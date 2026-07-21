#[derive(Clone, Debug)]
struct PendingOutcome {
    goal_id: GoalId,
    behavior_id: String,
    started_at_ms: u64,
    expected_progress: f32,
    expected_duration_ms: u64,
    metric: String,
    baseline: Option<f32>,
    tolerance: f32,
    start_pose: (f32, f32),
    start_target_distance_m: Option<f32>,
    target: Option<EntityId>,
    epistemic_question_id: Option<QuestionId>,
    start_uncertainty: Option<f32>,
}

pub struct GoalSystem {
    drives: DriveDynamics,
    goals: Vec<Box<dyn Goal>>,
    arbiter: GoalArbiter,
    pending: Option<PendingOutcome>,
    last_tick_ms: Option<u64>,
    processed_terminal_skill_executions: BTreeMap<GoalId, u64>,
}

impl Default for GoalSystem {
    fn default() -> Self {
        let goals = [
            "seek_charger",
            "escape_danger",
            "explore",
            "greet_person",
            "socialize",
            "rest",
            "investigate",
            "follow_task",
        ]
        .into_iter()
        .map(|id| Box::new(GoalModule::new(GoalId::new(id))) as Box<dyn Goal>)
        .collect();
        Self {
            drives: DriveDynamics::default(),
            goals,
            arbiter: GoalArbiter::default(),
            pending: None,
            last_tick_ms: None,
            processed_terminal_skill_executions: BTreeMap::new(),
        }
    }
}

impl GoalSystem {
    pub fn with_goals(goals: Vec<Box<dyn Goal>>) -> Self {
        Self {
            goals,
            drives: DriveDynamics::default(),
            arbiter: GoalArbiter::default(),
            pending: None,
            last_tick_ms: None,
            processed_terminal_skill_executions: BTreeMap::new(),
        }
    }

    pub fn register_goal(&mut self, goal: Box<dyn Goal>) -> Result<()> {
        if self
            .goals
            .iter()
            .any(|registered| registered.id() == goal.id())
        {
            return Err(anyhow!("goal {} is already registered", goal.id().as_str()));
        }
        self.goals.push(goal);
        Ok(())
    }

    pub fn add_drive_impulses(&mut self, impulses: DriveSense) {
        self.drives.add_impulses(impulses);
    }

    pub fn seed_drives(&mut self, drives: DriveSense) {
        self.drives.seed_from(drives);
    }

    pub fn observe_skill_status(&mut self, status: &SkillStatus) {
        let Some(goal_id) = status.request.goal_id.as_ref() else {
            return;
        };
        if status.phase == SkillPhase::Terminal
            && status.execution_id != 0
            && self
                .processed_terminal_skill_executions
                .get(goal_id)
                .is_some_and(|processed| *processed == status.execution_id)
        {
            return;
        }
        let Some(goal) = self.goals.iter_mut().find(|goal| goal.id() == goal_id) else {
            return;
        };
        let previous_progress = goal
            .runtime()
            .last_progress_observation
            .as_ref()
            .and_then(|observation| observation.progress);
        let measured_progress = if status.outcome == Some(SkillOutcome::SafetyPreempted) {
            None
        } else {
            status.progress
        };
        goal.runtime_mut().last_progress_observation = Some(ProgressObservation {
            observed_at_ms: status.updated_at_ms,
            progress: measured_progress,
            source: if status.outcome == Some(SkillOutcome::SafetyPreempted) {
                "autonomic_safety_preemption".to_string()
            } else {
                "possessor_skill".to_string()
            },
            outcome: status.outcome,
        });
        goal.runtime_mut().attempts = goal.runtime().attempts.max(status.attempts);
        if let Some(progress) = measured_progress {
            let delta = previous_progress.map_or(progress, |previous| progress - previous);
            goal.runtime_mut().progress_trend =
                (0.8 * goal.runtime().progress_trend + 0.2 * delta).clamp(-1.0, 1.0);
            goal.runtime_mut().recent_progress =
                (0.7 * goal.runtime().recent_progress + 0.3 * progress).clamp(0.0, 1.0);
            if delta > f32::EPSILON && progress >= status.request.progress_tolerance.max(0.01) {
                goal.runtime_mut().last_progress_at_ms = Some(status.updated_at_ms);
            }
        }
        if status.phase == SkillPhase::Terminal {
            if status.execution_id != 0 {
                self.processed_terminal_skill_executions
                    .insert(goal_id.clone(), status.execution_id);
            }
            goal.runtime_mut().last_skill_outcome = status.outcome;
            if matches!(
                status.outcome,
                Some(
                    SkillOutcome::Failed
                        | SkillOutcome::TimedOut
                        | SkillOutcome::Cancelled
                        | SkillOutcome::AuthorityLost
                        | SkillOutcome::TransportLost
                        | SkillOutcome::CapabilityUnavailable
                        | SkillOutcome::ResourcePreempted
                        | SkillOutcome::PostconditionFailed
                        | SkillOutcome::ScriptError
                        | SkillOutcome::BudgetExceeded
                        | SkillOutcome::TargetStale
                        | SkillOutcome::Unavailable
                )
            ) {
                goal.runtime_mut().failed_attempts =
                    goal.runtime().failed_attempts.saturating_add(1);
                goal.runtime_mut().frustration = (goal.runtime().frustration + 0.2).clamp(0.0, 1.0);
            }
        }
    }

    pub fn world_model_update_context(&self) -> WorldModelUpdateContext {
        let drive_summary = |drive: &HomeostaticDrive| DriveSelfSummary {
            desired: drive.desired,
            actual: drive.actual,
            predicted: drive.predicted,
            error: drive.error,
            satisfaction: drive.satisfaction,
            activation: drive.activation,
        };
        let active_goal = self
            .arbiter
            .current_goal()
            .map(|goal| goal.as_str().to_string());
        let active_status = active_goal
            .as_ref()
            .and_then(|active| self.goals.iter().find(|goal| goal.id().as_str() == active))
            .map(|goal| goal.runtime());
        let commitment_age_ms = self
            .arbiter
            .commitment
            .as_ref()
            .zip(self.last_tick_ms)
            .map(|(commitment, now_ms)| now_ms.saturating_sub(commitment.entered_at_ms))
            .unwrap_or(0);
        WorldModelUpdateContext {
            active_goal,
            goal_status: self
                .goals
                .iter()
                .map(|goal| (goal.id().as_str().to_string(), goal.runtime().snapshot()))
                .collect(),
            registered_goals: self
                .goals
                .iter()
                .map(|goal| goal.id().as_str().to_string())
                .collect(),
            registered_behaviors: REGISTERED_BEHAVIORS
                .iter()
                .map(|behavior| (*behavior).to_string())
                .collect(),
            drive_summaries: BTreeMap::from([
                (
                    "energy".to_string(),
                    drive_summary(&self.drives.snapshot.energy),
                ),
                (
                    "safety".to_string(),
                    drive_summary(&self.drives.snapshot.safety),
                ),
                (
                    "curiosity".to_string(),
                    drive_summary(&self.drives.snapshot.curiosity),
                ),
                (
                    "social".to_string(),
                    drive_summary(&self.drives.snapshot.social),
                ),
                (
                    "rest".to_string(),
                    drive_summary(&self.drives.snapshot.rest),
                ),
                (
                    "certainty".to_string(),
                    drive_summary(&self.drives.snapshot.certainty),
                ),
            ]),
            commitment_age_ms,
            active_behavior: self
                .pending
                .as_ref()
                .map(|pending| pending.behavior_id.clone()),
            expected_progress: self
                .pending
                .as_ref()
                .map(|pending| pending.expected_progress),
            recent_progress: active_status.map(|status| status.recent_progress),
            uncertainty: self.drives.snapshot.certainty.activation,
            strategy_failure_pressure: active_status
                .map(|status| status.frustration)
                .unwrap_or(0.0),
            temporal_expectations: self
                .pending
                .iter()
                .map(|pending| PendingTemporalExpectation {
                    subject: format!("behavior:{}", pending.behavior_id),
                    expected_interval: TimeInterval {
                        domain: ClockDomain::Predicted,
                        start_ms: pending.started_at_ms,
                        end_ms: Some(
                            pending
                                .started_at_ms
                                .saturating_add(pending.expected_duration_ms),
                        ),
                        uncertainty_ms: pending.expected_duration_ms / 4,
                    },
                    confidence: pending.expected_progress,
                    provenance: Vec::new(),
                })
                .collect(),
            epistemic_attempt: self.pending.as_ref().and_then(|pending| {
                pending
                    .epistemic_question_id
                    .as_ref()
                    .map(|question_id| EpistemicAttempt {
                        question_id: question_id.clone(),
                        behavior_id: pending.behavior_id.clone(),
                        started_at_ms: pending.started_at_ms,
                    })
            }),
            ..WorldModelUpdateContext::default()
        }
    }

    pub fn tick(
        &mut self,
        world: &WorldModelSnapshot,
        proposals: &[ActionPrimitive],
    ) -> Result<GoalCycle> {
        let world = world.clone();
        let previous_pending = self.pending.clone();
        self.observe_pending_outcome(&world);
        let drives = self.drives.update(&world);
        let mut interpretations = Vec::with_capacity(self.goals.len());
        let mut evaluations = Vec::with_capacity(self.goals.len());
        for goal in &mut self.goals {
            let runtime = goal.runtime().clone();
            let interpretation = goal.perceive(&GoalInterpretationContext {
                world: &world,
                drives: &drives,
                runtime: &runtime,
                suggestions: proposals,
            })?;
            let evaluation = goal.evaluate(
                &interpretation,
                &GoalEvaluationContext {
                    world: &world,
                    drives: &drives,
                    runtime: &runtime,
                },
            )?;
            interpretations.push(interpretation);
            evaluations.push(evaluation);
        }

        let previous_goal = self.arbiter.current_goal().cloned();
        let selection = self.arbiter.select(world.t_ms, &evaluations);
        if selection.switched {
            if let Some(previous) = previous_goal {
                if let Some(goal) = self.goals.iter_mut().find(|goal| goal.id() == &previous) {
                    goal.exit(
                        selection
                            .exit_reason
                            .clone()
                            .unwrap_or(GoalExitReason::Superseded),
                    );
                }
            }
            if let Some(selected) = selection.selected_goal.as_ref() {
                if let Some(goal) = self.goals.iter_mut().find(|goal| goal.id() == selected) {
                    let runtime = goal.runtime().clone();
                    goal.enter(&GoalExecutionContext {
                        world: &world,
                        runtime: &runtime,
                    });
                }
            }
        }
        let dt_ms = self
            .last_tick_ms
            .map(|last| world.t_ms.saturating_sub(last))
            .unwrap_or(0);
        self.last_tick_ms = Some(world.t_ms);
        let behavior = if let Some(goal_id) = selection.selected_goal.as_ref() {
            let index = self
                .goals
                .iter()
                .position(|goal| goal.id() == goal_id)
                .ok_or_else(|| anyhow!("selected goal is not registered"))?;
            let elapsed = self.goals[index]
                .runtime()
                .elapsed_time_ms
                .saturating_add(dt_ms);
            self.goals[index].runtime_mut().elapsed_time_ms = elapsed;
            let evaluation = evaluations
                .iter()
                .find(|evaluation| &evaluation.goal_id == goal_id)
                .ok_or_else(|| anyhow!("selected goal has no immutable evaluation"))?;
            let runtime = self.goals[index].runtime().clone();
            let decision = self.goals[index].execute(
                &GoalExecutionContext {
                    world: &world,
                    runtime: &runtime,
                },
                evaluation,
            )?;
            let begins_new_attempt = self
                .pending
                .as_ref()
                .map(|pending| {
                    pending.goal_id != decision.goal_id
                        || pending.behavior_id != decision.behavior_id
                })
                .unwrap_or(true);
            if begins_new_attempt {
                self.goals[index].runtime_mut().attempts =
                    self.goals[index].runtime().attempts.saturating_add(1);
                let start_uncertainty = decision
                    .affordance
                    .epistemic_question_id
                    .as_ref()
                    .and_then(|question_id| {
                        world
                            .epistemic
                            .active_questions
                            .iter()
                            .find(|question| &question.question_id == question_id)
                            .map(|question| question.current_uncertainty)
                    });
                let (progress_metric, progress_baseline, progress_tolerance) =
                    if decision.affordance.epistemic_question_id.is_some() {
                        ("uncertainty_reduction".to_string(), start_uncertainty, 0.1)
                    } else if let Some(request) = decision.affordance.skill_request.as_ref() {
                        let metric = if request.progress_metric.is_empty() {
                            skill_progress_metric(request.skill_id).to_string()
                        } else {
                            request.progress_metric.clone()
                        };
                        let baseline = request.progress_baseline.or_else(|| {
                            decision
                                .affordance
                                .target
                                .as_ref()
                                .and_then(|id| world.entities.get(id))
                                .and_then(|entity| entity.distance_m)
                        });
                        (metric, baseline, request.progress_tolerance)
                    } else {
                        ("world_displacement".to_string(), Some(0.0), 0.1)
                    };
                self.goals[index].runtime_mut().progress_expectation = Some(ProgressExpectation {
                    behavior_id: decision.behavior_id.clone(),
                    baseline: progress_baseline,
                    expected_progress: decision.affordance.expected_progress,
                    horizon_ms: decision.affordance.expected_duration_ms,
                    tolerance: progress_tolerance,
                    deadline_ms: world
                        .t_ms
                        .saturating_add(decision.affordance.expected_duration_ms),
                    metric: progress_metric.clone(),
                });
                self.pending = Some(PendingOutcome {
                    goal_id: decision.goal_id.clone(),
                    behavior_id: decision.behavior_id.clone(),
                    started_at_ms: world.t_ms,
                    expected_progress: decision.affordance.expected_progress,
                    expected_duration_ms: decision.affordance.expected_duration_ms,
                    metric: progress_metric,
                    baseline: progress_baseline,
                    tolerance: progress_tolerance,
                    start_pose: (world.self_model.pose.x_m, world.self_model.pose.y_m),
                    start_target_distance_m: decision
                        .affordance
                        .target
                        .as_ref()
                        .and_then(|id| world.entities.get(id))
                        .and_then(|entity| entity.distance_m),
                    target: decision.affordance.target.clone(),
                    epistemic_question_id: decision.affordance.epistemic_question_id.clone(),
                    start_uncertainty,
                });
            }
            Some(decision)
        } else {
            None
        };
        let progress = self.progress_reports(
            &evaluations,
            &selection,
            behavior.as_ref(),
            previous_pending.as_ref(),
            world.t_ms,
        );
        Ok(GoalCycle {
            schema_version: GOAL_CYCLE_SCHEMA_VERSION,
            world,
            drives,
            interpretations,
            evaluations,
            selection,
            behavior,
            progress,
        })
    }

    /// Quiesce deliberative control for sleep without pausing homeostatic dynamics.
    /// Any possessor-layer skill request becomes stale and must be rebuilt from a
    /// fresh world model after waking.
    pub fn suspend_for_sleep(&mut self, world: &WorldModelSnapshot) -> GoalCycle {
        if let Some(goal_id) = self.arbiter.release() {
            if let Some(goal) = self.goals.iter_mut().find(|goal| goal.id() == &goal_id) {
                goal.exit(GoalExitReason::Sleep);
            }
        }
        self.pending = None;
        self.last_tick_ms = Some(world.t_ms);
        GoalCycle {
            schema_version: GOAL_CYCLE_SCHEMA_VERSION,
            world: world.clone(),
            drives: self.drives.update(world),
            selection: GoalSelection {
                reason: "deliberative goals quiesced for sleep".to_string(),
                ..GoalSelection::default()
            },
            ..GoalCycle::default()
        }
    }

    fn observe_pending_outcome(&mut self, world: &WorldModelSnapshot) {
        let Some(pending) = self.pending.clone() else {
            return;
        };
        let elapsed = world.t_ms.saturating_sub(pending.started_at_ms);
        let dx = world.self_model.pose.x_m - pending.start_pose.0;
        let dy = world.self_model.pose.y_m - pending.start_pose.1;
        let movement_progress = ((dx * dx + dy * dy).sqrt() / 0.5).clamp(0.0, 1.0);
        let target_progress = pending
            .target
            .as_ref()
            .and_then(|target| world.entities.get(target))
            .filter(|entity| {
                entity.distance_meta.as_ref().is_some_and(|meta| {
                    !matches!(
                        meta.freshness,
                        Freshness::Stale | Freshness::Invalidated | Freshness::Missing
                    )
                })
            })
            .and_then(|entity| entity.distance_m)
            .zip(pending.start_target_distance_m)
            .map(|(current, start)| ((start - current) / start.max(0.1)).clamp(0.0, 1.0));
        let epistemic_progress = pending
            .epistemic_question_id
            .as_ref()
            .zip(pending.start_uncertainty)
            .map(|(question_id, start)| {
                let current = world
                    .epistemic
                    .active_questions
                    .iter()
                    .find(|question| &question.question_id == question_id)
                    .map(|question| question.current_uncertainty)
                    .or_else(|| {
                        world
                            .epistemic
                            .recent_outcomes
                            .iter()
                            .rev()
                            .find(|outcome| &outcome.question_id == question_id)
                            .map(|outcome| outcome.uncertainty_after)
                    })
                    .unwrap_or(0.0);
                ((start - current) / start.max(0.01)).clamp(0.0, 1.0)
            });
        let observed = if pending.behavior_id == "dock" && world.self_model.charging {
            Some(1.0)
        } else if pending.epistemic_question_id.is_some() {
            epistemic_progress
        } else if pending.target.is_some() {
            target_progress
        } else {
            Some(movement_progress)
        };
        let attempt_finished = elapsed >= pending.expected_duration_ms
            || (pending.behavior_id == "dock" && world.self_model.charging);
        if let Some(goal) = self
            .goals
            .iter_mut()
            .find(|goal| goal.id() == &pending.goal_id)
        {
            let previous_progress = goal
                .runtime()
                .last_progress_observation
                .as_ref()
                .and_then(|observation| observation.progress);
            goal.runtime_mut().last_progress_observation = Some(ProgressObservation {
                observed_at_ms: world.t_ms,
                progress: observed,
                source: if observed.is_some() {
                    "canonical_world_model".to_string()
                } else {
                    "canonical_world_model_unmeasurable".to_string()
                },
                outcome: None,
            });
            if let Some(confidence) = goal
                .last_evaluation()
                .map(|evaluation| evaluation.competence.confidence)
            {
                let trend = confidence - goal.runtime().last_confidence.unwrap_or(confidence);
                let confidence_trend =
                    (0.8 * goal.runtime().confidence_trend + 0.2 * trend).clamp(-1.0, 1.0);
                goal.runtime_mut().confidence_trend = confidence_trend;
                goal.runtime_mut().last_confidence = Some(confidence);
            }
            if let Some(observed) = observed {
                let delta = previous_progress.map_or(observed, |previous| observed - previous);
                goal.runtime_mut().progress_trend =
                    (0.8 * goal.runtime().progress_trend + 0.2 * delta).clamp(-1.0, 1.0);
                let recent_progress =
                    (0.7 * goal.runtime().recent_progress + 0.3 * observed).clamp(0.0, 1.0);
                goal.runtime_mut().recent_progress = recent_progress;
                if delta > f32::EPSILON && observed >= pending.tolerance.max(0.01) {
                    goal.runtime_mut().last_progress_at_ms = Some(world.t_ms);
                }
                if attempt_finished && observed + pending.tolerance < pending.expected_progress {
                    goal.runtime_mut().failed_attempts =
                        goal.runtime().failed_attempts.saturating_add(1);
                }
                let progress_deficit = (pending.expected_progress - observed).max(0.0);
                let failed = (goal.runtime().failed_attempts as f32 / 5.0).clamp(0.0, 1.0);
                let falling_confidence = (-goal.runtime().confidence_trend).max(0.0);
                let target_frustration =
                    (0.5 * progress_deficit + 0.3 * failed + 0.2 * falling_confidence)
                        .clamp(0.0, 1.0);
                let alpha = if target_frustration > goal.runtime().frustration {
                    0.20
                } else {
                    0.07
                };
                let frustration = goal.runtime().frustration
                    + (target_frustration - goal.runtime().frustration) * alpha;
                goal.runtime_mut().frustration = frustration;
            }
        }
        if attempt_finished {
            self.pending = None;
        }
    }

    fn progress_reports(
        &self,
        evaluations: &[GoalEvaluation],
        selection: &GoalSelection,
        behavior: Option<&BehaviorDecision>,
        previous_pending: Option<&PendingOutcome>,
        t_ms: u64,
    ) -> Vec<GoalProgressReport> {
        self.goals
            .iter()
            .map(|goal| {
                let runtime = goal.runtime();
                let evaluation = evaluations
                    .iter()
                    .find(|evaluation| evaluation.goal_id == *goal.id());
                let selected = selection.selected_goal.as_ref() == Some(goal.id());
                let selected_behavior = behavior
                    .filter(|decision| decision.goal_id == *goal.id())
                    .map(|decision| decision.behavior_id.clone());
                let previous_behavior = previous_pending
                    .filter(|pending| pending.goal_id == *goal.id())
                    .map(|pending| pending.behavior_id.clone());
                let observation = runtime.last_progress_observation.clone();
                let observed_this_tick = observation
                    .as_ref()
                    .is_some_and(|observation| observation.observed_at_ms == t_ms);
                let expectation = if observed_this_tick {
                    previous_pending
                        .filter(|pending| pending.goal_id == *goal.id())
                        .map(|pending| ProgressExpectation {
                            behavior_id: pending.behavior_id.clone(),
                            baseline: pending.baseline,
                            expected_progress: pending.expected_progress,
                            horizon_ms: pending.expected_duration_ms,
                            tolerance: pending.tolerance,
                            deadline_ms: pending
                                .started_at_ms
                                .saturating_add(pending.expected_duration_ms),
                            metric: pending.metric.clone(),
                        })
                } else {
                    runtime.progress_expectation.clone()
                };
                let (response, reason) = if evaluation.is_some_and(|evaluation| {
                    evaluation.disposition == GoalDisposition::Failed
                }) && previous_behavior.is_some()
                {
                    (
                        StrategyProgressResponse::Abandoned,
                        format!(
                            "goal abandoned after {} failed attempts with strategy failure {:.2}",
                            runtime.failed_attempts, runtime.frustration
                        ),
                    )
                } else if !selected {
                    (
                        StrategyProgressResponse::Inactive,
                        "goal was not selected this tick".to_string(),
                    )
                } else if selected_behavior.as_deref() == Some("request_charge_help")
                    && previous_behavior.as_deref() != Some("request_charge_help")
                {
                    (
                        StrategyProgressResponse::HelpRequested,
                        format!(
                            "bounded escalation requested help after {} failed attempts with strategy failure {:.2}",
                            runtime.failed_attempts, runtime.frustration
                        ),
                    )
                } else if previous_behavior.is_none() {
                    (
                        StrategyProgressResponse::Started,
                        "started the highest-utility available strategy".to_string(),
                    )
                } else if previous_behavior != selected_behavior {
                    (
                        StrategyProgressResponse::Changed,
                        format!(
                            "changed strategy after observed progress {} against expected {} with strategy failure {:.2}",
                            format_progress(observation.as_ref().and_then(|value| value.progress)),
                            format_progress(expectation.as_ref().map(|value| value.expected_progress)),
                            runtime.frustration
                        ),
                    )
                } else {
                    (
                        StrategyProgressResponse::Retained,
                        format!(
                            "retained strategy with observed progress {} against expected {} and strategy failure {:.2}",
                            format_progress(observation.as_ref().and_then(|value| value.progress)),
                            format_progress(expectation.as_ref().map(|value| value.expected_progress)),
                            runtime.frustration
                        ),
                    )
                };
                GoalProgressReport {
                    goal_id: goal.id().clone(),
                    selected_behavior,
                    previous_behavior,
                    expectation,
                    observation,
                    attempts: runtime.attempts,
                    failed_attempts: runtime.failed_attempts,
                    recent_progress: runtime.recent_progress,
                    progress_trend: runtime.progress_trend,
                    last_progress_at_ms: runtime.last_progress_at_ms,
                    strategy_failure: runtime.frustration,
                    response,
                    reason,
                }
            })
            .collect()
    }
}

fn format_progress(progress: Option<f32>) -> String {
    progress
        .map(|progress| format!("{progress:.2}"))
        .unwrap_or_else(|| "unknown".to_string())
}
