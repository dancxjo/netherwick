pub trait Goal: Send {
    fn id(&self) -> &GoalId;

    fn perceive(
        &mut self,
        context: &GoalInterpretationContext<'_>,
    ) -> Result<GoalInterpretationSnapshot>;

    fn evaluate(
        &mut self,
        perception: &GoalInterpretationSnapshot,
        context: &GoalEvaluationContext<'_>,
    ) -> Result<GoalEvaluation>;

    fn execute(
        &mut self,
        context: &GoalExecutionContext<'_>,
        evaluation: &GoalEvaluation,
    ) -> Result<BehaviorDecision>;

    fn enter(&mut self, context: &GoalExecutionContext<'_>);
    fn exit(&mut self, reason: GoalExitReason);
    fn runtime(&self) -> &GoalRuntimeState;
    fn runtime_mut(&mut self) -> &mut GoalRuntimeState;
    fn last_evaluation(&self) -> Option<&GoalEvaluation>;
}

pub trait GoalInterpreter: Send {
    fn interpret(
        &self,
        state: &InterpreterState,
        context: &GoalInterpretationContext<'_>,
    ) -> Result<(GoalInterpretation, InterpreterState)>;
}

pub trait GoalEvaluator: Send {
    fn evaluate(
        &self,
        state: &EvaluatorState,
        interpretation: &GoalInterpretation,
        context: &GoalEvaluationContext<'_>,
    ) -> Result<(GoalEvaluation, EvaluatorState)>;
}

pub trait GoalExecutor: Send {
    fn execute(
        &self,
        state: &ExecutorState,
        evaluation: &GoalEvaluation,
        context: &GoalExecutionContext<'_>,
    ) -> Result<(BehaviorDecision, ExecutorState)>;
}

pub struct GoalModule {
    pub id: GoalId,
    interpreter: Box<dyn GoalInterpreter>,
    evaluator: Box<dyn GoalEvaluator>,
    executor: Box<dyn GoalExecutor>,
    interpreter_state: InterpreterState,
    evaluator_state: EvaluatorState,
    executor_state: ExecutorState,
    pub runtime: GoalRuntimeState,
    last_interpretation: Option<GoalInterpretation>,
    last_evaluation: Option<GoalEvaluation>,
}

impl GoalModule {
    fn new(id: GoalId) -> Self {
        Self::from_components(
            id.clone(),
            Box::new(RuleGoalInterpreter { id: id.clone() }),
            Box::new(RuleGoalEvaluator { id: id.clone() }),
            Box::new(UtilityGoalExecutor { id }),
        )
    }

    pub fn from_components(
        id: GoalId,
        interpreter: Box<dyn GoalInterpreter>,
        evaluator: Box<dyn GoalEvaluator>,
        executor: Box<dyn GoalExecutor>,
    ) -> Self {
        Self {
            interpreter,
            evaluator,
            executor,
            id,
            interpreter_state: InterpreterState::default(),
            evaluator_state: EvaluatorState::default(),
            executor_state: ExecutorState::default(),
            runtime: GoalRuntimeState::default(),
            last_interpretation: None,
            last_evaluation: None,
        }
    }

    pub fn replace_interpreter(&mut self, interpreter: Box<dyn GoalInterpreter>) {
        self.interpreter = interpreter;
        self.interpreter_state = InterpreterState::default();
    }

    pub fn replace_evaluator(&mut self, evaluator: Box<dyn GoalEvaluator>) {
        self.evaluator = evaluator;
        self.evaluator_state = EvaluatorState::default();
    }

    pub fn replace_executor(&mut self, executor: Box<dyn GoalExecutor>) {
        self.executor = executor;
        self.executor_state = ExecutorState::default();
    }

    fn interpret(&mut self, context: &GoalInterpretationContext<'_>) -> Result<GoalInterpretation> {
        let (interpretation, next) = self
            .interpreter
            .interpret(&self.interpreter_state, context)?;
        self.interpreter_state = next;
        self.last_interpretation = Some(interpretation.clone());
        Ok(interpretation)
    }

    fn evaluate(
        &mut self,
        interpretation: &GoalInterpretation,
        context: &GoalEvaluationContext<'_>,
    ) -> Result<GoalEvaluation> {
        let (evaluation, next) =
            self.evaluator
                .evaluate(&self.evaluator_state, interpretation, context)?;
        self.evaluator_state = next;
        self.last_evaluation = Some(evaluation.clone());
        Ok(evaluation)
    }

    fn execute(
        &mut self,
        evaluation: &GoalEvaluation,
        context: &GoalExecutionContext<'_>,
    ) -> Result<BehaviorDecision> {
        let (decision, next) = self
            .executor
            .execute(&self.executor_state, evaluation, context)?;
        self.executor_state = next;
        Ok(decision)
    }
}

impl Goal for GoalModule {
    fn id(&self) -> &GoalId {
        &self.id
    }

    fn perceive(
        &mut self,
        context: &GoalInterpretationContext<'_>,
    ) -> Result<GoalInterpretationSnapshot> {
        self.interpret(context)
    }

    fn evaluate(
        &mut self,
        perception: &GoalInterpretationSnapshot,
        context: &GoalEvaluationContext<'_>,
    ) -> Result<GoalEvaluation> {
        GoalModule::evaluate(self, perception, context)
    }

    fn execute(
        &mut self,
        context: &GoalExecutionContext<'_>,
        evaluation: &GoalEvaluation,
    ) -> Result<BehaviorDecision> {
        GoalModule::execute(self, evaluation, context)
    }

    fn enter(&mut self, _context: &GoalExecutionContext<'_>) {
        self.runtime.active = true;
        self.runtime.elapsed_time_ms = 0;
        self.runtime.recent_progress = 0.0;
        self.runtime.last_exit_reason = None;
        self.executor_state.last_behavior_id = None;
        self.executor_state.executions = 0;
        self.executor_state.committed_turn_direction = None;
    }

    fn exit(&mut self, reason: GoalExitReason) {
        self.runtime.active = false;
        self.runtime.elapsed_time_ms = 0;
        if matches!(
            reason,
            GoalExitReason::Satisfied | GoalExitReason::Completed
        ) {
            self.runtime.attempts = 0;
            self.runtime.failed_attempts = 0;
            self.runtime.recent_progress = 0.0;
            self.runtime.progress_trend = 0.0;
            self.runtime.last_progress_at_ms = None;
            self.runtime.frustration = 0.0;
        }
        self.runtime.last_exit_reason = Some(reason);
    }

    fn runtime(&self) -> &GoalRuntimeState {
        &self.runtime
    }

    fn runtime_mut(&mut self) -> &mut GoalRuntimeState {
        &mut self.runtime
    }

    fn last_evaluation(&self) -> Option<&GoalEvaluation> {
        self.last_evaluation.as_ref()
    }
}

struct RuleGoalInterpreter {
    id: GoalId,
}

impl GoalInterpreter for RuleGoalInterpreter {
    fn interpret(
        &self,
        state: &InterpreterState,
        context: &GoalInterpretationContext<'_>,
    ) -> Result<(GoalInterpretation, InterpreterState)> {
        let target_kind = match self.id.as_str() {
            "seek_charger" => Some(WorldEntityKind::Charger),
            "socialize" | "greet_person" => Some(WorldEntityKind::Person),
            "investigate" => Some(WorldEntityKind::SoundSource),
            _ => None,
        };
        let target = target_kind.and_then(|kind| {
            context
                .world
                .entities
                .values()
                .filter(|entity| entity.kind == kind)
                .max_by(|left, right| {
                    goal_entity_score(left, context.world)
                        .total_cmp(&goal_entity_score(right, context.world))
                })
        });
        let target_relative = target.and_then(|entity| {
            entity.pose.map(|pose| {
                let self_pose = context.world.self_model.pose;
                let dx = pose.x_m - self_pose.x_m;
                let dy = pose.y_m - self_pose.y_m;
                let distance = (dx * dx + dy * dy).sqrt();
                let bearing = normalize_angle(dy.atan2(dx) - self_pose.heading_rad);
                (bearing, distance)
            })
        });
        let danger = context.drives.safety.activation;
        let stalled_goal_frustration = context
            .world
            .self_model
            .goal_status
            .values()
            .map(|status| status.frustration)
            .fold(0.0f32, f32::max);
        let interpretation = GoalInterpretation {
            goal_id: self.id.clone(),
            target: target.map(|entity| entity.id.clone()),
            target_confidence: target.map(|entity| entity.confidence).unwrap_or(0.0),
            target_bearing_rad: target_relative
                .map(|(bearing, _)| bearing)
                .or_else(|| target.and_then(|entity| entity.bearing_rad)),
            target_distance_m: target_relative
                .map(|(_, distance)| distance)
                .or_else(|| target.and_then(|entity| entity.distance_m)),
            target_reachable: target.is_some_and(|entity| entity.reachability.reachable),
            danger,
            novelty: context
                .world
                .context
                .novelty
                .as_ref()
                .map(|belief| belief.value)
                .unwrap_or(0.0),
            social_presence: context.drives.social.actual,
            uncertainty: context.drives.certainty.activation,
            stalled_goal_frustration,
            epistemic_question_id: context
                .world
                .epistemic
                .most_important_question()
                .map(|question| question.question_id.clone()),
            epistemic_question_family: context
                .world
                .epistemic
                .most_important_question()
                .map(|question| question.family),
            epistemic_importance: context
                .world
                .epistemic
                .most_important_question()
                .map(|question| question.importance)
                .unwrap_or(0.0),
            expected_information_gain: context
                .world
                .epistemic
                .affordances
                .iter()
                .filter(|affordance| affordance.available)
                .map(|affordance| affordance.expected_information_gain)
                .fold(0.0f32, f32::max),
            suggestions: context.suggestions.to_vec(),
            provenance: target
                .map(|entity| entity.provenance.clone())
                .unwrap_or_default(),
        };
        Ok((
            interpretation,
            InterpreterState {
                last_world_revision: context.world.revision,
                updates: state.updates.saturating_add(1),
            },
        ))
    }
}

fn goal_entity_score(entity: &WorldEntity, world: &WorldModelSnapshot) -> f32 {
    let distance = entity
        .pose
        .map(|pose| {
            let dx = pose.x_m - world.self_model.pose.x_m;
            let dy = pose.y_m - world.self_model.pose.y_m;
            (dx * dx + dy * dy).sqrt()
        })
        .or(entity.distance_m)
        .unwrap_or(10.0);
    let reachability = if entity.reachability.reachable {
        1.0
    } else {
        0.10
    };
    reachability * entity.confidence / (1.0 + distance.max(0.0))
}

fn normalize_angle(mut angle: f32) -> f32 {
    while angle > std::f32::consts::PI {
        angle -= std::f32::consts::TAU;
    }
    while angle < -std::f32::consts::PI {
        angle += std::f32::consts::TAU;
    }
    angle
}

struct RuleGoalEvaluator {
    id: GoalId,
}

impl GoalEvaluator for RuleGoalEvaluator {
    fn evaluate(
        &self,
        state: &EvaluatorState,
        interpretation: &GoalInterpretation,
        context: &GoalEvaluationContext<'_>,
    ) -> Result<(GoalEvaluation, EvaluatorState)> {
        let (activation, urgency, satisfaction, mut affordances, mut contributions) =
            match self.id.as_str() {
                "seek_charger" => evaluate_seek_charger(interpretation, context),
                "escape_danger" => evaluate_escape(interpretation, context),
                "explore" => evaluate_explore(interpretation, context),
                "greet_person" => evaluate_greet_person(interpretation, context),
                "socialize" => evaluate_socialize(interpretation, context),
                "rest" => evaluate_rest(interpretation, context),
                "investigate" => evaluate_investigate(interpretation, context),
                "follow_task" => evaluate_follow_task(interpretation, context),
                unknown => return Err(anyhow!("unknown goal {unknown}")),
            };

        if let Some(reign) = context.world.authority.as_ref().map(|belief| &belief.input) {
            if matches!(reign.mode, ReignMode::Assist | ReignMode::Suggest) {
                if let Some(action) = reign.command.to_action() {
                    if let Some(matching_confidence) = affordances
                        .iter()
                        .filter(|affordance| {
                            affordance.available && affordance.action.as_ref() == Some(&action)
                        })
                        .map(|affordance| affordance.confidence)
                        .max_by(f32::total_cmp)
                    {
                        let scale = if reign.mode == ReignMode::Assist {
                            0.60
                        } else {
                            0.25 * matching_confidence.clamp(0.0, 1.0)
                        };
                        contributions.push(EvaluationContribution {
                            source: format!("reign.{:?}", reign.mode).to_lowercase(),
                            value: scale * reign.priority.clamp(0.0, 1.0),
                        });
                    }
                }
            }
        }
        let bias = contributions
            .iter()
            .filter(|contribution| contribution.source.starts_with("reign."))
            .map(|contribution| contribution.value)
            .sum::<f32>();
        let activation = (activation + bias).clamp(0.0, 1.0);
        for affordance in &mut affordances {
            affordance.available &= affordance.action.is_some();
            if let Some(capability) = affordance.action.as_ref().and_then(required_capability) {
                if !context
                    .world
                    .self_model
                    .capabilities
                    .is_available(capability)
                    || !context
                        .world
                        .self_model
                        .capabilities
                        .is_authorized(capability)
                {
                    affordance.available = false;
                    affordance.rejection_reason = Some(
                        context
                            .world
                            .self_model
                            .capabilities
                            .execution_block_reason(capability)
                            .unwrap_or("required capability is unavailable")
                            .to_string(),
                    );
                }
            }
        }
        let confidence = affordances
            .iter()
            .filter(|affordance| affordance.available)
            .map(|affordance| affordance.confidence)
            .fold(0.0f32, f32::max)
            .clamp(0.0, 1.0);
        let escaped_with_clearance = self.id.as_str() == "escape_danger"
            && context.runtime.recent_progress >= 0.01
            && !context.world.self_model.stuck
            && !context.world.self_model.contact
            && context
                .world
                .self_model
                .range_nearest_m
                .unwrap_or(f32::INFINITY)
                >= 0.25;
        let disposition = if self.id.as_str() == "seek_charger" && context.world.self_model.charging
        {
            GoalDisposition::Completed
        } else if escaped_with_clearance {
            GoalDisposition::Completed
        } else if self.id.as_str() == "follow_task" && affordances.is_empty() {
            GoalDisposition::Completed
        } else if context.runtime.failed_attempts >= 8 && context.runtime.frustration > 0.9 {
            GoalDisposition::Failed
        } else if satisfaction >= 0.999 && activation <= 0.05 {
            GoalDisposition::Satisfied
        } else {
            GoalDisposition::Active
        };
        let evaluation = GoalEvaluation {
            goal_id: self.id.clone(),
            t_ms: context.world.t_ms,
            world_revision: context.world.revision,
            disposition,
            motivation: Motivation {
                activation,
                urgency: urgency.clamp(0.0, 1.0),
                satisfaction: satisfaction.clamp(0.0, 1.0),
            },
            competence: Competence {
                confidence,
                affordances,
            },
            contributions,
            semantic_explanation: (self.id.as_str() == "seek_charger")
                .then(|| interpretation.target.as_ref())
                .flatten()
                .map(|target| context.world.semantic.charger_explanation(target)),
        };
        Ok((
            evaluation,
            EvaluatorState {
                evaluations: state.evaluations.saturating_add(1),
                last_activation: activation,
            },
        ))
    }
}

struct UtilityGoalExecutor {
    id: GoalId,
}

impl GoalExecutor for UtilityGoalExecutor {
    fn execute(
        &self,
        state: &ExecutorState,
        evaluation: &GoalEvaluation,
        context: &GoalExecutionContext<'_>,
    ) -> Result<(BehaviorDecision, ExecutorState)> {
        let mut candidates = evaluation
            .competence
            .affordances
            .iter()
            .filter(|affordance| affordance.available && affordance.action.is_some())
            .collect::<Vec<_>>();
        if self.id.as_str() == "escape_danger" {
            let phase = state.executions % 13;
            let next = if context.world.self_model.contact && phase <= 1 {
                "reverse_from_danger"
            } else {
                match phase {
                    0..=8 => "turn_toward_clearance",
                    9..=11 => "probe_clearance",
                    _ => "inspect_clearance",
                }
            };
            if let Some(index) = candidates
                .iter()
                .position(|affordance| affordance.behavior_id == next)
            {
                candidates.swap(0, index);
            }
        } else if self.id.as_str() == "seek_charger" {
            let request_help =
                evaluation.motivation.urgency > 0.8 && context.runtime.frustration > 0.6;
            candidates.sort_by(|left, right| {
                charger_behavior_rank(&left.behavior_id, request_help)
                    .cmp(&charger_behavior_rank(&right.behavior_id, request_help))
                    .then_with(|| right.utility().total_cmp(&left.utility()))
            });
        } else if context.runtime.frustration > 0.6 {
            candidates.sort_by(|left, right| {
                let left_repeat =
                    state.last_behavior_id.as_deref() == Some(left.behavior_id.as_str());
                let right_repeat =
                    state.last_behavior_id.as_deref() == Some(right.behavior_id.as_str());
                let left_utility = left.utility() - if left_repeat { 0.35 } else { 0.0 };
                let right_utility = right.utility() - if right_repeat { 0.35 } else { 0.0 };
                right_utility.total_cmp(&left_utility)
            });
        } else {
            candidates.sort_by(|left, right| right.utility().total_cmp(&left.utility()));
        }
        let mut affordance = candidates
            .first()
            .copied()
            .ok_or_else(|| {
                anyhow!(
                    "goal {} has no executable affordance",
                    evaluation.goal_id.as_str()
                )
            })?
            .clone();
        let mut action = affordance
            .action
            .clone()
            .ok_or_else(|| anyhow!("selected affordance has no action"))?;
        if let Some(request) = &mut affordance.skill_request {
            request.goal_id = Some(evaluation.goal_id.clone());
            request.behavior_id = Some(affordance.behavior_id.clone());
        }
        let mut committed_turn_direction = state.committed_turn_direction.clone();
        if self.id.as_str() == "escape_danger" {
            if let ActionPrimitive::Turn {
                direction,
                intensity,
                duration_ms,
            } = &action
            {
                let direction = committed_turn_direction
                    .clone()
                    .unwrap_or_else(|| direction.clone());
                committed_turn_direction = Some(direction.clone());
                action = ActionPrimitive::Turn {
                    direction,
                    intensity: *intensity,
                    duration_ms: *duration_ms,
                };
                affordance.action = Some(action.clone());
            }
            if (state.executions + 1) % 13 == 0 {
                committed_turn_direction = None;
            }
        }
        let decision = BehaviorDecision {
            goal_id: evaluation.goal_id.clone(),
            behavior_id: affordance.behavior_id.clone(),
            action,
            affordance,
        };
        Ok((
            decision.clone(),
            ExecutorState {
                executions: state.executions.saturating_add(1),
                last_behavior_id: Some(decision.behavior_id),
                committed_turn_direction,
            },
        ))
    }
}

fn charger_behavior_rank(behavior_id: &str, request_help: bool) -> u8 {
    match behavior_id {
        "dock" => 0,
        "approach_charger" => 1,
        "turn_toward_charger" => 2,
        "request_charge_help" if request_help => 3,
        "systematic_charger_search" if request_help => 4,
        "inspect_for_charger" if request_help => 5,
        "systematic_charger_search" => 3,
        "inspect_for_charger" => 4,
        "request_charge_help" => 5,
        _ => 6,
    }
}

fn contribution(source: &str, value: f32) -> EvaluationContribution {
    EvaluationContribution {
        source: source.to_string(),
        value,
    }
}

fn affordance(
    behavior_id: &str,
    action: ActionPrimitive,
    confidence: f32,
    reward: f32,
    progress: f32,
    risk: f32,
    energy: f32,
    duration_ms: u64,
    target: Option<EntityId>,
    provenance: &[EvidenceRef],
) -> Affordance {
    Affordance {
        behavior_id: behavior_id.to_string(),
        available: true,
        rejection_reason: None,
        confidence: confidence.clamp(0.0, 1.0),
        expected_reward: reward.clamp(-1.0, 1.0),
        expected_progress: progress.clamp(0.0, 1.0),
        expected_risk: risk.clamp(0.0, 1.0),
        expected_energy_cost: energy.clamp(0.0, 1.0),
        expected_duration_ms: duration_ms,
        expected_information_gain: 0.0,
        expected_uncertainty_after: None,
        epistemic_question_id: None,
        target,
        bearing_rad: None,
        skill_request: None,
        action: Some(action),
        provenance: provenance.to_vec(),
        semantic_relation_ids: Vec::new(),
    }
}

fn rejected_affordance(
    behavior_id: &str,
    reason: impl Into<String>,
    target: Option<EntityId>,
    bearing_rad: Option<f32>,
    provenance: &[EvidenceRef],
) -> Affordance {
    Affordance {
        behavior_id: behavior_id.to_string(),
        available: false,
        rejection_reason: Some(reason.into()),
        target,
        bearing_rad,
        provenance: provenance.to_vec(),
        ..Affordance::default()
    }
}

const REGISTERED_BEHAVIORS: &[&str] = &[
    "dock",
    "turn_toward_charger",
    "approach_charger",
    "inspect_for_charger",
    "systematic_charger_search",
    "request_charge_help",
    "reverse_from_danger",
    "turn_toward_clearance",
    "probe_clearance",
    "inspect_clearance",
    "wander",
    "frontier_follow",
    "inspect_novelty",
    "orient_to_person",
    "approach_person",
    "greet_person",
    "speak",
    "rest",
    "investigate_sound",
    "orient_for_charger_evidence",
    "inspect_charger_hypothesis",
    "search_for_charger_evidence",
    "scan_clearance",
    "inspect_path",
    "stop_and_observe_path",
    "inspect_person_identity",
    "listen_for_identity",
    "ask_identity_clarification",
    "listen_for_direction",
    "orient_for_sound_parallax",
    "inspect_failure_context",
    "compare_failure_prediction",
    "follow_task",
];

fn required_capability(action: &ActionPrimitive) -> Option<&'static str> {
    match action {
        ActionPrimitive::Go { .. }
        | ActionPrimitive::Drive { .. }
        | ActionPrimitive::Turn { .. }
        | ActionPrimitive::Approach { .. }
        | ActionPrimitive::Dock
        | ActionPrimitive::Explore { .. } => Some("actuator:drive"),
        ActionPrimitive::Speak { .. } | ActionPrimitive::Chirp { .. } => Some("actuator:speaker"),
        ActionPrimitive::Inspect {
            target: InspectTarget::Charger | InspectTarget::Person,
        } => Some("sensor:vision"),
        ActionPrimitive::Inspect { .. } => None,
        ActionPrimitive::Stop => None,
    }
}
