#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GoalArbiterConfig {
    pub minimum_dwell_ms: u64,
    pub persistence_bonus: f32,
    pub switching_cost: f32,
}

impl Default for GoalArbiterConfig {
    fn default() -> Self {
        Self {
            minimum_dwell_ms: 750,
            persistence_bonus: 0.10,
            switching_cost: 0.15,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GoalCommitment {
    pub goal_id: GoalId,
    pub entered_at_ms: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GoalSelection {
    pub selected_goal: Option<GoalId>,
    pub incumbent_goal: Option<GoalId>,
    pub challenger_goal: Option<GoalId>,
    pub switched: bool,
    pub retained_by_commitment: bool,
    pub reason: String,
    pub exit_reason: Option<GoalExitReason>,
    pub commitment_age_ms: u64,
    pub incumbent_activation: Option<f32>,
    pub challenger_activation: Option<f32>,
    pub required_activation: Option<f32>,
    pub effective_switching_cost: f32,
    pub effective_minimum_dwell_ms: u64,
}

#[derive(Clone, Debug, Default)]
pub struct GoalArbiter {
    pub config: GoalArbiterConfig,
    commitment: Option<GoalCommitment>,
}

impl GoalArbiter {
    pub fn current_goal(&self) -> Option<&GoalId> {
        self.commitment.as_ref().map(|value| &value.goal_id)
    }

    fn release(&mut self) -> Option<GoalId> {
        self.commitment.take().map(|commitment| commitment.goal_id)
    }

    pub fn select(&mut self, now_ms: u64, evaluations: &[GoalEvaluation]) -> GoalSelection {
        let eligible = evaluations
            .iter()
            .filter(|evaluation| {
                evaluation.disposition == GoalDisposition::Active
                    && evaluation
                        .competence
                        .affordances
                        .iter()
                        .any(|affordance| affordance.available && affordance.action.is_some())
            })
            .collect::<Vec<_>>();
        let challenger = eligible.iter().copied().max_by(|left, right| {
            left.motivation
                .activation
                .total_cmp(&right.motivation.activation)
                .then_with(|| right.goal_id.cmp(&left.goal_id))
        });
        let incumbent_id = self.current_goal().cloned();
        let incumbent_evaluation = incumbent_id.as_ref().and_then(|id| {
            evaluations
                .iter()
                .find(|evaluation| &evaluation.goal_id == id)
        });
        let incumbent = incumbent_id.as_ref().and_then(|id| {
            eligible
                .iter()
                .copied()
                .find(|evaluation| &evaluation.goal_id == id)
        });

        let Some(challenger) = challenger else {
            let released = incumbent_id.is_some();
            self.commitment = None;
            return GoalSelection {
                incumbent_goal: incumbent_id,
                switched: released,
                exit_reason: incumbent_evaluation.map(goal_exit_reason),
                reason: "no eligible goal evaluation".to_string(),
                ..GoalSelection::default()
            };
        };

        let Some(commitment) = self.commitment.as_ref() else {
            self.commitment = Some(GoalCommitment {
                goal_id: challenger.goal_id.clone(),
                entered_at_ms: now_ms,
            });
            return GoalSelection {
                selected_goal: Some(challenger.goal_id.clone()),
                challenger_goal: Some(challenger.goal_id.clone()),
                switched: true,
                reason: "selected initial goal".to_string(),
                challenger_activation: Some(challenger.motivation.activation),
                ..GoalSelection::default()
            };
        };

        let Some(incumbent) = incumbent else {
            let old = commitment.goal_id.clone();
            self.commitment = Some(GoalCommitment {
                goal_id: challenger.goal_id.clone(),
                entered_at_ms: now_ms,
            });
            return GoalSelection {
                selected_goal: Some(challenger.goal_id.clone()),
                incumbent_goal: Some(old),
                challenger_goal: Some(challenger.goal_id.clone()),
                switched: true,
                reason: "incumbent completed, failed, or lost all affordances".to_string(),
                exit_reason: incumbent_evaluation.map(goal_exit_reason),
                incumbent_activation: incumbent_evaluation
                    .map(|evaluation| evaluation.motivation.activation),
                challenger_activation: Some(challenger.motivation.activation),
                ..GoalSelection::default()
            };
        };

        if challenger.goal_id == incumbent.goal_id {
            return GoalSelection {
                selected_goal: Some(incumbent.goal_id.clone()),
                incumbent_goal: Some(incumbent.goal_id.clone()),
                challenger_goal: Some(challenger.goal_id.clone()),
                reason: "incumbent remains most active".to_string(),
                commitment_age_ms: now_ms.saturating_sub(commitment.entered_at_ms),
                incumbent_activation: Some(incumbent.motivation.activation),
                challenger_activation: Some(challenger.motivation.activation),
                ..GoalSelection::default()
            };
        }

        let urgency = challenger.motivation.urgency.clamp(0.0, 1.0);
        let effective_switching_cost = self.config.switching_cost * (1.0 - urgency);
        let effective_minimum_dwell_ms =
            (self.config.minimum_dwell_ms as f32 * (1.0 - urgency)).round() as u64;
        let dwell_ms = now_ms.saturating_sub(commitment.entered_at_ms);
        let required_activation = incumbent.motivation.activation
            + self.config.persistence_bonus
            + effective_switching_cost;
        if dwell_ms >= effective_minimum_dwell_ms
            && challenger.motivation.activation > required_activation
        {
            let old = commitment.goal_id.clone();
            self.commitment = Some(GoalCommitment {
                goal_id: challenger.goal_id.clone(),
                entered_at_ms: now_ms,
            });
            GoalSelection {
                selected_goal: Some(challenger.goal_id.clone()),
                incumbent_goal: Some(old),
                challenger_goal: Some(challenger.goal_id.clone()),
                switched: true,
                reason: "challenger overcame persistence and switching cost".to_string(),
                exit_reason: Some(GoalExitReason::Superseded),
                commitment_age_ms: dwell_ms,
                incumbent_activation: Some(incumbent.motivation.activation),
                challenger_activation: Some(challenger.motivation.activation),
                required_activation: Some(required_activation),
                effective_switching_cost,
                effective_minimum_dwell_ms,
                ..GoalSelection::default()
            }
        } else {
            GoalSelection {
                selected_goal: Some(incumbent.goal_id.clone()),
                incumbent_goal: Some(incumbent.goal_id.clone()),
                challenger_goal: Some(challenger.goal_id.clone()),
                retained_by_commitment: true,
                reason: if dwell_ms < effective_minimum_dwell_ms {
                    "incumbent retained during commitment dwell".to_string()
                } else {
                    "challenger did not overcome persistence and switching cost".to_string()
                },
                commitment_age_ms: dwell_ms,
                incumbent_activation: Some(incumbent.motivation.activation),
                challenger_activation: Some(challenger.motivation.activation),
                required_activation: Some(required_activation),
                effective_switching_cost,
                effective_minimum_dwell_ms,
                ..GoalSelection::default()
            }
        }
    }
}

fn goal_exit_reason(evaluation: &GoalEvaluation) -> GoalExitReason {
    match evaluation.disposition {
        GoalDisposition::Satisfied => GoalExitReason::Satisfied,
        GoalDisposition::Completed => GoalExitReason::Completed,
        GoalDisposition::Failed => GoalExitReason::Failed,
        GoalDisposition::Active => GoalExitReason::LostSafeAffordances,
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GoalCycle {
    pub schema_version: u32,
    pub world: WorldModelSnapshot,
    pub drives: DriveSnapshot,
    pub interpretations: Vec<GoalInterpretationSnapshot>,
    pub evaluations: Vec<GoalEvaluation>,
    pub selection: GoalSelection,
    pub behavior: Option<BehaviorDecision>,
    #[serde(default)]
    pub progress: Vec<GoalProgressReport>,
}

const GOAL_CYCLE_SCHEMA_VERSION: u32 = 2;
