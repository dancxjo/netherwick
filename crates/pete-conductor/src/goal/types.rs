#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GoalId(pub String);

impl GoalId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HomeostaticDrive {
    pub desired: f32,
    pub actual: f32,
    pub predicted: f32,
    pub error: f32,
    pub predicted_error: f32,
    pub satisfaction: f32,
    pub activation: f32,
}

impl HomeostaticDrive {
    fn update(&mut self, desired: f32, actual: f32, predicted: f32, dt_s: f32, impulse: f32) {
        self.desired = desired.clamp(0.0, 1.0);
        self.actual = actual.clamp(0.0, 1.0);
        self.predicted = predicted.clamp(0.0, 1.0);
        self.error = (self.desired - self.actual).max(0.0).clamp(0.0, 1.0);
        self.predicted_error = (self.desired - self.predicted).max(0.0).clamp(0.0, 1.0);
        self.satisfaction = (1.0 - self.error).clamp(0.0, 1.0);
        let target = (0.65 * self.error + 0.35 * self.predicted_error + impulse).clamp(0.0, 1.0);
        let tau_s = if target > self.activation { 0.5 } else { 5.0 };
        let alpha = if dt_s <= 0.0 {
            1.0
        } else {
            (1.0 - (-dt_s / tau_s).exp()).clamp(0.0, 1.0)
        };
        self.activation += (target - self.activation) * alpha;
        self.activation = self.activation.clamp(0.0, 1.0);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DriveSnapshot {
    pub schema_version: u32,
    pub t_ms: u64,
    pub energy: HomeostaticDrive,
    pub safety: HomeostaticDrive,
    pub curiosity: HomeostaticDrive,
    pub social: HomeostaticDrive,
    pub rest: HomeostaticDrive,
    pub certainty: HomeostaticDrive,
}

impl DriveSnapshot {
    pub fn legacy_sense(&self) -> DriveSense {
        DriveSense {
            battery_hunger: self.energy.activation,
            danger_avoidance: self.safety.activation,
            curiosity: self.curiosity.activation,
            social_interest: self.social.activation,
            fatigue: self.rest.activation,
            uncertainty_pressure: self.certainty.activation,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DriveDynamics {
    last_t_ms: Option<u64>,
    fatigue: f32,
    snapshot: DriveSnapshot,
    seeded: bool,
    pending_impulses: DriveSense,
}

impl DriveDynamics {
    pub fn seed_from(&mut self, drives: DriveSense) {
        if !self.seeded {
            self.add_impulses(drives);
            self.seeded = true;
        }
    }

    pub fn add_impulses(&mut self, impulses: DriveSense) {
        self.pending_impulses.battery_hunger =
            (self.pending_impulses.battery_hunger + impulses.battery_hunger).clamp(0.0, 1.0);
        self.pending_impulses.danger_avoidance =
            (self.pending_impulses.danger_avoidance + impulses.danger_avoidance).clamp(0.0, 1.0);
        self.pending_impulses.curiosity =
            (self.pending_impulses.curiosity + impulses.curiosity).clamp(0.0, 1.0);
        self.pending_impulses.social_interest =
            (self.pending_impulses.social_interest + impulses.social_interest).clamp(0.0, 1.0);
        self.pending_impulses.fatigue =
            (self.pending_impulses.fatigue + impulses.fatigue).clamp(0.0, 1.0);
        self.pending_impulses.uncertainty_pressure = (self.pending_impulses.uncertainty_pressure
            + impulses.uncertainty_pressure)
            .clamp(0.0, 1.0);
    }

    pub fn update(&mut self, world: &WorldModelSnapshot) -> DriveSnapshot {
        self.seeded = true;
        let impulses = std::mem::take(&mut self.pending_impulses);
        let dt_s = self
            .last_t_ms
            .map(|last| world.t_ms.saturating_sub(last) as f32 / 1_000.0)
            .unwrap_or(0.0)
            .clamp(0.0, 5.0);
        self.last_t_ms = Some(world.t_ms);

        let fatigue_delta = if world.self_model.charging {
            -0.01 * dt_s
        } else if world.self_model.moving {
            0.003 * dt_s
        } else {
            0.001 * dt_s
        };
        self.fatigue = (self.fatigue + fatigue_delta).clamp(0.0, 1.0);

        let predicted_energy = (world.self_model.battery_level
            + world
                .context
                .expected_battery_delta
                .as_ref()
                .map(|belief| belief.value)
                .unwrap_or(-0.01))
        .clamp(0.0, 1.0);
        self.snapshot.energy.update(
            0.80,
            if world.self_model.charging {
                1.0
            } else {
                world.self_model.battery_level
            },
            if world.self_model.charging {
                1.0
            } else {
                predicted_energy
            },
            dt_s,
            impulses.battery_hunger * 0.35,
        );

        let predicted_danger = world
            .hazards
            .predicted_risk
            .as_ref()
            .map(|belief| belief.value)
            .unwrap_or(0.0);
        let contact = if world.self_model.contact { 1.0 } else { 0.0 };
        let immediate_risk = world
            .hazards
            .immediate_risk
            .as_ref()
            .map(|belief| belief.value)
            .unwrap_or(0.0);
        let remembered_risk = world
            .hazards
            .remembered_risk
            .as_ref()
            .map(|belief| belief.value)
            .unwrap_or(0.0);
        let risk = predicted_danger
            .max(remembered_risk)
            .max(immediate_risk)
            .max(contact);
        self.snapshot.safety.update(
            0.95,
            1.0 - risk,
            1.0 - predicted_danger,
            dt_s,
            (contact * 0.4).max(impulses.danger_avoidance * 0.4),
        );

        let novelty = world
            .context
            .novelty
            .as_ref()
            .map(|belief| belief.value)
            .unwrap_or(0.0);
        let surprise = world
            .context
            .surprise
            .as_ref()
            .map(|belief| belief.value)
            .unwrap_or(0.0);
        let weighted_uncertainty = world.epistemic.weighted_uncertainty();
        let expected_information_gain = world
            .epistemic
            .affordances
            .iter()
            .filter(|affordance| affordance.available)
            .map(|affordance| affordance.expected_information_gain)
            .fold(0.0f32, f32::max);
        let information_satisfaction = (1.0 - weighted_uncertainty).clamp(0.0, 1.0);
        let predicted_information_satisfaction =
            (1.0 - (weighted_uncertainty - expected_information_gain).max(0.0)).clamp(0.0, 1.0);
        self.snapshot.curiosity.update(
            0.80,
            information_satisfaction,
            predicted_information_satisfaction,
            dt_s,
            (surprise * 0.20)
                .max(novelty * 0.10)
                .max(impulses.curiosity * 0.25),
        );

        let person_confidence = world
            .social
            .present_people()
            .map(|person| person.presence.confidence)
            .fold(0.0f32, f32::max);
        self.snapshot.social.update(
            0.50,
            person_confidence,
            person_confidence,
            dt_s,
            impulses.social_interest * 0.25,
        );
        self.snapshot.rest.update(
            0.80,
            1.0 - self.fatigue,
            1.0 - self.fatigue,
            dt_s,
            impulses.fatigue * 0.25,
        );
        let llm_certainty = world
            .context
            .llm_confidence
            .as_ref()
            .map(|belief| belief.value)
            .unwrap_or(1.0);
        let certainty = (1.0
            - world
                .context
                .prediction_uncertainty
                .as_ref()
                .map(|belief| belief.value)
                .unwrap_or(0.0))
        .min(llm_certainty)
        .clamp(0.0, 1.0);
        self.snapshot.certainty.update(
            0.85,
            certainty,
            certainty,
            dt_s,
            impulses.uncertainty_pressure * 0.25,
        );
        self.snapshot.schema_version = 1;
        self.snapshot.t_ms = world.t_ms;
        self.snapshot.clone()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Motivation {
    pub activation: f32,
    pub urgency: f32,
    pub satisfaction: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Affordance {
    pub behavior_id: String,
    pub available: bool,
    pub rejection_reason: Option<String>,
    pub confidence: f32,
    pub expected_reward: f32,
    pub expected_progress: f32,
    pub expected_risk: f32,
    pub expected_energy_cost: f32,
    pub expected_duration_ms: u64,
    #[serde(default)]
    pub expected_information_gain: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_uncertainty_after: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epistemic_question_id: Option<QuestionId>,
    pub target: Option<EntityId>,
    pub bearing_rad: Option<f32>,
    pub skill_request: Option<SkillRequest>,
    pub action: Option<ActionPrimitive>,
    pub provenance: Vec<EvidenceRef>,
    #[serde(default)]
    pub semantic_relation_ids: Vec<SemanticRelationId>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillId {
    #[default]
    StopAndStabilize,
    TurnTowardTarget,
    FollowBearing,
    ApproachTarget,
    BackAway,
    InspectTarget,
    WallFollow,
    AlignWithDock,
    SystematicSearch,
    HoldHeading,
    RetreatFromCliff,
    ReleasePersistentBumper,
    TurnBy,
    DriveDistance,
    Undock,
    SearchForDock,
    ReturnToDock,
    RuntimeLoaded,
}

fn skill_progress_metric(skill_id: SkillId) -> &'static str {
    match skill_id {
        SkillId::TurnTowardTarget
        | SkillId::FollowBearing
        | SkillId::HoldHeading
        | SkillId::TurnBy => "bearing_error",
        SkillId::ApproachTarget
        | SkillId::AlignWithDock
        | SkillId::SearchForDock
        | SkillId::ReturnToDock => "target_distance",
        SkillId::BackAway => "reverse_displacement",
        SkillId::RetreatFromCliff
        | SkillId::ReleasePersistentBumper
        | SkillId::DriveDistance
        | SkillId::Undock => "reverse_displacement",
        SkillId::InspectTarget => "uncertainty_reduction",
        SkillId::WallFollow => "path_progress",
        SkillId::SystematicSearch => "frontier_coverage",
        SkillId::StopAndStabilize => "motion_stability",
        SkillId::RuntimeLoaded => "goal_progress",
    }
}

fn default_progress_tolerance() -> f32 {
    0.1
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkillRequest {
    pub skill_id: SkillId,
    /// Fully qualified runtime implementation ID for `RuntimeLoaded`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implementation_id: Option<String>,
    pub goal_id: Option<GoalId>,
    pub behavior_id: Option<String>,
    pub target: Option<EntityId>,
    pub bearing_rad: Option<f32>,
    pub range_m: Option<f32>,
    pub stop_range_m: Option<f32>,
    pub maximum_duration_ms: u64,
    pub expected_progress: f32,
    #[serde(default)]
    pub progress_metric: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress_baseline: Option<f32>,
    #[serde(default = "default_progress_tolerance")]
    pub progress_tolerance: f32,
}

impl Default for SkillRequest {
    fn default() -> Self {
        Self {
            skill_id: SkillId::default(),
            implementation_id: None,
            goal_id: None,
            behavior_id: None,
            target: None,
            bearing_rad: None,
            range_m: None,
            stop_range_m: None,
            maximum_duration_ms: 0,
            expected_progress: 0.0,
            progress_metric: String::new(),
            progress_baseline: None,
            progress_tolerance: default_progress_tolerance(),
        }
    }
}

impl SkillRequest {
    pub fn runtime_loaded(implementation_id: impl Into<String>) -> Self {
        Self {
            skill_id: SkillId::RuntimeLoaded,
            implementation_id: Some(implementation_id.into()),
            progress_metric: "goal_progress".to_string(),
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillPhase {
    #[default]
    Requested,
    Running,
    Terminal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillOutcome {
    Completed,
    Failed,
    TimedOut,
    Cancelled,
    SafetyPreempted,
    AuthorityLost,
    TransportLost,
    CapabilityUnavailable,
    ResourcePreempted,
    PostconditionFailed,
    ScriptError,
    BudgetExceeded,
    /// Retained for decoding historical status; new skill runtimes use a
    /// broad typed outcome plus a structured `target_stale` detail.
    TargetStale,
    /// Retained for decoding historical status.
    Unavailable,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SkillScriptStatus {
    pub skill_id: String,
    pub source_hash: String,
    pub source_path: String,
    pub current_function: Option<String>,
    pub current_operation: Option<String>,
    #[serde(default)]
    pub held_resources: Vec<String>,
    #[serde(default)]
    pub waiting_resources: Vec<String>,
    #[serde(default)]
    pub active_children: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SkillStatus {
    pub request: SkillRequest,
    /// Stable identifier for one possessor execution. Retries receive a new id.
    #[serde(default)]
    pub execution_id: u64,
    pub phase: SkillPhase,
    pub outcome: Option<SkillOutcome>,
    pub progress: Option<f32>,
    /// Number of executions of this intention, not motor refreshes.
    #[serde(default)]
    pub attempts: u32,
    /// Motor/command refreshes sent during this execution.
    #[serde(default)]
    pub dispatch_count: u32,
    pub started_at_ms: Option<u64>,
    pub updated_at_ms: u64,
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<SkillScriptStatus>,
}

impl Affordance {
    fn utility(&self) -> f32 {
        0.20 * self.confidence
            + 0.20 * self.expected_reward
            + 0.30 * self.expected_progress
            + 0.30 * self.expected_information_gain
            - 0.25 * self.expected_risk
            - 0.10 * self.expected_energy_cost
    }

    fn with_bearing(mut self, bearing_rad: Option<f32>) -> Self {
        self.bearing_rad = bearing_rad;
        self
    }

    fn with_skill(mut self, skill_id: SkillId, stop_range_m: Option<f32>) -> Self {
        let progress_baseline = match skill_id {
            SkillId::TurnTowardTarget | SkillId::FollowBearing | SkillId::HoldHeading => {
                self.bearing_rad.map(f32::abs)
            }
            SkillId::BackAway
            | SkillId::SystematicSearch
            | SkillId::RetreatFromCliff
            | SkillId::ReleasePersistentBumper
            | SkillId::DriveDistance
            | SkillId::Undock
            | SkillId::SearchForDock
            | SkillId::ReturnToDock => Some(0.0),
            SkillId::TurnBy => self.bearing_rad.map(f32::abs),
            SkillId::RuntimeLoaded => Some(0.0),
            _ => None,
        };
        self.skill_request = Some(SkillRequest {
            skill_id,
            implementation_id: None,
            goal_id: None,
            behavior_id: None,
            target: self.target.clone(),
            bearing_rad: self.bearing_rad,
            range_m: None,
            stop_range_m,
            maximum_duration_ms: self.expected_duration_ms,
            expected_progress: self.expected_progress,
            progress_metric: skill_progress_metric(skill_id).to_string(),
            progress_baseline,
            progress_tolerance: 0.1,
        });
        self
    }

    fn with_skill_range(mut self, range_m: Option<f32>) -> Self {
        if let Some(request) = &mut self.skill_request {
            request.range_m = range_m;
            if matches!(
                request.skill_id,
                SkillId::ApproachTarget | SkillId::AlignWithDock
            ) {
                request.progress_baseline = range_m;
            }
        }
        self
    }

    fn with_runtime_skill(mut self, implementation_id: impl Into<String>) -> Self {
        self = self.with_skill(SkillId::RuntimeLoaded, None);
        if let Some(request) = &mut self.skill_request {
            request.implementation_id = Some(implementation_id.into());
        }
        self
    }

    fn with_epistemic(mut self, affordance: &EpistemicAffordance) -> Self {
        self.epistemic_question_id = Some(affordance.question_id.clone());
        self.expected_information_gain = affordance.expected_information_gain;
        self.expected_uncertainty_after = Some(affordance.expected_uncertainty_after);
        self
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Competence {
    pub confidence: f32,
    pub affordances: Vec<Affordance>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EvaluationContribution {
    pub source: String,
    pub value: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GoalEvaluation {
    pub goal_id: GoalId,
    pub t_ms: u64,
    pub world_revision: u64,
    pub disposition: GoalDisposition,
    pub motivation: Motivation,
    pub competence: Competence,
    pub contributions: Vec<EvaluationContribution>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_explanation: Option<SemanticExplanation>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GoalInterpretation {
    pub goal_id: GoalId,
    pub target: Option<EntityId>,
    pub target_confidence: f32,
    pub target_bearing_rad: Option<f32>,
    pub target_distance_m: Option<f32>,
    pub target_reachable: bool,
    pub danger: f32,
    pub novelty: f32,
    pub social_presence: f32,
    pub uncertainty: f32,
    pub stalled_goal_frustration: f32,
    pub epistemic_question_id: Option<QuestionId>,
    pub epistemic_question_family: Option<EpistemicQuestionFamily>,
    pub epistemic_importance: f32,
    pub expected_information_gain: f32,
    pub suggestions: Vec<ActionPrimitive>,
    pub provenance: Vec<EvidenceRef>,
}

pub type GoalInterpretationSnapshot = GoalInterpretation;
pub type GoalPerceptionSnapshot = GoalInterpretationSnapshot;
pub type GoalPerceptionContext<'a> = GoalInterpretationContext<'a>;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InterpreterState {
    pub last_world_revision: u64,
    pub updates: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EvaluatorState {
    pub evaluations: u64,
    pub last_activation: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecutorState {
    pub executions: u64,
    pub last_behavior_id: Option<String>,
    pub committed_turn_direction: Option<TurnDir>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GoalRuntimeState {
    pub active: bool,
    pub elapsed_time_ms: u64,
    #[serde(default)]
    pub attempts: u32,
    pub failed_attempts: u32,
    pub recent_progress: f32,
    #[serde(default)]
    pub progress_trend: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_progress_at_ms: Option<u64>,
    pub confidence_trend: f32,
    pub frustration: f32,
    pub last_confidence: Option<f32>,
    pub last_exit_reason: Option<GoalExitReason>,
    pub progress_expectation: Option<ProgressExpectation>,
    pub last_progress_observation: Option<ProgressObservation>,
    pub last_skill_outcome: Option<SkillOutcome>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProgressExpectation {
    pub behavior_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<f32>,
    pub expected_progress: f32,
    #[serde(default)]
    pub horizon_ms: u64,
    #[serde(default = "default_progress_tolerance")]
    pub tolerance: f32,
    pub deadline_ms: u64,
    pub metric: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProgressObservation {
    pub observed_at_ms: u64,
    pub progress: Option<f32>,
    pub source: String,
    pub outcome: Option<SkillOutcome>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategyProgressResponse {
    #[default]
    Inactive,
    Started,
    Retained,
    Changed,
    HelpRequested,
    Abandoned,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GoalProgressReport {
    pub goal_id: GoalId,
    pub selected_behavior: Option<String>,
    pub previous_behavior: Option<String>,
    pub expectation: Option<ProgressExpectation>,
    pub observation: Option<ProgressObservation>,
    #[serde(default)]
    pub attempts: u32,
    pub failed_attempts: u32,
    pub recent_progress: f32,
    #[serde(default)]
    pub progress_trend: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_progress_at_ms: Option<u64>,
    pub strategy_failure: f32,
    pub response: StrategyProgressResponse,
    pub reason: String,
}

impl GoalRuntimeState {
    fn snapshot(&self) -> GoalStatusBelief {
        GoalStatusBelief {
            meta: Default::default(),
            active: self.active,
            elapsed_time_ms: self.elapsed_time_ms,
            attempts: self.attempts,
            failed_attempts: self.failed_attempts,
            recent_progress: self.recent_progress,
            progress_trend: self.progress_trend,
            last_progress_at_ms: self.last_progress_at_ms,
            confidence_trend: self.confidence_trend,
            frustration: self.frustration,
            last_exit_reason: self
                .last_exit_reason
                .as_ref()
                .map(|reason| format!("{reason:?}").to_ascii_lowercase()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BehaviorDecision {
    pub goal_id: GoalId,
    pub behavior_id: String,
    pub action: ActionPrimitive,
    pub affordance: Affordance,
}

pub struct GoalInterpretationContext<'a> {
    pub world: &'a WorldModelSnapshot,
    pub drives: &'a DriveSnapshot,
    pub runtime: &'a GoalRuntimeState,
    pub suggestions: &'a [ActionPrimitive],
}

pub struct GoalEvaluationContext<'a> {
    pub world: &'a WorldModelSnapshot,
    pub drives: &'a DriveSnapshot,
    pub runtime: &'a GoalRuntimeState,
}

pub struct GoalExecutionContext<'a> {
    pub world: &'a WorldModelSnapshot,
    pub runtime: &'a GoalRuntimeState,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalExitReason {
    Superseded,
    Sleep,
    Satisfied,
    Completed,
    Failed,
    LostSafeAffordances,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalDisposition {
    #[default]
    Active,
    Satisfied,
    Completed,
    Failed,
}
