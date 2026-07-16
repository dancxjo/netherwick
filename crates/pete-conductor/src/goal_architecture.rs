use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use pete_actions::{
    ActionPrimitive, ApproachTarget, ExploreStyle, InspectTarget, ReignMode, TurnDir,
};
use pete_now::{
    ClockDomain, DriveSelfSummary, DriveSense, EntityId, EpistemicActionKind, EpistemicAffordance,
    EpistemicAttempt, EpistemicQuestionFamily, EvidenceRef, GoalStatusBelief,
    PendingTemporalExpectation, QuestionId, SemanticBehaviorId, SemanticConceptId,
    SemanticExplanation, SemanticNodeRef, SemanticPredicate, SemanticRelationId, TimeInterval,
    WorldEntity, WorldEntityKind, WorldModelSnapshot, WorldModelUpdateContext,
};
use serde::{Deserialize, Serialize};

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
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SkillRequest {
    pub skill_id: SkillId,
    pub goal_id: Option<GoalId>,
    pub behavior_id: Option<String>,
    pub target: Option<EntityId>,
    pub bearing_rad: Option<f32>,
    pub range_m: Option<f32>,
    pub stop_range_m: Option<f32>,
    pub maximum_duration_ms: u64,
    pub expected_progress: f32,
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
    SafetyPreempted,
    AuthorityLost,
    TargetStale,
    Unavailable,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SkillStatus {
    pub request: SkillRequest,
    pub phase: SkillPhase,
    pub outcome: Option<SkillOutcome>,
    pub progress: Option<f32>,
    pub attempts: u32,
    pub started_at_ms: Option<u64>,
    pub updated_at_ms: u64,
    pub reason: Option<String>,
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
        self.skill_request = Some(SkillRequest {
            skill_id,
            goal_id: None,
            behavior_id: None,
            target: self.target.clone(),
            bearing_rad: self.bearing_rad,
            range_m: None,
            stop_range_m,
            maximum_duration_ms: self.expected_duration_ms,
            expected_progress: self.expected_progress,
        });
        self
    }

    fn with_skill_range(mut self, range_m: Option<f32>) -> Self {
        if let Some(request) = &mut self.skill_request {
            request.range_m = range_m;
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
    pub failed_attempts: u32,
    pub recent_progress: f32,
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
    pub expected_progress: f32,
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

impl GoalRuntimeState {
    fn snapshot(&self) -> GoalStatusBelief {
        GoalStatusBelief {
            meta: Default::default(),
            active: self.active,
            elapsed_time_ms: self.elapsed_time_ms,
            failed_attempts: self.failed_attempts,
            recent_progress: self.recent_progress,
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
            self.runtime.failed_attempts = 0;
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
            "socialize" => Some(WorldEntityKind::Person),
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

type EvaluationParts = (f32, f32, f32, Vec<Affordance>, Vec<EvaluationContribution>);

fn evaluate_seek_charger(
    interpretation: &GoalInterpretation,
    context: &GoalEvaluationContext<'_>,
) -> EvaluationParts {
    let energy = context.drives.energy.activation;
    let urgency = ((0.25 - context.world.self_model.battery_level) / 0.20).clamp(0.0, 1.0);
    let confidence = interpretation.target_confidence;
    let mut affordances = Vec::new();
    match interpretation.target_distance_m {
        Some(distance) if distance <= 0.35 && confidence >= 0.65 => {
            affordances.push(
                affordance(
                    "dock",
                    ActionPrimitive::Dock,
                    confidence,
                    1.0,
                    1.0,
                    0.05,
                    0.02,
                    2_000,
                    interpretation.target.clone(),
                    &interpretation.provenance,
                )
                .with_bearing(interpretation.target_bearing_rad)
                .with_skill(SkillId::AlignWithDock, Some(0.20))
                .with_skill_range(interpretation.target_distance_m),
            );
        }
        Some(distance) if distance > 0.35 => affordances.push(rejected_affordance(
            "dock",
            "charger is outside docking range",
            interpretation.target.clone(),
            interpretation.target_bearing_rad,
            &interpretation.provenance,
        )),
        Some(_) => affordances.push(rejected_affordance(
            "dock",
            "charger confidence is too low for docking",
            interpretation.target.clone(),
            interpretation.target_bearing_rad,
            &interpretation.provenance,
        )),
        None => affordances.push(rejected_affordance(
            "dock",
            "no localized charger target",
            None,
            None,
            &interpretation.provenance,
        )),
    }
    if let Some(bearing) = interpretation.target_bearing_rad {
        if confidence < 0.35 {
            affordances.push(rejected_affordance(
                "approach_charger",
                "charger confidence is too low for locomotion",
                interpretation.target.clone(),
                Some(bearing),
                &interpretation.provenance,
            ));
        } else if !interpretation.target_reachable {
            affordances.push(rejected_affordance(
                "approach_charger",
                "the charger target is not currently reachable",
                interpretation.target.clone(),
                Some(bearing),
                &interpretation.provenance,
            ));
        } else if bearing.abs() > 0.20 {
            affordances.push(
                affordance(
                    "turn_toward_charger",
                    ActionPrimitive::Turn {
                        direction: if bearing >= 0.0 {
                            TurnDir::Left
                        } else {
                            TurnDir::Right
                        },
                        intensity: 0.4,
                        duration_ms: 700,
                    },
                    confidence,
                    0.65,
                    0.75,
                    interpretation.danger * 0.25,
                    0.05,
                    700,
                    interpretation.target.clone(),
                    &interpretation.provenance,
                )
                .with_bearing(Some(bearing))
                .with_skill(SkillId::TurnTowardTarget, None)
                .with_skill_range(interpretation.target_distance_m),
            );
        } else {
            affordances.push(
                affordance(
                    "approach_charger",
                    ActionPrimitive::Approach {
                        target: ApproachTarget::Charger,
                    },
                    confidence,
                    0.8,
                    0.9,
                    interpretation.danger,
                    0.15,
                    1_000,
                    interpretation.target.clone(),
                    &interpretation.provenance,
                )
                .with_bearing(Some(bearing))
                .with_skill(SkillId::ApproachTarget, Some(0.30))
                .with_skill_range(interpretation.target_distance_m),
            );
        }
    } else {
        affordances.push(rejected_affordance(
            "approach_charger",
            "charger bearing is unknown",
            interpretation.target.clone(),
            None,
            &interpretation.provenance,
        ));
    }
    affordances.push(affordance(
        "inspect_for_charger",
        ActionPrimitive::Inspect {
            target: InspectTarget::Charger,
        },
        (1.0 - confidence).max(0.35),
        0.35,
        0.35,
        interpretation.danger * 0.25,
        0.03,
        750,
        interpretation.target.clone(),
        &interpretation.provenance,
    ));
    affordances.push(
        affordance(
            "systematic_charger_search",
            ActionPrimitive::Explore {
                style: ExploreStyle::WallFollow,
                duration_ms: 1_000,
            },
            (1.0 - confidence).max(0.25),
            0.8,
            0.20,
            interpretation.danger,
            0.2,
            1_000,
            None,
            &interpretation.provenance,
        )
        .with_skill(SkillId::SystematicSearch, None),
    );
    if urgency > 0.8 && confidence < 0.2 && context.runtime.frustration > 0.6 {
        affordances.push(affordance(
            "request_charge_help",
            ActionPrimitive::Speak {
                // Solresol: "Help! I'm hungry!" (dosido = help; dsod = hungry).
                text: "Dosido! Dore dsod!".to_string(),
            },
            0.9,
            0.55,
            0.5,
            0.0,
            0.0,
            2_000,
            None,
            &[],
        ));
    }
    if let Some(question) = context
        .world
        .epistemic
        .active_questions
        .iter()
        .find(|question| question.family == EpistemicQuestionFamily::ChargerIdentityOrBearing)
    {
        for goal_affordance in &mut affordances {
            let epistemic_behavior = match goal_affordance.behavior_id.as_str() {
                "turn_toward_charger" => Some("orient_for_charger_evidence"),
                "inspect_for_charger" => Some("inspect_charger_hypothesis"),
                "systematic_charger_search" => Some("search_for_charger_evidence"),
                _ => None,
            };
            let Some(epistemic_behavior) = epistemic_behavior else {
                continue;
            };
            if let Some(epistemic) = context
                .world
                .epistemic
                .affordances
                .iter()
                .find(|candidate| {
                    candidate.question_id == question.question_id
                        && candidate.behavior_id == epistemic_behavior
                })
            {
                goal_affordance.epistemic_question_id = Some(question.question_id.clone());
                goal_affordance.expected_information_gain = epistemic.expected_information_gain;
                goal_affordance.expected_uncertainty_after =
                    Some(epistemic.expected_uncertainty_after);
            }
        }
    }
    let dock_available = affordances
        .iter()
        .any(|affordance| affordance.behavior_id == "dock" && affordance.available);
    if context.world.self_model.contact && !dock_available {
        for affordance in &mut affordances {
            affordance.available = false;
            affordance.rejection_reason = Some(
                "immediate contact must be cleared before charger seeking resumes".to_string(),
            );
        }
    }
    for goal_affordance in &mut affordances {
        goal_affordance.semantic_relation_ids = charger_affordance_semantics(
            context.world,
            interpretation.target.as_ref(),
            &goal_affordance.behavior_id,
        );
    }
    let semantic_confidence = context
        .world
        .semantic
        .relations
        .values()
        .filter(|relation| {
            relation.subject == SemanticNodeRef::Concept(SemanticConceptId("charger".to_string()))
                && matches!(
                    relation.predicate,
                    SemanticPredicate::Restores | SemanticPredicate::SatisfiesDrive
                )
        })
        .map(|relation| relation.confidence)
        .fold(0.0f32, f32::max);
    (
        (0.85 * energy + 0.15 * confidence).clamp(0.0, 1.0),
        urgency,
        context.drives.energy.satisfaction,
        affordances,
        vec![
            contribution("drive.energy", energy),
            contribution("world.charger_confidence", confidence),
            contribution("semantic.charger_energy_meaning", semantic_confidence),
        ],
    )
}

fn charger_affordance_semantics(
    world: &WorldModelSnapshot,
    target: Option<&EntityId>,
    behavior_id: &str,
) -> Vec<SemanticRelationId> {
    let charger = SemanticNodeRef::Concept(SemanticConceptId("charger".to_string()));
    let semantic_behavior = match behavior_id {
        "dock" => Some("dock"),
        "approach_charger" | "turn_toward_charger" => Some("approach_charger"),
        _ => None,
    };
    world
        .semantic
        .relations
        .values()
        .filter(|relation| {
            (relation.subject == charger
                && matches!(
                    relation.predicate,
                    SemanticPredicate::Restores
                        | SemanticPredicate::SatisfiesDrive
                        | SemanticPredicate::HelpsGoal
                ))
                || semantic_behavior.is_some_and(|behavior| {
                    relation.subject == charger
                        && relation.predicate == SemanticPredicate::Affords
                        && relation.object
                            == SemanticNodeRef::Behavior(SemanticBehaviorId(behavior.to_string()))
                })
                || target.is_some_and(|target| {
                    (relation.subject == SemanticNodeRef::Entity(target.clone())
                        && relation.predicate == SemanticPredicate::IsA
                        && relation.object == charger)
                        || (relation.predicate == SemanticPredicate::Blocks
                            && relation.object == SemanticNodeRef::Entity(target.clone()))
                })
        })
        .map(|relation| relation.id.clone())
        .collect()
}

fn evaluate_escape(
    interpretation: &GoalInterpretation,
    context: &GoalEvaluationContext<'_>,
) -> EvaluationParts {
    let danger = interpretation.danger;
    let contact = context.world.self_model.contact;
    let stuck = context.world.self_model.stuck;
    let corner_trap = context
        .world
        .self_model
        .stuck_trap_kind
        .as_ref()
        .is_some_and(|belief| belief.value == pete_now::StuckTrapKind::Corner);
    let confidence = context
        .world
        .context
        .map_confidence
        .as_ref()
        .map(|belief| belief.value)
        .unwrap_or(0.0)
        .max(0.5);
    let direction = if context.world.self_model.bump_left {
        TurnDir::Right
    } else if context.world.self_model.contact {
        TurnDir::Left
    } else if context
        .world
        .local_geometry
        .right_clearance_m
        .as_ref()
        .map(|belief| belief.value)
        .unwrap_or(0.0)
        > context
            .world
            .local_geometry
            .left_clearance_m
            .as_ref()
            .map(|belief| belief.value)
            .unwrap_or(0.0)
    {
        TurnDir::Right
    } else if let Some(bearing) = context
        .world
        .context
        .safe_bearing_rad
        .as_ref()
        .map(|belief| belief.value)
    {
        if bearing >= 0.0 {
            TurnDir::Left
        } else {
            TurnDir::Right
        }
    } else {
        TurnDir::Left
    };
    let mut affordances = Vec::new();
    if contact || (stuck && !corner_trap) {
        affordances.push(
            affordance(
                "reverse_from_danger",
                ActionPrimitive::Go {
                    intensity: -0.18,
                    duration_ms: 300,
                },
                0.95,
                0.7,
                0.8,
                0.1,
                0.08,
                300,
                None,
                &[],
            )
            .with_skill(SkillId::BackAway, None),
        );
    }
    let clearance_bearing = Some(match &direction {
        TurnDir::Left => 0.75,
        TurnDir::Right => -0.75,
    });
    affordances.push(
        affordance(
            "turn_toward_clearance",
            ActionPrimitive::Turn {
                direction: direction.clone(),
                intensity: 0.75,
                duration_ms: 500,
            },
            confidence,
            0.65,
            0.7,
            0.15,
            0.08,
            500,
            None,
            &[],
        )
        .with_bearing(clearance_bearing)
        .with_skill(SkillId::TurnTowardTarget, None),
    );
    let center_clearance = context
        .world
        .local_geometry
        .center_clearance_m
        .as_ref()
        .map(|belief| belief.value);
    if center_clearance.is_some_and(|clearance| clearance >= 0.30) || (corner_trap && !contact) {
        affordances.push(affordance(
            "probe_clearance",
            ActionPrimitive::Go {
                intensity: 0.14,
                duration_ms: 300,
            },
            confidence,
            0.55,
            0.65,
            0.15,
            0.05,
            300,
            None,
            &[],
        ));
    } else {
        affordances.push(rejected_affordance(
            "probe_clearance",
            "center clearance is below 0.30 m or unknown",
            None,
            None,
            &[],
        ));
    }
    affordances.push(affordance(
        "inspect_clearance",
        ActionPrimitive::Inspect {
            target: InspectTarget::Novelty,
        },
        confidence * (1.0 - interpretation.danger * 0.5),
        0.5,
        0.35,
        0.0,
        0.01,
        500,
        None,
        &[],
    ));
    (
        danger.max(if contact { 1.0 } else { 0.0 }),
        danger.max(if contact { 1.0 } else { 0.0 }),
        context.drives.safety.satisfaction,
        affordances,
        vec![contribution("drive.safety", danger)],
    )
}

fn evaluate_explore(
    interpretation: &GoalInterpretation,
    context: &GoalEvaluationContext<'_>,
) -> EvaluationParts {
    let drives = context.drives;
    let activation = (0.15 + 0.65 * drives.curiosity.activation
        - 0.55 * drives.energy.activation
        - 0.65 * drives.safety.activation
        - 0.50 * drives.rest.activation
        - 0.25 * drives.certainty.activation)
        .clamp(0.0, 1.0);
    let action = if let Some(bearing) = context
        .world
        .context
        .frontier_bearing_rad
        .as_ref()
        .map(|belief| belief.value)
    {
        ActionPrimitive::Turn {
            direction: if bearing >= 0.0 {
                TurnDir::Left
            } else {
                TurnDir::Right
            },
            intensity: 0.35,
            duration_ms: 500,
        }
    } else if interpretation.novelty > 0.55 {
        ActionPrimitive::Inspect {
            target: InspectTarget::Novelty,
        }
    } else {
        ActionPrimitive::Explore {
            style: ExploreStyle::RandomWalk,
            duration_ms: 1_000,
        }
    };
    (
        activation,
        0.1,
        drives.curiosity.satisfaction,
        vec![affordance(
            "explore_frontier",
            action,
            (1.0 - interpretation.danger).clamp(0.0, 1.0),
            0.45,
            0.6,
            interpretation.danger,
            0.2,
            1_000,
            None,
            &[],
        )],
        vec![contribution("drive.curiosity", drives.curiosity.activation)],
    )
}

fn evaluate_socialize(
    interpretation: &GoalInterpretation,
    context: &GoalEvaluationContext<'_>,
) -> EvaluationParts {
    let social = context.drives.social.activation;
    let person = context.world.social.most_relevant_person();
    let identity = person.and_then(|person| person.best_identity());
    let identity_confidence = identity.map(|identity| identity.confidence).unwrap_or(0.0);
    let confidence = person
        .map(|person| person.presence.confidence)
        .unwrap_or(interpretation.target_confidence)
        .max(identity_confidence * 0.8);
    let identity_uncertain = person.is_some_and(|person| person.identity_is_uncertain());
    let person_target = person.map(|person| EntityId(person.person_id.0.clone()));
    let person_distance = person
        .and_then(|person| person.location.as_ref())
        .and_then(|location| location.distance_m)
        .or(interpretation.target_distance_m);
    let person_bearing = person
        .and_then(|person| person.location.as_ref())
        .and_then(|location| location.bearing_rad)
        .or(interpretation.target_bearing_rad);
    let action = match person_distance {
        Some(distance) if distance <= 0.8 && identity_uncertain => ActionPrimitive::Speak {
            text: "Hello. What should I call you?".to_string(),
        },
        Some(distance) if distance <= 0.8 => ActionPrimitive::Speak {
            text: person
                .and_then(|person| person.preferred_name.as_ref())
                .map(|name| format!("Hello, {}.", name.value))
                .unwrap_or_else(|| "Hello.".to_string()),
        },
        Some(_) => ActionPrimitive::Approach {
            target: ApproachTarget::Person,
        },
        None => ActionPrimitive::Inspect {
            target: InspectTarget::Person,
        },
    };
    let mut engagement = affordance(
        if identity_uncertain {
            "clarify_person_identity"
        } else {
            "social_engagement"
        },
        action.clone(),
        confidence.max(0.25),
        0.55,
        0.55,
        interpretation.danger,
        0.1,
        1_000,
        person_target.or_else(|| interpretation.target.clone()),
        person
            .map(|person| person.meta.provenance.as_slice())
            .unwrap_or(&interpretation.provenance),
    )
    .with_bearing(person_bearing);
    if matches!(action, ActionPrimitive::Approach { .. }) {
        engagement = engagement
            .with_skill(SkillId::ApproachTarget, Some(0.75))
            .with_skill_range(person_distance);
    }
    if identity_uncertain {
        if let Some(epistemic) = context
            .world
            .epistemic
            .affordances
            .iter()
            .filter(|affordance| affordance.action_kind == EpistemicActionKind::AskPerson)
            .max_by(|left, right| {
                left.epistemic_utility()
                    .total_cmp(&right.epistemic_utility())
            })
        {
            engagement = engagement.with_epistemic(epistemic);
        }
    }
    let pending_request = context
        .world
        .social
        .active_interaction
        .as_ref()
        .is_some_and(|interaction| !interaction.unresolved_requests.is_empty());
    (
        (0.70 * social + 0.30 * confidence + if pending_request { 0.20 } else { 0.0 }
            - 0.60 * interpretation.danger
            - 0.40 * context.drives.rest.activation)
            .clamp(0.0, 1.0),
        0.2,
        context.drives.social.satisfaction,
        vec![engagement],
        vec![
            contribution("drive.social", social),
            contribution("world.social.person_confidence", confidence),
            contribution("world.social.pending_request", pending_request as u8 as f32),
        ],
    )
}

fn evaluate_rest(
    _interpretation: &GoalInterpretation,
    context: &GoalEvaluationContext<'_>,
) -> EvaluationParts {
    let rest = context.drives.rest.activation;
    (
        rest,
        if context.world.self_model.charging {
            0.8
        } else {
            rest * 0.5
        },
        context.drives.rest.satisfaction,
        vec![affordance(
            "remain_stationary",
            ActionPrimitive::Stop,
            1.0,
            0.35,
            0.5,
            0.0,
            0.0,
            1_000,
            None,
            &[],
        )],
        vec![contribution("drive.rest", rest)],
    )
}

fn evaluate_investigate(
    interpretation: &GoalInterpretation,
    context: &GoalEvaluationContext<'_>,
) -> EvaluationParts {
    let uncertainty = context.drives.certainty.activation;
    let frustration = interpretation.stalled_goal_frustration;
    let question = context.world.epistemic.most_important_question();
    let epistemic_pressure = question
        .map(|question| question.importance * question.current_uncertainty)
        .unwrap_or(0.0);
    let mut affordances = question
        .map(|question| {
            context
                .world
                .epistemic
                .affordances_for(&question.question_id)
                .filter(|affordance| affordance.available)
                .map(conductor_epistemic_affordance)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if affordances.is_empty() {
        affordances.push(affordance(
            "gather_evidence",
            if interpretation.target.is_some() {
                ActionPrimitive::Inspect {
                    target: InspectTarget::Sound,
                }
            } else {
                ActionPrimitive::Inspect {
                    target: InspectTarget::Novelty,
                }
            },
            (1.0 - uncertainty).max(0.3),
            0.45,
            0.6,
            interpretation.danger * 0.25,
            0.05,
            750,
            interpretation.target.clone(),
            &interpretation.provenance,
        ));
    }
    (
        (0.50 * uncertainty
            + 0.55 * epistemic_pressure
            + 0.25
                * context
                    .world
                    .context
                    .surprise
                    .as_ref()
                    .map(|belief| belief.value)
                    .unwrap_or(0.0)
            + 0.35 * frustration
            - 0.50 * interpretation.danger)
            .clamp(0.0, 1.0),
        (0.25 + frustration * 0.5).clamp(0.0, 1.0),
        context.drives.certainty.satisfaction,
        affordances,
        vec![
            contribution("drive.certainty", uncertainty),
            contribution("world.epistemic.question_pressure", epistemic_pressure),
            contribution("self.stalled_goal", frustration),
        ],
    )
}

fn conductor_epistemic_affordance(source: &EpistemicAffordance) -> Affordance {
    let inspect_target = if source.affected_belief.0.contains("charger") {
        InspectTarget::Charger
    } else if source.affected_belief.0.contains("person") {
        InspectTarget::Person
    } else if source.affected_belief.0.contains("sound") {
        InspectTarget::Sound
    } else {
        InspectTarget::Novelty
    };
    let (action, skill) = match source.action_kind {
        EpistemicActionKind::OrientToBearing if source.bearing_rad.is_some() => (
            ActionPrimitive::Turn {
                direction: if source.bearing_rad.unwrap_or_default() >= 0.0 {
                    TurnDir::Left
                } else {
                    TurnDir::Right
                },
                intensity: 0.3,
                duration_ms: source.duration_ms,
            },
            Some(SkillId::TurnTowardTarget),
        ),
        EpistemicActionKind::SystematicSearch => (
            ActionPrimitive::Explore {
                style: ExploreStyle::WallFollow,
                duration_ms: source.duration_ms,
            },
            Some(SkillId::SystematicSearch),
        ),
        EpistemicActionKind::ScanClearance => (
            ActionPrimitive::Inspect {
                target: InspectTarget::Novelty,
            },
            Some(SkillId::InspectTarget),
        ),
        EpistemicActionKind::Listen => (
            ActionPrimitive::Inspect {
                target: InspectTarget::Sound,
            },
            Some(SkillId::InspectTarget),
        ),
        EpistemicActionKind::AskPerson => (
            ActionPrimitive::Speak {
                text: "Hello. What should I call you?".to_string(),
            },
            None,
        ),
        EpistemicActionKind::StopAndObserve | EpistemicActionKind::ComparePrediction => {
            (ActionPrimitive::Stop, Some(SkillId::StopAndStabilize))
        }
        EpistemicActionKind::InspectTarget
        | EpistemicActionKind::OrientToBearing
        | EpistemicActionKind::Unknown => (
            ActionPrimitive::Inspect {
                target: inspect_target,
            },
            Some(SkillId::InspectTarget),
        ),
    };
    let mut result = affordance(
        &source.behavior_id,
        action,
        source.confidence,
        source.expected_information_gain,
        source.expected_information_gain,
        source.risk,
        source.energy_cost,
        source.duration_ms,
        source.target.clone(),
        &[],
    )
    .with_bearing(source.bearing_rad)
    .with_epistemic(source);
    if let Some(skill) = skill {
        result = result.with_skill(skill, None);
    }
    result
}

fn evaluate_follow_task(
    interpretation: &GoalInterpretation,
    context: &GoalEvaluationContext<'_>,
) -> EvaluationParts {
    let affordances = interpretation
        .suggestions
        .iter()
        .cloned()
        .enumerate()
        .map(|(index, action)| {
            let mut task_affordance = affordance(
                &format!("task_proposal_{index}"),
                action.clone(),
                context
                    .world
                    .context
                    .llm_confidence
                    .as_ref()
                    .map(|belief| belief.value)
                    .unwrap_or(0.5)
                    .max(0.5),
                0.5,
                0.5,
                interpretation.danger,
                0.1,
                1_000,
                None,
                &[],
            );
            if matches!(action, ActionPrimitive::Go { intensity, .. } if intensity < 0.0) {
                task_affordance = task_affordance.with_skill(SkillId::BackAway, None);
            }
            task_affordance
        })
        .collect::<Vec<_>>();
    let activation = if affordances.is_empty() { 0.0 } else { 0.45 };
    (
        activation,
        0.3,
        if affordances.is_empty() { 1.0 } else { 0.0 },
        affordances,
        vec![contribution(
            "proposal.count",
            interpretation.suggestions.len() as f32,
        )],
    )
}

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
}

#[derive(Clone, Debug)]
struct PendingOutcome {
    goal_id: GoalId,
    behavior_id: String,
    started_at_ms: u64,
    expected_progress: f32,
    expected_duration_ms: u64,
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
}

impl Default for GoalSystem {
    fn default() -> Self {
        let goals = [
            "seek_charger",
            "escape_danger",
            "explore",
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
        let Some(goal) = self.goals.iter_mut().find(|goal| goal.id() == goal_id) else {
            return;
        };
        goal.runtime_mut().last_progress_observation = Some(ProgressObservation {
            observed_at_ms: status.updated_at_ms,
            progress: status.progress,
            source: "possessor_skill".to_string(),
            outcome: status.outcome,
        });
        if let Some(progress) = status.progress {
            goal.runtime_mut().recent_progress =
                (0.7 * goal.runtime().recent_progress + 0.3 * progress).clamp(0.0, 1.0);
        }
        if status.phase == SkillPhase::Terminal {
            goal.runtime_mut().last_skill_outcome = status.outcome;
            goal.runtime_mut().progress_expectation = None;
            if matches!(
                status.outcome,
                Some(
                    SkillOutcome::Failed
                        | SkillOutcome::TimedOut
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
                let progress_metric = if decision.affordance.epistemic_question_id.is_some() {
                    "uncertainty_reduction"
                } else {
                    match decision
                        .affordance
                        .skill_request
                        .as_ref()
                        .map(|request| request.skill_id)
                    {
                        Some(
                            SkillId::TurnTowardTarget
                            | SkillId::FollowBearing
                            | SkillId::HoldHeading,
                        ) => "bearing_error",
                        Some(SkillId::ApproachTarget | SkillId::AlignWithDock) => "target_distance",
                        Some(SkillId::BackAway) => "reverse_displacement",
                        Some(_) => "skill_specific",
                        None => "world_displacement",
                    }
                };
                self.goals[index].runtime_mut().progress_expectation = Some(ProgressExpectation {
                    behavior_id: decision.behavior_id.clone(),
                    expected_progress: decision.affordance.expected_progress,
                    deadline_ms: world
                        .t_ms
                        .saturating_add(decision.affordance.expected_duration_ms),
                    metric: progress_metric.to_string(),
                });
                self.pending = Some(PendingOutcome {
                    goal_id: decision.goal_id.clone(),
                    behavior_id: decision.behavior_id.clone(),
                    started_at_ms: world.t_ms,
                    expected_progress: decision.affordance.expected_progress,
                    expected_duration_ms: decision.affordance.expected_duration_ms,
                    start_pose: (world.self_model.pose.x_m, world.self_model.pose.y_m),
                    start_target_distance_m: decision
                        .affordance
                        .target
                        .as_ref()
                        .and_then(|id| world.entities.get(id))
                        .and_then(|entity| entity.distance_m),
                    target: decision.affordance.target.clone(),
                    epistemic_question_id: decision.affordance.epistemic_question_id.clone(),
                    start_uncertainty: decision.affordance.epistemic_question_id.as_ref().and_then(
                        |question_id| {
                            world
                                .epistemic
                                .active_questions
                                .iter()
                                .find(|question| &question.question_id == question_id)
                                .map(|question| question.current_uncertainty)
                        },
                    ),
                });
            }
            Some(decision)
        } else {
            None
        };
        Ok(GoalCycle {
            schema_version: 1,
            world,
            drives,
            interpretations,
            evaluations,
            selection,
            behavior,
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
            schema_version: 1,
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
            1.0
        } else {
            epistemic_progress
                .or(target_progress)
                .unwrap_or(movement_progress)
        };
        let attempt_finished = elapsed >= pending.expected_duration_ms
            || (pending.behavior_id == "dock" && world.self_model.charging);
        if let Some(goal) = self
            .goals
            .iter_mut()
            .find(|goal| goal.id() == &pending.goal_id)
        {
            let recent_progress =
                (0.7 * goal.runtime().recent_progress + 0.3 * observed).clamp(0.0, 1.0);
            goal.runtime_mut().recent_progress = recent_progress;
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
            if attempt_finished && observed + 0.1 < pending.expected_progress {
                goal.runtime_mut().failed_attempts =
                    goal.runtime().failed_attempts.saturating_add(1);
            }
            let progress_deficit = (pending.expected_progress - observed).max(0.0);
            let failed = (goal.runtime().failed_attempts as f32 / 5.0).clamp(0.0, 1.0);
            let falling_confidence = (-goal.runtime().confidence_trend).max(0.0);
            let target_frustration =
                (0.5 * progress_deficit + 0.3 * failed + 0.2 * falling_confidence).clamp(0.0, 1.0);
            let alpha = if target_frustration > goal.runtime().frustration {
                0.20
            } else {
                0.07
            };
            let frustration = goal.runtime().frustration
                + (target_frustration - goal.runtime().frustration) * alpha;
            goal.runtime_mut().frustration = frustration;
        }
        if attempt_finished {
            self.pending = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pete_body::BodySense;
    use pete_now::{
        Now, ObjectClass, ObjectObservation, ObjectObservationSource, WorldModelUpdater,
    };

    fn evaluation(id: &str, activation: f32, urgency: f32) -> GoalEvaluation {
        GoalEvaluation {
            goal_id: GoalId::new(id),
            motivation: Motivation {
                activation,
                urgency,
                satisfaction: 0.0,
            },
            competence: Competence {
                confidence: 1.0,
                affordances: vec![affordance(
                    "test",
                    ActionPrimitive::Stop,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    100,
                    None,
                    &[],
                )],
            },
            ..GoalEvaluation::default()
        }
    }

    fn tick_with_canonical_world(
        system: &mut GoalSystem,
        updater: &mut WorldModelUpdater,
        now: Now,
    ) -> GoalCycle {
        let now = updater.update(now, system.world_model_update_context());
        system.tick(&now.world, &[]).unwrap()
    }

    #[test]
    fn world_model_keeps_entity_identity_across_occlusion() {
        let mut updater = WorldModelUpdater::default();
        let mut now = Now::blank(100, BodySense::default());
        now.objects.observations.push(ObjectObservation {
            label: "dock 17".to_string(),
            class: ObjectClass::Charger,
            bearing_rad: 0.2,
            distance_m: Some(1.5),
            confidence: 0.9,
            source: ObjectObservationSource::Sim,
        });
        let first = updater
            .update(now.clone(), WorldModelUpdateContext::default())
            .world;
        now.t_ms = 500;
        now.objects.observations.clear();
        let second = updater
            .update(now, WorldModelUpdateContext::default())
            .world;
        assert_eq!(
            first.entities.keys().collect::<Vec<_>>(),
            second.entities.keys().collect::<Vec<_>>()
        );
        assert_eq!(second.entities.values().next().unwrap().confidence, 0.9);
    }

    #[test]
    fn goal_interpretation_recomputes_relative_bearing_from_world_pose() {
        let mut system = GoalSystem::default();
        let mut updater = WorldModelUpdater::default();
        let mut now = Now::blank(100, BodySense::default());
        now.body.battery_level = 0.2;
        now.objects.observations.push(ObjectObservation {
            label: "dock".to_string(),
            class: ObjectClass::Charger,
            bearing_rad: 0.0,
            distance_m: Some(2.0),
            confidence: 0.9,
            source: ObjectObservationSource::Sim,
        });
        tick_with_canonical_world(&mut system, &mut updater, now.clone());

        now.t_ms = 200;
        now.objects.observations.clear();
        now.body.odometry.heading_rad = std::f32::consts::FRAC_PI_2;
        let cycle = tick_with_canonical_world(&mut system, &mut updater, now);
        let charge = cycle
            .interpretations
            .iter()
            .find(|interpretation| interpretation.goal_id == GoalId::new("seek_charger"))
            .unwrap();
        assert!((charge.target_bearing_rad.unwrap() + std::f32::consts::FRAC_PI_2).abs() < 0.001);
    }

    #[test]
    fn goal_commitment_rejects_small_oscillations() {
        let mut arbiter = GoalArbiter::default();
        let first = arbiter.select(
            0,
            &[
                evaluation("explore", 0.51, 0.0),
                evaluation("charge", 0.50, 0.0),
            ],
        );
        assert_eq!(first.selected_goal, Some(GoalId::new("explore")));
        let second = arbiter.select(
            1_000,
            &[
                evaluation("explore", 0.49, 0.0),
                evaluation("charge", 0.52, 0.0),
            ],
        );
        assert_eq!(second.selected_goal, Some(GoalId::new("explore")));
        assert!(second.retained_by_commitment);
    }

    #[test]
    fn arbitration_is_deterministic_and_does_not_modify_evaluations() {
        let mut alpha = evaluation("alpha", 0.5, 0.0);
        alpha.contributions.push(EvaluationContribution {
            source: "direct_observation".to_string(),
            value: 100.0,
        });
        alpha.competence.affordances[0]
            .provenance
            .push(EvidenceRef {
                id: "sensor:a".to_string(),
                ..EvidenceRef::default()
            });
        let mut beta = evaluation("beta", 0.5, 0.0);
        beta.contributions.push(EvaluationContribution {
            source: "memory_recall".to_string(),
            value: -100.0,
        });
        let evaluations = vec![alpha.clone(), beta.clone()];
        let original = evaluations.clone();
        let first = GoalArbiter::default().select(0, &evaluations);
        assert_eq!(evaluations, original);

        let reversed = vec![beta, alpha];
        let second = GoalArbiter::default().select(0, &reversed);
        assert_eq!(first.selected_goal, second.selected_goal);
    }

    #[test]
    fn goal_components_are_independently_replaceable() {
        let id = GoalId::new("rest");
        let mut goal = GoalModule::new(id.clone());
        goal.interpreter_state.updates = 3;
        goal.evaluator_state.evaluations = 4;
        goal.executor_state.executions = 5;

        goal.replace_interpreter(Box::new(RuleGoalInterpreter { id: id.clone() }));
        assert_eq!(goal.interpreter_state, InterpreterState::default());
        assert_eq!(goal.evaluator_state.evaluations, 4);
        assert_eq!(goal.executor_state.executions, 5);

        goal.replace_evaluator(Box::new(RuleGoalEvaluator { id: id.clone() }));
        assert_eq!(goal.evaluator_state, EvaluatorState::default());
        assert_eq!(goal.executor_state.executions, 5);

        goal.replace_executor(Box::new(UtilityGoalExecutor { id }));
        assert_eq!(goal.executor_state, ExecutorState::default());
    }

    #[test]
    fn adding_a_registered_goal_does_not_change_the_arbiter() {
        let mut system =
            GoalSystem::with_goals(vec![Box::new(GoalModule::new(GoalId::new("rest")))]);
        system
            .register_goal(Box::new(GoalModule::new(GoalId::new("explore"))))
            .unwrap();
        let mut updater = WorldModelUpdater::default();
        let cycle = tick_with_canonical_world(
            &mut system,
            &mut updater,
            Now::blank(0, BodySense::default()),
        );
        assert_eq!(cycle.evaluations.len(), 2);
        assert!(cycle
            .evaluations
            .iter()
            .any(|evaluation| evaluation.goal_id == GoalId::new("explore")));
    }

    #[test]
    fn urgency_reduces_commitment_cost_without_becoming_activation() {
        let mut arbiter = GoalArbiter::default();
        arbiter.select(0, &[evaluation("explore", 0.4, 0.0)]);
        let switched = arbiter.select(
            10,
            &[
                evaluation("explore", 0.4, 0.0),
                evaluation("charge", 0.51, 1.0),
            ],
        );
        assert_eq!(switched.selected_goal, Some(GoalId::new("charge")));
        assert!(switched.switched);
        assert_eq!(switched.effective_minimum_dwell_ms, 0);
    }

    #[test]
    fn completed_goal_releases_commitment_immediately() {
        let mut arbiter = GoalArbiter::default();
        arbiter.select(0, &[evaluation("charge", 0.9, 0.0)]);
        let mut completed = evaluation("charge", 0.9, 0.0);
        completed.disposition = GoalDisposition::Completed;
        let selection = arbiter.select(10, &[completed, evaluation("explore", 0.2, 0.0)]);
        assert_eq!(selection.selected_goal, Some(GoalId::new("explore")));
        assert_eq!(selection.exit_reason, Some(GoalExitReason::Completed));
        assert!(selection.switched);
    }

    #[test]
    fn transient_drive_impulse_decays_and_ordinary_frames_do_not_reset_it() {
        let mut dynamics = DriveDynamics::default();
        let mut world = WorldModelSnapshot::default();
        let mut body = BodySense::default();
        body.battery_level = 0.8;
        world.self_model.battery_level = body.battery_level;
        dynamics.update(&world);
        dynamics.add_impulses(DriveSense {
            battery_hunger: 1.0,
            ..DriveSense::default()
        });
        world.t_ms = 100;
        let pulsed = dynamics.update(&world).energy.activation;
        world.t_ms = 200;
        let recovered = dynamics.update(&world).energy.activation;
        assert!(pulsed > 0.05);
        assert!(recovered < pulsed);
    }

    #[test]
    fn low_confidence_urgent_charge_searches_instead_of_docking() {
        let mut system = GoalSystem::default();
        let mut updater = WorldModelUpdater::default();
        let mut body = BodySense::default();
        body.battery_level = 0.05;
        let now = Now::blank(1_000, body);
        let cycle = tick_with_canonical_world(&mut system, &mut updater, now);
        let behavior = cycle.behavior.unwrap();
        assert_eq!(behavior.goal_id, GoalId::new("seek_charger"));
        assert_eq!(behavior.behavior_id, "systematic_charger_search");
        assert!(matches!(behavior.action, ActionPrimitive::Explore { .. }));
        assert!(behavior.affordance.epistemic_question_id.is_some());
        assert!(behavior.affordance.expected_information_gain > 0.0);
    }

    #[test]
    fn low_confidence_localized_charger_rejects_direct_locomotion() {
        let mut system = GoalSystem::default();
        let mut updater = WorldModelUpdater::default();
        let mut body = BodySense::default();
        body.battery_level = 0.05;
        let mut now = Now::blank(1_000, body);
        now.objects.observations.push(ObjectObservation {
            label: "uncertain dock".to_string(),
            class: ObjectClass::Charger,
            bearing_rad: 0.0,
            distance_m: Some(0.2),
            confidence: 0.1,
            source: ObjectObservationSource::Sim,
        });
        let cycle = tick_with_canonical_world(&mut system, &mut updater, now);
        assert_eq!(
            cycle.behavior.as_ref().unwrap().behavior_id,
            "systematic_charger_search"
        );
        let evaluation = cycle
            .evaluations
            .iter()
            .find(|evaluation| evaluation.goal_id == GoalId::new("seek_charger"))
            .unwrap();
        let dock = evaluation
            .competence
            .affordances
            .iter()
            .find(|affordance| affordance.behavior_id == "dock")
            .unwrap();
        assert!(!dock.available);
        assert!(dock.rejection_reason.is_some());
    }

    #[test]
    fn goal_competence_uses_canonical_drive_capability() {
        let mut system = GoalSystem::default();
        let mut updater = WorldModelUpdater::default();
        let mut body = BodySense::default();
        body.battery_level = 0.05;
        body.health.health = 0.1;
        let mut now = Now::blank(1_000, body);
        now.objects.observations.push(ObjectObservation {
            label: "dock".to_string(),
            class: ObjectClass::Charger,
            bearing_rad: 0.0,
            distance_m: Some(1.0),
            confidence: 0.95,
            source: ObjectObservationSource::Sim,
        });
        let cycle = tick_with_canonical_world(&mut system, &mut updater, now);
        let evaluation = cycle
            .evaluations
            .iter()
            .find(|evaluation| evaluation.goal_id == GoalId::new("seek_charger"))
            .unwrap();
        let approach = evaluation
            .competence
            .affordances
            .iter()
            .find(|affordance| affordance.behavior_id == "approach_charger")
            .unwrap();
        assert!(!approach.available);
        assert_eq!(
            approach.rejection_reason.as_deref(),
            Some("drive is unsafe or body health is degraded")
        );
        assert!(!cycle
            .world
            .self_model
            .capabilities
            .is_available("actuator:drive"));
    }

    #[test]
    fn occluded_charger_selects_search_instead_of_direct_approach() {
        let mut system = GoalSystem::default();
        let mut updater = WorldModelUpdater::default();
        let mut body = BodySense::default();
        body.battery_level = 0.05;
        let mut now = Now::blank(1_000, body);
        now.range.nearest_m = Some(0.5);
        now.objects.observations.extend([
            ObjectObservation {
                label: "blocking obstacle".to_string(),
                class: ObjectClass::Obstacle,
                bearing_rad: 0.02,
                distance_m: Some(0.5),
                confidence: 0.95,
                source: ObjectObservationSource::Sim,
            },
            ObjectObservation {
                label: "dock".to_string(),
                class: ObjectClass::Charger,
                bearing_rad: 0.0,
                distance_m: Some(2.0),
                confidence: 0.95,
                source: ObjectObservationSource::Sim,
            },
        ]);
        let cycle = tick_with_canonical_world(&mut system, &mut updater, now);
        assert_eq!(
            cycle.behavior.as_ref().unwrap().behavior_id,
            "systematic_charger_search"
        );
        let charge = cycle
            .evaluations
            .iter()
            .find(|evaluation| evaluation.goal_id == GoalId::new("seek_charger"))
            .unwrap();
        let approach = charge
            .competence
            .affordances
            .iter()
            .find(|affordance| affordance.behavior_id == "approach_charger")
            .unwrap();
        assert!(!approach.available);
        assert!(approach
            .rejection_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("not currently reachable")));
    }

    #[test]
    fn obstacle_contact_releases_charge_commitment_to_escape() {
        let mut system = GoalSystem::default();
        let mut updater = WorldModelUpdater::default();
        let mut body = BodySense::default();
        body.battery_level = 0.05;
        body.flags.bump_right = true;
        let mut now = Now::blank(1_000, body);
        now.objects.observations.push(ObjectObservation {
            label: "dock".to_string(),
            class: ObjectClass::Charger,
            bearing_rad: 0.0,
            distance_m: Some(1.5),
            confidence: 0.9,
            source: ObjectObservationSource::Sim,
        });
        let cycle = tick_with_canonical_world(&mut system, &mut updater, now);
        assert_eq!(
            cycle.selection.selected_goal,
            Some(GoalId::new("escape_danger"))
        );
        let charge = cycle
            .evaluations
            .iter()
            .find(|evaluation| evaluation.goal_id == GoalId::new("seek_charger"))
            .unwrap();
        assert!(charge
            .competence
            .affordances
            .iter()
            .all(|affordance| !affordance.available));
    }

    #[test]
    fn escape_goal_sequences_behaviors_without_resetting_goal_commitment() {
        let mut system = GoalSystem::default();
        let mut updater = WorldModelUpdater::default();
        let mut body = BodySense::default();
        body.battery_level = 1.0;
        let mut now = Now::blank(0, body);
        now.memory.place_danger = 1.0;
        now.memory.map_confidence = 1.0;
        now.range.beams = vec![1.0; 9];
        let mut behaviors = Vec::new();
        for tick in 0..13 {
            now.t_ms = tick * 100;
            let cycle = tick_with_canonical_world(&mut system, &mut updater, now.clone());
            assert_eq!(
                cycle.selection.selected_goal,
                Some(GoalId::new("escape_danger"))
            );
            behaviors.push(cycle.behavior.unwrap().behavior_id);
        }
        assert!(behaviors[..9]
            .iter()
            .all(|behavior| behavior == "turn_toward_clearance"));
        assert!(behaviors[9..12]
            .iter()
            .all(|behavior| behavior == "probe_clearance"));
        assert_eq!(behaviors[12], "inspect_clearance");
    }

    #[test]
    fn high_confidence_nearby_charger_affords_docking() {
        let mut system = GoalSystem::default();
        let mut updater = WorldModelUpdater::default();
        let mut body = BodySense::default();
        body.battery_level = 0.05;
        let mut now = Now::blank(1_000, body);
        now.objects.observations.push(ObjectObservation {
            label: "dock".to_string(),
            class: ObjectClass::Charger,
            bearing_rad: 0.0,
            distance_m: Some(0.2),
            confidence: 0.98,
            source: ObjectObservationSource::Sim,
        });
        let cycle = tick_with_canonical_world(&mut system, &mut updater, now);
        let behavior = cycle.behavior.unwrap();
        assert_eq!(behavior.goal_id, GoalId::new("seek_charger"));
        assert_eq!(behavior.behavior_id, "dock");
        assert_eq!(behavior.action, ActionPrimitive::Dock);
        assert_eq!(
            behavior
                .affordance
                .skill_request
                .as_ref()
                .map(|request| request.skill_id),
            Some(SkillId::AlignWithDock)
        );
    }

    #[test]
    fn urgent_aligned_charger_approach_requests_possessor_skill() {
        let mut system = GoalSystem::default();
        let mut updater = WorldModelUpdater::default();
        let mut body = BodySense::default();
        body.battery_level = 0.05;
        let mut now = Now::blank(1_000, body);
        now.objects.observations.push(ObjectObservation {
            label: "dock".to_string(),
            class: ObjectClass::Charger,
            bearing_rad: 0.1,
            distance_m: Some(2.0),
            confidence: 0.98,
            source: ObjectObservationSource::Sim,
        });
        let cycle = tick_with_canonical_world(&mut system, &mut updater, now);
        let behavior = cycle.behavior.unwrap();
        assert_eq!(behavior.behavior_id, "approach_charger");
        assert_eq!(
            behavior.action,
            ActionPrimitive::Approach {
                target: ApproachTarget::Charger
            }
        );
        let skill = behavior.affordance.skill_request.unwrap();
        assert_eq!(skill.skill_id, SkillId::ApproachTarget);
        assert_eq!(skill.range_m, Some(2.0));
        assert_eq!(skill.stop_range_m, Some(0.30));
    }

    #[test]
    fn failed_expected_progress_builds_runtime_frustration() {
        let mut system = GoalSystem::default();
        let mut updater = WorldModelUpdater::default();
        let mut body = BodySense::default();
        body.battery_level = 0.05;
        let first = Now::blank(1_000, body.clone());
        tick_with_canonical_world(&mut system, &mut updater, first);
        let second = Now::blank(2_100, body);
        tick_with_canonical_world(&mut system, &mut updater, second);
        let charge = system
            .goals
            .iter()
            .find(|goal| goal.id() == &GoalId::new("seek_charger"))
            .unwrap();
        assert_eq!(charge.runtime().failed_attempts, 1);
        assert!(charge.runtime().frustration > 0.0);
    }

    #[test]
    fn possessor_skill_failure_updates_goal_progress_without_switching_goal() {
        let mut system = GoalSystem::default();
        let mut updater = WorldModelUpdater::default();
        let mut body = BodySense::default();
        body.battery_level = 0.05;
        let mut now = Now::blank(1_000, body);
        now.objects.observations.push(ObjectObservation {
            label: "dock".to_string(),
            class: ObjectClass::Charger,
            bearing_rad: 0.1,
            distance_m: Some(2.0),
            confidence: 0.98,
            source: ObjectObservationSource::Sim,
        });
        let cycle = tick_with_canonical_world(&mut system, &mut updater, now);
        let request = cycle.behavior.unwrap().affordance.skill_request.unwrap();

        system.observe_skill_status(&SkillStatus {
            request,
            phase: SkillPhase::Terminal,
            outcome: Some(SkillOutcome::TimedOut),
            progress: None,
            attempts: 2,
            started_at_ms: Some(1_000),
            updated_at_ms: 2_000,
            reason: Some("no target progress".to_string()),
        });

        let charge = system
            .goals
            .iter()
            .find(|goal| goal.id() == &GoalId::new("seek_charger"))
            .unwrap();
        assert_eq!(charge.runtime().failed_attempts, 1);
        assert_eq!(
            charge.runtime().last_skill_outcome,
            Some(SkillOutcome::TimedOut)
        );
        assert!(charge.runtime().last_progress_observation.is_some());
        assert_eq!(
            system.arbiter.current_goal(),
            Some(&GoalId::new("seek_charger"))
        );
    }

    #[test]
    fn reusable_skills_are_claimed_by_multiple_goals() {
        let mut updater = WorldModelUpdater::default();

        let mut charger_now = Now::blank(1_000, BodySense::default());
        charger_now.body.battery_level = 0.05;
        charger_now.objects.observations.push(ObjectObservation {
            label: "dock".to_string(),
            class: ObjectClass::Charger,
            bearing_rad: 0.8,
            distance_m: Some(2.0),
            confidence: 0.95,
            source: ObjectObservationSource::Sim,
        });
        let mut charger_system = GoalSystem::default();
        let charger = tick_with_canonical_world(&mut charger_system, &mut updater, charger_now);
        let mut aligned_now = Now::blank(3_500, BodySense::default());
        aligned_now.body.battery_level = 0.05;
        aligned_now.objects.observations.push(ObjectObservation {
            label: "aligned dock".to_string(),
            class: ObjectClass::Charger,
            bearing_rad: 0.1,
            distance_m: Some(2.0),
            confidence: 0.95,
            source: ObjectObservationSource::Sim,
        });
        let mut aligned_system = GoalSystem::default();
        let aligned = tick_with_canonical_world(&mut aligned_system, &mut updater, aligned_now);
        assert!(charger
            .evaluations
            .iter()
            .find(|evaluation| evaluation.goal_id == GoalId::new("seek_charger"))
            .unwrap()
            .competence
            .affordances
            .iter()
            .any(|affordance| affordance
                .skill_request
                .as_ref()
                .map(|request| request.skill_id)
                == Some(SkillId::TurnTowardTarget)));

        let mut escape_now = Now::blank(2_000, BodySense::default());
        escape_now.body.flags.bump_left = true;
        escape_now.memory.place_danger = 1.0;
        escape_now.memory.map_confidence = 1.0;
        let mut escape_system = GoalSystem::default();
        let escape = tick_with_canonical_world(&mut escape_system, &mut updater, escape_now);
        assert!(escape
            .evaluations
            .iter()
            .find(|evaluation| evaluation.goal_id == GoalId::new("escape_danger"))
            .unwrap()
            .competence
            .affordances
            .iter()
            .any(|affordance| affordance
                .skill_request
                .as_ref()
                .map(|request| request.skill_id)
                == Some(SkillId::TurnTowardTarget)));

        let mut person_now = Now::blank(3_000, BodySense::default());
        person_now.objects.observations.push(ObjectObservation {
            label: "person".to_string(),
            class: ObjectClass::Person,
            bearing_rad: 0.1,
            distance_m: Some(2.0),
            confidence: 0.9,
            source: ObjectObservationSource::Sim,
        });
        let mut social_system = GoalSystem::default();
        let social = tick_with_canonical_world(&mut social_system, &mut updater, person_now);
        assert!(social
            .evaluations
            .iter()
            .find(|evaluation| evaluation.goal_id == GoalId::new("socialize"))
            .unwrap()
            .competence
            .affordances
            .iter()
            .any(|affordance| affordance
                .skill_request
                .as_ref()
                .map(|request| request.skill_id)
                == Some(SkillId::ApproachTarget)));
        assert!(aligned
            .evaluations
            .iter()
            .find(|evaluation| evaluation.goal_id == GoalId::new("seek_charger"))
            .unwrap()
            .competence
            .affordances
            .iter()
            .any(|affordance| affordance
                .skill_request
                .as_ref()
                .map(|request| request.skill_id)
                == Some(SkillId::ApproachTarget)));

        let task_now = Now::blank(4_000, BodySense::default());
        let task_world = updater.update(task_now, WorldModelUpdateContext::default());
        let mut task_system = GoalSystem::default();
        let task = task_system
            .tick(
                &task_world.world,
                &[ActionPrimitive::Go {
                    intensity: -0.2,
                    duration_ms: 300,
                }],
            )
            .unwrap();
        assert!(task
            .evaluations
            .iter()
            .find(|evaluation| evaluation.goal_id == GoalId::new("follow_task"))
            .unwrap()
            .competence
            .affordances
            .iter()
            .any(|affordance| affordance
                .skill_request
                .as_ref()
                .map(|request| request.skill_id)
                == Some(SkillId::BackAway)));
        assert!(escape
            .evaluations
            .iter()
            .find(|evaluation| evaluation.goal_id == GoalId::new("escape_danger"))
            .unwrap()
            .competence
            .affordances
            .iter()
            .any(|affordance| affordance
                .skill_request
                .as_ref()
                .map(|request| request.skill_id)
                == Some(SkillId::BackAway)));
    }

    #[test]
    fn absent_llm_opinion_does_not_create_uncertainty_pressure() {
        let mut system = GoalSystem::default();
        let mut updater = WorldModelUpdater::default();
        let mut body = BodySense::default();
        body.battery_level = 0.35;
        let cycle = tick_with_canonical_world(&mut system, &mut updater, Now::blank(1_000, body));
        assert_eq!(cycle.drives.certainty.activation, 0.0);
        assert_eq!(
            cycle.selection.selected_goal,
            Some(GoalId::new("seek_charger"))
        );
        assert_eq!(
            cycle.behavior.unwrap().behavior_id,
            "systematic_charger_search"
        );
    }

    #[test]
    fn investigate_publishes_three_targeted_information_gathering_behaviors() {
        let mut system = GoalSystem::default();
        let mut updater = WorldModelUpdater::default();
        let mut body = BodySense::default();
        body.battery_level = 1.0;
        let cycle = tick_with_canonical_world(&mut system, &mut updater, Now::blank(1_000, body));
        let investigate = cycle
            .evaluations
            .iter()
            .find(|evaluation| evaluation.goal_id == GoalId::new("investigate"))
            .unwrap();
        let available = investigate
            .competence
            .affordances
            .iter()
            .filter(|affordance| affordance.available)
            .collect::<Vec<_>>();
        assert!(available.len() >= 3);
        for expected in ["scan_clearance", "inspect_path", "stop_and_observe_path"] {
            let affordance = available
                .iter()
                .find(|affordance| affordance.behavior_id == expected)
                .unwrap();
            assert!(affordance.epistemic_question_id.is_some());
            assert!(affordance.expected_information_gain > 0.0);
        }
    }

    #[test]
    fn social_goal_uses_shared_identity_beliefs_for_its_greeting() {
        let mut uncertain_updater = WorldModelUpdater::default();
        let mut uncertain_now = Now::blank(1_000, BodySense::default());
        uncertain_now.objects.observations.push(ObjectObservation {
            label: "person".to_string(),
            class: ObjectClass::Person,
            bearing_rad: 0.0,
            distance_m: Some(0.6),
            confidence: 0.9,
            source: ObjectObservationSource::Kinect,
        });
        let uncertain_world = uncertain_updater
            .update(uncertain_now, WorldModelUpdateContext::default())
            .world;
        let mut uncertain_system = GoalSystem::default();
        let uncertain = uncertain_system.tick(&uncertain_world, &[]).unwrap();
        let uncertain_social = uncertain
            .evaluations
            .iter()
            .find(|evaluation| evaluation.goal_id == GoalId::new("socialize"))
            .unwrap();
        assert_eq!(
            uncertain_social.competence.affordances[0].action,
            Some(ActionPrimitive::Speak {
                text: "Hello. What should I call you?".to_string()
            })
        );

        let mut known_updater = WorldModelUpdater::default();
        let mut known_now = Now::blank(1_000, BodySense::default());
        known_now.objects.observations.push(ObjectObservation {
            label: "Alex".to_string(),
            class: ObjectClass::Person,
            bearing_rad: 0.0,
            distance_m: Some(0.6),
            confidence: 0.9,
            source: ObjectObservationSource::Kinect,
        });
        let known_world = known_updater
            .update(known_now, WorldModelUpdateContext::default())
            .world;
        let mut known_system = GoalSystem::default();
        let known = known_system.tick(&known_world, &[]).unwrap();
        let known_social = known
            .evaluations
            .iter()
            .find(|evaluation| evaluation.goal_id == GoalId::new("socialize"))
            .unwrap();
        assert_eq!(
            known_social.competence.affordances[0].action,
            Some(ActionPrimitive::Speak {
                text: "Hello, Alex.".to_string()
            })
        );
    }

    #[test]
    fn behavior_expectations_use_the_predicted_clock_domain() {
        let mut system = GoalSystem::default();
        let mut updater = WorldModelUpdater::default();
        let mut body = BodySense::default();
        body.battery_level = 0.05;
        tick_with_canonical_world(&mut system, &mut updater, Now::blank(1_000, body));
        let context = system.world_model_update_context();
        assert_eq!(context.temporal_expectations.len(), 1);
        assert_eq!(
            context.temporal_expectations[0].expected_interval.domain,
            ClockDomain::Predicted
        );
    }
}
