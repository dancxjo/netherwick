use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use pete_actions::{
    ActionPrimitive, ApproachTarget, ExploreStyle, InspectTarget, ReignMode, TurnDir,
};
use pete_now::{DriveSense, Now, ObjectClass};
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

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EntityId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorldEntityKind {
    Charger,
    Person,
    Obstacle,
    SoundSource,
    Landmark,
    Door,
    Region,
    #[default]
    Unknown,
}

impl From<&ObjectClass> for WorldEntityKind {
    fn from(value: &ObjectClass) -> Self {
        match value {
            ObjectClass::Charger => Self::Charger,
            ObjectClass::Person => Self::Person,
            ObjectClass::Obstacle => Self::Obstacle,
            ObjectClass::SoundSource => Self::SoundSource,
            ObjectClass::Landmark => Self::Landmark,
            ObjectClass::Unknown => Self::Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorldPose {
    pub x_m: f32,
    pub y_m: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReachabilityEstimate {
    pub reachable: bool,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRef {
    pub source: String,
    pub key: String,
    pub observed_at_ms: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorldEntity {
    pub id: EntityId,
    pub kind: WorldEntityKind,
    pub label: String,
    pub last_observed_at_ms: u64,
    pub confidence: f32,
    pub pose: Option<WorldPose>,
    pub bearing_rad: Option<f32>,
    pub distance_m: Option<f32>,
    pub reachability: ReachabilityEstimate,
    pub attributes: BTreeMap<String, f32>,
    pub provenance: Vec<EvidenceRef>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GoalStatusSnapshot {
    pub elapsed_time_ms: u64,
    pub failed_attempts: u32,
    pub recent_progress: f32,
    pub confidence_trend: f32,
    pub frustration: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SelfModelSnapshot {
    pub battery_level: f32,
    pub charging: bool,
    pub active_goal: Option<GoalId>,
    pub goal_status: BTreeMap<GoalId, GoalStatusSnapshot>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorldModelSnapshot {
    pub schema_version: u32,
    pub revision: u64,
    pub t_ms: u64,
    pub entities: BTreeMap<EntityId, WorldEntity>,
    pub self_model: SelfModelSnapshot,
}

#[derive(Clone, Debug, Default)]
pub struct WorldModelUpdater {
    revision: u64,
    entities: BTreeMap<EntityId, WorldEntity>,
}

impl WorldModelUpdater {
    pub fn update(
        &mut self,
        now: &Now,
        active_goal: Option<GoalId>,
        goal_status: BTreeMap<GoalId, GoalStatusSnapshot>,
    ) -> WorldModelSnapshot {
        for entity in self.entities.values_mut() {
            let age_ms = now.t_ms.saturating_sub(entity.last_observed_at_ms);
            if age_ms > 1_000 {
                let decay = (age_ms.saturating_sub(1_000) as f32 / 15_000.0).clamp(0.0, 1.0);
                entity.confidence = (entity.confidence * (1.0 - decay)).clamp(0.0, 1.0);
                entity.reachability.confidence =
                    entity.reachability.confidence.min(entity.confidence);
            }
        }

        for observation in &now.objects.observations {
            let kind = WorldEntityKind::from(&observation.class);
            let id = EntityId(format!(
                "{}:{}",
                entity_kind_key(&kind),
                normalized_label(&observation.label)
            ));
            let pose = observation.distance_m.map(|distance| {
                let heading = now.body.odometry.heading_rad + observation.bearing_rad;
                WorldPose {
                    x_m: now.body.odometry.x_m + heading.cos() * distance,
                    y_m: now.body.odometry.y_m + heading.sin() * distance,
                }
            });
            let range_clear = now.range.nearest_m.unwrap_or(f32::INFINITY) > 0.18;
            let reachable = observation.distance_m.is_some() && range_clear;
            let source = format!("object.{:?}", observation.source).to_lowercase();
            self.entities.insert(
                id.clone(),
                WorldEntity {
                    id,
                    kind,
                    label: observation.label.clone(),
                    last_observed_at_ms: now.t_ms,
                    confidence: observation.confidence.clamp(0.0, 1.0),
                    pose,
                    bearing_rad: Some(observation.bearing_rad),
                    distance_m: observation.distance_m,
                    reachability: ReachabilityEstimate {
                        reachable,
                        confidence: observation.confidence.clamp(0.0, 1.0),
                    },
                    attributes: BTreeMap::new(),
                    provenance: vec![EvidenceRef {
                        source,
                        key: observation.label.clone(),
                        observed_at_ms: now.t_ms,
                    }],
                },
            );
        }

        if !now.ear.features.is_empty()
            || now
                .ear
                .transcript
                .as_deref()
                .is_some_and(|v| !v.trim().is_empty())
        {
            let id = EntityId("sound_source:current".to_string());
            self.entities.insert(
                id.clone(),
                WorldEntity {
                    id,
                    kind: WorldEntityKind::SoundSource,
                    label: now
                        .ear
                        .transcript
                        .clone()
                        .unwrap_or_else(|| "unidentified sound".to_string()),
                    last_observed_at_ms: now.t_ms,
                    confidence: now.ear.asr.confidence.clamp(0.2, 1.0),
                    reachability: ReachabilityEstimate {
                        reachable: false,
                        confidence: 0.2,
                    },
                    provenance: vec![EvidenceRef {
                        source: "ear".to_string(),
                        key: "sound_source".to_string(),
                        observed_at_ms: now.t_ms,
                    }],
                    ..WorldEntity::default()
                },
            );
        }

        self.entities.retain(|_, entity| {
            now.t_ms.saturating_sub(entity.last_observed_at_ms) <= 60_000
                && entity.confidence > 0.01
        });
        self.revision = self.revision.saturating_add(1);
        WorldModelSnapshot {
            schema_version: 1,
            revision: self.revision,
            t_ms: now.t_ms,
            entities: self.entities.clone(),
            self_model: SelfModelSnapshot {
                battery_level: now.body.battery_level,
                charging: now.body.charging,
                active_goal,
                goal_status,
            },
        }
    }
}

fn entity_kind_key(kind: &WorldEntityKind) -> &'static str {
    match kind {
        WorldEntityKind::Charger => "charger",
        WorldEntityKind::Person => "person",
        WorldEntityKind::Obstacle => "obstacle",
        WorldEntityKind::SoundSource => "sound_source",
        WorldEntityKind::Landmark => "landmark",
        WorldEntityKind::Door => "door",
        WorldEntityKind::Region => "region",
        WorldEntityKind::Unknown => "unknown",
    }
}

fn normalized_label(label: &str) -> String {
    let normalized = label
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    if normalized.is_empty() {
        "unlabeled".to_string()
    } else {
        normalized
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
}

impl DriveDynamics {
    pub fn update(&mut self, now: &Now, world: &WorldModelSnapshot) -> DriveSnapshot {
        let dt_s = self
            .last_t_ms
            .map(|last| now.t_ms.saturating_sub(last) as f32 / 1_000.0)
            .unwrap_or(0.0)
            .clamp(0.0, 5.0);
        self.last_t_ms = Some(now.t_ms);

        let moving =
            now.body.velocity.forward_m_s.abs() > 0.01 || now.body.velocity.turn_rad_s.abs() > 0.01;
        let fatigue_delta = if now.body.charging {
            -0.01 * dt_s
        } else if moving {
            0.003 * dt_s
        } else {
            0.001 * dt_s
        };
        self.fatigue = (self.fatigue + fatigue_delta).clamp(0.0, 1.0);

        let predicted_energy = (now.body.battery_level
            + now
                .predictions
                .charge_model
                .or(now.predictions.charge_hardcoded)
                .map(|p| p.expected_battery_delta)
                .unwrap_or(-0.01))
        .clamp(0.0, 1.0);
        self.snapshot.energy.update(
            0.80,
            if now.body.charging {
                1.0
            } else {
                now.body.battery_level
            },
            if now.body.charging {
                1.0
            } else {
                predicted_energy
            },
            dt_s,
            now.drives.battery_hunger.clamp(0.0, 1.0) * 0.35,
        );

        let predicted_danger = now
            .predictions
            .danger_model
            .or(now.predictions.danger_hardcoded)
            .map(|p| {
                p.bump_risk
                    .max(p.cliff_risk)
                    .max(p.wheel_drop_risk)
                    .max(p.stuck_risk)
            })
            .unwrap_or(0.0);
        let contact = if now.body.flags.bump_left
            || now.body.flags.bump_right
            || now.body.flags.wall
            || now.body.flags.wheel_drop
        {
            1.0
        } else {
            0.0
        };
        let range_risk = now
            .range
            .nearest_m
            .map(|distance| ((0.35 - distance) / 0.35).clamp(0.0, 1.0))
            .unwrap_or(0.0);
        let risk = predicted_danger
            .max(now.memory.place_danger)
            .max(range_risk)
            .max(contact);
        self.snapshot.safety.update(
            0.95,
            1.0 - risk,
            1.0 - predicted_danger,
            dt_s,
            (contact * 0.4).max(now.drives.danger_avoidance.clamp(0.0, 1.0) * 0.4),
        );

        let novelty = now.memory.place_novelty.clamp(0.0, 1.0);
        self.snapshot.curiosity.update(
            0.60,
            novelty,
            novelty.max(now.surprise.total),
            dt_s,
            (now.surprise.total.clamp(0.0, 1.0) * 0.25)
                .max(now.drives.curiosity.clamp(0.0, 1.0) * 0.25),
        );

        let person_confidence = world
            .entities
            .values()
            .filter(|entity| entity.kind == WorldEntityKind::Person)
            .map(|entity| entity.confidence)
            .fold(0.0f32, f32::max)
            .max(now.memory.place_social_value);
        self.snapshot
            .social
            .update(0.50, person_confidence, person_confidence, dt_s, 0.0);
        self.snapshot
            .rest
            .update(0.80, 1.0 - self.fatigue, 1.0 - self.fatigue, dt_s, 0.0);
        let llm_certainty = if now.llm.command_summary.is_some() || now.llm.critique.is_some() {
            now.llm.confidence.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let certainty = (1.0 - now.predictions.uncertainty)
            .min(llm_certainty)
            .clamp(0.0, 1.0);
        self.snapshot
            .certainty
            .update(0.85, certainty, certainty, dt_s, 0.0);
        self.snapshot.schema_version = 1;
        self.snapshot.t_ms = now.t_ms;
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
    pub confidence: f32,
    pub expected_reward: f32,
    pub expected_progress: f32,
    pub expected_risk: f32,
    pub expected_energy_cost: f32,
    pub expected_duration_ms: u64,
    pub target: Option<EntityId>,
    pub action: Option<ActionPrimitive>,
    pub provenance: Vec<EvidenceRef>,
}

impl Affordance {
    fn utility(&self) -> f32 {
        0.25 * self.confidence + 0.25 * self.expected_reward + 0.35 * self.expected_progress
            - 0.25 * self.expected_risk
            - 0.10 * self.expected_energy_cost
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
    pub motivation: Motivation,
    pub competence: Competence,
    pub contributions: Vec<EvaluationContribution>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GoalInterpretation {
    pub goal_id: GoalId,
    pub target: Option<EntityId>,
    pub target_confidence: f32,
    pub target_bearing_rad: Option<f32>,
    pub target_distance_m: Option<f32>,
    pub danger: f32,
    pub novelty: f32,
    pub social_presence: f32,
    pub uncertainty: f32,
    pub stalled_goal_frustration: f32,
    pub suggestions: Vec<ActionPrimitive>,
    pub provenance: Vec<EvidenceRef>,
}

pub type GoalInterpretationSnapshot = GoalInterpretation;

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
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GoalRuntimeState {
    pub elapsed_time_ms: u64,
    pub failed_attempts: u32,
    pub recent_progress: f32,
    pub confidence_trend: f32,
    pub frustration: f32,
    pub last_confidence: Option<f32>,
}

impl GoalRuntimeState {
    fn snapshot(&self) -> GoalStatusSnapshot {
        GoalStatusSnapshot {
            elapsed_time_ms: self.elapsed_time_ms,
            failed_attempts: self.failed_attempts,
            recent_progress: self.recent_progress,
            confidence_trend: self.confidence_trend,
            frustration: self.frustration,
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
    pub now: &'a Now,
    pub world: &'a WorldModelSnapshot,
    pub drives: &'a DriveSnapshot,
    pub runtime: &'a GoalRuntimeState,
    pub proposals: &'a [ActionPrimitive],
}

pub struct GoalEvaluationContext<'a> {
    pub now: &'a Now,
    pub world: &'a WorldModelSnapshot,
    pub drives: &'a DriveSnapshot,
    pub runtime: &'a GoalRuntimeState,
}

pub struct GoalExecutionContext<'a> {
    pub world: &'a WorldModelSnapshot,
    pub runtime: &'a GoalRuntimeState,
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
        Self {
            interpreter: Box::new(RuleGoalInterpreter { id: id.clone() }),
            evaluator: Box::new(RuleGoalEvaluator { id: id.clone() }),
            executor: Box::new(UtilityGoalExecutor),
            id,
            interpreter_state: InterpreterState::default(),
            evaluator_state: EvaluatorState::default(),
            executor_state: ExecutorState::default(),
            runtime: GoalRuntimeState::default(),
            last_interpretation: None,
            last_evaluation: None,
        }
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
                    goal_entity_score(left, context.now)
                        .total_cmp(&goal_entity_score(right, context.now))
                })
        });
        let target_relative = target.and_then(|entity| {
            entity.pose.map(|pose| {
                let dx = pose.x_m - context.now.body.odometry.x_m;
                let dy = pose.y_m - context.now.body.odometry.y_m;
                let distance = (dx * dx + dy * dy).sqrt();
                let bearing = normalize_angle(dy.atan2(dx) - context.now.body.odometry.heading_rad);
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
            target_confidence: target.map(|entity| entity.confidence).unwrap_or_else(|| {
                if self.id.as_str() == "seek_charger" {
                    context
                        .now
                        .predictions
                        .charge_model
                        .or(context.now.predictions.charge_hardcoded)
                        .map(|p| p.confidence * p.charge_probability)
                        .unwrap_or(context.now.memory.place_charge_value)
                } else {
                    0.0
                }
            }),
            target_bearing_rad: target_relative
                .map(|(bearing, _)| bearing)
                .or_else(|| target.and_then(|entity| entity.bearing_rad))
                .or_else(|| {
                    (self.id.as_str() == "seek_charger")
                        .then_some(context.now.memory.nearby_best_charge_direction_rad)
                        .flatten()
                }),
            target_distance_m: target_relative
                .map(|(_, distance)| distance)
                .or_else(|| target.and_then(|entity| entity.distance_m)),
            danger,
            novelty: context.now.memory.place_novelty,
            social_presence: context.drives.social.actual,
            uncertainty: context.drives.certainty.activation,
            stalled_goal_frustration,
            suggestions: context.proposals.to_vec(),
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

fn goal_entity_score(entity: &WorldEntity, now: &Now) -> f32 {
    let distance = entity
        .pose
        .map(|pose| {
            let dx = pose.x_m - now.body.odometry.x_m;
            let dy = pose.y_m - now.body.odometry.y_m;
            (dx * dx + dy * dy).sqrt()
        })
        .or(entity.distance_m)
        .unwrap_or(10.0);
    entity.confidence / (1.0 + distance.max(0.0))
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

        if let Some(reign) = context.now.reign.latest.as_ref() {
            if matches!(reign.mode, ReignMode::Assist | ReignMode::Suggest) {
                if let Some(action) = reign.command.to_action() {
                    if affordances
                        .iter()
                        .any(|affordance| affordance.action.as_ref() == Some(&action))
                    {
                        let scale = if reign.mode == ReignMode::Assist {
                            0.60
                        } else {
                            0.25
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
        }
        let confidence = affordances
            .iter()
            .filter(|affordance| affordance.available)
            .map(|affordance| affordance.confidence)
            .fold(0.0f32, f32::max)
            .clamp(0.0, 1.0);
        let evaluation = GoalEvaluation {
            goal_id: self.id.clone(),
            t_ms: context.now.t_ms,
            world_revision: context.world.revision,
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

struct UtilityGoalExecutor;

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
        if context.runtime.frustration > 0.6 {
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
        let affordance = candidates
            .first()
            .copied()
            .ok_or_else(|| {
                anyhow!(
                    "goal {} has no executable affordance",
                    evaluation.goal_id.as_str()
                )
            })?
            .clone();
        let action = affordance
            .action
            .clone()
            .ok_or_else(|| anyhow!("selected affordance has no action"))?;
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
            },
        ))
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
        confidence: confidence.clamp(0.0, 1.0),
        expected_reward: reward.clamp(-1.0, 1.0),
        expected_progress: progress.clamp(0.0, 1.0),
        expected_risk: risk.clamp(0.0, 1.0),
        expected_energy_cost: energy.clamp(0.0, 1.0),
        expected_duration_ms: duration_ms,
        target,
        action: Some(action),
        provenance: provenance.to_vec(),
    }
}

type EvaluationParts = (f32, f32, f32, Vec<Affordance>, Vec<EvaluationContribution>);

fn evaluate_seek_charger(
    interpretation: &GoalInterpretation,
    context: &GoalEvaluationContext<'_>,
) -> EvaluationParts {
    let energy = context.drives.energy.activation;
    let urgency = ((0.25 - context.now.body.battery_level) / 0.20).clamp(0.0, 1.0);
    let confidence = interpretation.target_confidence;
    let mut affordances = Vec::new();
    if let Some(distance) = interpretation.target_distance_m {
        if distance <= 0.35 {
            affordances.push(affordance(
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
            ));
        }
    }
    if let Some(bearing) = interpretation.target_bearing_rad {
        if bearing.abs() > 0.20 {
            affordances.push(affordance(
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
            ));
        } else {
            affordances.push(affordance(
                "approach_charger",
                ActionPrimitive::Drive {
                    forward: 0.40,
                    turn: (bearing * 1.2).clamp(-0.35, 0.35),
                    duration_ms: 1_000,
                },
                confidence,
                0.8,
                0.9,
                interpretation.danger,
                0.15,
                1_000,
                interpretation.target.clone(),
                &interpretation.provenance,
            ));
        }
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
    affordances.push(affordance(
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
    ));
    if urgency > 0.8 && confidence < 0.2 && context.runtime.frustration > 0.6 {
        affordances.push(affordance(
            "request_charge_help",
            ActionPrimitive::Speak {
                text: "I need help finding the charger.".to_string(),
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
    (
        (0.85 * energy + 0.15 * confidence).clamp(0.0, 1.0),
        urgency,
        context.drives.energy.satisfaction,
        affordances,
        vec![
            contribution("drive.energy", energy),
            contribution("world.charger_confidence", confidence),
        ],
    )
}

fn evaluate_escape(
    interpretation: &GoalInterpretation,
    context: &GoalEvaluationContext<'_>,
) -> EvaluationParts {
    let danger = interpretation.danger;
    let contact = context.now.body.flags.bump_left
        || context.now.body.flags.bump_right
        || context.now.body.flags.wall;
    let confidence = context.now.memory.map_confidence.max(0.5);
    let direction = context
        .now
        .memory
        .nearby_best_safe_direction_rad
        .map(|bearing| {
            if bearing >= 0.0 {
                TurnDir::Left
            } else {
                TurnDir::Right
            }
        })
        .unwrap_or_else(|| {
            if context.now.body.flags.bump_left {
                TurnDir::Right
            } else {
                TurnDir::Left
            }
        });
    let mut affordances = Vec::new();
    if contact {
        affordances.push(affordance(
            "reverse_from_contact",
            ActionPrimitive::Go {
                intensity: -0.18,
                duration_ms: 300,
            },
            0.95,
            0.7,
            0.8,
            0.1,
            0.05,
            300,
            None,
            &[],
        ));
    }
    affordances.push(affordance(
        "turn_toward_clearance",
        ActionPrimitive::Turn {
            direction,
            intensity: 0.55,
            duration_ms: 800,
        },
        confidence,
        0.65,
        0.7,
        0.15,
        0.08,
        800,
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
    let action = if let Some(bearing) = context.now.memory.nearby_frontier_direction_rad {
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
    let confidence = interpretation.target_confidence;
    let action = match interpretation.target_distance_m {
        Some(distance) if distance <= 0.8 => ActionPrimitive::Speak {
            text: "Hello.".to_string(),
        },
        Some(_) => ActionPrimitive::Approach {
            target: ApproachTarget::Person,
        },
        None => ActionPrimitive::Inspect {
            target: InspectTarget::Person,
        },
    };
    (
        (0.70 * social + 0.30 * confidence
            - 0.60 * interpretation.danger
            - 0.40 * context.drives.rest.activation)
            .clamp(0.0, 1.0),
        0.2,
        context.drives.social.satisfaction,
        vec![affordance(
            "social_engagement",
            action,
            confidence.max(0.25),
            0.55,
            0.55,
            interpretation.danger,
            0.1,
            1_000,
            interpretation.target.clone(),
            &interpretation.provenance,
        )],
        vec![
            contribution("drive.social", social),
            contribution("world.person_confidence", confidence),
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
        if context.now.body.charging {
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
    let action = if interpretation.target.is_some() {
        ActionPrimitive::Inspect {
            target: InspectTarget::Sound,
        }
    } else {
        ActionPrimitive::Inspect {
            target: InspectTarget::Novelty,
        }
    };
    (
        (0.65 * uncertainty + 0.25 * context.now.surprise.total + 0.35 * frustration
            - 0.50 * interpretation.danger)
            .clamp(0.0, 1.0),
        (0.25 + frustration * 0.5).clamp(0.0, 1.0),
        context.drives.certainty.satisfaction,
        vec![affordance(
            "gather_evidence",
            action,
            (1.0 - uncertainty).max(0.3),
            0.45,
            0.6,
            interpretation.danger * 0.25,
            0.05,
            750,
            interpretation.target.clone(),
            &interpretation.provenance,
        )],
        vec![
            contribution("drive.certainty", uncertainty),
            contribution("self.stalled_goal", frustration),
        ],
    )
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
            affordance(
                &format!("task_proposal_{index}"),
                action,
                context.now.llm.confidence.max(0.5),
                0.5,
                0.5,
                interpretation.danger,
                0.1,
                1_000,
                None,
                &[],
            )
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
    pub switched: bool,
    pub retained_by_commitment: bool,
    pub reason: String,
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

    pub fn select(&mut self, now_ms: u64, evaluations: &[GoalEvaluation]) -> GoalSelection {
        let eligible = evaluations
            .iter()
            .filter(|evaluation| {
                (evaluation.motivation.satisfaction < 0.999
                    || evaluation.motivation.activation > 0.05)
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
        let incumbent = incumbent_id.as_ref().and_then(|id| {
            eligible
                .iter()
                .copied()
                .find(|evaluation| &evaluation.goal_id == id)
        });

        let Some(challenger) = challenger else {
            self.commitment = None;
            return GoalSelection {
                incumbent_goal: incumbent_id,
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
                switched: true,
                reason: "selected initial goal".to_string(),
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
                switched: true,
                reason: "incumbent completed, failed, or lost all affordances".to_string(),
                ..GoalSelection::default()
            };
        };

        if challenger.goal_id == incumbent.goal_id {
            return GoalSelection {
                selected_goal: Some(incumbent.goal_id.clone()),
                incumbent_goal: Some(incumbent.goal_id.clone()),
                reason: "incumbent remains most active".to_string(),
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
                switched: true,
                reason: "challenger overcame persistence and switching cost".to_string(),
                effective_switching_cost,
                effective_minimum_dwell_ms,
                ..GoalSelection::default()
            }
        } else {
            GoalSelection {
                selected_goal: Some(incumbent.goal_id.clone()),
                incumbent_goal: Some(incumbent.goal_id.clone()),
                retained_by_commitment: true,
                reason: if dwell_ms < effective_minimum_dwell_ms {
                    "incumbent retained during commitment dwell".to_string()
                } else {
                    "challenger did not overcome persistence and switching cost".to_string()
                },
                effective_switching_cost,
                effective_minimum_dwell_ms,
                ..GoalSelection::default()
            }
        }
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
}

pub struct GoalSystem {
    world: WorldModelUpdater,
    drives: DriveDynamics,
    goals: Vec<GoalModule>,
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
        .map(|id| GoalModule::new(GoalId::new(id)))
        .collect();
        Self {
            world: WorldModelUpdater::default(),
            drives: DriveDynamics::default(),
            goals,
            arbiter: GoalArbiter::default(),
            pending: None,
            last_tick_ms: None,
        }
    }
}

impl GoalSystem {
    pub fn tick(&mut self, now: &Now, proposals: &[ActionPrimitive]) -> Result<GoalCycle> {
        let goal_status = self
            .goals
            .iter()
            .map(|goal| (goal.id.clone(), goal.runtime.snapshot()))
            .collect();
        let world = self
            .world
            .update(now, self.arbiter.current_goal().cloned(), goal_status);
        self.observe_pending_outcome(now, &world);
        let drives = self.drives.update(now, &world);
        let mut interpretations = Vec::with_capacity(self.goals.len());
        let mut evaluations = Vec::with_capacity(self.goals.len());
        for goal in &mut self.goals {
            let runtime = goal.runtime.clone();
            let interpretation = goal.interpret(&GoalInterpretationContext {
                now,
                world: &world,
                drives: &drives,
                runtime: &runtime,
                proposals,
            })?;
            let evaluation = goal.evaluate(
                &interpretation,
                &GoalEvaluationContext {
                    now,
                    world: &world,
                    drives: &drives,
                    runtime: &runtime,
                },
            )?;
            interpretations.push(interpretation);
            evaluations.push(evaluation);
        }

        let previous_goal = self.arbiter.current_goal().cloned();
        let selection = self.arbiter.select(now.t_ms, &evaluations);
        if selection.switched {
            if let Some(previous) = previous_goal {
                if let Some(goal) = self.goals.iter_mut().find(|goal| goal.id == previous) {
                    goal.runtime.elapsed_time_ms = 0;
                }
            }
        }
        let dt_ms = self
            .last_tick_ms
            .map(|last| now.t_ms.saturating_sub(last))
            .unwrap_or(0);
        self.last_tick_ms = Some(now.t_ms);
        let behavior = if let Some(goal_id) = selection.selected_goal.as_ref() {
            let index = self
                .goals
                .iter()
                .position(|goal| &goal.id == goal_id)
                .ok_or_else(|| anyhow!("selected goal is not registered"))?;
            self.goals[index].runtime.elapsed_time_ms = self.goals[index]
                .runtime
                .elapsed_time_ms
                .saturating_add(dt_ms);
            let evaluation = evaluations
                .iter()
                .find(|evaluation| &evaluation.goal_id == goal_id)
                .ok_or_else(|| anyhow!("selected goal has no immutable evaluation"))?;
            let runtime = self.goals[index].runtime.clone();
            let decision = self.goals[index].execute(
                evaluation,
                &GoalExecutionContext {
                    world: &world,
                    runtime: &runtime,
                },
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
                self.pending = Some(PendingOutcome {
                    goal_id: decision.goal_id.clone(),
                    behavior_id: decision.behavior_id.clone(),
                    started_at_ms: now.t_ms,
                    expected_progress: decision.affordance.expected_progress,
                    expected_duration_ms: decision.affordance.expected_duration_ms,
                    start_pose: (now.body.odometry.x_m, now.body.odometry.y_m),
                    start_target_distance_m: decision
                        .affordance
                        .target
                        .as_ref()
                        .and_then(|id| world.entities.get(id))
                        .and_then(|entity| entity.distance_m),
                    target: decision.affordance.target.clone(),
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

    fn observe_pending_outcome(&mut self, now: &Now, world: &WorldModelSnapshot) {
        let Some(pending) = self.pending.clone() else {
            return;
        };
        let elapsed = now.t_ms.saturating_sub(pending.started_at_ms);
        let dx = now.body.odometry.x_m - pending.start_pose.0;
        let dy = now.body.odometry.y_m - pending.start_pose.1;
        let movement_progress = ((dx * dx + dy * dy).sqrt() / 0.5).clamp(0.0, 1.0);
        let target_progress = pending
            .target
            .as_ref()
            .and_then(|target| world.entities.get(target))
            .and_then(|entity| entity.distance_m)
            .zip(pending.start_target_distance_m)
            .map(|(current, start)| ((start - current) / start.max(0.1)).clamp(0.0, 1.0));
        let observed = if pending.behavior_id == "dock" && now.body.charging {
            1.0
        } else {
            target_progress.unwrap_or(movement_progress)
        };
        let attempt_finished = elapsed >= pending.expected_duration_ms
            || (pending.behavior_id == "dock" && now.body.charging);
        if let Some(goal) = self
            .goals
            .iter_mut()
            .find(|goal| goal.id == pending.goal_id)
        {
            goal.runtime.recent_progress =
                (0.7 * goal.runtime.recent_progress + 0.3 * observed).clamp(0.0, 1.0);
            if let Some(evaluation) = goal.last_evaluation.as_ref() {
                let trend = evaluation.competence.confidence
                    - goal
                        .runtime
                        .last_confidence
                        .unwrap_or(evaluation.competence.confidence);
                goal.runtime.confidence_trend =
                    (0.8 * goal.runtime.confidence_trend + 0.2 * trend).clamp(-1.0, 1.0);
                goal.runtime.last_confidence = Some(evaluation.competence.confidence);
            }
            if attempt_finished && observed + 0.1 < pending.expected_progress {
                goal.runtime.failed_attempts = goal.runtime.failed_attempts.saturating_add(1);
            }
            let progress_deficit = (pending.expected_progress - observed).max(0.0);
            let failed = (goal.runtime.failed_attempts as f32 / 5.0).clamp(0.0, 1.0);
            let falling_confidence = (-goal.runtime.confidence_trend).max(0.0);
            let target_frustration =
                (0.5 * progress_deficit + 0.3 * failed + 0.2 * falling_confidence).clamp(0.0, 1.0);
            let alpha = if target_frustration > goal.runtime.frustration {
                0.20
            } else {
                0.07
            };
            goal.runtime.frustration += (target_frustration - goal.runtime.frustration) * alpha;
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
    use pete_now::{ObjectObservation, ObjectObservationSource};

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
        let first = updater.update(&now, None, BTreeMap::new());
        now.t_ms = 500;
        now.objects.observations.clear();
        let second = updater.update(&now, None, BTreeMap::new());
        assert_eq!(
            first.entities.keys().collect::<Vec<_>>(),
            second.entities.keys().collect::<Vec<_>>()
        );
        assert_eq!(second.entities.values().next().unwrap().confidence, 0.9);
    }

    #[test]
    fn goal_interpretation_recomputes_relative_bearing_from_world_pose() {
        let mut system = GoalSystem::default();
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
        system.tick(&now, &[]).unwrap();

        now.t_ms = 200;
        now.objects.observations.clear();
        now.body.odometry.heading_rad = std::f32::consts::FRAC_PI_2;
        let cycle = system.tick(&now, &[]).unwrap();
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
    fn low_confidence_urgent_charge_searches_instead_of_docking() {
        let mut system = GoalSystem::default();
        let mut body = BodySense::default();
        body.battery_level = 0.05;
        let now = Now::blank(1_000, body);
        let cycle = system.tick(&now, &[]).unwrap();
        let behavior = cycle.behavior.unwrap();
        assert_eq!(behavior.goal_id, GoalId::new("seek_charger"));
        assert_eq!(behavior.behavior_id, "systematic_charger_search");
        assert!(matches!(behavior.action, ActionPrimitive::Explore { .. }));
    }

    #[test]
    fn high_confidence_nearby_charger_affords_docking() {
        let mut system = GoalSystem::default();
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
        let cycle = system.tick(&now, &[]).unwrap();
        let behavior = cycle.behavior.unwrap();
        assert_eq!(behavior.goal_id, GoalId::new("seek_charger"));
        assert_eq!(behavior.behavior_id, "dock");
        assert_eq!(behavior.action, ActionPrimitive::Dock);
    }

    #[test]
    fn urgent_aligned_charger_approach_uses_bounded_fast_drive() {
        let mut system = GoalSystem::default();
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
        let cycle = system.tick(&now, &[]).unwrap();
        let behavior = cycle.behavior.unwrap();
        assert_eq!(behavior.behavior_id, "approach_charger");
        assert!(matches!(
            behavior.action,
            ActionPrimitive::Drive { forward, turn, .. }
                if forward == 0.40 && turn.abs() <= 0.35
        ));
    }

    #[test]
    fn failed_expected_progress_builds_runtime_frustration() {
        let mut system = GoalSystem::default();
        let mut body = BodySense::default();
        body.battery_level = 0.05;
        let first = Now::blank(1_000, body.clone());
        system.tick(&first, &[]).unwrap();
        let second = Now::blank(2_100, body);
        system.tick(&second, &[]).unwrap();
        let charge = system
            .goals
            .iter()
            .find(|goal| goal.id == GoalId::new("seek_charger"))
            .unwrap();
        assert_eq!(charge.runtime.failed_attempts, 1);
        assert!(charge.runtime.frustration > 0.0);
    }

    #[test]
    fn absent_llm_opinion_does_not_create_uncertainty_pressure() {
        let mut system = GoalSystem::default();
        let mut body = BodySense::default();
        body.battery_level = 0.35;
        let cycle = system.tick(&Now::blank(1_000, body), &[]).unwrap();
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
}
