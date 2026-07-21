mod sleep;
mod social_exam;

pub use sleep::*;
pub use social_exam::*;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use pete_actions::{
    action_to_motor_command, ActionPrimitive, ApproachTarget, ChirpPattern, ExploreStyle,
    InspectTarget, LlmActionProposal, LlmAdvisoryAction, LlmAdvisoryActionDisposition,
    LlmAdvisoryActionSource, ReignInput, ReignMode, ReignOutcome, ReignSource, TurnDir,
};
use pete_autonomic::{SafetyLayer, SafetyReason};
use pete_behaviors::{
    BehaviorConfig, BehaviorImplementation, BehaviorNodeState, BehaviorNodeUpdate, BehaviorRegime,
    BehaviorRegistryConfig, ErasedBehaviorRunRecord, FallbackPolicy, FunctionBehavior,
    ReplaceableBehavior, TargetExtractor, TrainingSample, TrainingSource,
};
use pete_body::{BodyFlags, BodySense};
use pete_cockpit::{
    Cockpit, CockpitEventKind, ContactWithdrawalEvent, ContactWithdrawalOutcome, DockIrCue,
    MotionCommand, MotorCommand, SafeCockpit, SafeStopReason, SafetyLatchKind, StatusSummary,
};
use pete_conductor::{
    Conductor, ConductorInput, GoalSystem, NavigationIntent, SimpleConductor, SkillId,
    SkillOutcome, SkillPhase, SkillRequest, SkillStatus,
};
use pete_core::{Pose2, Provenance, Reward, TimeMs};
use pete_events::{default_event_bus, DriveName, EventBus, EventContext, EventExtractor, Response};
use pete_experience::{
    action_features, action_value_input_from_transition_like,
    action_value_target_from_reward_surprise, charge_input_from_transition_like,
    charge_target_from_transition_like, danger_input_from_transition_like,
    danger_target_from_transition_like, ear_next_input_from_transition_like,
    ear_next_target_from_now, experience_decode_target_from_now,
    eye_next_input_from_transition_like, eye_next_target_from_now, ActionValueInput,
    ActionValueOutput, BaselineRewardComputer, BaselineSurpriseComputer, ChargeInput, ChargeOutput,
    DangerInput, DangerOutput, EarNextInput, EarNextOutput, Experience, ExperienceBehaviorInput,
    ExperienceBehaviorOutput, ExperienceDecodeOutput, ExperienceEncodeInput, ExperienceInstant,
    ExperienceLatent, EyeNextInput, EyeNextOutput, FutureInput, FuturePrediction, FuturePredictor,
    Impression, Prediction, RewardComputer, Sensation, StasisFuturePredictor, SurpriseComputer,
};
use pete_ledger::{
    ExperienceFrame, ExperienceTransition, LedgerWriter, PendingFrame, TransitionBuilder,
};
use pete_llm::{Combobulation, LiveImageCognition, LiveImageEnricher, LlmAgent, LlmTickResult};
use pete_map::{observation_from_now, LocalMap, LoopClosureCandidateInput, MAP_EXTENSION_NAME};
use pete_memory::{
    attach_memory_links_to_frame, place_recognition_input_from_query_now, MemoryStore,
    PlaceRecognitionCandidate, PlaceRecognitionKind, Recall, RecallBundle, RecallQuery,
};
use pete_models::{
    read_action_value_metadata, read_charge_metadata, read_danger_metadata, read_ear_next_metadata,
    read_experience_autoencoder_metadata, read_eye_next_metadata, read_future_metadata,
    ActionValueNetTrainer, ChargeNetTrainer, CopyCurrentEarPredictor, CopyCurrentEyePredictor,
    DangerNetTrainer, EarNextNetTrainer, ExperienceAutoencoderTrainer, EyeNextNetTrainer,
    FutureNetTrainer, HardcodedActionValuePredictor, HardcodedChargePredictor,
    HardcodedDangerPredictor,
};
use pete_neat::{
    HardcodedLocomotionBehavior, LocomotionInput, LocomotionOutput, LocomotionTracker,
    NeatLocomotionBehavior,
};
use pete_now::{
    ActionValuePrediction, ActiveControlSummary, BeliefMeta, BeliefSourceKind, ChargePrediction,
    CognitiveServiceBelief, ControlProvenance, DangerPrediction, DriveSense, EarPrediction,
    EntityId, EvidenceRef, ExtensionSense, EyePrediction, Freshness, MemorySense, Now, ObjectClass,
    ObjectObservation, ObjectObservationSource, ReignSense, SafetySense, SemanticBehaviorId,
    SemanticEvidenceObservation, SemanticGroundingKind, SemanticNodeRef, SemanticOutcomeId,
    SemanticPredicate, SurpriseSense, WorldModelSnapshot, WorldModelUpdater,
};
use pete_sensors::{
    anticipate_surfaces, FrameProcessor, NowBuilder, SenseProducer, SurfaceExtractor,
    SurfaceExtractorOutput, World, WorldSnapshot,
};
use pete_sim::{SimCockpit, VirtualWorld};
use pete_skills::{
    BodyResource, HazardKind, HostOperation, LuaSkillConfig, LuaSkillRuntime, OperationContext,
    OrganDriver, OrganPoll, PrimitiveIntent, SkillFailure,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::task::JoinHandle;
use tsrun::{js_value_to_json, Interpreter, JsError, StepResult};
use uuid::Uuid;

pub struct MinimalRuntime<L, M, R, C, S, A>
where
    L: LedgerWriter + Sync,
    M: MemoryStore,
    R: Recall + Sync,
    C: Conductor,
    S: SafetyLayer,
    A: LlmAgent + 'static,
{
    pub ledger: L,
    pub memory_store: M,
    pub memory_recall: R,
    pub conductor: C,
    pub safety: S,
    /// Optional higher cognition is shared only with a background job.  The
    /// control tick never takes this mutex; it is exclusively an ownership
    /// device for the spawned provider future.
    pub llm: Arc<tokio::sync::Mutex<A>>,
    pub extractor: EventExtractor,
    pub bus: EventBus,
    pub reign_queue: Arc<Mutex<ReignQueue>>,
    pub predictor: StasisFuturePredictor,
    pub models: RuntimeModelStack,
    pub action_selector_mode: ActionSelectorMode,
    pub surprise_computer: BaselineSurpriseComputer,
    pub reward_computer: BaselineRewardComputer,
    pub transition_builder: TransitionBuilder,
    pub behavior_training_hub: BehaviorTrainingHub,
    pub surface_extractor: SurfaceExtractor,
    pub inline_learning: InlineLearningConfig,
    pub nudge_policy: NudgePolicy,
    pub local_map: LocalMap,
    pub last_behavior_runs: Vec<ErasedBehaviorRunRecord>,
    locomotion_tracker: LocomotionTracker,
    chirp_events: ChirpEventState,
    nudge: NudgeController,
    goal_system: GoalSystem,
    world_model: WorldModelUpdater,
    sleep_controller: SleepController,
    semantic_outcomes: SemanticOutcomeTracker,
    last_active_control: Option<ActiveControlSummary>,
    cognition: RuntimeCognition,
}

const COGNITION_DEADLINE_MS: u64 = 2_000;
/// Leave a quiet period after every terminal provider outcome so a fast or
/// disabled provider cannot turn the organism tick into a request generator.
const COGNITION_COOLDOWN_MS: u64 = 2_000;

struct RuntimeCognition {
    pending: Option<PendingLlmCognition>,
    next_request_at_ms: u64,
    last_sense: pete_now::LlmSense,
    last_sense_valid_until_ms: u64,
    last_outcome: Option<CognitionOutcome>,
    provider_declared_available: bool,
    provider_unavailable_reason: Option<String>,
}

impl RuntimeCognition {
    fn from_agent(agent: &impl LlmAgent) -> Self {
        Self {
            pending: None,
            next_request_at_ms: 0,
            last_sense: pete_now::LlmSense::default(),
            last_sense_valid_until_ms: 0,
            last_outcome: None,
            provider_declared_available: agent.enhanced_cognition_available(),
            provider_unavailable_reason: agent
                .enhanced_cognition_unavailable_reason()
                .map(str::to_string),
        }
    }
}

struct PendingLlmCognition {
    snapshot_ref: String,
    requested_at_ms: u64,
    deadline_ms: u64,
    task: JoinHandle<Result<(Option<Combobulation>, LlmTickResult)>>,
}

#[derive(Clone, Debug)]
enum CognitionOutcome {
    Accepted,
    Expired,
    Failed(String),
    Cancelled,
}

struct AcceptedLlmCognition {
    reflection: Option<Combobulation>,
    tick: LlmTickResult,
    snapshot_ref: String,
    requested_at_ms: u64,
    observed_at_ms: u64,
}

#[derive(Clone, Debug, Default)]
struct SemanticOutcomeTracker {
    previous: Option<SemanticActionState>,
    pending: Vec<SemanticEvidenceObservation>,
}

#[derive(Clone, Debug)]
struct SemanticActionState {
    t_ms: u64,
    behavior_id: String,
    target_id: Option<EntityId>,
    charger_distance_m: Option<f32>,
    clearance_m: Option<f32>,
    charging: bool,
}

impl SemanticOutcomeTracker {
    fn take_pending(&mut self) -> Vec<SemanticEvidenceObservation> {
        std::mem::take(&mut self.pending)
    }

    fn observe_outcome(&mut self, world: &WorldModelSnapshot) {
        let Some(previous) = self.previous.as_ref() else {
            return;
        };
        let mut observations = Vec::new();
        let current_charger_distance = previous
            .target_id
            .as_ref()
            .and_then(|target_id| canonical_entity_distance(world, target_id));
        let progress_evidence = |key: &str| EvidenceRef {
            id: format!(
                "semantic:action-outcome:{key}:{}:{}",
                previous.t_ms, world.t_ms
            ),
            source: "runtime.action_outcome".to_string(),
            key: key.to_string(),
            observed_at_ms: world.t_ms,
            transformation_lineage: vec!["pete_runtime::SemanticOutcomeTracker".to_string()],
            implementation_version: Some("2".to_string()),
        };
        if previous.behavior_id == "approach_charger"
            && previous
                .charger_distance_m
                .zip(current_charger_distance)
                .is_some_and(|(before, after)| after + 0.02 < before)
        {
            observations.push(SemanticEvidenceObservation::supported(
                SemanticNodeRef::Behavior(SemanticBehaviorId("approach_charger".to_string())),
                SemanticPredicate::Predicts,
                SemanticNodeRef::Outcome(SemanticOutcomeId(
                    "target_distance_decreases".to_string(),
                )),
                0.85,
                SemanticGroundingKind::ActionOutcome,
                progress_evidence("approach_reduced_charger_distance"),
            ));
        }
        if previous.behavior_id == "dock" && !previous.charging && world.self_model.charging {
            observations.push(SemanticEvidenceObservation::supported(
                SemanticNodeRef::Behavior(SemanticBehaviorId("dock".to_string())),
                SemanticPredicate::Predicts,
                SemanticNodeRef::Outcome(SemanticOutcomeId("charging_started".to_string())),
                0.95,
                SemanticGroundingKind::ActionOutcome,
                progress_evidence("dock_started_charging"),
            ));
        }
        if previous.behavior_id == "back_away"
            && previous
                .clearance_m
                .zip(
                    world
                        .local_geometry
                        .nearest_m
                        .as_ref()
                        .filter(|belief| belief.meta.freshness == Freshness::Current)
                        .map(|belief| belief.value),
                )
                .is_some_and(|(before, after)| after > before + 0.02)
        {
            observations.push(SemanticEvidenceObservation::supported(
                SemanticNodeRef::Behavior(SemanticBehaviorId("back_away".to_string())),
                SemanticPredicate::Predicts,
                SemanticNodeRef::Outcome(SemanticOutcomeId("clearance_increases".to_string())),
                0.85,
                SemanticGroundingKind::ActionOutcome,
                progress_evidence("back_away_increased_clearance"),
            ));
        }
        self.pending.extend(observations);
    }

    fn remember(
        &mut self,
        world: &WorldModelSnapshot,
        behavior: Option<&pete_conductor::BehaviorDecision>,
    ) {
        self.previous = behavior.map(|behavior| {
            let target_id = behavior.affordance.target.clone();
            SemanticActionState {
                t_ms: world.t_ms,
                behavior_id: behavior.behavior_id.clone(),
                charger_distance_m: target_id
                    .as_ref()
                    .and_then(|target_id| canonical_entity_distance(world, target_id)),
                target_id,
                clearance_m: world
                    .local_geometry
                    .nearest_m
                    .as_ref()
                    .filter(|belief| belief.meta.freshness == Freshness::Current)
                    .map(|belief| belief.value),
                charging: world.self_model.charging,
            }
        });
    }
}

fn canonical_entity_distance(world: &WorldModelSnapshot, target_id: &EntityId) -> Option<f32> {
    world
        .entities
        .get(target_id)
        .filter(|entity| entity.kind == pete_now::WorldEntityKind::Charger)
        .filter(|entity| {
            entity.distance_meta.as_ref().is_some_and(|meta| {
                !matches!(
                    meta.freshness,
                    Freshness::Stale | Freshness::Invalidated | Freshness::Missing
                )
            })
        })
        .and_then(|entity| entity.distance_m)
}

fn runtime_sleep_input(
    now: &Now,
    expected_external_power: bool,
    accelerator_available: bool,
) -> SleepTickInput {
    let flags = &now.body.flags;
    let safety_event = if flags.wheel_drop {
        Some("wheel_drop".to_string())
    } else if flags.cliff_left
        || flags.cliff_front_left
        || flags.cliff_front_right
        || flags.cliff_right
    {
        Some("cliff".to_string())
    } else if flags.bump_left || flags.bump_right {
        Some("contact".to_string())
    } else {
        None
    };
    let extension_bool = |key: &str| {
        now.extensions
            .get(key)
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    };
    let fatigue_activation = now
        .world
        .self_model
        .motivation
        .drives
        .get("rest")
        .map(|drive| drive.activation)
        .unwrap_or(now.drives.fatigue)
        .max(now.drives.fatigue);
    let direct_reign_active = now.reign.active
        && (now.reign.mode == Some(ReignMode::Direct)
            || now
                .reign
                .latest
                .as_ref()
                .is_some_and(|input| input.mode == ReignMode::Direct));
    let stopped =
        now.body.velocity.forward_m_s.abs() <= 0.01 && now.body.velocity.turn_rad_s.abs() <= 0.01;
    let body_communication_stable =
        now.body.last_update_ms == 0 || now.t_ms.saturating_sub(now.body.last_update_ms) <= 2_000;
    let critical_battery = now.body.battery_level <= 0.08;
    let unresolved_urgent_need = safety_event.is_some()
        || (now.body.battery_level <= 0.15 && !now.body.charging)
        || now.drives.danger_avoidance >= 0.80;
    let completed_episode_refs = now
        .world
        .temporal
        .recently_completed
        .iter()
        .map(|episode| episode.episode_id.0.clone())
        .collect::<Vec<_>>();
    let failed_behavior_refs = now
        .world
        .self_model
        .goal_status
        .iter()
        .filter(|(_, status)| status.failed_attempts > 0)
        .map(|(goal_id, status)| format!("goal:{goal_id}:failures:{}", status.failed_attempts))
        .collect::<Vec<_>>();
    let semantic_relation_refs = now
        .world
        .semantic
        .relations
        .keys()
        .take(128)
        .map(|relation_id| relation_id.0.clone())
        .collect::<Vec<_>>();
    SleepTickInput {
        now_ms: now.t_ms,
        fatigue_activation,
        charging: now.body.charging,
        docked: now.body.charging,
        stopped,
        direct_reign_active,
        unresolved_urgent_need,
        body_communication_stable,
        active_skill_interruptible: true,
        critical_battery,
        external_power_lost: expected_external_power && !now.body.charging,
        safety_event,
        important_social_cue: extension_bool("sleep.important_social_cue"),
        operator_sleep_request: extension_bool("sleep.request"),
        operator_wake_request: extension_bool("wake.request"),
        accelerator_available,
        thermal_fraction: now
            .extensions
            .get("body.thermal_fraction")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32,
        completed_episode_refs,
        failed_behavior_refs,
        semantic_relation_refs,
    }
}

#[derive(Clone, Debug, Default)]
struct ChirpEventState {
    last_charging: Option<bool>,
    last_awake: Option<bool>,
    last_object_count: usize,
    last_object_familiarity: f32,
    last_place_familiarity: f32,
    last_places_visited: u32,
    last_similar_situation_count: u16,
    last_surprise_high: bool,
    last_chosen_docking: bool,
}

impl ChirpEventState {
    fn emit_pre_selection_chirps(&mut self, now: &mut Now, notes: &mut Vec<String>) -> Result<()> {
        let object_count = now.objects.observations.len() + now.objects.vectors.len();
        if object_count > 0 && self.last_object_count == 0 {
            append_event_script_chirp(now, notes, "saw-something", ChirpPattern::SawSomething)?;
        }
        if now.memory.object_familiarity >= 0.70 && self.last_object_familiarity < 0.70 {
            append_event_script_chirp(
                now,
                notes,
                "object-recognized",
                ChirpPattern::ObjectRecognized,
            )?;
        }
        if now.memory.place_familiarity >= 0.70 && self.last_place_familiarity < 0.70 {
            append_event_script_chirp(
                now,
                notes,
                "place-recognized",
                ChirpPattern::PlaceRecognized,
            )?;
        }
        let learned = (self.last_places_visited > 0
            && now.memory.places_visited > self.last_places_visited)
            || (self.last_similar_situation_count > 0
                && now.memory.similar_situation_count > self.last_similar_situation_count);
        if learned {
            append_event_script_chirp(now, notes, "learned", ChirpPattern::Learned)?;
        }
        let surprise_high = now.surprise.total >= 0.70 || now.surprise.prediction_error >= 0.70;
        if surprise_high && !self.last_surprise_high {
            append_event_script_chirp(now, notes, "surprise", ChirpPattern::Surprise)?;
        }
        if matches!(self.last_charging, Some(false)) && now.body.charging {
            append_event_script_chirp(
                now,
                notes,
                "charging-started",
                ChirpPattern::ChargingStarted,
            )?;
        }
        let awake = now.drives.fatigue < 0.80;
        if matches!(self.last_awake, Some(true)) && !awake {
            append_event_script_chirp(now, notes, "sleep", ChirpPattern::Sleep)?;
        } else if matches!(self.last_awake, Some(false)) && awake {
            append_event_script_chirp(now, notes, "wake", ChirpPattern::Wake)?;
        }

        self.last_charging = Some(now.body.charging);
        self.last_awake = Some(awake);
        self.last_object_count = object_count;
        self.last_object_familiarity = now.memory.object_familiarity;
        self.last_place_familiarity = now.memory.place_familiarity;
        self.last_places_visited = now.memory.places_visited;
        self.last_similar_situation_count = now.memory.similar_situation_count;
        self.last_surprise_high = surprise_high;
        Ok(())
    }

    fn emit_post_selection_chirps(
        &mut self,
        now: &mut Now,
        notes: &mut Vec<String>,
        chosen_action: &ActionPrimitive,
    ) -> Result<()> {
        let docking = matches!(chosen_action, ActionPrimitive::Dock);
        if docking && !self.last_chosen_docking {
            let pattern = if charger_visible(now) {
                ChirpPattern::GoalAcquired
            } else {
                ChirpPattern::Docking
            };
            append_event_script_chirp(now, notes, "docking", pattern)?;
        }
        self.last_chosen_docking = docking;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ActionSelectorMode {
    #[default]
    Baseline,
    Random,
    ModelAssisted,
    Scripted,
    GoalShadow,
    Goal,
}

impl ActionSelectorMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Random => "random",
            Self::ModelAssisted => "model-assisted",
            Self::Scripted => "scripted",
            Self::GoalShadow => "goal-shadow",
            Self::Goal => "goal",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InlineLearningMode {
    #[default]
    Off,
    ShadowOnly,
    WorldOutcome,
}

impl InlineLearningMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::ShadowOnly => "shadow-only",
            Self::WorldOutcome => "world-outcome",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InlineLearningBehaviors {
    pub danger: bool,
    pub charge: bool,
    pub future: bool,
    pub action_value: bool,
    pub eye_next: bool,
    pub ear_next: bool,
    pub experience: bool,
}

impl Default for InlineLearningBehaviors {
    fn default() -> Self {
        Self {
            danger: true,
            charge: true,
            future: true,
            action_value: true,
            eye_next: true,
            ear_next: true,
            experience: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InlineLearningConfig {
    pub mode: InlineLearningMode,
    pub behaviors: InlineLearningBehaviors,
    pub max_train_steps_per_tick: usize,
}

impl Default for InlineLearningConfig {
    fn default() -> Self {
        Self {
            mode: InlineLearningMode::Off,
            behaviors: InlineLearningBehaviors::default(),
            max_train_steps_per_tick: 1,
        }
    }
}

impl InlineLearningConfig {
    pub fn disabled() -> Self {
        Self::default()
    }

    pub fn is_enabled(&self) -> bool {
        self.mode != InlineLearningMode::Off && self.max_train_steps_per_tick > 0
    }

    pub fn training_mode_label(&self) -> &'static str {
        match self.mode {
            InlineLearningMode::Off => "collecting",
            InlineLearningMode::ShadowOnly => "inline-shadow",
            InlineLearningMode::WorldOutcome => "inline-world-outcome",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InlineLearningTickStatus {
    pub enabled: bool,
    pub mode: InlineLearningMode,
    pub samples_observed: usize,
    pub train_steps_used: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NudgePolicy {
    pub enabled: bool,
    pub idle_after_ms: u64,
    pub max_nudges_per_minute: u32,
    pub max_forward_intensity: f32,
    pub max_turn_intensity: f32,
    pub require_clearance_m: f32,
    pub prefer_turn_when_clearance_low: bool,
    pub cooldown_ms: u64,
}

impl Default for NudgePolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            idle_after_ms: 4_000,
            max_nudges_per_minute: 6,
            max_forward_intensity: 0.15,
            max_turn_intensity: 0.25,
            require_clearance_m: 0.35,
            prefer_turn_when_clearance_low: true,
            cooldown_ms: 5_000,
        }
    }
}

impl NudgePolicy {
    pub fn virtual_default() -> Self {
        let mut policy = Self::default();
        policy.enabled = true;
        policy.idle_after_ms = 1_200;
        policy.cooldown_ms = 2_500;
        policy
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NudgeStatus {
    pub idle_ms: u64,
    pub last_nudge_ms: Option<u64>,
    pub nudge_count_recent: u32,
    pub nudge_blocked_reason: Option<String>,
    pub active_nudge: bool,
}

#[derive(Clone, Debug)]
struct NudgeController {
    status: NudgeStatus,
    last_pose: Option<pete_core::Pose2>,
    idle_started_at_ms: Option<u64>,
    recent_nudges: VecDeque<u64>,
    last_motor: MotorCommand,
}

impl Default for NudgeController {
    fn default() -> Self {
        Self {
            status: NudgeStatus::default(),
            last_pose: None,
            idle_started_at_ms: None,
            recent_nudges: VecDeque::new(),
            last_motor: MotorCommand::stop(),
        }
    }
}

impl NudgeController {
    fn propose(&mut self, now: &Now, policy: NudgePolicy) -> Option<ActionPrimitive> {
        self.prune_recent(now.t_ms);
        self.status.nudge_count_recent = self.recent_nudges.len() as u32;
        self.status.active_nudge = self
            .status
            .last_nudge_ms
            .map(|last| now.t_ms.saturating_sub(last) < 1_500)
            .unwrap_or(false);

        let low_motion =
            self.last_motor.forward.abs() < 0.02 && now.body.velocity.forward_m_s.abs() < 0.02;
        let low_pose_delta = self
            .last_pose
            .map(|pose| pose_delta_small(pose, now.body.odometry))
            .unwrap_or(true);
        if !low_motion || !low_pose_delta {
            self.idle_started_at_ms = Some(now.t_ms);
            self.status.idle_ms = 0;
            self.status.nudge_blocked_reason = None;
            self.last_pose = Some(now.body.odometry);
            return None;
        }

        let idle_started_at = *self.idle_started_at_ms.get_or_insert(now.t_ms);
        self.status.idle_ms = now.t_ms.saturating_sub(idle_started_at);
        self.last_pose = Some(now.body.odometry);

        if !policy.enabled {
            self.status.nudge_blocked_reason = Some("prod mode disabled".to_string());
            return None;
        }
        if let Some(last) = self.status.last_nudge_ms {
            if now.t_ms.saturating_sub(last) < policy.cooldown_ms {
                self.status.nudge_blocked_reason = Some("prod cooldown active".to_string());
                return None;
            }
        }
        if self.status.idle_ms < policy.idle_after_ms {
            self.status.nudge_blocked_reason = Some("not idle long enough".to_string());
            return None;
        }
        if self.recent_nudges.len() as u32 >= policy.max_nudges_per_minute {
            self.status.nudge_blocked_reason = Some("prod rate limit active".to_string());
            return None;
        }
        if let Some(reason) = nudge_general_block_reason(now) {
            self.status.nudge_blocked_reason = Some(reason);
            return None;
        }

        let action = choose_nudge_action(now, policy, self.recent_nudges.len());
        if let Some(reason) = nudge_action_block_reason(now, &action, policy) {
            self.status.nudge_blocked_reason = Some(reason);
            return None;
        }
        self.record_nudge(now.t_ms);
        self.status.nudge_blocked_reason = None;
        Some(action)
    }

    fn observe_motor(&mut self, motor: MotorCommand) {
        self.last_motor = motor;
    }

    fn record_nudge(&mut self, t_ms: u64) {
        self.status.last_nudge_ms = Some(t_ms);
        self.status.active_nudge = true;
        self.recent_nudges.push_back(t_ms);
        self.prune_recent(t_ms);
        self.status.nudge_count_recent = self.recent_nudges.len() as u32;
        self.idle_started_at_ms = Some(t_ms);
        self.status.idle_ms = 0;
    }

    fn prune_recent(&mut self, t_ms: u64) {
        while self
            .recent_nudges
            .front()
            .map(|stamp| t_ms.saturating_sub(*stamp) > 60_000)
            .unwrap_or(false)
        {
            self.recent_nudges.pop_front();
        }
    }
}

pub fn nudge_action_block_reason(
    now: &Now,
    action: &ActionPrimitive,
    policy: NudgePolicy,
) -> Option<String> {
    if let Some(reason) = nudge_general_block_reason(now) {
        return Some(reason);
    }
    let motor = action_to_motor_command(Some(action));
    if motor.forward > 0.0 && !forward_clear(now, policy.require_clearance_m) {
        return Some(format!(
            "forward path clearance is below {:.2} m",
            policy.require_clearance_m
        ));
    }
    None
}

pub fn nudge_action_block_reason_for_snapshot(
    snapshot: &WorldSnapshot,
    action: &ActionPrimitive,
    policy: NudgePolicy,
) -> Option<String> {
    nudge_action_block_reason(
        &snapshot.to_now(snapshot.body.last_update_ms),
        action,
        policy,
    )
}

fn choose_nudge_action(now: &Now, policy: NudgePolicy, recent_count: usize) -> ActionPrimitive {
    let turn_intensity = 0.20_f32.min(policy.max_turn_intensity);
    if !forward_clear(now, policy.require_clearance_m) && policy.prefer_turn_when_clearance_low {
        return ActionPrimitive::Turn {
            direction: clearer_turn_direction(now),
            intensity: turn_intensity,
            duration_ms: 600,
        };
    }

    match recent_count % 3 {
        0 => ActionPrimitive::Turn {
            direction: clearer_turn_direction(now),
            intensity: turn_intensity,
            duration_ms: 600,
        },
        1 => ActionPrimitive::Explore {
            style: ExploreStyle::RandomWalk,
            duration_ms: 1_000,
        },
        _ => ActionPrimitive::Go {
            intensity: 0.12_f32.min(policy.max_forward_intensity),
            duration_ms: 500,
        },
    }
}

fn nudge_general_block_reason(now: &Now) -> Option<String> {
    if now.body.flags.wheel_drop {
        return Some("wheel drop detected".to_string());
    }
    if now.body.battery_level <= SafetyConfigForNudge::CRITICAL_BATTERY {
        return Some("battery is critical".to_string());
    }
    if now
        .extensions
        .get("safety.vetoed")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
    {
        return Some("active safety override".to_string());
    }
    if sim_stuck_active(now) {
        return Some("stuck recovery active".to_string());
    }
    None
}

struct SafetyConfigForNudge;

impl SafetyConfigForNudge {
    const CRITICAL_BATTERY: f32 = 0.10;
}

fn is_near_zero_motor(motor: MotorCommand) -> bool {
    motor.forward.abs() < 0.02 && motor.turn.abs() < 0.04
}

fn pose_delta_small(left: pete_core::Pose2, right: pete_core::Pose2) -> bool {
    let dx = left.x_m - right.x_m;
    let dy = left.y_m - right.y_m;
    let distance = (dx * dx + dy * dy).sqrt();
    distance < 0.025
}

fn forward_clear(now: &Now, clearance_m: f32) -> bool {
    now.range
        .nearest_m
        .map(|nearest| nearest >= clearance_m)
        .unwrap_or(true)
}

fn clearer_turn_direction(now: &Now) -> TurnDir {
    let (left, _center, right) = beam_clearance_buckets(&now.range.beams);
    if right > left {
        TurnDir::Right
    } else {
        TurnDir::Left
    }
}

fn sim_stuck_active(now: &Now) -> bool {
    now.extensions
        .get("sim.stuck")
        .and_then(|value| value.get("values"))
        .and_then(|value| value.as_array())
        .and_then(|values| values.first())
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0)
        > 0.0
}

fn sim_world_extension_score(now: &Now, index: usize) -> f32 {
    now.extensions
        .get("sim.world")
        .and_then(|value| value.get("values"))
        .and_then(|value| value.as_array())
        .and_then(|values| values.get(index))
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0) as f32
}

fn apply_create_ir_charger_cue(now: &mut Now) {
    let Some(cue) = DockIrCue::from_character(now.body.infrared_character) else {
        return;
    };
    now.objects.observations.push(ObjectObservation {
        label: "home base IR".to_string(),
        class: ObjectClass::Charger,
        bearing_rad: cue.bearing_hint_rad(),
        // The beacon proves direction and identity, not metric range. In
        // particular, force-field reception must never be mistaken for dock
        // contact or charging.
        distance_m: None,
        confidence: cue.visible_score(),
        source: ObjectObservationSource::CreateIr,
    });
}

fn charger_signal_scores(now: &Now) -> (f32, f32) {
    let mut near = sim_world_extension_score(now, 3);
    let mut visible = sim_world_extension_score(now, 4);
    if let Some(cue) = DockIrCue::from_character(now.body.infrared_character) {
        near = near.max(cue.near_score());
        visible = visible.max(cue.visible_score());
    }
    (near, visible)
}

fn apply_recent_trap_memory_hints(now: &mut Now) {
    let Some(values) = now
        .extensions
        .get("sim.stuck")
        .and_then(|value| value.get("values"))
        .and_then(|value| value.as_array())
    else {
        return;
    };
    let active = values
        .first()
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0)
        > 0.0;
    let event_started = values
        .get(6)
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0)
        > 0.0;
    let repeated = values
        .get(12)
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0)
        .max(0.0) as f32;
    let trap_kind = values
        .get(10)
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0);
    if !(active || event_started || repeated > 0.0 || trap_kind > 0.0) {
        return;
    }
    let turn_sign = values
        .get(5)
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0) as f32;
    now.memory.recent_trap_confidence = (0.6 + repeated.min(2.0) * 0.15).clamp(0.0, 1.0);
    now.memory.recent_trap_direction_rad = Some(if turn_sign < 0.0 {
        -std::f32::consts::FRAC_PI_2
    } else if turn_sign > 0.0 {
        std::f32::consts::FRAC_PI_2
    } else {
        0.0
    });
}

fn place_candidate_to_loop_input(
    candidate: &PlaceRecognitionCandidate,
    source_frame_id: Option<String>,
    query_input: Option<&pete_memory::PlaceRecognitionInput>,
) -> LoopClosureCandidateInput {
    LoopClosureCandidateInput {
        target_pose: Pose2 {
            x_m: candidate.cell.center_x_m,
            y_m: candidate.cell.center_y_m,
            heading_rad: query_input
                .and_then(|input| input.pose)
                .map(|pose| pose.heading_rad)
                .unwrap_or(0.0),
        },
        confidence: candidate.confidence,
        similarity: candidate.similarity,
        kind: match candidate.kind {
            PlaceRecognitionKind::SamePlace => "same_place",
            PlaceRecognitionKind::SimilarPlace => "similar_place",
            PlaceRecognitionKind::EntityConstellation => "entity_constellation",
        }
        .to_string(),
        target_frame_id: candidate
            .source_instant_frame_id
            .clone()
            .or_else(|| candidate.source_frame_id.clone()),
        source_frame_id,
        source_experience_id: candidate.source_experience_id.clone(),
        source_instant_frame_id: candidate.source_instant_frame_id.clone(),
        source_vector_refs: candidate.source_vector_refs.clone(),
        source_vector_id: Some(candidate.source_vector_id.clone()),
        query_vector_id: candidate.query_vector_id.clone(),
        query_experience_id: candidate
            .query_experience_id
            .clone()
            .or_else(|| query_input.and_then(|input| input.experience_id.clone())),
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ActionSelectionDecision {
    pub mode: ActionSelectorMode,
    pub candidates: Vec<ActionSelectionCandidateScore>,
    pub selected_action: Option<ActionPrimitive>,
    pub baseline_action: Option<ActionPrimitive>,
    pub selected_score: Option<f32>,
    pub safety_overrode: bool,
    pub fallback_warnings: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_goal: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_behavior: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_selected_goal: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_selected_behavior: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_goal_action: Option<ActionPrimitive>,
    #[serde(default)]
    pub shadow_diverged_from_baseline: bool,
    #[serde(default)]
    pub goal_switched: bool,
    #[serde(default)]
    pub goal_retained_by_commitment: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal_selection_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionSelectionCandidateScore {
    pub action: ActionPrimitive,
    pub score: f32,
    pub danger: f32,
    pub charge: f32,
    pub action_value: f32,
    pub curiosity: f32,
    pub collision_risk: f32,
    pub low_battery_risk: f32,
    pub repeat_penalty: f32,
    pub fallback_used: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MapMemoryDecisionDebug {
    pub influenced: bool,
    #[serde(default)]
    pub corrected_map_trusted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corrected_map_untrusted_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub navigation_intent: Option<NavigationIntent>,
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_string: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal_value: Option<f32>,
    #[serde(default)]
    pub signal_confidence: f32,
    #[serde(default)]
    pub confidence: f32,
    pub place_danger: f32,
    pub place_charge_value: f32,
    pub place_novelty: f32,
    pub safe_direction_rad: Option<f32>,
    pub charge_direction_rad: Option<f32>,
    pub frontier_direction_rad: Option<f32>,
    pub recent_trap_direction_rad: Option<f32>,
    pub map_confidence: f32,
    pub recent_trap_confidence: f32,
    pub selected_action: Option<ActionPrimitive>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_action: Option<ActionPrimitive>,
    #[serde(default)]
    pub safety_overrode: bool,
}

impl Default for ActionSelectionCandidateScore {
    fn default() -> Self {
        Self {
            action: ActionPrimitive::Stop,
            score: 0.0,
            danger: 0.0,
            charge: 0.0,
            action_value: 0.0,
            curiosity: 0.0,
            collision_risk: 0.0,
            low_battery_risk: 0.0,
            repeat_penalty: 0.0,
            fallback_used: false,
        }
    }
}

#[derive(Default)]
pub struct RuntimeModelStack {
    pub behaviors: BehaviorRegistry,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BumpEventInput {
    pub t_ms: TimeMs,
    pub bump_left: bool,
    pub bump_right: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RobotInitializedEventInput {
    pub t_ms: TimeMs,
    pub mode: String,
    pub body: String,
    pub battery_percent: Option<u32>,
    pub charging: Option<bool>,
    pub active_sensors: usize,
    pub requested_sensors: usize,
    pub ledger: String,
    pub tick_ms: u64,
    pub dashboard: Option<String>,
    pub capture: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventScriptAction {
    Say { text: String },
    Chirp { pattern: ChirpPattern },
    Song { name: String },
    Stop,
    Rotate { deg: i16 },
    Go,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EventScriptOutput {
    pub actions: Vec<EventScriptAction>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SafeScriptAction {
    pub requested: EventScriptAction,
    pub action: Option<ActionPrimitive>,
    pub desired_motor: MotorCommand,
    pub final_motor: MotorCommand,
    pub vetoed: bool,
    pub safety_reason: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SafeScriptSequence {
    pub actions: Vec<SafeScriptAction>,
}

impl RuntimeModelStack {
    pub fn with_danger_shadow_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_danger_metadata(path)?;
        let mut stack = Self::default();
        stack.behaviors.danger = danger_behavior(
            BehaviorRegime::ShadowInfer,
            Some(DangerNetTrainer::load_checkpoint(path, metadata.input_dim)?),
            FallbackPolicy::UseHardcoded,
        );
        Ok(stack)
    }

    pub fn with_charge_shadow_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_charge_metadata(path)?;
        let mut stack = Self::default();
        stack.behaviors.charge = charge_behavior(
            BehaviorRegime::ShadowInfer,
            Some(ChargeNetTrainer::load_checkpoint(path, metadata.input_dim)?),
            FallbackPolicy::UseHardcoded,
        );
        Ok(stack)
    }

    pub fn with_action_value_shadow_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_action_value_metadata(path)?;
        let mut stack = Self::default();
        stack.behaviors.action_value = action_value_behavior(
            BehaviorRegime::ShadowInfer,
            Some(ActionValueNetTrainer::load_checkpoint(
                path,
                metadata.input_dim,
            )?),
            FallbackPolicy::UseHardcoded,
        );
        Ok(stack)
    }

    pub fn with_eye_next_shadow_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_eye_next_metadata(path)?;
        let mut stack = Self::default();
        stack.behaviors.eye_next = eye_next_behavior(
            BehaviorRegime::ShadowInfer,
            Some(EyeNextNetTrainer::load_checkpoint(
                path,
                metadata.input_dim,
            )?),
            FallbackPolicy::UseHardcoded,
        );
        Ok(stack)
    }

    pub fn with_ear_next_shadow_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_ear_next_metadata(path)?;
        let mut stack = Self::default();
        stack.behaviors.ear_next = ear_next_behavior(
            BehaviorRegime::ShadowInfer,
            Some(EarNextNetTrainer::load_checkpoint(
                path,
                metadata.input_dim,
            )?),
            FallbackPolicy::UseHardcoded,
        );
        Ok(stack)
    }

    pub fn with_experience_shadow_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_experience_autoencoder_metadata(path)?;
        let mut stack = Self::default();
        stack.behaviors.experience = experience_behavior(
            BehaviorRegime::ShadowInfer,
            Some(ExperienceAutoencoderTrainer::load_checkpoint(
                path,
                metadata.input_dim,
            )?),
            FallbackPolicy::UseHardcoded,
        );
        Ok(stack)
    }

    pub fn with_future_shadow_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        Self::with_future_checkpoint(path, BehaviorRegime::ShadowInfer)
    }

    pub fn with_future_checkpoint(path: impl AsRef<Path>, mode: BehaviorRegime) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_future_metadata(path)?;
        let mut stack = Self::default();
        stack.behaviors.future = future_behavior(
            mode,
            Some(FutureNetTrainer::load_checkpoint(
                path,
                metadata.input_dim,
                metadata.latent_dim,
            )?),
            FallbackPolicy::UseHardcoded,
        );
        Ok(stack)
    }

    pub fn with_shadow_checkpoints(
        danger_path: Option<&Path>,
        charge_path: Option<&Path>,
        action_value_path: Option<&Path>,
        future_path: Option<&Path>,
        eye_next_path: Option<&Path>,
        ear_next_path: Option<&Path>,
        experience_path: Option<&Path>,
    ) -> Result<Self> {
        let mut stack = Self::default();
        if let Some(path) = danger_path {
            let metadata = read_danger_metadata(path)?;
            stack.behaviors.danger = danger_behavior(
                BehaviorRegime::ShadowInfer,
                Some(DangerNetTrainer::load_checkpoint(path, metadata.input_dim)?),
                FallbackPolicy::UseHardcoded,
            );
        }
        if let Some(path) = charge_path {
            let metadata = read_charge_metadata(path)?;
            stack.behaviors.charge = charge_behavior(
                BehaviorRegime::ShadowInfer,
                Some(ChargeNetTrainer::load_checkpoint(path, metadata.input_dim)?),
                FallbackPolicy::UseHardcoded,
            );
        }
        if let Some(path) = action_value_path {
            let metadata = read_action_value_metadata(path)?;
            stack.behaviors.action_value = action_value_behavior(
                BehaviorRegime::ShadowInfer,
                Some(ActionValueNetTrainer::load_checkpoint(
                    path,
                    metadata.input_dim,
                )?),
                FallbackPolicy::UseHardcoded,
            );
        }
        if let Some(path) = future_path {
            let metadata = read_future_metadata(path)?;
            stack.behaviors.future = future_behavior(
                BehaviorRegime::ShadowInfer,
                Some(FutureNetTrainer::load_checkpoint(
                    path,
                    metadata.input_dim,
                    metadata.latent_dim,
                )?),
                FallbackPolicy::UseHardcoded,
            );
        }
        if let Some(path) = eye_next_path {
            let metadata = read_eye_next_metadata(path)?;
            stack.behaviors.eye_next = eye_next_behavior(
                BehaviorRegime::ShadowInfer,
                Some(EyeNextNetTrainer::load_checkpoint(
                    path,
                    metadata.input_dim,
                )?),
                FallbackPolicy::UseHardcoded,
            );
        }
        if let Some(path) = ear_next_path {
            let metadata = read_ear_next_metadata(path)?;
            stack.behaviors.ear_next = ear_next_behavior(
                BehaviorRegime::ShadowInfer,
                Some(EarNextNetTrainer::load_checkpoint(
                    path,
                    metadata.input_dim,
                )?),
                FallbackPolicy::UseHardcoded,
            );
        }
        if let Some(path) = experience_path {
            let metadata = read_experience_autoencoder_metadata(path)?;
            stack.behaviors.experience = experience_behavior(
                BehaviorRegime::ShadowInfer,
                Some(ExperienceAutoencoderTrainer::load_checkpoint(
                    path,
                    metadata.input_dim,
                )?),
                FallbackPolicy::UseHardcoded,
            );
        }
        Ok(stack)
    }

    pub fn from_models_config(path: impl AsRef<Path>) -> Result<Self> {
        let config: BehaviorRegistryConfig = toml::from_str(&std::fs::read_to_string(path)?)?;
        Self::from_behavior_config(&config)
    }

    pub fn from_behavior_config(config: &BehaviorRegistryConfig) -> Result<Self> {
        let mut stack = Self::default();
        if let Some(behavior) = config.behavior.get("locomotion") {
            stack.behaviors.locomotion = locomotion_behavior(
                behavior.regime,
                load_locomotion_behavior(behavior)?,
                behavior.fallback,
            );
        }
        if let Some(behavior) = config.behavior.get("danger") {
            stack.behaviors.danger = danger_behavior(
                behavior.regime,
                load_danger_behavior_trainer(behavior)?,
                behavior.fallback,
            );
        }
        if let Some(behavior) = config.behavior.get("charge") {
            stack.behaviors.charge = charge_behavior(
                behavior.regime,
                load_charge_behavior_trainer(behavior)?,
                behavior.fallback,
            );
        }
        if let Some(behavior) = config.behavior.get("future") {
            stack.behaviors.future = future_behavior(
                behavior.regime,
                load_future_behavior_trainer(behavior)?,
                behavior.fallback,
            );
        }
        if let Some(behavior) = config.behavior.get("action_value") {
            stack.behaviors.action_value = action_value_behavior(
                behavior.regime,
                load_action_value_behavior_trainer(behavior)?,
                behavior.fallback,
            );
        }
        if let Some(behavior) = config.behavior.get("conductor") {
            stack.behaviors.conductor = conductor_behavior(
                behavior.regime,
                &behavior.hardcoded,
                behavior.model.clone(),
                behavior.fallback,
            );
        }
        if let Some(behavior) = config.behavior.get("eye_next") {
            stack.behaviors.eye_next = eye_next_behavior(
                behavior.regime,
                load_eye_next_behavior_trainer(behavior)?,
                behavior.fallback,
            );
        }
        if let Some(behavior) = config.behavior.get("ear_next") {
            stack.behaviors.ear_next = ear_next_behavior(
                behavior.regime,
                load_ear_next_behavior_trainer(behavior)?,
                behavior.fallback,
            );
        }
        if let Some(behavior) = config.behavior.get("experience") {
            stack.behaviors.experience = experience_behavior(
                behavior.regime,
                load_experience_behavior_trainer(behavior)?,
                behavior.fallback,
            );
        }
        if let Some(behavior) = config.behavior.get("event_bump") {
            stack.behaviors.event_bump =
                bump_event_behavior(behavior.regime, behavior.model.clone(), behavior.fallback);
        }
        if let Some(behavior) = config.behavior.get("event_robot_initialized") {
            stack.behaviors.event_robot_initialized = robot_initialized_event_behavior(
                behavior.regime,
                behavior.model.clone(),
                behavior.fallback,
            );
        }
        Ok(stack)
    }

    pub fn behavior_node_states(
        &self,
        last_runs: &[ErasedBehaviorRunRecord],
    ) -> Vec<BehaviorNodeState> {
        let last = |id: &str| {
            last_runs
                .iter()
                .rev()
                .find(|run| run.behavior_id == id)
                .cloned()
        };
        vec![
            behavior_node_state(
                "Locomotion",
                "locomotion",
                "Locomotion",
                self.behaviors.locomotion.regime,
                self.behaviors.locomotion.hardcoded_id(),
                self.behaviors.locomotion.model_id(),
                self.behaviors.locomotion.fallback,
                vec![impl_id(
                    "locomotion.hardcoded_wander.v0",
                    "Hardcoded wander/reflex",
                )],
                vec![impl_id("locomotion.neat.v0", "NEAT locomotion v0")],
                last("locomotion"),
            ),
            behavior_node_state(
                "Experience",
                "experience",
                "Experience",
                self.behaviors.experience.regime,
                self.behaviors.experience.hardcoded_id(),
                self.behaviors.experience.model_id(),
                self.behaviors.experience.fallback,
                vec![impl_id("experience.no_latent_yet", "No latent yet")],
                vec![impl_id("experience.autoencoder.v0", "Autoencoder v0")],
                last("experience"),
            ),
            behavior_node_state(
                "Danger",
                "danger",
                "Danger",
                self.behaviors.danger.regime,
                self.behaviors.danger.hardcoded_id(),
                self.behaviors.danger.model_id(),
                self.behaviors.danger.fallback,
                vec![impl_id("danger.range_bumper", "Range/bumper")],
                vec![impl_id("danger.burn.v0", "Burn v0")],
                last("danger"),
            ),
            behavior_node_state(
                "Charge",
                "charge",
                "Charge",
                self.behaviors.charge.regime,
                self.behaviors.charge.hardcoded_id(),
                self.behaviors.charge.model_id(),
                self.behaviors.charge.fallback,
                vec![impl_id(
                    "charge.sensor_battery_delta",
                    "Sensor/battery delta",
                )],
                vec![impl_id("charge.burn.v0", "Burn v0")],
                last("charge"),
            ),
            behavior_node_state(
                "Future",
                "future",
                "Future",
                self.behaviors.future.regime,
                self.behaviors.future.hardcoded_id(),
                self.behaviors.future.model_id(),
                self.behaviors.future.fallback,
                vec![impl_id("future.stasis", "Stasis")],
                vec![impl_id("future.burn.v0", "Burn v0")],
                last("future"),
            ),
            behavior_node_state(
                "Conductor",
                "conductor",
                "Conductor",
                self.behaviors.conductor.regime,
                self.behaviors.conductor.hardcoded_id(),
                self.behaviors.conductor.model_id(),
                self.behaviors.conductor.fallback,
                vec![
                    impl_id("conductor.simple_v0", "Simple conductor"),
                    impl_id("action_selector.baseline", "Baseline selector"),
                    impl_id("reign.teacher", "Reign teacher"),
                ],
                vec![
                    impl_id("conductor.burn.v0", "Conductor Burn v0"),
                    impl_id("action_selector.burn.v0", "Action selector Burn v0"),
                ],
                last("conductor"),
            ),
            behavior_node_state(
                "ActionValue",
                "action_value",
                "ActionValue",
                self.behaviors.action_value.regime,
                self.behaviors.action_value.hardcoded_id(),
                self.behaviors.action_value.model_id(),
                self.behaviors.action_value.fallback,
                vec![impl_id("action_value.handcoded", "Handcoded value")],
                vec![impl_id("action_value.burn.v0", "Burn v0")],
                last("action_value"),
            ),
            behavior_node_state(
                "EyeNext",
                "eye_next",
                "EyeNext",
                self.behaviors.eye_next.regime,
                self.behaviors.eye_next.hardcoded_id(),
                self.behaviors.eye_next.model_id(),
                self.behaviors.eye_next.fallback,
                vec![impl_id("eye.copy_current", "Copy current")],
                vec![impl_id("eye.burn.next_v0", "Burn next v0")],
                last("eye_next"),
            ),
            behavior_node_state(
                "EarNext",
                "ear_next",
                "EarNext",
                self.behaviors.ear_next.regime,
                self.behaviors.ear_next.hardcoded_id(),
                self.behaviors.ear_next.model_id(),
                self.behaviors.ear_next.fallback,
                vec![impl_id("ear.copy_current", "Copy current")],
                vec![impl_id("ear.burn.next_v0", "Burn next v0")],
                last("ear_next"),
            ),
            behavior_node_state(
                "EventRobotInitialized",
                "event_robot_initialized",
                "on(robot-initialized)",
                self.behaviors.event_robot_initialized.regime,
                self.behaviors.event_robot_initialized.hardcoded_id(),
                self.behaviors.event_robot_initialized.model_id(),
                self.behaviors.event_robot_initialized.fallback,
                vec![impl_id(
                    "script.on_robot_initialized.ts.v0",
                    "TypeScript script teacher",
                )],
                vec![impl_id("event.robot_initialized.shadow.v0", "Shadow model")],
                last("event_robot_initialized"),
            ),
            behavior_node_state(
                "EventBump",
                "event_bump",
                "on(bump)",
                self.behaviors.event_bump.regime,
                self.behaviors.event_bump.hardcoded_id(),
                self.behaviors.event_bump.model_id(),
                self.behaviors.event_bump.fallback,
                vec![impl_id("script.on_bump.ts.v0", "TypeScript script teacher")],
                vec![impl_id("event.bump.shadow.v0", "Shadow model")],
                last("event_bump"),
            ),
        ]
    }

    pub fn apply_behavior_node_update(&mut self, node_id: &str, update: &BehaviorNodeUpdate) {
        let id = normalize_behavior_node_id(node_id);
        if id == "conductor" {
            let regime = effective_training_regime(
                update
                    .selected_regime
                    .unwrap_or(self.behaviors.conductor.regime),
                update.training_enabled,
            );
            let hardcoded = update
                .selected_hardcoded
                .as_deref()
                .unwrap_or_else(|| self.behaviors.conductor.hardcoded_id());
            let model = update
                .selected_model
                .clone()
                .or_else(|| self.behaviors.conductor.model_id().map(str::to_string));
            let fallback = update
                .fallback_policy
                .unwrap_or(self.behaviors.conductor.fallback);
            self.behaviors.conductor = conductor_behavior(regime, hardcoded, model, fallback);
            return;
        }
        macro_rules! update_behavior {
            ($field:ident) => {{
                if let Some(regime) = update.selected_regime {
                    self.behaviors.$field.regime =
                        effective_training_regime(regime, update.training_enabled);
                }
                if let Some(fallback) = update.fallback_policy {
                    self.behaviors.$field.fallback = fallback;
                }
            }};
        }
        match id.as_str() {
            "locomotion" => update_behavior!(locomotion),
            "experience" => update_behavior!(experience),
            "danger" => update_behavior!(danger),
            "charge" => update_behavior!(charge),
            "future" => update_behavior!(future),
            "action_value" => update_behavior!(action_value),
            "eye_next" => update_behavior!(eye_next),
            "ear_next" => update_behavior!(ear_next),
            "event_robot_initialized" => update_behavior!(event_robot_initialized),
            "event_bump" => update_behavior!(event_bump),
            _ => {}
        }
    }
}

fn impl_id(id: &str, label: &str) -> BehaviorImplementation {
    BehaviorImplementation {
        id: id.to_string(),
        label: label.to_string(),
    }
}

fn behavior_node_state(
    node_id: &str,
    behavior_id: &str,
    label: &str,
    regime: BehaviorRegime,
    hardcoded_id: &str,
    model_id: Option<&str>,
    fallback: FallbackPolicy,
    hardcoded_implementations: Vec<BehaviorImplementation>,
    model_implementations: Vec<BehaviorImplementation>,
    last_run: Option<ErasedBehaviorRunRecord>,
) -> BehaviorNodeState {
    let training_enabled = matches!(
        regime,
        BehaviorRegime::ShadowTrain | BehaviorRegime::ModelTrainAndInfer
    );
    BehaviorNodeState {
        node_id: node_id.to_string(),
        behavior_id: behavior_id.to_string(),
        label: label.to_string(),
        allowed_regimes: vec![
            BehaviorRegime::Hardcoded,
            BehaviorRegime::ShadowTrain,
            BehaviorRegime::ShadowInfer,
            BehaviorRegime::ModelInfer,
            BehaviorRegime::ModelTrainAndInfer,
            BehaviorRegime::Compare,
        ],
        hardcoded_implementations,
        model_implementations,
        selected_regime: regime,
        selected_hardcoded: hardcoded_id.to_string(),
        selected_model: model_id.map(str::to_string),
        checkpoint_path: None,
        fallback_policy: fallback,
        training_enabled,
        last_run,
        samples_observed: 0,
        train_steps_used: 0,
        missing_model_or_checkpoint: model_id.is_none()
            && !matches!(regime, BehaviorRegime::Hardcoded),
    }
}

fn normalize_behavior_node_id(node_id: &str) -> String {
    match node_id {
        "ActionValue" => "action_value".to_string(),
        "EyeNext" => "eye_next".to_string(),
        "EarNext" => "ear_next".to_string(),
        "EventRobotInitialized" => "event_robot_initialized".to_string(),
        "EventBump" => "event_bump".to_string(),
        other => other.to_ascii_lowercase().replace('-', "_"),
    }
}

fn effective_training_regime(
    regime: BehaviorRegime,
    training_enabled: Option<bool>,
) -> BehaviorRegime {
    if training_enabled.unwrap_or(true) {
        return regime;
    }
    match regime {
        BehaviorRegime::ShadowTrain => BehaviorRegime::ShadowInfer,
        BehaviorRegime::ModelTrainAndInfer => BehaviorRegime::ModelInfer,
        other => other,
    }
}

pub struct BehaviorRegistry {
    pub locomotion: ReplaceableBehavior<LocomotionInput, LocomotionOutput>,
    pub experience: ReplaceableBehavior<ExperienceBehaviorInput, ExperienceBehaviorOutput>,
    pub danger: ReplaceableBehavior<SituatedDangerInput, DangerOutput>,
    pub charge: ReplaceableBehavior<SituatedChargeInput, ChargeOutput>,
    pub future: ReplaceableBehavior<FutureInput, FuturePrediction>,
    pub action_value: ReplaceableBehavior<SituatedActionValueInput, ActionValueOutput>,
    pub conductor: ReplaceableBehavior<ConductorInput, ActionPrimitive>,
    pub eye_next: ReplaceableBehavior<SituatedEyeNextInput, EyeNextOutput>,
    pub ear_next: ReplaceableBehavior<SituatedEarNextInput, EarNextOutput>,
    pub event_robot_initialized: ReplaceableBehavior<RobotInitializedEventInput, EventScriptOutput>,
    pub event_bump: ReplaceableBehavior<BumpEventInput, EventScriptOutput>,
}

impl Default for BehaviorRegistry {
    fn default() -> Self {
        Self {
            locomotion: locomotion_behavior(
                BehaviorRegime::Hardcoded,
                None,
                FallbackPolicy::UseHardcoded,
            ),
            experience: experience_behavior(
                BehaviorRegime::Hardcoded,
                None,
                FallbackPolicy::UseHardcoded,
            ),
            danger: danger_behavior(
                BehaviorRegime::Hardcoded,
                None,
                FallbackPolicy::UseHardcoded,
            ),
            charge: charge_behavior(
                BehaviorRegime::Hardcoded,
                None,
                FallbackPolicy::UseHardcoded,
            ),
            future: future_behavior(
                BehaviorRegime::Hardcoded,
                None,
                FallbackPolicy::UseHardcoded,
            ),
            action_value: action_value_behavior(
                BehaviorRegime::Hardcoded,
                None,
                FallbackPolicy::UseHardcoded,
            ),
            conductor: conductor_behavior(
                BehaviorRegime::Hardcoded,
                "conductor.simple_v0",
                Some("conductor.burn.v0".to_string()),
                FallbackPolicy::StopSafely,
            ),
            eye_next: eye_next_behavior(
                BehaviorRegime::Hardcoded,
                None,
                FallbackPolicy::UseHardcoded,
            ),
            ear_next: ear_next_behavior(
                BehaviorRegime::Hardcoded,
                None,
                FallbackPolicy::UseHardcoded,
            ),
            event_robot_initialized: robot_initialized_event_behavior(
                BehaviorRegime::ShadowTrain,
                Some("event.robot_initialized.shadow.v0".to_string()),
                FallbackPolicy::UseHardcoded,
            ),
            event_bump: bump_event_behavior(
                BehaviorRegime::ShadowTrain,
                Some("event.bump.shadow.v0".to_string()),
                FallbackPolicy::UseHardcoded,
            ),
        }
    }
}

pub struct BehaviorTrainingHub {
    pub danger_extractor: Box<dyn TargetExtractor<ExperienceTransition, DangerInput, DangerOutput>>,
    pub charge_extractor: Box<dyn TargetExtractor<ExperienceTransition, ChargeInput, ChargeOutput>>,
    pub future_extractor:
        Box<dyn TargetExtractor<ExperienceTransition, FutureInput, FuturePrediction>>,
    pub action_value_extractor:
        Box<dyn TargetExtractor<ExperienceTransition, ActionValueInput, ActionValueOutput>>,
    pub eye_next_extractor:
        Box<dyn TargetExtractor<ExperienceTransition, EyeNextInput, EyeNextOutput>>,
    pub ear_next_extractor:
        Box<dyn TargetExtractor<ExperienceTransition, EarNextInput, EarNextOutput>>,
}

impl Default for BehaviorTrainingHub {
    fn default() -> Self {
        Self {
            danger_extractor: Box::new(DangerTargetExtractor),
            charge_extractor: Box::new(ChargeTargetExtractor),
            future_extractor: Box::new(FutureTargetExtractor { offset_ms: 1_000 }),
            action_value_extractor: Box::new(ActionValueTargetExtractor),
            eye_next_extractor: Box::new(EyeNextTargetExtractor { offset_ms: 100 }),
            ear_next_extractor: Box::new(EarNextTargetExtractor { offset_ms: 100 }),
        }
    }
}

pub struct DangerTargetExtractor;

impl TargetExtractor<ExperienceTransition, DangerInput, DangerOutput> for DangerTargetExtractor {
    fn extract(
        &self,
        transition: &ExperienceTransition,
    ) -> Result<Option<TrainingSample<DangerInput, DangerOutput>>> {
        let input = danger_input_from_transition_like(
            &transition.before_z,
            transition.action.as_ref(),
            &transition.before,
        );
        let target = danger_target_from_transition_like(
            &transition.before,
            transition.action.as_ref(),
            &transition.after,
        );
        Ok(Some(TrainingSample {
            input,
            expected: DangerOutput {
                bump_risk: target.bump,
                cliff_risk: target.cliff,
                wheel_drop_risk: target.wheel_drop,
                stuck_risk: target.stuck,
                confidence: 1.0,
            },
            actual: None,
            reward: Some(transition.reward.value),
            weight: 1.0,
            source: TrainingSource::WorldOutcome,
            t_ms: transition.created_at_ms,
        }))
    }
}

pub struct ChargeTargetExtractor;

impl TargetExtractor<ExperienceTransition, ChargeInput, ChargeOutput> for ChargeTargetExtractor {
    fn extract(
        &self,
        transition: &ExperienceTransition,
    ) -> Result<Option<TrainingSample<ChargeInput, ChargeOutput>>> {
        let input = charge_input_from_transition_like(
            &transition.before_z,
            transition.action.as_ref(),
            &transition.before,
        );
        let target = charge_target_from_transition_like(
            &transition.before,
            transition.action.as_ref(),
            &transition.after,
        );
        Ok(Some(TrainingSample {
            input,
            expected: ChargeOutput {
                charge_probability: target.charging_started,
                expected_battery_delta: target.battery_delta,
                dock_likelihood: target.charging_after,
                confidence: 1.0,
            },
            actual: None,
            reward: Some(transition.reward.value),
            weight: 1.0,
            source: TrainingSource::WorldOutcome,
            t_ms: transition.created_at_ms,
        }))
    }
}

pub struct FutureTargetExtractor {
    pub offset_ms: TimeMs,
}

impl TargetExtractor<ExperienceTransition, FutureInput, FuturePrediction>
    for FutureTargetExtractor
{
    fn extract(
        &self,
        transition: &ExperienceTransition,
    ) -> Result<Option<TrainingSample<FutureInput, FuturePrediction>>> {
        let action = match transition.action.clone() {
            Some(action) => action,
            None => return Ok(None),
        };
        Ok(Some(TrainingSample {
            input: FutureInput {
                latent: transition.before_z.clone(),
                action,
                offset_ms: self.offset_ms,
            },
            expected: FuturePrediction {
                offset_ms: self.offset_ms,
                predicted_z: transition.after_z.z.clone(),
                confidence: transition.after_z.confidence,
                summary: Some("Observed next latent state.".to_string()),
            },
            actual: None,
            reward: Some(transition.reward.value),
            weight: 1.0,
            source: TrainingSource::WorldOutcome,
            t_ms: transition.created_at_ms,
        }))
    }
}

pub struct ActionValueTargetExtractor;

impl TargetExtractor<ExperienceTransition, ActionValueInput, ActionValueOutput>
    for ActionValueTargetExtractor
{
    fn extract(
        &self,
        transition: &ExperienceTransition,
    ) -> Result<Option<TrainingSample<ActionValueInput, ActionValueOutput>>> {
        let target =
            action_value_target_from_reward_surprise(&transition.reward, &transition.surprise);
        Ok(Some(TrainingSample {
            input: action_value_input_from_transition_like(
                &transition.before_z,
                transition.action.as_ref(),
                &transition.before,
            ),
            expected: ActionValueOutput {
                value: target.value.clamp(-1.0, 1.0),
                confidence: 1.0,
            },
            actual: None,
            reward: Some(transition.reward.value),
            weight: 1.0,
            source: TrainingSource::WorldOutcome,
            t_ms: transition.created_at_ms,
        }))
    }
}

pub struct EyeNextTargetExtractor {
    pub offset_ms: TimeMs,
}

impl TargetExtractor<ExperienceTransition, EyeNextInput, EyeNextOutput> for EyeNextTargetExtractor {
    fn extract(
        &self,
        transition: &ExperienceTransition,
    ) -> Result<Option<TrainingSample<EyeNextInput, EyeNextOutput>>> {
        let Some(target) = eye_next_target_from_now(&transition.after) else {
            return Ok(None);
        };
        Ok(Some(TrainingSample {
            input: eye_next_input_from_transition_like(
                &transition.before_z,
                transition.action.as_ref(),
                &transition.before,
                self.offset_ms,
            ),
            expected: EyeNextOutput {
                width: target.width,
                height: target.height,
                rgb: target.rgb,
                confidence: 1.0,
            },
            actual: None,
            reward: Some(transition.reward.value),
            weight: 1.0,
            source: TrainingSource::WorldOutcome,
            t_ms: transition.created_at_ms,
        }))
    }
}

pub struct EarNextTargetExtractor {
    pub offset_ms: TimeMs,
}

impl TargetExtractor<ExperienceTransition, EarNextInput, EarNextOutput> for EarNextTargetExtractor {
    fn extract(
        &self,
        transition: &ExperienceTransition,
    ) -> Result<Option<TrainingSample<EarNextInput, EarNextOutput>>> {
        let Some(target) = ear_next_target_from_now(&transition.after) else {
            return Ok(None);
        };
        Ok(Some(TrainingSample {
            input: ear_next_input_from_transition_like(
                &transition.before_z,
                transition.action.as_ref(),
                &transition.before,
                self.offset_ms,
            ),
            expected: EarNextOutput {
                sample_rate_hz: target.sample_rate_hz,
                channels: target.channels,
                pcm: target.pcm,
                features: target.features,
                confidence: 1.0,
            },
            actual: None,
            reward: Some(transition.reward.value),
            weight: 1.0,
            source: TrainingSource::WorldOutcome,
            t_ms: transition.created_at_ms,
        }))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SituatedDangerInput {
    pub input: DangerInput,
    pub now: Now,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SituatedChargeInput {
    pub input: ChargeInput,
    pub now: Now,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SituatedActionValueInput {
    pub input: ActionValueInput,
    pub now: Now,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SituatedEyeNextInput {
    pub input: EyeNextInput,
    pub now: Now,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SituatedEarNextInput {
    pub input: EarNextInput,
    pub now: Now,
}

struct HardcodedExperienceBehavior;

impl FunctionBehavior<ExperienceBehaviorInput, ExperienceBehaviorOutput>
    for HardcodedExperienceBehavior
{
    fn id(&self) -> &'static str {
        "experience.no_latent_yet"
    }

    fn infer(&mut self, input: &ExperienceBehaviorInput) -> Result<ExperienceBehaviorOutput> {
        Ok(ExperienceBehaviorOutput {
            latent: ExperienceLatent {
                t_ms: input.now.t_ms,
                z: Vec::new(),
                reconstruction_error: 0.0,
                prediction_error: 0.0,
                confidence: 0.0,
            },
            reconstruction: None,
            reconstruction_loss: None,
            confidence: 0.0,
        })
    }
}

struct LearnedExperienceBehavior {
    model: ExperienceAutoencoderTrainer,
}

impl FunctionBehavior<ExperienceBehaviorInput, ExperienceBehaviorOutput>
    for LearnedExperienceBehavior
{
    fn id(&self) -> &'static str {
        "experience.autoencoder.v0"
    }

    fn infer(&mut self, input: &ExperienceBehaviorInput) -> Result<ExperienceBehaviorOutput> {
        let encode_input = ExperienceEncodeInput {
            sense_vectors: input.sense_vectors.clone(),
        };
        let prediction = self.model.predict(&encode_input)?;
        let target = experience_decode_target_from_now(&input.now);
        let reconstruction_loss = experience_reconstruction_loss_flat(&prediction.decoded, &target);
        let latent = ExperienceLatent {
            t_ms: input.now.t_ms,
            z: prediction.encoded.z.clone(),
            reconstruction_error: reconstruction_loss,
            prediction_error: 0.0,
            confidence: prediction.encoded.confidence,
        };
        Ok(ExperienceBehaviorOutput {
            latent,
            reconstruction: Some(prediction.decoded),
            reconstruction_loss: Some(reconstruction_loss),
            confidence: prediction.encoded.confidence,
        })
    }

    fn observe(
        &mut self,
        sample: &TrainingSample<ExperienceBehaviorInput, ExperienceBehaviorOutput>,
    ) -> Result<()> {
        let encode_input = ExperienceEncodeInput {
            sense_vectors: sample.input.sense_vectors.clone(),
        };
        let target = experience_decode_target_from_now(&sample.input.now);
        self.model.train_step(&encode_input, &target)?;
        Ok(())
    }
}

struct HardcodedDangerBehavior;

impl FunctionBehavior<SituatedDangerInput, DangerOutput> for HardcodedDangerBehavior {
    fn id(&self) -> &'static str {
        "danger.range_bumper"
    }

    fn infer(&mut self, input: &SituatedDangerInput) -> Result<DangerOutput> {
        Ok(HardcodedDangerPredictor.predict_from_now(&input.now, &input.input))
    }
}

struct DangerModelBehavior {
    trainer: DangerNetTrainer,
}

impl FunctionBehavior<SituatedDangerInput, DangerOutput> for DangerModelBehavior {
    fn id(&self) -> &'static str {
        "danger.burn.v0"
    }

    fn infer(&mut self, input: &SituatedDangerInput) -> Result<DangerOutput> {
        self.trainer.predict(&input.input)
    }

    fn observe(
        &mut self,
        sample: &TrainingSample<SituatedDangerInput, DangerOutput>,
    ) -> Result<()> {
        let target = pete_experience::DangerTarget {
            bump: sample.expected.bump_risk,
            cliff: sample.expected.cliff_risk,
            wheel_drop: sample.expected.wheel_drop_risk,
            stuck: sample.expected.stuck_risk,
        };
        self.trainer.train_step(&sample.input.input, &target)?;
        Ok(())
    }
}

struct HardcodedChargeBehavior;

impl FunctionBehavior<SituatedChargeInput, ChargeOutput> for HardcodedChargeBehavior {
    fn id(&self) -> &'static str {
        "charge.sensor_battery_delta"
    }

    fn infer(&mut self, input: &SituatedChargeInput) -> Result<ChargeOutput> {
        Ok(HardcodedChargePredictor.predict_from_now(&input.now, &input.input))
    }
}

struct ChargeModelBehavior {
    trainer: ChargeNetTrainer,
}

impl FunctionBehavior<SituatedChargeInput, ChargeOutput> for ChargeModelBehavior {
    fn id(&self) -> &'static str {
        "charge.burn.v0"
    }

    fn infer(&mut self, input: &SituatedChargeInput) -> Result<ChargeOutput> {
        self.trainer.predict(&input.input)
    }

    fn observe(
        &mut self,
        sample: &TrainingSample<SituatedChargeInput, ChargeOutput>,
    ) -> Result<()> {
        let target = pete_experience::ChargeTarget {
            charging_started: sample.expected.charge_probability,
            battery_delta: sample.expected.expected_battery_delta,
            charging_after: sample.expected.dock_likelihood,
        };
        self.trainer.train_step(&sample.input.input, &target)?;
        Ok(())
    }
}

struct HardcodedActionValueBehavior;

impl FunctionBehavior<SituatedActionValueInput, ActionValueOutput>
    for HardcodedActionValueBehavior
{
    fn id(&self) -> &'static str {
        "action_value.handcoded"
    }

    fn infer(&mut self, input: &SituatedActionValueInput) -> Result<ActionValueOutput> {
        Ok(HardcodedActionValuePredictor.predict_from_now(&input.now, &input.input))
    }
}

struct ActionValueModelBehavior {
    trainer: ActionValueNetTrainer,
}

impl FunctionBehavior<SituatedActionValueInput, ActionValueOutput> for ActionValueModelBehavior {
    fn id(&self) -> &'static str {
        "action_value.burn.v0"
    }

    fn infer(&mut self, input: &SituatedActionValueInput) -> Result<ActionValueOutput> {
        self.trainer.predict(&input.input)
    }

    fn observe(
        &mut self,
        sample: &TrainingSample<SituatedActionValueInput, ActionValueOutput>,
    ) -> Result<()> {
        let target = pete_experience::ActionValueTarget {
            value: sample.expected.value,
        };
        self.trainer.train_step(&sample.input.input, &target)?;
        Ok(())
    }
}

struct HardcodedEyeNextBehavior;

impl FunctionBehavior<SituatedEyeNextInput, EyeNextOutput> for HardcodedEyeNextBehavior {
    fn id(&self) -> &'static str {
        "eye.copy_current"
    }

    fn infer(&mut self, input: &SituatedEyeNextInput) -> Result<EyeNextOutput> {
        Ok(CopyCurrentEyePredictor.predict_from_now(&input.now, &input.input))
    }
}

struct EyeNextModelBehavior {
    trainer: EyeNextNetTrainer,
}

impl FunctionBehavior<SituatedEyeNextInput, EyeNextOutput> for EyeNextModelBehavior {
    fn id(&self) -> &'static str {
        "eye.burn.next_v0"
    }

    fn infer(&mut self, input: &SituatedEyeNextInput) -> Result<EyeNextOutput> {
        self.trainer.predict(&input.input)
    }

    fn observe(
        &mut self,
        sample: &TrainingSample<SituatedEyeNextInput, EyeNextOutput>,
    ) -> Result<()> {
        let target = pete_experience::EyeNextTarget {
            width: sample.expected.width,
            height: sample.expected.height,
            rgb: sample.expected.rgb.clone(),
        };
        self.trainer.train_step(&sample.input.input, &target)?;
        Ok(())
    }
}

struct HardcodedEarNextBehavior;

impl FunctionBehavior<SituatedEarNextInput, EarNextOutput> for HardcodedEarNextBehavior {
    fn id(&self) -> &'static str {
        "ear.copy_current"
    }

    fn infer(&mut self, input: &SituatedEarNextInput) -> Result<EarNextOutput> {
        Ok(CopyCurrentEarPredictor.predict_from_now(&input.now, &input.input))
    }
}

struct EarNextModelBehavior {
    trainer: EarNextNetTrainer,
}

impl FunctionBehavior<SituatedEarNextInput, EarNextOutput> for EarNextModelBehavior {
    fn id(&self) -> &'static str {
        "ear.burn.next_v0"
    }

    fn infer(&mut self, input: &SituatedEarNextInput) -> Result<EarNextOutput> {
        self.trainer.predict(&input.input)
    }

    fn observe(
        &mut self,
        sample: &TrainingSample<SituatedEarNextInput, EarNextOutput>,
    ) -> Result<()> {
        let target = pete_experience::EarNextTarget {
            sample_rate_hz: sample.expected.sample_rate_hz,
            channels: sample.expected.channels,
            pcm: sample.expected.pcm.clone(),
            features: sample.expected.features.clone(),
        };
        self.trainer.train_step(&sample.input.input, &target)?;
        Ok(())
    }
}

struct StasisFutureBehavior {
    predictor: StasisFuturePredictor,
}

impl FunctionBehavior<FutureInput, FuturePrediction> for StasisFutureBehavior {
    fn id(&self) -> &'static str {
        "future.stasis"
    }

    fn infer(&mut self, input: &FutureInput) -> Result<FuturePrediction> {
        self.predictor
            .predict(&input.latent, &input.action, input.offset_ms)
    }
}

struct FutureModelBehavior {
    trainer: FutureNetTrainer,
}

impl FunctionBehavior<FutureInput, FuturePrediction> for FutureModelBehavior {
    fn id(&self) -> &'static str {
        "future.burn.v0"
    }

    fn infer(&mut self, input: &FutureInput) -> Result<FuturePrediction> {
        let mut input = input.clone();
        if input.flat_features().len() != self.trainer.input_dim() {
            input.latent.z.resize(self.trainer.latent_dim(), 0.0);
            input.latent.z.truncate(self.trainer.latent_dim());
            let expected_input_dim = self.trainer.latent_dim() + action_features(None).len() + 1;
            if expected_input_dim != self.trainer.input_dim() {
                return Err(anyhow::anyhow!(
                    "future checkpoint input dimension mismatch: checkpoint expects {}, adapted runtime input would be {}",
                    self.trainer.input_dim(),
                    expected_input_dim
                ));
            }
        }
        self.trainer.predict(&input)
    }

    fn observe(&mut self, sample: &TrainingSample<FutureInput, FuturePrediction>) -> Result<()> {
        self.trainer
            .train_step(&sample.input, &sample.expected.predicted_z)?;
        Ok(())
    }
}

fn locomotion_behavior(
    regime: BehaviorRegime,
    model: Option<NeatLocomotionBehavior>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<LocomotionInput, LocomotionOutput> {
    ReplaceableBehavior::new(
        "locomotion",
        regime,
        Box::new(HardcodedLocomotionBehavior::default()),
        model.map(|model| Box::new(model) as Box<_>),
        fallback,
    )
}

fn danger_behavior(
    regime: BehaviorRegime,
    trainer: Option<DangerNetTrainer>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<SituatedDangerInput, DangerOutput> {
    ReplaceableBehavior::new(
        "danger",
        regime,
        Box::new(HardcodedDangerBehavior),
        trainer.map(|trainer| Box::new(DangerModelBehavior { trainer }) as Box<_>),
        fallback,
    )
}

fn experience_behavior(
    regime: BehaviorRegime,
    trainer: Option<ExperienceAutoencoderTrainer>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<ExperienceBehaviorInput, ExperienceBehaviorOutput> {
    ReplaceableBehavior::new(
        "experience",
        regime,
        Box::new(HardcodedExperienceBehavior),
        trainer.map(|trainer| Box::new(LearnedExperienceBehavior { model: trainer }) as Box<_>),
        fallback,
    )
}

fn charge_behavior(
    regime: BehaviorRegime,
    trainer: Option<ChargeNetTrainer>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<SituatedChargeInput, ChargeOutput> {
    ReplaceableBehavior::new(
        "charge",
        regime,
        Box::new(HardcodedChargeBehavior),
        trainer.map(|trainer| Box::new(ChargeModelBehavior { trainer }) as Box<_>),
        fallback,
    )
}

fn future_behavior(
    regime: BehaviorRegime,
    trainer: Option<FutureNetTrainer>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<FutureInput, FuturePrediction> {
    ReplaceableBehavior::new(
        "future",
        regime,
        Box::new(StasisFutureBehavior {
            predictor: StasisFuturePredictor,
        }),
        trainer.map(|trainer| Box::new(FutureModelBehavior { trainer }) as Box<_>),
        fallback,
    )
}

fn action_value_behavior(
    regime: BehaviorRegime,
    trainer: Option<ActionValueNetTrainer>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<SituatedActionValueInput, ActionValueOutput> {
    ReplaceableBehavior::new(
        "action_value",
        regime,
        Box::new(HardcodedActionValueBehavior),
        trainer.map(|trainer| Box::new(ActionValueModelBehavior { trainer }) as Box<_>),
        fallback,
    )
}

fn conductor_behavior(
    regime: BehaviorRegime,
    hardcoded_id: &str,
    model_id: Option<String>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<ConductorInput, ActionPrimitive> {
    ReplaceableBehavior::new(
        "conductor",
        regime,
        Box::new(HardcodedConductorBehavior {
            id: known_conductor_hardcoded_id(hardcoded_id),
        }),
        model_id.map(|id| {
            Box::new(ShadowActionSelectorModel {
                id,
                last_observed: None,
                samples_seen: 0,
            }) as Box<_>
        }),
        fallback,
    )
}

fn known_conductor_hardcoded_id(id: &str) -> &'static str {
    match id {
        "action_selector.baseline" => "action_selector.baseline",
        "reign.teacher" => "reign.teacher",
        _ => "conductor.simple_v0",
    }
}

struct HardcodedConductorBehavior {
    id: &'static str,
}

impl FunctionBehavior<ConductorInput, ActionPrimitive> for HardcodedConductorBehavior {
    fn id(&self) -> &'static str {
        self.id
    }

    fn infer(&mut self, input: &ConductorInput) -> Result<ActionPrimitive> {
        match self.id {
            "reign.teacher" => input
                .reign
                .latest
                .as_ref()
                .and_then(|input| input.command.to_action())
                .or_else(|| input.proposals.last().cloned())
                .map(Ok)
                .unwrap_or_else(|| Ok(ActionPrimitive::Stop)),
            "action_selector.baseline" => {
                Ok(input
                    .proposals
                    .last()
                    .cloned()
                    .unwrap_or(ActionPrimitive::Explore {
                        style: ExploreStyle::RandomWalk,
                        duration_ms: 1_000,
                    }))
            }
            _ => SimpleConductor::default().choose(input.clone()),
        }
    }
}

struct ShadowActionSelectorModel {
    id: String,
    last_observed: Option<ActionPrimitive>,
    samples_seen: usize,
}

impl FunctionBehavior<ConductorInput, ActionPrimitive> for ShadowActionSelectorModel {
    fn id(&self) -> &'static str {
        "conductor.burn.v0"
    }

    fn infer(&mut self, _input: &ConductorInput) -> Result<ActionPrimitive> {
        self.last_observed
            .clone()
            .ok_or_else(|| anyhow::anyhow!("{} has no observed teacher samples", self.id))
    }

    fn observe(&mut self, sample: &TrainingSample<ConductorInput, ActionPrimitive>) -> Result<()> {
        if sample.source != TrainingSource::SafetyVeto {
            self.last_observed = Some(sample.expected.clone());
            self.samples_seen = self.samples_seen.saturating_add(1);
        }
        Ok(())
    }
}

struct BumpScriptBehavior;

impl FunctionBehavior<BumpEventInput, EventScriptOutput> for BumpScriptBehavior {
    fn id(&self) -> &'static str {
        "script.on_bump.ts.v0"
    }

    fn infer(&mut self, input: &BumpEventInput) -> Result<EventScriptOutput> {
        execute_event_script_typescript(BUMP_SCRIPT, input)
    }
}

const BUMP_SCRIPT: &str = r#"
const r = random();
const lament =
  r < 0.20 ? say("Uh-oh") :
  r < 0.40 ? say("Oh no!") :
  r < 0.60 ? say("Oopsie!") :
  r < 0.80 ? say("Oh dear!") :
             song("mournful_bump");

[
  chirp("Warning"),
  lament,
  stop(),
  rotate(180),
  go()
]
"#;

struct RobotInitializedScriptBehavior;

impl FunctionBehavior<RobotInitializedEventInput, EventScriptOutput>
    for RobotInitializedScriptBehavior
{
    fn id(&self) -> &'static str {
        "script.on_robot_initialized.ts.v0"
    }

    fn infer(&mut self, input: &RobotInitializedEventInput) -> Result<EventScriptOutput> {
        execute_event_script_typescript(ROBOT_INITIALIZED_SCRIPT, input)
    }
}

const ROBOT_INITIALIZED_SCRIPT: &str = r#"
[
  song("bring_up"),
  chirp("Wake"),
  chirp("Hello"),
  say(`Pete robot initialization complete in ${input.mode} mode.`),
  say(`${input.body}.`),
  input.battery_percent === null
    ? say("Battery status is unavailable.")
    : say(`Battery is ${input.battery_percent} percent and ${input.charging ? "charging" : "not charging"}.`),
  input.requested_sensors === 0
    ? say("No optional sensors requested.")
    : say(`${input.active_sensors} of ${input.requested_sensors} optional sensors initialized.`),
  say(`Ledger is ready at ${input.ledger}.`),
  say(`Tick rate is ${input.tick_ms} milliseconds.`),
  input.dashboard ? say(`Dashboard is listening at ${input.dashboard}.`) : say("Dashboard is not enabled."),
  input.capture ? say(`Capture recording is armed at ${input.capture}.`) : say("Capture recording is not enabled."),
  input.mode === "read-only"
    ? say("Read only mode is active. Motors are suppressed.")
    : say("Slow mode is active. Guarded motor commands are enabled."),
  chirp("Confirm")
]
"#;

struct EventScriptShadowModel {
    id: &'static str,
    last_observed: Option<EventScriptOutput>,
    samples_seen: usize,
}

impl<I> FunctionBehavior<I, EventScriptOutput> for EventScriptShadowModel
where
    I: Send,
{
    fn id(&self) -> &'static str {
        self.id
    }

    fn infer(&mut self, _input: &I) -> Result<EventScriptOutput> {
        self.last_observed
            .clone()
            .ok_or_else(|| anyhow::anyhow!("{} has no observed script samples", self.id))
    }

    fn observe(&mut self, sample: &TrainingSample<I, EventScriptOutput>) -> Result<()> {
        self.last_observed = Some(sample.expected.clone());
        self.samples_seen = self.samples_seen.saturating_add(1);
        Ok(())
    }
}

fn robot_initialized_event_behavior(
    regime: BehaviorRegime,
    model_id: Option<String>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<RobotInitializedEventInput, EventScriptOutput> {
    ReplaceableBehavior::new(
        "event_robot_initialized",
        regime,
        Box::new(RobotInitializedScriptBehavior),
        model_id.map(|_| {
            Box::new(EventScriptShadowModel {
                id: "event.robot_initialized.shadow.v0",
                last_observed: None,
                samples_seen: 0,
            }) as Box<_>
        }),
        fallback,
    )
}

fn bump_event_behavior(
    regime: BehaviorRegime,
    model_id: Option<String>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<BumpEventInput, EventScriptOutput> {
    ReplaceableBehavior::new(
        "event_bump",
        regime,
        Box::new(BumpScriptBehavior),
        model_id.map(|_| {
            Box::new(EventScriptShadowModel {
                id: "event.bump.shadow.v0",
                last_observed: None,
                samples_seen: 0,
            }) as Box<_>
        }),
        fallback,
    )
}

fn eye_next_behavior(
    regime: BehaviorRegime,
    trainer: Option<EyeNextNetTrainer>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<SituatedEyeNextInput, EyeNextOutput> {
    ReplaceableBehavior::new(
        "eye_next",
        regime,
        Box::new(HardcodedEyeNextBehavior),
        trainer.map(|trainer| Box::new(EyeNextModelBehavior { trainer }) as Box<_>),
        fallback,
    )
}

fn ear_next_behavior(
    regime: BehaviorRegime,
    trainer: Option<EarNextNetTrainer>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<SituatedEarNextInput, EarNextOutput> {
    ReplaceableBehavior::new(
        "ear_next",
        regime,
        Box::new(HardcodedEarNextBehavior),
        trainer.map(|trainer| Box::new(EarNextModelBehavior { trainer }) as Box<_>),
        fallback,
    )
}

fn load_locomotion_behavior(behavior: &BehaviorConfig) -> Result<Option<NeatLocomotionBehavior>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    let checkpoint_path = Path::new(checkpoint);
    let artifact = if checkpoint_path.extension().is_some() {
        checkpoint_path.to_path_buf()
    } else {
        checkpoint_path.join("locomotion-neat.json")
    };
    if !artifact.exists() {
        return Ok(None);
    }
    Ok(Some(NeatLocomotionBehavior::load(checkpoint)?))
}

fn load_danger_behavior_trainer(behavior: &BehaviorConfig) -> Result<Option<DangerNetTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_danger_metadata(checkpoint)?;
    Ok(Some(DangerNetTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
    )?))
}

fn load_charge_behavior_trainer(behavior: &BehaviorConfig) -> Result<Option<ChargeNetTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_charge_metadata(checkpoint)?;
    Ok(Some(ChargeNetTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
    )?))
}

fn load_action_value_behavior_trainer(
    behavior: &BehaviorConfig,
) -> Result<Option<ActionValueNetTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_action_value_metadata(checkpoint)?;
    Ok(Some(ActionValueNetTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
    )?))
}

fn load_future_behavior_trainer(behavior: &BehaviorConfig) -> Result<Option<FutureNetTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_future_metadata(checkpoint)?;
    Ok(Some(FutureNetTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
        metadata.latent_dim,
    )?))
}

fn load_eye_next_behavior_trainer(behavior: &BehaviorConfig) -> Result<Option<EyeNextNetTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_eye_next_metadata(checkpoint)?;
    Ok(Some(EyeNextNetTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
    )?))
}

fn load_ear_next_behavior_trainer(behavior: &BehaviorConfig) -> Result<Option<EarNextNetTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_ear_next_metadata(checkpoint)?;
    Ok(Some(EarNextNetTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
    )?))
}

fn load_experience_behavior_trainer(
    behavior: &BehaviorConfig,
) -> Result<Option<ExperienceAutoencoderTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_experience_autoencoder_metadata(checkpoint)?;
    Ok(Some(ExperienceAutoencoderTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
    )?))
}

fn danger_behavior_input(
    now: &Now,
    latent: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
) -> SituatedDangerInput {
    SituatedDangerInput {
        input: DangerInput::from_parts(latent.z.clone(), action, now),
        now: now.clone(),
    }
}

fn charge_behavior_input(
    now: &Now,
    latent: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
) -> SituatedChargeInput {
    SituatedChargeInput {
        input: ChargeInput::from_parts(latent.z.clone(), action, now),
        now: now.clone(),
    }
}

fn action_value_behavior_input(
    now: &Now,
    latent: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
    danger: Option<DangerOutput>,
    charge: Option<ChargeOutput>,
) -> SituatedActionValueInput {
    SituatedActionValueInput {
        input: ActionValueInput::from_parts_with_predictions(
            latent.z.clone(),
            action,
            now,
            danger,
            charge,
        ),
        now: now.clone(),
    }
}

fn eye_next_behavior_input(
    now: &Now,
    latent: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
    offset_ms: TimeMs,
) -> SituatedEyeNextInput {
    SituatedEyeNextInput {
        input: EyeNextInput::from_parts(latent.z.clone(), action, now, offset_ms),
        now: now.clone(),
    }
}

fn ear_next_behavior_input(
    now: &Now,
    latent: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
    offset_ms: TimeMs,
) -> SituatedEarNextInput {
    SituatedEarNextInput {
        input: EarNextInput::from_parts(latent.z.clone(), action, now, offset_ms),
        now: now.clone(),
    }
}

fn danger_disagreement(left: &DangerOutput, right: &DangerOutput) -> f32 {
    let deltas = [
        (left.bump_risk - right.bump_risk).abs(),
        (left.cliff_risk - right.cliff_risk).abs(),
        (left.wheel_drop_risk - right.wheel_drop_risk).abs(),
        (left.stuck_risk - right.stuck_risk).abs(),
    ];
    deltas.iter().sum::<f32>() / deltas.len() as f32
}

fn action_value_disagreement(left: &ActionValueOutput, right: &ActionValueOutput) -> f32 {
    (left.value - right.value).abs()
}

fn charge_disagreement(left: &ChargeOutput, right: &ChargeOutput) -> f32 {
    let deltas = [
        (left.charge_probability - right.charge_probability).abs(),
        (left.expected_battery_delta - right.expected_battery_delta).abs(),
        (left.dock_likelihood - right.dock_likelihood).abs(),
    ];
    deltas.iter().sum::<f32>() / deltas.len() as f32
}

fn eye_next_disagreement(left: &EyeNextOutput, right: &EyeNextOutput) -> f32 {
    let len = left.rgb.len().max(right.rgb.len());
    if len == 0 {
        return 0.0;
    }
    (0..len)
        .map(|idx| {
            let left = left.rgb.get(idx).copied().unwrap_or_default() as f32 / 255.0;
            let right = right.rgb.get(idx).copied().unwrap_or_default() as f32 / 255.0;
            (left - right).abs()
        })
        .sum::<f32>()
        / len as f32
}

fn ear_next_disagreement(left: &EarNextOutput, right: &EarNextOutput) -> f32 {
    let len = left.features.len().max(right.features.len());
    if len == 0 {
        return 0.0;
    }
    (0..len)
        .map(|idx| {
            let left = left.features.get(idx).copied().unwrap_or_default();
            let right = right.features.get(idx).copied().unwrap_or_default();
            (left - right).abs()
        })
        .sum::<f32>()
        / len as f32
}

fn experience_reconstruction_loss_flat(
    output: &ExperienceDecodeOutput,
    target: &ExperienceDecodeOutput,
) -> f32 {
    let output = output.flat_features();
    let target = target.flat_features();
    let len = output.len().max(target.len());
    if len == 0 {
        return 0.0;
    }
    (0..len)
        .map(|idx| {
            let actual = output.get(idx).copied().unwrap_or_default();
            let expected = target.get(idx).copied().unwrap_or_default();
            let delta = actual - expected;
            delta * delta
        })
        .sum::<f32>()
        / len as f32
}

fn experience_disagreement(
    left: &ExperienceBehaviorOutput,
    right: &ExperienceBehaviorOutput,
) -> f32 {
    let a = &left.latent.z;
    let b = &right.latent.z;
    let len = a.len().max(b.len());
    if len == 0 {
        return 0.0;
    }
    let sum: f32 = (0..len)
        .map(|idx| {
            let delta =
                a.get(idx).copied().unwrap_or_default() - b.get(idx).copied().unwrap_or_default();
            delta * delta
        })
        .sum();
    sum.sqrt()
}

#[derive(Clone, Debug, Default)]
pub struct ReignQueue {
    pending: VecDeque<ReignInput>,
    latest: Option<ReignInput>,
    clear_sequence: u64,
}

impl ReignQueue {
    pub fn push(&mut self, input: ReignInput) {
        self.latest = Some(input.clone());
        self.pending.push_back(input);
    }

    pub fn latest_active(&self, now_ms: TimeMs) -> Option<ReignInput> {
        self.pending
            .iter()
            .rev()
            .find(|input| input.expires_at_ms > now_ms)
            .cloned()
    }

    pub fn drain_expired(&mut self, now_ms: TimeMs) {
        self.pending.retain(|input| input.expires_at_ms > now_ms);
        if self
            .latest
            .as_ref()
            .map(|input| input.expires_at_ms <= now_ms)
            .unwrap_or(false)
        {
            self.latest = self.latest_active(now_ms);
        }
    }

    pub fn clear(&mut self) {
        self.pending.clear();
        self.latest = None;
        self.clear_sequence = self.clear_sequence.saturating_add(1);
    }

    pub fn sense(&self, now_ms: TimeMs) -> ReignSense {
        let latest = self.latest_active(now_ms);
        let active = latest.is_some();
        ReignSense {
            active,
            mode: latest.as_ref().map(|input| input.mode.clone()),
            last_command_age_ms: latest
                .as_ref()
                .map(|input| now_ms.saturating_sub(input.issued_at_ms)),
            human_override_pressure: latest
                .as_ref()
                .map(|input| input.priority.clamp(0.0, 1.0))
                .unwrap_or(0.0),
            latest,
            pending_count: self
                .pending
                .iter()
                .filter(|input| input.expires_at_ms > now_ms)
                .count(),
            clear_sequence: self.clear_sequence,
        }
    }
}

fn mechanical_reign_action(
    input: &Option<ReignInput>,
    selector_mode: ActionSelectorMode,
) -> Option<ActionPrimitive> {
    let input = input.as_ref()?;
    let goal_mode = selector_mode == ActionSelectorMode::Goal;
    let mechanical = matches!(input.mode, pete_actions::ReignMode::Direct)
        || (!goal_mode && matches!(input.mode, pete_actions::ReignMode::Assist));
    if !mechanical {
        return None;
    }
    input.command.to_action()
}

fn reign_input_drives_sim_directly(input: &ReignInput) -> bool {
    matches!(
        input.mode,
        pete_actions::ReignMode::Direct | pete_actions::ReignMode::Assist
    ) && input.command.to_action().is_some()
}

impl<L, M, R, C, S, A> MinimalRuntime<L, M, R, C, S, A>
where
    L: LedgerWriter + Sync,
    M: MemoryStore,
    R: Recall + Sync,
    C: Conductor,
    S: SafetyLayer,
    A: LlmAgent + 'static,
{
    pub fn new(
        ledger: L,
        memory_store: M,
        memory_recall: R,
        conductor: C,
        safety: S,
        llm: A,
    ) -> Self {
        let cognition = RuntimeCognition::from_agent(&llm);
        Self {
            ledger,
            memory_store,
            memory_recall,
            conductor,
            safety,
            llm: Arc::new(tokio::sync::Mutex::new(llm)),
            extractor: EventExtractor::default(),
            bus: default_event_bus(),
            reign_queue: Arc::new(Mutex::new(ReignQueue::default())),
            predictor: StasisFuturePredictor,
            models: RuntimeModelStack::default(),
            action_selector_mode: ActionSelectorMode::Baseline,
            surprise_computer: BaselineSurpriseComputer,
            reward_computer: BaselineRewardComputer,
            transition_builder: TransitionBuilder::new(),
            behavior_training_hub: BehaviorTrainingHub::default(),
            surface_extractor: SurfaceExtractor::default(),
            inline_learning: InlineLearningConfig::default(),
            nudge_policy: NudgePolicy::default(),
            local_map: LocalMap::default(),
            last_behavior_runs: Vec::new(),
            locomotion_tracker: LocomotionTracker::default(),
            chirp_events: ChirpEventState::default(),
            nudge: NudgeController::default(),
            goal_system: GoalSystem::default(),
            world_model: WorldModelUpdater::default(),
            sleep_controller: SleepController::default(),
            semantic_outcomes: SemanticOutcomeTracker::default(),
            last_active_control: None,
            cognition,
        }
    }

    pub fn with_reign_queue(
        ledger: L,
        memory_store: M,
        memory_recall: R,
        conductor: C,
        safety: S,
        llm: A,
        reign_queue: Arc<Mutex<ReignQueue>>,
    ) -> Self {
        let cognition = RuntimeCognition::from_agent(&llm);
        Self {
            ledger,
            memory_store,
            memory_recall,
            conductor,
            safety,
            llm: Arc::new(tokio::sync::Mutex::new(llm)),
            extractor: EventExtractor::default(),
            bus: default_event_bus(),
            reign_queue,
            predictor: StasisFuturePredictor,
            models: RuntimeModelStack::default(),
            action_selector_mode: ActionSelectorMode::Baseline,
            surprise_computer: BaselineSurpriseComputer,
            reward_computer: BaselineRewardComputer,
            transition_builder: TransitionBuilder::new(),
            behavior_training_hub: BehaviorTrainingHub::default(),
            surface_extractor: SurfaceExtractor::default(),
            inline_learning: InlineLearningConfig::default(),
            nudge_policy: NudgePolicy::default(),
            local_map: LocalMap::default(),
            last_behavior_runs: Vec::new(),
            locomotion_tracker: LocomotionTracker::default(),
            chirp_events: ChirpEventState::default(),
            nudge: NudgeController::default(),
            goal_system: GoalSystem::default(),
            world_model: WorldModelUpdater::default(),
            sleep_controller: SleepController::default(),
            semantic_outcomes: SemanticOutcomeTracker::default(),
            last_active_control: None,
            cognition,
        }
    }

    pub fn with_default_events(
        ledger: L,
        memory_store: M,
        memory_recall: R,
        conductor: C,
        safety: S,
        llm: A,
    ) -> Self {
        Self::new(ledger, memory_store, memory_recall, conductor, safety, llm)
    }

    pub fn with_models(mut self, models: RuntimeModelStack) -> Self {
        self.models = models;
        self
    }

    pub fn with_action_selector_mode(mut self, mode: ActionSelectorMode) -> Self {
        self.action_selector_mode = mode;
        self
    }

    pub fn with_inline_learning(mut self, config: InlineLearningConfig) -> Self {
        self.inline_learning = config;
        self
    }

    pub fn with_nudge_policy(mut self, policy: NudgePolicy) -> Self {
        self.nudge_policy = policy;
        self
    }

    pub fn with_local_map(mut self, local_map: LocalMap) -> Self {
        self.local_map = local_map;
        self
    }

    pub fn nudge_status(&self) -> NudgeStatus {
        self.nudge.status.clone()
    }

    /// Cancel optional cognition without disturbing local control state.
    pub fn cancel_cognition(&mut self) {
        if let Some(pending) = self.cognition.pending.take() {
            self.cognition.next_request_at_ms = pending
                .requested_at_ms
                .saturating_add(COGNITION_COOLDOWN_MS);
            pending.task.abort();
            self.cognition.last_outcome = Some(CognitionOutcome::Cancelled);
        }
    }

    pub fn behavior_node_states(&self) -> Vec<BehaviorNodeState> {
        self.models.behavior_node_states(&self.last_behavior_runs)
    }

    /// Poll a previous request and enqueue the current immutable view.
    ///
    /// `JoinHandle::is_finished` is deliberately checked before awaiting it,
    /// making the only await here a ready-value extraction rather than model
    /// or network I/O. Provider output is reduced to `LlmSense` and typed
    /// evidence. Decisions cross only as discarded advisory telemetry;
    /// conscious commands and executable actions never cross this boundary.
    async fn advance_cognition(
        &mut self,
        now: &Now,
        impressions: &[Impression],
        embodied: &pete_experience::EmbodiedContext,
        latent: &ExperienceLatent,
        futures: &[FuturePrediction],
        recall_summary: &str,
        notes: &mut Vec<String>,
    ) -> Option<AcceptedLlmCognition> {
        if now.t_ms > self.cognition.last_sense_valid_until_ms {
            self.cognition.last_sense = pete_now::LlmSense::default();
        }
        let mut accepted = None;
        if self
            .cognition
            .pending
            .as_ref()
            .is_some_and(|pending| now.t_ms > pending.deadline_ms)
        {
            let pending = self.cognition.pending.take().expect("expired task");
            pending.task.abort();
            self.cognition.next_request_at_ms = now.t_ms.saturating_add(COGNITION_COOLDOWN_MS);
            self.cognition.last_outcome = Some(CognitionOutcome::Expired);
        }
        if self
            .cognition
            .pending
            .as_ref()
            .is_some_and(|pending| pending.task.is_finished())
        {
            let pending = self.cognition.pending.take().expect("finished task");
            self.cognition.next_request_at_ms = now.t_ms.saturating_add(COGNITION_COOLDOWN_MS);
            match pending.task.await {
                Err(error) => {
                    let outcome = if error.is_cancelled() {
                        CognitionOutcome::Cancelled
                    } else {
                        CognitionOutcome::Failed(error.to_string())
                    };
                    self.cognition.last_outcome = Some(outcome);
                }
                Ok(Err(error)) => {
                    self.cognition.last_outcome = Some(CognitionOutcome::Failed(error.to_string()));
                }
                Ok(Ok((_reflection, _result))) if now.t_ms > pending.deadline_ms => {
                    self.cognition.last_outcome = Some(CognitionOutcome::Expired);
                }
                Ok(Ok((reflection, result))) => {
                    self.cognition.last_sense = result.sense.clone();
                    self.cognition.last_sense_valid_until_ms =
                        now.t_ms.saturating_add(COGNITION_DEADLINE_MS);
                    self.cognition.last_outcome = Some(CognitionOutcome::Accepted);
                    accepted = Some(AcceptedLlmCognition {
                        reflection,
                        tick: result,
                        snapshot_ref: pending.snapshot_ref,
                        requested_at_ms: pending.requested_at_ms,
                        observed_at_ms: now.t_ms,
                    });
                }
            }
        }

        if self.cognition.provider_declared_available
            && self.cognition.pending.is_none()
            && now.t_ms >= self.cognition.next_request_at_ms
        {
            let llm = Arc::clone(&self.llm);
            let request_now = now.clone();
            let request_impressions = impressions.to_vec();
            let request_embodied = embodied.clone();
            let request_latent = latent.clone();
            let request_futures = futures.to_vec();
            let request_recall = recall_summary.to_string();
            let task = tokio::spawn(async move {
                let mut agent = llm.lock().await;
                let reflection = agent
                    .combobulate(
                        &request_now,
                        &request_impressions,
                        Some(&request_embodied),
                        &request_latent,
                        &request_futures,
                        &request_recall,
                    )
                    .await?;
                let awareness = reflection.as_ref().map(|value| value.summary.as_str());
                let tick = agent
                    .maybe_tick(
                        &request_now,
                        Some(&request_embodied),
                        &request_latent,
                        &request_futures,
                        &request_recall,
                        awareness,
                    )
                    .await?;
                Ok((reflection, tick))
            });
            self.cognition.pending = Some(PendingLlmCognition {
                snapshot_ref: now
                    .extensions
                    .get("frame_id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("unknown-frame")
                    .to_string(),
                requested_at_ms: now.t_ms,
                deadline_ms: now.t_ms.saturating_add(COGNITION_DEADLINE_MS),
                task,
            });
        }
        if let Some(outcome) = self.cognition.last_outcome.as_ref() {
            notes.push(match outcome {
                CognitionOutcome::Accepted => "LlmProviderOutcome: accepted".to_string(),
                CognitionOutcome::Expired => "LlmProviderOutcome: expired".to_string(),
                CognitionOutcome::Cancelled => "LlmProviderOutcome: cancelled".to_string(),
                CognitionOutcome::Failed(error) => format!("LlmProviderOutcome: failed: {error}"),
            });
        }
        accepted
    }

    pub fn apply_behavior_node_update(&mut self, node_id: &str, update: &BehaviorNodeUpdate) {
        self.models.apply_behavior_node_update(node_id, update);
    }

    pub async fn tick(
        &mut self,
        mut now: Now,
        _latent: ExperienceLatent,
        mut futures: Vec<FuturePrediction>,
    ) -> Result<RuntimeTick> {
        let frame_id = Uuid::new_v4();
        now.extensions.insert(
            "frame_id".to_string(),
            serde_json::Value::String(frame_id.to_string()),
        );
        apply_create_ir_charger_cue(&mut now);
        {
            let mut reign_queue = self
                .reign_queue
                .lock()
                .map_err(|_| anyhow::anyhow!("reign queue lock poisoned"))?;
            reign_queue.drain_expired(now.t_ms);
            now.reign = reign_queue.sense(now.t_ms);
        }
        let reign_input = now.reign.latest.clone();
        let reign_action = reign_input
            .as_ref()
            .and_then(|input| input.command.to_action());
        let mechanical_reign_action =
            mechanical_reign_action(&reign_input, self.action_selector_mode);

        let mut behavior_runs: Vec<ErasedBehaviorRunRecord> = Vec::new();
        let experience_input = ExperienceBehaviorInput::from_now(&now);
        let experience_run = self
            .models
            .behaviors
            .experience
            .infer(&experience_input, now.t_ms)?;
        let mut experience_record = experience_run.record;
        if let (Some(hard), Some(model)) = (
            experience_record.hardcoded_output.as_ref(),
            experience_record.model_output.as_ref(),
        ) {
            experience_record.disagreement = Some(experience_disagreement(hard, model));
        }
        if let Some(model_output) = experience_record.model_output.as_ref() {
            if let Some(loss) = model_output.reconstruction_loss {
                now.extensions.insert(
                    "experience.autoencoder".to_string(),
                    serde_json::json!({
                        "reconstruction_loss": loss,
                        "z_dim": model_output.latent.z.len(),
                    }),
                );
            }
        }
        behavior_runs.push(experience_record.erase());
        let latent = experience_run.chosen.latent.clone();

        let mut recall_query = RecallQuery::from_now(&now);
        let place_recognition_input =
            place_recognition_input_from_query_now(&now, Some(&latent), "runtime-pre-frame");
        recall_query.place_recognition_input = Some(place_recognition_input.clone());
        let loop_min_confidence = self.local_map.config.pose_graph_min_loop_confidence;
        let live_loop_candidates = self
            .memory_recall
            .loop_closure_candidates(&recall_query, loop_min_confidence, 10)
            .await?
            .iter()
            .map(|candidate| {
                place_candidate_to_loop_input(
                    candidate,
                    Some(frame_id.to_string()),
                    Some(&place_recognition_input),
                )
            })
            .collect::<Vec<_>>();
        let recall = self.memory_recall.recall(recall_query).await?;
        now.memory = recall.sense.clone();
        apply_recent_trap_memory_hints(&mut now);
        now.extensions.insert(
            "memory.place".to_string(),
            serde_json::json!({
                "danger": now.memory.place_danger,
                "charge": now.memory.place_charge_value,
                "social": now.memory.place_social_value,
                "novelty": now.memory.place_novelty,
                "confidence": now.memory.map_confidence,
                "places_visited": now.memory.places_visited,
                "nearby_best_charge_direction_rad": now.memory.nearby_best_charge_direction_rad,
                "nearby_best_safe_direction_rad": now.memory.nearby_best_safe_direction_rad,
                "nearby_frontier_direction_rad": now.memory.nearby_frontier_direction_rad,
                "recent_trap_direction_rad": now.memory.recent_trap_direction_rad,
                "recent_trap_confidence": now.memory.recent_trap_confidence,
            }),
        );
        if let Some(semantic_map) = &recall.semantic_map {
            now.extensions.insert(
                "memory.semantic_map".to_string(),
                serde_json::to_value(semantic_map)?,
            );
        }

        let mut surface_output_for_anticipation: Option<SurfaceExtractorOutput> = None;
        if !now.kinect.depth_m.is_empty()
            && now.kinect.depth_width > 0
            && now.kinect.depth_height > 0
        {
            let surface_output =
                self.surface_extractor
                    .process(&now.kinect, now.body.odometry, now.t_ms);
            surface_output_for_anticipation = Some(surface_output.clone());
            now.extensions.insert(
                "surface.scene_graph".to_string(),
                serde_json::json!({
                    "diagnostics": surface_output.diagnostics.clone(),
                    "floor": surface_output.floor.clone(),
                    "surfaces": surface_output.stable_surfaces.clone(),
                    "clusters": surface_output.clusters.clone(),
                    "navigation": surface_output.scene_graph.navigation.clone(),
                    "calibration_hint": surface_output.diagnostics.calibration_hint,
                    "obstacle_grid": {
                        "resolution_m": surface_output.obstacle_grid.resolution_m,
                        "half_extent_m": surface_output.obstacle_grid.half_extent_m,
                        "cells": surface_output.obstacle_grid.cells.clone(),
                    },
                }),
            );
        }

        let embodied_now = pete_experience::embody_now(&now).await?;
        let mut sensations = embodied_now.sensations;
        let mut impressions = embodied_now.impressions;
        if let Some(summary) = embodied_now.experience.summary_impression.clone() {
            impressions.push(summary);
        }
        let (direct_sensations, direct_impressions) = derive_direct_impressions_from_now(&now);
        sensations.extend(direct_sensations);
        impressions.extend(direct_impressions);
        let (recall_sensations, recall_impressions) =
            embodied_recall_sensations_and_impressions(&recall);
        let recall_sensation_ids = recall_sensations
            .iter()
            .map(|sensation| sensation.id)
            .collect::<Vec<_>>();
        let recall_impression_ids = recall_impressions
            .iter()
            .map(|impression| impression.id)
            .collect::<Vec<_>>();
        sensations.extend(recall_sensations);
        impressions.extend(recall_impressions);
        let mut experiences = derive_direct_experiences(&impressions, &sensations, now.t_ms);
        let mut embodied_experience = embodied_now.experience;
        embodied_experience
            .sensation_ids
            .extend(recall_sensation_ids);
        embodied_experience
            .impression_ids
            .extend(recall_impression_ids);
        if futures.is_empty() {
            let (predicted, records) =
                predict_baseline_futures(&mut self.models.behaviors.future, &latent, now.t_ms)?;
            futures = predicted;
            behavior_runs.extend(records);
        }
        let mut teachings = Vec::new();
        let mut notes = Vec::new();
        let mut drive_impulses = DriveSense::default();
        if let Some(stuck_values) = now
            .extensions
            .get("sim.stuck")
            .and_then(|value| value.get("values"))
            .and_then(|value| value.as_array())
        {
            let active = stuck_values
                .first()
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0;
            let corner = stuck_values
                .get(1)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0;
            let duration_ms = stuck_values
                .get(3)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0);
            let phase = stuck_values
                .get(4)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0);
            let started = stuck_values
                .get(6)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0;
            let recovered = stuck_values
                .get(7)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0;
            let dead_battery = stuck_values
                .get(8)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0)
                > 0.0;
            if dead_battery {
                notes.push(
                    "VirtualDeadBattery: battery reached 0%; virtual motion stopped".to_string(),
                );
            }
            if started {
                notes.push("StuckDetected: classified as stuck/corner-trap".to_string());
            }
            if active {
                notes.push(format!(
                    "StuckRecovery: class={}, phase={}, duration_ms={duration_ms:.0}",
                    if corner { "corner-trap" } else { "stuck" },
                    stuck_phase_label(phase),
                ));
            }
            if recovered {
                notes.push("StuckRecovery: recovered and resumed exploration".to_string());
            }
        }
        if now
            .extensions
            .get("safety/read_only_veto")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            notes.push("source = real_robot_read_only".to_string());
            notes.push("mode = read_only".to_string());
            notes.push("motor_applied = false".to_string());
            notes.push("ReadOnlyActionSuppressed: motion suppressed by read-only mode".to_string());
        }
        let map_observation = observation_from_now(&now, self.local_map.config);
        let map_summary = self
            .local_map
            .integrate_observation_with_loop_candidates(map_observation, &live_loop_candidates);
        now.extensions.insert(
            MAP_EXTENSION_NAME.to_string(),
            serde_json::to_value(&map_summary)?,
        );
        notes.push(format!(
            "ScanMatchedMap: {} cells ({} occupied, {} free); occupancy scan matching corrects odometry before integration",
            map_summary.cells, map_summary.occupied_cells, map_summary.free_cells
        ));
        let corrected_map_trust = corrected_map_trust_status(&now);
        if !corrected_map_trust.trusted {
            notes.push(format!(
                "MapTrustGate: navigation will not trust spatial memory until corrected SLAM is ready ({})",
                corrected_map_trust
                    .reason
                    .as_deref()
                    .unwrap_or("corrected map is not trusted")
            ));
        }
        let mut proposed_actions = Vec::new();

        let events = self.extractor.events_from_now(&now, Some(&recall));
        let ctx = EventContext {
            now: &now,
            latent: Some(&latent),
            recall: Some(&recall),
            predicted_futures: &futures,
            llm: Some(&now.llm),
            safety: None,
        };
        let event_output = self.bus.dispatch_all(&ctx, events)?;
        apply_responses(
            &mut now,
            event_output.responses,
            &mut sensations,
            &mut impressions,
            &mut experiences,
            &mut teachings,
            &mut notes,
            &mut drive_impulses,
        );
        let (event_script_forced_action, event_script_records) =
            self.run_event_scripts(&mut now, &mut notes, &mut proposed_actions)?;
        behavior_runs.extend(event_script_records);
        self.chirp_events
            .emit_pre_selection_chirps(&mut now, &mut notes)?;
        let runtime_instant = ExperienceInstant::from_parts(
            Some(&embodied_experience),
            &sensations,
            &impressions,
            &futures,
            &recall.recollections,
            &now,
            None,
            None,
            "runtime-live",
        );
        let embodied_context = runtime_instant.embodied_context();

        let accepted_llm = self
            .advance_cognition(
                &now,
                &impressions,
                &embodied_context,
                &latent,
                &futures,
                &recall.first_person_summary,
                &mut notes,
            )
            .await;
        now.llm = self.cognition.last_sense.clone();
        if let Some(accepted) = accepted_llm.as_ref() {
            if let Some(reflection) = accepted.reflection.as_ref() {
                append_combobulation(
                    &mut sensations,
                    &mut impressions,
                    &mut experiences,
                    accepted.requested_at_ms,
                    accepted.observed_at_ms,
                    &accepted.snapshot_ref,
                    reflection,
                );
            }
            apply_llm_tick(
                &accepted.tick,
                accepted.requested_at_ms,
                accepted.observed_at_ms,
                &accepted.snapshot_ref,
                &mut sensations,
                &mut impressions,
                &mut experiences,
                &mut teachings,
            );
        }
        // Higher cognition is advisory. Even a valid response cannot become a
        // Cockpit proposal; local goals, skills, Reign, and safety own motion.
        let combobulation = accepted_llm
            .as_ref()
            .and_then(|accepted| accepted.reflection.clone());
        let llm_advisory_action = accepted_llm.as_ref().and_then(|accepted| {
            accepted
                .tick
                .decision
                .as_ref()
                .and_then(|decision| decision.action.clone())
                .map(|action| LlmAdvisoryAction {
                    action,
                    source: LlmAdvisoryActionSource::ProviderDecision,
                    input_snapshot_ref: accepted.snapshot_ref.clone(),
                    disposition: LlmAdvisoryActionDisposition::DiscardedAtAdvisoryBoundary,
                })
        });
        let llm_tick = accepted_llm
            .map(|accepted| accepted.tick)
            .unwrap_or_default();
        let llm_command_action = None;
        let mut llm_action_proposal = LlmActionProposal {
            proposed_action: llm_command_action.clone(),
            advisory_action: llm_advisory_action.clone(),
            ignored_reason: llm_advisory_action.as_ref().map(|advisory| {
                format!(
                    "provider suggested {:?}; discarded at advisory boundary",
                    advisory.action
                )
            }),
            ..LlmActionProposal::default()
        };
        if let Some(advisory) = llm_advisory_action.as_ref() {
            notes.push(format!(
                "LlmAdvisoryAction: provider suggested {:?}; discarded at advisory boundary (input_snapshot_ref={})",
                advisory.action, advisory.input_snapshot_ref
            ));
        }
        let llm_has_safety_reason = crate::llm_explicit_safety_reason(&llm_tick);
        let direct_reign_active = reign_input
            .as_ref()
            .is_some_and(|input| input.mode == pete_actions::ReignMode::Direct);
        let mechanical_reign_action_for_selection =
            if mechanical_reign_action.is_some() && llm_has_safety_reason && !direct_reign_active {
                notes.push(
                    "LlmActionProposal: explicit safety reason allowed competition with Reign"
                        .to_string(),
                );
                None
            } else {
                mechanical_reign_action.clone()
            };

        let nudge_proposal = self.nudge.propose(&now, self.nudge_policy);
        if let Some(action) = nudge_proposal.clone() {
            notes.push(format!("ProdNudge: proposed {:?}", action));
            proposed_actions.push(action);
        } else if let Some(reason) = self.nudge.status.nudge_blocked_reason.clone() {
            notes.push(format!("ProdNudgeBlocked: {reason}"));
        }

        let mut proposals = proposed_actions.clone();
        if self.action_selector_mode == ActionSelectorMode::Goal {
            if let Some(reign_action) = now
                .reign
                .latest
                .as_ref()
                .filter(|input| matches!(input.mode, ReignMode::Assist | ReignMode::Suggest))
                .and_then(|input| input.command.to_action())
            {
                // In goal mode Assist/Suggest is consumed as an affordance-matched
                // bias by GoalSystem, not reintroduced as a generic task proposal.
                proposals.retain(|proposal| proposal != &reign_action);
            }
        }
        if let Some(action) = llm_command_action.clone() {
            notes.push(format!("LlmActionProposal: proposed {:?}", action));
            proposals.push(action);
        }
        if let Some(action) = event_script_forced_action.clone() {
            push_unique_action(&mut proposals, action);
        }
        self.goal_system
            .add_drive_impulses(std::mem::take(&mut drive_impulses));
        self.goal_system.seed_drives(now.drives.clone());
        let mut world_context = self.goal_system.world_model_update_context();
        if let Some(experience_id) = runtime_instant.experience_id.as_ref() {
            world_context
                .continuity
                .recent_experience_refs
                .push(format!("{experience_id:?}"));
        }
        if let Some(previous_control) = self.last_active_control.as_ref() {
            if let Some(action) = previous_control.action_kind.as_ref() {
                world_context
                    .continuity
                    .recent_self_action_refs
                    .push(action.clone());
            }
            world_context
                .continuity
                .recent_outcome_refs
                .extend(previous_control.veto_reasons.iter().cloned());
        }
        world_context.active_control = self.last_active_control.clone();
        world_context.semantic_observations = self.semantic_outcomes.take_pending();
        let cognition_busy = self.cognition.pending.is_some();
        let cognition_failure = self
            .cognition
            .last_outcome
            .as_ref()
            .and_then(|outcome| match outcome {
                CognitionOutcome::Failed(error) => Some(error.clone()),
                CognitionOutcome::Expired => Some("latest request expired".to_string()),
                CognitionOutcome::Cancelled => Some("latest request was cancelled".to_string()),
                _ => None,
            })
            .or_else(|| {
                (!self.cognition.provider_declared_available)
                    .then(|| self.cognition.provider_unavailable_reason.clone())
                    .flatten()
            });
        // Occupancy and health are separate: a pending task means the healthy
        // service is busy, not unavailable. The post-request cooldown is idle,
        // healthy time rather than either an outage or request occupancy.
        let enhanced_cognition_available =
            self.cognition.provider_declared_available && cognition_failure.is_none();
        world_context.cognitive_services.insert(
            "rich_language".to_string(),
            CognitiveServiceBelief {
                available: enhanced_cognition_available,
                busy: cognition_busy,
                confidence: 1.0,
                unavailable_reason: cognition_failure,
                meta: BeliefMeta {
                    confidence: 1.0,
                    observed_at_ms: now.t_ms,
                    valid_at_ms: now.t_ms,
                    freshness: Freshness::Current,
                    source_kind: BeliefSourceKind::Map,
                    ..BeliefMeta::default()
                },
                ..CognitiveServiceBelief::default()
            },
        );
        now = self.world_model.update(now, world_context);
        self.semantic_outcomes.observe_outcome(&now.world);
        now.extensions.insert(
            "self_model".to_string(),
            serde_json::to_value(&now.world.self_model)?,
        );
        now.extensions.insert(
            "temporal_context".to_string(),
            serde_json::to_value(&now.world.temporal)?,
        );
        now.extensions.insert(
            "social_world".to_string(),
            serde_json::to_value(&now.world.social)?,
        );
        now.extensions.insert(
            "epistemic_state".to_string(),
            serde_json::to_value(&now.world.epistemic)?,
        );
        now.extensions.insert(
            "semantic_graph".to_string(),
            serde_json::to_value(&now.world.semantic)?,
        );
        let sleep_input = runtime_sleep_input(
            &now,
            self.sleep_controller.expects_external_power(),
            enhanced_cognition_available,
        );
        let sleep_snapshot = self.sleep_controller.tick(sleep_input);
        let sleeping = self.sleep_controller.requires_quiescence();
        now.extensions
            .insert("sleep".to_string(), serde_json::to_value(&sleep_snapshot)?);
        let goal_cycle = if sleeping {
            self.goal_system.suspend_for_sleep(&now.world)
        } else {
            self.goal_system.tick(&now.world, &proposals)?
        };
        now.drives = goal_cycle.drives.legacy_sense();
        let goal_action = goal_cycle
            .behavior
            .as_ref()
            .map(|behavior| behavior.action.clone());
        let mut goal_skill_request = goal_cycle
            .behavior
            .as_ref()
            .and_then(|behavior| behavior.affordance.skill_request.clone());
        if goal_cycle
            .selection
            .selected_goal
            .as_ref()
            .is_some_and(|goal| goal.as_str() == "seek_charger")
        {
            if let (Some(request), Some(cue)) = (
                goal_skill_request.as_mut(),
                DockIrCue::from_character(now.body.infrared_character),
            ) {
                if matches!(
                    request.skill_id,
                    SkillId::TurnTowardTarget | SkillId::ApproachTarget | SkillId::AlignWithDock
                ) {
                    request.skill_id = SkillId::AlignWithDock;
                    request.bearing_rad = Some(cue.bearing_hint_rad());
                    request.progress_metric = "dock_ir_alignment".to_string();
                }
            }
        }
        now.extensions.insert(
            "goal_system".to_string(),
            serde_json::to_value(&goal_cycle)?,
        );
        let mut action_value_candidates =
            action_value_candidate_actions(&proposals, reign_action.as_ref(), &llm_tick);

        let conductor_memory =
            memory_for_navigation_with_map_trust(now.memory.clone(), corrected_map_trust);
        let mut baseline_action = self.conductor.choose(ConductorInput {
            latent: latent.clone(),
            drives: now.drives.clone(),
            memory: conductor_memory,
            predictions: now.predictions.clone(),
            surprise: now.surprise.clone(),
            llm: now.llm.clone(),
            safety: SafetySense::default(),
            reign: now.reign.clone(),
            range: now.range.clone(),
            body: now.body.clone(),
            charger_near_score: charger_signal_scores(&now).0,
            charger_visible_score: charger_signal_scores(&now).1,
            proposals: proposals.clone(),
        })?;
        let conductor_navigation_goal = Box::new(
            self.conductor
                .navigation_goal()
                .cloned()
                .unwrap_or_else(|| pete_conductor::NavigationGoalDecision {
                    intent: NavigationIntent::FollowProposal,
                    action: baseline_action.clone(),
                    confidence: 0.5,
                    reason:
                        "conductor selected an action without structured navigation diagnostics"
                            .to_string(),
                }),
        );
        now.extensions.insert(
            "conductor.navigation_goal".to_string(),
            serde_json::to_value(conductor_navigation_goal.as_ref())?,
        );
        if let Some(action) = mechanical_reign_action_for_selection.as_ref() {
            baseline_action = action.clone();
        }
        if recovery_candidate_context(&now) && is_recovery_locomotion_action(&baseline_action) {
            push_unique_action(&mut action_value_candidates, baseline_action.clone());
        }
        if memory_navigation_candidate_context(&now, &baseline_action) {
            push_unique_action(&mut action_value_candidates, baseline_action.clone());
        }

        let mut model_predictions = Vec::new();
        let mut hardcoded_predictions = Vec::new();
        let mut candidate_scores = Vec::new();
        for action in &action_value_candidates {
            let candidate_now = now_with_surface_anticipation(
                &now,
                surface_output_for_anticipation.as_ref(),
                action,
            );
            let candidate_danger_input =
                danger_behavior_input(&candidate_now, &latent, Some(action));
            let candidate_danger = self
                .models
                .behaviors
                .danger
                .infer(&candidate_danger_input, now.t_ms)?;
            let mut candidate_danger_record = candidate_danger.record;
            if let (Some(hard), Some(model)) = (
                candidate_danger_record.hardcoded_output.as_ref(),
                candidate_danger_record.model_output.as_ref(),
            ) {
                candidate_danger_record.disagreement = Some(danger_disagreement(hard, model));
            }
            let candidate_danger_output = candidate_danger_record
                .model_output
                .as_ref()
                .copied()
                .or(candidate_danger_record.selected_output.as_ref().copied());
            let candidate_danger_had_fallback = candidate_danger_record.model_output.is_none();
            behavior_runs.push(candidate_danger_record.erase());

            let candidate_charge_input = charge_behavior_input(&now, &latent, Some(action));
            let candidate_charge = self
                .models
                .behaviors
                .charge
                .infer(&candidate_charge_input, now.t_ms)?;
            let mut candidate_charge_record = candidate_charge.record;
            if let (Some(hard), Some(model)) = (
                candidate_charge_record.hardcoded_output.as_ref(),
                candidate_charge_record.model_output.as_ref(),
            ) {
                candidate_charge_record.disagreement = Some(charge_disagreement(hard, model));
            }
            let candidate_charge_output = candidate_charge_record
                .model_output
                .as_ref()
                .copied()
                .or(candidate_charge_record.selected_output.as_ref().copied());
            let candidate_charge_had_fallback = candidate_charge_record.model_output.is_none();
            behavior_runs.push(candidate_charge_record.erase());

            let candidate_action_value_input = action_value_behavior_input(
                &candidate_now,
                &latent,
                Some(action),
                candidate_danger_output,
                candidate_charge_output,
            );
            let action_value_run = self
                .models
                .behaviors
                .action_value
                .infer(&candidate_action_value_input, now.t_ms)?;
            let mut action_value_record = action_value_run.record;
            if let (Some(hard), Some(model)) = (
                action_value_record.hardcoded_output.as_ref(),
                action_value_record.model_output.as_ref(),
            ) {
                action_value_record.disagreement = Some(action_value_disagreement(hard, model));
            }
            if let Some(model) = action_value_record.model_output.as_ref() {
                model_predictions.push(action_value_prediction(action.clone(), *model));
            }
            if let Some(hardcoded) = action_value_record.hardcoded_output.as_ref() {
                hardcoded_predictions.push(action_value_prediction(action.clone(), *hardcoded));
            }
            let action_value_output = action_value_record
                .model_output
                .as_ref()
                .copied()
                .or(action_value_record.selected_output.as_ref().copied());
            let action_value_had_fallback = action_value_record.model_output.is_none();
            behavior_runs.push(action_value_record.erase());

            let mut candidate_score = score_action_candidate(
                &candidate_now,
                action,
                CandidateModelSignals {
                    danger: candidate_danger_output,
                    charge: candidate_charge_output,
                    action_value: action_value_output,
                },
                Some(&baseline_action),
            );
            candidate_score.fallback_used = candidate_score.fallback_used
                || candidate_danger_had_fallback
                || candidate_charge_had_fallback
                || action_value_had_fallback;
            candidate_scores.push(candidate_score);
        }
        now.predictions.action_values_model = model_predictions;
        now.predictions.action_values_hardcoded = hardcoded_predictions;

        let mut action_selection = select_action_from_scores(
            self.action_selector_mode,
            &now,
            baseline_action.clone(),
            candidate_scores,
        );
        action_selection.shadow_selected_goal = goal_cycle
            .selection
            .selected_goal
            .as_ref()
            .map(|goal| goal.as_str().to_string());
        action_selection.shadow_selected_behavior = goal_cycle
            .behavior
            .as_ref()
            .map(|behavior| behavior.behavior_id.clone());
        action_selection.shadow_goal_action = goal_action.clone();
        action_selection.shadow_diverged_from_baseline = goal_action
            .as_ref()
            .is_some_and(|action| action != &baseline_action);
        action_selection.goal_switched = goal_cycle.selection.switched;
        action_selection.goal_retained_by_commitment = goal_cycle.selection.retained_by_commitment;
        action_selection.goal_selection_reason = Some(goal_cycle.selection.reason.clone());
        if self.action_selector_mode == ActionSelectorMode::Goal {
            action_selection.selected_goal = action_selection.shadow_selected_goal.clone();
            action_selection.selected_behavior = goal_cycle
                .behavior
                .as_ref()
                .map(|behavior| behavior.behavior_id.clone());
            action_selection.selected_action = goal_action.clone();
            action_selection.selected_score = goal_cycle
                .selection
                .selected_goal
                .as_ref()
                .and_then(|selected| {
                    goal_cycle
                        .evaluations
                        .iter()
                        .find(|evaluation| &evaluation.goal_id == selected)
                })
                .map(|evaluation| evaluation.motivation.activation);
            action_selection.safety_overrode = false;
        }
        if let Some(action) = mechanical_reign_action_for_selection.as_ref() {
            action_selection.selected_action = Some(action.clone());
            action_selection.selected_score = None;
            action_selection.safety_overrode = false;
        } else if self.action_selector_mode != ActionSelectorMode::Goal {
            if let Some(action) = event_script_forced_action.as_ref() {
                action_selection.selected_action = Some(action.clone());
                action_selection.selected_score = None;
                action_selection.safety_overrode = false;
            }
        }
        for warning in &action_selection.fallback_warnings {
            notes.push(warning.clone());
        }
        now.extensions.insert(
            "action_selector".to_string(),
            serde_json::to_value(&action_selection)?,
        );
        let teacher_action = action_selection
            .selected_action
            .clone()
            .unwrap_or(baseline_action);
        let mut conductor_proposals = action_value_candidates.clone();
        conductor_proposals.push(teacher_action.clone());
        let conductor_behavior_input = ConductorInput {
            latent: latent.clone(),
            drives: now.drives.clone(),
            memory: now.memory.clone(),
            predictions: now.predictions.clone(),
            surprise: now.surprise.clone(),
            llm: now.llm.clone(),
            safety: SafetySense::default(),
            reign: now.reign.clone(),
            range: now.range.clone(),
            body: now.body.clone(),
            charger_near_score: charger_signal_scores(&now).0,
            charger_visible_score: charger_signal_scores(&now).1,
            proposals: conductor_proposals,
        };
        let teacher_source = if now.reign.active {
            TrainingSource::HumanReign
        } else {
            TrainingSource::HardcodedTeacher
        };
        let conductor_run = self.models.behaviors.conductor.infer_with_teacher_source(
            &conductor_behavior_input,
            now.t_ms,
            teacher_source,
        )?;
        let mut conductor_record = conductor_run.record;
        let conductor_controls = matches!(
            self.models.behaviors.conductor.regime,
            BehaviorRegime::ModelInfer | BehaviorRegime::ModelTrainAndInfer
        ) && self.action_selector_mode != ActionSelectorMode::Goal;
        let conductor_selected_action = if conductor_controls {
            conductor_run.chosen
        } else {
            teacher_action.clone()
        };
        conductor_record.selected_output = Some(conductor_selected_action.clone());
        if mechanical_reign_action_for_selection.is_some()
            && !matches!(conductor_record.selected_output, Some(ref action) if Some(action) == mechanical_reign_action_for_selection.as_ref())
        {
            conductor_record.selected_output = mechanical_reign_action_for_selection.clone();
        }
        let mut chosen_action = mechanical_reign_action_for_selection
            .clone()
            .or_else(|| {
                (self.action_selector_mode != ActionSelectorMode::Goal)
                    .then(|| event_script_forced_action.clone())
                    .flatten()
            })
            .unwrap_or(conductor_selected_action);
        if sleeping && mechanical_reign_action_for_selection.is_none() {
            chosen_action = ActionPrimitive::Stop;
        }
        let locomotion_input = self
            .locomotion_tracker
            .observe(now.t_ms, &now.body, &now.range);
        let locomotion_run = self
            .models
            .behaviors
            .locomotion
            .infer_with_disagreement(&locomotion_input, now.t_ms)?;
        let locomotion_output = locomotion_run.chosen.bounded(0.6, 1.0);
        let locomotion_applied = mechanical_reign_action_for_selection.is_none()
            && (event_script_forced_action.is_none()
                || self.action_selector_mode == ActionSelectorMode::Goal)
            && matches!(&chosen_action, ActionPrimitive::Explore { .. });
        if locomotion_applied {
            let duration_ms = match &chosen_action {
                ActionPrimitive::Explore { duration_ms, .. } => *duration_ms,
                _ => 1_000,
            };
            chosen_action = ActionPrimitive::Drive {
                forward: locomotion_output.forward_velocity_m_s,
                turn: locomotion_output.angular_velocity_rad_s,
                duration_ms,
            };
        }
        now.extensions.insert(
            "locomotion.nervous_system".to_string(),
            serde_json::json!({
                "schema_version": pete_neat::LOCOMOTION_SCHEMA_VERSION,
                "input": locomotion_input,
                "output": locomotion_output,
                "applied": locomotion_applied,
                "safety_authority": false,
            }),
        );
        behavior_runs.push(locomotion_run.record.erase());
        self.chirp_events
            .emit_post_selection_chirps(&mut now, &mut notes, &chosen_action)?;
        let mut map_memory_decision = map_memory_decision_debug(
            &now,
            &chosen_action,
            action_selection.baseline_action.as_ref(),
            mechanical_reign_action_for_selection.is_some() || event_script_forced_action.is_some(),
        );
        now = now_with_surface_anticipation(
            &now,
            surface_output_for_anticipation.as_ref(),
            &chosen_action,
        );
        let conductor_selected_output = conductor_record.selected_output.clone();
        behavior_runs.push(conductor_record.erase());

        if let Some(proposed) = llm_action_proposal.proposed_action.as_ref() {
            llm_action_proposal.accepted = proposed == &chosen_action;
            llm_action_proposal.final_action = Some(chosen_action.clone());
            if !llm_action_proposal.accepted && llm_action_proposal.ignored_reason.is_none() {
                llm_action_proposal.ignored_reason = if mechanical_reign_action_for_selection
                    .is_some()
                {
                    Some("safe active Reign command outranked LLM action".to_string())
                } else if mechanical_reign_action.is_some() && llm_has_safety_reason {
                    Some("LLM safety rationale competed with Reign but conductor selected another action".to_string())
                } else if event_script_forced_action.is_some() {
                    Some("event script action outranked LLM action".to_string())
                } else {
                    Some("conductor selected a different action".to_string())
                };
            }
        }

        let danger_input = danger_behavior_input(&now, &latent, Some(&chosen_action));
        let danger_run = self
            .models
            .behaviors
            .danger
            .infer(&danger_input, now.t_ms)?;
        let mut danger_record = danger_run.record;
        if let (Some(hard), Some(model)) = (
            danger_record.hardcoded_output.as_ref(),
            danger_record.model_output.as_ref(),
        ) {
            danger_record.disagreement = Some(danger_disagreement(hard, model));
        }
        if let Some(model) = danger_record.model_output.as_ref() {
            now.predictions.danger_model = Some(danger_prediction(*model));
        }
        if let Some(hardcoded) = danger_record.hardcoded_output.as_ref() {
            now.predictions.danger_hardcoded = Some(danger_prediction(*hardcoded));
        }
        behavior_runs.push(danger_record.erase());

        let charge_input = charge_behavior_input(&now, &latent, Some(&chosen_action));
        let charge_run = self
            .models
            .behaviors
            .charge
            .infer(&charge_input, now.t_ms)?;
        let mut charge_record = charge_run.record;
        if let (Some(hard), Some(model)) = (
            charge_record.hardcoded_output.as_ref(),
            charge_record.model_output.as_ref(),
        ) {
            charge_record.disagreement = Some(charge_disagreement(hard, model));
        }
        if let Some(model) = charge_record.model_output.as_ref() {
            now.predictions.charge_model = Some(charge_prediction(*model));
        }
        if let Some(hardcoded) = charge_record.hardcoded_output.as_ref() {
            now.predictions.charge_hardcoded = Some(charge_prediction(*hardcoded));
        }
        behavior_runs.push(charge_record.erase());

        let eye_next_input = eye_next_behavior_input(&now, &latent, Some(&chosen_action), 100);
        let eye_next_run = self
            .models
            .behaviors
            .eye_next
            .infer(&eye_next_input, now.t_ms)?;
        let mut eye_next_record = eye_next_run.record;
        if let (Some(hard), Some(model)) = (
            eye_next_record.hardcoded_output.as_ref(),
            eye_next_record.model_output.as_ref(),
        ) {
            eye_next_record.disagreement = Some(eye_next_disagreement(hard, model));
        }
        if let Some(model) = eye_next_record.model_output.as_ref() {
            now.predictions.eye_next_model = Some(eye_prediction(model));
        }
        if let Some(hardcoded) = eye_next_record.hardcoded_output.as_ref() {
            now.predictions.eye_next_hardcoded = Some(eye_prediction(hardcoded));
        }
        behavior_runs.push(eye_next_record.erase());

        let ear_next_input = ear_next_behavior_input(&now, &latent, Some(&chosen_action), 100);
        let ear_next_run = self
            .models
            .behaviors
            .ear_next
            .infer(&ear_next_input, now.t_ms)?;
        let mut ear_next_record = ear_next_run.record;
        if let (Some(hard), Some(model)) = (
            ear_next_record.hardcoded_output.as_ref(),
            ear_next_record.model_output.as_ref(),
        ) {
            ear_next_record.disagreement = Some(ear_next_disagreement(hard, model));
        }
        if let Some(model) = ear_next_record.model_output.as_ref() {
            now.predictions.ear_next_model = Some(ear_prediction(model));
        }
        if let Some(hardcoded) = ear_next_record.hardcoded_output.as_ref() {
            now.predictions.ear_next_hardcoded = Some(ear_prediction(hardcoded));
        }
        behavior_runs.push(ear_next_record.erase());

        let selected_goal_for_safety = (self.action_selector_mode == ActionSelectorMode::Goal
            && mechanical_reign_action_for_selection.is_none())
        .then(|| goal_cycle.selection.selected_goal.as_ref())
        .flatten()
        .map(|goal| goal.as_str());
        let desired_motor = action_to_motor_command(Some(&chosen_action));
        let safety = self.safety.filter_action(
            &now,
            selected_goal_for_safety,
            &chosen_action,
            desired_motor,
        );
        let control_provenance = if safety.vetoed {
            ControlProvenance::SafetyVeto
        } else if safety.reason == Some(SafetyReason::Contact) {
            ControlProvenance::AutonomicReflex
        } else if direct_reign_active {
            ControlProvenance::HumanDirect
        } else if now
            .reign
            .latest
            .as_ref()
            .is_some_and(|input| input.mode == ReignMode::Assist)
        {
            ControlProvenance::HumanAssist
        } else {
            ControlProvenance::Autonomous
        };
        let executed_goal_behavior = goal_cycle.behavior.as_ref().filter(|behavior| {
            self.action_selector_mode == ActionSelectorMode::Goal
                && !sleeping
                && !locomotion_applied
                && control_provenance == ControlProvenance::Autonomous
                && behavior.action == chosen_action
                && !safety.vetoed
                && safety.reason.is_none()
                && safety.command == desired_motor
        });
        self.last_active_control = Some(ActiveControlSummary {
            goal_id: executed_goal_behavior.map(|behavior| behavior.goal_id.as_str().to_string()),
            behavior_id: executed_goal_behavior.map(|behavior| behavior.behavior_id.clone()),
            action_kind: Some(format!("{chosen_action:?}")),
            provenance: control_provenance,
            safety_preempted: safety.reason.is_some(),
            veto_reasons: safety
                .reason
                .as_ref()
                .map(|reason| vec![describe_safety_reason(Some(reason.clone())).to_string()])
                .unwrap_or_default(),
            unable_to_act_reason: safety
                .vetoed
                .then(|| describe_safety_reason(safety.reason.clone()).to_string()),
            ..ActiveControlSummary::default()
        });
        action_selection.safety_overrode = safety.vetoed;
        now.extensions.insert(
            "action_selector".to_string(),
            serde_json::to_value(&action_selection)?,
        );
        now.extensions.insert(
            "goal_system.outcome".to_string(),
            serde_json::json!({
                "schema_version": 1,
                "world_revision": goal_cycle.world.revision,
                "selected_goal": goal_cycle.selection.selected_goal.clone(),
                "selected_behavior": goal_cycle.behavior.as_ref().map(|behavior| &behavior.behavior_id),
                "executed_goal_behavior": executed_goal_behavior.map(|behavior| &behavior.behavior_id),
                "selected_primitive": chosen_action.clone(),
                "safety": {
                    "vetoed": safety.vetoed,
                    "reason": safety
                        .reason
                        .clone()
                        .map(|reason| describe_safety_reason(Some(reason))),
                    "final_motor": safety.command,
                },
                "shadow_diverged_from_baseline": action_selection.shadow_diverged_from_baseline,
            }),
        );
        self.locomotion_tracker.observe_command(LocomotionOutput {
            forward_velocity_m_s: safety.command.forward,
            angular_velocity_rad_s: safety.command.turn,
            recovery_activation: locomotion_output.recovery_activation,
        });
        map_memory_decision.safety_overrode = safety.vetoed;
        self.nudge.observe_motor(safety.command);
        now.extensions.insert(
            "motor_gate".to_string(),
            serde_json::json!({
                "desired_motor": desired_motor,
                "final_motor": safety.command,
                "motor_applied": !is_near_zero_motor(safety.command),
                "vetoed": safety.vetoed,
                "safety_reason": safety.reason.clone().map(Some).map(describe_safety_reason),
            }),
        );
        now.extensions.insert(
            "action.motion_bridge".to_string(),
            serde_json::json!({
                "llm_action": llm_action_proposal.proposed_action.clone(),
                "llm_advisory_action": llm_action_proposal.advisory_action.clone(),
                "selected_action": action_selection.selected_action.clone(),
                "conductor_selected_action": conductor_selected_output.clone(),
                "conductor_navigation_goal": conductor_navigation_goal.as_ref(),
                "chosen_action": chosen_action.clone(),
                "map_memory_decision": map_memory_decision.clone(),
                "desired_motor": action_to_motor_command(Some(&chosen_action)),
                "final_motor": safety.command,
                "safety_override": safety.vetoed,
                "safety_reason": safety.reason.clone().map(Some).map(describe_safety_reason),
            }),
        );
        if map_memory_decision.influenced {
            notes.push(format!(
                "MapMemoryDecision reason={:?} action={:?} danger={:.2} charge={:.2} novelty={:.2}",
                map_memory_decision.reason,
                map_memory_decision.selected_action,
                map_memory_decision.place_danger,
                map_memory_decision.place_charge_value,
                map_memory_decision.place_novelty
            ));
        }
        notes.push(format!(
            "ActionMotorBridge llm_action={:?} llm_advisory_action={:?} selected_action={:?} conductor_selected_action={:?} chosen_action={:?} desired_motor={:?} final_motor={:?} safety_override={}",
            llm_action_proposal.proposed_action,
            llm_action_proposal.advisory_action,
            action_selection.selected_action,
            conductor_selected_output,
            chosen_action,
            action_to_motor_command(Some(&chosen_action)),
            safety.command,
            safety.vetoed
        ));
        now.extensions.insert(
            "prod.nudge".to_string(),
            serde_json::to_value(&self.nudge.status)?,
        );
        if safety.vetoed {
            if llm_action_proposal.accepted {
                let reason = describe_safety_reason(safety.reason.clone()).to_string();
                llm_action_proposal.safety_vetoed = true;
                llm_action_proposal.safety_reason = Some(reason.clone());
                notes.push(format!("LlmActionProposalSafetyVeto: {reason}"));
                teachings.push(pete_llm::LlmTeaching {
                    t_ms: now.t_ms,
                    summary: format!("Safety vetoed LLM action {:?}", chosen_action),
                    critique: Some(format!("LLM proposed an unsafe action: {reason}")),
                    counterfactuals: Vec::new(),
                    memory_notes: vec![format!(
                        "Avoid repeating LLM action {:?} when safety reports {reason}",
                        chosen_action
                    )],
                    confidence: now.llm.confidence,
                });
            }
            now.extensions
                .insert("safety.vetoed".to_string(), serde_json::Value::Bool(true));
            let veto_ctx = EventContext {
                now: &now,
                latent: Some(&latent),
                recall: Some(&recall),
                predicted_futures: &futures,
                llm: Some(&now.llm),
                safety: Some(&safety),
            };
            let veto_events = self
                .extractor
                .events_from_safety(&now, &chosen_action, &safety);
            let veto_output = self.bus.dispatch_all(&veto_ctx, veto_events)?;
            apply_responses(
                &mut now,
                veto_output.responses,
                &mut sensations,
                &mut impressions,
                &mut experiences,
                &mut teachings,
                &mut notes,
                &mut drive_impulses,
            );
            self.goal_system
                .add_drive_impulses(std::mem::take(&mut drive_impulses));
            notes.push(format!(
                "Safety vetoed {:?}: {}",
                chosen_action,
                describe_safety_reason(safety.reason.clone())
            ));
        }
        now.extensions.insert(
            "llm.action_proposal".to_string(),
            serde_json::to_value(&llm_action_proposal)?,
        );

        attach_structured_predictions_to_experience(
            &mut embodied_experience,
            &futures,
            &now,
            Some(&chosen_action),
        );
        experiences.push(embodied_experience);

        let reign_outcome = reign_input.as_ref().map(|input| {
            let accepted_by_conductor = reign_action
                .as_ref()
                .map(|action| action == &chosen_action)
                .unwrap_or(false);
            ReignOutcome {
                input_id: input.id,
                accepted_by_conductor,
                vetoed_by_safety: safety.vetoed,
                final_action: Some(chosen_action.clone()),
                reason: if safety.vetoed {
                    Some(describe_safety_reason(safety.reason.clone()).to_string())
                } else if accepted_by_conductor {
                    None
                } else {
                    Some("conductor chose another action".to_string())
                },
            }
        });

        if experiences.is_empty() {
            experiences.push(Experience::new(
                "realtime.state",
                format!(
                    "I am at t={}ms with battery {:.2}.",
                    now.t_ms, now.body.battery_level
                ),
                Vec::new(),
                Vec::new(),
                now.t_ms,
                now.t_ms,
            ));
        }

        self.last_behavior_runs = behavior_runs.clone();
        let mut frame = ExperienceFrame {
            id: frame_id,
            t_ms: now.t_ms,
            now: now.clone(),
            sensations,
            impressions,
            experiences: experiences.clone(),
            z: Some(latent.clone()),
            chosen_action: Some(chosen_action.clone()),
            conscious_command: llm_tick.conscious_command.clone(),
            reign_input,
            reign_outcome,
            predicted_futures: futures.clone(),
            behavior_runs,
            actual_next: None,
            reward: Reward::default(),
            surprise: now.surprise.clone(),
            memory_recall: recall.hits.clone(),
            recollections: recall.recollections.clone(),
            llm_teaching: teachings.clone(),
            counterfactuals: teachings
                .iter()
                .flat_map(|teaching| teaching.counterfactuals.clone())
                .collect(),
            notes,
        };
        attach_memory_links_to_frame(&mut frame);

        self.ledger.append(&frame).await?;
        self.memory_recall.observe_frame(&frame).await?;
        let surprise_computer = self.surprise_computer.clone();
        let reward_computer = self.reward_computer.clone();
        let mut inline_learning = InlineLearningTickStatus {
            enabled: self.inline_learning.is_enabled(),
            mode: self.inline_learning.mode,
            samples_observed: 0,
            train_steps_used: 0,
        };
        if let Some(transition) = self.transition_builder.observe(
            PendingFrame {
                frame_id: frame.id,
                now: frame.now.clone(),
                z: latent,
                action: frame.chosen_action.clone(),
                predicted_futures: frame.predicted_futures.clone(),
            },
            |previous, current| {
                let surprise = surprise_computer.compute(
                    &previous.predicted_futures,
                    &current.z,
                    &current.now,
                );
                reward_computer.compute(
                    &previous.now,
                    previous.action.as_ref(),
                    &current.now,
                    &surprise,
                )
            },
            |previous, current| {
                surprise_computer.compute(&previous.predicted_futures, &current.z, &current.now)
            },
        ) {
            self.ledger.append_transition(&transition).await?;
            self.memory_recall.observe_transition(&transition).await?;
            inline_learning = self.observe_inline_learning(&transition)?;
        }
        self.memory_store.store(&frame).await?;

        self.semantic_outcomes
            .remember(&now.world, executed_goal_behavior);
        Ok(RuntimeTick {
            frame,
            experience: experiences.last().cloned().unwrap_or_else(|| {
                Experience::new(
                    "realtime.state",
                    "I am active.",
                    Vec::new(),
                    Vec::new(),
                    now.t_ms,
                    now.t_ms,
                )
            }),
            chosen_action: Some(chosen_action),
            skill_request: (self.action_selector_mode == ActionSelectorMode::Goal
                && mechanical_reign_action_for_selection.is_none()
                && !sleeping)
                .then_some(goal_skill_request)
                .flatten(),
            skill_status: None,
            recall,
            llm: llm_tick,
            combobulation,
            inline_learning,
        })
    }

    fn run_event_scripts(
        &mut self,
        now: &mut Now,
        notes: &mut Vec<String>,
        proposed_actions: &mut Vec<ActionPrimitive>,
    ) -> Result<(Option<ActionPrimitive>, Vec<ErasedBehaviorRunRecord>)> {
        let mut behavior_runs = Vec::new();
        let forced_action = None;
        let mut safe_sequences = serde_json::Map::new();

        if let Some(input) = robot_initialized_event_input(now) {
            let run = self
                .models
                .behaviors
                .event_robot_initialized
                .infer_with_teacher_source(&input, now.t_ms, TrainingSource::HardcodedTeacher)?;
            let sequence = safety_trace_script_actions(&mut self.safety, now, &run.chosen);
            for action in run
                .chosen
                .actions
                .iter()
                .filter_map(script_action_to_primitive)
            {
                proposed_actions.push(action);
            }
            safe_sequences.insert(
                "robot-initialized".to_string(),
                serde_json::to_value(&sequence)?,
            );
            notes.push("EventScript:on(robot-initialized) emitted bring-up sequence".to_string());
            behavior_runs.push(run.record.erase());
        }

        if now.body.flags.bump_left || now.body.flags.bump_right {
            let input = BumpEventInput {
                t_ms: now.t_ms,
                bump_left: now.body.flags.bump_left,
                bump_right: now.body.flags.bump_right,
            };
            let run = self.models.behaviors.event_bump.infer_with_teacher_source(
                &input,
                now.t_ms,
                TrainingSource::HardcodedTeacher,
            )?;
            let sequence = safety_trace_script_actions(&mut self.safety, now, &run.chosen);
            if let Some(first) = first_motor_script_action(&run.chosen) {
                proposed_actions.push(first);
            }
            safe_sequences.insert("bump".to_string(), serde_json::to_value(&sequence)?);
            notes.push(
                "EventScript:on(bump) emitted random lament -> Stop -> Rotate(180) -> Go"
                    .to_string(),
            );
            let mut record = run.record;
            record.selected_output = Some(EventScriptOutput {
                actions: sequence
                    .actions
                    .iter()
                    .map(|action| action.requested.clone())
                    .collect(),
            });
            record.confidence = Some(if sequence.actions.iter().any(|action| action.vetoed) {
                0.5
            } else {
                1.0
            });
            behavior_runs.push(record.erase());
        }

        if !safe_sequences.is_empty() {
            now.extensions.insert(
                "event_scripts".to_string(),
                serde_json::Value::Object(safe_sequences),
            );
            notes.push("EventScript: safety filtered every emitted action".to_string());
        }

        Ok((forced_action, behavior_runs))
    }

    fn observe_inline_learning(
        &mut self,
        transition: &ExperienceTransition,
    ) -> Result<InlineLearningTickStatus> {
        let mut status = InlineLearningTickStatus {
            enabled: self.inline_learning.is_enabled(),
            mode: self.inline_learning.mode,
            samples_observed: 0,
            train_steps_used: 0,
        };
        if !self.inline_learning.is_enabled() {
            return Ok(status);
        }
        if self.inline_learning.mode != InlineLearningMode::WorldOutcome {
            return Ok(status);
        }

        let mut remaining = self.inline_learning.max_train_steps_per_tick;
        if self.inline_learning.behaviors.danger && remaining > 0 {
            if let Some(sample) = self
                .behavior_training_hub
                .danger_extractor
                .extract(transition)?
            {
                let sample = TrainingSample {
                    input: SituatedDangerInput {
                        input: sample.input,
                        now: transition.before.clone(),
                    },
                    expected: sample.expected,
                    actual: sample.actual,
                    reward: sample.reward,
                    weight: sample.weight,
                    source: sample.source,
                    t_ms: sample.t_ms,
                };
                self.models.behaviors.danger.observe(&sample)?;
                remaining = remaining.saturating_sub(1);
                status.samples_observed = status.samples_observed.saturating_add(1);
                status.train_steps_used = status.train_steps_used.saturating_add(1);
            }
        }
        if self.inline_learning.behaviors.charge && remaining > 0 {
            if let Some(sample) = self
                .behavior_training_hub
                .charge_extractor
                .extract(transition)?
            {
                let sample = TrainingSample {
                    input: SituatedChargeInput {
                        input: sample.input,
                        now: transition.before.clone(),
                    },
                    expected: sample.expected,
                    actual: sample.actual,
                    reward: sample.reward,
                    weight: sample.weight,
                    source: sample.source,
                    t_ms: sample.t_ms,
                };
                self.models.behaviors.charge.observe(&sample)?;
                remaining = remaining.saturating_sub(1);
                status.samples_observed = status.samples_observed.saturating_add(1);
                status.train_steps_used = status.train_steps_used.saturating_add(1);
            }
        }
        if self.inline_learning.behaviors.future && remaining > 0 {
            if let Some(sample) = self
                .behavior_training_hub
                .future_extractor
                .extract(transition)?
            {
                self.models.behaviors.future.observe(&sample)?;
                remaining = remaining.saturating_sub(1);
                status.samples_observed = status.samples_observed.saturating_add(1);
                status.train_steps_used = status.train_steps_used.saturating_add(1);
            }
        }
        if self.inline_learning.behaviors.action_value && remaining > 0 {
            if let Some(sample) = self
                .behavior_training_hub
                .action_value_extractor
                .extract(transition)?
            {
                let sample = TrainingSample {
                    input: SituatedActionValueInput {
                        input: sample.input,
                        now: transition.before.clone(),
                    },
                    expected: sample.expected,
                    actual: sample.actual,
                    reward: sample.reward,
                    weight: sample.weight,
                    source: sample.source,
                    t_ms: sample.t_ms,
                };
                self.models.behaviors.action_value.observe(&sample)?;
                remaining = remaining.saturating_sub(1);
                status.samples_observed = status.samples_observed.saturating_add(1);
                status.train_steps_used = status.train_steps_used.saturating_add(1);
            }
        }
        if self.inline_learning.behaviors.eye_next && remaining > 0 {
            if let Some(sample) = self
                .behavior_training_hub
                .eye_next_extractor
                .extract(transition)?
            {
                let sample = TrainingSample {
                    input: SituatedEyeNextInput {
                        input: sample.input,
                        now: transition.before.clone(),
                    },
                    expected: sample.expected,
                    actual: sample.actual,
                    reward: sample.reward,
                    weight: sample.weight,
                    source: sample.source,
                    t_ms: sample.t_ms,
                };
                self.models.behaviors.eye_next.observe(&sample)?;
                remaining = remaining.saturating_sub(1);
                status.samples_observed = status.samples_observed.saturating_add(1);
                status.train_steps_used = status.train_steps_used.saturating_add(1);
            }
        }
        if self.inline_learning.behaviors.ear_next && remaining > 0 {
            if let Some(sample) = self
                .behavior_training_hub
                .ear_next_extractor
                .extract(transition)?
            {
                let sample = TrainingSample {
                    input: SituatedEarNextInput {
                        input: sample.input,
                        now: transition.before.clone(),
                    },
                    expected: sample.expected,
                    actual: sample.actual,
                    reward: sample.reward,
                    weight: sample.weight,
                    source: sample.source,
                    t_ms: sample.t_ms,
                };
                self.models.behaviors.ear_next.observe(&sample)?;
                remaining = remaining.saturating_sub(1);
                status.samples_observed = status.samples_observed.saturating_add(1);
                status.train_steps_used = status.train_steps_used.saturating_add(1);
            }
        }
        if self.inline_learning.behaviors.experience && remaining > 0 {
            let input = ExperienceBehaviorInput::from_now(&transition.before);
            let sample = TrainingSample {
                input,
                expected: ExperienceBehaviorOutput {
                    latent: transition.before_z.clone(),
                    reconstruction: None,
                    reconstruction_loss: None,
                    confidence: transition.before_z.confidence,
                },
                actual: None,
                reward: Some(transition.reward.value),
                weight: 1.0,
                source: TrainingSource::WorldOutcome,
                t_ms: transition.created_at_ms,
            };
            self.models.behaviors.experience.observe(&sample)?;
            status.samples_observed = status.samples_observed.saturating_add(1);
            status.train_steps_used = status.train_steps_used.saturating_add(1);
        }

        Ok(status)
    }
}

fn danger_prediction(output: DangerOutput) -> DangerPrediction {
    DangerPrediction {
        bump_risk: output.bump_risk,
        cliff_risk: output.cliff_risk,
        wheel_drop_risk: output.wheel_drop_risk,
        stuck_risk: output.stuck_risk,
        confidence: output.confidence,
    }
}

fn charge_prediction(output: ChargeOutput) -> ChargePrediction {
    ChargePrediction {
        charge_probability: output.charge_probability,
        expected_battery_delta: output.expected_battery_delta,
        dock_likelihood: output.dock_likelihood,
        confidence: output.confidence,
    }
}

fn stuck_phase_label(code: f64) -> &'static str {
    match code.round() as i32 {
        1 => "stop",
        2 => "reverse",
        3 => "turn-away",
        _ => "none",
    }
}

fn action_value_prediction(
    action: ActionPrimitive,
    output: ActionValueOutput,
) -> ActionValuePrediction {
    ActionValuePrediction {
        action,
        value: output.value,
        confidence: output.confidence,
    }
}

fn eye_prediction(output: &EyeNextOutput) -> EyePrediction {
    EyePrediction {
        width: output.width,
        height: output.height,
        rgb: output.rgb.clone(),
        confidence: output.confidence,
    }
}

fn ear_prediction(output: &EarNextOutput) -> EarPrediction {
    EarPrediction {
        sample_rate_hz: output.sample_rate_hz,
        channels: output.channels,
        pcm: output.pcm.clone(),
        features: output.features.clone(),
        confidence: output.confidence,
    }
}

fn robot_initialized_event_input(now: &Now) -> Option<RobotInitializedEventInput> {
    let init = now.extensions.get("robot.initialization")?;
    Some(RobotInitializedEventInput {
        t_ms: now.t_ms,
        mode: init
            .get("mode")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown")
            .to_string(),
        body: init
            .get("body")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown body")
            .to_string(),
        battery_percent: init
            .get("battery_percent")
            .and_then(|value| value.as_u64())
            .and_then(|value| u32::try_from(value).ok()),
        charging: init.get("charging").and_then(|value| value.as_bool()),
        active_sensors: init
            .get("active_sensors")
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as usize,
        requested_sensors: init
            .get("requested_sensors")
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as usize,
        ledger: init
            .get("ledger")
            .and_then(|value| value.as_str())
            .unwrap_or("unknown ledger")
            .to_string(),
        tick_ms: init
            .get("tick_ms")
            .and_then(|value| value.as_u64())
            .unwrap_or(0),
        dashboard: init
            .get("dashboard")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        capture: init
            .get("capture")
            .and_then(|value| value.as_str())
            .map(str::to_string),
    })
}

fn safety_trace_script_actions<S>(
    safety: &mut S,
    now: &Now,
    output: &EventScriptOutput,
) -> SafeScriptSequence
where
    S: SafetyLayer,
{
    SafeScriptSequence {
        actions: output
            .actions
            .iter()
            .map(|requested| {
                let action = script_action_to_primitive(requested);
                let desired_motor = action_to_motor_command(action.as_ref());
                let decision = safety.filter(now, desired_motor);
                SafeScriptAction {
                    requested: requested.clone(),
                    action,
                    desired_motor,
                    final_motor: decision.command,
                    vetoed: decision.vetoed,
                    safety_reason: decision
                        .reason
                        .map(|reason| describe_safety_reason(Some(reason)).to_string()),
                }
            })
            .collect(),
    }
}

fn first_motor_script_action(output: &EventScriptOutput) -> Option<ActionPrimitive> {
    output
        .actions
        .iter()
        .filter_map(script_action_to_primitive)
        .find(|action| {
            !matches!(
                action,
                ActionPrimitive::Speak { .. } | ActionPrimitive::Chirp { .. }
            )
        })
}

fn script_action_to_primitive(action: &EventScriptAction) -> Option<ActionPrimitive> {
    match action {
        EventScriptAction::Say { text } => Some(ActionPrimitive::Speak { text: text.clone() }),
        EventScriptAction::Chirp { pattern } => Some(ActionPrimitive::Chirp {
            pattern: pattern.clone(),
        }),
        EventScriptAction::Song { .. } => None,
        EventScriptAction::Stop => Some(ActionPrimitive::Stop),
        EventScriptAction::Rotate { deg } => Some(ActionPrimitive::Turn {
            direction: if *deg >= 0 {
                TurnDir::Left
            } else {
                TurnDir::Right
            },
            intensity: 0.5,
            duration_ms: ((*deg as i32).unsigned_abs() as u64 * 10).max(500),
        }),
        EventScriptAction::Go => Some(ActionPrimitive::Go {
            intensity: 0.15,
            duration_ms: 500,
        }),
    }
}

fn append_event_script_chirp(
    now: &mut Now,
    notes: &mut Vec<String>,
    event_name: &str,
    pattern: ChirpPattern,
) -> Result<()> {
    let sequence = SafeScriptSequence {
        actions: vec![SafeScriptAction {
            requested: EventScriptAction::Chirp {
                pattern: pattern.clone(),
            },
            action: Some(ActionPrimitive::Chirp {
                pattern: pattern.clone(),
            }),
            desired_motor: MotorCommand::stop(),
            final_motor: MotorCommand::stop(),
            vetoed: false,
            safety_reason: None,
        }],
    };
    let event_scripts = now
        .extensions
        .entry("event_scripts".to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if !event_scripts.is_object() {
        *event_scripts = serde_json::Value::Object(serde_json::Map::new());
    }
    if let Some(object) = event_scripts.as_object_mut() {
        object.insert(event_name.to_string(), serde_json::to_value(sequence)?);
    }
    notes.push(format!(
        "EventScript:on({event_name}) emitted chirp({pattern:?})"
    ));
    Ok(())
}

fn charger_visible(now: &Now) -> bool {
    now.objects.observations.iter().any(|observation| {
        observation.class == ObjectClass::Charger && observation.confidence >= 0.4
    }) || charger_signal_scores(now).1 >= 0.5
}

fn execute_event_script_typescript<I>(script: &str, input: &I) -> Result<EventScriptOutput>
where
    I: Serialize,
{
    let input_json =
        serde_json::to_string(input).context("failed to serialize event script input")?;
    let random_value = rand::random::<f64>();
    let source = format!(
        r#"
const input = {input_json};
const __peteRandom = {random_value};
function random() {{
  return __peteRandom;
}}
function say(text) {{
  return {{ type: "say", text: String(text) }};
}}
function chirp(pattern) {{
  return {{ type: "chirp", pattern: String(pattern) }};
}}
function song(name) {{
  return {{ type: "song", name: String(name) }};
}}
function stop() {{
  return {{ type: "stop" }};
}}
function rotate(deg) {{
  return {{ type: "rotate", deg: Number(deg) }};
}}
function go() {{
  return {{ type: "go" }};
}}
{script}
"#
    );
    let mut interp = Interpreter::new();
    interp
        .prepare(
            &source,
            Some(tsrun::ModulePath::new("/pete-event-script.ts")),
        )
        .map_err(tsrun_error)?;
    let value = loop {
        match interp.step().map_err(tsrun_error)? {
            StepResult::Continue => continue,
            StepResult::Complete(value) => break value,
            StepResult::NeedImports(imports) => {
                let names = imports
                    .iter()
                    .map(|request| request.specifier.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                anyhow::bail!("unsupported TypeScript import(s): {names}");
            }
            StepResult::Suspended { .. } => {
                anyhow::bail!(
                    "TypeScript event script suspended; async host commands are not enabled"
                )
            }
            StepResult::Done => return Ok(EventScriptOutput::default()),
        }
    };
    let value = js_value_to_json(value.value()).map_err(tsrun_error)?;
    if value.get("actions").is_some() {
        serde_json::from_value(value).context("failed to parse TypeScript event script output")
    } else {
        let actions = serde_json::from_value(value)
            .context("failed to parse TypeScript event script action list")?;
        Ok(EventScriptOutput { actions })
    }
}

fn tsrun_error(err: JsError) -> anyhow::Error {
    anyhow::anyhow!("TypeScript event script failed: {err}")
}

pub struct RuntimeTick {
    pub frame: ExperienceFrame,
    pub experience: Experience,
    pub chosen_action: Option<ActionPrimitive>,
    pub skill_request: Option<SkillRequest>,
    pub skill_status: Option<SkillStatus>,
    pub recall: RecallBundle,
    pub llm: LlmTickResult,
    pub combobulation: Option<Combobulation>,
    pub inline_learning: InlineLearningTickStatus,
}

#[async_trait::async_trait]
pub trait RuntimeLoop {
    async fn tick(
        &mut self,
        now: Now,
        latent: ExperienceLatent,
        futures: Vec<FuturePrediction>,
    ) -> Result<RuntimeTick>;

    fn reign_sense(&self, _now_ms: TimeMs) -> Result<ReignSense> {
        Ok(ReignSense::default())
    }

    fn observe_skill_status(&mut self, _status: &SkillStatus) {}
}

#[async_trait::async_trait]
impl<L, M, R, C, S, A> RuntimeLoop for MinimalRuntime<L, M, R, C, S, A>
where
    L: LedgerWriter + Sync + Send,
    M: MemoryStore + Send,
    R: Recall + Send + Sync,
    C: Conductor + Send,
    S: SafetyLayer + Send,
    A: LlmAgent + Send + 'static,
{
    fn reign_sense(&self, now_ms: TimeMs) -> Result<ReignSense> {
        let mut reign_queue = self
            .reign_queue
            .lock()
            .map_err(|_| anyhow::anyhow!("reign queue lock poisoned"))?;
        reign_queue.drain_expired(now_ms);
        Ok(reign_queue.sense(now_ms))
    }

    fn observe_skill_status(&mut self, status: &SkillStatus) {
        self.goal_system.observe_skill_status(status);
    }

    async fn tick(
        &mut self,
        now: Now,
        latent: ExperienceLatent,
        futures: Vec<FuturePrediction>,
    ) -> Result<RuntimeTick> {
        MinimalRuntime::tick(self, now, latent, futures).await
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RobotMode {
    ReadOnly,
    Slow,
    Disabled,
}

pub struct RealRobotRunner<R, C> {
    pub mode: RobotMode,
    pub cockpit: SafeCockpit<C>,
    pub sensors: Vec<Box<dyn SenseProducer + Send>>,
    pub runtime: R,
    pub tick_ms: u64,
    pub tick_count: usize,
    /// Allow executive-selected motion to reach real slow hardware. Direct
    /// WebRemote/Gamepad commands remain available regardless of this gate.
    pub autonomous_motion: bool,
    now_builder: NowBuilder,
    frame_processor: FrameProcessor,
    live_image_cognition: LiveImageCognition,
    robot_initialization: Option<serde_json::Value>,
    brainstem_interface: Option<serde_json::Value>,
    possession_recovery: PossessionRecoveryState,
    motion_rejection: MotionRejectionState,
    possessor_skills: PossessorSkillRuntime,
    sensor_poll_health: Vec<SensorPollHealth>,
}

#[derive(Clone, Debug, Default)]
struct SensorPollHealth {
    name: String,
    available: bool,
    consecutive_failures: u32,
    last_error: Option<String>,
    last_report_ms: TimeMs,
    last_success_ms: TimeMs,
}

const SENSOR_FAILURE_REPORT_INTERVAL_MS: TimeMs = 30_000;

struct PossessorSkillRuntime {
    lua: Option<LuaSkillRuntime>,
    load_error: Option<String>,
    driver: EmbodiedLuaDriverState,
    status: Option<SkillStatus>,
    provenance: Option<serde_json::Value>,
    last_reload_check_ms: TimeMs,
}

impl Default for PossessorSkillRuntime {
    fn default() -> Self {
        let mut config = LuaSkillConfig::default();
        config.directory = std::env::var_os("PETE_MOTHERBRAIN_SKILL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../skills/motherbrain")
            });
        match LuaSkillRuntime::load(config) {
            Ok(lua) => Self {
                lua: Some(lua),
                load_error: None,
                driver: EmbodiedLuaDriverState::default(),
                status: None,
                provenance: None,
                last_reload_check_ms: 0,
            },
            Err(error) => Self {
                lua: None,
                load_error: Some(error.to_string()),
                driver: EmbodiedLuaDriverState::default(),
                status: None,
                provenance: None,
                last_reload_check_ms: 0,
            },
        }
    }
}

#[derive(Clone, Debug, Default)]
struct LuaDriverOperationState {
    start_x_m: f32,
    start_y_m: f32,
    start_heading_rad: f32,
    last_dispatch_ms: TimeMs,
}

#[derive(Clone, Debug, Default)]
struct EmbodiedLuaDriverState {
    operations: std::collections::HashMap<u64, LuaDriverOperationState>,
}

struct RealLuaOrganDriver<'a, C> {
    cockpit: &'a mut C,
    request: &'a SkillRequest,
    status: &'a StatusSummary,
    home_base_contact: bool,
    state: &'a mut EmbodiedLuaDriverState,
    command_sent: bool,
}

impl<C: Cockpit> OrganDriver for RealLuaOrganDriver<'_, C> {
    fn poll(
        &mut self,
        operation: &HostOperation,
        context: OperationContext,
        now: &Now,
        _events: &pete_cockpit::EventBatch,
    ) -> OrganPoll {
        let state = self
            .state
            .operations
            .entry(context.operation_id)
            .or_insert_with(|| LuaDriverOperationState {
                start_x_m: now.body.odometry.x_m,
                start_y_m: now.body.odometry.y_m,
                start_heading_rad: now.body.odometry.heading_rad,
                last_dispatch_ms: 0,
            });
        let dispatch_due = state.last_dispatch_ms == 0
            || context.now_ms.saturating_sub(state.last_dispatch_ms) >= 100;
        let mut primitive = None;
        let outcome = match operation {
            HostOperation::Stop => {
                if context.first_poll {
                    if let Err(error) = self.cockpit.stop() {
                        return OrganPoll::Failed(cockpit_skill_failure(operation, error));
                    }
                    self.command_sent = true;
                }
                OrganPoll::Completed(json!({"stopped": true}))
            }
            HostOperation::FaceBearing { bearing_rad } => {
                let requested_bearing = self.request.bearing_rad.unwrap_or(*bearing_rad);
                let turned = angle_delta(now.body.odometry.heading_rad, state.start_heading_rad);
                let remaining = if self.request.target.is_some() {
                    requested_bearing
                } else {
                    angle_delta(requested_bearing, turned)
                };
                if remaining.abs() <= 0.10 {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"bearing_error": remaining, "turned_rad": turned}),
                        &mut self.command_sent,
                    )
                } else if dispatch_due {
                    let available_s = self
                        .request
                        .maximum_duration_ms
                        .saturating_sub(100)
                        .max(100) as f32
                        / 1_000.0;
                    let angular_rad_s = (remaining.abs() / available_s * 1.25).clamp(0.25, 2.5)
                        * remaining.signum();
                    let angular = radians_to_mrad(angular_rad_s);
                    match self.cockpit.cmd_vel(0, angular, context.primitive_ttl_ms) {
                        Ok(()) => {
                            self.command_sent = true;
                            primitive = Some(primitive_intent(
                                context,
                                operation,
                                json!({"linear_mm_s": 0, "angular_mrad_s": angular, "ttl_ms": context.primitive_ttl_ms, "remaining_rad": remaining}),
                            ));
                            OrganPoll::Pending {
                                progress: Some(("bearing_error".into(), remaining.abs())),
                                primitive: None,
                            }
                        }
                        Err(error) => OrganPoll::Failed(cockpit_skill_failure(operation, error)),
                    }
                } else {
                    OrganPoll::Pending {
                        progress: Some(("bearing_error".into(), remaining.abs())),
                        primitive: None,
                    }
                }
            }
            HostOperation::FollowBearing {
                bearing_rad,
                linear_m_s,
            } => {
                let bearing = self.request.bearing_rad.unwrap_or(*bearing_rad);
                if bearing.abs() <= 0.10 {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"bearing_error": bearing}),
                        &mut self.command_sent,
                    )
                } else if dispatch_due {
                    with_operation_progress(
                        dispatch_velocity(
                            self.cockpit,
                            operation,
                            context,
                            (*linear_m_s * 1_000.0).round() as i16,
                            radians_to_mrad(bearing).clamp(-500, 500),
                            &mut self.command_sent,
                            &mut primitive,
                        ),
                        Some(("bearing_error".into(), bearing.abs())),
                    )
                } else {
                    OrganPoll::Pending {
                        progress: Some(("bearing_error".into(), bearing.abs())),
                        primitive: None,
                    }
                }
            }
            HostOperation::HoldHeading {
                heading_rad,
                linear_m_s,
            } => {
                let error = self.request.bearing_rad.unwrap_or(*heading_rad);
                if error.abs() <= 0.10 {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"heading_error": error}),
                        &mut self.command_sent,
                    )
                } else if dispatch_due {
                    with_operation_progress(
                        dispatch_velocity(
                            self.cockpit,
                            operation,
                            context,
                            (*linear_m_s * 1_000.0).round() as i16,
                            radians_to_mrad(error).clamp(-400, 400),
                            &mut self.command_sent,
                            &mut primitive,
                        ),
                        Some(("bearing_error".into(), error.abs())),
                    )
                } else {
                    OrganPoll::Pending {
                        progress: Some(("heading_error".into(), error.abs())),
                        primitive: None,
                    }
                }
            }
            HostOperation::Approach { stop_range_m, .. } => {
                let Some(range) = self.request.range_m else {
                    return OrganPoll::Failed(
                        SkillFailure::new(
                            SkillOutcome::Failed,
                            "target_stale",
                            "approach target has no current range",
                        )
                        .for_operation(operation),
                    );
                };
                let Some(bearing) = self.request.bearing_rad else {
                    return OrganPoll::Failed(
                        SkillFailure::new(
                            SkillOutcome::Failed,
                            "target_stale",
                            "approach target has no current bearing",
                        )
                        .for_operation(operation),
                    );
                };
                if range <= *stop_range_m {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"range_m": range, "stop_range_m": stop_range_m}),
                        &mut self.command_sent,
                    )
                } else if dispatch_due {
                    with_operation_progress(
                        dispatch_velocity(
                            self.cockpit,
                            operation,
                            context,
                            50,
                            radians_to_mrad(bearing).clamp(-500, 500),
                            &mut self.command_sent,
                            &mut primitive,
                        ),
                        Some(("target_distance".into(), range)),
                    )
                } else {
                    OrganPoll::Pending {
                        progress: Some(("target_distance".into(), range)),
                        primitive: None,
                    }
                }
            }
            HostOperation::AlignWithDock => {
                if now.body.charging || self.home_base_contact {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"charging": true}),
                        &mut self.command_sent,
                    )
                } else {
                    let Some(cue) = DockIrCue::from_character(now.body.infrared_character) else {
                        return OrganPoll::Failed(
                            SkillFailure::new(
                                SkillOutcome::Failed,
                                "target_stale",
                                "Home Base IR gradient disappeared",
                            )
                            .for_operation(operation),
                        );
                    };
                    if dispatch_due {
                        with_operation_progress(
                            dispatch_velocity(
                                self.cockpit,
                                operation,
                                context,
                                50,
                                cue.steering_mrad_s(400),
                                &mut self.command_sent,
                                &mut primitive,
                            ),
                            self.request
                                .range_m
                                .map(|range| ("target_distance".into(), range)),
                        )
                    } else {
                        OrganPoll::Pending {
                            progress: self
                                .request
                                .range_m
                                .map(|range| ("target_distance".into(), range)),
                            primitive: None,
                        }
                    }
                }
            }
            HostOperation::SearchForDockSignal => {
                if self.home_base_contact
                    || DockIrCue::from_character(now.body.infrared_character).is_some()
                {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"infrared_character": now.body.infrared_character}),
                        &mut self.command_sent,
                    )
                } else if dispatch_due {
                    dispatch_velocity(
                        self.cockpit,
                        operation,
                        context,
                        0,
                        300,
                        &mut self.command_sent,
                        &mut primitive,
                    )
                } else {
                    OrganPoll::Pending {
                        progress: None,
                        primitive: None,
                    }
                }
            }
            HostOperation::VerifyCharging => {
                if now.body.charging || self.home_base_contact {
                    OrganPoll::Completed(json!({"charging": true}))
                } else if context.elapsed_ms >= 1_000 {
                    OrganPoll::Failed(
                        SkillFailure::new(
                            SkillOutcome::PostconditionFailed,
                            "charging_not_verified",
                            "Home Base contact did not produce charging",
                        )
                        .for_operation(operation),
                    )
                } else {
                    OrganPoll::Pending {
                        progress: None,
                        primitive: None,
                    }
                }
            }
            HostOperation::Drive {
                linear_m_s,
                duration_ms,
            } => {
                let progress = if self.request.progress_metric == "reverse_displacement" {
                    let expected_distance =
                        linear_m_s.abs() * (*duration_ms as f32 / 1_000.0).max(0.001);
                    (
                        "reverse_displacement".to_string(),
                        distance_from_start(state, &now.body) / expected_distance,
                    )
                } else {
                    (
                        "duration".to_string(),
                        context.elapsed_ms as f32 / (*duration_ms).max(1) as f32,
                    )
                };
                if context.elapsed_ms >= *duration_ms {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"duration_ms": duration_ms}),
                        &mut self.command_sent,
                    )
                } else if dispatch_due {
                    with_operation_progress(
                        dispatch_velocity(
                            self.cockpit,
                            operation,
                            context,
                            (*linear_m_s * 1_000.0).round() as i16,
                            0,
                            &mut self.command_sent,
                            &mut primitive,
                        ),
                        Some(progress),
                    )
                } else {
                    OrganPoll::Pending {
                        progress: Some(progress),
                        primitive: None,
                    }
                }
            }
            HostOperation::DriveDistance {
                distance_m,
                velocity_m_s,
            } => {
                let travelled =
                    distance_from_start(state, &now.body) * velocity_m_s.signum().max(-1.0);
                if travelled.abs() >= distance_m.abs() {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"distance_m": travelled}),
                        &mut self.command_sent,
                    )
                } else if dispatch_due {
                    with_operation_progress(
                        dispatch_velocity(
                            self.cockpit,
                            operation,
                            context,
                            (*velocity_m_s * 1_000.0).round() as i16,
                            0,
                            &mut self.command_sent,
                            &mut primitive,
                        ),
                        Some((
                            "reverse_displacement".into(),
                            travelled.abs() / distance_m.abs().max(0.001),
                        )),
                    )
                } else {
                    OrganPoll::Pending {
                        progress: Some((
                            "reverse_displacement".into(),
                            travelled.abs() / distance_m.abs().max(0.001),
                        )),
                        primitive: None,
                    }
                }
            }
            HostOperation::TurnBy { angle_rad } => {
                let turned = angle_delta(now.body.odometry.heading_rad, state.start_heading_rad);
                if turned.abs() >= angle_rad.abs() {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"turned_rad": turned}),
                        &mut self.command_sent,
                    )
                } else if dispatch_due {
                    with_operation_progress(
                        dispatch_velocity(
                            self.cockpit,
                            operation,
                            context,
                            0,
                            if *angle_rad < 0.0 { -300 } else { 300 },
                            &mut self.command_sent,
                            &mut primitive,
                        ),
                        Some((
                            "frontier_coverage".into(),
                            turned.abs() / angle_rad.abs().max(0.001),
                        )),
                    )
                } else {
                    OrganPoll::Pending {
                        progress: Some((
                            "frontier_coverage".into(),
                            turned.abs() / angle_rad.abs().max(0.001),
                        )),
                        primitive: None,
                    }
                }
            }
            HostOperation::FollowWall { side, .. } => {
                if !now.body.flags.wall {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"wall_clear": true}),
                        &mut self.command_sent,
                    )
                } else if dispatch_due {
                    with_operation_progress(
                        dispatch_velocity(
                            self.cockpit,
                            operation,
                            context,
                            45,
                            if side == "left" { 120 } else { -120 },
                            &mut self.command_sent,
                            &mut primitive,
                        ),
                        Some((
                            "path_progress".into(),
                            distance_from_start(state, &now.body).clamp(0.0, 1.0),
                        )),
                    )
                } else {
                    OrganPoll::Pending {
                        progress: None,
                        primitive: None,
                    }
                }
            }
            HostOperation::Undock => {
                if context.first_poll {
                    match self.cockpit.cmd_vel(-1, 0, 10) {
                        Ok(()) => {
                            self.command_sent = true;
                            primitive = Some(primitive_intent(
                                context,
                                operation,
                                json!({"linear_mm_s": -1, "angular_mrad_s": 0, "ttl_ms": 10}),
                            ));
                        }
                        Err(error) => {
                            return OrganPoll::Failed(cockpit_skill_failure(operation, error));
                        }
                    }
                }
                if !now.body.charging && context.elapsed_ms >= 1_500 {
                    OrganPoll::Completed(json!({"undocked": true}))
                } else {
                    OrganPoll::Pending {
                        progress: Some((
                            "dock_departure".into(),
                            context.elapsed_ms as f32 / 1_500.0,
                        )),
                        primitive: None,
                    }
                }
            }
            HostOperation::Retreat { hazard, distance_m } => {
                let incompatible_sensor = match hazard {
                    HazardKind::BumperFront => {
                        now.body.flags.cliff_left
                            || now.body.flags.cliff_front_left
                            || now.body.flags.cliff_front_right
                            || now.body.flags.cliff_right
                    }
                    HazardKind::Cliff => now.body.flags.bump_left || now.body.flags.bump_right,
                };
                let imu_absolute = self.status.imu.health.as_deref() == Some("fault")
                    || self
                        .status
                        .imu
                        .tilt_magnitude_mrad
                        .is_some_and(|value| value >= 650)
                    || self
                        .status
                        .imu
                        .impact_score_mm_s2
                        .is_some_and(|value| value >= 18_000);
                if self.status.estop_latched == Some(true)
                    || now.body.flags.wheel_drop
                    || now.body.charging
                    || incompatible_sensor
                    || imu_absolute
                {
                    return OrganPoll::Failed(
                        SkillFailure::new(
                            SkillOutcome::SafetyPreempted,
                            "absolute_hazard",
                            "an absolute or incompatible hazard forbids careful retreat",
                        )
                        .for_operation(operation),
                    );
                }
                if self.status.safety_latch_kind != Some(hazard.latch()) {
                    return OrganPoll::Failed(
                        SkillFailure::new(
                            SkillOutcome::SafetyPreempted,
                            "absolute_or_mismatched_hazard",
                            format!(
                                "Brainstem latch {:?} does not match requested {:?} retreat",
                                self.status.safety_latch_kind, hazard
                            ),
                        )
                        .for_operation(operation),
                    );
                }
                let active = match hazard {
                    HazardKind::BumperFront => {
                        now.body.flags.bump_left || now.body.flags.bump_right
                    }
                    HazardKind::Cliff => {
                        now.body.flags.cliff_left
                            || now.body.flags.cliff_front_left
                            || now.body.flags.cliff_front_right
                            || now.body.flags.cliff_right
                    }
                };
                let travelled = distance_from_start(state, &now.body);
                if !active {
                    if let Err(error) = self.cockpit.stop() {
                        return OrganPoll::Failed(cockpit_skill_failure(operation, error));
                    }
                    self.command_sent = true;
                    OrganPoll::Completed(json!({
                        "hazard": hazard,
                        "distance_m": travelled,
                        "clear": true,
                    }))
                } else if state.last_dispatch_ms == 0
                    || context.now_ms.saturating_sub(state.last_dispatch_ms)
                        >= context.primitive_ttl_ms as u64
                {
                    let Some(generation) = self.status.safety_hazard_generation.filter(|v| *v > 0)
                    else {
                        return OrganPoll::Failed(
                            SkillFailure::new(
                                SkillOutcome::Failed,
                                "hazard_generation_unavailable",
                                "Brainstem did not report an acknowledged hazard generation",
                            )
                            .for_operation(operation),
                        );
                    };
                    match self.cockpit.escape_motion(
                        hazard.latch(),
                        generation,
                        -100,
                        0,
                        context.primitive_ttl_ms,
                    ) {
                        Ok(()) => {
                            self.command_sent = true;
                            primitive = Some(primitive_intent(
                                context,
                                operation,
                                json!({
                                    "hazard": hazard,
                                    "hazard_generation": generation,
                                    "linear_mm_s": -100,
                                    "angular_mrad_s": 0,
                                    "ttl_ms": context.primitive_ttl_ms,
                                }),
                            ));
                            OrganPoll::Pending {
                                progress: Some((
                                    "reverse_displacement".into(),
                                    travelled / distance_m.max(0.001),
                                )),
                                primitive: None,
                            }
                        }
                        Err(error) => OrganPoll::Failed(cockpit_skill_failure(operation, error)),
                    }
                } else {
                    OrganPoll::Pending {
                        progress: Some((
                            "reverse_displacement".into(),
                            travelled / distance_m.max(0.001),
                        )),
                        primitive: None,
                    }
                }
            }
            HostOperation::CompleteHazardRecovery { hazard } => {
                let active = match hazard {
                    HazardKind::BumperFront => {
                        now.body.flags.bump_left || now.body.flags.bump_right
                    }
                    HazardKind::Cliff => {
                        now.body.flags.cliff_left
                            || now.body.flags.cliff_front_left
                            || now.body.flags.cliff_front_right
                            || now.body.flags.cliff_right
                    }
                };
                if self.status.safety_latch_kind != Some(hazard.latch())
                    || active
                    || now.body.flags.wheel_drop
                    || now.body.charging
                {
                    OrganPoll::Failed(
                        SkillFailure::new(
                            SkillOutcome::PostconditionFailed,
                            "hazard_not_clear",
                            "acknowledged hazard may be cleared only after its sensor is clear",
                        )
                        .for_operation(operation),
                    )
                } else if let Err(error) = self.cockpit.stop() {
                    OrganPoll::Failed(cockpit_skill_failure(operation, error))
                } else if let Err(error) = self.cockpit.clear_safety_latch(hazard.latch()) {
                    OrganPoll::Failed(cockpit_skill_failure(operation, error))
                } else {
                    self.command_sent = true;
                    OrganPoll::Completed(json!({
                        "hazard": hazard,
                        "clear": true,
                        "stopped": true,
                    }))
                }
            }
            HostOperation::ReleasePersistentBumper => OrganPoll::Failed(
                SkillFailure::new(
                    SkillOutcome::ScriptError,
                    "invalid_host_sequence",
                    "releasePersistentBumper policy must run through carefully and retreat",
                )
                .for_operation(operation),
            ),
            HostOperation::Observe { target } => OrganPoll::Completed(json!({
                "entity_id": target.id(),
                "observed_at_ms": now.t_ms,
                "provenance": "canonical_now",
            })),
            HostOperation::WaitUntil {
                predicate,
                timeout_ms: _,
            } => match predicate.as_str() {
                "charging" if now.body.charging => OrganPoll::Completed(json!(true)),
                "contact_clear" if !(now.body.flags.bump_left || now.body.flags.bump_right) => {
                    OrganPoll::Completed(json!(true))
                }
                "cliff_clear"
                    if !(now.body.flags.cliff_left
                        || now.body.flags.cliff_front_left
                        || now.body.flags.cliff_front_right
                        || now.body.flags.cliff_right) =>
                {
                    OrganPoll::Completed(json!(true))
                }
                _ => OrganPoll::Pending {
                    progress: None,
                    primitive: None,
                },
            },
            HostOperation::PlayFeedback { pattern } => {
                let kind = match pattern.as_str() {
                    "ok" => pete_cockpit::FeedbackKind::Ok,
                    "error" => pete_cockpit::FeedbackKind::Error,
                    "armed" => pete_cockpit::FeedbackKind::Armed,
                    "lost_target" => pete_cockpit::FeedbackKind::LostTarget,
                    "dock_seen" => pete_cockpit::FeedbackKind::DockSeen,
                    "danger" => pete_cockpit::FeedbackKind::Danger,
                    _ => return OrganPoll::Failed(SkillFailure::capability(operation)),
                };
                match self.cockpit.play_feedback(kind) {
                    Ok(()) => {
                        self.command_sent = true;
                        OrganPoll::Completed(json!({"played": pattern}))
                    }
                    Err(error) => OrganPoll::Failed(cockpit_skill_failure(operation, error)),
                }
            }
            HostOperation::Say { text } => {
                if context.first_poll {
                    primitive = Some(primitive_intent(
                        context,
                        operation,
                        json!({
                            "text": text,
                            "delivery": "motherbrain_mouth_queue",
                        }),
                    ));
                    OrganPoll::Pending {
                        progress: None,
                        primitive: None,
                    }
                } else {
                    OrganPoll::Completed(json!({
                        "text": text,
                        "attempted": true,
                    }))
                }
            }
            HostOperation::Scan
            | HostOperation::LookAt { .. }
            | HostOperation::Grasp { .. }
            | HostOperation::Release { .. }
            | HostOperation::BringToMouth { .. }
            | HostOperation::Chew
            | HostOperation::Swallow => OrganPoll::Failed(SkillFailure::capability(operation)),
        };
        if primitive.is_some() {
            state.last_dispatch_ms = context.now_ms;
        }
        match outcome {
            OrganPoll::Pending { progress, .. } => OrganPoll::Pending {
                progress,
                primitive,
            },
            terminal => {
                self.state.operations.remove(&context.operation_id);
                terminal
            }
        }
    }

    fn stop(&mut self, resource: BodyResource, _reason: &SkillFailure) {
        if resource == BodyResource::Locomotion {
            let _ = self.cockpit.stop();
            self.command_sent = true;
        }
    }
}

struct CockpitStopDriver<'a, C> {
    cockpit: &'a mut C,
}

impl<C: Cockpit> OrganDriver for CockpitStopDriver<'_, C> {
    fn poll(
        &mut self,
        operation: &HostOperation,
        _context: OperationContext,
        _now: &Now,
        _events: &pete_cockpit::EventBatch,
    ) -> OrganPoll {
        OrganPoll::Failed(
            SkillFailure::new(
                SkillOutcome::ResourcePreempted,
                "foreground_replaced",
                "operation was replaced before it could be polled",
            )
            .for_operation(operation),
        )
    }

    fn stop(&mut self, resource: BodyResource, _reason: &SkillFailure) {
        if resource == BodyResource::Locomotion {
            let _ = self.cockpit.stop();
        }
    }
}

impl PossessorSkillRuntime {
    fn active_request(&self) -> Option<SkillRequest> {
        self.status
            .as_ref()
            .filter(|status| status.phase != SkillPhase::Terminal)
            .map(|status| status.request.clone())
    }

    fn request_for_tick(&self, candidate: Option<SkillRequest>) -> Option<SkillRequest> {
        match (self.active_request(), candidate) {
            (Some(active), Some(candidate))
                if active.skill_id == candidate.skill_id
                    && active.implementation_id == candidate.implementation_id
                    && active.goal_id == candidate.goal_id =>
            {
                Some(candidate)
            }
            (Some(active), _) => Some(active),
            (None, candidate) => candidate,
        }
    }

    fn annotate_now(&self, now: &mut Now) {
        if let Some(provenance) = &self.provenance {
            now.extensions.insert(
                "motherbrain.skill_execution".to_string(),
                provenance.clone(),
            );
        }
    }

    fn step<C: Cockpit>(
        &mut self,
        cockpit: &mut C,
        request: &SkillRequest,
        now: &Now,
        status_summary: &StatusSummary,
        home_base_contact: bool,
        events: &pete_cockpit::EventBatch,
        now_ms: TimeMs,
    ) -> (SkillStatus, bool) {
        let Some(lua) = self.lua.as_mut() else {
            let status = SkillStatus {
                request: request.clone(),
                phase: SkillPhase::Terminal,
                outcome: Some(SkillOutcome::ScriptError),
                updated_at_ms: now_ms,
                reason: self.load_error.clone(),
                ..SkillStatus::default()
            };
            self.status = Some(status.clone());
            return (status, false);
        };
        if now_ms.saturating_sub(self.last_reload_check_ms) >= 1_000 {
            let _ = lua.reload();
            self.last_reload_check_ms = now_ms;
        }
        if let Some(current) = lua.active_skill_id() {
            let wanted = if request.skill_id == SkillId::RuntimeLoaded {
                request.implementation_id.clone().unwrap_or_default()
            } else {
                format!(
                    "motherbrain.{}",
                    match request.skill_id {
                        SkillId::StopAndStabilize => "stopAndStabilize",
                        SkillId::TurnTowardTarget => "turnTowardTarget",
                        SkillId::FollowBearing => "followBearingSkill",
                        SkillId::ApproachTarget => "approachTarget",
                        SkillId::BackAway => "driveFor",
                        SkillId::InspectTarget => "inspectObject",
                        SkillId::WallFollow => "wallFollow",
                        SkillId::AlignWithDock => "alignWithDockSkill",
                        SkillId::SystematicSearch => "systematicSearch",
                        SkillId::HoldHeading => "holdHeadingSkill",
                        SkillId::RetreatFromCliff => "retreatFromCliff",
                        SkillId::ReleasePersistentBumper => "releasePersistentBumper",
                        SkillId::TurnBy => "turnBySkill",
                        SkillId::DriveDistance => "driveDistanceSkill",
                        SkillId::Undock => "undockSkill",
                        SkillId::SearchForDock => "searchForDock",
                        SkillId::ReturnToDock => "returnToDock",
                        SkillId::RuntimeLoaded => unreachable!(),
                    }
                )
            };
            if current != wanted {
                if matches!(
                    request.skill_id,
                    SkillId::RetreatFromCliff | SkillId::ReleasePersistentBumper
                ) {
                    let mut stop_driver = CockpitStopDriver { cockpit };
                    let _ = lua.cancel(
                        &mut stop_driver,
                        SkillOutcome::ResourcePreempted,
                        "safety_recovery_preempted_foreground",
                        "acknowledged bodily hazard replaced the foreground skill",
                        now_ms,
                    );
                    self.provenance = lua
                        .execution_record()
                        .and_then(|record| serde_json::to_value(record).ok());
                    let _ = lua.take_terminal();
                    self.driver.operations.clear();
                } else {
                    // Competing ordinary goals remain pending. Only explicit
                    // higher authority calls the cancellation API.
                    return (
                        self.status.clone().unwrap_or_else(|| SkillStatus {
                            request: request.clone(),
                            phase: SkillPhase::Running,
                            updated_at_ms: now_ms,
                            reason: Some(format!("foreground skill {current} retains commitment")),
                            ..SkillStatus::default()
                        }),
                        false,
                    );
                }
            }
        }
        if !lua.is_active() {
            if let Err(error) = lua.start(request.clone(), now) {
                let status = SkillStatus {
                    request: request.clone(),
                    phase: SkillPhase::Terminal,
                    outcome: Some(SkillOutcome::ScriptError),
                    updated_at_ms: now_ms,
                    reason: Some(error.to_string()),
                    ..SkillStatus::default()
                };
                self.status = Some(status.clone());
                return (status, false);
            }
        }
        let mut driver = RealLuaOrganDriver {
            cockpit,
            request,
            status: status_summary,
            home_base_contact,
            state: &mut self.driver,
            command_sent: false,
        };
        let status = lua
            .step(now, events, &mut driver)
            .unwrap_or_else(|| SkillStatus {
                request: request.clone(),
                phase: SkillPhase::Terminal,
                outcome: Some(SkillOutcome::ScriptError),
                updated_at_ms: now_ms,
                reason: Some("Lua runtime lost the foreground invocation".into()),
                ..SkillStatus::default()
            });
        let command_sent = driver.command_sent;
        self.status = Some(status.clone());
        self.provenance = lua
            .execution_record()
            .and_then(|record| serde_json::to_value(record).ok());
        if status.phase == SkillPhase::Terminal {
            let _ = lua.take_terminal();
        }
        (status, command_sent)
    }
}

fn radians_to_mrad(value: f32) -> i16 {
    (value * 1_000.0)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn distance_from_start(state: &LuaDriverOperationState, body: &BodySense) -> f32 {
    (body.odometry.x_m - state.start_x_m).hypot(body.odometry.y_m - state.start_y_m)
}

fn angle_delta(current: f32, initial: f32) -> f32 {
    let mut delta = current - initial;
    while delta > std::f32::consts::PI {
        delta -= std::f32::consts::TAU;
    }
    while delta < -std::f32::consts::PI {
        delta += std::f32::consts::TAU;
    }
    delta
}

fn primitive_intent(
    context: OperationContext,
    operation: &HostOperation,
    detail: serde_json::Value,
) -> PrimitiveIntent {
    PrimitiveIntent {
        operation_id: context.operation_id,
        child_id: context.child_id,
        operation: operation.name().to_string(),
        resource: operation.resource(),
        emitted_at_ms: context.now_ms,
        detail,
    }
}

fn dispatch_velocity<C: Cockpit>(
    cockpit: &mut C,
    operation: &HostOperation,
    context: OperationContext,
    linear_mm_s: i16,
    angular_mrad_s: i16,
    command_sent: &mut bool,
    primitive: &mut Option<PrimitiveIntent>,
) -> OrganPoll {
    match cockpit.cmd_vel(linear_mm_s, angular_mrad_s, context.primitive_ttl_ms) {
        Ok(()) => {
            *command_sent = true;
            *primitive = Some(primitive_intent(
                context,
                operation,
                json!({
                    "linear_mm_s": linear_mm_s,
                    "angular_mrad_s": angular_mrad_s,
                    "ttl_ms": context.primitive_ttl_ms,
                }),
            ));
            OrganPoll::Pending {
                progress: None,
                primitive: None,
            }
        }
        Err(error) => OrganPoll::Failed(cockpit_skill_failure(operation, error)),
    }
}

fn with_operation_progress(poll: OrganPoll, progress: Option<(String, f32)>) -> OrganPoll {
    match poll {
        OrganPoll::Pending { primitive, .. } => OrganPoll::Pending {
            progress,
            primitive,
        },
        terminal => terminal,
    }
}

fn complete_stopped<C: Cockpit>(
    cockpit: &mut C,
    operation: &HostOperation,
    value: serde_json::Value,
    command_sent: &mut bool,
) -> OrganPoll {
    match cockpit.stop() {
        Ok(()) => {
            *command_sent = true;
            OrganPoll::Completed(value)
        }
        Err(error) => OrganPoll::Failed(cockpit_skill_failure(operation, error)),
    }
}

fn cockpit_skill_failure(
    operation: &HostOperation,
    error: pete_cockpit::CockpitError,
) -> SkillFailure {
    SkillFailure::new(
        SkillOutcome::Failed,
        "body_command_rejected",
        error.to_string(),
    )
    .for_operation(operation)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PossessionRecoveryPhase {
    Idle,
    BrainstemReflex,
    WaitingForSensorClear,
    Escaping,
}

#[derive(Clone, Debug)]
struct PossessionRecoveryState {
    latch: Option<SafetyLatchKind>,
    hazard_generation: u32,
    phase: PossessionRecoveryPhase,
    turn_direction: TurnDir,
    active_since_ms: TimeMs,
    last_command_ms: TimeMs,
    command_attempts: u32,
    stuck_stop_sent: bool,
    brainstem_reflex_observed: bool,
    last_reflex_outcome: Option<ContactWithdrawalOutcome>,
    last_observed_x_m: f32,
    last_observed_y_m: f32,
    last_observed_heading_rad: f32,
    observed_linear_m: f32,
    observed_turn_rad: f32,
}

impl Default for PossessionRecoveryState {
    fn default() -> Self {
        Self {
            latch: None,
            hazard_generation: 0,
            phase: PossessionRecoveryPhase::Idle,
            turn_direction: TurnDir::Left,
            active_since_ms: 0,
            last_command_ms: 0,
            command_attempts: 0,
            stuck_stop_sent: false,
            brainstem_reflex_observed: false,
            last_reflex_outcome: None,
            last_observed_x_m: 0.0,
            last_observed_y_m: 0.0,
            last_observed_heading_rad: 0.0,
            observed_linear_m: 0.0,
            observed_turn_rad: 0.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct MotionRejectionState {
    first_ms: TimeMs,
    last_ms: TimeMs,
    blocked_until_ms: TimeMs,
    latest_command_id: u32,
    latest_reason: Option<String>,
    count: u32,
    stuck: bool,
    stuck_stop_sent: bool,
}

const POSSESSION_RECOVERY_STUCK_AFTER_MS: TimeMs = 15_000;
const POSSESSION_ESCAPE_TTL_MS: u32 = 250;
const MOTION_REJECTION_WINDOW_MS: TimeMs = 5_000;
const MOTION_REJECTION_BASE_BACKOFF_MS: TimeMs = 1_000;
const MOTION_REJECTION_MAX_BACKOFF_MS: TimeMs = 5_000;
const MOTION_REJECTION_STUCK_AFTER: u32 = 3;

#[derive(Clone, Debug)]
struct PossessionRecoveryDecision {
    block_reason: Option<String>,
    command_sent: bool,
    action: Option<ActionPrimitive>,
    motor: Option<MotorCommand>,
    debug: serde_json::Value,
}

impl<R, C> RealRobotRunner<R, C>
where
    R: RuntimeLoop + Send,
    C: Cockpit + Send,
{
    pub fn new(
        mode: RobotMode,
        cockpit: C,
        sensors: Vec<Box<dyn SenseProducer + Send>>,
        runtime: R,
    ) -> Self {
        let sensor_poll_health = sensors
            .iter()
            .map(|sensor| SensorPollHealth {
                name: sensor.source_name().to_string(),
                ..SensorPollHealth::default()
            })
            .collect();
        Self {
            mode,
            cockpit: SafeCockpit::new(cockpit),
            sensors,
            runtime,
            tick_ms: 100,
            tick_count: 0,
            autonomous_motion: false,
            now_builder: NowBuilder::new(),
            frame_processor: FrameProcessor::new(),
            live_image_cognition: LiveImageCognition::new(None),
            robot_initialization: None,
            brainstem_interface: None,
            possession_recovery: PossessionRecoveryState::default(),
            motion_rejection: MotionRejectionState::default(),
            possessor_skills: PossessorSkillRuntime::default(),
            sensor_poll_health,
        }
    }

    pub fn with_frame_processor(mut self, frame_processor: FrameProcessor) -> Self {
        self.frame_processor = frame_processor;
        self
    }

    pub fn with_live_image_enricher(mut self, enricher: Option<LiveImageEnricher>) -> Self {
        self.live_image_cognition = LiveImageCognition::new(enricher);
        self
    }

    pub fn with_robot_initialization(mut self, initialization: serde_json::Value) -> Self {
        self.robot_initialization = Some(initialization);
        self
    }

    pub fn with_brainstem_interface(mut self, capabilities: serde_json::Value) -> Self {
        self.brainstem_interface = Some(capabilities);
        self
    }

    pub fn with_autonomous_motion(mut self, enabled: bool) -> Self {
        self.autonomous_motion = enabled;
        self
    }

    pub async fn run_read_only(&mut self, steps: Option<usize>) -> Result<()> {
        self.run_read_only_observing(steps, |_, _| {}).await
    }

    pub async fn run_read_only_observing<F>(
        &mut self,
        steps: Option<usize>,
        mut observe: F,
    ) -> Result<()>
    where
        F: FnMut(&WorldSnapshot, &RuntimeTick),
    {
        if self.mode != RobotMode::ReadOnly {
            anyhow::bail!("only read-only robot mode is implemented");
        }

        while steps.map(|limit| self.tick_count < limit).unwrap_or(true) {
            let (snapshot, tick) = self.tick_read_only().await?;
            observe(&snapshot, &tick);
            tokio::time::sleep(std::time::Duration::from_millis(self.tick_ms)).await;
        }
        Ok(())
    }

    pub async fn tick_read_only(&mut self) -> Result<(WorldSnapshot, RuntimeTick)> {
        if self.mode != RobotMode::ReadOnly {
            anyhow::bail!("only read-only robot mode is implemented");
        }

        let body = body_sense_from_cockpit_status(self.cockpit.refresh_status()?, wall_time_ms());
        let brainstem_events = self.cockpit.poll_events()?;
        let mut packets = poll_sensors_lossy(&mut self.sensors, &mut self.sensor_poll_health).await;
        let t_ms = body.last_update_ms.max(wall_time_ms());
        self.frame_processor.process_packets(t_ms, &mut packets);
        let mut now = self.now_builder.build(t_ms, body, packets)?;
        insert_sensor_health(&mut now, &self.sensor_poll_health);
        now.self_sense.mode = Some("read-only".to_string());
        now.extensions.insert(
            "source".to_string(),
            serde_json::Value::String("real_robot_read_only".to_string()),
        );
        now.extensions.insert(
            "mode".to_string(),
            serde_json::Value::String("read_only".to_string()),
        );
        now.extensions.insert(
            "safety/read_only_veto".to_string(),
            serde_json::Value::Bool(true),
        );
        now.extensions.insert(
            "read_only_motor_gate".to_string(),
            serde_json::json!({
                "motor_applied": false,
                "final_motor": MotorCommand::stop(),
                "safety_reason": "ReadOnlyMode",
            }),
        );
        self.insert_robot_initialization(&mut now);
        self.insert_brainstem_interface(&mut now, &brainstem_events);
        self.possessor_skills.annotate_now(&mut now);
        enrich_now_latest_image(&mut self.live_image_cognition, &mut now).await;

        let tick = self
            .runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await?;
        let mut snapshot = self.now_builder.snapshot();
        snapshot.eye = tick.frame.now.eye.clone();
        annotate_snapshot_from_tick(&mut snapshot, &tick);
        self.tick_count = self.tick_count.saturating_add(1);
        Ok((snapshot, tick))
    }

    pub async fn tick_slow_manual(&mut self) -> Result<(WorldSnapshot, RuntimeTick)> {
        if self.mode != RobotMode::Slow {
            anyhow::bail!("slow manual tick requires RobotMode::Slow");
        }

        let pre_body_t_ms = wall_time_ms();
        let brainstem_events = self.poll_slow_possession_events()?;
        if let Some(input) = self.runtime.reign_sense(pre_body_t_ms)?.latest {
            if reign_input_outputs_real_slow_directly(&input) {
                let body_before = pete_body::BodySense {
                    last_update_ms: pre_body_t_ms,
                    ..pete_body::BodySense::default()
                };
                let mut now =
                    self.now_builder
                        .build(pre_body_t_ms, body_before.clone(), Vec::new())?;
                now.self_sense.mode = Some("slow".to_string());
                self.insert_possession_snapshot(&mut now);
                now.extensions.insert(
                    "source".to_string(),
                    serde_json::Value::String("real_robot".to_string()),
                );
                now.extensions.insert(
                    "mode".to_string(),
                    serde_json::Value::String("slow".to_string()),
                );
                self.insert_robot_initialization(&mut now);
                self.insert_brainstem_interface(&mut now, &brainstem_events);
                now.reign.latest = Some(input.clone());
                now.reign.active = true;
                now.reign.mode = Some(input.mode.clone());
                now.reign.human_override_pressure = input.priority.clamp(0.0, 1.0);
                now.reign.last_command_age_ms =
                    Some(pre_body_t_ms.saturating_sub(input.issued_at_ms));
                let tick = synthetic_slow_manual_tick(
                    now,
                    input,
                    MotorCommand::stop(),
                    MotorCommand::stop(),
                    None,
                    &body_before,
                )?;
                let mut snapshot = self.now_builder.snapshot();
                snapshot.eye = tick.frame.now.eye.clone();
                annotate_snapshot_from_tick(&mut snapshot, &tick);
                self.tick_count = self.tick_count.saturating_add(1);
                return Ok((snapshot, tick));
            }
        }

        let status_before = self.cockpit.refresh_status()?;
        let body_before = body_sense_from_cockpit_status(status_before.clone(), wall_time_ms());
        let recovery_decision =
            self.apply_possession_recovery(&body_before, &brainstem_events, &status_before)?;
        let mut packets = poll_sensors_lossy(&mut self.sensors, &mut self.sensor_poll_health).await;
        let t_ms = body_before.last_update_ms.max(wall_time_ms());
        self.frame_processor.process_packets(t_ms, &mut packets);
        let mut now = self.now_builder.build(t_ms, body_before.clone(), packets)?;
        insert_sensor_health(&mut now, &self.sensor_poll_health);
        now.self_sense.mode = Some("slow".to_string());
        self.insert_possession_snapshot(&mut now);
        now.extensions.insert(
            "source".to_string(),
            serde_json::Value::String("real_robot".to_string()),
        );
        now.extensions.insert(
            "mode".to_string(),
            serde_json::Value::String("slow".to_string()),
        );
        self.insert_robot_initialization(&mut now);
        self.insert_brainstem_interface(&mut now, &brainstem_events);
        self.possessor_skills.annotate_now(&mut now);

        now.reign = self.runtime.reign_sense(t_ms)?;
        if let Some(input) = now.reign.latest.clone() {
            if reign_input_outputs_real_slow_directly(&input) {
                let tick = synthetic_slow_manual_tick(
                    now,
                    input,
                    MotorCommand::stop(),
                    MotorCommand::stop(),
                    None,
                    &body_before,
                )?;
                let mut snapshot = self.now_builder.snapshot();
                snapshot.eye = tick.frame.now.eye.clone();
                annotate_snapshot_from_tick(&mut snapshot, &tick);
                self.tick_count = self.tick_count.saturating_add(1);
                return Ok((snapshot, tick));
            }
            if reign_input_drives_real_slow(&input) {
                let desired_motor =
                    action_to_motor_command(input.command.to_action().as_ref()).clamped(0.05, 0.5);
                let mut block_reason = recovery_decision
                    .block_reason
                    .clone()
                    .or_else(|| self.motion_rejection_block_reason(wall_time_ms()))
                    .or_else(|| real_slow_body_block_reason(&body_before));
                let mut final_motor = if block_reason.is_none() {
                    desired_motor
                } else {
                    MotorCommand::stop()
                };
                if !recovery_decision.command_sent {
                    if let Some(block) =
                        apply_slow_possession_motor(&mut self.cockpit, final_motor)?
                    {
                        match block {
                            SlowPossessionMotionBlock::SafetyLatch(latch) => {
                                self.start_possession_recovery(latch, &body_before);
                                block_reason = Some(format!("recovering {latch:?} safety latch"));
                            }
                            SlowPossessionMotionBlock::CommandRejected { command_id, reason } => {
                                block_reason =
                                    Some(self.record_motion_rejection(command_id, &reason));
                            }
                        }
                        final_motor = MotorCommand::stop();
                    } else if !is_near_zero_motor(final_motor) {
                        self.motion_rejection = MotionRejectionState::default();
                    }
                }
                let tick = synthetic_slow_manual_tick(
                    now,
                    input,
                    desired_motor,
                    final_motor,
                    block_reason,
                    &body_before,
                )?;
                let mut snapshot = self.now_builder.snapshot();
                snapshot.eye = tick.frame.now.eye.clone();
                annotate_snapshot_from_tick(&mut snapshot, &tick);
                self.tick_count = self.tick_count.saturating_add(1);
                return Ok((snapshot, tick));
            }
        }

        enrich_now_latest_image(&mut self.live_image_cognition, &mut now).await;

        let mut tick = self
            .runtime
            .tick(now.clone(), ExperienceLatent::default(), Vec::new())
            .await?;
        let original_chosen_action = tick.chosen_action.clone();
        let chosen_motor = final_motor_from_tick(&tick).clamped(0.05, 0.5);
        let manual_drive = tick
            .frame
            .reign_input
            .as_ref()
            .map(reign_input_drives_real_slow)
            .unwrap_or(false);
        let mut block_reason = recovery_decision
            .block_reason
            .clone()
            .or_else(|| self.motion_rejection_block_reason(wall_time_ms()))
            .or_else(|| {
                real_slow_motor_block_reason(
                    &body_before,
                    &tick,
                    manual_drive,
                    self.autonomous_motion,
                )
            });
        let mut final_motor = if block_reason.is_none() {
            chosen_motor
        } else {
            MotorCommand::stop()
        };
        let recovery_request = self.possession_recovery_skill_request(&brainstem_events);
        let reflex_preemption = brainstem_events.events.iter().any(|event| {
            matches!(
                event.kind,
                CockpitEventKind::ContactWithdrawalStarted
                    | CockpitEventKind::SafetyTripped
                    | CockpitEventKind::EStopLatched
            )
        });
        let interrupted_request = reflex_preemption
            .then(|| self.possessor_skills.status.as_ref())
            .flatten()
            .filter(|status| status.phase != SkillPhase::Terminal)
            .map(|status| status.request.clone());
        let selected_skill_request =
            recovery_request
                .clone()
                .or(interrupted_request)
                .or_else(|| {
                    self.possessor_skills
                        .request_for_tick(tick.skill_request.clone())
                });
        let possessor_skill_owns_motion = selected_skill_request.is_some();
        let mut recovery_motion_sent = false;
        if let Some(request) = selected_skill_request {
            tick.skill_request = Some(request.clone());
            final_motor = MotorCommand::stop();
            if (block_reason.is_none() || recovery_request.is_some() || reflex_preemption)
                && !recovery_decision.command_sent
            {
                let (status, command_sent) = self.possessor_skills.step(
                    self.cockpit.client_mut(),
                    &request,
                    &tick.frame.now,
                    &status_before,
                    status_before.battery.home_base(),
                    &brainstem_events,
                    t_ms,
                );
                recovery_motion_sent = command_sent
                    && recovery_request.is_some()
                    && status.script.as_ref().is_some_and(|script| {
                        script.current_operation.as_deref() == Some("retreat")
                    });
                self.runtime.observe_skill_status(&status);
                if recovery_request.is_some()
                    && status.phase == SkillPhase::Terminal
                    && status.outcome == Some(SkillOutcome::Completed)
                {
                    self.finish_possession_recovery();
                }
                tick.skill_status = Some(status);
                self.possessor_skills.annotate_now(&mut tick.frame.now);
            }
        }
        if !recovery_decision.command_sent && !possessor_skill_owns_motion {
            if let Some(block) = apply_slow_possession_motor(&mut self.cockpit, final_motor)? {
                match block {
                    SlowPossessionMotionBlock::SafetyLatch(latch) => {
                        self.start_possession_recovery(latch, &body_before);
                        block_reason = Some(format!("recovering {latch:?} safety latch"));
                    }
                    SlowPossessionMotionBlock::CommandRejected { command_id, reason } => {
                        block_reason = Some(self.record_motion_rejection(command_id, &reason));
                    }
                }
                final_motor = MotorCommand::stop();
            } else if !is_near_zero_motor(final_motor) {
                self.motion_rejection = MotionRejectionState::default();
            }
        }
        if self.motion_rejection_block_reason(wall_time_ms()).is_some() {
            tick.chosen_action = Some(ActionPrimitive::Stop);
            tick.frame.chosen_action = Some(ActionPrimitive::Stop);
            if self.motion_rejection.stuck && !self.motion_rejection.stuck_stop_sent {
                self.cockpit.client_mut().stop()?;
                self.motion_rejection.stuck_stop_sent = true;
            }
        } else if let Some(recovery_action) = recovery_decision.action.clone() {
            tick.chosen_action = Some(recovery_action.clone());
            tick.frame.chosen_action = Some(recovery_action);
        } else if recovery_motion_sent {
            let recovery_action = ActionPrimitive::Go {
                intensity: -0.25,
                duration_ms: POSSESSION_ESCAPE_TTL_MS as TimeMs,
            };
            tick.chosen_action = Some(recovery_action.clone());
            tick.frame.chosen_action = Some(recovery_action);
        } else if possessor_skill_owns_motion {
            tick.chosen_action = Some(ActionPrimitive::Stop);
            tick.frame.chosen_action = Some(ActionPrimitive::Stop);
        }

        let mut snapshot = self.now_builder.snapshot();
        snapshot.eye = tick.frame.now.eye.clone();
        annotate_snapshot_from_tick(&mut snapshot, &tick);
        let mut action_debug = snapshot
            .action_debug
            .take()
            .unwrap_or_else(|| serde_json::json!({}));
        if !action_debug.is_object() {
            action_debug = serde_json::json!({});
        }
        let reported_robot_motor = if recovery_motion_sent {
            MotorCommand {
                forward: -0.10,
                turn: 0.0,
            }
        } else if recovery_decision.command_sent {
            recovery_decision.motor.unwrap_or_else(MotorCommand::stop)
        } else if recovery_decision.block_reason.is_some() {
            MotorCommand::stop()
        } else {
            final_motor
        };
        if let Some(object) = action_debug.as_object_mut() {
            object.insert("body_pose_before".to_string(), pose_json(&body_before));
            object.insert("body_pose_after".to_string(), pose_json(&snapshot.body));
            object.insert(
                "desired_motor".to_string(),
                serde_json::to_value(chosen_motor)?,
            );
            object.insert(
                "final_motor".to_string(),
                serde_json::to_value(final_motor)?,
            );
            object.insert(
                "motion_sent_to_robot".to_string(),
                serde_json::to_value(motor_command_to_motion(reported_robot_motor))?,
            );
            object.insert(
                "motion_sent_to_sim".to_string(),
                serde_json::to_value(motor_command_to_motion(final_motor))?,
            );
            object.insert(
                "motor_applied".to_string(),
                serde_json::json!(!is_near_zero_motor(reported_robot_motor)),
            );
            object.insert(
                "runtime_chosen_action".to_string(),
                serde_json::to_value(original_chosen_action)?,
            );
            object.insert(
                "recovery_action".to_string(),
                serde_json::to_value(recovery_decision.action.clone())?,
            );
            object.insert(
                "recovery_motor".to_string(),
                recovery_decision
                    .motor
                    .map(serde_json::to_value)
                    .transpose()?
                    .unwrap_or(serde_json::Value::Null),
            );
            object.insert(
                "manual_hardware_gate".to_string(),
                serde_json::json!(manual_drive),
            );
            object.insert(
                "autonomous_hardware_gate".to_string(),
                serde_json::json!(self.autonomous_motion),
            );
            object.insert(
                "possession_recovery".to_string(),
                recovery_decision.debug.clone(),
            );
            object.insert(
                "possessor_skill_request".to_string(),
                serde_json::to_value(&tick.skill_request)?,
            );
            object.insert(
                "possessor_skill_status".to_string(),
                serde_json::to_value(&tick.skill_status)?,
            );
            object.insert(
                "possessor_skill_execution".to_string(),
                self.possessor_skills
                    .provenance
                    .clone()
                    .unwrap_or(serde_json::Value::Null),
            );
            object.insert(
                "motion_rejection".to_string(),
                motion_rejection_debug(&self.motion_rejection),
            );
            object.insert(
                "why_not_moving".to_string(),
                block_reason
                    .clone()
                    .map(serde_json::Value::String)
                    .unwrap_or(serde_json::Value::Null),
            );
        }
        snapshot.action_debug = Some(action_debug);
        self.tick_count = self.tick_count.saturating_add(1);
        Ok((snapshot, tick))
    }

    fn insert_robot_initialization(&mut self, now: &mut Now) {
        if self.tick_count == 0 {
            if let Some(initialization) = self.robot_initialization.take() {
                now.extensions
                    .insert("robot.initialization".to_string(), initialization);
            }
        }
    }

    fn insert_possession_snapshot(&mut self, now: &mut Now) {
        if let Some(snapshot) = self.cockpit.client_mut().possession_snapshot() {
            now.extensions.insert(
                "brainstem.possession".to_string(),
                serde_json::to_value(snapshot).unwrap_or_else(|error| {
                    serde_json::json!({"possessed": false, "refusal_reason": error.to_string()})
                }),
            );
        }
    }

    fn poll_slow_possession_events(&mut self) -> Result<pete_cockpit::EventBatch> {
        let events = self.cockpit.poll_events_allowing_history_gap()?;
        if events.dropped_before_seq > 0 {
            if self.tick_count == 0 {
                eprintln!(
                    "slow possession recovered from pre-loop event history gap before sequence {}",
                    events.dropped_before_seq
                );
            } else {
                eprintln!(
                    "slow possession recovered from event history gap before sequence {}; stopping before continuing",
                    events.dropped_before_seq
                );
                self.cockpit.client_mut().stop()?;
            }
        }
        Ok(events)
    }

    fn apply_possession_recovery(
        &mut self,
        body: &BodySense,
        events: &pete_cockpit::EventBatch,
        status: &StatusSummary,
    ) -> Result<PossessionRecoveryDecision> {
        for event in &events.events {
            if let Some(reflex) = event.contact_withdrawal() {
                match reflex {
                    ContactWithdrawalEvent::Started { repeated_count, .. } => {
                        self.start_possession_recovery(SafetyLatchKind::Bump, body);
                        self.possession_recovery.phase = PossessionRecoveryPhase::BrainstemReflex;
                        self.possession_recovery.brainstem_reflex_observed = true;
                        self.possession_recovery.command_attempts = u32::from(repeated_count);
                        self.possession_recovery.last_command_ms = wall_time_ms();
                        self.possession_recovery.last_reflex_outcome = None;
                    }
                    ContactWithdrawalEvent::Completed { outcome, .. } => {
                        self.possession_recovery.brainstem_reflex_observed = true;
                        self.possession_recovery.last_reflex_outcome = Some(outcome);
                        self.possession_recovery.phase =
                            PossessionRecoveryPhase::WaitingForSensorClear;
                    }
                }
                continue;
            }
            match event.kind {
                CockpitEventKind::SafetyTripped => {
                    if let Some(kind) = safety_latch_kind_from_event_code(event.a) {
                        self.start_possession_recovery_generation(kind, event.seq, body);
                    }
                }
                CockpitEventKind::SafetyCleared => {
                    if self.possession_recovery.latch == safety_latch_kind_from_event_code(event.a)
                    {
                        self.possession_recovery = PossessionRecoveryState::default();
                    }
                }
                CockpitEventKind::EStopLatched => {
                    // The event contract does not distinguish an operator
                    // E-stop from any internally generated stop. Without
                    // trustworthy provenance, fail closed and require an
                    // explicit operator clear.
                    self.finish_possession_recovery();
                }
                _ => {}
            }
        }

        if status.estop_latched == Some(true) {
            self.possession_recovery = PossessionRecoveryState::default();
            return Ok(PossessionRecoveryDecision {
                block_reason: Some(
                    "operator E-stop is latched; explicit operator clear required".to_string(),
                ),
                command_sent: false,
                action: Some(ActionPrimitive::Stop),
                motor: Some(MotorCommand::stop()),
                debug: possession_recovery_debug(&self.possession_recovery, None, false),
            });
        }

        if self.possession_recovery.latch.is_none()
            && status.safety_tripped == Some(true)
            && status.estop_latched != Some(true)
        {
            if let Some(kind) = status
                .safety_latch_kind
                .or_else(|| infer_safety_latch_from_sensors(body))
            {
                self.start_possession_recovery_generation(
                    kind,
                    status.safety_hazard_generation.unwrap_or(0),
                    body,
                );
            }
        }

        let Some(latch) = self.possession_recovery.latch else {
            return Ok(PossessionRecoveryDecision {
                block_reason: None,
                command_sent: false,
                action: None,
                motor: None,
                debug: possession_recovery_debug(&self.possession_recovery, None, false),
            });
        };

        self.observe_possession_recovery_motion(body);
        let mut command_sent = false;
        let mut action = None;
        let mut motor = None;
        let mut reason = format!("recovering {latch:?} safety latch");
        match latch {
            SafetyLatchKind::Bump => {
                if self.possession_recovery.phase == PossessionRecoveryPhase::BrainstemReflex {
                    reason =
                        "brainstem contact-withdrawal reflex owns motion; possessor is observing"
                            .to_string();
                } else {
                    reason = "bumper recovery is delegated to the foreground Lua releasePersistentBumper skill"
                        .to_string();
                }
            }
            SafetyLatchKind::Cliff => {
                reason = "cliff recovery is delegated to the foreground Lua retreatFromCliff skill"
                    .to_string();
            }
            SafetyLatchKind::WheelDrop => {
                if body.flags.wheel_drop {
                    action = Some(ActionPrimitive::Stop);
                    motor = Some(MotorCommand::stop());
                    self.cockpit.client_mut().stop()?;
                    command_sent = true;
                } else {
                    self.cockpit.client_mut().clear_safety_latch(latch)?;
                    command_sent = true;
                    self.finish_possession_recovery();
                }
            }
            SafetyLatchKind::Charging => {
                if body.charging {
                    action = Some(ActionPrimitive::Stop);
                    motor = Some(MotorCommand::stop());
                    self.cockpit.client_mut().stop()?;
                    command_sent = true;
                } else {
                    self.cockpit.client_mut().clear_safety_latch(latch)?;
                    command_sent = true;
                    self.finish_possession_recovery();
                }
            }
            SafetyLatchKind::Heartbeat => {
                self.cockpit.client_mut().clear_safety_latch(latch)?;
                self.finish_possession_recovery();
                command_sent = true;
            }
            SafetyLatchKind::Tilt | SafetyLatchKind::Impact => {
                if imu_recovery_clear(status, latch) {
                    self.cockpit.client_mut().clear_safety_latch(latch)?;
                    self.finish_possession_recovery();
                    command_sent = true;
                } else {
                    action = Some(ActionPrimitive::Stop);
                    motor = Some(MotorCommand::stop());
                    self.cockpit.client_mut().stop()?;
                    command_sent = true;
                }
            }
        }

        Ok(PossessionRecoveryDecision {
            block_reason: Some(reason),
            command_sent,
            action,
            motor,
            debug: possession_recovery_debug(&self.possession_recovery, Some(latch), command_sent),
        })
    }

    fn possession_recovery_skill_request(
        &self,
        events: &pete_cockpit::EventBatch,
    ) -> Option<SkillRequest> {
        if self.possession_recovery.phase == PossessionRecoveryPhase::BrainstemReflex
            || self.possession_recovery.hazard_generation == 0
            || events.events.iter().any(|event| {
                matches!(
                    event.kind,
                    CockpitEventKind::ContactWithdrawalStarted | CockpitEventKind::SafetyTripped
                )
            })
        {
            return None;
        }
        let skill_id = match self.possession_recovery.latch? {
            SafetyLatchKind::Bump => SkillId::ReleasePersistentBumper,
            SafetyLatchKind::Cliff => SkillId::RetreatFromCliff,
            _ => return None,
        };
        Some(SkillRequest {
            skill_id,
            goal_id: Some(pete_conductor::GoalId::new("escape_danger")),
            behavior_id: Some("acknowledged_hazard_recovery".to_string()),
            maximum_duration_ms: POSSESSION_RECOVERY_STUCK_AFTER_MS,
            expected_progress: 1.0,
            progress_metric: "reverse_displacement".to_string(),
            progress_baseline: Some(0.0),
            progress_tolerance: 0.1,
            ..SkillRequest::default()
        })
    }

    fn start_possession_recovery(&mut self, latch: SafetyLatchKind, body: &BodySense) {
        self.start_possession_recovery_generation(latch, 0, body);
    }

    fn start_possession_recovery_generation(
        &mut self,
        latch: SafetyLatchKind,
        hazard_generation: u32,
        body: &BodySense,
    ) {
        let now_ms = wall_time_ms();
        let latch_changed = self.possession_recovery.latch != Some(latch);
        self.possession_recovery.latch = Some(latch);
        if hazard_generation != 0 {
            self.possession_recovery.hazard_generation = hazard_generation;
        }
        self.possession_recovery.phase = PossessionRecoveryPhase::WaitingForSensorClear;
        self.possession_recovery.turn_direction = recovery_turn_direction_for_latch(latch, body);
        if latch_changed || self.possession_recovery.active_since_ms == 0 {
            self.possession_recovery.active_since_ms = now_ms;
            self.possession_recovery.last_command_ms = 0;
            self.possession_recovery.command_attempts = 0;
            self.possession_recovery.stuck_stop_sent = false;
            self.possession_recovery.last_observed_x_m = body.odometry.x_m;
            self.possession_recovery.last_observed_y_m = body.odometry.y_m;
            self.possession_recovery.last_observed_heading_rad = body.odometry.heading_rad;
            self.possession_recovery.observed_linear_m = 0.0;
            self.possession_recovery.observed_turn_rad = 0.0;
        }
    }

    fn observe_possession_recovery_motion(&mut self, body: &BodySense) {
        if self.possession_recovery.latch.is_none() {
            return;
        }
        let dx = body.odometry.x_m - self.possession_recovery.last_observed_x_m;
        let dy = body.odometry.y_m - self.possession_recovery.last_observed_y_m;
        let distance = dx.hypot(dy);
        let heading_delta =
            body.odometry.heading_rad - self.possession_recovery.last_observed_heading_rad;
        self.possession_recovery.observed_linear_m += distance;
        self.possession_recovery.observed_turn_rad += heading_delta;
        self.possession_recovery.last_observed_x_m = body.odometry.x_m;
        self.possession_recovery.last_observed_y_m = body.odometry.y_m;
        self.possession_recovery.last_observed_heading_rad = body.odometry.heading_rad;
    }

    fn finish_possession_recovery(&mut self) {
        self.possession_recovery.latch = None;
        self.possession_recovery.hazard_generation = 0;
        self.possession_recovery.phase = PossessionRecoveryPhase::Idle;
        self.possession_recovery.active_since_ms = 0;
        self.possession_recovery.last_command_ms = 0;
        self.possession_recovery.brainstem_reflex_observed = false;
    }

    fn motion_rejection_block_reason(&self, now_ms: TimeMs) -> Option<String> {
        let reason = self
            .motion_rejection
            .latest_reason
            .as_deref()
            .unwrap_or("unknown rejection");
        if self.motion_rejection.stuck {
            return Some(format!(
                "brainstem repeatedly rejected motion; latest command #{}: {}; operator intervention needed",
                self.motion_rejection.latest_command_id, reason
            ));
        }
        if self.motion_rejection.blocked_until_ms > now_ms {
            return Some(format!(
                "brainstem rejected motion command #{}: {}; pausing motion retries for {} ms",
                self.motion_rejection.latest_command_id,
                reason,
                self.motion_rejection
                    .blocked_until_ms
                    .saturating_sub(now_ms)
            ));
        }
        None
    }

    fn record_motion_rejection(&mut self, command_id: u32, reason: &str) -> String {
        let now_ms = wall_time_ms();
        if self.motion_rejection.first_ms == 0
            || now_ms.saturating_sub(self.motion_rejection.first_ms) > MOTION_REJECTION_WINDOW_MS
        {
            self.motion_rejection = MotionRejectionState {
                first_ms: now_ms,
                ..MotionRejectionState::default()
            };
        }
        self.motion_rejection.last_ms = now_ms;
        self.motion_rejection.latest_command_id = command_id;
        self.motion_rejection.latest_reason = Some(reason.to_string());
        self.motion_rejection.count = self.motion_rejection.count.saturating_add(1);
        let backoff = MOTION_REJECTION_BASE_BACKOFF_MS
            .saturating_mul(u64::from(self.motion_rejection.count))
            .min(MOTION_REJECTION_MAX_BACKOFF_MS);
        self.motion_rejection.blocked_until_ms = now_ms.saturating_add(backoff);
        if self.motion_rejection.count >= MOTION_REJECTION_STUCK_AFTER {
            self.motion_rejection.stuck = true;
        }

        if self.motion_rejection.stuck {
            format!(
                "brainstem repeatedly rejected motion; latest command #{command_id}: {reason}; operator intervention needed"
            )
        } else {
            format!(
                "brainstem rejected motion command #{command_id}: {reason}; pausing motion retries for {backoff} ms"
            )
        }
    }

    fn insert_brainstem_interface(&self, now: &mut Now, events: &pete_cockpit::EventBatch) {
        if let Some(capabilities) = &self.brainstem_interface {
            now.extensions.insert(
                "brainstem.interface".to_string(),
                serde_json::json!({
                    "capabilities": capabilities,
                    "source": "brainstem",
                    "underlying_body_private": true,
                }),
            );
        }
        now.extensions.insert(
            "brainstem.events".to_string(),
            serde_json::to_value(events).unwrap_or_else(
                |error| serde_json::json!({"error": error.to_string(), "events": []}),
            ),
        );
    }
}

async fn enrich_now_latest_image(cognition: &mut LiveImageCognition, now: &mut Now) {
    let update = cognition
        .poll_and_submit(now.eye_frame.as_ref(), now.world.revision, now.t_ms)
        .await;
    now.extensions.insert(
        "cognition.registry".to_string(),
        serde_json::to_value(&update.registry)
            .unwrap_or_else(|error| serde_json::json!({"error": error.to_string()})),
    );
    if let Some(response) = update.response {
        now.extensions.insert(
            "cognition.describe_scene.last_response".to_string(),
            serde_json::to_value(&response)
                .unwrap_or_else(|error| serde_json::json!({"error": error.to_string()})),
        );
        if let Some(failure) = response.response.failure {
            now.extensions.insert(
                "vision.image_enrichment_error".to_string(),
                serde_json::json!(failure),
            );
        }
    }
    if let Some(enrichment) = update.enrichment {
        now.eye
            .image_description_vectors
            .push(enrichment.image_description_vector);
        now.eye.scene_vectors.push(enrichment.scene_vector);
        now.extensions.insert(
            "vision.latest_image_description".to_string(),
            serde_json::json!({
                "text": enrichment.description,
                "source_frame_id": now
                    .eye
                    .scene_vectors
                    .last()
                    .and_then(|vector| vector.source_frame_id.clone()),
                "scene_vector_count": now.eye.scene_vectors.len(),
                "image_description_vector_count": now.eye.image_description_vectors.len(),
            }),
        );
    }
}

async fn poll_sensors_lossy(
    sensors: &mut [Box<dyn SenseProducer + Send>],
    health: &mut Vec<SensorPollHealth>,
) -> Vec<pete_sensors::SensePacket> {
    let mut packets = Vec::new();
    if health.len() != sensors.len() {
        health.clear();
        health.extend(sensors.iter().map(|sensor| SensorPollHealth {
            name: sensor.source_name().to_string(),
            ..SensorPollHealth::default()
        }));
    }
    for (sensor, health) in sensors.iter_mut().zip(health.iter_mut()) {
        let now_ms = wall_time_ms();
        match tokio::time::timeout(std::time::Duration::from_millis(25), sensor.poll()).await {
            Ok(Ok(packet)) => {
                if health.consecutive_failures > 0 {
                    eprintln!(
                        "optional sensor {} recovered after {} failed polls; brainstem body evidence remained active",
                        health.name, health.consecutive_failures
                    );
                }
                health.available = true;
                health.consecutive_failures = 0;
                health.last_error = None;
                health.last_success_ms = now_ms;
                packets.push(packet);
            }
            Ok(Err(error)) => record_optional_sensor_failure(health, error.to_string(), now_ms),
            Err(_) => record_optional_sensor_failure(health, "poll timed out".to_string(), now_ms),
        }
    }
    packets
}

fn record_optional_sensor_failure(health: &mut SensorPollHealth, error: String, now_ms: TimeMs) {
    health.available = false;
    health.consecutive_failures = health.consecutive_failures.saturating_add(1);
    health.last_error = Some(error.clone());
    if health.last_report_ms == 0
        || now_ms.saturating_sub(health.last_report_ms) >= SENSOR_FAILURE_REPORT_INTERVAL_MS
    {
        eprintln!(
            "optional sensor {} unavailable; continuing with brainstem body evidence: {} ({} failed polls; repeated reports suppressed for 30s)",
            health.name, error, health.consecutive_failures
        );
        health.last_report_ms = now_ms;
    }
}

fn insert_sensor_health(now: &mut Now, health: &[SensorPollHealth]) {
    now.extensions.insert(
        "sensor.health".to_string(),
        serde_json::Value::Array(
            health
                .iter()
                .map(|health| {
                    serde_json::json!({
                        "name": health.name,
                        "available": health.available,
                        "consecutive_failures": health.consecutive_failures,
                        "last_error": health.last_error,
                        "last_success_ms": health.last_success_ms,
                        "body_evidence_independent": true,
                    })
                })
                .collect(),
        ),
    );
}

fn wall_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn final_motor_from_tick(tick: &RuntimeTick) -> MotorCommand {
    tick.frame
        .now
        .extensions
        .get("motor_gate")
        .and_then(|value| value.get("final_motor"))
        .and_then(|value| serde_json::from_value::<MotorCommand>(value.clone()).ok())
        .unwrap_or_else(|| action_to_motor_command(tick.chosen_action.as_ref()))
}

fn annotate_snapshot_from_tick(snapshot: &mut WorldSnapshot, tick: &RuntimeTick) {
    snapshot.final_selected_action = tick.chosen_action.clone();
    snapshot.llm_action_proposal = tick
        .frame
        .now
        .extensions
        .get("llm.action_proposal")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok());
    snapshot.action_debug = tick
        .frame
        .now
        .extensions
        .get("action.motion_bridge")
        .cloned();
    if let Some(possession) = tick.frame.now.extensions.get("brainstem.possession") {
        let debug = snapshot
            .action_debug
            .get_or_insert_with(|| serde_json::json!({}));
        if !debug.is_object() {
            *debug = serde_json::json!({});
        }
        debug
            .as_object_mut()
            .expect("action debug was normalized to an object")
            .insert("brainstem_possession".to_string(), possession.clone());
    }
}

fn motor_command_to_motion(motor: MotorCommand) -> MotionCommand {
    if is_near_zero_motor(motor) {
        MotionCommand::Stop
    } else if motor.turn == 0.0 {
        MotionCommand::Forward {
            speed_m_s: motor.forward,
        }
    } else if motor.forward == 0.0 {
        MotionCommand::Turn {
            turn_rad_s: motor.turn,
        }
    } else {
        MotionCommand::Drive {
            forward_m_s: motor.forward,
            turn_rad_s: motor.turn,
        }
    }
}

fn apply_safe_cockpit_motor<C: Cockpit>(
    cockpit: &mut SafeCockpit<C>,
    motor: MotorCommand,
) -> Result<()> {
    if is_near_zero_motor(motor) {
        cockpit.stop().map_err(anyhow::Error::from)
    } else {
        cockpit
            .pulse_motion(
                pete_cockpit::meters_per_second_to_mm_s(motor.forward),
                pete_cockpit::radians_per_second_to_mrad_s(motor.turn),
            )
            .map_err(anyhow::Error::from)
    }
}

fn apply_slow_possession_motor<C: Cockpit>(
    cockpit: &mut SafeCockpit<C>,
    motor: MotorCommand,
) -> Result<Option<SlowPossessionMotionBlock>> {
    if is_near_zero_motor(motor) {
        cockpit.client_mut().stop().map_err(anyhow::Error::from)?;
        return Ok(None);
    }
    match cockpit
        .pulse_motion(
            pete_cockpit::meters_per_second_to_mm_s(motor.forward),
            pete_cockpit::radians_per_second_to_mrad_s(motor.turn),
        )
        .map_err(anyhow::Error::from)
    {
        Ok(()) => Ok(None),
        Err(error) if is_missed_events_error(&error) => {
            eprintln!(
                "slow possession recovered from event history gap during motion safety poll; stopping before continuing"
            );
            cockpit.client_mut().stop().map_err(anyhow::Error::from)?;
            Ok(None)
        }
        Err(error) if motion_stopped_latch(&error).is_some() => {
            let latch = motion_stopped_latch(&error);
            eprintln!("{error}; slow possession stopping before continuing");
            cockpit.client_mut().stop().map_err(anyhow::Error::from)?;
            Ok(latch.map(SlowPossessionMotionBlock::SafetyLatch))
        }
        Err(error) if command_rejection(&error).is_some() => {
            let (command_id, reason) = command_rejection(&error).expect("rejection was present");
            eprintln!("{error}; slow possession stopping before continuing");
            cockpit.client_mut().stop().map_err(anyhow::Error::from)?;
            Ok(Some(SlowPossessionMotionBlock::CommandRejected {
                command_id,
                reason,
            }))
        }
        Err(error) => Err(error),
    }
}

#[derive(Clone, Debug)]
enum SlowPossessionMotionBlock {
    SafetyLatch(SafetyLatchKind),
    CommandRejected { command_id: u32, reason: String },
}

fn is_missed_events_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<pete_cockpit::CockpitError>()
            .is_some_and(|cockpit| {
                matches!(cockpit, pete_cockpit::CockpitError::MissedEvents { .. })
            })
    })
}

fn motion_stopped_latch(error: &anyhow::Error) -> Option<SafetyLatchKind> {
    error.chain().find_map(|cause| {
        let cockpit = cause.downcast_ref::<pete_cockpit::CockpitError>()?;
        match cockpit {
            pete_cockpit::CockpitError::MotionStopped { reasons } => {
                latch_from_stop_reasons(reasons)
            }
            pete_cockpit::CockpitError::Policy(reason)
                if reason.starts_with("motion stopped by ") =>
            {
                Some(SafetyLatchKind::Heartbeat)
            }
            _ => None,
        }
    })
}

fn command_rejection(error: &anyhow::Error) -> Option<(u32, String)> {
    error.chain().find_map(|cause| {
        let cockpit = cause.downcast_ref::<pete_cockpit::CockpitError>()?;
        match cockpit {
            pete_cockpit::CockpitError::Rejected { command_id, reason } => {
                Some((*command_id, reason.clone()))
            }
            _ => None,
        }
    })
}

fn latch_from_stop_reasons(reasons: &[SafeStopReason]) -> Option<SafetyLatchKind> {
    reasons
        .iter()
        .find_map(|reason| match reason {
            SafeStopReason::SafetyTripped { latch } => *latch,
            _ => None,
        })
        .or_else(|| {
            reasons.iter().find_map(|reason| match reason {
                SafeStopReason::HeartbeatExpired => Some(SafetyLatchKind::Heartbeat),
                _ => None,
            })
        })
}

fn apply_safe_cockpit_motion<C: Cockpit>(
    cockpit: &mut SafeCockpit<C>,
    motion: &MotionCommand,
) -> Result<()> {
    apply_safe_cockpit_motor(cockpit, motion.to_motor_command())
}

pub fn body_sense_from_cockpit_status(status: StatusSummary, last_update_ms: TimeMs) -> BodySense {
    let charging = status.battery.charging_state.unwrap_or(0) != 0
        || status.battery.charging_indicator.unwrap_or(false);
    let home_base = status.battery.home_base();
    let packet_update_ms = status
        .body_packet_age_ms
        .filter(|_| status.body_packet_complete == Some(true))
        .map(|age_ms| last_update_ms.saturating_sub(u64::from(age_ms)))
        .unwrap_or(if charging { last_update_ms } else { 0 });
    BodySense {
        battery_level: status
            .battery
            .percent
            .map(|percent| percent as f32 / 100.0)
            .unwrap_or(1.0),
        charging,
        infrared_character: status.infrared_character.unwrap_or(0),
        flags: BodyFlags {
            // Create 1 can report its dock contacts as both bumpers plus all
            // four cliff bits. Packet 34 is the authoritative Home Base
            // discriminator. Keep those raw bits in Cockpit status, but do
            // not promote dock geometry into upstream collision evidence.
            bump_left: !home_base && status.contact.bump_left.unwrap_or(false),
            bump_right: !home_base && status.contact.bump_right.unwrap_or(false),
            cliff_left: !home_base && status.contact.cliff_left.unwrap_or(false),
            cliff_front_left: !home_base && status.contact.cliff_front_left.unwrap_or(false),
            cliff_front_right: !home_base && status.contact.cliff_front_right.unwrap_or(false),
            cliff_right: !home_base && status.contact.cliff_right.unwrap_or(false),
            wheel_drop: status.contact.wheel_drop.unwrap_or(false),
            wall: status.contact.wall.unwrap_or(false),
            virtual_wall: status.contact.virtual_wall.unwrap_or(false),
        },
        odometry: Pose2 {
            x_m: status.odometry.distance_mm.unwrap_or(0) as f32 / 1000.0,
            y_m: 0.0,
            heading_rad: status.odometry.heading_mrad.unwrap_or(0) as f32 / 1000.0,
        },
        last_update_ms: packet_update_ms,
        ..BodySense::default()
    }
}

fn safety_latch_kind_from_event_code(code: u32) -> Option<SafetyLatchKind> {
    match code {
        1 => Some(SafetyLatchKind::Bump),
        2 => Some(SafetyLatchKind::Cliff),
        3 => Some(SafetyLatchKind::WheelDrop),
        5 => Some(SafetyLatchKind::Heartbeat),
        6 => Some(SafetyLatchKind::Tilt),
        7 => Some(SafetyLatchKind::Impact),
        8 => Some(SafetyLatchKind::Charging),
        _ => None,
    }
}

fn infer_safety_latch_from_sensors(body: &BodySense) -> Option<SafetyLatchKind> {
    if bump_active(body) {
        Some(SafetyLatchKind::Bump)
    } else if cliff_active(body) {
        Some(SafetyLatchKind::Cliff)
    } else if body.flags.wheel_drop {
        Some(SafetyLatchKind::WheelDrop)
    } else if body.charging {
        Some(SafetyLatchKind::Charging)
    } else {
        None
    }
}

fn bump_active(body: &BodySense) -> bool {
    body.flags.bump_left || body.flags.bump_right
}

fn cliff_active(body: &BodySense) -> bool {
    body.flags.cliff_left
        || body.flags.cliff_front_left
        || body.flags.cliff_front_right
        || body.flags.cliff_right
}

fn recovery_turn_direction_for_latch(kind: SafetyLatchKind, body: &BodySense) -> TurnDir {
    match kind {
        SafetyLatchKind::Bump => contact_turn_direction(body),
        SafetyLatchKind::Cliff => {
            if body.flags.cliff_left || body.flags.cliff_front_left {
                TurnDir::Right
            } else if body.flags.cliff_right || body.flags.cliff_front_right {
                TurnDir::Left
            } else {
                TurnDir::Left
            }
        }
        _ => TurnDir::Left,
    }
}

fn contact_turn_direction(body: &BodySense) -> TurnDir {
    match (body.flags.bump_left, body.flags.bump_right) {
        (true, false) => TurnDir::Right,
        (false, true) => TurnDir::Left,
        _ => TurnDir::Left,
    }
}

fn imu_recovery_clear(status: &StatusSummary, latch: SafetyLatchKind) -> bool {
    match latch {
        SafetyLatchKind::Tilt => status
            .imu
            .tilt_magnitude_mrad
            .is_none_or(|value| value < 650),
        SafetyLatchKind::Impact => status
            .imu
            .impact_score_mm_s2
            .is_none_or(|value| value < 18_000),
        _ => false,
    }
}

fn possession_recovery_debug(
    state: &PossessionRecoveryState,
    active_latch: Option<SafetyLatchKind>,
    command_sent: bool,
) -> serde_json::Value {
    let latch = active_latch.or(state.latch);
    let intended_motion = match (state.phase, latch) {
        (PossessionRecoveryPhase::BrainstemReflex, _) => {
            serde_json::json!({"owner": "brainstem_contact_withdrawal"})
        }
        (
            PossessionRecoveryPhase::WaitingForSensorClear | PossessionRecoveryPhase::Escaping,
            Some(SafetyLatchKind::Bump | SafetyLatchKind::Cliff),
        ) => serde_json::json!({
            "linear": "reverse",
            "ttl_ms": POSSESSION_ESCAPE_TTL_MS,
        }),
        (_, Some(_)) => serde_json::json!({"linear": "stop"}),
        _ => serde_json::Value::Null,
    };
    serde_json::json!({
        "latched": latch.map(|latch| format!("{latch:?}")),
        "hazard_generation": state.hazard_generation,
        "phase": format!("{:?}", state.phase),
        "turn_direction": format!("{:?}", state.turn_direction),
        "active_since_ms": state.active_since_ms,
        "last_command_ms": state.last_command_ms,
        "command_attempts": state.command_attempts,
        "stuck_stop_sent": state.stuck_stop_sent,
        "brainstem_reflex_observed": state.brainstem_reflex_observed,
        "last_reflex_outcome": state.last_reflex_outcome.map(|outcome| format!("{outcome:?}")),
        "command_sent": command_sent,
        "intended_motion": intended_motion,
        "commanded_motion": if command_sent && state.phase == PossessionRecoveryPhase::Escaping {
            serde_json::json!({
                "linear": "reverse",
                "ttl_ms": POSSESSION_ESCAPE_TTL_MS,
            })
        } else if command_sent {
            serde_json::json!({"linear": "stop"})
        } else {
            serde_json::Value::Null
        },
        "observed_motion": {
            "linear_displacement_m": state.observed_linear_m,
            "heading_change_rad": state.observed_turn_rad,
        },
    })
}

fn motion_rejection_debug(state: &MotionRejectionState) -> serde_json::Value {
    serde_json::json!({
        "first_ms": state.first_ms,
        "last_ms": state.last_ms,
        "blocked_until_ms": state.blocked_until_ms,
        "latest_command_id": state.latest_command_id,
        "latest_reason": state.latest_reason,
        "count": state.count,
        "stuck": state.stuck,
        "stuck_stop_sent": state.stuck_stop_sent,
    })
}

fn synthetic_slow_manual_tick(
    mut now: Now,
    input: ReignInput,
    desired_motor: MotorCommand,
    final_motor: MotorCommand,
    block_reason: Option<String>,
    body_before: &pete_body::BodySense,
) -> Result<RuntimeTick> {
    let action = input.command.to_action();
    let motor_applied = !is_near_zero_motor(final_motor);
    now.extensions.insert(
        "action.motion_bridge".to_string(),
        serde_json::json!({
            "selected_action": action,
            "chosen_action": action,
            "desired_motor": desired_motor,
            "final_motor": final_motor,
            "motor_applied": motor_applied,
            "manual_hardware_gate": true,
            "body_pose_before": pose_json(body_before),
            "body_pose_after": pose_json(&now.body),
            "motion_sent_to_robot": motor_command_to_motion(final_motor),
            "motion_sent_to_sim": motor_command_to_motion(final_motor),
            "why_not_moving": block_reason,
            "runtime_bypassed": true,
        }),
    );
    let experience = Experience::new(
        "real_robot_slow_manual",
        "Direct WebRemote slow hardware command.",
        Vec::new(),
        Vec::new(),
        now.t_ms,
        now.t_ms,
    );
    Ok(RuntimeTick {
        frame: ExperienceFrame {
            id: Uuid::new_v4(),
            t_ms: now.t_ms,
            now,
            sensations: Vec::new(),
            impressions: Vec::new(),
            experiences: vec![experience.clone()],
            z: Some(ExperienceLatent::default()),
            chosen_action: action.clone(),
            conscious_command: None,
            reign_input: Some(input),
            reign_outcome: None,
            predicted_futures: Vec::new(),
            behavior_runs: Vec::new(),
            actual_next: None,
            reward: Reward::default(),
            surprise: SurpriseSense::default(),
            memory_recall: Vec::new(),
            recollections: Vec::new(),
            llm_teaching: Vec::new(),
            counterfactuals: Vec::new(),
            notes: vec!["RealSlowManualRuntimeBypass: direct hardware command".to_string()],
        },
        experience,
        chosen_action: action,
        skill_request: None,
        skill_status: None,
        recall: RecallBundle::default(),
        llm: LlmTickResult::default(),
        combobulation: None,
        inline_learning: InlineLearningTickStatus::default(),
    })
}

fn reign_input_drives_real_slow(input: &ReignInput) -> bool {
    if !matches!(input.source, ReignSource::WebRemote | ReignSource::Gamepad)
        || input.mode != ReignMode::Direct
    {
        return false;
    }
    matches!(
        input.command,
        pete_actions::ReignCommand::Go { .. }
            | pete_actions::ReignCommand::Reverse { .. }
            | pete_actions::ReignCommand::Drive { .. }
            | pete_actions::ReignCommand::Turn { .. }
            | pete_actions::ReignCommand::Stop
    )
}

fn reign_input_outputs_real_slow_directly(input: &ReignInput) -> bool {
    if input.source != ReignSource::WebRemote || input.mode != ReignMode::Direct {
        return false;
    }
    matches!(
        input.command,
        pete_actions::ReignCommand::Speak { .. } | pete_actions::ReignCommand::Chirp { .. }
    )
}

fn real_slow_body_block_reason(body: &pete_body::BodySense) -> Option<String> {
    if body.charging {
        return Some("charging active".to_string());
    }
    if body.flags.wheel_drop {
        return Some("wheel drop active".to_string());
    }
    if body.flags.cliff_left
        || body.flags.cliff_front_left
        || body.flags.cliff_front_right
        || body.flags.cliff_right
    {
        return Some("cliff sensor active".to_string());
    }
    if body.battery_level <= 0.10 && !body.charging {
        return Some("battery is critical".to_string());
    }
    None
}

fn real_slow_motor_block_reason(
    body: &pete_body::BodySense,
    tick: &RuntimeTick,
    manual_drive: bool,
    autonomous_motion: bool,
) -> Option<String> {
    if !manual_drive && !autonomous_motion {
        return Some(
            "real slow mode requires active WebRemote/Gamepad Direct command or explicit autonomous motion authorization"
                .to_string(),
        );
    }
    if let Some(reason) = real_slow_body_block_reason(body) {
        return Some(reason);
    }
    tick.frame
        .now
        .extensions
        .get("motor_gate")
        .and_then(|value| value.get("safety_reason"))
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

fn pose_json(body: &pete_body::BodySense) -> serde_json::Value {
    serde_json::json!({
        "x_m": body.odometry.x_m,
        "y_m": body.odometry.y_m,
        "heading_rad": body.odometry.heading_rad,
    })
}

fn movement_delta_m(before: &pete_body::BodySense, after: &pete_body::BodySense) -> f32 {
    distance_between_points(
        (before.odometry.x_m, before.odometry.y_m),
        (after.odometry.x_m, after.odometry.y_m),
    )
}

fn not_moving_reason(
    final_motor: MotorCommand,
    motion: &MotionCommand,
    before: &pete_body::BodySense,
    after: &pete_body::BodySense,
    movement_delta: f32,
    reset_or_dead: bool,
    tick: &RuntimeTick,
) -> Option<String> {
    if movement_delta >= 0.005 {
        return None;
    }
    if reset_or_dead {
        return Some("dead battery or stuck reset prevented motor application".to_string());
    }
    if is_near_zero_motor(final_motor) || matches!(motion, MotionCommand::Stop) {
        return Some(
            tick.frame
                .now
                .extensions
                .get("motor_gate")
                .and_then(|value| value.get("safety_reason"))
                .and_then(|value| value.as_str())
                .map(|reason| format!("final motor was stop: {reason}"))
                .unwrap_or_else(|| "final motor was stop or near zero".to_string()),
        );
    }
    if after.flags.wall || after.flags.bump_left || after.flags.bump_right {
        return Some("sim collision blocked commanded motion".to_string());
    }
    if before.odometry.x_m == after.odometry.x_m
        && before.odometry.y_m == after.odometry.y_m
        && before.odometry.heading_rad != after.odometry.heading_rad
    {
        return Some("turn-only motion changed heading without translation".to_string());
    }
    Some("non-stop motion was sent but pose delta was near zero".to_string())
}

pub struct SimRunner<R> {
    pub runtime: R,
    pub world: VirtualWorld,
    pub cockpit: SafeCockpit<SimCockpit>,
    pub tick_count: usize,
    pub tick_ms: u64,
    stuck: StuckRecoveryController,
    possessor_skills: PossessorSkillRuntime,
}

const STUCK_LOW_DISPLACEMENT_TICKS: usize = 6;
const STUCK_WINDOW_DISPLACEMENT_EPSILON_M: f32 = 0.015;
const NEAR_ARENA_WALL_M: f32 = 0.32;
const SAME_TRAP_RADIUS_M: f32 = 0.18;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum RecoveryPhase {
    #[default]
    None,
    Stop,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum TrapKind {
    #[default]
    Unknown,
    Wall,
    Corner,
    Column,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct StuckStatus {
    active: bool,
    corner_trap: bool,
    trap_kind: TrapKind,
    stuck_ticks: usize,
    duration_ticks: usize,
    phase: RecoveryPhase,
    turn_sign: f32,
    recovery_attempts: usize,
    repeated_trap_count: usize,
    clearance_m: Option<f32>,
    event_started: bool,
    recovered: bool,
    dead_battery: bool,
    reset_due: bool,
}

#[derive(Clone, Debug, Default)]
struct StuckRecoveryController {
    last_position: Option<(f32, f32)>,
    displacement_window: VecDeque<f32>,
    commanded_window: VecDeque<bool>,
    stuck_ticks: usize,
    active: bool,
    corner_trap: bool,
    trap_kind: TrapKind,
    duration_ticks: usize,
    phase: RecoveryPhase,
    phase_ticks_remaining: usize,
    turn_sign: f32,
    recovery_attempts: usize,
    repeated_trap_count: usize,
    trap_anchor: Option<(f32, f32)>,
    last_failed_turn_sign: Option<f32>,
    clearance_m: Option<f32>,
    event_started: bool,
    recovered: bool,
    dead_battery: bool,
    reset_due: bool,
}

impl StuckRecoveryController {
    fn annotate_snapshot(&mut self, snapshot: &mut WorldSnapshot, tick_ms: u64) {
        self.dead_battery = is_dead_battery(snapshot);
        snapshot
            .extensions
            .retain(|extension| extension.name != "sim.stuck");
        snapshot.extensions.push(self.extension(tick_ms));
        self.event_started = false;
        self.recovered = false;
    }

    fn observe(&mut self, snapshot: &WorldSnapshot, action: Option<&ActionPrimitive>) {
        let was_active = self.active;
        let position = (snapshot.body.odometry.x_m, snapshot.body.odometry.y_m);
        let step_distance = self
            .last_position
            .map(|last| distance_between_points(last, position))
            .unwrap_or(f32::INFINITY);
        let commanded_motion = action_is_commanded_motion(action);
        self.dead_battery = is_dead_battery(snapshot);
        self.push_motion_sample(step_distance, commanded_motion);
        self.clearance_m = snapshot.range.nearest_m;
        let trap_kind = classify_trap_kind(snapshot);
        let trapped = trap_kind != TrapKind::Unknown;
        let stationary_column_or_corner = matches!(trap_kind, TrapKind::Column | TrapKind::Corner)
            && self.rolling_stationary()
            && matches!(action, Some(ActionPrimitive::Stop) | None);
        let low_displacement = (self.rolling_low_displacement() || stationary_column_or_corner)
            && !snapshot.body.charging;
        self.stuck_ticks = if low_displacement {
            self.stuck_ticks
                .saturating_add(1)
                .max(STUCK_LOW_DISPLACEMENT_TICKS)
        } else {
            if !self.active && step_distance > STUCK_WINDOW_DISPLACEMENT_EPSILON_M {
                self.recovery_attempts = 0;
            }
            0
        };

        if !self.active
            && (commanded_motion || stationary_column_or_corner)
            && self.stuck_ticks >= STUCK_LOW_DISPLACEMENT_TICKS
            && trapped
            && !self.dead_battery
        {
            self.active = true;
            self.corner_trap = trap_kind == TrapKind::Corner;
            self.trap_kind = trap_kind;
            self.duration_ticks = 1;
            self.phase = RecoveryPhase::Stop;
            self.phase_ticks_remaining = 1;
            if self
                .trap_anchor
                .map(|anchor| distance_between_points(anchor, position) <= SAME_TRAP_RADIUS_M)
                .unwrap_or(false)
            {
                self.repeated_trap_count = self.repeated_trap_count.saturating_add(1);
            } else {
                self.repeated_trap_count = 0;
                self.trap_anchor = Some(position);
                self.recovery_attempts = 0;
                self.last_failed_turn_sign = None;
            }
            self.recovery_attempts = self.recovery_attempts.saturating_add(1);
            self.turn_sign = recovery_turn_sign(snapshot, self.last_failed_turn_sign);
            self.event_started = true;
        } else if was_active {
            self.duration_ticks = self.duration_ticks.saturating_add(1);
        }

        let recovery_displacement = self
            .trap_anchor
            .map(|anchor| distance_between_points(anchor, position))
            .unwrap_or(0.0);
        if self.active && recovery_displacement >= STUCK_WINDOW_DISPLACEMENT_EPSILON_M {
            self.finish_recovery_success();
        }

        self.last_position = Some(position);
    }

    fn finish_recovery_success(&mut self) {
        self.active = false;
        self.corner_trap = false;
        self.trap_kind = TrapKind::Unknown;
        self.stuck_ticks = 0;
        self.phase = RecoveryPhase::None;
        self.phase_ticks_remaining = 0;
        self.trap_anchor = None;
        self.last_failed_turn_sign = None;
        self.recovered = true;
        self.displacement_window.clear();
        self.commanded_window.clear();
    }

    fn push_motion_sample(&mut self, step_distance: f32, commanded_motion: bool) {
        if step_distance.is_finite() {
            self.displacement_window.push_back(step_distance.max(0.0));
            self.commanded_window.push_back(commanded_motion);
        }
        while self.displacement_window.len() > STUCK_LOW_DISPLACEMENT_TICKS {
            self.displacement_window.pop_front();
        }
        while self.commanded_window.len() > STUCK_LOW_DISPLACEMENT_TICKS {
            self.commanded_window.pop_front();
        }
    }

    fn rolling_low_displacement(&self) -> bool {
        self.displacement_window.len() >= STUCK_LOW_DISPLACEMENT_TICKS
            && self.commanded_window.len() >= STUCK_LOW_DISPLACEMENT_TICKS
            && self.commanded_window.iter().all(|commanded| *commanded)
            && self.displacement_window.iter().sum::<f32>() < STUCK_WINDOW_DISPLACEMENT_EPSILON_M
    }

    fn rolling_stationary(&self) -> bool {
        self.displacement_window.len() >= STUCK_LOW_DISPLACEMENT_TICKS
            && self.displacement_window.iter().sum::<f32>() < STUCK_WINDOW_DISPLACEMENT_EPSILON_M
    }

    fn extension(&self, tick_ms: u64) -> ExtensionSense {
        let status = self.status();
        ExtensionSense {
            schema_version: 1,
            name: "sim.stuck".to_string(),
            values: vec![
                status.active as u8 as f32,
                status.corner_trap as u8 as f32,
                status.stuck_ticks as f32,
                (status.duration_ticks as u64).saturating_mul(tick_ms) as f32,
                recovery_phase_code(status.phase),
                status.turn_sign,
                status.event_started as u8 as f32,
                status.recovered as u8 as f32,
                status.dead_battery as u8 as f32,
                status.reset_due as u8 as f32,
                trap_kind_code(status.trap_kind),
                status.recovery_attempts as f32,
                status.repeated_trap_count as f32,
                status.clearance_m.unwrap_or(-1.0),
            ],
        }
    }

    fn status(&self) -> StuckStatus {
        StuckStatus {
            active: self.active,
            corner_trap: self.corner_trap,
            trap_kind: self.trap_kind,
            stuck_ticks: self.stuck_ticks,
            duration_ticks: self.duration_ticks,
            phase: self.phase,
            turn_sign: self.turn_sign,
            recovery_attempts: self.recovery_attempts,
            repeated_trap_count: self.repeated_trap_count,
            clearance_m: self.clearance_m,
            event_started: self.event_started,
            recovered: self.recovered,
            dead_battery: self.dead_battery,
            reset_due: self.reset_due,
        }
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

fn recovery_phase_code(phase: RecoveryPhase) -> f32 {
    match phase {
        RecoveryPhase::None => 0.0,
        RecoveryPhase::Stop => 1.0,
    }
}

fn trap_kind_code(kind: TrapKind) -> f32 {
    match kind {
        TrapKind::Unknown => 0.0,
        TrapKind::Wall => 1.0,
        TrapKind::Corner => 2.0,
        TrapKind::Column => 3.0,
    }
}

fn action_is_commanded_motion(action: Option<&ActionPrimitive>) -> bool {
    let motor = action_to_motor_command(action);
    motor.forward.abs() > 0.05 || motor.turn.abs() > 0.05
}

fn classify_trap_kind(snapshot: &WorldSnapshot) -> TrapKind {
    let body = &snapshot.body;
    let collision = body.flags.wall
        || body.flags.bump_left
        || body.flags.bump_right
        || body.flags.cliff_front_left
        || body.flags.cliff_front_right;
    let near = snapshot.range.nearest_m.unwrap_or(10.0) < NEAR_ARENA_WALL_M;
    let near_arena_wall = arena_bounds(snapshot)
        .map(|(width_m, height_m)| {
            snapshot.body.odometry.x_m < NEAR_ARENA_WALL_M
                || snapshot.body.odometry.y_m < NEAR_ARENA_WALL_M
                || width_m - snapshot.body.odometry.x_m < NEAR_ARENA_WALL_M
                || height_m - snapshot.body.odometry.y_m < NEAR_ARENA_WALL_M
        })
        .unwrap_or(false);
    let beams = &snapshot.range.beams;
    let (left, _center, right) = beam_clearance_buckets(beams);
    let side_constrained = left < 0.16 && right < 0.16;
    if near_arena_wall && side_constrained {
        TrapKind::Corner
    } else if near_arena_wall || body.flags.wall {
        TrapKind::Wall
    } else if collision || near {
        TrapKind::Column
    } else {
        TrapKind::Unknown
    }
}

fn recovery_turn_sign(snapshot: &WorldSnapshot, last_failed_turn_sign: Option<f32>) -> f32 {
    if let Some(sign) = bump_escape_turn_sign(snapshot) {
        return sign;
    }
    if let Some(last_failed) = last_failed_turn_sign {
        return -last_failed;
    }
    turn_toward_clearer_side(snapshot)
}

fn bump_escape_turn_sign(snapshot: &WorldSnapshot) -> Option<f32> {
    match (
        snapshot.body.flags.bump_left,
        snapshot.body.flags.bump_right,
        snapshot.body.flags.wall,
    ) {
        (true, false, _) => Some(-1.0),
        (false, true, _) => Some(1.0),
        (_, _, true) | (true, true, _) => Some(turn_toward_clearer_side(snapshot)),
        _ => None,
    }
}

fn turn_toward_clearer_side(snapshot: &WorldSnapshot) -> f32 {
    let beams = &snapshot.range.beams;
    if beams.len() < 2 {
        return 1.0;
    }
    let (left, _, right) = beam_clearance_buckets(beams);
    if left <= right {
        -1.0
    } else {
        1.0
    }
}

fn beam_clearance_buckets(beams: &[f32]) -> (f32, f32, f32) {
    if beams.is_empty() {
        return (1.0, 1.0, 1.0);
    }
    let third = (beams.len() / 3).max(1);
    let left_end = third.min(beams.len());
    let right_start = beams.len().saturating_sub(third);
    let center_start = left_end.saturating_sub(1).min(beams.len());
    let center_end = (right_start + 1).min(beams.len()).max(center_start + 1);
    let left = beams[..left_end].iter().copied().fold(1.0, f32::min);
    let center = beams[center_start..center_end]
        .iter()
        .copied()
        .fold(1.0, f32::min);
    let right = beams[right_start..].iter().copied().fold(1.0, f32::min);
    (left, center, right)
}

fn arena_bounds(snapshot: &WorldSnapshot) -> Option<(f32, f32)> {
    let world = snapshot
        .extensions
        .iter()
        .find(|extension| extension.name == "sim.world")?;
    let width_m = world.values.first().copied()?;
    let height_m = world.values.get(1).copied()?;
    (width_m > 0.0 && height_m > 0.0).then_some((width_m, height_m))
}

fn is_dead_battery(snapshot: &WorldSnapshot) -> bool {
    snapshot.body.battery_level <= f32::EPSILON && !snapshot.body.charging
}

fn sim_stuck_reset_due(snapshot: &WorldSnapshot) -> bool {
    snapshot
        .extensions
        .iter()
        .find(|extension| extension.name == "sim.stuck")
        .and_then(|extension| extension.values.get(9))
        .copied()
        .unwrap_or(0.0)
        > 0.0
}

fn distance_between_points(left: (f32, f32), right: (f32, f32)) -> f32 {
    let dx = left.0 - right.0;
    let dy = left.1 - right.1;
    (dx * dx + dy * dy).sqrt()
}

impl<R> SimRunner<R>
where
    R: RuntimeLoop + Send,
{
    pub fn new(runtime: R, world: VirtualWorld, motors: SimCockpit) -> Self {
        Self {
            runtime,
            world,
            cockpit: SafeCockpit::new(motors),
            tick_count: 0,
            tick_ms: 100,
            stuck: StuckRecoveryController::default(),
            possessor_skills: PossessorSkillRuntime::default(),
        }
    }

    pub async fn run_steps(&mut self, steps: usize) -> Result<()> {
        self.run_steps_observing(steps, |_| {}).await
    }

    pub async fn run_steps_observing<F>(&mut self, steps: usize, mut observe: F) -> Result<()>
    where
        F: FnMut(&WorldSnapshot),
    {
        self.run_steps_observing_ticks(steps, |snapshot, _tick| observe(snapshot))
            .await
    }

    pub async fn run_steps_observing_ticks<F>(&mut self, steps: usize, mut observe: F) -> Result<()>
    where
        F: FnMut(&WorldSnapshot, &RuntimeTick),
    {
        for _ in 0..steps {
            let mut snapshot = self.world.snapshot().await?;
            self.stuck.annotate_snapshot(&mut snapshot, self.tick_ms);
            let reset_after_tick = sim_stuck_reset_due(&snapshot);
            let body_pose_before = snapshot.body.clone();
            let mut now = snapshot.to_now(snapshot.body.last_update_ms);
            self.possessor_skills.annotate_now(&mut now);
            let mut tick = self
                .runtime
                .tick(now.clone(), ExperienceLatent::default(), Vec::new())
                .await?;
            let mut lua_skill_owns_motion = false;
            let selected_skill_request = self
                .possessor_skills
                .request_for_tick(tick.skill_request.clone());
            if let Some(request) = selected_skill_request {
                tick.skill_request = Some(request.clone());
                let status_summary = self.cockpit.refresh_status()?;
                let events = self.cockpit.poll_events_allowing_history_gap()?;
                let (status, _) = self.possessor_skills.step(
                    self.cockpit.client_mut(),
                    &request,
                    &tick.frame.now,
                    &status_summary,
                    status_summary.battery.home_base(),
                    &events,
                    now.t_ms,
                );
                self.runtime.observe_skill_status(&status);
                tick.skill_status = Some(status);
                self.possessor_skills.annotate_now(&mut tick.frame.now);
                lua_skill_owns_motion = true;
            }
            let final_motor = if lua_skill_owns_motion {
                self.world
                    .last_motion_sent()
                    .unwrap_or(MotionCommand::Stop)
                    .to_motor_command()
            } else {
                final_motor_from_tick(&tick)
            };
            let mut motion = motor_command_to_motion(final_motor);
            let mut motion_sent_to_sim = Some(serde_json::to_value(&motion)?);
            let reset_or_dead = is_dead_battery(&snapshot) || reset_after_tick;
            self.stuck.observe(&snapshot, tick.chosen_action.as_ref());
            let manual_reign_driving = tick
                .frame
                .reign_input
                .as_ref()
                .map(reign_input_drives_sim_directly)
                .unwrap_or(false);
            let observed_stuck_extension = self.stuck.extension(self.tick_ms);
            if is_dead_battery(&snapshot) || reset_after_tick {
                self.world.reset_body_to_spawn();
                self.stuck.reset();
                motion = MotionCommand::Stop;
                motion_sent_to_sim = None;
            } else if !lua_skill_owns_motion {
                let _ = manual_reign_driving;
                apply_safe_cockpit_motion(&mut self.cockpit, &motion)?;
            };
            let mut after_snapshot = self.world.snapshot().await?;
            annotate_snapshot_from_tick(&mut after_snapshot, &tick);
            let movement_delta = movement_delta_m(&body_pose_before, &after_snapshot.body);
            let why_not_moving = not_moving_reason(
                final_motor,
                &motion,
                &body_pose_before,
                &after_snapshot.body,
                movement_delta,
                reset_or_dead,
                &tick,
            );
            let mut action_debug = after_snapshot
                .action_debug
                .take()
                .unwrap_or_else(|| serde_json::json!({}));
            if !action_debug.is_object() {
                action_debug = serde_json::json!({});
            }
            if let Some(object) = action_debug.as_object_mut() {
                object.insert("body_pose_before".to_string(), pose_json(&body_pose_before));
                object.insert(
                    "body_pose_after".to_string(),
                    pose_json(&after_snapshot.body),
                );
                object.insert(
                    "movement_delta".to_string(),
                    serde_json::json!(movement_delta),
                );
                object.insert(
                    "motion_sent_to_sim".to_string(),
                    motion_sent_to_sim.unwrap_or(serde_json::Value::Null),
                );
                object.insert(
                    "motor_applied".to_string(),
                    serde_json::json!(movement_delta >= 0.005),
                );
                object.insert(
                    "why_not_moving".to_string(),
                    why_not_moving
                        .clone()
                        .map(serde_json::Value::String)
                        .unwrap_or(serde_json::Value::Null),
                );
            }
            after_snapshot.action_debug = Some(action_debug);
            after_snapshot
                .extensions
                .retain(|extension| extension.name != "sim.stuck");
            after_snapshot.extensions.push(observed_stuck_extension);
            self.stuck.event_started = false;
            self.stuck.recovered = false;
            observe(&after_snapshot, &tick);
            self.tick_count = self.tick_count.saturating_add(1);
        }
        Ok(())
    }
}

fn predict_baseline_futures(
    predictor: &mut ReplaceableBehavior<FutureInput, FuturePrediction>,
    latent: &ExperienceLatent,
    t_ms: TimeMs,
) -> Result<(Vec<FuturePrediction>, Vec<ErasedBehaviorRunRecord>)> {
    let mut out = Vec::new();
    let mut records = Vec::new();
    for action in default_candidate_actions() {
        for offset_ms in [100, 500, 1_000, 5_000] {
            let input = FutureInput {
                latent: latent.clone(),
                action: action.clone(),
                offset_ms,
            };
            let run = predictor.infer(&input, t_ms)?;
            out.push(run.chosen);
            records.push(run.record.erase());
        }
    }
    Ok((out, records))
}

fn attach_structured_predictions_to_experience(
    experience: &mut Experience,
    futures: &[FuturePrediction],
    now: &Now,
    action: Option<&ActionPrimitive>,
) {
    for future in futures.iter().take(2) {
        let text = future
            .summary
            .clone()
            .unwrap_or_else(|| "latent future estimated from ExperienceLatent".to_string());
        experience.predictions.push(Prediction {
            offset_ms: future.offset_ms,
            text: format!("next_state: {text}"),
            confidence: future.confidence.clamp(0.0, 1.0),
            vector: None,
        });
    }

    let danger = now
        .predictions
        .danger_model
        .or(now.predictions.danger_hardcoded);
    if let Some(danger) = danger {
        experience.predictions.push(Prediction {
            offset_ms: 100,
            text: format!(
                "hazard: bump={:.2} cliff={:.2} wheel_drop={:.2} stuck={:.2}",
                danger.bump_risk, danger.cliff_risk, danger.wheel_drop_risk, danger.stuck_risk
            ),
            confidence: danger.confidence.clamp(0.0, 1.0),
            vector: None,
        });
    }

    let charge = now
        .predictions
        .charge_model
        .or(now.predictions.charge_hardcoded);
    if let Some(charge) = charge {
        experience.predictions.push(Prediction {
            offset_ms: 500,
            text: format!(
                "charge: probability={:.2} battery_delta={:.3} dock={:.2}",
                charge.charge_probability, charge.expected_battery_delta, charge.dock_likelihood
            ),
            confidence: charge.confidence.clamp(0.0, 1.0),
            vector: None,
        });
    }

    let action_value = now
        .predictions
        .action_values_model
        .iter()
        .chain(now.predictions.action_values_hardcoded.iter())
        .find(|prediction| {
            action
                .map(|action| prediction.action == *action)
                .unwrap_or(true)
        });
    if let Some(action_value) = action_value {
        experience.predictions.push(Prediction {
            offset_ms: 250,
            text: format!(
                "action_value: action={:?} value={:.2}",
                action_value.action, action_value.value
            ),
            confidence: action_value.confidence.clamp(0.0, 1.0),
            vector: None,
        });
    }

    if !now.predictions.expected_events.is_empty() {
        experience.predictions.push(Prediction {
            offset_ms: 500,
            text: format!(
                "social_object_changes: expected_events={}",
                now.predictions.expected_events.join(", ")
            ),
            confidence: (1.0 - now.predictions.uncertainty).clamp(0.0, 1.0),
            vector: None,
        });
    }

    experience.predictions.push(Prediction {
        offset_ms: 500,
        text: format!(
            "uncertainty: {:.2}",
            now.predictions.uncertainty.clamp(0.0, 1.0)
        ),
        confidence: (1.0 - now.predictions.uncertainty).clamp(0.05, 1.0),
        vector: None,
    });
}

fn default_candidate_actions() -> Vec<ActionPrimitive> {
    vec![
        ActionPrimitive::Stop,
        ActionPrimitive::Go {
            intensity: 0.15,
            duration_ms: 1_000,
        },
        ActionPrimitive::Go {
            intensity: -0.12,
            duration_ms: 750,
        },
        ActionPrimitive::Turn {
            direction: TurnDir::Left,
            intensity: 0.25,
            duration_ms: 750,
        },
        ActionPrimitive::Turn {
            direction: TurnDir::Right,
            intensity: 0.25,
            duration_ms: 750,
        },
        ActionPrimitive::Inspect {
            target: InspectTarget::Novelty,
        },
        ActionPrimitive::Approach {
            target: ApproachTarget::Charger,
        },
        ActionPrimitive::Dock,
        ActionPrimitive::Explore {
            style: ExploreStyle::Wander,
            duration_ms: 2_000,
        },
    ]
}

#[derive(Clone, Copy, Debug, Default)]
struct CandidateModelSignals {
    danger: Option<DangerOutput>,
    charge: Option<ChargeOutput>,
    action_value: Option<ActionValueOutput>,
}

fn score_action_candidate(
    now: &Now,
    action: &ActionPrimitive,
    signals: CandidateModelSignals,
    previous_action: Option<&ActionPrimitive>,
) -> ActionSelectionCandidateScore {
    let danger = signals
        .danger
        .map(max_danger_risk)
        .unwrap_or_else(|| fallback_collision_risk(now, action));
    let charge = signals.charge.map(charge_score).unwrap_or_else(|| {
        if matches!(
            action,
            ActionPrimitive::Dock
                | ActionPrimitive::Approach {
                    target: ApproachTarget::Charger
                }
        ) {
            now.memory.place_charge_value.max(0.1)
        } else {
            0.0
        }
    });
    let action_value = signals.action_value.map(|value| value.value).unwrap_or(0.0);
    let (charger_near, charger_visible) = charger_signal_scores(now);
    let charger_contact_plausible = now.body.charging || charger_near >= 0.92;
    let charger_approach_bonus = if matches!(
        action,
        ActionPrimitive::Approach {
            target: ApproachTarget::Charger
        }
    ) {
        if charger_contact_plausible {
            0.08
        } else {
            let memory = now.memory.place_charge_value.clamp(0.0, 1.0);
            (charger_visible.max(charger_near) * 0.35 + memory * 0.18).min(0.45)
        }
    } else {
        0.0
    };
    let dock_distance_penalty =
        if matches!(action, ActionPrimitive::Dock) && !charger_contact_plausible {
            if charger_visible >= 0.20 || charger_near >= 0.25 {
                0.65
            } else {
                0.95
            }
        } else {
            0.0
        };
    let curiosity = curiosity_action_bonus(now, action);
    let collision_risk = fallback_collision_risk(now, action).max(danger);
    let low_battery_risk = if now.body.battery_level <= 0.2
        && matches!(
            action,
            ActionPrimitive::Go { .. } | ActionPrimitive::Explore { .. }
        ) {
        0.25
    } else {
        0.0
    };
    let repeat_penalty = if previous_action == Some(action) {
        0.03
    } else {
        0.0
    };
    let recovery_bonus = recovery_candidate_bonus(now, action, previous_action);
    let fallback_used =
        signals.danger.is_none() || signals.charge.is_none() || signals.action_value.is_none();
    let score = (-1.6 * danger)
        + (1.2 * charge)
        + action_value
        + curiosity
        + recovery_bonus
        + charger_approach_bonus
        - (0.8 * collision_risk)
        - low_battery_risk
        - dock_distance_penalty
        - repeat_penalty;

    ActionSelectionCandidateScore {
        action: action.clone(),
        score,
        danger,
        charge,
        action_value,
        curiosity,
        collision_risk,
        low_battery_risk,
        repeat_penalty,
        fallback_used,
    }
}

fn curiosity_action_bonus(now: &Now, action: &ActionPrimitive) -> f32 {
    let curiosity = now.drives.curiosity.clamp(0.0, 1.0);
    let novelty = now.memory.place_novelty.clamp(0.0, 1.0);
    let pressure = curiosity.max(novelty * 0.75);
    match action {
        ActionPrimitive::Explore { .. } => pressure * 0.24,
        ActionPrimitive::Inspect {
            target: InspectTarget::Novelty,
        } => pressure * 0.22,
        ActionPrimitive::Turn { .. } => pressure * 0.10,
        ActionPrimitive::Go { intensity, .. } if *intensity > 0.0 => pressure * 0.06,
        _ => 0.0,
    }
}

fn map_memory_decision_debug(
    now: &Now,
    chosen_action: &ActionPrimitive,
    baseline_action: Option<&ActionPrimitive>,
    forced_action: bool,
) -> MapMemoryDecisionDebug {
    let corrected_map_trust = corrected_map_trust_status(now);
    let mut debug = MapMemoryDecisionDebug {
        corrected_map_trusted: corrected_map_trust.trusted,
        corrected_map_untrusted_reason: corrected_map_trust.reason.clone(),
        place_danger: now.memory.place_danger,
        place_charge_value: now.memory.place_charge_value,
        place_novelty: now.memory.place_novelty,
        safe_direction_rad: now.memory.nearby_best_safe_direction_rad,
        charge_direction_rad: now.memory.nearby_best_charge_direction_rad,
        frontier_direction_rad: now.memory.nearby_frontier_direction_rad,
        recent_trap_direction_rad: now.memory.recent_trap_direction_rad,
        map_confidence: now.memory.map_confidence,
        recent_trap_confidence: now.memory.recent_trap_confidence,
        selected_action: Some(chosen_action.clone()),
        chosen_action: Some(chosen_action.clone()),
        ..MapMemoryDecisionDebug::default()
    };
    if forced_action || baseline_action != Some(chosen_action) {
        return debug;
    }

    debug.reason = map_memory_decision_reason(now, chosen_action);
    debug.influenced = debug.reason.is_some();
    if let Some(reason) = debug.reason.as_deref() {
        debug.navigation_intent = Some(map_memory_navigation_intent(reason));
        debug.reason_string = Some(map_memory_reason_string(reason, now));
        debug.signal = Some(map_memory_signal(reason));
        debug.signal_value = map_memory_signal_value(reason, now);
        debug.signal_confidence = map_memory_confidence(reason, now);
        debug.confidence = debug.signal_confidence;
    }
    debug
}

#[derive(Clone, Debug, Default, PartialEq)]
struct CorrectedMapTrustStatus {
    trusted: bool,
    reason: Option<String>,
}

fn corrected_map_trust_status(now: &Now) -> CorrectedMapTrustStatus {
    if let Some(sensor_truth) = now
        .extensions
        .get("sensor_truth")
        .or_else(|| now.extensions.get("geometry.sensor_truth"))
    {
        if sensor_truth
            .get("ready_for_real_slam")
            .and_then(serde_json::Value::as_bool)
            == Some(false)
        {
            return CorrectedMapTrustStatus {
                trusted: false,
                reason: Some("sensor_truth.ready_for_real_slam is false".to_string()),
            };
        }
    }

    let Some(map) = now.extensions.get(MAP_EXTENSION_NAME) else {
        return CorrectedMapTrustStatus {
            trusted: false,
            reason: Some(format!("{MAP_EXTENSION_NAME} summary is missing")),
        };
    };
    if let Some(slam_status) = map.get("slam_status") {
        let mode = slam_status
            .get("mode")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("odometry_only");
        if mode != "loop_closed_pose_graph" {
            let detail = slam_status
                .get("reasons")
                .and_then(serde_json::Value::as_array)
                .and_then(|reasons| reasons.iter().find_map(serde_json::Value::as_str))
                .unwrap_or("map is not in loop-closed pose-graph SLAM mode");
            return CorrectedMapTrustStatus {
                trusted: false,
                reason: Some(format!("slam_status.mode is {mode}: {detail}")),
            };
        }
        return CorrectedMapTrustStatus {
            trusted: true,
            reason: None,
        };
    }

    let accepted = map
        .get("loop_closures_accepted")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if accepted == 0 {
        return CorrectedMapTrustStatus {
            trusted: false,
            reason: Some("no accepted loop-closure edges in the live pose graph".to_string()),
        };
    }
    let optimized_nodes = map
        .get("pose_graph_optimization")
        .and_then(|value| value.get("optimized_nodes"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let active_edges = map
        .get("pose_graph_optimization")
        .and_then(|value| value.get("active_edges"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if optimized_nodes == 0 || active_edges == 0 {
        return CorrectedMapTrustStatus {
            trusted: false,
            reason: Some("pose graph has not optimized corrected live nodes".to_string()),
        };
    }
    let remap_submaps = map
        .get("remap")
        .and_then(|value| value.get("submaps"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let remap_generation = map
        .get("remap")
        .and_then(|value| value.get("generation"))
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    if remap_submaps == 0 || remap_generation == 0 {
        return CorrectedMapTrustStatus {
            trusted: false,
            reason: Some("occupancy has not been rebuilt from corrected submaps".to_string()),
        };
    }

    CorrectedMapTrustStatus {
        trusted: true,
        reason: None,
    }
}

fn memory_for_navigation_with_map_trust(
    mut memory: MemorySense,
    trust: CorrectedMapTrustStatus,
) -> MemorySense {
    if trust.trusted {
        return memory;
    }
    memory.place_danger = 0.0;
    memory.place_charge_value = 0.0;
    memory.place_social_value = 0.0;
    memory.place_novelty = 0.0;
    memory.nearby_best_charge_direction_rad = None;
    memory.nearby_best_safe_direction_rad = None;
    memory.nearby_frontier_direction_rad = None;
    memory.recent_trap_direction_rad = None;
    memory.recent_trap_confidence = 0.0;
    memory.map_confidence = 0.0;
    memory
}

fn memory_navigation_candidate_context(now: &Now, action: &ActionPrimitive) -> bool {
    if !corrected_map_trust_status(now).trusted {
        return false;
    }
    map_memory_decision_reason(now, action).is_some()
}

fn map_memory_decision_reason(now: &Now, action: &ActionPrimitive) -> Option<String> {
    if !corrected_map_trust_status(now).trusted {
        return None;
    }
    const CRITICAL_BATTERY: f32 = 0.10;
    const LOW_BATTERY: f32 = 0.20;
    const DANGER_THRESHOLD: f32 = 0.70;
    const NOVELTY_THRESHOLD: f32 = 0.50;

    if now.memory.place_danger >= DANGER_THRESHOLD {
        if let ActionPrimitive::Turn { direction, .. } = action {
            if let Some(bearing) = now.memory.nearby_best_safe_direction_rad {
                let expected = if bearing < 0.0 {
                    TurnDir::Right
                } else {
                    TurnDir::Left
                };
                if direction == &expected {
                    return Some("danger_safe_direction".to_string());
                }
            }
            return Some("danger_current_cell".to_string());
        }
    }

    if now.memory.recent_trap_confidence >= 0.6 {
        if let ActionPrimitive::Turn { direction, .. } = action {
            if let Some(bearing) = now.memory.nearby_best_safe_direction_rad {
                let expected = if bearing < 0.0 {
                    TurnDir::Right
                } else {
                    TurnDir::Left
                };
                if direction == &expected {
                    return Some("recent_trap_safe_direction".to_string());
                }
            }
            return Some("recent_trap_turn".to_string());
        }
    }

    if now.body.battery_level <= LOW_BATTERY && now.memory.place_charge_value > 0.5 {
        match action {
            ActionPrimitive::Turn { direction, .. } => {
                if let Some(bearing) = now.memory.nearby_best_charge_direction_rad {
                    let expected = if bearing < 0.0 {
                        TurnDir::Right
                    } else {
                        TurnDir::Left
                    };
                    if bearing.abs() > 0.20 && direction == &expected {
                        return Some("charge_direction_turn".to_string());
                    }
                }
            }
            ActionPrimitive::Approach {
                target: ApproachTarget::Charger,
            } => return Some("charge_direction_aligned".to_string()),
            _ => {}
        }
    }

    if now.body.battery_level <= CRITICAL_BATTERY
        && matches!(action, ActionPrimitive::Stop)
        && now.memory.place_charge_value < 0.25
        && now.memory.nearby_best_charge_direction_rad.is_none()
    {
        return Some("charge_low_confidence_fallback".to_string());
    }

    if now.memory.place_novelty >= NOVELTY_THRESHOLD && now.memory.place_danger < DANGER_THRESHOLD {
        if let ActionPrimitive::Turn { direction, .. } = action {
            if let Some(bearing) = now.memory.nearby_frontier_direction_rad {
                let expected = if bearing < 0.0 {
                    TurnDir::Right
                } else {
                    TurnDir::Left
                };
                if direction == &expected {
                    return Some("frontier_direction_turn".to_string());
                }
            }
        }
        if matches!(
            action,
            ActionPrimitive::Inspect {
                target: InspectTarget::Novelty
            }
        ) {
            return Some("safe_novelty_inspect".to_string());
        }
    }

    None
}

fn map_memory_navigation_intent(reason: &str) -> NavigationIntent {
    if reason.starts_with("danger_") {
        NavigationIntent::AvoidKnownDangerCell
    } else if reason.starts_with("recent_trap_") {
        NavigationIntent::ReturnToFamiliarSafeCell
    } else if reason == "charge_low_confidence_fallback" {
        NavigationIntent::StopAskForHelpWhenUncertain
    } else if reason.starts_with("charge_") {
        NavigationIntent::GoTowardKnownCharger
    } else if reason.starts_with("frontier_") || reason.starts_with("safe_novelty_") {
        NavigationIntent::InspectSafeNovelFrontier
    } else {
        NavigationIntent::Explore
    }
}

fn map_memory_signal(reason: &str) -> String {
    match reason {
        "danger_safe_direction" => "memory.nearby_best_safe_direction_rad",
        "danger_current_cell" => "memory.place_danger",
        "recent_trap_safe_direction" => {
            "memory.recent_trap_confidence+nearby_best_safe_direction_rad"
        }
        "recent_trap_turn" => "memory.recent_trap_confidence",
        "charge_direction_turn" => "memory.nearby_best_charge_direction_rad",
        "charge_direction_aligned" => "memory.place_charge_value",
        "charge_low_confidence_fallback" => "memory.nearby_best_charge_direction_rad",
        "frontier_direction_turn" => "memory.nearby_frontier_direction_rad",
        "safe_novelty_inspect" => "memory.place_novelty",
        _ => "memory.map",
    }
    .to_string()
}

fn map_memory_signal_value(reason: &str, now: &Now) -> Option<f32> {
    match reason {
        "danger_safe_direction" | "recent_trap_safe_direction" => {
            now.memory.nearby_best_safe_direction_rad
        }
        "danger_current_cell" => Some(now.memory.place_danger),
        "recent_trap_turn" => Some(now.memory.recent_trap_confidence),
        "charge_direction_turn" => now.memory.nearby_best_charge_direction_rad,
        "charge_direction_aligned" => Some(now.memory.place_charge_value),
        "charge_low_confidence_fallback" => now.memory.nearby_best_charge_direction_rad,
        "frontier_direction_turn" => now.memory.nearby_frontier_direction_rad,
        "safe_novelty_inspect" => Some(now.memory.place_novelty),
        _ => None,
    }
}

fn map_memory_confidence(reason: &str, now: &Now) -> f32 {
    let (charger_near, charger_visible) = charger_signal_scores(now);
    let charge_confidence = charger_near
        .max(charger_visible)
        .max(now.memory.place_charge_value)
        .clamp(0.0, 1.0);
    match reason {
        reason if reason.starts_with("danger_") => now.memory.place_danger.clamp(0.0, 1.0),
        reason if reason.starts_with("recent_trap_") => {
            now.memory.recent_trap_confidence.clamp(0.0, 1.0)
        }
        reason if reason.starts_with("charge_") => charge_confidence,
        "frontier_direction_turn" | "safe_novelty_inspect" => {
            now.memory.place_novelty.clamp(0.0, 1.0)
        }
        _ => now.memory.map_confidence.clamp(0.0, 1.0),
    }
}

fn map_memory_reason_string(reason: &str, now: &Now) -> String {
    match reason {
        "danger_safe_direction" => format!(
            "avoiding remembered danger {:.2} using safe bearing {:?}",
            now.memory.place_danger, now.memory.nearby_best_safe_direction_rad
        ),
        "danger_current_cell" => format!(
            "avoiding remembered/current danger {:.2} with local range clearance",
            now.memory.place_danger
        ),
        "recent_trap_safe_direction" => format!(
            "returning toward familiar safe cell from trap confidence {:.2}",
            now.memory.recent_trap_confidence
        ),
        "recent_trap_turn" => format!(
            "turning away from recent trap confidence {:.2}",
            now.memory.recent_trap_confidence
        ),
        "charge_direction_turn" => format!(
            "turning toward remembered charger bearing {:?} with charge value {:.2}",
            now.memory.nearby_best_charge_direction_rad, now.memory.place_charge_value
        ),
        "charge_direction_aligned" => format!(
            "approaching charger from remembered charge value {:.2}",
            now.memory.place_charge_value
        ),
        "charge_low_confidence_fallback" => format!(
            "critical battery but charger memory is too weak: charge value {:.2}, bearing {:?}",
            now.memory.place_charge_value, now.memory.nearby_best_charge_direction_rad
        ),
        "frontier_direction_turn" => format!(
            "inspecting safe novel frontier bearing {:?} with novelty {:.2}",
            now.memory.nearby_frontier_direction_rad, now.memory.place_novelty
        ),
        "safe_novelty_inspect" => format!(
            "inspecting safe novel place with novelty {:.2}",
            now.memory.place_novelty
        ),
        _ => "memory/map signal influenced navigation".to_string(),
    }
}

fn select_action_from_scores(
    mode: ActionSelectorMode,
    now: &Now,
    baseline_action: ActionPrimitive,
    candidates: Vec<ActionSelectionCandidateScore>,
) -> ActionSelectionDecision {
    if mode != ActionSelectorMode::Baseline {
        if let Some(action) = hard_safety_action(now) {
            return ActionSelectionDecision {
                mode,
                candidates,
                selected_action: Some(action),
                baseline_action: Some(baseline_action),
                selected_score: None,
                safety_overrode: true,
                fallback_warnings: fallback_warnings_for_mode(mode),
                ..ActionSelectionDecision::default()
            };
        }
    }

    let selected = match mode {
        ActionSelectorMode::Baseline => Some(ActionSelectionCandidateScore {
            action: baseline_action.clone(),
            score: 0.0,
            ..ActionSelectionCandidateScore::default()
        }),
        ActionSelectorMode::Scripted => candidates
            .iter()
            .find(|candidate| {
                matches!(
                    candidate.action,
                    ActionPrimitive::Approach {
                        target: ApproachTarget::Charger
                    } | ActionPrimitive::Dock
                )
            })
            .cloned()
            .or_else(|| candidates.first().cloned()),
        ActionSelectorMode::Random => {
            if candidates.is_empty() {
                None
            } else {
                candidates
                    .get(now.t_ms as usize % candidates.len())
                    .cloned()
            }
        }
        ActionSelectorMode::ModelAssisted => candidates
            .iter()
            .max_by(|left, right| {
                left.score
                    .partial_cmp(&right.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned(),
        ActionSelectorMode::GoalShadow | ActionSelectorMode::Goal => {
            Some(ActionSelectionCandidateScore {
                action: baseline_action.clone(),
                score: 0.0,
                ..ActionSelectionCandidateScore::default()
            })
        }
    };
    let fallback_warnings = if candidates.iter().any(|candidate| candidate.fallback_used) {
        fallback_warnings_for_mode(mode)
    } else {
        Vec::new()
    };
    ActionSelectionDecision {
        mode,
        selected_action: selected
            .as_ref()
            .map(|candidate| candidate.action.clone())
            .or(Some(baseline_action.clone())),
        selected_score: selected.as_ref().map(|candidate| candidate.score),
        candidates,
        baseline_action: Some(baseline_action),
        safety_overrode: false,
        fallback_warnings,
        ..ActionSelectionDecision::default()
    }
}

fn fallback_warnings_for_mode(mode: ActionSelectorMode) -> Vec<String> {
    if mode == ActionSelectorMode::ModelAssisted {
        vec!["model-assisted selector used hardcoded fallback estimates".to_string()]
    } else {
        Vec::new()
    }
}

fn recovery_candidate_bonus(
    now: &Now,
    action: &ActionPrimitive,
    baseline_action: Option<&ActionPrimitive>,
) -> f32 {
    if !recovery_candidate_context(now) || !is_recovery_locomotion_action(action) {
        return 0.0;
    }
    if baseline_action == Some(action) {
        3.0
    } else {
        0.75
    }
}

fn recovery_candidate_context(now: &Now) -> bool {
    let contact = now.body.flags.bump_left || now.body.flags.bump_right || now.body.flags.wall;
    let close_range = now
        .range
        .nearest_m
        .map(|nearest| nearest < 0.35)
        .unwrap_or(false);
    contact || close_range || sim_stuck_active(now)
}

fn is_recovery_locomotion_action(action: &ActionPrimitive) -> bool {
    match action {
        ActionPrimitive::Go { intensity, .. } => intensity.abs() <= 0.25,
        ActionPrimitive::Turn { intensity, .. } => *intensity >= 0.5,
        _ => false,
    }
}

fn hard_safety_action(now: &Now) -> Option<ActionPrimitive> {
    if now.body.flags.wheel_drop {
        return Some(ActionPrimitive::Stop);
    }
    if now.body.flags.cliff_left
        || now.body.flags.cliff_front_left
        || now.body.flags.cliff_front_right
        || now.body.flags.cliff_right
    {
        return Some(ActionPrimitive::Stop);
    }
    if now.body.flags.bump_left || now.body.flags.bump_right || now.body.flags.wall {
        let direction = if now.body.flags.bump_left && !now.body.flags.bump_right {
            TurnDir::Right
        } else if now.body.flags.bump_right && !now.body.flags.bump_left {
            TurnDir::Left
        } else if range_clearer_on_right(&now.range.beams) {
            TurnDir::Right
        } else {
            TurnDir::Left
        };
        return Some(ActionPrimitive::Turn {
            direction,
            intensity: 0.7,
            duration_ms: 1_200,
        });
    }
    if now.body.battery_level <= 0.10 && !charger_reachable_signal(now) {
        return Some(ActionPrimitive::Stop);
    }
    let danger = now
        .predictions
        .danger_model
        .or(now.predictions.danger_hardcoded)
        .map(|prediction| {
            prediction
                .bump_risk
                .max(prediction.cliff_risk)
                .max(prediction.wheel_drop_risk)
                .max(prediction.stuck_risk)
        })
        .unwrap_or(0.0);
    if danger >= 0.70 {
        return Some(ActionPrimitive::Turn {
            direction: TurnDir::Left,
            intensity: 0.5,
            duration_ms: 1_000,
        });
    }
    None
}

fn charger_reachable_signal(now: &Now) -> bool {
    let (charger_near, charger_visible) = charger_signal_scores(now);
    now.body.charging
        || charger_near >= 0.25
        || charger_visible >= 0.20
        || now.memory.place_charge_value >= 0.5
        || now.memory.nearby_best_charge_direction_rad.is_some()
        || now
            .predictions
            .charge_model
            .or(now.predictions.charge_hardcoded)
            .map(|prediction| prediction.charge_probability >= 0.7)
            .unwrap_or(false)
}

fn range_clearer_on_right(beams: &[f32]) -> bool {
    if beams.len() < 2 {
        return false;
    }
    let (left, _, right) = beam_clearance_buckets(beams);
    right > left
}

fn max_danger_risk(output: DangerOutput) -> f32 {
    output
        .bump_risk
        .max(output.cliff_risk)
        .max(output.wheel_drop_risk)
        .max(output.stuck_risk)
        .clamp(0.0, 1.0)
}

fn charge_score(output: ChargeOutput) -> f32 {
    (output.charge_probability + output.dock_likelihood + output.expected_battery_delta.max(0.0))
        .clamp(0.0, 1.0)
}

fn fallback_collision_risk(now: &Now, action: &ActionPrimitive) -> f32 {
    let forward = action_to_motor_command(Some(action)).forward;
    let anticipated_risk = anticipated_surface_collision_risk(now);
    let nearest_risk = now
        .range
        .nearest_m
        .map(|nearest| ((0.35 - nearest) / 0.35).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let contact_risk =
        if now.body.flags.bump_left || now.body.flags.bump_right || now.body.flags.wall {
            1.0
        } else {
            0.0
        };
    if forward <= 0.0 {
        return anticipated_risk.max(contact_risk);
    }
    nearest_risk.max(contact_risk).max(anticipated_risk)
}

fn now_with_surface_anticipation(
    now: &Now,
    surface_output: Option<&SurfaceExtractorOutput>,
    action: &ActionPrimitive,
) -> Now {
    let Some(surface_output) = surface_output else {
        return now.clone();
    };
    let mut next = now.clone();
    let frames = anticipate_surfaces(surface_output, now.body.odometry, action);
    let max_risk = frames
        .iter()
        .map(|frame| frame.navigation.collision_risk)
        .fold(0.0f32, f32::max);
    let nearest_front = frames
        .iter()
        .filter_map(|frame| frame.navigation.front_clear_m)
        .min_by(|left, right| left.total_cmp(right));
    let anticipation_value = serde_json::json!({
        "action": action,
        "frames": frames,
        "max_collision_risk": max_risk,
        "nearest_front_clear_m": nearest_front,
    });
    let entry = next
        .extensions
        .entry("surface.scene_graph".to_string())
        .or_insert_with(|| serde_json::json!({}));
    if let Some(object) = entry.as_object_mut() {
        object.insert("anticipation".to_string(), anticipation_value);
    }
    next
}

fn anticipated_surface_collision_risk(now: &Now) -> f32 {
    now.extensions
        .get("surface.scene_graph")
        .and_then(|value| value.get("anticipation"))
        .and_then(|value| value.get("max_collision_risk"))
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0) as f32
}

fn action_value_candidate_actions(
    proposals: &[ActionPrimitive],
    reign_action: Option<&ActionPrimitive>,
    llm_tick: &LlmTickResult,
) -> Vec<ActionPrimitive> {
    let mut candidates = default_candidate_actions();
    if let Some(action) = reign_action {
        push_unique_action(&mut candidates, action.clone());
    }
    if let Some(action) = llm_tick
        .conscious_command
        .as_ref()
        .and_then(|cmd| cmd.action.clone())
    {
        push_unique_action(&mut candidates, action);
    }
    for action in proposals {
        push_unique_action(&mut candidates, action.clone());
    }
    candidates
}

fn llm_explicit_safety_reason(llm_tick: &LlmTickResult) -> bool {
    let Some(decision) = llm_tick.decision.as_ref() else {
        return false;
    };
    [
        Some(decision.summary.as_str()),
        decision.critique.as_deref(),
    ]
    .into_iter()
    .flatten()
    .any(|text| {
        let text = text.to_ascii_lowercase();
        [
            "safety",
            "unsafe",
            "danger",
            "hazard",
            "collision",
            "cliff",
            "wheel drop",
            "blocked",
            "veto",
        ]
        .iter()
        .any(|needle| text.contains(needle))
    })
}

fn push_unique_action(actions: &mut Vec<ActionPrimitive>, action: ActionPrimitive) {
    if !actions.iter().any(|existing| existing == &action) {
        actions.push(action);
    }
}

fn apply_responses(
    now: &mut Now,
    responses: Vec<Response>,
    sensations: &mut Vec<Sensation>,
    impressions: &mut Vec<Impression>,
    experiences: &mut Vec<Experience>,
    teachings: &mut Vec<pete_llm::LlmTeaching>,
    notes: &mut Vec<String>,
    drive_impulses: &mut DriveSense,
) {
    for response in responses {
        match response {
            Response::Emit(_) => {}
            Response::AddSensation(sensation) => sensations.push(sensation),
            Response::AddImpression(impression) => impressions.push(impression),
            Response::AddExperience(experience) => experiences.push(experience),
            Response::AddDriveImpulse { name, value } => {
                add_drive_impulse(drive_impulses, &name, value)
            }
            Response::SetMemorySense(memory) => now.memory = memory,
            Response::Teach(teaching) => teachings.push(teaching),
            Response::AddMemoryNote(note) => notes.push(note),
        }
    }
}

fn apply_llm_tick(
    llm_tick: &LlmTickResult,
    occurred_at_ms: u64,
    observed_at_ms: u64,
    snapshot_ref: &str,
    sensations: &mut Vec<Sensation>,
    impressions: &mut Vec<Impression>,
    experiences: &mut Vec<Experience>,
    teachings: &mut Vec<pete_llm::LlmTeaching>,
) {
    if let Some(command) = &llm_tick.conscious_command {
        let sensation = Sensation::new(
            "llm.command",
            "llm",
            occurred_at_ms,
            observed_at_ms,
            serde_json::json!({
                "summary": command.summary,
                "action": command.action,
                "input_snapshot_ref": snapshot_ref,
            }),
        )
        .with_summary(command.summary.clone())
        .with_provenance(Provenance::direct().with_stage("llm"));
        let impression = Impression::new(
            "llm.command.observation",
            command.summary.clone(),
            vec![sensation.id],
            sensation.occurred_at_ms,
            sensation.observed_at_ms,
        )
        .with_confidence(llm_tick.sense.confidence);
        let experience = Experience::new(
            "llm.command",
            command.summary.clone(),
            vec![impression.id],
            vec![sensation.id],
            sensation.occurred_at_ms,
            sensation.observed_at_ms,
        );
        sensations.push(sensation);
        impressions.push(impression);
        experiences.push(experience);
    }

    if let Some(critique) = &llm_tick.sense.critique {
        let sensation = Sensation::new(
            "llm.critique",
            "llm",
            occurred_at_ms,
            observed_at_ms,
            serde_json::json!({
                "critique": critique,
                "input_snapshot_ref": snapshot_ref,
            }),
        )
        .with_summary(critique.clone())
        .with_provenance(Provenance::direct().with_stage("llm"));
        let impression = Impression::new(
            "llm.critique.observation",
            critique.clone(),
            vec![sensation.id],
            sensation.occurred_at_ms,
            sensation.observed_at_ms,
        )
        .with_confidence(llm_tick.sense.confidence);
        sensations.push(sensation);
        impressions.push(impression);
    }

    teachings.extend(llm_tick.teaching.clone());
}

fn append_combobulation(
    sensations: &mut Vec<Sensation>,
    impressions: &mut Vec<Impression>,
    experiences: &mut Vec<Experience>,
    occurred_at_ms: u64,
    observed_at_ms: u64,
    snapshot_ref: &str,
    combobulation: &Combobulation,
) {
    let sensation = Sensation::new(
        "llm.combobulation",
        "llm",
        occurred_at_ms,
        observed_at_ms,
        serde_json::json!({
            "summary": combobulation.summary,
            "confidence": combobulation.confidence,
            "input_snapshot_ref": snapshot_ref,
        }),
    )
    .with_summary(combobulation.summary.clone())
    .with_provenance(Provenance::direct().with_stage("combobulator"));
    let impression = Impression::new(
        "llm.combobulation.observation",
        combobulation.summary.clone(),
        vec![sensation.id],
        occurred_at_ms,
        observed_at_ms,
    )
    .with_confidence(combobulation.confidence);
    let experience = Experience::new(
        "llm.combobulation",
        combobulation.summary.clone(),
        vec![impression.id],
        vec![sensation.id],
        occurred_at_ms,
        observed_at_ms,
    );
    sensations.push(sensation);
    impressions.push(impression);
    experiences.push(experience);
}

fn embodied_recall_sensations_and_impressions(
    recall: &RecallBundle,
) -> (Vec<Sensation>, Vec<Impression>) {
    let mut sensations = Vec::new();
    let mut impressions = Vec::new();
    for recollection in &recall.recollections {
        let sensation = recollection.sensation.clone();
        if let Some(impression) = sensation.impression.clone() {
            impressions.push(impression);
        }
        sensations.push(sensation);
    }
    (sensations, impressions)
}

fn derive_direct_impressions_from_now(now: &Now) -> (Vec<Sensation>, Vec<Impression>) {
    let mut sensations = Vec::new();
    let mut impressions = Vec::new();
    let floor_feel = if now.body.flags.cliff_left
        || now.body.flags.cliff_front_left
        || now.body.flags.cliff_front_right
        || now.body.flags.cliff_right
    {
        "the floor feels like it falls away near me"
    } else if now.body.cliff_sensors.max() > 0.0 {
        "the floor feels steady, though my cliff IR signal is uncertain"
    } else {
        "the floor feels steady under me"
    };
    let contact_feel = if now.body.flags.bump_left || now.body.flags.bump_right {
        "my body feels blocked by contact"
    } else if now.body.flags.wall || now.body.flags.virtual_wall {
        "I feel a boundary close to me"
    } else {
        "my body feels unblocked"
    };
    let wheel_feel = if now.body.flags.wheel_drop {
        "one wheel feels unsupported"
    } else {
        "my wheels feel supported"
    };
    let charging_feel = if now.body.charging {
        "charging feels present"
    } else {
        "I do not feel charging contact"
    };
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "body.state",
        "body",
        format!(
            "My body feels {:.0}% full of power; {charging_feel}, {contact_feel}, {wheel_feel}, and {floor_feel}. I feel myself moving forward {:.2} m/s and turning {:.2} rad/s, with my body centered near ({:.2}, {:.2}) and facing {:.2} radians.",
            now.body.battery_level * 100.0,
            now.body.velocity.forward_m_s,
            now.body.velocity.turn_rad_s,
            now.body.odometry.x_m,
            now.body.odometry.y_m,
            now.body.odometry.heading_rad,
        ),
        0.9,
    );
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "eye.state",
        "eye",
        format!(
            "I am seeing through {} frame feature sets, with {} image vectors, {} image-description vectors, and {} scene vectors available.",
            now.eye.frames.len(),
            now.eye.image_vectors.len(),
            now.eye.image_description_vectors.len(),
            now.eye.scene_vectors.len(),
        ),
        0.6,
    );
    let transcript = now
        .ear
        .asr
        .transcript
        .as_deref()
        .or(now.ear.transcript.as_deref());
    if let Some(transcript) = transcript {
        let transcript = transcript.trim();
        if !transcript.is_empty() {
            push_now_input_impression(
                &mut sensations,
                &mut impressions,
                now.t_ms,
                "audio.transcript",
                "ear",
                asr_hearing_impression_text(
                    transcript,
                    now.ear.asr.is_final,
                    now.ear.asr.confidence,
                ),
                now.ear.asr.confidence.max(0.35),
            );
        }
    }
    if let Some(possible) = now
        .ear
        .asr
        .possible_transcript
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        push_now_input_impression(
            &mut sensations,
            &mut impressions,
            now.t_ms,
            "audio.possible_speech",
            "ear",
            asr_possible_speech_impression_text(possible, now.ear.asr.confidence),
            now.ear.asr.confidence.max(0.25),
        );
    }
    if let Some(committed) = now
        .ear
        .asr
        .committed_transcript
        .as_deref()
        .or_else(|| now.ear.asr.is_final.then_some(transcript).flatten())
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        push_now_input_impression(
            &mut sensations,
            &mut impressions,
            now.t_ms,
            "audio.committed_speech",
            "ear",
            asr_committed_speech_impression_text(committed, now.ear.asr.confidence),
            now.ear.asr.confidence.max(0.35),
        );
    }
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "ear.state",
        "ear",
        format!(
            "I am hearing through {} audio feature sets; my speech recognition final state is {}, confidence is {:.2}, word count is {:?}, and sequence is {:?}-{:?}.",
            now.ear.features.len(),
            now.ear.asr.is_final,
            now.ear.asr.confidence,
            now.ear.asr.word_count,
            now.ear.asr.sequence_start,
            now.ear.asr.sequence_end,
        ),
        0.6,
    );
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "range.state",
        "range",
        format!(
            "I sense the nearest obstacle at {:?} meters, from {} range beam samples.",
            now.range.nearest_m,
            now.range.beams.len(),
        ),
        0.7,
    );
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "imu.state",
        "imu",
        format!(
            "I feel my orientation through {} values, acceleration through {} values, and angular velocity through {} values.",
            now.imu.orientation.len(),
            now.imu.acceleration.len(),
            now.imu.angular_velocity.len(),
        ),
        0.5,
    );
    if let Some(gps) = &now.gps {
        push_now_input_impression(
            &mut sensations,
            &mut impressions,
            now.t_ms,
            "gps.state",
            "gps",
            format!(
                "I am located near latitude {:.6}, longitude {:.6}, altitude {:?} meters.",
                gps.lat, gps.lon, gps.altitude_m
            ),
            0.6,
        );
    }
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "identity.state",
        "identity",
        format!(
            "I have {} face vectors and {} voice vectors available for recognizing who may be present.",
            now.face.vectors.len(),
            now.voice.vectors.len(),
        ),
        0.5,
    );
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "kinect.state",
        "kinect",
        format!(
            "I sense the room with {} Kinect color feature sets, {} depth samples, {} IR samples, {} skeletons, and audio angle {:?} at confidence {:.2}.",
            now.kinect.color_features.len(),
            now.kinect.depth_m.len(),
            now.kinect.ir.len(),
            now.kinect.skeletons.len(),
            now.kinect.audio_angle_rad,
            now.kinect.audio_confidence,
        ),
        0.5,
    );
    if let Some(surface_graph) = now.extensions.get("surface.scene_graph") {
        push_now_input_impression(
            &mut sensations,
            &mut impressions,
            now.t_ms,
            "surface.scene_graph",
            "surface",
            summarize_surface_scene_graph(surface_graph),
            0.75,
        );
    }
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "memory.state",
        "memory",
        format!(
            "I remember this place with familiarity {:.2}, danger {:.2}, charge value {:.2}, social value {:.2}, novelty {:.2}, {} similar situations, warning {:?}, and graph summary {:?}.",
            now.memory.place_familiarity,
            now.memory.place_danger,
            now.memory.place_charge_value,
            now.memory.place_social_value,
            now.memory.place_novelty,
            now.memory.similar_situation_count,
            now.memory.remembered_warning,
            now.memory.graph_context_summary,
        ),
        0.7,
    );
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "prediction.state",
        "predictions",
        format!(
            "I expect events {:?} with uncertainty {:.2}; my danger model says {:?}, hardcoded danger says {:?}, charge model says {:?}, hardcoded charge says {:?}, and I have {} model action values plus {} hardcoded action values.",
            now.predictions.expected_events,
            now.predictions.uncertainty,
            now.predictions.danger_model,
            now.predictions.danger_hardcoded,
            now.predictions.charge_model,
            now.predictions.charge_hardcoded,
            now.predictions.action_values_model.len(),
            now.predictions.action_values_hardcoded.len(),
        ),
        0.7,
    );
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "surprise.state",
        "surprise",
        format!(
            "I feel surprise at {:.2}, with prediction error {:.2}.",
            now.surprise.total, now.surprise.prediction_error
        ),
        0.7,
    );
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "drive.state",
        "drives",
        format!(
            "I feel battery hunger {:.2}, danger avoidance {:.2}, curiosity {:.2}, social interest {:.2}, fatigue {:.2}, and uncertainty pressure {:.2}.",
            now.drives.battery_hunger,
            now.drives.danger_avoidance,
            now.drives.curiosity,
            now.drives.social_interest,
            now.drives.fatigue,
            now.drives.uncertainty_pressure,
        ),
        0.7,
    );
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "reign.state",
        "reign",
        format!(
            "Remote reign control active {}, mode {:?}, with {} pending commands, age {:?} ms, override pressure {:.2}, and latest command {}.",
            now.reign.active,
            now.reign.mode,
            now.reign.pending_count,
            now.reign.last_command_age_ms,
            now.reign.human_override_pressure,
            now.reign
                .latest
                .as_ref()
                .map(summarize_reign_command_for_runtime)
                .unwrap_or_else(|| "none".to_string()),
        ),
        0.7,
    );
    push_now_input_impression(
        &mut sensations,
        &mut impressions,
        now.t_ms,
        "self.state",
        "self",
        format!(
            "I am pursuing active goal {:?}, and my mode is {:?}.",
            now.self_sense.active_goal, now.self_sense.mode
        ),
        0.6,
    );
    if !now.extensions.is_empty() {
        push_now_input_impression(
            &mut sensations,
            &mut impressions,
            now.t_ms,
            "extension.state",
            "extensions",
            format!(
                "I have extension context from {}.",
                now.extensions
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            0.5,
        );
    }
    (sensations, impressions)
}

fn summarize_surface_scene_graph(value: &serde_json::Value) -> String {
    let floor_confidence = value
        .get("floor")
        .and_then(|floor| floor.get("confidence"))
        .and_then(|confidence| confidence.as_f64());
    let surfaces = value
        .get("surfaces")
        .and_then(|surfaces| surfaces.as_array())
        .map_or(0, Vec::len);
    let clusters = value
        .get("clusters")
        .and_then(|clusters| clusters.as_array())
        .map_or(0, Vec::len);
    let moving_clusters = value
        .get("clusters")
        .and_then(|clusters| clusters.as_array())
        .map(|clusters| {
            clusters
                .iter()
                .filter(|cluster| {
                    cluster
                        .get("moving")
                        .and_then(|moving| moving.as_bool())
                        .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0);
    let hints = value
        .get("clusters")
        .and_then(|clusters| clusters.as_array())
        .map(|clusters| {
            clusters
                .iter()
                .filter_map(|cluster| cluster.get("semantic_hint").and_then(|hint| hint.as_str()))
                .take(4)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let front_clear_m = value
        .get("navigation")
        .and_then(|navigation| navigation.get("front_clear_m"))
        .and_then(|clearance| clearance.as_f64());
    let left_clear_m = value
        .get("navigation")
        .and_then(|navigation| navigation.get("left_clear_m"))
        .and_then(|clearance| clearance.as_f64());
    let right_clear_m = value
        .get("navigation")
        .and_then(|navigation| navigation.get("right_clear_m"))
        .and_then(|clearance| clearance.as_f64());
    let calibration = summarize_surface_calibration_hint(value.get("calibration_hint"));
    format!(
        "I perceive persistent geometry: floor confidence {}, {} stable surfaces, {} leftover clusters ({} moving; hints: {}), navigation clearance front {}, left {}, right {}, and calibration {}.",
        format_optional_magnitude(floor_confidence, ""),
        surfaces,
        clusters,
        moving_clusters,
        if hints.is_empty() {
            "none".to_string()
        } else {
            hints.join(", ")
        },
        format_optional_magnitude(front_clear_m, "m"),
        format_optional_magnitude(left_clear_m, "m"),
        format_optional_magnitude(right_clear_m, "m"),
        calibration
    )
}

fn summarize_surface_calibration_hint(value: Option<&serde_json::Value>) -> String {
    let Some(value) = value else {
        return "unknown".to_string();
    };
    let height_error = value
        .get("floor_height_error_m")
        .and_then(|value| value.as_f64());
    let tilt_deg = value
        .get("floor_tilt_rad")
        .and_then(|value| value.as_f64())
        .map(|value| value.to_degrees());
    match (height_error, tilt_deg) {
        (Some(height), Some(tilt)) => {
            format!("floor offset {height:.2}m and tilt {tilt:.1} degrees")
        }
        (Some(height), None) => format!("floor offset {height:.2}m"),
        (None, Some(tilt)) => format!("floor tilt {tilt:.1} degrees"),
        (None, None) => "unknown".to_string(),
    }
}

fn format_optional_magnitude(value: Option<f64>, unit: &str) -> String {
    value
        .map(|value| format!("{value:.2}{unit}"))
        .unwrap_or_else(|| "unknown".to_string())
}

fn push_now_input_impression(
    sensations: &mut Vec<Sensation>,
    impressions: &mut Vec<Impression>,
    t_ms: u64,
    kind: &str,
    source: &str,
    text: String,
    confidence: f32,
) {
    let text = ensure_natural_confidence_text(&text, confidence);
    let sensation = Sensation::new(
        kind,
        source,
        t_ms,
        t_ms,
        serde_json::json!({ "text": text }),
    )
    .with_summary(text.clone())
    .with_provenance(Provenance::direct().with_stage("now"));
    let impression = Impression::new(
        format!("{kind}.impression"),
        text,
        vec![sensation.id],
        t_ms,
        t_ms,
    )
    .with_confidence(confidence)
    .with_payload(serde_json::json!({
        "generator": "mechanical",
        "faculty": format!("{source}.mechanical_impression"),
        "source_experience_kind": kind,
        "source": source,
    }));
    sensations.push(sensation);
    impressions.push(impression);
}

fn asr_hearing_impression_text(transcript: &str, is_final: bool, confidence: f32) -> String {
    let transcript = transcript.trim();
    let confidence = confidence.clamp(0.0, 1.0);
    if is_final {
        if confidence >= 0.85 {
            format!("I'm confident I finally heard \"{transcript}\".")
        } else if confidence >= 0.60 {
            format!("I'm pretty sure I finally heard \"{transcript}\".")
        } else {
            format!("I think I finally heard \"{transcript}\".")
        }
    } else if confidence >= 0.85 {
        format!("I'm pretty sure I'm hearing \"{transcript}\".")
    } else if confidence >= 0.45 {
        format!("I think I heard \"{transcript}\".")
    } else {
        format!("I may have heard \"{transcript}\".")
    }
}

fn asr_possible_speech_impression_text(transcript: &str, confidence: f32) -> String {
    let transcript = transcript.trim();
    if confidence >= 0.75 {
        format!("I'm probably hearing the possible speech \"{transcript}\".")
    } else if confidence >= 0.45 {
        format!("I think the possible speech is \"{transcript}\".")
    } else {
        format!("I may be hearing possible speech like \"{transcript}\".")
    }
}

fn asr_committed_speech_impression_text(transcript: &str, confidence: f32) -> String {
    let transcript = transcript.trim();
    if confidence >= 0.85 {
        format!("I'm confident I can commit the heard speech as \"{transcript}\".")
    } else if confidence >= 0.60 {
        format!("I'm pretty sure I can commit the heard speech as \"{transcript}\".")
    } else {
        format!("I think I can commit the heard speech as \"{transcript}\".")
    }
}

fn ensure_natural_confidence_text(text: &str, confidence: f32) -> String {
    if starts_with_natural_confidence(text) {
        return text.to_string();
    }

    let claim = lower_first_char(text.trim());
    match confidence.clamp(0.0, 1.0) {
        value if value >= 0.85 => format!("I'm confident that {claim}"),
        value if value >= 0.65 => format!("I'm pretty sure that {claim}"),
        value if value >= 0.40 => format!("I think {claim}"),
        _ => format!("I'm not sure, but I think {claim}"),
    }
}

fn starts_with_natural_confidence(text: &str) -> bool {
    let text = text.trim();
    text.starts_with("I'm confident")
        || text.starts_with("I'm pretty sure")
        || text.starts_with("I think")
        || text.starts_with("I may have")
        || text.starts_with("I'm not sure")
}

fn lower_first_char(text: &str) -> String {
    let mut chars = text.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    first.to_lowercase().chain(chars).collect()
}

fn summarize_reign_command_for_runtime(input: &pete_actions::ReignInput) -> String {
    match &input.command {
        pete_actions::ReignCommand::Stop => "Stop".to_string(),
        pete_actions::ReignCommand::Go {
            intensity,
            duration_ms,
        } => format!("Go intensity {:.2} for {}ms", intensity, duration_ms),
        pete_actions::ReignCommand::Reverse {
            intensity,
            duration_ms,
        } => format!("Reverse intensity {:.2} for {}ms", intensity, duration_ms),
        pete_actions::ReignCommand::Drive {
            forward,
            turn,
            duration_ms,
        } => format!(
            "Drive forward {:.2}, turn {:.2} for {}ms",
            forward, turn, duration_ms
        ),
        pete_actions::ReignCommand::Turn {
            direction,
            intensity,
            duration_ms,
        } => format!(
            "Turn {:?} intensity {:.2} for {}ms",
            direction, intensity, duration_ms
        ),
        pete_actions::ReignCommand::Inspect { target } => format!("Inspect {:?}", target),
        pete_actions::ReignCommand::Approach { target } => format!("Approach {:?}", target),
        pete_actions::ReignCommand::Dock => "Dock".to_string(),
        pete_actions::ReignCommand::Explore { duration_ms } => {
            format!("Explore for {}ms", duration_ms)
        }
        pete_actions::ReignCommand::Speak { text } => format!("Speak {text}"),
        pete_actions::ReignCommand::Chirp { pattern } => format!("Chirp {:?}", pattern),
        pete_actions::ReignCommand::SetMode { mode } => format!("Set mode {:?}", mode),
    }
}

fn derive_direct_experiences(
    impressions: &[Impression],
    sensations: &[Sensation],
    t_ms: u64,
) -> Vec<Experience> {
    if impressions.is_empty() || sensations.is_empty() {
        return Vec::new();
    }
    vec![Experience::new(
        "realtime.situation",
        impressions
            .iter()
            .map(|value| value.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        impressions.iter().map(|value| value.id).collect(),
        sensations.iter().map(|value| value.id).collect(),
        t_ms,
        t_ms,
    )]
}

fn add_drive_impulse(drives: &mut DriveSense, name: &DriveName, value: f32) {
    match name {
        DriveName::BatteryHunger => {
            drives.battery_hunger = (drives.battery_hunger + value).clamp(0.0, 1.0)
        }
        DriveName::DangerAvoidance => {
            drives.danger_avoidance = (drives.danger_avoidance + value).clamp(0.0, 1.0)
        }
        DriveName::Curiosity => drives.curiosity = (drives.curiosity + value).clamp(0.0, 1.0),
        DriveName::SocialInterest => {
            drives.social_interest = (drives.social_interest + value).clamp(0.0, 1.0)
        }
        DriveName::Fatigue => drives.fatigue = (drives.fatigue + value).clamp(0.0, 1.0),
        DriveName::UncertaintyPressure => {
            drives.uncertainty_pressure = (drives.uncertainty_pressure + value).clamp(0.0, 1.0)
        }
    }
}

fn describe_safety_reason(reason: Option<SafetyReason>) -> &'static str {
    match reason {
        Some(SafetyReason::Charging) => "charging",
        Some(SafetyReason::WheelDrop) => "wheel drop",
        Some(SafetyReason::Cliff) => "cliff",
        Some(SafetyReason::BatteryCritical) => "critical battery",
        Some(SafetyReason::StaleSensors) => "stale sensors",
        Some(SafetyReason::LostBodyComms) => "lost body comms",
        Some(SafetyReason::MotorOutOfRange) => "motor out of range",
        Some(SafetyReason::HighDanger) => "high danger",
        Some(SafetyReason::RawLlmMotorRejected) => "raw llm motor rejected",
        Some(SafetyReason::ReadOnlyMode) => "read-only mode",
        Some(SafetyReason::Contact) => "contact",
        None => "unknown reason",
    }
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
