use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use pete_actions::{
    action_to_motor_command, ActionPrimitive, ApproachTarget, ChirpPattern, ExploreStyle,
    InspectTarget, LlmActionProposal, ReignInput, ReignMode, ReignOutcome, ReignSource, TurnDir,
};
use pete_autonomic::{SafetyLayer, SafetyReason};
use pete_behaviors::{
    BehaviorConfig, BehaviorImplementation, BehaviorNodeState, BehaviorNodeUpdate, BehaviorRegime,
    BehaviorRegistryConfig, ErasedBehaviorRunRecord, FallbackPolicy, FunctionBehavior,
    ReplaceableBehavior, TargetExtractor, TrainingSample, TrainingSource,
};
use pete_body::{BodyFlags, BodySense};
use pete_cockpit::{
    bump_escape_duration_ms, Cockpit, CockpitEventKind, EscapeDirection, MotionCommand,
    MotorCommand, SafeCockpit, SafeStopReason, SafetyLatchKind, StatusSummary,
    BUMP_ESCAPE_BACKOFF_DURATION_MS,
};
use pete_conductor::{Conductor, ConductorInput, GoalSystem, NavigationIntent, SimpleConductor};
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
use pete_llm::{Combobulation, LiveImageEnricher, LlmAgent, LlmTickResult};
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
    ExtensionSense, EyePrediction, Freshness, MemorySense, Now, ObjectClass, ReignSense,
    SafetySense, SurpriseSense, WorldModelUpdater,
};
use pete_sensors::{
    anticipate_surfaces, FrameProcessor, NowBuilder, SenseProducer, SurfaceExtractor,
    SurfaceExtractorOutput, World, WorldSnapshot,
};
use pete_sim::{SimCockpit, VirtualWorld};
use serde::{Deserialize, Serialize};
use tsrun::{js_value_to_json, Interpreter, JsError, StepResult};
use uuid::Uuid;

pub struct MinimalRuntime<L, M, R, C, S, A>
where
    L: LedgerWriter + Sync,
    M: MemoryStore,
    R: Recall + Sync,
    C: Conductor,
    S: SafetyLayer,
    A: LlmAgent,
{
    pub ledger: L,
    pub memory_store: M,
    pub memory_recall: R,
    pub conductor: C,
    pub safety: S,
    pub llm: A,
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
    last_active_control: Option<ActiveControlSummary>,
}

#[derive(Clone, Debug, Default)]
struct ChirpEventState {
    last_charging: Option<bool>,
    last_awake: Option<bool>,
    last_object_count: usize,
    last_face_present: bool,
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
        self.last_face_present = face_present(now);
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
pub struct FaceDetectedEventInput {
    pub t_ms: TimeMs,
    pub recognized: bool,
    pub person: FacePerson,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FacePerson {
    pub id: String,
    pub name: Option<String>,
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
        if let Some(behavior) = config.behavior.get("event_face_detected") {
            stack.behaviors.event_face_detected = face_detected_event_behavior(
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
            behavior_node_state(
                "EventFaceDetected",
                "event_face_detected",
                "on(face-detected)",
                self.behaviors.event_face_detected.regime,
                self.behaviors.event_face_detected.hardcoded_id(),
                self.behaviors.event_face_detected.model_id(),
                self.behaviors.event_face_detected.fallback,
                vec![impl_id(
                    "script.on_face_detected.v0",
                    "Script hardcoded teacher",
                )],
                vec![impl_id("event.face_detected.shadow.v0", "Shadow model")],
                last("event_face_detected"),
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
            "event_face_detected" => update_behavior!(event_face_detected),
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
        "EventFaceDetected" => "event_face_detected".to_string(),
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
    pub event_face_detected: ReplaceableBehavior<FaceDetectedEventInput, EventScriptOutput>,
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
            event_face_detected: face_detected_event_behavior(
                BehaviorRegime::ShadowTrain,
                Some("event.face_detected.shadow.v0".to_string()),
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

struct FaceDetectedScriptBehavior;

impl FunctionBehavior<FaceDetectedEventInput, EventScriptOutput> for FaceDetectedScriptBehavior {
    fn id(&self) -> &'static str {
        "script.on_face_detected.v0"
    }

    fn infer(&mut self, input: &FaceDetectedEventInput) -> Result<EventScriptOutput> {
        let label = if input.recognized {
            input
                .person
                .name
                .as_ref()
                .filter(|name| !name.trim().is_empty())
                .cloned()
                .unwrap_or_else(|| format!("Acquaintance {}", input.person.id))
        } else {
            format!("Stranger {}", input.person.id)
        };
        Ok(EventScriptOutput {
            actions: vec![
                EventScriptAction::Chirp {
                    pattern: if input.recognized {
                        ChirpPattern::PersonRecognized
                    } else {
                        ChirpPattern::Hello
                    },
                },
                EventScriptAction::Say {
                    text: format!("Hello {label}"),
                },
            ],
        })
    }
}

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

fn face_detected_event_behavior(
    regime: BehaviorRegime,
    model_id: Option<String>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<FaceDetectedEventInput, EventScriptOutput> {
    ReplaceableBehavior::new(
        "event_face_detected",
        regime,
        Box::new(FaceDetectedScriptBehavior),
        model_id.map(|_| {
            Box::new(EventScriptShadowModel {
                id: "event.face_detected.shadow.v0",
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
    A: LlmAgent,
{
    pub fn new(
        ledger: L,
        memory_store: M,
        memory_recall: R,
        conductor: C,
        safety: S,
        llm: A,
    ) -> Self {
        Self {
            ledger,
            memory_store,
            memory_recall,
            conductor,
            safety,
            llm,
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
            last_active_control: None,
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
        Self {
            ledger,
            memory_store,
            memory_recall,
            conductor,
            safety,
            llm,
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
            last_active_control: None,
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

    pub fn behavior_node_states(&self) -> Vec<BehaviorNodeState> {
        self.models.behavior_node_states(&self.last_behavior_runs)
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
            self.run_event_scripts(&mut now, &recall, &mut notes, &mut proposed_actions)?;
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

        let combobulation = match self
            .llm
            .combobulate(
                &now,
                &impressions,
                Some(&embodied_context),
                &latent,
                &futures,
                &recall.first_person_summary,
            )
            .await
        {
            Ok(value) => value,
            Err(error) => {
                notes.push(format!("LlmCombobulationSkipped: {error}"));
                None
            }
        };

        let awareness_summary = combobulation.as_ref().map(|value| value.summary.as_str());
        let llm_tick = match self
            .llm
            .maybe_tick(
                &now,
                Some(&embodied_context),
                &latent,
                &futures,
                &recall.first_person_summary,
                awareness_summary,
            )
            .await
        {
            Ok(value) => value,
            Err(error) => {
                notes.push(format!("LlmTickSkipped: {error}"));
                LlmTickResult::default()
            }
        };
        now.llm = llm_tick.sense.clone();
        apply_llm_tick(
            &llm_tick,
            &mut sensations,
            &mut impressions,
            &mut experiences,
            &mut teachings,
        );
        let llm_command_action = llm_tick
            .decision
            .as_ref()
            .and_then(|decision| decision.action.clone());
        let mut llm_action_proposal = LlmActionProposal {
            proposed_action: llm_command_action.clone(),
            ignored_reason: if llm_command_action.is_none() {
                llm_tick
                    .decision
                    .as_ref()
                    .map(|_| "no executable action proposed".to_string())
            } else {
                None
            },
            ..LlmActionProposal::default()
        };
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
        world_context.active_control = self.last_active_control.clone();
        let enhanced_cognition_available = self.llm.enhanced_cognition_available();
        world_context.cognitive_services.insert(
            "rich_language".to_string(),
            CognitiveServiceBelief {
                available: enhanced_cognition_available,
                confidence: 1.0,
                unavailable_reason: (!enhanced_cognition_available).then(|| {
                    self.llm
                        .enhanced_cognition_unavailable_reason()
                        .unwrap_or("enhanced cognition is unavailable")
                        .to_string()
                }),
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
        let goal_cycle = self.goal_system.tick(&now.world, &proposals)?;
        now.drives = goal_cycle.drives.legacy_sense();
        let goal_action = goal_cycle
            .behavior
            .as_ref()
            .map(|behavior| behavior.action.clone());
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
            charger_near_score: sim_world_extension_score(&now, 3),
            charger_visible_score: sim_world_extension_score(&now, 4),
            proposals: proposals.clone(),
        })?;
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
            charger_near_score: sim_world_extension_score(&now, 3),
            charger_visible_score: sim_world_extension_score(&now, 4),
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
        let safety = self.safety.filter_action(
            &now,
            selected_goal_for_safety,
            &chosen_action,
            action_to_motor_command(Some(&chosen_action)),
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
        self.last_active_control = Some(ActiveControlSummary {
            goal_id: goal_cycle
                .selection
                .selected_goal
                .as_ref()
                .map(|goal| goal.as_str().to_string()),
            behavior_id: goal_cycle
                .behavior
                .as_ref()
                .map(|behavior| behavior.behavior_id.clone()),
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
                "desired_motor": action_to_motor_command(Some(&chosen_action)),
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
                "selected_action": action_selection.selected_action.clone(),
                "conductor_selected_action": conductor_selected_output.clone(),
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
            "ActionMotorBridge llm_action={:?} selected_action={:?} conductor_selected_action={:?} chosen_action={:?} desired_motor={:?} final_motor={:?} safety_override={}",
            llm_action_proposal.proposed_action,
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

        if let Some(combobulation) = &combobulation {
            append_combobulation(
                &mut sensations,
                &mut impressions,
                &mut experiences,
                now.t_ms,
                combobulation,
            );
        }

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
            recall,
            llm: llm_tick,
            combobulation,
            inline_learning,
        })
    }

    fn run_event_scripts(
        &mut self,
        now: &mut Now,
        recall: &RecallBundle,
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

        if let Some(input) = if self.chirp_events.last_face_present {
            None
        } else {
            face_detected_event_input(now, recall)
        } {
            let run = self
                .models
                .behaviors
                .event_face_detected
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
                "face-detected".to_string(),
                serde_json::to_value(&sequence)?,
            );
            if let Some(EventScriptAction::Say { text }) = run.chosen.actions.first() {
                notes.push(format!(
                    "EventScript:on(face-detected) emitted say({text:?})"
                ));
            }
            behavior_runs.push(run.record.erase());
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

fn face_detected_event_input(now: &Now, recall: &RecallBundle) -> Option<FaceDetectedEventInput> {
    if now.face.embeddings.is_empty() && now.face.vectors.is_empty() {
        return None;
    }
    let recognized = now.memory.face_familiarity >= 0.70 || !recall.hits.is_empty();
    let recalled_name = recall
        .hits
        .first()
        .map(|hit| hit.summary.trim().to_string())
        .filter(|summary| !summary.is_empty());
    let id = now
        .face
        .vectors
        .first()
        .map(|artifact| artifact.point_id.clone())
        .or_else(|| recalled_name.as_ref().map(|name| stable_person_id(name)))
        .unwrap_or_else(|| format!("face-{}", now.t_ms));
    Some(FaceDetectedEventInput {
        t_ms: now.t_ms,
        recognized,
        person: FacePerson {
            id,
            name: recognized.then_some(recalled_name).flatten(),
        },
    })
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

fn stable_person_id(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase()
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
    }) || sim_world_extension_score(now, 4) >= 0.5
}

fn face_present(now: &Now) -> bool {
    !now.face.embeddings.is_empty() || !now.face.vectors.is_empty()
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
}

#[async_trait::async_trait]
impl<L, M, R, C, S, A> RuntimeLoop for MinimalRuntime<L, M, R, C, S, A>
where
    L: LedgerWriter + Sync + Send,
    M: MemoryStore + Send,
    R: Recall + Send + Sync,
    C: Conductor + Send,
    S: SafetyLayer + Send,
    A: LlmAgent + Send,
{
    fn reign_sense(&self, now_ms: TimeMs) -> Result<ReignSense> {
        let mut reign_queue = self
            .reign_queue
            .lock()
            .map_err(|_| anyhow::anyhow!("reign queue lock poisoned"))?;
        reign_queue.drain_expired(now_ms);
        Ok(reign_queue.sense(now_ms))
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
    live_image_enricher: Option<LiveImageEnricher>,
    robot_initialization: Option<serde_json::Value>,
    brainstem_interface: Option<serde_json::Value>,
    possession_recovery: PossessionRecoveryState,
    motion_rejection: MotionRejectionState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PossessionRecoveryPhase {
    Idle,
    WaitingForSensorClear,
    Escaping,
    Stuck,
}

#[derive(Clone, Debug)]
struct PossessionRecoveryState {
    latch: Option<SafetyLatchKind>,
    phase: PossessionRecoveryPhase,
    turn_direction: TurnDir,
    active_since_ms: TimeMs,
    last_command_ms: TimeMs,
    command_attempts: u32,
    stuck_stop_sent: bool,
}

impl Default for PossessionRecoveryState {
    fn default() -> Self {
        Self {
            latch: None,
            phase: PossessionRecoveryPhase::Idle,
            turn_direction: TurnDir::Left,
            active_since_ms: 0,
            last_command_ms: 0,
            command_attempts: 0,
            stuck_stop_sent: false,
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

const POSSESSION_RECOVERY_COMMAND_COOLDOWN_MS: TimeMs = 1_000;
const POSSESSION_RECOVERY_STUCK_AFTER_MS: TimeMs = 15_000;
const POSSESSION_RECOVERY_MAX_ATTEMPTS: u32 = 3;
const POSSESSION_BUMP_ESCAPE_BACKOFF_MM_S: i16 = 50;
const POSSESSION_BUMP_ESCAPE_TURN_MRAD_S: i16 = 500;
const POSSESSION_BUMP_ESCAPE_SETTLE_MS: TimeMs = 250;
const POSSESSION_BUMP_ESCAPE_DURATION_MS: TimeMs =
    bump_escape_duration_ms(POSSESSION_BUMP_ESCAPE_TURN_MRAD_S) as TimeMs
        + POSSESSION_BUMP_ESCAPE_SETTLE_MS;
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
            live_image_enricher: None,
            robot_initialization: None,
            brainstem_interface: None,
            possession_recovery: PossessionRecoveryState::default(),
            motion_rejection: MotionRejectionState::default(),
        }
    }

    pub fn with_frame_processor(mut self, frame_processor: FrameProcessor) -> Self {
        self.frame_processor = frame_processor;
        self
    }

    pub fn with_live_image_enricher(mut self, enricher: Option<LiveImageEnricher>) -> Self {
        self.live_image_enricher = enricher;
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
        let mut packets = poll_sensors_lossy(&mut self.sensors).await;
        let t_ms = body.last_update_ms.max(wall_time_ms());
        self.frame_processor.process_packets(t_ms, &mut packets);
        let mut now = self.now_builder.build(t_ms, body, packets)?;
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
        enrich_now_latest_image(&mut self.live_image_enricher, &mut now).await;

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
        let mut packets = poll_sensors_lossy(&mut self.sensors).await;
        let t_ms = body_before.last_update_ms.max(wall_time_ms());
        self.frame_processor.process_packets(t_ms, &mut packets);
        let mut now = self.now_builder.build(t_ms, body_before.clone(), packets)?;
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

        enrich_now_latest_image(&mut self.live_image_enricher, &mut now).await;

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
        if !recovery_decision.command_sent {
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
                serde_json::to_value(motor_command_to_motion(
                    recovery_decision.motor.unwrap_or(final_motor),
                ))?,
            );
            object.insert(
                "motion_sent_to_sim".to_string(),
                serde_json::to_value(motor_command_to_motion(final_motor))?,
            );
            object.insert(
                "motor_applied".to_string(),
                serde_json::json!(recovery_decision
                    .motor
                    .map(|motor| !is_near_zero_motor(motor))
                    .unwrap_or(!is_near_zero_motor(final_motor))),
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
            match event.kind {
                CockpitEventKind::SafetyTripped => {
                    if let Some(kind) = safety_latch_kind_from_event_code(event.a) {
                        self.start_possession_recovery(kind, body);
                    }
                }
                CockpitEventKind::SafetyCleared => {
                    if self.possession_recovery.latch == safety_latch_kind_from_event_code(event.a)
                    {
                        self.possession_recovery = PossessionRecoveryState::default();
                    }
                }
                CockpitEventKind::EStopLatched => {
                    self.possession_recovery = PossessionRecoveryState::default();
                }
                _ => {}
            }
        }

        if self.possession_recovery.latch.is_none()
            && status.safety_tripped == Some(true)
            && status.estop_latched != Some(true)
        {
            if let Some(kind) = status
                .safety_latch_kind
                .or_else(|| infer_safety_latch_from_sensors(body))
            {
                self.start_possession_recovery(kind, body);
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

        let mut command_sent = true;
        let mut action = None;
        let mut motor = None;
        let now_ms = wall_time_ms();
        let recovery_age_ms = recovery_age_ms(&self.possession_recovery, now_ms);
        let mut reason = format!("recovering {latch:?} safety latch");
        match latch {
            SafetyLatchKind::Bump => {
                let escape_elapsed_ms =
                    now_ms.saturating_sub(self.possession_recovery.last_command_ms);
                if self.possession_recovery.phase == PossessionRecoveryPhase::Escaping
                    && escape_elapsed_ms < POSSESSION_BUMP_ESCAPE_DURATION_MS
                {
                    command_sent = true;
                    if escape_elapsed_ms < BUMP_ESCAPE_BACKOFF_DURATION_MS as TimeMs {
                        action = Some(ActionPrimitive::Go {
                            intensity: -0.2,
                            duration_ms: BUMP_ESCAPE_BACKOFF_DURATION_MS as TimeMs,
                        });
                        motor = Some(MotorCommand {
                            forward: -0.05,
                            turn: 0.0,
                        });
                        reason = format!(
                            "recovering {latch:?} safety latch; reversing for escape attempt {}",
                            self.possession_recovery.command_attempts
                        );
                    } else {
                        action = Some(ActionPrimitive::Turn {
                            direction: TurnDir::Right,
                            intensity: 0.5,
                            duration_ms: POSSESSION_BUMP_ESCAPE_DURATION_MS
                                .saturating_sub(BUMP_ESCAPE_BACKOFF_DURATION_MS as TimeMs),
                        });
                        motor = Some(MotorCommand {
                            forward: 0.0,
                            turn: -0.5,
                        });
                        reason = format!(
                            "recovering {latch:?} safety latch; turning clockwise 90 degrees for escape attempt {}",
                            self.possession_recovery.command_attempts
                        );
                    }
                } else if self.possession_recovery.phase == PossessionRecoveryPhase::Escaping
                    && !bump_active(body)
                {
                    self.cockpit.client_mut().stop()?;
                    self.cockpit.client_mut().clear_safety_latch(latch)?;
                    self.possession_recovery = PossessionRecoveryState::default();
                    action = Some(ActionPrimitive::Stop);
                    motor = Some(MotorCommand::stop());
                    reason = format!(
                        "recovered {latch:?} safety latch after reverse and clockwise 90 degree turn"
                    );
                } else {
                    self.possession_recovery.phase = PossessionRecoveryPhase::WaitingForSensorClear;
                    action = Some(ActionPrimitive::Go {
                        intensity: -0.2,
                        duration_ms: BUMP_ESCAPE_BACKOFF_DURATION_MS as TimeMs,
                    });
                    motor = Some(MotorCommand {
                        forward: -0.05,
                        turn: 0.0,
                    });
                    if possession_recovery_is_stuck(&self.possession_recovery, now_ms) {
                        self.possession_recovery.phase = PossessionRecoveryPhase::Stuck;
                        action = Some(ActionPrimitive::Stop);
                        motor = Some(MotorCommand::stop());
                        reason = format!(
                            "{latch:?} recovery stuck after reverse and clockwise turns; bumper still active after {} ms and {} escape attempts; operator intervention needed",
                            recovery_age_ms, self.possession_recovery.command_attempts
                        );
                        if !self.possession_recovery.stuck_stop_sent {
                            self.cockpit.client_mut().stop()?;
                            self.possession_recovery.stuck_stop_sent = true;
                        }
                    } else if possession_recovery_command_due(&self.possession_recovery, now_ms) {
                        self.possession_recovery.command_attempts =
                            self.possession_recovery.command_attempts.saturating_add(1);
                        self.possession_recovery.last_command_ms = now_ms;
                        self.possession_recovery.phase = PossessionRecoveryPhase::Escaping;
                        reason = format!(
                            "recovering {latch:?} safety latch; escape attempt {}: reverse then turn clockwise 90 degrees",
                            self.possession_recovery.command_attempts
                        );
                        self.cockpit.client_mut().bump_escape(
                            EscapeDirection::Right,
                            POSSESSION_BUMP_ESCAPE_BACKOFF_MM_S,
                            POSSESSION_BUMP_ESCAPE_TURN_MRAD_S,
                        )?;
                    } else {
                        reason = format!(
                            "recovering {latch:?} safety latch; escape in progress for {} ms",
                            now_ms.saturating_sub(self.possession_recovery.last_command_ms)
                        );
                    }
                }
            }
            SafetyLatchKind::Cliff => {
                if cliff_active(body) {
                    self.possession_recovery.phase = PossessionRecoveryPhase::WaitingForSensorClear;
                    self.possession_recovery.turn_direction =
                        recovery_turn_direction_for_latch(latch, body);
                    action = Some(ActionPrimitive::Go {
                        intensity: -0.25,
                        duration_ms: 900,
                    });
                    motor = Some(MotorCommand {
                        forward: -0.10,
                        turn: 0.0,
                    });
                    if possession_recovery_is_stuck(&self.possession_recovery, now_ms) {
                        self.possession_recovery.phase = PossessionRecoveryPhase::Stuck;
                        action = Some(ActionPrimitive::Stop);
                        motor = Some(MotorCommand::stop());
                        reason = format!(
                            "{latch:?} recovery stuck; cliff still active after {} ms and {} escape attempts; operator intervention needed",
                            recovery_age_ms, self.possession_recovery.command_attempts
                        );
                        if !self.possession_recovery.stuck_stop_sent {
                            self.cockpit.client_mut().stop()?;
                            self.possession_recovery.stuck_stop_sent = true;
                        }
                    } else if possession_recovery_command_due(&self.possession_recovery, now_ms) {
                        self.possession_recovery.command_attempts =
                            self.possession_recovery.command_attempts.saturating_add(1);
                        self.possession_recovery.last_command_ms = now_ms;
                        reason = format!(
                            "recovering {latch:?} safety latch; escape attempt {}",
                            self.possession_recovery.command_attempts
                        );
                        self.cockpit.client_mut().bump_escape(
                            escape_direction(self.possession_recovery.turn_direction.clone()),
                            100,
                            900,
                        )?;
                    } else {
                        reason = format!(
                            "recovering {latch:?} safety latch; escape in progress for {} ms",
                            now_ms.saturating_sub(self.possession_recovery.last_command_ms)
                        );
                    }
                } else {
                    self.possession_recovery.phase = PossessionRecoveryPhase::Escaping;
                    self.cockpit.client_mut().clear_safety_latch(latch)?;
                    self.possession_recovery = PossessionRecoveryState::default();
                }
            }
            SafetyLatchKind::WheelDrop => {
                if body.flags.wheel_drop {
                    action = Some(ActionPrimitive::Stop);
                    motor = Some(MotorCommand::stop());
                    self.cockpit.client_mut().stop()?;
                } else {
                    self.cockpit.client_mut().clear_safety_latch(latch)?;
                    self.possession_recovery = PossessionRecoveryState::default();
                }
            }
            SafetyLatchKind::Charging => {
                if body.charging {
                    action = Some(ActionPrimitive::Stop);
                    motor = Some(MotorCommand::stop());
                    self.cockpit.client_mut().stop()?;
                } else {
                    self.cockpit.client_mut().clear_safety_latch(latch)?;
                    self.possession_recovery = PossessionRecoveryState::default();
                }
            }
            SafetyLatchKind::Heartbeat => {
                self.cockpit.client_mut().clear_safety_latch(latch)?;
                self.possession_recovery = PossessionRecoveryState::default();
                command_sent = false;
            }
            SafetyLatchKind::Tilt | SafetyLatchKind::Impact => {
                if imu_recovery_clear(status, latch) {
                    self.cockpit.client_mut().clear_safety_latch(latch)?;
                    self.possession_recovery = PossessionRecoveryState::default();
                    command_sent = false;
                } else {
                    action = Some(ActionPrimitive::Stop);
                    motor = Some(MotorCommand::stop());
                    self.cockpit.client_mut().stop()?;
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

    fn start_possession_recovery(&mut self, latch: SafetyLatchKind, body: &BodySense) {
        let now_ms = wall_time_ms();
        let latch_changed = self.possession_recovery.latch != Some(latch);
        self.possession_recovery.latch = Some(latch);
        self.possession_recovery.phase = PossessionRecoveryPhase::WaitingForSensorClear;
        self.possession_recovery.turn_direction = recovery_turn_direction_for_latch(latch, body);
        if latch_changed || self.possession_recovery.active_since_ms == 0 {
            self.possession_recovery.active_since_ms = now_ms;
            self.possession_recovery.last_command_ms = 0;
            self.possession_recovery.command_attempts = 0;
            self.possession_recovery.stuck_stop_sent = false;
        }
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

async fn enrich_now_latest_image(enricher: &mut Option<LiveImageEnricher>, now: &mut Now) {
    let Some(enricher) = enricher.as_mut() else {
        return;
    };
    let Some(frame) = now.eye_frame.as_ref() else {
        return;
    };
    match enricher.enrich_latest(frame).await {
        Ok(Some(enrichment)) => {
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
        Ok(None) => {}
        Err(error) => {
            now.extensions.insert(
                "vision.image_enrichment_error".to_string(),
                serde_json::json!(error.to_string()),
            );
        }
    }
}

async fn poll_sensors_lossy(
    sensors: &mut [Box<dyn SenseProducer + Send>],
) -> Vec<pete_sensors::SensePacket> {
    let mut packets = Vec::new();
    for sensor in sensors {
        match tokio::time::timeout(std::time::Duration::from_millis(25), sensor.poll()).await {
            Ok(Ok(packet)) => packets.push(packet),
            Ok(Err(error)) => eprintln!("sensor poll failed; continuing without packet: {error}"),
            Err(_) => eprintln!("sensor poll timed out; continuing without packet"),
        }
    }
    packets
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
        flags: BodyFlags {
            bump_left: status.contact.bump_left.unwrap_or(false),
            bump_right: status.contact.bump_right.unwrap_or(false),
            cliff_left: status.contact.cliff_left.unwrap_or(false),
            cliff_front_left: status.contact.cliff_front_left.unwrap_or(false),
            cliff_front_right: status.contact.cliff_front_right.unwrap_or(false),
            cliff_right: status.contact.cliff_right.unwrap_or(false),
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

fn escape_direction(direction: TurnDir) -> EscapeDirection {
    match direction {
        TurnDir::Left => EscapeDirection::Left,
        TurnDir::Right => EscapeDirection::Right,
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

fn recovery_age_ms(state: &PossessionRecoveryState, now_ms: TimeMs) -> TimeMs {
    if state.active_since_ms == 0 {
        0
    } else {
        now_ms.saturating_sub(state.active_since_ms)
    }
}

fn possession_recovery_command_due(state: &PossessionRecoveryState, now_ms: TimeMs) -> bool {
    state.last_command_ms == 0
        || now_ms.saturating_sub(state.last_command_ms) >= POSSESSION_RECOVERY_COMMAND_COOLDOWN_MS
}

fn possession_recovery_is_stuck(state: &PossessionRecoveryState, now_ms: TimeMs) -> bool {
    recovery_age_ms(state, now_ms) >= POSSESSION_RECOVERY_STUCK_AFTER_MS
        || state.command_attempts >= POSSESSION_RECOVERY_MAX_ATTEMPTS
}

fn possession_recovery_debug(
    state: &PossessionRecoveryState,
    active_latch: Option<SafetyLatchKind>,
    command_sent: bool,
) -> serde_json::Value {
    serde_json::json!({
        "latched": active_latch.or(state.latch).map(|latch| format!("{latch:?}")),
        "phase": format!("{:?}", state.phase),
        "turn_direction": format!("{:?}", state.turn_direction),
        "active_since_ms": state.active_since_ms,
        "last_command_ms": state.last_command_ms,
        "command_attempts": state.command_attempts,
        "stuck_stop_sent": state.stuck_stop_sent,
        "command_sent": command_sent,
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

        if self.active && step_distance > STUCK_WINDOW_DISPLACEMENT_EPSILON_M {
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
            let now = snapshot.to_now(snapshot.body.last_update_ms);
            let tick = self
                .runtime
                .tick(now, ExperienceLatent::default(), Vec::new())
                .await?;
            let final_motor = final_motor_from_tick(&tick);
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
            } else {
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
    let charger_near = sim_world_extension_score(now, 3);
    let charger_visible = sim_world_extension_score(now, 4);
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
    let charge_confidence = sim_world_extension_score(now, 3)
        .max(sim_world_extension_score(now, 4))
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
    now.body.charging
        || sim_world_extension_score(now, 3) >= 0.25
        || sim_world_extension_score(now, 4) >= 0.20
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
    sensations: &mut Vec<Sensation>,
    impressions: &mut Vec<Impression>,
    experiences: &mut Vec<Experience>,
    teachings: &mut Vec<pete_llm::LlmTeaching>,
) {
    if let Some(command) = &llm_tick.conscious_command {
        let sensation = Sensation::new(
            "llm.command",
            "llm",
            llm_tick
                .teaching
                .first()
                .map(|value| value.t_ms)
                .unwrap_or_default(),
            llm_tick
                .teaching
                .first()
                .map(|value| value.t_ms)
                .unwrap_or_default(),
            serde_json::json!({
                "summary": command.summary,
                "action": command.action,
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
            llm_tick
                .teaching
                .first()
                .map(|value| value.t_ms)
                .unwrap_or_default(),
            llm_tick
                .teaching
                .first()
                .map(|value| value.t_ms)
                .unwrap_or_default(),
            serde_json::json!({ "critique": critique }),
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
    t_ms: u64,
    combobulation: &Combobulation,
) {
    let sensation = Sensation::new(
        "llm.combobulation",
        "llm",
        t_ms,
        t_ms,
        serde_json::json!({
            "summary": combobulation.summary,
            "confidence": combobulation.confidence,
        }),
    )
    .with_summary(combobulation.summary.clone())
    .with_provenance(Provenance::direct().with_stage("combobulator"));
    let impression = Impression::new(
        "llm.combobulation.observation",
        combobulation.summary.clone(),
        vec![sensation.id],
        t_ms,
        t_ms,
    )
    .with_confidence(combobulation.confidence);
    let experience = Experience::new(
        "llm.combobulation",
        combobulation.summary.clone(),
        vec![impression.id],
        vec![sensation.id],
        t_ms,
        t_ms,
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
            "I have {} face embeddings, {} face vectors, {} voice embeddings, and {} voice vectors available for recognizing who may be present.",
            now.face.embeddings.len(),
            now.face.vectors.len(),
            now.voice.embeddings.len(),
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
mod tests {
    use super::*;
    use pete_actions::{ChirpPattern, ReignCommand, ReignMode, ReignSource};
    use pete_autonomic::SimpleSafety;
    use pete_body::BodySense;
    use pete_cockpit::{
        establish_session, CockpitCapabilities, CockpitRequest, CockpitResponse, CockpitStatus,
        EventBatch, HandshakeHello, MotherbrainPossession, SimCockpit,
    };
    use pete_conductor::{Conductor, ConductorInput, SimpleConductor};
    use pete_experience::{
        embody_now, experience_encode_input_from_now, EmbodiedContext, Modality,
        SensationPayloadKind,
    };
    use pete_ledger::{ExperienceFrame, ExperienceTransition, JsonlLedger, LedgerReader};
    use pete_llm::{
        ConsciousCommand, LlmDecision, LlmReviewRequest, LlmScientificReview, LlmTickResult,
    };
    use pete_map::MapConfig;
    use pete_memory::InMemoryExperienceStore;
    use pete_models::{
        ActionValueNetTrainer, ChargeNetTrainer, DangerNetTrainer, EarNextNetTrainer,
        ExperienceAutoencoderTrainer, FutureNetTrainer,
    };
    use pete_now::{Now, SurpriseSense, VectorArtifact, SCENE_VECTOR_COLLECTION};
    use pete_sensors::World;
    use pete_sim::{
        build_scenario, ArenaConfig, ScenarioConfig, ScenarioKind, SimObject, VirtualWorld,
    };
    use serde_json::Value;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn mark_corrected_map_trusted(now: &mut Now) {
        now.extensions.insert(
            MAP_EXTENSION_NAME.to_string(),
            serde_json::json!({
                "slam_status": {
                    "mode": "loop_closed_pose_graph",
                    "local_scan_matching_active": true,
                    "loop_closure_active": true,
                    "pose_graph_optimized": true,
                    "occupancy_remapped_from_pose_graph": true,
                    "reasons": []
                },
                "loop_closures_accepted": 1,
                "pose_graph_optimization": {
                    "optimized_nodes": 2,
                    "active_edges": 2
                },
                "remap": {
                    "generation": 1,
                    "submaps": 1
                }
            }),
        );
    }

    #[test]
    fn physical_charging_indicator_populates_body_charging_without_oi_state() {
        let status = CockpitStatus {
            raw: serde_json::json!({
                "uptime_ms": 1_000,
                "current_runtime_state": "idle",
                "oi_mode": "safe",
                "create_sensors": {
                    "last_packet_id": 0,
                    "complete_packet_count": 1,
                    "last_complete_packet_timestamp_ms": 1_000,
                    "charging_state": 0,
                    "charging_indicator": "on"
                }
            })
            .to_string(),
        }
        .summary();

        let body = body_sense_from_cockpit_status(status, 123);

        assert!(body.charging);
        assert_eq!(body.last_update_ms, 123);
    }

    #[test]
    fn body_timestamp_tracks_complete_create_packet_age() {
        let status = CockpitStatus {
            raw: serde_json::json!({
                "uptime_ms": 2_000,
                "current_runtime_state": "idle",
                "oi_mode": "safe",
                "create_sensors": {
                    "last_packet_id": 0,
                    "complete_packet_count": 4,
                    "last_complete_packet_timestamp_ms": 1_250,
                    "bump_left": false
                }
            })
            .to_string(),
        }
        .summary();

        let body = body_sense_from_cockpit_status(status, 10_000);

        assert_eq!(body.last_update_ms, 9_250);
        let decision = SimpleSafety::default().filter(
            &Now::blank(10_000, body),
            MotorCommand {
                forward: 0.2,
                turn: 0.0,
            },
        );
        assert_eq!(decision.reason, Some(SafetyReason::StaleSensors));
        assert_eq!(decision.command, MotorCommand::stop());
    }

    #[test]
    fn incomplete_create_packet_never_refreshes_body_timestamp() {
        let status = CockpitStatus {
            raw: serde_json::json!({
                "uptime_ms": 2_000,
                "last_uart_packet_timestamp_ms": 1_990,
                "uart_rx_packets": 4,
                "current_runtime_state": "idle",
                "create_sensors": {
                    "last_packet_id": 35,
                    "complete_packet_count": 0
                }
            })
            .to_string(),
        }
        .summary();

        assert_eq!(
            body_sense_from_cockpit_status(status, 10_000).last_update_ms,
            0
        );
    }

    #[test]
    fn map_memory_debug_records_intent_confidence_and_signal() {
        let mut now = Now::blank(100, BodySense::default());
        mark_corrected_map_trusted(&mut now);
        now.memory.place_danger = 0.9;
        now.memory.nearby_best_safe_direction_rad = Some(-0.8);
        let action = ActionPrimitive::Turn {
            direction: TurnDir::Right,
            intensity: 0.5,
            duration_ms: 1_000,
        };

        let debug = map_memory_decision_debug(&now, &action, Some(&action), false);

        assert!(debug.influenced);
        assert_eq!(
            debug.navigation_intent,
            Some(NavigationIntent::AvoidKnownDangerCell)
        );
        assert_eq!(debug.reason.as_deref(), Some("danger_safe_direction"));
        assert_eq!(
            debug.signal.as_deref(),
            Some("memory.nearby_best_safe_direction_rad")
        );
        assert_eq!(debug.signal_value, Some(-0.8));
        assert_eq!(debug.signal_confidence, 0.9);
        assert_eq!(debug.confidence, 0.9);
        assert_eq!(debug.chosen_action.as_ref(), Some(&action));
        assert!(!debug.safety_overrode);
        assert!(debug.reason_string.unwrap().contains("remembered danger"));
    }

    #[test]
    fn map_memory_debug_records_low_confidence_charge_fallback() {
        let mut body = BodySense::default();
        body.battery_level = 0.05;
        let mut now = Now::blank(100, body);
        mark_corrected_map_trusted(&mut now);
        now.memory.place_charge_value = 0.1;
        let action = ActionPrimitive::Stop;

        let debug = map_memory_decision_debug(&now, &action, Some(&action), false);

        assert!(debug.influenced);
        assert_eq!(
            debug.navigation_intent,
            Some(NavigationIntent::StopAskForHelpWhenUncertain)
        );
        assert_eq!(
            debug.reason.as_deref(),
            Some("charge_low_confidence_fallback")
        );
        assert_eq!(
            debug.signal.as_deref(),
            Some("memory.nearby_best_charge_direction_rad")
        );
        assert_eq!(debug.signal_value, None);
        assert!(debug.signal_confidence < 0.35);
        assert_eq!(debug.chosen_action, Some(ActionPrimitive::Stop));
        assert!(debug
            .reason_string
            .as_deref()
            .unwrap_or_default()
            .contains("too weak"));
    }

    #[test]
    fn map_memory_debug_rejects_navigation_when_corrected_map_is_untrusted() {
        let mut now = Now::blank(100, BodySense::default());
        now.memory.place_danger = 0.9;
        now.memory.nearby_best_safe_direction_rad = Some(-0.8);
        let action = ActionPrimitive::Turn {
            direction: TurnDir::Right,
            intensity: 0.5,
            duration_ms: 1_000,
        };

        let debug = map_memory_decision_debug(&now, &action, Some(&action), false);

        assert!(!debug.influenced);
        assert!(!debug.corrected_map_trusted);
        assert!(debug
            .corrected_map_untrusted_reason
            .as_deref()
            .unwrap_or_default()
            .contains("summary is missing"));
        assert!(!memory_navigation_candidate_context(&now, &action));
    }

    #[test]
    fn map_memory_debug_rejects_local_scan_match_without_loop_closed_slam() {
        let mut now = Now::blank(100, BodySense::default());
        now.extensions.insert(
            MAP_EXTENSION_NAME.to_string(),
            serde_json::json!({
                "slam_status": {
                    "mode": "local_scan_matched",
                    "local_scan_matching_active": true,
                    "loop_closure_active": false,
                    "pose_graph_optimized": true,
                    "occupancy_remapped_from_pose_graph": true,
                    "reasons": ["no loop-closure candidate has been accepted yet"]
                }
            }),
        );
        now.memory.place_danger = 0.9;
        now.memory.nearby_best_safe_direction_rad = Some(-0.8);
        let action = ActionPrimitive::Turn {
            direction: TurnDir::Right,
            intensity: 0.5,
            duration_ms: 1_000,
        };

        let debug = map_memory_decision_debug(&now, &action, Some(&action), false);

        assert!(!debug.influenced);
        assert!(!debug.corrected_map_trusted);
        assert!(debug
            .corrected_map_untrusted_reason
            .as_deref()
            .unwrap_or_default()
            .contains("slam_status.mode is local_scan_matched"));
        assert!(!memory_navigation_candidate_context(&now, &action));
    }

    fn idle_now(t_ms: u64) -> Now {
        let mut body = BodySense::default();
        body.last_update_ms = t_ms;
        let mut now = Now::blank(t_ms, body);
        now.range.nearest_m = Some(1.0);
        now.range.beams = vec![1.0, 1.0, 1.0];
        now
    }

    fn mapped_scene_now(t_ms: u64, x_m: f32, point_id: &str) -> Now {
        let mut body = test_body(x_m, 0.0, 0.8, t_ms);
        body.odometry.heading_rad = 0.0;
        let mut now = Now::blank(t_ms, body);
        now.range.nearest_m = Some(1.0);
        now.range.beams = vec![1.0];
        now.eye.scene_vectors =
            vec![
                VectorArtifact::new(SCENE_VECTOR_COLLECTION, point_id, vec![1.0, 0.0, 0.0])
                    .with_occurred_at_ms(t_ms),
            ];
        now
    }

    #[tokio::test]
    async fn ledger_frame_with_asr_metadata_shows_audio_child_sensations() {
        let root = test_ledger_root("asr-audio-child-sensations");
        let ledger = JsonlLedger::new(&root);
        let mut now = Now::blank(1_000, BodySense::default());
        now.ear.asr = pete_now::AsrSense {
            transcript: Some("hello from replay".to_string()),
            is_final: true,
            confidence: 0.84,
            start_ms: Some(250),
            end_ms: Some(950),
            duration_ms: Some(700),
            word_count: Some(3),
            ..pete_now::AsrSense::default()
        };

        let embodied = embody_now(&now).await.unwrap();
        let frame = ExperienceFrame {
            id: Uuid::new_v4(),
            t_ms: now.t_ms,
            now,
            sensations: embodied.sensations,
            impressions: embodied.impressions,
            experiences: vec![embodied.experience],
            z: None,
            chosen_action: None,
            conscious_command: None,
            reign_input: None,
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
            notes: vec!["asr ledger smoke".to_string()],
        };
        ledger.append(&frame).await.unwrap();

        let frames = ledger.recent(1).await.unwrap();
        let readback = frames.first().expect("ledger frame");
        assert!(readback.sensations.iter().any(|sensation| {
            sensation.payload_kind == SensationPayloadKind::SpeechSegment
                && sensation.parent_id.is_some()
                && sensation.payload.get("text").and_then(Value::as_str)
                    == Some("hello from replay")
        }));
        assert!(readback.sensations.iter().any(|sensation| {
            sensation.payload_kind == SensationPayloadKind::TranscriptSpan
                && sensation.parent_id.is_some()
        }));
    }

    #[tokio::test]
    async fn embodied_eval_coverage_contract_reports_memory_recall() {
        let report = pete_memory::deterministic_embodied_eval_report()
            .await
            .unwrap();

        assert!(report.passed(), "{:?}", report.failures);
        assert!(report.experience_latent_count > 0);
        assert!(report.summary_impression_count > 0);
        assert!(report.prediction_count > 0);
        assert!(report.memory_link_count > 0);
        assert!(report.recall_sensation_count > 0);
        assert!(report.recall_impression_count > 0);
        assert!(report.lineage_edge_count > 0);
    }

    fn test_conductor_input(action: ActionPrimitive) -> ConductorInput {
        ConductorInput {
            latent: ExperienceLatent::default(),
            drives: DriveSense::default(),
            memory: pete_now::MemorySense::default(),
            predictions: pete_now::PredictionSense::default(),
            surprise: SurpriseSense::default(),
            llm: pete_now::LlmSense::default(),
            safety: SafetySense::default(),
            reign: ReignSense::default(),
            range: pete_now::RangeSense::default(),
            body: BodySense::default(),
            charger_near_score: 0.0,
            charger_visible_score: 0.0,
            proposals: vec![action],
        }
    }

    #[test]
    fn conductor_shadow_train_returns_reign_teacher_action_and_observes_model() {
        let teacher_action = ActionPrimitive::Turn {
            direction: TurnDir::Right,
            intensity: 0.4,
            duration_ms: 500,
        };
        let mut input = test_conductor_input(ActionPrimitive::Stop);
        input.reign.active = true;
        input.reign.latest = Some(ReignInput {
            id: Uuid::new_v4(),
            issued_at_ms: 10,
            expires_at_ms: 500,
            source: ReignSource::WebRemote,
            mode: ReignMode::Direct,
            command: ReignCommand::Turn {
                direction: TurnDir::Right,
                intensity: 0.4,
                duration_ms: 500,
            },
            priority: 1.0,
            note: None,
        });
        let mut behavior = conductor_behavior(
            BehaviorRegime::ShadowTrain,
            "reign.teacher",
            Some("conductor.burn.v0".to_string()),
            FallbackPolicy::UseHardcoded,
        );

        let run = behavior
            .infer_with_teacher_source(&input, 10, TrainingSource::HumanReign)
            .unwrap();

        assert_eq!(run.chosen, teacher_action);
        assert!(run.training_sample_emitted);
        assert_eq!(run.record.model_output, None);
    }

    #[test]
    fn conductor_model_infer_falls_back_to_hardcoded_when_model_has_no_sample() {
        let teacher_action = ActionPrimitive::Dock;
        let input = test_conductor_input(teacher_action.clone());
        let mut behavior = conductor_behavior(
            BehaviorRegime::ModelInfer,
            "action_selector.baseline",
            Some("conductor.burn.v0".to_string()),
            FallbackPolicy::UseHardcoded,
        );

        let run = behavior.infer(&input, 10).unwrap();

        assert_eq!(run.chosen, teacher_action);
        assert!(run.fallback_used);
        assert!(run.record.error.is_some());
    }

    #[test]
    fn bump_script_hardcoded_returns_escape_sequence() {
        let mut behavior = bump_event_behavior(
            BehaviorRegime::Hardcoded,
            Some("event.bump.shadow.v0".to_string()),
            FallbackPolicy::UseHardcoded,
        );

        let run = behavior.infer(&BumpEventInput::default(), 10).unwrap();

        assert!(run.chosen.actions.iter().any(is_bump_lament_action));
        let recovery_actions = run
            .chosen
            .actions
            .iter()
            .filter(|action| {
                matches!(
                    action,
                    EventScriptAction::Stop
                        | EventScriptAction::Rotate { .. }
                        | EventScriptAction::Go
                )
            })
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            recovery_actions,
            vec![
                EventScriptAction::Stop,
                EventScriptAction::Rotate { deg: 180 },
                EventScriptAction::Go,
            ]
        );
    }

    #[test]
    fn bump_script_shadow_train_returns_teacher_and_observes_model() {
        let mut behavior = bump_event_behavior(
            BehaviorRegime::ShadowTrain,
            Some("event.bump.shadow.v0".to_string()),
            FallbackPolicy::UseHardcoded,
        );

        let run = behavior.infer(&BumpEventInput::default(), 10).unwrap();

        assert!(run.chosen.actions.iter().any(is_bump_lament_action));
        let recovery_actions = run
            .chosen
            .actions
            .iter()
            .filter(|action| {
                matches!(
                    action,
                    EventScriptAction::Stop
                        | EventScriptAction::Rotate { .. }
                        | EventScriptAction::Go
                )
            })
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(
            recovery_actions,
            vec![
                EventScriptAction::Stop,
                EventScriptAction::Rotate { deg: 180 },
                EventScriptAction::Go,
            ]
        );
        assert!(run.training_sample_emitted);
        assert_eq!(run.record.hardcoded_output, Some(run.chosen));
    }

    fn is_bump_lament_action(action: &EventScriptAction) -> bool {
        match action {
            EventScriptAction::Say { text } => {
                matches!(text.as_str(), "Uh-oh" | "Oh no!" | "Oopsie!" | "Oh dear!")
            }
            EventScriptAction::Song { name } => name == "mournful_bump",
            _ => false,
        }
    }

    #[test]
    fn face_detected_script_greets_named_unnamed_and_stranger_faces() {
        let mut behavior = face_detected_event_behavior(
            BehaviorRegime::Hardcoded,
            Some("event.face_detected.shadow.v0".to_string()),
            FallbackPolicy::UseHardcoded,
        );
        let named = FaceDetectedEventInput {
            t_ms: 10,
            recognized: true,
            person: FacePerson {
                id: "p1".to_string(),
                name: Some("Ada".to_string()),
            },
        };
        let unnamed = FaceDetectedEventInput {
            t_ms: 10,
            recognized: true,
            person: FacePerson {
                id: "p2".to_string(),
                name: None,
            },
        };
        let stranger = FaceDetectedEventInput {
            t_ms: 10,
            recognized: false,
            person: FacePerson {
                id: "p3".to_string(),
                name: None,
            },
        };

        assert!(behavior.infer(&named, 10).unwrap().chosen.actions.contains(
            &EventScriptAction::Say {
                text: "Hello Ada".to_string()
            }
        ));
        assert!(behavior
            .infer(&unnamed, 10)
            .unwrap()
            .chosen
            .actions
            .contains(&EventScriptAction::Say {
                text: "Hello Acquaintance p2".to_string()
            }));
        assert!(behavior
            .infer(&stranger, 10)
            .unwrap()
            .chosen
            .actions
            .contains(&EventScriptAction::Say {
                text: "Hello Stranger p3".to_string()
            }));
    }

    #[test]
    fn safety_veto_prevents_unsafe_script_movement_and_records_context() {
        let mut now = idle_now(10);
        now.body.battery_level = 0.05;
        let mut safety = SimpleSafety::default();
        let output = EventScriptOutput {
            actions: vec![
                EventScriptAction::Stop,
                EventScriptAction::Rotate { deg: 180 },
                EventScriptAction::Go,
            ],
        };

        let sequence = safety_trace_script_actions(&mut safety, &now, &output);

        let go = sequence.actions.last().unwrap();
        assert_eq!(go.requested, EventScriptAction::Go);
        assert!(go.vetoed);
        assert_eq!(go.final_motor, MotorCommand::stop());
        assert_eq!(go.safety_reason.as_deref(), Some("critical battery"));
    }

    fn prime_idle(controller: &mut NudgeController, now: &Now, policy: NudgePolicy) {
        let mut first = now.clone();
        first.t_ms = now.t_ms.saturating_sub(policy.idle_after_ms);
        first.body.last_update_ms = first.t_ms;
        assert!(controller.propose(&first, policy).is_none());
    }

    #[test]
    fn nudge_refuses_wheel_drop() {
        let policy = NudgePolicy::virtual_default();
        let mut controller = NudgeController::default();
        let mut now = idle_now(5_000);
        now.body.flags.wheel_drop = true;
        prime_idle(&mut controller, &now, policy);

        assert!(controller.propose(&now, policy).is_none());
        assert_eq!(
            controller.status.nudge_blocked_reason.as_deref(),
            Some("wheel drop detected")
        );
    }

    #[test]
    fn nudge_refuses_critical_battery() {
        let policy = NudgePolicy::virtual_default();
        let mut controller = NudgeController::default();
        let mut now = idle_now(5_000);
        now.body.battery_level = 0.05;
        prime_idle(&mut controller, &now, policy);

        assert!(controller.propose(&now, policy).is_none());
        assert_eq!(
            controller.status.nudge_blocked_reason.as_deref(),
            Some("battery is critical")
        );
    }

    #[test]
    fn nudge_avoids_forward_when_obstacle_too_close() {
        let policy = NudgePolicy::virtual_default();
        let mut now = idle_now(5_000);
        now.range.nearest_m = Some(0.2);
        let action = ActionPrimitive::Go {
            intensity: 0.12,
            duration_ms: 500,
        };

        assert!(nudge_action_block_reason(&now, &action, policy)
            .unwrap()
            .contains("clearance"));
    }

    #[test]
    fn turn_nudge_allowed_when_forward_path_blocked() {
        let policy = NudgePolicy::virtual_default();
        let mut controller = NudgeController::default();
        let mut now = idle_now(5_000);
        now.range.nearest_m = Some(0.2);
        now.range.beams = vec![0.2, 0.2, 0.8];
        prime_idle(&mut controller, &now, policy);

        let action = controller.propose(&now, policy).unwrap();
        assert!(matches!(
            action,
            ActionPrimitive::Turn {
                direction: TurnDir::Right,
                ..
            }
        ));
    }

    #[test]
    fn nudge_cooldown_prevents_repeated_twitching() {
        let policy = NudgePolicy::virtual_default();
        let mut controller = NudgeController::default();
        let now = idle_now(5_000);
        prime_idle(&mut controller, &now, policy);
        assert!(controller.propose(&now, policy).is_some());

        let later = idle_now(6_000);
        assert!(controller.propose(&later, policy).is_none());
        assert_eq!(
            controller.status.nudge_blocked_reason.as_deref(),
            Some("prod cooldown active")
        );
    }

    #[test]
    fn default_candidates_include_novelty_inspection() {
        assert!(default_candidate_actions().iter().any(|action| {
            matches!(
                action,
                ActionPrimitive::Inspect {
                    target: InspectTarget::Novelty
                }
            )
        }));
    }

    #[test]
    fn curiosity_scores_inspection_above_stopping() {
        let mut now = idle_now(1_000);
        now.drives.curiosity = 0.8;
        now.memory.place_novelty = 0.7;

        let stop = score_action_candidate(
            &now,
            &ActionPrimitive::Stop,
            CandidateModelSignals::default(),
            None,
        );
        let inspect = score_action_candidate(
            &now,
            &ActionPrimitive::Inspect {
                target: InspectTarget::Novelty,
            },
            CandidateModelSignals::default(),
            None,
        );

        assert!(inspect.score > stop.score);
        assert!(inspect.curiosity > 0.0);
    }

    struct StubRuntime;

    #[async_trait::async_trait]
    impl RuntimeLoop for StubRuntime {
        async fn tick(
            &mut self,
            now: Now,
            _latent: ExperienceLatent,
            _futures: Vec<FuturePrediction>,
        ) -> Result<RuntimeTick> {
            let reign_input = now.reign.latest.clone();
            let action = ActionPrimitive::Go {
                intensity: 0.2,
                duration_ms: 100,
            };
            let experience =
                Experience::new("test", "test", Vec::new(), Vec::new(), now.t_ms, now.t_ms);
            Ok(RuntimeTick {
                frame: ExperienceFrame {
                    id: Uuid::new_v4(),
                    t_ms: now.t_ms,
                    now,
                    sensations: Vec::new(),
                    impressions: Vec::new(),
                    experiences: vec![experience.clone()],
                    z: Some(ExperienceLatent::default()),
                    chosen_action: Some(action.clone()),
                    conscious_command: None,
                    reign_input,
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
                    notes: Vec::new(),
                },
                experience,
                chosen_action: Some(action),
                recall: RecallBundle::default(),
                llm: LlmTickResult::default(),
                combobulation: None,
                inline_learning: InlineLearningTickStatus::default(),
            })
        }
    }

    struct SlowRuntime {
        tick_attempts: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl RuntimeLoop for SlowRuntime {
        async fn tick(
            &mut self,
            now: Now,
            _latent: ExperienceLatent,
            _futures: Vec<FuturePrediction>,
        ) -> Result<RuntimeTick> {
            self.tick_attempts.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            let experience =
                Experience::new("test", "test", Vec::new(), Vec::new(), now.t_ms, now.t_ms);
            Ok(RuntimeTick {
                frame: ExperienceFrame {
                    id: Uuid::new_v4(),
                    t_ms: now.t_ms,
                    now,
                    sensations: Vec::new(),
                    impressions: Vec::new(),
                    experiences: vec![experience.clone()],
                    z: Some(ExperienceLatent::default()),
                    chosen_action: Some(ActionPrimitive::Stop),
                    conscious_command: None,
                    reign_input: None,
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
                    notes: Vec::new(),
                },
                experience,
                chosen_action: Some(ActionPrimitive::Stop),
                recall: RecallBundle::default(),
                llm: LlmTickResult::default(),
                combobulation: None,
                inline_learning: InlineLearningTickStatus::default(),
            })
        }
    }

    #[derive(Clone)]
    struct SharedSimCockpit(Arc<Mutex<SimCockpit>>);

    impl Cockpit for SharedSimCockpit {
        fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
            self.0.lock().unwrap().execute(request)
        }

        fn handshake(
            &mut self,
            hello: pete_cockpit::HandshakeHello,
        ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
            self.0.lock().unwrap().handshake(hello)
        }

        fn execute_in_session(
            &mut self,
            session: &pete_cockpit::CockpitSession,
            request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            self.0.lock().unwrap().execute_in_session(session, request)
        }

        fn execute_with_lease(
            &mut self,
            session: &pete_cockpit::CockpitSession,
            lease: &pete_cockpit::ControlLease,
            request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            self.0
                .lock()
                .unwrap()
                .execute_with_lease(session, lease, request)
        }

        fn execute_with_service_lease(
            &mut self,
            session: &pete_cockpit::CockpitSession,
            lease: &pete_cockpit::ServiceLease,
            request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            self.0
                .lock()
                .unwrap()
                .execute_with_service_lease(session, lease, request)
        }
    }

    #[test]
    fn production_possession_composes_bump_stop_and_conductor_recovery() {
        let sim = Arc::new(Mutex::new(SimCockpit::new().with_event_capacity(256)));
        let session = establish_session(
            SharedSimCockpit(Arc::clone(&sim)),
            HandshakeHello::motherbrain("pete-runtime-recovery-test"),
            None,
        )
        .unwrap();
        let possession = MotherbrainPossession::acquire(session, 5_000).unwrap();
        let mut cockpit = SafeCockpit::new(possession);

        cockpit.pulse_motion(40, 0).unwrap();
        sim.lock().unwrap().set_bump(true, false);

        let stop_events = cockpit.poll_events().unwrap();
        assert!(stop_events.has_stop_reason());
        let status = cockpit.refresh_status().unwrap();
        let body = body_sense_from_cockpit_status(status, 1_000);
        assert!(body.flags.bump_left);

        let mut conductor = SimpleConductor::default();
        let mut input = test_conductor_input(ActionPrimitive::Stop);
        input.body = body;
        let first_recovery = conductor.choose(input).unwrap();
        assert!(matches!(
            first_recovery,
            ActionPrimitive::Go {
                intensity,
                duration_ms: 300
            } if intensity < 0.0
        ));

        sim.lock().unwrap().set_bump(false, false);
        cockpit
            .client_mut()
            .clear_safety_latch(SafetyLatchKind::Bump)
            .unwrap();
        let clear_events = cockpit.poll_events().unwrap();
        assert!(clear_events
            .events
            .iter()
            .any(|event| event.kind == pete_cockpit::CockpitEventKind::SafetyCleared));
        let cleared_body = body_sense_from_cockpit_status(cockpit.refresh_status().unwrap(), 1_100);
        assert!(!cleared_body.flags.bump_left);

        let mut cleared_input = test_conductor_input(ActionPrimitive::Stop);
        cleared_input.body = cleared_body.clone();
        let reverse = conductor.choose(cleared_input.clone()).unwrap();
        let reverse_motor = action_to_motor_command(Some(&reverse));
        assert!(reverse_motor.forward < 0.0);
        apply_slow_possession_motor(&mut cockpit, reverse_motor).unwrap();

        let turn = conductor.choose(cleared_input).unwrap();
        assert!(matches!(
            turn,
            ActionPrimitive::Turn {
                direction: TurnDir::Right,
                ..
            }
        ));
        let turn_motor = action_to_motor_command(Some(&turn));
        assert!(turn_motor.turn.abs() > 0.0);
        apply_slow_possession_motor(&mut cockpit, turn_motor).unwrap();

        assert!(cockpit.client_mut().snapshot().possessed);
        let events = sim.lock().unwrap().get_events_since(0).unwrap();
        let stop_index = events
            .events
            .iter()
            .position(|event| event.kind == pete_cockpit::CockpitEventKind::MotionStopped)
            .unwrap();
        assert!(events.events[stop_index + 1..]
            .iter()
            .any(|event| event.kind == pete_cockpit::CockpitEventKind::MotionRequested));
    }

    #[tokio::test]
    async fn normal_possession_run_random_walk_bump_stop_and_recovery() {
        let sim = Arc::new(Mutex::new(SimCockpit::new().with_event_capacity(256)));
        let session = establish_session(
            SharedSimCockpit(Arc::clone(&sim)),
            HandshakeHello::motherbrain("pete-runtime-normal-bump-test"),
            None,
        )
        .unwrap();
        let possession = MotherbrainPossession::acquire(session, 5_000).unwrap();
        let ledger_root = test_ledger_root("normal-possession-bump-recovery");
        let ledger = JsonlLedger::new(&ledger_root);
        let memory = InMemoryExperienceStore::new();
        let runtime = MinimalRuntime::new(
            ledger,
            memory.clone(),
            memory,
            SimpleConductor::default(),
            SimpleSafety::default(),
            pete_llm::NoopLlmAgent,
        );
        let mut runner = RealRobotRunner::new(RobotMode::Slow, possession, Vec::new(), runtime)
            .with_autonomous_motion(true);
        runner.cockpit.resync_event_cursor_from_status().unwrap();

        let (_first_snapshot, first_tick) = runner.tick_slow_manual().await.unwrap();
        assert!(matches!(
            first_tick.chosen_action,
            Some(ActionPrimitive::Drive { forward, turn, .. })
                if (forward - 0.2).abs() < 0.001 && (turn - 0.1).abs() < 0.001
        ));

        sim.lock().unwrap().set_bump(true, false);
        let (_bump_snapshot, bump_tick) = runner.tick_slow_manual().await.unwrap();
        assert!(bump_tick.frame.now.body.flags.bump_left);
        assert!(matches!(
            bump_tick.chosen_action,
            Some(ActionPrimitive::Go { intensity, .. }) if intensity < 0.0
        ));

        sim.lock().unwrap().set_bump(false, false);
        runner.possession_recovery.last_command_ms =
            wall_time_ms().saturating_sub(BUMP_ESCAPE_BACKOFF_DURATION_MS as TimeMs + 1);
        let (_turn_snapshot, turn_tick) = runner.tick_slow_manual().await.unwrap();
        assert!(matches!(
            turn_tick.chosen_action,
            Some(ActionPrimitive::Turn {
                direction: TurnDir::Right,
                ..
            })
        ));

        let events = sim.lock().unwrap().get_events_since(0).unwrap();
        let bump_index = events
            .events
            .iter()
            .position(|event| event.kind == pete_cockpit::CockpitEventKind::BumpChanged)
            .unwrap();
        let stop_index = events.events[bump_index..]
            .iter()
            .position(|event| event.kind == pete_cockpit::CockpitEventKind::MotionStopped)
            .map(|index| bump_index + index)
            .unwrap();
        let safety_index = events.events[bump_index..]
            .iter()
            .position(|event| event.kind == pete_cockpit::CockpitEventKind::SafetyTripped)
            .map(|index| bump_index + index)
            .unwrap();
        assert!(bump_index < safety_index && safety_index < stop_index);
        assert!(runner.cockpit.client_mut().snapshot().possessed);
        let _ = fs::remove_dir_all(ledger_root);
    }

    struct CountingCockpit {
        motor_attempts: Arc<AtomicUsize>,
        motors: Arc<Mutex<Vec<MotorCommand>>>,
        body: BodySense,
    }

    impl Cockpit for CountingCockpit {
        fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
            match request {
                CockpitRequest::GetStatus => Ok(CockpitResponse::Status(self.get_status()?)),
                CockpitRequest::GetCapabilities => {
                    Ok(CockpitResponse::Capabilities(self.get_capabilities()?))
                }
                CockpitRequest::GetEvents { since_seq } => {
                    Ok(CockpitResponse::Events(self.get_events_since(since_seq)?))
                }
                CockpitRequest::Stop => {
                    self.motor_attempts.fetch_add(1, Ordering::SeqCst);
                    self.motors.lock().unwrap().push(MotorCommand::stop());
                    Ok(CockpitResponse::Accepted)
                }
                CockpitRequest::CmdVel {
                    linear_mm_s,
                    angular_mrad_s,
                    ..
                } => {
                    self.motor_attempts.fetch_add(1, Ordering::SeqCst);
                    self.motors.lock().unwrap().push(MotorCommand {
                        forward: linear_mm_s as f32 / 1000.0,
                        turn: angular_mrad_s as f32 / 1000.0,
                    });
                    Ok(CockpitResponse::Accepted)
                }
                CockpitRequest::HeartbeatStop { .. } => Ok(CockpitResponse::Accepted),
                _ => Ok(CockpitResponse::Accepted),
            }
        }

        fn handshake(
            &mut self,
            _hello: pete_cockpit::HandshakeHello,
        ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
            Err(pete_cockpit::CockpitError::Policy(
                "test cockpit has no handshake peer".into(),
            ))
        }

        fn execute_in_session(
            &mut self,
            _session: &pete_cockpit::CockpitSession,
            request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            self.execute(request)
        }

        fn execute_with_lease(
            &mut self,
            _session: &pete_cockpit::CockpitSession,
            _lease: &pete_cockpit::ControlLease,
            request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            self.execute(request)
        }

        fn execute_with_service_lease(
            &mut self,
            _session: &pete_cockpit::CockpitSession,
            _lease: &pete_cockpit::ServiceLease,
            _request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            Err(pete_cockpit::CockpitError::Policy(
                "test cockpit has no service mode".into(),
            ))
        }

        fn get_status(&mut self) -> pete_cockpit::Result<CockpitStatus> {
            Ok(CockpitStatus {
                raw: serde_json::json!({
                    "uptime_ms": 1_000,
                    "current_runtime_state": "test",
                    "oi_mode": "safe",
                    "current_command": "stop",
                    "create_sensors": {
                        "last_packet_id": 0,
                        "complete_packet_count": 1,
                        "last_complete_packet_timestamp_ms": 1_000,
                        "bump_left": self.body.flags.bump_left,
                        "bump_right": self.body.flags.bump_right,
                        "wheel_drop": self.body.flags.wheel_drop,
                        "wall": self.body.flags.wall,
                        "virtual_wall": self.body.flags.virtual_wall,
                        "cliff_left": self.body.flags.cliff_left,
                        "cliff_front_left": self.body.flags.cliff_front_left,
                        "cliff_front_right": self.body.flags.cliff_front_right,
                        "cliff_right": self.body.flags.cliff_right,
                        "charge_mah": (self.body.battery_level.clamp(0.0, 1.0) * 2600.0).round() as u32,
                        "capacity_mah": 2600,
                        "charging_state": if self.body.charging { 1 } else { 0 },
                    },
                    "odometry": {
                        "distance_mm": (self.body.odometry.x_m * 1000.0).round() as i32,
                        "heading_mrad": (self.body.odometry.heading_rad * 1000.0).round() as i32,
                        "reset_count": 0,
                    }
                })
                .to_string(),
            })
        }

        fn get_capabilities(&mut self) -> pete_cockpit::Result<CockpitCapabilities> {
            Ok(CockpitCapabilities {
                body_kind: "test".to_string(),
                drive: "differential".to_string(),
                verbs: [
                    "status",
                    "get_capabilities",
                    "get_events",
                    "stop",
                    "cmd_vel",
                    "heartbeat_stop",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
                sensors: Vec::new(),
                outputs: Vec::new(),
                safety: Vec::new(),
                events: Vec::new(),
                limits: pete_cockpit::CockpitLimits {
                    max_linear_mm_s: 500,
                    max_angular_mrad_s: 4_000,
                    min_ttl_ms: 1,
                    max_ttl_ms: 60_000,
                },
            })
        }

        fn get_events_since(&mut self, since_seq: u32) -> pete_cockpit::Result<EventBatch> {
            Ok(EventBatch {
                since_seq,
                oldest_seq: 1,
                next_seq: since_seq.saturating_add(1),
                dropped_before_seq: 0,
                events: Vec::new(),
            })
        }
    }

    struct LatchedStatusCockpit {
        clear_attempts: Arc<Mutex<Vec<SafetyLatchKind>>>,
        latch: SafetyLatchKind,
        safety_tripped: bool,
    }

    impl Cockpit for LatchedStatusCockpit {
        fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
            match request {
                CockpitRequest::GetStatus => Ok(CockpitResponse::Status(self.get_status()?)),
                CockpitRequest::GetCapabilities => {
                    Ok(CockpitResponse::Capabilities(self.get_capabilities()?))
                }
                CockpitRequest::GetEvents { since_seq } => {
                    Ok(CockpitResponse::Events(self.get_events_since(since_seq)?))
                }
                CockpitRequest::ClearSafetyLatch { latch } => {
                    self.clear_attempts.lock().unwrap().push(latch);
                    self.safety_tripped = false;
                    Ok(CockpitResponse::Accepted)
                }
                CockpitRequest::Stop | CockpitRequest::HeartbeatStop { .. } => {
                    Ok(CockpitResponse::Accepted)
                }
                _ => Ok(CockpitResponse::Accepted),
            }
        }

        fn handshake(
            &mut self,
            _hello: pete_cockpit::HandshakeHello,
        ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
            Err(pete_cockpit::CockpitError::Policy(
                "test cockpit has no handshake peer".into(),
            ))
        }

        fn execute_in_session(
            &mut self,
            _session: &pete_cockpit::CockpitSession,
            request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            self.execute(request)
        }

        fn execute_with_lease(
            &mut self,
            _session: &pete_cockpit::CockpitSession,
            _lease: &pete_cockpit::ControlLease,
            request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            self.execute(request)
        }

        fn execute_with_service_lease(
            &mut self,
            _session: &pete_cockpit::CockpitSession,
            _lease: &pete_cockpit::ServiceLease,
            _request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            Err(pete_cockpit::CockpitError::Policy(
                "test cockpit has no service mode".into(),
            ))
        }

        fn get_status(&mut self) -> pete_cockpit::Result<CockpitStatus> {
            Ok(CockpitStatus {
                raw: serde_json::json!({
                    "uptime_ms": 1_000,
                    "current_runtime_state": "idle",
                    "oi_mode": "safe",
                    "current_command": "stop",
                    "estop_latched": false,
                    "safety_tripped": self.safety_tripped,
                    "safety_latch_kind": self.latch,
                    "create_sensors": {
                        "last_packet_id": 0,
                        "complete_packet_count": 1,
                        "last_complete_packet_timestamp_ms": 1_000,
                        "charging_state": 0,
                    },
                    "imu": {
                        "health": "ok",
                        "tilt_magnitude_mrad": 0,
                        "impact_score_mm_s2": 0,
                    }
                })
                .to_string(),
            })
        }

        fn get_capabilities(&mut self) -> pete_cockpit::Result<CockpitCapabilities> {
            Ok(CockpitCapabilities {
                body_kind: "test".to_string(),
                drive: "differential".to_string(),
                verbs: [
                    "status",
                    "get_capabilities",
                    "get_events",
                    "stop",
                    "cmd_vel",
                    "heartbeat_stop",
                    "clear_safety_latch",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
                sensors: Vec::new(),
                outputs: Vec::new(),
                safety: Vec::new(),
                events: Vec::new(),
                limits: pete_cockpit::CockpitLimits {
                    max_linear_mm_s: 500,
                    max_angular_mrad_s: 4_000,
                    min_ttl_ms: 1,
                    max_ttl_ms: 60_000,
                },
            })
        }

        fn get_events_since(&mut self, since_seq: u32) -> pete_cockpit::Result<EventBatch> {
            Ok(EventBatch {
                since_seq,
                oldest_seq: 1,
                next_seq: since_seq.saturating_add(1),
                dropped_before_seq: 0,
                events: Vec::new(),
            })
        }
    }

    struct ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc<AtomicUsize>,
        bump_escape_commands: Arc<Mutex<Vec<(EscapeDirection, i16, i16)>>>,
        stop_attempts: Arc<AtomicUsize>,
        clear_attempts: Arc<AtomicUsize>,
        bump_active: bool,
    }

    impl Cockpit for ActiveBumpRecoveryCockpit {
        fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
            match request {
                CockpitRequest::GetStatus => Ok(CockpitResponse::Status(self.get_status()?)),
                CockpitRequest::GetCapabilities => {
                    Ok(CockpitResponse::Capabilities(self.get_capabilities()?))
                }
                CockpitRequest::GetEvents { since_seq } => {
                    Ok(CockpitResponse::Events(self.get_events_since(since_seq)?))
                }
                CockpitRequest::BumpEscape {
                    direction,
                    backoff_mm_s,
                    turn_angular_mrad_s,
                } => {
                    self.bump_escape_attempts.fetch_add(1, Ordering::SeqCst);
                    self.bump_escape_commands.lock().unwrap().push((
                        direction,
                        backoff_mm_s,
                        turn_angular_mrad_s,
                    ));
                    Ok(CockpitResponse::Accepted)
                }
                CockpitRequest::Stop => {
                    self.stop_attempts.fetch_add(1, Ordering::SeqCst);
                    Ok(CockpitResponse::Accepted)
                }
                CockpitRequest::ClearSafetyLatch {
                    latch: SafetyLatchKind::Bump,
                } => {
                    self.clear_attempts.fetch_add(1, Ordering::SeqCst);
                    Ok(CockpitResponse::Accepted)
                }
                CockpitRequest::HeartbeatStop { .. } => Ok(CockpitResponse::Accepted),
                _ => Ok(CockpitResponse::Accepted),
            }
        }

        fn handshake(
            &mut self,
            _hello: pete_cockpit::HandshakeHello,
        ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
            Err(pete_cockpit::CockpitError::Policy(
                "test cockpit has no handshake peer".into(),
            ))
        }

        fn execute_in_session(
            &mut self,
            _session: &pete_cockpit::CockpitSession,
            request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            self.execute(request)
        }

        fn execute_with_lease(
            &mut self,
            _session: &pete_cockpit::CockpitSession,
            _lease: &pete_cockpit::ControlLease,
            request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            self.execute(request)
        }

        fn execute_with_service_lease(
            &mut self,
            _session: &pete_cockpit::CockpitSession,
            _lease: &pete_cockpit::ServiceLease,
            _request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            Err(pete_cockpit::CockpitError::Policy(
                "test cockpit has no service mode".into(),
            ))
        }

        fn get_status(&mut self) -> pete_cockpit::Result<CockpitStatus> {
            Ok(CockpitStatus {
                raw: serde_json::json!({
                    "uptime_ms": 1_000,
                    "current_runtime_state": "idle",
                    "oi_mode": "safe",
                    "current_command": "stop",
                    "estop_latched": false,
                    "safety_tripped": true,
                    "safety_latch_kind": "bump",
                    "create_sensors": {
                        "last_packet_id": 0,
                        "complete_packet_count": 1,
                        "last_complete_packet_timestamp_ms": 1_000,
                        "bump_left": self.bump_active,
                        "bump_right": false,
                        "charging_state": 0,
                    },
                    "imu": {
                        "health": "ok",
                        "tilt_magnitude_mrad": 0,
                        "impact_score_mm_s2": 0,
                    }
                })
                .to_string(),
            })
        }

        fn get_capabilities(&mut self) -> pete_cockpit::Result<CockpitCapabilities> {
            Ok(CockpitCapabilities {
                body_kind: "test".to_string(),
                drive: "differential".to_string(),
                verbs: [
                    "status",
                    "get_capabilities",
                    "get_events",
                    "stop",
                    "bump_escape",
                    "heartbeat_stop",
                ]
                .into_iter()
                .map(str::to_string)
                .collect(),
                sensors: Vec::new(),
                outputs: Vec::new(),
                safety: Vec::new(),
                events: Vec::new(),
                limits: pete_cockpit::CockpitLimits {
                    max_linear_mm_s: 500,
                    max_angular_mrad_s: 4_000,
                    min_ttl_ms: 1,
                    max_ttl_ms: 60_000,
                },
            })
        }

        fn get_events_since(&mut self, since_seq: u32) -> pete_cockpit::Result<EventBatch> {
            Ok(EventBatch {
                since_seq,
                oldest_seq: 1,
                next_seq: since_seq.saturating_add(1),
                dropped_before_seq: 0,
                events: Vec::new(),
            })
        }
    }

    struct HistoryGapCockpit {
        inner: CountingCockpit,
        event_polls: Arc<AtomicUsize>,
        gap_poll: usize,
    }

    impl Cockpit for HistoryGapCockpit {
        fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
            self.inner.execute(request)
        }

        fn handshake(
            &mut self,
            hello: pete_cockpit::HandshakeHello,
        ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
            self.inner.handshake(hello)
        }

        fn execute_in_session(
            &mut self,
            session: &pete_cockpit::CockpitSession,
            request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            self.inner.execute_in_session(session, request)
        }

        fn execute_with_lease(
            &mut self,
            session: &pete_cockpit::CockpitSession,
            lease: &pete_cockpit::ControlLease,
            request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            self.inner.execute_with_lease(session, lease, request)
        }

        fn execute_with_service_lease(
            &mut self,
            session: &pete_cockpit::CockpitSession,
            lease: &pete_cockpit::ServiceLease,
            request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            self.inner
                .execute_with_service_lease(session, lease, request)
        }

        fn get_status(&mut self) -> pete_cockpit::Result<CockpitStatus> {
            self.inner.get_status()
        }

        fn get_capabilities(&mut self) -> pete_cockpit::Result<CockpitCapabilities> {
            self.inner.get_capabilities()
        }

        fn get_events_since(&mut self, since_seq: u32) -> pete_cockpit::Result<EventBatch> {
            let poll = self.event_polls.fetch_add(1, Ordering::SeqCst);
            let inject_gap = poll == self.gap_poll;
            Ok(EventBatch {
                since_seq,
                oldest_seq: if inject_gap {
                    since_seq.saturating_add(2)
                } else {
                    1
                },
                next_seq: since_seq.saturating_add(2),
                dropped_before_seq: if inject_gap {
                    since_seq.saturating_add(2)
                } else {
                    0
                },
                events: Vec::new(),
            })
        }
    }

    struct MotionStopEventsCockpit {
        inner: CountingCockpit,
        event_polls: Arc<AtomicUsize>,
    }

    impl Cockpit for MotionStopEventsCockpit {
        fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
            self.inner.execute(request)
        }

        fn handshake(
            &mut self,
            hello: pete_cockpit::HandshakeHello,
        ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
            self.inner.handshake(hello)
        }

        fn execute_in_session(
            &mut self,
            session: &pete_cockpit::CockpitSession,
            request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            self.inner.execute_in_session(session, request)
        }

        fn execute_with_lease(
            &mut self,
            session: &pete_cockpit::CockpitSession,
            lease: &pete_cockpit::ControlLease,
            request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            self.inner.execute_with_lease(session, lease, request)
        }

        fn execute_with_service_lease(
            &mut self,
            session: &pete_cockpit::CockpitSession,
            lease: &pete_cockpit::ServiceLease,
            request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            self.inner
                .execute_with_service_lease(session, lease, request)
        }

        fn get_status(&mut self) -> pete_cockpit::Result<CockpitStatus> {
            self.inner.get_status()
        }

        fn get_capabilities(&mut self) -> pete_cockpit::Result<CockpitCapabilities> {
            self.inner.get_capabilities()
        }

        fn get_events_since(&mut self, since_seq: u32) -> pete_cockpit::Result<EventBatch> {
            let poll = self.event_polls.fetch_add(1, Ordering::SeqCst);
            let events = if poll == 1 {
                vec![
                    pete_cockpit::CockpitEvent {
                        seq: since_seq.saturating_add(1),
                        kind: pete_cockpit::CockpitEventKind::HeartbeatExpired,
                        a: 0,
                        b: 0,
                        c: 0,
                    },
                    pete_cockpit::CockpitEvent {
                        seq: since_seq.saturating_add(2),
                        kind: pete_cockpit::CockpitEventKind::SafetyTripped,
                        a: 1,
                        b: 0,
                        c: 0,
                    },
                ]
            } else {
                Vec::new()
            };
            Ok(EventBatch {
                since_seq,
                oldest_seq: 1,
                next_seq: since_seq.saturating_add(events.len() as u32 + 1),
                dropped_before_seq: 0,
                events,
            })
        }
    }

    struct RejectingMotionCockpit {
        inner: CountingCockpit,
        rejection_attempts: Arc<AtomicUsize>,
    }

    impl Cockpit for RejectingMotionCockpit {
        fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
            if matches!(&request, CockpitRequest::CmdVel { .. }) {
                self.rejection_attempts.fetch_add(1, Ordering::SeqCst);
                return Err(pete_cockpit::CockpitError::Rejected {
                    command_id: 42,
                    reason: "stale_sequence".to_string(),
                });
            }
            self.inner.execute(request)
        }

        fn handshake(
            &mut self,
            hello: pete_cockpit::HandshakeHello,
        ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
            self.inner.handshake(hello)
        }

        fn execute_in_session(
            &mut self,
            session: &pete_cockpit::CockpitSession,
            request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            self.inner.execute_in_session(session, request)
        }

        fn execute_with_lease(
            &mut self,
            session: &pete_cockpit::CockpitSession,
            lease: &pete_cockpit::ControlLease,
            request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            self.inner.execute_with_lease(session, lease, request)
        }

        fn execute_with_service_lease(
            &mut self,
            session: &pete_cockpit::CockpitSession,
            lease: &pete_cockpit::ServiceLease,
            request: CockpitRequest,
        ) -> pete_cockpit::Result<CockpitResponse> {
            self.inner
                .execute_with_service_lease(session, lease, request)
        }

        fn get_status(&mut self) -> pete_cockpit::Result<CockpitStatus> {
            self.inner.get_status()
        }

        fn get_capabilities(&mut self) -> pete_cockpit::Result<CockpitCapabilities> {
            self.inner.get_capabilities()
        }

        fn get_events_since(&mut self, since_seq: u32) -> pete_cockpit::Result<EventBatch> {
            Ok(EventBatch {
                since_seq,
                oldest_seq: 1,
                next_seq: since_seq.saturating_add(1),
                dropped_before_seq: 0,
                events: Vec::new(),
            })
        }
    }

    struct FailingSensor;

    #[async_trait::async_trait]
    impl SenseProducer for FailingSensor {
        async fn poll(&mut self) -> Result<pete_sensors::SensePacket> {
            anyhow::bail!("simulated sensor timeout")
        }
    }

    #[tokio::test]
    async fn real_robot_read_only_runner_never_applies_motor() {
        let motor_attempts = Arc::new(AtomicUsize::new(0));
        let motors = Arc::new(Mutex::new(Vec::new()));
        let body = CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        };
        let mut runner =
            RealRobotRunner::new(RobotMode::ReadOnly, Box::new(body), Vec::new(), StubRuntime);

        let (_snapshot, tick) = runner.tick_read_only().await.unwrap();

        assert!(matches!(
            tick.chosen_action,
            Some(ActionPrimitive::Go { .. })
        ));
        assert_eq!(motor_attempts.load(Ordering::SeqCst), 0);
        assert!(motors.lock().unwrap().is_empty());
        assert_eq!(
            tick.frame
                .now
                .extensions
                .get("safety/read_only_veto")
                .and_then(|value| value.as_bool()),
            Some(true)
        );
    }

    #[tokio::test]
    async fn real_robot_read_only_runner_publishes_snapshot_when_optional_sensor_fails() {
        let body = CountingCockpit {
            motor_attempts: Arc::new(AtomicUsize::new(0)),
            motors: Arc::new(Mutex::new(Vec::new())),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        };
        let sensors: Vec<Box<dyn SenseProducer + Send>> = vec![Box::new(FailingSensor)];
        let mut runner =
            RealRobotRunner::new(RobotMode::ReadOnly, Box::new(body), sensors, StubRuntime);

        let (snapshot, _tick) = runner.tick_read_only().await.unwrap();

        assert!(snapshot.body.last_update_ms >= 100);
        assert_eq!(runner.tick_count, 1);
    }

    #[tokio::test]
    async fn real_robot_slow_runner_without_webremote_direct_sends_stop() {
        let motor_attempts = Arc::new(AtomicUsize::new(0));
        let motors = Arc::new(Mutex::new(Vec::new()));
        let body = CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        };
        let mut runner =
            RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), StubRuntime);

        let (_snapshot, _tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(motor_attempts.load(Ordering::SeqCst), 1);
        assert_eq!(motors.lock().unwrap().as_slice(), &[MotorCommand::stop()]);
    }

    #[tokio::test]
    async fn real_robot_slow_runner_clears_latch_reported_by_status() {
        let clear_attempts = Arc::new(Mutex::new(Vec::new()));
        let body = LatchedStatusCockpit {
            clear_attempts: Arc::clone(&clear_attempts),
            latch: SafetyLatchKind::Tilt,
            safety_tripped: true,
        };
        let mut runner =
            RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), StubRuntime)
                .with_autonomous_motion(true);

        let (snapshot, _tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(
            clear_attempts.lock().unwrap().as_slice(),
            &[SafetyLatchKind::Tilt]
        );
        assert_eq!(
            snapshot
                .action_debug
                .as_ref()
                .and_then(|debug| debug.get("possession_recovery"))
                .and_then(|debug| debug.get("latched")),
            Some(&serde_json::json!("Tilt"))
        );
    }

    #[tokio::test]
    async fn real_robot_slow_runner_reports_active_bump_recovery_as_chosen_action() {
        let bump_escape_attempts = Arc::new(AtomicUsize::new(0));
        let bump_escape_commands = Arc::new(Mutex::new(Vec::new()));
        let stop_attempts = Arc::new(AtomicUsize::new(0));
        let body = ActiveBumpRecoveryCockpit {
            bump_escape_attempts: Arc::clone(&bump_escape_attempts),
            bump_escape_commands: Arc::clone(&bump_escape_commands),
            stop_attempts,
            clear_attempts: Arc::new(AtomicUsize::new(0)),
            bump_active: true,
        };
        let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
            .with_autonomous_motion(true);

        let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(bump_escape_attempts.load(Ordering::SeqCst), 1);
        assert_eq!(
            bump_escape_commands.lock().unwrap().as_slice(),
            &[(EscapeDirection::Right, 50, 500)]
        );
        assert_eq!(
            tick.chosen_action,
            Some(ActionPrimitive::Go {
                intensity: -0.2,
                duration_ms: BUMP_ESCAPE_BACKOFF_DURATION_MS as TimeMs,
            })
        );
        let debug = snapshot.action_debug.as_ref().unwrap();
        assert_eq!(
            debug.get("runtime_chosen_action"),
            Some(
                &serde_json::to_value(ActionPrimitive::Go {
                    intensity: 0.2,
                    duration_ms: 100,
                })
                .unwrap()
            )
        );
        assert_eq!(
            debug.get("motion_sent_to_robot"),
            Some(
                &serde_json::to_value(motor_command_to_motion(MotorCommand {
                    forward: -0.05,
                    turn: 0.0,
                }))
                .unwrap()
            )
        );
        assert_eq!(debug.get("motor_applied"), Some(&serde_json::json!(true)));
        assert_eq!(
            debug
                .get("possession_recovery")
                .and_then(|debug| debug.get("latched")),
            Some(&serde_json::json!("Bump"))
        );
    }

    #[tokio::test]
    async fn real_robot_slow_runner_does_not_spam_bump_escape_during_recovery_cooldown() {
        let bump_escape_attempts = Arc::new(AtomicUsize::new(0));
        let bump_escape_commands = Arc::new(Mutex::new(Vec::new()));
        let stop_attempts = Arc::new(AtomicUsize::new(0));
        let body = ActiveBumpRecoveryCockpit {
            bump_escape_attempts: Arc::clone(&bump_escape_attempts),
            bump_escape_commands,
            stop_attempts,
            clear_attempts: Arc::new(AtomicUsize::new(0)),
            bump_active: true,
        };
        let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
            .with_autonomous_motion(true);

        let (_first_snapshot, _first_tick) = runner.tick_slow_manual().await.unwrap();
        let (second_snapshot, second_tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(bump_escape_attempts.load(Ordering::SeqCst), 1);
        assert_eq!(
            second_tick.chosen_action,
            Some(ActionPrimitive::Go {
                intensity: -0.2,
                duration_ms: BUMP_ESCAPE_BACKOFF_DURATION_MS as TimeMs,
            })
        );
        assert!(second_snapshot
            .action_debug
            .as_ref()
            .and_then(|debug| debug.get("why_not_moving"))
            .and_then(|reason| reason.as_str())
            .is_some_and(|reason| reason.contains("reversing")));
    }

    #[tokio::test]
    async fn real_robot_slow_runner_reports_stuck_bump_recovery_after_repeated_attempts() {
        let bump_escape_attempts = Arc::new(AtomicUsize::new(0));
        let bump_escape_commands = Arc::new(Mutex::new(Vec::new()));
        let stop_attempts = Arc::new(AtomicUsize::new(0));
        let body = ActiveBumpRecoveryCockpit {
            bump_escape_attempts: Arc::clone(&bump_escape_attempts),
            bump_escape_commands,
            stop_attempts: Arc::clone(&stop_attempts),
            clear_attempts: Arc::new(AtomicUsize::new(0)),
            bump_active: true,
        };
        let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
            .with_autonomous_motion(true);
        runner.possession_recovery.latch = Some(SafetyLatchKind::Bump);
        runner.possession_recovery.phase = PossessionRecoveryPhase::WaitingForSensorClear;
        runner.possession_recovery.active_since_ms =
            wall_time_ms().saturating_sub(POSSESSION_RECOVERY_STUCK_AFTER_MS + 1);
        runner.possession_recovery.command_attempts = POSSESSION_RECOVERY_MAX_ATTEMPTS;

        let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(bump_escape_attempts.load(Ordering::SeqCst), 0);
        assert_eq!(stop_attempts.load(Ordering::SeqCst), 1);
        assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
        let debug = snapshot.action_debug.as_ref().unwrap();
        assert!(debug
            .get("why_not_moving")
            .and_then(|reason| reason.as_str())
            .is_some_and(|reason| reason.contains("operator intervention needed")));
        assert_eq!(
            debug
                .get("possession_recovery")
                .and_then(|debug| debug.get("phase")),
            Some(&serde_json::json!("Stuck"))
        );
    }

    #[tokio::test]
    async fn real_robot_slow_runner_escapes_even_if_momentary_bump_already_cleared() {
        let bump_escape_attempts = Arc::new(AtomicUsize::new(0));
        let bump_escape_commands = Arc::new(Mutex::new(Vec::new()));
        let body = ActiveBumpRecoveryCockpit {
            bump_escape_attempts: Arc::clone(&bump_escape_attempts),
            bump_escape_commands: Arc::clone(&bump_escape_commands),
            stop_attempts: Arc::new(AtomicUsize::new(0)),
            clear_attempts: Arc::new(AtomicUsize::new(0)),
            bump_active: false,
        };
        let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
            .with_autonomous_motion(true);

        let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(bump_escape_attempts.load(Ordering::SeqCst), 1);
        assert_eq!(
            bump_escape_commands.lock().unwrap().as_slice(),
            &[(EscapeDirection::Right, 50, 500)]
        );
        assert!(matches!(
            tick.chosen_action,
            Some(ActionPrimitive::Go { .. })
        ));
    }

    #[tokio::test]
    async fn real_robot_slow_runner_reports_clockwise_turn_phase_without_resubmitting() {
        let bump_escape_attempts = Arc::new(AtomicUsize::new(0));
        let body = ActiveBumpRecoveryCockpit {
            bump_escape_attempts: Arc::clone(&bump_escape_attempts),
            bump_escape_commands: Arc::new(Mutex::new(Vec::new())),
            stop_attempts: Arc::new(AtomicUsize::new(0)),
            clear_attempts: Arc::new(AtomicUsize::new(0)),
            bump_active: true,
        };
        let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
            .with_autonomous_motion(true);
        runner.possession_recovery.latch = Some(SafetyLatchKind::Bump);
        runner.possession_recovery.phase = PossessionRecoveryPhase::Escaping;
        runner.possession_recovery.command_attempts = 1;
        runner.possession_recovery.last_command_ms =
            wall_time_ms().saturating_sub(BUMP_ESCAPE_BACKOFF_DURATION_MS as TimeMs + 1);

        let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(bump_escape_attempts.load(Ordering::SeqCst), 0);
        assert!(matches!(
            tick.chosen_action,
            Some(ActionPrimitive::Turn {
                direction: TurnDir::Right,
                ..
            })
        ));
        assert!(snapshot
            .action_debug
            .as_ref()
            .and_then(|debug| debug.get("why_not_moving"))
            .and_then(|reason| reason.as_str())
            .is_some_and(|reason| reason.contains("clockwise 90 degrees")));
    }

    #[tokio::test]
    async fn real_robot_slow_runner_clears_bump_only_after_escape_finishes() {
        let stop_attempts = Arc::new(AtomicUsize::new(0));
        let clear_attempts = Arc::new(AtomicUsize::new(0));
        let body = ActiveBumpRecoveryCockpit {
            bump_escape_attempts: Arc::new(AtomicUsize::new(0)),
            bump_escape_commands: Arc::new(Mutex::new(Vec::new())),
            stop_attempts: Arc::clone(&stop_attempts),
            clear_attempts: Arc::clone(&clear_attempts),
            bump_active: false,
        };
        let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
            .with_autonomous_motion(true);
        runner.possession_recovery.latch = Some(SafetyLatchKind::Bump);
        runner.possession_recovery.phase = PossessionRecoveryPhase::Escaping;
        runner.possession_recovery.command_attempts = 1;
        runner.possession_recovery.last_command_ms =
            wall_time_ms().saturating_sub(POSSESSION_BUMP_ESCAPE_DURATION_MS + 1);

        let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(stop_attempts.load(Ordering::SeqCst), 1);
        assert_eq!(clear_attempts.load(Ordering::SeqCst), 1);
        assert_eq!(runner.possession_recovery.latch, None);
        assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
    }

    #[tokio::test]
    async fn real_robot_slow_runner_applies_executive_motion_when_explicitly_authorized() {
        let motor_attempts = Arc::new(AtomicUsize::new(0));
        let motors = Arc::new(Mutex::new(Vec::new()));
        let body = CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        };
        let mut runner =
            RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), StubRuntime)
                .with_brainstem_interface(serde_json::json!({
                    "verbs": ["status", "get_events", "cmd_vel"]
                }))
                .with_autonomous_motion(true);

        let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(motor_attempts.load(Ordering::SeqCst), 1);
        assert_eq!(
            motors.lock().unwrap().as_slice(),
            &[MotorCommand {
                forward: 0.05,
                turn: 0.0,
            }]
        );
        assert_eq!(
            snapshot
                .action_debug
                .as_ref()
                .and_then(|debug| debug.get("autonomous_hardware_gate"))
                .and_then(|value| value.as_bool()),
            Some(true)
        );
        assert_eq!(
            tick.frame
                .now
                .extensions
                .get("brainstem.events")
                .and_then(|extension| extension.get("events"))
                .and_then(|events| events.as_array())
                .map(Vec::len),
            Some(0)
        );
        assert_eq!(
            tick.frame.now.extensions["brainstem.interface"]["underlying_body_private"],
            serde_json::json!(true)
        );
    }

    #[tokio::test]
    async fn real_robot_slow_runner_waits_for_runtime_tick_without_backoff() {
        let motor_attempts = Arc::new(AtomicUsize::new(0));
        let motors = Arc::new(Mutex::new(Vec::new()));
        let body = CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        };
        let tick_attempts = Arc::new(AtomicUsize::new(0));
        let runtime = SlowRuntime {
            tick_attempts: Arc::clone(&tick_attempts),
        };
        let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), runtime);
        runner.tick_ms = 25;

        let (_first_snapshot, first_tick) = runner.tick_slow_manual().await.unwrap();
        let (_second_snapshot, second_tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(tick_attempts.load(Ordering::SeqCst), 2);
        assert_eq!(motor_attempts.load(Ordering::SeqCst), 2);
        assert_eq!(
            motors.lock().unwrap().as_slice(),
            &[MotorCommand::stop(), MotorCommand::stop()]
        );
        assert!(first_tick.frame.notes.is_empty());
        assert!(second_tick.frame.notes.is_empty());
    }

    #[tokio::test]
    async fn real_robot_slow_runner_recovers_history_gap_by_stopping() {
        let motor_attempts = Arc::new(AtomicUsize::new(0));
        let motors = Arc::new(Mutex::new(Vec::new()));
        let event_polls = Arc::new(AtomicUsize::new(0));
        let body = HistoryGapCockpit {
            inner: CountingCockpit {
                motor_attempts: Arc::clone(&motor_attempts),
                motors: Arc::clone(&motors),
                body: BodySense {
                    last_update_ms: 100,
                    ..BodySense::default()
                },
            },
            event_polls: Arc::clone(&event_polls),
            gap_poll: 1,
        };
        let runtime = SlowRuntime {
            tick_attempts: Arc::new(AtomicUsize::new(0)),
        };
        let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), runtime);

        runner.tick_slow_manual().await.unwrap();
        let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(event_polls.load(Ordering::SeqCst), 2);
        assert_eq!(
            tick.frame.now.extensions["brainstem.events"]["dropped_before_seq"],
            serde_json::json!(3)
        );
        assert!(motor_attempts.load(Ordering::SeqCst) >= 2);
        assert_eq!(motors.lock().unwrap().last(), Some(&MotorCommand::stop()));
    }

    #[tokio::test]
    async fn real_robot_slow_runner_recovers_motion_safety_poll_history_gap_by_stopping() {
        let motor_attempts = Arc::new(AtomicUsize::new(0));
        let motors = Arc::new(Mutex::new(Vec::new()));
        let event_polls = Arc::new(AtomicUsize::new(0));
        let body = HistoryGapCockpit {
            inner: CountingCockpit {
                motor_attempts: Arc::clone(&motor_attempts),
                motors: Arc::clone(&motors),
                body: BodySense {
                    last_update_ms: 100,
                    ..BodySense::default()
                },
            },
            event_polls: Arc::clone(&event_polls),
            gap_poll: 1,
        };
        let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
            .with_autonomous_motion(true);

        let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(
            tick.chosen_action,
            Some(ActionPrimitive::Go {
                intensity: 0.2,
                duration_ms: 100,
            })
        );
        assert_eq!(event_polls.load(Ordering::SeqCst), 2);
        assert!(motor_attempts.load(Ordering::SeqCst) >= 2);
        assert_eq!(motors.lock().unwrap().last(), Some(&MotorCommand::stop()));
    }

    #[tokio::test]
    async fn real_robot_slow_runner_recovers_motion_stop_events_by_stopping() {
        let motor_attempts = Arc::new(AtomicUsize::new(0));
        let motors = Arc::new(Mutex::new(Vec::new()));
        let event_polls = Arc::new(AtomicUsize::new(0));
        let body = MotionStopEventsCockpit {
            inner: CountingCockpit {
                motor_attempts: Arc::clone(&motor_attempts),
                motors: Arc::clone(&motors),
                body: BodySense {
                    last_update_ms: 100,
                    ..BodySense::default()
                },
            },
            event_polls: Arc::clone(&event_polls),
        };
        let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
            .with_autonomous_motion(true);

        let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(
            tick.chosen_action,
            Some(ActionPrimitive::Go {
                intensity: 0.2,
                duration_ms: 100,
            })
        );
        assert_eq!(event_polls.load(Ordering::SeqCst), 2);
        assert!(motor_attempts.load(Ordering::SeqCst) >= 2);
        assert_eq!(motors.lock().unwrap().last(), Some(&MotorCommand::stop()));

        let (recovery_snapshot, _recovery_tick) = runner.tick_slow_manual().await.unwrap();
        assert_eq!(
            recovery_snapshot
                .action_debug
                .as_ref()
                .and_then(|debug| debug.get("possession_recovery"))
                .and_then(|debug| debug.get("latched")),
            Some(&serde_json::json!("Bump"))
        );
    }

    #[tokio::test]
    async fn real_robot_slow_runner_treats_command_rejected_as_motion_feedback() {
        let motor_attempts = Arc::new(AtomicUsize::new(0));
        let motors = Arc::new(Mutex::new(Vec::new()));
        let rejection_attempts = Arc::new(AtomicUsize::new(0));
        let body = RejectingMotionCockpit {
            inner: CountingCockpit {
                motor_attempts: Arc::clone(&motor_attempts),
                motors: Arc::clone(&motors),
                body: BodySense {
                    last_update_ms: 100,
                    ..BodySense::default()
                },
            },
            rejection_attempts: Arc::clone(&rejection_attempts),
        };
        let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
            .with_autonomous_motion(true);

        let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
        assert_eq!(rejection_attempts.load(Ordering::SeqCst), 1);
        let motors = motors.lock().unwrap();
        assert_eq!(motors.last(), Some(&MotorCommand::stop()));
        assert_eq!(
            snapshot
                .action_debug
                .as_ref()
                .and_then(|debug| debug.get("why_not_moving"))
                .and_then(|reason| reason.as_str()),
            Some(
                "brainstem rejected motion command #42: stale_sequence; pausing motion retries for 1000 ms"
            )
        );
        assert_eq!(
            snapshot
                .action_debug
                .as_ref()
                .and_then(|debug| debug.get("motor_applied"))
                .and_then(|value| value.as_bool()),
            Some(false)
        );
    }

    #[tokio::test]
    async fn real_robot_slow_runner_pauses_motion_after_command_rejection() {
        let motor_attempts = Arc::new(AtomicUsize::new(0));
        let motors = Arc::new(Mutex::new(Vec::new()));
        let rejection_attempts = Arc::new(AtomicUsize::new(0));
        let body = RejectingMotionCockpit {
            inner: CountingCockpit {
                motor_attempts: Arc::clone(&motor_attempts),
                motors: Arc::clone(&motors),
                body: BodySense {
                    last_update_ms: 100,
                    ..BodySense::default()
                },
            },
            rejection_attempts: Arc::clone(&rejection_attempts),
        };
        let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
            .with_autonomous_motion(true);

        let (_first_snapshot, _first_tick) = runner.tick_slow_manual().await.unwrap();
        let (second_snapshot, second_tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(second_tick.chosen_action, Some(ActionPrimitive::Stop));
        assert_eq!(rejection_attempts.load(Ordering::SeqCst), 1);
        assert_eq!(motors.lock().unwrap().last(), Some(&MotorCommand::stop()));
        assert!(second_snapshot
            .action_debug
            .as_ref()
            .and_then(|debug| debug.get("why_not_moving"))
            .and_then(|reason| reason.as_str())
            .is_some_and(|reason| reason.contains("pausing motion retries")));
        assert_eq!(
            second_snapshot
                .action_debug
                .as_ref()
                .and_then(|debug| debug.get("motion_rejection"))
                .and_then(|debug| debug.get("count")),
            Some(&serde_json::json!(1))
        );
    }

    #[tokio::test]
    async fn real_robot_slow_runner_latches_stuck_after_repeated_command_rejections() {
        let motor_attempts = Arc::new(AtomicUsize::new(0));
        let motors = Arc::new(Mutex::new(Vec::new()));
        let rejection_attempts = Arc::new(AtomicUsize::new(0));
        let body = RejectingMotionCockpit {
            inner: CountingCockpit {
                motor_attempts: Arc::clone(&motor_attempts),
                motors: Arc::clone(&motors),
                body: BodySense {
                    last_update_ms: 100,
                    ..BodySense::default()
                },
            },
            rejection_attempts: Arc::clone(&rejection_attempts),
        };
        let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
            .with_autonomous_motion(true);
        let now_ms = wall_time_ms();
        runner.motion_rejection = MotionRejectionState {
            first_ms: now_ms,
            last_ms: now_ms,
            latest_command_id: 41,
            latest_reason: Some("busy".to_string()),
            count: MOTION_REJECTION_STUCK_AFTER - 1,
            ..MotionRejectionState::default()
        };

        let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
        assert!(snapshot
            .action_debug
            .as_ref()
            .and_then(|debug| debug.get("why_not_moving"))
            .and_then(|reason| reason.as_str())
            .is_some_and(|reason| reason.contains("operator intervention needed")));
        assert_eq!(
            snapshot
                .action_debug
                .as_ref()
                .and_then(|debug| debug.get("motion_rejection"))
                .and_then(|debug| debug.get("stuck")),
            Some(&serde_json::json!(true))
        );
    }

    struct ManualRuntime;

    #[async_trait::async_trait]
    impl RuntimeLoop for ManualRuntime {
        async fn tick(
            &mut self,
            mut now: Now,
            _latent: ExperienceLatent,
            _futures: Vec<FuturePrediction>,
        ) -> Result<RuntimeTick> {
            let input = ReignInput {
                id: Uuid::new_v4(),
                issued_at_ms: now.t_ms,
                expires_at_ms: now.t_ms + 300,
                source: ReignSource::WebRemote,
                mode: ReignMode::Direct,
                command: pete_actions::ReignCommand::Go {
                    intensity: 0.50,
                    duration_ms: 300,
                },
                priority: 1.0,
                note: None,
            };
            now.reign.latest = Some(input.clone());
            let action = input.command.to_action().unwrap();
            let experience =
                Experience::new("test", "test", Vec::new(), Vec::new(), now.t_ms, now.t_ms);
            Ok(RuntimeTick {
                frame: ExperienceFrame {
                    id: Uuid::new_v4(),
                    t_ms: now.t_ms,
                    now,
                    sensations: Vec::new(),
                    impressions: Vec::new(),
                    experiences: vec![experience.clone()],
                    z: Some(ExperienceLatent::default()),
                    chosen_action: Some(action.clone()),
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
                    notes: Vec::new(),
                },
                experience,
                chosen_action: Some(action),
                recall: RecallBundle::default(),
                llm: LlmTickResult::default(),
                combobulation: None,
                inline_learning: InlineLearningTickStatus::default(),
            })
        }
    }

    struct QueueOnlyRuntime {
        queue: Arc<Mutex<ReignQueue>>,
        tick_attempts: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl RuntimeLoop for QueueOnlyRuntime {
        async fn tick(
            &mut self,
            _now: Now,
            _latent: ExperienceLatent,
            _futures: Vec<FuturePrediction>,
        ) -> Result<RuntimeTick> {
            self.tick_attempts.fetch_add(1, Ordering::SeqCst);
            anyhow::bail!("slow direct hardware should bypass runtime tick")
        }

        fn reign_sense(&self, now_ms: TimeMs) -> Result<ReignSense> {
            Ok(self.queue.lock().unwrap().sense(now_ms))
        }
    }

    #[tokio::test]
    async fn real_robot_slow_runner_applies_only_clamped_webremote_direct_motor() {
        let motor_attempts = Arc::new(AtomicUsize::new(0));
        let motors = Arc::new(Mutex::new(Vec::new()));
        let body = CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        };
        let mut runner =
            RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), ManualRuntime);

        let (_snapshot, _tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(motor_attempts.load(Ordering::SeqCst), 1);
        assert_eq!(
            motors.lock().unwrap().as_slice(),
            &[MotorCommand {
                forward: 0.05,
                turn: 0.0
            }]
        );
    }

    #[tokio::test]
    async fn real_robot_slow_direct_webremote_bypasses_slow_runtime_tick() {
        let motor_attempts = Arc::new(AtomicUsize::new(0));
        let motors = Arc::new(Mutex::new(Vec::new()));
        let mut body_sense = BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        };
        body_sense.cliff_sensors.front_left = 0.96;
        body_sense.cliff_sensors.front_right = 0.82;
        let body = CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: body_sense,
        };
        let queue = Arc::new(Mutex::new(ReignQueue::default()));
        queue.lock().unwrap().push(ReignInput {
            id: Uuid::new_v4(),
            issued_at_ms: 100,
            expires_at_ms: wall_time_ms().saturating_add(500),
            source: ReignSource::WebRemote,
            mode: ReignMode::Direct,
            command: ReignCommand::Go {
                intensity: 0.50,
                duration_ms: 300,
            },
            priority: 1.0,
            note: None,
        });
        let tick_attempts = Arc::new(AtomicUsize::new(0));
        let runtime = QueueOnlyRuntime {
            queue,
            tick_attempts: Arc::clone(&tick_attempts),
        };
        let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), runtime);

        let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(tick_attempts.load(Ordering::SeqCst), 0);
        assert_eq!(motor_attempts.load(Ordering::SeqCst), 1);
        assert_eq!(
            motors.lock().unwrap().as_slice(),
            &[MotorCommand {
                forward: 0.05,
                turn: 0.0
            }]
        );
        assert_eq!(
            tick.frame
                .now
                .extensions
                .get("action.motion_bridge")
                .and_then(|value| value.get("runtime_bypassed"))
                .and_then(|value| value.as_bool()),
            Some(true)
        );
    }

    #[tokio::test]
    async fn real_robot_slow_direct_webremote_stops_locally_while_charging() {
        let motor_attempts = Arc::new(AtomicUsize::new(0));
        let motors = Arc::new(Mutex::new(Vec::new()));
        let body = CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                charging: true,
                last_update_ms: 100,
                ..BodySense::default()
            },
        };
        let queue = Arc::new(Mutex::new(ReignQueue::default()));
        queue.lock().unwrap().push(ReignInput {
            id: Uuid::new_v4(),
            issued_at_ms: 100,
            expires_at_ms: wall_time_ms().saturating_add(500),
            source: ReignSource::WebRemote,
            mode: ReignMode::Direct,
            command: ReignCommand::Go {
                intensity: 0.50,
                duration_ms: 300,
            },
            priority: 1.0,
            note: None,
        });
        let tick_attempts = Arc::new(AtomicUsize::new(0));
        let runtime = QueueOnlyRuntime {
            queue,
            tick_attempts: Arc::clone(&tick_attempts),
        };
        let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), runtime);

        let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(tick_attempts.load(Ordering::SeqCst), 0);
        assert_eq!(motor_attempts.load(Ordering::SeqCst), 1);
        assert_eq!(motors.lock().unwrap().as_slice(), &[MotorCommand::stop()]);
        assert_eq!(
            tick.frame
                .now
                .extensions
                .get("action.motion_bridge")
                .and_then(|value| value.get("why_not_moving"))
                .and_then(|value| value.as_str()),
            Some("charging active")
        );
    }

    #[tokio::test]
    async fn real_robot_slow_direct_gamepad_bypasses_slow_runtime_tick() {
        let motor_attempts = Arc::new(AtomicUsize::new(0));
        let motors = Arc::new(Mutex::new(Vec::new()));
        let body = CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        };
        let queue = Arc::new(Mutex::new(ReignQueue::default()));
        queue.lock().unwrap().push(ReignInput {
            id: Uuid::new_v4(),
            issued_at_ms: 100,
            expires_at_ms: wall_time_ms().saturating_add(500),
            source: ReignSource::Gamepad,
            mode: ReignMode::Direct,
            command: ReignCommand::Drive {
                forward: 0.50,
                turn: -0.50,
                duration_ms: 300,
            },
            priority: 1.0,
            note: None,
        });
        let tick_attempts = Arc::new(AtomicUsize::new(0));
        let runtime = QueueOnlyRuntime {
            queue,
            tick_attempts: Arc::clone(&tick_attempts),
        };
        let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), runtime);

        let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(tick_attempts.load(Ordering::SeqCst), 0);
        assert_eq!(motor_attempts.load(Ordering::SeqCst), 1);
        assert_eq!(
            motors.lock().unwrap().as_slice(),
            &[MotorCommand {
                forward: 0.05,
                turn: -0.5
            }]
        );
        assert!(matches!(
            tick.frame.reign_input.as_ref().map(|input| &input.source),
            Some(ReignSource::Gamepad)
        ));
    }

    #[tokio::test]
    async fn real_robot_slow_direct_webremote_chirp_bypasses_runtime_without_motor() {
        let motor_attempts = Arc::new(AtomicUsize::new(0));
        let motors = Arc::new(Mutex::new(Vec::new()));
        let body = CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        };
        let queue = Arc::new(Mutex::new(ReignQueue::default()));
        queue.lock().unwrap().push(ReignInput {
            id: Uuid::new_v4(),
            issued_at_ms: 100,
            expires_at_ms: wall_time_ms().saturating_add(500),
            source: ReignSource::WebRemote,
            mode: ReignMode::Direct,
            command: ReignCommand::Chirp {
                pattern: ChirpPattern::Confirm,
            },
            priority: 1.0,
            note: None,
        });
        let tick_attempts = Arc::new(AtomicUsize::new(0));
        let runtime = QueueOnlyRuntime {
            queue,
            tick_attempts: Arc::clone(&tick_attempts),
        };
        let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), runtime);

        let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(tick_attempts.load(Ordering::SeqCst), 0);
        assert_eq!(motor_attempts.load(Ordering::SeqCst), 0);
        assert!(motors.lock().unwrap().is_empty());
        assert!(matches!(
            tick.chosen_action,
            Some(ActionPrimitive::Chirp {
                pattern: ChirpPattern::Confirm
            })
        ));
        assert!(matches!(
            tick.frame.reign_input.as_ref().map(|input| &input.command),
            Some(ReignCommand::Chirp {
                pattern: ChirpPattern::Confirm
            })
        ));
    }

    #[tokio::test]
    async fn real_robot_slow_direct_webremote_speak_bypasses_runtime_without_motor() {
        let motor_attempts = Arc::new(AtomicUsize::new(0));
        let motors = Arc::new(Mutex::new(Vec::new()));
        let body = CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        };
        let queue = Arc::new(Mutex::new(ReignQueue::default()));
        queue.lock().unwrap().push(ReignInput {
            id: Uuid::new_v4(),
            issued_at_ms: 100,
            expires_at_ms: wall_time_ms().saturating_add(500),
            source: ReignSource::WebRemote,
            mode: ReignMode::Direct,
            command: ReignCommand::Speak {
                text: "hello from reign".to_string(),
            },
            priority: 1.0,
            note: None,
        });
        let tick_attempts = Arc::new(AtomicUsize::new(0));
        let runtime = QueueOnlyRuntime {
            queue,
            tick_attempts: Arc::clone(&tick_attempts),
        };
        let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), runtime);

        let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

        assert_eq!(tick_attempts.load(Ordering::SeqCst), 0);
        assert_eq!(motor_attempts.load(Ordering::SeqCst), 0);
        assert!(motors.lock().unwrap().is_empty());
        assert!(matches!(
            tick.chosen_action,
            Some(ActionPrimitive::Speak { ref text }) if text == "hello from reign"
        ));
        assert!(matches!(
            tick.frame.reign_input.as_ref().map(|input| &input.command),
            Some(ReignCommand::Speak { text }) if text == "hello from reign"
        ));
    }

    #[tokio::test]
    async fn tick_adds_combobulated_experience() {
        let ledger = JsonlLedger::new("/tmp/pete-runtime-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            pete_llm::NoopLlmAgent,
        );
        let mut now = Now::blank(100, BodySense::default());
        now.ear.transcript = Some("hello world".to_string());

        let tick = runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();

        assert!(tick
            .frame
            .experiences
            .iter()
            .any(|experience| experience.text.contains("hello world")));
    }

    #[tokio::test]
    async fn tick_persists_recalled_experiences_as_memory_sensations() {
        let ledger = JsonlLedger::new("/tmp/pete-runtime-memory-recall-sensations-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            pete_llm::NoopLlmAgent,
        );
        let mut first = Now::blank(100, BodySense::default());
        first.ear.transcript = Some("charger alcove".to_string());
        runtime
            .tick(first, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();

        let mut second = Now::blank(200, BodySense::default());
        second.ear.transcript = Some("charger alcove".to_string());
        let tick = runtime
            .tick(second, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();

        let recall_sensation = tick
            .frame
            .sensations
            .iter()
            .find(|sensation| {
                sensation.modality == Modality::Memory
                    && sensation.payload_kind == SensationPayloadKind::MemoryRecall
                    && sensation.kind == "memory.recall.experience"
            })
            .expect("memory recall sensation");
        assert!(recall_sensation
            .payload
            .get("original_frame_id")
            .and_then(Value::as_str)
            .is_some());
        assert!(tick.frame.impressions.iter().any(|impression| {
            impression.sensation_id == Some(recall_sensation.id)
                && impression.text.starts_with("I remember")
        }));
        let context = tick.frame.embodied_context();
        assert!(context.sensations.iter().any(|sensation| {
            sensation.id == recall_sensation.id
                && sensation.modality == Modality::Memory
                && sensation.payload_kind == SensationPayloadKind::MemoryRecall
        }));
    }

    #[tokio::test]
    async fn tick_feeds_memory_loop_candidates_into_live_map() {
        let root = test_ledger_root("runtime-live-loop-closure");
        let ledger = JsonlLedger::new(&root);
        let config = MapConfig {
            resolution_m: 0.25,
            pose_graph_min_node_distance_m: 0.01,
            pose_graph_max_ticks_between_nodes: 1,
            ..MapConfig::default()
        };
        let mut runtime = test_runtime(ledger, FixedConductor::new(ActionPrimitive::Stop))
            .with_local_map(LocalMap::new(config));

        for step in 0..5 {
            runtime
                .tick(
                    mapped_scene_now(100 + step * 100, 0.0, &format!("seed-{step}")),
                    ExperienceLatent::default(),
                    Vec::new(),
                )
                .await
                .unwrap();
        }

        let tick = runtime
            .tick(
                mapped_scene_now(700, 0.05, "return"),
                ExperienceLatent::default(),
                Vec::new(),
            )
            .await
            .unwrap();
        let frame_id = tick.frame.id.to_string();

        assert_eq!(
            tick.frame
                .now
                .extensions
                .get("frame_id")
                .and_then(Value::as_str),
            Some(frame_id.as_str())
        );
        let summary = runtime.local_map.summary();
        assert!(
            summary.loop_closures_accepted > 0,
            "expected live map to accept a memory loop closure, got {summary:?}"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn analog_cliff_risk_alone_does_not_say_floor_falls_away() {
        let mut now = Now::blank(100, BodySense::default());
        now.body.cliff_sensors.front_left = 0.96;
        now.body.cliff_sensors.front_right = 0.82;

        let (_sensations, impressions) = derive_direct_impressions_from_now(&now);
        let body_text = impressions
            .iter()
            .find(|impression| impression.kind == "body.state.impression")
            .map(|impression| impression.text.as_str())
            .unwrap();

        assert!(!body_text.contains("floor feels like it falls away near me"));
        assert!(body_text.contains("cliff IR signal is uncertain"));
    }

    #[test]
    fn cockpit_charging_indicator_sets_body_charging() {
        let status = StatusSummary::from_raw(
            r#"{"create_sensors":{"charging_state":0,"charging_indicator":"on","charge_mah":1300,"capacity_mah":2600}}"#,
        );

        let body = body_sense_from_cockpit_status(status, 42);

        assert!(body.charging);
        assert_eq!(body.battery_level, 0.5);
        assert_eq!(body.last_update_ms, 42);
    }

    #[test]
    fn real_slow_blocks_charging_body() {
        let mut body = BodySense::default();
        body.charging = true;

        assert_eq!(
            real_slow_body_block_reason(&body).as_deref(),
            Some("charging active")
        );
    }

    #[test]
    fn direct_now_impressions_are_first_person_present() {
        let mut now = Now::blank(100, BodySense::default());
        now.ear.transcript = Some("hello world".to_string());
        now.body.flags.cliff_front_left = true;
        now.body.cliff_sensors.front_left = 0.8;
        now.extensions.insert(
            "test.context".to_string(),
            serde_json::json!({ "ok": true }),
        );

        let (_sensations, impressions) = derive_direct_impressions_from_now(&now);
        let body_text = impressions
            .iter()
            .find(|impression| impression.kind == "body.state.impression")
            .map(|impression| impression.text.as_str())
            .unwrap();
        assert!(body_text.contains("floor feels like it falls away near me"));
        assert!(!body_text.contains("cliffs L/FL/FR/R"));
        assert!(!body_text.contains("cliff levels"));

        assert!(!impressions.is_empty());
        for impression in impressions {
            assert!(
                impression.text.starts_with("I ")
                    || impression.text.starts_with("I'm ")
                    || impression.text.starts_with("My "),
                "impression should manifest embodiment in first person: {}",
                impression.text
            );
            assert!(
                impression.text.contains("confident")
                    || impression.text.contains("pretty sure")
                    || impression.text.contains("I think")
                    || impression.text.contains("may have")
                    || impression.text.contains("not sure"),
                "impression should express confidence in natural language: {}",
                impression.text
            );
            assert_eq!(
                impression
                    .payload
                    .get("generator")
                    .and_then(|value| value.as_str()),
                Some("mechanical")
            );
        }
    }

    #[test]
    fn surface_scene_graph_becomes_spatial_impression() {
        let mut now = Now::blank(100, BodySense::default());
        now.extensions.insert(
            "surface.scene_graph".to_string(),
            serde_json::json!({
                "floor": {"confidence": 0.82},
                "surfaces": [{"id": "floor"}, {"id": "wall_1"}],
                "clusters": [{"id": "cluster_1"}],
                "navigation": {
                    "front_clear_m": 0.6,
                    "left_clear_m": 1.4,
                    "right_clear_m": 0.3
                }
            }),
        );

        let (_sensations, impressions) = derive_direct_impressions_from_now(&now);
        let surface_text = impressions
            .iter()
            .find(|impression| impression.kind == "surface.scene_graph.impression")
            .map(|impression| impression.text.as_str())
            .unwrap();

        assert!(surface_text.contains("persistent geometry"));
        assert!(surface_text.contains("2 stable surfaces"));
        assert!(surface_text.contains("1 leftover clusters"));
        assert!(surface_text.contains("front 0.60m"));
    }

    #[test]
    fn asr_impressions_phrase_partial_and_final_confidence_naturally() {
        let mut partial = Now::blank(100, BodySense::default());
        partial.ear.asr = pete_now::AsrSense {
            transcript: Some("come over here".to_string()),
            is_final: false,
            confidence: 0.52,
            ..pete_now::AsrSense::default()
        };
        let (_sensations, partial_impressions) = derive_direct_impressions_from_now(&partial);
        let partial_text = partial_impressions
            .iter()
            .find(|impression| impression.kind == "audio.transcript.impression")
            .map(|impression| impression.text.as_str())
            .unwrap();
        assert_eq!(partial_text, "I think I heard \"come over here\".");

        let mut final_now = Now::blank(100, BodySense::default());
        final_now.ear.asr = pete_now::AsrSense {
            transcript: Some("come over here".to_string()),
            is_final: true,
            confidence: 0.93,
            ..pete_now::AsrSense::default()
        };
        let (_sensations, final_impressions) = derive_direct_impressions_from_now(&final_now);
        let final_text = final_impressions
            .iter()
            .find(|impression| impression.kind == "audio.transcript.impression")
            .map(|impression| impression.text.as_str())
            .unwrap();
        assert_eq!(
            final_text,
            "I'm confident I finally heard \"come over here\"."
        );
    }

    #[test]
    fn asr_possible_and_committed_speech_become_direct_impressions() {
        let mut now = Now::blank(100, BodySense::default());
        now.ear.asr = pete_now::AsrSense {
            transcript: Some("open the door".to_string()),
            possible_transcript: Some("open the".to_string()),
            committed_transcript: Some("open the door".to_string()),
            is_final: true,
            confidence: 0.72,
            ..pete_now::AsrSense::default()
        };

        let (sensations, impressions) = derive_direct_impressions_from_now(&now);

        assert!(sensations
            .iter()
            .any(|sensation| sensation.kind == "audio.possible_speech"));
        assert!(sensations
            .iter()
            .any(|sensation| sensation.kind == "audio.committed_speech"));
        assert!(impressions.iter().any(|impression| {
            impression.kind == "audio.possible_speech.impression"
                && impression.text.contains("possible speech")
                && impression.text.contains("open the")
        }));
        assert!(impressions.iter().any(|impression| {
            impression.kind == "audio.committed_speech.impression"
                && impression.text.contains("commit")
                && impression.text.contains("open the door")
        }));
    }

    #[test]
    fn model_assisted_safety_override_beats_high_score_candidate() {
        let mut body = BodySense::default();
        body.flags.wheel_drop = true;
        let now = Now::blank(100, body);
        let baseline = ActionPrimitive::Go {
            intensity: 0.15,
            duration_ms: 1_000,
        };
        let decision = select_action_from_scores(
            ActionSelectorMode::ModelAssisted,
            &now,
            baseline,
            vec![ActionSelectionCandidateScore {
                action: ActionPrimitive::Go {
                    intensity: 0.15,
                    duration_ms: 1_000,
                },
                score: 10.0,
                ..ActionSelectionCandidateScore::default()
            }],
        );

        assert_eq!(decision.selected_action, Some(ActionPrimitive::Stop));
        assert!(decision.safety_overrode);
    }

    #[test]
    fn model_assisted_does_not_yield_to_close_range_alone() {
        let body = BodySense::default();
        let mut now = Now::blank(100, body);
        now.range.nearest_m = Some(0.12);
        let baseline = ActionPrimitive::Go {
            intensity: -0.18,
            duration_ms: 300,
        };
        let decision = select_action_from_scores(
            ActionSelectorMode::ModelAssisted,
            &now,
            baseline.clone(),
            vec![ActionSelectionCandidateScore {
                action: ActionPrimitive::Turn {
                    direction: TurnDir::Right,
                    intensity: 0.25,
                    duration_ms: 750,
                },
                score: 10.0,
                ..ActionSelectionCandidateScore::default()
            }],
        );

        assert_ne!(decision.selected_action, Some(baseline));
        assert!(!decision.safety_overrode);
        assert!(decision.fallback_warnings.is_empty());
    }

    #[test]
    fn close_range_scores_baseline_recovery_candidate() {
        let body = BodySense::default();
        let mut now = Now::blank(100, body);
        now.range.nearest_m = Some(0.12);
        let baseline = ActionPrimitive::Turn {
            direction: TurnDir::Left,
            intensity: 0.75,
            duration_ms: 500,
        };
        let model_signals = CandidateModelSignals {
            danger: Some(DangerOutput {
                confidence: 1.0,
                ..Default::default()
            }),
            charge: Some(ChargeOutput {
                confidence: 1.0,
                ..Default::default()
            }),
            action_value: Some(ActionValueOutput {
                confidence: 1.0,
                ..Default::default()
            }),
        };

        let recovery = score_action_candidate(&now, &baseline, model_signals, Some(&baseline));
        let default_turn = score_action_candidate(
            &now,
            &ActionPrimitive::Turn {
                direction: TurnDir::Right,
                intensity: 0.25,
                duration_ms: 750,
            },
            model_signals,
            Some(&baseline),
        );

        assert!(recovery.score > default_turn.score);
        assert!(!recovery.fallback_used);
    }

    #[test]
    fn model_assisted_scores_active_stuck_recovery_candidate() {
        let body = BodySense::default();
        let mut now = Now::blank(100, body);
        now.extensions.insert(
            "sim.stuck".to_string(),
            serde_json::json!({
                "schema_version": 1,
                "values": [1.0, 0.0, 6.0, 100.0, 1.0, -1.0, 0.0, 0.0]
            }),
        );
        let baseline = ActionPrimitive::Go {
            intensity: -0.18,
            duration_ms: 300,
        };
        let model_signals = CandidateModelSignals {
            danger: Some(DangerOutput {
                confidence: 1.0,
                ..Default::default()
            }),
            charge: Some(ChargeOutput {
                confidence: 1.0,
                ..Default::default()
            }),
            action_value: Some(ActionValueOutput {
                confidence: 1.0,
                ..Default::default()
            }),
        };
        let recovery = score_action_candidate(&now, &baseline, model_signals, Some(&baseline));
        let turn = score_action_candidate(
            &now,
            &ActionPrimitive::Turn {
                direction: TurnDir::Right,
                intensity: 0.25,
                duration_ms: 750,
            },
            model_signals,
            Some(&baseline),
        );
        let decision = select_action_from_scores(
            ActionSelectorMode::ModelAssisted,
            &now,
            baseline.clone(),
            vec![turn, recovery],
        );

        assert_eq!(decision.selected_action, Some(baseline));
        assert!(decision.selected_score.unwrap_or_default() > 0.0);
        assert!(!decision.safety_overrode);
        assert!(decision.fallback_warnings.is_empty());
    }

    #[test]
    fn sim_stuck_extension_sets_recent_trap_memory_hints() {
        let mut now = Now::blank(100, BodySense::default());
        now.extensions.insert(
            "sim.stuck".to_string(),
            serde_json::json!({
                "schema_version": 1,
                "values": [1.0, 1.0, 6.0, 600.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.0]
            }),
        );

        apply_recent_trap_memory_hints(&mut now);

        assert!(now.memory.recent_trap_confidence >= 0.6);
        assert!(now.memory.recent_trap_direction_rad.unwrap() < 0.0);
    }

    #[test]
    fn scoring_prefers_charger_when_charge_value_is_high() {
        let now = Now::blank(100, BodySense::default());
        let stop = score_action_candidate(
            &now,
            &ActionPrimitive::Stop,
            CandidateModelSignals::default(),
            None,
        );
        let charger = score_action_candidate(
            &now,
            &ActionPrimitive::Approach {
                target: ApproachTarget::Charger,
            },
            CandidateModelSignals {
                charge: Some(ChargeOutput {
                    charge_probability: 0.8,
                    expected_battery_delta: 0.1,
                    dock_likelihood: 0.7,
                    confidence: 1.0,
                }),
                ..CandidateModelSignals::default()
            },
            None,
        );

        assert!(charger.score > stop.score);
    }

    #[test]
    fn charger_approach_is_a_default_action_value_candidate() {
        let candidates = action_value_candidate_actions(&[], None, &LlmTickResult::default());

        assert!(candidates.contains(&ActionPrimitive::Approach {
            target: ApproachTarget::Charger
        }));
    }

    #[test]
    fn scoring_prefers_approach_over_dock_when_charger_visible_but_not_contacted() {
        let mut now = Now::blank(100, BodySense::default());
        now.body.battery_level = 0.15;
        now.memory.place_charge_value = 0.7;
        now.extensions.insert(
            "sim.world".to_string(),
            serde_json::json!({
                "schema_version": 1,
                "values": [4.0, 4.0, 1.0, 0.35, 0.65]
            }),
        );
        let signals = CandidateModelSignals {
            charge: Some(ChargeOutput {
                charge_probability: 0.85,
                expected_battery_delta: 0.02,
                dock_likelihood: 0.35,
                confidence: 1.0,
            }),
            action_value: Some(ActionValueOutput {
                value: 0.1,
                confidence: 1.0,
            }),
            ..CandidateModelSignals::default()
        };

        let approach = score_action_candidate(
            &now,
            &ActionPrimitive::Approach {
                target: ApproachTarget::Charger,
            },
            signals,
            None,
        );
        let dock = score_action_candidate(&now, &ActionPrimitive::Dock, signals, None);

        assert!(approach.score > dock.score);
    }

    #[test]
    fn scoring_avoids_high_danger_candidate() {
        let now = Now::blank(100, BodySense::default());
        let safe = score_action_candidate(
            &now,
            &ActionPrimitive::Stop,
            CandidateModelSignals::default(),
            None,
        );
        let dangerous = score_action_candidate(
            &now,
            &ActionPrimitive::Go {
                intensity: 0.15,
                duration_ms: 1_000,
            },
            CandidateModelSignals {
                danger: Some(DangerOutput {
                    bump_risk: 0.95,
                    confidence: 1.0,
                    ..DangerOutput::default()
                }),
                ..CandidateModelSignals::default()
            },
            None,
        );

        assert!(safe.score > dangerous.score);
    }

    #[test]
    fn missing_model_signals_fall_back_with_warning() {
        let now = Now::blank(100, BodySense::default());
        let candidate = score_action_candidate(
            &now,
            &ActionPrimitive::Stop,
            CandidateModelSignals::default(),
            None,
        );
        let decision = select_action_from_scores(
            ActionSelectorMode::ModelAssisted,
            &now,
            ActionPrimitive::Stop,
            vec![candidate],
        );

        assert!(!decision.fallback_warnings.is_empty());
        assert!(decision.candidates[0].fallback_used);
    }

    #[tokio::test]
    async fn model_assisted_tick_logs_compact_decision_info() {
        let ledger = JsonlLedger::new("/tmp/pete-runtime-action-selector-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            pete_llm::NoopLlmAgent,
        )
        .with_action_selector_mode(ActionSelectorMode::ModelAssisted);

        let tick = runtime
            .tick(
                Now::blank(100, BodySense::default()),
                ExperienceLatent::default(),
                Vec::new(),
            )
            .await
            .unwrap();
        let decision = tick
            .frame
            .now
            .extensions
            .get("action_selector")
            .cloned()
            .and_then(|value| serde_json::from_value::<ActionSelectionDecision>(value).ok())
            .unwrap();

        assert_eq!(decision.mode, ActionSelectorMode::ModelAssisted);
        assert!(!decision.candidates.is_empty());
        assert!(decision.selected_action.is_some());
    }

    #[tokio::test]
    async fn goal_shadow_records_evaluation_without_replacing_baseline() {
        let ledger = JsonlLedger::new("/tmp/pete-runtime-goal-shadow-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            FixedConductor::new(ActionPrimitive::Stop),
            SimpleSafety::default(),
            pete_llm::NoopLlmAgent,
        )
        .with_action_selector_mode(ActionSelectorMode::GoalShadow);
        let tick = runtime
            .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();
        assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
        let decision = serde_json::from_value::<ActionSelectionDecision>(
            tick.frame.now.extensions["action_selector"].clone(),
        )
        .unwrap();
        assert_eq!(decision.mode, ActionSelectorMode::GoalShadow);
        assert!(decision.selected_goal.is_none());
        assert!(decision.shadow_selected_goal.is_some());
        assert!(tick.frame.now.extensions.contains_key("goal_system"));
    }

    #[tokio::test]
    async fn goal_mode_executes_goal_behavior_and_publishes_homeostatic_drives() {
        let ledger = JsonlLedger::new("/tmp/pete-runtime-goal-mode-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            FixedConductor::new(ActionPrimitive::Stop),
            SimpleSafety::default(),
            pete_llm::NoopLlmAgent,
        )
        .with_action_selector_mode(ActionSelectorMode::Goal);
        let mut now = idle_now(100);
        now.body.battery_level = 0.05;
        let tick = runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();
        let decision = serde_json::from_value::<ActionSelectionDecision>(
            tick.frame.now.extensions["action_selector"].clone(),
        )
        .unwrap();
        assert_eq!(decision.selected_goal.as_deref(), Some("seek_charger"));
        assert!(matches!(
            decision.selected_behavior.as_deref(),
            Some("inspect_for_charger" | "systematic_charger_search")
        ));
        assert_ne!(tick.chosen_action, Some(ActionPrimitive::Dock));
        assert!(tick.frame.now.drives.battery_hunger > 0.5);
    }

    #[tokio::test]
    async fn goal_mode_assist_is_only_an_affordance_bias_but_direct_still_overrides() {
        let build_runtime = |path: &'static str| {
            let ledger = JsonlLedger::new(path);
            let memory = InMemoryExperienceStore::new();
            let recall = memory.clone();
            MinimalRuntime::new(
                ledger,
                memory,
                recall,
                FixedConductor::new(ActionPrimitive::Stop),
                SimpleSafety::default(),
                pete_llm::NoopLlmAgent,
            )
            .with_action_selector_mode(ActionSelectorMode::Goal)
        };
        let mut assisted = build_runtime("/tmp/pete-runtime-goal-assist-test");
        assisted.reign_queue.lock().unwrap().push(test_reign_input(
            100,
            ReignMode::Assist,
            ReignCommand::Dock,
            2_000,
        ));
        let assisted_tick = assisted
            .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();
        assert_ne!(assisted_tick.chosen_action, Some(ActionPrimitive::Dock));

        let mut direct = build_runtime("/tmp/pete-runtime-goal-direct-test");
        direct.reign_queue.lock().unwrap().push(test_reign_input(
            100,
            ReignMode::Direct,
            ReignCommand::Dock,
            2_000,
        ));
        let direct_tick = direct
            .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();
        assert_eq!(direct_tick.chosen_action, Some(ActionPrimitive::Dock));
    }

    #[test]
    fn memory_backed_baseline_action_is_a_selector_candidate_context() {
        let mut now = idle_now(100);
        mark_corrected_map_trusted(&mut now);
        now.memory.place_danger = 0.9;
        now.memory.nearby_best_safe_direction_rad = Some(-0.8);
        let memory_action = ActionPrimitive::Turn {
            direction: TurnDir::Right,
            intensity: 0.5,
            duration_ms: 1_000,
        };
        let default_action = ActionPrimitive::Go {
            intensity: 0.15,
            duration_ms: 1_000,
        };

        assert!(memory_navigation_candidate_context(&now, &memory_action));
        assert!(!memory_navigation_candidate_context(&now, &default_action));
    }

    #[tokio::test]
    async fn direct_reign_overrides_model_assisted_selector() {
        let ledger = JsonlLedger::new("/tmp/pete-runtime-reign-model-assisted-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            pete_llm::NoopLlmAgent,
        )
        .with_action_selector_mode(ActionSelectorMode::ModelAssisted);
        let command = ReignCommand::Turn {
            direction: TurnDir::Right,
            intensity: 0.5,
            duration_ms: 500,
        };
        runtime.reign_queue.lock().unwrap().push(test_reign_input(
            100,
            ReignMode::Direct,
            command.clone(),
            2_000,
        ));
        let mut now = idle_now(100);
        now.drives.curiosity = 1.0;

        let tick = runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();

        assert_eq!(tick.chosen_action, command.to_action());
        let decision = tick
            .frame
            .now
            .extensions
            .get("action_selector")
            .cloned()
            .and_then(|value| serde_json::from_value::<ActionSelectionDecision>(value).ok())
            .unwrap();
        assert_eq!(decision.selected_action, command.to_action());
        assert!(tick
            .frame
            .reign_outcome
            .as_ref()
            .map(|outcome| outcome.accepted_by_conductor)
            .unwrap_or(false));
    }

    #[tokio::test]
    async fn assist_reign_overrides_model_assisted_selector_immediately() {
        let ledger = JsonlLedger::new("/tmp/pete-runtime-assist-reign-model-assisted-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            pete_llm::NoopLlmAgent,
        )
        .with_action_selector_mode(ActionSelectorMode::ModelAssisted);
        let command = ReignCommand::Turn {
            direction: TurnDir::Right,
            intensity: 0.5,
            duration_ms: 500,
        };
        runtime.reign_queue.lock().unwrap().push(test_reign_input(
            100,
            ReignMode::Assist,
            command.clone(),
            2_000,
        ));
        let mut now = idle_now(100);
        now.drives.curiosity = 1.0;

        let tick = runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();

        assert_eq!(tick.chosen_action, command.to_action());
        let decision = tick
            .frame
            .now
            .extensions
            .get("action_selector")
            .cloned()
            .and_then(|value| serde_json::from_value::<ActionSelectionDecision>(value).ok())
            .unwrap();
        assert_eq!(decision.selected_action, command.to_action());
        assert!(tick
            .frame
            .reign_outcome
            .as_ref()
            .map(|outcome| outcome.accepted_by_conductor)
            .unwrap_or(false));
    }

    #[tokio::test]
    async fn observe_or_suggest_reign_does_not_mechanically_override_selector() {
        for mode in [ReignMode::ObserveOnly, ReignMode::Suggest] {
            let ledger = JsonlLedger::new(format!("/tmp/pete-runtime-non-driving-reign-{mode:?}"));
            let memory = InMemoryExperienceStore::new();
            let recall = memory.clone();
            let mut runtime = MinimalRuntime::new(
                ledger,
                memory,
                recall,
                FixedConductor::new(ActionPrimitive::Stop),
                SimpleSafety::default(),
                pete_llm::NoopLlmAgent,
            );
            let command = ReignCommand::Turn {
                direction: TurnDir::Right,
                intensity: 0.5,
                duration_ms: 500,
            };
            runtime
                .reign_queue
                .lock()
                .unwrap()
                .push(test_reign_input(100, mode, command, 2_000));

            let tick = runtime
                .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
                .await
                .unwrap();

            assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
            assert!(tick.frame.reign_input.is_some());
            assert!(!tick
                .frame
                .reign_outcome
                .as_ref()
                .map(|outcome| outcome.accepted_by_conductor)
                .unwrap_or(true));
        }
    }

    #[tokio::test]
    async fn stop_reign_becomes_now_event_and_chosen_action() {
        let ledger = JsonlLedger::new("/tmp/pete-runtime-reign-stop-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            pete_llm::NoopLlmAgent,
        );
        runtime.reign_queue.lock().unwrap().push(test_reign_input(
            100,
            ReignMode::Direct,
            ReignCommand::Stop,
            2_000,
        ));
        let mut body = BodySense::default();
        body.last_update_ms = 100;
        let now = Now::blank(100, body);

        let tick = runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();

        assert!(tick.frame.now.reign.active);
        assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
        assert!(tick
            .frame
            .sensations
            .iter()
            .any(|sensation| sensation.kind == "reign.command"));
        assert!(tick
            .frame
            .reign_input
            .as_ref()
            .map(|input| matches!(input.command, ReignCommand::Stop))
            .unwrap_or(false));
        assert!(tick
            .frame
            .reign_outcome
            .as_ref()
            .map(|outcome| outcome.accepted_by_conductor)
            .unwrap_or(false));
    }

    #[test]
    fn expired_reign_disappears_from_sense() {
        let mut queue = ReignQueue::default();
        queue.push(test_reign_input(
            100,
            ReignMode::Direct,
            ReignCommand::Stop,
            100,
        ));

        queue.drain_expired(250);
        let sense = queue.sense(250);

        assert!(!sense.active);
        assert!(sense.latest.is_none());
        assert_eq!(sense.pending_count, 0);
    }

    #[test]
    fn clear_marks_reign_sense_for_event_extraction() {
        let mut queue = ReignQueue::default();
        queue.push(test_reign_input(
            100,
            ReignMode::Direct,
            ReignCommand::Stop,
            1_000,
        ));

        queue.clear();
        let sense = queue.sense(150);

        assert!(!sense.active);
        assert!(sense.latest.is_none());
        assert_eq!(sense.clear_sequence, 1);
    }

    #[tokio::test]
    async fn safety_veto_beats_direct_go_reign_at_cliff() {
        let ledger = JsonlLedger::new("/tmp/pete-runtime-reign-safety-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            pete_llm::NoopLlmAgent,
        );
        runtime.reign_queue.lock().unwrap().push(test_reign_input(
            100,
            ReignMode::Direct,
            ReignCommand::Go {
                intensity: 0.5,
                duration_ms: 500,
            },
            2_000,
        ));
        let mut body = BodySense::default();
        body.flags.cliff_front_left = true;
        body.last_update_ms = 100;
        let now = Now::blank(100, body);

        let tick = runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();

        assert_eq!(
            tick.chosen_action,
            Some(ActionPrimitive::Go {
                intensity: 0.5,
                duration_ms: 500,
            })
        );
        assert!(tick
            .frame
            .reign_outcome
            .as_ref()
            .map(|outcome| outcome.vetoed_by_safety)
            .unwrap_or(false));
        assert!(tick
            .frame
            .notes
            .iter()
            .any(|note| note.contains("Safety vetoed")));
        let motor_gate = tick.frame.now.extensions.get("motor_gate").unwrap();
        assert_eq!(
            serde_json::from_value::<MotorCommand>(motor_gate["final_motor"].clone()).unwrap(),
            MotorCommand::stop()
        );
        assert_eq!(motor_gate["safety_reason"], "cliff");
    }

    #[tokio::test]
    async fn sim_runner_writes_frames_and_transitions() {
        let root = test_ledger_root("sim-runner-writes");
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(ledger.clone(), SimpleConductor::default());
        let (world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(10).await.unwrap();

        let frames = ledger.recent(20).await.unwrap();
        let transitions = read_transitions(&root);
        assert!(frames.len() >= 10);
        assert!(transitions.len() >= 9);
        assert!(transitions.iter().any(|transition| {
            transition.before.body.odometry.x_m != transition.after.body.odometry.x_m
                || transition.before.body.odometry.y_m != transition.after.body.odometry.y_m
        }));
    }

    #[tokio::test]
    async fn tick_records_erased_behavior_runs() {
        let root = test_ledger_root("runtime-behavior-runs");
        let ledger = JsonlLedger::new(&root);
        let mut runtime = test_runtime(ledger, FixedConductor::new(ActionPrimitive::Stop));

        let tick = runtime
            .tick(
                Now::blank(100, test_body(1.0, 1.0, 0.8, 100)),
                ExperienceLatent::default(),
                Vec::new(),
            )
            .await
            .unwrap();

        for behavior_id in [
            "danger",
            "charge",
            "future",
            "action_value",
            "eye_next",
            "ear_next",
        ] {
            assert!(
                tick.frame
                    .behavior_runs
                    .iter()
                    .any(|run| run.behavior_id == behavior_id),
                "missing behavior run for {behavior_id}"
            );
        }

        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn tick_runs_bump_event_script_and_records_safety_trace() {
        let root = test_ledger_root("runtime-bump-event-script");
        let ledger = JsonlLedger::new(&root);
        let mut runtime = test_runtime(
            ledger,
            FixedConductor::new(ActionPrimitive::Go {
                intensity: 0.3,
                duration_ms: 500,
            }),
        );
        let mut body = test_body(1.0, 1.0, 0.05, 100);
        body.flags.bump_left = true;
        let now = Now::blank(100, body);

        let tick = runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();

        assert_eq!(
            tick.chosen_action,
            Some(ActionPrimitive::Go {
                intensity: 0.3,
                duration_ms: 500
            })
        );
        assert!(tick
            .frame
            .now
            .extensions
            .get("safety.vetoed")
            .and_then(|value| value.as_bool())
            .unwrap_or(false));
        assert!(tick.frame.behavior_runs.iter().any(
            |run| run.behavior_id == "event_bump" && run.regime == BehaviorRegime::ShadowTrain
        ));
        let sequence = tick
            .frame
            .now
            .extensions
            .get("event_scripts")
            .and_then(|value| value.get("bump"))
            .cloned()
            .and_then(|value| serde_json::from_value::<SafeScriptSequence>(value).ok())
            .unwrap();
        assert_eq!(sequence.actions.len(), 5);
        assert!(matches!(
            sequence.actions.first().map(|action| &action.requested),
            Some(EventScriptAction::Chirp {
                pattern: ChirpPattern::Warning
            })
        ));
        assert!(matches!(
            sequence.actions.get(1).map(|action| &action.requested),
            Some(EventScriptAction::Say { .. } | EventScriptAction::Song { .. })
        ));
        assert!(sequence.actions.last().unwrap().vetoed);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn target_extractors_create_danger_charge_future_and_action_value_samples() {
        let action = ActionPrimitive::Go {
            intensity: 0.4,
            duration_ms: 1_000,
        };
        let mut before = Now::blank(100, test_body(1.0, 1.0, 0.0, 100));
        before.body.battery_level = 0.5;
        let mut after = before.clone();
        after.t_ms = 200;
        after.body.last_update_ms = 200;
        after.body.flags.bump_left = true;
        after.body.battery_level = 0.55;
        after.body.charging = true;
        after.eye.frames = vec![vec![0.25, 0.5, 0.75]];
        after.ear.features = vec![vec![0.2, 0.4], vec![0.6, 0.8]];
        let transition = ExperienceTransition {
            id: Uuid::new_v4(),
            before_frame_id: Uuid::new_v4(),
            before,
            before_z: ExperienceLatent {
                t_ms: 100,
                z: vec![0.1, 0.2],
                reconstruction_error: 0.0,
                prediction_error: 0.0,
                confidence: 0.8,
            },
            action: Some(action.clone()),
            predicted_futures: Vec::new(),
            after,
            after_z: ExperienceLatent {
                t_ms: 200,
                z: vec![0.3, 0.4],
                reconstruction_error: 0.0,
                prediction_error: 0.0,
                confidence: 0.9,
            },
            reward: Reward { value: 0.25 },
            surprise: SurpriseSense {
                total: 0.6,
                prediction_error: 0.1,
                ..SurpriseSense::default()
            },
            created_at_ms: 200,
        };

        let danger = DangerTargetExtractor.extract(&transition).unwrap().unwrap();
        assert_eq!(danger.source, TrainingSource::WorldOutcome);
        assert_eq!(danger.expected.bump_risk, 1.0);

        let charge = ChargeTargetExtractor.extract(&transition).unwrap().unwrap();
        assert_eq!(charge.expected.charge_probability, 1.0);
        assert!(charge.expected.expected_battery_delta > 0.0);

        let future = FutureTargetExtractor { offset_ms: 1_000 }
            .extract(&transition)
            .unwrap()
            .unwrap();
        assert_eq!(future.input.action, action);
        assert_eq!(future.expected.predicted_z, vec![0.3, 0.4]);

        let action_value = ActionValueTargetExtractor
            .extract(&transition)
            .unwrap()
            .unwrap();
        assert_eq!(action_value.source, TrainingSource::WorldOutcome);
        assert!((action_value.expected.value - 0.18).abs() < 0.0001);
        assert_eq!(action_value.expected.confidence, 1.0);

        let eye_next = EyeNextTargetExtractor { offset_ms: 100 }
            .extract(&transition)
            .unwrap()
            .unwrap();
        assert_eq!(eye_next.source, TrainingSource::WorldOutcome);
        assert_eq!(eye_next.expected.width, 64);
        assert_eq!(eye_next.expected.height, 48);
        assert_eq!(eye_next.expected.rgb.len(), 64 * 48 * 3);

        let ear_next = EarNextTargetExtractor { offset_ms: 100 }
            .extract(&transition)
            .unwrap()
            .unwrap();
        assert_eq!(ear_next.source, TrainingSource::WorldOutcome);
        assert_eq!(ear_next.expected.features, vec![0.2, 0.4, 0.6, 0.8]);
        assert!(ear_next.expected.pcm.is_empty());
    }

    #[test]
    fn ear_next_target_extractor_skips_missing_ear_frame() {
        let before = Now::blank(100, test_body(1.0, 1.0, 0.0, 100));
        let mut after = before.clone();
        after.t_ms = 200;
        let transition = ExperienceTransition {
            id: Uuid::new_v4(),
            before_frame_id: Uuid::new_v4(),
            before,
            before_z: ExperienceLatent {
                t_ms: 100,
                z: vec![0.1, 0.2],
                reconstruction_error: 0.0,
                prediction_error: 0.0,
                confidence: 0.8,
            },
            action: Some(ActionPrimitive::Stop),
            predicted_futures: Vec::new(),
            after,
            after_z: ExperienceLatent::default(),
            reward: Reward { value: 0.0 },
            surprise: SurpriseSense::default(),
            created_at_ms: 200,
        };

        let sample = EarNextTargetExtractor { offset_ms: 100 }
            .extract(&transition)
            .unwrap();

        assert!(sample.is_none());
    }

    #[test]
    fn behavior_registry_default_has_all_replaceable_slots() {
        let mut registry = BehaviorRegistry::default();
        let now = Now::blank(100, test_body(1.0, 1.0, 0.0, 100));
        let latent = ExperienceLatent {
            t_ms: 100,
            z: vec![0.0; 4],
            reconstruction_error: 0.0,
            prediction_error: 0.0,
            confidence: 0.8,
        };
        let action = ActionPrimitive::Dock;

        let locomotion = registry
            .locomotion
            .infer(&LocomotionInput::default(), 100)
            .unwrap();
        let danger = registry
            .danger
            .infer(&danger_behavior_input(&now, &latent, Some(&action)), 100)
            .unwrap();
        let charge = registry
            .charge
            .infer(&charge_behavior_input(&now, &latent, Some(&action)), 100)
            .unwrap();
        let future = registry
            .future
            .infer(
                &FutureInput {
                    latent: latent.clone(),
                    action: action.clone(),
                    offset_ms: 1_000,
                },
                100,
            )
            .unwrap();
        let action_value = registry
            .action_value
            .infer(
                &action_value_behavior_input(&now, &latent, Some(&action), None, None),
                100,
            )
            .unwrap();
        let eye_next = registry
            .eye_next
            .infer(
                &eye_next_behavior_input(&now, &latent, Some(&action), 100),
                100,
            )
            .unwrap();
        let ear_next = registry
            .ear_next
            .infer(
                &ear_next_behavior_input(&now, &latent, Some(&action), 100),
                100,
            )
            .unwrap();
        let experience = registry
            .experience
            .infer(&ExperienceBehaviorInput::from_now(&now), 100)
            .unwrap();

        assert_eq!(locomotion.record.behavior_id, "locomotion");
        assert_eq!(experience.record.behavior_id, "experience");
        assert_eq!(danger.record.behavior_id, "danger");
        assert_eq!(charge.record.behavior_id, "charge");
        assert_eq!(future.record.behavior_id, "future");
        assert_eq!(action_value.record.behavior_id, "action_value");
        assert_eq!(eye_next.record.behavior_id, "eye_next");
        assert_eq!(ear_next.record.behavior_id, "ear_next");
        assert!(locomotion.record.hardcoded_output.is_some());
        assert!(experience.record.hardcoded_output.is_some());
        assert!(danger.record.hardcoded_output.is_some());
        assert!(charge.record.hardcoded_output.is_some());
        assert!(future.record.hardcoded_output.is_some());
        assert!(action_value.record.hardcoded_output.is_some());
        assert!(eye_next.record.hardcoded_output.is_some());
        assert!(ear_next.record.hardcoded_output.is_some());
    }

    #[test]
    fn action_value_hardcoded_regime_returns_hardcoded_output() {
        let now = Now::blank(100, test_body(1.0, 1.0, 0.2, 100));
        let latent = ExperienceLatent {
            t_ms: 100,
            z: vec![0.0; 4],
            confidence: 0.8,
            ..ExperienceLatent::default()
        };
        let input =
            action_value_behavior_input(&now, &latent, Some(&ActionPrimitive::Dock), None, None);
        let mut behavior = action_value_behavior(
            BehaviorRegime::Hardcoded,
            None,
            FallbackPolicy::UseHardcoded,
        );

        let run = behavior.infer(&input, 100).unwrap();

        assert!(run.record.hardcoded_output.is_some());
        assert!(run.record.model_output.is_none());
        assert_eq!(run.record.selected_output, run.record.hardcoded_output);
    }

    #[test]
    fn action_value_shadow_infer_records_model_and_selects_hardcoded() {
        let now = Now::blank(100, test_body(1.0, 1.0, 0.2, 100));
        let latent = ExperienceLatent {
            t_ms: 100,
            z: vec![0.0; 4],
            confidence: 0.8,
            ..ExperienceLatent::default()
        };
        let input =
            action_value_behavior_input(&now, &latent, Some(&ActionPrimitive::Dock), None, None);
        let trainer = ActionValueNetTrainer::new(input.input.flat_features().len());
        let mut behavior = action_value_behavior(
            BehaviorRegime::ShadowInfer,
            Some(trainer),
            FallbackPolicy::UseHardcoded,
        );

        let run = behavior.infer(&input, 100).unwrap();

        assert!(run.record.hardcoded_output.is_some());
        assert!(run.record.model_output.is_some());
        assert_eq!(run.record.selected_output, run.record.hardcoded_output);
    }

    #[test]
    fn action_value_config_with_missing_checkpoint_falls_back_cleanly() {
        let config: BehaviorRegistryConfig = toml::from_str(
            r#"
            [behavior.action_value]
            regime = "shadow_infer"
            hardcoded = "action_value.handcoded"
            model = "action_value.burn.v0"
            checkpoint = "/tmp/pete-missing-action-value-checkpoint"
            fallback = "use_hardcoded"
            "#,
        )
        .unwrap();
        let mut stack = RuntimeModelStack::from_behavior_config(&config).unwrap();
        assert_eq!(
            stack.behaviors.action_value.regime,
            BehaviorRegime::ShadowInfer
        );

        let now = Now::blank(100, test_body(1.0, 1.0, 0.2, 100));
        let latent = ExperienceLatent {
            t_ms: 100,
            z: vec![0.0; 4],
            confidence: 0.8,
            ..ExperienceLatent::default()
        };
        let input =
            action_value_behavior_input(&now, &latent, Some(&ActionPrimitive::Dock), None, None);
        let run = stack
            .behaviors
            .action_value
            .infer(&input, now.t_ms)
            .unwrap();

        assert!(run.record.hardcoded_output.is_some());
        assert!(run.record.model_output.is_none());
    }

    #[tokio::test]
    async fn sim_runner_applies_chosen_action_to_world() {
        let ledger = JsonlLedger::new(test_ledger_root("sim-runner-action-world"));
        let runtime = test_runtime(
            ledger,
            FixedConductor::new(ActionPrimitive::Go {
                intensity: 0.4,
                duration_ms: 1_000,
            }),
        );
        let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.5, 7));
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(1).await.unwrap();
        let snapshot = runner.world.snapshot().await.unwrap();

        assert!(snapshot.body.odometry.x_m > 1.0);
        assert_eq!(runner.tick_count, 1);
    }

    #[tokio::test]
    async fn sim_runner_go_and_explore_send_non_stop_motion_and_change_pose() {
        for (name, action) in [
            (
                "go",
                ActionPrimitive::Go {
                    intensity: 0.4,
                    duration_ms: 1_000,
                },
            ),
            (
                "explore",
                ActionPrimitive::Explore {
                    style: ExploreStyle::RandomWalk,
                    duration_ms: 1_000,
                },
            ),
        ] {
            let ledger =
                JsonlLedger::new(test_ledger_root(&format!("sim-runner-{name}-motor-bridge")));
            let runtime = test_runtime(ledger, FixedConductor::new(action.clone()));
            let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
            world.set_body(test_body(1.0, 1.0, 0.5, 7));
            let mut runner = SimRunner::new(runtime, world, motors);
            let start = runner.world.body();
            let mut saw_non_zero_final_motor = false;
            let expected_selected_action = match &action {
                ActionPrimitive::Explore { duration_ms, .. } => ActionPrimitive::Drive {
                    forward: 0.2,
                    turn: 0.1,
                    duration_ms: *duration_ms,
                },
                _ => action.clone(),
            };

            runner
                .run_steps_observing_ticks(5, |snapshot, tick| {
                    let final_motor = final_motor_from_tick(tick);
                    if !is_near_zero_motor(final_motor) {
                        saw_non_zero_final_motor = true;
                    }
                    assert_eq!(
                        snapshot.final_selected_action,
                        Some(expected_selected_action.clone())
                    );
                })
                .await
                .unwrap();

            let end = runner.world.body();
            let delta = movement_delta_m(&start, &end);
            assert!(
                delta > 0.005,
                "{name} should move the simulated body, delta was {delta}"
            );
            assert!(saw_non_zero_final_motor, "{name} final motor was zero");
            assert!(
                !matches!(
                    runner.world.last_motion_sent(),
                    Some(MotionCommand::Stop) | None
                ),
                "{name} did not send non-stop motion to sim"
            );
        }
    }

    #[tokio::test]
    async fn sim_runner_reaches_charger_gets_positive_reward() {
        let root = test_ledger_root("sim-runner-charger-reward");
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(
            ledger,
            FixedConductor::new(ActionPrimitive::Go {
                intensity: 0.4,
                duration_ms: 1_000,
            }),
        );
        let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        let mut body = test_body(1.0, 1.0, 0.2, 7);
        body.battery_level = 0.2;
        world.set_body(body);
        world.add_object(SimObject::charger("charger", "charger", 1.38, 1.0, 0.18));
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(2).await.unwrap();
        let transitions = read_transitions(&root);

        let transition = transitions.last().unwrap();
        assert!(transition.after.body.charging);
        assert!(transition.reward.value > 0.0);
        assert!(transition.surprise.total > 0.0);
    }

    #[tokio::test]
    async fn sim_runner_collision_sets_bump_and_negative_reward() {
        let root = test_ledger_root("sim-runner-collision-reward");
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(
            ledger,
            FixedConductor::new(ActionPrimitive::Go {
                intensity: 0.4,
                duration_ms: 1_000,
            }),
        );
        let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.8, 7));
        world.add_object(SimObject::obstacle("box", "box", 1.31, 1.0, 0.1));
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(2).await.unwrap();
        let transitions = read_transitions(&root);

        let transition = transitions.last().unwrap();
        assert!(transition.after.body.flags.bump_left || transition.after.body.flags.bump_right);
        assert!(transition.reward.value < 0.0);
        assert!(transition.surprise.total > 0.0);
    }

    #[tokio::test]
    async fn sim_runner_resets_dead_uncharging_battery_and_records_critique() {
        let root = test_ledger_root("sim-runner-dead-battery-reset");
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(
            ledger.clone(),
            FixedConductor::new(ActionPrimitive::Go {
                intensity: 0.4,
                duration_ms: 1_000,
            }),
        );
        let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.0, 7));
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(1).await.unwrap();
        let snapshot = runner.world.snapshot().await.unwrap();
        let frames = ledger.recent(5).await.unwrap();
        let frame = frames.last().unwrap();

        assert_eq!(snapshot.body.battery_level, 1.0);
        assert!(!snapshot.body.charging);
        assert_eq!(snapshot.body.odometry.x_m, 2.0);
        assert_eq!(snapshot.body.odometry.y_m, 2.0);
        assert!(frame.llm_teaching.iter().any(|teaching| teaching
            .critique
            .as_deref()
            .is_some_and(|critique| { critique.contains("Dead battery away from the charger") })));
        assert!(frame
            .notes
            .iter()
            .any(|note| note.contains("VirtualDeadBattery")));

        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn sim_runner_gives_stuck_body_recovery_time_before_reset() {
        let root = test_ledger_root("sim-runner-stuck-recovery-time");
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(
            ledger.clone(),
            FixedConductor::new(ActionPrimitive::Go {
                intensity: 0.4,
                duration_ms: 1_000,
            }),
        );
        let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        let mut body = test_body(0.2, 0.2, 1.0, 7);
        body.odometry.heading_rad = std::f32::consts::PI;
        world.set_body(body);
        let mut runner = SimRunner::new(runtime, world, motors);

        runner
            .run_steps(STUCK_LOW_DISPLACEMENT_TICKS + 2)
            .await
            .unwrap();
        let snapshot = runner.world.snapshot().await.unwrap();
        let frames = ledger.recent(10).await.unwrap();

        assert_ne!(snapshot.body.odometry.x_m, 2.0);
        assert_ne!(snapshot.body.odometry.y_m, 2.0);
        assert!(!frames
            .iter()
            .any(|frame| frame.notes.iter().any(|note| note.contains("VirtualStuck"))));

        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn sim_with_danger_checkpoint_writes_shadow_predictions() {
        let root = test_ledger_root("sim-runner-danger-shadow");
        let checkpoint = danger_checkpoint_root("sim-runner-danger-shadow");
        let action = ActionPrimitive::Go {
            intensity: 0.4,
            duration_ms: 1_000,
        };
        write_test_danger_checkpoint(&checkpoint, action.clone());
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(ledger.clone(), FixedConductor::new(action))
            .with_models(RuntimeModelStack::with_danger_shadow_checkpoint(&checkpoint).unwrap());
        let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.8, 7));
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(1).await.unwrap();
        let frames = ledger.recent(5).await.unwrap();
        let frame = frames.last().unwrap();

        assert!(frame.now.predictions.danger_model.is_some());
        assert!(frame.now.predictions.danger_hardcoded.is_some());

        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(checkpoint);
    }

    #[tokio::test]
    async fn sim_attaches_fallback_predictions_to_embodied_experience() {
        let root = test_ledger_root("sim-runner-embodied-predictions");
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(ledger.clone(), FixedConductor::new(ActionPrimitive::Stop));
        let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.8, 7));
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(1).await.unwrap();
        let frames = ledger.recent(5).await.unwrap();
        let experience = frames.last().unwrap().experiences.last().unwrap();

        assert!(frames.last().unwrap().z.is_some());
        assert!(experience
            .predictions
            .iter()
            .any(|prediction| prediction.text.starts_with("hazard:")));
        assert!(experience
            .predictions
            .iter()
            .any(|prediction| prediction.text.starts_with("uncertainty:")));

        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn danger_shadow_prediction_does_not_bypass_safety() {
        let root = test_ledger_root("sim-runner-danger-shadow-safety");
        let checkpoint = danger_checkpoint_root("sim-runner-danger-shadow-safety");
        let action = ActionPrimitive::Go {
            intensity: 0.5,
            duration_ms: 500,
        };
        write_test_danger_checkpoint(&checkpoint, action.clone());
        let ledger = JsonlLedger::new(&root);
        let mut runtime = test_runtime(ledger, FixedConductor::new(action.clone()))
            .with_models(RuntimeModelStack::with_danger_shadow_checkpoint(&checkpoint).unwrap());
        let mut body = BodySense::default();
        body.flags.cliff_left = true;
        body.last_update_ms = 100;

        let tick = runtime
            .tick(
                Now::blank(100, body),
                ExperienceLatent::default(),
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(tick.chosen_action, Some(action));
        assert!(tick.frame.now.predictions.danger_model.is_some());
        assert!(tick
            .frame
            .notes
            .iter()
            .any(|note| note.contains("Safety vetoed")));

        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(checkpoint);
    }

    #[tokio::test]
    async fn sim_with_charge_checkpoint_writes_shadow_predictions() {
        let root = test_ledger_root("sim-runner-charge-shadow");
        let checkpoint = danger_checkpoint_root("sim-runner-charge-shadow");
        let action = ActionPrimitive::Dock;
        write_test_charge_checkpoint(&checkpoint, action.clone());
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(ledger.clone(), FixedConductor::new(action))
            .with_models(RuntimeModelStack::with_charge_shadow_checkpoint(&checkpoint).unwrap());
        let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.2, 7));
        world.add_object(SimObject::charger("charger", "charger", 1.2, 1.0, 0.18));
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(1).await.unwrap();
        let frames = ledger.recent(5).await.unwrap();
        let frame = frames.last().unwrap();

        assert!(frame.now.predictions.charge_model.is_some());
        assert!(frame.now.predictions.charge_hardcoded.is_some());

        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(checkpoint);
    }

    #[tokio::test]
    async fn sim_with_action_value_checkpoint_writes_shadow_predictions() {
        let root = test_ledger_root("sim-runner-action-value-shadow");
        let checkpoint = danger_checkpoint_root("sim-runner-action-value-shadow");
        let action = ActionPrimitive::Dock;
        write_test_action_value_checkpoint(&checkpoint, action.clone());
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(ledger.clone(), FixedConductor::new(action)).with_models(
            RuntimeModelStack::with_action_value_shadow_checkpoint(&checkpoint).unwrap(),
        );
        let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.2, 7));
        world.add_object(SimObject::charger("charger", "charger", 1.2, 1.0, 0.18));
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(1).await.unwrap();
        let frames = ledger.recent(5).await.unwrap();
        let frame = frames.last().unwrap();

        assert!(!frame.now.predictions.action_values_model.is_empty());
        assert!(!frame.now.predictions.action_values_hardcoded.is_empty());

        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(checkpoint);
    }

    #[tokio::test]
    async fn action_value_shadow_mode_does_not_override_conductor() {
        let root = test_ledger_root("sim-runner-action-value-shadow-choice");
        let checkpoint = danger_checkpoint_root("sim-runner-action-value-shadow-choice");
        write_test_action_value_checkpoint(&checkpoint, ActionPrimitive::Dock);
        let chosen = ActionPrimitive::Stop;
        let ledger = JsonlLedger::new(&root);
        let mut runtime = test_runtime(ledger, FixedConductor::new(chosen.clone())).with_models(
            RuntimeModelStack::with_action_value_shadow_checkpoint(&checkpoint).unwrap(),
        );

        let tick = runtime
            .tick(
                Now::blank(100, test_body(1.0, 1.0, 0.8, 100)),
                ExperienceLatent::default(),
                Vec::new(),
            )
            .await
            .unwrap();

        assert_eq!(tick.chosen_action, Some(chosen));
        assert!(!tick.frame.now.predictions.action_values_model.is_empty());

        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(checkpoint);
    }

    #[tokio::test]
    async fn sim_with_future_checkpoint_records_shadow_future_runs() {
        let root = test_ledger_root("sim-runner-future-shadow");
        let checkpoint = danger_checkpoint_root("sim-runner-future-shadow");
        write_test_future_checkpoint(&checkpoint, ActionPrimitive::Stop);
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(ledger.clone(), FixedConductor::new(ActionPrimitive::Stop))
            .with_models(RuntimeModelStack::with_future_shadow_checkpoint(&checkpoint).unwrap());
        let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.8, 7));
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(1).await.unwrap();
        let frames = ledger.recent(5).await.unwrap();
        let frame = frames.last().unwrap();
        let run = frame
            .behavior_runs
            .iter()
            .find(|run| run.behavior_id == "future" && run.model_json.is_some())
            .unwrap();

        assert_eq!(run.regime, BehaviorRegime::ShadowInfer);
        assert!(run.hardcoded_json.is_some());
        assert!(run.selected_json.is_some());
        assert!(!frame.predicted_futures.is_empty());

        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(checkpoint);
    }

    #[tokio::test]
    async fn inline_world_outcome_learning_observes_transition_sample() {
        let root = test_ledger_root("inline-world-outcome");
        let checkpoint = danger_checkpoint_root("inline-world-outcome");
        write_test_future_checkpoint(&checkpoint, ActionPrimitive::Stop);
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(ledger.clone(), FixedConductor::new(ActionPrimitive::Stop))
            .with_models(RuntimeModelStack::with_future_shadow_checkpoint(&checkpoint).unwrap())
            .with_inline_learning(InlineLearningConfig {
                mode: InlineLearningMode::WorldOutcome,
                behaviors: InlineLearningBehaviors {
                    danger: false,
                    charge: false,
                    future: true,
                    action_value: false,
                    eye_next: false,
                    ear_next: false,
                    experience: false,
                },
                max_train_steps_per_tick: 1,
            });
        let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.8, 7));
        let mut runner = SimRunner::new(runtime, world, motors);
        let mut observed_samples = 0usize;

        runner
            .run_steps_observing_ticks(3, |_snapshot, tick| {
                observed_samples =
                    observed_samples.saturating_add(tick.inline_learning.samples_observed);
            })
            .await
            .unwrap();

        assert!(observed_samples > 0);
        assert!(!read_transitions(&root).is_empty());

        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(checkpoint);
    }

    #[tokio::test]
    async fn disabled_inline_learning_reports_no_weight_updates() {
        let root = test_ledger_root("inline-disabled");
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(ledger.clone(), FixedConductor::new(ActionPrimitive::Stop));
        let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.8, 7));
        let mut runner = SimRunner::new(runtime, world, motors);
        let mut statuses = Vec::new();

        runner
            .run_steps_observing_ticks(3, |_snapshot, tick| {
                statuses.push(tick.inline_learning.clone());
            })
            .await
            .unwrap();

        assert!(statuses.iter().all(|status| !status.enabled));
        assert!(statuses
            .iter()
            .all(|status| status.samples_observed == 0 && status.train_steps_used == 0));

        let _ = fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn sim_with_ear_next_checkpoint_writes_shadow_prediction() {
        let root = test_ledger_root("sim-runner-ear-next-shadow");
        let checkpoint = danger_checkpoint_root("sim-runner-ear-next-shadow");
        let action = ActionPrimitive::Stop;
        write_test_ear_next_checkpoint(&checkpoint, action.clone());
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(ledger.clone(), FixedConductor::new(action))
            .with_models(RuntimeModelStack::with_ear_next_shadow_checkpoint(&checkpoint).unwrap());
        let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.8, 7));
        world.add_object(SimObject {
            id: "speaker".to_string(),
            label: "speaker".to_string(),
            kind: pete_sim::SimObjectKind::SoundSource {
                label: "speaker".to_string(),
            },
            x_m: 1.5,
            y_m: 1.2,
            radius_m: 0.12,
            color_rgb: [80, 80, 220],
            emits_sound: true,
            spoken_text: Some("listen to the room".to_string()),
            charge_rate: 0.0,
        });
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(1).await.unwrap();
        let frames = ledger.recent(5).await.unwrap();
        let frame = frames.last().unwrap();

        assert!(frame.now.predictions.ear_next_model.is_some());
        assert!(frame.now.predictions.ear_next_hardcoded.is_some());
        assert!(frame
            .behavior_runs
            .iter()
            .any(|run| run.behavior_id == "ear_next"));

        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(checkpoint);
    }

    #[tokio::test]
    async fn ear_next_shadow_mode_does_not_override_safety_or_action() {
        let root = test_ledger_root("sim-runner-ear-next-shadow-safety");
        let checkpoint = danger_checkpoint_root("sim-runner-ear-next-shadow-safety");
        let action = ActionPrimitive::Go {
            intensity: 0.5,
            duration_ms: 500,
        };
        write_test_ear_next_checkpoint(&checkpoint, action.clone());
        let ledger = JsonlLedger::new(&root);
        let mut runtime = test_runtime(ledger, FixedConductor::new(action.clone()))
            .with_models(RuntimeModelStack::with_ear_next_shadow_checkpoint(&checkpoint).unwrap());
        let mut body = BodySense::default();
        body.flags.cliff_left = true;
        body.last_update_ms = 100;
        let mut now = Now::blank(100, body);
        now.ear.features = vec![vec![0.2, 0.4, 0.6, 0.8]];

        let tick = runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();

        assert_eq!(tick.chosen_action, Some(action));
        assert!(tick.frame.now.predictions.ear_next_model.is_some());
        assert!(tick
            .frame
            .notes
            .iter()
            .any(|note| note.contains("Safety vetoed")));

        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(checkpoint);
    }

    #[tokio::test]
    async fn sim_with_experience_checkpoint_records_autoencoder_behavior_run() {
        let root = test_ledger_root("sim-runner-experience-shadow");
        let checkpoint = danger_checkpoint_root("sim-runner-experience-shadow");
        write_test_experience_checkpoint(&checkpoint);
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(ledger.clone(), FixedConductor::new(ActionPrimitive::Stop))
            .with_models(
                RuntimeModelStack::with_experience_shadow_checkpoint(&checkpoint).unwrap(),
            );
        let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        let mut body = test_body(1.0, 1.0, 0.8, 7);
        body.velocity.forward_m_s = 0.1;
        world.set_body(body);
        world.add_object(SimObject {
            id: "speaker".to_string(),
            label: "speaker".to_string(),
            kind: pete_sim::SimObjectKind::SoundSource {
                label: "speaker".to_string(),
            },
            x_m: 1.5,
            y_m: 1.2,
            radius_m: 0.12,
            color_rgb: [80, 80, 220],
            emits_sound: true,
            spoken_text: Some("the walls are awake".to_string()),
            charge_rate: 0.0,
        });
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(1).await.unwrap();
        let frames = ledger.recent(5).await.unwrap();
        let frame = frames.last().unwrap();
        let run = frame
            .behavior_runs
            .iter()
            .find(|run| run.behavior_id == "experience")
            .unwrap();

        assert_eq!(run.regime, BehaviorRegime::ShadowInfer);
        assert!(run.hardcoded_json.is_some());
        assert!(run.model_json.is_some());
        assert!(run.disagreement.unwrap_or_default().is_finite());
        assert!(frame.now.extensions.contains_key("experience.autoencoder"));

        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(checkpoint);
    }

    #[test]
    fn missing_experience_checkpoint_returns_no_latent_yet() {
        let config: BehaviorRegistryConfig = toml::from_str(
            r#"
            [behavior.experience]
            regime = "shadow_infer"
            hardcoded = "experience.no_latent_yet"
            model = "experience.autoencoder.v0"
            checkpoint = "/tmp/pete-missing-experience-checkpoint"
            fallback = "use_hardcoded"
            "#,
        )
        .unwrap();
        let mut stack = RuntimeModelStack::from_behavior_config(&config).unwrap();
        let now = Now::blank(100, test_body(1.0, 1.0, 0.8, 100));
        let run = stack
            .behaviors
            .experience
            .infer(&ExperienceBehaviorInput::from_now(&now), now.t_ms)
            .unwrap();

        assert_eq!(run.record.regime, BehaviorRegime::ShadowInfer);
        assert!(run.record.hardcoded_output.is_some());
        assert!(run.record.model_output.is_none());
        assert_eq!(run.chosen, run.record.hardcoded_output.unwrap());
        assert!(run.chosen.latent.z.is_empty());
        assert_eq!(run.chosen.confidence, 0.0);
    }

    #[tokio::test]
    async fn shared_reign_queue_controls_next_sim_tick() {
        let root = test_ledger_root("sim-runner-shared-reign");
        let ledger = JsonlLedger::new(&root);
        let queue = Arc::new(Mutex::new(ReignQueue::default()));
        queue.lock().unwrap().push(test_reign_input(
            7,
            ReignMode::Direct,
            ReignCommand::Turn {
                direction: pete_actions::TurnDir::Left,
                intensity: 0.5,
                duration_ms: 500,
            },
            2_000,
        ));
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let runtime = MinimalRuntime::with_reign_queue(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            pete_llm::NoopLlmAgent,
            queue,
        );
        let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.8, 7));
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(1).await.unwrap();
        let snapshot = runner.world.snapshot().await.unwrap();
        let frames = JsonlLedger::new(&root).recent(5).await.unwrap();
        let frame = frames.last().unwrap();

        assert!(snapshot.body.odometry.heading_rad > 0.0);
        assert!(frame.now.reign.active);
        assert!(frame
            .sensations
            .iter()
            .any(|sensation| sensation.kind == "reign.command"));
        assert!(frame.reign_input.is_some());
        assert!(frame
            .reign_outcome
            .as_ref()
            .map(|outcome| outcome.accepted_by_conductor)
            .unwrap_or(false));
    }

    #[tokio::test]
    async fn direct_reign_reverse_drives_sim_while_stuck_active() {
        let root = test_ledger_root("sim-runner-reign-reverse-interrupts-stuck");
        let ledger = JsonlLedger::new(&root);
        let queue = Arc::new(Mutex::new(ReignQueue::default()));
        queue.lock().unwrap().push(test_reign_input(
            7,
            ReignMode::Direct,
            ReignCommand::Reverse {
                intensity: 0.5,
                duration_ms: 500,
            },
            2_000,
        ));
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let runtime = MinimalRuntime::with_reign_queue(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            pete_llm::NoopLlmAgent,
            queue,
        );
        let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        world.set_body(test_body(1.0, 1.0, 1.0, 7));
        let mut runner = SimRunner::new(runtime, world, motors);
        runner.stuck.active = true;
        runner.stuck.phase = RecoveryPhase::Stop;
        runner.stuck.phase_ticks_remaining = 1;
        runner.stuck.turn_sign = 1.0;

        let mut observed_debug = None;
        runner
            .run_steps_observing(1, |snapshot| {
                observed_debug = snapshot.action_debug.clone();
            })
            .await
            .unwrap();
        let debug = observed_debug.unwrap();
        let motion = debug.get("motion_sent_to_sim").cloned().unwrap();

        let motion = serde_json::from_value::<MotionCommand>(motion.clone())
            .unwrap_or_else(|error| panic!("motion decode failed: {error}; debug={debug}"));
        assert_eq!(motion, MotionCommand::Forward { speed_m_s: -0.5 });
    }

    #[tokio::test]
    async fn column_trap_scenario_recovers_within_budget() {
        let root = test_ledger_root("sim-runner-column-trap-recovery");
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(ledger, SimpleConductor::default());
        let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ColumnTrap, 7));
        let start = (
            scenario.metadata.body.odometry.x_m,
            scenario.metadata.body.odometry.y_m,
        );
        let mut runner = SimRunner::new(runtime, scenario.world, scenario.motors);
        let mut saw_column = false;
        let mut recovered = false;

        runner
            .run_steps_observing(90, |snapshot| {
                if let Some(stuck) = snapshot
                    .extensions
                    .iter()
                    .find(|extension| extension.name == "sim.stuck")
                {
                    saw_column |= stuck.values.get(10).copied() == Some(3.0);
                    recovered |= stuck.values.get(7).copied() == Some(1.0);
                }
            })
            .await
            .unwrap();
        let end = runner.world.body();
        let distance = distance_between_points(start, (end.odometry.x_m, end.odometry.y_m));

        assert!(saw_column);
        assert!(recovered);
        assert!(distance > 0.10, "distance after recovery was {distance}");
    }

    #[derive(Clone, Copy, Debug, Default)]
    struct TrapRunMetrics {
        collision_frames: usize,
        stuck_frames: usize,
        recovered: bool,
        distance_m: f32,
    }

    async fn run_column_trap_metrics<C>(
        ledger_name: &str,
        conductor: C,
        steps: usize,
    ) -> TrapRunMetrics
    where
        C: Conductor + Send + 'static,
    {
        let root = test_ledger_root(ledger_name);
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(ledger, conductor);
        let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ColumnTrap, 7));
        let start = (
            scenario.metadata.body.odometry.x_m,
            scenario.metadata.body.odometry.y_m,
        );
        let mut runner = SimRunner::new(runtime, scenario.world, scenario.motors);
        let mut metrics = TrapRunMetrics::default();

        runner
            .run_steps_observing(steps, |snapshot| {
                let flags = &snapshot.body.flags;
                if flags.wall
                    || flags.bump_left
                    || flags.bump_right
                    || flags.cliff_front_left
                    || flags.cliff_front_right
                {
                    metrics.collision_frames += 1;
                }
                if let Some(stuck) = snapshot
                    .extensions
                    .iter()
                    .find(|extension| extension.name == "sim.stuck")
                {
                    metrics.stuck_frames +=
                        (stuck.values.first().copied().unwrap_or_default() > 0.0) as usize;
                    metrics.recovered |= stuck.values.get(7).copied() == Some(1.0);
                }
            })
            .await
            .unwrap();
        let end = runner.world.body();
        metrics.distance_m = distance_between_points(start, (end.odometry.x_m, end.odometry.y_m));
        metrics
    }

    #[tokio::test]
    async fn column_trap_recovery_beats_plain_explore_baseline() {
        let plain = run_column_trap_metrics(
            "sim-runner-column-trap-plain-explore",
            FixedConductor::new(ActionPrimitive::Explore {
                style: ExploreStyle::RandomWalk,
                duration_ms: 1_000,
            }),
            120,
        )
        .await;
        let recovered = run_column_trap_metrics(
            "sim-runner-column-trap-simple-recovery-comparison",
            SimpleConductor::default(),
            120,
        )
        .await;

        assert!(
            recovered.recovered,
            "expected recovery event, got {recovered:?}"
        );
        assert!(
            recovered.collision_frames < plain.collision_frames / 2,
            "recovery should reduce repeated collision frames; plain={plain:?} recovered={recovered:?}"
        );
        assert!(
            recovered.distance_m > plain.distance_m,
            "recovery should make more progress than plain explore; plain={plain:?} recovered={recovered:?}"
        );
        assert!(
            recovered.stuck_frames < plain.stuck_frames,
            "recovery should reduce repeated stuck frames; plain={plain:?} recovered={recovered:?}"
        );
    }

    #[derive(Clone, Debug)]
    struct FixedConductor {
        action: ActionPrimitive,
    }

    impl FixedConductor {
        fn new(action: ActionPrimitive) -> Self {
            Self { action }
        }
    }

    impl Conductor for FixedConductor {
        fn choose(&mut self, _input: ConductorInput) -> Result<ActionPrimitive> {
            Ok(self.action.clone())
        }
    }

    #[derive(Clone, Debug)]
    struct FixedLlmAgent {
        action: ActionPrimitive,
    }

    #[async_trait::async_trait]
    impl LlmAgent for FixedLlmAgent {
        async fn combobulate(
            &mut self,
            _now: &Now,
            _impressions: &[Impression],
            _embodied: Option<&EmbodiedContext>,
            _z: &ExperienceLatent,
            _futures: &[FuturePrediction],
            _recall_summary: &str,
        ) -> Result<Option<Combobulation>> {
            Ok(None)
        }

        async fn maybe_tick(
            &mut self,
            _now: &Now,
            _embodied: Option<&EmbodiedContext>,
            _z: &ExperienceLatent,
            _futures: &[FuturePrediction],
            _recall_summary: &str,
            _awareness_summary: Option<&str>,
        ) -> Result<LlmTickResult> {
            Ok(LlmTickResult {
                sense: pete_now::LlmSense {
                    schema_version: 1,
                    command_summary: Some("test command".to_string()),
                    critique: None,
                    confidence: 1.0,
                },
                conscious_command: Some(ConsciousCommand {
                    summary: "test command".to_string(),
                    action: Some(self.action.clone()),
                }),
                decision: Some(LlmDecision {
                    summary: "test command".to_string(),
                    action: Some(self.action.clone()),
                    confidence: 1.0,
                    ..LlmDecision::default()
                }),
                teaching: Vec::new(),
            })
        }

        async fn scientific_review(
            &mut self,
            _request: &LlmReviewRequest,
        ) -> Result<Option<LlmScientificReview>> {
            Ok(None)
        }
    }

    fn test_runtime<C>(
        ledger: JsonlLedger,
        conductor: C,
    ) -> MinimalRuntime<
        JsonlLedger,
        InMemoryExperienceStore,
        InMemoryExperienceStore,
        C,
        SimpleSafety,
        pete_llm::NoopLlmAgent,
    >
    where
        C: Conductor,
    {
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        MinimalRuntime::new(
            ledger,
            memory,
            recall,
            conductor,
            SimpleSafety::default(),
            pete_llm::NoopLlmAgent,
        )
    }

    #[tokio::test]
    async fn llm_command_action_overrides_default_curiosity_drive() {
        let ledger = JsonlLedger::new("/tmp/pete-runtime-llm-command-action-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let llm_action = ActionPrimitive::Go {
            intensity: 0.3,
            duration_ms: 700,
        };
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            FixedLlmAgent {
                action: llm_action.clone(),
            },
        );
        let mut now = idle_now(100);
        now.drives.curiosity = 1.0;

        let tick = runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();

        assert_eq!(tick.chosen_action, Some(llm_action.clone()));
        assert_eq!(
            tick.frame
                .conscious_command
                .as_ref()
                .and_then(|cmd| cmd.action.clone()),
            Some(llm_action.clone())
        );
        let decision = tick
            .frame
            .now
            .extensions
            .get("action_selector")
            .cloned()
            .and_then(|value| serde_json::from_value::<ActionSelectionDecision>(value).ok())
            .unwrap();
        assert_eq!(decision.selected_action, Some(llm_action));
    }

    #[tokio::test]
    async fn active_safe_reign_wins_over_llm_action() {
        let ledger = JsonlLedger::new("/tmp/pete-runtime-llm-reign-wins-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let queue = Arc::new(Mutex::new(ReignQueue::default()));
        let reign_command = ReignCommand::Turn {
            direction: TurnDir::Left,
            intensity: 0.4,
            duration_ms: 500,
        };
        queue.lock().unwrap().push(test_reign_input(
            100,
            ReignMode::Direct,
            reign_command.clone(),
            1_000,
        ));
        let mut runtime = MinimalRuntime::with_reign_queue(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            FixedLlmAgent {
                action: ActionPrimitive::Explore {
                    style: ExploreStyle::RandomWalk,
                    duration_ms: 1_000,
                },
            },
            queue,
        );
        let now = idle_now(100);

        let tick = runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();
        let proposal = tick
            .frame
            .now
            .extensions
            .get("llm.action_proposal")
            .cloned()
            .and_then(|value| serde_json::from_value::<LlmActionProposal>(value).ok())
            .unwrap();

        assert_eq!(tick.chosen_action, reign_command.to_action());
        assert!(!proposal.accepted);
        assert_eq!(
            proposal.ignored_reason.as_deref(),
            Some("safe active Reign command outranked LLM action")
        );
    }

    #[tokio::test]
    async fn unsafe_llm_action_is_safety_vetoed_and_recorded() {
        let ledger = JsonlLedger::new("/tmp/pete-runtime-llm-safety-veto-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            FixedLlmAgent {
                action: ActionPrimitive::Go {
                    intensity: 0.3,
                    duration_ms: 700,
                },
            },
        );
        let mut now = idle_now(100);
        now.body.flags.cliff_left = true;

        let tick = runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();
        let proposal = tick
            .frame
            .now
            .extensions
            .get("llm.action_proposal")
            .cloned()
            .and_then(|value| serde_json::from_value::<LlmActionProposal>(value).ok())
            .unwrap();

        assert_eq!(
            proposal.proposed_action,
            Some(ActionPrimitive::Go {
                intensity: 0.3,
                duration_ms: 700,
            })
        );
        assert!(proposal.accepted);
        assert!(proposal.safety_vetoed);
        assert_eq!(proposal.safety_reason.as_deref(), Some("cliff"));
        assert!(tick.frame.llm_teaching.iter().any(|teaching| teaching
            .critique
            .as_deref()
            .unwrap_or_default()
            .contains("unsafe action")));
    }

    fn arena() -> ArenaConfig {
        ArenaConfig {
            width_m: 4.0,
            height_m: 4.0,
        }
    }

    fn test_body(x_m: f32, y_m: f32, battery_level: f32, last_update_ms: u64) -> BodySense {
        let mut body = BodySense::default();
        body.odometry.x_m = x_m;
        body.odometry.y_m = y_m;
        body.battery_level = battery_level;
        body.last_update_ms = last_update_ms;
        body
    }

    fn stuck_test_snapshot(x_m: f32, y_m: f32, battery_level: f32) -> WorldSnapshot {
        let mut snapshot = WorldSnapshot::default();
        snapshot.body = test_body(x_m, y_m, battery_level, 100);
        snapshot.range.nearest_m = Some(0.12);
        snapshot.range.beams = vec![0.05, 0.08, 0.10, 0.09, 0.05];
        snapshot.extensions.push(ExtensionSense {
            schema_version: 1,
            name: "sim.world".to_string(),
            values: vec![4.0, 4.0, 0.0],
        });
        snapshot
    }

    #[test]
    fn stuck_detector_uses_rolling_low_displacement_window() {
        let mut detector = StuckRecoveryController::default();
        let action = ActionPrimitive::Explore {
            style: ExploreStyle::RandomWalk,
            duration_ms: 1_000,
        };

        for _ in 0..=STUCK_LOW_DISPLACEMENT_TICKS {
            detector.observe(&stuck_test_snapshot(0.2, 0.2, 1.0), Some(&action));
        }

        let status = detector.status();
        assert!(status.active);
        assert!(status.corner_trap);
        assert_eq!(status.stuck_ticks, STUCK_LOW_DISPLACEMENT_TICKS);
        assert!(status.event_started);
        assert_eq!(status.recovery_attempts, 1);
        assert_eq!(status.duration_ticks, 1);
        assert!(!status.reset_due);
    }

    #[test]
    fn recovered_stuck_event_reports_attempt_and_duration() {
        let mut detector = StuckRecoveryController::default();
        let action = ActionPrimitive::Explore {
            style: ExploreStyle::RandomWalk,
            duration_ms: 1_000,
        };

        for _ in 0..=STUCK_LOW_DISPLACEMENT_TICKS {
            detector.observe(&stuck_test_snapshot(0.2, 0.2, 1.0), Some(&action));
        }
        detector.observe(&stuck_test_snapshot(0.2, 0.2, 1.0), Some(&action));
        detector.observe(&stuck_test_snapshot(0.3, 0.2, 1.0), Some(&action));
        let status = detector.status();
        assert!(!status.active);
        assert!(status.recovered);
        assert_eq!(status.recovery_attempts, 1);
        assert!(status.duration_ticks >= 2);

        let extension = detector.extension(100);
        assert_eq!(extension.values.get(7).copied(), Some(1.0));
        assert_eq!(extension.values.get(11).copied(), Some(1.0));
        assert!(extension.values.get(3).copied().unwrap_or_default() >= 200.0);
    }

    #[test]
    fn repeated_stuck_escalates_recovery_instead_of_resetting() {
        let mut detector = StuckRecoveryController::default();
        let action = ActionPrimitive::Explore {
            style: ExploreStyle::RandomWalk,
            duration_ms: 1_000,
        };

        for _ in 0..=STUCK_LOW_DISPLACEMENT_TICKS {
            detector.observe(&stuck_test_snapshot(0.2, 0.2, 1.0), Some(&action));
        }
        detector.finish_recovery_success();
        detector.clearance_m = Some(0.10);
        detector.recovery_attempts = 1;
        detector.trap_anchor = Some((0.2, 0.2));

        for _ in 0..=STUCK_LOW_DISPLACEMENT_TICKS {
            detector.observe(&stuck_test_snapshot(0.2, 0.2, 1.0), Some(&action));
        }
        let mut snapshot = stuck_test_snapshot(0.2, 0.2, 1.0);
        detector.annotate_snapshot(&mut snapshot, 100);

        let status = detector.status();
        assert!(!status.reset_due);
        assert!(status.active);
        assert_eq!(status.repeated_trap_count, 1);
        let values = &snapshot
            .extensions
            .iter()
            .find(|extension| extension.name == "sim.stuck")
            .unwrap()
            .values;
        assert_eq!(values.get(9).copied(), Some(0.0));
        assert_eq!(values.get(12).copied(), Some(1.0));
    }

    #[test]
    fn dead_battery_state_is_reported_without_starting_recovery() {
        let mut detector = StuckRecoveryController::default();
        let action = ActionPrimitive::Explore {
            style: ExploreStyle::RandomWalk,
            duration_ms: 1_000,
        };

        for _ in 0..=STUCK_LOW_DISPLACEMENT_TICKS {
            detector.observe(&stuck_test_snapshot(0.2, 0.2, 0.0), Some(&action));
        }
        let mut snapshot = stuck_test_snapshot(0.2, 0.2, 0.0);
        detector.annotate_snapshot(&mut snapshot, 100);

        let status = detector.status();
        assert!(status.dead_battery);
        assert!(!status.active);
        let values = &snapshot
            .extensions
            .iter()
            .find(|extension| extension.name == "sim.stuck")
            .unwrap()
            .values;
        assert_eq!(values.get(8).copied(), Some(1.0));
    }

    #[test]
    fn stopped_column_trap_still_triggers_stuck_recovery() {
        let mut detector = StuckRecoveryController::default();
        let action = ActionPrimitive::Stop;

        for _ in 0..=STUCK_LOW_DISPLACEMENT_TICKS {
            detector.observe(&stuck_test_snapshot(2.0, 2.0, 1.0), Some(&action));
        }

        let status = detector.status();
        assert!(status.active);
        assert_eq!(status.trap_kind, TrapKind::Column);
        assert_eq!(status.stuck_ticks, STUCK_LOW_DISPLACEMENT_TICKS);
    }

    #[test]
    fn bump_left_chooses_rightward_escape() {
        let mut body = test_body(1.0, 1.0, 1.0, 100);
        body.flags.bump_left = true;
        let now = Now::blank(100, body);

        assert_eq!(
            hard_safety_action(&now),
            Some(ActionPrimitive::Turn {
                direction: TurnDir::Right,
                intensity: 0.7,
                duration_ms: 1_200
            })
        );
    }

    #[test]
    fn bump_right_chooses_leftward_escape() {
        let mut body = test_body(1.0, 1.0, 1.0, 100);
        body.flags.bump_right = true;
        let now = Now::blank(100, body);

        assert_eq!(
            hard_safety_action(&now),
            Some(ActionPrimitive::Turn {
                direction: TurnDir::Left,
                intensity: 0.7,
                duration_ms: 1_200
            })
        );
    }

    #[test]
    fn every_cliff_sensor_selects_stop_before_hardware_gate() {
        for sensor in ["left", "front_left", "front_right", "right"] {
            let mut body = test_body(1.0, 1.0, 1.0, 100);
            match sensor {
                "left" => body.flags.cliff_left = true,
                "front_left" => body.flags.cliff_front_left = true,
                "front_right" => body.flags.cliff_front_right = true,
                "right" => body.flags.cliff_right = true,
                _ => unreachable!(),
            }
            let now = Now::blank(100, body.clone());

            assert_eq!(
                hard_safety_action(&now),
                Some(ActionPrimitive::Stop),
                "{sensor}"
            );
            assert_eq!(
                real_slow_body_block_reason(&body).as_deref(),
                Some("cliff sensor active"),
                "{sensor}"
            );
        }
    }

    fn test_ledger_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("pete-{name}-{}", Uuid::new_v4()));
        let _ = fs::remove_dir_all(&root);
        root
    }

    fn danger_checkpoint_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("pete-{name}-checkpoint-{}", Uuid::new_v4()));
        let _ = fs::remove_dir_all(&root);
        root
    }

    fn write_test_danger_checkpoint(root: &Path, action: ActionPrimitive) {
        let mut body = test_body(1.0, 1.0, 0.8, 7);
        body.velocity.forward_m_s = 0.05;
        let now = Now::blank(100, body);
        let input = DangerInput::from_parts(Vec::new(), Some(&action), &now);
        let mut trainer = DangerNetTrainer::new(input.flat_features().len());
        trainer
            .train_step(
                &input,
                &pete_experience::DangerTarget {
                    bump: 0.2,
                    ..pete_experience::DangerTarget::default()
                },
            )
            .unwrap();
        trainer.save_checkpoint(root).unwrap();
    }

    fn write_test_charge_checkpoint(root: &Path, action: ActionPrimitive) {
        let mut body = test_body(1.0, 1.0, 0.2, 7);
        body.charging = false;
        let now = Now::blank(100, body);
        let input = ChargeInput::from_parts(Vec::new(), Some(&action), &now);
        let mut trainer = ChargeNetTrainer::new(input.flat_features().len());
        trainer
            .train_step(
                &input,
                &pete_experience::ChargeTarget {
                    charging_started: 1.0,
                    battery_delta: 0.03,
                    charging_after: 1.0,
                },
            )
            .unwrap();
        trainer.save_checkpoint(root).unwrap();
    }

    fn write_test_action_value_checkpoint(root: &Path, action: ActionPrimitive) {
        let mut body = test_body(1.0, 1.0, 0.2, 7);
        body.charging = false;
        let now = Now::blank(100, body);
        let input = ActionValueInput::from_parts(Vec::new(), Some(&action), &now);
        let mut trainer = ActionValueNetTrainer::new(input.flat_features().len());
        trainer
            .train_step(&input, &pete_experience::ActionValueTarget { value: 0.25 })
            .unwrap();
        trainer.save_checkpoint(root).unwrap();
    }

    fn write_test_future_checkpoint(root: &Path, action: ActionPrimitive) {
        let now = Now::blank(100, test_body(1.0, 1.0, 0.8, 100));
        let latent = ExperienceLatent {
            t_ms: now.t_ms,
            z: Vec::new(),
            reconstruction_error: 0.0,
            prediction_error: 0.0,
            confidence: 0.0,
        };
        let input = FutureInput {
            latent: latent.clone(),
            action,
            offset_ms: 100,
        };
        let mut trainer = FutureNetTrainer::new(input.flat_features().len(), 1);
        trainer.train_step(&input, &[0.0]).unwrap();
        trainer.save_checkpoint(root).unwrap();
    }

    fn write_test_ear_next_checkpoint(root: &Path, action: ActionPrimitive) {
        let body = test_body(1.0, 1.0, 0.8, 7);
        let mut now = Now::blank(100, body);
        now.ear.features = vec![vec![0.2, 0.4, 0.6, 0.8]];
        let input = EarNextInput::from_parts(Vec::new(), Some(&action), &now, 100);
        let mut trainer = EarNextNetTrainer::new(input.flat_features().len(), 4);
        trainer
            .train_step(
                &input,
                &pete_experience::EarNextTarget {
                    features: vec![0.2, 0.4, 0.6, 0.8],
                    ..pete_experience::EarNextTarget::default()
                },
            )
            .unwrap();
        trainer.save_checkpoint(root).unwrap();
    }

    fn write_test_experience_checkpoint(root: &Path) {
        let mut body = test_body(1.0, 1.0, 0.8, 7);
        body.velocity.forward_m_s = 0.1;
        let mut now = Now::blank(100, body);
        now.eye.frames = vec![vec![0.2, 0.4, 0.6, 0.8]];
        now.ear.features = vec![vec![0.2, 0.4, 0.6, 0.8]];
        now.memory.place_familiarity = 0.6;
        now.drives.curiosity = 0.4;
        let input = experience_encode_input_from_now(&now);
        let target = experience_decode_target_from_now(&now);
        let mut trainer = ExperienceAutoencoderTrainer::new(
            input.flat_features().len(),
            8,
            target.feature_lengths(),
        );
        trainer.train_step(&input, &target).unwrap();
        trainer.save_checkpoint(root).unwrap();
    }

    fn read_transitions(root: &Path) -> Vec<ExperienceTransition> {
        let mut out = Vec::new();
        read_transition_paths(root, &mut out);
        out
    }

    fn read_transition_paths(path: &Path, out: &mut Vec<ExperienceTransition>) {
        let Ok(entries) = fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                read_transition_paths(&path, out);
            } else if path.file_name().and_then(|name| name.to_str()) == Some("transitions.jsonl") {
                let Ok(contents) = fs::read_to_string(path) else {
                    continue;
                };
                out.extend(
                    contents
                        .lines()
                        .filter(|line| !line.trim().is_empty())
                        .filter_map(|line| serde_json::from_str(line).ok()),
                );
            }
        }
    }

    #[test]
    fn robot_initialized_typescript_behavior_emits_bringup_mouth_sequence() {
        let mut behavior = RobotInitializedScriptBehavior;
        let input = RobotInitializedEventInput {
            t_ms: 42,
            mode: "read-only".to_string(),
            body: "mock Create body connected".to_string(),
            battery_percent: Some(100),
            charging: Some(false),
            active_sensors: 2,
            requested_sensors: 3,
            ledger: "data/ledger/test".to_string(),
            tick_ms: 100,
            dashboard: Some("127.0.0.1:3000".to_string()),
            capture: None,
        };

        let output = behavior.infer(&input).unwrap();

        assert!(matches!(
            output.actions.first(),
            Some(EventScriptAction::Song { name }) if name == "bring_up"
        ));
        assert!(output.actions.iter().any(|action| matches!(
            action,
            EventScriptAction::Chirp {
                pattern: ChirpPattern::Confirm
            }
        )));
        assert!(output.actions.iter().any(|action| matches!(
            action,
            EventScriptAction::Say { text }
                if text.contains("Pete robot initialization complete")
        )));
    }

    fn test_reign_input(
        issued_at_ms: u64,
        mode: ReignMode,
        command: ReignCommand,
        ttl_ms: u64,
    ) -> ReignInput {
        ReignInput {
            id: Uuid::new_v4(),
            issued_at_ms,
            expires_at_ms: issued_at_ms + ttl_ms,
            source: ReignSource::WebRemote,
            mode,
            command,
            priority: 1.0,
            note: None,
        }
    }
}
