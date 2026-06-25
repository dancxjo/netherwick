use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use netherwick_actions::{
    action_to_motor_command, ActionPrimitive, ApproachTarget, ExploreStyle, InspectTarget,
    LlmActionProposal, ReignInput, ReignOutcome, TurnDir,
};
use netherwick_autonomic::{SafetyLayer, SafetyReason};
use netherwick_behaviors::{
    BehaviorConfig, BehaviorImplementation, BehaviorNodeState, BehaviorNodeUpdate, BehaviorRegime,
    BehaviorRegistryConfig, ErasedBehaviorRunRecord, FallbackPolicy, FunctionBehavior,
    ReplaceableBehavior, TargetExtractor, TrainingSample, TrainingSource,
};
use netherwick_body::{MotionCommand, MotorCommand, MotorComplex, RobotBody};
use netherwick_conductor::{Conductor, ConductorInput, SimpleConductor};
use netherwick_core::{Provenance, Reward, TimeMs};
use netherwick_events::{
    default_event_bus, DriveName, EventBus, EventContext, EventExtractor, Response,
};
use netherwick_experience::{
    action_features, action_value_input_from_transition_like, charge_input_from_transition_like,
    charge_target_from_transition_like, danger_input_from_transition_like,
    danger_target_from_transition_like, ear_next_input_from_transition_like,
    ear_next_target_from_now, experience_decode_target_from_now,
    eye_next_input_from_transition_like, eye_next_target_from_now, ActionValueInput,
    ActionValueOutput, BaselineRewardComputer, BaselineSurpriseComputer, ChargeInput, ChargeOutput,
    DangerInput, DangerOutput, EarNextInput, EarNextOutput, Experience, ExperienceBehaviorInput,
    ExperienceBehaviorOutput, ExperienceDecodeOutput, ExperienceEncodeInput, ExperienceEncoder,
    ExperienceLatent, EyeNextInput, EyeNextOutput, FeatureExperienceEncoder, FutureInput,
    FuturePrediction, FuturePredictor, Impression, RewardComputer, Sensation,
    StasisFuturePredictor, SurpriseComputer,
};
use netherwick_ledger::{
    ExperienceFrame, ExperienceTransition, LedgerWriter, PendingFrame, TransitionBuilder,
};
use netherwick_llm::{Combobulation, LlmAgent, LlmTickResult};
use netherwick_memory::{MemoryStore, Recall, RecallBundle, RecallQuery};
use netherwick_models::{
    read_action_value_metadata, read_charge_metadata, read_danger_metadata, read_ear_next_metadata,
    read_experience_autoencoder_metadata, read_eye_next_metadata, read_future_metadata,
    ActionValueNetTrainer, ChargeNetTrainer, CopyCurrentEarPredictor, CopyCurrentEyePredictor,
    DangerNetTrainer, EarNextNetTrainer, ExperienceAutoencoderTrainer, EyeNextNetTrainer,
    FutureNetTrainer, HardcodedActionValuePredictor, HardcodedChargePredictor,
    HardcodedDangerPredictor,
};
use netherwick_now::{
    ActionValuePrediction, ChargePrediction, DangerPrediction, DriveSense, EarPrediction,
    ExtensionSense, EyePrediction, Now, ReignSense, SafetySense,
};
use netherwick_sensors::{NowBuilder, SenseProducer, World, WorldSnapshot};
use netherwick_sim::{SimMotorComplex, VirtualWorld};
use serde::{Deserialize, Serialize};
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
    pub encoder: FeatureExperienceEncoder,
    pub predictor: StasisFuturePredictor,
    pub models: RuntimeModelStack,
    pub action_selector_mode: ActionSelectorMode,
    pub surprise_computer: BaselineSurpriseComputer,
    pub reward_computer: BaselineRewardComputer,
    pub transition_builder: TransitionBuilder,
    pub behavior_training_hub: BehaviorTrainingHub,
    pub inline_learning: InlineLearningConfig,
    pub nudge_policy: NudgePolicy,
    pub last_behavior_runs: Vec<ErasedBehaviorRunRecord>,
    nudge: NudgeController,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ActionSelectorMode {
    #[default]
    Baseline,
    Random,
    ModelAssisted,
    Scripted,
}

impl ActionSelectorMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Random => "random",
            Self::ModelAssisted => "model-assisted",
            Self::Scripted => "scripted",
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
    last_pose: Option<netherwick_core::Pose2>,
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

fn pose_delta_small(left: netherwick_core::Pose2, right: netherwick_core::Pose2) -> bool {
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

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ActionSelectionDecision {
    pub mode: ActionSelectorMode,
    pub candidates: Vec<ActionSelectionCandidateScore>,
    pub selected_action: Option<ActionPrimitive>,
    pub baseline_action: Option<ActionPrimitive>,
    pub selected_score: Option<f32>,
    pub safety_overrode: bool,
    pub fallback_warnings: Vec<String>,
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
                "Experience",
                "experience",
                "Experience",
                self.behaviors.experience.regime,
                self.behaviors.experience.hardcoded_id(),
                self.behaviors.experience.model_id(),
                self.behaviors.experience.fallback,
                vec![impl_id("experience.feature_encoder", "Feature encoder")],
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
                "EventBump",
                "event_bump",
                "on(bump)",
                self.behaviors.event_bump.regime,
                self.behaviors.event_bump.hardcoded_id(),
                self.behaviors.event_bump.model_id(),
                self.behaviors.event_bump.fallback,
                vec![impl_id("script.on_bump.v0", "Script hardcoded teacher")],
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
            "experience" => update_behavior!(experience),
            "danger" => update_behavior!(danger),
            "charge" => update_behavior!(charge),
            "future" => update_behavior!(future),
            "action_value" => update_behavior!(action_value),
            "eye_next" => update_behavior!(eye_next),
            "ear_next" => update_behavior!(ear_next),
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
    pub experience: ReplaceableBehavior<ExperienceBehaviorInput, ExperienceBehaviorOutput>,
    pub danger: ReplaceableBehavior<SituatedDangerInput, DangerOutput>,
    pub charge: ReplaceableBehavior<SituatedChargeInput, ChargeOutput>,
    pub future: ReplaceableBehavior<FutureInput, FuturePrediction>,
    pub action_value: ReplaceableBehavior<SituatedActionValueInput, ActionValueOutput>,
    pub conductor: ReplaceableBehavior<ConductorInput, ActionPrimitive>,
    pub eye_next: ReplaceableBehavior<SituatedEyeNextInput, EyeNextOutput>,
    pub ear_next: ReplaceableBehavior<SituatedEarNextInput, EarNextOutput>,
    pub event_bump: ReplaceableBehavior<BumpEventInput, EventScriptOutput>,
    pub event_face_detected: ReplaceableBehavior<FaceDetectedEventInput, EventScriptOutput>,
}

impl Default for BehaviorRegistry {
    fn default() -> Self {
        Self {
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
        Ok(Some(TrainingSample {
            input: action_value_input_from_transition_like(
                &transition.before_z,
                transition.action.as_ref(),
                &transition.before,
            ),
            expected: ActionValueOutput {
                value: transition.reward.value.clamp(-1.0, 1.0),
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

struct HardcodedExperienceBehavior {
    pub encoder: FeatureExperienceEncoder,
}

impl FunctionBehavior<ExperienceBehaviorInput, ExperienceBehaviorOutput>
    for HardcodedExperienceBehavior
{
    fn id(&self) -> &'static str {
        "experience.feature_encoder"
    }

    fn infer(&mut self, input: &ExperienceBehaviorInput) -> Result<ExperienceBehaviorOutput> {
        let latent = self.encoder.encode(&input.now)?;
        Ok(ExperienceBehaviorOutput {
            latent,
            reconstruction: None,
            reconstruction_loss: None,
            confidence: 1.0,
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
        let target = netherwick_experience::DangerTarget {
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
        let target = netherwick_experience::ChargeTarget {
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
        let target = netherwick_experience::ActionValueTarget {
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
        let target = netherwick_experience::EyeNextTarget {
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
        let target = netherwick_experience::EarNextTarget {
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
        Box::new(HardcodedExperienceBehavior {
            encoder: FeatureExperienceEncoder::new(),
        }),
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
        "script.on_bump.v0"
    }

    fn infer(&mut self, _input: &BumpEventInput) -> Result<EventScriptOutput> {
        Ok(EventScriptOutput {
            actions: vec![
                EventScriptAction::Stop,
                EventScriptAction::Rotate { deg: 180 },
                EventScriptAction::Go,
            ],
        })
    }
}

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
            actions: vec![EventScriptAction::Say {
                text: format!("Hello {label}"),
            }],
        })
    }
}

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

fn mechanical_reign_action(input: &Option<ReignInput>) -> Option<ActionPrimitive> {
    let input = input.as_ref()?;
    if !matches!(
        input.mode,
        netherwick_actions::ReignMode::Direct | netherwick_actions::ReignMode::Assist
    ) {
        return None;
    }
    input.command.to_action()
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
            encoder: FeatureExperienceEncoder::new(),
            predictor: StasisFuturePredictor,
            models: RuntimeModelStack::default(),
            action_selector_mode: ActionSelectorMode::Baseline,
            surprise_computer: BaselineSurpriseComputer,
            reward_computer: BaselineRewardComputer,
            transition_builder: TransitionBuilder::new(),
            behavior_training_hub: BehaviorTrainingHub::default(),
            inline_learning: InlineLearningConfig::default(),
            nudge_policy: NudgePolicy::default(),
            last_behavior_runs: Vec::new(),
            nudge: NudgeController::default(),
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
            encoder: FeatureExperienceEncoder::new(),
            predictor: StasisFuturePredictor,
            models: RuntimeModelStack::default(),
            action_selector_mode: ActionSelectorMode::Baseline,
            surprise_computer: BaselineSurpriseComputer,
            reward_computer: BaselineRewardComputer,
            transition_builder: TransitionBuilder::new(),
            behavior_training_hub: BehaviorTrainingHub::default(),
            inline_learning: InlineLearningConfig::default(),
            nudge_policy: NudgePolicy::default(),
            last_behavior_runs: Vec::new(),
            nudge: NudgeController::default(),
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
        let mechanical_reign_action = mechanical_reign_action(&reign_input);

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
        if futures.is_empty() {
            let (predicted, records) =
                predict_baseline_futures(&mut self.models.behaviors.future, &latent, now.t_ms)?;
            futures = predicted;
            behavior_runs.extend(records);
        }

        let recall = self
            .memory_recall
            .recall(RecallQuery::from_now(&now))
            .await?;
        now.memory = recall.sense.clone();
        now.extensions.insert(
            "memory.place".to_string(),
            serde_json::json!({
                "danger": now.memory.place_danger,
                "charge": now.memory.place_charge_value,
                "social": now.memory.place_social_value,
                "novelty": now.memory.place_novelty,
                "places_visited": now.memory.places_visited,
                "nearby_best_charge_direction_rad": now.memory.nearby_best_charge_direction_rad,
                "nearby_best_safe_direction_rad": now.memory.nearby_best_safe_direction_rad,
            }),
        );

        let (mut sensations, mut impressions) = derive_direct_impressions_from_now(&now);
        let mut experiences = derive_direct_experiences(&impressions, &sensations, now.t_ms);
        let mut teachings = Vec::new();
        let mut notes = Vec::new();
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
            &mut proposed_actions,
        );
        let (event_script_forced_action, event_script_records) =
            self.run_event_scripts(&mut now, &recall, &mut notes, &mut proposed_actions)?;
        behavior_runs.extend(event_script_records);

        let combobulation = self
            .llm
            .combobulate(
                &now,
                &impressions,
                &latent,
                &futures,
                &recall.first_person_summary,
            )
            .await?;

        let awareness_summary = combobulation.as_ref().map(|value| value.summary.as_str());
        let llm_tick = self
            .llm
            .maybe_tick(
                &now,
                &latent,
                &futures,
                &recall.first_person_summary,
                awareness_summary,
            )
            .await?;
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
        let mechanical_reign_action_for_selection =
            if mechanical_reign_action.is_some() && llm_has_safety_reason {
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
        if let Some(action) = llm_command_action.clone() {
            notes.push(format!("LlmActionProposal: proposed {:?}", action));
            proposals.push(action);
        }
        let action_value_candidates =
            action_value_candidate_actions(&proposals, reign_action.as_ref(), &llm_tick);

        let mut baseline_action = self.conductor.choose(ConductorInput {
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
            proposals,
        })?;
        if let Some(action) = mechanical_reign_action_for_selection.as_ref() {
            baseline_action = action.clone();
        }

        let mut model_predictions = Vec::new();
        let mut hardcoded_predictions = Vec::new();
        let mut candidate_scores = Vec::new();
        for action in &action_value_candidates {
            let candidate_danger_input = danger_behavior_input(&now, &latent, Some(action));
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
                &now,
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
                &now,
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
        if let Some(action) = mechanical_reign_action_for_selection.as_ref() {
            action_selection.selected_action = Some(action.clone());
            action_selection.selected_score = None;
            action_selection.safety_overrode = false;
        } else if let Some(action) = event_script_forced_action.as_ref() {
            action_selection.selected_action = Some(action.clone());
            action_selection.selected_score = None;
            action_selection.safety_overrode = false;
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
        );
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
        let chosen_action = mechanical_reign_action_for_selection
            .clone()
            .or_else(|| event_script_forced_action.clone())
            .unwrap_or(conductor_selected_action);
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

        let safety = self
            .safety
            .filter(&now, action_to_motor_command(Some(&chosen_action)));
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
            "prod.nudge".to_string(),
            serde_json::to_value(&self.nudge.status)?,
        );
        if safety.vetoed {
            if llm_action_proposal.accepted {
                let reason = describe_safety_reason(safety.reason.clone()).to_string();
                llm_action_proposal.safety_vetoed = true;
                llm_action_proposal.safety_reason = Some(reason.clone());
                notes.push(format!("LlmActionProposalSafetyVeto: {reason}"));
                teachings.push(netherwick_llm::LlmTeaching {
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
                &mut proposed_actions,
            );
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
        let frame = ExperienceFrame {
            id: Uuid::new_v4(),
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

        self.ledger.append(&frame).await?;
        self.memory_recall.observe_now(&frame.now).await?;
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
        let mut forced_action = None;
        let mut safe_sequences = serde_json::Map::new();

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
                forced_action = Some(first.clone());
                proposed_actions.push(first);
            }
            safe_sequences.insert("bump".to_string(), serde_json::to_value(&sequence)?);
            notes.push("EventScript:on(bump) emitted Stop -> Rotate(180) -> Go".to_string());
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

        if let Some(input) = face_detected_event_input(now, recall) {
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
        .find(|action| !matches!(action, ActionPrimitive::Speak { .. }))
}

fn script_action_to_primitive(action: &EventScriptAction) -> Option<ActionPrimitive> {
    match action {
        EventScriptAction::Say { text } => Some(ActionPrimitive::Speak { text: text.clone() }),
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

pub struct RealRobotRunner<R> {
    pub mode: RobotMode,
    pub body: Box<dyn RobotBody + Send>,
    pub sensors: Vec<Box<dyn SenseProducer + Send>>,
    pub runtime: R,
    pub tick_ms: u64,
    pub tick_count: usize,
    now_builder: NowBuilder,
}

impl<R> RealRobotRunner<R>
where
    R: RuntimeLoop + Send,
{
    pub fn new(
        mode: RobotMode,
        body: Box<dyn RobotBody + Send>,
        sensors: Vec<Box<dyn SenseProducer + Send>>,
        runtime: R,
    ) -> Self {
        Self {
            mode,
            body,
            sensors,
            runtime,
            tick_ms: 100,
            tick_count: 0,
            now_builder: NowBuilder::new(),
        }
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

        let body = self.body.read_body().await?;
        let mut packets = Vec::new();
        for sensor in &mut self.sensors {
            packets.push(sensor.poll().await?);
        }
        let t_ms = body.last_update_ms.max(wall_time_ms());
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

        let tick = self
            .runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await?;
        let mut snapshot = self.now_builder.snapshot();
        annotate_snapshot_from_tick(&mut snapshot, &tick);
        self.tick_count = self.tick_count.saturating_add(1);
        Ok((snapshot, tick))
    }
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

pub struct SimRunner<R> {
    pub runtime: R,
    pub world: VirtualWorld,
    pub motors: SimMotorComplex,
    pub tick_count: usize,
    pub tick_ms: u64,
    stuck: StuckRecoveryController,
}

const STUCK_LOW_DISPLACEMENT_TICKS: usize = 6;
const STUCK_WINDOW_DISPLACEMENT_EPSILON_M: f32 = 0.015;
const NEAR_ARENA_WALL_M: f32 = 0.32;
const RECOVERY_CLEARANCE_M: f32 = 0.18;
const SAME_TRAP_RADIUS_M: f32 = 0.18;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum RecoveryPhase {
    #[default]
    None,
    Stop,
    Reverse,
    Turn,
    Probe,
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
        let position = (snapshot.body.odometry.x_m, snapshot.body.odometry.y_m);
        let step_distance = self
            .last_position
            .map(|last| distance_between_points(last, position))
            .unwrap_or(f32::INFINITY);
        let commanded_motion = action_is_commanded_motion(action);
        self.dead_battery = is_dead_battery(snapshot);
        self.push_motion_sample(step_distance, commanded_motion);
        let low_displacement = self.rolling_low_displacement() && !snapshot.body.charging;
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

        self.clearance_m = snapshot.range.nearest_m;
        let trap_kind = classify_trap_kind(snapshot);
        let trapped = trap_kind != TrapKind::Unknown;
        if !self.active
            && commanded_motion
            && self.stuck_ticks >= STUCK_LOW_DISPLACEMENT_TICKS
            && trapped
            && !self.dead_battery
        {
            self.active = true;
            self.corner_trap = trap_kind == TrapKind::Corner;
            self.trap_kind = trap_kind;
            self.duration_ticks = 0;
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
            self.turn_sign = recovery_turn_sign(snapshot, self.last_failed_turn_sign);
            self.event_started = true;
        }

        self.last_position = Some(position);
    }

    fn recovery_motion(&mut self) -> Option<MotionCommand> {
        if !self.active {
            return None;
        }
        self.duration_ticks = self.duration_ticks.saturating_add(1);
        let motion = match self.phase {
            RecoveryPhase::Stop => MotionCommand::Stop,
            RecoveryPhase::Reverse => MotionCommand::Forward {
                speed_m_s: -reverse_speed(self.recovery_attempts),
            },
            RecoveryPhase::Turn => MotionCommand::Turn {
                turn_rad_s: self.turn_sign * turn_speed(self.recovery_attempts),
            },
            RecoveryPhase::Probe => MotionCommand::Drive {
                forward_m_s: 0.04,
                turn_rad_s: self.turn_sign * 0.25,
            },
            RecoveryPhase::None => return None,
        };
        self.advance_phase();
        Some(motion)
    }

    fn advance_phase(&mut self) {
        self.phase_ticks_remaining = self.phase_ticks_remaining.saturating_sub(1);
        if self.phase_ticks_remaining > 0 {
            return;
        }
        match self.phase {
            RecoveryPhase::Stop => {
                self.phase = RecoveryPhase::Reverse;
                self.phase_ticks_remaining = reverse_ticks(self.recovery_attempts);
            }
            RecoveryPhase::Reverse => {
                self.phase = RecoveryPhase::Turn;
                self.phase_ticks_remaining = turn_ticks(self.recovery_attempts);
            }
            RecoveryPhase::Turn => {
                self.phase = RecoveryPhase::Probe;
                self.phase_ticks_remaining = 1;
            }
            RecoveryPhase::Probe => {
                if self.clearance_achieved() {
                    self.finish_recovery_success();
                } else {
                    self.last_failed_turn_sign = Some(self.turn_sign);
                    self.recovery_attempts = self.recovery_attempts.saturating_add(1);
                    self.repeated_trap_count = self.repeated_trap_count.saturating_add(1);
                    self.turn_sign = -self.turn_sign;
                    self.phase = RecoveryPhase::Stop;
                    self.phase_ticks_remaining = 1;
                }
            }
            RecoveryPhase::None => {}
        }
    }

    fn clearance_achieved(&self) -> bool {
        self.clearance_m
            .map(|nearest| nearest > RECOVERY_CLEARANCE_M)
            .unwrap_or(true)
    }

    fn finish_recovery_success(&mut self) {
        self.active = false;
        self.corner_trap = false;
        self.trap_kind = TrapKind::Unknown;
        self.stuck_ticks = 0;
        self.duration_ticks = 0;
        self.phase = RecoveryPhase::None;
        self.phase_ticks_remaining = 0;
        self.recovery_attempts = 0;
        self.repeated_trap_count = 0;
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
        RecoveryPhase::Reverse => 2.0,
        RecoveryPhase::Turn => 3.0,
        RecoveryPhase::Probe => 4.0,
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

fn reverse_ticks(attempt: usize) -> usize {
    (6 + attempt.saturating_mul(2)).min(12)
}

fn turn_ticks(attempt: usize) -> usize {
    (12 + attempt.saturating_mul(4)).min(24)
}

fn reverse_speed(attempt: usize) -> f32 {
    (0.18 + attempt as f32 * 0.03).min(0.28)
}

fn turn_speed(attempt: usize) -> f32 {
    (0.8 + attempt as f32 * 0.1).min(1.0)
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
    pub fn new(runtime: R, world: VirtualWorld, motors: SimMotorComplex) -> Self {
        Self {
            runtime,
            world,
            motors,
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
            let now = snapshot.to_now(snapshot.body.last_update_ms);
            let tick = self
                .runtime
                .tick(now, ExperienceLatent::default(), Vec::new())
                .await?;
            annotate_snapshot_from_tick(&mut snapshot, &tick);
            observe(&snapshot, &tick);
            self.stuck.observe(&snapshot, tick.chosen_action.as_ref());
            if is_dead_battery(&snapshot) || reset_after_tick {
                self.world.reset_body_to_spawn();
                self.stuck.reset();
            } else {
                let motion = self
                    .stuck
                    .recovery_motion()
                    .unwrap_or_else(|| motor_command_to_motion(final_motor_from_tick(&tick)));
                self.motors.send(motion).await?;
            };
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
    let fallback_used =
        signals.danger.is_none() || signals.charge.is_none() || signals.action_value.is_none();
    let score = (-1.6 * danger) + (1.2 * charge) + action_value + curiosity
        - (0.8 * collision_risk)
        - low_battery_risk
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
    }
}

fn fallback_warnings_for_mode(mode: ActionSelectorMode) -> Vec<String> {
    if mode == ActionSelectorMode::ModelAssisted {
        vec!["model-assisted selector used hardcoded fallback estimates".to_string()]
    } else {
        Vec::new()
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
        return Some(ActionPrimitive::Go {
            intensity: -0.12,
            duration_ms: 750,
        });
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
    if now.body.battery_level <= 0.10 {
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
    if forward <= 0.0 {
        return 0.0;
    }
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
    nearest_risk.max(contact_risk)
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
    teachings: &mut Vec<netherwick_llm::LlmTeaching>,
    notes: &mut Vec<String>,
    proposed_actions: &mut Vec<ActionPrimitive>,
) {
    for response in responses {
        match response {
            Response::Emit(_) => {}
            Response::AddSensation(sensation) => sensations.push(sensation),
            Response::AddImpression(impression) => impressions.push(impression),
            Response::AddExperience(experience) => experiences.push(experience),
            Response::ProposeAction(action) => proposed_actions.push(action),
            Response::SetDrive { name, value } => set_drive(&mut now.drives, &name, value),
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
    teachings: &mut Vec<netherwick_llm::LlmTeaching>,
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

fn derive_direct_impressions_from_now(now: &Now) -> (Vec<Sensation>, Vec<Impression>) {
    let mut sensations = Vec::new();
    let mut impressions = Vec::new();
    let floor_feel = if now.body.flags.cliff_left
        || now.body.flags.cliff_front_left
        || now.body.flags.cliff_front_right
        || now.body.flags.cliff_right
        || now.body.cliff_sensors.max() >= 0.5
    {
        "the floor feels like it falls away near me"
    } else if now.body.cliff_sensors.max() > 0.0 {
        "the floor feels mostly steady with a faint edge-sense"
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

fn summarize_reign_command_for_runtime(input: &netherwick_actions::ReignInput) -> String {
    match &input.command {
        netherwick_actions::ReignCommand::Stop => "Stop".to_string(),
        netherwick_actions::ReignCommand::Go {
            intensity,
            duration_ms,
        } => format!("Go intensity {:.2} for {}ms", intensity, duration_ms),
        netherwick_actions::ReignCommand::Turn {
            direction,
            intensity,
            duration_ms,
        } => format!(
            "Turn {:?} intensity {:.2} for {}ms",
            direction, intensity, duration_ms
        ),
        netherwick_actions::ReignCommand::Inspect { target } => format!("Inspect {:?}", target),
        netherwick_actions::ReignCommand::Approach { target } => format!("Approach {:?}", target),
        netherwick_actions::ReignCommand::Dock => "Dock".to_string(),
        netherwick_actions::ReignCommand::Explore { duration_ms } => {
            format!("Explore for {}ms", duration_ms)
        }
        netherwick_actions::ReignCommand::Speak { text } => format!("Speak {text}"),
        netherwick_actions::ReignCommand::SetMode { mode } => format!("Set mode {:?}", mode),
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

fn set_drive(drives: &mut DriveSense, name: &DriveName, value: f32) {
    match name {
        DriveName::BatteryHunger => drives.battery_hunger = value,
        DriveName::DangerAvoidance => drives.danger_avoidance = value,
        DriveName::Curiosity => drives.curiosity = value,
        DriveName::SocialInterest => drives.social_interest = value,
        DriveName::Fatigue => drives.fatigue = value,
        DriveName::UncertaintyPressure => drives.uncertainty_pressure = value,
    }
}

fn describe_safety_reason(reason: Option<SafetyReason>) -> &'static str {
    match reason {
        Some(SafetyReason::WheelDrop) => "wheel drop",
        Some(SafetyReason::Cliff) => "cliff",
        Some(SafetyReason::BatteryCritical) => "critical battery",
        Some(SafetyReason::StaleSensors) => "stale sensors",
        Some(SafetyReason::LostBodyComms) => "lost body comms",
        Some(SafetyReason::MotorOutOfRange) => "motor out of range",
        Some(SafetyReason::HighDanger) => "high danger",
        Some(SafetyReason::RawLlmMotorRejected) => "raw llm motor rejected",
        Some(SafetyReason::ReadOnlyMode) => "read-only mode",
        None => "unknown reason",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use netherwick_actions::{ReignCommand, ReignMode, ReignSource};
    use netherwick_autonomic::SimpleSafety;
    use netherwick_body::{BodySense, MotorCommand, RobotBody};
    use netherwick_conductor::{Conductor, ConductorInput, SimpleConductor};
    use netherwick_experience::experience_encode_input_from_now;
    use netherwick_ledger::{ExperienceTransition, JsonlLedger, LedgerReader};
    use netherwick_llm::{ConsciousCommand, LlmDecision, LlmTickResult};
    use netherwick_memory::InMemoryExperienceStore;
    use netherwick_models::{
        ActionValueNetTrainer, ChargeNetTrainer, DangerNetTrainer, EarNextNetTrainer,
        ExperienceAutoencoderTrainer, FutureNetTrainer,
    };
    use netherwick_now::{Now, SurpriseSense};
    use netherwick_sensors::World;
    use netherwick_sim::{
        build_scenario, ArenaConfig, ScenarioConfig, ScenarioKind, SimObject, VirtualWorld,
    };
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn idle_now(t_ms: u64) -> Now {
        let mut body = BodySense::default();
        body.last_update_ms = t_ms;
        let mut now = Now::blank(t_ms, body);
        now.range.nearest_m = Some(1.0);
        now.range.beams = vec![1.0, 1.0, 1.0];
        now
    }

    fn test_conductor_input(action: ActionPrimitive) -> ConductorInput {
        ConductorInput {
            latent: ExperienceLatent::default(),
            drives: DriveSense::default(),
            memory: netherwick_now::MemorySense::default(),
            predictions: netherwick_now::PredictionSense::default(),
            surprise: SurpriseSense::default(),
            llm: netherwick_now::LlmSense::default(),
            safety: SafetySense::default(),
            reign: ReignSense::default(),
            range: netherwick_now::RangeSense::default(),
            body: BodySense::default(),
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

        assert_eq!(
            run.chosen.actions,
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

        assert_eq!(
            run.chosen.actions,
            vec![
                EventScriptAction::Stop,
                EventScriptAction::Rotate { deg: 180 },
                EventScriptAction::Go,
            ]
        );
        assert!(run.training_sample_emitted);
        assert_eq!(run.record.hardcoded_output, Some(run.chosen));
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

        assert_eq!(
            behavior.infer(&named, 10).unwrap().chosen.actions,
            vec![EventScriptAction::Say {
                text: "Hello Ada".to_string()
            }]
        );
        assert_eq!(
            behavior.infer(&unnamed, 10).unwrap().chosen.actions,
            vec![EventScriptAction::Say {
                text: "Hello Acquaintance p2".to_string()
            }]
        );
        assert_eq!(
            behavior.infer(&stranger, 10).unwrap().chosen.actions,
            vec![EventScriptAction::Say {
                text: "Hello Stranger p3".to_string()
            }]
        );
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
                chosen_action: Some(action),
                recall: RecallBundle::default(),
                llm: LlmTickResult::default(),
                combobulation: None,
                inline_learning: InlineLearningTickStatus::default(),
            })
        }
    }

    struct CountingBody {
        motor_attempts: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl RobotBody for CountingBody {
        async fn read_body(&mut self) -> Result<BodySense> {
            Ok(BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            })
        }

        async fn apply_motor(&mut self, _cmd: MotorCommand) -> Result<()> {
            self.motor_attempts.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[tokio::test]
    async fn real_robot_read_only_runner_never_applies_motor() {
        let motor_attempts = Arc::new(AtomicUsize::new(0));
        let body = CountingBody {
            motor_attempts: Arc::clone(&motor_attempts),
        };
        let mut runner =
            RealRobotRunner::new(RobotMode::ReadOnly, Box::new(body), Vec::new(), StubRuntime);

        let (_snapshot, tick) = runner.tick_read_only().await.unwrap();

        assert!(matches!(
            tick.chosen_action,
            Some(ActionPrimitive::Go { .. })
        ));
        assert_eq!(motor_attempts.load(Ordering::SeqCst), 0);
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
    async fn tick_adds_combobulated_experience() {
        let ledger = JsonlLedger::new("/tmp/netherwick-runtime-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            netherwick_llm::NoopLlmAgent,
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
    fn asr_impressions_phrase_partial_and_final_confidence_naturally() {
        let mut partial = Now::blank(100, BodySense::default());
        partial.ear.asr = netherwick_now::AsrSense {
            transcript: Some("come over here".to_string()),
            is_final: false,
            confidence: 0.52,
            ..netherwick_now::AsrSense::default()
        };
        let (_sensations, partial_impressions) = derive_direct_impressions_from_now(&partial);
        let partial_text = partial_impressions
            .iter()
            .find(|impression| impression.kind == "audio.transcript.impression")
            .map(|impression| impression.text.as_str())
            .unwrap();
        assert_eq!(partial_text, "I think I heard \"come over here\".");

        let mut final_now = Now::blank(100, BodySense::default());
        final_now.ear.asr = netherwick_now::AsrSense {
            transcript: Some("come over here".to_string()),
            is_final: true,
            confidence: 0.93,
            ..netherwick_now::AsrSense::default()
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
        let ledger = JsonlLedger::new("/tmp/netherwick-runtime-action-selector-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            netherwick_llm::NoopLlmAgent,
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
    async fn direct_reign_overrides_model_assisted_selector() {
        let ledger = JsonlLedger::new("/tmp/netherwick-runtime-reign-model-assisted-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            netherwick_llm::NoopLlmAgent,
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
        let ledger = JsonlLedger::new("/tmp/netherwick-runtime-assist-reign-model-assisted-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            netherwick_llm::NoopLlmAgent,
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
            let ledger = JsonlLedger::new(format!(
                "/tmp/netherwick-runtime-non-driving-reign-{mode:?}"
            ));
            let memory = InMemoryExperienceStore::new();
            let recall = memory.clone();
            let mut runtime = MinimalRuntime::new(
                ledger,
                memory,
                recall,
                FixedConductor::new(ActionPrimitive::Stop),
                SimpleSafety::default(),
                netherwick_llm::NoopLlmAgent,
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
        let ledger = JsonlLedger::new("/tmp/netherwick-runtime-reign-stop-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            netherwick_llm::NoopLlmAgent,
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
        let ledger = JsonlLedger::new("/tmp/netherwick-runtime-reign-safety-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            netherwick_llm::NoopLlmAgent,
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
        body.flags.cliff_left = true;
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
    }

    #[tokio::test]
    async fn sim_runner_writes_frames_and_transitions() {
        let root = test_ledger_root("sim-runner-writes");
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(ledger.clone(), SimpleConductor::default());
        let (world, motors) = VirtualWorld::new_with_motor(7, arena());
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

        assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
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
        assert_eq!(sequence.actions.len(), 3);
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
            surprise: SurpriseSense::default(),
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
        assert_eq!(action_value.expected.value, 0.25);
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

        assert_eq!(experience.record.behavior_id, "experience");
        assert_eq!(danger.record.behavior_id, "danger");
        assert_eq!(charge.record.behavior_id, "charge");
        assert_eq!(future.record.behavior_id, "future");
        assert_eq!(action_value.record.behavior_id, "action_value");
        assert_eq!(eye_next.record.behavior_id, "eye_next");
        assert_eq!(ear_next.record.behavior_id, "ear_next");
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
            checkpoint = "/tmp/netherwick-missing-action-value-checkpoint"
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
        let (mut world, motors) = VirtualWorld::new_with_motor(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.5, 7));
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(1).await.unwrap();
        let snapshot = runner.world.snapshot().await.unwrap();

        assert!(snapshot.body.odometry.x_m > 1.0);
        assert_eq!(runner.tick_count, 1);
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
        let (mut world, motors) = VirtualWorld::new_with_motor(7, arena());
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
        let (mut world, motors) = VirtualWorld::new_with_motor(7, arena());
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
        let (mut world, motors) = VirtualWorld::new_with_motor(7, arena());
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
        let (mut world, motors) = VirtualWorld::new_with_motor(7, arena());
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
        let (mut world, motors) = VirtualWorld::new_with_motor(7, arena());
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
        let (mut world, motors) = VirtualWorld::new_with_motor(7, arena());
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
        let (mut world, motors) = VirtualWorld::new_with_motor(7, arena());
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
        let (mut world, motors) = VirtualWorld::new_with_motor(7, arena());
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
        let (mut world, motors) = VirtualWorld::new_with_motor(7, arena());
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
        let (mut world, motors) = VirtualWorld::new_with_motor(7, arena());
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
        let (mut world, motors) = VirtualWorld::new_with_motor(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.8, 7));
        world.add_object(SimObject {
            id: "speaker".to_string(),
            label: "speaker".to_string(),
            kind: netherwick_sim::SimObjectKind::SoundSource {
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
        let (mut world, motors) = VirtualWorld::new_with_motor(7, arena());
        let mut body = test_body(1.0, 1.0, 0.8, 7);
        body.velocity.forward_m_s = 0.1;
        world.set_body(body);
        world.add_object(SimObject {
            id: "speaker".to_string(),
            label: "speaker".to_string(),
            kind: netherwick_sim::SimObjectKind::SoundSource {
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
    fn missing_experience_checkpoint_falls_back_to_feature_encoder() {
        let config: BehaviorRegistryConfig = toml::from_str(
            r#"
            [behavior.experience]
            regime = "shadow_infer"
            hardcoded = "experience.feature_encoder"
            model = "experience.autoencoder.v0"
            checkpoint = "/tmp/netherwick-missing-experience-checkpoint"
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
                direction: netherwick_actions::TurnDir::Left,
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
            netherwick_llm::NoopLlmAgent,
            queue,
        );
        let (mut world, motors) = VirtualWorld::new_with_motor(7, arena());
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
    async fn column_trap_scenario_recovers_within_budget() {
        let root = test_ledger_root("sim-runner-column-trap-recovery");
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(
            ledger,
            FixedConductor::new(ActionPrimitive::Go {
                intensity: 0.2,
                duration_ms: 1_000,
            }),
        );
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
            _z: &ExperienceLatent,
            _futures: &[FuturePrediction],
            _recall_summary: &str,
        ) -> Result<Option<Combobulation>> {
            Ok(None)
        }

        async fn maybe_tick(
            &mut self,
            _now: &Now,
            _z: &ExperienceLatent,
            _futures: &[FuturePrediction],
            _recall_summary: &str,
            _awareness_summary: Option<&str>,
        ) -> Result<LlmTickResult> {
            Ok(LlmTickResult {
                sense: netherwick_now::LlmSense {
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
        netherwick_llm::NoopLlmAgent,
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
            netherwick_llm::NoopLlmAgent,
        )
    }

    #[tokio::test]
    async fn llm_command_action_overrides_default_curiosity_drive() {
        let ledger = JsonlLedger::new("/tmp/netherwick-runtime-llm-command-action-test");
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
        let ledger = JsonlLedger::new("/tmp/netherwick-runtime-llm-reign-wins-test");
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
        let ledger = JsonlLedger::new("/tmp/netherwick-runtime-llm-safety-veto-test");
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
        assert!(!status.reset_due);
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
        detector.clearance_m = Some(0.8);
        while detector.recovery_motion().is_some() {}
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
    fn recovery_transition_runs_stop_reverse_turn_probe_then_releases() {
        let mut detector = StuckRecoveryController::default();
        detector.active = true;
        detector.corner_trap = true;
        detector.phase = RecoveryPhase::Stop;
        detector.phase_ticks_remaining = 1;
        detector.turn_sign = -1.0;
        detector.clearance_m = Some(0.8);

        assert_eq!(detector.recovery_motion(), Some(MotionCommand::Stop));
        assert_eq!(detector.phase, RecoveryPhase::Reverse);
        for _ in 0..6 {
            assert_eq!(
                detector.recovery_motion(),
                Some(MotionCommand::Forward { speed_m_s: -0.18 })
            );
        }
        assert_eq!(detector.phase, RecoveryPhase::Turn);
        for _ in 0..12 {
            assert_eq!(
                detector.recovery_motion(),
                Some(MotionCommand::Turn { turn_rad_s: -0.8 })
            );
        }
        assert_eq!(
            detector.recovery_motion(),
            Some(MotionCommand::Drive {
                forward_m_s: 0.04,
                turn_rad_s: -0.25
            })
        );

        assert!(!detector.status().active);
        assert!(detector.status().recovered);
        assert_eq!(detector.recovery_motion(), None);
    }

    #[test]
    fn failed_left_recovery_tries_right_next() {
        let mut detector = StuckRecoveryController {
            active: true,
            phase: RecoveryPhase::Stop,
            phase_ticks_remaining: 1,
            turn_sign: 1.0,
            clearance_m: Some(0.10),
            ..StuckRecoveryController::default()
        };

        while detector.phase != RecoveryPhase::Probe {
            assert!(detector.recovery_motion().is_some());
        }
        assert!(detector.recovery_motion().is_some());

        let status = detector.status();
        assert!(status.active);
        assert_eq!(status.turn_sign, -1.0);
        assert_eq!(status.recovery_attempts, 1);
    }

    #[test]
    fn recovery_does_not_resume_while_clearance_is_too_low() {
        let mut detector = StuckRecoveryController {
            active: true,
            phase: RecoveryPhase::Probe,
            phase_ticks_remaining: 1,
            turn_sign: -1.0,
            clearance_m: Some(0.12),
            ..StuckRecoveryController::default()
        };

        assert!(detector.recovery_motion().is_some());

        assert!(detector.status().active);
        assert_eq!(detector.status().phase, RecoveryPhase::Stop);
        assert!(!detector.status().recovered);
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

    fn test_ledger_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("netherwick-{name}-{}", Uuid::new_v4()));
        let _ = fs::remove_dir_all(&root);
        root
    }

    fn danger_checkpoint_root(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("netherwick-{name}-checkpoint-{}", Uuid::new_v4()));
        let _ = fs::remove_dir_all(&root);
        root
    }

    fn write_test_danger_checkpoint(root: &Path, action: ActionPrimitive) {
        let mut body = test_body(1.0, 1.0, 0.8, 7);
        body.velocity.forward_m_s = 0.05;
        let now = Now::blank(100, body);
        let mut encoder = FeatureExperienceEncoder::new();
        let latent = encoder.encode(&now).unwrap();
        let input = DangerInput::from_parts(latent.z, Some(&action), &now);
        let mut trainer = DangerNetTrainer::new(input.flat_features().len());
        trainer
            .train_step(
                &input,
                &netherwick_experience::DangerTarget {
                    bump: 0.2,
                    ..netherwick_experience::DangerTarget::default()
                },
            )
            .unwrap();
        trainer.save_checkpoint(root).unwrap();
    }

    fn write_test_charge_checkpoint(root: &Path, action: ActionPrimitive) {
        let mut body = test_body(1.0, 1.0, 0.2, 7);
        body.charging = false;
        let now = Now::blank(100, body);
        let mut encoder = FeatureExperienceEncoder::new();
        let latent = encoder.encode(&now).unwrap();
        let input = ChargeInput::from_parts(latent.z, Some(&action), &now);
        let mut trainer = ChargeNetTrainer::new(input.flat_features().len());
        trainer
            .train_step(
                &input,
                &netherwick_experience::ChargeTarget {
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
        let mut encoder = FeatureExperienceEncoder::new();
        let latent = encoder.encode(&now).unwrap();
        let input = ActionValueInput::from_parts(latent.z, Some(&action), &now);
        let mut trainer = ActionValueNetTrainer::new(input.flat_features().len());
        trainer
            .train_step(
                &input,
                &netherwick_experience::ActionValueTarget { value: 0.25 },
            )
            .unwrap();
        trainer.save_checkpoint(root).unwrap();
    }

    fn write_test_future_checkpoint(root: &Path, action: ActionPrimitive) {
        let now = Now::blank(100, test_body(1.0, 1.0, 0.8, 100));
        let mut encoder = FeatureExperienceEncoder::new();
        let latent = encoder.encode(&now).unwrap();
        let input = FutureInput {
            latent: latent.clone(),
            action,
            offset_ms: 100,
        };
        let mut trainer = FutureNetTrainer::new(input.flat_features().len(), input.latent.z.len());
        trainer.train_step(&input, &latent.z).unwrap();
        trainer.save_checkpoint(root).unwrap();
    }

    fn write_test_ear_next_checkpoint(root: &Path, action: ActionPrimitive) {
        let body = test_body(1.0, 1.0, 0.8, 7);
        let mut now = Now::blank(100, body);
        now.ear.features = vec![vec![0.2, 0.4, 0.6, 0.8]];
        let mut encoder = FeatureExperienceEncoder::new();
        let latent = encoder.encode(&now).unwrap();
        let input = EarNextInput::from_parts(latent.z, Some(&action), &now, 100);
        let mut trainer = EarNextNetTrainer::new(input.flat_features().len(), 4);
        trainer
            .train_step(
                &input,
                &netherwick_experience::EarNextTarget {
                    features: vec![0.2, 0.4, 0.6, 0.8],
                    ..netherwick_experience::EarNextTarget::default()
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
