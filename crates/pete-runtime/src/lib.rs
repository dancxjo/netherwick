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
    EntityId, EvidenceRef, ExtensionSense, EyePrediction, Freshness, ImuMotionContext, ImuSense,
    MemorySense, Now, ObjectClass, ObjectObservation, ObjectObservationSource, ReignSense,
    SafetySense, SemanticBehaviorId, SemanticEvidenceObservation, SemanticGroundingKind,
    SemanticNodeRef, SemanticOutcomeId, SemanticPredicate, SurpriseSense, WorldModelSnapshot,
    WorldModelUpdater,
};
use pete_sensors::{
    anticipate_surfaces, FrameProcessor, ImuArbiter, ImuCandidateMetadata, ImuSelection,
    ImuSourceOverride, NowBuilder, SenseProducer, SurfaceExtractor, SurfaceExtractorOutput, World,
    WorldSnapshot,
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

// Runtime domains share one namespace so this split does not change the public API.
include!("runtime/cognition.rs");
include!("runtime/policy.rs");
include!("runtime/models.rs");
include!("runtime/core.rs");
include!("runtime/real_robot.rs");
include!("runtime/simulation.rs");
include!("runtime/selection.rs");
include!("runtime/perception.rs");

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
