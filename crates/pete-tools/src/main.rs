use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fs;
use std::future::Future;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::Command as ProcessCommand;
use std::time::{Duration, Instant};

use anyhow::{Context, Error as AnyhowError, Result};
use chrono::Utc;
use clap::{Parser, Subcommand, ValueEnum};
use pete_actions::ActionPrimitive;
use pete_actions::{ApproachTarget, ExploreStyle, TurnDir};
use pete_autonomic::{SafetyLayer, SimpleSafety};
use pete_behaviors::{BehaviorConfig, BehaviorRegime, ErasedBehaviorRunRecord, FallbackPolicy};
use pete_body::{BodySense, BodySong, BodyTone};
use pete_cockpit::{
    establish_diagnostic_session, establish_session, Cockpit, CockpitError, CockpitEventKind,
    HandshakeHello, HttpCockpit, MotherbrainPossession, MotionCommand, SafeCockpit,
    SafetyLatchKind, SimCockpit as LocalSimCockpit, SongTone, UartCockpit, UdpCockpit,
};
use pete_conductor::{
    Conductor, ConductorInput, GoalProgressReport, SimpleConductor, StrategyProgressResponse,
};
use pete_ledger::{ExperienceFrame, ExperienceTransition, JsonlLedger, LedgerReader, LedgerWriter};
use pete_llm::{ConfiguredLlmAgent, LiveImageEnricher, LlmConfig, LlmProvider};
use pete_map::{
    observation_from_now, transform_point_to_world, LocalMap, LoopClosureCandidateInput,
    OrientationEstimate, Point3D, PointCloudConfig, PointCloudFrame, PoseEdgeSource,
    PoseGraphBuilder, PoseGraphConfig, PoseGraphReport, VoxelPointCloud,
};
use pete_memory::{
    deterministic_embodied_eval_report_with_omissions, place_memory_report_from_frames,
    place_recognition_input_from_frame, place_recognition_vectors_from_input, BindingRelation,
    DurableExperienceStore, EmbodiedEvalOmission, EntityConstellationState, EntityMemory,
    InMemoryExperienceStore, PlaceMemory, PlaceMemoryReport, PlaceRecognitionCandidate,
    PlaceRecognitionKind,
};
use pete_models::MODEL_REGISTRY;
use pete_mouth::QueuedPiperCpalMouth;
use pete_neat::{
    evaluate_locomotion_promotion, verify_locomotion_promotion_artifacts, CandidateEvaluation,
    CurriculumStage, EpisodeMetrics as NeatEpisodeMetrics, FitnessTraits, Genome, GenomeState,
    LocomotionCheckpoint, LocomotionOutput, LocomotionPromotionEvidence, LocomotionPromotionPolicy,
    LocomotionTracker, NeatConfig, NicheQualificationEvidence, NoveltyArchive, Population,
    QualityDiversityDescriptor, QualityDiversityEntry, SelectionSummary,
};
use pete_now::{
    EarSense, ExtensionSense, KinectSense, Now, RangeExtrinsics, RangeSense, SurpriseSense,
};
use pete_runtime::{
    body_sense_from_cockpit_status, ActionSelectionDecision, ActionSelectorMode,
    InlineLearningBehaviors, InlineLearningConfig, InlineLearningMode, MinimalRuntime, NudgePolicy,
    RealRobotRunner, ReignQueue, RobotMode, RuntimeLoop, RuntimeModelStack, RuntimeTick, SimRunner,
};
use pete_sensors::{
    AsrToolConfig, CameraSenseProvider, DepthRangeProjectionConfig, EyeFrame, EyeFrameFormat,
    FrameProcessor, GpsSenseProvider, ImuSenseProvider, ImuSourceOverride, Lfcd2SenseProvider,
    MicrophoneSenseProvider, PcmAudioFrame, SensePacket, SenseProducer, World, WorldSnapshot,
};
#[cfg(feature = "kinect-freenect")]
use pete_sensors::{FreenectKinectProvider, KinectRgbAdjustment};
use pete_server::{
    LiveSceneMetadata, LiveViewState, SceneArena, SceneObject, SceneSensorCalibration, SceneSession,
};
use pete_sim::{build_scenario, ScenarioConfig, ScenarioKind, SimObjectKind};
use pete_training::dream_policy::{
    load_best_genome, train_dream_policy, DreamLevel, DreamTrainingConfig,
};
use pete_training::{
    evaluate_behavior, load_models_config, promote_behavior_config, train_behavior,
    train_latent_round_trip, train_unified_experience, write_models_config,
    EvaluateBehaviorRequest, TrainBehaviorRequest, TrainLatentRoundTripRequest,
    TrainUnifiedExperienceRequest, TrainableBehavior,
};
use pete_worldlab::{
    export_pointcloud_for_frame, rewrite_frames, update_manifest, CaptureExportContext,
    CaptureReader, CaptureReplayRunner, CaptureSource, CaptureStreams, CaptureWriter,
};
use rand::{prelude::SliceRandom, rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// Command domains share this binary namespace to preserve CLI behavior.
include!("commands/cli.rs");
include!("commands/simulation.rs");
include!("commands/neat_state.rs");
include!("commands/neat_training.rs");
include!("commands/neat_evaluation.rs");
include!("commands/scenario.rs");
include!("commands/capture.rs");
include!("commands/possession.rs");
include!("commands/hardware.rs");
include!("commands/models.rs");
include!("commands/reports.rs");
include!("commands/vision.rs");
include!("commands/shadow_flight.rs");
include!("commands/shadow_score.rs");

#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
