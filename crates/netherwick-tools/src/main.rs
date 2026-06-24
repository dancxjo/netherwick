use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::time::Duration;

use anyhow::{Context, Result};
use chrono::Utc;
use clap::{Parser, Subcommand, ValueEnum};
use netherwick_actions::ActionPrimitive;
use netherwick_actions::{ApproachTarget, ExploreStyle, TurnDir};
use netherwick_autonomic::SimpleSafety;
use netherwick_behaviors::{BehaviorRegime, ErasedBehaviorRunRecord};
use netherwick_body::RobotBody;
use netherwick_conductor::{Conductor, ConductorInput, SimpleConductor};
use netherwick_create1::{Create1Body, MockCreate1Body};
use netherwick_ledger::{
    ExperienceFrame, ExperienceTransition, JsonlLedger, LedgerReader, LedgerWriter,
};
use netherwick_llm::{ConfiguredLlmAgent, LlmConfig, LlmProvider};
use netherwick_memory::{
    place_memory_report_from_frames, InMemoryExperienceStore, PlaceMemoryReport,
};
use netherwick_models::MODEL_REGISTRY;
use netherwick_now::{EarSense, KinectSense, RangeSense};
use netherwick_runtime::{
    ActionSelectionDecision, ActionSelectorMode, InlineLearningBehaviors, InlineLearningConfig,
    InlineLearningMode, MinimalRuntime, RealRobotRunner, RobotMode, RuntimeLoop, RuntimeModelStack,
    RuntimeTick, SimRunner,
};
use netherwick_sensors::{
    CameraSenseProvider, EyeFrame, EyeFrameFormat, GpsSenseProvider, ImuSenseProvider,
    MicrophoneSenseProvider, PcmAudioFrame, SensePacket, SenseProducer, WorldSnapshot,
};
use netherwick_server::{
    LiveSceneMetadata, LiveViewState, SceneArena, SceneObject, SceneSensorCalibration, SceneSession,
};
use netherwick_sim::{build_scenario, ScenarioConfig, ScenarioKind, SimObjectKind};
use netherwick_training::{
    evaluate_behavior, load_models_config, promote_behavior_config, train_behavior,
    EvaluateBehaviorRequest, TrainBehaviorRequest, TrainableBehavior,
};
use netherwick_worldlab::{
    export_pointcloud_for_frame, export_snapshot_assets, rewrite_frames, update_manifest,
    CaptureReader, CaptureReplayRunner, CaptureSource, CaptureStreams, CaptureWriter,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Parser)]
#[command(name = "netherwick")]
#[command(about = "Netherwick CLI entrypoint")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Sim(SimArgs),
    SimCurriculum(SimCurriculumArgs),
    EvalScenario(EvalScenarioArgs),
    MemoryInspect(MemoryInspectArgs),
    Robot(RobotArgs),
    HardwareEnv(HardwareEnvArgs),
    Replay,
    CaptureSim(CaptureSimArgs),
    CaptureReal(CaptureRealArgs),
    CaptureAssets(CaptureAssetsArgs),
    InspectCapture(InspectCaptureArgs),
    ReplayCapture(ReplayCaptureArgs),
    ReplayCounterfactual(ReplayCounterfactualArgs),
    Train(TrainCommand),
    Evaluate(EvaluateCommand),
    Promote(PromoteCommand),
    InspectLedger(InspectLedgerArgs),
    ModelRegister(ModelRegisterArgs),
    ModelStatus,
    ModelPromote(ModelPromoteArgs),
    CompareScenarioReports(CompareScenarioReportsArgs),
    Dashboard,
    VirtualReport(VirtualReportArgs),
    RetinaMockSend(RetinaMockSendArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    let cli = Cli::parse();
    match cli.command {
        Command::Sim(args) => run_sim(args).await,
        Command::SimCurriculum(args) => run_sim_curriculum(args).await,
        Command::EvalScenario(args) => run_eval_scenario(args).await,
        Command::MemoryInspect(args) => memory_inspect(args).await,
        Command::Robot(args) => run_robot(args).await,
        Command::HardwareEnv(args) => hardware_env(args).await,
        Command::CaptureSim(args) => capture_sim(args).await,
        Command::CaptureReal(args) => capture_real(args).await,
        Command::CaptureAssets(args) => capture_assets(args).await,
        Command::InspectCapture(args) => inspect_capture(args).await,
        Command::ReplayCapture(args) => replay_capture(args).await,
        Command::ReplayCounterfactual(args) => replay_counterfactual(args).await,
        Command::InspectLedger(args) => inspect_ledger(args).await,
        Command::Train(command) => run_train(command).await,
        Command::Evaluate(command) => run_evaluate(command).await,
        Command::Promote(command) => run_promote(command),
        Command::ModelRegister(args) => model_register(args),
        Command::ModelStatus => model_status(),
        Command::ModelPromote(args) => model_promote(args),
        Command::CompareScenarioReports(args) => compare_scenario_reports_command(args),
        Command::VirtualReport(args) => run_virtual_report(args).await,
        Command::RetinaMockSend(args) => run_retina_mock_send(args).await,
        other => {
            println!("selected command: {:?}", other);
            Ok(())
        }
    }
}

#[derive(Debug, Parser)]
struct SimArgs {
    #[arg(long, default_value_t = 50)]
    steps: usize,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long, value_enum, default_value = "mixed-room")]
    scenario: ScenarioArg,
    #[arg(long, default_value = "data/ledger")]
    ledger: String,
    #[arg(long)]
    danger_checkpoint: Option<String>,
    #[arg(long, value_enum, default_value = "off")]
    danger_mode: DangerMode,
    #[arg(long)]
    charge_checkpoint: Option<String>,
    #[arg(long, value_enum, default_value = "off")]
    charge_mode: ChargeMode,
    #[arg(long)]
    action_value_checkpoint: Option<String>,
    #[arg(long, value_enum, default_value = "off")]
    action_value_mode: ActionValueMode,
    #[arg(long)]
    future_checkpoint: Option<String>,
    #[arg(long, value_enum, default_value = "hardcoded")]
    future_mode: FutureMode,
    #[arg(long)]
    eye_next_checkpoint: Option<String>,
    #[arg(long, value_enum, default_value = "off")]
    eye_next_mode: EyeNextMode,
    #[arg(long)]
    ear_next_checkpoint: Option<String>,
    #[arg(long, value_enum, default_value = "off")]
    ear_next_mode: EarNextMode,
    #[arg(long)]
    experience_checkpoint: Option<String>,
    #[arg(long, value_enum, default_value = "off")]
    experience_mode: ExperienceMode,
    #[arg(long, value_enum, default_value = "baseline")]
    action_selector: CliActionSelectorMode,
    #[arg(long, env = "NETHERWICK_INLINE_LEARNING")]
    inline_learning: bool,
    #[arg(
        long,
        value_enum,
        default_value = "off",
        env = "NETHERWICK_INLINE_LEARNING_MODE"
    )]
    inline_learning_mode: InlineLearningModeArg,
    #[arg(
        long,
        default_value_t = 1,
        env = "NETHERWICK_INLINE_TRAIN_STEPS_PER_TICK"
    )]
    inline_train_steps_per_tick: usize,
    #[arg(long, env = "NETHERWICK_INLINE_BEHAVIORS")]
    inline_behaviors: Option<String>,
    #[arg(long)]
    live: bool,
    #[arg(long, default_value = "127.0.0.1:8787")]
    live_addr: SocketAddr,
    #[arg(long)]
    live_tls: bool,
    #[arg(long, default_value = "certs/netherwick-dev.crt")]
    live_tls_cert: String,
    #[arg(long, default_value = "certs/netherwick-dev.key")]
    live_tls_key: String,
    #[arg(long, default_value_t = 100)]
    tick_delay_ms: u64,
    #[command(flatten)]
    llm: LlmArgs,
}

#[derive(Debug, Parser)]
struct SimCurriculumArgs {
    #[arg(long, value_enum)]
    scenario: ScenarioArg,
    #[arg(long, default_value_t = 20)]
    episodes: usize,
    #[arg(long, default_value_t = 300)]
    steps: usize,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long, default_value = "data/ledger/curriculum")]
    out: String,
    #[arg(long)]
    capture_root: Option<String>,
    #[arg(long, default_value_t = 100)]
    tick_ms: u64,
    #[arg(long, default_value_t = 0.1)]
    validation_ratio: f32,
    #[arg(long, default_value_t = 0.1)]
    test_ratio: f32,
    #[command(flatten)]
    llm: LlmArgs,
}

#[derive(Debug, Parser)]
struct EvalScenarioArgs {
    #[arg(long, value_enum)]
    scenario: ScenarioArg,
    #[arg(long, default_value_t = 20)]
    episodes: usize,
    #[arg(long, default_value_t = 300)]
    steps: usize,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long, default_value_t = 100)]
    tick_ms: u64,
    #[arg(long)]
    out: Option<String>,
    #[arg(long)]
    ledger: Option<String>,
    #[arg(long)]
    capture_root: Option<String>,
    #[arg(long)]
    memory_report: bool,
    #[arg(long)]
    danger_checkpoint: Option<String>,
    #[arg(long, value_enum, default_value = "off")]
    danger_mode: DangerMode,
    #[arg(long)]
    charge_checkpoint: Option<String>,
    #[arg(long, value_enum, default_value = "off")]
    charge_mode: ChargeMode,
    #[arg(long)]
    action_value_checkpoint: Option<String>,
    #[arg(long, value_enum, default_value = "off")]
    action_value_mode: ActionValueMode,
    #[arg(long)]
    future_checkpoint: Option<String>,
    #[arg(long, value_enum, default_value = "hardcoded")]
    future_mode: FutureMode,
    #[arg(long)]
    eye_next_checkpoint: Option<String>,
    #[arg(long, value_enum, default_value = "off")]
    eye_next_mode: EyeNextMode,
    #[arg(long)]
    ear_next_checkpoint: Option<String>,
    #[arg(long, value_enum, default_value = "off")]
    ear_next_mode: EarNextMode,
    #[arg(long)]
    experience_checkpoint: Option<String>,
    #[arg(long, value_enum, default_value = "off")]
    experience_mode: ExperienceMode,
    #[arg(long, value_enum, default_value = "baseline")]
    action_selector: CliActionSelectorMode,
    #[command(flatten)]
    llm: LlmArgs,
}

#[derive(Debug, Parser)]
struct MemoryInspectArgs {
    #[arg(long)]
    ledger: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CliActionSelectorMode {
    Baseline,
    Random,
    ModelAssisted,
    Scripted,
}

impl From<CliActionSelectorMode> for ActionSelectorMode {
    fn from(value: CliActionSelectorMode) -> Self {
        match value {
            CliActionSelectorMode::Baseline => ActionSelectorMode::Baseline,
            CliActionSelectorMode::Random => ActionSelectorMode::Random,
            CliActionSelectorMode::ModelAssisted => ActionSelectorMode::ModelAssisted,
            CliActionSelectorMode::Scripted => ActionSelectorMode::Scripted,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum InlineLearningModeArg {
    Off,
    ShadowOnly,
    WorldOutcome,
}

impl From<InlineLearningModeArg> for InlineLearningMode {
    fn from(value: InlineLearningModeArg) -> Self {
        match value {
            InlineLearningModeArg::Off => InlineLearningMode::Off,
            InlineLearningModeArg::ShadowOnly => InlineLearningMode::ShadowOnly,
            InlineLearningModeArg::WorldOutcome => InlineLearningMode::WorldOutcome,
        }
    }
}

#[derive(Clone, Debug, Default, Parser)]
struct LlmArgs {
    #[arg(long)]
    llm_config: Option<String>,
    #[arg(long, value_enum)]
    llm_provider: Option<CliLlmProvider>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CliLlmProvider {
    Disabled,
    Ollama,
}

impl From<CliLlmProvider> for LlmProvider {
    fn from(value: CliLlmProvider) -> Self {
        match value {
            CliLlmProvider::Disabled => LlmProvider::Disabled,
            CliLlmProvider::Ollama => LlmProvider::Ollama,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ScenarioArg {
    EmptyRoom,
    ObstacleAvoidance,
    CornerTrap,
    ColumnTrap,
    ChargerSeeking,
    PersonSpeakerRoom,
    MixedRoom,
}

impl From<ScenarioArg> for ScenarioKind {
    fn from(value: ScenarioArg) -> Self {
        match value {
            ScenarioArg::EmptyRoom => ScenarioKind::EmptyRoom,
            ScenarioArg::ObstacleAvoidance => ScenarioKind::ObstacleAvoidance,
            ScenarioArg::CornerTrap => ScenarioKind::CornerTrap,
            ScenarioArg::ColumnTrap => ScenarioKind::ColumnTrap,
            ScenarioArg::ChargerSeeking => ScenarioKind::ChargerSeeking,
            ScenarioArg::PersonSpeakerRoom => ScenarioKind::PersonAndSpeaker,
            ScenarioArg::MixedRoom => ScenarioKind::MixedRoom,
        }
    }
}

#[derive(Debug, Parser)]
struct RobotArgs {
    #[arg(long, value_enum, default_value = "read-only")]
    mode: RobotModeArg,
    #[arg(long, default_value = "auto")]
    create_port: String,
    #[arg(long, default_value_t = 57_600)]
    create_baud: u32,
    #[arg(long, default_value = "data/ledger/robot-readonly")]
    ledger: String,
    #[arg(long)]
    camera: Option<String>,
    #[arg(long)]
    mic: Option<String>,
    #[arg(long)]
    imu: Option<String>,
    #[arg(long)]
    gps: Option<String>,
    #[arg(long)]
    capture: Option<String>,
    #[arg(long)]
    dashboard: Option<SocketAddr>,
    #[arg(long, default_value_t = 100)]
    tick_ms: u64,
    #[arg(long)]
    steps: Option<usize>,
    #[arg(long)]
    duration_seconds: Option<u64>,
    #[arg(long)]
    require_camera: bool,
    #[arg(long)]
    require_mic: bool,
    #[arg(long)]
    require_imu: bool,
    #[arg(long)]
    require_gps: bool,
    #[command(flatten)]
    llm: LlmArgs,
}

#[derive(Debug, Parser)]
struct HardwareEnvArgs {
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum RobotModeArg {
    ReadOnly,
    Slow,
    Disabled,
}

#[derive(Debug, Parser)]
struct CaptureSimArgs {
    #[arg(long, default_value = "data/captures/sim-test")]
    out: String,
    #[arg(long, default_value_t = 100)]
    steps: usize,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long, default_value_t = 100)]
    tick_ms: u64,
    #[command(flatten)]
    llm: LlmArgs,
}

#[derive(Debug, Parser)]
struct CaptureRealArgs {
    #[arg(long, default_value_t = 60)]
    duration_seconds: u64,
    #[arg(long, default_value = "data/captures/real/rpi5-smoke")]
    out: String,
    #[arg(long)]
    ledger: Option<String>,
    #[arg(long, default_value_t = 100)]
    tick_ms: u64,
    #[arg(long, default_value = "auto")]
    create_port: String,
    #[arg(long, default_value_t = 57_600)]
    create_baud: u32,
    #[arg(long)]
    camera: Option<String>,
    #[arg(long)]
    mic: Option<String>,
    #[arg(long)]
    imu: Option<String>,
    #[arg(long)]
    gps: Option<String>,
    #[arg(long)]
    mock: bool,
    #[arg(long)]
    export_rgb: bool,
    #[arg(long)]
    export_depth: bool,
    #[arg(long)]
    export_audio: bool,
    #[arg(long)]
    export_pointcloud: bool,
    #[arg(long, default_value_t = 4)]
    pointcloud_stride: usize,
    #[command(flatten)]
    llm: LlmArgs,
}

#[derive(Debug, Parser)]
struct CaptureAssetsArgs {
    #[arg(long)]
    capture: String,
    #[arg(long)]
    pointcloud: bool,
    #[arg(long, default_value_t = 4)]
    stride: usize,
    #[arg(long, default_value_t = 8.0)]
    max_depth_m: f32,
}

#[derive(Debug, Parser)]
struct InspectCaptureArgs {
    path: String,
}

#[derive(Debug, Parser)]
struct ReplayCaptureArgs {
    #[arg(long)]
    capture: String,
    #[arg(long, default_value = "data/ledger/replay-test")]
    ledger: String,
    #[command(flatten)]
    llm: LlmArgs,
}

#[derive(Debug, Parser)]
struct ReplayCounterfactualArgs {
    #[arg(long)]
    capture: String,
    #[arg(long)]
    edit: Vec<String>,
    #[arg(long, default_value = "baseline")]
    policy: String,
    #[arg(long)]
    actions: Option<String>,
    #[arg(long)]
    steps: Option<usize>,
    #[arg(long)]
    out_ledger: Option<String>,
    #[arg(long)]
    out_report: Option<String>,
    #[command(flatten)]
    llm: LlmArgs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DangerMode {
    Off,
    ShadowInfer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ChargeMode {
    Off,
    ShadowInfer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ActionValueMode {
    Off,
    ShadowInfer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum FutureMode {
    Hardcoded,
    ShadowInfer,
    ModelInfer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum EyeNextMode {
    Off,
    ShadowInfer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum EarNextMode {
    Off,
    ShadowInfer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ExperienceMode {
    Off,
    ShadowInfer,
    ModelInfer,
}

#[derive(Debug, Parser)]
struct TrainCommand {
    #[command(subcommand)]
    model: TrainModel,
}

#[derive(Debug, Subcommand)]
enum TrainModel {
    Behavior(TrainBehaviorArgs),
    Danger(TrainDangerArgs),
    Charge(TrainChargeArgs),
    ActionValue(TrainActionValueArgs),
    Future(TrainFutureArgs),
    EyeNext(TrainEyeNextArgs),
    EarNext(TrainEarNextArgs),
    Experience(TrainExperienceArgs),
    Virtual(TrainVirtualArgs),
}

#[derive(Debug, Parser)]
struct TrainBehaviorArgs {
    behavior: String,
    #[arg(long, default_value = "data/ledger")]
    ledger: String,
    #[arg(long, default_value_t = 5)]
    epochs: usize,
    #[arg(long)]
    checkpoint: Option<String>,
    #[arg(long, default_value_t = 0.2)]
    validation_split: f32,
    #[arg(long, default_value_t = 7)]
    seed: u64,
}

#[derive(Debug, Parser)]
struct TrainDangerArgs {
    #[arg(long, default_value = "data/ledger")]
    ledger: String,
    #[arg(long, default_value_t = 5)]
    epochs: usize,
    #[arg(long, default_value = "data/models/danger_v0")]
    checkpoint: String,
}

#[derive(Debug, Parser)]
struct TrainChargeArgs {
    #[arg(long, default_value = "data/ledger")]
    ledger: String,
    #[arg(long, default_value_t = 5)]
    epochs: usize,
    #[arg(long, default_value = "data/models/charge_v0")]
    checkpoint: String,
}

#[derive(Debug, Parser)]
struct TrainActionValueArgs {
    #[arg(long, default_value = "data/ledger")]
    ledger: String,
    #[arg(long, default_value_t = 5)]
    epochs: usize,
    #[arg(long, default_value = "data/models/action_value_v0")]
    checkpoint: String,
}

#[derive(Debug, Parser)]
struct TrainFutureArgs {
    #[arg(long, default_value = "data/ledger")]
    ledger: String,
    #[arg(long, default_value_t = 5)]
    epochs: usize,
    #[arg(long, default_value = "data/models/future_v0")]
    checkpoint: String,
}

#[derive(Debug, Parser)]
struct TrainEyeNextArgs {
    #[arg(long, default_value = "data/ledger")]
    ledger: String,
    #[arg(long, default_value_t = 5)]
    epochs: usize,
    #[arg(long, default_value = "data/models/eye_next_v0")]
    checkpoint: String,
}

#[derive(Debug, Parser)]
struct TrainEarNextArgs {
    #[arg(long, default_value = "data/ledger")]
    ledger: String,
    #[arg(long, default_value_t = 5)]
    epochs: usize,
    #[arg(long, default_value = "data/models/ear_next_v0")]
    checkpoint: String,
}

#[derive(Debug, Parser)]
struct TrainExperienceArgs {
    #[arg(long, default_value = "data/ledger")]
    ledger: String,
    #[arg(long, default_value_t = 5)]
    epochs: usize,
    #[arg(long, default_value = "data/models/experience_v0")]
    checkpoint: String,
}

#[derive(Debug, Parser)]
struct InspectLedgerArgs {
    #[arg(
        long,
        default_value = "data/ledger/virtual-live",
        env = "NETHERWICK_LEDGER"
    )]
    ledger: String,
}

#[derive(Debug, Parser)]
struct VirtualReportArgs {
    #[arg(
        long,
        default_value = "data/ledger/virtual-live",
        env = "NETHERWICK_LEDGER"
    )]
    ledger: String,
    #[arg(long, default_value = "data/reports/virtual/latest.json")]
    out: String,
}

#[derive(Debug, Parser)]
struct TrainVirtualArgs {
    #[arg(
        long,
        default_value = "data/ledger/virtual-live",
        env = "NETHERWICK_LEDGER"
    )]
    ledger: String,
    #[arg(
        long,
        default_value = "data/models/virtual/latest",
        env = "NETHERWICK_MODEL_OUT"
    )]
    out_dir: String,
    #[arg(long, default_value = "data/reports/virtual/latest.json")]
    report_out: String,
    #[arg(long, default_value_t = 5, env = "NETHERWICK_EPOCHS")]
    epochs: usize,
    #[arg(long)]
    allow_safety_critical_inference: bool,
}

#[derive(Debug, Parser)]
struct EvaluateCommand {
    #[command(subcommand)]
    model: EvaluateModel,
}

#[derive(Debug, Subcommand)]
enum EvaluateModel {
    Behavior(EvaluateBehaviorArgs),
}

#[derive(Debug, Parser)]
struct EvaluateBehaviorArgs {
    behavior: String,
    #[arg(long, default_value = "data/ledger")]
    ledger: String,
    #[arg(long)]
    checkpoint: Option<String>,
    #[arg(long)]
    max_samples: Option<usize>,
    #[arg(long)]
    out: Option<String>,
}

#[derive(Debug, Parser)]
struct PromoteCommand {
    #[command(subcommand)]
    model: PromoteModel,
}

#[derive(Debug, Subcommand)]
enum PromoteModel {
    Behavior(PromoteBehaviorArgs),
}

#[derive(Debug, Parser)]
struct PromoteBehaviorArgs {
    behavior: String,
    #[arg(long)]
    checkpoint: Option<String>,
    #[arg(long, default_value = "configs/models.toml")]
    config: String,
    #[arg(long, value_enum)]
    mode: PromoteMode,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PromoteMode {
    ShadowInfer,
    ModelInfer,
    ShadowTrain,
}

#[derive(Debug, Parser)]
struct ModelRegisterArgs {
    #[arg(long)]
    behavior: String,
    #[arg(long)]
    checkpoint: String,
    #[arg(long)]
    training_ledger: Option<String>,
    #[arg(long)]
    training_command: Option<String>,
    #[arg(long)]
    behavior_report: Option<String>,
    #[arg(long)]
    scenario_report: Option<String>,
    #[arg(long)]
    name: String,
    #[arg(long)]
    notes: Vec<String>,
    #[arg(long)]
    parent: Option<String>,
    #[arg(long, default_value = "data/models/registry.json")]
    registry: String,
    #[arg(long)]
    overwrite: bool,
}

#[derive(Debug, Parser)]
struct ModelPromoteArgs {
    #[arg(long)]
    behavior: String,
    #[arg(long)]
    name: String,
    #[arg(long, value_enum)]
    target: ModelStatus,
    #[arg(long)]
    baseline_report: Option<String>,
    #[arg(long)]
    candidate_report: Option<String>,
    #[arg(long, default_value = "data/models/registry.json")]
    registry: String,
    #[arg(long)]
    allow_safety_critical_inference: bool,
    #[arg(long)]
    notes: Vec<String>,
}

#[derive(Debug, Parser)]
struct CompareScenarioReportsArgs {
    #[arg(long)]
    baseline: String,
    #[arg(long)]
    candidate: String,
}

async fn run_sim(args: SimArgs) -> Result<()> {
    let ledger = JsonlLedger::new(&args.ledger);
    let flags = RuntimeModelFlags::from(&args);
    let (models, model_loading) = load_runtime_models_from_flags(&flags)?;
    let action_selector_mode = ActionSelectorMode::from(args.action_selector);
    let inline_learning = inline_learning_config_from_sim_args(&args)?;
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let llm = configured_llm_agent(&args.llm)?;
    let mut runtime = MinimalRuntime::with_default_events(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        llm,
    )
    .with_action_selector_mode(action_selector_mode)
    .with_inline_learning(inline_learning.clone());
    if let Some(models) = models {
        runtime = runtime.with_models(models);
    }

    let scenario_kind: ScenarioKind = args.scenario.into();
    let scenario = build_scenario(ScenarioConfig::new(scenario_kind, args.seed));
    let live_metadata = live_scene_metadata_from_scenario(&scenario.metadata);
    let world = scenario.world;
    let motors = scenario.motors;
    let mut runner = SimRunner::new(runtime, world, motors);
    if args.live {
        let live_state = LiveViewState::new().with_virtual_retina(true);
        live_state.update_inline_learning(inline_learning.clone());
        live_state.update_scene_metadata(live_metadata);
        live_state.update_session(SceneSession {
            mode: "virtual-live".to_string(),
            scenario: Some(scenario_kind.slug().to_string()),
            seed: Some(args.seed),
            source: "sim".to_string(),
            tick_ms: Some(args.tick_delay_ms),
        });
        live_state.update_training_status(netherwick_server::LiveTrainingStatus {
            training_mode: inline_learning.training_mode_label().to_string(),
            ledger_path: Some(args.ledger.clone()),
            frames_written: 0,
            transitions_written: 0,
            models_loaded: loaded_model_names(&model_loading),
            model_modes: model_modes_from_flags(&flags),
            action_selector_mode: action_selector_mode.as_str().to_string(),
            weights_updating: inline_learning.is_enabled(),
        });
        let server_state = live_state.clone();
        let live_addr = args.live_addr;
        if args.live_tls {
            let cert_path = args.live_tls_cert.clone();
            let key_path = args.live_tls_key.clone();
            tokio::spawn(async move {
                if let Err(error) = netherwick_server::serve_live_view_tls(
                    live_addr,
                    server_state,
                    cert_path,
                    key_path,
                )
                .await
                {
                    eprintln!("live robot HTTPS view server stopped: {error}");
                }
            });
        } else {
            tokio::spawn(async move {
                if let Err(error) =
                    netherwick_server::serve_live_view(live_addr, server_state).await
                {
                    eprintln!("live robot view server stopped: {error}");
                }
            });
        }
        let scheme = if args.live_tls { "https" } else { "http" };
        println!();
        println!("Netherwick virtual theater is running.");
        if inline_learning.is_enabled() {
            println!(
                "Virtual training theater is collecting experience and running {} inline learning.",
                inline_learning.mode.as_str()
            );
        } else {
            println!("Virtual training theater is collecting experience.");
            println!("Models are not updated online in this run.");
            println!("Train later with `cargo run --bin netherwick -- train behavior ...`");
        }
        println!();
        println!("Desktop:");
        println!("  {scheme}://127.0.0.1:{}/view/3d", args.live_addr.port());
        println!();
        println!("Bound address:");
        println!("  {scheme}://{}/view/3d", args.live_addr);
        println!();
        println!("Scene JSON:");
        println!("  {scheme}://{}/view/scene", args.live_addr);
        if args.live_tls {
            println!();
            println!("If your headset warns about the certificate, trust the local dev certificate or install the generated CA/cert.");
            println!("This serves robot/sim sensor data on the LAN. Use only on trusted networks.");
        }
        for _ in 0..args.steps {
            let current_inline_learning = live_state.inline_learning();
            runner.runtime.inline_learning = current_inline_learning.clone();
            let eye_frame = live_state.take_pending_retina_frame();
            if let Some(mut frame) = eye_frame {
                frame.source = Some("babylon-robot-eye".to_string());
                runner.world.set_retina_frame(Some(frame));
                live_state.record_ledger_write();
            } else {
                runner.world.set_retina_frame(None);
            }
            runner
                .run_steps_observing(1, |snapshot| live_state.update(snapshot.clone()))
                .await?;
            live_state.update_training_status(netherwick_server::LiveTrainingStatus {
                training_mode: current_inline_learning.training_mode_label().to_string(),
                ledger_path: Some(args.ledger.clone()),
                frames_written: runner.tick_count,
                transitions_written: runner.tick_count.saturating_sub(1),
                models_loaded: loaded_model_names(&model_loading),
                model_modes: model_modes_from_flags(&flags),
                action_selector_mode: action_selector_mode.as_str().to_string(),
                weights_updating: current_inline_learning.is_enabled(),
            });
            tokio::time::sleep(Duration::from_millis(args.tick_delay_ms)).await;
        }
    } else {
        runner.run_steps(args.steps).await?;
    }
    println!(
        "sim complete: {} ticks, seed {}, ledger {}, action_selector {:?}, danger_mode {:?}, charge_mode {:?}, action_value_mode {:?}, eye_next_mode {:?}, ear_next_mode {:?}, experience_mode {:?}",
        runner.tick_count,
        args.seed,
        args.ledger,
        args.action_selector,
        args.danger_mode,
        args.charge_mode,
        args.action_value_mode,
        args.eye_next_mode,
        args.ear_next_mode,
        args.experience_mode
    );
    Ok(())
}

fn loaded_model_names(report: &RuntimeModelLoadReport) -> Vec<String> {
    let mut names = report
        .loaded_checkpoints
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    names.sort();
    names
}

fn inline_learning_config_from_sim_args(args: &SimArgs) -> Result<InlineLearningConfig> {
    let mut mode = InlineLearningMode::from(args.inline_learning_mode);
    if args.inline_learning && mode == InlineLearningMode::Off {
        mode = InlineLearningMode::WorldOutcome;
    }
    Ok(InlineLearningConfig {
        mode,
        behaviors: inline_learning_behaviors(args.inline_behaviors.as_deref())?,
        max_train_steps_per_tick: args.inline_train_steps_per_tick,
    })
}

fn inline_learning_behaviors(list: Option<&str>) -> Result<InlineLearningBehaviors> {
    let Some(list) = list else {
        return Ok(InlineLearningBehaviors::default());
    };
    if list.trim().is_empty() || list.trim().eq_ignore_ascii_case("all") {
        return Ok(InlineLearningBehaviors::default());
    }
    let mut behaviors = InlineLearningBehaviors {
        danger: false,
        charge: false,
        future: false,
        action_value: false,
        eye_next: false,
        ear_next: false,
        experience: false,
    };
    for raw in list.split(',') {
        match raw.trim().replace('-', "_").as_str() {
            "" => {}
            "danger" => behaviors.danger = true,
            "charge" => behaviors.charge = true,
            "future" => behaviors.future = true,
            "action_value" => behaviors.action_value = true,
            "eye_next" => behaviors.eye_next = true,
            "ear_next" => behaviors.ear_next = true,
            "experience" => behaviors.experience = true,
            other => anyhow::bail!(
                "unknown inline behavior '{other}', expected one of danger,charge,future,action_value,eye_next,ear_next,experience"
            ),
        }
    }
    Ok(behaviors)
}

fn live_scene_metadata_from_scenario(
    metadata: &netherwick_sim::ScenarioMetadata,
) -> LiveSceneMetadata {
    LiveSceneMetadata {
        arena: Some(SceneArena {
            width_m: metadata.arena.width_m,
            height_m: metadata.arena.height_m,
        }),
        objects: metadata
            .objects
            .iter()
            .map(|object| SceneObject {
                id: object.id.clone(),
                kind: match &object.kind {
                    SimObjectKind::Obstacle => "obstacle",
                    SimObjectKind::Charger => "charger",
                    SimObjectKind::Person { .. } => "person",
                    SimObjectKind::SoundSource { .. } => "speaker",
                    SimObjectKind::Landmark { .. } => "landmark",
                }
                .to_string(),
                x_m: object.x_m,
                y_m: object.y_m,
                radius_m: object.radius_m,
                label: Some(object.label.clone()),
                color_rgb: Some(object.color_rgb),
            })
            .collect(),
        sensor_calibration: Some(SceneSensorCalibration {
            compact_depth_fov_rad: std::f32::consts::PI * 0.68,
            ..SceneSensorCalibration::sim_default()
        }),
    }
}

async fn run_sim_curriculum(args: SimCurriculumArgs) -> Result<()> {
    if args.validation_ratio < 0.0 || args.test_ratio < 0.0 {
        anyhow::bail!("validation and test ratios must be non-negative");
    }
    if args.validation_ratio + args.test_ratio >= 1.0 {
        anyhow::bail!("validation_ratio + test_ratio must be less than 1.0");
    }

    let kind = ScenarioKind::from(args.scenario);
    let ledger = JsonlLedger::new(&args.out);
    let mut total_ticks = 0usize;
    let mut capture_count = 0usize;
    let mut manifest_episodes = Vec::with_capacity(args.episodes);

    for episode_index in 0..args.episodes {
        let episode_seed = args.seed.saturating_add(episode_index as u64);
        let scenario = build_scenario(ScenarioConfig::new(kind, episode_seed));
        let object_count = scenario.metadata.objects.len();
        let object_summary = scenario_object_summary(&scenario.metadata.objects);
        let runtime = default_runtime(ledger.clone(), &args.llm)?;
        let mut runner = SimRunner::new(runtime, scenario.world, scenario.motors);
        runner.tick_ms = args.tick_ms;
        let mut capture_path_for_manifest = None;

        if let Some(root) = &args.capture_root {
            let mut snapshots = Vec::with_capacity(args.steps);
            runner
                .run_steps_observing(args.steps, |snapshot| snapshots.push(snapshot.clone()))
                .await?;
            let capture_path = Path::new(root).join(format!("episode-{episode_index:03}"));
            capture_path_for_manifest = Some(capture_path.to_string_lossy().to_string());
            let mut writer =
                CaptureWriter::create(&capture_path, CaptureSource::Sim, Some(args.tick_ms))
                    .await?;
            writer.manifest_mut().scenario = Some(scenario.metadata.clone());
            for snapshot in snapshots {
                writer
                    .append_snapshot(snapshot.body.last_update_ms, snapshot, Vec::new())
                    .await?;
            }
            writer.finish().await?;
            capture_count = capture_count.saturating_add(1);
        } else {
            runner.run_steps(args.steps).await?;
        }
        total_ticks = total_ticks.saturating_add(runner.tick_count);
        manifest_episodes.push(serde_json::json!({
            "index": episode_index,
            "split": curriculum_split(
                episode_index,
                args.episodes,
                args.validation_ratio,
                args.test_ratio,
            ),
            "scenario": kind.slug(),
            "seed": episode_seed,
            "steps": args.steps,
            "ticks": runner.tick_count,
            "arena": scenario.metadata.arena,
            "spawn": {
                "x_m": scenario.metadata.body.odometry.x_m,
                "y_m": scenario.metadata.body.odometry.y_m,
                "heading_rad": scenario.metadata.body.odometry.heading_rad,
                "battery_level": scenario.metadata.body.battery_level,
            },
            "object_count": object_count,
            "objects": object_summary,
            "capture": capture_path_for_manifest,
        }));
        println!(
            "episode {} complete: scenario {}, seed {}, ticks {}, objects {}",
            episode_index,
            kind.slug(),
            episode_seed,
            runner.tick_count,
            object_count
        );
    }

    let manifest = serde_json::json!({
        "schema_version": 1,
        "scenario": kind.slug(),
        "base_seed": args.seed,
        "episodes": args.episodes,
        "steps_per_episode": args.steps,
        "tick_ms": args.tick_ms,
        "ledger": args.out,
        "capture_root": args.capture_root,
        "splits": {
            "train": manifest_episodes.iter().filter(|episode| episode["split"] == "train").count(),
            "validation": manifest_episodes.iter().filter(|episode| episode["split"] == "validation").count(),
            "test": manifest_episodes.iter().filter(|episode| episode["split"] == "test").count(),
            "validation_ratio": args.validation_ratio,
            "test_ratio": args.test_ratio,
        },
        "episodes_detail": manifest_episodes,
    });
    fs::create_dir_all(&args.out)?;
    let manifest_path = Path::new(&args.out).join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;

    let transitions = ledger.transitions().await?;
    println!(
        "sim curriculum complete: scenario {}, episodes {}, ticks {}, ledger {}, transitions {}, captures {}, manifest {}",
        kind.slug(),
        args.episodes,
        total_ticks,
        args.out,
        transitions.len(),
        capture_count,
        manifest_path.display()
    );
    Ok(())
}

fn curriculum_split(
    episode_index: usize,
    episode_count: usize,
    validation_ratio: f32,
    test_ratio: f32,
) -> &'static str {
    let validation_count = ((episode_count as f32) * validation_ratio).round() as usize;
    let test_count = ((episode_count as f32) * test_ratio).round() as usize;
    let train_count = episode_count.saturating_sub(validation_count + test_count);
    if episode_index < train_count {
        "train"
    } else if episode_index < train_count + validation_count {
        "validation"
    } else {
        "test"
    }
}

fn scenario_object_summary(objects: &[netherwick_sim::SimObject]) -> serde_json::Value {
    let mut chargers = 0usize;
    let mut obstacles = 0usize;
    let mut people = 0usize;
    let mut speakers = 0usize;
    let mut landmarks = 0usize;

    for object in objects {
        match &object.kind {
            netherwick_sim::SimObjectKind::Charger => chargers += 1,
            netherwick_sim::SimObjectKind::Obstacle => obstacles += 1,
            netherwick_sim::SimObjectKind::Person { .. } => people += 1,
            netherwick_sim::SimObjectKind::SoundSource { .. } => speakers += 1,
            netherwick_sim::SimObjectKind::Landmark { .. } => landmarks += 1,
        }
    }

    serde_json::json!({
        "chargers": chargers,
        "obstacles": obstacles,
        "people": people,
        "speakers": speakers,
        "landmarks": landmarks,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ScenarioEvaluationReport {
    schema_version: u32,
    scenario: String,
    base_seed: u64,
    episodes: usize,
    steps_per_episode: usize,
    tick_ms: u64,
    action_selector_mode: String,
    model_modes: HashMap<String, String>,
    model_loading: RuntimeModelLoadReport,
    ledger: Option<String>,
    capture_root: Option<String>,
    summary: ScenarioEvaluationSummary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    memory: Option<ScenarioMemorySummary>,
    episodes_detail: Vec<ScenarioEpisodeReport>,
    recommendation: String,
    warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct ScenarioEvaluationSummary {
    success_rate: f32,
    collision_rate: f32,
    mean_collisions_per_episode: f32,
    mean_battery_delta: f32,
    mean_final_battery: f32,
    mean_distance_to_charger_final_m: Option<f32>,
    mean_nearest_obstacle_m: Option<f32>,
    mean_distance_traveled_m: f32,
    mean_ticks_survived: f32,
    #[serde(default)]
    stuck_count: usize,
    #[serde(default)]
    trap_kind_counts: HashMap<String, usize>,
    #[serde(default)]
    recovery_attempts: usize,
    #[serde(default)]
    stuck_duration: Option<f32>,
    #[serde(default)]
    mean_stuck_duration: Option<f32>,
    #[serde(default)]
    recovery_success_rate: Option<f32>,
    #[serde(default)]
    mean_recovery_ticks: Option<f32>,
    #[serde(default)]
    repeated_trap_count: usize,
    #[serde(default)]
    dead_battery_tick: Option<usize>,
    #[serde(default)]
    distance_after_recovery_m: Option<f32>,
    mean_safety_interventions: f32,
    behavior_run_records: usize,
    model_fallbacks: usize,
    model_assisted_decisions: usize,
    action_selector_safety_overrides: usize,
    mean_chosen_score: Option<f32>,
    mean_candidate_score: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ScenarioEpisodeReport {
    index: usize,
    seed: u64,
    success: bool,
    ticks: usize,
    collisions: usize,
    wall_hits: usize,
    bumper_hits: usize,
    cliff_hits: usize,
    charging_ticks: usize,
    first_charge_tick: Option<usize>,
    started_battery: f32,
    final_battery: f32,
    battery_delta: f32,
    min_nearest_obstacle_m: Option<f32>,
    mean_nearest_obstacle_m: Option<f32>,
    final_distance_to_charger_m: Option<f32>,
    final_distance_to_person_m: Option<f32>,
    final_distance_to_speaker_m: Option<f32>,
    distance_traveled_m: f32,
    #[serde(default)]
    stuck_ticks: usize,
    #[serde(default)]
    stuck_count: usize,
    #[serde(default)]
    trap_kind_counts: HashMap<String, usize>,
    #[serde(default)]
    recovery_attempts: usize,
    #[serde(default)]
    stuck_duration: Option<f32>,
    #[serde(default)]
    mean_stuck_duration: Option<f32>,
    #[serde(default)]
    recovery_success_rate: Option<f32>,
    #[serde(default)]
    mean_recovery_ticks: Option<f32>,
    #[serde(default)]
    repeated_trap_count: usize,
    #[serde(default)]
    dead_battery_tick: Option<usize>,
    #[serde(default)]
    distance_after_recovery_m: Option<f32>,
    unique_actions: Vec<String>,
    safety_interventions: usize,
    behavior_run_records: usize,
    model_fallbacks: usize,
    model_assisted_decisions: usize,
    action_selector_safety_overrides: usize,
    action_selector_fallbacks: usize,
    mean_chosen_score: Option<f32>,
    mean_candidate_score: Option<f32>,
    ticks_with_eye_frames: usize,
    ticks_with_ear_features: usize,
    ticks_with_voice_embeddings: usize,
    ticks_with_face_embeddings: usize,
    ticks_with_kinect_skeletons: usize,
    ticks_with_future_predictions: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    memory: Option<ScenarioEpisodeMemoryReport>,
    capture: Option<String>,
    ledger: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct ScenarioMemorySummary {
    places_visited: usize,
    mean_places_visited_per_episode: f32,
    charge_memory_hit_rate: Option<f32>,
    danger_memory_hit_rate: Option<f32>,
    social_memory_hit_rate: Option<f32>,
    novelty_decay_sane: bool,
    warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct ScenarioEpisodeMemoryReport {
    places_visited: usize,
    charge_memory_ticks: usize,
    charge_opportunity_ticks: usize,
    charge_memory_hit_rate: Option<f32>,
    danger_memory_ticks: usize,
    danger_opportunity_ticks: usize,
    danger_memory_hit_rate: Option<f32>,
    social_memory_ticks: usize,
    social_opportunity_ticks: usize,
    social_memory_hit_rate: Option<f32>,
    first_novelty: Option<f32>,
    final_novelty: Option<f32>,
    novelty_decayed: bool,
}

#[derive(Clone, Debug)]
struct EpisodeMetricBuilder {
    kind: ScenarioKind,
    metadata: netherwick_sim::ScenarioMetadata,
    index: usize,
    seed: u64,
    ledger: Option<String>,
    capture: Option<String>,
    ticks: usize,
    collisions: usize,
    wall_hits: usize,
    bumper_hits: usize,
    cliff_hits: usize,
    charging_ticks: usize,
    first_charge_tick: Option<usize>,
    started_battery: Option<f32>,
    final_battery: f32,
    min_nearest_obstacle_m: Option<f32>,
    nearest_obstacle_sum: f32,
    nearest_obstacle_count: usize,
    start_position: Option<(f32, f32)>,
    last_position: Option<(f32, f32)>,
    distance_traveled_m: f32,
    stuck_ticks: usize,
    stuck_count: usize,
    trap_kind_counts: HashMap<String, usize>,
    recovery_attempts: usize,
    stuck_duration_sum_ms: f32,
    stuck_duration_count: usize,
    active_stuck_duration_ms: Option<f32>,
    recovery_successes: usize,
    recovery_ticks_sum: usize,
    recovery_tick_count: usize,
    repeated_trap_count: usize,
    distance_at_last_recovery_m: Option<f32>,
    dead_battery_tick: Option<usize>,
    unique_actions: BTreeSet<String>,
    safety_interventions: usize,
    behavior_run_records: usize,
    model_fallbacks: usize,
    model_assisted_decisions: usize,
    action_selector_safety_overrides: usize,
    action_selector_fallbacks: usize,
    chosen_score_sum: f32,
    chosen_score_count: usize,
    candidate_score_sum: f32,
    candidate_score_count: usize,
    ticks_with_eye_frames: usize,
    ticks_with_ear_features: usize,
    ticks_with_voice_embeddings: usize,
    ticks_with_face_embeddings: usize,
    ticks_with_kinect_skeletons: usize,
    ticks_with_future_predictions: usize,
    memory: ScenarioEpisodeMemoryBuilder,
}

impl EpisodeMetricBuilder {
    fn new(
        kind: ScenarioKind,
        metadata: netherwick_sim::ScenarioMetadata,
        index: usize,
        seed: u64,
        ledger: Option<String>,
        capture: Option<String>,
    ) -> Self {
        Self {
            kind,
            metadata,
            index,
            seed,
            ledger,
            capture,
            ticks: 0,
            collisions: 0,
            wall_hits: 0,
            bumper_hits: 0,
            cliff_hits: 0,
            charging_ticks: 0,
            first_charge_tick: None,
            started_battery: None,
            final_battery: 0.0,
            min_nearest_obstacle_m: None,
            nearest_obstacle_sum: 0.0,
            nearest_obstacle_count: 0,
            start_position: None,
            last_position: None,
            distance_traveled_m: 0.0,
            stuck_ticks: 0,
            stuck_count: 0,
            trap_kind_counts: HashMap::new(),
            recovery_attempts: 0,
            stuck_duration_sum_ms: 0.0,
            stuck_duration_count: 0,
            active_stuck_duration_ms: None,
            recovery_successes: 0,
            recovery_ticks_sum: 0,
            recovery_tick_count: 0,
            repeated_trap_count: 0,
            distance_at_last_recovery_m: None,
            dead_battery_tick: None,
            unique_actions: BTreeSet::new(),
            safety_interventions: 0,
            behavior_run_records: 0,
            model_fallbacks: 0,
            model_assisted_decisions: 0,
            action_selector_safety_overrides: 0,
            action_selector_fallbacks: 0,
            chosen_score_sum: 0.0,
            chosen_score_count: 0,
            candidate_score_sum: 0.0,
            candidate_score_count: 0,
            ticks_with_eye_frames: 0,
            ticks_with_ear_features: 0,
            ticks_with_voice_embeddings: 0,
            ticks_with_face_embeddings: 0,
            ticks_with_kinect_skeletons: 0,
            ticks_with_future_predictions: 0,
            memory: ScenarioEpisodeMemoryBuilder::default(),
        }
    }

    fn observe(&mut self, snapshot: &WorldSnapshot, tick: &RuntimeTick) {
        self.ticks = self.ticks.saturating_add(1);
        let body = &snapshot.body;
        self.started_battery.get_or_insert(body.battery_level);
        self.final_battery = body.battery_level;
        if self.dead_battery_tick.is_none() && body.battery_level <= f32::EPSILON && !body.charging
        {
            self.dead_battery_tick = Some(self.ticks.saturating_sub(1));
        }
        let position = (body.odometry.x_m, body.odometry.y_m);
        if self.start_position.is_none() {
            self.start_position = Some(position);
        }
        if let Some(last) = self.last_position.replace(position) {
            let step_distance = distance_between(last, position);
            self.distance_traveled_m += step_distance;
        }

        let bumper = body.flags.bump_left || body.flags.bump_right;
        let cliff = body.flags.cliff_left
            || body.flags.cliff_front_left
            || body.flags.cliff_front_right
            || body.flags.cliff_right;
        let collision = bumper || body.flags.wall || cliff;
        if collision {
            self.collisions = self.collisions.saturating_add(1);
        }
        if body.flags.wall {
            self.wall_hits = self.wall_hits.saturating_add(1);
        }
        if bumper {
            self.bumper_hits = self.bumper_hits.saturating_add(1);
        }
        if cliff {
            self.cliff_hits = self.cliff_hits.saturating_add(1);
        }
        if body.charging {
            if self.first_charge_tick.is_none() {
                self.first_charge_tick = Some(self.ticks.saturating_sub(1));
            }
            self.charging_ticks = self.charging_ticks.saturating_add(1);
        }
        if let Some(nearest) = snapshot.range.nearest_m {
            self.min_nearest_obstacle_m = Some(
                self.min_nearest_obstacle_m
                    .map(|value| value.min(nearest))
                    .unwrap_or(nearest),
            );
            self.nearest_obstacle_sum += nearest;
            self.nearest_obstacle_count = self.nearest_obstacle_count.saturating_add(1);
        }
        if let Some(action) = &tick.chosen_action {
            self.unique_actions.insert(format!("{action:?}"));
        }
        self.observe_stuck(snapshot);
        if tick
            .frame
            .now
            .extensions
            .get("safety.vetoed")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            self.safety_interventions = self.safety_interventions.saturating_add(1);
        }
        self.observe_behavior_runs(&tick.frame.behavior_runs);
        self.observe_action_selector(tick);
        if snapshot.eye_frame.is_some() || !snapshot.eye.frames.is_empty() {
            self.ticks_with_eye_frames = self.ticks_with_eye_frames.saturating_add(1);
        }
        if !snapshot.ear.features.is_empty() || snapshot.ear_pcm.is_some() {
            self.ticks_with_ear_features = self.ticks_with_ear_features.saturating_add(1);
        }
        if !snapshot.voice.embeddings.is_empty() {
            self.ticks_with_voice_embeddings = self.ticks_with_voice_embeddings.saturating_add(1);
        }
        if !snapshot.face.embeddings.is_empty() {
            self.ticks_with_face_embeddings = self.ticks_with_face_embeddings.saturating_add(1);
        }
        if !snapshot.kinect.skeletons.is_empty() {
            self.ticks_with_kinect_skeletons = self.ticks_with_kinect_skeletons.saturating_add(1);
        }
        if !tick.frame.predicted_futures.is_empty() {
            self.ticks_with_future_predictions =
                self.ticks_with_future_predictions.saturating_add(1);
        }
        self.memory.observe(snapshot, tick);
    }

    fn observe_stuck(&mut self, snapshot: &WorldSnapshot) {
        let Some(extension) = snapshot
            .extensions
            .iter()
            .find(|extension| extension.name == "sim.stuck")
        else {
            return;
        };
        let values = &extension.values;
        let active = values.first().copied().unwrap_or(0.0) > 0.0;
        let duration_ms = values.get(3).copied().unwrap_or(0.0).max(0.0);
        let event_started = values.get(6).copied().unwrap_or(0.0) > 0.0;
        let recovered = values.get(7).copied().unwrap_or(0.0) > 0.0;
        let trap_kind = trap_kind_label(values.get(10).copied().unwrap_or(0.0));
        let attempts = values.get(11).copied().unwrap_or(0.0).max(0.0) as usize;
        let repeated = values.get(12).copied().unwrap_or(0.0).max(0.0) as usize;
        if event_started {
            self.stuck_count = self.stuck_count.saturating_add(1);
            self.active_stuck_duration_ms = Some(duration_ms);
            if let Some(kind) = trap_kind {
                *self.trap_kind_counts.entry(kind.to_string()).or_default() += 1;
            }
        }
        if active {
            self.stuck_ticks = self.stuck_ticks.saturating_add(1);
            self.active_stuck_duration_ms = Some(duration_ms);
        }
        self.recovery_attempts = self.recovery_attempts.max(attempts);
        self.repeated_trap_count = self.repeated_trap_count.max(repeated);
        if recovered {
            self.recovery_successes = self.recovery_successes.saturating_add(1);
            if let Some(duration) = self.active_stuck_duration_ms.take() {
                self.stuck_duration_sum_ms += duration;
                self.stuck_duration_count = self.stuck_duration_count.saturating_add(1);
                self.recovery_ticks_sum = self
                    .recovery_ticks_sum
                    .saturating_add((duration / 100.0).round().max(0.0) as usize);
                self.recovery_tick_count = self.recovery_tick_count.saturating_add(1);
            }
            self.distance_at_last_recovery_m = Some(self.distance_traveled_m);
        }
    }

    fn observe_behavior_runs(&mut self, records: &[ErasedBehaviorRunRecord]) {
        self.behavior_run_records = self.behavior_run_records.saturating_add(records.len());
        self.model_fallbacks = self.model_fallbacks.saturating_add(
            records
                .iter()
                .filter(|record| {
                    record.error.is_some()
                        || (record.regime == BehaviorRegime::ModelInfer
                            && record.model_json.is_none()
                            && record.hardcoded_json.is_some())
                })
                .count(),
        );
    }

    fn observe_action_selector(&mut self, tick: &RuntimeTick) {
        let Some(value) = tick.frame.now.extensions.get("action_selector") else {
            return;
        };
        let Ok(decision) = serde_json::from_value::<ActionSelectionDecision>(value.clone()) else {
            return;
        };
        if decision.mode == ActionSelectorMode::ModelAssisted {
            self.model_assisted_decisions = self.model_assisted_decisions.saturating_add(1);
        }
        if decision.safety_overrode {
            self.action_selector_safety_overrides =
                self.action_selector_safety_overrides.saturating_add(1);
        }
        if !decision.fallback_warnings.is_empty()
            || decision
                .candidates
                .iter()
                .any(|candidate| candidate.fallback_used)
        {
            self.action_selector_fallbacks = self.action_selector_fallbacks.saturating_add(1);
        }
        if let Some(score) = decision.selected_score {
            self.chosen_score_sum += score;
            self.chosen_score_count = self.chosen_score_count.saturating_add(1);
        }
        for candidate in decision.candidates {
            self.candidate_score_sum += candidate.score;
            self.candidate_score_count = self.candidate_score_count.saturating_add(1);
        }
    }

    fn finish(self) -> ScenarioEpisodeReport {
        let final_position = self.last_position.unwrap_or_else(|| {
            (
                self.metadata.body.odometry.x_m,
                self.metadata.body.odometry.y_m,
            )
        });
        let started_battery = self
            .started_battery
            .unwrap_or(self.metadata.body.battery_level);
        let final_distance_to_charger_m =
            nearest_object_distance(final_position, &self.metadata.objects, |kind| {
                matches!(kind, netherwick_sim::SimObjectKind::Charger)
            });
        let final_distance_to_person_m =
            nearest_object_distance(final_position, &self.metadata.objects, |kind| {
                matches!(kind, netherwick_sim::SimObjectKind::Person { .. })
            });
        let final_distance_to_speaker_m =
            nearest_object_distance(final_position, &self.metadata.objects, |kind| {
                matches!(kind, netherwick_sim::SimObjectKind::SoundSource { .. })
            });
        let mean_nearest_obstacle_m = if self.nearest_obstacle_count == 0 {
            None
        } else {
            Some(self.nearest_obstacle_sum / self.nearest_obstacle_count as f32)
        };
        let mut stuck_duration_sum_ms = self.stuck_duration_sum_ms;
        let mut stuck_duration_count = self.stuck_duration_count;
        if let Some(duration) = self.active_stuck_duration_ms {
            stuck_duration_sum_ms += duration;
            stuck_duration_count = stuck_duration_count.saturating_add(1);
        }
        let stuck_duration = (stuck_duration_count > 0)
            .then_some(stuck_duration_sum_ms / stuck_duration_count as f32);
        let mut report = ScenarioEpisodeReport {
            index: self.index,
            seed: self.seed,
            success: false,
            ticks: self.ticks,
            collisions: self.collisions,
            wall_hits: self.wall_hits,
            bumper_hits: self.bumper_hits,
            cliff_hits: self.cliff_hits,
            charging_ticks: self.charging_ticks,
            first_charge_tick: self.first_charge_tick,
            started_battery,
            final_battery: self.final_battery,
            battery_delta: self.final_battery - started_battery,
            min_nearest_obstacle_m: self.min_nearest_obstacle_m,
            mean_nearest_obstacle_m,
            final_distance_to_charger_m,
            final_distance_to_person_m,
            final_distance_to_speaker_m,
            distance_traveled_m: self.distance_traveled_m,
            stuck_ticks: self.stuck_ticks,
            stuck_count: self.stuck_count,
            trap_kind_counts: self.trap_kind_counts,
            recovery_attempts: self.recovery_attempts,
            stuck_duration,
            mean_stuck_duration: stuck_duration,
            recovery_success_rate: (self.stuck_count > 0)
                .then_some(self.recovery_successes as f32 / self.stuck_count as f32),
            mean_recovery_ticks: (self.recovery_tick_count > 0)
                .then_some(self.recovery_ticks_sum as f32 / self.recovery_tick_count as f32),
            repeated_trap_count: self.repeated_trap_count,
            dead_battery_tick: self.dead_battery_tick,
            distance_after_recovery_m: self
                .distance_at_last_recovery_m
                .map(|distance| (self.distance_traveled_m - distance).max(0.0)),
            unique_actions: self.unique_actions.into_iter().collect(),
            safety_interventions: self.safety_interventions,
            behavior_run_records: self.behavior_run_records,
            model_fallbacks: self.model_fallbacks,
            model_assisted_decisions: self.model_assisted_decisions,
            action_selector_safety_overrides: self.action_selector_safety_overrides,
            action_selector_fallbacks: self.action_selector_fallbacks,
            mean_chosen_score: (self.chosen_score_count > 0)
                .then_some(self.chosen_score_sum / self.chosen_score_count as f32),
            mean_candidate_score: (self.candidate_score_count > 0)
                .then_some(self.candidate_score_sum / self.candidate_score_count as f32),
            ticks_with_eye_frames: self.ticks_with_eye_frames,
            ticks_with_ear_features: self.ticks_with_ear_features,
            ticks_with_voice_embeddings: self.ticks_with_voice_embeddings,
            ticks_with_face_embeddings: self.ticks_with_face_embeddings,
            ticks_with_kinect_skeletons: self.ticks_with_kinect_skeletons,
            ticks_with_future_predictions: self.ticks_with_future_predictions,
            memory: Some(self.memory.finish()),
            capture: self.capture,
            ledger: self.ledger,
        };
        report.success = episode_success(self.kind, &report);
        report
    }
}

#[derive(Clone, Debug, Default)]
struct ScenarioEpisodeMemoryBuilder {
    max_places_visited: usize,
    charge_memory_ticks: usize,
    charge_opportunity_ticks: usize,
    danger_memory_ticks: usize,
    danger_opportunity_ticks: usize,
    social_memory_ticks: usize,
    social_opportunity_ticks: usize,
    first_novelty: Option<f32>,
    final_novelty: Option<f32>,
}

impl ScenarioEpisodeMemoryBuilder {
    fn observe(&mut self, snapshot: &WorldSnapshot, tick: &RuntimeTick) {
        let memory = &tick.frame.now.memory;
        self.max_places_visited = self.max_places_visited.max(memory.places_visited as usize);
        self.first_novelty.get_or_insert(memory.place_novelty);
        self.final_novelty = Some(memory.place_novelty);

        let charger_near = snapshot.body.charging
            || sim_world_score(snapshot, 3).max(sim_world_score(snapshot, 4)) >= 0.3;
        if charger_near {
            self.charge_opportunity_ticks = self.charge_opportunity_ticks.saturating_add(1);
        }
        if memory.place_charge_value >= 0.3 {
            self.charge_memory_ticks = self.charge_memory_ticks.saturating_add(1);
        }

        let danger_near = snapshot.body.flags.bump_left
            || snapshot.body.flags.bump_right
            || snapshot.body.flags.wall
            || snapshot.body.flags.cliff_left
            || snapshot.body.flags.cliff_front_left
            || snapshot.body.flags.cliff_front_right
            || snapshot.body.flags.cliff_right
            || snapshot
                .range
                .nearest_m
                .map(|nearest| nearest <= 0.35)
                .unwrap_or(false);
        if danger_near {
            self.danger_opportunity_ticks = self.danger_opportunity_ticks.saturating_add(1);
        }
        if memory.place_danger >= 0.3 {
            self.danger_memory_ticks = self.danger_memory_ticks.saturating_add(1);
        }

        let social_seen = !snapshot.face.embeddings.is_empty()
            || !snapshot.voice.embeddings.is_empty()
            || !snapshot.kinect.skeletons.is_empty();
        if social_seen {
            self.social_opportunity_ticks = self.social_opportunity_ticks.saturating_add(1);
        }
        if memory.place_social_value >= 0.3 {
            self.social_memory_ticks = self.social_memory_ticks.saturating_add(1);
        }
    }

    fn finish(self) -> ScenarioEpisodeMemoryReport {
        ScenarioEpisodeMemoryReport {
            places_visited: self.max_places_visited,
            charge_memory_ticks: self.charge_memory_ticks,
            charge_opportunity_ticks: self.charge_opportunity_ticks,
            charge_memory_hit_rate: hit_rate(
                self.charge_memory_ticks,
                self.charge_opportunity_ticks,
            ),
            danger_memory_ticks: self.danger_memory_ticks,
            danger_opportunity_ticks: self.danger_opportunity_ticks,
            danger_memory_hit_rate: hit_rate(
                self.danger_memory_ticks,
                self.danger_opportunity_ticks,
            ),
            social_memory_ticks: self.social_memory_ticks,
            social_opportunity_ticks: self.social_opportunity_ticks,
            social_memory_hit_rate: hit_rate(
                self.social_memory_ticks,
                self.social_opportunity_ticks,
            ),
            first_novelty: self.first_novelty,
            final_novelty: self.final_novelty,
            novelty_decayed: self
                .first_novelty
                .zip(self.final_novelty)
                .map(|(first, final_value)| final_value <= first)
                .unwrap_or(false),
        }
    }
}

async fn run_eval_scenario(args: EvalScenarioArgs) -> Result<()> {
    let kind = ScenarioKind::from(args.scenario);
    let flags = RuntimeModelFlags::from(&args);
    let mut model_loading = load_runtime_models_from_flags(&flags)?.1;
    if args.future_mode == FutureMode::ModelInfer {
        model_loading.blocked_model_infer.push(
            "future model-infer is limited to prediction behavior; motor safety remains hardcoded"
                .to_string(),
        );
    }
    if args.experience_mode == ExperienceMode::ModelInfer {
        model_loading.blocked_model_infer.push(
            "experience model-infer changes latent encoding only; motor safety remains hardcoded"
                .to_string(),
        );
    }

    let mut episodes_detail = Vec::with_capacity(args.episodes);
    for episode_index in 0..args.episodes {
        let episode_seed = args.seed.saturating_add(episode_index as u64);
        let scenario = build_scenario(ScenarioConfig::new(kind, episode_seed));
        let capture = args.capture_root.as_ref().map(|root| {
            Path::new(root)
                .join(format!("episode-{episode_index:03}"))
                .to_string_lossy()
                .to_string()
        });
        let builder = EpisodeMetricBuilder::new(
            kind,
            scenario.metadata.clone(),
            episode_index,
            episode_seed,
            args.ledger.clone(),
            capture.clone(),
        );
        let (episode, warnings) = if let Some(ledger_path) = &args.ledger {
            let mut runtime = default_runtime(JsonlLedger::new(ledger_path), &args.llm)?;
            runtime = runtime.with_action_selector_mode(args.action_selector.into());
            if let Some(models) = load_runtime_models_from_flags(&flags)?.0 {
                runtime = runtime.with_models(models);
            }
            run_eval_episode(runtime, scenario.world, scenario.motors, &args, builder).await?
        } else {
            let mut runtime = default_noop_runtime(&args.llm)?;
            runtime = runtime.with_action_selector_mode(args.action_selector.into());
            if let Some(models) = load_runtime_models_from_flags(&flags)?.0 {
                runtime = runtime.with_models(models);
            }
            run_eval_episode(runtime, scenario.world, scenario.motors, &args, builder).await?
        };
        model_loading.warnings.extend(warnings);
        println!(
            "eval episode {} complete: scenario {}, seed {}, ticks {}, success {}, collisions {}",
            episode.index,
            kind.slug(),
            episode.seed,
            episode.ticks,
            episode.success,
            episode.collisions
        );
        episodes_detail.push(episode);
    }

    let summary = summarize_episodes(&episodes_detail);
    let memory = args
        .memory_report
        .then(|| summarize_episode_memory(&episodes_detail));
    let recommendation = scenario_recommendation(args.episodes, &summary);
    let report = ScenarioEvaluationReport {
        schema_version: 1,
        scenario: kind.slug().to_string(),
        base_seed: args.seed,
        episodes: args.episodes,
        steps_per_episode: args.steps,
        tick_ms: args.tick_ms,
        action_selector_mode: ActionSelectorMode::from(args.action_selector)
            .as_str()
            .to_string(),
        model_modes: model_modes_from_flags(&flags),
        model_loading: model_loading.clone(),
        ledger: args.ledger.clone(),
        capture_root: args.capture_root.clone(),
        summary,
        memory,
        episodes_detail,
        recommendation,
        warnings: model_loading.warnings.clone(),
    };

    let bytes = serde_json::to_vec_pretty(&report)?;
    if let Some(out) = &args.out {
        if let Some(parent) = Path::new(out).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(out, &bytes)?;
        println!("scenario evaluation report written: {out}");
    } else {
        println!("{}", String::from_utf8_lossy(&bytes));
    }
    Ok(())
}

async fn run_eval_episode<R>(
    runtime: R,
    world: netherwick_sim::VirtualWorld,
    motors: netherwick_sim::SimMotorComplex,
    args: &EvalScenarioArgs,
    mut metrics: EpisodeMetricBuilder,
) -> Result<(ScenarioEpisodeReport, Vec<String>)>
where
    R: RuntimeLoop + Send,
{
    let mut warnings = Vec::new();
    let mut runner = SimRunner::new(runtime, world, motors);
    runner.tick_ms = args.tick_ms;
    let mut snapshots = Vec::new();
    runner
        .run_steps_observing_ticks(args.steps, |snapshot, tick| {
            if metrics.capture.is_some() {
                snapshots.push(snapshot.clone());
            }
            metrics.observe(snapshot, tick);
        })
        .await?;

    if let Some(capture_path) = &metrics.capture {
        let mut writer =
            CaptureWriter::create(capture_path, CaptureSource::Sim, Some(args.tick_ms)).await?;
        writer.manifest_mut().scenario = Some(metrics.metadata.clone());
        for snapshot in snapshots {
            writer
                .append_snapshot(snapshot.body.last_update_ms, snapshot, Vec::new())
                .await?;
        }
        writer.finish().await?;
    }

    if runner.tick_count < args.steps {
        warnings.push(format!(
            "episode {} stopped after {} ticks before requested {} steps",
            metrics.index, runner.tick_count, args.steps
        ));
    }
    Ok((metrics.finish(), warnings))
}

fn configured_llm_agent(args: &LlmArgs) -> Result<ConfiguredLlmAgent> {
    let mut config = match &args.llm_config {
        Some(path) => LlmConfig::load(path)?,
        None => LlmConfig::default(),
    };
    if let Some(provider) = args.llm_provider {
        config.provider = provider.into();
    }
    ConfiguredLlmAgent::from_config(config)
}

fn default_noop_runtime(
    llm_args: &LlmArgs,
) -> Result<
    MinimalRuntime<
        NoopLedger,
        InMemoryExperienceStore,
        InMemoryExperienceStore,
        SimpleConductor,
        SimpleSafety,
        ConfiguredLlmAgent,
    >,
> {
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    Ok(MinimalRuntime::with_default_events(
        NoopLedger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        configured_llm_agent(llm_args)?,
    ))
}

fn episode_success(kind: ScenarioKind, episode: &ScenarioEpisodeReport) -> bool {
    match kind {
        ScenarioKind::EmptyRoom => episode.ticks > 0 && episode.collisions == 0,
        ScenarioKind::ObstacleAvoidance => {
            episode.ticks > 0
                && episode.collisions <= (episode.ticks / 50).max(1)
                && episode.stuck_ticks < episode.ticks / 2
                && episode.distance_traveled_m > 0.05
        }
        ScenarioKind::CornerTrap => {
            episode.stuck_count > 0
                && episode.recovery_success_rate.unwrap_or(0.0) > 0.0
                && episode.distance_traveled_m > 0.02
        }
        ScenarioKind::ColumnTrap => {
            episode.stuck_count > 0
                && episode.recovery_success_rate.unwrap_or(0.0) > 0.0
                && episode.distance_after_recovery_m.unwrap_or(0.0) > 0.08
        }
        ScenarioKind::ChargerSeeking => episode.charging_ticks > 0 || episode.battery_delta > 0.03,
        ScenarioKind::PersonAndSpeaker => {
            episode.ticks > 0
                && episode.collisions == 0
                && (episode.ticks_with_face_embeddings > 0
                    || episode.ticks_with_voice_embeddings > 0
                    || episode.ticks_with_kinect_skeletons > 0
                    || episode.ticks_with_ear_features > 0)
        }
        ScenarioKind::MixedRoom => {
            episode.ticks > 0
                && episode.collisions <= (episode.ticks / 40).max(1)
                && (episode.charging_ticks > 0
                    || episode.ticks_with_face_embeddings > 0
                    || episode.ticks_with_voice_embeddings > 0)
        }
    }
}

fn summarize_episodes(episodes: &[ScenarioEpisodeReport]) -> ScenarioEvaluationSummary {
    if episodes.is_empty() {
        return ScenarioEvaluationSummary::default();
    }
    let count = episodes.len() as f32;
    let total_ticks: usize = episodes.iter().map(|episode| episode.ticks).sum();
    let total_collisions: usize = episodes.iter().map(|episode| episode.collisions).sum();
    let mut trap_kind_counts = HashMap::new();
    for episode in episodes {
        for (kind, count) in &episode.trap_kind_counts {
            *trap_kind_counts.entry(kind.clone()).or_default() += count;
        }
    }
    ScenarioEvaluationSummary {
        success_rate: episodes.iter().filter(|episode| episode.success).count() as f32 / count,
        collision_rate: if total_ticks == 0 {
            0.0
        } else {
            total_collisions as f32 / total_ticks as f32
        },
        mean_collisions_per_episode: total_collisions as f32 / count,
        mean_battery_delta: mean(episodes.iter().map(|episode| episode.battery_delta)),
        mean_final_battery: mean(episodes.iter().map(|episode| episode.final_battery)),
        mean_distance_to_charger_final_m: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.final_distance_to_charger_m),
        ),
        mean_nearest_obstacle_m: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.mean_nearest_obstacle_m),
        ),
        mean_distance_traveled_m: mean(episodes.iter().map(|episode| episode.distance_traveled_m)),
        mean_ticks_survived: mean(episodes.iter().map(|episode| episode.ticks as f32)),
        stuck_count: episodes.iter().map(|episode| episode.stuck_count).sum(),
        trap_kind_counts,
        recovery_attempts: episodes
            .iter()
            .map(|episode| episode.recovery_attempts)
            .sum(),
        stuck_duration: mean_optional(episodes.iter().filter_map(|episode| episode.stuck_duration)),
        mean_stuck_duration: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.mean_stuck_duration),
        ),
        recovery_success_rate: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.recovery_success_rate),
        ),
        mean_recovery_ticks: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.mean_recovery_ticks),
        ),
        repeated_trap_count: episodes
            .iter()
            .map(|episode| episode.repeated_trap_count)
            .sum(),
        dead_battery_tick: episodes
            .iter()
            .filter_map(|episode| episode.dead_battery_tick)
            .min(),
        distance_after_recovery_m: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.distance_after_recovery_m),
        ),
        mean_safety_interventions: mean(
            episodes
                .iter()
                .map(|episode| episode.safety_interventions as f32),
        ),
        behavior_run_records: episodes
            .iter()
            .map(|episode| episode.behavior_run_records)
            .sum(),
        model_fallbacks: episodes.iter().map(|episode| episode.model_fallbacks).sum(),
        model_assisted_decisions: episodes
            .iter()
            .map(|episode| episode.model_assisted_decisions)
            .sum(),
        action_selector_safety_overrides: episodes
            .iter()
            .map(|episode| episode.action_selector_safety_overrides)
            .sum(),
        mean_chosen_score: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.mean_chosen_score),
        ),
        mean_candidate_score: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.mean_candidate_score),
        ),
    }
}

fn trap_kind_label(code: f32) -> Option<&'static str> {
    match code.round() as i32 {
        1 => Some("wall"),
        2 => Some("corner"),
        3 => Some("column"),
        _ => None,
    }
}

fn summarize_episode_memory(episodes: &[ScenarioEpisodeReport]) -> ScenarioMemorySummary {
    let memory_reports = episodes
        .iter()
        .filter_map(|episode| episode.memory.as_ref())
        .collect::<Vec<_>>();
    if memory_reports.is_empty() {
        return ScenarioMemorySummary {
            novelty_decay_sane: false,
            warnings: vec!["no episode memory reports".to_string()],
            ..ScenarioMemorySummary::default()
        };
    }
    let places_visited = memory_reports
        .iter()
        .map(|memory| memory.places_visited)
        .max()
        .unwrap_or(0);
    let mut warnings = Vec::new();
    if places_visited == 0 {
        warnings.push("memory observed zero places".to_string());
    }
    let novelty_decay_sane = memory_reports.iter().any(|memory| memory.novelty_decayed);
    if !novelty_decay_sane {
        warnings.push("novelty did not decay in any episode".to_string());
    }
    ScenarioMemorySummary {
        places_visited,
        mean_places_visited_per_episode: mean(
            memory_reports
                .iter()
                .map(|memory| memory.places_visited as f32),
        ),
        charge_memory_hit_rate: aggregate_hit_rate(
            memory_reports
                .iter()
                .map(|memory| (memory.charge_memory_ticks, memory.charge_opportunity_ticks)),
        ),
        danger_memory_hit_rate: aggregate_hit_rate(
            memory_reports
                .iter()
                .map(|memory| (memory.danger_memory_ticks, memory.danger_opportunity_ticks)),
        ),
        social_memory_hit_rate: aggregate_hit_rate(
            memory_reports
                .iter()
                .map(|memory| (memory.social_memory_ticks, memory.social_opportunity_ticks)),
        ),
        novelty_decay_sane,
        warnings,
    }
}

fn scenario_recommendation(episodes: usize, summary: &ScenarioEvaluationSummary) -> String {
    if episodes < 3 {
        "insufficient_data".to_string()
    } else if summary.collision_rate > 0.10 || summary.mean_collisions_per_episode > 5.0 {
        "reject_or_continue_training".to_string()
    } else if summary.success_rate >= 0.80 && summary.collision_rate <= 0.02 {
        "candidate_for_more_eval".to_string()
    } else {
        "continue_training".to_string()
    }
}

fn nearest_object_distance<F>(
    position: (f32, f32),
    objects: &[netherwick_sim::SimObject],
    matches_kind: F,
) -> Option<f32>
where
    F: Fn(&netherwick_sim::SimObjectKind) -> bool,
{
    objects
        .iter()
        .filter(|object| matches_kind(&object.kind))
        .map(|object| {
            (distance_between(position, (object.x_m, object.y_m)) - object.radius_m).max(0.0)
        })
        .min_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
}

fn distance_between(left: (f32, f32), right: (f32, f32)) -> f32 {
    let dx = left.0 - right.0;
    let dy = left.1 - right.1;
    ((dx * dx) + (dy * dy)).sqrt()
}

fn hit_rate(hits: usize, opportunities: usize) -> Option<f32> {
    (opportunities > 0).then_some(hits as f32 / opportunities as f32)
}

fn aggregate_hit_rate(pairs: impl Iterator<Item = (usize, usize)>) -> Option<f32> {
    let (hits, opportunities) = pairs.fold((0usize, 0usize), |acc, pair| {
        (acc.0.saturating_add(pair.0), acc.1.saturating_add(pair.1))
    });
    hit_rate(hits, opportunities)
}

fn sim_world_score(snapshot: &WorldSnapshot, index: usize) -> f32 {
    snapshot
        .extensions
        .iter()
        .find(|extension| extension.name == "sim.world")
        .and_then(|extension| extension.values.get(index).copied())
        .unwrap_or(0.0)
}

fn mean(values: impl Iterator<Item = f32>) -> f32 {
    let mut count = 0usize;
    let mut sum = 0.0;
    for value in values {
        count = count.saturating_add(1);
        sum += value;
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

fn mean_optional(values: impl Iterator<Item = f32>) -> Option<f32> {
    let mut count = 0usize;
    let mut sum = 0.0;
    for value in values {
        count = count.saturating_add(1);
        sum += value;
    }
    (count > 0).then_some(sum / count as f32)
}

async fn capture_sim(args: CaptureSimArgs) -> Result<()> {
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let runtime = MinimalRuntime::with_default_events(
        NoopLedger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        configured_llm_agent(&args.llm)?,
    );
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::MixedRoom, args.seed));
    let world = scenario.world;
    let motors = scenario.motors;
    let mut runner = SimRunner::new(runtime, world, motors);
    let mut snapshots = Vec::new();
    runner
        .run_steps_observing(args.steps, |snapshot| snapshots.push(snapshot.clone()))
        .await?;

    let mut writer =
        CaptureWriter::create(&args.out, CaptureSource::Sim, Some(args.tick_ms)).await?;
    writer.manifest_mut().scenario = Some(scenario.metadata);
    for snapshot in snapshots {
        let t_ms = snapshot.body.last_update_ms;
        writer.append_snapshot(t_ms, snapshot, Vec::new()).await?;
    }
    let manifest = writer.finish().await?;

    println!(
        "capture complete: {} frames, seed {}, out {}, tick_ms {:?}",
        manifest.frame_count, args.seed, args.out, manifest.tick_ms
    );
    Ok(())
}

async fn replay_capture(args: ReplayCaptureArgs) -> Result<()> {
    let reader = CaptureReader::open(&args.capture).await?;
    let ledger = JsonlLedger::new(&args.ledger);
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let runtime = MinimalRuntime::with_default_events(
        ledger.clone(),
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        configured_llm_agent(&args.llm)?,
    );
    let mut runner = CaptureReplayRunner::new(runtime, reader);
    let summary = runner.replay().await?;
    let transitions = ledger.transitions().await?;

    println!(
        "replay complete: {} frames replayed, {} runtime ticks, ledger {}, {} transitions written",
        summary.frames_replayed,
        summary.runtime_ticks,
        args.ledger,
        transitions.len()
    );
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
enum CounterfactualEdit {
    MoveObject {
        kind: CounterfactualObjectKind,
        id: Option<String>,
        x_m: f32,
        y_m: f32,
    },
    RemoveObstacle {
        id: Option<String>,
    },
    AddObstacle {
        x_m: f32,
        y_m: f32,
        radius_m: f32,
    },
    SetBattery {
        value: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CounterfactualObjectKind {
    Charger,
    Person,
    Speaker,
}

#[derive(Clone, Debug, PartialEq)]
enum CounterfactualPolicy {
    Baseline,
    Stop,
    TurnLeftOnDanger,
    TurnRightOnDanger,
    SeekCharge,
    RandomWalk { seed: u64 },
    Scripted(Vec<ActionPrimitive>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct CounterfactualReport {
    schema_version: u32,
    source_capture: String,
    reconstructable: bool,
    edits: Vec<String>,
    policy: String,
    steps: usize,
    summary: CounterfactualSummary,
    warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct CounterfactualSummary {
    collisions: usize,
    charging_ticks: usize,
    battery_delta: f32,
    distance_traveled: f32,
    final_distance_to_charger_m: Option<f32>,
}

#[derive(Clone, Debug)]
struct CounterfactualConductor {
    policy: CounterfactualPolicy,
    baseline: SimpleConductor,
    rng: StdRng,
    scripted_index: usize,
}

impl CounterfactualConductor {
    fn new(policy: CounterfactualPolicy) -> Self {
        let seed = match policy {
            CounterfactualPolicy::RandomWalk { seed } => seed,
            _ => 0,
        };
        Self {
            policy,
            baseline: SimpleConductor::default(),
            rng: StdRng::seed_from_u64(seed),
            scripted_index: 0,
        }
    }
}

impl Conductor for CounterfactualConductor {
    fn choose(&mut self, input: ConductorInput) -> Result<ActionPrimitive> {
        match &self.policy {
            CounterfactualPolicy::Baseline => self.baseline.choose(input),
            CounterfactualPolicy::Stop => Ok(ActionPrimitive::Stop),
            CounterfactualPolicy::TurnLeftOnDanger => {
                if danger_present(&input) {
                    Ok(turn_action(TurnDir::Left))
                } else {
                    self.baseline.choose(input)
                }
            }
            CounterfactualPolicy::TurnRightOnDanger => {
                if danger_present(&input) {
                    Ok(turn_action(TurnDir::Right))
                } else {
                    self.baseline.choose(input)
                }
            }
            CounterfactualPolicy::SeekCharge => Ok(ActionPrimitive::Approach {
                target: ApproachTarget::Charger,
            }),
            CounterfactualPolicy::RandomWalk { .. } => {
                if self.rng.gen_bool(0.25) || danger_present(&input) {
                    let direction = if self.rng.gen_bool(0.5) {
                        TurnDir::Left
                    } else {
                        TurnDir::Right
                    };
                    Ok(turn_action(direction))
                } else {
                    Ok(ActionPrimitive::Go {
                        intensity: 0.25,
                        duration_ms: 1_000,
                    })
                }
            }
            CounterfactualPolicy::Scripted(actions) => {
                let action = actions
                    .get(self.scripted_index)
                    .cloned()
                    .unwrap_or(ActionPrimitive::Stop);
                self.scripted_index = self.scripted_index.saturating_add(1);
                Ok(action)
            }
        }
    }
}

async fn replay_counterfactual(args: ReplayCounterfactualArgs) -> Result<()> {
    let reader = CaptureReader::open(&args.capture).await?;
    let manifest = reader.manifest().clone();
    let Some(mut metadata) = manifest.scenario.clone() else {
        anyhow::bail!(
            "passive captures without reconstructable sim metadata cannot yet be counterfactually replayed"
        );
    };
    let frames = reader.read_frames().await?;
    let steps = args.steps.unwrap_or(frames.len()).max(1);
    let edits = args
        .edit
        .iter()
        .map(|edit| parse_counterfactual_edit(edit))
        .collect::<Result<Vec<_>>>()?;
    let mut warnings = Vec::new();
    apply_counterfactual_edits(&mut metadata, &edits, &mut warnings)?;
    let policy = parse_counterfactual_policy(&args.policy, args.actions.as_deref())?;

    let (mut world, motors) =
        netherwick_sim::VirtualWorld::new_with_motor(metadata.seed, metadata.arena);
    world.set_body(metadata.body.clone());
    world.set_objects(metadata.objects.clone());

    if let Some(ledger_path) = &args.out_ledger {
        let runtime =
            counterfactual_runtime(JsonlLedger::new(ledger_path), policy.clone(), &args.llm)?;
        let report = run_counterfactual_sim(
            runtime, world, motors, &metadata, &manifest, &args, steps, policy, warnings,
        )
        .await?;
        write_or_print_counterfactual_report(&args, &report)?;
        let transitions = JsonlLedger::new(ledger_path).transitions().await?;
        println!(
            "counterfactual replay complete: {} steps, ledger {}, transitions {}, report {}",
            steps,
            ledger_path,
            transitions.len(),
            args.out_report.as_deref().unwrap_or("stdout")
        );
    } else {
        let runtime = counterfactual_runtime(NoopLedger, policy.clone(), &args.llm)?;
        let report = run_counterfactual_sim(
            runtime, world, motors, &metadata, &manifest, &args, steps, policy, warnings,
        )
        .await?;
        write_or_print_counterfactual_report(&args, &report)?;
        println!(
            "counterfactual replay complete: {} steps, report {}",
            steps,
            args.out_report.as_deref().unwrap_or("stdout")
        );
    }
    Ok(())
}

fn counterfactual_runtime<L>(
    ledger: L,
    policy: CounterfactualPolicy,
    llm: &LlmArgs,
) -> Result<
    MinimalRuntime<
        L,
        InMemoryExperienceStore,
        InMemoryExperienceStore,
        CounterfactualConductor,
        SimpleSafety,
        ConfiguredLlmAgent,
    >,
>
where
    L: LedgerWriter + Sync + Send,
{
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    Ok(MinimalRuntime::with_default_events(
        ledger,
        memory,
        recall,
        CounterfactualConductor::new(policy),
        SimpleSafety::default(),
        configured_llm_agent(llm)?,
    ))
}

async fn run_counterfactual_sim<R>(
    runtime: R,
    world: netherwick_sim::VirtualWorld,
    motors: netherwick_sim::SimMotorComplex,
    metadata: &netherwick_sim::ScenarioMetadata,
    manifest: &netherwick_worldlab::CaptureManifest,
    args: &ReplayCounterfactualArgs,
    steps: usize,
    policy: CounterfactualPolicy,
    warnings: Vec<String>,
) -> Result<CounterfactualReport>
where
    R: RuntimeLoop + Send,
{
    let mut metrics = EpisodeMetricBuilder::new(
        metadata.kind,
        metadata.clone(),
        0,
        metadata.seed,
        args.out_ledger.clone(),
        Some(args.capture.clone()),
    );
    let mut runner = SimRunner::new(runtime, world, motors);
    runner.tick_ms = manifest.tick_ms.unwrap_or(100);
    runner
        .run_steps_observing_ticks(steps, |snapshot, tick| metrics.observe(snapshot, tick))
        .await?;
    let episode = metrics.finish();
    Ok(CounterfactualReport {
        schema_version: 1,
        source_capture: args.capture.clone(),
        reconstructable: true,
        edits: args.edit.clone(),
        policy: counterfactual_policy_label(&policy),
        steps: episode.ticks,
        summary: CounterfactualSummary {
            collisions: episode.collisions,
            charging_ticks: episode.charging_ticks,
            battery_delta: episode.battery_delta,
            distance_traveled: episode.distance_traveled_m,
            final_distance_to_charger_m: episode.final_distance_to_charger_m,
        },
        warnings,
    })
}

fn write_or_print_counterfactual_report(
    args: &ReplayCounterfactualArgs,
    report: &CounterfactualReport,
) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(report)?;
    if let Some(out) = &args.out_report {
        if let Some(parent) = Path::new(out).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(out, bytes)?;
    } else {
        println!("{}", String::from_utf8_lossy(&bytes));
    }
    Ok(())
}

fn parse_counterfactual_policy(
    policy: &str,
    actions: Option<&str>,
) -> Result<CounterfactualPolicy> {
    if let Some(actions) = actions {
        return Ok(CounterfactualPolicy::Scripted(parse_scripted_actions(
            actions,
        )?));
    }
    if policy == "baseline" {
        Ok(CounterfactualPolicy::Baseline)
    } else if policy == "stop" {
        Ok(CounterfactualPolicy::Stop)
    } else if policy == "turn-left-on-danger" {
        Ok(CounterfactualPolicy::TurnLeftOnDanger)
    } else if policy == "turn-right-on-danger" {
        Ok(CounterfactualPolicy::TurnRightOnDanger)
    } else if policy == "seek-charge" {
        Ok(CounterfactualPolicy::SeekCharge)
    } else if policy == "random-walk" {
        Ok(CounterfactualPolicy::RandomWalk { seed: 0 })
    } else if let Some(rest) = policy.strip_prefix("random-walk:seed=") {
        Ok(CounterfactualPolicy::RandomWalk {
            seed: rest.parse().context("invalid random-walk seed")?,
        })
    } else {
        anyhow::bail!("unknown counterfactual policy '{policy}'")
    }
}

fn counterfactual_policy_label(policy: &CounterfactualPolicy) -> String {
    match policy {
        CounterfactualPolicy::Baseline => "baseline".to_string(),
        CounterfactualPolicy::Stop => "stop".to_string(),
        CounterfactualPolicy::TurnLeftOnDanger => "turn-left-on-danger".to_string(),
        CounterfactualPolicy::TurnRightOnDanger => "turn-right-on-danger".to_string(),
        CounterfactualPolicy::SeekCharge => "seek-charge".to_string(),
        CounterfactualPolicy::RandomWalk { seed } => format!("random-walk:seed={seed}"),
        CounterfactualPolicy::Scripted(_) => "scripted".to_string(),
    }
}

fn parse_scripted_actions(actions: &str) -> Result<Vec<ActionPrimitive>> {
    actions
        .split(',')
        .map(|token| match token.trim() {
            "forward" | "go" => Ok(ActionPrimitive::Go {
                intensity: 0.25,
                duration_ms: 1_000,
            }),
            "left" => Ok(turn_action(TurnDir::Left)),
            "right" => Ok(turn_action(TurnDir::Right)),
            "stop" => Ok(ActionPrimitive::Stop),
            "dock" => Ok(ActionPrimitive::Dock),
            "wander" | "random-walk" => Ok(ActionPrimitive::Explore {
                style: ExploreStyle::RandomWalk,
                duration_ms: 1_000,
            }),
            other => anyhow::bail!("unknown scripted action '{other}'"),
        })
        .collect()
}

fn turn_action(direction: TurnDir) -> ActionPrimitive {
    ActionPrimitive::Turn {
        direction,
        intensity: 0.6,
        duration_ms: 1_000,
    }
}

fn danger_present(input: &ConductorInput) -> bool {
    input.body.flags.bump_left
        || input.body.flags.bump_right
        || input.body.flags.wall
        || input.body.flags.cliff_left
        || input.body.flags.cliff_front_left
        || input.body.flags.cliff_front_right
        || input.body.flags.cliff_right
        || input.drives.danger_avoidance >= 0.5
}

fn parse_counterfactual_edit(input: &str) -> Result<CounterfactualEdit> {
    let (name, rest) = input
        .split_once(':')
        .map(|(name, rest)| (name.trim(), rest.trim()))
        .unwrap_or((input.trim(), ""));
    let fields = parse_edit_fields(rest)?;
    match name {
        "move-charger" => parse_move_edit(CounterfactualObjectKind::Charger, fields),
        "move-person" => parse_move_edit(CounterfactualObjectKind::Person, fields),
        "move-speaker" => parse_move_edit(CounterfactualObjectKind::Speaker, fields),
        "remove-obstacle" => Ok(CounterfactualEdit::RemoveObstacle {
            id: fields.get("id").cloned(),
        }),
        "add-obstacle" => Ok(CounterfactualEdit::AddObstacle {
            x_m: required_f32(&fields, "x")?,
            y_m: required_f32(&fields, "y")?,
            radius_m: required_f32(&fields, "radius")?,
        }),
        "set-battery" => Ok(CounterfactualEdit::SetBattery {
            value: required_f32(&fields, "value")?.clamp(0.0, 1.0),
        }),
        _ => anyhow::bail!("unknown counterfactual edit '{name}'"),
    }
}

fn parse_move_edit(
    kind: CounterfactualObjectKind,
    fields: HashMap<String, String>,
) -> Result<CounterfactualEdit> {
    Ok(CounterfactualEdit::MoveObject {
        kind,
        id: fields.get("id").cloned(),
        x_m: required_f32(&fields, "x")?,
        y_m: required_f32(&fields, "y")?,
    })
}

fn parse_edit_fields(input: &str) -> Result<HashMap<String, String>> {
    let mut fields = HashMap::new();
    if input.trim().is_empty() {
        return Ok(fields);
    }
    for part in input.split(',') {
        let (key, value) = part
            .split_once('=')
            .with_context(|| format!("invalid edit field '{part}', expected key=value"))?;
        fields.insert(key.trim().to_string(), value.trim().to_string());
    }
    Ok(fields)
}

fn required_f32(fields: &HashMap<String, String>, key: &str) -> Result<f32> {
    fields
        .get(key)
        .with_context(|| format!("missing required edit field '{key}'"))?
        .parse()
        .with_context(|| format!("invalid float for edit field '{key}'"))
}

fn apply_counterfactual_edits(
    metadata: &mut netherwick_sim::ScenarioMetadata,
    edits: &[CounterfactualEdit],
    warnings: &mut Vec<String>,
) -> Result<()> {
    for edit in edits {
        match edit {
            CounterfactualEdit::MoveObject { kind, id, x_m, y_m } => {
                let object = find_counterfactual_object_mut(&mut metadata.objects, *kind, id)?;
                object.x_m = *x_m;
                object.y_m = *y_m;
                if id.is_none() {
                    warnings.push(format!(
                        "{} edit used first matching object because no id was provided",
                        object_kind_label(*kind)
                    ));
                }
            }
            CounterfactualEdit::RemoveObstacle { id } => {
                let index = metadata
                    .objects
                    .iter()
                    .position(|object| {
                        matches!(object.kind, netherwick_sim::SimObjectKind::Obstacle)
                            && id.as_ref().map(|id| id == &object.id).unwrap_or(true)
                    })
                    .with_context(|| {
                        if let Some(id) = id {
                            format!("obstacle '{id}' not found")
                        } else {
                            "no obstacle found to remove".to_string()
                        }
                    })?;
                metadata.objects.remove(index);
                if id.is_none() {
                    warnings.push(
                        "remove-obstacle edit used first obstacle because no id was provided"
                            .to_string(),
                    );
                }
            }
            CounterfactualEdit::AddObstacle { x_m, y_m, radius_m } => {
                let index = metadata
                    .objects
                    .iter()
                    .filter(|object| matches!(object.kind, netherwick_sim::SimObjectKind::Obstacle))
                    .count();
                metadata.objects.push(netherwick_sim::SimObject::obstacle(
                    format!("counterfactual-obstacle-{index}"),
                    format!("counterfactual obstacle {index}"),
                    *x_m,
                    *y_m,
                    *radius_m,
                ));
            }
            CounterfactualEdit::SetBattery { value } => {
                metadata.body.battery_level = *value;
                metadata.body.charging = false;
            }
        }
    }
    Ok(())
}

fn find_counterfactual_object_mut<'a>(
    objects: &'a mut [netherwick_sim::SimObject],
    kind: CounterfactualObjectKind,
    id: &Option<String>,
) -> Result<&'a mut netherwick_sim::SimObject> {
    objects
        .iter_mut()
        .find(|object| {
            object_matches_counterfactual_kind(&object.kind, kind)
                && id.as_ref().map(|id| id == &object.id).unwrap_or(true)
        })
        .with_context(|| {
            if let Some(id) = id {
                format!("{} '{id}' not found", object_kind_label(kind))
            } else {
                format!("no {} found", object_kind_label(kind))
            }
        })
}

fn object_matches_counterfactual_kind(
    kind: &netherwick_sim::SimObjectKind,
    edit_kind: CounterfactualObjectKind,
) -> bool {
    match edit_kind {
        CounterfactualObjectKind::Charger => matches!(kind, netherwick_sim::SimObjectKind::Charger),
        CounterfactualObjectKind::Person => {
            matches!(kind, netherwick_sim::SimObjectKind::Person { .. })
        }
        CounterfactualObjectKind::Speaker => {
            matches!(kind, netherwick_sim::SimObjectKind::SoundSource { .. })
        }
    }
}

fn object_kind_label(kind: CounterfactualObjectKind) -> &'static str {
    match kind {
        CounterfactualObjectKind::Charger => "charger",
        CounterfactualObjectKind::Person => "person",
        CounterfactualObjectKind::Speaker => "speaker",
    }
}

async fn hardware_env(args: HardwareEnvArgs) -> Result<()> {
    let report = collect_hardware_env_report().await;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("hardware environment");
    println!("  os: {}", report["os"].as_str().unwrap_or("unknown"));
    println!(
        "  architecture: {}",
        report["architecture"].as_str().unwrap_or("unknown")
    );
    println!(
        "  cpu: {}",
        report["cpu_model"].as_str().unwrap_or("unknown")
    );
    println!(
        "  memory: {} kB",
        report["memory_total_kb"].as_u64().unwrap_or(0)
    );
    println!(
        "  raspberry-pi-like: {}",
        report["raspberry_pi_like"].as_bool().unwrap_or(false)
    );
    print_json_list("  create serial candidates", &report["serial_devices"]);
    print_json_list("  cameras", &report["camera_devices"]);
    print_json_list("  audio inputs", &report["audio_input_devices"]);
    println!(
        "  libfreenect/freenect: {}",
        report["kinect"]["freenect_available"]
            .as_bool()
            .unwrap_or(false)
    );
    println!("  data dirs writable:");
    if let Some(object) = report["data_dirs_writable"].as_object() {
        for (path, writable) in object {
            println!("    {path}: {}", writable.as_bool().unwrap_or(false));
        }
    }
    print_json_list("  warnings", &report["warnings"]);
    Ok(())
}

async fn capture_real(args: CaptureRealArgs) -> Result<()> {
    if args.duration_seconds == 0 {
        anyhow::bail!("--duration-seconds must be greater than zero");
    }

    let env_report = collect_hardware_env_report().await;
    let mut warnings = Vec::new();
    let mut device_availability = serde_json::json!({
        "mock": args.mock,
        "create": null,
        "camera": null,
        "microphone": null,
        "imu": null,
        "gps": null,
        "kinect": env_report["kinect"].clone(),
    });

    let create_port = if args.create_port == "auto" {
        env_report["serial_devices"]
            .as_array()
            .and_then(|devices| devices.first())
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    } else {
        Some(args.create_port.clone())
    };

    let body: Box<dyn RobotBody + Send> = if args.mock || create_port.as_deref() == Some("mock") {
        device_availability["create"] = serde_json::json!({"present": true, "source": "mock"});
        Box::new(MockCreate1Body::new())
    } else if let Some(create_port) = create_port {
        match Create1Body::connect(&create_port, args.create_baud).await {
            Ok(body) => {
                device_availability["create"] = serde_json::json!({
                    "present": true,
                    "port": create_port,
                    "baud": args.create_baud
                });
                Box::new(body)
            }
            Err(error) => {
                anyhow::bail!("failed to open Create serial device {create_port}: {error}");
            }
        }
    } else {
        warnings
            .push("Create serial device not found; no body hardware stream available".to_string());
        device_availability["create"] =
            serde_json::json!({"present": false, "reason": "no serial candidate"});
        Box::new(MockCreate1Body::new())
    };

    let mut sensors: Vec<Box<dyn SenseProducer + Send>> = Vec::new();
    if args.mock {
        sensors.push(Box::new(MockEyeProducer::default()));
        sensors.push(Box::new(MockEarProducer::default()));
        sensors.push(Box::new(MockRangeProducer::default()));
        sensors.push(Box::new(MockKinectProducer::default()));
        device_availability["camera"] = serde_json::json!({"present": true, "source": "mock"});
        device_availability["microphone"] = serde_json::json!({"present": true, "source": "mock"});
        device_availability["kinect"] = serde_json::json!({"present": true, "source": "mock"});
    } else {
        add_optional_real_sensors(&args, &mut sensors, &mut device_availability, &mut warnings);
    }
    let no_real_create = device_availability["create"]["present"].as_bool() != Some(true);
    if !args.mock && no_real_create && sensors.is_empty() {
        anyhow::bail!(
            "no usable devices found: no Create serial device and no requested sensor initialized"
        );
    }

    let requested_frames = duration_to_steps(args.duration_seconds, args.tick_ms);
    if let Some(ledger_path) = &args.ledger {
        let runtime = default_runtime(JsonlLedger::new(ledger_path), &args.llm)?;
        capture_real_with_runtime(
            args,
            runtime,
            body,
            sensors,
            env_report,
            device_availability,
            warnings,
            requested_frames,
        )
        .await
    } else {
        let runtime = default_noop_runtime(&args.llm)?;
        capture_real_with_runtime(
            args,
            runtime,
            body,
            sensors,
            env_report,
            device_availability,
            warnings,
            requested_frames,
        )
        .await
    }
}

#[allow(clippy::too_many_arguments)]
async fn capture_real_with_runtime<R>(
    args: CaptureRealArgs,
    runtime: R,
    body: Box<dyn RobotBody + Send>,
    sensors: Vec<Box<dyn SenseProducer + Send>>,
    env_report: Value,
    device_availability: Value,
    mut warnings: Vec<String>,
    requested_frames: usize,
) -> Result<()>
where
    R: RuntimeLoop + Send,
{
    let mut runner = RealRobotRunner::new(RobotMode::ReadOnly, body, sensors, runtime);
    runner.tick_ms = args.tick_ms;
    let mut writer =
        CaptureWriter::create(&args.out, CaptureSource::RealRobot, Some(args.tick_ms)).await?;
    {
        let manifest = writer.manifest_mut();
        manifest.machine = Some(machine_info_from_env(&env_report));
        manifest.command_args = std::env::args().collect();
        manifest.device_availability = device_availability;
        manifest
            .notes
            .push("capture-real is capture-only; motors are not commanded".to_string());
        if args.export_rgb || args.export_depth || args.export_audio {
            manifest.notes.push(
                "asset export enabled; frame asset paths are relative to capture root".to_string(),
            );
        }
    }
    if args.export_pointcloud
        && !warnings
            .iter()
            .any(|warning| warning.contains("uncalibrated point cloud"))
    {
        warnings
            .push("uncalibrated point cloud: using approximate placeholder intrinsics".to_string());
    }

    let mut stream_counts = StreamCounts::default();
    let mut events_written = 0usize;
    let mut frame_index = 0u64;
    for _ in 0..requested_frames {
        let (snapshot, tick) = runner.tick_read_only().await?;
        stream_counts.observe(&snapshot);
        if tick
            .frame
            .notes
            .iter()
            .any(|note| note.contains("ReadOnlyActionSuppressed"))
        {
            events_written = events_written.saturating_add(1);
        }
        let export = export_snapshot_assets(
            writer.root(),
            frame_index,
            &snapshot,
            args.export_rgb,
            args.export_depth,
            args.export_audio,
        )?;
        writer
            .append_snapshot_with_assets(
                snapshot.body.last_update_ms,
                snapshot,
                Vec::new(),
                export.assets,
                (!export
                    .metadata
                    .as_object()
                    .map(|m| m.is_empty())
                    .unwrap_or(true))
                .then_some(export.metadata),
            )
            .await?;
        frame_index = frame_index.saturating_add(1);
        tokio::time::sleep(Duration::from_millis(args.tick_ms)).await;
    }

    let streams = stream_counts.streams();
    warnings.extend(stream_counts.warnings());
    if stream_counts.useful_stream_count() == 0 {
        anyhow::bail!("no usable body or sensor streams were captured");
    }
    {
        let manifest = writer.manifest_mut();
        manifest.streams = streams;
        manifest.warnings = warnings.clone();
        if let Some(ledger) = &args.ledger {
            manifest.notes.push(format!("ledger: {ledger}"));
        }
    }
    let manifest = writer.finish().await?;
    println!(
        "capture-real complete: {} frames, out {}, streams {:?}, warnings {}, motor_applied false",
        manifest.frame_count,
        args.out,
        manifest.streams.present,
        manifest.warnings.len()
    );
    if events_written > 0 {
        println!("  read-only motor suppressions observed: {events_written}");
    }
    if args.export_pointcloud {
        capture_assets(CaptureAssetsArgs {
            capture: args.out.clone(),
            pointcloud: true,
            stride: args.pointcloud_stride,
            max_depth_m: 8.0,
        })
        .await?;
    }
    Ok(())
}

async fn capture_assets(args: CaptureAssetsArgs) -> Result<()> {
    if !args.pointcloud {
        anyhow::bail!("no asset conversion requested; pass --pointcloud");
    }
    let root = PathBuf::from(&args.capture);
    let reader = CaptureReader::open(&root).await?;
    let mut manifest = reader.manifest().clone();
    let mut frames = reader.read_frames().await?;
    let mut exported = 0usize;
    for frame in &mut frames {
        if export_pointcloud_for_frame(&root, frame, args.max_depth_m, args.stride)?.is_some() {
            exported = exported.saturating_add(1);
        }
    }
    rewrite_frames(&root, &frames).await?;
    if exported > 0
        && !manifest
            .warnings
            .iter()
            .any(|warning| warning.contains("uncalibrated point cloud"))
    {
        manifest
            .warnings
            .push("uncalibrated point cloud: using approximate placeholder intrinsics".to_string());
    }
    update_manifest(&root, &manifest).await?;
    println!(
        "capture-assets complete: pointcloud {} frames, capture {}, stride {}",
        exported, args.capture, args.stride
    );
    Ok(())
}

async fn inspect_capture(args: InspectCaptureArgs) -> Result<()> {
    let report = inspect_capture_report(&args.path).await?;
    println!("capture: {}", report.path.display());
    println!("  frames: {}", report.frame_count);
    println!("  duration_ms: {}", report.duration_ms.unwrap_or(0));
    println!(
        "  streams present: {}",
        join_or_none(&report.streams_present)
    );
    println!(
        "  streams missing: {}",
        join_or_none(&report.streams_missing)
    );
    println!(
        "  first/last timestamps: {:?} / {:?}",
        report.first_timestamp_ms, report.last_timestamp_ms
    );
    println!("  events: {}", report.event_count);
    println!("  assets:");
    for (kind, count) in &report.asset_counts {
        println!("    {kind}: {count}");
    }
    for detail in &report.asset_details {
        println!("    {detail}");
    }
    println!("  warnings: {}", report.warnings.len());
    for warning in &report.warnings {
        println!("    - {warning}");
    }
    Ok(())
}

async fn run_robot(args: RobotArgs) -> Result<()> {
    if args.mode != RobotModeArg::ReadOnly {
        anyhow::bail!("only --mode read-only is implemented for real robot bring-up");
    }
    let mut sensors: Vec<Box<dyn SenseProducer + Send>> = Vec::new();

    if let Some(device) = &args.camera {
        match CameraSenseProvider::new(device) {
            Ok(provider) => sensors.push(Box::new(provider)),
            Err(err) => {
                if args.require_camera {
                    anyhow::bail!("failed to initialize camera: {err}");
                } else {
                    println!("failed to initialize camera: {err}; continuing without it");
                }
            }
        }
    }

    if let Some(device) = &args.mic {
        let pref_name = if device == "default" {
            None
        } else {
            Some(device.as_str())
        };
        match MicrophoneSenseProvider::new(pref_name) {
            Ok(provider) => sensors.push(Box::new(provider)),
            Err(err) => {
                if args.require_mic {
                    anyhow::bail!("failed to initialize mic: {err}");
                } else {
                    println!("failed to initialize mic: {err}; continuing without it");
                }
            }
        }
    }

    if let Some(device) = &args.gps {
        match GpsSenseProvider::new(device, 9600) {
            Ok(provider) => sensors.push(Box::new(provider)),
            Err(err) => {
                if args.require_gps {
                    anyhow::bail!("failed to initialize gps: {err}");
                } else {
                    println!("failed to initialize gps: {err}; continuing without it");
                }
            }
        }
    }

    if let Some(device) = &args.imu {
        match ImuSenseProvider::new(device) {
            Ok(provider) => sensors.push(Box::new(provider)),
            Err(err) => {
                if args.require_imu {
                    anyhow::bail!("failed to initialize imu: {err}");
                } else {
                    println!("failed to initialize imu: {err}; continuing without it");
                }
            }
        }
    }

    let create_port = if args.create_port == "auto" {
        let env_report = collect_hardware_env_report().await;
        env_report["serial_devices"]
            .as_array()
            .and_then(|devices| devices.first())
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    } else {
        Some(args.create_port.clone())
    };
    let body: Box<dyn RobotBody + Send> = if create_port.as_deref() == Some("mock") {
        Box::new(MockCreate1Body::new())
    } else if let Some(create_port) = create_port {
        Box::new(Create1Body::connect(&create_port, args.create_baud).await?)
    } else {
        anyhow::bail!(
            "no Create serial device found; pass --create-port /dev/ttyUSB0 or --create-port mock"
        );
    };
    let ledger = JsonlLedger::new(&args.ledger);
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let runtime = MinimalRuntime::with_default_events(
        ledger.clone(),
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        configured_llm_agent(&args.llm)?,
    );
    let mut runner = RealRobotRunner::new(RobotMode::ReadOnly, body, sensors, runtime);
    runner.tick_ms = args.tick_ms;

    let live_state = args.dashboard.map(|addr| {
        let live_state = LiveViewState::new();
        let server_state = live_state.clone();
        tokio::spawn(async move {
            if let Err(error) = netherwick_server::serve_live_view(addr, server_state).await {
                eprintln!("live robot view server stopped: {error}");
            }
        });
        println!("read-only robot dashboard: http://{addr}/view");
        live_state
    });

    let mut capture = match &args.capture {
        Some(path) => {
            Some(CaptureWriter::create(path, CaptureSource::RealRobot, Some(args.tick_ms)).await?)
        }
        None => None,
    };

    let max_steps = args.steps.or_else(|| {
        args.duration_seconds
            .map(|seconds| duration_to_steps(seconds, args.tick_ms))
    });
    while max_steps
        .map(|limit| runner.tick_count < limit)
        .unwrap_or(true)
    {
        let (snapshot, tick) = runner.tick_read_only().await?;
        if let Some(live_state) = &live_state {
            live_state.update(snapshot.clone());
        }
        if let Some(writer) = capture.as_mut() {
            writer
                .append_snapshot(snapshot.body.last_update_ms, snapshot.clone(), Vec::new())
                .await?;
        }
        println!(
            "robot read-only tick {}: battery {:.2}, chosen {:?}, motor_applied false",
            runner.tick_count, tick.frame.now.body.battery_level, tick.chosen_action
        );
        tokio::time::sleep(Duration::from_millis(args.tick_ms)).await;
    }

    let capture_summary = if let Some(writer) = capture {
        let manifest = writer.finish().await?;
        format!(
            ", capture {}, {} frames",
            args.capture.as_deref().unwrap_or_default(),
            manifest.frame_count
        )
    } else {
        String::new()
    };
    let transitions = ledger.transitions().await?;
    println!(
        "robot read-only complete: {} ticks, ledger {}, {} transitions{}, motor_applied false",
        runner.tick_count,
        args.ledger,
        transitions.len(),
        capture_summary
    );
    Ok(())
}

fn default_runtime(
    ledger: JsonlLedger,
    llm_args: &LlmArgs,
) -> Result<
    MinimalRuntime<
        JsonlLedger,
        InMemoryExperienceStore,
        InMemoryExperienceStore,
        SimpleConductor,
        SimpleSafety,
        ConfiguredLlmAgent,
    >,
> {
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    Ok(MinimalRuntime::with_default_events(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        configured_llm_agent(llm_args)?,
    ))
}

fn duration_to_steps(duration_seconds: u64, tick_ms: u64) -> usize {
    let tick_ms = tick_ms.max(1);
    let total_ms = duration_seconds.saturating_mul(1000);
    total_ms.div_ceil(tick_ms).max(1) as usize
}

fn add_optional_real_sensors(
    args: &CaptureRealArgs,
    sensors: &mut Vec<Box<dyn SenseProducer + Send>>,
    availability: &mut Value,
    warnings: &mut Vec<String>,
) {
    if let Some(device) = &args.camera {
        match CameraSenseProvider::new(device) {
            Ok(provider) => {
                sensors.push(Box::new(provider));
                availability["camera"] = serde_json::json!({"present": true, "device": device});
            }
            Err(error) => {
                availability["camera"] = serde_json::json!({"present": false, "device": device, "error": error.to_string()});
                warnings.push(format!("camera unavailable: {error}"));
            }
        }
    } else {
        availability["camera"] = serde_json::json!({"present": false, "reason": "not requested"});
        warnings.push("camera not requested; RGB stream missing".to_string());
    }

    if let Some(device) = &args.mic {
        let pref_name = (device != "default").then_some(device.as_str());
        match MicrophoneSenseProvider::new(pref_name) {
            Ok(provider) => {
                sensors.push(Box::new(provider));
                availability["microphone"] = serde_json::json!({"present": true, "device": device});
            }
            Err(error) => {
                availability["microphone"] = serde_json::json!({"present": false, "device": device, "error": error.to_string()});
                warnings.push(format!("microphone unavailable: {error}"));
            }
        }
    } else {
        availability["microphone"] =
            serde_json::json!({"present": false, "reason": "not requested"});
        warnings.push("microphone not requested; audio stream missing".to_string());
    }

    if let Some(device) = &args.gps {
        match GpsSenseProvider::new(device, 9600) {
            Ok(provider) => {
                sensors.push(Box::new(provider));
                availability["gps"] = serde_json::json!({"present": true, "device": device});
            }
            Err(error) => {
                availability["gps"] = serde_json::json!({"present": false, "device": device, "error": error.to_string()});
                warnings.push(format!("gps unavailable: {error}"));
            }
        }
    } else {
        availability["gps"] = serde_json::json!({"present": false, "reason": "not requested"});
    }

    if let Some(device) = &args.imu {
        match ImuSenseProvider::new(device) {
            Ok(provider) => {
                sensors.push(Box::new(provider));
                availability["imu"] = serde_json::json!({"present": true, "device": device});
            }
            Err(error) => {
                availability["imu"] = serde_json::json!({"present": false, "device": device, "error": error.to_string()});
                warnings.push(format!("imu unavailable: {error}"));
            }
        }
    } else {
        availability["imu"] = serde_json::json!({"present": false, "reason": "not requested"});
    }

    if availability["kinect"]["freenect_available"].as_bool() != Some(true) {
        warnings.push("Kinect/libfreenect not detected; depth stream missing".to_string());
    }
}

#[derive(Default)]
struct MockEyeProducer {
    tick: u64,
}

#[async_trait::async_trait]
impl SenseProducer for MockEyeProducer {
    async fn poll(&mut self) -> Result<SensePacket> {
        self.tick = self.tick.saturating_add(1);
        let base = (self.tick % 16) as f32 / 16.0;
        let b = (base * 255.0).round() as u8;
        Ok(SensePacket::EyeFrame(EyeFrame {
            captured_at_ms: Utc::now().timestamp_millis().max(0) as u64,
            width: 2,
            height: 2,
            format: EyeFrameFormat::Rgb8,
            bytes: vec![b, 64, 128, 128, b, 64, 64, 128, b, 255, 255, 255],
            source: None,
        }))
    }
}

#[derive(Default)]
struct MockEarProducer {
    tick: u64,
}

#[async_trait::async_trait]
impl SenseProducer for MockEarProducer {
    async fn poll(&mut self) -> Result<SensePacket> {
        self.tick = self.tick.saturating_add(1);
        Ok(if self.tick % 2 == 0 {
            SensePacket::Ear(EarSense {
                schema_version: 1,
                features: vec![vec![0.1, 0.2, 0.1]],
                transcript: None,
                ..EarSense::default()
            })
        } else {
            SensePacket::EarPcm(PcmAudioFrame {
                captured_at_ms: Utc::now().timestamp_millis().max(0) as u64,
                sample_rate_hz: 16_000,
                channels: 1,
                samples: vec![0, 128, -128, 64],
            })
        })
    }
}

#[derive(Default)]
struct MockRangeProducer;

#[async_trait::async_trait]
impl SenseProducer for MockRangeProducer {
    async fn poll(&mut self) -> Result<SensePacket> {
        Ok(SensePacket::Range(RangeSense {
            schema_version: 1,
            beams: vec![1.2, 1.0, 0.8],
            nearest_m: Some(0.8),
        }))
    }
}

#[derive(Default)]
struct MockKinectProducer;

#[async_trait::async_trait]
impl SenseProducer for MockKinectProducer {
    async fn poll(&mut self) -> Result<SensePacket> {
        Ok(SensePacket::Kinect(KinectSense {
            schema_version: 1,
            color_features: vec![vec![0.2, 0.4, 0.6]],
            depth_m: vec![0.8, 1.0, 1.2],
            audio_angle_rad: Some(0.0),
            audio_confidence: 0.75,
            ..KinectSense::default()
        }))
    }
}

#[derive(Clone, Debug, Default)]
struct NoopLedger;

#[async_trait::async_trait]
impl LedgerWriter for NoopLedger {
    async fn append(&self, _frame: &ExperienceFrame) -> Result<()> {
        Ok(())
    }

    async fn append_transition(&self, _transition: &ExperienceTransition) -> Result<()> {
        Ok(())
    }
}

async fn collect_hardware_env_report() -> Value {
    let serial_devices = list_matching_paths(&["/dev/ttyUSB", "/dev/ttyACM", "/dev/serial/by-id/"]);
    let camera_devices = list_matching_paths(&["/dev/video"]);
    let audio_input_devices = audio_input_devices();
    let warnings = hardware_env_warnings(&serial_devices, &camera_devices, &audio_input_devices);
    serde_json::json!({
        "os": std::env::consts::OS,
        "architecture": std::env::consts::ARCH,
        "cpu_model": cpu_model(),
        "memory_total_kb": memory_total_kb(),
        "serial_devices": serial_devices,
        "camera_devices": camera_devices,
        "audio_input_devices": audio_input_devices,
        "kinect": {
            "freenect_available": command_exists("freenect-glview") || command_exists("freenect-camtest") || pkg_config_exists("libfreenect"),
            "freenect_glview": command_exists("freenect-glview"),
            "freenect_camtest": command_exists("freenect-camtest"),
            "pkg_config_libfreenect": pkg_config_exists("libfreenect"),
        },
        "permissions": {
            "groups": current_groups(),
            "serial_group_hint": "dialout",
            "video_group_hint": "video",
            "audio_group_hint": "audio",
        },
        "data_dirs_writable": {
            "data": directory_writable(Path::new("data")),
            "data/captures/real": directory_writable(Path::new("data/captures/real")),
            "data/ledger/real": directory_writable(Path::new("data/ledger/real")),
        },
        "raspberry_pi_like": raspberry_pi_like(),
        "warnings": warnings,
    })
}

fn hardware_env_warnings(
    serial_devices: &[String],
    camera_devices: &[String],
    audio_input_devices: &[String],
) -> Vec<String> {
    let mut warnings = Vec::new();
    let groups = current_groups();
    if serial_devices.is_empty() {
        warnings.push("no likely Create serial devices found under /dev/ttyUSB*, /dev/ttyACM*, or /dev/serial/by-id".to_string());
    }
    if camera_devices.is_empty() {
        warnings.push("no /dev/video* camera devices found".to_string());
    }
    if audio_input_devices.is_empty() {
        warnings.push("no audio input devices detected by arecord or /proc/asound".to_string());
    }
    for group in ["dialout", "video", "audio"] {
        if !groups.iter().any(|item| item == group) {
            warnings.push(format!(
                "current user is not in `{group}` group; hardware permissions may fail"
            ));
        }
    }
    warnings
}

fn machine_info_from_env(report: &Value) -> Value {
    serde_json::json!({
        "os": report["os"].clone(),
        "architecture": report["architecture"].clone(),
        "cpu_model": report["cpu_model"].clone(),
        "memory_total_kb": report["memory_total_kb"].clone(),
        "raspberry_pi_like": report["raspberry_pi_like"].clone(),
    })
}

fn cpu_model() -> Option<String> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok()?;
    cpuinfo.lines().find_map(|line| {
        line.strip_prefix("Model")
            .or_else(|| line.strip_prefix("model name"))
            .and_then(|line| {
                line.split_once(':')
                    .map(|(_, value)| value.trim().to_string())
            })
            .filter(|value| !value.is_empty())
    })
}

fn memory_total_kb() -> Option<u64> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    meminfo.lines().find_map(|line| {
        line.strip_prefix("MemTotal:")
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|value| value.parse().ok())
    })
}

fn raspberry_pi_like() -> bool {
    let model = fs::read_to_string("/proc/device-tree/model")
        .or_else(|_| fs::read_to_string("/sys/firmware/devicetree/base/model"))
        .unwrap_or_default()
        .to_lowercase();
    model.contains("raspberry pi")
}

fn list_matching_paths(prefixes: &[&str]) -> Vec<String> {
    let mut paths = Vec::new();
    for prefix in prefixes {
        let path = Path::new(prefix);
        if path.is_dir() {
            if let Ok(entries) = fs::read_dir(path) {
                for entry in entries.flatten() {
                    paths.push(entry.path().to_string_lossy().to_string());
                }
            }
            continue;
        }
        let Some(parent) = path.parent() else {
            continue;
        };
        let Some(name_prefix) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if let Ok(entries) = fs::read_dir(parent) {
            for entry in entries.flatten() {
                if entry
                    .file_name()
                    .to_str()
                    .map(|name| name.starts_with(name_prefix))
                    .unwrap_or(false)
                {
                    paths.push(entry.path().to_string_lossy().to_string());
                }
            }
        }
    }
    paths.sort();
    paths.dedup();
    paths
}

fn audio_input_devices() -> Vec<String> {
    if let Ok(output) = ProcessCommand::new("arecord").arg("-l").output() {
        if output.status.success() {
            return String::from_utf8_lossy(&output.stdout)
                .lines()
                .filter(|line| line.trim_start().starts_with("card "))
                .map(|line| line.trim().to_string())
                .collect();
        }
    }
    let proc_asound = Path::new("/proc/asound/cards");
    fs::read_to_string(proc_asound)
        .ok()
        .map(|text| {
            text.lines()
                .filter(|line| line.contains('[') && line.contains(']'))
                .map(|line| line.trim().to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn command_exists(command: &str) -> bool {
    ProcessCommand::new("sh")
        .arg("-c")
        .arg(format!("command -v {command} >/dev/null 2>&1"))
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn pkg_config_exists(package: &str) -> bool {
    ProcessCommand::new("pkg-config")
        .arg("--exists")
        .arg(package)
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn current_groups() -> Vec<String> {
    ProcessCommand::new("id")
        .arg("-nG")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .split_whitespace()
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn directory_writable(path: &Path) -> bool {
    if fs::create_dir_all(path).is_err() {
        return false;
    }
    let probe = path.join(".netherwick-write-test");
    match fs::write(&probe, b"ok") {
        Ok(()) => {
            let _ = fs::remove_file(probe);
            true
        }
        Err(_) => false,
    }
}

fn print_json_list(label: &str, value: &Value) {
    println!("{label}:");
    let Some(items) = value.as_array() else {
        println!("    none");
        return;
    };
    if items.is_empty() {
        println!("    none");
        return;
    }
    for item in items {
        if let Some(text) = item.as_str() {
            println!("    - {text}");
        } else {
            println!("    - {item}");
        }
    }
}

#[derive(Clone, Debug, Default)]
struct StreamCounts {
    body: usize,
    rgb: usize,
    depth: usize,
    audio: usize,
    range: usize,
    imu: usize,
    gps: usize,
    kinect: usize,
}

impl StreamCounts {
    fn observe(&mut self, snapshot: &WorldSnapshot) {
        self.body = self.body.saturating_add(1);
        if snapshot.eye_frame.is_some() || !snapshot.eye.frames.is_empty() {
            self.rgb = self.rgb.saturating_add(1);
        }
        if snapshot.ear_pcm.is_some() || !snapshot.ear.features.is_empty() {
            self.audio = self.audio.saturating_add(1);
        }
        if !snapshot.kinect.depth_m.is_empty() {
            self.depth = self.depth.saturating_add(1);
        }
        if !snapshot.range.beams.is_empty() || snapshot.range.nearest_m.is_some() {
            self.range = self.range.saturating_add(1);
        }
        if !snapshot.imu.orientation.is_empty()
            || !snapshot.imu.acceleration.is_empty()
            || !snapshot.imu.angular_velocity.is_empty()
        {
            self.imu = self.imu.saturating_add(1);
        }
        if snapshot.gps.is_some() {
            self.gps = self.gps.saturating_add(1);
        }
        if !snapshot.kinect.color_features.is_empty()
            || !snapshot.kinect.depth_m.is_empty()
            || !snapshot.kinect.skeletons.is_empty()
        {
            self.kinect = self.kinect.saturating_add(1);
        }
    }

    fn streams(&self) -> CaptureStreams {
        let all = [
            ("body", self.body),
            ("rgb", self.rgb),
            ("depth", self.depth),
            ("audio", self.audio),
            ("range", self.range),
            ("imu", self.imu),
            ("gps", self.gps),
            ("kinect", self.kinect),
        ];
        CaptureStreams {
            present: all
                .iter()
                .filter(|(_, count)| *count > 0)
                .map(|(name, _)| (*name).to_string())
                .collect(),
            missing: all
                .iter()
                .filter(|(_, count)| *count == 0)
                .map(|(name, _)| (*name).to_string())
                .collect(),
        }
    }

    fn warnings(&self) -> Vec<String> {
        self.streams()
            .missing
            .into_iter()
            .filter(|name| name != "gps" && name != "imu")
            .map(|name| format!("{name} stream missing"))
            .collect()
    }

    fn useful_stream_count(&self) -> usize {
        [
            self.body,
            self.rgb,
            self.depth,
            self.audio,
            self.range,
            self.imu,
            self.gps,
            self.kinect,
        ]
        .into_iter()
        .filter(|count| *count > 0)
        .count()
    }
}

#[derive(Clone, Debug)]
struct CaptureInspectionReport {
    path: PathBuf,
    frame_count: usize,
    duration_ms: Option<u64>,
    streams_present: Vec<String>,
    streams_missing: Vec<String>,
    first_timestamp_ms: Option<u64>,
    last_timestamp_ms: Option<u64>,
    event_count: usize,
    asset_counts: Vec<(String, usize)>,
    asset_details: Vec<String>,
    warnings: Vec<String>,
}

async fn inspect_capture_report(path: impl AsRef<Path>) -> Result<CaptureInspectionReport> {
    let path = path.as_ref().to_path_buf();
    let reader = CaptureReader::open(&path).await?;
    let frames = reader.read_frames().await?;
    let mut stream_counts = StreamCounts::default();
    let mut event_count = 0usize;
    for frame in &frames {
        stream_counts.observe(&frame.snapshot);
        event_count = event_count.saturating_add(frame.events.len());
    }
    event_count = event_count.saturating_add(count_jsonl_lines(&path.join("events.jsonl"))?);
    let first_timestamp_ms = frames.first().map(|frame| frame.t_ms);
    let last_timestamp_ms = frames.last().map(|frame| frame.t_ms);
    let duration_ms = first_timestamp_ms
        .zip(last_timestamp_ms)
        .map(|(first, last)| last.saturating_sub(first));
    let streams = if reader.manifest().streams.present.is_empty()
        && reader.manifest().streams.missing.is_empty()
    {
        stream_counts.streams()
    } else {
        reader.manifest().streams.clone()
    };
    Ok(CaptureInspectionReport {
        path: path.clone(),
        frame_count: frames.len(),
        duration_ms,
        streams_present: streams.present,
        streams_missing: streams.missing,
        first_timestamp_ms,
        last_timestamp_ms,
        event_count,
        asset_counts: asset_counts(&path),
        asset_details: asset_details(&frames),
        warnings: reader.manifest().warnings.clone(),
    })
}

fn asset_details(frames: &[netherwick_worldlab::CaptureFrameRecord]) -> Vec<String> {
    let mut details = Vec::new();
    let mut seen = BTreeSet::new();
    for frame in frames {
        let Some(metadata) = frame.stream_metadata.as_ref().and_then(Value::as_object) else {
            continue;
        };
        for kind in ["rgb", "depth", "audio", "pointcloud"] {
            if seen.contains(kind) {
                continue;
            }
            let Some(value) = metadata.get(kind).and_then(Value::as_object) else {
                continue;
            };
            let detail = match kind {
                "rgb" | "depth" => format!(
                    "{kind} metadata: {}x{}, {}",
                    value.get("width").and_then(Value::as_u64).unwrap_or(0),
                    value.get("height").and_then(Value::as_u64).unwrap_or(0),
                    value
                        .get("format")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                ),
                "audio" => format!(
                    "audio metadata: {} Hz, {} channel(s), {}",
                    value
                        .get("sample_rate_hz")
                        .and_then(Value::as_u64)
                        .unwrap_or(0),
                    value.get("channels").and_then(Value::as_u64).unwrap_or(0),
                    value
                        .get("format")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                ),
                "pointcloud" => format!(
                    "pointcloud metadata: {} vertices, {}, calibration {}",
                    value.get("vertices").and_then(Value::as_u64).unwrap_or(0),
                    value
                        .get("format")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown"),
                    value
                        .get("calibration")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                ),
                _ => continue,
            };
            seen.insert(kind);
            details.push(detail);
        }
    }
    details
}

fn asset_counts(root: &Path) -> Vec<(String, usize)> {
    ["rgb", "depth", "audio", "pointcloud"]
        .into_iter()
        .map(|kind| {
            let path = root.join("assets").join(kind);
            (kind.to_string(), count_files(&path))
        })
        .collect()
}

fn count_files(path: &Path) -> usize {
    fs::read_dir(path)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.flatten())
        .filter(|entry| {
            entry
                .file_type()
                .map(|kind| kind.is_file())
                .unwrap_or(false)
        })
        .count()
}

fn count_jsonl_lines(path: &Path) -> Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    Ok(fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count())
}

fn join_or_none(items: &[String]) -> String {
    if items.is_empty() {
        "none".to_string()
    } else {
        items.join(", ")
    }
}

#[derive(Clone, Debug)]
struct RuntimeModelFlags<'a> {
    danger_checkpoint: Option<&'a str>,
    danger_mode: DangerMode,
    charge_checkpoint: Option<&'a str>,
    charge_mode: ChargeMode,
    action_value_checkpoint: Option<&'a str>,
    action_value_mode: ActionValueMode,
    future_checkpoint: Option<&'a str>,
    future_mode: FutureMode,
    eye_next_checkpoint: Option<&'a str>,
    eye_next_mode: EyeNextMode,
    ear_next_checkpoint: Option<&'a str>,
    ear_next_mode: EarNextMode,
    experience_checkpoint: Option<&'a str>,
    experience_mode: ExperienceMode,
}

impl<'a> From<&'a SimArgs> for RuntimeModelFlags<'a> {
    fn from(args: &'a SimArgs) -> Self {
        Self {
            danger_checkpoint: args.danger_checkpoint.as_deref(),
            danger_mode: args.danger_mode,
            charge_checkpoint: args.charge_checkpoint.as_deref(),
            charge_mode: args.charge_mode,
            action_value_checkpoint: args.action_value_checkpoint.as_deref(),
            action_value_mode: args.action_value_mode,
            future_checkpoint: args.future_checkpoint.as_deref(),
            future_mode: args.future_mode,
            eye_next_checkpoint: args.eye_next_checkpoint.as_deref(),
            eye_next_mode: args.eye_next_mode,
            ear_next_checkpoint: args.ear_next_checkpoint.as_deref(),
            ear_next_mode: args.ear_next_mode,
            experience_checkpoint: args.experience_checkpoint.as_deref(),
            experience_mode: args.experience_mode,
        }
    }
}

impl<'a> From<&'a EvalScenarioArgs> for RuntimeModelFlags<'a> {
    fn from(args: &'a EvalScenarioArgs) -> Self {
        Self {
            danger_checkpoint: args.danger_checkpoint.as_deref(),
            danger_mode: args.danger_mode,
            charge_checkpoint: args.charge_checkpoint.as_deref(),
            charge_mode: args.charge_mode,
            action_value_checkpoint: args.action_value_checkpoint.as_deref(),
            action_value_mode: args.action_value_mode,
            future_checkpoint: args.future_checkpoint.as_deref(),
            future_mode: args.future_mode,
            eye_next_checkpoint: args.eye_next_checkpoint.as_deref(),
            eye_next_mode: args.eye_next_mode,
            ear_next_checkpoint: args.ear_next_checkpoint.as_deref(),
            ear_next_mode: args.ear_next_mode,
            experience_checkpoint: args.experience_checkpoint.as_deref(),
            experience_mode: args.experience_mode,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct RuntimeModelLoadReport {
    requested_checkpoints: HashMap<String, Option<String>>,
    loaded_checkpoints: HashMap<String, String>,
    active_modes: HashMap<String, String>,
    blocked_model_infer: Vec<String>,
    warnings: Vec<String>,
}

fn load_runtime_models_from_flags(
    flags: &RuntimeModelFlags<'_>,
) -> Result<(Option<RuntimeModelStack>, RuntimeModelLoadReport)> {
    let mut report = RuntimeModelLoadReport::default();
    report.active_modes = model_modes_from_flags(flags);
    report.requested_checkpoints.insert(
        "danger".to_string(),
        flags.danger_checkpoint.map(ToOwned::to_owned),
    );
    report.requested_checkpoints.insert(
        "charge".to_string(),
        flags.charge_checkpoint.map(ToOwned::to_owned),
    );
    report.requested_checkpoints.insert(
        "action_value".to_string(),
        flags.action_value_checkpoint.map(ToOwned::to_owned),
    );
    report.requested_checkpoints.insert(
        "future".to_string(),
        flags.future_checkpoint.map(ToOwned::to_owned),
    );
    report.requested_checkpoints.insert(
        "eye_next".to_string(),
        flags.eye_next_checkpoint.map(ToOwned::to_owned),
    );
    report.requested_checkpoints.insert(
        "ear_next".to_string(),
        flags.ear_next_checkpoint.map(ToOwned::to_owned),
    );
    report.requested_checkpoints.insert(
        "experience".to_string(),
        flags.experience_checkpoint.map(ToOwned::to_owned),
    );

    if flags.danger_mode != DangerMode::ShadowInfer
        && flags.charge_mode != ChargeMode::ShadowInfer
        && flags.action_value_mode != ActionValueMode::ShadowInfer
        && flags.future_mode == FutureMode::Hardcoded
        && flags.eye_next_mode != EyeNextMode::ShadowInfer
        && flags.ear_next_mode != EarNextMode::ShadowInfer
        && flags.experience_mode == ExperienceMode::Off
    {
        return Ok((None, report));
    }
    let mut checkpoint_path = |behavior: &str, checkpoint: Option<&str>, enabled: bool| {
        if !enabled {
            return None;
        }
        match checkpoint {
            Some(checkpoint) if Path::new(checkpoint).exists() => {
                let path = PathBuf::from(checkpoint);
                println!("loaded {behavior} checkpoint: {}", path.display());
                report
                    .loaded_checkpoints
                    .insert(behavior.to_string(), checkpoint.to_string());
                Some(path)
            }
            Some(checkpoint) => {
                let warning =
                    format!("{behavior} inference disabled: checkpoint not found at {checkpoint}");
                println!("{warning}");
                report.warnings.push(warning);
                None
            }
            None => {
                let warning =
                    format!("{behavior} inference disabled: no --{behavior}-checkpoint provided");
                println!("{warning}");
                report.warnings.push(warning);
                None
            }
        }
    };
    let danger_path = checkpoint_path(
        "danger",
        flags.danger_checkpoint,
        flags.danger_mode == DangerMode::ShadowInfer,
    );
    let charge_path = checkpoint_path(
        "charge",
        flags.charge_checkpoint,
        flags.charge_mode == ChargeMode::ShadowInfer,
    );
    let action_value_path = checkpoint_path(
        "action_value",
        flags.action_value_checkpoint,
        flags.action_value_mode == ActionValueMode::ShadowInfer,
    );
    let future_path = checkpoint_path(
        "future",
        flags.future_checkpoint,
        flags.future_mode != FutureMode::Hardcoded,
    );
    let eye_next_path = checkpoint_path(
        "eye_next",
        flags.eye_next_checkpoint,
        flags.eye_next_mode == EyeNextMode::ShadowInfer,
    );
    let ear_next_path = checkpoint_path(
        "ear_next",
        flags.ear_next_checkpoint,
        flags.ear_next_mode == EarNextMode::ShadowInfer,
    );
    let experience_path = checkpoint_path(
        "experience",
        flags.experience_checkpoint,
        flags.experience_mode != ExperienceMode::Off,
    );
    if danger_path.is_none()
        && charge_path.is_none()
        && action_value_path.is_none()
        && future_path.is_none()
        && eye_next_path.is_none()
        && ear_next_path.is_none()
        && experience_path.is_none()
    {
        return Ok((None, report));
    }

    let mut models = RuntimeModelStack::with_shadow_checkpoints(
        danger_path.as_deref(),
        charge_path.as_deref(),
        action_value_path.as_deref(),
        future_path.as_deref(),
        eye_next_path.as_deref(),
        ear_next_path.as_deref(),
        experience_path.as_deref(),
    )?;
    if future_path.is_some() && flags.future_mode == FutureMode::ModelInfer {
        models.behaviors.future.regime = BehaviorRegime::ModelInfer;
    }
    if experience_path.is_some() && flags.experience_mode == ExperienceMode::ModelInfer {
        models.behaviors.experience.regime = BehaviorRegime::ModelInfer;
    }
    Ok((Some(models), report))
}

fn model_modes_from_flags(flags: &RuntimeModelFlags<'_>) -> HashMap<String, String> {
    HashMap::from([
        (
            "danger".to_string(),
            mode_name(flags.danger_mode).to_string(),
        ),
        (
            "charge".to_string(),
            mode_name(flags.charge_mode).to_string(),
        ),
        (
            "action_value".to_string(),
            mode_name(flags.action_value_mode).to_string(),
        ),
        (
            "future".to_string(),
            mode_name(flags.future_mode).to_string(),
        ),
        (
            "eye_next".to_string(),
            mode_name(flags.eye_next_mode).to_string(),
        ),
        (
            "ear_next".to_string(),
            mode_name(flags.ear_next_mode).to_string(),
        ),
        (
            "experience".to_string(),
            mode_name(flags.experience_mode).to_string(),
        ),
    ])
}

fn mode_name<T: std::fmt::Debug>(mode: T) -> &'static str {
    match format!("{mode:?}").as_str() {
        "Off" => "off",
        "Hardcoded" => "hardcoded",
        "ShadowInfer" => "shadow-infer",
        "ModelInfer" => "model-infer",
        _ => "unknown",
    }
}

async fn inspect_ledger(args: InspectLedgerArgs) -> Result<()> {
    let ledger = JsonlLedger::new(&args.ledger);
    let frames = ledger.recent(10).await?;
    if frames.is_empty() {
        println!("ledger is empty");
        return Ok(());
    }

    for frame in frames {
        print_frame(&frame);
    }
    Ok(())
}

async fn memory_inspect(args: MemoryInspectArgs) -> Result<()> {
    let ledger = JsonlLedger::new(&args.ledger);
    let frames = ledger.range(0, u64::MAX).await?;
    let report = place_memory_report_from_frames(&frames);
    print_memory_report(&args.ledger, frames.len(), &report);
    Ok(())
}

fn print_memory_report(source: &str, frame_count: usize, report: &PlaceMemoryReport) {
    println!("memory report: {source}");
    println!("  frames: {frame_count}");
    println!("  places_visited: {}", report.places_visited);
    println!("  coverage_m2: {:.2}", report.coverage_m2);
    println!("  novelty_mean: {:.3}", report.novelty_mean);
    print_place_cells("top danger cells", &report.top_danger_cells);
    print_place_cells("top charge cells", &report.top_charge_cells);
    print_place_cells("top social cells", &report.top_social_cells);
    if report.warnings.is_empty() {
        println!("  warnings: none");
    } else {
        println!("  warnings:");
        for warning in &report.warnings {
            println!("    - {warning}");
        }
    }
}

fn print_place_cells(label: &str, cells: &[netherwick_memory::PlaceCellSummary]) {
    println!("  {label}:");
    if cells.is_empty() {
        println!("    none");
        return;
    }
    for cell in cells {
        println!(
            "    - cell=({}, {}) center=({:.2}, {:.2}) score={:.3} visits={} confidence={:.3}",
            cell.x,
            cell.y,
            cell.center_x_m,
            cell.center_y_m,
            cell.score,
            cell.visit_count,
            cell.confidence
        );
    }
}

async fn run_train(command: TrainCommand) -> Result<()> {
    match command.model {
        TrainModel::Behavior(args) => train_behavior_command(args).await,
        TrainModel::Danger(args) => {
            train_behavior_command(TrainBehaviorArgs {
                behavior: "danger".to_string(),
                ledger: args.ledger,
                epochs: args.epochs,
                checkpoint: Some(args.checkpoint),
                validation_split: 0.2,
                seed: 7,
            })
            .await
        }
        TrainModel::Charge(args) => {
            train_behavior_command(TrainBehaviorArgs {
                behavior: "charge".to_string(),
                ledger: args.ledger,
                epochs: args.epochs,
                checkpoint: Some(args.checkpoint),
                validation_split: 0.2,
                seed: 7,
            })
            .await
        }
        TrainModel::ActionValue(args) => {
            train_behavior_command(TrainBehaviorArgs {
                behavior: "action-value".to_string(),
                ledger: args.ledger,
                epochs: args.epochs,
                checkpoint: Some(args.checkpoint),
                validation_split: 0.2,
                seed: 7,
            })
            .await
        }
        TrainModel::Future(args) => {
            train_behavior_command(TrainBehaviorArgs {
                behavior: "future".to_string(),
                ledger: args.ledger,
                epochs: args.epochs,
                checkpoint: Some(args.checkpoint),
                validation_split: 0.2,
                seed: 7,
            })
            .await
        }
        TrainModel::EyeNext(args) => {
            train_behavior_command(TrainBehaviorArgs {
                behavior: "eye-next".to_string(),
                ledger: args.ledger,
                epochs: args.epochs,
                checkpoint: Some(args.checkpoint),
                validation_split: 0.2,
                seed: 7,
            })
            .await
        }
        TrainModel::EarNext(args) => {
            train_behavior_command(TrainBehaviorArgs {
                behavior: "ear-next".to_string(),
                ledger: args.ledger,
                epochs: args.epochs,
                checkpoint: Some(args.checkpoint),
                validation_split: 0.2,
                seed: 7,
            })
            .await
        }
        TrainModel::Experience(args) => {
            train_behavior_command(TrainBehaviorArgs {
                behavior: "experience".to_string(),
                ledger: args.ledger,
                epochs: args.epochs,
                checkpoint: Some(args.checkpoint),
                validation_split: 0.2,
                seed: 7,
            })
            .await
        }
        TrainModel::Virtual(args) => run_train_virtual(args).await,
    }
}

async fn train_behavior_command(args: TrainBehaviorArgs) -> Result<()> {
    let behavior: TrainableBehavior = args.behavior.parse()?;
    let checkpoint = args
        .checkpoint
        .unwrap_or_else(|| default_checkpoint(&behavior).to_string());
    let summary = train_behavior(TrainBehaviorRequest {
        behavior: behavior.clone(),
        ledger_path: args.ledger.into(),
        checkpoint_path: checkpoint.clone().into(),
        epochs: args.epochs,
        validation_split: args.validation_split,
        seed: args.seed,
    })
    .await?;
    println!(
        "{} training complete: {} transitions, {} train samples, {} eval samples, {} epochs, {} samples seen, metrics {}",
        behavior,
        summary.transition_count,
        summary.train_sample_count,
        summary.eval_sample_count,
        summary.epochs,
        summary.samples_seen,
        summary.metrics_path.display()
    );
    println!(
        "saved {} checkpoint: {}",
        behavior,
        summary.checkpoint_path.display()
    );
    if let Some(last_loss) = summary.last_loss {
        println!("last_loss: {:.6}", last_loss);
    }
    println!("best_loss: {:?}", summary.best_loss);
    print_evaluation_report(&summary.evaluation)?;
    Ok(())
}

async fn run_evaluate(command: EvaluateCommand) -> Result<()> {
    match command.model {
        EvaluateModel::Behavior(args) => {
            let behavior: TrainableBehavior = args.behavior.parse()?;
            let checkpoint = args
                .checkpoint
                .unwrap_or_else(|| default_checkpoint(&behavior).to_string());
            let report = evaluate_behavior(EvaluateBehaviorRequest {
                behavior,
                ledger_path: args.ledger.into(),
                checkpoint_path: checkpoint.clone().into(),
                max_samples: args.max_samples,
            })
            .await?;
            let checkpoint_evaluation_path = Path::new(&checkpoint).join("evaluation.json");
            let json = serde_json::to_string_pretty(&report)?;
            std::fs::write(&checkpoint_evaluation_path, &json)?;
            println!(
                "Saved checkpoint evaluation report to {}",
                checkpoint_evaluation_path.display()
            );
            if let Some(out_path) = &args.out {
                std::fs::write(out_path, &json)?;
                println!("Saved evaluation report to {}", out_path);
            }
            print_evaluation_report(&report)
        }
    }
}

fn run_promote(command: PromoteCommand) -> Result<()> {
    match command.model {
        PromoteModel::Behavior(args) => {
            let behavior: TrainableBehavior = args.behavior.parse()?;
            let checkpoint = args
                .checkpoint
                .unwrap_or_else(|| default_checkpoint(&behavior).to_string());
            let regime = match args.mode {
                PromoteMode::ShadowInfer => BehaviorRegime::ShadowInfer,
                PromoteMode::ModelInfer => BehaviorRegime::ModelInfer,
                PromoteMode::ShadowTrain => BehaviorRegime::ShadowTrain,
            };
            promote_behavior_config(
                behavior.clone(),
                checkpoint.clone().into(),
                Path::new(&args.config),
                regime,
            )?;
            println!(
                "promoted {} in {}: regime {:?}, checkpoint {}",
                behavior, args.config, regime, checkpoint
            );
            Ok(())
        }
    }
}

fn default_checkpoint(behavior: &TrainableBehavior) -> &'static str {
    match behavior {
        TrainableBehavior::Danger => "data/models/danger_v0",
        TrainableBehavior::Charge => "data/models/charge_v0",
        TrainableBehavior::ActionValue => "data/models/action_value_v0",
        TrainableBehavior::EyeNext => "data/models/eye_next_v0",
        TrainableBehavior::EarNext => "data/models/ear_next_v0",
        TrainableBehavior::Experience => "data/models/experience_v0",
        TrainableBehavior::Future => "data/models/future_v0",
    }
}

fn print_evaluation_report(report: &netherwick_training::BehaviorEvaluationReport) -> Result<()> {
    println!("evaluation behavior: {}", report.behavior);
    println!("checkpoint: {}", report.checkpoint_path.display());
    println!("sample_count: {}", report.sample_count);
    println!("model_loss_mean: {:.6}", report.model_loss_mean);
    println!("hardcoded_loss_mean: {:?}", report.hardcoded_loss_mean);
    println!("selected_loss_mean: {:?}", report.selected_loss_mean);
    println!(
        "model_better_than_hardcoded: {:?}",
        report.model_better_than_hardcoded
    );
    println!("improvement_ratio: {:?}", report.improvement_ratio);
    println!("recommendation: {:?}", report.recommendation);
    for warning in &report.warnings {
        println!("warning: {warning}");
    }
    Ok(())
}

const DEFAULT_REGISTRY_PATH: &str = "data/models/registry.json";

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ModelRegistry {
    schema_version: u32,
    entries: Vec<ModelRegistryEntry>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self {
            schema_version: 1,
            entries: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ModelRegistryEntry {
    schema_version: u32,
    name: String,
    behavior: TrainableBehavior,
    checkpoint: String,
    created_at: Option<String>,
    training: ModelTrainingRecord,
    reports: ModelReportRecord,
    scenario_names: Vec<String>,
    metrics: ModelMetricsSummary,
    allowed_modes: Vec<String>,
    status: ModelStatus,
    warnings: Vec<String>,
    notes: Vec<String>,
    parent_model: Option<String>,
    git_commit: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ModelTrainingRecord {
    ledger: Option<String>,
    command: Option<String>,
    epochs: Option<usize>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ModelReportRecord {
    behavior: Option<String>,
    scenario: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ModelMetricsSummary {
    behavior_loss: Option<f32>,
    scenario_success_rate: Option<f32>,
    collision_rate: Option<f32>,
    battery_delta: Option<f32>,
    fallback_count: Option<usize>,
    episodes: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum ModelStatus {
    Registered,
    Shadow,
    Inference,
    Retired,
    Rejected,
}

impl ModelStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Registered => "registered",
            Self::Shadow => "shadow",
            Self::Inference => "inference",
            Self::Retired => "retired",
            Self::Rejected => "rejected",
        }
    }
}

#[derive(Clone, Debug)]
struct ScenarioReportComparison {
    outcome: ComparisonOutcome,
    success_rate_delta: f32,
    collision_rate_delta: f32,
    battery_delta_delta: f32,
    fallback_delta: isize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ComparisonOutcome {
    Improved,
    Regressed,
    Inconclusive,
}

impl ComparisonOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Improved => "improved",
            Self::Regressed => "regressed",
            Self::Inconclusive => "inconclusive",
        }
    }
}

fn model_register(args: ModelRegisterArgs) -> Result<()> {
    let behavior: TrainableBehavior = args.behavior.parse()?;
    let mut registry = load_model_registry(Path::new(&args.registry))?;
    if registry
        .entries
        .iter()
        .any(|entry| entry.name == args.name && entry.behavior == behavior)
        && !args.overwrite
    {
        anyhow::bail!(
            "model {} for {} already exists; pass --overwrite to replace it",
            args.name,
            behavior
        );
    }

    let behavior_report = args
        .behavior_report
        .as_deref()
        .map(load_behavior_report)
        .transpose()?;
    let scenario_report = args
        .scenario_report
        .as_deref()
        .map(load_scenario_report)
        .transpose()?;
    let mut warnings = Vec::new();
    if !Path::new(&args.checkpoint).exists() {
        warnings.push(format!("checkpoint missing: {}", args.checkpoint));
    }
    if let Some(path) = &args.behavior_report {
        if !Path::new(path).exists() {
            warnings.push(format!("behavior report missing: {path}"));
        }
    } else if behavior == TrainableBehavior::Danger {
        warnings.push("danger registration lacks a behavior evaluation report".to_string());
    }
    if let Some(path) = &args.scenario_report {
        if !Path::new(path).exists() {
            warnings.push(format!("scenario report missing: {path}"));
        }
    }

    let metrics = ModelMetricsSummary {
        behavior_loss: behavior_report
            .as_ref()
            .map(|report| report.model_loss_mean),
        scenario_success_rate: scenario_report
            .as_ref()
            .map(|report| report.summary.success_rate),
        collision_rate: scenario_report
            .as_ref()
            .map(|report| report.summary.collision_rate),
        battery_delta: scenario_report
            .as_ref()
            .map(|report| report.summary.mean_battery_delta),
        fallback_count: scenario_report
            .as_ref()
            .map(|report| report.summary.model_fallbacks),
        episodes: scenario_report.as_ref().map(|report| report.episodes),
    };
    let entry = ModelRegistryEntry {
        schema_version: 1,
        name: args.name.clone(),
        behavior: behavior.clone(),
        checkpoint: args.checkpoint,
        created_at: Some(Utc::now().to_rfc3339()),
        training: ModelTrainingRecord {
            ledger: args.training_ledger,
            command: args.training_command.or_else(|| Some(command_summary())),
            epochs: None,
        },
        reports: ModelReportRecord {
            behavior: args.behavior_report,
            scenario: args.scenario_report,
        },
        scenario_names: scenario_report
            .as_ref()
            .map(|report| vec![report.scenario.clone()])
            .unwrap_or_default(),
        metrics,
        allowed_modes: allowed_modes_for_status(&behavior, ModelStatus::Registered),
        status: ModelStatus::Registered,
        warnings,
        notes: args.notes,
        parent_model: args.parent,
        git_commit: current_git_commit(),
    };

    registry
        .entries
        .retain(|entry| !(entry.name == args.name && entry.behavior == behavior));
    registry.entries.push(entry);
    write_model_registry(Path::new(&args.registry), &registry)?;
    println!(
        "registered {} model {} in {}",
        behavior, args.name, args.registry
    );
    Ok(())
}

fn model_promote(args: ModelPromoteArgs) -> Result<()> {
    let behavior: TrainableBehavior = args.behavior.parse()?;
    let path = Path::new(&args.registry);
    let mut registry = load_model_registry(path)?;
    let Some(index) = registry
        .entries
        .iter()
        .position(|entry| entry.name == args.name && entry.behavior == behavior)
    else {
        anyhow::bail!("model {} for {} is not registered", args.name, behavior);
    };

    let candidate_path = args
        .candidate_report
        .clone()
        .or_else(|| registry.entries[index].reports.scenario.clone());
    let baseline_report = args
        .baseline_report
        .as_deref()
        .map(load_scenario_report)
        .transpose()?;
    let candidate_report = candidate_path
        .as_deref()
        .map(load_scenario_report)
        .transpose()?;
    let comparison = match (&baseline_report, &candidate_report) {
        (Some(baseline), Some(candidate)) => Some(compare_scenario_reports(baseline, candidate)),
        _ => None,
    };
    let decision = promotion_gate(
        &registry.entries[index],
        args.target,
        baseline_report.as_ref(),
        candidate_report.as_ref(),
        comparison.as_ref(),
        args.allow_safety_critical_inference,
    );

    if !decision.allowed {
        for warning in decision.warnings {
            println!("warning: {warning}");
        }
        anyhow::bail!(
            "promotion refused: {} {} -> {}",
            behavior,
            args.name,
            args.target.as_str()
        );
    }

    {
        let entry = &mut registry.entries[index];
        entry.status = args.target;
        entry.allowed_modes = allowed_modes_for_status(&behavior, args.target);
        entry.warnings = merge_warnings(&entry.warnings, &decision.warnings);
        entry.notes.extend(args.notes);
        if let Some(path) = args.candidate_report {
            entry.reports.scenario = Some(path);
        }
        if let Some(report) = candidate_report {
            entry.scenario_names = vec![report.scenario.clone()];
            entry.metrics.scenario_success_rate = Some(report.summary.success_rate);
            entry.metrics.collision_rate = Some(report.summary.collision_rate);
            entry.metrics.battery_delta = Some(report.summary.mean_battery_delta);
            entry.metrics.fallback_count = Some(report.summary.model_fallbacks);
            entry.metrics.episodes = Some(report.episodes);
        }
    }
    write_model_registry(path, &registry)?;
    println!(
        "promoted {} model {} to {}",
        behavior,
        args.name,
        args.target.as_str()
    );
    if let Some(comparison) = comparison {
        print_scenario_comparison(&comparison);
    }
    Ok(())
}

fn compare_scenario_reports_command(args: CompareScenarioReportsArgs) -> Result<()> {
    let baseline = load_scenario_report(&args.baseline)?;
    let candidate = load_scenario_report(&args.candidate)?;
    let comparison = compare_scenario_reports(&baseline, &candidate);
    print_scenario_comparison(&comparison);
    Ok(())
}

fn load_model_registry(path: &Path) -> Result<ModelRegistry> {
    if !path.exists() {
        return Ok(ModelRegistry::default());
    }
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn write_model_registry(path: &Path, registry: &ModelRegistry) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temp_path = path.with_extension("json.tmp");
    fs::write(&temp_path, serde_json::to_vec_pretty(registry)?)?;
    fs::rename(temp_path, path)?;
    Ok(())
}

fn load_behavior_report(path: &str) -> Result<netherwick_training::BehaviorEvaluationReport> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn load_scenario_report(path: &str) -> Result<ScenarioEvaluationReport> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn compare_scenario_reports(
    baseline: &ScenarioEvaluationReport,
    candidate: &ScenarioEvaluationReport,
) -> ScenarioReportComparison {
    let success_rate_delta = candidate.summary.success_rate - baseline.summary.success_rate;
    let collision_rate_delta = candidate.summary.collision_rate - baseline.summary.collision_rate;
    let battery_delta_delta =
        candidate.summary.mean_battery_delta - baseline.summary.mean_battery_delta;
    let fallback_delta =
        candidate.summary.model_fallbacks as isize - baseline.summary.model_fallbacks as isize;
    let outcome =
        if baseline.scenario != candidate.scenario || baseline.episodes != candidate.episodes {
            ComparisonOutcome::Inconclusive
        } else if success_rate_delta < -0.01
            || collision_rate_delta > 0.005
            || battery_delta_delta < -0.02
            || fallback_delta > 0
        {
            ComparisonOutcome::Regressed
        } else if success_rate_delta > 0.01
            || collision_rate_delta < -0.005
            || battery_delta_delta > 0.02
        {
            ComparisonOutcome::Improved
        } else {
            ComparisonOutcome::Inconclusive
        };
    ScenarioReportComparison {
        outcome,
        success_rate_delta,
        collision_rate_delta,
        battery_delta_delta,
        fallback_delta,
    }
}

fn print_scenario_comparison(comparison: &ScenarioReportComparison) {
    println!("comparison: {}", comparison.outcome.as_str());
    println!("success_rate_delta: {:.6}", comparison.success_rate_delta);
    println!(
        "collision_rate_delta: {:.6}",
        comparison.collision_rate_delta
    );
    println!("battery_delta_delta: {:.6}", comparison.battery_delta_delta);
    println!("fallback_delta: {}", comparison.fallback_delta);
}

#[derive(Clone, Debug)]
struct PromotionGateDecision {
    allowed: bool,
    warnings: Vec<String>,
}

fn promotion_gate(
    entry: &ModelRegistryEntry,
    target: ModelStatus,
    baseline: Option<&ScenarioEvaluationReport>,
    candidate: Option<&ScenarioEvaluationReport>,
    comparison: Option<&ScenarioReportComparison>,
    allow_safety_critical_inference: bool,
) -> PromotionGateDecision {
    let mut warnings = Vec::new();
    if !Path::new(&entry.checkpoint).exists() {
        warnings.push(format!("checkpoint missing: {}", entry.checkpoint));
    }
    if matches!(
        target,
        ModelStatus::Retired | ModelStatus::Rejected | ModelStatus::Registered
    ) {
        return PromotionGateDecision {
            allowed: true,
            warnings,
        };
    }
    if target == ModelStatus::Shadow {
        if entry.reports.scenario.is_none() {
            warnings.push("shadow requires a scenario evaluation report".to_string());
        }
        return PromotionGateDecision {
            allowed: warnings.is_empty(),
            warnings,
        };
    }
    if target != ModelStatus::Inference {
        return PromotionGateDecision {
            allowed: false,
            warnings: vec!["unknown promotion target".to_string()],
        };
    }
    if is_safety_critical_behavior(&entry.behavior) && !allow_safety_critical_inference {
        warnings.push(
            "safety-critical inference requires --allow-safety-critical-inference".to_string(),
        );
    }
    let Some(candidate) = candidate else {
        warnings.push("inference promotion requires a candidate scenario report".to_string());
        return PromotionGateDecision {
            allowed: false,
            warnings,
        };
    };
    if candidate.episodes < 10 {
        warnings.push(format!(
            "not enough scenario episodes for inference: {} < 10",
            candidate.episodes
        ));
    }
    if candidate.summary.model_fallbacks > 0 {
        warnings.push(format!(
            "model fallback count is not zero: {}",
            candidate.summary.model_fallbacks
        ));
    }
    if candidate.summary.collision_rate > 0.05 {
        warnings.push(format!(
            "candidate collision rate too high: {:.4}",
            candidate.summary.collision_rate
        ));
    }
    if let Some(comparison) = comparison {
        if comparison.outcome == ComparisonOutcome::Regressed {
            warnings.push("candidate scenario report regressed against baseline".to_string());
        }
    } else if baseline.is_none() && is_safety_critical_behavior(&entry.behavior) {
        warnings
            .push("safety-critical inference requires baseline comparison evidence".to_string());
    }
    match entry.behavior {
        TrainableBehavior::Danger => {
            if let Some(comparison) = comparison {
                if comparison.collision_rate_delta > 0.002 {
                    warnings.push(format!(
                        "danger collision rate worse than baseline by {:.4}",
                        comparison.collision_rate_delta
                    ));
                }
            }
        }
        TrainableBehavior::Charge => {
            if candidate.summary.success_rate < 0.70 {
                warnings.push(format!(
                    "charger success rate below threshold: {:.3}",
                    candidate.summary.success_rate
                ));
            }
            if candidate.summary.mean_battery_delta < -0.05 {
                warnings.push(format!(
                    "charger battery delta unacceptable: {:.3}",
                    candidate.summary.mean_battery_delta
                ));
            }
        }
        TrainableBehavior::ActionValue => {
            if candidate.scenario != "mixed-room" {
                warnings
                    .push("action-value inference requires mixed-room scenario eval".to_string());
            }
        }
        TrainableBehavior::Future => {
            warnings.push(
                "future inference is not a direct motor-control promotion; keep hardcoded fallback available"
                    .to_string(),
            );
        }
        TrainableBehavior::EyeNext | TrainableBehavior::EarNext | TrainableBehavior::Experience => {
            if entry.behavior == TrainableBehavior::Experience {
                warnings.push(
                    "experience inference changes latent encoding; only use where it cannot directly command motors"
                        .to_string(),
                );
            }
        }
    }
    PromotionGateDecision {
        allowed: warnings.is_empty(),
        warnings,
    }
}

fn allowed_modes_for_status(behavior: &TrainableBehavior, status: ModelStatus) -> Vec<String> {
    let mut modes = vec!["off".to_string(), "hardcoded".to_string()];
    if matches!(status, ModelStatus::Shadow | ModelStatus::Inference) {
        modes.push("shadow-infer".to_string());
    }
    if status == ModelStatus::Inference
        && matches!(
            behavior,
            TrainableBehavior::Future
                | TrainableBehavior::EyeNext
                | TrainableBehavior::EarNext
                | TrainableBehavior::Experience
        )
    {
        modes.push("model-infer".to_string());
    }
    modes
}

fn merge_warnings(left: &[String], right: &[String]) -> Vec<String> {
    let mut warnings = left.to_vec();
    for warning in right {
        if !warnings.contains(warning) {
            warnings.push(warning.clone());
        }
    }
    warnings
}

fn command_summary() -> String {
    std::env::args().collect::<Vec<_>>().join(" ")
}

fn current_git_commit() -> Option<String> {
    let output = ProcessCommand::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|text| !text.is_empty())
}

fn model_status() -> Result<()> {
    print_registry_status(Path::new(DEFAULT_REGISTRY_PATH))?;
    println!();
    println!("registered models:");
    for model in MODEL_REGISTRY {
        println!("  - {model}");
    }

    let config_path = Path::new("configs/models.toml");
    println!();
    println!("models config: {}", config_path.display());
    let config = match load_models_config(config_path) {
        Ok(config) => Some(config),
        Err(error) => {
            println!("  unavailable: {error}");
            None
        }
    };

    println!();
    println!("behavior instrument panel:");
    for behavior in trainable_behaviors() {
        print_behavior_status(behavior, config.as_ref())?;
    }

    println!();
    println!("checkpoint directories:");
    print_model_directories(Path::new("data/models"))?;
    Ok(())
}

fn print_registry_status(path: &Path) -> Result<()> {
    let registry = load_model_registry(path)?;
    println!("model registry: {}", path.display());
    if registry.entries.is_empty() {
        println!("  no registry entries");
        return Ok(());
    }
    println!(
        "{:<14} {:<24} {:<10} {:<32} {:<32} recommendation/warnings",
        "behavior", "name", "status", "checkpoint", "scenario report"
    );
    for entry in registry.entries {
        let report = entry.reports.scenario.as_deref().unwrap_or("-");
        let recommendation = registry_recommendation(&entry);
        println!(
            "{:<14} {:<24} {:<10} {:<32} {:<32} {}",
            entry.behavior,
            entry.name,
            entry.status.as_str(),
            entry.checkpoint,
            report,
            recommendation
        );
    }
    Ok(())
}

fn registry_recommendation(entry: &ModelRegistryEntry) -> String {
    if !entry.warnings.is_empty() {
        return entry.warnings.join("; ");
    }
    match entry.status {
        ModelStatus::Registered => "run scenario eval, then promote to shadow".to_string(),
        ModelStatus::Shadow => {
            if is_safety_critical_behavior(&entry.behavior) {
                "collect baseline comparison before inference".to_string()
            } else {
                "eligible for cautious inference review".to_string()
            }
        }
        ModelStatus::Inference => "allowed for configured inference surfaces".to_string(),
        ModelStatus::Retired => "retired".to_string(),
        ModelStatus::Rejected => "rejected".to_string(),
    }
}

fn trainable_behaviors() -> &'static [TrainableBehavior] {
    &[
        TrainableBehavior::Danger,
        TrainableBehavior::Charge,
        TrainableBehavior::ActionValue,
        TrainableBehavior::Future,
        TrainableBehavior::EyeNext,
        TrainableBehavior::EarNext,
        TrainableBehavior::Experience,
    ]
}

fn print_behavior_status(
    behavior: &TrainableBehavior,
    config: Option<&netherwick_behaviors::BehaviorRegistryConfig>,
) -> Result<()> {
    let key = behavior.config_key();
    let configured = config.and_then(|config| config.behavior.get(key));
    let checkpoint = configured
        .and_then(|entry| entry.checkpoint.as_deref())
        .unwrap_or_else(|| default_checkpoint(behavior));
    let checkpoint_path = Path::new(checkpoint);
    let checkpoint_present = checkpoint_path.is_dir();
    let metadata = read_json_optional(&checkpoint_path.join("metadata.json"))?;
    let evaluation = read_json_optional(&checkpoint_path.join("evaluation.json"))?;
    let latest_metric = read_latest_metric(&checkpoint_path.join("metrics.jsonl"))?;

    println!("  - {}", behavior);
    println!(
        "      checkpoint: {} ({})",
        checkpoint_path.display(),
        if checkpoint_present {
            "present"
        } else {
            "missing"
        }
    );
    println!(
        "      samples_seen: {}",
        json_field(&metadata, "samples_seen").unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "      best_loss: {}",
        json_field(&metadata, "best_loss").unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "      latest_eval_loss: {}",
        json_field(&evaluation, "model_loss_mean").unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "      hardcoded_loss: {}",
        json_field(&evaluation, "hardcoded_loss_mean").unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "      improvement_ratio: {}",
        json_field(&evaluation, "improvement_ratio").unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "      current regime: {}",
        configured
            .map(|entry| format!("{:?}", entry.regime))
            .unwrap_or_else(|| "unconfigured".to_string())
    );
    println!(
        "      recommended regime: {}",
        recommended_regime(behavior, evaluation.as_ref())
    );
    println!(
        "      safety-critical? {}",
        if is_safety_critical_behavior(behavior) {
            "yes"
        } else {
            "no"
        }
    );
    println!(
        "      last metrics timestamp: {}",
        latest_metric
            .as_ref()
            .and_then(|metric| json_field(&Some(metric.clone()), "t_ms"))
            .unwrap_or_else(|| "unknown".to_string())
    );

    if let Some(entry) = configured {
        println!("      hardcoded: {}", entry.hardcoded);
        println!("      model: {}", entry.model.as_deref().unwrap_or("none"));
        println!("      fallback: {:?}", entry.fallback);
    } else {
        println!("      hardcoded: {}", behavior.default_hardcoded_id());
        println!("      model: {}", behavior.default_model_id());
        println!("      fallback: UseHardcoded");
    }
    if let Some(warnings) = evaluation
        .as_ref()
        .and_then(|json| json.get("warnings"))
        .and_then(Value::as_array)
    {
        for warning in warnings.iter().filter_map(Value::as_str) {
            println!("      warning: {warning}");
        }
    }
    Ok(())
}

fn read_json_optional(path: &Path) -> Result<Option<Value>> {
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_slice(&fs::read(path)?)?))
}

fn read_latest_metric(path: &Path) -> Result<Option<Value>> {
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(path)?;
    let Some(line) = text.lines().rev().find(|line| !line.trim().is_empty()) else {
        return Ok(None);
    };
    Ok(Some(serde_json::from_str(line)?))
}

fn json_field(json: &Option<Value>, field: &str) -> Option<String> {
    json.as_ref().and_then(|json| {
        json.get(field).map(|value| match value {
            Value::String(text) => text.clone(),
            Value::Null => "null".to_string(),
            other => other.to_string(),
        })
    })
}

fn recommended_regime(behavior: &TrainableBehavior, evaluation: Option<&Value>) -> String {
    let recommendation = evaluation
        .and_then(|json| json.get("recommendation"))
        .and_then(Value::as_str);
    match recommendation {
        Some("promote_to_model_infer") if is_safety_critical_behavior(behavior) => {
            "shadow_infer (model_infer blocked for safety-critical behavior)".to_string()
        }
        Some("promote_to_model_infer") => "model_infer".to_string(),
        Some("shadow_infer") => "shadow_infer".to_string(),
        Some("shadow_train") => "shadow_train".to_string(),
        Some("keep_hardcoded") => "hardcoded".to_string(),
        Some("reject_checkpoint") => "hardcoded (reject checkpoint)".to_string(),
        Some(other) => format!("unknown ({other})"),
        None => "unknown".to_string(),
    }
}

fn is_safety_critical_behavior(behavior: &TrainableBehavior) -> bool {
    matches!(
        behavior,
        TrainableBehavior::Danger
            | TrainableBehavior::ActionValue
            | TrainableBehavior::Experience
            | TrainableBehavior::Future
    )
}

fn print_model_directories(path: &Path) -> Result<()> {
    if !path.exists() {
        println!("  missing {}", path.display());
        return Ok(());
    }
    let mut directories = fs::read_dir(path)?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            entry
                .file_type()
                .ok()
                .filter(|file_type| file_type.is_dir())
                .map(|_| entry.path())
        })
        .collect::<Vec<_>>();
    directories.sort();
    if directories.is_empty() {
        println!("  none below {}", path.display());
        return Ok(());
    }
    for directory in directories {
        let metadata = directory.join("metadata.json").exists();
        let evaluation = directory.join("evaluation.json").exists();
        let metrics = directory.join("metrics.jsonl").exists();
        println!(
            "  - {} (metadata={}, evaluation={}, metrics={})",
            directory.display(),
            metadata,
            evaluation,
            metrics
        );
    }
    Ok(())
}

fn print_frame(frame: &ExperienceFrame) {
    println!("frame {} @ {}ms", frame.id, frame.t_ms);
    println!("  summary: {}", frame.summary_text());
    println!("  action: {:?}", frame.chosen_action);
    println!("  recalls: {}", frame.memory_recall.len());
    println!("  recollections: {}", frame.recollections.len());
    if let Some(experience) = frame.experiences.last() {
        println!("  experience: {}", experience.text);
    }
    if let Some(transcript) = &frame.now.ear.transcript {
        println!("  heard: {}", transcript);
    }
    if let Some(eye_frame) = &frame.now.eye_frame {
        println!(
            "  eye_frame: {}x{} ({:?}) source={:?}",
            eye_frame.width,
            eye_frame.height,
            eye_frame.format,
            eye_frame.source.as_deref().unwrap_or("none")
        );
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct VirtualRunReport {
    pub total_frames: usize,
    pub total_transitions: usize,
    pub total_eye_frames: usize,
    pub total_ear_frames: usize,
    pub total_stuck_trap_events: usize,
    pub battery_delta: f32,
    pub duration_seconds: f64,
    pub eye_sources: HashMap<String, usize>,
    pub retina_coverage: f32,
    pub collisions: usize,
    pub collision_rate: f32,
    pub charger_contacts: usize,
    pub charging_ticks: usize,
    pub battery_recovery_success: bool,
    pub stuck_recovery_attempts: usize,
    pub stuck_recovery_successes: usize,
    pub trap_kinds: HashMap<String, usize>,
    pub ledger_gaps: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VirtualTrainingReport {
    pub timestamp: String,
    pub run_report: VirtualRunReport,
    pub models: HashMap<String, ModelTrainingStatus>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelTrainingStatus {
    pub name: String,
    pub trained: bool,
    pub previous_status: String,
    pub new_status: String,
    pub recommended_action: String,
    pub warnings: Vec<String>,
    pub loss: Option<f32>,
    pub baseline_collision_rate: Option<f32>,
    pub candidate_collision_rate: Option<f32>,
    pub baseline_success_rate: Option<f32>,
    pub candidate_success_rate: Option<f32>,
}

async fn run_virtual_report(args: VirtualReportArgs) -> Result<()> {
    let report = generate_virtual_report(&args.ledger).await?;
    let parent = Path::new(&args.out).parent();
    if let Some(p) = parent {
        if !p.as_os_str().is_empty() {
            fs::create_dir_all(p)?;
        }
    }
    let content = serde_json::to_string_pretty(&report)?;
    fs::write(&args.out, content)?;
    println!("virtual run report written to {}", args.out);
    Ok(())
}

async fn generate_virtual_report(ledger_path: &str) -> Result<VirtualRunReport> {
    let ledger = JsonlLedger::new(ledger_path);
    let frames = ledger.frames().await?;
    let transitions = ledger.transitions().await?;

    let total_frames = frames.len();
    let total_transitions = transitions.len();

    let mut total_eye_frames = 0;
    let mut total_ear_frames = 0;
    let mut total_stuck_trap_events = 0;

    let mut eye_sources = HashMap::new();
    let mut babylon_eye_frames = 0;
    let mut collisions = 0;
    let mut charging_ticks = 0;
    let mut charger_contacts = 0;
    let mut was_charging = false;
    let mut stuck_recovery_attempts = 0;
    let mut stuck_recovery_successes = 0;
    let mut trap_kinds = HashMap::new();
    let mut ledger_gaps = Vec::new();
    let mut warnings = Vec::new();

    let mut min_battery = 1.0f32;
    let mut max_after_min = 0.0f32;
    let mut prev_t_ms = None;

    for frame in &frames {
        if !frame.now.eye.frames.is_empty() || !frame.now.eye.image_vectors.is_empty() {
            total_eye_frames += 1;
        }
        if !frame.now.ear.features.is_empty() || frame.now.ear.transcript.is_some() {
            total_ear_frames += 1;
        }

        // 1. Eye source tracking
        if let Some(eye_frame) = &frame.now.eye_frame {
            let src = eye_frame
                .source
                .clone()
                .unwrap_or_else(|| "none".to_string());
            *eye_sources.entry(src.clone()).or_insert(0) += 1;
            if src == "babylon-robot-eye" {
                babylon_eye_frames += 1;
            }
        }

        // 2. Collision tracking
        if frame.now.body.flags.bump_left || frame.now.body.flags.bump_right {
            collisions += 1;
        }

        // 3. Charger & Battery tracking
        if frame.now.body.charging {
            charging_ticks += 1;
            if !was_charging {
                charger_contacts += 1;
            }
            was_charging = true;
        } else {
            was_charging = false;
        }

        let bat = frame.now.body.battery_level;
        if bat < min_battery {
            min_battery = bat;
            max_after_min = bat;
        } else if bat > max_after_min {
            max_after_min = bat;
        }

        // 4. Stuck recovery / Trap tracking
        if let Some(val) = frame.now.extensions.get("sim.stuck") {
            if let Ok(values) = serde_json::from_value::<Vec<f32>>(val.clone()) {
                let event_started = values.get(6).copied().unwrap_or(0.0) > 0.0;
                let recovered = values.get(7).copied().unwrap_or(0.0) > 0.0;
                let trap_code = values.get(10).copied().unwrap_or(0.0);

                if event_started {
                    total_stuck_trap_events += 1;
                    stuck_recovery_attempts += 1;
                    let trap_name = match trap_code {
                        1.0 => "Wall",
                        2.0 => "Corner",
                        3.0 => "Column",
                        _ => "Unknown",
                    }
                    .to_string();
                    *trap_kinds.entry(trap_name).or_insert(0) += 1;
                }
                if recovered {
                    stuck_recovery_successes += 1;
                }
            }
        }

        // 5. Gap tracking
        if let Some(prev) = prev_t_ms {
            let diff = frame.t_ms.saturating_sub(prev);
            if diff > 500 {
                ledger_gaps.push(format!(
                    "gap of {}ms between {}ms and {}ms",
                    diff, prev, frame.t_ms
                ));
            }
        }
        prev_t_ms = Some(frame.t_ms);
    }

    let battery_delta = if let (Some(first), Some(last)) = (frames.first(), frames.last()) {
        first.now.body.battery_level - last.now.body.battery_level
    } else {
        0.0
    };

    let battery_recovery_success = max_after_min - min_battery >= 0.05;

    let duration_seconds = if let (Some(first), Some(last)) = (frames.first(), frames.last()) {
        (last.t_ms.saturating_sub(first.t_ms) as f64) / 1000.0
    } else {
        0.0
    };

    let collision_rate = if total_frames > 0 {
        collisions as f32 / total_frames as f32
    } else {
        0.0
    };

    let retina_coverage = if total_frames > 0 {
        babylon_eye_frames as f32 / total_frames as f32
    } else {
        0.0
    };

    if total_frames == 0 {
        warnings.push("ledger is empty".to_string());
    } else if babylon_eye_frames == 0 {
        warnings.push("no retina frames from babylon-robot-eye found in ledger".to_string());
    }

    Ok(VirtualRunReport {
        total_frames,
        total_transitions,
        total_eye_frames,
        total_ear_frames,
        total_stuck_trap_events,
        battery_delta,
        duration_seconds,
        eye_sources,
        retina_coverage,
        collisions,
        collision_rate,
        charger_contacts,
        charging_ticks,
        battery_recovery_success,
        stuck_recovery_attempts,
        stuck_recovery_successes,
        trap_kinds,
        ledger_gaps,
        warnings,
    })
}

async fn run_train_virtual(args: TrainVirtualArgs) -> Result<()> {
    println!("Starting virtual training pipeline...");
    println!("Ledger: {}", args.ledger);
    println!("Out Dir: {}", args.out_dir);

    // 1. Generate run report
    let run_report = generate_virtual_report(&args.ledger).await?;
    println!("Run report generated successfully.");

    // Create out_dir
    fs::create_dir_all(&args.out_dir)?;

    // 2. Train selected behaviors
    let behaviors = vec![
        TrainableBehavior::Danger,
        TrainableBehavior::Charge,
        TrainableBehavior::EyeNext,
        TrainableBehavior::EarNext,
        TrainableBehavior::Future,
    ];

    let mut trained_summaries = HashMap::new();
    for behavior in &behaviors {
        let checkpoint_path = Path::new(&args.out_dir).join(behavior.config_key());
        println!("Training behavior model: {:?}", behavior);
        let summary = train_behavior(TrainBehaviorRequest {
            behavior: behavior.clone(),
            ledger_path: PathBuf::from(&args.ledger),
            checkpoint_path,
            epochs: args.epochs,
            validation_split: 0.2,
            seed: 7,
        })
        .await?;
        trained_summaries.insert(behavior.clone(), summary);
    }

    // 3. Run scenario evaluations
    println!("Running baseline scenario evaluation (all models Off)...");
    let baseline_report_path = Path::new(&args.out_dir).join("baseline-scenario.json");
    let baseline_args = EvalScenarioArgs {
        scenario: ScenarioArg::MixedRoom,
        episodes: 10,
        steps: 100,
        seed: 7,
        tick_ms: 100,
        out: Some(baseline_report_path.to_string_lossy().to_string()),
        ledger: None,
        capture_root: None,
        memory_report: false,
        danger_checkpoint: None,
        danger_mode: DangerMode::Off,
        charge_checkpoint: None,
        charge_mode: ChargeMode::Off,
        action_value_checkpoint: None,
        action_value_mode: ActionValueMode::Off,
        future_checkpoint: None,
        future_mode: FutureMode::Hardcoded,
        eye_next_checkpoint: None,
        eye_next_mode: EyeNextMode::Off,
        ear_next_checkpoint: None,
        ear_next_mode: EarNextMode::Off,
        experience_checkpoint: None,
        experience_mode: ExperienceMode::Off,
        action_selector: CliActionSelectorMode::Baseline,
        llm: LlmArgs::default(),
    };
    run_eval_scenario(baseline_args).await?;
    let baseline_report = load_scenario_report(&baseline_report_path.to_string_lossy())?;

    println!("Running candidate scenario evaluation (new models ShadowInfer)...");
    let candidate_report_path = Path::new(&args.out_dir).join("candidate-scenario.json");
    let candidate_args = EvalScenarioArgs {
        scenario: ScenarioArg::MixedRoom,
        episodes: 10,
        steps: 100,
        seed: 7,
        tick_ms: 100,
        out: Some(candidate_report_path.to_string_lossy().to_string()),
        ledger: None,
        capture_root: None,
        memory_report: false,
        danger_checkpoint: Some(
            Path::new(&args.out_dir)
                .join("danger")
                .to_string_lossy()
                .to_string(),
        ),
        danger_mode: DangerMode::ShadowInfer,
        charge_checkpoint: Some(
            Path::new(&args.out_dir)
                .join("charge")
                .to_string_lossy()
                .to_string(),
        ),
        charge_mode: ChargeMode::ShadowInfer,
        action_value_checkpoint: None,
        action_value_mode: ActionValueMode::Off,
        future_checkpoint: Some(
            Path::new(&args.out_dir)
                .join("future")
                .to_string_lossy()
                .to_string(),
        ),
        future_mode: FutureMode::ShadowInfer,
        eye_next_checkpoint: Some(
            Path::new(&args.out_dir)
                .join("eye_next")
                .to_string_lossy()
                .to_string(),
        ),
        eye_next_mode: EyeNextMode::ShadowInfer,
        ear_next_checkpoint: Some(
            Path::new(&args.out_dir)
                .join("ear_next")
                .to_string_lossy()
                .to_string(),
        ),
        ear_next_mode: EarNextMode::ShadowInfer,
        experience_checkpoint: None,
        experience_mode: ExperienceMode::Off,
        action_selector: CliActionSelectorMode::Baseline,
        llm: LlmArgs::default(),
    };
    run_eval_scenario(candidate_args).await?;
    let candidate_report = load_scenario_report(&candidate_report_path.to_string_lossy())?;

    // Compare scenario reports
    let comparison = compare_scenario_reports(&baseline_report, &candidate_report);
    println!("Evaluation comparison result: {:?}", comparison.outcome);

    // 4. Update/register models and run promotion gates
    let registry_path = Path::new("data/models/registry.json");
    let mut model_statuses = HashMap::new();
    let timestamp = Utc::now().format("%Y%m%d_%H%M").to_string();

    for behavior in &behaviors {
        let name = format!("{}_virtual_{}", behavior.config_key(), timestamp);
        let checkpoint = Path::new(&args.out_dir)
            .join(behavior.config_key())
            .to_string_lossy()
            .to_string();
        let behavior_report = Path::new(&checkpoint).join("evaluation.json");

        println!("Registering candidate model {}...", name);
        model_register(ModelRegisterArgs {
            behavior: behavior.cli_name().to_string(),
            checkpoint: checkpoint.clone(),
            training_ledger: Some(args.ledger.clone()),
            training_command: Some("just train virtual".to_string()),
            behavior_report: Some(behavior_report.to_string_lossy().to_string()),
            scenario_report: Some(candidate_report_path.to_string_lossy().to_string()),
            name: name.clone(),
            notes: vec!["Automatically trained via virtual pipeline".to_string()],
            parent: None,
            registry: registry_path.to_string_lossy().to_string(),
            overwrite: true,
        })?;

        // Load the registry to get the entry we just registered
        let registry = load_model_registry(registry_path)?;
        let entry = registry
            .entries
            .iter()
            .find(|e| e.name == name && e.behavior == *behavior)
            .unwrap()
            .clone();

        // Determine recommended promotion status
        // First test Inference promotion
        let inference_decision = promotion_gate(
            &entry,
            ModelStatus::Inference,
            Some(&baseline_report),
            Some(&candidate_report),
            Some(&comparison),
            args.allow_safety_critical_inference,
        );

        let mut new_status = ModelStatus::Registered;
        let mut recommended_action = "keep hardcoded".to_string();
        let mut warnings = Vec::new();

        if inference_decision.allowed {
            new_status = ModelStatus::Inference;
            recommended_action = "inference".to_string();
        } else {
            // Test Shadow promotion
            let shadow_decision = promotion_gate(
                &entry,
                ModelStatus::Shadow,
                Some(&baseline_report),
                Some(&candidate_report),
                Some(&comparison),
                args.allow_safety_critical_inference,
            );
            if shadow_decision.allowed {
                new_status = ModelStatus::Shadow;
                recommended_action = "shadow".to_string();
            } else {
                // Collect warnings for why promotion failed
                warnings.extend(inference_decision.warnings);
                warnings.extend(shadow_decision.warnings);
            }
        }

        // Apply promotion if recommended status is higher than Registered
        if new_status != ModelStatus::Registered {
            println!("Promoting model {} to {}...", name, new_status.as_str());
            model_promote(ModelPromoteArgs {
                behavior: behavior.cli_name().to_string(),
                name: name.clone(),
                target: new_status,
                baseline_report: Some(baseline_report_path.to_string_lossy().to_string()),
                candidate_report: Some(candidate_report_path.to_string_lossy().to_string()),
                registry: registry_path.to_string_lossy().to_string(),
                allow_safety_critical_inference: args.allow_safety_critical_inference,
                notes: vec!["Automatically promoted via virtual pipeline".to_string()],
            })?;
        }

        let loss = trained_summaries.get(behavior).and_then(|s| s.last_loss);

        model_statuses.insert(
            behavior.config_key().to_string(),
            ModelTrainingStatus {
                name,
                trained: true,
                previous_status: "registered".to_string(),
                new_status: new_status.as_str().to_string(),
                recommended_action,
                warnings,
                loss,
                baseline_collision_rate: Some(baseline_report.summary.collision_rate),
                candidate_collision_rate: Some(candidate_report.summary.collision_rate),
                baseline_success_rate: Some(baseline_report.summary.success_rate),
                candidate_success_rate: Some(candidate_report.summary.success_rate),
            },
        );
    }

    // 5. Write final consolidated training report
    let final_report = VirtualTrainingReport {
        timestamp: Utc::now().to_rfc3339(),
        run_report,
        models: model_statuses,
        warnings: if comparison.outcome == ComparisonOutcome::Regressed {
            vec![
                "Candidate models overall regressed on MixedRoom scenario against baseline"
                    .to_string(),
            ]
        } else {
            Vec::new()
        },
    };

    let parent = Path::new(&args.report_out).parent();
    if let Some(p) = parent {
        fs::create_dir_all(p)?;
    }
    fs::write(
        &args.report_out,
        serde_json::to_string_pretty(&final_report)?,
    )?;
    println!(
        "Consolidated training report written to {}",
        args.report_out
    );

    Ok(())
}

#[derive(Debug, Parser)]
struct RetinaMockSendArgs {
    /// Server URL
    #[arg(long, default_value = "https://localhost:8443")]
    url: String,

    /// Frame rate (FPS)
    #[arg(long, default_value = "5")]
    fps: u64,

    /// Width of mock image
    #[arg(long, default_value = "160")]
    width: u32,

    /// Height of mock image
    #[arg(long, default_value = "90")]
    height: u32,

    /// Color pattern: "solid-red", "solid-green", "solid-blue", "gradient", or "noise"
    #[arg(long, default_value = "gradient")]
    pattern: String,
}

fn generate_mock_image_base64(
    width: u32,
    height: u32,
    pattern: &str,
    frame_index: usize,
) -> Result<String> {
    use base64::Engine;
    use image::codecs::png::PngEncoder;
    use image::ImageEncoder;
    use image::{Rgb, RgbImage};

    let mut img = RgbImage::new(width, height);

    match pattern {
        "solid-red" => {
            for pixel in img.pixels_mut() {
                *pixel = Rgb([255, 0, 0]);
            }
        }
        "solid-green" => {
            for pixel in img.pixels_mut() {
                *pixel = Rgb([0, 255, 0]);
            }
        }
        "solid-blue" => {
            for pixel in img.pixels_mut() {
                *pixel = Rgb([0, 0, 255]);
            }
        }
        "gradient" => {
            for (x, y, pixel) in img.enumerate_pixels_mut() {
                let r = ((x as f32 / width as f32) * 255.0) as u8;
                let g = ((y as f32 / height as f32) * 255.0) as u8;
                let b = ((frame_index * 10) % 256) as u8;
                *pixel = Rgb([r, g, b]);
            }
        }
        "noise" => {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for pixel in img.pixels_mut() {
                *pixel = Rgb([rng.gen(), rng.gen(), rng.gen()]);
            }
        }
        _ => {
            for (x, y, pixel) in img.enumerate_pixels_mut() {
                let g = ((x as f32 / width as f32) * 255.0) as u8;
                let b = ((y as f32 / height as f32) * 255.0) as u8;
                *pixel = Rgb([0, g, b]);
            }
        }
    }

    let mut png_bytes = Vec::new();
    PngEncoder::new(&mut png_bytes)
        .write_image(&img, width, height, image::ColorType::Rgb8.into())
        .context("failed to encode mock image as PNG")?;

    let encoded = base64::engine::general_purpose::STANDARD.encode(&png_bytes);
    Ok(encoded)
}

async fn run_retina_mock_send(args: RetinaMockSendArgs) -> Result<()> {
    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .context("failed to build reqwest client")?;

    let url = format!("{}/view/retina-frame", args.url.trim_end_matches('/'));
    println!(
        "Starting mock retina stream to {url} at {} FPS ({}x{})...",
        args.fps, args.width, args.height
    );

    let interval = Duration::from_millis(1000 / args.fps.max(1));
    let mut interval_timer = tokio::time::interval(interval);
    let mut frame_index = 0;

    let start_time = std::time::Instant::now();

    loop {
        interval_timer.tick().await;

        let t_ms = start_time.elapsed().as_millis() as u64;
        let base64_str =
            match generate_mock_image_base64(args.width, args.height, &args.pattern, frame_index) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error generating mock image: {e}");
                    continue;
                }
            };

        let payload = serde_json::json!({
            "schema_version": 1,
            "source": "babylon-robot-eye",
            "t_ms": t_ms,
            "frame_index": frame_index,
            "width": args.width,
            "height": args.height,
            "format": "Rgb8",
            "encoding": "base64",
            "data": format!("data:image/png;base64,{base64_str}")
        });

        let res = client.post(&url).json(&payload).send().await;

        match res {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    println!(
                        "[frame {}] Sent successfully (t_ms = {})",
                        frame_index, t_ms
                    );
                } else {
                    let err_text = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "unknown error".to_string());
                    eprintln!(
                        "[frame {}] FAILED with status {}: {}",
                        frame_index, status, err_text
                    );
                }
            }
            Err(e) => {
                eprintln!("[frame {}] Request error: {}", frame_index, e);
            }
        }

        frame_index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use netherwick_actions::ActionPrimitive;
    use netherwick_body::BodySense;
    use netherwick_core::Reward;
    use netherwick_experience::ExperienceLatent;
    use netherwick_now::{ExtensionSense, Now, SurpriseSense};
    use netherwick_sensors::World;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(prefix: &str) -> std::path::PathBuf {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        std::env::temp_dir().join(format!("{prefix}_{now_ms}"))
    }

    #[test]
    fn sim_args_parse_virtual_live_tls_flags() {
        let cli = Cli::try_parse_from([
            "netherwick",
            "sim",
            "--live",
            "--live-tls",
            "--live-addr",
            "0.0.0.0:9443",
            "--live-tls-cert",
            "certs/test.crt",
            "--live-tls-key",
            "certs/test.key",
            "--scenario",
            "charger-seeking",
            "--steps",
            "123",
        ])
        .unwrap();

        let Command::Sim(args) = cli.command else {
            panic!("expected sim command");
        };
        assert!(args.live);
        assert!(args.live_tls);
        assert_eq!(args.live_addr.port(), 9443);
        assert_eq!(args.live_tls_cert, "certs/test.crt");
        assert_eq!(args.live_tls_key, "certs/test.key");
        assert_eq!(args.scenario, ScenarioArg::ChargerSeeking);
        assert_eq!(args.steps, 123);
    }

    #[test]
    fn scenario_arg_parses_all_public_slugs() {
        for (slug, expected) in [
            ("empty-room", ScenarioArg::EmptyRoom),
            ("obstacle-avoidance", ScenarioArg::ObstacleAvoidance),
            ("corner-trap", ScenarioArg::CornerTrap),
            ("column-trap", ScenarioArg::ColumnTrap),
            ("charger-seeking", ScenarioArg::ChargerSeeking),
            ("person-speaker-room", ScenarioArg::PersonSpeakerRoom),
            ("mixed-room", ScenarioArg::MixedRoom),
        ] {
            let cli = Cli::try_parse_from(["netherwick", "sim", "--scenario", slug]).unwrap();
            let Command::Sim(args) = cli.command else {
                panic!("expected sim command");
            };
            assert_eq!(args.scenario, expected);
        }
    }

    fn eval_args(
        scenario: ScenarioArg,
        episodes: usize,
        steps: usize,
        out: Option<String>,
    ) -> EvalScenarioArgs {
        EvalScenarioArgs {
            scenario,
            episodes,
            steps,
            seed: 7,
            tick_ms: 100,
            out,
            ledger: None,
            capture_root: None,
            memory_report: false,
            danger_checkpoint: None,
            danger_mode: DangerMode::Off,
            charge_checkpoint: None,
            charge_mode: ChargeMode::Off,
            action_value_checkpoint: None,
            action_value_mode: ActionValueMode::Off,
            future_checkpoint: None,
            future_mode: FutureMode::Hardcoded,
            eye_next_checkpoint: None,
            eye_next_mode: EyeNextMode::Off,
            ear_next_checkpoint: None,
            ear_next_mode: EarNextMode::Off,
            experience_checkpoint: None,
            experience_mode: ExperienceMode::Off,
            action_selector: CliActionSelectorMode::Baseline,
            llm: LlmArgs::default(),
        }
    }

    fn replay_counterfactual_args(
        capture: String,
        out_ledger: Option<String>,
        out_report: Option<String>,
    ) -> ReplayCounterfactualArgs {
        ReplayCounterfactualArgs {
            capture,
            edit: Vec::new(),
            policy: "baseline".to_string(),
            actions: None,
            steps: Some(4),
            out_ledger,
            out_report,
            llm: LlmArgs::default(),
        }
    }

    #[test]
    fn counterfactual_edit_parser_parses_supported_edits() {
        assert_eq!(
            parse_counterfactual_edit("move-charger:x=1.0,y=2.0").unwrap(),
            CounterfactualEdit::MoveObject {
                kind: CounterfactualObjectKind::Charger,
                id: None,
                x_m: 1.0,
                y_m: 2.0,
            }
        );
        assert_eq!(
            parse_counterfactual_edit("set-battery:value=0.42").unwrap(),
            CounterfactualEdit::SetBattery { value: 0.42 }
        );
        assert!(parse_counterfactual_edit("move-moon:x=1,y=2")
            .unwrap_err()
            .to_string()
            .contains("unknown counterfactual edit"));
    }

    #[test]
    fn counterfactual_edits_move_charger_and_set_battery() {
        let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ChargerSeeking, 9));
        let mut metadata = scenario.metadata;
        let edits = vec![
            parse_counterfactual_edit("move-charger:x=1.0,y=1.0").unwrap(),
            parse_counterfactual_edit("set-battery:value=0.75").unwrap(),
        ];
        let mut warnings = Vec::new();

        apply_counterfactual_edits(&mut metadata, &edits, &mut warnings).unwrap();

        let charger = metadata
            .objects
            .iter()
            .find(|object| matches!(object.kind, netherwick_sim::SimObjectKind::Charger))
            .unwrap();
        assert_eq!((charger.x_m, charger.y_m), (1.0, 1.0));
        assert_eq!(metadata.body.battery_level, 0.75);
        assert!(warnings
            .iter()
            .any(|warning| warning.contains("first matching object")));
    }

    #[test]
    fn counterfactual_report_serializes_schema() {
        let report = CounterfactualReport {
            schema_version: 1,
            source_capture: "capture".to_string(),
            reconstructable: true,
            edits: vec!["set-battery:value=0.5".to_string()],
            policy: "stop".to_string(),
            steps: 3,
            summary: CounterfactualSummary {
                collisions: 0,
                charging_ticks: 1,
                battery_delta: 0.1,
                distance_traveled: 0.2,
                final_distance_to_charger_m: Some(0.3),
            },
            warnings: Vec::new(),
        };

        let value: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&report).unwrap()).unwrap();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["summary"]["charging_ticks"], 1);
    }

    #[tokio::test]
    async fn replay_counterfactual_baseline_writes_ledger_and_report() {
        let temp_dir = temp_path("netherwick_counterfactual_baseline");
        let capture_dir = temp_dir.join("capture");
        let ledger_dir = temp_dir.join("ledger");
        let report_path = temp_dir.join("report.json");
        let mut scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ChargerSeeking, 77));
        let snapshot = scenario.world.snapshot().await.unwrap();
        let mut writer = CaptureWriter::create(&capture_dir, CaptureSource::Sim, Some(100))
            .await
            .unwrap();
        writer.manifest_mut().scenario = Some(scenario.metadata);
        writer
            .append_snapshot(snapshot.body.last_update_ms, snapshot, Vec::new())
            .await
            .unwrap();
        writer.finish().await.unwrap();

        let args = replay_counterfactual_args(
            capture_dir.to_string_lossy().to_string(),
            Some(ledger_dir.to_string_lossy().to_string()),
            Some(report_path.to_string_lossy().to_string()),
        );
        replay_counterfactual(args).await.unwrap();

        let transitions = JsonlLedger::new(&ledger_dir).transitions().await.unwrap();
        assert!(!transitions.is_empty());
        let report: CounterfactualReport =
            serde_json::from_slice(&fs::read(&report_path).unwrap()).unwrap();
        assert!(report.reconstructable);
        assert_eq!(report.steps, 4);
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn replay_counterfactual_with_moved_charger_writes_report() {
        let temp_dir = temp_path("netherwick_counterfactual_moved_charger");
        let capture_dir = temp_dir.join("capture");
        let report_path = temp_dir.join("report.json");
        let mut scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ChargerSeeking, 78));
        let snapshot = scenario.world.snapshot().await.unwrap();
        let mut writer = CaptureWriter::create(&capture_dir, CaptureSource::Sim, Some(100))
            .await
            .unwrap();
        writer.manifest_mut().scenario = Some(scenario.metadata);
        writer
            .append_snapshot(snapshot.body.last_update_ms, snapshot, Vec::new())
            .await
            .unwrap();
        writer.finish().await.unwrap();

        let mut args = replay_counterfactual_args(
            capture_dir.to_string_lossy().to_string(),
            None,
            Some(report_path.to_string_lossy().to_string()),
        );
        args.edit = vec!["move-charger:x=1.0,y=1.0".to_string()];
        args.policy = "seek-charge".to_string();
        replay_counterfactual(args).await.unwrap();

        let report: CounterfactualReport =
            serde_json::from_slice(&fs::read(&report_path).unwrap()).unwrap();
        assert_eq!(report.edits, vec!["move-charger:x=1.0,y=1.0"]);
        assert_eq!(report.policy, "seek-charge");
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("first matching object")));
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn replay_counterfactual_passive_capture_fails_clearly() {
        let temp_dir = temp_path("netherwick_counterfactual_passive");
        let capture_dir = temp_dir.join("capture");
        let mut writer = CaptureWriter::create(&capture_dir, CaptureSource::Replay, Some(100))
            .await
            .unwrap();
        writer
            .append_snapshot(0, WorldSnapshot::default(), Vec::new())
            .await
            .unwrap();
        writer.finish().await.unwrap();

        let err = replay_counterfactual(replay_counterfactual_args(
            capture_dir.to_string_lossy().to_string(),
            None,
            None,
        ))
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains(
            "passive captures without reconstructable sim metadata cannot yet be counterfactually replayed"
        ));
        let _ = fs::remove_dir_all(&temp_dir);
    }

    fn tick_with_action(action: ActionPrimitive) -> RuntimeTick {
        let now = Now::blank(100, BodySense::default());
        RuntimeTick {
            frame: ExperienceFrame {
                id: uuid::Uuid::new_v4(),
                t_ms: 100,
                now,
                sensations: Vec::new(),
                impressions: Vec::new(),
                experiences: Vec::new(),
                z: None,
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
            experience: netherwick_experience::Experience::new(
                "test",
                "test",
                Vec::new(),
                Vec::new(),
                100,
                100,
            ),
            chosen_action: Some(action),
            recall: Default::default(),
            llm: Default::default(),
            combobulation: None,
            inline_learning: Default::default(),
        }
    }

    #[test]
    fn scenario_report_round_trips_json() {
        let report = ScenarioEvaluationReport {
            schema_version: 1,
            scenario: "empty-room".to_string(),
            base_seed: 7,
            episodes: 1,
            steps_per_episode: 2,
            tick_ms: 100,
            action_selector_mode: "baseline".to_string(),
            model_modes: HashMap::new(),
            model_loading: RuntimeModelLoadReport::default(),
            ledger: None,
            capture_root: None,
            summary: ScenarioEvaluationSummary::default(),
            memory: None,
            episodes_detail: Vec::new(),
            recommendation: "insufficient_data".to_string(),
            warnings: Vec::new(),
        };
        let encoded = serde_json::to_string(&report).unwrap();
        let decoded: ScenarioEvaluationReport = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.scenario, "empty-room");
        assert_eq!(decoded.schema_version, 1);
    }

    #[test]
    fn obstacle_metrics_count_collision_flags() {
        let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ObstacleAvoidance, 11));
        let mut metrics = EpisodeMetricBuilder::new(
            ScenarioKind::ObstacleAvoidance,
            scenario.metadata,
            0,
            11,
            None,
            None,
        );
        let mut snapshot = WorldSnapshot::default();
        snapshot.body.flags.bump_left = true;
        snapshot.body.flags.wall = true;
        snapshot.body.flags.cliff_front_left = true;
        snapshot.range.nearest_m = Some(0.2);
        metrics.observe(
            &snapshot,
            &tick_with_action(ActionPrimitive::Go {
                intensity: 0.2,
                duration_ms: 100,
            }),
        );
        let report = metrics.finish();
        assert_eq!(report.collisions, 1);
        assert_eq!(report.wall_hits, 1);
        assert_eq!(report.bumper_hits, 1);
        assert_eq!(report.cliff_hits, 1);
        assert_eq!(report.min_nearest_obstacle_m, Some(0.2));
    }

    #[test]
    fn stuck_metrics_count_recovery_events() {
        let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::CornerTrap, 11));
        let mut metrics = EpisodeMetricBuilder::new(
            ScenarioKind::CornerTrap,
            scenario.metadata,
            0,
            11,
            None,
            None,
        );
        let mut started = WorldSnapshot::default();
        started.extensions.push(ExtensionSense {
            schema_version: 1,
            name: "sim.stuck".to_string(),
            values: vec![1.0, 1.0, 6.0, 100.0, 1.0, -1.0, 1.0, 0.0],
        });
        metrics.observe(
            &started,
            &tick_with_action(ActionPrimitive::Explore {
                style: netherwick_actions::ExploreStyle::RandomWalk,
                duration_ms: 100,
            }),
        );
        let mut recovered = started.clone();
        recovered.body.odometry.x_m = 0.1;
        recovered.extensions[0].values = vec![0.0, 0.0, 0.0, 900.0, 0.0, -1.0, 0.0, 1.0];
        metrics.observe(
            &recovered,
            &tick_with_action(ActionPrimitive::Explore {
                style: netherwick_actions::ExploreStyle::RandomWalk,
                duration_ms: 100,
            }),
        );

        let report = metrics.finish();
        assert_eq!(report.stuck_count, 1);
        assert_eq!(report.stuck_ticks, 1);
        assert_eq!(report.stuck_duration, Some(100.0));
        assert_eq!(report.mean_stuck_duration, Some(100.0));
        assert_eq!(report.recovery_success_rate, Some(1.0));
    }

    #[test]
    fn metrics_record_dead_battery_tick() {
        let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::EmptyRoom, 12));
        let mut metrics = EpisodeMetricBuilder::new(
            ScenarioKind::EmptyRoom,
            scenario.metadata,
            0,
            12,
            None,
            None,
        );
        let mut alive = WorldSnapshot::default();
        alive.body.battery_level = 0.01;
        metrics.observe(&alive, &tick_with_action(ActionPrimitive::Stop));
        let mut dead = alive.clone();
        dead.body.battery_level = 0.0;
        metrics.observe(&dead, &tick_with_action(ActionPrimitive::Stop));

        let report = metrics.finish();
        assert_eq!(report.dead_battery_tick, Some(1));
    }

    #[test]
    fn charger_metrics_detect_success_and_battery_delta() {
        let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ChargerSeeking, 12));
        let mut metrics = EpisodeMetricBuilder::new(
            ScenarioKind::ChargerSeeking,
            scenario.metadata,
            0,
            12,
            None,
            None,
        );
        let mut start = WorldSnapshot::default();
        start.body.battery_level = 0.2;
        metrics.observe(&start, &tick_with_action(ActionPrimitive::Stop));
        let mut charged = start.clone();
        charged.body.battery_level = 0.26;
        charged.body.charging = true;
        metrics.observe(&charged, &tick_with_action(ActionPrimitive::Stop));
        let report = metrics.finish();
        assert_eq!(report.charging_ticks, 1);
        assert!(report.battery_delta > 0.05);
        assert!(report.success);
    }

    #[test]
    fn social_metrics_detect_projected_senses() {
        let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::PersonAndSpeaker, 13));
        let mut metrics = EpisodeMetricBuilder::new(
            ScenarioKind::PersonAndSpeaker,
            scenario.metadata,
            0,
            13,
            None,
            None,
        );
        let mut snapshot = WorldSnapshot::default();
        snapshot.eye.frames.push(vec![0.1, 0.2]);
        snapshot.ear.features.push(vec![0.3]);
        snapshot.voice.embeddings.push(vec![0.4]);
        snapshot.face.embeddings.push(vec![0.5]);
        snapshot.kinect.skeletons.push(Default::default());
        metrics.observe(&snapshot, &tick_with_action(ActionPrimitive::Stop));
        let report = metrics.finish();
        assert_eq!(report.ticks_with_eye_frames, 1);
        assert_eq!(report.ticks_with_ear_features, 1);
        assert_eq!(report.ticks_with_voice_embeddings, 1);
        assert_eq!(report.ticks_with_face_embeddings, 1);
        assert_eq!(report.ticks_with_kinect_skeletons, 1);
        assert!(report.success);
    }

    #[test]
    fn recommendation_logic_classifies_common_outcomes() {
        let strong = ScenarioEvaluationSummary {
            success_rate: 0.9,
            collision_rate: 0.01,
            ..ScenarioEvaluationSummary::default()
        };
        assert_eq!(
            scenario_recommendation(10, &strong),
            "candidate_for_more_eval"
        );
        assert_eq!(scenario_recommendation(2, &strong), "insufficient_data");
        let risky = ScenarioEvaluationSummary {
            success_rate: 0.9,
            collision_rate: 0.2,
            ..ScenarioEvaluationSummary::default()
        };
        assert_eq!(
            scenario_recommendation(10, &risky),
            "reject_or_continue_training"
        );
    }

    #[tokio::test]
    async fn hardware_env_report_has_expected_shape() {
        let report = collect_hardware_env_report().await;
        assert!(report.get("os").is_some());
        assert!(report.get("architecture").is_some());
        assert!(report.get("serial_devices").unwrap().is_array());
        assert!(report.get("camera_devices").unwrap().is_array());
        assert!(report.get("audio_input_devices").unwrap().is_array());
        assert!(report.get("kinect").unwrap().is_object());
        assert!(report.get("data_dirs_writable").unwrap().is_object());
    }

    #[test]
    fn missing_streams_generate_warnings() {
        let mut counts = StreamCounts::default();
        counts.observe(&WorldSnapshot::default());
        let streams = counts.streams();
        assert!(streams.present.contains(&"body".to_string()));
        assert!(streams.missing.contains(&"rgb".to_string()));
        assert!(counts
            .warnings()
            .iter()
            .any(|warning| warning == "rgb stream missing"));
    }

    #[tokio::test]
    async fn inspect_capture_reads_tiny_fake_capture() {
        let temp_dir = temp_path("netherwick_inspect_capture");
        let mut writer = CaptureWriter::create(&temp_dir, CaptureSource::RealRobot, Some(100))
            .await
            .unwrap();
        let mut snapshot = WorldSnapshot::default();
        snapshot.eye.frames.push(vec![0.1, 0.2]);
        writer
            .append_snapshot(100, snapshot, Vec::new())
            .await
            .unwrap();
        writer.finish().await.unwrap();

        let report = inspect_capture_report(&temp_dir).await.unwrap();
        assert_eq!(report.frame_count, 1);
        assert!(report.streams_present.contains(&"rgb".to_string()));
        assert!(report.streams_missing.contains(&"audio".to_string()));
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn capture_real_mock_writes_manifest_and_frames() {
        let temp_dir = temp_path("netherwick_capture_real_mock");
        let args = CaptureRealArgs {
            duration_seconds: 1,
            out: temp_dir.to_string_lossy().to_string(),
            ledger: None,
            tick_ms: 1000,
            create_port: "mock".to_string(),
            create_baud: 57_600,
            camera: None,
            mic: None,
            imu: None,
            gps: None,
            mock: true,
            export_rgb: false,
            export_depth: false,
            export_audio: false,
            export_pointcloud: false,
            pointcloud_stride: 4,
            llm: LlmArgs::default(),
        };

        capture_real(args).await.unwrap();
        assert!(temp_dir.join("manifest.json").exists());
        assert!(temp_dir.join("frames.jsonl").exists());
        let report = inspect_capture_report(&temp_dir).await.unwrap();
        assert_eq!(report.frame_count, 1);
        assert!(report.streams_present.contains(&"body".to_string()));
        assert!(report.streams_present.contains(&"audio".to_string()));
        assert!(report.streams_present.contains(&"depth".to_string()));
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn capture_real_mock_exports_assets_and_pointclouds() {
        let temp_dir = temp_path("netherwick_capture_real_mock_assets");
        let args = CaptureRealArgs {
            duration_seconds: 1,
            out: temp_dir.to_string_lossy().to_string(),
            ledger: None,
            tick_ms: 1000,
            create_port: "mock".to_string(),
            create_baud: 57_600,
            camera: None,
            mic: None,
            imu: None,
            gps: None,
            mock: true,
            export_rgb: true,
            export_depth: true,
            export_audio: true,
            export_pointcloud: false,
            pointcloud_stride: 4,
            llm: LlmArgs::default(),
        };

        capture_real(args).await.unwrap();
        capture_assets(CaptureAssetsArgs {
            capture: temp_dir.to_string_lossy().to_string(),
            pointcloud: true,
            stride: 1,
            max_depth_m: 8.0,
        })
        .await
        .unwrap();

        let report = inspect_capture_report(&temp_dir).await.unwrap();
        assert_eq!(
            report.asset_counts,
            vec![
                ("rgb".to_string(), 1),
                ("depth".to_string(), 1),
                ("audio".to_string(), 1),
                ("pointcloud".to_string(), 1)
            ]
        );
        assert!(report
            .asset_details
            .iter()
            .any(|detail| detail.contains("rgb metadata: 2x2")));
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("uncalibrated point cloud")));
        let replay = replay_capture(ReplayCaptureArgs {
            capture: temp_dir.to_string_lossy().to_string(),
            ledger: temp_dir.join("ledger").to_string_lossy().to_string(),
            llm: LlmArgs::default(),
        })
        .await;
        assert!(replay.is_ok());
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn eval_scenario_empty_room_smoke_runs() {
        let args = eval_args(ScenarioArg::EmptyRoom, 1, 3, None);
        run_eval_scenario(args).await.unwrap();
    }

    #[tokio::test]
    async fn eval_scenario_obstacle_writes_report() {
        let temp_dir = temp_path("netherwick_eval_scenario_obstacle");
        let out = temp_dir.join("obstacle.json");
        let mut args = eval_args(
            ScenarioArg::ObstacleAvoidance,
            3,
            5,
            Some(out.to_string_lossy().to_string()),
        );
        args.memory_report = true;
        run_eval_scenario(args).await.unwrap();
        let report: serde_json::Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
        assert_eq!(report["scenario"], "obstacle-avoidance");
        assert_eq!(report["action_selector_mode"], "baseline");
        assert_eq!(report["episodes_detail"].as_array().unwrap().len(), 3);
        assert!(report["memory"]["places_visited"].as_u64().unwrap_or(0) > 0);
        assert!(
            report["episodes_detail"][0]["memory"]["danger_memory_ticks"]
                .as_u64()
                .unwrap_or(0)
                > 0
        );
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn eval_scenario_model_assisted_empty_room_runs_and_reports_stats() {
        let temp_dir = temp_path("netherwick_eval_scenario_model_assisted_empty");
        let out = temp_dir.join("empty-model-assisted.json");
        let mut args = eval_args(
            ScenarioArg::EmptyRoom,
            1,
            3,
            Some(out.to_string_lossy().to_string()),
        );
        args.action_selector = CliActionSelectorMode::ModelAssisted;
        run_eval_scenario(args).await.unwrap();
        let report: serde_json::Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
        assert_eq!(report["action_selector_mode"], "model-assisted");
        assert_eq!(report["summary"]["model_assisted_decisions"], 3);
        assert!(report["summary"]["mean_candidate_score"].is_number());
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn eval_scenario_model_assisted_charger_seeking_runs() {
        let mut args = eval_args(ScenarioArg::ChargerSeeking, 1, 3, None);
        args.action_selector = CliActionSelectorMode::ModelAssisted;
        run_eval_scenario(args).await.unwrap();
    }

    #[tokio::test]
    async fn eval_scenario_optional_ledger_writes_transitions() {
        let temp_dir = temp_path("netherwick_eval_scenario_ledger");
        let ledger_dir = temp_dir.join("ledger");
        let out = temp_dir.join("empty.json");
        let mut args = eval_args(
            ScenarioArg::EmptyRoom,
            1,
            4,
            Some(out.to_string_lossy().to_string()),
        );
        args.ledger = Some(ledger_dir.to_string_lossy().to_string());
        run_eval_scenario(args).await.unwrap();
        let transitions = JsonlLedger::new(&ledger_dir).transitions().await.unwrap();
        assert!(!transitions.is_empty());
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn sim_curriculum_writes_one_capture_per_episode() {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let temp_dir = std::env::temp_dir().join(format!("netherwick_curriculum_test_{now_ms}"));
        let ledger_dir = temp_dir.join("ledger");
        let capture_root = temp_dir.join("captures");

        let args = SimCurriculumArgs {
            scenario: ScenarioArg::PersonSpeakerRoom,
            episodes: 2,
            steps: 3,
            seed: 7,
            out: ledger_dir.to_str().unwrap().to_string(),
            capture_root: Some(capture_root.to_str().unwrap().to_string()),
            tick_ms: 100,
            validation_ratio: 0.25,
            test_ratio: 0.25,
            llm: LlmArgs::default(),
        };

        run_sim_curriculum(args).await.unwrap();

        let manifest_path = ledger_dir.join("manifest.json");
        assert!(manifest_path.exists());
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        assert_eq!(manifest["scenario"], "person-speaker-room");
        assert_eq!(manifest["episodes"], 2);
        assert_eq!(manifest["splits"]["train"], 0);
        assert_eq!(manifest["splits"]["validation"], 1);
        assert_eq!(manifest["splits"]["test"], 1);
        assert_eq!(
            manifest["episodes_detail"][0]["capture"],
            capture_root
                .join("episode-000")
                .to_string_lossy()
                .to_string()
        );
        assert!(capture_root
            .join("episode-000")
            .join("manifest.json")
            .exists());
        assert!(capture_root
            .join("episode-001")
            .join("manifest.json")
            .exists());
        let transitions = JsonlLedger::new(&ledger_dir).transitions().await.unwrap();
        assert!(!transitions.is_empty());

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_evaluate_behavior_command_writes_json_to_out() {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let temp_dir = std::env::temp_dir().join(format!("netherwick_eval_test_{}", now_ms));
        let ledger_dir = temp_dir.join("ledger");
        let session_dir = ledger_dir.join("2026-06-24");
        fs::create_dir_all(&session_dir).unwrap();

        let checkpoint_dir = temp_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).unwrap();

        // Write 5 mock transitions to have enough data for training and validation splits
        let mut transitions = Vec::new();
        for i in 0..5 {
            let transition = ExperienceTransition {
                id: uuid::Uuid::new_v4(),
                before_frame_id: uuid::Uuid::new_v4(),
                before: Now::blank(100 + i * 100, BodySense::default()),
                before_z: ExperienceLatent {
                    t_ms: 100 + i * 100,
                    z: vec![0.1; 4],
                    ..ExperienceLatent::default()
                },
                action: Some(ActionPrimitive::Stop),
                predicted_futures: Vec::new(),
                after: Now::blank(200 + i * 100, BodySense::default()),
                after_z: ExperienceLatent {
                    t_ms: 200 + i * 100,
                    z: vec![0.2; 4],
                    ..ExperienceLatent::default()
                },
                reward: Reward { value: 0.0 },
                surprise: SurpriseSense::default(),
                created_at_ms: 200 + i * 100,
            };
            transitions.push(transition);
        }

        let transitions_file = session_dir.join("transitions.jsonl");
        let mut content = String::new();
        for t in &transitions {
            content.push_str(&serde_json::to_string(t).unwrap());
            content.push('\n');
        }
        fs::write(&transitions_file, content).unwrap();

        // Train first to create the checkpoint and metadata
        netherwick_training::train_behavior(netherwick_training::TrainBehaviorRequest {
            behavior: netherwick_training::TrainableBehavior::Danger,
            ledger_path: ledger_dir.clone(),
            checkpoint_path: checkpoint_dir.clone(),
            epochs: 1,
            validation_split: 0.2,
            seed: 42,
        })
        .await
        .unwrap();

        // Prepare output path
        let out_json_path = temp_dir.join("report.json");

        let args = EvaluateBehaviorArgs {
            behavior: "danger".to_string(),
            ledger: ledger_dir.to_str().unwrap().to_string(),
            checkpoint: Some(checkpoint_dir.to_str().unwrap().to_string()),
            max_samples: None,
            out: Some(out_json_path.to_str().unwrap().to_string()),
        };

        let cmd = EvaluateCommand {
            model: EvaluateModel::Behavior(args),
        };

        let res = run_evaluate(cmd).await;
        assert!(res.is_ok(), "run_evaluate failed: {:?}", res.err());

        // Verify report file exists and has correct behavior name
        assert!(out_json_path.exists());
        let report_content = fs::read_to_string(&out_json_path).unwrap();
        let report: serde_json::Value = serde_json::from_str(&report_content).unwrap();
        assert_eq!(report["behavior"], "danger");

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
