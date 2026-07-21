const DEFAULT_LIVE_LLM_TIMEOUT_MS: u64 = 300_000;
const DEFAULT_MPU6050_IMU_DEVICE: &str = "/dev/i2c-1";
const DEFAULT_WHISPER_MODEL_FILENAME: &str = "ggml-tiny.en.bin";
const CREATE_SENSOR_STREAM_PACKET_ID: u8 = 0;
const CREATE_SENSOR_STREAM_PERIOD_MS: u32 = 250;
const CREATE_SENSOR_FRESHNESS_MAX_AGE_MS: u32 = 500;
const CREATE_SENSOR_READY_TIMEOUT_MS: u64 = 3_000;
const POSSESSION_SHUTDOWN_BUSY_RETRY_ATTEMPTS: usize = 20;
const POSSESSION_SHUTDOWN_BUSY_RETRY_DELAY: Duration = Duration::from_millis(100);

#[derive(Parser)]
#[command(name = "pete")]
#[command(about = "Pete CLI entrypoint")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Sim(SimArgs),
    SimCurriculum(SimCurriculumArgs),
    NeatTrain(NeatTrainArgs),
    DreamTrain(DreamTrainArgs),
    EvalScenario(EvalScenarioArgs),
    SocialExam(SocialExamArgs),
    MemoryInspect(MemoryInspectArgs),
    Mouth(MouthArgs),
    Robot(RobotArgs),
    WhisperTranscribe(WhisperTranscribeArgs),
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
    PoseGraphReport(PoseGraphReportArgs),
    GeometryDebug(GeometryDebugArgs),
    RepresentationReport(RepresentationReportArgs),
    ModelRegister(ModelRegisterArgs),
    ModelStatus,
    ModelPromote(ModelPromoteArgs),
    CompareScenarioReports(CompareScenarioReportsArgs),
    Dashboard,
    VirtualReport(VirtualReportArgs),
    RetinaMockSend(RetinaMockSendArgs),
    EmbodiedDemo(EmbodiedDemoArgs),
    EmbodiedEval(EmbodiedEvalArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    let cli = Cli::parse();
    match cli.command {
        Command::Sim(args) => run_sim(args).await,
        Command::SimCurriculum(args) => run_sim_curriculum(args).await,
        Command::NeatTrain(args) => run_neat_train(args).await,
        Command::DreamTrain(args) => run_dream_train(args).await,
        Command::EvalScenario(args) => run_eval_scenario(args).await,
        Command::SocialExam(args) => run_social_exam_command(args).await,
        Command::MemoryInspect(args) => memory_inspect(args).await,
        Command::Mouth(args) => run_mouth(args),
        Command::Robot(args) => run_robot(args).await,
        Command::WhisperTranscribe(args) => run_whisper_transcribe(args),
        Command::HardwareEnv(args) => hardware_env(args).await,
        Command::CaptureSim(args) => capture_sim(args).await,
        Command::CaptureReal(args) => capture_real(args).await,
        Command::CaptureAssets(args) => capture_assets(args).await,
        Command::InspectCapture(args) => inspect_capture(args).await,
        Command::ReplayCapture(args) => replay_capture(args).await,
        Command::ReplayCounterfactual(args) => replay_counterfactual(args).await,
        Command::InspectLedger(args) => inspect_ledger(args).await,
        Command::PoseGraphReport(args) => run_pose_graph_report(args).await,
        Command::GeometryDebug(args) => run_geometry_debug(args).await,
        Command::RepresentationReport(args) => run_representation_report(args).await,
        Command::Train(command) => run_train(command).await,
        Command::Evaluate(command) => run_evaluate(command).await,
        Command::Promote(command) => run_promote(command),
        Command::ModelRegister(args) => model_register(args),
        Command::ModelStatus => model_status(),
        Command::ModelPromote(args) => model_promote(args),
        Command::CompareScenarioReports(args) => compare_scenario_reports_command(args),
        Command::VirtualReport(args) => run_virtual_report(args).await,
        Command::RetinaMockSend(args) => run_retina_mock_send(args).await,
        Command::EmbodiedDemo(args) => run_embodied_demo(args).await,
        Command::EmbodiedEval(args) => run_embodied_eval(args).await,
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
    #[arg(long, value_enum, default_value = "goal")]
    action_selector: CliActionSelectorMode,
    #[arg(long, env = "PETE_DREAM_POLICY_CHECKPOINT")]
    dream_policy_checkpoint: Option<String>,
    #[arg(long, env = "PETE_INLINE_LEARNING")]
    inline_learning: bool,
    #[arg(
        long,
        value_enum,
        default_value = "world-outcome",
        env = "PETE_INLINE_LEARNING_MODE"
    )]
    inline_learning_mode: InlineLearningModeArg,
    #[arg(long, default_value_t = 1, env = "PETE_INLINE_TRAIN_STEPS_PER_TICK")]
    inline_train_steps_per_tick: usize,
    #[arg(long, env = "PETE_INLINE_BEHAVIORS")]
    inline_behaviors: Option<String>,
    #[arg(long)]
    live: bool,
    #[arg(long, default_value = "127.0.0.1:8787")]
    live_addr: SocketAddr,
    #[arg(long)]
    live_tls: bool,
    #[arg(long, default_value = "certs/pete-dev.crt")]
    live_tls_cert: String,
    #[arg(long, default_value = "certs/pete-dev.key")]
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
struct NeatTrainArgs {
    /// Stable replaceable behavior id. V0 supports only `locomotion`.
    behavior: String,
    #[arg(long, default_value_t = 6)]
    generations_per_stage: usize,
    #[arg(long, default_value_t = 32)]
    population: usize,
    #[arg(long, default_value_t = 3)]
    episodes_per_genome: usize,
    #[arg(long, default_value_t = 220)]
    steps: usize,
    #[arg(long, default_value_t = 500)]
    transfer_episodes: usize,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    /// Held-out audit seed root. These seeds are never used for selection.
    #[arg(long, default_value_t = 9_000_001)]
    heldout_seed: u64,
    /// Validation seed root. Validation worlds never participate in selection.
    #[arg(long, default_value_t = 8_000_001)]
    validation_seed: u64,
    /// Validate the current champion every N generations.
    #[arg(long, default_value_t = 4)]
    validation_every: usize,
    /// Consecutive validation passes required before advancing a stage.
    #[arg(long, default_value_t = 2)]
    validation_passes: usize,
    #[arg(long, default_value_t = 2.2)]
    compatibility_threshold: f32,
    #[arg(long, default_value_t = 0.05)]
    compatibility_threshold_floor: f32,
    #[arg(long, default_value_t = 4)]
    target_species_min: usize,
    #[arg(long, default_value_t = 9)]
    target_species_max: usize,
    #[arg(long, default_value = "data/models/locomotion_neat_v0")]
    checkpoint: String,
    #[arg(long, default_value = "data/reports/neat/locomotion")]
    report_dir: String,
    /// Atomic full evolutionary-state checkpoint.
    #[arg(
        long,
        default_value = "data/reports/neat/locomotion/trainer-state.json"
    )]
    state_checkpoint: String,
    /// Resume a full evolutionary-state checkpoint.
    #[arg(long)]
    resume: Option<String>,
    /// Migrate and rewrite a resumed state without evaluating a generation.
    #[arg(long)]
    migrate_only: bool,
    /// Reconstruct a founder population from a completed legacy report.
    #[arg(long)]
    founders_report: Option<String>,
    /// First stage for a reconstructed founder population.
    #[arg(long)]
    start_stage: Option<String>,
    #[arg(long, default_value = "data/captures/neat/locomotion")]
    capture_root: String,
    #[arg(long, default_value = "configs/models.toml")]
    models_config: String,
    /// Capture every Nth generation champion; stage champions are always captured.
    #[arg(long, default_value_t = 2)]
    capture_every: usize,
    /// Selection bonus multiplier for population-level behavioral novelty.
    #[arg(long, default_value_t = 25.0)]
    novelty_weight: f32,
    /// Number of nearest behavior descriptors used to score novelty.
    #[arg(long, default_value_t = 10)]
    novelty_neighbors: usize,
    /// Maximum historical behavior descriptors retained for novelty search.
    #[arg(long, default_value_t = 512)]
    novelty_archive_limit: usize,
    /// Maximum historically difficult procedural worlds retained for replay/mutation.
    #[arg(long, default_value_t = 128)]
    world_archive_limit: usize,
    /// Fraction of each generation's episodes reserved for old hard-world replay.
    #[arg(long, default_value_t = 0.25)]
    world_replay_ratio: f32,
    /// Fraction of each generation's episodes mutated from archived hard worlds.
    #[arg(long, default_value_t = 0.25)]
    world_mutation_ratio: f32,
    /// Fraction of late-stage training worlds drawn from earlier curriculum stages.
    #[arg(long, default_value_t = 0.20)]
    rehearsal_ratio: f32,
    /// Paired held-out episodes used to qualify specialist archive labels.
    #[arg(long, default_value_t = 16)]
    niche_audit_episodes: usize,
    /// Leave the trained candidate as a report artifact instead of promoting winners.
    #[arg(long)]
    no_promote: bool,
}

#[derive(Debug, Parser)]
struct DreamTrainArgs {
    #[arg(long, value_enum, default_value = "motion")]
    start_level: DreamLevelArg,
    #[arg(long, default_value_t = 30)]
    generations: usize,
    #[arg(long, default_value_t = 32)]
    population: usize,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long, default_value_t = 12)]
    hidden_dim: usize,
    #[arg(long, default_value = "data/models/dream-policy/neat")]
    checkpoint_dir: String,
    #[arg(long, default_value = "datasets/dream-policy/v0/episodes")]
    dataset_dir: String,
    #[arg(long, default_value_t = true, num_args = 0..=1, default_missing_value = "true")]
    export_dataset: bool,
    #[arg(long, default_value_t = false)]
    detailed_logs: bool,
    #[arg(long, default_value_t = false)]
    clear: bool,
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
    #[arg(long, value_enum, default_value = "goal")]
    action_selector: CliActionSelectorMode,
    #[command(flatten)]
    llm: LlmArgs,
}

#[derive(Debug, Parser)]
struct SocialExamArgs {
    /// Write the complete machine-readable exam report to this path.
    #[arg(long)]
    out: Option<String>,
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
    GoalShadow,
    Goal,
}

impl From<CliActionSelectorMode> for ActionSelectorMode {
    fn from(value: CliActionSelectorMode) -> Self {
        match value {
            CliActionSelectorMode::Baseline => ActionSelectorMode::Baseline,
            CliActionSelectorMode::Random => ActionSelectorMode::Random,
            CliActionSelectorMode::ModelAssisted => ActionSelectorMode::ModelAssisted,
            CliActionSelectorMode::Scripted => ActionSelectorMode::Scripted,
            CliActionSelectorMode::GoalShadow => ActionSelectorMode::GoalShadow,
            CliActionSelectorMode::Goal => ActionSelectorMode::Goal,
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
    ConcaveTrap,
    ColumnTrap,
    ChargerSeeking,
    PersonSpeakerRoom,
    MixedRoom,
    Dream,
}

impl From<ScenarioArg> for ScenarioKind {
    fn from(value: ScenarioArg) -> Self {
        match value {
            ScenarioArg::EmptyRoom => ScenarioKind::EmptyRoom,
            ScenarioArg::ObstacleAvoidance => ScenarioKind::ObstacleAvoidance,
            ScenarioArg::CornerTrap => ScenarioKind::CornerTrap,
            ScenarioArg::ConcaveTrap => ScenarioKind::ConcaveTrap,
            ScenarioArg::ColumnTrap => ScenarioKind::ColumnTrap,
            ScenarioArg::ChargerSeeking => ScenarioKind::ChargerSeeking,
            ScenarioArg::PersonSpeakerRoom => ScenarioKind::PersonAndSpeaker,
            ScenarioArg::MixedRoom => ScenarioKind::MixedRoom,
            ScenarioArg::Dream => ScenarioKind::Dream,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum DreamLevelArg {
    Motion,
    ObstacleAvoidance,
    EscapeTrap,
    ChargerSeeking,
    SocialInspection,
    PlaceMemory,
    WeirdDream,
}

impl From<DreamLevelArg> for DreamLevel {
    fn from(value: DreamLevelArg) -> Self {
        match value {
            DreamLevelArg::Motion => DreamLevel::Motion,
            DreamLevelArg::ObstacleAvoidance => DreamLevel::ObstacleAvoidance,
            DreamLevelArg::EscapeTrap => DreamLevel::EscapeTrap,
            DreamLevelArg::ChargerSeeking => DreamLevel::ChargerSeeking,
            DreamLevelArg::SocialInspection => DreamLevel::SocialInspection,
            DreamLevelArg::PlaceMemory => DreamLevel::PlaceMemory,
            DreamLevelArg::WeirdDream => DreamLevel::WeirdDream,
        }
    }
}

#[derive(Debug, Parser)]
struct RobotArgs {
    #[arg(long, value_enum, default_value = "read-only")]
    mode: RobotModeArg,
    #[arg(long, value_enum, default_value = "uart")]
    cockpit: CockpitBackendArg,
    #[arg(long, default_value = "auto")]
    create_port: String,
    /// Brainstem HTTP address used by the Wi-Fi Cockpit backend.
    #[arg(
        long,
        default_value = "192.168.4.1:80",
        env = "PETE_BRAINSTEM_HTTP_HOST"
    )]
    brainstem_host: String,
    /// Loopback address of an RPi 5 Brainstem process.
    #[arg(
        long,
        default_value = "127.0.0.1:8787",
        env = "PETE_BRAINSTEM_LOCAL_ADDR"
    )]
    brainstem_local: SocketAddr,
    #[arg(long, default_value_t = 57_600)]
    create_baud: u32,
    #[arg(long, default_value = "data/ledger/robot-readonly")]
    ledger: String,
    #[arg(long)]
    camera: Option<String>,
    #[arg(long)]
    kinect_depth: bool,
    #[arg(long, default_value_t = 0)]
    kinect_index: i32,
    #[arg(long, default_value_t = 0.32)]
    kinect_rgb_target_luma: f32,
    #[arg(long, default_value_t = 3.0)]
    kinect_rgb_auto_gain_max: f32,
    #[arg(long, default_value_t = 1.0)]
    kinect_rgb_gain: f32,
    #[arg(long, default_value_t = 0.80)]
    kinect_rgb_gamma: f32,
    #[arg(long, default_value_t = 0.0)]
    kinect_rgb_brightness: f32,
    #[arg(long)]
    kinect_rgb_raw: bool,
    #[arg(long)]
    mic: Option<String>,
    #[arg(long)]
    asr_command: Option<String>,
    #[arg(long)]
    imu: Option<String>,
    /// Fusion IMU policy. Automatic discovery is the normal mode; overrides never bypass trust.
    #[arg(long, value_enum, default_value = "auto", env = "PETE_IMU_SOURCE")]
    imu_source: ImuSourceArg,
    #[arg(long)]
    gps: Option<String>,
    /// HLS-LFCD2 / LDS-01 serial device. Omit for best-effort auto-detection.
    #[arg(long, env = "LIDAR_SERIAL_PORT")]
    lidar: Option<String>,
    /// Counter-clockwise mounting offset from the robot's forward axis.
    #[arg(long, default_value_t = 0.0, env = "LIDAR_YAW_DEG")]
    lidar_yaw_deg: f32,
    /// Downward tilt of the lidar scan plane; positive values look toward the ground ahead.
    #[arg(long, default_value_t = 0.0, env = "LIDAR_PITCH_DEG")]
    lidar_pitch_deg: f32,
    #[arg(long, default_value_t = 0.0, env = "LIDAR_ROLL_DEG")]
    lidar_roll_deg: f32,
    #[arg(long, default_value_t = 0.0, env = "LIDAR_HEIGHT_M")]
    lidar_height_m: f32,
    #[arg(long, default_value_t = 0.0, env = "LIDAR_FORWARD_M")]
    lidar_forward_m: f32,
    #[arg(long, default_value_t = 0.0, env = "LIDAR_LEFT_M")]
    lidar_left_m: f32,
    #[arg(long)]
    capture: Option<String>,
    #[arg(long)]
    dashboard: Option<SocketAddr>,
    #[arg(long)]
    dashboard_tls: bool,
    #[arg(long, default_value = "certs/pete-dev.crt")]
    dashboard_tls_cert: String,
    #[arg(long, default_value = "certs/pete-dev.key")]
    dashboard_tls_key: String,
    #[arg(long, default_value_t = 100)]
    tick_ms: u64,
    /// Expected brainstem device identity. Required for physical possession.
    #[arg(long)]
    brainstem_device_id: Option<String>,
    /// Expected boot identity. Required for physical possession and reconnect.
    #[arg(long)]
    brainstem_boot_id: Option<String>,
    #[arg(long, default_value_t = 50)]
    max_linear_mm_s: i16,
    #[arg(long, default_value_t = 500)]
    max_angular_mrad_s: i16,
    /// Permit executive-selected actions to drive physical wheels in regular possession mode.
    #[arg(long)]
    autonomous_motion: bool,
    /// Acknowledge that direct-RPi floor motion can persist through a whole-Pi freeze.
    #[arg(long, conflicts_with = "wheels_off_floor")]
    acknowledge_no_independent_watchdog: bool,
    /// Run the guarded physical bump-to-recovery possession smoke test and exit.
    #[arg(long)]
    recovery_smoke: bool,
    /// Calibrate IMU down from gravity, then run a tiny in-place spin probe and exit.
    #[arg(long)]
    orientation_probe: bool,
    /// Confirm that physical drive wheels are clear of the floor.
    #[arg(long)]
    wheels_off_floor: bool,
    #[arg(long, default_value_t = 250)]
    reconnect_initial_backoff_ms: u64,
    #[arg(long, default_value_t = 5_000)]
    reconnect_max_backoff_ms: u64,
    #[arg(long)]
    steps: Option<usize>,
    #[arg(long)]
    duration_seconds: Option<u64>,
    #[arg(long)]
    require_camera: bool,
    #[arg(long)]
    require_kinect: bool,
    #[arg(long)]
    require_mic: bool,
    #[arg(long)]
    require_imu: bool,
    #[arg(long)]
    require_gps: bool,
    #[arg(long)]
    require_lidar: bool,
    #[arg(long)]
    require_llm: bool,
    /// Time allowed for configured sensors to emit their first usable packet.
    #[arg(
        long,
        default_value_t = 3_000,
        env = "PETE_SENSOR_READINESS_TIMEOUT_MS"
    )]
    sensor_readiness_timeout_ms: u64,
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
    /// Motor-enabled possession mode.
    Regular,
    ReadOnly,
    PossessionSlow,
    Disabled,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
enum ImuSourceArg {
    #[default]
    Auto,
    Brainstem,
    LocalI2c,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CockpitBackendArg {
    Sim,
    Uart,
    Wifi,
    Local,
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
    #[arg(long, value_enum, default_value = "uart")]
    cockpit: CockpitBackendArg,
    #[arg(long, default_value = "auto")]
    create_port: String,
    /// Brainstem HTTP address used by the Wi-Fi Cockpit backend.
    #[arg(
        long,
        default_value = "192.168.4.1:80",
        env = "PETE_BRAINSTEM_HTTP_HOST"
    )]
    brainstem_host: String,
    /// Loopback address of an RPi 5 Brainstem process.
    #[arg(
        long,
        default_value = "127.0.0.1:8787",
        env = "PETE_BRAINSTEM_LOCAL_ADDR"
    )]
    brainstem_local: SocketAddr,
    #[arg(long, default_value_t = 57_600)]
    create_baud: u32,
    #[arg(long)]
    camera: Option<String>,
    #[arg(long)]
    kinect_depth: bool,
    #[arg(long, default_value_t = 0)]
    kinect_index: i32,
    #[arg(long, default_value_t = 0.32)]
    kinect_rgb_target_luma: f32,
    #[arg(long, default_value_t = 3.0)]
    kinect_rgb_auto_gain_max: f32,
    #[arg(long, default_value_t = 1.0)]
    kinect_rgb_gain: f32,
    #[arg(long, default_value_t = 0.80)]
    kinect_rgb_gamma: f32,
    #[arg(long, default_value_t = 0.0)]
    kinect_rgb_brightness: f32,
    #[arg(long)]
    kinect_rgb_raw: bool,
    #[arg(long)]
    mic: Option<String>,
    #[arg(long)]
    imu: Option<String>,
    #[arg(long)]
    gps: Option<String>,
    /// HLS-LFCD2 / LDS-01 serial device. Omit for best-effort auto-detection.
    #[arg(long, env = "LIDAR_SERIAL_PORT")]
    lidar: Option<String>,
    /// Counter-clockwise mounting offset from the robot's forward axis.
    #[arg(long, default_value_t = 0.0, env = "LIDAR_YAW_DEG")]
    lidar_yaw_deg: f32,
    /// Downward tilt of the lidar scan plane; positive values look toward the ground ahead.
    #[arg(long, default_value_t = 0.0, env = "LIDAR_PITCH_DEG")]
    lidar_pitch_deg: f32,
    #[arg(long, default_value_t = 0.0, env = "LIDAR_ROLL_DEG")]
    lidar_roll_deg: f32,
    #[arg(long, default_value_t = 0.0, env = "LIDAR_HEIGHT_M")]
    lidar_height_m: f32,
    #[arg(long, default_value_t = 0.0, env = "LIDAR_FORWARD_M")]
    lidar_forward_m: f32,
    #[arg(long, default_value_t = 0.0, env = "LIDAR_LEFT_M")]
    lidar_left_m: f32,
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
    #[arg(long)]
    world_pointcloud: bool,
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
    LatentRoundTrip(TrainLatentRoundTripArgs),
    UnifiedExperience(TrainUnifiedExperienceArgs),
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
struct TrainLatentRoundTripArgs {
    #[arg(long, default_value = "data/ledger")]
    ledger: String,
    #[arg(long, default_value_t = 5)]
    epochs: usize,
    #[arg(long, default_value = "data/models/latent_round_trip_v0")]
    checkpoint: String,
    #[arg(long, default_value = "data/reports/latent-round-trip.json")]
    report: String,
    #[arg(long, default_value_t = 16)]
    z_dim: usize,
    #[arg(long, default_value_t = 0.2)]
    validation_split: f32,
    #[arg(long, default_value_t = 7)]
    seed: u64,
    #[arg(long)]
    codebook_size: Option<usize>,
}

#[derive(Debug, Parser)]
struct TrainUnifiedExperienceArgs {
    #[arg(long, default_value = "data/ledger")]
    ledger: String,
    #[arg(long, default_value_t = 5)]
    epochs: usize,
    #[arg(long, default_value = "data/models/unified_experience/latest")]
    checkpoint: String,
    #[arg(long, default_value = "data/reports/unified-experience/latest.json")]
    report: String,
    #[arg(long, default_value_t = 16)]
    z_dim: usize,
    #[arg(long, default_value_t = 16)]
    teacher_dim: usize,
    #[arg(long, default_value_t = 0.2)]
    validation_split: f32,
    #[arg(long, default_value_t = 7)]
    seed: u64,
}

#[derive(Debug, Parser)]
struct InspectLedgerArgs {
    #[arg(long, default_value = "data/ledger/virtual-live", env = "PETE_LEDGER")]
    ledger: String,
}

#[derive(Debug, Parser)]
struct VirtualReportArgs {
    #[arg(long, default_value = "data/ledger/virtual-live", env = "PETE_LEDGER")]
    ledger: String,
    #[arg(long, default_value = "data/reports/virtual/latest.json")]
    out: String,
}

#[derive(Debug, Parser)]
struct PoseGraphReportArgs {
    #[arg(long, default_value = "data/ledger/virtual-live", env = "PETE_LEDGER")]
    ledger: String,
    #[arg(long)]
    capture: Option<String>,
    #[arg(long, default_value = "data/reports/pose-graph/latest.json")]
    out: String,
    #[arg(long, default_value_t = 0.25)]
    min_node_distance_m: f32,
    #[arg(long, default_value_t = 15.0)]
    min_node_degrees: f32,
    #[arg(long, default_value_t = 10)]
    max_ticks_between_nodes: u64,
    #[arg(long, default_value_t = 0.85)]
    min_loop_confidence: f32,
}

#[derive(Debug, Parser)]
struct GeometryDebugArgs {
    #[arg(long)]
    capture: Option<String>,
    #[arg(long)]
    live_now_url: Option<String>,
    #[arg(long, default_value = "data/reports/geometry/latest.json")]
    out: String,
    #[arg(long, default_value_t = 16)]
    samples: usize,
    #[arg(long, default_value_t = 0.02)]
    max_below_floor_ratio: f32,
    #[arg(long, default_value_t = 200)]
    max_body_timestamp_age_ms: u64,
    #[arg(long, default_value_t = 200)]
    max_kinect_timestamp_age_ms: u64,
    #[arg(long, default_value_t = 200)]
    max_imu_timestamp_age_ms: u64,
    #[arg(long, default_value_t = 100)]
    max_kinect_imu_skew_ms: u64,
    #[arg(long, default_value_t = 100)]
    max_kinect_body_skew_ms: u64,
    #[arg(long, default_value_t = 50)]
    max_rgbd_skew_ms: u64,
    #[arg(long, default_value_t = 2)]
    min_depth_frames: usize,
    #[arg(long, default_value_t = 45.0)]
    min_stationary_rotation_deg: f32,
    #[arg(long, default_value_t = 0.20)]
    max_stationary_translation_m: f32,
    #[arg(long, default_value_t = 0.05)]
    min_stationary_stable_voxel_ratio: f32,
    #[arg(long, default_value_t = 1.50)]
    max_stationary_stable_z_span_m: f32,
}

#[derive(Debug, Parser)]
struct RepresentationReportArgs {
    #[arg(long, default_value = "data/ledger/virtual-live", env = "PETE_LEDGER")]
    ledger: String,
    #[arg(long)]
    capture: Option<String>,
    #[arg(long, default_value = "data/reports/representation/latest.json")]
    out: String,
}

#[derive(Debug, Parser)]
struct EmbodiedDemoArgs {
    #[arg(long)]
    json: bool,
    #[arg(long)]
    ledger: Option<String>,
}

#[derive(Debug, Parser)]
struct EmbodiedEvalArgs {
    #[arg(long, value_enum, default_value = "deterministic")]
    fixture: EmbodiedEvalFixtureArg,
    #[arg(long)]
    json: bool,
    #[arg(long, value_enum)]
    omit: Vec<EmbodiedEvalOmissionArg>,
}

#[derive(Clone, Debug, ValueEnum)]
enum EmbodiedEvalFixtureArg {
    Deterministic,
}

#[derive(Clone, Debug, ValueEnum)]
enum EmbodiedEvalOmissionArg {
    PrimarySensations,
    Descendants,
    Vectors,
    Impressions,
    FusedExperience,
    SummaryImpression,
    Predictions,
    MemoryPersistence,
    MemoryLinks,
    Recall,
}

impl From<EmbodiedEvalOmissionArg> for EmbodiedEvalOmission {
    fn from(value: EmbodiedEvalOmissionArg) -> Self {
        match value {
            EmbodiedEvalOmissionArg::PrimarySensations => Self::PrimarySensations,
            EmbodiedEvalOmissionArg::Descendants => Self::Descendants,
            EmbodiedEvalOmissionArg::Vectors => Self::Vectors,
            EmbodiedEvalOmissionArg::Impressions => Self::Impressions,
            EmbodiedEvalOmissionArg::FusedExperience => Self::FusedExperience,
            EmbodiedEvalOmissionArg::SummaryImpression => Self::SummaryImpression,
            EmbodiedEvalOmissionArg::Predictions => Self::Predictions,
            EmbodiedEvalOmissionArg::MemoryPersistence => Self::MemoryPersistence,
            EmbodiedEvalOmissionArg::MemoryLinks => Self::MemoryLinks,
            EmbodiedEvalOmissionArg::Recall => Self::Recall,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct RepresentationHealthReport {
    schema_version: u32,
    frame_count: usize,
    input: RepresentationInputSummary,
    warnings: Vec<String>,
    entity_memory: RepresentationEntityMemorySummary,
    map: RepresentationMapSummary,
    pose_graph: RepresentationPoseGraphSummary,
    place_recognition: RepresentationPlaceRecognitionSummary,
    return_to_start: ReturnToStartValidation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ReturnToStartValidation {
    evaluated: bool,
    passed: bool,
    measured_loop_constraint: bool,
    graph_error_before: f32,
    graph_error_after: f32,
    graph_error_reduced: bool,
    wall_overlap_before: Option<f32>,
    wall_overlap_after: Option<f32>,
    wall_overlap_improved: bool,
    raw_final_distance_to_start_m: Option<f32>,
    corrected_final_distance_to_start_m: Option<f32>,
    max_corrected_distance_to_start_m: f32,
    corrected_pose_near_start: bool,
    reasons: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct RepresentationInputSummary {
    source_type: String,
    source_path: String,
    provenance: HashMap<String, usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct RepresentationEntityMemorySummary {
    total_entities: usize,
    active_entities: usize,
    occluded_entities: usize,
    vanished_entities: usize,
    revived_entities: usize,
    modality_support_counts: HashMap<String, usize>,
    constellation_edges_by_relation: HashMap<String, usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct RepresentationMapSummary {
    local_occupancy_cell_count: usize,
    pose_history_length: usize,
    point_cloud_voxel_count: usize,
    stable_voxel_count: usize,
    transient_voxel_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct RepresentationPoseGraphSummary {
    node_count: usize,
    odometry_edge_count: usize,
    loop_candidate_count: usize,
    loop_accepted_count: usize,
    loop_rejected_count: usize,
    confidence_distribution: RepresentationConfidenceDistribution,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct RepresentationPlaceRecognitionSummary {
    candidates_emitted: usize,
    candidate_kinds: HashMap<String, usize>,
    confidence_distribution: RepresentationConfidenceDistribution,
    repeated_place_hints: Vec<String>,
    warnings: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct RepresentationConfidenceDistribution {
    min: Option<f32>,
    max: Option<f32>,
    mean: Option<f32>,
    buckets: HashMap<String, usize>,
}

#[derive(Debug, Parser)]
struct TrainVirtualArgs {
    #[arg(long, default_value = "data/ledger/virtual-live", env = "PETE_LEDGER")]
    ledger: String,
    #[arg(
        long,
        default_value = "data/models/virtual/latest",
        env = "PETE_MODEL_OUT"
    )]
    out_dir: String,
    #[arg(long, default_value = "data/reports/virtual/latest.json")]
    report_out: String,
    #[arg(long, default_value_t = 5, env = "PETE_EPOCHS")]
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
    comparison_report: Option<String>,
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
    #[arg(long)]
    comparison_report: Option<String>,
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
    #[arg(long)]
    name: Option<String>,
    #[arg(long)]
    out: Option<String>,
}

#[derive(Debug, Parser)]
struct MouthArgs {
    #[arg(default_value = "Hello. My name is Pete.")]
    text: String,
}

#[derive(Debug, Parser)]
struct WhisperTranscribeArgs {
    #[arg(long)]
    model: Option<PathBuf>,
    wav: PathBuf,
}
