use std::fs;
use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use netherwick_autonomic::SimpleSafety;
use netherwick_behaviors::BehaviorRegime;
use netherwick_body::RobotBody;
use netherwick_conductor::SimpleConductor;
use netherwick_create1::{Create1Body, MockCreate1Body};
use netherwick_ledger::{
    ExperienceFrame, ExperienceTransition, JsonlLedger, LedgerReader, LedgerWriter,
};
use netherwick_llm::NoopLlmAgent;
use netherwick_memory::InMemoryExperienceStore;
use netherwick_models::MODEL_REGISTRY;
use netherwick_runtime::{
    MinimalRuntime, RealRobotRunner, RobotMode, RuntimeModelStack, SimRunner,
};
use netherwick_sensors::{
    CameraSenseProvider, GpsSenseProvider, ImuSenseProvider, MicrophoneSenseProvider, SenseProducer,
};
use netherwick_server::LiveViewState;
use netherwick_sim::{build_scenario, default_sim_world, ScenarioConfig, ScenarioKind};
use netherwick_training::{
    evaluate_behavior, load_models_config, promote_behavior_config, train_behavior,
    EvaluateBehaviorRequest, TrainBehaviorRequest, TrainableBehavior,
};
use netherwick_worldlab::{CaptureReader, CaptureReplayRunner, CaptureSource, CaptureWriter};
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
    Robot(RobotArgs),
    Replay,
    CaptureSim(CaptureSimArgs),
    ReplayCapture(ReplayCaptureArgs),
    Train(TrainCommand),
    Evaluate(EvaluateCommand),
    Promote(PromoteCommand),
    InspectLedger,
    ModelStatus,
    Dashboard,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Sim(args) => run_sim(args).await,
        Command::SimCurriculum(args) => run_sim_curriculum(args).await,
        Command::Robot(args) => run_robot(args).await,
        Command::CaptureSim(args) => capture_sim(args).await,
        Command::ReplayCapture(args) => replay_capture(args).await,
        Command::InspectLedger => inspect_ledger().await,
        Command::Train(command) => run_train(command).await,
        Command::Evaluate(command) => run_evaluate(command).await,
        Command::Promote(command) => run_promote(command),
        Command::ModelStatus => model_status(),
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
    #[arg(long)]
    live: bool,
    #[arg(long, default_value = "127.0.0.1:8787")]
    live_addr: SocketAddr,
    #[arg(long, default_value_t = 100)]
    tick_delay_ms: u64,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ScenarioArg {
    EmptyRoom,
    ObstacleAvoidance,
    ChargerSeeking,
    PersonSpeakerRoom,
    MixedRoom,
}

impl From<ScenarioArg> for ScenarioKind {
    fn from(value: ScenarioArg) -> Self {
        match value {
            ScenarioArg::EmptyRoom => ScenarioKind::EmptyRoom,
            ScenarioArg::ObstacleAvoidance => ScenarioKind::ObstacleAvoidance,
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
    #[arg(long, default_value = "mock")]
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
    require_camera: bool,
    #[arg(long)]
    require_mic: bool,
    #[arg(long)]
    require_imu: bool,
    #[arg(long)]
    require_gps: bool,
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
}

#[derive(Debug, Parser)]
struct ReplayCaptureArgs {
    #[arg(long)]
    capture: String,
    #[arg(long, default_value = "data/ledger/replay-test")]
    ledger: String,
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

async fn run_sim(args: SimArgs) -> Result<()> {
    let ledger = JsonlLedger::new(&args.ledger);
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime::with_default_events(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        NoopLlmAgent,
    );
    if let Some(models) = load_runtime_models(&args)? {
        runtime = runtime.with_models(models);
    }

    let (world, motors) = default_sim_world(args.seed);
    let mut runner = SimRunner::new(runtime, world, motors);
    if args.live {
        let live_state = LiveViewState::new();
        let server_state = live_state.clone();
        let live_addr = args.live_addr;
        tokio::spawn(async move {
            if let Err(error) = netherwick_server::serve_live_view(live_addr, server_state).await {
                eprintln!("live robot view server stopped: {error}");
            }
        });
        println!("live robot view: http://{}/view", args.live_addr);
        for _ in 0..args.steps {
            runner
                .run_steps_observing(1, |snapshot| live_state.update(snapshot.clone()))
                .await?;
            tokio::time::sleep(Duration::from_millis(args.tick_delay_ms)).await;
        }
    } else {
        runner.run_steps(args.steps).await?;
    }
    println!(
        "sim complete: {} ticks, seed {}, ledger {}, danger_mode {:?}, charge_mode {:?}, action_value_mode {:?}, eye_next_mode {:?}, ear_next_mode {:?}, experience_mode {:?}",
        runner.tick_count,
        args.seed,
        args.ledger,
        args.danger_mode,
        args.charge_mode,
        args.action_value_mode,
        args.eye_next_mode,
        args.ear_next_mode,
        args.experience_mode
    );
    Ok(())
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
        let runtime = default_runtime(ledger.clone());
        let mut runner = SimRunner::new(runtime, scenario.world, scenario.motors);
        runner.tick_ms = args.tick_ms;
        let mut capture_path_for_manifest = None;

        if let Some(root) = &args.capture_root {
            let mut snapshots = Vec::with_capacity(args.steps);
            runner
                .run_steps_observing(args.steps, |snapshot| snapshots.push(snapshot.clone()))
                .await?;
            let capture_path = Path::new(root).join(format!("episode-{episode_index:05}"));
            capture_path_for_manifest = Some(capture_path.to_string_lossy().to_string());
            let mut writer =
                CaptureWriter::create(&capture_path, CaptureSource::Sim, Some(args.tick_ms))
                    .await?;
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

async fn capture_sim(args: CaptureSimArgs) -> Result<()> {
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let runtime = MinimalRuntime::with_default_events(
        NoopLedger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        NoopLlmAgent,
    );
    let (world, motors) = default_sim_world(args.seed);
    let mut runner = SimRunner::new(runtime, world, motors);
    let mut snapshots = Vec::new();
    runner
        .run_steps_observing(args.steps, |snapshot| snapshots.push(snapshot.clone()))
        .await?;

    let mut writer =
        CaptureWriter::create(&args.out, CaptureSource::Sim, Some(args.tick_ms)).await?;
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
        NoopLlmAgent,
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

    let body: Box<dyn RobotBody + Send> = if args.create_port == "mock" {
        Box::new(MockCreate1Body::new())
    } else {
        Box::new(Create1Body::connect(&args.create_port, args.create_baud).await?)
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
        NoopLlmAgent,
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

    while args
        .steps
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
) -> MinimalRuntime<
    JsonlLedger,
    InMemoryExperienceStore,
    InMemoryExperienceStore,
    SimpleConductor,
    SimpleSafety,
    NoopLlmAgent,
> {
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    MinimalRuntime::with_default_events(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        NoopLlmAgent,
    )
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

fn load_runtime_models(args: &SimArgs) -> Result<Option<RuntimeModelStack>> {
    if args.danger_mode != DangerMode::ShadowInfer
        && args.charge_mode != ChargeMode::ShadowInfer
        && args.action_value_mode != ActionValueMode::ShadowInfer
        && args.future_mode == FutureMode::Hardcoded
        && args.eye_next_mode != EyeNextMode::ShadowInfer
        && args.ear_next_mode != EarNextMode::ShadowInfer
        && args.experience_mode == ExperienceMode::Off
    {
        return Ok(None);
    }
    let danger_path = if args.danger_mode == DangerMode::ShadowInfer {
        match &args.danger_checkpoint {
            Some(checkpoint) if Path::new(checkpoint).exists() => {
                let path = Path::new(checkpoint);
                println!("loaded danger checkpoint: {}", path.display());
                Some(path)
            }
            Some(checkpoint) => {
                println!(
                    "danger shadow inference disabled: checkpoint not found at {}",
                    checkpoint
                );
                None
            }
            None => {
                println!("danger shadow inference disabled: no --danger-checkpoint provided");
                None
            }
        }
    } else {
        None
    };
    let charge_path = if args.charge_mode == ChargeMode::ShadowInfer {
        match &args.charge_checkpoint {
            Some(checkpoint) if Path::new(checkpoint).exists() => {
                let path = Path::new(checkpoint);
                println!("loaded charge checkpoint: {}", path.display());
                Some(path)
            }
            Some(checkpoint) => {
                println!(
                    "charge shadow inference disabled: checkpoint not found at {}",
                    checkpoint
                );
                None
            }
            None => {
                println!("charge shadow inference disabled: no --charge-checkpoint provided");
                None
            }
        }
    } else {
        None
    };
    let action_value_path = if args.action_value_mode == ActionValueMode::ShadowInfer {
        match &args.action_value_checkpoint {
            Some(checkpoint) if Path::new(checkpoint).exists() => {
                let path = Path::new(checkpoint);
                println!("loaded action-value checkpoint: {}", path.display());
                Some(path)
            }
            Some(checkpoint) => {
                println!(
                    "action-value shadow inference disabled: checkpoint not found at {}",
                    checkpoint
                );
                None
            }
            None => {
                println!(
                    "action-value shadow inference disabled: no --action-value-checkpoint provided"
                );
                None
            }
        }
    } else {
        None
    };
    let future_path = if args.future_mode != FutureMode::Hardcoded {
        match &args.future_checkpoint {
            Some(checkpoint) if Path::new(checkpoint).exists() => {
                let path = Path::new(checkpoint);
                println!("loaded future checkpoint: {}", path.display());
                Some(path)
            }
            Some(checkpoint) => {
                println!(
                    "future inference disabled: checkpoint not found at {}",
                    checkpoint
                );
                None
            }
            None => {
                println!("future inference disabled: no --future-checkpoint provided");
                None
            }
        }
    } else {
        None
    };
    let eye_next_path = if args.eye_next_mode == EyeNextMode::ShadowInfer {
        match &args.eye_next_checkpoint {
            Some(checkpoint) if Path::new(checkpoint).exists() => {
                let path = Path::new(checkpoint);
                println!("loaded eye-next checkpoint: {}", path.display());
                Some(path)
            }
            Some(checkpoint) => {
                println!(
                    "eye-next shadow inference disabled: checkpoint not found at {}",
                    checkpoint
                );
                None
            }
            None => {
                println!("eye-next shadow inference disabled: no --eye-next-checkpoint provided");
                None
            }
        }
    } else {
        None
    };
    let ear_next_path = if args.ear_next_mode == EarNextMode::ShadowInfer {
        match &args.ear_next_checkpoint {
            Some(checkpoint) if Path::new(checkpoint).exists() => {
                let path = Path::new(checkpoint);
                println!("loaded ear-next checkpoint: {}", path.display());
                Some(path)
            }
            Some(checkpoint) => {
                println!(
                    "ear-next shadow inference disabled: checkpoint not found at {}",
                    checkpoint
                );
                None
            }
            None => {
                println!("ear-next shadow inference disabled: no --ear-next-checkpoint provided");
                None
            }
        }
    } else {
        None
    };
    let experience_path = if args.experience_mode != ExperienceMode::Off {
        match &args.experience_checkpoint {
            Some(checkpoint) if Path::new(checkpoint).exists() => {
                let path = Path::new(checkpoint);
                println!("loaded experience checkpoint: {}", path.display());
                Some(path)
            }
            Some(checkpoint) => {
                println!(
                    "experience inference disabled: checkpoint not found at {}",
                    checkpoint
                );
                None
            }
            None => {
                println!("experience inference disabled: no --experience-checkpoint provided");
                None
            }
        }
    } else {
        None
    };
    if danger_path.is_none()
        && charge_path.is_none()
        && action_value_path.is_none()
        && future_path.is_none()
        && eye_next_path.is_none()
        && ear_next_path.is_none()
        && experience_path.is_none()
    {
        return Ok(None);
    }

    let mut models = RuntimeModelStack::with_shadow_checkpoints(
        danger_path,
        charge_path,
        action_value_path,
        future_path,
        eye_next_path,
        ear_next_path,
        experience_path,
    )?;
    if future_path.is_some() && args.future_mode == FutureMode::ModelInfer {
        models.behaviors.future.regime = BehaviorRegime::ModelInfer;
    }
    if experience_path.is_some() && args.experience_mode == ExperienceMode::ModelInfer {
        models.behaviors.experience.regime = BehaviorRegime::ModelInfer;
    }
    Ok(Some(models))
}

async fn inspect_ledger() -> Result<()> {
    let ledger = JsonlLedger::new("data/ledger");
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

fn model_status() -> Result<()> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use netherwick_actions::ActionPrimitive;
    use netherwick_body::BodySense;
    use netherwick_core::Reward;
    use netherwick_experience::ExperienceLatent;
    use netherwick_now::{Now, SurpriseSense};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

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
                .join("episode-00000")
                .to_string_lossy()
                .to_string()
        );
        assert!(capture_root
            .join("episode-00000")
            .join("manifest.json")
            .exists());
        assert!(capture_root
            .join("episode-00001")
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
