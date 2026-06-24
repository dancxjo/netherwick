use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use netherwick_autonomic::SimpleSafety;
use netherwick_behaviors::BehaviorRegime;
use netherwick_body::{BodySense, RobotBody};
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
use netherwick_sensors::SenseProducer;
use netherwick_server::LiveViewState;
use netherwick_sim::{ArenaConfig, SimMotorComplex, SimObject, SimObjectKind, VirtualWorld};
use netherwick_training::{
    evaluate_behavior, promote_behavior_config, train_behavior, EvaluateBehaviorRequest,
    TrainBehaviorRequest, TrainableBehavior,
};
use netherwick_worldlab::{CaptureReader, CaptureReplayRunner, CaptureSource, CaptureWriter};

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

fn default_sim_world(seed: u64) -> (VirtualWorld, SimMotorComplex) {
    let (mut world, motors) = VirtualWorld::new_with_motor(
        seed,
        ArenaConfig {
            width_m: 8.0,
            height_m: 8.0,
        },
    );
    let mut body = BodySense::default();
    body.last_update_ms = seed;
    body.odometry.x_m = 1.0;
    body.odometry.y_m = 1.0;
    world.set_body(body);
    world.add_object(SimObject::charger("charger", "charger", 1.9, 1.0, 0.25));
    world.add_object(SimObject::obstacle("crate", "crate", 3.2, 2.3, 0.35));
    world.add_object(SimObject {
        id: "person".to_string(),
        label: "person".to_string(),
        kind: SimObjectKind::Person {
            identity: Some("sim-person".to_string()),
        },
        x_m: 2.4,
        y_m: 1.6,
        radius_m: 0.22,
        color_rgb: [220, 180, 140],
        emits_sound: false,
        charge_rate: 0.0,
    });
    world.add_object(SimObject {
        id: "speaker".to_string(),
        label: "speaker".to_string(),
        kind: SimObjectKind::SoundSource {
            label: "speaker".to_string(),
        },
        x_m: 1.5,
        y_m: 2.2,
        radius_m: 0.12,
        color_rgb: [80, 80, 220],
        emits_sound: true,
        charge_rate: 0.0,
    });
    (world, motors)
}

async fn run_robot(args: RobotArgs) -> Result<()> {
    if args.mode != RobotModeArg::ReadOnly {
        anyhow::bail!("only --mode read-only is implemented for real robot bring-up");
    }
    validate_optional_sensor("camera", args.camera.as_deref(), args.require_camera)?;
    validate_optional_sensor("mic", args.mic.as_deref(), args.require_mic)?;
    validate_optional_sensor("imu", args.imu.as_deref(), args.require_imu)?;
    validate_optional_sensor("gps", args.gps.as_deref(), args.require_gps)?;

    let body: Box<dyn RobotBody + Send> = if args.create_port == "mock" {
        Box::new(MockCreate1Body::new())
    } else {
        Box::new(Create1Body::connect(&args.create_port, args.create_baud).await?)
    };
    let sensors: Vec<Box<dyn SenseProducer + Send>> = Vec::new();
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

fn validate_optional_sensor(kind: &str, device: Option<&str>, required: bool) -> Result<()> {
    if device.is_none() {
        return Ok(());
    }
    let message = format!("{kind} provider is not wired into this build yet");
    if required {
        anyhow::bail!(message);
    }
    println!("{message}; continuing without it");
    Ok(())
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
        && args.experience_mode != ExperienceMode::ShadowInfer
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
    let experience_path = if args.experience_mode == ExperienceMode::ShadowInfer {
        match &args.experience_checkpoint {
            Some(checkpoint) if Path::new(checkpoint).exists() => {
                let path = Path::new(checkpoint);
                println!("loaded experience checkpoint: {}", path.display());
                Some(path)
            }
            Some(checkpoint) => {
                println!(
                    "experience shadow inference disabled: checkpoint not found at {}",
                    checkpoint
                );
                None
            }
            None => {
                println!(
                    "experience shadow inference disabled: no --experience-checkpoint provided"
                );
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

    let models = RuntimeModelStack::with_shadow_checkpoints(
        danger_path,
        charge_path,
        action_value_path,
        future_path,
        eye_next_path,
        ear_next_path,
        experience_path,
    )?;
    let models = if future_path.is_some() && args.future_mode == FutureMode::ModelInfer {
        if danger_path.is_none()
            && charge_path.is_none()
            && action_value_path.is_none()
            && eye_next_path.is_none()
            && ear_next_path.is_none()
            && experience_path.is_none()
        {
            RuntimeModelStack::with_future_checkpoint(
                future_path.unwrap(),
                BehaviorRegime::ModelInfer,
            )?
        } else {
            let mut models = models;
            models.behaviors.future.regime = BehaviorRegime::ModelInfer;
            models
        }
    } else {
        models
    };
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
                checkpoint_path: checkpoint.into(),
                max_samples: args.max_samples,
            })
            .await?;
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
    println!(
        "DangerPredictor: shadow-train ready; metrics: data/ledger/danger-shadow-metrics.jsonl; checkpoint: data/models/danger_v0"
    );
    println!(
        "ChargePredictor: shadow-train ready; metrics: data/ledger/charge-shadow-metrics.jsonl; checkpoint: data/models/charge_v0"
    );
    println!(
        "ActionValueNet: shadow-train ready; metrics: data/ledger/action-value-shadow-metrics.jsonl; checkpoint: data/models/action_value_v0"
    );
    println!(
        "FuturePredictor: hardcoded stasis; train ready with transitions; metrics: data/models/future_v0/future-shadow-metrics.jsonl; checkpoint: data/models/future_v0; behavior-regime: hardcoded/shadow-infer/model-infer"
    );
    println!(
        "EyeNextPredictor: shadow-train ready; metrics: data/ledger/eye-next-shadow-metrics.jsonl; checkpoint: data/models/eye_next_v0"
    );
    println!(
        "EarNextPredictor: shadow-train ready; metrics: data/ledger/ear-next-shadow-metrics.jsonl; checkpoint: data/models/ear_next_v0"
    );
    println!(
        "ExperienceAutoencoder: shadow-train ready; metrics: data/ledger/experience-autoencoder-shadow-metrics.jsonl; checkpoint: data/models/experience_v0"
    );
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
