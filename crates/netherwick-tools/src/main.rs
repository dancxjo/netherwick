use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use netherwick_autonomic::SimpleSafety;
use netherwick_behaviors::BehaviorRegime;
use netherwick_body::BodySense;
use netherwick_conductor::SimpleConductor;
use netherwick_experience::{
    action_value_input_from_transition_like, action_value_target_from_reward_surprise,
    charge_input_from_transition_like, charge_target_from_transition_like,
    danger_input_from_transition_like, danger_target_from_transition_like,
    ear_next_input_from_transition_like, ear_next_target_from_now,
    experience_decode_target_from_now, experience_encode_input_from_now,
    eye_next_input_from_transition_like, eye_next_target_from_now, FuturePredictor,
    StasisFuturePredictor,
};
use netherwick_ledger::{
    future_input_from_transition, future_target_from_transition, ExperienceFrame,
    ExperienceTransition, JsonlLedger, LedgerReader, LedgerWriter,
};
use netherwick_llm::NoopLlmAgent;
use netherwick_memory::InMemoryExperienceStore;
use netherwick_models::{
    ActionValueNetTrainer, ChargeNetTrainer, DangerNetTrainer, EarNextNetTrainer,
    ExperienceAutoencoderTrainer, EyeNextNetTrainer, FutureNetTrainer, MODEL_REGISTRY,
};
use netherwick_runtime::{MinimalRuntime, RuntimeModelStack, SimRunner};
use netherwick_server::LiveViewState;
use netherwick_sim::{ArenaConfig, SimMotorComplex, SimObject, SimObjectKind, VirtualWorld};
use netherwick_worldlab::{CaptureReader, CaptureReplayRunner, CaptureSource, CaptureWriter};
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;

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
    Robot,
    Replay,
    CaptureSim(CaptureSimArgs),
    ReplayCapture(ReplayCaptureArgs),
    Train(TrainCommand),
    InspectLedger,
    ModelStatus,
    Dashboard,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Sim(args) => run_sim(args).await,
        Command::CaptureSim(args) => capture_sim(args).await,
        Command::ReplayCapture(args) => replay_capture(args).await,
        Command::InspectLedger => inspect_ledger().await,
        Command::Train(command) => run_train(command).await,
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
    Danger(TrainDangerArgs),
    Charge(TrainChargeArgs),
    ActionValue(TrainActionValueArgs),
    Future(TrainFutureArgs),
    EyeNext(TrainEyeNextArgs),
    EarNext(TrainEarNextArgs),
    Experience(TrainExperienceArgs),
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
        TrainModel::Danger(args) => train_danger(args).await,
        TrainModel::Charge(args) => train_charge(args).await,
        TrainModel::ActionValue(args) => train_action_value(args).await,
        TrainModel::Future(args) => train_future(args).await,
        TrainModel::EyeNext(args) => train_eye_next(args).await,
        TrainModel::EarNext(args) => train_ear_next(args).await,
        TrainModel::Experience(args) => train_experience(args).await,
    }
}

async fn train_danger(args: TrainDangerArgs) -> Result<()> {
    let ledger = JsonlLedger::new(&args.ledger);
    let transitions = ledger.transitions().await?;
    if transitions.is_empty() {
        println!(
            "danger training skipped: no transitions found in {}",
            args.ledger
        );
        return Ok(());
    }

    let mut samples = Vec::new();
    for transition in &transitions {
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
        samples.push((
            transition.created_at_ms,
            transition.before.clone(),
            input,
            target,
        ));
    }

    let input_dim = samples
        .first()
        .map(|(_, _, input, _)| input.flat_features().len())
        .unwrap_or(0);
    let mut trainer = DangerNetTrainer::new(input_dim);
    let metrics_path = std::path::Path::new(&args.ledger).join("danger-shadow-metrics.jsonl");
    let mut metrics_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&metrics_path)
        .await?;

    let mut last_loss = 0.0;
    let mut seen = 0_u64;
    for _ in 0..args.epochs {
        for (observed_at_ms, before, input, target) in &samples {
            if input.flat_features().len() != trainer.input_dim() {
                continue;
            }
            let metric = trainer.shadow_compare(*observed_at_ms, before, input, target)?;
            let line = serde_json::to_string(&metric)?;
            metrics_file.write_all(line.as_bytes()).await?;
            metrics_file.write_all(b"\n").await?;

            let stats = trainer.train_step(input, target)?;
            last_loss = stats.loss;
            seen = stats.samples_seen;
        }
    }

    println!(
        "danger training complete: {} transitions, {} epochs, {} samples, last_loss {:.6}, metrics {}",
        samples.len(),
        args.epochs,
        seen,
        last_loss,
        metrics_path.display()
    );
    trainer.save_checkpoint(&args.checkpoint)?;
    println!("saved danger checkpoint: {}", args.checkpoint);
    println!("samples_seen: {}", trainer.samples_seen());
    println!("last_loss: {:.6}", last_loss);
    println!("best_loss: {:?}", trainer.best_loss());
    Ok(())
}

async fn train_charge(args: TrainChargeArgs) -> Result<()> {
    let ledger = JsonlLedger::new(&args.ledger);
    let transitions = ledger.transitions().await?;
    if transitions.is_empty() {
        println!(
            "charge training skipped: no transitions found in {}",
            args.ledger
        );
        return Ok(());
    }

    let mut samples = Vec::new();
    for transition in &transitions {
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
        samples.push((
            transition.created_at_ms,
            transition.before.clone(),
            input,
            target,
        ));
    }

    let input_dim = samples
        .first()
        .map(|(_, _, input, _)| input.flat_features().len())
        .unwrap_or(0);
    let mut trainer = ChargeNetTrainer::new(input_dim);
    let metrics_path = std::path::Path::new(&args.ledger).join("charge-shadow-metrics.jsonl");
    let mut metrics_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&metrics_path)
        .await?;

    let mut last_loss = 0.0;
    let mut seen = 0_u64;
    for _ in 0..args.epochs {
        for (observed_at_ms, before, input, target) in &samples {
            if input.flat_features().len() != trainer.input_dim() {
                continue;
            }
            let metric = trainer.shadow_compare(*observed_at_ms, before, input, target)?;
            let line = serde_json::to_string(&metric)?;
            metrics_file.write_all(line.as_bytes()).await?;
            metrics_file.write_all(b"\n").await?;

            let stats = trainer.train_step(input, target)?;
            last_loss = stats.loss;
            seen = stats.samples_seen;
        }
    }

    println!(
        "charge training complete: {} transitions, {} epochs, {} samples, last_loss {:.6}, metrics {}",
        samples.len(),
        args.epochs,
        seen,
        last_loss,
        metrics_path.display()
    );
    trainer.save_checkpoint(&args.checkpoint)?;
    println!("saved charge checkpoint: {}", args.checkpoint);
    println!("samples_seen: {}", trainer.samples_seen());
    println!("last_loss: {:.6}", last_loss);
    println!("best_loss: {:?}", trainer.best_loss());
    Ok(())
}

async fn train_action_value(args: TrainActionValueArgs) -> Result<()> {
    let ledger = JsonlLedger::new(&args.ledger);
    let transitions = ledger.transitions().await?;
    if transitions.is_empty() {
        println!(
            "action-value training skipped: no transitions found in {}",
            args.ledger
        );
        return Ok(());
    }

    let mut samples = Vec::new();
    for transition in &transitions {
        let input = action_value_input_from_transition_like(
            &transition.before_z,
            transition.action.as_ref(),
            &transition.before,
        );
        let target =
            action_value_target_from_reward_surprise(&transition.reward, &transition.surprise);
        samples.push((
            transition.created_at_ms,
            transition.before.clone(),
            input,
            target,
        ));
    }

    let input_dim = samples
        .first()
        .map(|(_, _, input, _)| input.flat_features().len())
        .unwrap_or(0);
    let mut trainer = ActionValueNetTrainer::new(input_dim);
    let metrics_path = std::path::Path::new(&args.ledger).join("action-value-shadow-metrics.jsonl");
    let mut metrics_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&metrics_path)
        .await?;

    let mut last_loss = 0.0;
    let mut seen = 0_u64;
    for _ in 0..args.epochs {
        for (observed_at_ms, before, input, target) in &samples {
            if input.flat_features().len() != trainer.input_dim() {
                continue;
            }
            let metric = trainer.shadow_compare(*observed_at_ms, before, input, target)?;
            let line = serde_json::to_string(&metric)?;
            metrics_file.write_all(line.as_bytes()).await?;
            metrics_file.write_all(b"\n").await?;

            let stats = trainer.train_step(input, target)?;
            last_loss = stats.loss;
            seen = stats.samples_seen;
        }
    }

    println!(
        "action-value training complete: {} transitions, {} epochs, {} samples, last_loss {:.6}, metrics {}",
        samples.len(),
        args.epochs,
        seen,
        last_loss,
        metrics_path.display()
    );
    trainer.save_checkpoint(&args.checkpoint)?;
    println!("saved action-value checkpoint: {}", args.checkpoint);
    println!("samples_seen: {}", trainer.samples_seen());
    println!("last_loss: {:.6}", last_loss);
    println!("best_loss: {:?}", trainer.best_loss());
    Ok(())
}

async fn train_future(args: TrainFutureArgs) -> Result<()> {
    let ledger = JsonlLedger::new(&args.ledger);
    let transitions = ledger.transitions().await?;
    if transitions.is_empty() {
        println!(
            "future training skipped: no transitions found in {}",
            args.ledger
        );
        return Ok(());
    }

    let mut samples = Vec::new();
    for transition in &transitions {
        let Some(input) = future_input_from_transition(transition, 1_000) else {
            continue;
        };
        let target = future_target_from_transition(transition);
        if target.is_empty() {
            continue;
        }
        samples.push((transition.created_at_ms, input, target));
    }
    if samples.is_empty() {
        println!(
            "future training skipped: no usable transitions with actions and after_z in {}",
            args.ledger
        );
        return Ok(());
    }

    let input_dim = samples
        .first()
        .map(|(_, input, _)| input.flat_features().len())
        .unwrap_or(0);
    let latent_dim = samples
        .first()
        .map(|(_, _, target)| target.len())
        .unwrap_or(0);
    let mut trainer = FutureNetTrainer::new(input_dim, latent_dim);
    let checkpoint_dir = std::path::Path::new(&args.checkpoint);
    tokio::fs::create_dir_all(checkpoint_dir).await?;
    let metrics_path = checkpoint_dir.join("future-shadow-metrics.jsonl");
    let mut metrics_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&metrics_path)
        .await?;

    let mut last_loss = 0.0;
    let mut seen = 0_u64;
    let mut stasis = StasisFuturePredictor;
    for _ in 0..args.epochs {
        for (_, input, target) in &samples {
            if input.flat_features().len() != trainer.input_dim() || target.len() != latent_dim {
                continue;
            }
            let hardcoded = stasis.predict(&input.latent, &input.action, input.offset_ms)?;
            let metric = trainer.shadow_compare(input, &hardcoded, target)?;
            let line = serde_json::to_string(&metric)?;
            metrics_file.write_all(line.as_bytes()).await?;
            metrics_file.write_all(b"\n").await?;

            let stats = trainer.train_step(input, target)?;
            last_loss = stats.loss;
            seen = stats.samples_seen;
        }
    }

    println!(
        "future training complete: {} transitions, {} epochs, {} samples, last_loss {:.6}, metrics {}",
        samples.len(),
        args.epochs,
        seen,
        last_loss,
        metrics_path.display()
    );
    trainer.save_checkpoint(&args.checkpoint)?;
    println!("saved future checkpoint: {}", args.checkpoint);
    println!("samples_seen: {}", trainer.samples_seen());
    println!("last_loss: {:.6}", last_loss);
    println!("best_loss: {:?}", trainer.best_loss());
    Ok(())
}

async fn train_eye_next(args: TrainEyeNextArgs) -> Result<()> {
    let ledger = JsonlLedger::new(&args.ledger);
    let transitions = ledger.transitions().await?;
    if transitions.is_empty() {
        println!(
            "eye-next training skipped: no transitions found in {}",
            args.ledger
        );
        return Ok(());
    }

    let mut samples = Vec::new();
    for transition in &transitions {
        let Some(target) = eye_next_target_from_now(&transition.after) else {
            continue;
        };
        let input = eye_next_input_from_transition_like(
            &transition.before_z,
            transition.action.as_ref(),
            &transition.before,
            100,
        );
        samples.push((
            transition.created_at_ms,
            transition.before.clone(),
            input,
            target,
        ));
    }
    if samples.is_empty() {
        println!(
            "eye-next training skipped: no transitions with eye frames found in {}",
            args.ledger
        );
        return Ok(());
    }

    let (input_dim, width, height) = samples
        .first()
        .map(|(_, _, input, target)| (input.flat_features().len(), target.width, target.height))
        .unwrap_or((0, 64, 48));
    let mut trainer = EyeNextNetTrainer::new(input_dim, width, height);
    let metrics_path = std::path::Path::new(&args.ledger).join("eye-next-shadow-metrics.jsonl");
    let mut metrics_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&metrics_path)
        .await?;

    let mut last_loss = 0.0;
    let mut seen = 0_u64;
    for _ in 0..args.epochs {
        for (observed_at_ms, before, input, target) in &samples {
            if input.flat_features().len() != trainer.input_dim() {
                continue;
            }
            let metric = trainer.shadow_compare(*observed_at_ms, before, input, target)?;
            let line = serde_json::to_string(&metric)?;
            metrics_file.write_all(line.as_bytes()).await?;
            metrics_file.write_all(b"\n").await?;

            let stats = trainer.train_step(input, target)?;
            last_loss = stats.loss;
            seen = stats.samples_seen;
        }
    }

    println!(
        "eye-next training complete: {} transitions, {} epochs, {} samples, last_loss {:.6}, metrics {}",
        samples.len(),
        args.epochs,
        seen,
        last_loss,
        metrics_path.display()
    );
    trainer.save_checkpoint(&args.checkpoint)?;
    println!("saved eye-next checkpoint: {}", args.checkpoint);
    println!("samples_seen: {}", trainer.samples_seen());
    println!("last_loss: {:.6}", last_loss);
    println!("best_loss: {:?}", trainer.best_loss());
    Ok(())
}

async fn train_ear_next(args: TrainEarNextArgs) -> Result<()> {
    let ledger = JsonlLedger::new(&args.ledger);
    let transitions = ledger.transitions().await?;
    if transitions.is_empty() {
        println!(
            "ear-next training skipped: no transitions found in {}",
            args.ledger
        );
        return Ok(());
    }

    let mut samples = Vec::new();
    for transition in &transitions {
        let Some(target) = ear_next_target_from_now(&transition.after) else {
            continue;
        };
        let input = ear_next_input_from_transition_like(
            &transition.before_z,
            transition.action.as_ref(),
            &transition.before,
            100,
        );
        samples.push((
            transition.created_at_ms,
            transition.before.clone(),
            input,
            target,
        ));
    }
    if samples.is_empty() {
        println!(
            "ear-next training skipped: no transitions with ear features found in {}",
            args.ledger
        );
        return Ok(());
    }

    let (input_dim, output_dim, sample_rate_hz, channels) = samples
        .first()
        .map(|(_, _, input, target)| {
            (
                input.flat_features().len(),
                target.features.len(),
                target.sample_rate_hz,
                target.channels,
            )
        })
        .unwrap_or((0, 0, 0, 0));
    let mut trainer =
        EarNextNetTrainer::with_audio_shape(input_dim, output_dim, sample_rate_hz, channels);
    let metrics_path = std::path::Path::new(&args.ledger).join("ear-next-shadow-metrics.jsonl");
    let mut metrics_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&metrics_path)
        .await?;

    let mut last_loss = 0.0;
    let mut seen = 0_u64;
    for _ in 0..args.epochs {
        for (observed_at_ms, before, input, target) in &samples {
            if input.flat_features().len() != trainer.input_dim()
                || target.features.len() != trainer.output_dim()
            {
                continue;
            }
            let metric = trainer.shadow_compare(*observed_at_ms, before, input, target)?;
            let line = serde_json::to_string(&metric)?;
            metrics_file.write_all(line.as_bytes()).await?;
            metrics_file.write_all(b"\n").await?;

            let stats = trainer.train_step(input, target)?;
            last_loss = stats.loss;
            seen = stats.samples_seen;
        }
    }

    println!(
        "ear-next training complete: {} transitions, {} epochs, {} samples, last_loss {:.6}, metrics {}",
        samples.len(),
        args.epochs,
        seen,
        last_loss,
        metrics_path.display()
    );
    trainer.save_checkpoint(&args.checkpoint)?;
    println!("saved ear-next checkpoint: {}", args.checkpoint);
    println!("samples_seen: {}", trainer.samples_seen());
    println!("last_loss: {:.6}", last_loss);
    println!("best_loss: {:?}", trainer.best_loss());
    Ok(())
}

async fn train_experience(args: TrainExperienceArgs) -> Result<()> {
    let ledger = JsonlLedger::new(&args.ledger);
    let transitions = ledger.transitions().await?;
    if transitions.is_empty() {
        println!(
            "experience training skipped: no transitions found in {}",
            args.ledger
        );
        return Ok(());
    }

    let mut samples = Vec::new();
    for transition in &transitions {
        for (observed_at_ms, now) in [
            (transition.created_at_ms, &transition.before),
            (transition.created_at_ms, &transition.after),
        ] {
            let input = experience_encode_input_from_now(now);
            let target = experience_decode_target_from_now(now);
            if input.flat_features().is_empty() || target.flat_features().is_empty() {
                continue;
            }
            samples.push((observed_at_ms, input, target));
        }
    }
    if samples.is_empty() {
        println!(
            "experience training skipped: no vectorized frames found in {}",
            args.ledger
        );
        return Ok(());
    }

    let (input_dim, decode_lengths) = samples
        .first()
        .map(|(_, input, target)| (input.flat_features().len(), target.feature_lengths()))
        .unwrap_or_default();
    let z_dim = input_dim.clamp(8, 32);
    let mut trainer = ExperienceAutoencoderTrainer::new(input_dim, z_dim, decode_lengths);
    let metrics_path =
        std::path::Path::new(&args.ledger).join("experience-autoencoder-shadow-metrics.jsonl");
    let mut metrics_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&metrics_path)
        .await?;

    let mut last_loss = 0.0;
    let mut seen = 0_u64;
    for _ in 0..args.epochs {
        for (observed_at_ms, input, target) in &samples {
            if input.flat_features().len() != trainer.input_dim()
                || target.feature_lengths() != trainer.decode_lengths()
            {
                continue;
            }
            let metric = trainer.shadow_compare(*observed_at_ms, input, target)?;
            let line = serde_json::to_string(&metric)?;
            metrics_file.write_all(line.as_bytes()).await?;
            metrics_file.write_all(b"\n").await?;

            let stats = trainer.train_step(input, target)?;
            last_loss = stats.loss;
            seen = stats.samples_seen;
        }
    }

    println!(
        "experience training complete: {} examples, {} epochs, {} samples, last_loss {:.6}, metrics {}",
        samples.len(),
        args.epochs,
        seen,
        last_loss,
        metrics_path.display()
    );
    trainer.save_checkpoint(&args.checkpoint)?;
    println!("saved experience checkpoint: {}", args.checkpoint);
    println!("samples_seen: {}", trainer.samples_seen());
    println!("z_dim: {}", trainer.z_dim());
    println!("last_loss: {:.6}", last_loss);
    println!("best_loss: {:?}", trainer.best_loss());
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
