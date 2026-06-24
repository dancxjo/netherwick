use std::path::Path;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use netherwick_autonomic::SimpleSafety;
use netherwick_body::BodySense;
use netherwick_conductor::SimpleConductor;
use netherwick_experience::{
    action_value_input_from_transition_like, action_value_target_from_reward_surprise,
    charge_input_from_transition_like, charge_target_from_transition_like,
    danger_input_from_transition_like, danger_target_from_transition_like,
};
use netherwick_ledger::{ExperienceFrame, JsonlLedger, LedgerReader};
use netherwick_llm::NoopLlmAgent;
use netherwick_memory::InMemoryExperienceStore;
use netherwick_models::{
    ActionValueNetTrainer, ChargeNetTrainer, DangerNetTrainer, MODEL_REGISTRY,
};
use netherwick_runtime::{MinimalRuntime, RuntimeModelStack, SimRunner};
use netherwick_sim::{ArenaConfig, SimObject, SimObjectKind, VirtualWorld};
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

    let (mut world, motors) = VirtualWorld::new_with_motor(
        args.seed,
        ArenaConfig {
            width_m: 8.0,
            height_m: 8.0,
        },
    );
    let mut body = BodySense::default();
    body.last_update_ms = args.seed;
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

    let mut runner = SimRunner::new(runtime, world, motors);
    runner.run_steps(args.steps).await?;
    println!(
        "sim complete: {} ticks, seed {}, ledger {}, danger_mode {:?}, charge_mode {:?}, action_value_mode {:?}",
        runner.tick_count,
        args.seed,
        args.ledger,
        args.danger_mode,
        args.charge_mode,
        args.action_value_mode
    );
    Ok(())
}

fn load_runtime_models(args: &SimArgs) -> Result<Option<RuntimeModelStack>> {
    if args.danger_mode != DangerMode::ShadowInfer
        && args.charge_mode != ChargeMode::ShadowInfer
        && args.action_value_mode != ActionValueMode::ShadowInfer
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
    if danger_path.is_none() && charge_path.is_none() && action_value_path.is_none() {
        return Ok(None);
    }

    let models =
        RuntimeModelStack::with_shadow_checkpoints(danger_path, charge_path, action_value_path)?;
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
