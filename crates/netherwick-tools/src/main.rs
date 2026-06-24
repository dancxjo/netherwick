use anyhow::Result;
use clap::{Parser, Subcommand};
use netherwick_autonomic::SimpleSafety;
use netherwick_body::BodySense;
use netherwick_conductor::SimpleConductor;
use netherwick_ledger::{ExperienceFrame, JsonlLedger, LedgerReader};
use netherwick_llm::NoopLlmAgent;
use netherwick_memory::InMemoryExperienceStore;
use netherwick_runtime::{MinimalRuntime, SimRunner};
use netherwick_sim::{ArenaConfig, SimObject, SimObjectKind, VirtualWorld};

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
    Train,
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
}

async fn run_sim(args: SimArgs) -> Result<()> {
    let ledger = JsonlLedger::new(&args.ledger);
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let runtime = MinimalRuntime::with_default_events(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        NoopLlmAgent,
    );

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
        "sim complete: {} ticks, seed {}, ledger {}",
        runner.tick_count, args.seed, args.ledger
    );
    Ok(())
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
