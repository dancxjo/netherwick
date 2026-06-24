use anyhow::Result;
use clap::{Parser, Subcommand};
use netherwick_autonomic::SimpleSafety;
use netherwick_body::RobotBody;
use netherwick_conductor::SimpleConductor;
use netherwick_experience::ExperienceLatent;
use netherwick_ledger::{ExperienceFrame, JsonlLedger, LedgerReader};
use netherwick_llm::NoopLlmAgent;
use netherwick_memory::InMemoryExperienceStore;
use netherwick_now::Now;
use netherwick_runtime::MinimalRuntime;
use netherwick_sim::SimBody;

#[derive(Parser)]
#[command(name = "netherwick")]
#[command(about = "Netherwick CLI entrypoint")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Sim,
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
        Command::Sim => run_sim_tick().await,
        Command::InspectLedger => inspect_ledger().await,
        other => {
            println!("selected command: {:?}", other);
            Ok(())
        }
    }
}

async fn run_sim_tick() -> Result<()> {
    let ledger = JsonlLedger::new("data/ledger");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime {
        ledger,
        memory_store: memory,
        memory_recall: recall,
        conductor: SimpleConductor::default(),
        safety: SimpleSafety::default(),
        llm: NoopLlmAgent,
    };

    let mut body = SimBody::new(7);
    let body_sense = body.read_body().await?;
    let mut now = Now::blank(100, body_sense);
    now.ear.transcript = Some("hello from the simulator".to_string());
    let latent = ExperienceLatent {
        t_ms: now.t_ms,
        z: vec![0.1, 0.2, 0.3],
        reconstruction_error: 0.0,
        prediction_error: 0.0,
        confidence: 1.0,
    };
    let tick = runtime.tick(now, latent, Vec::new()).await?;
    println!("experience: {}", tick.experience.text);
    println!("action: {:?}", tick.chosen_action);
    println!("recall: {}", tick.recall.first_person_summary);
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
