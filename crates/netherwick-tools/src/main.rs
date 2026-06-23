use clap::{Parser, Subcommand};

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

fn main() {
    let cli = Cli::parse();
    println!("selected command: {:?}", cli.command);
}
