use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "cargo xtask")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Check,
    Sim,
    SimDashboard,
    Train { target: String },
    Replay { target: String },
    RobotSlow,
}

fn main() {
    let cli = Cli::parse();
    println!("xtask {:?}", cli.command);
}
