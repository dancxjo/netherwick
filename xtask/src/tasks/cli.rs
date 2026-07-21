#[derive(Parser)]
#[command(name = "cargo xtask", about = "Netherwick workspace automation")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Setup,
    SetupSystem,
    SetupDocker,
    SetupUser,
    SetupPicoBootsel,
    SetupRust,
    SetupOrt,
    SetupKinect,
    SetupKinectFromSource,
    SetupTts,
    SetupWhisper,
    Fetch,
    ForebrainBuild,
    ForebrainValidate {
        config: PathBuf,
    },
    ForebrainProvision {
        inventory: PathBuf,
    },
    ForebrainProvisionCheck {
        inventory: PathBuf,
    },
    ForebrainRemove {
        inventory: PathBuf,
    },
    Fmt,
    Check,
    Test,
    Clippy,
    Clean,
    BrainstemBuild,
    BrainstemUf2,
    BrainstemFetchCyw43,
    BrainstemPicoWBuild,
    BrainstemPicoWUf2,
    Flash,
    Skull,
    ComposeConfig,
    ComposeBuild,
    Servers,
    LiveServer,
    ServerLogs {
        service: String,
    },
    StopServers,
    Sim,
    Say {
        text: String,
    },
    Transcribe {
        wav: PathBuf,
    },
    Robot {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    Possess {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Guide a human through the physical safety QA checklist and record evidence.
    PhysicalQa {
        /// Print the complete QA plan without touching hardware or prompting.
        #[arg(long)]
        plan: bool,
        /// Write the session record to this path instead of the timestamped default.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    Go {
        target: Option<String>,
    },
    Train {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    Evolve {
        clear: Option<String>,
    },
    EvolveQuality {
        clear: Option<String>,
    },
    EvolveInfinite {
        clear: Option<String>,
    },
    DevCert,
    VirtualUrl,
    Run {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    RehearseModels,
    EvalScenarioSmoke,
    InspectLedger,
    HardwareEnv,
    RealCockpit,
    /// Open the lightweight transport-neutral Brainstem cockpit CLI.
    Cockpit {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    CodexSync,
    PicoBootselMount {
        umount: bool,
        kernel_name: String,
    },
    NeatGenerationLimit {
        state_path: PathBuf,
        explicit_limit: Option<u64>,
        #[arg(default_value_t = 120)]
        default_limit: u64,
        #[arg(default_value_t = 120)]
        increment: u64,
    },
}

fn main() {
    if let Err(error) = run(Cli::parse().command) {
        eprintln!("xtask: {error}");
        std::process::exit(1);
    }
}

fn run(command: Command) -> Result<()> {
    match command {
        Command::Setup => {
            for command in [
                Command::SetupSystem,
                Command::SetupDocker,
                Command::SetupUser,
                Command::SetupRust,
                Command::SetupPicoBootsel,
                Command::SetupKinect,
                Command::SetupOrt,
                Command::SetupTts,
                Command::SetupWhisper,
            ] {
                run(command)?;
            }
            println!("pete Linux setup complete\nnext: cargo check && just sim");
            Ok(())
        }
        Command::SetupSystem => {
            run_program(ProcessCommand::new("sudo").args(["apt-get", "update"]))?;
            run_program(ProcessCommand::new("sudo").args([
                "apt-get", "install", "-y", "build-essential", "pkg-config", "cmake", "ninja-build",
                "git", "curl", "just", "clang", "libclang-dev", "ffmpeg", "i2c-tools", "v4l-utils",
                "libasound2-dev", "libgomp1", "libssl-dev", "libudev-dev", "libusb-1.0-0-dev",
                "libv4l-dev", "udisks2",
            ]))
        }
        Command::SetupDocker => {
            if !program_exists("docker") {
                run_program(ProcessCommand::new("sh").args(["-c", "curl -fsSL https://get.docker.com | sudo sh"]))?;
            }
            Ok(())
        }
        Command::SetupUser => {
            let user = env::var("USER").unwrap_or_else(|_| "root".to_owned());
            for group in ["docker", "dialout", "video", "audio", "i2c", "plugdev"] {
                if ProcessCommand::new("getent").args(["group", group]).status()?.success() {
                    run_program(ProcessCommand::new("sudo").args(["usermod", "-aG", group, &user]))?;
                }
            }
            println!("User setup complete. Log out and back in for group changes to take effect.");
            Ok(())
        }
        Command::SetupPicoBootsel => setup_pico_bootsel(),
        Command::SetupRust => {
            if !program_exists("cargo") {
                run_program(ProcessCommand::new("sh").args(["-c", "curl https://sh.rustup.rs -sSf | sh -s -- -y"]))?;
            }
            run_program(rust_tool("rustup").args(["target", "add", "thumbv6m-none-eabi"]))?;
            if !program_exists("elf2uf2-rs") {
                run_program(rust_tool("cargo").args(["install", "elf2uf2-rs"]))?;
            }
            Ok(())
        }
        Command::SetupOrt => pete(["check", "-p", "pete-mouth"]),
        Command::SetupKinect => {
            if ProcessCommand::new("apt-cache").args(["show", "libfreenect-dev"]).status()?.success() {
                run_program(ProcessCommand::new("sudo").args(["apt-get", "install", "-y", "libfreenect-dev", "freenect"]))
            } else {
                fail("libfreenect-dev is unavailable; run `just setup-kinect-from-source`")
            }
        }
        Command::SetupKinectFromSource => setup_kinect_from_source(),
        Command::SetupTts => fetch_asset(
            &data_home().join("tongues/models/piper/en_US-ryan-medium.onnx"),
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        ).and_then(|_| fetch_asset(
            &data_home().join("tongues/models/piper/en_US-ryan-medium.onnx.json"),
            "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
        )),
        Command::SetupWhisper => fetch_asset(
            &data_home().join("pete/models/whisper/ggml-tiny.en.bin"),
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
        ),
        Command::Fetch => { run(Command::SetupTts)?; run(Command::SetupWhisper) }
        Command::ForebrainBuild => pete(["build", "--locked", "--release", "-p", "pete-higher-brain"]),
        Command::ForebrainValidate { config } => pete(["run", "-q", "-p", "pete-higher-brain", "--", "validate-node", "--config", path(&config)?]),
        Command::ForebrainProvision { inventory } => ansible(["-i", path(&inventory)?, "provisioning/forebrain/site.yml"]),
        Command::ForebrainProvisionCheck { inventory } => ansible(["--check", "--diff", "-i", path(&inventory)?, "provisioning/forebrain/site.yml"]),
        Command::ForebrainRemove { inventory } => ansible(["-i", path(&inventory)?, "provisioning/forebrain/remove.yml"]),
        Command::Fmt => pete(["fmt", "--all"]),
        Command::Check => pete(["check", "--workspace"]),
        Command::Test => pete(["test", "--workspace"]),
        Command::Clippy => pete(["clippy", "--workspace", "--all-targets", "--", "-D", "warnings"]),
        Command::Clean => pete(["clean"]),
        Command::BrainstemBuild => brainstem(["build", "--release"]),
        Command::BrainstemUf2 => { run(Command::BrainstemBuild)?; elf_to_uf2("pete-brainstem.uf2") }
        Command::BrainstemFetchCyw43 => fetch_cyw43(),
        Command::BrainstemPicoWBuild => { run(Command::BrainstemFetchCyw43)?; brainstem(["build", "--release", "--no-default-features", "--features", "pico-w,service-mode,operator-debug"]) }
        Command::BrainstemPicoWUf2 => { run(Command::BrainstemPicoWBuild)?; elf_to_uf2("pete-brainstem-pico-w.uf2") }
        Command::Flash => flash(),
        Command::Skull => skull(),
        Command::ComposeConfig => docker(["compose", "config"]),
        Command::ComposeBuild => docker(["compose", "build", "pete-live"]),
        Command::Servers => docker(["compose", "up", "-d", "neo4j", "qdrant"]),
        Command::LiveServer => docker(["compose", "--profile", "pete", "up", "-d", "neo4j", "qdrant", "pete-live"]),
        Command::ServerLogs { service } => docker(["compose", "logs", "-f", &service]),
        Command::StopServers => docker(["compose", "down"]),
        Command::Sim => pete_tools(["sim"], &[]),
        Command::Say { text } => pete_tools(["mouth", &text], &[("PETE_TTS_OUTPUT_DEVICE", env_or("PETE_TTS_OUTPUT_DEVICE", ""))]),
        Command::Transcribe { wav } => pete_tools(["whisper-transcribe", path(&wav)?], &[]),
        Command::Robot { args } => robot(&args, &[]),
        Command::Possess { args } => possess(&args),
        Command::PhysicalQa { plan, out } => physical_qa::run(plan, out, possess),
        Command::Go { target } => go(target.as_deref().unwrap_or("virtual")),
        Command::Train { args } => train(&args),
        Command::Evolve { clear } => evolve(clear.as_deref(), false),
        Command::EvolveQuality { clear } => evolve(clear.as_deref(), true),
        Command::EvolveInfinite { clear } => evolve_infinite(clear.as_deref()),
        Command::DevCert => dev_cert(),
        Command::VirtualUrl => { println!("https://{}:{}/view/3d", lan_ip(), env_or("PETE_LIVE_PORT", "8787")); Ok(()) }
        Command::Run { args } => pete_tools(args.iter().map(String::as_str), &[]),
        Command::RehearseModels => rehearse_models(),
        Command::EvalScenarioSmoke => eval_scenario_smoke(),
        Command::InspectLedger => pete_tools(["inspect-ledger"], &[]),
        Command::HardwareEnv => pete_tools(["hardware-env"], &[]),
        Command::RealCockpit => { println!("Cockpit backend: {}\nCockpit port: {}", cockpit_backend()?, env_or("PETE_COCKPIT_PORT", "auto")); Ok(()) }
        Command::Cockpit { args } => pete_cockpit(args.iter().map(String::as_str)),
        Command::CodexSync => codex_sync(),
        Command::PicoBootselMount { umount, kernel_name } => pico_bootsel_mount(umount, &kernel_name),
        Command::NeatGenerationLimit { state_path, explicit_limit, default_limit, increment } => {
            println!("{}", neat_generation_limit(&state_path, explicit_limit, default_limit, increment));
            Ok(())
        }
    }
}
