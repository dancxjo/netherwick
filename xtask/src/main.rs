use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
mod physical_qa;
#[cfg(unix)]
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::{
    env,
    error::Error,
    ffi::OsStr,
    fs,
    io::{self, BufRead, BufReader, IsTerminal, Read, Write},
    path::{Path, PathBuf},
    process::{Command as ProcessCommand, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, SystemTime},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

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

fn pete<const N: usize>(args: [&str; N]) -> Result<()> {
    run_program(ProcessCommand::new("cargo").args(args))
}
fn ansible<const N: usize>(args: [&str; N]) -> Result<()> {
    run_program(ProcessCommand::new("ansible-playbook").args(args))
}
fn docker<const N: usize>(args: [&str; N]) -> Result<()> {
    run_program(ProcessCommand::new("docker").args(args))
}
fn brainstem<const N: usize>(args: [&str; N]) -> Result<()> {
    run_program(
        ProcessCommand::new("cargo")
            .args(args)
            .current_dir("crates/pete-brainstem"),
    )
}
fn path(value: &Path) -> Result<&str> {
    value
        .to_str()
        .ok_or_else(|| io::Error::other("path is not UTF-8").into())
}
fn fail<T>(message: impl Into<String>) -> Result<T> {
    Err(io::Error::other(message.into()).into())
}

fn run_program(command: &mut ProcessCommand) -> Result<()> {
    let status = command.status()?;
    if status.success() {
        Ok(())
    } else {
        fail(format!("command failed with {status}"))
    }
}

fn pete_tools<'a>(args: impl IntoIterator<Item = &'a str>, envs: &[(&str, String)]) -> Result<()> {
    let mut command = ProcessCommand::new("cargo");
    command.args(["run", "-p", "pete-tools", "--"]);
    command.args(args);
    for (key, value) in envs {
        command.env(key, value);
    }
    run_program(&mut command)
}

fn pete_cockpit<'a>(args: impl IntoIterator<Item = &'a str>) -> Result<()> {
    let mut command = ProcessCommand::new("cargo");
    command.args(["run", "-q", "-p", "pete-cockpit", "--bin", "pete-cockpit", "--"]);
    command.args(args);
    command.env("CARGO_BUILD_JOBS", env_or("CARGO_BUILD_JOBS", "1"));
    run_program(&mut command)
}

fn env_or(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_owned())
}
fn env_flag(name: &str) -> bool {
    matches!(env_or(name, "").as_str(), "1" | "true" | "on" | "yes")
}
fn program_exists(name: &str) -> bool {
    ProcessCommand::new("sh")
        .args(["-c", &format!("command -v {name} >/dev/null 2>&1")])
        .status()
        .is_ok_and(|s| s.success())
}

fn rust_tool(name: &str) -> ProcessCommand {
    if program_exists(name) {
        ProcessCommand::new(name)
    } else {
        ProcessCommand::new(
            PathBuf::from(env_or("HOME", "."))
                .join(".cargo/bin")
                .join(name),
        )
    }
}
fn data_home() -> PathBuf {
    env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env_or("HOME", ".")).join(".local/share"))
}

fn fetch_asset(destination: &Path, url: &str) -> Result<()> {
    if destination.is_file() && destination.metadata()?.len() > 0 {
        return Ok(());
    }
    fs::create_dir_all(
        destination
            .parent()
            .ok_or_else(|| io::Error::other("asset has no parent"))?,
    )?;
    run_program(ProcessCommand::new("curl").args([
        "-fL",
        "--retry",
        "3",
        "--retry-delay",
        "2",
        "-o",
        path(destination)?,
        url,
    ]))
}

fn setup_pico_bootsel() -> Result<()> {
    pete(["build", "--release", "-p", "xtask"])?;
    let user = env_or("SUDO_USER", &env_or("USER", "root"));
    let group = output("id", &["-gn", &user])?;
    run_program(ProcessCommand::new("sudo").args([
        "install",
        "-d",
        "-m",
        "0755",
        "/usr/local/lib/netherwick",
    ]))?;
    run_program(ProcessCommand::new("sudo").args([
        "install",
        "-m",
        "0755",
        "target/release/xtask",
        "/usr/local/lib/netherwick/pico-bootsel-mount",
    ]))?;
    run_program(ProcessCommand::new("sudo").args([
        "install",
        "-m",
        "0644",
        "configs/systemd/netherwick-pico-bootsel-mount@.service",
        "/etc/systemd/system/netherwick-pico-bootsel-mount@.service",
    ]))?;
    run_program(ProcessCommand::new("sudo").args([
        "install",
        "-m",
        "0644",
        "configs/udev/99-netherwick-pico-bootsel.rules",
        "/etc/udev/rules.d/99-netherwick-pico-bootsel.rules",
    ]))?;
    let defaults = env::temp_dir().join(format!("netherwick-pico-bootsel-{}", std::process::id()));
    fs::write(
        &defaults,
        format!(
            "PICO_BOOTSEL_USER={user}\nPICO_BOOTSEL_GROUP={group}\nPICO_BOOTSEL_MOUNT_BASE=/media\n"
        ),
    )?;
    run_program(ProcessCommand::new("sudo").args([
        "install",
        "-m",
        "0644",
        path(&defaults)?,
        "/etc/default/netherwick-pico-bootsel",
    ]))?;
    fs::remove_file(defaults)?;
    run_program(ProcessCommand::new("sudo").args(["systemctl", "daemon-reload"]))?;
    run_program(ProcessCommand::new("sudo").args(["udevadm", "control", "--reload-rules"]))?;
    let _ = ProcessCommand::new("sudo")
        .args([
            "udevadm",
            "trigger",
            "--subsystem-match=block",
            "--property-match=ID_FS_LABEL=RPI-RP2",
        ])
        .status();
    println!("Pico BOOTSEL automount installed for {user}:{group}.");
    Ok(())
}

fn setup_kinect_from_source() -> Result<()> {
    let source = Path::new(".vendor/libfreenect");
    if !source.join(".git").is_dir() {
        run_program(ProcessCommand::new("git").args([
            "clone",
            "https://github.com/OpenKinect/libfreenect.git",
            path(source)?,
        ]))?;
    }
    run_program(ProcessCommand::new("cmake").args([
        "-S",
        path(source)?,
        "-B",
        ".vendor/libfreenect/build",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_CPP=ON",
        "-DBUILD_AUDIO=ON",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_OPENNI2_DRIVER=OFF",
    ]))?;
    run_program(ProcessCommand::new("cmake").args(["--build", ".vendor/libfreenect/build", "-j"]))?;
    run_program(ProcessCommand::new("sudo").args([
        "cmake",
        "--install",
        ".vendor/libfreenect/build",
    ]))
}

fn fetch_cyw43() -> Result<()> {
    let directory = Path::new("crates/pete-brainstem/firmware/cyw43");
    fs::create_dir_all(directory)?;
    let base = format!(
        "https://raw.githubusercontent.com/embassy-rs/embassy/{}/cyw43-firmware",
        env_or("CYW43_FIRMWARE_REF", "main")
    );
    for file in [
        "43439A0.bin",
        "43439A0_clm.bin",
        "nvram_rp2040.bin",
        "LICENSE-permissive-binary-license-1.0.txt",
    ] {
        run_program(ProcessCommand::new("curl").args([
            "-fL",
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "-o",
            path(&directory.join(file))?,
            &format!("{base}/{file}"),
        ]))?;
    }
    Ok(())
}

fn elf_to_uf2(name: &str) -> Result<()> {
    run_program(ProcessCommand::new("elf2uf2-rs").args([
        "crates/pete-brainstem/target/thumbv6m-none-eabi/release/pete-brainstem",
        &format!("crates/pete-brainstem/target/thumbv6m-none-eabi/release/{name}"),
    ]))
}

fn flash() -> Result<()> {
    run(Command::BrainstemPicoWUf2)?;
    let uf2 = Path::new(
        "crates/pete-brainstem/target/thumbv6m-none-eabi/release/pete-brainstem-pico-w.uf2",
    );
    if !uf2.is_file() || uf2.metadata()?.len() == 0 {
        return fail(format!("UF2 not found: {}", uf2.display()));
    }
    let mut mount = bootsel_mount().or_else(mount_bootsel_block);
    if mount.is_none() {
        println!("Requesting authorized BOOTSEL via USB CDC");
        let mut request = ProcessCommand::new("cargo");
        request
            .args([
                "run",
                "-q",
                "-p",
                "pete-cockpit",
                "--example",
                "service_bootsel",
            ])
            .env("PETE_BOOTSEL_USB", "1");
        let mut requested = request.status()?.success();
        let bootsel_url = env_or("PICO_W_BOOTSEL_URL", "http://192.168.4.1/command");
        let host = bootsel_host(&bootsel_url);
        if !requested && connected_to_brainstem_wifi(&host) {
            println!("USB BOOTSEL failed; requesting authorized BOOTSEL via {bootsel_url}");
            let mut request = ProcessCommand::new("cargo");
            request
                .args([
                    "run",
                    "-q",
                    "-p",
                    "pete-cockpit",
                    "--example",
                    "service_bootsel",
                ])
                .env("PETE_BRAINSTEM_HTTP_HOST", &host);
            requested = request.status()?.success();
        }
        if !requested {
            if !env_flag("PETE_ALLOW_LEGACY_BOOTSEL") {
                return fail(
                    "authorized BOOTSEL failed; set PETE_ALLOW_LEGACY_BOOTSEL=1 only for explicit development recovery",
                );
            }
            if !connected_to_brainstem_wifi(&host) {
                return fail("legacy BOOTSEL requires an active Pete brainstem Wi-Fi connection");
            }
            eprintln!("WARNING: USING UNAUDITED LEGACY BOOTSEL DEVELOPMENT FALLBACK");
            run_program(ProcessCommand::new("curl").args([
                "-fsS",
                "--max-time",
                "3",
                "-H",
                "Content-Type: application/json",
                "-d",
                r#"{"kind":"bootsel","command_id":1}"#,
                &bootsel_url,
            ]))?;
        }
        println!("Waiting for RPI-RP2 mount");
        let timeout = env_or("PICO_W_MOUNT_TIMEOUT_SECS", "30")
            .parse::<u64>()
            .unwrap_or(30);
        for _ in 0..timeout {
            mount = bootsel_mount().or_else(mount_bootsel_block);
            if mount.is_some() {
                break;
            }
            thread::sleep(Duration::from_secs(1));
        }
    }
    let mount =
        mount.ok_or_else(|| io::Error::other("RPI-RP2 mount was not found; set PICO_W_MOUNT"))?;
    println!("Copying {} to {}", uf2.display(), mount.display());
    fs::copy(uf2, mount.join("pete-brainstem-pico-w.uf2"))?;
    run_program(&mut ProcessCommand::new("sync"))?;
    println!("Flash copy complete");
    Ok(())
}

fn bootsel_mount() -> Option<PathBuf> {
    let explicit = env_or("PICO_W_MOUNT", "");
    let candidates = [
        PathBuf::from(explicit),
        PathBuf::from(format!("/media/{}/RPI-RP2", env_or("USER", "root"))),
        PathBuf::from(format!("/run/media/{}/RPI-RP2", env_or("USER", "root"))),
        PathBuf::from("/media/RPI-RP2"),
        PathBuf::from("/Volumes/RPI-RP2"),
    ];
    candidates
        .into_iter()
        .find(|candidate| {
            !candidate.as_os_str().is_empty()
                && candidate.is_dir()
                && ProcessCommand::new("test")
                    .args(["-w", path(candidate).unwrap_or("")])
                    .status()
                    .is_ok_and(|status| status.success())
        })
        .or_else(|| {
            output("lsblk", &["-rpo", "LABEL,MOUNTPOINT"])
                .ok()?
                .lines()
                .find_map(|line| {
                    let mut fields = line.split_whitespace();
                    if fields.next()? != "RPI-RP2" {
                        return None;
                    }
                    Some(PathBuf::from(fields.next()?))
                })
                .filter(|candidate| candidate.is_dir())
        })
}

fn mount_bootsel_block() -> Option<PathBuf> {
    let listing = output("lsblk", &["-rnpo", "LABEL,PATH,FSTYPE,MOUNTPOINT"]).ok()?;
    let block = listing.lines().find_map(|line| {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        (fields.len() >= 3 && fields[0] == "RPI-RP2" && fields[2] == "vfat")
            .then(|| fields[1].to_owned())
    })?;
    if program_exists("udisksctl")
        && ProcessCommand::new("udisksctl")
            .args(["mount", "-b", &block])
            .status()
            .is_ok_and(|status| status.success())
    {
        return bootsel_mount();
    }
    let mount = PathBuf::from(format!("/media/{}/RPI-RP2", env_or("USER", "root")));
    if run_program(ProcessCommand::new("sudo").args(["mkdir", "-p", path(&mount).ok()?])).is_err()
        || run_program(ProcessCommand::new("sudo").args([
            "mount",
            "-t",
            "vfat",
            "-o",
            &format!(
                "uid={},gid={},umask=022",
                output("id", &["-u"]).ok()?,
                output("id", &["-g"]).ok()?
            ),
            &block,
            path(&mount).ok()?,
        ]))
        .is_err()
    {
        return None;
    }
    bootsel_mount()
}

fn bootsel_host(url: &str) -> String {
    let host = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url)
        .split('/')
        .next()
        .unwrap_or(url);
    if host.contains(':') {
        host.to_owned()
    } else {
        format!("{host}:80")
    }
}

fn connected_to_brainstem_wifi(host: &str) -> bool {
    let pete_network = if program_exists("nmcli") {
        output("nmcli", &["-t", "-f", "active,ssid", "dev", "wifi"])
            .unwrap_or_default()
            .lines()
            .any(|line| line.to_ascii_lowercase().starts_with("yes:pete-"))
    } else if program_exists("iwgetid") {
        output("iwgetid", &["-r"])
            .unwrap_or_default()
            .to_ascii_lowercase()
            .starts_with("pete-")
    } else {
        false
    };
    pete_network && curl_ok(&format!("http://{host}/status.json"))
}

fn skull() -> Result<()> {
    let python = if Path::new("skeleton/brainstem/.venv/bin/python").is_file() {
        "skeleton/brainstem/.venv/bin/python"
    } else {
        "python"
    };
    run_program(
        ProcessCommand::new(python)
            .arg("skull.py")
            .current_dir("skeleton/brainstem"),
    )
}

fn cockpit_backend() -> Result<String> {
    if let Ok(value) = env::var("PETE_COCKPIT_BACKEND") {
        if !value.is_empty() {
            return Ok(value);
        }
    }
    let configured = env_or("PETE_COCKPIT_PORT", "auto");
    let port = if configured == "auto" {
        serial_candidates().into_iter().next().unwrap_or_default()
    } else {
        configured
    };
    if !env_flag("PETE_SKIP_COCKPIT_UART") && !port.is_empty() {
        let status = ProcessCommand::new("cargo")
            .args([
                "run",
                "-q",
                "-p",
                "pete-cockpit",
                "--example",
                "contract_check",
                "--",
                "uart",
                &port,
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()?;
        if status.success() {
            return Ok("uart".to_owned());
        }
    }
    let wifi = output("nmcli", &["-t", "-f", "active,ssid", "dev", "wifi"])
        .unwrap_or_default()
        .lines()
        .any(|line| line.to_ascii_lowercase().starts_with("yes:pete-"));
    if wifi
        && ProcessCommand::new("curl")
            .args([
                "-fsS",
                "--connect-timeout",
                "1",
                "--max-time",
                "2",
                &format!(
                    "http://{}/status.json",
                    env_or("PETE_BRAINSTEM_HTTP_HOST", "192.168.4.1:80")
                ),
            ])
            .status()
            .is_ok_and(|s| s.success())
    {
        return Ok("wifi".to_owned());
    }
    Ok(if env_flag("PETE_SKIP_COCKPIT_UART") {
        "none"
    } else {
        "uart"
    }
    .to_owned())
}

fn serial_candidates() -> Vec<String> {
    let mut candidates = Vec::new();
    for directory in ["/dev/serial/by-id", "/dev"] {
        if let Ok(entries) = fs::read_dir(directory) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.contains("Pete_Brainstem")
                    || (directory == "/dev"
                        && (name.starts_with("ttyACM") || name.starts_with("ttyUSB")))
                {
                    candidates.push(entry.path().display().to_string());
                }
            }
        }
    }
    candidates.sort();
    candidates
}

fn robot(args: &[String], overrides: &[(&str, String)]) -> Result<()> {
    run_program(&mut robot_process(args, overrides)?)
}

fn robot_process(args: &[String], overrides: &[(&str, String)]) -> Result<ProcessCommand> {
    ensure_memory_servers()?;
    dev_cert()?;
    let mut command = vec![
        "robot".to_owned(),
        "--mode".to_owned(),
        override_or_env(overrides, "PETE_ROBOT_MODE", "read-only"),
        "--cockpit".to_owned(),
        match override_value(overrides, "PETE_COCKPIT_BACKEND") {
            Some(backend) => backend,
            None => cockpit_backend()?,
        },
        "--create-port".to_owned(),
        override_or_env(overrides, "PETE_COCKPIT_PORT", "auto"),
        "--ledger".to_owned(),
        override_or_env(overrides, "PETE_ROBOT_LEDGER", "data/ledger/real/robot"),
    ];
    optional_arg(&mut command, "CAMERA_DEVICE", "/dev/video0", "--camera");
    optional_arg(&mut command, "MIC_DEVICE", "", "--mic");
    optional_arg(&mut command, "IMU_DEVICE", "", "--imu");
    optional_arg(&mut command, "GPS_SERIAL_PORT", "", "--gps");
    if let Ok(lidar) = env::var("LIDAR_SERIAL_PORT") {
        if !lidar.is_empty() {
            command.extend([
                "--lidar".to_owned(),
                lidar,
                "--lidar-yaw-deg".to_owned(),
                env_or("LIDAR_YAW_DEG", "0"),
                "--lidar-pitch-deg".to_owned(),
                env_or("LIDAR_PITCH_DEG", "0"),
                "--lidar-roll-deg".to_owned(),
                env_or("LIDAR_ROLL_DEG", "0"),
                "--lidar-height-m".to_owned(),
                env_or("LIDAR_HEIGHT_M", "0"),
                "--lidar-forward-m".to_owned(),
                env_or("LIDAR_FORWARD_M", "0"),
                "--lidar-left-m".to_owned(),
                env_or("LIDAR_LEFT_M", "0"),
            ]);
        }
    }
    if env_flag("PETE_KINECT_DEPTH")
        || env::var_os("PETE_KINECT_DEPTH").is_none_or(|value| value != OsStr::new("0"))
    {
        command.extend([
            "--kinect-depth".to_owned(),
            "--kinect-rgb-target-luma".to_owned(),
            env_or("KINECT_RGB_TARGET_LUMA", "0.32"),
            "--kinect-rgb-auto-gain-max".to_owned(),
            env_or("KINECT_RGB_AUTO_GAIN_MAX", "3.0"),
            "--kinect-rgb-gain".to_owned(),
            env_or("KINECT_RGB_GAIN", "1.0"),
            "--kinect-rgb-gamma".to_owned(),
            env_or("KINECT_RGB_GAMMA", "0.80"),
            "--kinect-rgb-brightness".to_owned(),
            env_or("KINECT_RGB_BRIGHTNESS", "0.0"),
        ]);
    }
    command.extend([
        "--dashboard".to_owned(),
        env_or("PETE_ROBOT_DASHBOARD", "0.0.0.0:3000"),
        "--dashboard-tls".to_owned(),
        "--dashboard-tls-cert".to_owned(),
        env_or("PETE_ROBOT_DASHBOARD_TLS_CERT", "certs/pete-dev.crt"),
        "--dashboard-tls-key".to_owned(),
        env_or("PETE_ROBOT_DASHBOARD_TLS_KEY", "certs/pete-dev.key"),
    ]);
    command.extend(args.iter().cloned());
    let mut process = ProcessCommand::new("cargo");
    process.args(["run", "-p", "pete-tools", "--"]);
    process.args(&command);
    process.env(
        "PETE_TTS_OUTPUT_DEVICE",
        env_or("PETE_TTS_OUTPUT_DEVICE", ""),
    );
    for (key, value) in overrides {
        process.env(key, value);
    }
    Ok(process)
}

fn override_value(overrides: &[(&str, String)], name: &str) -> Option<String> {
    overrides
        .iter()
        .find_map(|(key, value)| (*key == name).then(|| value.clone()))
}

fn override_or_env(overrides: &[(&str, String)], name: &str, default: &str) -> String {
    override_value(overrides, name).unwrap_or_else(|| env_or(name, default))
}

fn optional_arg(command: &mut Vec<String>, env_name: &str, default: &str, flag: &str) {
    let value = env_or(env_name, default);
    if !value.is_empty() {
        command.extend([flag.to_owned(), value]);
    }
}

fn possess(args: &[String]) -> Result<()> {
    let mut device = env::var("PETE_BRAINSTEM_DEVICE_ID")
        .map_err(|_| io::Error::other("set PETE_BRAINSTEM_DEVICE_ID in .env"))?;
    let mut boot = env_or("PETE_BRAINSTEM_BOOT_ID", "unknown");
    let mut port = env_or("PETE_COCKPIT_PORT", "auto");
    if port != "auto" && !port.is_empty() && !Path::new(&port).exists() {
        if let Some(detected) = single_brainstem_port() {
            println!(
                "Configured PETE_COCKPIT_PORT is missing: {port}\nDetected one wired brainstem candidate: {}",
                detected.display()
            );
            let (live_device, live_boot) = bootstrap_brainstem(Some(&detected))?;
            if live_device != device && !env_flag("PETE_ACCEPT_BRAINSTEM_REPLACEMENT") {
                return fail(format!(
                    "detected brainstem {live_device}, but .env pins {device}; rerun with PETE_ACCEPT_BRAINSTEM_REPLACEMENT=1 to accept the wired replacement"
                ));
            }
            device = live_device;
            boot = live_boot;
            port = detected.display().to_string();
            set_dotenv("PETE_BRAINSTEM_DEVICE_ID", &device)?;
            set_dotenv("PETE_BRAINSTEM_BOOT_ID", &boot)?;
            set_dotenv("PETE_COCKPIT_PORT", &port)?;
            println!(
                "Updated .env brainstem pin from wired USB: device={device} boot={boot} port={port}"
            );
        }
    }
    let backend_was_explicit =
        env::var("PETE_COCKPIT_BACKEND").is_ok_and(|value| !value.is_empty());
    let mut backend = if backend_was_explicit {
        env_or("PETE_COCKPIT_BACKEND", "uart")
    } else if port != "auto" && Path::new(&port).exists() {
        "uart".to_owned()
    } else {
        cockpit_backend()?
    };
    let (mut status, mut log) = possession_attempt(args, &device, &boot, &backend, &port)?;
    if status.success() {
        return Ok(());
    }
    let host = env_or("PETE_BRAINSTEM_HTTP_HOST", "192.168.4.1:80");
    if !backend_was_explicit && backend == "uart" && connected_to_brainstem_wifi(&host) {
        backend = "wifi".to_owned();
        println!("Brainstem USB/UART failed; retrying possession over Pete Wi-Fi at {host}.");
        (status, log) = possession_attempt(args, &device, &boot, &backend, &port)?;
        if status.success() {
            return Ok(());
        }
    }
    if backend == "wifi" && log.contains("reason_code: InvalidIdentity") {
        println!("Wi-Fi identity continuity is not established; bootstrapping the pinned brainstem over USB.");
        bootstrap_brainstem(single_brainstem_port().as_deref())?;
        (status, log) = possession_attempt(args, &device, &boot, &backend, &port)?;
        if status.success() {
            return Ok(());
        }
    }
    if let Some(live_boot) = boot_identity_mismatch(&log) {
        set_dotenv("PETE_BRAINSTEM_BOOT_ID", &live_boot)?;
        println!(
            "Accepted current boot identity for pinned device {device}; updated .env and retrying."
        );
        (status, _) = possession_attempt(args, &device, &live_boot, &backend, &port)?;
    }
    if status.success() {
        Ok(())
    } else {
        fail(format!("possession exited with {status}"))
    }
}

fn possession_attempt(
    args: &[String],
    device: &str,
    boot: &str,
    backend: &str,
    port: &str,
) -> Result<(std::process::ExitStatus, String)> {
    let endpoint = if backend == "wifi" {
        env_or("PETE_BRAINSTEM_HTTP_HOST", "192.168.4.1:80")
    } else {
        port.to_owned()
    };
    println!(
        "Taking brainstem possession over {backend} at {endpoint}\ndevice={device} boot={boot}\nlimits: 50 mm/s linear, 500 mrad/s angular; exit performs STOP then exorcize"
    );
    let mut robot_args = vec![
        "--brainstem-device-id".to_owned(),
        device.to_owned(),
        "--brainstem-boot-id".to_owned(),
        boot.to_owned(),
        "--max-linear-mm-s".to_owned(),
        "50".to_owned(),
        "--max-angular-mrad-s".to_owned(),
        "500".to_owned(),
        "--autonomous-motion".to_owned(),
        "--imu".to_owned(),
        "none".to_owned(),
        "--gps".to_owned(),
        "none".to_owned(),
        "--llm-provider".to_owned(),
        "disabled".to_owned(),
        "--capture".to_owned(),
        env_or("PETE_POSSESSION_CAPTURE", "data/captures/real/possession"),
    ];
    robot_args.extend(args.iter().cloned());
    let mut command = robot_process(
        &robot_args,
        &[
            ("PETE_ROBOT_MODE", "possession-slow".to_owned()),
            ("PETE_COCKPIT_BACKEND", backend.to_owned()),
            ("PETE_COCKPIT_PORT", port.to_owned()),
            (
                "PETE_ROBOT_LEDGER",
                env_or("PETE_POSSESSION_LEDGER", "data/ledger/real/possession"),
            ),
            ("CAMERA_DEVICE", "".to_owned()),
            ("MIC_DEVICE", "".to_owned()),
            ("IMU_DEVICE", "".to_owned()),
            ("GPS_SERIAL_PORT", "".to_owned()),
            ("PETE_KINECT_DEPTH", "0".to_owned()),
        ],
    )?;
    run_program_captured(&mut command)
}

fn run_program_captured(
    command: &mut ProcessCommand,
) -> Result<(std::process::ExitStatus, String)> {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let log = Arc::new(Mutex::new(String::new()));
    let stdout = stream_and_capture(
        child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("missing child stdout"))?,
        false,
        Arc::clone(&log),
    );
    let stderr = stream_and_capture(
        child
            .stderr
            .take()
            .ok_or_else(|| io::Error::other("missing child stderr"))?,
        true,
        Arc::clone(&log),
    );
    let status = child.wait()?;
    stdout
        .join()
        .map_err(|_| io::Error::other("stdout reader panicked"))??;
    stderr
        .join()
        .map_err(|_| io::Error::other("stderr reader panicked"))??;
    let captured = log
        .lock()
        .map_err(|_| io::Error::other("captured output lock poisoned"))?
        .clone();
    Ok((status, captured))
}

fn stream_and_capture<R: Read + Send + 'static>(
    reader: R,
    is_stderr: bool,
    log: Arc<Mutex<String>>,
) -> thread::JoinHandle<io::Result<()>> {
    thread::spawn(move || {
        for line in BufReader::new(reader).lines() {
            let line = line?;
            if is_stderr {
                eprintln!("{line}");
            } else {
                println!("{line}");
            }
            let mut captured = log
                .lock()
                .map_err(|_| io::Error::other("captured output lock poisoned"))?;
            captured.push_str(&line);
            captured.push('\n');
        }
        Ok(())
    })
}

fn single_brainstem_port() -> Option<PathBuf> {
    let candidates = fs::read_dir("/dev/serial/by-id")
        .ok()?
        .flatten()
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .contains("Pete_Brainstem")
        })
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    (candidates.len() == 1).then(|| candidates[0].clone())
}

fn bootstrap_brainstem(port: Option<&Path>) -> Result<(String, String)> {
    let mut command = ProcessCommand::new("cargo");
    command
        .args([
            "run",
            "-q",
            "-p",
            "pete-cockpit",
            "--example",
            "motherbrain_bootstrap",
            "--",
            "--identity-only",
        ])
        .env("PETE_BRAINSTEM_DEVICE_ID", "");
    if let Some(port) = port {
        command.env("PETE_COCKPIT_PORT", port);
    }
    let output = command.output()?;
    print!("{}", String::from_utf8_lossy(&output.stdout));
    eprint!("{}", String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        return fail(format!(
            "USB identity bootstrap failed with {}",
            output.status
        ));
    }
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let device = prefixed_value(&text, "brainstem identity: ")
        .ok_or_else(|| io::Error::other("USB bootstrap did not report a brainstem identity"))?;
    let boot = prefixed_value(&text, "brainstem boot: ").ok_or_else(|| {
        io::Error::other("USB bootstrap did not report a brainstem boot identity")
    })?;
    Ok((device, boot))
}

fn prefixed_value(text: &str, prefix: &str) -> Option<String> {
    text.lines()
        .filter_map(|line| line.strip_prefix(prefix))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .next_back()
        .map(str::to_owned)
}

fn boot_identity_mismatch(log: &str) -> Option<String> {
    log.lines()
        .filter_map(|line| {
            line.strip_prefix("Error: brainstem boot identity mismatch: expected ")?
                .split_once(" received ")
                .map(|(_, received)| {
                    received
                        .split_whitespace()
                        .next()
                        .unwrap_or(received)
                        .to_owned()
                })
        })
        .next_back()
}

fn set_dotenv(key: &str, value: &str) -> Result<()> {
    let path = Path::new(".env");
    let existing = fs::read_to_string(path).unwrap_or_default();
    let mut replaced = false;
    let mut lines = existing
        .lines()
        .map(|line| {
            if line.starts_with(&format!("{key}=")) {
                replaced = true;
                format!("{key}={value}")
            } else {
                line.to_owned()
            }
        })
        .collect::<Vec<_>>();
    if !replaced {
        lines.push(format!("{key}={value}"));
    }
    fs::write(path, format!("{}\n", lines.join("\n")))?;
    Ok(())
}

fn ensure_memory_servers() -> Result<()> {
    let neo4j = env_or(
        "PETE_NEO4J_HTTP_URL",
        &format!(
            "http://127.0.0.1:{}",
            env_or("PETE_NEO4J_HTTP_PORT", "7474")
        ),
    );
    let qdrant = env_or(
        "PETE_QDRANT_URL",
        &format!(
            "http://127.0.0.1:{}",
            env_or("PETE_QDRANT_HTTP_PORT", "6333")
        ),
    );
    if curl_ok(&neo4j)
        && (curl_ok(&format!("{}/readyz", qdrant.trim_end_matches('/')))
            || curl_ok(&format!("{}/collections", qdrant.trim_end_matches('/'))))
    {
        return Ok(());
    }
    run(Command::Servers)
}

fn curl_ok(url: &str) -> bool {
    ProcessCommand::new("curl")
        .args(["-fsS", "--max-time", "2", url])
        .status()
        .is_ok_and(|s| s.success())
}

fn dev_cert() -> Result<()> {
    let cert = PathBuf::from(env_or(
        "PETE_ROBOT_DASHBOARD_TLS_CERT",
        "certs/pete-dev.crt",
    ));
    let key = PathBuf::from(env_or("PETE_ROBOT_DASHBOARD_TLS_KEY", "certs/pete-dev.key"));
    if cert.is_file() && key.is_file() {
        return Ok(());
    }
    fs::create_dir_all(cert.parent().unwrap_or(Path::new(".")))?;
    fs::create_dir_all(key.parent().unwrap_or(Path::new(".")))?;
    let san = if lan_ip() == "127.0.0.1" {
        String::new()
    } else {
        format!(",IP:{}", lan_ip())
    };
    run_program(ProcessCommand::new("openssl").args([
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        path(&key)?,
        "-out",
        path(&cert)?,
        "-days",
        "365",
        "-subj",
        "/CN=pete.local",
        "-addext",
        &format!("subjectAltName=DNS:localhost,DNS:pete.local,IP:127.0.0.1{san}"),
    ]))
}

fn lan_ip() -> String {
    output("hostname", &["-I"])
        .ok()
        .and_then(|ips| ips.split_whitespace().next().map(str::to_owned))
        .unwrap_or_else(|| "127.0.0.1".to_owned())
}

fn go(target: &str) -> Result<()> {
    if target != "virtual" {
        return fail("usage: just go virtual");
    }
    dev_cert()?;
    let port = env_or("PETE_LIVE_PORT", "8787");
    let lan_ip = lan_ip();
    println!(
        "Pete Dream World is starting.\nVirtual training theater is collecting experience.\nDesktop: https://127.0.0.1:{port}/view/3d\nHeadset/LAN: https://{lan_ip}:{port}/view/3d\nScene JSON: https://{lan_ip}:{port}/view/scene\nThis serves robot/dream-world sensor data on the LAN; use only on trusted networks."
    );
    if program_exists("qrencode") {
        let _ = ProcessCommand::new("qrencode")
            .args([
                "-t",
                "ANSIUTF8",
                &format!("https://{lan_ip}:{port}/view/3d"),
            ])
            .status();
    }
    pete(["build", "-p", "pete-tools"])?;
    let mut child = ProcessCommand::new("target/debug/pete")
        .args([
            "sim",
            "--live",
            "--live-tls",
            "--live-addr",
            &format!("0.0.0.0:{port}"),
            "--live-tls-cert",
            "certs/pete-dev.crt",
            "--live-tls-key",
            "certs/pete-dev.key",
            "--action-selector",
            &env_or("PETE_ACTION_SELECTOR", "goal"),
            "--scenario",
            &env_or("PETE_SCENARIO", "dream"),
            "--steps",
            &env_or("PETE_SIM_STEPS", "1000000000"),
            "--tick-delay-ms",
            &env_or("PETE_TICK_DELAY_MS", "100"),
            "--ledger",
            &env_or("PETE_LEDGER", "data/ledger/virtual-live"),
        ])
        .spawn()?;
    if env_flag("PETE_OPEN_BROWSER") && program_exists("xdg-open") {
        if program_exists("curl") {
            for _ in 0..80 {
                if ProcessCommand::new("curl")
                    .args(["-kfsS", &format!("https://127.0.0.1:{port}/view/scene")])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .is_ok_and(|status| status.success())
                {
                    break;
                }
                if child.try_wait()?.is_some() {
                    break;
                }
                thread::sleep(Duration::from_millis(250));
            }
        } else {
            thread::sleep(Duration::from_secs(2));
        }
        let _ = ProcessCommand::new("xdg-open")
            .arg(format!("https://127.0.0.1:{port}/view/3d"))
            .spawn();
    }
    let status = child.wait()?;
    if status.success() {
        Ok(())
    } else {
        fail(format!("virtual world exited with {status}"))
    }
}

fn train(args: &[String]) -> Result<()> {
    if args.first().is_some_and(|value| value == "--neat") {
        if args.len() != 2 {
            return fail("usage: just train --neat locomotion");
        }
        return train_neat(&args[1]);
    }
    if args.len() > 1 || args.first().is_some_and(|value| value != "virtual") {
        return fail("usage: just train virtual | just train --neat locomotion");
    }
    pete_tools(
        [
            "train",
            "virtual",
            "--ledger",
            &env_or("PETE_LEDGER", "data/ledger/virtual-live"),
            "--out-dir",
            &env_or("PETE_MODEL_OUT", "data/models/virtual/latest"),
            "--report-out",
            &env_or("PETE_REPORT_OUT", "data/reports/virtual/latest.json"),
            "--epochs",
            &env_or("PETE_EPOCHS", "5"),
        ],
        &[],
    )
}

fn train_neat(behavior: &str) -> Result<()> {
    let report_dir = env_or("PETE_NEAT_REPORT_DIR", "data/reports/neat/locomotion-v2");
    let migrated = PathBuf::from(&report_dir).join("trainer-state-schema3-leave-start-region.json");
    let state = env::var("PETE_NEAT_STATE_CHECKPOINT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            if migrated.is_file() {
                migrated
            } else {
                PathBuf::from(&report_dir).join("trainer-state.json")
            }
        });
    let resume = env_or("PETE_NEAT_RESUME", "");
    let founders = env_or("PETE_NEAT_FOUNDERS_REPORT", "");
    let start_stage = env_or("PETE_NEAT_START_STAGE", "");
    if !resume.is_empty() && !founders.is_empty() {
        return fail("PETE_NEAT_RESUME and PETE_NEAT_FOUNDERS_REPORT are mutually exclusive");
    }
    let mut extra = Vec::new();
    let continuation;
    if !resume.is_empty() {
        extra.extend(["--resume".to_owned(), resume.clone()]);
        continuation = PathBuf::from(resume);
        println!("NEAT continuation: resuming {}", continuation.display());
    } else if !founders.is_empty() {
        println!("NEAT continuation: reconstructing founders from {founders}");
        extra.extend(["--founders-report".to_owned(), founders]);
        extra.extend([
            "--start-stage".to_owned(),
            if start_stage.is_empty() {
                "leave-start-region".to_owned()
            } else {
                start_stage.clone()
            },
        ]);
        continuation = PathBuf::new();
    } else if !env_flag("PETE_NEAT_FRESH") && state.is_file() {
        extra.extend(["--resume".to_owned(), path(&state)?.to_owned()]);
        continuation = state.clone();
        println!("NEAT continuation: resuming {}", state.display());
    } else if !env_flag("PETE_NEAT_FRESH")
        && Path::new("data/reports/neat/locomotion/training-report.json").is_file()
    {
        let founders = "data/reports/neat/locomotion/training-report.json";
        println!("NEAT continuation: reconstructing founders from {founders}");
        extra.extend(["--founders-report".to_owned(), founders.to_owned()]);
        extra.extend([
            "--start-stage".to_owned(),
            if start_stage.is_empty() {
                "leave-start-region".to_owned()
            } else {
                start_stage.clone()
            },
        ]);
        continuation = PathBuf::new();
    } else {
        if !start_stage.is_empty() {
            extra.extend(["--start-stage".to_owned(), start_stage]);
        }
        continuation = PathBuf::new();
        println!("NEAT continuation: starting a fresh competence-gated run");
    }
    let generations = neat_generation_limit(
        &continuation,
        env::var("PETE_NEAT_GENERATIONS_PER_STAGE")
            .ok()
            .and_then(|value| value.parse().ok()),
        120,
        120,
    )
    .to_string();
    let mut values = vec![
        "neat-train".to_owned(),
        behavior.to_owned(),
        "--generations-per-stage".to_owned(),
        generations,
        "--population".to_owned(),
        env_or("PETE_NEAT_POPULATION", "64"),
        "--episodes-per-genome".to_owned(),
        env_or("PETE_NEAT_EPISODES_PER_GENOME", "12"),
        "--steps".to_owned(),
        env_or("PETE_NEAT_STEPS", "300"),
        "--transfer-episodes".to_owned(),
        env_or("PETE_NEAT_TRANSFER_EPISODES", "500"),
        "--seed".to_owned(),
        env_or("PETE_NEAT_SEED", "7"),
        "--heldout-seed".to_owned(),
        env_or("PETE_NEAT_HELDOUT_SEED", "9000001"),
        "--validation-seed".to_owned(),
        env_or("PETE_NEAT_VALIDATION_SEED", "8000001"),
        "--validation-every".to_owned(),
        env_or("PETE_NEAT_VALIDATION_EVERY", "4"),
        "--validation-passes".to_owned(),
        env_or("PETE_NEAT_VALIDATION_PASSES", "2"),
        "--compatibility-threshold".to_owned(),
        env_or("PETE_NEAT_COMPATIBILITY_THRESHOLD", "2.2"),
        "--compatibility-threshold-floor".to_owned(),
        env_or("PETE_NEAT_COMPATIBILITY_THRESHOLD_FLOOR", "0.05"),
        "--target-species-min".to_owned(),
        env_or("PETE_NEAT_TARGET_SPECIES_MIN", "4"),
        "--target-species-max".to_owned(),
        env_or("PETE_NEAT_TARGET_SPECIES_MAX", "9"),
        "--checkpoint".to_owned(),
        env_or("PETE_NEAT_CHECKPOINT", "data/models/locomotion_neat_v0"),
        "--report-dir".to_owned(),
        report_dir,
        "--state-checkpoint".to_owned(),
        path(&state)?.to_owned(),
        "--capture-root".to_owned(),
        env_or("PETE_NEAT_CAPTURE_ROOT", "data/captures/neat/locomotion-v2"),
        "--capture-every".to_owned(),
        env_or("PETE_NEAT_CAPTURE_EVERY", "2"),
        "--rehearsal-ratio".to_owned(),
        env_or("PETE_NEAT_REHEARSAL_RATIO", "0.20"),
        "--niche-audit-episodes".to_owned(),
        env_or("PETE_NEAT_NICHE_AUDIT_EPISODES", "16"),
        "--models-config".to_owned(),
        env_or("PETE_NEAT_MODELS_CONFIG", "configs/models.toml"),
    ];
    if env_flag("PETE_NEAT_NO_PROMOTE") {
        values.push("--no-promote".to_owned());
    }
    values.extend(extra);
    pete_tools(values.iter().map(String::as_str), &[])
}

fn neat_generation_limit(state: &Path, explicit: Option<u64>, default: u64, increment: u64) -> u64 {
    explicit
        .or_else(|| {
            fs::read_to_string(state)
                .ok()
                .and_then(|json| {
                    json.split("\"generation_in_stage\"")
                        .nth(1)?
                        .split(':')
                        .nth(1)?
                        .trim_start()
                        .split(|character: char| !character.is_ascii_digit())
                        .next()?
                        .parse::<u64>()
                        .ok()
                })
                .map(|completed| completed + increment)
        })
        .unwrap_or(default)
}

fn evolve(clear: Option<&str>, quality: bool) -> Result<()> {
    let prefix = if quality {
        "PETE_NEAT_QUALITY_"
    } else {
        "PETE_NEAT_"
    };
    let generations = env_or(
        &format!("{prefix}GENERATIONS"),
        if quality { "36" } else { "12" },
    );
    let population = env_or(
        &format!("{prefix}POPULATION"),
        if quality { "64" } else { "24" },
    );
    let hidden = env_or(
        &format!("{prefix}HIDDEN_DIM"),
        if quality { "14" } else { "10" },
    );
    let seed = env::var("PETE_NEAT_SEED").unwrap_or_else(|_| {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .to_string()
    });
    let mut args = vec![
        "dream-train".to_owned(),
        "--start-level".to_owned(),
        env_or("PETE_NEAT_START_LEVEL", "motion"),
        "--generations".to_owned(),
        generations,
        "--population".to_owned(),
        population,
        "--seed".to_owned(),
        seed,
        "--hidden-dim".to_owned(),
        hidden,
        "--checkpoint-dir".to_owned(),
        env_or("PETE_NEAT_CHECKPOINT_DIR", "data/models/dream-policy/neat"),
        "--dataset-dir".to_owned(),
        env_or("PETE_NEAT_DATASET_DIR", "datasets/dream-policy/v0/episodes"),
        "--export-dataset".to_owned(),
        env_or("PETE_NEAT_EXPORT_DATASET", "false"),
    ];
    args.push("--detailed-logs".to_owned());
    if clear.is_some_and(|value| value == "true" || value == "--clear" || value == "clear=true") {
        args.push("--clear".to_owned());
    }
    println!(
        "Dream NEAT {}: {} generations, population {}, seed {}",
        if quality { "quality" } else { "fast" },
        args[4],
        args[6],
        args[8]
    );
    pete(["build", "--release", "-p", "pete-tools"])?;
    run_program(ProcessCommand::new("target/release/pete").args(args))
}

fn evolve_infinite(clear: Option<&str>) -> Result<()> {
    let dataset = PathBuf::from(env_or(
        "PETE_NEAT_DATASET_DIR",
        "datasets/dream-policy/v0/episodes",
    ));
    let export_dataset = env_flag("PETE_NEAT_EXPORT_DATASET");
    let checkpoint = PathBuf::from(env_or(
        "PETE_NEAT_CHECKPOINT_DIR",
        "data/models/dream-policy/neat",
    ))
    .join("evolve-best.json");
    let report_root = PathBuf::from(env_or(
        "PETE_EVOLVE_BENCHMARK_ROOT",
        "data/reports/scenario/evolve",
    ));
    let ledger_root = PathBuf::from(env_or(
        "PETE_EVOLVE_BENCHMARK_LEDGER_ROOT",
        "data/ledger/evolve-benchmark",
    ));
    let benchmark_every = env_u64("PETE_EVOLVE_BENCHMARK_EVERY", 10);
    let benchmark_steps = env_u64("PETE_EVOLVE_BENCHMARK_STEPS", 160).to_string();
    let max_runs = env_u64("PETE_EVOLVE_BENCHMARK_MAX_RUNS", 64) as usize;
    let benchmark_age = env_u64("PETE_EVOLVE_BENCHMARK_MAX_AGE_DAYS", 21);
    println!(
        "evolve-infinite: clear={} benchmark_every={benchmark_every} export_dataset={export_dataset}",
        clear.unwrap_or("false")
    );
    for iteration in 1_u64.. {
        println!("iteration #{iteration}");
        evolve(clear, true)?;
        if export_dataset {
            prune_dataset(&dataset)?;
        } else {
            println!("dataset: export disabled; skipping dataset retention");
        }
        if benchmark_every > 0 && iteration % benchmark_every == 0 {
            run_evolution_benchmarks(
                iteration,
                &benchmark_steps,
                &checkpoint,
                &report_root,
                &ledger_root,
            )?;
        }
        prune_directories(&report_root, max_runs, benchmark_age)?;
        prune_directories(&ledger_root, max_runs, benchmark_age)?;
        println!(
            "bench-retain: reports={}, ledgers={}",
            directory_count(&report_root),
            directory_count(&ledger_root)
        );
    }
    Ok(())
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn prune_dataset(root: &Path) -> Result<()> {
    fs::create_dir_all(root)?;
    let before = evolution_dataset_files(root)?;
    let max_age = Duration::from_secs(env_u64("PETE_DATASET_MAX_AGE_DAYS", 10) * 86_400);
    let max_files = env_u64("PETE_DATASET_MAX_FILES", 8_000) as usize;
    let max_bytes = env_u64("PETE_DATASET_MAX_BYTES", 536_870_912);
    let now = SystemTime::now();
    let mut retained = Vec::new();
    for (path, modified, size) in before.iter().cloned() {
        if !max_age.is_zero() && now.duration_since(modified).unwrap_or_default() > max_age {
            fs::remove_file(path)?;
        } else {
            retained.push((path, modified, size));
        }
    }
    retained.sort_by_key(|(_, modified, _)| *modified);
    while (max_files > 0 && retained.len() > max_files)
        || (max_bytes > 0 && retained.iter().map(|(_, _, size)| size).sum::<u64>() > max_bytes)
    {
        let (path, _, _) = retained.remove(0);
        fs::remove_file(path)?;
    }
    let size = retained.iter().map(|(_, _, size)| size).sum::<u64>();
    println!(
        "dataset: files {} -> {}, size={} bytes",
        before.len(),
        retained.len(),
        size
    );
    Ok(())
}

fn evolution_dataset_files(root: &Path) -> Result<Vec<(PathBuf, SystemTime, u64)>> {
    let mut files = Vec::new();
    collect_files(root, &mut files)?;
    Ok(files
        .into_iter()
        .filter_map(|path| {
            let name = path.file_name()?.to_string_lossy();
            if !name.starts_with("level-")
                || !name.contains("-seed-")
                || !name.contains("-genome-")
                || !name.ends_with(".jsonl")
            {
                return None;
            }
            let metadata = path.metadata().ok()?;
            Some((path, metadata.modified().ok()?, metadata.len()))
        })
        .collect())
}

fn collect_files(root: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if !root.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_files(&path, files)?;
        } else if path.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

fn run_evolution_benchmarks(
    iteration: u64,
    steps: &str,
    checkpoint: &Path,
    report_root: &Path,
    ledger_root: &Path,
) -> Result<()> {
    if !checkpoint.is_file() {
        println!("benchmark: skipped (missing {})", checkpoint.display());
        return Ok(());
    }
    let stamp = output("date", &["-u", "+%Y%m%dT%H%M%SZ"])?;
    let name = format!("{stamp}-iter-{iteration}");
    let reports = report_root.join(&name);
    let ledgers = ledger_root.join(name);
    fs::create_dir_all(&reports)?;
    fs::create_dir_all(&ledgers)?;
    for (scenario, seed) in [
        ("obstacle-avoidance", "701"),
        ("corner-trap", "1701"),
        ("column-trap", "2701"),
    ] {
        let ledger = ledgers.join(scenario);
        let report = reports.join(format!("{scenario}.json"));
        let _ = fs::remove_dir_all(&ledger);
        run_program(ProcessCommand::new("target/release/pete").args([
            "sim",
            "--scenario",
            scenario,
            "--steps",
            steps,
            "--tick-delay-ms",
            "0",
            "--seed",
            seed,
            "--ledger",
            path(&ledger)?,
            "--dream-policy-checkpoint",
            path(checkpoint)?,
        ]))?;
        run_program(ProcessCommand::new("target/release/pete").args([
            "virtual-report",
            "--ledger",
            path(&ledger)?,
            "--out",
            path(&report)?,
        ]))?;
    }
    println!("benchmark: reports at {}", reports.display());
    Ok(())
}

fn prune_directories(root: &Path, max_count: usize, max_age_days: u64) -> Result<()> {
    fs::create_dir_all(root)?;
    let now = SystemTime::now();
    let max_age = Duration::from_secs(max_age_days * 86_400);
    let mut directories = fs::read_dir(root)?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .filter_map(|path| Some((path.clone(), path.metadata().ok()?.modified().ok()?)))
        .collect::<Vec<_>>();
    for (path, modified) in &directories {
        if !max_age.is_zero() && now.duration_since(*modified).unwrap_or_default() > max_age {
            fs::remove_dir_all(path)?;
        }
    }
    directories.retain(|(path, _)| path.is_dir());
    directories.sort_by_key(|(_, modified)| *modified);
    if max_count > 0 && directories.len() > max_count {
        let remove = directories.len() - max_count;
        for (path, _) in directories.into_iter().take(remove) {
            fs::remove_dir_all(path)?;
        }
    }
    Ok(())
}

fn directory_count(root: &Path) -> usize {
    fs::read_dir(root)
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| entry.path().is_dir())
                .count()
        })
        .unwrap_or(0)
}

fn rehearse_models() -> Result<()> {
    for args in [
        ["sim", "--steps", "200", "--ledger", "data/ledger/sim1"].as_slice(),
        [
            "train",
            "behavior",
            "danger",
            "--ledger",
            "data/ledger/sim1",
        ]
        .as_slice(),
        [
            "train",
            "behavior",
            "charge",
            "--ledger",
            "data/ledger/sim1",
        ]
        .as_slice(),
        [
            "train",
            "behavior",
            "future",
            "--ledger",
            "data/ledger/sim1",
        ]
        .as_slice(),
        [
            "evaluate",
            "behavior",
            "danger",
            "--ledger",
            "data/ledger/sim1",
        ]
        .as_slice(),
        ["model-status"].as_slice(),
        [
            "sim",
            "--steps",
            "200",
            "--danger-checkpoint",
            "data/models/danger_v0",
            "--danger-mode",
            "shadow-infer",
        ]
        .as_slice(),
        [
            "robot",
            "--mode",
            "read-only",
            "--cockpit",
            "sim",
            "--steps",
            "20",
            "--capture",
            "data/captures/mock-readonly",
        ]
        .as_slice(),
        ["replay-capture", "--capture", "data/captures/mock-readonly"].as_slice(),
    ] {
        pete_tools(args.iter().copied(), &[])?;
    }
    Ok(())
}

fn eval_scenario_smoke() -> Result<()> {
    for args in [
        [
            "empty-room",
            "2",
            "10",
            "data/reports/scenario/empty-smoke.json",
        ],
        [
            "obstacle-avoidance",
            "2",
            "10",
            "data/reports/scenario/obstacle-smoke.json",
        ],
        [
            "corner-trap",
            "1",
            "40",
            "data/reports/scenario/corner-trap-smoke.json",
        ],
        [
            "charger-seeking",
            "2",
            "10",
            "data/reports/scenario/charge-smoke.json",
        ],
    ] {
        pete_tools(
            [
                "eval-scenario",
                "--scenario",
                args[0],
                "--episodes",
                args[1],
                "--steps",
                args[2],
                "--out",
                args[3],
            ],
            &[],
        )?;
    }
    Ok(())
}

fn pico_bootsel_mount(umount: bool, kernel_name: &str) -> Result<()> {
    let user = env_or(
        "PICO_BOOTSEL_USER",
        &env_or("SUDO_USER", &env_or("USER", "")),
    );
    let uid = output("id", &["-u", &user]).unwrap_or_else(|_| "0".to_owned());
    let group = env_or("PICO_BOOTSEL_GROUP", &user);
    let gid = output("getent", &["group", &group])
        .ok()
        .and_then(|entry| entry.split(':').nth(2).map(str::to_owned))
        .unwrap_or_else(|| uid.clone());
    let mount = env::var("PICO_BOOTSEL_MOUNT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env_or("PICO_BOOTSEL_MOUNT_BASE", "/media")).join(if uid == "0" {
                "RPI-RP2".to_owned()
            } else {
                format!("{user}/RPI-RP2")
            })
        });
    if umount {
        if is_mountpoint(&mount) {
            run_program(ProcessCommand::new("umount").arg(&mount))?;
        }
        let _ = fs::remove_dir(&mount);
        return Ok(());
    }
    let device = format!("/dev/{kernel_name}");
    #[cfg(unix)]
    let is_block_device = fs::metadata(&device)
        .map(|metadata| metadata.file_type().is_block_device())
        .unwrap_or(false);
    #[cfg(not(unix))]
    let is_block_device = Path::new(&device).exists();
    if !is_block_device {
        return fail(format!("BOOTSEL block device not found: {device}"));
    }
    fs::create_dir_all(&mount)?;
    if !is_mountpoint(&mount) {
        run_program(ProcessCommand::new("mount").args([
            "-t",
            "vfat",
            "-o",
            &format!("uid={uid},gid={gid},umask=022,noatime,flush"),
            &device,
            path(&mount)?,
        ]))?;
    }
    #[cfg(unix)]
    fs::set_permissions(&mount, fs::Permissions::from_mode(0o755))?;
    println!(
        "Mounted RPI-RP2 at {} for uid={uid} gid={gid}",
        mount.display()
    );
    Ok(())
}

fn is_mountpoint(path: &Path) -> bool {
    ProcessCommand::new("mountpoint")
        .args(["-q", path.to_str().unwrap_or("")])
        .status()
        .is_ok_and(|status| status.success())
}

fn codex_sync() -> Result<()> {
    if !Path::new(".git").is_dir() {
        return fail("codex-sync: no git repository in the current directory");
    }
    let status = output("git", &["status", "--short", "--branch"])?;
    if status.lines().count() <= 1 {
        run_program(ProcessCommand::new("git").args(["pull", "--ff-only"]))?;
        if status.contains("[ahead ") {
            run_program(ProcessCommand::new("git").arg("push"))?;
        }
        return Ok(());
    }
    let prompt = "Inspect `git diff`, `git diff --cached`, and recent commits. Treat already staged changes as candidate work even when another agent or person staged them: include every ready staged or unstaged semantic change in CHANGELOG.md under Unreleased without removing releases. Summarize and classify each change as ready or ongoing, commit only ready semantic groups, then git pull --ff-only and git push. Do not run CI or create extra files; leave uncertain work uncommitted.";
    let summary_path =
        env::temp_dir().join(format!("netherwick-codex-sync-{}.md", std::process::id()));
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg}").expect("valid spinner template"),
    );
    spinner.set_message("Codex is reviewing and syncing the worktree");
    spinner.enable_steady_tick(Duration::from_millis(120));
    let terminal_title = TerminalTitleSpinner::start("Codex is reviewing and syncing the worktree");
    let mut child = ProcessCommand::new("codex")
        .args([
            "--ask-for-approval",
            "never",
            "exec",
            "--sandbox",
            "danger-full-access",
            "--ephemeral",
            "--output-last-message",
            path(&summary_path)?,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    child
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("failed to open codex stdin"))?
        .write_all(prompt.as_bytes())?;
    let status = child.wait()?;
    spinner.finish_and_clear();
    drop(terminal_title);
    let summary = fs::read_to_string(&summary_path).unwrap_or_default();
    let _ = fs::remove_file(&summary_path);
    if !status.success() {
        return fail(format!("codex-sync failed with {status}"));
    }
    if !summary.trim().is_empty() {
        println!("{summary}");
    }
    Ok(())
}

struct TerminalTitleSpinner {
    running: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl TerminalTitleSpinner {
    fn start(message: &str) -> Option<Self> {
        if !io::stderr().is_terminal() {
            return None;
        }

        let running = Arc::new(AtomicBool::new(true));
        let spinner_running = Arc::clone(&running);
        let message = terminal_title_text(message);
        let thread = thread::spawn(move || {
            const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let mut frame = 0;
            while spinner_running.load(Ordering::Relaxed) {
                let mut stderr = io::stderr().lock();
                let _ = write!(stderr, "\x1b]0;{} {}\x07", FRAMES[frame], message);
                let _ = stderr.flush();
                frame = (frame + 1) % FRAMES.len();
                thread::sleep(Duration::from_millis(120));
            }
        });

        Some(Self {
            running,
            thread: Some(thread),
        })
    }
}

impl Drop for TerminalTitleSpinner {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        let mut stderr = io::stderr().lock();
        let _ = write!(stderr, "\x1b]0;Netherwick\x07");
        let _ = stderr.flush();
    }
}

fn terminal_title_text(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_control())
        .collect()
}

fn output(program: &str, args: &[&str]) -> Result<String> {
    let result = ProcessCommand::new(program).args(args).output()?;
    if result.status.success() {
        Ok(String::from_utf8_lossy(&result.stdout).trim().to_owned())
    } else {
        fail(format!("{program} failed with {}", result.status))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        boot_identity_mismatch, bootsel_host, evolution_dataset_files, neat_generation_limit,
        prefixed_value, terminal_title_text, Cli,
    };
    use clap::Parser;
    use std::fs;

    #[test]
    fn continuation_limit_extends_completed_generation() {
        let state = std::env::temp_dir().join(format!("xtask-neat-limit-{}", std::process::id()));
        fs::write(&state, r#"{"generation_in_stage":243}"#).unwrap();
        assert_eq!(neat_generation_limit(&state, None, 120, 120), 363);
        fs::write(&state, r#"{"generation_in_stage":363}"#).unwrap();
        assert_eq!(neat_generation_limit(&state, None, 120, 120), 483);
        assert_eq!(neat_generation_limit(&state, Some(77), 120, 120), 77);
        let _ = fs::remove_file(state);
    }

    #[test]
    fn terminal_title_text_omits_terminal_control_characters() {
        assert_eq!(
            terminal_title_text("sync\u{1b}]0;spoof\u{7}"),
            "sync]0;spoof"
        );
    }

    #[test]
    fn hardware_identity_output_is_parsed_conservatively() {
        let output = "brainstem identity: pete-17\nbrainstem boot: boot-4\n";
        assert_eq!(
            prefixed_value(output, "brainstem identity: ").as_deref(),
            Some("pete-17")
        );
        assert_eq!(
            boot_identity_mismatch(
                "Error: brainstem boot identity mismatch: expected boot-3 received boot-4"
            )
            .as_deref(),
            Some("boot-4")
        );
        assert_eq!(bootsel_host("http://192.168.4.1/command"), "192.168.4.1:80");
    }

    #[test]
    fn dataset_retention_only_claims_evolution_episodes() {
        let root = std::env::temp_dir().join(format!("xtask-dataset-files-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("nested")).unwrap();
        fs::write(root.join("nested/level-1-seed-2-genome-3.jsonl"), "episode").unwrap();
        fs::write(root.join("keep-me.jsonl"), "other").unwrap();
        let files = evolution_dataset_files(&root).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].0.ends_with("level-1-seed-2-genome-3.jsonl"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn public_alias_commands_remain_parseable() {
        for command in [
            "check",
            "flash",
            "possess",
            "go",
            "train",
            "evolve-infinite",
            "codex-sync",
        ] {
            Cli::try_parse_from(["xtask", command]).unwrap();
        }
    }
}
