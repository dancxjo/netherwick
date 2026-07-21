use std::{
    fs,
    io::{self, Read, Write},
    os::unix::{
        fs::{FileTypeExt, PermissionsExt},
        net::{UnixListener, UnixStream},
    },
    path::{Path, PathBuf},
    process::Command as ProcessCommand,
    thread,
    time::{Duration, Instant},
};

use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, Subcommand};
use gpio_cdev::{Chip, LineHandle, LineRequestFlags};
use i2cdev::{core::I2CDevice, linux::LinuxI2CDevice};
use pete_ups::{
    decode_max1704x, input_is_active, output_level, HardwareProfile, UpsTelemetry,
    MAX1704X_SOC_REGISTER, MAX1704X_VCELL_REGISTER,
};

const DEFAULT_PROFILE: &str = "/etc/pete/x1202.toml";
const DEFAULT_STATE_FILE: &str = "/run/pete/ups.json";
const DEFAULT_CONTROL_SOCKET: &str = "/run/pete/ups-control.sock";

#[derive(Parser)]
#[command(about = "Geekworm X1202 telemetry and control for Pete's motherbrain")]
struct Args {
    /// Explicit X1202 hardware profile. Verify its gpiochip offsets with gpioinfo.
    #[arg(long, default_value = DEFAULT_PROFILE)]
    profile: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Status,
    Monitor {
        #[arg(long, default_value = DEFAULT_STATE_FILE)]
        state_file: PathBuf,
        #[arg(long, default_value = DEFAULT_CONTROL_SOCKET)]
        control_socket: PathBuf,
        #[arg(long, default_value_t = 2)]
        interval_seconds: u64,
        /// Gracefully power off after this many consecutive critical samples
        /// while external power is absent. Zero disables automatic shutdown.
        #[arg(long, default_value_t = 0)]
        critical_samples: u32,
        #[arg(long, default_value_t = 10.0)]
        critical_percent: f32,
    },
    /// Ask the running monitor to change its daemon-owned charging output.
    Charge {
        #[arg(value_enum)]
        state: ChargeState,
        #[arg(long, default_value = DEFAULT_CONTROL_SOCKET)]
        control_socket: PathBuf,
    },
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum ChargeState {
    Enable,
    Disable,
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Status => {
            let profile = load_profile(&args.profile)?;
            let ac_present = request_input_line(&profile)?;
            let telemetry = read_telemetry(&profile, &ac_present, None)?;
            println!("{}", serde_json::to_string_pretty(&telemetry)?);
        }
        Command::Monitor {
            state_file,
            control_socket,
            interval_seconds,
            critical_samples,
            critical_percent,
        } => run_monitor(
            load_profile(&args.profile)?,
            &state_file,
            &control_socket,
            interval_seconds,
            critical_samples,
            critical_percent,
        )?,
        Command::Charge {
            state,
            control_socket,
        } => request_charging(&control_socket, matches!(state, ChargeState::Enable))?,
    }
    Ok(())
}

fn load_profile(path: &Path) -> Result<HardwareProfile> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("read X1202 hardware profile {}", path.display()))?;
    let profile: HardwareProfile = toml::from_str(&text)
        .with_context(|| format!("parse X1202 hardware profile {}", path.display()))?;
    profile
        .validate()
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("validate X1202 hardware profile {}", path.display()))?;
    Ok(profile)
}

fn run_monitor(
    profile: HardwareProfile,
    state_file: &Path,
    control_socket: &Path,
    interval_seconds: u64,
    critical_samples: u32,
    critical_percent: f32,
) -> Result<()> {
    let ac_present = request_input_line(&profile)?;
    let charge_enable = request_charge_line(&profile, profile.charging_enabled_on_start)?;
    let (listener, _socket_guard) = bind_control_socket(control_socket)?;
    let mut charging_enabled = profile.charging_enabled_on_start;
    let mut consecutive_critical = 0u32;
    let interval = Duration::from_secs(interval_seconds.max(1));
    let mut next_sample = Instant::now();

    loop {
        poll_charge_requests(&listener, &charge_enable, &profile, &mut charging_enabled)?;

        let now = Instant::now();
        if now >= next_sample {
            match read_telemetry(&profile, &ac_present, Some(charging_enabled)) {
                Ok(value) => {
                    write_state(state_file, &value)?;
                    if !value.external_power_present && value.battery_percent <= critical_percent {
                        consecutive_critical = consecutive_critical.saturating_add(1);
                    } else {
                        consecutive_critical = 0;
                    }
                    if critical_samples > 0 && consecutive_critical >= critical_samples {
                        request_poweroff()?;
                        return Ok(());
                    }
                }
                Err(error) => {
                    consecutive_critical = 0;
                    eprintln!("x1202 telemetry unavailable: {error:#}");
                }
            }
            next_sample = now + interval;
        }
        thread::sleep(Duration::from_millis(50));
    }
}

fn request_input_line(profile: &HardwareProfile) -> Result<LineHandle> {
    let mut chip = Chip::new(&profile.gpiochip)
        .with_context(|| format!("open GPIO chip {}", profile.gpiochip.display()))?;
    chip.get_line(profile.external_power_offset)?
        .request(LineRequestFlags::INPUT, 0, "pete-ups-ac")
        .context("request external-power GPIO as input")
}

fn request_charge_line(profile: &HardwareProfile, enabled: bool) -> Result<LineHandle> {
    let mut chip = Chip::new(&profile.gpiochip)
        .with_context(|| format!("open GPIO chip {}", profile.gpiochip.display()))?;
    let level = output_level(enabled, profile.charge_enable_active_low);
    chip.get_line(profile.charge_enable_offset)?
        .request(LineRequestFlags::OUTPUT, level, "pete-ups-charge")
        .context("request charge-enable GPIO as persistent output")
}

fn read_telemetry(
    profile: &HardwareProfile,
    ac_present: &LineHandle,
    charging_enabled: Option<bool>,
) -> Result<UpsTelemetry> {
    let mut gauge =
        LinuxI2CDevice::new(&profile.i2c_device, profile.i2c_address).with_context(|| {
            format!(
                "open X1202 fuel gauge at 0x{:02x} on {}",
                profile.i2c_address,
                profile.i2c_device.display()
            )
        })?;
    let vcell = gauge
        .smbus_read_word_data(MAX1704X_VCELL_REGISTER)?
        .swap_bytes()
        .to_be_bytes();
    let soc = gauge
        .smbus_read_word_data(MAX1704X_SOC_REGISTER)?
        .swap_bytes()
        .to_be_bytes();
    let (battery_voltage_v, battery_percent) = decode_max1704x(vcell, soc);
    let external_power_present =
        input_is_active(ac_present.get_value()?, profile.external_power_active_high);
    let sampled_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    Ok(UpsTelemetry {
        hardware_profile: profile.name.clone(),
        battery_voltage_v,
        battery_percent,
        external_power_present,
        charging_enabled,
        sampled_at_ms,
        fuel_gauge_observed_at_ms: sampled_at_ms,
        external_power_observed_at_ms: sampled_at_ms,
        charging_command_observed_at_ms: sampled_at_ms,
        battery_current_a: None,
        battery_current_observable: false,
        confidence: 1.0,
    })
}

fn bind_control_socket(path: &Path) -> Result<(UnixListener, SocketGuard)> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_socket() => fs::remove_file(path)?,
        Ok(_) => bail!("refusing to replace non-socket path {}", path.display()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let listener = UnixListener::bind(path)
        .with_context(|| format!("bind charging-control socket {}", path.display()))?;
    listener.set_nonblocking(true)?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o660))?;
    Ok((listener, SocketGuard(path.to_owned())))
}

fn poll_charge_requests(
    listener: &UnixListener,
    charge_enable: &LineHandle,
    profile: &HardwareProfile,
    charging_enabled: &mut bool,
) -> Result<()> {
    loop {
        let (mut stream, _) = match listener.accept() {
            Ok(connection) => connection,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(()),
            Err(error) => return Err(error.into()),
        };
        if let Err(error) =
            handle_charge_request(&mut stream, charge_enable, profile, charging_enabled)
        {
            let _ = writeln!(stream, "ERR {error:#}");
            eprintln!("charging-control request failed: {error:#}");
        }
    }
}

fn handle_charge_request(
    stream: &mut UnixStream,
    charge_enable: &LineHandle,
    profile: &HardwareProfile,
    charging_enabled: &mut bool,
) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_millis(250)))?;
    let mut request = String::new();
    Read::by_ref(stream).take(64).read_to_string(&mut request)?;
    let requested = match request.trim() {
        "enable" => Some(true),
        "disable" => Some(false),
        _ => None,
    };
    if let Some(enabled) = requested {
        let level = output_level(enabled, profile.charge_enable_active_low);
        match charge_enable.set_value(level) {
            Ok(()) => {
                *charging_enabled = enabled;
                writeln!(
                    &mut *stream,
                    "OK charging {}",
                    if enabled { "enabled" } else { "disabled" }
                )?;
            }
            Err(error) => {
                writeln!(&mut *stream, "ERR failed to set charging output: {error}")?;
            }
        }
    } else {
        writeln!(&mut *stream, "ERR expected enable or disable")?;
    }
    Ok(())
}

fn request_charging(socket: &Path, enabled: bool) -> Result<()> {
    let mut stream = UnixStream::connect(socket)
        .with_context(|| format!("connect to pete-ups monitor at {}", socket.display()))?;
    writeln!(stream, "{}", if enabled { "enable" } else { "disable" })?;
    stream.shutdown(std::net::Shutdown::Write)?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    let response = response.trim();
    ensure!(response.starts_with("OK "), "{response}");
    println!("{}", response.trim_start_matches("OK "));
    Ok(())
}

fn request_poweroff() -> Result<()> {
    let status = ProcessCommand::new("systemctl")
        .arg("poweroff")
        .status()
        .context("request graceful OS shutdown")?;
    ensure!(status.success(), "systemctl poweroff failed with {status}");
    Ok(())
}

fn write_state(path: &Path, value: &UpsTelemetry) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

struct SocketGuard(PathBuf);

impl Drop for SocketGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
