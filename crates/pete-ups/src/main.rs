use std::{fs, path::PathBuf, process::Command as ProcessCommand, thread, time::Duration};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use gpio_cdev::{Chip, LineRequestFlags};
use i2cdev::{core::I2CDevice, linux::LinuxI2CDevice};
use pete_ups::{
    decode_max1704x, UpsTelemetry, MAX1704X_SOC_REGISTER, MAX1704X_VCELL_REGISTER,
    X1202_I2C_ADDRESS,
};

const AC_PRESENT_GPIO: u32 = 6;
const CHARGE_ENABLE_GPIO: u32 = 16;

#[derive(Parser)]
#[command(about = "Geekworm X1202 telemetry and control for Pete's motherbrain")]
struct Args {
    #[arg(long, default_value = "/dev/i2c-1")]
    i2c: PathBuf,
    #[arg(long, default_value = "/dev/gpiochip0")]
    gpiochip: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Status,
    Monitor {
        #[arg(long, default_value = "/run/pete/ups.json")]
        state_file: PathBuf,
        #[arg(long, default_value_t = 2)]
        interval_seconds: u64,
        /// Gracefully power off after this many consecutive critical samples
        /// while external power is absent. Zero disables automatic shutdown.
        #[arg(long, default_value_t = 0)]
        critical_samples: u32,
        #[arg(long, default_value_t = 10.0)]
        critical_percent: f32,
    },
    Charge {
        #[arg(value_enum)]
        state: ChargeState,
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
        Command::Status => println!("{}", serde_json::to_string_pretty(&read_telemetry(&args)?)?),
        Command::Monitor {
            ref state_file,
            interval_seconds,
            critical_samples,
            critical_percent,
        } => {
            let mut consecutive_critical = 0u32;
            loop {
                match read_telemetry(&args) {
                    Ok(value) => {
                        write_state(state_file, &value)?;
                        if !value.external_power_present
                            && value.battery_percent <= critical_percent
                        {
                            consecutive_critical = consecutive_critical.saturating_add(1);
                        } else {
                            consecutive_critical = 0;
                        }
                        if critical_samples > 0 && consecutive_critical >= critical_samples {
                            ProcessCommand::new("systemctl")
                                .arg("poweroff")
                                .status()
                                .context("request graceful OS shutdown")?;
                            return Ok(());
                        }
                    }
                    Err(error) => {
                        consecutive_critical = 0;
                        eprintln!("x1202 telemetry unavailable: {error:#}");
                    }
                }
                thread::sleep(Duration::from_secs(interval_seconds.max(1)));
            }
        }
        Command::Charge { state } => set_charging(&args, matches!(state, ChargeState::Enable))?,
    }
    Ok(())
}

fn read_telemetry(args: &Args) -> Result<UpsTelemetry> {
    let mut gauge = LinuxI2CDevice::new(&args.i2c, X1202_I2C_ADDRESS)
        .with_context(|| format!("open X1202 fuel gauge on {}", args.i2c.display()))?;
    let vcell = gauge
        .smbus_read_word_data(MAX1704X_VCELL_REGISTER)?
        .swap_bytes()
        .to_be_bytes();
    let soc = gauge
        .smbus_read_word_data(MAX1704X_SOC_REGISTER)?
        .swap_bytes()
        .to_be_bytes();
    let (battery_voltage_v, battery_percent) = decode_max1704x(vcell, soc);
    let mut chip = Chip::new(&args.gpiochip)?;
    let external_power_present = chip
        .get_line(AC_PRESENT_GPIO)?
        .request(LineRequestFlags::INPUT, 0, "pete-ups-ac")?
        .get_value()?
        != 0;
    Ok(UpsTelemetry {
        hardware_profile: "geekworm-x1202-max17040g-r1",
        battery_voltage_v,
        battery_percent,
        external_power_present,
        charging_enabled: None,
    })
}

fn set_charging(args: &Args, enabled: bool) -> Result<()> {
    let mut chip = Chip::new(&args.gpiochip)?;
    let line = chip.get_line(CHARGE_ENABLE_GPIO)?.request(
        LineRequestFlags::OUTPUT,
        (!enabled) as u8,
        "pete-ups-charge",
    )?;
    line.set_value((!enabled) as u8)?;
    println!("charging {}", if enabled { "enabled" } else { "disabled" });
    Ok(())
}

fn write_state(path: &PathBuf, value: &UpsTelemetry) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}
