use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub const X1202_I2C_ADDRESS: u16 = 0x36;
pub const MAX1704X_VCELL_REGISTER: u8 = 0x02;
pub const MAX1704X_SOC_REGISTER: u8 = 0x04;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UpsTelemetry {
    pub hardware_profile: String,
    pub battery_voltage_v: f32,
    pub battery_percent: f32,
    pub external_power_present: bool,
    /// `Some` when published by the monitor that owns the charging GPIO.
    pub charging_enabled: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub name: String,
    pub i2c_device: PathBuf,
    pub i2c_address: u16,
    pub gpiochip: PathBuf,
    pub external_power_offset: u32,
    pub external_power_active_high: bool,
    pub charge_enable_offset: u32,
    pub charge_enable_active_low: bool,
    pub charging_enabled_on_start: bool,
}

impl HardwareProfile {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.name.trim().is_empty() {
            return Err("hardware profile name must not be empty");
        }
        if self.i2c_address > 0x7f {
            return Err("I2C address must be a 7-bit address");
        }
        if self.external_power_offset == self.charge_enable_offset {
            return Err("external-power and charge-enable GPIO offsets must differ");
        }
        Ok(())
    }
}

pub fn input_is_active(level: u8, active_high: bool) -> bool {
    (level != 0) == active_high
}

pub fn output_level(active: bool, active_low: bool) -> u8 {
    u8::from(active != active_low)
}

/// MAX1704x words arrive MSB first. VCELL has a 1.25 mV/LSB scale after
/// discarding its low nibble; SOC is an 8.8 fixed-point percentage.
pub fn decode_max1704x(vcell: [u8; 2], soc: [u8; 2]) -> (f32, f32) {
    let raw_voltage = u16::from_be_bytes(vcell) >> 4;
    let voltage = raw_voltage as f32 * 0.00125;
    let percent = (soc[0] as f32 + soc[1] as f32 / 256.0).clamp(0.0, 100.0);
    (voltage, percent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_datasheet_register_layout() {
        let (voltage, percent) = decode_max1704x([0xbc, 0x00], [0x4b, 0x80]);
        assert!((voltage - 3.76).abs() < 0.001);
        assert!((percent - 75.5).abs() < 0.001);
    }

    #[test]
    fn clamps_invalid_soc() {
        let (_, percent) = decode_max1704x([0, 0], [0xff, 0xff]);
        assert_eq!(percent, 100.0);
    }

    #[test]
    fn applies_configured_gpio_polarity() {
        assert!(input_is_active(1, true));
        assert!(input_is_active(0, false));
        assert_eq!(output_level(true, true), 0);
        assert_eq!(output_level(false, true), 1);
    }

    #[test]
    fn parses_and_validates_hardware_profile() {
        let profile: HardwareProfile = toml::from_str(
            r#"
name = "x1202-test"
i2c_device = "/dev/i2c-1"
i2c_address = 0x36
gpiochip = "/dev/gpiochip4"
external_power_offset = 6
external_power_active_high = true
charge_enable_offset = 16
charge_enable_active_low = true
charging_enabled_on_start = true
"#,
        )
        .unwrap();
        assert_eq!(profile.i2c_address, 0x36);
        assert_eq!(profile.gpiochip, PathBuf::from("/dev/gpiochip4"));
        assert_eq!(profile.validate(), Ok(()));
    }
}
