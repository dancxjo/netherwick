use serde::{Deserialize, Serialize};

pub const X1202_I2C_ADDRESS: u16 = 0x36;
pub const MAX1704X_VCELL_REGISTER: u8 = 0x02;
pub const MAX1704X_SOC_REGISTER: u8 = 0x04;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct UpsTelemetry {
    pub hardware_profile: &'static str,
    pub battery_voltage_v: f32,
    pub battery_percent: f32,
    pub external_power_present: bool,
    /// `None` unless this process owns GPIO16; reading it by changing the line
    /// to input would itself alter the charging control circuit.
    pub charging_enabled: Option<bool>,
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
}
