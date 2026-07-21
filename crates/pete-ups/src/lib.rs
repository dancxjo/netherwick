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
    #[serde(default)]
    pub sampled_at_ms: u64,
    #[serde(default)]
    pub fuel_gauge_observed_at_ms: u64,
    #[serde(default)]
    pub external_power_observed_at_ms: u64,
    #[serde(default)]
    pub charging_command_observed_at_ms: u64,
    /// MAX17040G does not expose battery current; this must remain `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub battery_current_a: Option<f32>,
    #[serde(default)]
    pub battery_current_observable: bool,
    #[serde(default)]
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CreatePowerEvidence {
    pub observed_at_ms: u64,
    pub stopped: bool,
    pub docked: bool,
    pub home_base_contact: bool,
    pub dock_ir_visible: bool,
    /// Raw Create OI charging-state code. Zero is not charging.
    pub charging_state: Option<u8>,
    pub motion_authority_active: bool,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PowerEvidencePolicy {
    pub maximum_evidence_age_ms: u64,
    pub minimum_confidence: f32,
    pub minimum_battery_percent: f32,
    pub minimum_trend_window_ms: u64,
    pub minimum_voltage_rise_v: f32,
    pub minimum_soc_rise_percent: f32,
}

impl Default for PowerEvidencePolicy {
    fn default() -> Self {
        Self {
            maximum_evidence_age_ms: 2_000,
            minimum_confidence: 0.8,
            minimum_battery_percent: 20.0,
            minimum_trend_window_ms: 10_000,
            minimum_voltage_rise_v: 0.005,
            minimum_soc_rise_percent: 0.02,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChargingInference {
    #[default]
    Unobservable,
    NotCharging,
    LikelyCharging,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PowerEvidenceAge {
    pub fuel_gauge_ms: Option<u64>,
    pub external_power_ms: Option<u64>,
    pub charging_command_ms: Option<u64>,
    pub create_ms: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConsolidationPowerAssessment {
    pub evaluated_at_ms: u64,
    pub external_power_present: Option<bool>,
    pub charging_enabled: Option<bool>,
    pub battery_voltage_v: Option<f32>,
    pub battery_percent: Option<f32>,
    pub battery_current_a: Option<f32>,
    pub battery_current_observable: bool,
    pub battery_charging: ChargingInference,
    pub battery_charging_confidence: f32,
    pub create_stopped: Option<bool>,
    pub create_docked: Option<bool>,
    pub home_base_contact: Option<bool>,
    pub dock_ir_visible: Option<bool>,
    pub create_charging: Option<bool>,
    pub create_charging_state: Option<u8>,
    pub motion_authority_active: Option<bool>,
    pub ages: PowerEvidenceAge,
    pub consolidation_ready: bool,
    pub action: String,
    pub reasons: Vec<String>,
    pub evidence_sources: Vec<String>,
}

pub fn assess_consolidation_power(
    now_ms: u64,
    current_ups: Option<&UpsTelemetry>,
    previous_ups: Option<&UpsTelemetry>,
    create: Option<&CreatePowerEvidence>,
    policy: &PowerEvidencePolicy,
) -> ConsolidationPowerAssessment {
    let mut assessment = ConsolidationPowerAssessment {
        evaluated_at_ms: now_ms,
        action: "pause".to_string(),
        ..ConsolidationPowerAssessment::default()
    };
    if let Some(ups) = current_ups {
        assessment.ages.fuel_gauge_ms = age(now_ms, ups.fuel_gauge_observed_at_ms);
        assessment.ages.external_power_ms = age(now_ms, ups.external_power_observed_at_ms);
        assessment.ages.charging_command_ms = age(now_ms, ups.charging_command_observed_at_ms);
        assessment.battery_current_a = ups.battery_current_a;
        assessment.battery_current_observable = ups.battery_current_observable;
        let confidence_ok = ups.confidence >= policy.minimum_confidence;
        if confidence_ok
            && evidence_fresh(
                now_ms,
                ups.external_power_observed_at_ms,
                policy.maximum_evidence_age_ms,
            )
        {
            assessment.external_power_present = Some(ups.external_power_present);
        } else {
            assessment
                .reasons
                .push("GPIO6 external-power evidence is stale or low-confidence".to_string());
        }
        if confidence_ok
            && evidence_fresh(
                now_ms,
                ups.charging_command_observed_at_ms,
                policy.maximum_evidence_age_ms,
            )
        {
            assessment.charging_enabled = ups.charging_enabled;
        } else {
            assessment
                .reasons
                .push("GPIO16 charging-command evidence is stale or low-confidence".to_string());
        }
        if confidence_ok
            && evidence_fresh(
                now_ms,
                ups.fuel_gauge_observed_at_ms,
                policy.maximum_evidence_age_ms,
            )
        {
            assessment.battery_voltage_v = Some(ups.battery_voltage_v);
            assessment.battery_percent = Some(ups.battery_percent);
        } else {
            assessment
                .reasons
                .push("MAX17040G fuel-gauge evidence is stale or low-confidence".to_string());
        }
        if confidence_ok {
            assessment
                .evidence_sources
                .push("x1202_pogo_max17040g_gpio6_gpio16".to_string());
        }
    } else {
        assessment
            .reasons
            .push("X1202 evidence is unavailable".to_string());
    }

    let create_fresh = create.is_some_and(|value| {
        evidence_fresh(now_ms, value.observed_at_ms, policy.maximum_evidence_age_ms)
            && value.confidence >= policy.minimum_confidence
    });
    if let Some(create) = create {
        assessment.ages.create_ms = age(now_ms, create.observed_at_ms);
        if create_fresh {
            assessment.create_stopped = Some(create.stopped);
            assessment.create_docked = Some(create.docked);
            assessment.home_base_contact = Some(create.home_base_contact);
            assessment.dock_ir_visible = Some(create.dock_ir_visible);
            assessment.create_charging_state = create.charging_state;
            assessment.create_charging = create.charging_state.map(create_state_is_charging);
            assessment.motion_authority_active = Some(create.motion_authority_active);
            assessment
                .evidence_sources
                .push("create_oi_and_dock_observation".to_string());
        } else {
            assessment
                .reasons
                .push("Create dock/charging evidence is stale or low-confidence".to_string());
        }
    } else {
        assessment
            .reasons
            .push("Create dock/charging evidence is unavailable".to_string());
    }

    let trend = charging_trend(previous_ups, current_ups, policy);
    assessment.battery_charging = if assessment.external_power_present == Some(false)
        || assessment.charging_enabled == Some(false)
    {
        ChargingInference::NotCharging
    } else if assessment.external_power_present == Some(true)
        && assessment.charging_enabled == Some(true)
        && trend.is_some_and(|positive| positive)
    {
        ChargingInference::LikelyCharging
    } else {
        ChargingInference::Unobservable
    };
    assessment.battery_charging_confidence = match assessment.battery_charging {
        ChargingInference::LikelyCharging => 0.75,
        ChargingInference::NotCharging => 0.95,
        ChargingInference::Unobservable => 0.0,
    };
    if !assessment.battery_current_observable {
        assessment
            .reasons
            .push("battery current is unavailable on MAX17040G; charging is inferred".to_string());
    }

    require_gate(
        &mut assessment.reasons,
        assessment.create_stopped == Some(true),
        "Create is not freshly confirmed stopped",
    );
    require_gate(
        &mut assessment.reasons,
        assessment.create_docked == Some(true),
        "Create is not freshly confirmed docked",
    );
    require_gate(
        &mut assessment.reasons,
        assessment.create_charging == Some(true),
        "fresh Create OI charging state is absent",
    );
    require_gate(
        &mut assessment.reasons,
        assessment.external_power_present == Some(true),
        "fresh GPIO6 external-power evidence is absent",
    );
    require_gate(
        &mut assessment.reasons,
        assessment.charging_enabled == Some(true),
        "GPIO16 charging-enable command is not freshly enabled",
    );
    require_gate(
        &mut assessment.reasons,
        assessment.motion_authority_active == Some(false),
        "motion authority remains active",
    );
    require_gate(
        &mut assessment.reasons,
        assessment
            .battery_percent
            .is_some_and(|value| value >= policy.minimum_battery_percent),
        "battery health policy is not satisfied",
    );
    let blocking_reasons = assessment
        .reasons
        .iter()
        .filter(|reason| !reason.contains("battery current is unavailable"))
        .count();
    assessment.consolidation_ready = blocking_reasons == 0;
    assessment.action = if assessment.consolidation_ready {
        "proceed".to_string()
    } else if assessment.external_power_present == Some(false) {
        "pause_external_power_lost".to_string()
    } else {
        "pause".to_string()
    };
    assessment
}

fn charging_trend(
    previous: Option<&UpsTelemetry>,
    current: Option<&UpsTelemetry>,
    policy: &PowerEvidencePolicy,
) -> Option<bool> {
    let (previous, current) = previous.zip(current)?;
    let elapsed = current.sampled_at_ms.checked_sub(previous.sampled_at_ms)?;
    if elapsed < policy.minimum_trend_window_ms {
        return None;
    }
    Some(
        current.battery_voltage_v - previous.battery_voltage_v >= policy.minimum_voltage_rise_v
            || current.battery_percent - previous.battery_percent
                >= policy.minimum_soc_rise_percent,
    )
}

fn create_state_is_charging(state: u8) -> bool {
    matches!(state, 1..=5)
}

fn evidence_fresh(now_ms: u64, observed_at_ms: u64, maximum_age_ms: u64) -> bool {
    observed_at_ms > 0 && observed_at_ms <= now_ms && now_ms - observed_at_ms <= maximum_age_ms
}

fn age(now_ms: u64, observed_at_ms: u64) -> Option<u64> {
    (observed_at_ms > 0 && observed_at_ms <= now_ms).then_some(now_ms - observed_at_ms)
}

fn require_gate(reasons: &mut Vec<String>, passed: bool, reason: &str) {
    if !passed && !reasons.iter().any(|existing| existing == reason) {
        reasons.push(reason.to_string());
    }
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

    fn ups(at_ms: u64, external: bool, enabled: bool, voltage: f32, soc: f32) -> UpsTelemetry {
        UpsTelemetry {
            hardware_profile: "x1202-synthetic".to_string(),
            battery_voltage_v: voltage,
            battery_percent: soc,
            external_power_present: external,
            charging_enabled: Some(enabled),
            sampled_at_ms: at_ms,
            fuel_gauge_observed_at_ms: at_ms,
            external_power_observed_at_ms: at_ms,
            charging_command_observed_at_ms: at_ms,
            battery_current_a: None,
            battery_current_observable: false,
            confidence: 1.0,
        }
    }

    fn create(at_ms: u64, docked: bool, charging_state: Option<u8>) -> CreatePowerEvidence {
        CreatePowerEvidence {
            observed_at_ms: at_ms,
            stopped: true,
            docked,
            home_base_contact: docked,
            dock_ir_visible: docked,
            charging_state,
            motion_authority_active: false,
            confidence: 1.0,
        }
    }

    #[test]
    fn no_single_contact_command_gpio_or_soc_can_imply_consolidation() {
        let policy = PowerEvidencePolicy::default();
        let now = 20_000;
        let cases = [
            (Some(ups(now, true, true, 4.0, 80.0)), None),
            (
                Some(ups(now, false, true, 4.0, 80.0)),
                Some(create(now, true, Some(2))),
            ),
            (
                Some(ups(now, true, false, 4.0, 80.0)),
                Some(create(now, true, Some(2))),
            ),
            (
                Some(ups(now, true, true, 4.0, 80.0)),
                Some(create(now, true, Some(0))),
            ),
            (
                Some(ups(now, true, true, 4.0, 80.0)),
                Some(create(now, false, Some(2))),
            ),
        ];
        for (ups, create) in cases {
            let assessment =
                assess_consolidation_power(now, ups.as_ref(), None, create.as_ref(), &policy);
            assert!(!assessment.consolidation_ready, "{assessment:?}");
        }
    }

    #[test]
    fn successful_charge_uses_independent_fresh_evidence_without_current() {
        let policy = PowerEvidencePolicy::default();
        let previous = ups(10_000, true, true, 3.90, 50.0);
        let current = ups(20_000, true, true, 3.92, 50.1);
        let assessment = assess_consolidation_power(
            20_000,
            Some(&current),
            Some(&previous),
            Some(&create(20_000, true, Some(2))),
            &policy,
        );
        assert!(assessment.consolidation_ready, "{assessment:?}");
        assert_eq!(
            assessment.battery_charging,
            ChargingInference::LikelyCharging
        );
        assert!(!assessment.battery_current_observable);
        assert_eq!(assessment.battery_current_a, None);
        assert_eq!(assessment.action, "proceed");
    }

    #[test]
    fn power_loss_stale_evidence_and_delayed_charge_start_pause_safely() {
        let policy = PowerEvidencePolicy::default();
        let create_charging = create(20_000, true, Some(2));
        let removed = ups(20_000, false, true, 3.9, 50.0);
        let removed_assessment = assess_consolidation_power(
            20_000,
            Some(&removed),
            None,
            Some(&create_charging),
            &policy,
        );
        assert_eq!(removed_assessment.action, "pause_external_power_lost");

        let stale = ups(10_000, true, true, 3.9, 50.0);
        let stale_assessment =
            assess_consolidation_power(20_000, Some(&stale), None, Some(&create_charging), &policy);
        assert!(!stale_assessment.consolidation_ready);
        assert_eq!(stale_assessment.external_power_present, None);

        let delayed = assess_consolidation_power(
            20_000,
            Some(&ups(20_000, true, true, 3.9, 50.0)),
            None,
            Some(&create(20_000, true, Some(0))),
            &policy,
        );
        assert!(!delayed.consolidation_ready);
        assert!(delayed
            .reasons
            .iter()
            .any(|reason| reason.contains("Create OI charging")));
    }

    #[test]
    fn dirty_contact_or_ir_without_oi_charging_remains_not_ready() {
        let now = 20_000;
        let mut dirty = create(now, true, None);
        dirty.home_base_contact = true;
        dirty.dock_ir_visible = true;
        let assessment = assess_consolidation_power(
            now,
            Some(&ups(now, true, true, 4.0, 80.0)),
            None,
            Some(&dirty),
            &PowerEvidencePolicy::default(),
        );
        assert!(!assessment.consolidation_ready);
        assert_eq!(assessment.create_charging, None);
    }
}
