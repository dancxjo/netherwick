use pete_cockpit::MotorCommand;
use pete_now::Now;
use serde::{Deserialize, Serialize};

pub trait SafetyLayer {
    fn filter(&mut self, now: &Now, desired: MotorCommand) -> SafetyDecision;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyReason {
    WheelDrop,
    Cliff,
    BatteryCritical,
    StaleSensors,
    LostBodyComms,
    MotorOutOfRange,
    HighDanger,
    RawLlmMotorRejected,
    ReadOnlyMode,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutonomicEvent {
    Stop,
    Reverse,
    Turn,
    Clamp,
    Veto,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SafetyDecision {
    pub command: MotorCommand,
    pub vetoed: bool,
    pub reason: Option<SafetyReason>,
    pub events: Vec<AutonomicEvent>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SafetyConfig {
    pub max_forward: f32,
    pub max_turn: f32,
    pub critical_battery: f32,
    pub low_battery: f32,
    pub stale_sensor_ms: u64,
    pub lost_body_timeout_ms: u64,
    pub allow_llm_raw_motor: bool,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            max_forward: 0.6,
            max_turn: 1.0,
            critical_battery: 0.10,
            low_battery: 0.20,
            stale_sensor_ms: 500,
            lost_body_timeout_ms: 1000,
            allow_llm_raw_motor: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SimpleSafety {
    pub config: SafetyConfig,
}

impl SafetyLayer for SimpleSafety {
    fn filter(&mut self, now: &Now, desired: MotorCommand) -> SafetyDecision {
        let mut command = desired.clamped(self.config.max_forward, self.config.max_turn);
        let mut events = Vec::new();
        let mut reason = None;
        let mut vetoed = false;

        if command != desired {
            events.push(AutonomicEvent::Clamp);
            reason = Some(SafetyReason::MotorOutOfRange);
        }

        if now.body.flags.wheel_drop {
            command = MotorCommand::stop();
            events.extend([AutonomicEvent::Stop, AutonomicEvent::Veto]);
            reason = Some(SafetyReason::WheelDrop);
            vetoed = true;
        } else if now.body.flags.cliff_left || now.body.flags.cliff_right {
            command = MotorCommand {
                forward: -0.2,
                turn: 0.4,
            };
            events.extend([
                AutonomicEvent::Reverse,
                AutonomicEvent::Turn,
                AutonomicEvent::Veto,
            ]);
            reason = Some(SafetyReason::Cliff);
            vetoed = true;
        } else if now.body.battery_level <= self.config.critical_battery && desired.forward > 0.0 {
            command = MotorCommand::stop();
            events.extend([AutonomicEvent::Stop, AutonomicEvent::Veto]);
            reason = Some(SafetyReason::BatteryCritical);
            vetoed = true;
        } else if now.t_ms.saturating_sub(now.body.last_update_ms) > self.config.stale_sensor_ms {
            command = MotorCommand::stop();
            events.extend([AutonomicEvent::Stop, AutonomicEvent::Veto]);
            reason = Some(SafetyReason::StaleSensors);
            vetoed = true;
        }

        SafetyDecision {
            command,
            vetoed,
            reason,
            events,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pete_body::BodySense;
    use pete_now::Now;

    #[test]
    fn clamps_and_vetoes_wheel_drop() {
        let mut safety = SimpleSafety::default();
        let mut body = BodySense::default();
        body.flags.wheel_drop = true;
        body.last_update_ms = 10;
        let now = Now::blank(10, body);

        let decision = safety.filter(
            &now,
            MotorCommand {
                forward: 2.0,
                turn: 2.0,
            },
        );

        assert!(decision.vetoed);
        assert_eq!(decision.command, MotorCommand::stop());
        assert_eq!(decision.reason, Some(SafetyReason::WheelDrop));
    }
}
