use pete_cockpit::MotorCommand;
use pete_now::Now;
use serde::{Deserialize, Serialize};

pub trait SafetyLayer {
    fn filter(&mut self, now: &Now, desired: MotorCommand) -> SafetyDecision;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyReason {
    Charging,
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

        if now.body.charging {
            command = MotorCommand::stop();
            events.extend([AutonomicEvent::Stop, AutonomicEvent::Veto]);
            reason = Some(SafetyReason::Charging);
            vetoed = true;
        } else if now.body.flags.wheel_drop {
            command = MotorCommand::stop();
            events.extend([AutonomicEvent::Stop, AutonomicEvent::Veto]);
            reason = Some(SafetyReason::WheelDrop);
            vetoed = true;
        } else if now.body.flags.cliff_left
            || now.body.flags.cliff_front_left
            || now.body.flags.cliff_front_right
            || now.body.flags.cliff_right
        {
            command = MotorCommand::stop();
            events.extend([AutonomicEvent::Stop, AutonomicEvent::Veto]);
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

    #[test]
    fn charging_vetoes_motion_and_overrides_cliff_escape() {
        let mut safety = SimpleSafety::default();
        let mut body = BodySense::default();
        body.charging = true;
        body.flags.cliff_left = true;
        body.last_update_ms = 10;
        let now = Now::blank(10, body);

        let decision = safety.filter(
            &now,
            MotorCommand {
                forward: 0.4,
                turn: 0.3,
            },
        );

        assert!(decision.vetoed);
        assert_eq!(decision.command, MotorCommand::stop());
        assert_eq!(decision.reason, Some(SafetyReason::Charging));
        assert_eq!(
            decision.events,
            vec![AutonomicEvent::Stop, AutonomicEvent::Veto]
        );
    }

    #[test]
    fn charging_transition_stops_previously_allowed_motion() {
        let mut safety = SimpleSafety::default();
        let mut body = BodySense::default();
        body.last_update_ms = 10;
        let desired = MotorCommand {
            forward: 0.4,
            turn: 0.0,
        };

        let moving = safety.filter(&Now::blank(10, body.clone()), desired);
        assert!(!moving.vetoed);
        assert_eq!(moving.command, desired);

        body.charging = true;
        body.last_update_ms = 20;
        let stopped = safety.filter(&Now::blank(20, body), desired);
        assert!(stopped.vetoed);
        assert_eq!(stopped.command, MotorCommand::stop());
        assert_eq!(stopped.reason, Some(SafetyReason::Charging));
    }

    #[test]
    fn every_digital_cliff_sensor_vetoes_with_stop() {
        for sensor in ["left", "front_left", "front_right", "right"] {
            let mut body = BodySense::default();
            match sensor {
                "left" => body.flags.cliff_left = true,
                "front_left" => body.flags.cliff_front_left = true,
                "front_right" => body.flags.cliff_front_right = true,
                "right" => body.flags.cliff_right = true,
                _ => unreachable!(),
            }
            body.last_update_ms = 10;
            let decision = SimpleSafety::default().filter(
                &Now::blank(10, body),
                MotorCommand {
                    forward: 0.4,
                    turn: 0.3,
                },
            );

            assert!(decision.vetoed, "{sensor} cliff did not veto");
            assert_eq!(decision.command, MotorCommand::stop(), "{sensor}");
            assert_eq!(decision.reason, Some(SafetyReason::Cliff), "{sensor}");
            assert_eq!(
                decision.events,
                vec![AutonomicEvent::Stop, AutonomicEvent::Veto],
                "{sensor}"
            );
        }
    }
}
