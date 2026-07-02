use netherwick_body::{MotionCommand, MotorCommand};
use netherwick_core::TimeMs;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurnDir {
    Left,
    Right,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InspectTarget {
    Novelty,
    Charger,
    Person,
    Sound,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApproachTarget {
    Charger,
    Person,
    Sound,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExploreStyle {
    Wander,
    RandomWalk,
    WallFollow,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChirpPattern {
    Confirm,
    Warning,
    Curious,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyClass {
    Safe,
    Cautious,
    Critical,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ActionPrimitive {
    Stop,
    Go {
        intensity: f32,
        duration_ms: TimeMs,
    },
    Drive {
        forward: f32,
        turn: f32,
        duration_ms: TimeMs,
    },
    Turn {
        direction: TurnDir,
        intensity: f32,
        duration_ms: TimeMs,
    },
    Inspect {
        target: InspectTarget,
    },
    Approach {
        target: ApproachTarget,
    },
    Dock,
    Explore {
        style: ExploreStyle,
        duration_ms: TimeMs,
    },
    Speak {
        text: String,
    },
    Chirp {
        pattern: ChirpPattern,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ReignCommand {
    Stop,
    Go {
        intensity: f32,
        duration_ms: TimeMs,
    },
    Reverse {
        intensity: f32,
        duration_ms: TimeMs,
    },
    Drive {
        forward: f32,
        turn: f32,
        duration_ms: TimeMs,
    },
    Turn {
        direction: TurnDir,
        intensity: f32,
        duration_ms: TimeMs,
    },
    Inspect {
        target: InspectTarget,
    },
    Approach {
        target: ApproachTarget,
    },
    Dock,
    Explore {
        duration_ms: TimeMs,
    },
    Speak {
        text: String,
    },
    Chirp {
        pattern: ChirpPattern,
    },
    SetMode {
        mode: ReignMode,
    },
}

impl ReignCommand {
    pub fn default_ttl_ms(&self) -> TimeMs {
        match self {
            Self::Stop => 2_000,
            Self::Go { duration_ms, .. }
            | Self::Reverse { duration_ms, .. }
            | Self::Drive { duration_ms, .. }
            | Self::Turn { duration_ms, .. } => duration_ms.saturating_add(500),
            Self::Dock | Self::Explore { .. } | Self::Approach { .. } => 5_000,
            Self::Inspect { .. } => 5_000,
            Self::Speak { .. } => 10_000,
            Self::Chirp { .. } => 2_000,
            Self::SetMode { .. } => 2_000,
        }
    }

    pub fn to_action(&self) -> Option<ActionPrimitive> {
        match self {
            Self::Stop => Some(ActionPrimitive::Stop),
            Self::Go {
                intensity,
                duration_ms,
            } => Some(ActionPrimitive::Go {
                intensity: *intensity,
                duration_ms: *duration_ms,
            }),
            Self::Reverse {
                intensity,
                duration_ms,
            } => Some(ActionPrimitive::Go {
                intensity: -*intensity,
                duration_ms: *duration_ms,
            }),
            Self::Drive {
                forward,
                turn,
                duration_ms,
            } => Some(ActionPrimitive::Drive {
                forward: *forward,
                turn: *turn,
                duration_ms: *duration_ms,
            }),
            Self::Turn {
                direction,
                intensity,
                duration_ms,
            } => Some(ActionPrimitive::Turn {
                direction: direction.clone(),
                intensity: *intensity,
                duration_ms: *duration_ms,
            }),
            Self::Inspect { target } => Some(ActionPrimitive::Inspect {
                target: target.clone(),
            }),
            Self::Approach { target } => Some(ActionPrimitive::Approach {
                target: target.clone(),
            }),
            Self::Dock => Some(ActionPrimitive::Dock),
            Self::Explore { duration_ms } => Some(ActionPrimitive::Explore {
                style: ExploreStyle::Wander,
                duration_ms: *duration_ms,
            }),
            Self::Speak { text } => Some(ActionPrimitive::Speak { text: text.clone() }),
            Self::Chirp { pattern } => Some(ActionPrimitive::Chirp {
                pattern: pattern.clone(),
            }),
            Self::SetMode { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReignMode {
    ObserveOnly,
    Suggest,
    Assist,
    Direct,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReignInput {
    pub id: Uuid,
    pub issued_at_ms: TimeMs,
    pub expires_at_ms: TimeMs,
    pub source: ReignSource,
    pub mode: ReignMode,
    pub command: ReignCommand,
    pub priority: f32,
    pub note: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReignSource {
    WebRemote,
    Keyboard,
    Gamepad,
    Script,
    HumanSupervisor,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReignOutcome {
    pub input_id: Uuid,
    pub accepted_by_conductor: bool,
    pub vetoed_by_safety: bool,
    pub final_action: Option<ActionPrimitive>,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmActionProposal {
    pub proposed_action: Option<ActionPrimitive>,
    pub accepted: bool,
    pub safety_vetoed: bool,
    pub final_action: Option<ActionPrimitive>,
    pub ignored_reason: Option<String>,
    pub safety_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CommandLease {
    pub command: ActionPrimitive,
    pub issued_at_ms: TimeMs,
    pub expires_at_ms: TimeMs,
    pub interruptible: bool,
    pub safety_class: SafetyClass,
}

pub fn action_to_motion(action: Option<&ActionPrimitive>) -> MotionCommand {
    match action_to_motor_command(action) {
        MotorCommand { forward, turn } if forward == 0.0 && turn == 0.0 => MotionCommand::Stop,
        MotorCommand { forward, turn } if turn == 0.0 => {
            MotionCommand::Forward { speed_m_s: forward }
        }
        MotorCommand { forward, turn } if forward == 0.0 => {
            MotionCommand::Turn { turn_rad_s: turn }
        }
        MotorCommand { forward, turn } => MotionCommand::Drive {
            forward_m_s: forward,
            turn_rad_s: turn,
        },
    }
}

pub fn action_to_motor_command(action: Option<&ActionPrimitive>) -> MotorCommand {
    let Some(action) = action else {
        return MotorCommand::stop();
    };
    match action {
        ActionPrimitive::Stop => MotorCommand::stop(),
        ActionPrimitive::Go { intensity, .. } => MotorCommand {
            forward: *intensity,
            turn: 0.0,
        },
        ActionPrimitive::Drive { forward, turn, .. } => MotorCommand {
            forward: *forward,
            turn: *turn,
        },
        ActionPrimitive::Turn {
            direction,
            intensity,
            ..
        } => MotorCommand {
            forward: 0.0,
            turn: match direction {
                TurnDir::Left => *intensity,
                TurnDir::Right => -*intensity,
            },
        },
        ActionPrimitive::Inspect { target } => match target {
            InspectTarget::Novelty | InspectTarget::Sound | InspectTarget::Person => MotorCommand {
                forward: 0.0,
                turn: 0.16,
            },
            _ => MotorCommand::stop(),
        },
        ActionPrimitive::Approach { target } => match target {
            ApproachTarget::Charger | ApproachTarget::Person | ApproachTarget::Sound => {
                MotorCommand {
                    forward: 0.2,
                    turn: 0.0,
                }
            }
        },
        ActionPrimitive::Dock => MotorCommand {
            forward: 0.15,
            turn: 0.0,
        },
        ActionPrimitive::Explore { style, .. } => match style {
            ExploreStyle::Wander | ExploreStyle::RandomWalk => MotorCommand {
                forward: 0.2,
                turn: 0.1,
            },
            ExploreStyle::WallFollow => MotorCommand {
                forward: 0.15,
                turn: 0.2,
            },
        },
        ActionPrimitive::Speak { .. } | ActionPrimitive::Chirp { .. } => MotorCommand::stop(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inspect_novelty_scans_instead_of_stopping() {
        let motor = action_to_motor_command(Some(&ActionPrimitive::Inspect {
            target: InspectTarget::Novelty,
        }));

        assert_eq!(motor.forward, 0.0);
        assert!(motor.turn.abs() > 0.0);
    }

    #[test]
    fn reign_reverse_maps_to_negative_go_action() {
        let action = ReignCommand::Reverse {
            intensity: 0.4,
            duration_ms: 350,
        }
        .to_action();

        assert_eq!(
            action,
            Some(ActionPrimitive::Go {
                intensity: -0.4,
                duration_ms: 350
            })
        );
        let motor = action_to_motor_command(action.as_ref());
        assert_eq!(motor.forward, -0.4);
        assert_eq!(motor.turn, 0.0);
    }

    #[test]
    fn reign_drive_maps_to_combined_motor_command() {
        let action = ReignCommand::Drive {
            forward: 0.35,
            turn: -0.42,
            duration_ms: 320,
        }
        .to_action();

        assert_eq!(
            action,
            Some(ActionPrimitive::Drive {
                forward: 0.35,
                turn: -0.42,
                duration_ms: 320
            })
        );
        let motor = action_to_motor_command(action.as_ref());
        assert_eq!(motor.forward, 0.35);
        assert_eq!(motor.turn, -0.42);
    }

    #[test]
    fn reign_chirp_maps_to_chirp_action_without_motion() {
        let action = ReignCommand::Chirp {
            pattern: ChirpPattern::Confirm,
        }
        .to_action();

        assert_eq!(
            action,
            Some(ActionPrimitive::Chirp {
                pattern: ChirpPattern::Confirm
            })
        );
        assert_eq!(
            action_to_motor_command(action.as_ref()),
            MotorCommand::stop()
        );
    }
}
