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
    SetMode {
        mode: ReignMode,
    },
}

impl ReignCommand {
    pub fn default_ttl_ms(&self) -> TimeMs {
        match self {
            Self::Stop => 2_000,
            Self::Go { duration_ms, .. } | Self::Turn { duration_ms, .. } => {
                duration_ms.saturating_add(500)
            }
            Self::Dock | Self::Explore { .. } | Self::Approach { .. } => 5_000,
            Self::Inspect { .. } => 5_000,
            Self::Speak { .. } => 10_000,
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CommandLease {
    pub command: ActionPrimitive,
    pub issued_at_ms: TimeMs,
    pub expires_at_ms: TimeMs,
    pub interruptible: bool,
    pub safety_class: SafetyClass,
}
