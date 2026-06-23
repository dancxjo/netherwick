use netherwick_core::TimeMs;
use serde::{Deserialize, Serialize};

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
pub struct CommandLease {
    pub command: ActionPrimitive,
    pub issued_at_ms: TimeMs,
    pub expires_at_ms: TimeMs,
    pub interruptible: bool,
    pub safety_class: SafetyClass,
}
