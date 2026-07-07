use pete_core::{Pose2, TimeMs};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BodyTone {
    pub note: u8,
    pub duration_64ths: u8,
}

impl BodyTone {
    pub fn new(note: u8, duration_64ths: u8) -> Self {
        Self {
            note,
            duration_64ths,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BodySong {
    pub tones: Vec<BodyTone>,
}

impl BodySong {
    pub fn new(tones: impl Into<Vec<BodyTone>>) -> Self {
        Self {
            tones: tones.into(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BodyFlags {
    pub bump_left: bool,
    pub bump_right: bool,
    pub cliff_left: bool,
    pub cliff_front_left: bool,
    pub cliff_front_right: bool,
    pub cliff_right: bool,
    pub wheel_drop: bool,
    pub wall: bool,
    pub virtual_wall: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CliffSensors {
    pub left: f32,
    pub front_left: f32,
    pub front_right: f32,
    pub right: f32,
}

impl CliffSensors {
    pub fn max(self) -> f32 {
        self.left
            .max(self.front_left)
            .max(self.front_right)
            .max(self.right)
            .clamp(0.0, 1.0)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Velocity {
    pub forward_m_s: f32,
    pub turn_rad_s: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BodyHealth {
    pub strain: f32,
    pub health: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodySense {
    pub battery_level: f32,
    pub charging: bool,
    #[serde(default)]
    pub cliff_sensors: CliffSensors,
    pub flags: BodyFlags,
    pub odometry: Pose2,
    pub velocity: Velocity,
    pub health: BodyHealth,
    pub last_update_ms: TimeMs,
}

impl Default for BodySense {
    fn default() -> Self {
        Self {
            battery_level: 1.0,
            charging: false,
            cliff_sensors: CliffSensors::default(),
            flags: BodyFlags::default(),
            odometry: Pose2::default(),
            velocity: Velocity::default(),
            health: BodyHealth {
                strain: 0.0,
                health: 1.0,
            },
            last_update_ms: 0,
        }
    }
}
