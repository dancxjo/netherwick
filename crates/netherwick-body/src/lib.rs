use anyhow::Result;
use async_trait::async_trait;
use netherwick_core::{Pose2, TimeMs};
use serde::{Deserialize, Serialize};

#[async_trait]
pub trait RobotBody {
    async fn read_body(&mut self) -> Result<BodySense>;
    async fn apply_motor(&mut self, cmd: MotorCommand) -> Result<()>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MotorCommand {
    pub forward: f32,
    pub turn: f32,
}

impl MotorCommand {
    pub fn stop() -> Self {
        Self::default()
    }

    pub fn clamped(self, max_forward: f32, max_turn: f32) -> Self {
        Self {
            forward: self.forward.clamp(-max_forward, max_forward),
            turn: self.turn.clamp(-max_turn, max_turn),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BodyFlags {
    pub bump_left: bool,
    pub bump_right: bool,
    pub cliff_left: bool,
    pub cliff_right: bool,
    pub wheel_drop: bool,
    pub wall: bool,
    pub virtual_wall: bool,
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
