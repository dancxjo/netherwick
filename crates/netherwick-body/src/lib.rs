use anyhow::Result;
use async_trait::async_trait;
use netherwick_core::{Pose2, TimeMs};
use serde::{Deserialize, Serialize};

#[async_trait]
pub trait RobotBody {
    async fn read_body(&mut self) -> Result<BodySense>;
    async fn apply_motor(&mut self, cmd: MotorCommand) -> Result<()>;
}

#[async_trait]
pub trait MotorComplex {
    async fn send(&mut self, command: MotionCommand) -> Result<BodySense>;
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MotionCommand {
    Stop,
    Forward { speed_m_s: f32 },
    Turn { turn_rad_s: f32 },
    Drive { forward_m_s: f32, turn_rad_s: f32 },
}

impl Default for MotionCommand {
    fn default() -> Self {
        Self::Stop
    }
}

impl MotionCommand {
    pub fn to_motor_command(&self) -> MotorCommand {
        match self {
            Self::Stop => MotorCommand::stop(),
            Self::Forward { speed_m_s } => MotorCommand {
                forward: *speed_m_s,
                turn: 0.0,
            },
            Self::Turn { turn_rad_s } => MotorCommand {
                forward: 0.0,
                turn: *turn_rad_s,
            },
            Self::Drive {
                forward_m_s,
                turn_rad_s,
            } => MotorCommand {
                forward: *forward_m_s,
                turn: *turn_rad_s,
            },
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
