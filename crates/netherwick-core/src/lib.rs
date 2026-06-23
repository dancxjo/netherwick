use serde::{Deserialize, Serialize};

pub type TimeMs = u64;
pub type BehaviorId = String;
pub type ModelId = String;
pub type SensorId = String;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VecF32(pub Vec<f32>);

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Confidence(pub f32);

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Pose2 {
    pub x_m: f32,
    pub y_m: f32,
    pub heading_rad: f32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Goal {
    Explore,
    Dock,
    Rest,
    Escape,
    Inspect,
    Approach,
    Speak,
}

impl Default for Goal {
    fn default() -> Self {
        Self::Explore
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Reward {
    pub value: f32,
}
