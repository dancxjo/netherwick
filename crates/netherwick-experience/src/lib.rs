use netherwick_core::TimeMs;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExperienceLatent {
    pub t_ms: TimeMs,
    pub z: Vec<f32>,
    pub reconstruction_error: f32,
    pub prediction_error: f32,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FuturePrediction {
    pub offset_ms: TimeMs,
    pub predicted_z: Vec<f32>,
    pub confidence: f32,
    pub summary: Option<String>,
}
