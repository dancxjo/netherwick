use std::collections::BTreeMap;

use netherwick_actions::ActionPrimitive;
use netherwick_body::BodySense;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

pub type ExtensionMap = BTreeMap<String, Value>;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EyeSense {
    pub schema_version: u32,
    pub frames: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EarSense {
    pub schema_version: u32,
    pub features: Vec<Vec<f32>>,
    pub transcript: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RangeSense {
    pub schema_version: u32,
    pub beams: Vec<f32>,
    pub nearest_m: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ImuSense {
    pub schema_version: u32,
    pub orientation: Vec<f32>,
    pub acceleration: Vec<f32>,
    pub angular_velocity: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GpsSense {
    pub schema_version: u32,
    pub lat: f64,
    pub lon: f64,
    pub altitude_m: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PredictionSense {
    pub schema_version: u32,
    pub expected_events: Vec<String>,
    pub uncertainty: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SurpriseSense {
    pub schema_version: u32,
    pub total: f32,
    pub prediction_error: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DriveSense {
    pub battery_hunger: f32,
    pub danger_avoidance: f32,
    pub curiosity: f32,
    pub social_interest: f32,
    pub fatigue: f32,
    pub uncertainty_pressure: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmSense {
    pub schema_version: u32,
    pub command_summary: Option<String>,
    pub critique: Option<String>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SelfSense {
    pub schema_version: u32,
    pub active_goal: Option<String>,
    pub mode: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MemorySense {
    pub place_familiarity: f32,
    pub place_danger: f32,
    pub place_charge_value: f32,
    pub face_familiarity: f32,
    pub voice_familiarity: f32,
    pub similar_situation_count: u16,
    pub best_remembered_action: Option<ActionPrimitive>,
    pub remembered_warning: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SafetySense {
    pub schema_version: u32,
    pub vetoed: bool,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FaceSense {
    pub schema_version: u32,
    pub embeddings: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VoiceSense {
    pub schema_version: u32,
    pub embeddings: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct KinectJointSense {
    pub joint_name: String,
    pub position_m: [f32; 3],
    pub tracking_confidence: f32,
    pub tracked: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct KinectSkeletonSense {
    pub tracking_id: u64,
    pub lean_xy: [f32; 2],
    pub joints: Vec<KinectJointSense>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct KinectSense {
    pub schema_version: u32,
    pub color_features: Vec<Vec<f32>>,
    pub depth_m: Vec<f32>,
    pub ir: Vec<f32>,
    pub player_index: Vec<u8>,
    pub audio_angle_rad: Option<f32>,
    pub audio_confidence: f32,
    pub floor_clip_plane: Vec<f32>,
    pub skeletons: Vec<KinectSkeletonSense>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExtensionSense {
    pub schema_version: u32,
    pub name: String,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RecallHit {
    pub frame_id: Option<Uuid>,
    pub score: f32,
    pub summary: String,
    pub warning: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Now {
    pub t_ms: u64,
    pub body: BodySense,
    pub eye: EyeSense,
    pub ear: EarSense,
    pub face: FaceSense,
    pub voice: VoiceSense,
    pub range: RangeSense,
    pub imu: ImuSense,
    pub gps: Option<GpsSense>,
    pub kinect: KinectSense,
    pub memory: MemorySense,
    pub predictions: PredictionSense,
    pub surprise: SurpriseSense,
    pub drives: DriveSense,
    pub llm: LlmSense,
    pub self_sense: SelfSense,
    pub extensions: ExtensionMap,
}

impl Now {
    pub fn blank(t_ms: u64, body: BodySense) -> Self {
        Self {
            t_ms,
            body,
            eye: EyeSense {
                schema_version: 1,
                ..EyeSense::default()
            },
            ear: EarSense {
                schema_version: 1,
                ..EarSense::default()
            },
            face: FaceSense {
                schema_version: 1,
                ..FaceSense::default()
            },
            voice: VoiceSense {
                schema_version: 1,
                ..VoiceSense::default()
            },
            range: RangeSense {
                schema_version: 1,
                ..RangeSense::default()
            },
            imu: ImuSense {
                schema_version: 1,
                ..ImuSense::default()
            },
            gps: None,
            kinect: KinectSense {
                schema_version: 1,
                ..KinectSense::default()
            },
            memory: MemorySense::default(),
            predictions: PredictionSense {
                schema_version: 1,
                ..PredictionSense::default()
            },
            surprise: SurpriseSense {
                schema_version: 1,
                ..SurpriseSense::default()
            },
            drives: DriveSense::default(),
            llm: LlmSense {
                schema_version: 1,
                ..LlmSense::default()
            },
            self_sense: SelfSense {
                schema_version: 1,
                ..SelfSense::default()
            },
            extensions: ExtensionMap::default(),
        }
    }
}

pub trait SenseVectorizer {
    fn sense_name(&self) -> &'static str;
    fn schema_version(&self) -> u32;
    fn encode(&self, now: &Now) -> Vec<f32>;
}
