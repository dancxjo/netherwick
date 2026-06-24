use std::collections::BTreeMap;

use netherwick_actions::{ActionPrimitive, ReignInput, ReignMode};
use netherwick_body::BodySense;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

pub type ExtensionMap = BTreeMap<String, Value>;

pub const MEMORY_VECTOR_COLLECTION: &str = "memories";
pub const IMAGE_VECTOR_COLLECTION: &str = "images";
pub const IMAGE_DESCRIPTION_VECTOR_COLLECTION: &str = "image_descriptions";
pub const SCENE_VECTOR_COLLECTION: &str = "scene_vectors";
pub const FACE_VECTOR_COLLECTION: &str = "faces";
pub const VOICE_VECTOR_COLLECTION: &str = "voices";
pub const GEOLOCATION_VECTOR_COLLECTION: &str = "geolocations";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VectorArtifact {
    pub collection: String,
    pub point_id: String,
    pub vector: Vec<f32>,
    pub model: Option<String>,
    pub source_id: Option<String>,
    pub source_frame_id: Option<String>,
    pub occurred_at_ms: Option<u64>,
}

impl VectorArtifact {
    pub fn new(
        collection: impl Into<String>,
        point_id: impl Into<String>,
        vector: Vec<f32>,
    ) -> Self {
        Self {
            collection: collection.into(),
            point_id: point_id.into(),
            vector,
            model: None,
            source_id: None,
            source_frame_id: None,
            occurred_at_ms: None,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = Some(source_id.into());
        self
    }

    pub fn with_source_frame_id(mut self, source_frame_id: impl Into<String>) -> Self {
        self.source_frame_id = Some(source_frame_id.into());
        self
    }

    pub fn with_occurred_at_ms(mut self, occurred_at_ms: u64) -> Self {
        self.occurred_at_ms = Some(occurred_at_ms);
        self
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EyeSense {
    pub schema_version: u32,
    pub frames: Vec<Vec<f32>>,
    #[serde(default)]
    pub image_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub image_description_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub scene_vectors: Vec<VectorArtifact>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EarSense {
    pub schema_version: u32,
    pub features: Vec<Vec<f32>>,
    pub transcript: Option<String>,
    #[serde(default)]
    pub asr: AsrSense,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AsrSense {
    pub transcript: Option<String>,
    pub is_final: bool,
    pub confidence: f32,
    pub sequence_start: Option<u64>,
    pub sequence_end: Option<u64>,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
    pub duration_ms: Option<u64>,
    pub sample_rate_hz: Option<u32>,
    pub word_count: Option<u16>,
    pub speaker_confidence: Option<f32>,
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
    pub danger_model: Option<DangerPrediction>,
    pub danger_hardcoded: Option<DangerPrediction>,
    pub charge_model: Option<ChargePrediction>,
    pub charge_hardcoded: Option<ChargePrediction>,
    #[serde(default)]
    pub action_values_model: Vec<ActionValuePrediction>,
    #[serde(default)]
    pub action_values_hardcoded: Vec<ActionValuePrediction>,
    pub eye_next_model: Option<EyePrediction>,
    pub eye_next_hardcoded: Option<EyePrediction>,
    pub ear_next_model: Option<EarPrediction>,
    pub ear_next_hardcoded: Option<EarPrediction>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DangerPrediction {
    pub bump_risk: f32,
    pub cliff_risk: f32,
    pub wheel_drop_risk: f32,
    pub stuck_risk: f32,
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChargePrediction {
    pub charge_probability: f32,
    pub expected_battery_delta: f32,
    pub dock_likelihood: f32,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionValuePrediction {
    pub action: ActionPrimitive,
    pub value: f32,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EyePrediction {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EarPrediction {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub pcm: Vec<i16>,
    pub features: Vec<f32>,
    pub confidence: f32,
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
pub struct GraphEntity {
    pub id: String,
    pub labels: Vec<String>,
    pub summary: String,
    pub score: f32,
}

impl GraphEntity {
    pub fn has_label(&self, label: &str) -> bool {
        self.labels.iter().any(|candidate| candidate == label)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphEdge {
    pub from: String,
    pub to: String,
    pub relationship: String,
    pub summary: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MemorySense {
    #[serde(default)]
    pub place_familiarity: f32,
    #[serde(default)]
    pub place_danger: f32,
    #[serde(default)]
    pub place_charge_value: f32,
    #[serde(default)]
    pub place_social_value: f32,
    #[serde(default)]
    pub place_novelty: f32,
    #[serde(default)]
    pub nearby_best_charge_direction_rad: Option<f32>,
    #[serde(default)]
    pub nearby_best_safe_direction_rad: Option<f32>,
    #[serde(default)]
    pub places_visited: u32,
    pub face_familiarity: f32,
    pub voice_familiarity: f32,
    pub similar_situation_count: u16,
    pub best_remembered_action: Option<ActionPrimitive>,
    pub remembered_warning: Option<String>,
    #[serde(default)]
    pub remembered_entities: Vec<GraphEntity>,
    #[serde(default)]
    pub remembered_relationships: Vec<GraphEdge>,
    #[serde(default)]
    pub graph_context_summary: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SafetySense {
    pub schema_version: u32,
    pub vetoed: bool,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReignSense {
    pub active: bool,
    pub mode: Option<ReignMode>,
    pub latest: Option<ReignInput>,
    pub pending_count: usize,
    pub last_command_age_ms: Option<u64>,
    pub human_override_pressure: f32,
    pub clear_sequence: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FaceSense {
    pub schema_version: u32,
    pub embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    pub vectors: Vec<VectorArtifact>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VoiceSense {
    pub schema_version: u32,
    pub embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    pub vectors: Vec<VectorArtifact>,
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
    #[serde(default)]
    pub graph_context: Vec<GraphEntity>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Now {
    pub t_ms: u64,
    pub body: BodySense,
    #[serde(default)]
    pub eye: EyeSense,
    #[serde(default)]
    pub ear: EarSense,
    #[serde(default)]
    pub face: FaceSense,
    #[serde(default)]
    pub voice: VoiceSense,
    #[serde(default)]
    pub range: RangeSense,
    #[serde(default)]
    pub imu: ImuSense,
    #[serde(default)]
    pub gps: Option<GpsSense>,
    #[serde(default)]
    pub kinect: KinectSense,
    #[serde(default)]
    pub memory: MemorySense,
    #[serde(default)]
    pub predictions: PredictionSense,
    #[serde(default)]
    pub surprise: SurpriseSense,
    #[serde(default)]
    pub drives: DriveSense,
    #[serde(default)]
    pub llm: LlmSense,
    #[serde(default)]
    pub reign: ReignSense,
    #[serde(default)]
    pub self_sense: SelfSense,
    #[serde(default)]
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
                schema_version: 2,
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
            reign: ReignSense::default(),
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
