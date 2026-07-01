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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EyeFrameFormat {
    Gray8,
    Rgb8,
    Bgr8,
    Yuyv422,
    Uyvy422,
    BayerGrbg8,
    BayerRggb8,
    BayerBggr8,
    BayerGbrg8,
    Mjpeg,
    Unknown(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EyeFrame {
    pub captured_at_ms: u64,
    pub width: u32,
    pub height: u32,
    pub format: EyeFrameFormat,
    pub bytes: Vec<u8>,
    #[serde(default)]
    pub source: Option<String>,
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectClass {
    Obstacle,
    Charger,
    Person,
    SoundSource,
    Landmark,
    Unknown,
}

impl Default for ObjectClass {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectObservationSource {
    Sim,
    Kinect,
    Captioner,
    HumanLabel,
    Unknown,
}

impl Default for ObjectObservationSource {
    fn default() -> Self {
        Self::Unknown
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ObjectObservation {
    pub label: String,
    pub class: ObjectClass,
    pub bearing_rad: f32,
    pub distance_m: Option<f32>,
    pub confidence: f32,
    pub source: ObjectObservationSource,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ObjectSense {
    pub schema_version: u32,
    #[serde(default)]
    pub observations: Vec<ObjectObservation>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub possible_transcript: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub committed_transcript: Option<String>,
    pub is_final: bool,
    pub confidence: f32,
    pub sequence_start: Option<u64>,
    pub sequence_end: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stable_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unstable_text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stable_word_prefix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stable_word_count: Option<u16>,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
    pub duration_ms: Option<u64>,
    pub sample_rate_hz: Option<u32>,
    pub word_count: Option<u16>,
    pub speaker_confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidate_events: Vec<TranscriptCandidateEvent>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TranscriptChunk {
    pub text: String,
    pub is_final: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TranscriptCandidateId(pub u64);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptReplacementReason {
    HeadChanged { stable_prefix_len: usize },
    Restarted,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptCandidateEvent {
    CandidateStarted {
        id: TranscriptCandidateId,
    },
    CandidateUpdated {
        id: TranscriptCandidateId,
        text: String,
        stable_prefix_len: usize,
        confidence: Option<f32>,
    },
    CandidateReplaced {
        old: TranscriptCandidateId,
        new: TranscriptCandidateId,
        reason: TranscriptReplacementReason,
    },
    CandidateFinalized {
        id: TranscriptCandidateId,
        text: String,
        confidence: Option<f32>,
    },
    CandidateCancelled {
        id: TranscriptCandidateId,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TranscriptStabilityState {
    pub candidate_id: TranscriptCandidateId,
    pub text: String,
    pub stable_prefix_len: usize,
    pub stable_text: String,
    pub unstable_text: String,
    pub stable_word_prefix: Option<String>,
    pub stable_word_count: usize,
    pub confidence: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TranscriptCandidateTracker {
    next_id: u64,
    active: Option<ActiveTranscriptCandidate>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ActiveTranscriptCandidate {
    id: TranscriptCandidateId,
    text: String,
}

impl TranscriptCandidateTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ingest_chunk(&mut self, chunk: TranscriptChunk) -> Vec<TranscriptCandidateEvent> {
        self.ingest_candidate(chunk.text, None, chunk.is_final)
    }

    pub fn ingest_candidate(
        &mut self,
        text: impl Into<String>,
        confidence: Option<f32>,
        is_final: bool,
    ) -> Vec<TranscriptCandidateEvent> {
        let text = text.into();
        if text.is_empty() {
            return if is_final {
                self.cancel_active()
            } else {
                Vec::new()
            };
        }

        let mut events = Vec::new();

        if let Some(active) = self.active.take() {
            if active.text == text {
                if is_final {
                    events.push(TranscriptCandidateEvent::CandidateFinalized {
                        id: active.id,
                        text,
                        confidence,
                    });
                } else {
                    let stable_prefix_len = text.len();
                    self.active = Some(ActiveTranscriptCandidate {
                        id: active.id,
                        text: text.clone(),
                    });
                    events.push(TranscriptCandidateEvent::CandidateUpdated {
                        id: active.id,
                        text,
                        stable_prefix_len,
                        confidence,
                    });
                }
                return events;
            }

            let stable_prefix_len = stable_prefix_len(&active.text, &text);
            if stable_prefix_len < active.text.len() {
                let new_id = self.next_id();
                events.push(TranscriptCandidateEvent::CandidateReplaced {
                    old: active.id,
                    new: new_id,
                    reason: TranscriptReplacementReason::HeadChanged { stable_prefix_len },
                });
                events.push(TranscriptCandidateEvent::CandidateStarted { id: new_id });

                if is_final {
                    events.push(TranscriptCandidateEvent::CandidateFinalized {
                        id: new_id,
                        text,
                        confidence,
                    });
                } else {
                    self.active = Some(ActiveTranscriptCandidate {
                        id: new_id,
                        text: text.clone(),
                    });
                    events.push(TranscriptCandidateEvent::CandidateUpdated {
                        id: new_id,
                        text,
                        stable_prefix_len,
                        confidence,
                    });
                }
                return events;
            }

            if is_final {
                events.push(TranscriptCandidateEvent::CandidateFinalized {
                    id: active.id,
                    text,
                    confidence,
                });
            } else {
                self.active = Some(ActiveTranscriptCandidate {
                    id: active.id,
                    text: text.clone(),
                });
                events.push(TranscriptCandidateEvent::CandidateUpdated {
                    id: active.id,
                    text,
                    stable_prefix_len,
                    confidence,
                });
            }
            return events;
        }

        let id = self.next_id();
        events.push(TranscriptCandidateEvent::CandidateStarted { id });
        if is_final {
            events.push(TranscriptCandidateEvent::CandidateFinalized {
                id,
                text,
                confidence,
            });
        } else {
            let stable_prefix_len = text.len();
            self.active = Some(ActiveTranscriptCandidate {
                id,
                text: text.clone(),
            });
            events.push(TranscriptCandidateEvent::CandidateUpdated {
                id,
                text,
                stable_prefix_len,
                confidence,
            });
        }
        events
    }

    pub fn cancel_active(&mut self) -> Vec<TranscriptCandidateEvent> {
        let Some(active) = self.active.take() else {
            return Vec::new();
        };
        vec![TranscriptCandidateEvent::CandidateCancelled { id: active.id }]
    }

    fn next_id(&mut self) -> TranscriptCandidateId {
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("transcript candidate id space exhausted");
        TranscriptCandidateId(self.next_id)
    }
}

impl TranscriptStabilityState {
    pub fn from_parts(
        candidate_id: TranscriptCandidateId,
        text: &str,
        stable_prefix_len: usize,
        confidence: Option<f32>,
    ) -> Self {
        let split = stable_prefix_len.min(text.len());
        let split = if text.is_char_boundary(split) {
            split
        } else {
            text.char_indices()
                .map(|(idx, _)| idx)
                .take_while(|idx| *idx < split)
                .last()
                .unwrap_or_default()
        };
        let (stable_text, unstable_text) = text.split_at(split);
        let stable_word_split = if stable_text
            .chars()
            .next_back()
            .is_some_and(char::is_whitespace)
        {
            stable_text.trim_end().len()
        } else {
            stable_text
                .char_indices()
                .rev()
                .find_map(|(idx, ch)| ch.is_whitespace().then_some(idx + ch.len_utf8()))
                .unwrap_or_default()
        };
        let stable_word_prefix = stable_text[..stable_word_split].trim_end();
        Self {
            candidate_id,
            text: text.to_string(),
            stable_prefix_len: split,
            stable_text: stable_text.to_string(),
            unstable_text: unstable_text.to_string(),
            stable_word_prefix: (!stable_word_prefix.is_empty())
                .then(|| stable_word_prefix.to_string()),
            stable_word_count: stable_word_prefix.split_whitespace().count(),
            confidence,
        }
    }
}

pub fn shared_prefix_len(previous: &str, next: &str) -> usize {
    let mut len = 0;
    let mut previous_chars = previous.char_indices();
    let mut next_chars = next.char_indices();
    loop {
        match (previous_chars.next(), next_chars.next()) {
            (Some((idx, previous_char)), Some((_, next_char))) if previous_char == next_char => {
                len = idx + previous_char.len_utf8();
            }
            _ => break,
        }
    }
    len
}

pub fn stable_prefix_len(previous: &str, next: &str) -> usize {
    let shared = shared_prefix_len(previous, next);
    if shared == 0 {
        return 0;
    }
    if shared == previous.len() || shared == next.len() {
        return shared;
    }
    last_word_boundary_at_or_before(previous, shared)
        .zip(last_word_boundary_at_or_before(next, shared))
        .map(|(previous_boundary, next_boundary)| previous_boundary.min(next_boundary))
        .unwrap_or(shared)
}

fn last_word_boundary_at_or_before(text: &str, limit: usize) -> Option<usize> {
    let mut capped = limit.min(text.len());
    while capped > 0 && !text.is_char_boundary(capped) {
        capped -= 1;
    }
    if capped == 0 {
        return None;
    }
    let mut last_boundary = None;
    for (idx, ch) in text[..capped].char_indices() {
        if ch.is_whitespace() {
            last_boundary = Some(idx + ch.len_utf8());
        }
    }
    if capped < text.len() {
        let previous = text[..capped].chars().next_back();
        let next = text[capped..].chars().next();
        if let (Some(previous), Some(next)) = (previous, next) {
            if previous.is_whitespace() || next.is_whitespace() {
                last_boundary = Some(capped);
            }
        }
    } else {
        last_boundary = Some(capped);
    }
    last_boundary
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
    #[serde(default)]
    pub captured_at_ms: u64,
    /// Orientation uses radians in `[roll, pitch, yaw]` order when all axes are available.
    /// Hardware MPU-6050 samples provide roll/pitch from gravity and no absolute yaw, so they
    /// emit two values. Legacy one-value samples are treated as yaw-only heading.
    pub orientation: Vec<f32>,
    /// Linear acceleration in g units, `[x, y, z]`.
    pub acceleration: Vec<f32>,
    /// Angular velocity in radians per second, `[x, y, z]`.
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
    #[serde(default)]
    pub score: f32,
    #[serde(default)]
    pub payload: Value,
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
    pub nearby_frontier_direction_rad: Option<f32>,
    #[serde(default)]
    pub recent_trap_direction_rad: Option<f32>,
    #[serde(default)]
    pub map_confidence: f32,
    #[serde(default)]
    pub recent_trap_confidence: f32,
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
    #[serde(default)]
    pub captured_at_ms: u64,
    pub color_features: Vec<Vec<f32>>,
    pub depth_m: Vec<f32>,
    #[serde(default)]
    pub depth_width: u32,
    #[serde(default)]
    pub depth_height: u32,
    #[serde(default)]
    pub depth_fx: f32,
    #[serde(default)]
    pub depth_fy: f32,
    #[serde(default)]
    pub depth_cx: f32,
    #[serde(default)]
    pub depth_cy: f32,
    #[serde(default)]
    pub min_depth_m: f32,
    #[serde(default)]
    pub max_depth_m: f32,
    #[serde(default)]
    pub depth_coordinate_system: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eye_frame: Option<EyeFrame>,
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
    pub objects: ObjectSense,
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
            eye_frame: None,
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
            objects: ObjectSense {
                schema_version: 1,
                ..ObjectSense::default()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_tracker_emits_started_then_finalized_for_final_only_asr() {
        let mut tracker = TranscriptCandidateTracker::new();

        let events = tracker.ingest_chunk(TranscriptChunk {
            text: "hello there".to_string(),
            is_final: true,
        });

        assert_eq!(
            events,
            vec![
                TranscriptCandidateEvent::CandidateStarted {
                    id: TranscriptCandidateId(1)
                },
                TranscriptCandidateEvent::CandidateFinalized {
                    id: TranscriptCandidateId(1),
                    text: "hello there".to_string(),
                    confidence: None,
                },
            ]
        );
    }

    #[test]
    fn transcript_tracker_keeps_word_boundary_stable_prefix() {
        let mut tracker = TranscriptCandidateTracker::new();
        let _ = tracker.ingest_candidate("can you tell", Some(0.4), false);

        let events = tracker.ingest_candidate("can you help", Some(0.5), false);

        assert_eq!(
            events,
            vec![
                TranscriptCandidateEvent::CandidateReplaced {
                    old: TranscriptCandidateId(1),
                    new: TranscriptCandidateId(2),
                    reason: TranscriptReplacementReason::HeadChanged {
                        stable_prefix_len: "can you ".len(),
                    },
                },
                TranscriptCandidateEvent::CandidateStarted {
                    id: TranscriptCandidateId(2),
                },
                TranscriptCandidateEvent::CandidateUpdated {
                    id: TranscriptCandidateId(2),
                    text: "can you help".to_string(),
                    stable_prefix_len: "can you ".len(),
                    confidence: Some(0.5),
                },
            ]
        );
    }

    #[test]
    fn transcript_stability_tracks_stable_word_prefix() {
        let state = TranscriptStabilityState::from_parts(
            TranscriptCandidateId(7),
            "hello wor",
            "hello wor".len(),
            Some(0.6),
        );

        assert_eq!(state.stable_text, "hello wor");
        assert_eq!(state.stable_word_prefix.as_deref(), Some("hello"));
        assert_eq!(state.stable_word_count, 1);
        assert_eq!(state.unstable_text, "");
        assert_eq!(state.confidence, Some(0.6));
    }
}
