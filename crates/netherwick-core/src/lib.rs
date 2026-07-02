use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

pub type TimeMs = u64;
pub type BehaviorId = String;
pub type ModelId = String;
pub type SensorId = String;
pub type SensationId = Uuid;
pub type ImpressionId = Uuid;
pub type ExperienceId = Uuid;
pub type FeatureId = Uuid;
pub type FrameId = String;
pub type VectorId = String;

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Pose3 {
    pub x_m: f32,
    pub y_m: f32,
    pub z_m: f32,
    pub roll_rad: f32,
    pub pitch_rad: f32,
    pub yaw_rad: f32,
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceKind {
    Direct,
    DerivedFromSensations { sensation_ids: Vec<SensationId> },
    DerivedFromImpressions { impression_ids: Vec<ImpressionId> },
    MemoryRecall { experience_id: ExperienceId },
}

impl Default for ProvenanceKind {
    fn default() -> Self {
        Self::Direct
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub kind: ProvenanceKind,
    pub stage_chain: Vec<String>,
    pub metadata: Value,
}

impl Provenance {
    pub fn direct() -> Self {
        Self::default()
    }

    pub fn derived_from_sensations(sensation_ids: impl IntoIterator<Item = SensationId>) -> Self {
        Self {
            kind: ProvenanceKind::DerivedFromSensations {
                sensation_ids: sensation_ids.into_iter().collect(),
            },
            ..Self::default()
        }
    }

    pub fn derived_from_impressions(
        impression_ids: impl IntoIterator<Item = ImpressionId>,
    ) -> Self {
        Self {
            kind: ProvenanceKind::DerivedFromImpressions {
                impression_ids: impression_ids.into_iter().collect(),
            },
            ..Self::default()
        }
    }

    pub fn memory_recall(experience_id: ExperienceId) -> Self {
        Self {
            kind: ProvenanceKind::MemoryRecall { experience_id },
            ..Self::default()
        }
    }

    pub fn with_stage(mut self, stage: impl Into<String>) -> Self {
        self.stage_chain.push(stage.into());
        self
    }

    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureModality {
    Vision,
    Geometry,
    Motion,
    Audio,
    Language,
    Body,
    Memory,
    Prediction,
    #[default]
    Other,
}

impl FeatureModality {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Vision => "vision",
            Self::Geometry => "geometry",
            Self::Motion => "motion",
            Self::Audio => "audio",
            Self::Language => "language",
            Self::Body => "body",
            Self::Memory => "memory",
            Self::Prediction => "prediction",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureType {
    ObjectDetection,
    ImageRegion,
    SemanticSegmentation,
    RgbPatch,
    ImageDescriptor,
    FaceObservation,
    Voxel,
    PointCloudFeature,
    Plane,
    Corner,
    Blob,
    OccupancyCell,
    OdometryEvent,
    ImuEvent,
    OpticalFlow,
    MotionVector,
    SpeechSegment,
    VoiceEmbedding,
    SoundEvent,
    DirectionEstimate,
    Transcript,
    ImageDescription,
    LlmLabel,
    HumanCorrection,
    Battery,
    Bumper,
    Cliff,
    DockState,
    WheelDrop,
    SafetyMode,
    RobotMode,
    RecalledExperience,
    RememberedPlace,
    RememberedEntity,
    RememberedAction,
    FuturePrediction,
    Surprise,
    Counterfactual,
    ActionProposal,
    VectorArtifact,
    #[default]
    Other,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VectorRef {
    pub collection: String,
    pub point_id: VectorId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
}

impl VectorRef {
    pub fn new(collection: impl Into<String>, point_id: impl Into<String>) -> Self {
        Self {
            collection: collection.into(),
            point_id: point_id.into(),
            model: None,
            source_id: None,
        }
    }

    pub fn with_model(mut self, model: Option<String>) -> Self {
        self.model = model;
        self
    }

    pub fn with_source_id(mut self, source_id: Option<String>) -> Self {
        self.source_id = source_id;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Feature {
    pub id: FeatureId,
    pub feature_type: FeatureType,
    pub modality: FeatureModality,
    pub created_at_ms: TimeMs,
    pub confidence: f32,
    pub provenance: Provenance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_frame: Option<FrameId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_sensor: Option<SensorId>,
    #[serde(default)]
    pub vector_refs: Vec<VectorRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world_pose: Option<Pose3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_pose: Option<Pose3>,
    #[serde(default)]
    pub metadata: Value,
}

impl Feature {
    pub fn new(
        feature_type: FeatureType,
        modality: FeatureModality,
        created_at_ms: TimeMs,
        confidence: f32,
        provenance: Provenance,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            feature_type,
            modality,
            created_at_ms,
            confidence: confidence.clamp(0.0, 1.0),
            provenance,
            source_frame: None,
            source_sensor: None,
            vector_refs: Vec::new(),
            world_pose: None,
            local_pose: None,
            metadata: Value::Null,
        }
    }

    pub fn with_source_frame(mut self, source_frame: impl Into<String>) -> Self {
        self.source_frame = Some(source_frame.into());
        self
    }

    pub fn with_optional_source_frame(mut self, source_frame: Option<String>) -> Self {
        self.source_frame = source_frame;
        self
    }

    pub fn with_source_sensor(mut self, source_sensor: impl Into<String>) -> Self {
        self.source_sensor = Some(source_sensor.into());
        self
    }

    pub fn with_vector_ref(mut self, vector_ref: VectorRef) -> Self {
        self.vector_refs.push(vector_ref);
        self
    }

    pub fn with_local_pose(mut self, local_pose: Pose3) -> Self {
        self.local_pose = Some(local_pose);
        self
    }

    pub fn with_world_pose(mut self, world_pose: Pose3) -> Self {
        self.world_pose = Some(world_pose);
        self
    }

    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FeatureRegistry {
    features: BTreeMap<FeatureId, Feature>,
    by_modality: BTreeMap<FeatureModality, BTreeSet<FeatureId>>,
    by_source_frame: BTreeMap<FrameId, BTreeSet<FeatureId>>,
    by_source_sensor: BTreeMap<SensorId, BTreeSet<FeatureId>>,
    by_vector_id: BTreeMap<VectorId, BTreeSet<FeatureId>>,
    by_provenance: BTreeMap<String, BTreeSet<FeatureId>>,
}

impl FeatureRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    pub fn insert(&mut self, feature: Feature) -> Option<Feature> {
        let id = feature.id;
        if let Some(previous) = self.features.get(&id).cloned() {
            self.remove_from_indexes(&previous);
        }
        self.add_to_indexes(&feature);
        let replaced = self.features.insert(id, feature);
        if let Some(previous) = &replaced {
            debug_assert_eq!(previous.id, id);
        }
        replaced
    }

    pub fn extend(&mut self, features: impl IntoIterator<Item = Feature>) {
        for feature in features {
            self.insert(feature);
        }
    }

    pub fn get(&self, id: &FeatureId) -> Option<&Feature> {
        self.features.get(id)
    }

    pub fn by_modality(&self, modality: FeatureModality) -> Vec<&Feature> {
        self.lookup(self.by_modality.get(&modality))
    }

    pub fn by_time_window(&self, start_ms: TimeMs, end_ms: TimeMs) -> Vec<&Feature> {
        self.features
            .values()
            .filter(|feature| feature.created_at_ms >= start_ms && feature.created_at_ms <= end_ms)
            .collect()
    }

    pub fn by_source_frame(&self, source_frame: &str) -> Vec<&Feature> {
        self.lookup(self.by_source_frame.get(source_frame))
    }

    pub fn by_source_sensor(&self, source_sensor: &str) -> Vec<&Feature> {
        self.lookup(self.by_source_sensor.get(source_sensor))
    }

    pub fn by_vector_id(&self, vector_id: &str) -> Vec<&Feature> {
        self.lookup(self.by_vector_id.get(vector_id))
    }

    pub fn by_provenance(&self, provenance: &Provenance) -> Vec<&Feature> {
        let key = provenance_key(provenance);
        self.lookup(self.by_provenance.get(&key))
    }

    pub fn all(&self) -> impl Iterator<Item = &Feature> {
        self.features.values()
    }

    fn lookup<'a>(&'a self, ids: Option<&'a BTreeSet<FeatureId>>) -> Vec<&'a Feature> {
        ids.into_iter()
            .flat_map(|ids| ids.iter())
            .filter_map(|id| self.features.get(id))
            .collect()
    }

    fn add_to_indexes(&mut self, feature: &Feature) {
        self.by_modality
            .entry(feature.modality.clone())
            .or_default()
            .insert(feature.id);
        if let Some(source_frame) = &feature.source_frame {
            self.by_source_frame
                .entry(source_frame.clone())
                .or_default()
                .insert(feature.id);
        }
        if let Some(source_sensor) = &feature.source_sensor {
            self.by_source_sensor
                .entry(source_sensor.clone())
                .or_default()
                .insert(feature.id);
        }
        for vector_ref in &feature.vector_refs {
            self.by_vector_id
                .entry(vector_ref.point_id.clone())
                .or_default()
                .insert(feature.id);
        }
        self.by_provenance
            .entry(provenance_key(&feature.provenance))
            .or_default()
            .insert(feature.id);
    }

    fn remove_from_indexes(&mut self, feature: &Feature) {
        remove_indexed_id(&mut self.by_modality, &feature.modality, &feature.id);
        if let Some(source_frame) = &feature.source_frame {
            remove_indexed_id(&mut self.by_source_frame, source_frame, &feature.id);
        }
        if let Some(source_sensor) = &feature.source_sensor {
            remove_indexed_id(&mut self.by_source_sensor, source_sensor, &feature.id);
        }
        for vector_ref in &feature.vector_refs {
            remove_indexed_id(&mut self.by_vector_id, &vector_ref.point_id, &feature.id);
        }
        remove_indexed_id(
            &mut self.by_provenance,
            &provenance_key(&feature.provenance),
            &feature.id,
        );
    }
}

fn remove_indexed_id<K: Ord + Clone>(
    index: &mut BTreeMap<K, BTreeSet<FeatureId>>,
    key: &K,
    id: &FeatureId,
) {
    if let Some(ids) = index.get_mut(key) {
        ids.remove(id);
        if ids.is_empty() {
            index.remove(key);
        }
    }
}

fn provenance_key(provenance: &Provenance) -> String {
    serde_json::to_string(provenance).unwrap_or_else(|_| format!("{:?}", provenance.kind))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn feature_registry_indexes_common_lookup_dimensions() {
        let provenance = Provenance::direct().with_stage("test");
        let feature = Feature::new(
            FeatureType::FaceObservation,
            FeatureModality::Vision,
            100,
            0.8,
            provenance.clone(),
        )
        .with_source_frame("frame-1")
        .with_source_sensor("kinect")
        .with_vector_ref(VectorRef::new("faces", "face-1"))
        .with_metadata(json!({ "label": "face" }));
        let id = feature.id;

        let mut registry = FeatureRegistry::new();
        registry.insert(feature);

        assert_eq!(registry.get(&id).map(|feature| feature.id), Some(id));
        assert_eq!(registry.by_modality(FeatureModality::Vision).len(), 1);
        assert_eq!(registry.by_time_window(50, 150).len(), 1);
        assert_eq!(registry.by_source_frame("frame-1").len(), 1);
        assert_eq!(registry.by_source_sensor("kinect").len(), 1);
        assert_eq!(registry.by_vector_id("face-1").len(), 1);
        assert_eq!(registry.by_provenance(&provenance).len(), 1);
    }

    #[test]
    fn feature_registry_reindex_replacement() {
        let mut first = Feature::new(
            FeatureType::VoiceEmbedding,
            FeatureModality::Audio,
            100,
            0.8,
            Provenance::direct().with_stage("test"),
        )
        .with_source_sensor("mic-a")
        .with_vector_ref(VectorRef::new("voices", "voice-a"));
        let id = first.id;

        let mut second = Feature::new(
            FeatureType::VoiceEmbedding,
            FeatureModality::Audio,
            200,
            0.9,
            Provenance::direct().with_stage("test"),
        )
        .with_source_sensor("mic-b")
        .with_vector_ref(VectorRef::new("voices", "voice-b"));
        second.id = id;
        first.id = id;

        let mut registry = FeatureRegistry::new();
        registry.insert(first);
        registry.insert(second);

        assert!(registry.by_source_sensor("mic-a").is_empty());
        assert!(registry.by_vector_id("voice-a").is_empty());
        assert_eq!(registry.by_source_sensor("mic-b").len(), 1);
        assert_eq!(registry.by_vector_id("voice-b").len(), 1);
    }
}
