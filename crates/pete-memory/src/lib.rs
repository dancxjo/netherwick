use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use pete_actions::{ActionPrimitive, TurnDir};
use pete_body::{BodyFlags, BodySense, Velocity};
use pete_core::{Feature, FeatureId, Goal, Pose2, Reward};
use pete_experience::{
    EmbodiedContext, EmbodiedPipeline, EmbodiedVectorCoverage, Experience, ExperienceFuser,
    FuturePrediction, Impression, InstantCoverage, MemoryLink, Modality, RecalledExperience,
    SensationPayloadKind, VectorEmbedding,
};
use pete_ledger::{ExperienceFrame, ExperienceTransition};
use pete_now::{
    AsrSense, EarSense, Episode, EpisodeKind, EpistemicSnapshot, EyeFrame, EyeFrameFormat,
    GraphEdge, GraphEntity, InteractionState, KinectJointSense, KinectSense, KinectSkeletonSense,
    MemorySense, Now, ObjectClass, ObjectObservation, ObjectObservationSource, PersonId,
    RangeSense, RecallHit, SemanticGraphSnapshot, SemanticNodeRef, SocialWorldSnapshot,
    SurpriseSense, TemporalContext, VectorArtifact, FACE_VECTOR_COLLECTION,
    SCENE_VECTOR_COLLECTION, VOICE_VECTOR_COLLECTION,
};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

const DEFAULT_CELL_SIZE_M: f32 = 0.5;
const SCORE_DECAY_PER_TICK: f32 = 0.995;
const CELL_CONFIDENCE_DECAY_PER_TICK: f32 = 0.999;
const RECALL_RADIUS_CELLS: i32 = 4;
const PLACE_RECOGNITION_MIN_CONFIDENCE: f32 = 0.55;
pub const SENSATION_VECTOR_COLLECTION: &str = "sensations";
pub const EXPERIENCE_VECTOR_COLLECTION: &str = "experiences";

#[async_trait]
pub trait MemoryStore {
    async fn store(&self, frame: &ExperienceFrame) -> Result<()>;
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn upsert_vectors(&self, record: &MemoryRecord) -> Result<()>;
}

#[async_trait]
pub trait GraphStore: Send + Sync {
    async fn upsert_graph(&self, record: &MemoryRecord) -> Result<()>;
}

#[async_trait]
pub trait GraphIntelligence: Send + Sync {
    async fn upsert_intelligence(&self, document: &GraphIntelligenceDocument) -> Result<()>;
    async fn feature_or_cluster_intelligence(
        &self,
        node_id: &str,
        limit: usize,
    ) -> Result<FeatureClusterIntelligence>;
    async fn constellation_intelligence(
        &self,
        constellation_id: &str,
        limit: usize,
    ) -> Result<ConstellationIntelligence>;
    async fn ambiguity_intelligence(
        &self,
        family_or_target_id: &str,
        limit: usize,
    ) -> Result<AmbiguityIntelligence>;
    async fn action_outcome_intelligence(
        &self,
        action_id: &str,
        limit: usize,
    ) -> Result<ActionOutcomeIntelligence>;
    async fn local_community(
        &self,
        start_node_id: &str,
        max_depth: u32,
        min_weight: f32,
        limit: usize,
    ) -> Result<GraphCommunity>;
    async fn graph_recall(
        &self,
        query: GraphRecallQuery,
        limit: usize,
    ) -> Result<GraphRecallBundle>;
    async fn consistency_checks(&self, limit: usize) -> Result<Vec<GraphReviewRecord>>;
}

#[async_trait]
pub trait Recall {
    async fn observe_now(&self, _now: &Now) -> Result<()> {
        Ok(())
    }

    async fn observe_frame(&self, frame: &ExperienceFrame) -> Result<()> {
        self.observe_now(&frame.now).await
    }

    async fn observe_transition(&self, _transition: &ExperienceTransition) -> Result<()> {
        Ok(())
    }

    async fn loop_closure_candidates(
        &self,
        _query: &RecallQuery,
        _min_confidence: f32,
        _limit: usize,
    ) -> Result<Vec<PlaceRecognitionCandidate>> {
        Ok(Vec::new())
    }

    async fn recall(&self, query: RecallQuery) -> Result<RecallBundle>;
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RecallQuery {
    pub now_text: String,
    pub pose: Option<Pose2>,
    pub scene_vector: Option<Vec<f32>>,
    #[serde(default)]
    pub scene_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub place_recognition_input: Option<PlaceRecognitionInput>,
    #[serde(default)]
    pub face_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub object_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub voice_vectors: Vec<VectorArtifact>,
    pub battery: f32,
    pub active_goal: Option<Goal>,
    pub proposed_action: Option<ActionPrimitive>,
}

impl RecallQuery {
    pub fn from_now(now: &Now) -> Self {
        Self {
            now_text: format!("t_ms={} battery={:.2}", now.t_ms, now.body.battery_level),
            pose: Some(now.body.odometry),
            scene_vector: now
                .eye
                .scene_vectors
                .last()
                .map(|artifact| artifact.vector.clone()),
            scene_vectors: now.eye.scene_vectors.clone(),
            place_recognition_input: None,
            face_vectors: now.face.vectors.clone(),
            object_vectors: now.objects.vectors.clone(),
            voice_vectors: now.voice.vectors.clone(),
            battery: now.body.battery_level,
            active_goal: now
                .self_sense
                .active_goal
                .as_ref()
                .map(|goal| match goal.as_str() {
                    "dock" => Goal::Dock,
                    "rest" => Goal::Rest,
                    "escape" => Goal::Escape,
                    "inspect" => Goal::Inspect,
                    "approach" => Goal::Approach,
                    "speak" => Goal::Speak,
                    _ => Goal::Explore,
                }),
            proposed_action: now.memory.best_remembered_action.clone(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RecallBundle {
    pub hits: Vec<RecallHit>,
    pub sense: MemorySense,
    pub first_person_summary: String,
    pub recollections: Vec<RecalledExperience>,
    #[serde(default)]
    pub semantic_map: Option<SemanticMapOverlay>,
    #[serde(default)]
    pub place_recognition_candidates: Vec<PlaceRecognitionCandidate>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PlaceCellKey {
    pub x: i32,
    pub y: i32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionOutcomeSummary {
    pub action: ActionPrimitive,
    pub count: u32,
    pub mean_reward: f32,
    pub last_seen_tick: u64,
}

impl Default for ActionOutcomeSummary {
    fn default() -> Self {
        Self {
            action: ActionPrimitive::Stop,
            count: 0,
            mean_reward: 0.0,
            last_seen_tick: 0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SemanticCell {
    pub key: PlaceCellKey,
    #[serde(default)]
    pub occupancy_cell: Option<PlaceCellKey>,
    pub center_x_m: f32,
    pub center_y_m: f32,
    pub visit_count: u32,
    pub last_seen_tick: u64,
    pub danger_score: f32,
    pub charge_score: f32,
    pub social_score: f32,
    pub novelty_score: f32,
    pub confidence: f32,
    #[serde(default)]
    pub last_observed_objects: Vec<String>,
    #[serde(default)]
    pub associated_scene_vectors: Vec<String>,
    #[serde(default)]
    pub associated_face_vectors: Vec<String>,
    #[serde(default)]
    pub associated_object_vectors: Vec<String>,
    #[serde(default)]
    pub associated_voice_vectors: Vec<String>,
    #[serde(default)]
    pub successful_actions: Vec<ActionOutcomeSummary>,
    #[serde(default)]
    pub failed_actions: Vec<ActionOutcomeSummary>,
}

pub type PlaceCell = SemanticCell;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlaceRecognitionKind {
    SamePlace,
    SimilarPlace,
    EntityConstellation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlaceRecognitionCandidate {
    pub kind: PlaceRecognitionKind,
    pub cell: PlaceCellSummary,
    pub source_vector_id: String,
    pub source_frame_id: Option<String>,
    #[serde(default)]
    pub source_experience_id: Option<String>,
    #[serde(default)]
    pub source_instant_frame_id: Option<String>,
    #[serde(default)]
    pub source_vector_refs: Vec<String>,
    pub query_vector_id: Option<String>,
    #[serde(default)]
    pub query_experience_id: Option<String>,
    pub similarity: f32,
    pub confidence: f32,
    #[serde(default)]
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlaceSceneEmbedding {
    pub cell_key: PlaceCellKey,
    pub artifact: VectorArtifact,
    #[serde(default)]
    pub input: Option<PlaceRecognitionInput>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlaceRecognitionInput {
    pub experience_id: Option<String>,
    pub instant_frame_id: Option<String>,
    pub experience_latent_vector: Option<VectorArtifact>,
    #[serde(default)]
    pub teacher_vector_refs: Vec<String>,
    #[serde(default)]
    pub compact_range_summary: Option<CompactRangeSummary>,
    #[serde(default)]
    pub compact_depth_summary: Option<CompactDepthSummary>,
    #[serde(default)]
    pub object_labels: Vec<String>,
    #[serde(default)]
    pub person_labels: Vec<String>,
    #[serde(default)]
    pub voice_labels: Vec<String>,
    pub action: Option<ActionPrimitive>,
    pub pose: Option<Pose2>,
    pub window_start_ms: u64,
    pub window_end_ms: u64,
    pub provenance: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CompactRangeSummary {
    pub beam_count: usize,
    pub nearest_m: Option<f32>,
    pub mean_m: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CompactDepthSummary {
    pub sample_count: usize,
    pub min_m: Option<f32>,
    pub max_m: Option<f32>,
    pub mean_m: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlaceRecognitionOutput {
    pub candidates: Vec<PlaceRecognitionCandidate>,
    pub rejected: Vec<PlaceRecognitionRejection>,
    pub not_enough_evidence: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlaceRecognitionRejection {
    pub source_vector_id: String,
    pub query_vector_id: Option<String>,
    pub similarity: f32,
    pub confidence: f32,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlaceMemoryConfig {
    pub cell_size_m: f32,
}

impl Default for PlaceMemoryConfig {
    fn default() -> Self {
        Self {
            cell_size_m: DEFAULT_CELL_SIZE_M,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlaceMemoryFeatures {
    pub current_place_danger: f32,
    pub current_place_charge: f32,
    pub current_place_social: f32,
    pub current_place_novelty: f32,
    pub current_place_familiarity: f32,
    pub current_place_confidence: f32,
    pub nearby_best_charge_direction_rad: Option<f32>,
    pub nearby_best_safe_direction_rad: Option<f32>,
    pub nearby_frontier_direction_rad: Option<f32>,
    pub places_visited: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlaceCellSummary {
    pub x: i32,
    pub y: i32,
    pub center_x_m: f32,
    pub center_y_m: f32,
    pub score: f32,
    pub visit_count: u32,
    pub last_seen_tick: u64,
    pub confidence: f32,
    #[serde(default)]
    pub last_observed_objects: Vec<String>,
    #[serde(default)]
    pub associated_scene_vectors: Vec<String>,
    #[serde(default)]
    pub associated_face_vectors: Vec<String>,
    #[serde(default)]
    pub associated_object_vectors: Vec<String>,
    #[serde(default)]
    pub associated_voice_vectors: Vec<String>,
    #[serde(default)]
    pub successful_actions: Vec<ActionOutcomeSummary>,
    #[serde(default)]
    pub failed_actions: Vec<ActionOutcomeSummary>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlaceMemoryReport {
    pub places_visited: usize,
    pub coverage_m2: f32,
    pub top_danger_cells: Vec<PlaceCellSummary>,
    pub top_charge_cells: Vec<PlaceCellSummary>,
    pub top_social_cells: Vec<PlaceCellSummary>,
    pub novelty_mean: f32,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SemanticMapOverlay {
    pub schema_version: u32,
    pub cell_size_m: f32,
    pub places_visited: usize,
    pub coverage_m2: f32,
    pub current: Option<PlaceCellSummary>,
    pub danger_cells: Vec<PlaceCellSummary>,
    pub charge_cells: Vec<PlaceCellSummary>,
    pub social_cells: Vec<PlaceCellSummary>,
    pub novelty_cells: Vec<PlaceCellSummary>,
    #[serde(default)]
    pub place_recognition_candidates: Vec<PlaceRecognitionCandidate>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlaceMemory {
    pub config: PlaceMemoryConfig,
    pub cells: BTreeMap<PlaceCellKey, PlaceCell>,
    #[serde(default)]
    pub scene_embeddings: BTreeMap<String, PlaceSceneEmbedding>,
    last_tick: Option<u64>,
}

impl Default for PlaceMemory {
    fn default() -> Self {
        Self {
            config: PlaceMemoryConfig::default(),
            cells: BTreeMap::new(),
            scene_embeddings: BTreeMap::new(),
            last_tick: None,
        }
    }
}

impl PlaceMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn quantize(&self, x_m: f32, y_m: f32) -> PlaceCellKey {
        let cell_size = self.config.cell_size_m.max(0.01);
        PlaceCellKey {
            x: (x_m / cell_size).floor() as i32,
            y: (y_m / cell_size).floor() as i32,
        }
    }

    pub fn observe_now(&mut self, now: &Now) -> PlaceMemoryFeatures {
        let elapsed_ticks = self
            .last_tick
            .map(|last| now.t_ms.saturating_sub(last) / 100)
            .unwrap_or(1)
            .max(1);
        self.decay(elapsed_ticks);
        self.last_tick = Some(now.t_ms);

        let key = self.quantize(now.body.odometry.x_m, now.body.odometry.y_m);
        let was_known = self.cells.contains_key(&key);
        let cell_size = self.config.cell_size_m.max(0.01);
        let danger_signal = danger_signal(now);
        let charge_signal = charge_signal(now);
        let social_signal = social_signal(now);
        let objects = observed_object_summary(now);
        let cell = self.cells.entry(key).or_insert_with(|| PlaceCell {
            key,
            occupancy_cell: Some(key),
            center_x_m: (key.x as f32 + 0.5) * cell_size,
            center_y_m: (key.y as f32 + 0.5) * cell_size,
            novelty_score: 1.0,
            ..PlaceCell::default()
        });
        cell.visit_count = cell.visit_count.saturating_add(1);
        cell.last_seen_tick = now.t_ms;
        cell.danger_score = cell.danger_score.max(danger_signal).clamp(0.0, 1.0);
        cell.charge_score = cell.charge_score.max(charge_signal).clamp(0.0, 1.0);
        cell.social_score = cell.social_score.max(social_signal).clamp(0.0, 1.0);
        cell.novelty_score = if was_known {
            (cell.novelty_score * 0.75).clamp(0.0, 1.0)
        } else {
            1.0
        };
        cell.confidence = (cell.confidence + 0.2).clamp(0.0, 1.0);
        if !objects.is_empty() {
            cell.last_observed_objects = objects;
        }
        merge_vector_ids(&mut cell.associated_scene_vectors, &now.eye.scene_vectors);
        merge_vector_ids(&mut cell.associated_face_vectors, &now.face.vectors);
        merge_vector_ids(&mut cell.associated_object_vectors, &now.objects.vectors);
        merge_vector_ids(&mut cell.associated_voice_vectors, &now.voice.vectors);
        let scene_vectors = scene_vectors_with_frame_id(
            &now.eye.scene_vectors,
            now.extensions
                .get("frame_id")
                .and_then(|value| value.as_str()),
        );
        self.store_scene_embeddings(key, &scene_vectors);
        self.features_at(now.body.odometry.x_m, now.body.odometry.y_m)
    }

    pub fn observe_frame(&mut self, frame: &ExperienceFrame) -> PlaceMemoryFeatures {
        let features = self.observe_now(&frame.now);
        let key = self.quantize(frame.now.body.odometry.x_m, frame.now.body.odometry.y_m);
        let scene_vectors = scene_vectors_from_now(&frame.now, frame.id, frame.t_ms);
        let place_input = place_recognition_input_from_frame(frame);
        let place_vectors = place_recognition_vectors_from_input(&place_input);
        if !scene_vectors.is_empty() {
            if let Some(cell) = self.cells.get_mut(&key) {
                merge_vector_ids(&mut cell.associated_scene_vectors, &scene_vectors);
            }
            self.store_scene_embeddings(key, &scene_vectors);
        }
        if !place_vectors.is_empty() {
            if let Some(cell) = self.cells.get_mut(&key) {
                merge_vector_ids(&mut cell.associated_scene_vectors, &place_vectors);
            }
            self.store_place_embeddings(key, &place_vectors, Some(place_input));
        }
        let Some(action) = frame.chosen_action.as_ref() else {
            return features;
        };
        self.observe_action_outcome(key, action, frame.reward.value, frame.now.t_ms);
        features
    }

    pub fn observe_transition(&mut self, transition: &ExperienceTransition) {
        let Some(action) = transition.action.as_ref() else {
            return;
        };
        let key = self.quantize(
            transition.before.body.odometry.x_m,
            transition.before.body.odometry.y_m,
        );
        self.observe_action_outcome(
            key,
            action,
            transition.reward.value,
            transition.created_at_ms,
        );
    }

    pub fn features_at(&self, x_m: f32, y_m: f32) -> PlaceMemoryFeatures {
        let key = self.quantize(x_m, y_m);
        let current = self.cells.get(&key);
        PlaceMemoryFeatures {
            current_place_danger: current.map(|cell| cell.danger_score).unwrap_or(0.0),
            current_place_charge: current.map(|cell| cell.charge_score).unwrap_or(0.0),
            current_place_social: current.map(|cell| cell.social_score).unwrap_or(0.0),
            current_place_novelty: current.map(|cell| cell.novelty_score).unwrap_or(1.0),
            current_place_familiarity: current
                .map(|cell| (cell.visit_count as f32 / 5.0).clamp(0.0, 1.0))
                .unwrap_or(0.0),
            current_place_confidence: current.map(|cell| cell.confidence).unwrap_or(0.0),
            nearby_best_charge_direction_rad: self
                .best_direction_from(key, x_m, y_m, |cell| cell.charge_score * cell.confidence),
            nearby_best_safe_direction_rad: self.best_direction_from(key, x_m, y_m, |cell| {
                (1.0 - cell.danger_score) * cell.confidence * (0.25 + cell.visit_count as f32)
            }),
            nearby_frontier_direction_rad: self.best_direction_from(key, x_m, y_m, |cell| {
                cell.novelty_score * (1.0 - cell.danger_score) * cell.confidence
            }),
            places_visited: self.cells.len().try_into().unwrap_or(u32::MAX),
        }
    }

    pub fn semantic_overlay_at(&self, x_m: f32, y_m: f32) -> SemanticMapOverlay {
        self.semantic_overlay_with_query(
            Some(self.quantize(x_m, y_m)),
            &[],
            PLACE_RECOGNITION_MIN_CONFIDENCE,
        )
    }

    pub fn semantic_overlay(&self, current_key: Option<PlaceCellKey>) -> SemanticMapOverlay {
        self.semantic_overlay_with_query(current_key, &[], PLACE_RECOGNITION_MIN_CONFIDENCE)
    }

    pub fn semantic_overlay_with_query(
        &self,
        current_key: Option<PlaceCellKey>,
        query_vectors: &[VectorArtifact],
        min_confidence: f32,
    ) -> SemanticMapOverlay {
        let report = self.report();
        SemanticMapOverlay {
            schema_version: 1,
            cell_size_m: self.config.cell_size_m,
            places_visited: report.places_visited,
            coverage_m2: report.coverage_m2,
            current: current_key
                .and_then(|key| self.cells.get(&key))
                .map(|cell| summarize_cell(cell, cell.confidence)),
            danger_cells: top_cells(&self.cells, |cell| cell.danger_score),
            charge_cells: top_cells(&self.cells, |cell| cell.charge_score),
            social_cells: top_cells(&self.cells, |cell| cell.social_score),
            novelty_cells: top_cells(&self.cells, |cell| cell.novelty_score),
            place_recognition_candidates: self.recognize_places(
                current_key,
                query_vectors,
                min_confidence,
                5,
            ),
        }
    }

    pub fn recognize_places(
        &self,
        current_key: Option<PlaceCellKey>,
        query_vectors: &[VectorArtifact],
        min_confidence: f32,
        limit: usize,
    ) -> Vec<PlaceRecognitionCandidate> {
        let mut candidates = Vec::new();
        let output =
            self.recognize_places_report(current_key, query_vectors, min_confidence, limit);
        candidates.extend(output.candidates);
        candidates
    }

    pub fn recognize_places_report(
        &self,
        current_key: Option<PlaceCellKey>,
        query_vectors: &[VectorArtifact],
        min_confidence: f32,
        limit: usize,
    ) -> PlaceRecognitionOutput {
        let mut candidates = Vec::new();
        let mut rejected = Vec::new();
        if query_vectors.iter().all(|query| query.vector.is_empty()) {
            return PlaceRecognitionOutput {
                candidates,
                rejected,
                not_enough_evidence: Some(
                    "no fused Experience latent or teacher vector was available".to_string(),
                ),
            };
        }
        if self.scene_embeddings.is_empty() {
            return PlaceRecognitionOutput {
                candidates,
                rejected,
                not_enough_evidence: Some(
                    "no stored place-recognition vectors have been observed".to_string(),
                ),
            };
        }
        for query in query_vectors {
            if query.vector.is_empty() {
                continue;
            }
            for stored in self.scene_embeddings.values() {
                let Some(cell) = self.cells.get(&stored.cell_key) else {
                    continue;
                };
                if stored.artifact.point_id == query.point_id {
                    continue;
                }
                let similarity = cosine_similarity(&query.vector, &stored.artifact.vector);
                let confidence =
                    (similarity * (0.5 + cell.confidence.clamp(0.0, 1.0) * 0.5)).clamp(0.0, 1.0);
                if confidence < min_confidence {
                    rejected.push(PlaceRecognitionRejection {
                        source_vector_id: stored.artifact.point_id.clone(),
                        query_vector_id: Some(query.point_id.clone()),
                        similarity,
                        confidence,
                        reason: format!(
                            "confidence {:.3} below threshold {:.3}",
                            confidence, min_confidence
                        ),
                    });
                    continue;
                }
                candidates.push(PlaceRecognitionCandidate {
                    kind: recognition_kind(current_key, stored.cell_key, similarity),
                    cell: summarize_cell(cell, confidence),
                    source_vector_id: stored.artifact.point_id.clone(),
                    source_frame_id: stored.artifact.source_frame_id.clone(),
                    source_experience_id: stored
                        .input
                        .as_ref()
                        .and_then(|input| input.experience_id.clone())
                        .or_else(|| stored.artifact.source_id.clone()),
                    source_instant_frame_id: stored
                        .input
                        .as_ref()
                        .and_then(|input| input.instant_frame_id.clone())
                        .or_else(|| stored.artifact.source_frame_id.clone()),
                    source_vector_refs: stored
                        .input
                        .as_ref()
                        .map(|input| input.teacher_vector_refs.clone())
                        .unwrap_or_default(),
                    query_vector_id: Some(query.point_id.clone()),
                    query_experience_id: query.source_id.clone(),
                    similarity,
                    confidence,
                    reason: candidate_reason(similarity, confidence, current_key, stored.cell_key),
                });
            }
        }
        candidates.sort_by(|left, right| {
            right
                .confidence
                .partial_cmp(&left.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.cell.last_seen_tick.cmp(&left.cell.last_seen_tick))
        });
        candidates.truncate(limit);
        rejected.sort_by(|left, right| {
            right
                .confidence
                .partial_cmp(&left.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rejected.truncate(limit);
        PlaceRecognitionOutput {
            not_enough_evidence: candidates
                .is_empty()
                .then_some("no stored place candidate met the confidence threshold".to_string()),
            candidates,
            rejected,
        }
    }

    /// Produce conservative loop-closure candidates by comparing current entity labels against
    /// previously observed cells' object labels using Jaccard overlap.  This complements the
    /// vector-embedding path and is intentionally conservative: the confidence is scaled down
    /// by a fixed factor so entity-constellation candidates only pass a high gate.
    pub fn recognize_entity_constellations(
        &self,
        current_key: Option<PlaceCellKey>,
        entity_labels: &[String],
        min_confidence: f32,
        limit: usize,
    ) -> Vec<PlaceRecognitionCandidate> {
        if entity_labels.is_empty() {
            return Vec::new();
        }
        let query_set: std::collections::BTreeSet<String> = entity_labels.iter().cloned().collect();
        let mut candidates = Vec::new();
        for (key, cell) in &self.cells {
            if current_key.as_ref() == Some(key) {
                continue;
            }
            if cell.last_observed_objects.is_empty() {
                continue;
            }
            let stored_set: std::collections::BTreeSet<String> =
                cell.last_observed_objects.iter().cloned().collect();
            let overlap = token_overlap(&query_set, &stored_set);
            if overlap <= 0.0 {
                continue;
            }
            // Conservative confidence: scale by cell confidence and a fixed 0.7 factor
            let confidence = (overlap * cell.confidence.clamp(0.0, 1.0) * 0.7).clamp(0.0, 1.0);
            if confidence < min_confidence {
                continue;
            }
            let source_vector_id = format!("entity-constellation:{}:{}", key.x, key.y);
            let shared_labels: Vec<String> = query_set.intersection(&stored_set).cloned().collect();
            candidates.push(PlaceRecognitionCandidate {
                kind: PlaceRecognitionKind::EntityConstellation,
                cell: summarize_cell(cell, confidence),
                source_vector_id,
                source_frame_id: None,
                source_experience_id: None,
                source_instant_frame_id: None,
                source_vector_refs: shared_labels,
                query_vector_id: None,
                query_experience_id: None,
                similarity: overlap,
                confidence,
                reason: format!(
                    "entity overlap {:.2} (shared: {}, stored: {}, query: {})",
                    overlap,
                    query_set.intersection(&stored_set).count(),
                    stored_set.len(),
                    query_set.len()
                ),
            });
        }
        candidates.sort_by(|left, right| {
            right
                .confidence
                .partial_cmp(&left.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.cell.last_seen_tick.cmp(&left.cell.last_seen_tick))
        });
        candidates.truncate(limit);
        candidates
    }

    pub fn report(&self) -> PlaceMemoryReport {
        let places_visited = self.cells.len();
        let coverage_m2 = places_visited as f32 * self.config.cell_size_m * self.config.cell_size_m;
        let novelty_mean = if places_visited == 0 {
            0.0
        } else {
            self.cells
                .values()
                .map(|cell| cell.novelty_score)
                .sum::<f32>()
                / places_visited as f32
        };
        let mut warnings = Vec::new();
        if places_visited == 0 {
            warnings.push("no place memory cells observed".to_string());
        }
        if !self.cells.values().any(|cell| cell.charge_score >= 0.5) {
            warnings.push("no strong charge memory cells".to_string());
        }
        if !self.cells.values().any(|cell| cell.danger_score >= 0.5) {
            warnings.push("no strong danger memory cells".to_string());
        }
        if !self.cells.values().any(|cell| cell.social_score >= 0.5) {
            warnings.push("no strong social memory cells".to_string());
        }
        PlaceMemoryReport {
            places_visited,
            coverage_m2,
            top_danger_cells: top_cells(&self.cells, |cell| cell.danger_score),
            top_charge_cells: top_cells(&self.cells, |cell| cell.charge_score),
            top_social_cells: top_cells(&self.cells, |cell| cell.social_score),
            novelty_mean,
            warnings,
        }
    }

    fn decay(&mut self, ticks: u64) {
        let score_factor = SCORE_DECAY_PER_TICK.powi(ticks.min(i32::MAX as u64) as i32);
        let confidence_factor =
            CELL_CONFIDENCE_DECAY_PER_TICK.powi(ticks.min(i32::MAX as u64) as i32);
        for cell in self.cells.values_mut() {
            cell.danger_score *= score_factor;
            cell.charge_score *= score_factor;
            cell.social_score *= score_factor;
            cell.confidence *= confidence_factor;
        }
    }

    fn observe_action_outcome(
        &mut self,
        key: PlaceCellKey,
        action: &ActionPrimitive,
        reward: f32,
        t_ms: u64,
    ) {
        if let Some(cell) = self.cells.get_mut(&key) {
            if reward >= 0.05 {
                update_action_outcome(&mut cell.successful_actions, action, reward, t_ms);
            } else if reward <= -0.05 {
                update_action_outcome(&mut cell.failed_actions, action, reward, t_ms);
            }
        }
    }

    fn store_scene_embeddings(&mut self, key: PlaceCellKey, artifacts: &[VectorArtifact]) {
        for artifact in artifacts {
            if artifact.point_id.trim().is_empty() || artifact.vector.is_empty() {
                continue;
            }
            self.scene_embeddings.insert(
                artifact.point_id.clone(),
                PlaceSceneEmbedding {
                    cell_key: key,
                    artifact: artifact.clone(),
                    input: None,
                },
            );
        }
        const MAX_PLACE_SCENE_EMBEDDINGS: usize = 512;
        while self.scene_embeddings.len() > MAX_PLACE_SCENE_EMBEDDINGS {
            let Some(oldest_key) = self.scene_embeddings.keys().next().cloned() else {
                break;
            };
            self.scene_embeddings.remove(&oldest_key);
        }
    }

    fn store_place_embeddings(
        &mut self,
        key: PlaceCellKey,
        artifacts: &[VectorArtifact],
        input: Option<PlaceRecognitionInput>,
    ) {
        for artifact in artifacts {
            if artifact.point_id.trim().is_empty() || artifact.vector.is_empty() {
                continue;
            }
            self.scene_embeddings.insert(
                artifact.point_id.clone(),
                PlaceSceneEmbedding {
                    cell_key: key,
                    artifact: artifact.clone(),
                    input: input.clone(),
                },
            );
        }
        const MAX_PLACE_SCENE_EMBEDDINGS: usize = 512;
        while self.scene_embeddings.len() > MAX_PLACE_SCENE_EMBEDDINGS {
            let Some(oldest_key) = self.scene_embeddings.keys().next().cloned() else {
                break;
            };
            self.scene_embeddings.remove(&oldest_key);
        }
    }

    fn best_direction_from(
        &self,
        key: PlaceCellKey,
        x_m: f32,
        y_m: f32,
        score: impl Fn(&PlaceCell) -> f32,
    ) -> Option<f32> {
        self.cells
            .values()
            .filter(|cell| {
                (cell.key.x - key.x).abs() <= RECALL_RADIUS_CELLS
                    && (cell.key.y - key.y).abs() <= RECALL_RADIUS_CELLS
                    && cell.key != key
            })
            .filter_map(|cell| {
                let value = score(cell);
                (value > 0.05).then_some((value, cell))
            })
            .max_by(|left, right| {
                left.0
                    .partial_cmp(&right.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, cell)| (cell.center_y_m - y_m).atan2(cell.center_x_m - x_m))
    }
}

/// How confident the system is that an entity is currently present.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityLifecycleState {
    /// Entity has been recently observed.
    #[default]
    Active,
    /// Entity was seen before but not in recent ticks; may return.
    Occluded,
    /// Entity has not been seen for a long time and is considered gone.
    Vanished,
}

/// Which sensing modalities have contributed evidence for this entity.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModalitySupport {
    /// Vector point IDs from the face/image collection.
    #[serde(default)]
    pub face_vector_ids: Vec<String>,
    /// Vector point IDs from the object identity/similarity collection.
    #[serde(default)]
    pub object_vector_ids: Vec<String>,
    /// Vector point IDs from the voice collection.
    #[serde(default)]
    pub voice_vector_ids: Vec<String>,
    /// Vector point IDs from the scene/depth collection.
    #[serde(default)]
    pub scene_vector_ids: Vec<String>,
    /// Free-form text labels contributed by LLM, captions, or human labels.
    #[serde(default)]
    pub text_labels: Vec<String>,
}

impl ModalitySupport {
    /// Number of distinct modalities that have contributed evidence.
    pub fn active_modalities(&self) -> usize {
        [
            !self.face_vector_ids.is_empty(),
            !self.object_vector_ids.is_empty(),
            !self.voice_vector_ids.is_empty(),
            !self.scene_vector_ids.is_empty(),
            !self.text_labels.is_empty(),
        ]
        .iter()
        .filter(|&&b| b)
        .count()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingRelation {
    #[default]
    CooccursInTime,
    CooccursInEstimatedSpace,
    MovesTogether,
    PredictsSameFutureEvents,
    NamedBy,
    ProjectsTo,
    HasColorAtPose,
    LikelySameEntity,
    ExplainsOutcome,
    Contradicts,
    RequiresReview,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BindingDecision {
    Accept,
    Reject,
    HoldAmbiguous,
    AskHuman,
    CollectMoreEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BindingEvidenceKind {
    TemporalOverlap,
    SpatialOverlap,
    VectorSimilarity,
    ProjectionAgreement,
    PoseAgreement,
    RepeatedCooccurrence,
    SingleCandidateContext,
    HumanConfirmed,
    LlmSuggested,
    Contradiction,
    SimultaneousConflict,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BindingEvidence {
    pub kind: BindingEvidenceKind,
    pub score: f32,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BindingCandidate {
    pub left_cluster_id: String,
    pub right_cluster_id: String,
    pub relation: BindingRelation,
    pub evidence: Vec<BindingEvidence>,
    pub confidence: f32,
    pub decision: BindingDecision,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrackingHypothesisKind {
    FaceIdentity,
    VoiceIdentity,
    CrossModalBinding,
    PlaceMatch,
    ObjectContinuity,
    #[default]
    Other,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisState {
    #[default]
    Active,
    Winning,
    Losing,
    NeedsReview,
    Rejected,
    Promoted,
    Expired,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackingHypothesis {
    pub id: String,
    pub family_id: String,
    pub kind: TrackingHypothesisKind,
    pub target_id: Option<String>,
    #[serde(default)]
    pub observation_ids: Vec<String>,
    #[serde(default)]
    pub binding_candidate_ids: Vec<String>,
    pub confidence: f32,
    #[serde(default)]
    pub evidence: Vec<BindingEvidence>,
    #[serde(default)]
    pub contradictions: Vec<String>,
    pub state: HypothesisState,
    pub first_seen_ms: u64,
    pub last_updated_ms: u64,
}

impl TrackingHypothesis {
    fn new(
        kind: TrackingHypothesisKind,
        family_id: String,
        target_id: Option<String>,
        observation_id: String,
        candidate_id: String,
        evidence: Vec<BindingEvidence>,
        t_ms: u64,
    ) -> Self {
        let target_slug = target_id
            .as_deref()
            .map(stable_slug)
            .unwrap_or_else(|| "new-entity".to_string());
        let id = format!(
            "hypothesis:{}:{}:{}",
            tracking_kind_slug(&kind),
            stable_slug(&family_id),
            target_slug
        );
        let mut hypothesis = Self {
            id,
            family_id,
            kind,
            target_id,
            observation_ids: vec![observation_id],
            binding_candidate_ids: vec![candidate_id],
            confidence: 0.0,
            evidence: Vec::new(),
            contradictions: Vec::new(),
            state: HypothesisState::Active,
            first_seen_ms: t_ms,
            last_updated_ms: t_ms,
        };
        hypothesis.add_evidence(evidence, t_ms);
        hypothesis
    }

    fn add_evidence(&mut self, evidence: Vec<BindingEvidence>, t_ms: u64) {
        self.last_updated_ms = t_ms;
        let previous_observations = self.observation_ids.len().max(1) as f32;
        self.evidence.extend(evidence);
        for item in &self.evidence {
            if matches!(
                item.kind,
                BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
            ) && !self.contradictions.contains(&item.reason)
            {
                self.contradictions.push(item.reason.clone());
            }
        }
        self.confidence = score_hypothesis_evidence(&self.evidence, previous_observations);
        if self.state != HypothesisState::Promoted && self.state != HypothesisState::Rejected {
            self.state = if has_hard_contradiction(&self.evidence) {
                HypothesisState::Rejected
            } else if !self.contradictions.is_empty() {
                HypothesisState::NeedsReview
            } else {
                HypothesisState::Active
            };
        }
    }
}

pub type ClusterId = String;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveredClusterKind {
    Face,
    Voice,
    RgbImage,
    Geometry,
    Object,
    Place,
    Action,
    Outcome,
    Label,
    BodyState,
    #[default]
    Other,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiscoveredCluster {
    pub id: ClusterId,
    pub modality: Modality,
    pub kind: DiscoveredClusterKind,
    pub first_seen_ms: u64,
    pub last_seen_ms: u64,
    pub confidence: f32,
    #[serde(default)]
    pub feature_ids: Vec<FeatureId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_frame_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub place_cell: Option<PlaceCellKey>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_pose: Option<Pose2>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

impl DiscoveredCluster {
    pub fn new(
        id: impl Into<String>,
        modality: Modality,
        kind: DiscoveredClusterKind,
        t_ms: u64,
        confidence: f32,
    ) -> Self {
        Self {
            id: id.into(),
            modality,
            kind,
            first_seen_ms: t_ms,
            last_seen_ms: t_ms,
            confidence: confidence.clamp(0.0, 1.0),
            feature_ids: Vec::new(),
            source_frame_id: None,
            place_cell: None,
            estimated_pose: None,
            metadata: serde_json::Value::Null,
        }
    }

    pub fn with_time_span(mut self, first_seen_ms: u64, last_seen_ms: u64) -> Self {
        self.first_seen_ms = first_seen_ms;
        self.last_seen_ms = last_seen_ms;
        self
    }

    pub fn with_source_frame_id(mut self, source_frame_id: impl Into<String>) -> Self {
        self.source_frame_id = Some(source_frame_id.into());
        self
    }

    pub fn with_place_cell(mut self, place_cell: PlaceCellKey) -> Self {
        self.place_cell = Some(place_cell);
        self
    }

    pub fn with_estimated_pose(mut self, estimated_pose: Pose2) -> Self {
        self.estimated_pose = Some(estimated_pose);
        self
    }

    pub fn with_feature_ids(mut self, feature_ids: Vec<FeatureId>) -> Self {
        self.feature_ids = feature_ids;
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BindingContext {
    pub t_ms: u64,
    pub time_window_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robot_pose: Option<Pose2>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_action: Option<ActionPrimitive>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_state: Option<BodySense>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_frame_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_place_cell: Option<PlaceCellKey>,
    #[serde(default)]
    pub recent_features: Vec<FeatureId>,
    #[serde(default)]
    pub recent_clusters: Vec<ClusterId>,
}

impl BindingContext {
    pub fn new(t_ms: u64) -> Self {
        Self {
            t_ms,
            time_window_ms: 1_000,
            ..Self::default()
        }
    }
}

pub trait CrossModalBindingEngine {
    fn propose_bindings(
        &mut self,
        context: &BindingContext,
        clusters: &[DiscoveredCluster],
    ) -> Vec<BindingCandidate>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct DefaultCrossModalBindingEngine {
    pub projection_error_threshold_px: f32,
    pub pose_distance_threshold_m: f32,
    pub action_outcome_min_lag_ms: u64,
    pub action_outcome_max_lag_ms: u64,
}

impl Default for DefaultCrossModalBindingEngine {
    fn default() -> Self {
        Self {
            projection_error_threshold_px: 5.0,
            pose_distance_threshold_m: 0.75,
            action_outcome_min_lag_ms: 50,
            action_outcome_max_lag_ms: 2_500,
        }
    }
}

impl CrossModalBindingEngine for DefaultCrossModalBindingEngine {
    fn propose_bindings(
        &mut self,
        context: &BindingContext,
        clusters: &[DiscoveredCluster],
    ) -> Vec<BindingCandidate> {
        let mut candidates = Vec::new();
        for left_index in 0..clusters.len() {
            for right_index in (left_index + 1)..clusters.len() {
                let left = &clusters[left_index];
                let right = &clusters[right_index];
                if left.id == right.id || left.modality == right.modality {
                    continue;
                }
                if let Some(candidate) = self.propose_pair(context, left, right, clusters) {
                    candidates.push(candidate);
                }
            }
        }
        candidates
    }
}

impl DefaultCrossModalBindingEngine {
    fn propose_pair(
        &self,
        context: &BindingContext,
        left: &DiscoveredCluster,
        right: &DiscoveredCluster,
        all_clusters: &[DiscoveredCluster],
    ) -> Option<BindingCandidate> {
        match (&left.kind, &right.kind) {
            (DiscoveredClusterKind::Face, DiscoveredClusterKind::Voice)
            | (DiscoveredClusterKind::Voice, DiscoveredClusterKind::Face) => {
                Some(self.face_voice_candidate(context, left, right, all_clusters))
            }
            (DiscoveredClusterKind::RgbImage, DiscoveredClusterKind::Geometry)
            | (DiscoveredClusterKind::Geometry, DiscoveredClusterKind::RgbImage) => {
                self.rgb_geometry_candidate(context, left, right)
            }
            (DiscoveredClusterKind::Object, DiscoveredClusterKind::Place)
            | (DiscoveredClusterKind::Place, DiscoveredClusterKind::Object) => {
                Some(self.object_place_candidate(context, left, right))
            }
            (DiscoveredClusterKind::Action, DiscoveredClusterKind::Outcome)
            | (DiscoveredClusterKind::Outcome, DiscoveredClusterKind::Action)
            | (DiscoveredClusterKind::Action, DiscoveredClusterKind::BodyState)
            | (DiscoveredClusterKind::BodyState, DiscoveredClusterKind::Action) => {
                self.action_outcome_candidate(context, left, right)
            }
            (DiscoveredClusterKind::Label, _) | (_, DiscoveredClusterKind::Label) => {
                Some(self.label_cluster_candidate(context, left, right))
            }
            _ => None,
        }
    }

    fn face_voice_candidate(
        &self,
        context: &BindingContext,
        left: &DiscoveredCluster,
        right: &DiscoveredCluster,
        all_clusters: &[DiscoveredCluster],
    ) -> BindingCandidate {
        let mut evidence = base_cross_modal_evidence(context, left, right);
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::VectorSimilarity,
            score: left.confidence.min(right.confidence).clamp(0.0, 1.0),
            reason: "face and voice clusters propose a possible person correspondence".to_string(),
        });
        add_recent_cooccurrence(context, left, right, &mut evidence);
        add_label_support(left, right, &mut evidence);

        let plausible_faces = all_clusters
            .iter()
            .filter(|cluster| cluster.kind == DiscoveredClusterKind::Face)
            .filter(|cluster| temporally_compatible(context, cluster, right))
            .count();
        let plausible_voices = all_clusters
            .iter()
            .filter(|cluster| cluster.kind == DiscoveredClusterKind::Voice)
            .filter(|cluster| temporally_compatible(context, cluster, left))
            .count();
        if plausible_faces == 1 || plausible_voices == 1 {
            evidence.push(BindingEvidence {
                kind: BindingEvidenceKind::SingleCandidateContext,
                score: 0.65,
                reason: "only one plausible face or voice cluster is active in the binding window"
                    .to_string(),
            });
        } else if plausible_faces > 1 && plausible_voices > 1 {
            evidence.push(BindingEvidence {
                kind: BindingEvidenceKind::SimultaneousConflict,
                score: 0.85,
                reason:
                    "multiple face and voice clusters are active; speaker identity is ambiguous"
                        .to_string(),
            });
        }

        proposal_candidate_from_evidence(
            left,
            right,
            BindingRelation::LikelySameEntity,
            evidence,
            "face/voice binding proposal",
        )
    }

    fn rgb_geometry_candidate(
        &self,
        context: &BindingContext,
        left: &DiscoveredCluster,
        right: &DiscoveredCluster,
    ) -> Option<BindingCandidate> {
        let mut evidence = base_cross_modal_evidence(context, left, right);
        let projection_error = projection_error_px(left, right);
        if let Some(error) = projection_error {
            if error <= self.projection_error_threshold_px {
                evidence.push(BindingEvidence {
                    kind: BindingEvidenceKind::ProjectionAgreement,
                    score: (1.0 - error / self.projection_error_threshold_px).clamp(0.0, 1.0),
                    reason: format!("RGB and geometry projections agree within {error:.2} px"),
                });
            } else {
                evidence.push(BindingEvidence {
                    kind: BindingEvidenceKind::Contradiction,
                    score: (error / self.projection_error_threshold_px).clamp(0.0, 1.0),
                    reason: format!("RGB/depth reprojection error {error:.2} px exceeds threshold"),
                });
            }
        }
        if let Some(distance) = pose_distance_m(left, right) {
            if distance <= self.pose_distance_threshold_m {
                evidence.push(BindingEvidence {
                    kind: BindingEvidenceKind::PoseAgreement,
                    score: (1.0 - distance / self.pose_distance_threshold_m).clamp(0.0, 1.0),
                    reason: format!("RGB and geometry world poses agree within {distance:.2} m"),
                });
            }
        }
        add_recent_cooccurrence(context, left, right, &mut evidence);

        let has_projection_or_pose_agreement = evidence
            .iter()
            .any(|evidence| evidence.kind == BindingEvidenceKind::ProjectionAgreement)
            || evidence
                .iter()
                .any(|evidence| evidence.kind == BindingEvidenceKind::PoseAgreement);
        let has_projection_contradiction = evidence
            .iter()
            .any(|evidence| evidence.kind == BindingEvidenceKind::Contradiction);
        if has_projection_or_pose_agreement || has_projection_contradiction {
            Some(proposal_candidate_from_evidence(
                left,
                right,
                BindingRelation::ProjectsTo,
                evidence,
                "RGB/geometry correspondence proposal",
            ))
        } else {
            None
        }
    }

    fn object_place_candidate(
        &self,
        context: &BindingContext,
        left: &DiscoveredCluster,
        right: &DiscoveredCluster,
    ) -> BindingCandidate {
        let mut evidence = base_cross_modal_evidence(context, left, right);
        if left.place_cell.is_some()
            && right.place_cell.is_some()
            && left.place_cell == right.place_cell
        {
            evidence.push(BindingEvidence {
                kind: BindingEvidenceKind::SpatialOverlap,
                score: 0.85,
                reason: "object and place cluster share a place cell".to_string(),
            });
        } else if context
            .current_place_cell
            .is_some_and(|cell| left.place_cell == Some(cell) || right.place_cell == Some(cell))
        {
            evidence.push(BindingEvidence {
                kind: BindingEvidenceKind::SpatialOverlap,
                score: 0.65,
                reason: "one cluster is compatible with the current place cell".to_string(),
            });
        }
        if metadata_bool(left, "moves_independently") || metadata_bool(right, "moves_independently")
        {
            evidence.push(BindingEvidence {
                kind: BindingEvidenceKind::Contradiction,
                score: 0.7,
                reason: "object cluster has evidence of independent motion".to_string(),
            });
        }
        add_repetition_evidence(left, right, &mut evidence);
        add_recent_cooccurrence(context, left, right, &mut evidence);

        proposal_candidate_from_evidence(
            left,
            right,
            BindingRelation::CooccursInEstimatedSpace,
            evidence,
            "object/place binding proposal",
        )
    }

    fn action_outcome_candidate(
        &self,
        context: &BindingContext,
        left: &DiscoveredCluster,
        right: &DiscoveredCluster,
    ) -> Option<BindingCandidate> {
        let (action, outcome) = if left.kind == DiscoveredClusterKind::Action {
            (left, right)
        } else {
            (right, left)
        };
        let lag_ms = outcome.first_seen_ms.saturating_sub(action.last_seen_ms);
        if lag_ms < self.action_outcome_min_lag_ms || lag_ms > self.action_outcome_max_lag_ms {
            return None;
        }

        let mut evidence = Vec::new();
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::TemporalOverlap,
            score: lag_score(
                lag_ms,
                self.action_outcome_min_lag_ms,
                self.action_outcome_max_lag_ms,
            ),
            reason: format!("outcome followed action after {lag_ms} ms"),
        });
        if context.active_action.is_some() {
            evidence.push(BindingEvidence {
                kind: BindingEvidenceKind::PoseAgreement,
                score: 0.55,
                reason: "binding context includes the active action that produced this window"
                    .to_string(),
            });
        }
        if let Some(body_state) = &context.body_state {
            if body_state.charging
                || body_state.flags.wheel_drop
                || body_state.flags.bump_left
                || body_state.flags.bump_right
            {
                evidence.push(BindingEvidence {
                    kind: BindingEvidenceKind::RepeatedCooccurrence,
                    score: 0.65,
                    reason: "body state contains concrete outcome evidence".to_string(),
                });
            }
        }
        add_repetition_evidence(left, right, &mut evidence);

        Some(proposal_candidate_from_evidence(
            action,
            outcome,
            BindingRelation::ExplainsOutcome,
            evidence,
            "action/outcome binding proposal",
        ))
    }

    fn label_cluster_candidate(
        &self,
        context: &BindingContext,
        left: &DiscoveredCluster,
        right: &DiscoveredCluster,
    ) -> BindingCandidate {
        let mut evidence = base_cross_modal_evidence(context, left, right);
        let trusted =
            metadata_bool(left, "trusted_source") || metadata_bool(right, "trusted_source");
        let llm = metadata_string(left, "source").as_deref() == Some("llm")
            || metadata_string(right, "source").as_deref() == Some("llm");
        if trusted {
            evidence.push(BindingEvidence {
                kind: BindingEvidenceKind::HumanConfirmed,
                score: 0.9,
                reason: "label came from a trusted source".to_string(),
            });
        } else if llm {
            evidence.push(BindingEvidence {
                kind: BindingEvidenceKind::LlmSuggested,
                score: 0.45,
                reason: "LLM label suggests this correspondence but needs support".to_string(),
            });
        }
        add_repetition_evidence(left, right, &mut evidence);
        add_recent_cooccurrence(context, left, right, &mut evidence);

        proposal_candidate_from_evidence(
            left,
            right,
            BindingRelation::NamedBy,
            evidence,
            "label/cluster binding proposal",
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BindingEdgeResult {
    pub edge: BindingEdge,
    pub created: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationPoint {
    pub id: String,
    pub modality: Modality,
    pub source: String,
    pub observed_at_ms: u64,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModalityCluster {
    pub id: String,
    pub modality: Modality,
    #[serde(default)]
    pub observation_point_ids: Vec<String>,
    pub evidence_count: u32,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BindingEdge {
    pub left_cluster_id: String,
    pub right_cluster_id: String,
    pub relation: BindingRelation,
    pub confidence: f32,
    pub evidence_count: u32,
    pub decay_per_tick: f32,
    pub last_seen_ms: u64,
}

impl BindingEdge {
    fn strengthen(&mut self, evidence: f32, t_ms: u64) {
        self.evidence_count = self.evidence_count.saturating_add(1);
        self.last_seen_ms = t_ms;
        self.confidence = (self.confidence + evidence.clamp(0.0, 1.0) * 0.2).clamp(0.0, 1.0);
    }

    fn weaken(&mut self, amount: f32) {
        self.confidence = (self.confidence * (1.0 - amount.clamp(0.0, 1.0))).clamp(0.0, 1.0);
    }

    pub fn is_strong(&self) -> bool {
        self.confidence >= 0.6
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityConstellationState {
    #[default]
    Weak,
    Strong,
    Merged,
    Split,
    Vanished,
    Revived,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EntityConstellation {
    #[serde(default)]
    pub observation_points: Vec<ObservationPoint>,
    #[serde(default)]
    pub modality_clusters: Vec<ModalityCluster>,
    #[serde(default)]
    pub binding_edges: Vec<BindingEdge>,
    #[serde(default)]
    pub binding_candidates: Vec<BindingCandidate>,
    pub state: EntityConstellationState,
    #[serde(default)]
    pub merged_entity_ids: Vec<String>,
    #[serde(default)]
    pub split_entity_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstellationKind {
    Person,
    Place,
    Object,
    Episode,
    Affordance,
    RiskPattern,
    ActionOutcome,
    #[default]
    Unknown,
}

impl ConstellationKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Person => "person",
            Self::Place => "place",
            Self::Object => "object",
            Self::Episode => "episode",
            Self::Affordance => "affordance",
            Self::RiskPattern => "risk_pattern",
            Self::ActionOutcome => "action_outcome",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstellationState {
    #[default]
    Candidate,
    Stable,
    Ambiguous,
    SplitNeeded,
    MergeNeeded,
    Retired,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Constellation {
    pub id: String,
    pub kind_hint: Option<String>,
    #[serde(default)]
    pub member_cluster_ids: Vec<String>,
    #[serde(default)]
    pub member_binding_ids: Vec<String>,
    #[serde(default)]
    pub supporting_feature_ids: Vec<FeatureId>,
    #[serde(default)]
    pub supporting_entity_ids: Vec<String>,
    #[serde(default)]
    pub supporting_place_cells: Vec<PlaceCellKey>,
    pub confidence: f32,
    pub stability: f32,
    pub prediction_value: f32,
    pub first_seen_ms: u64,
    pub last_seen_ms: u64,
    pub evidence_count: u32,
    pub state: ConstellationState,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstellationObservation {
    pub t_ms: u64,
    #[serde(default)]
    pub clusters: Vec<DiscoveredCluster>,
    #[serde(default)]
    pub accepted_bindings: Vec<BindingCandidate>,
    #[serde(default)]
    pub active_entity_ids: Vec<String>,
    #[serde(default)]
    pub place_cells: Vec<PlaceCellKey>,
    #[serde(default)]
    pub action_outcome_ids: Vec<String>,
    #[serde(default)]
    pub prediction_error_ids: Vec<String>,
    pub prediction_value: f32,
    #[serde(default)]
    pub llm_notes: Vec<String>,
}

impl Default for ConstellationObservation {
    fn default() -> Self {
        Self {
            t_ms: 0,
            clusters: Vec::new(),
            accepted_bindings: Vec::new(),
            active_entity_ids: Vec::new(),
            place_cells: Vec::new(),
            action_outcome_ids: Vec::new(),
            prediction_error_ids: Vec::new(),
            prediction_value: 0.0,
            llm_notes: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConstellationQuery {
    pub t_ms: u64,
    #[serde(default)]
    pub cluster_ids: Vec<String>,
    #[serde(default)]
    pub binding_ids: Vec<String>,
    #[serde(default)]
    pub feature_ids: Vec<FeatureId>,
    #[serde(default)]
    pub entity_ids: Vec<String>,
    #[serde(default)]
    pub place_cells: Vec<PlaceCellKey>,
    #[serde(default)]
    pub contradiction_ids: Vec<String>,
}

impl ConstellationQuery {
    pub fn from_observation(observation: &ConstellationObservation) -> Self {
        Self {
            t_ms: observation.t_ms,
            cluster_ids: observation
                .clusters
                .iter()
                .map(|cluster| cluster.id.clone())
                .collect(),
            binding_ids: observation
                .accepted_bindings
                .iter()
                .filter(|candidate| candidate.decision == BindingDecision::Accept)
                .map(binding_candidate_id)
                .collect(),
            feature_ids: observation
                .clusters
                .iter()
                .flat_map(|cluster| cluster.feature_ids.iter().copied())
                .collect(),
            entity_ids: observation.active_entity_ids.clone(),
            place_cells: observation.place_cells.clone(),
            contradiction_ids: observation
                .accepted_bindings
                .iter()
                .filter(|candidate| binding_has_conflict(candidate))
                .map(binding_candidate_id)
                .collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstellationMatch {
    pub constellation_id: String,
    pub score: f32,
    pub matched_cluster_ids: Vec<String>,
    pub matched_binding_ids: Vec<String>,
    pub missing_cluster_ids: Vec<String>,
    pub stale_penalty: f32,
    pub contradiction_penalty: f32,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstellationEngineConfig {
    pub promotion_confidence_threshold: f32,
    pub min_evidence_for_stable: u32,
    pub min_clusters_for_stable: usize,
    pub min_bindings_for_stable: usize,
    pub min_prediction_value_for_stable: f32,
    pub partial_match_threshold: f32,
    pub stale_after_ms: u64,
}

impl Default for ConstellationEngineConfig {
    fn default() -> Self {
        Self {
            promotion_confidence_threshold: 0.68,
            min_evidence_for_stable: 3,
            min_clusters_for_stable: 2,
            min_bindings_for_stable: 2,
            min_prediction_value_for_stable: 0.1,
            partial_match_threshold: 0.35,
            stale_after_ms: 60_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstellationEngine {
    pub constellations: BTreeMap<String, Constellation>,
    pub config: ConstellationEngineConfig,
    next_id: u64,
}

impl Default for ConstellationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstellationEngine {
    pub fn new() -> Self {
        Self {
            constellations: BTreeMap::new(),
            config: ConstellationEngineConfig::default(),
            next_id: 1,
        }
    }

    pub fn with_config(config: ConstellationEngineConfig) -> Self {
        Self {
            constellations: BTreeMap::new(),
            config,
            next_id: 1,
        }
    }

    pub fn observe(&mut self, observation: ConstellationObservation) -> Option<Constellation> {
        let accepted_bindings = observation
            .accepted_bindings
            .iter()
            .filter(|candidate| candidate.decision == BindingDecision::Accept)
            .collect::<Vec<_>>();
        let member_cluster_ids = cluster_ids_from_observation(&observation, &accepted_bindings);
        let member_binding_ids = accepted_bindings
            .iter()
            .map(|candidate| binding_candidate_id(candidate))
            .collect::<Vec<_>>();
        if member_cluster_ids.len() < 2 || member_binding_ids.is_empty() {
            return None;
        }

        let query = ConstellationQuery::from_observation(&observation);
        let match_id = self
            .best_match(&query)
            .filter(|matched| matched.score >= self.config.partial_match_threshold)
            .map(|matched| matched.constellation_id);

        let id = if let Some(id) = match_id {
            id
        } else {
            self.allocate_constellation_id(&member_cluster_ids)
        };
        if let Some(existing) = self.constellations.get_mut(&id) {
            merge_constellation_observation(
                existing,
                &observation,
                &member_cluster_ids,
                &member_binding_ids,
            );
        } else {
            let constellation = Constellation {
                id: id.clone(),
                kind_hint: infer_constellation_kind(&observation.clusters)
                    .map(|kind| kind.as_str().to_string()),
                member_cluster_ids,
                member_binding_ids,
                supporting_feature_ids: query.feature_ids,
                supporting_entity_ids: observation.active_entity_ids.clone(),
                supporting_place_cells: observation.place_cells.clone(),
                confidence: 0.0,
                stability: 0.0,
                prediction_value: observation.prediction_value.clamp(0.0, 1.0),
                first_seen_ms: observation.t_ms,
                last_seen_ms: observation.t_ms,
                evidence_count: 1,
                state: ConstellationState::Candidate,
                notes: observation.llm_notes.clone(),
            };
            self.constellations.insert(id.clone(), constellation);
        }

        let config = self.config.clone();
        let constellation = self.constellations.get_mut(&id)?;
        refresh_constellation_scores(constellation, &observation, &config);
        Some(constellation.clone())
    }

    pub fn best_match(&self, query: &ConstellationQuery) -> Option<ConstellationMatch> {
        self.matches(query, 1).into_iter().next()
    }

    pub fn matches(&self, query: &ConstellationQuery, limit: usize) -> Vec<ConstellationMatch> {
        let mut matches = self
            .constellations
            .values()
            .filter(|constellation| constellation.state != ConstellationState::Retired)
            .filter_map(|constellation| self.score_match(constellation, query))
            .collect::<Vec<_>>();
        matches.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(limit);
        matches
    }

    fn score_match(
        &self,
        constellation: &Constellation,
        query: &ConstellationQuery,
    ) -> Option<ConstellationMatch> {
        let constellation_clusters = constellation
            .member_cluster_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let query_clusters = query.cluster_ids.iter().cloned().collect::<BTreeSet<_>>();
        let matched_cluster_ids = constellation_clusters
            .intersection(&query_clusters)
            .cloned()
            .collect::<Vec<_>>();
        let missing_cluster_ids = constellation_clusters
            .difference(&query_clusters)
            .cloned()
            .collect::<Vec<_>>();

        let constellation_bindings = constellation
            .member_binding_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let query_bindings = query.binding_ids.iter().cloned().collect::<BTreeSet<_>>();
        let matched_binding_ids = constellation_bindings
            .intersection(&query_bindings)
            .cloned()
            .collect::<Vec<_>>();

        let cluster_score = overlap_score(
            matched_cluster_ids.len(),
            constellation.member_cluster_ids.len(),
        );
        let binding_score = overlap_score(
            matched_binding_ids.len(),
            constellation.member_binding_ids.len(),
        );
        let feature_score = overlap_score(
            intersection_count(&constellation.supporting_feature_ids, &query.feature_ids),
            constellation.supporting_feature_ids.len(),
        );
        let entity_score = overlap_score(
            intersection_count(&constellation.supporting_entity_ids, &query.entity_ids),
            constellation.supporting_entity_ids.len(),
        );
        let place_score = overlap_score(
            intersection_count(&constellation.supporting_place_cells, &query.place_cells),
            constellation.supporting_place_cells.len(),
        );

        let evidence_score = cluster_score * 0.45
            + binding_score * 0.3
            + place_score * 0.1
            + entity_score * 0.08
            + feature_score * 0.07;
        if evidence_score <= 0.0 {
            return None;
        }

        let stale_penalty = stale_penalty(
            query.t_ms.saturating_sub(constellation.last_seen_ms),
            self.config.stale_after_ms,
        );
        let contradiction_penalty = overlap_score(
            intersection_count(&constellation.member_binding_ids, &query.contradiction_ids),
            constellation.member_binding_ids.len(),
        ) * 0.45;
        let score = (evidence_score * (1.0 - stale_penalty) * (1.0 - contradiction_penalty))
            .clamp(0.0, 1.0);
        let reason = if matched_binding_ids.is_empty() && !matched_cluster_ids.is_empty() {
            "partial cluster match without all known bindings".to_string()
        } else if !missing_cluster_ids.is_empty() {
            "partial match with missing modalities".to_string()
        } else {
            "constellation evidence matches query".to_string()
        };
        Some(ConstellationMatch {
            constellation_id: constellation.id.clone(),
            score,
            matched_cluster_ids,
            matched_binding_ids,
            missing_cluster_ids,
            stale_penalty,
            contradiction_penalty,
            reason,
        })
    }

    fn allocate_constellation_id(&mut self, cluster_ids: &[String]) -> String {
        let first = cluster_ids
            .first()
            .map(|id| stable_slug(id))
            .filter(|slug| !slug.is_empty())
            .unwrap_or_else(|| "unknown".to_string());
        let id = format!("constellation:{}:{}", self.next_id, first);
        self.next_id = self.next_id.saturating_add(1);
        id
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssociationItemKind {
    Feature,
    Cluster,
    Binding,
    Constellation,
    Action,
    Outcome,
    BodyState,
    Prediction,
    Surprise,
    Memory,
    LlmNote,
    #[default]
    Other,
}

impl AssociationItemKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Feature => "feature",
            Self::Cluster => "cluster",
            Self::Binding => "binding",
            Self::Constellation => "constellation",
            Self::Action => "action",
            Self::Outcome => "outcome",
            Self::BodyState => "body_state",
            Self::Prediction => "prediction",
            Self::Surprise => "surprise",
            Self::Memory => "memory",
            Self::LlmNote => "llm_note",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssociationItem {
    pub id: String,
    pub kind: AssociationItemKind,
    pub confidence: f32,
}

impl AssociationItem {
    pub fn new(id: impl Into<String>, kind: AssociationItemKind, confidence: f32) -> Self {
        Self {
            id: id.into(),
            kind,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssociationRelation {
    #[default]
    CoOccursWith,
    Predicts,
    Follows,
    Suppresses,
    Contradicts,
    Explains,
    Enables,
    Prevents,
    PartOf,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssociationExample {
    pub frame_id: Option<String>,
    pub t_ms: u64,
    pub reason: String,
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssociationEdge {
    pub id: String,
    pub from_id: String,
    pub to_id: String,
    pub relation: AssociationRelation,
    pub confidence: f32,
    pub evidence_count: u32,
    pub prediction_gain: f32,
    pub contradiction_count: u32,
    pub first_seen_ms: u64,
    pub last_seen_ms: u64,
    #[serde(default)]
    pub examples: Vec<AssociationExample>,
}

impl AssociationEdge {
    fn new(
        from_id: String,
        to_id: String,
        relation: AssociationRelation,
        example: AssociationExample,
    ) -> Self {
        let id = association_edge_id(&from_id, &to_id, &relation);
        Self {
            id,
            from_id,
            to_id,
            relation,
            confidence: 0.0,
            evidence_count: 0,
            prediction_gain: 0.0,
            contradiction_count: 0,
            first_seen_ms: example.t_ms,
            last_seen_ms: example.t_ms,
            examples: Vec::new(),
        }
    }

    fn strengthen(&mut self, example: AssociationExample, prediction_gain: f32) {
        let score = example.score.clamp(0.0, 1.0);
        self.evidence_count = self.evidence_count.saturating_add(1);
        self.last_seen_ms = example.t_ms;
        self.confidence =
            (self.confidence * 0.78 + score * 0.22 + (self.evidence_count as f32).ln_1p() * 0.035)
                .clamp(0.0, 1.0);
        self.prediction_gain = (self.prediction_gain * 0.7 + prediction_gain * 0.3).clamp(0.0, 1.0);
        self.examples.push(example);
        const MAX_ASSOCIATION_EXAMPLES: usize = 12;
        if self.examples.len() > MAX_ASSOCIATION_EXAMPLES {
            let excess = self.examples.len() - MAX_ASSOCIATION_EXAMPLES;
            self.examples.drain(0..excess);
        }
    }

    fn weaken(&mut self, amount: f32) {
        self.confidence = (self.confidence * (1.0 - amount.clamp(0.0, 1.0))).clamp(0.0, 1.0);
        self.prediction_gain =
            (self.prediction_gain * (1.0 - amount.clamp(0.0, 1.0) * 0.5)).clamp(0.0, 1.0);
    }

    fn add_contradiction(&mut self, example: AssociationExample) {
        self.contradiction_count = self.contradiction_count.saturating_add(1);
        self.last_seen_ms = example.t_ms;
        self.confidence = (self.confidence * 0.72).clamp(0.0, 1.0);
        self.examples.push(example);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AssociationNegativeEvidence {
    pub present_id: String,
    pub absent_id: String,
    pub relation: AssociationRelation,
    pub reason: String,
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssociationTimeWindow {
    SameMoment,
    Within500Ms,
    Within2Sec,
    Within10Sec,
    NextFrame,
    NextActionOutcome,
}

impl AssociationTimeWindow {
    pub fn max_lag_ms(&self) -> u64 {
        match self {
            Self::SameMoment => 0,
            Self::Within500Ms => 500,
            Self::Within2Sec => 2_000,
            Self::Within10Sec => 10_000,
            Self::NextFrame => 2_000,
            Self::NextActionOutcome => 10_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssociationObservation {
    pub frame_id: Option<String>,
    pub t_ms: u64,
    #[serde(default)]
    pub active_items: Vec<AssociationItem>,
    #[serde(default)]
    pub outcome_items: Vec<AssociationItem>,
    #[serde(default)]
    pub prediction_error_items: Vec<AssociationItem>,
    #[serde(default)]
    pub memory_recall_items: Vec<AssociationItem>,
    #[serde(default)]
    pub llm_notes: Vec<String>,
    #[serde(default)]
    pub negative_evidence: Vec<AssociationNegativeEvidence>,
}

impl Default for AssociationObservation {
    fn default() -> Self {
        Self {
            frame_id: None,
            t_ms: 0,
            active_items: Vec::new(),
            outcome_items: Vec::new(),
            prediction_error_items: Vec::new(),
            memory_recall_items: Vec::new(),
            llm_notes: Vec::new(),
            negative_evidence: Vec::new(),
        }
    }
}

impl AssociationObservation {
    fn all_items(&self) -> Vec<AssociationItem> {
        let mut items = self
            .active_items
            .iter()
            .chain(self.outcome_items.iter())
            .chain(self.prediction_error_items.iter())
            .chain(self.memory_recall_items.iter())
            .cloned()
            .collect::<Vec<_>>();
        items.extend(self.llm_notes.iter().map(|note| {
            AssociationItem::new(
                format!("llm-note:{}", stable_slug(note)),
                AssociationItemKind::LlmNote,
                0.45,
            )
        }));
        dedupe_association_items(items)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssociationPrediction {
    pub source_id: String,
    pub predicted_id: String,
    pub relation: AssociationRelation,
    pub confidence: f32,
    pub prediction_gain: f32,
    pub evidence_count: u32,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveLearningTargetKind {
    BindingCandidate,
    TrackingHypothesis,
    Constellation,
    PlaceCandidate,
    ActionOutcome,
    PredictionFailure,
    #[default]
    Other,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveLearningActionKind {
    AskHuman,
    MoveOrRotate,
    WaitForEvidence,
    ReplayMemory,
    RequestLlmCritique,
    Diagnostic,
    #[default]
    Other,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveLearningState {
    #[default]
    Open,
    WaitingForSafety,
    WaitingForHuman,
    TestScheduled,
    Resolved,
    Abandoned,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InformationGatheringAction {
    pub kind: ActiveLearningActionKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<ActionPrimitive>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub human_question: Option<String>,
    pub expected_observation: String,
    pub disconfirming_observation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_safety_state: Option<String>,
    pub priority: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActiveLearningQuestion {
    pub id: String,
    pub target_id: String,
    pub target_kind: ActiveLearningTargetKind,
    pub question: String,
    pub uncertainty: f32,
    pub expected_information_gain: f32,
    pub risk: f32,
    pub proposed_tests: Vec<InformationGatheringAction>,
    pub state: ActiveLearningState,
}

impl ActiveLearningQuestion {
    pub fn best_test(&self) -> Option<&InformationGatheringAction> {
        self.proposed_tests.iter().max_by(|left, right| {
            left.priority
                .partial_cmp(&right.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PredictionFailure {
    pub id: String,
    pub target_id: String,
    pub predicted: String,
    pub observed: String,
    pub confidence: f32,
    pub surprise: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<ActionPrimitive>,
    #[serde(default)]
    pub possible_causes: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ActiveLearningReviewHint {
    pub id: String,
    pub target_id: String,
    pub target_kind: ActiveLearningTargetKind,
    #[serde(default)]
    pub suggested_tests: Vec<InformationGatheringAction>,
    #[serde(default)]
    pub human_review_prompts: Vec<String>,
    #[serde(default)]
    pub contradictions: Vec<String>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ActionIntentRecord {
    pub id: String,
    pub action: Option<ActionPrimitive>,
    pub frame_id: Option<String>,
    pub t_ms: u64,
    pub confidence: f32,
    pub state: String,
    pub reason: String,
    #[serde(default)]
    pub body_state_ids: Vec<String>,
    #[serde(default)]
    pub place_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OutcomeRecord {
    pub id: String,
    pub frame_id: Option<String>,
    pub t_ms: u64,
    pub reward: f32,
    pub success: Option<bool>,
    pub confidence: f32,
    pub state: String,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PredictionRecord {
    pub id: String,
    pub target_id: String,
    pub predicted: String,
    pub confidence: f32,
    pub t_ms: u64,
    pub state: String,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SurpriseRecord {
    pub id: String,
    pub target_id: String,
    pub observed: String,
    pub surprise: f32,
    pub confidence: f32,
    pub t_ms: u64,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmReviewRecord {
    pub id: String,
    pub target_id: String,
    pub target_kind: ActiveLearningTargetKind,
    pub confidence: f32,
    pub t_ms: u64,
    pub critique: String,
    #[serde(default)]
    pub contradictions: Vec<String>,
    #[serde(default)]
    pub suggested_questions: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HumanReviewRecord {
    pub id: String,
    pub target_id: String,
    pub target_kind: ActiveLearningTargetKind,
    pub confidence: f32,
    pub t_ms: u64,
    pub confirmation: String,
    pub reviewer: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphReviewRecord {
    pub id: String,
    pub target_id: String,
    pub review_kind: String,
    pub severity: f32,
    pub confidence: f32,
    pub t_ms: u64,
    pub reason: String,
    #[serde(default)]
    pub evidence_ids: Vec<String>,
    pub state: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphIntelligenceDocument {
    pub id: String,
    pub t_ms: u64,
    pub frame_id: Option<String>,
    pub provenance: String,
    pub confidence: f32,
    pub reason: String,
    #[serde(default)]
    pub source_frame_ids: Vec<String>,
    #[serde(default)]
    pub features: Vec<Feature>,
    #[serde(default)]
    pub clusters: Vec<DiscoveredCluster>,
    #[serde(default)]
    pub binding_candidates: Vec<BindingCandidate>,
    #[serde(default)]
    pub binding_edges: Vec<BindingEdge>,
    #[serde(default)]
    pub tracking_hypotheses: Vec<TrackingHypothesis>,
    #[serde(default)]
    pub constellations: Vec<Constellation>,
    #[serde(default)]
    pub associations: Vec<AssociationEdge>,
    #[serde(default)]
    pub action_intents: Vec<ActionIntentRecord>,
    #[serde(default)]
    pub outcomes: Vec<OutcomeRecord>,
    #[serde(default)]
    pub predictions: Vec<PredictionRecord>,
    #[serde(default)]
    pub surprises: Vec<SurpriseRecord>,
    #[serde(default)]
    pub llm_reviews: Vec<LlmReviewRecord>,
    #[serde(default)]
    pub human_reviews: Vec<HumanReviewRecord>,
    #[serde(default)]
    pub review_records: Vec<GraphReviewRecord>,
    #[serde(default)]
    pub learning_events: Vec<LearningEventRecord>,
    #[serde(default)]
    pub training_examples: Vec<TrainingExample>,
    #[serde(default)]
    pub replay_items: Vec<ReplayItem>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LearningEvent {
    FeatureObserved,
    ClusterStrengthened,
    ClusterSplit,
    ClusterMerged,
    BindingAccepted,
    BindingRejected,
    HypothesisPromoted,
    HypothesisExpired,
    ConstellationPromoted,
    PredictionSucceeded,
    PredictionFailed,
    SurpriseSpike,
    HumanCorrection,
    LlmCritiqueAccepted,
    AssociationStrengthened,
    ActiveLearningTaskCreated,
    #[default]
    Other,
}

impl LearningEvent {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::FeatureObserved => "feature_observed",
            Self::ClusterStrengthened => "cluster_strengthened",
            Self::ClusterSplit => "cluster_split",
            Self::ClusterMerged => "cluster_merged",
            Self::BindingAccepted => "binding_accepted",
            Self::BindingRejected => "binding_rejected",
            Self::HypothesisPromoted => "hypothesis_promoted",
            Self::HypothesisExpired => "hypothesis_expired",
            Self::ConstellationPromoted => "constellation_promoted",
            Self::PredictionSucceeded => "prediction_succeeded",
            Self::PredictionFailed => "prediction_failed",
            Self::SurpriseSpike => "surprise_spike",
            Self::HumanCorrection => "human_correction",
            Self::LlmCritiqueAccepted => "llm_critique_accepted",
            Self::AssociationStrengthened => "association_strengthened",
            Self::ActiveLearningTaskCreated => "active_learning_task_created",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LearningEventRecord {
    pub id: String,
    pub event: LearningEvent,
    pub target_id: String,
    pub t_ms: u64,
    pub confidence: f32,
    pub surprise: f32,
    pub novelty: f32,
    pub ambiguity: f32,
    pub contradiction: f32,
    pub trusted: bool,
    pub reason: String,
    #[serde(default)]
    pub evidence_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingExampleKind {
    PredictionPositive,
    PredictionNegative,
    BindingPositive,
    BindingNegative,
    ContrastiveNegative,
    AssociationPositive,
    AssociationNegative,
    ConstellationPositive,
    HumanTrustedPositive,
    LlmCritique,
    #[default]
    Other,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TrainingExample {
    pub id: String,
    pub kind: TrainingExampleKind,
    pub target_model: String,
    pub source_event_id: String,
    pub input_ref: String,
    pub target_ref: String,
    pub label: String,
    pub weight: f32,
    pub trusted: bool,
    pub reason: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

pub trait SelfTrainingTarget {
    fn generate_training_examples(
        &self,
        learning_event: &LearningEventRecord,
    ) -> Vec<TrainingExample>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DefaultSelfTrainingTarget {
    pub target_model: String,
}

impl Default for DefaultSelfTrainingTarget {
    fn default() -> Self {
        Self {
            target_model: "cognitive_loop".to_string(),
        }
    }
}

impl SelfTrainingTarget for DefaultSelfTrainingTarget {
    fn generate_training_examples(
        &self,
        learning_event: &LearningEventRecord,
    ) -> Vec<TrainingExample> {
        training_examples_for_event(learning_event, &self.target_model)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CurriculumScore {
    pub priority: f32,
    pub surprise: f32,
    pub novelty: f32,
    pub contradiction: f32,
    pub ambiguity: f32,
    pub human_confirmation: f32,
    pub prediction_improvement: f32,
    pub information_gain: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayItemState {
    #[default]
    Queued,
    Training,
    Archived,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReplayItem {
    pub id: String,
    pub event_id: String,
    pub source_frame_id: Option<String>,
    pub t_ms: u64,
    pub target_id: String,
    pub curriculum: CurriculumScore,
    pub decay_per_tick: f32,
    pub state: ReplayItemState,
    #[serde(default)]
    pub training_example_ids: Vec<String>,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplayBuffer {
    pub items: VecDeque<ReplayItem>,
    pub max_items: usize,
    pub archive_below_priority: f32,
}

impl Default for ReplayBuffer {
    fn default() -> Self {
        Self {
            items: VecDeque::new(),
            max_items: 512,
            archive_below_priority: 0.03,
        }
    }
}

impl ReplayBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, item: ReplayItem) {
        self.items.push_back(item);
        self.prioritize();
        while self.items.len() > self.max_items {
            self.items.pop_back();
        }
    }

    pub fn extend(&mut self, items: impl IntoIterator<Item = ReplayItem>) {
        for item in items {
            self.push(item);
        }
    }

    pub fn prioritize(&mut self) {
        let mut items = self.items.drain(..).collect::<Vec<_>>();
        items.sort_by(|left, right| {
            right
                .curriculum
                .priority
                .partial_cmp(&left.curriculum.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.t_ms.cmp(&left.t_ms))
        });
        self.items = items.into();
    }

    pub fn decay(&mut self, ticks: u64) {
        let factor = (1.0 - ticks as f32 * 0.01).clamp(0.0, 1.0);
        for item in &mut self.items {
            let decay = (item.decay_per_tick * ticks as f32).clamp(0.0, 0.95);
            item.curriculum.priority =
                (item.curriculum.priority * (1.0 - decay) * factor).clamp(0.0, 1.0);
            if item.curriculum.priority < self.archive_below_priority {
                item.state = ReplayItemState::Archived;
            }
        }
        self.prioritize();
    }

    pub fn queued(&self) -> Vec<ReplayItem> {
        self.items
            .iter()
            .filter(|item| item.state == ReplayItemState::Queued)
            .cloned()
            .collect()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LearningCycleReport {
    pub t_ms: u64,
    pub frame_id: Option<String>,
    #[serde(default)]
    pub learning_events: Vec<LearningEventRecord>,
    #[serde(default)]
    pub replay_items: Vec<ReplayItem>,
    #[serde(default)]
    pub training_examples: Vec<TrainingExample>,
    #[serde(default)]
    pub critique_tasks: Vec<ActiveLearningQuestion>,
    pub features_observed: usize,
    pub clusters_updated: usize,
    pub bindings_accepted: usize,
    pub bindings_rejected: usize,
    pub hypotheses_promoted: usize,
    pub hypotheses_expired: usize,
    pub constellations_promoted: usize,
    pub prediction_successes: usize,
    pub prediction_failures: usize,
    pub surprise: f32,
    pub active_learning_tasks: usize,
    pub human_review_requests: usize,
    pub what_observed: String,
    pub what_changed: String,
    pub what_surprised: String,
    pub what_became_stronger: String,
    pub what_became_weaker: String,
    pub what_to_investigate: String,
    pub what_to_remember: String,
    pub what_to_forget: String,
    pub what_to_train_on: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CognitiveDiagnosticsReport {
    pub summary: CognitiveDiagnosticsSummary,
    pub features: FeatureDiagnostics,
    pub clusters: ClusterDiagnostics,
    pub bindings: BindingDiagnostics,
    pub hypotheses: HypothesisDiagnostics,
    pub constellations: ConstellationDiagnostics,
    pub associations: AssociationDiagnostics,
    pub predictions: PredictionDiagnostics,
    pub active_learning: ActiveLearningDiagnostics,
    pub learning_cycle: LearningCycleReport,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CognitiveDiagnosticsSummary {
    pub feature_count: usize,
    pub cluster_count: usize,
    pub binding_candidate_count: usize,
    pub accepted_binding_count: usize,
    pub rejected_binding_count: usize,
    pub ambiguous_binding_count: usize,
    pub hypothesis_count: usize,
    pub competing_hypothesis_family_count: usize,
    pub constellation_count: usize,
    pub association_count: usize,
    pub prediction_count: usize,
    pub prediction_failure_count: usize,
    pub learning_event_count: usize,
    pub replay_item_count: usize,
    pub training_example_count: usize,
    pub llm_critique_count: usize,
    pub open_question_count: usize,
    pub contradiction_count: usize,
    pub review_prompt_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FeatureDiagnostics {
    pub items: Vec<FeatureInspectorItem>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureInspectorItem {
    pub feature_id: String,
    pub modality: String,
    pub feature_type: String,
    pub timestamp_ms: u64,
    pub confidence: f32,
    pub provenance: String,
    pub source_frame: Option<String>,
    pub source_sensor: Option<String>,
    pub vector_refs: Vec<VectorRefSummary>,
    pub pose: Option<PoseSummary>,
    pub metadata_summary: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VectorRefSummary {
    pub collection: String,
    pub point_id: String,
    pub model: Option<String>,
    pub source_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoseSummary {
    pub x_m: f32,
    pub y_m: f32,
    pub z_m: f32,
    pub yaw_rad: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClusterDiagnostics {
    pub items: Vec<ClusterInspectorItem>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterInspectorItem {
    pub cluster_id: String,
    pub modality: String,
    pub lifecycle: String,
    pub kind: String,
    pub centroid_vector: Option<String>,
    pub member_feature_ids: Vec<String>,
    pub evidence_count: u32,
    pub confidence: f32,
    pub radius_or_spread: Option<f32>,
    pub nearest_neighbors: Vec<String>,
    pub split_merge_suggestions: Vec<String>,
    pub source_frame: Option<String>,
    pub pose: Option<PoseSummary>,
    pub metadata_summary: serde_json::Value,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BindingDiagnostics {
    pub items: Vec<BindingInspectorItem>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BindingInspectorItem {
    pub binding_candidate_id: String,
    pub accepted_binding_edge_id: Option<String>,
    pub left_cluster_id: String,
    pub right_cluster_id: String,
    pub relation: String,
    pub decision: String,
    pub confidence: f32,
    pub evidence: Vec<BindingEvidenceInspectorItem>,
    pub rejection_reason: Option<String>,
    pub ambiguity_reason: Option<String>,
    pub contradictions: Vec<String>,
    pub review_status: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BindingEvidenceInspectorItem {
    pub kind: String,
    pub score: f32,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HypothesisDiagnostics {
    pub families: Vec<HypothesisFamilyInspectorItem>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HypothesisFamilyInspectorItem {
    pub family_id: String,
    pub competing_hypotheses: Vec<HypothesisInspectorItem>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HypothesisInspectorItem {
    pub hypothesis_id: String,
    pub kind: String,
    pub target_id: Option<String>,
    pub current_confidence: f32,
    pub evidence: Vec<BindingEvidenceInspectorItem>,
    pub contradictions: Vec<String>,
    pub state: String,
    pub why_not_promoted: Option<String>,
    pub what_would_resolve_it: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConstellationDiagnostics {
    pub items: Vec<ConstellationInspectorItem>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstellationInspectorItem {
    pub constellation_id: String,
    pub state: String,
    pub kind_hint: Option<String>,
    pub member_clusters: Vec<String>,
    pub member_bindings: Vec<String>,
    pub supporting_features: Vec<String>,
    pub supporting_places: Vec<String>,
    pub supporting_entities: Vec<String>,
    pub missing_expected_evidence: Vec<String>,
    pub contradiction_notes: Vec<String>,
    pub prediction_value: f32,
    pub stability: f32,
    pub suggested_tests: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AssociationDiagnostics {
    pub items: Vec<AssociationInspectorItem>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssociationInspectorItem {
    pub association_id: String,
    pub from_id: String,
    pub to_id: String,
    pub relation_type: String,
    pub confidence: f32,
    pub prediction_gain: f32,
    pub evidence_count: u32,
    pub examples: Vec<AssociationExample>,
    pub contradiction_count: u32,
    pub last_seen_ms: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PredictionDiagnostics {
    pub items: Vec<PredictionInspectorItem>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictionInspectorItem {
    pub current_prediction_id: String,
    pub predicted_next_observation: String,
    pub actual_next_observation: Option<String>,
    pub prediction_error: Option<f32>,
    pub surprise: f32,
    pub likely_explanation: Option<String>,
    pub related_associations: Vec<String>,
    pub related_constellations: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ActiveLearningDiagnostics {
    pub open_questions: Vec<ActiveLearningInspectorItem>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActiveLearningInspectorItem {
    pub question_id: String,
    pub target_id: String,
    pub target_kind: String,
    pub question: String,
    pub target_uncertainty: f32,
    pub proposed_tests: Vec<InformationGatheringAction>,
    pub expected_observation: Option<String>,
    pub disconfirming_observation: Option<String>,
    pub risk: f32,
    pub expected_information_gain: f32,
    pub human_question: Option<String>,
    pub safety_blocker: Option<String>,
    pub state: String,
}

impl CognitiveDiagnosticsReport {
    pub fn from_graph_document(document: &GraphIntelligenceDocument) -> Self {
        let mut planner = DefaultActiveLearningPlanner::default();
        let active_learning_input = ActiveLearningInput {
            context: ActiveLearningContext {
                t_ms: document.t_ms,
                ..ActiveLearningContext::default()
            },
            ambiguous_binding_candidates: document
                .binding_candidates
                .iter()
                .filter(|candidate| is_ambiguous_binding(candidate))
                .cloned()
                .collect(),
            tracking_hypotheses: document.tracking_hypotheses.clone(),
            constellations: document.constellations.clone(),
            association_edges: document.associations.clone(),
            prediction_failures: prediction_failures_from_document(document),
            llm_reviews: document
                .llm_reviews
                .iter()
                .map(active_learning_hint_from_llm_review)
                .collect(),
            ..ActiveLearningInput::default()
        };
        let questions = planner.plan(&active_learning_input);
        Self::from_parts(document, questions)
    }

    pub fn from_entity_memory_report(report: &EntityMemoryReport) -> Self {
        let mut planner = DefaultActiveLearningPlanner::default();
        let mut hypotheses = Vec::new();
        hypotheses.extend(report.active_tracking_hypotheses.clone());
        hypotheses.extend(report.review_tracking_hypotheses.clone());
        hypotheses.extend(report.promoted_tracking_hypotheses.clone());
        hypotheses.extend(report.expired_tracking_hypotheses.clone());
        hypotheses.sort_by(|left, right| left.id.cmp(&right.id));
        hypotheses.dedup_by(|left, right| left.id == right.id);

        let mut candidates = Vec::new();
        candidates.extend(report.accepted_binding_candidates.clone());
        candidates.extend(report.ambiguous_binding_candidates.clone());
        candidates.extend(report.rejected_binding_candidates.clone());
        let input = ActiveLearningInput {
            ambiguous_binding_candidates: report.ambiguous_binding_candidates.clone(),
            tracking_hypotheses: hypotheses.clone(),
            ..ActiveLearningInput::default()
        };
        let questions = planner.plan(&input);
        let document = GraphIntelligenceDocument {
            provenance: "entity_memory_report".to_string(),
            binding_candidates: candidates,
            tracking_hypotheses: hypotheses,
            ..GraphIntelligenceDocument::default()
        };
        Self::from_parts(&document, questions)
    }

    pub fn with_embodied_context(mut self, context: &EmbodiedContext) -> Self {
        let mut features = context
            .sensations
            .iter()
            .map(feature_item_from_embodied_sensation)
            .collect::<Vec<_>>();
        self.features.items.append(&mut features);
        self.predictions
            .items
            .extend(
                context
                    .predictions
                    .iter()
                    .enumerate()
                    .map(|(index, prediction)| PredictionInspectorItem {
                        current_prediction_id: format!("embodied-prediction:{index}"),
                        predicted_next_observation: prediction.text.clone(),
                        actual_next_observation: None,
                        prediction_error: None,
                        surprise: 0.0,
                        likely_explanation: None,
                        related_associations: Vec::new(),
                        related_constellations: Vec::new(),
                    }),
            );
        self.refresh_summary();
        self
    }

    fn from_parts(
        document: &GraphIntelligenceDocument,
        questions: Vec<ActiveLearningQuestion>,
    ) -> Self {
        let learning_cycle = LearningCycleReport::from_document(document, &questions);
        let mut report = Self {
            features: FeatureDiagnostics {
                items: document.features.iter().map(feature_item).collect(),
            },
            clusters: ClusterDiagnostics {
                items: document
                    .clusters
                    .iter()
                    .map(|cluster| cluster_item(cluster, &document.clusters))
                    .collect(),
            },
            bindings: BindingDiagnostics {
                items: document
                    .binding_candidates
                    .iter()
                    .map(|candidate| binding_item(candidate, &document.binding_edges))
                    .collect(),
            },
            hypotheses: HypothesisDiagnostics {
                families: hypothesis_families(&document.tracking_hypotheses),
            },
            constellations: ConstellationDiagnostics {
                items: document
                    .constellations
                    .iter()
                    .map(|constellation| constellation_item(constellation, &questions))
                    .collect(),
            },
            associations: AssociationDiagnostics {
                items: document.associations.iter().map(association_item).collect(),
            },
            predictions: PredictionDiagnostics {
                items: prediction_items(document),
            },
            active_learning: ActiveLearningDiagnostics {
                open_questions: questions.iter().map(active_learning_item).collect(),
            },
            learning_cycle,
            summary: CognitiveDiagnosticsSummary::default(),
        };
        report.refresh_summary();
        report
    }

    fn refresh_summary(&mut self) {
        let accepted_binding_count = self
            .bindings
            .items
            .iter()
            .filter(|item| item.decision == "accept")
            .count();
        let rejected_binding_count = self
            .bindings
            .items
            .iter()
            .filter(|item| item.decision == "reject")
            .count();
        let ambiguous_binding_count = self
            .bindings
            .items
            .iter()
            .filter(|item| item.ambiguity_reason.is_some())
            .count();
        let contradiction_count = self
            .bindings
            .items
            .iter()
            .map(|item| item.contradictions.len())
            .sum::<usize>()
            + self
                .hypotheses
                .families
                .iter()
                .flat_map(|family| family.competing_hypotheses.iter())
                .map(|hypothesis| hypothesis.contradictions.len())
                .sum::<usize>()
            + self
                .constellations
                .items
                .iter()
                .map(|item| item.contradiction_notes.len())
                .sum::<usize>()
            + self
                .associations
                .items
                .iter()
                .map(|item| item.contradiction_count as usize)
                .sum::<usize>();
        self.summary = CognitiveDiagnosticsSummary {
            feature_count: self.features.items.len(),
            cluster_count: self.clusters.items.len(),
            binding_candidate_count: self.bindings.items.len(),
            accepted_binding_count,
            rejected_binding_count,
            ambiguous_binding_count,
            hypothesis_count: self
                .hypotheses
                .families
                .iter()
                .map(|family| family.competing_hypotheses.len())
                .sum(),
            competing_hypothesis_family_count: self
                .hypotheses
                .families
                .iter()
                .filter(|family| family.competing_hypotheses.len() > 1)
                .count(),
            constellation_count: self.constellations.items.len(),
            association_count: self.associations.items.len(),
            prediction_count: self.predictions.items.len(),
            prediction_failure_count: self
                .predictions
                .items
                .iter()
                .filter(|item| item.prediction_error.is_some() || item.surprise > 0.0)
                .count(),
            learning_event_count: self.learning_cycle.learning_events.len(),
            replay_item_count: self.learning_cycle.replay_items.len(),
            training_example_count: self.learning_cycle.training_examples.len(),
            llm_critique_count: self
                .learning_cycle
                .critique_tasks
                .iter()
                .filter(|question| {
                    question
                        .proposed_tests
                        .iter()
                        .any(|test| test.kind == ActiveLearningActionKind::RequestLlmCritique)
                })
                .count(),
            open_question_count: self.active_learning.open_questions.len(),
            contradiction_count,
            review_prompt_count: self
                .active_learning
                .open_questions
                .iter()
                .filter(|item| item.human_question.is_some())
                .count(),
        };
    }
}

impl LearningCycleReport {
    pub fn from_document(
        document: &GraphIntelligenceDocument,
        questions: &[ActiveLearningQuestion],
    ) -> Self {
        let mut events = document.learning_events.clone();
        events.extend(derive_learning_events(document, questions));
        dedupe_learning_events(&mut events);

        let target = DefaultSelfTrainingTarget::default();
        let mut training_examples = document.training_examples.clone();
        for event in &events {
            training_examples.extend(target.generate_training_examples(event));
        }
        dedupe_training_examples(&mut training_examples);

        let mut replay_items = document.replay_items.clone();
        replay_items.extend(
            events
                .iter()
                .map(|event| replay_item_for_event(event, document, &training_examples)),
        );
        dedupe_replay_items(&mut replay_items);
        replay_items.sort_by(|left, right| {
            right
                .curriculum
                .priority
                .partial_cmp(&left.curriculum.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let prediction_failures = events
            .iter()
            .filter(|event| event.event == LearningEvent::PredictionFailed)
            .count();
        let prediction_successes = events
            .iter()
            .filter(|event| event.event == LearningEvent::PredictionSucceeded)
            .count();
        let surprise = events
            .iter()
            .map(|event| event.surprise)
            .fold(0.0_f32, f32::max);
        let critique_tasks = questions
            .iter()
            .filter(|question| {
                question
                    .proposed_tests
                    .iter()
                    .any(|test| test.kind == ActiveLearningActionKind::RequestLlmCritique)
            })
            .cloned()
            .collect::<Vec<_>>();
        let human_review_requests = questions
            .iter()
            .filter(|question| {
                question
                    .proposed_tests
                    .iter()
                    .any(|test| test.kind == ActiveLearningActionKind::AskHuman)
            })
            .count();

        Self {
            t_ms: document.t_ms,
            frame_id: document.frame_id.clone(),
            features_observed: document.features.len(),
            clusters_updated: document.clusters.len(),
            bindings_accepted: document
                .binding_candidates
                .iter()
                .filter(|candidate| candidate.decision == BindingDecision::Accept)
                .count(),
            bindings_rejected: document
                .binding_candidates
                .iter()
                .filter(|candidate| candidate.decision == BindingDecision::Reject)
                .count(),
            hypotheses_promoted: document
                .tracking_hypotheses
                .iter()
                .filter(|hypothesis| hypothesis.state == HypothesisState::Promoted)
                .count(),
            hypotheses_expired: document
                .tracking_hypotheses
                .iter()
                .filter(|hypothesis| hypothesis.state == HypothesisState::Expired)
                .count(),
            constellations_promoted: document
                .constellations
                .iter()
                .filter(|constellation| constellation.state == ConstellationState::Stable)
                .count(),
            prediction_successes,
            prediction_failures,
            surprise,
            active_learning_tasks: questions.len(),
            human_review_requests,
            what_observed: summarize_observations(document),
            what_changed: summarize_changes(&events),
            what_surprised: summarize_surprise(&events),
            what_became_stronger: summarize_strengthened(&events),
            what_became_weaker: summarize_weakened(&events),
            what_to_investigate: summarize_investigations(questions),
            what_to_remember: summarize_memory_targets(&replay_items),
            what_to_forget: summarize_forgetting(&replay_items),
            what_to_train_on: summarize_training_targets(&training_examples),
            learning_events: events,
            replay_items,
            training_examples,
            critique_tasks,
        }
    }
}

fn derive_learning_events(
    document: &GraphIntelligenceDocument,
    questions: &[ActiveLearningQuestion],
) -> Vec<LearningEventRecord> {
    let mut events = Vec::new();
    for feature in &document.features {
        events.push(LearningEventRecord {
            id: format!(
                "learning:{}:{}",
                LearningEvent::FeatureObserved.as_str(),
                feature.id
            ),
            event: LearningEvent::FeatureObserved,
            target_id: feature.id.to_string(),
            t_ms: feature.created_at_ms,
            confidence: feature.confidence,
            novelty: (1.0 - feature.confidence).clamp(0.0, 1.0),
            reason: format!("{:?} feature entered the registry", feature.feature_type),
            evidence_ids: vec![feature.id.to_string()],
            ..LearningEventRecord::default()
        });
    }
    for cluster in &document.clusters {
        if cluster.confidence >= 0.65 {
            events.push(LearningEventRecord {
                id: format!("learning:cluster_strengthened:{}", stable_slug(&cluster.id)),
                event: LearningEvent::ClusterStrengthened,
                target_id: cluster.id.clone(),
                t_ms: cluster.last_seen_ms,
                confidence: cluster.confidence,
                novelty: (1.0 / cluster.feature_ids.len().max(1) as f32).clamp(0.0, 1.0),
                reason: "cluster gained enough evidence to be useful downstream".to_string(),
                evidence_ids: cluster
                    .feature_ids
                    .iter()
                    .map(|id| id.to_string())
                    .collect(),
                ..LearningEventRecord::default()
            });
        }
    }
    for candidate in &document.binding_candidates {
        let event = match candidate.decision {
            BindingDecision::Accept => Some(LearningEvent::BindingAccepted),
            BindingDecision::Reject => Some(LearningEvent::BindingRejected),
            _ => None,
        };
        if let Some(event) = event {
            events.push(LearningEventRecord {
                id: format!(
                    "learning:{}:{}",
                    event.as_str(),
                    binding_candidate_id(candidate)
                ),
                event,
                target_id: binding_candidate_id(candidate),
                t_ms: document.t_ms,
                confidence: candidate.confidence,
                ambiguity: binding_uncertainty(candidate),
                contradiction: candidate
                    .evidence
                    .iter()
                    .filter(|evidence| binding_evidence_is_contradictory(evidence))
                    .map(|evidence| evidence.score)
                    .fold(0.0_f32, f32::max),
                trusted: candidate
                    .evidence
                    .iter()
                    .any(|evidence| evidence.kind == BindingEvidenceKind::HumanConfirmed),
                reason: candidate.reason.clone(),
                evidence_ids: vec![
                    candidate.left_cluster_id.clone(),
                    candidate.right_cluster_id.clone(),
                ],
                ..LearningEventRecord::default()
            });
        }
    }
    for hypothesis in &document.tracking_hypotheses {
        let event = match hypothesis.state {
            HypothesisState::Promoted => Some(LearningEvent::HypothesisPromoted),
            HypothesisState::Expired => Some(LearningEvent::HypothesisExpired),
            _ => None,
        };
        if let Some(event) = event {
            events.push(LearningEventRecord {
                id: format!(
                    "learning:{}:{}",
                    event.as_str(),
                    stable_slug(&hypothesis.id)
                ),
                event,
                target_id: hypothesis.id.clone(),
                t_ms: hypothesis.last_updated_ms,
                confidence: hypothesis.confidence,
                ambiguity: hypothesis_uncertainty(hypothesis),
                contradiction: (!hypothesis.contradictions.is_empty()) as u8 as f32,
                trusted: hypothesis
                    .evidence
                    .iter()
                    .any(|evidence| evidence.kind == BindingEvidenceKind::HumanConfirmed),
                reason: format!("{:?} hypothesis is {:?}", hypothesis.kind, hypothesis.state),
                evidence_ids: hypothesis.binding_candidate_ids.clone(),
                ..LearningEventRecord::default()
            });
        }
        if hypothesis
            .evidence
            .iter()
            .any(|evidence| evidence.kind == BindingEvidenceKind::HumanConfirmed)
        {
            events.push(LearningEventRecord {
                id: format!("learning:human_correction:{}", stable_slug(&hypothesis.id)),
                event: LearningEvent::HumanCorrection,
                target_id: hypothesis.id.clone(),
                t_ms: hypothesis.last_updated_ms,
                confidence: 1.0,
                trusted: true,
                reason: "human confirmation resolved a tracking hypothesis".to_string(),
                evidence_ids: hypothesis.binding_candidate_ids.clone(),
                ..LearningEventRecord::default()
            });
        }
    }
    for constellation in &document.constellations {
        if constellation.state == ConstellationState::Stable {
            events.push(LearningEventRecord {
                id: format!(
                    "learning:constellation_promoted:{}",
                    stable_slug(&constellation.id)
                ),
                event: LearningEvent::ConstellationPromoted,
                target_id: constellation.id.clone(),
                t_ms: constellation.last_seen_ms,
                confidence: constellation.confidence,
                novelty: (1.0 - constellation.stability).clamp(0.0, 1.0),
                reason: "constellation became stable enough to train recognizers".to_string(),
                evidence_ids: constellation.member_binding_ids.clone(),
                ..LearningEventRecord::default()
            });
        }
    }
    for edge in &document.associations {
        if edge.confidence >= 0.5 || edge.prediction_gain >= 0.1 {
            events.push(LearningEventRecord {
                id: format!(
                    "learning:association_strengthened:{}",
                    stable_slug(&edge.id)
                ),
                event: LearningEvent::AssociationStrengthened,
                target_id: edge.id.clone(),
                t_ms: edge.last_seen_ms,
                confidence: edge.confidence,
                contradiction: (edge.contradiction_count as f32
                    / edge.evidence_count.max(1) as f32)
                    .clamp(0.0, 1.0),
                reason: format!("association {:?} gained predictive evidence", edge.relation),
                evidence_ids: edge
                    .examples
                    .iter()
                    .filter_map(|e| e.frame_id.clone())
                    .collect(),
                ..LearningEventRecord::default()
            });
        }
    }
    for prediction in &document.predictions {
        if document
            .surprises
            .iter()
            .all(|surprise| surprise.target_id != prediction.id)
            && prediction.confidence >= 0.55
        {
            events.push(LearningEventRecord {
                id: format!(
                    "learning:prediction_succeeded:{}",
                    stable_slug(&prediction.id)
                ),
                event: LearningEvent::PredictionSucceeded,
                target_id: prediction.id.clone(),
                t_ms: prediction.t_ms,
                confidence: prediction.confidence,
                reason: "prediction had no matching surprise in this cycle".to_string(),
                evidence_ids: vec![prediction.target_id.clone()],
                ..LearningEventRecord::default()
            });
        }
    }
    for failure in prediction_failures_from_document(document) {
        events.push(LearningEventRecord {
            id: format!("learning:prediction_failed:{}", stable_slug(&failure.id)),
            event: LearningEvent::PredictionFailed,
            target_id: failure.target_id.clone(),
            t_ms: document.t_ms,
            confidence: failure.confidence,
            surprise: failure.surprise,
            contradiction: failure.surprise,
            reason: failure.possible_causes.join("; "),
            evidence_ids: vec![failure.id.clone()],
            ..LearningEventRecord::default()
        });
        if failure.surprise >= 0.7 {
            events.push(LearningEventRecord {
                id: format!("learning:surprise_spike:{}", stable_slug(&failure.id)),
                event: LearningEvent::SurpriseSpike,
                target_id: failure.target_id,
                t_ms: document.t_ms,
                confidence: failure.confidence,
                surprise: failure.surprise,
                reason: "prediction error exceeded surprise-spike threshold".to_string(),
                evidence_ids: vec![failure.id],
                ..LearningEventRecord::default()
            });
        }
    }
    for review in &document.llm_reviews {
        if review.confidence >= 0.5 {
            events.push(LearningEventRecord {
                id: format!("learning:llm_critique_accepted:{}", stable_slug(&review.id)),
                event: LearningEvent::LlmCritiqueAccepted,
                target_id: review.target_id.clone(),
                t_ms: review.t_ms,
                confidence: review.confidence,
                contradiction: (!review.contradictions.is_empty()) as u8 as f32,
                reason: review.critique.clone(),
                evidence_ids: review.suggested_questions.clone(),
                ..LearningEventRecord::default()
            });
        }
    }
    for review in &document.human_reviews {
        events.push(LearningEventRecord {
            id: format!("learning:human_correction:{}", stable_slug(&review.id)),
            event: LearningEvent::HumanCorrection,
            target_id: review.target_id.clone(),
            t_ms: review.t_ms,
            confidence: review.confidence.max(0.9),
            trusted: true,
            reason: review.confirmation.clone(),
            evidence_ids: Vec::new(),
            ..LearningEventRecord::default()
        });
    }
    for question in questions {
        events.push(LearningEventRecord {
            id: format!(
                "learning:active_learning_task_created:{}",
                stable_slug(&question.id)
            ),
            event: LearningEvent::ActiveLearningTaskCreated,
            target_id: question.target_id.clone(),
            t_ms: document.t_ms,
            confidence: 1.0 - question.uncertainty,
            ambiguity: question.uncertainty,
            novelty: question.expected_information_gain,
            reason: question.question.clone(),
            evidence_ids: question
                .proposed_tests
                .iter()
                .map(|test| test.expected_observation.clone())
                .collect(),
            ..LearningEventRecord::default()
        });
    }
    events
}

fn training_examples_for_event(
    event: &LearningEventRecord,
    default_model: &str,
) -> Vec<TrainingExample> {
    let mut examples = Vec::new();
    let (kind, model, label) = match event.event {
        LearningEvent::PredictionSucceeded => (
            TrainingExampleKind::PredictionPositive,
            "prediction_model",
            "prediction_succeeded",
        ),
        LearningEvent::PredictionFailed | LearningEvent::SurpriseSpike => (
            TrainingExampleKind::PredictionNegative,
            "prediction_model",
            "prediction_failed",
        ),
        LearningEvent::BindingAccepted => (
            TrainingExampleKind::BindingPositive,
            "binding_model",
            "binding_accepted",
        ),
        LearningEvent::BindingRejected => (
            TrainingExampleKind::BindingNegative,
            "binding_model",
            "binding_rejected",
        ),
        LearningEvent::AssociationStrengthened => (
            TrainingExampleKind::AssociationPositive,
            "association_model",
            "association_strengthened",
        ),
        LearningEvent::ConstellationPromoted => (
            TrainingExampleKind::ConstellationPositive,
            "constellation_recognizer",
            "constellation_stable",
        ),
        LearningEvent::HumanCorrection => (
            TrainingExampleKind::HumanTrustedPositive,
            "trusted_correction_model",
            "human_confirmed",
        ),
        LearningEvent::LlmCritiqueAccepted => (
            TrainingExampleKind::LlmCritique,
            "critique_filter",
            "llm_critique_accepted",
        ),
        _ => (
            TrainingExampleKind::Other,
            default_model,
            event.event.as_str(),
        ),
    };
    if kind != TrainingExampleKind::Other || event.trusted {
        examples.push(TrainingExample {
            id: format!("training:{}:{}", model, stable_slug(&event.id)),
            kind,
            target_model: model.to_string(),
            source_event_id: event.id.clone(),
            input_ref: event.target_id.clone(),
            target_ref: event.evidence_ids.first().cloned().unwrap_or_default(),
            label: label.to_string(),
            weight: training_weight(event),
            trusted: event.trusted,
            reason: event.reason.clone(),
            metadata: json!({
                "surprise": event.surprise,
                "novelty": event.novelty,
                "ambiguity": event.ambiguity,
                "contradiction": event.contradiction
            }),
        });
    }
    if event.event == LearningEvent::BindingRejected {
        examples.push(TrainingExample {
            id: format!("training:contrastive:{}", stable_slug(&event.id)),
            kind: TrainingExampleKind::ContrastiveNegative,
            target_model: "contrastive_binding_model".to_string(),
            source_event_id: event.id.clone(),
            input_ref: event.target_id.clone(),
            target_ref: event.evidence_ids.join("|"),
            label: "not_same_binding".to_string(),
            weight: training_weight(event).max(0.5),
            trusted: event.trusted,
            reason: event.reason.clone(),
            metadata: serde_json::Value::Null,
        });
    }
    examples
}

fn replay_item_for_event(
    event: &LearningEventRecord,
    document: &GraphIntelligenceDocument,
    examples: &[TrainingExample],
) -> ReplayItem {
    let curriculum = curriculum_score(event);
    ReplayItem {
        id: format!("replay:{}", stable_slug(&event.id)),
        event_id: event.id.clone(),
        source_frame_id: document.frame_id.clone(),
        t_ms: event.t_ms,
        target_id: event.target_id.clone(),
        curriculum,
        decay_per_tick: if event.trusted { 0.0005 } else { 0.0025 },
        state: ReplayItemState::Queued,
        training_example_ids: examples
            .iter()
            .filter(|example| example.source_event_id == event.id)
            .map(|example| example.id.clone())
            .collect(),
        reason: event.reason.clone(),
    }
}

fn curriculum_score(event: &LearningEventRecord) -> CurriculumScore {
    let human_confirmation = if event.trusted { 1.0 } else { 0.0 };
    let prediction_improvement = match event.event {
        LearningEvent::PredictionSucceeded | LearningEvent::PredictionFailed => event.confidence,
        _ => 0.0,
    };
    let information_gain = event
        .surprise
        .max(event.novelty)
        .max(event.ambiguity)
        .max(event.contradiction)
        .max(human_confirmation);
    let priority = (event.surprise * 0.24
        + event.novelty * 0.16
        + event.contradiction * 0.18
        + event.ambiguity * 0.14
        + human_confirmation * 0.18
        + prediction_improvement * 0.05
        + information_gain * 0.05)
        .max(match event.event {
            LearningEvent::PredictionFailed
            | LearningEvent::SurpriseSpike
            | LearningEvent::HumanCorrection => 0.75,
            LearningEvent::BindingRejected | LearningEvent::ConstellationPromoted => 0.55,
            LearningEvent::BindingAccepted | LearningEvent::AssociationStrengthened => 0.35,
            _ => 0.1,
        })
        .clamp(0.0, 1.0);
    CurriculumScore {
        priority,
        surprise: event.surprise,
        novelty: event.novelty,
        contradiction: event.contradiction,
        ambiguity: event.ambiguity,
        human_confirmation,
        prediction_improvement,
        information_gain,
    }
}

fn training_weight(event: &LearningEventRecord) -> f32 {
    (0.35
        + event.confidence * 0.25
        + event.surprise * 0.15
        + event.contradiction * 0.1
        + event.ambiguity * 0.05
        + if event.trusted { 0.35 } else { 0.0 })
    .clamp(0.05, 1.0)
}

fn dedupe_learning_events(events: &mut Vec<LearningEventRecord>) {
    let mut seen = BTreeSet::new();
    events.retain(|event| seen.insert(event.id.clone()));
}

fn dedupe_training_examples(examples: &mut Vec<TrainingExample>) {
    let mut seen = BTreeSet::new();
    examples.retain(|example| seen.insert(example.id.clone()));
}

fn dedupe_replay_items(items: &mut Vec<ReplayItem>) {
    let mut seen = BTreeSet::new();
    items.retain(|item| seen.insert(item.id.clone()));
}

fn summarize_observations(document: &GraphIntelligenceDocument) -> String {
    format!(
        "{} features, {} clusters, {} predictions, {} surprise records",
        document.features.len(),
        document.clusters.len(),
        document.predictions.len(),
        document.surprises.len()
    )
}

fn summarize_changes(events: &[LearningEventRecord]) -> String {
    let mut counts = BTreeMap::<&'static str, usize>::new();
    for event in events {
        *counts.entry(event.event.as_str()).or_default() += 1;
    }
    if counts.is_empty() {
        return "no learning events yet".to_string();
    }
    counts
        .into_iter()
        .map(|(event, count)| format!("{count} {event}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn summarize_surprise(events: &[LearningEventRecord]) -> String {
    events
        .iter()
        .filter(|event| event.surprise > 0.0)
        .max_by(|left, right| {
            left.surprise
                .partial_cmp(&right.surprise)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|event| {
            format!(
                "{} at {:.2}: {}",
                event.target_id, event.surprise, event.reason
            )
        })
        .unwrap_or_else(|| "nothing exceeded the surprise threshold".to_string())
}

fn summarize_strengthened(events: &[LearningEventRecord]) -> String {
    let targets = events
        .iter()
        .filter(|event| {
            matches!(
                event.event,
                LearningEvent::ClusterStrengthened
                    | LearningEvent::BindingAccepted
                    | LearningEvent::HypothesisPromoted
                    | LearningEvent::ConstellationPromoted
                    | LearningEvent::AssociationStrengthened
                    | LearningEvent::PredictionSucceeded
            )
        })
        .map(|event| event.target_id.clone())
        .take(5)
        .collect::<Vec<_>>();
    list_or_none(targets, "no subsystem strengthened this cycle")
}

fn summarize_weakened(events: &[LearningEventRecord]) -> String {
    let targets = events
        .iter()
        .filter(|event| {
            matches!(
                event.event,
                LearningEvent::BindingRejected
                    | LearningEvent::HypothesisExpired
                    | LearningEvent::PredictionFailed
                    | LearningEvent::SurpriseSpike
            )
        })
        .map(|event| event.target_id.clone())
        .take(5)
        .collect::<Vec<_>>();
    list_or_none(targets, "nothing was weakened beyond graceful decay")
}

fn summarize_investigations(questions: &[ActiveLearningQuestion]) -> String {
    list_or_none(
        questions
            .iter()
            .map(|question| question.question.clone())
            .take(5)
            .collect(),
        "no active investigation queued",
    )
}

fn summarize_memory_targets(items: &[ReplayItem]) -> String {
    list_or_none(
        items
            .iter()
            .filter(|item| item.curriculum.priority >= 0.5)
            .map(|item| item.target_id.clone())
            .take(5)
            .collect(),
        "no high-priority replay item",
    )
}

fn summarize_forgetting(items: &[ReplayItem]) -> String {
    list_or_none(
        items
            .iter()
            .filter(|item| item.state == ReplayItemState::Archived)
            .map(|item| item.target_id.clone())
            .take(5)
            .collect(),
        "old replay items should decay, not disappear immediately",
    )
}

fn summarize_training_targets(examples: &[TrainingExample]) -> String {
    let mut counts = BTreeMap::<String, usize>::new();
    for example in examples {
        *counts.entry(example.target_model.clone()).or_default() += 1;
    }
    if counts.is_empty() {
        return "no training examples generated".to_string();
    }
    counts
        .into_iter()
        .map(|(model, count)| format!("{count} for {model}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn list_or_none(items: Vec<String>, fallback: &str) -> String {
    if items.is_empty() {
        fallback.to_string()
    } else {
        items.join(", ")
    }
}

fn hypothesis_uncertainty(hypothesis: &TrackingHypothesis) -> f32 {
    (1.0 - hypothesis.confidence)
        .max(if hypothesis.contradictions.is_empty() {
            0.0
        } else {
            0.65
        })
        .clamp(0.0, 1.0)
}

fn feature_item(feature: &Feature) -> FeatureInspectorItem {
    FeatureInspectorItem {
        feature_id: feature.id.to_string(),
        modality: feature.modality.as_str().to_string(),
        feature_type: format!("{:?}", feature.feature_type),
        timestamp_ms: feature.created_at_ms,
        confidence: feature.confidence,
        provenance: summarize_json(&feature.provenance),
        source_frame: feature.source_frame.clone(),
        source_sensor: feature.source_sensor.clone(),
        vector_refs: feature
            .vector_refs
            .iter()
            .map(|vector| VectorRefSummary {
                collection: vector.collection.clone(),
                point_id: vector.point_id.clone(),
                model: vector.model.clone(),
                source_id: vector.source_id.clone(),
            })
            .collect(),
        pose: feature
            .world_pose
            .or(feature.local_pose)
            .map(|pose| PoseSummary {
                x_m: pose.x_m,
                y_m: pose.y_m,
                z_m: pose.z_m,
                yaw_rad: pose.yaw_rad,
            }),
        metadata_summary: summarize_value(&feature.metadata),
    }
}

fn feature_item_from_embodied_sensation(
    sensation: &pete_experience::EmbodiedSensationRef,
) -> FeatureInspectorItem {
    FeatureInspectorItem {
        feature_id: sensation.id.to_string(),
        modality: sensation.modality.as_str().to_string(),
        feature_type: sensation.payload_kind.as_str().to_string(),
        timestamp_ms: 0,
        confidence: 0.5,
        provenance: sensation.source.clone(),
        source_frame: None,
        source_sensor: Some(sensation.source.clone()),
        vector_refs: Vec::new(),
        pose: None,
        metadata_summary: json!({
            "kind": sensation.kind,
            "summary": sensation.summary,
            "parent_id": sensation.parent_id.map(|id| id.to_string()),
        }),
    }
}

fn cluster_item(cluster: &DiscoveredCluster, all: &[DiscoveredCluster]) -> ClusterInspectorItem {
    let nearest_neighbors = all
        .iter()
        .filter(|other| other.id != cluster.id && other.modality == cluster.modality)
        .take(5)
        .map(|other| other.id.clone())
        .collect::<Vec<_>>();
    let mut split_merge_suggestions = Vec::new();
    if cluster.confidence < 0.45 {
        split_merge_suggestions
            .push("low confidence; collect more evidence before merging".to_string());
    }
    if metadata_bool(cluster, "moves_independently") {
        split_merge_suggestions
            .push("independent motion suggests this cluster may need splitting".to_string());
    }
    ClusterInspectorItem {
        cluster_id: cluster.id.clone(),
        modality: cluster.modality.as_str().to_string(),
        lifecycle: if cluster.confidence >= 0.7 {
            "strong".to_string()
        } else if cluster.confidence >= 0.4 {
            "tentative".to_string()
        } else {
            "weak".to_string()
        },
        kind: format!("{:?}", cluster.kind),
        centroid_vector: cluster
            .metadata
            .get("centroid_vector_id")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        member_feature_ids: cluster
            .feature_ids
            .iter()
            .map(|id| id.to_string())
            .collect(),
        evidence_count: cluster.feature_ids.len().max(1) as u32,
        confidence: cluster.confidence,
        radius_or_spread: cluster
            .metadata
            .get("radius")
            .or_else(|| cluster.metadata.get("spread"))
            .and_then(value_as_f32),
        nearest_neighbors,
        split_merge_suggestions,
        source_frame: cluster.source_frame_id.clone(),
        pose: cluster.estimated_pose.map(|pose| PoseSummary {
            x_m: pose.x_m,
            y_m: pose.y_m,
            z_m: 0.0,
            yaw_rad: pose.heading_rad,
        }),
        metadata_summary: summarize_value(&cluster.metadata),
    }
}

fn binding_item(candidate: &BindingCandidate, edges: &[BindingEdge]) -> BindingInspectorItem {
    let contradictions = candidate
        .evidence
        .iter()
        .filter(|evidence| binding_evidence_is_contradictory(evidence))
        .map(|evidence| evidence.reason.clone())
        .collect::<Vec<_>>();
    let accepted_binding_edge_id = (candidate.decision == BindingDecision::Accept).then(|| {
        edges
            .iter()
            .find(|edge| {
                edge.left_cluster_id == candidate.left_cluster_id
                    && edge.right_cluster_id == candidate.right_cluster_id
                    && edge.relation == candidate.relation
            })
            .map(binding_edge_id)
            .unwrap_or_else(|| {
                binding_edge_id_from_parts(
                    &candidate.left_cluster_id,
                    &candidate.right_cluster_id,
                    &candidate.relation,
                )
            })
    });
    BindingInspectorItem {
        binding_candidate_id: binding_candidate_id(candidate),
        accepted_binding_edge_id,
        left_cluster_id: candidate.left_cluster_id.clone(),
        right_cluster_id: candidate.right_cluster_id.clone(),
        relation: binding_relation_slug(&candidate.relation).to_string(),
        decision: binding_decision_slug(&candidate.decision).to_string(),
        confidence: candidate.confidence,
        evidence: candidate
            .evidence
            .iter()
            .map(binding_evidence_item)
            .collect(),
        rejection_reason: (candidate.decision == BindingDecision::Reject)
            .then(|| candidate.reason.clone()),
        ambiguity_reason: binding_is_unresolved(candidate).then(|| candidate.reason.clone()),
        contradictions,
        review_status: match candidate.decision {
            BindingDecision::AskHuman => "needs_human_review",
            BindingDecision::HoldAmbiguous | BindingDecision::CollectMoreEvidence => {
                "needs_more_evidence"
            }
            BindingDecision::Reject => "rejected",
            BindingDecision::Accept => "accepted",
        }
        .to_string(),
    }
}

fn binding_is_unresolved(candidate: &BindingCandidate) -> bool {
    !matches!(
        candidate.decision,
        BindingDecision::Accept | BindingDecision::Reject
    ) && is_ambiguous_binding(candidate)
}

fn binding_evidence_item(evidence: &BindingEvidence) -> BindingEvidenceInspectorItem {
    BindingEvidenceInspectorItem {
        kind: binding_evidence_slug(&evidence.kind).to_string(),
        score: evidence.score,
        reason: evidence.reason.clone(),
    }
}

fn hypothesis_families(hypotheses: &[TrackingHypothesis]) -> Vec<HypothesisFamilyInspectorItem> {
    let mut by_family = BTreeMap::<String, Vec<HypothesisInspectorItem>>::new();
    for hypothesis in hypotheses {
        by_family
            .entry(hypothesis.family_id.clone())
            .or_default()
            .push(hypothesis_item(hypothesis));
    }
    by_family
        .into_iter()
        .map(|(family_id, mut competing_hypotheses)| {
            competing_hypotheses.sort_by(|left, right| {
                right
                    .current_confidence
                    .partial_cmp(&left.current_confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            HypothesisFamilyInspectorItem {
                family_id,
                competing_hypotheses,
            }
        })
        .collect()
}

fn hypothesis_item(hypothesis: &TrackingHypothesis) -> HypothesisInspectorItem {
    HypothesisInspectorItem {
        hypothesis_id: hypothesis.id.clone(),
        kind: format!("{:?}", hypothesis.kind),
        target_id: hypothesis.target_id.clone(),
        current_confidence: hypothesis.confidence,
        evidence: hypothesis
            .evidence
            .iter()
            .map(binding_evidence_item)
            .collect(),
        contradictions: hypothesis.contradictions.clone(),
        state: format!("{:?}", hypothesis.state),
        why_not_promoted: why_hypothesis_not_promoted(hypothesis),
        what_would_resolve_it: hypothesis_resolution_notes(hypothesis),
    }
}

fn why_hypothesis_not_promoted(hypothesis: &TrackingHypothesis) -> Option<String> {
    if hypothesis.state == HypothesisState::Promoted {
        return None;
    }
    if !hypothesis.contradictions.is_empty() {
        Some("contradictory evidence must be resolved first".to_string())
    } else if hypothesis.confidence < HYPOTHESIS_PROMOTION_THRESHOLD {
        Some(format!(
            "confidence {:.2} is below promotion threshold {:.2}",
            hypothesis.confidence, HYPOTHESIS_PROMOTION_THRESHOLD
        ))
    } else if matches!(hypothesis.state, HypothesisState::NeedsReview) {
        Some("hypothesis is waiting for review".to_string())
    } else {
        Some("hypothesis has not accumulated enough stable evidence".to_string())
    }
}

fn hypothesis_resolution_notes(hypothesis: &TrackingHypothesis) -> Vec<String> {
    let mut notes = Vec::new();
    if !hypothesis.contradictions.is_empty() {
        notes.push("resolve contradiction or reject the losing competitor".to_string());
    }
    if hypothesis.confidence < HYPOTHESIS_PROMOTION_THRESHOLD {
        notes
            .push("collect repeated supporting evidence in another observation window".to_string());
    }
    notes.push("human or LLM review can name the target or reject the match".to_string());
    notes
}

fn constellation_item(
    constellation: &Constellation,
    questions: &[ActiveLearningQuestion],
) -> ConstellationInspectorItem {
    let contradiction_notes = constellation
        .notes
        .iter()
        .filter(|note| note.to_lowercase().contains("contradict"))
        .cloned()
        .collect::<Vec<_>>();
    let mut missing_expected_evidence = Vec::new();
    if constellation.member_binding_ids.is_empty() {
        missing_expected_evidence.push("no accepted binding evidence yet".to_string());
    }
    if constellation.supporting_feature_ids.is_empty() {
        missing_expected_evidence.push("no supporting feature ids attached".to_string());
    }
    ConstellationInspectorItem {
        constellation_id: constellation.id.clone(),
        state: constellation_state_slug(&constellation.state).to_string(),
        kind_hint: constellation.kind_hint.clone(),
        member_clusters: constellation.member_cluster_ids.clone(),
        member_bindings: constellation.member_binding_ids.clone(),
        supporting_features: constellation
            .supporting_feature_ids
            .iter()
            .map(|id| id.to_string())
            .collect(),
        supporting_places: constellation
            .supporting_place_cells
            .iter()
            .map(|cell| format!("place-cell:{},{}", cell.x, cell.y))
            .collect(),
        supporting_entities: constellation.supporting_entity_ids.clone(),
        missing_expected_evidence,
        contradiction_notes,
        prediction_value: constellation.prediction_value,
        stability: constellation.stability,
        suggested_tests: questions
            .iter()
            .filter(|question| question.target_id == constellation.id)
            .flat_map(|question| {
                question
                    .proposed_tests
                    .iter()
                    .map(|test| test.expected_observation.clone())
            })
            .collect(),
    }
}

fn association_item(edge: &AssociationEdge) -> AssociationInspectorItem {
    AssociationInspectorItem {
        association_id: edge.id.clone(),
        from_id: edge.from_id.clone(),
        to_id: edge.to_id.clone(),
        relation_type: format!("{:?}", edge.relation),
        confidence: edge.confidence,
        prediction_gain: edge.prediction_gain,
        evidence_count: edge.evidence_count,
        examples: edge.examples.clone(),
        contradiction_count: edge.contradiction_count,
        last_seen_ms: edge.last_seen_ms,
    }
}

fn prediction_items(document: &GraphIntelligenceDocument) -> Vec<PredictionInspectorItem> {
    let mut items = document
        .predictions
        .iter()
        .map(|prediction| PredictionInspectorItem {
            current_prediction_id: prediction.id.clone(),
            predicted_next_observation: prediction.predicted.clone(),
            actual_next_observation: document
                .surprises
                .iter()
                .find(|surprise| surprise.target_id == prediction.id)
                .map(|surprise| surprise.observed.clone()),
            prediction_error: document
                .surprises
                .iter()
                .find(|surprise| surprise.target_id == prediction.id)
                .map(|surprise| surprise.surprise),
            surprise: document
                .surprises
                .iter()
                .filter(|surprise| surprise.target_id == prediction.id)
                .map(|surprise| surprise.surprise)
                .fold(0.0_f32, f32::max),
            likely_explanation: Some(prediction.reason.clone()).filter(|reason| !reason.is_empty()),
            related_associations: document
                .associations
                .iter()
                .filter(|edge| {
                    edge.from_id == prediction.target_id || edge.to_id == prediction.target_id
                })
                .map(|edge| edge.id.clone())
                .collect(),
            related_constellations: document
                .constellations
                .iter()
                .filter(|constellation| {
                    constellation
                        .member_cluster_ids
                        .iter()
                        .any(|id| id == &prediction.target_id)
                        || constellation
                            .member_binding_ids
                            .iter()
                            .any(|id| id == &prediction.target_id)
                })
                .map(|constellation| constellation.id.clone())
                .collect(),
        })
        .collect::<Vec<_>>();
    items.extend(document.surprises.iter().filter_map(|surprise| {
        document
            .predictions
            .iter()
            .any(|prediction| prediction.id == surprise.target_id)
            .then_some(())
            .is_none()
            .then(|| PredictionInspectorItem {
                current_prediction_id: surprise.target_id.clone(),
                predicted_next_observation: "unknown".to_string(),
                actual_next_observation: Some(surprise.observed.clone()),
                prediction_error: Some(surprise.surprise),
                surprise: surprise.surprise,
                likely_explanation: Some(surprise.reason.clone()),
                related_associations: Vec::new(),
                related_constellations: Vec::new(),
            })
    }));
    items
}

fn active_learning_item(question: &ActiveLearningQuestion) -> ActiveLearningInspectorItem {
    let best = question.best_test();
    ActiveLearningInspectorItem {
        question_id: question.id.clone(),
        target_id: question.target_id.clone(),
        target_kind: format!("{:?}", question.target_kind),
        question: question.question.clone(),
        target_uncertainty: question.uncertainty,
        proposed_tests: question.proposed_tests.clone(),
        expected_observation: best.map(|test| test.expected_observation.clone()),
        disconfirming_observation: best.map(|test| test.disconfirming_observation.clone()),
        risk: question.risk,
        expected_information_gain: question.expected_information_gain,
        human_question: best.and_then(|test| test.human_question.clone()),
        safety_blocker: question
            .proposed_tests
            .iter()
            .find_map(|test| test.required_safety_state.clone()),
        state: format!("{:?}", question.state),
    }
}

fn prediction_failures_from_document(
    document: &GraphIntelligenceDocument,
) -> Vec<PredictionFailure> {
    document
        .surprises
        .iter()
        .map(|surprise| {
            let prediction = document
                .predictions
                .iter()
                .find(|prediction| prediction.id == surprise.target_id);
            PredictionFailure {
                id: format!("prediction-failure:{}", stable_slug(&surprise.id)),
                target_id: surprise.target_id.clone(),
                predicted: prediction
                    .map(|prediction| prediction.predicted.clone())
                    .unwrap_or_else(|| "unknown".to_string()),
                observed: surprise.observed.clone(),
                confidence: surprise.confidence,
                surprise: surprise.surprise,
                action: None,
                possible_causes: vec![surprise.reason.clone()],
            }
        })
        .collect()
}

fn active_learning_hint_from_llm_review(review: &LlmReviewRecord) -> ActiveLearningReviewHint {
    ActiveLearningReviewHint {
        id: review.id.clone(),
        target_id: review.target_id.clone(),
        target_kind: review.target_kind.clone(),
        suggested_tests: Vec::new(),
        human_review_prompts: review.suggested_questions.clone(),
        contradictions: review.contradictions.clone(),
        confidence: review.confidence,
    }
}

fn summarize_json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "<unserializable>".to_string())
}

fn summarize_value(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(items) => json!({
            "kind": "array",
            "len": items.len(),
            "sample": items.iter().take(5).map(summarize_value).collect::<Vec<_>>(),
        }),
        serde_json::Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (key, value) in map.iter().take(12) {
                out.insert(key.clone(), summarize_value(value));
            }
            if map.len() > 12 {
                out.insert("_truncated_keys".to_string(), json!(map.len() - 12));
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::String(text) if text.len() > 160 => {
            json!({ "kind": "text", "char_count": text.len(), "preview": &text[..160] })
        }
        other => other.clone(),
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphFactSummary {
    pub id: String,
    pub kind: String,
    pub relation: Option<String>,
    pub confidence: f32,
    pub evidence_count: u32,
    pub t_ms: u64,
    pub state: Option<String>,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FeatureClusterIntelligence {
    pub node_id: String,
    pub clusters: Vec<GraphFactSummary>,
    pub bindings: Vec<GraphFactSummary>,
    pub supporting_evidence: Vec<GraphFactSummary>,
    pub contradictions: Vec<GraphFactSummary>,
    pub constellations: Vec<GraphFactSummary>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConstellationIntelligence {
    pub constellation_id: String,
    pub state: String,
    pub members: Vec<GraphFactSummary>,
    pub missing_members: Vec<String>,
    pub predictions: Vec<GraphFactSummary>,
    pub similar_constellations: Vec<GraphFactSummary>,
    pub contradictions: Vec<GraphFactSummary>,
    pub stability: f32,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AmbiguityIntelligence {
    pub target_id: String,
    pub competing_hypotheses: Vec<GraphFactSummary>,
    pub distinguishing_evidence: Vec<GraphFactSummary>,
    pub contradictions: Vec<GraphFactSummary>,
    pub human_question: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ActionOutcomeIntelligence {
    pub action_id: String,
    pub outcomes: Vec<GraphFactSummary>,
    pub preventing_body_states: Vec<GraphFactSummary>,
    pub risky_places: Vec<GraphFactSummary>,
    pub usual_next: Vec<GraphFactSummary>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphCommunityMember {
    pub node_id: String,
    pub labels: Vec<String>,
    pub score: f32,
    pub depth: u32,
    pub recurrence: u32,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphCommunity {
    pub start_node_id: String,
    pub max_depth: u32,
    pub min_weight: f32,
    pub members: Vec<GraphCommunityMember>,
    pub summary: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphRecallQuery {
    #[serde(default)]
    pub active_feature_ids: Vec<String>,
    #[serde(default)]
    pub active_cluster_ids: Vec<String>,
    #[serde(default)]
    pub active_constellation_ids: Vec<String>,
    #[serde(default)]
    pub action_ids: Vec<String>,
    #[serde(default)]
    pub place_ids: Vec<String>,
    pub min_confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphRecallBundle {
    pub query_ids: Vec<String>,
    pub nearby_memories: Vec<GraphFactSummary>,
    pub similar_constellations: Vec<GraphFactSummary>,
    pub likely_outcomes: Vec<GraphFactSummary>,
    pub previous_contradictions: Vec<GraphFactSummary>,
    pub human_confirmations: Vec<GraphFactSummary>,
    pub llm_critiques: Vec<GraphFactSummary>,
    pub action_successes: Vec<GraphFactSummary>,
    pub action_failures: Vec<GraphFactSummary>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MovementReadiness {
    pub safety_allows_motion: bool,
    pub robot_mode_allows_motion: bool,
    pub base_connected: bool,
    pub controller_ready: bool,
    pub body_state_ready: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub movement_responding: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl Default for MovementReadiness {
    fn default() -> Self {
        Self {
            safety_allows_motion: true,
            robot_mode_allows_motion: true,
            base_connected: true,
            controller_ready: true,
            body_state_ready: true,
            movement_responding: Some(true),
            reason: None,
        }
    }
}

impl MovementReadiness {
    pub fn allows_information_motion(&self) -> bool {
        self.safety_allows_motion
            && self.robot_mode_allows_motion
            && self.base_connected
            && self.controller_ready
            && self.body_state_ready
            && self.movement_responding.unwrap_or(true)
    }

    pub fn blocking_reason(&self) -> String {
        if let Some(reason) = &self.reason {
            return reason.clone();
        }
        if !self.safety_allows_motion {
            "safety veto is active".to_string()
        } else if !self.robot_mode_allows_motion {
            "robot mode does not allow motion".to_string()
        } else if !self.base_connected {
            "base connection is not ready".to_string()
        } else if !self.controller_ready {
            "controller state is not ready".to_string()
        } else if !self.body_state_ready {
            "body state is stale or uncertain".to_string()
        } else if self.movement_responding == Some(false) {
            "movement is not responding".to_string()
        } else {
            "movement readiness is unknown".to_string()
        }
    }
}

impl From<&BodySense> for MovementReadiness {
    fn from(body: &BodySense) -> Self {
        let safety_allows_motion = !body.flags.wheel_drop
            && !body.flags.cliff_left
            && !body.flags.cliff_front_left
            && !body.flags.cliff_front_right
            && !body.flags.cliff_right
            && body.battery_level > 0.10;
        let body_state_ready = body.health.health > 0.0 && body.last_update_ms > 0;
        let movement_responding = if body.charging {
            Some(false)
        } else {
            Some(true)
        };
        Self {
            safety_allows_motion,
            robot_mode_allows_motion: true,
            base_connected: true,
            controller_ready: true,
            body_state_ready,
            movement_responding,
            reason: (!safety_allows_motion)
                .then(|| "body safety flags or critical battery block motion".to_string())
                .or_else(|| {
                    (!body_state_ready).then(|| "body state is stale or unhealthy".to_string())
                })
                .or_else(|| {
                    body.charging
                        .then(|| "body reports charging/docked".to_string())
                }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActiveLearningContext {
    pub t_ms: u64,
    #[serde(default)]
    pub available_actions: Vec<ActionPrimitive>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body_state: Option<BodySense>,
    pub movement_readiness: MovementReadiness,
    #[serde(default)]
    pub completed_test_ids: BTreeSet<String>,
    #[serde(default)]
    pub available_sensors: BTreeSet<String>,
}

impl Default for ActiveLearningContext {
    fn default() -> Self {
        Self {
            t_ms: 0,
            available_actions: Vec::new(),
            body_state: None,
            movement_readiness: MovementReadiness::default(),
            completed_test_ids: BTreeSet::new(),
            available_sensors: BTreeSet::new(),
        }
    }
}

impl ActiveLearningContext {
    pub fn from_binding_context(context: &BindingContext) -> Self {
        Self {
            t_ms: context.t_ms,
            available_actions: context.active_action.clone().into_iter().collect(),
            body_state: context.body_state.clone(),
            movement_readiness: context
                .body_state
                .as_ref()
                .map(MovementReadiness::from)
                .unwrap_or_default(),
            completed_test_ids: BTreeSet::new(),
            available_sensors: BTreeSet::new(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ActiveLearningInput {
    pub context: ActiveLearningContext,
    #[serde(default)]
    pub ambiguous_binding_candidates: Vec<BindingCandidate>,
    #[serde(default)]
    pub tracking_hypotheses: Vec<TrackingHypothesis>,
    #[serde(default)]
    pub constellations: Vec<Constellation>,
    #[serde(default)]
    pub place_candidates: Vec<PlaceRecognitionCandidate>,
    #[serde(default)]
    pub association_edges: Vec<AssociationEdge>,
    #[serde(default)]
    pub prediction_failures: Vec<PredictionFailure>,
    #[serde(default)]
    pub llm_reviews: Vec<ActiveLearningReviewHint>,
}

pub trait ActiveLearningPlanner {
    fn plan(&mut self, input: &ActiveLearningInput) -> Vec<ActiveLearningQuestion>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActiveLearningPlannerConfig {
    pub max_questions: usize,
    pub ask_human_uncertainty_threshold: f32,
    pub motion_risk: f32,
    pub human_question_risk: f32,
    pub wait_risk: f32,
    pub memory_risk: f32,
    pub llm_risk: f32,
}

impl Default for ActiveLearningPlannerConfig {
    fn default() -> Self {
        Self {
            max_questions: 16,
            ask_human_uncertainty_threshold: 0.25,
            motion_risk: 0.35,
            human_question_risk: 0.05,
            wait_risk: 0.02,
            memory_risk: 0.01,
            llm_risk: 0.03,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DefaultActiveLearningPlanner {
    pub config: ActiveLearningPlannerConfig,
}

impl Default for DefaultActiveLearningPlanner {
    fn default() -> Self {
        Self {
            config: ActiveLearningPlannerConfig::default(),
        }
    }
}

impl ActiveLearningPlanner for DefaultActiveLearningPlanner {
    fn plan(&mut self, input: &ActiveLearningInput) -> Vec<ActiveLearningQuestion> {
        let mut questions = Vec::new();
        for candidate in &input.ambiguous_binding_candidates {
            if is_ambiguous_binding(candidate) {
                questions.push(self.binding_question(candidate, input));
            }
        }
        questions.extend(self.tracking_questions(input));
        for constellation in &input.constellations {
            if matches!(
                constellation.state,
                ConstellationState::Ambiguous
                    | ConstellationState::SplitNeeded
                    | ConstellationState::MergeNeeded
                    | ConstellationState::Candidate
            ) {
                questions.push(self.constellation_question(constellation, input));
            }
        }
        for candidate in &input.place_candidates {
            if candidate.confidence < 0.72 {
                questions.push(self.place_question(candidate, input));
            }
        }
        for edge in &input.association_edges {
            if edge.contradiction_count > 0 || edge.confidence < 0.45 {
                questions.push(self.association_question(edge, input));
            }
        }
        for failure in &input.prediction_failures {
            questions.push(self.prediction_failure_question(failure, input));
        }
        for review in &input.llm_reviews {
            questions.push(self.review_question(review, input));
        }

        for question in &mut questions {
            question.proposed_tests.retain(|test| {
                let id = information_action_id(&question.target_id, test);
                !input.context.completed_test_ids.contains(&id)
            });
            question.proposed_tests.sort_by(|left, right| {
                right
                    .priority
                    .partial_cmp(&left.priority)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            question.expected_information_gain = question
                .best_test()
                .map(|test| test.priority)
                .unwrap_or(0.0);
            question.risk = question
                .proposed_tests
                .iter()
                .map(|test| action_risk(test, &self.config))
                .fold(0.0_f32, f32::max);
            question.state = infer_active_learning_state(question);
        }
        questions.retain(|question| !question.proposed_tests.is_empty());
        questions.sort_by(|left, right| {
            active_learning_score(right)
                .partial_cmp(&active_learning_score(left))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        questions.truncate(self.config.max_questions);
        questions
    }
}

impl DefaultActiveLearningPlanner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: ActiveLearningPlannerConfig) -> Self {
        Self { config }
    }

    fn binding_question(
        &self,
        candidate: &BindingCandidate,
        input: &ActiveLearningInput,
    ) -> ActiveLearningQuestion {
        let target_id = binding_candidate_id(candidate);
        let uncertainty = binding_uncertainty(candidate);
        let mut tests = vec![
            ask_human_action(
                binding_human_question(candidate),
                "human confirmation would settle the label or identity relationship",
                "human rejects the proposed relationship",
                self.config.human_question_risk,
                uncertainty,
            ),
            replay_memory_action(
                "compare candidate clusters against prior reviewed bindings",
                "memory contains a prior co-occurrence, label, or contradiction",
                "memory has no matching support or points to a different entity",
                self.config.memory_risk,
                uncertainty * 0.75,
            ),
            wait_action(
                "wait for another co-occurrence window",
                "the same clusters appear together again",
                "only one cluster reappears or a stronger competitor appears",
                self.config.wait_risk,
                uncertainty * 0.55,
            ),
        ];
        if binding_benefits_from_viewpoint(candidate) {
            tests.extend(self.motion_or_diagnostic_tests(
                input,
                "look again from a slightly different angle",
                "the relationship remains geometrically coherent after a small turn",
                "the clusters separate, reproject poorly, or one disappears",
                uncertainty * 0.9,
            ));
        }
        if candidate
            .evidence
            .iter()
            .any(|evidence| evidence.kind == BindingEvidenceKind::SimultaneousConflict)
        {
            tests.push(llm_critique_action(
                "ask for critique of the competing binding evidence",
                "LLM identifies which observation would separate the candidates",
                "LLM finds the evidence underdetermined and recommends human review",
                self.config.llm_risk,
                uncertainty * 0.45,
            ));
        }
        ActiveLearningQuestion {
            id: format!("active-learning:{target_id}"),
            target_id,
            target_kind: ActiveLearningTargetKind::BindingCandidate,
            question: format!(
                "What test would best decide whether {} {:?} {}?",
                candidate.left_cluster_id, candidate.relation, candidate.right_cluster_id
            ),
            uncertainty,
            expected_information_gain: 0.0,
            risk: 0.0,
            proposed_tests: tests,
            state: ActiveLearningState::Open,
        }
    }

    fn tracking_questions(&self, input: &ActiveLearningInput) -> Vec<ActiveLearningQuestion> {
        let mut by_family = BTreeMap::<String, Vec<&TrackingHypothesis>>::new();
        for hypothesis in &input.tracking_hypotheses {
            if matches!(
                hypothesis.state,
                HypothesisState::NeedsReview | HypothesisState::Winning | HypothesisState::Losing
            ) {
                by_family
                    .entry(hypothesis.family_id.clone())
                    .or_default()
                    .push(hypothesis);
            }
        }

        by_family
            .into_iter()
            .filter_map(|(family_id, mut family)| {
                family.sort_by(|left, right| {
                    right
                        .confidence
                        .partial_cmp(&left.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let first = *family.first()?;
                let second_confidence = family.get(1).map(|h| h.confidence).unwrap_or(0.0);
                let uncertainty = (1.0 - (first.confidence - second_confidence).abs())
                    .max(1.0 - first.confidence)
                    .clamp(0.0, 1.0);
                let target = first
                    .target_id
                    .clone()
                    .unwrap_or_else(|| "unknown target".to_string());
                let mut tests = vec![
                    replay_memory_action(
                        "compare competing tracking hypotheses with reviewed memory",
                        "one hypothesis has stronger historical support",
                        "memory supports a different target or no known target",
                        self.config.memory_risk,
                        uncertainty * 0.8,
                    ),
                    wait_action(
                        "wait for another observation of the same track family",
                        "the next observation strengthens one hypothesis",
                        "the next observation strengthens a competitor",
                        self.config.wait_risk,
                        uncertainty * 0.55,
                    ),
                ];
                if uncertainty >= self.config.ask_human_uncertainty_threshold {
                    tests.push(ask_human_action(
                        format!("Does this observation belong to {target}?"),
                        "human names or rejects the target",
                        "human says the observation belongs to someone or something else",
                        self.config.human_question_risk,
                        uncertainty,
                    ));
                }
                Some(ActiveLearningQuestion {
                    id: format!("active-learning:tracking:{}", stable_slug(&family_id)),
                    target_id: first.id.clone(),
                    target_kind: ActiveLearningTargetKind::TrackingHypothesis,
                    question: format!("Which tracking hypothesis in {family_id} should survive?"),
                    uncertainty,
                    expected_information_gain: 0.0,
                    risk: 0.0,
                    proposed_tests: tests,
                    state: ActiveLearningState::Open,
                })
            })
            .collect()
    }

    fn constellation_question(
        &self,
        constellation: &Constellation,
        input: &ActiveLearningInput,
    ) -> ActiveLearningQuestion {
        let uncertainty = (1.0 - constellation.confidence)
            .max(
                if matches!(constellation.state, ConstellationState::SplitNeeded) {
                    0.7
                } else {
                    0.35
                },
            )
            .clamp(0.0, 1.0);
        let mut tests = vec![
            replay_memory_action(
                "compare this constellation with previous scene or entity constellations",
                "the member pattern matches one known constellation",
                "the pattern maps to two different known constellations",
                self.config.memory_risk,
                uncertainty * 0.75,
            ),
            llm_critique_action(
                "ask what would disprove this constellation",
                "LLM suggests a concrete split, merge, or missing-modality test",
                "LLM finds no discriminating observation",
                self.config.llm_risk,
                uncertainty * 0.4,
            ),
        ];
        if matches!(constellation.state, ConstellationState::SplitNeeded) {
            tests.extend(self.motion_or_diagnostic_tests(
                input,
                "rotate slightly and reobserve the fused cluster",
                "the cluster remains a single coherent object",
                "the cluster separates into multiple objects or tracks",
                uncertainty,
            ));
        } else {
            tests.push(wait_action(
                "wait for a missing modality or repeated co-occurrence",
                "missing member evidence appears with the same constellation",
                "the constellation dissolves or contradicts itself",
                self.config.wait_risk,
                uncertainty * 0.5,
            ));
        }
        ActiveLearningQuestion {
            id: format!(
                "active-learning:constellation:{}",
                stable_slug(&constellation.id)
            ),
            target_id: constellation.id.clone(),
            target_kind: ActiveLearningTargetKind::Constellation,
            question: "What observation would clarify this constellation?".to_string(),
            uncertainty,
            expected_information_gain: 0.0,
            risk: 0.0,
            proposed_tests: tests,
            state: ActiveLearningState::Open,
        }
    }

    fn place_question(
        &self,
        candidate: &PlaceRecognitionCandidate,
        input: &ActiveLearningInput,
    ) -> ActiveLearningQuestion {
        let uncertainty = (1.0 - candidate.confidence).clamp(0.0, 1.0);
        let mut tests = vec![replay_memory_action(
            "compare current scene vectors and place-cell evidence with prior place memory",
            "the same room has matching scene and entity anchors",
            "similar geometry lacks entity or scene-vector support",
            self.config.memory_risk,
            uncertainty * 0.9,
        )];
        tests.extend(self.motion_or_diagnostic_tests(
            input,
            "turn slightly and compare the place candidate again",
            "place anchors remain consistent after viewpoint change",
            "the candidate only matched from the original viewpoint",
            uncertainty * 0.7,
        ));
        ActiveLearningQuestion {
            id: format!(
                "active-learning:place:{}:{}",
                stable_slug(&candidate.cell.x.to_string()),
                stable_slug(&candidate.source_vector_id)
            ),
            target_id: candidate.source_vector_id.clone(),
            target_kind: ActiveLearningTargetKind::PlaceCandidate,
            question: "Is this the same place as the recalled candidate?".to_string(),
            uncertainty,
            expected_information_gain: 0.0,
            risk: 0.0,
            proposed_tests: tests,
            state: ActiveLearningState::Open,
        }
    }

    fn association_question(
        &self,
        edge: &AssociationEdge,
        _input: &ActiveLearningInput,
    ) -> ActiveLearningQuestion {
        let contradiction = edge.contradiction_count as f32 / edge.evidence_count.max(1) as f32;
        let uncertainty = (1.0 - edge.confidence).max(contradiction).clamp(0.0, 1.0);
        let tests = vec![
            replay_memory_action(
                "replay examples supporting and contradicting this association",
                "examples show a consistent condition that explains the contradiction",
                "examples remain mutually incompatible",
                self.config.memory_risk,
                uncertainty * 0.85,
            ),
            wait_action(
                "wait for the predicted item to appear or fail again",
                "the association predicts the next observation",
                "the expected observation does not appear",
                self.config.wait_risk,
                uncertainty * 0.45,
            ),
        ];
        ActiveLearningQuestion {
            id: format!("active-learning:association:{}", stable_slug(&edge.id)),
            target_id: edge.id.clone(),
            target_kind: ActiveLearningTargetKind::ActionOutcome,
            question: format!(
                "Does {} {:?} {} reliably?",
                edge.from_id, edge.relation, edge.to_id
            ),
            uncertainty,
            expected_information_gain: 0.0,
            risk: 0.0,
            proposed_tests: tests,
            state: ActiveLearningState::Open,
        }
    }

    fn prediction_failure_question(
        &self,
        failure: &PredictionFailure,
        input: &ActiveLearningInput,
    ) -> ActiveLearningQuestion {
        let uncertainty = failure
            .surprise
            .max(1.0 - failure.confidence)
            .clamp(0.0, 1.0);
        let mut tests = vec![
            replay_memory_action(
                "compare the failed prediction with similar past outcomes",
                "a past case explains the mismatch",
                "no past case explains the observed outcome",
                self.config.memory_risk,
                uncertainty * 0.75,
            ),
            llm_critique_action(
                "ask what hidden cause could explain this prediction failure",
                "LLM proposes a falsifiable hidden-state test",
                "LLM cannot separate the possible causes",
                self.config.llm_risk,
                uncertainty * 0.35,
            ),
        ];
        if failure.action.as_ref().is_some_and(is_motion_action) {
            if input.context.movement_readiness.allows_information_motion() {
                tests.push(wait_action(
                    "observe the next movement outcome before retrying any motion",
                    "the body reports matching odometry or velocity",
                    "the command path still produces no movement",
                    self.config.wait_risk,
                    uncertainty * 0.55,
                ));
            } else {
                tests.push(diagnostic_action(
                    "test command-to-base path before using motion for disambiguation",
                    "controller, base, and body state agree that motion commands can be attempted",
                    "safety, mode, base, controller, or body state still blocks motion",
                    input.context.movement_readiness.blocking_reason(),
                    uncertainty * 0.95,
                ));
            }
        }
        ActiveLearningQuestion {
            id: format!(
                "active-learning:prediction-failure:{}",
                stable_slug(&failure.id)
            ),
            target_id: failure.id.clone(),
            target_kind: ActiveLearningTargetKind::PredictionFailure,
            question: format!(
                "Why did prediction '{}' become observation '{}'?",
                failure.predicted, failure.observed
            ),
            uncertainty,
            expected_information_gain: 0.0,
            risk: 0.0,
            proposed_tests: tests,
            state: ActiveLearningState::Open,
        }
    }

    fn review_question(
        &self,
        review: &ActiveLearningReviewHint,
        _input: &ActiveLearningInput,
    ) -> ActiveLearningQuestion {
        let uncertainty = (1.0 - review.confidence)
            .max(if review.contradictions.is_empty() {
                0.25
            } else {
                0.65
            })
            .clamp(0.0, 1.0);
        let mut tests = review.suggested_tests.clone();
        tests.extend(review.human_review_prompts.iter().map(|prompt| {
            ask_human_action(
                prompt.clone(),
                "human review resolves the LLM-raised ambiguity",
                "human review rejects the LLM-raised interpretation",
                self.config.human_question_risk,
                uncertainty * 0.9,
            )
        }));
        if tests.is_empty() {
            tests.push(llm_critique_action(
                "ask for one concrete disambiguating test",
                "LLM proposes a low-risk observation that could separate hypotheses",
                "LLM cannot suggest a falsifiable next observation",
                self.config.llm_risk,
                uncertainty * 0.4,
            ));
        }
        ActiveLearningQuestion {
            id: format!("active-learning:review:{}", stable_slug(&review.id)),
            target_id: review.target_id.clone(),
            target_kind: review.target_kind.clone(),
            question: "Which review-suggested test should be tried next?".to_string(),
            uncertainty,
            expected_information_gain: 0.0,
            risk: 0.0,
            proposed_tests: tests,
            state: ActiveLearningState::Open,
        }
    }

    fn motion_or_diagnostic_tests(
        &self,
        input: &ActiveLearningInput,
        action_label: &str,
        expected: &str,
        disconfirming: &str,
        information_gain: f32,
    ) -> Vec<InformationGatheringAction> {
        if !input.context.movement_readiness.allows_information_motion() {
            return vec![diagnostic_action(
                "test command-to-base path before motion-based disambiguation",
                "motion stack reports safety, robot mode, base, controller, and body state ready",
                "one of safety, robot mode, base, controller, or body state remains blocked",
                input.context.movement_readiness.blocking_reason(),
                information_gain,
            )];
        }
        let action = input
            .context
            .available_actions
            .iter()
            .find(|action| is_motion_action(action))
            .cloned()
            .unwrap_or(ActionPrimitive::Turn {
                direction: TurnDir::Left,
                intensity: 0.12,
                duration_ms: 300,
            });
        vec![InformationGatheringAction {
            kind: ActiveLearningActionKind::MoveOrRotate,
            action: Some(action),
            human_question: None,
            expected_observation: expected.to_string(),
            disconfirming_observation: disconfirming.to_string(),
            required_safety_state: Some(
                "safety permits motion; robot mode, base, controller, and body state ready"
                    .to_string(),
            ),
            priority: information_gain * (1.0 - self.config.motion_risk),
        }
        .with_expected_label(action_label)]
    }
}

trait WithExpectedLabel {
    fn with_expected_label(self, label: &str) -> Self;
}

impl WithExpectedLabel for InformationGatheringAction {
    fn with_expected_label(mut self, label: &str) -> Self {
        self.expected_observation = format!("{label}: {}", self.expected_observation);
        self
    }
}

fn is_ambiguous_binding(candidate: &BindingCandidate) -> bool {
    matches!(
        candidate.decision,
        BindingDecision::HoldAmbiguous
            | BindingDecision::AskHuman
            | BindingDecision::CollectMoreEvidence
    ) || candidate.confidence < 0.6
        || candidate.evidence.iter().any(|evidence| {
            matches!(
                evidence.kind,
                BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
            )
        })
}

fn binding_uncertainty(candidate: &BindingCandidate) -> f32 {
    let contradiction = candidate
        .evidence
        .iter()
        .filter(|evidence| {
            matches!(
                evidence.kind,
                BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
            )
        })
        .map(|evidence| evidence.score)
        .fold(0.0_f32, f32::max);
    (1.0 - candidate.confidence)
        .max(contradiction)
        .clamp(0.0, 1.0)
}

fn binding_human_question(candidate: &BindingCandidate) -> String {
    match candidate.relation {
        BindingRelation::LikelySameEntity => format!(
            "Do {} and {} refer to the same entity?",
            candidate.left_cluster_id, candidate.right_cluster_id
        ),
        BindingRelation::NamedBy => format!(
            "Is {} the right label for {}?",
            candidate.left_cluster_id, candidate.right_cluster_id
        ),
        BindingRelation::ExplainsOutcome => format!(
            "Did {} cause or explain {}?",
            candidate.left_cluster_id, candidate.right_cluster_id
        ),
        _ => format!(
            "Is the proposed relationship between {} and {} correct?",
            candidate.left_cluster_id, candidate.right_cluster_id
        ),
    }
}

fn binding_benefits_from_viewpoint(candidate: &BindingCandidate) -> bool {
    matches!(
        candidate.relation,
        BindingRelation::ProjectsTo
            | BindingRelation::CooccursInEstimatedSpace
            | BindingRelation::HasColorAtPose
            | BindingRelation::MovesTogether
            | BindingRelation::LikelySameEntity
    )
}

fn ask_human_action(
    question: impl Into<String>,
    expected: impl Into<String>,
    disconfirming: impl Into<String>,
    risk: f32,
    information_gain: f32,
) -> InformationGatheringAction {
    InformationGatheringAction {
        kind: ActiveLearningActionKind::AskHuman,
        action: None,
        human_question: Some(question.into()),
        expected_observation: expected.into(),
        disconfirming_observation: disconfirming.into(),
        required_safety_state: None,
        priority: priority_from_gain_and_risk(information_gain, risk),
    }
}

fn wait_action(
    reason: impl Into<String>,
    expected: impl Into<String>,
    disconfirming: impl Into<String>,
    risk: f32,
    information_gain: f32,
) -> InformationGatheringAction {
    InformationGatheringAction {
        kind: ActiveLearningActionKind::WaitForEvidence,
        action: None,
        human_question: None,
        expected_observation: format!("{}: {}", reason.into(), expected.into()),
        disconfirming_observation: disconfirming.into(),
        required_safety_state: None,
        priority: priority_from_gain_and_risk(information_gain, risk),
    }
}

fn replay_memory_action(
    reason: impl Into<String>,
    expected: impl Into<String>,
    disconfirming: impl Into<String>,
    risk: f32,
    information_gain: f32,
) -> InformationGatheringAction {
    InformationGatheringAction {
        kind: ActiveLearningActionKind::ReplayMemory,
        action: None,
        human_question: None,
        expected_observation: format!("{}: {}", reason.into(), expected.into()),
        disconfirming_observation: disconfirming.into(),
        required_safety_state: None,
        priority: priority_from_gain_and_risk(information_gain, risk),
    }
}

fn llm_critique_action(
    reason: impl Into<String>,
    expected: impl Into<String>,
    disconfirming: impl Into<String>,
    risk: f32,
    information_gain: f32,
) -> InformationGatheringAction {
    InformationGatheringAction {
        kind: ActiveLearningActionKind::RequestLlmCritique,
        action: None,
        human_question: None,
        expected_observation: format!("{}: {}", reason.into(), expected.into()),
        disconfirming_observation: disconfirming.into(),
        required_safety_state: None,
        priority: priority_from_gain_and_risk(information_gain, risk),
    }
}

fn diagnostic_action(
    reason: impl Into<String>,
    expected: impl Into<String>,
    disconfirming: impl Into<String>,
    blocking_reason: impl Into<String>,
    information_gain: f32,
) -> InformationGatheringAction {
    InformationGatheringAction {
        kind: ActiveLearningActionKind::Diagnostic,
        action: None,
        human_question: None,
        expected_observation: format!("{}: {}", reason.into(), expected.into()),
        disconfirming_observation: disconfirming.into(),
        required_safety_state: Some(blocking_reason.into()),
        priority: priority_from_gain_and_risk(information_gain, 0.04),
    }
}

fn priority_from_gain_and_risk(information_gain: f32, risk: f32) -> f32 {
    (information_gain.clamp(0.0, 1.0) * (1.0 - risk.clamp(0.0, 1.0))).clamp(0.0, 1.0)
}

fn information_action_id(target_id: &str, action: &InformationGatheringAction) -> String {
    let detail = action
        .human_question
        .as_deref()
        .or_else(|| Some(action.expected_observation.as_str()))
        .unwrap_or_default();
    format!(
        "active-test:{}:{}:{}",
        stable_slug(target_id),
        active_learning_action_kind_slug(&action.kind),
        stable_slug(detail)
    )
}

fn active_learning_action_kind_slug(kind: &ActiveLearningActionKind) -> &'static str {
    match kind {
        ActiveLearningActionKind::AskHuman => "ask-human",
        ActiveLearningActionKind::MoveOrRotate => "move-or-rotate",
        ActiveLearningActionKind::WaitForEvidence => "wait",
        ActiveLearningActionKind::ReplayMemory => "replay-memory",
        ActiveLearningActionKind::RequestLlmCritique => "llm-critique",
        ActiveLearningActionKind::Diagnostic => "diagnostic",
        ActiveLearningActionKind::Other => "other",
    }
}

fn action_risk(action: &InformationGatheringAction, config: &ActiveLearningPlannerConfig) -> f32 {
    match action.kind {
        ActiveLearningActionKind::AskHuman => config.human_question_risk,
        ActiveLearningActionKind::MoveOrRotate => config.motion_risk,
        ActiveLearningActionKind::WaitForEvidence => config.wait_risk,
        ActiveLearningActionKind::ReplayMemory => config.memory_risk,
        ActiveLearningActionKind::RequestLlmCritique => config.llm_risk,
        ActiveLearningActionKind::Diagnostic => 0.04,
        ActiveLearningActionKind::Other => 0.1,
    }
}

fn infer_active_learning_state(question: &ActiveLearningQuestion) -> ActiveLearningState {
    if question.proposed_tests.iter().any(|test| {
        matches!(
            test.kind,
            ActiveLearningActionKind::Diagnostic | ActiveLearningActionKind::MoveOrRotate
        ) && test.required_safety_state.is_some()
    }) && question
        .proposed_tests
        .iter()
        .all(|test| test.kind == ActiveLearningActionKind::Diagnostic)
    {
        ActiveLearningState::WaitingForSafety
    } else if question
        .best_test()
        .is_some_and(|test| test.kind == ActiveLearningActionKind::AskHuman)
    {
        ActiveLearningState::WaitingForHuman
    } else {
        ActiveLearningState::Open
    }
}

fn active_learning_score(question: &ActiveLearningQuestion) -> f32 {
    (question.uncertainty * 0.55 + question.expected_information_gain * 0.35 - question.risk * 0.10)
        .clamp(0.0, 1.0)
}

fn is_motion_action(action: &ActionPrimitive) -> bool {
    matches!(
        action,
        ActionPrimitive::Go { .. }
            | ActionPrimitive::Drive { .. }
            | ActionPrimitive::Turn { .. }
            | ActionPrimitive::Approach { .. }
            | ActionPrimitive::Dock
            | ActionPrimitive::Explore { .. }
    )
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssociationLearningConfig {
    pub same_moment_window_ms: u64,
    pub short_sequence_window_ms: u64,
    pub long_sequence_window_ms: u64,
    pub max_recent_observations: usize,
    pub decay_per_tick: f32,
    pub min_prediction_gain: f32,
}

impl Default for AssociationLearningConfig {
    fn default() -> Self {
        Self {
            same_moment_window_ms: 0,
            short_sequence_window_ms: 2_000,
            long_sequence_window_ms: 10_000,
            max_recent_observations: 32,
            decay_per_tick: 0.025,
            min_prediction_gain: 0.02,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct AssociationItemStats {
    present_count: u32,
    last_seen_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssociationLearningEngine {
    pub edges: BTreeMap<String, AssociationEdge>,
    pub config: AssociationLearningConfig,
    recent: VecDeque<AssociationObservation>,
    item_stats: BTreeMap<String, AssociationItemStats>,
    observation_count: u32,
}

impl Default for AssociationLearningEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AssociationLearningEngine {
    pub fn new() -> Self {
        Self {
            edges: BTreeMap::new(),
            config: AssociationLearningConfig::default(),
            recent: VecDeque::new(),
            item_stats: BTreeMap::new(),
            observation_count: 0,
        }
    }

    pub fn with_config(config: AssociationLearningConfig) -> Self {
        Self {
            edges: BTreeMap::new(),
            config,
            recent: VecDeque::new(),
            item_stats: BTreeMap::new(),
            observation_count: 0,
        }
    }

    pub fn observe(&mut self, observation: AssociationObservation) -> Vec<AssociationEdge> {
        self.observation_count = self.observation_count.saturating_add(1);
        let current_items = observation.all_items();
        for item in &current_items {
            let stats = self.item_stats.entry(item.id.clone()).or_default();
            stats.present_count = stats.present_count.saturating_add(1);
            stats.last_seen_ms = observation.t_ms;
        }

        self.learn_cooccurrences(&observation, &current_items);
        self.learn_sequences(&observation, &current_items);
        self.learn_negative_evidence(&observation);

        self.recent.push_back(observation);
        while self.recent.len() > self.config.max_recent_observations {
            self.recent.pop_front();
        }
        self.edges.values().cloned().collect()
    }

    pub fn decay(&mut self, ticks: u64) {
        let amount = (self.config.decay_per_tick * ticks as f32).clamp(0.0, 0.95);
        for edge in self.edges.values_mut() {
            edge.weaken(amount);
        }
    }

    pub fn predictions_for(
        &self,
        active_ids: &[String],
        min_confidence: f32,
        limit: usize,
    ) -> Vec<AssociationPrediction> {
        let active = active_ids.iter().cloned().collect::<BTreeSet<_>>();
        let mut predictions = self
            .edges
            .values()
            .filter(|edge| active.contains(&edge.from_id))
            .filter(|edge| {
                matches!(
                    edge.relation,
                    AssociationRelation::Predicts
                        | AssociationRelation::Follows
                        | AssociationRelation::Enables
                        | AssociationRelation::Explains
                )
            })
            .filter(|edge| edge.confidence >= min_confidence)
            .map(|edge| AssociationPrediction {
                source_id: edge.from_id.clone(),
                predicted_id: edge.to_id.clone(),
                relation: edge.relation.clone(),
                confidence: edge.confidence,
                prediction_gain: edge.prediction_gain,
                evidence_count: edge.evidence_count,
                reason: format!(
                    "{} {} {} with gain {:.2}",
                    edge.from_id,
                    association_relation_slug(&edge.relation),
                    edge.to_id,
                    edge.prediction_gain
                ),
            })
            .collect::<Vec<_>>();
        predictions.sort_by(|left, right| {
            right
                .confidence
                .partial_cmp(&left.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    right
                        .prediction_gain
                        .partial_cmp(&left.prediction_gain)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        predictions.truncate(limit);
        predictions
    }

    fn learn_cooccurrences(
        &mut self,
        observation: &AssociationObservation,
        current_items: &[AssociationItem],
    ) {
        for left_index in 0..current_items.len() {
            for right_index in (left_index + 1)..current_items.len() {
                let left = &current_items[left_index];
                let right = &current_items[right_index];
                let (from, to) = canonical_association_pair(left, right);
                self.upsert_association(
                    from,
                    to,
                    AssociationRelation::CoOccursWith,
                    AssociationExample {
                        frame_id: observation.frame_id.clone(),
                        t_ms: observation.t_ms,
                        reason: "items appeared in the same observation window".to_string(),
                        score: left.confidence.min(right.confidence),
                    },
                );
            }
        }
    }

    fn learn_sequences(
        &mut self,
        observation: &AssociationObservation,
        current_items: &[AssociationItem],
    ) {
        let recent = self.recent.iter().cloned().collect::<Vec<_>>();
        for prior in recent.iter().rev() {
            let lag_ms = observation.t_ms.saturating_sub(prior.t_ms);
            if lag_ms > self.config.long_sequence_window_ms {
                break;
            }
            let prior_items = prior.all_items();
            for from in &prior_items {
                for to in current_items {
                    if from.id == to.id {
                        continue;
                    }
                    let relation = sequence_relation(to, lag_ms, &self.config);
                    self.upsert_association(
                        &from.id,
                        &to.id,
                        relation,
                        AssociationExample {
                            frame_id: observation.frame_id.clone(),
                            t_ms: observation.t_ms,
                            reason: format!("{} preceded {} by {lag_ms} ms", from.id, to.id),
                            score: from.confidence.min(to.confidence)
                                * lag_score_for_association(lag_ms),
                        },
                    );
                }
            }
        }
    }

    fn learn_negative_evidence(&mut self, observation: &AssociationObservation) {
        for item in &observation.negative_evidence {
            let relation = match item.relation {
                AssociationRelation::Suppresses
                | AssociationRelation::Contradicts
                | AssociationRelation::Prevents => item.relation.clone(),
                _ => AssociationRelation::Suppresses,
            };
            let edge = self.upsert_association(
                &item.present_id,
                &item.absent_id,
                relation.clone(),
                AssociationExample {
                    frame_id: observation.frame_id.clone(),
                    t_ms: observation.t_ms,
                    reason: item.reason.clone(),
                    score: item.score.clamp(0.0, 1.0),
                },
            );
            if relation == AssociationRelation::Contradicts {
                edge.add_contradiction(AssociationExample {
                    frame_id: observation.frame_id.clone(),
                    t_ms: observation.t_ms,
                    reason: item.reason.clone(),
                    score: item.score.clamp(0.0, 1.0),
                });
            }
        }
    }

    fn upsert_association(
        &mut self,
        from_id: &str,
        to_id: &str,
        relation: AssociationRelation,
        example: AssociationExample,
    ) -> &mut AssociationEdge {
        let id = association_edge_id(from_id, to_id, &relation);
        let prediction_gain =
            self.prediction_gain_estimate(from_id, to_id, example.score.clamp(0.0, 1.0));
        let edge = self.edges.entry(id).or_insert_with(|| {
            AssociationEdge::new(
                from_id.to_string(),
                to_id.to_string(),
                relation,
                example.clone(),
            )
        });
        edge.strengthen(example, prediction_gain);
        edge
    }

    fn prediction_gain_estimate(&self, from_id: &str, to_id: &str, fallback_score: f32) -> f32 {
        let from_count = self
            .item_stats
            .get(from_id)
            .map(|stats| stats.present_count)
            .unwrap_or(1)
            .max(1) as f32;
        let to_count = self
            .item_stats
            .get(to_id)
            .map(|stats| stats.present_count)
            .unwrap_or(0) as f32;
        let total = self.observation_count.max(1) as f32;
        let edge_count = self
            .edges
            .get(&association_edge_id(
                from_id,
                to_id,
                &AssociationRelation::Predicts,
            ))
            .or_else(|| {
                self.edges.get(&association_edge_id(
                    from_id,
                    to_id,
                    &AssociationRelation::Follows,
                ))
            })
            .or_else(|| {
                self.edges.get(&association_edge_id(
                    from_id,
                    to_id,
                    &AssociationRelation::CoOccursWith,
                ))
            })
            .map(|edge| edge.evidence_count as f32)
            .unwrap_or(0.0)
            + 1.0;
        let p_b = (to_count / total).clamp(0.0, 1.0);
        let p_b_given_a = (edge_count / from_count).clamp(0.0, 1.0);
        let gain = (p_b_given_a - p_b).max(0.0);
        gain.max(approximate_mutual_information(p_b, p_b_given_a))
            .max(fallback_score * self.config.min_prediction_gain)
            .clamp(0.0, 1.0)
    }
}

/// A provisional, persistent record of an observed entity.
///
/// Entities begin as thin hypotheses from a single detection and grow stronger
/// as repeated observations merge into the same record.  Multiple sensing
/// modalities (face, voice, depth/motion, text) may support the same entity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntityHypothesis {
    /// Stable identifier derived from entity class + label.
    pub id: String,
    /// Coarse semantic class (e.g. "person", "obstacle", "charger").
    pub kind: String,
    /// Labels seen for this entity, most-recently observed first.
    pub labels: Vec<String>,
    /// Provisional display name (may carry a trailing `?` when uncertain).
    pub display_name: Option<String>,
    /// Millisecond timestamp of the very first observation.
    pub first_seen_ms: u64,
    /// Millisecond timestamp of the most recent observation.
    pub last_seen_ms: u64,
    /// Total number of individual observations merged into this record.
    pub observation_count: u32,
    /// Belief strength in [0, 1].  Increases on re-observation, decays over time.
    pub confidence: f32,
    /// Current lifecycle state.
    pub lifecycle: EntityLifecycleState,
    /// Map cells where this entity has been observed.
    pub location_cells: Vec<PlaceCellKey>,
    /// Cross-modal evidence links.
    pub modality_support: ModalitySupport,
    /// Entity-centered SLAM graph over recurring multimodal clusters.
    #[serde(default)]
    pub constellation: EntityConstellation,
}

impl EntityHypothesis {
    /// Create a new hypothesis from a single `ObjectObservation`.
    pub fn from_observation(
        observation: &ObjectObservation,
        t_ms: u64,
        cell_key: Option<PlaceCellKey>,
    ) -> Self {
        let kind = object_class_slug(&observation.class).to_string();
        let id = format!("entity:{}:{}", kind, stable_slug(&observation.label));
        let label = observation.label.clone();
        let display_name = Some(label.clone());
        let location_cells = cell_key.into_iter().collect();
        let mut entity = Self {
            id,
            kind,
            labels: vec![label],
            display_name,
            first_seen_ms: t_ms,
            last_seen_ms: t_ms,
            observation_count: 1,
            confidence: observation.confidence.clamp(0.0, 1.0),
            lifecycle: EntityLifecycleState::Active,
            location_cells,
            modality_support: ModalitySupport::default(),
            constellation: EntityConstellation::default(),
        };
        let point = entity.push_observation_point(
            Modality::Vision,
            format!("object:{}", observation.label),
            observation.confidence,
            t_ms,
        );
        entity.upsert_cluster(
            Modality::Vision,
            format!("object:{}", stable_slug(&observation.label)),
            point,
            observation.confidence,
        );
        entity
    }

    /// Merge a new observation into this existing hypothesis.
    ///
    /// Confidence is nudged upward; repeated observations strengthen the record.
    pub fn merge_observation(
        &mut self,
        observation: &ObjectObservation,
        t_ms: u64,
        cell_key: Option<PlaceCellKey>,
    ) {
        let was_inactive = self.lifecycle != EntityLifecycleState::Active;
        self.last_seen_ms = t_ms;
        self.observation_count = self.observation_count.saturating_add(1);
        // Exponential moving average biased toward the new value on re-sighting.
        self.confidence =
            (self.confidence * 0.7 + observation.confidence.clamp(0.0, 1.0) * 0.3).clamp(0.0, 1.0);
        self.lifecycle = EntityLifecycleState::Active;
        if was_inactive {
            self.constellation.state = EntityConstellationState::Revived;
        }
        if !self.labels.contains(&observation.label) {
            self.labels.insert(0, observation.label.clone());
        }
        if let Some(key) = cell_key {
            if !self.location_cells.contains(&key) {
                self.location_cells.push(key);
            }
        }
        let point = self.push_observation_point(
            Modality::Vision,
            format!("object:{}", observation.label),
            observation.confidence,
            t_ms,
        );
        self.upsert_cluster(
            Modality::Vision,
            format!("object:{}", stable_slug(&observation.label)),
            point,
            observation.confidence,
        );
    }

    /// Add face vector evidence.
    pub fn add_face_vector(&mut self, point_id: impl Into<String>) -> String {
        let id = point_id.into();
        if !self.modality_support.face_vector_ids.contains(&id) {
            self.modality_support.face_vector_ids.push(id.clone());
        }
        let point = self.push_observation_point(
            Modality::Vision,
            format!("face:{id}"),
            0.8,
            self.last_seen_ms,
        );
        self.upsert_cluster(Modality::Vision, format!("face:{id}"), point, 0.8)
    }

    /// Add object vector evidence.
    pub fn add_object_vector(&mut self, point_id: impl Into<String>) -> String {
        let id = point_id.into();
        if !self.modality_support.object_vector_ids.contains(&id) {
            self.modality_support.object_vector_ids.push(id.clone());
        }
        let point = self.push_observation_point(
            Modality::Vision,
            format!("object-vector:{id}"),
            0.75,
            self.last_seen_ms,
        );
        self.upsert_cluster(Modality::Vision, format!("object-vector:{id}"), point, 0.75)
    }

    /// Add voice vector evidence.
    pub fn add_voice_vector(&mut self, point_id: impl Into<String>) -> String {
        let id = point_id.into();
        if !self.modality_support.voice_vector_ids.contains(&id) {
            self.modality_support.voice_vector_ids.push(id.clone());
        }
        let point = self.push_observation_point(
            Modality::Audio,
            format!("voice:{id}"),
            0.8,
            self.last_seen_ms,
        );
        self.upsert_cluster(Modality::Audio, format!("voice:{id}"), point, 0.8)
    }

    /// Add scene/depth vector evidence.
    pub fn add_scene_vector(&mut self, point_id: impl Into<String>) -> String {
        let id = point_id.into();
        if !self.modality_support.scene_vector_ids.contains(&id) {
            self.modality_support.scene_vector_ids.push(id.clone());
        }
        let point = self.push_observation_point(
            Modality::Depth,
            format!("scene:{id}"),
            0.75,
            self.last_seen_ms,
        );
        self.upsert_cluster(Modality::Depth, format!("scene:{id}"), point, 0.75)
    }

    pub fn add_text_label(&mut self, label: impl Into<String>, confidence: f32, t_ms: u64) {
        let text = label.into().trim().to_string();
        if text.is_empty() {
            return;
        }
        if !self.modality_support.text_labels.contains(&text) {
            self.modality_support.text_labels.push(text.clone());
        }
        if self.display_name.is_none() {
            self.display_name = Some(format!("{text}?"));
        }
        let point = self.push_observation_point(
            Modality::Language,
            format!("text:{text}"),
            confidence,
            t_ms,
        );
        let text_cluster = self.upsert_cluster(
            Modality::Language,
            format!("text:{}", stable_slug(&text)),
            point,
            confidence,
        );
        self.bind_with_object_cluster(text_cluster, BindingRelation::NamedBy, confidence, t_ms);
    }

    fn push_observation_point(
        &mut self,
        modality: Modality,
        source: String,
        confidence: f32,
        t_ms: u64,
    ) -> String {
        let point_id = format!(
            "point:{}:{}:{}",
            modality.as_str(),
            stable_slug(&source),
            self.constellation.observation_points.len() + 1
        );
        self.constellation
            .observation_points
            .push(ObservationPoint {
                id: point_id.clone(),
                modality,
                source,
                observed_at_ms: t_ms,
                confidence: confidence.clamp(0.0, 1.0),
            });
        point_id
    }

    fn upsert_cluster(
        &mut self,
        modality: Modality,
        cluster_key: String,
        point_id: String,
        confidence: f32,
    ) -> String {
        let cluster_id = format!(
            "cluster:{}:{}",
            modality.as_str(),
            stable_slug(&cluster_key)
        );
        if let Some(cluster) = self
            .constellation
            .modality_clusters
            .iter_mut()
            .find(|cluster| cluster.id == cluster_id)
        {
            if !cluster.observation_point_ids.contains(&point_id) {
                cluster.observation_point_ids.push(point_id);
            }
            cluster.evidence_count = cluster.evidence_count.saturating_add(1);
            cluster.confidence =
                (cluster.confidence * 0.7 + confidence.clamp(0.0, 1.0) * 0.3).clamp(0.0, 1.0);
        } else {
            self.constellation.modality_clusters.push(ModalityCluster {
                id: cluster_id.clone(),
                modality,
                observation_point_ids: vec![point_id],
                evidence_count: 1,
                confidence: confidence.clamp(0.0, 1.0),
            });
        }
        cluster_id
    }

    fn bind_with_object_cluster(
        &mut self,
        cluster_id: String,
        relation: BindingRelation,
        confidence: f32,
        t_ms: u64,
    ) {
        let Some(object_cluster_id) = self
            .constellation
            .modality_clusters
            .iter()
            .find(|cluster| cluster.id.starts_with("cluster:vision:object"))
            .map(|cluster| cluster.id.clone())
        else {
            return;
        };
        if object_cluster_id == cluster_id {
            return;
        }
        let (left_cluster_id, right_cluster_id) = if object_cluster_id <= cluster_id {
            (object_cluster_id, cluster_id)
        } else {
            (cluster_id, object_cluster_id)
        };
        self.upsert_binding_edge(
            left_cluster_id,
            right_cluster_id,
            relation,
            confidence,
            t_ms,
        );
    }

    fn primary_object_cluster_id(&self) -> Option<String> {
        self.constellation
            .modality_clusters
            .iter()
            .find(|cluster| cluster.id.starts_with("cluster:vision:object"))
            .map(|cluster| cluster.id.clone())
    }

    pub fn upsert_binding_edge(
        &mut self,
        left_cluster_id: String,
        right_cluster_id: String,
        relation: BindingRelation,
        confidence: f32,
        t_ms: u64,
    ) -> BindingEdgeResult {
        let (left_cluster_id, right_cluster_id) = if left_cluster_id <= right_cluster_id {
            (left_cluster_id, right_cluster_id)
        } else {
            (right_cluster_id, left_cluster_id)
        };
        if let Some(index) = self.constellation.binding_edges.iter().position(|edge| {
            edge.left_cluster_id == left_cluster_id
                && edge.right_cluster_id == right_cluster_id
                && edge.relation == relation
        }) {
            self.constellation.binding_edges[index].strengthen(confidence, t_ms);
            let edge = self.constellation.binding_edges[index].clone();
            self.refresh_constellation_state();
            return BindingEdgeResult {
                edge,
                created: false,
            };
        }

        let mut edge = BindingEdge {
            left_cluster_id,
            right_cluster_id,
            relation,
            confidence: 0.1,
            evidence_count: 0,
            decay_per_tick: 0.01,
            last_seen_ms: t_ms,
        };
        edge.strengthen(confidence, t_ms);
        self.constellation.binding_edges.push(edge.clone());
        self.refresh_constellation_state();
        BindingEdgeResult {
            edge,
            created: true,
        }
    }

    fn record_binding_candidate(&mut self, candidate: BindingCandidate) {
        self.constellation.binding_candidates.push(candidate);
        const MAX_BINDING_CANDIDATES: usize = 64;
        if self.constellation.binding_candidates.len() > MAX_BINDING_CANDIDATES {
            let excess = self.constellation.binding_candidates.len() - MAX_BINDING_CANDIDATES;
            self.constellation.binding_candidates.drain(0..excess);
        }
    }

    fn refresh_constellation_state(&mut self) {
        if matches!(
            self.constellation.state,
            EntityConstellationState::Merged
                | EntityConstellationState::Split
                | EntityConstellationState::Vanished
        ) {
            return;
        }
        let strong_edges = self
            .constellation
            .binding_edges
            .iter()
            .filter(|edge| edge.is_strong())
            .count();
        let total_edge_evidence = self
            .constellation
            .binding_edges
            .iter()
            .map(|edge| edge.evidence_count)
            .sum::<u32>();
        let active_modalities = self.modality_support.active_modalities();
        let has_major_contradiction =
            self.constellation
                .binding_candidates
                .iter()
                .any(|candidate| {
                    candidate.decision == BindingDecision::Reject
                        && candidate.evidence.iter().any(|evidence| {
                            matches!(
                                evidence.kind,
                                BindingEvidenceKind::Contradiction
                                    | BindingEvidenceKind::SimultaneousConflict
                            )
                        })
                });
        self.constellation.state = if !has_major_contradiction
            && (strong_edges >= 2
                || (self.constellation.binding_edges.len() >= 2
                    && active_modalities >= 3
                    && total_edge_evidence >= 3))
        {
            EntityConstellationState::Strong
        } else {
            EntityConstellationState::Weak
        };
    }

    fn decay_bindings(&mut self, decay_factor: f32) {
        for edge in &mut self.constellation.binding_edges {
            edge.weaken(decay_factor * edge.decay_per_tick.max(0.01));
        }
        self.refresh_constellation_state();
    }
}

/// A lightweight summary of one entity for API responses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntityHypothesisSummary {
    pub id: String,
    pub kind: String,
    pub display_name: Option<String>,
    pub labels: Vec<String>,
    pub text_labels: Vec<String>,
    pub confidence: f32,
    pub lifecycle: EntityLifecycleState,
    pub observation_count: u32,
    pub first_seen_ms: u64,
    pub last_seen_ms: u64,
    pub location_cells: Vec<PlaceCellKey>,
    pub active_modalities: usize,
    pub constellation_state: EntityConstellationState,
    pub observation_points: Vec<ObservationPoint>,
    pub modality_clusters: Vec<ModalityCluster>,
    pub binding_edges: Vec<BindingEdge>,
    #[serde(default)]
    pub accepted_binding_candidates: Vec<BindingCandidate>,
    #[serde(default)]
    pub ambiguous_binding_candidates: Vec<BindingCandidate>,
    #[serde(default)]
    pub rejected_binding_candidates: Vec<BindingCandidate>,
}

impl From<&EntityHypothesis> for EntityHypothesisSummary {
    fn from(h: &EntityHypothesis) -> Self {
        Self {
            id: h.id.clone(),
            kind: h.kind.clone(),
            display_name: h.display_name.clone(),
            labels: h.labels.clone(),
            text_labels: h.modality_support.text_labels.clone(),
            confidence: h.confidence,
            lifecycle: h.lifecycle.clone(),
            observation_count: h.observation_count,
            first_seen_ms: h.first_seen_ms,
            last_seen_ms: h.last_seen_ms,
            location_cells: h.location_cells.clone(),
            active_modalities: h.modality_support.active_modalities(),
            constellation_state: h.constellation.state.clone(),
            observation_points: h.constellation.observation_points.clone(),
            modality_clusters: h.constellation.modality_clusters.clone(),
            binding_edges: h.constellation.binding_edges.clone(),
            accepted_binding_candidates: h
                .constellation
                .binding_candidates
                .iter()
                .filter(|candidate| candidate.decision == BindingDecision::Accept)
                .cloned()
                .collect(),
            ambiguous_binding_candidates: h
                .constellation
                .binding_candidates
                .iter()
                .filter(|candidate| {
                    matches!(
                        candidate.decision,
                        BindingDecision::HoldAmbiguous
                            | BindingDecision::AskHuman
                            | BindingDecision::CollectMoreEvidence
                    )
                })
                .cloned()
                .collect(),
            rejected_binding_candidates: h
                .constellation
                .binding_candidates
                .iter()
                .filter(|candidate| candidate.decision == BindingDecision::Reject)
                .cloned()
                .collect(),
        }
    }
}

/// Dashboard-level report over all entity hypotheses.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EntityMemoryReport {
    pub total_entities: usize,
    pub active_entities: usize,
    pub occluded_entities: usize,
    pub vanished_entities: usize,
    #[serde(default)]
    pub active_tracking_hypotheses: Vec<TrackingHypothesis>,
    #[serde(default)]
    pub review_tracking_hypotheses: Vec<TrackingHypothesis>,
    #[serde(default)]
    pub promoted_tracking_hypotheses: Vec<TrackingHypothesis>,
    #[serde(default)]
    pub expired_tracking_hypotheses: Vec<TrackingHypothesis>,
    #[serde(default)]
    pub accepted_binding_candidates: Vec<BindingCandidate>,
    #[serde(default)]
    pub ambiguous_binding_candidates: Vec<BindingCandidate>,
    #[serde(default)]
    pub rejected_binding_candidates: Vec<BindingCandidate>,
    /// Top entities ranked by confidence (active ones first).
    pub top_entities: Vec<EntityHypothesisSummary>,
}

const ENTITY_CONFIDENCE_DECAY_PER_TICK: f32 = 0.998;
const ENTITY_OCCLUDE_THRESHOLD: f32 = 0.25;
const ENTITY_VANISH_THRESHOLD: f32 = 0.05;
const HYPOTHESIS_CONFIDENCE_DECAY_PER_TICK: f32 = 0.999;
const HYPOTHESIS_PROMOTION_THRESHOLD: f32 = 0.72;
const HYPOTHESIS_REVIEW_MARGIN: f32 = 0.08;
const HYPOTHESIS_STALE_MS: u64 = 30_000;
const HYPOTHESIS_REVIEW_STALE_MS: u64 = 120_000;

/// Stores and maintains all persistent entity hypotheses.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EntityMemory {
    /// All known entity records keyed by entity id.
    pub entities: BTreeMap<String, EntityHypothesis>,
    #[serde(default)]
    pub binding_candidates: Vec<BindingCandidate>,
    #[serde(default)]
    pub tracking_hypotheses: BTreeMap<String, TrackingHypothesis>,
    last_tick: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VectorBindingKind {
    Face,
    Voice,
    Scene,
}

impl EntityMemory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a single `Now` snapshot: merge object observations and update
    /// cross-modal evidence.
    pub fn observe_now(&mut self, now: &Now, cell_key: Option<PlaceCellKey>) {
        let elapsed_ticks = self
            .last_tick
            .map(|last| now.t_ms.saturating_sub(last) / 100)
            .unwrap_or(1)
            .max(1);
        self.decay(elapsed_ticks, now.t_ms);
        self.last_tick = Some(now.t_ms);

        for observation in &now.objects.observations {
            let kind = object_class_slug(&observation.class).to_string();
            let id = format!("entity:{}:{}", kind, stable_slug(&observation.label));
            if let Some(existing) = self.entities.get_mut(&id) {
                existing.merge_observation(observation, now.t_ms, cell_key);
            } else {
                let hypothesis =
                    EntityHypothesis::from_observation(observation, now.t_ms, cell_key);
                self.entities.insert(id, hypothesis);
            }
        }

        let current_entity_ids = now
            .objects
            .observations
            .iter()
            .map(|observation| {
                format!(
                    "entity:{}:{}",
                    object_class_slug(&observation.class),
                    stable_slug(&observation.label)
                )
            })
            .collect::<BTreeSet<_>>();

        // Face vectors propose person bindings; they do not fan out to every person.
        for artifact in &now.face.vectors {
            self.admit_vector_artifact(
                artifact,
                VectorBindingKind::Face,
                now.t_ms,
                cell_key,
                &current_entity_ids,
            );
        }

        // Attach object vectors to active non-person entities, or to an explicit source entity.
        for artifact in &now.objects.vectors {
            let object_ids: Vec<String> = if let Some(source_id) = artifact.source_id.as_ref() {
                vec![source_id.clone()]
            } else {
                self.entities
                    .values()
                    .filter(|entity| {
                        entity.lifecycle == EntityLifecycleState::Active
                            && !entity.id.starts_with("entity:person:")
                    })
                    .map(|entity| entity.id.clone())
                    .collect()
            };
            for id in object_ids {
                if let Some(entity) = self.entities.get_mut(&id) {
                    entity.add_object_vector(&artifact.point_id);
                }
            }
        }

        // Voice vectors propose speaker bindings; ambiguity is preserved for review.
        for artifact in &now.voice.vectors {
            self.admit_vector_artifact(
                artifact,
                VectorBindingKind::Voice,
                now.t_ms,
                cell_key,
                &current_entity_ids,
            );
        }

        // Scene vectors bind only when there is explicit spatial/object context.
        for artifact in &now.eye.scene_vectors {
            self.admit_vector_artifact(
                artifact,
                VectorBindingKind::Scene,
                now.t_ms,
                cell_key,
                &current_entity_ids,
            );
        }

        let text_labels = now
            .ear
            .transcript
            .as_ref()
            .into_iter()
            .chain(now.ear.asr.transcript.as_ref())
            .map(|text| text.trim())
            .filter(|text| !text.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();
        if !text_labels.is_empty() {
            let active_ids = self
                .entities
                .values()
                .filter(|entity| entity.lifecycle == EntityLifecycleState::Active)
                .map(|entity| entity.id.clone())
                .collect::<Vec<_>>();
            for id in active_ids {
                if let Some(entity) = self.entities.get_mut(&id) {
                    for text in &text_labels {
                        entity.add_text_label(text.clone(), 0.6, now.t_ms);
                    }
                }
            }
        }
    }

    fn admit_vector_artifact(
        &mut self,
        artifact: &VectorArtifact,
        kind: VectorBindingKind,
        t_ms: u64,
        cell_key: Option<PlaceCellKey>,
        current_entity_ids: &BTreeSet<String>,
    ) {
        let plausible_ids = self.plausible_entity_ids(artifact, kind, t_ms, cell_key);
        if plausible_ids.is_empty() {
            let reason = match kind {
                VectorBindingKind::Face => "face vector observed but no plausible person entity",
                VectorBindingKind::Voice => "voice observed but no plausible person entity",
                VectorBindingKind::Scene => {
                    "scene vector active but no spatially compatible object cluster"
                }
            };
            self.record_binding_candidate(BindingCandidate {
                left_cluster_id: "unresolved".to_string(),
                right_cluster_id: vector_cluster_id(kind, &artifact.point_id),
                relation: BindingRelation::RequiresReview,
                evidence: vec![BindingEvidence {
                    kind: BindingEvidenceKind::VectorSimilarity,
                    score: 0.25,
                    reason: "single vector artifact without compatible entity context".to_string(),
                }],
                confidence: 0.0,
                decision: BindingDecision::CollectMoreEvidence,
                reason: reason.to_string(),
            });
            self.upsert_new_entity_hypothesis(artifact, kind, t_ms);
            return;
        }

        let family_id = tracking_family_id(kind, &artifact.point_id);
        let mut candidate_ids = Vec::new();
        for entity_id in plausible_ids.clone() {
            let Some(entity) = self.entities.get(&entity_id) else {
                continue;
            };
            let Some(object_cluster_id) = entity.primary_object_cluster_id() else {
                continue;
            };
            let right_cluster_id = vector_cluster_id(kind, &artifact.point_id);
            let candidate = qualify_binding_candidate(
                entity,
                artifact,
                kind,
                object_cluster_id,
                right_cluster_id,
                t_ms,
                cell_key,
                plausible_ids.len(),
                current_entity_ids.contains(&entity_id),
            );
            let candidate_id = binding_candidate_id(&candidate);
            candidate_ids.push(candidate_id.clone());
            if let Some(entity) = self.entities.get_mut(&entity_id) {
                entity.record_binding_candidate(candidate.clone());
            }
            self.upsert_tracking_hypothesis(
                tracking_kind_from_vector(kind),
                family_id.clone(),
                Some(entity_id),
                artifact.point_id.clone(),
                candidate_id,
                candidate.evidence,
                t_ms,
            );
        }
        self.upsert_unknown_competitor_hypothesis(artifact, kind, &family_id, &candidate_ids, t_ms);
        self.evaluate_hypothesis_family(&family_id, artifact, kind, t_ms);
    }

    fn plausible_entity_ids(
        &self,
        artifact: &VectorArtifact,
        kind: VectorBindingKind,
        t_ms: u64,
        cell_key: Option<PlaceCellKey>,
    ) -> Vec<String> {
        if let Some(source_id) = artifact.source_id.as_ref() {
            if self.entities.contains_key(source_id) {
                return vec![source_id.clone()];
            }
        }
        self.entities
            .values()
            .filter(|entity| match kind {
                VectorBindingKind::Face | VectorBindingKind::Voice => entity.kind == "person",
                VectorBindingKind::Scene => entity.lifecycle == EntityLifecycleState::Active,
            })
            .filter(|entity| {
                if entity.lifecycle != EntityLifecycleState::Active {
                    return false;
                }
                let recent = t_ms.saturating_sub(entity.last_seen_ms) <= 1_000;
                let same_cell = cell_key
                    .map(|key| entity.location_cells.contains(&key))
                    .unwrap_or(false);
                let prior_support = match kind {
                    VectorBindingKind::Face => !entity.modality_support.face_vector_ids.is_empty(),
                    VectorBindingKind::Voice => {
                        !entity.modality_support.voice_vector_ids.is_empty()
                    }
                    VectorBindingKind::Scene => {
                        !entity.modality_support.scene_vector_ids.is_empty()
                    }
                };
                let explicit_label = !entity.modality_support.text_labels.is_empty();
                match kind {
                    VectorBindingKind::Face | VectorBindingKind::Voice => {
                        recent || same_cell || prior_support || explicit_label
                    }
                    VectorBindingKind::Scene => same_cell || prior_support,
                }
            })
            .map(|entity| entity.id.clone())
            .collect()
    }

    fn record_binding_candidate(&mut self, candidate: BindingCandidate) {
        self.binding_candidates.push(candidate);
        const MAX_BINDING_CANDIDATES: usize = 128;
        if self.binding_candidates.len() > MAX_BINDING_CANDIDATES {
            let excess = self.binding_candidates.len() - MAX_BINDING_CANDIDATES;
            self.binding_candidates.drain(0..excess);
        }
    }

    fn upsert_new_entity_hypothesis(
        &mut self,
        artifact: &VectorArtifact,
        kind: VectorBindingKind,
        t_ms: u64,
    ) {
        let family_id = tracking_family_id(kind, &artifact.point_id);
        self.upsert_unknown_competitor_hypothesis(artifact, kind, &family_id, &[], t_ms);
    }

    fn upsert_unknown_competitor_hypothesis(
        &mut self,
        artifact: &VectorArtifact,
        kind: VectorBindingKind,
        family_id: &str,
        competing_candidate_ids: &[String],
        t_ms: u64,
    ) {
        let mut evidence = vec![BindingEvidence {
            kind: BindingEvidenceKind::VectorSimilarity,
            score: if competing_candidate_ids.is_empty() {
                0.45
            } else {
                0.22
            },
            reason: match kind {
                VectorBindingKind::Face => "face may belong to a new unknown person".to_string(),
                VectorBindingKind::Voice => "voice may belong to a new unknown speaker".to_string(),
                VectorBindingKind::Scene => {
                    "scene may describe a new place or object context".to_string()
                }
            },
        }];
        if competing_candidate_ids.len() > 1 {
            evidence.push(BindingEvidence {
                kind: BindingEvidenceKind::SimultaneousConflict,
                score: 0.5,
                reason: "known-entity competitors are still unresolved".to_string(),
            });
        }
        let candidate_id = format!(
            "candidate:{}:{}:new",
            tracking_kind_slug(&tracking_kind_from_vector(kind)),
            stable_slug(&artifact.point_id)
        );
        self.upsert_tracking_hypothesis(
            tracking_kind_from_vector(kind),
            family_id.to_string(),
            None,
            artifact.point_id.clone(),
            candidate_id,
            evidence,
            t_ms,
        );
    }

    fn upsert_tracking_hypothesis(
        &mut self,
        kind: TrackingHypothesisKind,
        family_id: String,
        target_id: Option<String>,
        observation_id: String,
        candidate_id: String,
        evidence: Vec<BindingEvidence>,
        t_ms: u64,
    ) {
        let target_slug = target_id
            .as_deref()
            .map(stable_slug)
            .unwrap_or_else(|| "new-entity".to_string());
        let id = format!(
            "hypothesis:{}:{}:{}",
            tracking_kind_slug(&kind),
            stable_slug(&family_id),
            target_slug
        );
        if let Some(existing) = self.tracking_hypotheses.get_mut(&id) {
            if !existing.observation_ids.contains(&observation_id) {
                existing.observation_ids.push(observation_id);
            }
            if !existing.binding_candidate_ids.contains(&candidate_id) {
                existing.binding_candidate_ids.push(candidate_id);
            }
            existing.add_evidence(evidence, t_ms);
        } else {
            self.tracking_hypotheses.insert(
                id,
                TrackingHypothesis::new(
                    kind,
                    family_id,
                    target_id,
                    observation_id,
                    candidate_id,
                    evidence,
                    t_ms,
                ),
            );
        }
    }

    fn evaluate_hypothesis_family(
        &mut self,
        family_id: &str,
        artifact: &VectorArtifact,
        kind: VectorBindingKind,
        t_ms: u64,
    ) {
        let mut family = self
            .tracking_hypotheses
            .values()
            .filter(|hypothesis| {
                hypothesis.family_id == family_id
                    && !matches!(
                        hypothesis.state,
                        HypothesisState::Rejected | HypothesisState::Expired
                    )
            })
            .cloned()
            .collect::<Vec<_>>();
        if family.is_empty() {
            return;
        }
        family.sort_by(|left, right| {
            right
                .confidence
                .partial_cmp(&left.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let winner = family[0].clone();
        let runner_up_confidence = family.get(1).map(|hypothesis| hypothesis.confidence);
        let near_equal_competitor = runner_up_confidence
            .map(|confidence| winner.confidence - confidence < HYPOTHESIS_REVIEW_MARGIN)
            .unwrap_or(false);
        let promotable = self.hypothesis_passes_promotion_gates(&winner, near_equal_competitor);

        if promotable {
            self.promote_tracking_hypothesis(&winner, artifact, kind, t_ms);
            for hypothesis in family {
                if let Some(stored) = self.tracking_hypotheses.get_mut(&hypothesis.id) {
                    stored.state = if hypothesis.id == winner.id {
                        HypothesisState::Promoted
                    } else {
                        HypothesisState::Rejected
                    };
                    stored.last_updated_ms = t_ms;
                }
            }
            return;
        }

        for (index, hypothesis) in family.iter().enumerate() {
            if let Some(stored) = self.tracking_hypotheses.get_mut(&hypothesis.id) {
                if has_hard_contradiction(&stored.evidence) {
                    stored.state = HypothesisState::Rejected;
                } else if near_equal_competitor || !stored.contradictions.is_empty() {
                    stored.state = HypothesisState::NeedsReview;
                } else if index == 0 {
                    stored.state = HypothesisState::Winning;
                } else {
                    stored.state = HypothesisState::Losing;
                }
            }
        }
    }

    fn hypothesis_passes_promotion_gates(
        &self,
        hypothesis: &TrackingHypothesis,
        near_equal_competitor: bool,
    ) -> bool {
        if hypothesis.target_id.is_none() || near_equal_competitor {
            return false;
        }
        if hypothesis.confidence < HYPOTHESIS_PROMOTION_THRESHOLD {
            return false;
        }
        if has_hard_contradiction(&hypothesis.evidence) {
            return false;
        }
        let independent_evidence_types = hypothesis
            .evidence
            .iter()
            .filter(|evidence| {
                !matches!(
                    evidence.kind,
                    BindingEvidenceKind::Contradiction
                        | BindingEvidenceKind::SimultaneousConflict
                        | BindingEvidenceKind::VectorSimilarity
                        | BindingEvidenceKind::LlmSuggested
                )
            })
            .map(|evidence| binding_evidence_kind_rank(&evidence.kind))
            .collect::<BTreeSet<_>>()
            .len();
        let human_confirmed = hypothesis
            .evidence
            .iter()
            .any(|evidence| evidence.kind == BindingEvidenceKind::HumanConfirmed);
        human_confirmed || (hypothesis.evidence.len() >= 3 && independent_evidence_types >= 2)
    }

    fn promote_tracking_hypothesis(
        &mut self,
        hypothesis: &TrackingHypothesis,
        artifact: &VectorArtifact,
        kind: VectorBindingKind,
        t_ms: u64,
    ) {
        let Some(entity_id) = hypothesis.target_id.as_ref() else {
            return;
        };
        let Some(entity) = self.entities.get_mut(entity_id) else {
            return;
        };
        let Some(object_cluster_id) = entity.primary_object_cluster_id() else {
            return;
        };
        let actual_cluster_id = match kind {
            VectorBindingKind::Face => entity.add_face_vector(&artifact.point_id),
            VectorBindingKind::Voice => entity.add_voice_vector(&artifact.point_id),
            VectorBindingKind::Scene => entity.add_scene_vector(&artifact.point_id),
        };
        entity.upsert_binding_edge(
            object_cluster_id,
            actual_cluster_id,
            match kind {
                VectorBindingKind::Face | VectorBindingKind::Voice => {
                    BindingRelation::LikelySameEntity
                }
                VectorBindingKind::Scene => BindingRelation::ProjectsTo,
            },
            hypothesis.confidence,
            t_ms,
        );
    }

    pub fn confirm_tracking_hypothesis(&mut self, hypothesis_id: &str, t_ms: u64) -> bool {
        let Some(hypothesis) = self.tracking_hypotheses.get_mut(hypothesis_id) else {
            return false;
        };
        hypothesis.add_evidence(
            vec![BindingEvidence {
                kind: BindingEvidenceKind::HumanConfirmed,
                score: 1.0,
                reason: "human confirmed this hypothesis".to_string(),
            }],
            t_ms,
        );
        true
    }

    pub fn observe_frame(&mut self, frame: &ExperienceFrame, cell_key: Option<PlaceCellKey>) {
        self.observe_now(&frame.now, cell_key);
        if self.entities.is_empty() {
            return;
        }
        let active_ids = self
            .entities
            .values()
            .filter(|entity| entity.lifecycle == EntityLifecycleState::Active)
            .map(|entity| entity.id.clone())
            .collect::<Vec<_>>();
        for entity_id in active_ids {
            if let Some(entity) = self.entities.get_mut(&entity_id) {
                for experience in &frame.experiences {
                    let point = entity.push_observation_point(
                        Modality::Memory,
                        format!("experience:{}", experience.id),
                        experience.salience,
                        frame.t_ms,
                    );
                    let cluster = entity.upsert_cluster(
                        Modality::Memory,
                        format!("experience:{}", experience.id),
                        point,
                        experience.salience,
                    );
                    entity.bind_with_object_cluster(
                        cluster,
                        BindingRelation::PredictsSameFutureEvents,
                        experience.salience,
                        frame.t_ms,
                    );
                }
                for impression in &frame.impressions {
                    entity.add_text_label(
                        impression.text.clone(),
                        impression.confidence,
                        frame.t_ms,
                    );
                }
            }
        }
    }

    /// Decay confidence of all entities.  Entities whose confidence falls
    /// below threshold transition to `Occluded` or `Vanished`.
    fn decay(&mut self, ticks: u64, now_ms: u64) {
        let factor = ENTITY_CONFIDENCE_DECAY_PER_TICK.powi(ticks as i32);
        let hypothesis_factor = HYPOTHESIS_CONFIDENCE_DECAY_PER_TICK.powi(ticks as i32);
        for entity in self.entities.values_mut() {
            if entity.lifecycle == EntityLifecycleState::Vanished {
                continue;
            }
            entity.confidence = (entity.confidence * factor).clamp(0.0, 1.0);
            entity.lifecycle = if entity.confidence < ENTITY_VANISH_THRESHOLD {
                EntityLifecycleState::Vanished
            } else if entity.confidence < ENTITY_OCCLUDE_THRESHOLD {
                EntityLifecycleState::Occluded
            } else {
                EntityLifecycleState::Active
            };
            if entity.lifecycle == EntityLifecycleState::Vanished {
                entity.constellation.state = EntityConstellationState::Vanished;
            }
            entity.decay_bindings((1.0 - factor).clamp(0.0, 1.0));
        }
        for hypothesis in self.tracking_hypotheses.values_mut() {
            if matches!(
                hypothesis.state,
                HypothesisState::Promoted | HypothesisState::Rejected | HypothesisState::Expired
            ) {
                continue;
            }
            hypothesis.confidence = (hypothesis.confidence * hypothesis_factor).clamp(0.0, 1.0);
            let stale_ms = now_ms.saturating_sub(hypothesis.last_updated_ms);
            if hypothesis.confidence < 0.25 && stale_ms >= HYPOTHESIS_STALE_MS {
                hypothesis.state = HypothesisState::Expired;
            } else if hypothesis.state == HypothesisState::NeedsReview
                && stale_ms >= HYPOTHESIS_REVIEW_STALE_MS
            {
                hypothesis.state = HypothesisState::Expired;
            }
        }
    }

    pub fn merge_entities(&mut self, primary_id: &str, secondary_id: &str) -> bool {
        if primary_id == secondary_id {
            return false;
        }
        let Some(mut secondary) = self.entities.remove(secondary_id) else {
            return false;
        };
        let Some(primary) = self.entities.get_mut(primary_id) else {
            self.entities.insert(secondary_id.to_string(), secondary);
            return false;
        };
        primary.observation_count = primary
            .observation_count
            .saturating_add(secondary.observation_count);
        primary.confidence = primary.confidence.max(secondary.confidence);
        for label in secondary.labels.drain(..) {
            if !primary.labels.contains(&label) {
                primary.labels.push(label);
            }
        }
        primary
            .constellation
            .merged_entity_ids
            .push(secondary_id.to_string());
        primary.constellation.state = EntityConstellationState::Merged;
        true
    }

    pub fn split_entity(&mut self, entity_id: &str, suffix: &str) -> Option<String> {
        let mut child = self.entities.get(entity_id)?.clone();
        let child_id = format!("{entity_id}:split:{}", stable_slug(suffix));
        child.id = child_id.clone();
        child.confidence = (child.confidence * 0.6).clamp(0.0, 1.0);
        child.constellation.state = EntityConstellationState::Split;
        if let Some(parent) = self.entities.get_mut(entity_id) {
            parent.constellation.split_entity_ids.push(child_id.clone());
            parent.constellation.state = EntityConstellationState::Split;
        }
        self.entities.insert(child_id.clone(), child);
        Some(child_id)
    }

    /// Build a summary report for dashboard/API consumption.
    pub fn report(&self) -> EntityMemoryReport {
        let total_entities = self.entities.len();
        let active_entities = self
            .entities
            .values()
            .filter(|e| e.lifecycle == EntityLifecycleState::Active)
            .count();
        let occluded_entities = self
            .entities
            .values()
            .filter(|e| e.lifecycle == EntityLifecycleState::Occluded)
            .count();
        let vanished_entities = self
            .entities
            .values()
            .filter(|e| e.lifecycle == EntityLifecycleState::Vanished)
            .count();

        let mut sorted: Vec<&EntityHypothesis> = self.entities.values().collect();
        sorted.sort_by(|a, b| {
            // Active before occluded before vanished, then by confidence descending.
            let state_order = |e: &EntityHypothesis| match e.lifecycle {
                EntityLifecycleState::Active => 0u8,
                EntityLifecycleState::Occluded => 1,
                EntityLifecycleState::Vanished => 2,
            };
            state_order(a).cmp(&state_order(b)).then_with(|| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        sorted.truncate(20);
        let top_entities = sorted
            .iter()
            .map(|e| EntityHypothesisSummary::from(*e))
            .collect();

        let all_candidates = self
            .binding_candidates
            .iter()
            .cloned()
            .chain(
                self.entities
                    .values()
                    .flat_map(|entity| entity.constellation.binding_candidates.iter().cloned()),
            )
            .collect::<Vec<_>>();
        let accepted_binding_candidates = all_candidates
            .iter()
            .filter(|candidate| candidate.decision == BindingDecision::Accept)
            .cloned()
            .collect();
        let ambiguous_binding_candidates = all_candidates
            .iter()
            .filter(|candidate| {
                matches!(
                    candidate.decision,
                    BindingDecision::HoldAmbiguous
                        | BindingDecision::AskHuman
                        | BindingDecision::CollectMoreEvidence
                )
            })
            .cloned()
            .collect();
        let rejected_binding_candidates = all_candidates
            .iter()
            .filter(|candidate| candidate.decision == BindingDecision::Reject)
            .cloned()
            .collect();
        let active_tracking_hypotheses = self
            .tracking_hypotheses
            .values()
            .filter(|hypothesis| {
                matches!(
                    hypothesis.state,
                    HypothesisState::Active | HypothesisState::Winning | HypothesisState::Losing
                )
            })
            .cloned()
            .collect();
        let review_tracking_hypotheses = self
            .tracking_hypotheses
            .values()
            .filter(|hypothesis| hypothesis.state == HypothesisState::NeedsReview)
            .cloned()
            .collect();
        let promoted_tracking_hypotheses = self
            .tracking_hypotheses
            .values()
            .filter(|hypothesis| hypothesis.state == HypothesisState::Promoted)
            .cloned()
            .collect();
        let expired_tracking_hypotheses = self
            .tracking_hypotheses
            .values()
            .filter(|hypothesis| hypothesis.state == HypothesisState::Expired)
            .cloned()
            .collect();

        EntityMemoryReport {
            total_entities,
            active_entities,
            occluded_entities,
            vanished_entities,
            active_tracking_hypotheses,
            review_tracking_hypotheses,
            promoted_tracking_hypotheses,
            expired_tracking_hypotheses,
            accepted_binding_candidates,
            ambiguous_binding_candidates,
            rejected_binding_candidates,
            top_entities,
        }
    }
}

fn base_cross_modal_evidence(
    context: &BindingContext,
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
) -> Vec<BindingEvidence> {
    let mut evidence = Vec::new();
    if temporally_compatible(context, left, right) {
        let delta_ms = cluster_time_delta_ms(left, right);
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::TemporalOverlap,
            score: (1.0 - delta_ms as f32 / context.time_window_ms.max(1) as f32).clamp(0.0, 1.0),
            reason: format!("{} and {} occurred within {delta_ms} ms", left.id, right.id),
        });
    }
    if source_frame_matches(context, left, right) {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::ProjectionAgreement,
            score: 0.55,
            reason: "clusters share a source frame context".to_string(),
        });
    }
    if let Some(distance) = pose_distance_m(left, right) {
        if distance <= 0.75 {
            evidence.push(BindingEvidence {
                kind: BindingEvidenceKind::SpatialOverlap,
                score: (1.0 - distance / 0.75).clamp(0.0, 1.0),
                reason: format!("cluster poses are within {distance:.2} m"),
            });
        }
    }
    evidence
}

fn candidate_from_evidence(
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
    relation: BindingRelation,
    evidence: Vec<BindingEvidence>,
    fallback_reason: &str,
) -> BindingCandidate {
    let has_human_confirmation = evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::HumanConfirmed);
    let has_hard_contradiction = evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::Contradiction);
    let has_conflict = evidence.iter().any(|item| {
        matches!(
            item.kind,
            BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
        )
    });
    let independent_positive_kinds = evidence
        .iter()
        .filter(|item| {
            !matches!(
                item.kind,
                BindingEvidenceKind::Contradiction
                    | BindingEvidenceKind::SimultaneousConflict
                    | BindingEvidenceKind::VectorSimilarity
                    | BindingEvidenceKind::LlmSuggested
            )
        })
        .map(|item| binding_evidence_kind_rank(&item.kind))
        .collect::<BTreeSet<_>>()
        .len();
    let mean_score = if evidence.is_empty() {
        0.0
    } else {
        evidence
            .iter()
            .map(|item| item.score.clamp(0.0, 1.0))
            .sum::<f32>()
            / evidence.len() as f32
    };
    let mut confidence = if has_human_confirmation {
        mean_score.max(0.9)
    } else {
        (mean_score * (independent_positive_kinds as f32 / 3.0).clamp(0.25, 1.0)).clamp(0.0, 1.0)
    };
    if has_conflict {
        confidence *= 0.35;
    }

    let (decision, reason) = if has_hard_contradiction {
        (
            BindingDecision::Reject,
            "candidate contains contradictory cross-modal evidence".to_string(),
        )
    } else if has_conflict {
        (
            BindingDecision::HoldAmbiguous,
            "candidate is plausible but has competing cross-modal evidence".to_string(),
        )
    } else if has_human_confirmation {
        (
            BindingDecision::Accept,
            "candidate has trusted human/source confirmation".to_string(),
        )
    } else if independent_positive_kinds >= 2 {
        (
            BindingDecision::Accept,
            "candidate has at least two independent cross-modal evidence types".to_string(),
        )
    } else if evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::LlmSuggested)
        && independent_positive_kinds == 0
    {
        (
            BindingDecision::CollectMoreEvidence,
            "LLM suggestion alone is not enough to bind clusters".to_string(),
        )
    } else if evidence.is_empty() {
        (
            BindingDecision::CollectMoreEvidence,
            fallback_reason.to_string(),
        )
    } else {
        (
            BindingDecision::CollectMoreEvidence,
            "candidate needs more independent evidence before admission".to_string(),
        )
    };

    BindingCandidate {
        left_cluster_id: left.id.clone(),
        right_cluster_id: right.id.clone(),
        relation,
        evidence,
        confidence: confidence.clamp(0.0, 1.0),
        decision,
        reason,
    }
}

fn proposal_candidate_from_evidence(
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
    relation: BindingRelation,
    evidence: Vec<BindingEvidence>,
    fallback_reason: &str,
) -> BindingCandidate {
    let mut candidate = candidate_from_evidence(left, right, relation, evidence, fallback_reason);
    if candidate.decision == BindingDecision::Accept {
        candidate.decision = BindingDecision::CollectMoreEvidence;
        candidate.reason =
            "candidate is proposal-only; conservative binding admission must accept it".to_string();
    }
    candidate
}

fn temporally_compatible(
    context: &BindingContext,
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
) -> bool {
    cluster_time_delta_ms(left, right) <= context.time_window_ms
}

fn cluster_time_delta_ms(left: &DiscoveredCluster, right: &DiscoveredCluster) -> u64 {
    if left.last_seen_ms < right.first_seen_ms {
        right.first_seen_ms.saturating_sub(left.last_seen_ms)
    } else if right.last_seen_ms < left.first_seen_ms {
        left.first_seen_ms.saturating_sub(right.last_seen_ms)
    } else {
        0
    }
}

fn source_frame_matches(
    context: &BindingContext,
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
) -> bool {
    if left.source_frame_id.is_some()
        && right.source_frame_id.is_some()
        && left.source_frame_id == right.source_frame_id
    {
        return true;
    }
    context.source_frame_id.as_ref().is_some_and(|frame| {
        left.source_frame_id.as_ref() == Some(frame)
            || right.source_frame_id.as_ref() == Some(frame)
    })
}

fn add_recent_cooccurrence(
    context: &BindingContext,
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
    evidence: &mut Vec<BindingEvidence>,
) {
    if context.recent_clusters.contains(&left.id) && context.recent_clusters.contains(&right.id) {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::RepeatedCooccurrence,
            score: 0.7,
            reason: "both clusters appeared in recent binding context".to_string(),
        });
    }
    if !left.feature_ids.is_empty()
        && !right.feature_ids.is_empty()
        && left
            .feature_ids
            .iter()
            .any(|id| context.recent_features.contains(id))
        && right
            .feature_ids
            .iter()
            .any(|id| context.recent_features.contains(id))
    {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::RepeatedCooccurrence,
            score: 0.7,
            reason: "both clusters reference recently observed features".to_string(),
        });
    }
}

fn add_repetition_evidence(
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
    evidence: &mut Vec<BindingEvidence>,
) {
    let repeats =
        metadata_u64(left, "cooccurrence_count").max(metadata_u64(right, "cooccurrence_count"));
    if repeats >= 2 {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::RepeatedCooccurrence,
            score: (repeats as f32 / 5.0).clamp(0.0, 1.0),
            reason: format!("clusters have repeated together in {repeats} observations"),
        });
    }
}

fn add_label_support(
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
    evidence: &mut Vec<BindingEvidence>,
) {
    let left_label = metadata_string(left, "label");
    let right_label = metadata_string(right, "label");
    if left_label.is_some() && left_label == right_label {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::HumanConfirmed,
            score: 0.85,
            reason: "clusters share a supporting label".to_string(),
        });
    }
}

fn projection_error_px(left: &DiscoveredCluster, right: &DiscoveredCluster) -> Option<f32> {
    let left_x = metadata_f32(left, "image_x");
    let left_y = metadata_f32(left, "image_y");
    let right_x =
        metadata_f32(right, "projected_image_x").or_else(|| metadata_f32(right, "image_x"));
    let right_y =
        metadata_f32(right, "projected_image_y").or_else(|| metadata_f32(right, "image_y"));
    left_x
        .zip(left_y)
        .zip(right_x.zip(right_y))
        .map(|((lx, ly), (rx, ry))| ((lx - rx).powi(2) + (ly - ry).powi(2)).sqrt())
        .or_else(|| {
            let right_x = metadata_f32(right, "image_x");
            let right_y = metadata_f32(right, "image_y");
            let left_x =
                metadata_f32(left, "projected_image_x").or_else(|| metadata_f32(left, "image_x"));
            let left_y =
                metadata_f32(left, "projected_image_y").or_else(|| metadata_f32(left, "image_y"));
            left_x
                .zip(left_y)
                .zip(right_x.zip(right_y))
                .map(|((lx, ly), (rx, ry))| ((lx - rx).powi(2) + (ly - ry).powi(2)).sqrt())
        })
}

fn pose_distance_m(left: &DiscoveredCluster, right: &DiscoveredCluster) -> Option<f32> {
    left.estimated_pose
        .zip(right.estimated_pose)
        .map(|(left, right)| {
            ((left.x_m - right.x_m).powi(2) + (left.y_m - right.y_m).powi(2)).sqrt()
        })
}

fn lag_score(lag_ms: u64, min_ms: u64, max_ms: u64) -> f32 {
    if lag_ms < min_ms || lag_ms > max_ms {
        return 0.0;
    }
    let midpoint = (min_ms + max_ms) as f32 / 2.0;
    let half_span = (max_ms.saturating_sub(min_ms)).max(1) as f32 / 2.0;
    (1.0 - ((lag_ms as f32 - midpoint).abs() / half_span)).clamp(0.1, 1.0)
}

fn metadata_f32(cluster: &DiscoveredCluster, key: &str) -> Option<f32> {
    cluster
        .metadata
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .map(|value| value as f32)
}

fn metadata_u64(cluster: &DiscoveredCluster, key: &str) -> u64 {
    cluster
        .metadata
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default()
}

fn metadata_bool(cluster: &DiscoveredCluster, key: &str) -> bool {
    cluster
        .metadata
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn metadata_string(cluster: &DiscoveredCluster, key: &str) -> Option<String> {
    cluster
        .metadata
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

fn qualify_binding_candidate(
    entity: &EntityHypothesis,
    artifact: &VectorArtifact,
    kind: VectorBindingKind,
    left_cluster_id: String,
    right_cluster_id: String,
    t_ms: u64,
    cell_key: Option<PlaceCellKey>,
    plausible_count: usize,
    current_object_observed: bool,
) -> BindingCandidate {
    let mut evidence = Vec::new();
    evidence.push(BindingEvidence {
        kind: BindingEvidenceKind::VectorSimilarity,
        score: 0.45,
        reason: "vector artifact proposes a possible cross-modal correspondence".to_string(),
    });

    if artifact.source_id.as_deref() == Some(entity.id.as_str()) {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::HumanConfirmed,
            score: 1.0,
            reason: "vector source explicitly names this entity".to_string(),
        });
    } else if artifact
        .source_id
        .as_deref()
        .is_some_and(|source_id| source_id.starts_with("entity:"))
    {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::Contradiction,
            score: 1.0,
            reason: format!(
                "vector source names {}, not {}",
                artifact.source_id.as_deref().unwrap_or("unknown"),
                entity.id
            ),
        });
    }
    if t_ms.saturating_sub(entity.last_seen_ms) <= 1_000 {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::TemporalOverlap,
            score: 0.75,
            reason: "entity was observed in the current temporal window".to_string(),
        });
    }
    if cell_key
        .map(|key| entity.location_cells.contains(&key))
        .unwrap_or(false)
    {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::SpatialOverlap,
            score: 0.75,
            reason: "entity has a compatible current map cell".to_string(),
        });
    }
    if current_object_observed {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::ProjectionAgreement,
            score: 0.7,
            reason: "a current object observation anchors this entity".to_string(),
        });
    }
    if plausible_count == 1 {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::SingleCandidateContext,
            score: 0.65,
            reason: "only one plausible entity matched this vector context".to_string(),
        });
    } else if plausible_count > 1 {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::SimultaneousConflict,
            score: 0.8,
            reason: match kind {
                VectorBindingKind::Face => {
                    "face vector close to multiple active person entities".to_string()
                }
                VectorBindingKind::Voice => {
                    "voice observed while multiple person hypotheses are active".to_string()
                }
                VectorBindingKind::Scene => {
                    "scene vector has multiple spatially plausible entities".to_string()
                }
            },
        });
    }
    if entity.constellation.binding_edges.iter().any(|edge| {
        edge.left_cluster_id == right_cluster_id || edge.right_cluster_id == right_cluster_id
    }) {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::RepeatedCooccurrence,
            score: 0.8,
            reason: "prior binding history supports this correspondence".to_string(),
        });
    }

    let has_human_confirmation = evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::HumanConfirmed);
    let has_conflict = evidence.iter().any(|item| {
        matches!(
            item.kind,
            BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
        )
    });
    let has_hard_contradiction = evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::Contradiction);
    let independent_positive_kinds = evidence
        .iter()
        .filter(|item| {
            !matches!(
                item.kind,
                BindingEvidenceKind::Contradiction
                    | BindingEvidenceKind::SimultaneousConflict
                    | BindingEvidenceKind::VectorSimilarity
            )
        })
        .map(|item| binding_evidence_kind_rank(&item.kind))
        .collect::<BTreeSet<_>>()
        .len();
    let mean_score = if evidence.is_empty() {
        0.0
    } else {
        evidence
            .iter()
            .map(|item| item.score.clamp(0.0, 1.0))
            .sum::<f32>()
            / evidence.len() as f32
    };
    let mut confidence = if has_human_confirmation {
        mean_score.max(0.9)
    } else {
        (mean_score * (independent_positive_kinds as f32 / 3.0).clamp(0.25, 1.0)).clamp(0.0, 1.0)
    };
    if has_conflict {
        confidence *= 0.35;
    }

    let (decision, reason) = if has_hard_contradiction {
        (
            BindingDecision::Reject,
            "candidate contradicts explicit entity source evidence".to_string(),
        )
    } else if has_human_confirmation {
        (
            BindingDecision::Accept,
            "human-confirmed or explicit source binding".to_string(),
        )
    } else if has_conflict {
        (
            BindingDecision::HoldAmbiguous,
            match kind {
                VectorBindingKind::Face => "face vector close to multiple active person entities",
                VectorBindingKind::Voice => {
                    "voice observed while multiple person hypotheses active"
                }
                VectorBindingKind::Scene => {
                    "scene vector active but multiple spatially compatible entities exist"
                }
            }
            .to_string(),
        )
    } else if independent_positive_kinds >= 2 {
        (
            BindingDecision::Accept,
            "candidate has at least two independent supporting evidence types".to_string(),
        )
    } else if evidence.len() == 1 {
        (
            BindingDecision::CollectMoreEvidence,
            "single vector similarity without supporting temporal/spatial evidence".to_string(),
        )
    } else {
        (
            BindingDecision::CollectMoreEvidence,
            "projection agreement missing or evidence is not yet independent".to_string(),
        )
    };

    BindingCandidate {
        left_cluster_id,
        right_cluster_id,
        relation: match kind {
            VectorBindingKind::Face | VectorBindingKind::Voice => BindingRelation::LikelySameEntity,
            VectorBindingKind::Scene => BindingRelation::ProjectsTo,
        },
        evidence,
        confidence: confidence.clamp(0.0, 1.0),
        decision,
        reason,
    }
}

fn vector_cluster_id(kind: VectorBindingKind, point_id: &str) -> String {
    let key = match kind {
        VectorBindingKind::Face => format!("face:{point_id}"),
        VectorBindingKind::Voice => format!("voice:{point_id}"),
        VectorBindingKind::Scene => format!("scene:{point_id}"),
    };
    let modality = match kind {
        VectorBindingKind::Face => Modality::Vision,
        VectorBindingKind::Voice => Modality::Audio,
        VectorBindingKind::Scene => Modality::Depth,
    };
    format!("cluster:{}:{}", modality.as_str(), stable_slug(&key))
}

fn binding_evidence_kind_rank(kind: &BindingEvidenceKind) -> u8 {
    match kind {
        BindingEvidenceKind::TemporalOverlap => 1,
        BindingEvidenceKind::SpatialOverlap => 2,
        BindingEvidenceKind::VectorSimilarity => 3,
        BindingEvidenceKind::ProjectionAgreement => 4,
        BindingEvidenceKind::PoseAgreement => 5,
        BindingEvidenceKind::RepeatedCooccurrence => 6,
        BindingEvidenceKind::SingleCandidateContext => 7,
        BindingEvidenceKind::HumanConfirmed => 8,
        BindingEvidenceKind::LlmSuggested => 9,
        BindingEvidenceKind::Contradiction => 10,
        BindingEvidenceKind::SimultaneousConflict => 11,
    }
}

fn tracking_kind_from_vector(kind: VectorBindingKind) -> TrackingHypothesisKind {
    match kind {
        VectorBindingKind::Face => TrackingHypothesisKind::FaceIdentity,
        VectorBindingKind::Voice => TrackingHypothesisKind::VoiceIdentity,
        VectorBindingKind::Scene => TrackingHypothesisKind::PlaceMatch,
    }
}

fn tracking_kind_slug(kind: &TrackingHypothesisKind) -> &'static str {
    match kind {
        TrackingHypothesisKind::FaceIdentity => "face-identity",
        TrackingHypothesisKind::VoiceIdentity => "voice-identity",
        TrackingHypothesisKind::CrossModalBinding => "cross-modal",
        TrackingHypothesisKind::PlaceMatch => "place",
        TrackingHypothesisKind::ObjectContinuity => "object-continuity",
        TrackingHypothesisKind::Other => "other",
    }
}

fn tracking_family_id(kind: VectorBindingKind, observation_id: &str) -> String {
    format!(
        "{}:{}",
        tracking_kind_slug(&tracking_kind_from_vector(kind)),
        observation_id
    )
}

fn binding_candidate_id(candidate: &BindingCandidate) -> String {
    format!(
        "candidate:{}:{}:{}",
        stable_slug(&candidate.left_cluster_id),
        stable_slug(&candidate.right_cluster_id),
        binding_relation_label_for_id(&candidate.relation)
    )
}

fn binding_relation_label_for_id(relation: &BindingRelation) -> &'static str {
    match relation {
        BindingRelation::CooccursInTime => "time",
        BindingRelation::CooccursInEstimatedSpace => "space",
        BindingRelation::MovesTogether => "moves",
        BindingRelation::PredictsSameFutureEvents => "future",
        BindingRelation::NamedBy => "named-by",
        BindingRelation::ProjectsTo => "projects-to",
        BindingRelation::HasColorAtPose => "color-pose",
        BindingRelation::LikelySameEntity => "same-entity",
        BindingRelation::ExplainsOutcome => "outcome",
        BindingRelation::Contradicts => "contradicts",
        BindingRelation::RequiresReview => "review",
    }
}

fn has_hard_contradiction(evidence: &[BindingEvidence]) -> bool {
    evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::Contradiction)
}

fn score_hypothesis_evidence(evidence: &[BindingEvidence], repeated_observations: f32) -> f32 {
    if evidence.is_empty() {
        return 0.0;
    }
    let human_confirmed = evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::HumanConfirmed);
    let positive = evidence
        .iter()
        .filter(|item| {
            !matches!(
                item.kind,
                BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
            )
        })
        .map(|item| item.score.clamp(0.0, 1.0))
        .sum::<f32>();
    let positive_count = evidence
        .iter()
        .filter(|item| {
            !matches!(
                item.kind,
                BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
            )
        })
        .count()
        .max(1) as f32;
    let independent_positive_kinds = evidence
        .iter()
        .filter(|item| {
            !matches!(
                item.kind,
                BindingEvidenceKind::Contradiction
                    | BindingEvidenceKind::SimultaneousConflict
                    | BindingEvidenceKind::VectorSimilarity
                    | BindingEvidenceKind::LlmSuggested
            )
        })
        .map(|item| binding_evidence_kind_rank(&item.kind))
        .collect::<BTreeSet<_>>()
        .len() as f32;
    let contradiction_count = evidence
        .iter()
        .filter(|item| {
            matches!(
                item.kind,
                BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
            )
        })
        .count() as f32;
    let repetition_bonus = ((repeated_observations - 1.0).max(0.0) * 0.08).min(0.18);
    let independence_bonus = (independent_positive_kinds * 0.08).min(0.24);
    let mut score = positive / positive_count + repetition_bonus + independence_bonus;
    if human_confirmed {
        score = score.max(0.92);
    }
    score -= contradiction_count * 0.18;
    score.clamp(0.0, 1.0)
}

fn cluster_ids_from_observation(
    observation: &ConstellationObservation,
    accepted_bindings: &[&BindingCandidate],
) -> Vec<String> {
    let mut ids = accepted_bindings
        .iter()
        .flat_map(|candidate| {
            [
                candidate.left_cluster_id.clone(),
                candidate.right_cluster_id.clone(),
            ]
        })
        .chain(
            observation
                .clusters
                .iter()
                .map(|cluster| cluster.id.clone()),
        )
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    ids.sort();
    ids
}

fn merge_constellation_observation(
    constellation: &mut Constellation,
    observation: &ConstellationObservation,
    member_cluster_ids: &[String],
    member_binding_ids: &[String],
) {
    merge_unique(&mut constellation.member_cluster_ids, member_cluster_ids);
    merge_unique(&mut constellation.member_binding_ids, member_binding_ids);
    let feature_ids = observation
        .clusters
        .iter()
        .flat_map(|cluster| cluster.feature_ids.iter().copied())
        .collect::<Vec<_>>();
    merge_unique(&mut constellation.supporting_feature_ids, &feature_ids);
    merge_unique(
        &mut constellation.supporting_entity_ids,
        &observation.active_entity_ids,
    );
    merge_unique(
        &mut constellation.supporting_place_cells,
        &observation.place_cells,
    );
    merge_unique(&mut constellation.notes, &observation.llm_notes);
    constellation.last_seen_ms = observation.t_ms;
    constellation.evidence_count = constellation.evidence_count.saturating_add(1);
    constellation.prediction_value = (constellation.prediction_value * 0.75
        + observation.prediction_value * 0.25)
        .clamp(0.0, 1.0);
    if constellation.kind_hint.is_none() {
        constellation.kind_hint =
            infer_constellation_kind(&observation.clusters).map(|kind| kind.as_str().to_string());
    }
}

fn refresh_constellation_scores(
    constellation: &mut Constellation,
    observation: &ConstellationObservation,
    config: &ConstellationEngineConfig,
) {
    let accepted_bindings = observation
        .accepted_bindings
        .iter()
        .filter(|candidate| candidate.decision == BindingDecision::Accept)
        .collect::<Vec<_>>();
    let positive_binding_score = if accepted_bindings.is_empty() {
        0.0
    } else {
        accepted_bindings
            .iter()
            .map(|candidate| candidate.confidence.clamp(0.0, 1.0))
            .sum::<f32>()
            / accepted_bindings.len() as f32
    };
    let recurrence_score = (constellation.evidence_count as f32
        / config.min_evidence_for_stable.max(1) as f32)
        .clamp(0.0, 1.0);
    let cluster_score = (constellation.member_cluster_ids.len() as f32
        / config.min_clusters_for_stable.max(1) as f32)
        .clamp(0.0, 1.0);
    let binding_score = (constellation.member_binding_ids.len() as f32
        / config.min_bindings_for_stable.max(1) as f32)
        .clamp(0.0, 1.0);
    let contradiction_count = observation
        .accepted_bindings
        .iter()
        .filter(|candidate| binding_has_conflict(candidate))
        .count();
    let contradiction_penalty = (contradiction_count as f32 * 0.25).min(0.65);

    constellation.stability =
        (recurrence_score * 0.5 + binding_score * 0.3 + cluster_score * 0.2).clamp(0.0, 1.0);
    constellation.confidence = (positive_binding_score * 0.45
        + constellation.stability * 0.35
        + constellation.prediction_value * 0.2
        - contradiction_penalty)
        .clamp(0.0, 1.0);

    if evidence_suggests_split(observation) {
        constellation.state = ConstellationState::SplitNeeded;
        return;
    }
    if contradiction_count > 0 {
        constellation.state = ConstellationState::Ambiguous;
        return;
    }
    let promotable = constellation.member_cluster_ids.len() >= config.min_clusters_for_stable
        && constellation.member_binding_ids.len() >= config.min_bindings_for_stable
        && constellation.evidence_count >= config.min_evidence_for_stable
        && constellation.confidence >= config.promotion_confidence_threshold
        && constellation.prediction_value >= config.min_prediction_value_for_stable;
    constellation.state = if promotable {
        ConstellationState::Stable
    } else {
        ConstellationState::Candidate
    };
}

fn binding_has_conflict(candidate: &BindingCandidate) -> bool {
    candidate.evidence.iter().any(|evidence| {
        matches!(
            evidence.kind,
            BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
        )
    })
}

fn evidence_suggests_split(observation: &ConstellationObservation) -> bool {
    observation.accepted_bindings.iter().any(|candidate| {
        candidate
            .evidence
            .iter()
            .any(|evidence| evidence.kind == BindingEvidenceKind::SimultaneousConflict)
    }) || observation.llm_notes.iter().any(|note| {
        let note = note.to_ascii_lowercase();
        note.contains("split")
            || note.contains("fused")
            || note.contains("fusion")
            || note.contains("two patterns")
    })
}

fn infer_constellation_kind(clusters: &[DiscoveredCluster]) -> Option<ConstellationKind> {
    let kinds = clusters
        .iter()
        .map(|cluster| cluster.kind.clone())
        .collect::<BTreeSet<_>>();
    if kinds.contains(&DiscoveredClusterKind::Face) || kinds.contains(&DiscoveredClusterKind::Voice)
    {
        Some(ConstellationKind::Person)
    } else if kinds.contains(&DiscoveredClusterKind::Action)
        || kinds.contains(&DiscoveredClusterKind::Outcome)
        || kinds.contains(&DiscoveredClusterKind::BodyState)
    {
        Some(ConstellationKind::ActionOutcome)
    } else if kinds.contains(&DiscoveredClusterKind::Place) {
        Some(ConstellationKind::Place)
    } else if kinds.contains(&DiscoveredClusterKind::Object)
        || kinds.contains(&DiscoveredClusterKind::Geometry)
        || kinds.contains(&DiscoveredClusterKind::RgbImage)
    {
        Some(ConstellationKind::Object)
    } else {
        None
    }
}

fn overlap_score(matched: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        (matched as f32 / total as f32).clamp(0.0, 1.0)
    }
}

fn stale_penalty(age_ms: u64, stale_after_ms: u64) -> f32 {
    if stale_after_ms == 0 || age_ms <= stale_after_ms {
        0.0
    } else {
        ((age_ms - stale_after_ms) as f32 / (stale_after_ms * 4) as f32).clamp(0.0, 0.6)
    }
}

fn intersection_count<T>(left: &[T], right: &[T]) -> usize
where
    T: Ord + Clone,
{
    let left = left.iter().cloned().collect::<BTreeSet<_>>();
    let right = right.iter().cloned().collect::<BTreeSet<_>>();
    left.intersection(&right).count()
}

fn merge_unique<T>(target: &mut Vec<T>, incoming: &[T])
where
    T: Ord + Clone,
{
    let mut seen = target.iter().cloned().collect::<BTreeSet<_>>();
    for item in incoming {
        if seen.insert(item.clone()) {
            target.push(item.clone());
        }
    }
    target.sort();
}

fn association_edge_id(from_id: &str, to_id: &str, relation: &AssociationRelation) -> String {
    format!(
        "association:{}:{}:{}",
        association_relation_slug(relation),
        stable_slug(from_id),
        stable_slug(to_id)
    )
}

fn association_relation_slug(relation: &AssociationRelation) -> &'static str {
    match relation {
        AssociationRelation::CoOccursWith => "co-occurs-with",
        AssociationRelation::Predicts => "predicts",
        AssociationRelation::Follows => "follows",
        AssociationRelation::Suppresses => "suppresses",
        AssociationRelation::Contradicts => "contradicts",
        AssociationRelation::Explains => "explains",
        AssociationRelation::Enables => "enables",
        AssociationRelation::Prevents => "prevents",
        AssociationRelation::PartOf => "part-of",
    }
}

fn dedupe_association_items(items: Vec<AssociationItem>) -> Vec<AssociationItem> {
    let mut by_id = BTreeMap::<String, AssociationItem>::new();
    for item in items {
        by_id
            .entry(item.id.clone())
            .and_modify(|existing| {
                existing.confidence = existing.confidence.max(item.confidence);
            })
            .or_insert(item);
    }
    by_id.into_values().collect()
}

fn canonical_association_pair<'a>(
    left: &'a AssociationItem,
    right: &'a AssociationItem,
) -> (&'a str, &'a str) {
    if left.id <= right.id {
        (&left.id, &right.id)
    } else {
        (&right.id, &left.id)
    }
}

fn sequence_relation(
    to: &AssociationItem,
    lag_ms: u64,
    config: &AssociationLearningConfig,
) -> AssociationRelation {
    if matches!(
        to.kind,
        AssociationItemKind::Outcome
            | AssociationItemKind::Prediction
            | AssociationItemKind::Surprise
            | AssociationItemKind::BodyState
    ) && lag_ms <= config.long_sequence_window_ms
    {
        AssociationRelation::Predicts
    } else if lag_ms <= config.short_sequence_window_ms {
        AssociationRelation::Follows
    } else {
        AssociationRelation::Follows
    }
}

fn lag_score_for_association(lag_ms: u64) -> f32 {
    match lag_ms {
        0..=500 => 1.0,
        501..=2_000 => 0.8,
        2_001..=10_000 => 0.55,
        _ => 0.2,
    }
}

fn approximate_mutual_information(p_b: f32, p_b_given_a: f32) -> f32 {
    let p_b = p_b.clamp(0.001, 0.999);
    let p_b_given_a = p_b_given_a.clamp(0.001, 0.999);
    if p_b_given_a <= p_b {
        return 0.0;
    }
    (p_b_given_a * (p_b_given_a / p_b).ln()).clamp(0.0, 1.0)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub frame_id: uuid::Uuid,
    pub t_ms: u64,
    pub summary: String,
    #[serde(default)]
    pub graph_entities: Vec<GraphEntity>,
    #[serde(default)]
    pub graph_relationships: Vec<GraphEdge>,
    #[serde(default)]
    pub scene_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub face_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub object_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub voice_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub sensation_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub experience_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub vector_payloads: BTreeMap<String, serde_json::Value>,
    pub battery: f32,
    pub active_goal: Option<Goal>,
    pub chosen_action: Option<ActionPrimitive>,
    pub warning: Option<String>,
    pub experience: Option<Experience>,
    #[serde(default)]
    pub temporal_context: TemporalContext,
    #[serde(default)]
    pub social_world: SocialWorldSnapshot,
    #[serde(default)]
    pub epistemic_state: EpistemicSnapshot,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QdrantConfig {
    pub url: String,
}

impl QdrantConfig {
    pub fn from_env() -> Option<Self> {
        std::env::var("PETE_QDRANT_URL")
            .ok()
            .filter(|url| !url.trim().is_empty())
            .map(|url| Self { url })
    }
}

#[derive(Clone)]
pub struct QdrantVectorStore {
    client: reqwest::Client,
    config: QdrantConfig,
}

impl QdrantVectorStore {
    pub fn new(config: QdrantConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
        }
    }

    pub fn from_env() -> Option<Self> {
        QdrantConfig::from_env().map(Self::new)
    }

    async fn ensure_collection(&self, collection: &str, vector_size: usize) -> Result<()> {
        if vector_size == 0 {
            return Ok(());
        }
        let url = format!(
            "{}/collections/{}",
            self.config.url.trim_end_matches('/'),
            collection
        );
        let response = self
            .client
            .put(url)
            .json(&json!({
                "vectors": {
                    "size": vector_size,
                    "distance": "Cosine"
                }
            }))
            .send()
            .await
            .context("creating qdrant collection")?;
        if response.status().is_success() || response.status() == StatusCode::CONFLICT {
            return Ok(());
        }
        Err(anyhow!(
            "qdrant collection create failed for {collection}: HTTP {}",
            response.status()
        ))
    }
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn upsert_vectors(&self, record: &MemoryRecord) -> Result<()> {
        let mut by_collection: BTreeMap<&str, Vec<&VectorArtifact>> = BTreeMap::new();
        for artifact in record_all_vectors(record) {
            by_collection
                .entry(artifact.collection.as_str())
                .or_default()
                .push(artifact);
        }

        for (collection, artifacts) in by_collection {
            let Some(first) = artifacts.first() else {
                continue;
            };
            self.ensure_collection(collection, first.vector.len())
                .await?;
            let points = artifacts
                .into_iter()
                .filter(|artifact| !artifact.vector.is_empty())
                .map(|artifact| {
                    let mut payload = json!({
                        "collection": artifact.collection,
                        "point_id": artifact.point_id,
                        "frame_id": record.frame_id.to_string(),
                        "source_frame_id": artifact.source_frame_id,
                        "source_id": artifact.source_id,
                        "model": artifact.model,
                        "dim": artifact.vector.len(),
                        "occurred_at_ms": artifact.occurred_at_ms.or(Some(record.t_ms)),
                        "summary": record.summary,
                        "neo4j_node_id": vector_node_id(artifact),
                    });
                    if let Some(extra) = record.vector_payloads.get(&vector_payload_key(artifact)) {
                        merge_json_object(&mut payload, extra);
                    }
                    json!({
                        "id": stable_qdrant_point_id(&artifact.collection, &artifact.point_id),
                        "vector": artifact.vector,
                        "payload": payload
                    })
                })
                .collect::<Vec<_>>();
            if points.is_empty() {
                continue;
            }
            let url = format!(
                "{}/collections/{}/points?wait=true",
                self.config.url.trim_end_matches('/'),
                collection
            );
            let response = self
                .client
                .put(url)
                .json(&json!({ "points": points }))
                .send()
                .await
                .with_context(|| format!("upserting qdrant points into {collection}"))?;
            if !response.status().is_success() {
                return Err(anyhow!(
                    "qdrant upsert failed for {collection}: HTTP {}",
                    response.status()
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Neo4jConfig {
    pub http_url: String,
    pub user: String,
    pub password: String,
    pub database: String,
}

impl Neo4jConfig {
    pub fn from_env() -> Option<Self> {
        let user = std::env::var("PETE_NEO4J_USER").ok()?;
        let password = std::env::var("PETE_NEO4J_PASSWORD").ok()?;
        let http_url = std::env::var("PETE_NEO4J_HTTP_URL")
            .ok()
            .or_else(|| {
                std::env::var("PETE_NEO4J_URI")
                    .ok()
                    .and_then(|uri| neo4j_http_url_from_uri(&uri))
            })
            .unwrap_or_else(|| "http://localhost:7474".to_string());
        let database = std::env::var("PETE_NEO4J_DATABASE").unwrap_or_else(|_| "neo4j".to_string());
        Some(Self {
            http_url,
            user,
            password,
            database,
        })
    }
}

#[derive(Clone)]
pub struct Neo4jGraphStore {
    client: reqwest::Client,
    config: Neo4jConfig,
    legacy_related_migration: Arc<tokio::sync::OnceCell<()>>,
}

impl Neo4jGraphStore {
    pub fn new(config: Neo4jConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
            legacy_related_migration: Arc::new(tokio::sync::OnceCell::new()),
        }
    }

    pub fn from_env() -> Option<Self> {
        Neo4jConfig::from_env().map(Self::new)
    }

    async fn migrate_legacy_related_edges(&self) -> Result<()> {
        self.legacy_related_migration
            .get_or_try_init(|| async {
                self.run_cypher(NEO4J_LEGACY_RELATED_EDGE_MIGRATION_CYPHER, json!({}))
                    .await
            })
            .await?;
        Ok(())
    }

    async fn run_cypher(&self, statement: &str, parameters: serde_json::Value) -> Result<()> {
        let url = format!(
            "{}/db/{}/tx/commit",
            self.config.http_url.trim_end_matches('/'),
            self.config.database
        );
        let response = self
            .client
            .post(url)
            .basic_auth(&self.config.user, Some(&self.config.password))
            .json(&json!({
                "statements": [{
                    "statement": statement,
                    "parameters": parameters
                }]
            }))
            .send()
            .await
            .context("running neo4j cypher")?;
        if !response.status().is_success() {
            return Err(anyhow!("neo4j cypher failed: HTTP {}", response.status()));
        }
        let body = response
            .json::<serde_json::Value>()
            .await
            .context("reading neo4j response")?;
        let errors = body
            .get("errors")
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap_or_default();
        if !errors.is_empty() {
            return Err(anyhow!("neo4j cypher errors: {errors:?}"));
        }
        Ok(())
    }

    async fn query_cypher(
        &self,
        statement: &str,
        parameters: serde_json::Value,
    ) -> Result<Vec<Vec<serde_json::Value>>> {
        let url = format!(
            "{}/db/{}/tx/commit",
            self.config.http_url.trim_end_matches('/'),
            self.config.database
        );
        let response = self
            .client
            .post(url)
            .basic_auth(&self.config.user, Some(&self.config.password))
            .json(&json!({
                "statements": [{
                    "statement": statement,
                    "parameters": parameters,
                    "resultDataContents": ["row"]
                }]
            }))
            .send()
            .await
            .context("querying neo4j cypher")?;
        if !response.status().is_success() {
            return Err(anyhow!("neo4j query failed: HTTP {}", response.status()));
        }
        let body = response
            .json::<serde_json::Value>()
            .await
            .context("reading neo4j query response")?;
        let errors = body
            .get("errors")
            .and_then(|value| value.as_array())
            .cloned()
            .unwrap_or_default();
        if !errors.is_empty() {
            return Err(anyhow!("neo4j query errors: {errors:?}"));
        }
        Ok(body
            .get("results")
            .and_then(|results| results.get(0))
            .and_then(|result| result.get("data"))
            .and_then(|data| data.as_array())
            .into_iter()
            .flatten()
            .filter_map(|item| item.get("row").and_then(|row| row.as_array()).cloned())
            .collect())
    }
}

#[async_trait]
impl GraphStore for Neo4jGraphStore {
    async fn upsert_graph(&self, record: &MemoryRecord) -> Result<()> {
        // Releases before stable edge ids merged RELATED edges by kind. Backfill
        // the surviving projection in one transaction before any new merge;
        // never delete historical relationships during an ordinary write.
        self.migrate_legacy_related_edges().await?;

        let entities = neo4j_entity_params(record);
        let relationships = neo4j_relationship_params(record);

        self.run_cypher(
            NEO4J_GRAPH_UPSERT_CYPHER,
            json!({
                "entities": entities,
                "relationships": relationships,
            }),
        )
        .await
    }
}

const NEO4J_LEGACY_RELATED_EDGE_MIGRATION_CYPHER: &str = r#"
MATCH (from:MemoryNode)-[legacy:RELATED]->(to:MemoryNode)
WHERE legacy.edge_id IS NULL
SET legacy.edge_id =
    CASE
        WHEN legacy.kind STARTS WITH 'SEMANTIC_'
             AND legacy.payload_json CONTAINS '"id":"'
        THEN split(split(legacy.payload_json, '"id":"')[1], '"')[0]
        ELSE 'graph-edge:'
             + toString(size(from.id)) + ':' + from.id + ':'
             + toString(size(legacy.kind)) + ':' + legacy.kind + ':'
             + toString(size(to.id)) + ':' + to.id
    END,
    legacy.edge_identity_migrated = true
"#;

const NEO4J_GRAPH_UPSERT_CYPHER: &str = r#"
UNWIND $entities AS entity
MERGE (n:MemoryNode {id: entity.id})
SET n.labels = entity.labels,
    n.summary = entity.summary,
    n.score = entity.score,
    n.frame_id = entity.frame_id,
    n.t_ms = entity.t_ms
WITH collect(n) AS ignored
UNWIND $relationships AS relationship
MATCH (from:MemoryNode {id: relationship.from})
MATCH (to:MemoryNode {id: relationship.to})
MERGE (from)-[r:RELATED {edge_id: relationship.edge_id}]->(to)
SET r.kind = relationship.kind,
    r.summary = relationship.summary,
    r.score = relationship.score,
    r.payload_json = relationship.payload_json,
    r.frame_id = relationship.frame_id,
    r.t_ms = relationship.t_ms
REMOVE r.payload
"#;

#[async_trait]
impl GraphIntelligence for Neo4jGraphStore {
    async fn upsert_intelligence(&self, document: &GraphIntelligenceDocument) -> Result<()> {
        let params = neo4j_intelligence_params(document);
        for statement in neo4j_intelligence_upsert_statements() {
            self.run_cypher(statement, params.clone()).await?;
        }
        Ok(())
    }

    async fn feature_or_cluster_intelligence(
        &self,
        node_id: &str,
        limit: usize,
    ) -> Result<FeatureClusterIntelligence> {
        let rows = self
            .query_cypher(
                r#"
MATCH (n {id: $id})
OPTIONAL MATCH (n)-[:MEMBER_OF]->(cluster:Cluster)
OPTIONAL MATCH (n)-[:BOUND_TO]-(bound:Cluster)
OPTIONAL MATCH (bc:BindingCandidate)-[:FROM|TO]->(n)
OPTIONAL MATCH (bc)-[:SUPPORTED_BY]->(support:Evidence)
OPTIONAL MATCH (bc)-[:REJECTED_BECAUSE]->(contradiction:Evidence)
OPTIONAL MATCH (co:Constellation)-[:HAS_MEMBER]->(n)
RETURN collect(DISTINCT cluster)[0..$limit],
       collect(DISTINCT bound)[0..$limit],
       collect(DISTINCT bc)[0..$limit],
       collect(DISTINCT support)[0..$limit],
       collect(DISTINCT contradiction)[0..$limit],
       collect(DISTINCT co)[0..$limit]
"#,
                json!({"id": node_id, "limit": limit as i64}),
            )
            .await?;
        let row = rows.first().cloned().unwrap_or_default();
        Ok(FeatureClusterIntelligence {
            node_id: node_id.to_string(),
            clusters: summaries_from_row(row.first()),
            bindings: summaries_from_row(row.get(1))
                .into_iter()
                .chain(summaries_from_row(row.get(2)))
                .collect(),
            supporting_evidence: summaries_from_row(row.get(3)),
            contradictions: summaries_from_row(row.get(4)),
            constellations: summaries_from_row(row.get(5)),
        })
    }

    async fn constellation_intelligence(
        &self,
        constellation_id: &str,
        limit: usize,
    ) -> Result<ConstellationIntelligence> {
        let rows = self
            .query_cypher(
                r#"
MATCH (co:Constellation {id: $id})
OPTIONAL MATCH (co)-[:HAS_MEMBER]->(member)
OPTIONAL MATCH (co)-[:SUPPORTED_BY]->(binding:BindingEdge)
OPTIONAL MATCH (co)<-[:TO]-(a:Association)-[:TO]->(prediction)
OPTIONAL MATCH (similar:Constellation)
WHERE similar.id <> co.id AND any(id IN co.member_cluster_ids WHERE id IN similar.member_cluster_ids)
OPTIONAL MATCH (co)<-[:CRITIQUES]-(lr:LlmReview)
RETURN co,
       collect(DISTINCT member)[0..$limit] + collect(DISTINCT binding)[0..$limit],
       collect(DISTINCT prediction)[0..$limit],
       collect(DISTINCT similar)[0..$limit],
       collect(DISTINCT lr)[0..$limit]
"#,
                json!({"id": constellation_id, "limit": limit as i64}),
            )
            .await?;
        let row = rows.first().cloned().unwrap_or_default();
        let constellation = row.first().map(summary_from_value).unwrap_or_default();
        let members = summaries_from_row(row.get(1));
        let known_member_ids = members
            .iter()
            .map(|member| member.id.clone())
            .collect::<BTreeSet<_>>();
        let expected = row
            .first()
            .and_then(|value| value.get("member_cluster_ids"))
            .and_then(|value| value.as_array())
            .into_iter()
            .flatten()
            .filter_map(|value| value.as_str().map(str::to_string))
            .collect::<Vec<_>>();
        Ok(ConstellationIntelligence {
            constellation_id: constellation_id.to_string(),
            state: constellation.state.clone().unwrap_or_default(),
            missing_members: expected
                .into_iter()
                .filter(|id| !known_member_ids.contains(id))
                .collect(),
            predictions: summaries_from_row(row.get(2)),
            similar_constellations: summaries_from_row(row.get(3)),
            contradictions: summaries_from_row(row.get(4)),
            stability: row
                .first()
                .and_then(|value| value.get("stability"))
                .and_then(value_as_f32)
                .unwrap_or_default(),
            reason: constellation.reason,
            members,
        })
    }

    async fn ambiguity_intelligence(
        &self,
        family_or_target_id: &str,
        limit: usize,
    ) -> Result<AmbiguityIntelligence> {
        let rows = self
            .query_cypher(
                r#"
MATCH (h:TrackingHypothesis)
WHERE h.family_id = $id OR h.target_id = $id OR h.id = $id
OPTIONAL MATCH (h)-[:SUPPORTS]->(bc:BindingCandidate)
OPTIONAL MATCH (bc)-[:SUPPORTED_BY|REJECTED_BECAUSE]->(e:Evidence)
OPTIONAL MATCH (h)<-[:CRITIQUES]-(lr:LlmReview)
RETURN collect(DISTINCT h)[0..$limit],
       collect(DISTINCT e)[0..$limit],
       collect(DISTINCT lr)[0..$limit]
"#,
                json!({"id": family_or_target_id, "limit": limit as i64}),
            )
            .await?;
        let row = rows.first().cloned().unwrap_or_default();
        let reviews = summaries_from_row(row.get(2));
        let question = reviews
            .iter()
            .find(|review| !review.reason.is_empty())
            .map(|review| review.reason.clone())
            .or_else(|| {
                Some(format!(
                    "Which hypothesis best explains {family_or_target_id}?"
                ))
            });
        Ok(AmbiguityIntelligence {
            target_id: family_or_target_id.to_string(),
            competing_hypotheses: summaries_from_row(row.first()),
            distinguishing_evidence: summaries_from_row(row.get(1)),
            contradictions: reviews.clone(),
            human_question: question,
        })
    }

    async fn action_outcome_intelligence(
        &self,
        action_id: &str,
        limit: usize,
    ) -> Result<ActionOutcomeIntelligence> {
        let rows = self
            .query_cypher(
                r#"
MATCH (a:ActionIntent {id: $id})
OPTIONAL MATCH (a)-[:RESULTED_IN]->(outcome:Outcome)
OPTIONAL MATCH (a)<-[:FROM]-(assoc:Association)-[:TO]->(next)
OPTIONAL MATCH (place:Place)<-[:FROM]-(risk:Association {relation: 'prevents'})
OPTIONAL MATCH (body:BodyState)<-[:FROM]-(prevent:Association {relation: 'prevents'})
RETURN collect(DISTINCT outcome)[0..$limit],
       collect(DISTINCT next)[0..$limit],
       collect(DISTINCT place)[0..$limit],
       collect(DISTINCT body)[0..$limit]
"#,
                json!({"id": action_id, "limit": limit as i64}),
            )
            .await?;
        let row = rows.first().cloned().unwrap_or_default();
        Ok(ActionOutcomeIntelligence {
            action_id: action_id.to_string(),
            outcomes: summaries_from_row(row.first()),
            usual_next: summaries_from_row(row.get(1)),
            risky_places: summaries_from_row(row.get(2)),
            preventing_body_states: summaries_from_row(row.get(3)),
        })
    }

    async fn local_community(
        &self,
        start_node_id: &str,
        max_depth: u32,
        min_weight: f32,
        limit: usize,
    ) -> Result<GraphCommunity> {
        let rows = self
            .query_cypher(
                r#"
MATCH path = (start {id: $id})-[rels*1..4]-(node)
WHERE length(path) <= $max_depth
WITH node, rels, length(path) AS depth,
     reduce(score = 0.0, r IN rels | score + coalesce(r.confidence, r.score, 0.1)) AS weight
WHERE weight >= $min_weight
RETURN node.id, labels(node), weight, depth, size(rels),
       coalesce(node.reason, node.summary, node.current_state, '')
ORDER BY weight DESC, depth ASC
LIMIT $limit
"#,
                json!({
                    "id": start_node_id,
                    "max_depth": max_depth.min(4) as i64,
                    "min_weight": min_weight,
                    "limit": limit as i64,
                }),
            )
            .await?;
        let members = rows
            .into_iter()
            .map(|row| GraphCommunityMember {
                node_id: row
                    .first()
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                labels: row
                    .get(1)
                    .and_then(|v| v.as_array())
                    .into_iter()
                    .flatten()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect(),
                score: row.get(2).and_then(value_as_f32).unwrap_or_default(),
                depth: row.get(3).and_then(|v| v.as_u64()).unwrap_or_default() as u32,
                recurrence: row.get(4).and_then(|v| v.as_u64()).unwrap_or_default() as u32,
                reason: row
                    .get(5)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
            })
            .collect::<Vec<_>>();
        Ok(GraphCommunity {
            start_node_id: start_node_id.to_string(),
            max_depth,
            min_weight,
            summary: format!("{} strong nearby graph members", members.len()),
            members,
        })
    }

    async fn graph_recall(
        &self,
        query: GraphRecallQuery,
        limit: usize,
    ) -> Result<GraphRecallBundle> {
        let query_ids = graph_recall_query_ids(&query);
        let rows = self
            .query_cypher(
                r#"
MATCH (seed)
WHERE seed.id IN $ids
OPTIONAL MATCH (seed)-[*1..2]-(near:MemoryNode)
OPTIONAL MATCH (co:Constellation)
WHERE co.id IN $ids OR any(id IN co.member_cluster_ids WHERE id IN $ids)
OPTIONAL MATCH (seed)<-[:FROM]-(assoc:Association)-[:TO]->(outcome:Outcome)
OPTIONAL MATCH (seed)<-[:REVIEWS|CRITIQUES]-(review)
OPTIONAL MATCH (hr:HumanReview)-[:CONFIRMS]->(seed)
OPTIONAL MATCH (lr:LlmReview)-[:CRITIQUES]->(seed)
OPTIONAL MATCH (ai:ActionIntent)-[:RESULTED_IN]->(action_outcome:Outcome)
WHERE ai.id IN $ids
RETURN collect(DISTINCT near)[0..$limit],
       collect(DISTINCT co)[0..$limit],
       collect(DISTINCT outcome)[0..$limit],
       collect(DISTINCT review)[0..$limit],
       collect(DISTINCT hr)[0..$limit],
       collect(DISTINCT lr)[0..$limit],
       collect(DISTINCT action_outcome)[0..$limit]
"#,
                json!({"ids": query_ids, "limit": limit as i64}),
            )
            .await?;
        let row = rows.first().cloned().unwrap_or_default();
        let action_outcomes = summaries_from_row(row.get(6));
        Ok(GraphRecallBundle {
            query_ids: graph_recall_query_ids(&query),
            nearby_memories: summaries_from_row(row.first()),
            similar_constellations: summaries_from_row(row.get(1)),
            likely_outcomes: summaries_from_row(row.get(2)),
            previous_contradictions: summaries_from_row(row.get(3)),
            human_confirmations: summaries_from_row(row.get(4)),
            llm_critiques: summaries_from_row(row.get(5)),
            action_successes: action_outcomes
                .iter()
                .filter(|summary| summary.state.as_deref() == Some("succeeded"))
                .cloned()
                .collect(),
            action_failures: action_outcomes
                .into_iter()
                .filter(|summary| summary.state.as_deref() == Some("failed"))
                .collect(),
        })
    }

    async fn consistency_checks(&self, limit: usize) -> Result<Vec<GraphReviewRecord>> {
        let rows = self
            .query_cypher(
                r#"
MATCH (target)
WHERE (target:BindingCandidate AND target.decision IN ['reject', 'hold_ambiguous', 'ask_human'])
   OR (target:TrackingHypothesis AND target.current_state IN ['needs_review', 'rejected'])
   OR (target:Constellation AND target.current_state IN ['ambiguous', 'split_needed', 'merge_needed'])
   OR (target:Association AND coalesce(target.contradiction_count, 0) > 0)
   OR (target:Prediction)-[:FAILED_WITH]->(:Surprise)
RETURN target.id, labels(target)[0], coalesce(target.confidence, 0.5),
       coalesce(target.last_updated_ms, target.last_seen_ms, target.t_ms, 0),
       coalesce(target.reason, target.current_state, target.decision, 'suspicious graph state')
ORDER BY coalesce(target.last_updated_ms, target.last_seen_ms, target.t_ms, 0) DESC
LIMIT $limit
"#,
                json!({"limit": limit as i64}),
            )
            .await?;
        Ok(rows
            .into_iter()
            .enumerate()
            .map(|(index, row)| {
                let target_id = row.first().and_then(|v| v.as_str()).unwrap_or_default();
                let kind = row.get(1).and_then(|v| v.as_str()).unwrap_or("graph");
                GraphReviewRecord {
                    id: format!(
                        "graph-review:{}:{}",
                        stable_slug(kind),
                        stable_slug(target_id)
                    ),
                    target_id: target_id.to_string(),
                    review_kind: kind.to_string(),
                    severity: 1.0 - row.get(2).and_then(value_as_f32).unwrap_or(0.5),
                    confidence: row.get(2).and_then(value_as_f32).unwrap_or(0.5),
                    t_ms: row.get(3).and_then(|v| v.as_u64()).unwrap_or(index as u64),
                    reason: row
                        .get(4)
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    evidence_ids: Vec::new(),
                    state: "open".to_string(),
                }
            })
            .collect())
    }
}

fn neo4j_intelligence_upsert_statements() -> &'static [&'static str] {
    &[
        r#"
MERGE (doc:GraphIntelligenceWrite {id: $document.id})
SET doc.t_ms = $document.t_ms,
    doc.frame_id = $document.frame_id,
    doc.provenance = $document.provenance,
    doc.confidence = $document.confidence,
    doc.reason = $document.reason,
    doc.source_frame_ids = $document.source_frame_ids
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $features AS feature
MERGE (f:Feature {id: feature.id})
SET f.feature_type = feature.feature_type,
    f.modality = feature.modality,
    f.created_at_ms = feature.created_at_ms,
    f.confidence = feature.confidence,
    f.provenance_json = feature.provenance_json,
    f.source_frame = feature.source_frame,
    f.source_sensor = feature.source_sensor,
    f.vector_refs_json = feature.vector_refs_json,
    f.metadata_json = feature.metadata_json,
    f.current_state = feature.current_state,
    f.reason = feature.reason
MERGE (doc)-[:ASSERTS]->(f)
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $clusters AS cluster
MERGE (c:Cluster {id: cluster.id})
SET c.modality = cluster.modality,
    c.kind = cluster.kind,
    c.first_seen_ms = cluster.first_seen_ms,
    c.last_seen_ms = cluster.last_seen_ms,
    c.confidence = cluster.confidence,
    c.evidence_count = cluster.evidence_count,
    c.source_frame_id = cluster.source_frame_id,
    c.current_state = cluster.current_state,
    c.reason = cluster.reason,
    c.metadata_json = cluster.metadata_json
MERGE (doc)-[:ASSERTS]->(c)
"#,
        r#"
UNWIND $cluster_features AS rel
MATCH (f:Feature {id: rel.feature_id})
MATCH (c:Cluster {id: rel.cluster_id})
MERGE (f)-[r:MEMBER_OF]->(c)
SET r.confidence = rel.confidence,
    r.t_ms = rel.t_ms,
    r.provenance = rel.provenance,
    r.source_frame_ids = rel.source_frame_ids
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $binding_candidates AS candidate
MERGE (bc:BindingCandidate {id: candidate.id})
SET bc.left_cluster_id = candidate.left_cluster_id,
    bc.right_cluster_id = candidate.right_cluster_id,
    bc.relation = candidate.relation,
    bc.confidence = candidate.confidence,
    bc.evidence_count = candidate.evidence_count,
    bc.decision = candidate.decision,
    bc.current_state = candidate.current_state,
    bc.reason = candidate.reason,
    bc.t_ms = candidate.t_ms,
    bc.provenance = candidate.provenance,
    bc.source_frame_ids = candidate.source_frame_ids
MERGE (doc)-[:ASSERTS]->(bc)
WITH candidate, bc
OPTIONAL MATCH (left:Cluster {id: candidate.left_cluster_id})
OPTIONAL MATCH (right:Cluster {id: candidate.right_cluster_id})
FOREACH (_ IN CASE WHEN left IS NULL THEN [] ELSE [1] END | MERGE (bc)-[:FROM]->(left))
FOREACH (_ IN CASE WHEN right IS NULL THEN [] ELSE [1] END | MERGE (bc)-[:TO]->(right))
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $binding_edges AS edge
MERGE (be:BindingEdge {id: edge.id})
SET be.left_cluster_id = edge.left_cluster_id,
    be.right_cluster_id = edge.right_cluster_id,
    be.relation = edge.relation,
    be.confidence = edge.confidence,
    be.evidence_count = edge.evidence_count,
    be.last_seen_ms = edge.last_seen_ms,
    be.current_state = edge.current_state,
    be.reason = edge.reason
MERGE (doc)-[:ASSERTS]->(be)
WITH edge, be
OPTIONAL MATCH (left:Cluster {id: edge.left_cluster_id})
OPTIONAL MATCH (right:Cluster {id: edge.right_cluster_id})
FOREACH (_ IN CASE WHEN left IS NULL OR right IS NULL THEN [] ELSE [1] END |
    MERGE (left)-[r:BOUND_TO]->(right)
    SET r.binding_id = edge.id,
        r.relation = edge.relation,
        r.confidence = edge.confidence,
        r.evidence_count = edge.evidence_count,
        r.last_seen_ms = edge.last_seen_ms)
"#,
        r#"
UNWIND $candidate_edges AS rel
MATCH (bc:BindingCandidate {id: rel.candidate_id})
MATCH (be:BindingEdge {id: rel.binding_id})
MERGE (bc)-[r:PROPOSES]->(be)
SET r.confidence = rel.confidence,
    r.reason = rel.reason,
    r.t_ms = rel.t_ms
"#,
        r#"
UNWIND $candidate_evidence AS evidence
MERGE (e:Evidence {id: evidence.id})
SET e.kind = evidence.kind,
    e.score = evidence.score,
    e.reason = evidence.reason,
    e.t_ms = evidence.t_ms,
    e.current_state = evidence.current_state
WITH evidence, e
MATCH (bc:BindingCandidate {id: evidence.candidate_id})
FOREACH (_ IN CASE WHEN evidence.contradictory THEN [1] ELSE [] END | MERGE (bc)-[:REJECTED_BECAUSE]->(e))
FOREACH (_ IN CASE WHEN evidence.contradictory THEN [] ELSE [1] END | MERGE (bc)-[:SUPPORTED_BY]->(e))
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $tracking_hypotheses AS hyp
MERGE (h:TrackingHypothesis {id: hyp.id})
SET h.family_id = hyp.family_id,
    h.kind = hyp.kind,
    h.target_id = hyp.target_id,
    h.confidence = hyp.confidence,
    h.evidence_count = hyp.evidence_count,
    h.current_state = hyp.current_state,
    h.first_seen_ms = hyp.first_seen_ms,
    h.last_updated_ms = hyp.last_updated_ms,
    h.contradictions = hyp.contradictions,
    h.reason = hyp.reason
MERGE (doc)-[:ASSERTS]->(h)
"#,
        r#"
UNWIND $hypothesis_candidates AS rel
MATCH (h:TrackingHypothesis {id: rel.hypothesis_id})
MATCH (bc:BindingCandidate {id: rel.candidate_id})
MERGE (h)-[r:SUPPORTS]->(bc)
SET r.confidence = rel.confidence,
    r.t_ms = rel.t_ms
"#,
        r#"
UNWIND $hypothesis_competitions AS rel
MATCH (left:TrackingHypothesis {id: rel.left_id})
MATCH (right:TrackingHypothesis {id: rel.right_id})
MERGE (left)-[r:COMPETES_WITH]->(right)
SET r.family_id = rel.family_id,
    r.confidence = rel.confidence,
    r.t_ms = rel.t_ms
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $constellations AS constellation
MERGE (co:Constellation {id: constellation.id})
SET co.kind_hint = constellation.kind_hint,
    co.member_cluster_ids = constellation.member_cluster_ids,
    co.member_binding_ids = constellation.member_binding_ids,
    co.confidence = constellation.confidence,
    co.stability = constellation.stability,
    co.prediction_value = constellation.prediction_value,
    co.first_seen_ms = constellation.first_seen_ms,
    co.last_seen_ms = constellation.last_seen_ms,
    co.evidence_count = constellation.evidence_count,
    co.current_state = constellation.current_state,
    co.reason = constellation.reason,
    co.notes = constellation.notes
MERGE (doc)-[:ASSERTS]->(co)
"#,
        r#"
UNWIND $constellation_members AS rel
MATCH (co:Constellation {id: rel.constellation_id})
OPTIONAL MATCH (c:Cluster {id: rel.member_id})
OPTIONAL MATCH (be:BindingEdge {id: rel.member_id})
FOREACH (_ IN CASE WHEN c IS NULL THEN [] ELSE [1] END | MERGE (co)-[r:HAS_MEMBER]->(c) SET r.confidence = rel.confidence, r.t_ms = rel.t_ms)
FOREACH (_ IN CASE WHEN be IS NULL THEN [] ELSE [1] END | MERGE (co)-[r:SUPPORTED_BY]->(be) SET r.confidence = rel.confidence, r.t_ms = rel.t_ms)
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $associations AS assoc
MERGE (a:Association {id: assoc.id})
SET a.from_id = assoc.from_id,
    a.to_id = assoc.to_id,
    a.relation = assoc.relation,
    a.confidence = assoc.confidence,
    a.evidence_count = assoc.evidence_count,
    a.prediction_gain = assoc.prediction_gain,
    a.contradiction_count = assoc.contradiction_count,
    a.first_seen_ms = assoc.first_seen_ms,
    a.last_seen_ms = assoc.last_seen_ms,
    a.current_state = assoc.current_state,
    a.reason = assoc.reason,
    a.examples_json = assoc.examples_json
MERGE (doc)-[:ASSERTS]->(a)
WITH assoc, a
OPTIONAL MATCH (from {id: assoc.from_id})
OPTIONAL MATCH (to {id: assoc.to_id})
FOREACH (_ IN CASE WHEN from IS NULL THEN [] ELSE [1] END | MERGE (a)-[:FROM]->(from))
FOREACH (_ IN CASE WHEN to IS NULL THEN [] ELSE [1] END | MERGE (a)-[:TO]->(to))
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $action_intents AS action
MERGE (ai:ActionIntent {id: action.id})
SET ai.action_json = action.action_json,
    ai.frame_id = action.frame_id,
    ai.t_ms = action.t_ms,
    ai.confidence = action.confidence,
    ai.current_state = action.current_state,
    ai.reason = action.reason
MERGE (doc)-[:ASSERTS]->(ai)
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $outcomes AS outcome
MERGE (o:Outcome {id: outcome.id})
SET o.frame_id = outcome.frame_id,
    o.t_ms = outcome.t_ms,
    o.reward = outcome.reward,
    o.success = outcome.success,
    o.confidence = outcome.confidence,
    o.current_state = outcome.current_state,
    o.reason = outcome.reason
MERGE (doc)-[:ASSERTS]->(o)
"#,
        r#"
UNWIND $action_outcomes AS rel
MATCH (ai:ActionIntent {id: rel.action_id})
MATCH (o:Outcome {id: rel.outcome_id})
MERGE (ai)-[r:RESULTED_IN]->(o)
SET r.confidence = rel.confidence,
    r.t_ms = rel.t_ms,
    r.reason = rel.reason
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $predictions AS prediction
MERGE (p:Prediction {id: prediction.id})
SET p.target_id = prediction.target_id,
    p.predicted = prediction.predicted,
    p.confidence = prediction.confidence,
    p.t_ms = prediction.t_ms,
    p.current_state = prediction.current_state,
    p.reason = prediction.reason
MERGE (doc)-[:ASSERTS]->(p)
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $surprises AS surprise
MERGE (s:Surprise {id: surprise.id})
SET s.target_id = surprise.target_id,
    s.observed = surprise.observed,
    s.surprise = surprise.surprise,
    s.confidence = surprise.confidence,
    s.t_ms = surprise.t_ms,
    s.reason = surprise.reason
MERGE (doc)-[:ASSERTS]->(s)
"#,
        r#"
UNWIND $prediction_failures AS rel
MATCH (p:Prediction {id: rel.prediction_id})
MATCH (s:Surprise {id: rel.surprise_id})
MERGE (p)-[r:FAILED_WITH]->(s)
SET r.confidence = rel.confidence,
    r.t_ms = rel.t_ms,
    r.reason = rel.reason
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $llm_reviews AS review
MERGE (lr:LlmReview {id: review.id})
SET lr.target_id = review.target_id,
    lr.target_kind = review.target_kind,
    lr.confidence = review.confidence,
    lr.t_ms = review.t_ms,
    lr.critique = review.critique,
    lr.contradictions = review.contradictions,
    lr.suggested_questions = review.suggested_questions,
    lr.current_state = review.current_state
MERGE (doc)-[:ASSERTS]->(lr)
WITH review, lr
OPTIONAL MATCH (target {id: review.target_id})
FOREACH (_ IN CASE WHEN target IS NULL THEN [] ELSE [1] END | MERGE (lr)-[:CRITIQUES]->(target))
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $human_reviews AS review
MERGE (hr:HumanReview {id: review.id})
SET hr.target_id = review.target_id,
    hr.target_kind = review.target_kind,
    hr.confidence = review.confidence,
    hr.t_ms = review.t_ms,
    hr.confirmation = review.confirmation,
    hr.reviewer = review.reviewer,
    hr.current_state = review.current_state
MERGE (doc)-[:ASSERTS]->(hr)
WITH review, hr
OPTIONAL MATCH (target {id: review.target_id})
FOREACH (_ IN CASE WHEN target IS NULL THEN [] ELSE [1] END | MERGE (hr)-[:CONFIRMS]->(target))
"#,
        r#"
MATCH (doc:GraphIntelligenceWrite {id: $document.id})
UNWIND $review_records AS review
MERGE (gr:GraphReview {id: review.id})
SET gr.target_id = review.target_id,
    gr.review_kind = review.review_kind,
    gr.severity = review.severity,
    gr.confidence = review.confidence,
    gr.t_ms = review.t_ms,
    gr.reason = review.reason,
    gr.evidence_ids = review.evidence_ids,
    gr.current_state = review.current_state
MERGE (doc)-[:ASSERTS]->(gr)
WITH review, gr
OPTIONAL MATCH (target {id: review.target_id})
FOREACH (_ IN CASE WHEN target IS NULL THEN [] ELSE [1] END | MERGE (gr)-[:REVIEWS]->(target))
"#,
    ]
}

fn neo4j_intelligence_params(document: &GraphIntelligenceDocument) -> serde_json::Value {
    let document_meta = json!({
        "id": document.id,
        "t_ms": document.t_ms,
        "frame_id": document.frame_id,
        "provenance": document.provenance,
        "confidence": document.confidence,
        "reason": document.reason,
        "source_frame_ids": document.source_frame_ids,
    });
    json!({
        "document": document_meta,
        "features": document.features.iter().map(|feature| json!({
            "id": feature.id.to_string(),
            "feature_type": format!("{:?}", feature.feature_type),
            "modality": feature.modality.as_str(),
            "created_at_ms": feature.created_at_ms,
            "confidence": feature.confidence,
            "provenance_json": json_string(&feature.provenance),
            "source_frame": feature.source_frame,
            "source_sensor": feature.source_sensor,
            "vector_refs_json": json_string(&feature.vector_refs),
            "metadata_json": json_string(&feature.metadata),
            "current_state": "observed",
            "reason": document.reason,
        })).collect::<Vec<_>>(),
        "clusters": document.clusters.iter().map(|cluster| json!({
            "id": cluster.id,
            "modality": cluster.modality.as_str(),
            "kind": format!("{:?}", cluster.kind),
            "first_seen_ms": cluster.first_seen_ms,
            "last_seen_ms": cluster.last_seen_ms,
            "confidence": cluster.confidence,
            "evidence_count": cluster.feature_ids.len() as u32,
            "source_frame_id": cluster.source_frame_id,
            "current_state": "active",
            "reason": document.reason,
            "metadata_json": json_string(&cluster.metadata),
        })).collect::<Vec<_>>(),
        "cluster_features": document.clusters.iter().flat_map(|cluster| {
            cluster.feature_ids.iter().map(|feature_id| json!({
                "cluster_id": cluster.id,
                "feature_id": feature_id.to_string(),
                "confidence": cluster.confidence,
                "t_ms": cluster.last_seen_ms,
                "provenance": document.provenance,
                "source_frame_ids": document.source_frame_ids,
            }))
        }).collect::<Vec<_>>(),
        "binding_candidates": document.binding_candidates.iter().map(|candidate| {
            let id = binding_candidate_id(candidate);
            json!({
                "id": id,
                "left_cluster_id": candidate.left_cluster_id,
                "right_cluster_id": candidate.right_cluster_id,
                "relation": binding_relation_slug(&candidate.relation),
                "confidence": candidate.confidence,
                "evidence_count": candidate.evidence.len() as u32,
                "decision": binding_decision_slug(&candidate.decision),
                "current_state": binding_decision_slug(&candidate.decision),
                "reason": candidate.reason,
                "t_ms": document.t_ms,
                "provenance": document.provenance,
                "source_frame_ids": document.source_frame_ids,
            })
        }).collect::<Vec<_>>(),
        "binding_edges": document.binding_edges.iter().map(|edge| {
            let id = binding_edge_id(edge);
            json!({
                "id": id,
                "left_cluster_id": edge.left_cluster_id,
                "right_cluster_id": edge.right_cluster_id,
                "relation": binding_relation_slug(&edge.relation),
                "confidence": edge.confidence,
                "evidence_count": edge.evidence_count,
                "last_seen_ms": edge.last_seen_ms,
                "current_state": if edge.is_strong() { "accepted" } else { "provisional" },
                "reason": format!("{} evidence events", edge.evidence_count),
            })
        }).collect::<Vec<_>>(),
        "candidate_edges": document.binding_candidates.iter().map(|candidate| json!({
            "candidate_id": binding_candidate_id(candidate),
            "binding_id": binding_edge_id_from_parts(&candidate.left_cluster_id, &candidate.right_cluster_id, &candidate.relation),
            "confidence": candidate.confidence,
            "reason": candidate.reason,
            "t_ms": document.t_ms,
        })).collect::<Vec<_>>(),
        "candidate_evidence": document.binding_candidates.iter().flat_map(|candidate| {
            let candidate_id = binding_candidate_id(candidate);
            candidate.evidence.iter().enumerate().map(move |(index, evidence)| json!({
                "id": format!("evidence:{}:{}", stable_slug(&candidate_id), index),
                "candidate_id": candidate_id,
                "kind": binding_evidence_slug(&evidence.kind),
                "score": evidence.score,
                "reason": evidence.reason,
                "t_ms": document.t_ms,
                "current_state": if binding_evidence_is_contradictory(evidence) { "contradictory" } else { "supporting" },
                "contradictory": binding_evidence_is_contradictory(evidence),
            }))
        }).collect::<Vec<_>>(),
        "tracking_hypotheses": document.tracking_hypotheses.iter().map(|hypothesis| json!({
            "id": hypothesis.id,
            "family_id": hypothesis.family_id,
            "kind": format!("{:?}", hypothesis.kind),
            "target_id": hypothesis.target_id,
            "confidence": hypothesis.confidence,
            "evidence_count": hypothesis.evidence.len() as u32,
            "current_state": format!("{:?}", hypothesis.state).to_lowercase(),
            "first_seen_ms": hypothesis.first_seen_ms,
            "last_updated_ms": hypothesis.last_updated_ms,
            "contradictions": hypothesis.contradictions,
            "reason": hypothesis.contradictions.first().cloned().unwrap_or_else(|| "tracking hypothesis evidence".to_string()),
        })).collect::<Vec<_>>(),
        "hypothesis_candidates": document.tracking_hypotheses.iter().flat_map(|hypothesis| {
            hypothesis.binding_candidate_ids.iter().map(|candidate_id| json!({
                "hypothesis_id": hypothesis.id,
                "candidate_id": candidate_id,
                "confidence": hypothesis.confidence,
                "t_ms": hypothesis.last_updated_ms,
            }))
        }).collect::<Vec<_>>(),
        "hypothesis_competitions": hypothesis_competition_params(&document.tracking_hypotheses),
        "constellations": document.constellations.iter().map(|constellation| json!({
            "id": constellation.id,
            "kind_hint": constellation.kind_hint,
            "member_cluster_ids": constellation.member_cluster_ids,
            "member_binding_ids": constellation.member_binding_ids,
            "confidence": constellation.confidence,
            "stability": constellation.stability,
            "prediction_value": constellation.prediction_value,
            "first_seen_ms": constellation.first_seen_ms,
            "last_seen_ms": constellation.last_seen_ms,
            "evidence_count": constellation.evidence_count,
            "current_state": constellation_state_slug(&constellation.state),
            "reason": constellation.notes.first().cloned().unwrap_or_else(|| "constellation evidence".to_string()),
            "notes": constellation.notes,
        })).collect::<Vec<_>>(),
        "constellation_members": document.constellations.iter().flat_map(|constellation| {
            constellation.member_cluster_ids.iter().chain(constellation.member_binding_ids.iter()).map(|member_id| json!({
                "constellation_id": constellation.id,
                "member_id": member_id,
                "confidence": constellation.confidence,
                "t_ms": constellation.last_seen_ms,
            }))
        }).collect::<Vec<_>>(),
        "associations": document.associations.iter().map(|edge| json!({
            "id": edge.id,
            "from_id": edge.from_id,
            "to_id": edge.to_id,
            "relation": association_relation_slug(&edge.relation),
            "confidence": edge.confidence,
            "evidence_count": edge.evidence_count,
            "prediction_gain": edge.prediction_gain,
            "contradiction_count": edge.contradiction_count,
            "first_seen_ms": edge.first_seen_ms,
            "last_seen_ms": edge.last_seen_ms,
            "current_state": if edge.contradiction_count > 0 { "needs_review" } else { "active" },
            "reason": edge.examples.last().map(|example| example.reason.clone()).unwrap_or_else(|| "association evidence".to_string()),
            "examples_json": json_string(&edge.examples),
        })).collect::<Vec<_>>(),
        "action_intents": document.action_intents.iter().map(|action| json!({
            "id": action.id,
            "action_json": json_string(&action.action),
            "frame_id": action.frame_id,
            "t_ms": action.t_ms,
            "confidence": action.confidence,
            "current_state": action.state,
            "reason": action.reason,
        })).collect::<Vec<_>>(),
        "outcomes": document.outcomes.iter().map(|outcome| json!({
            "id": outcome.id,
            "frame_id": outcome.frame_id,
            "t_ms": outcome.t_ms,
            "reward": outcome.reward,
            "success": outcome.success,
            "confidence": outcome.confidence,
            "current_state": outcome.state,
            "reason": outcome.reason,
        })).collect::<Vec<_>>(),
        "action_outcomes": document.action_intents.iter().zip(document.outcomes.iter()).map(|(action, outcome)| json!({
            "action_id": action.id,
            "outcome_id": outcome.id,
            "confidence": action.confidence.min(outcome.confidence),
            "t_ms": outcome.t_ms,
            "reason": outcome.reason,
        })).collect::<Vec<_>>(),
        "predictions": document.predictions.iter().map(|prediction| json!({
            "id": prediction.id,
            "target_id": prediction.target_id,
            "predicted": prediction.predicted,
            "confidence": prediction.confidence,
            "t_ms": prediction.t_ms,
            "current_state": prediction.state,
            "reason": prediction.reason,
        })).collect::<Vec<_>>(),
        "surprises": document.surprises.iter().map(|surprise| json!({
            "id": surprise.id,
            "target_id": surprise.target_id,
            "observed": surprise.observed,
            "surprise": surprise.surprise,
            "confidence": surprise.confidence,
            "t_ms": surprise.t_ms,
            "reason": surprise.reason,
        })).collect::<Vec<_>>(),
        "prediction_failures": document.predictions.iter().flat_map(|prediction| {
            document.surprises.iter().filter(move |surprise| surprise.target_id == prediction.target_id).map(move |surprise| json!({
                "prediction_id": prediction.id,
                "surprise_id": surprise.id,
                "confidence": prediction.confidence.min(surprise.confidence),
                "t_ms": surprise.t_ms,
                "reason": surprise.reason,
            }))
        }).collect::<Vec<_>>(),
        "llm_reviews": document.llm_reviews.iter().map(|review| json!({
            "id": review.id,
            "target_id": review.target_id,
            "target_kind": format!("{:?}", review.target_kind),
            "confidence": review.confidence,
            "t_ms": review.t_ms,
            "critique": review.critique,
            "contradictions": review.contradictions,
            "suggested_questions": review.suggested_questions,
            "current_state": if review.contradictions.is_empty() { "open" } else { "needs_review" },
        })).collect::<Vec<_>>(),
        "human_reviews": document.human_reviews.iter().map(|review| json!({
            "id": review.id,
            "target_id": review.target_id,
            "target_kind": format!("{:?}", review.target_kind),
            "confidence": review.confidence,
            "t_ms": review.t_ms,
            "confirmation": review.confirmation,
            "reviewer": review.reviewer,
            "current_state": "confirmed",
        })).collect::<Vec<_>>(),
        "review_records": document.review_records.iter().map(|review| json!({
            "id": review.id,
            "target_id": review.target_id,
            "review_kind": review.review_kind,
            "severity": review.severity,
            "confidence": review.confidence,
            "t_ms": review.t_ms,
            "reason": review.reason,
            "evidence_ids": review.evidence_ids,
            "current_state": review.state,
        })).collect::<Vec<_>>(),
    })
}

fn hypothesis_competition_params(hypotheses: &[TrackingHypothesis]) -> Vec<serde_json::Value> {
    let mut by_family = BTreeMap::<String, Vec<&TrackingHypothesis>>::new();
    for hypothesis in hypotheses {
        by_family
            .entry(hypothesis.family_id.clone())
            .or_default()
            .push(hypothesis);
    }
    by_family
        .into_iter()
        .flat_map(|(family_id, hypotheses)| {
            let mut params = Vec::new();
            for left in 0..hypotheses.len() {
                for right in (left + 1)..hypotheses.len() {
                    params.push(json!({
                        "left_id": hypotheses[left].id,
                        "right_id": hypotheses[right].id,
                        "family_id": family_id,
                        "confidence": hypotheses[left].confidence.min(hypotheses[right].confidence),
                        "t_ms": hypotheses[left].last_updated_ms.max(hypotheses[right].last_updated_ms),
                    }));
                }
            }
            params
        })
        .collect()
}

fn json_string(value: &impl Serialize) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}

fn summaries_from_row(value: Option<&serde_json::Value>) -> Vec<GraphFactSummary> {
    value
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .map(summary_from_value)
        .filter(|summary| !summary.id.is_empty())
        .collect()
}

fn summary_from_value(value: &serde_json::Value) -> GraphFactSummary {
    GraphFactSummary {
        id: value
            .get("id")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string(),
        kind: value
            .get("kind")
            .or_else(|| value.get("kind_hint"))
            .or_else(|| value.get("feature_type"))
            .or_else(|| value.get("relation"))
            .and_then(|value| value.as_str())
            .unwrap_or("graph_fact")
            .to_string(),
        relation: value
            .get("relation")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        confidence: value
            .get("confidence")
            .or_else(|| value.get("score"))
            .and_then(value_as_f32)
            .unwrap_or_default(),
        evidence_count: value
            .get("evidence_count")
            .and_then(|value| value.as_u64())
            .unwrap_or_default() as u32,
        t_ms: value
            .get("t_ms")
            .or_else(|| value.get("last_seen_ms"))
            .or_else(|| value.get("last_updated_ms"))
            .and_then(|value| value.as_u64())
            .unwrap_or_default(),
        state: value
            .get("current_state")
            .or_else(|| value.get("decision"))
            .and_then(|value| value.as_str())
            .map(str::to_string),
        reason: value
            .get("reason")
            .or_else(|| value.get("summary"))
            .or_else(|| value.get("critique"))
            .or_else(|| value.get("confirmation"))
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string(),
    }
}

fn value_as_f32(value: &serde_json::Value) -> Option<f32> {
    value.as_f64().map(|value| value as f32)
}

fn graph_recall_query_ids(query: &GraphRecallQuery) -> Vec<String> {
    query
        .active_feature_ids
        .iter()
        .chain(query.active_cluster_ids.iter())
        .chain(query.active_constellation_ids.iter())
        .chain(query.action_ids.iter())
        .chain(query.place_ids.iter())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn binding_edge_id(edge: &BindingEdge) -> String {
    binding_edge_id_from_parts(
        &edge.left_cluster_id,
        &edge.right_cluster_id,
        &edge.relation,
    )
}

fn binding_edge_id_from_parts(left: &str, right: &str, relation: &BindingRelation) -> String {
    format!(
        "binding-edge:{}:{}:{}",
        stable_slug(left),
        binding_relation_slug(relation),
        stable_slug(right)
    )
}

fn binding_relation_slug(relation: &BindingRelation) -> &'static str {
    match relation {
        BindingRelation::CooccursInTime => "cooccurs_in_time",
        BindingRelation::CooccursInEstimatedSpace => "cooccurs_in_estimated_space",
        BindingRelation::MovesTogether => "moves_together",
        BindingRelation::PredictsSameFutureEvents => "predicts_same_future_events",
        BindingRelation::NamedBy => "named_by",
        BindingRelation::ProjectsTo => "projects_to",
        BindingRelation::HasColorAtPose => "has_color_at_pose",
        BindingRelation::LikelySameEntity => "likely_same_entity",
        BindingRelation::ExplainsOutcome => "explains_outcome",
        BindingRelation::Contradicts => "contradicts",
        BindingRelation::RequiresReview => "requires_review",
    }
}

fn binding_decision_slug(decision: &BindingDecision) -> &'static str {
    match decision {
        BindingDecision::Accept => "accept",
        BindingDecision::Reject => "reject",
        BindingDecision::HoldAmbiguous => "hold_ambiguous",
        BindingDecision::AskHuman => "ask_human",
        BindingDecision::CollectMoreEvidence => "collect_more_evidence",
    }
}

fn binding_evidence_slug(kind: &BindingEvidenceKind) -> &'static str {
    match kind {
        BindingEvidenceKind::TemporalOverlap => "temporal_overlap",
        BindingEvidenceKind::SpatialOverlap => "spatial_overlap",
        BindingEvidenceKind::VectorSimilarity => "vector_similarity",
        BindingEvidenceKind::ProjectionAgreement => "projection_agreement",
        BindingEvidenceKind::PoseAgreement => "pose_agreement",
        BindingEvidenceKind::RepeatedCooccurrence => "repeated_cooccurrence",
        BindingEvidenceKind::SingleCandidateContext => "single_candidate_context",
        BindingEvidenceKind::HumanConfirmed => "human_confirmed",
        BindingEvidenceKind::LlmSuggested => "llm_suggested",
        BindingEvidenceKind::Contradiction => "contradiction",
        BindingEvidenceKind::SimultaneousConflict => "simultaneous_conflict",
    }
}

fn binding_evidence_is_contradictory(evidence: &BindingEvidence) -> bool {
    matches!(
        evidence.kind,
        BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
    )
}

fn constellation_state_slug(state: &ConstellationState) -> &'static str {
    match state {
        ConstellationState::Candidate => "candidate",
        ConstellationState::Stable => "stable",
        ConstellationState::Ambiguous => "ambiguous",
        ConstellationState::SplitNeeded => "split_needed",
        ConstellationState::MergeNeeded => "merge_needed",
        ConstellationState::Retired => "retired",
    }
}

fn neo4j_entity_params(record: &MemoryRecord) -> Vec<serde_json::Value> {
    record
        .graph_entities
        .iter()
        .map(|entity| {
            json!({
                "id": entity.id,
                "labels": entity.labels,
                "summary": entity.summary,
                "score": entity.score,
                "frame_id": record.frame_id.to_string(),
                "t_ms": record.t_ms,
            })
        })
        .collect()
}

fn neo4j_relationship_params(record: &MemoryRecord) -> Vec<serde_json::Value> {
    record
        .graph_relationships
        .iter()
        .map(|edge| {
            json!({
                "edge_id": graph_edge_id(edge),
                "from": edge.from,
                "to": edge.to,
                "kind": edge.relationship,
                "summary": edge.summary,
                "score": edge.score,
                "payload_json": edge.payload.to_string(),
                "frame_id": record.frame_id.to_string(),
                "t_ms": record.t_ms,
            })
        })
        .collect()
}

#[derive(Clone, Default)]
pub struct InMemoryExperienceStore {
    records: Arc<Mutex<Vec<MemoryRecord>>>,
    places: Arc<Mutex<PlaceMemory>>,
    entities: Arc<Mutex<EntityMemory>>,
}

impl InMemoryExperienceStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> Vec<MemoryRecord> {
        self.records.lock().expect("memory mutex poisoned").clone()
    }

    pub fn place_snapshot(&self) -> PlaceMemory {
        self.places
            .lock()
            .expect("place memory mutex poisoned")
            .clone()
    }

    pub fn place_report(&self) -> PlaceMemoryReport {
        self.place_snapshot().report()
    }

    pub fn entity_snapshot(&self) -> EntityMemory {
        self.entities
            .lock()
            .expect("entity memory mutex poisoned")
            .clone()
    }

    pub fn entity_report(&self) -> EntityMemoryReport {
        self.entity_snapshot().report()
    }

    pub fn last_social_interaction(&self, person_id: &PersonId) -> Option<InteractionState> {
        self.records
            .lock()
            .expect("memory mutex poisoned")
            .iter()
            .rev()
            .find_map(|record| {
                record
                    .social_world
                    .active_interaction
                    .iter()
                    .chain(record.social_world.recent_interactions.iter().rev())
                    .find(|interaction| interaction.participants.contains(person_id))
                    .cloned()
            })
    }

    pub fn completed_episodes(&self, kind: EpisodeKind) -> Vec<Episode> {
        let mut episodes = BTreeMap::new();
        for record in self.records.lock().expect("memory mutex poisoned").iter() {
            for episode in &record.temporal_context.recently_completed {
                if episode.kind == kind {
                    episodes.insert(episode.episode_id.clone(), episode.clone());
                }
            }
        }
        episodes.into_values().collect()
    }
}

#[async_trait]
impl MemoryStore for InMemoryExperienceStore {
    async fn store(&self, frame: &ExperienceFrame) -> Result<()> {
        self.store_record(memory_record_from_frame(frame)?).await
    }
}

impl InMemoryExperienceStore {
    async fn store_record(&self, record: MemoryRecord) -> Result<()> {
        self.records
            .lock()
            .expect("memory mutex poisoned")
            .push(record);
        Ok(())
    }
}

#[derive(Clone, Default)]
pub struct DurableExperienceStore {
    inner: InMemoryExperienceStore,
    vector_stores: Vec<Arc<dyn VectorStore>>,
    graph_stores: Vec<Arc<dyn GraphStore>>,
}

impl DurableExperienceStore {
    pub fn new(inner: InMemoryExperienceStore) -> Self {
        Self {
            inner,
            vector_stores: Vec::new(),
            graph_stores: Vec::new(),
        }
    }

    pub fn from_env() -> Self {
        let mut store = Self::new(InMemoryExperienceStore::new());
        if let Some(qdrant) = QdrantVectorStore::from_env() {
            store = store.with_vector_store(qdrant);
        }
        if let Some(neo4j) = Neo4jGraphStore::from_env() {
            store = store.with_graph_store(neo4j);
        }
        store
    }

    pub fn with_vector_store(mut self, store: impl VectorStore + 'static) -> Self {
        self.vector_stores.push(Arc::new(store));
        self
    }

    pub fn with_graph_store(mut self, store: impl GraphStore + 'static) -> Self {
        self.graph_stores.push(Arc::new(store));
        self
    }

    pub fn snapshot(&self) -> Vec<MemoryRecord> {
        self.inner.snapshot()
    }

    pub fn place_snapshot(&self) -> PlaceMemory {
        self.inner.place_snapshot()
    }

    pub fn place_report(&self) -> PlaceMemoryReport {
        self.inner.place_report()
    }

    pub fn entity_snapshot(&self) -> EntityMemory {
        self.inner.entity_snapshot()
    }

    pub fn entity_report(&self) -> EntityMemoryReport {
        self.inner.entity_report()
    }
}

#[async_trait]
impl MemoryStore for DurableExperienceStore {
    async fn store(&self, frame: &ExperienceFrame) -> Result<()> {
        let record = memory_record_from_frame(frame)?;
        self.inner.store_record(record.clone()).await?;
        for vector_store in &self.vector_stores {
            if let Err(error) = vector_store.upsert_vectors(&record).await {
                eprintln!("memory vector store write failed: {error:#}");
            }
        }
        for graph_store in &self.graph_stores {
            if let Err(error) = graph_store.upsert_graph(&record).await {
                eprintln!("memory graph store write failed: {error:#}");
            }
        }
        Ok(())
    }
}

#[async_trait]
impl Recall for DurableExperienceStore {
    async fn observe_now(&self, now: &Now) -> Result<()> {
        self.inner.observe_now(now).await
    }

    async fn observe_frame(&self, frame: &ExperienceFrame) -> Result<()> {
        self.inner.observe_frame(frame).await
    }

    async fn observe_transition(&self, transition: &ExperienceTransition) -> Result<()> {
        self.inner.observe_transition(transition).await
    }

    async fn loop_closure_candidates(
        &self,
        query: &RecallQuery,
        min_confidence: f32,
        limit: usize,
    ) -> Result<Vec<PlaceRecognitionCandidate>> {
        self.inner
            .loop_closure_candidates(query, min_confidence, limit)
            .await
    }

    async fn recall(&self, query: RecallQuery) -> Result<RecallBundle> {
        self.inner.recall(query).await
    }
}

#[async_trait]
impl Recall for InMemoryExperienceStore {
    async fn observe_now(&self, now: &Now) -> Result<()> {
        let cell_key = {
            let places = self.places.lock().expect("place memory mutex poisoned");
            Some(places.quantize(now.body.odometry.x_m, now.body.odometry.y_m))
        };
        self.places
            .lock()
            .expect("place memory mutex poisoned")
            .observe_now(now);
        self.entities
            .lock()
            .expect("entity memory mutex poisoned")
            .observe_now(now, cell_key);
        Ok(())
    }

    async fn observe_frame(&self, frame: &ExperienceFrame) -> Result<()> {
        let cell_key = {
            let places = self.places.lock().expect("place memory mutex poisoned");
            Some(places.quantize(frame.now.body.odometry.x_m, frame.now.body.odometry.y_m))
        };
        self.places
            .lock()
            .expect("place memory mutex poisoned")
            .observe_frame(frame);
        self.entities
            .lock()
            .expect("entity memory mutex poisoned")
            .observe_frame(frame, cell_key);
        Ok(())
    }

    async fn observe_transition(&self, transition: &ExperienceTransition) -> Result<()> {
        self.places
            .lock()
            .expect("place memory mutex poisoned")
            .observe_transition(transition);
        Ok(())
    }

    async fn loop_closure_candidates(
        &self,
        query: &RecallQuery,
        min_confidence: f32,
        limit: usize,
    ) -> Result<Vec<PlaceRecognitionCandidate>> {
        let places = self.places.lock().expect("place memory mutex poisoned");
        let current_key = query.pose.map(|pose| places.quantize(pose.x_m, pose.y_m));
        let mut query_vectors = query.scene_vectors.clone();
        if let Some(input) = query.place_recognition_input.as_ref() {
            query_vectors.extend(place_recognition_vectors_from_input(input));
        }
        let mut candidates =
            places.recognize_places(current_key, &query_vectors, min_confidence, limit);
        let mut entity_labels = query
            .place_recognition_input
            .as_ref()
            .map(|input| {
                input
                    .object_labels
                    .iter()
                    .chain(input.person_labels.iter())
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        entity_labels.sort();
        entity_labels.dedup();
        candidates.extend(places.recognize_entity_constellations(
            current_key,
            &entity_labels,
            min_confidence,
            limit,
        ));
        candidates.sort_by(|left, right| {
            right
                .confidence
                .partial_cmp(&left.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.cell.last_seen_tick.cmp(&left.cell.last_seen_tick))
        });
        candidates.truncate(limit);
        Ok(candidates)
    }

    async fn recall(&self, query: RecallQuery) -> Result<RecallBundle> {
        let place_features = query
            .pose
            .map(|pose| {
                self.places
                    .lock()
                    .expect("place memory mutex poisoned")
                    .features_at(pose.x_m, pose.y_m)
            })
            .unwrap_or_default();
        let records = self.snapshot();
        let mut scored = records
            .into_iter()
            .filter_map(|record| score_record(&query, record))
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.1.t_ms.cmp(&left.1.t_ms))
        });

        let mut hits = Vec::new();
        let mut recollections = Vec::new();
        let mut seen_actions = BTreeSet::new();
        let mut place_familiarity = 0.0f32;
        let mut place_danger = 0.0f32;
        let mut place_charge_value = 0.0f32;
        let mut face_familiarity = 0.0f32;
        let mut object_familiarity = 0.0f32;
        let mut voice_familiarity = 0.0f32;
        let mut remembered_warning = None;
        let mut best_remembered_action = None;
        let mut remembered_entities = Vec::new();
        let mut remembered_relationships = Vec::new();

        for (index, (score, record)) in scored.into_iter().take(5).enumerate() {
            hits.push(RecallHit {
                frame_id: Some(record.frame_id),
                score,
                summary: record.summary.clone(),
                warning: record.warning.clone(),
                graph_context: scored_entities(&record, score),
            });
            let scene_score = max_vector_similarity(
                query_scene_vectors(&query),
                record.scene_vectors.iter().collect(),
            );
            let face_score = max_vector_similarity(
                query_face_vectors(&query),
                record.face_vectors.iter().collect(),
            );
            let object_score = max_vector_similarity(
                query_object_vectors(&query),
                record.object_vectors.iter().collect(),
            );
            let voice_score = max_vector_similarity(
                query_voice_vectors(&query),
                record.voice_vectors.iter().collect(),
            );

            place_familiarity = place_familiarity.max(score).max(scene_score);
            if record.summary.to_ascii_lowercase().contains("danger") || record.warning.is_some() {
                place_danger = place_danger.max(score);
            }
            if matches!(record.active_goal, Some(Goal::Dock)) || record.summary.contains("charge") {
                place_charge_value = place_charge_value.max(score);
            }
            if has_face_query(&query) {
                face_familiarity = face_familiarity.max(score).max(face_score);
            }
            if has_object_query(&query) {
                object_familiarity = object_familiarity.max(score).max(object_score);
            }
            if has_voice_query(&query) {
                voice_familiarity = voice_familiarity.max(score).max(voice_score);
            }
            if remembered_warning.is_none() {
                remembered_warning = record.warning.clone();
            }
            if best_remembered_action.is_none() {
                if let Some(action) = &record.chosen_action {
                    let key = format!("{action:?}");
                    if seen_actions.insert(key) {
                        best_remembered_action = Some(action.clone());
                    }
                }
            }
            remembered_entities.extend(scored_entities(&record, score));
            remembered_relationships.extend(record.graph_relationships.clone());
            if let Some(experience) = record.experience.as_ref() {
                let original_vector_ids = recall_vector_ids(&record);
                let mut sensation = experience.to_recall_sensation_with_lineage(
                    query_pose_time_hint(&query, index as u64),
                    score,
                    "memory-recall",
                    Some(record.frame_id),
                    original_vector_ids.clone(),
                );
                let impression = experience.to_recall_impression(&sensation, score);
                sensation.impression = Some(impression);
                recollections.push(RecalledExperience {
                    score,
                    experience: experience.clone(),
                    sensation,
                    original_frame_id: Some(record.frame_id),
                    original_vector_ids,
                });
            }
        }
        remembered_entities = dedupe_entities(remembered_entities, 12);
        remembered_relationships = dedupe_relationships(remembered_relationships, 16);
        let graph_context_summary = graph_context_summary(&remembered_entities);

        let sense = MemorySense {
            place_familiarity: place_familiarity.max(place_features.current_place_familiarity),
            place_danger: place_danger.max(place_features.current_place_danger),
            place_charge_value: place_charge_value.max(place_features.current_place_charge),
            place_social_value: place_features.current_place_social,
            place_novelty: place_features.current_place_novelty,
            nearby_best_charge_direction_rad: place_features.nearby_best_charge_direction_rad,
            nearby_best_safe_direction_rad: place_features.nearby_best_safe_direction_rad,
            nearby_frontier_direction_rad: place_features.nearby_frontier_direction_rad,
            recent_trap_direction_rad: None,
            map_confidence: place_features.current_place_confidence,
            recent_trap_confidence: 0.0,
            places_visited: place_features.places_visited,
            face_familiarity,
            object_familiarity,
            voice_familiarity,
            similar_situation_count: hits.len().try_into().unwrap_or(u16::MAX),
            best_remembered_action,
            remembered_warning,
            remembered_entities,
            remembered_relationships,
            graph_context_summary,
        };
        let place_query_vectors = place_query_vectors_from_query(&query);
        let (semantic_map, place_recognition_candidates) = {
            let places = self.places.lock().expect("place memory mutex poisoned");
            let current_key = query.pose.map(|pose| places.quantize(pose.x_m, pose.y_m));
            let semantic_map = current_key.map(|key| {
                places.semantic_overlay_with_query(
                    Some(key),
                    &place_query_vectors,
                    PLACE_RECOGNITION_MIN_CONFIDENCE,
                )
            });
            let candidates = places.recognize_places(
                current_key,
                &place_query_vectors,
                PLACE_RECOGNITION_MIN_CONFIDENCE,
                5,
            );
            (semantic_map, candidates)
        };
        let first_person_summary = if hits.is_empty() {
            "I do not remember a similar situation yet.".to_string()
        } else {
            format!(
                "I remember {} similar moments. The closest one was: {}",
                hits.len(),
                hits[0].summary
            )
        };

        Ok(RecallBundle {
            hits,
            sense,
            first_person_summary,
            recollections,
            semantic_map,
            place_recognition_candidates,
        })
    }
}

pub fn memory_record_from_frame(frame: &ExperienceFrame) -> Result<MemoryRecord> {
    let warning = frame
        .memory_recall
        .iter()
        .find_map(|hit| hit.warning.clone())
        .or_else(|| frame.now.memory.remembered_warning.clone());
    let scene_vectors = scene_vectors_from_now(&frame.now, frame.id, frame.t_ms);
    let face_vectors = frame.now.face.vectors.clone();
    let object_vectors = frame.now.objects.vectors.clone();
    let voice_vectors = frame.now.voice.vectors.clone();
    let linked_experiences = experiences_with_memory_links(
        frame,
        &scene_vectors,
        &face_vectors,
        &object_vectors,
        &voice_vectors,
    );
    let (sensation_vectors, mut vector_payloads) = sensation_vectors_from_frame(frame);
    let experience_vectors = experience_vectors_from_frame(frame, &mut vector_payloads);
    let (graph_entities, graph_relationships) = graph_context_from_frame(
        frame,
        &linked_experiences,
        &scene_vectors,
        &face_vectors,
        &object_vectors,
        &voice_vectors,
    );
    Ok(MemoryRecord {
        frame_id: frame.id,
        t_ms: frame.t_ms,
        summary: frame.summary_text(),
        graph_entities,
        graph_relationships,
        scene_vectors,
        face_vectors,
        object_vectors,
        voice_vectors,
        sensation_vectors,
        experience_vectors,
        vector_payloads,
        battery: frame.now.body.battery_level,
        active_goal: RecallQuery::from_now(&frame.now).active_goal,
        chosen_action: frame.chosen_action.clone(),
        warning,
        experience: linked_experiences.last().cloned(),
        temporal_context: frame.now.world.temporal.clone(),
        social_world: frame.now.world.social.clone(),
        epistemic_state: frame.now.world.epistemic.clone(),
    })
}

pub fn attach_memory_links_to_frame(frame: &mut ExperienceFrame) {
    let scene_vectors = scene_vectors_from_now(&frame.now, frame.id, frame.t_ms);
    let face_vectors = frame.now.face.vectors.clone();
    let object_vectors = frame.now.objects.vectors.clone();
    let voice_vectors = frame.now.voice.vectors.clone();
    let links = memory_links_from_frame(
        frame,
        &scene_vectors,
        &face_vectors,
        &object_vectors,
        &voice_vectors,
    );
    for experience in &mut frame.experiences {
        merge_memory_links(&mut experience.memory_links, links.clone());
    }
}

fn experiences_with_memory_links(
    frame: &ExperienceFrame,
    scene_vectors: &[VectorArtifact],
    face_vectors: &[VectorArtifact],
    object_vectors: &[VectorArtifact],
    voice_vectors: &[VectorArtifact],
) -> Vec<Experience> {
    let links = memory_links_from_frame(
        frame,
        scene_vectors,
        face_vectors,
        object_vectors,
        voice_vectors,
    );
    frame
        .experiences
        .iter()
        .cloned()
        .map(|mut experience| {
            merge_memory_links(&mut experience.memory_links, links.clone());
            experience
        })
        .collect()
}

pub fn place_memory_report_from_frames(frames: &[ExperienceFrame]) -> PlaceMemoryReport {
    place_memory_from_frames(frames).report()
}

pub fn place_memory_from_frames(frames: &[ExperienceFrame]) -> PlaceMemory {
    let mut memory = PlaceMemory::new();
    for frame in frames {
        memory.observe_frame(frame);
    }
    memory
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EmbodiedEvalOmission {
    PrimarySensations,
    Descendants,
    Vectors,
    Impressions,
    FusedExperience,
    SummaryImpression,
    Predictions,
    MemoryPersistence,
    MemoryLinks,
    Recall,
}

impl EmbodiedEvalOmission {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PrimarySensations => "primary-sensations",
            Self::Descendants => "descendants",
            Self::Vectors => "vectors",
            Self::Impressions => "impressions",
            Self::FusedExperience => "fused-experience",
            Self::SummaryImpression => "summary-impression",
            Self::Predictions => "predictions",
            Self::MemoryPersistence => "memory-persistence",
            Self::MemoryLinks => "memory-links",
            Self::Recall => "recall",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedPipelineCoverageReport {
    pub schema_version: u32,
    pub fixture: String,
    pub placeholder: bool,
    pub placeholder_vector_count: usize,
    pub frame_count: usize,
    pub instant_count: usize,
    pub instant_teacher_vector_count: usize,
    pub instant_missing_modality_count: usize,
    pub primary_sensation_count: usize,
    pub descendant_sensation_count: usize,
    pub vectorized_sensation_count: usize,
    pub impression_count: usize,
    pub summary_impression_count: usize,
    pub experience_latent_count: usize,
    pub prediction_count: usize,
    pub memory_link_count: usize,
    pub recall_sensation_count: usize,
    pub recall_impression_count: usize,
    pub place_recognition_candidate_count: usize,
    pub lineage_edge_count: usize,
    pub input_modalities: Vec<String>,
    #[serde(default)]
    pub instant_coverage: Vec<InstantCoverage>,
    pub vector_coverage: EmbodiedVectorCoverage,
    pub warnings: Vec<String>,
    pub failures: Vec<String>,
}

impl EmbodiedPipelineCoverageReport {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

pub async fn deterministic_embodied_eval_report() -> Result<EmbodiedPipelineCoverageReport> {
    deterministic_embodied_eval_report_with_omissions(&[]).await
}

pub async fn deterministic_embodied_eval_report_with_omissions(
    omissions: &[EmbodiedEvalOmission],
) -> Result<EmbodiedPipelineCoverageReport> {
    let store = InMemoryExperienceStore::new();
    let prior_now = deterministic_embodied_fixture_now(1_000, 0.0);
    let mut prior = build_embodied_eval_frame(prior_now, None, omissions).await?;
    if !omitted(omissions, EmbodiedEvalOmission::MemoryLinks) {
        attach_memory_links_to_frame(&mut prior);
    }
    if !omitted(omissions, EmbodiedEvalOmission::MemoryPersistence) {
        store.store(&prior).await?;
        store.observe_frame(&prior).await?;
    }

    let current_now = deterministic_embodied_fixture_now(1_750, 0.08);
    let recall = if omitted(omissions, EmbodiedEvalOmission::Recall)
        || omitted(omissions, EmbodiedEvalOmission::MemoryPersistence)
    {
        None
    } else {
        Some(store.recall(RecallQuery::from_now(&current_now)).await?)
    };
    let mut current = build_embodied_eval_frame(current_now, recall.as_ref(), omissions).await?;
    if !omitted(omissions, EmbodiedEvalOmission::MemoryLinks) {
        attach_memory_links_to_frame(&mut current);
    }
    if !omitted(omissions, EmbodiedEvalOmission::MemoryPersistence) {
        store.store(&current).await?;
        store.observe_frame(&current).await?;
    }

    let persisted_frame_count = store.snapshot().len();
    let mut frames = vec![prior, current];
    let mut report = coverage_report_from_frames("deterministic", &frames);
    report.place_recognition_candidate_count = recall
        .as_ref()
        .map(|recall| recall.place_recognition_candidates.len())
        .unwrap_or_default();
    report.frame_count = persisted_frame_count.max(frames.len());
    if omitted(omissions, EmbodiedEvalOmission::MemoryPersistence) {
        report.frame_count = persisted_frame_count;
    }
    report.warnings.extend(
        omissions
            .iter()
            .map(|stage| format!("omitted {}", stage.as_str())),
    );
    evaluate_required_embodied_coverage(&mut report);
    frames.clear();
    Ok(report)
}

pub fn deterministic_embodied_fixture_now(t_ms: u64, pose_offset_m: f32) -> Now {
    let mut body = BodySense {
        battery_level: 0.72,
        charging: false,
        flags: BodyFlags {
            wall: true,
            ..BodyFlags::default()
        },
        odometry: Pose2 {
            x_m: 1.25 + pose_offset_m,
            y_m: -0.35,
            heading_rad: 0.18,
        },
        velocity: Velocity {
            forward_m_s: 0.06,
            turn_rad_s: 0.01,
        },
        last_update_ms: t_ms,
        ..BodySense::default()
    };
    body.cliff_sensors.front_left = 0.08;

    let mut rgb = vec![9_u8; 12 * 8 * 3];
    for y in 2..6 {
        for x in 4..8 {
            let idx = (y * 12 + x) * 3;
            rgb[idx] = 210;
            rgb[idx + 1] = 160;
            rgb[idx + 2] = 80;
        }
    }

    let mut now = Now::blank(t_ms, body);
    now.eye_frame = Some(EyeFrame {
        captured_at_ms: t_ms.saturating_sub(12),
        width: 12,
        height: 8,
        format: EyeFrameFormat::Rgb8,
        bytes: rgb,
        source: Some("fixture.synthetic_camera".to_string()),
    });
    now.eye.scene_vectors.push(
        VectorArtifact::new(
            SCENE_VECTOR_COLLECTION,
            "fixture-scene",
            vec![1.0, 0.0, 0.25, 0.5],
        )
        .with_model("fixture.scene.vector.v1")
        .with_source_frame_id("fixture-frame")
        .with_occurred_at_ms(t_ms),
    );
    now.range = RangeSense {
        schema_version: 1,
        beams: vec![0.42, 0.55, 1.2, 0.9, 0.48],
        nearest_m: Some(0.42),
        ..RangeSense::default()
    };
    now.kinect = KinectSense {
        schema_version: 1,
        depth_m: vec![0.72, 0.74, 0.81, 0.92, 1.05, 1.1],
        depth_width: 3,
        depth_height: 2,
        min_depth_m: 0.72,
        max_depth_m: 1.1,
        depth_coordinate_system: Some("fixture-depth-camera".to_string()),
        skeletons: vec![KinectSkeletonSense {
            tracking_id: 7,
            lean_xy: [0.02, -0.01],
            joints: vec![KinectJointSense {
                joint_name: "head".to_string(),
                position_m: [0.4, 0.1, 1.2],
                tracking_confidence: 0.8,
                tracked: true,
            }],
        }],
        ..KinectSense::default()
    };
    now.ear = EarSense {
        schema_version: 1,
        features: vec![vec![0.1, 0.2, 0.15, 0.05]],
        transcript: Some("fixture voice says remember the charger alcove".to_string()),
        transcript_vectors: vec![pete_now::VectorArtifact::new(
            "transcripts",
            "fixture-asr-transcript",
            vec![0.21, 0.34, 0.55, 0.89],
        )
        .with_model("pete.text.hashing.v1")
        .with_source_id("fixture-asr")
        .with_occurred_at_ms(t_ms)],
        asr: AsrSense {
            transcript: Some("fixture voice says remember the charger alcove".to_string()),
            is_final: true,
            confidence: 0.91,
            start_ms: Some(t_ms.saturating_sub(360)),
            end_ms: Some(t_ms),
            duration_ms: Some(360),
            sample_rate_hz: Some(16_000),
            word_count: Some(7),
            speaker_confidence: Some(0.77),
            ..AsrSense::default()
        },
    };
    now.voice.vectors.push(
        VectorArtifact::new(
            VOICE_VECTOR_COLLECTION,
            "fixture-voice",
            vec![0.2, 0.4, 0.6, 0.8],
        )
        .with_model("fixture.voice.vector.v1")
        .with_source_id("speaker:fixture")
        .with_occurred_at_ms(t_ms),
    );
    now.face.vectors.push(
        VectorArtifact::new(
            FACE_VECTOR_COLLECTION,
            "fixture-face",
            vec![0.8, 0.6, 0.4, 0.2],
        )
        .with_model("fixture.face.vector.v1")
        .with_source_id("person:fixture")
        .with_source_frame_id("fixture-frame")
        .with_occurred_at_ms(t_ms),
    );
    now.objects.observations.push(ObjectObservation {
        label: "charger alcove".to_string(),
        class: ObjectClass::Charger,
        bearing_rad: 0.15,
        distance_m: Some(0.9),
        confidence: 0.82,
        source: ObjectObservationSource::Captioner,
    });
    now
}

async fn build_embodied_eval_frame(
    now: Now,
    recall: Option<&RecallBundle>,
    omissions: &[EmbodiedEvalOmission],
) -> Result<ExperienceFrame> {
    let pipeline = EmbodiedPipeline::new();
    let mut sensations = Vec::new();
    let mut impressions = Vec::new();

    if !omitted(omissions, EmbodiedEvalOmission::PrimarySensations) {
        for primary in pete_experience::primary_sensations_from_now(&now) {
            let batch = pipeline.ingest_primary(primary).await?;
            sensations.extend(batch.sensations);
            impressions.extend(batch.impressions);
        }
    }
    if omitted(omissions, EmbodiedEvalOmission::Descendants) {
        let retained_ids = sensations
            .iter()
            .filter(|sensation| sensation.parent_id.is_none())
            .map(|sensation| sensation.id)
            .collect::<BTreeSet<_>>();
        sensations.retain(|sensation| sensation.parent_id.is_none());
        impressions.retain(|impression| {
            impression
                .sensation_id
                .map(|id| retained_ids.contains(&id))
                .unwrap_or(false)
        });
    }
    if omitted(omissions, EmbodiedEvalOmission::Vectors) {
        for sensation in &mut sensations {
            sensation.vector = None;
            if let Some(impression) = &mut sensation.impression {
                impression.vector = None;
            }
        }
        for impression in &mut impressions {
            impression.vector = None;
        }
    }
    if omitted(omissions, EmbodiedEvalOmission::Impressions) {
        for sensation in &mut sensations {
            sensation.impression = None;
        }
        impressions.clear();
    }

    let mut experience = if omitted(omissions, EmbodiedEvalOmission::FusedExperience)
        || sensations.is_empty()
        || impressions.is_empty()
    {
        None
    } else {
        let mut fused = ExperienceFuser::new(750).fuse(&sensations, &impressions)?;
        if omitted(omissions, EmbodiedEvalOmission::SummaryImpression) {
            fused.summary_impression = None;
        }
        if omitted(omissions, EmbodiedEvalOmission::Predictions) {
            fused.predictions.clear();
        }
        Some(fused)
    };

    if let (Some(experience), Some(recall)) = (&mut experience, recall) {
        for recollection in &recall.recollections {
            sensations.push(recollection.sensation.clone());
            if let Some(impression) = recollection.sensation.impression.clone() {
                impressions.push(impression.clone());
                experience.impression_ids.push(impression.id);
            }
            experience.sensation_ids.push(recollection.sensation.id);
        }
    }

    let summary_impression = experience
        .as_ref()
        .and_then(|experience| experience.summary_impression.clone());
    if let Some(summary) = summary_impression {
        impressions.push(summary);
    }
    let latent = pete_experience::ExperienceLatent {
        t_ms: now.t_ms,
        z: vec![
            (now.t_ms as f32 / 1_000.0).sin(),
            now.body.battery_level,
            now.body.odometry.x_m,
            now.body.odometry.y_m,
        ],
        reconstruction_error: 0.0,
        prediction_error: 0.0,
        confidence: 0.5,
    };

    Ok(ExperienceFrame {
        id: uuid::Uuid::new_v4(),
        t_ms: now.t_ms,
        now,
        sensations,
        impressions,
        experiences: experience.into_iter().collect(),
        z: Some(latent),
        chosen_action: Some(ActionPrimitive::Inspect {
            target: pete_actions::InspectTarget::Novelty,
        }),
        conscious_command: None,
        reign_input: None,
        reign_outcome: None,
        predicted_futures: vec![FuturePrediction {
            offset_ms: 750,
            predicted_z: vec![0.1, 0.2, 0.3, 0.4],
            confidence: 0.31,
            summary: Some("fallback latent future remains near the charger alcove".to_string()),
        }],
        behavior_runs: Vec::new(),
        actual_next: None,
        reward: Reward::default(),
        surprise: SurpriseSense::default(),
        memory_recall: recall.map(|recall| recall.hits.clone()).unwrap_or_default(),
        recollections: recall
            .map(|recall| recall.recollections.clone())
            .unwrap_or_default(),
        llm_teaching: Vec::new(),
        counterfactuals: Vec::new(),
        notes: vec!["deterministic embodied eval fixture".to_string()],
    })
}

fn coverage_report_from_frames(
    fixture: impl Into<String>,
    frames: &[ExperienceFrame],
) -> EmbodiedPipelineCoverageReport {
    let mut report = EmbodiedPipelineCoverageReport {
        schema_version: 1,
        fixture: fixture.into(),
        frame_count: frames.len(),
        ..EmbodiedPipelineCoverageReport::default()
    };
    let mut modalities = BTreeSet::new();
    for frame in frames {
        let instant = frame.experience_instant();
        let instant_coverage = instant.coverage();
        report.instant_count += 1;
        report.instant_teacher_vector_count += instant.teacher_vectors.len();
        report.instant_missing_modality_count += instant.missing_modalities.len();
        report.primary_sensation_count += instant.primary_sensations.len();
        report.descendant_sensation_count += instant.descendant_sensations.len();
        report.vectorized_sensation_count += frame
            .sensations
            .iter()
            .filter(|sensation| sensation.vector.is_some())
            .count();
        report.placeholder_vector_count += frame
            .sensations
            .iter()
            .filter_map(|sensation| sensation.vector.as_ref())
            .filter(|vector| vector.is_fallback)
            .count()
            + frame
                .impressions
                .iter()
                .filter_map(|impression| impression.vector.as_ref())
                .filter(|vector| vector.is_fallback)
                .count();
        report.impression_count += frame
            .impressions
            .iter()
            .filter(|impression| impression.sensation_id.is_some() || !impression.about.is_empty())
            .count();
        report.summary_impression_count += frame
            .experiences
            .iter()
            .filter(|experience| experience.summary_impression.is_some())
            .count();
        report.experience_latent_count += usize::from(frame.z.is_some());
        report.prediction_count += instant.predictions.len();
        report.memory_link_count += instant.memory_links.len();
        report.recall_sensation_count += frame
            .sensations
            .iter()
            .filter(|sensation| {
                sensation.modality == Modality::Memory
                    && sensation.payload_kind == SensationPayloadKind::MemoryRecall
            })
            .count();
        report.recall_impression_count += frame
            .impressions
            .iter()
            .filter(|impression| impression.kind == "memory.recall.impression")
            .count();
        report.lineage_edge_count += instant.lineage.len();
        modalities.extend(instant_coverage.present_modalities.iter().cloned());
        report.instant_coverage.push(instant_coverage);
        let coverage = EmbodiedVectorCoverage::from_parts(
            &frame.sensations,
            &frame.impressions,
            frame.experiences.last(),
        );
        merge_vector_coverage(&mut report.vector_coverage, coverage);
    }
    report.input_modalities = modalities.into_iter().collect();
    report.placeholder = report.placeholder_vector_count > 0;
    report
}

fn merge_vector_coverage(target: &mut EmbodiedVectorCoverage, incoming: EmbodiedVectorCoverage) {
    target.image += incoming.image;
    target.face += incoming.face;
    target.voice += incoming.voice;
    target.transcript += incoming.transcript;
    target.impression += incoming.impression;
    target.experience += incoming.experience;
    target.fallback_count += incoming.fallback_count;
}

fn evaluate_required_embodied_coverage(report: &mut EmbodiedPipelineCoverageReport) {
    required_stage(report.instant_count, "no instants", &mut report.failures);
    required_stage(
        report.instant_teacher_vector_count,
        "no instant teacher vectors",
        &mut report.failures,
    );
    required_stage(
        report.primary_sensation_count,
        "no primary sensations",
        &mut report.failures,
    );
    required_stage(
        report.descendant_sensation_count,
        "no descendants",
        &mut report.failures,
    );
    required_stage(
        report.vectorized_sensation_count,
        "no vectors",
        &mut report.failures,
    );
    required_stage(
        report.impression_count,
        "no impressions",
        &mut report.failures,
    );
    required_stage(
        report.experience_latent_count,
        "no learned experience latent",
        &mut report.failures,
    );
    required_stage(
        report.summary_impression_count,
        "no summary impression",
        &mut report.failures,
    );
    required_stage(
        report.prediction_count,
        "no prediction",
        &mut report.failures,
    );
    required_stage(
        report.memory_link_count,
        "no memory persistence/link",
        &mut report.failures,
    );
    required_stage(
        report.frame_count,
        "no memory persistence/link",
        &mut report.failures,
    );
    required_stage(
        report
            .recall_sensation_count
            .min(report.recall_impression_count),
        "no recall",
        &mut report.failures,
    );
    required_stage(
        report.place_recognition_candidate_count,
        "no place recognition",
        &mut report.failures,
    );
    required_stage(
        report.lineage_edge_count,
        "no lineage",
        &mut report.warnings,
    );
}

fn required_stage(count: usize, message: &str, messages: &mut Vec<String>) {
    if count == 0 && !messages.iter().any(|existing| existing == message) {
        messages.push(message.to_string());
    }
}

fn omitted(omissions: &[EmbodiedEvalOmission], stage: EmbodiedEvalOmission) -> bool {
    omissions.iter().any(|candidate| *candidate == stage)
}

fn danger_signal(now: &Now) -> f32 {
    let body = &now.body;
    let bumper = body.flags.bump_left || body.flags.bump_right;
    let cliff = body.flags.cliff_left
        || body.flags.cliff_front_left
        || body.flags.cliff_front_right
        || body.flags.cliff_right;
    let cliff_sensor = body.cliff_sensors.max();
    let nearest = now.range.nearest_m.unwrap_or(10.0);
    let range_risk = (1.0 - nearest / 0.7).clamp(0.0, 1.0);
    [
        if bumper { 1.0 } else { 0.0 },
        if body.flags.wall { 0.85 } else { 0.0 },
        if cliff {
            1.0
        } else {
            cliff_sensor.clamp(0.0, 1.0)
        },
        range_risk,
    ]
    .into_iter()
    .fold(0.0, f32::max)
}

fn charge_signal(now: &Now) -> f32 {
    let sim_score = now
        .extensions
        .get("sim.world")
        .and_then(|value| value.get("values"))
        .and_then(|value| value.as_array())
        .and_then(|values| {
            let near = values
                .get(3)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0) as f32;
            let visible = values
                .get(4)
                .and_then(|value| value.as_f64())
                .unwrap_or(0.0) as f32;
            Some(near.max(visible))
        })
        .unwrap_or(0.0);
    [if now.body.charging { 1.0 } else { 0.0 }, sim_score]
        .into_iter()
        .fold(0.0, f32::max)
}

fn social_signal(now: &Now) -> f32 {
    let visual = !now.face.vectors.is_empty() as u8 as f32;
    let voice = !now.voice.vectors.is_empty() as u8 as f32;
    let skeleton = (!now.kinect.skeletons.is_empty()) as u8 as f32;
    let transcript = now
        .ear
        .transcript
        .as_ref()
        .map(|text| (!text.trim().is_empty()) as u8 as f32)
        .unwrap_or(0.0);
    visual.max(voice).max(skeleton).max(transcript)
}

fn observed_object_summary(now: &Now) -> Vec<String> {
    let mut objects = now
        .objects
        .observations
        .iter()
        .filter(|observation| observation.confidence >= 0.3)
        .map(|observation| observation.label.clone())
        .collect::<Vec<_>>();
    if danger_signal(now) >= 0.5 {
        push_unique_object(&mut objects, "danger");
    }
    if charge_signal(now) >= 0.5 {
        push_unique_object(&mut objects, "charger");
    }
    if social_signal(now) >= 0.5 {
        push_unique_object(&mut objects, "person_or_speaker");
    }
    objects
}

fn push_unique_object(objects: &mut Vec<String>, value: &str) {
    if !objects.iter().any(|object| object == value) {
        objects.push(value.to_string());
    }
}

fn scene_vectors_with_frame_id(
    artifacts: &[VectorArtifact],
    frame_id: Option<&str>,
) -> Vec<VectorArtifact> {
    let Some(frame_id) = frame_id else {
        return artifacts.to_vec();
    };
    artifacts
        .iter()
        .cloned()
        .map(|mut artifact| {
            if artifact.source_frame_id.is_none() {
                artifact.source_frame_id = Some(frame_id.to_string());
            }
            artifact
        })
        .collect()
}

fn merge_vector_ids(target: &mut Vec<String>, artifacts: &[VectorArtifact]) {
    for artifact in artifacts {
        if artifact.point_id.trim().is_empty() {
            continue;
        }
        if !target.iter().any(|existing| existing == &artifact.point_id) {
            target.push(artifact.point_id.clone());
        }
    }
    const MAX_ASSOCIATED_VECTORS: usize = 12;
    if target.len() > MAX_ASSOCIATED_VECTORS {
        target.drain(0..target.len() - MAX_ASSOCIATED_VECTORS);
    }
}

fn update_action_outcome(
    outcomes: &mut Vec<ActionOutcomeSummary>,
    action: &ActionPrimitive,
    reward: f32,
    t_ms: u64,
) {
    if let Some(existing) = outcomes
        .iter_mut()
        .find(|candidate| candidate.action == *action)
    {
        let prior_total = existing.mean_reward * existing.count as f32;
        existing.count = existing.count.saturating_add(1);
        existing.mean_reward = (prior_total + reward) / existing.count.max(1) as f32;
        existing.last_seen_tick = t_ms;
    } else {
        outcomes.push(ActionOutcomeSummary {
            action: action.clone(),
            count: 1,
            mean_reward: reward,
            last_seen_tick: t_ms,
        });
    }
    outcomes.sort_by(|left, right| {
        right
            .mean_reward
            .abs()
            .partial_cmp(&left.mean_reward.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.last_seen_tick.cmp(&left.last_seen_tick))
    });
    const MAX_ACTION_OUTCOMES: usize = 8;
    outcomes.truncate(MAX_ACTION_OUTCOMES);
}

fn top_cells(
    cells: &BTreeMap<PlaceCellKey, PlaceCell>,
    score: impl Fn(&PlaceCell) -> f32,
) -> Vec<PlaceCellSummary> {
    let mut scored = cells
        .values()
        .map(|cell| (score(cell), cell))
        .filter(|(score, _)| *score > 0.0)
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored
        .into_iter()
        .take(5)
        .map(|(score, cell)| summarize_cell(cell, score))
        .collect()
}

fn summarize_cell(cell: &PlaceCell, score: f32) -> PlaceCellSummary {
    PlaceCellSummary {
        x: cell.key.x,
        y: cell.key.y,
        center_x_m: cell.center_x_m,
        center_y_m: cell.center_y_m,
        score,
        visit_count: cell.visit_count,
        last_seen_tick: cell.last_seen_tick,
        confidence: cell.confidence,
        last_observed_objects: cell.last_observed_objects.clone(),
        associated_scene_vectors: cell.associated_scene_vectors.clone(),
        associated_face_vectors: cell.associated_face_vectors.clone(),
        associated_object_vectors: cell.associated_object_vectors.clone(),
        associated_voice_vectors: cell.associated_voice_vectors.clone(),
        successful_actions: cell.successful_actions.clone(),
        failed_actions: cell.failed_actions.clone(),
    }
}

fn recognition_kind(
    current_key: Option<PlaceCellKey>,
    candidate_key: PlaceCellKey,
    similarity: f32,
) -> PlaceRecognitionKind {
    if current_key == Some(candidate_key) || similarity >= 0.92 {
        PlaceRecognitionKind::SamePlace
    } else {
        PlaceRecognitionKind::SimilarPlace
    }
}

fn candidate_reason(
    similarity: f32,
    confidence: f32,
    current_key: Option<PlaceCellKey>,
    candidate_key: PlaceCellKey,
) -> String {
    let kind = if current_key == Some(candidate_key) {
        "same map cell"
    } else if similarity >= 0.92 {
        "high latent similarity"
    } else {
        "moderate latent similarity"
    };
    format!("{kind}; similarity={similarity:.3}; confidence={confidence:.3}")
}

fn score_record(query: &RecallQuery, record: MemoryRecord) -> Option<(f32, MemoryRecord)> {
    let query_tokens = tokenize(&query.now_text);
    let summary_tokens = tokenize(&record.summary);
    let overlap = token_overlap(&query_tokens, &summary_tokens);
    let battery_distance = (query.battery - record.battery).abs();
    let battery_score = (1.0 - battery_distance).clamp(0.0, 1.0);
    let goal_score = if query.active_goal.is_some() && query.active_goal == record.active_goal {
        1.0
    } else {
        0.0
    };
    let action_score =
        if query.proposed_action.is_some() && query.proposed_action == record.chosen_action {
            0.8
        } else {
            0.0
        };
    let vector_score = max_vector_similarity(query_all_vectors(query), record_all_vectors(&record));
    let score = (overlap * 0.4)
        + (battery_score * 0.15)
        + (goal_score * 0.15)
        + (action_score * 0.1)
        + (vector_score * 0.2);
    if score <= 0.05 {
        return None;
    }
    Some((score, record))
}

fn graph_context_from_frame(
    frame: &ExperienceFrame,
    experiences: &[Experience],
    scene_vectors: &[VectorArtifact],
    face_vectors: &[VectorArtifact],
    object_vectors: &[VectorArtifact],
    voice_vectors: &[VectorArtifact],
) -> (Vec<GraphEntity>, Vec<GraphEdge>) {
    let frame_id = frame.id.to_string();
    let experience_id = format!("experience:{frame_id}");
    let mut entities = vec![
        GraphEntity {
            id: format!("frame:{frame_id}"),
            labels: vec!["Frame".to_string(), "Memory".to_string()],
            summary: format!("runtime frame {frame_id}"),
            score: 1.0,
        },
        GraphEntity {
            id: experience_id.clone(),
            labels: vec!["Experience".to_string(), "Memory".to_string()],
            summary: frame.summary_text(),
            score: 1.0,
        },
    ];
    let mut relationships = vec![graph_edge(
        format!("frame:{frame_id}"),
        experience_id.clone(),
        "HAS_MEMORY_SUMMARY",
        None,
    )];

    let pose = frame.now.body.odometry;
    let place_id = place_id_for_pose(pose);
    entities.push(GraphEntity {
        id: place_id.clone(),
        labels: vec!["Place".to_string()],
        summary: format!("place near x={:.1}m y={:.1}m", pose.x_m, pose.y_m),
        score: 1.0,
    });
    relationships.push(graph_edge(
        experience_id.clone(),
        place_id,
        "OCCURRED_AT",
        None,
    ));

    append_world_model_memory(
        &frame.now.world.social,
        &frame.now.world.temporal,
        &experience_id,
        &mut entities,
        &mut relationships,
    );
    append_semantic_graph_memory(&frame.now.world.semantic, &mut entities, &mut relationships);

    for artifact in scene_vectors {
        let vector_id = vector_node_id(artifact);
        entities.push(vector_entity(artifact, "scene"));
        relationships.push(graph_edge(
            experience_id.clone(),
            vector_id,
            "HAS_SCENE_VECTOR",
            None,
        ));
    }

    for sensation in &frame.sensations {
        let sensation_id = sensation_node_id(sensation.id);
        entities.push(GraphEntity {
            id: sensation_id.clone(),
            labels: vec![
                "Sensation".to_string(),
                sensation.modality.as_str().to_string(),
                sensation.payload_kind.as_str().to_string(),
            ],
            summary: sensation
                .summary
                .clone()
                .unwrap_or_else(|| sensation.kind.clone()),
            score: sensation.metadata.confidence.unwrap_or(1.0),
        });
        relationships.push(graph_edge(
            format!("frame:{frame_id}"),
            sensation_id.clone(),
            "HAS_SENSATION",
            Some(sensation.kind.clone()),
        ));
        if let Some(parent_id) = sensation.parent_id {
            relationships.push(graph_edge(
                sensation_id.clone(),
                sensation_node_id(parent_id),
                "DERIVED_FROM_SENSATION",
                None,
            ));
        }
        if let Some(embedding) = &sensation.vector {
            let artifact = embodied_vector_artifact(
                SENSATION_VECTOR_COLLECTION,
                &format!("{}:sensation:{}", frame.id, sensation.id),
                embedding,
                frame.id,
                sensation.id.to_string(),
                sensation.occurred_at_ms,
            );
            entities.push(vector_entity(&artifact, "sensation"));
            relationships.push(graph_edge(
                sensation_id.clone(),
                vector_node_id(&artifact),
                "HAS_SENSATION_VECTOR",
                Some(format!("{} dimensions", embedding.dim)),
            ));
        }
        if let Some(impression) = &sensation.impression {
            let impression_id = impression_node_id(impression.id);
            entities.push(impression_entity(impression));
            relationships.push(graph_edge(
                sensation_id,
                impression_id,
                "HAS_IMPRESSION",
                Some(impression.text.clone()),
            ));
        }
    }

    for impression in &frame.impressions {
        entities.push(impression_entity(impression));
        for sensation_id in &impression.about {
            relationships.push(graph_edge(
                sensation_node_id(*sensation_id),
                impression_node_id(impression.id),
                "HAS_IMPRESSION",
                Some(impression.text.clone()),
            ));
        }
    }

    for experience in experiences {
        let canonical_experience_id = experience_node_id(experience.id);
        entities.push(GraphEntity {
            id: canonical_experience_id.clone(),
            labels: vec!["Experience".to_string(), "EmbodiedExperience".to_string()],
            summary: experience_summary_text(experience),
            score: experience.salience,
        });
        relationships.push(graph_edge(
            format!("frame:{frame_id}"),
            canonical_experience_id.clone(),
            "HAS_EXPERIENCE",
            Some(experience.kind.clone()),
        ));
        relationships.push(graph_edge(
            experience_id.clone(),
            canonical_experience_id.clone(),
            "SUMMARIZES_EXPERIENCE",
            None,
        ));
        let (artifact, _, _, _) = experience_vector_artifact(frame, experience);
        entities.push(vector_entity(&artifact, "experience"));
        relationships.push(graph_edge(
            canonical_experience_id.clone(),
            vector_node_id(&artifact),
            "HAS_FUSED_VECTOR",
            Some(format!("{} dimensions", artifact.vector.len())),
        ));
        for sensation_id in &experience.sensation_ids {
            relationships.push(graph_edge(
                canonical_experience_id.clone(),
                sensation_node_id(*sensation_id),
                "INTEGRATES_SENSATION",
                None,
            ));
        }
        for impression_id in &experience.impression_ids {
            relationships.push(graph_edge(
                canonical_experience_id.clone(),
                impression_node_id(*impression_id),
                "INTEGRATES_IMPRESSION",
                None,
            ));
        }
        if let Some(impression) = &experience.summary_impression {
            entities.push(impression_entity(impression));
            relationships.push(graph_edge(
                canonical_experience_id.clone(),
                impression_node_id(impression.id),
                "HAS_SUMMARY_IMPRESSION",
                Some(impression.text.clone()),
            ));
        }
        for link in &experience.memory_links {
            if let Some(entity) = memory_link_entity(link) {
                entities.push(entity);
            }
            relationships.push(memory_link_edge(canonical_experience_id.clone(), link));
        }
    }

    for (index, artifact) in face_vectors.iter().enumerate() {
        let person_id = artifact
            .source_id
            .clone()
            .unwrap_or_else(|| format!("person:face:{frame_id}:{index}"));
        entities.push(GraphEntity {
            id: person_id.clone(),
            labels: vec![
                "Person".to_string(),
                "FaceInstance".to_string(),
                "Entity".to_string(),
            ],
            summary: "person seen by face vector".to_string(),
            score: 1.0,
        });
        entities.push(vector_entity(artifact, "face"));
        relationships.push(graph_edge(
            experience_id.clone(),
            person_id.clone(),
            "SAW_PERSON",
            None,
        ));
        relationships.push(graph_edge(
            person_id,
            vector_node_id(artifact),
            "HAS_FACE_VECTOR",
            None,
        ));
    }

    for (index, artifact) in object_vectors.iter().enumerate() {
        let object_id = artifact
            .source_id
            .clone()
            .unwrap_or_else(|| format!("object:vector:{frame_id}:{index}"));
        entities.push(GraphEntity {
            id: object_id.clone(),
            labels: vec![
                "Object".to_string(),
                "ObjectInstance".to_string(),
                "Entity".to_string(),
            ],
            summary: "object seen by visual vector".to_string(),
            score: 1.0,
        });
        entities.push(vector_entity(artifact, "object"));
        relationships.push(graph_edge(
            experience_id.clone(),
            object_id.clone(),
            "SAW_OBJECT",
            None,
        ));
        relationships.push(graph_edge(
            object_id,
            vector_node_id(artifact),
            "HAS_OBJECT_VECTOR",
            None,
        ));
    }

    for (index, artifact) in voice_vectors.iter().enumerate() {
        let person_id = artifact
            .source_id
            .clone()
            .unwrap_or_else(|| format!("person:voice:{frame_id}:{index}"));
        entities.push(GraphEntity {
            id: person_id.clone(),
            labels: vec![
                "Person".to_string(),
                "VoiceSignature".to_string(),
                "Entity".to_string(),
            ],
            summary: "person heard by voice vector".to_string(),
            score: 1.0,
        });
        entities.push(vector_entity(artifact, "voice"));
        relationships.push(graph_edge(
            experience_id.clone(),
            person_id.clone(),
            "HEARD_PERSON",
            None,
        ));
        relationships.push(graph_edge(
            person_id,
            vector_node_id(artifact),
            "HAS_VOICE_VECTOR",
            None,
        ));
    }

    (
        dedupe_entities(entities, usize::MAX),
        dedupe_relationships(relationships, usize::MAX),
    )
}

fn append_semantic_graph_memory(
    semantic: &SemanticGraphSnapshot,
    entities: &mut Vec<GraphEntity>,
    relationships: &mut Vec<GraphEdge>,
) {
    for relation in semantic.relations.values() {
        for node in [&relation.subject, &relation.object] {
            entities.push(GraphEntity {
                id: node.stable_key(),
                labels: semantic_node_labels(node),
                summary: node.stable_key(),
                score: relation.confidence,
            });
        }
        relationships.push(GraphEdge {
            id: Some(relation.id.0.clone()),
            from: relation.subject.stable_key(),
            to: relation.object.stable_key(),
            relationship: format!("SEMANTIC_{:?}", relation.predicate).to_ascii_uppercase(),
            summary: Some(format!(
                "grounded semantic relation {:?} ({:?})",
                relation.predicate, relation.status
            )),
            score: relation.confidence,
            payload: serde_json::to_value(relation).unwrap_or(serde_json::Value::Null),
        });
    }
}

fn semantic_node_labels(node: &SemanticNodeRef) -> Vec<String> {
    let kind = match node {
        SemanticNodeRef::Entity(_) => "Entity",
        SemanticNodeRef::Place(_) => "Place",
        SemanticNodeRef::Person(_) => "Person",
        SemanticNodeRef::Action(_) => "Action",
        SemanticNodeRef::Skill(_) => "Skill",
        SemanticNodeRef::Behavior(_) => "Behavior",
        SemanticNodeRef::Goal(_) => "Goal",
        SemanticNodeRef::Drive(_) => "Drive",
        SemanticNodeRef::Outcome(_) => "Outcome",
        SemanticNodeRef::Property(_) => "Property",
        SemanticNodeRef::Concept(_) => "Concept",
        SemanticNodeRef::Episode(_) => "Episode",
    };
    vec!["SemanticNode".to_string(), kind.to_string()]
}

fn append_world_model_memory(
    social: &SocialWorldSnapshot,
    temporal: &TemporalContext,
    experience_id: &str,
    entities: &mut Vec<GraphEntity>,
    relationships: &mut Vec<GraphEdge>,
) {
    for person in social.people.values() {
        let name = person
            .preferred_name
            .as_ref()
            .map(|name| name.value.as_str())
            .or_else(|| {
                person
                    .best_identity()
                    .and_then(|identity| identity.display_name.as_deref())
            })
            .unwrap_or("unknown person");
        entities.push(GraphEntity {
            id: person.person_id.0.clone(),
            labels: vec![
                "Person".to_string(),
                "SocialPersonBelief".to_string(),
                "Entity".to_string(),
            ],
            summary: format!("social belief about {name}"),
            score: person.current_identity_confidence,
        });
        relationships.push(GraphEdge {
            id: None,
            from: experience_id.to_string(),
            to: person.person_id.0.clone(),
            relationship: if person.presence.present {
                "INTERACTED_WITH_OR_OBSERVED".to_string()
            } else {
                "REMEMBERS_PERSON".to_string()
            },
            summary: Some(format!(
                "presence={} identity_confidence={:.3}",
                person.presence.present, person.current_identity_confidence
            )),
            score: person
                .presence
                .confidence
                .max(person.current_identity_confidence),
            payload: json!({
                "present": person.presence.present,
                "presence_freshness": person.presence.freshness,
                "last_seen_at_ms": person.presence.last_seen_at_ms,
                "identity_hypotheses": person.identity_hypotheses,
            }),
        });
    }

    for relationship in social.relationships.values() {
        let relationship_id = format!("relationship:{}", relationship.relationship_id.0);
        let confidence = relationship
            .relationship_kinds
            .iter()
            .map(|kind| kind.confidence)
            .fold(0.0f32, f32::max);
        entities.push(GraphEntity {
            id: relationship_id.clone(),
            labels: vec!["Relationship".to_string(), "SocialBelief".to_string()],
            summary: format!("relationship belief for {}", relationship.person_id.0),
            score: confidence,
        });
        relationships.push(GraphEdge {
            id: None,
            from: relationship_id,
            to: relationship.person_id.0.clone(),
            relationship: "RELATES_TO_PERSON".to_string(),
            summary: Some(format!(
                "trust={:.3} affiliation={:.3}",
                relationship.trust, relationship.affiliation
            )),
            score: confidence,
            payload: json!({
                "relationship_kinds": relationship.relationship_kinds,
                "trust": relationship.trust,
                "affiliation": relationship.affiliation,
                "authority": relationship.caregiving_or_authority,
            }),
        });
    }

    for interaction in social
        .recent_interactions
        .iter()
        .chain(social.active_interaction.iter())
    {
        let interaction_id = interaction.interaction_id.0.clone();
        entities.push(GraphEntity {
            id: interaction_id.clone(),
            labels: vec!["Interaction".to_string(), "Episode".to_string()],
            summary: format!("social interaction in phase {:?}", interaction.phase),
            score: 1.0,
        });
        relationships.push(GraphEdge {
            id: None,
            from: experience_id.to_string(),
            to: interaction_id.clone(),
            relationship: "HAS_SOCIAL_INTERACTION".to_string(),
            summary: None,
            score: 1.0,
            payload: json!({
                "started_at_ms": interaction.started_at_ms,
                "last_activity_ms": interaction.last_activity_ms,
                "ended_at_ms": interaction.ended_at_ms,
                "phase": interaction.phase,
            }),
        });
        for participant in &interaction.participants {
            relationships.push(graph_edge(
                interaction_id.clone(),
                participant.0.clone(),
                "HAS_PARTICIPANT",
                None,
            ));
        }
    }

    for episode in temporal
        .recently_completed
        .iter()
        .chain(temporal.active_episodes.iter())
    {
        let episode_id = episode.episode_id.0.clone();
        entities.push(GraphEntity {
            id: episode_id.clone(),
            labels: vec!["Episode".to_string(), format!("{:?}", episode.kind)],
            summary: format!("{:?} episode", episode.kind),
            score: episode.confidence,
        });
        relationships.push(GraphEdge {
            id: None,
            from: experience_id.to_string(),
            to: episode_id.clone(),
            relationship: "HAS_TEMPORAL_EPISODE".to_string(),
            summary: episode.closure_reason.map(|reason| format!("{reason:?}")),
            score: episode.confidence,
            payload: json!({
                "clock_domain": episode.interval.domain,
                "start_ms": episode.interval.start_ms,
                "end_ms": episode.interval.end_ms,
                "active_goals": episode.active_goals,
                "significant_events": episode.significant_events,
            }),
        });
        for participant in &episode.participants {
            relationships.push(graph_edge(
                episode_id.clone(),
                participant.0.clone(),
                "HAS_PARTICIPANT",
                None,
            ));
        }
        for preceding in &episode.preceding_episode_refs {
            relationships.push(graph_edge(
                episode_id.clone(),
                preceding.0.clone(),
                "FOLLOWS_EPISODE",
                None,
            ));
        }
    }
}

fn place_id_for_pose(pose: Pose2) -> String {
    format!(
        "place:grid:{:.0}:{:.0}",
        (pose.x_m * 2.0).round(),
        (pose.y_m * 2.0).round()
    )
}

fn graph_edge(
    from: impl Into<String>,
    to: impl Into<String>,
    relationship: impl Into<String>,
    summary: Option<String>,
) -> GraphEdge {
    GraphEdge {
        id: None,
        from: from.into(),
        to: to.into(),
        relationship: relationship.into(),
        summary,
        score: 1.0,
        payload: serde_json::Value::Null,
    }
}

fn memory_link_edge(from: impl Into<String>, link: &MemoryLink) -> GraphEdge {
    GraphEdge {
        id: None,
        from: from.into(),
        to: link.target_id.clone(),
        relationship: link.relation.clone(),
        summary: link
            .payload
            .get("text")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        score: link.score,
        payload: link.payload.clone(),
    }
}

fn memory_link_entity(link: &MemoryLink) -> Option<GraphEntity> {
    let target_kind = link
        .payload
        .get("target_kind")
        .and_then(|value| value.as_str())?;
    let text = link
        .payload
        .get("text")
        .and_then(|value| value.as_str())
        .unwrap_or(&link.target_id)
        .to_string();
    let labels = match target_kind {
        "place" => vec!["Place".to_string(), "Entity".to_string()],
        "object" => {
            let mut labels = vec!["Object".to_string(), "Entity".to_string()];
            if let Some(class) = link.payload.get("class").and_then(|value| value.as_str()) {
                labels.push(class.to_string());
            }
            labels
        }
        "person" => vec!["Person".to_string(), "Entity".to_string()],
        "surface" => vec!["Surface".to_string(), "Entity".to_string()],
        "experience" => vec!["Experience".to_string(), "Memory".to_string()],
        _ => return None,
    };
    Some(GraphEntity {
        id: link.target_id.clone(),
        labels,
        summary: text,
        score: link.score,
    })
}

fn vector_node_id(artifact: &VectorArtifact) -> String {
    format!("vector:{}:{}", artifact.collection, artifact.point_id)
}

fn sensation_node_id(id: uuid::Uuid) -> String {
    format!("sensation:{id}")
}

fn impression_node_id(id: uuid::Uuid) -> String {
    format!("impression:{id}")
}

fn experience_node_id(id: uuid::Uuid) -> String {
    format!("embodied_experience:{id}")
}

fn impression_entity(impression: &Impression) -> GraphEntity {
    GraphEntity {
        id: impression_node_id(impression.id),
        labels: vec!["Impression".to_string(), impression.kind.clone()],
        summary: impression.text.clone(),
        score: impression.confidence,
    }
}

fn experience_summary_text(experience: &Experience) -> String {
    experience
        .summary_impression
        .as_ref()
        .map(|impression| impression.text.clone())
        .unwrap_or_else(|| experience.text.clone())
}

fn vector_entity(artifact: &VectorArtifact, kind: &str) -> GraphEntity {
    GraphEntity {
        id: vector_node_id(artifact),
        labels: vec!["Vector".to_string()],
        summary: format!(
            "{kind} vector in {} with {} dimensions",
            artifact.collection,
            artifact.vector.len()
        ),
        score: 1.0,
    }
}

fn scored_entities(record: &MemoryRecord, score: f32) -> Vec<GraphEntity> {
    record
        .graph_entities
        .iter()
        .filter(|entity| !entity.has_label("Vector"))
        .map(|entity| {
            let mut entity = entity.clone();
            entity.score = score;
            entity
        })
        .collect()
}

fn dedupe_entities(entities: Vec<GraphEntity>, limit: usize) -> Vec<GraphEntity> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for entity in entities {
        if seen.insert(entity.id.clone()) {
            out.push(entity);
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}

fn dedupe_relationships(relationships: Vec<GraphEdge>, limit: usize) -> Vec<GraphEdge> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for relationship in relationships {
        let key = graph_edge_id(&relationship);
        if seen.insert(key) {
            out.push(relationship);
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}

fn graph_edge_id(edge: &GraphEdge) -> String {
    edge.id
        .as_deref()
        .filter(|id| !id.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| {
            // Character-length-prefix the structural fallback so identifiers
            // containing separators cannot alias another relationship triple,
            // and Neo4j can reproduce the same id with Cypher `size()`.
            format!(
                "graph-edge:{}:{}:{}:{}:{}:{}",
                edge.from.chars().count(),
                edge.from,
                edge.relationship.chars().count(),
                edge.relationship,
                edge.to.chars().count(),
                edge.to
            )
        })
}

fn graph_context_summary(entities: &[GraphEntity]) -> Option<String> {
    let people = entities
        .iter()
        .filter(|entity| entity.has_label("Person"))
        .count();
    let places = entities
        .iter()
        .filter(|entity| entity.has_label("Place"))
        .count();
    let experiences = entities
        .iter()
        .filter(|entity| entity.has_label("Experience"))
        .count();
    if people == 0 && places == 0 && experiences == 0 {
        return None;
    }
    Some(format!(
        "Graph recall: {people} person nodes, {places} place nodes, {experiences} experience nodes."
    ))
}

fn scene_vectors_from_now(now: &Now, frame_id: uuid::Uuid, t_ms: u64) -> Vec<VectorArtifact> {
    if !now.eye.scene_vectors.is_empty() {
        return now.eye.scene_vectors.clone();
    }
    now.eye
        .frames
        .last()
        .map(|vector| {
            VectorArtifact::new(
                SCENE_VECTOR_COLLECTION,
                format!(
                    "{}:scene:{}",
                    frame_id,
                    now.eye.frames.len().saturating_sub(1)
                ),
                vector.clone(),
            )
            .with_source_frame_id(frame_id.to_string())
            .with_occurred_at_ms(t_ms)
        })
        .into_iter()
        .collect()
}

fn sensation_vectors_from_frame(
    frame: &ExperienceFrame,
) -> (Vec<VectorArtifact>, BTreeMap<String, serde_json::Value>) {
    let mut payloads = BTreeMap::new();
    let artifacts = frame
        .sensations
        .iter()
        .filter_map(|sensation| {
            let embedding = sensation.vector.as_ref()?;
            let artifact = embodied_vector_artifact(
                SENSATION_VECTOR_COLLECTION,
                &format!("{}:sensation:{}", frame.id, sensation.id),
                embedding,
                frame.id,
                sensation.id.to_string(),
                sensation.occurred_at_ms,
            );
            payloads.insert(
                vector_payload_key(&artifact),
                json!({
                    "payload_kind": sensation.payload_kind.as_str(),
                    "modality": sensation.modality.as_str(),
                    "sensation_id": sensation.id.to_string(),
                    "parent_sensation_id": sensation.parent_id.map(|id| id.to_string()),
                    "source_sensation_id": embedding.source_sensation_id.to_string(),
                    "model_id": embedding.model_id,
                    "dim": embedding.dim,
                    "observed_at_ms": sensation.observed_at_ms,
                    "occurred_at_ms": sensation.occurred_at_ms,
                    "generated_at_ms": embedding.generated_at_ms,
                    "sensation_kind": sensation.kind,
                    "source": sensation.source,
                    "summary": sensation.summary,
                    "labels": sensation.metadata.labels,
                    "confidence": sensation.metadata.confidence,
                    "provenance": sensation.provenance,
                }),
            );
            Some(artifact)
        })
        .collect();
    (artifacts, payloads)
}

fn experience_vectors_from_frame(
    frame: &ExperienceFrame,
    payloads: &mut BTreeMap<String, serde_json::Value>,
) -> Vec<VectorArtifact> {
    frame
        .experiences
        .iter()
        .map(|experience| {
            let summary_impression = experience.summary_impression.as_ref();
            let (artifact, model_id, generated_at_ms, summary_text) =
                experience_vector_artifact(frame, experience);
            payloads.insert(
                vector_payload_key(&artifact),
                json!({
                    "experience_id": experience.id.to_string(),
                    "experience_kind": experience.kind,
                    "summary": experience.text,
                    "summary_impression_id": summary_impression.map(|impression| impression.id.to_string()),
                    "summary_impression_text": summary_text,
                    "impression_ids": experience.impression_ids.iter().map(ToString::to_string).collect::<Vec<_>>(),
                    "sensation_ids": experience.sensation_ids.iter().map(ToString::to_string).collect::<Vec<_>>(),
                    "model_id": model_id,
                    "dim": artifact.vector.len(),
                    "observed_at_ms": experience.observed_at_ms,
                    "occurred_at_ms": experience.occurred_at_ms,
                    "generated_at_ms": generated_at_ms,
                }),
            );
            artifact
        })
        .collect()
}

fn deterministic_text_vector(text: &str, dim: usize) -> Vec<f32> {
    let dim = dim.max(1);
    let mut vector = vec![0.0; dim];
    for token in text.split_whitespace() {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        token.to_ascii_lowercase().hash(&mut hasher);
        let hash = hasher.finish();
        let index = hash as usize % dim;
        let sign = if (hash >> 63) == 0 { 1.0 } else { -1.0 };
        vector[index] += sign;
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

fn experience_vector_artifact(
    frame: &ExperienceFrame,
    experience: &Experience,
) -> (VectorArtifact, String, u64, String) {
    let summary_impression = experience.summary_impression.as_ref();
    let summary_text = summary_impression
        .map(|impression| impression.text.clone())
        .unwrap_or_else(|| experience.text.clone());
    let (vector, model_id, generated_at_ms) = if let Some(latent) = &frame.z {
        (
            latent.z.clone(),
            "pete.experience.latent".to_string(),
            latent.t_ms,
        )
    } else if let Some(embedding) =
        summary_impression.and_then(|impression| impression.vector.as_ref())
    {
        (
            embedding.vector.clone(),
            embedding.model_id.clone(),
            embedding.generated_at_ms,
        )
    } else {
        (
            deterministic_text_vector(&summary_text, 16),
            "pete.text.hashing.v1".to_string(),
            frame.t_ms,
        )
    };
    let artifact = VectorArtifact::new(
        EXPERIENCE_VECTOR_COLLECTION,
        format!("{}:experience:{}", frame.id, experience.id),
        vector,
    )
    .with_model(model_id.clone())
    .with_source_id(experience.id.to_string())
    .with_source_frame_id(frame.id.to_string())
    .with_occurred_at_ms(experience.occurred_at_ms);
    (artifact, model_id, generated_at_ms, summary_text)
}

fn memory_links_from_frame(
    frame: &ExperienceFrame,
    _scene_vectors: &[VectorArtifact],
    face_vectors: &[VectorArtifact],
    object_vectors: &[VectorArtifact],
    voice_vectors: &[VectorArtifact],
) -> Vec<MemoryLink> {
    let mut links = Vec::new();
    let pose = frame.now.body.odometry;
    let place_score = if frame.now.memory.places_visited > 0 {
        frame.now.memory.place_familiarity.max(0.75)
    } else {
        frame.now.memory.place_familiarity.max(0.5)
    };
    links.push(MemoryLink {
        target_id: place_id_for_pose(pose),
        relation: "occurred_at_place".to_string(),
        score: place_score.clamp(0.0, 1.0),
        payload: json!({
            "target_kind": "place",
            "text": format!("place near x={:.1}m y={:.1}m", pose.x_m, pose.y_m),
            "x_m": pose.x_m,
            "y_m": pose.y_m,
            "heading_rad": pose.heading_rad,
        }),
    });

    for observation in &frame.now.objects.observations {
        let target_id = object_observation_id(observation);
        links.push(MemoryLink {
            target_id,
            relation: "observed_object".to_string(),
            score: observation.confidence.clamp(0.0, 1.0),
            payload: json!({
                "target_kind": "object",
                "text": observation.label,
                "label": observation.label,
                "class": object_class_slug(&observation.class),
                "bearing_rad": observation.bearing_rad,
                "distance_m": observation.distance_m,
                "source": object_source_slug(&observation.source),
            }),
        });
    }

    for (index, artifact) in face_vectors.iter().enumerate() {
        let target_id = artifact
            .source_id
            .clone()
            .unwrap_or_else(|| format!("person:face:{}:{index}", frame.id));
        links.push(MemoryLink {
            target_id,
            relation: "saw_face".to_string(),
            score: 1.0,
            payload: json!({
                "target_kind": "person",
                "text": "face observed",
                "vector_id": vector_node_id(artifact),
                "collection": artifact.collection,
                "point_id": artifact.point_id,
            }),
        });
    }

    for (index, artifact) in object_vectors.iter().enumerate() {
        let target_id = artifact
            .source_id
            .clone()
            .unwrap_or_else(|| format!("object:vector:{}:{index}", frame.id));
        links.push(MemoryLink {
            target_id,
            relation: "saw_object_vector".to_string(),
            score: 1.0,
            payload: json!({
                "target_kind": "object",
                "text": "object visual vector observed",
                "vector_id": vector_node_id(artifact),
                "collection": artifact.collection,
                "point_id": artifact.point_id,
            }),
        });
    }

    for (index, artifact) in voice_vectors.iter().enumerate() {
        let target_id = artifact
            .source_id
            .clone()
            .unwrap_or_else(|| format!("person:voice:{}:{index}", frame.id));
        links.push(MemoryLink {
            target_id,
            relation: "heard_voice".to_string(),
            score: 1.0,
            payload: json!({
                "target_kind": "person",
                "text": "voice observed",
                "vector_id": vector_node_id(artifact),
                "collection": artifact.collection,
                "point_id": artifact.point_id,
            }),
        });
    }

    if let Some(surface_graph) = frame.now.extensions.get("surface.scene_graph") {
        links.extend(surface_memory_links(surface_graph));
    }

    links.extend(frame.recollections.iter().map(|recollection| MemoryLink {
        target_id: experience_node_id(recollection.experience.id),
        relation: "similar_to_experience".to_string(),
        score: recollection.score.clamp(0.0, 1.0),
        payload: json!({
            "target_kind": "experience",
            "text": recollection.experience.text,
            "original_frame_id": recollection.original_frame_id.map(|id| id.to_string()),
            "original_vector_ids": recollection.original_vector_ids,
        }),
    }));

    dedupe_memory_links(links)
}

fn surface_memory_links(surface_graph: &serde_json::Value) -> Vec<MemoryLink> {
    let mut links = Vec::new();
    if let Some(floor) = surface_graph.get("floor") {
        if let Some(link) = surface_link_from_value(floor, "near_surface") {
            links.push(link);
        }
    }
    if let Some(surfaces) = surface_graph
        .get("surfaces")
        .and_then(|value| value.as_array())
    {
        links.extend(
            surfaces
                .iter()
                .filter_map(|surface| surface_link_from_value(surface, "near_surface")),
        );
    }
    if let Some(clusters) = surface_graph
        .get("clusters")
        .and_then(|value| value.as_array())
    {
        links.extend(
            clusters
                .iter()
                .filter_map(|cluster| surface_link_from_value(cluster, "observed_surface_cluster")),
        );
    }
    links
}

fn surface_link_from_value(value: &serde_json::Value, relation: &str) -> Option<MemoryLink> {
    let id = value.get("id").and_then(|id| id.as_str())?;
    let confidence = value
        .get("confidence")
        .and_then(|confidence| confidence.as_f64())
        .unwrap_or(1.0) as f32;
    Some(MemoryLink {
        target_id: format!("surface:{id}"),
        relation: relation.to_string(),
        score: confidence.clamp(0.0, 1.0),
        payload: json!({
            "target_kind": "surface",
            "text": format!("surface {id}"),
            "surface_id": id,
            "kind": value.get("kind").cloned(),
        }),
    })
}

fn merge_memory_links(existing: &mut Vec<MemoryLink>, incoming: Vec<MemoryLink>) {
    let mut seen = existing
        .iter()
        .map(memory_link_key)
        .collect::<BTreeSet<_>>();
    for link in incoming {
        if seen.insert(memory_link_key(&link)) {
            existing.push(link);
        }
    }
}

fn dedupe_memory_links(links: Vec<MemoryLink>) -> Vec<MemoryLink> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for link in links {
        if seen.insert(memory_link_key(&link)) {
            out.push(link);
        }
    }
    out
}

fn memory_link_key(link: &MemoryLink) -> String {
    format!("{}:{}", link.relation, link.target_id)
}

fn object_observation_id(observation: &pete_now::ObjectObservation) -> String {
    format!(
        "object:{}:{}:{}",
        object_source_slug(&observation.source),
        object_class_slug(&observation.class),
        stable_slug(&observation.label)
    )
}

fn object_class_slug(class: &pete_now::ObjectClass) -> &'static str {
    match class {
        pete_now::ObjectClass::Obstacle => "obstacle",
        pete_now::ObjectClass::Charger => "charger",
        pete_now::ObjectClass::Person => "person",
        pete_now::ObjectClass::SoundSource => "sound_source",
        pete_now::ObjectClass::Landmark => "landmark",
        pete_now::ObjectClass::Unknown => "unknown",
    }
}

fn object_source_slug(source: &pete_now::ObjectObservationSource) -> &'static str {
    match source {
        pete_now::ObjectObservationSource::Sim => "sim",
        pete_now::ObjectObservationSource::Kinect => "kinect",
        pete_now::ObjectObservationSource::Captioner => "captioner",
        pete_now::ObjectObservationSource::CreateIr => "create_ir",
        pete_now::ObjectObservationSource::HumanLabel => "human_label",
        pete_now::ObjectObservationSource::Unknown => "unknown",
    }
}

fn stable_slug(value: &str) -> String {
    let mut slug = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    slug.trim_matches('-').to_string()
}

fn embodied_vector_artifact(
    collection: &str,
    point_id: &str,
    embedding: &VectorEmbedding,
    frame_id: uuid::Uuid,
    source_id: String,
    occurred_at_ms: u64,
) -> VectorArtifact {
    VectorArtifact::new(collection, point_id, embedding.vector.clone())
        .with_model(embedding.model_id.clone())
        .with_source_id(source_id)
        .with_source_frame_id(frame_id.to_string())
        .with_occurred_at_ms(occurred_at_ms)
}

fn query_scene_vectors(query: &RecallQuery) -> Vec<&[f32]> {
    let mut vectors = query
        .scene_vectors
        .iter()
        .map(|artifact| artifact.vector.as_slice())
        .collect::<Vec<_>>();
    if let Some(vector) = &query.scene_vector {
        vectors.push(vector.as_slice());
    }
    vectors
}

fn place_query_vectors_from_query(query: &RecallQuery) -> Vec<VectorArtifact> {
    let mut vectors = query.scene_vectors.clone();
    if let Some(vector) = &query.scene_vector {
        vectors.push(VectorArtifact::new(
            SCENE_VECTOR_COLLECTION,
            "query:legacy-scene-vector",
            vector.clone(),
        ));
    }
    if let Some(input) = &query.place_recognition_input {
        vectors.extend(place_recognition_vectors_from_input(input));
    }
    vectors
}

pub fn place_recognition_input_from_frame(frame: &ExperienceFrame) -> PlaceRecognitionInput {
    let instant = frame.experience_instant();
    let experience_id = instant.experience_id.map(|id| id.to_string());
    let experience_latent_vector = frame.z.as_ref().map(|latent| {
        let artifact = VectorArtifact::new(
            EXPERIENCE_VECTOR_COLLECTION,
            format!("{}:experience-latent", frame.id),
            latent.z.clone(),
        )
        .with_model("pete.experience.latent")
        .with_source_frame_id(frame.id.to_string())
        .with_occurred_at_ms(frame.t_ms);
        if let Some(experience_id) = &experience_id {
            artifact.with_source_id(experience_id.clone())
        } else {
            artifact
        }
    });
    PlaceRecognitionInput {
        experience_id,
        instant_frame_id: Some(frame.id.to_string()),
        experience_latent_vector,
        teacher_vector_refs: instant
            .teacher_vectors
            .iter()
            .map(|vector| embodied_vector_ref_id(&vector.metadata))
            .collect(),
        compact_range_summary: compact_range_summary(&frame.now),
        compact_depth_summary: compact_depth_summary(&frame.now),
        object_labels: object_labels(&frame.now, None),
        person_labels: object_labels(&frame.now, Some(ObjectClass::Person)),
        voice_labels: voice_labels(&frame.now),
        action: frame.chosen_action.clone(),
        pose: Some(frame.now.body.odometry),
        window_start_ms: instant.window_start_ms,
        window_end_ms: instant.window_end_ms,
        provenance: format!(
            "{}:{}",
            instant.provenance.source,
            instant
                .provenance
                .source_frame_id
                .as_deref()
                .unwrap_or("unknown-frame")
        ),
    }
}

pub fn place_recognition_input_from_query_now(
    now: &Now,
    latent: Option<&pete_experience::ExperienceLatent>,
    provenance: impl Into<String>,
) -> PlaceRecognitionInput {
    let experience_latent_vector = latent.map(|latent| {
        VectorArtifact::new(
            EXPERIENCE_VECTOR_COLLECTION,
            format!("query:{}:experience-latent", now.t_ms),
            latent.z.clone(),
        )
        .with_model("pete.experience.latent")
        .with_occurred_at_ms(now.t_ms)
    });
    PlaceRecognitionInput {
        experience_id: None,
        instant_frame_id: None,
        experience_latent_vector,
        teacher_vector_refs: now
            .eye
            .scene_vectors
            .iter()
            .chain(now.face.vectors.iter())
            .chain(now.objects.vectors.iter())
            .chain(now.voice.vectors.iter())
            .map(|artifact| format!("{}:{}", artifact.collection, artifact.point_id))
            .collect(),
        compact_range_summary: compact_range_summary(now),
        compact_depth_summary: compact_depth_summary(now),
        object_labels: object_labels(now, None),
        person_labels: object_labels(now, Some(ObjectClass::Person)),
        voice_labels: voice_labels(now),
        action: now.memory.best_remembered_action.clone(),
        pose: Some(now.body.odometry),
        window_start_ms: now.t_ms,
        window_end_ms: now.t_ms,
        provenance: provenance.into(),
    }
}

pub fn place_recognition_vectors_from_input(input: &PlaceRecognitionInput) -> Vec<VectorArtifact> {
    input
        .experience_latent_vector
        .iter()
        .cloned()
        .collect::<Vec<_>>()
}

fn compact_range_summary(now: &Now) -> Option<CompactRangeSummary> {
    let finite = now
        .range
        .beams
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() && now.range.nearest_m.is_none() {
        return None;
    }
    let mean_m = (!finite.is_empty()).then(|| finite.iter().sum::<f32>() / finite.len() as f32);
    Some(CompactRangeSummary {
        beam_count: now.range.beams.len(),
        nearest_m: now.range.nearest_m,
        mean_m,
    })
}

fn compact_depth_summary(now: &Now) -> Option<CompactDepthSummary> {
    let finite = now
        .kinect
        .depth_m
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return None;
    }
    let min_m = finite.iter().copied().reduce(f32::min);
    let max_m = finite.iter().copied().reduce(f32::max);
    let mean_m = Some(finite.iter().sum::<f32>() / finite.len() as f32);
    Some(CompactDepthSummary {
        sample_count: finite.len(),
        min_m,
        max_m,
        mean_m,
    })
}

fn object_labels(now: &Now, class: Option<ObjectClass>) -> Vec<String> {
    let mut labels = now
        .objects
        .observations
        .iter()
        .filter(|observation| {
            class
                .as_ref()
                .map(|class| observation.class == *class)
                .unwrap_or(true)
        })
        .map(|observation| observation.label.clone())
        .collect::<Vec<_>>();
    labels.sort();
    labels.dedup();
    labels
}

fn voice_labels(now: &Now) -> Vec<String> {
    now.ear
        .transcript
        .as_ref()
        .into_iter()
        .chain(now.ear.asr.transcript.as_ref())
        .map(|text| text.trim())
        .filter(|text| !text.is_empty())
        .map(str::to_string)
        .collect()
}

fn embodied_vector_ref_id(vector: &pete_experience::EmbodiedVectorRef) -> String {
    format!(
        "{}:{}:{}:{}",
        vector.collection, vector.model_id, vector.source_sensation_id, vector.dim
    )
}

fn query_face_vectors(query: &RecallQuery) -> Vec<&[f32]> {
    query
        .face_vectors
        .iter()
        .map(|artifact| artifact.vector.as_slice())
        .collect()
}

fn query_object_vectors(query: &RecallQuery) -> Vec<&[f32]> {
    query
        .object_vectors
        .iter()
        .map(|artifact| artifact.vector.as_slice())
        .collect()
}

fn query_voice_vectors(query: &RecallQuery) -> Vec<&[f32]> {
    query
        .voice_vectors
        .iter()
        .map(|artifact| artifact.vector.as_slice())
        .collect()
}

fn recall_vector_ids(record: &MemoryRecord) -> Vec<String> {
    let mut ids = record
        .experience_vectors
        .iter()
        .chain(record.sensation_vectors.iter())
        .chain(record.scene_vectors.iter())
        .chain(record.face_vectors.iter())
        .chain(record.object_vectors.iter())
        .chain(record.voice_vectors.iter())
        .map(|artifact| format!("{}:{}", artifact.collection, artifact.point_id))
        .collect::<Vec<_>>();
    ids.sort();
    ids.dedup();
    ids
}

fn query_all_vectors(query: &RecallQuery) -> Vec<&[f32]> {
    let mut vectors = query_scene_vectors(query);
    vectors.extend(query_face_vectors(query));
    vectors.extend(query_object_vectors(query));
    vectors.extend(query_voice_vectors(query));
    vectors
}

fn record_all_vectors(record: &MemoryRecord) -> Vec<&VectorArtifact> {
    record
        .scene_vectors
        .iter()
        .chain(record.face_vectors.iter())
        .chain(record.object_vectors.iter())
        .chain(record.voice_vectors.iter())
        .chain(record.sensation_vectors.iter())
        .chain(record.experience_vectors.iter())
        .collect()
}

fn vector_payload_key(artifact: &VectorArtifact) -> String {
    format!("{}:{}", artifact.collection, artifact.point_id)
}

fn merge_json_object(base: &mut serde_json::Value, extra: &serde_json::Value) {
    let (Some(base), Some(extra)) = (base.as_object_mut(), extra.as_object()) else {
        return;
    };
    for (key, value) in extra {
        base.insert(key.clone(), value.clone());
    }
}

fn has_face_query(query: &RecallQuery) -> bool {
    !query.face_vectors.is_empty()
}

fn has_object_query(query: &RecallQuery) -> bool {
    !query.object_vectors.is_empty()
}

fn has_voice_query(query: &RecallQuery) -> bool {
    !query.voice_vectors.is_empty()
}

fn max_vector_similarity(query_vectors: Vec<&[f32]>, record_vectors: Vec<&VectorArtifact>) -> f32 {
    query_vectors
        .into_iter()
        .flat_map(|query| {
            record_vectors
                .iter()
                .map(move |record| cosine_similarity(query, &record.vector))
        })
        .fold(0.0f32, f32::max)
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.is_empty() || right.is_empty() || left.len() != right.len() {
        return 0.0;
    }
    let (mut dot, mut left_norm, mut right_norm) = (0.0f32, 0.0f32, 0.0f32);
    for (left, right) in left.iter().zip(right) {
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm <= f32::EPSILON || right_norm <= f32::EPSILON {
        return 0.0;
    }
    (dot / (left_norm.sqrt() * right_norm.sqrt())).clamp(0.0, 1.0)
}

fn tokenize(text: &str) -> BTreeSet<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect()
}

fn token_overlap(left: &BTreeSet<String>, right: &BTreeSet<String>) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let shared = left.intersection(right).count() as f32;
    let total = left.union(right).count() as f32;
    if total <= f32::EPSILON {
        0.0
    } else {
        shared / total
    }
}

fn query_pose_time_hint(query: &RecallQuery, ordinal: u64) -> u64 {
    let pose_hint = query
        .pose
        .map(|pose| ((pose.x_m.abs() + pose.y_m.abs()) * 100.0) as u64)
        .unwrap_or(0);
    pose_hint.saturating_add(ordinal)
}

fn stable_qdrant_point_id(collection: &str, point_id: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    collection.hash(&mut hasher);
    point_id.hash(&mut hasher);
    hasher.finish()
}

fn neo4j_http_url_from_uri(uri: &str) -> Option<String> {
    let trimmed = uri.trim();
    let rest = trimmed
        .strip_prefix("bolt://")
        .or_else(|| trimmed.strip_prefix("neo4j://"))?;
    let host = rest.split('/').next().unwrap_or(rest);
    let host_without_port = host.split(':').next().unwrap_or(host);
    if host_without_port.is_empty() {
        return None;
    }
    Some(format!("http://{host_without_port}:7474"))
}

#[cfg(test)]
mod cognitive_diagnostics_tests {
    use super::*;
    use pete_core::{FeatureModality, FeatureType, Provenance, VectorRef};

    fn candidate(
        left: &str,
        right: &str,
        decision: BindingDecision,
        evidence: Vec<BindingEvidence>,
    ) -> BindingCandidate {
        BindingCandidate {
            left_cluster_id: left.to_string(),
            right_cluster_id: right.to_string(),
            relation: BindingRelation::LikelySameEntity,
            confidence: match decision {
                BindingDecision::Accept => 0.88,
                BindingDecision::Reject => 0.2,
                _ => 0.48,
            },
            decision,
            reason: "test decision reason".to_string(),
            evidence,
        }
    }

    fn evidence(kind: BindingEvidenceKind, reason: &str) -> BindingEvidence {
        BindingEvidence {
            kind,
            score: 0.7,
            reason: reason.to_string(),
        }
    }

    fn diagnostic_document() -> GraphIntelligenceDocument {
        let mut feature = Feature::new(
            FeatureType::FaceObservation,
            FeatureModality::Vision,
            100,
            0.82,
            Provenance::direct().with_stage("test"),
        )
        .with_source_frame("frame-a")
        .with_vector_ref(VectorRef::new("faces", "face-vector-a"))
        .with_metadata(json!({
            "raw_vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "caption": "face candidate"
        }));
        let feature_id = feature.id;
        feature.metadata["large_text"] = json!("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz");

        let accepted = candidate(
            "face:a",
            "voice:a",
            BindingDecision::Accept,
            vec![evidence(
                BindingEvidenceKind::TemporalOverlap,
                "same window",
            )],
        );
        let rejected = candidate(
            "face:a",
            "voice:b",
            BindingDecision::Reject,
            vec![evidence(
                BindingEvidenceKind::Contradiction,
                "voice belongs to another face",
            )],
        );
        let ambiguous = candidate(
            "face:a",
            "voice:c",
            BindingDecision::HoldAmbiguous,
            vec![evidence(
                BindingEvidenceKind::SimultaneousConflict,
                "two voices are active",
            )],
        );
        GraphIntelligenceDocument {
            id: "doc:test".to_string(),
            t_ms: 120,
            provenance: "test".to_string(),
            features: vec![feature],
            clusters: vec![
                DiscoveredCluster::new(
                    "face:a",
                    Modality::Vision,
                    DiscoveredClusterKind::Face,
                    100,
                    0.8,
                )
                .with_feature_ids(vec![feature_id]),
                DiscoveredCluster::new(
                    "voice:a",
                    Modality::Audio,
                    DiscoveredClusterKind::Voice,
                    100,
                    0.8,
                ),
            ],
            binding_candidates: vec![accepted.clone(), rejected, ambiguous],
            binding_edges: vec![BindingEdge {
                left_cluster_id: accepted.left_cluster_id.clone(),
                right_cluster_id: accepted.right_cluster_id.clone(),
                relation: accepted.relation.clone(),
                confidence: 0.9,
                evidence_count: 2,
                decay_per_tick: 0.01,
                last_seen_ms: 120,
            }],
            tracking_hypotheses: vec![
                TrackingHypothesis {
                    id: "hypothesis:face:a:ada".to_string(),
                    family_id: "face:a".to_string(),
                    kind: TrackingHypothesisKind::FaceIdentity,
                    target_id: Some("entity:ada".to_string()),
                    observation_ids: vec!["obs:a".to_string()],
                    binding_candidate_ids: vec!["candidate:a".to_string()],
                    confidence: 0.51,
                    evidence: vec![evidence(
                        BindingEvidenceKind::VectorSimilarity,
                        "close face vector",
                    )],
                    contradictions: Vec::new(),
                    state: HypothesisState::Winning,
                    first_seen_ms: 100,
                    last_updated_ms: 120,
                },
                TrackingHypothesis {
                    id: "hypothesis:face:a:grace".to_string(),
                    family_id: "face:a".to_string(),
                    kind: TrackingHypothesisKind::FaceIdentity,
                    target_id: Some("entity:grace".to_string()),
                    observation_ids: vec!["obs:a".to_string()],
                    binding_candidate_ids: vec!["candidate:b".to_string()],
                    confidence: 0.49,
                    evidence: vec![evidence(
                        BindingEvidenceKind::VectorSimilarity,
                        "also close",
                    )],
                    contradictions: vec!["face vector close to two people".to_string()],
                    state: HypothesisState::NeedsReview,
                    first_seen_ms: 100,
                    last_updated_ms: 120,
                },
            ],
            constellations: vec![Constellation {
                id: "constellation:person:a".to_string(),
                kind_hint: Some("person".to_string()),
                member_cluster_ids: vec!["face:a".to_string(), "voice:a".to_string()],
                member_binding_ids: vec![binding_candidate_id(&accepted)],
                supporting_feature_ids: Vec::new(),
                supporting_entity_ids: vec!["entity:ada".to_string()],
                supporting_place_cells: vec![PlaceCellKey { x: 1, y: 2 }],
                confidence: 0.55,
                stability: 0.35,
                prediction_value: 0.4,
                first_seen_ms: 100,
                last_seen_ms: 120,
                evidence_count: 1,
                state: ConstellationState::Candidate,
                notes: vec!["missing voice confirmation".to_string()],
            }],
            associations: vec![AssociationEdge {
                id: "association:face-predicts-voice".to_string(),
                from_id: "face:a".to_string(),
                to_id: "voice:a".to_string(),
                relation: AssociationRelation::Predicts,
                confidence: 0.62,
                evidence_count: 3,
                prediction_gain: 0.2,
                contradiction_count: 1,
                first_seen_ms: 100,
                last_seen_ms: 120,
                examples: vec![AssociationExample {
                    frame_id: Some("frame-a".to_string()),
                    t_ms: 120,
                    reason: "face preceded voice".to_string(),
                    score: 0.7,
                }],
            }],
            predictions: vec![PredictionRecord {
                id: "prediction:voice".to_string(),
                target_id: "voice:a".to_string(),
                predicted: "voice continues".to_string(),
                confidence: 0.6,
                t_ms: 120,
                state: "open".to_string(),
                reason: "association predicts voice".to_string(),
            }],
            surprises: vec![SurpriseRecord {
                id: "surprise:voice".to_string(),
                target_id: "prediction:voice".to_string(),
                observed: "voice stopped".to_string(),
                surprise: 0.8,
                confidence: 0.7,
                t_ms: 130,
                reason: "speaker stopped unexpectedly".to_string(),
            }],
            ..GraphIntelligenceDocument::default()
        }
    }

    #[test]
    fn cognitive_report_serializes_and_summarizes_sensitive_data() {
        let report = CognitiveDiagnosticsReport::from_graph_document(&diagnostic_document());
        let value = serde_json::to_value(&report).expect("serializable report");

        assert_eq!(value["summary"]["feature_count"], 1);
        assert_eq!(
            value["features"]["items"][0]["metadata_summary"]["raw_vector"]["kind"],
            "array"
        );
        assert!(
            value["features"]["items"][0]["metadata_summary"]["raw_vector"]["vector"].is_null()
        );
    }

    #[test]
    fn binding_inspector_includes_accepted_rejected_and_ambiguous_candidates() {
        let report = CognitiveDiagnosticsReport::from_graph_document(&diagnostic_document());

        assert_eq!(report.summary.accepted_binding_count, 1);
        assert_eq!(report.summary.rejected_binding_count, 1);
        assert_eq!(report.summary.ambiguous_binding_count, 1);
        assert!(report
            .bindings
            .items
            .iter()
            .any(|item| item.accepted_binding_edge_id.is_some()));
        assert!(report
            .bindings
            .items
            .iter()
            .any(|item| item.rejection_reason.is_some()));
        assert!(report
            .bindings
            .items
            .iter()
            .any(|item| item.ambiguity_reason.is_some()));
    }

    #[test]
    fn hypothesis_constellation_active_learning_and_summary_are_inspectable() {
        let report = CognitiveDiagnosticsReport::from_graph_document(&diagnostic_document());

        assert_eq!(report.hypotheses.families.len(), 1);
        assert_eq!(report.hypotheses.families[0].competing_hypotheses.len(), 2);
        assert!(report.constellations.items[0]
            .member_clusters
            .contains(&"face:a".to_string()));
        assert!(!report.constellations.items[0]
            .missing_expected_evidence
            .is_empty());
        assert!(!report.active_learning.open_questions.is_empty());
        assert_eq!(report.summary.cluster_count, 2);
        assert_eq!(report.summary.constellation_count, 1);
        assert_eq!(report.summary.association_count, 1);
        assert_eq!(report.summary.prediction_failure_count, 1);
    }

    #[test]
    fn learning_cycle_turns_person_greeting_into_replay_and_training() {
        let feature = Feature::new(
            FeatureType::FaceObservation,
            FeatureModality::Vision,
            1_000,
            0.88,
            Provenance::direct().with_stage("observe"),
        );
        let feature_id = feature.id;
        let accepted = BindingCandidate {
            left_cluster_id: "cluster:person-face:ada".to_string(),
            right_cluster_id: "cluster:greeting:hello".to_string(),
            relation: BindingRelation::NamedBy,
            evidence: vec![
                evidence(
                    BindingEvidenceKind::TemporalOverlap,
                    "face and greeting co-occurred",
                ),
                evidence(
                    BindingEvidenceKind::RepeatedCooccurrence,
                    "greeting repeated",
                ),
            ],
            confidence: 0.91,
            decision: BindingDecision::Accept,
            reason: "person greeting binding accepted".to_string(),
        };
        let document = GraphIntelligenceDocument {
            id: "doc:greeting".to_string(),
            t_ms: 1_000,
            frame_id: Some("frame:greeting".to_string()),
            provenance: "integration_test".to_string(),
            features: vec![feature],
            clusters: vec![
                DiscoveredCluster::new(
                    "cluster:person-face:ada",
                    Modality::Vision,
                    DiscoveredClusterKind::Face,
                    1_000,
                    0.9,
                )
                .with_feature_ids(vec![feature_id]),
                DiscoveredCluster::new(
                    "cluster:greeting:hello",
                    Modality::Language,
                    DiscoveredClusterKind::Label,
                    1_000,
                    0.85,
                ),
            ],
            binding_candidates: vec![accepted.clone()],
            constellations: vec![Constellation {
                id: "constellation:person:greeting".to_string(),
                kind_hint: Some("person".to_string()),
                member_cluster_ids: vec![
                    "cluster:person-face:ada".to_string(),
                    "cluster:greeting:hello".to_string(),
                ],
                member_binding_ids: vec![binding_candidate_id(&accepted)],
                supporting_feature_ids: vec![feature_id],
                supporting_entity_ids: vec!["entity:person:ada".to_string()],
                supporting_place_cells: Vec::new(),
                confidence: 0.82,
                stability: 0.78,
                prediction_value: 0.72,
                first_seen_ms: 900,
                last_seen_ms: 1_000,
                evidence_count: 3,
                state: ConstellationState::Stable,
                notes: Vec::new(),
            }],
            associations: vec![AssociationEdge {
                id: "association:person-predicts-greeting".to_string(),
                from_id: "constellation:person:greeting".to_string(),
                to_id: "outcome:greeting".to_string(),
                relation: AssociationRelation::Predicts,
                confidence: 0.74,
                evidence_count: 4,
                prediction_gain: 0.22,
                contradiction_count: 0,
                first_seen_ms: 900,
                last_seen_ms: 1_000,
                examples: vec![AssociationExample {
                    frame_id: Some("frame:greeting".to_string()),
                    t_ms: 1_000,
                    reason: "person appearance predicted greeting".to_string(),
                    score: 0.8,
                }],
            }],
            predictions: vec![PredictionRecord {
                id: "prediction:greeting".to_string(),
                target_id: "outcome:greeting".to_string(),
                predicted: "person says hello".to_string(),
                confidence: 0.76,
                t_ms: 1_000,
                state: "succeeded".to_string(),
                reason: "association predicted greeting".to_string(),
            }],
            ..GraphIntelligenceDocument::default()
        };

        let report = CognitiveDiagnosticsReport::from_graph_document(&document);
        let event_kinds = report
            .learning_cycle
            .learning_events
            .iter()
            .map(|event| event.event.clone())
            .collect::<BTreeSet<_>>();

        assert!(event_kinds.contains(&LearningEvent::FeatureObserved));
        assert!(event_kinds.contains(&LearningEvent::ClusterStrengthened));
        assert!(event_kinds.contains(&LearningEvent::BindingAccepted));
        assert!(event_kinds.contains(&LearningEvent::ConstellationPromoted));
        assert!(event_kinds.contains(&LearningEvent::AssociationStrengthened));
        assert!(event_kinds.contains(&LearningEvent::PredictionSucceeded));
        assert!(report
            .learning_cycle
            .training_examples
            .iter()
            .any(|example| example.target_model == "prediction_model"
                && example.label == "prediction_succeeded"));
        assert!(report
            .learning_cycle
            .replay_items
            .iter()
            .any(|item| item.target_id == "constellation:person:greeting"
                && item.curriculum.priority >= 0.5));
        assert_eq!(
            report.summary.training_example_count,
            report.learning_cycle.training_examples.len()
        );
    }

    #[test]
    fn learning_cycle_marks_human_confirmed_ambiguity_as_trusted_training() {
        let hypothesis = TrackingHypothesis {
            id: "hypothesis:face:unknown:ada".to_string(),
            family_id: "family:ambiguous-face".to_string(),
            kind: TrackingHypothesisKind::FaceIdentity,
            target_id: Some("entity:person:ada".to_string()),
            observation_ids: vec!["face-vector:ambiguous".to_string()],
            binding_candidate_ids: vec!["binding:ambiguous-face-ada".to_string()],
            confidence: 0.93,
            evidence: vec![
                evidence(
                    BindingEvidenceKind::VectorSimilarity,
                    "face vector is close",
                ),
                BindingEvidence {
                    kind: BindingEvidenceKind::HumanConfirmed,
                    score: 1.0,
                    reason: "human confirmed this is Ada".to_string(),
                },
            ],
            contradictions: Vec::new(),
            state: HypothesisState::Promoted,
            first_seen_ms: 2_000,
            last_updated_ms: 2_400,
        };
        let document = GraphIntelligenceDocument {
            id: "doc:human-confirmed".to_string(),
            t_ms: 2_400,
            provenance: "integration_test".to_string(),
            binding_candidates: vec![BindingCandidate {
                left_cluster_id: "cluster:face:unknown".to_string(),
                right_cluster_id: "entity:person:ada".to_string(),
                relation: BindingRelation::LikelySameEntity,
                evidence: hypothesis.evidence.clone(),
                confidence: 0.93,
                decision: BindingDecision::Accept,
                reason: "human confirmation promoted the binding".to_string(),
            }],
            tracking_hypotheses: vec![
                hypothesis,
                TrackingHypothesis {
                    id: "hypothesis:face:unknown:other".to_string(),
                    family_id: "family:ambiguous-face".to_string(),
                    kind: TrackingHypothesisKind::FaceIdentity,
                    target_id: Some("entity:person:other".to_string()),
                    observation_ids: vec!["face-vector:ambiguous".to_string()],
                    binding_candidate_ids: vec!["binding:ambiguous-face-other".to_string()],
                    confidence: 0.21,
                    evidence: vec![evidence(
                        BindingEvidenceKind::SimultaneousConflict,
                        "human confirmation rejected this competitor",
                    )],
                    contradictions: vec!["human confirmation selected Ada".to_string()],
                    state: HypothesisState::Rejected,
                    first_seen_ms: 2_000,
                    last_updated_ms: 2_400,
                },
            ],
            human_reviews: vec![HumanReviewRecord {
                id: "human-review:ada".to_string(),
                target_id: "hypothesis:face:unknown:ada".to_string(),
                target_kind: ActiveLearningTargetKind::TrackingHypothesis,
                confidence: 1.0,
                t_ms: 2_400,
                confirmation: "That face is Ada".to_string(),
                reviewer: Some("test-human".to_string()),
            }],
            ..GraphIntelligenceDocument::default()
        };

        let report = CognitiveDiagnosticsReport::from_graph_document(&document);

        assert!(report
            .learning_cycle
            .learning_events
            .iter()
            .any(|event| event.event == LearningEvent::HumanCorrection && event.trusted));
        assert!(report
            .learning_cycle
            .training_examples
            .iter()
            .any(
                |example| example.kind == TrainingExampleKind::HumanTrustedPositive
                    && example.trusted
                    && example.weight >= 0.9
            ));
        assert!(report.learning_cycle.replay_items.iter().any(|item| item
            .curriculum
            .human_confirmation
            == 1.0
            && item.curriculum.priority >= 0.75));
        assert_eq!(report.learning_cycle.hypotheses_promoted, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pete_body::BodySense;
    use pete_experience::{
        Experience, ExperienceLatent, Impression, Modality, RecalledExperience, Sensation,
        SensationMetadata, SensationPayload, SensationPayloadKind, SensationSource,
        VectorEmbedding,
    };
    use pete_ledger::ExperienceFrame;
    use pete_now::{
        EpisodeKind, FaceSense, ObjectClass, ObjectObservation, ObjectObservationSource, PersonId,
        SemanticConceptId, SemanticContext, SemanticDriveId, SemanticPredicate, SemanticRelation,
        SemanticRelationId, SemanticRelationStatus, SurpriseSense, VectorArtifact,
        WorldModelUpdateContext, WorldModelUpdater, FACE_VECTOR_COLLECTION,
        OBJECT_VECTOR_COLLECTION, SCENE_VECTOR_COLLECTION, VOICE_VECTOR_COLLECTION,
    };

    fn empty_frame(now: Now) -> ExperienceFrame {
        ExperienceFrame {
            id: uuid::Uuid::new_v4(),
            t_ms: now.t_ms,
            now,
            sensations: Vec::new(),
            impressions: Vec::new(),
            experiences: Vec::new(),
            z: None,
            chosen_action: None,
            conscious_command: None,
            reign_input: None,
            reign_outcome: None,
            predicted_futures: Vec::new(),
            behavior_runs: Vec::new(),
            actual_next: None,
            reward: Default::default(),
            surprise: SurpriseSense::default(),
            memory_recall: Vec::new(),
            recollections: Vec::new(),
            llm_teaching: Vec::new(),
            counterfactuals: Vec::new(),
            notes: vec!["saw a familiar person".to_string()],
        }
    }

    #[tokio::test]
    async fn durable_records_preserve_temporal_and_social_world_models() {
        let mut updater = WorldModelUpdater::default();
        let mut present = Now::blank(0, BodySense::default());
        present.body.charging = true;
        present.objects.observations.push(ObjectObservation {
            label: "Alex".to_string(),
            class: ObjectClass::Person,
            bearing_rad: 0.1,
            distance_m: Some(0.8),
            confidence: 0.9,
            source: ObjectObservationSource::Kinect,
        });
        updater.update(present, WorldModelUpdateContext::default());

        let closed = updater.update(
            Now::blank(2_000, BodySense::default()),
            WorldModelUpdateContext::default(),
        );
        let frame = empty_frame(closed);
        let store = InMemoryExperienceStore::new();
        store.store(&frame).await.unwrap();

        assert_eq!(store.completed_episodes(EpisodeKind::Charging).len(), 1);
        let person_id = PersonId("person:alex".to_string());
        assert!(store.last_social_interaction(&person_id).is_some());
        let record = store.snapshot().pop().unwrap();
        assert!(record.social_world.people.contains_key(&person_id));
        assert!(record
            .temporal_context
            .recently_completed
            .iter()
            .any(|episode| episode.kind == EpisodeKind::Conversation));
        assert!(record
            .graph_entities
            .iter()
            .any(|entity| { entity.id == person_id.0 && entity.has_label("SocialPersonBelief") }));
        assert!(record
            .graph_entities
            .iter()
            .any(|entity| entity.has_label("Charging")));
        assert!(record
            .graph_relationships
            .iter()
            .any(|relationship| { relationship.relationship == "HAS_TEMPORAL_EPISODE" }));
        assert!(record
            .graph_entities
            .iter()
            .any(|entity| entity.id == "concept:charger" && entity.has_label("SemanticNode")));
        assert!(record.graph_relationships.iter().any(|relationship| {
            relationship.from == "concept:charger"
                && relationship.to == "drive:energy"
                && relationship.relationship == "SEMANTIC_RESTORES"
                && relationship.payload.get("supporting_evidence").is_some()
        }));
    }

    #[test]
    fn context_distinct_semantic_relations_keep_distinct_persistence_ids() {
        let subject = SemanticNodeRef::Concept(SemanticConceptId("charger".to_string()));
        let object = SemanticNodeRef::Drive(SemanticDriveId("energy".to_string()));
        let mut semantic = SemanticGraphSnapshot::default();

        for route_state in ["clear", "blocked"] {
            let id = SemanticRelationId(format!("semantic:test:restores:{route_state}"));
            semantic.relations.insert(
                id.clone(),
                SemanticRelation {
                    id,
                    subject: subject.clone(),
                    predicate: SemanticPredicate::Restores,
                    object: object.clone(),
                    context: SemanticContext {
                        agent: Some("pete".to_string()),
                        conditions: BTreeMap::from([(
                            "route_state".to_string(),
                            route_state.to_string(),
                        )]),
                        ..SemanticContext::default()
                    },
                    confidence: 0.8,
                    status: SemanticRelationStatus::ContextLimited,
                    ..SemanticRelation::default()
                },
            );
        }

        let mut now = Now::blank(1_000, BodySense::default());
        now.world.semantic = semantic;
        let record = memory_record_from_frame(&empty_frame(now)).expect("memory record");
        let semantic_edges = record
            .graph_relationships
            .iter()
            .filter(|edge| edge.relationship == "SEMANTIC_RESTORES")
            .collect::<Vec<_>>();

        assert_eq!(semantic_edges.len(), 2);
        assert_eq!(
            semantic_edges
                .iter()
                .filter_map(|edge| edge.id.as_deref())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "semantic:test:restores:blocked",
                "semantic:test:restores:clear",
            ])
        );
        assert_eq!(
            semantic_edges
                .iter()
                .filter_map(|edge| {
                    edge.payload
                        .pointer("/context/conditions/route_state")
                        .and_then(|value| value.as_str())
                })
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["blocked", "clear"])
        );

        let neo4j_params = neo4j_relationship_params(&record);
        assert!(neo4j_params
            .iter()
            .filter(|param| param["kind"] == "SEMANTIC_RESTORES")
            .all(|param| param["payload_json"]
                .as_str()
                .is_some_and(|payload| payload.contains("\"id\":\"semantic:test:restores:"))));
        let neo4j_edge_ids = neo4j_params
            .into_iter()
            .filter(|param| param["kind"] == "SEMANTIC_RESTORES")
            .filter_map(|param| param["edge_id"].as_str().map(str::to_string))
            .collect::<BTreeSet<_>>();
        assert_eq!(
            neo4j_edge_ids,
            BTreeSet::from([
                "semantic:test:restores:blocked".to_string(),
                "semantic:test:restores:clear".to_string(),
            ])
        );
        assert!(NEO4J_GRAPH_UPSERT_CYPHER.contains("{edge_id: relationship.edge_id}"));
        assert!(NEO4J_LEGACY_RELATED_EDGE_MIGRATION_CYPHER.contains("WHERE legacy.edge_id IS NULL"));
        assert!(NEO4J_LEGACY_RELATED_EDGE_MIGRATION_CYPHER.contains("SET legacy.edge_id"));
        assert!(NEO4J_LEGACY_RELATED_EDGE_MIGRATION_CYPHER.contains("legacy.payload_json"));
        assert!(!NEO4J_LEGACY_RELATED_EDGE_MIGRATION_CYPHER.contains("DELETE"));
    }

    fn test_cluster(
        id: &str,
        kind: DiscoveredClusterKind,
        modality: Modality,
    ) -> DiscoveredCluster {
        DiscoveredCluster::new(id, modality, kind, 1_000, 0.9)
    }

    fn accepted_binding(left: &str, right: &str, relation: BindingRelation) -> BindingCandidate {
        BindingCandidate {
            left_cluster_id: left.to_string(),
            right_cluster_id: right.to_string(),
            relation,
            evidence: vec![
                BindingEvidence {
                    kind: BindingEvidenceKind::TemporalOverlap,
                    score: 0.9,
                    reason: "test temporal support".to_string(),
                },
                BindingEvidence {
                    kind: BindingEvidenceKind::RepeatedCooccurrence,
                    score: 0.85,
                    reason: "test repeated support".to_string(),
                },
            ],
            confidence: 0.9,
            decision: BindingDecision::Accept,
            reason: "test accepted binding".to_string(),
        }
    }

    fn person_constellation_observation(t_ms: u64) -> ConstellationObservation {
        ConstellationObservation {
            t_ms,
            clusters: vec![
                test_cluster("face:travis", DiscoveredClusterKind::Face, Modality::Vision),
                test_cluster(
                    "voice:travis",
                    DiscoveredClusterKind::Voice,
                    Modality::Audio,
                ),
                test_cluster(
                    "label:travis",
                    DiscoveredClusterKind::Label,
                    Modality::Language,
                ),
            ],
            accepted_bindings: vec![
                accepted_binding(
                    "face:travis",
                    "voice:travis",
                    BindingRelation::LikelySameEntity,
                ),
                accepted_binding("face:travis", "label:travis", BindingRelation::NamedBy),
            ],
            active_entity_ids: vec!["entity:person:travis".to_string()],
            place_cells: vec![PlaceCellKey { x: 2, y: -1 }],
            prediction_value: 0.65,
            ..ConstellationObservation::default()
        }
    }

    #[test]
    fn repeated_accepted_bindings_form_candidate_constellation() {
        let mut engine = ConstellationEngine::new();
        let constellation = engine
            .observe(ConstellationObservation {
                t_ms: 1_000,
                clusters: vec![
                    test_cluster("face:a", DiscoveredClusterKind::Face, Modality::Vision),
                    test_cluster("voice:a", DiscoveredClusterKind::Voice, Modality::Audio),
                ],
                accepted_bindings: vec![accepted_binding(
                    "face:a",
                    "voice:a",
                    BindingRelation::LikelySameEntity,
                )],
                prediction_value: 0.4,
                ..ConstellationObservation::default()
            })
            .expect("candidate");

        assert_eq!(constellation.state, ConstellationState::Candidate);
        assert_eq!(constellation.member_cluster_ids.len(), 2);
        assert_eq!(constellation.member_binding_ids.len(), 1);
    }

    #[test]
    fn candidate_becomes_stable_only_after_repeated_evidence() {
        let mut engine = ConstellationEngine::new();
        let first = engine
            .observe(person_constellation_observation(1_000))
            .expect("first candidate");
        assert_eq!(first.state, ConstellationState::Candidate);

        engine.observe(person_constellation_observation(2_000));
        let stable = engine
            .observe(person_constellation_observation(3_000))
            .expect("stable constellation");

        assert_eq!(stable.state, ConstellationState::Stable);
        assert!(stable.confidence >= engine.config.promotion_confidence_threshold);
        assert_eq!(stable.evidence_count, 3);
    }

    #[test]
    fn constellation_is_not_promoted_from_one_strong_cluster_alone() {
        let mut engine = ConstellationEngine::new();
        for t_ms in [1_000, 2_000, 3_000, 4_000] {
            let admitted = engine.observe(ConstellationObservation {
                t_ms,
                clusters: vec![test_cluster(
                    "face:solo",
                    DiscoveredClusterKind::Face,
                    Modality::Vision,
                )],
                prediction_value: 1.0,
                ..ConstellationObservation::default()
            });
            assert!(admitted.is_none());
        }
        assert!(engine.constellations.is_empty());
    }

    #[test]
    fn partial_match_retrieves_known_constellation() {
        let mut engine = ConstellationEngine::new();
        for t_ms in [1_000, 2_000, 3_000] {
            engine.observe(person_constellation_observation(t_ms));
        }

        let matched = engine
            .best_match(&ConstellationQuery {
                t_ms: 4_000,
                cluster_ids: vec!["face:travis".to_string(), "voice:travis".to_string()],
                place_cells: vec![PlaceCellKey { x: 2, y: -1 }],
                ..ConstellationQuery::default()
            })
            .expect("partial match");

        assert!(matched.score >= engine.config.partial_match_threshold);
        assert!(matched
            .matched_cluster_ids
            .contains(&"face:travis".to_string()));
        assert!(matched
            .missing_cluster_ids
            .contains(&"label:travis".to_string()));
    }

    #[test]
    fn missing_modality_does_not_destroy_match() {
        let mut engine = ConstellationEngine::new();
        for t_ms in [1_000, 2_000, 3_000] {
            engine.observe(person_constellation_observation(t_ms));
        }

        let full = engine
            .best_match(&ConstellationQuery {
                t_ms: 4_000,
                cluster_ids: vec![
                    "face:travis".to_string(),
                    "voice:travis".to_string(),
                    "label:travis".to_string(),
                ],
                place_cells: vec![PlaceCellKey { x: 2, y: -1 }],
                ..ConstellationQuery::default()
            })
            .expect("full match");
        let partial = engine
            .best_match(&ConstellationQuery {
                t_ms: 4_000,
                cluster_ids: vec!["face:travis".to_string(), "voice:travis".to_string()],
                place_cells: vec![PlaceCellKey { x: 2, y: -1 }],
                ..ConstellationQuery::default()
            })
            .expect("partial match");

        assert!(partial.score > 0.0);
        assert!(partial.score < full.score);
        assert!(partial.score >= engine.config.partial_match_threshold);
    }

    #[test]
    fn contradiction_lowers_confidence() {
        let mut engine = ConstellationEngine::new();
        for t_ms in [1_000, 2_000, 3_000] {
            engine.observe(person_constellation_observation(t_ms));
        }
        let known_binding_id = engine
            .constellations
            .values()
            .next()
            .unwrap()
            .member_binding_ids
            .first()
            .cloned()
            .unwrap();

        let clean = engine
            .best_match(&ConstellationQuery {
                t_ms: 4_000,
                cluster_ids: vec!["face:travis".to_string(), "voice:travis".to_string()],
                binding_ids: vec![known_binding_id.clone()],
                ..ConstellationQuery::default()
            })
            .expect("clean match");
        let contradicted = engine
            .best_match(&ConstellationQuery {
                t_ms: 4_000,
                cluster_ids: vec!["face:travis".to_string(), "voice:travis".to_string()],
                binding_ids: vec![known_binding_id.clone()],
                contradiction_ids: vec![known_binding_id],
                ..ConstellationQuery::default()
            })
            .expect("contradicted match");

        assert!(contradicted.score < clean.score);
        assert!(contradicted.contradiction_penalty > 0.0);
    }

    #[test]
    fn split_needed_state_appears_when_evidence_suggests_fusion() {
        let mut engine = ConstellationEngine::new();
        let mut binding = accepted_binding(
            "object:patch",
            "geometry:blob",
            BindingRelation::CooccursInEstimatedSpace,
        );
        binding.evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::SimultaneousConflict,
            score: 0.8,
            reason: "two object tracks may have fused".to_string(),
        });

        let constellation = engine
            .observe(ConstellationObservation {
                t_ms: 1_000,
                clusters: vec![
                    test_cluster(
                        "object:patch",
                        DiscoveredClusterKind::Object,
                        Modality::Vision,
                    ),
                    test_cluster(
                        "geometry:blob",
                        DiscoveredClusterKind::Geometry,
                        Modality::Depth,
                    ),
                ],
                accepted_bindings: vec![binding],
                prediction_value: 0.4,
                llm_notes: vec!["this may be two fused patterns".to_string()],
                ..ConstellationObservation::default()
            })
            .expect("split-needed constellation");

        assert_eq!(constellation.state, ConstellationState::SplitNeeded);
        assert!(constellation.confidence < 0.9);
    }

    fn association_item(id: &str, kind: AssociationItemKind, confidence: f32) -> AssociationItem {
        AssociationItem::new(id, kind, confidence)
    }

    #[test]
    fn repeated_cooccurrence_creates_association() {
        let mut engine = AssociationLearningEngine::new();
        for t_ms in [1_000, 1_100, 1_200] {
            engine.observe(AssociationObservation {
                t_ms,
                active_items: vec![
                    association_item("cluster:face:travis", AssociationItemKind::Cluster, 0.9),
                    association_item("cluster:voice:travis", AssociationItemKind::Cluster, 0.85),
                ],
                ..AssociationObservation::default()
            });
        }

        let id = association_edge_id(
            "cluster:face:travis",
            "cluster:voice:travis",
            &AssociationRelation::CoOccursWith,
        );
        let edge = engine.edges.get(&id).expect("co-occurrence edge");
        assert_eq!(edge.relation, AssociationRelation::CoOccursWith);
        assert_eq!(edge.evidence_count, 3);
        assert!(edge.confidence > 0.4);
    }

    #[test]
    fn repeated_sequence_creates_predicts_or_follows() {
        let mut engine = AssociationLearningEngine::new();
        for base in [1_000, 3_000, 5_000] {
            engine.observe(AssociationObservation {
                t_ms: base,
                active_items: vec![association_item(
                    "action:forward",
                    AssociationItemKind::Action,
                    0.9,
                )],
                ..AssociationObservation::default()
            });
            engine.observe(AssociationObservation {
                t_ms: base + 300,
                outcome_items: vec![association_item(
                    "outcome:no-movement",
                    AssociationItemKind::Outcome,
                    0.95,
                )],
                ..AssociationObservation::default()
            });
        }

        let id = association_edge_id(
            "action:forward",
            "outcome:no-movement",
            &AssociationRelation::Predicts,
        );
        let edge = engine.edges.get(&id).expect("prediction edge");
        assert_eq!(edge.relation, AssociationRelation::Predicts);
        assert!(edge.evidence_count >= 3);
        assert!(edge.prediction_gain > 0.0);

        let predictions = engine.predictions_for(&["action:forward".to_string()], 0.1, 3);
        assert!(predictions
            .iter()
            .any(|prediction| prediction.predicted_id == "outcome:no-movement"));
    }

    #[test]
    fn association_confidence_increases_with_evidence() {
        let mut engine = AssociationLearningEngine::new();
        engine.observe(AssociationObservation {
            t_ms: 1_000,
            active_items: vec![
                association_item("place:charger", AssociationItemKind::Constellation, 0.8),
                association_item("body:charging", AssociationItemKind::BodyState, 0.8),
            ],
            ..AssociationObservation::default()
        });
        let id = association_edge_id(
            "body:charging",
            "place:charger",
            &AssociationRelation::CoOccursWith,
        );
        let first_confidence = engine.edges.get(&id).unwrap().confidence;

        for t_ms in [1_500, 2_000, 2_500] {
            engine.observe(AssociationObservation {
                t_ms,
                active_items: vec![
                    association_item("place:charger", AssociationItemKind::Constellation, 0.9),
                    association_item("body:charging", AssociationItemKind::BodyState, 0.9),
                ],
                ..AssociationObservation::default()
            });
        }
        let later_confidence = engine.edges.get(&id).unwrap().confidence;

        assert!(later_confidence > first_confidence);
        assert_eq!(engine.edges.get(&id).unwrap().evidence_count, 4);
    }

    #[test]
    fn active_learning_asks_human_for_ambiguous_identity_binding() {
        let candidate = BindingCandidate {
            left_cluster_id: "face:unknown".to_string(),
            right_cluster_id: "voice:travis-or-tim".to_string(),
            relation: BindingRelation::LikelySameEntity,
            evidence: vec![BindingEvidence {
                kind: BindingEvidenceKind::SimultaneousConflict,
                score: 0.8,
                reason: "two person candidates are active".to_string(),
            }],
            confidence: 0.42,
            decision: BindingDecision::AskHuman,
            reason: "identity is ambiguous".to_string(),
        };
        let mut planner = DefaultActiveLearningPlanner::new();
        let questions = planner.plan(&ActiveLearningInput {
            ambiguous_binding_candidates: vec![candidate],
            ..ActiveLearningInput::default()
        });

        assert_eq!(questions.len(), 1);
        assert_eq!(
            questions[0].target_kind,
            ActiveLearningTargetKind::BindingCandidate
        );
        assert!(questions[0]
            .proposed_tests
            .iter()
            .any(|test| test.kind == ActiveLearningActionKind::AskHuman));
        assert_eq!(questions[0].state, ActiveLearningState::WaitingForHuman);
    }

    #[test]
    fn active_learning_motion_test_stays_proposal_when_safe() {
        let candidate = BindingCandidate {
            left_cluster_id: "rgb:patch".to_string(),
            right_cluster_id: "geometry:blob".to_string(),
            relation: BindingRelation::ProjectsTo,
            evidence: vec![BindingEvidence {
                kind: BindingEvidenceKind::ProjectionAgreement,
                score: 0.45,
                reason: "weak reprojection support".to_string(),
            }],
            confidence: 0.5,
            decision: BindingDecision::CollectMoreEvidence,
            reason: "needs viewpoint evidence".to_string(),
        };
        let mut body = BodySense::default();
        body.last_update_ms = 1_000;
        let mut planner = DefaultActiveLearningPlanner::new();
        let questions = planner.plan(&ActiveLearningInput {
            context: ActiveLearningContext {
                t_ms: 1_000,
                body_state: Some(body.clone()),
                movement_readiness: MovementReadiness::from(&body),
                ..ActiveLearningContext::default()
            },
            ambiguous_binding_candidates: vec![candidate],
            ..ActiveLearningInput::default()
        });

        let motion = questions[0]
            .proposed_tests
            .iter()
            .find(|test| test.kind == ActiveLearningActionKind::MoveOrRotate)
            .expect("motion proposal");
        assert!(matches!(motion.action, Some(ActionPrimitive::Turn { .. })));
        assert!(motion.required_safety_state.is_some());
    }

    #[test]
    fn active_learning_uses_diagnostic_when_movement_is_broken() {
        let failure = PredictionFailure {
            id: "failure:no-motion".to_string(),
            target_id: "action:forward".to_string(),
            predicted: "odometry should change".to_string(),
            observed: "pose delta near zero".to_string(),
            confidence: 0.4,
            surprise: 0.9,
            action: Some(ActionPrimitive::Go {
                intensity: 0.2,
                duration_ms: 300,
            }),
            possible_causes: vec!["base disconnected".to_string()],
        };
        let mut planner = DefaultActiveLearningPlanner::new();
        let questions = planner.plan(&ActiveLearningInput {
            context: ActiveLearningContext {
                movement_readiness: MovementReadiness {
                    base_connected: false,
                    movement_responding: Some(false),
                    reason: Some("base connection is down".to_string()),
                    ..MovementReadiness::default()
                },
                ..ActiveLearningContext::default()
            },
            prediction_failures: vec![failure],
            ..ActiveLearningInput::default()
        });

        assert!(questions[0].proposed_tests.iter().any(|test| test.kind
            == ActiveLearningActionKind::Diagnostic
            && test.action.is_none()
            && test
                .required_safety_state
                .as_deref()
                .is_some_and(|state| state.contains("base connection"))));
    }

    #[test]
    fn active_learning_replays_memory_for_place_candidate() {
        let candidate = PlaceRecognitionCandidate {
            kind: PlaceRecognitionKind::SamePlace,
            cell: PlaceCellSummary {
                x: 1,
                y: 2,
                center_x_m: 0.5,
                center_y_m: 1.0,
                score: 0.4,
                visit_count: 2,
                last_seen_tick: 10,
                confidence: 0.4,
                last_observed_objects: Vec::new(),
                associated_scene_vectors: Vec::new(),
                associated_face_vectors: Vec::new(),
                associated_object_vectors: Vec::new(),
                associated_voice_vectors: Vec::new(),
                successful_actions: Vec::new(),
                failed_actions: Vec::new(),
            },
            source_vector_id: "scene:old".to_string(),
            source_frame_id: None,
            source_experience_id: None,
            source_instant_frame_id: None,
            source_vector_refs: Vec::new(),
            query_vector_id: Some("scene:now".to_string()),
            query_experience_id: None,
            similarity: 0.55,
            confidence: 0.45,
            reason: "weak scene similarity".to_string(),
        };
        let mut planner = DefaultActiveLearningPlanner::new();
        let questions = planner.plan(&ActiveLearningInput {
            place_candidates: vec![candidate],
            ..ActiveLearningInput::default()
        });

        assert!(questions[0]
            .proposed_tests
            .iter()
            .any(|test| test.kind == ActiveLearningActionKind::ReplayMemory));
    }

    #[test]
    fn association_decays_without_evidence() {
        let mut engine = AssociationLearningEngine::new();
        engine.observe(AssociationObservation {
            t_ms: 1_000,
            active_items: vec![
                association_item("plane:wall", AssociationItemKind::Cluster, 0.9),
                association_item("action:forward-unsafe", AssociationItemKind::Outcome, 0.9),
            ],
            ..AssociationObservation::default()
        });
        let id = association_edge_id(
            "action:forward-unsafe",
            "plane:wall",
            &AssociationRelation::CoOccursWith,
        );
        let before = engine.edges.get(&id).unwrap().confidence;
        engine.decay(10);
        let after = engine.edges.get(&id).unwrap().confidence;

        assert!(after < before);
        assert!(after > 0.0);
    }

    #[tokio::test]
    async fn store_preserves_typed_vector_artifacts() {
        let mut now = Now::blank(123, BodySense::default());
        now.face = FaceSense {
            schema_version: 1,
            vectors: vec![
                VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-1", vec![1.0, 0.0])
                    .with_source_id("face-crop-1"),
            ],
        };
        now.eye.scene_vectors.push(VectorArtifact::new(
            SCENE_VECTOR_COLLECTION,
            "scene-1",
            vec![0.0, 1.0],
        ));

        let store = InMemoryExperienceStore::new();
        store.store(&empty_frame(now)).await.unwrap();

        let snapshot = store.snapshot();
        assert_eq!(snapshot[0].face_vectors[0].point_id, "face-1");
        assert_eq!(
            snapshot[0].face_vectors[0].source_id.as_deref(),
            Some("face-crop-1")
        );
        assert_eq!(
            snapshot[0].scene_vectors[0].collection,
            SCENE_VECTOR_COLLECTION
        );
        assert!(snapshot[0]
            .graph_entities
            .iter()
            .any(|entity| entity.has_label("Person")));
        assert!(snapshot[0]
            .graph_entities
            .iter()
            .any(|entity| entity.has_label("Place")));
        assert!(snapshot[0]
            .graph_relationships
            .iter()
            .any(|edge| edge.relationship == "HAS_FACE_VECTOR"));
    }

    #[tokio::test]
    async fn recall_returns_graph_context_as_memory_sense() {
        let mut now = Now::blank(123, BodySense::default());
        now.face.vectors =
            vec![
                VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-1", vec![1.0, 0.0])
                    .with_source_id("person:ada"),
            ];

        let store = InMemoryExperienceStore::new();
        store.store(&empty_frame(now)).await.unwrap();

        let recall = store
            .recall(RecallQuery {
                face_vectors: vec![VectorArtifact::new(
                    FACE_VECTOR_COLLECTION,
                    "query-face",
                    vec![1.0, 0.0],
                )],
                battery: 1.0,
                ..RecallQuery::default()
            })
            .await
            .unwrap();

        assert!(recall
            .sense
            .remembered_entities
            .iter()
            .any(|entity| entity.id == "person:ada" && entity.has_label("Person")));
        assert!(recall
            .sense
            .remembered_entities
            .iter()
            .any(|entity| entity.has_label("Place")));
        assert!(recall.sense.graph_context_summary.is_some());
        assert!(recall.hits[0]
            .graph_context
            .iter()
            .any(|entity| entity.has_label("Person")));
    }

    #[tokio::test]
    async fn recall_returns_embodied_memory_recall_sensations_with_lineage() {
        let now = Now::blank(500, BodySense::default());
        let mut frame = empty_frame(now);
        let source_sensation_id = uuid::Uuid::new_v4();
        let experience = Experience::new(
            "embodied.now",
            "I notice a familiar charger alcove.",
            Vec::new(),
            vec![source_sensation_id],
            450,
            500,
        );
        frame.z = Some(ExperienceLatent {
            t_ms: frame.t_ms,
            z: vec![1.0, 0.0, 0.0],
            confidence: 0.9,
            ..ExperienceLatent::default()
        });
        frame.experiences = vec![experience.clone()];
        let frame_id = frame.id;

        let store = InMemoryExperienceStore::new();
        store.store(&frame).await.unwrap();
        let recall = store
            .recall(RecallQuery {
                scene_vectors: vec![VectorArtifact::new(
                    EXPERIENCE_VECTOR_COLLECTION,
                    "query-experience",
                    vec![1.0, 0.0, 0.0],
                )],
                battery: frame.now.body.battery_level,
                ..RecallQuery::default()
            })
            .await
            .unwrap();

        let recollection = recall.recollections.first().expect("recollection");
        assert_eq!(recollection.original_frame_id, Some(frame_id));
        assert!(recollection
            .original_vector_ids
            .iter()
            .any(|id| id.contains(&experience.id.to_string())));
        assert_eq!(recollection.sensation.modality, Modality::Memory);
        assert_eq!(
            recollection.sensation.payload_kind,
            SensationPayloadKind::MemoryRecall
        );
        assert!(matches!(
            recollection.sensation.provenance.kind,
            pete_core::ProvenanceKind::MemoryRecall { experience_id }
                if experience_id == experience.id
        ));
        assert!(recollection
            .sensation
            .impression
            .as_ref()
            .is_some_and(|impression| impression.text.starts_with("I remember")));
    }

    #[tokio::test]
    async fn object_vectors_are_memorized_and_recalled_like_faces() {
        let mut now = now_at(1_000, 0.0, 0.0);
        now.objects.observations.push(ObjectObservation {
            label: "red cup".to_string(),
            class: ObjectClass::Landmark,
            bearing_rad: 0.1,
            distance_m: Some(1.2),
            confidence: 0.9,
            source: ObjectObservationSource::Kinect,
        });
        now.objects.vectors.push(
            VectorArtifact::new(OBJECT_VECTOR_COLLECTION, "object-red-cup", vec![1.0, 0.0])
                .with_model("test.object.embedding")
                .with_source_id("entity:landmark:red-cup"),
        );
        let frame = empty_frame(now);
        let store = InMemoryExperienceStore::new();
        store.store(&frame).await.unwrap();

        let record = store.snapshot().pop().expect("stored record");
        assert_eq!(record.object_vectors.len(), 1);
        assert!(record
            .graph_relationships
            .iter()
            .any(|edge| edge.relationship == "HAS_OBJECT_VECTOR"));

        let recall = store
            .recall(RecallQuery {
                object_vectors: vec![VectorArtifact::new(
                    OBJECT_VECTOR_COLLECTION,
                    "object-query",
                    vec![1.0, 0.0],
                )],
                ..RecallQuery::default()
            })
            .await
            .unwrap();

        assert_eq!(recall.hits.len(), 1);
        assert!(recall.sense.object_familiarity > 0.99);
    }

    #[test]
    fn deterministic_embodied_fixture_exercises_hardware_free_modalities() {
        let now = deterministic_embodied_fixture_now(1_000, 0.0);
        assert!(now
            .eye_frame
            .as_ref()
            .is_some_and(|frame| !frame.bytes.is_empty()));
        assert_eq!(now.range.nearest_m, Some(0.42));
        assert!(!now.kinect.depth_m.is_empty());
        assert_eq!(now.ear.asr.sample_rate_hz, Some(16_000));
        assert!(now.ear.asr.transcript.is_some());
        assert!(now.body.flags.wall);

        let primary = pete_experience::primary_sensations_from_now(&now);
        assert!(primary
            .iter()
            .any(|sensation| sensation.payload_kind == SensationPayloadKind::ImageBytes));
        assert!(primary
            .iter()
            .any(|sensation| sensation.payload_kind == SensationPayloadKind::DepthFrame));
        assert!(primary
            .iter()
            .any(|sensation| sensation.payload_kind == SensationPayloadKind::LidarScan));
        assert!(primary
            .iter()
            .any(|sensation| sensation.payload_kind == SensationPayloadKind::AudioPcm));
        assert!(primary
            .iter()
            .any(|sensation| sensation.payload_kind == SensationPayloadKind::ContactEvent));
    }

    #[tokio::test]
    async fn deterministic_embodied_frame_preserves_lineage_vectors_and_experience_outputs() {
        let frame = super::build_embodied_eval_frame(
            deterministic_embodied_fixture_now(1_000, 0.0),
            None,
            &[],
        )
        .await
        .unwrap();

        let visual_parent_ids = frame
            .sensations
            .iter()
            .filter(|sensation| {
                sensation.modality == Modality::Vision
                    && sensation.payload_kind == SensationPayloadKind::ImageBytes
            })
            .map(|sensation| sensation.id)
            .collect::<BTreeSet<_>>();
        assert!(frame.sensations.iter().any(|sensation| {
            sensation.modality == Modality::Vision
                && sensation.payload_kind == SensationPayloadKind::Crop
                && sensation
                    .parent_id
                    .is_some_and(|parent_id| visual_parent_ids.contains(&parent_id))
        }));
        assert!(frame.sensations.iter().any(|sensation| {
            sensation.vector.as_ref().is_some_and(|vector| {
                !vector.model_id.is_empty()
                    && vector.dim == vector.vector.len()
                    && !vector.purpose.is_empty()
            })
        }));
        assert!(frame
            .impressions
            .iter()
            .any(|impression| impression.sensation_id.is_some()));
        let experience = frame.experiences.last().expect("experience");
        assert!(experience.summary_impression.is_some());
        assert!(!experience.predictions.is_empty());
    }

    #[tokio::test]
    async fn deterministic_replay_produces_identical_instant_shape() {
        let left = super::build_embodied_eval_frame(
            deterministic_embodied_fixture_now(1_000, 0.0),
            None,
            &[],
        )
        .await
        .unwrap()
        .experience_instant();
        let right = super::build_embodied_eval_frame(
            deterministic_embodied_fixture_now(1_000, 0.0),
            None,
            &[],
        )
        .await
        .unwrap()
        .experience_instant();

        assert_eq!(stable_instant_shape(&left), stable_instant_shape(&right));
    }

    #[tokio::test]
    async fn instant_conversion_preserves_lineage_vectors_predictions_and_memory_links() {
        let mut frame = super::build_embodied_eval_frame(
            deterministic_embodied_fixture_now(1_000, 0.0),
            None,
            &[],
        )
        .await
        .unwrap();
        attach_memory_links_to_frame(&mut frame);

        let instant = frame.experience_instant();
        assert!(!instant.lineage.is_empty());
        assert!(instant.teacher_vectors.iter().any(|vector| vector
            .metadata
            .model_id
            .contains("fixture")
            || vector.metadata.model_id.contains("pete")));
        assert!(instant
            .teacher_vectors
            .iter()
            .all(|vector| vector.metadata.dim == vector.vector.len()));
        assert!(!instant.predictions.is_empty());
        assert!(!instant.memory_links.is_empty());

        let context = instant.embodied_context();
        assert_eq!(context.lineage, instant.lineage);
        assert_eq!(context.predictions, instant.predictions);
        assert_eq!(context.memory_links, instant.memory_links);
    }

    #[tokio::test]
    async fn instant_missing_modalities_are_explicit_in_coverage() {
        let frame = super::build_embodied_eval_frame(
            deterministic_embodied_fixture_now(1_000, 0.0),
            None,
            &[EmbodiedEvalOmission::Vectors],
        )
        .await
        .unwrap();
        let instant = frame.experience_instant();
        let coverage = instant.coverage();

        assert!(!instant.missing_modalities.is_empty());
        assert!(!coverage.missing_modalities.is_empty());
        assert_eq!(
            coverage.sensation_count,
            instant.primary_sensations.len() + instant.descendant_sensations.len()
        );
        assert_eq!(coverage.vector_count, instant.teacher_vectors.len());
    }

    #[tokio::test]
    async fn deterministic_embodied_eval_reports_full_coverage_and_recall() {
        let report = deterministic_embodied_eval_report().await.unwrap();

        assert!(report.passed(), "{:?}", report.failures);
        assert_eq!(report.frame_count, 2);
        assert_eq!(report.instant_count, 2);
        assert!(report.instant_teacher_vector_count > 0);
        assert!(report.instant_missing_modality_count > 0);
        assert!(report.primary_sensation_count > 0);
        assert!(report.descendant_sensation_count > 0);
        assert!(report.vectorized_sensation_count > 0);
        assert!(report.impression_count > 0);
        assert!(report.summary_impression_count > 0);
        assert!(report.experience_latent_count > 0);
        assert!(report.prediction_count > 0);
        assert!(report.memory_link_count > 0);
        assert!(report.recall_sensation_count > 0);
        assert!(report.recall_impression_count > 0);
        assert!(report.place_recognition_candidate_count > 0);
        assert!(report.lineage_edge_count > 0);
        assert!(report.input_modalities.contains(&"vision".to_string()));
        assert!(report.input_modalities.contains(&"depth".to_string()));
        assert!(report.input_modalities.contains(&"lidar".to_string()));
        assert!(report.input_modalities.contains(&"audio".to_string()));
        assert_eq!(report.instant_coverage.len(), report.instant_count);
        assert!(report
            .instant_coverage
            .iter()
            .all(|coverage| coverage.sensation_count > 0));
    }

    #[tokio::test]
    async fn deterministic_embodied_eval_reports_deliberately_missing_stages() {
        let report = deterministic_embodied_eval_report_with_omissions(&[
            EmbodiedEvalOmission::Vectors,
            EmbodiedEvalOmission::Recall,
        ])
        .await
        .unwrap();

        assert!(!report.passed());
        assert!(report
            .failures
            .iter()
            .any(|failure| failure == "no vectors"));
        assert!(report.failures.iter().any(|failure| failure == "no recall"));
        assert!(report
            .failures
            .iter()
            .any(|failure| failure == "no place recognition"));
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning == "omitted vectors"));
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning == "omitted recall"));
    }

    fn stable_instant_shape(instant: &pete_experience::ExperienceInstant) -> serde_json::Value {
        serde_json::json!({
            "schema_version": instant.schema_version,
            "t_ms": instant.t_ms,
            "window_start_ms": instant.window_start_ms,
            "window_end_ms": instant.window_end_ms,
            "primary": instant.primary_sensations.iter().map(|sensation| {
                serde_json::json!({
                    "parent": sensation.parent_id.is_some(),
                    "modality": sensation.modality.as_str(),
                    "payload_kind": sensation.payload_kind.as_str(),
                    "kind": sensation.kind,
                    "source": sensation.source,
                })
            }).collect::<Vec<_>>(),
            "descendant": instant.descendant_sensations.iter().map(|sensation| {
                serde_json::json!({
                    "parent": sensation.parent_id.is_some(),
                    "modality": sensation.modality.as_str(),
                    "payload_kind": sensation.payload_kind.as_str(),
                    "kind": sensation.kind,
                    "source": sensation.source,
                })
            }).collect::<Vec<_>>(),
            "vectors": instant.teacher_vectors.iter().map(|vector| {
                serde_json::json!({
                    "dim": vector.metadata.dim,
                    "model_id": vector.metadata.model_id,
                    "purpose": vector.metadata.purpose,
                    "collection": vector.metadata.collection,
                    "modality": vector.metadata.modality.as_str(),
                    "payload_kind": vector.metadata.payload_kind.as_str(),
                })
            }).collect::<Vec<_>>(),
            "coverage": instant.coverage(),
        })
    }

    #[test]
    fn vector_similarity_scores_matching_vectors() {
        let query = vec![vec![1.0, 0.0]];
        let records = vec![VectorArtifact::new(
            FACE_VECTOR_COLLECTION,
            "face-1",
            vec![1.0, 0.0],
        )];
        let score = max_vector_similarity(
            query.iter().map(Vec::as_slice).collect(),
            records.iter().collect(),
        );

        assert!((score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn memory_record_includes_embodied_vectors_and_lineage() {
        let now = Now::blank(123, BodySense::default());
        let mut frame = empty_frame(now);

        let primary = Sensation::primary(
            Modality::Vision,
            SensationSource {
                name: "camera.front".to_string(),
                device_id: Some("cam-1".to_string()),
                frame_id: Some("camera-frame-1".to_string()),
            },
            120,
            123,
            SensationPayload::image_metadata(640, 480, "rgb8", 921_600),
        )
        .with_summary("wide camera frame")
        .with_vector(VectorEmbedding::new(
            vec![1.0, 0.0, 0.0],
            "vision.sensation.test",
            Modality::Vision,
            SensationPayloadKind::ImageBytes,
            uuid::Uuid::nil(),
            124,
        ));
        let primary_id = primary.id;
        let mut primary = primary;
        primary.vector.as_mut().unwrap().source_sensation_id = primary_id;

        let crop_impression = Impression::new(
            "focus",
            "I focus on a bright crop.",
            vec![primary_id],
            120,
            123,
        )
        .with_confidence(0.82);
        let mut crop = Sensation::descendant(
            &primary,
            "vision.crop",
            SensationPayloadKind::Crop,
            serde_json::json!({"bbox": [10, 20, 64, 64]}),
            SensationMetadata {
                confidence: Some(0.91),
                labels: vec!["bright".to_string()],
                ..SensationMetadata::default()
            },
            "cropper",
        )
        .with_summary("bright crop")
        .with_vector(VectorEmbedding::new(
            vec![0.0, 1.0, 0.0],
            "vision.crop.test",
            Modality::Vision,
            SensationPayloadKind::Crop,
            uuid::Uuid::nil(),
            125,
        ))
        .with_impression(crop_impression.clone());
        crop.vector.as_mut().unwrap().source_sensation_id = crop.id;

        let experience_impression = Impression::new(
            "summary",
            "I see and focus on something bright.",
            vec![primary_id, crop.id],
            120,
            126,
        )
        .with_confidence(0.9);
        let mut experience = Experience::new(
            "visual_focus",
            "I see and focus on something bright.",
            vec![crop_impression.id, experience_impression.id],
            vec![primary_id, crop.id],
            120,
            126,
        );
        experience.summary_impression =
            Some(experience_impression.clone().for_experience(experience.id));

        frame.sensations = vec![primary.clone(), crop.clone()];
        frame.impressions = vec![crop_impression.clone(), experience_impression.clone()];
        frame.experiences = vec![experience.clone()];

        let record = memory_record_from_frame(&frame).unwrap();

        assert_eq!(record.sensation_vectors.len(), 2);
        assert_eq!(record.experience_vectors.len(), 1);
        assert!(record
            .sensation_vectors
            .iter()
            .all(|artifact| artifact.collection == SENSATION_VECTOR_COLLECTION));
        assert_eq!(
            record.experience_vectors[0].collection,
            EXPERIENCE_VECTOR_COLLECTION
        );

        let crop_id = crop.id.to_string();
        let crop_artifact = record
            .sensation_vectors
            .iter()
            .find(|artifact| artifact.source_id.as_deref() == Some(crop_id.as_str()))
            .unwrap();
        let crop_payload = record
            .vector_payloads
            .get(&vector_payload_key(crop_artifact))
            .unwrap();
        assert_eq!(
            crop_payload["parent_sensation_id"],
            serde_json::json!(primary_id.to_string())
        );
        assert_eq!(crop_payload["payload_kind"], "crop");
        assert_eq!(crop_payload["model_id"], "vision.crop.test");
        assert_eq!(crop_payload["dim"], 3);

        let exp_payload = record
            .vector_payloads
            .get(&vector_payload_key(&record.experience_vectors[0]))
            .unwrap();
        assert_eq!(
            exp_payload["summary_impression_text"],
            "I see and focus on something bright."
        );
        assert_eq!(exp_payload["experience_id"], experience.id.to_string());

        assert!(record.graph_relationships.iter().any(|edge| {
            edge.from == format!("frame:{}", frame.id)
                && edge.to == experience_node_id(experience.id)
                && edge.relationship == "HAS_EXPERIENCE"
        }));
        assert!(record.graph_relationships.iter().any(|edge| {
            edge.from == experience_node_id(experience.id)
                && edge.to == sensation_node_id(crop.id)
                && edge.relationship == "INTEGRATES_SENSATION"
        }));
        assert!(record.graph_relationships.iter().any(|edge| {
            edge.from == experience_node_id(experience.id)
                && edge.to == impression_node_id(experience_impression.id)
                && edge.relationship == "INTEGRATES_IMPRESSION"
        }));
        assert!(record.graph_relationships.iter().any(|edge| {
            edge.from == sensation_node_id(crop.id)
                && edge.to == sensation_node_id(primary_id)
                && edge.relationship == "DERIVED_FROM_SENSATION"
        }));
        assert!(record.graph_relationships.iter().any(|edge| {
            edge.from == sensation_node_id(crop.id)
                && edge.to == impression_node_id(crop_impression.id)
                && edge.relationship == "HAS_IMPRESSION"
        }));
        assert!(record.graph_relationships.iter().any(|edge| {
            edge.from == experience_node_id(experience.id)
                && edge.relationship == "HAS_FUSED_VECTOR"
        }));
    }

    #[test]
    fn memory_record_links_experience_to_place_objects_people_surfaces_and_recalls() {
        let mut now = now_at(1_000, 1.25, -0.25);
        now.memory.places_visited = 3;
        now.memory.place_familiarity = 0.62;
        now.objects.observations.push(ObjectObservation {
            label: "Charging Dock".to_string(),
            class: ObjectClass::Charger,
            bearing_rad: 0.2,
            distance_m: Some(1.4),
            confidence: 0.83,
            source: ObjectObservationSource::Sim,
        });
        now.face.vectors.push(
            VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-link-1", vec![1.0, 0.0])
                .with_source_id("person:ada"),
        );
        now.voice.vectors.push(
            VectorArtifact::new(VOICE_VECTOR_COLLECTION, "voice-link-1", vec![0.0, 1.0])
                .with_source_id("person:ada"),
        );
        now.extensions.insert(
            "surface.scene_graph".to_string(),
            json!({
                "floor": {"id": "floor", "kind": "floor", "confidence": 0.91},
                "surfaces": [{"id": "wall-east", "kind": "vertical_plane", "confidence": 0.77}],
                "clusters": [{"id": "box-1", "confidence": 0.66}],
            }),
        );

        let mut frame = empty_frame(now);
        let source_sensation_id = uuid::Uuid::new_v4();
        let experience = Experience::new(
            "embodied.now",
            "I see the charging dock near the wall.",
            Vec::new(),
            vec![source_sensation_id],
            950,
            1_000,
        );
        let recalled = Experience::new(
            "embodied.now",
            "I previously found the dock here.",
            Vec::new(),
            Vec::new(),
            500,
            550,
        );
        let recalled_id = recalled.id;
        frame.recollections.push(RecalledExperience {
            experience: recalled,
            score: 0.72,
            original_frame_id: Some(uuid::Uuid::new_v4()),
            original_vector_ids: vec!["experiences:prior".to_string()],
            sensation: Sensation::primary(
                Modality::Memory,
                SensationSource::default(),
                500,
                1_000,
                SensationPayload::structured(json!({})),
            ),
        });
        frame.experiences = vec![experience.clone()];

        let record = memory_record_from_frame(&frame).unwrap();
        let stored_experience = record.experience.as_ref().expect("stored experience");
        let links = &stored_experience.memory_links;

        assert!(links.iter().any(|link| {
            link.relation == "occurred_at_place"
                && link.target_id == place_id_for_pose(frame.now.body.odometry)
                && link.score >= 0.62
        }));
        assert!(links.iter().any(|link| {
            link.relation == "observed_object"
                && link.target_id == "object:sim:charger:charging-dock"
                && (link.score - 0.83).abs() < f32::EPSILON
        }));
        assert!(links
            .iter()
            .any(|link| link.relation == "saw_face" && link.target_id == "person:ada"));
        assert!(links
            .iter()
            .any(|link| link.relation == "heard_voice" && link.target_id == "person:ada"));
        assert!(links.iter().any(|link| {
            link.relation == "near_surface" && link.target_id == "surface:wall-east"
        }));
        assert!(links.iter().any(|link| {
            link.relation == "similar_to_experience"
                && link.target_id == experience_node_id(recalled_id)
                && (link.score - 0.72).abs() < f32::EPSILON
        }));

        let graph_experience_id = experience_node_id(experience.id);
        let object_edge = record
            .graph_relationships
            .iter()
            .find(|edge| {
                edge.from == graph_experience_id
                    && edge.to == "object:sim:charger:charging-dock"
                    && edge.relationship == "observed_object"
            })
            .expect("object memory link edge");
        assert!((object_edge.score - 0.83).abs() < f32::EPSILON);
        assert_eq!(object_edge.payload["class"], "charger");
        assert!(record.graph_entities.iter().any(|entity| {
            entity.id == "object:sim:charger:charging-dock" && entity.has_label("Object")
        }));
    }

    #[test]
    fn qdrant_point_ids_are_stable() {
        assert_eq!(
            stable_qdrant_point_id("faces", "frame:face:0"),
            stable_qdrant_point_id("faces", "frame:face:0")
        );
        assert_ne!(
            stable_qdrant_point_id("faces", "frame:face:0"),
            stable_qdrant_point_id("voices", "frame:face:0")
        );
    }

    #[tokio::test]
    async fn qdrant_vector_store_upserts_face_vectors_into_faces_collection() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::sync::mpsc;
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test qdrant");
        let addr = listener.local_addr().expect("test qdrant addr");
        let (tx, rx) = mpsc::channel();
        let server = thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().expect("accept qdrant request");
                let mut request = Vec::new();
                let mut buffer = [0_u8; 1024];
                loop {
                    let read = stream.read(&mut buffer).expect("read qdrant request");
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..read]);
                    if request.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let request_text = String::from_utf8_lossy(&request).to_string();
                tx.send(request_text).expect("send qdrant request");
                let body = r#"{"result":true,"status":"ok"}"#;
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                    body.len(),
                    body
                )
                .expect("write qdrant response");
            }
        });

        let store = QdrantVectorStore::new(QdrantConfig {
            url: format!("http://{addr}"),
        });
        let record = MemoryRecord {
            frame_id: uuid::Uuid::new_v4(),
            t_ms: 100,
            summary: "face frame".to_string(),
            graph_entities: Vec::new(),
            graph_relationships: Vec::new(),
            scene_vectors: Vec::new(),
            face_vectors: vec![VectorArtifact::new(
                FACE_VECTOR_COLLECTION,
                "face-qdrant",
                vec![0.1, 0.2],
            )
            .with_model("test.face.detector")
            .with_source_frame_id("eye-frame")],
            object_vectors: Vec::new(),
            voice_vectors: Vec::new(),
            sensation_vectors: Vec::new(),
            experience_vectors: Vec::new(),
            vector_payloads: BTreeMap::new(),
            battery: 1.0,
            active_goal: None,
            chosen_action: None,
            warning: None,
            experience: None,
            temporal_context: TemporalContext::default(),
            social_world: SocialWorldSnapshot::default(),
            epistemic_state: EpistemicSnapshot::default(),
        };

        store.upsert_vectors(&record).await.expect("upsert vectors");

        let create = rx.recv().expect("collection create request");
        let upsert = rx.recv().expect("point upsert request");
        server.join().expect("qdrant mock server");
        assert!(create.starts_with("PUT /collections/faces "));
        assert!(upsert.starts_with("PUT /collections/faces/points?wait=true "));
    }

    #[test]
    fn neo4j_http_url_can_be_derived_from_bolt_uri() {
        assert_eq!(
            neo4j_http_url_from_uri("bolt://neo4j:7687"),
            Some("http://neo4j:7474".to_string())
        );
        assert_eq!(
            neo4j_http_url_from_uri("neo4j://localhost:7687"),
            Some("http://localhost:7474".to_string())
        );
        assert_eq!(neo4j_http_url_from_uri("http://localhost:7474"), None);
    }

    #[test]
    fn neo4j_relationship_params_serialize_nested_payloads() {
        let record = MemoryRecord {
            frame_id: uuid::Uuid::new_v4(),
            t_ms: 1_000,
            summary: "place memory".to_string(),
            graph_entities: Vec::new(),
            graph_relationships: vec![GraphEdge {
                id: None,
                from: "embodied_experience:test".to_string(),
                to: "place:grid:0:0".to_string(),
                relationship: "occurred_at_place".to_string(),
                summary: Some("place near x=0.0m y=0.0m".to_string()),
                score: 1.0,
                payload: json!({
                    "target_kind": "place",
                    "text": "place near x=0.0m y=0.0m",
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "heading_rad": 0.0,
                }),
            }],
            scene_vectors: Vec::new(),
            face_vectors: Vec::new(),
            object_vectors: Vec::new(),
            voice_vectors: Vec::new(),
            sensation_vectors: Vec::new(),
            experience_vectors: Vec::new(),
            vector_payloads: BTreeMap::new(),
            battery: 1.0,
            active_goal: None,
            chosen_action: None,
            warning: None,
            experience: None,
            temporal_context: TemporalContext::default(),
            social_world: SocialWorldSnapshot::default(),
            epistemic_state: EpistemicSnapshot::default(),
        };

        let params = neo4j_relationship_params(&record);
        let payload_json = params[0]["payload_json"]
            .as_str()
            .expect("payload serialized as string");
        let payload: serde_json::Value =
            serde_json::from_str(payload_json).expect("payload_json is valid json");

        assert!(params[0].get("payload").is_none());
        assert!(params[0]["edge_id"]
            .as_str()
            .is_some_and(|id| id.starts_with("graph-edge:")));
        assert_eq!(payload["target_kind"], "place");
        assert_eq!(payload["heading_rad"], 0.0);
    }

    #[test]
    fn neo4j_intelligence_params_preserve_uncertainty_and_reviews() {
        let feature = Feature::new(
            pete_core::FeatureType::FaceObservation,
            pete_core::FeatureModality::Vision,
            1_000,
            0.74,
            pete_core::Provenance::direct(),
        )
        .with_source_frame("frame-a");
        let feature_id = feature.id;
        let candidate = BindingCandidate {
            left_cluster_id: "cluster:face:a".to_string(),
            right_cluster_id: "cluster:voice:b".to_string(),
            relation: BindingRelation::LikelySameEntity,
            evidence: vec![
                BindingEvidence {
                    kind: BindingEvidenceKind::VectorSimilarity,
                    score: 0.68,
                    reason: "face and voice recur close together".to_string(),
                },
                BindingEvidence {
                    kind: BindingEvidenceKind::Contradiction,
                    score: 0.82,
                    reason: "same identity appears in incompatible places".to_string(),
                },
            ],
            confidence: 0.41,
            decision: BindingDecision::HoldAmbiguous,
            reason: "cross-modal identity needs review".to_string(),
        };
        let document = GraphIntelligenceDocument {
            id: "graph-doc:test".to_string(),
            t_ms: 1_000,
            frame_id: Some("frame-a".to_string()),
            provenance: "unit-test".to_string(),
            confidence: 0.6,
            reason: "testing typed graph persistence".to_string(),
            source_frame_ids: vec!["frame-a".to_string()],
            features: vec![feature],
            clusters: vec![
                DiscoveredCluster::new(
                    "cluster:face:a",
                    Modality::Vision,
                    DiscoveredClusterKind::Face,
                    1_000,
                    0.7,
                )
                .with_feature_ids(vec![feature_id]),
                DiscoveredCluster::new(
                    "cluster:voice:b",
                    Modality::Audio,
                    DiscoveredClusterKind::Voice,
                    1_000,
                    0.65,
                ),
            ],
            binding_candidates: vec![candidate],
            constellations: vec![Constellation {
                id: "constellation:person:maybe".to_string(),
                kind_hint: Some("person".to_string()),
                member_cluster_ids: vec![
                    "cluster:face:a".to_string(),
                    "cluster:voice:b".to_string(),
                ],
                member_binding_ids: vec![
                    "binding-edge:cluster-face-a:likely_same_entity:cluster-voice-b".to_string(),
                ],
                supporting_feature_ids: vec![feature_id],
                supporting_entity_ids: Vec::new(),
                supporting_place_cells: Vec::new(),
                confidence: 0.48,
                stability: 0.2,
                prediction_value: 0.1,
                first_seen_ms: 1_000,
                last_seen_ms: 1_000,
                evidence_count: 2,
                state: ConstellationState::Ambiguous,
                notes: vec!["voice may belong to a different person".to_string()],
            }],
            llm_reviews: vec![LlmReviewRecord {
                id: "llm-review:1".to_string(),
                target_id: "binding-candidate:cluster-face-a:likely_same_entity:cluster-voice-b"
                    .to_string(),
                target_kind: ActiveLearningTargetKind::BindingCandidate,
                confidence: 0.55,
                t_ms: 1_010,
                critique: "ask whether the face and voice are the same person".to_string(),
                contradictions: vec!["place mismatch".to_string()],
                suggested_questions: vec!["Is this the same person?".to_string()],
            }],
            human_reviews: vec![HumanReviewRecord {
                id: "human-review:1".to_string(),
                target_id: "hypothesis:face_identity:track-a:pete".to_string(),
                target_kind: ActiveLearningTargetKind::TrackingHypothesis,
                confidence: 0.9,
                t_ms: 1_020,
                confirmation: "not the same person".to_string(),
                reviewer: Some("tester".to_string()),
            }],
            review_records: vec![GraphReviewRecord {
                id: "graph-review:1".to_string(),
                target_id: "constellation:person:maybe".to_string(),
                review_kind: "mutually_exclusive_label".to_string(),
                severity: 0.8,
                confidence: 0.7,
                t_ms: 1_030,
                reason: "constellation has conflicting person labels".to_string(),
                evidence_ids: vec!["evidence:place-mismatch".to_string()],
                state: "open".to_string(),
            }],
            ..GraphIntelligenceDocument::default()
        };

        let params = neo4j_intelligence_params(&document);
        assert_eq!(params["document"]["source_frame_ids"][0], "frame-a");
        assert_eq!(
            params["binding_candidates"][0]["decision"],
            "hold_ambiguous"
        );
        assert_eq!(params["candidate_evidence"].as_array().unwrap().len(), 2);
        assert_eq!(
            params["candidate_evidence"][1]["current_state"],
            "contradictory"
        );
        assert_eq!(params["constellations"][0]["current_state"], "ambiguous");
        assert_eq!(params["llm_reviews"][0]["current_state"], "needs_review");
        assert_eq!(params["human_reviews"][0]["current_state"], "confirmed");
        assert_eq!(
            params["review_records"][0]["state"],
            serde_json::Value::Null
        );
        assert_eq!(params["review_records"][0]["current_state"], "open");
        assert!(neo4j_intelligence_upsert_statements()
            .iter()
            .any(|statement| statement.contains("COMPETES_WITH")));
        assert!(neo4j_intelligence_upsert_statements()
            .iter()
            .any(|statement| statement.contains("FAILED_WITH")));
    }

    fn now_at(t_ms: u64, x_m: f32, y_m: f32) -> Now {
        let mut body = BodySense::default();
        body.odometry.x_m = x_m;
        body.odometry.y_m = y_m;
        Now::blank(t_ms, body)
    }

    fn add_sim_world_extension(now: &mut Now, charger_near: f32, charger_visible: f32) {
        now.extensions.insert(
            "sim.world".to_string(),
            serde_json::json!({
                "schema_version": 1,
                "values": [8.0, 8.0, 1.0, charger_near, charger_visible],
            }),
        );
    }

    #[test]
    fn place_quantization_is_stable_inside_cell() {
        let memory = PlaceMemory::new();

        assert_eq!(memory.quantize(1.01, 2.01), memory.quantize(1.49, 2.49));
        assert_ne!(memory.quantize(1.49, 2.49), memory.quantize(1.51, 2.49));
    }

    #[test]
    fn danger_update_increases_danger_score() {
        let mut memory = PlaceMemory::new();
        let mut now = now_at(100, 1.0, 1.0);
        now.body.flags.bump_left = true;

        let features = memory.observe_now(&now);

        assert!(features.current_place_danger >= 0.9);
        assert_eq!(features.places_visited, 1);
    }

    #[test]
    fn charge_update_increases_charge_score() {
        let mut memory = PlaceMemory::new();
        let mut now = now_at(100, 1.0, 1.0);
        add_sim_world_extension(&mut now, 0.8, 0.2);

        let features = memory.observe_now(&now);

        assert!(features.current_place_charge >= 0.8);
    }

    #[test]
    fn social_update_increases_social_score() {
        let mut memory = PlaceMemory::new();
        let mut now = now_at(100, 1.0, 1.0);
        now.face.vectors.push(VectorArtifact::new(
            FACE_VECTOR_COLLECTION,
            "face-social",
            vec![1.0, 0.0],
        ));

        let features = memory.observe_now(&now);

        assert!(features.current_place_social >= 1.0);
    }

    #[test]
    fn scores_decay_between_observations() {
        let mut memory = PlaceMemory::new();
        let mut now = now_at(100, 1.0, 1.0);
        now.body.flags.bump_left = true;
        let first = memory.observe_now(&now);

        let second = memory.observe_now(&now_at(10_100, 1.0, 1.0));

        assert!(second.current_place_danger < first.current_place_danger);
    }

    #[test]
    fn recall_returns_nearby_charger_direction() {
        let mut memory = PlaceMemory::new();
        let mut charge_now = now_at(100, 1.0, 0.0);
        add_sim_world_extension(&mut charge_now, 1.0, 0.0);
        memory.observe_now(&charge_now);

        let features = memory.features_at(0.0, 0.0);

        let direction = features.nearby_best_charge_direction_rad.unwrap();
        assert!(direction.abs() < 0.4);
    }

    #[test]
    fn recall_returns_safe_frontier_direction_and_confidence() {
        let mut memory = PlaceMemory::new();
        memory.observe_now(&now_at(100, 0.0, 0.0));
        memory.observe_now(&now_at(200, 0.0, 1.0));

        let features = memory.features_at(0.0, 0.0);

        let direction = features.nearby_frontier_direction_rad.unwrap();
        assert!(direction > 0.5);
        assert!(features.current_place_confidence > 0.0);
    }

    #[test]
    fn semantic_cells_keep_vector_and_action_associations() {
        let mut memory = PlaceMemory::new();
        let mut now = now_at(100, 1.0, 1.0);
        now.eye.scene_vectors.push(VectorArtifact::new(
            SCENE_VECTOR_COLLECTION,
            "scene-place-1",
            vec![0.0, 1.0],
        ));
        now.face.vectors.push(VectorArtifact::new(
            FACE_VECTOR_COLLECTION,
            "face-place-1",
            vec![1.0, 0.0],
        ));
        let mut frame = empty_frame(now);
        frame.chosen_action = Some(ActionPrimitive::Inspect {
            target: pete_actions::InspectTarget::Charger,
        });
        frame.reward.value = 0.4;

        memory.observe_frame(&frame);
        let cell = memory
            .cells
            .values()
            .next()
            .expect("semantic cell should be created");

        assert_eq!(cell.occupancy_cell, Some(cell.key));
        assert_eq!(cell.associated_scene_vectors, vec!["scene-place-1"]);
        assert_eq!(cell.associated_face_vectors, vec!["face-place-1"]);
        assert_eq!(cell.successful_actions.len(), 1);
        assert_eq!(cell.successful_actions[0].count, 1);
    }

    #[test]
    fn semantic_overlay_exposes_scores_for_dashboard() {
        let mut memory = PlaceMemory::new();
        let mut danger_now = now_at(100, 1.0, 1.0);
        danger_now.body.flags.bump_left = true;
        memory.observe_now(&danger_now);
        let mut charge_now = now_at(200, 1.5, 1.0);
        add_sim_world_extension(&mut charge_now, 1.0, 0.0);
        memory.observe_now(&charge_now);

        let overlay = memory.semantic_overlay_at(1.0, 1.0);

        assert_eq!(overlay.schema_version, 1);
        assert!(overlay.current.is_some());
        assert!(!overlay.danger_cells.is_empty());
        assert!(!overlay.charge_cells.is_empty());
    }

    #[test]
    fn place_recognition_scores_revisits_above_unrelated_locations() {
        let mut memory = PlaceMemory::new();
        let first_frame_id = uuid::Uuid::new_v4();
        let unrelated_frame_id = uuid::Uuid::new_v4();
        let mut first = now_at(100, 1.0, 1.0);
        first.eye.scene_vectors.push(
            VectorArtifact::new(SCENE_VECTOR_COLLECTION, "scene-first", vec![1.0, 0.0, 0.0])
                .with_source_frame_id(first_frame_id.to_string()),
        );
        let mut unrelated = now_at(200, 4.0, 1.0);
        unrelated.eye.scene_vectors.push(
            VectorArtifact::new(
                SCENE_VECTOR_COLLECTION,
                "scene-unrelated",
                vec![0.0, 1.0, 0.0],
            )
            .with_source_frame_id(unrelated_frame_id.to_string()),
        );
        memory.observe_now(&first);
        memory.observe_now(&first);
        memory.observe_now(&unrelated);

        let query = VectorArtifact::new(
            SCENE_VECTOR_COLLECTION,
            "scene-query",
            vec![0.98, 0.02, 0.0],
        );
        let candidates = memory.recognize_places(
            Some(memory.quantize(1.02, 1.02)),
            &[query],
            PLACE_RECOGNITION_MIN_CONFIDENCE,
            5,
        );

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].source_vector_id, "scene-first");
        assert!(matches!(
            candidates[0].kind,
            PlaceRecognitionKind::SamePlace
        ));
        assert!(candidates[0].similarity > 0.99);
        assert!(candidates[0].confidence >= PLACE_RECOGNITION_MIN_CONFIDENCE);
        assert_eq!(
            candidates[0].source_frame_id.as_deref(),
            Some(first_frame_id.to_string().as_str())
        );
    }

    #[test]
    fn observe_now_stamps_scene_vectors_with_frame_id_extension() {
        let mut memory = PlaceMemory::new();
        let mut observed = now_at(100, 1.0, 1.0);
        observed.extensions.insert(
            "frame_id".to_string(),
            serde_json::Value::String("live-frame-1".to_string()),
        );
        observed.eye.scene_vectors.push(VectorArtifact::new(
            SCENE_VECTOR_COLLECTION,
            "scene-observed",
            vec![1.0, 0.0, 0.0],
        ));
        memory.observe_now(&observed);

        let query =
            VectorArtifact::new(SCENE_VECTOR_COLLECTION, "scene-query", vec![1.0, 0.0, 0.0]);
        let candidates = memory.recognize_places(
            Some(memory.quantize(1.0, 1.0)),
            &[query],
            PLACE_RECOGNITION_MIN_CONFIDENCE,
            5,
        );

        assert_eq!(candidates.len(), 1);
        assert_eq!(
            candidates[0].source_frame_id.as_deref(),
            Some("live-frame-1")
        );
        assert_eq!(
            candidates[0].source_instant_frame_id.as_deref(),
            Some("live-frame-1")
        );
    }

    #[test]
    fn place_recognition_rejects_low_confidence_candidates() {
        let mut memory = PlaceMemory::new();
        let mut observed = now_at(100, 2.0, 1.0);
        observed.eye.scene_vectors.push(VectorArtifact::new(
            SCENE_VECTOR_COLLECTION,
            "scene-observed",
            vec![1.0, 0.0, 0.0],
        ));
        memory.observe_now(&observed);

        let query =
            VectorArtifact::new(SCENE_VECTOR_COLLECTION, "scene-query", vec![0.0, 1.0, 0.0]);
        let candidates = memory.recognize_places(
            Some(memory.quantize(2.0, 1.0)),
            &[query],
            PLACE_RECOGNITION_MIN_CONFIDENCE,
            5,
        );

        assert!(candidates.is_empty());
    }

    #[tokio::test]
    async fn recall_and_semantic_overlay_include_place_candidates() {
        let store = InMemoryExperienceStore::new();
        let mut frame = empty_frame(now_at(100, 1.0, 1.0));
        frame.now.eye.scene_vectors.push(
            VectorArtifact::new(SCENE_VECTOR_COLLECTION, "scene-stored", vec![1.0, 0.0])
                .with_source_frame_id(frame.id.to_string()),
        );
        store.observe_frame(&frame).await.unwrap();

        let recall = store
            .recall(RecallQuery {
                pose: Some(frame.now.body.odometry),
                scene_vectors: vec![VectorArtifact::new(
                    SCENE_VECTOR_COLLECTION,
                    "scene-query",
                    vec![1.0, 0.0],
                )],
                battery: frame.now.body.battery_level,
                ..RecallQuery::default()
            })
            .await
            .unwrap();

        assert_eq!(recall.place_recognition_candidates.len(), 1);
        let candidate = &recall.place_recognition_candidates[0];
        assert_eq!(candidate.source_vector_id, "scene-stored");
        assert_eq!(candidate.query_vector_id.as_deref(), Some("scene-query"));
        assert!(candidate.confidence >= PLACE_RECOGNITION_MIN_CONFIDENCE);
        assert!(recall
            .semantic_map
            .as_ref()
            .is_some_and(|overlay| !overlay.place_recognition_candidates.is_empty()));
    }

    #[tokio::test]
    async fn recall_loop_closure_candidates_use_place_memory_evidence() {
        let store = InMemoryExperienceStore::new();
        let mut observed = now_at(100, 1.0, 1.0);
        observed.eye.scene_vectors.push(
            VectorArtifact::new(SCENE_VECTOR_COLLECTION, "scene-stored", vec![1.0, 0.0])
                .with_source_frame_id("stored-frame"),
        );
        store.observe_now(&observed).await.unwrap();

        let candidates = store
            .loop_closure_candidates(
                &RecallQuery {
                    pose: Some(now_at(200, 1.1, 1.0).body.odometry),
                    scene_vectors: vec![VectorArtifact::new(
                        SCENE_VECTOR_COLLECTION,
                        "scene-query",
                        vec![1.0, 0.0],
                    )],
                    ..RecallQuery::default()
                },
                PLACE_RECOGNITION_MIN_CONFIDENCE,
                5,
            )
            .await
            .unwrap();

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].source_vector_id, "scene-stored");
        assert_eq!(
            candidates[0].source_frame_id.as_deref(),
            Some("stored-frame")
        );
    }

    #[tokio::test]
    async fn place_recognition_uses_experience_latents_and_lineage() {
        let store = InMemoryExperienceStore::new();
        let mut frame = empty_frame(now_at(100, 1.0, 1.0));
        frame.now.range.beams = vec![0.4, 0.7, 1.0];
        frame.now.range.nearest_m = Some(0.4);
        frame.now.objects.observations.push(ObjectObservation {
            label: "charger alcove".to_string(),
            class: ObjectClass::Landmark,
            confidence: 0.9,
            source: ObjectObservationSource::Sim,
            ..ObjectObservation::default()
        });
        let sensation_id = uuid::Uuid::new_v4();
        let experience = Experience::new(
            "embodied.place",
            "charger alcove near the wall",
            Vec::new(),
            vec![sensation_id],
            80,
            100,
        );
        frame.z = Some(ExperienceLatent {
            t_ms: frame.t_ms,
            z: vec![1.0, 0.0, 0.0, 0.0],
            confidence: 0.95,
            ..ExperienceLatent::default()
        });
        let experience_id = experience.id.to_string();
        frame.experiences.push(experience);
        store.observe_frame(&frame).await.unwrap();

        let query_now = now_at(140, 1.02, 1.01);
        let latent = ExperienceLatent {
            t_ms: 140,
            z: vec![0.99, 0.01, 0.0, 0.0],
            confidence: 0.95,
            ..ExperienceLatent::default()
        };
        let mut query = RecallQuery::from_now(&query_now);
        query.place_recognition_input = Some(place_recognition_input_from_query_now(
            &query_now,
            Some(&latent),
            "test-query",
        ));
        let recall = store.recall(query).await.unwrap();

        let candidate = recall
            .place_recognition_candidates
            .first()
            .expect("place candidate from learned latent");
        assert_eq!(
            candidate.source_experience_id.as_deref(),
            Some(experience_id.as_str())
        );
        assert_eq!(
            candidate.source_instant_frame_id.as_deref(),
            Some(frame.id.to_string().as_str())
        );
        assert!(candidate.source_vector_id.contains(":experience-latent"));
        assert!(candidate.reason.contains("confidence="));
    }

    #[test]
    fn place_recognition_report_marks_not_enough_evidence_and_rejections() {
        let mut memory = PlaceMemory::new();
        let mut observed = now_at(100, 2.0, 1.0);
        observed.eye.scene_vectors.push(VectorArtifact::new(
            SCENE_VECTOR_COLLECTION,
            "scene-observed",
            vec![1.0, 0.0, 0.0],
        ));
        memory.observe_now(&observed);

        let empty = memory.recognize_places_report(
            Some(memory.quantize(2.0, 1.0)),
            &[],
            PLACE_RECOGNITION_MIN_CONFIDENCE,
            5,
        );
        assert!(empty.not_enough_evidence.is_some());

        let low = memory.recognize_places_report(
            Some(memory.quantize(2.0, 1.0)),
            &[VectorArtifact::new(
                SCENE_VECTOR_COLLECTION,
                "scene-query",
                vec![0.0, 1.0, 0.0],
            )],
            PLACE_RECOGNITION_MIN_CONFIDENCE,
            5,
        );
        assert!(low.candidates.is_empty());
        assert_eq!(low.rejected.len(), 1);
        assert!(low.rejected[0].reason.contains("below threshold"));
    }

    #[test]
    fn ledger_replay_reconstructs_semantic_map_cells() {
        let mut first = empty_frame(now_at(100, 1.0, 1.0));
        first.now.body.flags.bump_left = true;
        first.chosen_action = Some(ActionPrimitive::Go {
            intensity: 0.3,
            duration_ms: 200,
        });
        first.reward.value = -0.6;
        let mut second = empty_frame(now_at(200, 1.0, 1.0));
        second.chosen_action = first.chosen_action.clone();
        second.reward.value = 0.3;

        let memory = place_memory_from_frames(&[first, second]);
        let report = memory.report();
        let cell = memory
            .cells
            .values()
            .next()
            .expect("replayed semantic cell");

        assert_eq!(report.places_visited, 1);
        assert!(cell.danger_score > 0.0);
        assert_eq!(cell.failed_actions.len(), 1);
        assert_eq!(cell.successful_actions.len(), 1);
        assert!(cell.novelty_score < 1.0);
    }

    #[test]
    fn compact_memory_features_serialize() {
        let features = PlaceMemoryFeatures {
            current_place_danger: 0.2,
            current_place_charge: 0.7,
            current_place_social: 0.3,
            current_place_novelty: 0.4,
            current_place_familiarity: 0.5,
            current_place_confidence: 0.6,
            nearby_best_charge_direction_rad: Some(1.0),
            nearby_best_safe_direction_rad: None,
            nearby_frontier_direction_rad: Some(-0.5),
            places_visited: 3,
        };

        let json = serde_json::to_value(&features).unwrap();

        let charge = json["current_place_charge"].as_f64().unwrap();
        assert!((charge - 0.7).abs() < 0.000_001);
        assert_eq!(json["places_visited"], 3);
    }

    #[test]
    fn entity_constellation_candidates_are_generated_and_gated_by_overlap() {
        let mut memory = PlaceMemory::new();

        // Observe a place with several entities at (2.0, 1.0)
        let mut first = now_at(100, 2.0, 1.0);
        first.objects.observations.push(ObjectObservation {
            label: "chair".to_string(),
            class: ObjectClass::Unknown,
            bearing_rad: 0.1,
            distance_m: Some(1.0),
            confidence: 0.9,
            source: ObjectObservationSource::Sim,
        });
        first.objects.observations.push(ObjectObservation {
            label: "desk".to_string(),
            class: ObjectClass::Unknown,
            bearing_rad: 0.2,
            distance_m: Some(1.5),
            confidence: 0.8,
            source: ObjectObservationSource::Sim,
        });
        memory.observe_now(&first);

        // Query from a different cell (5.0, 1.0) with the same entities -> strong overlap
        let current_key_different = Some(memory.quantize(5.0, 1.0));
        let strong_labels = vec!["chair".to_string(), "desk".to_string()];
        let candidates =
            memory.recognize_entity_constellations(current_key_different, &strong_labels, 0.0, 5);

        assert_eq!(candidates.len(), 1, "should find the overlapping cell");
        assert!(matches!(
            candidates[0].kind,
            PlaceRecognitionKind::EntityConstellation
        ));
        assert!(
            (candidates[0].similarity - 1.0).abs() < 0.001,
            "full overlap"
        );
        assert!(candidates[0].confidence > 0.0);
        assert!(candidates[0].reason.contains("entity overlap"));

        // Query from a different cell with no shared entities -> no candidates
        let weak_labels = vec!["robot".to_string(), "unknown_label".to_string()];
        let no_candidates =
            memory.recognize_entity_constellations(current_key_different, &weak_labels, 0.0, 5);
        assert!(
            no_candidates.is_empty(),
            "no shared entities means no candidates"
        );

        // Empty labels -> no candidates
        let empty = memory.recognize_entity_constellations(current_key_different, &[], 0.0, 5);
        assert!(empty.is_empty(), "empty query labels yields no candidates");
    }

    #[test]
    fn entity_constellation_skips_current_cell_and_low_confidence() {
        let mut memory = PlaceMemory::new();
        let mut obs = now_at(100, 2.0, 1.0);
        obs.objects.observations.push(ObjectObservation {
            label: "lamp".to_string(),
            class: ObjectClass::Unknown,
            bearing_rad: 0.0,
            distance_m: Some(0.8),
            confidence: 0.9,
            source: ObjectObservationSource::Sim,
        });
        memory.observe_now(&obs);

        let current_key = Some(memory.quantize(2.0, 1.0));
        let labels = vec!["lamp".to_string()];

        // Should not return self-match
        let self_candidates = memory.recognize_entity_constellations(current_key, &labels, 0.0, 5);
        assert!(
            self_candidates.is_empty(),
            "should not return the current cell as a loop candidate"
        );

        // High confidence gate filters low-overlap candidates
        let different_key = Some(memory.quantize(10.0, 10.0));
        let high_gate = memory.recognize_entity_constellations(different_key, &labels, 0.99, 5);
        assert!(
            high_gate.is_empty(),
            "low confidence candidate should be filtered by high gate"
        );
    }

    // ── EntityMemory tests ───────────────────────────────────────────────────

    fn make_object_observation(
        label: &str,
        class: ObjectClass,
        confidence: f32,
    ) -> ObjectObservation {
        ObjectObservation {
            label: label.to_string(),
            class,
            bearing_rad: 0.0,
            distance_m: Some(1.0),
            confidence,
            source: ObjectObservationSource::Sim,
        }
    }

    #[test]
    fn entity_memory_repeated_observation_merges_not_duplicates() {
        let mut memory = EntityMemory::new();

        let mut now1 = now_at(100, 1.0, 1.0);
        now1.objects
            .observations
            .push(make_object_observation("chair", ObjectClass::Unknown, 0.8));
        memory.observe_now(&now1, Some(PlaceCellKey { x: 2, y: 2 }));

        let mut now2 = now_at(200, 1.0, 1.0);
        now2.objects
            .observations
            .push(make_object_observation("chair", ObjectClass::Unknown, 0.7));
        memory.observe_now(&now2, Some(PlaceCellKey { x: 2, y: 2 }));

        assert_eq!(
            memory.entities.len(),
            1,
            "repeated observation must merge, not duplicate"
        );
        let entity = memory.entities.values().next().unwrap();
        assert_eq!(entity.observation_count, 2);
        assert_eq!(entity.first_seen_ms, 100);
        assert_eq!(entity.last_seen_ms, 200);
    }

    #[test]
    fn entity_memory_confidence_increases_on_re_observation() {
        let mut memory = EntityMemory::new();

        let mut now1 = now_at(100, 1.0, 1.0);
        now1.objects
            .observations
            .push(make_object_observation("desk", ObjectClass::Unknown, 0.5));
        memory.observe_now(&now1, None);
        let confidence_after_first = memory.entities.values().next().unwrap().confidence;

        let mut now2 = now_at(200, 1.0, 1.0);
        now2.objects
            .observations
            .push(make_object_observation("desk", ObjectClass::Unknown, 0.9));
        memory.observe_now(&now2, None);
        let confidence_after_second = memory.entities.values().next().unwrap().confidence;

        assert!(
            confidence_after_second > confidence_after_first * 0.9,
            "confidence should remain stable or grow on re-observation: {confidence_after_first} -> {confidence_after_second}"
        );
    }

    #[test]
    fn entity_memory_stale_entity_transitions_to_occluded_then_vanished() {
        let mut memory = EntityMemory::new();

        let mut now1 = now_at(100, 0.0, 0.0);
        now1.objects
            .observations
            .push(make_object_observation("cup", ObjectClass::Unknown, 0.6));
        memory.observe_now(&now1, None);

        // Simulate many decay ticks by calling observe_now with no objects and a large time gap
        let mut stale = now_at(1_000_000, 0.0, 0.0); // ~10 000 ticks later
        stale.objects.observations.clear();
        memory.observe_now(&stale, None);

        let entity = memory.entities.values().next().unwrap();
        assert!(
            entity.lifecycle != EntityLifecycleState::Active,
            "entity should not remain Active after a long absence"
        );
    }

    #[test]
    fn entity_memory_report_counts_lifecycle_states() {
        let mut memory = EntityMemory::new();

        // Active entity
        let mut now1 = now_at(100, 0.0, 0.0);
        now1.objects
            .observations
            .push(make_object_observation("sofa", ObjectClass::Unknown, 0.9));
        memory.observe_now(&now1, None);

        // A second entity that will go stale
        let mut now2 = now_at(200, 5.0, 5.0);
        now2.objects
            .observations
            .push(make_object_observation("lamp", ObjectClass::Unknown, 0.6));
        memory.observe_now(&now2, None);

        // Age both entities hard
        let stale = now_at(10_000_000, 0.0, 0.0);
        memory.observe_now(&stale, None);

        let report = memory.report();
        assert_eq!(report.total_entities, 2);
        assert!(
            report.active_entities < 2,
            "both should have decayed away from Active"
        );
    }

    #[test]
    fn entity_memory_records_map_cell_linkage() {
        let mut memory = EntityMemory::new();
        let cell_a = PlaceCellKey { x: 3, y: 4 };
        let cell_b = PlaceCellKey { x: 7, y: 8 };

        let mut now1 = now_at(100, 0.0, 0.0);
        now1.objects.observations.push(make_object_observation(
            "robot",
            ObjectClass::Obstacle,
            0.85,
        ));
        memory.observe_now(&now1, Some(cell_a));

        let mut now2 = now_at(200, 3.5, 4.0);
        now2.objects.observations.push(make_object_observation(
            "robot",
            ObjectClass::Obstacle,
            0.8,
        ));
        memory.observe_now(&now2, Some(cell_b));

        let entity = memory.entities.values().next().unwrap();
        assert!(
            entity.location_cells.contains(&cell_a),
            "first observed cell must be linked"
        );
        assert!(
            entity.location_cells.contains(&cell_b),
            "second observed cell must be linked"
        );
    }

    #[test]
    fn entity_memory_revives_after_re_sighting() {
        let mut memory = EntityMemory::new();

        let mut now1 = now_at(100, 0.0, 0.0);
        now1.objects
            .observations
            .push(make_object_observation("box", ObjectClass::Unknown, 0.7));
        memory.observe_now(&now1, None);

        // Age it into Occluded/Vanished
        let stale = now_at(500_000, 0.0, 0.0);
        memory.observe_now(&stale, None);
        let lifecycle_before = memory.entities.values().next().unwrap().lifecycle.clone();

        // Re-observe
        let mut now3 = now_at(600_000, 0.0, 0.0);
        now3.objects
            .observations
            .push(make_object_observation("box", ObjectClass::Unknown, 0.8));
        memory.observe_now(&now3, None);

        let entity = memory.entities.values().next().unwrap();
        assert_eq!(
            entity.lifecycle,
            EntityLifecycleState::Active,
            "entity should revive to Active on re-sighting; was {lifecycle_before:?}"
        );
    }

    #[test]
    fn entity_constellation_strengthens_with_cross_modal_repetition() {
        let mut memory = EntityMemory::new();
        let mut now1 = now_at(100, 0.0, 0.0);
        now1.objects
            .observations
            .push(make_object_observation("pete", ObjectClass::Person, 0.8));
        now1.face.vectors.push(VectorArtifact::new(
            FACE_VECTOR_COLLECTION,
            "face-pete",
            vec![1.0, 0.0],
        ));
        now1.voice.vectors.push(VectorArtifact::new(
            VOICE_VECTOR_COLLECTION,
            "voice-pete",
            vec![1.0, 0.0],
        ));
        memory.observe_now(&now1, None);

        let entity_id = "entity:person:pete";
        let first_edge_evidence = memory
            .entities
            .get(entity_id)
            .and_then(|entity| entity.constellation.binding_edges.first())
            .map(|edge| (edge.evidence_count, edge.confidence))
            .expect("binding edge on first multimodal observation");

        let mut now2 = now_at(200, 0.1, 0.0);
        now2.objects
            .observations
            .push(make_object_observation("pete", ObjectClass::Person, 0.9));
        now2.face.vectors.push(VectorArtifact::new(
            FACE_VECTOR_COLLECTION,
            "face-pete",
            vec![1.0, 0.0],
        ));
        now2.voice.vectors.push(VectorArtifact::new(
            VOICE_VECTOR_COLLECTION,
            "voice-pete",
            vec![1.0, 0.0],
        ));
        memory.observe_now(&now2, None);

        let entity = memory.entities.get(entity_id).expect("person entity");
        assert!(
            entity.constellation.binding_edges.iter().any(|edge| {
                edge.evidence_count > first_edge_evidence.0
                    && edge.confidence > first_edge_evidence.1
            }),
            "repeated co-occurrence should strengthen at least one binding edge"
        );
    }

    #[test]
    fn entity_constellation_attaches_provisional_text_labels() {
        let mut memory = EntityMemory::new();
        let mut now = now_at(100, 0.0, 0.0);
        now.objects.observations.push(make_object_observation(
            "person-nearby",
            ObjectClass::Person,
            0.8,
        ));
        now.ear.transcript = Some("Travis".to_string());
        memory.observe_now(&now, None);

        let entity = memory
            .entities
            .get("entity:person:person-nearby")
            .expect("person entity");
        assert!(entity
            .modality_support
            .text_labels
            .contains(&"Travis".to_string()));
        assert_eq!(
            entity.display_name.as_deref(),
            Some("person-nearby"),
            "text remains provisional and does not override sensory label"
        );
        assert!(
            entity
                .constellation
                .binding_edges
                .iter()
                .any(|edge| edge.relation == BindingRelation::NamedBy),
            "named_by edge should connect text cluster to the entity constellation"
        );
    }

    #[test]
    fn face_vector_is_not_attached_to_every_active_person() {
        let mut memory = EntityMemory::new();
        let mut now = now_at(100, 0.0, 0.0);
        now.objects
            .observations
            .push(make_object_observation("ada", ObjectClass::Person, 0.8));
        now.objects
            .observations
            .push(make_object_observation("grace", ObjectClass::Person, 0.8));
        now.face.vectors.push(VectorArtifact::new(
            FACE_VECTOR_COLLECTION,
            "face-ambiguous",
            vec![1.0, 0.0],
        ));

        memory.observe_now(&now, Some(PlaceCellKey { x: 0, y: 0 }));

        assert!(memory
            .entities
            .values()
            .all(|entity| entity.modality_support.face_vector_ids.is_empty()));
        assert!(memory
            .report()
            .ambiguous_binding_candidates
            .iter()
            .any(|candidate| candidate.reason.contains("face vector close")));
    }

    #[test]
    fn ambiguous_face_observation_creates_competing_tracking_hypotheses() {
        let mut memory = EntityMemory::new();
        let mut now = now_at(100, 0.0, 0.0);
        now.objects
            .observations
            .push(make_object_observation("travis", ObjectClass::Person, 0.8));
        now.objects
            .observations
            .push(make_object_observation("tim", ObjectClass::Person, 0.8));
        now.face.vectors.push(VectorArtifact::new(
            FACE_VECTOR_COLLECTION,
            "face-ambiguous-ticket5",
            vec![1.0, 0.0],
        ));

        memory.observe_now(&now, Some(PlaceCellKey { x: 0, y: 0 }));

        let family = memory
            .tracking_hypotheses
            .values()
            .filter(|hypothesis| hypothesis.family_id == "face-identity:face-ambiguous-ticket5")
            .collect::<Vec<_>>();
        assert_eq!(
            family.len(),
            3,
            "two known-person hypotheses plus one unknown-person hypothesis should remain visible"
        );
        assert!(family
            .iter()
            .any(|hypothesis| { hypothesis.target_id.as_deref() == Some("entity:person:travis") }));
        assert!(family
            .iter()
            .any(|hypothesis| hypothesis.target_id.as_deref() == Some("entity:person:tim")));
        assert!(family
            .iter()
            .any(|hypothesis| hypothesis.target_id.is_none()));
        assert!(family
            .iter()
            .any(|hypothesis| { matches!(hypothesis.state, HypothesisState::NeedsReview) }));
        assert!(memory
            .entities
            .values()
            .all(|entity| entity.modality_support.face_vector_ids.is_empty()));
    }

    #[test]
    fn voice_vector_is_not_attached_to_every_active_person() {
        let mut memory = EntityMemory::new();
        let mut now = now_at(100, 0.0, 0.0);
        now.objects
            .observations
            .push(make_object_observation("ada", ObjectClass::Person, 0.8));
        now.objects
            .observations
            .push(make_object_observation("grace", ObjectClass::Person, 0.8));
        now.voice.vectors.push(VectorArtifact::new(
            VOICE_VECTOR_COLLECTION,
            "voice-ambiguous",
            vec![0.0, 1.0],
        ));

        memory.observe_now(&now, Some(PlaceCellKey { x: 0, y: 0 }));

        assert!(memory
            .entities
            .values()
            .all(|entity| entity.modality_support.voice_vector_ids.is_empty()));
        assert!(!memory.report().ambiguous_binding_candidates.is_empty());
    }

    #[test]
    fn ambiguous_voice_observation_creates_competing_tracking_hypotheses() {
        let mut memory = EntityMemory::new();
        let mut now = now_at(100, 0.0, 0.0);
        now.objects
            .observations
            .push(make_object_observation("travis", ObjectClass::Person, 0.8));
        now.objects
            .observations
            .push(make_object_observation("tim", ObjectClass::Person, 0.8));
        now.voice.vectors.push(VectorArtifact::new(
            VOICE_VECTOR_COLLECTION,
            "voice-ambiguous-ticket5",
            vec![0.0, 1.0],
        ));

        memory.observe_now(&now, Some(PlaceCellKey { x: 0, y: 0 }));

        let family = memory
            .tracking_hypotheses
            .values()
            .filter(|hypothesis| hypothesis.family_id == "voice-identity:voice-ambiguous-ticket5")
            .collect::<Vec<_>>();
        assert_eq!(family.len(), 3);
        assert!(family
            .iter()
            .all(|hypothesis| { hypothesis.kind == TrackingHypothesisKind::VoiceIdentity }));
        assert!(memory
            .entities
            .values()
            .all(|entity| entity.modality_support.voice_vector_ids.is_empty()));
    }

    #[test]
    fn single_clear_candidate_promotes_identity_binding() {
        let mut memory = EntityMemory::new();
        let cell = PlaceCellKey { x: 1, y: 1 };
        let mut now = now_at(100, 0.0, 0.0);
        now.objects
            .observations
            .push(make_object_observation("ada", ObjectClass::Person, 0.9));
        now.face.vectors.push(VectorArtifact::new(
            FACE_VECTOR_COLLECTION,
            "face-ada-ticket5",
            vec![1.0, 0.0],
        ));

        memory.observe_now(&now, Some(cell));

        let ada = memory.entities.get("entity:person:ada").unwrap();
        assert_eq!(
            ada.modality_support.face_vector_ids,
            vec!["face-ada-ticket5".to_string()]
        );
        assert!(memory
            .tracking_hypotheses
            .values()
            .any(|hypothesis| hypothesis.state == HypothesisState::Promoted
                && hypothesis.target_id.as_deref() == Some("entity:person:ada")));
    }

    #[test]
    fn close_identity_candidates_remain_unresolved_for_review() {
        let mut memory = EntityMemory::new();
        let mut now = now_at(100, 0.0, 0.0);
        now.objects
            .observations
            .push(make_object_observation("travis", ObjectClass::Person, 0.8));
        now.objects
            .observations
            .push(make_object_observation("tim", ObjectClass::Person, 0.8));
        now.face.vectors.push(VectorArtifact::new(
            FACE_VECTOR_COLLECTION,
            "face-close-ticket5",
            vec![1.0, 0.0],
        ));

        memory.observe_now(&now, Some(PlaceCellKey { x: 0, y: 0 }));

        let report = memory.report();
        assert!(
            report.review_tracking_hypotheses.len() >= 2,
            "near-equal known-person hypotheses should be visible for review"
        );
        assert!(report.promoted_tracking_hypotheses.is_empty());
    }

    #[test]
    fn contradiction_lowers_hypothesis_confidence() {
        let positive = score_hypothesis_evidence(
            &[
                BindingEvidence {
                    kind: BindingEvidenceKind::VectorSimilarity,
                    score: 0.8,
                    reason: "high vector similarity".to_string(),
                },
                BindingEvidence {
                    kind: BindingEvidenceKind::TemporalOverlap,
                    score: 0.8,
                    reason: "same time window".to_string(),
                },
                BindingEvidence {
                    kind: BindingEvidenceKind::SpatialOverlap,
                    score: 0.8,
                    reason: "same place".to_string(),
                },
            ],
            1.0,
        );
        let contradicted = score_hypothesis_evidence(
            &[
                BindingEvidence {
                    kind: BindingEvidenceKind::VectorSimilarity,
                    score: 0.8,
                    reason: "high vector similarity".to_string(),
                },
                BindingEvidence {
                    kind: BindingEvidenceKind::TemporalOverlap,
                    score: 0.8,
                    reason: "same time window".to_string(),
                },
                BindingEvidence {
                    kind: BindingEvidenceKind::Contradiction,
                    score: 1.0,
                    reason: "same person appears in incompatible places".to_string(),
                },
            ],
            1.0,
        );

        assert!(contradicted < positive);
    }

    #[test]
    fn human_confirmation_promotes_correct_hypothesis() {
        let mut memory = EntityMemory::new();
        let mut first = now_at(100, 0.0, 0.0);
        first
            .objects
            .observations
            .push(make_object_observation("ada", ObjectClass::Person, 0.9));
        memory.observe_now(&first, None);

        let mut later = now_at(10_000, 3.0, 3.0);
        later.face.vectors.push(
            VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-human-confirmed", vec![1.0])
                .with_source_id("entity:person:ada"),
        );
        memory.observe_now(&later, None);

        let ada = memory.entities.get("entity:person:ada").unwrap();
        assert_eq!(
            ada.modality_support.face_vector_ids,
            vec!["face-human-confirmed".to_string()]
        );
        assert!(memory
            .report()
            .promoted_tracking_hypotheses
            .iter()
            .any(|hypothesis| hypothesis.target_id.as_deref() == Some("entity:person:ada")));
    }

    #[test]
    fn stale_weak_hypotheses_expire() {
        let mut memory = EntityMemory::new();
        let mut now = now_at(100, 0.0, 0.0);
        now.face.vectors.push(VectorArtifact::new(
            FACE_VECTOR_COLLECTION,
            "face-stale-ticket5",
            vec![1.0],
        ));
        memory.observe_now(&now, None);

        let later = now_at(120_000, 0.0, 0.0);
        memory.observe_now(&later, None);

        assert!(memory
            .tracking_hypotheses
            .values()
            .any(|hypothesis| hypothesis.state == HypothesisState::Expired));
    }

    #[test]
    fn promoted_hypothesis_does_not_merge_unrelated_entities() {
        let mut memory = EntityMemory::new();
        let mut now = now_at(100, 0.0, 0.0);
        now.objects
            .observations
            .push(make_object_observation("ada", ObjectClass::Person, 0.9));
        now.objects
            .observations
            .push(make_object_observation("grace", ObjectClass::Person, 0.9));
        memory.observe_now(&now, None);

        let mut confirmed = now_at(200, 0.0, 0.0);
        confirmed.face.vectors.push(
            VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-ada-confirmed", vec![1.0])
                .with_source_id("entity:person:ada"),
        );
        memory.observe_now(&confirmed, None);

        assert!(memory.entities.contains_key("entity:person:ada"));
        assert!(memory.entities.contains_key("entity:person:grace"));
        assert_eq!(memory.entities.len(), 2);
        assert!(memory
            .entities
            .get("entity:person:grace")
            .unwrap()
            .modality_support
            .face_vector_ids
            .is_empty());
    }

    #[test]
    fn scene_vector_is_not_attached_to_every_active_entity() {
        let mut memory = EntityMemory::new();
        let mut first = now_at(100, 0.0, 0.0);
        first
            .objects
            .observations
            .push(make_object_observation("cup", ObjectClass::Unknown, 0.8));
        memory.observe_now(&first, Some(PlaceCellKey { x: 0, y: 0 }));

        let mut second = now_at(200, 2.0, 2.0);
        second.objects.observations.push(make_object_observation(
            "lamp",
            ObjectClass::Unknown,
            0.8,
        ));
        second.eye.scene_vectors.push(VectorArtifact::new(
            SCENE_VECTOR_COLLECTION,
            "scene-current",
            vec![0.5, 0.5],
        ));
        memory.observe_now(&second, Some(PlaceCellKey { x: 4, y: 4 }));

        let cup = memory.entities.get("entity:unknown:cup").unwrap();
        let lamp = memory.entities.get("entity:unknown:lamp").unwrap();
        assert!(cup.modality_support.scene_vector_ids.is_empty());
        assert_eq!(
            lamp.modality_support.scene_vector_ids,
            vec!["scene-current".to_string()]
        );
    }

    #[test]
    fn candidate_with_only_one_weak_evidence_source_is_held() {
        let observation = make_object_observation("ada", ObjectClass::Person, 0.8);
        let entity = EntityHypothesis::from_observation(&observation, 100, None);
        let candidate = qualify_binding_candidate(
            &entity,
            &VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-only", vec![1.0]),
            VectorBindingKind::Face,
            entity.primary_object_cluster_id().unwrap(),
            vector_cluster_id(VectorBindingKind::Face, "face-only"),
            5_000,
            None,
            0,
            false,
        );

        assert_ne!(candidate.decision, BindingDecision::Accept);
        assert!(candidate.reason.contains("single vector"));
    }

    #[test]
    fn candidate_with_temporal_and_spatial_evidence_is_accepted() {
        let observation = make_object_observation("ada", ObjectClass::Person, 0.8);
        let cell = PlaceCellKey { x: 1, y: 2 };
        let entity = EntityHypothesis::from_observation(&observation, 100, Some(cell));
        let candidate = qualify_binding_candidate(
            &entity,
            &VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-ada", vec![1.0]),
            VectorBindingKind::Face,
            entity.primary_object_cluster_id().unwrap(),
            vector_cluster_id(VectorBindingKind::Face, "face-ada"),
            150,
            Some(cell),
            1,
            false,
        );

        assert_eq!(candidate.decision, BindingDecision::Accept);
    }

    #[test]
    fn rejected_candidate_includes_useful_reason() {
        let observation = make_object_observation("ada", ObjectClass::Person, 0.8);
        let entity = EntityHypothesis::from_observation(&observation, 100, None);
        let candidate = qualify_binding_candidate(
            &entity,
            &VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-grace", vec![1.0])
                .with_source_id("entity:person:grace"),
            VectorBindingKind::Face,
            entity.primary_object_cluster_id().unwrap(),
            vector_cluster_id(VectorBindingKind::Face, "face-grace"),
            150,
            None,
            1,
            false,
        );

        assert_eq!(candidate.decision, BindingDecision::Reject);
        assert!(candidate
            .reason
            .contains("contradicts explicit entity source"));
        assert!(candidate.evidence.iter().any(|evidence| {
            evidence.kind == BindingEvidenceKind::Contradiction
                && evidence.reason.contains("entity:person:grace")
                && evidence.reason.contains("entity:person:ada")
        }));
    }

    #[test]
    fn cross_modal_engine_proposes_face_voice_candidate_without_mutation() {
        let mut engine = DefaultCrossModalBindingEngine::default();
        let context = BindingContext::new(1_000);
        let clusters = vec![
            DiscoveredCluster::new(
                "face-a",
                Modality::Vision,
                DiscoveredClusterKind::Face,
                1_000,
                0.9,
            ),
            DiscoveredCluster::new(
                "voice-a",
                Modality::Audio,
                DiscoveredClusterKind::Voice,
                1_050,
                0.85,
            ),
        ];

        let candidates = engine.propose_bindings(&context, &clusters);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].relation, BindingRelation::LikelySameEntity);
        assert!(candidates[0]
            .evidence
            .iter()
            .any(|evidence| evidence.kind == BindingEvidenceKind::TemporalOverlap));
        assert!(candidates[0]
            .evidence
            .iter()
            .any(|evidence| evidence.kind == BindingEvidenceKind::SingleCandidateContext));
        assert_ne!(candidates[0].decision, BindingDecision::Accept);
        assert!(candidates[0].reason.contains("admission must accept"));
        assert_eq!(
            clusters.len(),
            2,
            "engine must not mutate clusters or entities"
        );
    }

    #[test]
    fn cross_modal_engine_holds_ambiguous_face_voice_context() {
        let mut engine = DefaultCrossModalBindingEngine::default();
        let context = BindingContext::new(1_000);
        let clusters = vec![
            DiscoveredCluster::new(
                "face-a",
                Modality::Vision,
                DiscoveredClusterKind::Face,
                1_000,
                0.9,
            ),
            DiscoveredCluster::new(
                "face-b",
                Modality::Vision,
                DiscoveredClusterKind::Face,
                1_010,
                0.9,
            ),
            DiscoveredCluster::new(
                "voice-a",
                Modality::Audio,
                DiscoveredClusterKind::Voice,
                1_020,
                0.9,
            ),
            DiscoveredCluster::new(
                "voice-b",
                Modality::Audio,
                DiscoveredClusterKind::Voice,
                1_030,
                0.9,
            ),
        ];

        let candidates = engine.propose_bindings(&context, &clusters);

        assert!(!candidates.is_empty());
        assert!(candidates.iter().all(|candidate| {
            candidate.decision == BindingDecision::HoldAmbiguous
                && candidate
                    .evidence
                    .iter()
                    .any(|evidence| evidence.kind == BindingEvidenceKind::SimultaneousConflict)
        }));
    }

    #[test]
    fn cross_modal_engine_proposes_rgb_geometry_projection() {
        let mut engine = DefaultCrossModalBindingEngine::default();
        let context = BindingContext {
            source_frame_id: Some("frame-1".to_string()),
            ..BindingContext::new(1_000)
        };
        let rgb = DiscoveredCluster::new(
            "rgb-patch",
            Modality::Vision,
            DiscoveredClusterKind::RgbImage,
            1_000,
            0.8,
        )
        .with_source_frame_id("frame-1")
        .with_metadata(json!({ "image_x": 100.0, "image_y": 50.0 }));
        let depth = DiscoveredCluster::new(
            "voxel",
            Modality::Depth,
            DiscoveredClusterKind::Geometry,
            1_000,
            0.8,
        )
        .with_source_frame_id("frame-1")
        .with_metadata(json!({ "projected_image_x": 102.0, "projected_image_y": 53.0 }));

        let candidates = engine.propose_bindings(&context, &[rgb, depth]);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].relation, BindingRelation::ProjectsTo);
        assert!(candidates[0]
            .evidence
            .iter()
            .any(|evidence| evidence.kind == BindingEvidenceKind::ProjectionAgreement));
    }

    #[test]
    fn cross_modal_engine_rejects_rgb_geometry_projection_disagreement() {
        let mut engine = DefaultCrossModalBindingEngine::default();
        let context = BindingContext {
            source_frame_id: Some("frame-1".to_string()),
            ..BindingContext::new(1_000)
        };
        let rgb = DiscoveredCluster::new(
            "rgb-patch",
            Modality::Vision,
            DiscoveredClusterKind::RgbImage,
            1_000,
            0.8,
        )
        .with_source_frame_id("frame-1")
        .with_metadata(json!({ "image_x": 100.0, "image_y": 50.0 }));
        let depth = DiscoveredCluster::new(
            "voxel",
            Modality::Depth,
            DiscoveredClusterKind::Geometry,
            1_000,
            0.8,
        )
        .with_source_frame_id("frame-1")
        .with_metadata(json!({ "projected_image_x": 180.0, "projected_image_y": 110.0 }));

        let candidates = engine.propose_bindings(&context, &[rgb, depth]);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].relation, BindingRelation::ProjectsTo);
        assert_eq!(candidates[0].decision, BindingDecision::Reject);
        assert!(candidates[0]
            .evidence
            .iter()
            .any(|evidence| evidence.kind == BindingEvidenceKind::Contradiction));
    }

    #[test]
    fn cross_modal_engine_proposes_object_place_binding() {
        let mut engine = DefaultCrossModalBindingEngine::default();
        let cell = PlaceCellKey { x: 2, y: 3 };
        let context = BindingContext {
            current_place_cell: Some(cell),
            ..BindingContext::new(1_000)
        };
        let object = DiscoveredCluster::new(
            "charger-object",
            Modality::Vision,
            DiscoveredClusterKind::Object,
            1_000,
            0.9,
        )
        .with_place_cell(cell)
        .with_metadata(json!({ "cooccurrence_count": 3 }));
        let place = DiscoveredCluster::new(
            "dock-place",
            Modality::Memory,
            DiscoveredClusterKind::Place,
            950,
            0.8,
        )
        .with_place_cell(cell);

        let candidates = engine.propose_bindings(&context, &[object, place]);

        assert_eq!(candidates.len(), 1);
        assert_eq!(
            candidates[0].relation,
            BindingRelation::CooccursInEstimatedSpace
        );
        assert!(candidates[0]
            .evidence
            .iter()
            .any(|evidence| evidence.kind == BindingEvidenceKind::SpatialOverlap));
        assert!(candidates[0]
            .evidence
            .iter()
            .any(|evidence| evidence.kind == BindingEvidenceKind::RepeatedCooccurrence));
    }

    #[test]
    fn cross_modal_engine_proposes_action_outcome_binding() {
        let mut engine = DefaultCrossModalBindingEngine::default();
        let context = BindingContext {
            active_action: Some(ActionPrimitive::Go {
                intensity: 0.5,
                duration_ms: 500,
            }),
            body_state: Some(BodySense {
                flags: BodyFlags {
                    bump_left: true,
                    ..BodyFlags::default()
                },
                ..BodySense::default()
            }),
            ..BindingContext::new(1_000)
        };
        let action = DiscoveredCluster::new(
            "go-forward",
            Modality::Odometry,
            DiscoveredClusterKind::Action,
            1_000,
            0.8,
        );
        let outcome = DiscoveredCluster::new(
            "bump-left",
            Modality::Touch,
            DiscoveredClusterKind::Outcome,
            1_400,
            0.9,
        );

        let candidates = engine.propose_bindings(&context, &[action, outcome]);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].relation, BindingRelation::ExplainsOutcome);
        assert!(candidates[0]
            .evidence
            .iter()
            .any(|evidence| evidence.reason.contains("outcome followed action")));
    }

    #[test]
    fn cross_modal_engine_keeps_llm_label_alone_weak() {
        let mut engine = DefaultCrossModalBindingEngine::default();
        let context = BindingContext::new(1_000);
        let label = DiscoveredCluster::new(
            "llm-label-chair",
            Modality::Language,
            DiscoveredClusterKind::Label,
            1_000,
            0.7,
        )
        .with_metadata(json!({ "source": "llm", "label": "chair" }));
        let object = DiscoveredCluster::new(
            "object-blob",
            Modality::Vision,
            DiscoveredClusterKind::Object,
            5_000,
            0.7,
        );

        let candidates = engine.propose_bindings(&context, &[label, object]);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].relation, BindingRelation::NamedBy);
        assert_eq!(candidates[0].decision, BindingDecision::CollectMoreEvidence);
        assert!(candidates[0].reason.contains("LLM suggestion alone"));
    }

    #[test]
    fn cross_modal_engine_never_accepts_without_admission() {
        let mut engine = DefaultCrossModalBindingEngine::default();
        let cell = PlaceCellKey { x: 2, y: 3 };
        let context = BindingContext {
            current_place_cell: Some(cell),
            recent_clusters: vec!["charger-object".to_string(), "dock-place".to_string()],
            ..BindingContext::new(1_000)
        };
        let object = DiscoveredCluster::new(
            "charger-object",
            Modality::Vision,
            DiscoveredClusterKind::Object,
            1_000,
            0.95,
        )
        .with_place_cell(cell)
        .with_metadata(json!({ "cooccurrence_count": 5 }));
        let place = DiscoveredCluster::new(
            "dock-place",
            Modality::Memory,
            DiscoveredClusterKind::Place,
            1_000,
            0.95,
        )
        .with_place_cell(cell);

        let candidates = engine.propose_bindings(&context, &[object, place]);

        assert_eq!(candidates.len(), 1);
        assert_ne!(candidates[0].decision, BindingDecision::Accept);
        assert!(candidates[0].reason.contains("admission must accept"));
    }

    #[test]
    fn accepted_candidates_strengthen_existing_binding_edges() {
        let mut memory = EntityMemory::new();
        let cell = PlaceCellKey { x: 0, y: 0 };
        let mut now1 = now_at(100, 0.0, 0.0);
        now1.objects
            .observations
            .push(make_object_observation("ada", ObjectClass::Person, 0.8));
        now1.face.vectors.push(VectorArtifact::new(
            FACE_VECTOR_COLLECTION,
            "face-ada",
            vec![1.0, 0.0],
        ));
        memory.observe_now(&now1, Some(cell));

        let before = memory.entities["entity:person:ada"]
            .constellation
            .binding_edges[0]
            .clone();

        let mut now2 = now_at(200, 0.0, 0.0);
        now2.objects
            .observations
            .push(make_object_observation("ada", ObjectClass::Person, 0.9));
        now2.face.vectors.push(VectorArtifact::new(
            FACE_VECTOR_COLLECTION,
            "face-ada",
            vec![1.0, 0.0],
        ));
        memory.observe_now(&now2, Some(cell));

        let after = &memory.entities["entity:person:ada"]
            .constellation
            .binding_edges[0];
        assert!(after.evidence_count > before.evidence_count);
        assert!(after.confidence > before.confidence);
    }

    #[test]
    fn strong_constellation_requires_multiple_supporting_bindings() {
        let observation = make_object_observation("ada", ObjectClass::Person, 0.8);
        let mut entity = EntityHypothesis::from_observation(&observation, 100, None);
        let object_cluster = entity.primary_object_cluster_id().unwrap();
        let face_cluster = entity.add_face_vector("face-ada");
        for step in 0..5 {
            entity.upsert_binding_edge(
                object_cluster.clone(),
                face_cluster.clone(),
                BindingRelation::LikelySameEntity,
                1.0,
                100 + step,
            );
        }
        assert_eq!(entity.constellation.state, EntityConstellationState::Weak);

        let voice_cluster = entity.add_voice_vector("voice-ada");
        for step in 0..5 {
            entity.upsert_binding_edge(
                object_cluster.clone(),
                voice_cluster.clone(),
                BindingRelation::LikelySameEntity,
                1.0,
                200 + step,
            );
        }
        assert_eq!(entity.constellation.state, EntityConstellationState::Strong);
    }

    #[test]
    fn binding_admission_does_not_merge_entities() {
        let mut memory = EntityMemory::new();
        let mut now = now_at(100, 0.0, 0.0);
        now.objects
            .observations
            .push(make_object_observation("ada", ObjectClass::Person, 0.8));
        now.objects
            .observations
            .push(make_object_observation("grace", ObjectClass::Person, 0.8));
        now.face.vectors.push(VectorArtifact::new(
            FACE_VECTOR_COLLECTION,
            "face-ambiguous",
            vec![1.0, 0.0],
        ));

        memory.observe_now(&now, Some(PlaceCellKey { x: 0, y: 0 }));

        assert_eq!(memory.entities.len(), 2);
        assert!(memory.entities.contains_key("entity:person:ada"));
        assert!(memory.entities.contains_key("entity:person:grace"));
    }

    #[test]
    fn entity_constellation_supports_experience_binding_and_merge_split_states() {
        let mut memory = EntityMemory::new();
        let mut frame = empty_frame(now_at(100, 1.0, 1.0));
        frame.now.objects.observations.push(make_object_observation(
            "charger",
            ObjectClass::Charger,
            0.9,
        ));
        let sensation_id = uuid::Uuid::new_v4();
        let mut experience = Experience::new(
            "embodied.place",
            "charger alcove",
            Vec::new(),
            vec![sensation_id],
            90,
            100,
        );
        experience.salience = 0.9;
        frame.experiences.push(experience);
        frame.impressions.push(
            Impression::new("memory.impression", "charger", Vec::new(), 90, 100)
                .with_confidence(0.8),
        );
        memory.observe_frame(&frame, Some(PlaceCellKey { x: 2, y: 2 }));

        let entity_id = "entity:charger:charger".to_string();
        let entity = memory.entities.get(&entity_id).expect("charger entity");
        assert!(
            entity
                .constellation
                .modality_clusters
                .iter()
                .any(|cluster| cluster.modality == Modality::Memory),
            "experience-level memory clusters should bind into the entity constellation"
        );

        let split_id = memory
            .split_entity(&entity_id, "left")
            .expect("split child id");
        assert!(memory.merge_entities(&entity_id, &split_id));
        let merged = memory.entities.get(&entity_id).expect("merged entity");
        assert_eq!(merged.constellation.state, EntityConstellationState::Merged);
    }
}
