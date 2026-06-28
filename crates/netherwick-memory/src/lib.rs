use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use netherwick_actions::ActionPrimitive;
use netherwick_core::{Goal, Pose2};
use netherwick_experience::{
    Experience, Impression, MemoryLink, RecalledExperience, VectorEmbedding,
};
use netherwick_ledger::{ExperienceFrame, ExperienceTransition};
use netherwick_now::{
    GraphEdge, GraphEntity, MemorySense, Now, RecallHit, VectorArtifact, FACE_VECTOR_COLLECTION,
    SCENE_VECTOR_COLLECTION, VOICE_VECTOR_COLLECTION,
};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
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

    async fn recall(&self, query: RecallQuery) -> Result<RecallBundle>;
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RecallQuery {
    pub now_text: String,
    pub pose: Option<Pose2>,
    pub scene_vector: Option<Vec<f32>>,
    #[serde(default)]
    pub scene_vectors: Vec<VectorArtifact>,
    pub face_vectors: Vec<Vec<f32>>,
    #[serde(default)]
    pub face_vector_artifacts: Vec<VectorArtifact>,
    pub voice_vectors: Vec<Vec<f32>>,
    #[serde(default)]
    pub voice_vector_artifacts: Vec<VectorArtifact>,
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
            face_vectors: now.face.embeddings.clone(),
            face_vector_artifacts: now.face.vectors.clone(),
            voice_vectors: now.voice.embeddings.clone(),
            voice_vector_artifacts: now.voice.vectors.clone(),
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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlaceRecognitionCandidate {
    pub kind: PlaceRecognitionKind,
    pub cell: PlaceCellSummary,
    pub source_vector_id: String,
    pub source_frame_id: Option<String>,
    pub query_vector_id: Option<String>,
    pub similarity: f32,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlaceSceneEmbedding {
    pub cell_key: PlaceCellKey,
    pub artifact: VectorArtifact,
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
        merge_vector_ids(&mut cell.associated_voice_vectors, &now.voice.vectors);
        self.store_scene_embeddings(key, &now.eye.scene_vectors);
        self.features_at(now.body.odometry.x_m, now.body.odometry.y_m)
    }

    pub fn observe_frame(&mut self, frame: &ExperienceFrame) -> PlaceMemoryFeatures {
        let features = self.observe_now(&frame.now);
        let key = self.quantize(frame.now.body.odometry.x_m, frame.now.body.odometry.y_m);
        let scene_vectors = scene_vectors_from_now(&frame.now, frame.id, frame.t_ms);
        if !scene_vectors.is_empty() {
            if let Some(cell) = self.cells.get_mut(&key) {
                merge_vector_ids(&mut cell.associated_scene_vectors, &scene_vectors);
            }
            self.store_scene_embeddings(key, &scene_vectors);
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
                    continue;
                }
                candidates.push(PlaceRecognitionCandidate {
                    kind: recognition_kind(current_key, stored.cell_key, similarity),
                    cell: summarize_cell(cell, confidence),
                    source_vector_id: stored.artifact.point_id.clone(),
                    source_frame_id: stored.artifact.source_frame_id.clone(),
                    query_vector_id: Some(query.point_id.clone()),
                    similarity,
                    confidence,
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
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QdrantConfig {
    pub url: String,
}

impl QdrantConfig {
    pub fn from_env() -> Option<Self> {
        std::env::var("NETHERWICK_QDRANT_URL")
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
        let user = std::env::var("NETHERWICK_NEO4J_USER").ok()?;
        let password = std::env::var("NETHERWICK_NEO4J_PASSWORD").ok()?;
        let http_url = std::env::var("NETHERWICK_NEO4J_HTTP_URL")
            .ok()
            .or_else(|| {
                std::env::var("NETHERWICK_NEO4J_URI")
                    .ok()
                    .and_then(|uri| neo4j_http_url_from_uri(&uri))
            })
            .unwrap_or_else(|| "http://localhost:7474".to_string());
        let database =
            std::env::var("NETHERWICK_NEO4J_DATABASE").unwrap_or_else(|_| "neo4j".to_string());
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
}

impl Neo4jGraphStore {
    pub fn new(config: Neo4jConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
        }
    }

    pub fn from_env() -> Option<Self> {
        Neo4jConfig::from_env().map(Self::new)
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
}

#[async_trait]
impl GraphStore for Neo4jGraphStore {
    async fn upsert_graph(&self, record: &MemoryRecord) -> Result<()> {
        let entities = record
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
            .collect::<Vec<_>>();
        let relationships = record
            .graph_relationships
            .iter()
            .map(|edge| {
                json!({
                    "from": edge.from,
                    "to": edge.to,
                    "kind": edge.relationship,
                    "summary": edge.summary,
                    "score": edge.score,
                    "payload": edge.payload,
                    "frame_id": record.frame_id.to_string(),
                    "t_ms": record.t_ms,
                })
            })
            .collect::<Vec<_>>();

        self.run_cypher(
            r#"
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
MERGE (from)-[r:RELATED {kind: relationship.kind}]->(to)
SET r.summary = relationship.summary,
    r.score = relationship.score,
    r.payload = relationship.payload,
    r.frame_id = relationship.frame_id,
    r.t_ms = relationship.t_ms
"#,
            json!({
                "entities": entities,
                "relationships": relationships,
            }),
        )
        .await
    }
}

#[derive(Clone, Default)]
pub struct InMemoryExperienceStore {
    records: Arc<Mutex<Vec<MemoryRecord>>>,
    places: Arc<Mutex<PlaceMemory>>,
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

    async fn recall(&self, query: RecallQuery) -> Result<RecallBundle> {
        self.inner.recall(query).await
    }
}

#[async_trait]
impl Recall for InMemoryExperienceStore {
    async fn observe_now(&self, now: &Now) -> Result<()> {
        self.places
            .lock()
            .expect("place memory mutex poisoned")
            .observe_now(now);
        Ok(())
    }

    async fn observe_frame(&self, frame: &ExperienceFrame) -> Result<()> {
        self.places
            .lock()
            .expect("place memory mutex poisoned")
            .observe_frame(frame);
        Ok(())
    }

    async fn observe_transition(&self, transition: &ExperienceTransition) -> Result<()> {
        self.places
            .lock()
            .expect("place memory mutex poisoned")
            .observe_transition(transition);
        Ok(())
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
    let face_vectors = vector_artifacts_from_now(
        FACE_VECTOR_COLLECTION,
        &frame.now.face.vectors,
        &frame.now.face.embeddings,
        frame.id,
        frame.t_ms,
    );
    let voice_vectors = vector_artifacts_from_now(
        VOICE_VECTOR_COLLECTION,
        &frame.now.voice.vectors,
        &frame.now.voice.embeddings,
        frame.id,
        frame.t_ms,
    );
    let linked_experiences =
        experiences_with_memory_links(frame, &scene_vectors, &face_vectors, &voice_vectors);
    let (sensation_vectors, mut vector_payloads) = sensation_vectors_from_frame(frame);
    let (experience_vectors, experience_payloads) =
        experience_vectors_from_experiences(frame, &linked_experiences);
    vector_payloads.extend(experience_payloads);
    let (graph_entities, graph_relationships) = graph_context_from_frame(
        frame,
        &linked_experiences,
        &scene_vectors,
        &face_vectors,
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
        voice_vectors,
        sensation_vectors,
        experience_vectors,
        vector_payloads,
        battery: frame.now.body.battery_level,
        active_goal: RecallQuery::from_now(&frame.now).active_goal,
        chosen_action: frame.chosen_action.clone(),
        warning,
        experience: linked_experiences.last().cloned(),
    })
}

pub fn attach_memory_links_to_frame(frame: &mut ExperienceFrame) {
    let scene_vectors = scene_vectors_from_now(&frame.now, frame.id, frame.t_ms);
    let face_vectors = vector_artifacts_from_now(
        FACE_VECTOR_COLLECTION,
        &frame.now.face.vectors,
        &frame.now.face.embeddings,
        frame.id,
        frame.t_ms,
    );
    let voice_vectors = vector_artifacts_from_now(
        VOICE_VECTOR_COLLECTION,
        &frame.now.voice.vectors,
        &frame.now.voice.embeddings,
        frame.id,
        frame.t_ms,
    );
    let links = memory_links_from_frame(frame, &scene_vectors, &face_vectors, &voice_vectors);
    for experience in &mut frame.experiences {
        merge_memory_links(&mut experience.memory_links, links.clone());
    }
}

fn experiences_with_memory_links(
    frame: &ExperienceFrame,
    scene_vectors: &[VectorArtifact],
    face_vectors: &[VectorArtifact],
    voice_vectors: &[VectorArtifact],
) -> Vec<Experience> {
    let links = memory_links_from_frame(frame, scene_vectors, face_vectors, voice_vectors);
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
    let visual = (!now.face.embeddings.is_empty() || !now.face.vectors.is_empty()) as u8 as f32;
    let voice = (!now.voice.embeddings.is_empty() || !now.voice.vectors.is_empty()) as u8 as f32;
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
        if let Some(embedding) = &experience.fused_vector {
            let artifact = embodied_vector_artifact(
                EXPERIENCE_VECTOR_COLLECTION,
                &format!("{}:experience:{}", frame.id, experience.id),
                embedding,
                frame.id,
                experience.id.to_string(),
                experience.occurred_at_ms,
            );
            entities.push(vector_entity(&artifact, "experience"));
            relationships.push(graph_edge(
                canonical_experience_id.clone(),
                vector_node_id(&artifact),
                "HAS_FUSED_VECTOR",
                Some(format!("{} dimensions", embedding.dim)),
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
        let key = format!(
            "{}:{}:{}",
            relationship.from, relationship.relationship, relationship.to
        );
        if seen.insert(key) {
            out.push(relationship);
        }
        if out.len() >= limit {
            break;
        }
    }
    out
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

fn vector_artifacts_from_now(
    collection: &str,
    artifacts: &[VectorArtifact],
    legacy_embeddings: &[Vec<f32>],
    frame_id: uuid::Uuid,
    t_ms: u64,
) -> Vec<VectorArtifact> {
    if !artifacts.is_empty() {
        return artifacts.to_vec();
    }
    legacy_embeddings
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            VectorArtifact::new(
                collection,
                format!("{frame_id}:{collection}:{index}"),
                vector.clone(),
            )
            .with_source_frame_id(frame_id.to_string())
            .with_occurred_at_ms(t_ms)
        })
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

fn experience_vectors_from_experiences(
    frame: &ExperienceFrame,
    experiences: &[Experience],
) -> (Vec<VectorArtifact>, BTreeMap<String, serde_json::Value>) {
    let mut payloads = BTreeMap::new();
    let artifacts = experiences
        .iter()
        .filter_map(|experience| {
            let embedding = experience.fused_vector.as_ref()?;
            let artifact = embodied_vector_artifact(
                EXPERIENCE_VECTOR_COLLECTION,
                &format!("{}:experience:{}", frame.id, experience.id),
                embedding,
                frame.id,
                experience.id.to_string(),
                experience.occurred_at_ms,
            );
            payloads.insert(
                vector_payload_key(&artifact),
                json!({
                    "payload_kind": embedding.payload_kind.as_str(),
                    "modality": embedding.modality.as_str(),
                    "experience_id": experience.id.to_string(),
                    "source_sensation_id": embedding.source_sensation_id.to_string(),
                    "sensation_ids": experience.sensation_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
                    "impression_ids": experience.impression_ids.iter().map(|id| id.to_string()).collect::<Vec<_>>(),
                    "model_id": embedding.model_id,
                    "dim": embedding.dim,
                    "observed_at_ms": experience.observed_at_ms,
                    "occurred_at_ms": experience.occurred_at_ms,
                    "window_start_ms": experience.window_start_ms,
                    "window_end_ms": experience.window_end_ms,
                    "generated_at_ms": embedding.generated_at_ms,
                    "experience_kind": experience.kind,
                    "summary": experience_summary_text(experience),
                    "summary_impression_text": experience.summary_impression.as_ref().map(|impression| impression.text.clone()),
                    "salience": experience.salience,
                    "tags": experience.tags,
                    "memory_links": experience.memory_links,
                }),
            );
            Some(artifact)
        })
        .collect();
    (artifacts, payloads)
}

fn memory_links_from_frame(
    frame: &ExperienceFrame,
    _scene_vectors: &[VectorArtifact],
    face_vectors: &[VectorArtifact],
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

fn object_observation_id(observation: &netherwick_now::ObjectObservation) -> String {
    format!(
        "object:{}:{}:{}",
        object_source_slug(&observation.source),
        object_class_slug(&observation.class),
        stable_slug(&observation.label)
    )
}

fn object_class_slug(class: &netherwick_now::ObjectClass) -> &'static str {
    match class {
        netherwick_now::ObjectClass::Obstacle => "obstacle",
        netherwick_now::ObjectClass::Charger => "charger",
        netherwick_now::ObjectClass::Person => "person",
        netherwick_now::ObjectClass::SoundSource => "sound_source",
        netherwick_now::ObjectClass::Landmark => "landmark",
        netherwick_now::ObjectClass::Unknown => "unknown",
    }
}

fn object_source_slug(source: &netherwick_now::ObjectObservationSource) -> &'static str {
    match source {
        netherwick_now::ObjectObservationSource::Sim => "sim",
        netherwick_now::ObjectObservationSource::Kinect => "kinect",
        netherwick_now::ObjectObservationSource::Captioner => "captioner",
        netherwick_now::ObjectObservationSource::HumanLabel => "human_label",
        netherwick_now::ObjectObservationSource::Unknown => "unknown",
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
    vectors
}

fn query_face_vectors(query: &RecallQuery) -> Vec<&[f32]> {
    let mut vectors = query
        .face_vector_artifacts
        .iter()
        .map(|artifact| artifact.vector.as_slice())
        .collect::<Vec<_>>();
    vectors.extend(query.face_vectors.iter().map(Vec::as_slice));
    vectors
}

fn query_voice_vectors(query: &RecallQuery) -> Vec<&[f32]> {
    let mut vectors = query
        .voice_vector_artifacts
        .iter()
        .map(|artifact| artifact.vector.as_slice())
        .collect::<Vec<_>>();
    vectors.extend(query.voice_vectors.iter().map(Vec::as_slice));
    vectors
}

fn recall_vector_ids(record: &MemoryRecord) -> Vec<String> {
    let mut ids = record
        .experience_vectors
        .iter()
        .chain(record.sensation_vectors.iter())
        .chain(record.scene_vectors.iter())
        .chain(record.face_vectors.iter())
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
    vectors.extend(query_voice_vectors(query));
    vectors
}

fn record_all_vectors(record: &MemoryRecord) -> Vec<&VectorArtifact> {
    record
        .scene_vectors
        .iter()
        .chain(record.face_vectors.iter())
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
    !query.face_vectors.is_empty() || !query.face_vector_artifacts.is_empty()
}

fn has_voice_query(query: &RecallQuery) -> bool {
    !query.voice_vectors.is_empty() || !query.voice_vector_artifacts.is_empty()
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
mod tests {
    use super::*;
    use netherwick_body::BodySense;
    use netherwick_experience::{
        Experience, Impression, Modality, RecalledExperience, Sensation, SensationMetadata,
        SensationPayload, SensationPayloadKind, SensationSource, VectorEmbedding,
    };
    use netherwick_ledger::ExperienceFrame;
    use netherwick_now::{
        FaceSense, ObjectClass, ObjectObservation, ObjectObservationSource, SurpriseSense,
        VectorArtifact, FACE_VECTOR_COLLECTION, SCENE_VECTOR_COLLECTION, VOICE_VECTOR_COLLECTION,
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
    async fn store_preserves_typed_vector_artifacts() {
        let mut now = Now::blank(123, BodySense::default());
        now.face = FaceSense {
            schema_version: 1,
            embeddings: Vec::new(),
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
                face_vector_artifacts: vec![VectorArtifact::new(
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
        let mut experience = Experience::new(
            "embodied.now",
            "I notice a familiar charger alcove.",
            Vec::new(),
            vec![source_sensation_id],
            450,
            500,
        );
        experience.fused_vector = Some(VectorEmbedding::new(
            vec![1.0, 0.0, 0.0],
            "unit.experience.v0",
            Modality::Other,
            SensationPayloadKind::Structured,
            source_sensation_id,
            500,
        ));
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
            netherwick_core::ProvenanceKind::MemoryRecall { experience_id }
                if experience_id == experience.id
        ));
        assert!(recollection
            .sensation
            .impression
            .as_ref()
            .is_some_and(|impression| impression.text.starts_with("I remember")));
    }

    #[test]
    fn legacy_embeddings_become_collection_artifacts() {
        let frame_id = uuid::Uuid::new_v4();
        let artifacts = vector_artifacts_from_now(
            FACE_VECTOR_COLLECTION,
            &[],
            &[vec![0.25, 0.75]],
            frame_id,
            99,
        );

        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].collection, FACE_VECTOR_COLLECTION);
        assert_eq!(artifacts[0].vector, vec![0.25, 0.75]);
        let expected_frame_id = frame_id.to_string();
        assert_eq!(
            artifacts[0].source_frame_id.as_deref(),
            Some(expected_frame_id.as_str())
        );
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
        experience.fused_vector = Some(VectorEmbedding::new(
            vec![0.5, 0.5, 0.0],
            "netherwick.fusion.test",
            Modality::Other,
            SensationPayloadKind::Structured,
            primary_id,
            126,
        ));
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
        now.face.embeddings.push(vec![1.0, 0.0]);

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
            target: netherwick_actions::InspectTarget::Charger,
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
}
