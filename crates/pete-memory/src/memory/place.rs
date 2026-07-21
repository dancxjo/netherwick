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
