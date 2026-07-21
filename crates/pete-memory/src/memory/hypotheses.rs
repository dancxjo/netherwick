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

