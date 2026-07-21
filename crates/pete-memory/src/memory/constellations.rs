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

