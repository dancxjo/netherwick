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
