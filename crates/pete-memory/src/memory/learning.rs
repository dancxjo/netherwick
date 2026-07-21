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

