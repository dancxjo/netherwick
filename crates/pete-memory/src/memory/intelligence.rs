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

