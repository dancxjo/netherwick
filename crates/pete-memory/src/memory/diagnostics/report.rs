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
