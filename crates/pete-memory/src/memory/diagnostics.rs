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

