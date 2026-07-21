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
