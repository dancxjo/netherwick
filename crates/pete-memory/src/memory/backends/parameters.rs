fn neo4j_intelligence_params(document: &GraphIntelligenceDocument) -> serde_json::Value {
    let document_meta = json!({
        "id": document.id,
        "t_ms": document.t_ms,
        "frame_id": document.frame_id,
        "provenance": document.provenance,
        "confidence": document.confidence,
        "reason": document.reason,
        "source_frame_ids": document.source_frame_ids,
    });
    json!({
        "document": document_meta,
        "features": document.features.iter().map(|feature| json!({
            "id": feature.id.to_string(),
            "feature_type": format!("{:?}", feature.feature_type),
            "modality": feature.modality.as_str(),
            "created_at_ms": feature.created_at_ms,
            "confidence": feature.confidence,
            "provenance_json": json_string(&feature.provenance),
            "source_frame": feature.source_frame,
            "source_sensor": feature.source_sensor,
            "vector_refs_json": json_string(&feature.vector_refs),
            "metadata_json": json_string(&feature.metadata),
            "current_state": "observed",
            "reason": document.reason,
        })).collect::<Vec<_>>(),
        "clusters": document.clusters.iter().map(|cluster| json!({
            "id": cluster.id,
            "modality": cluster.modality.as_str(),
            "kind": format!("{:?}", cluster.kind),
            "first_seen_ms": cluster.first_seen_ms,
            "last_seen_ms": cluster.last_seen_ms,
            "confidence": cluster.confidence,
            "evidence_count": cluster.feature_ids.len() as u32,
            "source_frame_id": cluster.source_frame_id,
            "current_state": "active",
            "reason": document.reason,
            "metadata_json": json_string(&cluster.metadata),
        })).collect::<Vec<_>>(),
        "cluster_features": document.clusters.iter().flat_map(|cluster| {
            cluster.feature_ids.iter().map(|feature_id| json!({
                "cluster_id": cluster.id,
                "feature_id": feature_id.to_string(),
                "confidence": cluster.confidence,
                "t_ms": cluster.last_seen_ms,
                "provenance": document.provenance,
                "source_frame_ids": document.source_frame_ids,
            }))
        }).collect::<Vec<_>>(),
        "binding_candidates": document.binding_candidates.iter().map(|candidate| {
            let id = binding_candidate_id(candidate);
            json!({
                "id": id,
                "left_cluster_id": candidate.left_cluster_id,
                "right_cluster_id": candidate.right_cluster_id,
                "relation": binding_relation_slug(&candidate.relation),
                "confidence": candidate.confidence,
                "evidence_count": candidate.evidence.len() as u32,
                "decision": binding_decision_slug(&candidate.decision),
                "current_state": binding_decision_slug(&candidate.decision),
                "reason": candidate.reason,
                "t_ms": document.t_ms,
                "provenance": document.provenance,
                "source_frame_ids": document.source_frame_ids,
            })
        }).collect::<Vec<_>>(),
        "binding_edges": document.binding_edges.iter().map(|edge| {
            let id = binding_edge_id(edge);
            json!({
                "id": id,
                "left_cluster_id": edge.left_cluster_id,
                "right_cluster_id": edge.right_cluster_id,
                "relation": binding_relation_slug(&edge.relation),
                "confidence": edge.confidence,
                "evidence_count": edge.evidence_count,
                "last_seen_ms": edge.last_seen_ms,
                "current_state": if edge.is_strong() { "accepted" } else { "provisional" },
                "reason": format!("{} evidence events", edge.evidence_count),
            })
        }).collect::<Vec<_>>(),
        "candidate_edges": document.binding_candidates.iter().map(|candidate| json!({
            "candidate_id": binding_candidate_id(candidate),
            "binding_id": binding_edge_id_from_parts(&candidate.left_cluster_id, &candidate.right_cluster_id, &candidate.relation),
            "confidence": candidate.confidence,
            "reason": candidate.reason,
            "t_ms": document.t_ms,
        })).collect::<Vec<_>>(),
        "candidate_evidence": document.binding_candidates.iter().flat_map(|candidate| {
            let candidate_id = binding_candidate_id(candidate);
            candidate.evidence.iter().enumerate().map(move |(index, evidence)| json!({
                "id": format!("evidence:{}:{}", stable_slug(&candidate_id), index),
                "candidate_id": candidate_id,
                "kind": binding_evidence_slug(&evidence.kind),
                "score": evidence.score,
                "reason": evidence.reason,
                "t_ms": document.t_ms,
                "current_state": if binding_evidence_is_contradictory(evidence) { "contradictory" } else { "supporting" },
                "contradictory": binding_evidence_is_contradictory(evidence),
            }))
        }).collect::<Vec<_>>(),
        "tracking_hypotheses": document.tracking_hypotheses.iter().map(|hypothesis| json!({
            "id": hypothesis.id,
            "family_id": hypothesis.family_id,
            "kind": format!("{:?}", hypothesis.kind),
            "target_id": hypothesis.target_id,
            "confidence": hypothesis.confidence,
            "evidence_count": hypothesis.evidence.len() as u32,
            "current_state": format!("{:?}", hypothesis.state).to_lowercase(),
            "first_seen_ms": hypothesis.first_seen_ms,
            "last_updated_ms": hypothesis.last_updated_ms,
            "contradictions": hypothesis.contradictions,
            "reason": hypothesis.contradictions.first().cloned().unwrap_or_else(|| "tracking hypothesis evidence".to_string()),
        })).collect::<Vec<_>>(),
        "hypothesis_candidates": document.tracking_hypotheses.iter().flat_map(|hypothesis| {
            hypothesis.binding_candidate_ids.iter().map(|candidate_id| json!({
                "hypothesis_id": hypothesis.id,
                "candidate_id": candidate_id,
                "confidence": hypothesis.confidence,
                "t_ms": hypothesis.last_updated_ms,
            }))
        }).collect::<Vec<_>>(),
        "hypothesis_competitions": hypothesis_competition_params(&document.tracking_hypotheses),
        "constellations": document.constellations.iter().map(|constellation| json!({
            "id": constellation.id,
            "kind_hint": constellation.kind_hint,
            "member_cluster_ids": constellation.member_cluster_ids,
            "member_binding_ids": constellation.member_binding_ids,
            "confidence": constellation.confidence,
            "stability": constellation.stability,
            "prediction_value": constellation.prediction_value,
            "first_seen_ms": constellation.first_seen_ms,
            "last_seen_ms": constellation.last_seen_ms,
            "evidence_count": constellation.evidence_count,
            "current_state": constellation_state_slug(&constellation.state),
            "reason": constellation.notes.first().cloned().unwrap_or_else(|| "constellation evidence".to_string()),
            "notes": constellation.notes,
        })).collect::<Vec<_>>(),
        "constellation_members": document.constellations.iter().flat_map(|constellation| {
            constellation.member_cluster_ids.iter().chain(constellation.member_binding_ids.iter()).map(|member_id| json!({
                "constellation_id": constellation.id,
                "member_id": member_id,
                "confidence": constellation.confidence,
                "t_ms": constellation.last_seen_ms,
            }))
        }).collect::<Vec<_>>(),
        "associations": document.associations.iter().map(|edge| json!({
            "id": edge.id,
            "from_id": edge.from_id,
            "to_id": edge.to_id,
            "relation": association_relation_slug(&edge.relation),
            "confidence": edge.confidence,
            "evidence_count": edge.evidence_count,
            "prediction_gain": edge.prediction_gain,
            "contradiction_count": edge.contradiction_count,
            "first_seen_ms": edge.first_seen_ms,
            "last_seen_ms": edge.last_seen_ms,
            "current_state": if edge.contradiction_count > 0 { "needs_review" } else { "active" },
            "reason": edge.examples.last().map(|example| example.reason.clone()).unwrap_or_else(|| "association evidence".to_string()),
            "examples_json": json_string(&edge.examples),
        })).collect::<Vec<_>>(),
        "action_intents": document.action_intents.iter().map(|action| json!({
            "id": action.id,
            "action_json": json_string(&action.action),
            "frame_id": action.frame_id,
            "t_ms": action.t_ms,
            "confidence": action.confidence,
            "current_state": action.state,
            "reason": action.reason,
        })).collect::<Vec<_>>(),
        "outcomes": document.outcomes.iter().map(|outcome| json!({
            "id": outcome.id,
            "frame_id": outcome.frame_id,
            "t_ms": outcome.t_ms,
            "reward": outcome.reward,
            "success": outcome.success,
            "confidence": outcome.confidence,
            "current_state": outcome.state,
            "reason": outcome.reason,
        })).collect::<Vec<_>>(),
        "action_outcomes": document.action_intents.iter().zip(document.outcomes.iter()).map(|(action, outcome)| json!({
            "action_id": action.id,
            "outcome_id": outcome.id,
            "confidence": action.confidence.min(outcome.confidence),
            "t_ms": outcome.t_ms,
            "reason": outcome.reason,
        })).collect::<Vec<_>>(),
        "predictions": document.predictions.iter().map(|prediction| json!({
            "id": prediction.id,
            "target_id": prediction.target_id,
            "predicted": prediction.predicted,
            "confidence": prediction.confidence,
            "t_ms": prediction.t_ms,
            "current_state": prediction.state,
            "reason": prediction.reason,
        })).collect::<Vec<_>>(),
        "surprises": document.surprises.iter().map(|surprise| json!({
            "id": surprise.id,
            "target_id": surprise.target_id,
            "observed": surprise.observed,
            "surprise": surprise.surprise,
            "confidence": surprise.confidence,
            "t_ms": surprise.t_ms,
            "reason": surprise.reason,
        })).collect::<Vec<_>>(),
        "prediction_failures": document.predictions.iter().flat_map(|prediction| {
            document.surprises.iter().filter(move |surprise| surprise.target_id == prediction.target_id).map(move |surprise| json!({
                "prediction_id": prediction.id,
                "surprise_id": surprise.id,
                "confidence": prediction.confidence.min(surprise.confidence),
                "t_ms": surprise.t_ms,
                "reason": surprise.reason,
            }))
        }).collect::<Vec<_>>(),
        "llm_reviews": document.llm_reviews.iter().map(|review| json!({
            "id": review.id,
            "target_id": review.target_id,
            "target_kind": format!("{:?}", review.target_kind),
            "confidence": review.confidence,
            "t_ms": review.t_ms,
            "critique": review.critique,
            "contradictions": review.contradictions,
            "suggested_questions": review.suggested_questions,
            "current_state": if review.contradictions.is_empty() { "open" } else { "needs_review" },
        })).collect::<Vec<_>>(),
        "human_reviews": document.human_reviews.iter().map(|review| json!({
            "id": review.id,
            "target_id": review.target_id,
            "target_kind": format!("{:?}", review.target_kind),
            "confidence": review.confidence,
            "t_ms": review.t_ms,
            "confirmation": review.confirmation,
            "reviewer": review.reviewer,
            "current_state": "confirmed",
        })).collect::<Vec<_>>(),
        "review_records": document.review_records.iter().map(|review| json!({
            "id": review.id,
            "target_id": review.target_id,
            "review_kind": review.review_kind,
            "severity": review.severity,
            "confidence": review.confidence,
            "t_ms": review.t_ms,
            "reason": review.reason,
            "evidence_ids": review.evidence_ids,
            "current_state": review.state,
        })).collect::<Vec<_>>(),
    })
}

fn hypothesis_competition_params(hypotheses: &[TrackingHypothesis]) -> Vec<serde_json::Value> {
    let mut by_family = BTreeMap::<String, Vec<&TrackingHypothesis>>::new();
    for hypothesis in hypotheses {
        by_family
            .entry(hypothesis.family_id.clone())
            .or_default()
            .push(hypothesis);
    }
    by_family
        .into_iter()
        .flat_map(|(family_id, hypotheses)| {
            let mut params = Vec::new();
            for left in 0..hypotheses.len() {
                for right in (left + 1)..hypotheses.len() {
                    params.push(json!({
                        "left_id": hypotheses[left].id,
                        "right_id": hypotheses[right].id,
                        "family_id": family_id,
                        "confidence": hypotheses[left].confidence.min(hypotheses[right].confidence),
                        "t_ms": hypotheses[left].last_updated_ms.max(hypotheses[right].last_updated_ms),
                    }));
                }
            }
            params
        })
        .collect()
}

fn json_string(value: &impl Serialize) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}

fn summaries_from_row(value: Option<&serde_json::Value>) -> Vec<GraphFactSummary> {
    value
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .map(summary_from_value)
        .filter(|summary| !summary.id.is_empty())
        .collect()
}

fn summary_from_value(value: &serde_json::Value) -> GraphFactSummary {
    GraphFactSummary {
        id: value
            .get("id")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string(),
        kind: value
            .get("kind")
            .or_else(|| value.get("kind_hint"))
            .or_else(|| value.get("feature_type"))
            .or_else(|| value.get("relation"))
            .and_then(|value| value.as_str())
            .unwrap_or("graph_fact")
            .to_string(),
        relation: value
            .get("relation")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        confidence: value
            .get("confidence")
            .or_else(|| value.get("score"))
            .and_then(value_as_f32)
            .unwrap_or_default(),
        evidence_count: value
            .get("evidence_count")
            .and_then(|value| value.as_u64())
            .unwrap_or_default() as u32,
        t_ms: value
            .get("t_ms")
            .or_else(|| value.get("last_seen_ms"))
            .or_else(|| value.get("last_updated_ms"))
            .and_then(|value| value.as_u64())
            .unwrap_or_default(),
        state: value
            .get("current_state")
            .or_else(|| value.get("decision"))
            .and_then(|value| value.as_str())
            .map(str::to_string),
        reason: value
            .get("reason")
            .or_else(|| value.get("summary"))
            .or_else(|| value.get("critique"))
            .or_else(|| value.get("confirmation"))
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string(),
    }
}

fn value_as_f32(value: &serde_json::Value) -> Option<f32> {
    value.as_f64().map(|value| value as f32)
}

fn graph_recall_query_ids(query: &GraphRecallQuery) -> Vec<String> {
    query
        .active_feature_ids
        .iter()
        .chain(query.active_cluster_ids.iter())
        .chain(query.active_constellation_ids.iter())
        .chain(query.action_ids.iter())
        .chain(query.place_ids.iter())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn binding_edge_id(edge: &BindingEdge) -> String {
    binding_edge_id_from_parts(
        &edge.left_cluster_id,
        &edge.right_cluster_id,
        &edge.relation,
    )
}

fn binding_edge_id_from_parts(left: &str, right: &str, relation: &BindingRelation) -> String {
    format!(
        "binding-edge:{}:{}:{}",
        stable_slug(left),
        binding_relation_slug(relation),
        stable_slug(right)
    )
}

fn binding_relation_slug(relation: &BindingRelation) -> &'static str {
    match relation {
        BindingRelation::CooccursInTime => "cooccurs_in_time",
        BindingRelation::CooccursInEstimatedSpace => "cooccurs_in_estimated_space",
        BindingRelation::MovesTogether => "moves_together",
        BindingRelation::PredictsSameFutureEvents => "predicts_same_future_events",
        BindingRelation::NamedBy => "named_by",
        BindingRelation::ProjectsTo => "projects_to",
        BindingRelation::HasColorAtPose => "has_color_at_pose",
        BindingRelation::LikelySameEntity => "likely_same_entity",
        BindingRelation::ExplainsOutcome => "explains_outcome",
        BindingRelation::Contradicts => "contradicts",
        BindingRelation::RequiresReview => "requires_review",
    }
}

fn binding_decision_slug(decision: &BindingDecision) -> &'static str {
    match decision {
        BindingDecision::Accept => "accept",
        BindingDecision::Reject => "reject",
        BindingDecision::HoldAmbiguous => "hold_ambiguous",
        BindingDecision::AskHuman => "ask_human",
        BindingDecision::CollectMoreEvidence => "collect_more_evidence",
    }
}

fn binding_evidence_slug(kind: &BindingEvidenceKind) -> &'static str {
    match kind {
        BindingEvidenceKind::TemporalOverlap => "temporal_overlap",
        BindingEvidenceKind::SpatialOverlap => "spatial_overlap",
        BindingEvidenceKind::VectorSimilarity => "vector_similarity",
        BindingEvidenceKind::ProjectionAgreement => "projection_agreement",
        BindingEvidenceKind::PoseAgreement => "pose_agreement",
        BindingEvidenceKind::RepeatedCooccurrence => "repeated_cooccurrence",
        BindingEvidenceKind::SingleCandidateContext => "single_candidate_context",
        BindingEvidenceKind::HumanConfirmed => "human_confirmed",
        BindingEvidenceKind::LlmSuggested => "llm_suggested",
        BindingEvidenceKind::Contradiction => "contradiction",
        BindingEvidenceKind::SimultaneousConflict => "simultaneous_conflict",
    }
}

fn binding_evidence_is_contradictory(evidence: &BindingEvidence) -> bool {
    matches!(
        evidence.kind,
        BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
    )
}

fn constellation_state_slug(state: &ConstellationState) -> &'static str {
    match state {
        ConstellationState::Candidate => "candidate",
        ConstellationState::Stable => "stable",
        ConstellationState::Ambiguous => "ambiguous",
        ConstellationState::SplitNeeded => "split_needed",
        ConstellationState::MergeNeeded => "merge_needed",
        ConstellationState::Retired => "retired",
    }
}

fn neo4j_entity_params(record: &MemoryRecord) -> Vec<serde_json::Value> {
    record
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
        .collect()
}

fn neo4j_relationship_params(record: &MemoryRecord) -> Vec<serde_json::Value> {
    record
        .graph_relationships
        .iter()
        .map(|edge| {
            json!({
                "edge_id": graph_edge_id(edge),
                "from": edge.from,
                "to": edge.to,
                "kind": edge.relationship,
                "summary": edge.summary,
                "score": edge.score,
                "payload_json": edge.payload.to_string(),
                "frame_id": record.frame_id.to_string(),
                "t_ms": record.t_ms,
            })
        })
        .collect()
}
