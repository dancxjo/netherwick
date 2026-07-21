fn base_cross_modal_evidence(
    context: &BindingContext,
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
) -> Vec<BindingEvidence> {
    let mut evidence = Vec::new();
    if temporally_compatible(context, left, right) {
        let delta_ms = cluster_time_delta_ms(left, right);
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::TemporalOverlap,
            score: (1.0 - delta_ms as f32 / context.time_window_ms.max(1) as f32).clamp(0.0, 1.0),
            reason: format!("{} and {} occurred within {delta_ms} ms", left.id, right.id),
        });
    }
    if source_frame_matches(context, left, right) {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::ProjectionAgreement,
            score: 0.55,
            reason: "clusters share a source frame context".to_string(),
        });
    }
    if let Some(distance) = pose_distance_m(left, right) {
        if distance <= 0.75 {
            evidence.push(BindingEvidence {
                kind: BindingEvidenceKind::SpatialOverlap,
                score: (1.0 - distance / 0.75).clamp(0.0, 1.0),
                reason: format!("cluster poses are within {distance:.2} m"),
            });
        }
    }
    evidence
}

fn candidate_from_evidence(
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
    relation: BindingRelation,
    evidence: Vec<BindingEvidence>,
    fallback_reason: &str,
) -> BindingCandidate {
    let has_human_confirmation = evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::HumanConfirmed);
    let has_hard_contradiction = evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::Contradiction);
    let has_conflict = evidence.iter().any(|item| {
        matches!(
            item.kind,
            BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
        )
    });
    let independent_positive_kinds = evidence
        .iter()
        .filter(|item| {
            !matches!(
                item.kind,
                BindingEvidenceKind::Contradiction
                    | BindingEvidenceKind::SimultaneousConflict
                    | BindingEvidenceKind::VectorSimilarity
                    | BindingEvidenceKind::LlmSuggested
            )
        })
        .map(|item| binding_evidence_kind_rank(&item.kind))
        .collect::<BTreeSet<_>>()
        .len();
    let mean_score = if evidence.is_empty() {
        0.0
    } else {
        evidence
            .iter()
            .map(|item| item.score.clamp(0.0, 1.0))
            .sum::<f32>()
            / evidence.len() as f32
    };
    let mut confidence = if has_human_confirmation {
        mean_score.max(0.9)
    } else {
        (mean_score * (independent_positive_kinds as f32 / 3.0).clamp(0.25, 1.0)).clamp(0.0, 1.0)
    };
    if has_conflict {
        confidence *= 0.35;
    }

    let (decision, reason) = if has_hard_contradiction {
        (
            BindingDecision::Reject,
            "candidate contains contradictory cross-modal evidence".to_string(),
        )
    } else if has_conflict {
        (
            BindingDecision::HoldAmbiguous,
            "candidate is plausible but has competing cross-modal evidence".to_string(),
        )
    } else if has_human_confirmation {
        (
            BindingDecision::Accept,
            "candidate has trusted human/source confirmation".to_string(),
        )
    } else if independent_positive_kinds >= 2 {
        (
            BindingDecision::Accept,
            "candidate has at least two independent cross-modal evidence types".to_string(),
        )
    } else if evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::LlmSuggested)
        && independent_positive_kinds == 0
    {
        (
            BindingDecision::CollectMoreEvidence,
            "LLM suggestion alone is not enough to bind clusters".to_string(),
        )
    } else if evidence.is_empty() {
        (
            BindingDecision::CollectMoreEvidence,
            fallback_reason.to_string(),
        )
    } else {
        (
            BindingDecision::CollectMoreEvidence,
            "candidate needs more independent evidence before admission".to_string(),
        )
    };

    BindingCandidate {
        left_cluster_id: left.id.clone(),
        right_cluster_id: right.id.clone(),
        relation,
        evidence,
        confidence: confidence.clamp(0.0, 1.0),
        decision,
        reason,
    }
}

fn proposal_candidate_from_evidence(
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
    relation: BindingRelation,
    evidence: Vec<BindingEvidence>,
    fallback_reason: &str,
) -> BindingCandidate {
    let mut candidate = candidate_from_evidence(left, right, relation, evidence, fallback_reason);
    if candidate.decision == BindingDecision::Accept {
        candidate.decision = BindingDecision::CollectMoreEvidence;
        candidate.reason =
            "candidate is proposal-only; conservative binding admission must accept it".to_string();
    }
    candidate
}

fn temporally_compatible(
    context: &BindingContext,
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
) -> bool {
    cluster_time_delta_ms(left, right) <= context.time_window_ms
}

fn cluster_time_delta_ms(left: &DiscoveredCluster, right: &DiscoveredCluster) -> u64 {
    if left.last_seen_ms < right.first_seen_ms {
        right.first_seen_ms.saturating_sub(left.last_seen_ms)
    } else if right.last_seen_ms < left.first_seen_ms {
        left.first_seen_ms.saturating_sub(right.last_seen_ms)
    } else {
        0
    }
}

fn source_frame_matches(
    context: &BindingContext,
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
) -> bool {
    if left.source_frame_id.is_some()
        && right.source_frame_id.is_some()
        && left.source_frame_id == right.source_frame_id
    {
        return true;
    }
    context.source_frame_id.as_ref().is_some_and(|frame| {
        left.source_frame_id.as_ref() == Some(frame)
            || right.source_frame_id.as_ref() == Some(frame)
    })
}

fn add_recent_cooccurrence(
    context: &BindingContext,
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
    evidence: &mut Vec<BindingEvidence>,
) {
    if context.recent_clusters.contains(&left.id) && context.recent_clusters.contains(&right.id) {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::RepeatedCooccurrence,
            score: 0.7,
            reason: "both clusters appeared in recent binding context".to_string(),
        });
    }
    if !left.feature_ids.is_empty()
        && !right.feature_ids.is_empty()
        && left
            .feature_ids
            .iter()
            .any(|id| context.recent_features.contains(id))
        && right
            .feature_ids
            .iter()
            .any(|id| context.recent_features.contains(id))
    {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::RepeatedCooccurrence,
            score: 0.7,
            reason: "both clusters reference recently observed features".to_string(),
        });
    }
}

fn add_repetition_evidence(
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
    evidence: &mut Vec<BindingEvidence>,
) {
    let repeats =
        metadata_u64(left, "cooccurrence_count").max(metadata_u64(right, "cooccurrence_count"));
    if repeats >= 2 {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::RepeatedCooccurrence,
            score: (repeats as f32 / 5.0).clamp(0.0, 1.0),
            reason: format!("clusters have repeated together in {repeats} observations"),
        });
    }
}

fn add_label_support(
    left: &DiscoveredCluster,
    right: &DiscoveredCluster,
    evidence: &mut Vec<BindingEvidence>,
) {
    let left_label = metadata_string(left, "label");
    let right_label = metadata_string(right, "label");
    if left_label.is_some() && left_label == right_label {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::HumanConfirmed,
            score: 0.85,
            reason: "clusters share a supporting label".to_string(),
        });
    }
}

fn projection_error_px(left: &DiscoveredCluster, right: &DiscoveredCluster) -> Option<f32> {
    let left_x = metadata_f32(left, "image_x");
    let left_y = metadata_f32(left, "image_y");
    let right_x =
        metadata_f32(right, "projected_image_x").or_else(|| metadata_f32(right, "image_x"));
    let right_y =
        metadata_f32(right, "projected_image_y").or_else(|| metadata_f32(right, "image_y"));
    left_x
        .zip(left_y)
        .zip(right_x.zip(right_y))
        .map(|((lx, ly), (rx, ry))| ((lx - rx).powi(2) + (ly - ry).powi(2)).sqrt())
        .or_else(|| {
            let right_x = metadata_f32(right, "image_x");
            let right_y = metadata_f32(right, "image_y");
            let left_x =
                metadata_f32(left, "projected_image_x").or_else(|| metadata_f32(left, "image_x"));
            let left_y =
                metadata_f32(left, "projected_image_y").or_else(|| metadata_f32(left, "image_y"));
            left_x
                .zip(left_y)
                .zip(right_x.zip(right_y))
                .map(|((lx, ly), (rx, ry))| ((lx - rx).powi(2) + (ly - ry).powi(2)).sqrt())
        })
}

fn pose_distance_m(left: &DiscoveredCluster, right: &DiscoveredCluster) -> Option<f32> {
    left.estimated_pose
        .zip(right.estimated_pose)
        .map(|(left, right)| {
            ((left.x_m - right.x_m).powi(2) + (left.y_m - right.y_m).powi(2)).sqrt()
        })
}

fn lag_score(lag_ms: u64, min_ms: u64, max_ms: u64) -> f32 {
    if lag_ms < min_ms || lag_ms > max_ms {
        return 0.0;
    }
    let midpoint = (min_ms + max_ms) as f32 / 2.0;
    let half_span = (max_ms.saturating_sub(min_ms)).max(1) as f32 / 2.0;
    (1.0 - ((lag_ms as f32 - midpoint).abs() / half_span)).clamp(0.1, 1.0)
}

fn metadata_f32(cluster: &DiscoveredCluster, key: &str) -> Option<f32> {
    cluster
        .metadata
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .map(|value| value as f32)
}

fn metadata_u64(cluster: &DiscoveredCluster, key: &str) -> u64 {
    cluster
        .metadata
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default()
}

fn metadata_bool(cluster: &DiscoveredCluster, key: &str) -> bool {
    cluster
        .metadata
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn metadata_string(cluster: &DiscoveredCluster, key: &str) -> Option<String> {
    cluster
        .metadata
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}
