fn cluster_ids_from_observation(
    observation: &ConstellationObservation,
    accepted_bindings: &[&BindingCandidate],
) -> Vec<String> {
    let mut ids = accepted_bindings
        .iter()
        .flat_map(|candidate| {
            [
                candidate.left_cluster_id.clone(),
                candidate.right_cluster_id.clone(),
            ]
        })
        .chain(
            observation
                .clusters
                .iter()
                .map(|cluster| cluster.id.clone()),
        )
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    ids.sort();
    ids
}

fn merge_constellation_observation(
    constellation: &mut Constellation,
    observation: &ConstellationObservation,
    member_cluster_ids: &[String],
    member_binding_ids: &[String],
) {
    merge_unique(&mut constellation.member_cluster_ids, member_cluster_ids);
    merge_unique(&mut constellation.member_binding_ids, member_binding_ids);
    let feature_ids = observation
        .clusters
        .iter()
        .flat_map(|cluster| cluster.feature_ids.iter().copied())
        .collect::<Vec<_>>();
    merge_unique(&mut constellation.supporting_feature_ids, &feature_ids);
    merge_unique(
        &mut constellation.supporting_entity_ids,
        &observation.active_entity_ids,
    );
    merge_unique(
        &mut constellation.supporting_place_cells,
        &observation.place_cells,
    );
    merge_unique(&mut constellation.notes, &observation.llm_notes);
    constellation.last_seen_ms = observation.t_ms;
    constellation.evidence_count = constellation.evidence_count.saturating_add(1);
    constellation.prediction_value = (constellation.prediction_value * 0.75
        + observation.prediction_value * 0.25)
        .clamp(0.0, 1.0);
    if constellation.kind_hint.is_none() {
        constellation.kind_hint =
            infer_constellation_kind(&observation.clusters).map(|kind| kind.as_str().to_string());
    }
}

fn refresh_constellation_scores(
    constellation: &mut Constellation,
    observation: &ConstellationObservation,
    config: &ConstellationEngineConfig,
) {
    let accepted_bindings = observation
        .accepted_bindings
        .iter()
        .filter(|candidate| candidate.decision == BindingDecision::Accept)
        .collect::<Vec<_>>();
    let positive_binding_score = if accepted_bindings.is_empty() {
        0.0
    } else {
        accepted_bindings
            .iter()
            .map(|candidate| candidate.confidence.clamp(0.0, 1.0))
            .sum::<f32>()
            / accepted_bindings.len() as f32
    };
    let recurrence_score = (constellation.evidence_count as f32
        / config.min_evidence_for_stable.max(1) as f32)
        .clamp(0.0, 1.0);
    let cluster_score = (constellation.member_cluster_ids.len() as f32
        / config.min_clusters_for_stable.max(1) as f32)
        .clamp(0.0, 1.0);
    let binding_score = (constellation.member_binding_ids.len() as f32
        / config.min_bindings_for_stable.max(1) as f32)
        .clamp(0.0, 1.0);
    let contradiction_count = observation
        .accepted_bindings
        .iter()
        .filter(|candidate| binding_has_conflict(candidate))
        .count();
    let contradiction_penalty = (contradiction_count as f32 * 0.25).min(0.65);

    constellation.stability =
        (recurrence_score * 0.5 + binding_score * 0.3 + cluster_score * 0.2).clamp(0.0, 1.0);
    constellation.confidence = (positive_binding_score * 0.45
        + constellation.stability * 0.35
        + constellation.prediction_value * 0.2
        - contradiction_penalty)
        .clamp(0.0, 1.0);

    if evidence_suggests_split(observation) {
        constellation.state = ConstellationState::SplitNeeded;
        return;
    }
    if contradiction_count > 0 {
        constellation.state = ConstellationState::Ambiguous;
        return;
    }
    let promotable = constellation.member_cluster_ids.len() >= config.min_clusters_for_stable
        && constellation.member_binding_ids.len() >= config.min_bindings_for_stable
        && constellation.evidence_count >= config.min_evidence_for_stable
        && constellation.confidence >= config.promotion_confidence_threshold
        && constellation.prediction_value >= config.min_prediction_value_for_stable;
    constellation.state = if promotable {
        ConstellationState::Stable
    } else {
        ConstellationState::Candidate
    };
}

fn binding_has_conflict(candidate: &BindingCandidate) -> bool {
    candidate.evidence.iter().any(|evidence| {
        matches!(
            evidence.kind,
            BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
        )
    })
}

fn evidence_suggests_split(observation: &ConstellationObservation) -> bool {
    observation.accepted_bindings.iter().any(|candidate| {
        candidate
            .evidence
            .iter()
            .any(|evidence| evidence.kind == BindingEvidenceKind::SimultaneousConflict)
    }) || observation.llm_notes.iter().any(|note| {
        let note = note.to_ascii_lowercase();
        note.contains("split")
            || note.contains("fused")
            || note.contains("fusion")
            || note.contains("two patterns")
    })
}

fn infer_constellation_kind(clusters: &[DiscoveredCluster]) -> Option<ConstellationKind> {
    let kinds = clusters
        .iter()
        .map(|cluster| cluster.kind.clone())
        .collect::<BTreeSet<_>>();
    if kinds.contains(&DiscoveredClusterKind::Face) || kinds.contains(&DiscoveredClusterKind::Voice)
    {
        Some(ConstellationKind::Person)
    } else if kinds.contains(&DiscoveredClusterKind::Action)
        || kinds.contains(&DiscoveredClusterKind::Outcome)
        || kinds.contains(&DiscoveredClusterKind::BodyState)
    {
        Some(ConstellationKind::ActionOutcome)
    } else if kinds.contains(&DiscoveredClusterKind::Place) {
        Some(ConstellationKind::Place)
    } else if kinds.contains(&DiscoveredClusterKind::Object)
        || kinds.contains(&DiscoveredClusterKind::Geometry)
        || kinds.contains(&DiscoveredClusterKind::RgbImage)
    {
        Some(ConstellationKind::Object)
    } else {
        None
    }
}

fn overlap_score(matched: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        (matched as f32 / total as f32).clamp(0.0, 1.0)
    }
}

fn stale_penalty(age_ms: u64, stale_after_ms: u64) -> f32 {
    if stale_after_ms == 0 || age_ms <= stale_after_ms {
        0.0
    } else {
        ((age_ms - stale_after_ms) as f32 / (stale_after_ms * 4) as f32).clamp(0.0, 0.6)
    }
}

fn intersection_count<T>(left: &[T], right: &[T]) -> usize
where
    T: Ord + Clone,
{
    let left = left.iter().cloned().collect::<BTreeSet<_>>();
    let right = right.iter().cloned().collect::<BTreeSet<_>>();
    left.intersection(&right).count()
}

fn merge_unique<T>(target: &mut Vec<T>, incoming: &[T])
where
    T: Ord + Clone,
{
    let mut seen = target.iter().cloned().collect::<BTreeSet<_>>();
    for item in incoming {
        if seen.insert(item.clone()) {
            target.push(item.clone());
        }
    }
    target.sort();
}

fn association_edge_id(from_id: &str, to_id: &str, relation: &AssociationRelation) -> String {
    format!(
        "association:{}:{}:{}",
        association_relation_slug(relation),
        stable_slug(from_id),
        stable_slug(to_id)
    )
}

fn association_relation_slug(relation: &AssociationRelation) -> &'static str {
    match relation {
        AssociationRelation::CoOccursWith => "co-occurs-with",
        AssociationRelation::Predicts => "predicts",
        AssociationRelation::Follows => "follows",
        AssociationRelation::Suppresses => "suppresses",
        AssociationRelation::Contradicts => "contradicts",
        AssociationRelation::Explains => "explains",
        AssociationRelation::Enables => "enables",
        AssociationRelation::Prevents => "prevents",
        AssociationRelation::PartOf => "part-of",
    }
}

fn dedupe_association_items(items: Vec<AssociationItem>) -> Vec<AssociationItem> {
    let mut by_id = BTreeMap::<String, AssociationItem>::new();
    for item in items {
        by_id
            .entry(item.id.clone())
            .and_modify(|existing| {
                existing.confidence = existing.confidence.max(item.confidence);
            })
            .or_insert(item);
    }
    by_id.into_values().collect()
}

fn canonical_association_pair<'a>(
    left: &'a AssociationItem,
    right: &'a AssociationItem,
) -> (&'a str, &'a str) {
    if left.id <= right.id {
        (&left.id, &right.id)
    } else {
        (&right.id, &left.id)
    }
}

fn sequence_relation(
    to: &AssociationItem,
    lag_ms: u64,
    config: &AssociationLearningConfig,
) -> AssociationRelation {
    if matches!(
        to.kind,
        AssociationItemKind::Outcome
            | AssociationItemKind::Prediction
            | AssociationItemKind::Surprise
            | AssociationItemKind::BodyState
    ) && lag_ms <= config.long_sequence_window_ms
    {
        AssociationRelation::Predicts
    } else if lag_ms <= config.short_sequence_window_ms {
        AssociationRelation::Follows
    } else {
        AssociationRelation::Follows
    }
}

fn lag_score_for_association(lag_ms: u64) -> f32 {
    match lag_ms {
        0..=500 => 1.0,
        501..=2_000 => 0.8,
        2_001..=10_000 => 0.55,
        _ => 0.2,
    }
}

fn approximate_mutual_information(p_b: f32, p_b_given_a: f32) -> f32 {
    let p_b = p_b.clamp(0.001, 0.999);
    let p_b_given_a = p_b_given_a.clamp(0.001, 0.999);
    if p_b_given_a <= p_b {
        return 0.0;
    }
    (p_b_given_a * (p_b_given_a / p_b).ln()).clamp(0.0, 1.0)
}
