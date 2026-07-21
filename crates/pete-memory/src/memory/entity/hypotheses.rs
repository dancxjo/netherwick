fn qualify_binding_candidate(
    entity: &EntityHypothesis,
    artifact: &VectorArtifact,
    kind: VectorBindingKind,
    left_cluster_id: String,
    right_cluster_id: String,
    t_ms: u64,
    cell_key: Option<PlaceCellKey>,
    plausible_count: usize,
    current_object_observed: bool,
) -> BindingCandidate {
    let mut evidence = Vec::new();
    evidence.push(BindingEvidence {
        kind: BindingEvidenceKind::VectorSimilarity,
        score: 0.45,
        reason: "vector artifact proposes a possible cross-modal correspondence".to_string(),
    });

    if artifact.source_id.as_deref() == Some(entity.id.as_str()) {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::HumanConfirmed,
            score: 1.0,
            reason: "vector source explicitly names this entity".to_string(),
        });
    } else if artifact
        .source_id
        .as_deref()
        .is_some_and(|source_id| source_id.starts_with("entity:"))
    {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::Contradiction,
            score: 1.0,
            reason: format!(
                "vector source names {}, not {}",
                artifact.source_id.as_deref().unwrap_or("unknown"),
                entity.id
            ),
        });
    }
    if t_ms.saturating_sub(entity.last_seen_ms) <= 1_000 {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::TemporalOverlap,
            score: 0.75,
            reason: "entity was observed in the current temporal window".to_string(),
        });
    }
    if cell_key
        .map(|key| entity.location_cells.contains(&key))
        .unwrap_or(false)
    {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::SpatialOverlap,
            score: 0.75,
            reason: "entity has a compatible current map cell".to_string(),
        });
    }
    if current_object_observed {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::ProjectionAgreement,
            score: 0.7,
            reason: "a current object observation anchors this entity".to_string(),
        });
    }
    if plausible_count == 1 {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::SingleCandidateContext,
            score: 0.65,
            reason: "only one plausible entity matched this vector context".to_string(),
        });
    } else if plausible_count > 1 {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::SimultaneousConflict,
            score: 0.8,
            reason: match kind {
                VectorBindingKind::Face => {
                    "face vector close to multiple active person entities".to_string()
                }
                VectorBindingKind::Voice => {
                    "voice observed while multiple person hypotheses are active".to_string()
                }
                VectorBindingKind::Scene => {
                    "scene vector has multiple spatially plausible entities".to_string()
                }
            },
        });
    }
    if entity.constellation.binding_edges.iter().any(|edge| {
        edge.left_cluster_id == right_cluster_id || edge.right_cluster_id == right_cluster_id
    }) {
        evidence.push(BindingEvidence {
            kind: BindingEvidenceKind::RepeatedCooccurrence,
            score: 0.8,
            reason: "prior binding history supports this correspondence".to_string(),
        });
    }

    let has_human_confirmation = evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::HumanConfirmed);
    let has_conflict = evidence.iter().any(|item| {
        matches!(
            item.kind,
            BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
        )
    });
    let has_hard_contradiction = evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::Contradiction);
    let independent_positive_kinds = evidence
        .iter()
        .filter(|item| {
            !matches!(
                item.kind,
                BindingEvidenceKind::Contradiction
                    | BindingEvidenceKind::SimultaneousConflict
                    | BindingEvidenceKind::VectorSimilarity
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
            "candidate contradicts explicit entity source evidence".to_string(),
        )
    } else if has_human_confirmation {
        (
            BindingDecision::Accept,
            "human-confirmed or explicit source binding".to_string(),
        )
    } else if has_conflict {
        (
            BindingDecision::HoldAmbiguous,
            match kind {
                VectorBindingKind::Face => "face vector close to multiple active person entities",
                VectorBindingKind::Voice => {
                    "voice observed while multiple person hypotheses active"
                }
                VectorBindingKind::Scene => {
                    "scene vector active but multiple spatially compatible entities exist"
                }
            }
            .to_string(),
        )
    } else if independent_positive_kinds >= 2 {
        (
            BindingDecision::Accept,
            "candidate has at least two independent supporting evidence types".to_string(),
        )
    } else if evidence.len() == 1 {
        (
            BindingDecision::CollectMoreEvidence,
            "single vector similarity without supporting temporal/spatial evidence".to_string(),
        )
    } else {
        (
            BindingDecision::CollectMoreEvidence,
            "projection agreement missing or evidence is not yet independent".to_string(),
        )
    };

    BindingCandidate {
        left_cluster_id,
        right_cluster_id,
        relation: match kind {
            VectorBindingKind::Face | VectorBindingKind::Voice => BindingRelation::LikelySameEntity,
            VectorBindingKind::Scene => BindingRelation::ProjectsTo,
        },
        evidence,
        confidence: confidence.clamp(0.0, 1.0),
        decision,
        reason,
    }
}

fn vector_cluster_id(kind: VectorBindingKind, point_id: &str) -> String {
    let key = match kind {
        VectorBindingKind::Face => format!("face:{point_id}"),
        VectorBindingKind::Voice => format!("voice:{point_id}"),
        VectorBindingKind::Scene => format!("scene:{point_id}"),
    };
    let modality = match kind {
        VectorBindingKind::Face => Modality::Vision,
        VectorBindingKind::Voice => Modality::Audio,
        VectorBindingKind::Scene => Modality::Depth,
    };
    format!("cluster:{}:{}", modality.as_str(), stable_slug(&key))
}

fn binding_evidence_kind_rank(kind: &BindingEvidenceKind) -> u8 {
    match kind {
        BindingEvidenceKind::TemporalOverlap => 1,
        BindingEvidenceKind::SpatialOverlap => 2,
        BindingEvidenceKind::VectorSimilarity => 3,
        BindingEvidenceKind::ProjectionAgreement => 4,
        BindingEvidenceKind::PoseAgreement => 5,
        BindingEvidenceKind::RepeatedCooccurrence => 6,
        BindingEvidenceKind::SingleCandidateContext => 7,
        BindingEvidenceKind::HumanConfirmed => 8,
        BindingEvidenceKind::LlmSuggested => 9,
        BindingEvidenceKind::Contradiction => 10,
        BindingEvidenceKind::SimultaneousConflict => 11,
    }
}

fn tracking_kind_from_vector(kind: VectorBindingKind) -> TrackingHypothesisKind {
    match kind {
        VectorBindingKind::Face => TrackingHypothesisKind::FaceIdentity,
        VectorBindingKind::Voice => TrackingHypothesisKind::VoiceIdentity,
        VectorBindingKind::Scene => TrackingHypothesisKind::PlaceMatch,
    }
}

fn tracking_kind_slug(kind: &TrackingHypothesisKind) -> &'static str {
    match kind {
        TrackingHypothesisKind::FaceIdentity => "face-identity",
        TrackingHypothesisKind::VoiceIdentity => "voice-identity",
        TrackingHypothesisKind::CrossModalBinding => "cross-modal",
        TrackingHypothesisKind::PlaceMatch => "place",
        TrackingHypothesisKind::ObjectContinuity => "object-continuity",
        TrackingHypothesisKind::Other => "other",
    }
}

fn tracking_family_id(kind: VectorBindingKind, observation_id: &str) -> String {
    format!(
        "{}:{}",
        tracking_kind_slug(&tracking_kind_from_vector(kind)),
        observation_id
    )
}

fn binding_candidate_id(candidate: &BindingCandidate) -> String {
    format!(
        "candidate:{}:{}:{}",
        stable_slug(&candidate.left_cluster_id),
        stable_slug(&candidate.right_cluster_id),
        binding_relation_label_for_id(&candidate.relation)
    )
}

fn binding_relation_label_for_id(relation: &BindingRelation) -> &'static str {
    match relation {
        BindingRelation::CooccursInTime => "time",
        BindingRelation::CooccursInEstimatedSpace => "space",
        BindingRelation::MovesTogether => "moves",
        BindingRelation::PredictsSameFutureEvents => "future",
        BindingRelation::NamedBy => "named-by",
        BindingRelation::ProjectsTo => "projects-to",
        BindingRelation::HasColorAtPose => "color-pose",
        BindingRelation::LikelySameEntity => "same-entity",
        BindingRelation::ExplainsOutcome => "outcome",
        BindingRelation::Contradicts => "contradicts",
        BindingRelation::RequiresReview => "review",
    }
}

fn has_hard_contradiction(evidence: &[BindingEvidence]) -> bool {
    evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::Contradiction)
}

fn score_hypothesis_evidence(evidence: &[BindingEvidence], repeated_observations: f32) -> f32 {
    if evidence.is_empty() {
        return 0.0;
    }
    let human_confirmed = evidence
        .iter()
        .any(|item| item.kind == BindingEvidenceKind::HumanConfirmed);
    let positive = evidence
        .iter()
        .filter(|item| {
            !matches!(
                item.kind,
                BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
            )
        })
        .map(|item| item.score.clamp(0.0, 1.0))
        .sum::<f32>();
    let positive_count = evidence
        .iter()
        .filter(|item| {
            !matches!(
                item.kind,
                BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
            )
        })
        .count()
        .max(1) as f32;
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
        .len() as f32;
    let contradiction_count = evidence
        .iter()
        .filter(|item| {
            matches!(
                item.kind,
                BindingEvidenceKind::Contradiction | BindingEvidenceKind::SimultaneousConflict
            )
        })
        .count() as f32;
    let repetition_bonus = ((repeated_observations - 1.0).max(0.0) * 0.08).min(0.18);
    let independence_bonus = (independent_positive_kinds * 0.08).min(0.24);
    let mut score = positive / positive_count + repetition_bonus + independence_bonus;
    if human_confirmed {
        score = score.max(0.92);
    }
    score -= contradiction_count * 0.18;
    score.clamp(0.0, 1.0)
}
