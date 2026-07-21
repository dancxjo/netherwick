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
