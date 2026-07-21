use super::*;
use pete_core::{FeatureModality, FeatureType, Provenance, VectorRef};

fn candidate(
    left: &str,
    right: &str,
    decision: BindingDecision,
    evidence: Vec<BindingEvidence>,
) -> BindingCandidate {
    BindingCandidate {
        left_cluster_id: left.to_string(),
        right_cluster_id: right.to_string(),
        relation: BindingRelation::LikelySameEntity,
        confidence: match decision {
            BindingDecision::Accept => 0.88,
            BindingDecision::Reject => 0.2,
            _ => 0.48,
        },
        decision,
        reason: "test decision reason".to_string(),
        evidence,
    }
}

fn evidence(kind: BindingEvidenceKind, reason: &str) -> BindingEvidence {
    BindingEvidence {
        kind,
        score: 0.7,
        reason: reason.to_string(),
    }
}

fn diagnostic_document() -> GraphIntelligenceDocument {
    let mut feature = Feature::new(
        FeatureType::FaceObservation,
        FeatureModality::Vision,
        100,
        0.82,
        Provenance::direct().with_stage("test"),
    )
    .with_source_frame("frame-a")
    .with_vector_ref(VectorRef::new("faces", "face-vector-a"))
    .with_metadata(json!({
        "raw_vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "caption": "face candidate"
    }));
    let feature_id = feature.id;
    feature.metadata["large_text"] = json!("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz");

    let accepted = candidate(
        "face:a",
        "voice:a",
        BindingDecision::Accept,
        vec![evidence(
            BindingEvidenceKind::TemporalOverlap,
            "same window",
        )],
    );
    let rejected = candidate(
        "face:a",
        "voice:b",
        BindingDecision::Reject,
        vec![evidence(
            BindingEvidenceKind::Contradiction,
            "voice belongs to another face",
        )],
    );
    let ambiguous = candidate(
        "face:a",
        "voice:c",
        BindingDecision::HoldAmbiguous,
        vec![evidence(
            BindingEvidenceKind::SimultaneousConflict,
            "two voices are active",
        )],
    );
    GraphIntelligenceDocument {
        id: "doc:test".to_string(),
        t_ms: 120,
        provenance: "test".to_string(),
        features: vec![feature],
        clusters: vec![
            DiscoveredCluster::new(
                "face:a",
                Modality::Vision,
                DiscoveredClusterKind::Face,
                100,
                0.8,
            )
            .with_feature_ids(vec![feature_id]),
            DiscoveredCluster::new(
                "voice:a",
                Modality::Audio,
                DiscoveredClusterKind::Voice,
                100,
                0.8,
            ),
        ],
        binding_candidates: vec![accepted.clone(), rejected, ambiguous],
        binding_edges: vec![BindingEdge {
            left_cluster_id: accepted.left_cluster_id.clone(),
            right_cluster_id: accepted.right_cluster_id.clone(),
            relation: accepted.relation.clone(),
            confidence: 0.9,
            evidence_count: 2,
            decay_per_tick: 0.01,
            last_seen_ms: 120,
        }],
        tracking_hypotheses: vec![
            TrackingHypothesis {
                id: "hypothesis:face:a:ada".to_string(),
                family_id: "face:a".to_string(),
                kind: TrackingHypothesisKind::FaceIdentity,
                target_id: Some("entity:ada".to_string()),
                observation_ids: vec!["obs:a".to_string()],
                binding_candidate_ids: vec!["candidate:a".to_string()],
                confidence: 0.51,
                evidence: vec![evidence(
                    BindingEvidenceKind::VectorSimilarity,
                    "close face vector",
                )],
                contradictions: Vec::new(),
                state: HypothesisState::Winning,
                first_seen_ms: 100,
                last_updated_ms: 120,
            },
            TrackingHypothesis {
                id: "hypothesis:face:a:grace".to_string(),
                family_id: "face:a".to_string(),
                kind: TrackingHypothesisKind::FaceIdentity,
                target_id: Some("entity:grace".to_string()),
                observation_ids: vec!["obs:a".to_string()],
                binding_candidate_ids: vec!["candidate:b".to_string()],
                confidence: 0.49,
                evidence: vec![evidence(
                    BindingEvidenceKind::VectorSimilarity,
                    "also close",
                )],
                contradictions: vec!["face vector close to two people".to_string()],
                state: HypothesisState::NeedsReview,
                first_seen_ms: 100,
                last_updated_ms: 120,
            },
        ],
        constellations: vec![Constellation {
            id: "constellation:person:a".to_string(),
            kind_hint: Some("person".to_string()),
            member_cluster_ids: vec!["face:a".to_string(), "voice:a".to_string()],
            member_binding_ids: vec![binding_candidate_id(&accepted)],
            supporting_feature_ids: Vec::new(),
            supporting_entity_ids: vec!["entity:ada".to_string()],
            supporting_place_cells: vec![PlaceCellKey { x: 1, y: 2 }],
            confidence: 0.55,
            stability: 0.35,
            prediction_value: 0.4,
            first_seen_ms: 100,
            last_seen_ms: 120,
            evidence_count: 1,
            state: ConstellationState::Candidate,
            notes: vec!["missing voice confirmation".to_string()],
        }],
        associations: vec![AssociationEdge {
            id: "association:face-predicts-voice".to_string(),
            from_id: "face:a".to_string(),
            to_id: "voice:a".to_string(),
            relation: AssociationRelation::Predicts,
            confidence: 0.62,
            evidence_count: 3,
            prediction_gain: 0.2,
            contradiction_count: 1,
            first_seen_ms: 100,
            last_seen_ms: 120,
            examples: vec![AssociationExample {
                frame_id: Some("frame-a".to_string()),
                t_ms: 120,
                reason: "face preceded voice".to_string(),
                score: 0.7,
            }],
        }],
        predictions: vec![PredictionRecord {
            id: "prediction:voice".to_string(),
            target_id: "voice:a".to_string(),
            predicted: "voice continues".to_string(),
            confidence: 0.6,
            t_ms: 120,
            state: "open".to_string(),
            reason: "association predicts voice".to_string(),
        }],
        surprises: vec![SurpriseRecord {
            id: "surprise:voice".to_string(),
            target_id: "prediction:voice".to_string(),
            observed: "voice stopped".to_string(),
            surprise: 0.8,
            confidence: 0.7,
            t_ms: 130,
            reason: "speaker stopped unexpectedly".to_string(),
        }],
        ..GraphIntelligenceDocument::default()
    }
}

#[test]
fn cognitive_report_serializes_and_summarizes_sensitive_data() {
    let report = CognitiveDiagnosticsReport::from_graph_document(&diagnostic_document());
    let value = serde_json::to_value(&report).expect("serializable report");

    assert_eq!(value["summary"]["feature_count"], 1);
    assert_eq!(
        value["features"]["items"][0]["metadata_summary"]["raw_vector"]["kind"],
        "array"
    );
    assert!(value["features"]["items"][0]["metadata_summary"]["raw_vector"]["vector"].is_null());
}

#[test]
fn binding_inspector_includes_accepted_rejected_and_ambiguous_candidates() {
    let report = CognitiveDiagnosticsReport::from_graph_document(&diagnostic_document());

    assert_eq!(report.summary.accepted_binding_count, 1);
    assert_eq!(report.summary.rejected_binding_count, 1);
    assert_eq!(report.summary.ambiguous_binding_count, 1);
    assert!(report
        .bindings
        .items
        .iter()
        .any(|item| item.accepted_binding_edge_id.is_some()));
    assert!(report
        .bindings
        .items
        .iter()
        .any(|item| item.rejection_reason.is_some()));
    assert!(report
        .bindings
        .items
        .iter()
        .any(|item| item.ambiguity_reason.is_some()));
}

#[test]
fn hypothesis_constellation_active_learning_and_summary_are_inspectable() {
    let report = CognitiveDiagnosticsReport::from_graph_document(&diagnostic_document());

    assert_eq!(report.hypotheses.families.len(), 1);
    assert_eq!(report.hypotheses.families[0].competing_hypotheses.len(), 2);
    assert!(report.constellations.items[0]
        .member_clusters
        .contains(&"face:a".to_string()));
    assert!(!report.constellations.items[0]
        .missing_expected_evidence
        .is_empty());
    assert!(!report.active_learning.open_questions.is_empty());
    assert_eq!(report.summary.cluster_count, 2);
    assert_eq!(report.summary.constellation_count, 1);
    assert_eq!(report.summary.association_count, 1);
    assert_eq!(report.summary.prediction_failure_count, 1);
}

#[test]
fn learning_cycle_turns_person_greeting_into_replay_and_training() {
    let feature = Feature::new(
        FeatureType::FaceObservation,
        FeatureModality::Vision,
        1_000,
        0.88,
        Provenance::direct().with_stage("observe"),
    );
    let feature_id = feature.id;
    let accepted = BindingCandidate {
        left_cluster_id: "cluster:person-face:ada".to_string(),
        right_cluster_id: "cluster:greeting:hello".to_string(),
        relation: BindingRelation::NamedBy,
        evidence: vec![
            evidence(
                BindingEvidenceKind::TemporalOverlap,
                "face and greeting co-occurred",
            ),
            evidence(
                BindingEvidenceKind::RepeatedCooccurrence,
                "greeting repeated",
            ),
        ],
        confidence: 0.91,
        decision: BindingDecision::Accept,
        reason: "person greeting binding accepted".to_string(),
    };
    let document = GraphIntelligenceDocument {
        id: "doc:greeting".to_string(),
        t_ms: 1_000,
        frame_id: Some("frame:greeting".to_string()),
        provenance: "integration_test".to_string(),
        features: vec![feature],
        clusters: vec![
            DiscoveredCluster::new(
                "cluster:person-face:ada",
                Modality::Vision,
                DiscoveredClusterKind::Face,
                1_000,
                0.9,
            )
            .with_feature_ids(vec![feature_id]),
            DiscoveredCluster::new(
                "cluster:greeting:hello",
                Modality::Language,
                DiscoveredClusterKind::Label,
                1_000,
                0.85,
            ),
        ],
        binding_candidates: vec![accepted.clone()],
        constellations: vec![Constellation {
            id: "constellation:person:greeting".to_string(),
            kind_hint: Some("person".to_string()),
            member_cluster_ids: vec![
                "cluster:person-face:ada".to_string(),
                "cluster:greeting:hello".to_string(),
            ],
            member_binding_ids: vec![binding_candidate_id(&accepted)],
            supporting_feature_ids: vec![feature_id],
            supporting_entity_ids: vec!["entity:person:ada".to_string()],
            supporting_place_cells: Vec::new(),
            confidence: 0.82,
            stability: 0.78,
            prediction_value: 0.72,
            first_seen_ms: 900,
            last_seen_ms: 1_000,
            evidence_count: 3,
            state: ConstellationState::Stable,
            notes: Vec::new(),
        }],
        associations: vec![AssociationEdge {
            id: "association:person-predicts-greeting".to_string(),
            from_id: "constellation:person:greeting".to_string(),
            to_id: "outcome:greeting".to_string(),
            relation: AssociationRelation::Predicts,
            confidence: 0.74,
            evidence_count: 4,
            prediction_gain: 0.22,
            contradiction_count: 0,
            first_seen_ms: 900,
            last_seen_ms: 1_000,
            examples: vec![AssociationExample {
                frame_id: Some("frame:greeting".to_string()),
                t_ms: 1_000,
                reason: "person appearance predicted greeting".to_string(),
                score: 0.8,
            }],
        }],
        predictions: vec![PredictionRecord {
            id: "prediction:greeting".to_string(),
            target_id: "outcome:greeting".to_string(),
            predicted: "person says hello".to_string(),
            confidence: 0.76,
            t_ms: 1_000,
            state: "succeeded".to_string(),
            reason: "association predicted greeting".to_string(),
        }],
        ..GraphIntelligenceDocument::default()
    };

    let report = CognitiveDiagnosticsReport::from_graph_document(&document);
    let event_kinds = report
        .learning_cycle
        .learning_events
        .iter()
        .map(|event| event.event.clone())
        .collect::<BTreeSet<_>>();

    assert!(event_kinds.contains(&LearningEvent::FeatureObserved));
    assert!(event_kinds.contains(&LearningEvent::ClusterStrengthened));
    assert!(event_kinds.contains(&LearningEvent::BindingAccepted));
    assert!(event_kinds.contains(&LearningEvent::ConstellationPromoted));
    assert!(event_kinds.contains(&LearningEvent::AssociationStrengthened));
    assert!(event_kinds.contains(&LearningEvent::PredictionSucceeded));
    assert!(report
        .learning_cycle
        .training_examples
        .iter()
        .any(|example| example.target_model == "prediction_model"
            && example.label == "prediction_succeeded"));
    assert!(report
        .learning_cycle
        .replay_items
        .iter()
        .any(|item| item.target_id == "constellation:person:greeting"
            && item.curriculum.priority >= 0.5));
    assert_eq!(
        report.summary.training_example_count,
        report.learning_cycle.training_examples.len()
    );
}

#[test]
fn learning_cycle_marks_human_confirmed_ambiguity_as_trusted_training() {
    let hypothesis = TrackingHypothesis {
        id: "hypothesis:face:unknown:ada".to_string(),
        family_id: "family:ambiguous-face".to_string(),
        kind: TrackingHypothesisKind::FaceIdentity,
        target_id: Some("entity:person:ada".to_string()),
        observation_ids: vec!["face-vector:ambiguous".to_string()],
        binding_candidate_ids: vec!["binding:ambiguous-face-ada".to_string()],
        confidence: 0.93,
        evidence: vec![
            evidence(
                BindingEvidenceKind::VectorSimilarity,
                "face vector is close",
            ),
            BindingEvidence {
                kind: BindingEvidenceKind::HumanConfirmed,
                score: 1.0,
                reason: "human confirmed this is Ada".to_string(),
            },
        ],
        contradictions: Vec::new(),
        state: HypothesisState::Promoted,
        first_seen_ms: 2_000,
        last_updated_ms: 2_400,
    };
    let document = GraphIntelligenceDocument {
        id: "doc:human-confirmed".to_string(),
        t_ms: 2_400,
        provenance: "integration_test".to_string(),
        binding_candidates: vec![BindingCandidate {
            left_cluster_id: "cluster:face:unknown".to_string(),
            right_cluster_id: "entity:person:ada".to_string(),
            relation: BindingRelation::LikelySameEntity,
            evidence: hypothesis.evidence.clone(),
            confidence: 0.93,
            decision: BindingDecision::Accept,
            reason: "human confirmation promoted the binding".to_string(),
        }],
        tracking_hypotheses: vec![
            hypothesis,
            TrackingHypothesis {
                id: "hypothesis:face:unknown:other".to_string(),
                family_id: "family:ambiguous-face".to_string(),
                kind: TrackingHypothesisKind::FaceIdentity,
                target_id: Some("entity:person:other".to_string()),
                observation_ids: vec!["face-vector:ambiguous".to_string()],
                binding_candidate_ids: vec!["binding:ambiguous-face-other".to_string()],
                confidence: 0.21,
                evidence: vec![evidence(
                    BindingEvidenceKind::SimultaneousConflict,
                    "human confirmation rejected this competitor",
                )],
                contradictions: vec!["human confirmation selected Ada".to_string()],
                state: HypothesisState::Rejected,
                first_seen_ms: 2_000,
                last_updated_ms: 2_400,
            },
        ],
        human_reviews: vec![HumanReviewRecord {
            id: "human-review:ada".to_string(),
            target_id: "hypothesis:face:unknown:ada".to_string(),
            target_kind: ActiveLearningTargetKind::TrackingHypothesis,
            confidence: 1.0,
            t_ms: 2_400,
            confirmation: "That face is Ada".to_string(),
            reviewer: Some("test-human".to_string()),
        }],
        ..GraphIntelligenceDocument::default()
    };

    let report = CognitiveDiagnosticsReport::from_graph_document(&document);

    assert!(report
        .learning_cycle
        .learning_events
        .iter()
        .any(|event| event.event == LearningEvent::HumanCorrection && event.trusted));
    assert!(report
        .learning_cycle
        .training_examples
        .iter()
        .any(
            |example| example.kind == TrainingExampleKind::HumanTrustedPositive
                && example.trusted
                && example.weight >= 0.9
        ));
    assert!(report.learning_cycle.replay_items.iter().any(|item| item
        .curriculum
        .human_confirmation
        == 1.0
        && item.curriculum.priority >= 0.75));
    assert_eq!(report.learning_cycle.hypotheses_promoted, 1);
}
