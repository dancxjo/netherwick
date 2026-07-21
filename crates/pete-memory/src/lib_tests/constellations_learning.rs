#[test]
fn repeated_accepted_bindings_form_candidate_constellation() {
    let mut engine = ConstellationEngine::new();
    let constellation = engine
        .observe(ConstellationObservation {
            t_ms: 1_000,
            clusters: vec![
                test_cluster("face:a", DiscoveredClusterKind::Face, Modality::Vision),
                test_cluster("voice:a", DiscoveredClusterKind::Voice, Modality::Audio),
            ],
            accepted_bindings: vec![accepted_binding(
                "face:a",
                "voice:a",
                BindingRelation::LikelySameEntity,
            )],
            prediction_value: 0.4,
            ..ConstellationObservation::default()
        })
        .expect("candidate");

    assert_eq!(constellation.state, ConstellationState::Candidate);
    assert_eq!(constellation.member_cluster_ids.len(), 2);
    assert_eq!(constellation.member_binding_ids.len(), 1);
}

#[test]
fn candidate_becomes_stable_only_after_repeated_evidence() {
    let mut engine = ConstellationEngine::new();
    let first = engine
        .observe(person_constellation_observation(1_000))
        .expect("first candidate");
    assert_eq!(first.state, ConstellationState::Candidate);

    engine.observe(person_constellation_observation(2_000));
    let stable = engine
        .observe(person_constellation_observation(3_000))
        .expect("stable constellation");

    assert_eq!(stable.state, ConstellationState::Stable);
    assert!(stable.confidence >= engine.config.promotion_confidence_threshold);
    assert_eq!(stable.evidence_count, 3);
}

#[test]
fn constellation_is_not_promoted_from_one_strong_cluster_alone() {
    let mut engine = ConstellationEngine::new();
    for t_ms in [1_000, 2_000, 3_000, 4_000] {
        let admitted = engine.observe(ConstellationObservation {
            t_ms,
            clusters: vec![test_cluster(
                "face:solo",
                DiscoveredClusterKind::Face,
                Modality::Vision,
            )],
            prediction_value: 1.0,
            ..ConstellationObservation::default()
        });
        assert!(admitted.is_none());
    }
    assert!(engine.constellations.is_empty());
}

#[test]
fn partial_match_retrieves_known_constellation() {
    let mut engine = ConstellationEngine::new();
    for t_ms in [1_000, 2_000, 3_000] {
        engine.observe(person_constellation_observation(t_ms));
    }

    let matched = engine
        .best_match(&ConstellationQuery {
            t_ms: 4_000,
            cluster_ids: vec!["face:travis".to_string(), "voice:travis".to_string()],
            place_cells: vec![PlaceCellKey { x: 2, y: -1 }],
            ..ConstellationQuery::default()
        })
        .expect("partial match");

    assert!(matched.score >= engine.config.partial_match_threshold);
    assert!(matched
        .matched_cluster_ids
        .contains(&"face:travis".to_string()));
    assert!(matched
        .missing_cluster_ids
        .contains(&"label:travis".to_string()));
}

#[test]
fn missing_modality_does_not_destroy_match() {
    let mut engine = ConstellationEngine::new();
    for t_ms in [1_000, 2_000, 3_000] {
        engine.observe(person_constellation_observation(t_ms));
    }

    let full = engine
        .best_match(&ConstellationQuery {
            t_ms: 4_000,
            cluster_ids: vec![
                "face:travis".to_string(),
                "voice:travis".to_string(),
                "label:travis".to_string(),
            ],
            place_cells: vec![PlaceCellKey { x: 2, y: -1 }],
            ..ConstellationQuery::default()
        })
        .expect("full match");
    let partial = engine
        .best_match(&ConstellationQuery {
            t_ms: 4_000,
            cluster_ids: vec!["face:travis".to_string(), "voice:travis".to_string()],
            place_cells: vec![PlaceCellKey { x: 2, y: -1 }],
            ..ConstellationQuery::default()
        })
        .expect("partial match");

    assert!(partial.score > 0.0);
    assert!(partial.score < full.score);
    assert!(partial.score >= engine.config.partial_match_threshold);
}

#[test]
fn contradiction_lowers_confidence() {
    let mut engine = ConstellationEngine::new();
    for t_ms in [1_000, 2_000, 3_000] {
        engine.observe(person_constellation_observation(t_ms));
    }
    let known_binding_id = engine
        .constellations
        .values()
        .next()
        .unwrap()
        .member_binding_ids
        .first()
        .cloned()
        .unwrap();

    let clean = engine
        .best_match(&ConstellationQuery {
            t_ms: 4_000,
            cluster_ids: vec!["face:travis".to_string(), "voice:travis".to_string()],
            binding_ids: vec![known_binding_id.clone()],
            ..ConstellationQuery::default()
        })
        .expect("clean match");
    let contradicted = engine
        .best_match(&ConstellationQuery {
            t_ms: 4_000,
            cluster_ids: vec!["face:travis".to_string(), "voice:travis".to_string()],
            binding_ids: vec![known_binding_id.clone()],
            contradiction_ids: vec![known_binding_id],
            ..ConstellationQuery::default()
        })
        .expect("contradicted match");

    assert!(contradicted.score < clean.score);
    assert!(contradicted.contradiction_penalty > 0.0);
}

#[test]
fn split_needed_state_appears_when_evidence_suggests_fusion() {
    let mut engine = ConstellationEngine::new();
    let mut binding = accepted_binding(
        "object:patch",
        "geometry:blob",
        BindingRelation::CooccursInEstimatedSpace,
    );
    binding.evidence.push(BindingEvidence {
        kind: BindingEvidenceKind::SimultaneousConflict,
        score: 0.8,
        reason: "two object tracks may have fused".to_string(),
    });

    let constellation = engine
        .observe(ConstellationObservation {
            t_ms: 1_000,
            clusters: vec![
                test_cluster(
                    "object:patch",
                    DiscoveredClusterKind::Object,
                    Modality::Vision,
                ),
                test_cluster(
                    "geometry:blob",
                    DiscoveredClusterKind::Geometry,
                    Modality::Depth,
                ),
            ],
            accepted_bindings: vec![binding],
            prediction_value: 0.4,
            llm_notes: vec!["this may be two fused patterns".to_string()],
            ..ConstellationObservation::default()
        })
        .expect("split-needed constellation");

    assert_eq!(constellation.state, ConstellationState::SplitNeeded);
    assert!(constellation.confidence < 0.9);
}

fn association_item(id: &str, kind: AssociationItemKind, confidence: f32) -> AssociationItem {
    AssociationItem::new(id, kind, confidence)
}

#[test]
fn repeated_cooccurrence_creates_association() {
    let mut engine = AssociationLearningEngine::new();
    for t_ms in [1_000, 1_100, 1_200] {
        engine.observe(AssociationObservation {
            t_ms,
            active_items: vec![
                association_item("cluster:face:travis", AssociationItemKind::Cluster, 0.9),
                association_item("cluster:voice:travis", AssociationItemKind::Cluster, 0.85),
            ],
            ..AssociationObservation::default()
        });
    }

    let id = association_edge_id(
        "cluster:face:travis",
        "cluster:voice:travis",
        &AssociationRelation::CoOccursWith,
    );
    let edge = engine.edges.get(&id).expect("co-occurrence edge");
    assert_eq!(edge.relation, AssociationRelation::CoOccursWith);
    assert_eq!(edge.evidence_count, 3);
    assert!(edge.confidence > 0.4);
}

#[test]
fn repeated_sequence_creates_predicts_or_follows() {
    let mut engine = AssociationLearningEngine::new();
    for base in [1_000, 3_000, 5_000] {
        engine.observe(AssociationObservation {
            t_ms: base,
            active_items: vec![association_item(
                "action:forward",
                AssociationItemKind::Action,
                0.9,
            )],
            ..AssociationObservation::default()
        });
        engine.observe(AssociationObservation {
            t_ms: base + 300,
            outcome_items: vec![association_item(
                "outcome:no-movement",
                AssociationItemKind::Outcome,
                0.95,
            )],
            ..AssociationObservation::default()
        });
    }

    let id = association_edge_id(
        "action:forward",
        "outcome:no-movement",
        &AssociationRelation::Predicts,
    );
    let edge = engine.edges.get(&id).expect("prediction edge");
    assert_eq!(edge.relation, AssociationRelation::Predicts);
    assert!(edge.evidence_count >= 3);
    assert!(edge.prediction_gain > 0.0);

    let predictions = engine.predictions_for(&["action:forward".to_string()], 0.1, 3);
    assert!(predictions
        .iter()
        .any(|prediction| prediction.predicted_id == "outcome:no-movement"));
}

#[test]
fn association_confidence_increases_with_evidence() {
    let mut engine = AssociationLearningEngine::new();
    engine.observe(AssociationObservation {
        t_ms: 1_000,
        active_items: vec![
            association_item("place:charger", AssociationItemKind::Constellation, 0.8),
            association_item("body:charging", AssociationItemKind::BodyState, 0.8),
        ],
        ..AssociationObservation::default()
    });
    let id = association_edge_id(
        "body:charging",
        "place:charger",
        &AssociationRelation::CoOccursWith,
    );
    let first_confidence = engine.edges.get(&id).unwrap().confidence;

    for t_ms in [1_500, 2_000, 2_500] {
        engine.observe(AssociationObservation {
            t_ms,
            active_items: vec![
                association_item("place:charger", AssociationItemKind::Constellation, 0.9),
                association_item("body:charging", AssociationItemKind::BodyState, 0.9),
            ],
            ..AssociationObservation::default()
        });
    }
    let later_confidence = engine.edges.get(&id).unwrap().confidence;

    assert!(later_confidence > first_confidence);
    assert_eq!(engine.edges.get(&id).unwrap().evidence_count, 4);
}

#[test]
fn active_learning_asks_human_for_ambiguous_identity_binding() {
    let candidate = BindingCandidate {
        left_cluster_id: "face:unknown".to_string(),
        right_cluster_id: "voice:travis-or-tim".to_string(),
        relation: BindingRelation::LikelySameEntity,
        evidence: vec![BindingEvidence {
            kind: BindingEvidenceKind::SimultaneousConflict,
            score: 0.8,
            reason: "two person candidates are active".to_string(),
        }],
        confidence: 0.42,
        decision: BindingDecision::AskHuman,
        reason: "identity is ambiguous".to_string(),
    };
    let mut planner = DefaultActiveLearningPlanner::new();
    let questions = planner.plan(&ActiveLearningInput {
        ambiguous_binding_candidates: vec![candidate],
        ..ActiveLearningInput::default()
    });

    assert_eq!(questions.len(), 1);
    assert_eq!(
        questions[0].target_kind,
        ActiveLearningTargetKind::BindingCandidate
    );
    assert!(questions[0]
        .proposed_tests
        .iter()
        .any(|test| test.kind == ActiveLearningActionKind::AskHuman));
    assert_eq!(questions[0].state, ActiveLearningState::WaitingForHuman);
}

#[test]
fn active_learning_motion_test_stays_proposal_when_safe() {
    let candidate = BindingCandidate {
        left_cluster_id: "rgb:patch".to_string(),
        right_cluster_id: "geometry:blob".to_string(),
        relation: BindingRelation::ProjectsTo,
        evidence: vec![BindingEvidence {
            kind: BindingEvidenceKind::ProjectionAgreement,
            score: 0.45,
            reason: "weak reprojection support".to_string(),
        }],
        confidence: 0.5,
        decision: BindingDecision::CollectMoreEvidence,
        reason: "needs viewpoint evidence".to_string(),
    };
    let mut body = BodySense::default();
    body.last_update_ms = 1_000;
    let mut planner = DefaultActiveLearningPlanner::new();
    let questions = planner.plan(&ActiveLearningInput {
        context: ActiveLearningContext {
            t_ms: 1_000,
            body_state: Some(body.clone()),
            movement_readiness: MovementReadiness::from(&body),
            ..ActiveLearningContext::default()
        },
        ambiguous_binding_candidates: vec![candidate],
        ..ActiveLearningInput::default()
    });

    let motion = questions[0]
        .proposed_tests
        .iter()
        .find(|test| test.kind == ActiveLearningActionKind::MoveOrRotate)
        .expect("motion proposal");
    assert!(matches!(motion.action, Some(ActionPrimitive::Turn { .. })));
    assert!(motion.required_safety_state.is_some());
}

#[test]
fn active_learning_uses_diagnostic_when_movement_is_broken() {
    let failure = PredictionFailure {
        id: "failure:no-motion".to_string(),
        target_id: "action:forward".to_string(),
        predicted: "odometry should change".to_string(),
        observed: "pose delta near zero".to_string(),
        confidence: 0.4,
        surprise: 0.9,
        action: Some(ActionPrimitive::Go {
            intensity: 0.2,
            duration_ms: 300,
        }),
        possible_causes: vec!["base disconnected".to_string()],
    };
    let mut planner = DefaultActiveLearningPlanner::new();
    let questions = planner.plan(&ActiveLearningInput {
        context: ActiveLearningContext {
            movement_readiness: MovementReadiness {
                base_connected: false,
                movement_responding: Some(false),
                reason: Some("base connection is down".to_string()),
                ..MovementReadiness::default()
            },
            ..ActiveLearningContext::default()
        },
        prediction_failures: vec![failure],
        ..ActiveLearningInput::default()
    });

    assert!(questions[0].proposed_tests.iter().any(|test| test.kind
        == ActiveLearningActionKind::Diagnostic
        && test.action.is_none()
        && test
            .required_safety_state
            .as_deref()
            .is_some_and(|state| state.contains("base connection"))));
}

#[test]
fn active_learning_replays_memory_for_place_candidate() {
    let candidate = PlaceRecognitionCandidate {
        kind: PlaceRecognitionKind::SamePlace,
        cell: PlaceCellSummary {
            x: 1,
            y: 2,
            center_x_m: 0.5,
            center_y_m: 1.0,
            score: 0.4,
            visit_count: 2,
            last_seen_tick: 10,
            confidence: 0.4,
            last_observed_objects: Vec::new(),
            associated_scene_vectors: Vec::new(),
            associated_face_vectors: Vec::new(),
            associated_object_vectors: Vec::new(),
            associated_voice_vectors: Vec::new(),
            successful_actions: Vec::new(),
            failed_actions: Vec::new(),
        },
        source_vector_id: "scene:old".to_string(),
        source_frame_id: None,
        source_experience_id: None,
        source_instant_frame_id: None,
        source_vector_refs: Vec::new(),
        query_vector_id: Some("scene:now".to_string()),
        query_experience_id: None,
        similarity: 0.55,
        confidence: 0.45,
        reason: "weak scene similarity".to_string(),
    };
    let mut planner = DefaultActiveLearningPlanner::new();
    let questions = planner.plan(&ActiveLearningInput {
        place_candidates: vec![candidate],
        ..ActiveLearningInput::default()
    });

    assert!(questions[0]
        .proposed_tests
        .iter()
        .any(|test| test.kind == ActiveLearningActionKind::ReplayMemory));
}

#[test]
fn association_decays_without_evidence() {
    let mut engine = AssociationLearningEngine::new();
    engine.observe(AssociationObservation {
        t_ms: 1_000,
        active_items: vec![
            association_item("plane:wall", AssociationItemKind::Cluster, 0.9),
            association_item("action:forward-unsafe", AssociationItemKind::Outcome, 0.9),
        ],
        ..AssociationObservation::default()
    });
    let id = association_edge_id(
        "action:forward-unsafe",
        "plane:wall",
        &AssociationRelation::CoOccursWith,
    );
    let before = engine.edges.get(&id).unwrap().confidence;
    engine.decay(10);
    let after = engine.edges.get(&id).unwrap().confidence;

    assert!(after < before);
    assert!(after > 0.0);
}
