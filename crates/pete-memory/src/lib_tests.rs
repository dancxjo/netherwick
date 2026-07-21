
use super::*;
use pete_body::BodySense;
use pete_experience::{
    Experience, ExperienceLatent, Impression, Modality, RecalledExperience, Sensation,
    SensationMetadata, SensationPayload, SensationPayloadKind, SensationSource, VectorEmbedding,
};
use pete_ledger::ExperienceFrame;
use pete_now::{
    EpisodeKind, FaceSense, ObjectClass, ObjectObservation, ObjectObservationSource, PersonId,
    SemanticConceptId, SemanticContext, SemanticDriveId, SemanticPredicate, SemanticRelation,
    SemanticRelationId, SemanticRelationStatus, SurpriseSense, VectorArtifact,
    WorldModelUpdateContext, WorldModelUpdater, FACE_VECTOR_COLLECTION, OBJECT_VECTOR_COLLECTION,
    SCENE_VECTOR_COLLECTION, VOICE_VECTOR_COLLECTION,
};

fn empty_frame(now: Now) -> ExperienceFrame {
    ExperienceFrame {
        id: uuid::Uuid::new_v4(),
        t_ms: now.t_ms,
        now,
        sensations: Vec::new(),
        impressions: Vec::new(),
        experiences: Vec::new(),
        z: None,
        chosen_action: None,
        conscious_command: None,
        reign_input: None,
        reign_outcome: None,
        predicted_futures: Vec::new(),
        behavior_runs: Vec::new(),
        actual_next: None,
        reward: Default::default(),
        surprise: SurpriseSense::default(),
        memory_recall: Vec::new(),
        recollections: Vec::new(),
        llm_teaching: Vec::new(),
        counterfactuals: Vec::new(),
        notes: vec!["saw a familiar person".to_string()],
    }
}

#[tokio::test]
async fn durable_records_preserve_temporal_and_social_world_models() {
    let mut updater = WorldModelUpdater::default();
    let mut present = Now::blank(0, BodySense::default());
    present.body.charging = true;
    present.objects.observations.push(ObjectObservation {
        label: "Alex".to_string(),
        class: ObjectClass::Person,
        bearing_rad: 0.1,
        distance_m: Some(0.8),
        confidence: 0.9,
        source: ObjectObservationSource::Kinect,
    });
    updater.update(present, WorldModelUpdateContext::default());

    let closed = updater.update(
        Now::blank(2_000, BodySense::default()),
        WorldModelUpdateContext::default(),
    );
    let frame = empty_frame(closed);
    let store = InMemoryExperienceStore::new();
    store.store(&frame).await.unwrap();

    assert_eq!(store.completed_episodes(EpisodeKind::Charging).len(), 1);
    let person_id = PersonId("person:alex".to_string());
    assert!(store.last_social_interaction(&person_id).is_some());
    let record = store.snapshot().pop().unwrap();
    assert!(record.social_world.people.contains_key(&person_id));
    assert!(record
        .temporal_context
        .recently_completed
        .iter()
        .any(|episode| episode.kind == EpisodeKind::Conversation));
    assert!(record
        .graph_entities
        .iter()
        .any(|entity| { entity.id == person_id.0 && entity.has_label("SocialPersonBelief") }));
    assert!(record
        .graph_entities
        .iter()
        .any(|entity| entity.has_label("Charging")));
    assert!(record
        .graph_relationships
        .iter()
        .any(|relationship| { relationship.relationship == "HAS_TEMPORAL_EPISODE" }));
    assert!(record
        .graph_entities
        .iter()
        .any(|entity| entity.id == "concept:charger" && entity.has_label("SemanticNode")));
    assert!(record.graph_relationships.iter().any(|relationship| {
        relationship.from == "concept:charger"
            && relationship.to == "drive:energy"
            && relationship.relationship == "SEMANTIC_RESTORES"
            && relationship.payload.get("supporting_evidence").is_some()
    }));
}

#[test]
fn context_distinct_semantic_relations_keep_distinct_persistence_ids() {
    let subject = SemanticNodeRef::Concept(SemanticConceptId("charger".to_string()));
    let object = SemanticNodeRef::Drive(SemanticDriveId("energy".to_string()));
    let mut semantic = SemanticGraphSnapshot::default();

    for route_state in ["clear", "blocked"] {
        let id = SemanticRelationId(format!("semantic:test:restores:{route_state}"));
        semantic.relations.insert(
            id.clone(),
            SemanticRelation {
                id,
                subject: subject.clone(),
                predicate: SemanticPredicate::Restores,
                object: object.clone(),
                context: SemanticContext {
                    agent: Some("pete".to_string()),
                    conditions: BTreeMap::from([(
                        "route_state".to_string(),
                        route_state.to_string(),
                    )]),
                    ..SemanticContext::default()
                },
                confidence: 0.8,
                status: SemanticRelationStatus::ContextLimited,
                ..SemanticRelation::default()
            },
        );
    }

    let mut now = Now::blank(1_000, BodySense::default());
    now.world.semantic = semantic;
    let record = memory_record_from_frame(&empty_frame(now)).expect("memory record");
    let semantic_edges = record
        .graph_relationships
        .iter()
        .filter(|edge| edge.relationship == "SEMANTIC_RESTORES")
        .collect::<Vec<_>>();

    assert_eq!(semantic_edges.len(), 2);
    assert_eq!(
        semantic_edges
            .iter()
            .filter_map(|edge| edge.id.as_deref())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "semantic:test:restores:blocked",
            "semantic:test:restores:clear",
        ])
    );
    assert_eq!(
        semantic_edges
            .iter()
            .filter_map(|edge| {
                edge.payload
                    .pointer("/context/conditions/route_state")
                    .and_then(|value| value.as_str())
            })
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["blocked", "clear"])
    );

    let neo4j_params = neo4j_relationship_params(&record);
    assert!(neo4j_params
        .iter()
        .filter(|param| param["kind"] == "SEMANTIC_RESTORES")
        .all(|param| param["payload_json"]
            .as_str()
            .is_some_and(|payload| payload.contains("\"id\":\"semantic:test:restores:"))));
    let neo4j_edge_ids = neo4j_params
        .into_iter()
        .filter(|param| param["kind"] == "SEMANTIC_RESTORES")
        .filter_map(|param| param["edge_id"].as_str().map(str::to_string))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        neo4j_edge_ids,
        BTreeSet::from([
            "semantic:test:restores:blocked".to_string(),
            "semantic:test:restores:clear".to_string(),
        ])
    );
    assert!(NEO4J_GRAPH_UPSERT_CYPHER.contains("{edge_id: relationship.edge_id}"));
    assert!(NEO4J_LEGACY_RELATED_EDGE_MIGRATION_CYPHER.contains("WHERE legacy.edge_id IS NULL"));
    assert!(NEO4J_LEGACY_RELATED_EDGE_MIGRATION_CYPHER.contains("SET legacy.edge_id"));
    assert!(NEO4J_LEGACY_RELATED_EDGE_MIGRATION_CYPHER.contains("legacy.payload_json"));
    assert!(!NEO4J_LEGACY_RELATED_EDGE_MIGRATION_CYPHER.contains("DELETE"));
}

fn test_cluster(id: &str, kind: DiscoveredClusterKind, modality: Modality) -> DiscoveredCluster {
    DiscoveredCluster::new(id, modality, kind, 1_000, 0.9)
}

fn accepted_binding(left: &str, right: &str, relation: BindingRelation) -> BindingCandidate {
    BindingCandidate {
        left_cluster_id: left.to_string(),
        right_cluster_id: right.to_string(),
        relation,
        evidence: vec![
            BindingEvidence {
                kind: BindingEvidenceKind::TemporalOverlap,
                score: 0.9,
                reason: "test temporal support".to_string(),
            },
            BindingEvidence {
                kind: BindingEvidenceKind::RepeatedCooccurrence,
                score: 0.85,
                reason: "test repeated support".to_string(),
            },
        ],
        confidence: 0.9,
        decision: BindingDecision::Accept,
        reason: "test accepted binding".to_string(),
    }
}

fn person_constellation_observation(t_ms: u64) -> ConstellationObservation {
    ConstellationObservation {
        t_ms,
        clusters: vec![
            test_cluster("face:travis", DiscoveredClusterKind::Face, Modality::Vision),
            test_cluster(
                "voice:travis",
                DiscoveredClusterKind::Voice,
                Modality::Audio,
            ),
            test_cluster(
                "label:travis",
                DiscoveredClusterKind::Label,
                Modality::Language,
            ),
        ],
        accepted_bindings: vec![
            accepted_binding(
                "face:travis",
                "voice:travis",
                BindingRelation::LikelySameEntity,
            ),
            accepted_binding("face:travis", "label:travis", BindingRelation::NamedBy),
        ],
        active_entity_ids: vec!["entity:person:travis".to_string()],
        place_cells: vec![PlaceCellKey { x: 2, y: -1 }],
        prediction_value: 0.65,
        ..ConstellationObservation::default()
    }
}

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

#[tokio::test]
async fn store_preserves_typed_vector_artifacts() {
    let mut now = Now::blank(123, BodySense::default());
    now.face = FaceSense {
        schema_version: 1,
        vectors: vec![
            VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-1", vec![1.0, 0.0])
                .with_source_id("face-crop-1"),
        ],
    };
    now.eye.scene_vectors.push(VectorArtifact::new(
        SCENE_VECTOR_COLLECTION,
        "scene-1",
        vec![0.0, 1.0],
    ));

    let store = InMemoryExperienceStore::new();
    store.store(&empty_frame(now)).await.unwrap();

    let snapshot = store.snapshot();
    assert_eq!(snapshot[0].face_vectors[0].point_id, "face-1");
    assert_eq!(
        snapshot[0].face_vectors[0].source_id.as_deref(),
        Some("face-crop-1")
    );
    assert_eq!(
        snapshot[0].scene_vectors[0].collection,
        SCENE_VECTOR_COLLECTION
    );
    assert!(snapshot[0]
        .graph_entities
        .iter()
        .any(|entity| entity.has_label("Person")));
    assert!(snapshot[0]
        .graph_entities
        .iter()
        .any(|entity| entity.has_label("Place")));
    assert!(snapshot[0]
        .graph_relationships
        .iter()
        .any(|edge| edge.relationship == "HAS_FACE_VECTOR"));
}

#[tokio::test]
async fn recall_returns_graph_context_as_memory_sense() {
    let mut now = Now::blank(123, BodySense::default());
    now.face.vectors = vec![
        VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-1", vec![1.0, 0.0])
            .with_source_id("person:ada"),
    ];

    let store = InMemoryExperienceStore::new();
    store.store(&empty_frame(now)).await.unwrap();

    let recall = store
        .recall(RecallQuery {
            face_vectors: vec![VectorArtifact::new(
                FACE_VECTOR_COLLECTION,
                "query-face",
                vec![1.0, 0.0],
            )],
            battery: 1.0,
            ..RecallQuery::default()
        })
        .await
        .unwrap();

    assert!(recall
        .sense
        .remembered_entities
        .iter()
        .any(|entity| entity.id == "person:ada" && entity.has_label("Person")));
    assert!(recall
        .sense
        .remembered_entities
        .iter()
        .any(|entity| entity.has_label("Place")));
    assert!(recall.sense.graph_context_summary.is_some());
    assert!(recall.hits[0]
        .graph_context
        .iter()
        .any(|entity| entity.has_label("Person")));
}

#[tokio::test]
async fn recall_returns_embodied_memory_recall_sensations_with_lineage() {
    let now = Now::blank(500, BodySense::default());
    let mut frame = empty_frame(now);
    let source_sensation_id = uuid::Uuid::new_v4();
    let experience = Experience::new(
        "embodied.now",
        "I notice a familiar charger alcove.",
        Vec::new(),
        vec![source_sensation_id],
        450,
        500,
    );
    frame.z = Some(ExperienceLatent {
        t_ms: frame.t_ms,
        z: vec![1.0, 0.0, 0.0],
        confidence: 0.9,
        ..ExperienceLatent::default()
    });
    frame.experiences = vec![experience.clone()];
    let frame_id = frame.id;

    let store = InMemoryExperienceStore::new();
    store.store(&frame).await.unwrap();
    let recall = store
        .recall(RecallQuery {
            scene_vectors: vec![VectorArtifact::new(
                EXPERIENCE_VECTOR_COLLECTION,
                "query-experience",
                vec![1.0, 0.0, 0.0],
            )],
            battery: frame.now.body.battery_level,
            ..RecallQuery::default()
        })
        .await
        .unwrap();

    let recollection = recall.recollections.first().expect("recollection");
    assert_eq!(recollection.original_frame_id, Some(frame_id));
    assert!(recollection
        .original_vector_ids
        .iter()
        .any(|id| id.contains(&experience.id.to_string())));
    assert_eq!(recollection.sensation.modality, Modality::Memory);
    assert_eq!(
        recollection.sensation.payload_kind,
        SensationPayloadKind::MemoryRecall
    );
    assert!(matches!(
        recollection.sensation.provenance.kind,
        pete_core::ProvenanceKind::MemoryRecall { experience_id }
            if experience_id == experience.id
    ));
    assert!(recollection
        .sensation
        .impression
        .as_ref()
        .is_some_and(|impression| impression.text.starts_with("I remember")));
}

#[tokio::test]
async fn object_vectors_are_memorized_and_recalled_like_faces() {
    let mut now = now_at(1_000, 0.0, 0.0);
    now.objects.observations.push(ObjectObservation {
        label: "red cup".to_string(),
        class: ObjectClass::Landmark,
        bearing_rad: 0.1,
        distance_m: Some(1.2),
        confidence: 0.9,
        source: ObjectObservationSource::Kinect,
    });
    now.objects.vectors.push(
        VectorArtifact::new(OBJECT_VECTOR_COLLECTION, "object-red-cup", vec![1.0, 0.0])
            .with_model("test.object.embedding")
            .with_source_id("entity:landmark:red-cup"),
    );
    let frame = empty_frame(now);
    let store = InMemoryExperienceStore::new();
    store.store(&frame).await.unwrap();

    let record = store.snapshot().pop().expect("stored record");
    assert_eq!(record.object_vectors.len(), 1);
    assert!(record
        .graph_relationships
        .iter()
        .any(|edge| edge.relationship == "HAS_OBJECT_VECTOR"));

    let recall = store
        .recall(RecallQuery {
            object_vectors: vec![VectorArtifact::new(
                OBJECT_VECTOR_COLLECTION,
                "object-query",
                vec![1.0, 0.0],
            )],
            ..RecallQuery::default()
        })
        .await
        .unwrap();

    assert_eq!(recall.hits.len(), 1);
    assert!(recall.sense.object_familiarity > 0.99);
}

#[test]
fn deterministic_embodied_fixture_exercises_hardware_free_modalities() {
    let now = deterministic_embodied_fixture_now(1_000, 0.0);
    assert!(now
        .eye_frame
        .as_ref()
        .is_some_and(|frame| !frame.bytes.is_empty()));
    assert_eq!(now.range.nearest_m, Some(0.42));
    assert!(!now.kinect.depth_m.is_empty());
    assert_eq!(now.ear.asr.sample_rate_hz, Some(16_000));
    assert!(now.ear.asr.transcript.is_some());
    assert!(now.body.flags.wall);

    let primary = pete_experience::primary_sensations_from_now(&now);
    assert!(primary
        .iter()
        .any(|sensation| sensation.payload_kind == SensationPayloadKind::ImageBytes));
    assert!(primary
        .iter()
        .any(|sensation| sensation.payload_kind == SensationPayloadKind::DepthFrame));
    assert!(primary
        .iter()
        .any(|sensation| sensation.payload_kind == SensationPayloadKind::LidarScan));
    assert!(primary
        .iter()
        .any(|sensation| sensation.payload_kind == SensationPayloadKind::AudioPcm));
    assert!(primary
        .iter()
        .any(|sensation| sensation.payload_kind == SensationPayloadKind::ContactEvent));
}

#[tokio::test]
async fn deterministic_embodied_frame_preserves_lineage_vectors_and_experience_outputs() {
    let frame =
        super::build_embodied_eval_frame(deterministic_embodied_fixture_now(1_000, 0.0), None, &[])
            .await
            .unwrap();

    let visual_parent_ids = frame
        .sensations
        .iter()
        .filter(|sensation| {
            sensation.modality == Modality::Vision
                && sensation.payload_kind == SensationPayloadKind::ImageBytes
        })
        .map(|sensation| sensation.id)
        .collect::<BTreeSet<_>>();
    assert!(frame.sensations.iter().any(|sensation| {
        sensation.modality == Modality::Vision
            && sensation.payload_kind == SensationPayloadKind::Crop
            && sensation
                .parent_id
                .is_some_and(|parent_id| visual_parent_ids.contains(&parent_id))
    }));
    assert!(frame.sensations.iter().any(|sensation| {
        sensation.vector.as_ref().is_some_and(|vector| {
            !vector.model_id.is_empty()
                && vector.dim == vector.vector.len()
                && !vector.purpose.is_empty()
        })
    }));
    assert!(frame
        .impressions
        .iter()
        .any(|impression| impression.sensation_id.is_some()));
    let experience = frame.experiences.last().expect("experience");
    assert!(experience.summary_impression.is_some());
    assert!(!experience.predictions.is_empty());
}

#[tokio::test]
async fn deterministic_replay_produces_identical_instant_shape() {
    let left =
        super::build_embodied_eval_frame(deterministic_embodied_fixture_now(1_000, 0.0), None, &[])
            .await
            .unwrap()
            .experience_instant();
    let right =
        super::build_embodied_eval_frame(deterministic_embodied_fixture_now(1_000, 0.0), None, &[])
            .await
            .unwrap()
            .experience_instant();

    assert_eq!(stable_instant_shape(&left), stable_instant_shape(&right));
}

#[tokio::test]
async fn instant_conversion_preserves_lineage_vectors_predictions_and_memory_links() {
    let mut frame =
        super::build_embodied_eval_frame(deterministic_embodied_fixture_now(1_000, 0.0), None, &[])
            .await
            .unwrap();
    attach_memory_links_to_frame(&mut frame);

    let instant = frame.experience_instant();
    assert!(!instant.lineage.is_empty());
    assert!(instant.teacher_vectors.iter().any(|vector| vector
        .metadata
        .model_id
        .contains("fixture")
        || vector.metadata.model_id.contains("pete")));
    assert!(instant
        .teacher_vectors
        .iter()
        .all(|vector| vector.metadata.dim == vector.vector.len()));
    assert!(!instant.predictions.is_empty());
    assert!(!instant.memory_links.is_empty());

    let context = instant.embodied_context();
    assert_eq!(context.lineage, instant.lineage);
    assert_eq!(context.predictions, instant.predictions);
    assert_eq!(context.memory_links, instant.memory_links);
}

#[tokio::test]
async fn instant_missing_modalities_are_explicit_in_coverage() {
    let frame = super::build_embodied_eval_frame(
        deterministic_embodied_fixture_now(1_000, 0.0),
        None,
        &[EmbodiedEvalOmission::Vectors],
    )
    .await
    .unwrap();
    let instant = frame.experience_instant();
    let coverage = instant.coverage();

    assert!(!instant.missing_modalities.is_empty());
    assert!(!coverage.missing_modalities.is_empty());
    assert_eq!(
        coverage.sensation_count,
        instant.primary_sensations.len() + instant.descendant_sensations.len()
    );
    assert_eq!(coverage.vector_count, instant.teacher_vectors.len());
}

#[tokio::test]
async fn deterministic_embodied_eval_reports_full_coverage_and_recall() {
    let report = deterministic_embodied_eval_report().await.unwrap();

    assert!(report.passed(), "{:?}", report.failures);
    assert_eq!(report.frame_count, 2);
    assert_eq!(report.instant_count, 2);
    assert!(report.instant_teacher_vector_count > 0);
    assert!(report.instant_missing_modality_count > 0);
    assert!(report.primary_sensation_count > 0);
    assert!(report.descendant_sensation_count > 0);
    assert!(report.vectorized_sensation_count > 0);
    assert!(report.impression_count > 0);
    assert!(report.summary_impression_count > 0);
    assert!(report.experience_latent_count > 0);
    assert!(report.prediction_count > 0);
    assert!(report.memory_link_count > 0);
    assert!(report.recall_sensation_count > 0);
    assert!(report.recall_impression_count > 0);
    assert!(report.place_recognition_candidate_count > 0);
    assert!(report.lineage_edge_count > 0);
    assert!(report.input_modalities.contains(&"vision".to_string()));
    assert!(report.input_modalities.contains(&"depth".to_string()));
    assert!(report.input_modalities.contains(&"lidar".to_string()));
    assert!(report.input_modalities.contains(&"audio".to_string()));
    assert_eq!(report.instant_coverage.len(), report.instant_count);
    assert!(report
        .instant_coverage
        .iter()
        .all(|coverage| coverage.sensation_count > 0));
}

#[tokio::test]
async fn deterministic_embodied_eval_reports_deliberately_missing_stages() {
    let report = deterministic_embodied_eval_report_with_omissions(&[
        EmbodiedEvalOmission::Vectors,
        EmbodiedEvalOmission::Recall,
    ])
    .await
    .unwrap();

    assert!(!report.passed());
    assert!(report
        .failures
        .iter()
        .any(|failure| failure == "no vectors"));
    assert!(report.failures.iter().any(|failure| failure == "no recall"));
    assert!(report
        .failures
        .iter()
        .any(|failure| failure == "no place recognition"));
    assert!(report
        .warnings
        .iter()
        .any(|warning| warning == "omitted vectors"));
    assert!(report
        .warnings
        .iter()
        .any(|warning| warning == "omitted recall"));
}

fn stable_instant_shape(instant: &pete_experience::ExperienceInstant) -> serde_json::Value {
    serde_json::json!({
        "schema_version": instant.schema_version,
        "t_ms": instant.t_ms,
        "window_start_ms": instant.window_start_ms,
        "window_end_ms": instant.window_end_ms,
        "primary": instant.primary_sensations.iter().map(|sensation| {
            serde_json::json!({
                "parent": sensation.parent_id.is_some(),
                "modality": sensation.modality.as_str(),
                "payload_kind": sensation.payload_kind.as_str(),
                "kind": sensation.kind,
                "source": sensation.source,
            })
        }).collect::<Vec<_>>(),
        "descendant": instant.descendant_sensations.iter().map(|sensation| {
            serde_json::json!({
                "parent": sensation.parent_id.is_some(),
                "modality": sensation.modality.as_str(),
                "payload_kind": sensation.payload_kind.as_str(),
                "kind": sensation.kind,
                "source": sensation.source,
            })
        }).collect::<Vec<_>>(),
        "vectors": instant.teacher_vectors.iter().map(|vector| {
            serde_json::json!({
                "dim": vector.metadata.dim,
                "model_id": vector.metadata.model_id,
                "purpose": vector.metadata.purpose,
                "collection": vector.metadata.collection,
                "modality": vector.metadata.modality.as_str(),
                "payload_kind": vector.metadata.payload_kind.as_str(),
            })
        }).collect::<Vec<_>>(),
        "coverage": instant.coverage(),
    })
}

#[test]
fn vector_similarity_scores_matching_vectors() {
    let query = vec![vec![1.0, 0.0]];
    let records = vec![VectorArtifact::new(
        FACE_VECTOR_COLLECTION,
        "face-1",
        vec![1.0, 0.0],
    )];
    let score = max_vector_similarity(
        query.iter().map(Vec::as_slice).collect(),
        records.iter().collect(),
    );

    assert!((score - 1.0).abs() < f32::EPSILON);
}

#[test]
fn memory_record_includes_embodied_vectors_and_lineage() {
    let now = Now::blank(123, BodySense::default());
    let mut frame = empty_frame(now);

    let primary = Sensation::primary(
        Modality::Vision,
        SensationSource {
            name: "camera.front".to_string(),
            device_id: Some("cam-1".to_string()),
            frame_id: Some("camera-frame-1".to_string()),
        },
        120,
        123,
        SensationPayload::image_metadata(640, 480, "rgb8", 921_600),
    )
    .with_summary("wide camera frame")
    .with_vector(VectorEmbedding::new(
        vec![1.0, 0.0, 0.0],
        "vision.sensation.test",
        Modality::Vision,
        SensationPayloadKind::ImageBytes,
        uuid::Uuid::nil(),
        124,
    ));
    let primary_id = primary.id;
    let mut primary = primary;
    primary.vector.as_mut().unwrap().source_sensation_id = primary_id;

    let crop_impression = Impression::new(
        "focus",
        "I focus on a bright crop.",
        vec![primary_id],
        120,
        123,
    )
    .with_confidence(0.82);
    let mut crop = Sensation::descendant(
        &primary,
        "vision.crop",
        SensationPayloadKind::Crop,
        serde_json::json!({"bbox": [10, 20, 64, 64]}),
        SensationMetadata {
            confidence: Some(0.91),
            labels: vec!["bright".to_string()],
            ..SensationMetadata::default()
        },
        "cropper",
    )
    .with_summary("bright crop")
    .with_vector(VectorEmbedding::new(
        vec![0.0, 1.0, 0.0],
        "vision.crop.test",
        Modality::Vision,
        SensationPayloadKind::Crop,
        uuid::Uuid::nil(),
        125,
    ))
    .with_impression(crop_impression.clone());
    crop.vector.as_mut().unwrap().source_sensation_id = crop.id;

    let experience_impression = Impression::new(
        "summary",
        "I see and focus on something bright.",
        vec![primary_id, crop.id],
        120,
        126,
    )
    .with_confidence(0.9);
    let mut experience = Experience::new(
        "visual_focus",
        "I see and focus on something bright.",
        vec![crop_impression.id, experience_impression.id],
        vec![primary_id, crop.id],
        120,
        126,
    );
    experience.summary_impression =
        Some(experience_impression.clone().for_experience(experience.id));

    frame.sensations = vec![primary.clone(), crop.clone()];
    frame.impressions = vec![crop_impression.clone(), experience_impression.clone()];
    frame.experiences = vec![experience.clone()];

    let record = memory_record_from_frame(&frame).unwrap();

    assert_eq!(record.sensation_vectors.len(), 2);
    assert_eq!(record.experience_vectors.len(), 1);
    assert!(record
        .sensation_vectors
        .iter()
        .all(|artifact| artifact.collection == SENSATION_VECTOR_COLLECTION));
    assert_eq!(
        record.experience_vectors[0].collection,
        EXPERIENCE_VECTOR_COLLECTION
    );

    let crop_id = crop.id.to_string();
    let crop_artifact = record
        .sensation_vectors
        .iter()
        .find(|artifact| artifact.source_id.as_deref() == Some(crop_id.as_str()))
        .unwrap();
    let crop_payload = record
        .vector_payloads
        .get(&vector_payload_key(crop_artifact))
        .unwrap();
    assert_eq!(
        crop_payload["parent_sensation_id"],
        serde_json::json!(primary_id.to_string())
    );
    assert_eq!(crop_payload["payload_kind"], "crop");
    assert_eq!(crop_payload["model_id"], "vision.crop.test");
    assert_eq!(crop_payload["dim"], 3);

    let exp_payload = record
        .vector_payloads
        .get(&vector_payload_key(&record.experience_vectors[0]))
        .unwrap();
    assert_eq!(
        exp_payload["summary_impression_text"],
        "I see and focus on something bright."
    );
    assert_eq!(exp_payload["experience_id"], experience.id.to_string());

    assert!(record.graph_relationships.iter().any(|edge| {
        edge.from == format!("frame:{}", frame.id)
            && edge.to == experience_node_id(experience.id)
            && edge.relationship == "HAS_EXPERIENCE"
    }));
    assert!(record.graph_relationships.iter().any(|edge| {
        edge.from == experience_node_id(experience.id)
            && edge.to == sensation_node_id(crop.id)
            && edge.relationship == "INTEGRATES_SENSATION"
    }));
    assert!(record.graph_relationships.iter().any(|edge| {
        edge.from == experience_node_id(experience.id)
            && edge.to == impression_node_id(experience_impression.id)
            && edge.relationship == "INTEGRATES_IMPRESSION"
    }));
    assert!(record.graph_relationships.iter().any(|edge| {
        edge.from == sensation_node_id(crop.id)
            && edge.to == sensation_node_id(primary_id)
            && edge.relationship == "DERIVED_FROM_SENSATION"
    }));
    assert!(record.graph_relationships.iter().any(|edge| {
        edge.from == sensation_node_id(crop.id)
            && edge.to == impression_node_id(crop_impression.id)
            && edge.relationship == "HAS_IMPRESSION"
    }));
    assert!(record.graph_relationships.iter().any(|edge| {
        edge.from == experience_node_id(experience.id) && edge.relationship == "HAS_FUSED_VECTOR"
    }));
}

#[test]
fn memory_record_links_experience_to_place_objects_people_surfaces_and_recalls() {
    let mut now = now_at(1_000, 1.25, -0.25);
    now.memory.places_visited = 3;
    now.memory.place_familiarity = 0.62;
    now.objects.observations.push(ObjectObservation {
        label: "Charging Dock".to_string(),
        class: ObjectClass::Charger,
        bearing_rad: 0.2,
        distance_m: Some(1.4),
        confidence: 0.83,
        source: ObjectObservationSource::Sim,
    });
    now.face.vectors.push(
        VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-link-1", vec![1.0, 0.0])
            .with_source_id("person:ada"),
    );
    now.voice.vectors.push(
        VectorArtifact::new(VOICE_VECTOR_COLLECTION, "voice-link-1", vec![0.0, 1.0])
            .with_source_id("person:ada"),
    );
    now.extensions.insert(
        "surface.scene_graph".to_string(),
        json!({
            "floor": {"id": "floor", "kind": "floor", "confidence": 0.91},
            "surfaces": [{"id": "wall-east", "kind": "vertical_plane", "confidence": 0.77}],
            "clusters": [{"id": "box-1", "confidence": 0.66}],
        }),
    );

    let mut frame = empty_frame(now);
    let source_sensation_id = uuid::Uuid::new_v4();
    let experience = Experience::new(
        "embodied.now",
        "I see the charging dock near the wall.",
        Vec::new(),
        vec![source_sensation_id],
        950,
        1_000,
    );
    let recalled = Experience::new(
        "embodied.now",
        "I previously found the dock here.",
        Vec::new(),
        Vec::new(),
        500,
        550,
    );
    let recalled_id = recalled.id;
    frame.recollections.push(RecalledExperience {
        experience: recalled,
        score: 0.72,
        original_frame_id: Some(uuid::Uuid::new_v4()),
        original_vector_ids: vec!["experiences:prior".to_string()],
        sensation: Sensation::primary(
            Modality::Memory,
            SensationSource::default(),
            500,
            1_000,
            SensationPayload::structured(json!({})),
        ),
    });
    frame.experiences = vec![experience.clone()];

    let record = memory_record_from_frame(&frame).unwrap();
    let stored_experience = record.experience.as_ref().expect("stored experience");
    let links = &stored_experience.memory_links;

    assert!(links.iter().any(|link| {
        link.relation == "occurred_at_place"
            && link.target_id == place_id_for_pose(frame.now.body.odometry)
            && link.score >= 0.62
    }));
    assert!(links.iter().any(|link| {
        link.relation == "observed_object"
            && link.target_id == "object:sim:charger:charging-dock"
            && (link.score - 0.83).abs() < f32::EPSILON
    }));
    assert!(links
        .iter()
        .any(|link| link.relation == "saw_face" && link.target_id == "person:ada"));
    assert!(links
        .iter()
        .any(|link| link.relation == "heard_voice" && link.target_id == "person:ada"));
    assert!(links
        .iter()
        .any(|link| { link.relation == "near_surface" && link.target_id == "surface:wall-east" }));
    assert!(links.iter().any(|link| {
        link.relation == "similar_to_experience"
            && link.target_id == experience_node_id(recalled_id)
            && (link.score - 0.72).abs() < f32::EPSILON
    }));

    let graph_experience_id = experience_node_id(experience.id);
    let object_edge = record
        .graph_relationships
        .iter()
        .find(|edge| {
            edge.from == graph_experience_id
                && edge.to == "object:sim:charger:charging-dock"
                && edge.relationship == "observed_object"
        })
        .expect("object memory link edge");
    assert!((object_edge.score - 0.83).abs() < f32::EPSILON);
    assert_eq!(object_edge.payload["class"], "charger");
    assert!(record.graph_entities.iter().any(|entity| {
        entity.id == "object:sim:charger:charging-dock" && entity.has_label("Object")
    }));
}

#[test]
fn qdrant_point_ids_are_stable() {
    assert_eq!(
        stable_qdrant_point_id("faces", "frame:face:0"),
        stable_qdrant_point_id("faces", "frame:face:0")
    );
    assert_ne!(
        stable_qdrant_point_id("faces", "frame:face:0"),
        stable_qdrant_point_id("voices", "frame:face:0")
    );
}

#[tokio::test]
async fn qdrant_vector_store_upserts_face_vectors_into_faces_collection() {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test qdrant");
    let addr = listener.local_addr().expect("test qdrant addr");
    let (tx, rx) = mpsc::channel();
    let server = thread::spawn(move || {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().expect("accept qdrant request");
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            loop {
                let read = stream.read(&mut buffer).expect("read qdrant request");
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let request_text = String::from_utf8_lossy(&request).to_string();
            tx.send(request_text).expect("send qdrant request");
            let body = r#"{"result":true,"status":"ok"}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                body.len(),
                body
            )
            .expect("write qdrant response");
        }
    });

    let store = QdrantVectorStore::new(QdrantConfig {
        url: format!("http://{addr}"),
    });
    let record = MemoryRecord {
        frame_id: uuid::Uuid::new_v4(),
        t_ms: 100,
        summary: "face frame".to_string(),
        graph_entities: Vec::new(),
        graph_relationships: Vec::new(),
        scene_vectors: Vec::new(),
        face_vectors: vec![VectorArtifact::new(
            FACE_VECTOR_COLLECTION,
            "face-qdrant",
            vec![0.1, 0.2],
        )
        .with_model("test.face.detector")
        .with_source_frame_id("eye-frame")],
        object_vectors: Vec::new(),
        voice_vectors: Vec::new(),
        sensation_vectors: Vec::new(),
        experience_vectors: Vec::new(),
        vector_payloads: BTreeMap::new(),
        battery: 1.0,
        active_goal: None,
        chosen_action: None,
        warning: None,
        experience: None,
        temporal_context: TemporalContext::default(),
        social_world: SocialWorldSnapshot::default(),
        epistemic_state: EpistemicSnapshot::default(),
    };

    store.upsert_vectors(&record).await.expect("upsert vectors");

    let create = rx.recv().expect("collection create request");
    let upsert = rx.recv().expect("point upsert request");
    server.join().expect("qdrant mock server");
    assert!(create.starts_with("PUT /collections/faces "));
    assert!(upsert.starts_with("PUT /collections/faces/points?wait=true "));
}

#[test]
fn neo4j_http_url_can_be_derived_from_bolt_uri() {
    assert_eq!(
        neo4j_http_url_from_uri("bolt://neo4j:7687"),
        Some("http://neo4j:7474".to_string())
    );
    assert_eq!(
        neo4j_http_url_from_uri("neo4j://localhost:7687"),
        Some("http://localhost:7474".to_string())
    );
    assert_eq!(neo4j_http_url_from_uri("http://localhost:7474"), None);
}

#[test]
fn neo4j_relationship_params_serialize_nested_payloads() {
    let record = MemoryRecord {
        frame_id: uuid::Uuid::new_v4(),
        t_ms: 1_000,
        summary: "place memory".to_string(),
        graph_entities: Vec::new(),
        graph_relationships: vec![GraphEdge {
            id: None,
            from: "embodied_experience:test".to_string(),
            to: "place:grid:0:0".to_string(),
            relationship: "occurred_at_place".to_string(),
            summary: Some("place near x=0.0m y=0.0m".to_string()),
            score: 1.0,
            payload: json!({
                "target_kind": "place",
                "text": "place near x=0.0m y=0.0m",
                "x_m": 0.0,
                "y_m": 0.0,
                "heading_rad": 0.0,
            }),
        }],
        scene_vectors: Vec::new(),
        face_vectors: Vec::new(),
        object_vectors: Vec::new(),
        voice_vectors: Vec::new(),
        sensation_vectors: Vec::new(),
        experience_vectors: Vec::new(),
        vector_payloads: BTreeMap::new(),
        battery: 1.0,
        active_goal: None,
        chosen_action: None,
        warning: None,
        experience: None,
        temporal_context: TemporalContext::default(),
        social_world: SocialWorldSnapshot::default(),
        epistemic_state: EpistemicSnapshot::default(),
    };

    let params = neo4j_relationship_params(&record);
    let payload_json = params[0]["payload_json"]
        .as_str()
        .expect("payload serialized as string");
    let payload: serde_json::Value =
        serde_json::from_str(payload_json).expect("payload_json is valid json");

    assert!(params[0].get("payload").is_none());
    assert!(params[0]["edge_id"]
        .as_str()
        .is_some_and(|id| id.starts_with("graph-edge:")));
    assert_eq!(payload["target_kind"], "place");
    assert_eq!(payload["heading_rad"], 0.0);
}

#[test]
fn neo4j_intelligence_params_preserve_uncertainty_and_reviews() {
    let feature = Feature::new(
        pete_core::FeatureType::FaceObservation,
        pete_core::FeatureModality::Vision,
        1_000,
        0.74,
        pete_core::Provenance::direct(),
    )
    .with_source_frame("frame-a");
    let feature_id = feature.id;
    let candidate = BindingCandidate {
        left_cluster_id: "cluster:face:a".to_string(),
        right_cluster_id: "cluster:voice:b".to_string(),
        relation: BindingRelation::LikelySameEntity,
        evidence: vec![
            BindingEvidence {
                kind: BindingEvidenceKind::VectorSimilarity,
                score: 0.68,
                reason: "face and voice recur close together".to_string(),
            },
            BindingEvidence {
                kind: BindingEvidenceKind::Contradiction,
                score: 0.82,
                reason: "same identity appears in incompatible places".to_string(),
            },
        ],
        confidence: 0.41,
        decision: BindingDecision::HoldAmbiguous,
        reason: "cross-modal identity needs review".to_string(),
    };
    let document = GraphIntelligenceDocument {
        id: "graph-doc:test".to_string(),
        t_ms: 1_000,
        frame_id: Some("frame-a".to_string()),
        provenance: "unit-test".to_string(),
        confidence: 0.6,
        reason: "testing typed graph persistence".to_string(),
        source_frame_ids: vec!["frame-a".to_string()],
        features: vec![feature],
        clusters: vec![
            DiscoveredCluster::new(
                "cluster:face:a",
                Modality::Vision,
                DiscoveredClusterKind::Face,
                1_000,
                0.7,
            )
            .with_feature_ids(vec![feature_id]),
            DiscoveredCluster::new(
                "cluster:voice:b",
                Modality::Audio,
                DiscoveredClusterKind::Voice,
                1_000,
                0.65,
            ),
        ],
        binding_candidates: vec![candidate],
        constellations: vec![Constellation {
            id: "constellation:person:maybe".to_string(),
            kind_hint: Some("person".to_string()),
            member_cluster_ids: vec!["cluster:face:a".to_string(), "cluster:voice:b".to_string()],
            member_binding_ids: vec![
                "binding-edge:cluster-face-a:likely_same_entity:cluster-voice-b".to_string(),
            ],
            supporting_feature_ids: vec![feature_id],
            supporting_entity_ids: Vec::new(),
            supporting_place_cells: Vec::new(),
            confidence: 0.48,
            stability: 0.2,
            prediction_value: 0.1,
            first_seen_ms: 1_000,
            last_seen_ms: 1_000,
            evidence_count: 2,
            state: ConstellationState::Ambiguous,
            notes: vec!["voice may belong to a different person".to_string()],
        }],
        llm_reviews: vec![LlmReviewRecord {
            id: "llm-review:1".to_string(),
            target_id: "binding-candidate:cluster-face-a:likely_same_entity:cluster-voice-b"
                .to_string(),
            target_kind: ActiveLearningTargetKind::BindingCandidate,
            confidence: 0.55,
            t_ms: 1_010,
            critique: "ask whether the face and voice are the same person".to_string(),
            contradictions: vec!["place mismatch".to_string()],
            suggested_questions: vec!["Is this the same person?".to_string()],
        }],
        human_reviews: vec![HumanReviewRecord {
            id: "human-review:1".to_string(),
            target_id: "hypothesis:face_identity:track-a:pete".to_string(),
            target_kind: ActiveLearningTargetKind::TrackingHypothesis,
            confidence: 0.9,
            t_ms: 1_020,
            confirmation: "not the same person".to_string(),
            reviewer: Some("tester".to_string()),
        }],
        review_records: vec![GraphReviewRecord {
            id: "graph-review:1".to_string(),
            target_id: "constellation:person:maybe".to_string(),
            review_kind: "mutually_exclusive_label".to_string(),
            severity: 0.8,
            confidence: 0.7,
            t_ms: 1_030,
            reason: "constellation has conflicting person labels".to_string(),
            evidence_ids: vec!["evidence:place-mismatch".to_string()],
            state: "open".to_string(),
        }],
        ..GraphIntelligenceDocument::default()
    };

    let params = neo4j_intelligence_params(&document);
    assert_eq!(params["document"]["source_frame_ids"][0], "frame-a");
    assert_eq!(
        params["binding_candidates"][0]["decision"],
        "hold_ambiguous"
    );
    assert_eq!(params["candidate_evidence"].as_array().unwrap().len(), 2);
    assert_eq!(
        params["candidate_evidence"][1]["current_state"],
        "contradictory"
    );
    assert_eq!(params["constellations"][0]["current_state"], "ambiguous");
    assert_eq!(params["llm_reviews"][0]["current_state"], "needs_review");
    assert_eq!(params["human_reviews"][0]["current_state"], "confirmed");
    assert_eq!(
        params["review_records"][0]["state"],
        serde_json::Value::Null
    );
    assert_eq!(params["review_records"][0]["current_state"], "open");
    assert!(neo4j_intelligence_upsert_statements()
        .iter()
        .any(|statement| statement.contains("COMPETES_WITH")));
    assert!(neo4j_intelligence_upsert_statements()
        .iter()
        .any(|statement| statement.contains("FAILED_WITH")));
}

fn now_at(t_ms: u64, x_m: f32, y_m: f32) -> Now {
    let mut body = BodySense::default();
    body.odometry.x_m = x_m;
    body.odometry.y_m = y_m;
    Now::blank(t_ms, body)
}

fn add_sim_world_extension(now: &mut Now, charger_near: f32, charger_visible: f32) {
    now.extensions.insert(
        "sim.world".to_string(),
        serde_json::json!({
            "schema_version": 1,
            "values": [8.0, 8.0, 1.0, charger_near, charger_visible],
        }),
    );
}

#[test]
fn place_quantization_is_stable_inside_cell() {
    let memory = PlaceMemory::new();

    assert_eq!(memory.quantize(1.01, 2.01), memory.quantize(1.49, 2.49));
    assert_ne!(memory.quantize(1.49, 2.49), memory.quantize(1.51, 2.49));
}

#[test]
fn danger_update_increases_danger_score() {
    let mut memory = PlaceMemory::new();
    let mut now = now_at(100, 1.0, 1.0);
    now.body.flags.bump_left = true;

    let features = memory.observe_now(&now);

    assert!(features.current_place_danger >= 0.9);
    assert_eq!(features.places_visited, 1);
}

#[test]
fn charge_update_increases_charge_score() {
    let mut memory = PlaceMemory::new();
    let mut now = now_at(100, 1.0, 1.0);
    add_sim_world_extension(&mut now, 0.8, 0.2);

    let features = memory.observe_now(&now);

    assert!(features.current_place_charge >= 0.8);
}

#[test]
fn social_update_increases_social_score() {
    let mut memory = PlaceMemory::new();
    let mut now = now_at(100, 1.0, 1.0);
    now.face.vectors.push(VectorArtifact::new(
        FACE_VECTOR_COLLECTION,
        "face-social",
        vec![1.0, 0.0],
    ));

    let features = memory.observe_now(&now);

    assert!(features.current_place_social >= 1.0);
}

#[test]
fn scores_decay_between_observations() {
    let mut memory = PlaceMemory::new();
    let mut now = now_at(100, 1.0, 1.0);
    now.body.flags.bump_left = true;
    let first = memory.observe_now(&now);

    let second = memory.observe_now(&now_at(10_100, 1.0, 1.0));

    assert!(second.current_place_danger < first.current_place_danger);
}

#[test]
fn recall_returns_nearby_charger_direction() {
    let mut memory = PlaceMemory::new();
    let mut charge_now = now_at(100, 1.0, 0.0);
    add_sim_world_extension(&mut charge_now, 1.0, 0.0);
    memory.observe_now(&charge_now);

    let features = memory.features_at(0.0, 0.0);

    let direction = features.nearby_best_charge_direction_rad.unwrap();
    assert!(direction.abs() < 0.4);
}

#[test]
fn recall_returns_safe_frontier_direction_and_confidence() {
    let mut memory = PlaceMemory::new();
    memory.observe_now(&now_at(100, 0.0, 0.0));
    memory.observe_now(&now_at(200, 0.0, 1.0));

    let features = memory.features_at(0.0, 0.0);

    let direction = features.nearby_frontier_direction_rad.unwrap();
    assert!(direction > 0.5);
    assert!(features.current_place_confidence > 0.0);
}

#[test]
fn semantic_cells_keep_vector_and_action_associations() {
    let mut memory = PlaceMemory::new();
    let mut now = now_at(100, 1.0, 1.0);
    now.eye.scene_vectors.push(VectorArtifact::new(
        SCENE_VECTOR_COLLECTION,
        "scene-place-1",
        vec![0.0, 1.0],
    ));
    now.face.vectors.push(VectorArtifact::new(
        FACE_VECTOR_COLLECTION,
        "face-place-1",
        vec![1.0, 0.0],
    ));
    let mut frame = empty_frame(now);
    frame.chosen_action = Some(ActionPrimitive::Inspect {
        target: pete_actions::InspectTarget::Charger,
    });
    frame.reward.value = 0.4;

    memory.observe_frame(&frame);
    let cell = memory
        .cells
        .values()
        .next()
        .expect("semantic cell should be created");

    assert_eq!(cell.occupancy_cell, Some(cell.key));
    assert_eq!(cell.associated_scene_vectors, vec!["scene-place-1"]);
    assert_eq!(cell.associated_face_vectors, vec!["face-place-1"]);
    assert_eq!(cell.successful_actions.len(), 1);
    assert_eq!(cell.successful_actions[0].count, 1);
}

#[test]
fn semantic_overlay_exposes_scores_for_dashboard() {
    let mut memory = PlaceMemory::new();
    let mut danger_now = now_at(100, 1.0, 1.0);
    danger_now.body.flags.bump_left = true;
    memory.observe_now(&danger_now);
    let mut charge_now = now_at(200, 1.5, 1.0);
    add_sim_world_extension(&mut charge_now, 1.0, 0.0);
    memory.observe_now(&charge_now);

    let overlay = memory.semantic_overlay_at(1.0, 1.0);

    assert_eq!(overlay.schema_version, 1);
    assert!(overlay.current.is_some());
    assert!(!overlay.danger_cells.is_empty());
    assert!(!overlay.charge_cells.is_empty());
}

#[test]
fn place_recognition_scores_revisits_above_unrelated_locations() {
    let mut memory = PlaceMemory::new();
    let first_frame_id = uuid::Uuid::new_v4();
    let unrelated_frame_id = uuid::Uuid::new_v4();
    let mut first = now_at(100, 1.0, 1.0);
    first.eye.scene_vectors.push(
        VectorArtifact::new(SCENE_VECTOR_COLLECTION, "scene-first", vec![1.0, 0.0, 0.0])
            .with_source_frame_id(first_frame_id.to_string()),
    );
    let mut unrelated = now_at(200, 4.0, 1.0);
    unrelated.eye.scene_vectors.push(
        VectorArtifact::new(
            SCENE_VECTOR_COLLECTION,
            "scene-unrelated",
            vec![0.0, 1.0, 0.0],
        )
        .with_source_frame_id(unrelated_frame_id.to_string()),
    );
    memory.observe_now(&first);
    memory.observe_now(&first);
    memory.observe_now(&unrelated);

    let query = VectorArtifact::new(
        SCENE_VECTOR_COLLECTION,
        "scene-query",
        vec![0.98, 0.02, 0.0],
    );
    let candidates = memory.recognize_places(
        Some(memory.quantize(1.02, 1.02)),
        &[query],
        PLACE_RECOGNITION_MIN_CONFIDENCE,
        5,
    );

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].source_vector_id, "scene-first");
    assert!(matches!(
        candidates[0].kind,
        PlaceRecognitionKind::SamePlace
    ));
    assert!(candidates[0].similarity > 0.99);
    assert!(candidates[0].confidence >= PLACE_RECOGNITION_MIN_CONFIDENCE);
    assert_eq!(
        candidates[0].source_frame_id.as_deref(),
        Some(first_frame_id.to_string().as_str())
    );
}

#[test]
fn observe_now_stamps_scene_vectors_with_frame_id_extension() {
    let mut memory = PlaceMemory::new();
    let mut observed = now_at(100, 1.0, 1.0);
    observed.extensions.insert(
        "frame_id".to_string(),
        serde_json::Value::String("live-frame-1".to_string()),
    );
    observed.eye.scene_vectors.push(VectorArtifact::new(
        SCENE_VECTOR_COLLECTION,
        "scene-observed",
        vec![1.0, 0.0, 0.0],
    ));
    memory.observe_now(&observed);

    let query = VectorArtifact::new(SCENE_VECTOR_COLLECTION, "scene-query", vec![1.0, 0.0, 0.0]);
    let candidates = memory.recognize_places(
        Some(memory.quantize(1.0, 1.0)),
        &[query],
        PLACE_RECOGNITION_MIN_CONFIDENCE,
        5,
    );

    assert_eq!(candidates.len(), 1);
    assert_eq!(
        candidates[0].source_frame_id.as_deref(),
        Some("live-frame-1")
    );
    assert_eq!(
        candidates[0].source_instant_frame_id.as_deref(),
        Some("live-frame-1")
    );
}

#[test]
fn place_recognition_rejects_low_confidence_candidates() {
    let mut memory = PlaceMemory::new();
    let mut observed = now_at(100, 2.0, 1.0);
    observed.eye.scene_vectors.push(VectorArtifact::new(
        SCENE_VECTOR_COLLECTION,
        "scene-observed",
        vec![1.0, 0.0, 0.0],
    ));
    memory.observe_now(&observed);

    let query = VectorArtifact::new(SCENE_VECTOR_COLLECTION, "scene-query", vec![0.0, 1.0, 0.0]);
    let candidates = memory.recognize_places(
        Some(memory.quantize(2.0, 1.0)),
        &[query],
        PLACE_RECOGNITION_MIN_CONFIDENCE,
        5,
    );

    assert!(candidates.is_empty());
}

#[tokio::test]
async fn recall_and_semantic_overlay_include_place_candidates() {
    let store = InMemoryExperienceStore::new();
    let mut frame = empty_frame(now_at(100, 1.0, 1.0));
    frame.now.eye.scene_vectors.push(
        VectorArtifact::new(SCENE_VECTOR_COLLECTION, "scene-stored", vec![1.0, 0.0])
            .with_source_frame_id(frame.id.to_string()),
    );
    store.observe_frame(&frame).await.unwrap();

    let recall = store
        .recall(RecallQuery {
            pose: Some(frame.now.body.odometry),
            scene_vectors: vec![VectorArtifact::new(
                SCENE_VECTOR_COLLECTION,
                "scene-query",
                vec![1.0, 0.0],
            )],
            battery: frame.now.body.battery_level,
            ..RecallQuery::default()
        })
        .await
        .unwrap();

    assert_eq!(recall.place_recognition_candidates.len(), 1);
    let candidate = &recall.place_recognition_candidates[0];
    assert_eq!(candidate.source_vector_id, "scene-stored");
    assert_eq!(candidate.query_vector_id.as_deref(), Some("scene-query"));
    assert!(candidate.confidence >= PLACE_RECOGNITION_MIN_CONFIDENCE);
    assert!(recall
        .semantic_map
        .as_ref()
        .is_some_and(|overlay| !overlay.place_recognition_candidates.is_empty()));
}

#[tokio::test]
async fn recall_loop_closure_candidates_use_place_memory_evidence() {
    let store = InMemoryExperienceStore::new();
    let mut observed = now_at(100, 1.0, 1.0);
    observed.eye.scene_vectors.push(
        VectorArtifact::new(SCENE_VECTOR_COLLECTION, "scene-stored", vec![1.0, 0.0])
            .with_source_frame_id("stored-frame"),
    );
    store.observe_now(&observed).await.unwrap();

    let candidates = store
        .loop_closure_candidates(
            &RecallQuery {
                pose: Some(now_at(200, 1.1, 1.0).body.odometry),
                scene_vectors: vec![VectorArtifact::new(
                    SCENE_VECTOR_COLLECTION,
                    "scene-query",
                    vec![1.0, 0.0],
                )],
                ..RecallQuery::default()
            },
            PLACE_RECOGNITION_MIN_CONFIDENCE,
            5,
        )
        .await
        .unwrap();

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].source_vector_id, "scene-stored");
    assert_eq!(
        candidates[0].source_frame_id.as_deref(),
        Some("stored-frame")
    );
}

#[tokio::test]
async fn place_recognition_uses_experience_latents_and_lineage() {
    let store = InMemoryExperienceStore::new();
    let mut frame = empty_frame(now_at(100, 1.0, 1.0));
    frame.now.range.beams = vec![0.4, 0.7, 1.0];
    frame.now.range.nearest_m = Some(0.4);
    frame.now.objects.observations.push(ObjectObservation {
        label: "charger alcove".to_string(),
        class: ObjectClass::Landmark,
        confidence: 0.9,
        source: ObjectObservationSource::Sim,
        ..ObjectObservation::default()
    });
    let sensation_id = uuid::Uuid::new_v4();
    let experience = Experience::new(
        "embodied.place",
        "charger alcove near the wall",
        Vec::new(),
        vec![sensation_id],
        80,
        100,
    );
    frame.z = Some(ExperienceLatent {
        t_ms: frame.t_ms,
        z: vec![1.0, 0.0, 0.0, 0.0],
        confidence: 0.95,
        ..ExperienceLatent::default()
    });
    let experience_id = experience.id.to_string();
    frame.experiences.push(experience);
    store.observe_frame(&frame).await.unwrap();

    let query_now = now_at(140, 1.02, 1.01);
    let latent = ExperienceLatent {
        t_ms: 140,
        z: vec![0.99, 0.01, 0.0, 0.0],
        confidence: 0.95,
        ..ExperienceLatent::default()
    };
    let mut query = RecallQuery::from_now(&query_now);
    query.place_recognition_input = Some(place_recognition_input_from_query_now(
        &query_now,
        Some(&latent),
        "test-query",
    ));
    let recall = store.recall(query).await.unwrap();

    let candidate = recall
        .place_recognition_candidates
        .first()
        .expect("place candidate from learned latent");
    assert_eq!(
        candidate.source_experience_id.as_deref(),
        Some(experience_id.as_str())
    );
    assert_eq!(
        candidate.source_instant_frame_id.as_deref(),
        Some(frame.id.to_string().as_str())
    );
    assert!(candidate.source_vector_id.contains(":experience-latent"));
    assert!(candidate.reason.contains("confidence="));
}

#[test]
fn place_recognition_report_marks_not_enough_evidence_and_rejections() {
    let mut memory = PlaceMemory::new();
    let mut observed = now_at(100, 2.0, 1.0);
    observed.eye.scene_vectors.push(VectorArtifact::new(
        SCENE_VECTOR_COLLECTION,
        "scene-observed",
        vec![1.0, 0.0, 0.0],
    ));
    memory.observe_now(&observed);

    let empty = memory.recognize_places_report(
        Some(memory.quantize(2.0, 1.0)),
        &[],
        PLACE_RECOGNITION_MIN_CONFIDENCE,
        5,
    );
    assert!(empty.not_enough_evidence.is_some());

    let low = memory.recognize_places_report(
        Some(memory.quantize(2.0, 1.0)),
        &[VectorArtifact::new(
            SCENE_VECTOR_COLLECTION,
            "scene-query",
            vec![0.0, 1.0, 0.0],
        )],
        PLACE_RECOGNITION_MIN_CONFIDENCE,
        5,
    );
    assert!(low.candidates.is_empty());
    assert_eq!(low.rejected.len(), 1);
    assert!(low.rejected[0].reason.contains("below threshold"));
}

#[test]
fn ledger_replay_reconstructs_semantic_map_cells() {
    let mut first = empty_frame(now_at(100, 1.0, 1.0));
    first.now.body.flags.bump_left = true;
    first.chosen_action = Some(ActionPrimitive::Go {
        intensity: 0.3,
        duration_ms: 200,
    });
    first.reward.value = -0.6;
    let mut second = empty_frame(now_at(200, 1.0, 1.0));
    second.chosen_action = first.chosen_action.clone();
    second.reward.value = 0.3;

    let memory = place_memory_from_frames(&[first, second]);
    let report = memory.report();
    let cell = memory
        .cells
        .values()
        .next()
        .expect("replayed semantic cell");

    assert_eq!(report.places_visited, 1);
    assert!(cell.danger_score > 0.0);
    assert_eq!(cell.failed_actions.len(), 1);
    assert_eq!(cell.successful_actions.len(), 1);
    assert!(cell.novelty_score < 1.0);
}

#[test]
fn compact_memory_features_serialize() {
    let features = PlaceMemoryFeatures {
        current_place_danger: 0.2,
        current_place_charge: 0.7,
        current_place_social: 0.3,
        current_place_novelty: 0.4,
        current_place_familiarity: 0.5,
        current_place_confidence: 0.6,
        nearby_best_charge_direction_rad: Some(1.0),
        nearby_best_safe_direction_rad: None,
        nearby_frontier_direction_rad: Some(-0.5),
        places_visited: 3,
    };

    let json = serde_json::to_value(&features).unwrap();

    let charge = json["current_place_charge"].as_f64().unwrap();
    assert!((charge - 0.7).abs() < 0.000_001);
    assert_eq!(json["places_visited"], 3);
}

#[test]
fn entity_constellation_candidates_are_generated_and_gated_by_overlap() {
    let mut memory = PlaceMemory::new();

    // Observe a place with several entities at (2.0, 1.0)
    let mut first = now_at(100, 2.0, 1.0);
    first.objects.observations.push(ObjectObservation {
        label: "chair".to_string(),
        class: ObjectClass::Unknown,
        bearing_rad: 0.1,
        distance_m: Some(1.0),
        confidence: 0.9,
        source: ObjectObservationSource::Sim,
    });
    first.objects.observations.push(ObjectObservation {
        label: "desk".to_string(),
        class: ObjectClass::Unknown,
        bearing_rad: 0.2,
        distance_m: Some(1.5),
        confidence: 0.8,
        source: ObjectObservationSource::Sim,
    });
    memory.observe_now(&first);

    // Query from a different cell (5.0, 1.0) with the same entities -> strong overlap
    let current_key_different = Some(memory.quantize(5.0, 1.0));
    let strong_labels = vec!["chair".to_string(), "desk".to_string()];
    let candidates =
        memory.recognize_entity_constellations(current_key_different, &strong_labels, 0.0, 5);

    assert_eq!(candidates.len(), 1, "should find the overlapping cell");
    assert!(matches!(
        candidates[0].kind,
        PlaceRecognitionKind::EntityConstellation
    ));
    assert!(
        (candidates[0].similarity - 1.0).abs() < 0.001,
        "full overlap"
    );
    assert!(candidates[0].confidence > 0.0);
    assert!(candidates[0].reason.contains("entity overlap"));

    // Query from a different cell with no shared entities -> no candidates
    let weak_labels = vec!["robot".to_string(), "unknown_label".to_string()];
    let no_candidates =
        memory.recognize_entity_constellations(current_key_different, &weak_labels, 0.0, 5);
    assert!(
        no_candidates.is_empty(),
        "no shared entities means no candidates"
    );

    // Empty labels -> no candidates
    let empty = memory.recognize_entity_constellations(current_key_different, &[], 0.0, 5);
    assert!(empty.is_empty(), "empty query labels yields no candidates");
}

#[test]
fn entity_constellation_skips_current_cell_and_low_confidence() {
    let mut memory = PlaceMemory::new();
    let mut obs = now_at(100, 2.0, 1.0);
    obs.objects.observations.push(ObjectObservation {
        label: "lamp".to_string(),
        class: ObjectClass::Unknown,
        bearing_rad: 0.0,
        distance_m: Some(0.8),
        confidence: 0.9,
        source: ObjectObservationSource::Sim,
    });
    memory.observe_now(&obs);

    let current_key = Some(memory.quantize(2.0, 1.0));
    let labels = vec!["lamp".to_string()];

    // Should not return self-match
    let self_candidates = memory.recognize_entity_constellations(current_key, &labels, 0.0, 5);
    assert!(
        self_candidates.is_empty(),
        "should not return the current cell as a loop candidate"
    );

    // High confidence gate filters low-overlap candidates
    let different_key = Some(memory.quantize(10.0, 10.0));
    let high_gate = memory.recognize_entity_constellations(different_key, &labels, 0.99, 5);
    assert!(
        high_gate.is_empty(),
        "low confidence candidate should be filtered by high gate"
    );
}

// ── EntityMemory tests ───────────────────────────────────────────────────

fn make_object_observation(label: &str, class: ObjectClass, confidence: f32) -> ObjectObservation {
    ObjectObservation {
        label: label.to_string(),
        class,
        bearing_rad: 0.0,
        distance_m: Some(1.0),
        confidence,
        source: ObjectObservationSource::Sim,
    }
}

#[test]
fn entity_memory_repeated_observation_merges_not_duplicates() {
    let mut memory = EntityMemory::new();

    let mut now1 = now_at(100, 1.0, 1.0);
    now1.objects
        .observations
        .push(make_object_observation("chair", ObjectClass::Unknown, 0.8));
    memory.observe_now(&now1, Some(PlaceCellKey { x: 2, y: 2 }));

    let mut now2 = now_at(200, 1.0, 1.0);
    now2.objects
        .observations
        .push(make_object_observation("chair", ObjectClass::Unknown, 0.7));
    memory.observe_now(&now2, Some(PlaceCellKey { x: 2, y: 2 }));

    assert_eq!(
        memory.entities.len(),
        1,
        "repeated observation must merge, not duplicate"
    );
    let entity = memory.entities.values().next().unwrap();
    assert_eq!(entity.observation_count, 2);
    assert_eq!(entity.first_seen_ms, 100);
    assert_eq!(entity.last_seen_ms, 200);
}

#[test]
fn entity_memory_confidence_increases_on_re_observation() {
    let mut memory = EntityMemory::new();

    let mut now1 = now_at(100, 1.0, 1.0);
    now1.objects
        .observations
        .push(make_object_observation("desk", ObjectClass::Unknown, 0.5));
    memory.observe_now(&now1, None);
    let confidence_after_first = memory.entities.values().next().unwrap().confidence;

    let mut now2 = now_at(200, 1.0, 1.0);
    now2.objects
        .observations
        .push(make_object_observation("desk", ObjectClass::Unknown, 0.9));
    memory.observe_now(&now2, None);
    let confidence_after_second = memory.entities.values().next().unwrap().confidence;

    assert!(
            confidence_after_second > confidence_after_first * 0.9,
            "confidence should remain stable or grow on re-observation: {confidence_after_first} -> {confidence_after_second}"
        );
}

#[test]
fn entity_memory_stale_entity_transitions_to_occluded_then_vanished() {
    let mut memory = EntityMemory::new();

    let mut now1 = now_at(100, 0.0, 0.0);
    now1.objects
        .observations
        .push(make_object_observation("cup", ObjectClass::Unknown, 0.6));
    memory.observe_now(&now1, None);

    // Simulate many decay ticks by calling observe_now with no objects and a large time gap
    let mut stale = now_at(1_000_000, 0.0, 0.0); // ~10 000 ticks later
    stale.objects.observations.clear();
    memory.observe_now(&stale, None);

    let entity = memory.entities.values().next().unwrap();
    assert!(
        entity.lifecycle != EntityLifecycleState::Active,
        "entity should not remain Active after a long absence"
    );
}

#[test]
fn entity_memory_report_counts_lifecycle_states() {
    let mut memory = EntityMemory::new();

    // Active entity
    let mut now1 = now_at(100, 0.0, 0.0);
    now1.objects
        .observations
        .push(make_object_observation("sofa", ObjectClass::Unknown, 0.9));
    memory.observe_now(&now1, None);

    // A second entity that will go stale
    let mut now2 = now_at(200, 5.0, 5.0);
    now2.objects
        .observations
        .push(make_object_observation("lamp", ObjectClass::Unknown, 0.6));
    memory.observe_now(&now2, None);

    // Age both entities hard
    let stale = now_at(10_000_000, 0.0, 0.0);
    memory.observe_now(&stale, None);

    let report = memory.report();
    assert_eq!(report.total_entities, 2);
    assert!(
        report.active_entities < 2,
        "both should have decayed away from Active"
    );
}

#[test]
fn entity_memory_records_map_cell_linkage() {
    let mut memory = EntityMemory::new();
    let cell_a = PlaceCellKey { x: 3, y: 4 };
    let cell_b = PlaceCellKey { x: 7, y: 8 };

    let mut now1 = now_at(100, 0.0, 0.0);
    now1.objects.observations.push(make_object_observation(
        "robot",
        ObjectClass::Obstacle,
        0.85,
    ));
    memory.observe_now(&now1, Some(cell_a));

    let mut now2 = now_at(200, 3.5, 4.0);
    now2.objects
        .observations
        .push(make_object_observation("robot", ObjectClass::Obstacle, 0.8));
    memory.observe_now(&now2, Some(cell_b));

    let entity = memory.entities.values().next().unwrap();
    assert!(
        entity.location_cells.contains(&cell_a),
        "first observed cell must be linked"
    );
    assert!(
        entity.location_cells.contains(&cell_b),
        "second observed cell must be linked"
    );
}

#[test]
fn entity_memory_revives_after_re_sighting() {
    let mut memory = EntityMemory::new();

    let mut now1 = now_at(100, 0.0, 0.0);
    now1.objects
        .observations
        .push(make_object_observation("box", ObjectClass::Unknown, 0.7));
    memory.observe_now(&now1, None);

    // Age it into Occluded/Vanished
    let stale = now_at(500_000, 0.0, 0.0);
    memory.observe_now(&stale, None);
    let lifecycle_before = memory.entities.values().next().unwrap().lifecycle.clone();

    // Re-observe
    let mut now3 = now_at(600_000, 0.0, 0.0);
    now3.objects
        .observations
        .push(make_object_observation("box", ObjectClass::Unknown, 0.8));
    memory.observe_now(&now3, None);

    let entity = memory.entities.values().next().unwrap();
    assert_eq!(
        entity.lifecycle,
        EntityLifecycleState::Active,
        "entity should revive to Active on re-sighting; was {lifecycle_before:?}"
    );
}

#[test]
fn entity_constellation_strengthens_with_cross_modal_repetition() {
    let mut memory = EntityMemory::new();
    let mut now1 = now_at(100, 0.0, 0.0);
    now1.objects
        .observations
        .push(make_object_observation("pete", ObjectClass::Person, 0.8));
    now1.face.vectors.push(VectorArtifact::new(
        FACE_VECTOR_COLLECTION,
        "face-pete",
        vec![1.0, 0.0],
    ));
    now1.voice.vectors.push(VectorArtifact::new(
        VOICE_VECTOR_COLLECTION,
        "voice-pete",
        vec![1.0, 0.0],
    ));
    memory.observe_now(&now1, None);

    let entity_id = "entity:person:pete";
    let first_edge_evidence = memory
        .entities
        .get(entity_id)
        .and_then(|entity| entity.constellation.binding_edges.first())
        .map(|edge| (edge.evidence_count, edge.confidence))
        .expect("binding edge on first multimodal observation");

    let mut now2 = now_at(200, 0.1, 0.0);
    now2.objects
        .observations
        .push(make_object_observation("pete", ObjectClass::Person, 0.9));
    now2.face.vectors.push(VectorArtifact::new(
        FACE_VECTOR_COLLECTION,
        "face-pete",
        vec![1.0, 0.0],
    ));
    now2.voice.vectors.push(VectorArtifact::new(
        VOICE_VECTOR_COLLECTION,
        "voice-pete",
        vec![1.0, 0.0],
    ));
    memory.observe_now(&now2, None);

    let entity = memory.entities.get(entity_id).expect("person entity");
    assert!(
        entity.constellation.binding_edges.iter().any(|edge| {
            edge.evidence_count > first_edge_evidence.0 && edge.confidence > first_edge_evidence.1
        }),
        "repeated co-occurrence should strengthen at least one binding edge"
    );
}

#[test]
fn entity_constellation_attaches_provisional_text_labels() {
    let mut memory = EntityMemory::new();
    let mut now = now_at(100, 0.0, 0.0);
    now.objects.observations.push(make_object_observation(
        "person-nearby",
        ObjectClass::Person,
        0.8,
    ));
    now.ear.transcript = Some("Travis".to_string());
    memory.observe_now(&now, None);

    let entity = memory
        .entities
        .get("entity:person:person-nearby")
        .expect("person entity");
    assert!(entity
        .modality_support
        .text_labels
        .contains(&"Travis".to_string()));
    assert_eq!(
        entity.display_name.as_deref(),
        Some("person-nearby"),
        "text remains provisional and does not override sensory label"
    );
    assert!(
        entity
            .constellation
            .binding_edges
            .iter()
            .any(|edge| edge.relation == BindingRelation::NamedBy),
        "named_by edge should connect text cluster to the entity constellation"
    );
}

#[test]
fn face_vector_is_not_attached_to_every_active_person() {
    let mut memory = EntityMemory::new();
    let mut now = now_at(100, 0.0, 0.0);
    now.objects
        .observations
        .push(make_object_observation("ada", ObjectClass::Person, 0.8));
    now.objects
        .observations
        .push(make_object_observation("grace", ObjectClass::Person, 0.8));
    now.face.vectors.push(VectorArtifact::new(
        FACE_VECTOR_COLLECTION,
        "face-ambiguous",
        vec![1.0, 0.0],
    ));

    memory.observe_now(&now, Some(PlaceCellKey { x: 0, y: 0 }));

    assert!(memory
        .entities
        .values()
        .all(|entity| entity.modality_support.face_vector_ids.is_empty()));
    assert!(memory
        .report()
        .ambiguous_binding_candidates
        .iter()
        .any(|candidate| candidate.reason.contains("face vector close")));
}

#[test]
fn ambiguous_face_observation_creates_competing_tracking_hypotheses() {
    let mut memory = EntityMemory::new();
    let mut now = now_at(100, 0.0, 0.0);
    now.objects
        .observations
        .push(make_object_observation("travis", ObjectClass::Person, 0.8));
    now.objects
        .observations
        .push(make_object_observation("tim", ObjectClass::Person, 0.8));
    now.face.vectors.push(VectorArtifact::new(
        FACE_VECTOR_COLLECTION,
        "face-ambiguous-ticket5",
        vec![1.0, 0.0],
    ));

    memory.observe_now(&now, Some(PlaceCellKey { x: 0, y: 0 }));

    let family = memory
        .tracking_hypotheses
        .values()
        .filter(|hypothesis| hypothesis.family_id == "face-identity:face-ambiguous-ticket5")
        .collect::<Vec<_>>();
    assert_eq!(
        family.len(),
        3,
        "two known-person hypotheses plus one unknown-person hypothesis should remain visible"
    );
    assert!(family
        .iter()
        .any(|hypothesis| { hypothesis.target_id.as_deref() == Some("entity:person:travis") }));
    assert!(family
        .iter()
        .any(|hypothesis| hypothesis.target_id.as_deref() == Some("entity:person:tim")));
    assert!(family
        .iter()
        .any(|hypothesis| hypothesis.target_id.is_none()));
    assert!(family
        .iter()
        .any(|hypothesis| { matches!(hypothesis.state, HypothesisState::NeedsReview) }));
    assert!(memory
        .entities
        .values()
        .all(|entity| entity.modality_support.face_vector_ids.is_empty()));
}

#[test]
fn voice_vector_is_not_attached_to_every_active_person() {
    let mut memory = EntityMemory::new();
    let mut now = now_at(100, 0.0, 0.0);
    now.objects
        .observations
        .push(make_object_observation("ada", ObjectClass::Person, 0.8));
    now.objects
        .observations
        .push(make_object_observation("grace", ObjectClass::Person, 0.8));
    now.voice.vectors.push(VectorArtifact::new(
        VOICE_VECTOR_COLLECTION,
        "voice-ambiguous",
        vec![0.0, 1.0],
    ));

    memory.observe_now(&now, Some(PlaceCellKey { x: 0, y: 0 }));

    assert!(memory
        .entities
        .values()
        .all(|entity| entity.modality_support.voice_vector_ids.is_empty()));
    assert!(!memory.report().ambiguous_binding_candidates.is_empty());
}

#[test]
fn ambiguous_voice_observation_creates_competing_tracking_hypotheses() {
    let mut memory = EntityMemory::new();
    let mut now = now_at(100, 0.0, 0.0);
    now.objects
        .observations
        .push(make_object_observation("travis", ObjectClass::Person, 0.8));
    now.objects
        .observations
        .push(make_object_observation("tim", ObjectClass::Person, 0.8));
    now.voice.vectors.push(VectorArtifact::new(
        VOICE_VECTOR_COLLECTION,
        "voice-ambiguous-ticket5",
        vec![0.0, 1.0],
    ));

    memory.observe_now(&now, Some(PlaceCellKey { x: 0, y: 0 }));

    let family = memory
        .tracking_hypotheses
        .values()
        .filter(|hypothesis| hypothesis.family_id == "voice-identity:voice-ambiguous-ticket5")
        .collect::<Vec<_>>();
    assert_eq!(family.len(), 3);
    assert!(family
        .iter()
        .all(|hypothesis| { hypothesis.kind == TrackingHypothesisKind::VoiceIdentity }));
    assert!(memory
        .entities
        .values()
        .all(|entity| entity.modality_support.voice_vector_ids.is_empty()));
}

#[test]
fn single_clear_candidate_promotes_identity_binding() {
    let mut memory = EntityMemory::new();
    let cell = PlaceCellKey { x: 1, y: 1 };
    let mut now = now_at(100, 0.0, 0.0);
    now.objects
        .observations
        .push(make_object_observation("ada", ObjectClass::Person, 0.9));
    now.face.vectors.push(VectorArtifact::new(
        FACE_VECTOR_COLLECTION,
        "face-ada-ticket5",
        vec![1.0, 0.0],
    ));

    memory.observe_now(&now, Some(cell));

    let ada = memory.entities.get("entity:person:ada").unwrap();
    assert_eq!(
        ada.modality_support.face_vector_ids,
        vec!["face-ada-ticket5".to_string()]
    );
    assert!(memory
        .tracking_hypotheses
        .values()
        .any(|hypothesis| hypothesis.state == HypothesisState::Promoted
            && hypothesis.target_id.as_deref() == Some("entity:person:ada")));
}

#[test]
fn close_identity_candidates_remain_unresolved_for_review() {
    let mut memory = EntityMemory::new();
    let mut now = now_at(100, 0.0, 0.0);
    now.objects
        .observations
        .push(make_object_observation("travis", ObjectClass::Person, 0.8));
    now.objects
        .observations
        .push(make_object_observation("tim", ObjectClass::Person, 0.8));
    now.face.vectors.push(VectorArtifact::new(
        FACE_VECTOR_COLLECTION,
        "face-close-ticket5",
        vec![1.0, 0.0],
    ));

    memory.observe_now(&now, Some(PlaceCellKey { x: 0, y: 0 }));

    let report = memory.report();
    assert!(
        report.review_tracking_hypotheses.len() >= 2,
        "near-equal known-person hypotheses should be visible for review"
    );
    assert!(report.promoted_tracking_hypotheses.is_empty());
}

#[test]
fn contradiction_lowers_hypothesis_confidence() {
    let positive = score_hypothesis_evidence(
        &[
            BindingEvidence {
                kind: BindingEvidenceKind::VectorSimilarity,
                score: 0.8,
                reason: "high vector similarity".to_string(),
            },
            BindingEvidence {
                kind: BindingEvidenceKind::TemporalOverlap,
                score: 0.8,
                reason: "same time window".to_string(),
            },
            BindingEvidence {
                kind: BindingEvidenceKind::SpatialOverlap,
                score: 0.8,
                reason: "same place".to_string(),
            },
        ],
        1.0,
    );
    let contradicted = score_hypothesis_evidence(
        &[
            BindingEvidence {
                kind: BindingEvidenceKind::VectorSimilarity,
                score: 0.8,
                reason: "high vector similarity".to_string(),
            },
            BindingEvidence {
                kind: BindingEvidenceKind::TemporalOverlap,
                score: 0.8,
                reason: "same time window".to_string(),
            },
            BindingEvidence {
                kind: BindingEvidenceKind::Contradiction,
                score: 1.0,
                reason: "same person appears in incompatible places".to_string(),
            },
        ],
        1.0,
    );

    assert!(contradicted < positive);
}

#[test]
fn human_confirmation_promotes_correct_hypothesis() {
    let mut memory = EntityMemory::new();
    let mut first = now_at(100, 0.0, 0.0);
    first
        .objects
        .observations
        .push(make_object_observation("ada", ObjectClass::Person, 0.9));
    memory.observe_now(&first, None);

    let mut later = now_at(10_000, 3.0, 3.0);
    later.face.vectors.push(
        VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-human-confirmed", vec![1.0])
            .with_source_id("entity:person:ada"),
    );
    memory.observe_now(&later, None);

    let ada = memory.entities.get("entity:person:ada").unwrap();
    assert_eq!(
        ada.modality_support.face_vector_ids,
        vec!["face-human-confirmed".to_string()]
    );
    assert!(memory
        .report()
        .promoted_tracking_hypotheses
        .iter()
        .any(|hypothesis| hypothesis.target_id.as_deref() == Some("entity:person:ada")));
}

#[test]
fn stale_weak_hypotheses_expire() {
    let mut memory = EntityMemory::new();
    let mut now = now_at(100, 0.0, 0.0);
    now.face.vectors.push(VectorArtifact::new(
        FACE_VECTOR_COLLECTION,
        "face-stale-ticket5",
        vec![1.0],
    ));
    memory.observe_now(&now, None);

    let later = now_at(120_000, 0.0, 0.0);
    memory.observe_now(&later, None);

    assert!(memory
        .tracking_hypotheses
        .values()
        .any(|hypothesis| hypothesis.state == HypothesisState::Expired));
}

#[test]
fn promoted_hypothesis_does_not_merge_unrelated_entities() {
    let mut memory = EntityMemory::new();
    let mut now = now_at(100, 0.0, 0.0);
    now.objects
        .observations
        .push(make_object_observation("ada", ObjectClass::Person, 0.9));
    now.objects
        .observations
        .push(make_object_observation("grace", ObjectClass::Person, 0.9));
    memory.observe_now(&now, None);

    let mut confirmed = now_at(200, 0.0, 0.0);
    confirmed.face.vectors.push(
        VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-ada-confirmed", vec![1.0])
            .with_source_id("entity:person:ada"),
    );
    memory.observe_now(&confirmed, None);

    assert!(memory.entities.contains_key("entity:person:ada"));
    assert!(memory.entities.contains_key("entity:person:grace"));
    assert_eq!(memory.entities.len(), 2);
    assert!(memory
        .entities
        .get("entity:person:grace")
        .unwrap()
        .modality_support
        .face_vector_ids
        .is_empty());
}

#[test]
fn scene_vector_is_not_attached_to_every_active_entity() {
    let mut memory = EntityMemory::new();
    let mut first = now_at(100, 0.0, 0.0);
    first
        .objects
        .observations
        .push(make_object_observation("cup", ObjectClass::Unknown, 0.8));
    memory.observe_now(&first, Some(PlaceCellKey { x: 0, y: 0 }));

    let mut second = now_at(200, 2.0, 2.0);
    second
        .objects
        .observations
        .push(make_object_observation("lamp", ObjectClass::Unknown, 0.8));
    second.eye.scene_vectors.push(VectorArtifact::new(
        SCENE_VECTOR_COLLECTION,
        "scene-current",
        vec![0.5, 0.5],
    ));
    memory.observe_now(&second, Some(PlaceCellKey { x: 4, y: 4 }));

    let cup = memory.entities.get("entity:unknown:cup").unwrap();
    let lamp = memory.entities.get("entity:unknown:lamp").unwrap();
    assert!(cup.modality_support.scene_vector_ids.is_empty());
    assert_eq!(
        lamp.modality_support.scene_vector_ids,
        vec!["scene-current".to_string()]
    );
}

#[test]
fn candidate_with_only_one_weak_evidence_source_is_held() {
    let observation = make_object_observation("ada", ObjectClass::Person, 0.8);
    let entity = EntityHypothesis::from_observation(&observation, 100, None);
    let candidate = qualify_binding_candidate(
        &entity,
        &VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-only", vec![1.0]),
        VectorBindingKind::Face,
        entity.primary_object_cluster_id().unwrap(),
        vector_cluster_id(VectorBindingKind::Face, "face-only"),
        5_000,
        None,
        0,
        false,
    );

    assert_ne!(candidate.decision, BindingDecision::Accept);
    assert!(candidate.reason.contains("single vector"));
}

#[test]
fn candidate_with_temporal_and_spatial_evidence_is_accepted() {
    let observation = make_object_observation("ada", ObjectClass::Person, 0.8);
    let cell = PlaceCellKey { x: 1, y: 2 };
    let entity = EntityHypothesis::from_observation(&observation, 100, Some(cell));
    let candidate = qualify_binding_candidate(
        &entity,
        &VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-ada", vec![1.0]),
        VectorBindingKind::Face,
        entity.primary_object_cluster_id().unwrap(),
        vector_cluster_id(VectorBindingKind::Face, "face-ada"),
        150,
        Some(cell),
        1,
        false,
    );

    assert_eq!(candidate.decision, BindingDecision::Accept);
}

#[test]
fn rejected_candidate_includes_useful_reason() {
    let observation = make_object_observation("ada", ObjectClass::Person, 0.8);
    let entity = EntityHypothesis::from_observation(&observation, 100, None);
    let candidate = qualify_binding_candidate(
        &entity,
        &VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-grace", vec![1.0])
            .with_source_id("entity:person:grace"),
        VectorBindingKind::Face,
        entity.primary_object_cluster_id().unwrap(),
        vector_cluster_id(VectorBindingKind::Face, "face-grace"),
        150,
        None,
        1,
        false,
    );

    assert_eq!(candidate.decision, BindingDecision::Reject);
    assert!(candidate
        .reason
        .contains("contradicts explicit entity source"));
    assert!(candidate.evidence.iter().any(|evidence| {
        evidence.kind == BindingEvidenceKind::Contradiction
            && evidence.reason.contains("entity:person:grace")
            && evidence.reason.contains("entity:person:ada")
    }));
}

#[test]
fn cross_modal_engine_proposes_face_voice_candidate_without_mutation() {
    let mut engine = DefaultCrossModalBindingEngine::default();
    let context = BindingContext::new(1_000);
    let clusters = vec![
        DiscoveredCluster::new(
            "face-a",
            Modality::Vision,
            DiscoveredClusterKind::Face,
            1_000,
            0.9,
        ),
        DiscoveredCluster::new(
            "voice-a",
            Modality::Audio,
            DiscoveredClusterKind::Voice,
            1_050,
            0.85,
        ),
    ];

    let candidates = engine.propose_bindings(&context, &clusters);

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].relation, BindingRelation::LikelySameEntity);
    assert!(candidates[0]
        .evidence
        .iter()
        .any(|evidence| evidence.kind == BindingEvidenceKind::TemporalOverlap));
    assert!(candidates[0]
        .evidence
        .iter()
        .any(|evidence| evidence.kind == BindingEvidenceKind::SingleCandidateContext));
    assert_ne!(candidates[0].decision, BindingDecision::Accept);
    assert!(candidates[0].reason.contains("admission must accept"));
    assert_eq!(
        clusters.len(),
        2,
        "engine must not mutate clusters or entities"
    );
}

#[test]
fn cross_modal_engine_holds_ambiguous_face_voice_context() {
    let mut engine = DefaultCrossModalBindingEngine::default();
    let context = BindingContext::new(1_000);
    let clusters = vec![
        DiscoveredCluster::new(
            "face-a",
            Modality::Vision,
            DiscoveredClusterKind::Face,
            1_000,
            0.9,
        ),
        DiscoveredCluster::new(
            "face-b",
            Modality::Vision,
            DiscoveredClusterKind::Face,
            1_010,
            0.9,
        ),
        DiscoveredCluster::new(
            "voice-a",
            Modality::Audio,
            DiscoveredClusterKind::Voice,
            1_020,
            0.9,
        ),
        DiscoveredCluster::new(
            "voice-b",
            Modality::Audio,
            DiscoveredClusterKind::Voice,
            1_030,
            0.9,
        ),
    ];

    let candidates = engine.propose_bindings(&context, &clusters);

    assert!(!candidates.is_empty());
    assert!(candidates.iter().all(|candidate| {
        candidate.decision == BindingDecision::HoldAmbiguous
            && candidate
                .evidence
                .iter()
                .any(|evidence| evidence.kind == BindingEvidenceKind::SimultaneousConflict)
    }));
}

#[test]
fn cross_modal_engine_proposes_rgb_geometry_projection() {
    let mut engine = DefaultCrossModalBindingEngine::default();
    let context = BindingContext {
        source_frame_id: Some("frame-1".to_string()),
        ..BindingContext::new(1_000)
    };
    let rgb = DiscoveredCluster::new(
        "rgb-patch",
        Modality::Vision,
        DiscoveredClusterKind::RgbImage,
        1_000,
        0.8,
    )
    .with_source_frame_id("frame-1")
    .with_metadata(json!({ "image_x": 100.0, "image_y": 50.0 }));
    let depth = DiscoveredCluster::new(
        "voxel",
        Modality::Depth,
        DiscoveredClusterKind::Geometry,
        1_000,
        0.8,
    )
    .with_source_frame_id("frame-1")
    .with_metadata(json!({ "projected_image_x": 102.0, "projected_image_y": 53.0 }));

    let candidates = engine.propose_bindings(&context, &[rgb, depth]);

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].relation, BindingRelation::ProjectsTo);
    assert!(candidates[0]
        .evidence
        .iter()
        .any(|evidence| evidence.kind == BindingEvidenceKind::ProjectionAgreement));
}

#[test]
fn cross_modal_engine_rejects_rgb_geometry_projection_disagreement() {
    let mut engine = DefaultCrossModalBindingEngine::default();
    let context = BindingContext {
        source_frame_id: Some("frame-1".to_string()),
        ..BindingContext::new(1_000)
    };
    let rgb = DiscoveredCluster::new(
        "rgb-patch",
        Modality::Vision,
        DiscoveredClusterKind::RgbImage,
        1_000,
        0.8,
    )
    .with_source_frame_id("frame-1")
    .with_metadata(json!({ "image_x": 100.0, "image_y": 50.0 }));
    let depth = DiscoveredCluster::new(
        "voxel",
        Modality::Depth,
        DiscoveredClusterKind::Geometry,
        1_000,
        0.8,
    )
    .with_source_frame_id("frame-1")
    .with_metadata(json!({ "projected_image_x": 180.0, "projected_image_y": 110.0 }));

    let candidates = engine.propose_bindings(&context, &[rgb, depth]);

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].relation, BindingRelation::ProjectsTo);
    assert_eq!(candidates[0].decision, BindingDecision::Reject);
    assert!(candidates[0]
        .evidence
        .iter()
        .any(|evidence| evidence.kind == BindingEvidenceKind::Contradiction));
}

#[test]
fn cross_modal_engine_proposes_object_place_binding() {
    let mut engine = DefaultCrossModalBindingEngine::default();
    let cell = PlaceCellKey { x: 2, y: 3 };
    let context = BindingContext {
        current_place_cell: Some(cell),
        ..BindingContext::new(1_000)
    };
    let object = DiscoveredCluster::new(
        "charger-object",
        Modality::Vision,
        DiscoveredClusterKind::Object,
        1_000,
        0.9,
    )
    .with_place_cell(cell)
    .with_metadata(json!({ "cooccurrence_count": 3 }));
    let place = DiscoveredCluster::new(
        "dock-place",
        Modality::Memory,
        DiscoveredClusterKind::Place,
        950,
        0.8,
    )
    .with_place_cell(cell);

    let candidates = engine.propose_bindings(&context, &[object, place]);

    assert_eq!(candidates.len(), 1);
    assert_eq!(
        candidates[0].relation,
        BindingRelation::CooccursInEstimatedSpace
    );
    assert!(candidates[0]
        .evidence
        .iter()
        .any(|evidence| evidence.kind == BindingEvidenceKind::SpatialOverlap));
    assert!(candidates[0]
        .evidence
        .iter()
        .any(|evidence| evidence.kind == BindingEvidenceKind::RepeatedCooccurrence));
}

#[test]
fn cross_modal_engine_proposes_action_outcome_binding() {
    let mut engine = DefaultCrossModalBindingEngine::default();
    let context = BindingContext {
        active_action: Some(ActionPrimitive::Go {
            intensity: 0.5,
            duration_ms: 500,
        }),
        body_state: Some(BodySense {
            flags: BodyFlags {
                bump_left: true,
                ..BodyFlags::default()
            },
            ..BodySense::default()
        }),
        ..BindingContext::new(1_000)
    };
    let action = DiscoveredCluster::new(
        "go-forward",
        Modality::Odometry,
        DiscoveredClusterKind::Action,
        1_000,
        0.8,
    );
    let outcome = DiscoveredCluster::new(
        "bump-left",
        Modality::Touch,
        DiscoveredClusterKind::Outcome,
        1_400,
        0.9,
    );

    let candidates = engine.propose_bindings(&context, &[action, outcome]);

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].relation, BindingRelation::ExplainsOutcome);
    assert!(candidates[0]
        .evidence
        .iter()
        .any(|evidence| evidence.reason.contains("outcome followed action")));
}

#[test]
fn cross_modal_engine_keeps_llm_label_alone_weak() {
    let mut engine = DefaultCrossModalBindingEngine::default();
    let context = BindingContext::new(1_000);
    let label = DiscoveredCluster::new(
        "llm-label-chair",
        Modality::Language,
        DiscoveredClusterKind::Label,
        1_000,
        0.7,
    )
    .with_metadata(json!({ "source": "llm", "label": "chair" }));
    let object = DiscoveredCluster::new(
        "object-blob",
        Modality::Vision,
        DiscoveredClusterKind::Object,
        5_000,
        0.7,
    );

    let candidates = engine.propose_bindings(&context, &[label, object]);

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].relation, BindingRelation::NamedBy);
    assert_eq!(candidates[0].decision, BindingDecision::CollectMoreEvidence);
    assert!(candidates[0].reason.contains("LLM suggestion alone"));
}

#[test]
fn cross_modal_engine_never_accepts_without_admission() {
    let mut engine = DefaultCrossModalBindingEngine::default();
    let cell = PlaceCellKey { x: 2, y: 3 };
    let context = BindingContext {
        current_place_cell: Some(cell),
        recent_clusters: vec!["charger-object".to_string(), "dock-place".to_string()],
        ..BindingContext::new(1_000)
    };
    let object = DiscoveredCluster::new(
        "charger-object",
        Modality::Vision,
        DiscoveredClusterKind::Object,
        1_000,
        0.95,
    )
    .with_place_cell(cell)
    .with_metadata(json!({ "cooccurrence_count": 5 }));
    let place = DiscoveredCluster::new(
        "dock-place",
        Modality::Memory,
        DiscoveredClusterKind::Place,
        1_000,
        0.95,
    )
    .with_place_cell(cell);

    let candidates = engine.propose_bindings(&context, &[object, place]);

    assert_eq!(candidates.len(), 1);
    assert_ne!(candidates[0].decision, BindingDecision::Accept);
    assert!(candidates[0].reason.contains("admission must accept"));
}

#[test]
fn accepted_candidates_strengthen_existing_binding_edges() {
    let mut memory = EntityMemory::new();
    let cell = PlaceCellKey { x: 0, y: 0 };
    let mut now1 = now_at(100, 0.0, 0.0);
    now1.objects
        .observations
        .push(make_object_observation("ada", ObjectClass::Person, 0.8));
    now1.face.vectors.push(VectorArtifact::new(
        FACE_VECTOR_COLLECTION,
        "face-ada",
        vec![1.0, 0.0],
    ));
    memory.observe_now(&now1, Some(cell));

    let before = memory.entities["entity:person:ada"]
        .constellation
        .binding_edges[0]
        .clone();

    let mut now2 = now_at(200, 0.0, 0.0);
    now2.objects
        .observations
        .push(make_object_observation("ada", ObjectClass::Person, 0.9));
    now2.face.vectors.push(VectorArtifact::new(
        FACE_VECTOR_COLLECTION,
        "face-ada",
        vec![1.0, 0.0],
    ));
    memory.observe_now(&now2, Some(cell));

    let after = &memory.entities["entity:person:ada"]
        .constellation
        .binding_edges[0];
    assert!(after.evidence_count > before.evidence_count);
    assert!(after.confidence > before.confidence);
}

#[test]
fn strong_constellation_requires_multiple_supporting_bindings() {
    let observation = make_object_observation("ada", ObjectClass::Person, 0.8);
    let mut entity = EntityHypothesis::from_observation(&observation, 100, None);
    let object_cluster = entity.primary_object_cluster_id().unwrap();
    let face_cluster = entity.add_face_vector("face-ada");
    for step in 0..5 {
        entity.upsert_binding_edge(
            object_cluster.clone(),
            face_cluster.clone(),
            BindingRelation::LikelySameEntity,
            1.0,
            100 + step,
        );
    }
    assert_eq!(entity.constellation.state, EntityConstellationState::Weak);

    let voice_cluster = entity.add_voice_vector("voice-ada");
    for step in 0..5 {
        entity.upsert_binding_edge(
            object_cluster.clone(),
            voice_cluster.clone(),
            BindingRelation::LikelySameEntity,
            1.0,
            200 + step,
        );
    }
    assert_eq!(entity.constellation.state, EntityConstellationState::Strong);
}

#[test]
fn binding_admission_does_not_merge_entities() {
    let mut memory = EntityMemory::new();
    let mut now = now_at(100, 0.0, 0.0);
    now.objects
        .observations
        .push(make_object_observation("ada", ObjectClass::Person, 0.8));
    now.objects
        .observations
        .push(make_object_observation("grace", ObjectClass::Person, 0.8));
    now.face.vectors.push(VectorArtifact::new(
        FACE_VECTOR_COLLECTION,
        "face-ambiguous",
        vec![1.0, 0.0],
    ));

    memory.observe_now(&now, Some(PlaceCellKey { x: 0, y: 0 }));

    assert_eq!(memory.entities.len(), 2);
    assert!(memory.entities.contains_key("entity:person:ada"));
    assert!(memory.entities.contains_key("entity:person:grace"));
}

#[test]
fn entity_constellation_supports_experience_binding_and_merge_split_states() {
    let mut memory = EntityMemory::new();
    let mut frame = empty_frame(now_at(100, 1.0, 1.0));
    frame.now.objects.observations.push(make_object_observation(
        "charger",
        ObjectClass::Charger,
        0.9,
    ));
    let sensation_id = uuid::Uuid::new_v4();
    let mut experience = Experience::new(
        "embodied.place",
        "charger alcove",
        Vec::new(),
        vec![sensation_id],
        90,
        100,
    );
    experience.salience = 0.9;
    frame.experiences.push(experience);
    frame.impressions.push(
        Impression::new("memory.impression", "charger", Vec::new(), 90, 100).with_confidence(0.8),
    );
    memory.observe_frame(&frame, Some(PlaceCellKey { x: 2, y: 2 }));

    let entity_id = "entity:charger:charger".to_string();
    let entity = memory.entities.get(&entity_id).expect("charger entity");
    assert!(
        entity
            .constellation
            .modality_clusters
            .iter()
            .any(|cluster| cluster.modality == Modality::Memory),
        "experience-level memory clusters should bind into the entity constellation"
    );

    let split_id = memory
        .split_entity(&entity_id, "left")
        .expect("split child id");
    assert!(memory.merge_entities(&entity_id, &split_id));
    let merged = memory.entities.get(&entity_id).expect("merged entity");
    assert_eq!(merged.constellation.state, EntityConstellationState::Merged);
}
