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
