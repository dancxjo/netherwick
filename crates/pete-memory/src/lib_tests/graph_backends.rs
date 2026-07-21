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
