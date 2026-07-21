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
