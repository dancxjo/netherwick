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
