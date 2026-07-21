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
