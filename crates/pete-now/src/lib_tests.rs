use super::*;

#[test]
fn transcript_tracker_emits_started_then_finalized_for_final_only_asr() {
    let mut tracker = TranscriptCandidateTracker::new();

    let events = tracker.ingest_chunk(TranscriptChunk {
        text: "hello there".to_string(),
        is_final: true,
    });

    assert_eq!(
        events,
        vec![
            TranscriptCandidateEvent::CandidateStarted {
                id: TranscriptCandidateId(1)
            },
            TranscriptCandidateEvent::CandidateFinalized {
                id: TranscriptCandidateId(1),
                text: "hello there".to_string(),
                confidence: None,
            },
        ]
    );
}

#[test]
fn transcript_tracker_keeps_word_boundary_stable_prefix() {
    let mut tracker = TranscriptCandidateTracker::new();
    let _ = tracker.ingest_candidate("can you tell", Some(0.4), false);

    let events = tracker.ingest_candidate("can you help", Some(0.5), false);

    assert_eq!(
        events,
        vec![
            TranscriptCandidateEvent::CandidateReplaced {
                old: TranscriptCandidateId(1),
                new: TranscriptCandidateId(2),
                reason: TranscriptReplacementReason::HeadChanged {
                    stable_prefix_len: "can you ".len(),
                },
            },
            TranscriptCandidateEvent::CandidateStarted {
                id: TranscriptCandidateId(2),
            },
            TranscriptCandidateEvent::CandidateUpdated {
                id: TranscriptCandidateId(2),
                text: "can you help".to_string(),
                stable_prefix_len: "can you ".len(),
                confidence: Some(0.5),
            },
        ]
    );
}

#[test]
fn transcript_stability_tracks_stable_word_prefix() {
    let state = TranscriptStabilityState::from_parts(
        TranscriptCandidateId(7),
        "hello wor",
        "hello wor".len(),
        Some(0.6),
    );

    assert_eq!(state.stable_text, "hello wor");
    assert_eq!(state.stable_word_prefix.as_deref(), Some("hello"));
    assert_eq!(state.stable_word_count, 1);
    assert_eq!(state.unstable_text, "");
    assert_eq!(state.confidence, Some(0.6));
}

#[test]
fn now_emits_queryable_features_across_modalities() {
    let mut now = Now::blank(1_000, BodySense::default());
    now.body.infrared_character = 248;
    now.objects.observations.push(ObjectObservation {
        label: "Ada".to_string(),
        class: ObjectClass::Person,
        bearing_rad: 0.25,
        distance_m: Some(1.5),
        confidence: 0.9,
        source: ObjectObservationSource::Kinect,
    });
    now.face.vectors.push(
        VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-ada", vec![1.0, 0.0])
            .with_source_frame_id("frame-1")
            .with_source_id("face-crop-1"),
    );
    now.voice.vectors.push(VectorArtifact::new(
        VOICE_VECTOR_COLLECTION,
        "voice-ada",
        vec![0.0, 1.0],
    ));
    now.ear.transcript = Some("hello Pete".to_string());
    now.ear.asr.transcript = Some("hello Pete".to_string());
    now.ear.asr.confidence = 0.7;
    now.ear.asr.is_final = true;
    now.imu.captured_at_ms = 990;
    now.imu.acceleration = vec![0.0, 0.0, 1.0];
    now.memory.remembered_entities.push(GraphEntity {
        id: "entity:person:ada".to_string(),
        labels: vec!["Person".to_string()],
        summary: "Ada was here before".to_string(),
        score: 0.8,
    });
    now.predictions
        .expected_events
        .push("speech_heard".to_string());
    now.predictions.uncertainty = 0.2;

    let registry = now.feature_registry();

    let create_ir = registry.by_source_sensor("create_ir");
    assert_eq!(create_ir.len(), 1);
    assert_eq!(create_ir[0].metadata["infrared_character"], 248);

    assert!(registry
        .by_modality(FeatureModality::Vision)
        .iter()
        .any(|feature| feature.feature_type == FeatureType::ObjectDetection));
    assert_eq!(registry.by_source_frame("frame-1").len(), 1);
    assert_eq!(registry.by_vector_id("face-ada").len(), 1);
    assert!(registry
        .by_modality(FeatureModality::Audio)
        .iter()
        .any(|feature| feature.feature_type == FeatureType::VoiceEmbedding));
    assert!(registry
        .by_modality(FeatureModality::Language)
        .iter()
        .any(|feature| feature.feature_type == FeatureType::Transcript));
    assert!(registry
        .by_modality(FeatureModality::Memory)
        .iter()
        .any(|feature| feature.feature_type == FeatureType::RememberedEntity));
    assert!(registry
        .by_modality(FeatureModality::Prediction)
        .iter()
        .any(|feature| feature.feature_type == FeatureType::FuturePrediction));
    assert!(!registry.by_time_window(990, 1_000).is_empty());
}
