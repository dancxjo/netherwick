
use super::*;
use pete_body::BodySense;
use pete_now::{AsrSense, Now};

#[test]
fn feature_encoder_produces_non_empty_latent() {
    let mut encoder = FeatureExperienceEncoder::new();
    let mut now = Now::blank(42, BodySense::default());
    now.memory.place_familiarity = 0.7;
    now.drives.curiosity = 0.5;

    let latent = encoder.encode(&now).unwrap();

    assert_eq!(latent.t_ms, 42);
    assert!(!latent.z.is_empty());
    assert!(latent.z.iter().any(|value| *value > 0.0));
}

#[test]
fn feature_encoder_consumes_create_ir_without_shifting_prior_channels() {
    let mut encoder = FeatureExperienceEncoder::new();
    let mut now = Now::blank(42, BodySense::default());
    now.body.infrared_character = 248;

    let latent = encoder.encode(&now).unwrap();
    let ir_features = &latent.z[latent.z.len() - 2..];

    assert_eq!(ir_features[0], 1.0);
    assert_eq!(ir_features[1], 248.0 / 255.0);
}

#[test]
fn now_with_sensors_produces_non_empty_experience_encoder_input() {
    let mut now = Now::blank(42, BodySense::default());
    now.eye.frames = vec![vec![0.2, 0.4, 0.6, 0.8]];
    now.ear.features = vec![vec![0.1, 0.3, 0.5, 0.7]];
    now.memory.place_familiarity = 0.7;
    now.drives.curiosity = 0.5;

    let input = experience_encode_input_from_now(&now);
    let target = experience_decode_target_from_now(&now);

    assert_eq!(input.sense_vectors.len(), 6);
    assert!(!input.flat_features().is_empty());
    assert_eq!(input.flat_features().len(), target.flat_features().len());
    assert_eq!(target.eye_features.len(), 16);
    assert_eq!(target.ear_features.len(), 16);
}

#[tokio::test]
async fn live_now_builds_canonical_experience_instant_with_missingness() {
    let mut now = Now::blank(42, BodySense::default());
    now.range.nearest_m = Some(0.8);
    now.range.beams = vec![0.8, 1.2, 1.6];
    now.kinect.depth_m = vec![0.7, 1.1, 1.5];
    now.kinect.depth_width = 3;
    now.kinect.depth_height = 1;
    now.ear.asr = AsrSense {
        transcript: Some("hello pete".to_string()),
        confidence: 0.8,
        ..AsrSense::default()
    };

    let instant = ExperienceInstant::from_now(&now, Some(ActionPrimitive::Stop))
        .await
        .unwrap();
    let input = ExperienceEncodeInput::from_instant(&instant);

    assert_eq!(instant.schema_version, 1);
    assert!(instant.primary_sensations.len() > 0);
    assert!(instant.descendant_sensations.len() > 0);
    assert!(instant.teacher_vectors.len() > 0);
    assert!(instant.lineage.len() > 0);
    assert!(instant
        .missing_modalities
        .iter()
        .any(|missing| missing.modality == Modality::Vision));
    assert_eq!(
        instant.modality_mask().len(),
        expected_instant_modalities().len()
    );
    assert!(!input.flat_features().is_empty());
}

#[test]
fn ear_features_include_finalized_asr_metadata() {
    let mut now = Now::blank(42, BodySense::default());
    now.ear.features = vec![vec![0.1, 0.3, 0.5, 0.7]];
    now.ear.asr = AsrSense {
        transcript: Some("hello world again".to_string()),
        is_final: true,
        confidence: 0.72,
        sequence_start: Some(10),
        sequence_end: Some(13),
        start_ms: Some(100),
        end_ms: Some(1_100),
        duration_ms: Some(1_000),
        sample_rate_hz: Some(16_000),
        word_count: Some(3),
        speaker_confidence: Some(0.6),
        ..AsrSense::default()
    };

    let features = ear_next_features(&now);

    assert_eq!(features.len(), 16);
    assert_eq!(&features[..4], &[0.1, 0.3, 0.5, 0.7]);
    assert_eq!(features[8], 1.0);
    assert_eq!(features[9], 1.0);
    assert_eq!(features[10], 0.72);
    assert!(features[11] > 0.0);
    assert!(features[13] > 0.0);
    assert!(features[14] > 0.0);
}

#[test]
fn transcript_only_asr_still_reaches_now_vector() {
    let mut now = Now::blank(42, BodySense::default());
    now.ear.transcript = Some("come over here".to_string());

    let target = experience_decode_target_from_now(&now);
    let asr = asr_features(&now);

    assert_eq!(target.ear_features.len(), 16);
    assert_eq!(target.ear_features[8], 1.0);
    assert_eq!(asr[0], 1.0);
    assert!(asr[3] > 0.0);
    assert!(target.flat_features().iter().any(|value| *value > 0.0));
}

#[test]
fn stasis_predictor_clones_latent_and_decays_confidence() {
    let mut predictor = StasisFuturePredictor;
    let latent = ExperienceLatent {
        t_ms: 10,
        z: vec![0.1, 0.2],
        confidence: 0.8,
        ..ExperienceLatent::default()
    };

    let near = predictor
        .predict(&latent, &ActionPrimitive::Stop, 1_000)
        .unwrap();
    let far = predictor
        .predict(&latent, &ActionPrimitive::Stop, 5_000)
        .unwrap();

    assert_eq!(near.predicted_z, latent.z);
    assert!(near.confidence > far.confidence);
    assert!(near
        .summary
        .as_deref()
        .unwrap_or_default()
        .contains("stable"));
}

#[test]
fn reward_tastes_low_battery_charging_as_good() {
    let computer = BaselineRewardComputer;
    let mut before = Now::blank(1, BodySense::default());
    before.body.battery_level = 0.2;
    before.body.charging = false;
    let mut after = before.clone();
    after.t_ms = 2;
    after.body.battery_level = 0.24;
    after.body.charging = true;

    let reward = computer.compute(
        &before,
        Some(&ActionPrimitive::Dock),
        &after,
        &SurpriseSense::default(),
    );

    assert!(reward.value > 0.0);
}

#[test]
fn reward_tastes_safe_discovery_as_good() {
    let computer = BaselineRewardComputer;
    let mut before = Now::blank(1, BodySense::default());
    before.memory.place_novelty = 0.1;
    before.memory.places_visited = 1;
    let mut after = before.clone();
    after.t_ms = 2;
    after.memory.place_novelty = 0.8;
    after.memory.places_visited = 2;
    after.body.odometry.x_m = 0.12;
    after.body.velocity.forward_m_s = 0.12;

    let reward = computer.compute(
        &before,
        Some(&ActionPrimitive::Explore {
            style: ExploreStyle::Wander,
            duration_ms: 1_000,
        }),
        &after,
        &SurpriseSense {
            total: 0.4,
            prediction_error: 0.4,
            ..SurpriseSense::default()
        },
    );

    assert!(reward.value > 0.08);
}

#[test]
fn reward_keeps_hazard_surprise_negative() {
    let computer = BaselineRewardComputer;
    let before = Now::blank(1, BodySense::default());
    let mut after = before.clone();
    after.t_ms = 2;
    after.body.flags.bump_left = true;

    let reward = computer.compute(
        &before,
        Some(&ActionPrimitive::Go {
            intensity: 0.2,
            duration_ms: 1_000,
        }),
        &after,
        &SurpriseSense {
            total: 0.6,
            prediction_error: 0.1,
            ..SurpriseSense::default()
        },
    );

    assert!(reward.value < -0.2);
}

#[test]
fn danger_target_marks_bump_and_cliff_labels() {
    let before = Now::blank(1, BodySense::default());
    let mut after = before.clone();
    after.body.flags.bump_left = true;
    after.body.flags.cliff_right = true;

    let target = danger_target_from_transition_like(&before, Some(&ActionPrimitive::Stop), &after);

    assert_eq!(target.bump, 1.0);
    assert_eq!(target.cliff, 1.0);
    assert_eq!(target.wheel_drop, 0.0);
}

#[test]
fn danger_target_marks_go_with_no_movement_as_stuck() {
    let before = Now::blank(1, BodySense::default());
    let mut after = before.clone();
    after.t_ms = 2;

    let target = danger_target_from_transition_like(
        &before,
        Some(&ActionPrimitive::Go {
            intensity: 0.4,
            duration_ms: 1_000,
        }),
        &after,
    );

    assert_eq!(target.stuck, 1.0);
}

#[test]
fn danger_action_features_are_fixed_width() {
    let now = Now::blank(1, BodySense::default());
    let stop = DangerInput::from_parts(vec![0.0], Some(&ActionPrimitive::Stop), &now);
    let go = DangerInput::from_parts(
        vec![0.0],
        Some(&ActionPrimitive::Go {
            intensity: 0.4,
            duration_ms: 1_000,
        }),
        &now,
    );
    let turn = DangerInput::from_parts(
        vec![0.0],
        Some(&ActionPrimitive::Turn {
            direction: pete_actions::TurnDir::Left,
            intensity: 0.4,
            duration_ms: 1_000,
        }),
        &now,
    );

    assert_eq!(stop.action_features.len(), go.action_features.len());
    assert_eq!(go.action_features.len(), turn.action_features.len());
}

#[test]
fn danger_input_includes_cliff_sensor_channels() {
    let mut now = Now::blank(1, BodySense::default());
    now.body.cliff_sensors.front_left = 0.8;

    let input = DangerInput::from_parts(vec![0.0], Some(&ActionPrimitive::Stop), &now);

    assert!(input.body_features.contains(&0.8));
    assert_eq!(input.body_features[3], 1.0);
}

#[test]
fn charge_target_marks_transition_onto_charger() {
    let mut before = Now::blank(1, BodySense::default());
    before.body.charging = false;
    before.body.battery_level = 0.2;
    let mut after = before.clone();
    after.body.charging = true;
    after.body.battery_level = 0.24;

    let target = charge_target_from_transition_like(&before, Some(&ActionPrimitive::Dock), &after);

    assert_eq!(target.charging_started, 1.0);
    assert_eq!(target.charging_after, 1.0);
    assert!(target.battery_delta > 0.0);
}

#[test]
fn charge_target_marks_transition_off_charger() {
    let mut before = Now::blank(1, BodySense::default());
    before.body.charging = true;
    let mut after = before.clone();
    after.body.charging = false;

    let target = charge_target_from_transition_like(&before, Some(&ActionPrimitive::Stop), &after);

    assert_eq!(target.charging_started, 0.0);
    assert_eq!(target.charging_after, 0.0);
}

#[test]
fn charge_input_includes_ir_sensor_summary() {
    let mut now = Now::blank(1, BodySense::default());
    now.kinect.ir = vec![0.1, 0.8, 0.9, 0.2];

    let input = ChargeInput::from_parts(vec![0.0], Some(&ActionPrimitive::Dock), &now);

    assert!(input.body_features.iter().any(|value| *value >= 0.8));
}

#[test]
fn action_value_input_includes_input_sensor_channels() {
    let mut now = Now::blank(1, BodySense::default());
    now.body.cliff_sensors.front_right = 0.7;
    now.kinect.ir = vec![0.1, 0.8, 0.9, 0.2];

    let input = ActionValueInput::from_parts(vec![0.0], Some(&ActionPrimitive::Dock), &now);

    assert!(input.body_features.contains(&0.7));
    assert!(input.body_features.iter().any(|value| *value >= 0.8));
}

#[test]
fn action_value_target_positive_for_charging_reward() {
    let reward = Reward { value: 0.35 };
    let surprise = SurpriseSense {
        total: 0.2,
        ..SurpriseSense::default()
    };

    let target = action_value_target_from_reward_surprise(&reward, &surprise);

    assert!(target.value > 0.0);
}

#[test]
fn action_value_target_values_safe_prediction_error() {
    let reward = Reward { value: 0.0 };
    let surprise = SurpriseSense {
        total: 0.5,
        prediction_error: 0.5,
        ..SurpriseSense::default()
    };

    let target = action_value_target_from_reward_surprise(&reward, &surprise);

    assert!(target.value > 0.0);
}

#[test]
fn action_value_target_negative_for_bump_or_cliff_transition() {
    let reward = Reward { value: -0.8 };
    let surprise = SurpriseSense {
        total: 0.4,
        ..SurpriseSense::default()
    };

    let target = action_value_target_from_reward_surprise(&reward, &surprise);

    assert!(target.value < 0.0);
}

#[test]
fn action_value_input_uses_prediction_channels() {
    let mut now = Now::blank(1, BodySense::default());
    now.predictions.danger_model = Some(pete_now::DangerPrediction {
        bump_risk: 0.7,
        confidence: 0.8,
        ..pete_now::DangerPrediction::default()
    });
    now.predictions.charge_model = Some(pete_now::ChargePrediction {
        charge_probability: 0.6,
        expected_battery_delta: 0.1,
        dock_likelihood: 0.5,
        confidence: 0.9,
    });

    let input = ActionValueInput::from_parts(vec![0.0], Some(&ActionPrimitive::Dock), &now);

    assert!(input.prediction_features.contains(&0.7));
    assert!(input.prediction_features.contains(&0.6));
}

#[test]
fn deterministic_extractor_preserves_visual_lineage() {
    let primary = Sensation::primary(
        Modality::Vision,
        SensationSource::new("test-camera"),
        100,
        100,
        SensationPayload::image_metadata(64, 48, "rgb8", 64 * 48 * 3),
    );

    let descendants = DeterministicDescendantExtractor.extract(&primary).unwrap();

    assert_eq!(descendants.len(), 1);
    assert_eq!(descendants[0].parent_id, Some(primary.id));
    assert_eq!(descendants[0].payload_kind, SensationPayloadKind::Crop);
    assert!(matches!(
        descendants[0].provenance.kind,
        pete_core::ProvenanceKind::DerivedFromSensations { .. }
    ));
}

#[test]
fn audio_extractor_derives_asr_voice_speech_and_transcript_spans() {
    let mut primary = Sensation::primary(
        Modality::Audio,
        SensationSource::new("test-ear"),
        1_000,
        1_900,
        SensationPayload {
            kind: SensationPayloadKind::AudioPcm,
            value: json!({
                "feature_sets": 4,
                "transcript": "hello there",
                "asr": {
                    "transcript": "hello there",
                    "is_final": true,
                    "confidence": 0.82,
                    "start_ms": 1_000,
                    "end_ms": 1_900,
                    "duration_ms": 900,
                    "sample_rate_hz": 16_000,
                    "word_count": 2,
                    "speaker_confidence": 0.61,
                },
            }),
        },
    );
    primary.metadata.duration_ms = Some(900);
    primary.metadata.confidence = Some(0.82);

    let descendants = DeterministicDescendantExtractor.extract(&primary).unwrap();

    assert!(descendants
        .iter()
        .any(|sensation| sensation.payload_kind == SensationPayloadKind::VoiceSegment));
    let speech = descendants
        .iter()
        .find(|sensation| sensation.payload_kind == SensationPayloadKind::SpeechSegment)
        .expect("speech span");
    assert_eq!(speech.parent_id, Some(primary.id));
    assert_eq!(speech.occurred_at_ms, 1_000);
    assert_eq!(speech.metadata.duration_ms, Some(900));
    assert_eq!(
        speech.payload.get("text").and_then(Value::as_str),
        Some("hello there")
    );
    assert!(speech
        .provenance
        .stage_chain
        .contains(&"descendant.audio_speech_span".to_string()));
    assert!(descendants
        .iter()
        .any(|sensation| sensation.payload_kind == SensationPayloadKind::TranscriptSpan));
    assert!(descendants.iter().any(|sensation| {
        sensation
            .summary
            .as_deref()
            .is_some_and(|summary| summary == "I hear someone say \"hello there\".")
    }));
}

#[test]
fn audio_extractor_reports_possible_and_committed_speech() {
    let mut primary = Sensation::primary(
        Modality::Audio,
        SensationSource::new("test-ear"),
        2_000,
        2_600,
        SensationPayload {
            kind: SensationPayloadKind::AudioPcm,
            value: json!({
                "feature_sets": 2,
                "transcript": "turn left",
                "asr": {
                    "transcript": "turn left",
                    "possible_transcript": "turn le",
                    "committed_transcript": "turn left",
                    "is_final": true,
                    "confidence": 0.78,
                    "candidate_id": 4,
                    "stable_text": "turn left",
                    "unstable_text": "",
                    "stable_word_prefix": "turn left",
                    "stable_word_count": 2,
                    "start_ms": 2_000,
                    "end_ms": 2_600,
                    "duration_ms": 600,
                    "word_count": 2,
                },
            }),
        },
    );
    primary.metadata.duration_ms = Some(600);
    primary.metadata.confidence = Some(0.78);

    let descendants = AudioDescendantExtractor.extract(&primary).unwrap();

    let possible = descendants
        .iter()
        .find(|sensation| sensation.kind == "audio.possible_speech")
        .expect("possible speech sensation");
    assert_eq!(
        possible.payload.get("commitment").and_then(Value::as_str),
        Some("possible")
    );
    assert_eq!(
        possible.payload.get("text").and_then(Value::as_str),
        Some("turn le")
    );
    assert!(possible
        .summary
        .as_deref()
        .is_some_and(|summary| summary.contains("may be hearing")));

    let committed = descendants
        .iter()
        .find(|sensation| sensation.kind == "audio.committed_speech")
        .expect("committed speech sensation");
    assert_eq!(
        committed.payload.get("commitment").and_then(Value::as_str),
        Some("committed")
    );
    assert_eq!(
        committed.payload.get("text").and_then(Value::as_str),
        Some("turn left")
    );
    assert!(committed
        .summary
        .as_deref()
        .is_some_and(|summary| summary.contains("commit")));
}

#[test]
fn audio_extractor_falls_back_to_deterministic_voice_windows() {
    let mut primary = Sensation::primary(
        Modality::Audio,
        SensationSource::new("test-ear"),
        2_000,
        4_600,
        SensationPayload {
            kind: SensationPayloadKind::AudioPcm,
            value: json!({
                "feature_sets": 130,
                "transcript": null,
                "asr": {},
            }),
        },
    );
    primary.metadata.duration_ms = Some(2_600);
    primary.metadata.confidence = Some(0.35);

    let descendants = AudioDescendantExtractor.extract(&primary).unwrap();

    assert_eq!(descendants.len(), 3);
    assert!(descendants
        .iter()
        .all(|sensation| sensation.payload_kind == SensationPayloadKind::VoiceSegment));
    assert!(descendants
        .iter()
        .all(|sensation| sensation.parent_id == Some(primary.id)));
    assert_eq!(
        descendants[0].payload.get("method").and_then(Value::as_str),
        Some("deterministic_audio_features")
    );
    assert_eq!(
        descendants[0].summary.as_deref(),
        Some("I hear a voice nearby.")
    );
}

fn visual_primary_with_rgb(width: u32, height: u32, rgb: Vec<u8>) -> Sensation {
    let mut payload = SensationPayload::image_metadata(width, height, "rgb8", rgb.len());
    payload.value["raw_bytes_b64"] =
        Value::String(base64::engine::general_purpose::STANDARD.encode(rgb));
    Sensation::primary(
        Modality::Vision,
        SensationSource::new("test-camera"),
        100,
        110,
        payload,
    )
}

#[test]
fn visual_detector_creates_face_crop_with_bbox_metadata() {
    let mut rgb = vec![8_u8; 64 * 48 * 3];
    for y in 12..32 {
        for x in 22..42 {
            let idx = (y * 64 + x) * 3;
            rgb[idx] = 225;
            rgb[idx + 1] = 168;
            rgb[idx + 2] = 115;
        }
    }
    let primary = visual_primary_with_rgb(64, 48, rgb);

    let descendants = VisualDescendantExtractor.extract(&primary).unwrap();

    assert_eq!(descendants.len(), 1);
    let crop = &descendants[0];
    assert_eq!(crop.parent_id, Some(primary.id));
    assert_eq!(crop.modality, Modality::Vision);
    assert_eq!(crop.payload_kind, SensationPayloadKind::Crop);
    assert_eq!(crop.kind, "vision.face_crop");
    assert_eq!(crop.metadata.bbox.unwrap().x, 22);
    assert!(crop.metadata.confidence.unwrap() > 0.4);
    assert!(crop.metadata.labels.contains(&"face".to_string()));
    assert_eq!(
        crop.metadata
            .properties
            .get("detection_kind")
            .and_then(Value::as_str),
        Some("face")
    );
    assert_eq!(
        crop.provenance.stage_chain,
        vec!["descendant.face_crop".to_string()]
    );
    assert!(crop.payload.get("raw_bytes_b64").is_some());
    assert!(crop.payload.get("crop_content_id").is_some());
}

#[test]
fn visual_extractor_falls_back_to_center_crop_without_detector_output() {
    let primary = Sensation::primary(
        Modality::Vision,
        SensationSource::new("test-camera"),
        100,
        100,
        SensationPayload::image_metadata(64, 48, "rgb8", 64 * 48 * 3),
    );

    let descendants = VisualDescendantExtractor.extract(&primary).unwrap();

    assert_eq!(descendants.len(), 1);
    assert_eq!(descendants[0].kind, "vision.crop");
    assert_eq!(descendants[0].parent_id, Some(primary.id));
    assert_eq!(
        descendants[0].payload.get("method").and_then(Value::as_str),
        Some("deterministic_center_crop")
    );
    assert_eq!(descendants[0].metadata.bbox.unwrap().x, 16);
}

#[tokio::test]
async fn embodied_pipeline_vectorizes_visual_crop_and_impression_text() {
    let mut rgb = vec![5_u8; 64 * 48 * 3];
    for y in 10..34 {
        for x in 20..44 {
            let idx = (y * 64 + x) * 3;
            rgb[idx] = 230;
            rgb[idx + 1] = 172;
            rgb[idx + 2] = 120;
        }
    }
    let primary = visual_primary_with_rgb(64, 48, rgb);

    let batch = EmbodiedPipeline::new()
        .ingest_primary(primary)
        .await
        .unwrap();

    let crop = batch
        .sensations
        .iter()
        .find(|sensation| sensation.kind == "vision.face_crop")
        .expect("face crop sensation");
    assert!(crop.vector.is_some(), "crop should be vectorized");
    assert_eq!(
        crop.vector.as_ref().map(|vector| vector.model_id.as_str()),
        Some("pete.image.frame_stats.v1")
    );
    assert_eq!(
        crop.vector.as_ref().map(|vector| vector.dim),
        Some(EMBODIED_FEATURE_VECTOR_DIM)
    );
    assert!(batch
        .impressions
        .iter()
        .any(|impression| impression.sensation_id == Some(crop.id)
            && impression.text == "I see a face close to me."));
}

#[tokio::test]
async fn vectorizer_registry_uses_configured_placeholder_fallback() {
    let mut config = EmbodiedVectorizerRegistryConfig::default();
    config.vectorizer.insert(
        "vision_crop".to_string(),
        EmbodiedVectorizerConfig {
            enabled: false,
            model: None,
            model_label: None,
            model_path: None,
            purpose: None,
            collection: None,
            fallback: Some("placeholder".to_string()),
        },
    );
    let registry = SensationVectorizerRegistry::from_config(&config);
    let primary = visual_primary_with_rgb(16, 16, vec![9; 16 * 16 * 3]);
    let crop = Sensation::descendant(
        &primary,
        "vision.crop",
        SensationPayloadKind::Crop,
        json!({"width": 8, "height": 8}),
        SensationMetadata::default(),
        "test",
    );

    let vector = registry.vectorize(&crop).await.unwrap().expect("vector");

    assert_eq!(vector.model_id, "pete.placeholder.v0");
    assert_eq!(vector.dim, PLACEHOLDER_VECTOR_DIM);
    assert_eq!(vector.source_sensation_id, crop.id);
    assert_eq!(vector.payload_kind, SensationPayloadKind::Crop);
    assert_eq!(vector.vectorizer_id, "pete.vectorizer.placeholder.v0");
    assert_eq!(vector.purpose, "face_identity");
    assert_eq!(vector.collection, "fallback_vectors");
    assert!(vector.is_fallback);
    assert!(vector.input_summary.contains("kind=vision.crop"));
}

#[tokio::test]
async fn vectorizer_registry_selects_configured_real_vectorizer_metadata() {
    let mut config = EmbodiedVectorizerRegistryConfig::default();
    config.vectorizer.insert(
        "vision_image".to_string(),
        EmbodiedVectorizerConfig {
            enabled: true,
            model: Some("pete.test.real_scene.v1".to_string()),
            model_label: Some("Test Scene Vectorizer".to_string()),
            model_path: None,
            purpose: Some("scene_similarity".to_string()),
            collection: Some("scene_vectors".to_string()),
            fallback: Some("placeholder".to_string()),
        },
    );
    let registry = SensationVectorizerRegistry::from_config(&config);
    let primary = visual_primary_with_rgb(16, 16, vec![9; 16 * 16 * 3]);

    let vector = registry.vectorize(&primary).await.unwrap().expect("vector");

    assert_eq!(vector.model_id, "pete.test.real_scene.v1");
    assert_eq!(vector.model_label, "Test Scene Vectorizer");
    assert_eq!(
        vector.vectorizer_id,
        "pete.vectorizer.vision_image.frame_stats.v1"
    );
    assert_eq!(vector.dim, EMBODIED_FEATURE_VECTOR_DIM);
    assert_eq!(vector.source_sensation_id, primary.id);
    assert_eq!(vector.purpose, "scene_similarity");
    assert_eq!(vector.collection, "scene_vectors");
    assert!(!vector.is_fallback);
    assert!(vector.input_summary.contains("size=16x16"));
}

#[tokio::test]
async fn missing_configured_model_path_falls_back_to_placeholder() {
    let mut config = EmbodiedVectorizerRegistryConfig::default();
    config.vectorizer.insert(
        "vision_image".to_string(),
        EmbodiedVectorizerConfig {
            enabled: true,
            model: Some("pete.clip.missing.v1".to_string()),
            model_label: None,
            model_path: Some("data/models/definitely-missing-openclip.onnx".to_string()),
            purpose: Some("scene_similarity".to_string()),
            collection: Some("scene_vectors".to_string()),
            fallback: Some("placeholder".to_string()),
        },
    );
    let registry = SensationVectorizerRegistry::from_config(&config);
    let primary = visual_primary_with_rgb(16, 16, vec![11; 16 * 16 * 3]);

    let vector = registry.vectorize(&primary).await.unwrap().expect("vector");

    assert_eq!(vector.model_id, "pete.placeholder.v0");
    assert_eq!(vector.dim, PLACEHOLDER_VECTOR_DIM);
    assert!(vector.is_fallback);
}

#[tokio::test]
async fn vectorizer_registry_preserves_precomputed_model_metadata() {
    let registry = SensationVectorizerRegistry::with_defaults();
    let mut now = Now::blank(310, BodySense::default());
    now.face.vectors.push(
        pete_now::VectorArtifact::new("faces", "face-vector-1", vec![0.2, 0.4, 0.6])
            .with_model("arcface.test.v0")
            .with_source_id("face-1")
            .with_occurred_at_ms(300),
    );
    let face = primary_sensations_from_now(&now)
        .into_iter()
        .find(|sensation| sensation.source == "face.features")
        .expect("face feature sensation");

    let vector = registry.vectorize(&face).await.unwrap().expect("vector");

    assert_eq!(vector.model_id, "arcface.test.v0");
    assert_eq!(vector.dim, 3);
    assert_eq!(vector.vector, vec![0.2, 0.4, 0.6]);
    assert_eq!(vector.source_sensation_id, face.id);
    assert_eq!(vector.generated_at_ms, face.observed_at_ms);
    assert_eq!(vector.vectorizer_id, "precomputed.faces.arcface.test.v0");
    assert_eq!(vector.purpose, "face_identity");
    assert_eq!(vector.collection, "faces");
    assert!(!vector.is_fallback);
    assert!(vector.input_summary.contains("face-vector-1"));
}

#[tokio::test]
async fn voice_precomputed_vectors_are_preserved_with_voice_identity_metadata() {
    let registry = SensationVectorizerRegistry::with_defaults();
    let mut now = Now::blank(410, BodySense::default());
    now.voice.vectors.push(
        pete_now::VectorArtifact::new("voices", "voice-vector-1", vec![0.1, 0.3, 0.5, 0.7])
            .with_model("pete.voice.test.v0")
            .with_source_id("voice-1")
            .with_occurred_at_ms(405),
    );
    let voice = primary_sensations_from_now(&now)
        .into_iter()
        .find(|sensation| sensation.source == "voice.features")
        .expect("voice feature sensation");

    let vector = registry.vectorize(&voice).await.unwrap().expect("vector");

    assert_eq!(vector.model_id, "pete.voice.test.v0");
    assert_eq!(vector.dim, 4);
    assert_eq!(vector.vector, vec![0.1, 0.3, 0.5, 0.7]);
    assert_eq!(vector.source_sensation_id, voice.id);
    assert_eq!(
        vector.vectorizer_id,
        "precomputed.voices.pete.voice.test.v0"
    );
    assert_eq!(vector.purpose, "voice_identity");
    assert_eq!(vector.collection, "voices");
    assert!(!vector.is_fallback);
    assert!(vector.input_summary.contains("voice-vector-1"));
}

#[tokio::test]
async fn embodied_demo_reports_multimodal_vector_coverage() {
    let demo = demo_embodied_experience(1_000).await.unwrap();

    assert!(demo.coverage.image > 0);
    assert!(demo.coverage.face > 0);
    assert!(demo.coverage.voice > 0);
    assert!(demo.coverage.transcript > 0);
    assert!(demo.coverage.impression > 0);
    assert!(demo.coverage.experience > 0);
    assert!(demo.coverage.fallback_count > 0);
    assert!(demo
        .sensations
        .iter()
        .filter_map(|sensation| sensation.vector.as_ref())
        .any(
            |vector| vector.vectorizer_id == "precomputed.faces.face_id/0.4.1"
                && vector.collection == "faces"
                && vector.purpose == "face_identity"
        ));
    assert!(demo
        .sensations
        .iter()
        .filter_map(|sensation| sensation.vector.as_ref())
        .any(
            |vector| vector.vectorizer_id == "precomputed.voices.pete/voice_vector/16d"
                && vector.collection == "voices"
                && vector.purpose == "voice_identity"
        ));
}

#[tokio::test]
async fn duplicate_image_frames_do_not_repeat_embeddings() {
    let registry = SensationVectorizerRegistry::with_defaults();
    let first = visual_primary_with_rgb(16, 16, vec![13; 16 * 16 * 3]);
    let second = visual_primary_with_rgb(16, 16, vec![13; 16 * 16 * 3]);

    let first_vector = registry.vectorize(&first).await.unwrap();
    let second_vector = registry.vectorize(&second).await.unwrap();

    assert!(first_vector.is_some());
    assert!(second_vector.is_none());
}

#[test]
fn primary_sensation_from_now_preserves_raw_visual_bytes() {
    let mut now = Now::blank(200, BodySense::default());
    now.eye_frame = Some(pete_now::EyeFrame {
        captured_at_ms: 190,
        width: 2,
        height: 2,
        format: pete_now::EyeFrameFormat::Rgb8,
        bytes: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        source: Some("unit-camera".to_string()),
    });

    let sensations = primary_sensations_from_now(&now);

    let vision = sensations
        .iter()
        .find(|sensation| sensation.payload_kind == SensationPayloadKind::ImageBytes)
        .expect("vision primary");
    assert_eq!(
        vision
            .metadata
            .properties
            .get("raw_bytes_present")
            .and_then(Value::as_bool),
        Some(true)
    );
    let encoded = vision
        .payload
        .get("raw_bytes_b64")
        .and_then(Value::as_str)
        .expect("raw bytes payload");
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .unwrap();
    assert_eq!(decoded, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
}

#[test]
fn experience_fuser_links_sensations_impressions_and_summary() {
    let mut sensation = Sensation::primary(
        Modality::Vision,
        SensationSource::new("test-camera"),
        100,
        110,
        SensationPayload::image_metadata(64, 48, "rgb8", 64 * 48 * 3),
    );
    sensation.vector = Some(VectorEmbedding::new(
        vec![0.1, 0.2, 0.3],
        "test-vectorizer",
        Modality::Vision,
        SensationPayloadKind::ImageBytes,
        sensation.id,
        110,
    ));
    let impression = TemplateImpressionGenerator.generate_for_sensation(&sensation);

    let experience = ExperienceFuser::new(750)
        .fuse(&[sensation.clone()], &[impression.clone()])
        .unwrap();

    assert_eq!(experience.sensation_ids, vec![sensation.id]);
    assert_eq!(experience.impression_ids, vec![impression.id]);
    assert_eq!(experience.window_start_ms, 100);
    assert_eq!(experience.window_end_ms, 110);
    let impression_vector = impression.vector.as_ref().expect("impression vector");
    assert_eq!(impression_vector.model_id, "pete.text.hashing.v1");
    assert_eq!(impression_vector.purpose, "impression_semantic");
    assert_eq!(impression_vector.collection, "impressions");
    assert_eq!(impression_vector.source_kind, "impression");
    assert!(!impression_vector.is_fallback);
    assert_eq!(
        experience
            .summary_impression
            .as_ref()
            .and_then(|summary| summary.experience_id),
        Some(experience.id)
    );
    let summary_vector = experience
        .summary_impression
        .as_ref()
        .and_then(|summary| summary.vector.as_ref())
        .expect("experience summary semantic vector");
    assert_eq!(summary_vector.model_id, "pete.text.hashing.v1");
    assert_eq!(summary_vector.purpose, "experience_semantic");
    assert_eq!(summary_vector.collection, "experiences");
    assert_eq!(summary_vector.source_kind, "experience");
    assert_eq!(summary_vector.source_sensation_id, experience.id);
    assert!(!summary_vector.is_fallback);
    assert!(experience.text.starts_with("I see"));
}

#[test]
fn primary_sensations_from_now_lifts_live_sensor_surfaces() {
    let mut now = Now::blank(200, BodySense::default());
    now.eye_frame = Some(pete_now::EyeFrame {
        captured_at_ms: 190,
        width: 32,
        height: 24,
        format: pete_now::EyeFrameFormat::Rgb8,
        bytes: vec![0; 32 * 24 * 3],
        source: Some("unit-camera".to_string()),
    });
    now.ear.asr.transcript = Some("hello".to_string());
    now.ear.asr.confidence = 0.8;
    now.range.nearest_m = Some(0.4);
    now.kinect.depth_m = vec![1.0, 1.2, 1.4, 1.6];
    now.kinect.depth_width = 2;
    now.kinect.depth_height = 2;

    let sensations = primary_sensations_from_now(&now);

    assert!(sensations
        .iter()
        .any(|sensation| sensation.payload_kind == SensationPayloadKind::ImageBytes));
    assert!(sensations
        .iter()
        .any(|sensation| sensation.payload_kind == SensationPayloadKind::AudioPcm));
    assert!(sensations
        .iter()
        .any(|sensation| sensation.payload_kind == SensationPayloadKind::LidarScan));
    assert!(sensations
        .iter()
        .any(|sensation| sensation.payload_kind == SensationPayloadKind::DepthFrame));
}

#[tokio::test]
async fn embodied_now_vectorizes_asr_audio_descendants() {
    let mut now = Now::blank(200, BodySense::default());
    now.ear.asr = AsrSense {
        transcript: Some("come closer".to_string()),
        is_final: true,
        confidence: 0.77,
        start_ms: Some(120),
        end_ms: Some(920),
        duration_ms: Some(800),
        word_count: Some(2),
        ..AsrSense::default()
    };

    let embodied = embody_now(&now).await.unwrap();

    let speech = embodied
        .sensations
        .iter()
        .find(|sensation| sensation.payload_kind == SensationPayloadKind::SpeechSegment)
        .expect("speech child sensation");
    assert!(speech.parent_id.is_some());
    assert!(speech.vector.is_some());
    assert_eq!(
        speech
            .vector
            .as_ref()
            .map(|vector| vector.model_id.as_str()),
        Some("pete.text.hashing.v1")
    );
    assert_eq!(
        speech.vector.as_ref().map(|vector| vector.purpose.as_str()),
        Some("transcript_semantic")
    );
    assert!(!speech.vector.as_ref().unwrap().is_fallback);
    assert_eq!(
        speech
            .impression
            .as_ref()
            .map(|impression| impression.text.as_str()),
        Some("I hear someone say \"come closer\".")
    );
    let speech_impression_vector = speech
        .impression
        .as_ref()
        .and_then(|impression| impression.vector.as_ref())
        .expect("speech impression semantic vector");
    assert_eq!(speech_impression_vector.model_id, "pete.text.hashing.v1");
    assert_eq!(speech_impression_vector.purpose, "impression_semantic");
    assert_eq!(speech_impression_vector.collection, "impressions");
    assert!(!speech_impression_vector.is_fallback);
    assert!(embodied
        .sensations
        .iter()
        .any(
            |sensation| sensation.payload_kind == SensationPayloadKind::TranscriptSpan
                && sensation.vector.is_some()
        ));
}

#[tokio::test]
async fn embodied_now_preserves_precomputed_asr_transcript_vector() {
    let mut now = Now::blank(300, BodySense::default());
    now.ear.transcript = Some("open the pod bay doors".to_string());
    now.ear.asr = AsrSense {
        transcript: now.ear.transcript.clone(),
        is_final: true,
        confidence: 0.88,
        start_ms: Some(240),
        end_ms: Some(300),
        duration_ms: Some(60),
        word_count: Some(5),
        ..AsrSense::default()
    };
    now.ear.transcript_vectors.push(
        pete_now::VectorArtifact::new(
            "transcripts",
            "asr-utterance-7-transcript",
            vec![0.25, 0.5, 0.75],
        )
        .with_model("pete.text.hashing.v1")
        .with_source_id("asr-utterance-7")
        .with_occurred_at_ms(300),
    );

    let embodied = embody_now(&now).await.unwrap();

    let transcript = embodied
        .sensations
        .iter()
        .find(|sensation| {
            sensation.source == "ear.transcript_vectors"
                && sensation.payload_kind == SensationPayloadKind::TranscriptSpan
        })
        .expect("precomputed transcript vector sensation");
    let vector = transcript
        .vector
        .as_ref()
        .expect("preserved transcript vector");
    assert_eq!(vector.vector, vec![0.25, 0.5, 0.75]);
    assert_eq!(vector.model_id, "pete.text.hashing.v1");
    assert_eq!(
        vector.vectorizer_id,
        "precomputed.transcripts.pete.text.hashing.v1"
    );
    assert_eq!(vector.purpose, "transcript_semantic");
    assert_eq!(vector.collection, "transcripts");
    assert!(!vector.is_fallback);
}

#[test]
fn embodied_context_from_current_experience_uses_traceable_sensation_lineage() {
    let primary = Sensation::primary(
        Modality::Vision,
        SensationSource::new("unit-camera"),
        100,
        105,
        SensationPayload::image_metadata(32, 24, "rgb8", 32 * 24 * 3),
    )
    .with_summary("I receive a visual frame.");
    let child = Sensation::descendant(
        &primary,
        "vision.crop.focus",
        SensationPayloadKind::Crop,
        json!({"x": 4, "y": 3, "width": 12, "height": 9}),
        SensationMetadata::default(),
        "focus",
    )
    .with_summary("I focus on a patch in the frame.")
    .with_vector(VectorEmbedding::new(
        vec![0.1, 0.2, 0.3],
        "unit.crop.v0",
        Modality::Vision,
        SensationPayloadKind::Crop,
        primary.id,
        106,
    ));
    let impression = Impression::new(
        "vision.focus.impression",
        "I see a frame and focus on part of it.",
        vec![primary.id, child.id],
        100,
        106,
    );
    let mut experience = Experience::new(
        "embodied.now",
        "I see a frame and focus on part of it.",
        vec![impression.id],
        vec![primary.id, child.id],
        100,
        106,
    );
    experience.predictions.push(Prediction {
        offset_ms: 750,
        text: "I expect the focused view to remain similar.".to_string(),
        confidence: 0.4,
        vector: None,
    });
    experience.memory_links.push(MemoryLink {
        target_id: "memory-1".to_string(),
        relation: "similar".to_string(),
        score: 0.7,
        payload: json!({"text": "A previous focused camera moment."}),
    });

    let context = EmbodiedContext::from_current_experience(
        Some(&experience),
        &[primary.clone(), child.clone()],
        &[impression],
        &[],
        &[],
    );

    assert_eq!(context.experience_id, Some(experience.id));
    assert_eq!(context.summary, experience.text);
    assert_eq!(context.sensations.len(), 2);
    assert_eq!(context.derived_sensation_count(), 1);
    assert_eq!(
        context.lineage,
        vec![EmbodiedLineageEdge {
            parent_id: primary.id,
            child_id: child.id,
        }]
    );
    assert_eq!(
        context
            .sensation_vectors
            .iter()
            .map(|vector| (vector.model_id.as_str(), vector.dim))
            .collect::<Vec<_>>(),
        vec![("unit.crop.v0", 3)]
    );
    assert_eq!(context.predictions.len(), 1);
    assert_eq!(context.memory_links.len(), 1);
}

#[test]
fn recalled_experience_becomes_memory_recall_sensation_with_impression() {
    let source_sensation_id = Uuid::new_v4();
    let original_frame_id = Uuid::new_v4();
    let experience = Experience::new(
        "embodied.now",
        "I see a charger by the wall.",
        Vec::new(),
        vec![source_sensation_id],
        1_000,
        1_100,
    );
    let sensation = experience.to_recall_sensation_with_lineage(
        2_000,
        0.82,
        "unit-recall",
        Some(original_frame_id),
        vec!["experiences:vector-1".to_string()],
    );
    let impression = experience.to_recall_impression(&sensation, 0.82);

    assert_eq!(sensation.modality, Modality::Memory);
    assert_eq!(sensation.payload_kind, SensationPayloadKind::MemoryRecall);
    assert!(matches!(
        sensation.provenance.kind,
        pete_core::ProvenanceKind::MemoryRecall { experience_id }
            if experience_id == experience.id
    ));
    assert_eq!(
        sensation
            .payload
            .get("original_frame_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        Some(original_frame_id.to_string())
    );
    assert_eq!(
        sensation
            .payload
            .get("original_vector_ids")
            .and_then(Value::as_array)
            .and_then(|values| values.first())
            .and_then(Value::as_str),
        Some("experiences:vector-1")
    );
    assert!(sensation.vector.is_none());
    assert!(impression.text.starts_with("I remember"));
    assert!(impression.text.contains("near here"));
    assert_eq!(impression.sensation_id, Some(sensation.id));
}
