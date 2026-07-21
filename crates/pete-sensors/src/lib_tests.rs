use super::*;
use std::io::Write;
use std::sync::Arc;

struct StaticFaceDetector;

fn trustworthy_imu(source: &str, captured_at_ms: u64, source_epoch: u64) -> ImuSense {
    ImuSense {
        schema_version: 2,
        captured_at_ms,
        orientation: vec![0.1, -0.1],
        acceleration: vec![0.0, 0.0, 1.0],
        angular_velocity: vec![0.0, 0.0, 0.0],
        temperature_c: None,
        orientation_confidence: 0.9,
        gyro_bias_calibrated: true,
        mounting_calibrated: true,
        calibration: None,
        orientation_source: Some(format!("{source}@{source_epoch}:accel_gyro_roll_pitch")),
    }
}

fn candidate_metadata(source: &str, source_epoch: u64, healthy: bool) -> ImuCandidateMetadata {
    ImuCandidateMetadata {
        source_id: source.to_string(),
        provenance: source.to_string(),
        healthy,
        clock_confidence: 0.9,
        clock_source: Some("test_host_clock".to_string()),
        source_epoch,
        reported_sample_age_ms: Some(5),
        supported_axes: vec![
            "roll".to_string(),
            "pitch".to_string(),
            "gyro_xyz".to_string(),
            "accel_xyz".to_string(),
        ],
    }
}

#[test]
fn imu_auto_selects_only_trustworthy_brainstem_and_none_without_candidates() {
    let mut arbiter = ImuArbiter::default();
    assert!(arbiter.select(1_000).selected_source.is_none());
    arbiter.observe(trustworthy_imu("brainstem_board_imu", 990, 0), 1_000);
    arbiter.observe(trustworthy_imu("brainstem_board_imu", 995, 0), 1_000);
    let selection = arbiter.select(1_000);
    assert_eq!(
        selection.selected_source.as_deref(),
        Some("brainstem_board_imu")
    );
    assert_eq!(
        selection.selected.as_ref().and_then(ImuSense::source_id),
        Some("brainstem_board_imu")
    );
}

#[test]
fn imu_auto_discovers_uncalibrated_brainstem_then_selects_after_calibration() {
    let mut arbiter = ImuArbiter::default();
    let mut uncalibrated = trustworthy_imu("brainstem_board_imu", 990, 0);
    uncalibrated.gyro_bias_calibrated = false;
    arbiter.observe(uncalibrated, 1_000);
    let rejected = arbiter.select(1_000);
    assert!(rejected.selected_source.is_none());
    assert!(rejected.diagnostics["candidates"][0]["rejection_reason"]
        .as_str()
        .is_some_and(|reason| reason.contains("gyro bias")));

    let mut mounting_unknown = trustworthy_imu("brainstem_board_imu", 995, 0);
    mounting_unknown.mounting_calibrated = false;
    arbiter.observe(mounting_unknown, 1_000);
    let rejected = arbiter.select(1_000);
    assert!(rejected.selected_source.is_none());
    assert!(rejected.diagnostics["candidates"][0]["rejection_reason"]
        .as_str()
        .is_some_and(|reason| reason.contains("mounting")));

    arbiter.observe(trustworthy_imu("brainstem_board_imu", 1_005, 0), 1_010);
    arbiter.observe(trustworthy_imu("brainstem_board_imu", 1_015, 0), 1_020);
    assert_eq!(
        arbiter.select(1_020).selected_source.as_deref(),
        Some("brainstem_board_imu")
    );
}

#[test]
fn schema3_imu_requires_trusted_adaptive_calibration() {
    let mut missing = trustworthy_imu("local_i2c_mpu6050", 990, 0);
    missing.schema_version = 3;
    let mut arbiter = ImuArbiter::default();
    arbiter.observe(missing, 1_000);
    let rejected = arbiter.select(1_000);
    assert!(rejected.selected_source.is_none());
    assert!(rejected.diagnostics["candidates"][0]["rejection_reason"]
        .as_str()
        .is_some_and(|reason| reason.contains("state is missing")));

    let mut estimator = pete_now::ImuCalibrationEstimator::new(
        pete_now::RigidTransform3::default(),
        false,
        0,
        pete_now::ImuCalibrationConfig::default(),
    );
    for index in 0..60 {
        estimator.observe(
            [0.0, 0.0, 1.0],
            [0.0; 3],
            None,
            pete_now::ImuMotionContext::default(),
            index * 10,
        );
    }
    for timestamp in [1_005, 1_010] {
        let mut trusted = trustworthy_imu("local_i2c_mpu6050", timestamp, 0);
        trusted.schema_version = 3;
        trusted.calibration = Some(estimator.state().clone());
        arbiter.observe(trusted, timestamp);
    }
    assert_eq!(
        arbiter.select(1_010).selected_source.as_deref(),
        Some("local_i2c_mpu6050")
    );
}

#[test]
fn now_builder_publishes_replayable_latency_epochs_and_correlated_rotation() {
    let mut builder = NowBuilder::new();
    let mut final_now = None;
    for index in 0..16_u64 {
        let t_ms = 10_000 + index * 100;
        let turning = index % 2 == 0;
        let mut body = BodySense {
            last_update_ms: t_ms - 30,
            ..BodySense::default()
        };
        body.velocity.turn_rad_s = if turning { 0.5 } else { 0.0 };
        let imu = ImuSense {
            schema_version: 2,
            captured_at_ms: t_ms - 10,
            angular_velocity: vec![0.0, 0.0, if turning { 0.5 } else { 0.0 }],
            orientation_source: Some("test_imu@0:synthetic".to_string()),
            ..ImuSense::default()
        };
        final_now = Some(
            builder
                .build(t_ms, body, vec![SensePacket::Imu(imu)])
                .unwrap(),
        );
    }
    let states: BTreeMap<String, pete_now::StreamLatencyCalibration> =
        serde_json::from_value(final_now.unwrap().extensions["sensor.latency_calibration"].clone())
            .unwrap();
    assert_eq!(
        states["body"].trust_state,
        pete_now::LatencyTrustState::Trusted
    );
    assert_eq!(
        states["imu"].trust_state,
        pete_now::LatencyTrustState::Trusted
    );
    assert_eq!(states["imu"].correlated_event_count, 8);
    assert_eq!(
        states["imu"].correlated_offset.as_ref().unwrap().median_ms,
        20.0
    );
    assert!(states["lidar"].optional);
    assert!(!states["lidar"].enabled);
}

#[test]
fn now_builder_captures_advisory_locomotion_calibration_without_applying_it() {
    let mut builder = NowBuilder::new();
    let conditions = pete_now::LocomotionConditions {
        surface: Some("synthetic_floor".to_string()),
        tire_condition: Some("nominal".to_string()),
    };
    for index in 0..5_u64 {
        assert!(builder.observe_straight_calibration_episode(
            pete_now::StraightCalibrationEpisode {
                captured_at_ms: 1_000 + index,
                direction: pete_now::TravelDirection::Forward,
                reported_distance_m: 2.0,
                actual_distance_m: 2.04,
                lateral_drift_m: 0.01,
                endpoint_heading_error_rad: 0.01,
                environmental_alignment_residual_m: 0.02,
                confidence: 0.95,
                repeated_traversal: true,
                loop_supported: true,
                conditions: conditions.clone(),
            }
        ));
        assert!(builder.observe_rotation_calibration_episode(
            pete_now::RotationCalibrationEpisode {
                captured_at_ms: 2_000 + index,
                direction: pete_now::RotationDirection::Clockwise,
                commanded_angle_rad: std::f32::consts::PI,
                wheel_odometry_angle_rad: std::f32::consts::PI,
                imu_angle_rad: Some(std::f32::consts::PI * 1.02),
                imu_trusted: true,
                environmental_angle_rad: Some(std::f32::consts::PI * 1.02),
                environmental_alignment_residual_m: 0.02,
                loop_angle_rad: Some(std::f32::consts::PI * 1.02),
                axle_center_displacement_m: 0.01,
                confidence: 0.95,
                conditions: conditions.clone(),
            }
        ));
    }
    let now = builder
        .build(3_000, BodySense::default(), Vec::new())
        .unwrap();
    let state: pete_now::LocomotionCalibrationState =
        serde_json::from_value(now.extensions["calibration.locomotion"].clone()).unwrap();
    assert_eq!(
        state.trust_state,
        pete_now::LocomotionCalibrationTrustState::Trusted
    );
    assert!(state.authority.contains("brainstem"));
    assert_eq!(builder.snapshot().locomotion_calibration.epoch, state.epoch);
}

#[test]
fn now_builder_exposes_power_evidence_in_dashboard_and_capture_snapshot() {
    let mut builder = NowBuilder::new();
    let assessment = serde_json::json!({
        "consolidation_ready": false,
        "action": "pause_external_power_lost",
        "external_power_present": false,
        "battery_current_a": null,
        "battery_current_observable": false,
        "ages": {"external_power_ms": 12},
    });
    builder.set_power_assessment(Some(assessment.clone()));
    let now = builder
        .build(1_000, BodySense::default(), Vec::new())
        .unwrap();
    assert_eq!(now.extensions["power.consolidation_readiness"], assessment);
    assert_eq!(builder.snapshot().power_assessment, Some(assessment));
}

#[test]
fn imu_auto_rejects_stale_unhealthy_future_and_untrusted_override() {
    let mut arbiter = ImuArbiter::default();
    arbiter.observe(trustworthy_imu("brainstem_board_imu", 990, 0), 1_000);
    arbiter.observe(trustworthy_imu("brainstem_board_imu", 995, 0), 1_000);
    assert!(arbiter.select(1_000).selected.is_some());
    let stale = arbiter.select(1_300);
    assert!(stale.selected_source.is_none());
    assert!(stale.selected.is_none());

    let unhealthy = trustworthy_imu("brainstem_board_imu", 1_305, 0);
    arbiter.observe_with_metadata(
        unhealthy,
        candidate_metadata("brainstem_board_imu", 0, false),
        1_310,
    );
    assert!(arbiter.select(1_310).selected_source.is_none());

    let mut future = trustworthy_imu("brainstem_board_imu", 1_500, 0);
    future.mounting_calibrated = false;
    let mut forced = ImuArbiter::new(ImuSourceOverride::Force("brainstem_board_imu".to_string()));
    forced.observe(future, 1_310);
    assert!(forced.select(1_310).selected_source.is_none());
}

#[test]
fn imu_auto_prefers_equivalent_brainstem_and_hysteresis_prevents_flapping() {
    let mut arbiter = ImuArbiter::default();
    for timestamp in [990, 995] {
        arbiter.observe(trustworthy_imu("local_i2c_mpu6050", timestamp, 0), 1_000);
        arbiter.observe(trustworthy_imu("brainstem_board_imu", timestamp, 0), 1_000);
    }
    assert_eq!(
        arbiter.select(1_000).selected_source.as_deref(),
        Some("brainstem_board_imu")
    );

    for (index, healthy) in [true, false, true, false].into_iter().enumerate() {
        let now = 1_010 + index as u64 * 10;
        let mut local = trustworthy_imu("local_i2c_mpu6050", now - 1, 0);
        local.orientation_confidence = 1.0;
        arbiter.observe_with_metadata(
            local,
            candidate_metadata("local_i2c_mpu6050", 0, healthy),
            now,
        );
        arbiter.observe(trustworthy_imu("brainstem_board_imu", now - 1, 0), now);
        assert_eq!(
            arbiter.select(now).selected_source.as_deref(),
            Some("brainstem_board_imu")
        );
    }
}

#[test]
fn imu_reconnect_changes_epoch_and_never_reuses_cross_source_history() {
    let mut arbiter = ImuArbiter::default();
    for timestamp in [990, 995] {
        arbiter.observe(trustworthy_imu("brainstem_board_imu", timestamp, 0), 1_000);
    }
    let initial = arbiter.select(1_000);
    assert!(initial.selected.is_some());
    let initial_epoch = initial.source_epoch;

    arbiter.observe(trustworthy_imu("brainstem_board_imu", 1_005, 1), 1_010);
    let rebuilding = arbiter.select(1_010);
    assert!(rebuilding.source_epoch > initial_epoch);
    assert!(rebuilding.source_changed);
    assert!(
        rebuilding.selected.is_none(),
        "new epoch needs fresh history"
    );
    for now in [1_020, 1_030] {
        arbiter.observe(trustworthy_imu("brainstem_board_imu", now - 5, 1), now);
        let selection = arbiter.select(now);
        if now < 1_030 {
            assert!(selection.selected.is_none());
        } else {
            assert!(selection.selected.is_some());
        }
    }

    arbiter.observe_unavailable("brainstem_board_imu", "disconnected", 1_040);
    let disappeared = arbiter.select(1_040);
    assert!(disappeared.selected.is_none());
    assert!(disappeared.selected_source.is_none());

    let mut disabled = ImuArbiter::new(ImuSourceOverride::Disabled);
    disabled.observe(trustworthy_imu("local_i2c_mpu6050", 1_025, 0), 1_030);
    assert!(disabled.select(1_030).selected_source.is_none());
}

#[test]
fn selected_imu_must_recover_stably_before_reselection() {
    let mut arbiter = ImuArbiter::default();
    for timestamp in [990, 995] {
        arbiter.observe(trustworthy_imu("brainstem_board_imu", timestamp, 0), 1_000);
    }
    assert!(arbiter.select(1_000).selected.is_some());

    let unhealthy = trustworthy_imu("brainstem_board_imu", 1_005, 0);
    arbiter.observe_with_metadata(
        unhealthy,
        candidate_metadata("brainstem_board_imu", 0, false),
        1_010,
    );
    assert!(arbiter.select(1_010).selected_source.is_none());

    for (index, now) in [1_020, 1_030, 1_040].into_iter().enumerate() {
        arbiter.observe(trustworthy_imu("brainstem_board_imu", now - 1, 0), now);
        let selection = arbiter.select(now);
        if index < 2 {
            assert!(selection.selected_source.is_none());
        } else {
            assert_eq!(
                selection.selected_source.as_deref(),
                Some("brainstem_board_imu")
            );
        }
    }
}

#[test]
fn stale_brainstem_cannot_overwrite_fresh_authoritative_local_history() {
    let mut arbiter = ImuArbiter::default();
    for timestamp in [990, 995] {
        let mut local = trustworthy_imu("local_i2c_mpu6050", timestamp, 0);
        local.orientation_confidence = 0.95;
        arbiter.observe(local, 1_000);
    }
    assert_eq!(
        arbiter.select(1_000).selected_source.as_deref(),
        Some("local_i2c_mpu6050")
    );

    arbiter.observe(trustworthy_imu("brainstem_board_imu", 700, 0), 1_010);
    let selection = arbiter.select(1_010);
    assert_eq!(
        selection.selected_source.as_deref(),
        Some("local_i2c_mpu6050")
    );
    assert_eq!(
        selection.selected.as_ref().and_then(ImuSense::source_id),
        Some("local_i2c_mpu6050")
    );
    let candidates = selection.diagnostics["candidates"]
        .as_array()
        .expect("candidate diagnostics");
    assert_eq!(candidates.len(), 2);
    assert!(candidates
        .iter()
        .all(|candidate| candidate["history_samples"] == 1 || candidate["history_samples"] == 2));
    assert!(candidates
        .iter()
        .all(|candidate| candidate["sample"]["captured_at_ms"].is_number()));
}

impl FaceDetector for StaticFaceDetector {
    fn detect_faces(&self, _frame: &EyeFrame) -> Result<Vec<FaceDetection>> {
        Ok(vec![FaceDetection {
            face_id: "face-static".to_string(),
            source_frame_id: None,
            embedding: vec![0.1, 0.2, 0.3],
            model: "test.face.detector".to_string(),
        }])
    }
}

struct StaticObjectDetector;

impl ObjectDetector for StaticObjectDetector {
    fn detect_objects(&self, _frame: &EyeFrame) -> Result<Vec<ObjectDetection>> {
        Ok(vec![ObjectDetection {
            object_id: "object-static".to_string(),
            label: "red cup".to_string(),
            class: ObjectClass::Landmark,
            bearing_rad: 0.25,
            distance_m: Some(1.5),
            confidence: 0.9,
            source: ObjectObservationSource::Kinect,
            source_frame_id: None,
            embedding: vec![0.7, 0.2, 0.1],
            model: "test.object.detector".to_string(),
        }])
    }
}

#[test]
fn frame_processor_vectorizes_detected_faces_into_face_collection() {
    let frame = EyeFrame {
        rgbd_frame_id: None,
        device_timestamp_ms: None,
        captured_at_ms: 42,
        width: 1,
        height: 1,
        format: EyeFrameFormat::Rgb8,
        bytes: vec![255, 128, 0],
        source: Some("unit-camera".to_string()),
    };
    let mut processor = FrameProcessor::new().with_face_detector(Arc::new(StaticFaceDetector));

    let processed = processor
        .process_frame(100, &frame)
        .expect("processed frame");

    assert_eq!(processed.face.vectors.len(), 1);
    let artifact = &processed.face.vectors[0];
    assert_eq!(artifact.collection, FACE_VECTOR_COLLECTION);
    assert_eq!(artifact.point_id, "face-static");
    assert_eq!(artifact.vector, vec![0.1, 0.2, 0.3]);
    assert_eq!(artifact.model.as_deref(), Some("test.face.detector"));
    assert_eq!(artifact.source_frame_id.as_deref(), Some("eye-42-1x1-3"));
    assert_eq!(artifact.occurred_at_ms, Some(100));
}

#[test]
fn frame_processor_vectorizes_detected_objects_into_object_collection() {
    let frame = EyeFrame {
        rgbd_frame_id: None,
        device_timestamp_ms: None,
        captured_at_ms: 42,
        width: 1,
        height: 1,
        format: EyeFrameFormat::Rgb8,
        bytes: vec![255, 128, 0],
        source: Some("unit-camera".to_string()),
    };
    let mut processor = FrameProcessor::new().with_object_detector(Arc::new(StaticObjectDetector));

    let processed = processor
        .process_frame(100, &frame)
        .expect("processed frame");

    assert_eq!(processed.objects.observations.len(), 1);
    assert_eq!(processed.objects.observations[0].label, "red cup");
    assert_eq!(processed.objects.vectors.len(), 1);
    let artifact = &processed.objects.vectors[0];
    assert_eq!(artifact.collection, OBJECT_VECTOR_COLLECTION);
    assert_eq!(artifact.point_id, "object-static");
    assert_eq!(artifact.vector, vec![0.7, 0.2, 0.1]);
    assert_eq!(artifact.model.as_deref(), Some("test.object.detector"));
    assert_eq!(artifact.source_frame_id.as_deref(), Some("eye-42-1x1-3"));
    assert_eq!(artifact.occurred_at_ms, Some(100));
}

#[test]
fn mutes_repeated_alsa_poll_descriptor_input_stream_error() {
    assert!(is_muted_cpal_input_stream_error(
            "A backend-specific error has occurred: ALSA function 'snd_pcm_poll_descriptors' failed with error 'Unknown errno (-32)'"
        ));
}

#[test]
fn keeps_other_cpal_input_stream_errors_visible() {
    assert!(!is_muted_cpal_input_stream_error(
            "A backend-specific error has occurred: ALSA function 'snd_pcm_readi' failed with error 'Input/output error'"
        ));
    assert!(!is_muted_cpal_input_stream_error(
            "A backend-specific error has occurred: ALSA function 'snd_pcm_poll_descriptors' failed with error 'Input/output error'"
        ));
}

#[tokio::test]
async fn kinect_replay_emits_kinect_then_eye_packet() {
    let root = std::env::temp_dir().join(format!("pete-kinect-replay-{}", unix_time_ms()));
    std::fs::create_dir_all(root.join("rgb")).unwrap();
    std::fs::create_dir_all(root.join("depth")).unwrap();
    std::fs::write(root.join("rgb/frame.raw"), [0u8, 128, 255]).unwrap();
    std::fs::write(root.join("depth/frame.json"), "[1.0,2.0]").unwrap();
    let mut manifest = File::create(root.join("timestamps.jsonl")).unwrap();
    writeln!(
        manifest,
        "{}",
        serde_json::json!({
            "t_ms": 1,
            "rgb_path": "rgb/frame.raw",
            "depth_path": "depth/frame.json"
        })
    )
    .unwrap();

    let mut provider = KinectReplayProvider::new(&root).unwrap();
    let first = provider.poll().await.unwrap();
    let second = provider.poll().await.unwrap();

    assert!(
        matches!(first, SensePacket::Kinect(KinectSense { depth_m, .. }) if depth_m == vec![1.0, 2.0])
    );
    assert!(matches!(second, SensePacket::Eye(EyeSense { frames, .. }) if frames.len() == 1));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn now_builder_maps_packets_and_marks_stale_sensor_ages() {
    let mut builder = NowBuilder::new();
    let first = builder
        .build(
            100,
            BodySense::default(),
            vec![
                SensePacket::Ear(EarSense {
                    transcript: Some("hello".to_string()),
                    ..EarSense::default()
                }),
                SensePacket::Range(RangeSense {
                    beams: vec![0.4],
                    nearest_m: Some(0.4),
                    ..RangeSense::default()
                }),
            ],
        )
        .unwrap();

    assert_eq!(first.ear.transcript.as_deref(), Some("hello"));
    assert_eq!(first.range.nearest_m, Some(0.4));

    let second = builder
        .build(250, BodySense::default(), Vec::new())
        .unwrap();
    assert_eq!(second.ear.transcript.as_deref(), Some("hello"));
    assert_eq!(second.range.nearest_m, Some(0.4));
    assert_eq!(
        second
            .extensions
            .get("sensor_status")
            .and_then(|status| status.get("age_ms"))
            .and_then(|age| age.get("ear"))
            .and_then(|age| age.as_u64()),
        Some(150)
    );
}

#[test]
fn now_builder_derives_range_beams_from_kinect_depth_when_range_absent() {
    let mut builder = NowBuilder::new();
    let now = builder
        .build(
            100,
            BodySense::default(),
            vec![SensePacket::Kinect(KinectSense {
                depth_m: vec![1.0, 2.0, 0.0, 3.0],
                depth_width: 4,
                depth_height: 1,
                min_depth_m: 0.2,
                max_depth_m: 4.0,
                ..KinectSense::default()
            })],
        )
        .unwrap();

    assert_eq!(now.range.beams.len(), 4);
    for (actual, expected) in now.range.beams.iter().zip([1.0, 2.0, 4.0, 3.0]) {
        assert!((actual - expected).abs() < 0.0001);
    }
    assert!(now
        .range
        .nearest_m
        .is_some_and(|nearest| (nearest - 1.0).abs() < 0.0001));
    assert_eq!(now.range.beam_angles_rad.len(), 4);
    assert_eq!(now.range.frame.as_deref(), Some("robot_base"));
    assert_eq!(now.range.source.as_deref(), Some("kinect_depth_image"));
    assert_eq!(
        now.extensions
            .get("sensor_status")
            .and_then(|status| status.get("last_update_ms"))
            .and_then(|updates| updates.get("range"))
            .and_then(|value| value.as_u64()),
        Some(100)
    );
}

#[test]
fn frame_processor_injects_calibrated_range_angles_from_kinect_depth() {
    let mut processor =
        FrameProcessor::new().with_kinect_range_projection(DepthRangeProjectionConfig {
            compact_depth_beam_count: 3,
            compact_depth_fov_rad: std::f32::consts::FRAC_PI_2,
            camera_yaw_rad: -std::f32::consts::FRAC_PI_2,
            min_depth_m: 0.1,
            max_depth_m: 3.0,
            ..DepthRangeProjectionConfig::default()
        });
    let mut packets = vec![SensePacket::Kinect(KinectSense {
        depth_m: vec![1.0, 1.0, 1.0],
        ..KinectSense::default()
    })];

    processor.process_packets(100, &mut packets);

    let range = packets
        .iter()
        .find_map(|packet| match packet {
            SensePacket::Range(range) => Some(range),
            _ => None,
        })
        .expect("calibrated range packet");
    assert_eq!(range.frame.as_deref(), Some("robot_base"));
    assert_eq!(range.source.as_deref(), Some("kinect_compact_depth"));
    assert_eq!(range.beam_angles_rad.len(), 3);
    assert!((range.beam_angles_rad[1] + std::f32::consts::FRAC_PI_2).abs() < 0.001);
}

#[test]
fn kinect_range_projection_uses_shared_metric_and_six_dof_calibration() {
    let mut builder = NowBuilder::new();
    let now = builder
        .build(
            100,
            BodySense::default(),
            vec![SensePacket::Kinect(KinectSense {
                schema_version: 1,
                captured_at_ms: 90,
                depth_m: vec![1.0],
                depth_width: 1,
                depth_height: 1,
                depth_fx: 1.0,
                depth_fy: 1.0,
                min_depth_m: 0.1,
                max_depth_m: 8.0,
                geometry_calibration: Some(pete_now::DepthGeometryCalibration {
                    depth: pete_now::CameraIntrinsics {
                        width: 1,
                        height: 1,
                        fx: 1.0,
                        fy: 1.0,
                        ..pete_now::CameraIntrinsics::default()
                    },
                    depth_scale: 2.0,
                    depth_to_base: pete_now::RigidTransform3 {
                        translation_m: [0.0, 0.4, 0.0],
                        rotation_rpy_rad: [
                            -std::f32::consts::FRAC_PI_2,
                            0.0,
                            -std::f32::consts::FRAC_PI_2,
                        ],
                    },
                    ..pete_now::DepthGeometryCalibration::default()
                }),
                ..KinectSense::default()
            })],
        )
        .unwrap();

    assert_eq!(now.range.captured_at_ms, 90);
    assert!((now.range.beams[0] - 2.0_f32.hypot(0.4)).abs() < 0.001);
    assert!((now.range.beam_angles_rad[0] - 0.4_f32.atan2(2.0)).abs() < 0.001);
}

#[test]
fn now_builder_keeps_explicit_range_over_kinect_fallback() {
    let mut builder = NowBuilder::new();
    let now = builder
        .build(
            100,
            BodySense::default(),
            vec![
                SensePacket::Range(RangeSense {
                    beams: vec![0.7],
                    nearest_m: Some(0.7),
                    ..RangeSense::default()
                }),
                SensePacket::Kinect(KinectSense {
                    depth_m: vec![1.0, 2.0, 3.0, 4.0],
                    depth_width: 4,
                    depth_height: 1,
                    ..KinectSense::default()
                }),
            ],
        )
        .unwrap();

    assert_eq!(now.range.beams, vec![0.7]);
    assert_eq!(now.range.nearest_m, Some(0.7));
}

#[test]
fn lfcd2_parser_builds_clockwise_native_segments_into_a_full_scan() {
    let extrinsics = RangeExtrinsics {
        forward_m: 0.18,
        height_m: 0.42,
        pitch_rad: 20.0_f32.to_radians(),
        yaw_rad: 0.25,
        ..RangeExtrinsics::default()
    };
    let mut parser = Lfcd2Parser::new(extrinsics);
    let mut stream = vec![0x00, 0x11, 0xfa, 0x01];
    for segment in 0..LFCD2_SEGMENTS_PER_SCAN {
        stream.extend_from_slice(&lfcd2_test_segment(segment));
    }

    assert!(parser.push(&stream[..777]).is_none());
    let scan = parser.push(&stream[777..]).expect("complete LFCD2 scan");

    assert_eq!(scan.schema_version, 1);
    assert_eq!(scan.beams.len(), LFCD2_BEAMS_PER_SCAN);
    assert_eq!(scan.beam_angles_rad.len(), LFCD2_BEAMS_PER_SCAN);
    assert_eq!(scan.frame.as_deref(), Some("hls_lfcd2"));
    assert_eq!(scan.source.as_deref(), Some("hls_lfcd2"));
    assert_eq!(scan.extrinsics, Some(extrinsics));
    assert!((scan.nearest_m.expect("nearest range") - 0.12).abs() < 0.0001);
    assert!((scan.beams[359] - 0.12).abs() < 0.0001);
    assert!((scan.beams[0] - 0.479).abs() < 0.0001);
    assert!(scan.beam_angles_rad[0].abs() < 0.0001);
    assert!((scan.beam_angles_rad[359] - 359.0_f32.to_radians()).abs() < 0.0001);
}

#[test]
fn lfcd2_parser_rejects_out_of_range_measurements() {
    let mut parser = Lfcd2Parser::new(RangeExtrinsics::default());
    let mut stream = Vec::new();
    for segment in 0..LFCD2_SEGMENTS_PER_SCAN {
        let mut packet = lfcd2_test_segment(segment);
        if segment == 0 {
            packet[6..8].copy_from_slice(&119_u16.to_le_bytes());
            packet[12..14].copy_from_slice(&3501_u16.to_le_bytes());
        }
        stream.extend_from_slice(&packet);
    }

    let scan = parser.push(&stream).expect("complete LFCD2 scan");
    assert_eq!(scan.beams[359], 0.0);
    assert_eq!(scan.beams[358], 0.0);
    assert!((scan.nearest_m.expect("nearest valid range") - 0.122).abs() < 0.0001);
}

fn lfcd2_test_segment(segment: usize) -> [u8; LFCD2_SEGMENT_BYTES] {
    let mut packet = [0u8; LFCD2_SEGMENT_BYTES];
    packet[0] = 0xfa;
    packet[1] = 0xa0 + segment as u8;
    packet[2..4].copy_from_slice(&3000_u16.to_le_bytes());
    for beam_in_segment in 0..LFCD2_BEAMS_PER_SEGMENT {
        let raw_index = segment * LFCD2_BEAMS_PER_SEGMENT + beam_in_segment;
        let offset = 4 + beam_in_segment * 6;
        packet[offset..offset + 2].copy_from_slice(&(1000_u16 + raw_index as u16).to_le_bytes());
        packet[offset + 2..offset + 4].copy_from_slice(&(120_u16 + raw_index as u16).to_le_bytes());
    }
    packet
}

#[test]
fn parses_mpu6050_device_specs() {
    assert_eq!(
        parse_mpu6050_device_spec("/dev/i2c-1").unwrap(),
        Mpu6050DeviceSpec {
            path: "/dev/i2c-1".to_string(),
            address: 0x68,
        }
    );
    assert_eq!(
        parse_mpu6050_device_spec("/dev/i2c-1@0x69").unwrap(),
        Mpu6050DeviceSpec {
            path: "/dev/i2c-1".to_string(),
            address: 0x69,
        }
    );
    assert_eq!(
        parse_mpu6050_device_spec("/dev/i2c-2:105").unwrap(),
        Mpu6050DeviceSpec {
            path: "/dev/i2c-2".to_string(),
            address: 0x69,
        }
    );
}

#[test]
fn converts_mpu6050_raw_samples_to_imu_sense() {
    let imu = mpu6050_samples_to_imu(
        [
            0x00, 0x00, // accel x = 0 g
            0x00, 0x00, // accel y = 0 g
            0x40, 0x00, // accel z = 1 g
            0x00, 0x00, // temperature, ignored
            0x00, 0x83, // gyro x = 1 deg/s => rad/s
            0xff, 0x7d, // gyro y = -1 deg/s => rad/s
            0x01, 0x06, // gyro z = 2 deg/s => rad/s
        ],
        1234,
    );

    assert_eq!(imu.schema_version, 1);
    assert_eq!(imu.captured_at_ms, 1234);
    assert_eq!(imu.acceleration, vec![0.0, 0.0, 1.0]);
    assert_eq!(imu.temperature_c, Some(36.53));
    assert!((imu.angular_velocity[0] - 1.0_f32.to_radians()).abs() < 0.0001);
    assert!((imu.angular_velocity[1] - (-1.0_f32).to_radians()).abs() < 0.0001);
    assert!((imu.angular_velocity[2] - 2.0_f32.to_radians()).abs() < 0.0001);
    assert_eq!(imu.orientation, vec![0.0, -0.0]);
}

#[test]
fn mpu6050_filter_calibrates_bias_and_uses_gyro_when_acceleration_is_untrusted() {
    let mut filter = Mpu6050OrientationFilter::new([0.0; 3], true);
    let mut filtered = ImuSense::default();
    for index in 0..MPU6050_GYRO_BIAS_SAMPLES {
        filtered = filter.update(ImuSense {
            captured_at_ms: u64::from(index) * 10,
            acceleration: vec![0.0, 0.0, 1.0],
            angular_velocity: vec![0.01, 0.0, 0.0],
            ..ImuSense::default()
        });
    }
    assert!(filtered.gyro_bias_calibrated);
    assert!(filtered.mounting_calibrated);
    assert!(filtered.orientation_confidence > 0.9);

    let moving = filter.update(ImuSense {
        captured_at_ms: 600,
        acceleration: vec![1.0, 0.0, 1.0],
        angular_velocity: vec![0.51, 0.0, 0.0],
        ..ImuSense::default()
    });
    assert!(moving.orientation[0] > 0.04);
    assert_eq!(
        moving.orientation_source.as_deref(),
        Some("local_i2c_mpu6050@0:adaptive_accel_gyro")
    );
    let calibration = moving.calibration.expect("adaptive calibration");
    assert!(calibration
        .gyro_variance
        .iter()
        .all(|value| value.is_finite()));
    assert!(calibration.temperature_c.is_none());
    assert!(!calibration.yaw_axis_observable);
}

#[test]
fn now_builder_interpolates_pose_and_imu_to_depth_exposure() {
    let mut builder = NowBuilder::new();
    let mut first_body = BodySense::default();
    first_body.last_update_ms = 100;
    builder
        .build(
            100,
            first_body,
            vec![SensePacket::Imu({
                let mut imu = trustworthy_imu("brainstem_board_imu", 100, 7);
                imu.orientation = vec![0.0, 0.0];
                imu
            })],
        )
        .unwrap();

    let mut second_body = BodySense::default();
    second_body.last_update_ms = 200;
    second_body.odometry.x_m = 1.0;
    second_body.odometry.heading_rad = std::f32::consts::FRAC_PI_2;
    let now = builder
        .build(
            200,
            second_body,
            vec![
                SensePacket::Imu({
                    let mut imu = trustworthy_imu("brainstem_board_imu", 200, 7);
                    imu.orientation = vec![0.2, -0.2];
                    imu
                }),
                SensePacket::Kinect(KinectSense {
                    schema_version: 2,
                    captured_at_ms: 150,
                    depth_m: vec![1.0],
                    depth_width: 1,
                    depth_height: 1,
                    depth_fx: 1.0,
                    depth_fy: 1.0,
                    ..KinectSense::default()
                }),
            ],
        )
        .unwrap();
    let alignment = now.kinect.fusion_alignment.expect("fusion alignment");
    assert!((alignment.pose.x_m - 0.5).abs() < 0.001);
    assert!((alignment.pose.heading_rad - std::f32::consts::FRAC_PI_4).abs() < 0.001);
    assert!((alignment.imu.orientation[0] - 0.1).abs() < 0.001);
    assert_eq!(alignment.imu.source_id(), Some("brainstem_board_imu"));
    assert_eq!(alignment.imu.source_epoch(), 7);
    assert!(alignment.imu.mounting_calibrated);
    assert!(alignment.imu.gyro_bias_calibrated);
    assert_eq!(alignment.pose_sample_skew_ms, 50);
    assert_eq!(alignment.imu_sample_skew_ms, 50);
}

#[test]
fn kinect_rgb_adjustment_brightens_dark_frames_without_changing_length() {
    let dark = vec![12_u8, 10, 8, 20, 18, 16];
    let adjusted = adjust_kinect_rgb(
        &dark,
        KinectRgbAdjustment {
            enabled: true,
            target_luma: 0.35,
            auto_gain_max: 4.0,
            gamma: 0.75,
            ..KinectRgbAdjustment::default()
        },
    );

    assert_eq!(adjusted.len(), dark.len());
    assert!(mean_rgb_luma(&adjusted) > mean_rgb_luma(&dark));
    assert!(adjusted.iter().zip(dark.iter()).any(|(new, old)| new > old));
}

#[test]
fn disabled_kinect_rgb_adjustment_preserves_bytes() {
    let rgb = vec![8_u8, 16, 24, 128, 96, 64];
    let adjusted = adjust_kinect_rgb(
        &rgb,
        KinectRgbAdjustment {
            enabled: false,
            ..KinectRgbAdjustment::default()
        },
    );

    assert_eq!(adjusted, rgb);
}

#[test]
fn asr_tool_emits_final_ear_sense_from_command_transcript() {
    let mut adapter = AsrTool::new(AsrToolConfig {
        command: Some("printf hello".to_string()),
        min_voice_rms: 0.01,
        min_chunk_ms: 100,
        max_chunk_ms: 8_000,
        silence_finalize_ms: 0,
    });
    let frame = PcmAudioFrame {
        captured_at_ms: 1_000,
        sample_rate_hz: 1_000,
        channels: 1,
        samples: vec![10_000; 250],
    };

    let ear = adapter.observe_frame(&frame).expect("final ASR sense");

    assert_eq!(ear.transcript.as_deref(), Some("hello"));
    assert_eq!(ear.asr.transcript.as_deref(), Some("hello"));
    assert!(ear.asr.is_final);
    assert_eq!(ear.asr.start_ms, Some(1_000));
    assert_eq!(ear.asr.duration_ms, Some(250));
    assert_eq!(ear.asr.sample_rate_hz, Some(1_000));
    assert_eq!(ear.asr.word_count, Some(1));
    assert_eq!(ear.asr.committed_transcript.as_deref(), Some("hello"));
    assert!(ear.asr.possible_transcript.is_none());
    assert_eq!(ear.transcript_vectors.len(), 1);
    let transcript_vector = &ear.transcript_vectors[0];
    assert_eq!(transcript_vector.collection, TRANSCRIPT_VECTOR_COLLECTION);
    assert_eq!(
        transcript_vector.model.as_deref(),
        Some(ASR_TRANSCRIPT_VECTOR_MODEL)
    );
    assert_eq!(transcript_vector.vector.len(), ASR_TRANSCRIPT_VECTOR_DIM);
    assert_eq!(transcript_vector.occurred_at_ms, Some(1_250));
    assert_eq!(ear.asr.candidate_id, Some(1));
    assert_eq!(ear.asr.stable_text.as_deref(), Some("hello"));
    assert!(ear
        .asr
        .candidate_events
        .iter()
        .any(|event| matches!(event, TranscriptCandidateEvent::CandidateFinalized { .. })));
}

#[test]
fn asr_tool_without_command_does_not_fabricate_words() {
    let mut adapter = AsrTool::new(AsrToolConfig {
        command: None,
        min_voice_rms: 0.01,
        min_chunk_ms: 100,
        max_chunk_ms: 8_000,
        silence_finalize_ms: 0,
    });
    let frame = PcmAudioFrame {
        captured_at_ms: 1_000,
        sample_rate_hz: 1_000,
        channels: 1,
        samples: vec![10_000; 250],
    };

    assert!(adapter.observe_frame(&frame).is_none());
}

#[test]
fn frame_processor_publishes_floor_evidence_without_requiring_lidar() {
    let calibration = pete_now::DepthGeometryCalibration {
        calibrated: true,
        depth: pete_now::CameraIntrinsics {
            width: 1,
            height: 1,
            fx: 1.0,
            fy: 1.0,
            ..pete_now::CameraIntrinsics::default()
        },
        depth_to_base: pete_now::RigidTransform3 {
            translation_m: [0.0, 0.0, 0.5],
            ..pete_now::RigidTransform3::default()
        },
        ..pete_now::DepthGeometryCalibration::default()
    };
    let mut snapshot = WorldSnapshot::default();
    snapshot.kinect = KinectSense {
        schema_version: 2,
        captured_at_ms: 100,
        geometry_calibration: Some(calibration),
        floor_clip_plane: vec![0.0, 0.0, 1.0, -0.6],
        ..KinectSense::default()
    };
    FrameProcessor::new().process_snapshot(100, &mut snapshot);

    let estimate = snapshot
        .kinect
        .live_geometry_calibration
        .expect("live floor estimate");
    assert_eq!(
        estimate.trust_state,
        pete_now::CalibrationTrustState::Estimating
    );
    assert_eq!(estimate.epoch.id, 0);
    assert_eq!(
        estimate
            .evidence_counts
            .get(&pete_now::CalibrationEvidenceSource::FloorPlane),
        Some(&1)
    );
    assert!(!estimate
        .evidence_counts
        .contains_key(&pete_now::CalibrationEvidenceSource::Lidar));
    assert!(estimate.rejection_reasons[0].contains("unobservable"));
}
