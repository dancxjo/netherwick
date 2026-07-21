use super::*;
use std::io::Write;
use std::sync::Arc;

struct StaticFaceDetector;

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
        Some("mpu6050_complementary_accel_gyro")
    );
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
            vec![SensePacket::Imu(ImuSense {
                captured_at_ms: 100,
                orientation: vec![0.0, 0.0],
                ..ImuSense::default()
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
                SensePacket::Imu(ImuSense {
                    captured_at_ms: 200,
                    orientation: vec![0.2, -0.2],
                    ..ImuSense::default()
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
