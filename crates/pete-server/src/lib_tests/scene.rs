#[test]
fn map_pose_graph_summary_exposes_loop_acceptance_and_rejection_reasons() {
    let mut map = LocalMap::default();
    map.pose_graph = PoseGraph {
        nodes: vec![
            PoseNode {
                id: "live-pose-0".to_string(),
                pose_estimate: PoseEstimate {
                    pose: Pose2::default(),
                    confidence: 0.9,
                    covariance: [0.05, 0.05, 0.1],
                    source: "test".to_string(),
                    t_ms: 100,
                },
                t_ms: 100,
                source_frame_id: Some("seed".to_string()),
            },
            PoseNode {
                id: "live-pose-1".to_string(),
                pose_estimate: PoseEstimate {
                    pose: Pose2 {
                        x_m: 0.05,
                        y_m: 0.0,
                        heading_rad: 0.0,
                    },
                    confidence: 0.9,
                    covariance: [0.05, 0.05, 0.1],
                    source: "test".to_string(),
                    t_ms: 200,
                },
                t_ms: 200,
                source_frame_id: Some("return".to_string()),
            },
        ],
        edges: vec![
            PoseEdge {
                from: "live-pose-1".to_string(),
                to: "live-pose-0".to_string(),
                transform: Pose2::default(),
                covariance: [0.04, 0.04, 0.08],
                confidence: 0.94,
                source: PoseEdgeSource::LoopClosureCandidate {
                    kind: "entity_constellation".to_string(),
                    target_frame_id: Some("seed".to_string()),
                    source_frame_id: Some("return".to_string()),
                    source_experience_id: None,
                    source_instant_frame_id: None,
                    source_vector_refs: Vec::new(),
                    source_vector_id: Some("constellation-seed".to_string()),
                    query_vector_id: Some("constellation-return".to_string()),
                    query_experience_id: None,
                    registration: None,
                },
                active: true,
                rejection_reason: None,
            },
            PoseEdge {
                from: "live-pose-1".to_string(),
                to: "unresolved".to_string(),
                transform: Pose2::default(),
                covariance: [0.2, 0.2, 0.4],
                confidence: 0.6,
                source: PoseEdgeSource::LoopClosureCandidate {
                    kind: "same_place".to_string(),
                    target_frame_id: Some("far-away".to_string()),
                    source_frame_id: Some("return".to_string()),
                    source_experience_id: None,
                    source_instant_frame_id: None,
                    source_vector_refs: Vec::new(),
                    source_vector_id: Some("place-seed".to_string()),
                    query_vector_id: Some("place-return".to_string()),
                    query_experience_id: None,
                    registration: None,
                },
                active: false,
                rejection_reason: Some("confidence 0.600 below gate 0.850".to_string()),
            },
        ],
    };

    let summary = map_pose_graph_summary(&map);

    assert_eq!(summary.loop_candidate_edges, 2);
    assert_eq!(summary.loop_candidate_active_edges, 1);
    assert_eq!(summary.loop_candidate_rejected_edges, 1);
    assert_eq!(
        summary.loop_candidate_rejection_reasons,
        vec!["confidence 0.600 below gate 0.850".to_string()]
    );
    assert_eq!(
        summary.latest_edge_source.as_deref(),
        Some("loop_closure_candidate")
    );
}

#[test]
fn missing_eye_kinect_and_audio_serialize_as_empty_or_null() {
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.battery_level = 0.0;
    let scene = snapshot_to_scene(
        &snapshot,
        None,
        None,
        LiveTrainingStatus::default(),
        NudgeStatus::default(),
        default_behavior_nodes(),
        None,
        None,
        HardwareControlStatus::unavailable("unit test"),
    );
    let value = serde_json::to_value(scene).unwrap();

    assert!(value["eye"].is_null());
    assert_eq!(value["dead_battery"].as_bool(), Some(true));
    assert_eq!(value["kinect"]["points"].as_array().unwrap().len(), 0);
    assert!(value["audio"].is_null());
    assert!(value["warnings"].as_array().unwrap().len() >= 3);
}

#[test]
fn mono_pcm_audio_reports_energy_without_bearing() {
    let mut snapshot = WorldSnapshot::default();
    snapshot.ear_pcm = Some(pete_sensors::PcmAudioFrame {
        captured_at_ms: 100,
        sample_rate_hz: 44_100,
        channels: 1,
        samples: vec![0, i16::MAX / 2, -(i16::MAX / 2)],
    });
    let scene = snapshot_to_scene(
        &snapshot,
        None,
        None,
        LiveTrainingStatus::default(),
        NudgeStatus::default(),
        default_behavior_nodes(),
        None,
        None,
        HardwareControlStatus::unavailable("unit test"),
    );
    let value = serde_json::to_value(scene).unwrap();

    assert!(value["audio"]["bearing_rad"].is_null());
    assert!(value["audio"]["energy"].as_f64().unwrap() > 0.0);
    assert!(value["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|warning| warning == "no audio bearing stream"));
}

#[test]
fn scene_packet_exposes_persistent_stable_world_belief_points() {
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.last_update_ms = 300;
    let mut cloud = VoxelPointCloud::default();
    for t_ms in [100, 200, 300] {
        cloud.integrate_observation(PointCloudObservation {
            frame: PointCloudFrame::OdometryWorld,
            pose: PoseEstimate {
                pose: Pose2::default(),
                confidence: 0.9,
                covariance: [0.01, 0.01, 0.02],
                source: "unit-test".to_string(),
                t_ms,
            },
            orientation: OrientationEstimate {
                roll_rad: Some(0.01),
                pitch_rad: Some(-0.02),
                yaw_rad: Some(0.0),
                roll_pitch_from_imu: true,
                yaw_source: YawSource::OdometryHeading,
            },
            points: vec![PointCloudPoint {
                position: Point3D {
                    x_m: 1.0,
                    y_m: 0.5,
                    z_m: 0.2,
                },
                color_rgb: None,
                confidence: 1.0,
                depth_index: None,
                depth_uv: None,
                depth_image_size: None,
                source_frame_id: None,
            }],
            source: "unit-test".to_string(),
            t_ms,
            metadata: serde_json::json!({}),
        });
    }

    let scene = snapshot_to_scene(
        &snapshot,
        None,
        None,
        LiveTrainingStatus::default(),
        NudgeStatus::default(),
        default_behavior_nodes(),
        Some(&cloud),
        None,
        HardwareControlStatus::unavailable("unit test"),
    );

    assert_eq!(
        scene.world_belief_layers,
        vec![
            "current rays",
            "raw point cloud",
            "raw camera-frame points",
            "robot-frame points",
            "world-frame points",
            "accumulated occupancy",
            "floor plane",
            "axes gizmo",
            "stable wall candidates"
        ]
    );
    assert_eq!(
        scene
            .kinect
            .accumulated_summary
            .as_ref()
            .unwrap()
            .stable_voxels,
        1
    );
    assert_eq!(scene.kinect.accumulated_points.len(), 1);
    assert!(scene.kinect.accumulated_points[0].stable);
    let belief = scene.kinect.local_world_belief.as_ref().unwrap();
    assert_eq!(belief.stable_voxels, 1);
    assert!(belief.orientation_status.roll_pitch_corrected);
    assert_eq!(scene.kinect.coordinate_system.as_deref(), Some("world"));
}

#[test]
fn live_state_does_not_accumulate_uncalibrated_real_depth_images() {
    let state = LiveViewState::new();
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.last_update_ms = 100;
    snapshot.kinect = KinectSense {
        depth_m: vec![1.0],
        depth_width: 1,
        depth_height: 1,
        depth_fx: 1.0,
        depth_fy: 1.0,
        min_depth_m: 0.1,
        max_depth_m: 3.0,
        depth_coordinate_system: Some("kinect_camera".to_string()),
        ..KinectSense::default()
    };

    state.update(snapshot.clone());

    assert_eq!(state.point_cloud_snapshot().summary().observations, 0);
    let scene = snapshot_to_scene(
        &snapshot,
        None,
        None,
        LiveTrainingStatus::default(),
        NudgeStatus::default(),
        default_behavior_nodes(),
        Some(&state.point_cloud_snapshot()),
        None,
        HardwareControlStatus::unavailable("unit test"),
    );
    assert!(scene
        .warnings
        .iter()
        .any(|warning| warning.contains("accumulated world cloud is disabled")));
}

#[test]
fn live_state_applies_scene_calibration_to_accumulated_point_cloud() {
    let state = LiveViewState::new();
    let calibration = SceneSensorCalibration {
        camera_height_m: 0.50,
        camera_forward_m: 0.10,
        camera_pitch_rad: 12.0_f32.to_radians(),
        ..SceneSensorCalibration::sim_default()
    };
    state.update_scene_metadata(LiveSceneMetadata {
        arena: None,
        objects: Vec::new(),
        sensor_calibration: Some(calibration),
    });
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.last_update_ms = 100;
    snapshot.kinect = KinectSense {
        depth_m: vec![1.0],
        depth_width: 1,
        depth_height: 1,
        depth_fx: 1.0,
        depth_fy: 1.0,
        min_depth_m: 0.1,
        max_depth_m: 3.0,
        depth_coordinate_system: Some("kinect_camera".to_string()),
        ..KinectSense::default()
    };

    state.update(snapshot);

    let cloud = state.point_cloud_snapshot();
    assert_eq!(cloud.summary().observations, 1);
    assert!((cloud.config.camera_height_m - 0.50).abs() < 0.001);
    assert!((cloud.config.camera_forward_m - 0.10).abs() < 0.001);
    assert!((cloud.config.camera_pitch_rad - 12.0_f32.to_radians()).abs() < 0.001);
}

#[test]
fn scene_body_cliff_uses_create_flags_not_analog_risk() {
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.cliff_sensors.front_left = 0.96;
    snapshot.body.cliff_sensors.front_right = 0.82;
    let scene = snapshot_to_scene(
        &snapshot,
        None,
        None,
        LiveTrainingStatus::default(),
        NudgeStatus::default(),
        default_behavior_nodes(),
        None,
        None,
        HardwareControlStatus::unavailable("unit test"),
    );

    assert!(!scene.body.cliff);

    snapshot.body.flags.cliff_front_left = true;
    let scene = snapshot_to_scene(
        &snapshot,
        None,
        None,
        LiveTrainingStatus::default(),
        NudgeStatus::default(),
        default_behavior_nodes(),
        None,
        None,
        HardwareControlStatus::unavailable("unit test"),
    );

    assert!(scene.body.cliff);
}

#[test]
fn compact_kinect_depth_projects_as_meter_range_fan() {
    let depths = vec![2.0; 32];
    let kinect = KinectSense {
        depth_m: depths,
        ..KinectSense::default()
    };
    let (points, diagnostics) =
        depth_points(&kinect, Some(SceneSensorCalibration::sim_default()), None);

    assert_eq!(points.len(), 32);
    assert_eq!(diagnostics.coordinate_system, "scene_robot_render");
    assert_eq!(
        diagnostics.render_frame.as_deref(),
        Some("scene: +x left, +y up, +z forward")
    );
    assert!((points[0].x + 1.847759).abs() < 0.001);
    assert!((points[0].z - 0.765367).abs() < 0.001);
    assert!((points[31].x - 1.847759).abs() < 0.001);
    assert!((points[31].z - 0.765367).abs() < 0.001);

    let near_center = &points[15];
    assert!(near_center.z > 1.99);
    assert!(near_center.x.abs() < 0.11);
}

#[test]
fn kinect_depth_image_projects_with_pinhole_intrinsics() {
    let kinect = KinectSense {
        depth_m: vec![1.0, 2.0, 0.0, 4.0],
        depth_width: 2,
        depth_height: 2,
        depth_fx: 2.0,
        depth_fy: 2.0,
        depth_cx: 0.5,
        depth_cy: 0.5,
        min_depth_m: 0.4,
        max_depth_m: 3.0,
        depth_coordinate_system: Some("kinect_camera".to_string()),
        ..KinectSense::default()
    };

    let (points, diagnostics) = depth_points(&kinect, None, None);

    assert_eq!(points.len(), 2);
    assert_eq!(diagnostics.depth_width, 2);
    assert_eq!(diagnostics.depth_height, 2);
    assert_eq!(diagnostics.valid_depth_count, 2);
    assert_eq!(diagnostics.skipped_depth_count, 1);
    assert_eq!(diagnostics.clipped_depth_count, 1);
    assert_eq!(diagnostics.coordinate_system, "kinect_camera");
    assert!((points[0].x + 0.25).abs() < 0.001);
    assert!((points[0].y + 0.25).abs() < 0.001);
    assert!((points[0].z - 1.0).abs() < 0.001);
    assert!((points[1].x - 0.5).abs() < 0.001);
    assert!((points[1].y + 0.5).abs() < 0.001);
    assert!((points[1].z - 2.0).abs() < 0.001);
}

#[test]
fn kinect_depth_image_projects_rgb_frame_colors_onto_points() {
    let kinect = KinectSense {
        depth_m: vec![1.0, 1.0, 1.0, 1.0],
        depth_width: 2,
        depth_height: 2,
        depth_fx: 2.0,
        depth_fy: 2.0,
        depth_cx: 0.5,
        depth_cy: 0.5,
        min_depth_m: 0.1,
        max_depth_m: 3.0,
        depth_coordinate_system: Some("kinect_camera".to_string()),
        ..KinectSense::default()
    };
    let eye = EyeFrame {
        captured_at_ms: 10,
        width: 2,
        height: 2,
        format: EyeFrameFormat::Rgb8,
        bytes: vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0],
        source: Some("kinect-freenect-rgb".to_string()),
    };
    let color = DepthColorImage::from_eye_frame(&eye).unwrap();

    let (points, diagnostics) = depth_points(&kinect, None, Some(&color));

    assert_eq!(diagnostics.coordinate_system, "kinect_camera");
    assert_eq!(points.len(), 4);
    assert_eq!((points[0].r, points[0].g, points[0].b), (255, 0, 0));
    assert_eq!((points[1].r, points[1].g, points[1].b), (0, 255, 0));
    assert_eq!((points[2].r, points[2].g, points[2].b), (0, 0, 255));
    assert_eq!((points[3].r, points[3].g, points[3].b), (255, 255, 0));
}

#[test]
fn depth_color_sampling_applies_calibrated_pixel_offsets() {
    let width = 80;
    let height = 80;
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            rgb.extend_from_slice(&[x as u8, y as u8, 0]);
        }
    }
    let color = DepthColorImage { width, height, rgb };

    assert_eq!(
        color.sample_depth_pixel_with_offset(10, 10, width, height, 0, 0),
        Some([10, 10, 0])
    );
    assert_eq!(
        color.sample_depth_pixel_with_offset(10, 10, width, height, 3, 7),
        Some([13, 17, 0])
    );
    assert_eq!(
        color.sample_depth_pixel_with_offset(0, 0, width, height, -3, -7),
        Some([0, 0, 0])
    );
}

#[test]
fn calibrated_kinect_depth_image_reports_below_floor_points() {
    let kinect = KinectSense {
        depth_m: vec![1.0],
        depth_width: 1,
        depth_height: 1,
        depth_fx: 1.0,
        depth_fy: 1.0,
        depth_cx: 0.0,
        depth_cy: 0.0,
        min_depth_m: 0.1,
        max_depth_m: 3.0,
        depth_coordinate_system: Some("kinect_camera".to_string()),
        ..KinectSense::default()
    };
    let calibration = SceneSensorCalibration {
        camera_height_m: 0.1,
        camera_pitch_rad: 0.25,
        ..SceneSensorCalibration::sim_default()
    };

    let (points, diagnostics) = depth_points(&kinect, Some(calibration), None);

    assert_eq!(points.len(), 1);
    assert_eq!(diagnostics.coordinate_system, "scene_robot_render");
    assert_eq!(
        diagnostics.point_coordinate_system.as_deref(),
        Some("scene axes derived from robot math frame")
    );
    assert_eq!(diagnostics.below_floor_count, 1);
    assert_eq!(diagnostics.below_floor_ratio, 1.0);
    assert!(diagnostics.min_z_m.unwrap() < 0.0);
    assert!(diagnostics.median_z_m.unwrap() < 0.0);
}
