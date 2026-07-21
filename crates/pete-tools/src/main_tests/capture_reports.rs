#[test]
fn sensor_only_live_publish_does_not_refresh_body_timestamp() {
    let live_state = LiveViewState::new().with_real_slow_hardware_control();
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.last_update_ms = 1_234;
    live_state.update(snapshot);

    publish_live_sensor_only_snapshot(
        &live_state,
        &SensePacket::EyeFrame(EyeFrame {
            captured_at_ms: 9_999,
            rgbd_frame_id: None,
            device_timestamp_ms: None,
            width: 1,
            height: 1,
            format: EyeFrameFormat::Rgb8,
            bytes: vec![0, 0, 0],
            source: Some("test".to_string()),
        }),
    );

    let latest = live_state.latest().unwrap();
    assert_eq!(latest.body.last_update_ms, 1_234);
    assert_eq!(
        latest.eye_frame.as_ref().map(|frame| frame.captured_at_ms),
        Some(9_999)
    );
}

#[test]
fn geometry_truth_rejects_stale_and_unsynchronized_kinect_imu_samples() {
    let timestamps = GeometryTimestampDiagnostics {
        frame_count: 3,
        depth_frame_count: 3,
        first_frame_t_ms: Some(1_000),
        last_frame_t_ms: Some(1_200),
        frame_timestamps_monotonic: true,
        median_frame_dt_ms: Some(100),
        max_frame_dt_ms: Some(100),
        body_last_update_age_ms: Some(0),
        body_timestamp_in_future: false,
        eye_frame_age_ms: None,
        ear_pcm_age_ms: None,
        kinect_capture_timestamp_present: true,
        kinect_capture_age_ms: None,
        kinect_timestamp_in_future: true,
        imu_capture_timestamp_present: true,
        imu_capture_age_ms: Some(23_000),
        imu_timestamp_in_future: false,
        kinect_imu_skew_ms: Some(22_950),
        kinect_body_skew_ms: Some(50),
        note: String::new(),
    };
    let imu = GeometryImuInterpretation {
        raw_orientation: vec![0.0, 0.0],
        assumed_units: "radians".to_string(),
        assumed_axis_order: "[roll, pitch]".to_string(),
        roll_deg: Some(0.0),
        pitch_deg: Some(0.0),
        yaw_deg: None,
        roll_pitch_correction_active: true,
        yaw_source: "OdometryHeading".to_string(),
        contract_known: true,
        contract_source: "test".to_string(),
        note: String::new(),
    };
    let stationary = StationaryRotationDiagnostics {
        evaluated: true,
        reason: "test".to_string(),
        frame_count: 3,
        direction: "counter_clockwise".to_string(),
        heading_delta_deg: 90.0,
        cumulative_rotation_deg: 90.0,
        final_heading_error_deg: 0.0,
        translation_delta_m: 0.0,
        max_axle_translation_m: 0.0,
        stationary_frames_before: 1,
        stationary_frames_after: 1,
        imu_integrated_rotation_deg: Some(90.0),
        imu_odometry_error_deg: Some(0.0),
        rotation_agreement: true,
        calibration_epoch_ids: vec![0],
        remount_detected: false,
        reconverged_after_remount: false,
        observability_gate_passed: true,
        insufficient_observability_exposed: false,
        covariance_gate_passed: true,
        optional_lidar_present: false,
        raw_points_seen: 100,
        voxel_count: 20,
        stable_voxel_count: 10,
        stable_voxel_ratio: 0.5,
        stable_z_span_m: Some(0.2),
        stable_z_median_m: Some(0.5),
    };
    let args = GeometryDebugArgs {
        capture: None,
        live_now_url: None,
        out: String::new(),
        samples: 16,
        max_below_floor_ratio: 0.02,
        max_body_timestamp_age_ms: 200,
        max_kinect_timestamp_age_ms: 200,
        max_imu_timestamp_age_ms: 200,
        max_kinect_imu_skew_ms: 100,
        max_kinect_body_skew_ms: 100,
        max_rgbd_skew_ms: 50,
        min_depth_frames: 2,
        min_stationary_rotation_deg: 45.0,
        max_stationary_translation_m: 0.2,
        min_stationary_stable_voxel_ratio: 0.05,
        max_stationary_stable_z_span_m: 1.5,
    };

    let report = sensor_truth_report(
        &WorldSnapshot::default(),
        false,
        &timestamps,
        &imu,
        0.0,
        &stationary,
        &args,
    );

    assert!(!report.ready_for_real_slam);
    for name in [
        "kinect_timestamp_fresh",
        "imu_timestamp_fresh",
        "kinect_imu_synchronized",
    ] {
        assert_eq!(
            report
                .gates
                .iter()
                .find(|gate| gate.name == name)
                .unwrap()
                .status,
            SensorTruthStatus::Fail
        );
    }
}

#[test]
fn return_to_start_gate_requires_measured_error_overlap_and_endpoint_improvement() {
    let pose = |x_m: f32| {
        let mut pose = BodySense::default().odometry;
        pose.x_m = x_m;
        pose
    };
    let estimate = |x_m: f32, t_ms: u64| pete_map::PoseEstimate {
        pose: pose(x_m),
        confidence: 0.9,
        covariance: [0.05, 0.05, 0.1],
        source: "test".to_string(),
        t_ms,
    };
    let mut map = LocalMap::default();
    map.pose_graph.nodes = vec![
        pete_map::PoseNode {
            id: "start".to_string(),
            pose_estimate: estimate(0.0, 0),
            t_ms: 0,
            source_frame_id: Some("start".to_string()),
        },
        pete_map::PoseNode {
            id: "away".to_string(),
            pose_estimate: estimate(1.0, 100),
            t_ms: 100,
            source_frame_id: Some("away".to_string()),
        },
        pete_map::PoseNode {
            id: "return".to_string(),
            pose_estimate: estimate(0.1, 200),
            t_ms: 200,
            source_frame_id: Some("return".to_string()),
        },
    ];
    map.pose_graph.edges.push(pete_map::PoseEdge {
        from: "return".to_string(),
        to: "start".to_string(),
        transform: pose(0.0),
        covariance: [0.02, 0.02, 0.04],
        confidence: 0.95,
        source: PoseEdgeSource::LoopClosureCandidate {
            kind: "test".to_string(),
            target_frame_id: Some("start".to_string()),
            source_frame_id: Some("return".to_string()),
            source_experience_id: None,
            source_instant_frame_id: None,
            source_vector_refs: Vec::new(),
            source_vector_id: None,
            query_vector_id: None,
            query_experience_id: None,
            registration: Some(pete_map::LoopRegistrationMeasurement {
                algorithm: "correlative_occupancy_submap_registration".to_string(),
                registered_pose: pose(0.0),
                score: 0.8,
                odometry_score: 0.2,
                geometric_overlap: 0.8,
                odometry_geometric_overlap: 0.2,
            }),
        },
        active: true,
        rejection_reason: None,
    });
    map.pose_graph_optimization.initial_mean_error = 0.3;
    map.pose_graph_optimization.final_mean_error = 0.1;
    for (x_m, t_ms) in [(0.0, 0), (0.4, 200)] {
        map.observations.push(pete_map::MapObservation {
            pose: estimate(x_m, t_ms),
            range_beams: Vec::new(),
            source_snapshot: serde_json::json!({"body":{"odometry":pose(x_m)}}),
            t_ms,
        });
    }

    let validation = return_to_start_validation(
        &map,
        &ReturnToStartCalibrationEvidence {
            epoch_ids: vec![2],
            saw_invalidated: false,
            last_epoch_trusted: true,
            uncertainty_reported: true,
            kinect_present: true,
            lidar_present: false,
        },
        Some(ReturnToStartPhysicalReference {
            direction: "counter_clockwise".to_string(),
            actual_endpoint_distance_m: 0.08,
            actual_orientation_error_deg: 2.0,
            distance_tolerance_m: 0.15,
            orientation_tolerance_deg: 5.0,
            notes: vec!["synthetic measured fixture".to_string()],
        }),
    );

    assert!(validation.evaluated);
    assert!(validation.passed);
    assert!(validation.graph_error_reduced);
    assert!(validation.wall_overlap_improved);
    assert!(validation.corrected_pose_near_start);
    assert!(validation.corrected_endpoint_improves_over_raw);
    assert!(validation.navigation_trusted);
    assert_eq!(validation.navigation_trust_decision, "trusted");
    assert_eq!(validation.geometry_mode, "kinect_only");
    assert_eq!(validation.raw_final_distance_to_start_m, Some(0.4));
    assert_eq!(validation.corrected_final_distance_to_start_m, Some(0.1));
}

#[test]
fn stationary_rotation_evaluator_accumulates_wrapped_full_turn_and_exposes_unobservability() {
    let headings = [
        0.0,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::PI,
        -std::f32::consts::FRAC_PI_2,
        0.0,
    ];
    let frames = headings
        .into_iter()
        .enumerate()
        .map(|(index, heading)| {
            let mut snapshot = WorldSnapshot::default();
            snapshot.body.odometry.heading_rad = heading;
            snapshot.body.velocity.turn_rad_s = if index == 0 || index == 4 { 0.0 } else { 0.5 };
            snapshot.kinect.depth_m = vec![1.0];
            snapshot.kinect.depth_width = 1;
            snapshot.kinect.depth_height = 1;
            snapshot.imu.schema_version = 2;
            snapshot.imu.captured_at_ms = 1_000 + index as u64 * 1_000;
            snapshot.imu.angular_velocity = vec![
                0.0,
                0.0,
                if index == 0 || index == 4 {
                    0.0
                } else {
                    std::f32::consts::TAU / 3.0
                },
            ];
            pete_worldlab::CaptureFrameRecord {
                index: index as u64,
                t_ms: 1_000 + index as u64 * 1_000,
                snapshot,
                events: Vec::new(),
                assets: pete_worldlab::CaptureFrameAssets::default(),
                stream_metadata: None,
            }
        })
        .collect::<Vec<_>>();
    let args = GeometryDebugArgs {
        capture: None,
        live_now_url: None,
        out: String::new(),
        samples: 16,
        max_below_floor_ratio: 0.02,
        max_body_timestamp_age_ms: 200,
        max_kinect_timestamp_age_ms: 200,
        max_imu_timestamp_age_ms: 200,
        max_kinect_imu_skew_ms: 100,
        max_kinect_body_skew_ms: 100,
        max_rgbd_skew_ms: 50,
        min_depth_frames: 2,
        min_stationary_rotation_deg: 300.0,
        max_stationary_translation_m: 0.2,
        min_stationary_stable_voxel_ratio: 0.05,
        max_stationary_stable_z_span_m: 1.5,
    };

    let diagnostics = stationary_rotation_diagnostics(&frames, &args);

    assert!(diagnostics.evaluated);
    assert!((diagnostics.cumulative_rotation_deg - 360.0).abs() < 0.1);
    assert!(diagnostics.final_heading_error_deg < 0.1);
    assert!(diagnostics.rotation_agreement);
    assert_eq!(diagnostics.stationary_frames_before, 1);
    assert_eq!(diagnostics.stationary_frames_after, 1);
    assert!(diagnostics.insufficient_observability_exposed);
    assert!(!diagnostics.observability_gate_passed);
    assert!(!diagnostics.optional_lidar_present);
}

#[test]
fn missing_streams_generate_warnings() {
    let mut counts = StreamCounts::default();
    counts.observe(&WorldSnapshot::default());
    let streams = counts.streams();
    assert!(streams.present.contains(&"body".to_string()));
    assert!(streams.missing.contains(&"rgb".to_string()));
    assert!(counts
        .warnings()
        .iter()
        .any(|warning| warning == "rgb stream missing"));
}

#[tokio::test]
async fn inspect_capture_reads_tiny_fake_capture() {
    let temp_dir = temp_path("pete_inspect_capture");
    let mut writer = CaptureWriter::create(&temp_dir, CaptureSource::RealRobot, Some(100))
        .await
        .unwrap();
    let mut snapshot = WorldSnapshot::default();
    snapshot.eye.frames.push(vec![0.1, 0.2]);
    writer
        .append_snapshot(100, snapshot, Vec::new())
        .await
        .unwrap();
    writer.finish().await.unwrap();

    let report = inspect_capture_report(&temp_dir).await.unwrap();
    assert_eq!(report.frame_count, 1);
    assert!(report.streams_present.contains(&"rgb".to_string()));
    assert!(report.streams_missing.contains(&"audio".to_string()));
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn inspect_capture_reports_asset_ranges_bytes_and_integrity() {
    let temp_dir = temp_path("pete_inspect_capture_assets");
    let mut writer = CaptureWriter::create(&temp_dir, CaptureSource::RealRobot, Some(100))
        .await
        .unwrap();
    let mut snapshot = WorldSnapshot::default();
    snapshot.kinect.captured_at_ms = 990;
    snapshot.kinect.depth_width = 1;
    snapshot.kinect.depth_height = 1;
    snapshot.kinect.depth_m = vec![1.0];
    snapshot.kinect.geometry_calibration = Some(pete_now::DepthGeometryCalibration {
        calibrated: true,
        depth: pete_now::CameraIntrinsics {
            width: 1,
            height: 1,
            fx: 1.0,
            fy: 1.0,
            cx: 0.0,
            cy: 0.0,
            distortion: [0.0; 5],
        },
        depth_scale: 1.0,
        ..pete_now::DepthGeometryCalibration::default()
    });
    writer
        .append_snapshot_with_exported_assets(
            1_000,
            snapshot,
            Vec::new(),
            false,
            true,
            false,
        )
        .await
        .unwrap();
    writer.finish().await.unwrap();

    let report = inspect_capture_report(&temp_dir).await.unwrap();
    let depth = report
        .asset_streams
        .iter()
        .find(|stream| stream.kind == "depth")
        .unwrap();
    assert_eq!(depth.count, 1);
    assert_eq!(depth.first_producer_ms, Some(990));
    assert_eq!(depth.last_producer_ms, Some(990));
    assert!(depth.bytes > 0);
    assert!(depth.missing_intervals.is_empty());
    assert_eq!(depth.checksum_failures, 0);
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn pose_graph_report_reads_capture_frames_and_gates_loop_candidates() {
    let temp_dir = temp_path("pete_pose_graph_capture");
    let mut writer = CaptureWriter::create(&temp_dir, CaptureSource::Sim, Some(100))
        .await
        .unwrap();
    let mut first = WorldSnapshot::default();
    first.body.odometry.x_m = 0.0;
    first.eye.scene_vectors.push(
        VectorArtifact::new(SCENE_VECTOR_COLLECTION, "scene-first", vec![1.0, 0.0])
            .with_source_frame_id("capture-frame-0"),
    );
    writer
        .append_snapshot(100, first, Vec::new())
        .await
        .unwrap();

    let mut second = WorldSnapshot::default();
    second.body.odometry.x_m = 1.0;
    second.eye.scene_vectors.push(VectorArtifact::new(
        SCENE_VECTOR_COLLECTION,
        "scene-query-strong",
        vec![1.0, 0.0],
    ));
    second.eye.scene_vectors.push(VectorArtifact::new(
        SCENE_VECTOR_COLLECTION,
        "scene-query-weak",
        vec![0.0, 1.0],
    ));
    writer
        .append_snapshot(200, second, Vec::new())
        .await
        .unwrap();
    writer.finish().await.unwrap();

    let args = PoseGraphReportArgs {
        ledger: "unused-when-capture-is-set".to_string(),
        capture: Some(temp_dir.to_string_lossy().to_string()),
        out: temp_dir
            .join("pose-graph.json")
            .to_string_lossy()
            .to_string(),
        min_node_distance_m: 0.5,
        min_node_degrees: 15.0,
        max_ticks_between_nodes: 10,
        min_loop_confidence: 0.55,
    };
    let report = generate_pose_graph_report(&args).await.unwrap();

    assert_eq!(report.nodes, 2);
    assert_eq!(report.odometry_edges, 1);
    assert_eq!(report.loop_candidate_edges, 2);
    assert_eq!(report.active_loop_candidate_edges, 1);
    assert_eq!(report.rejected_loop_candidates, 1);
    assert_eq!(
        report.rejected_candidates[0].reason,
        "confidence 0.000 below gate 0.550"
    );
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn pose_graph_report_uses_ledger_place_recognition_latents() {
    let temp_dir = temp_path("pete_pose_graph_ledger");
    let ledger = JsonlLedger::new(&temp_dir);

    let first = pose_graph_test_frame(100, 0.0, vec![1.0, 0.0, 0.0]);
    let first_experience_id = first.experiences[0].id.to_string();
    let first_frame_id = first.id.to_string();
    let second = pose_graph_test_frame(200, 1.0, vec![0.99, 0.01, 0.0]);
    let second_experience_id = second.experiences[0].id.to_string();
    ledger.append(&first).await.unwrap();
    ledger.append(&second).await.unwrap();

    let args = PoseGraphReportArgs {
        ledger: temp_dir.to_string_lossy().to_string(),
        capture: None,
        out: temp_dir
            .join("pose-graph.json")
            .to_string_lossy()
            .to_string(),
        min_node_distance_m: 0.5,
        min_node_degrees: 15.0,
        max_ticks_between_nodes: 10,
        min_loop_confidence: 0.55,
    };
    let report = generate_pose_graph_report(&args).await.unwrap();

    assert_eq!(report.nodes, 2);
    assert_eq!(report.odometry_edges, 1);
    assert_eq!(report.loop_candidate_edges, 1);
    assert_eq!(report.active_loop_candidate_edges, 1);
    assert_eq!(report.rejected_loop_candidates, 0);
    let loop_edge = report
        .graph
        .edges
        .iter()
        .find(|edge| {
            matches!(
                edge.source,
                pete_map::PoseEdgeSource::LoopClosureCandidate { .. }
            )
        })
        .expect("loop edge");
    match &loop_edge.source {
        pete_map::PoseEdgeSource::LoopClosureCandidate {
            target_frame_id,
            source_frame_id,
            source_experience_id,
            source_instant_frame_id,
            query_experience_id,
            ..
        } => {
            assert_eq!(target_frame_id.as_deref(), Some(first_frame_id.as_str()));
            assert_eq!(
                source_frame_id.as_deref(),
                Some(second.id.to_string().as_str())
            );
            assert_eq!(
                source_experience_id.as_deref(),
                Some(first_experience_id.as_str())
            );
            assert_eq!(
                source_instant_frame_id.as_deref(),
                Some(first_frame_id.as_str())
            );
            assert_eq!(
                query_experience_id.as_deref(),
                Some(second_experience_id.as_str())
            );
        }
        pete_map::PoseEdgeSource::Odometry => panic!("expected loop edge"),
        pete_map::PoseEdgeSource::ScanMatch { .. } => panic!("expected loop edge"),
    }
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn representation_report_writes_json_from_capture_fixture() {
    let temp_dir = temp_path("pete_representation_report_capture");
    let mut writer = CaptureWriter::create(&temp_dir, CaptureSource::Sim, Some(100))
        .await
        .unwrap();

    let mut first = WorldSnapshot::default();
    first.body.odometry.x_m = 0.0;
    first.range.nearest_m = Some(0.5);
    first.eye.scene_vectors.push(
        VectorArtifact::new(SCENE_VECTOR_COLLECTION, "scene-a", vec![1.0, 0.0])
            .with_source_frame_id("capture-frame-0"),
    );
    writer
        .append_snapshot(100, first, Vec::new())
        .await
        .unwrap();

    let mut second = WorldSnapshot::default();
    second.body.odometry.x_m = 0.8;
    second.range.nearest_m = Some(0.45);
    second.eye.scene_vectors.push(VectorArtifact::new(
        SCENE_VECTOR_COLLECTION,
        "scene-b",
        vec![0.98, 0.02],
    ));
    writer
        .append_snapshot(200, second, Vec::new())
        .await
        .unwrap();
    writer.finish().await.unwrap();

    let out = temp_dir.join("reports/representation/report.json");
    run_representation_report(RepresentationReportArgs {
        ledger: "unused-when-capture-is-set".to_string(),
        capture: Some(temp_dir.to_string_lossy().to_string()),
        physical_reference: None,
        out: out.to_string_lossy().to_string(),
    })
    .await
    .unwrap();

    let value: serde_json::Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(value["frame_count"], 2);
    assert_eq!(value["input"]["source_type"], "capture");
    assert!(value["entity_memory"].is_object());
    assert!(value["map"].is_object());
    assert!(value["pose_graph"].is_object());
    assert!(value["place_recognition"].is_object());
    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn calibration_replay_summary_records_epoch_and_trust_transitions() {
    let calibration = pete_now::DepthGeometryCalibration::default();
    let mut first_estimate = pete_now::CalibrationStateMachine::new(
        calibration.depth_to_base,
        100,
        pete_now::CalibrationStateConfig::default(),
    )
    .estimate()
    .clone();
    first_estimate.trust_state = pete_now::CalibrationTrustState::Trusted;
    first_estimate.residuals.floor_m = Some(0.01);
    let mut second_estimate = first_estimate.clone();
    second_estimate.trust_state = pete_now::CalibrationTrustState::Invalidated;
    second_estimate.epoch.id = 1;
    second_estimate.epoch.invalidation_reason = Some("synthetic remount".to_string());
    second_estimate.residuals.wall_m = Some(0.08);
    let frames = [first_estimate, second_estimate]
        .into_iter()
        .enumerate()
        .map(|(index, estimate)| {
            let mut snapshot = WorldSnapshot::default();
            snapshot.kinect.live_geometry_calibration = Some(estimate);
            pete_worldlab::CaptureFrameRecord {
                index: index as u64,
                t_ms: 100 + index as u64 * 100,
                snapshot,
                events: Vec::new(),
                assets: pete_worldlab::CaptureFrameAssets::default(),
                stream_metadata: None,
            }
        })
        .collect::<Vec<_>>();

    let summary = calibration_replay_summary(&frames);
    assert_eq!(summary.frames_with_estimate, 2);
    assert_eq!(summary.epoch_ids, [0, 1]);
    assert_eq!(summary.epoch_changes, 1);
    assert_eq!(summary.trusted_frames, 1);
    assert_eq!(summary.invalidated_frames, 1);
    assert_eq!(summary.maximum_floor_residual_m, Some(0.01));
    assert_eq!(summary.maximum_wall_residual_m, Some(0.08));
}

fn pose_graph_test_frame(t_ms: u64, x_m: f32, latent_vector: Vec<f32>) -> ExperienceFrame {
    let mut now = Now::blank(t_ms, BodySense::default());
    now.body.odometry.x_m = x_m;
    let sensation_id = uuid::Uuid::new_v4();
    let experience = Experience::new(
        "test.place",
        format!("test place at {x_m:.1}m"),
        Vec::new(),
        vec![sensation_id],
        t_ms,
        t_ms,
    );
    ExperienceFrame {
        id: uuid::Uuid::new_v4(),
        t_ms,
        now,
        sensations: Vec::new(),
        impressions: Vec::new(),
        experiences: vec![experience],
        z: Some(ExperienceLatent {
            t_ms,
            z: latent_vector,
            confidence: 1.0,
            ..ExperienceLatent::default()
        }),
        chosen_action: None,
        conscious_command: None,
        reign_input: None,
        reign_outcome: None,
        predicted_futures: Vec::new(),
        behavior_runs: Vec::new(),
        actual_next: None,
        reward: Reward::default(),
        surprise: SurpriseSense::default(),
        memory_recall: Vec::new(),
        recollections: Vec::new(),
        llm_teaching: Vec::new(),
        counterfactuals: Vec::new(),
        notes: Vec::new(),
    }
}

#[tokio::test]
#[ignore = "slow capture-real mock path can stall workspace test runs"]
async fn capture_real_mock_writes_manifest_and_frames() {
    let temp_dir = temp_path("pete_capture_real_mock");
    let args = CaptureRealArgs {
        duration_seconds: 1,
        out: temp_dir.to_string_lossy().to_string(),
        ledger: None,
        tick_ms: 1000,
        cockpit: CockpitBackendArg::Sim,
        create_port: "mock".to_string(),
        brainstem_host: "192.168.4.1:80".to_string(),
        brainstem_local: "127.0.0.1:8787".parse().unwrap(),
        create_baud: 57_600,
        camera: None,
        kinect_depth: false,
        kinect_index: 0,
        kinect_rgb_target_luma: 0.32,
        kinect_rgb_auto_gain_max: 3.0,
        kinect_rgb_gain: 1.0,
        kinect_rgb_gamma: 0.80,
        kinect_rgb_brightness: 0.0,
        kinect_rgb_raw: false,
        mic: None,
        imu: None,
        gps: None,
        lidar: None,
        lidar_yaw_deg: 0.0,
        lidar_pitch_deg: 0.0,
        lidar_roll_deg: 0.0,
        lidar_height_m: 0.0,
        lidar_forward_m: 0.0,
        lidar_left_m: 0.0,
        mock: true,
        export_rgb: false,
        export_depth: false,
        export_audio: false,
        export_pointcloud: false,
        pointcloud_stride: 4,
        llm: LlmArgs::default(),
    };

    capture_real(args).await.unwrap();
    assert!(temp_dir.join("manifest.json").exists());
    assert!(temp_dir.join("frames.jsonl").exists());
    let report = inspect_capture_report(&temp_dir).await.unwrap();
    assert_eq!(report.frame_count, 1);
    assert!(report.streams_present.contains(&"body".to_string()));
    assert!(report.streams_present.contains(&"audio".to_string()));
    assert!(report.streams_present.contains(&"depth".to_string()));
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
#[ignore = "slow capture-real mock path can stall workspace test runs"]
async fn capture_real_mock_exports_assets_and_pointclouds() {
    let temp_dir = temp_path("pete_capture_real_mock_assets");
    let args = CaptureRealArgs {
        duration_seconds: 1,
        out: temp_dir.to_string_lossy().to_string(),
        ledger: None,
        tick_ms: 1000,
        cockpit: CockpitBackendArg::Sim,
        create_port: "mock".to_string(),
        brainstem_host: "192.168.4.1:80".to_string(),
        brainstem_local: "127.0.0.1:8787".parse().unwrap(),
        create_baud: 57_600,
        camera: None,
        kinect_depth: false,
        kinect_index: 0,
        kinect_rgb_target_luma: 0.32,
        kinect_rgb_auto_gain_max: 3.0,
        kinect_rgb_gain: 1.0,
        kinect_rgb_gamma: 0.80,
        kinect_rgb_brightness: 0.0,
        kinect_rgb_raw: false,
        mic: None,
        imu: None,
        gps: None,
        lidar: None,
        lidar_yaw_deg: 0.0,
        lidar_pitch_deg: 0.0,
        lidar_roll_deg: 0.0,
        lidar_height_m: 0.0,
        lidar_forward_m: 0.0,
        lidar_left_m: 0.0,
        mock: true,
        export_rgb: true,
        export_depth: true,
        export_audio: true,
        export_pointcloud: false,
        pointcloud_stride: 4,
        llm: LlmArgs::default(),
    };

    capture_real(args).await.unwrap();
    capture_assets(CaptureAssetsArgs {
        capture: temp_dir.to_string_lossy().to_string(),
        pointcloud: true,
        world_pointcloud: true,
        stride: 1,
        max_depth_m: 8.0,
    })
    .await
    .unwrap();

    let report = inspect_capture_report(&temp_dir).await.unwrap();
    assert_eq!(
        report.asset_counts,
        vec![
            ("rgb".to_string(), 1),
            ("depth".to_string(), 1),
            ("audio".to_string(), 1),
            ("pointcloud".to_string(), 2)
        ]
    );
    assert!(report
        .asset_details
        .iter()
        .any(|detail| detail.contains("rgb metadata: 2x2")));
    assert!(report
        .warnings
        .iter()
        .any(|warning| warning.contains("uncalibrated point cloud")));
    let world_ply = temp_dir.join("assets/pointcloud/world-accumulated.ply");
    let world_ply_text = fs::read_to_string(&world_ply).unwrap();
    assert!(world_ply_text.contains("property float confidence"));
    assert!(world_ply_text.contains("property uchar stable"));
    let replay = replay_capture(ReplayCaptureArgs {
        capture: temp_dir.to_string_lossy().to_string(),
        ledger: temp_dir.join("ledger").to_string_lossy().to_string(),
        llm: LlmArgs::default(),
    })
    .await;
    assert!(replay.is_ok());
    let _ = fs::remove_dir_all(&temp_dir);
}
