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

