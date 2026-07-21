use super::*;
use pete_autonomic::SimpleSafety;
use pete_body::BodySense;
use pete_conductor::SimpleConductor;
use pete_ledger::JsonlLedger;
use pete_llm::NoopLlmAgent;
use pete_memory::InMemoryExperienceStore;
use pete_runtime::{MinimalRuntime, SimRunner};
use pete_sim::{ArenaConfig, VirtualWorld};
use tempfile::tempdir;

#[test]
fn capture_manifest_round_trips() {
    let manifest = CaptureManifest {
        id: "round-trip".to_string(),
        created_at_ms: 123,
        source: CaptureSource::Sim,
        schema_version: CAPTURE_SCHEMA_VERSION,
        frame_count: 2,
        tick_ms: Some(100),
        notes: vec!["small and sturdy".to_string()],
        machine: None,
        firmware_identity: Some(serde_json::json!({"build_id": "0.1.7+g1a2b3c4"})),
        brainstem_safety: Some(serde_json::json!({
            "class": "independent-watchdog",
            "independent_watchdog": true
        })),
        command_args: Vec::new(),
        device_availability: Value::Null,
        streams: CaptureStreams::default(),
        started_at_ms: Some(123),
        ended_at_ms: Some(456),
        warnings: Vec::new(),
        asset_layout: default_asset_layout(),
        scenario: None,
        writer_health: CaptureWriterHealth::default(),
    };

    let encoded = serde_json::to_string(&manifest).unwrap();
    let decoded: CaptureManifest = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, manifest);
}

#[test]
fn capture_asset_paths_are_deterministic() {
    assert_eq!(capture_asset_path("rgb", 7, "png"), "assets/rgb/000007.png");
    assert_eq!(
        capture_asset_path("depth", 42, "depth16.png"),
        "assets/depth/000042.depth16.png"
    );
}

#[test]
fn capture_frame_records_require_asset_and_stream_metadata_fields() {
    let encoded = serde_json::json!({
        "index": 0,
        "t_ms": 123,
        "snapshot": WorldSnapshot::default(),
        "events": []
    });

    assert!(serde_json::from_value::<CaptureFrameRecord>(encoded).is_err());
}

#[test]
fn frame_asset_references_serialize() {
    let record = CaptureFrameRecord {
        index: 1,
        t_ms: 1234,
        snapshot: WorldSnapshot::default(),
        events: Vec::new(),
        assets: CaptureFrameAssets {
            rgb: Some("assets/rgb/000001.png".to_string()),
            depth: Some("assets/depth/000001.depth16.png".to_string()),
            audio: None,
            pointcloud: Some("assets/pointcloud/000001.ply".to_string()),
            perception: None,
            ..CaptureFrameAssets::default()
        },
        stream_metadata: Some(serde_json::json!({"rgb": {"width": 2, "height": 2}})),
    };

    let encoded = serde_json::to_value(&record).unwrap();

    assert_eq!(encoded["assets"]["rgb"], "assets/rgb/000001.png");
    assert_eq!(
        encoded["assets"]["pointcloud"],
        "assets/pointcloud/000001.ply"
    );
    assert_eq!(encoded["stream_metadata"]["rgb"]["width"], 2);
}

#[test]
fn perception_export_writes_frame_json_asset() {
    let dir = tempdir().unwrap();
    let mut frame = CaptureFrameRecord {
        index: 123,
        t_ms: 456,
        snapshot: WorldSnapshot::default(),
        events: Vec::new(),
        assets: CaptureFrameAssets::default(),
        stream_metadata: None,
    };
    frame.snapshot.kinect.depth_width = 2;
    frame.snapshot.kinect.depth_height = 1;
    frame.snapshot.kinect.depth_m = vec![1.0, 2.0];
    frame.snapshot.kinect.depth_fx = 1.0;
    frame.snapshot.kinect.depth_fy = 1.0;
    frame.snapshot.kinect.min_depth_m = 0.1;
    frame.snapshot.kinect.max_depth_m = 8.0;

    let metadata = export_perception_for_frame(dir.path(), &mut frame)
        .unwrap()
        .unwrap();
    let rel = frame.assets.perception.as_deref().unwrap();
    let encoded = std_fs::read_to_string(dir.path().join(rel)).unwrap();
    let decoded: PerceptionFrame = serde_json::from_str(&encoded).unwrap();

    assert_eq!(rel, "assets/perception/000123.json");
    assert_eq!(metadata["perception"]["points"], 2);
    assert_eq!(decoded.points[1].depth.index, 1);
    assert_eq!(decoded.points[1].depth.u, 1);
}

#[test]
fn tiny_depth_map_exports_expected_ply_vertices() {
    let dir = tempdir().unwrap();
    let depth = DepthImage {
        width: 2,
        height: 2,
        units: DepthUnits::Millimeters,
        values_mm: vec![1000, 0, 2000, 9000],
    };
    let path = dir.path().join("tiny.ply");

    let vertices = write_pointcloud_ply(
        &path,
        &depth,
        CameraIntrinsics {
            fx: 2.0,
            fy: 2.0,
            cx: 0.5,
            cy: 0.5,
        },
        3.0,
        1,
    )
    .unwrap();
    let ply = std_fs::read_to_string(path).unwrap();

    assert_eq!(vertices, 2);
    assert!(ply.contains("element vertex 2"));
    let vertex_rows = ply
        .lines()
        .skip_while(|line| *line != "end_header")
        .skip(1)
        .count();
    assert_eq!(vertex_rows, 2);
}

#[test]
fn snapshot_depth_image_preserves_declared_dimensions() {
    let mut snapshot = WorldSnapshot::default();
    snapshot.kinect.depth_width = 2;
    snapshot.kinect.depth_height = 2;
    snapshot.kinect.depth_m = vec![1.0, 0.0, f32::NAN, 65.536];

    let depth = snapshot_depth_image(&snapshot).unwrap();

    assert_eq!(depth.width, 2);
    assert_eq!(depth.height, 2);
    assert_eq!(depth.values_mm, vec![1000, 0, 0, u16::MAX]);
}

#[test]
fn snapshot_depth_image_rejects_invalid_dimensions() {
    let mut snapshot = WorldSnapshot::default();
    snapshot.kinect.depth_width = 2;
    snapshot.kinect.depth_height = 2;
    snapshot.kinect.depth_m = vec![1.0, 2.0, 3.0];

    assert!(snapshot_depth_image(&snapshot).is_none());
}

#[tokio::test]
async fn capture_writer_writes_manifest_and_frames() {
    let dir = tempdir().unwrap();
    let mut writer = CaptureWriter::create(dir.path(), CaptureSource::Sim, Some(100))
        .await
        .unwrap();
    writer
        .append_snapshot(10, WorldSnapshot::default(), Vec::new())
        .await
        .unwrap();
    let manifest = writer.finish().await.unwrap();

    assert_eq!(manifest.frame_count, 1);
    assert!(dir.path().join(MANIFEST_FILE).exists());
    let frames = fs::read_to_string(dir.path().join(FRAMES_FILE))
        .await
        .unwrap();
    assert_eq!(frames.lines().count(), 1);
}

#[tokio::test]
async fn background_writer_exports_timestamped_checksummed_raw_streams() {
    let dir = tempdir().unwrap();
    let mut writer = CaptureWriter::create(dir.path(), CaptureSource::RealRobot, Some(100))
        .await
        .unwrap();
    let mut snapshot = WorldSnapshot::default();
    snapshot.eye_frame = Some(pete_now::EyeFrame {
        captured_at_ms: 990,
        rgbd_frame_id: Some("rgbd-1".to_string()),
        device_timestamp_ms: Some(12),
        width: 2,
        height: 1,
        format: EyeFrameFormat::Rgb8,
        bytes: vec![255, 0, 0, 0, 255, 0],
        source: Some("test-camera".to_string()),
    });
    snapshot.kinect.depth_width = 2;
    snapshot.kinect.depth_height = 1;
    snapshot.kinect.depth_m = vec![1.0, 2.0];
    snapshot.kinect.captured_at_ms = 992;
    snapshot.kinect.geometry_calibration = Some(pete_now::DepthGeometryCalibration {
        calibrated: true,
        depth: pete_now::CameraIntrinsics {
            width: 2,
            height: 1,
            fx: 2.0,
            fy: 2.0,
            cx: 0.5,
            cy: 0.0,
            distortion: [0.0; 5],
        },
        depth_scale: 1.0,
        ..pete_now::DepthGeometryCalibration::default()
    });
    snapshot.range.captured_at_ms = 995;
    snapshot.range.beams = vec![0.5, 0.75];
    snapshot.range.beam_time_offsets_ms = vec![-10, 0];
    snapshot.range.source = Some("lfcd2".to_string());
    snapshot.imu.captured_at_ms = 996;
    snapshot.imu.orientation = vec![0.0, 0.0];
    snapshot.ear_pcm = Some(PcmAudioFrame {
        captured_at_ms: 997,
        sample_rate_hz: 16_000,
        channels: 1,
        samples: vec![1, -1, 2, -2],
    });

    writer
        .append_snapshot_with_exported_assets_and_context(
            1_000,
            snapshot,
            Vec::new(),
            true,
            true,
            true,
            CaptureExportContext {
                imu_selection: Some(serde_json::json!({
                    "selected_source": "test-imu",
                    "candidates": [{"source_id": "test-imu", "sample": {"captured_at_ms": 996}}]
                })),
            },
        )
        .await
        .unwrap();
    let manifest = writer.finish().await.unwrap();
    let raw_record: Value = serde_json::from_str(
        std_fs::read_to_string(dir.path().join(FRAMES_FILE))
            .unwrap()
            .lines()
            .next()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(raw_record["snapshot"]["eye_frame"]["bytes"], serde_json::json!([]));
    assert_eq!(raw_record["snapshot"]["kinect"]["depth_m"], serde_json::json!([]));
    assert_eq!(raw_record["snapshot"]["ear_pcm"]["samples"], serde_json::json!([]));
    assert_eq!(raw_record["snapshot"]["range"]["beams"], serde_json::json!([]));
    let reader = CaptureReader::open(dir.path()).await.unwrap();
    let frames = reader.read_frames().await.unwrap();
    let frame = &frames[0];

    assert_eq!(manifest.writer_health.submitted_frames, 1);
    assert_eq!(manifest.writer_health.written_frames, 1);
    assert_eq!(manifest.writer_health.dropped_frames, 0);
    for kind in [
        "rgb",
        "camera",
        "depth",
        "lidar",
        "imu",
        "audio",
        "calibration",
        "pointcloud",
    ] {
        let metadata = &frame.stream_metadata.as_ref().unwrap()[kind];
        assert_eq!(metadata["status"], "written", "{kind}");
        assert_eq!(metadata["capture_t_ms"], 1_000, "{kind}");
        assert!(metadata["bytes"].as_u64().unwrap_or_default() > 0, "{kind}");
        assert_eq!(
            metadata["sha256"].as_str().unwrap_or_default().len(),
            64,
            "{kind}"
        );
    }
    assert!(frame
        .assets
        .lidar
        .as_ref()
        .is_some_and(|path| dir.path().join(path).exists()));
    assert!(frame
        .assets
        .imu
        .as_ref()
        .is_some_and(|path| dir.path().join(path).exists()));
    assert!(frame
        .assets
        .pointcloud
        .as_ref()
        .is_some_and(|path| dir.path().join(path).exists()));
    assert_eq!(frame.snapshot.kinect.depth_m, vec![1.0, 2.0]);
    assert_eq!(frame.snapshot.ear_pcm.as_ref().unwrap().samples, vec![1, -1, 2, -2]);
    assert_eq!(frame.snapshot.range.beams, vec![0.5, 0.75]);

    let expected_pointcloud_sha = frame.stream_metadata.as_ref().unwrap()["pointcloud"]["sha256"]
        .as_str()
        .unwrap()
        .to_string();
    let mut regenerated = frame.clone();
    export_pointcloud_for_frame(dir.path(), &mut regenerated, 8.0, 1)
        .unwrap()
        .unwrap();
    assert_eq!(
        sha256_file(dir.path().join(regenerated.assets.pointcloud.unwrap()).as_path()).unwrap(),
        expected_pointcloud_sha
    );
}

#[tokio::test]
async fn background_writer_drops_when_bounded_queue_is_full() {
    let dir = tempdir().unwrap();
    let mut writer = CaptureWriter::create(dir.path(), CaptureSource::RealRobot, Some(10))
        .await
        .unwrap();
    writer.set_background_write_delay(std::time::Duration::from_millis(50));

    let submitted = CAPTURE_QUEUE_CAPACITY as u64 + 12;
    for index in 0..submitted {
        writer
            .append_snapshot_with_exported_assets(
                index * 10,
                WorldSnapshot::default(),
                Vec::new(),
                true,
                true,
                true,
            )
            .await
            .unwrap();
    }
    let manifest = writer.finish().await.unwrap();

    assert_eq!(manifest.writer_health.submitted_frames, submitted);
    assert!(manifest.writer_health.dropped_frames > 0);
    assert_eq!(
        manifest.writer_health.written_frames + manifest.writer_health.dropped_frames,
        submitted
    );
    assert!(manifest
        .writer_health
        .dropped_assets
        .get("depth")
        .is_some_and(|count| *count > 0));
    assert!(manifest
        .warnings
        .iter()
        .any(|warning| warning.contains("writer queue dropped")));
}

#[tokio::test]
async fn capture_reader_reads_frames_in_order() {
    let dir = tempdir().unwrap();
    let mut writer = CaptureWriter::create(dir.path(), CaptureSource::Sim, Some(100))
        .await
        .unwrap();
    for t_ms in [100, 200, 300] {
        writer
            .append_snapshot(t_ms, WorldSnapshot::default(), Vec::new())
            .await
            .unwrap();
    }
    writer.finish().await.unwrap();

    let reader = CaptureReader::open(dir.path()).await.unwrap();
    let frames = reader.read_frames().await.unwrap();

    assert_eq!(
        frames.iter().map(|frame| frame.index).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(
        frames.iter().map(|frame| frame.t_ms).collect::<Vec<_>>(),
        vec![100, 200, 300]
    );
}

#[test]
fn serializable_snapshot_to_now_preserves_body_battery() {
    let mut snapshot = SerializableWorldSnapshot::default();
    snapshot.body = BodySense {
        battery_level: 0.42,
        last_update_ms: 500,
        ..BodySense::default()
    };

    let now = snapshot.to_now(600);

    assert_eq!(now.t_ms, 600);
    assert_eq!(now.body.battery_level, 0.42);
}

#[tokio::test]
async fn capture_sim_creates_nonempty_session() {
    let capture_dir = tempdir().unwrap();
    let ledger_dir = tempdir().unwrap();
    let runtime = test_runtime(ledger_dir.path());
    let (world, motors) = VirtualWorld::new_with_motor(
        7,
        ArenaConfig {
            width_m: 4.0,
            height_m: 4.0,
        },
    );
    let mut runner = SimRunner::new(runtime, world, motors);
    let mut snapshots = Vec::new();
    runner
        .run_steps_observing(3, |snapshot| snapshots.push(snapshot.clone()))
        .await
        .unwrap();

    let mut writer = CaptureWriter::create(capture_dir.path(), CaptureSource::Sim, Some(100))
        .await
        .unwrap();
    for snapshot in snapshots {
        writer
            .append_snapshot(snapshot.body.last_update_ms, snapshot, Vec::new())
            .await
            .unwrap();
    }
    let manifest = writer.finish().await.unwrap();
    let reader = CaptureReader::open(capture_dir.path()).await.unwrap();

    assert_eq!(manifest.frame_count, 3);
    assert_eq!(reader.read_frames().await.unwrap().len(), 3);
}

#[tokio::test]
async fn replay_capture_produces_runtime_ticks() {
    let capture_dir = tempdir().unwrap();
    let ledger_dir = tempdir().unwrap();
    let mut writer = CaptureWriter::create(capture_dir.path(), CaptureSource::Sim, Some(100))
        .await
        .unwrap();
    for t_ms in [100, 200, 300] {
        let mut snapshot = WorldSnapshot::default();
        snapshot.body.last_update_ms = t_ms;
        snapshot.body.battery_level = 1.0 - (t_ms as f32 / 1_000.0);
        writer
            .append_snapshot(t_ms, snapshot, Vec::new())
            .await
            .unwrap();
    }
    writer.finish().await.unwrap();

    let reader = CaptureReader::open(capture_dir.path()).await.unwrap();
    let ledger = JsonlLedger::new(ledger_dir.path());
    let runtime = test_runtime(ledger_dir.path());
    let mut runner = CaptureReplayRunner::new(runtime, reader);
    let summary = runner.replay().await.unwrap();
    let transitions = ledger.transitions().await.unwrap();

    assert_eq!(summary.frames_replayed, 3);
    assert_eq!(summary.runtime_ticks, 3);
    assert_eq!(transitions.len(), 2);
}

fn test_runtime(
    ledger_path: impl Into<std::path::PathBuf>,
) -> MinimalRuntime<
    JsonlLedger,
    InMemoryExperienceStore,
    InMemoryExperienceStore,
    SimpleConductor,
    SimpleSafety,
    NoopLlmAgent,
> {
    let memory = InMemoryExperienceStore::new();
    MinimalRuntime::with_default_events(
        JsonlLedger::new(ledger_path),
        memory.clone(),
        memory,
        SimpleConductor::default(),
        SimpleSafety::default(),
        NoopLlmAgent,
    )
}
