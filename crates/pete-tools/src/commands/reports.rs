#[derive(Debug, Serialize, Deserialize)]
struct VirtualRunReport {
    pub total_frames: usize,
    pub total_transitions: usize,
    pub total_eye_frames: usize,
    pub total_ear_frames: usize,
    pub total_stuck_trap_events: usize,
    pub battery_delta: f32,
    pub duration_seconds: f64,
    pub eye_sources: HashMap<String, usize>,
    pub retina_coverage: f32,
    pub collisions: usize,
    pub collision_rate: f32,
    pub charger_contacts: usize,
    pub charging_ticks: usize,
    pub battery_recovery_success: bool,
    pub stuck_recovery_attempts: usize,
    pub stuck_recovery_successes: usize,
    pub trap_kinds: HashMap<String, usize>,
    pub ledger_gaps: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VirtualTrainingReport {
    pub timestamp: String,
    pub run_report: VirtualRunReport,
    pub models: HashMap<String, ModelTrainingStatus>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelTrainingStatus {
    pub name: String,
    pub trained: bool,
    pub previous_status: String,
    pub new_status: String,
    pub recommended_action: String,
    pub warnings: Vec<String>,
    pub loss: Option<f32>,
    pub baseline_collision_rate: Option<f32>,
    pub candidate_collision_rate: Option<f32>,
    pub baseline_success_rate: Option<f32>,
    pub candidate_success_rate: Option<f32>,
}

async fn run_virtual_report(args: VirtualReportArgs) -> Result<()> {
    let report = generate_virtual_report(&args.ledger).await?;
    let parent = Path::new(&args.out).parent();
    if let Some(p) = parent {
        if !p.as_os_str().is_empty() {
            fs::create_dir_all(p)?;
        }
    }
    let content = serde_json::to_string_pretty(&report)?;
    fs::write(&args.out, content)?;
    println!("virtual run report written to {}", args.out);
    Ok(())
}

async fn run_pose_graph_report(args: PoseGraphReportArgs) -> Result<()> {
    let report = generate_pose_graph_report(&args).await?;
    let parent = Path::new(&args.out).parent();
    if let Some(p) = parent {
        if !p.as_os_str().is_empty() {
            fs::create_dir_all(p)?;
        }
    }
    fs::write(&args.out, serde_json::to_string_pretty(&report)?)?;
    println!(
        "pose graph report written to {} (nodes={}, odometry_edges={}, loop_candidates={}, rejected={})",
        args.out,
        report.nodes,
        report.odometry_edges,
        report.loop_candidate_edges,
        report.rejected_loop_candidates
    );
    Ok(())
}

async fn run_geometry_debug(args: GeometryDebugArgs) -> Result<()> {
    let report = generate_geometry_debug_report(&args).await?;
    if let Some(parent) = Path::new(&args.out).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "geometry debug report written to {} (frame={}, below_floor_ratio={:.3}, ready_for_real_slam={})",
        args.out,
        report.frame_index,
        report.floor_statistics.below_floor_ratio,
        report.sensor_truth.ready_for_real_slam
    );
    Ok(())
}

#[derive(Debug, Serialize)]
struct GeometryDebugReport {
    schema_version: u32,
    input_source: String,
    frame_index: u64,
    t_ms: u64,
    sensor_truth: SensorTruthReport,
    depth_projection: GeometryDepthProjection,
    calibration_extrinsics: GeometryExtrinsics,
    imu_orientation: GeometryImuInterpretation,
    timestamp_diagnostics: GeometryTimestampDiagnostics,
    stationary_rotation_diagnostics: StationaryRotationDiagnostics,
    coordinate_frame_conventions: Vec<String>,
    sample_transformed_points: Vec<GeometryPointSample>,
    floor_statistics: GeometryFloorStatistics,
    warnings: Vec<String>,
    hard_failures: Vec<String>,
}

#[derive(Debug, Serialize)]
struct GeometryDepthProjection {
    width: usize,
    height: usize,
    vector_len: usize,
    projection_source: String,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    min_depth_m: Option<f32>,
    median_depth_m: Option<f32>,
    max_depth_m: Option<f32>,
    skipped_depth_count: usize,
    clipped_depth_count: usize,
    sample_stride: usize,
}

#[derive(Debug, Serialize)]
struct SensorTruthReport {
    ready_for_real_slam: bool,
    gates: Vec<SensorTruthGate>,
}

#[derive(Debug, Serialize)]
struct SensorTruthGate {
    name: String,
    status: SensorTruthStatus,
    detail: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum SensorTruthStatus {
    Pass,
    Fail,
    NotApplicable,
}

#[derive(Debug, Serialize)]
struct GeometryExtrinsics {
    camera_height_m: f32,
    camera_forward_m: f32,
    camera_pitch_rad: f32,
    camera_roll_rad: f32,
    camera_yaw_rad: f32,
    rotation_order: String,
    base_mapping: String,
}

#[derive(Debug, Serialize)]
struct GeometryImuInterpretation {
    raw_orientation: Vec<f32>,
    assumed_units: String,
    assumed_axis_order: String,
    roll_deg: Option<f32>,
    pitch_deg: Option<f32>,
    yaw_deg: Option<f32>,
    roll_pitch_correction_active: bool,
    yaw_source: String,
    contract_known: bool,
    contract_source: String,
    note: String,
}

#[derive(Debug, Serialize)]
struct GeometryTimestampDiagnostics {
    frame_count: usize,
    depth_frame_count: usize,
    first_frame_t_ms: Option<u64>,
    last_frame_t_ms: Option<u64>,
    frame_timestamps_monotonic: bool,
    median_frame_dt_ms: Option<u64>,
    max_frame_dt_ms: Option<u64>,
    body_last_update_age_ms: Option<u64>,
    eye_frame_age_ms: Option<u64>,
    ear_pcm_age_ms: Option<u64>,
    kinect_capture_timestamp_present: bool,
    kinect_capture_age_ms: Option<u64>,
    imu_capture_timestamp_present: bool,
    imu_capture_age_ms: Option<u64>,
    note: String,
}

#[derive(Debug, Serialize)]
struct StationaryRotationDiagnostics {
    evaluated: bool,
    reason: String,
    frame_count: usize,
    heading_delta_deg: f32,
    translation_delta_m: f32,
    raw_points_seen: u64,
    voxel_count: usize,
    stable_voxel_count: usize,
    stable_voxel_ratio: f32,
    stable_z_span_m: Option<f32>,
    stable_z_median_m: Option<f32>,
}

#[derive(Debug, Serialize)]
struct GeometryPointSample {
    pixel_index: usize,
    u: usize,
    v: usize,
    depth_m: f32,
    camera_frame: Point3D,
    robot_base_frame: Point3D,
    world_frame: Point3D,
    render_scene_frame: Point3D,
}

#[derive(Debug, Serialize)]
struct GeometryFloorStatistics {
    sampled_points: usize,
    below_floor_count: usize,
    below_floor_ratio: f32,
    robot_base_below_floor_count: usize,
    robot_base_below_floor_ratio: f32,
    min_robot_base_z_m: Option<f32>,
    median_robot_base_z_m: Option<f32>,
    max_robot_base_z_m: Option<f32>,
    min_math_frame_z_m: Option<f32>,
    median_math_frame_z_m: Option<f32>,
    max_math_frame_z_m: Option<f32>,
    min_render_vertical_axis_m: Option<f32>,
    median_render_vertical_axis_m: Option<f32>,
    max_render_vertical_axis_m: Option<f32>,
}

async fn generate_geometry_debug_report(args: &GeometryDebugArgs) -> Result<GeometryDebugReport> {
    let input = geometry_debug_input(args).await?;
    let frames = input.frames;
    let record = frames
        .iter()
        .find(|frame| !frame.snapshot.kinect.depth_m.is_empty())
        .context("capture contains no frame with Kinect/depth data")?;
    let snapshot = &record.snapshot;
    let kinect = &snapshot.kinect;
    let mut warnings = Vec::new();
    let mut hard_failures = Vec::new();
    let projection = geometry_projection(kinect, &mut warnings);
    let config = PointCloudConfig::default();
    let orientation = pete_map::orientation_from_snapshot(snapshot);
    let imu_interpretation = geometry_imu_interpretation(&snapshot.imu.orientation, orientation);
    let pose = snapshot.body.odometry;
    let timestamp_diagnostics = geometry_timestamp_diagnostics(&frames, record);
    let stationary_rotation_diagnostics = stationary_rotation_diagnostics(&frames, args);
    let sample_stride = kinect.depth_m.len().div_ceil(args.samples.max(1)).max(1);
    let min_depth = positive_depth_or(kinect.min_depth_m, config.min_depth_m);
    let max_depth = positive_depth_or(kinect.max_depth_m, config.max_depth_m);
    let mut full_skipped = 0usize;
    let mut full_clipped = 0usize;
    let mut full_valid_depths = Vec::new();
    for depth in &kinect.depth_m {
        if !depth.is_finite() || *depth <= 0.0 {
            full_skipped = full_skipped.saturating_add(1);
        } else if *depth < min_depth || *depth > max_depth {
            full_clipped = full_clipped.saturating_add(1);
        } else {
            full_valid_depths.push(*depth);
        }
    }
    full_valid_depths.sort_by(|a, b| a.total_cmp(b));
    let mut samples = Vec::new();
    let mut robot_heights = Vec::new();
    let mut world_heights = Vec::new();
    let mut render_heights = Vec::new();
    for (index, depth) in kinect.depth_m.iter().enumerate() {
        if !depth.is_finite() || *depth <= 0.0 {
            continue;
        }
        if *depth < min_depth || *depth > max_depth {
            continue;
        }
        let u = index % projection.width;
        let v = index / projection.width;
        let camera = Point3D {
            x_m: (u as f32 - projection.cx) * *depth / projection.fx.max(f32::EPSILON),
            y_m: (v as f32 - projection.cy) * *depth / projection.fy.max(f32::EPSILON),
            z_m: *depth,
        };
        let robot = geometry_camera_to_robot(camera, config);
        let world = transform_point_to_world(camera, projection.frame, pose, orientation, config);
        let render = Point3D {
            x_m: robot.y_m,
            y_m: robot.z_m,
            z_m: robot.x_m,
        };
        robot_heights.push(robot.z_m);
        world_heights.push(world.z_m);
        render_heights.push(render.y_m);
        if samples.len() < args.samples && index % sample_stride == 0 {
            samples.push(GeometryPointSample {
                pixel_index: index,
                u,
                v,
                depth_m: *depth,
                camera_frame: camera,
                robot_base_frame: robot,
                world_frame: world,
                render_scene_frame: render,
            });
        }
    }
    let robot_base_below_floor_count = robot_heights.iter().filter(|z| **z < 0.0).count();
    let robot_base_below_floor_ratio = if robot_heights.is_empty() {
        0.0
    } else {
        robot_base_below_floor_count as f32 / robot_heights.len() as f32
    };
    let below_floor_count = world_heights.iter().filter(|z| **z < 0.0).count();
    let below_floor_ratio = if world_heights.is_empty() {
        0.0
    } else {
        below_floor_count as f32 / world_heights.len() as f32
    };
    if projection.source_is_fallback {
        warnings.push(
            "fallback intrinsics/projection are active; real Kinect geometry is not trustworthy"
                .to_string(),
        );
    }
    if !kinect
        .depth_coordinate_system
        .as_deref()
        .is_some_and(|s| s == "kinect_camera")
    {
        warnings.push(format!(
            "depth coordinate system metadata is {:?}; assuming Kinect camera frame",
            kinect.depth_coordinate_system
        ));
    }
    if !imu_interpretation.contract_known {
        warnings.push(format!(
            "IMU orientation contract is not sufficient for roll/pitch correction: {}",
            imu_interpretation.note
        ));
    }
    let sensor_truth = sensor_truth_report(
        projection.source_is_fallback,
        &timestamp_diagnostics,
        &imu_interpretation,
        below_floor_ratio,
        &stationary_rotation_diagnostics,
        args,
    );
    for gate in &sensor_truth.gates {
        if gate.status == SensorTruthStatus::Fail {
            hard_failures.push(format!("{}: {}", gate.name, gate.detail));
        }
    }
    Ok(GeometryDebugReport {
        schema_version: 1,
        input_source: input.source,
        frame_index: record.index,
        t_ms: record.t_ms,
        sensor_truth,
        depth_projection: GeometryDepthProjection {
            width: projection.width,
            height: projection.height,
            vector_len: kinect.depth_m.len(),
            projection_source: projection.source,
            fx: projection.fx,
            fy: projection.fy,
            cx: projection.cx,
            cy: projection.cy,
            min_depth_m: full_valid_depths.first().copied(),
            median_depth_m: median_sorted(&full_valid_depths),
            max_depth_m: full_valid_depths.last().copied(),
            skipped_depth_count: full_skipped,
            clipped_depth_count: full_clipped,
            sample_stride,
        },
        calibration_extrinsics: GeometryExtrinsics {
            camera_height_m: config.camera_height_m,
            camera_forward_m: config.camera_forward_m,
            camera_pitch_rad: config.camera_pitch_rad,
            camera_roll_rad: config.camera_roll_rad,
            camera_yaw_rad: config.camera_yaw_rad,
            rotation_order: "camera -> base [z,-x,-y], then pitch, roll, yaw, then translate".to_string(),
            base_mapping: "Kinect camera +x right, +y down, +z forward -> robot +x forward, +y left, +z up".to_string(),
        },
        imu_orientation: imu_interpretation,
        timestamp_diagnostics,
        stationary_rotation_diagnostics,
        coordinate_frame_conventions: vec![
            "Kinect camera frame: +x right, +y down, +z forward".to_string(),
            "Robot/base math frame: +x forward, +y left, +z up; floor is z=0".to_string(),
            "World/odometry math frame: +x odom forward/east, +y odom left/north, +z up".to_string(),
            "Scene render frame: Babylon +x world x/left-local, +y up, +z world y; robot forward is local -z".to_string(),
            "ScenePoint for calibrated live depth is scene_robot_render: x=robot_y, y=robot_z, z=robot_x".to_string(),
        ],
        sample_transformed_points: samples,
        floor_statistics: GeometryFloorStatistics {
            sampled_points: world_heights.len(),
            below_floor_count,
            below_floor_ratio,
            robot_base_below_floor_count,
            robot_base_below_floor_ratio,
            min_robot_base_z_m: min_sorted(robot_heights.clone()),
            median_robot_base_z_m: median_values(robot_heights.clone()),
            max_robot_base_z_m: max_sorted(robot_heights.clone()),
            min_math_frame_z_m: min_sorted(world_heights.clone()),
            median_math_frame_z_m: median_values(world_heights.clone()),
            max_math_frame_z_m: max_sorted(world_heights.clone()),
            min_render_vertical_axis_m: min_sorted(render_heights.clone()),
            median_render_vertical_axis_m: median_values(render_heights.clone()),
            max_render_vertical_axis_m: max_sorted(render_heights),
        },
        warnings,
        hard_failures,
    })
}

struct GeometryDebugInput {
    source: String,
    frames: Vec<pete_worldlab::CaptureFrameRecord>,
}

async fn geometry_debug_input(args: &GeometryDebugArgs) -> Result<GeometryDebugInput> {
    match (&args.capture, &args.live_now_url) {
        (Some(_), Some(_)) => {
            anyhow::bail!("pass only one of --capture or --live-now-url");
        }
        (Some(capture), None) => {
            let reader = CaptureReader::open(capture).await?;
            Ok(GeometryDebugInput {
                source: format!("capture:{capture}"),
                frames: reader.read_frames().await?,
            })
        }
        (None, Some(url)) => {
            let now = reqwest::Client::new()
                .get(url)
                .send()
                .await
                .with_context(|| format!("fetching live Now snapshot from {url}"))?
                .error_for_status()
                .with_context(|| format!("live Now endpoint returned an error for {url}"))?
                .json::<Now>()
                .await
                .with_context(|| format!("decoding live Now JSON from {url}"))?;
            let t_ms = now.t_ms;
            Ok(GeometryDebugInput {
                source: format!("live-now:{url}"),
                frames: vec![pete_worldlab::CaptureFrameRecord {
                    index: 0,
                    t_ms,
                    snapshot: world_snapshot_from_now(now),
                    events: Vec::new(),
                    assets: pete_worldlab::CaptureFrameAssets::default(),
                    stream_metadata: Some(serde_json::json!({
                        "source": "live_now_url",
                        "url": url
                    })),
                }],
            })
        }
        (None, None) => {
            anyhow::bail!("pass --capture <dir> or --live-now-url <url>");
        }
    }
}

fn world_snapshot_from_now(now: Now) -> WorldSnapshot {
    WorldSnapshot {
        body: now.body,
        eye_frame: now.eye_frame,
        eye: now.eye,
        ear: now.ear,
        range: now.range,
        imu: now.imu,
        gps: now.gps,
        kinect: now.kinect,
        objects: now.objects,
        face: now.face,
        voice: now.voice,
        extensions: Vec::new(),
        ..WorldSnapshot::default()
    }
}

struct GeometryProjection {
    width: usize,
    height: usize,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    frame: PointCloudFrame,
    source: String,
    source_is_fallback: bool,
}

fn geometry_projection(kinect: &KinectSense, warnings: &mut Vec<String>) -> GeometryProjection {
    let width = usize::try_from(kinect.depth_width).unwrap_or(0);
    let height = usize::try_from(kinect.depth_height).unwrap_or(0);
    if width > 0 && height > 0 && width.saturating_mul(height) == kinect.depth_m.len() {
        return GeometryProjection {
            width,
            height,
            fx: positive_depth_or(kinect.depth_fx, 594.0),
            fy: positive_depth_or(kinect.depth_fy, 591.0),
            cx: positive_depth_or(kinect.depth_cx, (width as f32 - 1.0) * 0.5),
            cy: positive_depth_or(kinect.depth_cy, (height as f32 - 1.0) * 0.5),
            frame: PointCloudFrame::KinectCamera,
            source: if kinect.depth_fx > 0.0 && kinect.depth_fy > 0.0 {
                "real_intrinsics".to_string()
            } else {
                "metadata_dimensions_with_fallback_intrinsics".to_string()
            },
            source_is_fallback: !(kinect.depth_fx > 0.0 && kinect.depth_fy > 0.0),
        };
    }
    warnings.push(
        "depth width/height do not match vector length; using legacy square fallback projection"
            .to_string(),
    );
    let width = (kinect.depth_m.len() as f32).sqrt().ceil().max(1.0) as usize;
    GeometryProjection {
        width,
        height: kinect.depth_m.len().div_ceil(width).max(1),
        fx: width as f32,
        fy: width as f32,
        cx: (width as f32 - 1.0) * 0.5,
        cy: (kinect.depth_m.len().div_ceil(width).max(1) as f32 - 1.0) * 0.5,
        frame: PointCloudFrame::DepthImageUnknown,
        source: "fallback_legacy_square_projection".to_string(),
        source_is_fallback: true,
    }
}

fn sensor_truth_report(
    depth_projection_is_fallback: bool,
    timestamps: &GeometryTimestampDiagnostics,
    imu: &GeometryImuInterpretation,
    below_floor_ratio: f32,
    stationary: &StationaryRotationDiagnostics,
    args: &GeometryDebugArgs,
) -> SensorTruthReport {
    let mut gates = Vec::new();
    gates.push(SensorTruthGate {
        name: "depth_intrinsics_non_fallback".to_string(),
        status: if depth_projection_is_fallback {
            SensorTruthStatus::Fail
        } else {
            SensorTruthStatus::Pass
        },
        detail: if depth_projection_is_fallback {
            "depth projection is using fallback intrinsics/projection".to_string()
        } else {
            "depth frame carries usable width/height and fx/fy intrinsics".to_string()
        },
    });
    gates.push(SensorTruthGate {
        name: "below_floor_ratio".to_string(),
        status: if below_floor_ratio <= args.max_below_floor_ratio {
            SensorTruthStatus::Pass
        } else {
            SensorTruthStatus::Fail
        },
        detail: format!(
            "below_floor_ratio={below_floor_ratio:.4}, threshold={:.4}",
            args.max_below_floor_ratio
        ),
    });
    gates.push(SensorTruthGate {
        name: "frame_timestamps_monotonic".to_string(),
        status: if timestamps.frame_timestamps_monotonic {
            SensorTruthStatus::Pass
        } else {
            SensorTruthStatus::Fail
        },
        detail: format!(
            "frames={}, median_dt_ms={:?}, max_dt_ms={:?}",
            timestamps.frame_count, timestamps.median_frame_dt_ms, timestamps.max_frame_dt_ms
        ),
    });
    let body_age_ok = timestamps
        .body_last_update_age_ms
        .is_some_and(|age| age <= args.max_body_timestamp_age_ms);
    gates.push(SensorTruthGate {
        name: "body_timestamp_fresh".to_string(),
        status: if body_age_ok {
            SensorTruthStatus::Pass
        } else {
            SensorTruthStatus::Fail
        },
        detail: format!(
            "body_last_update_age_ms={:?}, threshold={}",
            timestamps.body_last_update_age_ms, args.max_body_timestamp_age_ms
        ),
    });
    gates.push(SensorTruthGate {
        name: "kinect_timestamp_carried".to_string(),
        status: if timestamps.kinect_capture_timestamp_present {
            SensorTruthStatus::Pass
        } else {
            SensorTruthStatus::Fail
        },
        detail: format!(
            "kinect_captured_at_ms_present={}, age_ms={:?}",
            timestamps.kinect_capture_timestamp_present, timestamps.kinect_capture_age_ms
        ),
    });
    gates.push(SensorTruthGate {
        name: "imu_timestamp_carried".to_string(),
        status: if timestamps.imu_capture_timestamp_present {
            SensorTruthStatus::Pass
        } else {
            SensorTruthStatus::Fail
        },
        detail: format!(
            "imu_captured_at_ms_present={}, age_ms={:?}",
            timestamps.imu_capture_timestamp_present, timestamps.imu_capture_age_ms
        ),
    });
    let imu_ready = imu.contract_known && imu.roll_pitch_correction_active;
    gates.push(SensorTruthGate {
        name: "imu_roll_pitch_contract".to_string(),
        status: if imu_ready {
            SensorTruthStatus::Pass
        } else {
            SensorTruthStatus::Fail
        },
        detail: format!(
            "{}; roll_pitch_active={}",
            imu.contract_source, imu.roll_pitch_correction_active
        ),
    });
    let stationary_status = if !stationary.evaluated {
        SensorTruthStatus::NotApplicable
    } else if stationary.stable_voxel_ratio >= args.min_stationary_stable_voxel_ratio
        && stationary
            .stable_z_span_m
            .is_some_and(|span| span <= args.max_stationary_stable_z_span_m)
    {
        SensorTruthStatus::Pass
    } else {
        SensorTruthStatus::Fail
    };
    gates.push(SensorTruthGate {
        name: "stationary_rotation_cloud_stability".to_string(),
        status: stationary_status,
        detail: format!(
            "{}; heading_delta_deg={:.1}, translation_delta_m={:.3}, stable_ratio={:.3}, stable_z_span_m={:?}",
            stationary.reason,
            stationary.heading_delta_deg,
            stationary.translation_delta_m,
            stationary.stable_voxel_ratio,
            stationary.stable_z_span_m
        ),
    });
    let ready_for_real_slam = gates
        .iter()
        .all(|gate| gate.status == SensorTruthStatus::Pass);
    SensorTruthReport {
        ready_for_real_slam,
        gates,
    }
}

fn geometry_timestamp_diagnostics(
    frames: &[pete_worldlab::CaptureFrameRecord],
    selected: &pete_worldlab::CaptureFrameRecord,
) -> GeometryTimestampDiagnostics {
    let mut deltas = frames
        .windows(2)
        .map(|pair| pair[1].t_ms.saturating_sub(pair[0].t_ms))
        .collect::<Vec<_>>();
    deltas.sort_unstable();
    let frame_timestamps_monotonic = frames.windows(2).all(|pair| pair[1].t_ms >= pair[0].t_ms);
    let median_frame_dt_ms = if deltas.is_empty() {
        None
    } else {
        Some(deltas[deltas.len() / 2])
    };
    let max_frame_dt_ms = deltas.last().copied();
    let snapshot = &selected.snapshot;
    GeometryTimestampDiagnostics {
        frame_count: frames.len(),
        depth_frame_count: frames
            .iter()
            .filter(|frame| !frame.snapshot.kinect.depth_m.is_empty())
            .count(),
        first_frame_t_ms: frames.first().map(|frame| frame.t_ms),
        last_frame_t_ms: frames.last().map(|frame| frame.t_ms),
        frame_timestamps_monotonic,
        median_frame_dt_ms,
        max_frame_dt_ms,
        body_last_update_age_ms: Some(selected.t_ms.saturating_sub(snapshot.body.last_update_ms)),
        eye_frame_age_ms: snapshot
            .eye_frame
            .as_ref()
            .map(|frame| selected.t_ms.saturating_sub(frame.captured_at_ms)),
        ear_pcm_age_ms: snapshot
            .ear_pcm
            .as_ref()
            .map(|frame| selected.t_ms.saturating_sub(frame.captured_at_ms)),
        kinect_capture_timestamp_present: snapshot.kinect.captured_at_ms > 0,
        kinect_capture_age_ms: (snapshot.kinect.captured_at_ms > 0)
            .then(|| selected.t_ms.saturating_sub(snapshot.kinect.captured_at_ms)),
        imu_capture_timestamp_present: snapshot.imu.captured_at_ms > 0,
        imu_capture_age_ms: (snapshot.imu.captured_at_ms > 0)
            .then(|| selected.t_ms.saturating_sub(snapshot.imu.captured_at_ms)),
        note: "KinectSense and ImuSense carry individual capture timestamps when produced by current sensor providers; old captures may deserialize as 0 and fail these gates".to_string(),
    }
}

fn stationary_rotation_diagnostics(
    frames: &[pete_worldlab::CaptureFrameRecord],
    args: &GeometryDebugArgs,
) -> StationaryRotationDiagnostics {
    let depth_frames = frames
        .iter()
        .filter(|frame| !frame.snapshot.kinect.depth_m.is_empty())
        .collect::<Vec<_>>();
    if depth_frames.len() < 2 {
        return StationaryRotationDiagnostics {
            evaluated: false,
            reason: "capture has fewer than two depth frames".to_string(),
            frame_count: depth_frames.len(),
            heading_delta_deg: 0.0,
            translation_delta_m: 0.0,
            raw_points_seen: 0,
            voxel_count: 0,
            stable_voxel_count: 0,
            stable_voxel_ratio: 0.0,
            stable_z_span_m: None,
            stable_z_median_m: None,
        };
    }
    let first = depth_frames.first().unwrap();
    let last = depth_frames.last().unwrap();
    let first_pose = first.snapshot.body.odometry;
    let last_pose = last.snapshot.body.odometry;
    let heading_delta_deg =
        angle_delta_abs(last_pose.heading_rad, first_pose.heading_rad).to_degrees();
    let translation_delta_m = ((last_pose.x_m - first_pose.x_m).powi(2)
        + (last_pose.y_m - first_pose.y_m).powi(2))
    .sqrt();
    let stationary_candidate = heading_delta_deg >= args.min_stationary_rotation_deg
        && translation_delta_m <= args.max_stationary_translation_m;
    let mut cloud = VoxelPointCloud::default();
    for frame in &depth_frames {
        cloud.observe_snapshot(&frame.snapshot, frame.t_ms);
    }
    let summary = cloud.summary();
    let stable_z_values = cloud
        .points()
        .into_iter()
        .filter(|point| point.stable)
        .map(|point| point.position.z_m)
        .filter(|z| z.is_finite())
        .collect::<Vec<_>>();
    let stable_z_span_m = min_max_values(&stable_z_values).map(|(min, max)| max - min);
    let stable_z_median_m = median_values(stable_z_values.clone());
    let stable_voxel_ratio = if summary.voxels == 0 {
        0.0
    } else {
        summary.stable_voxels as f32 / summary.voxels as f32
    };
    StationaryRotationDiagnostics {
        evaluated: stationary_candidate,
        reason: if stationary_candidate {
            "capture looks like a stationary rotation test".to_string()
        } else {
            format!(
                "capture is not a stationary rotation test; requires heading_delta>={:.1}deg and translation<={:.2}m",
                args.min_stationary_rotation_deg, args.max_stationary_translation_m
            )
        },
        frame_count: depth_frames.len(),
        heading_delta_deg,
        translation_delta_m,
        raw_points_seen: summary.raw_points_seen,
        voxel_count: summary.voxels,
        stable_voxel_count: summary.stable_voxels,
        stable_voxel_ratio,
        stable_z_span_m,
        stable_z_median_m,
    }
}

fn geometry_camera_to_robot(point: Point3D, config: PointCloudConfig) -> Point3D {
    let base = Point3D {
        x_m: point.z_m,
        y_m: -point.x_m,
        z_m: -point.y_m,
    };
    let rotated = geometry_rotate_robot_extrinsic(
        base,
        config.camera_pitch_rad,
        config.camera_roll_rad,
        config.camera_yaw_rad,
    );
    Point3D {
        x_m: rotated.x_m + config.camera_forward_m,
        y_m: rotated.y_m,
        z_m: rotated.z_m + config.camera_height_m,
    }
}

fn geometry_rotate_robot_extrinsic(
    point: Point3D,
    pitch_rad: f32,
    roll_rad: f32,
    yaw_rad: f32,
) -> Point3D {
    let (pitch_sin, pitch_cos) = pitch_rad.sin_cos();
    let x = point.x_m * pitch_cos + point.z_m * pitch_sin;
    let y = point.y_m;
    let mut z = -point.x_m * pitch_sin + point.z_m * pitch_cos;
    let (roll_sin, roll_cos) = roll_rad.sin_cos();
    let rolled_y = y * roll_cos - z * roll_sin;
    z = y * roll_sin + z * roll_cos;
    let (yaw_sin, yaw_cos) = yaw_rad.sin_cos();
    Point3D {
        x_m: x * yaw_cos - rolled_y * yaw_sin,
        y_m: x * yaw_sin + rolled_y * yaw_cos,
        z_m: z,
    }
}

fn geometry_imu_interpretation(
    raw: &[f32],
    orientation: OrientationEstimate,
) -> GeometryImuInterpretation {
    let (contract_known, contract_source, note) = match raw.len() {
        2 => (
            true,
            "recognized MPU-6050 hardware shape: [roll, pitch] radians from gravity".to_string(),
            "hardware MPU-6050 has roll/pitch but no absolute yaw; yaw should come from odometry"
                .to_string(),
        ),
        3.. => (
            true,
            "recognized full orientation shape: [roll, pitch, yaw] radians".to_string(),
            "full orientation vector supplies roll/pitch/yaw in Pete order".to_string(),
        ),
        1 => (
            false,
            "invalid one-value orientation vector".to_string(),
            "one-value IMU orientation is ignored; yaw falls back to odometry".to_string(),
        ),
        _ => (
            false,
            "no orientation vector".to_string(),
            "no IMU orientation was present in this frame".to_string(),
        ),
    };
    GeometryImuInterpretation {
        raw_orientation: raw.to_vec(),
        assumed_units: "radians".to_string(),
        assumed_axis_order: match raw.len() {
            0 => "none".to_string(),
            1 => "invalid one-value vector".to_string(),
            2 => "[roll, pitch]".to_string(),
            _ => "[roll, pitch, yaw]".to_string(),
        },
        roll_deg: orientation.roll_rad.map(f32::to_degrees),
        pitch_deg: orientation.pitch_rad.map(f32::to_degrees),
        yaw_deg: orientation.yaw_rad.map(f32::to_degrees),
        roll_pitch_correction_active: orientation.roll_pitch_from_imu,
        yaw_source: format!("{:?}", orientation.yaw_source),
        contract_known,
        contract_source,
        note,
    }
}

fn positive_depth_or(value: f32, fallback: f32) -> f32 {
    if value > 0.0 {
        value
    } else {
        fallback
    }
}

fn median_values(mut values: Vec<f32>) -> Option<f32> {
    values.sort_by(|a, b| a.total_cmp(b));
    median_sorted(&values)
}

fn median_sorted(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        None
    } else {
        Some(values[values.len() / 2])
    }
}

fn min_sorted(mut values: Vec<f32>) -> Option<f32> {
    values.sort_by(|a, b| a.total_cmp(b));
    values.first().copied()
}

fn max_sorted(mut values: Vec<f32>) -> Option<f32> {
    values.sort_by(|a, b| a.total_cmp(b));
    values.last().copied()
}

fn min_max_values(values: &[f32]) -> Option<(f32, f32)> {
    let mut iter = values.iter().copied().filter(|value| value.is_finite());
    let first = iter.next()?;
    let mut min = first;
    let mut max = first;
    for value in iter {
        min = min.min(value);
        max = max.max(value);
    }
    Some((min, max))
}

fn angle_delta_abs(left: f32, right: f32) -> f32 {
    let mut delta = left - right;
    while delta > std::f32::consts::PI {
        delta -= std::f32::consts::TAU;
    }
    while delta < -std::f32::consts::PI {
        delta += std::f32::consts::TAU;
    }
    delta.abs()
}

async fn run_representation_report(args: RepresentationReportArgs) -> Result<()> {
    let report = generate_representation_report(&args).await?;
    if let Some(parent) = Path::new(&args.out).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "representation report written to {} (frames={}, entities={}, voxels={})",
        args.out,
        report.frame_count,
        report.entity_memory.total_entities,
        report.map.point_cloud_voxel_count
    );
    Ok(())
}

async fn generate_representation_report(
    args: &RepresentationReportArgs,
) -> Result<RepresentationHealthReport> {
    let mut warnings = BTreeSet::new();
    let mut provenance: HashMap<String, usize> = HashMap::new();
    let mut place_memory = PlaceMemory::new();
    let mut entity_memory = EntityMemory::new();
    let mut local_map = LocalMap::default();
    let mut point_cloud = VoxelPointCloud::default();
    let mut pose_graph = PoseGraphBuilder::new(PoseGraphConfig::default());
    let mut place_candidates = Vec::new();
    let mut place_recognition_warnings = BTreeSet::new();
    let mut frame_count = 0usize;

    let mut saw_range = false;
    let mut saw_scene_vectors = false;
    let mut saw_objects = false;
    let mut saw_depth = false;
    let mut saw_audio = false;

    let input = if let Some(capture) = args.capture.as_deref() {
        let reader = CaptureReader::open(capture).await?;
        let mut records = reader.read_frames().await?;
        records.sort_by_key(|record| record.t_ms);
        for record in records {
            frame_count += 1;
            *provenance
                .entry("capture_snapshot".to_string())
                .or_default() += 1;
            let frame_id = format!("capture-frame-{}", record.index);
            let mut now = record.snapshot.to_now(record.t_ms);
            set_now_frame_id(&mut now, &frame_id);
            saw_range |= !now.range.beams.is_empty() || now.range.nearest_m.is_some();
            saw_scene_vectors |= !now.eye.scene_vectors.is_empty();
            saw_objects |= !now.objects.observations.is_empty();
            saw_depth |= !record.snapshot.kinect.depth_m.is_empty();
            saw_audio |= !now.ear.features.is_empty() || now.ear.transcript.is_some();

            let current_key =
                Some(place_memory.quantize(now.body.odometry.x_m, now.body.odometry.y_m));
            let live_loop_candidates = live_loop_candidates_from_now(
                &place_memory,
                &now,
                current_key,
                Some(frame_id.clone()),
            );
            place_memory.observe_now(&now);
            entity_memory.observe_now(&now, current_key);
            let map_observation = observation_from_now(&now, local_map.config);
            local_map
                .integrate_observation_with_loop_candidates(map_observation, &live_loop_candidates);
            point_cloud.observe_snapshot(&record.snapshot, record.t_ms);
            observe_pose_graph_now(&mut pose_graph, &mut place_memory, &now, Some(frame_id));

            let output =
                place_memory.recognize_places_report(current_key, &now.eye.scene_vectors, 0.0, 20);
            if let Some(reason) = output.not_enough_evidence {
                place_recognition_warnings.insert(reason);
            }
            place_candidates.extend(output.candidates);
        }
        RepresentationInputSummary {
            source_type: "capture".to_string(),
            source_path: capture.to_string(),
            provenance,
        }
    } else {
        let ledger = JsonlLedger::new(&args.ledger);
        let mut frames = ledger.range(0, u64::MAX).await?;
        frames.sort_by_key(|frame| frame.t_ms);
        for frame in &frames {
            frame_count += 1;
            let place_input = place_recognition_input_from_frame(frame);
            *provenance
                .entry(place_input.provenance.clone())
                .or_default() += 1;

            let now = &frame.now;
            saw_range |= !now.range.beams.is_empty() || now.range.nearest_m.is_some();
            saw_scene_vectors |= !now.eye.scene_vectors.is_empty();
            saw_objects |= !now.objects.observations.is_empty();
            saw_depth |= !now.kinect.depth_m.is_empty();
            saw_audio |= !now.ear.features.is_empty() || now.ear.transcript.is_some();

            let current_key =
                Some(place_memory.quantize(now.body.odometry.x_m, now.body.odometry.y_m));
            let live_loop_candidates =
                live_loop_candidates_from_frame(&place_memory, frame, current_key);
            place_memory.observe_frame(frame);
            entity_memory.observe_now(now, current_key);
            let map_now = now_with_frame_id(now, &frame.id.to_string());
            let map_observation = observation_from_now(&map_now, local_map.config);
            local_map
                .integrate_observation_with_loop_candidates(map_observation, &live_loop_candidates);
            point_cloud.decay_stale(now.t_ms);
            observe_pose_graph_frame(&mut pose_graph, &mut place_memory, frame);

            let mut query_vectors = now.eye.scene_vectors.clone();
            query_vectors.extend(place_recognition_vectors_from_input(&place_input));
            let output = place_memory.recognize_places_report(current_key, &query_vectors, 0.0, 20);
            if let Some(reason) = output.not_enough_evidence {
                place_recognition_warnings.insert(reason);
            }
            place_candidates.extend(output.candidates);
        }
        RepresentationInputSummary {
            source_type: "ledger".to_string(),
            source_path: args.ledger.clone(),
            provenance,
        }
    };

    if frame_count == 0 {
        warnings.insert("no frames found in input".to_string());
    }
    if !saw_range {
        warnings.insert("range sensor data missing across all frames".to_string());
    }
    if !saw_scene_vectors {
        warnings.insert("scene vectors missing across all frames".to_string());
    }
    if !saw_objects {
        warnings.insert("object observations missing across all frames".to_string());
    }
    if !saw_depth {
        warnings.insert("depth channel missing across all frames".to_string());
    }
    if !saw_audio {
        warnings.insert("audio/transcript channel missing across all frames".to_string());
    }

    let entity_report = entity_memory.report();
    let revived_entities = entity_memory
        .entities
        .values()
        .filter(|entity| entity.constellation.state == EntityConstellationState::Revived)
        .count();
    let mut modality_support_counts = HashMap::new();
    let mut constellation_edges_by_relation = HashMap::new();
    for entity in entity_memory.entities.values() {
        if !entity.modality_support.face_vector_ids.is_empty() {
            *modality_support_counts
                .entry("face".to_string())
                .or_default() += 1;
        }
        if !entity.modality_support.voice_vector_ids.is_empty() {
            *modality_support_counts
                .entry("voice".to_string())
                .or_default() += 1;
        }
        if !entity.modality_support.scene_vector_ids.is_empty() {
            *modality_support_counts
                .entry("scene".to_string())
                .or_default() += 1;
        }
        if !entity.modality_support.text_labels.is_empty() {
            *modality_support_counts
                .entry("text".to_string())
                .or_default() += 1;
        }
        for edge in &entity.constellation.binding_edges {
            *constellation_edges_by_relation
                .entry(binding_relation_label(edge.relation.clone()).to_string())
                .or_default() += 1;
        }
    }

    let map_summary = local_map.summary();
    let point_cloud_summary = point_cloud.summary();
    if point_cloud_summary.observations == 0 {
        warnings.insert("point cloud received no usable observations".to_string());
    }

    let pose_graph_report = pose_graph.finish_report();
    let confidence_values = place_candidates
        .iter()
        .map(|candidate| candidate.confidence)
        .collect::<Vec<_>>();
    let mut candidate_kinds = HashMap::new();
    let mut same_place_cells: HashMap<(i32, i32), usize> = HashMap::new();
    for candidate in &place_candidates {
        let kind = match &candidate.kind {
            PlaceRecognitionKind::SamePlace => "same_place",
            PlaceRecognitionKind::SimilarPlace => "similar_place",
            PlaceRecognitionKind::EntityConstellation => "entity_constellation",
        };
        *candidate_kinds.entry(kind.to_string()).or_default() += 1;
        if matches!(&candidate.kind, PlaceRecognitionKind::SamePlace) {
            *same_place_cells
                .entry((candidate.cell.x, candidate.cell.y))
                .or_default() += 1;
        }
    }
    let repeated_place_hints = same_place_cells
        .iter()
        .filter(|(_, count)| **count > 1)
        .map(|((x, y), count)| format!("cell ({x}, {y}) recognized {count} times"))
        .take(5)
        .collect::<Vec<_>>();
    if place_candidates.is_empty() {
        place_recognition_warnings.insert("no place-recognition candidates emitted".to_string());
    }

    Ok(RepresentationHealthReport {
        schema_version: 1,
        frame_count,
        input,
        warnings: warnings.into_iter().collect(),
        entity_memory: RepresentationEntityMemorySummary {
            total_entities: entity_report.total_entities,
            active_entities: entity_report.active_entities,
            occluded_entities: entity_report.occluded_entities,
            vanished_entities: entity_report.vanished_entities,
            revived_entities,
            modality_support_counts,
            constellation_edges_by_relation,
        },
        map: RepresentationMapSummary {
            local_occupancy_cell_count: map_summary.occupied_cells,
            pose_history_length: local_map.pose_history.len(),
            point_cloud_voxel_count: point_cloud_summary.voxels,
            stable_voxel_count: point_cloud_summary.stable_voxels,
            transient_voxel_count: point_cloud_summary.transient_voxels,
        },
        pose_graph: RepresentationPoseGraphSummary {
            node_count: pose_graph_report.nodes,
            odometry_edge_count: pose_graph_report.odometry_edges,
            loop_candidate_count: pose_graph_report.loop_candidate_edges,
            loop_accepted_count: pose_graph_report.active_loop_candidate_edges,
            loop_rejected_count: pose_graph_report.rejected_loop_candidates,
            confidence_distribution: RepresentationConfidenceDistribution {
                min: pose_graph_report.confidence_distribution.min,
                max: pose_graph_report.confidence_distribution.max,
                mean: pose_graph_report.confidence_distribution.mean,
                buckets: pose_graph_report
                    .confidence_distribution
                    .buckets
                    .into_iter()
                    .collect(),
            },
        },
        place_recognition: RepresentationPlaceRecognitionSummary {
            candidates_emitted: place_candidates.len(),
            candidate_kinds,
            confidence_distribution: summarize_confidence_distribution(&confidence_values),
            repeated_place_hints,
            warnings: place_recognition_warnings.into_iter().collect(),
        },
    })
}

fn summarize_confidence_distribution(values: &[f32]) -> RepresentationConfidenceDistribution {
    let mut buckets = HashMap::new();
    for value in values {
        let bucket = if *value < 0.25 {
            "0.00-0.24"
        } else if *value < 0.5 {
            "0.25-0.49"
        } else if *value < 0.75 {
            "0.50-0.74"
        } else {
            "0.75-1.00"
        };
        *buckets.entry(bucket.to_string()).or_default() += 1;
    }
    let min = values
        .iter()
        .copied()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let max = values
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = (!values.is_empty()).then_some(values.iter().sum::<f32>() / values.len() as f32);
    RepresentationConfidenceDistribution {
        min,
        max,
        mean,
        buckets,
    }
}

fn binding_relation_label(relation: BindingRelation) -> &'static str {
    match relation {
        BindingRelation::CooccursInTime => "cooccurs_in_time",
        BindingRelation::CooccursInEstimatedSpace => "cooccurs_in_estimated_space",
        BindingRelation::MovesTogether => "moves_together",
        BindingRelation::PredictsSameFutureEvents => "predicts_same_future_events",
        BindingRelation::NamedBy => "named_by",
        BindingRelation::ProjectsTo => "projects_to",
        BindingRelation::HasColorAtPose => "has_color_at_pose",
        BindingRelation::LikelySameEntity => "likely_same_entity",
        BindingRelation::ExplainsOutcome => "explains_outcome",
        BindingRelation::Contradicts => "contradicts",
        BindingRelation::RequiresReview => "requires_review",
    }
}

async fn generate_pose_graph_report(args: &PoseGraphReportArgs) -> Result<PoseGraphReport> {
    let config = PoseGraphConfig {
        min_node_distance_m: args.min_node_distance_m,
        min_node_heading_delta_rad: args.min_node_degrees.to_radians(),
        max_ticks_between_nodes: args.max_ticks_between_nodes,
        min_loop_confidence: args.min_loop_confidence,
        ..PoseGraphConfig::default()
    };
    let mut builder = PoseGraphBuilder::new(config);
    let mut memory = PlaceMemory::new();

    if let Some(capture) = args.capture.as_deref() {
        let reader = CaptureReader::open(capture).await?;
        let mut records = reader.read_frames().await?;
        records.sort_by_key(|record| record.t_ms);
        for record in &records {
            let frame_id = format!("capture-frame-{}", record.index);
            let mut now = record.snapshot.to_now(record.t_ms);
            set_now_frame_id(&mut now, &frame_id);
            observe_pose_graph_now(&mut builder, &mut memory, &now, Some(frame_id));
            memory.observe_now(&now);
        }
    } else {
        let ledger = JsonlLedger::new(&args.ledger);
        let mut frames = ledger.range(0, u64::MAX).await?;
        frames.sort_by_key(|frame| frame.t_ms);
        for frame in &frames {
            observe_pose_graph_frame(&mut builder, &mut memory, frame);
            memory.observe_frame(frame);
        }
    }

    Ok(builder.finish_report())
}

fn observe_pose_graph_now(
    builder: &mut PoseGraphBuilder,
    memory: &mut PlaceMemory,
    now: &Now,
    source_frame_id: Option<String>,
) {
    let current_key = Some(memory.quantize(now.body.odometry.x_m, now.body.odometry.y_m));
    let place_candidates = memory.recognize_places(current_key, &now.eye.scene_vectors, 0.0, 20);
    let entity_labels = entity_labels_from_now(now);
    let entity_candidates =
        memory.recognize_entity_constellations(current_key, &entity_labels, 0.0, 10);
    let loop_candidates = place_candidates
        .iter()
        .chain(entity_candidates.iter())
        .map(|candidate| place_candidate_to_loop_input(candidate, source_frame_id.clone()))
        .collect::<Vec<_>>();
    builder.observe(
        now.body.odometry,
        now.t_ms,
        source_frame_id,
        &loop_candidates,
    );
}

fn live_loop_candidates_from_now(
    memory: &PlaceMemory,
    now: &Now,
    current_key: Option<pete_memory::PlaceCellKey>,
    source_frame_id: Option<String>,
) -> Vec<LoopClosureCandidateInput> {
    let place_candidates = memory.recognize_places(current_key, &now.eye.scene_vectors, 0.85, 10);
    let entity_labels = entity_labels_from_now(now);
    let entity_candidates =
        memory.recognize_entity_constellations(current_key, &entity_labels, 0.85, 10);
    place_candidates
        .iter()
        .chain(entity_candidates.iter())
        .map(|candidate| place_candidate_to_loop_input(candidate, source_frame_id.clone()))
        .collect()
}

fn observe_pose_graph_frame(
    builder: &mut PoseGraphBuilder,
    memory: &mut PlaceMemory,
    frame: &ExperienceFrame,
) {
    let current_key =
        Some(memory.quantize(frame.now.body.odometry.x_m, frame.now.body.odometry.y_m));
    let place_input = place_recognition_input_from_frame(frame);
    let mut query_vectors = frame.now.eye.scene_vectors.clone();
    query_vectors.extend(place_recognition_vectors_from_input(&place_input));
    let place_candidates = memory.recognize_places(current_key, &query_vectors, 0.0, 20);
    let entity_labels = entity_labels_from_place_input(&place_input);
    let entity_candidates =
        memory.recognize_entity_constellations(current_key, &entity_labels, 0.0, 10);
    let loop_candidates = place_candidates
        .iter()
        .chain(entity_candidates.iter())
        .map(|candidate| place_candidate_to_loop_input(candidate, Some(frame.id.to_string())))
        .collect::<Vec<_>>();
    builder.observe(
        frame.now.body.odometry,
        frame.t_ms,
        Some(frame.id.to_string()),
        &loop_candidates,
    );
}

fn live_loop_candidates_from_frame(
    memory: &PlaceMemory,
    frame: &ExperienceFrame,
    current_key: Option<pete_memory::PlaceCellKey>,
) -> Vec<LoopClosureCandidateInput> {
    let place_input = place_recognition_input_from_frame(frame);
    let mut query_vectors = frame.now.eye.scene_vectors.clone();
    query_vectors.extend(place_recognition_vectors_from_input(&place_input));
    let place_candidates = memory.recognize_places(current_key, &query_vectors, 0.85, 10);
    let entity_labels = entity_labels_from_place_input(&place_input);
    let entity_candidates =
        memory.recognize_entity_constellations(current_key, &entity_labels, 0.85, 10);
    place_candidates
        .iter()
        .chain(entity_candidates.iter())
        .map(|candidate| place_candidate_to_loop_input(candidate, Some(frame.id.to_string())))
        .collect()
}

fn now_with_frame_id(now: &Now, frame_id: &str) -> Now {
    let mut now = now.clone();
    set_now_frame_id(&mut now, frame_id);
    now
}

fn set_now_frame_id(now: &mut Now, frame_id: &str) {
    now.extensions.insert(
        "frame_id".to_string(),
        serde_json::Value::String(frame_id.to_string()),
    );
}

fn entity_labels_from_now(now: &Now) -> Vec<String> {
    let mut labels: Vec<String> = now
        .objects
        .observations
        .iter()
        .filter(|obs| obs.confidence >= 0.3)
        .map(|obs| obs.label.clone())
        .collect();
    labels.sort();
    labels.dedup();
    labels
}

fn entity_labels_from_place_input(input: &pete_memory::PlaceRecognitionInput) -> Vec<String> {
    let mut labels: Vec<String> = input
        .object_labels
        .iter()
        .chain(input.person_labels.iter())
        .cloned()
        .collect();
    labels.sort();
    labels.dedup();
    labels
}

fn place_candidate_to_loop_input(
    candidate: &PlaceRecognitionCandidate,
    source_frame_id: Option<String>,
) -> LoopClosureCandidateInput {
    LoopClosureCandidateInput {
        target_pose: pete_core::Pose2 {
            x_m: candidate.cell.center_x_m,
            y_m: candidate.cell.center_y_m,
            heading_rad: 0.0,
        },
        confidence: candidate.confidence,
        similarity: candidate.similarity,
        kind: match candidate.kind {
            PlaceRecognitionKind::SamePlace => "same_place",
            PlaceRecognitionKind::SimilarPlace => "similar_place",
            PlaceRecognitionKind::EntityConstellation => "entity_constellation",
        }
        .to_string(),
        target_frame_id: candidate
            .source_instant_frame_id
            .clone()
            .or_else(|| candidate.source_frame_id.clone()),
        source_frame_id,
        source_experience_id: candidate.source_experience_id.clone(),
        source_instant_frame_id: candidate.source_instant_frame_id.clone(),
        source_vector_refs: candidate.source_vector_refs.clone(),
        source_vector_id: Some(candidate.source_vector_id.clone()),
        query_vector_id: candidate.query_vector_id.clone(),
        query_experience_id: candidate.query_experience_id.clone(),
    }
}

async fn run_embodied_demo(args: EmbodiedDemoArgs) -> Result<()> {
    let now_ms = Utc::now().timestamp_millis().max(0) as u64;
    let demo = pete_experience::demo_embodied_experience(now_ms).await?;
    let mut impressions = demo.impressions.clone();
    if let Some(summary) = demo.experience.summary_impression.clone() {
        impressions.push(summary);
    }

    if let Some(root) = args.ledger.as_deref() {
        let ledger = JsonlLedger::new(root);
        let frame = ExperienceFrame {
            id: uuid::Uuid::new_v4(),
            t_ms: now_ms,
            now: Now::blank(now_ms, BodySense::default()),
            sensations: demo.sensations.clone(),
            impressions: impressions.clone(),
            experiences: vec![demo.experience.clone()],
            z: None,
            chosen_action: None,
            conscious_command: None,
            reign_input: None,
            reign_outcome: None,
            predicted_futures: Vec::new(),
            behavior_runs: Vec::new(),
            actual_next: None,
            reward: Default::default(),
            surprise: SurpriseSense::default(),
            memory_recall: Vec::new(),
            recollections: Vec::new(),
            llm_teaching: Vec::new(),
            counterfactuals: Vec::new(),
            notes: vec!["embodied demo pipeline".to_string()],
        };
        ledger.append(&frame).await?;
        println!("wrote embodied demo frame to {}", root);
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&demo)?);
        return Ok(());
    }

    println!("embodied experience {}", demo.experience.id);
    println!("  summary: {}", demo.experience.text);
    println!(
        "  vector coverage: image={} face={} voice={} transcript={} impression={} experience={} fallback_count={}",
        demo.coverage.image,
        demo.coverage.face,
        demo.coverage.voice,
        demo.coverage.transcript,
        demo.coverage.impression,
        demo.coverage.experience,
        demo.coverage.fallback_count
    );
    println!("  sensations: {}", demo.sensations.len());
    for sensation in &demo.sensations {
        let vector = sensation
            .vector
            .as_ref()
            .map(|embedding| {
                format!(
                    "{}d {} purpose={} vectorizer={} fallback={}",
                    embedding.dim,
                    embedding.model_id,
                    embedding.purpose,
                    embedding.vectorizer_id,
                    embedding.is_fallback
                )
            })
            .unwrap_or_else(|| "none".to_string());
        println!(
            "    - {} {:?}/{:?} parent={:?} vector={}",
            sensation.kind, sensation.modality, sensation.payload_kind, sensation.parent_id, vector
        );
    }
    println!("  impressions:");
    for impression in &impressions {
        let vector = impression
            .vector
            .as_ref()
            .map(|embedding| {
                format!(
                    "{}d {} purpose={} vectorizer={} fallback={}",
                    embedding.dim,
                    embedding.model_id,
                    embedding.purpose,
                    embedding.vectorizer_id,
                    embedding.is_fallback
                )
            })
            .unwrap_or_else(|| "none".to_string());
        println!("    - {} vector={}", impression.text, vector);
    }
    Ok(())
}

async fn run_embodied_eval(args: EmbodiedEvalArgs) -> Result<()> {
    match args.fixture {
        EmbodiedEvalFixtureArg::Deterministic => {}
    }
    let omissions = args
        .omit
        .into_iter()
        .map(EmbodiedEvalOmission::from)
        .collect::<Vec<_>>();
    let report = deterministic_embodied_eval_report_with_omissions(&omissions).await?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("embodied eval fixture={}", report.fixture);
        println!("  frames: {}", report.frame_count);
        println!("  instants: {}", report.instant_count);
        println!(
            "  instant teacher vectors: {}",
            report.instant_teacher_vector_count
        );
        println!(
            "  instant missing modalities: {}",
            report.instant_missing_modality_count
        );
        println!("  primary sensations: {}", report.primary_sensation_count);
        println!(
            "  descendant sensations: {}",
            report.descendant_sensation_count
        );
        println!(
            "  vectorized sensations: {}",
            report.vectorized_sensation_count
        );
        println!("  impressions: {}", report.impression_count);
        println!("  summary impressions: {}", report.summary_impression_count);
        println!(
            "  learned experience latents: {}",
            report.experience_latent_count
        );
        println!("  predictions: {}", report.prediction_count);
        println!("  memory links: {}", report.memory_link_count);
        println!("  recall sensations: {}", report.recall_sensation_count);
        println!("  recall impressions: {}", report.recall_impression_count);
        println!("  lineage edges: {}", report.lineage_edge_count);
        if !report.warnings.is_empty() {
            println!("  warnings:");
            for warning in &report.warnings {
                println!("    - {warning}");
            }
        }
        if !report.failures.is_empty() {
            println!("  failures:");
            for failure in &report.failures {
                println!("    - {failure}");
            }
        }
    }

    if report.passed() {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "embodied eval failed: {}",
            report.failures.join(", ")
        ))
    }
}

async fn generate_virtual_report(ledger_path: &str) -> Result<VirtualRunReport> {
    let ledger = JsonlLedger::new(ledger_path);
    let frames = ledger.frames().await?;
    let transitions = ledger.transitions().await?;

    let total_frames = frames.len();
    let total_transitions = transitions.len();

    let mut total_eye_frames = 0;
    let mut total_ear_frames = 0;
    let mut total_stuck_trap_events = 0;

    let mut eye_sources = HashMap::new();
    let mut babylon_eye_frames = 0;
    let mut collisions = 0;
    let mut charging_ticks = 0;
    let mut charger_contacts = 0;
    let mut was_charging = false;
    let mut stuck_recovery_attempts = 0;
    let mut stuck_recovery_successes = 0;
    let mut trap_kinds = HashMap::new();
    let mut ledger_gaps = Vec::new();
    let mut warnings = Vec::new();

    let mut min_battery = 1.0f32;
    let mut max_after_min = 0.0f32;
    let mut prev_t_ms = None;

    for frame in &frames {
        if !frame.now.eye.frames.is_empty() || !frame.now.eye.image_vectors.is_empty() {
            total_eye_frames += 1;
        }
        if !frame.now.ear.features.is_empty() || frame.now.ear.transcript.is_some() {
            total_ear_frames += 1;
        }

        // 1. Eye source tracking
        if let Some(eye_frame) = &frame.now.eye_frame {
            let src = eye_frame
                .source
                .clone()
                .unwrap_or_else(|| "none".to_string());
            *eye_sources.entry(src.clone()).or_insert(0) += 1;
            if src == "babylon-robot-eye" {
                babylon_eye_frames += 1;
            }
        }

        // 2. Collision tracking
        if frame.now.body.flags.bump_left || frame.now.body.flags.bump_right {
            collisions += 1;
        }

        // 3. Charger & Battery tracking
        if frame.now.body.charging {
            charging_ticks += 1;
            if !was_charging {
                charger_contacts += 1;
            }
            was_charging = true;
        } else {
            was_charging = false;
        }

        let bat = frame.now.body.battery_level;
        if bat < min_battery {
            min_battery = bat;
            max_after_min = bat;
        } else if bat > max_after_min {
            max_after_min = bat;
        }

        // 4. Stuck recovery / Trap tracking
        if let Some(val) = frame.now.extensions.get("sim.stuck") {
            if let Ok(values) = serde_json::from_value::<Vec<f32>>(val.clone()) {
                let event_started = values.get(6).copied().unwrap_or(0.0) > 0.0;
                let recovered = values.get(7).copied().unwrap_or(0.0) > 0.0;
                let trap_code = values.get(10).copied().unwrap_or(0.0);

                if event_started {
                    total_stuck_trap_events += 1;
                    stuck_recovery_attempts += 1;
                    let trap_name = match trap_code {
                        1.0 => "Wall",
                        2.0 => "Corner",
                        3.0 => "Column",
                        _ => "Unknown",
                    }
                    .to_string();
                    *trap_kinds.entry(trap_name).or_insert(0) += 1;
                }
                if recovered {
                    stuck_recovery_successes += 1;
                }
            }
        }

        // 5. Gap tracking
        if let Some(prev) = prev_t_ms {
            let diff = frame.t_ms.saturating_sub(prev);
            if diff > 500 {
                ledger_gaps.push(format!(
                    "gap of {}ms between {}ms and {}ms",
                    diff, prev, frame.t_ms
                ));
            }
        }
        prev_t_ms = Some(frame.t_ms);
    }

    let battery_delta = if let (Some(first), Some(last)) = (frames.first(), frames.last()) {
        first.now.body.battery_level - last.now.body.battery_level
    } else {
        0.0
    };

    let battery_recovery_success = max_after_min - min_battery >= 0.05;

    let duration_seconds = if let (Some(first), Some(last)) = (frames.first(), frames.last()) {
        (last.t_ms.saturating_sub(first.t_ms) as f64) / 1000.0
    } else {
        0.0
    };

    let collision_rate = if total_frames > 0 {
        collisions as f32 / total_frames as f32
    } else {
        0.0
    };

    let retina_coverage = if total_frames > 0 {
        babylon_eye_frames as f32 / total_frames as f32
    } else {
        0.0
    };

    if total_frames == 0 {
        warnings.push("ledger is empty".to_string());
    } else if babylon_eye_frames == 0 {
        warnings.push("no retina frames from babylon-robot-eye found in ledger".to_string());
    }

    Ok(VirtualRunReport {
        total_frames,
        total_transitions,
        total_eye_frames,
        total_ear_frames,
        total_stuck_trap_events,
        battery_delta,
        duration_seconds,
        eye_sources,
        retina_coverage,
        collisions,
        collision_rate,
        charger_contacts,
        charging_ticks,
        battery_recovery_success,
        stuck_recovery_attempts,
        stuck_recovery_successes,
        trap_kinds,
        ledger_gaps,
        warnings,
    })
}

async fn run_train_virtual(args: TrainVirtualArgs) -> Result<()> {
    println!("Starting virtual training pipeline...");
    println!("Ledger: {}", args.ledger);
    println!("Out Dir: {}", args.out_dir);

    // 1. Generate run report
    let run_report = generate_virtual_report(&args.ledger).await?;
    println!("Run report generated successfully.");

    // Create out_dir
    fs::create_dir_all(&args.out_dir)?;

    // 2. Train selected behaviors
    let behaviors = vec![
        TrainableBehavior::Danger,
        TrainableBehavior::Charge,
        TrainableBehavior::EyeNext,
        TrainableBehavior::EarNext,
        TrainableBehavior::Future,
    ];

    let mut trained_summaries = HashMap::new();
    for behavior in &behaviors {
        let checkpoint_path = Path::new(&args.out_dir).join(behavior.config_key());
        println!("Training behavior model: {:?}", behavior);
        let summary = train_behavior(TrainBehaviorRequest {
            behavior: behavior.clone(),
            ledger_path: PathBuf::from(&args.ledger),
            checkpoint_path,
            epochs: args.epochs,
            validation_split: 0.2,
            seed: 7,
        })
        .await?;
        trained_summaries.insert(behavior.clone(), summary);
    }

    // 3. Run scenario evaluations
    println!("Running baseline scenario evaluation (all models Off)...");
    let baseline_report_path = Path::new(&args.out_dir).join("baseline-scenario.json");
    let baseline_args = EvalScenarioArgs {
        scenario: ScenarioArg::MixedRoom,
        episodes: 10,
        steps: 100,
        seed: 7,
        tick_ms: 100,
        out: Some(baseline_report_path.to_string_lossy().to_string()),
        ledger: None,
        capture_root: None,
        memory_report: false,
        danger_checkpoint: None,
        danger_mode: DangerMode::Off,
        charge_checkpoint: None,
        charge_mode: ChargeMode::Off,
        action_value_checkpoint: None,
        action_value_mode: ActionValueMode::Off,
        future_checkpoint: None,
        future_mode: FutureMode::Hardcoded,
        eye_next_checkpoint: None,
        eye_next_mode: EyeNextMode::Off,
        ear_next_checkpoint: None,
        ear_next_mode: EarNextMode::Off,
        experience_checkpoint: None,
        experience_mode: ExperienceMode::Off,
        action_selector: CliActionSelectorMode::Baseline,
        llm: LlmArgs::default(),
    };
    run_eval_scenario(baseline_args).await?;
    let baseline_report = load_scenario_report(&baseline_report_path.to_string_lossy())?;

    println!("Running candidate scenario evaluation (new models ShadowInfer)...");
    let candidate_report_path = Path::new(&args.out_dir).join("candidate-scenario.json");
    let candidate_args = EvalScenarioArgs {
        scenario: ScenarioArg::MixedRoom,
        episodes: 10,
        steps: 100,
        seed: 7,
        tick_ms: 100,
        out: Some(candidate_report_path.to_string_lossy().to_string()),
        ledger: None,
        capture_root: None,
        memory_report: false,
        danger_checkpoint: Some(
            Path::new(&args.out_dir)
                .join("danger")
                .to_string_lossy()
                .to_string(),
        ),
        danger_mode: DangerMode::ShadowInfer,
        charge_checkpoint: Some(
            Path::new(&args.out_dir)
                .join("charge")
                .to_string_lossy()
                .to_string(),
        ),
        charge_mode: ChargeMode::ShadowInfer,
        action_value_checkpoint: None,
        action_value_mode: ActionValueMode::Off,
        future_checkpoint: Some(
            Path::new(&args.out_dir)
                .join("future")
                .to_string_lossy()
                .to_string(),
        ),
        future_mode: FutureMode::ShadowInfer,
        eye_next_checkpoint: Some(
            Path::new(&args.out_dir)
                .join("eye_next")
                .to_string_lossy()
                .to_string(),
        ),
        eye_next_mode: EyeNextMode::ShadowInfer,
        ear_next_checkpoint: Some(
            Path::new(&args.out_dir)
                .join("ear_next")
                .to_string_lossy()
                .to_string(),
        ),
        ear_next_mode: EarNextMode::ShadowInfer,
        experience_checkpoint: None,
        experience_mode: ExperienceMode::Off,
        action_selector: CliActionSelectorMode::Baseline,
        llm: LlmArgs::default(),
    };
    run_eval_scenario(candidate_args).await?;
    let candidate_report = load_scenario_report(&candidate_report_path.to_string_lossy())?;

    // Compare scenario reports
    let comparison_report_path = Path::new(&args.out_dir).join("comparison-scenario.json");
    let comparison = compare_scenario_reports(
        &baseline_report_path.to_string_lossy(),
        &candidate_report_path.to_string_lossy(),
        &baseline_report,
        &candidate_report,
    );
    write_scenario_comparison_report(&comparison_report_path, &comparison)?;
    println!(
        "Evaluation comparison recommendation: {}",
        comparison.recommendation.as_str()
    );

    // 4. Update/register models and run promotion gates
    let registry_path = Path::new("data/models/registry.json");
    let mut model_statuses = HashMap::new();
    let timestamp = Utc::now().format("%Y%m%d_%H%M").to_string();

    for behavior in &behaviors {
        let name = format!("{}_virtual_{}", behavior.config_key(), timestamp);
        let checkpoint = Path::new(&args.out_dir)
            .join(behavior.config_key())
            .to_string_lossy()
            .to_string();
        let behavior_report = Path::new(&checkpoint).join("evaluation.json");

        println!("Registering candidate model {}...", name);
        model_register(ModelRegisterArgs {
            behavior: behavior.cli_name().to_string(),
            checkpoint: checkpoint.clone(),
            training_ledger: Some(args.ledger.clone()),
            training_command: Some("just train virtual".to_string()),
            behavior_report: Some(behavior_report.to_string_lossy().to_string()),
            scenario_report: Some(candidate_report_path.to_string_lossy().to_string()),
            comparison_report: Some(comparison_report_path.to_string_lossy().to_string()),
            name: name.clone(),
            notes: vec!["Automatically trained via virtual pipeline".to_string()],
            parent: None,
            registry: registry_path.to_string_lossy().to_string(),
            overwrite: true,
        })?;

        // Load the registry to get the entry we just registered
        let registry = load_model_registry(registry_path)?;
        let entry = registry
            .entries
            .iter()
            .find(|e| e.name == name && e.behavior == *behavior)
            .unwrap()
            .clone();

        // Determine recommended promotion status
        // First test Inference promotion
        let inference_decision = promotion_gate(
            &entry,
            ModelStatus::Inference,
            Some(&baseline_report),
            Some(&candidate_report),
            Some(&comparison),
            args.allow_safety_critical_inference,
        );

        let mut new_status = ModelStatus::Registered;
        let mut recommended_action = "keep hardcoded".to_string();
        let mut warnings = Vec::new();

        if inference_decision.allowed {
            new_status = ModelStatus::Inference;
            recommended_action = "inference".to_string();
        } else {
            // Test Shadow promotion
            let shadow_decision = promotion_gate(
                &entry,
                ModelStatus::Shadow,
                Some(&baseline_report),
                Some(&candidate_report),
                Some(&comparison),
                args.allow_safety_critical_inference,
            );
            if shadow_decision.allowed {
                new_status = ModelStatus::Shadow;
                recommended_action = "shadow".to_string();
            } else {
                // Collect warnings for why promotion failed
                warnings.extend(inference_decision.warnings);
                warnings.extend(shadow_decision.warnings);
            }
        }

        // Apply promotion if recommended status is higher than Registered
        if new_status != ModelStatus::Registered {
            println!("Promoting model {} to {}...", name, new_status.as_str());
            model_promote(ModelPromoteArgs {
                behavior: behavior.cli_name().to_string(),
                name: name.clone(),
                target: new_status,
                baseline_report: Some(baseline_report_path.to_string_lossy().to_string()),
                candidate_report: Some(candidate_report_path.to_string_lossy().to_string()),
                comparison_report: Some(comparison_report_path.to_string_lossy().to_string()),
                registry: registry_path.to_string_lossy().to_string(),
                allow_safety_critical_inference: args.allow_safety_critical_inference,
                notes: vec!["Automatically promoted via virtual pipeline".to_string()],
            })?;
        }

        let loss = trained_summaries.get(behavior).and_then(|s| s.last_loss);

        model_statuses.insert(
            behavior.config_key().to_string(),
            ModelTrainingStatus {
                name,
                trained: true,
                previous_status: "registered".to_string(),
                new_status: new_status.as_str().to_string(),
                recommended_action,
                warnings,
                loss,
                baseline_collision_rate: Some(baseline_report.summary.collision_rate),
                candidate_collision_rate: Some(candidate_report.summary.collision_rate),
                baseline_success_rate: Some(baseline_report.summary.success_rate),
                candidate_success_rate: Some(candidate_report.summary.success_rate),
            },
        );
    }

    // 5. Write final consolidated training report
    let final_report = VirtualTrainingReport {
        timestamp: Utc::now().to_rfc3339(),
        run_report,
        models: model_statuses,
        warnings: if comparison.recommendation
            == ScenarioComparisonRecommendation::RegressionDetected
        {
            vec![
                "Candidate models overall regressed on MixedRoom scenario against baseline"
                    .to_string(),
            ]
        } else {
            Vec::new()
        },
    };

    let parent = Path::new(&args.report_out).parent();
    if let Some(p) = parent {
        fs::create_dir_all(p)?;
    }
    fs::write(
        &args.report_out,
        serde_json::to_string_pretty(&final_report)?,
    )?;
    println!(
        "Consolidated training report written to {}",
        args.report_out
    );

    Ok(())
}

#[derive(Debug, Parser)]
struct RetinaMockSendArgs {
    /// Server URL
    #[arg(long, default_value = "https://localhost:8443")]
    url: String,

    /// Frame rate (FPS)
    #[arg(long, default_value = "5")]
    fps: u64,

    /// Width of mock image
    #[arg(long, default_value = "160")]
    width: u32,

    /// Height of mock image
    #[arg(long, default_value = "90")]
    height: u32,

    /// Color pattern: "solid-red", "solid-green", "solid-blue", "gradient", or "noise"
    #[arg(long, default_value = "gradient")]
    pattern: String,
}

fn generate_mock_image_base64(
    width: u32,
    height: u32,
    pattern: &str,
    frame_index: usize,
) -> Result<String> {
    use base64::Engine;
    use image::codecs::png::PngEncoder;
    use image::ImageEncoder;
    use image::{Rgb, RgbImage};

    let mut img = RgbImage::new(width, height);

    match pattern {
        "solid-red" => {
            for pixel in img.pixels_mut() {
                *pixel = Rgb([255, 0, 0]);
            }
        }
        "solid-green" => {
            for pixel in img.pixels_mut() {
                *pixel = Rgb([0, 255, 0]);
            }
        }
        "solid-blue" => {
            for pixel in img.pixels_mut() {
                *pixel = Rgb([0, 0, 255]);
            }
        }
        "gradient" => {
            for (x, y, pixel) in img.enumerate_pixels_mut() {
                let r = ((x as f32 / width as f32) * 255.0) as u8;
                let g = ((y as f32 / height as f32) * 255.0) as u8;
                let b = ((frame_index * 10) % 256) as u8;
                *pixel = Rgb([r, g, b]);
            }
        }
        "noise" => {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for pixel in img.pixels_mut() {
                *pixel = Rgb([rng.gen(), rng.gen(), rng.gen()]);
            }
        }
        _ => {
            for (x, y, pixel) in img.enumerate_pixels_mut() {
                let g = ((x as f32 / width as f32) * 255.0) as u8;
                let b = ((y as f32 / height as f32) * 255.0) as u8;
                *pixel = Rgb([0, g, b]);
            }
        }
    }

    let mut png_bytes = Vec::new();
    PngEncoder::new(&mut png_bytes)
        .write_image(&img, width, height, image::ColorType::Rgb8.into())
        .context("failed to encode mock image as PNG")?;

    let encoded = base64::engine::general_purpose::STANDARD.encode(&png_bytes);
    Ok(encoded)
}

async fn run_retina_mock_send(args: RetinaMockSendArgs) -> Result<()> {
    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .context("failed to build reqwest client")?;

    let url = format!("{}/view/retina-frame", args.url.trim_end_matches('/'));
    println!(
        "Starting mock retina stream to {url} at {} FPS ({}x{})...",
        args.fps, args.width, args.height
    );

    let interval = Duration::from_millis(1000 / args.fps.max(1));
    let mut interval_timer = tokio::time::interval(interval);
    let mut frame_index = 0;

    let start_time = std::time::Instant::now();

    loop {
        interval_timer.tick().await;

        let t_ms = start_time.elapsed().as_millis() as u64;
        let base64_str =
            match generate_mock_image_base64(args.width, args.height, &args.pattern, frame_index) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error generating mock image: {e}");
                    continue;
                }
            };

        let payload = serde_json::json!({
            "schema_version": 1,
            "source": "babylon-robot-eye",
            "t_ms": t_ms,
            "frame_index": frame_index,
            "width": args.width,
            "height": args.height,
            "format": "Rgb8",
            "encoding": "base64",
            "data": format!("data:image/png;base64,{base64_str}")
        });

        let res = client.post(&url).json(&payload).send().await;

        match res {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    println!(
                        "[frame {}] Sent successfully (t_ms = {})",
                        frame_index, t_ms
                    );
                } else {
                    let err_text = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "unknown error".to_string());
                    eprintln!(
                        "[frame {}] FAILED with status {}: {}",
                        frame_index, status, err_text
                    );
                }
            }
            Err(e) => {
                eprintln!("[frame {}] Request error: {}", frame_index, e);
            }
        }

        frame_index += 1;
    }
}

