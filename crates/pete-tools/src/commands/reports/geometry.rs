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
    calibration_replay: CalibrationReplaySummary,
    imu_orientation: GeometryImuInterpretation,
    timestamp_diagnostics: GeometryTimestampDiagnostics,
    stationary_rotation_diagnostics: StationaryRotationDiagnostics,
    coordinate_frame_conventions: Vec<String>,
    sample_transformed_points: Vec<GeometryPointSample>,
    floor_statistics: GeometryFloorStatistics,
    warnings: Vec<String>,
    hard_failures: Vec<String>,
}

#[derive(Debug, Default, Serialize)]
struct CalibrationReplaySummary {
    frames_with_estimate: usize,
    epoch_ids: Vec<u64>,
    epoch_changes: usize,
    configured_frames: usize,
    estimating_frames: usize,
    trusted_frames: usize,
    degraded_frames: usize,
    invalidated_frames: usize,
    maximum_covariance: [f32; pete_now::TRANSFORM_DOF_COUNT],
    maximum_floor_residual_m: Option<f32>,
    maximum_wall_residual_m: Option<f32>,
    maximum_reprojection_residual_px: Option<f32>,
    maximum_map_consistency_residual_m: Option<f32>,
    evidence_counts: BTreeMap<String, u64>,
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
    camera_left_m: f32,
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
    body_timestamp_in_future: bool,
    eye_frame_age_ms: Option<u64>,
    ear_pcm_age_ms: Option<u64>,
    kinect_capture_timestamp_present: bool,
    kinect_capture_age_ms: Option<u64>,
    kinect_timestamp_in_future: bool,
    imu_capture_timestamp_present: bool,
    imu_capture_age_ms: Option<u64>,
    imu_timestamp_in_future: bool,
    kinect_imu_skew_ms: Option<u64>,
    kinect_body_skew_ms: Option<u64>,
    note: String,
}

#[derive(Debug, Serialize)]
struct StationaryRotationDiagnostics {
    evaluated: bool,
    reason: String,
    frame_count: usize,
    direction: String,
    heading_delta_deg: f32,
    cumulative_rotation_deg: f32,
    final_heading_error_deg: f32,
    translation_delta_m: f32,
    max_axle_translation_m: f32,
    stationary_frames_before: usize,
    stationary_frames_after: usize,
    imu_integrated_rotation_deg: Option<f32>,
    imu_odometry_error_deg: Option<f32>,
    rotation_agreement: bool,
    calibration_epoch_ids: Vec<u64>,
    remount_detected: bool,
    reconverged_after_remount: bool,
    observability_gate_passed: bool,
    insufficient_observability_exposed: bool,
    covariance_gate_passed: bool,
    optional_lidar_present: bool,
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
    let geometry = pete_now::DepthGeometry::from_kinect(kinect)
        .context("depth frame has no usable calibrated or declared intrinsics")?;
    let config = PointCloudConfig::default();
    let alignment = kinect.fusion_alignment.as_ref();
    let pose = alignment
        .map(|alignment| alignment.pose)
        .unwrap_or(snapshot.body.odometry);
    let imu = alignment
        .map(|alignment| &alignment.imu)
        .unwrap_or(&snapshot.imu);
    let orientation = pete_map::orientation_from_imu(imu, pose.heading_rad);
    let imu_interpretation = geometry_imu_interpretation(&imu.orientation, orientation);
    let timestamp_diagnostics = geometry_timestamp_diagnostics(&frames, record);
    let stationary_rotation_diagnostics = stationary_rotation_diagnostics(&frames, args);
    let calibration_replay = calibration_replay_summary(&frames);
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
        let Some(camera_xyz) = geometry.depth_pixel_to_camera(u as f32, v as f32, *depth) else {
            continue;
        };
        let camera = Point3D {
            x_m: camera_xyz[0],
            y_m: camera_xyz[1],
            z_m: camera_xyz[2],
        };
        let robot = if kinect.geometry_calibration.is_some() {
            let base = geometry.depth_point_to_base(camera_xyz);
            Point3D {
                x_m: base[0],
                y_m: base[1],
                z_m: base[2],
            }
        } else {
            geometry_camera_to_robot(camera, config)
        };
        let world =
            transform_point_to_world(robot, PointCloudFrame::RobotBase, pose, orientation, config);
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
        snapshot,
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
    let (camera_forward_m, camera_left_m, camera_height_m, camera_rotation_rpy) = kinect
        .geometry_calibration
        .map(|calibration| {
            (
                calibration.depth_to_base.translation_m[0],
                calibration.depth_to_base.translation_m[1],
                calibration.depth_to_base.translation_m[2],
                calibration.depth_to_base.rotation_rpy_rad,
            )
        })
        .unwrap_or((
            config.camera_forward_m,
            config.camera_left_m,
            config.camera_height_m,
            [
                config.camera_roll_rad,
                config.camera_pitch_rad,
                config.camera_yaw_rad,
            ],
        ));
    Ok(GeometryDebugReport {
        schema_version: 4,
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
            camera_height_m,
            camera_forward_m,
            camera_left_m,
            camera_pitch_rad: camera_rotation_rpy[1],
            camera_roll_rad: camera_rotation_rpy[0],
            camera_yaw_rad: camera_rotation_rpy[2],
            rotation_order:
                "calibrated source roll, pitch, yaw, then destination-frame translation"
                    .to_string(),
            base_mapping: "Kinect camera +x right, +y down, +z forward -> robot +x forward, +y left, +z up".to_string(),
        },
        calibration_replay,
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

fn calibration_replay_summary(
    frames: &[pete_worldlab::CaptureFrameRecord],
) -> CalibrationReplaySummary {
    let mut summary = CalibrationReplaySummary::default();
    let mut previous_epoch = None;
    for estimate in frames
        .iter()
        .filter_map(|frame| frame.snapshot.kinect.live_geometry_calibration.as_ref())
    {
        summary.frames_with_estimate += 1;
        if previous_epoch.is_some_and(|epoch| epoch != estimate.epoch.id) {
            summary.epoch_changes += 1;
        }
        previous_epoch = Some(estimate.epoch.id);
        if !summary.epoch_ids.contains(&estimate.epoch.id) {
            summary.epoch_ids.push(estimate.epoch.id);
        }
        match estimate.trust_state {
            pete_now::CalibrationTrustState::Configured => summary.configured_frames += 1,
            pete_now::CalibrationTrustState::Estimating => summary.estimating_frames += 1,
            pete_now::CalibrationTrustState::Trusted => summary.trusted_frames += 1,
            pete_now::CalibrationTrustState::Degraded => summary.degraded_frames += 1,
            pete_now::CalibrationTrustState::Invalidated => summary.invalidated_frames += 1,
        }
        for (index, covariance) in estimate.covariance.iter().enumerate() {
            summary.maximum_covariance[index] =
                summary.maximum_covariance[index].max(*covariance);
        }
        update_maximum(
            &mut summary.maximum_floor_residual_m,
            estimate.residuals.floor_m,
        );
        update_maximum(
            &mut summary.maximum_wall_residual_m,
            estimate.residuals.wall_m,
        );
        update_maximum(
            &mut summary.maximum_reprojection_residual_px,
            estimate.residuals.reprojection_px,
        );
        update_maximum(
            &mut summary.maximum_map_consistency_residual_m,
            estimate.residuals.map_consistency_m,
        );
        for (source, count) in &estimate.evidence_counts {
            let label = format!("{source:?}").to_ascii_lowercase();
            let total = summary.evidence_counts.entry(label).or_default();
            *total = (*total).max(u64::from(*count));
        }
    }
    summary
}

fn update_maximum(slot: &mut Option<f32>, value: Option<f32>) {
    if let Some(value) = value.map(f32::abs) {
        *slot = Some(slot.map_or(value, |current| current.max(value)));
    }
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
        source: "fallback_legacy_square_projection".to_string(),
        source_is_fallback: true,
    }
}

fn sensor_truth_report(
    snapshot: &WorldSnapshot,
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
        .is_some_and(|age| age <= args.max_body_timestamp_age_ms)
        && !timestamps.body_timestamp_in_future;
    gates.push(SensorTruthGate {
        name: "body_timestamp_fresh".to_string(),
        status: if body_age_ok {
            SensorTruthStatus::Pass
        } else {
            SensorTruthStatus::Fail
        },
        detail: format!(
            "body_last_update_age_ms={:?}, future={}, threshold={}",
            timestamps.body_last_update_age_ms,
            timestamps.body_timestamp_in_future,
            args.max_body_timestamp_age_ms
        ),
    });
    gates.push(SensorTruthGate {
        name: "multi_frame_depth_capture".to_string(),
        status: if timestamps.depth_frame_count >= args.min_depth_frames {
            SensorTruthStatus::Pass
        } else {
            SensorTruthStatus::Fail
        },
        detail: format!(
            "depth_frames={}, minimum={}",
            timestamps.depth_frame_count, args.min_depth_frames
        ),
    });
    let camera_calibrated = snapshot
        .kinect
        .geometry_calibration
        .is_some_and(|calibration| calibration.physical_validation_ready());
    gates.push(SensorTruthGate {
        name: "camera_geometry_calibrated".to_string(),
        status: if camera_calibrated {
            SensorTruthStatus::Pass
        } else {
            SensorTruthStatus::Fail
        },
        detail: "requires measured depth intrinsics/distortion, depth scale, full depth-to-base and RGB-D extrinsics, four 0.5-3.0m validation distances, <=2cm plane error, and <=3px RGB-D boundary error".to_string(),
    });
    let live_calibration_trusted = snapshot
        .kinect
        .live_geometry_calibration
        .as_ref()
        .is_some_and(|estimate| {
            estimate.trust_state == pete_now::CalibrationTrustState::Trusted
        });
    gates.push(SensorTruthGate {
        name: "live_kinect_mount_calibration".to_string(),
        status: if live_calibration_trusted {
            SensorTruthStatus::Pass
        } else {
            SensorTruthStatus::Fail
        },
        detail: snapshot
            .kinect
            .live_geometry_calibration
            .as_ref()
            .map(|estimate| {
                format!(
                    "epoch={}, state={:?}, confidence={:.3}, covariance={:?}, evidence={:?}, residuals={:?}, reasons={}",
                    estimate.epoch.id,
                    estimate.trust_state,
                    estimate.confidence,
                    estimate.covariance,
                    estimate.evidence_counts,
                    estimate.residuals,
                    estimate.rejection_reasons.join("; ")
                )
            })
            .unwrap_or_else(|| {
                "configured Kinect transform is only an initial guess; no live estimate is present"
                    .to_string()
            }),
    });
    let rgbd_skew_ms = snapshot.kinect.color_frame.as_ref().map(|color| {
        color
            .captured_at_ms
            .abs_diff(snapshot.kinect.captured_at_ms)
    });
    let rgbd_paired = snapshot.kinect.color_frame.as_ref().is_some_and(|color| {
        color.rgbd_frame_id.is_some()
            && color.rgbd_frame_id == snapshot.kinect.rgbd_frame_id
            && rgbd_skew_ms.is_some_and(|skew| skew <= args.max_rgbd_skew_ms)
    });
    gates.push(SensorTruthGate {
        name: "rgb_depth_paired".to_string(),
        status: if rgbd_paired {
            SensorTruthStatus::Pass
        } else {
            SensorTruthStatus::Fail
        },
        detail: format!(
            "shared_frame_id={}, capture_skew_ms={rgbd_skew_ms:?}, threshold={}",
            snapshot.kinect.rgbd_frame_id.is_some(),
            args.max_rgbd_skew_ms
        ),
    });
    let kinect_fresh = timestamps.kinect_capture_timestamp_present
        && !timestamps.kinect_timestamp_in_future
        && timestamps
            .kinect_capture_age_ms
            .is_some_and(|age| age <= args.max_kinect_timestamp_age_ms);
    gates.push(SensorTruthGate {
        name: "kinect_timestamp_fresh".to_string(),
        status: if kinect_fresh {
            SensorTruthStatus::Pass
        } else {
            SensorTruthStatus::Fail
        },
        detail: format!(
            "present={}, future={}, age_ms={:?}, threshold={}",
            timestamps.kinect_capture_timestamp_present,
            timestamps.kinect_timestamp_in_future,
            timestamps.kinect_capture_age_ms,
            args.max_kinect_timestamp_age_ms
        ),
    });
    let imu_fresh = timestamps.imu_capture_timestamp_present
        && !timestamps.imu_timestamp_in_future
        && timestamps
            .imu_capture_age_ms
            .is_some_and(|age| age <= args.max_imu_timestamp_age_ms);
    gates.push(SensorTruthGate {
        name: "imu_timestamp_fresh".to_string(),
        status: if imu_fresh {
            SensorTruthStatus::Pass
        } else {
            SensorTruthStatus::Fail
        },
        detail: format!(
            "present={}, future={}, age_ms={:?}, threshold={}",
            timestamps.imu_capture_timestamp_present,
            timestamps.imu_timestamp_in_future,
            timestamps.imu_capture_age_ms,
            args.max_imu_timestamp_age_ms
        ),
    });
    let sensors_synchronized = kinect_fresh
        && imu_fresh
        && timestamps
            .kinect_imu_skew_ms
            .is_some_and(|skew| skew <= args.max_kinect_imu_skew_ms);
    gates.push(SensorTruthGate {
        name: "kinect_imu_synchronized".to_string(),
        status: if sensors_synchronized {
            SensorTruthStatus::Pass
        } else {
            SensorTruthStatus::Fail
        },
        detail: format!(
            "capture_skew_ms={:?}, threshold={}",
            timestamps.kinect_imu_skew_ms, args.max_kinect_imu_skew_ms
        ),
    });
    let pose_synchronized = kinect_fresh
        && body_age_ok
        && timestamps
            .kinect_body_skew_ms
            .is_some_and(|skew| skew <= args.max_kinect_body_skew_ms);
    gates.push(SensorTruthGate {
        name: "kinect_body_pose_synchronized".to_string(),
        status: if pose_synchronized {
            SensorTruthStatus::Pass
        } else {
            SensorTruthStatus::Fail
        },
        detail: format!(
            "capture_skew_ms={:?}, threshold={}",
            timestamps.kinect_body_skew_ms, args.max_kinect_body_skew_ms
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
        && stationary.max_axle_translation_m <= args.max_stationary_translation_m
        && stationary.rotation_agreement
        && stationary.observability_gate_passed
        && stationary.covariance_gate_passed
        && (!stationary.remount_detected || stationary.reconverged_after_remount)
    {
        SensorTruthStatus::Pass
    } else {
        SensorTruthStatus::Fail
    };
    gates.push(SensorTruthGate {
        name: "stationary_rotation_cloud_stability".to_string(),
        status: stationary_status,
        detail: format!(
            "{}; cumulative_rotation_deg={:.1}, final_heading_error_deg={:.1}, max_axle_translation_m={:.3}, imu_error_deg={:?}, stable_ratio={:.3}, stable_z_span_m={:?}, epochs={:?}, remount={}, reconverged={}",
            stationary.reason,
            stationary.cumulative_rotation_deg,
            stationary.final_heading_error_deg,
            stationary.max_axle_translation_m,
            stationary.imu_odometry_error_deg,
            stationary.stable_voxel_ratio,
            stationary.stable_z_span_m,
            stationary.calibration_epoch_ids,
            stationary.remount_detected,
            stationary.reconverged_after_remount,
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
        body_last_update_age_ms: selected.t_ms.checked_sub(snapshot.body.last_update_ms),
        body_timestamp_in_future: snapshot.body.last_update_ms > selected.t_ms,
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
            .then(|| selected.t_ms.checked_sub(snapshot.kinect.captured_at_ms))
            .flatten(),
        kinect_timestamp_in_future: snapshot.kinect.captured_at_ms > selected.t_ms,
        imu_capture_timestamp_present: snapshot.imu.captured_at_ms > 0,
        imu_capture_age_ms: (snapshot.imu.captured_at_ms > 0)
            .then(|| selected.t_ms.checked_sub(snapshot.imu.captured_at_ms))
            .flatten(),
        imu_timestamp_in_future: snapshot.imu.captured_at_ms > selected.t_ms,
        kinect_imu_skew_ms: (snapshot.kinect.captured_at_ms > 0
            && snapshot.imu.captured_at_ms > 0)
            .then(|| snapshot.kinect.captured_at_ms.abs_diff(snapshot.imu.captured_at_ms)),
        kinect_body_skew_ms: snapshot
            .kinect
            .fusion_alignment
            .as_ref()
            .map(|alignment| alignment.pose_sample_skew_ms)
            .or_else(|| {
                (snapshot.kinect.captured_at_ms > 0 && snapshot.body.last_update_ms > 0)
                    .then(|| snapshot.kinect.captured_at_ms.abs_diff(snapshot.body.last_update_ms))
            }),
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
            direction: "unobservable".to_string(),
            heading_delta_deg: 0.0,
            cumulative_rotation_deg: 0.0,
            final_heading_error_deg: 0.0,
            translation_delta_m: 0.0,
            max_axle_translation_m: 0.0,
            stationary_frames_before: 0,
            stationary_frames_after: 0,
            imu_integrated_rotation_deg: None,
            imu_odometry_error_deg: None,
            rotation_agreement: false,
            calibration_epoch_ids: Vec::new(),
            remount_detected: false,
            reconverged_after_remount: false,
            observability_gate_passed: false,
            insufficient_observability_exposed: false,
            covariance_gate_passed: false,
            optional_lidar_present: false,
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
    let cumulative_rotation_rad = depth_frames
        .windows(2)
        .map(|window| {
            angle_delta_signed(
                window[1].snapshot.body.odometry.heading_rad,
                window[0].snapshot.body.odometry.heading_rad,
            )
        })
        .sum::<f32>();
    let cumulative_rotation_deg = cumulative_rotation_rad.to_degrees();
    let translation_delta_m = ((last_pose.x_m - first_pose.x_m).powi(2)
        + (last_pose.y_m - first_pose.y_m).powi(2))
    .sqrt();
    let max_axle_translation_m = depth_frames
        .iter()
        .map(|frame| {
            (frame.snapshot.body.odometry.x_m - first_pose.x_m)
                .hypot(frame.snapshot.body.odometry.y_m - first_pose.y_m)
        })
        .fold(0.0, f32::max);
    let stationary_candidate = cumulative_rotation_deg.abs() >= args.min_stationary_rotation_deg
        && max_axle_translation_m <= args.max_stationary_translation_m;
    let stationary_frames_before = depth_frames
        .iter()
        .take_while(|frame| frame_is_stationary(&frame.snapshot))
        .count();
    let stationary_frames_after = depth_frames
        .iter()
        .rev()
        .take_while(|frame| frame_is_stationary(&frame.snapshot))
        .count();
    let imu_integrated_rotation_rad = integrate_imu_rotation(&depth_frames);
    let imu_odometry_error_deg = imu_integrated_rotation_rad
        .map(|imu| (imu - cumulative_rotation_rad).abs().to_degrees());
    let rotation_agreement = imu_integrated_rotation_rad.is_some_and(|imu| {
        imu.signum() == cumulative_rotation_rad.signum()
            && (imu - cumulative_rotation_rad).abs()
                <= (15.0_f32.to_radians()).max(cumulative_rotation_rad.abs() * 0.10)
    });
    let estimates = depth_frames
        .iter()
        .filter_map(|frame| frame.snapshot.kinect.live_geometry_calibration.as_ref())
        .collect::<Vec<_>>();
    let mut calibration_epoch_ids = Vec::new();
    for estimate in &estimates {
        if !calibration_epoch_ids.contains(&estimate.epoch.id) {
            calibration_epoch_ids.push(estimate.epoch.id);
        }
    }
    let remount_detected = calibration_epoch_ids.len() > 1
        || estimates.iter().any(|estimate| {
            estimate.trust_state == pete_now::CalibrationTrustState::Invalidated
        });
    let reconverged_after_remount = remount_detected
        && estimates.last().is_some_and(|estimate| {
            estimate.trust_state == pete_now::CalibrationTrustState::Trusted
        });
    let observability_gate_passed = estimates.last().is_some_and(|estimate| {
        estimate.trust_state == pete_now::CalibrationTrustState::Trusted
            && estimate.observable_dofs.iter().all(|observable| *observable)
    });
    let insufficient_observability_exposed = estimates.is_empty()
        || estimates.iter().any(|estimate| {
            estimate.trust_state != pete_now::CalibrationTrustState::Trusted
                && (!estimate.observable_dofs.iter().all(|observable| *observable)
                    || !estimate.rejection_reasons.is_empty())
        });
    let covariance_limits = pete_now::CalibrationStateConfig::default().trusted_covariance;
    let covariance_gate_passed = estimates.last().is_some_and(|estimate| {
        estimate
            .covariance
            .iter()
            .zip(covariance_limits)
            .all(|(value, limit)| value.is_finite() && *value <= limit)
    });
    let optional_lidar_present = depth_frames.iter().any(|frame| {
        frame.snapshot.range.source.as_deref().is_some_and(|source| {
            source.contains("lidar") || source.contains("lfcd")
        })
    });
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
        direction: if cumulative_rotation_rad > 0.0 {
            "counter_clockwise".to_string()
        } else if cumulative_rotation_rad < 0.0 {
            "clockwise".to_string()
        } else {
            "unobservable".to_string()
        },
        heading_delta_deg,
        cumulative_rotation_deg,
        final_heading_error_deg: heading_delta_deg,
        translation_delta_m,
        max_axle_translation_m,
        stationary_frames_before,
        stationary_frames_after,
        imu_integrated_rotation_deg: imu_integrated_rotation_rad.map(f32::to_degrees),
        imu_odometry_error_deg,
        rotation_agreement,
        calibration_epoch_ids,
        remount_detected,
        reconverged_after_remount,
        observability_gate_passed,
        insufficient_observability_exposed,
        covariance_gate_passed,
        optional_lidar_present,
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
    angle_delta_signed(left, right).abs()
}

fn angle_delta_signed(left: f32, right: f32) -> f32 {
    let mut delta = left - right;
    while delta > std::f32::consts::PI {
        delta -= std::f32::consts::TAU;
    }
    while delta < -std::f32::consts::PI {
        delta += std::f32::consts::TAU;
    }
    delta
}

fn frame_is_stationary(snapshot: &WorldSnapshot) -> bool {
    snapshot.body.velocity.forward_m_s.abs() <= 0.01
        && snapshot.body.velocity.turn_rad_s.abs() <= 0.01
        && snapshot
            .imu
            .angular_velocity
            .get(2)
            .is_none_or(|rate| rate.abs() <= 0.02)
}

fn integrate_imu_rotation(
    frames: &[&pete_worldlab::CaptureFrameRecord],
) -> Option<f32> {
    let mut rotation = 0.0;
    let mut intervals = 0usize;
    for window in frames.windows(2) {
        let first = &window[0].snapshot.imu;
        let second = &window[1].snapshot.imu;
        let trusted = |imu: &pete_now::ImuSense| {
            imu.schema_version < 3
                || imu.calibration.as_ref().is_some_and(|calibration| {
                    calibration.trust_state == pete_now::ImuCalibrationTrustState::Trusted
                })
        };
        let Some((first_rate, second_rate)) = first
            .angular_velocity
            .get(2)
            .copied()
            .zip(second.angular_velocity.get(2).copied())
        else {
            continue;
        };
        if !trusted(first) || !trusted(second) {
            continue;
        }
        let first_time = nonzero_or(first.captured_at_ms, window[0].t_ms);
        let second_time = nonzero_or(second.captured_at_ms, window[1].t_ms);
        let dt_s = second_time.saturating_sub(first_time) as f32 / 1_000.0;
        if !(0.0..=1.0).contains(&dt_s) || dt_s == 0.0 {
            continue;
        }
        rotation += (first_rate + second_rate) * 0.5 * dt_s;
        intervals += 1;
    }
    (intervals > 0).then_some(rotation)
}

fn nonzero_or(value: u64, fallback: u64) -> u64 {
    if value == 0 { fallback } else { value }
}
