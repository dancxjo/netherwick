async fn get_capture_scene(
    Query(query): Query<CaptureSceneQuery>,
) -> Result<Json<LiveSceneResponse>, LiveViewError> {
    let reader = CaptureReader::open(&query.capture)
        .await
        .map_err(|error| LiveViewError::bad_request(format!("failed to open capture: {error}")))?;
    let frames = reader.read_frames().await.map_err(|error| {
        LiveViewError::bad_request(format!("failed to read capture frames: {error}"))
    })?;
    let record = frames.get(query.frame).ok_or_else(|| {
        LiveViewError::not_found(format!("capture frame {} was not found", query.frame))
    })?;
    let metadata = reader
        .manifest()
        .scenario
        .as_ref()
        .map(|scenario| LiveSceneMetadata {
            arena: Some(SceneArena {
                width_m: scenario.arena.width_m,
                height_m: scenario.arena.height_m,
            }),
            objects: scenario
                .objects
                .iter()
                .map(|object| SceneObject {
                    id: object.id.clone(),
                    kind: scene_object_kind(&format!("{:?}", object.kind)),
                    x_m: object.x_m,
                    y_m: object.y_m,
                    radius_m: object.radius_m,
                    label: Some(object.label.clone()),
                    color_rgb: Some(object.color_rgb),
                })
                .collect(),
            sensor_calibration: None,
        });
    let mut scene = snapshot_to_scene(
        &record.snapshot,
        metadata.as_ref(),
        None,
        LiveTrainingStatus::default(),
        NudgeStatus::default(),
        default_behavior_nodes(),
        Some(&accumulated_point_cloud_for_frames(&frames[..=query.frame])),
        None,
        HardwareControlStatus::unavailable("capture replay is not a hardware cockpit session"),
    );
    scene.t_ms = record.t_ms;
    if let Some(pointcloud) = &record.assets.pointcloud {
        let ply_path = query.capture.join(pointcloud);
        if ply_path.exists() {
            match std::fs::read_to_string(&ply_path) {
                Ok(content) => {
                    let mut points = Vec::new();
                    let mut in_header = true;
                    for line in content.lines() {
                        let line = line.trim();
                        if in_header {
                            if line == "end_header" {
                                in_header = false;
                            }
                            continue;
                        }
                        if line.is_empty() {
                            continue;
                        }
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 3 {
                            if let (Ok(x), Ok(y), Ok(z)) = (
                                parts[0].parse::<f32>(),
                                parts[1].parse::<f32>(),
                                parts[2].parse::<f32>(),
                            ) {
                                points.push(ScenePoint {
                                    x,
                                    y: -y,
                                    z,
                                    r: 180,
                                    g: 180,
                                    b: 240,
                                });
                            }
                        }
                    }
                    scene.kinect.points = points;
                    scene.kinect.coordinate_system = Some("camera".to_string());
                }
                Err(error) => {
                    scene.warnings.push(format!(
                        "failed to read PLY file {}: {error}",
                        ply_path.display()
                    ));
                }
            }
        } else {
            scene
                .warnings
                .push(format!("PLY file not found: {}", ply_path.display()));
        }
    }
    Ok(Json(scene))
}

fn accumulated_point_cloud_for_frames(
    frames: &[pete_worldlab::CaptureFrameRecord],
) -> VoxelPointCloud {
    let mut cloud = VoxelPointCloud::default();
    for frame in frames {
        cloud.observe_snapshot(&frame.snapshot, frame.t_ms);
    }
    cloud
}

fn accumulated_scene_points(
    point_cloud: &VoxelPointCloud,
    now_ms: TimeMs,
) -> Vec<SceneAccumulatedPoint> {
    const MAX_SCENE_POINTS: usize = 16_000;
    let points = point_cloud.points();
    let stride = points.len().div_ceil(MAX_SCENE_POINTS).max(1);
    points
        .into_iter()
        .step_by(stride)
        .map(|point| scene_accumulated_point(point, now_ms))
        .collect()
}

fn scene_accumulated_point(point: VoxelPoint, now_ms: TimeMs) -> SceneAccumulatedPoint {
    let [r, g, b] = point.color_rgb.unwrap_or(if point.stable {
        [124, 230, 174]
    } else {
        [190, 194, 246]
    });
    SceneAccumulatedPoint {
        x: point.position.x_m,
        y: point.position.y_m,
        z: point.position.z_m,
        r,
        g,
        b,
        confidence: point.confidence,
        age_ms: now_ms.saturating_sub(point.last_seen_ms),
        stable: point.stable,
        transient: point.transient,
    }
}

fn scene_imu_debug(snapshot: &WorldSnapshot) -> SceneImuDebug {
    let orientation = orientation_from_imu(&snapshot.imu, snapshot.body.odometry.heading_rad);
    SceneImuDebug {
        raw_orientation: snapshot.imu.orientation.clone(),
        assumed_units: "radians".to_string(),
        assumed_axis_order: match snapshot.imu.orientation.len() {
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
        contract_known: matches!(snapshot.imu.orientation.len(), 2..),
    }
}

fn local_world_belief_trust_warning(kinect: &SceneKinect) -> Option<String> {
    let belief = kinect.local_world_belief.as_ref()?;
    let mut reasons = Vec::new();
    if kinect.diagnostics.coordinate_system == "depth_image_unknown" {
        reasons.push("fallback depth projection is active".to_string());
    }
    if kinect.diagnostics.below_floor_ratio > 0.02 {
        reasons.push(format!(
            "below_floor_ratio {:.3} exceeds 0.020",
            kinect.diagnostics.below_floor_ratio
        ));
    }
    if !belief.orientation_status.roll_pitch_corrected {
        reasons.push("IMU roll/pitch correction is not active".to_string());
    }
    if belief.stable_surfaces.is_empty() && belief.stable_blobs.is_empty() {
        reasons.push("no stable LocalWorldBelief surfaces or blobs yet".to_string());
    }
    if reasons.is_empty() {
        None
    } else {
        Some(format!(
            "LocalWorldBelief geometry not trustworthy: {}",
            reasons.join("; ")
        ))
    }
}

async fn get_live_now(
    State(state): State<LiveViewState>,
) -> Result<Json<pete_now::Now>, LiveViewError> {
    let snapshot = state
        .latest()
        .ok_or_else(|| LiveViewError::unavailable("no live world snapshot has arrived yet"))?;
    Ok(Json(snapshot.to_now(snapshot.body.last_update_ms)))
}

async fn live_view_page() -> Html<&'static str> {
    Html(LIVE_VIEW_PAGE)
}

async fn cognitive_view_page() -> Html<&'static str> {
    Html(COGNITIVE_VIEW_PAGE)
}

async fn live_view_3d_page() -> Html<&'static str> {
    Html(LIVE_VIEW_3D_PAGE)
}

async fn get_llm_stream(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(stream_llm_events)
}

async fn stream_llm_events(mut socket: WebSocket) {
    let mut rx = pete_llm::subscribe_llm_streams();
    loop {
        match rx.recv().await {
            Ok(event) => {
                let Ok(text) = serde_json::to_string(&event) else {
                    continue;
                };
                if socket.send(Message::Text(text.into())).await.is_err() {
                    break;
                }
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
        }
    }
}

pub fn snapshot_to_scene(
    snapshot: &WorldSnapshot,
    metadata: Option<&LiveSceneMetadata>,
    session: Option<SceneSession>,
    training: LiveTrainingStatus,
    prod_state: NudgeStatus,
    behavior_nodes: Vec<BehaviorNodeState>,
    point_cloud: Option<&VoxelPointCloud>,
    retina_status: Option<RetinaStatusInfo>,
    hardware_control: HardwareControlStatus,
) -> LiveSceneResponse {
    let mut warnings = Vec::new();
    let body = &snapshot.body;
    let eye = match snapshot.eye_frame.as_ref() {
        Some(f) => {
            let (eye, frame_warnings) =
                scene_eye_from_frame(f, retina_status.as_ref(), snapshot.body.last_update_ms);
            warnings.extend(frame_warnings);
            Some(eye)
        }
        None => {
            if let Some(rs) = retina_status.as_ref() {
                if rs.connected {
                    Some(SceneEye {
                        width: 0,
                        height: 0,
                        format: "None".to_string(),
                        data_url: None,
                        mean_luma: 1.0,
                        non_background_ratio: 1.0,
                        source: "babylon-robot-eye".to_string(),
                        authoritative: false,
                        retina_connected: true,
                        retina_last_frame_age_ms: None,
                        frames_received: rs.frames_received,
                        frames_written_to_ledger: rs.frames_written_to_ledger,
                    })
                } else {
                    warnings.push("no eye frame stream".to_string());
                    None
                }
            } else {
                warnings.push("no eye frame stream".to_string());
                None
            }
        }
    };
    let sensor_calibration = metadata.and_then(|metadata| metadata.sensor_calibration);
    let mut kinect = scene_kinect_from_snapshot(snapshot, sensor_calibration, &mut warnings);
    if let Some(point_cloud) = point_cloud {
        kinect.accumulated_points =
            accumulated_scene_points(point_cloud, snapshot.body.last_update_ms);
        kinect.accumulated_summary = Some(point_cloud.summary());
        kinect.local_world_belief = Some(point_cloud.local_world_belief());
        if let Some(warning) = local_world_belief_trust_warning(&kinect) {
            warnings.push(warning);
        }
        if !kinect.accumulated_points.is_empty() {
            kinect.coordinate_system = Some("world".to_string());
        }
    }
    let audio_bearing = snapshot
        .kinect
        .audio_angle_rad
        .or_else(|| audio_bearing_from_objects(body.odometry.x_m, body.odometry.y_m, metadata));
    let pcm_energy = snapshot.ear_pcm.as_ref().map(pcm_audio_energy);
    let audio = audio_bearing
        .map(|bearing_rad| SceneAudio {
            bearing_rad: Some(bearing_rad),
            energy: snapshot
                .kinect
                .audio_confidence
                .max(pcm_energy.unwrap_or(0.0))
                .clamp(0.0, 1.0),
        })
        .or_else(|| {
            pcm_energy.map(|energy| SceneAudio {
                bearing_rad: None,
                energy,
            })
        });
    if audio_bearing.is_none() {
        warnings.push("no audio bearing stream".to_string());
    }
    let stuck = scene_stuck_from_snapshot(snapshot);
    let training_mode = training.training_mode.clone();
    let ledger_path = training.ledger_path.clone();
    let frames_written = training.frames_written;
    let transitions_written = training.transitions_written;
    let models_loaded = training.models_loaded.clone();
    let model_modes = training.model_modes.clone();
    let action_selector_mode = training.action_selector_mode.clone();
    let weights_updating = training.weights_updating;
    LiveSceneResponse {
        schema_version: 1,
        session,
        training,
        hardware_control,
        training_mode,
        ledger_path,
        frames_written,
        transitions_written,
        models_loaded,
        model_modes,
        behavior_nodes,
        action_selector_mode,
        weights_updating,
        t_ms: body.last_update_ms,
        body: SceneBody {
            x_m: body.odometry.x_m,
            y_m: body.odometry.y_m,
            heading_rad: body.odometry.heading_rad,
            battery_level: body.battery_level,
            charging: body.charging,
            bump_left: body.flags.bump_left,
            bump_right: body.flags.bump_right,
            cliff: body.flags.cliff_left
                || body.flags.cliff_front_left
                || body.flags.cliff_front_right
                || body.flags.cliff_right,
            wheel_drop: body.flags.wheel_drop,
        },
        range: scene_range_from_snapshot(snapshot),
        eye,
        kinect,
        imu: scene_imu_debug(snapshot),
        surface_perception: None,
        world_belief_layers: vec![
            "current rays",
            "raw point cloud",
            "raw camera-frame points",
            "robot-frame points",
            "world-frame points",
            "accumulated occupancy",
            "floor plane",
            "axes gizmo",
            "stable wall candidates",
        ],
        audio,
        objects: metadata
            .map(|metadata| metadata.objects.clone())
            .unwrap_or_default(),
        arena: metadata.and_then(|metadata| metadata.arena),
        sensor_calibration,
        action: scene_action_from_snapshot(snapshot),
        prod: SceneProd {
            idle_ms: prod_state.idle_ms,
            last_nudge_ms: prod_state.last_nudge_ms,
            nudge_count_recent: prod_state.nudge_count_recent,
            nudge_blocked_reason: prod_state.nudge_blocked_reason.clone(),
            active_nudge: prod_state.active_nudge,
        },
        idle_ms: prod_state.idle_ms,
        last_nudge_ms: prod_state.last_nudge_ms,
        nudge_count_recent: prod_state.nudge_count_recent,
        nudge_blocked_reason: prod_state.nudge_blocked_reason.clone(),
        active_nudge: prod_state.active_nudge,
        stuck: stuck.active,
        dead_battery: stuck.dead_battery,
        recovery_mode: stuck.recovery_phase.clone(),
        stuck_ticks: stuck.stuck_ticks,
        stuck_detail: stuck,
        mind: SceneMind {
            combobulation: snapshot
                .extensions
                .iter()
                .find(|extension| extension.name == "vision.frame_summary")
                .map(|extension| {
                    format!(
                        "vision summary vector with {} values",
                        extension.values.len()
                    )
                }),
            surprise: None,
        },
        warnings,
    }
}

fn scene_action_from_snapshot(snapshot: &WorldSnapshot) -> SceneAction {
    let forward = snapshot.body.velocity.forward_m_s;
    let turn = snapshot.body.velocity.turn_rad_s;
    let action_debug = snapshot.action_debug.as_ref();
    let latest = if forward.abs() < 0.01 && turn.abs() < 0.01 {
        Some("stop".to_string())
    } else if turn.abs() < 0.01 {
        Some(format!("forward {:.2} m/s", forward))
    } else if forward.abs() < 0.01 {
        Some(format!("turn {:.2} rad/s", turn))
    } else {
        Some(format!("drive {:.2} m/s, turn {:.2} rad/s", forward, turn))
    };
    SceneAction {
        latest,
        desired_motor: action_debug_value(action_debug, "desired_motor"),
        final_motor: action_debug_value(action_debug, "final_motor"),
        motion_sent: action_debug_value(action_debug, "motion_sent_to_sim"),
        motor_applied: action_debug
            .and_then(|debug| debug.get("motor_applied"))
            .and_then(|value| value.as_bool()),
        movement_delta: action_debug
            .and_then(|debug| debug.get("movement_delta"))
            .and_then(|value| serde_json::from_value(value.clone()).ok()),
        safety_override: action_debug
            .and_then(|debug| debug.get("safety_override"))
            .and_then(|value| value.as_bool())
            .unwrap_or_else(|| {
                snapshot.body.flags.wheel_drop
                    || snapshot.body.flags.cliff_left
                    || snapshot.body.flags.cliff_right
            }),
        not_moving_reason: action_debug
            .and_then(|debug| {
                debug
                    .get("why_not_moving")
                    .or_else(|| debug.get("not_moving_reason"))
            })
            .and_then(|value| value.as_str())
            .map(str::to_string),
        latest_llm_proposed_action: snapshot
            .llm_action_proposal
            .as_ref()
            .and_then(|proposal| proposal.proposed_action.clone()),
        latest_llm_advisory_action: snapshot
            .llm_action_proposal
            .as_ref()
            .and_then(|proposal| proposal.advisory_action.clone()),
        llm_action_accepted: snapshot
            .llm_action_proposal
            .as_ref()
            .map(|proposal| proposal.accepted),
        llm_action_safety_vetoed: snapshot
            .llm_action_proposal
            .as_ref()
            .map(|proposal| proposal.safety_vetoed),
        final_selected_action: snapshot.final_selected_action.clone().or_else(|| {
            snapshot
                .llm_action_proposal
                .as_ref()
                .and_then(|proposal| proposal.final_action.clone())
        }),
        llm_action_ignored_reason: snapshot
            .llm_action_proposal
            .as_ref()
            .and_then(|proposal| proposal.ignored_reason.clone()),
        llm_action_safety_reason: snapshot
            .llm_action_proposal
            .as_ref()
            .and_then(|proposal| proposal.safety_reason.clone()),
    }
}

fn action_debug_value<T>(debug: Option<&serde_json::Value>, key: &str) -> Option<T>
where
    T: serde::de::DeserializeOwned,
{
    debug
        .and_then(|debug| debug.get(key))
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
}

fn scene_stuck_from_snapshot(snapshot: &WorldSnapshot) -> SceneStuck {
    let Some(extension) = snapshot
        .extensions
        .iter()
        .find(|extension| extension.name == "sim.stuck")
    else {
        return SceneStuck {
            dead_battery: snapshot.body.battery_level <= f32::EPSILON && !snapshot.body.charging,
            ..SceneStuck::default()
        };
    };
    let values = &extension.values;
    let active = values.first().copied().unwrap_or(0.0) > 0.0;
    let corner_trap = values.get(1).copied().unwrap_or(0.0) > 0.0;
    let phase_code = values.get(4).copied().unwrap_or(0.0).round() as i32;
    let trap_kind = trap_kind_name(values.get(10).copied().unwrap_or(0.0));
    SceneStuck {
        active,
        class: if active || corner_trap || trap_kind.is_some() {
            Some(
                trap_kind
                    .map(|kind| format!("{kind}-trap"))
                    .unwrap_or_else(|| {
                        if corner_trap {
                            "corner-trap".to_string()
                        } else {
                            "stuck".to_string()
                        }
                    }),
            )
        } else {
            None
        },
        trap_kind: trap_kind.map(str::to_string),
        stuck_ticks: values.get(2).copied().unwrap_or(0.0).max(0.0) as usize,
        duration_ms: values.get(3).copied().unwrap_or(0.0).max(0.0) as u64,
        recovery_phase: match phase_code {
            1 => Some("stop".to_string()),
            2 => Some("reverse".to_string()),
            3 => Some("turn-away".to_string()),
            4 => Some("probe".to_string()),
            _ => None,
        },
        turn_direction: match values.get(5).copied().unwrap_or(0.0) {
            value if value < 0.0 => Some("right".to_string()),
            value if value > 0.0 => Some("left".to_string()),
            _ => None,
        },
        recovery_attempts: values.get(11).copied().unwrap_or(0.0).max(0.0) as usize,
        repeated_trap_count: values.get(12).copied().unwrap_or(0.0).max(0.0) as usize,
        clearance_m: values.get(13).copied().filter(|value| *value >= 0.0),
        event_started: values.get(6).copied().unwrap_or(0.0) > 0.0,
        recovered: values.get(7).copied().unwrap_or(0.0) > 0.0,
        dead_battery: values.get(8).copied().unwrap_or(0.0) > 0.0,
    }
}

fn trap_kind_name(code: f32) -> Option<&'static str> {
    match code.round() as i32 {
        1 => Some("wall"),
        2 => Some("corner"),
        3 => Some("column"),
        _ => None,
    }
}

fn scene_range_from_snapshot(snapshot: &WorldSnapshot) -> SceneRange {
    let beam_count = snapshot.range.beams.len().max(1);
    let explicit_angles = snapshot.range.beam_angles_rad.len() == snapshot.range.beams.len();
    let fov_rad = std::f32::consts::PI;
    SceneRange {
        nearest_m: snapshot.range.nearest_m,
        beams: snapshot
            .range
            .beams
            .iter()
            .enumerate()
            .map(|(index, distance_m)| {
                let ratio = if beam_count <= 1 {
                    0.5
                } else {
                    index as f32 / (beam_count - 1) as f32
                };
                let angle_rad = if explicit_angles {
                    snapshot.range.beam_angles_rad[index]
                } else {
                    -fov_rad * 0.5 + ratio * fov_rad
                };
                SceneRangeBeam {
                    angle_rad: finite_or_zero(angle_rad),
                    distance_m: finite_or_zero(*distance_m),
                    hit: snapshot
                        .range
                        .nearest_m
                        .map(|nearest| (*distance_m - nearest).abs() < 0.05)
                        .unwrap_or(false),
                }
            })
            .collect(),
    }
}

fn scene_eye_from_frame(
    frame: &pete_sensors::EyeFrame,
    retina_status: Option<&RetinaStatusInfo>,
    current_t_ms: u64,
) -> (SceneEye, Vec<String>) {
    let mut warnings = Vec::new();
    let (data_url, encode_warning) = encode_eye_data_url(frame);
    if let Some(warning) = encode_warning {
        warnings.push(warning);
    }
    let stats = eye_frame_stats(frame);
    if stats.mean_luma < 0.08 {
        warnings.push(format!(
            "eye frame is very dim (mean luma {:.3})",
            stats.mean_luma
        ));
    }

    let source = frame.source.clone().unwrap_or_else(|| "none".to_string());
    let authoritative = source == "babylon-robot-eye" || source == "real-camera";
    let retina_connected = retina_status.map(|s| s.connected).unwrap_or(false);

    let retina_last_frame_age_ms = if source == "babylon-robot-eye" {
        Some(current_t_ms.saturating_sub(frame.captured_at_ms))
    } else {
        None
    };

    let frames_received = retina_status.map(|s| s.frames_received).unwrap_or(0);
    let frames_written_to_ledger = retina_status
        .map(|s| s.frames_written_to_ledger)
        .unwrap_or(0);

    (
        SceneEye {
            width: frame.width,
            height: frame.height,
            format: format!("{:?}", frame.format),
            data_url,
            mean_luma: stats.mean_luma,
            non_background_ratio: stats.non_background_ratio,
            source,
            authoritative,
            retina_connected,
            retina_last_frame_age_ms,
            frames_received,
            frames_written_to_ledger,
        },
        warnings,
    )
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct EyeFrameStats {
    mean_luma: f32,
    non_background_ratio: f32,
}

fn eye_frame_stats(frame: &pete_sensors::EyeFrame) -> EyeFrameStats {
    let pixels = frame.width as usize * frame.height as usize;
    if pixels == 0 {
        return EyeFrameStats::default();
    }
    let mut luma_sum = 0.0f32;
    let mut non_background = 0usize;
    match frame.format {
        EyeFrameFormat::Gray8 => {
            for value in frame.bytes.iter().take(pixels) {
                let luma = *value as f32 / 255.0;
                luma_sum += luma;
                if luma > 0.08 {
                    non_background += 1;
                }
            }
        }
        EyeFrameFormat::Rgb8 | EyeFrameFormat::Bgr8 => {
            for pixel in frame.bytes.chunks_exact(3).take(pixels) {
                let (r, g, b) = match frame.format {
                    EyeFrameFormat::Bgr8 => (pixel[2], pixel[1], pixel[0]),
                    _ => (pixel[0], pixel[1], pixel[2]),
                };
                let luma = (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32) / 255.0;
                luma_sum += luma;
                if luma > 0.08 || r.abs_diff(g) > 8 || g.abs_diff(b) > 8 {
                    non_background += 1;
                }
            }
        }
        EyeFrameFormat::Yuyv422 | EyeFrameFormat::Uyvy422 => {
            for pair in frame.bytes.chunks_exact(4).take(pixels.div_ceil(2)) {
                let values = match frame.format {
                    EyeFrameFormat::Uyvy422 => [pair[1], pair[3]],
                    _ => [pair[0], pair[2]],
                };
                for value in values {
                    let luma = value as f32 / 255.0;
                    luma_sum += luma;
                    if luma > 0.08 {
                        non_background += 1;
                    }
                }
            }
        }
        EyeFrameFormat::BayerGrbg8
        | EyeFrameFormat::BayerRggb8
        | EyeFrameFormat::BayerBggr8
        | EyeFrameFormat::BayerGbrg8 => {
            for value in frame.bytes.iter().take(pixels) {
                let luma = *value as f32 / 255.0;
                luma_sum += luma;
                if luma > 0.08 {
                    non_background += 1;
                }
            }
        }
        EyeFrameFormat::Mjpeg | EyeFrameFormat::Unknown(_) => {}
    }
    EyeFrameStats {
        mean_luma: luma_sum / pixels as f32,
        non_background_ratio: non_background as f32 / pixels as f32,
    }
}

fn encode_eye_data_url(frame: &pete_sensors::EyeFrame) -> (Option<String>, Option<String>) {
    match frame.format {
        EyeFrameFormat::Mjpeg => {
            let encoded = base64::engine::general_purpose::STANDARD.encode(&frame.bytes);
            (Some(format!("data:image/jpeg;base64,{encoded}")), None)
        }
        EyeFrameFormat::Rgb8
        | EyeFrameFormat::Bgr8
        | EyeFrameFormat::Gray8
        | EyeFrameFormat::Yuyv422
        | EyeFrameFormat::Uyvy422
        | EyeFrameFormat::BayerGrbg8
        | EyeFrameFormat::BayerRggb8
        | EyeFrameFormat::BayerBggr8
        | EyeFrameFormat::BayerGbrg8 => {
            let rgb = match eye_frame_to_rgb(frame) {
                Ok(rgb) => rgb,
                Err(error) => return (None, Some(error)),
            };
            let mut png = Vec::new();
            let result = PngEncoder::new(&mut png).write_image(
                &rgb,
                frame.width,
                frame.height,
                ColorType::Rgb8.into(),
            );
            match result {
                Ok(()) => {
                    let encoded = base64::engine::general_purpose::STANDARD.encode(png);
                    (Some(format!("data:image/png;base64,{encoded}")), None)
                }
                Err(error) => (None, Some(format!("failed to encode eye PNG: {error}"))),
            }
        }
        EyeFrameFormat::Unknown(ref format) => {
            (None, Some(format!("unsupported eye frame format {format}")))
        }
    }
}

fn bayer8_to_rgb(bytes: &[u8], width: usize, height: usize, format: &EyeFrameFormat) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let (r, g, b) = bayer_pixel_to_rgb(bytes, width, height, x, y, format);
            rgb.extend_from_slice(&[r, g, b]);
        }
    }
    rgb
}

fn bayer_pixel_to_rgb(
    bytes: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    format: &EyeFrameFormat,
) -> (u8, u8, u8) {
    let value = bayer_sample(bytes, width, x, y);
    match bayer_color_at(x, y, format) {
        BayerColor::Red => (
            value,
            average_bayer_neighbors(bytes, width, height, x, y, format, &[BayerColor::Green]),
            average_bayer_neighbors(bytes, width, height, x, y, format, &[BayerColor::Blue]),
        ),
        BayerColor::Green => (
            average_bayer_neighbors(bytes, width, height, x, y, format, &[BayerColor::Red]),
            value,
            average_bayer_neighbors(bytes, width, height, x, y, format, &[BayerColor::Blue]),
        ),
        BayerColor::Blue => (
            average_bayer_neighbors(bytes, width, height, x, y, format, &[BayerColor::Red]),
            average_bayer_neighbors(bytes, width, height, x, y, format, &[BayerColor::Green]),
            value,
        ),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BayerColor {
    Red,
    Green,
    Blue,
}

fn bayer_color_at(x: usize, y: usize, format: &EyeFrameFormat) -> BayerColor {
    let even_x = x % 2 == 0;
    let even_y = y % 2 == 0;
    match format {
        EyeFrameFormat::BayerGrbg8 => match (even_y, even_x) {
            (true, true) | (false, false) => BayerColor::Green,
            (true, false) => BayerColor::Red,
            (false, true) => BayerColor::Blue,
        },
        EyeFrameFormat::BayerRggb8 => match (even_y, even_x) {
            (true, true) => BayerColor::Red,
            (true, false) | (false, true) => BayerColor::Green,
            (false, false) => BayerColor::Blue,
        },
        EyeFrameFormat::BayerBggr8 => match (even_y, even_x) {
            (true, true) => BayerColor::Blue,
            (true, false) | (false, true) => BayerColor::Green,
            (false, false) => BayerColor::Red,
        },
        EyeFrameFormat::BayerGbrg8 => match (even_y, even_x) {
            (true, true) | (false, false) => BayerColor::Green,
            (true, false) => BayerColor::Blue,
            (false, true) => BayerColor::Red,
        },
        _ => BayerColor::Green,
    }
}

fn average_bayer_neighbors(
    bytes: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    format: &EyeFrameFormat,
    colors: &[BayerColor],
) -> u8 {
    let mut sum = 0usize;
    let mut count = 0usize;
    let min_y = y.saturating_sub(1);
    let max_y = (y + 1).min(height.saturating_sub(1));
    let min_x = x.saturating_sub(1);
    let max_x = (x + 1).min(width.saturating_sub(1));
    for ny in min_y..=max_y {
        for nx in min_x..=max_x {
            if nx == x && ny == y {
                continue;
            }
            if colors.contains(&bayer_color_at(nx, ny, format)) {
                sum += bayer_sample(bytes, width, nx, ny) as usize;
                count += 1;
            }
        }
    }
    if count == 0 {
        bayer_sample(bytes, width, x, y)
    } else {
        (sum / count) as u8
    }
}

fn bayer_sample(bytes: &[u8], width: usize, x: usize, y: usize) -> u8 {
    bytes[y * width + x]
}

fn yuyv422_to_rgb(bytes: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(bytes.len() / 2 * 3);
    for pair in bytes.chunks_exact(4) {
        let y0 = pair[0];
        let u = pair[1];
        let y1 = pair[2];
        let v = pair[3];
        push_yuv_rgb(&mut rgb, y0, u, v);
        push_yuv_rgb(&mut rgb, y1, u, v);
    }
    rgb
}

fn uyvy422_to_rgb(bytes: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(bytes.len() / 2 * 3);
    for pair in bytes.chunks_exact(4) {
        let u = pair[0];
        let y0 = pair[1];
        let v = pair[2];
        let y1 = pair[3];
        push_yuv_rgb(&mut rgb, y0, u, v);
        push_yuv_rgb(&mut rgb, y1, u, v);
    }
    rgb
}

fn push_yuv_rgb(rgb: &mut Vec<u8>, y: u8, u: u8, v: u8) {
    let c = y as i32 - 16;
    let d = u as i32 - 128;
    let e = v as i32 - 128;
    rgb.push(((298 * c + 409 * e + 128) >> 8).clamp(0, 255) as u8);
    rgb.push(((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255) as u8);
    rgb.push(((298 * c + 516 * d + 128) >> 8).clamp(0, 255) as u8);
}

fn scene_kinect_from_snapshot(
    snapshot: &WorldSnapshot,
    calibration: Option<SceneSensorCalibration>,
    warnings: &mut Vec<String>,
) -> SceneKinect {
    let color = snapshot
        .eye_frame
        .as_ref()
        .and_then(DepthColorImage::from_eye_frame);
    let (points, diagnostics) = depth_points(&snapshot.kinect, calibration, color.as_ref());
    if points.is_empty() {
        warnings.push("no point cloud stream".to_string());
    }
    if diagnostics.coordinate_system == "depth_image_unknown" {
        warnings.push(
            "Kinect depth frame has no width/height metadata; using legacy approximate projection"
                .to_string(),
        );
    }
    if calibration.is_none() && snapshot.kinect.depth_width > 0 && snapshot.kinect.depth_height > 0
    {
        warnings.push(
            "Kinect depth image has no explicit calibration; accumulated world cloud is disabled"
                .to_string(),
        );
    }
    SceneKinect {
        points,
        accumulated_points: Vec::new(),
        accumulated_summary: None,
        local_world_belief: None,
        skeletons: snapshot.kinect.skeletons.clone(),
        coordinate_system: Some(diagnostics.coordinate_system.clone()),
        diagnostics,
    }
}

fn depth_points(
    kinect: &KinectSense,
    calibration: Option<SceneSensorCalibration>,
    color: Option<&DepthColorImage>,
) -> (Vec<ScenePoint>, SceneKinectDiagnostics) {
    const MAX_POINTS: usize = 12_000;
    let depth_m = &kinect.depth_m;
    if depth_m.is_empty() {
        return (
            Vec::new(),
            SceneKinectDiagnostics {
                coordinate_system: "none".to_string(),
                sample_stride: 1,
                ..SceneKinectDiagnostics::default()
            },
        );
    }
    if let Some(calibration) = calibration {
        if depth_m.len() == calibration.compact_depth_beam_count {
            let points = range_beam_points(depth_m, calibration);
            let mut stats = depth_stats(depth_m, 1, 0.0, 8.0, "scene_robot_render", 0, 0)
                .with_floor_stats(floor_stats_from_scene_points(&points));
            stats.point_coordinate_system =
                Some("scene axes derived from robot math frame".to_string());
            stats.math_frame =
                Some("robot/base: +x forward, +y left, +z up; floor z=0".to_string());
            stats.render_frame = Some("scene: +x left, +y up, +z forward".to_string());
            return (points, stats);
        }
    }
    if let Some(frame) = KinectDepthProjection::from_kinect(kinect) {
        return project_depth_image(depth_m, frame, calibration, color);
    }
    if depth_m.len() == 640 * 480 {
        return project_depth_image(
            depth_m,
            KinectDepthProjection {
                width: 640,
                height: 480,
                fx: 594.0,
                fy: 591.0,
                cx: 339.0,
                cy: 242.0,
                min_depth_m: 0.4,
                max_depth_m: 8.0,
                coordinate_system: "kinect_camera".to_string(),
            },
            calibration,
            color,
        );
    }
    let width = (depth_m.len() as f32).sqrt().ceil().max(1.0) as usize;
    let height = depth_m.len().div_ceil(width).max(1);
    let stride = (depth_m.len().div_ceil(MAX_POINTS)).max(1);
    let points = depth_m
        .iter()
        .enumerate()
        .step_by(stride)
        .filter_map(|(index, depth)| {
            if !depth.is_finite() || *depth <= 0.0 {
                return None;
            }
            let x = index % width;
            let y = index / width;
            let nx = (x as f32 / width as f32) - 0.5;
            let ny = (y as f32 / height as f32) - 0.5;
            let z = (depth * calibration.map_or(1.0, |calibration| calibration.depth_scale))
                .clamp(0.0, 8.0);
            let [r, g, b] = color
                .and_then(|color| {
                    let (offset_x, offset_y) = calibration
                        .map(SceneSensorCalibration::color_offset_px)
                        .unwrap_or_default();
                    color.sample_depth_pixel_with_offset(x, y, width, height, offset_x, offset_y)
                })
                .unwrap_or_else(|| depth_shade(z, 8.0));
            Some(ScenePoint {
                x: nx * z,
                y: ny * z,
                z,
                r,
                g,
                b,
            })
        })
        .collect();
    (
        points,
        depth_stats(
            depth_m,
            stride,
            0.0,
            8.0,
            "depth_image_unknown",
            width as u32,
            height as u32,
        ),
    )
}

#[derive(Clone, Debug)]
struct KinectDepthProjection {
    width: usize,
    height: usize,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    min_depth_m: f32,
    max_depth_m: f32,
    coordinate_system: String,
}

impl KinectDepthProjection {
    fn from_kinect(kinect: &KinectSense) -> Option<Self> {
        let width = usize::try_from(kinect.depth_width).ok()?;
        let height = usize::try_from(kinect.depth_height).ok()?;
        if width == 0 || height == 0 || width.checked_mul(height)? != kinect.depth_m.len() {
            return None;
        }
        let fx = if kinect.depth_fx > 0.0 {
            kinect.depth_fx
        } else {
            594.0
        };
        let fy = if kinect.depth_fy > 0.0 {
            kinect.depth_fy
        } else {
            591.0
        };
        let cx = if kinect.depth_cx > 0.0 {
            kinect.depth_cx
        } else {
            (width as f32 - 1.0) * 0.5
        };
        let cy = if kinect.depth_cy > 0.0 {
            kinect.depth_cy
        } else {
            (height as f32 - 1.0) * 0.5
        };
        let max_depth_m = if kinect.max_depth_m > 0.0 {
            kinect.max_depth_m
        } else {
            8.0
        };
        Some(Self {
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            min_depth_m: kinect.min_depth_m.max(0.0),
            max_depth_m,
            coordinate_system: kinect
                .depth_coordinate_system
                .clone()
                .filter(|system| system != "kinect_depth_image")
                .unwrap_or_else(|| "kinect_camera".to_string()),
        })
    }
}

#[derive(Clone, Debug)]
struct DepthColorImage {
    width: usize,
    height: usize,
    rgb: Vec<u8>,
}

impl DepthColorImage {
    fn from_eye_frame(frame: &EyeFrame) -> Option<Self> {
        let width = usize::try_from(frame.width).ok()?;
        let height = usize::try_from(frame.height).ok()?;
        if width == 0 || height == 0 {
            return None;
        }
        let rgb = eye_frame_to_rgb(frame).ok()?;
        if rgb.len() < width.checked_mul(height)?.checked_mul(3)? {
            return None;
        }
        Some(Self { width, height, rgb })
    }

    fn sample_depth_pixel_with_offset(
        &self,
        depth_x: usize,
        depth_y: usize,
        depth_width: usize,
        depth_height: usize,
        offset_x_px: i32,
        offset_y_px: i32,
    ) -> Option<[u8; 3]> {
        if depth_width == 0 || depth_height == 0 {
            return None;
        }
        let color_x = (depth_x.saturating_mul(self.width) / depth_width).min(self.width - 1);
        let color_y = (depth_y.saturating_mul(self.height) / depth_height).min(self.height - 1);
        self.sample_offset(color_x, color_y, offset_x_px, offset_y_px)
    }

    fn sample_offset(
        &self,
        x: usize,
        y: usize,
        offset_x_px: i32,
        offset_y_px: i32,
    ) -> Option<[u8; 3]> {
        let x = offset_index(x, offset_x_px, self.width)?;
        let y = offset_index(y, offset_y_px, self.height)?;
        self.sample(x, y)
    }

    fn sample(&self, x: usize, y: usize) -> Option<[u8; 3]> {
        let offset = y.checked_mul(self.width)?.checked_add(x)?.checked_mul(3)?;
        Some([
            *self.rgb.get(offset)?,
            *self.rgb.get(offset + 1)?,
            *self.rgb.get(offset + 2)?,
        ])
    }
}

fn offset_index(index: usize, offset: i32, len: usize) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let shifted = index as i64 + offset as i64;
    Some(shifted.clamp(0, len as i64 - 1) as usize)
}

fn depth_shade(depth_m: f32, max_depth_m: f32) -> [u8; 3] {
    let shade = ((1.0 - (depth_m / max_depth_m.max(f32::EPSILON))).clamp(0.15, 1.0) * 255.0) as u8;
    [shade, shade, shade]
}

fn project_depth_image(
    depth_m: &[f32],
    frame: KinectDepthProjection,
    calibration: Option<SceneSensorCalibration>,
    color: Option<&DepthColorImage>,
) -> (Vec<ScenePoint>, SceneKinectDiagnostics) {
    const MAX_POINTS: usize = 2_000;
    let stride = (depth_m.len().div_ceil(MAX_POINTS)).max(1);
    let mut points = Vec::with_capacity(MAX_POINTS.min(depth_m.len()));
    let calibrated = calibration.map(|calibration| DepthExtrinsics::from(calibration));
    let (color_offset_x_px, color_offset_y_px) = calibration
        .map(SceneSensorCalibration::color_offset_px)
        .unwrap_or_default();
    for (index, depth) in depth_m.iter().enumerate().step_by(stride) {
        if !depth.is_finite() || *depth <= 0.0 {
            continue;
        }
        if *depth < frame.min_depth_m || *depth > frame.max_depth_m {
            continue;
        }
        let u = (index % frame.width) as f32;
        let v = (index / frame.width) as f32;
        let z = *depth;
        let x = (u - frame.cx) * z / frame.fx.max(f32::EPSILON);
        let y = (v - frame.cy) * z / frame.fy.max(f32::EPSILON);
        let [r, g, b] = color
            .and_then(|color| {
                color.sample_depth_pixel_with_offset(
                    index % frame.width,
                    index / frame.width,
                    frame.width,
                    frame.height,
                    color_offset_x_px,
                    color_offset_y_px,
                )
            })
            .unwrap_or_else(|| depth_shade(z, frame.max_depth_m));
        let point = if let Some(extrinsics) = calibrated {
            let robot = camera_point_to_robot([x, y, z], extrinsics);
            scene_point_from_robot(robot, r, g, b)
        } else {
            ScenePoint { x, y, z, r, g, b }
        };
        points.push(point);
    }
    let coordinate_system = if calibrated.is_some() {
        "scene_robot_render"
    } else {
        &frame.coordinate_system
    };
    let mut diagnostics = depth_stats(
        depth_m,
        stride,
        frame.min_depth_m,
        frame.max_depth_m,
        coordinate_system,
        frame.width as u32,
        frame.height as u32,
    )
    .with_floor_stats(floor_stats_from_scene_points(&points));
    if calibrated.is_some() {
        diagnostics.point_coordinate_system =
            Some("scene axes derived from robot math frame".to_string());
        diagnostics.math_frame =
            Some("robot/base: +x forward, +y left, +z up; floor z=0".to_string());
        diagnostics.render_frame = Some("scene: +x left, +y up, +z forward".to_string());
    }
    (points, diagnostics)
}

fn scene_point_from_robot(robot: [f32; 3], r: u8, g: u8, b: u8) -> ScenePoint {
    ScenePoint {
        x: robot[1],
        y: robot[2],
        z: robot[0],
        r,
        g,
        b,
    }
}

fn depth_stats(
    depth_m: &[f32],
    sample_stride: usize,
    min_depth_m: f32,
    max_depth_m: f32,
    coordinate_system: &str,
    depth_width: u32,
    depth_height: u32,
) -> SceneKinectDiagnostics {
    let mut valid = Vec::new();
    let mut skipped = 0usize;
    let mut clipped = 0usize;
    for depth in depth_m {
        if !depth.is_finite() || *depth <= 0.0 {
            skipped = skipped.saturating_add(1);
        } else if *depth < min_depth_m || *depth > max_depth_m {
            clipped = clipped.saturating_add(1);
        } else {
            valid.push(*depth);
        }
    }
    valid.sort_by(|left, right| left.total_cmp(right));
    let median_depth_m = if valid.is_empty() {
        None
    } else {
        Some(valid[valid.len() / 2])
    };
    SceneKinectDiagnostics {
        depth_width,
        depth_height,
        valid_depth_count: valid.len(),
        skipped_depth_count: skipped,
        clipped_depth_count: clipped,
        min_depth_m: valid.first().copied(),
        median_depth_m,
        max_depth_m: valid.last().copied(),
        sample_stride,
        coordinate_system: coordinate_system.to_string(),
        point_coordinate_system: None,
        math_frame: None,
        render_frame: None,
        below_floor_count: 0,
        below_floor_ratio: 0.0,
        min_z_m: None,
        median_z_m: None,
        min_math_z_m: None,
        median_math_z_m: None,
        min_render_vertical_m: None,
        median_render_vertical_m: None,
        warnings: Vec::new(),
    }
}

impl SceneKinectDiagnostics {
    fn with_floor_stats(mut self, stats: FloorPointStats) -> Self {
        self.below_floor_count = stats.below_floor_count;
        self.below_floor_ratio = stats.below_floor_ratio;
        self.min_z_m = stats.min_z_m;
        self.median_z_m = stats.median_z_m;
        self.min_render_vertical_m = stats.min_z_m;
        self.median_render_vertical_m = stats.median_z_m;
        if self.coordinate_system == "scene_robot_render" {
            self.min_math_z_m = stats.min_z_m;
            self.median_math_z_m = stats.median_z_m;
        }
        self
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FloorPointStats {
    below_floor_count: usize,
    below_floor_ratio: f32,
    min_z_m: Option<f32>,
    median_z_m: Option<f32>,
}

fn floor_stats_from_scene_points(points: &[ScenePoint]) -> FloorPointStats {
    let mut heights = points
        .iter()
        .map(|point| point.y)
        .filter(|height| height.is_finite())
        .collect::<Vec<_>>();
    if heights.is_empty() {
        return FloorPointStats::default();
    }
    heights.sort_by(|left, right| left.total_cmp(right));
    let below_floor_count = heights.iter().filter(|height| **height < 0.0).count();
    FloorPointStats {
        below_floor_count,
        below_floor_ratio: below_floor_count as f32 / heights.len() as f32,
        min_z_m: heights.first().copied(),
        median_z_m: heights.get(heights.len() / 2).copied(),
    }
}

#[derive(Clone, Copy, Debug)]
struct DepthExtrinsics {
    forward_m: f32,
    height_m: f32,
    pitch_rad: f32,
    roll_rad: f32,
    yaw_rad: f32,
}

impl From<SceneSensorCalibration> for DepthExtrinsics {
    fn from(calibration: SceneSensorCalibration) -> Self {
        Self {
            forward_m: calibration.depth_camera_forward_m(),
            height_m: calibration.depth_camera_height_m(),
            pitch_rad: calibration.depth_camera_pitch_rad(),
            roll_rad: calibration.camera_roll_rad,
            yaw_rad: calibration.camera_yaw_rad,
        }
    }
}

fn camera_point_to_robot(camera: [f32; 3], extrinsics: DepthExtrinsics) -> [f32; 3] {
    let base = [camera[2], -camera[0], -camera[1]];
    apply_robot_extrinsics(base, extrinsics)
}

fn apply_robot_extrinsics(base: [f32; 3], extrinsics: DepthExtrinsics) -> [f32; 3] {
    let rotated = rotate_robot_extrinsic(
        base,
        extrinsics.pitch_rad,
        extrinsics.roll_rad,
        extrinsics.yaw_rad,
    );
    [
        rotated[0] + extrinsics.forward_m,
        rotated[1],
        rotated[2] + extrinsics.height_m,
    ]
}

fn rotate_robot_extrinsic(
    point: [f32; 3],
    pitch_rad: f32,
    roll_rad: f32,
    yaw_rad: f32,
) -> [f32; 3] {
    let (pitch_sin, pitch_cos) = pitch_rad.sin_cos();
    let mut x = point[0] * pitch_cos + point[2] * pitch_sin;
    let y = point[1];
    let mut z = -point[0] * pitch_sin + point[2] * pitch_cos;

    let (roll_sin, roll_cos) = roll_rad.sin_cos();
    let rolled_y = y * roll_cos - z * roll_sin;
    z = y * roll_sin + z * roll_cos;

    let (yaw_sin, yaw_cos) = yaw_rad.sin_cos();
    let yawed_x = x * yaw_cos - rolled_y * yaw_sin;
    let yawed_y = x * yaw_sin + rolled_y * yaw_cos;
    x = yawed_x;

    [x, yawed_y, z]
}

fn range_beam_points(depth_m: &[f32], calibration: SceneSensorCalibration) -> Vec<ScenePoint> {
    let beam_count = depth_m.len().max(1);
    let extrinsics = DepthExtrinsics::from(calibration);
    let fov_rad = calibration
        .compact_depth_fov_rad
        .clamp(0.01, std::f32::consts::TAU);
    let start = if beam_count == 1 { 0.0 } else { -fov_rad * 0.5 };
    let step = if beam_count == 1 {
        0.0
    } else {
        fov_rad / (beam_count - 1) as f32
    };
    depth_m
        .iter()
        .enumerate()
        .filter_map(|(index, depth)| {
            if !depth.is_finite() || *depth <= 0.0 {
                return None;
            }
            let distance = (depth * calibration.depth_scale).clamp(0.0, 8.0);
            let angle = start + step * index as f32;
            let shade = ((1.0 - (distance / 8.0)).clamp(0.15, 1.0) * 255.0) as u8;
            let robot = apply_robot_extrinsics(
                [angle.cos() * distance, angle.sin() * distance, 0.0],
                extrinsics,
            );
            Some(scene_point_from_robot(robot, shade, shade, shade))
        })
        .collect()
}

fn audio_bearing_from_objects(
    robot_x_m: f32,
    robot_y_m: f32,
    metadata: Option<&LiveSceneMetadata>,
) -> Option<f32> {
    metadata
        .into_iter()
        .flat_map(|metadata| metadata.objects.iter())
        .find(|object| {
            object.kind == "person" || object.kind == "speaker" || object.kind == "sound_source"
        })
        .map(|object| (object.y_m - robot_y_m).atan2(object.x_m - robot_x_m))
}

fn pcm_audio_energy(frame: &pete_sensors::PcmAudioFrame) -> f32 {
    if frame.samples.is_empty() {
        return 0.0;
    }
    let mean_square = frame
        .samples
        .iter()
        .map(|sample| {
            let normalized = *sample as f32 / i16::MAX as f32;
            normalized * normalized
        })
        .sum::<f32>()
        / frame.samples.len() as f32;
    mean_square.sqrt().clamp(0.0, 1.0)
}

fn scene_object_kind(debug_kind: &str) -> String {
    let lower = debug_kind.to_ascii_lowercase();
    if lower.contains("charger") {
        "charger"
    } else if lower.contains("person") {
        "person"
    } else if lower.contains("sound") || lower.contains("speaker") {
        "speaker"
    } else if lower.contains("landmark") {
        "landmark"
    } else {
        "obstacle"
    }
    .to_string()
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

