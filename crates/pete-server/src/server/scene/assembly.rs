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

async fn observatory_page() -> Html<&'static str> {
    Html(OBSERVATORY_PAGE)
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
