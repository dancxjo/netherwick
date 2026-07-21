#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BehaviorNodesResponse {
    pub schema_version: u32,
    pub nodes: Vec<BehaviorNodeState>,
    pub connections: Vec<ModelConnection>,
}

async fn get_behavior_nodes(State(state): State<LiveViewState>) -> Json<BehaviorNodesResponse> {
    Json(BehaviorNodesResponse {
        schema_version: 1,
        nodes: state.behavior_nodes(),
        connections: model_connections(),
    })
}

async fn post_behavior_node(
    State(state): State<LiveViewState>,
    AxumPath(id): AxumPath<String>,
    Json(update): Json<BehaviorNodeUpdate>,
) -> Result<Json<BehaviorNodeState>, LiveViewError> {
    state
        .update_behavior_node(&id, update)
        .map(Json)
        .ok_or_else(|| LiveViewError::not_found(format!("behavior node {id} was not found")))
}

async fn post_promote_behavior_node(
    State(state): State<LiveViewState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<BehaviorNodeState>, LiveViewError> {
    let current = state
        .behavior_nodes()
        .into_iter()
        .find(|node| {
            same_behavior_node(&node.node_id, &id) || same_behavior_node(&node.behavior_id, &id)
        })
        .ok_or_else(|| LiveViewError::not_found(format!("behavior node {id} was not found")))?;
    let gates_pass = current
        .last_run
        .as_ref()
        .and_then(|run| run.error.as_ref())
        .is_none()
        && current
            .last_run
            .as_ref()
            .and_then(|run| run.disagreement)
            .map(|value| value <= 0.25)
            .unwrap_or(true)
        && !current.missing_model_or_checkpoint;
    if !gates_pass {
        return Err(LiveViewError::bad_request(
            "promotion safety gates failed: resolve errors, disagreement, or missing model/checkpoint",
        ));
    }
    let update = BehaviorNodeUpdate {
        selected_regime: Some(BehaviorRegime::ModelInfer),
        ..BehaviorNodeUpdate::default()
    };
    state
        .update_behavior_node(&id, update)
        .map(Json)
        .ok_or_else(|| LiveViewError::not_found(format!("behavior node {id} was not found")))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetinaStatusInfo {
    pub enabled: bool,
    pub connected: bool,
    pub frames_received: usize,
    pub frames_written_to_ledger: usize,
}

#[derive(Deserialize)]
pub struct RetinaFramePayload {
    pub schema_version: u32,
    pub source: String,
    pub t_ms: u64,
    pub frame_index: usize,
    pub width: u32,
    pub height: u32,
    pub format: String,
    pub encoding: String,
    pub data: String,
}

async fn post_retina_frame(
    State(state): State<LiveViewState>,
    Json(payload): Json<RetinaFramePayload>,
) -> Result<StatusCode, LiveViewError> {
    let session = state.session();
    let is_virtual_live = session
        .as_ref()
        .map(|s| s.mode == "virtual-live")
        .unwrap_or(false);
    if !is_virtual_live {
        return Err(LiveViewError::forbidden(
            "retina frames are only accepted in virtual-live mode",
        ));
    }

    let mut base64_str = payload.data.as_str();
    if let Some(pos) = base64_str.find(',') {
        base64_str = &base64_str[pos + 1..];
    }
    let decoded_bytes = base64::engine::general_purpose::STANDARD
        .decode(base64_str)
        .map_err(|e| LiveViewError::bad_request(format!("invalid base64 image data: {e}")))?;

    let img = image::load_from_memory(&decoded_bytes)
        .map_err(|e| LiveViewError::bad_request(format!("failed to decode image format: {e}")))?;

    let width = img.width();
    let height = img.height();

    if width > state.retina_width || height > state.retina_height {
        return Err(LiveViewError::bad_request(format!(
            "retina frame dimensions ({}x{}) exceed configured limits ({}x{})",
            width, height, state.retina_width, state.retina_height
        )));
    }

    let sim_t_ms = state
        .latest()
        .map(|snap| snap.body.last_update_ms)
        .unwrap_or(0);
    if sim_t_ms > 0 && payload.t_ms + 1500 < sim_t_ms {
        return Err(LiveViewError::bad_request("retina frame is too stale"));
    }

    let raw_rgb = img.to_rgb8().into_raw();

    let format = match payload.format.as_str() {
        "Rgb8" => pete_sensors::EyeFrameFormat::Rgb8,
        "Bgr8" => pete_sensors::EyeFrameFormat::Bgr8,
        "Gray8" => pete_sensors::EyeFrameFormat::Gray8,
        "Mjpeg" => pete_sensors::EyeFrameFormat::Mjpeg,
        other => pete_sensors::EyeFrameFormat::Unknown(other.to_string()),
    };

    let eye_frame = pete_sensors::EyeFrame {
        captured_at_ms: payload.t_ms,
        width,
        height,
        format,
        bytes: raw_rgb,
        source: Some(payload.source.clone()),
    };

    {
        let mut rstate = state
            .retina_state
            .lock()
            .expect("retina state mutex poisoned");
        rstate.latest_frame = Some(eye_frame);
        rstate.has_new_frame = true;
        rstate.last_received_at = Some(std::time::Instant::now());
        rstate.frames_received += 1;
    }

    Ok(StatusCode::OK)
}

async fn post_calibration(
    State(state): State<LiveViewState>,
    Json(calibration): Json<SceneSensorCalibration>,
) -> Result<StatusCode, LiveViewError> {
    let mut metadata = state.scene_metadata().unwrap_or_default();
    metadata.sensor_calibration = Some(calibration);
    state.update_scene_metadata(metadata);
    Ok(StatusCode::OK)
}

#[derive(Clone, Debug, Serialize)]
pub struct InlineLearningControlResponse {
    pub config: InlineLearningConfig,
    pub enabled: bool,
    pub training_mode: String,
    pub weights_updating: bool,
}

fn inline_learning_response(config: InlineLearningConfig) -> InlineLearningControlResponse {
    InlineLearningControlResponse {
        enabled: config.is_enabled(),
        training_mode: config.training_mode_label().to_string(),
        weights_updating: config.is_enabled(),
        config,
    }
}

async fn get_inline_learning(
    State(state): State<LiveViewState>,
) -> Json<InlineLearningControlResponse> {
    Json(inline_learning_response(state.inline_learning()))
}

async fn post_inline_learning(
    State(state): State<LiveViewState>,
    Json(mut config): Json<InlineLearningConfig>,
) -> Result<Json<InlineLearningControlResponse>, LiveViewError> {
    if config.max_train_steps_per_tick > 64 {
        return Err(LiveViewError::bad_request(
            "max_train_steps_per_tick must be 64 or less",
        ));
    }
    if config.mode == InlineLearningMode::Off {
        config.max_train_steps_per_tick = config.max_train_steps_per_tick.max(1);
    }
    state.update_inline_learning(config.clone());
    Ok(Json(inline_learning_response(config)))
}

#[derive(Serialize)]
pub struct RetinaStatusResponse {
    pub enabled: bool,
    pub connected: bool,
    pub source: String,
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub frames_received: usize,
    pub frames_attached_to_snapshots: usize,
    pub frames_written_to_ledger: usize,
    pub latest_age_ms: Option<u64>,
    pub latest_luma: Option<f32>,
    pub warnings: Vec<String>,
}

async fn get_retina_status(State(state): State<LiveViewState>) -> Json<RetinaStatusResponse> {
    let rstate = state
        .retina_state
        .lock()
        .expect("retina state mutex poisoned");
    let connected = state.virtual_retina
        && rstate
            .last_received_at
            .map(|t| t.elapsed() < std::time::Duration::from_millis(1500))
            .unwrap_or(false);
    let latest_age_ms = rstate
        .last_received_at
        .map(|t| t.elapsed().as_millis() as u64);
    let latest_luma = rstate.latest_frame.as_ref().map(|frame| {
        let stats = eye_frame_stats(frame);
        stats.mean_luma
    });
    let source = rstate
        .latest_frame
        .as_ref()
        .and_then(|f| f.source.clone())
        .unwrap_or_else(|| "none".to_string());

    let mut warnings = rstate.warnings.clone();
    if state.virtual_retina {
        if rstate.frames_received == 0 {
            warnings.push("no retina frame received yet".to_string());
        } else if !connected {
            warnings.push("retina frame stream is stale/disconnected".to_string());
        }
    }

    Json(RetinaStatusResponse {
        enabled: state.virtual_retina,
        connected,
        source,
        width: state.retina_width,
        height: state.retina_height,
        fps: state.retina_fps,
        frames_received: rstate.frames_received,
        frames_attached_to_snapshots: rstate.frames_attached_to_snapshots,
        frames_written_to_ledger: rstate.frames_written_to_ledger,
        latest_age_ms,
        latest_luma,
        warnings,
    })
}

fn encode_eye_png_bytes(frame: &pete_sensors::EyeFrame) -> Result<Vec<u8>, String> {
    match frame.format {
        EyeFrameFormat::Mjpeg => Ok(frame.bytes.clone()),
        EyeFrameFormat::Rgb8
        | EyeFrameFormat::Bgr8
        | EyeFrameFormat::Gray8
        | EyeFrameFormat::Yuyv422
        | EyeFrameFormat::Uyvy422
        | EyeFrameFormat::BayerGrbg8
        | EyeFrameFormat::BayerRggb8
        | EyeFrameFormat::BayerBggr8
        | EyeFrameFormat::BayerGbrg8 => {
            let rgb = eye_frame_to_rgb(frame)?;
            let mut png = Vec::new();
            let result = PngEncoder::new(&mut png).write_image(
                &rgb,
                frame.width,
                frame.height,
                ColorType::Rgb8.into(),
            );
            match result {
                Ok(()) => Ok(png),
                Err(error) => Err(format!("failed to encode eye PNG: {error}")),
            }
        }
        EyeFrameFormat::Unknown(ref format) => {
            Err(format!("unsupported eye frame format {format}"))
        }
    }
}

fn eye_frame_to_rgb(frame: &pete_sensors::EyeFrame) -> Result<Vec<u8>, String> {
    let expected_len = match frame.format {
        EyeFrameFormat::Gray8
        | EyeFrameFormat::BayerGrbg8
        | EyeFrameFormat::BayerRggb8
        | EyeFrameFormat::BayerBggr8
        | EyeFrameFormat::BayerGbrg8 => frame.width as usize * frame.height as usize,
        EyeFrameFormat::Yuyv422 | EyeFrameFormat::Uyvy422 => {
            frame.width as usize * frame.height as usize * 2
        }
        EyeFrameFormat::Rgb8 | EyeFrameFormat::Bgr8 => {
            frame.width as usize * frame.height as usize * 3
        }
        EyeFrameFormat::Mjpeg | EyeFrameFormat::Unknown(_) => {
            return Err(format!("unsupported eye frame format {:?}", frame.format));
        }
    };
    if frame.bytes.len() < expected_len {
        return Err(format!(
            "eye frame has {} bytes, expected at least {}",
            frame.bytes.len(),
            expected_len
        ));
    }
    let bytes = &frame.bytes[..expected_len];
    let mut rgb = Vec::with_capacity(frame.width as usize * frame.height as usize * 3);
    match frame.format {
        EyeFrameFormat::Rgb8 => rgb.extend_from_slice(bytes),
        EyeFrameFormat::Bgr8 => {
            for pixel in bytes.chunks_exact(3) {
                rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
            }
        }
        EyeFrameFormat::Gray8 => {
            for value in bytes {
                rgb.extend_from_slice(&[*value, *value, *value]);
            }
        }
        EyeFrameFormat::Yuyv422 => {
            rgb.extend(yuyv422_to_rgb(bytes));
        }
        EyeFrameFormat::Uyvy422 => {
            rgb.extend(uyvy422_to_rgb(bytes));
        }
        EyeFrameFormat::BayerGrbg8
        | EyeFrameFormat::BayerRggb8
        | EyeFrameFormat::BayerBggr8
        | EyeFrameFormat::BayerGbrg8 => {
            rgb.extend(bayer8_to_rgb(
                bytes,
                frame.width as usize,
                frame.height as usize,
                &frame.format,
            ));
        }
        EyeFrameFormat::Mjpeg | EyeFrameFormat::Unknown(_) => {}
    }
    Ok(rgb)
}

async fn get_retina_latest(State(state): State<LiveViewState>) -> impl IntoResponse {
    let rstate = state
        .retina_state
        .lock()
        .expect("retina state mutex poisoned");
    if let Some(ref frame) = rstate.latest_frame {
        match encode_eye_png_bytes(frame) {
            Ok(bytes) => {
                let content_type = match frame.format {
                    EyeFrameFormat::Mjpeg => "image/jpeg",
                    _ => "image/png",
                };
                (
                    StatusCode::OK,
                    [(axum::http::header::CONTENT_TYPE, content_type)],
                    bytes,
                )
                    .into_response()
            }
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to encode frame: {err}"),
            )
                .into_response(),
        }
    } else {
        (StatusCode::NOT_FOUND, "no retina frame received yet").into_response()
    }
}

async fn get_models() -> Json<ModelsResponse> {
    Json(read_models_response(&default_model_root()))
}

async fn get_latest_training() -> impl IntoResponse {
    let report_path = Path::new("data/reports/virtual/latest.json");
    if report_path.exists() {
        if let Ok(content) = std::fs::read_to_string(report_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                return (StatusCode::OK, Json(json)).into_response();
            }
        }
    }
    let alt_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/reports/virtual/latest.json");
    if alt_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&alt_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                return (StatusCode::OK, Json(json)).into_response();
            }
        }
    }
    (
        StatusCode::OK,
        Json(serde_json::json!({ "status": "none" })),
    )
        .into_response()
}

fn default_model_root() -> PathBuf {
    let root = PathBuf::from("data/models");
    if root.exists() {
        root
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/models")
    }
}

fn read_models_response(root: &Path) -> ModelsResponse {
    let mut response = ModelsResponse {
        schema_version: 1,
        root: root.display().to_string(),
        connections: model_connections(),
        behavior_nodes: default_behavior_nodes(),
        ..ModelsResponse::default()
    };
    let registry = read_model_registry_summary(&root.join("registry.json"), &mut response.warnings);
    response.registry = registry.clone();

    let mut models = Vec::new();
    match fs::read_dir(root) {
        Ok(entries) => {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                if let Some(summary) = read_model_summary(&path, &registry) {
                    models.push(summary);
                }
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => response
            .warnings
            .push(format!("model root {} was not found", root.display())),
        Err(error) => response.warnings.push(format!(
            "failed to read model root {}: {error}",
            root.display()
        )),
    }

    for registered in &registry {
        if models
            .iter()
            .all(|model| model.checkpoint_path != registered.checkpoint_path)
        {
            models.push(ModelSummary {
                name: registered.name.clone(),
                behavior: Some(registered.behavior.clone()),
                checkpoint_path: registered.checkpoint_path.clone(),
                registered_status: Some(registered.status.clone()),
                allowed_modes: registered.allowed_modes.clone(),
                warnings: vec!["registered checkpoint has no local metadata summary".to_string()],
                ..ModelSummary::default()
            });
        }
    }

    models.sort_by(|left, right| {
        left.behavior
            .cmp(&right.behavior)
            .then_with(|| left.name.cmp(&right.name))
    });
    response.models = models;
    response
}

fn read_model_summary(path: &Path, registry: &[ModelRegistrySummary]) -> Option<ModelSummary> {
    let name = path.file_name()?.to_string_lossy().to_string();
    let checkpoint_path = path.display().to_string();
    let metadata = read_json_value(&path.join("metadata.json")).ok();
    let evaluation = read_json_value(&path.join("evaluation.json")).ok();
    let registry_entry = registry
        .iter()
        .find(|entry| same_path_text(&entry.checkpoint_path, &checkpoint_path));

    let behavior = evaluation
        .as_ref()
        .and_then(|value| value.get("behavior"))
        .and_then(|value| value.as_str())
        .or_else(|| registry_entry.map(|entry| entry.behavior.as_str()))
        .or_else(|| behavior_from_checkpoint_name(&name))
        .map(str::to_string);

    let mut warnings = Vec::new();
    if metadata.is_none() {
        warnings.push("metadata.json missing".to_string());
    }
    if evaluation.is_none() {
        warnings.push("evaluation.json missing".to_string());
    }

    Some(ModelSummary {
        name,
        behavior,
        checkpoint_path,
        samples_seen: metadata
            .as_ref()
            .and_then(|value| json_u64(value, "samples_seen")),
        best_loss: metadata
            .as_ref()
            .and_then(|value| json_f32(value, "best_loss")),
        input_dim: metadata
            .as_ref()
            .and_then(|value| json_u64(value, "input_dim")),
        output_dim: metadata
            .as_ref()
            .and_then(|value| json_u64(value, "output_dim")),
        latent_dim: metadata
            .as_ref()
            .and_then(|value| json_u64(value, "latent_dim")),
        width: metadata.as_ref().and_then(|value| json_u64(value, "width")),
        height: metadata
            .as_ref()
            .and_then(|value| json_u64(value, "height")),
        evaluation: evaluation.as_ref().map(model_evaluation_summary),
        metrics: read_metric_summary(&path.join("metrics.jsonl")),
        registered_status: registry_entry.map(|entry| entry.status.clone()),
        allowed_modes: registry_entry
            .map(|entry| entry.allowed_modes.clone())
            .unwrap_or_default(),
        warnings,
    })
}

fn model_evaluation_summary(value: &serde_json::Value) -> ModelEvaluationSummary {
    ModelEvaluationSummary {
        sample_count: json_u64(value, "sample_count"),
        model_loss_mean: json_f32(value, "model_loss_mean"),
        hardcoded_loss_mean: json_f32(value, "hardcoded_loss_mean"),
        selected_loss_mean: json_f32(value, "selected_loss_mean"),
        model_better_than_hardcoded: value
            .get("model_better_than_hardcoded")
            .and_then(|value| value.as_bool()),
        improvement_ratio: json_f32(value, "improvement_ratio"),
        recommendation: value
            .get("recommendation")
            .and_then(|value| value.as_str())
            .map(str::to_string),
    }
}

fn read_metric_summary(path: &Path) -> Option<ModelTrainingMetricSummary> {
    let text = fs::read_to_string(path).ok()?;
    let mut summary = ModelTrainingMetricSummary::default();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        summary.record_count += 1;
        summary.last_epoch = json_u64(&value, "epoch");
        summary.last_sample_index = json_u64(&value, "sample_index");
        summary.last_train_loss = json_f32(&value, "train_loss");
        summary.last_model_loss = json_f32(&value, "model_loss");
        summary.last_hardcoded_loss = json_f32(&value, "hardcoded_loss");
        summary.last_selected_loss = json_f32(&value, "selected_loss");
    }
    (summary.record_count > 0).then_some(summary)
}

fn read_model_registry_summary(
    path: &Path,
    warnings: &mut Vec<String>,
) -> Vec<ModelRegistrySummary> {
    let Ok(value) = read_json_value(path) else {
        warnings.push(format!(
            "model registry {} was not readable",
            path.display()
        ));
        return Vec::new();
    };
    value
        .get("entries")
        .and_then(|value| value.as_array())
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| {
                    Some(ModelRegistrySummary {
                        name: entry.get("name")?.as_str()?.to_string(),
                        behavior: entry.get("behavior")?.as_str()?.to_string(),
                        checkpoint_path: entry.get("checkpoint")?.as_str()?.to_string(),
                        training_ledger: entry
                            .get("training")
                            .and_then(|value| value.get("ledger"))
                            .and_then(|value| value.as_str())
                            .map(str::to_string),
                        behavior_report_path: entry
                            .get("reports")
                            .and_then(|value| value.get("behavior"))
                            .and_then(|value| value.as_str())
                            .map(str::to_string),
                        scenario_report_path: entry
                            .get("reports")
                            .and_then(|value| value.get("scenario"))
                            .and_then(|value| value.as_str())
                            .map(str::to_string),
                        status: entry
                            .get("status")
                            .and_then(|value| value.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                        allowed_modes: entry
                            .get("allowed_modes")
                            .and_then(|value| value.as_array())
                            .map(|values| {
                                values
                                    .iter()
                                    .filter_map(|value| value.as_str().map(str::to_string))
                                    .collect()
                            })
                            .unwrap_or_default(),
                        scenario_success_rate: entry
                            .get("metrics")
                            .and_then(|value| json_f32(value, "scenario_success_rate")),
                        collision_rate: entry
                            .get("metrics")
                            .and_then(|value| json_f32(value, "collision_rate")),
                        episodes: entry
                            .get("metrics")
                            .and_then(|value| json_u64(value, "episodes")),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn model_connections() -> Vec<ModelConnection> {
    [
        ("Sensors", "Now", "snapshot"),
        ("Now", "Experience", "encode"),
        ("Now", "Danger", "range/body/action"),
        ("Now", "Charge", "battery/dock cues"),
        ("Now", "EyeNext", "vision context"),
        ("Now", "EarNext", "audio context"),
        ("Now", "EventBump", "body bump event"),
        ("Now", "SocialWorld", "presence + identity"),
        ("Experience", "Future", "latent z"),
        ("EventBump", "Autonomic", "scripted escape"),
        ("SocialWorld", "GreetGoal", "new encounter"),
        ("GreetGoal", "Conductor", "eligible goal"),
        ("Danger", "Conductor", "risk"),
        ("Charge", "Conductor", "dock value"),
        ("Future", "Conductor", "imagined state"),
        ("Conductor", "ActionValue", "candidate actions"),
        ("Conductor", "LuaSkills", "foreground skill"),
        ("LuaSkills", "Body", "bounded organs"),
        ("ActionValue", "Autonomic", "ranked action"),
        ("Autonomic", "Body", "safe command"),
        ("Body", "Ledger", "transition"),
        ("Ledger", "Training", "replay"),
        ("Training", "Models", "checkpoints"),
    ]
    .into_iter()
    .map(|(from, to, label)| ModelConnection {
        from: from.to_string(),
        to: to.to_string(),
        label: label.to_string(),
    })
    .collect()
}

fn read_json_value(path: &Path) -> Result<serde_json::Value, io::Error> {
    let text = fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn json_f32(value: &serde_json::Value, key: &str) -> Option<f32> {
    value
        .get(key)
        .and_then(|value| value.as_f64())
        .filter(|value| value.is_finite())
        .map(|value| value as f32)
}

fn json_u64(value: &serde_json::Value, key: &str) -> Option<u64> {
    value.get(key).and_then(|value| value.as_u64())
}

fn same_path_text(left: &str, right: &str) -> bool {
    left.trim_end_matches('/') == right.trim_end_matches('/')
}

fn behavior_from_checkpoint_name(name: &str) -> Option<&'static str> {
    if name.starts_with("action_value") {
        Some("action_value")
    } else if name.starts_with("eye") {
        Some("eye_next")
    } else if name.starts_with("ear") {
        Some("ear_next")
    } else if name.starts_with("danger") {
        Some("danger")
    } else if name.starts_with("charge") {
        Some("charge")
    } else if name.starts_with("future") {
        Some("future")
    } else if name.starts_with("experience") {
        Some("experience")
    } else {
        None
    }
}

