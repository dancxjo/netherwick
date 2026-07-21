#[derive(Debug)]
pub struct LiveViewError {
    status: StatusCode,
    message: String,
}

impl LiveViewError {
    fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
        }
    }

    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
        }
    }

    fn forbidden(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            message: message.into(),
        }
    }
}

impl IntoResponse for LiveViewError {
    fn into_response(self) -> axum::response::Response {
        (
            self.status,
            Json(serde_json::json!({
                "error": self.message,
            })),
        )
            .into_response()
    }
}

async fn post_reign_command(
    State(state): State<ReignServerState>,
    Json(request): Json<ReignCommandRequest>,
) -> Result<Json<ReignInput>, ReignApiError> {
    Ok(Json(enqueue_reign_command(&state, request)?))
}

async fn get_reign_command_ws(
    ws: WebSocketUpgrade,
    State(state): State<ReignServerState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| stream_reign_commands(socket, state))
}

async fn stream_reign_commands(mut socket: WebSocket, state: ReignServerState) {
    while let Some(message) = socket.recv().await {
        let response = match message {
            Ok(Message::Text(text)) => reign_ws_response(&state, text.as_str()),
            Ok(Message::Binary(bytes)) => match std::str::from_utf8(&bytes) {
                Ok(text) => reign_ws_response(&state, text),
                Err(error) => ReignCommandWsResponse {
                    accepted: false,
                    input: None,
                    error: Some(format!("invalid utf-8 websocket command: {error}")),
                },
            },
            Ok(Message::Close(_)) => break,
            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => continue,
            Err(error) => {
                let response = ReignCommandWsResponse {
                    accepted: false,
                    input: None,
                    error: Some(format!("websocket command receive failed: {error}")),
                };
                let _ = send_reign_ws_response(&mut socket, &response).await;
                break;
            }
        };
        if send_reign_ws_response(&mut socket, &response)
            .await
            .is_err()
        {
            break;
        }
    }
}

fn reign_ws_response(state: &ReignServerState, text: &str) -> ReignCommandWsResponse {
    match serde_json::from_str::<ReignCommandRequest>(text) {
        Ok(request) => match enqueue_reign_command(state, request) {
            Ok(input) => ReignCommandWsResponse {
                accepted: true,
                input: Some(input),
                error: None,
            },
            Err(error) => ReignCommandWsResponse {
                accepted: false,
                input: None,
                error: Some(error.message),
            },
        },
        Err(error) => ReignCommandWsResponse {
            accepted: false,
            input: None,
            error: Some(format!("invalid reign command json: {error}")),
        },
    }
}

async fn send_reign_ws_response(
    socket: &mut WebSocket,
    response: &ReignCommandWsResponse,
) -> Result<(), axum::Error> {
    let text = serde_json::to_string(response).unwrap_or_else(|_| {
        "{\"accepted\":false,\"input\":null,\"error\":\"reign response serialization failed\"}"
            .to_string()
    });
    socket.send(Message::Text(text.into())).await
}

fn enqueue_reign_command(
    state: &ReignServerState,
    request: ReignCommandRequest,
) -> Result<ReignInput, ReignApiError> {
    let now_ms = wall_now_ms();
    let source = request.source.unwrap_or(ReignSource::WebRemote);
    let mut command = sanitize_command(request.command)?;
    let mut ttl_ms = request.ttl_ms.unwrap_or_else(|| command.default_ttl_ms());
    if state.hardware_control_status(now_ms).available && hardware_gate_applies_to_command(&command)
    {
        command = sanitize_hardware_command(command)?;
        ttl_ms = ttl_ms.clamp(HARDWARE_TTL_MIN_MS, HARDWARE_TTL_MAX_MS);
        enforce_hardware_command_gate(&state, &source, &request.mode, &command, now_ms)?;
    }
    let input = ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: now_ms,
        expires_at_ms: now_ms.saturating_add(ttl_ms.clamp(REIGN_TTL_MIN_MS, REIGN_TTL_MAX_MS)),
        source,
        mode: request.mode,
        command,
        priority: request.priority.clamp(0.0, 1.0),
        note: request.note.filter(|note| !note.trim().is_empty()),
    };
    state
        .queue
        .lock()
        .map_err(|_| ReignApiError::internal("reign queue lock poisoned"))?
        .push(input.clone());
    Ok(input)
}

async fn post_hardware_arm(
    State(state): State<ReignServerState>,
    Json(request): Json<HardwareArmRequest>,
) -> Result<Json<HardwareControlStatus>, ReignApiError> {
    Ok(Json(state.set_hardware_armed(request.armed)?))
}

async fn post_reign_prod(
    State(state): State<ReignServerState>,
    request: Option<Json<ProdRequest>>,
) -> Result<Json<ProdResponse>, ReignApiError> {
    let request = request.map(|Json(request)| request).unwrap_or_default();
    let snapshot = state.latest_snapshot().ok_or_else(|| {
        ReignApiError::bad_request("cannot prod before a live scene snapshot exists")
    })?;
    let command = prod_command_from_request(request)?;
    let action = command
        .to_action()
        .ok_or_else(|| ReignApiError::bad_request("prod command cannot be converted to action"))?;
    let policy = NudgePolicy::virtual_default();
    if let Some(reason) = nudge_action_block_reason_for_snapshot(&snapshot, &action, policy) {
        return Err(ReignApiError::forbidden(format!("prod refused: {reason}")));
    }

    let now_ms = wall_now_ms();
    let ttl_ms = command.default_ttl_ms().clamp(500, 5_000);
    let input = ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: now_ms,
        expires_at_ms: now_ms.saturating_add(ttl_ms),
        source: ReignSource::HumanSupervisor,
        mode: ReignMode::Assist,
        command,
        priority: 0.8,
        note: Some("manual prod".to_string()),
    };
    state
        .queue
        .lock()
        .map_err(|_| ReignApiError::internal("reign queue lock poisoned"))?
        .push(input.clone());
    Ok(Json(ProdResponse {
        accepted: true,
        input,
    }))
}

async fn get_reign_state(
    State(state): State<ReignServerState>,
) -> Result<Json<ReignSense>, ReignApiError> {
    let now_ms = wall_now_ms();
    let mut queue = state
        .queue
        .lock()
        .map_err(|_| ReignApiError::internal("reign queue lock poisoned"))?;
    queue.drain_expired(now_ms);
    Ok(Json(queue.sense(now_ms)))
}

fn prod_command_from_request(request: ProdRequest) -> Result<ReignCommand, ReignApiError> {
    let duration_ms = request.duration_ms.unwrap_or(1_500).clamp(250, 3_000);
    let intensity = match request.intensity {
        Some(value) if value.is_finite() => value,
        Some(_) => return Err(ReignApiError::bad_request("intensity must be finite")),
        None => 0.12,
    };
    let intensity = intensity.clamp(0.0, 0.25);
    let kind = request
        .kind
        .as_deref()
        .unwrap_or("explore")
        .trim()
        .to_ascii_lowercase();
    let command = match kind.as_str() {
        "" | "explore" | "random_walk" | "random-walk" => ReignCommand::Explore { duration_ms },
        "go" | "forward" => ReignCommand::Go {
            intensity: intensity.min(0.15),
            duration_ms: duration_ms.min(1_000),
        },
        "reverse" | "back" | "backward" => ReignCommand::Reverse {
            intensity: intensity.min(0.15),
            duration_ms: duration_ms.min(1_000),
        },
        "turn" | "left" => ReignCommand::Turn {
            direction: TurnDir::Left,
            intensity: intensity.min(0.25),
            duration_ms: duration_ms.min(1_000),
        },
        "right" => ReignCommand::Turn {
            direction: TurnDir::Right,
            intensity: intensity.min(0.25),
            duration_ms: duration_ms.min(1_000),
        },
        other => {
            return Err(ReignApiError::bad_request(format!(
                "unsupported prod kind '{other}'"
            )))
        }
    };
    sanitize_command(command)
}

async fn post_reign_clear(
    State(state): State<ReignServerState>,
) -> Result<Json<ReignSense>, ReignApiError> {
    let now_ms = wall_now_ms();
    let mut queue = state
        .queue
        .lock()
        .map_err(|_| ReignApiError::internal("reign queue lock poisoned"))?;
    queue.clear();
    Ok(Json(queue.sense(now_ms)))
}

async fn reign_page() -> Html<&'static str> {
    Html(REIGN_PAGE)
}

#[derive(Debug)]
pub struct ReignApiError {
    status: StatusCode,
    message: String,
}

impl ReignApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }

    fn forbidden(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            message: message.into(),
        }
    }
}

impl IntoResponse for ReignApiError {
    fn into_response(self) -> axum::response::Response {
        (
            self.status,
            Json(serde_json::json!({
                "error": self.message,
            })),
        )
            .into_response()
    }
}

fn sanitize_command(command: ReignCommand) -> Result<ReignCommand, ReignApiError> {
    Ok(match command {
        ReignCommand::Go {
            intensity,
            duration_ms,
        } => ReignCommand::Go {
            intensity: finite_intensity(intensity)?,
            duration_ms: duration_ms.clamp(50, 10_000),
        },
        ReignCommand::Reverse {
            intensity,
            duration_ms,
        } => ReignCommand::Reverse {
            intensity: finite_intensity(intensity)?,
            duration_ms: duration_ms.clamp(50, 10_000),
        },
        ReignCommand::Drive {
            forward,
            turn,
            duration_ms,
        } => ReignCommand::Drive {
            forward: finite_axis(forward)?,
            turn: finite_axis(turn)?,
            duration_ms: duration_ms.clamp(50, 10_000),
        },
        ReignCommand::Turn {
            direction,
            intensity,
            duration_ms,
        } => ReignCommand::Turn {
            direction,
            intensity: finite_intensity(intensity)?,
            duration_ms: duration_ms.clamp(50, 10_000),
        },
        ReignCommand::Explore { duration_ms } => ReignCommand::Explore {
            duration_ms: duration_ms.clamp(250, 30_000),
        },
        ReignCommand::Speak { text } => {
            let text = text.trim();
            if text.is_empty() {
                return Err(ReignApiError::bad_request("speak text cannot be empty"));
            }
            ReignCommand::Speak {
                text: text.chars().take(500).collect(),
            }
        }
        ReignCommand::Chirp { pattern } => ReignCommand::Chirp { pattern },
        other => other,
    })
}

fn hardware_gate_applies_to_command(command: &ReignCommand) -> bool {
    matches!(
        command,
        ReignCommand::Stop
            | ReignCommand::Go { .. }
            | ReignCommand::Reverse { .. }
            | ReignCommand::Drive { .. }
            | ReignCommand::Turn { .. }
    )
}

fn sanitize_hardware_command(command: ReignCommand) -> Result<ReignCommand, ReignApiError> {
    Ok(match command {
        ReignCommand::Go {
            intensity,
            duration_ms,
        } => ReignCommand::Go {
            intensity: finite_intensity(intensity)?.min(HARDWARE_MAX_FORWARD_INTENSITY),
            duration_ms: duration_ms.clamp(HARDWARE_TTL_MIN_MS, HARDWARE_TTL_MAX_MS),
        },
        ReignCommand::Reverse {
            intensity,
            duration_ms,
        } => ReignCommand::Reverse {
            intensity: finite_intensity(intensity)?.min(HARDWARE_MAX_FORWARD_INTENSITY),
            duration_ms: duration_ms.clamp(HARDWARE_TTL_MIN_MS, HARDWARE_TTL_MAX_MS),
        },
        ReignCommand::Drive {
            forward,
            turn,
            duration_ms,
        } => ReignCommand::Drive {
            forward: finite_axis(forward)?.clamp(
                -HARDWARE_MAX_FORWARD_INTENSITY,
                HARDWARE_MAX_FORWARD_INTENSITY,
            ),
            turn: finite_axis(turn)?
                .clamp(-HARDWARE_MAX_TURN_INTENSITY, HARDWARE_MAX_TURN_INTENSITY),
            duration_ms: duration_ms.clamp(HARDWARE_TTL_MIN_MS, HARDWARE_TTL_MAX_MS),
        },
        ReignCommand::Turn {
            direction,
            intensity,
            duration_ms,
        } => ReignCommand::Turn {
            direction,
            intensity: finite_intensity(intensity)?.min(HARDWARE_MAX_TURN_INTENSITY),
            duration_ms: duration_ms.clamp(HARDWARE_TTL_MIN_MS, HARDWARE_TTL_MAX_MS),
        },
        ReignCommand::Stop => ReignCommand::Stop,
        _ => {
            return Err(ReignApiError::forbidden(
                "hardware cockpit only accepts Stop, Go, Reverse, Drive, and Turn",
            ))
        }
    })
}

fn enforce_hardware_command_gate(
    state: &ReignServerState,
    source: &ReignSource,
    mode: &ReignMode,
    command: &ReignCommand,
    now_ms: TimeMs,
) -> Result<(), ReignApiError> {
    if matches!(command, ReignCommand::Stop) {
        return Ok(());
    }
    if !matches!(source, ReignSource::WebRemote | ReignSource::Gamepad) {
        return Err(ReignApiError::forbidden(
            "hardware cockpit only accepts WebRemote or Gamepad drive commands",
        ));
    }
    if mode != &ReignMode::Direct {
        return Err(ReignApiError::forbidden(
            "hardware cockpit drive commands must use Direct mode",
        ));
    }
    let status = state.hardware_control_status(now_ms);
    if !status.armed {
        return Err(ReignApiError::forbidden(
            "hardware cockpit is disarmed; movement rejected",
        ));
    }
    if let Some(reason) = status.reason {
        return Err(ReignApiError::forbidden(format!(
            "hardware cockpit movement rejected: {reason}"
        )));
    }
    Ok(())
}

fn finite_intensity(value: f32) -> Result<f32, ReignApiError> {
    if !value.is_finite() {
        return Err(ReignApiError::bad_request("intensity must be finite"));
    }
    Ok(value.clamp(0.0, 1.0))
}

fn finite_axis(value: f32) -> Result<f32, ReignApiError> {
    if !value.is_finite() {
        return Err(ReignApiError::bad_request("drive axis must be finite"));
    }
    Ok(value.clamp(-1.0, 1.0))
}

fn default_priority() -> f32 {
    0.8
}

fn default_reign_mode() -> ReignMode {
    ReignMode::Direct
}

fn wall_now_ms() -> TimeMs {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as TimeMs)
        .unwrap_or_default()
}

