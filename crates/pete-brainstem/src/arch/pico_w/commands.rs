fn handle_command_request<'a>(
    request: &[u8],
    buffer: &'a mut [u8],
) -> Result<&'a str, CommandParseError> {
    let body = request_body(request).ok_or(CommandParseError::BadRequest)?;
    let command_id = json_u32(body, "command_id").ok_or(CommandParseError::BadRequest)?;
    if json_str(body, "kind") == Some("register_network_endpoint") {
        return handle_network_registration_json(body, buffer).ok_or(CommandParseError::BadRequest);
    }
    if json_str(body, "kind") == Some("acquire_control_lease") {
        return handle_authority_json(body, buffer).ok_or(CommandParseError::BadRequest);
    }
    if json_str(body, "kind") == Some("acquire_service_lease") {
        return handle_service_authority_json(body, buffer).ok_or(CommandParseError::BadRequest);
    }
    let command = parse_command(command_id, body).ok_or(CommandParseError::BadRequest)?;
    if command_id == 0 {
        return render_command_response(buffer, false, command_id, "invalid_command_id")
            .ok_or(CommandParseError::BadRequest);
    }
    if matches!(command, BrainstemCommand::Status) {
        let snapshot = status::snapshot(Instant::now().as_millis() as u32);
        return status::render_json(snapshot, buffer).map_err(|_| CommandParseError::BadRequest);
    }
    if let BrainstemCommand::GetEvents { since_seq } = command {
        return status::render_events_json(since_seq, buffer).ok_or(CommandParseError::BadRequest);
    }
    if matches!(command, BrainstemCommand::GetCapabilities) {
        return render_capabilities_response(buffer, command_id)
            .ok_or(CommandParseError::BadRequest);
    }
    if matches!(command, BrainstemCommand::Ping) {
        return render_command_response(buffer, true, command_id, "pong")
            .ok_or(CommandParseError::BadRequest);
    }
    if matches!(command, BrainstemCommand::Bootsel) && !json_service_authority_valid(body) {
        return render_command_response(
            buffer,
            false,
            command_id,
            "service_authorization_required",
        )
        .ok_or(CommandParseError::BadRequest);
    }
    if command_requires_session(command) && !json_session_valid(body) {
        return render_command_response(buffer, false, command_id, "invalid_session")
            .ok_or(CommandParseError::BadRequest);
    }
    if command_requires_authority(command) && !json_authority_valid(body) {
        return render_command_response(buffer, false, command_id, "invalid_control_lease")
            .ok_or(CommandParseError::BadRequest);
    }
    if command_requires_service_authority(command) && !json_service_authority_valid(body) {
        return render_command_response(buffer, false, command_id, "invalid_service_lease")
            .ok_or(CommandParseError::BadRequest);
    }
    if matches!(command, BrainstemCommand::Bootsel) {
        return render_command_response(
            buffer,
            cfg!(feature = "service-mode"),
            command_id,
            if cfg!(feature = "service-mode") {
                "bootsel_accepted"
            } else {
                "service_operation_disabled"
            },
        )
        .ok_or(CommandParseError::BadRequest);
    }
    if let Err(reason) = submit_json_control_command(command_id, command, body) {
        return Err(CommandParseError::Busy(command_id, reason.as_str()));
    }
    render_command_response(buffer, true, command_id, "accepted")
        .ok_or(CommandParseError::BadRequest)
}

fn handle_websocket_message<'a>(body: &str, buffer: &'a mut [u8]) -> Option<&'a str> {
    if json_str(body, "kind") == Some("hello") {
        return handle_handshake_json(body, buffer, TransportKind::WebSocket as u8);
    }
    if json_str(body, "kind") == Some("register_network_endpoint") {
        return handle_network_registration_json(body, buffer);
    }
    if json_str(body, "kind") == Some("acquire_control_lease") {
        return handle_authority_json(body, buffer);
    }
    if json_str(body, "kind") == Some("acquire_service_lease") {
        return handle_service_authority_json(body, buffer);
    }
    if json_str(body, "kind") == Some("status") {
        let snapshot = status::snapshot(Instant::now().as_millis() as u32);
        return render_status_websocket_response(snapshot, buffer);
    }
    if json_str(body, "kind") == Some("get_capabilities") {
        let command_id = json_u32(body, "command_id")?;
        return render_capabilities_response(buffer, command_id);
    }
    if json_str(body, "kind") == Some("get_events") {
        let since_seq = json_u32(body, "since_seq")?;
        return status::render_events_json(since_seq, buffer);
    }
    if json_str(body, "kind") == Some("ping") {
        let command_id = json_u32(body, "command_id")?;
        return render_command_response(buffer, true, command_id, "pong");
    }

    if json_bool(body, "ack") == Some(false) {
        let command_id = json_u32(body, "command_id")?;
        let command = parse_command(command_id, body)?;
        if command_id == 0 {
            return render_command_response(buffer, false, command_id, "invalid_command_id");
        }
        if matches!(command, BrainstemCommand::Bootsel) && !json_service_authority_valid(body) {
            return render_command_response(
                buffer,
                false,
                command_id,
                "service_authorization_required",
            );
        }
        if command_requires_session(command) && !json_session_valid(body) {
            return render_command_response(buffer, false, command_id, "invalid_session");
        }
        if command_requires_authority(command) && !json_authority_valid(body) {
            return render_command_response(buffer, false, command_id, "invalid_control_lease");
        }
        if command_requires_service_authority(command) && !json_service_authority_valid(body) {
            return render_command_response(buffer, false, command_id, "invalid_service_lease");
        }
        if matches!(command, BrainstemCommand::Bootsel) {
            return render_command_response(
                buffer,
                false,
                command_id,
                "service_operation_disabled",
            );
        }
        match submit_json_control_command(command_id, command, body) {
            Ok(()) => None,
            Err(reason) => render_command_response(buffer, false, command_id, reason.as_str()),
        }
    } else {
        handle_websocket_command(body, buffer)
            .or(Some("{\"accepted\":false,\"message\":\"bad_request\"}\n"))
    }
}

fn handle_websocket_command<'a>(body: &str, buffer: &'a mut [u8]) -> Option<&'a str> {
    let command_id = json_u32(body, "command_id")?;
    let command = parse_command(command_id, body)?;
    if command_id == 0 {
        return render_command_response(buffer, false, command_id, "invalid_command_id");
    }
    if matches!(command, BrainstemCommand::Bootsel) && !json_service_authority_valid(body) {
        return render_command_response(
            buffer,
            false,
            command_id,
            "service_authorization_required",
        );
    }
    if command_requires_session(command) && !json_session_valid(body) {
        return render_command_response(buffer, false, command_id, "invalid_session");
    }
    if command_requires_authority(command) && !json_authority_valid(body) {
        return render_command_response(buffer, false, command_id, "invalid_control_lease");
    }
    if command_requires_service_authority(command) && !json_service_authority_valid(body) {
        return render_command_response(buffer, false, command_id, "invalid_service_lease");
    }
    if matches!(command, BrainstemCommand::Bootsel) {
        return render_command_response(buffer, false, command_id, "service_operation_disabled");
    }
    if let Err(reason) = submit_json_control_command(command_id, command, body) {
        return render_command_response(buffer, false, command_id, reason.as_str());
    }
    render_command_response(buffer, true, command_id, "accepted")
}

fn render_status_websocket_response<'a>(
    snapshot: status::BrainstemStatus,
    buffer: &'a mut [u8],
) -> Option<&'a str> {
    let body_len = {
        let body = status::render_json(snapshot, buffer).ok()?;
        body.len()
    };
    let prefix = br#"{"type":"status","#;
    let extra_len = prefix.len().checked_sub(1)?;
    let new_len = body_len.checked_add(extra_len)?;
    if new_len > buffer.len() {
        return None;
    }

    for i in (1..body_len).rev() {
        buffer[i + extra_len] = buffer[i];
    }
    buffer[..prefix.len()].copy_from_slice(prefix);
    core::str::from_utf8(&buffer[..new_len]).ok()
}

fn handle_forebrain_uart_line(uart: &mut Uart<'static, Blocking>, line: &[u8]) {
    if line.is_empty() {
        return;
    }

    let line = match core::str::from_utf8(line) {
        Ok(line) => line,
        Err(_) => {
            status::mark_forebrain_uart_error(status::ForebrainUartErrorCode::Utf8);
            submit_forebrain_stop();
            write_forebrain_uart_line(uart, b"ERR 0 utf8\n");
            return;
        }
    };

    if let Some(body) = line.strip_prefix("HELLO ") {
        let mut response = [0u8; 4096];
        if let Some(welcome) =
            handle_handshake_json(body, &mut response, TransportKind::HardwareUart as u8)
        {
            let prefix = if welcome.contains("\"kind\":\"reject\"") {
                b"REJECT ".as_slice()
            } else {
                b"WELCOME ".as_slice()
            };
            write_forebrain_uart_line(uart, prefix);
            write_forebrain_uart_line(uart, welcome.as_bytes());
            write_forebrain_uart_line(uart, b"\n");
        } else {
            write_forebrain_uart_line(uart, b"ERR 0 handshake\n");
        }
        return;
    }
    if line.starts_with("REGISTER_NETWORK_ENDPOINT ") {
        let mut response = heapless::String::<512>::new();
        handle_network_registration_compact(line, &mut response);
        write_forebrain_uart_line(uart, response.as_bytes());
        return;
    }
    if line.starts_with("ACQUIRE_CONTROL_LEASE ") {
        let mut response = heapless::String::<512>::new();
        handle_authority_compact(line, &mut response);
        write_forebrain_uart_line(uart, response.as_bytes());
        return;
    }
    if line.starts_with("ACQUIRE_SERVICE_LEASE ") {
        let mut response = heapless::String::<512>::new();
        handle_service_authority_compact(line, &mut response);
        write_forebrain_uart_line(uart, response.as_bytes());
        return;
    }

    let (command_line, session_id, lease_id, service_lease_id) = compact_envelope(line);
    let (seq, command) = match parse_forebrain_uart_command(command_line) {
        Ok(parsed) => parsed,
        Err(seq) => {
            status::mark_forebrain_uart_error(status::ForebrainUartErrorCode::Parse);
            submit_forebrain_stop();
            write_forebrain_uart_error(uart, seq, "parse");
            return;
        }
    };

    status::mark_forebrain_uart_command(seq, Instant::now().as_millis() as u32);
    if matches!(command, BrainstemCommand::Bootsel) {
        if !compact_service_authority_valid(session_id, service_lease_id, command) {
            write_forebrain_uart_error(uart, seq, "service_authorization_required");
            return;
        }
        write_forebrain_uart_error(uart, seq, "service_operation_disabled");
        return;
    }
    if matches!(command, BrainstemCommand::GetCapabilities) {
        write_forebrain_uart_capabilities(uart, seq);
        return;
    }
    if let BrainstemCommand::GetEvents { since_seq } = command {
        write_forebrain_uart_events(uart, seq, since_seq);
        return;
    }
    if command_requires_session(command) && !session_id.is_some_and(compact_session_valid) {
        write_forebrain_uart_error(uart, seq, "invalid_session");
        return;
    }
    if command_requires_authority(command)
        && !compact_authority_valid(command, session_id, lease_id)
    {
        write_forebrain_uart_error(uart, seq, "invalid_control_lease");
        return;
    }
    if command_requires_service_authority(command)
        && !compact_service_authority_valid(session_id, service_lease_id, command)
    {
        write_forebrain_uart_error(uart, seq, "invalid_service_lease");
        return;
    }

    if let Err(reason) = submit_compact_control_command(seq, command, session_id, service_lease_id)
    {
        status::mark_forebrain_uart_error(status::ForebrainUartErrorCode::Busy);
        if matches!(command, BrainstemCommand::CmdVel { .. }) {
            submit_forebrain_stop();
        }
        write_forebrain_uart_error(uart, seq, reason.as_str());
        return;
    }

    if matches!(command, BrainstemCommand::Status) {
        write_forebrain_uart_status(uart, seq);
    } else {
        write_forebrain_uart_ok(uart, seq);
    }
}

fn parse_forebrain_uart_command(line: &str) -> Result<(u32, BrainstemCommand), u32> {
    let mut parts = line.split_ascii_whitespace();
    let Some(kind) = parts.next() else {
        return Err(0);
    };
    let seq = parse_u32(parts.next()).ok_or(0u32)?;
    if seq == 0 {
        return Err(0);
    }

    let command = match kind {
        "PING" => BrainstemCommand::Ping,
        "BOOTSEL" => BrainstemCommand::Bootsel,
        "RESET_MOTHERBRAIN" => BrainstemCommand::ResetMotherbrain,
        "ARM" => BrainstemCommand::Arm,
        "DISARM" => BrainstemCommand::Disarm,
        "SET_MODE" => {
            BrainstemCommand::SetMode(parse_oi_mode(parts.next().ok_or(seq)?).ok_or(seq)?)
        }
        "STOP" => BrainstemCommand::Stop,
        "ESTOP" => BrainstemCommand::EStop,
        "CLEAR_ESTOP" => BrainstemCommand::ClearEStop,
        "CMD_VEL" => BrainstemCommand::CmdVel {
            seq,
            linear_mm_s: parse_i16(parts.next()).ok_or(seq)?,
            angular_mrad_s: parse_i16(parts.next()).ok_or(seq)?,
            ttl_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "DRIVE_DIRECT" => BrainstemCommand::DriveDirect {
            seq,
            left_mm_s: parse_i16(parts.next()).ok_or(seq)?,
            right_mm_s: parse_i16(parts.next()).ok_or(seq)?,
            ttl_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "DRIVE_ARC" => BrainstemCommand::DriveArc {
            seq,
            velocity_mm_s: parse_i16(parts.next()).ok_or(seq)?,
            radius_mm: parse_i16(parts.next()).ok_or(seq)?,
            ttl_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "FACE_BEARING" | "TRACK_BEARING" | "TURN_BY" | "DRIVE_FOR" | "BUMP_ESCAPE"
        | "HOLD_HEADING" | "TURN_TO_HEADING" | "ARC_FOR" | "CREEP_UNTIL" | "SCAN_ARC"
        | "DOCK_ALIGN" | "WALL_FOLLOW" | "WIGGLE_ALIGN" | "UNSTICK" | "CLIFF_GUARD"
        | "SET_SAFETY_POLICY" => BrainstemCommand::Unsupported { seq },
        "CLEAR_SAFETY_LATCH" => BrainstemCommand::ClearSafetyLatch {
            seq,
            kind: parse_safety_latch_kind(parts.next().ok_or(seq)?).ok_or(seq)?,
        },
        "CAREFUL_MODE" => BrainstemCommand::CarefulMode {
            seq,
            ttl_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "ESCAPE_MOTION" => BrainstemCommand::EscapeMotion {
            seq,
            kind: parse_safety_latch_kind(parts.next().ok_or(seq)?).ok_or(seq)?,
            hazard_generation: parse_u32(parts.next()).ok_or(seq)?,
            linear_mm_s: parse_i16(parts.next()).ok_or(seq)?,
            angular_mrad_s: parse_i16(parts.next()).ok_or(seq)?,
            ttl_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "HEARTBEAT_STOP" => BrainstemCommand::HeartbeatStop {
            seq,
            timeout_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "REQUEST_SENSORS" => BrainstemCommand::RequestSensors {
            seq,
            packet_id: parse_u32(parts.next()).ok_or(seq)? as u8,
        },
        "STREAM_SENSORS" => BrainstemCommand::StreamSensors {
            seq,
            enabled: parse_bool(parts.next()).ok_or(seq)?,
            packet_id: parse_u32(parts.next()).ok_or(seq)? as u8,
            period_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "CLEAR_MOTION_QUEUE" => BrainstemCommand::ClearMotionQueue { seq },
        "DEFINE_CHIRP" => {
            let kind = parse_feedback_kind(parts.next().ok_or(seq)?).ok_or(seq)?;
            let mut tones = [SongTone::default(); MAX_SONG_TONES];
            let mut tone_count = 0;
            while tone_count < MAX_SONG_TONES {
                let Some(note) = parts.next() else {
                    break;
                };
                let duration = parts.next().ok_or(seq)?;
                tones[tone_count] = SongTone {
                    note: parse_u32(Some(note)).ok_or(seq)? as u8,
                    duration_64ths: parse_u32(Some(duration)).ok_or(seq)? as u8,
                };
                tone_count += 1;
            }
            if tone_count == 0 {
                return Err(seq);
            }
            BrainstemCommand::DefineChirp {
                kind,
                tones,
                tone_count: tone_count as u8,
                seq,
            }
        }
        "PLAY_FEEDBACK" => BrainstemCommand::PlayFeedback {
            seq,
            kind: parse_feedback_kind(parts.next().ok_or(seq)?).ok_or(seq)?,
        },
        "SET_SILENT" => BrainstemCommand::SetAudioSilent {
            seq,
            silent: parse_bool(parts.next()).ok_or(seq)?,
        },
        "POWER_STATE" => BrainstemCommand::PowerState {
            seq,
            request: parse_power_request(parts.next().ok_or(seq)?).ok_or(seq)?,
        },
        "CREATE_POWER_ON" => BrainstemCommand::PowerState {
            seq,
            request: PowerStateRequest::Wake,
        },
        "CREATE_POWER_OFF" => BrainstemCommand::PowerState {
            seq,
            request: PowerStateRequest::Sleep,
        },
        "CALIBRATE_TURN" => BrainstemCommand::CalibrateTurn {
            seq,
            angular_mrad_s: parse_i16(parts.next()).ok_or(seq)?,
            duration_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "ORIENTATION_PROBE" => BrainstemCommand::OrientationProbe {
            seq,
            angular_mrad_s: parse_i16(parts.next()).ok_or(seq)?,
            duration_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "RESET_ODOMETRY" => BrainstemCommand::ResetOdometry { seq },
        "ZERO_IMU_ORIENTATION" => BrainstemCommand::ZeroImuOrientation { seq },
        "CLEAR_IMU_ORIENTATION" => BrainstemCommand::ClearImuOrientation { seq },
        "GET_CAPABILITIES" => BrainstemCommand::GetCapabilities,
        "GET_EVENTS" => BrainstemCommand::GetEvents {
            since_seq: parse_u32(parts.next()).ok_or(seq)?,
        },
        "STATUS" => BrainstemCommand::Status,
        "SONG_PLAY" => BrainstemCommand::SongPlay {
            id: parse_u32(parts.next()).ok_or(seq)? as u8,
        },
        "SONG_DEFINE" => {
            let id = parse_u32(parts.next()).ok_or(seq)? as u8;
            let mut tones = [SongTone::default(); MAX_SONG_TONES];
            let mut tone_count = 0;
            while tone_count < MAX_SONG_TONES {
                let Some(note) = parts.next() else {
                    break;
                };
                let duration = parts.next().ok_or(seq)?;
                tones[tone_count] = SongTone {
                    note: parse_u32(Some(note)).ok_or(seq)? as u8,
                    duration_64ths: parse_u32(Some(duration)).ok_or(seq)? as u8,
                };
                tone_count += 1;
            }
            if tone_count == 0 {
                return Err(seq);
            }
            BrainstemCommand::SongDefine {
                id,
                tones,
                tone_count: tone_count as u8,
                seq,
            }
        }
        "DOCK" => BrainstemCommand::Dock,
        "SET_LIGHTS" => {
            let led_bits = parse_u32(parts.next()).ok_or(seq)?;
            let color = parse_u32(parts.next()).ok_or(seq)?;
            let intensity = parse_u32(parts.next()).ok_or(seq)?;
            if led_bits > 0x0f || color > u8::MAX as u32 || intensity > u8::MAX as u32 {
                return Err(seq);
            }
            BrainstemCommand::SetLights {
                led_bits: led_bits as u8,
                color: color as u8,
                intensity: intensity as u8,
            }
        }
        _ => return Err(seq),
    };

    if !matches!(command, BrainstemCommand::Unsupported { .. }) && parts.next().is_some() {
        return Err(seq);
    }

    Ok((seq, command))
}

fn handle_compact_control_line<const N: usize>(
    line: &str,
    response: &mut heapless::String<N>,
    transport: u8,
) -> Option<bool> {
    response.clear();
    if let Some(body) = line.strip_prefix("HELLO ") {
        let mut buffer = [0u8; 4096];
        if let Some(welcome) = handle_handshake_json(body, &mut buffer, transport) {
            let prefix = if welcome.contains("\"kind\":\"reject\"") {
                "REJECT "
            } else {
                "WELCOME "
            };
            response.push_str(prefix).ok()?;
            response.push_str(welcome).ok()?;
            response.push('\n').ok()?;
            return Some(false);
        }
        return None;
    }
    if line.starts_with("REGISTER_NETWORK_ENDPOINT ") {
        handle_network_registration_compact(line, response);
        return Some(false);
    }
    if line.starts_with("ACQUIRE_CONTROL_LEASE ") {
        handle_authority_compact(line, response);
        return Some(false);
    }
    if line.starts_with("ACQUIRE_SERVICE_LEASE ") {
        handle_service_authority_compact(line, response);
        return Some(false);
    }
    let (command_line, session_id, lease_id, service_lease_id) = compact_envelope(line);
    let (seq, command) = match parse_forebrain_uart_command(command_line) {
        Ok(parsed) => parsed,
        Err(seq) => {
            let _ = writeln!(response, "ERR {seq} parse");
            return Some(false);
        }
    };

    match command {
        BrainstemCommand::Status | BrainstemCommand::Ping => {
            if write_compact_status_line(response, seq).is_err() {
                response.clear();
                let _ = writeln!(response, "ERR {seq} status_too_large");
            }
            Some(false)
        }
        BrainstemCommand::GetCapabilities => {
            let _ = capabilities::write_compact(response, &capabilities::current(), seq);
            Some(false)
        }
        BrainstemCommand::GetEvents { since_seq } => {
            let _ = write!(response, "OK {seq} ");
            let _ = status::write_compact_events(response, since_seq);
            Some(false)
        }
        BrainstemCommand::Bootsel => {
            if compact_service_authority_valid(session_id, service_lease_id, command)
                && cfg!(feature = "service-mode")
            {
                let _ = writeln!(response, "OK {seq} bootsel_accepted");
                Some(true)
            } else {
                let reason = if cfg!(feature = "service-mode") {
                    "service_authorization_required"
                } else {
                    "service_operation_disabled"
                };
                let _ = writeln!(response, "ERR {seq} {reason}");
                Some(false)
            }
        }
        command => {
            if command_requires_session(command) && !session_id.is_some_and(compact_session_valid) {
                let _ = writeln!(response, "ERR {seq} invalid_session");
                return Some(false);
            }
            if command_requires_authority(command)
                && !compact_authority_valid(command, session_id, lease_id)
            {
                let _ = writeln!(response, "ERR {seq} invalid_control_lease");
                return Some(false);
            }
            if command_requires_service_authority(command)
                && !compact_service_authority_valid(session_id, service_lease_id, command)
            {
                let _ = writeln!(response, "ERR {seq} invalid_service_lease");
                return Some(false);
            }
            match submit_compact_control_command(seq, command, session_id, service_lease_id) {
                Ok(()) => {
                    let _ = writeln!(response, "OK {seq}");
                }
                Err(reason) => {
                    let _ = writeln!(response, "ERR {seq} {}", reason.as_str());
                }
            }
            Some(false)
        }
    }
}

fn command_requires_session(command: BrainstemCommand) -> bool {
    !matches!(
        command,
        BrainstemCommand::Status
            | BrainstemCommand::Ping
            | BrainstemCommand::GetCapabilities
            | BrainstemCommand::GetEvents { .. }
            | BrainstemCommand::Stop
            | BrainstemCommand::EStop
            | BrainstemCommand::Unsupported { .. }
    )
}
fn command_requires_authority(command: BrainstemCommand) -> bool {
    command_requires_session(command)
        && !matches!(
            command,
            BrainstemCommand::Disarm | BrainstemCommand::SetAudioSilent { .. }
        )
        && !matches!(
            command,
            BrainstemCommand::RequestSensors { .. } | BrainstemCommand::StreamSensors { .. }
        )
        && !command_requires_service_authority(command)
}
fn command_requires_service_authority(command: BrainstemCommand) -> bool {
    matches!(
        command,
        BrainstemCommand::Bootsel
            | BrainstemCommand::RestartCreate
            | BrainstemCommand::ResetMotherbrain
    )
}

fn submit_json_control_command(
    command_id: u32,
    command: BrainstemCommand,
    body: &str,
) -> Result<(), status::CommandRejectReason> {
    if command_requires_service_authority(command) {
        let Some((session_hash, lease_hash)) = json_service_identity(body) else {
            return Err(status::CommandRejectReason::Unsupported);
        };
        status::submit_service_control_command(command_id, command, session_hash, lease_hash)
    } else {
        status::submit_control_command(command_id, command)
    }
}

fn submit_compact_control_command(
    command_id: u32,
    command: BrainstemCommand,
    session_id: Option<&str>,
    service_lease_id: Option<&str>,
) -> Result<(), status::CommandRejectReason> {
    if command_requires_service_authority(command) {
        let (Some(session_id), Some(lease_id)) = (session_id, service_lease_id) else {
            return Err(status::CommandRejectReason::Unsupported);
        };
        status::submit_service_control_command(
            command_id,
            command,
            session::token_hash(session_id),
            session::token_hash(lease_id),
        )
    } else {
        status::submit_control_command(command_id, command)
    }
}

fn json_service_identity(body: &str) -> Option<(u32, u32)> {
    Some((
        session::token_hash(json_str(body, "session_id")?),
        session::token_hash(json_str(body, "service_lease_id")?),
    ))
}

fn json_session_valid(body: &str) -> bool {
    json_str(body, "session_id").is_some_and(compact_session_valid)
}
fn json_authority_valid(body: &str) -> bool {
    let Some(session_id) = json_str(body, "session_id") else {
        return false;
    };
    let Some(lease_id) = json_str(body, "lease_id") else {
        return false;
    };
    let session_hash = session::token_hash(session_id);
    let lease_hash = session::token_hash(lease_id);
    let now = Instant::now().as_millis() as u32;
    if json_str(body, "kind") == Some("careful_mode")
        && (status::session_role(session_hash) != Some(3) || !cfg!(feature = "operator-debug"))
    {
        false
    } else if json_str(body, "kind") == Some("heartbeat_stop") {
        status::authority_heartbeat_valid(session_hash, lease_hash, now)
    } else {
        status::active_authority_matches(session_hash, lease_hash, now)
    }
}
fn json_service_authority_valid(body: &str) -> bool {
    let Some((session_hash, lease_hash)) = json_service_identity(body) else {
        return false;
    };
    status::active_service_authority_matches(
        session_hash,
        lease_hash,
        Instant::now().as_millis() as u32,
        json_str(body, "kind")
            .and_then(service_scope_code)
            .unwrap_or(0),
    )
}
fn compact_authority_valid(
    command: BrainstemCommand,
    session_id: Option<&str>,
    lease_id: Option<&str>,
) -> bool {
    match (session_id, lease_id) {
        (Some(session_id), Some(lease)) => {
            let session_hash = session::token_hash(session_id);
            let lease_hash = session::token_hash(lease);
            let now = Instant::now().as_millis() as u32;
            if matches!(command, BrainstemCommand::CarefulMode { .. })
                && (status::session_role(session_hash) != Some(3)
                    || !cfg!(feature = "operator-debug"))
            {
                false
            } else if let BrainstemCommand::HeartbeatStop { .. } = command {
                status::authority_heartbeat_valid(session_hash, lease_hash, now)
            } else {
                status::active_authority_matches(session_hash, lease_hash, now)
            }
        }
        _ => false,
    }
}

fn compact_session_valid(session_id: &str) -> bool {
    status::active_session_matches(session::token_hash(session_id))
}

fn compact_session(line: &str) -> (&str, Option<&str>) {
    let (command, session, _, _) = compact_envelope(line);
    (command, session)
}
fn compact_envelope(line: &str) -> (&str, Option<&str>, Option<&str>, Option<&str>) {
    let (without_service, service_lease) = match line.rsplit_once(" service_lease_id=") {
        Some((command, lease)) if !lease.contains(' ') => (command, Some(lease)),
        _ => (line, None),
    };
    let (without_lease, lease) = match without_service.rsplit_once(" lease_id=") {
        Some((command, lease)) if !lease.contains(' ') => (command, Some(lease)),
        _ => (without_service, None),
    };
    match without_lease.rsplit_once(" session_id=") {
        Some((command, session_id)) if !session_id.contains(' ') => {
            (command, Some(session_id), lease, service_lease)
        }
        _ => (without_lease, None, lease, service_lease),
    }
}

fn compact_service_authority_valid(
    session_id: Option<&str>,
    lease_id: Option<&str>,
    command: BrainstemCommand,
) -> bool {
    let scope = match command {
        BrainstemCommand::Bootsel => 1,
        BrainstemCommand::RestartCreate => 3,
        BrainstemCommand::ResetMotherbrain => 4,
        _ => 0,
    };
    match (session_id, lease_id) {
        (Some(session_id), Some(lease_id)) => status::active_service_authority_matches(
            session::token_hash(session_id),
            session::token_hash(lease_id),
            Instant::now().as_millis() as u32,
            scope,
        ),
        _ => false,
    }
}
