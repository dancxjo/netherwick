fn parse_command(command_id: u32, body: &str) -> Option<BrainstemCommand> {
    match json_str(body, "kind")? {
        "ping" => Some(BrainstemCommand::Ping),
        "bootsel" => Some(BrainstemCommand::Bootsel),
        "arm" => Some(BrainstemCommand::Arm),
        "set_mode" => Some(BrainstemCommand::SetMode(parse_oi_mode(json_str(
            body, "mode",
        )?)?)),
        "disarm" => Some(BrainstemCommand::Disarm),
        "stop" => Some(BrainstemCommand::Stop),
        "estop" => Some(BrainstemCommand::EStop),
        "clear_estop" => Some(BrainstemCommand::ClearEStop),
        "cmd_vel" => Some(BrainstemCommand::CmdVel {
            linear_mm_s: json_i16(body, "linear_mm_s")?,
            angular_mrad_s: json_i16(body, "angular_mrad_s")?,
            ttl_ms: json_u32(body, "ttl_ms").or_else(|| json_u32(body, "duration_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "drive_direct" => Some(BrainstemCommand::DriveDirect {
            left_mm_s: json_i16(body, "left_mm_s")?,
            right_mm_s: json_i16(body, "right_mm_s")?,
            ttl_ms: json_u32(body, "ttl_ms").or_else(|| json_u32(body, "duration_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "drive_arc" => Some(BrainstemCommand::DriveArc {
            velocity_mm_s: json_i16(body, "velocity_mm_s")?,
            radius_mm: json_i16(body, "radius_mm")?,
            ttl_ms: json_u32(body, "ttl_ms").or_else(|| json_u32(body, "duration_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "face_bearing" | "track_bearing" | "turn_by" | "drive_for" | "bump_escape"
        | "hold_heading" | "turn_to_heading" | "arc_for" | "creep_until" | "scan_arc"
        | "dock_align" | "wall_follow" | "wiggle_align" | "unstick" | "cliff_guard"
        | "set_safety_policy" => Some(BrainstemCommand::Unsupported {
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "clear_safety_latch" => Some(BrainstemCommand::ClearSafetyLatch {
            kind: parse_safety_latch_kind(json_str(body, "latch")?)?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "careful_mode" => Some(BrainstemCommand::CarefulMode {
            ttl_ms: json_u32(body, "ttl_ms").or_else(|| json_u32(body, "duration_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "escape_motion" => Some(BrainstemCommand::EscapeMotion {
            kind: parse_safety_latch_kind(json_str(body, "hazard")?)?,
            hazard_generation: json_u32(body, "hazard_generation")?,
            linear_mm_s: json_i16(body, "linear_mm_s")?,
            angular_mrad_s: json_i16(body, "angular_mrad_s")?,
            ttl_ms: json_u32(body, "ttl_ms").or_else(|| json_u32(body, "duration_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "heartbeat_stop" => Some(BrainstemCommand::HeartbeatStop {
            timeout_ms: json_u32(body, "timeout_ms").or_else(|| json_u32(body, "ttl_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "request_sensors" => Some(BrainstemCommand::RequestSensors {
            packet_id: json_u32(body, "packet_id")? as u8,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "stream_sensors" => Some(BrainstemCommand::StreamSensors {
            enabled: json_bool(body, "enabled").unwrap_or(true),
            packet_id: json_u32(body, "packet_id")
                .unwrap_or(body::CREATE_SENSOR_PROBE_PACKET as u32) as u8,
            period_ms: json_u32(body, "period_ms").unwrap_or(250),
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "clear_motion_queue" => Some(BrainstemCommand::ClearMotionQueue {
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "define_chirp" => {
            let (tones, tone_count) = parse_song_tones(json_str(body, "tones")?)?;
            Some(BrainstemCommand::DefineChirp {
                kind: parse_feedback_kind(json_str(body, "feedback")?)?,
                tones,
                tone_count,
                seq: json_u32(body, "seq").unwrap_or(command_id),
            })
        }
        "play_feedback" => Some(BrainstemCommand::PlayFeedback {
            kind: parse_feedback_kind(json_str(body, "feedback")?)?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "set_silent" | "set_audio_silent" => Some(BrainstemCommand::SetAudioSilent {
            silent: json_bool(body, "silent")?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "power_state" => Some(BrainstemCommand::PowerState {
            request: parse_power_request(json_str(body, "request")?)?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "create_power_on" => Some(BrainstemCommand::PowerState {
            request: PowerStateRequest::Wake,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "create_power_off" => Some(BrainstemCommand::PowerState {
            request: PowerStateRequest::Sleep,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "calibrate_turn" => Some(BrainstemCommand::CalibrateTurn {
            angular_mrad_s: json_i16(body, "angular_mrad_s")?,
            duration_ms: json_u32(body, "duration_ms")?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "orientation_probe" => Some(BrainstemCommand::OrientationProbe {
            angular_mrad_s: json_i16(body, "angular_mrad_s").unwrap_or(250),
            duration_ms: json_u32(body, "duration_ms").unwrap_or(400),
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "reset_odometry" => Some(BrainstemCommand::ResetOdometry {
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "zero_imu_orientation" => Some(BrainstemCommand::ZeroImuOrientation {
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "clear_imu_orientation" => Some(BrainstemCommand::ClearImuOrientation {
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "restart_create" => Some(BrainstemCommand::RestartCreate),
        "reset_motherbrain" => Some(BrainstemCommand::ResetMotherbrain),
        "get_capabilities" => Some(BrainstemCommand::GetCapabilities),
        "get_events" => Some(BrainstemCommand::GetEvents {
            since_seq: json_u32(body, "since_seq")?,
        }),
        "status" => Some(BrainstemCommand::Status),
        "song_play" => Some(BrainstemCommand::SongPlay {
            id: json_u32(body, "id")? as u8,
        }),
        "song_define" => {
            let (tones, tone_count) = parse_song_tones(json_str(body, "tones")?)?;
            Some(BrainstemCommand::SongDefine {
                id: json_u32(body, "id")? as u8,
                tones,
                tone_count,
                seq: json_u32(body, "seq").unwrap_or(command_id),
            })
        }
        "dock" => Some(BrainstemCommand::Dock),
        "set_lights" => {
            let led_bits = json_u32(body, "led_bits")?;
            let color = json_u32(body, "color")?;
            let intensity = json_u32(body, "intensity")?;
            if led_bits > 0x0f || color > u8::MAX as u32 || intensity > u8::MAX as u32 {
                return None;
            }
            Some(BrainstemCommand::SetLights {
                led_bits: led_bits as u8,
                color: color as u8,
                intensity: intensity as u8,
            })
        }
        _ => None,
    }
}

fn parse_safety_latch_kind(kind: &str) -> Option<SafetyLatchKind> {
    match kind {
        "bump" | "BUMP" => Some(SafetyLatchKind::Bump),
        "cliff" | "CLIFF" => Some(SafetyLatchKind::Cliff),
        "wheel_drop" | "WHEEL_DROP" => Some(SafetyLatchKind::WheelDrop),
        "heartbeat" | "HEARTBEAT" => Some(SafetyLatchKind::Heartbeat),
        "tilt" | "TILT" => Some(SafetyLatchKind::Tilt),
        "impact" | "IMPACT" => Some(SafetyLatchKind::Impact),
        "charging" | "CHARGING" => Some(SafetyLatchKind::Charging),
        _ => None,
    }
}

fn parse_feedback_kind(kind: &str) -> Option<FeedbackKind> {
    match kind {
        "ok" | "OK" => Some(FeedbackKind::Ok),
        "error" | "ERROR" => Some(FeedbackKind::Error),
        "armed" | "ARMED" => Some(FeedbackKind::Armed),
        "lost_target" | "LOST_TARGET" => Some(FeedbackKind::LostTarget),
        "dock_seen" | "DOCK_SEEN" => Some(FeedbackKind::DockSeen),
        "danger" | "DANGER" => Some(FeedbackKind::Danger),
        _ => None,
    }
}

fn parse_power_request(request: &str) -> Option<PowerStateRequest> {
    match request {
        "wake" | "WAKE" => Some(PowerStateRequest::Wake),
        "sleep" | "SLEEP" => Some(PowerStateRequest::Sleep),
        "start_oi" | "START_OI" => Some(PowerStateRequest::StartOi),
        "debug_baud_19200" | "DEBUG_BAUD_19200" => Some(PowerStateRequest::DebugBaud19200),
        "debug_baud_57600" | "DEBUG_BAUD_57600" => Some(PowerStateRequest::DebugBaud57600),
        "debug_baud_115200" | "DEBUG_BAUD_115200" => Some(PowerStateRequest::DebugBaud115200),
        _ => None,
    }
}

fn parse_song_tones(value: &str) -> Option<([SongTone; MAX_SONG_TONES], u8)> {
    let mut tones = [SongTone::default(); MAX_SONG_TONES];
    let mut tone_count = 0usize;
    for pair in value.split(',') {
        if tone_count >= MAX_SONG_TONES {
            return None;
        }
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let split = pair.find(':')?;
        let note = pair[..split].trim().parse::<u8>().ok()?;
        let duration_64ths = pair[split + 1..].trim().parse::<u8>().ok()?;
        tones[tone_count] = SongTone {
            note,
            duration_64ths,
        };
        tone_count += 1;
    }
    if tone_count == 0 {
        None
    } else {
        Some((tones, tone_count as u8))
    }
}

fn parse_oi_mode(mode: &str) -> Option<CreateOiMode> {
    match mode {
        "passive" | "PASSIVE" => Some(CreateOiMode::Passive),
        "safe" | "SAFE" => Some(CreateOiMode::Safe),
        "full" | "FULL" => Some(CreateOiMode::Full),
        _ => None,
    }
}

fn render_command_response<'a>(
    buffer: &'a mut [u8],
    accepted: bool,
    command_id: u32,
    message: &str,
) -> Option<&'a str> {
    let mut response = heapless::String::<128>::new();
    let _ = write!(
        response,
        "{{\"accepted\":{},\"command_id\":{},\"message\":\"{}\"}}\n",
        if accepted { "true" } else { "false" },
        command_id,
        message
    );
    let bytes = response.as_bytes();
    if bytes.len() > buffer.len() {
        return None;
    }
    buffer[..bytes.len()].copy_from_slice(bytes);
    core::str::from_utf8(&buffer[..bytes.len()]).ok()
}

fn render_capabilities_response(buffer: &mut [u8], command_id: u32) -> Option<&str> {
    capabilities::render_json(&capabilities::current(), command_id, buffer)
}

fn websocket_key(request: &[u8]) -> Option<&str> {
    let request = core::str::from_utf8(request).ok()?;
    for line in request.split("\r\n") {
        if let Some(value) = line.strip_prefix("Sec-WebSocket-Key:") {
            return Some(value.trim());
        }
    }
    None
}

fn websocket_accept_key<'a>(key: &str, buffer: &'a mut [u8]) -> Option<&'a str> {
    const GUID: &[u8] = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    let mut sha1 = Sha1::new();
    sha1.update(key.as_bytes());
    sha1.update(GUID);
    let digest = sha1.finalize();
    let len = base64_encode(&digest, buffer)?;
    core::str::from_utf8(&buffer[..len]).ok()
}

async fn write_websocket_upgrade(
    socket: &mut TcpSocket<'_>,
    accept_key: &str,
) -> Result<(), embassy_net::tcp::Error> {
    let mut header = heapless::String::<192>::new();
    let _ = write!(
        header,
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {}\r\n\r\n",
        accept_key
    );
    socket.write_all(header.as_bytes()).await?;
    flush_tcp_with_timeout(socket).await.map(|_| ())
}

async fn read_websocket_text<'a>(
    socket: &mut TcpSocket<'_>,
    payload: &'a mut [u8],
) -> Result<Option<&'a str>, embassy_net::tcp::Error> {
    let mut header = [0; 2];
    read_exact_tcp(socket, &mut header).await?;

    let opcode = header[0] & 0x0f;
    if opcode == 0x08 {
        return Ok(None);
    }
    if opcode != 0x01 {
        return Ok(Some(""));
    }

    let masked = header[1] & 0x80 != 0;
    let len = (header[1] & 0x7f) as usize;
    if !masked || len > payload.len() || len == 126 || len == 127 {
        return Ok(Some(""));
    }

    let mut mask = [0; 4];
    read_exact_tcp(socket, &mut mask).await?;
    read_exact_tcp(socket, &mut payload[..len]).await?;
    for i in 0..len {
        payload[i] ^= mask[i & 3];
    }

    Ok(core::str::from_utf8(&payload[..len]).ok())
}

async fn write_websocket_text(
    socket: &mut TcpSocket<'_>,
    payload: &[u8],
) -> Result<(), embassy_net::tcp::Error> {
    if payload.len() <= 125 {
        let header = [0x81, payload.len() as u8];
        socket.write_all(&header).await?;
    } else if payload.len() <= u16::MAX as usize {
        let len = payload.len() as u16;
        let header = [0x81, 126, (len >> 8) as u8, len as u8];
        socket.write_all(&header).await?;
    } else {
        return Ok(());
    }
    socket.write_all(payload).await?;
    flush_tcp_with_timeout(socket).await.map(|_| ())
}

async fn read_exact_tcp(
    socket: &mut TcpSocket<'_>,
    mut buffer: &mut [u8],
) -> Result<(), embassy_net::tcp::Error> {
    while !buffer.is_empty() {
        let n = socket.read(buffer).await?;
        if n == 0 {
            return Err(embassy_net::tcp::Error::ConnectionReset);
        }
        let tmp = buffer;
        buffer = &mut tmp[n..];
    }
    Ok(())
}

fn base64_encode(input: &[u8], output: &mut [u8]) -> Option<usize> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let output_len = input.len().div_ceil(3) * 4;
    if output_len > output.len() {
        return None;
    }

    let mut i = 0;
    let mut j = 0;
    while i < input.len() {
        let b0 = input[i];
        let b1 = if i + 1 < input.len() { input[i + 1] } else { 0 };
        let b2 = if i + 2 < input.len() { input[i + 2] } else { 0 };
        output[j] = TABLE[(b0 >> 2) as usize];
        output[j + 1] = TABLE[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize];
        output[j + 2] = if i + 1 < input.len() {
            TABLE[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize]
        } else {
            b'='
        };
        output[j + 3] = if i + 2 < input.len() {
            TABLE[(b2 & 0x3f) as usize]
        } else {
            b'='
        };
        i += 3;
        j += 4;
    }
    Some(output_len)
}

struct Sha1 {
    state: [u32; 5],
    len_bytes: u64,
    buffer: [u8; 64],
    buffer_len: usize,
}

impl Sha1 {
    fn new() -> Self {
        Self {
            state: [
                0x6745_2301,
                0xefcd_ab89,
                0x98ba_dcfe,
                0x1032_5476,
                0xc3d2_e1f0,
            ],
            len_bytes: 0,
            buffer: [0; 64],
            buffer_len: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.len_bytes = self.len_bytes.saturating_add(input.len() as u64);

        if self.buffer_len > 0 {
            let copy_len = (64 - self.buffer_len).min(input.len());
            self.buffer[self.buffer_len..self.buffer_len + copy_len]
                .copy_from_slice(&input[..copy_len]);
            self.buffer_len += copy_len;
            input = &input[copy_len..];
            if self.buffer_len == 64 {
                let block = self.buffer;
                self.process_block(&block);
                self.buffer_len = 0;
            }
        }

        while input.len() >= 64 {
            let mut block = [0; 64];
            block.copy_from_slice(&input[..64]);
            self.process_block(&block);
            input = &input[64..];
        }

        if !input.is_empty() {
            self.buffer[..input.len()].copy_from_slice(input);
            self.buffer_len = input.len();
        }
    }

    fn finalize(mut self) -> [u8; 20] {
        let bit_len = self.len_bytes.saturating_mul(8);
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;

        if self.buffer_len > 56 {
            for byte in &mut self.buffer[self.buffer_len..] {
                *byte = 0;
            }
            let block = self.buffer;
            self.process_block(&block);
            self.buffer_len = 0;
        }

        for byte in &mut self.buffer[self.buffer_len..56] {
            *byte = 0;
        }
        self.buffer[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buffer;
        self.process_block(&block);

        let mut out = [0; 20];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn process_block(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let mut a = self.state[0];
        let mut b = self.state[1];
        let mut c = self.state[2];
        let mut d = self.state[3];
        let mut e = self.state[4];

        for (i, word) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5a82_7999),
                20..=39 => (b ^ c ^ d, 0x6ed9_eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1b_bcdc),
                _ => (b ^ c ^ d, 0xca62_c1d6),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
    }
}

fn json_str<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    let value = json_value(body, key)?.strip_prefix('"')?;
    let end = value.find('"')?;
    Some(&value[..end])
}

fn json_value<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    let bytes = body.as_bytes();
    for (start, _) in body.match_indices(key) {
        let end = start.checked_add(key.len())?;
        if start == 0 || end >= bytes.len() || bytes[start - 1] != b'"' || bytes[end] != b'"' {
            continue;
        }
        let after_key = body[end + 1..].trim_start();
        return Some(after_key.strip_prefix(':')?.trim_start());
    }
    None
}

fn json_u32(body: &str, key: &str) -> Option<u32> {
    json_i32(body, key).and_then(|value| u32::try_from(value).ok())
}

fn json_i16(body: &str, key: &str) -> Option<i16> {
    json_i32(body, key).and_then(|value| i16::try_from(value).ok())
}

fn json_bool(body: &str, key: &str) -> Option<bool> {
    let value = json_value(body, key)?;
    let (parsed, rest) = if let Some(rest) = value.strip_prefix("true") {
        (true, rest)
    } else if let Some(rest) = value.strip_prefix("false") {
        (false, rest)
    } else {
        return None;
    };
    json_scalar_terminated(rest).then_some(parsed)
}

fn json_i32(body: &str, key: &str) -> Option<i32> {
    let value = json_value(body, key)?;
    let end = value
        .find(|c: char| !(c == '-' || c.is_ascii_digit()))
        .unwrap_or(value.len());
    if !json_scalar_terminated(&value[end..]) {
        return None;
    }
    value[..end].parse().ok()
}

fn json_scalar_terminated(rest: &str) -> bool {
    matches!(
        rest.trim_start().as_bytes().first(),
        Some(b',') | Some(b'}')
    )
}
