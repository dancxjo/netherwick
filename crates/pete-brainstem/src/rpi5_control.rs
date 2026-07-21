use std::fmt::Write as _;
use std::fs;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

use pete_cockpit_protocol::{ControlAuthority, EndpointRole, TransportKind};

use crate::capabilities;
use crate::commands::{
    BrainstemCommand, CreateOiMode, FeedbackKind, PowerStateRequest, SafetyLatchKind, SongTone,
    MAX_SONG_TONES,
};
use crate::{session, status};

static STARTED: OnceLock<Instant> = OnceLock::new();
static IDENTITY: OnceLock<Identity> = OnceLock::new();
static AUTHORITY_GENERATION: AtomicU32 = AtomicU32::new(0);

struct Identity {
    device_id: String,
    boot_id: String,
}

pub fn initialize_identity() -> Result<(), Box<dyn std::error::Error>> {
    STARTED.get_or_init(Instant::now);
    let machine_id = token_from_file("/etc/machine-id", "unknown-machine")?;
    let boot_id = token_from_file("/proc/sys/kernel/random/boot_id", "unknown-boot")?;
    let device_suffix = machine_id.chars().take(16).collect::<String>();
    let boot_suffix = boot_id.chars().take(32).collect::<String>();
    let _ = IDENTITY.set(Identity {
        device_id: format!("pete-brainstem-rpi5-{device_suffix}"),
        boot_id: format!("bsboot-{boot_suffix}"),
    });
    Ok(())
}

fn token_from_file(path: &str, fallback: &str) -> Result<String, std::io::Error> {
    let value = match fs::read_to_string(path) {
        Ok(value) => value,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => fallback.to_owned(),
        Err(error) => return Err(error),
    };
    Ok(value
        .trim()
        .chars()
        .filter(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
        .collect())
}

fn now_ms() -> u32 {
    STARTED.get_or_init(Instant::now).elapsed().as_millis() as u32
}

pub fn handle_packet(packet: &[u8]) -> String {
    let Ok(line) = std::str::from_utf8(packet) else {
        return "ERR 0 utf8\n".to_owned();
    };
    handle_line(line.trim())
}

fn handle_line(line: &str) -> String {
    if let Some(body) = line.strip_prefix("HELLO ") {
        return handle_handshake(body);
    }
    if line.starts_with("ACQUIRE_CONTROL_LEASE ") {
        return handle_authority(line);
    }
    if line.starts_with("REGISTER_NETWORK_ENDPOINT ") {
        let seq = line
            .split_ascii_whitespace()
            .nth(1)
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0);
        return format!("ERR {seq} local_brainstem_has_no_network_registration\n");
    }
    if line.starts_with("ACQUIRE_SERVICE_LEASE ") {
        let seq = line
            .split_ascii_whitespace()
            .nth(1)
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0);
        return format!("ERR {seq} service_operation_disabled\n");
    }

    let (command_line, session_id, lease_id, _service_lease_id) = compact_envelope(line);
    let (seq, command) = match parse_command(command_line) {
        Ok(parsed) => parsed,
        Err(seq) => return format!("ERR {seq} parse\n"),
    };

    match command {
        BrainstemCommand::Status | BrainstemCommand::Ping => compact_status(seq),
        BrainstemCommand::GetCapabilities => {
            let mut response = heapless::String::<4096>::new();
            if capabilities::write_compact(&mut response, &capabilities::current(), seq).is_err() {
                format!("ERR {seq} capabilities_too_large\n")
            } else {
                response.as_str().to_owned()
            }
        }
        BrainstemCommand::GetEvents { since_seq } => {
            let mut response = heapless::String::<4096>::new();
            let _ = write!(response, "OK {seq} ");
            if status::write_compact_events(&mut response, since_seq).is_err() {
                format!("ERR {seq} events_too_large\n")
            } else {
                response.as_str().to_owned()
            }
        }
        BrainstemCommand::Bootsel
        | BrainstemCommand::RestartCreate
        | BrainstemCommand::ResetMotherbrain
        | BrainstemCommand::PowerState { .. }
        | BrainstemCommand::OrientationProbe { .. }
        | BrainstemCommand::ZeroImuOrientation { .. }
        | BrainstemCommand::ClearImuOrientation { .. } => {
            format!("ERR {seq} service_operation_disabled\n")
        }
        command => {
            if command_requires_session(command) && !session_id.is_some_and(compact_session_valid) {
                return format!("ERR {seq} invalid_session\n");
            }
            if command_requires_authority(command)
                && !compact_authority_valid(command, session_id, lease_id)
            {
                return format!("ERR {seq} invalid_control_lease\n");
            }
            match status::submit_control_command(seq, command) {
                Ok(()) => format!("OK {seq}\n"),
                Err(reason) => format!("ERR {seq} {}\n", reason.as_str()),
            }
        }
    }
}

fn handle_handshake(body: &str) -> String {
    let hello = match session::parse_json(body) {
        Ok(hello) => hello,
        Err(reason) => return handshake_reject("", reason),
    };
    let Some(identity) = IDENTITY.get() else {
        return handshake_reject(
            hello.handshake_nonce.as_str(),
            session::RejectReason::InternalError,
        );
    };
    let accepted = match session::validate(&hello, &identity.device_id, &identity.boot_id) {
        Ok(accepted) => accepted,
        Err(reason) => return handshake_reject(hello.handshake_nonce.as_str(), reason),
    };
    let session_hash = session::token_hash(accepted.session_id.as_str());
    let peer_hash = session::token_hash(hello.device_id.as_str());
    if hello.role == session::EndpointRole::Motherbrain
        && hello.session_purpose == session::SessionPurpose::Control
    {
        status::request_session_replace(
            accepted.generation,
            session_hash,
            peer_hash,
            session::token_hash(hello.boot_id.as_str()),
        );
        if !wait_for(250, || status::session_replace_acked(accepted.generation)) {
            return handshake_reject(
                hello.handshake_nonce.as_str(),
                session::RejectReason::InternalError,
            );
        }
        status::mark_transport_changed(TransportKind::Udp as u8);
    } else {
        let role = match hello.role {
            session::EndpointRole::Motherbrain => 1,
            session::EndpointRole::Forebrain => 2,
            session::EndpointRole::Operator => 3,
            session::EndpointRole::ServiceTool => 4,
            _ => 0,
        };
        let purpose = match hello.session_purpose {
            session::SessionPurpose::Control => 1,
            session::SessionPurpose::Diagnostic => 2,
        };
        status::register_diagnostic_session(
            session_hash,
            peer_hash,
            session::token_hash(hello.boot_id.as_str()),
            role,
            purpose,
            TransportKind::Udp as u8,
        );
    }

    let caps = capabilities::current();
    let snapshot = status::snapshot(now_ms());
    let (estop_latched, safety_tripped, motion_interlock_latched, _) =
        status::session_safety_snapshot();
    let active_motion = snapshot.body_state == status::BodyState::Moving as u8;
    let mut contract = String::new();
    write_json_array(&mut contract, "verbs", caps.verbs);
    write_json_array(&mut contract, "sensors", caps.sensors);
    write_json_array(&mut contract, "outputs", caps.outputs);
    write_json_array(&mut contract, "safety", caps.safety);
    write_json_array(&mut contract, "events", caps.events);
    let mut software = heapless::String::<512>::new();
    let _ = crate::build_identity::write_json(&mut software, crate::build_identity::CURRENT);
    format!(
        concat!(
            "WELCOME {{\"kind\":\"welcome\",\"role\":\"brainstem\",",
            "\"device_id\":\"{}\",\"boot_id\":\"{}\",",
            "\"echoed_handshake_nonce\":\"{}\",\"session_id\":\"{}\",",
            "\"protocol_major\":1,\"protocol_minor\":{},",
            "\"supported_features\":[\"session_ids\",\"event_cursor\",\"heartbeat\",\"transport_failover\"],",
            "\"required_features\":[\"session_ids\"],",
            "\"heartbeat_min_ms\":250,\"heartbeat_max_ms\":2000,",
            "\"command_ttl_min_ms\":{},\"command_ttl_max_ms\":{},",
            "\"current_event_next_seq\":{},",
            "\"capability_contract\":{{\"body_kind\":\"{}\",\"drive\":\"{}\",{}",
            "\"limits\":{{\"max_linear_mm_s\":{},\"max_angular_mrad_s\":{},\"min_ttl_ms\":{},\"max_ttl_ms\":{}}}}},",
            "\"software\":{{\"software_name\":\"{}\",{}}},",
            "\"safety_snapshot\":{{\"armed\":false,\"estop_latched\":{},\"safety_tripped\":{},",
            "\"motion_interlock_latched\":{},\"active_motion\":{},\"runtime_state\":\"{}\"}}}}\n"
        ),
        identity.device_id,
        identity.boot_id,
        hello.handshake_nonce,
        accepted.session_id,
        accepted.negotiated_minor,
        caps.min_ttl_ms,
        caps.max_ttl_ms,
        snapshot.event_next_seq,
        caps.body_kind,
        caps.drive,
        contract,
        caps.max_linear_mm_s,
        caps.max_angular_mrad_s,
        caps.min_ttl_ms,
        caps.max_ttl_ms,
        caps.firmware_name,
        software,
        estop_latched,
        safety_tripped,
        motion_interlock_latched,
        active_motion,
        if active_motion { "moving" } else { "idle" },
    )
}

fn handshake_reject(nonce: &str, reason: session::RejectReason) -> String {
    status::mark_session_rejected(reason.code());
    format!(
        "REJECT {{\"kind\":\"reject\",\"echoed_handshake_nonce\":\"{nonce}\",\"reason_code\":\"{}\",\"message\":\"handshake rejected\",\"supported_protocol_major\":1,\"supported_minor_min\":0,\"supported_minor_max\":0}}\n",
        reason.as_str()
    )
}

fn write_json_array(response: &mut String, key: &str, values: &[&str]) {
    let _ = write!(response, "\"{key}\":[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            response.push(',');
        }
        let _ = write!(response, "\"{value}\"");
    }
    response.push_str("],");
}

fn handle_authority(line: &str) -> String {
    let (command, session_id, _, _) = compact_envelope(line);
    let mut fields = command.split_ascii_whitespace();
    let valid = fields.next() == Some("ACQUIRE_CONTROL_LEASE");
    let seq = fields
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0);
    let authority = fields.next().unwrap_or("");
    let ttl_ms = fields
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(2_000)
        .clamp(250, 60_000);
    let Some(session_id) = session_id else {
        return format!("ERR {seq} invalid_session\n");
    };
    let session_hash = session::token_hash(session_id);
    if !valid || !authority_policy_allows(session_hash, authority, now_ms()) {
        return format!("ERR {seq} authority_policy_rejected\n");
    }
    let generation = AUTHORITY_GENERATION
        .fetch_add(1, Ordering::Relaxed)
        .wrapping_add(1)
        .max(1);
    let lease_id = format!("lease-{generation:08x}-{session_hash:08x}");
    status::request_authority_transition(
        generation,
        session::token_hash(&lease_id),
        session_hash,
        now_ms().wrapping_add(ttl_ms),
    );
    if !wait_for(250, || status::authority_transition_acked(generation)) {
        return format!("ERR {seq} authority_transition_timeout\n");
    }
    format!(
        "OK {seq} CONTROL_LEASE_GRANTED lease_id={lease_id} session_id={session_id} owner_role={} authority={authority} ttl_ms={ttl_ms} generation={generation}\n",
        role_name(status::session_role(session_hash).unwrap_or(0))
    )
}

fn authority_policy_allows(session_hash: u32, authority: &str, now_ms: u32) -> bool {
    let Some(identity) = status::session_identity(session_hash) else {
        return false;
    };
    let role = match identity.role {
        1 => EndpointRole::Motherbrain,
        2 => EndpointRole::Forebrain,
        3 => EndpointRole::Operator,
        4 => EndpointRole::ServiceTool,
        _ => return false,
    };
    let purpose = if identity.purpose == 1 {
        session::SessionPurpose::Control
    } else {
        session::SessionPurpose::Diagnostic
    };
    let requested = match authority {
        "motherbrain" => ControlAuthority::Motherbrain,
        "operator_debug" => ControlAuthority::OperatorDebug,
        "forebrain_recovery" => ControlAuthority::ForebrainRecovery,
        _ => return false,
    };
    if !pete_cockpit_protocol::role_can_request_control(role, purpose, requested) {
        return false;
    }
    match requested {
        ControlAuthority::Motherbrain => true,
        ControlAuthority::OperatorDebug => cfg!(feature = "operator-debug"),
        ControlAuthority::ForebrainRecovery => {
            status::authority_expired(now_ms)
                && option_env!("PETE_RECOVERY_FOREBRAIN_ID").is_some_and(|device_id| {
                    status::session_peer_matches(session_hash, session::token_hash(device_id))
                })
        }
    }
}

fn wait_for(timeout_ms: u32, condition: impl Fn() -> bool) -> bool {
    for _ in 0..timeout_ms {
        if condition() {
            return true;
        }
        thread::sleep(Duration::from_millis(1));
    }
    condition()
}

fn role_name(role: u8) -> &'static str {
    match role {
        1 => "motherbrain",
        2 => "forebrain",
        3 => "operator",
        4 => "service_tool",
        _ => "unknown",
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
}

fn compact_authority_valid(
    command: BrainstemCommand,
    session_id: Option<&str>,
    lease_id: Option<&str>,
) -> bool {
    match (session_id, lease_id) {
        (Some(session_id), Some(lease_id)) => {
            let session_hash = session::token_hash(session_id);
            let lease_hash = session::token_hash(lease_id);
            if matches!(command, BrainstemCommand::CarefulMode { .. })
                && (status::session_role(session_hash) != Some(3)
                    || !cfg!(feature = "operator-debug"))
            {
                false
            } else if matches!(command, BrainstemCommand::HeartbeatStop { .. }) {
                status::authority_heartbeat_valid(session_hash, lease_hash, now_ms())
            } else {
                status::active_authority_matches(session_hash, lease_hash, now_ms())
            }
        }
        _ => false,
    }
}

fn compact_session_valid(session_id: &str) -> bool {
    status::active_session_matches(session::token_hash(session_id))
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

fn parse_command(line: &str) -> Result<(u32, BrainstemCommand), u32> {
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
        "RESTART_CREATE" => BrainstemCommand::RestartCreate,
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
            let (tones, tone_count) = parse_tones(&mut parts, seq)?;
            BrainstemCommand::DefineChirp {
                kind,
                tones,
                tone_count,
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
            let (tones, tone_count) = parse_tones(&mut parts, seq)?;
            BrainstemCommand::SongDefine {
                id,
                tones,
                tone_count,
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

fn parse_tones<'a>(
    parts: &mut impl Iterator<Item = &'a str>,
    seq: u32,
) -> Result<([SongTone; MAX_SONG_TONES], u8), u32> {
    let mut tones = [SongTone::default(); MAX_SONG_TONES];
    let mut count = 0usize;
    while count < MAX_SONG_TONES {
        let Some(note) = parts.next() else {
            break;
        };
        let duration = parts.next().ok_or(seq)?;
        tones[count] = SongTone {
            note: parse_u32(Some(note)).ok_or(seq)? as u8,
            duration_64ths: parse_u32(Some(duration)).ok_or(seq)? as u8,
        };
        count += 1;
    }
    if count == 0 {
        Err(seq)
    } else {
        Ok((tones, count as u8))
    }
}

fn parse_u32(value: Option<&str>) -> Option<u32> {
    value?.parse().ok()
}

fn parse_i16(value: Option<&str>) -> Option<i16> {
    value?.parse().ok()
}

fn parse_bool(value: Option<&str>) -> Option<bool> {
    match value? {
        "1" | "true" | "TRUE" | "clear" | "CLEAR" | "on" | "ON" | "enable" | "ENABLE" => Some(true),
        "0" | "false" | "FALSE" | "trip" | "TRIP" | "off" | "OFF" | "disable" | "DISABLE" => {
            Some(false)
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

fn parse_oi_mode(mode: &str) -> Option<CreateOiMode> {
    match mode {
        "passive" | "PASSIVE" => Some(CreateOiMode::Passive),
        "safe" | "SAFE" => Some(CreateOiMode::Safe),
        "full" | "FULL" => Some(CreateOiMode::Full),
        _ => None,
    }
}

fn compact_status(seq: u32) -> String {
    let snapshot = status::snapshot(now_ms());
    let (estop_latched, safety_tripped, motion_interlock_latched, safety_latch_kind) =
        status::session_safety_snapshot();
    let flags = snapshot.create_sensor_flags;
    format!(
        concat!(
            "OK {} STATUS uptime_ms={} runtime={} body={} action={} command={} pending={} ",
            "error={} error_uart={} power={} oi={} armed={} estop={} safety_tripped={} ",
            "safety_latch_kind={} safety_hazard_generation={} motion_interlock={} active_cmd_vel={} ",
            "event_next_seq={} uart_health={} uart_error={} create_rx_bytes={} create_rx_packets={} ",
            "create_last_packet_ms={} create_sensor_packet_id={} create_body_packets={} ",
            "create_last_body_packet_ms={} create_last_packet_len={} charging_sources={} create_flags={} ",
            "ir_byte={} bump_left={} bump_right={} wheel_drop={} cliff_left={} cliff_front_left={} ",
            "cliff_front_right={} cliff_right={} create_tx_bytes={} create_last_rx_byte={} ",
            "create_last_tx_byte={} create_last_rx_ms={} create_last_tx_ms={} ",
            "create_rx_errors={}/{}/{}/{}/{} wake_probe={}/{} forebrain_rx_bytes={} ",
            "forebrain_rx_lines={} imu_present={} imu_health={} imu_samples={} imu_age_ms={} ",
            "imu_poll_ms={} imu_yaw_mrad={} imu_pitch_mrad={} imu_roll_mrad={} ",
            "imu_yaw_rate_mrad_s={} imu_gyro_x_mrad_s={} imu_gyro_y_mrad_s={} ",
            "imu_gyro_z_mrad_s={} imu_accel_x_mm_s2={} imu_accel_y_mm_s2={} ",
            "imu_accel_z_mm_s2={} imu_accel_mag_mm_s2={} imu_tilt_mrad={} ",
            "imu_roughness_mm_s2={} imu_impact_mm_s2={} imu_motion_consistency={} ",
            "imu_calibration={} firmware_version={} git_commit={} git_dirty={} build_id={} ",
            "careful_mode={} careful_remaining_ms={} ",
            "audio_silent={} audio_last_requested={} audio_last_played={} audio_last_playback_ms={} ",
            "audio_suppressed={} audio_dropped={}\n"
        ),
        seq,
        snapshot.uptime_ms,
        snapshot.current_runtime_state,
        snapshot.body_state,
        snapshot.current_runtime_action,
        snapshot.current_command,
        snapshot.pending_command,
        snapshot.last_error,
        snapshot.last_error_uart_read_error,
        snapshot.create_power_state,
        snapshot.oi_mode,
        matches!(snapshot.oi_mode, 2 | 3),
        estop_latched,
        safety_tripped,
        compact_safety_latch_kind(safety_latch_kind),
        snapshot.safety_hazard_generation,
        motion_interlock_latched,
        snapshot.body_state == status::BodyState::Moving as u8,
        snapshot.event_next_seq,
        snapshot.uart_rx_health,
        snapshot.last_uart_read_error,
        snapshot.uart_rx_bytes,
        snapshot.uart_rx_packets,
        snapshot.last_uart_packet_timestamp_ms,
        snapshot.create_sensor_last_packet_id,
        snapshot.create_sensor_complete_packet_count,
        snapshot.create_sensor_last_complete_packet_timestamp_ms,
        snapshot.last_uart_packet_len,
        snapshot.create_sensor_charging_sources,
        snapshot.create_sensor_flags,
        snapshot.create_sensor_ir_byte,
        flags & (1 << 0) != 0,
        flags & (1 << 1) != 0,
        flags & (1 << 2) != 0,
        flags & (1 << 4) != 0,
        flags & (1 << 5) != 0,
        flags & (1 << 6) != 0,
        flags & (1 << 7) != 0,
        snapshot.uart_tx_bytes,
        snapshot.last_uart_rx_byte,
        snapshot.last_uart_tx_byte,
        snapshot.last_uart_rx_timestamp_ms,
        snapshot.last_uart_tx_timestamp_ms,
        snapshot.uart_rx_overruns,
        snapshot.uart_rx_breaks,
        snapshot.uart_rx_parity_errors,
        snapshot.uart_rx_framing_errors,
        snapshot.uart_rx_other_errors,
        snapshot.wake_probe_response_bytes,
        snapshot.wake_probe_expected_bytes,
        snapshot.forebrain_uart_rx_bytes,
        snapshot.forebrain_uart_rx_lines,
        snapshot.imu_present,
        snapshot.imu_health,
        snapshot.imu_sample_count,
        snapshot.imu_sample_age_ms,
        snapshot.imu_poll_period_ms,
        snapshot.imu_yaw_mrad,
        snapshot.imu_pitch_mrad,
        snapshot.imu_roll_mrad,
        snapshot.imu_yaw_rate_mrad_s,
        snapshot.imu_gyro_x_mrad_s,
        snapshot.imu_gyro_y_mrad_s,
        snapshot.imu_gyro_z_mrad_s,
        snapshot.imu_accel_x_mm_s2,
        snapshot.imu_accel_y_mm_s2,
        snapshot.imu_accel_z_mm_s2,
        snapshot.imu_accel_magnitude_mm_s2,
        snapshot.imu_tilt_magnitude_mrad,
        snapshot.imu_roughness_mm_s2,
        snapshot.imu_impact_score_mm_s2,
        snapshot.imu_motion_consistency,
        snapshot.imu_calibration_state,
        snapshot.firmware_version,
        snapshot.git_commit,
        snapshot.git_dirty,
        snapshot.build_id,
        status::careful_mode_remaining_ms(snapshot.uptime_ms) > 0,
        status::careful_mode_remaining_ms(snapshot.uptime_ms),
        snapshot.audio_silent,
        crate::audio::cue_name(snapshot.audio_last_requested_cue),
        crate::audio::cue_name(snapshot.audio_last_played_cue),
        snapshot.audio_last_playback_timestamp_ms,
        snapshot.audio_suppressed_by_silent_count,
        snapshot.audio_dropped_or_replaced_count,
    )
}

fn compact_safety_latch_kind(kind: Option<status::SafetyEventKind>) -> &'static str {
    match kind {
        None => "none",
        Some(status::SafetyEventKind::Bump) => "bump",
        Some(status::SafetyEventKind::Cliff) => "cliff",
        Some(status::SafetyEventKind::WheelDrop) => "wheel_drop",
        Some(status::SafetyEventKind::EStop) => "estop",
        Some(status::SafetyEventKind::Heartbeat) => "heartbeat",
        Some(status::SafetyEventKind::Tilt) => "tilt",
        Some(status::SafetyEventKind::Impact) => "impact",
        Some(status::SafetyEventKind::Charging) => "charging",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_envelope_keeps_local_authority_tokens_out_of_command_parser() {
        let (command, session, lease, service) = compact_envelope(
            "CMD_VEL 7 50 0 250 session_id=sess-1 lease_id=lease-1 service_lease_id=service-1",
        );
        assert_eq!(command, "CMD_VEL 7 50 0 250");
        assert_eq!(session, Some("sess-1"));
        assert_eq!(lease, Some("lease-1"));
        assert_eq!(service, Some("service-1"));
    }

    #[test]
    fn compact_parser_matches_create_wire_primitives() {
        assert!(matches!(
            parse_command("CMD_VEL 7 50 -100 250"),
            Ok((
                7,
                BrainstemCommand::CmdVel {
                    linear_mm_s: 50,
                    angular_mrad_s: -100,
                    ttl_ms: 250,
                    seq: 7
                }
            ))
        ));
        assert!(matches!(
            parse_command("FACE_BEARING 8 0 0"),
            Ok((8, BrainstemCommand::Unsupported { seq: 8 }))
        ));
        assert!(matches!(
            parse_command("SET_SILENT 9 true"),
            Ok((
                9,
                BrainstemCommand::SetAudioSilent {
                    silent: true,
                    seq: 9
                }
            ))
        ));
    }

    #[test]
    fn silent_mode_requires_a_session_but_not_control_authority() {
        let command = BrainstemCommand::SetAudioSilent {
            silent: true,
            seq: 9,
        };
        assert!(command_requires_session(command));
        assert!(!command_requires_authority(command));
    }

    #[test]
    fn rpi5_contract_does_not_claim_missing_pico_wiring() {
        let caps = capabilities::current();
        assert!(!caps.verbs.contains(&"power_state"));
        assert!(!caps.verbs.contains(&"orientation_probe"));
        assert!(!caps.outputs.contains(&"power_toggle"));
        assert!(!caps.sensors.contains(&"imu"));
        assert!(!caps.safety.contains(&"tilt"));
    }

    #[test]
    fn diagnostic_welcome_is_valid_compact_json() {
        initialize_identity().unwrap();
        let response = handle_line(concat!(
            "HELLO {",
            "\"role\":\"operator\",\"session_purpose\":\"diagnostic\",",
            "\"device_id\":\"operator-test\",\"boot_id\":\"operator-boot\",",
            "\"handshake_nonce\":\"rpi5-welcome-test\",",
            "\"protocol_major\":1,\"protocol_minor_min\":0,\"protocol_minor_max\":0,",
            "\"supported_features\":[\"session_ids\"],",
            "\"required_features\":[\"session_ids\"],",
            "\"preferred_heartbeat_ms\":500}"
        ));
        let json = response
            .strip_prefix("WELCOME ")
            .expect("diagnostic handshake should be accepted")
            .trim();
        let (_, used) = serde_json_core::from_str::<serde::de::IgnoredAny>(json)
            .expect("WELCOME payload should be valid JSON");
        assert_eq!(used, json.len());
        assert!(json.contains("\"software_name\":\"pete-brainstem-rpi5\""));
        assert!(json.contains("\"active_motion\":false"));
    }
}
