use core::fmt::{self, Write as _};

use heapless::String;

use crate::{body, commands::MAX_SONG_TONES};

pub const VERBS: &[&str] = &[
    "ping",
    "status",
    "get_capabilities",
    "get_events",
    "arm",
    "disarm",
    "stop",
    "estop",
    "clear_estop",
    "clear_motion_queue",
    "cmd_vel",
    "drive_direct",
    "drive_arc",
    "drive_for",
    "turn_by",
    "arc_for",
    "creep_until",
    "scan_arc",
    "face_bearing",
    "track_bearing",
    "hold_heading",
    "turn_to_heading",
    "dock_align",
    "wall_follow",
    "wiggle_align",
    "bump_escape",
    "unstick",
    "cliff_guard",
    "request_sensors",
    "stream_sensors",
    "set_safety_policy",
    "song_define",
    "song_play",
    "define_chirp",
    "play_feedback",
    "power_state",
    "calibrate_turn",
    "reset_odometry",
    "dock",
    "set_lights",
    "set_mode",
];

pub const SENSORS: &[&str] = &[
    "bump",
    "cliff",
    "wheel_drop",
    "wall",
    "virtual_wall",
    "ir",
    "buttons",
    "battery",
    "odometry_delta",
];

pub const OUTPUTS: &[&str] = &["lights", "song", "dock", "power_toggle", "brc"];
pub const SAFETY: &[&str] = &["bump", "cliff", "wheel_drop", "estop", "heartbeat"];
pub const FEEDBACK: &[&str] = &["ok", "error", "armed", "lost_target", "dock_seen", "danger"];
pub const EVENTS: &[&str] = &[
    "boot",
    "command_accepted",
    "command_rejected",
    "command_started",
    "command_completed",
    "command_interrupted",
    "command_timed_out",
    "body_power_requested",
    "body_power_changed",
    "body_mode_requested",
    "body_mode_changed",
    "telemetry_received",
    "sensor_frame_decoded",
    "motion_requested",
    "motion_stopped",
    "safety_tripped",
    "safety_cleared",
    "bump_changed",
    "cliff_changed",
    "wheel_drop_latched",
    "wheel_drop_cleared",
    "heartbeat_expired",
    "estop_latched",
    "estop_cleared",
    "error",
];

pub const SENSOR_PACKET_RANGE: &str = "0,7-31";
pub const SONG_SLOTS: u8 = 16;
pub const MIN_TTL_MS: u32 = 10;
pub const MAX_TTL_MS: u32 = 60_000;
pub const MAX_LINEAR_MM_S: i16 = 500;
pub const MAX_ANGULAR_MRAD_S: i16 = 4_000;

pub struct BrainstemCapabilities {
    pub firmware_name: &'static str,
    pub firmware_version: &'static str,
    pub body_name: &'static str,
    pub body_kind: &'static str,
    pub drive: &'static str,
    pub verbs: &'static [&'static str],
    pub sensors: &'static [&'static str],
    pub outputs: &'static [&'static str],
    pub safety: &'static [&'static str],
    pub feedback: &'static [&'static str],
    pub events: &'static [&'static str],
    pub sensor_packets: &'static str,
    pub max_song_tones: usize,
    pub song_slots: u8,
    pub max_linear_mm_s: i16,
    pub max_angular_mrad_s: i16,
    pub min_ttl_ms: u32,
    pub max_ttl_ms: u32,
}

pub fn current() -> BrainstemCapabilities {
    BrainstemCapabilities {
        firmware_name: env!("CARGO_PKG_NAME"),
        firmware_version: env!("CARGO_PKG_VERSION"),
        body_name: body::BODY_NAME,
        body_kind: body_kind_text(body::BODY_KIND),
        drive: drive_kind_text(body::DRIVE_KIND),
        verbs: VERBS,
        sensors: SENSORS,
        outputs: OUTPUTS,
        safety: SAFETY,
        feedback: FEEDBACK,
        events: EVENTS,
        sensor_packets: SENSOR_PACKET_RANGE,
        max_song_tones: MAX_SONG_TONES,
        song_slots: SONG_SLOTS,
        max_linear_mm_s: MAX_LINEAR_MM_S,
        max_angular_mrad_s: MAX_ANGULAR_MRAD_S,
        min_ttl_ms: MIN_TTL_MS,
        max_ttl_ms: MAX_TTL_MS,
    }
}

pub fn render_json<'a>(
    capabilities: &BrainstemCapabilities,
    command_id: u32,
    buffer: &'a mut [u8],
) -> Option<&'a str> {
    let mut response = String::<3072>::new();
    write!(
        response,
        "{{\"accepted\":true,\"command_id\":{},\"firmware\":\"{}\",\"version\":\"{}\",\"body\":\"{}\",\"body_kind\":\"{}\",\"drive\":\"{}\",",
        command_id,
        capabilities.firmware_name,
        capabilities.firmware_version,
        capabilities.body_name,
        capabilities.body_kind,
        capabilities.drive
    )
    .ok()?;
    write_json_str_array(&mut response, "verbs", capabilities.verbs).ok()?;
    write_json_str_array(&mut response, "sensors", capabilities.sensors).ok()?;
    write_json_str_array(&mut response, "outputs", capabilities.outputs).ok()?;
    write_json_str_array(&mut response, "safety", capabilities.safety).ok()?;
    write_json_str_array(&mut response, "feedback", capabilities.feedback).ok()?;
    write_json_str_array(&mut response, "events", capabilities.events).ok()?;
    write!(
        response,
        "\"limits\":{{\"max_linear_mm_s\":{},\"max_angular_mrad_s\":{},\"min_ttl_ms\":{},\"max_ttl_ms\":{}}},\"sensor_packets\":\"{}\",\"max_song_tones\":{},\"song_slots\":{}}}\n",
        capabilities.max_linear_mm_s,
        capabilities.max_angular_mrad_s,
        capabilities.min_ttl_ms,
        capabilities.max_ttl_ms,
        capabilities.sensor_packets,
        capabilities.max_song_tones,
        capabilities.song_slots
    )
    .ok()?;

    let bytes = response.as_bytes();
    if bytes.len() > buffer.len() {
        return None;
    }
    buffer[..bytes.len()].copy_from_slice(bytes);
    core::str::from_utf8(&buffer[..bytes.len()]).ok()
}

pub fn write_compact<const N: usize>(
    response: &mut String<N>,
    capabilities: &BrainstemCapabilities,
    seq: u32,
) -> fmt::Result {
    write!(
        response,
        "OK {seq} CAPABILITIES body_kind={} drive={} verbs=",
        capabilities.body_kind, capabilities.drive
    )?;
    write_csv(response, capabilities.verbs)?;
    write!(response, " sensors=")?;
    write_csv(response, capabilities.sensors)?;
    write!(response, " outputs=")?;
    write_csv(response, capabilities.outputs)?;
    write!(response, " safety=")?;
    write_csv(response, capabilities.safety)?;
    write!(response, " events=")?;
    write_csv(response, capabilities.events)?;
    write!(
        response,
        " limits=max_linear_mm_s:{},max_angular_mrad_s:{},min_ttl_ms:{},max_ttl_ms:{} max_tones={} song_slots={} feedback_slots={} sensor_packets={}\n",
        capabilities.max_linear_mm_s,
        capabilities.max_angular_mrad_s,
        capabilities.min_ttl_ms,
        capabilities.max_ttl_ms,
        capabilities.max_song_tones,
        capabilities.song_slots,
        capabilities.feedback.len(),
        capabilities.sensor_packets
    )
}

fn write_json_str_array<const N: usize>(
    response: &mut String<N>,
    key: &str,
    values: &[&str],
) -> fmt::Result {
    write!(response, "\"{}\":[", key)?;
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            response.push(',').map_err(|_| fmt::Error)?;
        }
        write!(response, "\"{}\"", value)?;
    }
    response.push_str("],").map_err(|_| fmt::Error)
}

fn write_csv<const N: usize>(response: &mut String<N>, values: &[&str]) -> fmt::Result {
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            response.push(',').map_err(|_| fmt::Error)?;
        }
        response.push_str(value).map_err(|_| fmt::Error)?;
    }
    Ok(())
}

fn body_kind_text(kind: body::BodyKind) -> &'static str {
    match kind {
        body::BodyKind::CreateOpenInterface => "create_oi",
    }
}

fn drive_kind_text(kind: body::DriveKind) -> &'static str {
    match kind {
        body::DriveKind::Differential => "differential",
    }
}
