#![allow(dead_code)]

use crate::capabilities::{BodyDriverCapabilities, BrainstemCapabilities};
use crate::commands::CreateOiMode;

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum BodyKind {
    CreateOpenInterface,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum DriveKind {
    Differential,
}

include!(concat!(env!("OUT_DIR"), "/body_config.rs"));

pub fn current_capabilities() -> BrainstemCapabilities {
    match BODY_KIND {
        BodyKind::CreateOpenInterface => create_oi_capabilities(),
    }
}

fn create_oi_capabilities() -> BrainstemCapabilities {
    BrainstemCapabilities {
        firmware_name: env!("CARGO_PKG_NAME"),
        firmware_version: env!("CARGO_PKG_VERSION"),
        body_name: BODY_NAME,
        body_kind: body_kind_text(BODY_KIND),
        drive: drive_kind_text(DRIVE_KIND),
        verbs: CAPABILITY_VERBS,
        sensors: CAPABILITY_SENSORS,
        outputs: CAPABILITY_OUTPUTS,
        safety: CAPABILITY_SAFETY,
        feedback: CAPABILITY_FEEDBACK,
        events: CAPABILITY_EVENTS,
        sensor_packets: CREATE_SUPPORTED_SENSOR_PACKETS,
        max_song_tones: CAPABILITY_MAX_SONG_TONES,
        song_slots: CAPABILITY_SONG_SLOTS,
        max_linear_mm_s: CAPABILITY_MAX_LINEAR_MM_S,
        max_angular_mrad_s: CAPABILITY_MAX_ANGULAR_MRAD_S,
        min_ttl_ms: CAPABILITY_MIN_TTL_MS,
        max_ttl_ms: CAPABILITY_MAX_TTL_MS,
        driver: BodyDriverCapabilities {
            modes: CREATE_SUPPORTED_MODES,
            sensor_packets: CREATE_SUPPORTED_SENSOR_PACKETS,
            has_brc: CREATE_BRC_ENABLED,
            has_power_toggle: true,
            has_lights: CAPABILITY_OUTPUTS.contains(&"lights"),
            has_songs: CAPABILITY_OUTPUTS.contains(&"song"),
            has_dock: CAPABILITY_OUTPUTS.contains(&"dock"),
        },
    }
}

fn body_kind_text(kind: BodyKind) -> &'static str {
    match kind {
        BodyKind::CreateOpenInterface => "create_oi",
    }
}

fn drive_kind_text(kind: DriveKind) -> &'static str {
    match kind {
        DriveKind::Differential => "differential",
    }
}
