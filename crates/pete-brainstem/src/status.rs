use core::{
    fmt::Write as _,
    sync::atomic::{AtomicU32, AtomicU8, Ordering},
};
pub use pete_cockpit_protocol::CommandRejectReason;

#[cfg(feature = "pico-w")]
use crate::audio::cue_name;
use crate::body;
use crate::commands::{
    BrainstemCommand, CreateOiMode, FeedbackKind, PowerStateRequest, RuntimeCommand,
    SafetyLatchKind, SongTone, MAX_SONG_TONES,
};
use crate::drivers::imu::{
    derive_sample, derive_sample_with_gravity_calibration, ImuGravityCalibration, ImuHealth,
    ImuSample, ImuVector,
};
use crate::events::{BrainstemError, BrainstemEvent, CreateSensorPacket};
use crate::hardware::UartReadError;

// Status domains share one namespace so atomics and wire contracts remain unchanged.
include!("status/state.rs");
include!("status/commands.rs");
include!("status/telemetry.rs");
include!("status/events.rs");
include!("status/snapshot.rs");
include!("status/render.rs");

#[cfg(test)]
#[path = "status_tests.rs"]
mod tests;
