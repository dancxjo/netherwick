use heapless::Deque;
use pete_cockpit_protocol::{CONTACT_WITHDRAWAL_DURATION_MS, CONTACT_WITHDRAWAL_SPEED_MM_S};

use crate::audio::{
    cue_tones, tone_duration_ms, AudioAnnunciator, AuditoryCue, CueRequestResult,
    AUTOMATIC_CUE_SLOT,
};
use crate::body;
use crate::commands::{
    BrainstemCommand, FeedbackKind, PowerStateRequest, RuntimeCommand, SafetyLatchKind, SongTone,
    ACQUIRE_CREATE_SCRIPT, DISARM_SCRIPT, MAX_SONG_TONES, RESTART_CREATE_SCRIPT,
};
use crate::drivers::{
    create_uart::{CreateUart, CREATE_BUTTON_LED_MASK, CREATE_LED_ADVANCE, CREATE_LED_PLAY},
    leds::Leds,
    timers::Timers,
};
use crate::events::{BrainstemError, BrainstemEvent};
use crate::hardware::BrainstemHardware;
use crate::network_registry;
use crate::status::{self, BodyState, RuntimeActionCode, RuntimeState};

const EVENT_QUEUE_CAPACITY: usize = 16;
const COMMAND_QUEUE_CAPACITY: usize = 16;
pub(crate) const RUNTIME_TICK_MS: u32 = 10;
const SENSOR_PROBE_PERIOD_MS: u32 = 100;
const FULL_MODE_REFRESH_PERIOD_MS: u32 = 1_000;
const HEALTHY_LIGHT_STEP_MS: u32 = 100;
const LOW_BATTERY_PERCENT: u32 = 20;
const LOW_BATTERY_AUDIO_CLEAR_PERCENT: u32 = 25;
const WAKE_PROBE_RESPONSE_BYTES_REQUIRED: u8 = 1;
const CREATE_AXLE_TRACK_MM: i32 = 258;
const FEEDBACK_SLOT_BASE: u8 = 10;
const FEEDBACK_KIND_COUNT: usize = 6;
const MOTHERBRAIN_RESET_PULSE_MS: u32 = 100;
const MOTHERBRAIN_RESET_COOLDOWN_MS: u32 = 30_000;
const MOTHERBRAIN_RESET_HISTORY_CAPACITY: usize = 16;
const MOTHERBRAIN_RESET_SERVICE_SCOPE: u8 = 4;
const CONNECTED_POWER_LED_COLOR: u8 = 128;
const CONNECTED_POWER_LED_INTENSITY: u8 = 255;
const ORIENTATION_PROBE_IMU_MAX_AGE_MS: u32 = body::IMU_POLL_PERIOD_MS * 5;
const ORIENTATION_PROBE_MIN_ACCEL_MM_S2: u16 = 7_000;
const ORIENTATION_PROBE_MAX_ACCEL_MM_S2: u16 = 13_000;
const CONTACT_REPEAT_WINDOW_MS: u32 = 2_000;
const DOCK_DEPARTURE_SPEED_MM_S: i16 = -200;
const DOCK_DEPARTURE_DURATION_MS: u32 = 1_500;
const CREATE_CHARGING_SOURCES_PACKET: u8 = 34;
const CREATE_CHARGING_SOURCES_POLL_PERIOD_MS: u32 = 250;
const CREATE_COMPLETE_SENSOR_PACKET: u8 = 0;
const CREATE_COMPLETE_SENSOR_POLL_PERIOD_MS: u32 = 750;
const CREATE_LINK_FRESHNESS_TIMEOUT_MS: u32 = 1_000;
// Accelerometer-derived tilt is contaminated by short acceleration and dock-ramp
// impacts. Keep impact detection immediate, but require the gravity-vector tilt
// threshold to remain crossed before turning it into a latched motion veto.
const IMU_TILT_LATCH_HOLD_MS: u32 = 100;
const CAREFUL_MODE_MIN_TTL_MS: u32 = 250;
const CAREFUL_MODE_MAX_TTL_MS: u32 = 15_000;

#[derive(Clone, Copy, Eq, PartialEq)]
enum RuntimeMode {
    Running,
    Idle,
    Error,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum SafetyResponse {
    Stop,
    ContactWithdrawal,
}

fn healthy_supervision_lights(phase: u8) -> (u8, u8, u8, u32) {
    let led_bits = if (phase / 8) & 1 == 0 {
        CREATE_LED_PLAY
    } else {
        CREATE_LED_ADVANCE
    };
    (
        led_bits,
        CONNECTED_POWER_LED_COLOR,
        CONNECTED_POWER_LED_INTENSITY,
        HEALTHY_LIGHT_STEP_MS,
    )
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ActiveAction {
    None,
    PowerPulse {
        release_at_ms: u32,
        wake_wait_until_ms: Option<u32>,
        power_on: bool,
    },
    WakeSettle {
        until_ms: u32,
    },
    WaitForCreate {
        deadline_ms: u32,
        next_probe_ms: u32,
        response_bytes: u8,
        oi_started: bool,
        allow_power_toggle_on_timeout: bool,
    },
    Settle {
        until_ms: u32,
    },
    Driving {
        stop_at_ms: u32,
    },
    DockDeparture {
        stop_at_ms: u32,
    },
}

#[derive(Clone, Copy)]
struct SensorStream {
    packet_id: u8,
    period_ms: u32,
    next_request_ms: u32,
}

#[derive(Clone, Copy)]
struct QueuedCommand {
    command_id: u32,
    command: RuntimeCommand,
    safety_recovery: bool,
}

impl QueuedCommand {
    fn new(command_id: u32, command: RuntimeCommand) -> Self {
        Self {
            command_id,
            command,
            safety_recovery: false,
        }
    }

    fn safety_recovery(command_id: u32, command: RuntimeCommand) -> Self {
        Self {
            command_id,
            command,
            safety_recovery: true,
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct MotherbrainResetIdentity {
    session_hash: u32,
    lease_hash: u32,
    command_id: u32,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum MotherbrainResetOutcome {
    Refused(status::MotherbrainResetRefusal),
    Asserted,
    Completed,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct MotherbrainResetRecord {
    identity: MotherbrainResetIdentity,
    outcome: MotherbrainResetOutcome,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct ActiveMotherbrainReset {
    identity: MotherbrainResetIdentity,
    release_at_ms: u32,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct ActiveContactWithdrawal {
    started_at_ms: u32,
    baseline_odometry_mm: i32,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct ActiveVelocity {
    linear_mm_s: i16,
    angular_mrad_s: i16,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct ActiveEscape {
    kind: SafetyLatchKind,
    hazard_generation: u32,
    linear_mm_s: i16,
    angular_mrad_s: i16,
    ttl_ms: u32,
}

pub struct Runtime<H>
where
    H: BrainstemHardware,
{
    hardware: H,
    events: Deque<BrainstemEvent, EVENT_QUEUE_CAPACITY>,
    commands: Deque<QueuedCommand, COMMAND_QUEUE_CAPACITY>,
    timers: Timers,
    create_uart: CreateUart,
    leds: Leds,
    mode: RuntimeMode,
    active: ActiveAction,
    active_command_id: Option<u32>,
    active_velocity: Option<ActiveVelocity>,
    active_escape: Option<ActiveEscape>,
    stop_sent: bool,
    heartbeat_stop_at_ms: Option<u32>,
    careful_mode_until_ms: Option<u32>,
    sensor_stream: Option<SensorStream>,
    next_charging_sources_poll_ms: u32,
    next_complete_sensor_poll_ms: u32,
    next_imu_poll_ms: u32,
    next_full_mode_refresh_ms: u32,
    next_supervision_light_ms: u32,
    supervision_light_phase: u8,
    safety_latched: bool,
    safety_latch_kind: Option<status::SafetyEventKind>,
    dock_departure_pending: bool,
    charging_interlock_latched: bool,
    chirps: [[SongTone; MAX_SONG_TONES]; FEEDBACK_KIND_COUNT],
    chirp_counts: [u8; FEEDBACK_KIND_COUNT],
    audio: AudioAnnunciator,
    song_durations_ms: [u32; 16],
    error_blink_next_ms: u32,
    error_blink_on: bool,
    error_blink_count: u8,
    idle_blink_next_ms: u32,
    idle_blink_on: bool,
    create_responsive: bool,
    estop_latched: bool,
    last_bump: bool,
    last_cliff: bool,
    last_wheel_drop: bool,
    safety_observation_initialized: bool,
    tilt_observed_since_ms: Option<u32>,
    active_motherbrain_reset: Option<ActiveMotherbrainReset>,
    motherbrain_reset_cooldown_until_ms: u32,
    motherbrain_reset_hardware_enabled: bool,
    motherbrain_reset_history: [Option<MotherbrainResetRecord>; MOTHERBRAIN_RESET_HISTORY_CAPACITY],
    motherbrain_reset_history_next: usize,
    safety_recovery_motion: bool,
    active_contact_withdrawal: Option<ActiveContactWithdrawal>,
    last_contact_withdrawal_at_ms: Option<u32>,
    repeated_contact_count: u8,
    last_observed_uart_rx_packets: u32,
    last_create_packet_at_ms: Option<u32>,
    low_battery_active: bool,
    charging_active: bool,
    imu_recovery_since_ms: Option<u32>,
    motion_inconsistency_cooldown_until_ms: u32,
    docking_active: bool,
    last_dock_ir: u8,
    restart_create_pending: bool,
    create_full_ready: bool,
    ever_create_full_ready: bool,
    imu_fault_active: bool,
    last_motion_inconsistent: bool,
}

// Runtime responsibilities use separate impl blocks without changing the type API.
include!("runtime/lifecycle.rs");
include!("runtime/command_queue.rs");
include!("runtime/execution.rs");
include!("runtime/sensors.rs");
include!("runtime/safety.rs");
include!("runtime/motion.rs");
include!("runtime/state.rs");
include!("runtime/helpers.rs");

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;
