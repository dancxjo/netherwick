use core::{
    fmt::Write as _,
    sync::atomic::{AtomicU32, AtomicU8, Ordering},
};

use crate::body;
use crate::commands::{
    BrainstemCommand, CreateOiMode, EscapeDirection, FeedbackKind, LightPattern, PowerStateRequest,
    RuntimeCommand, SafetyAction, SafetyPolicy, SongTone, MAX_SONG_TONES,
};
use crate::events::{BrainstemError, BrainstemEvent, CreateSensorPacket};
use crate::hardware::UartReadError;

const UNKNOWN: u8 = 0;
const OFF: u8 = 1;
const ON: u8 = 2;
const EVENT_LOG_CAPACITY: usize = 32;

static RUNTIME_STATE: AtomicU8 = AtomicU8::new(RuntimeState::Booting as u8);
static CREATE_POWER_STATE: AtomicU8 = AtomicU8::new(UNKNOWN);
static OI_MODE: AtomicU8 = AtomicU8::new(UNKNOWN);
static UART_RX_HEALTH: AtomicU8 = AtomicU8::new(UNKNOWN);
static CURRENT_COMMAND: AtomicU8 = AtomicU8::new(CommandCode::None as u8);
static LAST_ERROR: AtomicU8 = AtomicU8::new(ErrorCode::None as u8);
static LAST_ERROR_UART_READ_ERROR: AtomicU8 = AtomicU8::new(UartReadErrorCode::None as u8);
static DEMO_STATE: AtomicU8 = AtomicU8::new(DemoState::NotStarted as u8);
static LAST_UART_PACKET_TIMESTAMP_MS: AtomicU32 = AtomicU32::new(0);
static LAST_UART_READ_ERROR: AtomicU8 = AtomicU8::new(UartReadErrorCode::None as u8);
static UART_RX_BYTES: AtomicU32 = AtomicU32::new(0);
static UART_RX_PACKETS: AtomicU32 = AtomicU32::new(0);
static LAST_UART_PACKET_LEN: AtomicU32 = AtomicU32::new(0);
static WAKE_PROBE_RESPONSE_BYTES: AtomicU32 = AtomicU32::new(0);
static WAKE_PROBE_EXPECTED_BYTES: AtomicU32 = AtomicU32::new(0);
static CURRENT_RUNTIME_ACTION: AtomicU8 = AtomicU8::new(RuntimeActionCode::None as u8);
static LAST_ERROR_ACTION: AtomicU8 = AtomicU8::new(RuntimeActionCode::None as u8);
static WIFI_STATE: AtomicU8 = AtomicU8::new(WifiState::Off as u8);
static HTTPS_STATE: AtomicU8 = AtomicU8::new(HttpsState::Unavailable as u8);
static HTTP_REQUESTS: AtomicU32 = AtomicU32::new(0);
static DHCP_GRANTS: AtomicU32 = AtomicU32::new(0);
static LAST_WEB_REQUEST_TIMESTAMP_MS: AtomicU32 = AtomicU32::new(0);
static PENDING_LED_BLINKS: AtomicU8 = AtomicU8::new(0);
static PENDING_COMMAND_KIND: AtomicU8 = AtomicU8::new(ControlCommandCode::None as u8);
static PENDING_COMMAND_ID: AtomicU32 = AtomicU32::new(0);
static PENDING_COMMAND_A: AtomicU32 = AtomicU32::new(0);
static PENDING_COMMAND_B: AtomicU32 = AtomicU32::new(0);
static PENDING_COMMAND_C: AtomicU32 = AtomicU32::new(0);
static PENDING_COMMAND_D: AtomicU32 = AtomicU32::new(0);
static PENDING_COMMAND_DURATION_MS: AtomicU32 = AtomicU32::new(0);
static PENDING_COMMAND_SEQ: AtomicU32 = AtomicU32::new(0);
static PENDING_VELOCITY_KIND: AtomicU8 = AtomicU8::new(ControlCommandCode::None as u8);
static PENDING_VELOCITY_ID: AtomicU32 = AtomicU32::new(0);
static PENDING_VELOCITY_A: AtomicU32 = AtomicU32::new(0);
static PENDING_VELOCITY_B: AtomicU32 = AtomicU32::new(0);
static PENDING_VELOCITY_TTL_MS: AtomicU32 = AtomicU32::new(0);
static PENDING_VELOCITY_SEQ: AtomicU32 = AtomicU32::new(0);
static LAST_ACCEPTED_COMMAND_ID: AtomicU32 = AtomicU32::new(0);
static LAST_REJECTED_COMMAND_ID: AtomicU32 = AtomicU32::new(0);
static LAST_STARTED_COMMAND_ID: AtomicU32 = AtomicU32::new(0);
static LAST_COMPLETED_COMMAND_ID: AtomicU32 = AtomicU32::new(0);
static LAST_INTERRUPTED_COMMAND_ID: AtomicU32 = AtomicU32::new(0);
static LAST_TIMED_OUT_COMMAND_ID: AtomicU32 = AtomicU32::new(0);
static LAST_DISPATCHED_COMMAND_ID: AtomicU32 = AtomicU32::new(0);
static FOREBRAIN_UART_RX_BYTES: AtomicU32 = AtomicU32::new(0);
static FOREBRAIN_UART_RX_LINES: AtomicU32 = AtomicU32::new(0);
static FOREBRAIN_UART_LAST_SEQ: AtomicU32 = AtomicU32::new(0);
static FOREBRAIN_UART_LAST_ERROR: AtomicU8 = AtomicU8::new(ForebrainUartErrorCode::None as u8);
static FOREBRAIN_UART_LAST_RX_MS: AtomicU32 = AtomicU32::new(0);
static FOREBRAIN_UART_LAST_COMMAND_MS: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_LAST_PACKET_ID: AtomicU8 = AtomicU8::new(0);
static CREATE_SENSOR_FLAGS: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_DISTANCE_MM: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_ANGLE_MRAD: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_IR_BYTE: AtomicU8 = AtomicU8::new(0);
static CREATE_SENSOR_BUTTONS: AtomicU8 = AtomicU8::new(0);
static CREATE_SENSOR_CHARGING_STATE: AtomicU8 = AtomicU8::new(0);
static CREATE_SENSOR_VOLTAGE_MV: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_CURRENT_MA: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_TEMPERATURE_C: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_CHARGE_MAH: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_CAPACITY_MAH: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_CLIFF_LEFT_SIGNAL: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_CLIFF_FRONT_LEFT_SIGNAL: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_CLIFF_FRONT_RIGHT_SIGNAL: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_CLIFF_RIGHT_SIGNAL: AtomicU32 = AtomicU32::new(0);
static PENDING_SONG_TONES: [AtomicU32; MAX_SONG_TONES] =
    [const { AtomicU32::new(0) }; MAX_SONG_TONES];
static CREATE_SONG_LAST_DEFINED_ID: AtomicU8 = AtomicU8::new(0);
static CREATE_SONG_LAST_DEFINED_LEN: AtomicU8 = AtomicU8::new(0);
static CREATE_SONG_LAST_PLAYED_ID: AtomicU8 = AtomicU8::new(0);
static ODOMETRY_RESET_COUNT: AtomicU32 = AtomicU32::new(0);
static ODOMETRY_DISTANCE_MM: AtomicU32 = AtomicU32::new(0);
static ODOMETRY_HEADING_MRAD: AtomicU32 = AtomicU32::new(0);
static EVENT_NEXT_SEQ: AtomicU32 = AtomicU32::new(1);
static EVENT_SEQ: [AtomicU32; EVENT_LOG_CAPACITY] =
    [const { AtomicU32::new(0) }; EVENT_LOG_CAPACITY];
static EVENT_KIND: [AtomicU8; EVENT_LOG_CAPACITY] =
    [const { AtomicU8::new(PublicEventKind::None as u8) }; EVENT_LOG_CAPACITY];
static EVENT_A: [AtomicU32; EVENT_LOG_CAPACITY] = [const { AtomicU32::new(0) }; EVENT_LOG_CAPACITY];
static EVENT_B: [AtomicU32; EVENT_LOG_CAPACITY] = [const { AtomicU32::new(0) }; EVENT_LOG_CAPACITY];
static EVENT_C: [AtomicU32; EVENT_LOG_CAPACITY] = [const { AtomicU32::new(0) }; EVENT_LOG_CAPACITY];

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct BrainstemStatus {
    pub firmware_name: &'static str,
    pub firmware_version: &'static str,
    pub body_name: &'static str,
    pub body_kind: &'static str,
    pub create_uart_baud: u32,
    pub create_sensor_probe_packet: u8,
    pub uptime_ms: u32,
    pub current_runtime_state: u8,
    pub create_power_state: u8,
    pub oi_mode: u8,
    pub uart_rx_health: u8,
    pub last_uart_packet_timestamp_ms: u32,
    pub last_uart_read_error: u8,
    pub uart_rx_bytes: u32,
    pub uart_rx_packets: u32,
    pub last_uart_packet_len: u32,
    pub wake_probe_response_bytes: u32,
    pub wake_probe_expected_bytes: u32,
    pub current_command: u8,
    pub current_runtime_action: u8,
    pub last_error: u8,
    pub last_error_uart_read_error: u8,
    pub last_error_action: u8,
    pub demo_state: u8,
    pub wifi_state: u8,
    pub https_state: u8,
    pub http_requests: u32,
    pub dhcp_grants: u32,
    pub last_web_request_timestamp_ms: u32,
    pub pending_command: u8,
    pub pending_command_id: u32,
    pub last_accepted_command_id: u32,
    pub last_rejected_command_id: u32,
    pub last_started_command_id: u32,
    pub last_completed_command_id: u32,
    pub last_interrupted_command_id: u32,
    pub last_timed_out_command_id: u32,
    pub forebrain_uart_rx_bytes: u32,
    pub forebrain_uart_rx_lines: u32,
    pub forebrain_uart_last_seq: u32,
    pub forebrain_uart_last_error: u8,
    pub forebrain_uart_link_alive_ms: u32,
    pub forebrain_uart_last_command_age_ms: u32,
    pub create_sensor_last_packet_id: u8,
    pub create_sensor_flags: u32,
    pub create_sensor_distance_mm: i16,
    pub create_sensor_angle_mrad: i16,
    pub create_sensor_ir_byte: u8,
    pub create_sensor_buttons: u8,
    pub create_sensor_charging_state: u8,
    pub create_sensor_voltage_mv: u16,
    pub create_sensor_current_ma: i16,
    pub create_sensor_temperature_c: i8,
    pub create_sensor_charge_mah: u16,
    pub create_sensor_capacity_mah: u16,
    pub create_sensor_cliff_left_signal: u16,
    pub create_sensor_cliff_front_left_signal: u16,
    pub create_sensor_cliff_front_right_signal: u16,
    pub create_sensor_cliff_right_signal: u16,
    pub create_song_last_defined_id: u8,
    pub create_song_last_defined_len: u8,
    pub create_song_last_played_id: u8,
    pub odometry_reset_count: u32,
    pub odometry_distance_mm: i32,
    pub odometry_heading_mrad: i32,
    pub event_next_seq: u32,
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum RuntimeState {
    Booting = 1,
    RunningDemo = 2,
    Idle = 3,
    Error = 4,
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum DemoState {
    NotStarted = 1,
    WaitingForCreate = 2,
    OiStarted = 3,
    Moving = 4,
    PowerCycling = 5,
    Idle = 6,
    Error = 7,
}

#[derive(Clone, Copy)]
#[repr(u8)]
enum CommandCode {
    None = 0,
    WakeCreate = 1,
    SleepCreate = 2,
    PulseBrc = 3,
    StartOi = 4,
    SetOiPassive = 5,
    SetOiSafe = 6,
    SetOiFull = 7,
    Drive = 8,
    StopDrive = 9,
    Behavior = 10,
}

#[derive(Clone, Copy)]
#[repr(u8)]
enum ErrorCode {
    None = 0,
    CreateNoResponse = 1,
    UartFraming = 2,
    Timeout = 3,
    InvalidPacket = 4,
}

#[derive(Clone, Copy)]
#[repr(u8)]
enum UartReadErrorCode {
    None = 0,
    Overrun = 1,
    Break = 2,
    Parity = 3,
    Framing = 4,
    Other = 5,
}

#[derive(Clone, Copy)]
#[repr(u8)]
#[allow(dead_code)]
pub enum ForebrainUartErrorCode {
    None = 0,
    LineTooLong = 1,
    Utf8 = 2,
    Parse = 3,
    Busy = 4,
    Uart = 5,
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum RuntimeActionCode {
    None = 0,
    PowerPulse = 1,
    BrcLow = 2,
    BrcSettle = 3,
    WakeSettle = 4,
    WaitForCreate = 5,
    Settle = 6,
    Driving = 7,
}

#[derive(Clone, Copy)]
#[repr(u8)]
#[allow(dead_code)]
enum WifiState {
    Off = 0,
    Starting = 1,
    ApStarted = 2,
    ServicesStarted = 3,
    Error = 4,
}

#[derive(Clone, Copy)]
#[repr(u8)]
enum HttpsState {
    Unavailable = 0,
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum PublicEventKind {
    None = 0,
    Boot = 1,
    CommandAccepted = 2,
    CommandRejected = 3,
    CommandStarted = 4,
    CommandCompleted = 5,
    CommandInterrupted = 6,
    CommandTimedOut = 7,
    BodyPowerRequested = 8,
    BodyPowerChanged = 9,
    BodyModeRequested = 10,
    BodyModeChanged = 11,
    TelemetryReceived = 12,
    SensorFrameDecoded = 13,
    MotionRequested = 14,
    MotionStopped = 15,
    SafetyTripped = 16,
    SafetyCleared = 17,
    BumpChanged = 18,
    CliffChanged = 19,
    WheelDropLatched = 20,
    WheelDropCleared = 21,
    HeartbeatExpired = 22,
    EStopLatched = 23,
    EStopCleared = 24,
    Error = 25,
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum SafetyEventKind {
    Bump = 1,
    Cliff = 2,
    WheelDrop = 3,
    EStop = 4,
    Heartbeat = 5,
}

#[derive(Clone, Copy, Default)]
pub struct PublicEventRecord {
    pub seq: u32,
    pub kind: u8,
    pub a: u32,
    pub b: u32,
    pub c: u32,
}

#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(u8)]
enum ControlCommandCode {
    None = 0,
    Ping = 1,
    Arm = 2,
    Disarm = 3,
    EStop = 4,
    ClearEStop = 5,
    CmdVel = 6,
    Stop = 7,
    Status = 8,
    SongPlay = 9,
    Dock = 10,
    SetLights = 11,
    SetMode = 12,
    FaceBearing = 13,
    TrackBearing = 14,
    TurnBy = 15,
    DriveFor = 16,
    BumpEscape = 17,
    HeartbeatStop = 18,
    HoldHeading = 19,
    TurnToHeading = 20,
    ArcFor = 21,
    CreepUntil = 22,
    ScanArc = 23,
    DockAlign = 24,
    WallFollow = 25,
    WiggleAlign = 26,
    Unstick = 27,
    CliffGuard = 28,
    SongDefine = 29,
    DriveDirect = 30,
    DriveArc = 31,
    RequestSensors = 32,
    StreamSensors = 33,
    SetSafetyPolicy = 34,
    ClearMotionQueue = 35,
    DefineChirp = 36,
    PlayFeedback = 37,
    PowerState = 38,
    CalibrateTurn = 39,
    ResetOdometry = 40,
    GetCapabilities = 41,
    GetEvents = 42,
}

pub fn set_runtime_state(state: RuntimeState) {
    RUNTIME_STATE.store(state as u8, Ordering::Relaxed);
}

pub fn set_demo_state(state: DemoState) {
    DEMO_STATE.store(state as u8, Ordering::Relaxed);
}

pub fn set_command(command: Option<RuntimeCommand>) -> u8 {
    let code = match command {
        None => CommandCode::None,
        Some(RuntimeCommand::WakeCreate) => CommandCode::WakeCreate,
        Some(RuntimeCommand::SleepCreate) => CommandCode::SleepCreate,
        Some(RuntimeCommand::SetMode(CreateOiMode::Passive)) => CommandCode::SetOiPassive,
        Some(RuntimeCommand::SetMode(CreateOiMode::Safe)) => CommandCode::SetOiSafe,
        Some(RuntimeCommand::SetMode(CreateOiMode::Full)) => CommandCode::SetOiFull,
        Some(RuntimeCommand::Stop) => CommandCode::StopDrive,
        Some(RuntimeCommand::EStop) => CommandCode::StopDrive,
        Some(RuntimeCommand::ClearEStop) => CommandCode::None,
        Some(RuntimeCommand::DriveDirect { .. }) => CommandCode::Drive,
        Some(RuntimeCommand::CmdVel { .. }) => CommandCode::Drive,
        Some(RuntimeCommand::DriveArc { .. }) => CommandCode::Drive,
        Some(RuntimeCommand::FaceBearing { .. })
        | Some(RuntimeCommand::TrackBearing { .. })
        | Some(RuntimeCommand::TurnBy { .. })
        | Some(RuntimeCommand::DriveFor { .. })
        | Some(RuntimeCommand::BumpEscape { .. })
        | Some(RuntimeCommand::HoldHeading { .. })
        | Some(RuntimeCommand::TurnToHeading { .. })
        | Some(RuntimeCommand::ArcFor { .. })
        | Some(RuntimeCommand::CreepUntil { .. })
        | Some(RuntimeCommand::ScanArc { .. })
        | Some(RuntimeCommand::DockAlign { .. })
        | Some(RuntimeCommand::WallFollow { .. })
        | Some(RuntimeCommand::WiggleAlign { .. })
        | Some(RuntimeCommand::Unstick { .. })
        | Some(RuntimeCommand::CliffGuard { .. })
        | Some(RuntimeCommand::HeartbeatStop { .. }) => CommandCode::Behavior,
        Some(RuntimeCommand::PulseBrc) => CommandCode::PulseBrc,
        Some(RuntimeCommand::StartOi) => CommandCode::StartOi,
        Some(RuntimeCommand::Drive { .. }) => CommandCode::Drive,
        Some(RuntimeCommand::StopDrive) => CommandCode::StopDrive,
        Some(RuntimeCommand::RequestSensors { .. })
        | Some(RuntimeCommand::StreamSensors { .. })
        | Some(RuntimeCommand::SetSafetyPolicy { .. })
        | Some(RuntimeCommand::ClearMotionQueue)
        | Some(RuntimeCommand::DefineChirp { .. })
        | Some(RuntimeCommand::PlayFeedback { .. })
        | Some(RuntimeCommand::CalibrateTurn { .. })
        | Some(RuntimeCommand::ResetOdometry)
        | Some(RuntimeCommand::SongDefine { .. })
        | Some(RuntimeCommand::SongPlay { .. })
        | Some(RuntimeCommand::Dock)
        | Some(RuntimeCommand::SetLights { .. }) => CommandCode::None,
    };
    CURRENT_COMMAND.store(code as u8, Ordering::Relaxed);
    code as u8
}

pub fn last_dispatched_command_id() -> u32 {
    LAST_DISPATCHED_COMMAND_ID.load(Ordering::Relaxed)
}

pub fn mark_command_started(command_id: u32, command_code: u8) {
    LAST_STARTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
    record_public_event(
        PublicEventKind::CommandStarted,
        command_id,
        command_code as u32,
        0,
    );
}

#[cfg(feature = "pico-w")]
pub fn submit_control_command(command_id: u32, command: BrainstemCommand) -> bool {
    if matches!(
        command,
        BrainstemCommand::Status | BrainstemCommand::Ping | BrainstemCommand::GetEvents { .. }
    ) {
        LAST_ACCEPTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
        record_public_event(PublicEventKind::CommandAccepted, command_id, 0, 0);
        return true;
    }
    if matches!(command, BrainstemCommand::GetCapabilities) {
        LAST_ACCEPTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
        record_public_event(PublicEventKind::CommandAccepted, command_id, 0, 0);
        return true;
    }

    let Some((kind, a, b, c, d, duration_ms)) = encode_control_command(command) else {
        LAST_REJECTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
        record_public_event(PublicEventKind::CommandRejected, command_id, 0, 0);
        return false;
    };

    if kind == ControlCommandCode::CmdVel {
        let seq = command_seq(command);
        if !seq_is_current_or_newer(seq, PENDING_VELOCITY_SEQ.load(Ordering::Relaxed)) {
            LAST_REJECTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
            record_public_event(PublicEventKind::CommandRejected, command_id, seq, 0);
            return false;
        }

        PENDING_VELOCITY_ID.store(command_id, Ordering::Relaxed);
        PENDING_VELOCITY_A.store(a, Ordering::Relaxed);
        PENDING_VELOCITY_B.store(b, Ordering::Relaxed);
        PENDING_VELOCITY_TTL_MS.store(duration_ms.unwrap_or(0), Ordering::Relaxed);
        PENDING_VELOCITY_SEQ.store(seq, Ordering::Relaxed);
        PENDING_VELOCITY_KIND.store(ControlCommandCode::CmdVel as u8, Ordering::Relaxed);
        LAST_ACCEPTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
        record_public_event(PublicEventKind::CommandAccepted, command_id, seq, 0);
        return true;
    }

    if matches!(kind, ControlCommandCode::Stop | ControlCommandCode::EStop) {
        PENDING_VELOCITY_KIND.store(ControlCommandCode::None as u8, Ordering::Relaxed);
    } else if PENDING_COMMAND_KIND.load(Ordering::Relaxed) != ControlCommandCode::None as u8 {
        LAST_REJECTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
        record_public_event(PublicEventKind::CommandRejected, command_id, 0, 0);
        return false;
    }

    PENDING_COMMAND_ID.store(command_id, Ordering::Relaxed);
    PENDING_COMMAND_A.store(a, Ordering::Relaxed);
    PENDING_COMMAND_B.store(b, Ordering::Relaxed);
    PENDING_COMMAND_C.store(c, Ordering::Relaxed);
    PENDING_COMMAND_D.store(d, Ordering::Relaxed);
    PENDING_COMMAND_DURATION_MS.store(duration_ms.unwrap_or(0), Ordering::Relaxed);
    PENDING_COMMAND_SEQ.store(command_seq(command), Ordering::Relaxed);
    PENDING_COMMAND_KIND.store(kind as u8, Ordering::Relaxed);
    LAST_ACCEPTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
    record_public_event(
        PublicEventKind::CommandAccepted,
        command_id,
        command_seq(command),
        kind as u32,
    );
    true
}

pub fn take_control_command() -> Option<BrainstemCommand> {
    let kind = PENDING_COMMAND_KIND.load(Ordering::Relaxed);
    if kind != ControlCommandCode::None as u8 {
        let a = PENDING_COMMAND_A.load(Ordering::Relaxed);
        let b = PENDING_COMMAND_B.load(Ordering::Relaxed);
        let c = PENDING_COMMAND_C.load(Ordering::Relaxed);
        let d = PENDING_COMMAND_D.load(Ordering::Relaxed);
        let duration = match PENDING_COMMAND_DURATION_MS.load(Ordering::Relaxed) {
            0 => None,
            duration_ms => Some(duration_ms),
        };
        let seq = PENDING_COMMAND_SEQ.load(Ordering::Relaxed);
        let command_id = PENDING_COMMAND_ID.load(Ordering::Relaxed);
        PENDING_COMMAND_KIND.store(ControlCommandCode::None as u8, Ordering::Relaxed);
        LAST_DISPATCHED_COMMAND_ID.store(command_id, Ordering::Relaxed);

        return decode_control_command(kind, a, b, c, d, duration, seq);
    }

    let kind = PENDING_VELOCITY_KIND.load(Ordering::Relaxed);
    if kind != ControlCommandCode::CmdVel as u8 {
        return None;
    }

    let a = PENDING_VELOCITY_A.load(Ordering::Relaxed);
    let b = PENDING_VELOCITY_B.load(Ordering::Relaxed);
    let ttl_ms = PENDING_VELOCITY_TTL_MS.load(Ordering::Relaxed);
    let seq = PENDING_VELOCITY_SEQ.load(Ordering::Relaxed);
    let command_id = PENDING_VELOCITY_ID.load(Ordering::Relaxed);
    PENDING_VELOCITY_KIND.store(ControlCommandCode::None as u8, Ordering::Relaxed);
    LAST_DISPATCHED_COMMAND_ID.store(command_id, Ordering::Relaxed);

    Some(BrainstemCommand::CmdVel {
        linear_mm_s: decode_i16(a),
        angular_mrad_s: decode_i16(b),
        ttl_ms,
        seq,
    })
}

#[cfg(feature = "pico-w")]
fn encode_control_command(
    command: BrainstemCommand,
) -> Option<(ControlCommandCode, u32, u32, u32, u32, Option<u32>)> {
    match command {
        BrainstemCommand::Ping => Some((ControlCommandCode::Ping, 0, 0, 0, 0, None)),
        BrainstemCommand::Arm => Some((ControlCommandCode::Arm, 0, 0, 0, 0, None)),
        BrainstemCommand::Disarm => Some((ControlCommandCode::Disarm, 0, 0, 0, 0, None)),
        BrainstemCommand::EStop => Some((ControlCommandCode::EStop, 0, 0, 0, 0, None)),
        BrainstemCommand::ClearEStop => Some((ControlCommandCode::ClearEStop, 0, 0, 0, 0, None)),
        BrainstemCommand::Stop => Some((ControlCommandCode::Stop, 0, 0, 0, 0, None)),
        BrainstemCommand::Status => Some((ControlCommandCode::Status, 0, 0, 0, 0, None)),
        BrainstemCommand::Bootsel => None,
        BrainstemCommand::SetMode(mode) => Some((
            ControlCommandCode::SetMode,
            match mode {
                CreateOiMode::Passive => 1,
                CreateOiMode::Safe => 2,
                CreateOiMode::Full => 3,
            },
            0,
            0,
            0,
            None,
        )),
        BrainstemCommand::CmdVel {
            linear_mm_s,
            angular_mrad_s,
            ttl_ms,
            ..
        } => Some((
            ControlCommandCode::CmdVel,
            encode_i16(linear_mm_s),
            encode_i16(angular_mrad_s),
            0,
            0,
            Some(ttl_ms),
        )),
        BrainstemCommand::DriveDirect {
            left_mm_s,
            right_mm_s,
            ttl_ms,
            ..
        } => Some((
            ControlCommandCode::DriveDirect,
            encode_i16(left_mm_s),
            encode_i16(right_mm_s),
            0,
            0,
            Some(ttl_ms),
        )),
        BrainstemCommand::DriveArc {
            velocity_mm_s,
            radius_mm,
            ttl_ms,
            ..
        } => Some((
            ControlCommandCode::DriveArc,
            encode_i16(velocity_mm_s),
            encode_i16(radius_mm),
            0,
            0,
            Some(ttl_ms),
        )),
        BrainstemCommand::SongPlay { id } => {
            Some((ControlCommandCode::SongPlay, id as u32, 0, 0, 0, None))
        }
        BrainstemCommand::SongDefine {
            id,
            tones,
            tone_count,
            ..
        } => {
            let tone_count = tone_count.min(MAX_SONG_TONES as u8);
            store_pending_song_tones(&tones, tone_count);
            Some((
                ControlCommandCode::SongDefine,
                id as u32,
                tone_count as u32,
                0,
                0,
                None,
            ))
        }
        BrainstemCommand::Dock => Some((ControlCommandCode::Dock, 0, 0, 0, 0, None)),
        BrainstemCommand::SetLights { pattern } => Some((
            ControlCommandCode::SetLights,
            encode_light_pattern(pattern) as u32,
            0,
            0,
            0,
            None,
        )),
        BrainstemCommand::FaceBearing {
            bearing_mrad,
            max_angular_mrad_s,
            tolerance_mrad,
            ttl_ms,
            ..
        } => Some((
            ControlCommandCode::FaceBearing,
            encode_i16(bearing_mrad),
            encode_i16(max_angular_mrad_s),
            encode_i16(tolerance_mrad),
            0,
            Some(ttl_ms),
        )),
        BrainstemCommand::TrackBearing {
            bearing_mrad,
            range_mm,
            max_linear_mm_s,
            max_angular_mrad_s,
            stop_range_mm,
            ttl_ms,
            ..
        } => Some((
            ControlCommandCode::TrackBearing,
            encode_i16(bearing_mrad),
            range_mm as u32,
            encode_i16(max_linear_mm_s),
            pack_i16_u16(max_angular_mrad_s, stop_range_mm),
            Some(ttl_ms),
        )),
        BrainstemCommand::TurnBy {
            angle_mrad,
            angular_mrad_s,
            timeout_ms,
            ..
        } => Some((
            ControlCommandCode::TurnBy,
            encode_i16(angle_mrad),
            encode_i16(angular_mrad_s),
            0,
            0,
            Some(timeout_ms),
        )),
        BrainstemCommand::DriveFor {
            distance_mm,
            velocity_mm_s,
            timeout_ms,
            ..
        } => Some((
            ControlCommandCode::DriveFor,
            encode_i16(distance_mm),
            encode_i16(velocity_mm_s),
            0,
            0,
            Some(timeout_ms),
        )),
        BrainstemCommand::BumpEscape {
            direction,
            backoff_mm_s,
            turn_angular_mrad_s,
            ..
        } => Some((
            ControlCommandCode::BumpEscape,
            encode_escape_direction(direction) as u32,
            encode_i16(backoff_mm_s),
            encode_i16(turn_angular_mrad_s),
            0,
            None,
        )),
        BrainstemCommand::HeartbeatStop { timeout_ms, .. } => Some((
            ControlCommandCode::HeartbeatStop,
            0,
            0,
            0,
            0,
            Some(timeout_ms),
        )),
        BrainstemCommand::HoldHeading {
            heading_error_mrad,
            velocity_mm_s,
            max_angular_mrad_s,
            ttl_ms,
            ..
        } => Some((
            ControlCommandCode::HoldHeading,
            encode_i16(heading_error_mrad),
            encode_i16(velocity_mm_s),
            encode_i16(max_angular_mrad_s),
            0,
            Some(ttl_ms),
        )),
        BrainstemCommand::TurnToHeading {
            heading_error_mrad,
            angular_mrad_s,
            tolerance_mrad,
            timeout_ms,
            ..
        } => Some((
            ControlCommandCode::TurnToHeading,
            encode_i16(heading_error_mrad),
            encode_i16(angular_mrad_s),
            encode_i16(tolerance_mrad),
            0,
            Some(timeout_ms),
        )),
        BrainstemCommand::ArcFor {
            velocity_mm_s,
            radius_mm,
            duration_ms,
            ..
        } => Some((
            ControlCommandCode::ArcFor,
            encode_i16(velocity_mm_s),
            encode_i16(radius_mm),
            0,
            0,
            Some(duration_ms),
        )),
        BrainstemCommand::CreepUntil {
            velocity_mm_s,
            angular_mrad_s,
            timeout_ms,
            ..
        } => Some((
            ControlCommandCode::CreepUntil,
            encode_i16(velocity_mm_s),
            encode_i16(angular_mrad_s),
            0,
            0,
            Some(timeout_ms),
        )),
        BrainstemCommand::ScanArc {
            angle_mrad,
            angular_mrad_s,
            timeout_ms,
            ..
        } => Some((
            ControlCommandCode::ScanArc,
            encode_i16(angle_mrad),
            encode_i16(angular_mrad_s),
            0,
            0,
            Some(timeout_ms),
        )),
        BrainstemCommand::DockAlign {
            bearing_mrad,
            range_mm,
            max_linear_mm_s,
            max_angular_mrad_s,
            stop_range_mm,
            ttl_ms,
            ..
        } => Some((
            ControlCommandCode::DockAlign,
            encode_i16(bearing_mrad),
            range_mm as u32,
            encode_i16(max_linear_mm_s),
            pack_i16_u16(max_angular_mrad_s, stop_range_mm),
            Some(ttl_ms),
        )),
        BrainstemCommand::WallFollow {
            distance_error_mm,
            velocity_mm_s,
            max_angular_mrad_s,
            ttl_ms,
            ..
        } => Some((
            ControlCommandCode::WallFollow,
            encode_i16(distance_error_mm),
            encode_i16(velocity_mm_s),
            encode_i16(max_angular_mrad_s),
            0,
            Some(ttl_ms),
        )),
        BrainstemCommand::WiggleAlign {
            amplitude_mrad,
            angular_mrad_s,
            cycles,
            ..
        } => Some((
            ControlCommandCode::WiggleAlign,
            encode_i16(amplitude_mrad),
            encode_i16(angular_mrad_s),
            cycles as u32,
            0,
            None,
        )),
        BrainstemCommand::Unstick {
            direction,
            backoff_mm_s,
            turn_angular_mrad_s,
            ..
        } => Some((
            ControlCommandCode::Unstick,
            encode_escape_direction(direction) as u32,
            encode_i16(backoff_mm_s),
            encode_i16(turn_angular_mrad_s),
            0,
            None,
        )),
        BrainstemCommand::CliffGuard { clear, .. } => {
            Some((ControlCommandCode::CliffGuard, clear as u32, 0, 0, 0, None))
        }
        BrainstemCommand::RequestSensors { packet_id, .. } => Some((
            ControlCommandCode::RequestSensors,
            packet_id as u32,
            0,
            0,
            0,
            None,
        )),
        BrainstemCommand::StreamSensors {
            enabled,
            packet_id,
            period_ms,
            ..
        } => Some((
            ControlCommandCode::StreamSensors,
            enabled as u32,
            packet_id as u32,
            0,
            0,
            Some(period_ms),
        )),
        BrainstemCommand::SetSafetyPolicy { policy, .. } => Some((
            ControlCommandCode::SetSafetyPolicy,
            encode_safety_action(policy.bump) as u32,
            encode_safety_action(policy.cliff) as u32,
            policy.wheel_drop_latch as u32,
            0,
            None,
        )),
        BrainstemCommand::ClearMotionQueue { .. } => {
            Some((ControlCommandCode::ClearMotionQueue, 0, 0, 0, 0, None))
        }
        BrainstemCommand::DefineChirp {
            kind,
            tones,
            tone_count,
            ..
        } => {
            let tone_count = tone_count.min(MAX_SONG_TONES as u8);
            store_pending_song_tones(&tones, tone_count);
            Some((
                ControlCommandCode::DefineChirp,
                encode_feedback_kind(kind) as u32,
                tone_count as u32,
                0,
                0,
                None,
            ))
        }
        BrainstemCommand::PlayFeedback { kind, .. } => Some((
            ControlCommandCode::PlayFeedback,
            encode_feedback_kind(kind) as u32,
            0,
            0,
            0,
            None,
        )),
        BrainstemCommand::PowerState { request, .. } => Some((
            ControlCommandCode::PowerState,
            encode_power_request(request) as u32,
            0,
            0,
            0,
            None,
        )),
        BrainstemCommand::CalibrateTurn {
            angular_mrad_s,
            duration_ms,
            ..
        } => Some((
            ControlCommandCode::CalibrateTurn,
            encode_i16(angular_mrad_s),
            0,
            0,
            0,
            Some(duration_ms),
        )),
        BrainstemCommand::ResetOdometry { .. } => {
            Some((ControlCommandCode::ResetOdometry, 0, 0, 0, 0, None))
        }
        BrainstemCommand::GetCapabilities => {
            Some((ControlCommandCode::GetCapabilities, 0, 0, 0, 0, None))
        }
        BrainstemCommand::GetEvents { since_seq } => {
            Some((ControlCommandCode::GetEvents, since_seq, 0, 0, 0, None))
        }
    }
}

fn decode_control_command(
    kind: u8,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    duration_ms: Option<u32>,
    seq: u32,
) -> Option<BrainstemCommand> {
    match kind {
        x if x == ControlCommandCode::Ping as u8 => Some(BrainstemCommand::Ping),
        x if x == ControlCommandCode::Arm as u8 => Some(BrainstemCommand::Arm),
        x if x == ControlCommandCode::Disarm as u8 => Some(BrainstemCommand::Disarm),
        x if x == ControlCommandCode::Stop as u8 => Some(BrainstemCommand::Stop),
        x if x == ControlCommandCode::EStop as u8 => Some(BrainstemCommand::EStop),
        x if x == ControlCommandCode::ClearEStop as u8 => Some(BrainstemCommand::ClearEStop),
        x if x == ControlCommandCode::Status as u8 => Some(BrainstemCommand::Status),
        x if x == ControlCommandCode::SetMode as u8 => Some(BrainstemCommand::SetMode(match a {
            1 => CreateOiMode::Passive,
            2 => CreateOiMode::Safe,
            3 => CreateOiMode::Full,
            _ => return None,
        })),
        x if x == ControlCommandCode::CmdVel as u8 => Some(BrainstemCommand::CmdVel {
            linear_mm_s: decode_i16(a),
            angular_mrad_s: decode_i16(b),
            ttl_ms: duration_ms?,
            seq,
        }),
        x if x == ControlCommandCode::DriveDirect as u8 => Some(BrainstemCommand::DriveDirect {
            left_mm_s: decode_i16(a),
            right_mm_s: decode_i16(b),
            ttl_ms: duration_ms?,
            seq,
        }),
        x if x == ControlCommandCode::DriveArc as u8 => Some(BrainstemCommand::DriveArc {
            velocity_mm_s: decode_i16(a),
            radius_mm: decode_i16(b),
            ttl_ms: duration_ms?,
            seq,
        }),
        x if x == ControlCommandCode::SongPlay as u8 => {
            Some(BrainstemCommand::SongPlay { id: a as u8 })
        }
        x if x == ControlCommandCode::SongDefine as u8 => {
            let tone_count = (b as u8).min(MAX_SONG_TONES as u8);
            Some(BrainstemCommand::SongDefine {
                id: a as u8,
                tones: load_pending_song_tones(tone_count),
                tone_count,
                seq,
            })
        }
        x if x == ControlCommandCode::Dock as u8 => Some(BrainstemCommand::Dock),
        x if x == ControlCommandCode::SetLights as u8 => Some(BrainstemCommand::SetLights {
            pattern: decode_light_pattern(a as u8)?,
        }),
        x if x == ControlCommandCode::FaceBearing as u8 => Some(BrainstemCommand::FaceBearing {
            bearing_mrad: decode_i16(a),
            max_angular_mrad_s: decode_i16(b),
            tolerance_mrad: decode_i16(c),
            ttl_ms: duration_ms?,
            seq,
        }),
        x if x == ControlCommandCode::TrackBearing as u8 => {
            let (max_angular_mrad_s, stop_range_mm) = unpack_i16_u16(d);
            Some(BrainstemCommand::TrackBearing {
                bearing_mrad: decode_i16(a),
                range_mm: b as u16,
                max_linear_mm_s: decode_i16(c),
                max_angular_mrad_s,
                stop_range_mm,
                ttl_ms: duration_ms?,
                seq,
            })
        }
        x if x == ControlCommandCode::TurnBy as u8 => Some(BrainstemCommand::TurnBy {
            angle_mrad: decode_i16(a),
            angular_mrad_s: decode_i16(b),
            timeout_ms: duration_ms?,
            seq,
        }),
        x if x == ControlCommandCode::DriveFor as u8 => Some(BrainstemCommand::DriveFor {
            distance_mm: decode_i16(a),
            velocity_mm_s: decode_i16(b),
            timeout_ms: duration_ms?,
            seq,
        }),
        x if x == ControlCommandCode::BumpEscape as u8 => Some(BrainstemCommand::BumpEscape {
            direction: decode_escape_direction(a as u8)?,
            backoff_mm_s: decode_i16(b),
            turn_angular_mrad_s: decode_i16(c),
            seq,
        }),
        x if x == ControlCommandCode::HeartbeatStop as u8 => {
            Some(BrainstemCommand::HeartbeatStop {
                timeout_ms: duration_ms?,
                seq,
            })
        }
        x if x == ControlCommandCode::HoldHeading as u8 => Some(BrainstemCommand::HoldHeading {
            heading_error_mrad: decode_i16(a),
            velocity_mm_s: decode_i16(b),
            max_angular_mrad_s: decode_i16(c),
            ttl_ms: duration_ms?,
            seq,
        }),
        x if x == ControlCommandCode::TurnToHeading as u8 => {
            Some(BrainstemCommand::TurnToHeading {
                heading_error_mrad: decode_i16(a),
                angular_mrad_s: decode_i16(b),
                tolerance_mrad: decode_i16(c),
                timeout_ms: duration_ms?,
                seq,
            })
        }
        x if x == ControlCommandCode::ArcFor as u8 => Some(BrainstemCommand::ArcFor {
            velocity_mm_s: decode_i16(a),
            radius_mm: decode_i16(b),
            duration_ms: duration_ms?,
            seq,
        }),
        x if x == ControlCommandCode::CreepUntil as u8 => Some(BrainstemCommand::CreepUntil {
            velocity_mm_s: decode_i16(a),
            angular_mrad_s: decode_i16(b),
            timeout_ms: duration_ms?,
            seq,
        }),
        x if x == ControlCommandCode::ScanArc as u8 => Some(BrainstemCommand::ScanArc {
            angle_mrad: decode_i16(a),
            angular_mrad_s: decode_i16(b),
            timeout_ms: duration_ms?,
            seq,
        }),
        x if x == ControlCommandCode::DockAlign as u8 => {
            let (max_angular_mrad_s, stop_range_mm) = unpack_i16_u16(d);
            Some(BrainstemCommand::DockAlign {
                bearing_mrad: decode_i16(a),
                range_mm: b as u16,
                max_linear_mm_s: decode_i16(c),
                max_angular_mrad_s,
                stop_range_mm,
                ttl_ms: duration_ms?,
                seq,
            })
        }
        x if x == ControlCommandCode::WallFollow as u8 => Some(BrainstemCommand::WallFollow {
            distance_error_mm: decode_i16(a),
            velocity_mm_s: decode_i16(b),
            max_angular_mrad_s: decode_i16(c),
            ttl_ms: duration_ms?,
            seq,
        }),
        x if x == ControlCommandCode::WiggleAlign as u8 => Some(BrainstemCommand::WiggleAlign {
            amplitude_mrad: decode_i16(a),
            angular_mrad_s: decode_i16(b),
            cycles: c as u8,
            seq,
        }),
        x if x == ControlCommandCode::Unstick as u8 => Some(BrainstemCommand::Unstick {
            direction: decode_escape_direction(a as u8)?,
            backoff_mm_s: decode_i16(b),
            turn_angular_mrad_s: decode_i16(c),
            seq,
        }),
        x if x == ControlCommandCode::CliffGuard as u8 => {
            Some(BrainstemCommand::CliffGuard { clear: a != 0, seq })
        }
        x if x == ControlCommandCode::RequestSensors as u8 => {
            Some(BrainstemCommand::RequestSensors {
                packet_id: a as u8,
                seq,
            })
        }
        x if x == ControlCommandCode::StreamSensors as u8 => {
            Some(BrainstemCommand::StreamSensors {
                enabled: a != 0,
                packet_id: b as u8,
                period_ms: duration_ms?,
                seq,
            })
        }
        x if x == ControlCommandCode::SetSafetyPolicy as u8 => {
            Some(BrainstemCommand::SetSafetyPolicy {
                policy: SafetyPolicy {
                    bump: decode_safety_action(a as u8)?,
                    cliff: decode_safety_action(b as u8)?,
                    wheel_drop_latch: c != 0,
                },
                seq,
            })
        }
        x if x == ControlCommandCode::ClearMotionQueue as u8 => {
            Some(BrainstemCommand::ClearMotionQueue { seq })
        }
        x if x == ControlCommandCode::DefineChirp as u8 => {
            let tone_count = (b as u8).min(MAX_SONG_TONES as u8);
            Some(BrainstemCommand::DefineChirp {
                kind: decode_feedback_kind(a as u8)?,
                tones: load_pending_song_tones(tone_count),
                tone_count,
                seq,
            })
        }
        x if x == ControlCommandCode::PlayFeedback as u8 => Some(BrainstemCommand::PlayFeedback {
            kind: decode_feedback_kind(a as u8)?,
            seq,
        }),
        x if x == ControlCommandCode::PowerState as u8 => Some(BrainstemCommand::PowerState {
            request: decode_power_request(a as u8)?,
            seq,
        }),
        x if x == ControlCommandCode::CalibrateTurn as u8 => {
            Some(BrainstemCommand::CalibrateTurn {
                angular_mrad_s: decode_i16(a),
                duration_ms: duration_ms?,
                seq,
            })
        }
        x if x == ControlCommandCode::ResetOdometry as u8 => {
            Some(BrainstemCommand::ResetOdometry { seq })
        }
        x if x == ControlCommandCode::GetCapabilities as u8 => {
            Some(BrainstemCommand::GetCapabilities)
        }
        x if x == ControlCommandCode::GetEvents as u8 => {
            Some(BrainstemCommand::GetEvents { since_seq: a })
        }
        _ => None,
    }
}

#[cfg(feature = "pico-w")]
fn command_seq(command: BrainstemCommand) -> u32 {
    match command {
        BrainstemCommand::CmdVel { seq, .. }
        | BrainstemCommand::DriveDirect { seq, .. }
        | BrainstemCommand::DriveArc { seq, .. }
        | BrainstemCommand::FaceBearing { seq, .. }
        | BrainstemCommand::TrackBearing { seq, .. }
        | BrainstemCommand::TurnBy { seq, .. }
        | BrainstemCommand::DriveFor { seq, .. }
        | BrainstemCommand::BumpEscape { seq, .. }
        | BrainstemCommand::HoldHeading { seq, .. }
        | BrainstemCommand::TurnToHeading { seq, .. }
        | BrainstemCommand::ArcFor { seq, .. }
        | BrainstemCommand::CreepUntil { seq, .. }
        | BrainstemCommand::ScanArc { seq, .. }
        | BrainstemCommand::DockAlign { seq, .. }
        | BrainstemCommand::WallFollow { seq, .. }
        | BrainstemCommand::WiggleAlign { seq, .. }
        | BrainstemCommand::Unstick { seq, .. }
        | BrainstemCommand::CliffGuard { seq, .. }
        | BrainstemCommand::SongDefine { seq, .. }
        | BrainstemCommand::RequestSensors { seq, .. }
        | BrainstemCommand::StreamSensors { seq, .. }
        | BrainstemCommand::SetSafetyPolicy { seq, .. }
        | BrainstemCommand::ClearMotionQueue { seq, .. }
        | BrainstemCommand::DefineChirp { seq, .. }
        | BrainstemCommand::PlayFeedback { seq, .. }
        | BrainstemCommand::PowerState { seq, .. }
        | BrainstemCommand::CalibrateTurn { seq, .. }
        | BrainstemCommand::ResetOdometry { seq, .. }
        | BrainstemCommand::HeartbeatStop { seq, .. } => seq,
        _ => 0,
    }
}

#[cfg(feature = "pico-w")]
fn seq_is_current_or_newer(seq: u32, latest_seq: u32) -> bool {
    seq == latest_seq || seq.wrapping_sub(latest_seq) < u32::MAX / 2
}

#[cfg(feature = "pico-w")]
fn encode_i16(value: i16) -> u32 {
    value as u16 as u32
}

fn decode_i16(value: u32) -> i16 {
    value as u16 as i16
}

#[cfg(feature = "pico-w")]
fn pack_i16_u16(left: i16, right: u16) -> u32 {
    ((left as u16 as u32) << 16) | right as u32
}

fn unpack_i16_u16(value: u32) -> (i16, u16) {
    ((value >> 16) as u16 as i16, value as u16)
}

#[cfg(feature = "pico-w")]
fn encode_escape_direction(direction: EscapeDirection) -> u8 {
    match direction {
        EscapeDirection::Left => 1,
        EscapeDirection::Right => 2,
        EscapeDirection::Either => 3,
    }
}

fn decode_escape_direction(value: u8) -> Option<EscapeDirection> {
    match value {
        1 => Some(EscapeDirection::Left),
        2 => Some(EscapeDirection::Right),
        3 => Some(EscapeDirection::Either),
        _ => None,
    }
}

#[cfg(feature = "pico-w")]
fn encode_safety_action(action: SafetyAction) -> u8 {
    match action {
        SafetyAction::None => 0,
        SafetyAction::Stop => 1,
        SafetyAction::Backoff => 2,
        SafetyAction::BumpEscape => 3,
    }
}

fn decode_safety_action(value: u8) -> Option<SafetyAction> {
    match value {
        0 => Some(SafetyAction::None),
        1 => Some(SafetyAction::Stop),
        2 => Some(SafetyAction::Backoff),
        3 => Some(SafetyAction::BumpEscape),
        _ => None,
    }
}

#[cfg(feature = "pico-w")]
fn encode_feedback_kind(kind: FeedbackKind) -> u8 {
    match kind {
        FeedbackKind::Ok => 0,
        FeedbackKind::Error => 1,
        FeedbackKind::Armed => 2,
        FeedbackKind::LostTarget => 3,
        FeedbackKind::DockSeen => 4,
        FeedbackKind::Danger => 5,
    }
}

fn decode_feedback_kind(value: u8) -> Option<FeedbackKind> {
    match value {
        0 => Some(FeedbackKind::Ok),
        1 => Some(FeedbackKind::Error),
        2 => Some(FeedbackKind::Armed),
        3 => Some(FeedbackKind::LostTarget),
        4 => Some(FeedbackKind::DockSeen),
        5 => Some(FeedbackKind::Danger),
        _ => None,
    }
}

#[cfg(feature = "pico-w")]
fn encode_power_request(request: PowerStateRequest) -> u8 {
    match request {
        PowerStateRequest::Wake => 1,
        PowerStateRequest::Sleep => 2,
        PowerStateRequest::PulseBrc => 3,
        PowerStateRequest::StartOi => 4,
    }
}

fn decode_power_request(value: u8) -> Option<PowerStateRequest> {
    match value {
        1 => Some(PowerStateRequest::Wake),
        2 => Some(PowerStateRequest::Sleep),
        3 => Some(PowerStateRequest::PulseBrc),
        4 => Some(PowerStateRequest::StartOi),
        _ => None,
    }
}

#[cfg(feature = "pico-w")]
fn store_pending_song_tones(tones: &[SongTone; MAX_SONG_TONES], tone_count: u8) {
    let tone_count = tone_count.min(MAX_SONG_TONES as u8) as usize;
    for i in 0..MAX_SONG_TONES {
        let value = if i < tone_count {
            pack_song_tone(tones[i])
        } else {
            0
        };
        PENDING_SONG_TONES[i].store(value, Ordering::Relaxed);
    }
}

fn load_pending_song_tones(tone_count: u8) -> [SongTone; MAX_SONG_TONES] {
    let mut tones = [SongTone::default(); MAX_SONG_TONES];
    let tone_count = tone_count.min(MAX_SONG_TONES as u8) as usize;
    for i in 0..tone_count {
        tones[i] = unpack_song_tone(PENDING_SONG_TONES[i].load(Ordering::Relaxed));
    }
    tones
}

#[cfg(feature = "pico-w")]
fn pack_song_tone(tone: SongTone) -> u32 {
    ((tone.note as u32) << 8) | tone.duration_64ths as u32
}

fn unpack_song_tone(value: u32) -> SongTone {
    SongTone {
        note: (value >> 8) as u8,
        duration_64ths: value as u8,
    }
}

#[cfg(feature = "pico-w")]
fn encode_light_pattern(pattern: LightPattern) -> u8 {
    match pattern {
        LightPattern::Off => 0,
        LightPattern::Status => 1,
        LightPattern::Clean => 2,
        LightPattern::Dock => 3,
        LightPattern::Spot => 4,
        LightPattern::Max => 5,
    }
}

fn decode_light_pattern(value: u8) -> Option<LightPattern> {
    match value {
        0 => Some(LightPattern::Off),
        1 => Some(LightPattern::Status),
        2 => Some(LightPattern::Clean),
        3 => Some(LightPattern::Dock),
        4 => Some(LightPattern::Spot),
        5 => Some(LightPattern::Max),
        _ => None,
    }
}

pub fn set_runtime_action(action: RuntimeActionCode) {
    CURRENT_RUNTIME_ACTION.store(action as u8, Ordering::Relaxed);
}

pub fn set_create_power_on(on: bool) {
    CREATE_POWER_STATE.store(if on { ON } else { OFF }, Ordering::Relaxed);
}

pub fn set_create_power_unknown() {
    CREATE_POWER_STATE.store(UNKNOWN, Ordering::Relaxed);
}

pub fn set_oi_mode(mode: CreateOiMode) {
    let code = match mode {
        CreateOiMode::Passive => 1,
        CreateOiMode::Safe => 2,
        CreateOiMode::Full => 3,
    };
    OI_MODE.store(code, Ordering::Relaxed);
}

pub fn set_oi_mode_unknown() {
    OI_MODE.store(UNKNOWN, Ordering::Relaxed);
}

pub fn mark_uart_rx_ok(timestamp_ms: u32) {
    UART_RX_HEALTH.store(ON, Ordering::Relaxed);
    LAST_UART_READ_ERROR.store(UartReadErrorCode::None as u8, Ordering::Relaxed);
    LAST_UART_PACKET_TIMESTAMP_MS.store(timestamp_ms, Ordering::Relaxed);
}

pub fn mark_uart_packet(len: usize) {
    increment(&UART_RX_PACKETS);
    increment_by(&UART_RX_BYTES, len as u32);
    LAST_UART_PACKET_LEN.store(len as u32, Ordering::Relaxed);
}

pub fn mark_create_sensor_packet(packet_id: u8, sensors: CreateSensorPacket) {
    CREATE_SENSOR_LAST_PACKET_ID.store(packet_id, Ordering::Relaxed);
    CREATE_SENSOR_FLAGS.store(create_sensor_flags_bits(sensors), Ordering::Relaxed);
    CREATE_SENSOR_DISTANCE_MM.store(encode_signed_i16(sensors.distance_mm), Ordering::Relaxed);
    CREATE_SENSOR_ANGLE_MRAD.store(encode_signed_i16(sensors.angle_mrad), Ordering::Relaxed);
    CREATE_SENSOR_IR_BYTE.store(sensors.ir_byte, Ordering::Relaxed);
    CREATE_SENSOR_BUTTONS.store(sensors.buttons, Ordering::Relaxed);
    CREATE_SENSOR_CHARGING_STATE.store(sensors.charging_state, Ordering::Relaxed);
    CREATE_SENSOR_VOLTAGE_MV.store(sensors.voltage_mv as u32, Ordering::Relaxed);
    CREATE_SENSOR_CURRENT_MA.store(encode_signed_i16(sensors.current_ma), Ordering::Relaxed);
    CREATE_SENSOR_TEMPERATURE_C.store(encode_signed_i8(sensors.temperature_c), Ordering::Relaxed);
    CREATE_SENSOR_CHARGE_MAH.store(sensors.charge_mah as u32, Ordering::Relaxed);
    CREATE_SENSOR_CAPACITY_MAH.store(sensors.capacity_mah as u32, Ordering::Relaxed);
    CREATE_SENSOR_CLIFF_LEFT_SIGNAL.store(sensors.cliff_left_signal as u32, Ordering::Relaxed);
    CREATE_SENSOR_CLIFF_FRONT_LEFT_SIGNAL
        .store(sensors.cliff_front_left_signal as u32, Ordering::Relaxed);
    CREATE_SENSOR_CLIFF_FRONT_RIGHT_SIGNAL
        .store(sensors.cliff_front_right_signal as u32, Ordering::Relaxed);
    CREATE_SENSOR_CLIFF_RIGHT_SIGNAL.store(sensors.cliff_right_signal as u32, Ordering::Relaxed);
    // Create OI packets 0, 19, and 20 contain distance/angle deltas since
    // the last requested packet. Other packets are snapshots and must not be
    // integrated into odometry.
    if create_packet_has_distance_delta(packet_id) {
        add_signed(&ODOMETRY_DISTANCE_MM, sensors.distance_mm as i32);
    }
    if create_packet_has_angle_delta(packet_id) {
        add_signed(&ODOMETRY_HEADING_MRAD, sensors.angle_mrad as i32);
    }
}

pub fn mark_song_defined(id: u8, tone_count: u8) {
    CREATE_SONG_LAST_DEFINED_ID.store(id, Ordering::Relaxed);
    CREATE_SONG_LAST_DEFINED_LEN.store(tone_count.min(MAX_SONG_TONES as u8), Ordering::Relaxed);
}

pub fn mark_song_played(id: u8) {
    CREATE_SONG_LAST_PLAYED_ID.store(id, Ordering::Relaxed);
}

pub fn mark_odometry_reset() {
    increment(&ODOMETRY_RESET_COUNT);
    ODOMETRY_DISTANCE_MM.store(0, Ordering::Relaxed);
    ODOMETRY_HEADING_MRAD.store(0, Ordering::Relaxed);
}

pub fn mark_command_completed(command_id: u32) {
    LAST_COMPLETED_COMMAND_ID.store(command_id, Ordering::Relaxed);
    record_public_event(PublicEventKind::CommandCompleted, command_id, 0, 0);
}

pub fn mark_command_interrupted(command_id: u32) {
    LAST_INTERRUPTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
    record_public_event(PublicEventKind::CommandInterrupted, command_id, 0, 0);
}

pub fn mark_command_timed_out(command_id: u32) {
    LAST_TIMED_OUT_COMMAND_ID.store(command_id, Ordering::Relaxed);
    record_public_event(PublicEventKind::CommandTimedOut, command_id, 0, 0);
}

pub fn mark_safety_tripped(kind: SafetyEventKind) {
    record_public_event(PublicEventKind::SafetyTripped, kind as u32, 0, 0);
}

pub fn mark_safety_cleared(kind: SafetyEventKind) {
    record_public_event(PublicEventKind::SafetyCleared, kind as u32, 0, 0);
}

pub fn mark_bump_changed(active: bool) {
    record_public_event(PublicEventKind::BumpChanged, active as u32, 0, 0);
}

pub fn mark_cliff_changed(active: bool) {
    record_public_event(PublicEventKind::CliffChanged, active as u32, 0, 0);
}

pub fn mark_wheel_drop_latched() {
    record_public_event(PublicEventKind::WheelDropLatched, 1, 0, 0);
}

pub fn mark_wheel_drop_cleared() {
    record_public_event(PublicEventKind::WheelDropCleared, 0, 0, 0);
}

pub fn mark_heartbeat_expired() {
    record_public_event(PublicEventKind::HeartbeatExpired, 0, 0, 0);
}

pub fn mark_estop_latched() {
    record_public_event(PublicEventKind::EStopLatched, 1, 0, 0);
}

pub fn mark_estop_cleared() {
    record_public_event(PublicEventKind::EStopCleared, 0, 0, 0);
}

pub fn mark_uart_rx_error() {
    UART_RX_HEALTH.store(OFF, Ordering::Relaxed);
}

pub fn mark_uart_rx_error_detail(error: UartReadError) {
    UART_RX_HEALTH.store(OFF, Ordering::Relaxed);
    let code = match error {
        UartReadError::Overrun => UartReadErrorCode::Overrun,
        UartReadError::Break => UartReadErrorCode::Break,
        UartReadError::Parity => UartReadErrorCode::Parity,
        UartReadError::Framing => UartReadErrorCode::Framing,
        UartReadError::Other => UartReadErrorCode::Other,
    };
    LAST_UART_READ_ERROR.store(code as u8, Ordering::Relaxed);
}

#[cfg(feature = "pico-w")]
pub fn mark_forebrain_uart_rx_byte(uptime_ms: u32) {
    increment(&FOREBRAIN_UART_RX_BYTES);
    FOREBRAIN_UART_LAST_RX_MS.store(uptime_ms, Ordering::Relaxed);
}

#[cfg(feature = "pico-w")]
pub fn mark_forebrain_uart_command(seq: u32, uptime_ms: u32) {
    increment(&FOREBRAIN_UART_RX_LINES);
    FOREBRAIN_UART_LAST_SEQ.store(seq, Ordering::Relaxed);
    FOREBRAIN_UART_LAST_COMMAND_MS.store(uptime_ms, Ordering::Relaxed);
    FOREBRAIN_UART_LAST_ERROR.store(ForebrainUartErrorCode::None as u8, Ordering::Relaxed);
}

#[cfg(feature = "pico-w")]
pub fn mark_forebrain_uart_error(error: ForebrainUartErrorCode) {
    FOREBRAIN_UART_LAST_ERROR.store(error as u8, Ordering::Relaxed);
}

pub fn set_wake_probe_progress(response_bytes: u32, expected_bytes: u32) {
    WAKE_PROBE_RESPONSE_BYTES.store(response_bytes, Ordering::Relaxed);
    WAKE_PROBE_EXPECTED_BYTES.store(expected_bytes, Ordering::Relaxed);
}

pub fn set_error(error: BrainstemError) {
    let code = match error {
        BrainstemError::CreateNoResponse => ErrorCode::CreateNoResponse,
        BrainstemError::UartFraming => ErrorCode::UartFraming,
        BrainstemError::Timeout => ErrorCode::Timeout,
        BrainstemError::InvalidPacket => ErrorCode::InvalidPacket,
    };
    LAST_ERROR.store(code as u8, Ordering::Relaxed);
    LAST_ERROR_UART_READ_ERROR.store(
        LAST_UART_READ_ERROR.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    LAST_ERROR_ACTION.store(
        CURRENT_RUNTIME_ACTION.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    set_runtime_state(RuntimeState::Error);
    set_demo_state(DemoState::Error);
    request_led_blinks(8);
}

#[cfg(feature = "pico-w")]
pub fn mark_wifi_starting() {
    WIFI_STATE.store(WifiState::Starting as u8, Ordering::Relaxed);
    request_led_blinks(1);
}

#[cfg(feature = "pico-w")]
pub fn mark_wifi_ap_started() {
    WIFI_STATE.store(WifiState::ApStarted as u8, Ordering::Relaxed);
    request_led_blinks(2);
}

#[cfg(feature = "pico-w")]
pub fn mark_wifi_services_started() {
    WIFI_STATE.store(WifiState::ServicesStarted as u8, Ordering::Relaxed);
    request_led_blinks(3);
}

#[cfg(feature = "pico-w")]
#[allow(dead_code)]
pub fn mark_wifi_error() {
    WIFI_STATE.store(WifiState::Error as u8, Ordering::Relaxed);
    request_led_blinks(8);
}

#[cfg(feature = "pico-w")]
pub fn mark_http_request(uptime_ms: u32) {
    increment(&HTTP_REQUESTS);
    LAST_WEB_REQUEST_TIMESTAMP_MS.store(uptime_ms, Ordering::Relaxed);
    request_led_blinks(4);
}

#[cfg(feature = "pico-w")]
pub fn mark_http_response_flushed() {
    request_led_blinks(2);
}

#[cfg(feature = "pico-w")]
pub fn mark_http_response_error() {
    request_led_blinks(8);
}

#[cfg(feature = "pico-w")]
pub fn mark_dhcp_grant() {
    increment(&DHCP_GRANTS);
    request_led_blinks(5);
}

pub fn signal_event(event: &BrainstemEvent) {
    record_public_event_from_brainstem_event(event);
    let blinks = match event {
        BrainstemEvent::Boot => 1,
        BrainstemEvent::CreatePowerOnRequested | BrainstemEvent::CreatePowerOffRequested => 2,
        BrainstemEvent::CreatePowerToggled => 3,
        BrainstemEvent::CreateBrcPulseRequested | BrainstemEvent::CreateBrcPulsed => 4,
        BrainstemEvent::CreateOiStartRequested | BrainstemEvent::CreateOiModeRequested(_) => 5,
        BrainstemEvent::CreatePacketReceived { .. }
        | BrainstemEvent::CreateSensorPacketDecoded { .. } => 6,
        BrainstemEvent::DriveRequested { .. } | BrainstemEvent::DriveStopped => 7,
        BrainstemEvent::Error(_) => 8,
        BrainstemEvent::TickMs(_) => return,
    };
    request_led_blinks(blinks);
}

pub fn event_next_seq() -> u32 {
    EVENT_NEXT_SEQ.load(Ordering::Relaxed)
}

pub fn event_oldest_seq() -> u32 {
    event_next_seq()
        .saturating_sub(EVENT_LOG_CAPACITY as u32)
        .max(1)
}

pub fn event_dropped_before_seq(since_seq: u32) -> u32 {
    let oldest_seq = event_oldest_seq();
    if since_seq.saturating_add(1) < oldest_seq {
        oldest_seq
    } else {
        0
    }
}

pub fn collect_events_since<const N: usize>(
    since_seq: u32,
    out: &mut heapless::Vec<PublicEventRecord, N>,
) {
    let next_seq = EVENT_NEXT_SEQ.load(Ordering::Relaxed);
    let since_seq = since_seq.max(event_oldest_seq().saturating_sub(1));
    for seq in since_seq.saturating_add(1)..next_seq {
        let index = event_index(seq);
        if EVENT_SEQ[index].load(Ordering::Relaxed) != seq {
            continue;
        }
        let _ = out.push(PublicEventRecord {
            seq,
            kind: EVENT_KIND[index].load(Ordering::Relaxed),
            a: EVENT_A[index].load(Ordering::Relaxed),
            b: EVENT_B[index].load(Ordering::Relaxed),
            c: EVENT_C[index].load(Ordering::Relaxed),
        });
    }
}

#[cfg(feature = "pico-w")]
pub fn render_events_json<'a>(since_seq: u32, buffer: &'a mut [u8]) -> Option<&'a str> {
    let mut response = heapless::String::<2048>::new();
    let mut records = heapless::Vec::<PublicEventRecord, EVENT_LOG_CAPACITY>::new();
    collect_events_since(since_seq, &mut records);
    write!(
        response,
        "{{\"type\":\"events\",\"since_seq\":{},\"oldest_seq\":{},\"next_seq\":{},\"dropped_before_seq\":{},\"events\":[",
        since_seq,
        event_oldest_seq(),
        event_next_seq(),
        event_dropped_before_seq(since_seq)
    )
    .ok()?;
    for (index, record) in records.iter().enumerate() {
        if index > 0 {
            response.push(',').ok()?;
        }
        write!(
            response,
            "{{\"seq\":{},\"kind\":\"{}\",\"a\":{},\"b\":{},\"c\":{}}}",
            record.seq,
            public_event_kind_text(record.kind),
            record.a,
            record.b,
            record.c
        )
        .ok()?;
    }
    response.push_str("]}\n").ok()?;
    let bytes = response.as_bytes();
    if bytes.len() > buffer.len() {
        return None;
    }
    buffer[..bytes.len()].copy_from_slice(bytes);
    core::str::from_utf8(&buffer[..bytes.len()]).ok()
}

pub fn write_compact_events<const N: usize>(
    response: &mut heapless::String<N>,
    since_seq: u32,
) -> core::fmt::Result {
    let mut records = heapless::Vec::<PublicEventRecord, EVENT_LOG_CAPACITY>::new();
    collect_events_since(since_seq, &mut records);
    write!(
        response,
        "EVENTS since={} oldest={} next={} dropped_before={} count={}",
        since_seq,
        event_oldest_seq(),
        event_next_seq(),
        event_dropped_before_seq(since_seq),
        records.len()
    )?;
    for record in records {
        write!(
            response,
            " | {}:{}:{},{},{}",
            record.seq,
            public_event_kind_text(record.kind),
            record.a,
            record.b,
            record.c
        )?;
    }
    response.push('\n').map_err(|_| core::fmt::Error)
}

#[cfg(feature = "pico-w")]
pub fn take_led_blinks() -> Option<u8> {
    let blinks = PENDING_LED_BLINKS.load(Ordering::Relaxed);
    PENDING_LED_BLINKS.store(0, Ordering::Relaxed);
    match blinks {
        0 => None,
        blinks => Some(blinks),
    }
}

fn request_led_blinks(blinks: u8) {
    let blinks = blinks.min(9);
    if blinks > PENDING_LED_BLINKS.load(Ordering::Relaxed) {
        PENDING_LED_BLINKS.store(blinks, Ordering::Relaxed);
    }
}

fn increment(counter: &AtomicU32) {
    increment_by(counter, 1);
}

fn increment_by(counter: &AtomicU32, amount: u32) {
    counter.store(
        counter.load(Ordering::Relaxed).saturating_add(amount),
        Ordering::Relaxed,
    );
}

fn add_signed(counter: &AtomicU32, amount: i32) {
    let current = decode_signed_i32(counter.load(Ordering::Relaxed));
    counter.store(
        encode_signed_i32(current.saturating_add(amount)),
        Ordering::Relaxed,
    );
}

fn record_public_event(kind: PublicEventKind, a: u32, b: u32, c: u32) {
    let seq = EVENT_NEXT_SEQ.load(Ordering::Relaxed);
    EVENT_NEXT_SEQ.store(seq.wrapping_add(1).max(1), Ordering::Relaxed);
    let index = event_index(seq);
    EVENT_A[index].store(a, Ordering::Relaxed);
    EVENT_B[index].store(b, Ordering::Relaxed);
    EVENT_C[index].store(c, Ordering::Relaxed);
    EVENT_KIND[index].store(kind as u8, Ordering::Relaxed);
    EVENT_SEQ[index].store(seq, Ordering::Relaxed);
}

fn record_public_event_from_brainstem_event(event: &BrainstemEvent) {
    match event {
        BrainstemEvent::Boot => record_public_event(PublicEventKind::Boot, 0, 0, 0),
        BrainstemEvent::CreatePowerOnRequested => {
            record_public_event(PublicEventKind::BodyPowerRequested, 1, 0, 0)
        }
        BrainstemEvent::CreatePowerOffRequested => {
            record_public_event(PublicEventKind::BodyPowerRequested, 0, 0, 0)
        }
        BrainstemEvent::CreatePowerToggled => {
            record_public_event(PublicEventKind::BodyPowerChanged, 0, 0, 0)
        }
        BrainstemEvent::CreateOiStartRequested => {
            record_public_event(PublicEventKind::BodyModeRequested, 0, 0, 0)
        }
        BrainstemEvent::CreateOiModeRequested(mode) => record_public_event(
            PublicEventKind::BodyModeRequested,
            encode_oi_mode_public(*mode),
            0,
            0,
        ),
        BrainstemEvent::CreatePacketReceived { packet_id, bytes } => record_public_event(
            PublicEventKind::TelemetryReceived,
            *packet_id as u32,
            bytes.len() as u32,
            0,
        ),
        BrainstemEvent::CreateSensorPacketDecoded { packet_id, sensors } => record_public_event(
            PublicEventKind::SensorFrameDecoded,
            *packet_id as u32,
            create_sensor_flags_bits(*sensors),
            pack_i16_pair(sensors.distance_mm, sensors.angle_mrad),
        ),
        BrainstemEvent::DriveRequested {
            left_mm_s,
            right_mm_s,
            duration_ms,
        } => record_public_event(
            PublicEventKind::MotionRequested,
            pack_i16_pair(*left_mm_s, *right_mm_s),
            *duration_ms,
            0,
        ),
        BrainstemEvent::DriveStopped => {
            record_public_event(PublicEventKind::MotionStopped, 0, 0, 0)
        }
        BrainstemEvent::Error(error) => {
            record_public_event(PublicEventKind::Error, encode_error_public(*error), 0, 0)
        }
        BrainstemEvent::CreateBrcPulseRequested
        | BrainstemEvent::CreateBrcPulsed
        | BrainstemEvent::TickMs(_) => {}
    }
}

const fn event_index(seq: u32) -> usize {
    seq as usize % EVENT_LOG_CAPACITY
}

#[cfg(test)]
fn reset_event_log_for_test() {
    EVENT_NEXT_SEQ.store(1, Ordering::Relaxed);
    for index in 0..EVENT_LOG_CAPACITY {
        EVENT_SEQ[index].store(0, Ordering::Relaxed);
        EVENT_KIND[index].store(PublicEventKind::None as u8, Ordering::Relaxed);
        EVENT_A[index].store(0, Ordering::Relaxed);
        EVENT_B[index].store(0, Ordering::Relaxed);
        EVENT_C[index].store(0, Ordering::Relaxed);
    }
}

fn create_sensor_flags_bits(sensors: CreateSensorPacket) -> u32 {
    let flags = sensors.flags;
    (flags.bump_left as u32)
        | ((flags.bump_right as u32) << 1)
        | ((flags.wheel_drop as u32) << 2)
        | ((flags.wall as u32) << 3)
        | ((flags.cliff_left as u32) << 4)
        | ((flags.cliff_front_left as u32) << 5)
        | ((flags.cliff_front_right as u32) << 6)
        | ((flags.cliff_right as u32) << 7)
        | ((flags.virtual_wall as u32) << 8)
        | ((flags.overcurrent as u32) << 9)
}

fn create_packet_has_distance_delta(packet_id: u8) -> bool {
    matches!(packet_id, 0 | 19)
}

fn create_packet_has_angle_delta(packet_id: u8) -> bool {
    matches!(packet_id, 0 | 20)
}

fn encode_signed_i16(value: i16) -> u32 {
    value as u16 as u32
}

fn decode_signed_i16(value: u32) -> i16 {
    value as u16 as i16
}

fn encode_signed_i8(value: i8) -> u32 {
    value as u8 as u32
}

fn decode_signed_i8(value: u32) -> i8 {
    value as u8 as i8
}

fn encode_signed_i32(value: i32) -> u32 {
    value as u32
}

fn decode_signed_i32(value: u32) -> i32 {
    value as i32
}

fn pack_i16_pair(left: i16, right: i16) -> u32 {
    ((left as u16 as u32) << 16) | right as u16 as u32
}

fn encode_oi_mode_public(mode: CreateOiMode) -> u32 {
    match mode {
        CreateOiMode::Passive => 1,
        CreateOiMode::Safe => 2,
        CreateOiMode::Full => 3,
    }
}

fn encode_error_public(error: BrainstemError) -> u32 {
    match error {
        BrainstemError::CreateNoResponse => ErrorCode::CreateNoResponse as u32,
        BrainstemError::UartFraming => ErrorCode::UartFraming as u32,
        BrainstemError::Timeout => ErrorCode::Timeout as u32,
        BrainstemError::InvalidPacket => ErrorCode::InvalidPacket as u32,
    }
}

#[allow(dead_code)]
pub fn snapshot(uptime_ms: u32) -> BrainstemStatus {
    let pending_command = PENDING_COMMAND_KIND.load(Ordering::Relaxed);
    let pending_command_id = if pending_command == ControlCommandCode::None as u8 {
        PENDING_VELOCITY_ID.load(Ordering::Relaxed)
    } else {
        PENDING_COMMAND_ID.load(Ordering::Relaxed)
    };
    let pending_command = if pending_command == ControlCommandCode::None as u8 {
        PENDING_VELOCITY_KIND.load(Ordering::Relaxed)
    } else {
        pending_command
    };

    BrainstemStatus {
        firmware_name: env!("CARGO_PKG_NAME"),
        firmware_version: env!("CARGO_PKG_VERSION"),
        body_name: body::BODY_NAME,
        body_kind: body_kind(),
        create_uart_baud: body::CREATE_UART_BAUD,
        create_sensor_probe_packet: body::CREATE_SENSOR_PROBE_PACKET,
        uptime_ms,
        current_runtime_state: RUNTIME_STATE.load(Ordering::Relaxed),
        create_power_state: CREATE_POWER_STATE.load(Ordering::Relaxed),
        oi_mode: OI_MODE.load(Ordering::Relaxed),
        uart_rx_health: UART_RX_HEALTH.load(Ordering::Relaxed),
        last_uart_packet_timestamp_ms: LAST_UART_PACKET_TIMESTAMP_MS.load(Ordering::Relaxed),
        last_uart_read_error: LAST_UART_READ_ERROR.load(Ordering::Relaxed),
        uart_rx_bytes: UART_RX_BYTES.load(Ordering::Relaxed),
        uart_rx_packets: UART_RX_PACKETS.load(Ordering::Relaxed),
        last_uart_packet_len: LAST_UART_PACKET_LEN.load(Ordering::Relaxed),
        wake_probe_response_bytes: WAKE_PROBE_RESPONSE_BYTES.load(Ordering::Relaxed),
        wake_probe_expected_bytes: WAKE_PROBE_EXPECTED_BYTES.load(Ordering::Relaxed),
        current_command: CURRENT_COMMAND.load(Ordering::Relaxed),
        current_runtime_action: CURRENT_RUNTIME_ACTION.load(Ordering::Relaxed),
        last_error: LAST_ERROR.load(Ordering::Relaxed),
        last_error_uart_read_error: LAST_ERROR_UART_READ_ERROR.load(Ordering::Relaxed),
        last_error_action: LAST_ERROR_ACTION.load(Ordering::Relaxed),
        demo_state: DEMO_STATE.load(Ordering::Relaxed),
        wifi_state: WIFI_STATE.load(Ordering::Relaxed),
        https_state: HTTPS_STATE.load(Ordering::Relaxed),
        http_requests: HTTP_REQUESTS.load(Ordering::Relaxed),
        dhcp_grants: DHCP_GRANTS.load(Ordering::Relaxed),
        last_web_request_timestamp_ms: LAST_WEB_REQUEST_TIMESTAMP_MS.load(Ordering::Relaxed),
        pending_command,
        pending_command_id,
        last_accepted_command_id: LAST_ACCEPTED_COMMAND_ID.load(Ordering::Relaxed),
        last_rejected_command_id: LAST_REJECTED_COMMAND_ID.load(Ordering::Relaxed),
        last_started_command_id: LAST_STARTED_COMMAND_ID.load(Ordering::Relaxed),
        last_completed_command_id: LAST_COMPLETED_COMMAND_ID.load(Ordering::Relaxed),
        last_interrupted_command_id: LAST_INTERRUPTED_COMMAND_ID.load(Ordering::Relaxed),
        last_timed_out_command_id: LAST_TIMED_OUT_COMMAND_ID.load(Ordering::Relaxed),
        forebrain_uart_rx_bytes: FOREBRAIN_UART_RX_BYTES.load(Ordering::Relaxed),
        forebrain_uart_rx_lines: FOREBRAIN_UART_RX_LINES.load(Ordering::Relaxed),
        forebrain_uart_last_seq: FOREBRAIN_UART_LAST_SEQ.load(Ordering::Relaxed),
        forebrain_uart_last_error: FOREBRAIN_UART_LAST_ERROR.load(Ordering::Relaxed),
        forebrain_uart_link_alive_ms: elapsed_since(
            uptime_ms,
            FOREBRAIN_UART_LAST_RX_MS.load(Ordering::Relaxed),
        ),
        forebrain_uart_last_command_age_ms: elapsed_since(
            uptime_ms,
            FOREBRAIN_UART_LAST_COMMAND_MS.load(Ordering::Relaxed),
        ),
        create_sensor_last_packet_id: CREATE_SENSOR_LAST_PACKET_ID.load(Ordering::Relaxed),
        create_sensor_flags: CREATE_SENSOR_FLAGS.load(Ordering::Relaxed),
        create_sensor_distance_mm: decode_signed_i16(
            CREATE_SENSOR_DISTANCE_MM.load(Ordering::Relaxed),
        ),
        create_sensor_angle_mrad: decode_signed_i16(
            CREATE_SENSOR_ANGLE_MRAD.load(Ordering::Relaxed),
        ),
        create_sensor_ir_byte: CREATE_SENSOR_IR_BYTE.load(Ordering::Relaxed),
        create_sensor_buttons: CREATE_SENSOR_BUTTONS.load(Ordering::Relaxed),
        create_sensor_charging_state: CREATE_SENSOR_CHARGING_STATE.load(Ordering::Relaxed),
        create_sensor_voltage_mv: CREATE_SENSOR_VOLTAGE_MV.load(Ordering::Relaxed) as u16,
        create_sensor_current_ma: decode_signed_i16(
            CREATE_SENSOR_CURRENT_MA.load(Ordering::Relaxed),
        ),
        create_sensor_temperature_c: decode_signed_i8(
            CREATE_SENSOR_TEMPERATURE_C.load(Ordering::Relaxed),
        ),
        create_sensor_charge_mah: CREATE_SENSOR_CHARGE_MAH.load(Ordering::Relaxed) as u16,
        create_sensor_capacity_mah: CREATE_SENSOR_CAPACITY_MAH.load(Ordering::Relaxed) as u16,
        create_sensor_cliff_left_signal: CREATE_SENSOR_CLIFF_LEFT_SIGNAL.load(Ordering::Relaxed)
            as u16,
        create_sensor_cliff_front_left_signal: CREATE_SENSOR_CLIFF_FRONT_LEFT_SIGNAL
            .load(Ordering::Relaxed) as u16,
        create_sensor_cliff_front_right_signal: CREATE_SENSOR_CLIFF_FRONT_RIGHT_SIGNAL
            .load(Ordering::Relaxed) as u16,
        create_sensor_cliff_right_signal: CREATE_SENSOR_CLIFF_RIGHT_SIGNAL.load(Ordering::Relaxed)
            as u16,
        create_song_last_defined_id: CREATE_SONG_LAST_DEFINED_ID.load(Ordering::Relaxed),
        create_song_last_defined_len: CREATE_SONG_LAST_DEFINED_LEN.load(Ordering::Relaxed),
        create_song_last_played_id: CREATE_SONG_LAST_PLAYED_ID.load(Ordering::Relaxed),
        odometry_reset_count: ODOMETRY_RESET_COUNT.load(Ordering::Relaxed),
        odometry_distance_mm: decode_signed_i32(ODOMETRY_DISTANCE_MM.load(Ordering::Relaxed)),
        odometry_heading_mrad: decode_signed_i32(ODOMETRY_HEADING_MRAD.load(Ordering::Relaxed)),
        event_next_seq: EVENT_NEXT_SEQ.load(Ordering::Relaxed),
    }
}

fn elapsed_since(now_ms: u32, timestamp_ms: u32) -> u32 {
    if timestamp_ms == 0 {
        0
    } else {
        now_ms.wrapping_sub(timestamp_ms)
    }
}

#[cfg(feature = "pico-w")]
#[derive(serde::Serialize)]
struct StatusJson {
    firmware_name: &'static str,
    firmware_version: &'static str,
    body_name: &'static str,
    body_kind: &'static str,
    create_uart_baud: u32,
    create_sensor_probe_packet: u8,
    uptime_ms: u32,
    current_runtime_state: &'static str,
    create_power_state: &'static str,
    oi_mode: &'static str,
    uart_rx_health: &'static str,
    last_uart_packet_timestamp_ms: u32,
    last_uart_read_error: &'static str,
    uart_rx_bytes: u32,
    uart_rx_packets: u32,
    last_uart_packet_len: u32,
    wake_probe_response_bytes: u32,
    wake_probe_expected_bytes: u32,
    current_command: &'static str,
    current_runtime_action: &'static str,
    last_error: &'static str,
    last_error_uart_read_error: &'static str,
    last_error_action: &'static str,
    last_error_hint: &'static str,
    demo_state: &'static str,
    wifi_state: &'static str,
    https_state: &'static str,
    http_requests: u32,
    dhcp_grants: u32,
    last_web_request_timestamp_ms: u32,
    pending_command: &'static str,
    pending_command_id: u32,
    last_accepted_command_id: u32,
    last_rejected_command_id: u32,
    last_started_command_id: u32,
    last_completed_command_id: u32,
    last_interrupted_command_id: u32,
    last_timed_out_command_id: u32,
    event_next_seq: u32,
    create_songs: CreateSongStatusJson,
    odometry: OdometryStatusJson,
    create_sensors: CreateSensorStatusJson,
    forebrain_uart: ForebrainUartStatusJson,
}

#[cfg(feature = "pico-w")]
#[derive(serde::Serialize)]
struct CreateSongStatusJson {
    last_defined_id: u8,
    last_defined_len: u8,
    last_played_id: u8,
}

#[cfg(feature = "pico-w")]
#[derive(serde::Serialize)]
struct OdometryStatusJson {
    reset_count: u32,
    distance_mm: i32,
    heading_mrad: i32,
}

#[cfg(feature = "pico-w")]
#[derive(serde::Serialize)]
struct CreateSensorStatusJson {
    last_packet_id: u8,
    bump_left: bool,
    bump_right: bool,
    wheel_drop: bool,
    wall: bool,
    cliff_left: bool,
    cliff_front_left: bool,
    cliff_front_right: bool,
    cliff_right: bool,
    virtual_wall: bool,
    overcurrent: bool,
    distance_mm: i16,
    angle_mrad: i16,
    ir_byte: u8,
    buttons: u8,
    charging_state: u8,
    voltage_mv: u16,
    current_ma: i16,
    temperature_c: i8,
    charge_mah: u16,
    capacity_mah: u16,
    cliff_left_signal: u16,
    cliff_front_left_signal: u16,
    cliff_front_right_signal: u16,
    cliff_right_signal: u16,
}

#[cfg(feature = "pico-w")]
#[derive(serde::Serialize)]
struct ForebrainUartStatusJson {
    rx_bytes: u32,
    rx_lines: u32,
    last_seq: u32,
    last_error: &'static str,
    link_alive_ms: u32,
    last_command_age_ms: u32,
}

#[cfg(feature = "pico-w")]
pub fn render_json<'a>(snapshot: BrainstemStatus, buffer: &'a mut [u8]) -> Result<&'a str, ()> {
    let status = StatusJson {
        firmware_name: snapshot.firmware_name,
        firmware_version: snapshot.firmware_version,
        body_name: snapshot.body_name,
        body_kind: snapshot.body_kind,
        create_uart_baud: snapshot.create_uart_baud,
        create_sensor_probe_packet: snapshot.create_sensor_probe_packet,
        uptime_ms: snapshot.uptime_ms,
        current_runtime_state: runtime_state_text(snapshot.current_runtime_state),
        create_power_state: tri_state_text(snapshot.create_power_state),
        oi_mode: oi_mode_text(snapshot.oi_mode),
        uart_rx_health: uart_health_text(snapshot.uart_rx_health),
        last_uart_packet_timestamp_ms: snapshot.last_uart_packet_timestamp_ms,
        last_uart_read_error: uart_read_error_text(snapshot.last_uart_read_error),
        uart_rx_bytes: snapshot.uart_rx_bytes,
        uart_rx_packets: snapshot.uart_rx_packets,
        last_uart_packet_len: snapshot.last_uart_packet_len,
        wake_probe_response_bytes: snapshot.wake_probe_response_bytes,
        wake_probe_expected_bytes: snapshot.wake_probe_expected_bytes,
        current_command: command_text(snapshot.current_command),
        current_runtime_action: runtime_action_text(snapshot.current_runtime_action),
        last_error: error_text(snapshot.last_error),
        last_error_uart_read_error: uart_read_error_text(snapshot.last_error_uart_read_error),
        last_error_action: runtime_action_text(snapshot.last_error_action),
        last_error_hint: error_hint_text(snapshot),
        demo_state: demo_state_text(snapshot.demo_state),
        wifi_state: wifi_state_text(snapshot.wifi_state),
        https_state: https_state_text(snapshot.https_state),
        http_requests: snapshot.http_requests,
        dhcp_grants: snapshot.dhcp_grants,
        last_web_request_timestamp_ms: snapshot.last_web_request_timestamp_ms,
        pending_command: control_command_text(snapshot.pending_command),
        pending_command_id: snapshot.pending_command_id,
        last_accepted_command_id: snapshot.last_accepted_command_id,
        last_rejected_command_id: snapshot.last_rejected_command_id,
        last_started_command_id: snapshot.last_started_command_id,
        last_completed_command_id: snapshot.last_completed_command_id,
        last_interrupted_command_id: snapshot.last_interrupted_command_id,
        last_timed_out_command_id: snapshot.last_timed_out_command_id,
        event_next_seq: snapshot.event_next_seq,
        create_songs: CreateSongStatusJson {
            last_defined_id: snapshot.create_song_last_defined_id,
            last_defined_len: snapshot.create_song_last_defined_len,
            last_played_id: snapshot.create_song_last_played_id,
        },
        odometry: OdometryStatusJson {
            reset_count: snapshot.odometry_reset_count,
            distance_mm: snapshot.odometry_distance_mm,
            heading_mrad: snapshot.odometry_heading_mrad,
        },
        create_sensors: create_sensor_status_json(snapshot),
        forebrain_uart: ForebrainUartStatusJson {
            rx_bytes: snapshot.forebrain_uart_rx_bytes,
            rx_lines: snapshot.forebrain_uart_rx_lines,
            last_seq: snapshot.forebrain_uart_last_seq,
            last_error: forebrain_uart_error_text(snapshot.forebrain_uart_last_error),
            link_alive_ms: snapshot.forebrain_uart_link_alive_ms,
            last_command_age_ms: snapshot.forebrain_uart_last_command_age_ms,
        },
    };
    let len = serde_json_core::to_slice(&status, buffer).map_err(|_| ())?;
    core::str::from_utf8(&buffer[..len]).map_err(|_| ())
}

#[cfg(feature = "pico-w")]
fn create_sensor_status_json(snapshot: BrainstemStatus) -> CreateSensorStatusJson {
    let flags = snapshot.create_sensor_flags;
    CreateSensorStatusJson {
        last_packet_id: snapshot.create_sensor_last_packet_id,
        bump_left: flags & (1 << 0) != 0,
        bump_right: flags & (1 << 1) != 0,
        wheel_drop: flags & (1 << 2) != 0,
        wall: flags & (1 << 3) != 0,
        cliff_left: flags & (1 << 4) != 0,
        cliff_front_left: flags & (1 << 5) != 0,
        cliff_front_right: flags & (1 << 6) != 0,
        cliff_right: flags & (1 << 7) != 0,
        virtual_wall: flags & (1 << 8) != 0,
        overcurrent: flags & (1 << 9) != 0,
        distance_mm: snapshot.create_sensor_distance_mm,
        angle_mrad: snapshot.create_sensor_angle_mrad,
        ir_byte: snapshot.create_sensor_ir_byte,
        buttons: snapshot.create_sensor_buttons,
        charging_state: snapshot.create_sensor_charging_state,
        voltage_mv: snapshot.create_sensor_voltage_mv,
        current_ma: snapshot.create_sensor_current_ma,
        temperature_c: snapshot.create_sensor_temperature_c,
        charge_mah: snapshot.create_sensor_charge_mah,
        capacity_mah: snapshot.create_sensor_capacity_mah,
        cliff_left_signal: snapshot.create_sensor_cliff_left_signal,
        cliff_front_left_signal: snapshot.create_sensor_cliff_front_left_signal,
        cliff_front_right_signal: snapshot.create_sensor_cliff_front_right_signal,
        cliff_right_signal: snapshot.create_sensor_cliff_right_signal,
    }
}

#[allow(dead_code)]
fn body_kind() -> &'static str {
    match body::BODY_KIND {
        body::BodyKind::CreateOpenInterface => "create_oi",
    }
}

pub fn public_event_kind_text(code: u8) -> &'static str {
    match code {
        x if x == PublicEventKind::Boot as u8 => "boot",
        x if x == PublicEventKind::CommandAccepted as u8 => "command_accepted",
        x if x == PublicEventKind::CommandRejected as u8 => "command_rejected",
        x if x == PublicEventKind::CommandStarted as u8 => "command_started",
        x if x == PublicEventKind::CommandCompleted as u8 => "command_completed",
        x if x == PublicEventKind::CommandInterrupted as u8 => "command_interrupted",
        x if x == PublicEventKind::CommandTimedOut as u8 => "command_timed_out",
        x if x == PublicEventKind::BodyPowerRequested as u8 => "body_power_requested",
        x if x == PublicEventKind::BodyPowerChanged as u8 => "body_power_changed",
        x if x == PublicEventKind::BodyModeRequested as u8 => "body_mode_requested",
        x if x == PublicEventKind::BodyModeChanged as u8 => "body_mode_changed",
        x if x == PublicEventKind::TelemetryReceived as u8 => "telemetry_received",
        x if x == PublicEventKind::SensorFrameDecoded as u8 => "sensor_frame_decoded",
        x if x == PublicEventKind::MotionRequested as u8 => "motion_requested",
        x if x == PublicEventKind::MotionStopped as u8 => "motion_stopped",
        x if x == PublicEventKind::SafetyTripped as u8 => "safety_tripped",
        x if x == PublicEventKind::SafetyCleared as u8 => "safety_cleared",
        x if x == PublicEventKind::BumpChanged as u8 => "bump_changed",
        x if x == PublicEventKind::CliffChanged as u8 => "cliff_changed",
        x if x == PublicEventKind::WheelDropLatched as u8 => "wheel_drop_latched",
        x if x == PublicEventKind::WheelDropCleared as u8 => "wheel_drop_cleared",
        x if x == PublicEventKind::HeartbeatExpired as u8 => "heartbeat_expired",
        x if x == PublicEventKind::EStopLatched as u8 => "estop_latched",
        x if x == PublicEventKind::EStopCleared as u8 => "estop_cleared",
        x if x == PublicEventKind::Error as u8 => "error",
        _ => "none",
    }
}

#[cfg(feature = "pico-w")]
fn runtime_state_text(code: u8) -> &'static str {
    match code {
        x if x == RuntimeState::Booting as u8 => "booting",
        x if x == RuntimeState::RunningDemo as u8 => "running_demo",
        x if x == RuntimeState::Idle as u8 => "idle",
        x if x == RuntimeState::Error as u8 => "error",
        _ => "unknown",
    }
}

#[cfg(feature = "pico-w")]
fn demo_state_text(code: u8) -> &'static str {
    match code {
        x if x == DemoState::NotStarted as u8 => "not_started",
        x if x == DemoState::WaitingForCreate as u8 => "waiting_for_create",
        x if x == DemoState::OiStarted as u8 => "oi_started",
        x if x == DemoState::Moving as u8 => "moving",
        x if x == DemoState::PowerCycling as u8 => "power_cycling",
        x if x == DemoState::Idle as u8 => "idle",
        x if x == DemoState::Error as u8 => "error",
        _ => "unknown",
    }
}

#[cfg(feature = "pico-w")]
fn tri_state_text(code: u8) -> &'static str {
    match code {
        OFF => "off",
        ON => "on",
        _ => "unknown",
    }
}

#[cfg(feature = "pico-w")]
fn oi_mode_text(code: u8) -> &'static str {
    match code {
        1 => "passive",
        2 => "safe",
        3 => "full",
        _ => "unknown",
    }
}

#[cfg(feature = "pico-w")]
fn uart_health_text(code: u8) -> &'static str {
    match code {
        OFF => "error",
        ON => "ok",
        _ => "unknown",
    }
}

#[cfg(feature = "pico-w")]
fn command_text(code: u8) -> &'static str {
    match code {
        x if x == CommandCode::WakeCreate as u8 => "wake_create",
        x if x == CommandCode::SleepCreate as u8 => "sleep_create",
        x if x == CommandCode::PulseBrc as u8 => "pulse_brc",
        x if x == CommandCode::StartOi as u8 => "start_oi",
        x if x == CommandCode::SetOiPassive as u8 => "set_oi_passive",
        x if x == CommandCode::SetOiSafe as u8 => "set_oi_safe",
        x if x == CommandCode::SetOiFull as u8 => "set_oi_full",
        x if x == CommandCode::Drive as u8 => "drive",
        x if x == CommandCode::StopDrive as u8 => "stop_drive",
        _ => "none",
    }
}

#[cfg(feature = "pico-w")]
fn error_text(code: u8) -> &'static str {
    match code {
        x if x == ErrorCode::CreateNoResponse as u8 => "create_no_response",
        x if x == ErrorCode::UartFraming as u8 => "uart_framing",
        x if x == ErrorCode::Timeout as u8 => "timeout",
        x if x == ErrorCode::InvalidPacket as u8 => "invalid_packet",
        _ => "none",
    }
}

#[cfg(feature = "pico-w")]
fn uart_read_error_text(code: u8) -> &'static str {
    match code {
        x if x == UartReadErrorCode::Overrun as u8 => "overrun",
        x if x == UartReadErrorCode::Break as u8 => "break",
        x if x == UartReadErrorCode::Parity as u8 => "parity",
        x if x == UartReadErrorCode::Framing as u8 => "framing",
        x if x == UartReadErrorCode::Other as u8 => "other",
        _ => "none",
    }
}

#[cfg(feature = "pico-w")]
fn forebrain_uart_error_text(code: u8) -> &'static str {
    match code {
        x if x == ForebrainUartErrorCode::LineTooLong as u8 => "line_too_long",
        x if x == ForebrainUartErrorCode::Utf8 as u8 => "utf8",
        x if x == ForebrainUartErrorCode::Parse as u8 => "parse",
        x if x == ForebrainUartErrorCode::Busy as u8 => "busy",
        x if x == ForebrainUartErrorCode::Uart as u8 => "uart",
        _ => "none",
    }
}

#[cfg(feature = "pico-w")]
fn runtime_action_text(code: u8) -> &'static str {
    match code {
        x if x == RuntimeActionCode::PowerPulse as u8 => "power_pulse",
        x if x == RuntimeActionCode::BrcLow as u8 => "brc_low",
        x if x == RuntimeActionCode::BrcSettle as u8 => "brc_settle",
        x if x == RuntimeActionCode::WakeSettle as u8 => "wake_settle",
        x if x == RuntimeActionCode::WaitForCreate as u8 => "wait_for_create",
        x if x == RuntimeActionCode::Settle as u8 => "settle",
        x if x == RuntimeActionCode::Driving as u8 => "driving",
        _ => "none",
    }
}

#[cfg(feature = "pico-w")]
fn control_command_text(code: u8) -> &'static str {
    match code {
        x if x == ControlCommandCode::Ping as u8 => "ping",
        x if x == ControlCommandCode::Arm as u8 => "arm",
        x if x == ControlCommandCode::Disarm as u8 => "disarm",
        x if x == ControlCommandCode::EStop as u8 => "estop",
        x if x == ControlCommandCode::ClearEStop as u8 => "clear_estop",
        x if x == ControlCommandCode::CmdVel as u8 => "cmd_vel",
        x if x == ControlCommandCode::Stop as u8 => "stop",
        x if x == ControlCommandCode::Status as u8 => "status",
        x if x == ControlCommandCode::SongPlay as u8 => "song_play",
        x if x == ControlCommandCode::Dock as u8 => "dock",
        x if x == ControlCommandCode::SetLights as u8 => "set_lights",
        x if x == ControlCommandCode::SetMode as u8 => "set_mode",
        x if x == ControlCommandCode::FaceBearing as u8 => "face_bearing",
        x if x == ControlCommandCode::TrackBearing as u8 => "track_bearing",
        x if x == ControlCommandCode::TurnBy as u8 => "turn_by",
        x if x == ControlCommandCode::DriveFor as u8 => "drive_for",
        x if x == ControlCommandCode::BumpEscape as u8 => "bump_escape",
        x if x == ControlCommandCode::HeartbeatStop as u8 => "heartbeat_stop",
        x if x == ControlCommandCode::HoldHeading as u8 => "hold_heading",
        x if x == ControlCommandCode::TurnToHeading as u8 => "turn_to_heading",
        x if x == ControlCommandCode::ArcFor as u8 => "arc_for",
        x if x == ControlCommandCode::CreepUntil as u8 => "creep_until",
        x if x == ControlCommandCode::ScanArc as u8 => "scan_arc",
        x if x == ControlCommandCode::DockAlign as u8 => "dock_align",
        x if x == ControlCommandCode::WallFollow as u8 => "wall_follow",
        x if x == ControlCommandCode::WiggleAlign as u8 => "wiggle_align",
        x if x == ControlCommandCode::Unstick as u8 => "unstick",
        x if x == ControlCommandCode::CliffGuard as u8 => "cliff_guard",
        x if x == ControlCommandCode::SongDefine as u8 => "song_define",
        x if x == ControlCommandCode::DriveDirect as u8 => "drive_direct",
        x if x == ControlCommandCode::DriveArc as u8 => "drive_arc",
        x if x == ControlCommandCode::RequestSensors as u8 => "request_sensors",
        x if x == ControlCommandCode::StreamSensors as u8 => "stream_sensors",
        x if x == ControlCommandCode::SetSafetyPolicy as u8 => "set_safety_policy",
        x if x == ControlCommandCode::ClearMotionQueue as u8 => "clear_motion_queue",
        x if x == ControlCommandCode::DefineChirp as u8 => "define_chirp",
        x if x == ControlCommandCode::PlayFeedback as u8 => "play_feedback",
        x if x == ControlCommandCode::PowerState as u8 => "power_state",
        x if x == ControlCommandCode::CalibrateTurn as u8 => "calibrate_turn",
        x if x == ControlCommandCode::ResetOdometry as u8 => "reset_odometry",
        x if x == ControlCommandCode::GetCapabilities as u8 => "get_capabilities",
        _ => "none",
    }
}

#[cfg(feature = "pico-w")]
fn error_hint_text(snapshot: BrainstemStatus) -> &'static str {
    let uart_error = if snapshot.last_error_uart_read_error == UartReadErrorCode::None as u8 {
        snapshot.last_uart_read_error
    } else {
        snapshot.last_error_uart_read_error
    };

    match (snapshot.last_error, uart_error) {
        (error, uart)
            if error == ErrorCode::UartFraming as u8
                && uart == UartReadErrorCode::Framing as u8 =>
        {
            "UART RX saw an invalid stop bit before any valid Create byte; check TX/RX wiring, common ground, level shifting, baud 57600 8N1, and whether Create TX is idle-high."
        }
        (error, uart)
            if error == ErrorCode::UartFraming as u8
                && uart == UartReadErrorCode::Break as u8 =>
        {
            "UART RX saw a break condition; the RX line may be held low, shorted, inverted, or connected to the wrong signal."
        }
        (error, uart)
            if error == ErrorCode::UartFraming as u8
                && uart == UartReadErrorCode::Parity as u8 =>
        {
            "UART RX saw a parity mismatch; confirm the link is configured as 57600 8N1 with no parity."
        }
        (error, uart)
            if error == ErrorCode::UartFraming as u8
                && uart == UartReadErrorCode::Overrun as u8 =>
        {
            "UART RX overran; bytes arrived faster than the runtime drained them."
        }
        (error, uart)
            if error == ErrorCode::CreateNoResponse as u8
                && uart == UartReadErrorCode::Break as u8 =>
        {
            "Create did not produce valid UART bytes and RX saw a break condition; the RP2040 RX line is being held low, shorted, inverted, or connected to the wrong signal."
        }
        (error, uart)
            if error == ErrorCode::CreateNoResponse as u8
                && uart == UartReadErrorCode::Framing as u8 =>
        {
            "Create did not produce a valid sensor response and RX saw invalid stop bits; check TX/RX crossing, common ground, level shifting, and baud 57600 8N1."
        }
        (error, uart)
            if error == ErrorCode::CreateNoResponse as u8
                && uart == UartReadErrorCode::Overrun as u8 =>
        {
            "Create did not produce a complete wake-probe response before timeout; RX also overran, so stale or noisy incoming bytes may be flooding the UART."
        }
        (error, _) if error == ErrorCode::CreateNoResponse as u8 => {
            "Create did not produce any valid UART byte before the wake timeout; check power, wake wiring, Create baud, TX/RX crossing, common ground, and level shifting."
        }
        _ => "none",
    }
}

#[cfg(feature = "pico-w")]
fn wifi_state_text(code: u8) -> &'static str {
    match code {
        x if x == WifiState::Starting as u8 => "starting",
        x if x == WifiState::ApStarted as u8 => "ap_started",
        x if x == WifiState::ServicesStarted as u8 => "services_started",
        x if x == WifiState::Error as u8 => "error",
        _ => "off",
    }
}

#[cfg(feature = "pico-w")]
fn https_state_text(code: u8) -> &'static str {
    match code {
        x if x == HttpsState::Unavailable as u8 => "unavailable",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect<const N: usize>(since_seq: u32) -> heapless::Vec<PublicEventRecord, N> {
        let mut records = heapless::Vec::<PublicEventRecord, N>::new();
        collect_events_since(since_seq, &mut records);
        records
    }

    #[test]
    fn event_log_reports_oldest_next_and_dropped_before() {
        reset_event_log_for_test();
        mark_command_completed(10);
        mark_command_completed(11);

        assert_eq!(event_oldest_seq(), 1);
        assert_eq!(event_next_seq(), 3);
        assert_eq!(event_dropped_before_seq(0), 0);
        let records = collect::<4>(0);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].seq, 1);
        assert_eq!(records[1].seq, 2);
    }

    #[test]
    fn event_log_ring_overwrite_reports_dropped_before_seq() {
        reset_event_log_for_test();
        for command_id in 0..(EVENT_LOG_CAPACITY as u32 + 4) {
            mark_command_completed(command_id);
        }

        assert_eq!(event_next_seq(), EVENT_LOG_CAPACITY as u32 + 5);
        assert_eq!(event_oldest_seq(), 5);
        assert_eq!(event_dropped_before_seq(0), 5);
        let records = collect::<EVENT_LOG_CAPACITY>(0);
        assert_eq!(records.len(), EVENT_LOG_CAPACITY);
        assert_eq!(records[0].seq, 5);
        assert_eq!(records.last().unwrap().seq, EVENT_LOG_CAPACITY as u32 + 4);
    }

    #[test]
    fn generic_event_name_rendering_has_stable_fallback() {
        assert_eq!(
            public_event_kind_text(PublicEventKind::MotionRequested as u8),
            "motion_requested"
        );
        assert_eq!(public_event_kind_text(250), "none");
    }

    #[test]
    fn command_lifecycle_event_ordering() {
        reset_event_log_for_test();
        mark_command_started(42, ControlCommandCode::CmdVel as u8);
        mark_command_completed(42);

        let records = collect::<4>(0);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].kind, PublicEventKind::CommandStarted as u8);
        assert_eq!(records[0].a, 42);
        assert_eq!(records[0].b, ControlCommandCode::CmdVel as u8 as u32);
        assert_eq!(records[1].kind, PublicEventKind::CommandCompleted as u8);
        assert_eq!(records[1].a, 42);
    }

    #[test]
    fn safety_event_ordering() {
        reset_event_log_for_test();
        mark_estop_latched();
        mark_safety_tripped(SafetyEventKind::EStop);
        mark_estop_cleared();
        mark_safety_cleared(SafetyEventKind::EStop);

        let records = collect::<8>(0);
        let kinds: heapless::Vec<u8, 8> = records.iter().map(|record| record.kind).collect();
        assert_eq!(
            kinds.as_slice(),
            &[
                PublicEventKind::EStopLatched as u8,
                PublicEventKind::SafetyTripped as u8,
                PublicEventKind::EStopCleared as u8,
                PublicEventKind::SafetyCleared as u8,
            ]
        );
        assert_eq!(records[1].a, SafetyEventKind::EStop as u32);
        assert_eq!(records[3].a, SafetyEventKind::EStop as u32);
    }
}
