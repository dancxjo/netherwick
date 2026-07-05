use core::sync::atomic::{AtomicU32, AtomicU8, Ordering};

use crate::body;
use crate::commands::{BrainstemCommand, CreateOiMode};
use crate::events::{BrainstemError, BrainstemEvent};
use crate::hardware::UartReadError;

const UNKNOWN: u8 = 0;
const OFF: u8 = 1;
const ON: u8 = 2;

static RUNTIME_STATE: AtomicU8 = AtomicU8::new(RuntimeState::Booting as u8);
static CREATE_POWER_STATE: AtomicU8 = AtomicU8::new(UNKNOWN);
static OI_MODE: AtomicU8 = AtomicU8::new(UNKNOWN);
static UART_RX_HEALTH: AtomicU8 = AtomicU8::new(UNKNOWN);
static CURRENT_COMMAND: AtomicU8 = AtomicU8::new(CommandCode::None as u8);
static LAST_ERROR: AtomicU8 = AtomicU8::new(ErrorCode::None as u8);
static DEMO_STATE: AtomicU8 = AtomicU8::new(DemoState::NotStarted as u8);
static LAST_UART_PACKET_TIMESTAMP_MS: AtomicU32 = AtomicU32::new(0);
static LAST_UART_READ_ERROR: AtomicU8 = AtomicU8::new(UartReadErrorCode::None as u8);
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
static PENDING_COMMAND_DURATION_MS: AtomicU32 = AtomicU32::new(0);
static LAST_ACCEPTED_COMMAND_ID: AtomicU32 = AtomicU32::new(0);
static LAST_REJECTED_COMMAND_ID: AtomicU32 = AtomicU32::new(0);

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct BrainstemStatus {
    pub firmware_name: &'static str,
    pub firmware_version: &'static str,
    pub body_name: &'static str,
    pub body_kind: &'static str,
    pub uptime_ms: u32,
    pub current_runtime_state: u8,
    pub create_power_state: u8,
    pub oi_mode: u8,
    pub uart_rx_health: u8,
    pub last_uart_packet_timestamp_ms: u32,
    pub last_uart_read_error: u8,
    pub current_command: u8,
    pub current_runtime_action: u8,
    pub last_error: u8,
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
enum ControlCommandCode {
    None = 0,
    WakeCreate = 1,
    SleepCreate = 2,
    Stop = 3,
    EStop = 4,
    ClearEStop = 5,
    SetModePassive = 6,
    SetModeSafe = 7,
    SetModeFull = 8,
    DriveDirect = 9,
    CmdVel = 10,
    DriveArc = 11,
    Ping = 12,
}

pub fn set_runtime_state(state: RuntimeState) {
    RUNTIME_STATE.store(state as u8, Ordering::Relaxed);
}

pub fn set_demo_state(state: DemoState) {
    DEMO_STATE.store(state as u8, Ordering::Relaxed);
}

pub fn set_command(command: Option<BrainstemCommand>) {
    let code = match command {
        None => CommandCode::None,
        Some(BrainstemCommand::WakeCreate) => CommandCode::WakeCreate,
        Some(BrainstemCommand::SleepCreate) => CommandCode::SleepCreate,
        Some(BrainstemCommand::SetMode(CreateOiMode::Passive)) => CommandCode::SetOiPassive,
        Some(BrainstemCommand::SetMode(CreateOiMode::Safe)) => CommandCode::SetOiSafe,
        Some(BrainstemCommand::SetMode(CreateOiMode::Full)) => CommandCode::SetOiFull,
        Some(BrainstemCommand::Stop) => CommandCode::StopDrive,
        Some(BrainstemCommand::EStop) => CommandCode::StopDrive,
        Some(BrainstemCommand::ClearEStop) => CommandCode::None,
        Some(BrainstemCommand::DriveDirect { .. }) => CommandCode::Drive,
        Some(BrainstemCommand::CmdVel { .. }) => CommandCode::Drive,
        Some(BrainstemCommand::DriveArc { .. }) => CommandCode::Drive,
        Some(BrainstemCommand::Ping) => CommandCode::None,
        Some(BrainstemCommand::PulseBrc) => CommandCode::PulseBrc,
        Some(BrainstemCommand::StartOi) => CommandCode::StartOi,
        Some(BrainstemCommand::SetOiMode(CreateOiMode::Passive)) => CommandCode::SetOiPassive,
        Some(BrainstemCommand::SetOiMode(CreateOiMode::Safe)) => CommandCode::SetOiSafe,
        Some(BrainstemCommand::SetOiMode(CreateOiMode::Full)) => CommandCode::SetOiFull,
        Some(BrainstemCommand::Drive { .. }) => CommandCode::Drive,
        Some(BrainstemCommand::StopDrive) => CommandCode::StopDrive,
    };
    CURRENT_COMMAND.store(code as u8, Ordering::Relaxed);
}

#[cfg(feature = "pico-w")]
pub fn submit_control_command(command_id: u32, command: BrainstemCommand) -> bool {
    if PENDING_COMMAND_KIND.load(Ordering::Relaxed) != ControlCommandCode::None as u8 {
        LAST_REJECTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
        return false;
    }

    let Some((kind, a, b, duration_ms)) = encode_control_command(command) else {
        LAST_REJECTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
        return false;
    };

    PENDING_COMMAND_ID.store(command_id, Ordering::Relaxed);
    PENDING_COMMAND_A.store(a, Ordering::Relaxed);
    PENDING_COMMAND_B.store(b, Ordering::Relaxed);
    PENDING_COMMAND_DURATION_MS.store(duration_ms.unwrap_or(0), Ordering::Relaxed);
    PENDING_COMMAND_KIND.store(kind as u8, Ordering::Relaxed);
    LAST_ACCEPTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
    true
}

pub fn take_control_command() -> Option<BrainstemCommand> {
    let kind = PENDING_COMMAND_KIND.load(Ordering::Relaxed);
    if kind == ControlCommandCode::None as u8 {
        return None;
    }

    let a = PENDING_COMMAND_A.load(Ordering::Relaxed);
    let b = PENDING_COMMAND_B.load(Ordering::Relaxed);
    let duration = match PENDING_COMMAND_DURATION_MS.load(Ordering::Relaxed) {
        0 => None,
        duration_ms => Some(duration_ms),
    };
    PENDING_COMMAND_KIND.store(ControlCommandCode::None as u8, Ordering::Relaxed);

    decode_control_command(kind, a, b, duration)
}

#[cfg(feature = "pico-w")]
fn encode_control_command(
    command: BrainstemCommand,
) -> Option<(ControlCommandCode, u32, u32, Option<u32>)> {
    match command {
        BrainstemCommand::WakeCreate => Some((ControlCommandCode::WakeCreate, 0, 0, None)),
        BrainstemCommand::SleepCreate => Some((ControlCommandCode::SleepCreate, 0, 0, None)),
        BrainstemCommand::Stop | BrainstemCommand::StopDrive => {
            Some((ControlCommandCode::Stop, 0, 0, None))
        }
        BrainstemCommand::EStop => Some((ControlCommandCode::EStop, 0, 0, None)),
        BrainstemCommand::ClearEStop => Some((ControlCommandCode::ClearEStop, 0, 0, None)),
        BrainstemCommand::SetMode(CreateOiMode::Passive)
        | BrainstemCommand::SetOiMode(CreateOiMode::Passive) => {
            Some((ControlCommandCode::SetModePassive, 0, 0, None))
        }
        BrainstemCommand::SetMode(CreateOiMode::Safe)
        | BrainstemCommand::SetOiMode(CreateOiMode::Safe) => {
            Some((ControlCommandCode::SetModeSafe, 0, 0, None))
        }
        BrainstemCommand::SetMode(CreateOiMode::Full)
        | BrainstemCommand::SetOiMode(CreateOiMode::Full) => {
            Some((ControlCommandCode::SetModeFull, 0, 0, None))
        }
        BrainstemCommand::DriveDirect {
            left_mm_s,
            right_mm_s,
            duration_ms,
        } => Some((
            ControlCommandCode::DriveDirect,
            encode_i16(left_mm_s),
            encode_i16(right_mm_s),
            duration_ms,
        )),
        BrainstemCommand::CmdVel {
            linear_mm_s,
            angular_mrad_s,
            duration_ms,
        } => Some((
            ControlCommandCode::CmdVel,
            encode_i16(linear_mm_s),
            encode_i16(angular_mrad_s),
            duration_ms,
        )),
        BrainstemCommand::DriveArc {
            velocity_mm_s,
            radius_mm,
            duration_ms,
        } => Some((
            ControlCommandCode::DriveArc,
            encode_i16(velocity_mm_s),
            encode_i16(radius_mm),
            duration_ms,
        )),
        BrainstemCommand::Ping => Some((ControlCommandCode::Ping, 0, 0, None)),
        _ => None,
    }
}

fn decode_control_command(
    kind: u8,
    a: u32,
    b: u32,
    duration_ms: Option<u32>,
) -> Option<BrainstemCommand> {
    match kind {
        x if x == ControlCommandCode::WakeCreate as u8 => Some(BrainstemCommand::WakeCreate),
        x if x == ControlCommandCode::SleepCreate as u8 => Some(BrainstemCommand::SleepCreate),
        x if x == ControlCommandCode::Stop as u8 => Some(BrainstemCommand::Stop),
        x if x == ControlCommandCode::EStop as u8 => Some(BrainstemCommand::EStop),
        x if x == ControlCommandCode::ClearEStop as u8 => Some(BrainstemCommand::ClearEStop),
        x if x == ControlCommandCode::SetModePassive as u8 => {
            Some(BrainstemCommand::SetMode(CreateOiMode::Passive))
        }
        x if x == ControlCommandCode::SetModeSafe as u8 => {
            Some(BrainstemCommand::SetMode(CreateOiMode::Safe))
        }
        x if x == ControlCommandCode::SetModeFull as u8 => {
            Some(BrainstemCommand::SetMode(CreateOiMode::Full))
        }
        x if x == ControlCommandCode::DriveDirect as u8 => Some(BrainstemCommand::DriveDirect {
            left_mm_s: decode_i16(a),
            right_mm_s: decode_i16(b),
            duration_ms,
        }),
        x if x == ControlCommandCode::CmdVel as u8 => Some(BrainstemCommand::CmdVel {
            linear_mm_s: decode_i16(a),
            angular_mrad_s: decode_i16(b),
            duration_ms,
        }),
        x if x == ControlCommandCode::DriveArc as u8 => Some(BrainstemCommand::DriveArc {
            velocity_mm_s: decode_i16(a),
            radius_mm: decode_i16(b),
            duration_ms,
        }),
        x if x == ControlCommandCode::Ping as u8 => Some(BrainstemCommand::Ping),
        _ => None,
    }
}

#[cfg(feature = "pico-w")]
fn encode_i16(value: i16) -> u32 {
    value as u16 as u32
}

fn decode_i16(value: u32) -> i16 {
    value as u16 as i16
}

pub fn set_runtime_action(action: RuntimeActionCode) {
    CURRENT_RUNTIME_ACTION.store(action as u8, Ordering::Relaxed);
}

pub fn set_create_power_on(on: bool) {
    CREATE_POWER_STATE.store(if on { ON } else { OFF }, Ordering::Relaxed);
}

pub fn set_oi_mode(mode: CreateOiMode) {
    let code = match mode {
        CreateOiMode::Passive => 1,
        CreateOiMode::Safe => 2,
        CreateOiMode::Full => 3,
    };
    OI_MODE.store(code, Ordering::Relaxed);
}

pub fn mark_uart_rx_ok(timestamp_ms: u32) {
    UART_RX_HEALTH.store(ON, Ordering::Relaxed);
    LAST_UART_READ_ERROR.store(UartReadErrorCode::None as u8, Ordering::Relaxed);
    LAST_UART_PACKET_TIMESTAMP_MS.store(timestamp_ms, Ordering::Relaxed);
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

pub fn set_error(error: BrainstemError) {
    let code = match error {
        BrainstemError::CreateNoResponse => ErrorCode::CreateNoResponse,
        BrainstemError::UartFraming => ErrorCode::UartFraming,
        BrainstemError::Timeout => ErrorCode::Timeout,
        BrainstemError::InvalidPacket => ErrorCode::InvalidPacket,
    };
    LAST_ERROR.store(code as u8, Ordering::Relaxed);
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
    let blinks = match event {
        BrainstemEvent::Boot => 1,
        BrainstemEvent::CreatePowerOnRequested | BrainstemEvent::CreatePowerOffRequested => 2,
        BrainstemEvent::CreatePowerToggled => 3,
        BrainstemEvent::CreateBrcPulseRequested | BrainstemEvent::CreateBrcPulsed => 4,
        BrainstemEvent::CreateOiStartRequested | BrainstemEvent::CreateOiModeRequested(_) => 5,
        BrainstemEvent::CreatePacketReceived { .. } => 6,
        BrainstemEvent::DriveRequested { .. } | BrainstemEvent::DriveStopped => 7,
        BrainstemEvent::Error(_) => 8,
        BrainstemEvent::TickMs(_) => return,
    };
    request_led_blinks(blinks);
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

#[cfg(feature = "pico-w")]
fn increment(counter: &AtomicU32) {
    counter.store(
        counter.load(Ordering::Relaxed).saturating_add(1),
        Ordering::Relaxed,
    );
}

#[allow(dead_code)]
pub fn snapshot(uptime_ms: u32) -> BrainstemStatus {
    BrainstemStatus {
        firmware_name: env!("CARGO_PKG_NAME"),
        firmware_version: env!("CARGO_PKG_VERSION"),
        body_name: body::BODY_NAME,
        body_kind: body_kind(),
        uptime_ms,
        current_runtime_state: RUNTIME_STATE.load(Ordering::Relaxed),
        create_power_state: CREATE_POWER_STATE.load(Ordering::Relaxed),
        oi_mode: OI_MODE.load(Ordering::Relaxed),
        uart_rx_health: UART_RX_HEALTH.load(Ordering::Relaxed),
        last_uart_packet_timestamp_ms: LAST_UART_PACKET_TIMESTAMP_MS.load(Ordering::Relaxed),
        last_uart_read_error: LAST_UART_READ_ERROR.load(Ordering::Relaxed),
        current_command: CURRENT_COMMAND.load(Ordering::Relaxed),
        current_runtime_action: CURRENT_RUNTIME_ACTION.load(Ordering::Relaxed),
        last_error: LAST_ERROR.load(Ordering::Relaxed),
        last_error_action: LAST_ERROR_ACTION.load(Ordering::Relaxed),
        demo_state: DEMO_STATE.load(Ordering::Relaxed),
        wifi_state: WIFI_STATE.load(Ordering::Relaxed),
        https_state: HTTPS_STATE.load(Ordering::Relaxed),
        http_requests: HTTP_REQUESTS.load(Ordering::Relaxed),
        dhcp_grants: DHCP_GRANTS.load(Ordering::Relaxed),
        last_web_request_timestamp_ms: LAST_WEB_REQUEST_TIMESTAMP_MS.load(Ordering::Relaxed),
        pending_command: PENDING_COMMAND_KIND.load(Ordering::Relaxed),
        pending_command_id: PENDING_COMMAND_ID.load(Ordering::Relaxed),
        last_accepted_command_id: LAST_ACCEPTED_COMMAND_ID.load(Ordering::Relaxed),
        last_rejected_command_id: LAST_REJECTED_COMMAND_ID.load(Ordering::Relaxed),
    }
}

#[cfg(feature = "pico-w")]
#[derive(serde::Serialize)]
struct StatusJson {
    firmware_name: &'static str,
    firmware_version: &'static str,
    body_name: &'static str,
    body_kind: &'static str,
    uptime_ms: u32,
    current_runtime_state: &'static str,
    create_power_state: &'static str,
    oi_mode: &'static str,
    uart_rx_health: &'static str,
    last_uart_packet_timestamp_ms: u32,
    last_uart_read_error: &'static str,
    current_command: &'static str,
    current_runtime_action: &'static str,
    last_error: &'static str,
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
}

#[cfg(feature = "pico-w")]
pub fn render_json<'a>(snapshot: BrainstemStatus, buffer: &'a mut [u8]) -> Result<&'a str, ()> {
    let status = StatusJson {
        firmware_name: snapshot.firmware_name,
        firmware_version: snapshot.firmware_version,
        body_name: snapshot.body_name,
        body_kind: snapshot.body_kind,
        uptime_ms: snapshot.uptime_ms,
        current_runtime_state: runtime_state_text(snapshot.current_runtime_state),
        create_power_state: tri_state_text(snapshot.create_power_state),
        oi_mode: oi_mode_text(snapshot.oi_mode),
        uart_rx_health: uart_health_text(snapshot.uart_rx_health),
        last_uart_packet_timestamp_ms: snapshot.last_uart_packet_timestamp_ms,
        last_uart_read_error: uart_read_error_text(snapshot.last_uart_read_error),
        current_command: command_text(snapshot.current_command),
        current_runtime_action: runtime_action_text(snapshot.current_runtime_action),
        last_error: error_text(snapshot.last_error),
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
    };
    let len = serde_json_core::to_slice(&status, buffer).map_err(|_| ())?;
    core::str::from_utf8(&buffer[..len]).map_err(|_| ())
}

#[allow(dead_code)]
fn body_kind() -> &'static str {
    match body::BODY_KIND {
        body::BodyKind::CreateOpenInterface => "create_oi",
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
        x if x == ControlCommandCode::WakeCreate as u8 => "wake_create",
        x if x == ControlCommandCode::SleepCreate as u8 => "sleep_create",
        x if x == ControlCommandCode::Stop as u8 => "stop",
        x if x == ControlCommandCode::EStop as u8 => "estop",
        x if x == ControlCommandCode::ClearEStop as u8 => "clear_estop",
        x if x == ControlCommandCode::SetModePassive as u8 => "set_mode_passive",
        x if x == ControlCommandCode::SetModeSafe as u8 => "set_mode_safe",
        x if x == ControlCommandCode::SetModeFull as u8 => "set_mode_full",
        x if x == ControlCommandCode::DriveDirect as u8 => "drive_direct",
        x if x == ControlCommandCode::CmdVel as u8 => "cmd_vel",
        x if x == ControlCommandCode::DriveArc as u8 => "drive_arc",
        x if x == ControlCommandCode::Ping as u8 => "ping",
        _ => "none",
    }
}

#[cfg(feature = "pico-w")]
fn error_hint_text(snapshot: BrainstemStatus) -> &'static str {
    match (snapshot.last_error, snapshot.last_uart_read_error) {
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
