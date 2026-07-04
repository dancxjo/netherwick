use core::sync::atomic::{AtomicU32, AtomicU8, Ordering};

use crate::body;
use crate::commands::{BrainstemCommand, CreateOiMode};
use crate::events::BrainstemError;

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
    pub current_command: u8,
    pub last_error: u8,
    pub demo_state: u8,
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
    LAST_UART_PACKET_TIMESTAMP_MS.store(timestamp_ms, Ordering::Relaxed);
}

pub fn mark_uart_rx_error() {
    UART_RX_HEALTH.store(OFF, Ordering::Relaxed);
}

pub fn set_error(error: BrainstemError) {
    let code = match error {
        BrainstemError::CreateNoResponse => ErrorCode::CreateNoResponse,
        BrainstemError::UartFraming => ErrorCode::UartFraming,
        BrainstemError::Timeout => ErrorCode::Timeout,
        BrainstemError::InvalidPacket => ErrorCode::InvalidPacket,
    };
    LAST_ERROR.store(code as u8, Ordering::Relaxed);
    set_runtime_state(RuntimeState::Error);
    set_demo_state(DemoState::Error);
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
        current_command: CURRENT_COMMAND.load(Ordering::Relaxed),
        last_error: LAST_ERROR.load(Ordering::Relaxed),
        demo_state: DEMO_STATE.load(Ordering::Relaxed),
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
    current_command: &'static str,
    last_error: &'static str,
    demo_state: &'static str,
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
        current_command: command_text(snapshot.current_command),
        last_error: error_text(snapshot.last_error),
        demo_state: demo_state_text(snapshot.demo_state),
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
