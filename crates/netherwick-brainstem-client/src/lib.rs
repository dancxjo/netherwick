use std::io::{Read, Write};
use std::net::{SocketAddr, UdpSocket};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serialport::SerialPort;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, BrainstemClientError>;
const DEFAULT_SIM_EVENT_CAPACITY: usize = 32;

#[derive(Debug, Error)]
pub enum BrainstemClientError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serial error: {0}")]
    Serial(#[from] serialport::Error),
    #[error("bad brainstem response: {0}")]
    BadResponse(String),
    #[error("event history missed before sequence {dropped_before_seq}")]
    MissedEvents { dropped_before_seq: u32 },
}

pub trait BrainstemClient {
    fn get_status(&mut self) -> Result<BrainstemStatus>;
    fn get_capabilities(&mut self) -> Result<BrainstemCapabilities>;
    fn get_events_since(&mut self, since_seq: u32) -> Result<EventBatch>;
    fn arm(&mut self) -> Result<()>;
    fn disarm(&mut self) -> Result<()>;
    fn stop(&mut self) -> Result<()>;
    fn estop(&mut self) -> Result<()>;
    fn clear_estop(&mut self) -> Result<()>;
    fn cmd_vel(&mut self, linear_mm_s: i16, angular_mrad_s: i16, ttl_ms: u32) -> Result<()>;
    fn heartbeat_stop(&mut self, timeout_ms: u32) -> Result<()>;
    fn stream_sensors(&mut self, enabled: bool, packet_id: u8, period_ms: u32) -> Result<()>;
    fn reset_odometry(&mut self) -> Result<()>;
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BrainstemStatus {
    pub raw: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BrainstemCapabilities {
    pub body_kind: String,
    pub drive: String,
    pub verbs: Vec<String>,
    pub sensors: Vec<String>,
    pub outputs: Vec<String>,
    pub safety: Vec<String>,
    pub events: Vec<String>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct EventBatch {
    pub since_seq: u32,
    pub oldest_seq: u32,
    pub next_seq: u32,
    pub dropped_before_seq: u32,
    pub events: Vec<BrainstemEvent>,
}

impl EventBatch {
    pub fn ensure_no_missed_events(&self) -> Result<()> {
        if self.dropped_before_seq == 0 {
            Ok(())
        } else {
            Err(BrainstemClientError::MissedEvents {
                dropped_before_seq: self.dropped_before_seq,
            })
        }
    }

    pub fn has_stop_reason(&self) -> bool {
        self.events.iter().any(BrainstemEvent::is_stop_reason)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BrainstemEvent {
    pub seq: u32,
    pub kind: BrainstemEventKind,
    pub a: u32,
    pub b: u32,
    pub c: u32,
}

impl BrainstemEvent {
    pub fn is_stop_reason(&self) -> bool {
        matches!(
            self.kind,
            BrainstemEventKind::SafetyTripped
                | BrainstemEventKind::HeartbeatExpired
                | BrainstemEventKind::EStopLatched
        )
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum BrainstemEventKind {
    Boot,
    CommandAccepted,
    CommandRejected,
    CommandStarted,
    CommandCompleted,
    CommandInterrupted,
    CommandTimedOut,
    BodyPowerRequested,
    BodyPowerChanged,
    BodyModeRequested,
    BodyModeChanged,
    TelemetryReceived,
    SensorFrameDecoded,
    MotionRequested,
    MotionStopped,
    SafetyTripped,
    SafetyCleared,
    BumpChanged,
    CliffChanged,
    WheelDropLatched,
    WheelDropCleared,
    HeartbeatExpired,
    EStopLatched,
    EStopCleared,
    Error,
    Unknown(String),
}

impl From<&str> for BrainstemEventKind {
    fn from(kind: &str) -> Self {
        match kind {
            "boot" => Self::Boot,
            "command_accepted" => Self::CommandAccepted,
            "command_rejected" => Self::CommandRejected,
            "command_started" => Self::CommandStarted,
            "command_completed" => Self::CommandCompleted,
            "command_interrupted" => Self::CommandInterrupted,
            "command_timed_out" => Self::CommandTimedOut,
            "body_power_requested" => Self::BodyPowerRequested,
            "body_power_changed" => Self::BodyPowerChanged,
            "body_mode_requested" => Self::BodyModeRequested,
            "body_mode_changed" => Self::BodyModeChanged,
            "telemetry_received" => Self::TelemetryReceived,
            "sensor_frame_decoded" => Self::SensorFrameDecoded,
            "motion_requested" => Self::MotionRequested,
            "motion_stopped" => Self::MotionStopped,
            "safety_tripped" => Self::SafetyTripped,
            "safety_cleared" => Self::SafetyCleared,
            "bump_changed" => Self::BumpChanged,
            "cliff_changed" => Self::CliffChanged,
            "wheel_drop_latched" => Self::WheelDropLatched,
            "wheel_drop_cleared" => Self::WheelDropCleared,
            "heartbeat_expired" => Self::HeartbeatExpired,
            "estop_latched" => Self::EStopLatched,
            "estop_cleared" => Self::EStopCleared,
            "error" => Self::Error,
            other => Self::Unknown(other.to_owned()),
        }
    }
}

impl BrainstemEventKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Boot => "boot",
            Self::CommandAccepted => "command_accepted",
            Self::CommandRejected => "command_rejected",
            Self::CommandStarted => "command_started",
            Self::CommandCompleted => "command_completed",
            Self::CommandInterrupted => "command_interrupted",
            Self::CommandTimedOut => "command_timed_out",
            Self::BodyPowerRequested => "body_power_requested",
            Self::BodyPowerChanged => "body_power_changed",
            Self::BodyModeRequested => "body_mode_requested",
            Self::BodyModeChanged => "body_mode_changed",
            Self::TelemetryReceived => "telemetry_received",
            Self::SensorFrameDecoded => "sensor_frame_decoded",
            Self::MotionRequested => "motion_requested",
            Self::MotionStopped => "motion_stopped",
            Self::SafetyTripped => "safety_tripped",
            Self::SafetyCleared => "safety_cleared",
            Self::BumpChanged => "bump_changed",
            Self::CliffChanged => "cliff_changed",
            Self::WheelDropLatched => "wheel_drop_latched",
            Self::WheelDropCleared => "wheel_drop_cleared",
            Self::HeartbeatExpired => "heartbeat_expired",
            Self::EStopLatched => "estop_latched",
            Self::EStopCleared => "estop_cleared",
            Self::Error => "error",
            Self::Unknown(kind) => kind.as_str(),
        }
    }
}

#[derive(Debug, Clone)]
struct SimTimedAction {
    command_id: u32,
    complete_at_ms: u32,
}

#[derive(Debug, Clone)]
pub struct SimBrainstemClient {
    capabilities: BrainstemCapabilities,
    events: Vec<BrainstemEvent>,
    next_event_seq: u32,
    event_capacity: usize,
    now_ms: u32,
    next_command_id: u32,
    armed: bool,
    estop_latched: bool,
    safety_tripped: bool,
    active_cmd_vel: Option<SimTimedAction>,
    heartbeat_stop_at_ms: Option<u32>,
    odometry_reset_count: u32,
}

impl SimBrainstemClient {
    pub fn new() -> Self {
        let mut sim = Self {
            capabilities: BrainstemCapabilities {
                body_kind: "sim_create_oi".to_owned(),
                drive: "differential".to_owned(),
                verbs: [
                    "ping",
                    "arm",
                    "stop",
                    "disarm",
                    "estop",
                    "clear_estop",
                    "cmd_vel",
                    "heartbeat_stop",
                    "stream_sensors",
                    "reset_odometry",
                    "get_capabilities",
                    "get_events",
                ]
                .into_iter()
                .map(ToOwned::to_owned)
                .collect(),
                sensors: ["bump", "cliff", "wheel_drop", "battery", "odometry"]
                    .into_iter()
                    .map(ToOwned::to_owned)
                    .collect(),
                outputs: ["drive", "lights", "song"]
                    .into_iter()
                    .map(ToOwned::to_owned)
                    .collect(),
                safety: ["estop", "heartbeat", "bump", "cliff", "wheel_drop"]
                    .into_iter()
                    .map(ToOwned::to_owned)
                    .collect(),
                events: [
                    "boot",
                    "command_accepted",
                    "command_started",
                    "command_completed",
                    "command_interrupted",
                    "motion_requested",
                    "motion_stopped",
                    "safety_tripped",
                    "safety_cleared",
                    "heartbeat_expired",
                    "estop_latched",
                    "estop_cleared",
                ]
                .into_iter()
                .map(ToOwned::to_owned)
                .collect(),
            },
            events: Vec::new(),
            next_event_seq: 1,
            event_capacity: DEFAULT_SIM_EVENT_CAPACITY,
            now_ms: 0,
            next_command_id: 1,
            armed: false,
            estop_latched: false,
            safety_tripped: false,
            active_cmd_vel: None,
            heartbeat_stop_at_ms: None,
            odometry_reset_count: 0,
        };
        sim.push_event(BrainstemEventKind::Boot, 0, 0, 0);
        sim
    }

    pub fn with_event_capacity(mut self, event_capacity: usize) -> Self {
        self.event_capacity = event_capacity.max(1);
        self.enforce_event_capacity();
        self
    }

    pub fn advance_ms(&mut self, ms: u32) {
        self.now_ms = self.now_ms.wrapping_add(ms);
        self.complete_due_cmd_vel();
        self.expire_heartbeat_if_due();
    }

    pub fn trip_safety(&mut self) {
        if self.safety_tripped {
            return;
        }
        self.safety_tripped = true;
        self.interrupt_active_motion();
        self.push_event(BrainstemEventKind::SafetyTripped, 1, 0, 0);
        self.push_event(BrainstemEventKind::MotionStopped, 0, 0, 0);
    }

    pub fn odometry_reset_count(&self) -> u32 {
        self.odometry_reset_count
    }

    fn accept_command(&mut self) -> u32 {
        let id = self.next_command_id;
        self.next_command_id = self.next_command_id.wrapping_add(1).max(1);
        self.push_event(BrainstemEventKind::CommandAccepted, id, 0, 0);
        self.push_event(BrainstemEventKind::CommandStarted, id, 0, 0);
        id
    }

    fn complete_command(&mut self, id: u32) {
        self.push_event(BrainstemEventKind::CommandCompleted, id, 0, 0);
    }

    fn push_event(&mut self, kind: BrainstemEventKind, a: u32, b: u32, c: u32) {
        let seq = self.next_event_seq;
        self.next_event_seq = self.next_event_seq.wrapping_add(1).max(1);
        self.events.push(BrainstemEvent { seq, kind, a, b, c });
        self.enforce_event_capacity();
    }

    fn enforce_event_capacity(&mut self) {
        let overflow = self.events.len().saturating_sub(self.event_capacity);
        if overflow > 0 {
            self.events.drain(0..overflow);
        }
    }

    fn interrupt_active_motion(&mut self) {
        if let Some(active) = self.active_cmd_vel.take() {
            self.push_event(
                BrainstemEventKind::CommandInterrupted,
                active.command_id,
                0,
                0,
            );
        }
    }

    fn complete_due_cmd_vel(&mut self) {
        let Some(active) = self.active_cmd_vel.clone() else {
            return;
        };
        if !time_reached(self.now_ms, active.complete_at_ms) {
            return;
        }
        self.active_cmd_vel = None;
        self.push_event(BrainstemEventKind::MotionStopped, 0, 0, 0);
        self.complete_command(active.command_id);
    }

    fn expire_heartbeat_if_due(&mut self) {
        let Some(deadline_ms) = self.heartbeat_stop_at_ms else {
            return;
        };
        if !time_reached(self.now_ms, deadline_ms) {
            return;
        }
        self.heartbeat_stop_at_ms = None;
        self.interrupt_active_motion();
        self.safety_tripped = true;
        self.push_event(BrainstemEventKind::HeartbeatExpired, 0, 0, 0);
        self.push_event(BrainstemEventKind::SafetyTripped, 5, 0, 0);
        self.push_event(BrainstemEventKind::MotionStopped, 0, 0, 0);
    }

    fn oldest_seq(&self) -> u32 {
        self.events
            .first()
            .map(|event| event.seq)
            .unwrap_or(self.next_event_seq)
    }
}

impl Default for SimBrainstemClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BrainstemClient for SimBrainstemClient {
    fn get_status(&mut self) -> Result<BrainstemStatus> {
        self.complete_due_cmd_vel();
        self.expire_heartbeat_if_due();
        Ok(BrainstemStatus {
            raw: format!(
                "OK 0 STATUS sim=true now_ms={} armed={} estop={} safety_tripped={} active_cmd_vel={} odometry_resets={}",
                self.now_ms,
                self.armed,
                self.estop_latched,
                self.safety_tripped,
                self.active_cmd_vel.is_some(),
                self.odometry_reset_count
            ),
        })
    }

    fn get_capabilities(&mut self) -> Result<BrainstemCapabilities> {
        Ok(self.capabilities.clone())
    }

    fn get_events_since(&mut self, since_seq: u32) -> Result<EventBatch> {
        self.complete_due_cmd_vel();
        self.expire_heartbeat_if_due();
        let oldest_seq = self.oldest_seq();
        let dropped_before_seq = if since_seq.saturating_add(1) < oldest_seq {
            oldest_seq
        } else {
            0
        };
        Ok(EventBatch {
            since_seq,
            oldest_seq,
            next_seq: self.next_event_seq,
            dropped_before_seq,
            events: self
                .events
                .iter()
                .filter(|event| event.seq > since_seq)
                .cloned()
                .collect(),
        })
    }

    fn arm(&mut self) -> Result<()> {
        let id = self.accept_command();
        self.armed = true;
        self.complete_command(id);
        Ok(())
    }

    fn disarm(&mut self) -> Result<()> {
        let id = self.accept_command();
        self.interrupt_active_motion();
        self.armed = false;
        self.complete_command(id);
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        let id = self.accept_command();
        self.interrupt_active_motion();
        self.push_event(BrainstemEventKind::MotionStopped, 0, 0, 0);
        self.complete_command(id);
        Ok(())
    }

    fn estop(&mut self) -> Result<()> {
        let id = self.accept_command();
        self.interrupt_active_motion();
        self.estop_latched = true;
        self.safety_tripped = true;
        self.push_event(BrainstemEventKind::EStopLatched, 1, 0, 0);
        self.push_event(BrainstemEventKind::SafetyTripped, 4, 0, 0);
        self.push_event(BrainstemEventKind::MotionStopped, 0, 0, 0);
        self.complete_command(id);
        Ok(())
    }

    fn clear_estop(&mut self) -> Result<()> {
        let id = self.accept_command();
        self.estop_latched = false;
        self.safety_tripped = false;
        self.push_event(BrainstemEventKind::EStopCleared, 0, 0, 0);
        self.push_event(BrainstemEventKind::SafetyCleared, 4, 0, 0);
        self.complete_command(id);
        Ok(())
    }

    fn cmd_vel(&mut self, linear_mm_s: i16, angular_mrad_s: i16, ttl_ms: u32) -> Result<()> {
        let id = self.accept_command();
        if self.estop_latched || self.safety_tripped {
            self.push_event(BrainstemEventKind::CommandRejected, id, 0, 0);
            return Ok(());
        }
        self.interrupt_active_motion();
        self.push_event(
            BrainstemEventKind::MotionRequested,
            pack_i16_pair(linear_mm_s, angular_mrad_s),
            ttl_ms,
            0,
        );
        self.active_cmd_vel = Some(SimTimedAction {
            command_id: id,
            complete_at_ms: self.now_ms.wrapping_add(ttl_ms.max(1)),
        });
        Ok(())
    }

    fn heartbeat_stop(&mut self, timeout_ms: u32) -> Result<()> {
        let id = self.accept_command();
        self.heartbeat_stop_at_ms = Some(self.now_ms.wrapping_add(timeout_ms.max(1)));
        self.complete_command(id);
        Ok(())
    }

    fn stream_sensors(&mut self, _enabled: bool, _packet_id: u8, _period_ms: u32) -> Result<()> {
        let id = self.accept_command();
        self.complete_command(id);
        Ok(())
    }

    fn reset_odometry(&mut self) -> Result<()> {
        let id = self.accept_command();
        self.odometry_reset_count = self.odometry_reset_count.saturating_add(1);
        self.complete_command(id);
        Ok(())
    }
}

pub struct EventCursor {
    next_seq: u32,
}

impl EventCursor {
    pub fn new() -> Self {
        Self { next_seq: 0 }
    }

    pub fn next_seq(&self) -> u32 {
        self.next_seq
    }

    pub fn poll<C: BrainstemClient>(&mut self, client: &mut C) -> Result<EventBatch> {
        let batch = client.get_events_since(self.next_seq)?;
        batch.ensure_no_missed_events()?;
        self.next_seq = batch.next_seq.saturating_sub(1);
        Ok(batch)
    }
}

impl Default for EventCursor {
    fn default() -> Self {
        Self::new()
    }
}

pub struct UdpBrainstemClient {
    socket: UdpSocket,
    brainstem: SocketAddr,
    next_seq: u32,
    timeout: Duration,
}

impl UdpBrainstemClient {
    pub fn connect(brainstem: SocketAddr) -> Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        let timeout = Duration::from_millis(750);
        socket.set_read_timeout(Some(timeout))?;
        socket.set_write_timeout(Some(timeout))?;
        Ok(Self {
            socket,
            brainstem,
            next_seq: 1,
            timeout,
        })
    }

    pub fn set_timeout(&mut self, timeout: Duration) -> Result<()> {
        self.timeout = timeout;
        self.socket.set_read_timeout(Some(timeout))?;
        self.socket.set_write_timeout(Some(timeout))?;
        Ok(())
    }

    fn request(&mut self, line: String) -> Result<String> {
        self.socket.send_to(line.as_bytes(), self.brainstem)?;
        let mut buf = [0u8; 2048];
        let (len, _) = self.socket.recv_from(&mut buf)?;
        response_from_bytes(&buf[..len])
    }

    fn seq(&mut self) -> u32 {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1).max(1);
        seq
    }

    fn command(&mut self, kind: &str) -> Result<()> {
        let seq = self.seq();
        expect_ok(seq, &self.request(format!("{kind} {seq}\n"))?)
    }
}

impl BrainstemClient for UdpBrainstemClient {
    fn get_status(&mut self) -> Result<BrainstemStatus> {
        let seq = self.seq();
        let response = self.request(format!("STATUS {seq}\n"))?;
        expect_ok(seq, &response)?;
        Ok(BrainstemStatus { raw: response })
    }

    fn get_capabilities(&mut self) -> Result<BrainstemCapabilities> {
        let seq = self.seq();
        let response = self.request(format!("GET_CAPABILITIES {seq}\n"))?;
        parse_capabilities(seq, &response)
    }

    fn get_events_since(&mut self, since_seq: u32) -> Result<EventBatch> {
        let seq = self.seq();
        let response = self.request(format!("GET_EVENTS {seq} {since_seq}\n"))?;
        parse_events(seq, since_seq, &response)
    }

    fn arm(&mut self) -> Result<()> {
        self.command("ARM")
    }

    fn disarm(&mut self) -> Result<()> {
        self.command("DISARM")
    }

    fn stop(&mut self) -> Result<()> {
        self.command("STOP")
    }

    fn estop(&mut self) -> Result<()> {
        self.command("ESTOP")
    }

    fn clear_estop(&mut self) -> Result<()> {
        self.command("CLEAR_ESTOP")
    }

    fn cmd_vel(&mut self, linear_mm_s: i16, angular_mrad_s: i16, ttl_ms: u32) -> Result<()> {
        let seq = self.seq();
        expect_ok(
            seq,
            &self.request(format!(
                "CMD_VEL {seq} {linear_mm_s} {angular_mrad_s} {ttl_ms}\n"
            ))?,
        )
    }

    fn heartbeat_stop(&mut self, timeout_ms: u32) -> Result<()> {
        let seq = self.seq();
        expect_ok(
            seq,
            &self.request(format!("HEARTBEAT_STOP {seq} {timeout_ms}\n"))?,
        )
    }

    fn stream_sensors(&mut self, enabled: bool, packet_id: u8, period_ms: u32) -> Result<()> {
        let seq = self.seq();
        let enabled = if enabled { "true" } else { "false" };
        expect_ok(
            seq,
            &self.request(format!(
                "STREAM_SENSORS {seq} {enabled} {packet_id} {period_ms}\n"
            ))?,
        )
    }

    fn reset_odometry(&mut self) -> Result<()> {
        self.command("RESET_ODOMETRY")
    }
}

pub const DEFAULT_UART_BAUD_RATE: u32 = 115_200;
pub const DEFAULT_UART_TIMEOUT: Duration = Duration::from_millis(750);
pub const DEFAULT_UART_MAX_RESPONSE_LEN: usize = 2048;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct UartBrainstemClientConfig {
    pub path: PathBuf,
    pub baud_rate: u32,
    pub timeout: Duration,
    pub max_response_len: usize,
}

impl UartBrainstemClientConfig {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            baud_rate: DEFAULT_UART_BAUD_RATE,
            timeout: DEFAULT_UART_TIMEOUT,
            max_response_len: DEFAULT_UART_MAX_RESPONSE_LEN,
        }
    }

    pub fn with_baud_rate(mut self, baud_rate: u32) -> Self {
        self.baud_rate = baud_rate;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_max_response_len(mut self, max_response_len: usize) -> Self {
        self.max_response_len = max_response_len;
        self
    }
}

pub struct UartBrainstemClient {
    port: Box<dyn SerialPort>,
    next_seq: u32,
    timeout: Duration,
    max_response_len: usize,
}

impl UartBrainstemClient {
    pub fn connect(path: impl AsRef<Path>) -> Result<Self> {
        Self::connect_with_config(UartBrainstemClientConfig::new(path.as_ref()))
    }

    pub fn connect_with_config(config: UartBrainstemClientConfig) -> Result<Self> {
        let port = serialport::new(config.path.to_string_lossy(), config.baud_rate)
            .timeout(config.timeout)
            .open()?;
        Ok(Self {
            port,
            next_seq: 1,
            timeout: config.timeout,
            max_response_len: config.max_response_len,
        })
    }

    pub fn from_port(port: Box<dyn SerialPort>) -> Self {
        Self {
            port,
            next_seq: 1,
            timeout: DEFAULT_UART_TIMEOUT,
            max_response_len: DEFAULT_UART_MAX_RESPONSE_LEN,
        }
    }

    pub fn set_timeout(&mut self, timeout: Duration) -> Result<()> {
        self.timeout = timeout;
        self.port.set_timeout(timeout)?;
        Ok(())
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    fn request(&mut self, line: String) -> Result<String> {
        self.port.write_all(line.as_bytes())?;
        self.port.flush()?;
        read_line_response(&mut self.port, self.max_response_len)
    }

    fn seq(&mut self) -> u32 {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1).max(1);
        seq
    }

    fn command(&mut self, kind: &str) -> Result<()> {
        let seq = self.seq();
        expect_ok(seq, &self.request(format!("{kind} {seq}\n"))?)
    }
}

impl BrainstemClient for UartBrainstemClient {
    fn get_status(&mut self) -> Result<BrainstemStatus> {
        let seq = self.seq();
        let response = self.request(format!("STATUS {seq}\n"))?;
        expect_ok(seq, &response)?;
        Ok(BrainstemStatus { raw: response })
    }

    fn get_capabilities(&mut self) -> Result<BrainstemCapabilities> {
        let seq = self.seq();
        let response = self.request(format!("GET_CAPABILITIES {seq}\n"))?;
        parse_capabilities(seq, &response)
    }

    fn get_events_since(&mut self, since_seq: u32) -> Result<EventBatch> {
        let seq = self.seq();
        let response = self.request(format!("GET_EVENTS {seq} {since_seq}\n"))?;
        parse_events(seq, since_seq, &response)
    }

    fn arm(&mut self) -> Result<()> {
        self.command("ARM")
    }

    fn disarm(&mut self) -> Result<()> {
        self.command("DISARM")
    }

    fn stop(&mut self) -> Result<()> {
        self.command("STOP")
    }

    fn estop(&mut self) -> Result<()> {
        self.command("ESTOP")
    }

    fn clear_estop(&mut self) -> Result<()> {
        self.command("CLEAR_ESTOP")
    }

    fn cmd_vel(&mut self, linear_mm_s: i16, angular_mrad_s: i16, ttl_ms: u32) -> Result<()> {
        let seq = self.seq();
        expect_ok(
            seq,
            &self.request(format!(
                "CMD_VEL {seq} {linear_mm_s} {angular_mrad_s} {ttl_ms}\n"
            ))?,
        )
    }

    fn heartbeat_stop(&mut self, timeout_ms: u32) -> Result<()> {
        let seq = self.seq();
        expect_ok(
            seq,
            &self.request(format!("HEARTBEAT_STOP {seq} {timeout_ms}\n"))?,
        )
    }

    fn stream_sensors(&mut self, enabled: bool, packet_id: u8, period_ms: u32) -> Result<()> {
        let seq = self.seq();
        let enabled = if enabled { "true" } else { "false" };
        expect_ok(
            seq,
            &self.request(format!(
                "STREAM_SENSORS {seq} {enabled} {packet_id} {period_ms}\n"
            ))?,
        )
    }

    fn reset_odometry(&mut self) -> Result<()> {
        self.command("RESET_ODOMETRY")
    }
}

fn read_line_response(port: &mut Box<dyn SerialPort>, max_len: usize) -> Result<String> {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        match port.read(&mut byte) {
            Ok(0) => continue,
            Ok(_) if byte[0] == b'\n' => return response_from_bytes(&buf),
            Ok(_) if byte[0] == b'\r' => continue,
            Ok(_) => {
                if buf.len() >= max_len {
                    return Err(BrainstemClientError::BadResponse(
                        "response line exceeded maximum length".into(),
                    ));
                }
                buf.push(byte[0]);
            }
            Err(error) => return Err(error.into()),
        }
    }
}

fn response_from_bytes(bytes: &[u8]) -> Result<String> {
    let response = std::str::from_utf8(bytes)
        .map_err(|_| BrainstemClientError::BadResponse("response was not utf-8".into()))?
        .trim()
        .to_owned();
    Ok(response)
}

fn expect_ok(seq: u32, response: &str) -> Result<()> {
    let mut parts = response.split_ascii_whitespace();
    match (
        parts.next(),
        parts.next().and_then(|value| value.parse::<u32>().ok()),
    ) {
        (Some("OK"), Some(response_seq)) if response_seq == seq => Ok(()),
        _ => Err(BrainstemClientError::BadResponse(response.to_owned())),
    }
}

fn parse_capabilities(seq: u32, response: &str) -> Result<BrainstemCapabilities> {
    expect_ok(seq, response)?;
    let rest = response
        .strip_prefix(&format!("OK {seq} CAPABILITIES "))
        .ok_or_else(|| BrainstemClientError::BadResponse(response.to_owned()))?;
    Ok(BrainstemCapabilities {
        body_kind: value_for(rest, "body_kind").unwrap_or_default().to_owned(),
        drive: value_for(rest, "drive").unwrap_or_default().to_owned(),
        verbs: csv_for(rest, "verbs"),
        sensors: csv_for(rest, "sensors"),
        outputs: csv_for(rest, "outputs"),
        safety: csv_for(rest, "safety"),
        events: csv_for(rest, "events"),
    })
}

fn parse_events(seq: u32, since_seq: u32, response: &str) -> Result<EventBatch> {
    expect_ok(seq, response)?;
    let rest = response
        .strip_prefix(&format!("OK {seq} EVENTS "))
        .ok_or_else(|| BrainstemClientError::BadResponse(response.to_owned()))?;
    let header = rest.split('|').next().unwrap_or(rest);
    let dropped_before_seq = number_for(header, "dropped_before").unwrap_or(0);
    let mut batch = EventBatch {
        since_seq,
        oldest_seq: number_for(header, "oldest").unwrap_or(0),
        next_seq: number_for(header, "next").unwrap_or(since_seq),
        dropped_before_seq,
        events: Vec::new(),
    };
    let mut parsed_count = 0usize;
    for chunk in rest.split('|').skip(1) {
        let chunk = chunk.trim();
        let Some((seq_text, tail)) = chunk.split_once(':') else {
            return Err(BrainstemClientError::BadResponse(response.to_owned()));
        };
        let Some((kind_text, fields)) = tail.split_once(':') else {
            return Err(BrainstemClientError::BadResponse(response.to_owned()));
        };
        let mut nums = fields.split(',');
        let event_seq = seq_text
            .parse()
            .map_err(|_| BrainstemClientError::BadResponse(response.to_owned()))?;
        let a = nums
            .next()
            .ok_or_else(|| BrainstemClientError::BadResponse(response.to_owned()))?
            .parse()
            .map_err(|_| BrainstemClientError::BadResponse(response.to_owned()))?;
        let b = nums
            .next()
            .ok_or_else(|| BrainstemClientError::BadResponse(response.to_owned()))?
            .parse()
            .map_err(|_| BrainstemClientError::BadResponse(response.to_owned()))?;
        let c = nums
            .next()
            .ok_or_else(|| BrainstemClientError::BadResponse(response.to_owned()))?
            .parse()
            .map_err(|_| BrainstemClientError::BadResponse(response.to_owned()))?;
        if nums.next().is_some() {
            return Err(BrainstemClientError::BadResponse(response.to_owned()));
        }
        batch.events.push(BrainstemEvent {
            seq: event_seq,
            kind: BrainstemEventKind::from(kind_text),
            a,
            b,
            c,
        });
        parsed_count += 1;
    }
    if number_for(header, "count").is_some_and(|count| count as usize != parsed_count) {
        return Err(BrainstemClientError::BadResponse(response.to_owned()));
    }
    Ok(batch)
}

fn value_for<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{key}=");
    let start = line.find(&prefix)? + prefix.len();
    let tail = &line[start..];
    Some(tail.split_whitespace().next().unwrap_or(tail))
}

fn csv_for(line: &str, key: &str) -> Vec<String> {
    value_for(line, key)
        .unwrap_or("")
        .split(',')
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn number_for(line: &str, key: &str) -> Option<u32> {
    value_for(line, key)?.parse().ok()
}

fn pack_i16_pair(left: i16, right: i16) -> u32 {
    ((left as u16 as u32) << 16) | right as u16 as u32
}

fn time_reached(now_ms: u32, deadline_ms: u32) -> bool {
    now_ms.wrapping_sub(deadline_ms) < u32::MAX / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_capabilities_round_trip() {
        let mut sim = SimBrainstemClient::new();
        let caps = sim.get_capabilities().unwrap();
        assert_eq!(caps.body_kind, "sim_create_oi");
        assert_eq!(caps.drive, "differential");
        assert!(caps.verbs.contains(&"cmd_vel".to_owned()));
        assert!(caps.events.contains(&"safety_tripped".to_owned()));
    }

    #[test]
    fn simulator_event_cursor_happy_path() {
        let mut sim = SimBrainstemClient::new();
        let mut cursor = EventCursor::new();
        let boot = cursor.poll(&mut sim).unwrap();
        assert_eq!(boot.events[0].kind, BrainstemEventKind::Boot);
        sim.arm().unwrap();
        let batch = cursor.poll(&mut sim).unwrap();
        assert_eq!(cursor.next_seq(), batch.next_seq - 1);
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == BrainstemEventKind::CommandCompleted));
    }

    #[test]
    fn simulator_detects_missed_events_through_dropped_before_seq() {
        let mut sim = SimBrainstemClient::new().with_event_capacity(3);
        for _ in 0..4 {
            sim.arm().unwrap();
        }
        let batch = sim.get_events_since(0).unwrap();
        assert!(batch.dropped_before_seq > 0);
        assert!(matches!(
            batch.ensure_no_missed_events(),
            Err(BrainstemClientError::MissedEvents { .. })
        ));
    }

    #[test]
    fn simulator_arm_stop_disarm_lifecycle() {
        let mut sim = SimBrainstemClient::new();
        sim.arm().unwrap();
        sim.cmd_vel(50, 0, 100).unwrap();
        sim.stop().unwrap();
        sim.disarm().unwrap();
        let batch = sim.get_events_since(0).unwrap();
        let kinds: Vec<_> = batch.events.iter().map(|event| &event.kind).collect();
        assert!(kinds.contains(&&BrainstemEventKind::CommandInterrupted));
        assert!(kinds.contains(&&BrainstemEventKind::MotionStopped));
        assert!(kinds.contains(&&BrainstemEventKind::CommandCompleted));
    }

    #[test]
    fn simulator_cmd_vel_completes_after_ttl() {
        let mut sim = SimBrainstemClient::new();
        sim.cmd_vel(70, 10, 300).unwrap();
        sim.advance_ms(299);
        assert!(!sim
            .get_events_since(0)
            .unwrap()
            .events
            .iter()
            .any(|event| event.kind == BrainstemEventKind::MotionStopped));
        sim.advance_ms(1);
        let batch = sim.get_events_since(0).unwrap();
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == BrainstemEventKind::MotionStopped));
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == BrainstemEventKind::CommandCompleted));
    }

    #[test]
    fn simulator_estop_and_clear_estop() {
        let mut sim = SimBrainstemClient::new();
        sim.estop().unwrap();
        sim.clear_estop().unwrap();
        let batch = sim.get_events_since(0).unwrap();
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == BrainstemEventKind::EStopLatched));
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == BrainstemEventKind::EStopCleared));
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == BrainstemEventKind::SafetyCleared));
    }

    #[test]
    fn simulator_heartbeat_expiry_is_stop_reason() {
        let mut sim = SimBrainstemClient::new();
        sim.cmd_vel(70, 0, 1_000).unwrap();
        sim.heartbeat_stop(100).unwrap();
        sim.advance_ms(100);
        let batch = sim.get_events_since(0).unwrap();
        assert!(batch.has_stop_reason());
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == BrainstemEventKind::HeartbeatExpired));
    }

    #[test]
    fn simulator_safety_tripped_stops_motion_and_rejects_motion() {
        let mut sim = SimBrainstemClient::new();
        sim.cmd_vel(70, 0, 1_000).unwrap();
        sim.trip_safety();
        sim.cmd_vel(10, 0, 100).unwrap();
        let batch = sim.get_events_since(0).unwrap();
        assert!(batch.has_stop_reason());
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == BrainstemEventKind::CommandRejected));
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == BrainstemEventKind::MotionStopped));
    }

    #[test]
    fn simulator_reset_odometry() {
        let mut sim = SimBrainstemClient::new();
        sim.reset_odometry().unwrap();
        assert_eq!(sim.odometry_reset_count(), 1);
        let status = sim.get_status().unwrap();
        assert!(status.raw.contains("odometry_resets=1"));
    }

    #[test]
    fn parses_ok_and_err_responses() {
        assert!(expect_ok(2, "OK 2").is_ok());
        assert!(matches!(
            expect_ok(2, "ERR 2 parse"),
            Err(BrainstemClientError::BadResponse(_))
        ));
    }

    #[test]
    fn parses_status_response_as_raw_status() {
        expect_ok(9, "OK 9 STATUS runtime=idle demo=idle").unwrap();
        let status = BrainstemStatus {
            raw: "OK 9 STATUS runtime=idle demo=idle".to_owned(),
        };
        assert!(status.raw.contains("runtime=idle"));
    }

    #[test]
    fn parses_compact_events() {
        let batch = parse_events(
            7,
            12,
            "OK 7 EVENTS since=12 oldest=4 next=15 dropped_before=0 count=2 | 13:motion_requested:1,2,3 | 14:safety_tripped:2,0,0",
        )
        .unwrap();
        assert_eq!(batch.next_seq, 15);
        assert_eq!(batch.dropped_before_seq, 0);
        assert_eq!(batch.events.len(), 2);
        assert_eq!(batch.events[1].kind, BrainstemEventKind::SafetyTripped);
        assert!(batch.has_stop_reason());
    }

    #[test]
    fn parses_unknown_event_kinds() {
        let batch = parse_events(
            7,
            12,
            "OK 7 EVENTS since=12 oldest=13 next=14 dropped_before=0 count=1 | 13:new_future_event:1,2,3",
        )
        .unwrap();
        assert_eq!(
            batch.events[0].kind,
            BrainstemEventKind::Unknown("new_future_event".to_owned())
        );
        assert_eq!(batch.events[0].kind.as_str(), "new_future_event");
    }

    #[test]
    fn rejects_malformed_or_truncated_event_lines() {
        for line in [
            "OK 7 EVENTS since=12 oldest=13 next=14 dropped_before=0 count=1 | malformed",
            "OK 7 EVENTS since=12 oldest=13 next=14 dropped_before=0 count=1 | 13:motion_requested:1,2",
            "OK 7 EVENTS since=12 oldest=13 next=14 dropped_before=0 count=2 | 13:motion_requested:1,2,3",
        ] {
            assert!(matches!(
                parse_events(7, 12, line),
                Err(BrainstemClientError::BadResponse(_))
            ));
        }
    }

    #[test]
    fn parses_large_event_lists_near_response_buffer_limits() {
        let mut line =
            String::from("OK 7 EVENTS since=0 oldest=1 next=29 dropped_before=0 count=28");
        for seq in 1..29 {
            line.push_str(&format!(
                " | {seq}:motion_requested:{seq},{},{}",
                seq + 1,
                seq + 2
            ));
        }
        assert!(line.len() < DEFAULT_UART_MAX_RESPONSE_LEN);
        let batch = parse_events(7, 0, &line).unwrap();
        assert_eq!(batch.events.len(), 28);
        assert_eq!(batch.events.last().unwrap().seq, 28);
    }

    #[test]
    fn detects_missed_events() {
        let batch = parse_events(
            1,
            0,
            "OK 1 EVENTS since=0 oldest=20 next=52 dropped_before=20 count=0",
        )
        .unwrap();
        assert!(matches!(
            batch.ensure_no_missed_events(),
            Err(BrainstemClientError::MissedEvents {
                dropped_before_seq: 20
            })
        ));
    }

    #[test]
    fn parses_capabilities_without_body_specific_api() {
        let caps = parse_capabilities(
            3,
            "OK 3 CAPABILITIES body_kind=create_oi drive=differential verbs=arm,stop,cmd_vel sensors=bump,battery outputs=lights,song safety=bump,estop events=boot,safety_tripped limits=max_linear_mm_s:500 max_tones=16 song_slots=16 feedback_slots=6 sensor_packets=0,7-31",
        )
        .unwrap();
        assert_eq!(caps.drive, "differential");
        assert_eq!(caps.verbs, ["arm", "stop", "cmd_vel"]);
        assert_eq!(caps.events, ["boot", "safety_tripped"]);
    }

    #[test]
    fn uart_config_defaults_to_forebrain_baud() {
        let config = UartBrainstemClientConfig::new("/dev/ttyTEST0");
        assert_eq!(config.baud_rate, DEFAULT_UART_BAUD_RATE);
        assert_eq!(config.timeout, DEFAULT_UART_TIMEOUT);
        assert_eq!(config.max_response_len, DEFAULT_UART_MAX_RESPONSE_LEN);
    }

    #[test]
    fn malformed_response_maps_to_bad_response() {
        let err = expect_ok(2, "ERR 2 parse").unwrap_err();
        assert!(matches!(err, BrainstemClientError::BadResponse(_)));
    }

    #[test]
    fn mismatched_sequence_maps_to_bad_response() {
        let err = expect_ok(1, "OK 12").unwrap_err();
        assert!(matches!(err, BrainstemClientError::BadResponse(_)));
    }

    #[test]
    fn non_utf8_response_maps_to_bad_response() {
        let err = response_from_bytes(&[0xff]).unwrap_err();
        assert!(matches!(err, BrainstemClientError::BadResponse(_)));
    }
}
