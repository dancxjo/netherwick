use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs, UdpSocket};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serialport::SerialPort;
use thiserror::Error;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{connect, Message, WebSocket};

mod handshake;
pub use handshake::*;

pub type Result<T> = std::result::Result<T, CockpitError>;
const DEFAULT_SIM_EVENT_CAPACITY: usize = 32;
const POSSESSION_LEASE_RENEW_INTERVAL_MS: u32 = 1_000;
const POSSESSION_BUSY_RETRY_ATTEMPTS: usize = 300;
const POSSESSION_BUSY_RETRY_DELAY: Duration = Duration::from_millis(10);

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MotorCommand {
    pub forward: f32,
    pub turn: f32,
}

impl MotorCommand {
    pub fn stop() -> Self {
        Self::default()
    }

    pub fn clamped(self, max_forward: f32, max_turn: f32) -> Self {
        Self {
            forward: self.forward.clamp(-max_forward, max_forward),
            turn: self.turn.clamp(-max_turn, max_turn),
        }
    }

    pub fn to_cockpit_request(self, ttl_ms: u32) -> CockpitRequest {
        if self.forward == 0.0 && self.turn == 0.0 {
            CockpitRequest::Stop
        } else {
            CockpitRequest::CmdVel {
                linear_mm_s: meters_per_second_to_mm_s(self.forward),
                angular_mrad_s: radians_per_second_to_mrad_s(self.turn),
                ttl_ms,
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MotionCommand {
    Stop,
    Forward { speed_m_s: f32 },
    Turn { turn_rad_s: f32 },
    Drive { forward_m_s: f32, turn_rad_s: f32 },
}

impl Default for MotionCommand {
    fn default() -> Self {
        Self::Stop
    }
}

impl MotionCommand {
    pub fn to_motor_command(&self) -> MotorCommand {
        match self {
            Self::Stop => MotorCommand::stop(),
            Self::Forward { speed_m_s } => MotorCommand {
                forward: *speed_m_s,
                turn: 0.0,
            },
            Self::Turn { turn_rad_s } => MotorCommand {
                forward: 0.0,
                turn: *turn_rad_s,
            },
            Self::Drive {
                forward_m_s,
                turn_rad_s,
            } => MotorCommand {
                forward: *forward_m_s,
                turn: *turn_rad_s,
            },
        }
    }
}

pub fn meters_per_second_to_mm_s(value: f32) -> i16 {
    (value * 1000.0)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

pub fn radians_per_second_to_mrad_s(value: f32) -> i16 {
    (value * 1000.0)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

pub fn mm_s_to_meters_per_second(value: i16) -> f32 {
    value as f32 / 1000.0
}

pub fn mrad_s_to_radians_per_second(value: i16) -> f32 {
    value as f32 / 1000.0
}

#[derive(Debug, Error)]
pub enum CockpitError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serial error: {0}")]
    Serial(#[from] serialport::Error),
    #[error("websocket error: {0}")]
    WebSocket(#[from] tungstenite::Error),
    #[error("bad brainstem response: {0}")]
    BadResponse(String),
    #[error("event history missed before sequence {dropped_before_seq}")]
    MissedEvents { dropped_before_seq: u32 },
    #[error("brainstem rejected command {command_id}: {reason}")]
    Rejected { command_id: u32, reason: String },
    #[error("command rejected by policy: {0}")]
    Policy(String),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("handshake rejected ({0:?})")]
    HandshakeRejected(HandshakeReject),
    #[error("stale handshake response: expected nonce {expected}, received {received}")]
    StaleHandshake { expected: String, received: String },
    #[error("unsafe handshake response: {0}")]
    UnsafeHandshake(String),
    #[error("handshake frame exceeds {max} bytes")]
    FrameTooLarge { max: usize },
    #[error("request requires an active cockpit session")]
    SessionRequired,
    #[error("session {session_id} is not the active session")]
    InvalidSession { session_id: String },
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct CockpitLimits {
    pub max_linear_mm_s: i16,
    pub max_angular_mrad_s: i16,
    pub min_ttl_ms: u32,
    pub max_ttl_ms: u32,
}

pub const DEFAULT_INTERNAL_DOMAIN: &str = "pete.internal";
pub const RESERVED_NETWORK_NAMES: &[&str] = &[
    "pete",
    "brainstem",
    "motherbrain",
    "forebrain",
    "gateway",
    "control",
];

pub fn encode_dhcp_lease_identity(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AddressFamily {
    Ipv4,
    Ipv6,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct RegisterNetworkEndpoint {
    pub interface_id: String,
    pub address_family: AddressFamily,
    pub address: String,
    pub hostname: String,
    pub lease_identity: String,
    pub ttl_seconds: u32,
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct NetworkEndpointRegistered {
    pub session_id: String,
    pub fqdn: String,
    pub address: String,
    pub ttl_seconds: u32,
    pub registration_generation: u32,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct NetworkLease {
    pub leased_ip: String,
    pub client_mac: String,
    pub dhcp_client_identifier: String,
    pub requested_hostname: Option<String>,
    pub lease_start: u64,
    pub lease_expiry: u64,
}

impl Default for CockpitLimits {
    fn default() -> Self {
        Self {
            max_linear_mm_s: i16::MAX,
            max_angular_mrad_s: i16::MAX,
            min_ttl_ms: 1,
            max_ttl_ms: u32::MAX,
        }
    }
}

pub trait Cockpit {
    /// Production possession metadata, when this connector owns a scoped
    /// motherbrain lease. Read-only and legacy connectors return `None`.
    fn possession_snapshot(&self) -> Option<PossessionSnapshot> {
        None
    }

    /// Sequence immediately preceding the live event head established by a
    /// validated handshake. Consumers use this to avoid replaying unavailable
    /// history from boot.
    fn event_cursor_hint(&self) -> Option<u32> {
        None
    }

    /// Surrender motherbrain possession after stopping motion. This does not
    /// surrender the brainstem's independent ownership of Create OI.
    fn exorcize(&mut self) -> Result<()> {
        self.stop()
    }

    /// Unscoped transport operation. Production brainstems accept only
    /// read-only and emergency requests here; state-changing requests fail
    /// closed. Use `SessionCockpit`, `ControlCockpit`, or `ServiceCockpit` for
    /// scoped operations.
    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse>;

    fn handshake(&mut self, hello: HandshakeHello) -> Result<HandshakeOutcome>;

    fn execute_in_session(
        &mut self,
        session: &CockpitSession,
        request: CockpitRequest,
    ) -> Result<CockpitResponse>;

    fn execute_with_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ControlLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse>;

    fn execute_with_service_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ServiceLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse>;

    fn get_status(&mut self) -> Result<CockpitStatus> {
        match self.execute(CockpitRequest::GetStatus)? {
            CockpitResponse::Status(status) => Ok(status),
            other => Err(CockpitError::BadResponse(format!("{other:?}"))),
        }
    }

    fn get_capabilities(&mut self) -> Result<CockpitCapabilities> {
        match self.execute(CockpitRequest::GetCapabilities)? {
            CockpitResponse::Capabilities(capabilities) => Ok(capabilities),
            other => Err(CockpitError::BadResponse(format!("{other:?}"))),
        }
    }

    fn get_events_since(&mut self, since_seq: u32) -> Result<EventBatch> {
        match self.execute(CockpitRequest::GetEvents { since_seq })? {
            CockpitResponse::Events(events) => Ok(events),
            other => Err(CockpitError::BadResponse(format!("{other:?}"))),
        }
    }

    fn ping(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::Ping)?)
    }

    fn bootsel(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::Bootsel)?)
    }

    fn arm(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::Arm)?)
    }

    fn disarm(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::Disarm)?)
    }

    fn stop(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::Stop)?)
    }

    fn estop(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::EStop)?)
    }

    fn clear_estop(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::ClearEStop)?)
    }

    fn cmd_vel(&mut self, linear_mm_s: i16, angular_mrad_s: i16, ttl_ms: u32) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::CmdVel {
            linear_mm_s,
            angular_mrad_s,
            ttl_ms,
        })?)
    }

    fn drive_direct(&mut self, left_mm_s: i16, right_mm_s: i16, ttl_ms: u32) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::DriveDirect {
            left_mm_s,
            right_mm_s,
            ttl_ms,
        })?)
    }

    fn drive_arc(&mut self, velocity_mm_s: i16, radius_mm: i16, ttl_ms: u32) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::DriveArc {
            velocity_mm_s,
            radius_mm,
            ttl_ms,
        })?)
    }

    fn face_bearing(
        &mut self,
        bearing_mrad: i16,
        max_angular_mrad_s: i16,
        tolerance_mrad: i16,
        ttl_ms: u32,
    ) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::FaceBearing {
            bearing_mrad,
            max_angular_mrad_s,
            tolerance_mrad,
            ttl_ms,
        })?)
    }

    fn track_bearing(
        &mut self,
        bearing_mrad: i16,
        range_mm: u16,
        max_linear_mm_s: i16,
        max_angular_mrad_s: i16,
        stop_range_mm: u16,
        ttl_ms: u32,
    ) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::TrackBearing {
            bearing_mrad,
            range_mm,
            max_linear_mm_s,
            max_angular_mrad_s,
            stop_range_mm,
            ttl_ms,
        })?)
    }

    fn turn_by(&mut self, angle_mrad: i16, angular_mrad_s: i16, timeout_ms: u32) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::TurnBy {
            angle_mrad,
            angular_mrad_s,
            timeout_ms,
        })?)
    }

    fn drive_for(&mut self, distance_mm: i16, velocity_mm_s: i16, timeout_ms: u32) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::DriveFor {
            distance_mm,
            velocity_mm_s,
            timeout_ms,
        })?)
    }

    fn bump_escape(
        &mut self,
        direction: EscapeDirection,
        backoff_mm_s: i16,
        turn_angular_mrad_s: i16,
    ) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::BumpEscape {
            direction,
            backoff_mm_s,
            turn_angular_mrad_s,
        })?)
    }

    fn hold_heading(
        &mut self,
        heading_error_mrad: i16,
        velocity_mm_s: i16,
        max_angular_mrad_s: i16,
        ttl_ms: u32,
    ) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::HoldHeading {
            heading_error_mrad,
            velocity_mm_s,
            max_angular_mrad_s,
            ttl_ms,
        })?)
    }

    fn turn_to_heading(
        &mut self,
        heading_error_mrad: i16,
        angular_mrad_s: i16,
        tolerance_mrad: i16,
        timeout_ms: u32,
    ) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::TurnToHeading {
            heading_error_mrad,
            angular_mrad_s,
            tolerance_mrad,
            timeout_ms,
        })?)
    }

    fn arc_for(&mut self, velocity_mm_s: i16, radius_mm: i16, duration_ms: u32) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::ArcFor {
            velocity_mm_s,
            radius_mm,
            duration_ms,
        })?)
    }

    fn creep_until(
        &mut self,
        velocity_mm_s: i16,
        angular_mrad_s: i16,
        timeout_ms: u32,
    ) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::CreepUntil {
            velocity_mm_s,
            angular_mrad_s,
            timeout_ms,
        })?)
    }

    fn scan_arc(&mut self, angle_mrad: i16, angular_mrad_s: i16, timeout_ms: u32) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::ScanArc {
            angle_mrad,
            angular_mrad_s,
            timeout_ms,
        })?)
    }

    fn dock_align(
        &mut self,
        bearing_mrad: i16,
        range_mm: u16,
        max_linear_mm_s: i16,
        max_angular_mrad_s: i16,
        stop_range_mm: u16,
        ttl_ms: u32,
    ) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::DockAlign {
            bearing_mrad,
            range_mm,
            max_linear_mm_s,
            max_angular_mrad_s,
            stop_range_mm,
            ttl_ms,
        })?)
    }

    fn wall_follow(
        &mut self,
        distance_error_mm: i16,
        velocity_mm_s: i16,
        max_angular_mrad_s: i16,
        ttl_ms: u32,
    ) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::WallFollow {
            distance_error_mm,
            velocity_mm_s,
            max_angular_mrad_s,
            ttl_ms,
        })?)
    }

    fn wiggle_align(&mut self, amplitude_mrad: i16, angular_mrad_s: i16, cycles: u8) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::WiggleAlign {
            amplitude_mrad,
            angular_mrad_s,
            cycles,
        })?)
    }

    fn unstick(
        &mut self,
        direction: EscapeDirection,
        backoff_mm_s: i16,
        turn_angular_mrad_s: i16,
    ) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::Unstick {
            direction,
            backoff_mm_s,
            turn_angular_mrad_s,
        })?)
    }

    fn cliff_guard(&mut self, clear: bool) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::CliffGuard { clear })?)
    }

    fn heartbeat_stop(&mut self, timeout_ms: u32) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::HeartbeatStop { timeout_ms })?)
    }

    fn request_sensors(&mut self, packet_id: u8) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::RequestSensors { packet_id })?)
    }

    fn stream_sensors(&mut self, enabled: bool, packet_id: u8, period_ms: u32) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::StreamSensors {
            enabled,
            packet_id,
            period_ms,
        })?)
    }

    fn set_safety_policy(&mut self, policy: SafetyPolicy) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::SetSafetyPolicy { policy })?)
    }

    fn clear_motion_queue(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::ClearMotionQueue)?)
    }

    fn define_chirp(&mut self, kind: FeedbackKind, tones: &[SongTone]) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::DefineChirp {
            feedback: kind,
            tones: tones.to_vec(),
        })?)
    }

    fn play_feedback(&mut self, kind: FeedbackKind) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::PlayFeedback { feedback: kind })?)
    }

    fn power_state(&mut self, request: PowerStateRequest) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::PowerState { request })?)
    }

    fn calibrate_turn(&mut self, angular_mrad_s: i16, duration_ms: u32) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::CalibrateTurn {
            angular_mrad_s,
            duration_ms,
        })?)
    }

    fn reset_odometry(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::ResetOdometry)?)
    }

    fn zero_imu_orientation(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::ZeroImuOrientation)?)
    }

    fn clear_imu_orientation(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::ClearImuOrientation)?)
    }

    fn set_mode(&mut self, mode: CreateOiMode) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::SetMode { mode })?)
    }

    fn song_define(&mut self, id: u8, tones: &[SongTone]) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::SongDefine {
            id,
            tones: tones.to_vec(),
        })?)
    }

    fn song_play(&mut self, id: u8) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::SongPlay { id })?)
    }

    fn dock(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::Dock)?)
    }

    fn set_lights(&mut self, pattern: LightPattern) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::SetLights { pattern })?)
    }
}

impl<T: Cockpit + ?Sized> Cockpit for Box<T> {
    fn possession_snapshot(&self) -> Option<PossessionSnapshot> {
        (**self).possession_snapshot()
    }
    fn event_cursor_hint(&self) -> Option<u32> {
        (**self).event_cursor_hint()
    }

    fn exorcize(&mut self) -> Result<()> {
        (**self).exorcize()
    }

    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        (**self).execute(request)
    }

    fn handshake(&mut self, hello: HandshakeHello) -> Result<HandshakeOutcome> {
        (**self).handshake(hello)
    }

    fn execute_in_session(
        &mut self,
        session: &CockpitSession,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        (**self).execute_in_session(session, request)
    }

    fn execute_with_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ControlLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        (**self).execute_with_lease(session, lease, request)
    }

    fn execute_with_service_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ServiceLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        (**self).execute_with_service_lease(session, lease, request)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CreateOiMode {
    Passive,
    Safe,
    Full,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EscapeDirection {
    Left,
    Right,
    Either,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SafetyAction {
    None,
    Stop,
    Backoff,
    BumpEscape,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct SafetyPolicy {
    pub bump: SafetyAction,
    pub cliff: SafetyAction,
    pub wheel_drop_latch: bool,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackKind {
    Ok,
    Error,
    Armed,
    LostTarget,
    DockSeen,
    Danger,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct SongTone {
    pub note: u8,
    pub duration_64ths: u8,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PowerStateRequest {
    Wake,
    Sleep,
    PulseBrc,
    StartOi,
    DebugBaud19200,
    DebugBaud57600,
    DebugBaud115200,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LightPattern {
    Off,
    Status,
    Clean,
    Dock,
    Spot,
    Max,
}

impl CreateOiMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Passive => "passive",
            Self::Safe => "safe",
            Self::Full => "full",
        }
    }
}

impl EscapeDirection {
    fn as_str(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Right => "right",
            Self::Either => "either",
        }
    }
}

impl SafetyAction {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Stop => "stop",
            Self::Backoff => "backoff",
            Self::BumpEscape => "bump_escape",
        }
    }
}

impl FeedbackKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Error => "error",
            Self::Armed => "armed",
            Self::LostTarget => "lost_target",
            Self::DockSeen => "dock_seen",
            Self::Danger => "danger",
        }
    }
}

impl PowerStateRequest {
    fn as_str(self) -> &'static str {
        match self {
            Self::Wake => "wake",
            Self::Sleep => "sleep",
            Self::PulseBrc => "pulse_brc",
            Self::StartOi => "start_oi",
            Self::DebugBaud19200 => "debug_baud_19200",
            Self::DebugBaud57600 => "debug_baud_57600",
            Self::DebugBaud115200 => "debug_baud_115200",
        }
    }
}

impl LightPattern {
    fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Status => "status",
            Self::Clean => "clean",
            Self::Dock => "dock",
            Self::Spot => "spot",
            Self::Max => "max",
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct CockpitStatus {
    pub raw: String,
}

impl CockpitStatus {
    pub fn summary(&self) -> StatusSummary {
        StatusSummary::from_raw(&self.raw)
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct CockpitCapabilities {
    pub body_kind: String,
    pub drive: String,
    pub verbs: Vec<String>,
    pub sensors: Vec<String>,
    pub outputs: Vec<String>,
    pub safety: Vec<String>,
    pub events: Vec<String>,
    #[serde(default)]
    pub limits: CockpitLimits,
}

impl CockpitCapabilities {
    pub fn supports(&self, verb: &str) -> bool {
        self.verbs.iter().any(|candidate| candidate == verb)
    }

    pub fn contract(&self) -> CockpitContract {
        CockpitContract::new(self.clone())
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CockpitContract {
    capabilities: CockpitCapabilities,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ContractReport {
    pub missing_verbs: Vec<String>,
    pub extra_verbs: Vec<String>,
    pub optional_absent_verbs: Vec<String>,
    pub unknown_events: Vec<String>,
}

impl ContractReport {
    pub fn is_clean(&self) -> bool {
        self.missing_verbs.is_empty()
            && self.extra_verbs.is_empty()
            && self.unknown_events.is_empty()
    }
}

impl CockpitContract {
    pub fn new(capabilities: CockpitCapabilities) -> Self {
        Self { capabilities }
    }

    pub fn capabilities(&self) -> &CockpitCapabilities {
        &self.capabilities
    }

    pub fn supports(&self, verb: &str) -> bool {
        self.capabilities.supports(verb)
    }

    pub fn requires_capability(&self, request: &CockpitRequest) -> Option<&'static str> {
        request.required_capability()
    }

    pub fn validate_request(&self, request: &CockpitRequest) -> Result<()> {
        if let Some(verb) = self.requires_capability(request) {
            if !self.supports(verb) {
                return Err(CockpitError::Policy(format!(
                    "unsupported cockpit verb {verb}"
                )));
            }
        }
        self.validate_motion_limits(request)?;
        self.validate_ttl_limits(request)?;
        Ok(())
    }

    pub fn validate_motion_limits(&self, request: &CockpitRequest) -> Result<()> {
        let limits = &self.capabilities.limits;
        let max_linear = limits.max_linear_mm_s.abs();
        let max_angular = limits.max_angular_mrad_s.abs();
        let check_linear = |value: i16, name: &str| {
            if value.abs() > max_linear {
                Err(CockpitError::Policy(format!(
                    "{name} {value} mm/s exceeds max_linear_mm_s {max_linear}"
                )))
            } else {
                Ok(())
            }
        };
        let check_angular = |value: i16, name: &str| {
            if value.abs() > max_angular {
                Err(CockpitError::Policy(format!(
                    "{name} {value} mrad/s exceeds max_angular_mrad_s {max_angular}"
                )))
            } else {
                Ok(())
            }
        };
        match request {
            CockpitRequest::CmdVel {
                linear_mm_s,
                angular_mrad_s,
                ..
            } => {
                check_linear(*linear_mm_s, "linear_mm_s")?;
                check_angular(*angular_mrad_s, "angular_mrad_s")
            }
            CockpitRequest::DriveDirect {
                left_mm_s,
                right_mm_s,
                ..
            } => {
                check_linear(*left_mm_s, "left_mm_s")?;
                check_linear(*right_mm_s, "right_mm_s")
            }
            CockpitRequest::DriveArc { velocity_mm_s, .. }
            | CockpitRequest::ArcFor { velocity_mm_s, .. }
            | CockpitRequest::CreepUntil { velocity_mm_s, .. } => {
                check_linear(*velocity_mm_s, "velocity_mm_s")
            }
            CockpitRequest::HoldHeading {
                velocity_mm_s,
                max_angular_mrad_s,
                ..
            }
            | CockpitRequest::WallFollow {
                velocity_mm_s,
                max_angular_mrad_s,
                ..
            } => {
                check_linear(*velocity_mm_s, "velocity_mm_s")?;
                check_angular(*max_angular_mrad_s, "max_angular_mrad_s")
            }
            CockpitRequest::DriveFor { velocity_mm_s, .. } => {
                check_linear(*velocity_mm_s, "velocity_mm_s")
            }
            CockpitRequest::TrackBearing {
                max_linear_mm_s,
                max_angular_mrad_s,
                ..
            }
            | CockpitRequest::DockAlign {
                max_linear_mm_s,
                max_angular_mrad_s,
                ..
            } => {
                check_linear(*max_linear_mm_s, "max_linear_mm_s")?;
                check_angular(*max_angular_mrad_s, "max_angular_mrad_s")
            }
            CockpitRequest::FaceBearing {
                max_angular_mrad_s, ..
            } => check_angular(*max_angular_mrad_s, "max_angular_mrad_s"),
            CockpitRequest::TurnBy { angular_mrad_s, .. }
            | CockpitRequest::TurnToHeading { angular_mrad_s, .. }
            | CockpitRequest::ScanArc { angular_mrad_s, .. }
            | CockpitRequest::WiggleAlign { angular_mrad_s, .. }
            | CockpitRequest::CalibrateTurn { angular_mrad_s, .. } => {
                check_angular(*angular_mrad_s, "angular_mrad_s")
            }
            CockpitRequest::BumpEscape {
                backoff_mm_s,
                turn_angular_mrad_s,
                ..
            }
            | CockpitRequest::Unstick {
                backoff_mm_s,
                turn_angular_mrad_s,
                ..
            } => {
                check_linear(*backoff_mm_s, "backoff_mm_s")?;
                check_angular(*turn_angular_mrad_s, "turn_angular_mrad_s")
            }
            _ => Ok(()),
        }
    }

    pub fn validate_ttl_limits(&self, request: &CockpitRequest) -> Result<()> {
        let Some(ttl_ms) = request.ttl_or_timeout_ms() else {
            return Ok(());
        };
        let limits = &self.capabilities.limits;
        if ttl_ms < limits.min_ttl_ms || ttl_ms > limits.max_ttl_ms {
            return Err(CockpitError::Policy(format!(
                "ttl/timeout {ttl_ms} ms outside {}..={} ms",
                limits.min_ttl_ms, limits.max_ttl_ms
            )));
        }
        Ok(())
    }

    pub fn clamp_motion_request(&self, request: &CockpitRequest) -> CockpitRequest {
        let linear = self.capabilities.limits.max_linear_mm_s.abs();
        let angular = self.capabilities.limits.max_angular_mrad_s.abs();
        let ttl_min = self.capabilities.limits.min_ttl_ms;
        let ttl_max = self.capabilities.limits.max_ttl_ms;
        let clamp_linear = |value: i16| value.clamp(-linear, linear);
        let clamp_angular = |value: i16| value.clamp(-angular, angular);
        let clamp_ttl = |value: u32| value.clamp(ttl_min, ttl_max);
        match request {
            CockpitRequest::CmdVel {
                linear_mm_s,
                angular_mrad_s,
                ttl_ms,
            } => CockpitRequest::CmdVel {
                linear_mm_s: clamp_linear(*linear_mm_s),
                angular_mrad_s: clamp_angular(*angular_mrad_s),
                ttl_ms: clamp_ttl(*ttl_ms),
            },
            CockpitRequest::HeartbeatStop { timeout_ms } => CockpitRequest::HeartbeatStop {
                timeout_ms: clamp_ttl(*timeout_ms),
            },
            other => other.clone(),
        }
    }

    pub fn validate_event_vocabulary(&self) -> Result<()> {
        let unknown: Vec<_> = self
            .capabilities
            .events
            .iter()
            .filter(|event| {
                matches!(
                    CockpitEventKind::from(event.as_str()),
                    CockpitEventKind::Unknown(_)
                )
            })
            .cloned()
            .collect();
        if unknown.is_empty() {
            Ok(())
        } else {
            Err(CockpitError::Policy(format!(
                "unknown cockpit events: {}",
                unknown.join(",")
            )))
        }
    }

    pub fn validate_local_model(&self) -> ContractReport {
        let modeled_verbs: Vec<_> = CockpitRequest::capability_verbs()
            .into_iter()
            .map(ToOwned::to_owned)
            .collect();
        let optional_verbs = optional_cockpit_verbs();
        let missing_verbs = self
            .capabilities
            .verbs
            .iter()
            .filter(|verb| !modeled_verbs.iter().any(|modeled| modeled == *verb))
            .cloned()
            .collect();
        let extra_verbs = modeled_verbs
            .iter()
            .filter(|verb| {
                !self.capabilities.supports(verb)
                    && !optional_verbs
                        .iter()
                        .any(|optional| optional == &verb.as_str())
            })
            .cloned()
            .collect();
        let optional_absent_verbs = optional_verbs
            .iter()
            .filter(|verb| !self.capabilities.supports(verb))
            .map(|verb| (*verb).to_owned())
            .collect();
        let unknown_events = self
            .capabilities
            .events
            .iter()
            .filter(|event| {
                matches!(
                    CockpitEventKind::from(event.as_str()),
                    CockpitEventKind::Unknown(_)
                )
            })
            .cloned()
            .collect();
        ContractReport {
            missing_verbs,
            extra_verbs,
            optional_absent_verbs,
            unknown_events,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct EventBatch {
    pub since_seq: u32,
    pub oldest_seq: u32,
    pub next_seq: u32,
    pub dropped_before_seq: u32,
    pub events: Vec<CockpitEvent>,
}

impl EventBatch {
    pub fn ensure_no_missed_events(&self) -> Result<()> {
        if self.dropped_before_seq == 0 {
            Ok(())
        } else {
            Err(CockpitError::MissedEvents {
                dropped_before_seq: self.dropped_before_seq,
            })
        }
    }

    pub fn has_stop_reason(&self) -> bool {
        self.events.iter().any(CockpitEvent::is_stop_reason)
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct CockpitEvent {
    pub seq: u32,
    pub kind: CockpitEventKind,
    pub a: u32,
    pub b: u32,
    pub c: u32,
}

impl CockpitEvent {
    pub fn is_stop_reason(&self) -> bool {
        SafeStopReason::from_event(self).is_some()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SafeStopReason {
    SafetyTripped,
    HeartbeatExpired,
    EStopLatched,
}

impl SafeStopReason {
    pub fn from_event(event: &CockpitEvent) -> Option<Self> {
        match event.kind {
            CockpitEventKind::SafetyTripped => Some(Self::SafetyTripped),
            CockpitEventKind::HeartbeatExpired => Some(Self::HeartbeatExpired),
            CockpitEventKind::EStopLatched => Some(Self::EStopLatched),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct StatusSummary {
    pub raw: String,
    pub runtime_state: Option<String>,
    pub armed: Option<bool>,
    pub estop_latched: Option<bool>,
    pub safety_tripped: Option<bool>,
    pub active_motion: Option<bool>,
    pub event_next_seq: Option<u32>,
    pub contact: ContactSummary,
    pub battery: BatterySummary,
    pub odometry: OdometrySummary,
    pub imu: ImuSummary,
}

impl StatusSummary {
    pub fn from_raw(raw: &str) -> Self {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) {
            return Self::from_json(raw, &value);
        }
        Self {
            raw: raw.to_owned(),
            runtime_state: value_for(raw, "runtime").map(ToOwned::to_owned),
            armed: bool_for(raw, "armed"),
            estop_latched: bool_for(raw, "estop"),
            safety_tripped: bool_for(raw, "safety_tripped"),
            active_motion: bool_for(raw, "active_cmd_vel"),
            event_next_seq: number_for(raw, "event_next_seq"),
            contact: ContactSummary::from_raw(raw),
            battery: BatterySummary::from_raw(raw),
            odometry: OdometrySummary::from_raw(raw),
            imu: ImuSummary::from_raw(raw),
        }
    }

    fn from_json(raw: &str, value: &serde_json::Value) -> Self {
        let sensors = value.get("create_sensors");
        let safety_tripped = sensors.map(|sensors| {
            json_bool_value(sensors, "wheel_drop").unwrap_or(false)
                || json_bool_value(sensors, "cliff_left").unwrap_or(false)
                || json_bool_value(sensors, "cliff_front_left").unwrap_or(false)
                || json_bool_value(sensors, "cliff_front_right").unwrap_or(false)
                || json_bool_value(sensors, "cliff_right").unwrap_or(false)
        });
        Self {
            raw: raw.to_owned(),
            runtime_state: json_str_value(value, "current_runtime_state")
                .or_else(|| json_str_value(value, "runtime"))
                .map(ToOwned::to_owned),
            armed: json_str_value(value, "oi_mode").map(|mode| mode == "safe" || mode == "full"),
            estop_latched: None,
            safety_tripped,
            active_motion: json_str_value(value, "current_command")
                .map(|command| command == "drive"),
            event_next_seq: json_u32_value(value, "event_next_seq"),
            contact: ContactSummary::from_json(sensors),
            battery: BatterySummary::from_json(sensors),
            odometry: OdometrySummary::from_json(value.get("odometry")),
            imu: ImuSummary::from_json(value.get("imu")),
        }
    }
}

#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ContactSummary {
    pub bump_left: Option<bool>,
    pub bump_right: Option<bool>,
    pub wheel_drop: Option<bool>,
    pub wall: Option<bool>,
    pub virtual_wall: Option<bool>,
    pub cliff_left: Option<bool>,
    pub cliff_front_left: Option<bool>,
    pub cliff_front_right: Option<bool>,
    pub cliff_right: Option<bool>,
}

impl ContactSummary {
    pub fn any_contact(&self) -> Option<bool> {
        any_known_true([
            self.bump_left,
            self.bump_right,
            self.wall,
            self.virtual_wall,
        ])
    }

    pub fn any_safety_stop(&self) -> Option<bool> {
        any_known_true([
            self.wheel_drop,
            self.cliff_left,
            self.cliff_front_left,
            self.cliff_front_right,
            self.cliff_right,
        ])
    }

    fn from_raw(raw: &str) -> Self {
        Self {
            bump_left: bool_for(raw, "bump_left"),
            bump_right: bool_for(raw, "bump_right"),
            wheel_drop: bool_for(raw, "wheel_drop"),
            wall: bool_for(raw, "wall"),
            virtual_wall: bool_for(raw, "virtual_wall"),
            cliff_left: bool_for(raw, "cliff_left"),
            cliff_front_left: bool_for(raw, "cliff_front_left"),
            cliff_front_right: bool_for(raw, "cliff_front_right"),
            cliff_right: bool_for(raw, "cliff_right"),
        }
    }

    fn from_json(sensors: Option<&serde_json::Value>) -> Self {
        let Some(sensors) = sensors else {
            return Self::default();
        };
        Self {
            bump_left: json_bool_value(sensors, "bump_left"),
            bump_right: json_bool_value(sensors, "bump_right"),
            wheel_drop: json_bool_value(sensors, "wheel_drop"),
            wall: json_bool_value(sensors, "wall"),
            virtual_wall: json_bool_value(sensors, "virtual_wall"),
            cliff_left: json_bool_value(sensors, "cliff_left"),
            cliff_front_left: json_bool_value(sensors, "cliff_front_left"),
            cliff_front_right: json_bool_value(sensors, "cliff_front_right"),
            cliff_right: json_bool_value(sensors, "cliff_right"),
        }
    }
}

#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct BatterySummary {
    pub voltage_mv: Option<u32>,
    pub current_ma: Option<i32>,
    pub charge_mah: Option<u32>,
    pub capacity_mah: Option<u32>,
    pub percent: Option<u8>,
    pub charging_state: Option<u8>,
    pub charging_indicator: Option<bool>,
    pub low: Option<bool>,
}

impl BatterySummary {
    fn from_raw(raw: &str) -> Self {
        let charge_mah = number_for(raw, "charge_mah");
        let capacity_mah = number_for(raw, "capacity_mah");
        let percent = battery_percent(charge_mah, capacity_mah);
        Self {
            voltage_mv: number_for(raw, "voltage_mv"),
            current_ma: signed_number_for(raw, "current_ma"),
            charge_mah,
            capacity_mah,
            percent,
            charging_state: number_for(raw, "charging_state").map(|value| value as u8),
            charging_indicator: bool_for(raw, "charging_indicator"),
            low: percent.map(|value| value <= 20),
        }
    }

    fn from_json(sensors: Option<&serde_json::Value>) -> Self {
        let Some(sensors) = sensors else {
            return Self::default();
        };
        let charge_mah = json_u32_value(sensors, "charge_mah");
        let capacity_mah = json_u32_value(sensors, "capacity_mah");
        let percent = battery_percent(charge_mah, capacity_mah);
        Self {
            voltage_mv: json_u32_value(sensors, "voltage_mv"),
            current_ma: json_i32_value(sensors, "current_ma"),
            charge_mah,
            capacity_mah,
            percent,
            charging_state: json_u32_value(sensors, "charging_state").map(|value| value as u8),
            charging_indicator: json_tri_state_value(sensors, "charging_indicator"),
            low: percent.map(|value| value <= 20),
        }
    }
}

#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct OdometrySummary {
    pub reset_count: Option<u32>,
    pub distance_mm: Option<i32>,
    pub heading_mrad: Option<i32>,
}

impl OdometrySummary {
    fn from_raw(raw: &str) -> Self {
        Self {
            reset_count: number_for(raw, "odometry_resets"),
            distance_mm: signed_number_for(raw, "odometry_distance_mm"),
            heading_mrad: signed_number_for(raw, "odometry_heading_mrad"),
        }
    }

    fn from_json(odometry: Option<&serde_json::Value>) -> Self {
        let Some(odometry) = odometry else {
            return Self::default();
        };
        Self {
            reset_count: json_u32_value(odometry, "reset_count"),
            distance_mm: json_i32_value(odometry, "distance_mm"),
            heading_mrad: json_i32_value(odometry, "heading_mrad"),
        }
    }
}

#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ImuSummary {
    pub present: Option<String>,
    pub health: Option<String>,
    pub sample_age_ms: Option<u32>,
    pub poll_period_ms: Option<u32>,
    pub yaw_mrad: Option<i32>,
    pub yaw_rate_mrad_s: Option<i32>,
    pub accel_magnitude_mm_s2: Option<u32>,
    pub tilt_magnitude_mrad: Option<u32>,
    pub roughness_mm_s2: Option<u32>,
    pub impact_score_mm_s2: Option<u32>,
    pub motion_consistency: Option<String>,
    pub calibration: Option<String>,
}

impl ImuSummary {
    fn from_raw(raw: &str) -> Self {
        Self {
            present: value_for(raw, "imu_present").map(ToOwned::to_owned),
            health: value_for(raw, "imu_health").map(ToOwned::to_owned),
            sample_age_ms: number_for(raw, "imu_age_ms"),
            poll_period_ms: number_for(raw, "imu_poll_ms"),
            yaw_mrad: signed_number_for(raw, "imu_yaw_mrad"),
            yaw_rate_mrad_s: signed_number_for(raw, "imu_yaw_rate_mrad_s"),
            accel_magnitude_mm_s2: number_for(raw, "imu_accel_mag_mm_s2"),
            tilt_magnitude_mrad: number_for(raw, "imu_tilt_mrad"),
            roughness_mm_s2: number_for(raw, "imu_roughness_mm_s2"),
            impact_score_mm_s2: number_for(raw, "imu_impact_mm_s2"),
            motion_consistency: value_for(raw, "imu_motion_consistency").map(ToOwned::to_owned),
            calibration: value_for(raw, "imu_calibration").map(ToOwned::to_owned),
        }
    }

    fn from_json(imu: Option<&serde_json::Value>) -> Self {
        let Some(imu) = imu else {
            return Self::default();
        };
        Self {
            present: json_str_value(imu, "present").map(ToOwned::to_owned),
            health: json_str_value(imu, "health").map(ToOwned::to_owned),
            sample_age_ms: json_u32_value(imu, "sample_age_ms"),
            poll_period_ms: json_u32_value(imu, "poll_period_ms"),
            yaw_mrad: json_i32_value(imu, "yaw_mrad"),
            yaw_rate_mrad_s: json_i32_value(imu, "yaw_rate_mrad_s"),
            accel_magnitude_mm_s2: json_u32_value(imu, "accel_magnitude_mm_s2"),
            tilt_magnitude_mrad: json_u32_value(imu, "tilt_magnitude_mrad"),
            roughness_mm_s2: json_u32_value(imu, "roughness_mm_s2"),
            impact_score_mm_s2: json_u32_value(imu, "impact_score_mm_s2"),
            motion_consistency: json_str_value(imu, "motion_consistency").map(ToOwned::to_owned),
            calibration: json_str_value(imu, "calibration").map(ToOwned::to_owned),
        }
    }
}

fn any_known_true(values: impl IntoIterator<Item = Option<bool>>) -> Option<bool> {
    let mut saw_known = false;
    for value in values {
        match value {
            Some(true) => return Some(true),
            Some(false) => saw_known = true,
            None => {}
        }
    }
    saw_known.then_some(false)
}

fn battery_percent(charge_mah: Option<u32>, capacity_mah: Option<u32>) -> Option<u8> {
    let (Some(charge_mah), Some(capacity_mah)) = (charge_mah, capacity_mah) else {
        return None;
    };
    if capacity_mah == 0 {
        None
    } else {
        Some(((charge_mah * 100) / capacity_mah).min(100) as u8)
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CockpitRequest {
    Ping,
    Bootsel,
    RestartMpu,
    RestartCreate,
    ResetMotherbrain,
    GetStatus,
    GetCapabilities,
    GetEvents {
        since_seq: u32,
    },
    RegisterNetworkEndpoint(RegisterNetworkEndpoint),
    AcquireControlLease {
        authority: ControlAuthority,
        ttl_ms: u32,
    },
    AcquireServiceLease {
        scope: ServiceScope,
        ttl_ms: u32,
    },
    Arm,
    Disarm,
    Stop,
    EStop,
    ClearEStop,
    CmdVel {
        linear_mm_s: i16,
        angular_mrad_s: i16,
        ttl_ms: u32,
    },
    DriveDirect {
        left_mm_s: i16,
        right_mm_s: i16,
        ttl_ms: u32,
    },
    DriveArc {
        velocity_mm_s: i16,
        radius_mm: i16,
        ttl_ms: u32,
    },
    FaceBearing {
        bearing_mrad: i16,
        max_angular_mrad_s: i16,
        tolerance_mrad: i16,
        ttl_ms: u32,
    },
    TrackBearing {
        bearing_mrad: i16,
        range_mm: u16,
        max_linear_mm_s: i16,
        max_angular_mrad_s: i16,
        stop_range_mm: u16,
        ttl_ms: u32,
    },
    TurnBy {
        angle_mrad: i16,
        angular_mrad_s: i16,
        timeout_ms: u32,
    },
    DriveFor {
        distance_mm: i16,
        velocity_mm_s: i16,
        timeout_ms: u32,
    },
    BumpEscape {
        direction: EscapeDirection,
        backoff_mm_s: i16,
        turn_angular_mrad_s: i16,
    },
    HoldHeading {
        heading_error_mrad: i16,
        velocity_mm_s: i16,
        max_angular_mrad_s: i16,
        ttl_ms: u32,
    },
    TurnToHeading {
        heading_error_mrad: i16,
        angular_mrad_s: i16,
        tolerance_mrad: i16,
        timeout_ms: u32,
    },
    ArcFor {
        velocity_mm_s: i16,
        radius_mm: i16,
        duration_ms: u32,
    },
    CreepUntil {
        velocity_mm_s: i16,
        angular_mrad_s: i16,
        timeout_ms: u32,
    },
    ScanArc {
        angle_mrad: i16,
        angular_mrad_s: i16,
        timeout_ms: u32,
    },
    DockAlign {
        bearing_mrad: i16,
        range_mm: u16,
        max_linear_mm_s: i16,
        max_angular_mrad_s: i16,
        stop_range_mm: u16,
        ttl_ms: u32,
    },
    WallFollow {
        distance_error_mm: i16,
        velocity_mm_s: i16,
        max_angular_mrad_s: i16,
        ttl_ms: u32,
    },
    WiggleAlign {
        amplitude_mrad: i16,
        angular_mrad_s: i16,
        cycles: u8,
    },
    Unstick {
        direction: EscapeDirection,
        backoff_mm_s: i16,
        turn_angular_mrad_s: i16,
    },
    CliffGuard {
        clear: bool,
    },
    HeartbeatStop {
        timeout_ms: u32,
    },
    RequestSensors {
        packet_id: u8,
    },
    StreamSensors {
        enabled: bool,
        packet_id: u8,
        period_ms: u32,
    },
    SetSafetyPolicy {
        policy: SafetyPolicy,
    },
    ClearMotionQueue,
    DefineChirp {
        feedback: FeedbackKind,
        tones: Vec<SongTone>,
    },
    PlayFeedback {
        feedback: FeedbackKind,
    },
    PowerState {
        request: PowerStateRequest,
    },
    CalibrateTurn {
        angular_mrad_s: i16,
        duration_ms: u32,
    },
    ResetOdometry,
    ZeroImuOrientation,
    ClearImuOrientation,
    SetMode {
        mode: CreateOiMode,
    },
    SongDefine {
        id: u8,
        tones: Vec<SongTone>,
    },
    SongPlay {
        id: u8,
    },
    Dock,
    SetLights {
        pattern: LightPattern,
    },
}

impl CockpitRequest {
    fn required_service_scope(&self) -> Option<ServiceScope> {
        match self {
            Self::Bootsel => Some(ServiceScope::Bootsel),
            Self::RestartMpu => Some(ServiceScope::RestartMpu),
            Self::RestartCreate => Some(ServiceScope::RestartCreate),
            Self::ResetMotherbrain => Some(ServiceScope::ResetMotherbrain),
            _ => None,
        }
    }

    pub fn authorization_class(&self) -> AuthorizationClass {
        match self {
            Self::Ping | Self::GetStatus | Self::GetCapabilities | Self::GetEvents { .. } => {
                AuthorizationClass::ReadOnly
            }
            Self::Stop | Self::EStop => AuthorizationClass::Emergency,
            Self::RegisterNetworkEndpoint(_)
            | Self::AcquireControlLease { .. }
            | Self::AcquireServiceLease { .. }
            | Self::Disarm => AuthorizationClass::Session,
            Self::Bootsel | Self::RestartMpu | Self::RestartCreate | Self::ResetMotherbrain => {
                AuthorizationClass::ServiceLease
            }
            _ => AuthorizationClass::ControlLease,
        }
    }

    pub fn requires_session(&self) -> bool {
        !matches!(
            self.authorization_class(),
            AuthorizationClass::ReadOnly | AuthorizationClass::Emergency
        )
    }

    pub fn requires_control_authority(&self) -> bool {
        self.authorization_class() == AuthorizationClass::ControlLease
    }

    pub fn verb(&self) -> &'static str {
        match self {
            Self::Ping => "ping",
            Self::Bootsel => "bootsel",
            Self::RestartMpu => "restart_mpu",
            Self::RestartCreate => "restart_create",
            Self::ResetMotherbrain => "reset_motherbrain",
            Self::GetStatus => "status",
            Self::GetCapabilities => "get_capabilities",
            Self::GetEvents { .. } => "get_events",
            Self::RegisterNetworkEndpoint(_) => "register_network_endpoint",
            Self::AcquireControlLease { .. } => "acquire_control_lease",
            Self::AcquireServiceLease { .. } => "acquire_service_lease",
            Self::Arm => "arm",
            Self::Disarm => "disarm",
            Self::Stop => "stop",
            Self::EStop => "estop",
            Self::ClearEStop => "clear_estop",
            Self::CmdVel { .. } => "cmd_vel",
            Self::DriveDirect { .. } => "drive_direct",
            Self::DriveArc { .. } => "drive_arc",
            Self::FaceBearing { .. } => "face_bearing",
            Self::TrackBearing { .. } => "track_bearing",
            Self::TurnBy { .. } => "turn_by",
            Self::DriveFor { .. } => "drive_for",
            Self::BumpEscape { .. } => "bump_escape",
            Self::HoldHeading { .. } => "hold_heading",
            Self::TurnToHeading { .. } => "turn_to_heading",
            Self::ArcFor { .. } => "arc_for",
            Self::CreepUntil { .. } => "creep_until",
            Self::ScanArc { .. } => "scan_arc",
            Self::DockAlign { .. } => "dock_align",
            Self::WallFollow { .. } => "wall_follow",
            Self::WiggleAlign { .. } => "wiggle_align",
            Self::Unstick { .. } => "unstick",
            Self::CliffGuard { .. } => "cliff_guard",
            Self::HeartbeatStop { .. } => "heartbeat_stop",
            Self::RequestSensors { .. } => "request_sensors",
            Self::StreamSensors { .. } => "stream_sensors",
            Self::SetSafetyPolicy { .. } => "set_safety_policy",
            Self::ClearMotionQueue => "clear_motion_queue",
            Self::DefineChirp { .. } => "define_chirp",
            Self::PlayFeedback { .. } => "play_feedback",
            Self::PowerState { .. } => "power_state",
            Self::CalibrateTurn { .. } => "calibrate_turn",
            Self::ResetOdometry => "reset_odometry",
            Self::ZeroImuOrientation => "zero_imu_orientation",
            Self::ClearImuOrientation => "clear_imu_orientation",
            Self::SetMode { .. } => "set_mode",
            Self::SongDefine { .. } => "song_define",
            Self::SongPlay { .. } => "song_play",
            Self::Dock => "dock",
            Self::SetLights { .. } => "set_lights",
        }
    }

    pub fn required_capability(&self) -> Option<&'static str> {
        match self {
            Self::Bootsel
            | Self::RegisterNetworkEndpoint(_)
            | Self::AcquireControlLease { .. }
            | Self::AcquireServiceLease { .. } => None,
            other => Some(other.verb()),
        }
    }

    pub fn capability_verbs() -> Vec<&'static str> {
        sample_cockpit_capability_verbs()
    }

    fn ttl_or_timeout_ms(&self) -> Option<u32> {
        match self {
            Self::CmdVel { ttl_ms, .. }
            | Self::DriveDirect { ttl_ms, .. }
            | Self::DriveArc { ttl_ms, .. }
            | Self::FaceBearing { ttl_ms, .. }
            | Self::TrackBearing { ttl_ms, .. }
            | Self::HoldHeading { ttl_ms, .. }
            | Self::DockAlign { ttl_ms, .. }
            | Self::WallFollow { ttl_ms, .. } => Some(*ttl_ms),
            Self::TurnBy { timeout_ms, .. }
            | Self::DriveFor { timeout_ms, .. }
            | Self::CreepUntil { timeout_ms, .. }
            | Self::ScanArc { timeout_ms, .. }
            | Self::TurnToHeading { timeout_ms, .. } => Some(*timeout_ms),
            Self::ArcFor { duration_ms, .. } | Self::CalibrateTurn { duration_ms, .. } => {
                Some(*duration_ms)
            }
            Self::HeartbeatStop { timeout_ms } => Some(*timeout_ms),
            Self::StreamSensors { period_ms, .. } => Some(*period_ms),
            _ => None,
        }
    }

    pub fn apply<C: Cockpit>(&self, client: &mut C) -> Result<CockpitResponse> {
        match self {
            Self::Ping => client.ping().map(|()| CockpitResponse::Accepted),
            Self::Bootsel => client.bootsel().map(|()| CockpitResponse::Accepted),
            Self::RestartMpu | Self::RestartCreate | Self::ResetMotherbrain => {
                client.execute(self.clone())
            }
            Self::GetStatus => Ok(CockpitResponse::Status(client.get_status()?)),
            Self::GetCapabilities => Ok(CockpitResponse::Capabilities(client.get_capabilities()?)),
            Self::GetEvents { since_seq } => Ok(CockpitResponse::Events(
                client.get_events_since(*since_seq)?,
            )),
            Self::RegisterNetworkEndpoint(_) => Err(CockpitError::SessionRequired),
            Self::AcquireControlLease { .. } => Err(CockpitError::SessionRequired),
            Self::AcquireServiceLease { .. } => Err(CockpitError::SessionRequired),
            Self::Arm => client.arm().map(|()| CockpitResponse::Accepted),
            Self::Disarm => client.disarm().map(|()| CockpitResponse::Accepted),
            Self::Stop => client.stop().map(|()| CockpitResponse::Accepted),
            Self::EStop => client.estop().map(|()| CockpitResponse::Accepted),
            Self::ClearEStop => client.clear_estop().map(|()| CockpitResponse::Accepted),
            Self::CmdVel {
                linear_mm_s,
                angular_mrad_s,
                ttl_ms,
            } => client
                .cmd_vel(*linear_mm_s, *angular_mrad_s, *ttl_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::DriveDirect {
                left_mm_s,
                right_mm_s,
                ttl_ms,
            } => client
                .drive_direct(*left_mm_s, *right_mm_s, *ttl_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::DriveArc {
                velocity_mm_s,
                radius_mm,
                ttl_ms,
            } => client
                .drive_arc(*velocity_mm_s, *radius_mm, *ttl_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::FaceBearing {
                bearing_mrad,
                max_angular_mrad_s,
                tolerance_mrad,
                ttl_ms,
            } => client
                .face_bearing(*bearing_mrad, *max_angular_mrad_s, *tolerance_mrad, *ttl_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::TrackBearing {
                bearing_mrad,
                range_mm,
                max_linear_mm_s,
                max_angular_mrad_s,
                stop_range_mm,
                ttl_ms,
            } => client
                .track_bearing(
                    *bearing_mrad,
                    *range_mm,
                    *max_linear_mm_s,
                    *max_angular_mrad_s,
                    *stop_range_mm,
                    *ttl_ms,
                )
                .map(|()| CockpitResponse::Accepted),
            Self::TurnBy {
                angle_mrad,
                angular_mrad_s,
                timeout_ms,
            } => client
                .turn_by(*angle_mrad, *angular_mrad_s, *timeout_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::DriveFor {
                distance_mm,
                velocity_mm_s,
                timeout_ms,
            } => client
                .drive_for(*distance_mm, *velocity_mm_s, *timeout_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::BumpEscape {
                direction,
                backoff_mm_s,
                turn_angular_mrad_s,
            } => client
                .bump_escape(*direction, *backoff_mm_s, *turn_angular_mrad_s)
                .map(|()| CockpitResponse::Accepted),
            Self::HoldHeading {
                heading_error_mrad,
                velocity_mm_s,
                max_angular_mrad_s,
                ttl_ms,
            } => client
                .hold_heading(
                    *heading_error_mrad,
                    *velocity_mm_s,
                    *max_angular_mrad_s,
                    *ttl_ms,
                )
                .map(|()| CockpitResponse::Accepted),
            Self::TurnToHeading {
                heading_error_mrad,
                angular_mrad_s,
                tolerance_mrad,
                timeout_ms,
            } => client
                .turn_to_heading(
                    *heading_error_mrad,
                    *angular_mrad_s,
                    *tolerance_mrad,
                    *timeout_ms,
                )
                .map(|()| CockpitResponse::Accepted),
            Self::ArcFor {
                velocity_mm_s,
                radius_mm,
                duration_ms,
            } => client
                .arc_for(*velocity_mm_s, *radius_mm, *duration_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::CreepUntil {
                velocity_mm_s,
                angular_mrad_s,
                timeout_ms,
            } => client
                .creep_until(*velocity_mm_s, *angular_mrad_s, *timeout_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::ScanArc {
                angle_mrad,
                angular_mrad_s,
                timeout_ms,
            } => client
                .scan_arc(*angle_mrad, *angular_mrad_s, *timeout_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::DockAlign {
                bearing_mrad,
                range_mm,
                max_linear_mm_s,
                max_angular_mrad_s,
                stop_range_mm,
                ttl_ms,
            } => client
                .dock_align(
                    *bearing_mrad,
                    *range_mm,
                    *max_linear_mm_s,
                    *max_angular_mrad_s,
                    *stop_range_mm,
                    *ttl_ms,
                )
                .map(|()| CockpitResponse::Accepted),
            Self::WallFollow {
                distance_error_mm,
                velocity_mm_s,
                max_angular_mrad_s,
                ttl_ms,
            } => client
                .wall_follow(
                    *distance_error_mm,
                    *velocity_mm_s,
                    *max_angular_mrad_s,
                    *ttl_ms,
                )
                .map(|()| CockpitResponse::Accepted),
            Self::WiggleAlign {
                amplitude_mrad,
                angular_mrad_s,
                cycles,
            } => client
                .wiggle_align(*amplitude_mrad, *angular_mrad_s, *cycles)
                .map(|()| CockpitResponse::Accepted),
            Self::Unstick {
                direction,
                backoff_mm_s,
                turn_angular_mrad_s,
            } => client
                .unstick(*direction, *backoff_mm_s, *turn_angular_mrad_s)
                .map(|()| CockpitResponse::Accepted),
            Self::CliffGuard { clear } => client
                .cliff_guard(*clear)
                .map(|()| CockpitResponse::Accepted),
            Self::HeartbeatStop { timeout_ms } => client
                .heartbeat_stop(*timeout_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::RequestSensors { packet_id } => client
                .request_sensors(*packet_id)
                .map(|()| CockpitResponse::Accepted),
            Self::StreamSensors {
                enabled,
                packet_id,
                period_ms,
            } => client
                .stream_sensors(*enabled, *packet_id, *period_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::ResetOdometry => client.reset_odometry().map(|()| CockpitResponse::Accepted),
            Self::ZeroImuOrientation => client
                .zero_imu_orientation()
                .map(|()| CockpitResponse::Accepted),
            Self::ClearImuOrientation => client
                .clear_imu_orientation()
                .map(|()| CockpitResponse::Accepted),
            Self::SetSafetyPolicy { policy } => client
                .set_safety_policy(*policy)
                .map(|()| CockpitResponse::Accepted),
            Self::ClearMotionQueue => client
                .clear_motion_queue()
                .map(|()| CockpitResponse::Accepted),
            Self::DefineChirp { feedback, tones } => client
                .define_chirp(*feedback, tones)
                .map(|()| CockpitResponse::Accepted),
            Self::PlayFeedback { feedback } => client
                .play_feedback(*feedback)
                .map(|()| CockpitResponse::Accepted),
            Self::PowerState { request } => client
                .power_state(*request)
                .map(|()| CockpitResponse::Accepted),
            Self::CalibrateTurn {
                angular_mrad_s,
                duration_ms,
            } => client
                .calibrate_turn(*angular_mrad_s, *duration_ms)
                .map(|()| CockpitResponse::Accepted),
            Self::SetMode { mode } => client.set_mode(*mode).map(|()| CockpitResponse::Accepted),
            Self::SongDefine { id, tones } => client
                .song_define(*id, tones)
                .map(|()| CockpitResponse::Accepted),
            Self::SongPlay { id } => client.song_play(*id).map(|()| CockpitResponse::Accepted),
            Self::Dock => client.dock().map(|()| CockpitResponse::Accepted),
            Self::SetLights { pattern } => client
                .set_lights(*pattern)
                .map(|()| CockpitResponse::Accepted),
        }
    }

    pub fn to_firmware_json(&self, command_id: u32) -> Result<String> {
        let mut value = serde_json::to_value(self)?;
        if let Some(object) = value.as_object_mut() {
            object.insert("command_id".to_owned(), command_id.into());
            if self.needs_seq() {
                object.insert("seq".to_owned(), command_id.into());
            }
            rewrite_for_firmware_json(self, object);
            if let Some(kind) = object.get_mut("kind") {
                if kind == "get_status" {
                    *kind = "status".into();
                } else if kind == "e_stop" {
                    *kind = "estop".into();
                } else if kind == "clear_e_stop" {
                    *kind = "clear_estop".into();
                }
            }
        }
        Ok(serde_json::to_string(&value)?)
    }

    pub fn to_firmware_json_with_session(
        &self,
        command_id: u32,
        session_id: &str,
    ) -> Result<String> {
        let mut value: serde_json::Value =
            serde_json::from_str(&self.to_firmware_json(command_id)?)?;
        value
            .as_object_mut()
            .expect("request serializes as object")
            .insert(
                "session_id".to_owned(),
                serde_json::Value::String(session_id.to_owned()),
            );
        Ok(serde_json::to_string(&value)?)
    }

    pub fn to_firmware_json_with_authority(
        &self,
        command_id: u32,
        session_id: &str,
        lease_id: &str,
    ) -> Result<String> {
        let mut value: serde_json::Value =
            serde_json::from_str(&self.to_firmware_json_with_session(command_id, session_id)?)?;
        value
            .as_object_mut()
            .expect("request serializes as object")
            .insert("lease_id".into(), lease_id.into());
        Ok(serde_json::to_string(&value)?)
    }

    pub fn to_firmware_json_with_service_authority(
        &self,
        command_id: u32,
        session_id: &str,
        lease_id: &str,
    ) -> Result<String> {
        let mut value: serde_json::Value =
            serde_json::from_str(&self.to_firmware_json_with_session(command_id, session_id)?)?;
        value
            .as_object_mut()
            .expect("request serializes as object")
            .insert("service_lease_id".into(), lease_id.into());
        Ok(serde_json::to_string(&value)?)
    }

    fn needs_seq(&self) -> bool {
        !matches!(
            self,
            Self::Ping
                | Self::Bootsel
                | Self::RestartMpu
                | Self::RestartCreate
                | Self::ResetMotherbrain
                | Self::GetStatus
                | Self::GetCapabilities
                | Self::GetEvents { .. }
                | Self::Arm
                | Self::Disarm
                | Self::Stop
                | Self::EStop
                | Self::ClearEStop
                | Self::SetMode { .. }
                | Self::SongPlay { .. }
                | Self::Dock
                | Self::SetLights { .. }
        )
    }

    fn to_compact_line(&self, seq: u32) -> String {
        match self {
            Self::Ping => format!("PING {seq}\n"),
            Self::Bootsel => format!("BOOTSEL {seq}\n"),
            Self::RestartMpu => format!("RESTART_MPU {seq}\n"),
            Self::RestartCreate => format!("RESTART_CREATE {seq}\n"),
            Self::ResetMotherbrain => format!("RESET_MOTHERBRAIN {seq}\n"),
            Self::GetStatus => format!("STATUS {seq}\n"),
            Self::GetCapabilities => format!("GET_CAPABILITIES {seq}\n"),
            Self::GetEvents { since_seq } => format!("GET_EVENTS {seq} {since_seq}\n"),
            Self::RegisterNetworkEndpoint(registration) => format!(
                "REGISTER_NETWORK_ENDPOINT {seq} {} {} {} {} {}\n",
                registration.interface_id,
                registration.address,
                registration.hostname,
                registration.lease_identity,
                registration.ttl_seconds,
            ),
            Self::AcquireControlLease { authority, ttl_ms } => format!(
                "ACQUIRE_CONTROL_LEASE {seq} {} {ttl_ms}\n",
                match authority { ControlAuthority::Motherbrain => "motherbrain", ControlAuthority::ForebrainRecovery => "forebrain_recovery", ControlAuthority::OperatorDebug => "operator_debug" }
            ),
            Self::AcquireServiceLease { scope, ttl_ms } => format!(
                "ACQUIRE_SERVICE_LEASE {seq} {} {ttl_ms}\n",
                match scope {
                    ServiceScope::Bootsel => "bootsel",
                    ServiceScope::RestartMpu => "restart_mpu",
                    ServiceScope::RestartCreate => "restart_create",
                    ServiceScope::ResetMotherbrain => "reset_motherbrain",
                }
            ),
            Self::Arm => format!("ARM {seq}\n"),
            Self::Disarm => format!("DISARM {seq}\n"),
            Self::Stop => format!("STOP {seq}\n"),
            Self::EStop => format!("ESTOP {seq}\n"),
            Self::ClearEStop => format!("CLEAR_ESTOP {seq}\n"),
            Self::CmdVel {
                linear_mm_s,
                angular_mrad_s,
                ttl_ms,
            } => format!("CMD_VEL {seq} {linear_mm_s} {angular_mrad_s} {ttl_ms}\n"),
            Self::DriveDirect {
                left_mm_s,
                right_mm_s,
                ttl_ms,
            } => format!("DRIVE_DIRECT {seq} {left_mm_s} {right_mm_s} {ttl_ms}\n"),
            Self::DriveArc {
                velocity_mm_s,
                radius_mm,
                ttl_ms,
            } => format!("DRIVE_ARC {seq} {velocity_mm_s} {radius_mm} {ttl_ms}\n"),
            Self::FaceBearing {
                bearing_mrad,
                max_angular_mrad_s,
                tolerance_mrad,
                ttl_ms,
            } => format!(
                "FACE_BEARING {seq} {bearing_mrad} {max_angular_mrad_s} {tolerance_mrad} {ttl_ms}\n"
            ),
            Self::TrackBearing {
                bearing_mrad,
                range_mm,
                max_linear_mm_s,
                max_angular_mrad_s,
                stop_range_mm,
                ttl_ms,
            } => format!(
                "TRACK_BEARING {seq} {bearing_mrad} {range_mm} {max_linear_mm_s} {max_angular_mrad_s} {stop_range_mm} {ttl_ms}\n"
            ),
            Self::TurnBy {
                angle_mrad,
                angular_mrad_s,
                timeout_ms,
            } => format!("TURN_BY {seq} {angle_mrad} {angular_mrad_s} {timeout_ms}\n"),
            Self::DriveFor {
                distance_mm,
                velocity_mm_s,
                timeout_ms,
            } => format!("DRIVE_FOR {seq} {distance_mm} {velocity_mm_s} {timeout_ms}\n"),
            Self::BumpEscape {
                direction,
                backoff_mm_s,
                turn_angular_mrad_s,
            } => format!(
                "BUMP_ESCAPE {seq} {} {backoff_mm_s} {turn_angular_mrad_s}\n",
                direction.as_str()
            ),
            Self::HoldHeading {
                heading_error_mrad,
                velocity_mm_s,
                max_angular_mrad_s,
                ttl_ms,
            } => format!(
                "HOLD_HEADING {seq} {heading_error_mrad} {velocity_mm_s} {max_angular_mrad_s} {ttl_ms}\n"
            ),
            Self::TurnToHeading {
                heading_error_mrad,
                angular_mrad_s,
                tolerance_mrad,
                timeout_ms,
            } => format!(
                "TURN_TO_HEADING {seq} {heading_error_mrad} {angular_mrad_s} {tolerance_mrad} {timeout_ms}\n"
            ),
            Self::ArcFor {
                velocity_mm_s,
                radius_mm,
                duration_ms,
            } => format!("ARC_FOR {seq} {velocity_mm_s} {radius_mm} {duration_ms}\n"),
            Self::CreepUntil {
                velocity_mm_s,
                angular_mrad_s,
                timeout_ms,
            } => format!("CREEP_UNTIL {seq} {velocity_mm_s} {angular_mrad_s} {timeout_ms}\n"),
            Self::ScanArc {
                angle_mrad,
                angular_mrad_s,
                timeout_ms,
            } => format!("SCAN_ARC {seq} {angle_mrad} {angular_mrad_s} {timeout_ms}\n"),
            Self::DockAlign {
                bearing_mrad,
                range_mm,
                max_linear_mm_s,
                max_angular_mrad_s,
                stop_range_mm,
                ttl_ms,
            } => format!(
                "DOCK_ALIGN {seq} {bearing_mrad} {range_mm} {max_linear_mm_s} {max_angular_mrad_s} {stop_range_mm} {ttl_ms}\n"
            ),
            Self::WallFollow {
                distance_error_mm,
                velocity_mm_s,
                max_angular_mrad_s,
                ttl_ms,
            } => format!(
                "WALL_FOLLOW {seq} {distance_error_mm} {velocity_mm_s} {max_angular_mrad_s} {ttl_ms}\n"
            ),
            Self::WiggleAlign {
                amplitude_mrad,
                angular_mrad_s,
                cycles,
            } => format!("WIGGLE_ALIGN {seq} {amplitude_mrad} {angular_mrad_s} {cycles}\n"),
            Self::Unstick {
                direction,
                backoff_mm_s,
                turn_angular_mrad_s,
            } => format!(
                "UNSTICK {seq} {} {backoff_mm_s} {turn_angular_mrad_s}\n",
                direction.as_str()
            ),
            Self::CliffGuard { clear } => format!("CLIFF_GUARD {seq} {clear}\n"),
            Self::HeartbeatStop { timeout_ms } => format!("HEARTBEAT_STOP {seq} {timeout_ms}\n"),
            Self::RequestSensors { packet_id } => format!("REQUEST_SENSORS {seq} {packet_id}\n"),
            Self::StreamSensors {
                enabled,
                packet_id,
                period_ms,
            } => format!("STREAM_SENSORS {seq} {enabled} {packet_id} {period_ms}\n"),
            Self::SetSafetyPolicy { policy } => format!(
                "SET_SAFETY_POLICY {seq} {} {} {}\n",
                policy.bump.as_str(),
                policy.cliff.as_str(),
                policy.wheel_drop_latch
            ),
            Self::ClearMotionQueue => format!("CLEAR_MOTION_QUEUE {seq}\n"),
            Self::DefineChirp { feedback, tones } => {
                format!("DEFINE_CHIRP {seq} {}{}\n", feedback.as_str(), compact_tones(tones))
            }
            Self::PlayFeedback { feedback } => format!("PLAY_FEEDBACK {seq} {}\n", feedback.as_str()),
            Self::PowerState { request } => format!("POWER_STATE {seq} {}\n", request.as_str()),
            Self::CalibrateTurn {
                angular_mrad_s,
                duration_ms,
            } => format!("CALIBRATE_TURN {seq} {angular_mrad_s} {duration_ms}\n"),
            Self::ResetOdometry => format!("RESET_ODOMETRY {seq}\n"),
            Self::ZeroImuOrientation => format!("ZERO_IMU_ORIENTATION {seq}\n"),
            Self::ClearImuOrientation => format!("CLEAR_IMU_ORIENTATION {seq}\n"),
            Self::SetMode { mode } => format!("SET_MODE {seq} {}\n", mode.as_str()),
            Self::SongDefine { id, tones } => {
                format!("SONG_DEFINE {seq} {id}{}\n", compact_tones(tones))
            }
            Self::SongPlay { id } => format!("SONG_PLAY {seq} {id}\n"),
            Self::Dock => format!("DOCK {seq}\n"),
            Self::SetLights { pattern } => format!("SET_LIGHTS {seq} {}\n", pattern.as_str()),
        }
    }

    pub fn to_compact_line_with_session(&self, seq: u32, session_id: &str) -> String {
        let mut line = self.to_compact_line(seq);
        line.pop();
        line.push_str(" session_id=");
        line.push_str(session_id);
        line.push('\n');
        line
    }

    pub fn to_compact_line_with_authority(
        &self,
        seq: u32,
        session_id: &str,
        lease_id: &str,
    ) -> String {
        let mut line = self.to_compact_line_with_session(seq, session_id);
        line.pop();
        line.push_str(" lease_id=");
        line.push_str(lease_id);
        line.push('\n');
        line
    }

    pub fn to_compact_line_with_service_authority(
        &self,
        seq: u32,
        session_id: &str,
        lease_id: &str,
    ) -> String {
        let mut line = self.to_compact_line_with_session(seq, session_id);
        line.pop();
        line.push_str(" service_lease_id=");
        line.push_str(lease_id);
        line.push('\n');
        line
    }
    pub fn to_bridge_json(&self, command_id: u32) -> Result<String> {
        self.to_firmware_json(command_id)
    }
}

fn sample_cockpit_capability_verbs() -> Vec<&'static str> {
    vec![
        "ping",
        "status",
        "get_capabilities",
        "get_events",
        "restart_mpu",
        "restart_create",
        "reset_motherbrain",
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
        "heartbeat_stop",
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
        "zero_imu_orientation",
        "clear_imu_orientation",
        "dock",
        "set_lights",
        "set_mode",
    ]
}

fn optional_cockpit_verbs() -> Vec<&'static str> {
    vec!["reset_motherbrain"]
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CockpitResponse {
    Accepted,
    Rejected { message: String },
    Status(CockpitStatus),
    Capabilities(CockpitCapabilities),
    Events(EventBatch),
    NetworkEndpointRegistered(NetworkEndpointRegistered),
    ControlLeaseGranted(ControlLease),
    ServiceLeaseGranted(ServiceLease),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CommandAck {
    pub accepted: bool,
    pub command_id: u32,
    pub reason: String,
}

fn expect_accepted(response: CockpitResponse) -> Result<()> {
    match response {
        CockpitResponse::Accepted => Ok(()),
        CockpitResponse::Rejected { message } => Err(CockpitError::Rejected {
            command_id: 0,
            reason: message,
        }),
        other => Err(CockpitError::BadResponse(format!("{other:?}"))),
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct BrowserBridgeEnvelope {
    pub command_id: u32,
    pub request: CockpitRequest,
}

#[derive(Debug, Clone)]
pub struct AgentPolicy {
    pub motion_ttl_ms: u32,
    pub heartbeat_timeout_ms: u32,
}

impl Default for AgentPolicy {
    fn default() -> Self {
        Self {
            motion_ttl_ms: 300,
            heartbeat_timeout_ms: 900,
        }
    }
}

pub struct SafeCockpit<C> {
    client: C,
    cursor: EventCursor,
    policy: AgentPolicy,
    contract: Option<CockpitContract>,
}

impl<C: Cockpit> SafeCockpit<C> {
    pub fn new(client: C) -> Self {
        Self::with_policy(client, AgentPolicy::default())
    }

    pub fn with_policy(client: C, policy: AgentPolicy) -> Self {
        let cursor = EventCursor {
            next_seq: client.event_cursor_hint().unwrap_or(0),
        };
        Self {
            client,
            cursor,
            policy,
            contract: None,
        }
    }

    pub fn client_mut(&mut self) -> &mut C {
        &mut self.client
    }

    pub fn replace_client(&mut self, client: C) {
        self.cursor = EventCursor {
            next_seq: client.event_cursor_hint().unwrap_or(0),
        };
        self.client = client;
        self.contract = None;
    }

    pub fn refresh_status(&mut self) -> Result<StatusSummary> {
        let status = self.client.get_status()?.summary();
        if self.cursor.next_seq == 0 {
            if let Some(event_next_seq) = status.event_next_seq {
                self.cursor = EventCursor::from_event_next_seq(event_next_seq);
            }
        }
        Ok(status)
    }

    pub fn resync_event_cursor_from_status(&mut self) -> Result<StatusSummary> {
        let status = self.client.get_status()?.summary();
        if let Some(event_next_seq) = status.event_next_seq {
            self.cursor = EventCursor::from_event_next_seq(event_next_seq);
        }
        Ok(status)
    }

    pub fn poll_events_allowing_history_gap(&mut self) -> Result<EventBatch> {
        let batch = self.client.get_events_since(self.cursor.next_seq)?;
        self.cursor = EventCursor::from_event_next_seq(batch.next_seq);
        Ok(batch)
    }

    /// Consume the next cursor-bounded batch from the brainstem interface.
    /// A reported history gap is an error; callers never silently skip body
    /// events and pretend their view is continuous.
    pub fn poll_events(&mut self) -> Result<EventBatch> {
        self.cursor.poll(&mut self.client)
    }

    pub fn refresh_contract(&mut self) -> Result<&CockpitContract> {
        let capabilities = self.client.get_capabilities()?;
        let contract = CockpitContract::new(capabilities);
        contract.validate_event_vocabulary()?;
        self.contract = Some(contract);
        Ok(self.contract.as_ref().expect("contract was just set"))
    }

    fn ensure_contract(&mut self) -> Result<&CockpitContract> {
        if self.contract.is_none() {
            self.refresh_contract()?;
        }
        Ok(self.contract.as_ref().expect("contract is present"))
    }

    pub fn poll_safety_events(&mut self) -> Result<Vec<SafeStopReason>> {
        let batch = self.cursor.poll(&mut self.client)?;
        Ok(batch
            .events
            .iter()
            .filter_map(SafeStopReason::from_event)
            .collect())
    }

    pub fn pulse_motion(&mut self, linear_mm_s: i16, angular_mrad_s: i16) -> Result<()> {
        let status = self.refresh_status()?;
        if status.estop_latched == Some(true) || status.safety_tripped == Some(true) {
            return Err(CockpitError::Policy(
                "refusing motion while safety is latched".to_owned(),
            ));
        }
        let heartbeat_timeout_ms = self.policy.heartbeat_timeout_ms;
        let motion_ttl_ms = self.policy.motion_ttl_ms;
        let contract = self.ensure_contract()?;
        if !contract.supports("cmd_vel") {
            return Err(CockpitError::Policy(
                "refusing motion because cmd_vel is unsupported".to_owned(),
            ));
        }
        if heartbeat_timeout_ms > 0 && !contract.supports("heartbeat_stop") {
            return Err(CockpitError::Policy(
                "heartbeat policy requires heartbeat_stop capability".to_owned(),
            ));
        }
        let request = CockpitRequest::CmdVel {
            linear_mm_s,
            angular_mrad_s,
            ttl_ms: motion_ttl_ms,
        };
        let request = contract.clamp_motion_request(&request);
        contract.validate_request(&request)?;
        if heartbeat_timeout_ms > 0 {
            let heartbeat = CockpitRequest::HeartbeatStop {
                timeout_ms: heartbeat_timeout_ms,
            };
            let heartbeat = contract.clamp_motion_request(&heartbeat);
            contract.validate_request(&heartbeat)?;
            if let CockpitRequest::HeartbeatStop { timeout_ms } = heartbeat {
                self.client.heartbeat_stop(timeout_ms)?;
            }
        }
        let CockpitRequest::CmdVel {
            linear_mm_s,
            angular_mrad_s,
            ttl_ms,
        } = request
        else {
            unreachable!("request was constructed as cmd_vel")
        };
        self.client.cmd_vel(linear_mm_s, angular_mrad_s, ttl_ms)?;
        let stops = self.poll_safety_events()?;
        if !stops.is_empty() {
            let _ = self.client.stop();
            return Err(CockpitError::Policy(format!(
                "motion stopped by {:?}",
                stops
            )));
        }
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        self.client.stop()?;
        let _ = self.poll_safety_events()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum CockpitEventKind {
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
    WallChanged,
    VirtualWallChanged,
    BatteryLow,
    ChargingStateChanged,
    ButtonsChanged,
    IrChanged,
    HeartbeatExpired,
    EStopLatched,
    EStopCleared,
    ImuFrameReceived,
    ImuFault,
    TiltChanged,
    MotionInconsistencyDetected,
    ImpactDetected,
    SessionOpened,
    SessionReplaced,
    SessionClosed,
    SessionRejected,
    PeerRebootDetected,
    TransportChanged,
    DhcpLeaseChanged,
    DnsRegistrationChanged,
    AuthorityChanged,
    MotherbrainResetRequested,
    MotherbrainResetAsserted,
    MotherbrainResetCompleted,
    MotherbrainResetRefused,
    Error,
    Unknown(String),
}

impl From<&str> for CockpitEventKind {
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
            "wall_changed" => Self::WallChanged,
            "virtual_wall_changed" => Self::VirtualWallChanged,
            "battery_low" => Self::BatteryLow,
            "charging_state_changed" => Self::ChargingStateChanged,
            "buttons_changed" => Self::ButtonsChanged,
            "ir_changed" => Self::IrChanged,
            "heartbeat_expired" => Self::HeartbeatExpired,
            "estop_latched" => Self::EStopLatched,
            "estop_cleared" => Self::EStopCleared,
            "imu_frame_received" => Self::ImuFrameReceived,
            "imu_fault" => Self::ImuFault,
            "tilt_changed" => Self::TiltChanged,
            "motion_inconsistency_detected" => Self::MotionInconsistencyDetected,
            "impact_detected" => Self::ImpactDetected,
            "session_opened" => Self::SessionOpened,
            "session_replaced" => Self::SessionReplaced,
            "session_closed" => Self::SessionClosed,
            "session_rejected" => Self::SessionRejected,
            "peer_reboot_detected" => Self::PeerRebootDetected,
            "transport_changed" => Self::TransportChanged,
            "dhcp_lease_changed" => Self::DhcpLeaseChanged,
            "dns_registration_changed" => Self::DnsRegistrationChanged,
            "authority_changed" => Self::AuthorityChanged,
            "motherbrain_reset_requested" => Self::MotherbrainResetRequested,
            "motherbrain_reset_asserted" => Self::MotherbrainResetAsserted,
            "motherbrain_reset_completed" => Self::MotherbrainResetCompleted,
            "motherbrain_reset_refused" => Self::MotherbrainResetRefused,
            "error" => Self::Error,
            other => Self::Unknown(other.to_owned()),
        }
    }
}

impl CockpitEventKind {
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
            Self::WallChanged => "wall_changed",
            Self::VirtualWallChanged => "virtual_wall_changed",
            Self::BatteryLow => "battery_low",
            Self::ChargingStateChanged => "charging_state_changed",
            Self::ButtonsChanged => "buttons_changed",
            Self::IrChanged => "ir_changed",
            Self::HeartbeatExpired => "heartbeat_expired",
            Self::EStopLatched => "estop_latched",
            Self::EStopCleared => "estop_cleared",
            Self::ImuFrameReceived => "imu_frame_received",
            Self::ImuFault => "imu_fault",
            Self::TiltChanged => "tilt_changed",
            Self::MotionInconsistencyDetected => "motion_inconsistency_detected",
            Self::ImpactDetected => "impact_detected",
            Self::SessionOpened => "session_opened",
            Self::SessionReplaced => "session_replaced",
            Self::SessionClosed => "session_closed",
            Self::SessionRejected => "session_rejected",
            Self::PeerRebootDetected => "peer_reboot_detected",
            Self::TransportChanged => "transport_changed",
            Self::DhcpLeaseChanged => "dhcp_lease_changed",
            Self::DnsRegistrationChanged => "dns_registration_changed",
            Self::AuthorityChanged => "authority_changed",
            Self::MotherbrainResetRequested => "motherbrain_reset_requested",
            Self::MotherbrainResetAsserted => "motherbrain_reset_asserted",
            Self::MotherbrainResetCompleted => "motherbrain_reset_completed",
            Self::MotherbrainResetRefused => "motherbrain_reset_refused",
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
pub struct SimCockpit {
    capabilities: CockpitCapabilities,
    events: Vec<CockpitEvent>,
    next_event_seq: u32,
    event_capacity: usize,
    now_ms: u32,
    next_command_id: u32,
    armed: bool,
    estop_latched: bool,
    safety_tripped: bool,
    bump_left: bool,
    bump_right: bool,
    cliff: bool,
    wheel_drop: bool,
    wall: bool,
    virtual_wall: bool,
    buttons: u8,
    ir_byte: u8,
    charging_state: u8,
    battery_charge_mah: u32,
    battery_capacity_mah: u32,
    odometry_distance_mm: i32,
    odometry_heading_mrad: i32,
    active_cmd_vel: Option<SimTimedAction>,
    heartbeat_stop_at_ms: Option<u32>,
    odometry_reset_count: u32,
    imu_calibration: u8,
    device_id: String,
    boot_id: String,
    active_session: Option<CockpitSession>,
    sessions: Vec<CockpitSession>,
    control_lease: Option<ControlLease>,
    control_lease_expires_at_ms: Option<u32>,
    lease_generation: u32,
    session_serial: u64,
    last_hello: Option<(HandshakeHello, HandshakeResponse)>,
    leases: HashMap<String, NetworkLease>,
    dns_records: HashMap<String, (String, u64, u32)>,
    registration_generation: u32,
    registration_boot_id: Option<String>,
    internal_domain: String,
    operator_debug_allowed: bool,
    recovery_forebrain_device_id: Option<String>,
    allow_unscoped_bench_mode: bool,
    scoped_dispatch: bool,
}

impl SimCockpit {
    pub fn new() -> Self {
        let mut sim = Self {
            capabilities: CockpitCapabilities {
                body_kind: "sim_create_oi".to_owned(),
                drive: "differential".to_owned(),
                verbs: CockpitRequest::capability_verbs()
                    .into_iter()
                    .map(ToOwned::to_owned)
                    .collect(),
                sensors: [
                    "bump",
                    "cliff",
                    "wheel_drop",
                    "wall",
                    "virtual_wall",
                    "ir",
                    "buttons",
                    "battery",
                    "odometry_delta",
                    "imu",
                ]
                .into_iter()
                .map(ToOwned::to_owned)
                .collect(),
                outputs: ["drive", "lights", "song"]
                    .into_iter()
                    .map(ToOwned::to_owned)
                    .collect(),
                safety: [
                    "estop",
                    "heartbeat",
                    "bump",
                    "cliff",
                    "wheel_drop",
                    "tilt",
                    "impact",
                ]
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
                    "bump_changed",
                    "cliff_changed",
                    "wheel_drop_latched",
                    "wheel_drop_cleared",
                    "wall_changed",
                    "virtual_wall_changed",
                    "battery_low",
                    "charging_state_changed",
                    "buttons_changed",
                    "ir_changed",
                    "heartbeat_expired",
                    "estop_latched",
                    "estop_cleared",
                    "imu_frame_received",
                    "imu_fault",
                    "tilt_changed",
                    "motion_inconsistency_detected",
                    "impact_detected",
                ]
                .into_iter()
                .map(ToOwned::to_owned)
                .collect(),
                limits: CockpitLimits {
                    max_linear_mm_s: 500,
                    max_angular_mrad_s: 4_000,
                    min_ttl_ms: 10,
                    max_ttl_ms: 60_000,
                },
            },
            events: Vec::new(),
            next_event_seq: 1,
            event_capacity: DEFAULT_SIM_EVENT_CAPACITY,
            now_ms: 0,
            next_command_id: 1,
            armed: false,
            estop_latched: false,
            safety_tripped: false,
            bump_left: false,
            bump_right: false,
            cliff: false,
            wheel_drop: false,
            wall: false,
            virtual_wall: false,
            buttons: 0,
            ir_byte: 0,
            charging_state: 0,
            battery_charge_mah: 2600,
            battery_capacity_mah: 2600,
            odometry_distance_mm: 0,
            odometry_heading_mrad: 0,
            active_cmd_vel: None,
            heartbeat_stop_at_ms: None,
            odometry_reset_count: 0,
            imu_calibration: 3,
            device_id: "pete-brainstem-sim".into(),
            boot_id: fresh_sim_boot_id(),
            active_session: None,
            sessions: Vec::new(),
            control_lease: None,
            control_lease_expires_at_ms: None,
            lease_generation: 0,
            session_serial: 0,
            last_hello: None,
            leases: HashMap::new(),
            dns_records: HashMap::new(),
            registration_generation: 0,
            registration_boot_id: None,
            internal_domain: DEFAULT_INTERNAL_DOMAIN.into(),
            operator_debug_allowed: false,
            recovery_forebrain_device_id: None,
            allow_unscoped_bench_mode: false,
            scoped_dispatch: false,
        };
        sim.push_event(CockpitEventKind::Boot, 0, 0, 0);
        sim
    }

    pub fn with_event_capacity(mut self, event_capacity: usize) -> Self {
        self.event_capacity = event_capacity.max(1);
        self.enforce_event_capacity();
        self
    }

    pub fn with_capabilities(mut self, capabilities: CockpitCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    pub fn with_identity(
        mut self,
        device_id: impl Into<String>,
        boot_id: impl Into<String>,
    ) -> Self {
        self.device_id = device_id.into();
        self.boot_id = boot_id.into();
        self
    }

    pub fn with_takeover_policy(
        mut self,
        operator_debug_allowed: bool,
        recovery_forebrain_device_id: Option<String>,
    ) -> Self {
        self.operator_debug_allowed = operator_debug_allowed;
        self.recovery_forebrain_device_id = recovery_forebrain_device_id;
        self
    }

    /// Explicit compatibility escape hatch for local bench code that has not
    /// yet migrated to `SessionCockpit`. Never enabled by production
    /// connectors or by the simulator default.
    pub fn with_unscoped_bench_mode(mut self) -> Self {
        self.allow_unscoped_bench_mode = true;
        self
    }

    pub fn reboot(&mut self) {
        self.boot_id = fresh_sim_boot_id();
        self.active_session = None;
        self.sessions.clear();
        self.control_lease = None;
        self.control_lease_expires_at_ms = None;
        self.armed = false;
        self.interrupt_active_motion();
        self.heartbeat_stop_at_ms = None;
        self.last_hello = None;
        self.push_event(CockpitEventKind::Boot, 0, 0, 0);
    }

    pub fn active_session(&self) -> Option<&CockpitSession> {
        self.active_session.as_ref()
    }

    pub fn add_network_lease(&mut self, lease: NetworkLease) {
        let identity = lease.dhcp_client_identifier.clone();
        if let Some(hostname) = lease.requested_hostname.as_deref() {
            if !RESERVED_NETWORK_NAMES.contains(&hostname) {
                let fqdn = format!("{hostname}.{}", self.internal_domain);
                self.dns_records
                    .insert(fqdn, (lease.leased_ip.clone(), lease.lease_expiry, 0));
            }
        }
        self.leases.insert(identity, lease);
    }

    pub fn resolve_internal_name(&mut self, name: &str) -> Option<String> {
        self.expire_network_records();
        let canonical = match name {
            "pete" | "brainstem" => format!("brainstem.{}", self.internal_domain),
            other if !other.contains('.') => format!("{other}.{}", self.internal_domain),
            other => other.to_owned(),
        };
        if canonical == format!("brainstem.{}", self.internal_domain) {
            return Some("192.168.4.1".into());
        }
        self.dns_records
            .get(&canonical)
            .map(|record| record.0.clone())
    }

    fn expire_network_records(&mut self) {
        let now = (self.now_ms / 1_000) as u64;
        self.leases.retain(|_, lease| lease.lease_expiry > now);
        self.dns_records.retain(|_, (_, expiry, _)| *expiry > now);
    }

    fn register_network_endpoint(
        &mut self,
        session: &CockpitSession,
        registration: RegisterNetworkEndpoint,
    ) -> Result<CockpitResponse> {
        if session.local_role != EndpointRole::Motherbrain
            || session.local_purpose != SessionPurpose::Control
            || !self
                .active_session
                .as_ref()
                .is_some_and(|active| active.session_id == session.session_id)
            || registration.hostname != "motherbrain"
        {
            return Err(CockpitError::Policy(
                "reserved motherbrain registration requires the active motherbrain identity".into(),
            ));
        }
        self.expire_network_records();
        let lease = self
            .leases
            .get(&registration.lease_identity)
            .ok_or_else(|| {
                CockpitError::Policy(
                    "network registration does not match an active DHCP lease".into(),
                )
            })?;
        if lease.leased_ip != registration.address {
            return Err(CockpitError::Policy(
                "registered address does not match the DHCP lease".into(),
            ));
        }
        let fqdn = format!("motherbrain.{}", self.internal_domain);
        let duplicate = self.registration_boot_id.as_deref()
            == Some(session.local_boot_id.as_str())
            && self
                .dns_records
                .get(&fqdn)
                .is_some_and(|record| record.0 == registration.address);
        if !duplicate {
            self.registration_generation = self.registration_generation.wrapping_add(1).max(1);
        }
        self.registration_boot_id = Some(session.local_boot_id.clone());
        let ttl = registration
            .ttl_seconds
            .min(
                (lease
                    .lease_expiry
                    .saturating_sub((self.now_ms / 1_000) as u64)) as u32,
            )
            .max(1);
        self.dns_records.insert(
            fqdn.clone(),
            (
                registration.address.clone(),
                (self.now_ms / 1_000) as u64 + ttl as u64,
                self.registration_generation,
            ),
        );
        Ok(CockpitResponse::NetworkEndpointRegistered(
            NetworkEndpointRegistered {
                session_id: session.session_id.clone(),
                fqdn,
                address: registration.address,
                ttl_seconds: ttl,
                registration_generation: self.registration_generation,
            },
        ))
    }

    fn acquire_control_lease(
        &mut self,
        session: &CockpitSession,
        authority: ControlAuthority,
        ttl_ms: u32,
    ) -> Result<CockpitResponse> {
        let allowed = pete_cockpit_protocol::role_can_request_control(
            session.local_role,
            session.local_purpose,
            authority,
        );
        if !allowed {
            return Err(CockpitError::Policy(
                "role is not eligible for requested control authority".into(),
            ));
        }
        if authority == ControlAuthority::OperatorDebug && !self.operator_debug_allowed {
            return Err(CockpitError::Policy(
                "operator debug policy is disabled".into(),
            ));
        }
        if authority == ControlAuthority::ForebrainRecovery
            && self.recovery_forebrain_device_id.as_deref()
                != Some(session.local_device_id.as_str())
        {
            return Err(CockpitError::Policy(
                "forebrain is not the configured recovery identity".into(),
            ));
        }
        let lease_alive = self
            .control_lease_expires_at_ms
            .is_some_and(|deadline| !time_reached(self.now_ms, deadline));
        let continuing_owner = lease_alive
            && self.control_lease.as_ref().is_some_and(|lease| {
                lease.session_id == session.session_id && lease.authority == authority
            });
        if authority == ControlAuthority::ForebrainRecovery && lease_alive {
            return Err(CockpitError::Policy(
                "current controller lease has not expired".into(),
            ));
        }
        // A renewal by the same live owner atomically replaces only the lease.
        // A true ownership transition stops and clears inherited state.
        if !continuing_owner {
            self.interrupt_active_motion();
            self.heartbeat_stop_at_ms = None;
            self.armed = authority == ControlAuthority::Motherbrain;
        }
        self.control_lease = None;
        self.control_lease_expires_at_ms = None;
        self.lease_generation = self.lease_generation.wrapping_add(1).max(1);
        let lease = ControlLease {
            lease_id: format!(
                "lease-{}-{}",
                self.lease_generation,
                uuid::Uuid::new_v4().simple()
            ),
            session_id: session.session_id.clone(),
            owner_role: session.local_role,
            authority,
            ttl_ms: ttl_ms.clamp(250, 60_000),
            generation: self.lease_generation,
        };
        self.control_lease = Some(lease.clone());
        self.control_lease_expires_at_ms = Some(self.now_ms.wrapping_add(lease.ttl_ms));
        Ok(CockpitResponse::ControlLeaseGranted(lease))
    }

    fn require_scoped_dispatch(&self) -> Result<()> {
        if self.allow_unscoped_bench_mode || self.scoped_dispatch {
            Ok(())
        } else {
            Err(CockpitError::SessionRequired)
        }
    }

    pub fn advance_ms(&mut self, ms: u32) {
        self.now_ms = self.now_ms.wrapping_add(ms);
        if self
            .control_lease_expires_at_ms
            .is_some_and(|deadline| time_reached(self.now_ms, deadline))
        {
            self.interrupt_active_motion();
            self.heartbeat_stop_at_ms = None;
            self.armed = false;
            self.control_lease = None;
            self.control_lease_expires_at_ms = None;
        }
        self.complete_due_cmd_vel();
        self.expire_heartbeat_if_due();
    }

    pub fn trip_safety(&mut self) {
        if self.safety_tripped {
            return;
        }
        self.safety_tripped = true;
        self.interrupt_active_motion();
        self.push_event(CockpitEventKind::SafetyTripped, 1, 0, 0);
        self.push_event(CockpitEventKind::MotionStopped, 0, 0, 0);
    }

    pub fn set_bump(&mut self, left: bool, right: bool) {
        if self.bump_left == left && self.bump_right == right {
            return;
        }
        self.bump_left = left;
        self.bump_right = right;
        self.push_event(CockpitEventKind::BumpChanged, (left || right) as u32, 0, 0);
    }

    pub fn set_cliff(&mut self, active: bool) {
        if self.cliff == active {
            return;
        }
        self.cliff = active;
        self.push_event(CockpitEventKind::CliffChanged, active as u32, 0, 0);
    }

    pub fn set_wheel_drop(&mut self, active: bool) {
        if self.wheel_drop == active {
            return;
        }
        self.wheel_drop = active;
        if active {
            self.safety_tripped = true;
            self.interrupt_active_motion();
            self.push_event(CockpitEventKind::WheelDropLatched, 1, 0, 0);
            self.push_event(CockpitEventKind::SafetyTripped, 3, 0, 0);
            self.push_event(CockpitEventKind::MotionStopped, 0, 0, 0);
        } else {
            self.safety_tripped = self.estop_latched || self.cliff;
            self.push_event(CockpitEventKind::WheelDropCleared, 0, 0, 0);
            self.push_event(CockpitEventKind::SafetyCleared, 3, 0, 0);
        }
    }

    pub fn set_wall(&mut self, active: bool) {
        if self.wall == active {
            return;
        }
        self.wall = active;
        self.push_event(CockpitEventKind::WallChanged, active as u32, 0, 0);
    }

    pub fn set_virtual_wall(&mut self, active: bool) {
        if self.virtual_wall == active {
            return;
        }
        self.virtual_wall = active;
        self.push_event(CockpitEventKind::VirtualWallChanged, active as u32, 0, 0);
    }

    pub fn set_battery(&mut self, charge_mah: u32, capacity_mah: u32) {
        self.battery_charge_mah = charge_mah;
        self.battery_capacity_mah = capacity_mah;
        if self.battery_percent().is_some_and(|percent| percent <= 20) {
            self.push_event(
                CockpitEventKind::BatteryLow,
                self.battery_percent().unwrap_or(0) as u32,
                0,
                0,
            );
        }
    }

    pub fn set_charging_state(&mut self, state: u8) {
        if self.charging_state == state {
            return;
        }
        self.charging_state = state;
        self.push_event(CockpitEventKind::ChargingStateChanged, state as u32, 0, 0);
    }

    pub fn set_buttons(&mut self, buttons: u8) {
        if self.buttons == buttons {
            return;
        }
        self.buttons = buttons;
        self.push_event(CockpitEventKind::ButtonsChanged, buttons as u32, 0, 0);
    }

    pub fn set_ir_byte(&mut self, ir_byte: u8) {
        if self.ir_byte == ir_byte {
            return;
        }
        self.ir_byte = ir_byte;
        self.push_event(CockpitEventKind::IrChanged, ir_byte as u32, 0, 0);
    }

    pub fn odometry_reset_count(&self) -> u32 {
        self.odometry_reset_count
    }

    fn battery_percent(&self) -> Option<u8> {
        battery_percent(
            Some(self.battery_charge_mah),
            Some(self.battery_capacity_mah),
        )
    }

    fn accept_command(&mut self) -> u32 {
        let id = self.next_command_id;
        self.next_command_id = self.next_command_id.wrapping_add(1).max(1);
        self.push_event(CockpitEventKind::CommandAccepted, id, 0, 0);
        self.push_event(CockpitEventKind::CommandStarted, id, 0, 0);
        id
    }

    fn complete_command(&mut self, id: u32) {
        self.push_event(CockpitEventKind::CommandCompleted, id, 0, 0);
    }

    fn push_event(&mut self, kind: CockpitEventKind, a: u32, b: u32, c: u32) {
        let seq = self.next_event_seq;
        self.next_event_seq = self.next_event_seq.wrapping_add(1).max(1);
        self.events.push(CockpitEvent { seq, kind, a, b, c });
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
                CockpitEventKind::CommandInterrupted,
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
        self.push_event(CockpitEventKind::MotionStopped, 0, 0, 0);
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
        self.push_event(CockpitEventKind::HeartbeatExpired, 0, 0, 0);
        self.push_event(CockpitEventKind::SafetyTripped, 5, 0, 0);
        self.push_event(CockpitEventKind::MotionStopped, 0, 0, 0);
    }

    fn oldest_seq(&self) -> u32 {
        self.events
            .first()
            .map(|event| event.seq)
            .unwrap_or(self.next_event_seq)
    }
}

impl Default for SimCockpit {
    fn default() -> Self {
        Self::new()
    }
}

impl Cockpit for SimCockpit {
    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        if !self.allow_unscoped_bench_mode
            && !self.scoped_dispatch
            && !matches!(
                request.authorization_class(),
                AuthorizationClass::ReadOnly | AuthorizationClass::Emergency
            )
        {
            return Err(CockpitError::SessionRequired);
        }
        match request {
            CockpitRequest::GetStatus => Ok(CockpitResponse::Status(self.get_status()?)),
            CockpitRequest::GetCapabilities => {
                Ok(CockpitResponse::Capabilities(self.get_capabilities()?))
            }
            CockpitRequest::GetEvents { since_seq } => {
                Ok(CockpitResponse::Events(self.get_events_since(since_seq)?))
            }
            CockpitRequest::RegisterNetworkEndpoint(_) => Err(CockpitError::SessionRequired),
            CockpitRequest::Ping => {
                let id = self.accept_command();
                self.complete_command(id);
                Ok(CockpitResponse::Accepted)
            }
            CockpitRequest::Bootsel => {
                let id = self.accept_command();
                self.complete_command(id);
                Ok(CockpitResponse::Accepted)
            }
            CockpitRequest::Arm => self.arm().map(|()| CockpitResponse::Accepted),
            CockpitRequest::Disarm => self.disarm().map(|()| CockpitResponse::Accepted),
            CockpitRequest::Stop => self.stop().map(|()| CockpitResponse::Accepted),
            CockpitRequest::EStop => self.estop().map(|()| CockpitResponse::Accepted),
            CockpitRequest::ClearEStop => self.clear_estop().map(|()| CockpitResponse::Accepted),
            CockpitRequest::CmdVel {
                linear_mm_s,
                angular_mrad_s,
                ttl_ms,
            } => self
                .cmd_vel(linear_mm_s, angular_mrad_s, ttl_ms)
                .map(|()| CockpitResponse::Accepted),
            CockpitRequest::HeartbeatStop { timeout_ms } => self
                .heartbeat_stop(timeout_ms)
                .map(|()| CockpitResponse::Accepted),
            CockpitRequest::StreamSensors {
                enabled,
                packet_id,
                period_ms,
            } => self
                .stream_sensors(enabled, packet_id, period_ms)
                .map(|()| CockpitResponse::Accepted),
            CockpitRequest::ResetOdometry => {
                self.reset_odometry().map(|()| CockpitResponse::Accepted)
            }
            CockpitRequest::ZeroImuOrientation => self
                .zero_imu_orientation()
                .map(|()| CockpitResponse::Accepted),
            CockpitRequest::ClearImuOrientation => self
                .clear_imu_orientation()
                .map(|()| CockpitResponse::Accepted),
            _ => {
                let id = self.accept_command();
                self.complete_command(id);
                Ok(CockpitResponse::Accepted)
            }
        }
    }

    fn handshake(&mut self, hello: HandshakeHello) -> Result<HandshakeOutcome> {
        if let Some((cached_hello, cached_response)) = &self.last_hello {
            if cached_hello == &hello {
                return HandshakeOutcome::validate(&hello, cached_response.clone());
            }
        }
        self.session_serial = self.session_serial.wrapping_add(1).max(1);
        let establishes_primary = hello.role == EndpointRole::Motherbrain
            && hello.session_purpose == SessionPurpose::Control;
        let had_session = establishes_primary && self.active_session.is_some();
        let peer_reboot = self.active_session.as_ref().is_some_and(|session| {
            session.local_device_id == hello.device_id && session.local_boot_id != hello.boot_id
        });
        let response = negotiate(
            &hello,
            &self.device_id,
            &self.boot_id,
            self.capabilities.clone(),
            SafetySnapshot {
                armed: if establishes_primary {
                    false
                } else {
                    self.armed
                },
                estop_latched: self.estop_latched,
                safety_tripped: self.safety_tripped,
                active_motion: if establishes_primary {
                    false
                } else {
                    self.active_cmd_vel.is_some()
                },
                runtime_state: if self.active_cmd_vel.is_some() && !establishes_primary {
                    "moving".into()
                } else {
                    "idle".into()
                },
            },
            self.next_event_seq,
            SoftwareInfo {
                software_name: "pete-brainstem-sim".into(),
                software_version: env!("CARGO_PKG_VERSION").into(),
                build_id: "sim".into(),
            },
            self.session_serial,
        );
        if matches!(response, HandshakeResponse::Reject(_)) {
            self.push_event(CockpitEventKind::SessionRejected, 0, 0, 0);
        }
        let outcome = HandshakeOutcome::validate(&hello, response.clone())?;
        // Installing a valid session is a synchronous safety operation: old
        // motion and leases are revoked before WELCOME becomes observable.
        // Rejected/malformed hellos never mutate command state.
        if establishes_primary {
            self.sessions
                .retain(|session| session.local_role != EndpointRole::Motherbrain);
            self.dns_records
                .remove(&format!("motherbrain.{}", self.internal_domain));
            self.registration_boot_id = None;
            self.interrupt_active_motion();
            self.heartbeat_stop_at_ms = None;
            self.armed = false;
            self.control_lease = None;
            self.control_lease_expires_at_ms = None;
            self.active_session = Some(outcome.session.clone());
            self.push_event(
                if had_session {
                    CockpitEventKind::SessionReplaced
                } else {
                    CockpitEventKind::SessionOpened
                },
                0,
                0,
                0,
            );
        } else {
            self.push_event(CockpitEventKind::SessionOpened, 0, 0, 0);
        }
        self.sessions
            .retain(|session| session.session_id != outcome.session.session_id);
        self.sessions.push(outcome.session.clone());
        if peer_reboot {
            self.push_event(CockpitEventKind::PeerRebootDetected, 0, 0, 0);
        }
        self.last_hello = Some((hello, response));
        Ok(outcome)
    }

    fn execute_in_session(
        &mut self,
        session: &CockpitSession,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        if request.requires_session() {
            let valid = self
                .sessions
                .iter()
                .any(|active| active.session_id == session.session_id);
            if !valid {
                return Err(CockpitError::InvalidSession {
                    session_id: session.session_id.clone(),
                });
            }
        }
        if let CockpitRequest::RegisterNetworkEndpoint(registration) = request {
            return self.register_network_endpoint(session, registration);
        }
        if let CockpitRequest::AcquireControlLease { authority, ttl_ms } = request {
            return self.acquire_control_lease(session, authority, ttl_ms);
        }
        if matches!(request, CockpitRequest::AcquireServiceLease { .. }) {
            return Err(CockpitError::Policy(
                "service mode is disabled in this simulator".into(),
            ));
        }
        if request.authorization_class() == AuthorizationClass::ServiceLease {
            return Err(CockpitError::Policy(
                "request requires a separate service lease".into(),
            ));
        }
        if request.requires_control_authority() {
            let valid = self
                .control_lease
                .as_ref()
                .is_some_and(|lease| lease.session_id == session.session_id)
                && self
                    .control_lease_expires_at_ms
                    .is_some_and(|deadline| !time_reached(self.now_ms, deadline));
            if !valid {
                return Err(CockpitError::Policy(
                    "request requires the active control lease".into(),
                ));
            }
            if let CockpitRequest::HeartbeatStop { timeout_ms } = &request {
                self.control_lease_expires_at_ms =
                    Some(self.now_ms.wrapping_add((*timeout_ms).clamp(250, 60_000)));
            }
        }
        self.scoped_dispatch = true;
        let result = self.execute(request);
        self.scoped_dispatch = false;
        result
    }

    fn execute_with_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ControlLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        let valid = self.control_lease.as_ref().is_some_and(|active| {
            active.lease_id == lease.lease_id && active.session_id == session.session_id
        });
        if !valid {
            return Err(CockpitError::Policy(
                "unknown or replaced control lease".into(),
            ));
        }
        self.execute_in_session(session, request)
    }

    fn execute_with_service_lease(
        &mut self,
        _session: &CockpitSession,
        _lease: &ServiceLease,
        _request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        Err(CockpitError::Policy(
            "service mode is disabled in this simulator".into(),
        ))
    }

    fn get_status(&mut self) -> Result<CockpitStatus> {
        self.complete_due_cmd_vel();
        self.expire_heartbeat_if_due();
        Ok(CockpitStatus {
            raw: format!(
                "OK 0 STATUS sim=true now_ms={} armed={} estop={} safety_tripped={} active_cmd_vel={} bump_left={} bump_right={} cliff_left={} cliff_front_left={} cliff_front_right={} cliff_right={} wheel_drop={} wall={} virtual_wall={} ir_byte={} buttons={} charging_state={} charge_mah={} capacity_mah={} voltage_mv={} current_ma={} odometry_resets={} odometry_distance_mm={} odometry_heading_mrad={} imu_present=2 imu_health=1 imu_age_ms=0 imu_poll_ms=20 imu_yaw_mrad=0 imu_yaw_rate_mrad_s=0 imu_accel_mag_mm_s2=9807 imu_tilt_mrad=0 imu_roughness_mm_s2=0 imu_impact_mm_s2=0 imu_motion_consistency=1 imu_calibration={}",
                self.now_ms,
                self.armed,
                self.estop_latched,
                self.safety_tripped,
                self.active_cmd_vel.is_some(),
                self.bump_left,
                self.bump_right,
                self.cliff,
                self.cliff,
                self.cliff,
                self.cliff,
                self.wheel_drop,
                self.wall,
                self.virtual_wall,
                self.ir_byte,
                self.buttons,
                self.charging_state,
                self.battery_charge_mah,
                self.battery_capacity_mah,
                if self.battery_capacity_mah == 0 { 0 } else { 14_400 },
                0,
                self.odometry_reset_count,
                self.odometry_distance_mm,
                self.odometry_heading_mrad,
                self.imu_calibration
            ),
        })
    }

    fn get_capabilities(&mut self) -> Result<CockpitCapabilities> {
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
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        self.armed = true;
        self.complete_command(id);
        Ok(())
    }

    fn disarm(&mut self) -> Result<()> {
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        self.interrupt_active_motion();
        self.armed = false;
        self.complete_command(id);
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        let id = self.accept_command();
        self.interrupt_active_motion();
        self.push_event(CockpitEventKind::MotionStopped, 0, 0, 0);
        self.complete_command(id);
        Ok(())
    }

    fn estop(&mut self) -> Result<()> {
        let id = self.accept_command();
        self.interrupt_active_motion();
        self.estop_latched = true;
        self.safety_tripped = true;
        self.push_event(CockpitEventKind::EStopLatched, 1, 0, 0);
        self.push_event(CockpitEventKind::SafetyTripped, 4, 0, 0);
        self.push_event(CockpitEventKind::MotionStopped, 0, 0, 0);
        self.complete_command(id);
        Ok(())
    }

    fn clear_estop(&mut self) -> Result<()> {
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        self.estop_latched = false;
        self.safety_tripped = false;
        self.push_event(CockpitEventKind::EStopCleared, 0, 0, 0);
        self.push_event(CockpitEventKind::SafetyCleared, 4, 0, 0);
        self.complete_command(id);
        Ok(())
    }

    fn cmd_vel(&mut self, linear_mm_s: i16, angular_mrad_s: i16, ttl_ms: u32) -> Result<()> {
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        if self.estop_latched || self.safety_tripped {
            self.push_event(CockpitEventKind::CommandRejected, id, 0, 0);
            return Ok(());
        }
        self.interrupt_active_motion();
        self.push_event(
            CockpitEventKind::MotionRequested,
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
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        self.heartbeat_stop_at_ms = Some(self.now_ms.wrapping_add(timeout_ms.max(1)));
        self.complete_command(id);
        Ok(())
    }

    fn stream_sensors(&mut self, _enabled: bool, _packet_id: u8, _period_ms: u32) -> Result<()> {
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        self.complete_command(id);
        Ok(())
    }

    fn reset_odometry(&mut self) -> Result<()> {
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        self.odometry_reset_count = self.odometry_reset_count.saturating_add(1);
        self.odometry_distance_mm = 0;
        self.odometry_heading_mrad = 0;
        self.complete_command(id);
        Ok(())
    }

    fn zero_imu_orientation(&mut self) -> Result<()> {
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        self.imu_calibration = 3;
        self.complete_command(id);
        Ok(())
    }

    fn clear_imu_orientation(&mut self) -> Result<()> {
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        self.imu_calibration = 0;
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

    pub fn poll<C: Cockpit>(&mut self, client: &mut C) -> Result<EventBatch> {
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

impl EventCursor {
    pub fn from_event_next_seq(next_seq: u32) -> Self {
        Self {
            next_seq: next_seq.saturating_sub(1),
        }
    }
}

/// A transport-neutral, validated command session. Construction is only
/// possible through `establish_session`, so capabilities and safety state came
/// from the live WELCOME rather than a cache.
pub struct SessionCockpit<C> {
    connector: C,
    outcome: HandshakeOutcome,
    cursor: EventCursor,
    control_lease: Option<ControlLease>,
    service_lease: Option<ServiceLease>,
}

/// Production control boundary for a motherbrain-owned brainstem session.
///
/// Possession is the live `Motherbrain` lease itself. There is deliberately no
/// second local "armed" state and no Create OI surface here: callers can send
/// bounded body intents, STOP/DISARM, and inspect brainstem telemetry only.
pub struct MotherbrainPossession<C: Cockpit> {
    session: SessionCockpit<C>,
    lease_acquired_at: Instant,
    lease_ttl_ms: u32,
    renew_margin_ms: u32,
    motion_ttl_ms: u32,
    heartbeat_timeout_ms: u32,
    max_linear_mm_s: i16,
    max_angular_mrad_s: i16,
    motor_gate_open: bool,
    refusal_reason: Option<String>,
    last_applied_at: Option<Instant>,
    last_applied_command: Option<MotorCommand>,
    last_status: Option<StatusSummary>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PossessionSnapshot {
    pub brainstem_device_id: String,
    pub brainstem_boot_id: String,
    pub session_id: String,
    pub lease_id: String,
    pub lease_generation: u32,
    pub lease_remaining_ms: u32,
    pub possessed: bool,
    pub moving: Option<bool>,
    pub brainstem_armed: Option<bool>,
    pub body_health: Option<String>,
    pub uart_health: Option<String>,
    pub safety_tripped: Option<bool>,
    pub estop_latched: Option<bool>,
    pub refusal_reason: Option<String>,
    pub last_applied_command_age_ms: Option<u64>,
    pub last_applied_command: Option<MotorCommand>,
}

impl<C: Cockpit> MotherbrainPossession<C> {
    pub fn acquire(mut session: SessionCockpit<C>, lease_ttl_ms: u32) -> Result<Self> {
        let lease_ttl_ms = lease_ttl_ms.max(1_000);
        session.acquire_control(ControlAuthority::Motherbrain, lease_ttl_ms)?;
        // A newly acquired authority always begins stopped. Do not select or
        // otherwise expose Create OI modes from the motherbrain.
        expect_accepted(session.execute(CockpitRequest::Stop)?)?;
        let heartbeat_timeout_ms = 750;
        Ok(Self {
            session,
            lease_acquired_at: Instant::now(),
            lease_ttl_ms,
            renew_margin_ms: (lease_ttl_ms / 3).max(500),
            motion_ttl_ms: 300,
            heartbeat_timeout_ms,
            max_linear_mm_s: 50,
            max_angular_mrad_s: 500,
            motor_gate_open: true,
            refusal_reason: None,
            last_applied_at: None,
            last_applied_command: None,
            last_status: None,
        })
    }

    pub fn with_limits(mut self, linear_mm_s: i16, angular_mrad_s: i16) -> Self {
        self.max_linear_mm_s = linear_mm_s.abs().min(50);
        self.max_angular_mrad_s = angular_mrad_s.abs().min(500);
        self
    }

    pub fn snapshot(&self) -> PossessionSnapshot {
        let lease = self.session.control_lease.as_ref();
        let elapsed = self.lease_acquired_at.elapsed().as_millis() as u64;
        PossessionSnapshot {
            brainstem_device_id: self.session.session().peer_device_id.clone(),
            brainstem_boot_id: self.session.session().peer_boot_id.clone(),
            session_id: self.session.session().session_id.clone(),
            lease_id: lease
                .map(|lease| lease.lease_id.clone())
                .unwrap_or_default(),
            lease_generation: lease.map(|lease| lease.generation).unwrap_or_default(),
            lease_remaining_ms: u64::from(self.lease_ttl_ms)
                .saturating_sub(elapsed)
                .min(u64::from(u32::MAX)) as u32,
            possessed: self.motor_gate_open && lease.is_some(),
            moving: self
                .last_status
                .as_ref()
                .and_then(|status| status.active_motion),
            brainstem_armed: self.last_status.as_ref().and_then(|status| status.armed),
            body_health: self
                .last_status
                .as_ref()
                .and_then(|status| status.runtime_state.clone()),
            uart_health: self.last_status.as_ref().and_then(|status| {
                value_for(&status.raw, "create_uart_health")
                    .or_else(|| value_for(&status.raw, "uart_health"))
                    .map(ToOwned::to_owned)
            }),
            safety_tripped: self
                .last_status
                .as_ref()
                .and_then(|status| status.safety_tripped),
            estop_latched: self
                .last_status
                .as_ref()
                .and_then(|status| status.estop_latched),
            refusal_reason: self.refusal_reason.clone(),
            last_applied_command_age_ms: self
                .last_applied_at
                .map(|instant| instant.elapsed().as_millis().min(u128::from(u64::MAX)) as u64),
            last_applied_command: self.last_applied_command,
        }
    }

    pub fn maintain(&mut self) -> Result<()> {
        if !self.motor_gate_open {
            return Err(CockpitError::Policy(
                self.refusal_reason
                    .clone()
                    .unwrap_or_else(|| "not possessed".into()),
            ));
        }
        let renew_at = self
            .lease_ttl_ms
            .saturating_sub(self.renew_margin_ms)
            .min(POSSESSION_LEASE_RENEW_INTERVAL_MS);
        if self.lease_acquired_at.elapsed() < Duration::from_millis(u64::from(renew_at)) {
            return Ok(());
        }
        self.renew_control_lease()
    }

    fn renew_control_lease(&mut self) -> Result<()> {
        if let Err(error) = self
            .session
            .acquire_control(ControlAuthority::Motherbrain, self.lease_ttl_ms)
        {
            self.close_gate(format!("control lease renewal failed: {error}"));
            return Err(error);
        }
        self.lease_acquired_at = Instant::now();
        Ok(())
    }

    fn close_gate(&mut self, reason: String) {
        self.motor_gate_open = false;
        self.refusal_reason = Some(reason);
    }

    fn execute_with_busy_retry(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        let mut retried_after_lease_renewal = false;
        for attempt in 0..POSSESSION_BUSY_RETRY_ATTEMPTS {
            match self.session.execute(request.clone()) {
                Err(CockpitError::Rejected { reason, .. })
                    if reason == "busy" && attempt + 1 < POSSESSION_BUSY_RETRY_ATTEMPTS =>
                {
                    std::thread::sleep(POSSESSION_BUSY_RETRY_DELAY);
                }
                Err(error)
                    if request.authorization_class() == AuthorizationClass::ControlLease
                        && !retried_after_lease_renewal
                        && is_control_lease_rejection(&error) =>
                {
                    self.renew_control_lease()?;
                    retried_after_lease_renewal = true;
                }
                result => return result,
            }
        }
        unreachable!("bounded busy retry always returns on its final attempt")
    }

    pub fn exorcize(&mut self) -> Result<()> {
        let stop = self
            .execute_with_busy_retry(CockpitRequest::Stop)
            .and_then(expect_accepted);
        if let Err(error) = stop {
            self.close_gate("possession exorcized".into());
            return Err(error);
        }
        self.close_gate("possession exorcized".into());
        Ok(())
    }

    fn execute_scoped(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        if request.authorization_class() != AuthorizationClass::Emergency {
            self.maintain()?;
        }
        if matches!(request, CockpitRequest::SetMode { .. }) {
            return Err(CockpitError::Policy(
                "Create OI is private to the brainstem".into(),
            ));
        }
        if matches!(request, CockpitRequest::Arm) {
            return Err(CockpitError::Policy(
                "the motherbrain lease is the possession gate; no second arm layer exists".into(),
            ));
        }
        if matches!(request, CockpitRequest::PowerState { .. }) {
            return Err(CockpitError::Policy(
                "body power control is outside production possession".into(),
            ));
        }
        let request = match request {
            CockpitRequest::CmdVel {
                linear_mm_s,
                angular_mrad_s,
                ..
            } => {
                expect_accepted(self.execute_with_busy_retry(CockpitRequest::HeartbeatStop {
                    timeout_ms: self.heartbeat_timeout_ms,
                })?)?;
                CockpitRequest::CmdVel {
                    linear_mm_s: linear_mm_s.clamp(-self.max_linear_mm_s, self.max_linear_mm_s),
                    angular_mrad_s: angular_mrad_s
                        .clamp(-self.max_angular_mrad_s, self.max_angular_mrad_s),
                    ttl_ms: self.motion_ttl_ms,
                }
            }
            other => other,
        };
        let response = self.execute_with_busy_retry(request.clone());
        match response {
            Ok(response) => {
                match request {
                    CockpitRequest::CmdVel {
                        linear_mm_s,
                        angular_mrad_s,
                        ..
                    } => {
                        self.last_applied_at = Some(Instant::now());
                        self.last_applied_command = Some(MotorCommand {
                            forward: mm_s_to_meters_per_second(linear_mm_s),
                            turn: mrad_s_to_radians_per_second(angular_mrad_s),
                        });
                    }
                    CockpitRequest::Stop => {
                        self.last_applied_at = Some(Instant::now());
                        self.last_applied_command = Some(MotorCommand::stop());
                    }
                    _ => {}
                }
                Ok(response)
            }
            Err(error) => {
                if request.authorization_class() == AuthorizationClass::ControlLease {
                    self.close_gate(format!("scoped command failed: {error}"));
                }
                Err(error)
            }
        }
    }
}

fn is_control_lease_rejection(error: &CockpitError) -> bool {
    matches!(
        error,
        CockpitError::Rejected { reason, .. }
            if reason.contains("invalid_control_lease")
                || reason.contains("control_lease_required")
    )
}

impl<C: Cockpit> Drop for MotherbrainPossession<C> {
    fn drop(&mut self) {
        let _ = self.execute_with_busy_retry(CockpitRequest::Stop);
    }
}

impl<C: Cockpit> Cockpit for MotherbrainPossession<C> {
    fn possession_snapshot(&self) -> Option<PossessionSnapshot> {
        Some(self.snapshot())
    }

    fn event_cursor_hint(&self) -> Option<u32> {
        Some(self.session.cursor.next_seq())
    }

    fn exorcize(&mut self) -> Result<()> {
        MotherbrainPossession::exorcize(self)
    }

    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        match request.authorization_class() {
            AuthorizationClass::ReadOnly => {
                if self.motor_gate_open {
                    self.maintain()?;
                }
                let response = match self.session.execute(request) {
                    Ok(response) => response,
                    Err(error) => {
                        self.close_gate(format!("brainstem status/event failure: {error}"));
                        return Err(error);
                    }
                };
                if let CockpitResponse::Status(status) = &response {
                    let summary = status.summary();
                    self.last_status = Some(summary.clone());
                    if summary.estop_latched == Some(true) || summary.safety_tripped == Some(true) {
                        self.close_gate("brainstem safety refusal".into());
                    }
                }
                Ok(response)
            }
            AuthorizationClass::Emergency
            | AuthorizationClass::Session
            | AuthorizationClass::ControlLease => self.execute_scoped(request),
            AuthorizationClass::ServiceLease => Err(CockpitError::Policy(
                "service operations are outside motherbrain possession".into(),
            )),
        }
    }

    fn handshake(&mut self, _hello: HandshakeHello) -> Result<HandshakeOutcome> {
        Err(CockpitError::Policy(
            "possession session is already established".into(),
        ))
    }

    fn execute_in_session(
        &mut self,
        session: &CockpitSession,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        if session.session_id != self.session.session().session_id {
            return Err(CockpitError::Policy("session replacement detected".into()));
        }
        self.execute(request)
    }

    fn execute_with_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ControlLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        let current = self.session.control_lease.as_ref();
        if session.session_id != self.session.session().session_id
            || current.map(|value| (&value.lease_id, value.generation))
                != Some((&lease.lease_id, lease.generation))
        {
            return Err(CockpitError::Policy("superseded possession lease".into()));
        }
        self.execute(request)
    }

    fn execute_with_service_lease(
        &mut self,
        _session: &CockpitSession,
        _lease: &ServiceLease,
        _request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        Err(CockpitError::Policy(
            "service operations are outside motherbrain possession".into(),
        ))
    }
}

impl<C: Cockpit> SessionCockpit<C> {
    pub fn session(&self) -> &CockpitSession {
        &self.outcome.session
    }
    pub fn contract(&self) -> &CockpitContract {
        &self.outcome.contract
    }
    pub fn outcome(&self) -> &HandshakeOutcome {
        &self.outcome
    }
    pub fn connector_mut(&mut self) -> &mut C {
        &mut self.connector
    }

    pub fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        self.outcome.contract.validate_request(&request)?;
        match request.authorization_class() {
            AuthorizationClass::ControlLease => {
                let lease = self.control_lease.as_ref().ok_or_else(|| {
                    CockpitError::Policy("no control lease has been acquired".into())
                })?;
                self.connector
                    .execute_with_lease(&self.outcome.session, lease, request)
            }
            AuthorizationClass::ServiceLease => {
                let lease = self.service_lease.as_ref().ok_or_else(|| {
                    CockpitError::Policy("no service lease has been acquired".into())
                })?;
                if request.required_service_scope() != Some(lease.scope) {
                    return Err(CockpitError::Policy("service_scope_denied".into()));
                }
                self.connector
                    .execute_with_service_lease(&self.outcome.session, lease, request)
            }
            _ => self
                .connector
                .execute_in_session(&self.outcome.session, request),
        }
    }

    pub fn acquire_service(&mut self, scope: ServiceScope, ttl_ms: u32) -> Result<&ServiceLease> {
        let response = self.connector.execute_in_session(
            &self.outcome.session,
            CockpitRequest::AcquireServiceLease { scope, ttl_ms },
        )?;
        let CockpitResponse::ServiceLeaseGranted(lease) = response else {
            return Err(CockpitError::BadResponse(format!("{response:?}")));
        };
        self.control_lease = None;
        self.service_lease = Some(lease);
        Ok(self.service_lease.as_ref().expect("lease was installed"))
    }

    pub fn control(&mut self) -> Result<ControlCockpit<'_, C>> {
        if self.control_lease.is_none() {
            return Err(CockpitError::Policy(
                "no control lease has been acquired".into(),
            ));
        }
        Ok(ControlCockpit { session: self })
    }

    pub fn service(&mut self) -> Result<ServiceCockpit<'_, C>> {
        if self.service_lease.is_none() {
            return Err(CockpitError::Policy(
                "no service lease has been acquired".into(),
            ));
        }
        Ok(ServiceCockpit { session: self })
    }

    pub fn acquire_control(
        &mut self,
        authority: ControlAuthority,
        ttl_ms: u32,
    ) -> Result<&ControlLease> {
        let response = self.connector.execute_in_session(
            &self.outcome.session,
            CockpitRequest::AcquireControlLease { authority, ttl_ms },
        )?;
        let CockpitResponse::ControlLeaseGranted(lease) = response else {
            return Err(CockpitError::BadResponse(format!("{response:?}")));
        };
        self.service_lease = None;
        self.control_lease = Some(lease);
        Ok(self.control_lease.as_ref().expect("lease was installed"))
    }

    pub fn read_only(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        if request.requires_session() {
            return Err(CockpitError::Policy(
                "read_only accepts recovery requests only".into(),
            ));
        }
        self.connector.execute(request)
    }

    pub fn poll_events(&mut self) -> Result<EventBatch> {
        self.cursor.poll(&mut self.connector)
    }

    pub fn into_parts(self) -> (C, HandshakeOutcome) {
        (self.connector, self.outcome)
    }
}

/// Narrow handle exposing commands authorized by an installed control lease.
pub struct ControlCockpit<'a, C> {
    session: &'a mut SessionCockpit<C>,
}

impl<C: Cockpit> ControlCockpit<'_, C> {
    pub fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        if !matches!(
            request.authorization_class(),
            AuthorizationClass::ControlLease
                | AuthorizationClass::Emergency
                | AuthorizationClass::Session
        ) || matches!(
            request,
            CockpitRequest::AcquireControlLease { .. }
                | CockpitRequest::AcquireServiceLease { .. }
                | CockpitRequest::RegisterNetworkEndpoint(_)
        ) {
            return Err(CockpitError::Policy(
                "request is outside control-lease authority".into(),
            ));
        }
        self.session.execute(request)
    }

    pub fn cmd_vel(&mut self, linear_mm_s: i16, angular_mrad_s: i16, ttl_ms: u32) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::CmdVel {
            linear_mm_s,
            angular_mrad_s,
            ttl_ms,
        })?)
    }

    pub fn arm(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::Arm)?)
    }

    pub fn disarm(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::Disarm)?)
    }

    pub fn stop(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::Stop)?)
    }
}

/// Narrow handle exposing only separately authorized maintenance operations.
pub struct ServiceCockpit<'a, C> {
    session: &'a mut SessionCockpit<C>,
}

impl<C: Cockpit> ServiceCockpit<'_, C> {
    pub fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        if request.authorization_class() != AuthorizationClass::ServiceLease {
            return Err(CockpitError::Policy(
                "request is outside service-lease authority".into(),
            ));
        }
        self.session.execute(request)
    }

    pub fn restart_mpu(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::RestartMpu)?)
    }

    pub fn restart_create(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::RestartCreate)?)
    }

    pub fn reset_motherbrain(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::ResetMotherbrain)?)
    }

    pub fn bootsel(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::Bootsel)?)
    }
}

pub fn establish_session<C: Cockpit>(
    mut connector: C,
    hello: HandshakeHello,
    prior: Option<&CockpitSession>,
) -> Result<SessionCockpit<C>> {
    let outcome = connector.handshake(hello)?.classify_against(prior);
    outcome.contract.validate_event_vocabulary()?;
    let report = outcome.contract.validate_local_model();
    if !report.missing_verbs.is_empty() {
        return Err(CockpitError::Policy(format!(
            "live capability contract contains unsupported verbs: {}",
            report.missing_verbs.join(",")
        )));
    }
    if outcome.welcome.safety_snapshot.armed || outcome.welcome.safety_snapshot.active_motion {
        return Err(CockpitError::UnsafeHandshake(
            "brainstem is not ready-but-disarmed".into(),
        ));
    }
    let cursor = EventCursor::from_event_next_seq(outcome.event_cursor);
    Ok(SessionCockpit {
        connector,
        outcome,
        cursor,
        control_lease: None,
        service_lease: None,
    })
}

pub fn establish_diagnostic_session<C: Cockpit>(
    mut connector: C,
    mut hello: HandshakeHello,
    prior: Option<&CockpitSession>,
) -> Result<SessionCockpit<C>> {
    hello.session_purpose = SessionPurpose::Diagnostic;
    let outcome = connector.handshake(hello)?.classify_against(prior);
    outcome.contract.validate_event_vocabulary()?;
    let cursor = EventCursor::from_event_next_seq(outcome.event_cursor);
    Ok(SessionCockpit {
        connector,
        outcome,
        cursor,
        control_lease: None,
        service_lease: None,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementPolicy {
    Reject,
    Accept,
}

/// Owns exactly one command connector. Switching always performs a fresh
/// handshake; backup status probes should use a separate read-only connector.
pub struct FailoverCockpit {
    active: SessionCockpit<Box<dyn Cockpit>>,
}

impl FailoverCockpit {
    pub fn new(active: SessionCockpit<Box<dyn Cockpit>>) -> Self {
        Self { active }
    }
    pub fn active(&self) -> &SessionCockpit<Box<dyn Cockpit>> {
        &self.active
    }
    pub fn active_mut(&mut self) -> &mut SessionCockpit<Box<dyn Cockpit>> {
        &mut self.active
    }

    pub fn failover(
        &mut self,
        backup: Box<dyn Cockpit>,
        hello: HandshakeHello,
        replacement_policy: ReplacementPolicy,
    ) -> Result<()> {
        let prior = self.active.session().clone();
        let replacement = establish_session(backup, hello.new_attempt(), Some(&prior))?;
        if replacement.outcome().classification == ReconnectClassification::ReplacementBrainstem
            && replacement_policy == ReplacementPolicy::Reject
        {
            return Err(CockpitError::Policy(format!(
                "replacement brainstem {} requires explicit acceptance",
                replacement.session().peer_device_id
            )));
        }
        // Best effort on a failed physical lane. The new handshake already
        // synchronously stopped and disarmed the brainstem before WELCOME.
        let _ = self.active.execute(CockpitRequest::Stop);
        let _ = self.active.execute(CockpitRequest::Disarm);
        self.active = replacement;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MotherbrainBootstrap {
    pub hello: HandshakeHello,
    pub expected_brainstem_device_id: Option<String>,
}

#[derive(Debug, Error)]
pub enum BootstrapError {
    #[error("brainstem not found under /dev/serial/by-id")]
    BrainstemNotFound,
    #[error("serial open failed for {path}: {source}")]
    SerialOpenFailed { path: PathBuf, source: CockpitError },
    #[error("handshake failed on {path}: {source}")]
    HandshakeFailed { path: PathBuf, source: CockpitError },
    #[error("wrong brainstem identity: expected {expected}, received {received}")]
    WrongBrainstem { expected: String, received: String },
    #[error("unsafe welcome: {0}")]
    UnsafeWelcome(String),
    #[error("capability mismatch: {0}")]
    CapabilityMismatch(String),
    #[error("event history missed before sequence {dropped_before_seq}")]
    EventHistoryMissed { dropped_before_seq: u32 },
    #[error("brainstem network is not ready: {0}")]
    NetworkNotReady(String),
    #[error("network registration rejected: {0}")]
    NetworkRegistrationRejected(String),
    #[error("control lease rejected: {0}")]
    ControlLeaseRejected(String),
    #[error("DNS verification failed for {0}")]
    DnsVerificationFailed(String),
    #[error("all USB CDC candidates failed: {failures:?}")]
    CandidateFailures { failures: Vec<CandidateFailure> },
}

#[derive(Debug)]
pub struct CandidateFailure {
    pub path: PathBuf,
    pub cause: String,
}

impl MotherbrainBootstrap {
    pub fn from_host() -> Self {
        Self {
            hello: HandshakeHello::default_motherbrain(),
            expected_brainstem_device_id: None,
        }
    }

    /// USB CDC enumerates as a serial byte stream; `UartCockpit` is the shared
    /// line-protocol implementation, not an assertion that this is GPIO UART.
    pub fn connect_usb(
        &self,
    ) -> std::result::Result<SessionCockpit<Box<dyn Cockpit>>, BootstrapError> {
        let paths =
            discover_usb_serial_by_id().map_err(|error| BootstrapError::HandshakeFailed {
                path: PathBuf::from("/dev/serial/by-id"),
                source: error,
            })?;
        if paths.is_empty() {
            return Err(BootstrapError::BrainstemNotFound);
        }
        let mut errors = Vec::new();
        for path in paths {
            // Opening a Pico USB CDC endpoint and asserting DTR can race the
            // firmware's wait_connection loop, especially immediately after
            // flashing. Give it a short settle period and retry transient
            // open/handshake failures on this same pinned path.
            for attempt in 1..=3 {
                let connector = match UartCockpit::connect_with_config(
                    UartCockpitConfig::new(&path)
                        .with_timeout(Duration::from_secs(2))
                        .with_data_terminal_ready(true),
                ) {
                    Ok(connector) => connector,
                    Err(error) => {
                        errors.push(CandidateFailure {
                            path: path.clone(),
                            cause: format!(
                                "attempt {attempt}: {}",
                                BootstrapError::SerialOpenFailed {
                                    path: path.clone(),
                                    source: error,
                                }
                            ),
                        });
                        std::thread::sleep(Duration::from_millis(250));
                        continue;
                    }
                };
                std::thread::sleep(Duration::from_millis(250));
                match (|| -> Result<_> {
                    let ready = establish_session(
                        Box::new(connector) as Box<dyn Cockpit>,
                        self.hello.new_attempt(),
                        None,
                    )?;
                    self.validate_identity(ready.session())?;
                    Ok(ready)
                })() {
                    Ok(ready) => return Ok(ready),
                    Err(error) => errors.push(CandidateFailure {
                        path: path.clone(),
                        cause: format!("attempt {attempt}: {error}"),
                    }),
                }
                std::thread::sleep(Duration::from_millis(250));
            }
        }
        Err(BootstrapError::CandidateFailures { failures: errors })
    }

    pub fn connect_backup(
        &self,
        connector: Box<dyn Cockpit>,
        prior: &CockpitSession,
    ) -> Result<SessionCockpit<Box<dyn Cockpit>>> {
        let ready = establish_session(connector, self.hello.new_attempt(), Some(prior))?;
        self.validate_identity(ready.session())?;
        Ok(ready)
    }

    pub fn register_network(
        &self,
        ready: &mut SessionCockpit<Box<dyn Cockpit>>,
        endpoint: RegisterNetworkEndpoint,
    ) -> Result<NetworkEndpointRegistered> {
        match ready.execute(CockpitRequest::RegisterNetworkEndpoint(endpoint))? {
            CockpitResponse::NetworkEndpointRegistered(registered) => Ok(registered),
            response => Err(CockpitError::BadResponse(format!("{response:?}"))),
        }
    }

    fn validate_identity(&self, session: &CockpitSession) -> Result<()> {
        if self
            .expected_brainstem_device_id
            .as_deref()
            .is_some_and(|expected| expected != session.peer_device_id)
        {
            return Err(CockpitError::Policy(format!(
                "brainstem identity {} did not match configured {}",
                session.peer_device_id,
                self.expected_brainstem_device_id.as_deref().unwrap_or("")
            )));
        }
        Ok(())
    }
}

pub struct HttpCockpit {
    host: String,
    next_command_id: u32,
    timeout: Duration,
    active_session_id: Option<String>,
}

impl HttpCockpit {
    pub fn connect(host: impl Into<String>) -> Self {
        Self {
            host: host.into(),
            next_command_id: 1,
            timeout: Duration::from_millis(750),
            active_session_id: None,
        }
    }

    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    fn command_id(&mut self) -> u32 {
        let command_id = self.next_command_id;
        self.next_command_id = self.next_command_id.wrapping_add(1).max(1);
        command_id
    }

    fn post(&mut self, path: &str, body: &str) -> Result<String> {
        let addr = self
            .host
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| CockpitError::BadResponse("http host did not resolve".to_owned()))?;
        let mut stream = TcpStream::connect_timeout(&addr, self.timeout)?;
        stream.set_read_timeout(Some(self.timeout))?;
        stream.set_write_timeout(Some(self.timeout))?;
        write!(
            stream,
            "POST {path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            self.host,
            body.len(),
            body
        )?;
        stream.flush()?;
        let mut response = String::new();
        let mut bytes = [0u8; 1024];
        loop {
            match stream.read(&mut bytes) {
                Ok(0) => break,
                Ok(len) => response.push_str(&String::from_utf8_lossy(&bytes[..len])),
                Err(error)
                    if !response.is_empty()
                        && matches!(
                            error.kind(),
                            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                        ) =>
                {
                    break;
                }
                Err(error) => return Err(error.into()),
            }
        }
        http_body(&response)
    }
}

impl Cockpit for HttpCockpit {
    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        let command_id = self.command_id();
        let body = request.to_firmware_json(command_id)?;
        let response = self.post("/command", &body)?;
        parse_json_cockpit_response(command_id, &request, &response)
    }

    fn handshake(&mut self, hello: HandshakeHello) -> Result<HandshakeOutcome> {
        let body = serde_json::to_string(&hello)?;
        let response = self.post("/handshake", &body)?;
        let outcome = HandshakeOutcome::validate(&hello, serde_json::from_str(&response)?)?;
        self.active_session_id = Some(outcome.session.session_id.clone());
        Ok(outcome)
    }

    fn execute_in_session(
        &mut self,
        session: &CockpitSession,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let command_id = self.command_id();
        let body = request.to_firmware_json_with_session(command_id, &session.session_id)?;
        let response = self.post("/command", &body)?;
        parse_json_cockpit_response(command_id, &request, &response)
    }

    fn execute_with_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ControlLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let command_id = self.command_id();
        let body = request.to_firmware_json_with_authority(
            command_id,
            &session.session_id,
            &lease.lease_id,
        )?;
        let response = self.post("/command", &body)?;
        parse_json_cockpit_response(command_id, &request, &response)
    }

    fn execute_with_service_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ServiceLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let command_id = self.command_id();
        let body = request.to_firmware_json_with_service_authority(
            command_id,
            &session.session_id,
            &lease.lease_id,
        )?;
        let response = self.post("/command", &body)?;
        parse_json_cockpit_response(command_id, &request, &response)
    }
}

pub struct WebSocketCockpit {
    socket: WebSocket<MaybeTlsStream<TcpStream>>,
    next_command_id: u32,
    active_session_id: Option<String>,
}

impl WebSocketCockpit {
    pub fn connect_url(url: &str) -> Result<Self> {
        let (socket, _) = connect(url)?;
        Ok(Self {
            socket,
            next_command_id: 1,
            active_session_id: None,
        })
    }

    pub fn connect_pico_w(host: &str) -> Result<Self> {
        Self::connect_url(&format!("ws://{host}:81/control"))
    }

    fn command_id(&mut self) -> u32 {
        let command_id = self.next_command_id;
        self.next_command_id = self.next_command_id.wrapping_add(1).max(1);
        command_id
    }
}

impl Cockpit for WebSocketCockpit {
    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        let command_id = self.command_id();
        let body = request.to_firmware_json(command_id)?;
        self.socket.send(Message::Text(body.into()))?;
        loop {
            let message = self.socket.read()?;
            match message {
                Message::Text(text) => {
                    return parse_json_cockpit_response(command_id, &request, text.as_str());
                }
                Message::Binary(bytes) => {
                    let text = response_from_bytes(&bytes)?;
                    return parse_json_cockpit_response(command_id, &request, &text);
                }
                Message::Ping(bytes) => self.socket.send(Message::Pong(bytes))?,
                Message::Close(_) => {
                    return Err(CockpitError::BadResponse(
                        "websocket closed before response".to_owned(),
                    ));
                }
                _ => {}
            }
        }
    }

    fn handshake(&mut self, hello: HandshakeHello) -> Result<HandshakeOutcome> {
        let mut value = serde_json::to_value(&hello)?;
        value
            .as_object_mut()
            .expect("hello serializes as object")
            .insert("kind".into(), serde_json::Value::String("hello".into()));
        self.socket
            .send(Message::Text(serde_json::to_string(&value)?.into()))?;
        loop {
            match self.socket.read()? {
                Message::Text(text) => {
                    let outcome =
                        HandshakeOutcome::validate(&hello, serde_json::from_str(text.as_str())?)?;
                    self.active_session_id = Some(outcome.session.session_id.clone());
                    return Ok(outcome);
                }
                Message::Binary(bytes) => {
                    let outcome =
                        HandshakeOutcome::validate(&hello, serde_json::from_slice(&bytes)?)?;
                    self.active_session_id = Some(outcome.session.session_id.clone());
                    return Ok(outcome);
                }
                Message::Ping(bytes) => self.socket.send(Message::Pong(bytes))?,
                Message::Close(_) => {
                    return Err(CockpitError::BadResponse(
                        "websocket closed during handshake".into(),
                    ))
                }
                _ => {}
            }
        }
    }

    fn execute_in_session(
        &mut self,
        session: &CockpitSession,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let command_id = self.command_id();
        let body = request.to_firmware_json_with_session(command_id, &session.session_id)?;
        self.socket.send(Message::Text(body.into()))?;
        loop {
            match self.socket.read()? {
                Message::Text(text) => {
                    return parse_json_cockpit_response(command_id, &request, text.as_str())
                }
                Message::Binary(bytes) => {
                    return parse_json_cockpit_response(
                        command_id,
                        &request,
                        &response_from_bytes(&bytes)?,
                    )
                }
                Message::Ping(bytes) => self.socket.send(Message::Pong(bytes))?,
                Message::Close(_) => {
                    return Err(CockpitError::BadResponse(
                        "websocket closed before response".into(),
                    ))
                }
                _ => {}
            }
        }
    }

    fn execute_with_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ControlLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let command_id = self.command_id();
        let body = request.to_firmware_json_with_authority(
            command_id,
            &session.session_id,
            &lease.lease_id,
        )?;
        self.socket.send(Message::Text(body.into()))?;
        loop {
            match self.socket.read()? {
                Message::Text(text) => {
                    return parse_json_cockpit_response(command_id, &request, text.as_str())
                }
                Message::Binary(bytes) => {
                    return parse_json_cockpit_response(
                        command_id,
                        &request,
                        &response_from_bytes(&bytes)?,
                    )
                }
                Message::Ping(bytes) => self.socket.send(Message::Pong(bytes))?,
                Message::Close(_) => {
                    return Err(CockpitError::BadResponse(
                        "websocket closed before response".into(),
                    ))
                }
                _ => {}
            }
        }
    }

    fn execute_with_service_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ServiceLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let command_id = self.command_id();
        let body = request.to_firmware_json_with_service_authority(
            command_id,
            &session.session_id,
            &lease.lease_id,
        )?;
        self.socket.send(Message::Text(body.into()))?;
        loop {
            match self.socket.read()? {
                Message::Text(text) => {
                    return parse_json_cockpit_response(command_id, &request, text.as_str())
                }
                Message::Binary(bytes) => {
                    return parse_json_cockpit_response(
                        command_id,
                        &request,
                        &response_from_bytes(&bytes)?,
                    )
                }
                Message::Ping(bytes) => self.socket.send(Message::Pong(bytes))?,
                Message::Close(_) => {
                    return Err(CockpitError::BadResponse(
                        "websocket closed before response".into(),
                    ))
                }
                _ => {}
            }
        }
    }
}

pub struct UdpCockpit {
    socket: UdpSocket,
    brainstem: SocketAddr,
    next_seq: u32,
    timeout: Duration,
    active_session_id: Option<String>,
}

impl UdpCockpit {
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
            active_session_id: None,
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
        let mut buf = [0u8; MAX_COMPACT_HANDSHAKE_FRAME_LEN];
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

impl Cockpit for UdpCockpit {
    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        let seq = self.seq();
        let response = self.request(request.to_compact_line(seq))?;
        parse_compact_cockpit_response(seq, &request, &response)
    }

    fn handshake(&mut self, hello: HandshakeHello) -> Result<HandshakeOutcome> {
        let response = self.request(encode_compact_hello(&hello)?)?;
        let outcome = HandshakeOutcome::validate(&hello, decode_compact_response(&response)?)?;
        self.active_session_id = Some(outcome.session.session_id.clone());
        Ok(outcome)
    }

    fn execute_in_session(
        &mut self,
        session: &CockpitSession,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let seq = self.seq();
        let response =
            self.request(request.to_compact_line_with_session(seq, &session.session_id))?;
        parse_compact_cockpit_response(seq, &request, &response)
    }

    fn execute_with_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ControlLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let seq = self.seq();
        let response = self.request(request.to_compact_line_with_authority(
            seq,
            &session.session_id,
            &lease.lease_id,
        ))?;
        parse_compact_cockpit_response(seq, &request, &response)
    }

    fn execute_with_service_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ServiceLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let seq = self.seq();
        let response = self.request(request.to_compact_line_with_service_authority(
            seq,
            &session.session_id,
            &lease.lease_id,
        ))?;
        parse_compact_cockpit_response(seq, &request, &response)
    }

    fn get_status(&mut self) -> Result<CockpitStatus> {
        let seq = self.seq();
        let response = self.request(format!("STATUS {seq}\n"))?;
        expect_ok(seq, &response)?;
        Ok(CockpitStatus { raw: response })
    }

    fn get_capabilities(&mut self) -> Result<CockpitCapabilities> {
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

    fn zero_imu_orientation(&mut self) -> Result<()> {
        self.command("ZERO_IMU_ORIENTATION")
    }

    fn clear_imu_orientation(&mut self) -> Result<()> {
        self.command("CLEAR_IMU_ORIENTATION")
    }
}

pub const DEFAULT_UART_BAUD_RATE: u32 = 115_200;
pub const DEFAULT_UART_TIMEOUT: Duration = Duration::from_millis(750);
pub const DEFAULT_UART_MAX_RESPONSE_LEN: usize = MAX_COMPACT_HANDSHAKE_FRAME_LEN;

/// Enumerate stable Linux serial symlinks. Callers may further filter by USB
/// VID/PID or a configured identity hint, but must confirm identity by
/// handshake rather than trusting the filename.
pub fn discover_usb_serial_by_id() -> Result<Vec<PathBuf>> {
    let directory = Path::new("/dev/serial/by-id");
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut paths = std::fs::read_dir(directory)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct UartCockpitConfig {
    pub path: PathBuf,
    pub baud_rate: u32,
    pub timeout: Duration,
    pub max_response_len: usize,
    /// Assert DTR after opening. USB CDC firmware commonly uses DTR as its
    /// indication that a host is ready; hardware UART adapters need not.
    pub data_terminal_ready: bool,
}

impl UartCockpitConfig {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            baud_rate: DEFAULT_UART_BAUD_RATE,
            timeout: DEFAULT_UART_TIMEOUT,
            max_response_len: DEFAULT_UART_MAX_RESPONSE_LEN,
            data_terminal_ready: false,
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

    pub fn with_data_terminal_ready(mut self, ready: bool) -> Self {
        self.data_terminal_ready = ready;
        self
    }
}

pub struct UartCockpit {
    port: Box<dyn SerialPort>,
    next_seq: u32,
    timeout: Duration,
    max_response_len: usize,
    active_session_id: Option<String>,
}

impl UartCockpit {
    pub fn connect(path: impl AsRef<Path>) -> Result<Self> {
        Self::connect_with_config(UartCockpitConfig::new(path.as_ref()))
    }

    pub fn connect_with_config(config: UartCockpitConfig) -> Result<Self> {
        let mut port = serialport::new(config.path.to_string_lossy(), config.baud_rate)
            .timeout(config.timeout)
            .open()?;
        if config.data_terminal_ready {
            port.write_data_terminal_ready(true)?;
        }
        Ok(Self {
            port,
            next_seq: 1,
            timeout: config.timeout,
            max_response_len: config.max_response_len,
            active_session_id: None,
        })
    }

    pub fn from_port(port: Box<dyn SerialPort>) -> Self {
        Self {
            port,
            next_seq: 1,
            timeout: DEFAULT_UART_TIMEOUT,
            max_response_len: DEFAULT_UART_MAX_RESPONSE_LEN,
            active_session_id: None,
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

impl Cockpit for UartCockpit {
    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        let seq = self.seq();
        let response = self.request(request.to_compact_line(seq))?;
        parse_compact_cockpit_response(seq, &request, &response)
    }

    fn handshake(&mut self, hello: HandshakeHello) -> Result<HandshakeOutcome> {
        let response = self.request(encode_compact_hello(&hello)?)?;
        let outcome = HandshakeOutcome::validate(&hello, decode_compact_response(&response)?)?;
        self.active_session_id = Some(outcome.session.session_id.clone());
        Ok(outcome)
    }

    fn execute_in_session(
        &mut self,
        session: &CockpitSession,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let seq = self.seq();
        let response =
            self.request(request.to_compact_line_with_session(seq, &session.session_id))?;
        parse_compact_cockpit_response(seq, &request, &response)
    }

    fn execute_with_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ControlLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let seq = self.seq();
        let response = self.request(request.to_compact_line_with_authority(
            seq,
            &session.session_id,
            &lease.lease_id,
        ))?;
        parse_compact_cockpit_response(seq, &request, &response)
    }

    fn execute_with_service_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ServiceLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let seq = self.seq();
        let response = self.request(request.to_compact_line_with_service_authority(
            seq,
            &session.session_id,
            &lease.lease_id,
        ))?;
        parse_compact_cockpit_response(seq, &request, &response)
    }

    fn get_status(&mut self) -> Result<CockpitStatus> {
        let seq = self.seq();
        let response = self.request(format!("STATUS {seq}\n"))?;
        expect_ok(seq, &response)?;
        Ok(CockpitStatus { raw: response })
    }

    fn get_capabilities(&mut self) -> Result<CockpitCapabilities> {
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

    fn zero_imu_orientation(&mut self) -> Result<()> {
        self.command("ZERO_IMU_ORIENTATION")
    }

    fn clear_imu_orientation(&mut self) -> Result<()> {
        self.command("CLEAR_IMU_ORIENTATION")
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
                    return Err(CockpitError::BadResponse(
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
        .map_err(|_| CockpitError::BadResponse("response was not utf-8".into()))?
        .trim()
        .to_owned();
    Ok(response)
}

fn ensure_connector_session(active: &Option<String>, session: &CockpitSession) -> Result<()> {
    if active.as_deref() == Some(session.session_id.as_str()) {
        Ok(())
    } else {
        Err(CockpitError::InvalidSession {
            session_id: session.session_id.clone(),
        })
    }
}

fn expect_ok(seq: u32, response: &str) -> Result<()> {
    let mut parts = response.split_ascii_whitespace();
    match (
        parts.next(),
        parts.next().and_then(|value| value.parse::<u32>().ok()),
        parts.next(),
    ) {
        (Some("OK"), Some(response_seq), _) if response_seq == seq => Ok(()),
        (Some("ERR"), Some(response_seq), Some(reason))
            if response_seq == seq && is_compact_rejection_reason(reason) =>
        {
            Err(CockpitError::Rejected {
                command_id: seq,
                reason: reason.to_owned(),
            })
        }
        _ => Err(CockpitError::BadResponse(response.to_owned())),
    }
}

fn is_compact_rejection_reason(reason: &str) -> bool {
    matches!(
        reason,
        "busy"
            | "charging_busy"
            | "unsupported"
            | "invalid_session"
            | "session_required"
            | "invalid_control_lease"
            | "control_lease_required"
            | "invalid_service_lease"
            | "service_authorization_required"
            | "service_operation_disabled"
    )
}

fn parse_capabilities(seq: u32, response: &str) -> Result<CockpitCapabilities> {
    expect_ok(seq, response)?;
    let rest = response
        .strip_prefix(&format!("OK {seq} CAPABILITIES "))
        .ok_or_else(|| CockpitError::BadResponse(response.to_owned()))?;
    Ok(CockpitCapabilities {
        body_kind: value_for(rest, "body_kind").unwrap_or_default().to_owned(),
        drive: value_for(rest, "drive").unwrap_or_default().to_owned(),
        verbs: csv_for(rest, "verbs"),
        sensors: csv_for(rest, "sensors"),
        outputs: csv_for(rest, "outputs"),
        safety: csv_for(rest, "safety"),
        events: csv_for(rest, "events"),
        limits: parse_compact_limits(rest),
    })
}

fn parse_events(seq: u32, since_seq: u32, response: &str) -> Result<EventBatch> {
    expect_ok(seq, response)?;
    let rest = response
        .strip_prefix(&format!("OK {seq} EVENTS "))
        .ok_or_else(|| CockpitError::BadResponse(response.to_owned()))?;
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
            return Err(CockpitError::BadResponse(response.to_owned()));
        };
        let Some((kind_text, fields)) = tail.split_once(':') else {
            return Err(CockpitError::BadResponse(response.to_owned()));
        };
        let mut nums = fields.split(',');
        let event_seq = seq_text
            .parse()
            .map_err(|_| CockpitError::BadResponse(response.to_owned()))?;
        let a = nums
            .next()
            .ok_or_else(|| CockpitError::BadResponse(response.to_owned()))?
            .parse()
            .map_err(|_| CockpitError::BadResponse(response.to_owned()))?;
        let b = nums
            .next()
            .ok_or_else(|| CockpitError::BadResponse(response.to_owned()))?
            .parse()
            .map_err(|_| CockpitError::BadResponse(response.to_owned()))?;
        let c = nums
            .next()
            .ok_or_else(|| CockpitError::BadResponse(response.to_owned()))?
            .parse()
            .map_err(|_| CockpitError::BadResponse(response.to_owned()))?;
        if nums.next().is_some() {
            return Err(CockpitError::BadResponse(response.to_owned()));
        }
        batch.events.push(CockpitEvent {
            seq: event_seq,
            kind: CockpitEventKind::from(kind_text),
            a,
            b,
            c,
        });
        parsed_count += 1;
    }
    if number_for(header, "count").is_some_and(|count| count as usize != parsed_count) {
        return Err(CockpitError::BadResponse(response.to_owned()));
    }
    Ok(batch)
}

fn parse_compact_cockpit_response(
    seq: u32,
    request: &CockpitRequest,
    response: &str,
) -> Result<CockpitResponse> {
    match request {
        CockpitRequest::GetStatus => {
            expect_ok(seq, response)?;
            Ok(CockpitResponse::Status(CockpitStatus {
                raw: response.to_owned(),
            }))
        }
        CockpitRequest::GetCapabilities => Ok(CockpitResponse::Capabilities(parse_capabilities(
            seq, response,
        )?)),
        CockpitRequest::GetEvents { since_seq } => Ok(CockpitResponse::Events(parse_events(
            seq, *since_seq, response,
        )?)),
        CockpitRequest::RegisterNetworkEndpoint(_) => {
            expect_ok(seq, response)?;
            Ok(CockpitResponse::NetworkEndpointRegistered(
                NetworkEndpointRegistered {
                    session_id: value_for(response, "session_id")
                        .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                        .into(),
                    fqdn: value_for(response, "fqdn")
                        .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                        .into(),
                    address: value_for(response, "address")
                        .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                        .into(),
                    ttl_seconds: number_for(response, "ttl")
                        .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                    registration_generation: number_for(response, "generation")
                        .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                },
            ))
        }
        CockpitRequest::AcquireControlLease { .. } => {
            expect_ok(seq, response)?;
            let owner_role = serde_json::from_value(serde_json::Value::String(
                value_for(response, "owner_role")
                    .unwrap_or("unknown")
                    .into(),
            ))?;
            let authority = serde_json::from_value(serde_json::Value::String(
                value_for(response, "authority").unwrap_or("unknown").into(),
            ))?;
            Ok(CockpitResponse::ControlLeaseGranted(ControlLease {
                lease_id: value_for(response, "lease_id")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                session_id: value_for(response, "session_id")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                owner_role,
                authority,
                ttl_ms: number_for(response, "ttl_ms")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                generation: number_for(response, "generation")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
            }))
        }
        CockpitRequest::AcquireServiceLease { .. } => {
            expect_ok(seq, response)?;
            Ok(CockpitResponse::ServiceLeaseGranted(ServiceLease {
                lease_id: value_for(response, "lease_id")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                session_id: value_for(response, "session_id")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                owner_role: serde_json::from_value(serde_json::Value::String(
                    value_for(response, "owner_role")
                        .unwrap_or("unknown")
                        .into(),
                ))?,
                scope: serde_json::from_value(serde_json::Value::String(
                    value_for(response, "scope").unwrap_or("unknown").into(),
                ))?,
                ttl_ms: number_for(response, "ttl_ms")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                generation: number_for(response, "generation")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
            }))
        }
        _ => {
            expect_ok(seq, response)?;
            Ok(CockpitResponse::Accepted)
        }
    }
}

fn parse_json_cockpit_response(
    command_id: u32,
    request: &CockpitRequest,
    response: &str,
) -> Result<CockpitResponse> {
    let value: serde_json::Value = serde_json::from_str(response.trim())?;
    if value.get("accepted").and_then(serde_json::Value::as_bool) == Some(false) {
        let reason = json_str_value(&value, "message")
            .or_else(|| json_str_value(&value, "reason"))
            .unwrap_or("rejected")
            .to_owned();
        return Err(CockpitError::Rejected { command_id, reason });
    }

    match request {
        CockpitRequest::GetStatus => Ok(CockpitResponse::Status(CockpitStatus {
            raw: response.trim().to_owned(),
        })),
        CockpitRequest::GetCapabilities => Ok(CockpitResponse::Capabilities(
            parse_json_capabilities(&value)?,
        )),
        CockpitRequest::GetEvents { since_seq } => Ok(CockpitResponse::Events(parse_json_events(
            *since_seq, &value,
        )?)),
        CockpitRequest::RegisterNetworkEndpoint(_) => {
            let registered = NetworkEndpointRegistered {
                session_id: json_str_value(&value, "session_id")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                fqdn: json_str_value(&value, "fqdn")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                address: json_str_value(&value, "address")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                ttl_seconds: json_u32_value(&value, "ttl_seconds")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                registration_generation: json_u32_value(&value, "registration_generation")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
            };
            Ok(CockpitResponse::NetworkEndpointRegistered(registered))
        }
        CockpitRequest::AcquireControlLease { .. } => {
            Ok(CockpitResponse::ControlLeaseGranted(ControlLease {
                lease_id: json_str_value(&value, "lease_id")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                session_id: json_str_value(&value, "session_id")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                owner_role: serde_json::from_value(
                    value
                        .get("owner_role")
                        .cloned()
                        .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                )?,
                authority: serde_json::from_value(
                    value
                        .get("authority")
                        .cloned()
                        .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                )?,
                ttl_ms: json_u32_value(&value, "ttl_ms")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                generation: json_u32_value(&value, "generation")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
            }))
        }
        CockpitRequest::AcquireServiceLease { .. } => {
            Ok(CockpitResponse::ServiceLeaseGranted(ServiceLease {
                lease_id: json_str_value(&value, "lease_id")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                session_id: json_str_value(&value, "session_id")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                owner_role: serde_json::from_value(
                    value
                        .get("owner_role")
                        .cloned()
                        .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                )?,
                scope: serde_json::from_value(
                    value
                        .get("scope")
                        .cloned()
                        .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                )?,
                ttl_ms: json_u32_value(&value, "ttl_ms")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                generation: json_u32_value(&value, "generation")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
            }))
        }
        _ => {
            if value.get("accepted").and_then(serde_json::Value::as_bool) == Some(true) {
                Ok(CockpitResponse::Accepted)
            } else {
                Err(CockpitError::BadResponse(response.to_owned()))
            }
        }
    }
}

fn parse_json_capabilities(value: &serde_json::Value) -> Result<CockpitCapabilities> {
    Ok(CockpitCapabilities {
        body_kind: json_str_value(value, "body_kind")
            .unwrap_or_default()
            .to_owned(),
        drive: json_str_value(value, "drive")
            .unwrap_or_default()
            .to_owned(),
        verbs: json_string_array(value, "verbs"),
        sensors: json_string_array(value, "sensors"),
        outputs: json_string_array(value, "outputs"),
        safety: json_string_array(value, "safety"),
        events: json_string_array(value, "events"),
        limits: parse_json_limits(value),
    })
}

fn parse_compact_limits(line: &str) -> CockpitLimits {
    let Some(raw) = value_for(line, "limits") else {
        return CockpitLimits::default();
    };
    let mut limits = CockpitLimits::default();
    for item in raw.split(',') {
        let Some((key, value)) = item.split_once(':') else {
            continue;
        };
        match key {
            "max_linear_mm_s" => {
                if let Ok(value) = value.parse() {
                    limits.max_linear_mm_s = value;
                }
            }
            "max_angular_mrad_s" => {
                if let Ok(value) = value.parse() {
                    limits.max_angular_mrad_s = value;
                }
            }
            "min_ttl_ms" => {
                if let Ok(value) = value.parse() {
                    limits.min_ttl_ms = value;
                }
            }
            "max_ttl_ms" => {
                if let Ok(value) = value.parse() {
                    limits.max_ttl_ms = value;
                }
            }
            _ => {}
        }
    }
    limits
}

fn parse_json_limits(value: &serde_json::Value) -> CockpitLimits {
    let Some(limits) = value.get("limits") else {
        return CockpitLimits::default();
    };
    CockpitLimits {
        max_linear_mm_s: json_i16_value(limits, "max_linear_mm_s").unwrap_or(i16::MAX),
        max_angular_mrad_s: json_i16_value(limits, "max_angular_mrad_s").unwrap_or(i16::MAX),
        min_ttl_ms: json_u32_value(limits, "min_ttl_ms").unwrap_or(1),
        max_ttl_ms: json_u32_value(limits, "max_ttl_ms").unwrap_or(u32::MAX),
    }
}

fn parse_json_events(since_seq: u32, value: &serde_json::Value) -> Result<EventBatch> {
    let events = value
        .get("events")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| CockpitError::BadResponse(value.to_string()))?
        .iter()
        .map(|event| {
            Ok(CockpitEvent {
                seq: json_u32_value(event, "seq")
                    .ok_or_else(|| CockpitError::BadResponse(event.to_string()))?,
                kind: CockpitEventKind::from(json_str_value(event, "kind").unwrap_or("unknown")),
                a: json_u32_value(event, "a").unwrap_or(0),
                b: json_u32_value(event, "b").unwrap_or(0),
                c: json_u32_value(event, "c").unwrap_or(0),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(EventBatch {
        since_seq,
        oldest_seq: json_u32_value(value, "oldest_seq").unwrap_or(0),
        next_seq: json_u32_value(value, "next_seq").unwrap_or(since_seq),
        dropped_before_seq: json_u32_value(value, "dropped_before_seq").unwrap_or(0),
        events,
    })
}

fn json_string_array(value: &serde_json::Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
        .collect()
}

fn http_body(response: &str) -> Result<String> {
    let (head, body) = response
        .split_once("\r\n\r\n")
        .ok_or_else(|| CockpitError::BadResponse(response.to_owned()))?;
    let protocol_rejection = (head.starts_with("HTTP/1.1 409")
        || head.starts_with("HTTP/1.0 409")
        || head.starts_with("HTTP/1.1 422")
        || head.starts_with("HTTP/1.0 422"))
        && body.trim_start().starts_with('{');
    if !head.starts_with("HTTP/1.1 200") && !head.starts_with("HTTP/1.0 200") && !protocol_rejection
    {
        return Err(CockpitError::BadResponse(head.to_owned()));
    }
    Ok(body.trim().to_owned())
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

fn signed_number_for(line: &str, key: &str) -> Option<i32> {
    value_for(line, key)?.parse().ok()
}

fn bool_for(line: &str, key: &str) -> Option<bool> {
    match value_for(line, key)? {
        "true" | "1" | "on" | "yes" => Some(true),
        "false" | "0" | "off" | "no" => Some(false),
        _ => None,
    }
}

fn json_str_value<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    value.get(key)?.as_str()
}

fn json_bool_value(value: &serde_json::Value, key: &str) -> Option<bool> {
    value.get(key)?.as_bool()
}

fn json_tri_state_value(value: &serde_json::Value, key: &str) -> Option<bool> {
    match value.get(key)? {
        serde_json::Value::Bool(value) => Some(*value),
        serde_json::Value::String(value) => match value.as_str() {
            "true" | "1" | "on" | "yes" => Some(true),
            "false" | "0" | "off" | "no" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn json_u32_value(value: &serde_json::Value, key: &str) -> Option<u32> {
    value
        .get(key)?
        .as_u64()
        .and_then(|value| value.try_into().ok())
}

fn json_i32_value(value: &serde_json::Value, key: &str) -> Option<i32> {
    value
        .get(key)?
        .as_i64()
        .and_then(|value| value.try_into().ok())
}

fn json_i16_value(value: &serde_json::Value, key: &str) -> Option<i16> {
    value
        .get(key)?
        .as_i64()
        .and_then(|value| value.try_into().ok())
}

fn compact_tones(tones: &[SongTone]) -> String {
    let mut encoded = String::new();
    for tone in tones {
        encoded.push_str(&format!(" {} {}", tone.note, tone.duration_64ths));
    }
    encoded
}

fn rewrite_for_firmware_json(
    request: &CockpitRequest,
    object: &mut serde_json::Map<String, serde_json::Value>,
) {
    match request {
        CockpitRequest::DefineChirp { tones, .. } | CockpitRequest::SongDefine { tones, .. } => {
            object.insert(
                "tones".to_owned(),
                tones
                    .iter()
                    .map(|tone| format!("{}:{}", tone.note, tone.duration_64ths))
                    .collect::<Vec<_>>()
                    .join(",")
                    .into(),
            );
        }
        CockpitRequest::SetSafetyPolicy { policy } => {
            object.remove("policy");
            object.insert("bump_action".to_owned(), policy.bump.as_str().into());
            object.insert("cliff_action".to_owned(), policy.cliff.as_str().into());
            object.insert(
                "wheel_drop_latch".to_owned(),
                policy.wheel_drop_latch.into(),
            );
        }
        _ => {}
    }
}

fn pack_i16_pair(left: i16, right: i16) -> u32 {
    ((left as u16 as u32) << 16) | right as u16 as u32
}

fn time_reached(now_ms: u32, deadline_ms: u32) -> bool {
    now_ms.wrapping_sub(deadline_ms) < u32::MAX / 2
}

fn fresh_sim_boot_id() -> String {
    format!("simboot-{}", uuid::Uuid::new_v4().simple())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::thread;

    struct StopRejectingCockpit {
        inner: SimCockpit,
        reject_stop: bool,
        disarm_requests: usize,
    }

    struct BusyOnceCockpit {
        inner: SimCockpit,
        busy_remaining: usize,
        attempts: usize,
    }

    struct StaleLeaseOnceCockpit {
        inner: SimCockpit,
        invalid_remaining: usize,
        attempts: usize,
    }

    impl Cockpit for BusyOnceCockpit {
        fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
            self.inner.execute(request)
        }

        fn handshake(&mut self, hello: HandshakeHello) -> Result<HandshakeOutcome> {
            self.inner.handshake(hello)
        }

        fn execute_in_session(
            &mut self,
            session: &CockpitSession,
            request: CockpitRequest,
        ) -> Result<CockpitResponse> {
            self.inner.execute_in_session(session, request)
        }

        fn execute_with_lease(
            &mut self,
            session: &CockpitSession,
            lease: &ControlLease,
            request: CockpitRequest,
        ) -> Result<CockpitResponse> {
            self.attempts += 1;
            if self.busy_remaining > 0 {
                self.busy_remaining -= 1;
                return Err(CockpitError::Rejected {
                    command_id: self.attempts as u32,
                    reason: "busy".into(),
                });
            }
            self.inner.execute_with_lease(session, lease, request)
        }

        fn execute_with_service_lease(
            &mut self,
            session: &CockpitSession,
            lease: &ServiceLease,
            request: CockpitRequest,
        ) -> Result<CockpitResponse> {
            self.inner
                .execute_with_service_lease(session, lease, request)
        }
    }

    impl Cockpit for StaleLeaseOnceCockpit {
        fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
            self.inner.execute(request)
        }

        fn handshake(&mut self, hello: HandshakeHello) -> Result<HandshakeOutcome> {
            self.inner.handshake(hello)
        }

        fn execute_in_session(
            &mut self,
            session: &CockpitSession,
            request: CockpitRequest,
        ) -> Result<CockpitResponse> {
            self.inner.execute_in_session(session, request)
        }

        fn execute_with_lease(
            &mut self,
            session: &CockpitSession,
            lease: &ControlLease,
            request: CockpitRequest,
        ) -> Result<CockpitResponse> {
            self.attempts += 1;
            if self.invalid_remaining > 0 {
                self.invalid_remaining -= 1;
                return Err(CockpitError::Rejected {
                    command_id: self.attempts as u32,
                    reason: "invalid_control_lease".into(),
                });
            }
            self.inner.execute_with_lease(session, lease, request)
        }

        fn execute_with_service_lease(
            &mut self,
            session: &CockpitSession,
            lease: &ServiceLease,
            request: CockpitRequest,
        ) -> Result<CockpitResponse> {
            self.inner
                .execute_with_service_lease(session, lease, request)
        }
    }

    impl Cockpit for StopRejectingCockpit {
        fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
            self.inner.execute(request)
        }

        fn handshake(&mut self, hello: HandshakeHello) -> Result<HandshakeOutcome> {
            self.inner.handshake(hello)
        }

        fn execute_in_session(
            &mut self,
            session: &CockpitSession,
            request: CockpitRequest,
        ) -> Result<CockpitResponse> {
            if matches!(&request, CockpitRequest::Stop) && self.reject_stop {
                return Ok(CockpitResponse::Rejected {
                    message: "stop not acknowledged".into(),
                });
            }
            if matches!(&request, CockpitRequest::Disarm) {
                self.disarm_requests += 1;
            }
            self.inner.execute_in_session(session, request)
        }

        fn execute_with_lease(
            &mut self,
            session: &CockpitSession,
            lease: &ControlLease,
            request: CockpitRequest,
        ) -> Result<CockpitResponse> {
            self.inner.execute_with_lease(session, lease, request)
        }

        fn execute_with_service_lease(
            &mut self,
            session: &CockpitSession,
            lease: &ServiceLease,
            request: CockpitRequest,
        ) -> Result<CockpitResponse> {
            self.inner
                .execute_with_service_lease(session, lease, request)
        }
    }

    fn hello() -> HandshakeHello {
        HandshakeHello::motherbrain("pete-motherbrain-test")
    }

    fn conformance_caps() -> CockpitCapabilities {
        SimCockpit::new().get_capabilities().unwrap()
    }

    fn conformance_welcome(hello: &HandshakeHello) -> HandshakeResponse {
        negotiate(
            hello,
            "pete-brainstem-wire-test",
            "bsboot-wire-test",
            conformance_caps(),
            SafetySnapshot {
                armed: false,
                estop_latched: false,
                safety_tripped: false,
                active_motion: false,
                runtime_state: "idle".into(),
            },
            1,
            SoftwareInfo {
                software_name: "wire-test".into(),
                software_version: "1".into(),
                build_id: "test".into(),
            },
            1,
        )
    }

    fn compact_emulator_response(line: &str) -> String {
        if line.starts_with("HELLO ") {
            let hello = decode_compact_hello(line).unwrap();
            return encode_compact_response(&conformance_welcome(&hello)).unwrap();
        }
        let seq = line
            .split_ascii_whitespace()
            .nth(1)
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0);
        if line.starts_with("REGISTER_NETWORK_ENDPOINT ") {
            return format!("OK {seq} NETWORK_ENDPOINT_REGISTERED session_id=sess-wire fqdn=motherbrain.pete.internal address=192.168.4.2 ttl=60 generation=1\n");
        }
        if line.starts_with("ACQUIRE_CONTROL_LEASE ") {
            return format!("OK {seq} CONTROL_LEASE_GRANTED lease_id=lease-wire session_id=sess-wire owner_role=motherbrain authority=motherbrain ttl_ms=1000 generation=1\n");
        }
        if line.starts_with("BOOTSEL ") {
            return format!("ERR {seq} service_authorization_required\n");
        }
        if line.starts_with("CMD_VEL ") {
            return if line.contains(" session_id=") && line.contains(" lease_id=") {
                format!("OK {seq}\n")
            } else if line.contains(" session_id=") {
                format!("ERR {seq} invalid_control_lease\n")
            } else {
                format!("ERR {seq} invalid_session\n")
            };
        }
        format!("ERR {seq} unsupported\n")
    }

    fn json_emulator_response(body: &str) -> String {
        let value: serde_json::Value = serde_json::from_str(body).unwrap();
        if value.get("kind").and_then(serde_json::Value::as_str) == Some("hello")
            || value.get("handshake_nonce").is_some() && value.get("command_id").is_none()
        {
            let hello: HandshakeHello = serde_json::from_value(value).unwrap();
            return serde_json::to_string(&conformance_welcome(&hello)).unwrap();
        }
        let kind = value
            .get("kind")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        match kind {
            "register_network_endpoint" => serde_json::json!({
                "accepted": true,
                "session_id": "sess-wire",
                "fqdn": "motherbrain.pete.internal",
                "address": "192.168.4.2",
                "ttl_seconds": 60,
                "registration_generation": 1
            })
            .to_string(),
            "acquire_control_lease" => serde_json::json!({
                "accepted": true,
                "type": "control_lease_granted",
                "lease_id": "lease-wire",
                "session_id": "sess-wire",
                "owner_role": "motherbrain",
                "authority": "motherbrain",
                "ttl_ms": 1000,
                "generation": 1
            })
            .to_string(),
            "bootsel" => serde_json::json!({
                "accepted": false,
                "message": "service_authorization_required"
            })
            .to_string(),
            "cmd_vel" => {
                let accepted = value.get("session_id").is_some() && value.get("lease_id").is_some();
                if accepted {
                    serde_json::json!({"accepted": true}).to_string()
                } else {
                    serde_json::json!({
                        "accepted": false,
                        "message": if value.get("session_id").is_some() {
                            "invalid_control_lease"
                        } else {
                            "invalid_session"
                        }
                    })
                    .to_string()
                }
            }
            _ => serde_json::json!({"accepted": false, "message": "unsupported"}).to_string(),
        }
    }

    fn run_physical_connector_conformance<C: Cockpit>(connector: &mut C) {
        let hello = hello();
        let outcome = connector.handshake(hello).unwrap();
        assert_eq!(outcome.session.peer_device_id, "pete-brainstem-wire-test");
        let motion = CockpitRequest::CmdVel {
            linear_mm_s: 40,
            angular_mrad_s: 0,
            ttl_ms: 300,
        };
        assert!(connector.execute(motion.clone()).is_err());
        assert!(connector
            .execute_in_session(&outcome.session, motion.clone())
            .is_err());
        let registration = connector
            .execute_in_session(
                &outcome.session,
                CockpitRequest::RegisterNetworkEndpoint(RegisterNetworkEndpoint {
                    interface_id: "wlan1".into(),
                    address_family: AddressFamily::Ipv4,
                    address: "192.168.4.2".into(),
                    hostname: "motherbrain".into(),
                    lease_identity: "010203".into(),
                    ttl_seconds: 60,
                }),
            )
            .unwrap();
        assert!(matches!(
            registration,
            CockpitResponse::NetworkEndpointRegistered(_)
        ));
        let lease = match connector
            .execute_in_session(
                &outcome.session,
                CockpitRequest::AcquireControlLease {
                    authority: ControlAuthority::Motherbrain,
                    ttl_ms: 1_000,
                },
            )
            .unwrap()
        {
            CockpitResponse::ControlLeaseGranted(lease) => lease,
            other => panic!("{other:?}"),
        };
        assert!(connector
            .execute_with_lease(&outcome.session, &lease, motion)
            .is_ok());
        assert!(connector
            .execute_with_lease(&outcome.session, &lease, CockpitRequest::Bootsel)
            .is_err());
    }

    #[test]
    fn uart_connector_runs_shared_session_conformance() {
        let (host, mut device) = serialport::TTYPort::pair().unwrap();
        let server = thread::spawn(move || {
            let mut reader = BufReader::new(device.try_clone().unwrap());
            for _ in 0..7 {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                device
                    .write_all(compact_emulator_response(&line).as_bytes())
                    .unwrap();
                device.flush().unwrap();
            }
        });
        let mut connector = UartCockpit::from_port(Box::new(host));
        run_physical_connector_conformance(&mut connector);
        server.join().unwrap();
    }

    #[test]
    fn udp_connector_runs_shared_session_conformance() {
        let server_socket = UdpSocket::bind("127.0.0.1:0").unwrap();
        let address = server_socket.local_addr().unwrap();
        let server = thread::spawn(move || {
            let mut buffer = [0u8; MAX_COMPACT_HANDSHAKE_FRAME_LEN];
            for _ in 0..7 {
                let (len, peer) = server_socket.recv_from(&mut buffer).unwrap();
                let line = std::str::from_utf8(&buffer[..len]).unwrap();
                server_socket
                    .send_to(compact_emulator_response(line).as_bytes(), peer)
                    .unwrap();
            }
        });
        let mut connector = UdpCockpit::connect(address).unwrap();
        run_physical_connector_conformance(&mut connector);
        server.join().unwrap();
    }

    fn read_http_test_body(stream: &mut TcpStream) -> String {
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut bytes = Vec::new();
        let header_end = loop {
            let mut byte = [0u8; 1];
            stream.read_exact(&mut byte).unwrap();
            bytes.push(byte[0]);
            if bytes.ends_with(b"\r\n\r\n") {
                break bytes.len();
            }
        };
        let header = std::str::from_utf8(&bytes[..header_end]).unwrap();
        let content_length = header
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap();
        let mut body = vec![0u8; content_length];
        stream.read_exact(&mut body).unwrap();
        String::from_utf8(body).unwrap()
    }

    #[test]
    fn http_connector_runs_shared_session_conformance() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for _ in 0..7 {
                let (mut stream, _) = listener.accept().unwrap();
                let body = read_http_test_body(&mut stream);
                let response = json_emulator_response(&body);
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response.len(),
                    response
                )
                .unwrap();
                stream.flush().unwrap();
            }
        });
        let mut connector = HttpCockpit::connect(address.to_string());
        run_physical_connector_conformance(&mut connector);
        server.join().unwrap();
    }

    #[test]
    fn websocket_connector_runs_shared_session_conformance() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut socket = tungstenite::accept(stream).unwrap();
            for _ in 0..7 {
                let message = socket.read().unwrap();
                let body = message.into_text().unwrap();
                socket
                    .send(Message::Text(json_emulator_response(&body).into()))
                    .unwrap();
            }
        });
        let mut connector =
            WebSocketCockpit::connect_url(&format!("ws://{address}/control")).unwrap();
        run_physical_connector_conformance(&mut connector);
        server.join().unwrap();
    }

    #[test]
    fn process_boot_identity_is_stable_while_handshake_nonces_are_fresh() {
        let first = hello();
        let second = hello();
        assert_eq!(first.boot_id, second.boot_id);
        assert_ne!(first.handshake_nonce, second.handshake_nonce);
        assert_eq!(first.boot_id, first.new_attempt().boot_id);
        assert_ne!(first.handshake_nonce, first.new_attempt().handshake_nonce);
    }

    #[test]
    fn handshake_conformance_success_minor_and_json_compact_parity() {
        let mut sim = SimCockpit::new();
        let hello = hello();
        let outcome = sim.handshake(hello.clone()).unwrap();
        assert_eq!(outcome.welcome.protocol_minor, PROTOCOL_MINOR_MAX);
        assert_eq!(
            outcome.welcome.echoed_handshake_nonce,
            hello.handshake_nonce
        );
        assert!(!outcome.welcome.safety_snapshot.armed);
        assert!(!outcome.welcome.safety_snapshot.active_motion);
        assert_eq!(
            outcome.contract.capabilities(),
            &outcome.welcome.capability_contract
        );

        let hello_line = encode_compact_hello(&hello).unwrap();
        assert_eq!(decode_compact_hello(&hello_line).unwrap(), hello);
        let response = HandshakeResponse::Welcome(outcome.welcome.clone());
        let line = encode_compact_response(&response).unwrap();
        assert_eq!(decode_compact_response(&line).unwrap(), response);
        let json = serde_json::to_string(&response).unwrap();
        assert_eq!(
            serde_json::from_str::<HandshakeResponse>(&json).unwrap(),
            response
        );
    }

    #[test]
    fn handshake_structured_rejections_and_stale_nonce() {
        let caps = SimCockpit::new().get_capabilities().unwrap();
        let safety = SafetySnapshot {
            armed: false,
            estop_latched: false,
            safety_tripped: false,
            active_motion: false,
            runtime_state: "idle".into(),
        };
        let software = SoftwareInfo {
            software_name: "test".into(),
            software_version: "1".into(),
            build_id: "x".into(),
        };
        let mut major = hello();
        major.protocol_major += 1;
        assert!(matches!(
            negotiate(
                &major,
                "brain",
                "boot",
                caps.clone(),
                safety.clone(),
                1,
                software.clone(),
                1
            ),
            HandshakeResponse::Reject(HandshakeReject {
                reason_code: HandshakeRejectReason::ProtocolMajorMismatch,
                ..
            })
        ));
        let mut wrong_role = hello();
        wrong_role.role = EndpointRole::Brainstem;
        assert!(matches!(
            negotiate(
                &wrong_role,
                "brain",
                "boot",
                caps.clone(),
                safety.clone(),
                1,
                software.clone(),
                1
            ),
            HandshakeResponse::Reject(HandshakeReject {
                reason_code: HandshakeRejectReason::WrongRole,
                ..
            })
        ));
        let mut feature = hello();
        feature
            .required_features
            .push(HandshakeFeature::Unknown("future_required".into()));
        assert!(matches!(
            negotiate(&feature, "brain", "boot", caps, safety, 1, software, 1),
            HandshakeResponse::Reject(HandshakeReject {
                reason_code: HandshakeRejectReason::MissingRequiredFeature,
                ..
            })
        ));

        let mut sim = SimCockpit::new();
        let hello = hello();
        let mut response = match negotiate(
            &hello,
            "brain",
            "boot",
            sim.get_capabilities().unwrap(),
            SafetySnapshot {
                armed: false,
                estop_latched: false,
                safety_tripped: false,
                active_motion: false,
                runtime_state: "idle".into(),
            },
            1,
            SoftwareInfo {
                software_name: "x".into(),
                software_version: "1".into(),
                build_id: "x".into(),
            },
            1,
        ) {
            HandshakeResponse::Welcome(w) => w,
            _ => unreachable!(),
        };
        response.echoed_handshake_nonce = "old-nonce".into();
        assert!(matches!(
            HandshakeOutcome::validate(&hello, HandshakeResponse::Welcome(response)),
            Err(CockpitError::StaleHandshake { .. })
        ));
    }

    #[test]
    fn compact_handshake_rejects_malformed_and_oversized_frames() {
        assert!(decode_compact_hello("HELLO !!!\n").is_err());
        let oversized = format!("HELLO {}\n", "a".repeat(MAX_COMPACT_HANDSHAKE_FRAME_LEN));
        assert!(matches!(
            decode_compact_hello(&oversized),
            Err(CockpitError::FrameTooLarge { .. })
        ));
    }

    #[test]
    fn compact_line_decoder_handles_fragmentation_and_multiple_frames() {
        let mut decoder = CompactLineDecoder::new(32);
        assert!(decoder.push(b"HEL").unwrap().is_empty());
        assert_eq!(
            decoder.push(b"LO one\nPING two\npart").unwrap(),
            vec!["HELLO one", "PING two"]
        );
        assert_eq!(decoder.push(b"ial\n").unwrap(), vec!["partial"]);
        assert!(matches!(
            CompactLineDecoder::new(2).push(b"abc"),
            Err(CockpitError::FrameTooLarge { .. })
        ));
    }

    #[test]
    fn session_replacement_stops_disarms_and_rejects_old_session_without_clearing_estop() {
        let mut sim = SimCockpit::new();
        let first = sim.handshake(hello()).unwrap();
        sim.execute_in_session(
            &first.session,
            CockpitRequest::AcquireControlLease {
                authority: ControlAuthority::Motherbrain,
                ttl_ms: 1_000,
            },
        )
        .unwrap();
        sim.execute_in_session(&first.session, CockpitRequest::Arm)
            .unwrap();
        sim.execute_in_session(
            &first.session,
            CockpitRequest::CmdVel {
                linear_mm_s: 100,
                angular_mrad_s: 0,
                ttl_ms: 1_000,
            },
        )
        .unwrap();
        sim.execute_in_session(&first.session, CockpitRequest::EStop)
            .unwrap();

        let second = sim.handshake(hello()).unwrap();
        let cockpit_status = sim.get_status().unwrap();
        let status = cockpit_status.summary();
        assert_eq!(status.armed, Some(false));
        assert!(cockpit_status.raw.contains("active_cmd_vel=false"));
        assert_eq!(status.estop_latched, Some(true));
        assert_ne!(first.session.session_id, second.session.session_id);
        assert!(matches!(
            sim.execute_in_session(&first.session, CockpitRequest::Arm),
            Err(CockpitError::InvalidSession { .. })
        ));
    }

    #[test]
    fn reconnect_classifies_transport_reboot_and_replacement() {
        let mut sim = SimCockpit::new().with_identity("brain-a", "boot-a");
        let first = sim.handshake(hello()).unwrap();
        let reconnect = sim
            .handshake(hello())
            .unwrap()
            .classify_against(Some(&first.session));
        assert_eq!(
            reconnect.classification,
            ReconnectClassification::TransportReconnect
        );
        let mut rebooted = SimCockpit::new().with_identity("brain-a", "boot-b");
        let reboot = rebooted
            .handshake(hello())
            .unwrap()
            .classify_against(Some(&first.session));
        assert_eq!(
            reboot.classification,
            ReconnectClassification::BrainstemReboot
        );
        let mut replacement = SimCockpit::new().with_identity("brain-b", "boot-c");
        let replacement = replacement
            .handshake(hello())
            .unwrap()
            .classify_against(Some(&first.session));
        assert_eq!(
            replacement.classification,
            ReconnectClassification::ReplacementBrainstem
        );
    }

    #[test]
    fn duplicate_hello_is_idempotent() {
        let mut sim = SimCockpit::new();
        let hello = hello();
        let first = sim.handshake(hello.clone()).unwrap();
        let duplicate = sim.handshake(hello).unwrap();
        assert_eq!(first.session.session_id, duplicate.session.session_id);
    }

    #[test]
    fn failover_same_device_creates_fresh_single_authority_session() {
        let primary: Box<dyn Cockpit> =
            Box::new(SimCockpit::new().with_identity("brain-a", "boot-a"));
        let ready = establish_session(primary, hello(), None).unwrap();
        let old_session = ready.session().clone();
        let mut failover = FailoverCockpit::new(ready);
        let backup: Box<dyn Cockpit> =
            Box::new(SimCockpit::new().with_identity("brain-a", "boot-a"));
        failover
            .failover(backup, hello(), ReplacementPolicy::Reject)
            .unwrap();
        assert_ne!(
            old_session.session_id,
            failover.active().session().session_id
        );
        assert_eq!(
            failover.active().outcome().classification,
            ReconnectClassification::TransportReconnect
        );
        failover
            .active_mut()
            .acquire_control(ControlAuthority::Motherbrain, 1_000)
            .unwrap();
        assert!(failover.active_mut().execute(CockpitRequest::Arm).is_ok());
    }

    #[test]
    fn operator_debug_and_forebrain_recovery_transitions_stop_and_revoke() {
        let mut sim = SimCockpit::new().with_takeover_policy(true, Some("forebrain-alpha".into()));
        let mother = sim.handshake(hello()).unwrap();
        let mother_lease = match sim
            .execute_in_session(
                &mother.session,
                CockpitRequest::AcquireControlLease {
                    authority: ControlAuthority::Motherbrain,
                    ttl_ms: 500,
                },
            )
            .unwrap()
        {
            CockpitResponse::ControlLeaseGranted(lease) => lease,
            other => panic!("{other:?}"),
        };
        sim.execute_with_lease(&mother.session, &mother_lease, CockpitRequest::Arm)
            .unwrap();
        sim.execute_with_lease(
            &mother.session,
            &mother_lease,
            CockpitRequest::CmdVel {
                linear_mm_s: 100,
                angular_mrad_s: 0,
                ttl_ms: 1_000,
            },
        )
        .unwrap();

        let mut operator_hello = HandshakeHello::motherbrain("operator-laptop");
        operator_hello.role = EndpointRole::Operator;
        operator_hello.session_purpose = SessionPurpose::Control;
        let operator = sim.handshake(operator_hello).unwrap();
        assert!(sim
            .get_status()
            .unwrap()
            .raw
            .contains("active_cmd_vel=true"));
        let debug_lease = match sim
            .execute_in_session(
                &operator.session,
                CockpitRequest::AcquireControlLease {
                    authority: ControlAuthority::OperatorDebug,
                    ttl_ms: 500,
                },
            )
            .unwrap()
        {
            CockpitResponse::ControlLeaseGranted(lease) => lease,
            other => panic!("{other:?}"),
        };
        assert!(sim
            .get_status()
            .unwrap()
            .raw
            .contains("active_cmd_vel=false"));
        assert!(sim
            .execute_with_lease(&mother.session, &mother_lease, CockpitRequest::Arm)
            .is_err());

        let mut forebrain_hello = HandshakeHello::forebrain("forebrain-alpha");
        forebrain_hello.session_purpose = SessionPurpose::Control;
        forebrain_hello.handshake_nonce = "forebrain-recovery".into();
        let forebrain = sim.handshake(forebrain_hello).unwrap();
        assert!(sim
            .execute_in_session(
                &forebrain.session,
                CockpitRequest::AcquireControlLease {
                    authority: ControlAuthority::ForebrainRecovery,
                    ttl_ms: 500
                }
            )
            .is_err());
        sim.advance_ms(debug_lease.ttl_ms);
        let recovery = sim
            .execute_in_session(
                &forebrain.session,
                CockpitRequest::AcquireControlLease {
                    authority: ControlAuthority::ForebrainRecovery,
                    ttl_ms: 500,
                },
            )
            .unwrap();
        assert!(matches!(recovery, CockpitResponse::ControlLeaseGranted(_)));
        assert!(sim
            .execute_with_lease(&operator.session, &debug_lease, CockpitRequest::Arm)
            .is_err());
    }

    #[test]
    fn takeover_roles_are_default_deny() {
        let mut sim = SimCockpit::new();
        let operator = sim
            .handshake(HandshakeHello::operator("operator-laptop"))
            .unwrap();
        assert!(sim
            .execute_in_session(
                &operator.session,
                CockpitRequest::AcquireControlLease {
                    authority: ControlAuthority::OperatorDebug,
                    ttl_ms: 500,
                }
            )
            .is_err());
        let forebrain = sim
            .handshake(HandshakeHello::forebrain("forebrain-alpha"))
            .unwrap();
        assert!(sim
            .execute_in_session(
                &forebrain.session,
                CockpitRequest::AcquireControlLease {
                    authority: ControlAuthority::ForebrainRecovery,
                    ttl_ms: 500,
                }
            )
            .is_err());
    }

    #[test]
    fn authorization_classes_separate_emergency_control_and_service() {
        assert_eq!(
            CockpitRequest::GetStatus.authorization_class(),
            AuthorizationClass::ReadOnly
        );
        assert_eq!(
            CockpitRequest::EStop.authorization_class(),
            AuthorizationClass::Emergency
        );
        assert_eq!(
            CockpitRequest::Arm.authorization_class(),
            AuthorizationClass::ControlLease
        );
        for request in [
            CockpitRequest::Bootsel,
            CockpitRequest::RestartMpu,
            CockpitRequest::RestartCreate,
            CockpitRequest::ResetMotherbrain,
        ] {
            assert_eq!(
                request.authorization_class(),
                AuthorizationClass::ServiceLease
            );
        }
        let mut sim = SimCockpit::new();
        assert!(Cockpit::arm(&mut sim).is_err());
        assert!(Cockpit::cmd_vel(&mut sim, 10, 0, 100).is_err());
        assert!(Cockpit::stop(&mut sim).is_ok());
    }

    #[test]
    fn diagnostic_sessions_cannot_acquire_motion_authority_or_service_access() {
        let mut sim = SimCockpit::new().with_takeover_policy(true, Some("forebrain-alpha".into()));
        let operator = sim
            .handshake(HandshakeHello::operator("operator-laptop"))
            .unwrap();
        assert!(sim
            .execute_in_session(
                &operator.session,
                CockpitRequest::AcquireControlLease {
                    authority: ControlAuthority::OperatorDebug,
                    ttl_ms: 500,
                },
            )
            .is_err());
        let mother = sim.handshake(hello()).unwrap();
        let control = match sim
            .execute_in_session(
                &mother.session,
                CockpitRequest::AcquireControlLease {
                    authority: ControlAuthority::Motherbrain,
                    ttl_ms: 500,
                },
            )
            .unwrap()
        {
            CockpitResponse::ControlLeaseGranted(lease) => lease,
            other => panic!("{other:?}"),
        };
        assert!(sim
            .execute_with_lease(&mother.session, &control, CockpitRequest::Bootsel)
            .is_err());
        assert!(sim
            .execute_in_session(
                &mother.session,
                CockpitRequest::AcquireServiceLease {
                    scope: ServiceScope::Bootsel,
                    ttl_ms: 500,
                },
            )
            .is_err());
    }

    #[test]
    fn ready_connector_initializes_cursor_at_welcome_without_dropping_session_event() {
        let mut ready = establish_session(SimCockpit::new(), hello(), None).unwrap();
        let batch = ready.poll_events().unwrap();
        assert_eq!(batch.dropped_before_seq, 0);
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::SessionOpened));
    }

    fn motherbrain_registration(address: &str, lease: &str) -> RegisterNetworkEndpoint {
        RegisterNetworkEndpoint {
            interface_id: "wlan0".into(),
            address_family: AddressFamily::Ipv4,
            address: address.into(),
            hostname: "motherbrain".into(),
            lease_identity: lease.into(),
            ttl_seconds: 600,
        }
    }

    #[test]
    fn authenticated_motherbrain_registers_lease_and_reserved_dhcp_name_cannot() {
        let mut sim = SimCockpit::new();
        sim.add_network_lease(NetworkLease {
            leased_ip: "192.168.4.2".into(),
            client_mac: "02:00:00:00:00:02".into(),
            dhcp_client_identifier: "mb-client".into(),
            requested_hostname: Some("motherbrain".into()),
            lease_start: 0,
            lease_expiry: 3_600,
        });
        assert_eq!(sim.resolve_internal_name("motherbrain"), None);
        let outcome = sim.handshake(hello()).unwrap();
        let registration = motherbrain_registration("192.168.4.2", "mb-client");
        let response = sim
            .execute_in_session(
                &outcome.session,
                CockpitRequest::RegisterNetworkEndpoint(registration.clone()),
            )
            .unwrap();
        let CockpitResponse::NetworkEndpointRegistered(registered) = response else {
            panic!("registration response")
        };
        assert_eq!(
            registered.fqdn,
            format!("motherbrain.{DEFAULT_INTERNAL_DOMAIN}")
        );
        assert_eq!(
            sim.resolve_internal_name("motherbrain"),
            Some("192.168.4.2".into())
        );
        assert_eq!(
            sim.resolve_internal_name("pete"),
            Some("192.168.4.1".into())
        );
        assert_eq!(
            sim.resolve_internal_name("brainstem"),
            Some("192.168.4.1".into())
        );
        let duplicate = sim
            .execute_in_session(
                &outcome.session,
                CockpitRequest::RegisterNetworkEndpoint(registration.clone()),
            )
            .unwrap();
        let CockpitResponse::NetworkEndpointRegistered(duplicate) = duplicate else {
            panic!("duplicate registration response")
        };
        assert_eq!(
            registered.registration_generation,
            duplicate.registration_generation
        );

        let mut rebooted_hello = hello();
        rebooted_hello.boot_id = "mbboot-restarted".into();
        rebooted_hello.handshake_nonce = "hello-after-restart".into();
        let rebooted = sim.handshake(rebooted_hello).unwrap();
        let after_reboot = sim
            .execute_in_session(
                &rebooted.session,
                CockpitRequest::RegisterNetworkEndpoint(registration),
            )
            .unwrap();
        let CockpitResponse::NetworkEndpointRegistered(after_reboot) = after_reboot else {
            panic!("reboot registration response")
        };
        assert_ne!(
            duplicate.registration_generation,
            after_reboot.registration_generation
        );
    }

    #[test]
    fn network_registration_rejects_no_session_forebrain_and_lease_mismatch() {
        let mut sim = SimCockpit::new();
        let registration = motherbrain_registration("192.168.4.2", "mb-client");
        assert!(matches!(
            sim.execute(CockpitRequest::RegisterNetworkEndpoint(
                registration.clone()
            )),
            Err(CockpitError::SessionRequired)
        ));

        let mut forebrain_hello = HandshakeHello::forebrain("pete-forebrain-lab");
        forebrain_hello.handshake_nonce = "forebrain-attempt".into();
        let forebrain = sim.handshake(forebrain_hello).unwrap();
        assert!(matches!(
            sim.execute_in_session(
                &forebrain.session,
                CockpitRequest::RegisterNetworkEndpoint(registration.clone())
            ),
            Err(CockpitError::Policy(_))
        ));

        let outcome = sim.handshake(hello()).unwrap();
        assert!(matches!(
            sim.execute_in_session(
                &outcome.session,
                CockpitRequest::RegisterNetworkEndpoint(registration)
            ),
            Err(CockpitError::Policy(_))
        ));
    }

    #[test]
    fn expired_lease_removes_registered_dns_record() {
        let mut sim = SimCockpit::new();
        sim.add_network_lease(NetworkLease {
            leased_ip: "192.168.4.2".into(),
            client_mac: "mac".into(),
            dhcp_client_identifier: "lease".into(),
            requested_hostname: None,
            lease_start: 0,
            lease_expiry: 2,
        });
        let outcome = sim.handshake(hello()).unwrap();
        sim.execute_in_session(
            &outcome.session,
            CockpitRequest::RegisterNetworkEndpoint(motherbrain_registration(
                "192.168.4.2",
                "lease",
            )),
        )
        .unwrap();
        assert!(sim.resolve_internal_name("motherbrain").is_some());
        sim.advance_ms(2_000);
        assert_eq!(sim.resolve_internal_name("motherbrain"), None);
    }

    #[test]
    fn simulator_capabilities_round_trip() {
        let mut sim = SimCockpit::new();
        let caps = sim.get_capabilities().unwrap();
        assert_eq!(caps.body_kind, "sim_create_oi");
        assert_eq!(caps.drive, "differential");
        assert!(caps.verbs.contains(&"cmd_vel".to_owned()));
        assert!(caps.events.contains(&"safety_tripped".to_owned()));
        assert_eq!(caps.limits.max_linear_mm_s, 500);
    }

    #[test]
    fn cockpit_request_covers_public_firmware_verbs_from_body_toml() {
        let cockpit_verbs: BTreeSet<_> = sample_cockpit_requests()
            .into_iter()
            .map(|(verb, _, _)| verb)
            .filter(|verb| *verb != "bootsel")
            .collect();
        let firmware_verbs: BTreeSet<_> = body_toml_array("verbs").into_iter().collect();
        assert_eq!(cockpit_verbs, firmware_verbs);
    }

    #[test]
    fn cockpit_event_kind_covers_public_firmware_events_from_body_toml() {
        for event in body_toml_array("events") {
            assert!(
                !matches!(CockpitEventKind::from(event), CockpitEventKind::Unknown(_)),
                "body.toml event {event} is not modeled by CockpitEventKind"
            );
        }
    }

    #[test]
    fn body_toml_capabilities_validate_local_cockpit_model() {
        let contract = CockpitContract::new(body_toml_capabilities());
        let report = contract.validate_local_model();
        assert!(
            report.is_clean(),
            "missing={:?} extra={:?} unknown_events={:?}",
            report.missing_verbs,
            report.extra_verbs,
            report.unknown_events
        );
    }

    #[test]
    fn cockpit_requests_serialize_to_firmware_json_kinds() {
        for (verb, expected_json_kind, _) in sample_cockpit_requests() {
            let request = sample_request_for(verb);
            let json = request.to_firmware_json(7).unwrap();
            let value: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(
                value.get("kind").and_then(serde_json::Value::as_str),
                Some(expected_json_kind),
                "{verb} serialized as {json}"
            );
            assert_eq!(
                value.get("command_id").and_then(serde_json::Value::as_u64),
                Some(7)
            );
        }
    }

    #[test]
    fn cockpit_requests_serialize_to_compact_command_names() {
        for (verb, _, expected_compact_name) in sample_cockpit_requests() {
            let request = sample_request_for(verb);
            let line = request.to_compact_line(9);
            let first = line.split_ascii_whitespace().next().unwrap();
            assert_eq!(first, expected_compact_name, "{verb} serialized as {line}");
        }
    }

    #[test]
    fn firmware_json_rewrites_policy_and_tones() {
        let policy = CockpitRequest::SetSafetyPolicy {
            policy: SafetyPolicy {
                bump: SafetyAction::BumpEscape,
                cliff: SafetyAction::Backoff,
                wheel_drop_latch: true,
            },
        };
        let value: serde_json::Value =
            serde_json::from_str(&policy.to_firmware_json(1).unwrap()).unwrap();
        assert!(value.get("policy").is_none());
        assert_eq!(value["bump_action"], "bump_escape");
        assert_eq!(value["cliff_action"], "backoff");
        assert_eq!(value["wheel_drop_latch"], true);

        let song = CockpitRequest::SongDefine {
            id: 2,
            tones: vec![SongTone {
                note: 72,
                duration_64ths: 8,
            }],
        };
        let value: serde_json::Value =
            serde_json::from_str(&song.to_firmware_json(2).unwrap()).unwrap();
        assert_eq!(value["tones"], "72:8");
    }

    #[test]
    fn parses_json_accepted_and_rejected_command_responses() {
        let accepted = parse_json_cockpit_response(
            4,
            &CockpitRequest::Arm,
            r#"{"accepted":true,"command_id":4,"message":"accepted"}"#,
        )
        .unwrap();
        assert_eq!(accepted, CockpitResponse::Accepted);

        let rejected = parse_json_cockpit_response(
            5,
            &CockpitRequest::Arm,
            r#"{"accepted":false,"command_id":5,"message":"busy"}"#,
        )
        .unwrap_err();
        assert!(matches!(
            rejected,
            CockpitError::Rejected {
                command_id: 5,
                reason
            } if reason == "busy"
        ));
    }

    #[test]
    fn parses_json_status_capabilities_and_events() {
        let status = parse_json_cockpit_response(
            1,
            &CockpitRequest::GetStatus,
            r#"{"type":"status","current_runtime_state":"idle","oi_mode":"safe","event_next_seq":8,"create_sensors":{"charging_indicator":"on"}}"#,
        )
        .unwrap();
        let CockpitResponse::Status(status) = status else {
            panic!("expected status response");
        };
        let summary = status.summary();
        assert_eq!(summary.runtime_state.as_deref(), Some("idle"));
        assert_eq!(summary.armed, Some(true));
        assert_eq!(summary.event_next_seq, Some(8));
        assert_eq!(summary.battery.charging_indicator, Some(true));

        let caps = parse_json_cockpit_response(
            2,
            &CockpitRequest::GetCapabilities,
            r#"{"accepted":true,"command_id":2,"body_kind":"create_oi","drive":"differential","verbs":["arm","cmd_vel"],"sensors":["bump"],"outputs":["drive"],"safety":["estop"],"events":["boot","safety_tripped"]}"#,
        )
        .unwrap();
        let CockpitResponse::Capabilities(caps) = caps else {
            panic!("expected capabilities response");
        };
        assert_eq!(caps.body_kind, "create_oi");
        assert_eq!(caps.verbs, ["arm", "cmd_vel"]);
        assert_eq!(caps.limits.max_linear_mm_s, i16::MAX);

        let events = parse_json_cockpit_response(
            3,
            &CockpitRequest::GetEvents { since_seq: 6 },
            r#"{"type":"events","since_seq":6,"oldest_seq":4,"next_seq":9,"dropped_before_seq":0,"events":[{"seq":7,"kind":"safety_tripped","a":1,"b":0,"c":0},{"seq":8,"kind":"motion_stopped","a":0,"b":0,"c":0}]}"#,
        )
        .unwrap();
        let CockpitResponse::Events(events) = events else {
            panic!("expected events response");
        };
        assert_eq!(events.since_seq, 6);
        assert_eq!(events.next_seq, 9);
        assert!(events.has_stop_reason());
    }

    #[test]
    fn malformed_json_response_maps_to_json_error() {
        let err = parse_json_cockpit_response(1, &CockpitRequest::Arm, "{not-json").unwrap_err();
        assert!(matches!(err, CockpitError::Json(_)));
    }

    #[test]
    fn simulator_event_cursor_happy_path() {
        let mut sim = SimCockpit::new().with_unscoped_bench_mode();
        let mut cursor = EventCursor::new();
        let boot = cursor.poll(&mut sim).unwrap();
        assert_eq!(boot.events[0].kind, CockpitEventKind::Boot);
        sim.arm().unwrap();
        let batch = cursor.poll(&mut sim).unwrap();
        assert_eq!(cursor.next_seq(), batch.next_seq - 1);
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::CommandCompleted));
    }

    #[test]
    fn simulator_detects_missed_events_through_dropped_before_seq() {
        let mut sim = SimCockpit::new()
            .with_unscoped_bench_mode()
            .with_event_capacity(3);
        for _ in 0..4 {
            sim.arm().unwrap();
        }
        let batch = sim.get_events_since(0).unwrap();
        assert!(batch.dropped_before_seq > 0);
        assert!(matches!(
            batch.ensure_no_missed_events(),
            Err(CockpitError::MissedEvents { .. })
        ));
    }

    #[test]
    fn simulator_arm_stop_disarm_lifecycle() {
        let mut sim = SimCockpit::new().with_unscoped_bench_mode();
        sim.arm().unwrap();
        sim.cmd_vel(50, 0, 100).unwrap();
        sim.stop().unwrap();
        sim.disarm().unwrap();
        let batch = sim.get_events_since(0).unwrap();
        let kinds: Vec<_> = batch.events.iter().map(|event| &event.kind).collect();
        assert!(kinds.contains(&&CockpitEventKind::CommandInterrupted));
        assert!(kinds.contains(&&CockpitEventKind::MotionStopped));
        assert!(kinds.contains(&&CockpitEventKind::CommandCompleted));
    }

    #[test]
    fn simulator_cmd_vel_completes_after_ttl() {
        let mut sim = SimCockpit::new().with_unscoped_bench_mode();
        sim.cmd_vel(70, 10, 300).unwrap();
        sim.advance_ms(299);
        assert!(!sim
            .get_events_since(0)
            .unwrap()
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::MotionStopped));
        sim.advance_ms(1);
        let batch = sim.get_events_since(0).unwrap();
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::MotionStopped));
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::CommandCompleted));
    }

    #[test]
    fn simulator_estop_and_clear_estop() {
        let mut sim = SimCockpit::new().with_unscoped_bench_mode();
        sim.estop().unwrap();
        sim.clear_estop().unwrap();
        let batch = sim.get_events_since(0).unwrap();
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::EStopLatched));
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::EStopCleared));
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::SafetyCleared));
    }

    #[test]
    fn simulator_heartbeat_expiry_is_stop_reason() {
        let mut sim = SimCockpit::new().with_unscoped_bench_mode();
        sim.cmd_vel(70, 0, 1_000).unwrap();
        sim.heartbeat_stop(100).unwrap();
        sim.advance_ms(100);
        let batch = sim.get_events_since(0).unwrap();
        assert!(batch.has_stop_reason());
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::HeartbeatExpired));
    }

    #[test]
    fn simulator_command_rejection_alone_is_not_stop_reason() {
        let mut sim = SimCockpit::new().with_unscoped_bench_mode();
        sim.push_event(CockpitEventKind::CommandRejected, 7, 0, 0);
        let batch = sim.get_events_since(0).unwrap();

        assert!(!batch.has_stop_reason());
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::CommandRejected));
    }

    #[test]
    fn simulator_safety_tripped_stops_motion_and_rejects_motion() {
        let mut sim = SimCockpit::new().with_unscoped_bench_mode();
        sim.cmd_vel(70, 0, 1_000).unwrap();
        sim.trip_safety();
        sim.cmd_vel(10, 0, 100).unwrap();
        let batch = sim.get_events_since(0).unwrap();
        assert!(batch.has_stop_reason());
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::CommandRejected));
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::MotionStopped));
    }

    #[test]
    fn simulator_reset_odometry() {
        let mut sim = SimCockpit::new().with_unscoped_bench_mode();
        sim.reset_odometry().unwrap();
        assert_eq!(sim.odometry_reset_count(), 1);
        let status = sim.get_status().unwrap();
        assert!(status.raw.contains("odometry_resets=1"));
        assert_eq!(status.summary().odometry.reset_count, Some(1));
        assert_eq!(status.summary().odometry.distance_mm, Some(0));
    }

    #[test]
    fn simulator_builtin_sensor_edges_trip_and_clear() {
        let mut sim = SimCockpit::new();
        sim.set_bump(true, false);
        sim.set_bump(false, false);
        sim.set_cliff(true);
        sim.set_cliff(false);
        sim.set_wall(true);
        sim.set_wall(false);
        sim.set_virtual_wall(true);
        sim.set_virtual_wall(false);

        let batch = sim.get_events_since(0).unwrap();
        assert_eq!(
            batch
                .events
                .iter()
                .filter(|event| event.kind == CockpitEventKind::BumpChanged)
                .count(),
            2
        );
        assert_eq!(
            batch
                .events
                .iter()
                .filter(|event| event.kind == CockpitEventKind::CliffChanged)
                .count(),
            2
        );
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::WallChanged && event.a == 1));
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::VirtualWallChanged && event.a == 0));
    }

    #[test]
    fn simulator_wheel_drop_latches_and_clears() {
        let mut sim = SimCockpit::new().with_unscoped_bench_mode();
        sim.cmd_vel(70, 0, 1_000).unwrap();
        sim.set_wheel_drop(true);
        sim.set_wheel_drop(false);

        let batch = sim.get_events_since(0).unwrap();
        assert!(batch.has_stop_reason());
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::WheelDropLatched));
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::WheelDropCleared));
    }

    #[test]
    fn simulator_low_battery_and_charging_state_change() {
        let mut sim = SimCockpit::new();
        sim.set_battery(400, 2600);
        sim.set_charging_state(2);

        let status = sim.get_status().unwrap().summary();
        assert_eq!(status.battery.percent, Some(15));
        assert_eq!(status.battery.low, Some(true));
        assert_eq!(status.battery.charging_state, Some(2));

        let batch = sim.get_events_since(0).unwrap();
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::BatteryLow && event.a == 15));
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::ChargingStateChanged && event.a == 2));
    }

    #[test]
    fn simulator_buttons_and_ir_changes_are_events() {
        let mut sim = SimCockpit::new();
        sim.set_buttons(0b0000_0011);
        sim.set_ir_byte(248);

        let status = sim.get_status().unwrap().summary();
        assert_eq!(status.contact.any_contact(), Some(false));
        let batch = sim.get_events_since(0).unwrap();
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::ButtonsChanged && event.a == 3));
        assert!(batch
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::IrChanged && event.a == 248));
    }

    #[test]
    fn parses_ok_and_err_responses() {
        assert!(expect_ok(2, "OK 2").is_ok());
        assert!(matches!(
            expect_ok(2, "ERR 2 busy"),
            Err(CockpitError::Rejected {
                command_id: 2,
                reason
            }) if reason == "busy"
        ));
    }

    #[test]
    fn parses_status_response_as_raw_status() {
        expect_ok(9, "OK 9 STATUS runtime=idle demo=idle").unwrap();
        let status = CockpitStatus {
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
        assert_eq!(batch.events[1].kind, CockpitEventKind::SafetyTripped);
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
            CockpitEventKind::Unknown("new_future_event".to_owned())
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
                Err(CockpitError::BadResponse(_))
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
            Err(CockpitError::MissedEvents {
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
        assert_eq!(caps.limits.max_linear_mm_s, 500);
    }

    #[test]
    fn parses_json_capability_limits() {
        let caps = parse_json_capabilities(&serde_json::json!({
            "body_kind":"create_oi",
            "drive":"differential",
            "verbs":["cmd_vel"],
            "events":["boot"],
            "limits":{
                "max_linear_mm_s":120,
                "max_angular_mrad_s":800,
                "min_ttl_ms":20,
                "max_ttl_ms":900
            }
        }))
        .unwrap();
        assert_eq!(
            caps.limits,
            CockpitLimits {
                max_linear_mm_s: 120,
                max_angular_mrad_s: 800,
                min_ttl_ms: 20,
                max_ttl_ms: 900,
            }
        );
    }

    #[test]
    fn contract_rejects_unsupported_lights_music_and_step_verbs() {
        let contract =
            CockpitContract::new(sim_caps_without(&["set_lights", "song_play", "dock_align"]));
        assert!(matches!(
            contract.validate_request(&CockpitRequest::SetLights {
                pattern: LightPattern::Status
            }),
            Err(CockpitError::Policy(message)) if message.contains("set_lights")
        ));
        assert!(matches!(
            contract.validate_request(&CockpitRequest::SongPlay { id: 0 }),
            Err(CockpitError::Policy(message)) if message.contains("song_play")
        ));
        assert!(matches!(
            contract.validate_request(&CockpitRequest::DockAlign {
                bearing_mrad: 0,
                range_mm: 400,
                max_linear_mm_s: 80,
                max_angular_mrad_s: 500,
                stop_range_mm: 200,
                ttl_ms: 300,
            }),
            Err(CockpitError::Policy(message)) if message.contains("dock_align")
        ));
    }

    #[test]
    fn safe_cockpit_clamps_motion_to_body_limits() {
        let mut caps = sim_caps_with_all_verbs();
        caps.limits.max_linear_mm_s = 40;
        caps.limits.max_angular_mrad_s = 100;
        caps.limits.min_ttl_ms = 50;
        caps.limits.max_ttl_ms = 200;
        let sim = SimCockpit::new()
            .with_unscoped_bench_mode()
            .with_capabilities(caps);
        let mut safe = SafeCockpit::with_policy(
            sim,
            AgentPolicy {
                motion_ttl_ms: 500,
                heartbeat_timeout_ms: 500,
            },
        );
        safe.pulse_motion(120, 300).unwrap();
        let batch = safe.client_mut().get_events_since(0).unwrap();
        let motion = batch
            .events
            .iter()
            .find(|event| event.kind == CockpitEventKind::MotionRequested)
            .unwrap();
        assert_eq!(motion.a, pack_i16_pair(40, 100));
        assert_eq!(motion.b, 200);
    }

    #[test]
    fn safe_cockpit_requires_heartbeat_only_when_policy_uses_it() {
        let caps = sim_caps_without(&["heartbeat_stop"]);
        let sim = SimCockpit::new()
            .with_unscoped_bench_mode()
            .with_capabilities(caps.clone());
        let mut safe = SafeCockpit::new(sim);
        assert!(matches!(
            safe.pulse_motion(20, 0),
            Err(CockpitError::Policy(message)) if message.contains("heartbeat_stop")
        ));

        let sim = SimCockpit::new()
            .with_unscoped_bench_mode()
            .with_capabilities(caps);
        let mut safe = SafeCockpit::with_policy(
            sim,
            AgentPolicy {
                motion_ttl_ms: 100,
                heartbeat_timeout_ms: 0,
            },
        );
        safe.pulse_motion(20, 0).unwrap();
    }

    #[test]
    fn uart_config_defaults_to_forebrain_baud() {
        let config = UartCockpitConfig::new("/dev/ttyTEST0");
        assert_eq!(config.baud_rate, DEFAULT_UART_BAUD_RATE);
        assert_eq!(config.timeout, DEFAULT_UART_TIMEOUT);
        assert_eq!(config.max_response_len, DEFAULT_UART_MAX_RESPONSE_LEN);
    }

    #[test]
    fn malformed_response_maps_to_bad_response() {
        let err = expect_ok(2, "ERR 2 parse").unwrap_err();
        assert!(matches!(err, CockpitError::BadResponse(_)));
    }

    #[test]
    fn mismatched_sequence_maps_to_bad_response() {
        let err = expect_ok(1, "OK 12").unwrap_err();
        assert!(matches!(err, CockpitError::BadResponse(_)));
    }

    #[test]
    fn non_utf8_response_maps_to_bad_response() {
        let err = response_from_bytes(&[0xff]).unwrap_err();
        assert!(matches!(err, CockpitError::BadResponse(_)));
    }

    fn sample_cockpit_requests() -> Vec<(&'static str, &'static str, &'static str)> {
        vec![
            ("ping", "ping", "PING"),
            ("bootsel", "bootsel", "BOOTSEL"),
            ("restart_mpu", "restart_mpu", "RESTART_MPU"),
            ("restart_create", "restart_create", "RESTART_CREATE"),
            ("status", "status", "STATUS"),
            ("get_capabilities", "get_capabilities", "GET_CAPABILITIES"),
            ("get_events", "get_events", "GET_EVENTS"),
            ("arm", "arm", "ARM"),
            ("disarm", "disarm", "DISARM"),
            ("stop", "stop", "STOP"),
            ("estop", "estop", "ESTOP"),
            ("clear_estop", "clear_estop", "CLEAR_ESTOP"),
            (
                "clear_motion_queue",
                "clear_motion_queue",
                "CLEAR_MOTION_QUEUE",
            ),
            ("cmd_vel", "cmd_vel", "CMD_VEL"),
            ("drive_direct", "drive_direct", "DRIVE_DIRECT"),
            ("drive_arc", "drive_arc", "DRIVE_ARC"),
            ("drive_for", "drive_for", "DRIVE_FOR"),
            ("turn_by", "turn_by", "TURN_BY"),
            ("arc_for", "arc_for", "ARC_FOR"),
            ("creep_until", "creep_until", "CREEP_UNTIL"),
            ("scan_arc", "scan_arc", "SCAN_ARC"),
            ("face_bearing", "face_bearing", "FACE_BEARING"),
            ("track_bearing", "track_bearing", "TRACK_BEARING"),
            ("hold_heading", "hold_heading", "HOLD_HEADING"),
            ("turn_to_heading", "turn_to_heading", "TURN_TO_HEADING"),
            ("dock_align", "dock_align", "DOCK_ALIGN"),
            ("wall_follow", "wall_follow", "WALL_FOLLOW"),
            ("wiggle_align", "wiggle_align", "WIGGLE_ALIGN"),
            ("bump_escape", "bump_escape", "BUMP_ESCAPE"),
            ("unstick", "unstick", "UNSTICK"),
            ("cliff_guard", "cliff_guard", "CLIFF_GUARD"),
            ("heartbeat_stop", "heartbeat_stop", "HEARTBEAT_STOP"),
            ("request_sensors", "request_sensors", "REQUEST_SENSORS"),
            ("stream_sensors", "stream_sensors", "STREAM_SENSORS"),
            (
                "set_safety_policy",
                "set_safety_policy",
                "SET_SAFETY_POLICY",
            ),
            ("song_define", "song_define", "SONG_DEFINE"),
            ("song_play", "song_play", "SONG_PLAY"),
            ("define_chirp", "define_chirp", "DEFINE_CHIRP"),
            ("play_feedback", "play_feedback", "PLAY_FEEDBACK"),
            ("power_state", "power_state", "POWER_STATE"),
            ("calibrate_turn", "calibrate_turn", "CALIBRATE_TURN"),
            ("reset_odometry", "reset_odometry", "RESET_ODOMETRY"),
            (
                "zero_imu_orientation",
                "zero_imu_orientation",
                "ZERO_IMU_ORIENTATION",
            ),
            (
                "clear_imu_orientation",
                "clear_imu_orientation",
                "CLEAR_IMU_ORIENTATION",
            ),
            ("dock", "dock", "DOCK"),
            ("set_lights", "set_lights", "SET_LIGHTS"),
            ("set_mode", "set_mode", "SET_MODE"),
        ]
    }

    fn body_toml() -> toml::Value {
        include_str!("../../pete-brainstem/body.toml")
            .parse()
            .unwrap()
    }

    fn body_toml_array(key: &str) -> Vec<&'static str> {
        let body = body_toml();
        let values = body["capabilities"][key].as_array().unwrap();
        values
            .iter()
            .map(|value| {
                let value = value.as_str().unwrap().to_owned();
                Box::leak(value.into_boxed_str()) as &'static str
            })
            .collect()
    }

    fn body_toml_capabilities() -> CockpitCapabilities {
        let body = body_toml();
        let limits = &body["limits"];
        CockpitCapabilities {
            body_kind: body["body"]["kind"].as_str().unwrap().to_owned(),
            drive: body["body"]["drive"].as_str().unwrap().to_owned(),
            verbs: body_toml_array("verbs")
                .into_iter()
                .map(ToOwned::to_owned)
                .collect(),
            sensors: body_toml_array("sensors")
                .into_iter()
                .map(ToOwned::to_owned)
                .collect(),
            outputs: body_toml_array("outputs")
                .into_iter()
                .map(ToOwned::to_owned)
                .collect(),
            safety: body_toml_array("safety")
                .into_iter()
                .map(ToOwned::to_owned)
                .collect(),
            events: body_toml_array("events")
                .into_iter()
                .map(ToOwned::to_owned)
                .collect(),
            limits: CockpitLimits {
                max_linear_mm_s: limits["max_linear_mm_s"]
                    .as_integer()
                    .unwrap()
                    .try_into()
                    .unwrap(),
                max_angular_mrad_s: limits["max_angular_mrad_s"]
                    .as_integer()
                    .unwrap()
                    .try_into()
                    .unwrap(),
                min_ttl_ms: limits["min_ttl_ms"]
                    .as_integer()
                    .unwrap()
                    .try_into()
                    .unwrap(),
                max_ttl_ms: limits["max_ttl_ms"]
                    .as_integer()
                    .unwrap()
                    .try_into()
                    .unwrap(),
            },
        }
    }

    fn sim_caps_with_all_verbs() -> CockpitCapabilities {
        let mut caps = SimCockpit::new().get_capabilities().unwrap();
        caps.verbs = CockpitRequest::capability_verbs()
            .into_iter()
            .map(ToOwned::to_owned)
            .collect();
        caps.events = body_toml_array("events")
            .into_iter()
            .map(ToOwned::to_owned)
            .collect();
        caps
    }

    fn sim_caps_without(without: &[&str]) -> CockpitCapabilities {
        let mut caps = sim_caps_with_all_verbs();
        caps.verbs.retain(|verb| !without.contains(&verb.as_str()));
        caps
    }

    fn sample_request_for(verb: &str) -> CockpitRequest {
        match verb {
            "ping" => CockpitRequest::Ping,
            "bootsel" => CockpitRequest::Bootsel,
            "restart_mpu" => CockpitRequest::RestartMpu,
            "restart_create" => CockpitRequest::RestartCreate,
            "reset_motherbrain" => CockpitRequest::ResetMotherbrain,
            "status" => CockpitRequest::GetStatus,
            "get_capabilities" => CockpitRequest::GetCapabilities,
            "get_events" => CockpitRequest::GetEvents { since_seq: 3 },
            "arm" => CockpitRequest::Arm,
            "disarm" => CockpitRequest::Disarm,
            "stop" => CockpitRequest::Stop,
            "estop" => CockpitRequest::EStop,
            "clear_estop" => CockpitRequest::ClearEStop,
            "clear_motion_queue" => CockpitRequest::ClearMotionQueue,
            "cmd_vel" => CockpitRequest::CmdVel {
                linear_mm_s: 10,
                angular_mrad_s: 20,
                ttl_ms: 300,
            },
            "drive_direct" => CockpitRequest::DriveDirect {
                left_mm_s: 10,
                right_mm_s: 11,
                ttl_ms: 300,
            },
            "drive_arc" => CockpitRequest::DriveArc {
                velocity_mm_s: 10,
                radius_mm: 200,
                ttl_ms: 300,
            },
            "drive_for" => CockpitRequest::DriveFor {
                distance_mm: 300,
                velocity_mm_s: 80,
                timeout_ms: 2_000,
            },
            "turn_by" => CockpitRequest::TurnBy {
                angle_mrad: 1_570,
                angular_mrad_s: 800,
                timeout_ms: 2_000,
            },
            "arc_for" => CockpitRequest::ArcFor {
                velocity_mm_s: 80,
                radius_mm: 250,
                duration_ms: 1_000,
            },
            "creep_until" => CockpitRequest::CreepUntil {
                velocity_mm_s: 40,
                angular_mrad_s: 0,
                timeout_ms: 1_000,
            },
            "scan_arc" => CockpitRequest::ScanArc {
                angle_mrad: 3_140,
                angular_mrad_s: 500,
                timeout_ms: 4_000,
            },
            "face_bearing" => CockpitRequest::FaceBearing {
                bearing_mrad: 100,
                max_angular_mrad_s: 500,
                tolerance_mrad: 35,
                ttl_ms: 300,
            },
            "track_bearing" => CockpitRequest::TrackBearing {
                bearing_mrad: 100,
                range_mm: 900,
                max_linear_mm_s: 120,
                max_angular_mrad_s: 500,
                stop_range_mm: 250,
                ttl_ms: 300,
            },
            "hold_heading" => CockpitRequest::HoldHeading {
                heading_error_mrad: 100,
                velocity_mm_s: 80,
                max_angular_mrad_s: 500,
                ttl_ms: 300,
            },
            "turn_to_heading" => CockpitRequest::TurnToHeading {
                heading_error_mrad: 100,
                angular_mrad_s: 500,
                tolerance_mrad: 35,
                timeout_ms: 2_000,
            },
            "dock_align" => CockpitRequest::DockAlign {
                bearing_mrad: 50,
                range_mm: 600,
                max_linear_mm_s: 80,
                max_angular_mrad_s: 500,
                stop_range_mm: 250,
                ttl_ms: 300,
            },
            "wall_follow" => CockpitRequest::WallFollow {
                distance_error_mm: 20,
                velocity_mm_s: 80,
                max_angular_mrad_s: 400,
                ttl_ms: 300,
            },
            "wiggle_align" => CockpitRequest::WiggleAlign {
                amplitude_mrad: 200,
                angular_mrad_s: 500,
                cycles: 2,
            },
            "bump_escape" => CockpitRequest::BumpEscape {
                direction: EscapeDirection::Either,
                backoff_mm_s: 80,
                turn_angular_mrad_s: 900,
            },
            "unstick" => CockpitRequest::Unstick {
                direction: EscapeDirection::Either,
                backoff_mm_s: 90,
                turn_angular_mrad_s: 900,
            },
            "cliff_guard" => CockpitRequest::CliffGuard { clear: false },
            "heartbeat_stop" => CockpitRequest::HeartbeatStop { timeout_ms: 900 },
            "request_sensors" => CockpitRequest::RequestSensors { packet_id: 0 },
            "stream_sensors" => CockpitRequest::StreamSensors {
                enabled: true,
                packet_id: 0,
                period_ms: 250,
            },
            "set_safety_policy" => CockpitRequest::SetSafetyPolicy {
                policy: SafetyPolicy {
                    bump: SafetyAction::Stop,
                    cliff: SafetyAction::Stop,
                    wheel_drop_latch: true,
                },
            },
            "song_define" => CockpitRequest::SongDefine {
                id: 1,
                tones: sample_tones(),
            },
            "song_play" => CockpitRequest::SongPlay { id: 1 },
            "define_chirp" => CockpitRequest::DefineChirp {
                feedback: FeedbackKind::Ok,
                tones: sample_tones(),
            },
            "play_feedback" => CockpitRequest::PlayFeedback {
                feedback: FeedbackKind::Ok,
            },
            "power_state" => CockpitRequest::PowerState {
                request: PowerStateRequest::Wake,
            },
            "calibrate_turn" => CockpitRequest::CalibrateTurn {
                angular_mrad_s: 500,
                duration_ms: 1_000,
            },
            "reset_odometry" => CockpitRequest::ResetOdometry,
            "zero_imu_orientation" => CockpitRequest::ZeroImuOrientation,
            "clear_imu_orientation" => CockpitRequest::ClearImuOrientation,
            "dock" => CockpitRequest::Dock,
            "set_lights" => CockpitRequest::SetLights {
                pattern: LightPattern::Status,
            },
            "set_mode" => CockpitRequest::SetMode {
                mode: CreateOiMode::Safe,
            },
            other => panic!("missing sample for {other}"),
        }
    }

    fn sample_tones() -> Vec<SongTone> {
        vec![SongTone {
            note: 72,
            duration_64ths: 8,
        }]
    }

    #[test]
    fn production_possession_renews_and_replaces_lease() {
        let ready = establish_session(SimCockpit::new(), hello(), None).unwrap();
        let mut possession = MotherbrainPossession::acquire(ready, 1_000).unwrap();
        let first = possession.snapshot();
        possession.lease_acquired_at = Instant::now() - Duration::from_millis(800);
        possession.maintain().unwrap();
        let renewed = possession.snapshot();
        assert!(renewed.possessed);
        assert!(renewed.lease_generation > first.lease_generation);
        assert_ne!(renewed.lease_id, first.lease_id);
    }

    #[test]
    fn production_possession_renews_long_lease_on_short_cadence() {
        let ready = establish_session(SimCockpit::new(), hello(), None).unwrap();
        let mut possession = MotherbrainPossession::acquire(ready, 60_000).unwrap();
        let first = possession.snapshot();
        possession.lease_acquired_at =
            Instant::now() - Duration::from_millis(POSSESSION_LEASE_RENEW_INTERVAL_MS as u64 + 1);

        possession.maintain().unwrap();

        let renewed = possession.snapshot();
        assert!(renewed.possessed);
        assert!(renewed.lease_generation > first.lease_generation);
        assert_ne!(renewed.lease_id, first.lease_id);
    }

    #[test]
    fn production_possession_retries_transient_busy_commands() {
        let cockpit = BusyOnceCockpit {
            inner: SimCockpit::new(),
            busy_remaining: 0,
            attempts: 0,
        };
        let ready = establish_session(cockpit, hello(), None).unwrap();
        let mut possession = MotherbrainPossession::acquire(ready, 1_000).unwrap();
        possession.session.connector_mut().busy_remaining = 1;
        let attempts_before = possession.session.connector_mut().attempts;

        possession
            .execute(CockpitRequest::PlayFeedback {
                feedback: FeedbackKind::Ok,
            })
            .unwrap();

        assert_eq!(
            possession.session.connector_mut().attempts,
            attempts_before + 2
        );
        assert!(possession.snapshot().possessed);
    }

    #[test]
    fn production_possession_renews_and_retries_stale_control_lease() {
        let cockpit = StaleLeaseOnceCockpit {
            inner: SimCockpit::new(),
            invalid_remaining: 0,
            attempts: 0,
        };
        let ready = establish_session(cockpit, hello(), None).unwrap();
        let mut possession = MotherbrainPossession::acquire(ready, 60_000).unwrap();
        possession.session.connector_mut().invalid_remaining = 1;
        let first = possession.snapshot();

        possession
            .execute(CockpitRequest::PlayFeedback {
                feedback: FeedbackKind::Ok,
            })
            .unwrap();

        let renewed = possession.snapshot();
        assert_eq!(possession.session.connector_mut().attempts, 2);
        assert!(renewed.possessed);
        assert!(renewed.lease_generation > first.lease_generation);
        assert_ne!(renewed.lease_id, first.lease_id);
    }

    #[test]
    fn exorcize_closes_gate_only_after_stop_is_acknowledged() {
        let cockpit = StopRejectingCockpit {
            inner: SimCockpit::new(),
            reject_stop: false,
            disarm_requests: 0,
        };
        let ready = establish_session(cockpit, hello(), None).unwrap();
        let mut possession = MotherbrainPossession::acquire(ready, 1_000).unwrap();
        possession.session.connector_mut().reject_stop = true;

        assert!(possession.exorcize().is_err());
        assert!(!possession.snapshot().possessed);
        assert_eq!(possession.session.connector_mut().disarm_requests, 0);
    }

    #[test]
    fn exorcize_stops_without_disarming_create_oi() {
        let cockpit = StopRejectingCockpit {
            inner: SimCockpit::new(),
            reject_stop: false,
            disarm_requests: 0,
        };
        let ready = establish_session(cockpit, hello(), None).unwrap();
        let mut possession = MotherbrainPossession::acquire(ready, 1_000).unwrap();

        possession.exorcize().unwrap();

        assert!(!possession.snapshot().possessed);
        assert_eq!(possession.session.connector_mut().disarm_requests, 0);
    }

    #[test]
    fn renewal_failure_closes_motor_gate() {
        let ready = establish_session(SimCockpit::new(), hello(), None).unwrap();
        let mut possession = MotherbrainPossession::acquire(ready, 1_000).unwrap();
        possession.lease_acquired_at = Instant::now() - Duration::from_millis(800);
        possession
            .session
            .connector_mut()
            .handshake(hello().new_attempt())
            .unwrap();
        assert!(possession.maintain().is_err());
        assert!(!possession.snapshot().possessed);
        assert!(possession
            .execute(CockpitRequest::CmdVel {
                linear_mm_s: 1,
                angular_mrad_s: 0,
                ttl_ms: 100,
            })
            .is_err());
    }

    #[test]
    fn production_possession_clamps_motion_and_hides_oi() {
        let ready = establish_session(SimCockpit::new(), hello(), None).unwrap();
        let mut possession = MotherbrainPossession::acquire(ready, 1_000).unwrap();
        possession
            .execute(CockpitRequest::CmdVel {
                linear_mm_s: 500,
                angular_mrad_s: 5_000,
                ttl_ms: 10_000,
            })
            .unwrap();
        let events = possession.session.poll_events().unwrap();
        let motion = events
            .events
            .iter()
            .find(|event| event.kind == CockpitEventKind::MotionRequested)
            .unwrap();
        assert_eq!(motion.a, pack_i16_pair(50, 500));
        assert_eq!(motion.b, 300);
        assert!(possession
            .execute(CockpitRequest::SetMode {
                mode: CreateOiMode::Full,
            })
            .is_err());
    }
}
