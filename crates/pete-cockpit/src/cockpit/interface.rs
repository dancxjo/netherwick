const DEFAULT_SIM_EVENT_CAPACITY: usize = 32;
const POSSESSION_BUMP_ESCAPE_HEARTBEAT_MARGIN_MS: u32 = 1_000;
const POSSESSION_LEASE_RENEW_INTERVAL_MS: u32 = 1_000;
const POSSESSION_BUSY_RETRY_ATTEMPTS: usize = 300;
const POSSESSION_BUSY_RETRY_DELAY: Duration = Duration::from_millis(10);
const LEGACY_BUMP_ESCAPE_BACKOFF_DURATION_MS: u32 = 900;
const LEGACY_BUMP_ESCAPE_TURN_ANGLE_MRAD: u32 = 1_571;

const fn legacy_bump_escape_turn_duration_ms(turn_angular_mrad_s: i16) -> u32 {
    let angular_mrad_s = if turn_angular_mrad_s < 0 {
        -(turn_angular_mrad_s as i32)
    } else {
        turn_angular_mrad_s as i32
    } as u32;
    if angular_mrad_s == 0 {
        return 0;
    }
    LEGACY_BUMP_ESCAPE_TURN_ANGLE_MRAD
        .saturating_mul(1_000)
        .saturating_add(angular_mrad_s - 1)
        / angular_mrad_s
}

const fn legacy_bump_escape_duration_ms(turn_angular_mrad_s: i16) -> u32 {
    LEGACY_BUMP_ESCAPE_BACKOFF_DURATION_MS
        .saturating_add(legacy_bump_escape_turn_duration_ms(turn_angular_mrad_s))
}

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
    #[error("motion stopped by {reasons:?}")]
    MotionStopped { reasons: Vec<SafeStopReason> },
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

    /// Whether this cockpit already refreshes the motion heartbeat before
    /// forwarding velocity commands.
    fn manages_motion_heartbeat(&self) -> bool {
        false
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

    fn clear_safety_latch(&mut self, kind: SafetyLatchKind) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::ClearSafetyLatch { latch: kind })?)
    }

    fn careful_mode(&mut self, ttl_ms: u32) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::CarefulMode { ttl_ms })?)
    }

    fn escape_motion(
        &mut self,
        hazard: SafetyLatchKind,
        hazard_generation: u32,
        linear_mm_s: i16,
        angular_mrad_s: i16,
        ttl_ms: u32,
    ) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::EscapeMotion {
            hazard,
            hazard_generation,
            linear_mm_s,
            angular_mrad_s,
            ttl_ms,
        })?)
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

    fn set_audio_silent(&mut self, silent: bool) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::SetAudioSilent { silent })?)
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

    fn orientation_probe(&mut self, angular_mrad_s: i16, duration_ms: u32) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::OrientationProbe {
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

    fn manages_motion_heartbeat(&self) -> bool {
        (**self).manages_motion_heartbeat()
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

