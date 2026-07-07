use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs, UdpSocket};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serialport::SerialPort;
use thiserror::Error;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{connect, Message, WebSocket};

pub type Result<T> = std::result::Result<T, CockpitError>;
const DEFAULT_SIM_EVENT_CAPACITY: usize = 32;

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
}

pub trait Cockpit {
    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse>;

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
}

impl CockpitCapabilities {
    pub fn supports(&self, verb: &str) -> bool {
        self.verbs.iter().any(|candidate| candidate == verb)
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
    CommandRejected,
    CommandInterrupted,
}

impl SafeStopReason {
    pub fn from_event(event: &CockpitEvent) -> Option<Self> {
        match event.kind {
            CockpitEventKind::SafetyTripped => Some(Self::SafetyTripped),
            CockpitEventKind::HeartbeatExpired => Some(Self::HeartbeatExpired),
            CockpitEventKind::EStopLatched => Some(Self::EStopLatched),
            CockpitEventKind::CommandRejected => Some(Self::CommandRejected),
            CockpitEventKind::CommandInterrupted => Some(Self::CommandInterrupted),
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
        }
    }

    fn from_json(raw: &str, value: &serde_json::Value) -> Self {
        let sensors = value.get("create_sensors");
        let safety_tripped = sensors.map(|sensors| {
            json_bool_value(sensors, "bump_left").unwrap_or(false)
                || json_bool_value(sensors, "bump_right").unwrap_or(false)
                || json_bool_value(sensors, "wheel_drop").unwrap_or(false)
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
            active_motion: json_str_value(value, "current_command").map(|command| command == "drive"),
            event_next_seq: json_u32_value(value, "event_next_seq"),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CockpitRequest {
    Ping,
    Bootsel,
    GetStatus,
    GetCapabilities,
    GetEvents { since_seq: u32 },
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
    CliffGuard { clear: bool },
    HeartbeatStop { timeout_ms: u32 },
    RequestSensors { packet_id: u8 },
    StreamSensors {
        enabled: bool,
        packet_id: u8,
        period_ms: u32,
    },
    SetSafetyPolicy { policy: SafetyPolicy },
    ClearMotionQueue,
    DefineChirp {
        feedback: FeedbackKind,
        tones: Vec<SongTone>,
    },
    PlayFeedback { feedback: FeedbackKind },
    PowerState { request: PowerStateRequest },
    CalibrateTurn {
        angular_mrad_s: i16,
        duration_ms: u32,
    },
    ResetOdometry,
    SetMode { mode: CreateOiMode },
    SongDefine { id: u8, tones: Vec<SongTone> },
    SongPlay { id: u8 },
    Dock,
    SetLights { pattern: LightPattern },
}

impl CockpitRequest {
    pub fn apply<C: Cockpit>(&self, client: &mut C) -> Result<CockpitResponse> {
        match self {
            Self::Ping => client.ping().map(|()| CockpitResponse::Accepted),
            Self::Bootsel => client.bootsel().map(|()| CockpitResponse::Accepted),
            Self::GetStatus => Ok(CockpitResponse::Status(client.get_status()?)),
            Self::GetCapabilities => {
                Ok(CockpitResponse::Capabilities(client.get_capabilities()?))
            }
            Self::GetEvents { since_seq } => {
                Ok(CockpitResponse::Events(client.get_events_since(*since_seq)?))
            }
            Self::Arm => client.arm().map(|()| CockpitResponse::Accepted),
            Self::Disarm => client.disarm().map(|()| CockpitResponse::Accepted),
            Self::Stop => client.stop().map(|()| CockpitResponse::Accepted),
            Self::EStop => client.estop().map(|()| CockpitResponse::Accepted),
            Self::ClearEStop => client
                .clear_estop()
                .map(|()| CockpitResponse::Accepted),
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
                .hold_heading(*heading_error_mrad, *velocity_mm_s, *max_angular_mrad_s, *ttl_ms)
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
                .wall_follow(*distance_error_mm, *velocity_mm_s, *max_angular_mrad_s, *ttl_ms)
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
            Self::ResetOdometry => client
                .reset_odometry()
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
            Self::SetMode { mode } => client
                .set_mode(*mode)
                .map(|()| CockpitResponse::Accepted),
            Self::SongDefine { id, tones } => client
                .song_define(*id, tones)
                .map(|()| CockpitResponse::Accepted),
            Self::SongPlay { id } => client
                .song_play(*id)
                .map(|()| CockpitResponse::Accepted),
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

    fn needs_seq(&self) -> bool {
        !matches!(
            self,
            Self::Ping
                | Self::Bootsel
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
            Self::GetStatus => format!("STATUS {seq}\n"),
            Self::GetCapabilities => format!("GET_CAPABILITIES {seq}\n"),
            Self::GetEvents { since_seq } => format!("GET_EVENTS {seq} {since_seq}\n"),
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
            Self::SetMode { mode } => format!("SET_MODE {seq} {}\n", mode.as_str()),
            Self::SongDefine { id, tones } => {
                format!("SONG_DEFINE {seq} {id}{}\n", compact_tones(tones))
            }
            Self::SongPlay { id } => format!("SONG_PLAY {seq} {id}\n"),
            Self::Dock => format!("DOCK {seq}\n"),
            Self::SetLights { pattern } => format!("SET_LIGHTS {seq} {}\n", pattern.as_str()),
        }
    }
    pub fn to_bridge_json(&self, command_id: u32) -> Result<String> {
        self.to_firmware_json(command_id)
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CockpitResponse {
    Accepted,
    Rejected { message: String },
    Status(CockpitStatus),
    Capabilities(CockpitCapabilities),
    Events(EventBatch),
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
}

impl<C: Cockpit> SafeCockpit<C> {
    pub fn new(client: C) -> Self {
        Self::with_policy(client, AgentPolicy::default())
    }

    pub fn with_policy(client: C, policy: AgentPolicy) -> Self {
        Self {
            client,
            cursor: EventCursor::new(),
            policy,
        }
    }

    pub fn client_mut(&mut self) -> &mut C {
        &mut self.client
    }

    pub fn refresh_status(&mut self) -> Result<StatusSummary> {
        Ok(self.client.get_status()?.summary())
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
        self.client.heartbeat_stop(self.policy.heartbeat_timeout_ms)?;
        self.client
            .cmd_vel(linear_mm_s, angular_mrad_s, self.policy.motion_ttl_ms)?;
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
    HeartbeatExpired,
    EStopLatched,
    EStopCleared,
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
            "heartbeat_expired" => Self::HeartbeatExpired,
            "estop_latched" => Self::EStopLatched,
            "estop_cleared" => Self::EStopCleared,
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
    active_cmd_vel: Option<SimTimedAction>,
    heartbeat_stop_at_ms: Option<u32>,
    odometry_reset_count: u32,
}

impl SimCockpit {
    pub fn new() -> Self {
        let mut sim = Self {
            capabilities: CockpitCapabilities {
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
        sim.push_event(CockpitEventKind::Boot, 0, 0, 0);
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
        self.push_event(CockpitEventKind::SafetyTripped, 1, 0, 0);
        self.push_event(CockpitEventKind::MotionStopped, 0, 0, 0);
    }

    pub fn odometry_reset_count(&self) -> u32 {
        self.odometry_reset_count
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
        match request {
            CockpitRequest::GetStatus => Ok(CockpitResponse::Status(self.get_status()?)),
            CockpitRequest::GetCapabilities => {
                Ok(CockpitResponse::Capabilities(self.get_capabilities()?))
            }
            CockpitRequest::GetEvents { since_seq } => {
                Ok(CockpitResponse::Events(self.get_events_since(since_seq)?))
            }
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
            _ => {
                let id = self.accept_command();
                self.complete_command(id);
                Ok(CockpitResponse::Accepted)
            }
        }
    }

    fn get_status(&mut self) -> Result<CockpitStatus> {
        self.complete_due_cmd_vel();
        self.expire_heartbeat_if_due();
        Ok(CockpitStatus {
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
        let id = self.accept_command();
        self.estop_latched = false;
        self.safety_tripped = false;
        self.push_event(CockpitEventKind::EStopCleared, 0, 0, 0);
        self.push_event(CockpitEventKind::SafetyCleared, 4, 0, 0);
        self.complete_command(id);
        Ok(())
    }

    fn cmd_vel(&mut self, linear_mm_s: i16, angular_mrad_s: i16, ttl_ms: u32) -> Result<()> {
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

pub struct HttpCockpit {
    host: String,
    next_command_id: u32,
    timeout: Duration,
}

impl HttpCockpit {
    pub fn connect(host: impl Into<String>) -> Self {
        Self {
            host: host.into(),
            next_command_id: 1,
            timeout: Duration::from_millis(750),
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

    fn post_command(&mut self, body: &str) -> Result<String> {
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
            "POST /command HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            self.host,
            body.len(),
            body
        )?;
        stream.flush()?;
        let mut response = String::new();
        stream.read_to_string(&mut response)?;
        http_body(&response)
    }
}

impl Cockpit for HttpCockpit {
    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        let command_id = self.command_id();
        let body = request.to_firmware_json(command_id)?;
        let response = self.post_command(&body)?;
        parse_json_cockpit_response(command_id, &request, &response)
    }
}

pub struct WebSocketCockpit {
    socket: WebSocket<MaybeTlsStream<TcpStream>>,
    next_command_id: u32,
}

impl WebSocketCockpit {
    pub fn connect_url(url: &str) -> Result<Self> {
        let (socket, _) = connect(url)?;
        Ok(Self {
            socket,
            next_command_id: 1,
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
}

pub struct UdpCockpit {
    socket: UdpSocket,
    brainstem: SocketAddr,
    next_seq: u32,
    timeout: Duration,
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

impl Cockpit for UdpCockpit {
    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        let seq = self.seq();
        let response = self.request(request.to_compact_line(seq))?;
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
}

pub const DEFAULT_UART_BAUD_RATE: u32 = 115_200;
pub const DEFAULT_UART_TIMEOUT: Duration = Duration::from_millis(750);
pub const DEFAULT_UART_MAX_RESPONSE_LEN: usize = 2048;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct UartCockpitConfig {
    pub path: PathBuf,
    pub baud_rate: u32,
    pub timeout: Duration,
    pub max_response_len: usize,
}

impl UartCockpitConfig {
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

pub struct UartCockpit {
    port: Box<dyn SerialPort>,
    next_seq: u32,
    timeout: Duration,
    max_response_len: usize,
}

impl UartCockpit {
    pub fn connect(path: impl AsRef<Path>) -> Result<Self> {
        Self::connect_with_config(UartCockpitConfig::new(path.as_ref()))
    }

    pub fn connect_with_config(config: UartCockpitConfig) -> Result<Self> {
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

impl Cockpit for UartCockpit {
    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        let seq = self.seq();
        let response = self.request(request.to_compact_line(seq))?;
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

fn expect_ok(seq: u32, response: &str) -> Result<()> {
    let mut parts = response.split_ascii_whitespace();
    match (
        parts.next(),
        parts.next().and_then(|value| value.parse::<u32>().ok()),
    ) {
        (Some("OK"), Some(response_seq)) if response_seq == seq => Ok(()),
        _ => Err(CockpitError::BadResponse(response.to_owned())),
    }
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
        CockpitRequest::GetCapabilities => {
            Ok(CockpitResponse::Capabilities(parse_capabilities(seq, response)?))
        }
        CockpitRequest::GetEvents { since_seq } => {
            Ok(CockpitResponse::Events(parse_events(seq, *since_seq, response)?))
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
        CockpitRequest::GetEvents { since_seq } => {
            Ok(CockpitResponse::Events(parse_json_events(*since_seq, &value)?))
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
        body_kind: json_str_value(value, "body_kind").unwrap_or_default().to_owned(),
        drive: json_str_value(value, "drive").unwrap_or_default().to_owned(),
        verbs: json_string_array(value, "verbs"),
        sensors: json_string_array(value, "sensors"),
        outputs: json_string_array(value, "outputs"),
        safety: json_string_array(value, "safety"),
        events: json_string_array(value, "events"),
    })
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
    if !head.starts_with("HTTP/1.1 200") && !head.starts_with("HTTP/1.0 200") {
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

fn json_u32_value(value: &serde_json::Value, key: &str) -> Option<u32> {
    value.get(key)?.as_u64().and_then(|value| value.try_into().ok())
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
            object.insert("wheel_drop_latch".to_owned(), policy.wheel_drop_latch.into());
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn simulator_capabilities_round_trip() {
        let mut sim = SimCockpit::new();
        let caps = sim.get_capabilities().unwrap();
        assert_eq!(caps.body_kind, "sim_create_oi");
        assert_eq!(caps.drive, "differential");
        assert!(caps.verbs.contains(&"cmd_vel".to_owned()));
        assert!(caps.events.contains(&"safety_tripped".to_owned()));
    }

    #[test]
    fn cockpit_request_covers_public_firmware_verbs() {
        let cockpit_verbs: BTreeSet<_> = sample_cockpit_requests()
            .into_iter()
            .map(|(verb, _, _)| verb)
            .filter(|verb| *verb != "bootsel")
            .collect();
        let firmware_verbs: BTreeSet<_> = [
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
        ]
        .into_iter()
        .collect();
        assert_eq!(cockpit_verbs, firmware_verbs);
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
            assert_eq!(value.get("command_id").and_then(serde_json::Value::as_u64), Some(7));
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
        let value: serde_json::Value = serde_json::from_str(&song.to_firmware_json(2).unwrap()).unwrap();
        assert_eq!(value["tones"], "72:8");
    }

    #[test]
    fn simulator_event_cursor_happy_path() {
        let mut sim = SimCockpit::new();
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
        let mut sim = SimCockpit::new().with_event_capacity(3);
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
        let mut sim = SimCockpit::new();
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
        let mut sim = SimCockpit::new();
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
        let mut sim = SimCockpit::new();
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
        let mut sim = SimCockpit::new();
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
    fn simulator_safety_tripped_stops_motion_and_rejects_motion() {
        let mut sim = SimCockpit::new();
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
        let mut sim = SimCockpit::new();
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
            Err(CockpitError::BadResponse(_))
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
            ("status", "status", "STATUS"),
            ("get_capabilities", "get_capabilities", "GET_CAPABILITIES"),
            ("get_events", "get_events", "GET_EVENTS"),
            ("arm", "arm", "ARM"),
            ("disarm", "disarm", "DISARM"),
            ("stop", "stop", "STOP"),
            ("estop", "estop", "ESTOP"),
            ("clear_estop", "clear_estop", "CLEAR_ESTOP"),
            ("clear_motion_queue", "clear_motion_queue", "CLEAR_MOTION_QUEUE"),
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
            ("request_sensors", "request_sensors", "REQUEST_SENSORS"),
            ("stream_sensors", "stream_sensors", "STREAM_SENSORS"),
            ("set_safety_policy", "set_safety_policy", "SET_SAFETY_POLICY"),
            ("song_define", "song_define", "SONG_DEFINE"),
            ("song_play", "song_play", "SONG_PLAY"),
            ("define_chirp", "define_chirp", "DEFINE_CHIRP"),
            ("play_feedback", "play_feedback", "PLAY_FEEDBACK"),
            ("power_state", "power_state", "POWER_STATE"),
            ("calibrate_turn", "calibrate_turn", "CALIBRATE_TURN"),
            ("reset_odometry", "reset_odometry", "RESET_ODOMETRY"),
            ("dock", "dock", "DOCK"),
            ("set_lights", "set_lights", "SET_LIGHTS"),
            ("set_mode", "set_mode", "SET_MODE"),
        ]
    }

    fn sample_request_for(verb: &str) -> CockpitRequest {
        match verb {
            "ping" => CockpitRequest::Ping,
            "bootsel" => CockpitRequest::Bootsel,
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
}
