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
#[serde(rename_all = "snake_case")]
pub enum SafetyLatchKind {
    Bump,
    Cliff,
    WheelDrop,
    Heartbeat,
    Tilt,
    Impact,
    Charging,
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

impl SafetyLatchKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Bump => "bump",
            Self::Cliff => "cliff",
            Self::WheelDrop => "wheel_drop",
            Self::Heartbeat => "heartbeat",
            Self::Tilt => "tilt",
            Self::Impact => "impact",
            Self::Charging => "charging",
        }
    }

    fn from_str(value: &str) -> Option<Self> {
        match value {
            "bump" => Some(Self::Bump),
            "cliff" => Some(Self::Cliff),
            "wheel_drop" => Some(Self::WheelDrop),
            "heartbeat" => Some(Self::Heartbeat),
            "tilt" => Some(Self::Tilt),
            "impact" => Some(Self::Impact),
            "charging" => Some(Self::Charging),
            _ => None,
        }
    }

    fn from_event_code(code: u32) -> Option<Self> {
        match code {
            1 => Some(Self::Bump),
            2 => Some(Self::Cliff),
            3 => Some(Self::WheelDrop),
            5 => Some(Self::Heartbeat),
            6 => Some(Self::Tilt),
            7 => Some(Self::Impact),
            8 => Some(Self::Charging),
            _ => None,
        }
    }
}

fn safety_latch_kind_code(kind: SafetyLatchKind) -> u32 {
    match kind {
        SafetyLatchKind::Bump => 1,
        SafetyLatchKind::Cliff => 2,
        SafetyLatchKind::WheelDrop => 3,
        SafetyLatchKind::Heartbeat => 5,
        SafetyLatchKind::Tilt => 6,
        SafetyLatchKind::Impact => 7,
        SafetyLatchKind::Charging => 8,
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
    /// Whether motion supervision survives failure of the possessor's host.
    ///
    /// `None` preserves compatibility with pre-migration brainstem contracts;
    /// physical launchers resolve that legacy value from their selected backend.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub independent_watchdog: Option<bool>,
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
            | CockpitRequest::CalibrateTurn { angular_mrad_s, .. }
            | CockpitRequest::OrientationProbe { angular_mrad_s, .. } => {
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
        let tolerated_advertised_verbs = tolerated_advertised_verbs();
        let missing_verbs = self
            .capabilities
            .verbs
            .iter()
            .filter(|verb| {
                !modeled_verbs.iter().any(|modeled| modeled == *verb)
                    && !tolerated_advertised_verbs
                        .iter()
                        .any(|tolerated| tolerated == &verb.as_str())
            })
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

    pub fn command_rejection(&self) -> Option<CommandRejection> {
        if self.kind != CockpitEventKind::CommandRejected {
            return None;
        }
        Some(CommandRejection {
            command_id: self.a,
            command_seq: self.b,
            command_code: (self.c >> 8) as u8,
            reason: CommandRejectReason::from_code(self.c as u8),
        })
    }

    pub fn contact_withdrawal(&self) -> Option<ContactWithdrawalEvent> {
        ContactWithdrawalEvent::from_event(self)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContactWithdrawalOutcome {
    Completed,
    SafetyPreempted,
    Failed,
    Unknown(u8),
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum ContactWithdrawalEvent {
    Started {
        contact_bits: u8,
        repeated_count: u8,
        preempted_command_id: u32,
        reverse_speed_mm_s: u16,
        maximum_duration_ms: u16,
    },
    Completed {
        outcome: ContactWithdrawalOutcome,
        dominating_safety: Option<SafetyLatchKind>,
        final_stopped: bool,
        observed_displacement_mm: i32,
        elapsed_ms: u32,
    },
}

impl ContactWithdrawalEvent {
    pub fn from_event(event: &CockpitEvent) -> Option<Self> {
        match event.kind {
            CockpitEventKind::ContactWithdrawalStarted => Some(Self::Started {
                contact_bits: (event.a & 0b11) as u8,
                repeated_count: ((event.a >> 8) & 0xff) as u8,
                preempted_command_id: event.b,
                reverse_speed_mm_s: (event.c & 0xffff) as u16,
                maximum_duration_ms: (event.c >> 16) as u16,
            }),
            CockpitEventKind::ContactWithdrawalCompleted => Some(Self::Completed {
                outcome: match event.a as u8 {
                    1 => ContactWithdrawalOutcome::Completed,
                    2 => ContactWithdrawalOutcome::SafetyPreempted,
                    3 => ContactWithdrawalOutcome::Failed,
                    code => ContactWithdrawalOutcome::Unknown(code),
                },
                dominating_safety: SafetyLatchKind::from_event_code(((event.a >> 8) & 0xff) as u32),
                final_stopped: event.a & (1 << 16) != 0,
                observed_displacement_mm: event.b as i32,
                elapsed_ms: event.c,
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct CommandRejection {
    pub command_id: u32,
    pub command_seq: u32,
    pub command_code: u8,
    pub reason: CommandRejectReason,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SafeStopReason {
    SafetyTripped { latch: Option<SafetyLatchKind> },
    HeartbeatExpired,
    EStopLatched,
}

impl SafeStopReason {
    pub fn from_event(event: &CockpitEvent) -> Option<Self> {
        match event.kind {
            CockpitEventKind::SafetyTripped => Some(Self::SafetyTripped {
                latch: SafetyLatchKind::from_event_code(event.a),
            }),
            CockpitEventKind::HeartbeatExpired => Some(Self::HeartbeatExpired),
            CockpitEventKind::EStopLatched => Some(Self::EStopLatched),
            _ => None,
        }
    }
}

/// Directional Home Base evidence encoded by the Create OI IR character.
///
/// The dock transmits three independently recognizable components. Treating
/// them as a cue instead of an opaque byte lets higher layers follow the
/// red/green gradient without moving any safety policy out of the Brainstem.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct DockIrCue {
    pub red_buoy: bool,
    pub green_buoy: bool,
    pub force_field: bool,
}

impl DockIrCue {
    pub fn from_character(character: u8) -> Option<Self> {
        let cue = match character {
            242 => Self {
                red_buoy: false,
                green_buoy: false,
                force_field: true,
            },
            244 => Self {
                red_buoy: false,
                green_buoy: true,
                force_field: false,
            },
            246 => Self {
                red_buoy: false,
                green_buoy: true,
                force_field: true,
            },
            248 => Self {
                red_buoy: true,
                green_buoy: false,
                force_field: false,
            },
            250 => Self {
                red_buoy: true,
                green_buoy: false,
                force_field: true,
            },
            252 => Self {
                red_buoy: true,
                green_buoy: true,
                force_field: false,
            },
            254 => Self {
                red_buoy: true,
                green_buoy: true,
                force_field: true,
            },
            _ => return None,
        };
        Some(cue)
    }

    /// Signed turn hint in Pete's bearing convention: positive is left.
    pub fn bearing_hint_rad(self) -> f32 {
        match (self.red_buoy, self.green_buoy) {
            (true, false) => 0.35,
            (false, true) => -0.35,
            _ => 0.0,
        }
    }

    /// Steering command that follows the dock gradient toward both buoys.
    pub fn steering_mrad_s(self, correction_mrad_s: i16) -> i16 {
        let correction = correction_mrad_s.abs();
        match (self.red_buoy, self.green_buoy) {
            (true, false) => correction,
            (false, true) => -correction,
            _ => 0,
        }
    }

    /// A strong charger-visible cue, deliberately below contact certainty.
    pub fn visible_score(self) -> f32 {
        if self.red_buoy || self.green_buoy {
            0.85
        } else {
            0.65
        }
    }

    /// Force-field evidence raises proximity belief but never implies contact.
    pub fn near_score(self) -> f32 {
        if self.force_field {
            0.55
        } else {
            0.30
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct StatusSummary {
    pub raw: String,
    /// Brainstem monotonic clock at status generation. This is not a host timestamp.
    #[serde(default)]
    pub uptime_ms: Option<u32>,
    /// Firmware clock epoch. The host additionally advances its mapped epoch on reconnect
    /// or detected uptime regression because the RP2040 value restarts at zero after boot.
    #[serde(default)]
    pub clock_epoch: Option<u32>,
    pub runtime_state: Option<String>,
    pub armed: Option<bool>,
    pub estop_latched: Option<bool>,
    pub safety_tripped: Option<bool>,
    pub safety_latch_kind: Option<SafetyLatchKind>,
    #[serde(default)]
    pub safety_hazard_generation: Option<u32>,
    #[serde(default)]
    pub careful_mode_active: Option<bool>,
    #[serde(default)]
    pub careful_mode_remaining_ms: Option<u32>,
    #[serde(default)]
    pub audio_silent: Option<bool>,
    #[serde(default)]
    pub audio_last_requested_cue: Option<String>,
    #[serde(default)]
    pub audio_last_played_cue: Option<String>,
    #[serde(default)]
    pub audio_last_playback_timestamp_ms: Option<u32>,
    #[serde(default)]
    pub audio_suppressed_by_silent_count: Option<u32>,
    #[serde(default)]
    pub audio_dropped_or_replaced_count: Option<u32>,
    pub active_motion: Option<bool>,
    pub event_next_seq: Option<u32>,
    pub body_packet_count: Option<u32>,
    #[serde(default)]
    pub body_packet_timestamp_ms: Option<u32>,
    pub body_packet_age_ms: Option<u32>,
    pub body_packet_complete: Option<bool>,
    #[serde(default)]
    pub infrared_character: Option<u8>,
    pub contact: ContactSummary,
    pub battery: BatterySummary,
    pub odometry: OdometrySummary,
    pub imu: ImuSummary,
}

impl StatusSummary {
    pub fn has_fresh_complete_body_packet(&self, max_age_ms: u32) -> bool {
        self.body_packet_complete == Some(true)
            && self
                .body_packet_age_ms
                .is_some_and(|age_ms| age_ms <= max_age_ms)
    }

    pub fn from_raw(raw: &str) -> Self {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) {
            return Self::from_json(raw, &value);
        }
        let packet_count = number_for(raw, "create_body_packets");
        let imu = ImuSummary::from_raw(raw);
        let inferred_imu_latch = inferred_imu_safety_latch(&imu);
        let packet_age_ms = match (
            number_for(raw, "uptime_ms"),
            number_for(raw, "create_last_body_packet_ms"),
            packet_count,
        ) {
            (Some(uptime_ms), Some(packet_ms), Some(count)) if count > 0 => {
                Some(uptime_ms.wrapping_sub(packet_ms))
            }
            _ => None,
        };
        Self {
            raw: raw.to_owned(),
            uptime_ms: number_for(raw, "uptime_ms"),
            clock_epoch: number_for(raw, "clock_epoch"),
            runtime_state: value_for(raw, "runtime").map(ToOwned::to_owned),
            armed: bool_for(raw, "armed"),
            estop_latched: bool_for(raw, "estop"),
            safety_tripped: bool_for(raw, "safety_tripped").or(inferred_imu_latch.map(|_| true)),
            safety_latch_kind: value_for(raw, "safety_latch_kind")
                .and_then(SafetyLatchKind::from_str)
                .or(inferred_imu_latch),
            safety_hazard_generation: number_for(raw, "safety_hazard_generation"),
            careful_mode_active: bool_for(raw, "careful_mode"),
            careful_mode_remaining_ms: number_for(raw, "careful_remaining_ms"),
            audio_silent: bool_for(raw, "audio_silent"),
            audio_last_requested_cue: value_for(raw, "audio_last_requested").map(ToOwned::to_owned),
            audio_last_played_cue: value_for(raw, "audio_last_played").map(ToOwned::to_owned),
            audio_last_playback_timestamp_ms: number_for(raw, "audio_last_playback_ms"),
            audio_suppressed_by_silent_count: number_for(raw, "audio_suppressed"),
            audio_dropped_or_replaced_count: number_for(raw, "audio_dropped"),
            active_motion: bool_for(raw, "active_cmd_vel"),
            event_next_seq: number_for(raw, "event_next_seq"),
            body_packet_count: packet_count,
            body_packet_timestamp_ms: number_for(raw, "create_last_body_packet_ms"),
            body_packet_age_ms: packet_age_ms,
            body_packet_complete: Some(packet_count.unwrap_or(0) > 0),
            infrared_character: number_for(raw, "ir_byte").and_then(|value| value.try_into().ok()),
            contact: ContactSummary::from_raw(raw),
            battery: BatterySummary::from_raw(raw),
            odometry: OdometrySummary::from_raw(raw),
            imu,
        }
    }

    fn from_json(raw: &str, value: &serde_json::Value) -> Self {
        let sensors = value.get("create_sensors");
        let audio = value.get("audio");
        let packet_count =
            sensors.and_then(|sensors| json_u32_value(sensors, "complete_packet_count"));
        let packet_age_ms = match (
            json_u32_value(value, "uptime_ms"),
            sensors
                .and_then(|sensors| json_u32_value(sensors, "last_complete_packet_timestamp_ms")),
            packet_count,
        ) {
            (Some(uptime_ms), Some(packet_ms), Some(count)) if count > 0 => {
                Some(uptime_ms.wrapping_sub(packet_ms))
            }
            _ => None,
        };
        let sensor_safety_tripped = sensors.map(|sensors| {
            json_bool_value(sensors, "wheel_drop").unwrap_or(false)
                || json_bool_value(sensors, "cliff_left").unwrap_or(false)
                || json_bool_value(sensors, "cliff_front_left").unwrap_or(false)
                || json_bool_value(sensors, "cliff_front_right").unwrap_or(false)
                || json_bool_value(sensors, "cliff_right").unwrap_or(false)
        });
        Self {
            raw: raw.to_owned(),
            uptime_ms: json_u32_value(value, "uptime_ms"),
            clock_epoch: json_u32_value(value, "clock_epoch"),
            runtime_state: json_str_value(value, "current_runtime_state")
                .or_else(|| json_str_value(value, "runtime"))
                .map(ToOwned::to_owned),
            armed: json_str_value(value, "oi_mode").map(|mode| mode == "safe" || mode == "full"),
            estop_latched: json_bool_value(value, "estop_latched"),
            safety_tripped: json_bool_value(value, "safety_tripped").or(sensor_safety_tripped),
            safety_latch_kind: json_str_value(value, "safety_latch_kind")
                .and_then(SafetyLatchKind::from_str),
            safety_hazard_generation: json_u32_value(value, "safety_hazard_generation"),
            careful_mode_active: json_bool_value(value, "careful_mode_active"),
            careful_mode_remaining_ms: json_u32_value(value, "careful_mode_remaining_ms"),
            audio_silent: json_bool_value(value, "audio_silent")
                .or_else(|| audio.and_then(|audio| json_bool_value(audio, "silent"))),
            audio_last_requested_cue: audio
                .and_then(|audio| json_str_value(audio, "last_requested_cue"))
                .map(ToOwned::to_owned),
            audio_last_played_cue: audio
                .and_then(|audio| json_str_value(audio, "last_played_cue"))
                .map(ToOwned::to_owned),
            audio_last_playback_timestamp_ms: audio
                .and_then(|audio| json_u32_value(audio, "last_playback_timestamp_ms")),
            audio_suppressed_by_silent_count: audio
                .and_then(|audio| json_u32_value(audio, "suppressed_by_silent_count")),
            audio_dropped_or_replaced_count: audio
                .and_then(|audio| json_u32_value(audio, "dropped_or_replaced_count")),
            active_motion: json_str_value(value, "current_command")
                .map(|command| command == "drive"),
            event_next_seq: json_u32_value(value, "event_next_seq"),
            body_packet_count: packet_count,
            body_packet_timestamp_ms: sensors
                .and_then(|sensors| json_u32_value(sensors, "last_complete_packet_timestamp_ms")),
            body_packet_age_ms: packet_age_ms,
            body_packet_complete: Some(packet_count.unwrap_or(0) > 0),
            infrared_character: sensors
                .and_then(|sensors| json_u32_value(sensors, "ir_byte"))
                .and_then(|value| value.try_into().ok()),
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
    pub charging_sources: Option<u8>,
    pub charging_indicator: Option<bool>,
    pub low: Option<bool>,
}

impl BatterySummary {
    pub fn home_base(&self) -> bool {
        self.charging_sources
            .is_some_and(|sources| sources & 0b10 != 0)
    }

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
            charging_sources: number_for(raw, "charging_sources").map(|value| value as u8),
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
            charging_sources: json_u32_value(sensors, "charging_sources").map(|value| value as u8),
            charging_indicator: json_tri_state_value(sensors, "charging_indicator"),
            low: percent.map(|value| value <= 20),
        }
    }
}

#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct OdometrySummary {
    pub reset_count: Option<u32>,
    pub distance_mm: Option<i32>,
    pub x_mm: Option<i32>,
    pub y_mm: Option<i32>,
    pub heading_mrad: Option<i32>,
}

impl OdometrySummary {
    fn from_raw(raw: &str) -> Self {
        Self {
            reset_count: number_for(raw, "odometry_resets"),
            distance_mm: signed_number_for(raw, "odometry_distance_mm"),
            x_mm: signed_number_for(raw, "odometry_x_mm"),
            y_mm: signed_number_for(raw, "odometry_y_mm"),
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
            x_mm: json_i32_value(odometry, "x_mm"),
            y_mm: json_i32_value(odometry, "y_mm"),
            heading_mrad: json_i32_value(odometry, "heading_mrad"),
        }
    }
}

#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ImuSummary {
    pub present: Option<String>,
    pub health: Option<String>,
    pub sample_count: Option<u32>,
    pub sample_age_ms: Option<u32>,
    /// Exact sample timestamp in the brainstem monotonic clock.
    pub sample_timestamp_ms: Option<u32>,
    pub poll_period_ms: Option<u32>,
    pub yaw_mrad: Option<i32>,
    pub pitch_mrad: Option<i32>,
    pub roll_mrad: Option<i32>,
    pub yaw_rate_mrad_s: Option<i32>,
    pub angular_velocity_mrad_s: Axis3Summary,
    pub linear_acceleration_mm_s2: Axis3Summary,
    pub accel_magnitude_mm_s2: Option<u32>,
    pub tilt_magnitude_mrad: Option<u32>,
    pub roughness_mm_s2: Option<u32>,
    pub impact_score_mm_s2: Option<u32>,
    pub motion_consistency: Option<String>,
    pub calibration: Option<String>,
    pub orientation_confidence_permille: Option<u16>,
    pub gyro_bias_calibrated: Option<bool>,
    pub mounting_calibrated: Option<bool>,
    pub orientation_source: Option<String>,
}

impl ImuSummary {
    fn from_raw(raw: &str) -> Self {
        Self {
            present: value_for(raw, "imu_present").map(ToOwned::to_owned),
            health: value_for(raw, "imu_health").map(ToOwned::to_owned),
            sample_count: number_for(raw, "imu_samples")
                .or_else(|| number_for(raw, "imu_sample_count")),
            sample_age_ms: number_for(raw, "imu_age_ms"),
            sample_timestamp_ms: number_for(raw, "imu_sample_ms"),
            poll_period_ms: number_for(raw, "imu_poll_ms"),
            yaw_mrad: signed_number_for(raw, "imu_yaw_mrad"),
            pitch_mrad: signed_number_for(raw, "imu_pitch_mrad"),
            roll_mrad: signed_number_for(raw, "imu_roll_mrad"),
            yaw_rate_mrad_s: signed_number_for(raw, "imu_yaw_rate_mrad_s"),
            angular_velocity_mrad_s: Axis3Summary {
                x: signed_number_for(raw, "imu_gyro_x_mrad_s"),
                y: signed_number_for(raw, "imu_gyro_y_mrad_s"),
                z: signed_number_for(raw, "imu_gyro_z_mrad_s"),
            },
            linear_acceleration_mm_s2: Axis3Summary {
                x: signed_number_for(raw, "imu_accel_x_mm_s2"),
                y: signed_number_for(raw, "imu_accel_y_mm_s2"),
                z: signed_number_for(raw, "imu_accel_z_mm_s2"),
            },
            accel_magnitude_mm_s2: number_for(raw, "imu_accel_mag_mm_s2"),
            tilt_magnitude_mrad: number_for(raw, "imu_tilt_mrad"),
            roughness_mm_s2: number_for(raw, "imu_roughness_mm_s2"),
            impact_score_mm_s2: number_for(raw, "imu_impact_mm_s2"),
            motion_consistency: value_for(raw, "imu_motion_consistency").map(ToOwned::to_owned),
            calibration: value_for(raw, "imu_calibration").map(ToOwned::to_owned),
            orientation_confidence_permille: number_for(raw, "imu_orientation_confidence")
                .and_then(|value| value.try_into().ok()),
            gyro_bias_calibrated: bool_for(raw, "imu_gyro_bias_calibrated"),
            mounting_calibrated: bool_for(raw, "imu_mounting_calibrated"),
            orientation_source: value_for(raw, "imu_orientation_source").map(ToOwned::to_owned),
        }
    }

    fn from_json(imu: Option<&serde_json::Value>) -> Self {
        let Some(imu) = imu else {
            return Self::default();
        };
        Self {
            present: json_str_value(imu, "present").map(ToOwned::to_owned),
            health: json_str_value(imu, "health").map(ToOwned::to_owned),
            sample_count: json_u32_value(imu, "sample_count"),
            sample_age_ms: json_u32_value(imu, "sample_age_ms"),
            sample_timestamp_ms: json_u32_value(imu, "last_sample_timestamp_ms"),
            poll_period_ms: json_u32_value(imu, "poll_period_ms"),
            yaw_mrad: json_i32_value(imu, "yaw_mrad"),
            pitch_mrad: json_i32_value(imu, "pitch_mrad"),
            roll_mrad: json_i32_value(imu, "roll_mrad"),
            yaw_rate_mrad_s: json_i32_value(imu, "yaw_rate_mrad_s"),
            angular_velocity_mrad_s: Axis3Summary::from_json(imu.get("angular_velocity_mrad_s")),
            linear_acceleration_mm_s2: Axis3Summary::from_json(
                imu.get("linear_acceleration_mm_s2"),
            ),
            accel_magnitude_mm_s2: json_u32_value(imu, "accel_magnitude_mm_s2"),
            tilt_magnitude_mrad: json_u32_value(imu, "tilt_magnitude_mrad"),
            roughness_mm_s2: json_u32_value(imu, "roughness_mm_s2"),
            impact_score_mm_s2: json_u32_value(imu, "impact_score_mm_s2"),
            motion_consistency: json_str_value(imu, "motion_consistency").map(ToOwned::to_owned),
            calibration: json_str_value(imu, "calibration").map(ToOwned::to_owned),
            orientation_confidence_permille: json_u32_value(imu, "orientation_confidence_permille")
                .and_then(|value| value.try_into().ok()),
            gyro_bias_calibrated: json_bool_value(imu, "gyro_bias_calibrated"),
            mounting_calibrated: json_bool_value(imu, "mounting_calibrated"),
            orientation_source: json_str_value(imu, "orientation_source").map(ToOwned::to_owned),
        }
    }
}

fn inferred_imu_safety_latch(imu: &ImuSummary) -> Option<SafetyLatchKind> {
    if imu.tilt_magnitude_mrad.is_some_and(|value| value >= 650) {
        Some(SafetyLatchKind::Tilt)
    } else if imu.impact_score_mm_s2.is_some_and(|value| value >= 18_000) {
        Some(SafetyLatchKind::Impact)
    } else {
        None
    }
}

#[derive(Debug, Clone, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Axis3Summary {
    pub x: Option<i32>,
    pub y: Option<i32>,
    pub z: Option<i32>,
}

impl Axis3Summary {
    fn from_json(value: Option<&serde_json::Value>) -> Self {
        let Some(value) = value else {
            return Self::default();
        };
        Self {
            x: json_i32_value(value, "x"),
            y: json_i32_value(value, "y"),
            z: json_i32_value(value, "z"),
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
