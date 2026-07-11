pub const MAX_SONG_TONES: usize = 16;

#[derive(Clone, Copy, Eq, PartialEq)]
#[allow(dead_code)]
pub enum CreateOiMode {
    Passive,
    Safe,
    Full,
}

#[derive(Clone, Copy, Eq, PartialEq)]
#[allow(dead_code)]
pub enum BrainstemCommand {
    Ping,
    Arm,
    Disarm,
    EStop,
    ClearEStop,
    CmdVel {
        linear_mm_s: i16,
        angular_mrad_s: i16,
        ttl_ms: u32,
        seq: u32,
    },
    DriveDirect {
        left_mm_s: i16,
        right_mm_s: i16,
        ttl_ms: u32,
        seq: u32,
    },
    DriveArc {
        velocity_mm_s: i16,
        radius_mm: i16,
        ttl_ms: u32,
        seq: u32,
    },
    FaceBearing {
        bearing_mrad: i16,
        max_angular_mrad_s: i16,
        tolerance_mrad: i16,
        ttl_ms: u32,
        seq: u32,
    },
    TrackBearing {
        bearing_mrad: i16,
        range_mm: u16,
        max_linear_mm_s: i16,
        max_angular_mrad_s: i16,
        stop_range_mm: u16,
        ttl_ms: u32,
        seq: u32,
    },
    TurnBy {
        angle_mrad: i16,
        angular_mrad_s: i16,
        timeout_ms: u32,
        seq: u32,
    },
    DriveFor {
        distance_mm: i16,
        velocity_mm_s: i16,
        timeout_ms: u32,
        seq: u32,
    },
    BumpEscape {
        direction: EscapeDirection,
        backoff_mm_s: i16,
        turn_angular_mrad_s: i16,
        seq: u32,
    },
    HoldHeading {
        heading_error_mrad: i16,
        velocity_mm_s: i16,
        max_angular_mrad_s: i16,
        ttl_ms: u32,
        seq: u32,
    },
    TurnToHeading {
        heading_error_mrad: i16,
        angular_mrad_s: i16,
        tolerance_mrad: i16,
        timeout_ms: u32,
        seq: u32,
    },
    ArcFor {
        velocity_mm_s: i16,
        radius_mm: i16,
        duration_ms: u32,
        seq: u32,
    },
    CreepUntil {
        velocity_mm_s: i16,
        angular_mrad_s: i16,
        timeout_ms: u32,
        seq: u32,
    },
    ScanArc {
        angle_mrad: i16,
        angular_mrad_s: i16,
        timeout_ms: u32,
        seq: u32,
    },
    DockAlign {
        bearing_mrad: i16,
        range_mm: u16,
        max_linear_mm_s: i16,
        max_angular_mrad_s: i16,
        stop_range_mm: u16,
        ttl_ms: u32,
        seq: u32,
    },
    WallFollow {
        distance_error_mm: i16,
        velocity_mm_s: i16,
        max_angular_mrad_s: i16,
        ttl_ms: u32,
        seq: u32,
    },
    WiggleAlign {
        amplitude_mrad: i16,
        angular_mrad_s: i16,
        cycles: u8,
        seq: u32,
    },
    Unstick {
        direction: EscapeDirection,
        backoff_mm_s: i16,
        turn_angular_mrad_s: i16,
        seq: u32,
    },
    CliffGuard {
        clear: bool,
        seq: u32,
    },
    HeartbeatStop {
        timeout_ms: u32,
        seq: u32,
    },
    RequestSensors {
        packet_id: u8,
        seq: u32,
    },
    StreamSensors {
        enabled: bool,
        packet_id: u8,
        period_ms: u32,
        seq: u32,
    },
    SetSafetyPolicy {
        policy: SafetyPolicy,
        seq: u32,
    },
    ClearMotionQueue {
        seq: u32,
    },
    DefineChirp {
        kind: FeedbackKind,
        tones: [SongTone; MAX_SONG_TONES],
        tone_count: u8,
        seq: u32,
    },
    PlayFeedback {
        kind: FeedbackKind,
        seq: u32,
    },
    PowerState {
        request: PowerStateRequest,
        seq: u32,
    },
    CalibrateTurn {
        angular_mrad_s: i16,
        duration_ms: u32,
        seq: u32,
    },
    ResetOdometry {
        seq: u32,
    },
    ZeroImuOrientation {
        seq: u32,
    },
    ClearImuOrientation {
        seq: u32,
    },
    RestartMpu,
    RestartCreate,
    GetCapabilities,
    GetEvents {
        since_seq: u32,
    },
    Stop,
    Status,
    Bootsel,
    SetMode(CreateOiMode),
    SongDefine {
        id: u8,
        tones: [SongTone; MAX_SONG_TONES],
        tone_count: u8,
        seq: u32,
    },
    SongPlay {
        id: u8,
    },
    Dock,
    SetLights {
        led_bits: u8,
        color: u8,
        intensity: u8,
    },
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum EscapeDirection {
    Left,
    Right,
    Either,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum SafetyAction {
    None,
    Stop,
    Backoff,
    BumpEscape,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct SafetyPolicy {
    pub bump: SafetyAction,
    pub cliff: SafetyAction,
    pub wheel_drop_latch: bool,
}

impl Default for SafetyPolicy {
    fn default() -> Self {
        Self {
            bump: SafetyAction::Stop,
            cliff: SafetyAction::Stop,
            wheel_drop_latch: true,
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum FeedbackKind {
    Ok,
    Error,
    Armed,
    LostTarget,
    DockSeen,
    Danger,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum PowerStateRequest {
    Wake,
    Sleep,
    PulseBrc,
    StartOi,
}

#[derive(Clone, Copy, Eq, PartialEq, Default)]
pub struct SongTone {
    pub note: u8,
    pub duration_64ths: u8,
}

#[derive(Clone, Copy, Eq, PartialEq)]
#[allow(dead_code)]
pub(crate) enum RuntimeCommand {
    WakeCreate,
    SleepCreate,
    Stop,
    EStop,
    ClearEStop,
    DriveDirect {
        left_mm_s: i16,
        right_mm_s: i16,
        duration_ms: Option<u32>,
    },
    CmdVel {
        linear_mm_s: i16,
        angular_mrad_s: i16,
        duration_ms: Option<u32>,
    },
    FaceBearing {
        bearing_mrad: i16,
        max_angular_mrad_s: i16,
        tolerance_mrad: i16,
        duration_ms: u32,
    },
    TrackBearing {
        bearing_mrad: i16,
        range_mm: u16,
        max_linear_mm_s: i16,
        max_angular_mrad_s: i16,
        stop_range_mm: u16,
        duration_ms: u32,
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
        duration_ms: u32,
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
        duration_ms: u32,
    },
    WallFollow {
        distance_error_mm: i16,
        velocity_mm_s: i16,
        max_angular_mrad_s: i16,
        duration_ms: u32,
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
    DriveArc {
        velocity_mm_s: i16,
        radius_mm: i16,
        duration_ms: Option<u32>,
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
        kind: FeedbackKind,
        tones: [SongTone; MAX_SONG_TONES],
        tone_count: u8,
    },
    PlayFeedback {
        kind: FeedbackKind,
    },
    CalibrateTurn {
        angular_mrad_s: i16,
        duration_ms: u32,
    },
    ResetOdometry,
    ZeroImuOrientation,
    ClearImuOrientation,
    RestartMpu,
    PulseBrc,
    StartOi,
    SetMode(CreateOiMode),
    Drive {
        left_mm_s: i16,
        right_mm_s: i16,
        duration_ms: u32,
    },
    StopDrive,
    SongDefine {
        id: u8,
        tones: [SongTone; MAX_SONG_TONES],
        tone_count: u8,
    },
    SongPlay {
        id: u8,
    },
    Dock,
    SetLights {
        led_bits: u8,
        color: u8,
        intensity: u8,
    },
}

pub(crate) const ARM_SCRIPT: &[RuntimeCommand] = &[
    RuntimeCommand::WakeCreate,
    RuntimeCommand::PulseBrc,
    RuntimeCommand::StartOi,
    RuntimeCommand::SetMode(CreateOiMode::Safe),
];

pub(crate) const DISARM_SCRIPT: &[RuntimeCommand] =
    &[RuntimeCommand::Stop, RuntimeCommand::SleepCreate];

pub(crate) const RESTART_CREATE_SCRIPT: &[RuntimeCommand] = &[
    RuntimeCommand::Stop,
    RuntimeCommand::SleepCreate,
    RuntimeCommand::WakeCreate,
    RuntimeCommand::PulseBrc,
    RuntimeCommand::StartOi,
    RuntimeCommand::SetMode(CreateOiMode::Safe),
];

pub(crate) const DEMO_SCRIPT: &[RuntimeCommand] = &[
    RuntimeCommand::WakeCreate,
    RuntimeCommand::PulseBrc,
    RuntimeCommand::StartOi,
    RuntimeCommand::SetMode(CreateOiMode::Safe),
    RuntimeCommand::DriveDirect {
        left_mm_s: 100,
        right_mm_s: 100,
        duration_ms: Some(500),
    },
    RuntimeCommand::DriveDirect {
        left_mm_s: -80,
        right_mm_s: 80,
        duration_ms: Some(400),
    },
    RuntimeCommand::DriveDirect {
        left_mm_s: 80,
        right_mm_s: -80,
        duration_ms: Some(400),
    },
    RuntimeCommand::Stop,
];
