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
pub enum SafetyLatchKind {
    Bump,
    Cliff,
    WheelDrop,
    Tilt,
    Impact,
    Charging,
    Heartbeat,
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
    /// A recognized pre-migration wire verb. It is retained only long enough
    /// to return a typed `unsupported` result; no retired parameters enter the
    /// runtime command vocabulary.
    Unsupported {
        seq: u32,
    },
    ClearSafetyLatch {
        kind: SafetyLatchKind,
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
    OrientationProbe {
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
    RestartCreate,
    ResetMotherbrain,
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
    DebugBaud19200,
    DebugBaud57600,
    DebugBaud115200,
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
    ClearSafetyLatch {
        kind: SafetyLatchKind,
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
    OrientationProbe {
        angular_mrad_s: i16,
        duration_ms: u32,
    },
    ResetOdometry,
    ZeroImuOrientation,
    ClearImuOrientation,
    PulseBrc,
    StartOi,
    SetCreateBaud(u32),
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

pub(crate) const ACQUIRE_CREATE_SCRIPT: &[RuntimeCommand] = &[
    RuntimeCommand::WakeCreate,
    RuntimeCommand::PulseBrc,
    RuntimeCommand::StartOi,
    RuntimeCommand::SetMode(CreateOiMode::Full),
    RuntimeCommand::PlayFeedback {
        kind: FeedbackKind::Armed,
    },
];

// DISARM is retained as a legacy wire verb, but surrendering a controller must
// not surrender the brainstem's ownership of Create OI.  Stop the wheels and
// leave mode/power supervision to `maintain_full_mode`.
pub(crate) const DISARM_SCRIPT: &[RuntimeCommand] = &[RuntimeCommand::Stop];

pub(crate) const RESTART_CREATE_SCRIPT: &[RuntimeCommand] = &[
    RuntimeCommand::Stop,
    RuntimeCommand::SleepCreate,
    RuntimeCommand::WakeCreate,
    RuntimeCommand::PulseBrc,
    RuntimeCommand::StartOi,
    RuntimeCommand::SetMode(CreateOiMode::Full),
];
