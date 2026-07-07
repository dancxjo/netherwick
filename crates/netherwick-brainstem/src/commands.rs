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
    Stop,
    Status,
    Bootsel,
    SetMode(CreateOiMode),
    SongPlay {
        id: u8,
    },
    Dock,
    SetLights {
        pattern: LightPattern,
    },
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum LightPattern {
    Off,
    Status,
    Clean,
    Dock,
    Spot,
    Max,
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
    DriveArc {
        velocity_mm_s: i16,
        radius_mm: i16,
        duration_ms: Option<u32>,
    },
    PulseBrc,
    StartOi,
    SetMode(CreateOiMode),
    Drive {
        left_mm_s: i16,
        right_mm_s: i16,
        duration_ms: u32,
    },
    StopDrive,
    SongPlay {
        id: u8,
    },
    Dock,
    SetLights {
        pattern: LightPattern,
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
