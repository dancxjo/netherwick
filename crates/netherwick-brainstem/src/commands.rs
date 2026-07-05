#[derive(Clone, Copy, Eq, PartialEq)]
pub enum CreateOiMode {
    #[allow(dead_code)]
    Passive,
    Safe,
    #[allow(dead_code)]
    Full,
}

pub type BodyMode = CreateOiMode;

#[derive(Clone, Copy, Eq, PartialEq)]
#[allow(dead_code)]
pub enum BrainstemCommand {
    SetMode(BodyMode),
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
    Ping,

    // Legacy/demo aliases kept local to the firmware command script.
    PulseBrc,
    StartOi,
    SetOiMode(CreateOiMode),
    Drive {
        left_mm_s: i16,
        right_mm_s: i16,
        duration_ms: u32,
    },
    StopDrive,
}

pub const DEMO_SCRIPT: &[BrainstemCommand] = &[
    BrainstemCommand::WakeCreate,
    BrainstemCommand::PulseBrc,
    BrainstemCommand::StartOi,
    BrainstemCommand::SetMode(BodyMode::Safe),
    BrainstemCommand::DriveDirect {
        left_mm_s: 100,
        right_mm_s: 100,
        duration_ms: Some(500),
    },
    BrainstemCommand::DriveDirect {
        left_mm_s: -80,
        right_mm_s: 80,
        duration_ms: Some(400),
    },
    BrainstemCommand::DriveDirect {
        left_mm_s: 80,
        right_mm_s: -80,
        duration_ms: Some(400),
    },
    BrainstemCommand::Stop,
];
