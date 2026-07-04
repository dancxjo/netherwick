#[derive(Clone, Copy, Eq, PartialEq)]
pub enum CreateOiMode {
    #[allow(dead_code)]
    Passive,
    Safe,
    #[allow(dead_code)]
    Full,
}

#[derive(Clone, Copy, Eq, PartialEq)]
#[allow(dead_code)]
pub enum BrainstemCommand {
    WakeCreate,
    SleepCreate,
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
    BrainstemCommand::SetOiMode(CreateOiMode::Safe),
    BrainstemCommand::Drive {
        left_mm_s: 100,
        right_mm_s: 100,
        duration_ms: 500,
    },
    BrainstemCommand::Drive {
        left_mm_s: -80,
        right_mm_s: 80,
        duration_ms: 400,
    },
    BrainstemCommand::Drive {
        left_mm_s: 80,
        right_mm_s: -80,
        duration_ms: 400,
    },
    BrainstemCommand::StopDrive,
    BrainstemCommand::SleepCreate,
];
