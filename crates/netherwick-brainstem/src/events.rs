use heapless::Vec;

use crate::commands::CreateOiMode;

#[derive(Clone, Eq, PartialEq)]
pub enum BrainstemEvent {
    Boot,
    TickMs(u32),

    CreatePowerOnRequested,
    CreatePowerOffRequested,
    CreatePowerToggled,

    CreateBrcPulseRequested,
    CreateBrcPulsed,

    CreateOiStartRequested,
    CreateOiModeRequested(CreateOiMode),

    CreatePacketReceived {
        packet_id: u8,
        bytes: Vec<u8, 32>,
    },

    DriveRequested {
        left_mm_s: i16,
        right_mm_s: i16,
        duration_ms: u32,
    },

    DriveStopped,

    Error(BrainstemError),
}

#[derive(Clone, Copy, Eq, PartialEq)]
#[allow(dead_code)]
pub enum BrainstemError {
    CreateNoResponse,
    UartFraming,
    Timeout,
    InvalidPacket,
}
