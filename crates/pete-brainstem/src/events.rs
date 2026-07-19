use heapless::Vec;

use crate::commands::CreateOiMode;

#[derive(Clone, Copy, Eq, PartialEq, Default)]
pub struct CreateSensorPacket {
    pub flags: CreateSensorFlags,
    pub ir_byte: u8,
    pub buttons: u8,
    pub distance_mm: i16,
    pub angle_mrad: i16,
    pub charging_state: u8,
    pub charging_sources: u8,
    pub oi_mode: u8,
    pub voltage_mv: u16,
    pub current_ma: i16,
    pub temperature_c: i8,
    pub charge_mah: u16,
    pub capacity_mah: u16,
    pub cliff_left_signal: u16,
    pub cliff_front_left_signal: u16,
    pub cliff_front_right_signal: u16,
    pub cliff_right_signal: u16,
}

#[derive(Clone, Copy, Eq, PartialEq, Default)]
pub struct CreateSensorFlags {
    pub bump_left: bool,
    pub bump_right: bool,
    pub wheel_drop: bool,
    pub wall: bool,
    pub cliff_left: bool,
    pub cliff_front_left: bool,
    pub cliff_front_right: bool,
    pub cliff_right: bool,
    pub virtual_wall: bool,
    pub overcurrent: bool,
}

#[derive(Clone, Eq, PartialEq)]
pub enum BrainstemEvent {
    Boot,
    TickMs(u32),

    CreatePowerOnRequested,
    CreatePowerOffRequested,
    CreatePowerToggled,

    CreateOiStartRequested,
    CreateOiModeRequested(CreateOiMode),

    CreatePacketReceived {
        packet_id: u8,
        bytes: Vec<u8, 32>,
    },
    CreateSensorPacketDecoded {
        packet_id: u8,
        sensors: CreateSensorPacket,
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
