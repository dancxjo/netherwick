use heapless::{Deque, Vec};

use crate::commands::CreateOiMode;
use crate::events::{BrainstemError, BrainstemEvent};
use crate::hardware::{BrainstemHardware, SerialRead};
use crate::status;

const OI_START: u8 = 128;
const OI_SAFE: u8 = 131;
const OI_FULL: u8 = 132;
const OI_SENSORS: u8 = 142;
const OI_DRIVE: u8 = 137;

pub struct CreateUart;

impl CreateUart {
    pub const fn new() -> Self {
        Self
    }

    pub fn poll<H, const N: usize>(
        &mut self,
        hardware: &mut H,
        events: &mut Deque<BrainstemEvent, N>,
    ) where
        H: BrainstemHardware,
    {
        match hardware.read_byte() {
            SerialRead::Byte(byte) => {
                let mut bytes = Vec::new();
                let _ = bytes.push(byte);
                status::mark_uart_rx_ok(hardware.now_us() / 1_000);
                let event = BrainstemEvent::CreatePacketReceived {
                    packet_id: 0,
                    bytes,
                };
                status::signal_event(&event);
                let _ = events.push_back(event);
            }
            SerialRead::WouldBlock => {}
            SerialRead::Error(error) => {
                status::mark_uart_rx_error_detail(error);
                let event = BrainstemEvent::Error(BrainstemError::UartFraming);
                status::signal_event(&event);
                let _ = events.push_back(event);
            }
        }
    }

    pub fn request_sensor_packet<H>(
        &mut self,
        hardware: &mut H,
        packet_id: u8,
    ) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        self.send_bytes(hardware, &[OI_SENSORS, packet_id])
    }

    pub fn start_oi<H, const N: usize>(
        &mut self,
        hardware: &mut H,
        events: &mut Deque<BrainstemEvent, N>,
    ) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        let event = BrainstemEvent::CreateOiStartRequested;
        status::signal_event(&event);
        let _ = events.push_back(event);
        self.send_byte(hardware, OI_START)?;
        Ok(())
    }

    pub fn set_mode<H, const N: usize>(
        &mut self,
        hardware: &mut H,
        events: &mut Deque<BrainstemEvent, N>,
        mode: CreateOiMode,
    ) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        let event = BrainstemEvent::CreateOiModeRequested(mode);
        status::signal_event(&event);
        let _ = events.push_back(event);
        match mode {
            CreateOiMode::Passive => {}
            CreateOiMode::Safe => self.send_byte(hardware, OI_SAFE)?,
            CreateOiMode::Full => self.send_byte(hardware, OI_FULL)?,
        }
        Ok(())
    }

    pub fn drive_direct_start<H, const N: usize>(
        &mut self,
        hardware: &mut H,
        events: &mut Deque<BrainstemEvent, N>,
        left_mm_s: i16,
        right_mm_s: i16,
        duration_ms: u32,
    ) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        let event = BrainstemEvent::DriveRequested {
            left_mm_s,
            right_mm_s,
            duration_ms,
        };
        status::signal_event(&event);
        let _ = events.push_back(event);

        let velocity = ((left_mm_s as i32 + right_mm_s as i32) / 2) as i16;
        let radius = differential_radius_mm(left_mm_s, right_mm_s);
        self.drive(hardware, velocity, radius)
    }

    pub fn stop<H, const N: usize>(
        &mut self,
        hardware: &mut H,
        events: &mut Deque<BrainstemEvent, N>,
    ) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        self.drive(hardware, 0, 0)?;
        let event = BrainstemEvent::DriveStopped;
        status::signal_event(&event);
        let _ = events.push_back(event);
        Ok(())
    }

    fn drive<H>(
        &mut self,
        hardware: &mut H,
        velocity_mm_s: i16,
        radius_mm: i16,
    ) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        let velocity = velocity_mm_s.to_be_bytes();
        let radius = radius_mm.to_be_bytes();
        self.send_bytes(
            hardware,
            &[OI_DRIVE, velocity[0], velocity[1], radius[0], radius[1]],
        )
    }

    fn send_byte<H>(&mut self, hardware: &mut H, byte: u8) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        hardware
            .write_byte(byte)
            .map_err(|_| BrainstemError::UartFraming)?;
        hardware
            .flush_uart()
            .map_err(|_| BrainstemError::UartFraming)
    }

    fn send_bytes<H>(&mut self, hardware: &mut H, bytes: &[u8]) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        for byte in bytes {
            hardware
                .write_byte(*byte)
                .map_err(|_| BrainstemError::UartFraming)?;
        }
        hardware
            .flush_uart()
            .map_err(|_| BrainstemError::UartFraming)
    }
}

fn differential_radius_mm(left_mm_s: i16, right_mm_s: i16) -> i16 {
    match (left_mm_s, right_mm_s) {
        (left, right) if left == right => 0x8000u16 as i16,
        (left, right) if left == -right && right > 0 => 1,
        (left, right) if left == -right && right < 0 => -1,
        (left, right) if left > right => -1,
        _ => 1,
    }
}
