use heapless::{Deque, Vec};

use crate::body;
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
                let _ = events.push_back(BrainstemEvent::CreatePacketReceived {
                    packet_id: 0,
                    bytes,
                });
            }
            SerialRead::WouldBlock => {}
            SerialRead::Error => {
                status::mark_uart_rx_error();
                let _ = events.push_back(BrainstemEvent::Error(BrainstemError::UartFraming));
            }
        }
    }

    pub fn wait_for_response<H, const N: usize>(
        &mut self,
        hardware: &mut H,
        events: &mut Deque<BrainstemEvent, N>,
    ) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        let deadline = hardware
            .now_us()
            .wrapping_add(body::CREATE_RESPONSIVE_TIMEOUT_MS * 1_000);
        loop {
            self.send_bytes(hardware, &[OI_SENSORS, body::CREATE_SENSOR_PROBE_PACKET])?;
            let mut bytes = Vec::new();
            if self.read_packet_bytes(hardware, &mut bytes, 2).is_ok() {
                status::mark_uart_rx_ok(hardware.now_us() / 1_000);
                let _ = events.push_back(BrainstemEvent::CreatePacketReceived {
                    packet_id: body::CREATE_SENSOR_PROBE_PACKET,
                    bytes,
                });
                return Ok(());
            }
            if hardware.now_us().wrapping_sub(deadline) < u32::MAX / 2 {
                status::mark_uart_rx_error();
                let _ = events.push_back(BrainstemEvent::Error(BrainstemError::CreateNoResponse));
                return Err(BrainstemError::CreateNoResponse);
            }
            hardware.delay_ms(100);
        }
    }

    pub fn start_oi<H, const N: usize>(
        &mut self,
        hardware: &mut H,
        events: &mut Deque<BrainstemEvent, N>,
    ) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        let _ = events.push_back(BrainstemEvent::CreateOiStartRequested);
        self.send_byte(hardware, OI_START)?;
        hardware.delay_ms(body::POST_MODE_SETTLE_MS);
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
        let _ = events.push_back(BrainstemEvent::CreateOiModeRequested(mode));
        match mode {
            CreateOiMode::Passive => {}
            CreateOiMode::Safe => self.send_byte(hardware, OI_SAFE)?,
            CreateOiMode::Full => self.send_byte(hardware, OI_FULL)?,
        }
        hardware.delay_ms(body::POST_MODE_SETTLE_MS);
        Ok(())
    }

    pub fn drive_direct<H, const N: usize>(
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
        let _ = events.push_back(BrainstemEvent::DriveRequested {
            left_mm_s,
            right_mm_s,
            duration_ms,
        });

        let velocity = ((left_mm_s as i32 + right_mm_s as i32) / 2) as i16;
        let radius = differential_radius_mm(left_mm_s, right_mm_s);
        self.drive(hardware, velocity, radius)?;
        hardware.delay_ms(duration_ms);
        self.stop(hardware, events)
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
        let _ = events.push_back(BrainstemEvent::DriveStopped);
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

    fn read_packet_bytes<H>(
        &mut self,
        hardware: &mut H,
        bytes: &mut Vec<u8, 32>,
        len: usize,
    ) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        for _ in 0..len {
            let start = hardware.now_us();
            loop {
                match hardware.read_byte() {
                    SerialRead::Byte(byte) => {
                        bytes
                            .push(byte)
                            .map_err(|_| BrainstemError::InvalidPacket)?;
                        break;
                    }
                    SerialRead::WouldBlock => {
                        if hardware.now_us().wrapping_sub(start)
                            >= body::UART_BYTE_TIMEOUT_MS * 1_000
                        {
                            return Err(BrainstemError::Timeout);
                        }
                    }
                    SerialRead::Error => return Err(BrainstemError::UartFraming),
                }
            }
        }
        Ok(())
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
