use heapless::{Deque, Vec};

use crate::commands::CreateOiMode;
use crate::events::{BrainstemError, BrainstemEvent};
use crate::hardware::{BrainstemHardware, SerialRead, UartReadError};
use crate::status;

const OI_START: u8 = 128;
const OI_SAFE: u8 = 131;
const OI_FULL: u8 = 132;
const OI_SENSORS: u8 = 142;
const OI_DRIVE: u8 = 137;
const OI_DRIVE_DIRECT: u8 = 145;
const UART_DRAIN_LIMIT: usize = 128;

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
        let mut bytes = Vec::new();
        let mut drained = 0;

        loop {
            if drained >= UART_DRAIN_LIMIT {
                push_packet(events, &mut bytes);
                break;
            }

            match hardware.read_byte() {
                SerialRead::Byte(byte) => {
                    drained += 1;
                    status::mark_uart_rx_ok(hardware.now_us() / 1_000);
                    if bytes.push(byte).is_err() {
                        push_packet(events, &mut bytes);
                        let _ = bytes.push(byte);
                    }
                }
                SerialRead::WouldBlock => {
                    push_packet(events, &mut bytes);
                    break;
                }
                SerialRead::Error(UartReadError::Overrun) => {
                    status::mark_uart_rx_error_detail(UartReadError::Overrun);
                    push_packet(events, &mut bytes);
                    break;
                }
                SerialRead::Error(error) => {
                    status::mark_uart_rx_error_detail(error);
                    push_packet(events, &mut bytes);
                    break;
                }
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

    pub fn flush_rx<H>(&mut self, hardware: &mut H)
    where
        H: BrainstemHardware,
    {
        hardware.drain_uart_rx();
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
        let event = BrainstemEvent::DriveRequested {
            left_mm_s,
            right_mm_s,
            duration_ms,
        };
        status::signal_event(&event);
        let _ = events.push_back(event);

        let right = right_mm_s.to_be_bytes();
        let left = left_mm_s.to_be_bytes();
        self.send_bytes(
            hardware,
            &[OI_DRIVE_DIRECT, right[0], right[1], left[0], left[1]],
        )
    }

    pub fn drive_arc<H, const N: usize>(
        &mut self,
        hardware: &mut H,
        events: &mut Deque<BrainstemEvent, N>,
        velocity_mm_s: i16,
        radius_mm: i16,
    ) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        let event = BrainstemEvent::DriveRequested {
            left_mm_s: velocity_mm_s,
            right_mm_s: radius_mm,
            duration_ms: 0,
        };
        status::signal_event(&event);
        let _ = events.push_back(event);

        self.drive(hardware, velocity_mm_s, radius_mm)
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

fn push_packet<const N: usize>(events: &mut Deque<BrainstemEvent, N>, bytes: &mut Vec<u8, 32>) {
    if bytes.is_empty() {
        return;
    }

    let event = BrainstemEvent::CreatePacketReceived {
        packet_id: 0,
        bytes: core::mem::take(bytes),
    };
    status::mark_uart_packet(event_len(&event));
    status::signal_event(&event);
    let _ = events.push_back(event);
}

fn event_len(event: &BrainstemEvent) -> usize {
    match event {
        BrainstemEvent::CreatePacketReceived { bytes, .. } => bytes.len(),
        _ => 0,
    }
}
