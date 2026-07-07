use heapless::{Deque, Vec};

use crate::commands::{CreateOiMode, LightPattern, SongTone, MAX_SONG_TONES};
use crate::events::{BrainstemError, BrainstemEvent, CreateSensorFlags, CreateSensorPacket};
use crate::hardware::{BrainstemHardware, SerialRead, UartReadError};
use crate::status;

const OI_START: u8 = 128;
const OI_SAFE: u8 = 131;
const OI_FULL: u8 = 132;
const OI_SENSORS: u8 = 142;
const OI_DRIVE: u8 = 137;
const OI_LEDS: u8 = 139;
const OI_DEFINE_SONG: u8 = 140;
const OI_PLAY_SONG: u8 = 141;
const OI_SEEK_DOCK: u8 = 143;
const OI_DRIVE_DIRECT: u8 = 145;
const UART_DRAIN_LIMIT: usize = 128;

pub struct CreateUart {
    pending_sensor_packet: Option<u8>,
}

impl CreateUart {
    pub const fn new() -> Self {
        Self {
            pending_sensor_packet: None,
        }
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
                self.push_packet(events, &mut bytes);
                break;
            }

            match hardware.read_byte() {
                SerialRead::Byte(byte) => {
                    drained += 1;
                    status::mark_uart_rx_ok(hardware.now_us() / 1_000);
                    if bytes.push(byte).is_err() {
                        self.push_packet(events, &mut bytes);
                        let _ = bytes.push(byte);
                    }
                }
                SerialRead::WouldBlock => {
                    self.push_packet(events, &mut bytes);
                    break;
                }
                SerialRead::Error(UartReadError::Overrun) => {
                    status::mark_uart_rx_error_detail(UartReadError::Overrun);
                    self.push_packet(events, &mut bytes);
                    drained += 1;
                }
                SerialRead::Error(error) => {
                    status::mark_uart_rx_error_detail(error);
                    self.push_packet(events, &mut bytes);
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
        self.pending_sensor_packet = Some(packet_id);
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

    pub fn play_song<H, const N: usize>(
        &mut self,
        hardware: &mut H,
        _events: &mut Deque<BrainstemEvent, N>,
        id: u8,
    ) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        let id = id.min(15);
        self.send_bytes(hardware, &[OI_PLAY_SONG, id])?;
        status::mark_song_played(id);
        Ok(())
    }

    pub fn define_song<H, const N: usize>(
        &mut self,
        hardware: &mut H,
        _events: &mut Deque<BrainstemEvent, N>,
        id: u8,
        tones: &[SongTone; MAX_SONG_TONES],
        tone_count: u8,
    ) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        let id = id.min(15);
        let tone_count = tone_count.min(MAX_SONG_TONES as u8);
        if tone_count == 0 {
            return Ok(());
        }

        let mut bytes = [0u8; 3 + MAX_SONG_TONES * 2];
        bytes[0] = OI_DEFINE_SONG;
        bytes[1] = id;
        bytes[2] = tone_count;
        for i in 0..tone_count as usize {
            let tone = tones[i];
            let offset = 3 + i * 2;
            bytes[offset] = tone.note.clamp(31, 127);
            bytes[offset + 1] = tone.duration_64ths.max(1);
        }
        self.send_bytes(hardware, &bytes[..3 + tone_count as usize * 2])?;
        status::mark_song_defined(id, tone_count);
        Ok(())
    }

    pub fn seek_dock<H, const N: usize>(
        &mut self,
        hardware: &mut H,
        _events: &mut Deque<BrainstemEvent, N>,
    ) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        self.send_byte(hardware, OI_SEEK_DOCK)
    }

    pub fn set_lights<H, const N: usize>(
        &mut self,
        hardware: &mut H,
        _events: &mut Deque<BrainstemEvent, N>,
        pattern: LightPattern,
    ) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        let (bits, color, intensity) = match pattern {
            LightPattern::Off => (0, 0, 0),
            LightPattern::Status => (0b0010, 128, 160),
            LightPattern::Clean => (0b0010, 0, 255),
            LightPattern::Dock => (0b0100, 255, 255),
            LightPattern::Spot => (0b1000, 255, 180),
            LightPattern::Max => (0b1111, 255, 255),
        };
        self.send_bytes(hardware, &[OI_LEDS, bits, color, intensity])
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

    fn push_packet<const N: usize>(
        &mut self,
        events: &mut Deque<BrainstemEvent, N>,
        bytes: &mut Vec<u8, 32>,
    ) {
        if bytes.is_empty() {
            return;
        }

        let packet_id = self.pending_sensor_packet.take().unwrap_or(0);
        let raw_bytes = core::mem::take(bytes);
        status::mark_uart_packet(raw_bytes.len());

        if let Some(sensors) = decode_sensor_packet(packet_id, &raw_bytes) {
            status::mark_create_sensor_packet(packet_id, sensors);
            let event = BrainstemEvent::CreateSensorPacketDecoded { packet_id, sensors };
            status::signal_event(&event);
            let _ = events.push_back(event);
        }

        let event = BrainstemEvent::CreatePacketReceived {
            packet_id,
            bytes: raw_bytes,
        };
        status::signal_event(&event);
        let _ = events.push_back(event);
    }
}

fn decode_sensor_packet(packet_id: u8, bytes: &[u8]) -> Option<CreateSensorPacket> {
    let mut sensors = CreateSensorPacket::default();
    match packet_id {
        0 if bytes.len() >= 26 => {
            apply_bumps_wheel_drops(&mut sensors.flags, bytes[0]);
            sensors.flags.wall = bytes[1] != 0;
            sensors.flags.cliff_left = bytes[2] != 0;
            sensors.flags.cliff_front_left = bytes[3] != 0;
            sensors.flags.cliff_front_right = bytes[4] != 0;
            sensors.flags.cliff_right = bytes[5] != 0;
            sensors.flags.virtual_wall = bytes[6] != 0;
            sensors.flags.overcurrent = bytes[7] != 0;
            sensors.ir_byte = bytes[10];
            sensors.buttons = bytes[11];
            sensors.distance_mm = i16::from_be_bytes([bytes[12], bytes[13]]);
            let angle_deg = i16::from_be_bytes([bytes[14], bytes[15]]);
            sensors.angle_mrad = degrees_to_mrad(angle_deg);
            sensors.charging_state = bytes[16];
            sensors.voltage_mv = u16::from_be_bytes([bytes[17], bytes[18]]);
            sensors.current_ma = i16::from_be_bytes([bytes[19], bytes[20]]);
            sensors.temperature_c = bytes[21] as i8;
            sensors.charge_mah = u16::from_be_bytes([bytes[22], bytes[23]]);
            sensors.capacity_mah = u16::from_be_bytes([bytes[24], bytes[25]]);
            Some(sensors)
        }
        7 if bytes.len() == 1 => {
            apply_bumps_wheel_drops(&mut sensors.flags, bytes[0]);
            Some(sensors)
        }
        8 if bytes.len() == 1 => {
            sensors.flags.wall = bytes[0] != 0;
            Some(sensors)
        }
        9 if bytes.len() == 1 => {
            sensors.flags.cliff_left = bytes[0] != 0;
            Some(sensors)
        }
        10 if bytes.len() == 1 => {
            sensors.flags.cliff_front_left = bytes[0] != 0;
            Some(sensors)
        }
        11 if bytes.len() == 1 => {
            sensors.flags.cliff_front_right = bytes[0] != 0;
            Some(sensors)
        }
        12 if bytes.len() == 1 => {
            sensors.flags.cliff_right = bytes[0] != 0;
            Some(sensors)
        }
        13 if bytes.len() == 1 => {
            sensors.flags.virtual_wall = bytes[0] != 0;
            Some(sensors)
        }
        14 if bytes.len() == 1 => {
            sensors.flags.overcurrent = bytes[0] != 0;
            Some(sensors)
        }
        17 if bytes.len() == 1 => {
            sensors.ir_byte = bytes[0];
            Some(sensors)
        }
        18 if bytes.len() == 1 => {
            sensors.buttons = bytes[0];
            Some(sensors)
        }
        19 if bytes.len() == 2 => {
            sensors.distance_mm = i16::from_be_bytes([bytes[0], bytes[1]]);
            Some(sensors)
        }
        20 if bytes.len() == 2 => {
            let angle_deg = i16::from_be_bytes([bytes[0], bytes[1]]);
            sensors.angle_mrad = degrees_to_mrad(angle_deg);
            Some(sensors)
        }
        21 if bytes.len() == 1 => {
            sensors.charging_state = bytes[0];
            Some(sensors)
        }
        22 if bytes.len() == 2 => {
            sensors.voltage_mv = u16::from_be_bytes([bytes[0], bytes[1]]);
            Some(sensors)
        }
        23 if bytes.len() == 2 => {
            sensors.current_ma = i16::from_be_bytes([bytes[0], bytes[1]]);
            Some(sensors)
        }
        24 if bytes.len() == 1 => {
            sensors.temperature_c = bytes[0] as i8;
            Some(sensors)
        }
        25 if bytes.len() == 2 => {
            sensors.charge_mah = u16::from_be_bytes([bytes[0], bytes[1]]);
            Some(sensors)
        }
        26 if bytes.len() == 2 => {
            sensors.capacity_mah = u16::from_be_bytes([bytes[0], bytes[1]]);
            Some(sensors)
        }
        28 if bytes.len() == 2 => {
            sensors.cliff_left_signal = u16::from_be_bytes([bytes[0], bytes[1]]);
            Some(sensors)
        }
        29 if bytes.len() == 2 => {
            sensors.cliff_front_left_signal = u16::from_be_bytes([bytes[0], bytes[1]]);
            Some(sensors)
        }
        30 if bytes.len() == 2 => {
            sensors.cliff_front_right_signal = u16::from_be_bytes([bytes[0], bytes[1]]);
            Some(sensors)
        }
        31 if bytes.len() == 2 => {
            sensors.cliff_right_signal = u16::from_be_bytes([bytes[0], bytes[1]]);
            Some(sensors)
        }
        _ => None,
    }
}

fn apply_bumps_wheel_drops(flags: &mut CreateSensorFlags, byte: u8) {
    flags.bump_right = byte & 0b0000_0001 != 0;
    flags.bump_left = byte & 0b0000_0010 != 0;
    flags.wheel_drop = byte & 0b0001_1100 != 0;
}

fn degrees_to_mrad(degrees: i16) -> i16 {
    let mrad = degrees as i32 * 17_453 / 1_000;
    mrad.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}
