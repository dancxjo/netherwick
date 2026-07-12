use heapless::{Deque, Vec};

use crate::commands::{CreateOiMode, SongTone, MAX_SONG_TONES};
use crate::events::{BrainstemError, BrainstemEvent, CreateSensorFlags, CreateSensorPacket};
use crate::hardware::{BrainstemHardware, SerialRead, UartReadError};
use crate::status;

const OI_START: u8 = 128;
const OI_SAFE: u8 = 131;
const OI_FULL: u8 = 132;
const OI_DRIVE: u8 = 137;
const OI_LEDS: u8 = 139;
const OI_DEFINE_SONG: u8 = 140;
const OI_PLAY_SONG: u8 = 141;
const OI_SEEK_DOCK: u8 = 143;
const OI_DRIVE_DIRECT: u8 = 145;
const OI_STREAM: u8 = 148;
const OI_PAUSE_RESUME_STREAM: u8 = 150;
const OI_STREAM_HEADER: u8 = 19;
const UART_DRAIN_LIMIT: usize = 128;

#[derive(Clone, Copy)]
enum StreamState {
    Header,
    Length,
    PacketId,
    Data,
    Checksum,
}

pub struct CreateUart {
    pending_sensor_packet: Option<u8>,
    sensor_bytes: Vec<u8, 32>,
    stream_state: StreamState,
    stream_sum: u8,
}

impl CreateUart {
    pub const fn new() -> Self {
        Self {
            pending_sensor_packet: None,
            sensor_bytes: Vec::new(),
            stream_state: StreamState::Header,
            stream_sum: 0,
        }
    }

    pub fn poll<H, const N: usize>(
        &mut self,
        hardware: &mut H,
        events: &mut Deque<BrainstemEvent, N>,
    ) where
        H: BrainstemHardware,
    {
        let mut drained = 0;

        loop {
            if drained >= UART_DRAIN_LIMIT {
                break;
            }

            match hardware.read_byte() {
                SerialRead::Byte(byte) => {
                    drained += 1;
                    status::mark_uart_rx_byte(byte, hardware.now_us() / 1_000);
                    if self.pending_sensor_packet.is_some() {
                        self.process_stream_byte(hardware, events, byte);
                    }
                }
                SerialRead::WouldBlock => break,
                SerialRead::Error(UartReadError::Overrun) => {
                    status::mark_uart_rx_error_detail(UartReadError::Overrun);
                    self.reset_stream_parser();
                    drained += 1;
                }
                SerialRead::Error(error) => {
                    status::mark_uart_rx_error_detail(error);
                    self.reset_stream_parser();
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
        if sensor_packet_length(packet_id).is_none() {
            return Err(BrainstemError::InvalidPacket);
        }
        self.sensor_bytes.clear();
        self.pending_sensor_packet = Some(packet_id);
        self.reset_stream_parser();
        if let Err(error) = self.send_bytes(hardware, &[OI_STREAM, 1, packet_id]) {
            self.discard_pending_response();
            return Err(error);
        }
        Ok(())
    }

    pub fn flush_rx<H>(&mut self, hardware: &mut H)
    where
        H: BrainstemHardware,
    {
        hardware.drain_uart_rx();
        self.discard_pending_response();
        self.reset_stream_parser();
    }

    pub fn start_mode_stream<H>(&mut self, hardware: &mut H) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        self.request_sensor_packet(hardware, 35)
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
        led_bits: u8,
        color: u8,
        intensity: u8,
    ) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        self.send_bytes(hardware, &[OI_LEDS, led_bits & 0x0f, color, intensity])
    }

    /// Updates the brainstem-owned status animation without filling the public
    /// event queue with heartbeat frames.
    pub fn set_supervision_lights<H>(
        &mut self,
        hardware: &mut H,
        led_bits: u8,
        color: u8,
        intensity: u8,
    ) -> Result<(), BrainstemError>
    where
        H: BrainstemHardware,
    {
        self.send_bytes(hardware, &[OI_LEDS, led_bits & 0x03, color, intensity])
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
        status::mark_uart_tx_byte(byte, hardware.now_us() / 1_000);
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
            status::mark_uart_tx_byte(*byte, hardware.now_us() / 1_000);
        }
        hardware
            .flush_uart()
            .map_err(|_| BrainstemError::UartFraming)
    }

    fn push_packet<const N: usize>(&mut self, events: &mut Deque<BrainstemEvent, N>) {
        if self.sensor_bytes.is_empty() {
            return;
        }

        let packet_id = self.pending_sensor_packet.take().unwrap_or(0);
        let raw_bytes = core::mem::take(&mut self.sensor_bytes);
        if let Some(sensors) = decode_sensor_packet(packet_id, &raw_bytes) {
            status::mark_uart_packet(raw_bytes.len());
            status::mark_create_sensor_packet(packet_id, sensors);
            let event = BrainstemEvent::CreateSensorPacketDecoded { packet_id, sensors };
            status::signal_event(&event);
            let _ = events.push_back(event);
        } else if sensor_packet_is_decoded(packet_id) {
            let event = BrainstemEvent::Error(BrainstemError::InvalidPacket);
            status::signal_event(&event);
            let _ = events.push_back(event);
            return;
        }

        let event = BrainstemEvent::CreatePacketReceived {
            packet_id,
            bytes: raw_bytes,
        };
        status::signal_event(&event);
        let _ = events.push_back(event);
    }

    fn discard_pending_response(&mut self) {
        self.pending_sensor_packet = None;
        self.sensor_bytes.clear();
    }

    fn reset_stream_parser(&mut self) {
        self.stream_state = StreamState::Header;
        self.stream_sum = 0;
        self.sensor_bytes.clear();
    }

    fn process_stream_byte<H, const N: usize>(
        &mut self,
        hardware: &mut H,
        events: &mut Deque<BrainstemEvent, N>,
        byte: u8,
    ) where
        H: BrainstemHardware,
    {
        let Some(packet_id) = self.pending_sensor_packet else {
            return;
        };
        let Some(expected_len) = sensor_packet_length(packet_id) else {
            self.discard_pending_response();
            return;
        };

        match self.stream_state {
            StreamState::Header => {
                if byte == OI_STREAM_HEADER {
                    self.stream_sum = byte;
                    self.stream_state = StreamState::Length;
                }
            }
            StreamState::Length => {
                if usize::from(byte) == expected_len + 1 {
                    self.stream_sum = self.stream_sum.wrapping_add(byte);
                    self.stream_state = StreamState::PacketId;
                } else {
                    self.restart_stream_from(byte);
                }
            }
            StreamState::PacketId => {
                if byte == packet_id {
                    self.stream_sum = self.stream_sum.wrapping_add(byte);
                    self.sensor_bytes.clear();
                    self.stream_state = StreamState::Data;
                } else {
                    self.restart_stream_from(byte);
                }
            }
            StreamState::Data => {
                self.stream_sum = self.stream_sum.wrapping_add(byte);
                if self.sensor_bytes.push(byte).is_err() {
                    self.restart_stream_from(byte);
                } else if self.sensor_bytes.len() == expected_len {
                    self.stream_state = StreamState::Checksum;
                }
            }
            StreamState::Checksum => {
                if self.stream_sum.wrapping_add(byte) == 0 {
                    let _ = self.send_bytes(hardware, &[OI_PAUSE_RESUME_STREAM, 0]);
                    self.push_packet(events);
                    self.reset_stream_parser();
                } else {
                    // Leave the stream running and scan for the next header.
                    // A corrupt or partial frame must never be promoted to a
                    // sensor sample.
                    self.restart_stream_from(byte);
                }
            }
        }
    }

    fn restart_stream_from(&mut self, byte: u8) {
        self.sensor_bytes.clear();
        if byte == OI_STREAM_HEADER {
            self.stream_sum = byte;
            self.stream_state = StreamState::Length;
        } else {
            self.stream_sum = 0;
            self.stream_state = StreamState::Header;
        }
    }
}

fn sensor_packet_length(packet_id: u8) -> Option<usize> {
    match packet_id {
        0 => Some(26),
        7..=18 | 21 | 24 | 35 => Some(1),
        19 | 20 | 22 | 23 | 25..=31 => Some(2),
        _ => None,
    }
}

fn sensor_packet_is_decoded(packet_id: u8) -> bool {
    matches!(
        packet_id,
        0 | 7..=14 | 17..=26 | 28..=31 | 35
    )
}

fn decode_sensor_packet(packet_id: u8, bytes: &[u8]) -> Option<CreateSensorPacket> {
    let mut sensors = CreateSensorPacket::default();
    match packet_id {
        0 if bytes.len() == 26 && valid_group_zero(bytes) => {
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
        7 if bytes.len() == 1 && valid_bumps_wheel_drops(bytes[0]) => {
            apply_bumps_wheel_drops(&mut sensors.flags, bytes[0]);
            Some(sensors)
        }
        8 if bytes.len() == 1 && valid_bool(bytes[0]) => {
            sensors.flags.wall = bytes[0] != 0;
            Some(sensors)
        }
        9 if bytes.len() == 1 && valid_bool(bytes[0]) => {
            sensors.flags.cliff_left = bytes[0] != 0;
            Some(sensors)
        }
        10 if bytes.len() == 1 && valid_bool(bytes[0]) => {
            sensors.flags.cliff_front_left = bytes[0] != 0;
            Some(sensors)
        }
        11 if bytes.len() == 1 && valid_bool(bytes[0]) => {
            sensors.flags.cliff_front_right = bytes[0] != 0;
            Some(sensors)
        }
        12 if bytes.len() == 1 && valid_bool(bytes[0]) => {
            sensors.flags.cliff_right = bytes[0] != 0;
            Some(sensors)
        }
        13 if bytes.len() == 1 && valid_bool(bytes[0]) => {
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
        18 if bytes.len() == 1 && valid_buttons(bytes[0]) => {
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
        21 if bytes.len() == 1 && valid_charging_state(bytes[0]) => {
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
        35 if bytes.len() == 1 && bytes[0] <= 3 => {
            sensors.oi_mode = bytes[0];
            Some(sensors)
        }
        _ => None,
    }
}

fn valid_group_zero(bytes: &[u8]) -> bool {
    valid_bumps_wheel_drops(bytes[0])
        && bytes[1..=6].iter().all(|byte| valid_bool(*byte))
        && valid_buttons(bytes[11])
        && valid_charging_state(bytes[16])
}

fn valid_bumps_wheel_drops(byte: u8) -> bool {
    byte & !0b0001_1111 == 0
}

fn valid_bool(byte: u8) -> bool {
    byte <= 1
}

fn valid_buttons(byte: u8) -> bool {
    byte & !0b0000_1111 == 0
}

fn valid_charging_state(byte: u8) -> bool {
    byte <= 5
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct TestHardware {
        rx: VecDeque<SerialRead>,
        tx: std::vec::Vec<u8>,
    }

    impl TestHardware {
        fn new() -> Self {
            Self {
                rx: VecDeque::new(),
                tx: std::vec::Vec::new(),
            }
        }
    }

    impl BrainstemHardware for TestHardware {
        fn delay_ms(&mut self, _ms: u32) {}
        fn now_us(&mut self) -> u32 {
            0
        }
        fn feed_watchdog(&mut self) {}
        fn set_power_toggle(&mut self, _high: bool) {}
        fn set_brc(&mut self, _high: bool) {}
        fn set_indicators(&mut self, _on: bool) {}
        fn set_primary_indicator(&mut self, _on: bool) {}
        fn write_byte(&mut self, byte: u8) -> Result<(), ()> {
            self.tx.push(byte);
            Ok(())
        }
        fn flush_uart(&mut self) -> Result<(), ()> {
            Ok(())
        }
        fn read_byte(&mut self) -> SerialRead {
            self.rx.pop_front().unwrap_or(SerialRead::WouldBlock)
        }
    }

    fn valid_group_zero() -> [u8; 26] {
        let mut packet = [0; 26];
        packet[17..19].copy_from_slice(&14_400u16.to_be_bytes());
        packet[19..21].copy_from_slice(&(-250i16).to_be_bytes());
        packet[22..24].copy_from_slice(&1_800u16.to_be_bytes());
        packet[24..26].copy_from_slice(&3_000u16.to_be_bytes());
        packet
    }

    fn stream_frame(packet_id: u8, payload: &[u8]) -> std::vec::Vec<u8> {
        let mut frame = std::vec![OI_STREAM_HEADER, payload.len() as u8 + 1, packet_id];
        frame.extend_from_slice(payload);
        let sum = frame
            .iter()
            .copied()
            .fold(0u8, |sum, byte| sum.wrapping_add(byte));
        frame.push(0u8.wrapping_sub(sum));
        frame
    }

    #[test]
    fn group_zero_requires_exact_length() {
        let packet = valid_group_zero();
        assert!(decode_sensor_packet(0, &packet).is_some());

        let mut oversized = [0; 27];
        oversized[..26].copy_from_slice(&packet);
        assert!(decode_sensor_packet(0, &oversized).is_none());
        assert!(decode_sensor_packet(0, &packet[..25]).is_none());
    }

    #[test]
    fn group_zero_rejects_impossible_safety_and_status_fields() {
        let mut packet = valid_group_zero();
        packet[2] = b'1';
        assert!(decode_sensor_packet(0, &packet).is_none());

        packet = valid_group_zero();
        packet[16] = 90;
        assert!(decode_sensor_packet(0, &packet).is_none());

        packet = valid_group_zero();
        packet[11] = 0b1000_0000;
        assert!(decode_sensor_packet(0, &packet).is_none());
    }

    #[test]
    fn packet_lengths_cover_the_advertised_sensor_range() {
        assert_eq!(sensor_packet_length(0), Some(26));
        for packet_id in 7..=31 {
            assert!(sensor_packet_length(packet_id).is_some());
        }
        assert_eq!(sensor_packet_length(35), Some(1));
        assert_eq!(sensor_packet_length(6), None);
        assert_eq!(sensor_packet_length(32), None);
    }

    #[test]
    fn oi_mode_packet_is_strictly_validated() {
        for mode in 0..=3 {
            assert_eq!(decode_sensor_packet(35, &[mode]).unwrap().oi_mode, mode);
        }
        assert!(decode_sensor_packet(35, &[4]).is_none());
        assert!(decode_sensor_packet(35, &[0xf8]).is_none());
    }

    #[test]
    fn mode_stream_resynchronizes_and_requires_checksum() {
        let _guard = status::status_test_guard();
        let mut uart = CreateUart::new();
        let mut hardware = TestHardware::new();
        let mut events = Deque::<BrainstemEvent, 4>::new();
        let before = status::snapshot(0).uart_rx_packets;

        assert!(uart.start_mode_stream(&mut hardware).is_ok());
        hardware.rx.extend(
            [
                0xf8, 19, 9, // noise and a frame with the wrong length
                19, 2, 35, 3, 198, // corrupt checksum
                19, 2, 35, 3, 197, // valid Full-mode frame
            ]
            .into_iter()
            .map(SerialRead::Byte),
        );
        uart.poll(&mut hardware, &mut events);

        assert_eq!(status::snapshot(0).oi_mode, 3);
        assert_eq!(status::snapshot(0).uart_rx_packets, before.wrapping_add(1));
        assert!(hardware.tx.ends_with(&[OI_PAUSE_RESUME_STREAM, 0]));
    }

    #[test]
    fn set_lights_sends_raw_mechanical_values() {
        let mut uart = CreateUart::new();
        let mut hardware = TestHardware::new();
        let mut events = Deque::<BrainstemEvent, 1>::new();

        assert!(uart
            .set_lights(&mut hardware, &mut events, 0b1_0101, 128, 64)
            .is_ok());

        assert_eq!(hardware.tx, [OI_LEDS, 0b0101, 128, 64]);
    }

    #[test]
    fn poll_accumulates_fragmented_responses_until_exact_length() {
        let mut uart = CreateUart::new();
        let mut hardware = TestHardware::new();
        let mut events = Deque::<BrainstemEvent, 4>::new();
        let packet = valid_group_zero();
        let frame = stream_frame(0, &packet);

        assert!(uart.request_sensor_packet(&mut hardware, 0).is_ok());
        hardware
            .rx
            .extend(frame[..10].iter().copied().map(SerialRead::Byte));
        uart.poll(&mut hardware, &mut events);
        assert!(events.is_empty());
        assert_eq!(uart.sensor_bytes.len(), 7);

        hardware
            .rx
            .extend(frame[10..].iter().copied().map(SerialRead::Byte));
        uart.poll(&mut hardware, &mut events);
        assert_eq!(events.len(), 2);
        assert!(uart.pending_sensor_packet.is_none());
        assert!(uart.sensor_bytes.is_empty());
        assert!(hardware.tx.starts_with(&[OI_STREAM, 1, 0]));
        assert!(hardware.tx.ends_with(&[OI_PAUSE_RESUME_STREAM, 0]));
    }

    #[test]
    fn poll_resynchronizes_after_framing_error() {
        let mut uart = CreateUart::new();
        let mut hardware = TestHardware::new();
        let mut events = Deque::<BrainstemEvent, 4>::new();
        let packet = valid_group_zero();

        assert!(uart.request_sensor_packet(&mut hardware, 0).is_ok());
        hardware.rx.extend([
            SerialRead::Byte(OI_STREAM_HEADER),
            SerialRead::Byte(27),
            SerialRead::Byte(0),
            SerialRead::Byte(0),
            SerialRead::Error(UartReadError::Framing),
        ]);
        uart.poll(&mut hardware, &mut events);

        assert!(events.is_empty());
        assert_eq!(uart.pending_sensor_packet, Some(0));
        assert!(uart.sensor_bytes.is_empty());

        hardware
            .rx
            .extend(stream_frame(0, &packet).into_iter().map(SerialRead::Byte));
        uart.poll(&mut hardware, &mut events);
        assert_eq!(events.len(), 2);
        assert!(uart.pending_sensor_packet.is_none());
    }
}
