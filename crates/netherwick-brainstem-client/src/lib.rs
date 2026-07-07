use std::net::{SocketAddr, UdpSocket};
use std::time::Duration;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, BrainstemClientError>;

#[derive(Debug, Error)]
pub enum BrainstemClientError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad brainstem response: {0}")]
    BadResponse(String),
    #[error("event history missed before sequence {dropped_before_seq}")]
    MissedEvents { dropped_before_seq: u32 },
}

pub trait BrainstemClient {
    fn get_status(&mut self) -> Result<BrainstemStatus>;
    fn get_capabilities(&mut self) -> Result<BrainstemCapabilities>;
    fn get_events_since(&mut self, since_seq: u32) -> Result<EventBatch>;
    fn arm(&mut self) -> Result<()>;
    fn disarm(&mut self) -> Result<()>;
    fn stop(&mut self) -> Result<()>;
    fn estop(&mut self) -> Result<()>;
    fn clear_estop(&mut self) -> Result<()>;
    fn cmd_vel(&mut self, linear_mm_s: i16, angular_mrad_s: i16, ttl_ms: u32) -> Result<()>;
    fn heartbeat_stop(&mut self, timeout_ms: u32) -> Result<()>;
    fn stream_sensors(&mut self, enabled: bool, packet_id: u8, period_ms: u32) -> Result<()>;
    fn reset_odometry(&mut self) -> Result<()>;
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BrainstemStatus {
    pub raw: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BrainstemCapabilities {
    pub body_kind: String,
    pub drive: String,
    pub verbs: Vec<String>,
    pub sensors: Vec<String>,
    pub outputs: Vec<String>,
    pub safety: Vec<String>,
    pub events: Vec<String>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct EventBatch {
    pub since_seq: u32,
    pub oldest_seq: u32,
    pub next_seq: u32,
    pub dropped_before_seq: u32,
    pub events: Vec<BrainstemEvent>,
}

impl EventBatch {
    pub fn ensure_no_missed_events(&self) -> Result<()> {
        if self.dropped_before_seq == 0 {
            Ok(())
        } else {
            Err(BrainstemClientError::MissedEvents {
                dropped_before_seq: self.dropped_before_seq,
            })
        }
    }

    pub fn has_stop_reason(&self) -> bool {
        self.events.iter().any(BrainstemEvent::is_stop_reason)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BrainstemEvent {
    pub seq: u32,
    pub kind: BrainstemEventKind,
    pub a: u32,
    pub b: u32,
    pub c: u32,
}

impl BrainstemEvent {
    pub fn is_stop_reason(&self) -> bool {
        matches!(
            self.kind,
            BrainstemEventKind::SafetyTripped
                | BrainstemEventKind::HeartbeatExpired
                | BrainstemEventKind::EStopLatched
        )
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum BrainstemEventKind {
    Boot,
    CommandAccepted,
    CommandRejected,
    CommandStarted,
    CommandCompleted,
    CommandInterrupted,
    CommandTimedOut,
    BodyPowerRequested,
    BodyPowerChanged,
    BodyModeRequested,
    BodyModeChanged,
    TelemetryReceived,
    SensorFrameDecoded,
    MotionRequested,
    MotionStopped,
    SafetyTripped,
    SafetyCleared,
    BumpChanged,
    CliffChanged,
    WheelDropLatched,
    WheelDropCleared,
    HeartbeatExpired,
    EStopLatched,
    EStopCleared,
    Error,
    Unknown(String),
}

impl From<&str> for BrainstemEventKind {
    fn from(kind: &str) -> Self {
        match kind {
            "boot" => Self::Boot,
            "command_accepted" => Self::CommandAccepted,
            "command_rejected" => Self::CommandRejected,
            "command_started" => Self::CommandStarted,
            "command_completed" => Self::CommandCompleted,
            "command_interrupted" => Self::CommandInterrupted,
            "command_timed_out" => Self::CommandTimedOut,
            "body_power_requested" => Self::BodyPowerRequested,
            "body_power_changed" => Self::BodyPowerChanged,
            "body_mode_requested" => Self::BodyModeRequested,
            "body_mode_changed" => Self::BodyModeChanged,
            "telemetry_received" => Self::TelemetryReceived,
            "sensor_frame_decoded" => Self::SensorFrameDecoded,
            "motion_requested" => Self::MotionRequested,
            "motion_stopped" => Self::MotionStopped,
            "safety_tripped" => Self::SafetyTripped,
            "safety_cleared" => Self::SafetyCleared,
            "bump_changed" => Self::BumpChanged,
            "cliff_changed" => Self::CliffChanged,
            "wheel_drop_latched" => Self::WheelDropLatched,
            "wheel_drop_cleared" => Self::WheelDropCleared,
            "heartbeat_expired" => Self::HeartbeatExpired,
            "estop_latched" => Self::EStopLatched,
            "estop_cleared" => Self::EStopCleared,
            "error" => Self::Error,
            other => Self::Unknown(other.to_owned()),
        }
    }
}

pub struct EventCursor {
    next_seq: u32,
}

impl EventCursor {
    pub fn new() -> Self {
        Self { next_seq: 0 }
    }

    pub fn next_seq(&self) -> u32 {
        self.next_seq
    }

    pub fn poll<C: BrainstemClient>(&mut self, client: &mut C) -> Result<EventBatch> {
        let batch = client.get_events_since(self.next_seq)?;
        batch.ensure_no_missed_events()?;
        self.next_seq = batch.next_seq;
        Ok(batch)
    }
}

impl Default for EventCursor {
    fn default() -> Self {
        Self::new()
    }
}

pub struct UdpBrainstemClient {
    socket: UdpSocket,
    brainstem: SocketAddr,
    next_seq: u32,
    timeout: Duration,
}

impl UdpBrainstemClient {
    pub fn connect(brainstem: SocketAddr) -> Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        let timeout = Duration::from_millis(750);
        socket.set_read_timeout(Some(timeout))?;
        socket.set_write_timeout(Some(timeout))?;
        Ok(Self {
            socket,
            brainstem,
            next_seq: 1,
            timeout,
        })
    }

    pub fn set_timeout(&mut self, timeout: Duration) -> Result<()> {
        self.timeout = timeout;
        self.socket.set_read_timeout(Some(timeout))?;
        self.socket.set_write_timeout(Some(timeout))?;
        Ok(())
    }

    fn request(&mut self, line: String) -> Result<String> {
        self.socket.send_to(line.as_bytes(), self.brainstem)?;
        let mut buf = [0u8; 2048];
        let (len, _) = self.socket.recv_from(&mut buf)?;
        let response = std::str::from_utf8(&buf[..len])
            .map_err(|_| BrainstemClientError::BadResponse("response was not utf-8".into()))?
            .trim()
            .to_owned();
        Ok(response)
    }

    fn seq(&mut self) -> u32 {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1).max(1);
        seq
    }

    fn command(&mut self, kind: &str) -> Result<()> {
        let seq = self.seq();
        expect_ok(seq, &self.request(format!("{kind} {seq}\n"))?)
    }
}

impl BrainstemClient for UdpBrainstemClient {
    fn get_status(&mut self) -> Result<BrainstemStatus> {
        let seq = self.seq();
        let response = self.request(format!("STATUS {seq}\n"))?;
        expect_ok(seq, &response)?;
        Ok(BrainstemStatus { raw: response })
    }

    fn get_capabilities(&mut self) -> Result<BrainstemCapabilities> {
        let seq = self.seq();
        let response = self.request(format!("GET_CAPABILITIES {seq}\n"))?;
        parse_capabilities(seq, &response)
    }

    fn get_events_since(&mut self, since_seq: u32) -> Result<EventBatch> {
        let seq = self.seq();
        let response = self.request(format!("GET_EVENTS {seq} {since_seq}\n"))?;
        parse_events(seq, since_seq, &response)
    }

    fn arm(&mut self) -> Result<()> {
        self.command("ARM")
    }

    fn disarm(&mut self) -> Result<()> {
        self.command("DISARM")
    }

    fn stop(&mut self) -> Result<()> {
        self.command("STOP")
    }

    fn estop(&mut self) -> Result<()> {
        self.command("ESTOP")
    }

    fn clear_estop(&mut self) -> Result<()> {
        self.command("CLEAR_ESTOP")
    }

    fn cmd_vel(&mut self, linear_mm_s: i16, angular_mrad_s: i16, ttl_ms: u32) -> Result<()> {
        let seq = self.seq();
        expect_ok(
            seq,
            &self.request(format!(
                "CMD_VEL {seq} {linear_mm_s} {angular_mrad_s} {ttl_ms}\n"
            ))?,
        )
    }

    fn heartbeat_stop(&mut self, timeout_ms: u32) -> Result<()> {
        let seq = self.seq();
        expect_ok(
            seq,
            &self.request(format!("HEARTBEAT_STOP {seq} {timeout_ms}\n"))?,
        )
    }

    fn stream_sensors(&mut self, enabled: bool, packet_id: u8, period_ms: u32) -> Result<()> {
        let seq = self.seq();
        let enabled = if enabled { "true" } else { "false" };
        expect_ok(
            seq,
            &self.request(format!(
                "STREAM_SENSORS {seq} {enabled} {packet_id} {period_ms}\n"
            ))?,
        )
    }

    fn reset_odometry(&mut self) -> Result<()> {
        self.command("RESET_ODOMETRY")
    }
}

fn expect_ok(seq: u32, response: &str) -> Result<()> {
    let expected = format!("OK {seq}");
    if response.starts_with(&expected) {
        Ok(())
    } else {
        Err(BrainstemClientError::BadResponse(response.to_owned()))
    }
}

fn parse_capabilities(seq: u32, response: &str) -> Result<BrainstemCapabilities> {
    expect_ok(seq, response)?;
    let rest = response
        .strip_prefix(&format!("OK {seq} CAPABILITIES "))
        .ok_or_else(|| BrainstemClientError::BadResponse(response.to_owned()))?;
    Ok(BrainstemCapabilities {
        body_kind: value_for(rest, "body_kind").unwrap_or_default().to_owned(),
        drive: value_for(rest, "drive").unwrap_or_default().to_owned(),
        verbs: csv_for(rest, "verbs"),
        sensors: csv_for(rest, "sensors"),
        outputs: csv_for(rest, "outputs"),
        safety: csv_for(rest, "safety"),
        events: csv_for(rest, "events"),
    })
}

fn parse_events(seq: u32, since_seq: u32, response: &str) -> Result<EventBatch> {
    expect_ok(seq, response)?;
    let rest = response
        .strip_prefix(&format!("OK {seq} EVENTS "))
        .ok_or_else(|| BrainstemClientError::BadResponse(response.to_owned()))?;
    let header = rest.split('|').next().unwrap_or(rest);
    let dropped_before_seq = number_for(header, "dropped_before").unwrap_or(0);
    let mut batch = EventBatch {
        since_seq,
        oldest_seq: number_for(header, "oldest").unwrap_or(0),
        next_seq: number_for(header, "next").unwrap_or(since_seq),
        dropped_before_seq,
        events: Vec::new(),
    };
    for chunk in rest.split('|').skip(1) {
        let chunk = chunk.trim();
        let Some((seq_text, tail)) = chunk.split_once(':') else {
            continue;
        };
        let Some((kind_text, fields)) = tail.split_once(':') else {
            continue;
        };
        let mut nums = fields.split(',');
        batch.events.push(BrainstemEvent {
            seq: seq_text.parse().unwrap_or(0),
            kind: BrainstemEventKind::from(kind_text),
            a: nums.next().and_then(|n| n.parse().ok()).unwrap_or(0),
            b: nums.next().and_then(|n| n.parse().ok()).unwrap_or(0),
            c: nums.next().and_then(|n| n.parse().ok()).unwrap_or(0),
        });
    }
    Ok(batch)
}

fn value_for<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{key}=");
    let start = line.find(&prefix)? + prefix.len();
    let tail = &line[start..];
    Some(tail.split_whitespace().next().unwrap_or(tail))
}

fn csv_for(line: &str, key: &str) -> Vec<String> {
    value_for(line, key)
        .unwrap_or("")
        .split(',')
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn number_for(line: &str, key: &str) -> Option<u32> {
    value_for(line, key)?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_compact_events() {
        let batch = parse_events(
            7,
            12,
            "OK 7 EVENTS since=12 oldest=4 next=15 dropped_before=0 count=2 | 13:motion_requested:1,2,3 | 14:safety_tripped:2,0,0",
        )
        .unwrap();
        assert_eq!(batch.next_seq, 15);
        assert_eq!(batch.dropped_before_seq, 0);
        assert_eq!(batch.events.len(), 2);
        assert_eq!(batch.events[1].kind, BrainstemEventKind::SafetyTripped);
        assert!(batch.has_stop_reason());
    }

    #[test]
    fn detects_missed_events() {
        let batch = parse_events(
            1,
            0,
            "OK 1 EVENTS since=0 oldest=20 next=52 dropped_before=20 count=0",
        )
        .unwrap();
        assert!(matches!(
            batch.ensure_no_missed_events(),
            Err(BrainstemClientError::MissedEvents {
                dropped_before_seq: 20
            })
        ));
    }

    #[test]
    fn parses_capabilities_without_body_specific_api() {
        let caps = parse_capabilities(
            3,
            "OK 3 CAPABILITIES body_kind=create_oi drive=differential verbs=arm,stop,cmd_vel sensors=bump,battery outputs=lights,song safety=bump,estop events=boot,safety_tripped limits=max_linear_mm_s:500 max_tones=16 song_slots=16 feedback_slots=6 sensor_packets=0,7-31",
        )
        .unwrap();
        assert_eq!(caps.drive, "differential");
        assert_eq!(caps.verbs, ["arm", "stop", "cmd_vel"]);
        assert_eq!(caps.events, ["boot", "safety_tripped"]);
    }
}
