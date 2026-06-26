use anyhow::{Context, Result};
use async_trait::async_trait;
#[cfg(feature = "serial")]
use netherwick_body::CliffSensors;
use netherwick_body::{BodySense, MotionCommand, MotorCommand, MotorComplex, RobotBody};
use serde::{Deserialize, Serialize};
#[cfg(feature = "serial")]
use serialport::SerialPort;
#[cfg(feature = "serial")]
use std::io::Write;
#[cfg(feature = "serial")]
use std::time::Duration;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Create1Config {
    pub port: Option<String>,
    pub baud_rate: u32,
    pub wheel_base_m: f32,
    pub max_velocity_m_s: f32,
    #[serde(default)]
    pub open_mode: Create1OpenMode,
    pub use_safe_mode: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Create1OpenMode {
    #[default]
    Passive,
    Safe,
    Full,
}

impl Create1Config {
    pub fn normalized(mut self) -> Self {
        if self.baud_rate == 0 {
            self.baud_rate = 57_600;
        }
        if self.wheel_base_m <= 0.0 {
            self.wheel_base_m = 0.26;
        }
        if self.max_velocity_m_s <= 0.0 {
            self.max_velocity_m_s = 0.3;
        }
        if self.use_safe_mode {
            self.open_mode = Create1OpenMode::Safe;
        }
        self
    }
}

#[derive(Debug)]
pub struct Create1Body {
    body: BodySense,
    #[cfg(feature = "serial")]
    port: Option<Box<dyn SerialPort>>,
    config: Create1Config,
}

impl Default for Create1Body {
    fn default() -> Self {
        Self {
            body: BodySense::default(),
            #[cfg(feature = "serial")]
            port: None,
            config: Create1Config::default().normalized(),
        }
    }
}

impl Create1Body {
    pub fn new(config: Create1Config) -> Result<Self> {
        let config = config.normalized();
        let body = Self {
            body: BodySense::default(),
            #[cfg(feature = "serial")]
            port: None,
            config,
        };
        #[cfg(feature = "serial")]
        let mut body = body;
        #[cfg(feature = "serial")]
        if let Some(path) = body.config.port.clone() {
            let port = serialport::new(&path, body.config.baud_rate)
                .timeout(Duration::from_millis(500))
                .open()
                .with_context(|| format!("failed to open Create serial port {path}"))?;
            body.port = Some(port);
            body.initialize()?;
        }
        #[cfg(not(feature = "serial"))]
        if body.config.port.is_some() {
            anyhow::bail!(
                "Create1 serial port support requires the netherwick-create1 `serial` feature"
            );
        }
        Ok(body)
    }

    pub async fn connect(path: &str, baud: u32) -> Result<Self> {
        Self::connect_with_mode(path, baud, Create1OpenMode::Passive).await
    }

    pub async fn connect_with_mode(
        path: &str,
        baud: u32,
        open_mode: Create1OpenMode,
    ) -> Result<Self> {
        Self::new(Create1Config {
            port: Some(path.to_string()),
            baud_rate: baud,
            open_mode,
            ..Create1Config::default()
        })
    }

    #[cfg(feature = "serial")]
    fn initialize(&mut self) -> Result<()> {
        self.write_bytes(&[128])?;
        std::thread::sleep(Duration::from_millis(250));
        match self.config.open_mode {
            Create1OpenMode::Passive => {}
            Create1OpenMode::Safe => self.write_bytes(&[131])?,
            Create1OpenMode::Full => self.write_bytes(&[132])?,
        }
        if self.config.open_mode != Create1OpenMode::Passive {
            std::thread::sleep(Duration::from_millis(250));
        }
        Ok(())
    }

    #[cfg(feature = "serial")]
    fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        if let Some(port) = self.port.as_mut() {
            port.write_all(bytes)?;
            port.flush()?;
        }
        Ok(())
    }

    #[cfg(not(feature = "serial"))]
    fn write_bytes(&mut self, _bytes: &[u8]) -> Result<()> {
        Ok(())
    }

    fn drive_direct(&mut self, cmd: MotorCommand) -> Result<()> {
        let limited = cmd.clamped(self.config.max_velocity_m_s, self.config.max_velocity_m_s);
        let half_wheel_base = self.config.wheel_base_m / 2.0;
        let left_m_s = limited.forward - (limited.turn * half_wheel_base);
        let right_m_s = limited.forward + (limited.turn * half_wheel_base);
        let left_mm_s = meters_per_second_to_mm_per_second(left_m_s);
        let right_mm_s = meters_per_second_to_mm_per_second(right_m_s);
        let mut packet = vec![145];
        packet.extend_from_slice(&right_mm_s.to_be_bytes());
        packet.extend_from_slice(&left_mm_s.to_be_bytes());
        self.write_bytes(&packet)
    }

    #[cfg(feature = "serial")]
    fn refresh_sensors(&mut self) -> Result<()> {
        let Some(port) = self.port.as_mut() else {
            return Ok(());
        };
        const PACKETS: [(u8, usize); 12] = [
            (7, 1),
            (8, 1),
            (9, 1),
            (10, 1),
            (11, 1),
            (12, 1),
            (13, 1),
            (21, 1),
            (28, 2),
            (29, 2),
            (30, 2),
            (31, 2),
        ];
        let mut bytes = Vec::with_capacity(16);
        for (packet_id, packet_len) in PACKETS {
            read_sensor_packet(port.as_mut(), packet_id, packet_len, &mut bytes)?;
        }
        let bytes: [u8; 16] = bytes.try_into().map_err(|bytes: Vec<u8>| {
            anyhow::anyhow!("expected 16 Create sensor bytes, got {}", bytes.len())
        })?;
        let bumps = bytes[0];
        self.body.flags.bump_right = bumps & 0b0000_0001 != 0;
        self.body.flags.bump_left = bumps & 0b0000_0010 != 0;
        self.body.flags.wheel_drop = bumps & 0b0001_1100 != 0;
        self.body.flags.wall = bytes[1] != 0;
        self.body.flags.cliff_left = bytes[2] != 0;
        self.body.flags.cliff_front_left = bytes[3] != 0;
        self.body.flags.cliff_front_right = bytes[4] != 0;
        self.body.flags.cliff_right = bytes[5] != 0;
        self.body.flags.virtual_wall = bytes[6] != 0;
        self.body.charging = bytes[7] != 0;

        let left_signal = u16::from_be_bytes([bytes[8], bytes[9]]);
        let front_left_signal = u16::from_be_bytes([bytes[10], bytes[11]]);
        let front_right_signal = u16::from_be_bytes([bytes[12], bytes[13]]);
        let right_signal = u16::from_be_bytes([bytes[14], bytes[15]]);
        self.body.cliff_sensors = CliffSensors {
            left: create1_cliff_risk(left_signal, self.body.flags.cliff_left),
            front_left: create1_cliff_risk(front_left_signal, self.body.flags.cliff_front_left),
            front_right: create1_cliff_risk(front_right_signal, self.body.flags.cliff_front_right),
            right: create1_cliff_risk(right_signal, self.body.flags.cliff_right),
        };
        Ok(())
    }

    #[cfg(not(feature = "serial"))]
    fn refresh_sensors(&mut self) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct MockCreate1Body {
    body: BodySense,
    pub motor_attempts: usize,
}

impl MockCreate1Body {
    pub fn new() -> Self {
        Self {
            body: BodySense::default(),
            motor_attempts: 0,
        }
    }

    pub fn with_body(body: BodySense) -> Self {
        Self {
            body,
            motor_attempts: 0,
        }
    }
}

#[async_trait]
impl RobotBody for MockCreate1Body {
    async fn read_body(&mut self) -> Result<BodySense> {
        self.body.last_update_ms = self.body.last_update_ms.saturating_add(100);
        Ok(self.body.clone())
    }

    async fn apply_motor(&mut self, _cmd: MotorCommand) -> Result<()> {
        self.motor_attempts = self.motor_attempts.saturating_add(1);
        anyhow::bail!("MockCreate1Body refuses motor commands in read-only bring-up")
    }
}

#[async_trait]
impl RobotBody for Create1Body {
    async fn read_body(&mut self) -> Result<BodySense> {
        self.refresh_sensors()?;
        Ok(self.body.clone())
    }

    async fn apply_motor(&mut self, cmd: MotorCommand) -> Result<()> {
        self.drive_direct(cmd)?;
        self.body.velocity.forward_m_s = cmd.forward;
        self.body.velocity.turn_rad_s = cmd.turn;
        self.body.last_update_ms = self.body.last_update_ms.saturating_add(100);
        Ok(())
    }
}

#[async_trait]
impl MotorComplex for Create1Body {
    async fn send(&mut self, command: MotionCommand) -> Result<BodySense> {
        let motor = command.to_motor_command();
        self.apply_motor(motor).await?;
        self.read_body().await
    }
}

fn meters_per_second_to_mm_per_second(value: f32) -> i16 {
    let scaled = (value * 1000.0).round();
    scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

#[cfg(feature = "serial")]
fn create1_cliff_risk(signal: u16, tripped: bool) -> f32 {
    if tripped {
        1.0
    } else {
        (1.0 - signal as f32 / 4095.0).clamp(0.0, 1.0)
    }
}

#[cfg(feature = "serial")]
fn read_sensor_packet(
    port: &mut dyn SerialPort,
    packet_id: u8,
    packet_len: usize,
    output: &mut Vec<u8>,
) -> Result<()> {
    port.write_all(&[142, packet_id])?;
    port.flush()?;
    let start_len = output.len();
    output.resize(start_len + packet_len, 0);
    port.read_exact(&mut output[start_len..])?;
    Ok(())
}
