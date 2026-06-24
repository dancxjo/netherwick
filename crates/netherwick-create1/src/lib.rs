use anyhow::Result;
use async_trait::async_trait;
use netherwick_body::{BodySense, MotionCommand, MotorCommand, MotorComplex, RobotBody};
use serde::{Deserialize, Serialize};
use serialport::SerialPort;
use std::io::Write;
use std::time::Duration;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Create1Config {
    pub port: Option<String>,
    pub baud_rate: u32,
    pub wheel_base_m: f32,
    pub max_velocity_m_s: f32,
    pub use_safe_mode: bool,
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
        self
    }
}

#[derive(Debug)]
pub struct Create1Body {
    body: BodySense,
    port: Option<Box<dyn SerialPort>>,
    config: Create1Config,
}

impl Default for Create1Body {
    fn default() -> Self {
        Self {
            body: BodySense::default(),
            port: None,
            config: Create1Config::default().normalized(),
        }
    }
}

impl Create1Body {
    pub fn new(config: Create1Config) -> Result<Self> {
        let config = config.normalized();
        let mut body = Self {
            body: BodySense::default(),
            port: None,
            config,
        };
        if let Some(path) = body.config.port.clone() {
            let port = serialport::new(path, body.config.baud_rate)
                .timeout(Duration::from_millis(100))
                .open()?;
            body.port = Some(port);
            body.initialize()?;
        }
        Ok(body)
    }

    fn initialize(&mut self) -> Result<()> {
        self.write_bytes(&[128])?;
        self.write_bytes(&[if self.config.use_safe_mode { 131 } else { 132 }])?;
        Ok(())
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        if let Some(port) = self.port.as_mut() {
            port.write_all(bytes)?;
            port.flush()?;
        }
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
}

#[async_trait]
impl RobotBody for Create1Body {
    async fn read_body(&mut self) -> Result<BodySense> {
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
