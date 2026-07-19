use std::env;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, UdpSocket};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use serialport::SerialPort;

use crate::drivers::imu::{ImuHealth, ImuSample};
use crate::hardware::{BrainstemHardware, SerialRead, UartReadError};
use crate::runtime::{Runtime, RUNTIME_TICK_MS};

const DEFAULT_CREATE_BAUD: u32 = 57_600;
const DEFAULT_LISTEN: &str = "127.0.0.1:8787";
const CONTROL_PACKET_MAX: usize = 4096;

#[derive(Clone, Debug)]
pub struct Rpi5Config {
    pub create_port: PathBuf,
    pub create_baud: u32,
    pub listen: SocketAddr,
}

impl Rpi5Config {
    pub fn from_env() -> Result<Self, ConfigError> {
        let create_port = env::var_os("PETE_CREATE_PORT")
            .map(PathBuf::from)
            .ok_or(ConfigError::MissingCreatePort)?;
        let create_baud = env::var("PETE_CREATE_BAUD")
            .ok()
            .map(|value| {
                value
                    .parse()
                    .map_err(|_| ConfigError::InvalidCreateBaud(value))
            })
            .transpose()?
            .unwrap_or(DEFAULT_CREATE_BAUD);
        let listen_text = env::var("PETE_BRAINSTEM_LISTEN")
            .or_else(|_| env::var("PETE_BRAINSTEM_LOCAL_ADDR"))
            .unwrap_or_else(|_| DEFAULT_LISTEN.to_owned());
        let listen = parse_listen_address(listen_text)?;
        Ok(Self {
            create_port,
            create_baud,
            listen,
        })
    }
}

#[derive(Debug)]
pub enum ConfigError {
    MissingCreatePort,
    InvalidCreateBaud(String),
    InvalidListenAddress(String),
    NonLoopbackListenAddress(SocketAddr),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingCreatePort => write!(
                f,
                "PETE_CREATE_PORT must name the Create 1 side-port serial adapter"
            ),
            Self::InvalidCreateBaud(value) => {
                write!(f, "PETE_CREATE_BAUD is not a valid baud rate: {value}")
            }
            Self::InvalidListenAddress(value) => {
                write!(f, "PETE_BRAINSTEM_LISTEN is not a socket address: {value}")
            }
            Self::NonLoopbackListenAddress(address) => write!(
                f,
                "RPi 5 Brainstem control must bind loopback, not {address}"
            ),
        }
    }
}

fn parse_listen_address(text: String) -> Result<SocketAddr, ConfigError> {
    let listen =
        SocketAddr::from_str(&text).map_err(|_| ConfigError::InvalidListenAddress(text))?;
    if !listen.ip().is_loopback() {
        return Err(ConfigError::NonLoopbackListenAddress(listen));
    }
    Ok(listen)
}

impl std::error::Error for ConfigError {}

pub fn run(config: Rpi5Config) -> Result<(), Box<dyn std::error::Error>> {
    let hardware = Rpi5Hardware::open(&config)?;
    let control = UdpSocket::bind(config.listen)?;
    control.set_read_timeout(Some(Duration::from_millis(100)))?;

    let stopping = Arc::new(AtomicBool::new(false));
    let signal_stopping = Arc::clone(&stopping);
    ctrlc::set_handler(move || {
        signal_stopping.store(true, Ordering::Release);
    })?;

    crate::rpi5_control::initialize_identity()?;
    let control_stopping = Arc::clone(&stopping);
    let control_thread = thread::Builder::new()
        .name("pete-brainstem-control".into())
        .spawn(move || serve_control(control, control_stopping))?;

    eprintln!(
        "pete-brainstem-rpi5: Create OI {} at {} baud; Cockpit UDP {}",
        config.create_port.display(),
        config.create_baud,
        config.listen
    );

    let mut runtime = Runtime::new(hardware);
    runtime.start();
    while !stopping.load(Ordering::Acquire) {
        let tick_started = Instant::now();
        runtime.tick();
        let remaining = Duration::from_millis(u64::from(RUNTIME_TICK_MS))
            .saturating_sub(tick_started.elapsed());
        if !remaining.is_zero() {
            thread::sleep(remaining);
        }
    }
    runtime.shutdown();

    control_thread
        .join()
        .map_err(|_| "RPi 5 Brainstem control thread panicked")??;
    Ok(())
}

fn serve_control(socket: UdpSocket, stopping: Arc<AtomicBool>) -> io::Result<()> {
    let mut request = [0u8; CONTROL_PACKET_MAX];
    while !stopping.load(Ordering::Acquire) {
        let (length, peer) = match socket.recv_from(&mut request) {
            Ok(received) => received,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                continue;
            }
            Err(error) => return Err(error),
        };
        let response = crate::rpi5_control::handle_packet(&request[..length]);
        socket.send_to(response.as_bytes(), peer)?;
    }
    Ok(())
}

struct Rpi5Hardware {
    started: Instant,
    create: Box<dyn SerialPort>,
}

impl Rpi5Hardware {
    fn open(config: &Rpi5Config) -> Result<Self, serialport::Error> {
        let create = serialport::new(config.create_port.to_string_lossy(), config.create_baud)
            .timeout(Duration::from_millis(1))
            .open()?;
        Ok(Self {
            started: Instant::now(),
            create,
        })
    }
}

impl BrainstemHardware for Rpi5Hardware {
    fn delay_ms(&mut self, ms: u32) {
        thread::sleep(Duration::from_millis(u64::from(ms)));
    }

    fn now_us(&mut self) -> u32 {
        self.started.elapsed().as_micros() as u32
    }

    fn feed_watchdog(&mut self) {}

    fn begin_power_toggle_pulse(&mut self) {
        // A normal Create side-port USB serial lead exposes OI TX/RX/GND, not
        // the isolated power-button circuit owned by the Pico board.
    }

    fn end_power_toggle_pulse(&mut self) {
        // A normal Create side-port USB serial lead exposes OI TX/RX/GND, not
        // the isolated power-button circuit owned by the Pico board.
    }

    fn set_indicators(&mut self, _on: bool) {}

    fn set_primary_indicator(&mut self, _on: bool) {}

    fn write_byte(&mut self, byte: u8) -> Result<(), ()> {
        self.create.write_all(&[byte]).map_err(|_| ())
    }

    fn flush_uart(&mut self) -> Result<(), ()> {
        self.create.flush().map_err(|_| ())
    }

    fn read_byte(&mut self) -> SerialRead {
        let mut byte = [0u8; 1];
        match self.create.read(&mut byte) {
            Ok(1) => SerialRead::Byte(byte[0]),
            Ok(_) => SerialRead::WouldBlock,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                SerialRead::WouldBlock
            }
            Err(error) => SerialRead::Error(map_serial_error(error.kind())),
        }
    }

    fn set_create_uart_baud(&mut self, baud: u32) -> Result<(), ()> {
        self.create.set_baud_rate(baud).map_err(|_| ())
    }

    fn poll_imu_sample(&mut self, _now_ms: u32) -> Result<Option<ImuSample>, ImuHealth> {
        Err(ImuHealth::Absent)
    }
}

fn map_serial_error(kind: io::ErrorKind) -> UartReadError {
    match kind {
        io::ErrorKind::InvalidData => UartReadError::Framing,
        _ => UartReadError::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_address_must_remain_loopback() {
        assert_eq!(
            parse_listen_address("127.0.0.1:8787".into()).unwrap(),
            "127.0.0.1:8787".parse().unwrap()
        );
        assert!(matches!(
            parse_listen_address("0.0.0.0:8787".into()),
            Err(ConfigError::NonLoopbackListenAddress(_))
        ));
    }
}
