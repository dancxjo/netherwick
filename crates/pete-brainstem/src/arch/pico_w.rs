use core::{convert::Infallible, fmt::Write as _};

use cyw43::aligned_bytes;
use cyw43_pio::{PioSpi, DEFAULT_CLOCK_DIVIDER};
use embassy_executor::Spawner;
use embassy_futures::select::{select, Either};
use embassy_net::tcp::TcpSocket;
use embassy_net::udp::{PacketMetadata, UdpSocket};
use embassy_net::{
    Config as NetConfig, HardwareAddress, IpAddress, IpEndpoint, Ipv4Address, Ipv4Cidr, Stack,
    StackResources,
};
use embassy_net_driver::{Driver as NetDriver, RxToken as _, TxToken as _};
use embassy_rp::gpio::{Input, Level, Output, Pull};
use embassy_rp::i2c::{
    Async as I2cAsync, Config as I2cConfig, I2c, InterruptHandler as I2cInterruptHandler,
};
use embassy_rp::multicore::{spawn_core1, Stack as CoreStack};
use embassy_rp::peripherals::{
    DMA_CH0, I2C1, PIN_0, PIN_1, PIN_17, PIN_18, PIN_19, PIN_2, PIN_20, PIN_23, PIN_24, PIN_25,
    PIN_29, PIN_3, PIN_4, PIN_5, PIO0, UART0, UART1, USB,
};
use embassy_rp::pio::{InterruptHandler as PioInterruptHandler, Pio};
use embassy_rp::rom_data::reset_to_usb_boot;
use embassy_rp::uart::{
    Blocking, Config as UartConfig, DataBits, Error as UartError, Parity, StopBits, Uart,
};
use embassy_rp::{bind_interrupts, dma, Peri};
use embassy_time::{Duration, Instant, Timer};
use embassy_usb::class::cdc_acm::{CdcAcmClass, State as CdcAcmState};
use embassy_usb::UsbDevice;
use embedded_hal_nb::serial::Read as _;
use embedded_io_async::Write;
use pete_cockpit_protocol::TransportKind;
use portable_atomic::{AtomicU32, Ordering};
use static_cell::StaticCell;

use crate::body;
use crate::capabilities;
use crate::commands::{
    BrainstemCommand, CreateOiMode, FeedbackKind, PowerStateRequest, SafetyLatchKind, SongTone,
    MAX_SONG_TONES,
};
use crate::dhcp::{DhcpClient, DhcpGrant, DhcpLeaseState, DhcpRequest, DHCP_LEASE_SECONDS};
use crate::display::{self, DisplayNetwork, DisplaySafety, DisplayStatus};
use crate::drivers::imu::{decode_mpu6050_sample, ImuHealth};
use crate::hardware::{initialize_power_control, BrainstemHardware, SerialRead, UartReadError};
use crate::icmp::{
    process_icmp_echo_frame, IcmpEchoDisposition, IcmpRateLimit, NETWORK_FRAME_CAPACITY,
};
use crate::network_registry;
use crate::runtime::Runtime;
use crate::session;
use crate::status;

const AP_SSID_PREFIX: &str = "pete-";
const INSTANCE_ID_BASE: u32 = 36;
const INSTANCE_ID_MODULUS: u32 = INSTANCE_ID_BASE.pow(4);
const MDNS_NAME: &[u8] = b"\x04pete\x05local\x00";
const AP_CHANNEL: u8 = 6;
const AP_IP_OCTETS: [u8; 4] = [192, 168, 4, 1];
const AP_IP: Ipv4Address = Ipv4Address::new(192, 168, 4, 1);
const AP_INSTANCE_UNKNOWN: u32 = u32::MAX;
static AP_INSTANCE_ID: AtomicU32 = AtomicU32::new(AP_INSTANCE_UNKNOWN);
const HTTP_PORT: u16 = 80;
const HTTP_TASKS: usize = 3;
const WS_CONTROL_PORT: u16 = 81;
const UDP_CONTROL_PORT: u16 = 82;
const DNS_PORT: u16 = 53;
const MDNS_PORT: u16 = 5353;
const DHCP_SERVER_PORT: u16 = 67;
const DHCP_CLIENT_PORT: u16 = 68;
const HTTP_FLUSH_TIMEOUT_MS: u64 = 250;
const SSE_STATUS_INTERVAL_MS: u64 = 750;
const SSE_EVENT_CHECK_INTERVAL_MS: u64 = 100;
const LED_HEARTBEAT_INTERVAL_SECS: u64 = 15;
const LED_BLINK_ON_MS: u64 = 120;
const LED_BLINK_OFF_MS: u64 = 120;
const FOREBRAIN_UART_BAUD: u32 = 115_200;
const FOREBRAIN_LINE_MAX: usize = 1024;
const FOREBRAIN_POLL_MS: u64 = 2;
const FOREBRAIN_LINE_TIMEOUT_MS: u32 = 100;
const SENSOR_I2C_FREQUENCY_HZ: u32 = 100_000;
const SENSOR_I2C_TIMEOUT_MS: u64 = 25;
const IMU_RETRY_MS: u64 = 250;
const MPU6050_ADDRESS_LOW: u8 = 0x68;
const MPU6050_ADDRESS_HIGH: u8 = 0x69;
const MPU6050_WHO_AM_I: u8 = 0x75;
const MPU6050_PWR_MGMT_1: u8 = 0x6b;
const MPU6050_GYRO_CONFIG: u8 = 0x1b;
const MPU6050_ACCEL_CONFIG: u8 = 0x1c;
const MPU6050_ACCEL_XOUT_H: u8 = 0x3b;
const OLED_ADDRESSES: [u8; 2] = [0x3c, 0x3d];
const OLED_RETRY_MS: u32 = 5_000;
const OLED_IO_INTERVAL_MS: u32 = 4;
const OLED_I2C_TIMEOUT_MS: u64 = 8;
const OLED_CHUNK_BYTES: usize = 32;
const OLED_INIT_COMMANDS: [u8; 26] = [
    0x00, 0xae, 0xd5, 0x80, 0xa8, 0x1f, 0xd3, 0x00, 0x40, 0x8d, 0x14, 0x20, 0x02, 0xa1, 0xc8, 0xda,
    0x02, 0x81, 0x8f, 0xd9, 0xf1, 0xdb, 0x40, 0xa4, 0xa6, 0xaf,
];

static mut CORE1_STACK: CoreStack<8192> = CoreStack::new();
static BRAINSTEM_INSTANCE_ID: AtomicU32 = AtomicU32::new(0);
static BRAINSTEM_BOOT_ID: AtomicU32 = AtomicU32::new(0);
static AUTHORITY_GENERATION: AtomicU32 = AtomicU32::new(0);
static SERVICE_GENERATION: AtomicU32 = AtomicU32::new(0);

/// A narrow device-level ICMP responder. It has no route, socket, or robot
/// handles: it only consumes bounded IPv4 echo requests addressed to the AP.

// Pico W domains share one namespace to keep embedded paths and contracts stable.
include!("pico_w/hardware.rs");
include!("pico_w/wifi.rs");
include!("pico_w/peripherals.rs");
include!("pico_w/network.rs");
include!("pico_w/http.rs");
include!("pico_w/handshake.rs");
include!("pico_w/commands.rs");
include!("pico_w/authority.rs");
include!("pico_w/uart.rs");
include!("pico_w/json.rs");
include!("pico_w/discovery.rs");

#[cfg(test)]
#[path = "pico_w_tests.rs"]
mod tests;
