use core::fmt::Write as _;

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
use embassy_rp::gpio::{Input, Level, Output, OutputOpenDrain, Pull};
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
    BrainstemCommand, CreateOiMode, EscapeDirection, FeedbackKind, PowerStateRequest, SafetyAction,
    SafetyLatchKind, SafetyPolicy, SongTone, MAX_SONG_TONES,
};
use crate::dhcp::{DhcpClient, DhcpGrant, DhcpLeaseState, DhcpRequest, DHCP_LEASE_SECONDS};
use crate::drivers::imu::{decode_mpu6050_sample, ImuHealth};
use crate::hardware::{BrainstemHardware, SerialRead, UartReadError};
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
const IMU_I2C_FREQUENCY_HZ: u32 = 100_000;
const IMU_I2C_TIMEOUT_MS: u64 = 25;
const IMU_RETRY_MS: u64 = 250;
const MPU6050_ADDRESS_LOW: u8 = 0x68;
const MPU6050_ADDRESS_HIGH: u8 = 0x69;
const MPU6050_WHO_AM_I: u8 = 0x75;
const MPU6050_PWR_MGMT_1: u8 = 0x6b;
const MPU6050_GYRO_CONFIG: u8 = 0x1b;
const MPU6050_ACCEL_CONFIG: u8 = 0x1c;
const MPU6050_ACCEL_XOUT_H: u8 = 0x3b;

static mut CORE1_STACK: CoreStack<8192> = CoreStack::new();
static BRAINSTEM_INSTANCE_ID: AtomicU32 = AtomicU32::new(0);
static BRAINSTEM_BOOT_ID: AtomicU32 = AtomicU32::new(0);
static AUTHORITY_GENERATION: AtomicU32 = AtomicU32::new(0);
static SERVICE_GENERATION: AtomicU32 = AtomicU32::new(0);

type UsbDriver = embassy_rp::usb::Driver<'static, USB>;
type BrainstemUsbDevice = UsbDevice<'static, UsbDriver>;

bind_interrupts!(struct Irqs {
    PIO0_IRQ_0 => PioInterruptHandler<PIO0>;
    DMA_IRQ_0 => dma::InterruptHandler<DMA_CH0>;
    I2C1_IRQ => I2cInterruptHandler<I2C1>;
    USBCTRL_IRQ => embassy_rp::usb::InterruptHandler<USB>;
});

pub struct PicoWBrainstem {
    uart: Uart<'static, Blocking>,
    power_toggle: Output<'static>,
    brc: OutputOpenDrain<'static>,
    status_led: Output<'static>,
    charging_indicator: Input<'static>,
    #[cfg(motherbrain_reset_hardware)]
    motherbrain_reset: Output<'static>,
}

impl PicoWBrainstem {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        uart0: Peri<'static, UART0>,
        tx: Peri<'static, PIN_0>,
        rx: Peri<'static, PIN_1>,
        power_toggle: Peri<'static, PIN_18>,
        brc: Peri<'static, PIN_19>,
        status_led: Peri<'static, PIN_20>,
        charging_indicator: Peri<'static, PIN_17>,
        #[cfg(motherbrain_reset_hardware)] motherbrain_reset: Peri<
            'static,
            embassy_rp::peripherals::PIN_21,
        >,
    ) -> Self {
        let mut uart_config = UartConfig::default();
        uart_config.baudrate = body::CREATE_UART_BAUD;
        uart_config.data_bits = DataBits::DataBits8;
        uart_config.stop_bits = StopBits::STOP1;
        uart_config.parity = Parity::ParityNone;
        let uart = Uart::new_blocking(uart0, tx, rx, uart_config);
        // UART idles high. Keep the Create RX input at a defined idle level
        // while the robot or level shifter is unpowered instead of accepting
        // the RP2040 pad's reset pull-down as break/framing noise.
        rp_pac::PADS_BANK0.gpio(1).modify(|w| {
            w.set_pue(true);
            w.set_pde(false);
        });
        Self {
            uart,
            power_toggle: Output::new(power_toggle, Level::Low),
            brc: {
                let mut brc = OutputOpenDrain::new(brc, Level::High);
                brc.set_pullup(false);
                brc
            },
            status_led: Output::new(status_led, Level::Low),
            charging_indicator: Input::new(
                charging_indicator,
                if body::CREATE_CHARGING_INDICATOR_ACTIVE_HIGH {
                    Pull::Down
                } else {
                    Pull::Up
                },
            ),
            // The gate/base has an external pull-down. Construct it inactive
            // before any command transport starts.
            #[cfg(motherbrain_reset_hardware)]
            motherbrain_reset: Output::new(motherbrain_reset, Level::Low),
        }
    }
}

impl BrainstemHardware for PicoWBrainstem {
    fn delay_ms(&mut self, ms: u32) {
        embassy_time::block_for(Duration::from_millis(ms as u64));
    }

    fn now_us(&mut self) -> u32 {
        Instant::now().as_micros() as u32
    }

    fn feed_watchdog(&mut self) {
        // Watchdog plumbing is owned by the runtime safety lane; this Pico W
        // backend currently leaves the hardware watchdog disabled.
    }

    fn set_power_toggle(&mut self, high: bool) {
        self.power_toggle.set_level(level(high));
    }

    fn set_brc(&mut self, released: bool) {
        self.brc.set_level(level(released));
    }

    fn set_indicators(&mut self, on: bool) {
        self.status_led.set_level(level(on));
    }

    fn set_primary_indicator(&mut self, on: bool) {
        self.status_led.set_level(level(on));
    }

    fn set_motherbrain_reset(&mut self, asserted: bool) {
        #[cfg(motherbrain_reset_hardware)]
        self.motherbrain_reset.set_level(level(asserted));
        #[cfg(not(motherbrain_reset_hardware))]
        let _ = asserted;
    }

    fn write_byte(&mut self, byte: u8) -> Result<(), ()> {
        self.uart.blocking_write(&[byte]).map_err(|_| ())
    }

    fn flush_uart(&mut self) -> Result<(), ()> {
        self.uart.blocking_flush().map_err(|_| ())
    }

    fn read_byte(&mut self) -> SerialRead {
        match self.uart.read() {
            Ok(byte) => SerialRead::Byte(byte),
            Err(nb::Error::WouldBlock) => SerialRead::WouldBlock,
            Err(nb::Error::Other(error)) => SerialRead::Error(map_uart_error(error)),
        }
    }

    fn set_create_uart_baud(&mut self, baud: u32) -> Result<(), ()> {
        self.uart.set_baudrate(baud);
        Ok(())
    }

    fn charging_indicator_active(&mut self) -> Option<bool> {
        if body::CREATE_CHARGING_INDICATOR_ENABLED {
            Some(self.charging_indicator.is_high() == body::CREATE_CHARGING_INDICATOR_ACTIVE_HIGH)
        } else {
            None
        }
    }
}

fn map_uart_error(error: UartError) -> UartReadError {
    match error {
        UartError::Overrun => UartReadError::Overrun,
        UartError::Break => UartReadError::Break,
        UartError::Parity => UartReadError::Parity,
        UartError::Framing => UartReadError::Framing,
        _ => UartReadError::Other,
    }
}

pub fn spawn_safety_lane(p: embassy_rp::Peripherals) -> ! {
    let mut flash = embassy_rp::flash::Flash::<_, _, { 2 * 1024 * 1024 }>::new_blocking(p.FLASH);
    let mut unique_id = [0u8; 8];
    let _ = flash.blocking_unique_id(&mut unique_id);
    let instance = stable_board_id(&unique_id).max(1);
    BRAINSTEM_INSTANCE_ID.store(instance, Ordering::Release);
    BRAINSTEM_BOOT_ID.store(
        instance.rotate_left(11) ^ boot_entropy().max(1),
        Ordering::Release,
    );
    let hardware = PicoWBrainstem::new(
        p.UART0,
        p.PIN_0,
        p.PIN_1,
        p.PIN_18,
        p.PIN_19,
        p.PIN_20,
        p.PIN_17,
        #[cfg(motherbrain_reset_hardware)]
        p.PIN_21,
    );

    spawn_core1(
        p.CORE1,
        unsafe { &mut *core::ptr::addr_of_mut!(CORE1_STACK) },
        move || Runtime::new(hardware).run(),
    );

    spawn_wifi_lane(
        p.PIO0, p.DMA_CH0, p.PIN_23, p.PIN_24, p.PIN_25, p.PIN_29, p.UART1, p.PIN_4, p.PIN_5,
        p.I2C1, p.PIN_2, p.PIN_3, p.USB,
    );
}

#[allow(clippy::too_many_arguments)]
fn spawn_wifi_lane(
    pio0: Peri<'static, PIO0>,
    dma0: Peri<'static, DMA_CH0>,
    wifi_power: Peri<'static, PIN_23>,
    wifi_dio: Peri<'static, PIN_24>,
    wifi_cs: Peri<'static, PIN_25>,
    wifi_clk: Peri<'static, PIN_29>,
    forebrain_uart: Peri<'static, UART1>,
    forebrain_tx: Peri<'static, PIN_4>,
    forebrain_rx: Peri<'static, PIN_5>,
    i2c1: Peri<'static, I2C1>,
    i2c_sda: Peri<'static, PIN_2>,
    i2c_scl: Peri<'static, PIN_3>,
    usb: Peri<'static, USB>,
) -> ! {
    static EXECUTOR: StaticCell<embassy_executor::Executor> = StaticCell::new();
    let executor = EXECUTOR.init(embassy_executor::Executor::new());
    executor.run(|spawner| {
        spawner.spawn(
            wifi_task(spawner, pio0, dma0, wifi_power, wifi_dio, wifi_cs, wifi_clk)
                .expect("spawn wifi task"),
        );
        spawner.spawn(
            forebrain_uart_task(forebrain_uart, forebrain_tx, forebrain_rx)
                .expect("spawn forebrain uart task"),
        );
        spawner.spawn(imu_task(i2c1, i2c_sda, i2c_scl).expect("spawn imu task"));
        spawn_usb_cdc_tasks(&spawner, usb);
    })
}

#[embassy_executor::task]
#[allow(clippy::too_many_arguments)]
async fn wifi_task(
    spawner: Spawner,
    pio0: Peri<'static, PIO0>,
    dma0: Peri<'static, DMA_CH0>,
    wifi_power: Peri<'static, PIN_23>,
    wifi_dio: Peri<'static, PIN_24>,
    wifi_cs: Peri<'static, PIN_25>,
    wifi_clk: Peri<'static, PIN_29>,
) {
    status::mark_wifi_starting();
    if let Some((stack, mut control)) =
        start_wifi_ap(spawner, pio0, dma0, wifi_power, wifi_dio, wifi_cs, wifi_clk).await
    {
        status::mark_wifi_ap_started();
        let _ = control.gpio_set(0, false).await;
        for _ in 0..HTTP_TASKS {
            spawner.spawn(http_task(stack).expect("spawn http task"));
        }
        spawner.spawn(websocket_task(stack).expect("spawn websocket task"));
        spawner.spawn(udp_control_task(stack).expect("spawn udp control task"));
        spawner.spawn(dns_task(stack).expect("spawn dns task"));
        spawner.spawn(mdns_task(stack).expect("spawn mdns task"));
        spawner.spawn(dhcp_task(stack).expect("spawn dhcp task"));
        status::mark_wifi_services_started();
        onboard_led_loop(&mut control).await;
    }

    status::mark_wifi_error();
    loop {
        Timer::after_secs(LED_HEARTBEAT_INTERVAL_SECS).await;
    }
}

async fn start_wifi_ap(
    spawner: Spawner,
    pio0: Peri<'static, PIO0>,
    dma0: Peri<'static, DMA_CH0>,
    wifi_power: Peri<'static, PIN_23>,
    wifi_dio: Peri<'static, PIN_24>,
    wifi_cs: Peri<'static, PIN_25>,
    wifi_clk: Peri<'static, PIN_29>,
) -> Option<(Stack<'static>, cyw43::Control<'static>)> {
    let fw = aligned_bytes!("../../firmware/cyw43/43439A0.bin");
    let clm = aligned_bytes!("../../firmware/cyw43/43439A0_clm.bin");
    let nvram = aligned_bytes!("../../firmware/cyw43/nvram_rp2040.bin");

    let pwr = Output::new(wifi_power, Level::Low);
    let cs = Output::new(wifi_cs, Level::High);
    let mut pio = Pio::new(pio0, Irqs);
    let spi = PioSpi::new(
        &mut pio.common,
        pio.sm0,
        DEFAULT_CLOCK_DIVIDER,
        pio.irq0,
        cs,
        wifi_dio,
        wifi_clk,
        dma::Channel::new(dma0, Irqs),
    );

    static STATE: StaticCell<cyw43::State> = StaticCell::new();
    let state = STATE.init(cyw43::State::new());
    let (net_device, mut control, runner) = cyw43::new(state, pwr, spi, fw, nvram).await;
    spawner.spawn(cyw43_runner_task(runner).expect("spawn cyw43 runner"));

    control.init(clm).await;
    control
        .set_power_management(cyw43::PowerManagementMode::None)
        .await;

    let config = NetConfig::ipv4_static(embassy_net::StaticConfigV4 {
        address: Ipv4Cidr::new(AP_IP, 24),
        dns_servers: Default::default(),
        gateway: None,
    });

    static RESOURCES: StaticCell<StackResources<10>> = StaticCell::new();
    let (stack, runner) = embassy_net::new(
        net_device,
        config,
        RESOURCES.init(StackResources::new()),
        0x5eed,
    );
    let _ = stack.join_multicast_group(IpAddress::Ipv4(Ipv4Address::new(224, 0, 0, 251)));
    spawner.spawn(net_runner_task(runner).expect("spawn net runner"));

    let ssid = ap_ssid(stack.hardware_address());
    control.start_ap_open(&ssid, AP_CHANNEL).await;
    Some((stack, control))
}

fn ap_ssid(address: HardwareAddress) -> heapless::String<9> {
    let mut ssid = heapless::String::<9>::new();
    let _ = ssid.push_str(AP_SSID_PREFIX);
    let mut value = stable_instance_id(address);
    let mut digits = [b'0'; 4];
    for digit in digits.iter_mut().rev() {
        let remainder = (value % INSTANCE_ID_BASE) as u8;
        *digit = if remainder < 10 {
            b'0' + remainder
        } else {
            b'a' + (remainder - 10)
        };
        value /= INSTANCE_ID_BASE;
    }
    for digit in digits {
        let _ = ssid.push(digit as char);
    }
    ssid
}

fn stable_instance_id(address: HardwareAddress) -> u32 {
    let mut hash = 0x811c_9dc5u32;
    let HardwareAddress::Ethernet(address) = address;
    for byte in address.as_bytes() {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash % INSTANCE_ID_MODULUS
}

fn stable_board_id(unique_id: &[u8]) -> u32 {
    let mut hash = 0x811c_9dc5u32;
    for byte in unique_id {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

fn boot_entropy() -> u32 {
    let mut value = 0u32;
    for _ in 0..32 {
        for _ in 0..37 {
            cortex_m::asm::nop();
        }
        value = (value << 1) | rp_pac::ROSC.randombit().read().randombit() as u32;
    }
    value ^ Instant::now().as_micros() as u32
}

#[embassy_executor::task]
async fn cyw43_runner_task(
    runner: cyw43::Runner<'static, cyw43::SpiBus<Output<'static>, PioSpi<'static, PIO0, 0>>>,
) -> ! {
    runner.run().await
}

#[embassy_executor::task]
async fn net_runner_task(mut runner: embassy_net::Runner<'static, cyw43::NetDriver<'static>>) -> ! {
    runner.run().await
}

#[embassy_executor::task]
async fn imu_task(
    i2c1: Peri<'static, I2C1>,
    sda: Peri<'static, PIN_2>,
    scl: Peri<'static, PIN_3>,
) -> ! {
    if !body::IMU_ENABLED {
        status::mark_imu_health(ImuHealth::Absent);
        core::future::pending().await
    }

    let mut config = I2cConfig::default();
    config.frequency = IMU_I2C_FREQUENCY_HZ;
    let mut i2c = I2c::new_async(i2c1, scl, sda, Irqs, config);
    let mut address = None;

    loop {
        let active_address = match address {
            Some(address) => address,
            None => match initialize_imu(&mut i2c).await {
                Ok(found_address) => {
                    address = Some(found_address);
                    found_address
                }
                Err(health) => {
                    status::mark_imu_health(health);
                    Timer::after_millis(IMU_RETRY_MS).await;
                    continue;
                }
            },
        };

        let mut bytes = [0u8; 14];
        if imu_write_read_with_timeout(&mut i2c, active_address, MPU6050_ACCEL_XOUT_H, &mut bytes)
            .await
        {
            status::mark_imu_sample(decode_mpu6050_sample(
                Instant::now().as_millis() as u32,
                &bytes,
            ));
        } else {
            status::mark_imu_health(ImuHealth::Fault);
            address = None;
        }

        Timer::after_millis(body::IMU_POLL_PERIOD_MS.max(1) as u64).await;
    }
}

async fn initialize_imu(i2c: &mut I2c<'static, I2C1, I2cAsync>) -> Result<u8, ImuHealth> {
    for address in [MPU6050_ADDRESS_LOW, MPU6050_ADDRESS_HIGH] {
        let mut who_am_i = [0u8; 1];
        if !imu_write_read_with_timeout(i2c, address, MPU6050_WHO_AM_I, &mut who_am_i).await
            || (who_am_i[0] != 0x68 && who_am_i[0] != 0x70)
        {
            continue;
        }

        for command in [
            [MPU6050_PWR_MGMT_1, 0x00],
            [MPU6050_GYRO_CONFIG, 0x00],
            [MPU6050_ACCEL_CONFIG, 0x00],
        ] {
            if !imu_write_with_timeout(i2c, address, command).await {
                return Err(ImuHealth::Fault);
            }
        }
        return Ok(address);
    }
    Err(ImuHealth::Absent)
}

async fn imu_write_with_timeout(
    i2c: &mut I2c<'static, I2C1, I2cAsync>,
    address: u8,
    bytes: [u8; 2],
) -> bool {
    match select(
        i2c.write_async(address, bytes),
        Timer::after_millis(IMU_I2C_TIMEOUT_MS),
    )
    .await
    {
        Either::First(Ok(())) => true,
        Either::First(Err(_)) | Either::Second(()) => false,
    }
}

async fn imu_write_read_with_timeout(
    i2c: &mut I2c<'static, I2C1, I2cAsync>,
    address: u8,
    register: u8,
    bytes: &mut [u8],
) -> bool {
    match select(
        i2c.write_read_async(address, [register], bytes),
        Timer::after_millis(IMU_I2C_TIMEOUT_MS),
    )
    .await
    {
        Either::First(Ok(())) => true,
        Either::First(Err(_)) | Either::Second(()) => false,
    }
}

#[embassy_executor::task]
async fn forebrain_uart_task(
    uart1: Peri<'static, UART1>,
    tx: Peri<'static, PIN_4>,
    rx: Peri<'static, PIN_5>,
) -> ! {
    let mut uart_config = UartConfig::default();
    uart_config.baudrate = FOREBRAIN_UART_BAUD;
    uart_config.data_bits = DataBits::DataBits8;
    uart_config.stop_bits = StopBits::STOP1;
    uart_config.parity = Parity::ParityNone;

    let mut uart = Uart::new_blocking(uart1, tx, rx, uart_config);
    let mut line = heapless::Vec::<u8, FOREBRAIN_LINE_MAX>::new();
    let mut line_started_ms = 0;

    loop {
        let now_ms = Instant::now().as_millis() as u32;
        match uart.read() {
            Ok(byte) => {
                status::mark_forebrain_uart_rx_byte(now_ms);
                if line.is_empty() {
                    line_started_ms = now_ms;
                }

                match byte {
                    b'\r' => {}
                    b'\n' => {
                        handle_forebrain_uart_line(&mut uart, &line);
                        line.clear();
                        line_started_ms = 0;
                    }
                    byte => {
                        if line.push(byte).is_err() {
                            line.clear();
                            line_started_ms = 0;
                            status::mark_forebrain_uart_error(
                                status::ForebrainUartErrorCode::LineTooLong,
                            );
                            submit_forebrain_stop();
                            write_forebrain_uart_line(&mut uart, b"ERR 0 line_too_long\n");
                        }
                    }
                }
            }
            Err(nb::Error::WouldBlock) => {
                if !line.is_empty()
                    && now_ms.wrapping_sub(line_started_ms) > FOREBRAIN_LINE_TIMEOUT_MS
                {
                    line.clear();
                    line_started_ms = 0;
                    status::mark_forebrain_uart_error(status::ForebrainUartErrorCode::Parse);
                    submit_forebrain_stop();
                    write_forebrain_uart_line(&mut uart, b"ERR 0 timeout\n");
                }
                Timer::after_millis(FOREBRAIN_POLL_MS).await;
            }
            Err(nb::Error::Other(_)) => {
                line.clear();
                line_started_ms = 0;
                status::mark_forebrain_uart_error(status::ForebrainUartErrorCode::Uart);
                submit_forebrain_stop();
                write_forebrain_uart_line(&mut uart, b"ERR 0 uart\n");
                Timer::after_millis(FOREBRAIN_POLL_MS).await;
            }
        }
    }
}

fn spawn_usb_cdc_tasks(spawner: &Spawner, usb: Peri<'static, USB>) {
    let driver = embassy_rp::usb::Driver::new(usb, Irqs);
    let mut config = embassy_usb::Config::new(0x1209, 0x5054);
    config.manufacturer = Some("Pete Robotics");
    config.product = Some("Pete Brainstem Cockpit");
    let mut serial = heapless::String::<24>::new();
    let _ = write!(
        serial,
        "{:08x}",
        BRAINSTEM_INSTANCE_ID.load(Ordering::Acquire)
    );
    static SERIAL_NUMBER: StaticCell<heapless::String<24>> = StaticCell::new();
    config.serial_number = Some(SERIAL_NUMBER.init(serial).as_str());
    config.max_power = 100;
    config.max_packet_size_0 = 64;

    static CONFIG_DESCRIPTOR: StaticCell<[u8; 256]> = StaticCell::new();
    static BOS_DESCRIPTOR: StaticCell<[u8; 256]> = StaticCell::new();
    static CONTROL_BUF: StaticCell<[u8; 64]> = StaticCell::new();
    static CDC_STATE: StaticCell<CdcAcmState<'static>> = StaticCell::new();
    let mut builder = embassy_usb::Builder::new(
        driver,
        config,
        CONFIG_DESCRIPTOR.init([0; 256]),
        BOS_DESCRIPTOR.init([0; 256]),
        &mut [],
        CONTROL_BUF.init([0; 64]),
    );
    let class = CdcAcmClass::new(&mut builder, CDC_STATE.init(CdcAcmState::new()), 64);
    let device = builder.build();
    spawner.spawn(usb_device_task(device).expect("spawn USB device task"));
    spawner.spawn(usb_cdc_task(class).expect("spawn USB CDC task"));
}

#[embassy_executor::task]
async fn usb_device_task(mut device: BrainstemUsbDevice) -> ! {
    device.run().await
}

#[embassy_executor::task]
async fn usb_cdc_task(mut class: CdcAcmClass<'static, UsbDriver>) -> ! {
    let mut packet = [0u8; 64];
    let mut line = heapless::Vec::<u8, FOREBRAIN_LINE_MAX>::new();
    let mut response = heapless::String::<4096>::new();
    loop {
        class.wait_connection().await;
        line.clear();
        loop {
            let len = match class.read_packet(&mut packet).await {
                Ok(len) => len,
                Err(_) => break,
            };
            for byte in &packet[..len] {
                match *byte {
                    b'\r' => {}
                    b'\n' => {
                        if let Ok(command) = core::str::from_utf8(&line) {
                            let boot_to_usb = handle_compact_control_line(
                                command,
                                &mut response,
                                TransportKind::UsbCdc as u8,
                            )
                            .unwrap_or(false);
                            for chunk in response.as_bytes().chunks(64) {
                                if class.write_packet(chunk).await.is_err() {
                                    break;
                                }
                            }
                            if boot_to_usb {
                                Timer::after_millis(100).await;
                                reset_to_usb_boot(0, 0);
                            }
                        }
                        line.clear();
                    }
                    byte => {
                        if line.push(byte).is_err() {
                            line.clear();
                            let _ = class.write_packet(b"ERR 0 line_too_long\n").await;
                        }
                    }
                }
            }
        }
    }
}

#[embassy_executor::task(pool_size = 3)]
async fn http_task(stack: Stack<'static>) -> ! {
    let mut rx_buffer = [0; 1024];
    let mut tx_buffer = [0; 2048];
    let mut request = [0; 1024];
    let mut json = [0; 4096];

    loop {
        let mut socket = TcpSocket::new(stack, &mut rx_buffer, &mut tx_buffer);
        socket.set_timeout(Some(Duration::from_secs(4)));

        if socket.accept(HTTP_PORT).await.is_err() {
            continue;
        }

        let n = match read_http_request(&mut socket, &mut request).await {
            Ok(n) => n,
            Err(_) => {
                socket.abort();
                continue;
            }
        };

        let uptime_ms = Instant::now().as_millis() as u32;
        status::mark_http_request(uptime_ms);
        let method = request_method(&request[..n]);
        let path = request_path(&request[..n]);
        let bootsel_accepted = cfg!(feature = "service-mode")
            && method == Some("POST")
            && path == Some("/command")
            && request_body(&request[..n]).is_some_and(|body| {
                json_str(body, "kind") == Some("bootsel") && json_service_authority_valid(body)
            });
        let result = match (method, path) {
            (Some("GET"), Some("/") | Some("/index.html")) => {
                write_response(&mut socket, "text/html; charset=utf-8", index_html()).await
            }
            (Some("GET"), Some(path)) if path == "/events" || path.starts_with("/events?") => {
                let since_seq = request_sse_cursor(&request[..n]);
                stream_sse(&mut socket, &mut json, since_seq).await
            }
            (Some("GET"), Some("/status.json")) => {
                let snapshot = status::snapshot(uptime_ms);
                match status::render_json(snapshot, &mut json) {
                    Ok(body) => {
                        write_response(&mut socket, "application/json", body.as_bytes()).await
                    }
                    Err(_) => write_plain_status(&mut socket, 500, "Internal Server Error").await,
                }
            }
            (Some("GET"), Some("/network.json")) => {
                let now = Instant::now().as_millis() as u32;
                match render_network_diagnostics(&mut json, now) {
                    Some(body) => {
                        write_response(&mut socket, "application/json", body.as_bytes()).await
                    }
                    None => write_plain_status(&mut socket, 500, "Internal Server Error").await,
                }
            }
            (Some("GET"), Some("/sessions.json")) => {
                let now = Instant::now().as_millis() as u32;
                match render_session_diagnostics(&mut json, now) {
                    Some(body) => {
                        write_response(&mut socket, "application/json", body.as_bytes()).await
                    }
                    None => write_plain_status(&mut socket, 500, "Internal Server Error").await,
                }
            }
            (Some("POST"), Some("/command")) => {
                match handle_command_request(&request[..n], &mut json) {
                    Ok(body) => {
                        write_response(&mut socket, "application/json", body.as_bytes()).await
                    }
                    Err(CommandParseError::Busy(command_id, reason)) => {
                        let body =
                            render_command_response(json.as_mut(), false, command_id, reason);
                        match body {
                            Some(body) => {
                                write_response(&mut socket, "application/json", body.as_bytes())
                                    .await
                            }
                            None => {
                                write_plain_status(&mut socket, 500, "Internal Server Error").await
                            }
                        }
                    }
                    Err(CommandParseError::BadRequest) => {
                        write_plain_status(&mut socket, 400, "Bad Request").await
                    }
                }
            }
            (Some("POST"), Some("/handshake")) => {
                let body = request_body(&request[..n]);
                let malformed = body.is_none_or(|body| session::parse_json(body).is_err());
                if malformed {
                    let rejection = render_handshake_reject(
                        &mut json,
                        "",
                        session::RejectReason::InvalidIdentity,
                    )
                    .unwrap_or("{\"kind\":\"reject\",\"reason_code\":\"internal_error\"}");
                    write_response_status(
                        &mut socket,
                        400,
                        "Bad Request",
                        "application/json",
                        rejection.as_bytes(),
                    )
                    .await
                } else {
                    match handle_handshake_json(
                        body.unwrap_or(""),
                        &mut json,
                        TransportKind::Http as u8,
                    ) {
                        Some(body) if body.contains("\"kind\":\"reject\"") => {
                            write_response_status(
                                &mut socket,
                                409,
                                "Conflict",
                                "application/json",
                                body.as_bytes(),
                            )
                            .await
                        }
                        Some(body) => {
                            write_response(&mut socket, "application/json", body.as_bytes()).await
                        }
                        None => {
                            let rejection = render_handshake_reject(
                                &mut json,
                                "",
                                session::RejectReason::InternalError,
                            )
                            .unwrap_or("{\"kind\":\"reject\",\"reason_code\":\"internal_error\"}");
                            write_response_status(
                                &mut socket,
                                500,
                                "Internal Server Error",
                                "application/json",
                                rejection.as_bytes(),
                            )
                            .await
                        }
                    }
                }
            }
            _ => write_plain_status(&mut socket, 404, "Not Found").await,
        };

        match result {
            Ok(true) => {
                status::mark_http_response_flushed();
                socket.close();
                if bootsel_accepted {
                    Timer::after_millis(150).await;
                    reset_to_usb_boot(0, 0);
                }
            }
            Ok(false) => {
                status::mark_http_response_error();
                socket.abort();
            }
            Err(_) => {
                status::mark_http_response_error();
                socket.abort();
            }
        }
    }
}

#[embassy_executor::task]
async fn websocket_task(stack: Stack<'static>) -> ! {
    let mut rx_buffer = [0; 1024];
    let mut tx_buffer = [0; 2048];
    let mut request = [0; 512];
    let mut payload = [0; 1024];
    let mut response = [0; 4096];

    loop {
        let mut socket = TcpSocket::new(stack, &mut rx_buffer, &mut tx_buffer);
        socket.set_timeout(Some(Duration::from_secs(30)));

        if socket.accept(WS_CONTROL_PORT).await.is_err() {
            continue;
        }

        let n = match socket.read(&mut request).await {
            Ok(n) => n,
            Err(_) => {
                socket.abort();
                continue;
            }
        };

        let path = request_path(&request[..n]);
        let Some(key) = websocket_key(&request[..n]) else {
            let _ = write_plain_status(&mut socket, 400, "Bad Request").await;
            socket.abort();
            continue;
        };

        if path != Some("/control") {
            let _ = write_plain_status(&mut socket, 404, "Not Found").await;
            socket.abort();
            continue;
        }

        let Some(accept_key) = websocket_accept_key(key, &mut response) else {
            socket.abort();
            continue;
        };

        if write_websocket_upgrade(&mut socket, accept_key)
            .await
            .is_err()
        {
            socket.abort();
            continue;
        }

        loop {
            match read_websocket_text(&mut socket, &mut payload).await {
                Ok(Some(body)) => {
                    if let Some(reply) = handle_websocket_message(body, &mut response) {
                        if write_websocket_text(&mut socket, reply.as_bytes())
                            .await
                            .is_err()
                        {
                            socket.abort();
                            break;
                        }
                    }
                }
                Ok(None) => {
                    socket.abort();
                    break;
                }
                Err(_) => {
                    socket.abort();
                    break;
                }
            }
        }
    }
}

async fn onboard_led_loop(control: &mut cyw43::Control<'static>) -> ! {
    let mut next_heartbeat_ms = 0;
    loop {
        let now_ms = Instant::now().as_millis() as u64;
        if let Some(blinks) = status::take_led_blinks() {
            blink_onboard_led(control, blinks).await;
            Timer::after_millis(600).await;
            continue;
        }

        if now_ms >= next_heartbeat_ms {
            blink_onboard_led(control, 1).await;
            next_heartbeat_ms = now_ms.saturating_add(LED_HEARTBEAT_INTERVAL_SECS * 1_000);
        }

        Timer::after_millis(100).await;
    }
}

async fn blink_onboard_led(control: &mut cyw43::Control<'static>, blinks: u8) {
    for _ in 0..blinks {
        let _ = control.gpio_set(0, true).await;
        Timer::after_millis(LED_BLINK_ON_MS).await;
        let _ = control.gpio_set(0, false).await;
        Timer::after_millis(LED_BLINK_OFF_MS).await;
    }
}

#[embassy_executor::task]
async fn udp_control_task(stack: Stack<'static>) -> ! {
    let mut rx_meta = [PacketMetadata::EMPTY; 2];
    let mut rx_buffer = [0; 512];
    let mut tx_meta = [PacketMetadata::EMPTY; 2];
    let mut tx_buffer = [0; 2048];
    let mut request = [0; 1024];
    let mut response = heapless::String::<4096>::new();

    loop {
        let mut socket = UdpSocket::new(
            stack,
            &mut rx_meta,
            &mut rx_buffer,
            &mut tx_meta,
            &mut tx_buffer,
        );
        if socket.bind(UDP_CONTROL_PORT).is_err() {
            Timer::after_secs(5).await;
            continue;
        }

        loop {
            let Ok((len, endpoint)) = socket.recv_from(&mut request).await else {
                continue;
            };
            let Ok(line) = core::str::from_utf8(&request[..len]) else {
                continue;
            };
            let Some(boot_to_usb) =
                handle_compact_control_line(line.trim(), &mut response, TransportKind::Udp as u8)
            else {
                continue;
            };
            let _ = socket.send_to(response.as_bytes(), endpoint).await;
            if boot_to_usb {
                Timer::after_millis(100).await;
                reset_to_usb_boot(0, 0);
            }
        }
    }
}

#[embassy_executor::task]
async fn dns_task(stack: Stack<'static>) -> ! {
    let mut rx_meta = [PacketMetadata::EMPTY; 2];
    let mut rx_buffer = [0; 512];
    let mut tx_meta = [PacketMetadata::EMPTY; 2];
    let mut tx_buffer = [0; 512];
    let mut request = [0; 512];
    let mut response = [0; 512];

    loop {
        let mut socket = UdpSocket::new(
            stack,
            &mut rx_meta,
            &mut rx_buffer,
            &mut tx_meta,
            &mut tx_buffer,
        );
        if socket.bind(DNS_PORT).is_err() {
            Timer::after_secs(5).await;
            continue;
        }

        loop {
            let Ok((len, endpoint)) = socket.recv_from(&mut request).await else {
                continue;
            };
            let Some(reply) = build_dns_reply(&request[..len], &mut response) else {
                continue;
            };
            let _ = socket.send_to(reply, endpoint).await;
        }
    }
}

#[embassy_executor::task]
async fn mdns_task(stack: Stack<'static>) -> ! {
    let mut rx_meta = [PacketMetadata::EMPTY; 2];
    let mut rx_buffer = [0; 256];
    let mut tx_meta = [PacketMetadata::EMPTY; 2];
    let mut tx_buffer = [0; 768];
    let mut packet = [0; 768];
    let endpoint = IpEndpoint::new(IpAddress::Ipv4(Ipv4Address::new(224, 0, 0, 251)), MDNS_PORT);

    loop {
        let mut socket = UdpSocket::new(
            stack,
            &mut rx_meta,
            &mut rx_buffer,
            &mut tx_meta,
            &mut tx_buffer,
        );
        if socket.bind(MDNS_PORT).is_ok() {
            loop {
                let len = build_mdns_announcement(&mut packet);
                let _ = socket.send_to(&packet[..len], endpoint).await;
                Timer::after_secs(5).await;
            }
        }
        Timer::after_secs(5).await;
    }
}

#[embassy_executor::task]
async fn dhcp_task(stack: Stack<'static>) -> ! {
    let mut rx_meta = [PacketMetadata::EMPTY; 2];
    let mut rx_buffer = [0; 1024];
    let mut tx_meta = [PacketMetadata::EMPTY; 2];
    let mut tx_buffer = [0; 1024];
    let mut request = [0; 576];
    let mut response = [0; 576];
    let mut leases = DhcpLeaseState::new();
    let endpoint = IpEndpoint::new(
        IpAddress::Ipv4(Ipv4Address::new(255, 255, 255, 255)),
        DHCP_CLIENT_PORT,
    );

    loop {
        let mut socket = UdpSocket::new(
            stack,
            &mut rx_meta,
            &mut rx_buffer,
            &mut tx_meta,
            &mut tx_buffer,
        );

        if socket.bind(DHCP_SERVER_PORT).is_err() {
            Timer::after_secs(5).await;
            continue;
        }

        loop {
            let Ok((len, _meta)) = socket.recv_from(&mut request).await else {
                continue;
            };

            let Some(dhcp_request) = DhcpRequest::parse(&request[..len]) else {
                continue;
            };
            let Some(grant) = leases.grant(dhcp_request, Instant::now().as_millis() as u64) else {
                continue;
            };
            let client = dhcp_request.client();
            network_registry::record_lease(
                client.lease_identity(),
                grant.lease_ip(),
                (Instant::now().as_millis() as u32).wrapping_add(DHCP_LEASE_SECONDS * 1_000),
            );
            let Some(reply) = build_dhcp_reply(grant, &request[..len], &mut response) else {
                continue;
            };
            status::mark_dhcp_grant();
            let _ = socket.send_to(reply, endpoint).await;
        }
    }
}

async fn write_response(
    socket: &mut TcpSocket<'_>,
    content_type: &str,
    body: &[u8],
) -> Result<bool, embassy_net::tcp::Error> {
    let mut header = heapless::String::<192>::new();
    let _ = write!(
        header,
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        content_type,
        body.len()
    );
    socket.write_all(header.as_bytes()).await?;
    socket.write_all(body).await?;
    flush_tcp_with_timeout(socket).await
}

async fn stream_sse(
    socket: &mut TcpSocket<'_>,
    json: &mut [u8],
    mut since_seq: u32,
) -> Result<bool, embassy_net::tcp::Error> {
    socket
        .write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\nretry: 1000\r\n\r\n",
        )
        .await?;
    if !flush_tcp_with_timeout(socket).await? {
        return Ok(false);
    }

    let mut next_status_ms = 0;
    loop {
        let now_ms = Instant::now().as_millis() as u64;
        if now_ms >= next_status_ms {
            let snapshot = status::snapshot(now_ms as u32);
            let body = match status::render_json(snapshot, json) {
                Ok(body) => body,
                Err(_) => return Ok(false),
            };
            match write_sse_event(socket, "status", None, body).await {
                Ok(true) => {}
                Ok(false) | Err(_) => return Ok(true),
            }
            next_status_ms = now_ms.saturating_add(SSE_STATUS_INTERVAL_MS);
        }

        let event_next_seq = status::event_next_seq();
        if event_next_seq != since_seq.saturating_add(1) {
            let body = match status::render_events_json(since_seq, json) {
                Some(body) => body,
                None => return Ok(false),
            };
            let last_seq = json_u32(body, "next_seq")
                .unwrap_or(event_next_seq)
                .saturating_sub(1);
            match write_sse_event(socket, "events", Some(last_seq), body).await {
                Ok(true) => since_seq = last_seq,
                Ok(false) | Err(_) => return Ok(true),
            }
        }

        Timer::after_millis(SSE_EVENT_CHECK_INTERVAL_MS).await;
    }
}

async fn write_sse_event(
    socket: &mut TcpSocket<'_>,
    event: &str,
    id: Option<u32>,
    body: &str,
) -> Result<bool, embassy_net::tcp::Error> {
    let mut prefix = heapless::String::<64>::new();
    let _ = write!(prefix, "event: {event}\r\n");
    if let Some(id) = id {
        let _ = write!(prefix, "id: {id}\r\n");
    }
    let _ = prefix.push_str("data: ");
    socket.write_all(prefix.as_bytes()).await?;
    socket.write_all(body.trim_end().as_bytes()).await?;
    socket.write_all(b"\r\n\r\n").await?;
    flush_tcp_with_timeout(socket).await
}

async fn read_http_request(
    socket: &mut TcpSocket<'_>,
    buffer: &mut [u8],
) -> Result<usize, embassy_net::tcp::Error> {
    let mut used = 0;
    loop {
        if used == buffer.len() {
            return Ok(used);
        }
        let read = socket.read(&mut buffer[used..]).await?;
        if read == 0 {
            return Ok(used);
        }
        used += read;
        let Some(header_end) = buffer[..used]
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|index| index + 4)
        else {
            continue;
        };
        let header = core::str::from_utf8(&buffer[..header_end]).unwrap_or("");
        let content_length = header
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("Content-Length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
            .unwrap_or(0);
        if used >= header_end.saturating_add(content_length) {
            return Ok(used);
        }
    }
}

async fn write_response_status(
    socket: &mut TcpSocket<'_>,
    code: u16,
    text: &str,
    content_type: &str,
    body: &[u8],
) -> Result<bool, embassy_net::tcp::Error> {
    let mut header = heapless::String::<192>::new();
    let _ = write!(
        header,
        "HTTP/1.1 {code} {text}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    socket.write_all(header.as_bytes()).await?;
    socket.write_all(body).await?;
    flush_tcp_with_timeout(socket).await
}

async fn write_plain_status(
    socket: &mut TcpSocket<'_>,
    code: u16,
    text: &str,
) -> Result<bool, embassy_net::tcp::Error> {
    let mut header = heapless::String::<160>::new();
    let _ = write!(
        header,
        "HTTP/1.1 {} {}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        code,
        text,
        text.len(),
        text
    );
    socket.write_all(header.as_bytes()).await?;
    flush_tcp_with_timeout(socket).await
}

async fn flush_tcp_with_timeout(
    socket: &mut TcpSocket<'_>,
) -> Result<bool, embassy_net::tcp::Error> {
    match select(socket.flush(), Timer::after_millis(HTTP_FLUSH_TIMEOUT_MS)).await {
        Either::First(result) => result.map(|()| true),
        Either::Second(()) => Ok(false),
    }
}

fn request_path(request: &[u8]) -> Option<&str> {
    let line_end = request
        .windows(2)
        .position(|w| w == b"\r\n")
        .unwrap_or(request.len());
    let line = core::str::from_utf8(&request[..line_end]).ok()?;
    let mut parts = line.split(' ');
    let _method = parts.next()?;
    parts.next()
}

fn request_method(request: &[u8]) -> Option<&str> {
    let line_end = request
        .windows(2)
        .position(|w| w == b"\r\n")
        .unwrap_or(request.len());
    let line = core::str::from_utf8(&request[..line_end]).ok()?;
    line.split(' ').next()
}

fn request_sse_cursor(request: &[u8]) -> u32 {
    let requested = request_header(request, "Last-Event-ID")
        .and_then(|value| value.parse().ok())
        .or_else(|| {
            let query = request_path(request)?.split_once('?')?.1;
            query.split('&').find_map(|field| {
                let (name, value) = field.split_once('=')?;
                (name == "since").then(|| value.parse().ok()).flatten()
            })
        });
    let next_seq = status::event_next_seq();
    match requested {
        Some(cursor) if cursor < next_seq => cursor,
        Some(_) => 0,
        None => next_seq.saturating_sub(1),
    }
}

fn request_header<'a>(request: &'a [u8], wanted: &str) -> Option<&'a str> {
    let request = core::str::from_utf8(request).ok()?;
    request.lines().skip(1).find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case(wanted).then(|| value.trim())
    })
}

fn index_html() -> &'static [u8] {
    // Embedded browser cockpit mapping to the host-side pete-cockpit contract:
    //
    // UI action                    JSON kind             CockpitRequest                 Capability
    // joystick / drive pad          cmd_vel               CmdVel                         cmd_vel
    // active motion heartbeat       heartbeat_stop        HeartbeatStop                  heartbeat_stop
    // STOP                          stop                  Stop                           stop
    // E-STOP                        estop                 EStop                          estop
    // Clear E-Stop                  clear_estop           ClearEStop                     clear_estop
    // Dock                          dock                  Dock                           dock
    // Ping                          ping                  Ping                           ping
    // Drive 300                     drive_for             DriveFor                       drive_for
    // Turn L/R                      turn_by               TurnBy                         turn_by
    // Creep                         creep_until           CreepUntil                     creep_until
    // Scan                          scan_arc              ScanArc                        scan_arc
    // Wiggle                        wiggle_align          WiggleAlign                    wiggle_align
    // Bump Escape                   bump_escape           BumpEscape                     bump_escape
    // Unstick                       unstick               Unstick                        unstick
    // Cliff Stop                    cliff_guard           CliffGuard                     cliff_guard
    // Music Define / Play           song_define/play      SongDefine / SongPlay          song_define/song_play
    // Refresh                       reconnect /events SSE (no command)
    // BOOTSEL                       bootsel               Bootsel                        service/debug only
    br#"<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pete Brainstem</title>
<style>
:root{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#1f2728;background:#eef1ed;accent-color:#1d7580}
*{box-sizing:border-box}body{margin:0}.wrap{max-width:1280px;margin:auto;padding:14px}
header{display:grid;grid-template-columns:minmax(210px,1fr) auto;gap:12px;align-items:start;margin-bottom:12px}
h1{font-size:23px;line-height:1.05;margin:0;color:#121817}h2{font-size:12px;margin:0;color:#596361;text-transform:uppercase;font-weight:850}
.sub{font-size:13px;color:#697370;margin-top:5px}.top{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end;align-items:center}
.pill{font-size:12px;border:1px solid #c5cdc8;border-radius:999px;padding:5px 9px;background:#fff;color:#303936;white-space:nowrap}
.pill.ok{border-color:#5ca77a;background:#edf8f1}.pill.warn{border-color:#d4aa40;background:#fff7dc}.pill.bad{border-color:#cf6868;background:#fff0f0}
.check{display:inline-flex;align-items:center;gap:5px;font-size:12px;font-weight:750;color:#44504a;white-space:nowrap}.check input{width:16px;min-height:16px;margin:0}.lock{font-size:12px;border:1px solid #c5cdc8;border-radius:999px;padding:5px 9px;background:#fff;color:#68736c;font-weight:850;white-space:nowrap}.lock.ok{border-color:#5ca77a;background:#edf8f1;color:#287142}.lock.warn{border-color:#d4aa40;background:#fff7dc;color:#6d5510}
.layout{display:grid;grid-template-columns:minmax(340px,.95fr) minmax(0,1.45fr);gap:10px;align-items:start}
.side{display:grid;gap:10px}.station{background:#fff;border:1px solid #d7ded9;border-radius:8px;padding:11px;box-shadow:0 1px 2px #17241c10;display:grid;gap:10px}
.station-head{display:flex;align-items:center;justify-content:space-between;gap:8px}.station-head .pill{padding:4px 8px}.motion{position:sticky;top:10px}
.zone{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(180px,.85fr);gap:10px;align-items:start}.zone.slim{grid-template-columns:minmax(0,1fr) minmax(150px,.55fr)}
.controls{display:grid;gap:8px}.joy{min-height:326px;display:grid;place-items:center;touch-action:none;user-select:none;background:#f7f9f7;border:1px solid #e1e7e3;border-radius:8px}
.base{width:min(68vw,296px);height:min(68vw,296px);border-radius:50%;background:#e5ebe7;border:2px solid #c3ccc6;position:relative;box-shadow:inset 0 0 0 28px #f0f4f1}
.base:before,.base:after{content:"";position:absolute;background:#c8d1cb}.base:before{width:2px;height:82%;left:50%;top:9%}.base:after{height:2px;width:82%;left:9%;top:50%}
.nub{width:84px;height:84px;border-radius:50%;background:#1d7580;position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);box-shadow:0 8px 18px #13251c33;border:4px solid #fbfdfb}
.row{display:flex;gap:8px;flex-wrap:wrap}.row>*{flex:1 1 auto}.split{display:grid;grid-template-columns:1fr 1fr;gap:8px}
button{min-height:40px;border:1px solid #b9c2bd;border-radius:7px;background:#fff;color:#202722;font-weight:750;font-size:14px;letter-spacing:0;cursor:pointer}
button:active,.active{transform:translateY(1px);background:#eef2ef}button:disabled{opacity:.48;cursor:not-allowed}.primary{background:#dceee6;border-color:#8eb99f}.stop{background:#202522;color:#fff;border-color:#202522}.danger{background:#9d2830;color:#fff;border-color:#842029}.warnbtn{background:#fff3d6;border-color:#d8b24a}.blue{background:#e7f0fb;border-color:#9bbbe0}
.pad{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}.pad button{min-height:48px}.pad .center{grid-column:2}
.seg{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}.seg.three{grid-template-columns:repeat(3,minmax(0,1fr))}.seg button{min-height:38px;font-size:12px}
label{font-size:12px;color:#5b655f;font-weight:750}.slider,.field{display:grid;gap:6px}.slider input{width:100%}input,select{width:100%;min-height:40px;border:1px solid #cbd3ce;border-radius:7px;padding:8px;font:inherit;background:#fff}
.readout{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;font-size:13px}.readout.compact{grid-template-columns:1fr}.tile{background:#f6f8f6;border:1px solid #e1e6e2;border-radius:7px;padding:8px;min-height:50px}
.tile b{display:block;color:#4e5852;font-size:11px;text-transform:uppercase;margin-bottom:3px}.tile span,.tile div{overflow-wrap:anywhere}.wide{grid-column:1/-1}.muted{color:#68736c}.badtext{color:#a1262f}.oktext{color:#287142}
.imu{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.imu .tile{min-height:58px}.bar{height:7px;border-radius:999px;background:#dfe6e2;overflow:hidden;margin-top:6px}.bar i{display:block;height:100%;width:0;background:#1d7580}.bar.warn i{background:#d49832}.bar.bad i{background:#b12c37}
.log{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;line-height:1.45;max-height:132px;overflow:auto;white-space:pre-wrap}
@media(max-width:980px){.layout{grid-template-columns:1fr}.motion{position:static}.zone,.zone.slim{grid-template-columns:1fr}.imu{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:560px){.wrap{padding:10px}header{grid-template-columns:1fr}.top{justify-content:flex-start}.readout,.split,.imu{grid-template-columns:1fr}.seg,.seg.three{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
</head>
<body>
<div class="wrap">
<header><div><h1>Pete Brainstem</h1><div class="sub" id="headline">Waiting for status</div></div><div class="top"><span id="session" class="pill">no session</span><button id="sessionnew">New session</button><button id="controllease">Request control</button><span id="controlstate" class="lock">unlocked</span><label class="check"><input id="controlrefresh" type="checkbox">Keep control</label><span id="net" class="pill">connecting</span><span id="mode" class="pill">mode unknown</span><span id="safety" class="pill">safety unknown</span></div></header>
<div class="layout">
<section class="station motion">
<div class="station-head"><h2>Drive</h2><span id="cmd" class="pill">command unknown</span></div>
<div class="joy"><div id="base" class="base"><div id="nub" class="nub"></div></div></div>
<div class="split"><div class="slider"><label for="speed">Speed <span id="speedv">120</span> mm/s</label><input id="speed" type="range" min="40" max="260" value="120"></div><div class="slider"><label for="turn">Turn <span id="turnv">1200</span> mrad/s</label><input id="turn" type="range" min="300" max="2000" value="1200"></div></div>
<div class="pad">
<button class="primary center" data-drive="fwd">FWD</button>
<button data-drive="left">LEFT</button><button class="stop" id="padstop">STOP</button><button data-drive="right">RIGHT</button>
<button data-drive="back" class="center">BACK</button>
<button data-drive="spinl">SPIN L</button><button data-drive="slow">SLOW</button><button data-drive="spinr">SPIN R</button>
</div>
<div class="row"><button class="stop" id="stop">STOP</button><button data-action="drive_for">Drive 300</button><button data-action="turn_left">Turn L</button><button data-action="turn_right">Turn R</button></div>
<div class="row"><button data-action="creep">Creep</button><button data-action="scan">Scan</button><button data-action="wiggle">Wiggle</button></div>
</section>
<div class="side">
<section class="station">
<div class="station-head"><h2>Safety and Reflexes</h2><span class="pill">reflex guard</span></div>
<div class="zone slim">
<div class="readout">
<div class="tile wide"><b>Safety</b><span id="safetyread" class="muted">...</span></div>
<div class="tile"><b>Last error</b><span id="err" class="muted">...</span></div>
<div class="tile"><b>Events</b><span id="events" class="muted">...</span></div>
</div>
<div class="controls">
<div class="seg three"><button class="danger" id="estop">E-STOP</button><button id="clear">Clear E-Stop</button><button id="clearcharge">Clear Charge</button></div>
<div class="seg three"><button id="clearbump">Clear Bump</button><button id="clearwheel">Clear Wheel</button><button id="clearcliff">Clear Cliff</button></div>
<div class="seg"><button id="cleartilt">Clear Tilt</button><button id="clearimpact">Clear Impact</button></div>
<div class="seg"><button class="warnbtn" data-action="bump_escape">Bump Escape</button><button class="warnbtn" data-action="unstick">Unstick</button><button class="danger" data-action="cliff_trip">Cliff Stop</button><button data-action="cliff_clear">Clear Cliff</button></div>
<button class="blue" data-action="heartbeat">Heartbeat</button>
</div>
</div>
</section>
<section class="station">
<div class="station-head"><h2>Create Body</h2><span id="create" class="pill">create unknown</span></div>
<div class="zone">
<div class="readout">
<div class="tile"><b>Battery</b><span id="battery" class="muted">...</span></div>
<div class="tile"><b>Odometry</b><span id="odom" class="muted">...</span></div>
<div class="tile wide"><b>Sensors</b><span id="sensors" class="muted">...</span></div>
</div>
<div class="controls">
<div class="seg"><button id="dock">Dock</button><button id="ping">Ping</button></div>
<div class="seg"><button id="stream">Stream Sensors</button><button id="createon" class="blue">Create On</button></div>
<div class="seg"><button id="createoi">Start OI</button><button id="createbrc">Pulse BRC</button></div>
<div class="seg"><button id="createoff" class="warnbtn">Create Off</button><button id="createrestart" class="warnbtn">Restart Create</button></div>
</div>
</div>
</section>
<section class="station">
<div class="station-head"><h2>IMU</h2><span class="pill">orientation</span></div>
<div class="zone">
<div class="imu">
<div class="tile"><b>IMU health</b><span id="imuhealth" class="muted">...</span></div>
<div class="tile"><b>Yaw</b><span id="imuyaw" class="muted">...</span></div>
<div class="tile"><b>Accel</b><span id="imuaccel" class="muted">...</span><div id="imuaccelbar" class="bar"><i></i></div></div>
<div class="tile"><b>Tilt</b><span id="imutilt" class="muted">...</span><div id="imutiltbar" class="bar"><i></i></div></div>
<div class="tile"><b>Angular rate</b><span id="imurates" class="muted">...</span></div>
<div class="tile"><b>Roughness</b><span id="imurough" class="muted">...</span><div id="imuroughbar" class="bar"><i></i></div></div>
<div class="tile"><b>Impact</b><span id="imuimpact" class="muted">...</span><div id="imuimpactbar" class="bar"><i></i></div></div>
<div class="tile"><b>Motion</b><span id="imumotion" class="muted">...</span></div>
</div>
<div class="controls">
<button id="imuzero" class="primary">Zero IMU</button>
<button id="imuclear">Clear IMU</button>
</div>
</div>
</section>
<section class="station">
<div class="station-head"><h2>Session and Link</h2><button id="refresh">Refresh</button></div>
<div class="zone slim">
<div class="readout">
<div class="tile"><b>Runtime</b><span id="runtime" class="muted">...</span></div>
<div class="tile"><b>Uptime</b><span id="uptime" class="muted">...</span></div>
<div class="tile"><b>UART</b><span id="uart" class="muted">...</span></div>
<div class="tile"><b>Forebrain</b><span id="forebrain" class="muted">...</span></div>
<div class="tile"><b>Web</b><span id="web" class="muted">...</span></div>
<div class="tile"><b>Firmware</b><span id="firmware" class="muted">...</span></div>
</div>
<div class="controls"><div class="seg"><button id="mbreset" class="danger">Reset Motherbrain</button><button id="bootsel" class="danger">BOOTSEL</button></div></div>
</div>
</section>
<section class="station">
<div class="station-head"><h2>Music</h2><span id="music" class="pill">music unknown</span></div>
<div class="zone slim">
<div class="split"><div class="field"><label for="songid">Slot</label><input id="songid" inputmode="numeric" value="0"></div><div class="field"><label for="tones">Tones</label><input id="tones" value="72:8,76:8,79:16"></div></div>
<div class="controls"><div class="seg three"><button id="songdef" class="primary">Define</button><button id="songplay">Play</button><button id="song">Chirp</button></div></div>
</div>
</section>
<section class="station">
<div class="station-head"><h2>Activity</h2></div>
<div id="log" class="log muted">No commands yet</div>
</section>
</div>
</div>
</div>
<script>
let id=1,active=false,timer=0,controlRefreshTimer=0,last={x:0,y:0},ws=null,wsOpen=false,sse=null,sseOpen=false,driveKind='',lastDriveAt=0,lastHeartbeatAt=0,eventCursor=0,caps=null,lastStatus=null,sessionId='',controlLeaseId='',serviceLeaseId='',sensorStreamRequested=false;
const $=x=>document.getElementById(x),base=$('base'),nub=$('nub'),net=$('net'),log=$('log');
const seqKinds=new Set(['cmd_vel','drive_direct','drive_arc','face_bearing','track_bearing','turn_by','drive_for','bump_escape','hold_heading','turn_to_heading','arc_for','creep_until','scan_arc','dock_align','wall_follow','wiggle_align','unstick','cliff_guard','clear_safety_latch','heartbeat_stop','request_sensors','stream_sensors','set_safety_policy','clear_motion_queue','define_chirp','play_feedback','power_state','create_power_on','create_power_off','calibrate_turn','reset_odometry','zero_imu_orientation','clear_imu_orientation','song_define']);
const actionVerb={drive_for:'drive_for',turn_left:'turn_by',turn_right:'turn_by',creep:'creep_until',scan:'scan_arc',wiggle:'wiggle_align',bump_escape:'bump_escape',unstick:'unstick',cliff_trip:'cliff_guard',cliff_clear:'cliff_guard',heartbeat:'heartbeat_stop'};
function title(s){return (s||'unknown').replaceAll('_',' ')}
function pill(el,text,state){el.textContent=text;el.className='pill '+(state||'')}
function addLog(text){let t=new Date().toLocaleTimeString();log.textContent=(t+'  '+text+'\n'+(log.textContent==='No commands yet'?'':log.textContent)).slice(0,900)}
function hasVerb(v){return !!(caps&&caps.verbs&&caps.verbs.indexOf(v)>=0)}
function setEnabled(id,on){let e=$(id);if(e)e.disabled=!on}
function setEnabledAll(selector,on){document.querySelectorAll(selector).forEach(e=>e.disabled=!on)}
function chargeActive(cs){return cs.charging_indicator==='on'||(cs.charging_state>=1&&cs.charging_state<=3)}
function statusBlocksMotion(){let s=lastStatus||{},cs=s.create_sensors||{},imu=s.imu||{},imuDanger=imu.health==='fault'||(imu.health==='ok'&&((imu.tilt_magnitude_mrad||0)>=650||(imu.impact_score_mm_s2||0)>=18000)),safety=s.estop_latched||s.safety_tripped||s.motion_interlock_latched||cs.wheel_drop||cs.cliff_left||cs.cliff_front_left||cs.cliff_front_right||cs.cliff_right||imuDanger,charging=chargeActive(cs);return !!(safety||charging)}
function canSession(verb){return hasVerb(verb)&&!!sessionId}
function canControl(verb){return hasVerb(verb)&&!!sessionId&&!!controlLeaseId}
function canMotion(verb){return canControl(verb)&&!statusBlocksMotion()}
function canService(verb){return hasVerb(verb)&&!!sessionId&&!!serviceLeaseId}
function ensureSensorStream(){if(sensorStreamRequested||!sessionId||!hasVerb('stream_sensors'))return;sensorStreamRequested=true;sendCockpit({kind:'stream_sensors',enabled:true,packet_id:0,period_ms:250},false).then(j=>{if(j&&j.accepted===false)sensorStreamRequested=false})}
function token(prefix){let a=new Uint32Array(2);crypto.getRandomValues(a);return prefix+'-'+a[0].toString(16)+'-'+a[1].toString(16)}
function controlLock(text,state){let e=$('controlstate');e.textContent=text;e.className='lock '+(state||'')}
function refreshControlLock(){controlLock(controlLeaseId?'locked':'unlocked',controlLeaseId?'ok':($('controlrefresh').checked?'warn':''))}
function establishBrowserSession(){controlLeaseId='';serviceLeaseId='';sensorStreamRequested=false;applyCaps();let boot=sessionStorage.getItem('pete-browser-boot');if(!boot){boot=token('browserboot');sessionStorage.setItem('pete-browser-boot',boot)}let nonce=token('hello'),hello={role:'operator',session_purpose:'control',device_id:token('browser'),boot_id:boot,handshake_nonce:nonce,protocol_major:1,protocol_minor_min:0,protocol_minor_max:0,supported_features:['session_ids','event_cursor','heartbeat','transport_failover'],required_features:['session_ids'],preferred_heartbeat_ms:500};pill($('session'),'handshaking','warn');return fetch('/handshake',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(hello)}).then(r=>r.json()).then(j=>{if(j.kind!=='welcome'||j.echoed_handshake_nonce!==nonce)throw new Error(j.reason_code||'invalid welcome');sessionId=j.session_id;eventCursor=Math.max(0,(j.current_event_next_seq||1)-1);pill($('session'),'session '+sessionId.slice(-8),'ok');pill(net,'session HTTP','ok');addLog('session opened '+sessionId);applyCaps();requestCaps();connectSse();if($('controlrefresh').checked)acquireBrowserControl();return j}).catch(e=>{sessionId='';controlLeaseId='';serviceLeaseId='';sensorStreamRequested=false;applyCaps();pill($('session'),'session failed','bad');addLog('handshake failed '+e.message);throw e})}
function acquireBrowserControl(){if(!sessionId){addLog('open a session first');return Promise.resolve(null)}let body={kind:'acquire_control_lease',command_id:id++,session_id:sessionId,authority:'operator_debug',ttl_ms:60000};controlLock('refreshing','warn');return fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json()).then(j=>{if(j.accepted&&j.type==='control_lease_granted'){controlLeaseId=j.lease_id;serviceLeaseId='';pill($('session'),'operator control','ok');addLog('control lease '+j.lease_id+' (60 seconds)');applyCaps()}else{controlLeaseId='';handleReply(j);applyCaps()}refreshControlLock();return j}).catch(e=>{controlLeaseId='';refreshControlLock();addLog('control request failed '+e.message);return null})}
function syncControlRefresh(){clearInterval(controlRefreshTimer);controlRefreshTimer=0;refreshControlLock();if($('controlrefresh').checked){if(sessionId)acquireBrowserControl();controlRefreshTimer=setInterval(()=>{if(!sessionId)establishBrowserSession().catch(()=>{});else acquireBrowserControl()},45000)}}
function acquireService(scope){if(!sessionId){addLog('open a session first');return Promise.resolve(null)}let body={kind:'acquire_service_lease',command_id:id++,session_id:sessionId,scope,ttl_ms:5000};return fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json()).then(j=>{if(j.accepted&&j.type==='service_lease_granted'){serviceLeaseId=j.lease_id;controlLeaseId='';pill($('session'),'service '+scope,'warn');addLog('service lease '+scope);applyCaps()}else{serviceLeaseId='';handleReply(j);applyCaps()}refreshControlLock();return j}).catch(e=>{serviceLeaseId='';addLog('service request failed '+e.message);applyCaps();return null})}
function serviceCommand(scope,cmd){return acquireService(scope).then(j=>{if(!(j&&j.accepted))return j;return sendCockpit(cmd).then(r=>{serviceLeaseId='';applyCaps();if($('controlrefresh').checked&&scope!=='bootsel'&&scope!=='reset_motherbrain')acquireBrowserControl();return r})})}
function connectWs(){try{ws=new WebSocket('ws://'+location.hostname+':81/control');ws.onopen=()=>{wsOpen=true;pill(net,'control ws','ok');requestCaps()};ws.onclose=()=>{wsOpen=false;pill(net,'reconnecting','warn');setTimeout(connectWs,1000)};ws.onerror=()=>{wsOpen=false;pill(net,'ws error','warn')};ws.onmessage=e=>{try{handleReply(JSON.parse(e.data))}catch(_){}}}catch(_){wsOpen=false}}
function connectSse(){if(sse)sse.close();sseOpen=false;sse=new EventSource('/events?since='+eventCursor);sse.onopen=()=>{sseOpen=true;pill(net,'telemetry sse','ok')};sse.addEventListener('status',e=>{try{showStatus(JSON.parse(e.data))}catch(_){}});sse.addEventListener('events',e=>{try{handleEvents(JSON.parse(e.data))}catch(_){}});sse.onerror=()=>{sseOpen=false;pill(net,'sse reconnecting','warn')}}
function handleReply(j){if(j.type==='status'){showStatus(j);return}if(j.type==='events'){handleEvents(j);return}if(j.verbs){caps=j;applyCaps();pill(net,'capabilities','ok');ensureSensorStream();return}let ok=j.accepted!==false,reason=j.message||j.reason||'';if(!ok&&(reason==='invalid_control_lease'||reason==='control_lease_required')){controlLeaseId='';pill($('session'),'session; control expired','warn');addLog('control authority expired; press Request control')}if(!ok&&(reason==='invalid_service_lease'||reason==='service_authorization_required'||reason==='service_operation_disabled')){serviceLeaseId=''}if(!ok&&(reason==='invalid_session'||reason==='session_required')){sessionId='';controlLeaseId='';serviceLeaseId='';sensorStreamRequested=false;pill($('session'),'session expired','bad');addLog('session expired; press New session')}applyCaps();refreshControlLock();pill(net,ok?'accepted':'rejected',ok?'ok':'warn');if(!ok)addLog('rejected '+(reason||j.command_id||''));else if(j.message)addLog(j.message+' '+(j.command_id||''))}
function sendCockpit(o,ack){let cid=id++;o.command_id=cid;if(sessionId)o.session_id=sessionId;if(controlLeaseId)o.lease_id=controlLeaseId;if(serviceLeaseId)o.service_lease_id=serviceLeaseId;if(seqKinds.has(o.kind)&&o.seq===undefined)o.seq=cid;let body=JSON.stringify(o),name=o.kind==='cmd_vel'?'drive':o.kind;return fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body}).then(r=>r.json()).then(j=>{handleReply(j);return j}).catch(_=>{pill(net,'offline','bad');addLog('offline '+name)})}
function requestStatus(){return sendCockpit({kind:'status'},false)}
function requestCaps(){return sendCockpit({kind:'get_capabilities'},false).then(j=>{if(j&&j.verbs){caps=j;applyCaps();ensureSensorStream()}})}
function applyCaps(){let drive=canMotion('cmd_vel'),svc=!!sessionId,canClearLatch=canControl('clear_safety_latch');setEnabled('controllease',!!sessionId);setEnabled('stop',hasVerb('stop'));setEnabled('padstop',hasVerb('stop'));setEnabled('estop',hasVerb('estop'));setEnabled('clear',canSession('clear_estop'));setEnabled('clearcharge',canClearLatch&&!(lastStatus&&lastStatus.create_sensors&&chargeActive(lastStatus.create_sensors)));['clearbump','clearwheel','clearcliff','cleartilt','clearimpact'].forEach(id=>setEnabled(id,canClearLatch));setEnabled('stream',canSession('stream_sensors'));setEnabled('imuzero',canControl('zero_imu_orientation'));setEnabled('imuclear',canControl('clear_imu_orientation'));setEnabled('createrestart',svc&&hasVerb('restart_create'));setEnabled('mbreset',svc&&hasVerb('reset_motherbrain'));setEnabled('bootsel',svc&&hasVerb('bootsel'));setEnabled('createon',canControl('create_power_on'));setEnabled('createoff',canControl('create_power_off'));setEnabled('createbrc',canControl('power_state'));setEnabled('createoi',canControl('power_state'));setEnabledAll('[data-drive]',drive);setEnabled('speed',drive);setEnabled('turn',drive);base.style.pointerEvents=drive?'auto':'none';document.querySelectorAll('[data-action]').forEach(b=>{let v=actionVerb[b.dataset.action],motion=['drive_for','turn_left','turn_right','creep','scan','wiggle','bump_escape','unstick'].indexOf(b.dataset.action)>=0;b.disabled=!(v&&(motion?canMotion(v):canControl(v)))});setEnabled('dock',canMotion('dock'));setEnabled('songdef',canControl('song_define'));setEnabled('songplay',canControl('song_play'));setEnabled('song',canControl('song_define')&&canControl('song_play'));setEnabled('songid',canControl('song_define')||canControl('song_play'));setEnabled('tones',canControl('song_define'));setEnabled('ping',hasVerb('ping'));setEnabled('refresh',true);refreshControlLock();if(caps&&caps.limits){if(caps.limits.max_linear_mm_s)$('speed').max=caps.limits.max_linear_mm_s;if(caps.limits.max_angular_mrad_s)$('turn').max=caps.limits.max_angular_mrad_s}}
function releaseDriveUi(){let wasDriving=active||timer||driveKind;clearInterval(timer);timer=0;active=false;driveKind='';nub.style.left='50%';nub.style.top='50%';document.querySelectorAll('[data-drive].active').forEach(b=>b.classList.remove('active'));return wasDriving}
function stop(){releaseDriveUi();sendCockpit({kind:'stop'})}
function clearLatch(kind){return sendCockpit({kind:'clear_safety_latch',latch:kind}).then(requestStatus)}
function joyMax(){return {lin:+$('speed').value,ang:+$('turn').value}}
function paceDrive(fn){let now=Date.now();if(now-lastDriveAt<120)return;lastDriveAt=now;fn()}
function refreshHeartbeat(){if(!hasVerb('heartbeat_stop'))return;let now=Date.now();if(now-lastHeartbeatAt>550){lastHeartbeatAt=now;sendCockpit({kind:'heartbeat_stop',timeout_ms:900},false)}}
function pulseCmdVel(lin,ang){if(!hasVerb('cmd_vel')){addLog('unsupported cmd_vel');stop();return}refreshHeartbeat();sendCockpit({kind:'cmd_vel',linear_mm_s:lin,angular_mrad_s:ang,ttl_ms:320},false)}
function sendJoy(){paceDrive(()=>{let m=joyMax(),lin=Math.round(-last.y*m.lin),ang=Math.round(-last.x*m.ang);pulseCmdVel(lin,ang)})}
function sendDrive(){paceDrive(()=>{let m=joyMax(),lin=0,ang=0;if(driveKind==='fwd')lin=m.lin;if(driveKind==='back')lin=-m.lin;if(driveKind==='left')ang=m.ang;if(driveKind==='right')ang=-m.ang;if(driveKind==='spinl')ang=m.ang,lin=0;if(driveKind==='spinr')ang=-m.ang,lin=0;if(driveKind==='slow')lin=Math.round(m.lin*.45);pulseCmdVel(lin,ang)})}
function songSlot(){let n=parseInt($('songid').value,10);return Number.isFinite(n)?Math.max(0,Math.min(15,n)):0}
function defineSong(){return sendCockpit({kind:'song_define',id:songSlot(),tones:$('tones').value})}
function behavior(k){let v=actionVerb[k];if(v&&!hasVerb(v)){addLog('unsupported '+v);return}let m=joyMax();if(k==='drive_for')sendCockpit({kind:'drive_for',distance_mm:300,velocity_mm_s:m.lin,timeout_ms:3500});if(k==='turn_left')sendCockpit({kind:'turn_by',angle_mrad:1570,angular_mrad_s:m.ang,timeout_ms:2500});if(k==='turn_right')sendCockpit({kind:'turn_by',angle_mrad:-1570,angular_mrad_s:m.ang,timeout_ms:2500});if(k==='creep')sendCockpit({kind:'creep_until',velocity_mm_s:45,timeout_ms:1200});if(k==='scan')sendCockpit({kind:'scan_arc',angle_mrad:3140,angular_mrad_s:700,timeout_ms:6000});if(k==='wiggle')sendCockpit({kind:'wiggle_align',amplitude_mrad:240,angular_mrad_s:700,cycles:4});if(k==='bump_escape')sendCockpit({kind:'bump_escape',direction:'either'});if(k==='unstick')sendCockpit({kind:'unstick',direction:'either'});if(k==='cliff_trip')sendCockpit({kind:'cliff_guard',clear:false});if(k==='cliff_clear')sendCockpit({kind:'cliff_guard',clear:true});if(k==='heartbeat')sendCockpit({kind:'heartbeat_stop',timeout_ms:1200})}
function move(e){let r=base.getBoundingClientRect(),cx=r.left+r.width/2,cy=r.top+r.height/2,dx=e.clientX-cx,dy=e.clientY-cy,max=r.width*.34,d=Math.hypot(dx,dy);if(d>max){dx=dx/d*max;dy=dy/d*max}last={x:dx/max,y:dy/max};nub.style.left=(50+dx/r.width*100)+'%';nub.style.top=(50+dy/r.height*100)+'%';sendJoy()}
base.onpointerdown=e=>{active=true;base.setPointerCapture(e.pointerId);move(e);timer=setInterval(sendJoy,180)}
base.onpointermove=e=>{if(active)move(e)}
base.onpointerup=base.onpointercancel=stop
$('stop').onclick=stop;$('padstop').onclick=stop
$('sessionnew').onclick=establishBrowserSession
$('controllease').onclick=acquireBrowserControl
$('controlrefresh').onchange=syncControlRefresh
$('estop').onclick=()=>sendCockpit({kind:'estop'})
$('clear').onclick=()=>sendCockpit({kind:'clear_estop'})
$('clearcharge').onclick=()=>clearLatch('charging')
$('clearbump').onclick=()=>clearLatch('bump')
$('clearwheel').onclick=()=>clearLatch('wheel_drop')
$('clearcliff').onclick=()=>clearLatch('cliff')
$('cleartilt').onclick=()=>clearLatch('tilt')
$('clearimpact').onclick=()=>clearLatch('impact')
$('dock').onclick=()=>sendCockpit({kind:'dock'})
$('ping').onclick=()=>sendCockpit({kind:'ping'})
$('imuzero').onclick=()=>sendCockpit({kind:'zero_imu_orientation'}).then(requestStatus)
$('imuclear').onclick=()=>sendCockpit({kind:'clear_imu_orientation'}).then(requestStatus)
$('createrestart').onclick=()=>serviceCommand('restart_create',{kind:'restart_create'})
$('mbreset').onclick=()=>serviceCommand('reset_motherbrain',{kind:'reset_motherbrain'})
$('createon').onclick=()=>sendCockpit({kind:'create_power_on'})
$('createoff').onclick=()=>sendCockpit({kind:'create_power_off'})
$('createbrc').onclick=()=>sendCockpit({kind:'power_state',request:'pulse_brc'})
$('createoi').onclick=()=>sendCockpit({kind:'power_state',request:'start_oi'})
$('songdef').onclick=defineSong
$('songplay').onclick=()=>sendCockpit({kind:'song_play',id:songSlot()})
$('song').onclick=()=>defineSong().then(()=>sendCockpit({kind:'song_play',id:songSlot()}))
$('stream').onclick=()=>sendCockpit({kind:'stream_sensors',enabled:true,packet_id:0,period_ms:250})
$('bootsel').onclick=()=>serviceCommand('bootsel',{kind:'bootsel'})
$('refresh').onclick=connectSse
document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>behavior(b.dataset.action))
document.querySelectorAll('[data-drive]').forEach(b=>{b.onpointerdown=e=>{driveKind=b.dataset.drive;b.classList.add('active');sendDrive();timer=setInterval(sendDrive,190);b.setPointerCapture(e.pointerId)};b.onpointerup=b.onpointercancel=stop})
$('speed').oninput=()=>$('speedv').textContent=$('speed').value
$('turn').oninput=()=>$('turnv').textContent=$('turn').value
function time(ms){let s=Math.floor((ms||0)/1000),m=Math.floor(s/60),h=Math.floor(m/60);return h+'h '+(m%60)+'m '+(s%60)+'s'}
function flagList(cs){let f=[];if(cs.bump_left)f.push('bump L');if(cs.bump_right)f.push('bump R');if(cs.wall)f.push('wall');if(cs.virtual_wall)f.push('virtual wall');if(cs.wheel_drop)f.push('wheel drop');if(cs.cliff_left)f.push('cliff L');if(cs.cliff_front_left)f.push('cliff FL');if(cs.cliff_front_right)f.push('cliff FR');if(cs.cliff_right)f.push('cliff R');return f}
function battPct(cs){return cs.capacity_mah?Math.min(100,Math.round((cs.charge_mah||0)*100/cs.capacity_mah)):null}
function num(v,d=0){return typeof v==='number'&&isFinite(v)?v.toFixed(d):'--'}
function pctBar(id,value,max,badAt,warnAt){let e=$(id),i=e&&e.querySelector('i');if(!e||!i)return;let p=Math.max(0,Math.min(100,(value||0)*100/max));i.style.width=p+'%';e.className='bar '+((value||0)>=badAt?'bad':(value||0)>=warnAt?'warn':'')}
function imuClass(imu){let h=imu.health||'unknown',age=imu.sample_age_ms||0;if(h==='fault'||(h==='ok'&&age>2000))return'badtext';if(h!=='ok'||age>500)return'muted';return'oktext'}
function showImu(imu){imu=imu||{};let present=imu.present||'unknown',health=imu.health||'unknown',age=imu.sample_age_ms||0,poll=imu.poll_period_ms||0,yaw=(imu.yaw_mrad||0)/1000,pitch=(imu.pitch_mrad||0)/1000,roll=(imu.roll_mrad||0)/1000,rate=(imu.yaw_rate_mrad_s||0)/1000,acc=(imu.accel_magnitude_mm_s2||0)/1000,tilt=(imu.tilt_magnitude_mrad||0)/1000,rough=(imu.roughness_mm_s2||0)/1000,impact=(imu.impact_score_mm_s2||0)/1000,av=imu.angular_velocity_mrad_s||{},la=imu.linear_acceleration_mm_s2||{};let cls=imuClass(imu);$('imuhealth').textContent=title(health)+' / '+title(present)+' / samples '+(imu.sample_count||0)+' / age '+age+' ms / '+poll+' ms poll';$('imuhealth').className=cls;$('imuyaw').textContent='yaw '+num(yaw,2)+' / pitch '+num(pitch,2)+' / roll '+num(roll,2)+' rad';$('imuyaw').className=cls;$('imuaccel').textContent=num(acc,2)+' m/s\u00B2 / xyz '+num((la.x||0)/1000,2)+','+num((la.y||0)/1000,2)+','+num((la.z||0)/1000,2);$('imuaccel').className=acc>16?'badtext':acc>12?'muted':cls;$('imutilt').textContent=num(tilt,2)+' rad / '+num(tilt*57.2958,1)+' deg';$('imutilt').className=tilt>.65?'badtext':tilt>.35?'muted':cls;$('imurates').textContent='yaw '+num(rate,2)+' rad/s / xyz '+num((av.x||0)/1000,2)+','+num((av.y||0)/1000,2)+','+num((av.z||0)/1000,2);$('imurates').className=cls;$('imurough').textContent=num(rough,2)+' m/s\u00B2';$('imurough').className=rough>8?'badtext':rough>3?'muted':cls;$('imuimpact').textContent=num(impact,2)+' m/s\u00B2';$('imuimpact').className=impact>18?'badtext':impact>8?'muted':cls;$('imumotion').textContent=title(imu.motion_consistency||'unknown')+' / '+title(imu.calibration||'uncalibrated');$('imumotion').className=(imu.motion_consistency==='inconsistent'||imu.calibration==='uncalibrated')?'muted':cls;pctBar('imuaccelbar',imu.accel_magnitude_mm_s2||0,22000,18000,13000);pctBar('imutiltbar',imu.tilt_magnitude_mrad||0,1000,650,350);pctBar('imuroughbar',imu.roughness_mm_s2||0,12000,8000,3000);pctBar('imuimpactbar',imu.impact_score_mm_s2||0,22000,18000,8000)}
function showStatus(s){lastStatus=s;let cs=s.create_sensors||{},od=s.odometry||{},imu=s.imu||{},music=s.create_songs||{},fatal=s.current_runtime_state==='error'||(s.last_error&&s.last_error!=='none'),contact=cs.bump_left||cs.bump_right||cs.wall||cs.virtual_wall,charging=chargeActive(cs),imuOk=imu.health==='ok',imuDanger=imu.health==='fault'||(imuOk&&((imu.tilt_magnitude_mrad||0)>=650||(imu.impact_score_mm_s2||0)>=18000)),safetyStop=s.estop_latched||s.safety_tripped||s.motion_interlock_latched||charging||cs.wheel_drop||cs.cliff_left||cs.cliff_front_left||cs.cliff_front_right||cs.cliff_right||imuDanger,pct=battPct(cs),flags=flagList(cs),latchKind=s.safety_latch_kind&&s.safety_latch_kind!=='none'?s.safety_latch_kind:'';if(s.estop_latched)flags.push('e-stop');if(s.safety_tripped)flags.push(latchKind?title(latchKind)+' latch':'safety latch');if(s.motion_interlock_latched)flags.push('charge latch');if(charging)flags.push('charging');if(imuOk&&(imu.tilt_magnitude_mrad||0)>=650)flags.push('tilt');if(imuOk&&(imu.impact_score_mm_s2||0)>=18000)flags.push('impact');if(imuOk&&imu.motion_consistency==='inconsistent')flags.push('motion mismatch');let safetyText=flags.join(', ')||'clear';pill(net,wsOpen?'control ws':sseOpen?'telemetry sse':(s.wifi_state||'online'),'ok');pill($('mode'),title(s.oi_mode),(s.oi_mode==='safe'||s.oi_mode==='full')?'ok':'');pill($('safety'),safetyStop?'motion blocked':contact?'contact':'clear',safetyStop?'bad':contact?'warn':'ok');$('headline').textContent=title(s.current_runtime_state)+' / '+title(s.create_power_state)+' / '+title(s.uart_rx_health)+' / IMU '+title(imu.health||'unknown');$('runtime').textContent=title(s.current_runtime_state)+' / body '+title(s.body_state);$('uptime').textContent=time(s.uptime_ms);$('create').textContent=title(s.create_power_state)+' / '+title(s.oi_mode)+' / probe '+s.wake_probe_response_bytes+'/'+s.wake_probe_expected_bytes;$('safetyread').textContent=safetyText;$('safetyread').className=safetyStop?'badtext':contact?'muted':'oktext';$('uart').textContent=title(s.uart_rx_health)+' / '+title(s.last_uart_read_error)+' / '+s.uart_rx_packets+' packets';$('cmd').textContent=title(s.current_command)+' / pending '+title(s.pending_command)+' #'+s.pending_command_id;$('forebrain').textContent=(s.forebrain_uart?s.forebrain_uart.rx_lines:0)+' lines / '+title(s.forebrain_uart&&s.forebrain_uart.last_error);$('web').textContent=s.http_requests+' requests / '+s.dhcp_grants+' dhcp';$('sensors').textContent='pkt '+(cs.last_packet_id||0)+' / IR '+(cs.ir_byte||0)+' / buttons '+(cs.buttons||0)+' / cliff sig '+(cs.cliff_left_signal||0)+','+(cs.cliff_front_left_signal||0)+','+(cs.cliff_front_right_signal||0)+','+(cs.cliff_right_signal||0);$('battery').textContent=(pct===null?'--':pct+'%')+' / '+(cs.voltage_mv||0)+' mV / '+(cs.current_ma||0)+' mA / '+(cs.charge_mah||0)+'/'+(cs.capacity_mah||0)+' mAh / charge state '+(cs.charging_state||0)+' / charge pin '+title(cs.charging_indicator);$('battery').className=(charging||pct!==null&&pct<=20)?'badtext':'muted';$('odom').textContent='delta '+(cs.distance_mm||0)+' mm / '+(cs.angle_mrad||0)+' mrad / total '+(od.distance_mm||0)+' mm / '+(od.heading_mrad||0)+' mrad / resets '+(od.reset_count||0);showImu(imu);$('music').textContent='defined '+(music.last_defined_id||0)+' ('+(music.last_defined_len||0)+') / played '+(music.last_played_id||0);$('firmware').textContent=s.firmware_name+' '+s.firmware_version;$('err').textContent=fatal?title(s.last_error)+' / '+(s.last_error_hint||''): 'none';$('err').className=fatal?'badtext':'muted';applyCaps()}
function handleEvents(batch){let stopNeeded=false,refreshNeeded=false;eventCursor=Math.max(0,(batch.next_seq||1)-1);if(batch.dropped_before_seq){$('events').textContent='recovered after '+batch.dropped_before_seq;pill($('safety'),'event history recovered','warn');addLog('recovered event history after '+batch.dropped_before_seq);stopNeeded=true}else{$('events').textContent='cursor '+(batch.next_seq||0)+' / '+((batch.events||[]).length)+' new'}(batch.events||[]).forEach(e=>{let k=e.kind;if(['safety_tripped','heartbeat_expired','estop_latched','wheel_drop_latched'].indexOf(k)>=0){pill($('safety'),title(k),'bad');addLog('safety '+k+' '+(e.a||0));stopNeeded=true;refreshNeeded=true}else if(k==='safety_cleared'){pill($('safety'),'clear','ok');$('safetyread').textContent='clear';$('safetyread').className='oktext';addLog(k+' '+(e.a||0));refreshNeeded=true}else if(['imu_frame_received','imu_fault','tilt_changed','impact_detected','imu_calibration_changed'].indexOf(k)>=0){addLog(k+' '+(e.a||0));refreshNeeded=true}else if(['bump_changed','wall_changed','virtual_wall_changed','buttons_changed','ir_changed','charging_state_changed','battery_low','cliff_changed','wheel_drop_cleared'].indexOf(k)>=0){addLog(k+' '+(e.a||0));refreshNeeded=true}else if(['command_rejected','command_interrupted'].indexOf(k)>=0){pill($('safety'),title(k),'warn');addLog(k+' #'+(e.a||0));refreshNeeded=true}else if(k==='motion_stopped'){addLog('motion stopped')}else if(k==='error'){pill($('safety'),'fatal/error','bad');addLog('error '+(e.a||0));stopNeeded=true;refreshNeeded=true}});if(refreshNeeded)requestStatus();if(stopNeeded&&releaseDriveUi())sendCockpit({kind:'stop'})}
applyCaps();establishBrowserSession().catch(connectSse);
</script>
</body>
</html>
"#
}

enum CommandParseError {
    BadRequest,
    Busy(u32, &'static str),
}

fn handle_handshake_json<'a>(body: &str, buffer: &'a mut [u8], transport: u8) -> Option<&'a str> {
    let hello = match session::parse_json(body) {
        Ok(hello) => hello,
        Err(reason) => return render_handshake_reject(buffer, "", reason),
    };
    let mut device_id = heapless::String::<32>::new();
    let mut boot_id = heapless::String::<32>::new();
    let instance = BRAINSTEM_INSTANCE_ID.load(Ordering::Acquire);
    if instance == 0 {
        return render_handshake_reject(
            buffer,
            hello.handshake_nonce.as_str(),
            session::RejectReason::InvalidIdentity,
        );
    }
    let _ = write!(device_id, "pete-brainstem-{instance:04x}");
    let _ = write!(
        boot_id,
        "bsboot-{:08x}",
        BRAINSTEM_BOOT_ID.load(Ordering::Acquire)
    );
    let accepted = match session::validate(&hello, device_id.as_str(), boot_id.as_str()) {
        Ok(accepted) => accepted,
        Err(reason) => {
            return render_handshake_reject(buffer, hello.handshake_nonce.as_str(), reason)
        }
    };
    let session_hash = session::token_hash(accepted.session_id.as_str());
    let peer_hash = session::token_hash(hello.device_id.as_str());
    if hello.role == session::EndpointRole::Motherbrain
        && hello.session_purpose == session::SessionPurpose::Control
    {
        if transport != TransportKind::HardwareUart as u8
            && transport != TransportKind::UsbCdc as u8
            && !status::active_peer_matches(peer_hash)
        {
            return render_handshake_reject(
                buffer,
                hello.handshake_nonce.as_str(),
                session::RejectReason::InvalidIdentity,
            );
        }
        status::request_session_replace(
            accepted.generation,
            session_hash,
            peer_hash,
            session::token_hash(hello.boot_id.as_str()),
        );
        for _ in 0..250 {
            if status::session_replace_acked(accepted.generation) {
                break;
            }
            embassy_time::block_for(Duration::from_millis(1));
        }
        if !status::session_replace_acked(accepted.generation) {
            return None;
        }
        status::mark_transport_changed(transport);
    } else {
        let role = match hello.role {
            session::EndpointRole::Motherbrain => 1,
            session::EndpointRole::Forebrain => 2,
            session::EndpointRole::Operator => 3,
            session::EndpointRole::ServiceTool => 4,
            _ => 0,
        };
        let purpose = match hello.session_purpose {
            session::SessionPurpose::Control => 1,
            session::SessionPurpose::Diagnostic => 2,
        };
        status::register_diagnostic_session(
            session_hash,
            peer_hash,
            session::token_hash(hello.boot_id.as_str()),
            role,
            purpose,
            transport,
        );
    }
    render_handshake_welcome(
        buffer,
        &hello,
        &accepted,
        device_id.as_str(),
        boot_id.as_str(),
    )
}

fn render_handshake_reject<'a>(
    buffer: &'a mut [u8],
    nonce: &str,
    reason: session::RejectReason,
) -> Option<&'a str> {
    status::mark_session_rejected(reason.code());
    let mut response = heapless::String::<512>::new();
    write!(response, "{{\"kind\":\"reject\",\"echoed_handshake_nonce\":\"{nonce}\",\"reason_code\":\"{}\",\"message\":\"handshake rejected\",\"supported_protocol_major\":1,\"supported_minor_min\":0,\"supported_minor_max\":0}}", reason.as_str()).ok()?;
    copy_response(buffer, response.as_str())
}

fn render_handshake_welcome<'a>(
    buffer: &'a mut [u8],
    hello: &session::Hello,
    accepted: &session::AcceptedHello,
    device_id: &str,
    boot_id: &str,
) -> Option<&'a str> {
    let caps = capabilities::current();
    let snapshot = status::snapshot(Instant::now().as_millis() as u32);
    let (estop_latched, safety_tripped, motion_interlock_latched, _) =
        status::session_safety_snapshot();
    let mut response = heapless::String::<4096>::new();
    write!(response, "{{\"kind\":\"welcome\",\"role\":\"brainstem\",\"device_id\":\"{device_id}\",\"boot_id\":\"{boot_id}\",\"echoed_handshake_nonce\":\"{}\",\"session_id\":\"{}\",\"protocol_major\":1,\"protocol_minor\":{},\"supported_features\":[\"session_ids\",\"event_cursor\",\"heartbeat\",\"transport_failover\"],\"required_features\":[\"session_ids\"],\"heartbeat_min_ms\":250,\"heartbeat_max_ms\":2000,\"command_ttl_min_ms\":{},\"command_ttl_max_ms\":{},\"current_event_next_seq\":{},\"capability_contract\":{{\"body_kind\":\"{}\",\"drive\":\"{}\",", hello.handshake_nonce, accepted.session_id, accepted.negotiated_minor, caps.min_ttl_ms, caps.max_ttl_ms, snapshot.event_next_seq, caps.body_kind, caps.drive).ok()?;
    write_json_array(&mut response, "verbs", caps.verbs)?;
    write_json_array(&mut response, "sensors", caps.sensors)?;
    write_json_array(&mut response, "outputs", caps.outputs)?;
    write_json_array(&mut response, "safety", caps.safety)?;
    write_json_array(&mut response, "events", caps.events)?;
    let active_motion = snapshot.body_state == status::BodyState::Moving as u8;
    write!(response, "\"limits\":{{\"max_linear_mm_s\":{},\"max_angular_mrad_s\":{},\"min_ttl_ms\":{},\"max_ttl_ms\":{}}}}},\"software\":{{\"software_name\":\"{}\",\"software_version\":\"{}\",\"build_id\":\"{}\"}},\"safety_snapshot\":{{\"armed\":false,\"estop_latched\":{},\"safety_tripped\":{},\"motion_interlock_latched\":{},\"active_motion\":{},\"runtime_state\":\"{}\"}}}}", caps.max_linear_mm_s, caps.max_angular_mrad_s, caps.min_ttl_ms, caps.max_ttl_ms, caps.firmware_name, caps.firmware_version, option_env!("PETE_BUILD_ID").unwrap_or("development"), estop_latched, safety_tripped, motion_interlock_latched, active_motion, if active_motion { "moving" } else { "idle" }).ok()?;
    copy_response(buffer, response.as_str())
}

fn write_json_array<const N: usize>(
    response: &mut heapless::String<N>,
    key: &str,
    values: &[&str],
) -> Option<()> {
    write!(response, "\"{key}\":[").ok()?;
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            response.push(',').ok()?;
        }
        write!(response, "\"{value}\"").ok()?;
    }
    response.push_str("],").ok()?;
    Some(())
}

fn copy_response<'a>(buffer: &'a mut [u8], response: &str) -> Option<&'a str> {
    if response.len() > buffer.len() {
        return None;
    }
    buffer[..response.len()].copy_from_slice(response.as_bytes());
    core::str::from_utf8(&buffer[..response.len()]).ok()
}

fn render_network_diagnostics(buffer: &mut [u8], now_ms: u32) -> Option<&str> {
    let diagnostics = network_registry::diagnostics(now_ms);
    let mut response = heapless::String::<256>::new();
    write!(
        response,
        "{{\"active_leases\":{},\"registration_generation\":{},\"motherbrain_address\":",
        diagnostics.active_leases, diagnostics.registration_generation
    )
    .ok()?;
    if let Some(ip) = diagnostics.motherbrain_ip {
        write!(response, "\"{}.{}.{}.{}\"", ip[0], ip[1], ip[2], ip[3]).ok()?;
    } else {
        response.push_str("null").ok()?;
    }
    response.push('}').ok()?;
    copy_response(buffer, response.as_str())
}

fn render_session_diagnostics(buffer: &mut [u8], now_ms: u32) -> Option<&str> {
    let diagnostics = status::session_diagnostics(now_ms);
    let mut response = heapless::String::<256>::new();
    write!(response, "{{\"primary_session_generation\":{},\"diagnostic_sessions\":{},\"authority_generation\":{},\"authority_active\":{},\"service_authority_active\":{}}}", diagnostics.primary_session_generation, diagnostics.diagnostic_sessions, diagnostics.authority_generation, diagnostics.authority_active, diagnostics.service_authority_active).ok()?;
    copy_response(buffer, response.as_str())
}

fn handle_command_request<'a>(
    request: &[u8],
    buffer: &'a mut [u8],
) -> Result<&'a str, CommandParseError> {
    let body = request_body(request).ok_or(CommandParseError::BadRequest)?;
    let command_id = json_u32(body, "command_id").ok_or(CommandParseError::BadRequest)?;
    if json_str(body, "kind") == Some("register_network_endpoint") {
        return handle_network_registration_json(body, buffer).ok_or(CommandParseError::BadRequest);
    }
    if json_str(body, "kind") == Some("acquire_control_lease") {
        return handle_authority_json(body, buffer).ok_or(CommandParseError::BadRequest);
    }
    if json_str(body, "kind") == Some("acquire_service_lease") {
        return handle_service_authority_json(body, buffer).ok_or(CommandParseError::BadRequest);
    }
    let command = parse_command(command_id, body).ok_or(CommandParseError::BadRequest)?;
    if command_id == 0 {
        return render_command_response(buffer, false, command_id, "invalid_command_id")
            .ok_or(CommandParseError::BadRequest);
    }
    if matches!(command, BrainstemCommand::Status) {
        let snapshot = status::snapshot(Instant::now().as_millis() as u32);
        return status::render_json(snapshot, buffer).map_err(|_| CommandParseError::BadRequest);
    }
    if let BrainstemCommand::GetEvents { since_seq } = command {
        return status::render_events_json(since_seq, buffer).ok_or(CommandParseError::BadRequest);
    }
    if matches!(command, BrainstemCommand::GetCapabilities) {
        return render_capabilities_response(buffer, command_id)
            .ok_or(CommandParseError::BadRequest);
    }
    if matches!(command, BrainstemCommand::Ping) {
        return render_command_response(buffer, true, command_id, "pong")
            .ok_or(CommandParseError::BadRequest);
    }
    if matches!(command, BrainstemCommand::Bootsel) && !json_service_authority_valid(body) {
        return render_command_response(
            buffer,
            false,
            command_id,
            "service_authorization_required",
        )
        .ok_or(CommandParseError::BadRequest);
    }
    if command_requires_session(command) && !json_session_valid(body) {
        return render_command_response(buffer, false, command_id, "invalid_session")
            .ok_or(CommandParseError::BadRequest);
    }
    if command_requires_authority(command) && !json_authority_valid(body) {
        return render_command_response(buffer, false, command_id, "invalid_control_lease")
            .ok_or(CommandParseError::BadRequest);
    }
    if command_requires_service_authority(command) && !json_service_authority_valid(body) {
        return render_command_response(buffer, false, command_id, "invalid_service_lease")
            .ok_or(CommandParseError::BadRequest);
    }
    if matches!(command, BrainstemCommand::Bootsel) {
        return render_command_response(
            buffer,
            cfg!(feature = "service-mode"),
            command_id,
            if cfg!(feature = "service-mode") {
                "bootsel_accepted"
            } else {
                "service_operation_disabled"
            },
        )
        .ok_or(CommandParseError::BadRequest);
    }
    if !submit_json_control_command(command_id, command, body) {
        return Err(CommandParseError::Busy(command_id, control_busy_reason()));
    }
    render_command_response(buffer, true, command_id, "accepted")
        .ok_or(CommandParseError::BadRequest)
}

fn handle_websocket_message<'a>(body: &str, buffer: &'a mut [u8]) -> Option<&'a str> {
    if json_str(body, "kind") == Some("hello") {
        return handle_handshake_json(body, buffer, TransportKind::WebSocket as u8);
    }
    if json_str(body, "kind") == Some("register_network_endpoint") {
        return handle_network_registration_json(body, buffer);
    }
    if json_str(body, "kind") == Some("acquire_control_lease") {
        return handle_authority_json(body, buffer);
    }
    if json_str(body, "kind") == Some("acquire_service_lease") {
        return handle_service_authority_json(body, buffer);
    }
    if json_str(body, "kind") == Some("status") {
        let snapshot = status::snapshot(Instant::now().as_millis() as u32);
        return render_status_websocket_response(snapshot, buffer);
    }
    if json_str(body, "kind") == Some("get_capabilities") {
        let command_id = json_u32(body, "command_id")?;
        return render_capabilities_response(buffer, command_id);
    }
    if json_str(body, "kind") == Some("get_events") {
        let since_seq = json_u32(body, "since_seq").unwrap_or(0);
        return status::render_events_json(since_seq, buffer);
    }
    if json_str(body, "kind") == Some("ping") {
        let command_id = json_u32(body, "command_id")?;
        return render_command_response(buffer, true, command_id, "pong");
    }

    if json_bool(body, "ack") == Some(false) {
        let command_id = json_u32(body, "command_id")?;
        let command = parse_command(command_id, body)?;
        if command_id == 0 {
            return render_command_response(buffer, false, command_id, "invalid_command_id");
        }
        if matches!(command, BrainstemCommand::Bootsel) && !json_service_authority_valid(body) {
            return render_command_response(
                buffer,
                false,
                command_id,
                "service_authorization_required",
            );
        }
        if command_requires_session(command) && !json_session_valid(body) {
            return render_command_response(buffer, false, command_id, "invalid_session");
        }
        if command_requires_authority(command) && !json_authority_valid(body) {
            return render_command_response(buffer, false, command_id, "invalid_control_lease");
        }
        if command_requires_service_authority(command) && !json_service_authority_valid(body) {
            return render_command_response(buffer, false, command_id, "invalid_service_lease");
        }
        if matches!(command, BrainstemCommand::Bootsel) {
            return render_command_response(
                buffer,
                false,
                command_id,
                "service_operation_disabled",
            );
        }
        let accepted = submit_json_control_command(command_id, command, body);
        if accepted {
            None
        } else {
            render_command_response(buffer, false, command_id, control_busy_reason())
        }
    } else {
        handle_websocket_command(body, buffer)
            .or(Some("{\"accepted\":false,\"message\":\"bad_request\"}\n"))
    }
}

fn handle_websocket_command<'a>(body: &str, buffer: &'a mut [u8]) -> Option<&'a str> {
    let command_id = json_u32(body, "command_id")?;
    let command = parse_command(command_id, body)?;
    if command_id == 0 {
        return render_command_response(buffer, false, command_id, "invalid_command_id");
    }
    if matches!(command, BrainstemCommand::Bootsel) && !json_service_authority_valid(body) {
        return render_command_response(
            buffer,
            false,
            command_id,
            "service_authorization_required",
        );
    }
    if command_requires_session(command) && !json_session_valid(body) {
        return render_command_response(buffer, false, command_id, "invalid_session");
    }
    if command_requires_authority(command) && !json_authority_valid(body) {
        return render_command_response(buffer, false, command_id, "invalid_control_lease");
    }
    if command_requires_service_authority(command) && !json_service_authority_valid(body) {
        return render_command_response(buffer, false, command_id, "invalid_service_lease");
    }
    if matches!(command, BrainstemCommand::Bootsel) {
        return render_command_response(buffer, false, command_id, "service_operation_disabled");
    }
    if !submit_json_control_command(command_id, command, body) {
        return render_command_response(buffer, false, command_id, control_busy_reason());
    }
    render_command_response(buffer, true, command_id, "accepted")
}

fn render_status_websocket_response<'a>(
    snapshot: status::BrainstemStatus,
    buffer: &'a mut [u8],
) -> Option<&'a str> {
    let body_len = {
        let body = status::render_json(snapshot, buffer).ok()?;
        body.len()
    };
    let prefix = br#"{"type":"status","#;
    let extra_len = prefix.len().checked_sub(1)?;
    let new_len = body_len.checked_add(extra_len)?;
    if new_len > buffer.len() {
        return None;
    }

    for i in (1..body_len).rev() {
        buffer[i + extra_len] = buffer[i];
    }
    buffer[..prefix.len()].copy_from_slice(prefix);
    core::str::from_utf8(&buffer[..new_len]).ok()
}

fn handle_forebrain_uart_line(uart: &mut Uart<'static, Blocking>, line: &[u8]) {
    if line.is_empty() {
        return;
    }

    let line = match core::str::from_utf8(line) {
        Ok(line) => line,
        Err(_) => {
            status::mark_forebrain_uart_error(status::ForebrainUartErrorCode::Utf8);
            submit_forebrain_stop();
            write_forebrain_uart_line(uart, b"ERR 0 utf8\n");
            return;
        }
    };

    if let Some(body) = line.strip_prefix("HELLO ") {
        let mut response = [0u8; 4096];
        if let Some(welcome) =
            handle_handshake_json(body, &mut response, TransportKind::HardwareUart as u8)
        {
            let prefix = if welcome.contains("\"kind\":\"reject\"") {
                b"REJECT ".as_slice()
            } else {
                b"WELCOME ".as_slice()
            };
            write_forebrain_uart_line(uart, prefix);
            write_forebrain_uart_line(uart, welcome.as_bytes());
            write_forebrain_uart_line(uart, b"\n");
        } else {
            write_forebrain_uart_line(uart, b"ERR 0 handshake\n");
        }
        return;
    }
    if line.starts_with("REGISTER_NETWORK_ENDPOINT ") {
        let mut response = heapless::String::<512>::new();
        handle_network_registration_compact(line, &mut response);
        write_forebrain_uart_line(uart, response.as_bytes());
        return;
    }
    if line.starts_with("ACQUIRE_CONTROL_LEASE ") {
        let mut response = heapless::String::<512>::new();
        handle_authority_compact(line, &mut response);
        write_forebrain_uart_line(uart, response.as_bytes());
        return;
    }
    if line.starts_with("ACQUIRE_SERVICE_LEASE ") {
        let mut response = heapless::String::<512>::new();
        handle_service_authority_compact(line, &mut response);
        write_forebrain_uart_line(uart, response.as_bytes());
        return;
    }

    let (command_line, session_id, lease_id, service_lease_id) = compact_envelope(line);
    let (seq, command) = match parse_forebrain_uart_command(command_line) {
        Ok(parsed) => parsed,
        Err(seq) => {
            status::mark_forebrain_uart_error(status::ForebrainUartErrorCode::Parse);
            submit_forebrain_stop();
            write_forebrain_uart_error(uart, seq, "parse");
            return;
        }
    };

    status::mark_forebrain_uart_command(seq, Instant::now().as_millis() as u32);
    if matches!(command, BrainstemCommand::Bootsel) {
        if !compact_service_authority_valid(session_id, service_lease_id, command) {
            write_forebrain_uart_error(uart, seq, "service_authorization_required");
            return;
        }
        write_forebrain_uart_error(uart, seq, "service_operation_disabled");
        return;
    }
    if matches!(command, BrainstemCommand::GetCapabilities) {
        write_forebrain_uart_capabilities(uart, seq);
        return;
    }
    if let BrainstemCommand::GetEvents { since_seq } = command {
        write_forebrain_uart_events(uart, seq, since_seq);
        return;
    }
    if command_requires_session(command) && !session_id.is_some_and(compact_session_valid) {
        write_forebrain_uart_error(uart, seq, "invalid_session");
        return;
    }
    if command_requires_authority(command)
        && !compact_authority_valid(command, session_id, lease_id)
    {
        write_forebrain_uart_error(uart, seq, "invalid_control_lease");
        return;
    }
    if command_requires_service_authority(command)
        && !compact_service_authority_valid(session_id, service_lease_id, command)
    {
        write_forebrain_uart_error(uart, seq, "invalid_service_lease");
        return;
    }

    if !submit_compact_control_command(seq, command, session_id, service_lease_id) {
        status::mark_forebrain_uart_error(status::ForebrainUartErrorCode::Busy);
        if matches!(command, BrainstemCommand::CmdVel { .. }) {
            submit_forebrain_stop();
        }
        write_forebrain_uart_error(uart, seq, control_busy_reason());
        return;
    }

    if matches!(command, BrainstemCommand::Status) {
        write_forebrain_uart_status(uart, seq);
    } else {
        write_forebrain_uart_ok(uart, seq);
    }
}

fn parse_forebrain_uart_command(line: &str) -> Result<(u32, BrainstemCommand), u32> {
    let mut parts = line.split_ascii_whitespace();
    let Some(kind) = parts.next() else {
        return Err(0);
    };
    let seq = parse_u32(parts.next()).ok_or(0u32)?;
    if seq == 0 {
        return Err(0);
    }

    let command = match kind {
        "PING" => BrainstemCommand::Ping,
        "BOOTSEL" => BrainstemCommand::Bootsel,
        "RESET_MOTHERBRAIN" => BrainstemCommand::ResetMotherbrain,
        "ARM" => BrainstemCommand::Arm,
        "DISARM" => BrainstemCommand::Disarm,
        "SET_MODE" => {
            BrainstemCommand::SetMode(parse_oi_mode(parts.next().ok_or(seq)?).ok_or(seq)?)
        }
        "STOP" => BrainstemCommand::Stop,
        "ESTOP" => BrainstemCommand::EStop,
        "CLEAR_ESTOP" => BrainstemCommand::ClearEStop,
        "CMD_VEL" => BrainstemCommand::CmdVel {
            seq,
            linear_mm_s: parse_i16(parts.next()).ok_or(seq)?,
            angular_mrad_s: parse_i16(parts.next()).ok_or(seq)?,
            ttl_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "DRIVE_DIRECT" => BrainstemCommand::DriveDirect {
            seq,
            left_mm_s: parse_i16(parts.next()).ok_or(seq)?,
            right_mm_s: parse_i16(parts.next()).ok_or(seq)?,
            ttl_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "DRIVE_ARC" => BrainstemCommand::DriveArc {
            seq,
            velocity_mm_s: parse_i16(parts.next()).ok_or(seq)?,
            radius_mm: parse_i16(parts.next()).ok_or(seq)?,
            ttl_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "FACE_BEARING" => BrainstemCommand::FaceBearing {
            seq,
            bearing_mrad: parse_i16(parts.next()).ok_or(seq)?,
            max_angular_mrad_s: parse_i16(parts.next()).ok_or(seq)?,
            tolerance_mrad: parse_i16(parts.next()).ok_or(seq)?,
            ttl_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "TRACK_BEARING" => BrainstemCommand::TrackBearing {
            seq,
            bearing_mrad: parse_i16(parts.next()).ok_or(seq)?,
            range_mm: parse_u32(parts.next()).ok_or(seq)? as u16,
            max_linear_mm_s: parse_i16(parts.next()).ok_or(seq)?,
            max_angular_mrad_s: parse_i16(parts.next()).ok_or(seq)?,
            stop_range_mm: parse_u32(parts.next()).ok_or(seq)? as u16,
            ttl_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "TURN_BY" => BrainstemCommand::TurnBy {
            seq,
            angle_mrad: parse_i16(parts.next()).ok_or(seq)?,
            angular_mrad_s: parse_i16(parts.next()).ok_or(seq)?,
            timeout_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "DRIVE_FOR" => BrainstemCommand::DriveFor {
            seq,
            distance_mm: parse_i16(parts.next()).ok_or(seq)?,
            velocity_mm_s: parse_i16(parts.next()).ok_or(seq)?,
            timeout_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "BUMP_ESCAPE" => BrainstemCommand::BumpEscape {
            seq,
            direction: parse_escape_direction(parts.next().ok_or(seq)?).ok_or(seq)?,
            backoff_mm_s: parse_i16(parts.next()).ok_or(seq)?,
            turn_angular_mrad_s: parse_i16(parts.next()).ok_or(seq)?,
        },
        "HOLD_HEADING" => BrainstemCommand::HoldHeading {
            seq,
            heading_error_mrad: parse_i16(parts.next()).ok_or(seq)?,
            velocity_mm_s: parse_i16(parts.next()).ok_or(seq)?,
            max_angular_mrad_s: parse_i16(parts.next()).ok_or(seq)?,
            ttl_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "TURN_TO_HEADING" => BrainstemCommand::TurnToHeading {
            seq,
            heading_error_mrad: parse_i16(parts.next()).ok_or(seq)?,
            angular_mrad_s: parse_i16(parts.next()).ok_or(seq)?,
            tolerance_mrad: parse_i16(parts.next()).ok_or(seq)?,
            timeout_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "ARC_FOR" => BrainstemCommand::ArcFor {
            seq,
            velocity_mm_s: parse_i16(parts.next()).ok_or(seq)?,
            radius_mm: parse_i16(parts.next()).ok_or(seq)?,
            duration_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "CREEP_UNTIL" => BrainstemCommand::CreepUntil {
            seq,
            velocity_mm_s: parse_i16(parts.next()).ok_or(seq)?,
            angular_mrad_s: parse_i16(parts.next()).ok_or(seq)?,
            timeout_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "SCAN_ARC" => BrainstemCommand::ScanArc {
            seq,
            angle_mrad: parse_i16(parts.next()).ok_or(seq)?,
            angular_mrad_s: parse_i16(parts.next()).ok_or(seq)?,
            timeout_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "DOCK_ALIGN" => BrainstemCommand::DockAlign {
            seq,
            bearing_mrad: parse_i16(parts.next()).ok_or(seq)?,
            range_mm: parse_u32(parts.next()).ok_or(seq)? as u16,
            max_linear_mm_s: parse_i16(parts.next()).ok_or(seq)?,
            max_angular_mrad_s: parse_i16(parts.next()).ok_or(seq)?,
            stop_range_mm: parse_u32(parts.next()).ok_or(seq)? as u16,
            ttl_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "WALL_FOLLOW" => BrainstemCommand::WallFollow {
            seq,
            distance_error_mm: parse_i16(parts.next()).ok_or(seq)?,
            velocity_mm_s: parse_i16(parts.next()).ok_or(seq)?,
            max_angular_mrad_s: parse_i16(parts.next()).ok_or(seq)?,
            ttl_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "WIGGLE_ALIGN" => BrainstemCommand::WiggleAlign {
            seq,
            amplitude_mrad: parse_i16(parts.next()).ok_or(seq)?,
            angular_mrad_s: parse_i16(parts.next()).ok_or(seq)?,
            cycles: parse_u32(parts.next()).ok_or(seq)? as u8,
        },
        "UNSTICK" => BrainstemCommand::Unstick {
            seq,
            direction: parse_escape_direction(parts.next().ok_or(seq)?).ok_or(seq)?,
            backoff_mm_s: parse_i16(parts.next()).ok_or(seq)?,
            turn_angular_mrad_s: parse_i16(parts.next()).ok_or(seq)?,
        },
        "CLIFF_GUARD" => BrainstemCommand::CliffGuard {
            seq,
            clear: parse_bool(parts.next()).ok_or(seq)?,
        },
        "CLEAR_SAFETY_LATCH" => BrainstemCommand::ClearSafetyLatch {
            seq,
            kind: parse_safety_latch_kind(parts.next().ok_or(seq)?).ok_or(seq)?,
        },
        "HEARTBEAT_STOP" => BrainstemCommand::HeartbeatStop {
            seq,
            timeout_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "REQUEST_SENSORS" => BrainstemCommand::RequestSensors {
            seq,
            packet_id: parse_u32(parts.next()).ok_or(seq)? as u8,
        },
        "STREAM_SENSORS" => BrainstemCommand::StreamSensors {
            seq,
            enabled: parse_bool(parts.next()).ok_or(seq)?,
            packet_id: parse_u32(parts.next()).ok_or(seq)? as u8,
            period_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "SET_SAFETY_POLICY" => BrainstemCommand::SetSafetyPolicy {
            seq,
            policy: SafetyPolicy {
                bump: parse_safety_action(parts.next().ok_or(seq)?).ok_or(seq)?,
                cliff: parse_safety_action(parts.next().ok_or(seq)?).ok_or(seq)?,
                wheel_drop_latch: parse_bool(parts.next()).ok_or(seq)?,
            },
        },
        "CLEAR_MOTION_QUEUE" => BrainstemCommand::ClearMotionQueue { seq },
        "DEFINE_CHIRP" => {
            let kind = parse_feedback_kind(parts.next().ok_or(seq)?).ok_or(seq)?;
            let mut tones = [SongTone::default(); MAX_SONG_TONES];
            let mut tone_count = 0;
            while tone_count < MAX_SONG_TONES {
                let Some(note) = parts.next() else {
                    break;
                };
                let duration = parts.next().ok_or(seq)?;
                tones[tone_count] = SongTone {
                    note: parse_u32(Some(note)).ok_or(seq)? as u8,
                    duration_64ths: parse_u32(Some(duration)).ok_or(seq)? as u8,
                };
                tone_count += 1;
            }
            if tone_count == 0 {
                return Err(seq);
            }
            BrainstemCommand::DefineChirp {
                kind,
                tones,
                tone_count: tone_count as u8,
                seq,
            }
        }
        "PLAY_FEEDBACK" => BrainstemCommand::PlayFeedback {
            seq,
            kind: parse_feedback_kind(parts.next().ok_or(seq)?).ok_or(seq)?,
        },
        "POWER_STATE" => BrainstemCommand::PowerState {
            seq,
            request: parse_power_request(parts.next().ok_or(seq)?).ok_or(seq)?,
        },
        "CREATE_POWER_ON" => BrainstemCommand::PowerState {
            seq,
            request: PowerStateRequest::Wake,
        },
        "CREATE_POWER_OFF" => BrainstemCommand::PowerState {
            seq,
            request: PowerStateRequest::Sleep,
        },
        "CALIBRATE_TURN" => BrainstemCommand::CalibrateTurn {
            seq,
            angular_mrad_s: parse_i16(parts.next()).ok_or(seq)?,
            duration_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "RESET_ODOMETRY" => BrainstemCommand::ResetOdometry { seq },
        "ZERO_IMU_ORIENTATION" => BrainstemCommand::ZeroImuOrientation { seq },
        "CLEAR_IMU_ORIENTATION" => BrainstemCommand::ClearImuOrientation { seq },
        "GET_CAPABILITIES" => BrainstemCommand::GetCapabilities,
        "GET_EVENTS" => BrainstemCommand::GetEvents {
            since_seq: parse_u32(parts.next()).unwrap_or(0),
        },
        "STATUS" => BrainstemCommand::Status,
        "SONG_PLAY" => BrainstemCommand::SongPlay {
            id: parse_u32(parts.next()).ok_or(seq)? as u8,
        },
        "SONG_DEFINE" => {
            let id = parse_u32(parts.next()).ok_or(seq)? as u8;
            let mut tones = [SongTone::default(); MAX_SONG_TONES];
            let mut tone_count = 0;
            while tone_count < MAX_SONG_TONES {
                let Some(note) = parts.next() else {
                    break;
                };
                let duration = parts.next().ok_or(seq)?;
                tones[tone_count] = SongTone {
                    note: parse_u32(Some(note)).ok_or(seq)? as u8,
                    duration_64ths: parse_u32(Some(duration)).ok_or(seq)? as u8,
                };
                tone_count += 1;
            }
            if tone_count == 0 {
                return Err(seq);
            }
            BrainstemCommand::SongDefine {
                id,
                tones,
                tone_count: tone_count as u8,
                seq,
            }
        }
        "DOCK" => BrainstemCommand::Dock,
        "SET_LIGHTS" => {
            let led_bits = parse_u32(parts.next()).ok_or(seq)?;
            let color = parse_u32(parts.next()).ok_or(seq)?;
            let intensity = parse_u32(parts.next()).ok_or(seq)?;
            if led_bits > 0x0f || color > u8::MAX as u32 || intensity > u8::MAX as u32 {
                return Err(seq);
            }
            BrainstemCommand::SetLights {
                led_bits: led_bits as u8,
                color: color as u8,
                intensity: intensity as u8,
            }
        }
        _ => return Err(seq),
    };

    if parts.next().is_some() {
        return Err(seq);
    }

    Ok((seq, command))
}

fn handle_compact_control_line<const N: usize>(
    line: &str,
    response: &mut heapless::String<N>,
    transport: u8,
) -> Option<bool> {
    response.clear();
    if let Some(body) = line.strip_prefix("HELLO ") {
        let mut buffer = [0u8; 4096];
        if let Some(welcome) = handle_handshake_json(body, &mut buffer, transport) {
            let prefix = if welcome.contains("\"kind\":\"reject\"") {
                "REJECT "
            } else {
                "WELCOME "
            };
            response.push_str(prefix).ok()?;
            response.push_str(welcome).ok()?;
            response.push('\n').ok()?;
            return Some(false);
        }
        return None;
    }
    if line.starts_with("REGISTER_NETWORK_ENDPOINT ") {
        handle_network_registration_compact(line, response);
        return Some(false);
    }
    if line.starts_with("ACQUIRE_CONTROL_LEASE ") {
        handle_authority_compact(line, response);
        return Some(false);
    }
    if line.starts_with("ACQUIRE_SERVICE_LEASE ") {
        handle_service_authority_compact(line, response);
        return Some(false);
    }
    let (command_line, session_id, lease_id, service_lease_id) = compact_envelope(line);
    let (seq, command) = match parse_forebrain_uart_command(command_line) {
        Ok(parsed) => parsed,
        Err(seq) => {
            let _ = writeln!(response, "ERR {seq} parse");
            return Some(false);
        }
    };

    match command {
        BrainstemCommand::Status | BrainstemCommand::Ping => {
            write_compact_status_line(response, seq);
            Some(false)
        }
        BrainstemCommand::GetCapabilities => {
            let _ = capabilities::write_compact(response, &capabilities::current(), seq);
            Some(false)
        }
        BrainstemCommand::GetEvents { since_seq } => {
            let _ = write!(response, "OK {seq} ");
            let _ = status::write_compact_events(response, since_seq);
            Some(false)
        }
        BrainstemCommand::Bootsel => {
            if compact_service_authority_valid(session_id, service_lease_id, command)
                && cfg!(feature = "service-mode")
            {
                let _ = writeln!(response, "OK {seq} bootsel_accepted");
                Some(true)
            } else {
                let reason = if cfg!(feature = "service-mode") {
                    "service_authorization_required"
                } else {
                    "service_operation_disabled"
                };
                let _ = writeln!(response, "ERR {seq} {reason}");
                Some(false)
            }
        }
        command => {
            if command_requires_session(command) && !session_id.is_some_and(compact_session_valid) {
                let _ = writeln!(response, "ERR {seq} invalid_session");
                return Some(false);
            }
            if command_requires_authority(command)
                && !compact_authority_valid(command, session_id, lease_id)
            {
                let _ = writeln!(response, "ERR {seq} invalid_control_lease");
                return Some(false);
            }
            if command_requires_service_authority(command)
                && !compact_service_authority_valid(session_id, service_lease_id, command)
            {
                let _ = writeln!(response, "ERR {seq} invalid_service_lease");
                return Some(false);
            }
            if submit_compact_control_command(seq, command, session_id, service_lease_id) {
                let _ = writeln!(response, "OK {seq}");
            } else {
                let _ = writeln!(response, "ERR {seq} {}", control_busy_reason());
            }
            Some(false)
        }
    }
}

fn control_busy_reason() -> &'static str {
    let snapshot = status::snapshot(Instant::now().as_millis() as u32);
    if status::charging_interlock_active(&snapshot) {
        "charging_busy"
    } else {
        "busy"
    }
}

fn create_charging_active(snapshot: &status::BrainstemStatus) -> bool {
    snapshot.create_charging_indicator_state == 2
        || matches!(snapshot.create_sensor_charging_state, 1..=3)
}

fn command_requires_session(command: BrainstemCommand) -> bool {
    !matches!(
        command,
        BrainstemCommand::Status
            | BrainstemCommand::Ping
            | BrainstemCommand::GetCapabilities
            | BrainstemCommand::GetEvents { .. }
            | BrainstemCommand::Stop
            | BrainstemCommand::EStop
    )
}
fn command_requires_authority(command: BrainstemCommand) -> bool {
    command_requires_session(command)
        && !matches!(command, BrainstemCommand::Disarm)
        && !matches!(
            command,
            BrainstemCommand::RequestSensors { .. } | BrainstemCommand::StreamSensors { .. }
        )
        && !command_requires_service_authority(command)
}
fn command_requires_service_authority(command: BrainstemCommand) -> bool {
    matches!(
        command,
        BrainstemCommand::Bootsel
            | BrainstemCommand::RestartCreate
            | BrainstemCommand::ResetMotherbrain
    )
}

fn submit_json_control_command(command_id: u32, command: BrainstemCommand, body: &str) -> bool {
    if command_blocked_while_charging(command) {
        return false;
    }
    if command_requires_service_authority(command) {
        let Some((session_hash, lease_hash)) = json_service_identity(body) else {
            return false;
        };
        status::submit_service_control_command(command_id, command, session_hash, lease_hash)
    } else {
        status::submit_control_command(command_id, command)
    }
}

fn submit_compact_control_command(
    command_id: u32,
    command: BrainstemCommand,
    session_id: Option<&str>,
    service_lease_id: Option<&str>,
) -> bool {
    if command_blocked_while_charging(command) {
        return false;
    }
    if command_requires_service_authority(command) {
        let (Some(session_id), Some(lease_id)) = (session_id, service_lease_id) else {
            return false;
        };
        status::submit_service_control_command(
            command_id,
            command,
            session::token_hash(session_id),
            session::token_hash(lease_id),
        )
    } else {
        status::submit_control_command(command_id, command)
    }
}

fn command_blocked_while_charging(command: BrainstemCommand) -> bool {
    if !command_moves_body(command) {
        return false;
    }
    let snapshot = status::snapshot(Instant::now().as_millis() as u32);
    create_charging_active(&snapshot)
}

fn command_moves_body(command: BrainstemCommand) -> bool {
    matches!(
        command,
        BrainstemCommand::CmdVel { .. }
            | BrainstemCommand::DriveDirect { .. }
            | BrainstemCommand::DriveArc { .. }
            | BrainstemCommand::FaceBearing { .. }
            | BrainstemCommand::TrackBearing { .. }
            | BrainstemCommand::TurnBy { .. }
            | BrainstemCommand::DriveFor { .. }
            | BrainstemCommand::BumpEscape { .. }
            | BrainstemCommand::HoldHeading { .. }
            | BrainstemCommand::TurnToHeading { .. }
            | BrainstemCommand::ArcFor { .. }
            | BrainstemCommand::CreepUntil { .. }
            | BrainstemCommand::ScanArc { .. }
            | BrainstemCommand::DockAlign { .. }
            | BrainstemCommand::WallFollow { .. }
            | BrainstemCommand::WiggleAlign { .. }
            | BrainstemCommand::Unstick { .. }
            | BrainstemCommand::CalibrateTurn { .. }
            | BrainstemCommand::Dock
    )
}

fn json_service_identity(body: &str) -> Option<(u32, u32)> {
    Some((
        session::token_hash(json_str(body, "session_id")?),
        session::token_hash(json_str(body, "service_lease_id")?),
    ))
}

fn json_session_valid(body: &str) -> bool {
    json_str(body, "session_id").is_some_and(compact_session_valid)
}
fn json_authority_valid(body: &str) -> bool {
    let Some(session_id) = json_str(body, "session_id") else {
        return false;
    };
    let Some(lease_id) = json_str(body, "lease_id") else {
        return false;
    };
    let session_hash = session::token_hash(session_id);
    let lease_hash = session::token_hash(lease_id);
    let now = Instant::now().as_millis() as u32;
    if json_str(body, "kind") == Some("heartbeat_stop") {
        status::authority_heartbeat_valid(session_hash, lease_hash, now)
    } else {
        status::active_authority_matches(session_hash, lease_hash, now)
    }
}
fn json_service_authority_valid(body: &str) -> bool {
    let Some((session_hash, lease_hash)) = json_service_identity(body) else {
        return false;
    };
    status::active_service_authority_matches(
        session_hash,
        lease_hash,
        Instant::now().as_millis() as u32,
        json_str(body, "kind")
            .and_then(service_scope_code)
            .unwrap_or(0),
    )
}
fn compact_authority_valid(
    command: BrainstemCommand,
    session_id: Option<&str>,
    lease_id: Option<&str>,
) -> bool {
    match (session_id, lease_id) {
        (Some(session_id), Some(lease)) => {
            let session_hash = session::token_hash(session_id);
            let lease_hash = session::token_hash(lease);
            let now = Instant::now().as_millis() as u32;
            if let BrainstemCommand::HeartbeatStop { .. } = command {
                status::authority_heartbeat_valid(session_hash, lease_hash, now)
            } else {
                status::active_authority_matches(session_hash, lease_hash, now)
            }
        }
        _ => false,
    }
}

fn compact_session_valid(session_id: &str) -> bool {
    status::active_session_matches(session::token_hash(session_id))
}

fn compact_session(line: &str) -> (&str, Option<&str>) {
    let (command, session, _, _) = compact_envelope(line);
    (command, session)
}
fn compact_envelope(line: &str) -> (&str, Option<&str>, Option<&str>, Option<&str>) {
    let (without_service, service_lease) = match line.rsplit_once(" service_lease_id=") {
        Some((command, lease)) if !lease.contains(' ') => (command, Some(lease)),
        _ => (line, None),
    };
    let (without_lease, lease) = match without_service.rsplit_once(" lease_id=") {
        Some((command, lease)) if !lease.contains(' ') => (command, Some(lease)),
        _ => (without_service, None),
    };
    match without_lease.rsplit_once(" session_id=") {
        Some((command, session_id)) if !session_id.contains(' ') => {
            (command, Some(session_id), lease, service_lease)
        }
        _ => (without_lease, None, lease, service_lease),
    }
}

fn compact_service_authority_valid(
    session_id: Option<&str>,
    lease_id: Option<&str>,
    command: BrainstemCommand,
) -> bool {
    let scope = match command {
        BrainstemCommand::Bootsel => 1,
        BrainstemCommand::RestartCreate => 3,
        BrainstemCommand::ResetMotherbrain => 4,
        _ => 0,
    };
    match (session_id, lease_id) {
        (Some(session_id), Some(lease_id)) => status::active_service_authority_matches(
            session::token_hash(session_id),
            session::token_hash(lease_id),
            Instant::now().as_millis() as u32,
            scope,
        ),
        _ => false,
    }
}

fn handle_network_registration_json<'a>(body: &str, buffer: &'a mut [u8]) -> Option<&'a str> {
    let session_id = json_str(body, "session_id")?;
    let session_hash = session::token_hash(session_id);
    let identity = status::session_identity(session_hash)?;
    if !compact_session_valid(session_id)
        || identity.role != 1
        || identity.purpose != 1
        || json_str(body, "hostname") != Some("motherbrain")
    {
        return render_registration_reject(buffer, "invalid_session_or_identity");
    }
    let address = parse_ipv4(json_str(body, "address")?)?;
    let lease_identity = json_str(body, "lease_identity")?;
    let ttl = json_u32(body, "ttl_seconds")
        .unwrap_or(60)
        .clamp(1, DHCP_LEASE_SECONDS);
    let Some((ttl, generation)) = network_registry::register_motherbrain(
        lease_identity.as_bytes(),
        address,
        identity.peer_device_hash,
        identity.peer_boot_hash,
        ttl,
        Instant::now().as_millis() as u32,
    ) else {
        return render_registration_reject(buffer, "lease_mismatch");
    };
    let mut response = heapless::String::<512>::new();
    write!(response, "{{\"accepted\":true,\"session_id\":\"{session_id}\",\"fqdn\":\"motherbrain.pete.internal\",\"address\":\"{}.{}.{}.{}\",\"ttl_seconds\":{ttl},\"registration_generation\":{generation}}}", address[0], address[1], address[2], address[3]).ok()?;
    copy_response(buffer, response.as_str())
}

fn render_registration_reject<'a>(buffer: &'a mut [u8], reason: &str) -> Option<&'a str> {
    let mut response = heapless::String::<192>::new();
    write!(response, "{{\"accepted\":false,\"message\":\"{reason}\"}}").ok()?;
    copy_response(buffer, response.as_str())
}

fn handle_network_registration_compact<const N: usize>(
    line: &str,
    response: &mut heapless::String<N>,
) {
    response.clear();
    let (command, session_id) = compact_session(line);
    let mut fields = command.split_ascii_whitespace();
    let valid = fields.next() == Some("REGISTER_NETWORK_ENDPOINT");
    let seq = fields
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0);
    let _interface_id = fields.next();
    let address_text = fields.next();
    let hostname = fields.next();
    let lease_identity = fields.next();
    let ttl = fields
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(60);
    let Some(address) = address_text.and_then(parse_ipv4) else {
        let _ = writeln!(response, "ERR {seq} malformed_registration");
        return;
    };
    if !valid || hostname != Some("motherbrain") || !session_id.is_some_and(compact_session_valid) {
        let _ = writeln!(response, "ERR {seq} invalid_session_or_identity");
        return;
    }
    let session_identity = status::session_identity(session::token_hash(session_id.unwrap_or("")));
    let Some((ttl, generation)) = lease_identity.and_then(|lease_identity| {
        let peer = session_identity?;
        if peer.role != 1 || peer.purpose != 1 {
            return None;
        }
        network_registry::register_motherbrain(
            lease_identity.as_bytes(),
            address,
            peer.peer_device_hash,
            peer.peer_boot_hash,
            ttl,
            Instant::now().as_millis() as u32,
        )
    }) else {
        let _ = writeln!(response, "ERR {seq} lease_mismatch");
        return;
    };
    let _ = writeln!(response, "OK {seq} NETWORK_ENDPOINT_REGISTERED session_id={} fqdn=motherbrain.pete.internal address={}.{}.{}.{} ttl={} generation={}", session_id.unwrap_or(""), address[0], address[1], address[2], address[3], ttl, generation);
}

fn parse_ipv4(value: &str) -> Option<[u8; 4]> {
    let mut octets = [0u8; 4];
    let mut parts = value.split('.');
    for octet in &mut octets {
        *octet = parts.next()?.parse().ok()?;
    }
    if parts.next().is_some() {
        return None;
    }
    Some(octets)
}

fn handle_authority_json<'a>(body: &str, buffer: &'a mut [u8]) -> Option<&'a str> {
    let session_id = json_str(body, "session_id")?;
    let authority = json_str(body, "authority")?;
    let ttl_ms = json_u32(body, "ttl_ms").unwrap_or(2_000).clamp(250, 60_000);
    let session_hash = session::token_hash(session_id);
    if !authority_policy_allows(session_hash, authority, Instant::now().as_millis() as u32) {
        return render_registration_reject(buffer, "authority_policy_rejected");
    }
    let (lease_id, generation) = install_authority(session_hash, ttl_ms)?;
    let mut response = heapless::String::<512>::new();
    write!(response, "{{\"accepted\":true,\"type\":\"control_lease_granted\",\"lease_id\":\"{lease_id}\",\"session_id\":\"{session_id}\",\"owner_role\":\"{}\",\"authority\":\"{authority}\",\"ttl_ms\":{ttl_ms},\"generation\":{generation}}}", role_name(status::session_role(session_hash)?)).ok()?;
    copy_response(buffer, response.as_str())
}

fn handle_service_authority_json<'a>(body: &str, buffer: &'a mut [u8]) -> Option<&'a str> {
    let session_id = json_str(body, "session_id")?;
    let scope = json_str(body, "scope")?;
    let ttl_ms = json_u32(body, "ttl_ms").unwrap_or(2_000).clamp(250, 30_000);
    let session_hash = session::token_hash(session_id);
    let scope_code = service_scope_code(scope)?;
    if !service_policy_allows(session_hash) {
        return render_registration_reject(buffer, "service_policy_rejected");
    }
    let (lease_id, generation) = install_service_authority(session_hash, ttl_ms, scope_code)?;
    let mut response = heapless::String::<512>::new();
    write!(response, "{{\"accepted\":true,\"type\":\"service_lease_granted\",\"lease_id\":\"{lease_id}\",\"session_id\":\"{session_id}\",\"owner_role\":\"{}\",\"scope\":\"{scope}\",\"ttl_ms\":{ttl_ms},\"generation\":{generation}}}", role_name(status::session_role(session_hash)?)).ok()?;
    copy_response(buffer, response.as_str())
}

fn handle_service_authority_compact<const N: usize>(
    line: &str,
    response: &mut heapless::String<N>,
) {
    response.clear();
    let (command, session_id) = compact_session(line);
    let mut fields = command.split_ascii_whitespace();
    let valid = fields.next() == Some("ACQUIRE_SERVICE_LEASE");
    let seq = fields
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0);
    let scope = fields.next().unwrap_or("");
    let ttl_ms = fields
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(2_000)
        .clamp(250, 30_000);
    let Some(session_id) = session_id else {
        let _ = writeln!(response, "ERR {seq} invalid_session");
        return;
    };
    let session_hash = session::token_hash(session_id);
    let Some(scope_code) = service_scope_code(scope) else {
        let _ = writeln!(response, "ERR {seq} service_scope_denied");
        return;
    };
    if !valid || !service_policy_allows(session_hash) {
        let _ = writeln!(response, "ERR {seq} service_policy_rejected");
        return;
    }
    let Some((lease_id, generation)) = install_service_authority(session_hash, ttl_ms, scope_code)
    else {
        let _ = writeln!(response, "ERR {seq} authority_transition_timeout");
        return;
    };
    let _ = writeln!(response, "OK {seq} SERVICE_LEASE_GRANTED lease_id={lease_id} session_id={session_id} owner_role={} scope={scope} ttl_ms={ttl_ms} generation={generation}", role_name(status::session_role(session_hash).unwrap_or(0)));
}

fn service_policy_allows(session_hash: u32) -> bool {
    if !cfg!(feature = "service-mode") {
        return false;
    }
    let Some(identity) = status::session_identity(session_hash) else {
        return false;
    };
    let Some(role) = endpoint_role(identity.role) else {
        return false;
    };
    let purpose = if identity.purpose == 1 {
        session::SessionPurpose::Control
    } else {
        session::SessionPurpose::Diagnostic
    };
    let transport = match identity.transport {
        value if value == TransportKind::UsbCdc as u8 => TransportKind::UsbCdc,
        value if value == TransportKind::HardwareUart as u8 => TransportKind::HardwareUart,
        value if value == TransportKind::Http as u8 => TransportKind::Http,
        value if value == TransportKind::WebSocket as u8 => TransportKind::WebSocket,
        value if value == TransportKind::Udp as u8 => TransportKind::Udp,
        _ => return false,
    };
    pete_cockpit_protocol::role_can_request_service(role, purpose, transport)
}

fn install_service_authority(
    session_hash: u32,
    ttl_ms: u32,
    scope: u8,
) -> Option<(heapless::String<40>, u32)> {
    // Entering service authority uses the same synchronous stop/revoke barrier
    // as a controller transition, but installs a separate non-motion lease.
    let barrier_generation = AUTHORITY_GENERATION
        .fetch_add(1, Ordering::Relaxed)
        .wrapping_add(1)
        .max(1);
    status::request_authority_transition(barrier_generation, 0, 0, 0);
    for _ in 0..250 {
        if status::authority_transition_acked(barrier_generation) {
            break;
        }
        embassy_time::block_for(Duration::from_millis(1));
    }
    if !status::authority_transition_acked(barrier_generation) {
        return None;
    }
    let generation = SERVICE_GENERATION
        .fetch_add(1, Ordering::Relaxed)
        .wrapping_add(1)
        .max(1);
    let mut lease_id = heapless::String::<40>::new();
    write!(lease_id, "service-{generation:08x}-{session_hash:08x}").ok()?;
    status::install_service_authority(
        session_hash,
        session::token_hash(lease_id.as_str()),
        (Instant::now().as_millis() as u32).wrapping_add(ttl_ms),
        scope,
    );
    Some((lease_id, generation))
}

fn handle_authority_compact<const N: usize>(line: &str, response: &mut heapless::String<N>) {
    response.clear();
    let (command, session_id) = compact_session(line);
    let mut fields = command.split_ascii_whitespace();
    let valid = fields.next() == Some("ACQUIRE_CONTROL_LEASE");
    let seq = fields
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0);
    let authority = fields.next().unwrap_or("");
    let ttl_ms = fields
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(2_000)
        .clamp(250, 60_000);
    let Some(session_id) = session_id else {
        let _ = writeln!(response, "ERR {seq} invalid_session");
        return;
    };
    let session_hash = session::token_hash(session_id);
    if !valid
        || !authority_policy_allows(session_hash, authority, Instant::now().as_millis() as u32)
    {
        let _ = writeln!(response, "ERR {seq} authority_policy_rejected");
        return;
    }
    let Some((lease_id, generation)) = install_authority(session_hash, ttl_ms) else {
        let _ = writeln!(response, "ERR {seq} authority_transition_timeout");
        return;
    };
    let _ = writeln!(response, "OK {seq} CONTROL_LEASE_GRANTED lease_id={lease_id} session_id={session_id} owner_role={} authority={authority} ttl_ms={ttl_ms} generation={generation}", role_name(status::session_role(session_hash).unwrap_or(0)));
}

fn authority_policy_allows(session_hash: u32, authority: &str, now_ms: u32) -> bool {
    let Some(identity) = status::session_identity(session_hash) else {
        return false;
    };
    let Some(role) = endpoint_role(identity.role) else {
        return false;
    };
    let purpose = if identity.purpose == 1 {
        session::SessionPurpose::Control
    } else {
        session::SessionPurpose::Diagnostic
    };
    let requested = match authority {
        "motherbrain" => pete_cockpit_protocol::ControlAuthority::Motherbrain,
        "operator_debug" => pete_cockpit_protocol::ControlAuthority::OperatorDebug,
        "forebrain_recovery" => pete_cockpit_protocol::ControlAuthority::ForebrainRecovery,
        _ => return false,
    };
    if !pete_cockpit_protocol::role_can_request_control(role, purpose, requested) {
        return false;
    }
    match requested {
        pete_cockpit_protocol::ControlAuthority::Motherbrain => true,
        pete_cockpit_protocol::ControlAuthority::OperatorDebug => cfg!(feature = "operator-debug"),
        pete_cockpit_protocol::ControlAuthority::ForebrainRecovery => {
            status::authority_expired(now_ms)
                && option_env!("PETE_RECOVERY_FOREBRAIN_ID").is_some_and(|device_id| {
                    status::session_peer_matches(session_hash, session::token_hash(device_id))
                })
        }
    }
}

fn endpoint_role(role: u8) -> Option<session::EndpointRole> {
    match role {
        1 => Some(session::EndpointRole::Motherbrain),
        2 => Some(session::EndpointRole::Forebrain),
        3 => Some(session::EndpointRole::Operator),
        4 => Some(session::EndpointRole::ServiceTool),
        _ => None,
    }
}

fn install_authority(session_hash: u32, ttl_ms: u32) -> Option<(heapless::String<40>, u32)> {
    let generation = AUTHORITY_GENERATION
        .fetch_add(1, Ordering::Relaxed)
        .wrapping_add(1)
        .max(1);
    let mut lease_id = heapless::String::<40>::new();
    write!(lease_id, "lease-{generation:08x}-{session_hash:08x}").ok()?;
    let now = Instant::now().as_millis() as u32;
    status::request_authority_transition(
        generation,
        session::token_hash(lease_id.as_str()),
        session_hash,
        now.wrapping_add(ttl_ms),
    );
    for _ in 0..250 {
        if status::authority_transition_acked(generation) {
            return Some((lease_id, generation));
        }
        embassy_time::block_for(Duration::from_millis(1));
    }
    None
}

fn service_scope_code(scope: &str) -> Option<u8> {
    match scope {
        "bootsel" => Some(1),
        "restart_create" => Some(3),
        "reset_motherbrain" => Some(4),
        _ => None,
    }
}

fn role_name(role: u8) -> &'static str {
    match role {
        1 => "motherbrain",
        2 => "forebrain",
        3 => "operator",
        4 => "service_tool",
        _ => "unknown",
    }
}

fn parse_u32(value: Option<&str>) -> Option<u32> {
    value?.parse().ok()
}

fn parse_i16(value: Option<&str>) -> Option<i16> {
    value?.parse().ok()
}

fn parse_bool(value: Option<&str>) -> Option<bool> {
    match value? {
        "1" | "true" | "TRUE" | "clear" | "CLEAR" | "on" | "ON" | "enable" | "ENABLE" => Some(true),
        "0" | "false" | "FALSE" | "trip" | "TRIP" | "off" | "OFF" | "disable" | "DISABLE" => {
            Some(false)
        }
        _ => None,
    }
}

fn submit_forebrain_stop() {
    let _ = status::submit_control_command(0, BrainstemCommand::Stop);
}

fn write_forebrain_uart_ok(uart: &mut Uart<'static, Blocking>, seq: u32) {
    let mut response = heapless::String::<32>::new();
    let _ = writeln!(response, "OK {seq}");
    write_forebrain_uart_line(uart, response.as_bytes());
}

fn write_forebrain_uart_error(uart: &mut Uart<'static, Blocking>, seq: u32, error: &str) {
    let mut response = heapless::String::<48>::new();
    let _ = writeln!(response, "ERR {seq} {error}");
    write_forebrain_uart_line(uart, response.as_bytes());
}

fn write_forebrain_uart_status(uart: &mut Uart<'static, Blocking>, seq: u32) {
    let mut response = heapless::String::<384>::new();
    write_compact_status_line(&mut response, seq);
    write_forebrain_uart_line(uart, response.as_bytes());
}

fn write_forebrain_uart_capabilities(uart: &mut Uart<'static, Blocking>, seq: u32) {
    let mut response = heapless::String::<2048>::new();
    if capabilities::write_compact(&mut response, &capabilities::current(), seq).is_err() {
        write_forebrain_uart_error(uart, seq, "capabilities_too_large");
        return;
    }
    write_forebrain_uart_line(uart, response.as_bytes());
}

fn write_forebrain_uart_events(uart: &mut Uart<'static, Blocking>, seq: u32, since_seq: u32) {
    let mut response = heapless::String::<1024>::new();
    let _ = write!(response, "OK {seq} ");
    if status::write_compact_events(&mut response, since_seq).is_err() {
        write_forebrain_uart_error(uart, seq, "events_too_large");
        return;
    }
    write_forebrain_uart_line(uart, response.as_bytes());
}

fn write_compact_status_line<const N: usize>(response: &mut heapless::String<N>, seq: u32) {
    let snapshot = status::snapshot(Instant::now().as_millis() as u32);
    let _ = writeln!(
        response,
        "OK {seq} STATUS uptime_ms={} runtime={} body={} action={} command={} pending={} error={} error_uart={} power={} oi={} uart_health={} uart_error={} create_rx_bytes={} create_rx_packets={} create_last_packet_ms={} create_sensor_packet_id={} create_body_packets={} create_last_body_packet_ms={} create_last_packet_len={} create_tx_bytes={} create_last_rx_byte={} create_last_tx_byte={} create_last_rx_ms={} create_last_tx_ms={} create_rx_errors={}/{}/{}/{}/{} wake_probe={}/{} forebrain_rx_bytes={} forebrain_rx_lines={} imu_present={} imu_health={} imu_samples={} imu_age_ms={} imu_poll_ms={} imu_yaw_mrad={} imu_pitch_mrad={} imu_roll_mrad={} imu_yaw_rate_mrad_s={} imu_gyro_x_mrad_s={} imu_gyro_y_mrad_s={} imu_gyro_z_mrad_s={} imu_accel_x_mm_s2={} imu_accel_y_mm_s2={} imu_accel_z_mm_s2={} imu_accel_mag_mm_s2={} imu_tilt_mrad={} imu_roughness_mm_s2={} imu_impact_mm_s2={} imu_motion_consistency={} imu_calibration={}",
        snapshot.uptime_ms,
        snapshot.current_runtime_state,
        snapshot.body_state,
        snapshot.current_runtime_action,
        snapshot.current_command,
        snapshot.pending_command,
        snapshot.last_error,
        snapshot.last_error_uart_read_error,
        snapshot.create_power_state,
        snapshot.oi_mode,
        snapshot.uart_rx_health,
        snapshot.last_uart_read_error,
        snapshot.uart_rx_bytes,
        snapshot.uart_rx_packets,
        snapshot.last_uart_packet_timestamp_ms,
        snapshot.create_sensor_last_packet_id,
        snapshot.create_sensor_complete_packet_count,
        snapshot.create_sensor_last_complete_packet_timestamp_ms,
        snapshot.last_uart_packet_len,
        snapshot.uart_tx_bytes,
        snapshot.last_uart_rx_byte,
        snapshot.last_uart_tx_byte,
        snapshot.last_uart_rx_timestamp_ms,
        snapshot.last_uart_tx_timestamp_ms,
        snapshot.uart_rx_overruns,
        snapshot.uart_rx_breaks,
        snapshot.uart_rx_parity_errors,
        snapshot.uart_rx_framing_errors,
        snapshot.uart_rx_other_errors,
        snapshot.wake_probe_response_bytes,
        snapshot.wake_probe_expected_bytes,
        snapshot.forebrain_uart_rx_bytes,
        snapshot.forebrain_uart_rx_lines,
        snapshot.imu_present,
        snapshot.imu_health,
        snapshot.imu_sample_count,
        snapshot.imu_sample_age_ms,
        snapshot.imu_poll_period_ms,
        snapshot.imu_yaw_mrad,
        snapshot.imu_pitch_mrad,
        snapshot.imu_roll_mrad,
        snapshot.imu_yaw_rate_mrad_s,
        snapshot.imu_gyro_x_mrad_s,
        snapshot.imu_gyro_y_mrad_s,
        snapshot.imu_gyro_z_mrad_s,
        snapshot.imu_accel_x_mm_s2,
        snapshot.imu_accel_y_mm_s2,
        snapshot.imu_accel_z_mm_s2,
        snapshot.imu_accel_magnitude_mm_s2,
        snapshot.imu_tilt_magnitude_mrad,
        snapshot.imu_roughness_mm_s2,
        snapshot.imu_impact_score_mm_s2,
        snapshot.imu_motion_consistency,
        snapshot.imu_calibration_state
    );
}

fn write_forebrain_uart_line(uart: &mut Uart<'static, Blocking>, line: &[u8]) {
    if uart.blocking_write(line).is_err() || uart.blocking_flush().is_err() {
        status::mark_forebrain_uart_error(status::ForebrainUartErrorCode::Uart);
    }
}

fn request_body(request: &[u8]) -> Option<&str> {
    let body_start = request
        .windows(4)
        .position(|w| w == b"\r\n\r\n")?
        .checked_add(4)?;
    core::str::from_utf8(&request[body_start..]).ok()
}

fn parse_command(command_id: u32, body: &str) -> Option<BrainstemCommand> {
    match json_str(body, "kind")? {
        "ping" => Some(BrainstemCommand::Ping),
        "bootsel" => Some(BrainstemCommand::Bootsel),
        "arm" => Some(BrainstemCommand::Arm),
        "set_mode" => Some(BrainstemCommand::SetMode(parse_oi_mode(json_str(
            body, "mode",
        )?)?)),
        "disarm" => Some(BrainstemCommand::Disarm),
        "stop" => Some(BrainstemCommand::Stop),
        "estop" => Some(BrainstemCommand::EStop),
        "clear_estop" => Some(BrainstemCommand::ClearEStop),
        "cmd_vel" => Some(BrainstemCommand::CmdVel {
            linear_mm_s: json_i16(body, "linear_mm_s")?,
            angular_mrad_s: json_i16(body, "angular_mrad_s")?,
            ttl_ms: json_u32(body, "ttl_ms").or_else(|| json_u32(body, "duration_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "drive_direct" => Some(BrainstemCommand::DriveDirect {
            left_mm_s: json_i16(body, "left_mm_s")?,
            right_mm_s: json_i16(body, "right_mm_s")?,
            ttl_ms: json_u32(body, "ttl_ms").or_else(|| json_u32(body, "duration_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "drive_arc" => Some(BrainstemCommand::DriveArc {
            velocity_mm_s: json_i16(body, "velocity_mm_s")?,
            radius_mm: json_i16(body, "radius_mm")?,
            ttl_ms: json_u32(body, "ttl_ms").or_else(|| json_u32(body, "duration_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "face_bearing" => Some(BrainstemCommand::FaceBearing {
            bearing_mrad: json_i16(body, "bearing_mrad")?,
            max_angular_mrad_s: json_i16(body, "max_angular_mrad_s")?,
            tolerance_mrad: json_i16(body, "tolerance_mrad").unwrap_or(35),
            ttl_ms: json_u32(body, "ttl_ms").or_else(|| json_u32(body, "duration_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "track_bearing" => Some(BrainstemCommand::TrackBearing {
            bearing_mrad: json_i16(body, "bearing_mrad")?,
            range_mm: json_u32(body, "range_mm")? as u16,
            max_linear_mm_s: json_i16(body, "max_linear_mm_s")?,
            max_angular_mrad_s: json_i16(body, "max_angular_mrad_s")?,
            stop_range_mm: json_u32(body, "stop_range_mm").unwrap_or(0) as u16,
            ttl_ms: json_u32(body, "ttl_ms").or_else(|| json_u32(body, "duration_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "turn_by" => Some(BrainstemCommand::TurnBy {
            angle_mrad: json_i16(body, "angle_mrad")?,
            angular_mrad_s: json_i16(body, "angular_mrad_s")?,
            timeout_ms: json_u32(body, "timeout_ms").or_else(|| json_u32(body, "duration_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "drive_for" => Some(BrainstemCommand::DriveFor {
            distance_mm: json_i16(body, "distance_mm")?,
            velocity_mm_s: json_i16(body, "velocity_mm_s")?,
            timeout_ms: json_u32(body, "timeout_ms").or_else(|| json_u32(body, "duration_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "bump_escape" => Some(BrainstemCommand::BumpEscape {
            direction: parse_escape_direction(json_str(body, "direction").unwrap_or("either"))?,
            backoff_mm_s: json_i16(body, "backoff_mm_s").unwrap_or(80),
            turn_angular_mrad_s: json_i16(body, "turn_angular_mrad_s").unwrap_or(900),
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "hold_heading" => Some(BrainstemCommand::HoldHeading {
            heading_error_mrad: json_i16(body, "heading_error_mrad")?,
            velocity_mm_s: json_i16(body, "velocity_mm_s")?,
            max_angular_mrad_s: json_i16(body, "max_angular_mrad_s")?,
            ttl_ms: json_u32(body, "ttl_ms").or_else(|| json_u32(body, "duration_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "turn_to_heading" => Some(BrainstemCommand::TurnToHeading {
            heading_error_mrad: json_i16(body, "heading_error_mrad")?,
            angular_mrad_s: json_i16(body, "angular_mrad_s")?,
            tolerance_mrad: json_i16(body, "tolerance_mrad").unwrap_or(35),
            timeout_ms: json_u32(body, "timeout_ms").or_else(|| json_u32(body, "duration_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "arc_for" => Some(BrainstemCommand::ArcFor {
            velocity_mm_s: json_i16(body, "velocity_mm_s")?,
            radius_mm: json_i16(body, "radius_mm")?,
            duration_ms: json_u32(body, "duration_ms").or_else(|| json_u32(body, "ttl_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "creep_until" => Some(BrainstemCommand::CreepUntil {
            velocity_mm_s: json_i16(body, "velocity_mm_s")?,
            angular_mrad_s: json_i16(body, "angular_mrad_s").unwrap_or(0),
            timeout_ms: json_u32(body, "timeout_ms").or_else(|| json_u32(body, "duration_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "scan_arc" => Some(BrainstemCommand::ScanArc {
            angle_mrad: json_i16(body, "angle_mrad")?,
            angular_mrad_s: json_i16(body, "angular_mrad_s")?,
            timeout_ms: json_u32(body, "timeout_ms").or_else(|| json_u32(body, "duration_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "dock_align" => Some(BrainstemCommand::DockAlign {
            bearing_mrad: json_i16(body, "bearing_mrad")?,
            range_mm: json_u32(body, "range_mm")? as u16,
            max_linear_mm_s: json_i16(body, "max_linear_mm_s").unwrap_or(110),
            max_angular_mrad_s: json_i16(body, "max_angular_mrad_s").unwrap_or(650),
            stop_range_mm: json_u32(body, "stop_range_mm").unwrap_or(250) as u16,
            ttl_ms: json_u32(body, "ttl_ms").or_else(|| json_u32(body, "duration_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "wall_follow" => Some(BrainstemCommand::WallFollow {
            distance_error_mm: json_i16(body, "distance_error_mm")?,
            velocity_mm_s: json_i16(body, "velocity_mm_s")?,
            max_angular_mrad_s: json_i16(body, "max_angular_mrad_s")?,
            ttl_ms: json_u32(body, "ttl_ms").or_else(|| json_u32(body, "duration_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "wiggle_align" => Some(BrainstemCommand::WiggleAlign {
            amplitude_mrad: json_i16(body, "amplitude_mrad")?,
            angular_mrad_s: json_i16(body, "angular_mrad_s")?,
            cycles: json_u32(body, "cycles").unwrap_or(2) as u8,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "unstick" => Some(BrainstemCommand::Unstick {
            direction: parse_escape_direction(json_str(body, "direction").unwrap_or("either"))?,
            backoff_mm_s: json_i16(body, "backoff_mm_s").unwrap_or(100),
            turn_angular_mrad_s: json_i16(body, "turn_angular_mrad_s").unwrap_or(1_000),
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "cliff_guard" => Some(BrainstemCommand::CliffGuard {
            clear: json_bool(body, "clear").unwrap_or(false),
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "clear_safety_latch" => Some(BrainstemCommand::ClearSafetyLatch {
            kind: parse_safety_latch_kind(json_str(body, "latch")?)?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "heartbeat_stop" => Some(BrainstemCommand::HeartbeatStop {
            timeout_ms: json_u32(body, "timeout_ms").or_else(|| json_u32(body, "ttl_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "request_sensors" => Some(BrainstemCommand::RequestSensors {
            packet_id: json_u32(body, "packet_id")? as u8,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "stream_sensors" => Some(BrainstemCommand::StreamSensors {
            enabled: json_bool(body, "enabled").unwrap_or(true),
            packet_id: json_u32(body, "packet_id")
                .unwrap_or(body::CREATE_SENSOR_PROBE_PACKET as u32) as u8,
            period_ms: json_u32(body, "period_ms").unwrap_or(250),
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "set_safety_policy" => Some(BrainstemCommand::SetSafetyPolicy {
            policy: SafetyPolicy {
                bump: parse_safety_action(json_str(body, "bump_action").unwrap_or("stop"))?,
                cliff: parse_safety_action(json_str(body, "cliff_action").unwrap_or("stop"))?,
                wheel_drop_latch: json_bool(body, "wheel_drop_latch").unwrap_or(true),
            },
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "clear_motion_queue" => Some(BrainstemCommand::ClearMotionQueue {
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "define_chirp" => {
            let (tones, tone_count) = parse_song_tones(json_str(body, "tones")?)?;
            Some(BrainstemCommand::DefineChirp {
                kind: parse_feedback_kind(json_str(body, "feedback")?)?,
                tones,
                tone_count,
                seq: json_u32(body, "seq").unwrap_or(command_id),
            })
        }
        "play_feedback" => Some(BrainstemCommand::PlayFeedback {
            kind: parse_feedback_kind(json_str(body, "feedback")?)?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "power_state" => Some(BrainstemCommand::PowerState {
            request: parse_power_request(json_str(body, "request")?)?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "create_power_on" => Some(BrainstemCommand::PowerState {
            request: PowerStateRequest::Wake,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "create_power_off" => Some(BrainstemCommand::PowerState {
            request: PowerStateRequest::Sleep,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "calibrate_turn" => Some(BrainstemCommand::CalibrateTurn {
            angular_mrad_s: json_i16(body, "angular_mrad_s")?,
            duration_ms: json_u32(body, "duration_ms")?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "reset_odometry" => Some(BrainstemCommand::ResetOdometry {
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "zero_imu_orientation" => Some(BrainstemCommand::ZeroImuOrientation {
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "clear_imu_orientation" => Some(BrainstemCommand::ClearImuOrientation {
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "restart_create" => Some(BrainstemCommand::RestartCreate),
        "reset_motherbrain" => Some(BrainstemCommand::ResetMotherbrain),
        "get_capabilities" => Some(BrainstemCommand::GetCapabilities),
        "get_events" => Some(BrainstemCommand::GetEvents {
            since_seq: json_u32(body, "since_seq").unwrap_or(0),
        }),
        "status" => Some(BrainstemCommand::Status),
        "song_play" => Some(BrainstemCommand::SongPlay {
            id: json_u32(body, "id")? as u8,
        }),
        "song_define" => {
            let (tones, tone_count) = parse_song_tones(json_str(body, "tones")?)?;
            Some(BrainstemCommand::SongDefine {
                id: json_u32(body, "id")? as u8,
                tones,
                tone_count,
                seq: json_u32(body, "seq").unwrap_or(command_id),
            })
        }
        "dock" => Some(BrainstemCommand::Dock),
        "set_lights" => {
            let led_bits = json_u32(body, "led_bits")?;
            let color = json_u32(body, "color")?;
            let intensity = json_u32(body, "intensity")?;
            if led_bits > 0x0f || color > u8::MAX as u32 || intensity > u8::MAX as u32 {
                return None;
            }
            Some(BrainstemCommand::SetLights {
                led_bits: led_bits as u8,
                color: color as u8,
                intensity: intensity as u8,
            })
        }
        _ => None,
    }
}

fn parse_escape_direction(direction: &str) -> Option<EscapeDirection> {
    match direction {
        "left" | "LEFT" => Some(EscapeDirection::Left),
        "right" | "RIGHT" => Some(EscapeDirection::Right),
        "either" | "EITHER" => Some(EscapeDirection::Either),
        _ => None,
    }
}

fn parse_safety_action(action: &str) -> Option<SafetyAction> {
    match action {
        "none" | "NONE" => Some(SafetyAction::None),
        "stop" | "STOP" => Some(SafetyAction::Stop),
        "backoff" | "BACKOFF" => Some(SafetyAction::Backoff),
        "bump_escape" | "BUMP_ESCAPE" | "escape" | "ESCAPE" => Some(SafetyAction::BumpEscape),
        _ => None,
    }
}

fn parse_safety_latch_kind(kind: &str) -> Option<SafetyLatchKind> {
    match kind {
        "bump" | "BUMP" => Some(SafetyLatchKind::Bump),
        "cliff" | "CLIFF" => Some(SafetyLatchKind::Cliff),
        "wheel_drop" | "WHEEL_DROP" => Some(SafetyLatchKind::WheelDrop),
        "heartbeat" | "HEARTBEAT" => Some(SafetyLatchKind::Heartbeat),
        "tilt" | "TILT" => Some(SafetyLatchKind::Tilt),
        "impact" | "IMPACT" => Some(SafetyLatchKind::Impact),
        "charging" | "CHARGING" => Some(SafetyLatchKind::Charging),
        _ => None,
    }
}

fn parse_feedback_kind(kind: &str) -> Option<FeedbackKind> {
    match kind {
        "ok" | "OK" => Some(FeedbackKind::Ok),
        "error" | "ERROR" => Some(FeedbackKind::Error),
        "armed" | "ARMED" => Some(FeedbackKind::Armed),
        "lost_target" | "LOST_TARGET" => Some(FeedbackKind::LostTarget),
        "dock_seen" | "DOCK_SEEN" => Some(FeedbackKind::DockSeen),
        "danger" | "DANGER" => Some(FeedbackKind::Danger),
        _ => None,
    }
}

fn parse_power_request(request: &str) -> Option<PowerStateRequest> {
    match request {
        "wake" | "WAKE" => Some(PowerStateRequest::Wake),
        "sleep" | "SLEEP" => Some(PowerStateRequest::Sleep),
        "pulse_brc" | "PULSE_BRC" => Some(PowerStateRequest::PulseBrc),
        "start_oi" | "START_OI" => Some(PowerStateRequest::StartOi),
        "debug_baud_19200" | "DEBUG_BAUD_19200" => Some(PowerStateRequest::DebugBaud19200),
        "debug_baud_57600" | "DEBUG_BAUD_57600" => Some(PowerStateRequest::DebugBaud57600),
        "debug_baud_115200" | "DEBUG_BAUD_115200" => Some(PowerStateRequest::DebugBaud115200),
        _ => None,
    }
}

fn parse_song_tones(value: &str) -> Option<([SongTone; MAX_SONG_TONES], u8)> {
    let mut tones = [SongTone::default(); MAX_SONG_TONES];
    let mut tone_count = 0usize;
    for pair in value.split(',') {
        if tone_count >= MAX_SONG_TONES {
            return None;
        }
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let split = pair.find(':')?;
        let note = pair[..split].trim().parse::<u8>().ok()?;
        let duration_64ths = pair[split + 1..].trim().parse::<u8>().ok()?;
        tones[tone_count] = SongTone {
            note,
            duration_64ths,
        };
        tone_count += 1;
    }
    if tone_count == 0 {
        None
    } else {
        Some((tones, tone_count as u8))
    }
}

fn parse_oi_mode(mode: &str) -> Option<CreateOiMode> {
    match mode {
        "passive" | "PASSIVE" => Some(CreateOiMode::Passive),
        "safe" | "SAFE" => Some(CreateOiMode::Safe),
        "full" | "FULL" => Some(CreateOiMode::Full),
        _ => None,
    }
}

fn render_command_response<'a>(
    buffer: &'a mut [u8],
    accepted: bool,
    command_id: u32,
    message: &str,
) -> Option<&'a str> {
    let mut response = heapless::String::<128>::new();
    let _ = write!(
        response,
        "{{\"accepted\":{},\"command_id\":{},\"message\":\"{}\"}}\n",
        if accepted { "true" } else { "false" },
        command_id,
        message
    );
    let bytes = response.as_bytes();
    if bytes.len() > buffer.len() {
        return None;
    }
    buffer[..bytes.len()].copy_from_slice(bytes);
    core::str::from_utf8(&buffer[..bytes.len()]).ok()
}

fn render_capabilities_response(buffer: &mut [u8], command_id: u32) -> Option<&str> {
    capabilities::render_json(&capabilities::current(), command_id, buffer)
}

fn websocket_key(request: &[u8]) -> Option<&str> {
    let request = core::str::from_utf8(request).ok()?;
    for line in request.split("\r\n") {
        if let Some(value) = line.strip_prefix("Sec-WebSocket-Key:") {
            return Some(value.trim());
        }
    }
    None
}

fn websocket_accept_key<'a>(key: &str, buffer: &'a mut [u8]) -> Option<&'a str> {
    const GUID: &[u8] = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    let mut sha1 = Sha1::new();
    sha1.update(key.as_bytes());
    sha1.update(GUID);
    let digest = sha1.finalize();
    let len = base64_encode(&digest, buffer)?;
    core::str::from_utf8(&buffer[..len]).ok()
}

async fn write_websocket_upgrade(
    socket: &mut TcpSocket<'_>,
    accept_key: &str,
) -> Result<(), embassy_net::tcp::Error> {
    let mut header = heapless::String::<192>::new();
    let _ = write!(
        header,
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {}\r\n\r\n",
        accept_key
    );
    socket.write_all(header.as_bytes()).await?;
    flush_tcp_with_timeout(socket).await.map(|_| ())
}

async fn read_websocket_text<'a>(
    socket: &mut TcpSocket<'_>,
    payload: &'a mut [u8],
) -> Result<Option<&'a str>, embassy_net::tcp::Error> {
    let mut header = [0; 2];
    read_exact_tcp(socket, &mut header).await?;

    let opcode = header[0] & 0x0f;
    if opcode == 0x08 {
        return Ok(None);
    }
    if opcode != 0x01 {
        return Ok(Some(""));
    }

    let masked = header[1] & 0x80 != 0;
    let len = (header[1] & 0x7f) as usize;
    if !masked || len > payload.len() || len == 126 || len == 127 {
        return Ok(Some(""));
    }

    let mut mask = [0; 4];
    read_exact_tcp(socket, &mut mask).await?;
    read_exact_tcp(socket, &mut payload[..len]).await?;
    for i in 0..len {
        payload[i] ^= mask[i & 3];
    }

    Ok(core::str::from_utf8(&payload[..len]).ok())
}

async fn write_websocket_text(
    socket: &mut TcpSocket<'_>,
    payload: &[u8],
) -> Result<(), embassy_net::tcp::Error> {
    if payload.len() <= 125 {
        let header = [0x81, payload.len() as u8];
        socket.write_all(&header).await?;
    } else if payload.len() <= u16::MAX as usize {
        let len = payload.len() as u16;
        let header = [0x81, 126, (len >> 8) as u8, len as u8];
        socket.write_all(&header).await?;
    } else {
        return Ok(());
    }
    socket.write_all(payload).await?;
    flush_tcp_with_timeout(socket).await.map(|_| ())
}

async fn read_exact_tcp(
    socket: &mut TcpSocket<'_>,
    mut buffer: &mut [u8],
) -> Result<(), embassy_net::tcp::Error> {
    while !buffer.is_empty() {
        let n = socket.read(buffer).await?;
        if n == 0 {
            return Err(embassy_net::tcp::Error::ConnectionReset);
        }
        let tmp = buffer;
        buffer = &mut tmp[n..];
    }
    Ok(())
}

fn base64_encode(input: &[u8], output: &mut [u8]) -> Option<usize> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let output_len = input.len().div_ceil(3) * 4;
    if output_len > output.len() {
        return None;
    }

    let mut i = 0;
    let mut j = 0;
    while i < input.len() {
        let b0 = input[i];
        let b1 = if i + 1 < input.len() { input[i + 1] } else { 0 };
        let b2 = if i + 2 < input.len() { input[i + 2] } else { 0 };
        output[j] = TABLE[(b0 >> 2) as usize];
        output[j + 1] = TABLE[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize];
        output[j + 2] = if i + 1 < input.len() {
            TABLE[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize]
        } else {
            b'='
        };
        output[j + 3] = if i + 2 < input.len() {
            TABLE[(b2 & 0x3f) as usize]
        } else {
            b'='
        };
        i += 3;
        j += 4;
    }
    Some(output_len)
}

struct Sha1 {
    state: [u32; 5],
    len_bytes: u64,
    buffer: [u8; 64],
    buffer_len: usize,
}

impl Sha1 {
    fn new() -> Self {
        Self {
            state: [
                0x6745_2301,
                0xefcd_ab89,
                0x98ba_dcfe,
                0x1032_5476,
                0xc3d2_e1f0,
            ],
            len_bytes: 0,
            buffer: [0; 64],
            buffer_len: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.len_bytes = self.len_bytes.saturating_add(input.len() as u64);

        if self.buffer_len > 0 {
            let copy_len = (64 - self.buffer_len).min(input.len());
            self.buffer[self.buffer_len..self.buffer_len + copy_len]
                .copy_from_slice(&input[..copy_len]);
            self.buffer_len += copy_len;
            input = &input[copy_len..];
            if self.buffer_len == 64 {
                let block = self.buffer;
                self.process_block(&block);
                self.buffer_len = 0;
            }
        }

        while input.len() >= 64 {
            let mut block = [0; 64];
            block.copy_from_slice(&input[..64]);
            self.process_block(&block);
            input = &input[64..];
        }

        if !input.is_empty() {
            self.buffer[..input.len()].copy_from_slice(input);
            self.buffer_len = input.len();
        }
    }

    fn finalize(mut self) -> [u8; 20] {
        let bit_len = self.len_bytes.saturating_mul(8);
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;

        if self.buffer_len > 56 {
            for byte in &mut self.buffer[self.buffer_len..] {
                *byte = 0;
            }
            let block = self.buffer;
            self.process_block(&block);
            self.buffer_len = 0;
        }

        for byte in &mut self.buffer[self.buffer_len..56] {
            *byte = 0;
        }
        self.buffer[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buffer;
        self.process_block(&block);

        let mut out = [0; 20];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn process_block(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let mut a = self.state[0];
        let mut b = self.state[1];
        let mut c = self.state[2];
        let mut d = self.state[3];
        let mut e = self.state[4];

        for (i, word) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5a82_7999),
                20..=39 => (b ^ c ^ d, 0x6ed9_eba1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8f1b_bcdc),
                _ => (b ^ c ^ d, 0xca62_c1d6),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
    }
}

fn json_str<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    let value = json_value(body, key)?.strip_prefix('"')?;
    let end = value.find('"')?;
    Some(&value[..end])
}

fn json_value<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    let bytes = body.as_bytes();
    for (start, _) in body.match_indices(key) {
        let end = start.checked_add(key.len())?;
        if start == 0 || end >= bytes.len() || bytes[start - 1] != b'"' || bytes[end] != b'"' {
            continue;
        }
        let after_key = body[end + 1..].trim_start();
        return Some(after_key.strip_prefix(':')?.trim_start());
    }
    None
}

fn json_u32(body: &str, key: &str) -> Option<u32> {
    json_i32(body, key).and_then(|value| u32::try_from(value).ok())
}

fn json_i16(body: &str, key: &str) -> Option<i16> {
    json_i32(body, key).and_then(|value| i16::try_from(value).ok())
}

fn json_bool(body: &str, key: &str) -> Option<bool> {
    let value = json_value(body, key)?;
    if value.starts_with("true") {
        Some(true)
    } else if value.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn json_i32(body: &str, key: &str) -> Option<i32> {
    let value = json_value(body, key)?;
    let end = value
        .find(|c: char| !(c == '-' || c.is_ascii_digit()))
        .unwrap_or(value.len());
    value[..end].parse().ok()
}

fn build_mdns_announcement(packet: &mut [u8; 768]) -> usize {
    packet.fill(0);
    packet[2..4].copy_from_slice(&[0x84, 0x00]);
    packet[6..8].copy_from_slice(&9u16.to_be_bytes());
    let mut i = 12;
    i = mdns_a(packet, i, MDNS_NAME, AP_IP_OCTETS).unwrap_or(12);
    for (service, port) in [
        (
            b"\x0f_pete-brainstem\x04_tcp\x05local\x00".as_slice(),
            HTTP_PORT,
        ),
        (
            b"\x0b_pete-debug\x04_tcp\x05local\x00".as_slice(),
            HTTP_PORT,
        ),
        (
            b"\x0d_pete-control\x04_tcp\x05local\x00".as_slice(),
            WS_CONTROL_PORT,
        ),
        (
            b"\x0d_pete-control\x04_udp\x05local\x00".as_slice(),
            UDP_CONTROL_PORT,
        ),
    ] {
        let mut instance = heapless::Vec::<u8, 64>::new();
        let _ = instance.extend_from_slice(b"\x09brainstem");
        let _ = instance.extend_from_slice(service);
        i = mdns_ptr(packet, i, service, &instance).unwrap_or(i);
        i = mdns_srv(packet, i, &instance, port, MDNS_NAME).unwrap_or(i);
    }
    i
}

fn mdns_a(packet: &mut [u8], mut i: usize, name: &[u8], ip: [u8; 4]) -> Option<usize> {
    i = put_bytes(packet, i, name)?;
    i = put_bytes(packet, i, &[0, 1, 0x80, 1, 0, 0, 0, 120, 0, 4])?;
    put_bytes(packet, i, &ip)
}
fn mdns_ptr(packet: &mut [u8], mut i: usize, name: &[u8], target: &[u8]) -> Option<usize> {
    i = put_bytes(packet, i, name)?;
    i = put_bytes(packet, i, &[0, 12, 0, 1, 0, 0, 0, 120])?;
    i = put_bytes(packet, i, &(target.len() as u16).to_be_bytes())?;
    put_bytes(packet, i, target)
}
fn mdns_srv(
    packet: &mut [u8],
    mut i: usize,
    name: &[u8],
    port: u16,
    target: &[u8],
) -> Option<usize> {
    i = put_bytes(packet, i, name)?;
    i = put_bytes(packet, i, &[0, 33, 0x80, 1, 0, 0, 0, 120])?;
    i = put_bytes(packet, i, &((6 + target.len()) as u16).to_be_bytes())?;
    i = put_bytes(packet, i, &[0, 0, 0, 0])?;
    i = put_bytes(packet, i, &port.to_be_bytes())?;
    put_bytes(packet, i, target)
}
fn put_bytes(packet: &mut [u8], offset: usize, bytes: &[u8]) -> Option<usize> {
    let end = offset.checked_add(bytes.len())?;
    packet.get_mut(offset..end)?.copy_from_slice(bytes);
    Some(end)
}

fn build_dns_reply<'a>(query: &[u8], response: &'a mut [u8; 512]) -> Option<&'a [u8]> {
    let question = parse_dns_question(query)?;
    let answer_ip = dns_answer_ip(
        &query[12..question.name_end],
        Instant::now().as_millis() as u32,
    )?;
    if !matches!(question.qtype, 1 | 255) || !matches!(question.qclass, 1 | 255) {
        return None;
    }

    response[..question.end].copy_from_slice(&query[..question.end]);
    response[2] = 0x84 | (query[2] & 0x01); // response, authoritative, preserve RD
    response[3] = 0x00; // no error
    response[4] = 0x00;
    response[5] = 0x01; // echo only the first question
    response[6] = 0x00;
    response[7] = 0x01; // one answer
    response[8] = 0x00;
    response[9] = 0x00;
    response[10] = 0x00;
    response[11] = 0x00;

    let mut i = question.end;
    let answer = [
        0xc0,
        0x0c, // compressed name pointer to the original question name
        0x00,
        0x01, // A
        0x00,
        0x01, // IN
        0x00,
        0x00,
        0x00,
        0x3c, // TTL 60s
        0x00,
        0x04, // IPv4 length
        answer_ip[0],
        answer_ip[1],
        answer_ip[2],
        answer_ip[3],
    ];
    if i + answer.len() > response.len() {
        return None;
    }
    response[i..i + answer.len()].copy_from_slice(&answer);
    i += answer.len();
    Some(&response[..i])
}

struct DnsQuestion {
    name_end: usize,
    end: usize,
    qtype: u16,
    qclass: u16,
}

fn parse_dns_question(packet: &[u8]) -> Option<DnsQuestion> {
    if packet.len() < 17 || packet[2] & 0x80 != 0 {
        return None;
    }
    let question_count = u16::from_be_bytes([packet[4], packet[5]]);
    if question_count == 0 {
        return None;
    }

    let mut i = 12;
    loop {
        let len = *packet.get(i)? as usize;
        if len & 0xc0 != 0 {
            return None;
        }
        i += 1;
        if len == 0 {
            break;
        }
        i = i.checked_add(len)?;
        if i > packet.len() {
            return None;
        }
    }

    let name_end = i;
    let end = i.checked_add(4)?;
    if end > packet.len() {
        return None;
    }
    Some(DnsQuestion {
        name_end,
        end,
        qtype: u16::from_be_bytes([packet[i], packet[i + 1]]),
        qclass: u16::from_be_bytes([packet[i + 2], packet[i + 3]]),
    })
}

fn dns_answer_ip(name: &[u8], now_ms: u32) -> Option<[u8; 4]> {
    const PETE_INTERNAL: &[u8] = b"\x04pete\x08internal\x00";
    const BRAINSTEM_INTERNAL: &[u8] = b"\x09brainstem\x04pete\x08internal\x00";
    const GATEWAY_INTERNAL: &[u8] = b"\x07gateway\x04pete\x08internal\x00";
    const MOTHERBRAIN_INTERNAL: &[u8] = b"\x0bmotherbrain\x04pete\x08internal\x00";
    if dns_name_eq(name, PETE_INTERNAL)
        || dns_name_eq(name, BRAINSTEM_INTERNAL)
        || dns_name_eq(name, GATEWAY_INTERNAL)
        || dns_name_eq(name, MDNS_NAME)
    {
        return Some(AP_IP_OCTETS);
    }
    if dns_name_eq(name, MOTHERBRAIN_INTERNAL) {
        return network_registry::resolve_motherbrain(now_ms);
    }
    None
}

fn dns_name_eq(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| dns_byte_eq(*left, *right))
}

fn dns_byte_eq(left: u8, right: u8) -> bool {
    if left.is_ascii_alphabetic() && right.is_ascii_alphabetic() {
        left.to_ascii_lowercase() == right.to_ascii_lowercase()
    } else {
        left == right
    }
}

fn build_dhcp_reply<'a>(
    grant: DhcpGrant,
    request: &[u8],
    response: &'a mut [u8; 576],
) -> Option<&'a [u8]> {
    response.fill(0);
    response[0] = 2;
    response[1] = request[1];
    response[2] = request[2];
    response[3] = request[3];
    response[4..8].copy_from_slice(&request[4..8]);
    response[10..12].copy_from_slice(&request[10..12]);
    response[16..20].copy_from_slice(&grant.lease_ip());
    response[20..24].copy_from_slice(&AP_IP_OCTETS);
    response[28..44].copy_from_slice(&request[28..44]);
    response[236..240].copy_from_slice(&[99, 130, 83, 99]);

    let mut i = 240;
    i = write_dhcp_option(i, response, 53, &[grant.reply_message_type()])?;
    i = write_dhcp_option(i, response, 54, &AP_IP_OCTETS)?;
    i = write_dhcp_option(i, response, 51, &DHCP_LEASE_SECONDS.to_be_bytes())?;
    i = write_dhcp_option(i, response, 1, &[255, 255, 255, 0])?;
    i = write_dhcp_option(i, response, 3, &AP_IP_OCTETS)?;
    i = write_dhcp_option(i, response, 6, &AP_IP_OCTETS)?;
    response[i] = 255;
    Some(&response[..i + 1])
}

impl DhcpRequest {
    fn parse(packet: &[u8]) -> Option<Self> {
        if packet.len() < 240 || packet[0] != 1 || packet[1] != 1 || packet[2] < 6 {
            return None;
        }

        let mut hardware_address = [0; 6];
        hardware_address.copy_from_slice(&packet[28..34]);

        let client_identifier = dhcp_option(packet, 61).unwrap_or(&[]);
        let requested_hostname = dhcp_option(packet, 12).unwrap_or(&[]);
        Some(Self::new(
            dhcp_message_type(packet)?,
            DhcpClient::new(hardware_address).with_metadata(client_identifier, requested_hostname),
        ))
    }
}

fn dhcp_message_type(packet: &[u8]) -> Option<u8> {
    dhcp_option(packet, 53).and_then(|value| (value.len() == 1).then_some(value[0]))
}

fn dhcp_option(packet: &[u8], wanted: u8) -> Option<&[u8]> {
    if packet.len() < 240 || packet[236..240] != [99, 130, 83, 99] {
        return None;
    }

    let mut i = 240;
    while i < packet.len() {
        let option = packet[i];
        i += 1;
        match option {
            0 => continue,
            255 => return None,
            _ => {
                if i >= packet.len() {
                    return None;
                }
                let len = packet[i] as usize;
                i += 1;
                if i + len > packet.len() {
                    return None;
                }
                if option == wanted {
                    return Some(&packet[i..i + len]);
                }
                i += len;
            }
        }
    }
    None
}

fn write_dhcp_option(
    offset: usize,
    packet: &mut [u8; 576],
    option: u8,
    value: &[u8],
) -> Option<usize> {
    let end = offset.checked_add(2)?.checked_add(value.len())?;
    if end >= packet.len() || value.len() > u8::MAX as usize {
        return None;
    }
    packet[offset] = option;
    packet[offset + 1] = value.len() as u8;
    packet[offset + 2..end].copy_from_slice(value);
    Some(end)
}

fn level(high: bool) -> Level {
    if high {
        Level::High
    } else {
        Level::Low
    }
}
