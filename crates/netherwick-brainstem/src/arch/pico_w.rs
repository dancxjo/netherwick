use core::fmt::Write as _;

use cyw43::aligned_bytes;
use cyw43_pio::{DEFAULT_CLOCK_DIVIDER, PioSpi};
use embassy_executor::Spawner;
use embassy_net::tcp::TcpSocket;
use embassy_net::udp::{PacketMetadata, UdpSocket};
use embassy_net::{
    Config as NetConfig, IpAddress, IpEndpoint, Ipv4Address, Ipv4Cidr, Stack, StackResources,
};
use embassy_rp::gpio::{Level, Output};
use embassy_rp::multicore::{spawn_core1, Stack as CoreStack};
use embassy_rp::peripherals::{
    DMA_CH0, PIO0, PIN_16, PIN_17, PIN_18, PIN_19, PIN_20, PIN_23, PIN_24, PIN_25, PIN_29, UART0,
};
use embassy_rp::pio::{InterruptHandler, Pio};
use embassy_rp::uart::{Blocking, Config as UartConfig, DataBits, Parity, StopBits, Uart};
use embassy_rp::{bind_interrupts, dma, Peri};
use embassy_time::{Duration, Instant, Timer};
use embedded_hal_nb::serial::Read as _;
use embedded_io_async::Write;
use static_cell::StaticCell;

use crate::body;
use crate::hardware::{BrainstemHardware, SerialRead};
use crate::runtime::Runtime;
use crate::status;

const AP_SSID: &str = "pete-brainstem";
const MDNS_NAME: &[u8] = b"\x04pete\x05local\x00";
const AP_CHANNEL: u8 = 6;
const AP_IP: Ipv4Address = Ipv4Address::new(192, 168, 4, 1);
const HTTP_PORT: u16 = 80;
const MDNS_PORT: u16 = 5353;

static mut CORE1_STACK: CoreStack<8192> = CoreStack::new();

bind_interrupts!(struct Irqs {
    PIO0_IRQ_0 => InterruptHandler<PIO0>;
    DMA_IRQ_0 => dma::InterruptHandler<DMA_CH0>;
});

pub struct PicoWBrainstem {
    uart: Uart<'static, Blocking>,
    power_toggle: Output<'static>,
    brc: Output<'static>,
    status_led: Output<'static>,
}

impl PicoWBrainstem {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        uart0: Peri<'static, UART0>,
        tx: Peri<'static, PIN_16>,
        rx: Peri<'static, PIN_17>,
        power_toggle: Peri<'static, PIN_18>,
        brc: Peri<'static, PIN_19>,
        status_led: Peri<'static, PIN_20>,
    ) -> Self {
        let mut uart_config = UartConfig::default();
        uart_config.baudrate = body::CREATE_UART_BAUD;
        uart_config.data_bits = DataBits::DataBits8;
        uart_config.stop_bits = StopBits::STOP1;
        uart_config.parity = Parity::ParityNone;

        Self {
            uart: Uart::new_blocking(uart0, tx, rx, uart_config),
            power_toggle: Output::new(power_toggle, Level::Low),
            brc: Output::new(brc, Level::High),
            status_led: Output::new(status_led, Level::Low),
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

    fn set_power_toggle(&mut self, high: bool) {
        self.power_toggle.set_level(level(high));
    }

    fn set_brc(&mut self, high: bool) {
        self.brc.set_level(level(high));
    }

    fn set_indicators(&mut self, on: bool) {
        self.status_led.set_level(level(on));
    }

    fn set_primary_indicator(&mut self, on: bool) {
        self.status_led.set_level(level(on));
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
            Err(nb::Error::Other(_)) => SerialRead::Error,
        }
    }
}

pub fn spawn_safety_lane(p: embassy_rp::Peripherals) -> ! {
    let hardware = PicoWBrainstem::new(
        p.UART0, p.PIN_16, p.PIN_17, p.PIN_18, p.PIN_19, p.PIN_20,
    );

    spawn_core1(
        p.CORE1,
        unsafe { &mut *core::ptr::addr_of_mut!(CORE1_STACK) },
        move || Runtime::new(hardware).run_demo(),
    );

    spawn_wifi_lane(
        p.PIO0, p.DMA_CH0, p.PIN_23, p.PIN_24, p.PIN_25, p.PIN_29,
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
) -> ! {
    static EXECUTOR: StaticCell<embassy_executor::Executor> = StaticCell::new();
    let executor = EXECUTOR.init(embassy_executor::Executor::new());
    executor.run(|spawner| {
        spawner.spawn(wifi_task(
            spawner, pio0, dma0, wifi_power, wifi_dio, wifi_cs, wifi_clk,
        ).expect("spawn wifi task"));
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
    if let Some((stack, mut control)) =
        start_wifi_ap(spawner, pio0, dma0, wifi_power, wifi_dio, wifi_cs, wifi_clk).await
    {
        let _ = control.gpio_set(0, true).await;
        spawner.spawn(http_task(stack).expect("spawn http task"));
        spawner.spawn(mdns_task(stack).expect("spawn mdns task"));
    }

    loop {
        Timer::after_secs(60).await;
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
        .set_power_management(cyw43::PowerManagementMode::PowerSave)
        .await;

    let config = NetConfig::ipv4_static(embassy_net::StaticConfigV4 {
        address: Ipv4Cidr::new(AP_IP, 24),
        dns_servers: Default::default(),
        gateway: None,
    });

    static RESOURCES: StaticCell<StackResources<4>> = StaticCell::new();
    let (stack, runner) = embassy_net::new(net_device, config, RESOURCES.init(StackResources::new()), 0x5eed);
    let _ = stack.join_multicast_group(IpAddress::Ipv4(Ipv4Address::new(224, 0, 0, 251)));
    spawner.spawn(net_runner_task(runner).expect("spawn net runner"));

    control.start_ap_open(AP_SSID, AP_CHANNEL).await;
    Some((stack, control))
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
async fn http_task(stack: Stack<'static>) -> ! {
    let mut rx_buffer = [0; 1024];
    let mut tx_buffer = [0; 2048];
    let mut request = [0; 512];
    let mut json = [0; 768];

    loop {
        let mut socket = TcpSocket::new(stack, &mut rx_buffer, &mut tx_buffer);
        socket.set_timeout(Some(Duration::from_secs(4)));

        if socket.accept(HTTP_PORT).await.is_err() {
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
        match path {
            Some("/") | Some("/index.html") => {
                let body = index_html();
                write_response(&mut socket, "text/html; charset=utf-8", body.as_bytes()).await;
            }
            Some("/status.json") => {
                let uptime_ms = Instant::now().as_millis() as u32;
                let snapshot = status::snapshot(uptime_ms);
                match status::render_json(snapshot, &mut json) {
                    Ok(body) => {
                        write_response(&mut socket, "application/json", body.as_bytes()).await;
                    }
                    Err(_) => {
                        write_plain_status(&mut socket, 500, "Internal Server Error").await;
                    }
                }
            }
            _ => {
                write_plain_status(&mut socket, 404, "Not Found").await;
            }
        }

        socket.close();
    }
}

#[embassy_executor::task]
async fn mdns_task(stack: Stack<'static>) -> ! {
    let mut rx_meta = [PacketMetadata::EMPTY; 2];
    let mut rx_buffer = [0; 256];
    let mut tx_meta = [PacketMetadata::EMPTY; 2];
    let mut tx_buffer = [0; 256];
    let mut packet = [0; 96];
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

async fn write_response(socket: &mut TcpSocket<'_>, content_type: &str, body: &[u8]) {
    let mut header = heapless::String::<192>::new();
    let _ = write!(
        header,
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        content_type,
        body.len()
    );
    let _ = socket.write_all(header.as_bytes()).await;
    let _ = socket.write_all(body).await;
}

async fn write_plain_status(socket: &mut TcpSocket<'_>, code: u16, text: &str) {
    let mut header = heapless::String::<160>::new();
    let _ = write!(
        header,
        "HTTP/1.1 {} {}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        code,
        text,
        text.len(),
        text
    );
    let _ = socket.write_all(header.as_bytes()).await;
}

fn request_path(request: &[u8]) -> Option<&str> {
    let line_end = request.windows(2).position(|w| w == b"\r\n").unwrap_or(request.len());
    let line = core::str::from_utf8(&request[..line_end]).ok()?;
    let mut parts = line.split(' ');
    match (parts.next(), parts.next()) {
        (Some("GET"), Some(path)) => Some(path),
        _ => None,
    }
}

fn index_html() -> &'static str {
    "<!doctype html><html><head><meta charset=\"utf-8\"><title>Pete Brainstem</title></head><body><h1>Pete Brainstem</h1><p>Hostname: pete.local</p><p><a href=\"/status.json\">status.json</a></p></body></html>"
}

fn build_mdns_announcement(packet: &mut [u8; 96]) -> usize {
    let mut i = 0;
    let header = [
        0x00, 0x00, // transaction id
        0x84, 0x00, // response, authoritative answer
        0x00, 0x00, // questions
        0x00, 0x01, // answers
        0x00, 0x00, // authority records
        0x00, 0x00, // additional records
    ];
    packet[i..i + header.len()].copy_from_slice(&header);
    i += header.len();
    packet[i..i + MDNS_NAME.len()].copy_from_slice(MDNS_NAME);
    i += MDNS_NAME.len();
    let answer = [
        0x00, 0x01, // A
        0x80, 0x01, // IN, cache flush
        0x00, 0x00, 0x00, 0x78, // TTL 120s
        0x00, 0x04, // IPv4 length
        192, 168, 4, 1,
    ];
    packet[i..i + answer.len()].copy_from_slice(&answer);
    i + answer.len()
}

fn level(high: bool) -> Level {
    if high {
        Level::High
    } else {
        Level::Low
    }
}
