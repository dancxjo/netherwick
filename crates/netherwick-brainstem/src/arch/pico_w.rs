use core::fmt::Write as _;

use cyw43::aligned_bytes;
use cyw43_pio::{PioSpi, DEFAULT_CLOCK_DIVIDER};
use embassy_executor::Spawner;
use embassy_futures::select::{select, Either};
use embassy_net::tcp::TcpSocket;
use embassy_net::udp::{PacketMetadata, UdpSocket};
use embassy_net::{
    Config as NetConfig, IpAddress, IpEndpoint, Ipv4Address, Ipv4Cidr, Stack, StackResources,
};
use embassy_rp::gpio::{Level, Output};
use embassy_rp::multicore::{spawn_core1, Stack as CoreStack};
use embassy_rp::peripherals::{
    DMA_CH0, PIN_0, PIN_1, PIN_18, PIN_19, PIN_20, PIN_23, PIN_24, PIN_25, PIN_29, PIN_4, PIN_5,
    PIO0, UART0, UART1,
};
use embassy_rp::pio::{InterruptHandler, Pio};
use embassy_rp::rom_data::reset_to_usb_boot;
use embassy_rp::uart::{
    Blocking, Config as UartConfig, DataBits, Error as UartError, Parity, StopBits, Uart,
};
use embassy_rp::{bind_interrupts, dma, Peri};
use embassy_time::{Duration, Instant, Timer};
use embedded_hal_nb::serial::Read as _;
use embedded_io_async::Write;
use static_cell::StaticCell;

use crate::body;
use crate::commands::{BrainstemCommand, CreateOiMode, EscapeDirection, LightPattern};
use crate::hardware::{BrainstemHardware, SerialRead, UartReadError};
use crate::runtime::Runtime;
use crate::status;

const AP_SSID: &str = "pete-brainstem";
const MDNS_NAME: &[u8] = b"\x04pete\x05local\x00";
const AP_CHANNEL: u8 = 6;
const AP_IP_OCTETS: [u8; 4] = [192, 168, 4, 1];
const DHCP_LEASE_IP_OCTETS: [u8; 4] = [192, 168, 4, 2];
const AP_IP: Ipv4Address = Ipv4Address::new(192, 168, 4, 1);
const HTTP_PORT: u16 = 80;
const HTTP_TASKS: usize = 3;
const WS_CONTROL_PORT: u16 = 81;
const UDP_CONTROL_PORT: u16 = 82;
const DNS_PORT: u16 = 53;
const MDNS_PORT: u16 = 5353;
const DHCP_SERVER_PORT: u16 = 67;
const DHCP_CLIENT_PORT: u16 = 68;
const DHCP_LEASE_SECONDS: u32 = 3_600;
const DHCP_OFFER_HOLD_SECONDS: u32 = 30;
const HTTP_FLUSH_TIMEOUT_MS: u64 = 250;
const LED_HEARTBEAT_INTERVAL_SECS: u64 = 15;
const LED_BLINK_ON_MS: u64 = 120;
const LED_BLINK_OFF_MS: u64 = 120;
const FOREBRAIN_UART_BAUD: u32 = 115_200;
const FOREBRAIN_LINE_MAX: usize = 96;
const FOREBRAIN_POLL_MS: u64 = 2;
const FOREBRAIN_LINE_TIMEOUT_MS: u32 = 100;

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
        tx: Peri<'static, PIN_0>,
        rx: Peri<'static, PIN_1>,
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
            Err(nb::Error::Other(error)) => SerialRead::Error(map_uart_error(error)),
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
    let hardware = PicoWBrainstem::new(p.UART0, p.PIN_0, p.PIN_1, p.PIN_18, p.PIN_19, p.PIN_20);

    spawn_core1(
        p.CORE1,
        unsafe { &mut *core::ptr::addr_of_mut!(CORE1_STACK) },
        move || Runtime::new(hardware).run_demo(),
    );

    spawn_wifi_lane(
        p.PIO0, p.DMA_CH0, p.PIN_23, p.PIN_24, p.PIN_25, p.PIN_29, p.UART1, p.PIN_4, p.PIN_5,
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

#[embassy_executor::task(pool_size = 3)]
async fn http_task(stack: Stack<'static>) -> ! {
    let mut rx_buffer = [0; 1024];
    let mut tx_buffer = [0; 2048];
    let mut request = [0; 512];
    let mut json = [0; 1536];

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

        let uptime_ms = Instant::now().as_millis() as u32;
        status::mark_http_request(uptime_ms);
        let method = request_method(&request[..n]);
        let path = request_path(&request[..n]);
        let result = match (method, path) {
            (Some("GET"), Some("/") | Some("/index.html")) => {
                write_response(&mut socket, "text/html; charset=utf-8", index_html()).await
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
            (Some("POST"), Some("/command")) => {
                match handle_command_request(&request[..n], &mut json) {
                    Ok(body) => {
                        write_response(&mut socket, "application/json", body.as_bytes()).await
                    }
                    Err(CommandParseError::Busy(command_id)) => {
                        let body =
                            render_command_response(json.as_mut(), false, command_id, "busy");
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
            _ => write_plain_status(&mut socket, 404, "Not Found").await,
        };

        match result {
            Ok(true) => {
                status::mark_http_response_flushed();
                socket.close();
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
    let mut payload = [0; 256];
    let mut response = [0; 1536];

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
    let mut tx_buffer = [0; 512];
    let mut request = [0; 128];
    let mut response = heapless::String::<512>::new();

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
            let Some(boot_to_usb) = handle_udp_control_line(line.trim(), &mut response) else {
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

fn index_html() -> &'static [u8] {
    br#"<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pete Brainstem</title>
<style>
:root{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#1b211d;background:#eef2ed}
*{box-sizing:border-box}body{margin:0}.wrap{max-width:980px;margin:auto;padding:14px}
header{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:12px}
h1{font-size:22px;line-height:1.1;margin:0}h2{font-size:13px;margin:0 0 10px;color:#57615a;text-transform:uppercase;font-weight:800}
.sub{margin-top:4px;font-size:13px;color:#637067}.top{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}
.pill{font-size:12px;border:1px solid #cbd3cd;border-radius:999px;padding:5px 9px;background:#fff;color:#36413a}
.pill.ok{border-color:#8cc7a3;background:#eefbf3}.pill.warn{border-color:#e2bf62;background:#fff8df}.pill.bad{border-color:#d98282;background:#fff0f0}
.grid{display:grid;gap:10px}.panel{background:#fff;border:1px solid #d8ded9;border-radius:8px;padding:12px;box-shadow:0 1px 2px #16201512}
.hero{display:grid;gap:12px}.joy{min-height:285px;display:grid;place-items:center;touch-action:none;user-select:none;background:linear-gradient(180deg,#fbfcfb,#f3f7f3)}
.base{width:min(72vw,270px);height:min(72vw,270px);border-radius:50%;background:#e4ebe5;border:2px solid #c5d0c7;position:relative;box-shadow:inset 0 0 0 28px #edf3ee}
.base:before,.base:after{content:"";position:absolute;background:#cbd5ce}.base:before{width:2px;height:82%;left:50%;top:9%}.base:after{height:2px;width:82%;left:9%;top:50%}
.nub{width:86px;height:86px;border-radius:50%;background:#2d696f;position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);box-shadow:0 8px 18px #13251c33;border:4px solid #f7fbf8}
.row{display:flex;gap:8px;flex-wrap:wrap}.row>*{flex:1 1 auto}.stack{display:grid;gap:8px}.cluster{display:grid;gap:8px;margin-top:12px}
button{min-height:44px;border:1px solid #b6c0b8;border-radius:7px;background:#fff;color:#1f2822;font-weight:750;font-size:14px;letter-spacing:0}
button:active,.active{transform:translateY(1px);background:#eaf0ec}button:disabled{opacity:.55}.primary{background:#dcefe5;border-color:#9cc8ae}.stop{background:#202522;color:#fff;border-color:#202522}.danger{background:#8e242b;color:#fff;border-color:#75191f}
.pad{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}.pad button{min-height:48px}.pad .center{grid-column:2}.pad .widebtn{grid-column:1/-1}
.seg{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.seg button{min-height:38px;font-size:12px}
label{font-size:12px;color:#5c675f;font-weight:700}.slider{display:grid;gap:6px}.slider input{width:100%;accent-color:#2d696f}
.readout{display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px}.tile{background:#f5f7f5;border:1px solid #e1e6e2;border-radius:7px;padding:8px;min-height:48px}
.tile b{display:block;color:#4d5851;font-size:11px;text-transform:uppercase}.tile span{overflow-wrap:anywhere}.wide{grid-column:1/-1}.muted{color:#68736c}.log{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;line-height:1.45;max-height:96px;overflow:auto}
@media(min-width:760px){.grid{grid-template-columns:1.1fr .9fr}.wide{grid-column:1/-1}.hero{grid-template-columns:1fr}.joy{min-height:360px}.controls{align-self:start}}
</style>
</head>
<body>
<div class="wrap">
<header><div><h1>Pete Brainstem</h1><div class="sub" id="headline">Waiting for status</div></div><div class="top"><span id="net" class="pill">connecting</span><span id="mode" class="pill">mode unknown</span></div></header>
<div class="grid">
<section class="panel hero">
<div class="joy"><div id="base" class="base"><div id="nub" class="nub"></div></div></div>
<div class="row"><button class="stop" id="stop">STOP</button><button class="danger" id="estop">E-STOP</button><button id="clear">Clear E-Stop</button></div>
</section>
<section class="panel controls">
<h2>Drive</h2>
<div class="slider"><label for="speed">Speed <span id="speedv">120</span> mm/s</label><input id="speed" type="range" min="40" max="220" value="120"></div>
<div class="slider"><label for="turn">Turn <span id="turnv">1200</span> mrad/s</label><input id="turn" type="range" min="400" max="1800" value="1200"></div>
<div class="pad cluster">
<button class="primary center" data-drive="fwd">FWD</button>
<button data-drive="left">LEFT</button><button class="stop" id="padstop">STOP</button><button data-drive="right">RIGHT</button>
<button data-drive="back" class="center">BACK</button>
<button data-drive="spinl">SPIN L</button><button data-drive="slow">SLOW</button><button data-drive="spinr">SPIN R</button>
</div>
<div class="cluster"><h2>Mode</h2><div class="row"><button id="arm" class="primary">Arm</button><button id="safe">Safe</button><button id="full">Full</button><button id="disarm">Disarm</button><button id="dock">Dock</button><button id="ping">Ping</button></div></div>
<div class="cluster"><h2>Lights</h2><div class="seg"><button data-lights="off">Off</button><button data-lights="status">Status</button><button data-lights="clean">Clean</button><button data-lights="dock">Dock</button><button data-lights="spot">Spot</button><button data-lights="max">Max</button></div></div>
<div class="cluster"><div class="row"><button id="song">Song</button><button id="refresh">Refresh</button><button id="bootsel">BOOTSEL</button></div></div>
</section>
<section class="panel wide">
<h2>Telemetry</h2>
<div class="readout">
<div class="tile"><b>Runtime</b><span id="runtime" class="muted">...</span></div>
<div class="tile"><b>Uptime</b><span id="uptime" class="muted">...</span></div>
<div class="tile"><b>Create</b><span id="create" class="muted">...</span></div>
<div class="tile"><b>UART</b><span id="uart" class="muted">...</span></div>
<div class="tile"><b>Command</b><span id="cmd" class="muted">...</span></div>
<div class="tile"><b>Forebrain</b><span id="forebrain" class="muted">...</span></div>
<div class="tile"><b>Web</b><span id="web" class="muted">...</span></div>
<div class="tile"><b>Firmware</b><span id="firmware" class="muted">...</span></div>
<div class="tile wide"><b>Last error</b><span id="err" class="muted">...</span></div>
<div class="tile wide"><b>Activity</b><div id="log" class="log muted">No commands yet</div></div>
</div>
</section>
</div>
</div>
<script>
let id=1,active=false,timer=0,last={x:0,y:0},ws=null,wsOpen=false,driveKind='',statusBusy=false,lastDriveAt=0;
const $=x=>document.getElementById(x),base=$('base'),nub=$('nub'),net=$('net'),log=$('log');
function title(s){return (s||'unknown').replaceAll('_',' ')}
function pill(el,text,state){el.textContent=text;el.className='pill '+(state||'')}
function addLog(text){let t=new Date().toLocaleTimeString();log.textContent=(t+'  '+text+'\n'+(log.textContent==='No commands yet'?'':log.textContent)).slice(0,900)}
function connectWs(){try{ws=new WebSocket('ws://'+location.hostname+':81/control');ws.onopen=()=>{wsOpen=true;pill(net,'control ws','ok');refresh()};ws.onclose=()=>{wsOpen=false;pill(net,'reconnecting','warn');setTimeout(connectWs,1000)};ws.onerror=()=>{wsOpen=false;pill(net,'ws error','warn')};ws.onmessage=e=>{try{let j=JSON.parse(e.data);if(j.type==='status'){showStatus(j);return}pill(net,j.accepted?'accepted':'busy',j.accepted?'ok':'warn');addLog((j.accepted?'accepted ':'busy ')+j.command_id)}catch(_){}}}catch(_){wsOpen=false}}
function post(o,ack){let cid=id++;o.command_id=cid;if(ack===false)o.ack=false;if(o.kind==='cmd_vel'&&o.seq===undefined)o.seq=cid;let body=JSON.stringify(o),name=o.kind==='cmd_vel'?'drive':o.kind;if(wsOpen&&ws&&ws.readyState===1){if(ws.bufferedAmount<384){ws.send(body);if(ack!==false)addLog('sent '+name);return Promise.resolve({accepted:true})}pill(net,'throttled','warn');return Promise.resolve({accepted:false})}return fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body}).then(r=>r.json()).then(j=>{pill(net,j.accepted?'accepted':'busy',j.accepted?'ok':'warn');addLog((j.accepted?'accepted ':'busy ')+name);return j}).catch(_=>{pill(net,'offline','bad');addLog('offline '+name)})}
function stop(){clearInterval(timer);timer=0;active=false;driveKind='';nub.style.left='50%';nub.style.top='50%';document.querySelectorAll('.active').forEach(b=>b.classList.remove('active'));post({kind:'stop'})}
function joyMax(){return {lin:+$('speed').value,ang:+$('turn').value}}
function paceDrive(fn){let now=Date.now();if(now-lastDriveAt<120)return;lastDriveAt=now;fn()}
function sendJoy(){paceDrive(()=>{let m=joyMax(),lin=Math.round(-last.y*m.lin),ang=Math.round(-last.x*m.ang);post({kind:'cmd_vel',linear_mm_s:lin,angular_mrad_s:ang,ttl_ms:320},false)})}
function sendDrive(){paceDrive(()=>{let m=joyMax(),lin=0,ang=0;if(driveKind==='fwd')lin=m.lin;if(driveKind==='back')lin=-m.lin;if(driveKind==='left')ang=m.ang;if(driveKind==='right')ang=-m.ang;if(driveKind==='spinl')ang=m.ang,lin=0;if(driveKind==='spinr')ang=-m.ang,lin=0;if(driveKind==='slow')lin=Math.round(m.lin*.45);post({kind:'cmd_vel',linear_mm_s:lin,angular_mrad_s:ang,ttl_ms:320},false)})}
function move(e){let r=base.getBoundingClientRect(),cx=r.left+r.width/2,cy=r.top+r.height/2,dx=e.clientX-cx,dy=e.clientY-cy,max=r.width*.34,d=Math.hypot(dx,dy);if(d>max){dx=dx/d*max;dy=dy/d*max}last={x:dx/max,y:dy/max};nub.style.left=(50+dx/r.width*100)+'%';nub.style.top=(50+dy/r.height*100)+'%';sendJoy()}
base.onpointerdown=e=>{active=true;base.setPointerCapture(e.pointerId);move(e);timer=setInterval(sendJoy,180)}
base.onpointermove=e=>{if(active)move(e)}
base.onpointerup=base.onpointercancel=stop
$('stop').onclick=stop;$('padstop').onclick=stop
$('estop').onclick=()=>post({kind:'estop'})
$('clear').onclick=()=>post({kind:'clear_estop'})
$('arm').onclick=()=>post({kind:'arm'})
$('safe').onclick=()=>post({kind:'set_mode',mode:'safe'})
$('full').onclick=()=>post({kind:'set_mode',mode:'full'})
$('disarm').onclick=()=>post({kind:'disarm'})
$('dock').onclick=()=>post({kind:'dock'})
$('ping').onclick=()=>post({kind:'ping'})
$('song').onclick=()=>post({kind:'song_play',id:0})
$('bootsel').onclick=()=>post({kind:'bootsel'})
$('refresh').onclick=refresh
document.querySelectorAll('[data-lights]').forEach(b=>b.onclick=()=>post({kind:'set_lights',pattern:b.dataset.lights}))
document.querySelectorAll('[data-drive]').forEach(b=>{b.onpointerdown=e=>{driveKind=b.dataset.drive;b.classList.add('active');sendDrive();timer=setInterval(sendDrive,190);b.setPointerCapture(e.pointerId)};b.onpointerup=b.onpointercancel=stop})
$('speed').oninput=()=>$('speedv').textContent=$('speed').value
$('turn').oninput=()=>$('turnv').textContent=$('turn').value
function time(ms){let s=Math.floor((ms||0)/1000),m=Math.floor(s/60),h=Math.floor(m/60);return h+'h '+(m%60)+'m '+(s%60)+'s'}
function showStatus(s){let err=s.last_error&&s.last_error!=='none';pill(net,wsOpen?'control ws':(s.wifi_state||'online'),'ok');pill($('mode'),title(s.oi_mode),(s.oi_mode==='safe'||s.oi_mode==='full')?'ok':'');$('headline').textContent=title(s.current_runtime_state)+' / '+title(s.create_power_state)+' / '+title(s.uart_rx_health);$('runtime').textContent=title(s.current_runtime_state)+' / demo '+title(s.demo_state);$('uptime').textContent=time(s.uptime_ms);$('create').textContent=title(s.create_power_state)+' / '+title(s.oi_mode)+' / probe '+s.wake_probe_response_bytes+'/'+s.wake_probe_expected_bytes;$('uart').textContent=title(s.uart_rx_health)+' / '+title(s.last_uart_read_error)+' / '+s.uart_rx_packets+' packets';$('cmd').textContent=title(s.current_command)+' / pending '+title(s.pending_command)+' #'+s.pending_command_id;$('forebrain').textContent=(s.forebrain_uart?s.forebrain_uart.rx_lines:0)+' lines / '+title(s.forebrain_uart&&s.forebrain_uart.last_error);$('web').textContent=s.http_requests+' requests / '+s.dhcp_grants+' dhcp';$('firmware').textContent=s.firmware_name+' '+s.firmware_version;$('err').textContent=err?title(s.last_error)+' / '+(s.last_error_hint||''): 'none';$('err').className=err?'':'muted'}
function refresh(){if(statusBusy)return;statusBusy=true;if(wsOpen&&ws&&ws.readyState===1&&ws.bufferedAmount<384){ws.send(JSON.stringify({kind:'status',command_id:id++}));statusBusy=false;return}fetch('/status.json').then(r=>r.json()).then(showStatus).catch(_=>pill(net,'offline','bad')).finally(()=>statusBusy=false)}
setInterval(refresh,3500);refresh();
connectWs();
</script>
</body>
</html>
"#
}

enum CommandParseError {
    BadRequest,
    Busy(u32),
}

fn handle_command_request<'a>(
    request: &[u8],
    buffer: &'a mut [u8],
) -> Result<&'a str, CommandParseError> {
    let body = request_body(request).ok_or(CommandParseError::BadRequest)?;
    let command_id = json_u32(body, "command_id").ok_or(CommandParseError::BadRequest)?;
    let command = parse_command(command_id, body).ok_or(CommandParseError::BadRequest)?;
    if matches!(command, BrainstemCommand::Status) {
        let snapshot = status::snapshot(Instant::now().as_millis() as u32);
        return status::render_json(snapshot, buffer).map_err(|_| CommandParseError::BadRequest);
    }
    if matches!(command, BrainstemCommand::Ping) {
        return render_command_response(buffer, true, command_id, "pong")
            .ok_or(CommandParseError::BadRequest);
    }
    if matches!(command, BrainstemCommand::Bootsel) {
        return render_bootsel_response(buffer, command_id).ok_or(CommandParseError::BadRequest);
    }
    if !status::submit_control_command(command_id, command) {
        return Err(CommandParseError::Busy(command_id));
    }
    render_command_response(buffer, true, command_id, "accepted")
        .ok_or(CommandParseError::BadRequest)
}

fn handle_websocket_message<'a>(body: &str, buffer: &'a mut [u8]) -> Option<&'a str> {
    if json_str(body, "kind") == Some("status") {
        let snapshot = status::snapshot(Instant::now().as_millis() as u32);
        return render_status_websocket_response(snapshot, buffer);
    }
    if json_str(body, "kind") == Some("ping") {
        let command_id = json_u32(body, "command_id")?;
        return render_command_response(buffer, true, command_id, "pong");
    }

    if json_bool(body, "ack") == Some(false) {
        let command_id = json_u32(body, "command_id")?;
        let command = parse_command(command_id, body)?;
        if matches!(command, BrainstemCommand::Bootsel) {
            return render_bootsel_response(buffer, command_id);
        }
        let accepted = status::submit_control_command(command_id, command);
        if accepted {
            None
        } else {
            render_command_response(buffer, false, command_id, "busy")
        }
    } else {
        handle_websocket_command(body, buffer)
            .or(Some("{\"accepted\":false,\"message\":\"bad_request\"}\n"))
    }
}

fn handle_websocket_command<'a>(body: &str, buffer: &'a mut [u8]) -> Option<&'a str> {
    let command_id = json_u32(body, "command_id")?;
    let command = parse_command(command_id, body)?;
    if matches!(command, BrainstemCommand::Bootsel) {
        return render_bootsel_response(buffer, command_id);
    }
    if !status::submit_control_command(command_id, command) {
        return render_command_response(buffer, false, command_id, "busy");
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

    let (seq, command) = match parse_forebrain_uart_command(line) {
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
        write_forebrain_uart_ok(uart, seq);
        reset_to_usb_boot(0, 0);
    }

    if !status::submit_control_command(seq, command) {
        status::mark_forebrain_uart_error(status::ForebrainUartErrorCode::Busy);
        if matches!(command, BrainstemCommand::CmdVel { .. }) {
            submit_forebrain_stop();
        }
        write_forebrain_uart_error(uart, seq, "busy");
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

    let command = match kind {
        "PING" => BrainstemCommand::Ping,
        "BOOTSEL" => BrainstemCommand::Bootsel,
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
        "HEARTBEAT_STOP" => BrainstemCommand::HeartbeatStop {
            seq,
            timeout_ms: parse_u32(parts.next()).ok_or(seq)?,
        },
        "STATUS" => BrainstemCommand::Status,
        "SONG_PLAY" => BrainstemCommand::SongPlay {
            id: parse_u32(parts.next()).ok_or(seq)? as u8,
        },
        "DOCK" => BrainstemCommand::Dock,
        "SET_LIGHTS" => BrainstemCommand::SetLights {
            pattern: parse_light_pattern(parts.next().ok_or(seq)?).ok_or(seq)?,
        },
        _ => return Err(seq),
    };

    if parts.next().is_some() {
        return Err(seq);
    }

    Ok((seq, command))
}

fn handle_udp_control_line(line: &str, response: &mut heapless::String<512>) -> Option<bool> {
    response.clear();
    let (seq, command) = match parse_forebrain_uart_command(line) {
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
        BrainstemCommand::Bootsel => {
            let _ = writeln!(response, "OK {seq} bootsel");
            Some(true)
        }
        command => {
            if status::submit_control_command(seq, command) {
                let _ = writeln!(response, "OK {seq}");
            } else {
                let _ = writeln!(response, "ERR {seq} busy");
            }
            Some(false)
        }
    }
}

fn parse_u32(value: Option<&str>) -> Option<u32> {
    value?.parse().ok()
}

fn parse_i16(value: Option<&str>) -> Option<i16> {
    value?.parse().ok()
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

fn write_compact_status_line<const N: usize>(response: &mut heapless::String<N>, seq: u32) {
    let snapshot = status::snapshot(Instant::now().as_millis() as u32);
    let _ = writeln!(
        response,
        "OK {seq} STATUS runtime={} demo={} action={} command={} pending={} error={} error_uart={} power={} oi={} uart_health={} uart_error={} create_rx_bytes={} create_rx_packets={} create_last_packet_len={} wake_probe={}/{} forebrain_rx_bytes={} forebrain_rx_lines={}",
        snapshot.current_runtime_state,
        snapshot.demo_state,
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
        snapshot.last_uart_packet_len,
        snapshot.wake_probe_response_bytes,
        snapshot.wake_probe_expected_bytes,
        snapshot.forebrain_uart_rx_bytes,
        snapshot.forebrain_uart_rx_lines
    );
}

fn render_bootsel_response(buffer: &mut [u8], command_id: u32) -> Option<&str> {
    let body = render_command_response(buffer, true, command_id, "bootsel")?;
    reset_to_usb_boot(0, 0);
    Some(body)
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
        "heartbeat_stop" => Some(BrainstemCommand::HeartbeatStop {
            timeout_ms: json_u32(body, "timeout_ms").or_else(|| json_u32(body, "ttl_ms"))?,
            seq: json_u32(body, "seq").unwrap_or(command_id),
        }),
        "status" => Some(BrainstemCommand::Status),
        "song_play" => Some(BrainstemCommand::SongPlay {
            id: json_u32(body, "id")? as u8,
        }),
        "dock" => Some(BrainstemCommand::Dock),
        "set_lights" => Some(BrainstemCommand::SetLights {
            pattern: parse_light_pattern(json_str(body, "pattern")?)?,
        }),
        _ => None,
    }
}

fn parse_light_pattern(pattern: &str) -> Option<LightPattern> {
    match pattern {
        "off" => Some(LightPattern::Off),
        "status" => Some(LightPattern::Status),
        "clean" => Some(LightPattern::Clean),
        "dock" => Some(LightPattern::Dock),
        "spot" => Some(LightPattern::Spot),
        "max" => Some(LightPattern::Max),
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
    let key_start = body.find(key)?;
    let after_key = &body[key_start + key.len()..];
    let colon = after_key.find(':')?;
    let after_colon = after_key[colon + 1..].trim_start();
    let value = after_colon.strip_prefix('"')?;
    let end = value.find('"')?;
    Some(&value[..end])
}

fn json_u32(body: &str, key: &str) -> Option<u32> {
    json_i32(body, key).and_then(|value| u32::try_from(value).ok())
}

fn json_i16(body: &str, key: &str) -> Option<i16> {
    json_i32(body, key).and_then(|value| i16::try_from(value).ok())
}

fn json_bool(body: &str, key: &str) -> Option<bool> {
    let key_start = body.find(key)?;
    let after_key = &body[key_start + key.len()..];
    let colon = after_key.find(':')?;
    let value = after_key[colon + 1..].trim_start();
    if value.starts_with("true") {
        Some(true)
    } else if value.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

fn json_i32(body: &str, key: &str) -> Option<i32> {
    let key_start = body.find(key)?;
    let after_key = &body[key_start + key.len()..];
    let colon = after_key.find(':')?;
    let value = after_key[colon + 1..].trim_start();
    let end = value
        .find(|c: char| !(c == '-' || c.is_ascii_digit()))
        .unwrap_or(value.len());
    value[..end].parse().ok()
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

fn build_dns_reply<'a>(query: &[u8], response: &'a mut [u8; 512]) -> Option<&'a [u8]> {
    let question = parse_dns_question(query)?;
    if !dns_name_matches_pete(&query[12..question.name_end])
        || !matches!(question.qtype, 1 | 255)
        || !matches!(question.qclass, 1 | 255)
    {
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
        AP_IP_OCTETS[0],
        AP_IP_OCTETS[1],
        AP_IP_OCTETS[2],
        AP_IP_OCTETS[3],
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

fn dns_name_matches_pete(name: &[u8]) -> bool {
    name.len() == MDNS_NAME.len()
        && name
            .iter()
            .zip(MDNS_NAME.iter())
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
    response[16..20].copy_from_slice(&DHCP_LEASE_IP_OCTETS);
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

#[derive(Clone, Copy, Eq, PartialEq)]
struct DhcpClient {
    hardware_address: [u8; 6],
}

#[derive(Clone, Copy)]
struct DhcpRequest {
    message_type: u8,
    client: DhcpClient,
}

impl DhcpRequest {
    fn parse(packet: &[u8]) -> Option<Self> {
        if packet.len() < 240 || packet[0] != 1 || packet[1] != 1 || packet[2] < 6 {
            return None;
        }

        let mut hardware_address = [0; 6];
        hardware_address.copy_from_slice(&packet[28..34]);

        Some(Self {
            message_type: dhcp_message_type(packet)?,
            client: DhcpClient { hardware_address },
        })
    }
}

#[derive(Clone, Copy)]
struct DhcpLease {
    client: DhcpClient,
    expires_at_ms: u64,
}

#[derive(Clone, Copy)]
enum DhcpGrant {
    Offer,
    Ack,
}

impl DhcpGrant {
    fn reply_message_type(self) -> u8 {
        match self {
            Self::Offer => 2,
            Self::Ack => 5,
        }
    }
}

struct DhcpLeaseState {
    active: Option<DhcpLease>,
}

impl DhcpLeaseState {
    const fn new() -> Self {
        Self { active: None }
    }

    fn grant(&mut self, request: DhcpRequest, now_ms: u64) -> Option<DhcpGrant> {
        self.clear_expired(now_ms);

        match request.message_type {
            1 => self
                .reserve(request.client, now_ms, DHCP_OFFER_HOLD_SECONDS)
                .then_some(DhcpGrant::Offer),
            3 => self
                .reserve(request.client, now_ms, DHCP_LEASE_SECONDS)
                .then_some(DhcpGrant::Ack),
            7 => {
                self.release(request.client);
                None
            }
            _ => None,
        }
    }

    fn reserve(&mut self, client: DhcpClient, now_ms: u64, seconds: u32) -> bool {
        if let Some(active) = self.active {
            if active.client != client {
                return false;
            }
        }

        self.active = Some(DhcpLease {
            client,
            expires_at_ms: now_ms.saturating_add(seconds as u64 * 1_000),
        });
        true
    }

    fn release(&mut self, client: DhcpClient) {
        if self
            .active
            .map(|active| active.client == client)
            .unwrap_or(false)
        {
            self.active = None;
        }
    }

    fn clear_expired(&mut self, now_ms: u64) {
        if self
            .active
            .map(|active| now_ms >= active.expires_at_ms)
            .unwrap_or(false)
        {
            self.active = None;
        }
    }
}

fn dhcp_message_type(packet: &[u8]) -> Option<u8> {
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
                if option == 53 && len == 1 {
                    return Some(packet[i]);
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
