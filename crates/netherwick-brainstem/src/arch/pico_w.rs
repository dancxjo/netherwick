use core::fmt::Write as _;

use cyw43::aligned_bytes;
use cyw43_pio::{PioSpi, DEFAULT_CLOCK_DIVIDER};
use embassy_executor::Spawner;
use embassy_net::tcp::TcpSocket;
use embassy_net::udp::{PacketMetadata, UdpSocket};
use embassy_net::{
    Config as NetConfig, IpAddress, IpEndpoint, Ipv4Address, Ipv4Cidr, Stack, StackResources,
};
use embassy_rp::gpio::{Level, Output};
use embassy_rp::multicore::{spawn_core1, Stack as CoreStack};
use embassy_rp::peripherals::{
    DMA_CH0, PIN_0, PIN_1, PIN_18, PIN_19, PIN_20, PIN_23, PIN_24, PIN_25, PIN_29, PIO0, UART0,
};
use embassy_rp::pio::{InterruptHandler, Pio};
use embassy_rp::uart::{
    Blocking, Config as UartConfig, DataBits, Error as UartError, Parity, StopBits, Uart,
};
use embassy_rp::{bind_interrupts, dma, Peri};
use embassy_time::{Duration, Instant, Timer};
use embedded_hal_nb::serial::Read as _;
use embedded_io_async::Write;
use static_cell::StaticCell;

use crate::body;
use crate::commands::{BrainstemCommand, CreateOiMode};
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
const MDNS_PORT: u16 = 5353;
const DHCP_SERVER_PORT: u16 = 67;
const DHCP_CLIENT_PORT: u16 = 68;
const DHCP_LEASE_SECONDS: u32 = 3_600;
const DHCP_OFFER_HOLD_SECONDS: u32 = 30;
const LED_HEARTBEAT_INTERVAL_SECS: u64 = 15;
const LED_BLINK_ON_MS: u64 = 120;
const LED_BLINK_OFF_MS: u64 = 120;

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

    spawn_wifi_lane(p.PIO0, p.DMA_CH0, p.PIN_23, p.PIN_24, p.PIN_25, p.PIN_29);
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
        spawner.spawn(
            wifi_task(spawner, pio0, dma0, wifi_power, wifi_dio, wifi_cs, wifi_clk)
                .expect("spawn wifi task"),
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
        spawner.spawn(http_task(stack).expect("spawn http task"));
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
        .set_power_management(cyw43::PowerManagementMode::PowerSave)
        .await;

    let config = NetConfig::ipv4_static(embassy_net::StaticConfigV4 {
        address: Ipv4Cidr::new(AP_IP, 24),
        dns_servers: Default::default(),
        gateway: None,
    });

    static RESOURCES: StaticCell<StackResources<5>> = StaticCell::new();
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
                write_response(&mut socket, "text/plain; charset=utf-8", hello_body()).await
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
            Ok(()) => {
                socket.close();
                if socket.flush().await.is_ok() {
                    status::mark_http_response_flushed();
                } else {
                    status::mark_http_response_error();
                    socket.abort();
                    let _ = socket.flush().await;
                }
            }
            Err(_) => {
                status::mark_http_response_error();
                socket.abort();
                let _ = socket.flush().await;
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
) -> Result<(), embassy_net::tcp::Error> {
    let mut header = heapless::String::<192>::new();
    let _ = write!(
        header,
        "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        content_type,
        body.len()
    );
    socket.write_all(header.as_bytes()).await?;
    socket.write_all(body).await?;
    socket.flush().await
}

async fn write_plain_status(
    socket: &mut TcpSocket<'_>,
    code: u16,
    text: &str,
) -> Result<(), embassy_net::tcp::Error> {
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
    socket.flush().await
}

fn request_path(request: &[u8]) -> Option<&str> {
    let line_end = request
        .windows(2)
        .position(|w| w == b"\r\n")
        .unwrap_or(request.len());
    let line = core::str::from_utf8(&request[..line_end]).ok()?;
    let mut parts = line.split(' ');
    match (parts.next(), parts.next()) {
        (Some("GET"), Some(path)) => Some(path),
        _ => None,
    }
}

fn request_method(request: &[u8]) -> Option<&str> {
    let line_end = request
        .windows(2)
        .position(|w| w == b"\r\n")
        .unwrap_or(request.len());
    let line = core::str::from_utf8(&request[..line_end]).ok()?;
    line.split(' ').next()
}

fn hello_body() -> &'static [u8] {
    b"hello, I'm at least up\nstatus: /status.json\n"
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
    let command = parse_command(body).ok_or(CommandParseError::BadRequest)?;
    if !status::submit_control_command(command_id, command) {
        return Err(CommandParseError::Busy(command_id));
    }
    render_command_response(buffer, true, command_id, "accepted")
        .ok_or(CommandParseError::BadRequest)
}

fn request_body(request: &[u8]) -> Option<&str> {
    let body_start = request
        .windows(4)
        .position(|w| w == b"\r\n\r\n")?
        .checked_add(4)?;
    core::str::from_utf8(&request[body_start..]).ok()
}

fn parse_command(body: &str) -> Option<BrainstemCommand> {
    match json_str(body, "kind")? {
        "wake_create" | "wake" => Some(BrainstemCommand::WakeCreate),
        "sleep_create" | "sleep" => Some(BrainstemCommand::SleepCreate),
        "stop" => Some(BrainstemCommand::Stop),
        "estop" => Some(BrainstemCommand::EStop),
        "clear_estop" => Some(BrainstemCommand::ClearEStop),
        "set_mode" => match json_str(body, "mode")? {
            "passive" => Some(BrainstemCommand::SetMode(CreateOiMode::Passive)),
            "safe" => Some(BrainstemCommand::SetMode(CreateOiMode::Safe)),
            "full" => Some(BrainstemCommand::SetMode(CreateOiMode::Full)),
            _ => None,
        },
        "drive_direct" => Some(BrainstemCommand::DriveDirect {
            left_mm_s: json_i16(body, "left_mm_s")?,
            right_mm_s: json_i16(body, "right_mm_s")?,
            duration_ms: Some(json_u32(body, "duration_ms")?),
        }),
        "cmd_vel" => Some(BrainstemCommand::CmdVel {
            linear_mm_s: json_i16(body, "linear_mm_s")?,
            angular_mrad_s: json_i16(body, "angular_mrad_s")?,
            duration_ms: Some(json_u32(body, "duration_ms")?),
        }),
        "drive_arc" => Some(BrainstemCommand::DriveArc {
            velocity_mm_s: json_i16(body, "velocity_mm_s")?,
            radius_mm: json_i16(body, "radius_mm")?,
            duration_ms: Some(json_u32(body, "duration_ms")?),
        }),
        "ping" => Some(BrainstemCommand::Ping),
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
