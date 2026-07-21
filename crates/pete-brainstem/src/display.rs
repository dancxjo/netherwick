use core::fmt::Write as _;

use heapless::String;

use crate::status::{
    self, BodyState, BrainstemStatus, ImuHealthCode, RuntimeState, SafetyEventKind,
};

pub const WIDTH: usize = 128;
pub const HEIGHT: usize = 32;
pub const FRAMEBUFFER_BYTES: usize = WIDTH * HEIGHT / 8;
pub const REFRESH_PERIOD_MS: u32 = 200;
pub const PAGE_ROTATION_MS: u32 = 3_000;
pub const LINK_FRESHNESS_MS: u32 = 2_000;
pub const BATTERY_FRESHNESS_MS: u32 = 2_000;
pub const LOW_BATTERY_PERCENT: u32 = 20;

const LIVENESS_TOGGLE_MS: u32 = 500;

const LINE_CAPACITY: usize = 22;
const CREATE_POWER_OFF: u8 = 1;
const WIFI_SERVICES_STARTED: u8 = 3;
const WIFI_ERROR: u8 = 4;
const ERROR_CREATE_NO_RESPONSE: u8 = 1;
const ERROR_UART_FRAMING: u8 = 2;
const ERROR_TIMEOUT: u8 = 3;
const ERROR_INVALID_PACKET: u8 = 4;

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct DisplayNetwork {
    pub ssid_suffix: Option<u32>,
    pub active_leases: u8,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct DisplaySafety {
    pub estop_latched: bool,
    pub safety_latch_kind: Option<SafetyEventKind>,
}

impl DisplaySafety {
    pub fn current() -> Self {
        let (estop_latched, _, _, safety_latch_kind) = status::session_safety_snapshot();
        Self {
            estop_latched,
            safety_latch_kind,
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct DisplayStatus {
    runtime_state: u8,
    body_state: u8,
    create_power_state: u8,
    oi_mode: u8,
    oi_seen: bool,
    oi_fresh: bool,
    authority_active: bool,
    imu_enabled: bool,
    imu_health: u8,
    last_error: u8,
    wifi_state: u8,
    network: DisplayNetwork,
    battery: Option<BatteryStatus>,
    battery_stale: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BatteryStatus {
    percent: u8,
    charging: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DisplayPage {
    pub line1: String<LINE_CAPACITY>,
    pub line2: String<LINE_CAPACITY>,
    layout: DisplayLayout,
    liveness: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DisplayLayout {
    Dashboard {
        state: StateIcon,
        authority: AuthorityIcon,
    },
    Alert(AlertIcon),
    Battery(BatteryStatus),
    Network(NetworkStatus),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StateIcon {
    Boot,
    Ready,
    Run,
    Stop,
    Warn,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AuthorityIcon {
    Open,
    Active,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AlertIcon {
    Bump,
    Cliff,
    WheelDrop,
    EStop,
    Heartbeat,
    Tilt,
    Impact,
    Charging,
    OiLinkLost,
    LowBattery,
    BatteryStale,
    ImuOffline,
    WaitCreate,
    PowerOff,
    CreateNoRx,
    UartFraming,
    Timeout,
    InvalidPacket,
    RuntimeError,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NetworkStatus {
    ssid_suffix: Option<u32>,
    state: NetworkState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NetworkState {
    Starting,
    Ready,
    Lease(u8),
    Error,
}

impl DisplayStatus {
    pub fn from_snapshot(snapshot: &BrainstemStatus, network: DisplayNetwork) -> Self {
        let oi_seen = snapshot.uart_rx_packets > 0;
        let oi_fresh = oi_seen
            && snapshot
                .uptime_ms
                .wrapping_sub(snapshot.last_uart_packet_timestamp_ms)
                <= LINK_FRESHNESS_MS;
        let battery_fresh = snapshot.create_sensor_complete_packet_count > 0
            && snapshot
                .uptime_ms
                .wrapping_sub(snapshot.create_sensor_last_complete_packet_timestamp_ms)
                <= BATTERY_FRESHNESS_MS;
        let battery = battery_fresh.then(|| battery_status(snapshot)).flatten();
        let authority_active = status::session_diagnostics(snapshot.uptime_ms).authority_active;

        Self {
            runtime_state: snapshot.current_runtime_state,
            body_state: snapshot.body_state,
            create_power_state: snapshot.create_power_state,
            oi_mode: snapshot.oi_mode,
            oi_seen,
            oi_fresh,
            authority_active,
            imu_enabled: crate::body::IMU_ENABLED,
            imu_health: snapshot.imu_health,
            last_error: snapshot.last_error,
            wifi_state: snapshot.wifi_state,
            network,
            battery,
            battery_stale: snapshot.create_sensor_complete_packet_count > 0 && !battery_fresh,
        }
    }

    pub fn page(self, safety: DisplaySafety, now_ms: u32) -> DisplayPage {
        let mut selected = if safety.estop_latched {
            alert_page(AlertIcon::EStop)
        } else if let Some(kind) = safety.safety_latch_kind {
            safety_alert_page(kind)
        } else if self.runtime_state == RuntimeState::Error as u8
            || self.body_state == BodyState::Error as u8
            || self.last_error != 0
        {
            runtime_error_page(self.last_error)
        } else if self.create_power_state == CREATE_POWER_OFF {
            alert_page(AlertIcon::PowerOff)
        } else if self.oi_seen && !self.oi_fresh {
            alert_page(AlertIcon::OiLinkLost)
        } else if self
            .battery
            .is_some_and(|battery| u32::from(battery.percent) <= LOW_BATTERY_PERCENT)
        {
            alert_page(AlertIcon::LowBattery)
        } else if self.battery_stale {
            alert_page(AlertIcon::BatteryStale)
        } else if self.imu_enabled
            && matches!(
                self.imu_health,
                x if x == ImuHealthCode::Fault as u8 || x == ImuHealthCode::Absent as u8
            )
        {
            alert_page(AlertIcon::ImuOffline)
        } else {
            let rotation = (now_ms / PAGE_ROTATION_MS) % 3;
            if !self.oi_seen && rotation == 0 {
                network_page(self)
            } else if !self.oi_seen {
                alert_page(AlertIcon::WaitCreate)
            } else if let Some(battery) = self.battery.filter(|battery| battery.charging) {
                battery_page(battery)
            } else if self.wifi_state != WIFI_SERVICES_STARTED && rotation == 1 {
                network_page(self)
            } else if let Some(battery) = self.battery.filter(|_| rotation == 2) {
                battery_page(battery)
            } else {
                health_page(self)
            }
        };
        selected.liveness = (now_ms / LIVENESS_TOGGLE_MS) % 2 != 0;
        selected
    }
}

pub fn render(page: &DisplayPage) -> [u8; FRAMEBUFFER_BYTES] {
    let mut framebuffer = [0u8; FRAMEBUFFER_BYTES];
    match page.layout {
        DisplayLayout::Dashboard { state, authority } => {
            render_dashboard(&mut framebuffer, state, authority)
        }
        DisplayLayout::Alert(alert) => render_alert(&mut framebuffer, alert),
        DisplayLayout::Battery(battery) => render_battery_page(&mut framebuffer, battery),
        DisplayLayout::Network(network) => render_network_page(&mut framebuffer, network),
    }
    if page.liveness {
        set_pixel(&mut framebuffer, (WIDTH - 1) as i16, (HEIGHT - 1) as i16);
    }
    framebuffer
}

fn safety_alert_page(kind: SafetyEventKind) -> DisplayPage {
    let alert = match kind {
        SafetyEventKind::Bump => AlertIcon::Bump,
        SafetyEventKind::Cliff => AlertIcon::Cliff,
        SafetyEventKind::WheelDrop => AlertIcon::WheelDrop,
        SafetyEventKind::EStop => AlertIcon::EStop,
        SafetyEventKind::Heartbeat => AlertIcon::Heartbeat,
        SafetyEventKind::Tilt => AlertIcon::Tilt,
        SafetyEventKind::Impact => AlertIcon::Impact,
        SafetyEventKind::Charging => AlertIcon::Charging,
    };
    alert_page(alert)
}

fn runtime_error_page(error: u8) -> DisplayPage {
    alert_page(match error {
        ERROR_CREATE_NO_RESPONSE => AlertIcon::CreateNoRx,
        ERROR_UART_FRAMING => AlertIcon::UartFraming,
        ERROR_TIMEOUT => AlertIcon::Timeout,
        ERROR_INVALID_PACKET => AlertIcon::InvalidPacket,
        _ => AlertIcon::RuntimeError,
    })
}

fn alert_page(alert: AlertIcon) -> DisplayPage {
    let (line1, line2) = alert_text(alert);
    page(line1, line2, DisplayLayout::Alert(alert))
}

fn battery_page(battery: BatteryStatus) -> DisplayPage {
    let mut line1 = String::new();
    let mut line2 = String::new();
    let _ = write!(line1, "BATT {}%", battery.percent);
    let _ = line2.push_str(if battery.charging {
        "CHARGING"
    } else {
        "ON BATTERY"
    });
    DisplayPage {
        line1,
        line2,
        layout: DisplayLayout::Battery(battery),
        liveness: false,
    }
}

fn health_page(status: DisplayStatus) -> DisplayPage {
    let (state, state_icon) = if status.runtime_state == RuntimeState::Booting as u8
        || matches!(
            status.body_state,
            x if x == BodyState::NotStarted as u8 || x == BodyState::WaitingForCreate as u8
        ) {
        ("BOOT", StateIcon::Boot)
    } else if status.runtime_state == RuntimeState::Error as u8
        || status.body_state == BodyState::Error as u8
    {
        ("WARN", StateIcon::Warn)
    } else if status.body_state == BodyState::Moving as u8 {
        ("RUN", StateIcon::Run)
    } else if matches!(status.oi_mode, 2..=3) {
        ("READY", StateIcon::Ready)
    } else {
        ("STOP", StateIcon::Stop)
    };

    let mut line1 = String::new();
    let mut line2 = String::new();
    let _ = write!(line1, "PETE  {state}");
    let _ = write!(
        line2,
        "CTRL {}",
        if status.authority_active {
            "ACTIVE"
        } else {
            "OPEN"
        }
    );
    DisplayPage {
        line1,
        line2,
        layout: DisplayLayout::Dashboard {
            state: state_icon,
            authority: if status.authority_active {
                AuthorityIcon::Active
            } else {
                AuthorityIcon::Open
            },
        },
        liveness: false,
    }
}

fn network_page(status: DisplayStatus) -> DisplayPage {
    let state = if status.wifi_state == WIFI_ERROR {
        NetworkState::Error
    } else if status.wifi_state != WIFI_SERVICES_STARTED {
        NetworkState::Starting
    } else if status.network.active_leases > 0 {
        NetworkState::Lease(status.network.active_leases)
    } else {
        NetworkState::Ready
    };
    let network = NetworkStatus {
        ssid_suffix: status.network.ssid_suffix,
        state,
    };
    let ssid = ssid_text(network.ssid_suffix);
    let mut line2 = String::new();
    let _ = write!(line2, "192.168.4.1 {}", network_state_text(state));
    if let NetworkState::Lease(count) = state {
        let _ = write!(line2, " {count}");
    }
    DisplayPage {
        line1: ssid,
        line2,
        layout: DisplayLayout::Network(network),
        liveness: false,
    }
}

fn ssid_text(suffix: Option<u32>) -> String<LINE_CAPACITY> {
    let mut ssid = String::new();
    let _ = ssid.push_str("pete-");
    let Some(mut value) = suffix else {
        let _ = ssid.push_str("????");
        return ssid;
    };
    let mut digits = [b'0'; 4];
    for digit in digits.iter_mut().rev() {
        let remainder = (value % 36) as u8;
        *digit = if remainder < 10 {
            b'0' + remainder
        } else {
            b'a' + remainder - 10
        };
        value /= 36;
    }
    for digit in digits {
        let _ = ssid.push(digit as char);
    }
    ssid
}

fn network_state_text(state: NetworkState) -> &'static str {
    match state {
        NetworkState::Starting => "START",
        NetworkState::Ready => "READY",
        NetworkState::Lease(_) => "LEASE",
        NetworkState::Error => "ERROR",
    }
}

fn alert_text(alert: AlertIcon) -> (&'static str, &'static str) {
    match alert {
        AlertIcon::Bump => ("BUMP", ""),
        AlertIcon::Cliff => ("CLIFF", ""),
        AlertIcon::WheelDrop => ("WHEEL", "DROP"),
        AlertIcon::EStop => ("ESTOP", ""),
        AlertIcon::Heartbeat => ("CTRL", "LOST"),
        AlertIcon::Tilt => ("TILT", ""),
        AlertIcon::Impact => ("IMPACT", ""),
        AlertIcon::Charging => ("NO", "DRIVE"),
        AlertIcon::OiLinkLost => ("OI LINK", "LOST"),
        AlertIcon::LowBattery => ("LOW", "BATT"),
        AlertIcon::BatteryStale => ("BATT", "STALE"),
        AlertIcon::ImuOffline => ("IMU", "OFFLINE"),
        AlertIcon::WaitCreate => ("WAIT", "CREATE"),
        AlertIcon::PowerOff => ("POWER", "OFF"),
        AlertIcon::CreateNoRx => ("OI NO", "RX"),
        AlertIcon::UartFraming => ("UART", "FRAME"),
        AlertIcon::Timeout => ("TIME", "OUT"),
        AlertIcon::InvalidPacket => ("BAD", "PACKET"),
        AlertIcon::RuntimeError => ("RUNTIME", "ERROR"),
    }
}

fn battery_status(snapshot: &BrainstemStatus) -> Option<BatteryStatus> {
    let capacity = u32::from(snapshot.create_sensor_capacity_mah);
    let charge = u32::from(snapshot.create_sensor_charge_mah);
    if capacity == 0 || charge > capacity {
        return None;
    }
    Some(BatteryStatus {
        percent: ((charge * 100) / capacity).min(100) as u8,
        charging: status::charging_interlock_active(snapshot),
    })
}

fn page(line1: &str, line2: &str, layout: DisplayLayout) -> DisplayPage {
    let mut result = DisplayPage {
        line1: String::new(),
        line2: String::new(),
        layout,
        liveness: false,
    };
    let _ = result.line1.push_str(line1);
    let _ = result.line2.push_str(line2);
    result
}

fn render_dashboard(
    framebuffer: &mut [u8; FRAMEBUFFER_BYTES],
    state: StateIcon,
    authority: AuthorityIcon,
) {
    render_centered_text(framebuffer, 0, WIDTH, 1, 2, state_text(state));
    render_centered_text(
        framebuffer,
        0,
        WIDTH,
        17,
        2,
        match authority {
            AuthorityIcon::Open => "OPEN",
            AuthorityIcon::Active => "CTRL",
        },
    );
}

fn state_text(state: StateIcon) -> &'static str {
    match state {
        StateIcon::Boot => "BOOT",
        StateIcon::Ready => "READY",
        StateIcon::Run => "RUN",
        StateIcon::Stop => "STOP",
        StateIcon::Warn => "WARN",
    }
}

fn render_alert(framebuffer: &mut [u8; FRAMEBUFFER_BYTES], alert: AlertIcon) {
    draw_vline(framebuffer, 39, 2, 28);
    match alert {
        AlertIcon::Bump => {
            draw_circle(framebuffer, 16, 15, 8);
            for (x0, y0, x1, y1) in [
                (3, 15, 8, 15),
                (24, 15, 34, 15),
                (28, 11, 34, 15),
                (28, 19, 34, 15),
            ] {
                draw_line(framebuffer, x0, y0, x1, y1);
            }
        }
        AlertIcon::Cliff => {
            draw_rect(framebuffer, 4, 8, 19, 12);
            draw_circle(framebuffer, 9, 22, 3);
            draw_circle(framebuffer, 20, 22, 3);
            draw_hline(framebuffer, 2, 27, 22);
            draw_vline(framebuffer, 27, 21, 8);
            draw_line(framebuffer, 27, 29, 35, 29);
        }
        AlertIcon::WheelDrop => {
            draw_circle(framebuffer, 16, 11, 8);
            draw_vline(framebuffer, 16, 3, 17);
            draw_line(framebuffer, 11, 24, 16, 29);
            draw_line(framebuffer, 21, 24, 16, 29);
        }
        AlertIcon::EStop => {
            draw_octagon(framebuffer, 19, 15, 13);
            draw_vline(framebuffer, 19, 7, 10);
            fill_rect(framebuffer, 18, 21, 3, 3);
        }
        AlertIcon::Heartbeat => {
            draw_line(framebuffer, 3, 16, 9, 16);
            draw_line(framebuffer, 9, 16, 12, 7);
            draw_line(framebuffer, 12, 7, 17, 25);
            draw_line(framebuffer, 17, 25, 21, 12);
            draw_line(framebuffer, 21, 12, 24, 16);
            draw_line(framebuffer, 24, 16, 35, 16);
            draw_line(framebuffer, 4, 4, 34, 27);
        }
        AlertIcon::Tilt => {
            draw_rect(framebuffer, 8, 8, 18, 14);
            draw_line(framebuffer, 7, 23, 30, 5);
            draw_line(framebuffer, 4, 27, 35, 27);
        }
        AlertIcon::Impact => {
            for (x0, y0, x1, y1) in [
                (19, 1, 19, 9),
                (19, 21, 19, 30),
                (4, 15, 12, 15),
                (26, 15, 35, 15),
                (8, 5, 14, 11),
                (25, 20, 32, 27),
                (8, 26, 14, 20),
                (25, 10, 32, 3),
            ] {
                draw_line(framebuffer, x0, y0, x1, y1);
            }
            fill_rect(framebuffer, 15, 11, 9, 9);
        }
        AlertIcon::Charging => {
            draw_rect(framebuffer, 4, 7, 28, 18);
            fill_rect(framebuffer, 32, 12, 4, 8);
            draw_bolt(framebuffer, 14, 7);
        }
        AlertIcon::OiLinkLost => {
            draw_rect(framebuffer, 8, 7, 11, 10);
            draw_rect(framebuffer, 22, 7, 9, 10);
            draw_line(framebuffer, 17, 12, 22, 12);
            draw_line(framebuffer, 5, 3, 34, 25);
        }
        AlertIcon::LowBattery => {
            draw_rect(framebuffer, 5, 8, 27, 16);
            fill_rect(framebuffer, 32, 13, 3, 6);
            fill_rect(framebuffer, 8, 11, 4, 10);
        }
        AlertIcon::BatteryStale => {
            draw_rect(framebuffer, 5, 8, 27, 16);
            fill_rect(framebuffer, 32, 13, 3, 6);
            draw_line(framebuffer, 9, 11, 28, 21);
            draw_line(framebuffer, 28, 11, 9, 21);
        }
        AlertIcon::ImuOffline => {
            draw_rect(framebuffer, 8, 6, 20, 20);
            draw_line(framebuffer, 11, 20, 18, 12);
            draw_line(framebuffer, 18, 12, 25, 20);
            draw_line(framebuffer, 5, 3, 33, 28);
        }
        AlertIcon::WaitCreate => {
            draw_rect(framebuffer, 7, 8, 22, 15);
            draw_vline(framebuffer, 11, 3, 5);
            draw_vline(framebuffer, 24, 3, 5);
            draw_circle(framebuffer, 13, 15, 2);
            draw_circle(framebuffer, 23, 15, 2);
            draw_hline(framebuffer, 13, 20, 11);
        }
        AlertIcon::PowerOff => {
            draw_circle(framebuffer, 19, 16, 13);
            draw_vline(framebuffer, 19, 2, 15);
        }
        AlertIcon::CreateNoRx => {
            draw_rect(framebuffer, 7, 8, 23, 16);
            draw_vline(framebuffer, 12, 3, 5);
            draw_vline(framebuffer, 25, 3, 5);
            draw_line(framebuffer, 4, 3, 34, 28);
        }
        AlertIcon::UartFraming => {
            draw_rect(framebuffer, 5, 5, 29, 22);
            for y in [10, 16, 22] {
                draw_hline(framebuffer, 9, y, 21);
            }
            draw_line(framebuffer, 4, 3, 35, 29);
        }
        AlertIcon::Timeout => {
            draw_circle(framebuffer, 19, 16, 13);
            draw_vline(framebuffer, 19, 6, 11);
            draw_line(framebuffer, 19, 16, 27, 20);
        }
        AlertIcon::InvalidPacket => {
            draw_rect(framebuffer, 5, 6, 29, 20);
            draw_hline(framebuffer, 9, 11, 21);
            draw_hline(framebuffer, 9, 16, 16);
            draw_line(framebuffer, 4, 3, 35, 29);
        }
        AlertIcon::RuntimeError => {
            draw_triangle(framebuffer, 19, 2, 3, 28, 35, 28);
            draw_vline(framebuffer, 19, 9, 10);
            fill_rect(framebuffer, 18, 23, 3, 3);
        }
    }

    let (line1, line2) = alert_text(alert);
    if line2.is_empty() {
        render_centered_text(framebuffer, 41, WIDTH, 9, 2, line1);
    } else {
        render_centered_text(framebuffer, 41, WIDTH, 1, 2, line1);
        render_centered_text(framebuffer, 41, WIDTH, 17, 2, line2);
    }
}

fn render_network_page(framebuffer: &mut [u8; FRAMEBUFFER_BYTES], network: NetworkStatus) {
    draw_vline(framebuffer, 31, 2, 28);
    for radius in [5, 10, 15] {
        draw_arc_top(framebuffer, 15, 20, radius);
    }
    fill_rect(framebuffer, 13, 22, 5, 5);

    let ssid = ssid_text(network.ssid_suffix);
    render_text(framebuffer, 37, 1, 1, &ssid);
    render_text(framebuffer, 37, 12, 1, "192.168.4.1");
    match network.state {
        NetworkState::Starting => render_text(framebuffer, 37, 23, 1, "AP START"),
        NetworkState::Ready => render_text(framebuffer, 37, 23, 1, "AP READY"),
        NetworkState::Lease(count) => {
            let mut label = String::<12>::new();
            let _ = write!(label, "LEASE {count}");
            render_text(framebuffer, 37, 23, 1, &label);
        }
        NetworkState::Error => render_text(framebuffer, 37, 23, 1, "AP ERROR"),
    }
}

fn render_battery_page(framebuffer: &mut [u8; FRAMEBUFFER_BYTES], battery: BatteryStatus) {
    draw_rect(framebuffer, 3, 5, 40, 22);
    fill_rect(framebuffer, 43, 11, 4, 10);
    let fill = (u32::from(battery.percent) * 34 / 100) as usize;
    fill_rect(framebuffer, 6, 8, fill, 16);
    if battery.charging {
        draw_bolt(framebuffer, 20, 7);
    }

    let mut percent = String::<5>::new();
    let _ = write!(percent, "{}%", battery.percent);
    render_text(framebuffer, 53, 1, 2, &percent);
    render_centered_text(
        framebuffer,
        49,
        WIDTH,
        23,
        1,
        if battery.charging {
            "CHARGING"
        } else {
            "ON BATTERY"
        },
    );
}

fn draw_bolt(framebuffer: &mut [u8; FRAMEBUFFER_BYTES], x: usize, y: usize) {
    for (x0, y0, x1, y1) in [
        (x + 4, y, x, y + 8),
        (x, y + 8, x + 5, y + 8),
        (x + 5, y + 8, x + 1, y + 17),
    ] {
        invert_line(framebuffer, x0 as i16, y0 as i16, x1 as i16, y1 as i16);
    }
}

fn render_text(
    framebuffer: &mut [u8; FRAMEBUFFER_BYTES],
    x: usize,
    y: usize,
    scale: usize,
    text: &str,
) {
    debug_assert!(scale > 0);
    debug_assert!(x + text_pixel_width(text, scale) <= WIDTH);
    debug_assert!(y + 7 * scale <= HEIGHT);
    let mut cursor = x;
    for character in text.bytes() {
        for (glyph_x, column) in glyph(character).into_iter().enumerate() {
            for glyph_y in 0..7 {
                if column & (1 << glyph_y) != 0 {
                    fill_rect(
                        framebuffer,
                        cursor + glyph_x * scale,
                        y + glyph_y * scale,
                        scale,
                        scale,
                    );
                }
            }
        }
        cursor += 6 * scale;
    }
}

fn render_centered_text(
    framebuffer: &mut [u8; FRAMEBUFFER_BYTES],
    x_min: usize,
    x_max: usize,
    y: usize,
    scale: usize,
    text: &str,
) {
    let width = text_pixel_width(text, scale);
    debug_assert!(width <= x_max - x_min);
    render_text(
        framebuffer,
        x_min + (x_max - x_min - width) / 2,
        y,
        scale,
        text,
    );
}

fn text_pixel_width(text: &str, scale: usize) -> usize {
    text.len()
        .checked_mul(6 * scale)
        .unwrap_or(usize::MAX)
        .saturating_sub(scale)
}

fn set_pixel(framebuffer: &mut [u8; FRAMEBUFFER_BYTES], x: i16, y: i16) {
    if x < 0 || y < 0 || x >= WIDTH as i16 || y >= HEIGHT as i16 {
        return;
    }
    let x = x as usize;
    let y = y as usize;
    framebuffer[(y / 8) * WIDTH + x] |= 1 << (y & 7);
}

fn invert_pixel(framebuffer: &mut [u8; FRAMEBUFFER_BYTES], x: i16, y: i16) {
    if x < 0 || y < 0 || x >= WIDTH as i16 || y >= HEIGHT as i16 {
        return;
    }
    let x = x as usize;
    let y = y as usize;
    framebuffer[(y / 8) * WIDTH + x] ^= 1 << (y & 7);
}

fn fill_rect(
    framebuffer: &mut [u8; FRAMEBUFFER_BYTES],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) {
    for py in y..y.saturating_add(height).min(HEIGHT) {
        for px in x..x.saturating_add(width).min(WIDTH) {
            set_pixel(framebuffer, px as i16, py as i16);
        }
    }
}

fn draw_rect(
    framebuffer: &mut [u8; FRAMEBUFFER_BYTES],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) {
    draw_hline(framebuffer, x, y, width);
    draw_hline(framebuffer, x, y + height.saturating_sub(1), width);
    draw_vline(framebuffer, x, y, height);
    draw_vline(framebuffer, x + width.saturating_sub(1), y, height);
}

fn draw_hline(framebuffer: &mut [u8; FRAMEBUFFER_BYTES], x: usize, y: usize, width: usize) {
    fill_rect(framebuffer, x, y, width, 1);
}

fn draw_vline(framebuffer: &mut [u8; FRAMEBUFFER_BYTES], x: usize, y: usize, height: usize) {
    fill_rect(framebuffer, x, y, 1, height);
}

fn draw_line(
    framebuffer: &mut [u8; FRAMEBUFFER_BYTES],
    mut x0: i16,
    mut y0: i16,
    x1: i16,
    y1: i16,
) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        set_pixel(framebuffer, x0, y0);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let twice_error = error * 2;
        if twice_error >= dy {
            error += dy;
            x0 += sx;
        }
        if twice_error <= dx {
            error += dx;
            y0 += sy;
        }
    }
}

fn invert_line(
    framebuffer: &mut [u8; FRAMEBUFFER_BYTES],
    mut x0: i16,
    mut y0: i16,
    x1: i16,
    y1: i16,
) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut error = dx + dy;
    loop {
        invert_pixel(framebuffer, x0, y0);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let twice_error = error * 2;
        if twice_error >= dy {
            error += dy;
            x0 += sx;
        }
        if twice_error <= dx {
            error += dx;
            y0 += sy;
        }
    }
}

fn draw_circle(
    framebuffer: &mut [u8; FRAMEBUFFER_BYTES],
    center_x: i16,
    center_y: i16,
    radius: i16,
) {
    let inner = (radius - 1) * (radius - 1);
    let outer = (radius + 1) * (radius + 1);
    for y in center_y - radius - 1..=center_y + radius + 1 {
        for x in center_x - radius - 1..=center_x + radius + 1 {
            let dx = x - center_x;
            let dy = y - center_y;
            let distance = dx * dx + dy * dy;
            if distance >= inner && distance <= outer {
                set_pixel(framebuffer, x, y);
            }
        }
    }
}

fn draw_arc_top(
    framebuffer: &mut [u8; FRAMEBUFFER_BYTES],
    center_x: i16,
    center_y: i16,
    radius: i16,
) {
    let inner = (radius - 1) * (radius - 1);
    let outer = (radius + 1) * (radius + 1);
    for y in center_y - radius - 1..=center_y {
        for x in center_x - radius - 1..=center_x + radius + 1 {
            let dx = x - center_x;
            let dy = y - center_y;
            let distance = dx * dx + dy * dy;
            if distance >= inner && distance <= outer {
                set_pixel(framebuffer, x, y);
            }
        }
    }
}

fn draw_triangle(
    framebuffer: &mut [u8; FRAMEBUFFER_BYTES],
    x0: i16,
    y0: i16,
    x1: i16,
    y1: i16,
    x2: i16,
    y2: i16,
) {
    draw_line(framebuffer, x0, y0, x1, y1);
    draw_line(framebuffer, x1, y1, x2, y2);
    draw_line(framebuffer, x2, y2, x0, y0);
}

fn draw_octagon(
    framebuffer: &mut [u8; FRAMEBUFFER_BYTES],
    center_x: i16,
    center_y: i16,
    radius: i16,
) {
    let inset = radius / 3;
    let points = [
        (center_x - inset, center_y - radius),
        (center_x + inset, center_y - radius),
        (center_x + radius, center_y - inset),
        (center_x + radius, center_y + inset),
        (center_x + inset, center_y + radius),
        (center_x - inset, center_y + radius),
        (center_x - radius, center_y + inset),
        (center_x - radius, center_y - inset),
    ];
    for index in 0..points.len() {
        let (x0, y0) = points[index];
        let (x1, y1) = points[(index + 1) % points.len()];
        draw_line(framebuffer, x0, y0, x1, y1);
    }
}

#[rustfmt::skip]
fn glyph(character: u8) -> [u8; 5] {
    match character {
        b' ' => [0x00, 0x00, 0x00, 0x00, 0x00],
        b'%' => [0x63, 0x13, 0x08, 0x64, 0x63],
        b'-' => [0x08, 0x08, 0x08, 0x08, 0x08],
        b'/' => [0x40, 0x30, 0x08, 0x06, 0x01],
        b':' => [0x00, 0x36, 0x36, 0x00, 0x00],
        b'0' => [0x3e, 0x51, 0x49, 0x45, 0x3e],
        b'1' => [0x00, 0x42, 0x7f, 0x40, 0x00],
        b'2' => [0x42, 0x61, 0x51, 0x49, 0x46],
        b'3' => [0x21, 0x41, 0x45, 0x4b, 0x31],
        b'4' => [0x18, 0x14, 0x12, 0x7f, 0x10],
        b'5' => [0x27, 0x45, 0x45, 0x45, 0x39],
        b'6' => [0x3c, 0x4a, 0x49, 0x49, 0x30],
        b'7' => [0x01, 0x71, 0x09, 0x05, 0x03],
        b'8' => [0x36, 0x49, 0x49, 0x49, 0x36],
        b'9' => [0x06, 0x49, 0x49, 0x29, 0x1e],
        b'A' => [0x7e, 0x11, 0x11, 0x11, 0x7e],
        b'B' => [0x7f, 0x49, 0x49, 0x49, 0x36],
        b'C' => [0x3e, 0x41, 0x41, 0x41, 0x22],
        b'D' => [0x7f, 0x41, 0x41, 0x22, 0x1c],
        b'E' => [0x7f, 0x49, 0x49, 0x49, 0x41],
        b'F' => [0x7f, 0x09, 0x09, 0x09, 0x01],
        b'G' => [0x3e, 0x41, 0x49, 0x49, 0x7a],
        b'H' => [0x7f, 0x08, 0x08, 0x08, 0x7f],
        b'I' => [0x00, 0x41, 0x7f, 0x41, 0x00],
        b'J' => [0x20, 0x40, 0x41, 0x3f, 0x01],
        b'K' => [0x7f, 0x08, 0x14, 0x22, 0x41],
        b'L' => [0x7f, 0x40, 0x40, 0x40, 0x40],
        b'M' => [0x7f, 0x02, 0x0c, 0x02, 0x7f],
        b'N' => [0x7f, 0x04, 0x08, 0x10, 0x7f],
        b'O' => [0x3e, 0x41, 0x41, 0x41, 0x3e],
        b'P' => [0x7f, 0x09, 0x09, 0x09, 0x06],
        b'Q' => [0x3e, 0x41, 0x51, 0x21, 0x5e],
        b'R' => [0x7f, 0x09, 0x19, 0x29, 0x46],
        b'S' => [0x46, 0x49, 0x49, 0x49, 0x31],
        b'T' => [0x01, 0x01, 0x7f, 0x01, 0x01],
        b'U' => [0x3f, 0x40, 0x40, 0x40, 0x3f],
        b'V' => [0x1f, 0x20, 0x40, 0x20, 0x1f],
        b'W' => [0x3f, 0x40, 0x38, 0x40, 0x3f],
        b'X' => [0x63, 0x14, 0x08, 0x14, 0x63],
        b'Y' => [0x07, 0x08, 0x70, 0x08, 0x07],
        b'Z' => [0x61, 0x51, 0x49, 0x45, 0x43],
        _ => [0x02, 0x01, 0x51, 0x09, 0x06],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normal_status() -> DisplayStatus {
        DisplayStatus {
            runtime_state: RuntimeState::Idle as u8,
            body_state: BodyState::Idle as u8,
            create_power_state: 2,
            oi_mode: 3,
            oi_seen: true,
            oi_fresh: true,
            authority_active: false,
            imu_enabled: true,
            imu_health: ImuHealthCode::Ok as u8,
            last_error: 0,
            wifi_state: WIFI_SERVICES_STARTED,
            network: DisplayNetwork {
                ssid_suffix: Some(1_337_420),
                active_leases: 0,
            },
            battery: Some(BatteryStatus {
                percent: 73,
                charging: false,
            }),
            battery_stale: false,
        }
    }

    fn no_safety() -> DisplaySafety {
        DisplaySafety {
            estop_latched: false,
            safety_latch_kind: None,
        }
    }

    fn assert_lines(actual: DisplayPage, line1: &str, line2: &str) {
        assert_eq!(actual.line1.as_str(), line1);
        assert_eq!(actual.line2.as_str(), line2);
    }

    #[test]
    fn normal_pages_prioritize_large_status_and_real_battery() {
        let status = normal_status();
        assert_lines(status.page(no_safety(), 0), "PETE  READY", "CTRL OPEN");
        assert_lines(
            status.page(no_safety(), PAGE_ROTATION_MS),
            "PETE  READY",
            "CTRL OPEN",
        );
        assert_lines(
            status.page(no_safety(), PAGE_ROTATION_MS * 2),
            "BATT 73%",
            "ON BATTERY",
        );
    }

    #[test]
    fn active_control_authority_replaces_the_normal_imu_cell() {
        let mut status = normal_status();
        status.authority_active = true;
        assert_lines(status.page(no_safety(), 0), "PETE  READY", "CTRL ACTIVE");
    }

    #[test]
    fn network_failure_rotates_as_secondary_instead_of_monopolizing() {
        let mut status = normal_status();
        status.wifi_state = WIFI_ERROR;
        assert_lines(status.page(no_safety(), 0), "PETE  READY", "CTRL OPEN");
        assert_lines(
            status.page(no_safety(), PAGE_ROTATION_MS),
            "pete-snyk",
            "192.168.4.1 ERROR",
        );
        assert_lines(
            status.page(no_safety(), PAGE_ROTATION_MS * 2),
            "BATT 73%",
            "ON BATTERY",
        );
    }

    #[test]
    fn charging_is_a_persistent_positive_page() {
        let mut status = normal_status();
        status.battery = Some(BatteryStatus {
            percent: 73,
            charging: true,
        });
        for now_ms in [0, PAGE_ROTATION_MS, PAGE_ROTATION_MS * 2] {
            assert_lines(status.page(no_safety(), now_ms), "BATT 73%", "CHARGING");
        }
    }

    #[test]
    fn safety_and_fault_pages_have_stable_priority() {
        let mut status = normal_status();
        status.oi_fresh = false;
        status.battery = Some(BatteryStatus {
            percent: 10,
            charging: true,
        });
        status.imu_health = ImuHealthCode::Absent as u8;

        assert_lines(
            status.page(
                DisplaySafety {
                    estop_latched: true,
                    safety_latch_kind: Some(SafetyEventKind::Tilt),
                },
                0,
            ),
            "ESTOP",
            "",
        );
        assert_lines(
            status.page(
                DisplaySafety {
                    estop_latched: false,
                    safety_latch_kind: Some(SafetyEventKind::Impact),
                },
                0,
            ),
            "IMPACT",
            "",
        );
        assert_lines(status.page(no_safety(), 0), "OI LINK", "LOST");
    }

    #[test]
    fn invalid_or_missing_battery_never_creates_a_battery_page() {
        let mut status = normal_status();
        status.battery = None;
        assert_lines(
            status.page(no_safety(), PAGE_ROTATION_MS * 2),
            "PETE  READY",
            "CTRL OPEN",
        );
    }

    #[test]
    fn stale_battery_uses_a_full_alert_after_a_jitter_tolerant_window() {
        let mut snapshot = status::snapshot(10_000);
        snapshot.uptime_ms = 10_000;
        snapshot.create_sensor_complete_packet_count = 1;
        snapshot.create_sensor_capacity_mah = 100;
        snapshot.create_sensor_charge_mah = 10;
        snapshot.create_sensor_last_complete_packet_timestamp_ms =
            snapshot.uptime_ms - BATTERY_FRESHNESS_MS + 1;
        let network = DisplayNetwork {
            ssid_suffix: None,
            active_leases: 0,
        };

        let fresh = DisplayStatus::from_snapshot(&snapshot, network);
        assert_eq!(fresh.battery.map(|battery| battery.percent), Some(10));
        assert!(!fresh.battery_stale);

        snapshot.create_sensor_last_complete_packet_timestamp_ms =
            snapshot.uptime_ms - BATTERY_FRESHNESS_MS - 1;
        let stale = DisplayStatus::from_snapshot(&snapshot, network);
        assert_eq!(stale.battery, None);
        assert!(stale.battery_stale);
        assert_lines(stale.page(no_safety(), 0), "BATT", "STALE");
    }

    #[test]
    fn low_battery_and_offline_imu_use_existing_health_conditions() {
        let mut status = normal_status();
        status.battery = Some(BatteryStatus {
            percent: LOW_BATTERY_PERCENT as u8,
            charging: false,
        });
        assert_lines(status.page(no_safety(), 0), "LOW", "BATT");

        status.battery = Some(BatteryStatus {
            percent: 21,
            charging: false,
        });
        status.imu_health = ImuHealthCode::Fault as u8;
        assert_lines(status.page(no_safety(), 0), "IMU", "OFFLINE");
    }

    #[test]
    fn moving_and_passive_states_render_run_and_stop() {
        let mut status = normal_status();
        status.body_state = BodyState::Moving as u8;
        assert_lines(status.page(no_safety(), 0), "PETE  RUN", "CTRL OPEN");

        status.body_state = BodyState::Idle as u8;
        status.oi_mode = 1;
        assert_lines(status.page(no_safety(), 0), "PETE  STOP", "CTRL OPEN");
    }

    #[test]
    fn startup_create_and_runtime_diagnostics_are_explicit() {
        let mut status = normal_status();
        status.oi_seen = false;
        status.oi_fresh = false;
        status.body_state = BodyState::WaitingForCreate as u8;
        assert_lines(
            status.page(no_safety(), 0),
            "pete-snyk",
            "192.168.4.1 READY",
        );
        assert_lines(status.page(no_safety(), PAGE_ROTATION_MS), "WAIT", "CREATE");

        status.create_power_state = CREATE_POWER_OFF;
        assert_lines(status.page(no_safety(), 0), "POWER", "OFF");

        status.create_power_state = 2;
        status.runtime_state = RuntimeState::Error as u8;
        status.body_state = BodyState::Error as u8;
        for (error, expected) in [
            (ERROR_CREATE_NO_RESPONSE, ("OI NO", "RX")),
            (ERROR_UART_FRAMING, ("UART", "FRAME")),
            (ERROR_TIMEOUT, ("TIME", "OUT")),
            (ERROR_INVALID_PACKET, ("BAD", "PACKET")),
            (0, ("RUNTIME", "ERROR")),
        ] {
            status.last_error = error;
            assert_lines(status.page(no_safety(), 0), expected.0, expected.1);
        }
    }

    #[test]
    fn every_safety_latch_category_has_its_own_alert() {
        let status = normal_status();
        for (kind, expected) in [
            (SafetyEventKind::Bump, ("BUMP", "")),
            (SafetyEventKind::Cliff, ("CLIFF", "")),
            (SafetyEventKind::WheelDrop, ("WHEEL", "DROP")),
            (SafetyEventKind::EStop, ("ESTOP", "")),
            (SafetyEventKind::Heartbeat, ("CTRL", "LOST")),
            (SafetyEventKind::Tilt, ("TILT", "")),
            (SafetyEventKind::Impact, ("IMPACT", "")),
            (SafetyEventKind::Charging, ("NO", "DRIVE")),
        ] {
            assert_lines(
                status.page(
                    DisplaySafety {
                        estop_latched: false,
                        safety_latch_kind: Some(kind),
                    },
                    0,
                ),
                expected.0,
                expected.1,
            );
        }
    }

    #[test]
    fn network_page_reports_startup_readiness_and_active_dhcp_leases() {
        let mut status = normal_status();
        status.wifi_state = 1;
        assert_lines(network_page(status), "pete-snyk", "192.168.4.1 START");
        status.wifi_state = WIFI_SERVICES_STARTED;
        status.network.active_leases = 2;
        assert_lines(network_page(status), "pete-snyk", "192.168.4.1 LEASE 2");
        status.wifi_state = WIFI_ERROR;
        assert_lines(network_page(status), "pete-snyk", "192.168.4.1 ERROR");
    }

    #[test]
    fn normal_status_uses_both_double_height_text_bands() {
        let dashboard = render(&normal_status().page(no_safety(), 0));
        assert!(dashboard[..WIDTH * 2].iter().any(|byte| *byte != 0));
        assert!(dashboard[WIDTH * 2..].iter().any(|byte| *byte != 0));
    }

    #[test]
    fn liveness_pixel_toggles_without_changing_the_selected_page() {
        let status = normal_status();
        let off_page = status.page(no_safety(), 0);
        let on_page = status.page(no_safety(), LIVENESS_TOGGLE_MS);
        assert_eq!(off_page.line1, on_page.line1);
        assert_eq!(off_page.line2, on_page.line2);

        let off = render(&off_page);
        let on = render(&on_page);
        let differences = off
            .iter()
            .zip(on.iter())
            .filter(|(left, right)| left != right)
            .count();
        assert_eq!(differences, 1);
        assert_eq!(off[FRAMEBUFFER_BYTES - 1] ^ on[FRAMEBUFFER_BYTES - 1], 0x80);
    }

    fn framebuffer_hash(framebuffer: &[u8; FRAMEBUFFER_BYTES]) -> u64 {
        framebuffer
            .iter()
            .fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
            })
    }

    #[test]
    fn every_page_and_alert_matches_its_framebuffer_snapshot() {
        let status = normal_status();
        let mut boot = status;
        boot.runtime_state = RuntimeState::Booting as u8;
        let mut run = status;
        run.body_state = BodyState::Moving as u8;
        let mut stop = status;
        stop.oi_mode = 1;
        let mut warn = status;
        warn.runtime_state = RuntimeState::Error as u8;
        let mut controlled = status;
        controlled.authority_active = true;

        let network = |state| {
            page(
                "network",
                "",
                DisplayLayout::Network(NetworkStatus {
                    ssid_suffix: Some(1_337_420),
                    state,
                }),
            )
        };
        let alert = |icon| alert_page(icon);

        let pages = [
            ("dashboard_boot", health_page(boot)),
            ("dashboard_ready", health_page(status)),
            ("dashboard_run", health_page(run)),
            ("dashboard_stop", health_page(stop)),
            ("dashboard_warn", health_page(warn)),
            ("dashboard_controlled", health_page(controlled)),
            ("network_start", network(NetworkState::Starting)),
            ("network_ready", network(NetworkState::Ready)),
            ("network_lease", network(NetworkState::Lease(2))),
            ("network_error", network(NetworkState::Error)),
            (
                "battery",
                battery_page(BatteryStatus {
                    percent: 73,
                    charging: false,
                }),
            ),
            (
                "battery_charging",
                battery_page(BatteryStatus {
                    percent: 42,
                    charging: true,
                }),
            ),
            ("alert_bump", alert(AlertIcon::Bump)),
            ("alert_cliff", alert(AlertIcon::Cliff)),
            ("alert_wheel_drop", alert(AlertIcon::WheelDrop)),
            ("alert_estop", alert(AlertIcon::EStop)),
            ("alert_heartbeat", alert(AlertIcon::Heartbeat)),
            ("alert_tilt", alert(AlertIcon::Tilt)),
            ("alert_impact", alert(AlertIcon::Impact)),
            ("alert_charging", alert(AlertIcon::Charging)),
            ("alert_oi_link_lost", alert(AlertIcon::OiLinkLost)),
            ("alert_low_battery", alert(AlertIcon::LowBattery)),
            ("alert_battery_stale", alert(AlertIcon::BatteryStale)),
            ("alert_imu_offline", alert(AlertIcon::ImuOffline)),
            ("alert_wait_create", alert(AlertIcon::WaitCreate)),
            ("alert_power_off", alert(AlertIcon::PowerOff)),
            ("alert_create_no_rx", alert(AlertIcon::CreateNoRx)),
            ("alert_uart_framing", alert(AlertIcon::UartFraming)),
            ("alert_timeout", alert(AlertIcon::Timeout)),
            ("alert_invalid_packet", alert(AlertIcon::InvalidPacket)),
            ("alert_runtime_error", alert(AlertIcon::RuntimeError)),
        ];
        let expected = [
            0x7622_ed05_97b5_baf9,
            0x6742_ded5_5bfd_3ec3,
            0x7b1e_718d_b98d_10b9,
            0x24cc_85d2_5530_7bcf,
            0x5c33_5285_f482_079f,
            0x7fb9_312a_8a87_e61b,
            0xf7fe_64e2_9714_b28a,
            0xa1dd_390d_7fb4_0abc,
            0xfdab_7d1d_144f_86ad,
            0xc4b7_40fa_05d5_cd0e,
            0x945d_9d1a_02ec_af46,
            0xba9c_d6a8_f0d7_c0c1,
            0x12da_7ac2_b02f_ace2,
            0xb2e4_dee8_a08e_e256,
            0x6dfc_49f2_e1a4_c0ae,
            0xf182_9cdf_47ea_8586,
            0x3445_eed7_4e17_2226,
            0x05e4_a5b4_c9ab_35e6,
            0xc89b_9d26_35db_ea71,
            0x103e_5f07_8e33_c377,
            0x7900_5182_a418_2ed1,
            0x7742_6561_91ae_f3ec,
            0x8041_dfd5_0ce6_efec,
            0xfb71_5d25_25ab_1437,
            0x7daf_f066_7c51_80c6,
            0xc5ab_0cb8_2e95_be74,
            0x2811_ab0f_13a6_90d0,
            0x11f9_c1bd_36cf_e7bf,
            0x9554_cf17_f885_8c5a,
            0xfdc4_18ad_20c8_9eb4,
            0x2bfb_4a6d_1c28_fda2,
        ];
        let mut mismatches = 0;
        for ((name, page), expected_hash) in pages.iter().zip(expected) {
            let framebuffer = render(page);
            let actual_hash = framebuffer_hash(&framebuffer);
            if actual_hash != expected_hash {
                std::eprintln!("{name}: 0x{actual_hash:016x}");
                mismatches += 1;
            }
            assert!(
                framebuffer[..WIDTH * 3].iter().any(|byte| *byte != 0)
                    && framebuffer[WIDTH * 3..].iter().any(|byte| *byte != 0),
                "{name} must use both the upper and lower display bands"
            );
        }
        assert_eq!(mismatches, 0, "framebuffer snapshots changed");
    }
}
