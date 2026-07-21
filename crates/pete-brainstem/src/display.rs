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
#[path = "display_tests.rs"]
mod tests;
