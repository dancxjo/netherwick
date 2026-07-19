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
pub const LINK_FRESHNESS_MS: u32 = 1_000;
pub const LOW_BATTERY_PERCENT: u32 = 20;

const LINE_CAPACITY: usize = 22;

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
    oi_mode: u8,
    oi_seen: bool,
    oi_fresh: bool,
    imu_enabled: bool,
    imu_health: u8,
    battery: Option<BatteryStatus>,
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DisplayLayout {
    Dashboard {
        state: StateIcon,
        oi: HealthIcon,
        imu: HealthIcon,
        battery: Option<BatteryStatus>,
    },
    Alert(AlertIcon),
    Battery(BatteryStatus),
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
enum HealthIcon {
    Unknown,
    Ok,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AlertIcon {
    TiltImpact,
    EStop,
    OiLinkLost,
    LowBattery,
    ImuOffline,
}

impl DisplayStatus {
    pub fn from_snapshot(snapshot: &BrainstemStatus) -> Self {
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
                <= LINK_FRESHNESS_MS;
        let battery = battery_fresh.then(|| battery_status(snapshot)).flatten();

        Self {
            runtime_state: snapshot.current_runtime_state,
            body_state: snapshot.body_state,
            oi_mode: snapshot.oi_mode,
            oi_seen,
            oi_fresh,
            imu_enabled: crate::body::IMU_ENABLED,
            imu_health: snapshot.imu_health,
            battery,
        }
    }

    pub fn page(self, safety: DisplaySafety, now_ms: u32) -> DisplayPage {
        if safety.estop_latched {
            return warning_page("ESTOP");
        }
        if matches!(
            safety.safety_latch_kind,
            Some(SafetyEventKind::Tilt | SafetyEventKind::Impact)
        ) {
            return warning_page("TILT / IMPACT");
        }
        if self.oi_seen && !self.oi_fresh {
            return warning_page("OI LINK LOST");
        }
        if self
            .battery
            .is_some_and(|battery| u32::from(battery.percent) <= LOW_BATTERY_PERCENT)
        {
            return warning_page("LOW BATT");
        }
        if self.imu_enabled
            && matches!(
                self.imu_health,
                x if x == ImuHealthCode::Fault as u8 || x == ImuHealthCode::Absent as u8
            )
        {
            return warning_page("IMU OFFLINE");
        }

        if (now_ms / PAGE_ROTATION_MS) & 1 == 1 {
            if let Some(battery) = self.battery {
                return battery_page(battery);
            }
        }

        health_page(self)
    }
}

pub fn render(page: &DisplayPage) -> [u8; FRAMEBUFFER_BYTES] {
    let mut framebuffer = [0u8; FRAMEBUFFER_BYTES];
    match page.layout {
        DisplayLayout::Dashboard {
            state,
            oi,
            imu,
            battery,
        } => render_dashboard(&mut framebuffer, state, oi, imu, battery),
        DisplayLayout::Alert(alert) => render_alert(&mut framebuffer, alert),
        DisplayLayout::Battery(battery) => render_battery_page(&mut framebuffer, battery),
    }
    framebuffer
}

fn warning_page(reason: &str) -> DisplayPage {
    let alert = match reason {
        "ESTOP" => AlertIcon::EStop,
        "TILT / IMPACT" => AlertIcon::TiltImpact,
        "OI LINK LOST" => AlertIcon::OiLinkLost,
        "LOW BATT" => AlertIcon::LowBattery,
        "IMU OFFLINE" => AlertIcon::ImuOffline,
        _ => AlertIcon::EStop,
    };
    page("PETE  WARN", reason, DisplayLayout::Alert(alert))
}

fn battery_page(battery: BatteryStatus) -> DisplayPage {
    let mut line1 = String::new();
    let mut line2 = String::new();
    let _ = write!(line1, "BATT {}%", battery.percent);
    let _ = write!(
        line2,
        "CHARGING: {}",
        if battery.charging { "YES" } else { "NO" }
    );
    DisplayPage {
        line1,
        line2,
        layout: DisplayLayout::Battery(battery),
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
    let oi = if status.oi_fresh { "OK" } else { "--" };
    let imu = if !status.imu_enabled {
        "--"
    } else if status.imu_health == ImuHealthCode::Ok as u8 {
        "OK"
    } else {
        "--"
    };
    let _ = write!(line2, "OI {oi}  IMU {imu}");
    DisplayPage {
        line1,
        line2,
        layout: DisplayLayout::Dashboard {
            state: state_icon,
            oi: if status.oi_fresh {
                HealthIcon::Ok
            } else {
                HealthIcon::Unknown
            },
            imu: if status.imu_enabled && status.imu_health == ImuHealthCode::Ok as u8 {
                HealthIcon::Ok
            } else {
                HealthIcon::Unknown
            },
            battery: status.battery,
        },
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
    };
    let _ = result.line1.push_str(line1);
    let _ = result.line2.push_str(line2);
    result
}

fn render_dashboard(
    framebuffer: &mut [u8; FRAMEBUFFER_BYTES],
    state: StateIcon,
    oi: HealthIcon,
    imu: HealthIcon,
    battery: Option<BatteryStatus>,
) {
    for x in [32, 64, 96] {
        draw_vline(framebuffer, x, 3, 25);
    }
    render_state_icon(framebuffer, state);
    render_oi_icon(framebuffer, oi);
    render_imu_icon(framebuffer, imu);
    render_battery_icon(framebuffer, battery);
}

fn render_state_icon(framebuffer: &mut [u8; FRAMEBUFFER_BYTES], state: StateIcon) {
    match state {
        StateIcon::Boot => {
            draw_circle(framebuffer, 15, 11, 5);
            for (x0, y0, x1, y1) in [
                (15, 1, 15, 4),
                (15, 18, 15, 21),
                (5, 11, 8, 11),
                (22, 11, 25, 11),
                (8, 4, 10, 6),
                (20, 16, 22, 18),
                (8, 18, 10, 16),
                (20, 6, 22, 4),
            ] {
                draw_line(framebuffer, x0, y0, x1, y1);
            }
            render_text(framebuffer, 4, 24, 1, "BOOT");
        }
        StateIcon::Ready => {
            draw_circle(framebuffer, 15, 11, 10);
            draw_check(framebuffer, 8, 7, 1);
            render_text(framebuffer, 1, 24, 1, "READY");
        }
        StateIcon::Run => {
            fill_rect(framebuffer, 5, 8, 14, 7);
            fill_triangle_right(framebuffer, 19, 4, 10);
            render_text(framebuffer, 7, 24, 1, "RUN");
        }
        StateIcon::Stop => {
            fill_rect(framebuffer, 7, 3, 17, 17);
            render_text(framebuffer, 4, 24, 1, "STOP");
        }
        StateIcon::Warn => {
            draw_triangle(framebuffer, 15, 1, 3, 21, 27, 21);
            draw_vline(framebuffer, 15, 7, 7);
            fill_rect(framebuffer, 14, 17, 3, 3);
            render_text(framebuffer, 4, 24, 1, "WARN");
        }
    }
}

fn render_oi_icon(framebuffer: &mut [u8; FRAMEBUFFER_BYTES], health: HealthIcon) {
    draw_rect(framebuffer, 42, 5, 13, 11);
    draw_vline(framebuffer, 45, 1, 4);
    draw_vline(framebuffer, 51, 1, 4);
    draw_vline(framebuffer, 48, 16, 5);
    draw_line(framebuffer, 48, 20, 43, 22);
    draw_health_badge(framebuffer, 54, 4, health);
    render_text(framebuffer, 41, 24, 1, "OI");
}

fn render_imu_icon(framebuffer: &mut [u8; FRAMEBUFFER_BYTES], health: HealthIcon) {
    draw_rect(framebuffer, 72, 3, 17, 17);
    draw_line(framebuffer, 75, 14, 80, 9);
    draw_line(framebuffer, 80, 9, 86, 14);
    draw_circle(framebuffer, 80, 12, 2);
    draw_health_badge(framebuffer, 86, 4, health);
    render_text(framebuffer, 71, 24, 1, "IMU");
}

fn render_battery_icon(framebuffer: &mut [u8; FRAMEBUFFER_BYTES], battery: Option<BatteryStatus>) {
    draw_rect(framebuffer, 101, 5, 22, 14);
    fill_rect(framebuffer, 123, 9, 3, 6);
    if let Some(battery) = battery {
        let fill = (u32::from(battery.percent) * 18 / 100) as usize;
        fill_rect(framebuffer, 103, 7, fill, 10);
        if battery.charging {
            draw_bolt(framebuffer, 111, 6);
        }
        let mut label = String::<5>::new();
        let _ = write!(label, "{}%", battery.percent);
        let width = label.len() * 6 - 1;
        render_text(
            framebuffer,
            112usize.saturating_sub(width / 2),
            24,
            1,
            &label,
        );
    } else {
        render_text(framebuffer, 109, 8, 1, "?");
        render_text(framebuffer, 105, 24, 1, "BATT");
    }
}

fn render_alert(framebuffer: &mut [u8; FRAMEBUFFER_BYTES], alert: AlertIcon) {
    draw_vline(framebuffer, 39, 2, 28);
    match alert {
        AlertIcon::EStop => {
            draw_octagon(framebuffer, 19, 15, 13);
            draw_vline(framebuffer, 19, 7, 10);
            fill_rect(framebuffer, 18, 21, 3, 3);
            render_text(framebuffer, 51, 9, 2, "ESTOP");
        }
        AlertIcon::TiltImpact => {
            draw_rect(framebuffer, 8, 8, 18, 14);
            draw_line(framebuffer, 7, 22, 28, 4);
            draw_line(framebuffer, 7, 4, 30, 25);
            render_text(framebuffer, 46, 1, 2, "TILT");
            render_text(framebuffer, 46, 17, 2, "IMPACT");
        }
        AlertIcon::OiLinkLost => {
            draw_rect(framebuffer, 8, 7, 11, 10);
            draw_rect(framebuffer, 22, 7, 9, 10);
            draw_line(framebuffer, 17, 12, 22, 12);
            draw_line(framebuffer, 5, 3, 34, 25);
            render_text(framebuffer, 43, 1, 2, "OI LINK");
            render_text(framebuffer, 55, 17, 2, "LOST");
        }
        AlertIcon::LowBattery => {
            draw_rect(framebuffer, 5, 8, 27, 16);
            fill_rect(framebuffer, 32, 13, 3, 6);
            fill_rect(framebuffer, 8, 11, 4, 10);
            render_text(framebuffer, 49, 1, 2, "LOW");
            render_text(framebuffer, 43, 17, 2, "BATT");
        }
        AlertIcon::ImuOffline => {
            draw_rect(framebuffer, 8, 6, 20, 20);
            draw_line(framebuffer, 11, 20, 18, 12);
            draw_line(framebuffer, 18, 12, 25, 20);
            draw_line(framebuffer, 5, 3, 33, 28);
            render_text(framebuffer, 49, 1, 2, "IMU");
            render_text(framebuffer, 43, 17, 2, "OFFLINE");
        }
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
    render_text(
        framebuffer,
        59,
        23,
        1,
        if battery.charging {
            "CHG YES"
        } else {
            "CHG NO"
        },
    );
}

fn draw_health_badge(
    framebuffer: &mut [u8; FRAMEBUFFER_BYTES],
    x: usize,
    y: usize,
    health: HealthIcon,
) {
    match health {
        HealthIcon::Ok => draw_check(framebuffer, x, y, 1),
        HealthIcon::Unknown => draw_hline(framebuffer, x, y + 6, 7),
    }
}

fn draw_check(framebuffer: &mut [u8; FRAMEBUFFER_BYTES], x: usize, y: usize, scale: usize) {
    draw_line(
        framebuffer,
        x as i16,
        (y + 4 * scale) as i16,
        (x + 3 * scale) as i16,
        (y + 7 * scale) as i16,
    );
    draw_line(
        framebuffer,
        (x + 3 * scale) as i16,
        (y + 7 * scale) as i16,
        (x + 9 * scale) as i16,
        y as i16,
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

fn fill_triangle_right(framebuffer: &mut [u8; FRAMEBUFFER_BYTES], x: usize, y: usize, size: usize) {
    for column in 0..size {
        let half_height = size.saturating_sub(column) / 2;
        draw_vline(
            framebuffer,
            x + column,
            y + size / 2 - half_height,
            half_height * 2 + 1,
        );
    }
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
            oi_mode: 3,
            oi_seen: true,
            oi_fresh: true,
            imu_enabled: true,
            imu_health: ImuHealthCode::Ok as u8,
            battery: Some(BatteryStatus {
                percent: 73,
                charging: false,
            }),
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
    fn normal_page_rotates_to_real_battery_every_three_seconds() {
        let status = normal_status();
        assert_lines(status.page(no_safety(), 0), "PETE  READY", "OI OK  IMU OK");
        assert_lines(
            status.page(no_safety(), PAGE_ROTATION_MS),
            "BATT 73%",
            "CHARGING: NO",
        );
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
            "PETE  WARN",
            "ESTOP",
        );
        assert_lines(
            status.page(
                DisplaySafety {
                    estop_latched: false,
                    safety_latch_kind: Some(SafetyEventKind::Impact),
                },
                0,
            ),
            "PETE  WARN",
            "TILT / IMPACT",
        );
        assert_lines(status.page(no_safety(), 0), "PETE  WARN", "OI LINK LOST");
    }

    #[test]
    fn invalid_or_missing_battery_never_creates_a_battery_page() {
        let mut status = normal_status();
        status.battery = None;
        assert_lines(
            status.page(no_safety(), PAGE_ROTATION_MS),
            "PETE  READY",
            "OI OK  IMU OK",
        );
    }

    #[test]
    fn low_battery_and_offline_imu_use_existing_health_conditions() {
        let mut status = normal_status();
        status.battery = Some(BatteryStatus {
            percent: LOW_BATTERY_PERCENT as u8,
            charging: false,
        });
        assert_lines(status.page(no_safety(), 0), "PETE  WARN", "LOW BATT");

        status.battery = Some(BatteryStatus {
            percent: 21,
            charging: false,
        });
        status.imu_health = ImuHealthCode::Fault as u8;
        assert_lines(status.page(no_safety(), 0), "PETE  WARN", "IMU OFFLINE");
    }

    #[test]
    fn moving_and_passive_states_render_run_and_stop() {
        let mut status = normal_status();
        status.body_state = BodyState::Moving as u8;
        assert_lines(status.page(no_safety(), 0), "PETE  RUN", "OI OK  IMU OK");

        status.body_state = BodyState::Idle as u8;
        status.oi_mode = 1;
        assert_lines(status.page(no_safety(), 0), "PETE  STOP", "OI OK  IMU OK");
    }

    #[test]
    fn renderer_uses_all_four_icon_cells_and_large_alert_text() {
        let dashboard = render(&normal_status().page(no_safety(), 0));
        for cell in 0..4 {
            assert!(
                (0..HEIGHT).any(|y| dashboard[(y / 8) * WIDTH + cell * 32] != 0
                    || dashboard[(y / 8) * WIDTH + cell * 32 + 16] != 0),
                "dashboard cell {cell} should contain a visible icon"
            );
        }

        let alert = render(&warning_page("TILT / IMPACT"));
        assert!(alert.iter().filter(|byte| **byte != 0).count() > 80);
    }
}
