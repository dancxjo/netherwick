use std::{env, fs, path::PathBuf};

use serde::Deserialize;

#[allow(dead_code)]
#[derive(Deserialize)]
struct BodyToml {
    body: Body,
    create_oi: CreateOi,
    timing: Timing,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct BoardToml {
    board: Board,
    rp2040: Rp2040,
    pins: Pins,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Body {
    name: String,
    kind: String,
    drive: String,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct CreateOi {
    baud: u32,
    data_bits: u8,
    stop_bits: u8,
    default_mode: String,
    sensor_probe_packet: u8,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Timing {
    power_toggle_pulse_ms: u32,
    wake_wait_ms: u32,
    responsive_timeout_ms: u32,
    brc_low_pulse_ms: u32,
    post_brc_settle_ms: u32,
    post_start_settle_ms: u32,
    post_mode_settle_ms: u32,
    uart_byte_timeout_ms: u32,
    idle_blink_ms: u32,
    error_blink_ms: u32,
    error_pause_ms: u32,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Board {
    name: String,
    arch: String,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Rp2040 {
    xosc_crystal_freq_hz: u32,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Pins {
    create_uart: CreateUartPins,
    power_toggle: GpioPin,
    create_brc: OptionalGpioPin,
    leds: LedPins,
    create_device_detect: OptionalGpioPin,
    estop: OptionalGpioPin,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct CreateUartPins {
    tx: String,
    tx_gpio: u8,
    tx_physical_pin: u8,
    rx: String,
    rx_gpio: u8,
    rx_physical_pin: u8,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct GpioPin {
    pin: String,
    gpio: u8,
    physical_pin: Option<u8>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct OptionalGpioPin {
    enabled: bool,
    pin: String,
    gpio: u8,
    physical_pin: Option<u8>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct LedPins {
    onboard: String,
    onboard_gpio: u8,
    status: String,
    status_gpio: u8,
    status_physical_pin: Option<u8>,
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let body_path = manifest_dir.join("body.toml");
    let board_path = manifest_dir.join("board.toml");
    println!("cargo:rerun-if-changed={}", body_path.display());
    println!("cargo:rerun-if-changed={}", board_path.display());

    let body: BodyToml = toml::from_str(&fs::read_to_string(&body_path).unwrap()).unwrap();
    let board: BoardToml = toml::from_str(&fs::read_to_string(&board_path).unwrap()).unwrap();
    assert_eq!(body.body.kind, "create_oi");
    assert_eq!(body.body.drive, "differential");
    assert_eq!(body.create_oi.data_bits, 8);
    assert_eq!(body.create_oi.stop_bits, 1);
    assert_eq!(board.board.arch, "rp2040");

    let default_mode = match body.create_oi.default_mode.as_str() {
        "passive" => "CreateOiMode::Passive",
        "safe" => "CreateOiMode::Safe",
        "full" => "CreateOiMode::Full",
        other => panic!("unsupported create_oi.default_mode: {other}"),
    };

    let generated = format!(
        r#"pub const BODY_NAME: &str = {body_name:?};
pub const BOARD_NAME: &str = {board_name:?};
pub const BODY_KIND: BodyKind = BodyKind::CreateOpenInterface;
pub const DRIVE_KIND: DriveKind = DriveKind::Differential;

pub const CREATE_UART_BAUD: u32 = {baud};
pub const CREATE_DEFAULT_MODE: CreateOiMode = {default_mode};
pub const CREATE_SENSOR_PROBE_PACKET: u8 = {sensor_probe_packet};
pub const CREATE_BRC_ENABLED: bool = {create_brc_enabled};

pub const POWER_TOGGLE_PULSE_MS: u32 = {power_toggle_pulse_ms};
pub const CREATE_WAKE_WAIT_MS: u32 = {wake_wait_ms};
pub const CREATE_RESPONSIVE_TIMEOUT_MS: u32 = {responsive_timeout_ms};
pub const BRC_LOW_PULSE_MS: u32 = {brc_low_pulse_ms};
pub const POST_BRC_SETTLE_MS: u32 = {post_brc_settle_ms};
pub const POST_START_SETTLE_MS: u32 = {post_start_settle_ms};
pub const POST_MODE_SETTLE_MS: u32 = {post_mode_settle_ms};
pub const UART_BYTE_TIMEOUT_MS: u32 = {uart_byte_timeout_ms};
pub const IDLE_BLINK_MS: u32 = {idle_blink_ms};
pub const ERROR_BLINK_MS: u32 = {error_blink_ms};
pub const ERROR_PAUSE_MS: u32 = {error_pause_ms};

pub const XOSC_CRYSTAL_FREQ_HZ: u32 = {xosc_crystal_freq_hz};

pub const CREATE_TX_GPIO: u8 = {create_tx_gpio};
pub const CREATE_TX_PHYSICAL_PIN: u8 = {create_tx_physical_pin};
pub const CREATE_RX_GPIO: u8 = {create_rx_gpio};
pub const CREATE_RX_PHYSICAL_PIN: u8 = {create_rx_physical_pin};
pub const ONBOARD_LED_GPIO: u8 = {onboard_led_gpio};
pub const ONBOARD_LED_PIN: &str = {onboard_led_pin:?};
pub const CREATE_POWER_TOGGLE_PIN: &str = {create_power_toggle_pin:?};
pub const CREATE_POWER_TOGGLE_GPIO: u8 = {create_power_toggle_gpio};
pub const CREATE_BRC_PIN: &str = {create_brc_pin:?};
pub const CREATE_BRC_GPIO: u8 = {create_brc_gpio};
pub const EXTERNAL_LED_GPIO: u8 = {external_led_gpio};
pub const EXTERNAL_LED_PIN: &str = {external_led_pin:?};
pub const CREATE_DEVICE_DETECT_ENABLED: bool = {create_device_detect_enabled};
pub const CREATE_DEVICE_DETECT_PIN: &str = {create_device_detect_pin:?};
pub const CREATE_DEVICE_DETECT_GPIO: u8 = {create_device_detect_gpio};
pub const ESTOP_ENABLED: bool = {estop_enabled};
pub const ESTOP_PIN: &str = {estop_pin:?};
pub const ESTOP_GPIO: u8 = {estop_gpio};
"#,
        body_name = body.body.name,
        board_name = board.board.name,
        baud = body.create_oi.baud,
        default_mode = default_mode,
        sensor_probe_packet = body.create_oi.sensor_probe_packet,
        create_brc_enabled = board.pins.create_brc.enabled,
        power_toggle_pulse_ms = body.timing.power_toggle_pulse_ms,
        wake_wait_ms = body.timing.wake_wait_ms,
        responsive_timeout_ms = body.timing.responsive_timeout_ms,
        brc_low_pulse_ms = body.timing.brc_low_pulse_ms,
        post_brc_settle_ms = body.timing.post_brc_settle_ms,
        post_start_settle_ms = body.timing.post_start_settle_ms,
        post_mode_settle_ms = body.timing.post_mode_settle_ms,
        uart_byte_timeout_ms = body.timing.uart_byte_timeout_ms,
        idle_blink_ms = body.timing.idle_blink_ms,
        error_blink_ms = body.timing.error_blink_ms,
        error_pause_ms = body.timing.error_pause_ms,
        xosc_crystal_freq_hz = board.rp2040.xosc_crystal_freq_hz,
        create_tx_gpio = board.pins.create_uart.tx_gpio,
        create_tx_physical_pin = board.pins.create_uart.tx_physical_pin,
        create_rx_gpio = board.pins.create_uart.rx_gpio,
        create_rx_physical_pin = board.pins.create_uart.rx_physical_pin,
        onboard_led_gpio = board.pins.leds.onboard_gpio,
        onboard_led_pin = board.pins.leds.onboard,
        create_power_toggle_pin = board.pins.power_toggle.pin,
        create_power_toggle_gpio = board.pins.power_toggle.gpio,
        create_brc_pin = board.pins.create_brc.pin,
        create_brc_gpio = board.pins.create_brc.gpio,
        external_led_gpio = board.pins.leds.status_gpio,
        external_led_pin = board.pins.leds.status,
        create_device_detect_enabled = board.pins.create_device_detect.enabled,
        create_device_detect_pin = board.pins.create_device_detect.pin,
        create_device_detect_gpio = board.pins.create_device_detect.gpio,
        estop_enabled = board.pins.estop.enabled,
        estop_pin = board.pins.estop.pin,
        estop_gpio = board.pins.estop.gpio,
    );

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::write(out_dir.join("body_config.rs"), generated).unwrap();
}
