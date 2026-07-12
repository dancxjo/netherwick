use std::{env, fs, path::PathBuf};

use serde::Deserialize;

#[allow(dead_code)]
#[derive(Deserialize)]
struct BodyToml {
    body: Body,
    create_oi: CreateOi,
    timing: Timing,
    capabilities: Capabilities,
    limits: Limits,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct BoardToml {
    board: Board,
    rp2040: Rp2040,
    pins: Pins,
    i2c: I2c,
    imu: Imu,
    spi: Spi,
    pwm: Pwm,
    adc: Adc,
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
    supported_sensor_packets: String,
    supported_modes: Vec<String>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Capabilities {
    verbs: Vec<String>,
    sensors: Vec<String>,
    outputs: Vec<String>,
    safety: Vec<String>,
    feedback: Vec<String>,
    events: Vec<String>,
    song_slots: u8,
    max_song_tones: usize,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Limits {
    max_linear_mm_s: i16,
    max_angular_mrad_s: i16,
    min_ttl_ms: u32,
    max_ttl_ms: u32,
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
    motherbrain_reset: OptionalGpioPin,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct I2c {
    primary: I2cPins,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Spi {
    primary: SpiPins,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct SpiPins {
    mosi: String,
    mosi_gpio: u8,
    miso: String,
    miso_gpio: u8,
    sck: String,
    sck_gpio: u8,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Pwm {
    aux0: String,
    aux0_gpio: u8,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Adc {
    battery: String,
    battery_gpio: u8,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct I2cPins {
    sda: String,
    sda_gpio: u8,
    sda_physical_pin: u8,
    scl: String,
    scl_gpio: u8,
    scl_physical_pin: u8,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct Imu {
    enabled: bool,
    i2c_bus: String,
    poll_period_ms: u32,
    tilt_stop_mrad: i16,
    impact_stop_mm_s2: u16,
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

    let mut body: BodyToml = toml::from_str(&fs::read_to_string(&body_path).unwrap()).unwrap();
    let board: BoardToml = toml::from_str(&fs::read_to_string(&board_path).unwrap()).unwrap();
    assert_eq!(body.body.kind, "create_oi");
    assert_eq!(body.body.drive, "differential");
    assert_eq!(body.create_oi.data_bits, 8);
    assert_eq!(body.create_oi.stop_bits, 1);
    assert_eq!(board.board.arch, "rp2040");
    assert_eq!(board.imu.i2c_bus, "primary");
    validate_board_gpio_assignments(&board);
    println!("cargo:rustc-check-cfg=cfg(motherbrain_reset_hardware)");
    let motherbrain_reset_enabled = board.pins.motherbrain_reset.enabled
        && env::var_os("CARGO_FEATURE_MOTHERBRAIN_RESET").is_some();
    if motherbrain_reset_enabled {
        println!("cargo:rustc-cfg=motherbrain_reset_hardware");
        body.capabilities.verbs.push("reset_motherbrain".into());
        body.capabilities.outputs.push("motherbrain_reset".into());
        for event in [
            "motherbrain_reset_requested",
            "motherbrain_reset_asserted",
            "motherbrain_reset_completed",
            "motherbrain_reset_refused",
        ] {
            body.capabilities.events.push(event.into());
        }
    }

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
pub const CREATE_SUPPORTED_SENSOR_PACKETS: &str = {supported_sensor_packets:?};
pub const CREATE_SUPPORTED_MODES: &[&str] = {supported_modes};
pub const CREATE_BRC_ENABLED: bool = {create_brc_enabled};

pub const CAPABILITY_VERBS: &[&str] = {capability_verbs};
pub const CAPABILITY_SENSORS: &[&str] = {capability_sensors};
pub const CAPABILITY_OUTPUTS: &[&str] = {capability_outputs};
pub const CAPABILITY_SAFETY: &[&str] = {capability_safety};
pub const CAPABILITY_FEEDBACK: &[&str] = {capability_feedback};
pub const CAPABILITY_EVENTS: &[&str] = {capability_events};
pub const CAPABILITY_SONG_SLOTS: u8 = {song_slots};
pub const CAPABILITY_MAX_SONG_TONES: usize = {max_song_tones};
pub const CAPABILITY_MAX_LINEAR_MM_S: i16 = {max_linear_mm_s};
pub const CAPABILITY_MAX_ANGULAR_MRAD_S: i16 = {max_angular_mrad_s};
pub const CAPABILITY_MIN_TTL_MS: u32 = {min_ttl_ms};
pub const CAPABILITY_MAX_TTL_MS: u32 = {max_ttl_ms};

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
pub const MOTHERBRAIN_RESET_ENABLED: bool = {motherbrain_reset_enabled};

pub const I2C_PRIMARY_SDA_PIN: &str = {i2c_sda_pin:?};
pub const I2C_PRIMARY_SDA_GPIO: u8 = {i2c_sda_gpio};
pub const I2C_PRIMARY_SDA_PHYSICAL_PIN: u8 = {i2c_sda_physical_pin};
pub const I2C_PRIMARY_SCL_PIN: &str = {i2c_scl_pin:?};
pub const I2C_PRIMARY_SCL_GPIO: u8 = {i2c_scl_gpio};
pub const I2C_PRIMARY_SCL_PHYSICAL_PIN: u8 = {i2c_scl_physical_pin};
pub const IMU_ENABLED: bool = {imu_enabled};
pub const IMU_POLL_PERIOD_MS: u32 = {imu_poll_period_ms};
pub const IMU_TILT_STOP_MRAD: i16 = {imu_tilt_stop_mrad};
pub const IMU_IMPACT_STOP_MM_S2: u16 = {imu_impact_stop_mm_s2};
"#,
        body_name = body.body.name,
        board_name = board.board.name,
        baud = body.create_oi.baud,
        default_mode = default_mode,
        sensor_probe_packet = body.create_oi.sensor_probe_packet,
        supported_sensor_packets = body.create_oi.supported_sensor_packets,
        supported_modes = string_slice_literal(&body.create_oi.supported_modes),
        create_brc_enabled = board.pins.create_brc.enabled,
        capability_verbs = string_slice_literal(&body.capabilities.verbs),
        capability_sensors = string_slice_literal(&body.capabilities.sensors),
        capability_outputs = string_slice_literal(&body.capabilities.outputs),
        capability_safety = string_slice_literal(&body.capabilities.safety),
        capability_feedback = string_slice_literal(&body.capabilities.feedback),
        capability_events = string_slice_literal(&body.capabilities.events),
        song_slots = body.capabilities.song_slots,
        max_song_tones = body.capabilities.max_song_tones,
        max_linear_mm_s = body.limits.max_linear_mm_s,
        max_angular_mrad_s = body.limits.max_angular_mrad_s,
        min_ttl_ms = body.limits.min_ttl_ms,
        max_ttl_ms = body.limits.max_ttl_ms,
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
        motherbrain_reset_enabled = motherbrain_reset_enabled,
        i2c_sda_pin = board.i2c.primary.sda,
        i2c_sda_gpio = board.i2c.primary.sda_gpio,
        i2c_sda_physical_pin = board.i2c.primary.sda_physical_pin,
        i2c_scl_pin = board.i2c.primary.scl,
        i2c_scl_gpio = board.i2c.primary.scl_gpio,
        i2c_scl_physical_pin = board.i2c.primary.scl_physical_pin,
        imu_enabled = board.imu.enabled,
        imu_poll_period_ms = board.imu.poll_period_ms,
        imu_tilt_stop_mrad = board.imu.tilt_stop_mrad,
        imu_impact_stop_mm_s2 = board.imu.impact_stop_mm_s2,
    );

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let memory_layout = manifest_dir.join("memory.x");
    fs::copy(&memory_layout, out_dir.join("memory.x")).unwrap();
    println!("cargo:rerun-if-changed={}", memory_layout.display());
    println!("cargo:rustc-link-search={}", out_dir.display());
    if env::var("TARGET").as_deref() == Ok("thumbv6m-none-eabi") {
        println!("cargo:rustc-link-arg=-Tlink.x");
    }
    fs::write(out_dir.join("body_config.rs"), generated).unwrap();
}

fn string_slice_literal(values: &[String]) -> String {
    let mut literal = String::from("&[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            literal.push_str(", ");
        }
        literal.push_str(&format!("{value:?}"));
    }
    literal.push(']');
    literal
}

fn validate_board_gpio_assignments(board: &BoardToml) {
    let mut used = Vec::<(u8, &'static str)>::new();
    let mut claim = |gpio: u8, function: &'static str| {
        if let Some((_, previous)) = used.iter().find(|(used_gpio, _)| *used_gpio == gpio) {
            panic!(
                "board.toml GPIO collision: GP{gpio} is enabled for both {previous} and {function}"
            );
        }
        used.push((gpio, function));
    };

    claim(board.pins.create_uart.tx_gpio, "create_uart.tx");
    claim(board.pins.create_uart.rx_gpio, "create_uart.rx");
    claim(board.pins.power_toggle.gpio, "power_toggle");
    claim(board.pins.leds.onboard_gpio, "leds.onboard");
    claim(board.pins.leds.status_gpio, "leds.status");
    if board.pins.create_brc.enabled {
        claim(board.pins.create_brc.gpio, "create_brc");
    }
    if board.pins.create_device_detect.enabled {
        claim(board.pins.create_device_detect.gpio, "create_device_detect");
    }
    if board.pins.estop.enabled {
        claim(board.pins.estop.gpio, "estop");
    }
    if board.pins.motherbrain_reset.enabled {
        claim(board.pins.motherbrain_reset.gpio, "motherbrain_reset");
    }
    claim(board.i2c.primary.sda_gpio, "i2c.primary.sda");
    claim(board.i2c.primary.scl_gpio, "i2c.primary.scl");
    claim(board.spi.primary.mosi_gpio, "spi.primary.mosi");
    claim(board.spi.primary.miso_gpio, "spi.primary.miso");
    claim(board.spi.primary.sck_gpio, "spi.primary.sck");
    claim(board.pwm.aux0_gpio, "pwm.aux0");
    claim(board.adc.battery_gpio, "adc.battery");
}
