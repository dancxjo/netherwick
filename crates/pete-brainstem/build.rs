use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

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

#[derive(Deserialize)]
struct VerbClassificationToml {
    verbs: Vec<VerbClassification>,
}

#[allow(dead_code)]
#[derive(Deserialize)]
struct VerbClassification {
    names: Vec<String>,
    category: String,
    authority: String,
    sensors: String,
    bounds: String,
    preempted_by: String,
    owner: String,
    lifecycle: String,
    exposed: bool,
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
    txs_oe: GpioPin,
    leds: LedPins,
    create_charging_indicator: OptionalInputPin,
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
struct OptionalInputPin {
    enabled: bool,
    pin: String,
    gpio: u8,
    physical_pin: Option<u8>,
    active: String,
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
    let build_identity = build_identity(&manifest_dir);
    let body_path = manifest_dir.join("body.toml");
    let board_path = manifest_dir.join("board.toml");
    let classification_path = manifest_dir.join("verb-classification.toml");
    println!("cargo:rerun-if-changed={}", body_path.display());
    println!("cargo:rerun-if-changed={}", board_path.display());
    println!("cargo:rerun-if-changed={}", classification_path.display());

    let mut body: BodyToml = toml::from_str(&fs::read_to_string(&body_path).unwrap()).unwrap();
    let board: BoardToml = toml::from_str(&fs::read_to_string(&board_path).unwrap()).unwrap();
    let classifications: VerbClassificationToml =
        toml::from_str(&fs::read_to_string(&classification_path).unwrap()).unwrap();
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
    if env::var_os("CARGO_FEATURE_SERVICE_MODE").is_some()
        && !body.capabilities.verbs.iter().any(|verb| verb == "bootsel")
    {
        body.capabilities.verbs.push("bootsel".into());
    }
    validate_verb_classifications(&body.capabilities.verbs, &classifications);

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
pub const CREATE_BRC_ENABLED: bool = false;

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

pub const BUILD_GIT_COMMIT: &str = {git_commit:?};
pub const BUILD_GIT_COMMIT_SHORT: &str = {git_commit_short:?};
pub const BUILD_GIT_DIRTY: bool = {git_dirty};
pub const BUILD_TIMESTAMP: &str = {build_timestamp:?};
pub const BUILD_PROFILE: &str = {build_profile:?};
pub const BUILD_TARGET: &str = {build_target:?};
pub const BUILD_BACKEND: &str = {build_backend:?};
pub const BUILD_ID: &str = {build_id:?};

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
pub const CREATE_POWER_TOGGLE_PHYSICAL_PIN: u8 = {create_power_toggle_physical_pin};
pub const TXS_OE_PIN: &str = {txs_oe_pin:?};
pub const TXS_OE_GPIO: u8 = {txs_oe_gpio};
pub const TXS_OE_PHYSICAL_PIN: u8 = {txs_oe_physical_pin};
pub const EXTERNAL_LED_GPIO: u8 = {external_led_gpio};
pub const EXTERNAL_LED_PIN: &str = {external_led_pin:?};
pub const CREATE_CHARGING_INDICATOR_ENABLED: bool = {create_charging_indicator_enabled};
pub const CREATE_CHARGING_INDICATOR_PIN: &str = {create_charging_indicator_pin:?};
pub const CREATE_CHARGING_INDICATOR_GPIO: u8 = {create_charging_indicator_gpio};
pub const CREATE_CHARGING_INDICATOR_PHYSICAL_PIN: u8 = {create_charging_indicator_physical_pin};
pub const CREATE_CHARGING_INDICATOR_ACTIVE_HIGH: bool = {create_charging_indicator_active_high};
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
        git_commit = build_identity.git_commit,
        git_commit_short = build_identity.git_commit_short,
        git_dirty = build_identity.git_dirty,
        build_timestamp = build_identity.build_timestamp,
        build_profile = build_identity.build_profile,
        build_target = build_identity.build_target,
        build_backend = build_identity.build_backend,
        build_id = build_identity.build_id,
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
        create_power_toggle_physical_pin = board.pins.power_toggle.physical_pin.unwrap_or(0),
        txs_oe_pin = board.pins.txs_oe.pin,
        txs_oe_gpio = board.pins.txs_oe.gpio,
        txs_oe_physical_pin = board.pins.txs_oe.physical_pin.unwrap_or(0),
        external_led_gpio = board.pins.leds.status_gpio,
        external_led_pin = board.pins.leds.status,
        create_charging_indicator_enabled = board.pins.create_charging_indicator.enabled,
        create_charging_indicator_pin = board.pins.create_charging_indicator.pin,
        create_charging_indicator_gpio = board.pins.create_charging_indicator.gpio,
        create_charging_indicator_physical_pin = board
            .pins
            .create_charging_indicator
            .physical_pin
            .unwrap_or(0),
        create_charging_indicator_active_high =
            match board.pins.create_charging_indicator.active.as_str() {
                "high" => true,
                "low" => false,
                other => panic!("unsupported create_charging_indicator.active: {other}"),
            },
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

fn validate_verb_classifications(
    advertised_verbs: &[String],
    classifications: &VerbClassificationToml,
) {
    let mut classified = BTreeMap::new();
    for group in &classifications.verbs {
        assert!(
            matches!(
                group.category.as_str(),
                "primitive"
                    | "body-native"
                    | "telemetry"
                    | "service"
                    | "reflex"
                    | "deprecated/moved skill"
            ),
            "invalid Brainstem verb category: {}",
            group.category
        );
        for (field, value) in [
            ("authority", group.authority.as_str()),
            ("sensors", group.sensors.as_str()),
            ("bounds", group.bounds.as_str()),
            ("preempted_by", group.preempted_by.as_str()),
            ("owner", group.owner.as_str()),
            ("lifecycle", group.lifecycle.as_str()),
        ] {
            assert!(
                !value.trim().is_empty(),
                "Brainstem verb classification has an empty {field} field"
            );
        }
        for name in &group.names {
            assert!(
                classified.insert(name.as_str(), group.exposed).is_none(),
                "duplicate Brainstem verb classification: {name}"
            );
        }
    }
    let advertised = advertised_verbs
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    for verb in &advertised {
        assert_eq!(
            classified.get(verb).copied(),
            Some(true),
            "advertised Brainstem verb lacks an exposed classification: {verb}"
        );
    }
    for (verb, exposed) in classified {
        if exposed && !matches!(verb, "bootsel" | "reset_motherbrain") {
            assert!(
                advertised.contains(verb),
                "verb is classified as exposed but not advertised: {verb}"
            );
        }
    }
}

struct BuildIdentity {
    git_commit: String,
    git_commit_short: String,
    git_dirty: bool,
    build_timestamp: String,
    build_profile: String,
    build_target: String,
    build_backend: String,
    build_id: String,
}

fn build_identity(manifest_dir: &Path) -> BuildIdentity {
    for variable in [
        "PETE_GIT_COMMIT",
        "PETE_GIT_DIRTY",
        "PETE_BUILD_TIMESTAMP",
        "SOURCE_DATE_EPOCH",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    let version = env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "unknown".into());
    let repo_root = manifest_dir
        .ancestors()
        .find(|path| path.join(".git").exists());
    if let Some(repo_root) = repo_root {
        rerun_for_git_state(repo_root);
        rerun_for_tracked_sources(repo_root);
    }
    let override_commit = env::var("PETE_GIT_COMMIT")
        .ok()
        .filter(|value| !value.is_empty());
    let git_commit = override_commit
        .or_else(|| repo_root.and_then(git_commit))
        .unwrap_or_else(|| "unknown".into());
    let git_commit_short = if git_commit == "unknown" {
        "unknown".into()
    } else {
        git_commit.chars().take(12).collect()
    };
    let git_dirty = env::var("PETE_GIT_DIRTY")
        .ok()
        .and_then(|value| parse_bool(&value))
        .or_else(|| repo_root.and_then(git_dirty))
        .unwrap_or(false);
    let build_timestamp = env::var("PETE_BUILD_TIMESTAMP")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| {
            env::var("SOURCE_DATE_EPOCH")
                .ok()
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| "unknown".into());
    let build_profile = env::var("PROFILE").unwrap_or_else(|_| "unknown".into());
    let build_target = env::var("TARGET").unwrap_or_else(|_| "unknown".into());
    let build_backend = if env::var_os("CARGO_FEATURE_PICO_W").is_some() {
        "pico-w"
    } else {
        "rp2040"
    }
    .into();
    let build_id = format!(
        "{version}+g{git_commit_short}{}",
        if git_dirty { ".dirty" } else { "" }
    );
    BuildIdentity {
        git_commit,
        git_commit_short,
        git_dirty,
        build_timestamp,
        build_profile,
        build_target,
        build_backend,
        build_id,
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value {
        "1" | "true" | "yes" => Some(true),
        "0" | "false" | "no" => Some(false),
        _ => None,
    }
}

fn git_commit(repo_root: &Path) -> Option<String> {
    git_output(repo_root, &["rev-parse", "HEAD"])
}

fn git_dirty(repo_root: &Path) -> Option<bool> {
    Command::new("git")
        .args(["diff-index", "--quiet", "HEAD", "--"])
        .current_dir(repo_root)
        .status()
        .ok()
        .map(|status| !status.success())
}

fn git_output(repo_root: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo_root)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn rerun_for_git_state(repo_root: &Path) {
    for state_path in ["HEAD", "index", "packed-refs"] {
        if let Some(path) = git_output(repo_root, &["rev-parse", "--git-path", state_path]) {
            print_rerun_path(repo_root, &path);
        }
    }
    if let Some(reference) = git_output(repo_root, &["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git_output(repo_root, &["rev-parse", "--git-path", &reference]) {
            print_rerun_path(repo_root, &path);
        }
    }
}

fn print_rerun_path(repo_root: &Path, path: &str) {
    let path = Path::new(path);
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root.join(path)
    };
    println!("cargo:rerun-if-changed={}", path.display());
}

fn rerun_for_tracked_sources(repo_root: &Path) {
    let Ok(output) = Command::new("git")
        .args(["ls-files", "-z"])
        .current_dir(repo_root)
        .output()
    else {
        return;
    };
    if !output.status.success() {
        return;
    }
    for path in output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
    {
        println!(
            "cargo:rerun-if-changed={}",
            repo_root
                .join(String::from_utf8_lossy(path).as_ref())
                .display()
        );
    }
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
    claim(board.pins.txs_oe.gpio, "txs_oe");
    claim(board.pins.leds.onboard_gpio, "leds.onboard");
    claim(board.pins.leds.status_gpio, "leds.status");
    if board.pins.create_charging_indicator.enabled {
        claim(
            board.pins.create_charging_indicator.gpio,
            "create_charging_indicator",
        );
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
