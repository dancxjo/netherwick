const UNKNOWN: u8 = 0;
const OFF: u8 = 1;
const ON: u8 = 2;
const EVENT_LOG_CAPACITY: usize = 128;
// Keep transport responses bounded independently of the retained audit window.
// Consumers advance through the ring a page at a time using the returned
// `next_seq`, so a larger safety history does not overflow Pico UART buffers.
const EVENT_RESPONSE_CAPACITY: usize = 16;

static RUNTIME_STATE: AtomicU8 = AtomicU8::new(RuntimeState::Booting as u8);
static CREATE_POWER_STATE: AtomicU8 = AtomicU8::new(UNKNOWN);
static OI_MODE: AtomicU8 = AtomicU8::new(UNKNOWN);
static UART_RX_HEALTH: AtomicU8 = AtomicU8::new(UNKNOWN);
static CURRENT_COMMAND: AtomicU8 = AtomicU8::new(CommandCode::None as u8);
static LAST_ERROR: AtomicU8 = AtomicU8::new(ErrorCode::None as u8);
static LAST_ERROR_UART_READ_ERROR: AtomicU8 = AtomicU8::new(UartReadErrorCode::None as u8);
static BODY_STATE: AtomicU8 = AtomicU8::new(BodyState::NotStarted as u8);
static LAST_UART_PACKET_TIMESTAMP_MS: AtomicU32 = AtomicU32::new(0);
static LAST_UART_READ_ERROR: AtomicU8 = AtomicU8::new(UartReadErrorCode::None as u8);
static UART_RX_BYTES: AtomicU32 = AtomicU32::new(0);
static UART_RX_PACKETS: AtomicU32 = AtomicU32::new(0);
static LAST_UART_PACKET_LEN: AtomicU32 = AtomicU32::new(0);
static UART_TX_BYTES: AtomicU32 = AtomicU32::new(0);
static LAST_UART_RX_BYTE: AtomicU8 = AtomicU8::new(0);
static LAST_UART_TX_BYTE: AtomicU8 = AtomicU8::new(0);
static LAST_UART_RX_TIMESTAMP_MS: AtomicU32 = AtomicU32::new(0);
static LAST_UART_TX_TIMESTAMP_MS: AtomicU32 = AtomicU32::new(0);
static UART_RX_OVERRUNS: AtomicU32 = AtomicU32::new(0);
static UART_RX_BREAKS: AtomicU32 = AtomicU32::new(0);
static UART_RX_PARITY_ERRORS: AtomicU32 = AtomicU32::new(0);
static UART_RX_FRAMING_ERRORS: AtomicU32 = AtomicU32::new(0);
static UART_RX_OTHER_ERRORS: AtomicU32 = AtomicU32::new(0);
static WAKE_PROBE_RESPONSE_BYTES: AtomicU32 = AtomicU32::new(0);
static WAKE_PROBE_EXPECTED_BYTES: AtomicU32 = AtomicU32::new(0);
static CURRENT_RUNTIME_ACTION: AtomicU8 = AtomicU8::new(RuntimeActionCode::None as u8);
static LAST_ERROR_ACTION: AtomicU8 = AtomicU8::new(RuntimeActionCode::None as u8);
static WIFI_STATE: AtomicU8 = AtomicU8::new(WifiState::Off as u8);
static HTTPS_STATE: AtomicU8 = AtomicU8::new(HttpsState::Unavailable as u8);
static HTTP_REQUESTS: AtomicU32 = AtomicU32::new(0);
static DHCP_GRANTS: AtomicU32 = AtomicU32::new(0);
static ICMP_ECHO_REQUESTS: AtomicU32 = AtomicU32::new(0);
static ICMP_ECHO_REPLIES: AtomicU32 = AtomicU32::new(0);
static ICMP_DROPPED: AtomicU32 = AtomicU32::new(0);
static ICMP_RATE_LIMITED: AtomicU32 = AtomicU32::new(0);
static LAST_WEB_REQUEST_TIMESTAMP_MS: AtomicU32 = AtomicU32::new(0);
static PENDING_LED_BLINKS: AtomicU8 = AtomicU8::new(0);
static PENDING_COMMAND_KIND: AtomicU8 = AtomicU8::new(ControlCommandCode::None as u8);
static PENDING_COMMAND_ID: AtomicU32 = AtomicU32::new(0);
static PENDING_COMMAND_A: AtomicU32 = AtomicU32::new(0);
static PENDING_COMMAND_B: AtomicU32 = AtomicU32::new(0);
static PENDING_COMMAND_C: AtomicU32 = AtomicU32::new(0);
static PENDING_COMMAND_D: AtomicU32 = AtomicU32::new(0);
static PENDING_COMMAND_DURATION_MS: AtomicU32 = AtomicU32::new(0);
static PENDING_COMMAND_SEQ: AtomicU32 = AtomicU32::new(0);
static PENDING_COMMAND_SERVICE_SESSION_HASH: AtomicU32 = AtomicU32::new(0);
static PENDING_COMMAND_SERVICE_LEASE_HASH: AtomicU32 = AtomicU32::new(0);
static PENDING_VELOCITY_KIND: AtomicU8 = AtomicU8::new(ControlCommandCode::None as u8);
static PENDING_VELOCITY_ID: AtomicU32 = AtomicU32::new(0);
static PENDING_VELOCITY_A: AtomicU32 = AtomicU32::new(0);
static PENDING_VELOCITY_B: AtomicU32 = AtomicU32::new(0);
static PENDING_VELOCITY_TTL_MS: AtomicU32 = AtomicU32::new(0);
static PENDING_VELOCITY_SEQ: AtomicU32 = AtomicU32::new(0);
static PENDING_VELOCITY_IS_RENEWAL: AtomicU8 = AtomicU8::new(OFF);
static ACTIVE_VELOCITY_STREAM_ID: AtomicU32 = AtomicU32::new(0);
static ACTIVE_VELOCITY_STREAM_A: AtomicU32 = AtomicU32::new(0);
static ACTIVE_VELOCITY_STREAM_B: AtomicU32 = AtomicU32::new(0);
static ACTIVE_VELOCITY_STREAM: AtomicU8 = AtomicU8::new(OFF);
static LAST_ACCEPTED_COMMAND_ID: AtomicU32 = AtomicU32::new(0);
static LAST_REJECTED_COMMAND_ID: AtomicU32 = AtomicU32::new(0);
static LAST_STARTED_COMMAND_ID: AtomicU32 = AtomicU32::new(0);
static LAST_COMPLETED_COMMAND_ID: AtomicU32 = AtomicU32::new(0);
static LAST_INTERRUPTED_COMMAND_ID: AtomicU32 = AtomicU32::new(0);
static LAST_TIMED_OUT_COMMAND_ID: AtomicU32 = AtomicU32::new(0);
static LAST_DISPATCHED_COMMAND_ID: AtomicU32 = AtomicU32::new(0);
static LAST_DISPATCHED_SERVICE_SESSION_HASH: AtomicU32 = AtomicU32::new(0);
static LAST_DISPATCHED_SERVICE_LEASE_HASH: AtomicU32 = AtomicU32::new(0);
static FOREBRAIN_UART_RX_BYTES: AtomicU32 = AtomicU32::new(0);
static FOREBRAIN_UART_RX_LINES: AtomicU32 = AtomicU32::new(0);
static FOREBRAIN_UART_LAST_SEQ: AtomicU32 = AtomicU32::new(0);
static FOREBRAIN_UART_LAST_ERROR: AtomicU8 = AtomicU8::new(ForebrainUartErrorCode::None as u8);
static FOREBRAIN_UART_LAST_RX_MS: AtomicU32 = AtomicU32::new(0);
static FOREBRAIN_UART_LAST_COMMAND_MS: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_LAST_PACKET_ID: AtomicU8 = AtomicU8::new(0);
static CREATE_SENSOR_COMPLETE_PACKET_COUNT: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_LAST_COMPLETE_PACKET_TIMESTAMP_MS: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_FLAGS: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_DISTANCE_MM: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_ANGLE_MRAD: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_IR_BYTE: AtomicU8 = AtomicU8::new(0);
static CREATE_SENSOR_BUTTONS: AtomicU8 = AtomicU8::new(0);
static CREATE_SENSOR_CHARGING_STATE: AtomicU8 = AtomicU8::new(0);
static CREATE_SENSOR_CHARGING_SOURCES: AtomicU8 = AtomicU8::new(0);
static CREATE_CHARGING_INDICATOR_STATE: AtomicU8 = AtomicU8::new(UNKNOWN);
static CREATE_SENSOR_VOLTAGE_MV: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_CURRENT_MA: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_TEMPERATURE_C: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_CHARGE_MAH: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_CAPACITY_MAH: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_CLIFF_LEFT_SIGNAL: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_CLIFF_FRONT_LEFT_SIGNAL: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_CLIFF_FRONT_RIGHT_SIGNAL: AtomicU32 = AtomicU32::new(0);
static CREATE_SENSOR_CLIFF_RIGHT_SIGNAL: AtomicU32 = AtomicU32::new(0);
static BATTERY_LOW_LATCHED: AtomicU8 = AtomicU8::new(0);
static PENDING_SONG_TONES: [AtomicU32; MAX_SONG_TONES] =
    [const { AtomicU32::new(0) }; MAX_SONG_TONES];
static CREATE_SONG_LAST_DEFINED_ID: AtomicU8 = AtomicU8::new(0);
static CREATE_SONG_LAST_DEFINED_LEN: AtomicU8 = AtomicU8::new(0);
static CREATE_SONG_LAST_PLAYED_ID: AtomicU8 = AtomicU8::new(0);
static AUDIO_SILENT: AtomicU8 = AtomicU8::new(OFF);
static AUDIO_LAST_REQUESTED_CUE: AtomicU8 = AtomicU8::new(0);
static AUDIO_LAST_PLAYED_CUE: AtomicU8 = AtomicU8::new(0);
static AUDIO_LAST_PLAYBACK_TIMESTAMP_MS: AtomicU32 = AtomicU32::new(0);
static AUDIO_SUPPRESSED_BY_SILENT_COUNT: AtomicU32 = AtomicU32::new(0);
static AUDIO_DROPPED_OR_REPLACED_COUNT: AtomicU32 = AtomicU32::new(0);
static ODOMETRY_RESET_COUNT: AtomicU32 = AtomicU32::new(0);
static ODOMETRY_SEQUENCE: AtomicU32 = AtomicU32::new(0);
static ODOMETRY_DISTANCE_MM: AtomicU32 = AtomicU32::new(0);
static ODOMETRY_X_MM_Q10: AtomicU32 = AtomicU32::new(0);
static ODOMETRY_Y_MM_Q10: AtomicU32 = AtomicU32::new(0);
static ODOMETRY_HEADING_MRAD: AtomicU32 = AtomicU32::new(0);
static IMU_PRESENT: AtomicU8 = AtomicU8::new(UNKNOWN);
static IMU_HEALTH: AtomicU8 = AtomicU8::new(ImuHealthCode::Unknown as u8);
static IMU_LAST_SAMPLE_TIMESTAMP_MS: AtomicU32 = AtomicU32::new(0);
static IMU_SAMPLE_COUNT: AtomicU32 = AtomicU32::new(0);
static IMU_GYRO_X_MRAD_S: AtomicU32 = AtomicU32::new(0);
static IMU_GYRO_Y_MRAD_S: AtomicU32 = AtomicU32::new(0);
static IMU_GYRO_Z_MRAD_S: AtomicU32 = AtomicU32::new(0);
static IMU_ACCEL_X_MM_S2: AtomicU32 = AtomicU32::new(0);
static IMU_ACCEL_Y_MM_S2: AtomicU32 = AtomicU32::new(0);
static IMU_ACCEL_Z_MM_S2: AtomicU32 = AtomicU32::new(0);
static IMU_YAW_MRAD: AtomicU32 = AtomicU32::new(0);
static IMU_PITCH_MRAD: AtomicU32 = AtomicU32::new(0);
static IMU_ROLL_MRAD: AtomicU32 = AtomicU32::new(0);
static IMU_ACCEL_MAGNITUDE_MM_S2: AtomicU32 = AtomicU32::new(0);
static IMU_TILT_MAGNITUDE_MRAD: AtomicU32 = AtomicU32::new(0);
static IMU_ROUGHNESS_MM_S2: AtomicU32 = AtomicU32::new(0);
static IMU_IMPACT_SCORE_MM_S2: AtomicU32 = AtomicU32::new(0);
static IMU_MOTION_CONSISTENCY: AtomicU8 = AtomicU8::new(MotionConsistencyCode::Unknown as u8);
static IMU_CALIBRATION_STATE: AtomicU8 = AtomicU8::new(ImuCalibrationCode::Uncalibrated as u8);
static IMU_TILT_ACTIVE: AtomicU8 = AtomicU8::new(0);
static IMU_GRAVITY_CALIBRATED: AtomicU8 = AtomicU8::new(0);
static IMU_GRAVITY_REF_X_MM_S2: AtomicU32 = AtomicU32::new(0);
static IMU_GRAVITY_REF_Y_MM_S2: AtomicU32 = AtomicU32::new(0);
static IMU_GRAVITY_REF_Z_MM_S2: AtomicU32 = AtomicU32::new(0);
static IMU_GRAVITY_REF_MAGNITUDE_MM_S2: AtomicU32 = AtomicU32::new(0);
static IMU_GYRO_BIAS_SAMPLE_COUNT: AtomicU32 = AtomicU32::new(0);
static IMU_GYRO_BIAS_SUM_X: AtomicU32 = AtomicU32::new(0);
static IMU_GYRO_BIAS_SUM_Y: AtomicU32 = AtomicU32::new(0);
static IMU_GYRO_BIAS_SUM_Z: AtomicU32 = AtomicU32::new(0);
static IMU_GYRO_BIAS_X_MRAD_S: AtomicU32 = AtomicU32::new(0);
static IMU_GYRO_BIAS_Y_MRAD_S: AtomicU32 = AtomicU32::new(0);
static IMU_GYRO_BIAS_Z_MRAD_S: AtomicU32 = AtomicU32::new(0);
static EVENT_NEXT_SEQ: AtomicU32 = AtomicU32::new(1);
static EVENT_SEQ: [AtomicU32; EVENT_LOG_CAPACITY] =
    [const { AtomicU32::new(0) }; EVENT_LOG_CAPACITY];
static EVENT_KIND: [AtomicU8; EVENT_LOG_CAPACITY] =
    [const { AtomicU8::new(PublicEventKind::None as u8) }; EVENT_LOG_CAPACITY];
static EVENT_A: [AtomicU32; EVENT_LOG_CAPACITY] = [const { AtomicU32::new(0) }; EVENT_LOG_CAPACITY];
static EVENT_B: [AtomicU32; EVENT_LOG_CAPACITY] = [const { AtomicU32::new(0) }; EVENT_LOG_CAPACITY];
static EVENT_C: [AtomicU32; EVENT_LOG_CAPACITY] = [const { AtomicU32::new(0) }; EVENT_LOG_CAPACITY];
static SESSION_REPLACE_REQUEST: AtomicU32 = AtomicU32::new(0);
static SESSION_REPLACE_ACK: AtomicU32 = AtomicU32::new(0);
static PENDING_SESSION_HASH: AtomicU32 = AtomicU32::new(0);
static PENDING_PEER_DEVICE_HASH: AtomicU32 = AtomicU32::new(0);
static PENDING_PEER_BOOT_HASH: AtomicU32 = AtomicU32::new(0);
static ACTIVE_PEER_DEVICE_HASH: AtomicU32 = AtomicU32::new(0);
static ACTIVE_PEER_BOOT_HASH: AtomicU32 = AtomicU32::new(0);
const DIAGNOSTIC_SESSION_CAPACITY: usize = 4;
static DIAGNOSTIC_SESSION_HASH: [AtomicU32; DIAGNOSTIC_SESSION_CAPACITY] =
    [const { AtomicU32::new(0) }; DIAGNOSTIC_SESSION_CAPACITY];
static DIAGNOSTIC_PEER_HASH: [AtomicU32; DIAGNOSTIC_SESSION_CAPACITY] =
    [const { AtomicU32::new(0) }; DIAGNOSTIC_SESSION_CAPACITY];
static DIAGNOSTIC_PEER_BOOT_HASH: [AtomicU32; DIAGNOSTIC_SESSION_CAPACITY] =
    [const { AtomicU32::new(0) }; DIAGNOSTIC_SESSION_CAPACITY];
static DIAGNOSTIC_ROLE: [AtomicU8; DIAGNOSTIC_SESSION_CAPACITY] =
    [const { AtomicU8::new(0) }; DIAGNOSTIC_SESSION_CAPACITY];
static DIAGNOSTIC_PURPOSE: [AtomicU8; DIAGNOSTIC_SESSION_CAPACITY] =
    [const { AtomicU8::new(0) }; DIAGNOSTIC_SESSION_CAPACITY];
static DIAGNOSTIC_TRANSPORT: [AtomicU8; DIAGNOSTIC_SESSION_CAPACITY] =
    [const { AtomicU8::new(0) }; DIAGNOSTIC_SESSION_CAPACITY];
static ACTIVE_SESSION_HASH: AtomicU32 = AtomicU32::new(0);
static ACTIVE_SESSION_GENERATION: AtomicU32 = AtomicU32::new(0);
static SESSION_SAFETY_FLAGS: AtomicU32 = AtomicU32::new(0);
static SESSION_SAFETY_LATCH_KIND: AtomicU8 = AtomicU8::new(0);
static SAFETY_HAZARD_GENERATION: AtomicU32 = AtomicU32::new(0);
static CAREFUL_MODE_UNTIL_MS: AtomicU32 = AtomicU32::new(0);
static ACTIVE_TRANSPORT: AtomicU8 = AtomicU8::new(0);
static AUTHORITY_REQUEST: AtomicU32 = AtomicU32::new(0);
static AUTHORITY_ACK: AtomicU32 = AtomicU32::new(0);
static PENDING_LEASE_HASH: AtomicU32 = AtomicU32::new(0);
static PENDING_LEASE_SESSION_HASH: AtomicU32 = AtomicU32::new(0);
static ACTIVE_LEASE_HASH: AtomicU32 = AtomicU32::new(0);
static ACTIVE_LEASE_SESSION_HASH: AtomicU32 = AtomicU32::new(0);
static ACTIVE_LEASE_EXPIRES_MS: AtomicU32 = AtomicU32::new(0);
static ACTIVE_SERVICE_LEASE_HASH: AtomicU32 = AtomicU32::new(0);
static ACTIVE_SERVICE_SESSION_HASH: AtomicU32 = AtomicU32::new(0);
static ACTIVE_SERVICE_LEASE_EXPIRES_MS: AtomicU32 = AtomicU32::new(0);
static ACTIVE_SERVICE_SCOPE: AtomicU8 = AtomicU8::new(0);

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct BrainstemStatus {
    pub firmware_name: &'static str,
    pub firmware_version: &'static str,
    pub git_commit: &'static str,
    pub git_commit_short: &'static str,
    pub git_dirty: bool,
    pub build_timestamp: &'static str,
    pub build_profile: &'static str,
    pub build_target: &'static str,
    pub build_backend: &'static str,
    pub build_id: &'static str,
    pub body_name: &'static str,
    pub body_kind: &'static str,
    pub create_uart_baud: u32,
    pub create_sensor_probe_packet: u8,
    pub uptime_ms: u32,
    pub current_runtime_state: u8,
    pub create_power_state: u8,
    pub oi_mode: u8,
    pub uart_rx_health: u8,
    pub last_uart_packet_timestamp_ms: u32,
    pub last_uart_read_error: u8,
    pub uart_rx_bytes: u32,
    pub uart_rx_packets: u32,
    pub last_uart_packet_len: u32,
    pub uart_tx_bytes: u32,
    pub last_uart_rx_byte: u8,
    pub last_uart_tx_byte: u8,
    pub last_uart_rx_timestamp_ms: u32,
    pub last_uart_tx_timestamp_ms: u32,
    pub uart_rx_overruns: u32,
    pub uart_rx_breaks: u32,
    pub uart_rx_parity_errors: u32,
    pub uart_rx_framing_errors: u32,
    pub uart_rx_other_errors: u32,
    pub wake_probe_response_bytes: u32,
    pub wake_probe_expected_bytes: u32,
    pub current_command: u8,
    pub current_runtime_action: u8,
    pub last_error: u8,
    pub last_error_uart_read_error: u8,
    pub last_error_action: u8,
    pub body_state: u8,
    pub wifi_state: u8,
    pub https_state: u8,
    pub http_requests: u32,
    pub dhcp_grants: u32,
    pub icmp_echo_requests: u32,
    pub icmp_echo_replies: u32,
    pub icmp_dropped: u32,
    pub icmp_rate_limited: u32,
    pub last_web_request_timestamp_ms: u32,
    pub pending_command: u8,
    pub pending_command_id: u32,
    pub last_accepted_command_id: u32,
    pub last_rejected_command_id: u32,
    pub last_started_command_id: u32,
    pub last_completed_command_id: u32,
    pub last_interrupted_command_id: u32,
    pub last_timed_out_command_id: u32,
    pub forebrain_uart_rx_bytes: u32,
    pub forebrain_uart_rx_lines: u32,
    pub forebrain_uart_last_seq: u32,
    pub forebrain_uart_last_error: u8,
    pub forebrain_uart_link_alive_ms: u32,
    pub forebrain_uart_last_command_age_ms: u32,
    pub create_sensor_last_packet_id: u8,
    pub create_sensor_complete_packet_count: u32,
    pub create_sensor_last_complete_packet_timestamp_ms: u32,
    pub create_sensor_flags: u32,
    pub create_sensor_distance_mm: i16,
    pub create_sensor_angle_mrad: i16,
    pub create_sensor_ir_byte: u8,
    pub create_sensor_buttons: u8,
    pub create_sensor_charging_state: u8,
    pub create_sensor_charging_sources: u8,
    pub create_charging_indicator_state: u8,
    pub create_sensor_voltage_mv: u16,
    pub create_sensor_current_ma: i16,
    pub create_sensor_temperature_c: i8,
    pub create_sensor_charge_mah: u16,
    pub create_sensor_capacity_mah: u16,
    pub create_sensor_cliff_left_signal: u16,
    pub create_sensor_cliff_front_left_signal: u16,
    pub create_sensor_cliff_front_right_signal: u16,
    pub create_sensor_cliff_right_signal: u16,
    pub create_song_last_defined_id: u8,
    pub create_song_last_defined_len: u8,
    pub create_song_last_played_id: u8,
    pub audio_silent: bool,
    pub audio_last_requested_cue: u8,
    pub audio_last_played_cue: u8,
    pub audio_last_playback_timestamp_ms: u32,
    pub audio_suppressed_by_silent_count: u32,
    pub audio_dropped_or_replaced_count: u32,
    pub odometry_reset_count: u32,
    pub odometry_distance_mm: i32,
    pub odometry_x_mm: i32,
    pub odometry_y_mm: i32,
    pub odometry_heading_mrad: i32,
    pub imu_present: u8,
    pub imu_health: u8,
    pub imu_sample_count: u32,
    pub imu_last_sample_timestamp_ms: u32,
    pub imu_sample_age_ms: u32,
    pub imu_poll_period_ms: u32,
    pub imu_yaw_mrad: i32,
    pub imu_pitch_mrad: i16,
    pub imu_roll_mrad: i16,
    pub imu_yaw_rate_mrad_s: i16,
    pub imu_gyro_x_mrad_s: i16,
    pub imu_gyro_y_mrad_s: i16,
    pub imu_gyro_z_mrad_s: i16,
    pub imu_accel_x_mm_s2: i16,
    pub imu_accel_y_mm_s2: i16,
    pub imu_accel_z_mm_s2: i16,
    pub imu_accel_magnitude_mm_s2: u16,
    pub imu_tilt_magnitude_mrad: u16,
    pub imu_roughness_mm_s2: u16,
    pub imu_impact_score_mm_s2: u16,
    pub imu_motion_consistency: u8,
    pub imu_calibration_state: u8,
    pub imu_orientation_confidence_permille: u16,
    pub imu_gyro_bias_calibrated: bool,
    pub imu_mounting_calibrated: bool,
    pub imu_orientation_source: &'static str,
    pub safety_hazard_generation: u32,
    pub event_next_seq: u32,
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum RuntimeState {
    Booting = 1,
    Running = 2,
    Idle = 3,
    Error = 4,
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum BodyState {
    NotStarted = 1,
    WaitingForCreate = 2,
    OiStarted = 3,
    Moving = 4,
    PowerCycling = 5,
    Idle = 6,
    Error = 7,
}

#[derive(Clone, Copy)]
#[repr(u8)]
enum CommandCode {
    None = 0,
    WakeCreate = 1,
    SleepCreate = 2,
    StartOi = 4,
    SetOiPassive = 5,
    SetOiSafe = 6,
    SetOiFull = 7,
    Drive = 8,
    StopDrive = 9,
    Behavior = 10,
}

#[derive(Clone, Copy)]
#[repr(u8)]
enum ErrorCode {
    None = 0,
    CreateNoResponse = 1,
    UartFraming = 2,
    Timeout = 3,
    InvalidPacket = 4,
}

#[derive(Clone, Copy)]
#[repr(u8)]
enum UartReadErrorCode {
    None = 0,
    Overrun = 1,
    Break = 2,
    Parity = 3,
    Framing = 4,
    Other = 5,
}

#[derive(Clone, Copy)]
#[repr(u8)]
#[allow(dead_code)]
pub enum ForebrainUartErrorCode {
    None = 0,
    LineTooLong = 1,
    Utf8 = 2,
    Parse = 3,
    Busy = 4,
    Uart = 5,
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum RuntimeActionCode {
    None = 0,
    PowerPulse = 1,
    WakeSettle = 4,
    WaitForCreate = 5,
    Settle = 6,
    Driving = 7,
}

#[derive(Clone, Copy)]
#[repr(u8)]
#[allow(dead_code)]
enum WifiState {
    Off = 0,
    Starting = 1,
    ApStarted = 2,
    ServicesStarted = 3,
    Error = 4,
}

#[derive(Clone, Copy)]
#[repr(u8)]
enum HttpsState {
    Unavailable = 0,
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum PublicEventKind {
    None = 0,
    Boot = 1,
    CommandAccepted = 2,
    CommandRejected = 3,
    CommandStarted = 4,
    CommandCompleted = 5,
    CommandInterrupted = 6,
    CommandTimedOut = 7,
    BodyPowerRequested = 8,
    BodyPowerChanged = 9,
    BodyModeRequested = 10,
    BodyModeChanged = 11,
    TelemetryReceived = 12,
    SensorFrameDecoded = 13,
    MotionRequested = 14,
    MotionStopped = 15,
    SafetyTripped = 16,
    SafetyCleared = 17,
    BumpChanged = 18,
    CliffChanged = 19,
    WheelDropLatched = 20,
    WheelDropCleared = 21,
    WallChanged = 22,
    VirtualWallChanged = 23,
    BatteryLow = 24,
    ChargingStateChanged = 25,
    ButtonsChanged = 26,
    IrChanged = 27,
    HeartbeatExpired = 28,
    EStopLatched = 29,
    EStopCleared = 30,
    ImuFrameReceived = 31,
    ImuFault = 32,
    TiltChanged = 33,
    MotionInconsistencyDetected = 34,
    ImpactDetected = 35,
    ImuCalibrationChanged = 36,
    Error = 37,
    SessionOpened = 38,
    SessionReplaced = 39,
    SessionRejected = 40,
    TransportChanged = 41,
    PeerRebootDetected = 42,
    DhcpLeaseChanged = 43,
    DnsRegistrationChanged = 44,
    AuthorityChanged = 45,
    MotherbrainResetRequested = 46,
    MotherbrainResetAsserted = 47,
    MotherbrainResetCompleted = 48,
    MotherbrainResetRefused = 49,
    ContactWithdrawalStarted = 50,
    ContactWithdrawalCompleted = 51,
    CommandRenewed = 52,
    AudioStateChanged = 53,
}

#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(u8)]
pub enum ContactWithdrawalOutcome {
    Completed = 1,
    SafetyPreempted = 2,
    Failed = 3,
}

#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(u32)]
pub enum MotherbrainResetRefusal {
    HardwareDisabled = 1,
    UnsafeState = 2,
    Cooldown = 3,
    Duplicate = 4,
    InvalidServiceAuthority = 5,
    InvalidCommandId = 6,
}

#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(u8)]
pub enum SafetyEventKind {
    Bump = 1,
    Cliff = 2,
    WheelDrop = 3,
    EStop = 4,
    Heartbeat = 5,
    Tilt = 6,
    Impact = 7,
    Charging = 8,
}

fn safety_event_kind(code: u8) -> Option<SafetyEventKind> {
    match code {
        x if x == SafetyEventKind::Bump as u8 => Some(SafetyEventKind::Bump),
        x if x == SafetyEventKind::Cliff as u8 => Some(SafetyEventKind::Cliff),
        x if x == SafetyEventKind::WheelDrop as u8 => Some(SafetyEventKind::WheelDrop),
        x if x == SafetyEventKind::EStop as u8 => Some(SafetyEventKind::EStop),
        x if x == SafetyEventKind::Heartbeat as u8 => Some(SafetyEventKind::Heartbeat),
        x if x == SafetyEventKind::Tilt as u8 => Some(SafetyEventKind::Tilt),
        x if x == SafetyEventKind::Impact as u8 => Some(SafetyEventKind::Impact),
        x if x == SafetyEventKind::Charging as u8 => Some(SafetyEventKind::Charging),
        _ => None,
    }
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum ImuHealthCode {
    Unknown = 0,
    Ok = 1,
    Fault = 2,
    Absent = 3,
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum MotionConsistencyCode {
    Unknown = 0,
    Consistent = 1,
    Inconsistent = 2,
}

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum ImuCalibrationCode {
    Uncalibrated = 0,
    Calibrating = 1,
    Biased = 2,
    Ready = 3,
}

#[derive(Clone, Copy, Default)]
pub struct PublicEventRecord {
    pub seq: u32,
    pub kind: u8,
    pub a: u32,
    pub b: u32,
    pub c: u32,
}

#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(u8)]
enum ControlCommandCode {
    None = 0,
    Ping = 1,
    Arm = 2,
    Disarm = 3,
    EStop = 4,
    ClearEStop = 5,
    CmdVel = 6,
    Stop = 7,
    Status = 8,
    SongPlay = 9,
    Dock = 10,
    SetLights = 11,
    SetMode = 12,
    FaceBearing = 13,
    TrackBearing = 14,
    TurnBy = 15,
    DriveFor = 16,
    BumpEscape = 17,
    HeartbeatStop = 18,
    HoldHeading = 19,
    TurnToHeading = 20,
    ArcFor = 21,
    CreepUntil = 22,
    ScanArc = 23,
    DockAlign = 24,
    WallFollow = 25,
    WiggleAlign = 26,
    Unstick = 27,
    CliffGuard = 28,
    SongDefine = 29,
    DriveDirect = 30,
    DriveArc = 31,
    RequestSensors = 32,
    StreamSensors = 33,
    SetSafetyPolicy = 34,
    ClearMotionQueue = 35,
    DefineChirp = 36,
    PlayFeedback = 37,
    PowerState = 38,
    CalibrateTurn = 39,
    ResetOdometry = 40,
    GetCapabilities = 41,
    GetEvents = 42,
    ZeroImuOrientation = 43,
    ClearImuOrientation = 44,
    OrientationProbe = 45,
    RestartCreate = 46,
    ResetMotherbrain = 47,
    ClearSafetyLatch = 48,
    CarefulMode = 49,
    EscapeMotion = 50,
    SetAudioSilent = 51,
}
