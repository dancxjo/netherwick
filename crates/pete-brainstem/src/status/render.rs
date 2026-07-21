#[cfg(feature = "pico-w")]
#[derive(serde::Serialize)]
struct StatusJson {
    firmware_name: &'static str,
    firmware_version: &'static str,
    git_commit: &'static str,
    git_commit_short: &'static str,
    git_dirty: bool,
    build_timestamp: &'static str,
    build_profile: &'static str,
    build_target: &'static str,
    build_backend: &'static str,
    build_id: &'static str,
    body_name: &'static str,
    body_kind: &'static str,
    create_uart_baud: u32,
    create_sensor_probe_packet: u8,
    uptime_ms: u32,
    current_runtime_state: &'static str,
    create_power_state: &'static str,
    oi_mode: &'static str,
    uart_rx_health: &'static str,
    last_uart_packet_timestamp_ms: u32,
    last_uart_read_error: &'static str,
    uart_rx_bytes: u32,
    uart_rx_packets: u32,
    last_uart_packet_len: u32,
    uart_tx_bytes: u32,
    last_uart_rx_byte: u8,
    last_uart_tx_byte: u8,
    last_uart_rx_timestamp_ms: u32,
    last_uart_tx_timestamp_ms: u32,
    uart_rx_overruns: u32,
    uart_rx_breaks: u32,
    uart_rx_parity_errors: u32,
    uart_rx_framing_errors: u32,
    uart_rx_other_errors: u32,
    wake_probe_response_bytes: u32,
    wake_probe_expected_bytes: u32,
    current_command: &'static str,
    current_runtime_action: &'static str,
    last_error: &'static str,
    last_error_uart_read_error: &'static str,
    last_error_action: &'static str,
    last_error_hint: &'static str,
    body_state: &'static str,
    estop_latched: bool,
    safety_tripped: bool,
    safety_latch_kind: &'static str,
    safety_hazard_generation: u32,
    motion_interlock_latched: bool,
    careful_mode_active: bool,
    careful_mode_remaining_ms: u32,
    wifi_state: &'static str,
    https_state: &'static str,
    http_requests: u32,
    dhcp_grants: u32,
    icmp_echo_requests: u32,
    icmp_echo_replies: u32,
    icmp_dropped: u32,
    icmp_rate_limited: u32,
    last_web_request_timestamp_ms: u32,
    pending_command: &'static str,
    pending_command_id: u32,
    last_accepted_command_id: u32,
    last_rejected_command_id: u32,
    last_started_command_id: u32,
    last_completed_command_id: u32,
    last_interrupted_command_id: u32,
    last_timed_out_command_id: u32,
    event_next_seq: u32,
    audio_silent: bool,
    audio: AudioStatusJson,
    create_songs: CreateSongStatusJson,
    odometry: OdometryStatusJson,
    imu: ImuStatusJson,
    create_sensors: CreateSensorStatusJson,
    forebrain_uart: ForebrainUartStatusJson,
}

#[cfg(feature = "pico-w")]
#[derive(serde::Serialize)]
struct CreateSongStatusJson {
    last_defined_id: u8,
    last_defined_len: u8,
    last_played_id: u8,
}

#[cfg(feature = "pico-w")]
#[derive(serde::Serialize)]
struct AudioStatusJson {
    silent: bool,
    last_requested_cue: &'static str,
    last_played_cue: &'static str,
    last_playback_timestamp_ms: u32,
    suppressed_by_silent_count: u32,
    dropped_or_replaced_count: u32,
}

#[cfg(feature = "pico-w")]
#[derive(serde::Serialize)]
struct OdometryStatusJson {
    reset_count: u32,
    distance_mm: i32,
    x_mm: i32,
    y_mm: i32,
    heading_mrad: i32,
}

#[cfg(feature = "pico-w")]
#[derive(serde::Serialize)]
struct ImuStatusJson {
    present: &'static str,
    health: &'static str,
    sample_count: u32,
    last_sample_timestamp_ms: u32,
    sample_age_ms: u32,
    poll_period_ms: u32,
    yaw_mrad: i32,
    pitch_mrad: i16,
    roll_mrad: i16,
    yaw_rate_mrad_s: i16,
    angular_velocity_mrad_s: Axis3I16Json,
    linear_acceleration_mm_s2: Axis3I16Json,
    accel_magnitude_mm_s2: u16,
    tilt_magnitude_mrad: u16,
    roughness_mm_s2: u16,
    impact_score_mm_s2: u16,
    motion_consistency: &'static str,
    calibration: &'static str,
}

#[cfg(feature = "pico-w")]
#[derive(serde::Serialize)]
struct Axis3I16Json {
    x: i16,
    y: i16,
    z: i16,
}

#[cfg(feature = "pico-w")]
#[derive(serde::Serialize)]
struct CreateSensorStatusJson {
    last_packet_id: u8,
    complete_packet_count: u32,
    last_complete_packet_timestamp_ms: u32,
    bump_left: bool,
    bump_right: bool,
    wheel_drop: bool,
    wall: bool,
    cliff_left: bool,
    cliff_front_left: bool,
    cliff_front_right: bool,
    cliff_right: bool,
    virtual_wall: bool,
    overcurrent: bool,
    distance_mm: i16,
    angle_mrad: i16,
    ir_byte: u8,
    buttons: u8,
    charging_state: u8,
    charging_sources: u8,
    charging_indicator: &'static str,
    charging_indicator_level: &'static str,
    charging_indicator_pin: &'static str,
    charging_indicator_gpio: u8,
    charging_indicator_physical_pin: u8,
    charging_indicator_active_high: bool,
    voltage_mv: u16,
    current_ma: i16,
    temperature_c: i8,
    charge_mah: u16,
    capacity_mah: u16,
    cliff_left_signal: u16,
    cliff_front_left_signal: u16,
    cliff_front_right_signal: u16,
    cliff_right_signal: u16,
}

#[cfg(feature = "pico-w")]
#[derive(serde::Serialize)]
struct ForebrainUartStatusJson {
    rx_bytes: u32,
    rx_lines: u32,
    last_seq: u32,
    last_error: &'static str,
    link_alive_ms: u32,
    last_command_age_ms: u32,
}

#[cfg(feature = "pico-w")]
pub fn render_json<'a>(snapshot: BrainstemStatus, buffer: &'a mut [u8]) -> Result<&'a str, ()> {
    let (estop_latched, safety_tripped, motion_interlock_latched, safety_latch_kind) =
        session_safety_snapshot();
    let status = StatusJson {
        firmware_name: snapshot.firmware_name,
        firmware_version: snapshot.firmware_version,
        git_commit: snapshot.git_commit,
        git_commit_short: snapshot.git_commit_short,
        git_dirty: snapshot.git_dirty,
        build_timestamp: snapshot.build_timestamp,
        build_profile: snapshot.build_profile,
        build_target: snapshot.build_target,
        build_backend: snapshot.build_backend,
        build_id: snapshot.build_id,
        body_name: snapshot.body_name,
        body_kind: snapshot.body_kind,
        create_uart_baud: snapshot.create_uart_baud,
        create_sensor_probe_packet: snapshot.create_sensor_probe_packet,
        uptime_ms: snapshot.uptime_ms,
        current_runtime_state: runtime_state_text(snapshot.current_runtime_state),
        create_power_state: tri_state_text(snapshot.create_power_state),
        oi_mode: oi_mode_text(snapshot.oi_mode),
        uart_rx_health: uart_health_text(snapshot.uart_rx_health),
        last_uart_packet_timestamp_ms: snapshot.last_uart_packet_timestamp_ms,
        last_uart_read_error: uart_read_error_text(snapshot.last_uart_read_error),
        uart_rx_bytes: snapshot.uart_rx_bytes,
        uart_rx_packets: snapshot.uart_rx_packets,
        last_uart_packet_len: snapshot.last_uart_packet_len,
        uart_tx_bytes: snapshot.uart_tx_bytes,
        last_uart_rx_byte: snapshot.last_uart_rx_byte,
        last_uart_tx_byte: snapshot.last_uart_tx_byte,
        last_uart_rx_timestamp_ms: snapshot.last_uart_rx_timestamp_ms,
        last_uart_tx_timestamp_ms: snapshot.last_uart_tx_timestamp_ms,
        uart_rx_overruns: snapshot.uart_rx_overruns,
        uart_rx_breaks: snapshot.uart_rx_breaks,
        uart_rx_parity_errors: snapshot.uart_rx_parity_errors,
        uart_rx_framing_errors: snapshot.uart_rx_framing_errors,
        uart_rx_other_errors: snapshot.uart_rx_other_errors,
        wake_probe_response_bytes: snapshot.wake_probe_response_bytes,
        wake_probe_expected_bytes: snapshot.wake_probe_expected_bytes,
        current_command: command_text(snapshot.current_command),
        current_runtime_action: runtime_action_text(snapshot.current_runtime_action),
        last_error: error_text(snapshot.last_error),
        last_error_uart_read_error: uart_read_error_text(snapshot.last_error_uart_read_error),
        last_error_action: runtime_action_text(snapshot.last_error_action),
        last_error_hint: error_hint_text(snapshot),
        body_state: body_state_text(snapshot.body_state),
        estop_latched,
        safety_tripped,
        safety_latch_kind: safety_event_kind_text(safety_latch_kind),
        safety_hazard_generation: snapshot.safety_hazard_generation,
        motion_interlock_latched,
        careful_mode_active: careful_mode_remaining_ms(snapshot.uptime_ms) > 0,
        careful_mode_remaining_ms: careful_mode_remaining_ms(snapshot.uptime_ms),
        wifi_state: wifi_state_text(snapshot.wifi_state),
        https_state: https_state_text(snapshot.https_state),
        http_requests: snapshot.http_requests,
        dhcp_grants: snapshot.dhcp_grants,
        icmp_echo_requests: snapshot.icmp_echo_requests,
        icmp_echo_replies: snapshot.icmp_echo_replies,
        icmp_dropped: snapshot.icmp_dropped,
        icmp_rate_limited: snapshot.icmp_rate_limited,
        last_web_request_timestamp_ms: snapshot.last_web_request_timestamp_ms,
        pending_command: control_command_text(snapshot.pending_command),
        pending_command_id: snapshot.pending_command_id,
        last_accepted_command_id: snapshot.last_accepted_command_id,
        last_rejected_command_id: snapshot.last_rejected_command_id,
        last_started_command_id: snapshot.last_started_command_id,
        last_completed_command_id: snapshot.last_completed_command_id,
        last_interrupted_command_id: snapshot.last_interrupted_command_id,
        last_timed_out_command_id: snapshot.last_timed_out_command_id,
        event_next_seq: snapshot.event_next_seq,
        audio_silent: snapshot.audio_silent,
        audio: AudioStatusJson {
            silent: snapshot.audio_silent,
            last_requested_cue: cue_name(snapshot.audio_last_requested_cue),
            last_played_cue: cue_name(snapshot.audio_last_played_cue),
            last_playback_timestamp_ms: snapshot.audio_last_playback_timestamp_ms,
            suppressed_by_silent_count: snapshot.audio_suppressed_by_silent_count,
            dropped_or_replaced_count: snapshot.audio_dropped_or_replaced_count,
        },
        create_songs: CreateSongStatusJson {
            last_defined_id: snapshot.create_song_last_defined_id,
            last_defined_len: snapshot.create_song_last_defined_len,
            last_played_id: snapshot.create_song_last_played_id,
        },
        odometry: OdometryStatusJson {
            reset_count: snapshot.odometry_reset_count,
            distance_mm: snapshot.odometry_distance_mm,
            x_mm: snapshot.odometry_x_mm,
            y_mm: snapshot.odometry_y_mm,
            heading_mrad: snapshot.odometry_heading_mrad,
        },
        imu: ImuStatusJson {
            present: tri_state_text(snapshot.imu_present),
            health: imu_health_text(snapshot.imu_health),
            sample_count: snapshot.imu_sample_count,
            last_sample_timestamp_ms: snapshot.imu_last_sample_timestamp_ms,
            sample_age_ms: snapshot.imu_sample_age_ms,
            poll_period_ms: snapshot.imu_poll_period_ms,
            yaw_mrad: snapshot.imu_yaw_mrad,
            pitch_mrad: snapshot.imu_pitch_mrad,
            roll_mrad: snapshot.imu_roll_mrad,
            yaw_rate_mrad_s: snapshot.imu_yaw_rate_mrad_s,
            angular_velocity_mrad_s: Axis3I16Json {
                x: snapshot.imu_gyro_x_mrad_s,
                y: snapshot.imu_gyro_y_mrad_s,
                z: snapshot.imu_gyro_z_mrad_s,
            },
            linear_acceleration_mm_s2: Axis3I16Json {
                x: snapshot.imu_accel_x_mm_s2,
                y: snapshot.imu_accel_y_mm_s2,
                z: snapshot.imu_accel_z_mm_s2,
            },
            accel_magnitude_mm_s2: snapshot.imu_accel_magnitude_mm_s2,
            tilt_magnitude_mrad: snapshot.imu_tilt_magnitude_mrad,
            roughness_mm_s2: snapshot.imu_roughness_mm_s2,
            impact_score_mm_s2: snapshot.imu_impact_score_mm_s2,
            motion_consistency: motion_consistency_text(snapshot.imu_motion_consistency),
            calibration: imu_calibration_text(snapshot.imu_calibration_state),
        },
        create_sensors: create_sensor_status_json(snapshot),
        forebrain_uart: ForebrainUartStatusJson {
            rx_bytes: snapshot.forebrain_uart_rx_bytes,
            rx_lines: snapshot.forebrain_uart_rx_lines,
            last_seq: snapshot.forebrain_uart_last_seq,
            last_error: forebrain_uart_error_text(snapshot.forebrain_uart_last_error),
            link_alive_ms: snapshot.forebrain_uart_link_alive_ms,
            last_command_age_ms: snapshot.forebrain_uart_last_command_age_ms,
        },
    };
    let len = serde_json_core::to_slice(&status, buffer).map_err(|_| ())?;
    core::str::from_utf8(&buffer[..len]).map_err(|_| ())
}

#[cfg(feature = "pico-w")]
fn create_sensor_status_json(snapshot: BrainstemStatus) -> CreateSensorStatusJson {
    let flags = snapshot.create_sensor_flags;
    CreateSensorStatusJson {
        last_packet_id: snapshot.create_sensor_last_packet_id,
        complete_packet_count: snapshot.create_sensor_complete_packet_count,
        last_complete_packet_timestamp_ms: snapshot.create_sensor_last_complete_packet_timestamp_ms,
        bump_left: flags & (1 << 0) != 0,
        bump_right: flags & (1 << 1) != 0,
        wheel_drop: flags & (1 << 2) != 0,
        wall: flags & (1 << 3) != 0,
        cliff_left: flags & (1 << 4) != 0,
        cliff_front_left: flags & (1 << 5) != 0,
        cliff_front_right: flags & (1 << 6) != 0,
        cliff_right: flags & (1 << 7) != 0,
        virtual_wall: flags & (1 << 8) != 0,
        overcurrent: flags & (1 << 9) != 0,
        distance_mm: snapshot.create_sensor_distance_mm,
        angle_mrad: snapshot.create_sensor_angle_mrad,
        ir_byte: snapshot.create_sensor_ir_byte,
        buttons: snapshot.create_sensor_buttons,
        charging_state: snapshot.create_sensor_charging_state,
        charging_sources: snapshot.create_sensor_charging_sources,
        charging_indicator: tri_state_text(snapshot.create_charging_indicator_state),
        charging_indicator_level: charging_indicator_level_text(
            snapshot.create_charging_indicator_state,
        ),
        charging_indicator_pin: body::CREATE_CHARGING_INDICATOR_PIN,
        charging_indicator_gpio: body::CREATE_CHARGING_INDICATOR_GPIO,
        charging_indicator_physical_pin: body::CREATE_CHARGING_INDICATOR_PHYSICAL_PIN,
        charging_indicator_active_high: body::CREATE_CHARGING_INDICATOR_ACTIVE_HIGH,
        voltage_mv: snapshot.create_sensor_voltage_mv,
        current_ma: snapshot.create_sensor_current_ma,
        temperature_c: snapshot.create_sensor_temperature_c,
        charge_mah: snapshot.create_sensor_charge_mah,
        capacity_mah: snapshot.create_sensor_capacity_mah,
        cliff_left_signal: snapshot.create_sensor_cliff_left_signal,
        cliff_front_left_signal: snapshot.create_sensor_cliff_front_left_signal,
        cliff_front_right_signal: snapshot.create_sensor_cliff_front_right_signal,
        cliff_right_signal: snapshot.create_sensor_cliff_right_signal,
    }
}

fn charging_indicator_level_text(state: u8) -> &'static str {
    match state {
        ON if body::CREATE_CHARGING_INDICATOR_ACTIVE_HIGH => "high",
        ON => "low",
        OFF if body::CREATE_CHARGING_INDICATOR_ACTIVE_HIGH => "low",
        OFF => "high",
        _ => "unknown",
    }
}

fn safety_event_kind_text(kind: Option<SafetyEventKind>) -> &'static str {
    match kind {
        Some(SafetyEventKind::Bump) => "bump",
        Some(SafetyEventKind::Cliff) => "cliff",
        Some(SafetyEventKind::WheelDrop) => "wheel_drop",
        Some(SafetyEventKind::EStop) => "estop",
        Some(SafetyEventKind::Heartbeat) => "heartbeat",
        Some(SafetyEventKind::Tilt) => "tilt",
        Some(SafetyEventKind::Impact) => "impact",
        Some(SafetyEventKind::Charging) => "charging",
        None => "none",
    }
}

#[allow(dead_code)]
fn body_kind() -> &'static str {
    match body::BODY_KIND {
        body::BodyKind::CreateOpenInterface => "create_oi",
    }
}

pub fn public_event_kind_text(code: u8) -> &'static str {
    match code {
        x if x == PublicEventKind::Boot as u8 => "boot",
        x if x == PublicEventKind::CommandAccepted as u8 => "command_accepted",
        x if x == PublicEventKind::CommandRejected as u8 => "command_rejected",
        x if x == PublicEventKind::CommandStarted as u8 => "command_started",
        x if x == PublicEventKind::CommandCompleted as u8 => "command_completed",
        x if x == PublicEventKind::CommandInterrupted as u8 => "command_interrupted",
        x if x == PublicEventKind::CommandTimedOut as u8 => "command_timed_out",
        x if x == PublicEventKind::CommandRenewed as u8 => "command_renewed",
        x if x == PublicEventKind::BodyPowerRequested as u8 => "body_power_requested",
        x if x == PublicEventKind::BodyPowerChanged as u8 => "body_power_changed",
        x if x == PublicEventKind::BodyModeRequested as u8 => "body_mode_requested",
        x if x == PublicEventKind::BodyModeChanged as u8 => "body_mode_changed",
        x if x == PublicEventKind::TelemetryReceived as u8 => "telemetry_received",
        x if x == PublicEventKind::SensorFrameDecoded as u8 => "sensor_frame_decoded",
        x if x == PublicEventKind::MotionRequested as u8 => "motion_requested",
        x if x == PublicEventKind::MotionStopped as u8 => "motion_stopped",
        x if x == PublicEventKind::SafetyTripped as u8 => "safety_tripped",
        x if x == PublicEventKind::SafetyCleared as u8 => "safety_cleared",
        x if x == PublicEventKind::BumpChanged as u8 => "bump_changed",
        x if x == PublicEventKind::CliffChanged as u8 => "cliff_changed",
        x if x == PublicEventKind::WheelDropLatched as u8 => "wheel_drop_latched",
        x if x == PublicEventKind::WheelDropCleared as u8 => "wheel_drop_cleared",
        x if x == PublicEventKind::WallChanged as u8 => "wall_changed",
        x if x == PublicEventKind::VirtualWallChanged as u8 => "virtual_wall_changed",
        x if x == PublicEventKind::BatteryLow as u8 => "battery_low",
        x if x == PublicEventKind::ChargingStateChanged as u8 => "charging_state_changed",
        x if x == PublicEventKind::ButtonsChanged as u8 => "buttons_changed",
        x if x == PublicEventKind::IrChanged as u8 => "ir_changed",
        x if x == PublicEventKind::HeartbeatExpired as u8 => "heartbeat_expired",
        x if x == PublicEventKind::EStopLatched as u8 => "estop_latched",
        x if x == PublicEventKind::EStopCleared as u8 => "estop_cleared",
        x if x == PublicEventKind::ImuFrameReceived as u8 => "imu_frame_received",
        x if x == PublicEventKind::ImuFault as u8 => "imu_fault",
        x if x == PublicEventKind::TiltChanged as u8 => "tilt_changed",
        x if x == PublicEventKind::ImuCalibrationChanged as u8 => "imu_calibration_changed",
        x if x == PublicEventKind::MotionInconsistencyDetected as u8 => {
            "motion_inconsistency_detected"
        }
        x if x == PublicEventKind::ImpactDetected as u8 => "impact_detected",
        x if x == PublicEventKind::SessionOpened as u8 => "session_opened",
        x if x == PublicEventKind::SessionReplaced as u8 => "session_replaced",
        x if x == PublicEventKind::SessionRejected as u8 => "session_rejected",
        x if x == PublicEventKind::TransportChanged as u8 => "transport_changed",
        x if x == PublicEventKind::PeerRebootDetected as u8 => "peer_reboot_detected",
        x if x == PublicEventKind::DhcpLeaseChanged as u8 => "dhcp_lease_changed",
        x if x == PublicEventKind::DnsRegistrationChanged as u8 => "dns_registration_changed",
        x if x == PublicEventKind::AuthorityChanged as u8 => "authority_changed",
        x if x == PublicEventKind::MotherbrainResetRequested as u8 => "motherbrain_reset_requested",
        x if x == PublicEventKind::MotherbrainResetAsserted as u8 => "motherbrain_reset_asserted",
        x if x == PublicEventKind::MotherbrainResetCompleted as u8 => "motherbrain_reset_completed",
        x if x == PublicEventKind::MotherbrainResetRefused as u8 => "motherbrain_reset_refused",
        x if x == PublicEventKind::ContactWithdrawalStarted as u8 => "contact_withdrawal_started",
        x if x == PublicEventKind::ContactWithdrawalCompleted as u8 => {
            "contact_withdrawal_completed"
        }
        x if x == PublicEventKind::AudioStateChanged as u8 => "audio_state_changed",
        x if x == PublicEventKind::Error as u8 => "error",
        _ => "none",
    }
}

#[cfg(feature = "pico-w")]
fn runtime_state_text(code: u8) -> &'static str {
    match code {
        x if x == RuntimeState::Booting as u8 => "booting",
        x if x == RuntimeState::Running as u8 => "running",
        x if x == RuntimeState::Idle as u8 => "idle",
        x if x == RuntimeState::Error as u8 => "error",
        _ => "unknown",
    }
}

#[cfg(feature = "pico-w")]
fn body_state_text(code: u8) -> &'static str {
    match code {
        x if x == BodyState::NotStarted as u8 => "not_started",
        x if x == BodyState::WaitingForCreate as u8 => "waiting_for_create",
        x if x == BodyState::OiStarted as u8 => "oi_started",
        x if x == BodyState::Moving as u8 => "moving",
        x if x == BodyState::PowerCycling as u8 => "power_cycling",
        x if x == BodyState::Idle as u8 => "idle",
        x if x == BodyState::Error as u8 => "error",
        _ => "unknown",
    }
}

#[cfg(feature = "pico-w")]
fn tri_state_text(code: u8) -> &'static str {
    match code {
        OFF => "off",
        ON => "on",
        _ => "unknown",
    }
}

#[cfg(feature = "pico-w")]
fn oi_mode_text(code: u8) -> &'static str {
    match code {
        1 => "passive",
        2 => "safe",
        3 => "full",
        _ => "unknown",
    }
}

#[cfg(feature = "pico-w")]
fn uart_health_text(code: u8) -> &'static str {
    match code {
        OFF => "error",
        ON => "ok",
        _ => "unknown",
    }
}

#[cfg(feature = "pico-w")]
fn command_text(code: u8) -> &'static str {
    match code {
        x if x == CommandCode::WakeCreate as u8 => "wake_create",
        x if x == CommandCode::SleepCreate as u8 => "sleep_create",
        x if x == CommandCode::StartOi as u8 => "start_oi",
        x if x == CommandCode::SetOiPassive as u8 => "set_oi_passive",
        x if x == CommandCode::SetOiSafe as u8 => "set_oi_safe",
        x if x == CommandCode::SetOiFull as u8 => "set_oi_full",
        x if x == CommandCode::Drive as u8 => "drive",
        x if x == CommandCode::StopDrive as u8 => "stop_drive",
        _ => "none",
    }
}

#[cfg(feature = "pico-w")]
fn error_text(code: u8) -> &'static str {
    match code {
        x if x == ErrorCode::CreateNoResponse as u8 => "create_no_response",
        x if x == ErrorCode::UartFraming as u8 => "uart_framing",
        x if x == ErrorCode::Timeout as u8 => "timeout",
        x if x == ErrorCode::InvalidPacket as u8 => "invalid_packet",
        _ => "none",
    }
}

#[cfg(feature = "pico-w")]
fn uart_read_error_text(code: u8) -> &'static str {
    match code {
        x if x == UartReadErrorCode::Overrun as u8 => "overrun",
        x if x == UartReadErrorCode::Break as u8 => "break",
        x if x == UartReadErrorCode::Parity as u8 => "parity",
        x if x == UartReadErrorCode::Framing as u8 => "framing",
        x if x == UartReadErrorCode::Other as u8 => "other",
        _ => "none",
    }
}

#[cfg(feature = "pico-w")]
fn forebrain_uart_error_text(code: u8) -> &'static str {
    match code {
        x if x == ForebrainUartErrorCode::LineTooLong as u8 => "line_too_long",
        x if x == ForebrainUartErrorCode::Utf8 as u8 => "utf8",
        x if x == ForebrainUartErrorCode::Parse as u8 => "parse",
        x if x == ForebrainUartErrorCode::Busy as u8 => "busy",
        x if x == ForebrainUartErrorCode::Uart as u8 => "uart",
        _ => "none",
    }
}

#[cfg(feature = "pico-w")]
fn imu_health_text(code: u8) -> &'static str {
    match code {
        x if x == ImuHealthCode::Ok as u8 => "ok",
        x if x == ImuHealthCode::Fault as u8 => "fault",
        x if x == ImuHealthCode::Absent as u8 => "absent",
        _ => "unknown",
    }
}

#[cfg(feature = "pico-w")]
fn motion_consistency_text(code: u8) -> &'static str {
    match code {
        x if x == MotionConsistencyCode::Consistent as u8 => "consistent",
        x if x == MotionConsistencyCode::Inconsistent as u8 => "inconsistent",
        _ => "unknown",
    }
}

#[cfg(feature = "pico-w")]
fn imu_calibration_text(code: u8) -> &'static str {
    match code {
        x if x == ImuCalibrationCode::Calibrating as u8 => "calibrating",
        x if x == ImuCalibrationCode::Biased as u8 => "biased",
        x if x == ImuCalibrationCode::Ready as u8 => "ready",
        _ => "uncalibrated",
    }
}

#[cfg(feature = "pico-w")]
fn runtime_action_text(code: u8) -> &'static str {
    match code {
        x if x == RuntimeActionCode::PowerPulse as u8 => "power_pulse",
        x if x == RuntimeActionCode::WakeSettle as u8 => "wake_settle",
        x if x == RuntimeActionCode::WaitForCreate as u8 => "wait_for_create",
        x if x == RuntimeActionCode::Settle as u8 => "settle",
        x if x == RuntimeActionCode::Driving as u8 => "driving",
        _ => "none",
    }
}

#[cfg(feature = "pico-w")]
fn control_command_text(code: u8) -> &'static str {
    match code {
        x if x == ControlCommandCode::Ping as u8 => "ping",
        x if x == ControlCommandCode::Arm as u8 => "arm",
        x if x == ControlCommandCode::Disarm as u8 => "disarm",
        x if x == ControlCommandCode::EStop as u8 => "estop",
        x if x == ControlCommandCode::ClearEStop as u8 => "clear_estop",
        x if x == ControlCommandCode::CmdVel as u8 => "cmd_vel",
        x if x == ControlCommandCode::Stop as u8 => "stop",
        x if x == ControlCommandCode::Status as u8 => "status",
        x if x == ControlCommandCode::SongPlay as u8 => "song_play",
        x if x == ControlCommandCode::Dock as u8 => "dock",
        x if x == ControlCommandCode::SetLights as u8 => "set_lights",
        x if x == ControlCommandCode::SetMode as u8 => "set_mode",
        x if x == ControlCommandCode::FaceBearing as u8 => "face_bearing",
        x if x == ControlCommandCode::TrackBearing as u8 => "track_bearing",
        x if x == ControlCommandCode::TurnBy as u8 => "turn_by",
        x if x == ControlCommandCode::DriveFor as u8 => "drive_for",
        x if x == ControlCommandCode::BumpEscape as u8 => "bump_escape",
        x if x == ControlCommandCode::HeartbeatStop as u8 => "heartbeat_stop",
        x if x == ControlCommandCode::HoldHeading as u8 => "hold_heading",
        x if x == ControlCommandCode::TurnToHeading as u8 => "turn_to_heading",
        x if x == ControlCommandCode::ArcFor as u8 => "arc_for",
        x if x == ControlCommandCode::CreepUntil as u8 => "creep_until",
        x if x == ControlCommandCode::ScanArc as u8 => "scan_arc",
        x if x == ControlCommandCode::DockAlign as u8 => "dock_align",
        x if x == ControlCommandCode::WallFollow as u8 => "wall_follow",
        x if x == ControlCommandCode::WiggleAlign as u8 => "wiggle_align",
        x if x == ControlCommandCode::Unstick as u8 => "unstick",
        x if x == ControlCommandCode::CliffGuard as u8 => "cliff_guard",
        x if x == ControlCommandCode::ClearSafetyLatch as u8 => "clear_safety_latch",
        x if x == ControlCommandCode::CarefulMode as u8 => "careful_mode",
        x if x == ControlCommandCode::EscapeMotion as u8 => "escape_motion",
        x if x == ControlCommandCode::SongDefine as u8 => "song_define",
        x if x == ControlCommandCode::DriveDirect as u8 => "drive_direct",
        x if x == ControlCommandCode::DriveArc as u8 => "drive_arc",
        x if x == ControlCommandCode::RequestSensors as u8 => "request_sensors",
        x if x == ControlCommandCode::StreamSensors as u8 => "stream_sensors",
        x if x == ControlCommandCode::SetSafetyPolicy as u8 => "set_safety_policy",
        x if x == ControlCommandCode::ClearMotionQueue as u8 => "clear_motion_queue",
        x if x == ControlCommandCode::DefineChirp as u8 => "define_chirp",
        x if x == ControlCommandCode::PlayFeedback as u8 => "play_feedback",
        x if x == ControlCommandCode::SetAudioSilent as u8 => "set_silent",
        x if x == ControlCommandCode::PowerState as u8 => "power_state",
        x if x == ControlCommandCode::CalibrateTurn as u8 => "calibrate_turn",
        x if x == ControlCommandCode::OrientationProbe as u8 => "orientation_probe",
        x if x == ControlCommandCode::ResetOdometry as u8 => "reset_odometry",
        x if x == ControlCommandCode::ZeroImuOrientation as u8 => "zero_imu_orientation",
        x if x == ControlCommandCode::ClearImuOrientation as u8 => "clear_imu_orientation",
        x if x == ControlCommandCode::RestartCreate as u8 => "restart_create",
        x if x == ControlCommandCode::GetCapabilities as u8 => "get_capabilities",
        _ => "none",
    }
}

#[cfg(feature = "pico-w")]
fn error_hint_text(snapshot: BrainstemStatus) -> &'static str {
    let uart_error = if snapshot.last_error_uart_read_error == UartReadErrorCode::None as u8 {
        snapshot.last_uart_read_error
    } else {
        snapshot.last_error_uart_read_error
    };

    match (snapshot.last_error, uart_error) {
        (error, uart)
            if error == ErrorCode::UartFraming as u8
                && uart == UartReadErrorCode::Framing as u8 =>
        {
            "UART RX saw an invalid stop bit before any valid Create byte; check TX/RX wiring, common ground, level shifting, baud 57600 8N1, and whether Create TX is idle-high."
        }
        (error, uart)
            if error == ErrorCode::UartFraming as u8 && uart == UartReadErrorCode::Break as u8 =>
        {
            "UART RX saw a break condition; the RX line may be held low, shorted, inverted, or connected to the wrong signal."
        }
        (error, uart)
            if error == ErrorCode::UartFraming as u8 && uart == UartReadErrorCode::Parity as u8 =>
        {
            "UART RX saw a parity mismatch; confirm the link is configured as 57600 8N1 with no parity."
        }
        (error, uart)
            if error == ErrorCode::UartFraming as u8
                && uart == UartReadErrorCode::Overrun as u8 =>
        {
            "UART RX overran; bytes arrived faster than the runtime drained them."
        }
        (error, uart)
            if error == ErrorCode::CreateNoResponse as u8
                && uart == UartReadErrorCode::Break as u8 =>
        {
            "Create did not produce valid UART bytes and RX saw a break condition; the RP2040 RX line is being held low, shorted, inverted, or connected to the wrong signal."
        }
        (error, uart)
            if error == ErrorCode::CreateNoResponse as u8
                && uart == UartReadErrorCode::Framing as u8 =>
        {
            "Create did not produce a valid sensor response and RX saw invalid stop bits; check TX/RX crossing, common ground, level shifting, and baud 57600 8N1."
        }
        (error, uart)
            if error == ErrorCode::CreateNoResponse as u8
                && uart == UartReadErrorCode::Overrun as u8 =>
        {
            "Create did not produce a complete wake-probe response before timeout; RX also overran, so stale or noisy incoming bytes may be flooding the UART."
        }
        (error, _) if error == ErrorCode::CreateNoResponse as u8 => {
            "Create did not produce any valid UART byte before the wake timeout; check power, wake wiring, Create baud, TX/RX crossing, common ground, and level shifting."
        }
        _ => "none",
    }
}

#[cfg(feature = "pico-w")]
fn wifi_state_text(code: u8) -> &'static str {
    match code {
        x if x == WifiState::Starting as u8 => "starting",
        x if x == WifiState::ApStarted as u8 => "ap_started",
        x if x == WifiState::ServicesStarted as u8 => "services_started",
        x if x == WifiState::Error as u8 => "error",
        _ => "off",
    }
}

#[cfg(feature = "pico-w")]
fn https_state_text(code: u8) -> &'static str {
    match code {
        x if x == HttpsState::Unavailable as u8 => "unavailable",
        _ => "unknown",
    }
}
