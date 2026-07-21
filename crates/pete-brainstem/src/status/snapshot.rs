#[allow(dead_code)]
pub fn snapshot(uptime_ms: u32) -> BrainstemStatus {
    let pending_command = PENDING_COMMAND_KIND.load(Ordering::Relaxed);
    let pending_command_id = if pending_command == ControlCommandCode::None as u8 {
        PENDING_VELOCITY_ID.load(Ordering::Relaxed)
    } else {
        PENDING_COMMAND_ID.load(Ordering::Relaxed)
    };
    let pending_command = if pending_command == ControlCommandCode::None as u8 {
        PENDING_VELOCITY_KIND.load(Ordering::Relaxed)
    } else {
        pending_command
    };
    let (odometry_reset_count, odometry_distance_mm, odometry_x_mm, odometry_y_mm, odometry_heading_mrad) =
        coherent_odometry_snapshot();

    BrainstemStatus {
        firmware_name: env!("CARGO_PKG_NAME"),
        firmware_version: env!("CARGO_PKG_VERSION"),
        git_commit: crate::build_identity::CURRENT.git_commit,
        git_commit_short: crate::build_identity::CURRENT.git_commit_short,
        git_dirty: crate::build_identity::CURRENT.git_dirty,
        build_timestamp: crate::build_identity::CURRENT.build_timestamp,
        build_profile: crate::build_identity::CURRENT.build_profile,
        build_target: crate::build_identity::CURRENT.build_target,
        build_backend: crate::build_identity::CURRENT.build_backend,
        build_id: crate::build_identity::CURRENT.build_id,
        body_name: body::BODY_NAME,
        body_kind: body_kind(),
        create_uart_baud: body::CREATE_UART_BAUD,
        create_sensor_probe_packet: body::CREATE_SENSOR_PROBE_PACKET,
        uptime_ms,
        current_runtime_state: RUNTIME_STATE.load(Ordering::Relaxed),
        create_power_state: CREATE_POWER_STATE.load(Ordering::Relaxed),
        oi_mode: OI_MODE.load(Ordering::Relaxed),
        uart_rx_health: UART_RX_HEALTH.load(Ordering::Relaxed),
        last_uart_packet_timestamp_ms: LAST_UART_PACKET_TIMESTAMP_MS.load(Ordering::Relaxed),
        last_uart_read_error: LAST_UART_READ_ERROR.load(Ordering::Relaxed),
        uart_rx_bytes: UART_RX_BYTES.load(Ordering::Relaxed),
        uart_rx_packets: UART_RX_PACKETS.load(Ordering::Relaxed),
        last_uart_packet_len: LAST_UART_PACKET_LEN.load(Ordering::Relaxed),
        uart_tx_bytes: UART_TX_BYTES.load(Ordering::Relaxed),
        last_uart_rx_byte: LAST_UART_RX_BYTE.load(Ordering::Relaxed),
        last_uart_tx_byte: LAST_UART_TX_BYTE.load(Ordering::Relaxed),
        last_uart_rx_timestamp_ms: LAST_UART_RX_TIMESTAMP_MS.load(Ordering::Relaxed),
        last_uart_tx_timestamp_ms: LAST_UART_TX_TIMESTAMP_MS.load(Ordering::Relaxed),
        uart_rx_overruns: UART_RX_OVERRUNS.load(Ordering::Relaxed),
        uart_rx_breaks: UART_RX_BREAKS.load(Ordering::Relaxed),
        uart_rx_parity_errors: UART_RX_PARITY_ERRORS.load(Ordering::Relaxed),
        uart_rx_framing_errors: UART_RX_FRAMING_ERRORS.load(Ordering::Relaxed),
        uart_rx_other_errors: UART_RX_OTHER_ERRORS.load(Ordering::Relaxed),
        wake_probe_response_bytes: WAKE_PROBE_RESPONSE_BYTES.load(Ordering::Relaxed),
        wake_probe_expected_bytes: WAKE_PROBE_EXPECTED_BYTES.load(Ordering::Relaxed),
        current_command: CURRENT_COMMAND.load(Ordering::Relaxed),
        current_runtime_action: CURRENT_RUNTIME_ACTION.load(Ordering::Relaxed),
        last_error: LAST_ERROR.load(Ordering::Relaxed),
        last_error_uart_read_error: LAST_ERROR_UART_READ_ERROR.load(Ordering::Relaxed),
        last_error_action: LAST_ERROR_ACTION.load(Ordering::Relaxed),
        body_state: BODY_STATE.load(Ordering::Relaxed),
        wifi_state: WIFI_STATE.load(Ordering::Relaxed),
        https_state: HTTPS_STATE.load(Ordering::Relaxed),
        http_requests: HTTP_REQUESTS.load(Ordering::Relaxed),
        dhcp_grants: DHCP_GRANTS.load(Ordering::Relaxed),
        icmp_echo_requests: ICMP_ECHO_REQUESTS.load(Ordering::Relaxed),
        icmp_echo_replies: ICMP_ECHO_REPLIES.load(Ordering::Relaxed),
        icmp_dropped: ICMP_DROPPED.load(Ordering::Relaxed),
        icmp_rate_limited: ICMP_RATE_LIMITED.load(Ordering::Relaxed),
        last_web_request_timestamp_ms: LAST_WEB_REQUEST_TIMESTAMP_MS.load(Ordering::Relaxed),
        pending_command,
        pending_command_id,
        last_accepted_command_id: LAST_ACCEPTED_COMMAND_ID.load(Ordering::Relaxed),
        last_rejected_command_id: LAST_REJECTED_COMMAND_ID.load(Ordering::Relaxed),
        last_started_command_id: LAST_STARTED_COMMAND_ID.load(Ordering::Relaxed),
        last_completed_command_id: LAST_COMPLETED_COMMAND_ID.load(Ordering::Relaxed),
        last_interrupted_command_id: LAST_INTERRUPTED_COMMAND_ID.load(Ordering::Relaxed),
        last_timed_out_command_id: LAST_TIMED_OUT_COMMAND_ID.load(Ordering::Relaxed),
        forebrain_uart_rx_bytes: FOREBRAIN_UART_RX_BYTES.load(Ordering::Relaxed),
        forebrain_uart_rx_lines: FOREBRAIN_UART_RX_LINES.load(Ordering::Relaxed),
        forebrain_uart_last_seq: FOREBRAIN_UART_LAST_SEQ.load(Ordering::Relaxed),
        forebrain_uart_last_error: FOREBRAIN_UART_LAST_ERROR.load(Ordering::Relaxed),
        forebrain_uart_link_alive_ms: elapsed_since(
            uptime_ms,
            FOREBRAIN_UART_LAST_RX_MS.load(Ordering::Relaxed),
        ),
        forebrain_uart_last_command_age_ms: elapsed_since(
            uptime_ms,
            FOREBRAIN_UART_LAST_COMMAND_MS.load(Ordering::Relaxed),
        ),
        create_sensor_last_packet_id: CREATE_SENSOR_LAST_PACKET_ID.load(Ordering::Relaxed),
        create_sensor_complete_packet_count: CREATE_SENSOR_COMPLETE_PACKET_COUNT
            .load(Ordering::Relaxed),
        create_sensor_last_complete_packet_timestamp_ms:
            CREATE_SENSOR_LAST_COMPLETE_PACKET_TIMESTAMP_MS.load(Ordering::Relaxed),
        create_sensor_flags: CREATE_SENSOR_FLAGS.load(Ordering::Relaxed),
        create_sensor_distance_mm: decode_signed_i16(
            CREATE_SENSOR_DISTANCE_MM.load(Ordering::Relaxed),
        ),
        create_sensor_angle_mrad: decode_signed_i16(
            CREATE_SENSOR_ANGLE_MRAD.load(Ordering::Relaxed),
        ),
        create_sensor_ir_byte: CREATE_SENSOR_IR_BYTE.load(Ordering::Relaxed),
        create_sensor_buttons: CREATE_SENSOR_BUTTONS.load(Ordering::Relaxed),
        create_sensor_charging_state: CREATE_SENSOR_CHARGING_STATE.load(Ordering::Relaxed),
        create_sensor_charging_sources: CREATE_SENSOR_CHARGING_SOURCES.load(Ordering::Relaxed),
        create_charging_indicator_state: CREATE_CHARGING_INDICATOR_STATE.load(Ordering::Relaxed),
        create_sensor_voltage_mv: CREATE_SENSOR_VOLTAGE_MV.load(Ordering::Relaxed) as u16,
        create_sensor_current_ma: decode_signed_i16(
            CREATE_SENSOR_CURRENT_MA.load(Ordering::Relaxed),
        ),
        create_sensor_temperature_c: decode_signed_i8(
            CREATE_SENSOR_TEMPERATURE_C.load(Ordering::Relaxed),
        ),
        create_sensor_charge_mah: CREATE_SENSOR_CHARGE_MAH.load(Ordering::Relaxed) as u16,
        create_sensor_capacity_mah: CREATE_SENSOR_CAPACITY_MAH.load(Ordering::Relaxed) as u16,
        create_sensor_cliff_left_signal: CREATE_SENSOR_CLIFF_LEFT_SIGNAL.load(Ordering::Relaxed)
            as u16,
        create_sensor_cliff_front_left_signal: CREATE_SENSOR_CLIFF_FRONT_LEFT_SIGNAL
            .load(Ordering::Relaxed) as u16,
        create_sensor_cliff_front_right_signal: CREATE_SENSOR_CLIFF_FRONT_RIGHT_SIGNAL
            .load(Ordering::Relaxed) as u16,
        create_sensor_cliff_right_signal: CREATE_SENSOR_CLIFF_RIGHT_SIGNAL.load(Ordering::Relaxed)
            as u16,
        create_song_last_defined_id: CREATE_SONG_LAST_DEFINED_ID.load(Ordering::Relaxed),
        create_song_last_defined_len: CREATE_SONG_LAST_DEFINED_LEN.load(Ordering::Relaxed),
        create_song_last_played_id: CREATE_SONG_LAST_PLAYED_ID.load(Ordering::Relaxed),
        audio_silent: audio_silent(),
        audio_last_requested_cue: AUDIO_LAST_REQUESTED_CUE.load(Ordering::Relaxed),
        audio_last_played_cue: AUDIO_LAST_PLAYED_CUE.load(Ordering::Relaxed),
        audio_last_playback_timestamp_ms: AUDIO_LAST_PLAYBACK_TIMESTAMP_MS.load(Ordering::Relaxed),
        audio_suppressed_by_silent_count: AUDIO_SUPPRESSED_BY_SILENT_COUNT.load(Ordering::Relaxed),
        audio_dropped_or_replaced_count: AUDIO_DROPPED_OR_REPLACED_COUNT.load(Ordering::Relaxed),
        odometry_reset_count,
        odometry_distance_mm,
        odometry_x_mm,
        odometry_y_mm,
        odometry_heading_mrad,
        imu_present: IMU_PRESENT.load(Ordering::Relaxed),
        imu_health: IMU_HEALTH.load(Ordering::Relaxed),
        imu_sample_count: IMU_SAMPLE_COUNT.load(Ordering::Relaxed),
        imu_last_sample_timestamp_ms: IMU_LAST_SAMPLE_TIMESTAMP_MS.load(Ordering::Relaxed),
        imu_sample_age_ms: elapsed_since(
            uptime_ms,
            IMU_LAST_SAMPLE_TIMESTAMP_MS.load(Ordering::Relaxed),
        ),
        imu_poll_period_ms: body::IMU_POLL_PERIOD_MS,
        imu_yaw_mrad: decode_signed_i32(IMU_YAW_MRAD.load(Ordering::Relaxed)),
        imu_pitch_mrad: decode_signed_i16(IMU_PITCH_MRAD.load(Ordering::Relaxed)),
        imu_roll_mrad: decode_signed_i16(IMU_ROLL_MRAD.load(Ordering::Relaxed)),
        imu_yaw_rate_mrad_s: decode_signed_i16(IMU_GYRO_Z_MRAD_S.load(Ordering::Relaxed)),
        imu_gyro_x_mrad_s: decode_signed_i16(IMU_GYRO_X_MRAD_S.load(Ordering::Relaxed)),
        imu_gyro_y_mrad_s: decode_signed_i16(IMU_GYRO_Y_MRAD_S.load(Ordering::Relaxed)),
        imu_gyro_z_mrad_s: decode_signed_i16(IMU_GYRO_Z_MRAD_S.load(Ordering::Relaxed)),
        imu_accel_x_mm_s2: decode_signed_i16(IMU_ACCEL_X_MM_S2.load(Ordering::Relaxed)),
        imu_accel_y_mm_s2: decode_signed_i16(IMU_ACCEL_Y_MM_S2.load(Ordering::Relaxed)),
        imu_accel_z_mm_s2: decode_signed_i16(IMU_ACCEL_Z_MM_S2.load(Ordering::Relaxed)),
        imu_accel_magnitude_mm_s2: IMU_ACCEL_MAGNITUDE_MM_S2.load(Ordering::Relaxed) as u16,
        imu_tilt_magnitude_mrad: IMU_TILT_MAGNITUDE_MRAD.load(Ordering::Relaxed) as u16,
        imu_roughness_mm_s2: IMU_ROUGHNESS_MM_S2.load(Ordering::Relaxed) as u16,
        imu_impact_score_mm_s2: IMU_IMPACT_SCORE_MM_S2.load(Ordering::Relaxed) as u16,
        imu_motion_consistency: IMU_MOTION_CONSISTENCY.load(Ordering::Relaxed),
        imu_calibration_state: IMU_CALIBRATION_STATE.load(Ordering::Relaxed),
        safety_hazard_generation: safety_hazard_generation(),
        event_next_seq: EVENT_NEXT_SEQ.load(Ordering::Relaxed),
    }
}

fn coherent_odometry_snapshot() -> (u32, i32, i32, i32, i32) {
    loop {
        let before = ODOMETRY_SEQUENCE.load(Ordering::Acquire);
        if before & 1 != 0 {
            core::hint::spin_loop();
            continue;
        }
        let values = (
            ODOMETRY_RESET_COUNT.load(Ordering::Relaxed),
            decode_signed_i32(ODOMETRY_DISTANCE_MM.load(Ordering::Relaxed)),
            decode_signed_i32(ODOMETRY_X_MM_Q10.load(Ordering::Relaxed)) / 1024,
            decode_signed_i32(ODOMETRY_Y_MM_Q10.load(Ordering::Relaxed)) / 1024,
            decode_signed_i32(ODOMETRY_HEADING_MRAD.load(Ordering::Relaxed)),
        );
        let after = ODOMETRY_SEQUENCE.load(Ordering::Acquire);
        if before == after {
            return values;
        }
    }
}

fn elapsed_since(now_ms: u32, timestamp_ms: u32) -> u32 {
    if timestamp_ms == 0 {
        0
    } else {
        now_ms.wrapping_sub(timestamp_ms)
    }
}
