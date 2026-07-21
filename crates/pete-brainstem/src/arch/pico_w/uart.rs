fn submit_forebrain_stop() {
    let _ = status::submit_control_command(0, BrainstemCommand::Stop);
}

fn write_forebrain_uart_ok(uart: &mut Uart<'static, Blocking>, seq: u32) {
    let mut response = heapless::String::<32>::new();
    let _ = writeln!(response, "OK {seq}");
    write_forebrain_uart_line(uart, response.as_bytes());
}

fn write_forebrain_uart_error(uart: &mut Uart<'static, Blocking>, seq: u32, error: &str) {
    let mut response = heapless::String::<48>::new();
    let _ = writeln!(response, "ERR {seq} {error}");
    write_forebrain_uart_line(uart, response.as_bytes());
}

fn write_forebrain_uart_status(uart: &mut Uart<'static, Blocking>, seq: u32) {
    let mut response = heapless::String::<2048>::new();
    if write_compact_status_line(&mut response, seq).is_err() {
        write_forebrain_uart_error(uart, seq, "status_too_large");
        return;
    }
    write_forebrain_uart_line(uart, response.as_bytes());
}

fn write_forebrain_uart_capabilities(uart: &mut Uart<'static, Blocking>, seq: u32) {
    let mut response = heapless::String::<2048>::new();
    if capabilities::write_compact(&mut response, &capabilities::current(), seq).is_err() {
        write_forebrain_uart_error(uart, seq, "capabilities_too_large");
        return;
    }
    write_forebrain_uart_line(uart, response.as_bytes());
}

fn write_forebrain_uart_events(uart: &mut Uart<'static, Blocking>, seq: u32, since_seq: u32) {
    let mut response = heapless::String::<1024>::new();
    let _ = write!(response, "OK {seq} ");
    if status::write_compact_events(&mut response, since_seq).is_err() {
        write_forebrain_uart_error(uart, seq, "events_too_large");
        return;
    }
    write_forebrain_uart_line(uart, response.as_bytes());
}

fn write_compact_status_line<const N: usize>(
    response: &mut heapless::String<N>,
    seq: u32,
) -> core::fmt::Result {
    let snapshot = status::snapshot(Instant::now().as_millis() as u32);
    let (estop_latched, safety_tripped, motion_interlock_latched, safety_latch_kind) =
        status::session_safety_snapshot();
    let flags = snapshot.create_sensor_flags;
    writeln!(
        response,
        "OK {seq} STATUS uptime_ms={} runtime={} body={} action={} command={} pending={} error={} error_uart={} power={} oi={} armed={} estop={} safety_tripped={} safety_latch_kind={} safety_hazard_generation={} motion_interlock={} active_cmd_vel={} event_next_seq={} uart_health={} uart_error={} create_rx_bytes={} create_rx_packets={} create_last_packet_ms={} create_sensor_packet_id={} create_body_packets={} create_last_body_packet_ms={} create_last_packet_len={} charging_sources={} create_flags={} ir_byte={} bump_left={} bump_right={} wheel_drop={} cliff_left={} cliff_front_left={} cliff_front_right={} cliff_right={} create_tx_bytes={} create_last_rx_byte={} create_last_tx_byte={} create_last_rx_ms={} create_last_tx_ms={} create_rx_errors={}/{}/{}/{}/{} wake_probe={}/{} forebrain_rx_bytes={} forebrain_rx_lines={} imu_present={} imu_health={} imu_samples={} imu_age_ms={} imu_poll_ms={} imu_yaw_mrad={} imu_pitch_mrad={} imu_roll_mrad={} imu_yaw_rate_mrad_s={} imu_gyro_x_mrad_s={} imu_gyro_y_mrad_s={} imu_gyro_z_mrad_s={} imu_accel_x_mm_s2={} imu_accel_y_mm_s2={} imu_accel_z_mm_s2={} imu_accel_mag_mm_s2={} imu_tilt_mrad={} imu_roughness_mm_s2={} imu_impact_mm_s2={} imu_motion_consistency={} imu_calibration={} firmware_version={} git_commit={} git_dirty={} build_id={} careful_mode={} careful_remaining_ms={}",
        snapshot.uptime_ms,
        snapshot.current_runtime_state,
        snapshot.body_state,
        snapshot.current_runtime_action,
        snapshot.current_command,
        snapshot.pending_command,
        snapshot.last_error,
        snapshot.last_error_uart_read_error,
        snapshot.create_power_state,
        snapshot.oi_mode,
        matches!(snapshot.oi_mode, 2 | 3),
        estop_latched,
        safety_tripped,
        compact_safety_latch_kind(safety_latch_kind),
        snapshot.safety_hazard_generation,
        motion_interlock_latched,
        snapshot.body_state == status::BodyState::Moving as u8,
        snapshot.event_next_seq,
        snapshot.uart_rx_health,
        snapshot.last_uart_read_error,
        snapshot.uart_rx_bytes,
        snapshot.uart_rx_packets,
        snapshot.last_uart_packet_timestamp_ms,
        snapshot.create_sensor_last_packet_id,
        snapshot.create_sensor_complete_packet_count,
        snapshot.create_sensor_last_complete_packet_timestamp_ms,
        snapshot.last_uart_packet_len,
        snapshot.create_sensor_charging_sources,
        snapshot.create_sensor_flags,
        snapshot.create_sensor_ir_byte,
        flags & (1 << 0) != 0,
        flags & (1 << 1) != 0,
        flags & (1 << 2) != 0,
        flags & (1 << 4) != 0,
        flags & (1 << 5) != 0,
        flags & (1 << 6) != 0,
        flags & (1 << 7) != 0,
        snapshot.uart_tx_bytes,
        snapshot.last_uart_rx_byte,
        snapshot.last_uart_tx_byte,
        snapshot.last_uart_rx_timestamp_ms,
        snapshot.last_uart_tx_timestamp_ms,
        snapshot.uart_rx_overruns,
        snapshot.uart_rx_breaks,
        snapshot.uart_rx_parity_errors,
        snapshot.uart_rx_framing_errors,
        snapshot.uart_rx_other_errors,
        snapshot.wake_probe_response_bytes,
        snapshot.wake_probe_expected_bytes,
        snapshot.forebrain_uart_rx_bytes,
        snapshot.forebrain_uart_rx_lines,
        snapshot.imu_present,
        snapshot.imu_health,
        snapshot.imu_sample_count,
        snapshot.imu_sample_age_ms,
        snapshot.imu_poll_period_ms,
        snapshot.imu_yaw_mrad,
        snapshot.imu_pitch_mrad,
        snapshot.imu_roll_mrad,
        snapshot.imu_yaw_rate_mrad_s,
        snapshot.imu_gyro_x_mrad_s,
        snapshot.imu_gyro_y_mrad_s,
        snapshot.imu_gyro_z_mrad_s,
        snapshot.imu_accel_x_mm_s2,
        snapshot.imu_accel_y_mm_s2,
        snapshot.imu_accel_z_mm_s2,
        snapshot.imu_accel_magnitude_mm_s2,
        snapshot.imu_tilt_magnitude_mrad,
        snapshot.imu_roughness_mm_s2,
        snapshot.imu_impact_score_mm_s2,
        snapshot.imu_motion_consistency,
        snapshot.imu_calibration_state,
        snapshot.firmware_version,
        snapshot.git_commit,
        snapshot.git_dirty,
        snapshot.build_id,
        status::careful_mode_remaining_ms(snapshot.uptime_ms) > 0,
        status::careful_mode_remaining_ms(snapshot.uptime_ms)
    )
}

fn compact_safety_latch_kind(kind: Option<status::SafetyEventKind>) -> &'static str {
    match kind {
        None => "none",
        Some(status::SafetyEventKind::Bump) => "bump",
        Some(status::SafetyEventKind::Cliff) => "cliff",
        Some(status::SafetyEventKind::WheelDrop) => "wheel_drop",
        Some(status::SafetyEventKind::EStop) => "estop",
        Some(status::SafetyEventKind::Heartbeat) => "heartbeat",
        Some(status::SafetyEventKind::Tilt) => "tilt",
        Some(status::SafetyEventKind::Impact) => "impact",
        Some(status::SafetyEventKind::Charging) => "charging",
    }
}

fn write_forebrain_uart_line(uart: &mut Uart<'static, Blocking>, line: &[u8]) {
    if uart.blocking_write(line).is_err() || uart.blocking_flush().is_err() {
        status::mark_forebrain_uart_error(status::ForebrainUartErrorCode::Uart);
    }
}

fn request_body(request: &[u8]) -> Option<&str> {
    let body_start = request
        .windows(4)
        .position(|w| w == b"\r\n\r\n")?
        .checked_add(4)?;
    core::str::from_utf8(&request[body_start..]).ok()
}
