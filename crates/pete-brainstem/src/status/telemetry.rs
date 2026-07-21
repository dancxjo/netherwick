pub fn set_runtime_action(action: RuntimeActionCode) {
    CURRENT_RUNTIME_ACTION.store(action as u8, Ordering::Relaxed);
}

pub fn set_create_power_on(on: bool) {
    CREATE_POWER_STATE.store(if on { ON } else { OFF }, Ordering::Relaxed);
    if !on {
        clear_create_sensor_snapshot();
    }
}

pub fn set_create_power_unknown() {
    CREATE_POWER_STATE.store(UNKNOWN, Ordering::Relaxed);
    clear_create_sensor_snapshot();
}

pub fn known_create_power_state(state: u8) -> Option<bool> {
    match state {
        OFF => Some(false),
        ON => Some(true),
        _ => None,
    }
}

pub fn mark_create_charging_indicator(active: Option<bool>) {
    CREATE_CHARGING_INDICATOR_STATE.store(
        match active {
            Some(false) => OFF,
            Some(true) => ON,
            None => UNKNOWN,
        },
        Ordering::Relaxed,
    );
}

pub fn charging_interlock_active(snapshot: &BrainstemStatus) -> bool {
    snapshot.create_charging_indicator_state == ON
        || matches!(snapshot.create_sensor_charging_state, 1..=3)
}

pub fn set_oi_mode(mode: CreateOiMode) {
    let code = match mode {
        CreateOiMode::Passive => 1,
        CreateOiMode::Safe => 2,
        CreateOiMode::Full => 3,
    };
    OI_MODE.store(code, Ordering::Relaxed);
}

pub fn set_oi_mode_unknown() {
    OI_MODE.store(UNKNOWN, Ordering::Relaxed);
}

pub fn clear_create_sensor_snapshot() {
    CREATE_SENSOR_LAST_PACKET_ID.store(0, Ordering::Relaxed);
    CREATE_SENSOR_COMPLETE_PACKET_COUNT.store(0, Ordering::Relaxed);
    CREATE_SENSOR_LAST_COMPLETE_PACKET_TIMESTAMP_MS.store(0, Ordering::Relaxed);
    CREATE_SENSOR_FLAGS.store(0, Ordering::Relaxed);
    CREATE_SENSOR_DISTANCE_MM.store(0, Ordering::Relaxed);
    CREATE_SENSOR_ANGLE_MRAD.store(0, Ordering::Relaxed);
    CREATE_SENSOR_IR_BYTE.store(0, Ordering::Relaxed);
    CREATE_SENSOR_BUTTONS.store(0, Ordering::Relaxed);
    CREATE_SENSOR_CHARGING_STATE.store(0, Ordering::Relaxed);
    CREATE_SENSOR_CHARGING_SOURCES.store(0, Ordering::Relaxed);
    CREATE_SENSOR_VOLTAGE_MV.store(0, Ordering::Relaxed);
    CREATE_SENSOR_CURRENT_MA.store(0, Ordering::Relaxed);
    CREATE_SENSOR_TEMPERATURE_C.store(0, Ordering::Relaxed);
    CREATE_SENSOR_CHARGE_MAH.store(0, Ordering::Relaxed);
    CREATE_SENSOR_CAPACITY_MAH.store(0, Ordering::Relaxed);
    CREATE_SENSOR_CLIFF_LEFT_SIGNAL.store(0, Ordering::Relaxed);
    CREATE_SENSOR_CLIFF_FRONT_LEFT_SIGNAL.store(0, Ordering::Relaxed);
    CREATE_SENSOR_CLIFF_FRONT_RIGHT_SIGNAL.store(0, Ordering::Relaxed);
    CREATE_SENSOR_CLIFF_RIGHT_SIGNAL.store(0, Ordering::Relaxed);
}

pub fn mark_uart_rx_byte(byte: u8, timestamp_ms: u32) {
    LAST_UART_RX_BYTE.store(byte, Ordering::Relaxed);
    LAST_UART_RX_TIMESTAMP_MS.store(timestamp_ms, Ordering::Relaxed);
    increment(&UART_RX_BYTES);
}

pub fn mark_uart_tx_byte(byte: u8, timestamp_ms: u32) {
    LAST_UART_TX_BYTE.store(byte, Ordering::Relaxed);
    LAST_UART_TX_TIMESTAMP_MS.store(timestamp_ms, Ordering::Relaxed);
    increment(&UART_TX_BYTES);
}

pub fn mark_uart_packet(len: usize) {
    UART_RX_HEALTH.store(ON, Ordering::Relaxed);
    LAST_UART_READ_ERROR.store(UartReadErrorCode::None as u8, Ordering::Relaxed);
    LAST_UART_PACKET_TIMESTAMP_MS.store(
        LAST_UART_RX_TIMESTAMP_MS.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    increment(&UART_RX_PACKETS);
    LAST_UART_PACKET_LEN.store(len as u32, Ordering::Relaxed);
}

pub fn mark_create_sensor_packet(packet_id: u8, sensors: CreateSensorPacket) {
    if packet_id == 0 {
        increment(&CREATE_SENSOR_COMPLETE_PACKET_COUNT);
        CREATE_SENSOR_LAST_COMPLETE_PACKET_TIMESTAMP_MS.store(
            LAST_UART_PACKET_TIMESTAMP_MS.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }
    let old_flags = CREATE_SENSOR_FLAGS.load(Ordering::Relaxed);
    let new_flags =
        merge_create_sensor_flags(packet_id, old_flags, create_sensor_flags_bits(sensors));
    let old_ir_byte = CREATE_SENSOR_IR_BYTE.load(Ordering::Relaxed);
    let old_buttons = CREATE_SENSOR_BUTTONS.load(Ordering::Relaxed);
    let old_charging_state = CREATE_SENSOR_CHARGING_STATE.load(Ordering::Relaxed);
    let old_charge = CREATE_SENSOR_CHARGE_MAH.load(Ordering::Relaxed) as u16;
    let old_capacity = CREATE_SENSOR_CAPACITY_MAH.load(Ordering::Relaxed) as u16;
    let new_charge = if create_packet_has_charge(packet_id) {
        sensors.charge_mah
    } else {
        old_charge
    };
    let new_capacity = if create_packet_has_capacity(packet_id) {
        sensors.capacity_mah
    } else {
        old_capacity
    };

    CREATE_SENSOR_LAST_PACKET_ID.store(packet_id, Ordering::Relaxed);
    if packet_id == 35 {
        OI_MODE.store(sensors.oi_mode, Ordering::Relaxed);
    }
    CREATE_SENSOR_FLAGS.store(new_flags, Ordering::Relaxed);
    CREATE_SENSOR_DISTANCE_MM.store(encode_signed_i16(sensors.distance_mm), Ordering::Relaxed);
    CREATE_SENSOR_ANGLE_MRAD.store(encode_signed_i16(sensors.angle_mrad), Ordering::Relaxed);
    if create_packet_has_ir(packet_id) {
        CREATE_SENSOR_IR_BYTE.store(sensors.ir_byte, Ordering::Relaxed);
    }
    if create_packet_has_buttons(packet_id) {
        CREATE_SENSOR_BUTTONS.store(sensors.buttons, Ordering::Relaxed);
    }
    if create_packet_has_charging_state(packet_id) {
        CREATE_SENSOR_CHARGING_STATE.store(sensors.charging_state, Ordering::Relaxed);
    }
    if create_packet_has_charging_sources(packet_id) {
        CREATE_SENSOR_CHARGING_SOURCES.store(sensors.charging_sources, Ordering::Relaxed);
    }
    if create_packet_has_voltage(packet_id) {
        CREATE_SENSOR_VOLTAGE_MV.store(sensors.voltage_mv as u32, Ordering::Relaxed);
    }
    if create_packet_has_current(packet_id) {
        CREATE_SENSOR_CURRENT_MA.store(encode_signed_i16(sensors.current_ma), Ordering::Relaxed);
    }
    if create_packet_has_temperature(packet_id) {
        CREATE_SENSOR_TEMPERATURE_C
            .store(encode_signed_i8(sensors.temperature_c), Ordering::Relaxed);
    }
    if create_packet_has_charge(packet_id) {
        CREATE_SENSOR_CHARGE_MAH.store(sensors.charge_mah as u32, Ordering::Relaxed);
    }
    if create_packet_has_capacity(packet_id) {
        CREATE_SENSOR_CAPACITY_MAH.store(sensors.capacity_mah as u32, Ordering::Relaxed);
    }
    if create_packet_has_cliff_left_signal(packet_id) {
        CREATE_SENSOR_CLIFF_LEFT_SIGNAL.store(sensors.cliff_left_signal as u32, Ordering::Relaxed);
    }
    if create_packet_has_cliff_front_left_signal(packet_id) {
        CREATE_SENSOR_CLIFF_FRONT_LEFT_SIGNAL
            .store(sensors.cliff_front_left_signal as u32, Ordering::Relaxed);
    }
    if create_packet_has_cliff_front_right_signal(packet_id) {
        CREATE_SENSOR_CLIFF_FRONT_RIGHT_SIGNAL
            .store(sensors.cliff_front_right_signal as u32, Ordering::Relaxed);
    }
    if create_packet_has_cliff_right_signal(packet_id) {
        CREATE_SENSOR_CLIFF_RIGHT_SIGNAL
            .store(sensors.cliff_right_signal as u32, Ordering::Relaxed);
    }
    // Create OI packets 0, 19, and 20 contain distance/angle deltas since
    // the last requested packet. Other packets are snapshots and must not be
    // integrated into odometry.
    let distance_mm = create_packet_has_distance_delta(packet_id)
        .then_some(sensors.distance_mm as i32)
        .unwrap_or(0);
    let angle_mrad = create_packet_has_angle_delta(packet_id)
        .then_some(sensors.angle_mrad as i32)
        .unwrap_or(0);
    if distance_mm != 0 || angle_mrad != 0 {
        integrate_odometry_delta(distance_mm, angle_mrad);
    }

    record_sensor_edge_events(
        packet_id,
        old_flags,
        new_flags,
        old_ir_byte,
        sensors.ir_byte,
        old_buttons,
        sensors.buttons,
        old_charging_state,
        sensors.charging_state,
        old_charge,
        old_capacity,
        new_charge,
        new_capacity,
    );
}

pub fn mark_song_defined(id: u8, tone_count: u8) {
    CREATE_SONG_LAST_DEFINED_ID.store(id, Ordering::Relaxed);
    CREATE_SONG_LAST_DEFINED_LEN.store(tone_count.min(MAX_SONG_TONES as u8), Ordering::Relaxed);
}

pub fn mark_song_played(id: u8) {
    CREATE_SONG_LAST_PLAYED_ID.store(id, Ordering::Relaxed);
}

pub fn audio_silent() -> bool {
    AUDIO_SILENT.load(Ordering::Acquire) == ON
}

pub fn set_audio_silent(silent: bool) {
    let new = if silent { ON } else { OFF };
    let old = AUDIO_SILENT.load(Ordering::Acquire);
    AUDIO_SILENT.store(new, Ordering::Release);
    if old != new {
        record_public_event(PublicEventKind::AudioStateChanged, silent as u32, 0, 0);
    }
}

pub fn mark_audio_cue_requested(cue: u8) {
    AUDIO_LAST_REQUESTED_CUE.store(cue, Ordering::Relaxed);
}

pub fn mark_audio_cue_played(cue: u8, timestamp_ms: u32) {
    AUDIO_LAST_PLAYED_CUE.store(cue, Ordering::Relaxed);
    AUDIO_LAST_PLAYBACK_TIMESTAMP_MS.store(timestamp_ms, Ordering::Relaxed);
}

pub fn increment_audio_suppressed_by_silent() {
    increment(&AUDIO_SUPPRESSED_BY_SILENT_COUNT);
}

pub fn increment_audio_dropped_or_replaced(count: u32) {
    if count > 0 {
        let current = AUDIO_DROPPED_OR_REPLACED_COUNT.load(Ordering::Relaxed);
        AUDIO_DROPPED_OR_REPLACED_COUNT.store(current.wrapping_add(count), Ordering::Relaxed);
    }
}

#[cfg(test)]
pub fn reset_audio_observability() {
    AUDIO_SILENT.store(OFF, Ordering::Relaxed);
    AUDIO_LAST_REQUESTED_CUE.store(0, Ordering::Relaxed);
    AUDIO_LAST_PLAYED_CUE.store(0, Ordering::Relaxed);
    AUDIO_LAST_PLAYBACK_TIMESTAMP_MS.store(0, Ordering::Relaxed);
    AUDIO_SUPPRESSED_BY_SILENT_COUNT.store(0, Ordering::Relaxed);
    AUDIO_DROPPED_OR_REPLACED_COUNT.store(0, Ordering::Relaxed);
}

pub fn mark_odometry_reset() {
    begin_odometry_write();
    increment(&ODOMETRY_RESET_COUNT);
    ODOMETRY_DISTANCE_MM.store(0, Ordering::Relaxed);
    ODOMETRY_X_MM_Q10.store(0, Ordering::Relaxed);
    ODOMETRY_Y_MM_Q10.store(0, Ordering::Relaxed);
    ODOMETRY_HEADING_MRAD.store(0, Ordering::Relaxed);
    end_odometry_write();
    IMU_YAW_MRAD.store(0, Ordering::Relaxed);
}

const ODOMETRY_POSITION_SCALE: f32 = 1024.0;

fn integrate_odometry_delta(distance_mm: i32, angle_mrad: i32) {
    begin_odometry_write();
    let heading_mrad = decode_signed_i32(ODOMETRY_HEADING_MRAD.load(Ordering::Relaxed));
    // Integrate translation at the midpoint heading so a combined Create
    // distance/angle packet follows the measured arc instead of applying the
    // whole displacement before or after the turn.
    let midpoint_rad = (heading_mrad as f32 + angle_mrad as f32 * 0.5) / 1000.0;
    let dx_q10 =
        libm::roundf(distance_mm as f32 * libm::cosf(midpoint_rad) * ODOMETRY_POSITION_SCALE)
            as i32;
    let dy_q10 =
        libm::roundf(distance_mm as f32 * libm::sinf(midpoint_rad) * ODOMETRY_POSITION_SCALE)
            as i32;
    add_signed(&ODOMETRY_X_MM_Q10, dx_q10);
    add_signed(&ODOMETRY_Y_MM_Q10, dy_q10);
    add_signed(&ODOMETRY_DISTANCE_MM, distance_mm);
    add_signed(&ODOMETRY_HEADING_MRAD, angle_mrad);
    end_odometry_write();
}

fn begin_odometry_write() {
    let sequence = ODOMETRY_SEQUENCE.load(Ordering::Relaxed);
    ODOMETRY_SEQUENCE.store(sequence.wrapping_add(1), Ordering::Release);
}

fn end_odometry_write() {
    let sequence = ODOMETRY_SEQUENCE.load(Ordering::Relaxed);
    ODOMETRY_SEQUENCE.store(sequence.wrapping_add(1), Ordering::Release);
}

const IMU_GYRO_BIAS_REQUIRED_SAMPLES: u32 = 50;

pub fn mark_imu_sample(mut sample: ImuSample) {
    update_stationary_gyro_bias(sample);
    if imu_gyro_bias_calibrated() {
        sample.gyro_x_mrad_s = sample.gyro_x_mrad_s.saturating_sub(decode_signed_i16(
            IMU_GYRO_BIAS_X_MRAD_S.load(Ordering::Relaxed),
        ));
        sample.gyro_y_mrad_s = sample.gyro_y_mrad_s.saturating_sub(decode_signed_i16(
            IMU_GYRO_BIAS_Y_MRAD_S.load(Ordering::Relaxed),
        ));
        sample.gyro_z_mrad_s = sample.gyro_z_mrad_s.saturating_sub(decode_signed_i16(
            IMU_GYRO_BIAS_Z_MRAD_S.load(Ordering::Relaxed),
        ));
    }
    let previous_timestamp = IMU_LAST_SAMPLE_TIMESTAMP_MS.load(Ordering::Relaxed);
    let previous_yaw = decode_signed_i32(IMU_YAW_MRAD.load(Ordering::Relaxed));
    let previous_accel = IMU_ACCEL_MAGNITUDE_MM_S2.load(Ordering::Relaxed) as u16;
    let calibration = imu_gravity_calibration();
    let derived = match calibration {
        Some(calibration) => derive_sample_with_gravity_calibration(
            previous_yaw,
            previous_timestamp,
            previous_accel,
            sample,
            calibration,
        ),
        None => derive_sample(previous_yaw, previous_timestamp, previous_accel, sample),
    };
    let old_health = IMU_HEALTH.load(Ordering::Relaxed);
    let old_tilt = IMU_TILT_ACTIVE.load(Ordering::Relaxed) != 0;
    let new_tilt = derived.tilt_magnitude_mrad as i16 >= body::IMU_TILT_STOP_MRAD;

    IMU_PRESENT.store(ON, Ordering::Relaxed);
    IMU_HEALTH.store(ImuHealthCode::Ok as u8, Ordering::Relaxed);
    IMU_LAST_SAMPLE_TIMESTAMP_MS.store(sample.timestamp_ms, Ordering::Relaxed);
    IMU_GYRO_X_MRAD_S.store(encode_signed_i16(sample.gyro_x_mrad_s), Ordering::Relaxed);
    IMU_GYRO_Y_MRAD_S.store(encode_signed_i16(sample.gyro_y_mrad_s), Ordering::Relaxed);
    IMU_GYRO_Z_MRAD_S.store(encode_signed_i16(sample.gyro_z_mrad_s), Ordering::Relaxed);
    IMU_ACCEL_X_MM_S2.store(encode_signed_i16(sample.accel_x_mm_s2), Ordering::Relaxed);
    IMU_ACCEL_Y_MM_S2.store(encode_signed_i16(sample.accel_y_mm_s2), Ordering::Relaxed);
    IMU_ACCEL_Z_MM_S2.store(encode_signed_i16(sample.accel_z_mm_s2), Ordering::Relaxed);
    IMU_YAW_MRAD.store(encode_signed_i32(derived.yaw_mrad), Ordering::Relaxed);
    IMU_PITCH_MRAD.store(encode_signed_i16(derived.pitch_mrad), Ordering::Relaxed);
    IMU_ROLL_MRAD.store(encode_signed_i16(derived.roll_mrad), Ordering::Relaxed);
    IMU_ACCEL_MAGNITUDE_MM_S2.store(derived.accel_magnitude_mm_s2 as u32, Ordering::Relaxed);
    IMU_TILT_MAGNITUDE_MRAD.store(derived.tilt_magnitude_mrad as u32, Ordering::Relaxed);
    IMU_ROUGHNESS_MM_S2.store(derived.roughness_mm_s2 as u32, Ordering::Relaxed);
    IMU_IMPACT_SCORE_MM_S2.store(derived.impact_score_mm_s2 as u32, Ordering::Relaxed);
    IMU_CALIBRATION_STATE.store(
        if calibration.is_some() {
            ImuCalibrationCode::Ready as u8
        } else {
            ImuCalibrationCode::Uncalibrated as u8
        },
        Ordering::Relaxed,
    );
    IMU_MOTION_CONSISTENCY.store(MotionConsistencyCode::Consistent as u8, Ordering::Relaxed);
    IMU_TILT_ACTIVE.store(new_tilt as u8, Ordering::Relaxed);
    increment(&IMU_SAMPLE_COUNT);
    if old_health != ImuHealthCode::Ok as u8 {
        record_public_event(PublicEventKind::ImuFault, ImuHealthCode::Ok as u32, 0, 0);
    }
    if old_tilt != new_tilt {
        record_public_event(
            PublicEventKind::TiltChanged,
            new_tilt as u32,
            derived.tilt_magnitude_mrad as u32,
            0,
        );
    }
    if derived.impact_score_mm_s2 >= body::IMU_IMPACT_STOP_MM_S2 {
        record_public_event(
            PublicEventKind::ImpactDetected,
            derived.impact_score_mm_s2 as u32,
            derived.accel_magnitude_mm_s2 as u32,
            0,
        );
    }
}

pub fn imu_gyro_bias_calibrated() -> bool {
    IMU_GYRO_BIAS_SAMPLE_COUNT.load(Ordering::Relaxed) >= IMU_GYRO_BIAS_REQUIRED_SAMPLES
}

fn update_stationary_gyro_bias(sample: ImuSample) {
    if imu_gyro_bias_calibrated() {
        return;
    }
    let accel = crate::drivers::imu::gravity_vector(sample);
    let accel_square = (accel.x_mm_s2 as i32)
        .saturating_mul(accel.x_mm_s2 as i32)
        .saturating_add((accel.y_mm_s2 as i32).saturating_mul(accel.y_mm_s2 as i32))
        .saturating_add((accel.z_mm_s2 as i32).saturating_mul(accel.z_mm_s2 as i32));
    let plausible_gravity = (9_200_i32.saturating_mul(9_200)..=10_400_i32.saturating_mul(10_400))
        .contains(&accel_square);
    let quiet = sample.gyro_x_mrad_s.abs() <= 80
        && sample.gyro_y_mrad_s.abs() <= 80
        && sample.gyro_z_mrad_s.abs() <= 80;
    if !plausible_gravity || !quiet {
        return;
    }
    add_signed(&IMU_GYRO_BIAS_SUM_X, sample.gyro_x_mrad_s as i32);
    add_signed(&IMU_GYRO_BIAS_SUM_Y, sample.gyro_y_mrad_s as i32);
    add_signed(&IMU_GYRO_BIAS_SUM_Z, sample.gyro_z_mrad_s as i32);
    let count = IMU_GYRO_BIAS_SAMPLE_COUNT
        .load(Ordering::Relaxed)
        .saturating_add(1);
    IMU_GYRO_BIAS_SAMPLE_COUNT.store(count, Ordering::Relaxed);
    if count == IMU_GYRO_BIAS_REQUIRED_SAMPLES {
        for (sum, bias) in [
            (&IMU_GYRO_BIAS_SUM_X, &IMU_GYRO_BIAS_X_MRAD_S),
            (&IMU_GYRO_BIAS_SUM_Y, &IMU_GYRO_BIAS_Y_MRAD_S),
            (&IMU_GYRO_BIAS_SUM_Z, &IMU_GYRO_BIAS_Z_MRAD_S),
        ] {
            let average = decode_signed_i32(sum.load(Ordering::Relaxed))
                / IMU_GYRO_BIAS_REQUIRED_SAMPLES as i32;
            bias.store(encode_signed_i16(average as i16), Ordering::Relaxed);
        }
    }
}

pub fn zero_imu_orientation_from_gravity() -> bool {
    let sample_count = IMU_SAMPLE_COUNT.load(Ordering::Relaxed);
    if sample_count == 0 {
        return false;
    }
    let sample = ImuSample {
        timestamp_ms: IMU_LAST_SAMPLE_TIMESTAMP_MS.load(Ordering::Relaxed),
        gyro_x_mrad_s: decode_signed_i16(IMU_GYRO_X_MRAD_S.load(Ordering::Relaxed)),
        gyro_y_mrad_s: decode_signed_i16(IMU_GYRO_Y_MRAD_S.load(Ordering::Relaxed)),
        gyro_z_mrad_s: decode_signed_i16(IMU_GYRO_Z_MRAD_S.load(Ordering::Relaxed)),
        accel_x_mm_s2: decode_signed_i16(IMU_ACCEL_X_MM_S2.load(Ordering::Relaxed)),
        accel_y_mm_s2: decode_signed_i16(IMU_ACCEL_Y_MM_S2.load(Ordering::Relaxed)),
        accel_z_mm_s2: decode_signed_i16(IMU_ACCEL_Z_MM_S2.load(Ordering::Relaxed)),
    };
    let Some(calibration) = ImuGravityCalibration::from_stationary_sample(sample) else {
        IMU_CALIBRATION_STATE.store(ImuCalibrationCode::Uncalibrated as u8, Ordering::Relaxed);
        return false;
    };

    IMU_GRAVITY_REF_X_MM_S2.store(
        encode_signed_i16(calibration.reference.x_mm_s2),
        Ordering::Relaxed,
    );
    IMU_GRAVITY_REF_Y_MM_S2.store(
        encode_signed_i16(calibration.reference.y_mm_s2),
        Ordering::Relaxed,
    );
    IMU_GRAVITY_REF_Z_MM_S2.store(
        encode_signed_i16(calibration.reference.z_mm_s2),
        Ordering::Relaxed,
    );
    IMU_GRAVITY_REF_MAGNITUDE_MM_S2.store(
        calibration.reference_magnitude_mm_s2 as u32,
        Ordering::Relaxed,
    );
    IMU_GRAVITY_CALIBRATED.store(1, Ordering::Relaxed);
    IMU_YAW_MRAD.store(0, Ordering::Relaxed);
    IMU_PITCH_MRAD.store(0, Ordering::Relaxed);
    IMU_ROLL_MRAD.store(0, Ordering::Relaxed);
    IMU_TILT_MAGNITUDE_MRAD.store(0, Ordering::Relaxed);
    IMU_TILT_ACTIVE.store(0, Ordering::Relaxed);
    IMU_CALIBRATION_STATE.store(ImuCalibrationCode::Ready as u8, Ordering::Relaxed);
    record_public_event(
        PublicEventKind::ImuCalibrationChanged,
        ImuCalibrationCode::Ready as u32,
        calibration.reference_magnitude_mm_s2 as u32,
        sample_count,
    );
    true
}

pub fn clear_imu_orientation_calibration() {
    IMU_GRAVITY_CALIBRATED.store(0, Ordering::Relaxed);
    IMU_GRAVITY_REF_X_MM_S2.store(0, Ordering::Relaxed);
    IMU_GRAVITY_REF_Y_MM_S2.store(0, Ordering::Relaxed);
    IMU_GRAVITY_REF_Z_MM_S2.store(0, Ordering::Relaxed);
    IMU_GRAVITY_REF_MAGNITUDE_MM_S2.store(0, Ordering::Relaxed);
    IMU_GYRO_BIAS_SAMPLE_COUNT.store(0, Ordering::Relaxed);
    IMU_GYRO_BIAS_SUM_X.store(0, Ordering::Relaxed);
    IMU_GYRO_BIAS_SUM_Y.store(0, Ordering::Relaxed);
    IMU_GYRO_BIAS_SUM_Z.store(0, Ordering::Relaxed);
    IMU_GYRO_BIAS_X_MRAD_S.store(0, Ordering::Relaxed);
    IMU_GYRO_BIAS_Y_MRAD_S.store(0, Ordering::Relaxed);
    IMU_GYRO_BIAS_Z_MRAD_S.store(0, Ordering::Relaxed);
    IMU_CALIBRATION_STATE.store(ImuCalibrationCode::Uncalibrated as u8, Ordering::Relaxed);
    record_public_event(
        PublicEventKind::ImuCalibrationChanged,
        ImuCalibrationCode::Uncalibrated as u32,
        0,
        IMU_SAMPLE_COUNT.load(Ordering::Relaxed),
    );
}

pub fn imu_calibrated_down() -> Option<ImuVector> {
    imu_gravity_calibration().map(ImuGravityCalibration::down)
}

fn imu_gravity_calibration() -> Option<ImuGravityCalibration> {
    if IMU_GRAVITY_CALIBRATED.load(Ordering::Relaxed) == 0 {
        return None;
    }
    let reference_magnitude_mm_s2 = IMU_GRAVITY_REF_MAGNITUDE_MM_S2.load(Ordering::Relaxed) as u16;
    if reference_magnitude_mm_s2 == 0 {
        return None;
    }
    Some(ImuGravityCalibration {
        reference: ImuVector::new(
            decode_signed_i16(IMU_GRAVITY_REF_X_MM_S2.load(Ordering::Relaxed)),
            decode_signed_i16(IMU_GRAVITY_REF_Y_MM_S2.load(Ordering::Relaxed)),
            decode_signed_i16(IMU_GRAVITY_REF_Z_MM_S2.load(Ordering::Relaxed)),
        ),
        reference_magnitude_mm_s2,
    })
}

pub fn mark_imu_health(health: ImuHealth) {
    let code = match health {
        ImuHealth::Unknown => ImuHealthCode::Unknown,
        ImuHealth::Ok => ImuHealthCode::Ok,
        ImuHealth::Fault => ImuHealthCode::Fault,
        ImuHealth::Absent => ImuHealthCode::Absent,
    };
    let old = IMU_HEALTH.load(Ordering::Relaxed);
    IMU_HEALTH.store(code as u8, Ordering::Relaxed);
    IMU_PRESENT.store(
        if matches!(health, ImuHealth::Absent) {
            OFF
        } else {
            ON
        },
        Ordering::Relaxed,
    );
    if old != code as u8 {
        record_public_event(PublicEventKind::ImuFault, code as u32, 0, 0);
    }
}

pub fn mark_motion_inconsistency(expected: i16, observed: i16) {
    IMU_MOTION_CONSISTENCY.store(MotionConsistencyCode::Inconsistent as u8, Ordering::Relaxed);
    record_public_event(
        PublicEventKind::MotionInconsistencyDetected,
        expected as u16 as u32,
        observed as u16 as u32,
        0,
    );
}

pub fn mark_command_completed(command_id: u32) {
    LAST_COMPLETED_COMMAND_ID.store(command_id, Ordering::Relaxed);
    record_public_event(PublicEventKind::CommandCompleted, command_id, 0, 0);
}

pub fn mark_command_interrupted(command_id: u32) {
    LAST_INTERRUPTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
    record_public_event(PublicEventKind::CommandInterrupted, command_id, 0, 0);
}

pub fn mark_command_timed_out(command_id: u32) {
    LAST_TIMED_OUT_COMMAND_ID.store(command_id, Ordering::Relaxed);
    record_public_event(PublicEventKind::CommandTimedOut, command_id, 0, 0);
}

pub fn mark_safety_tripped(kind: SafetyEventKind) {
    let generation = record_public_event(PublicEventKind::SafetyTripped, kind as u32, 0, 0);
    SAFETY_HAZARD_GENERATION.store(generation, Ordering::Release);
}

pub fn mark_safety_cleared(kind: SafetyEventKind) {
    record_public_event(PublicEventKind::SafetyCleared, kind as u32, 0, 0);
    SAFETY_HAZARD_GENERATION.store(0, Ordering::Release);
}

/// Records the start of the unconditional contact-withdrawal reflex.
///
/// Payload layout is stable and deliberately compact:
/// `a[1:0]` contact side bits, `a[15:8]` repeated-contact count;
/// `b` preempted command id; `c[15:0]` reverse speed magnitude in mm/s,
/// `c[31:16]` maximum duration in milliseconds.
pub fn mark_contact_withdrawal_started(
    contact_bits: u8,
    repeated_count: u8,
    preempted_command_id: u32,
    reverse_speed_mm_s: u16,
    duration_ms: u16,
) {
    let trigger = u32::from(contact_bits & 0b11) | (u32::from(repeated_count) << 8);
    let bounds = u32::from(reverse_speed_mm_s) | (u32::from(duration_ms) << 16);
    record_public_event(
        PublicEventKind::ContactWithdrawalStarted,
        trigger,
        preempted_command_id,
        bounds,
    );
}

/// Records the terminal reflex outcome. `a[7:0]` is the outcome,
/// `a[15:8]` is an optional dominating SafetyEventKind, and `a[16]` confirms
/// that a stop was sent. `b` is signed observed displacement encoded as i32;
/// `c` is elapsed milliseconds.
pub fn mark_contact_withdrawal_completed(
    outcome: ContactWithdrawalOutcome,
    dominating_safety: Option<SafetyEventKind>,
    final_stopped: bool,
    observed_displacement_mm: i32,
    elapsed_ms: u32,
) {
    let summary = u32::from(outcome as u8)
        | (dominating_safety.map_or(0, |kind| u32::from(kind as u8)) << 8)
        | (u32::from(final_stopped) << 16);
    record_public_event(
        PublicEventKind::ContactWithdrawalCompleted,
        summary,
        observed_displacement_mm as u32,
        elapsed_ms,
    );
}

pub fn mark_bump_changed(active: bool) {
    record_public_event(PublicEventKind::BumpChanged, active as u32, 0, 0);
}

pub fn mark_cliff_changed(active: bool) {
    record_public_event(PublicEventKind::CliffChanged, active as u32, 0, 0);
}

pub fn mark_wall_changed(active: bool) {
    record_public_event(PublicEventKind::WallChanged, active as u32, 0, 0);
}

pub fn mark_virtual_wall_changed(active: bool) {
    record_public_event(PublicEventKind::VirtualWallChanged, active as u32, 0, 0);
}

pub fn mark_battery_low(percent: u8) {
    record_public_event(PublicEventKind::BatteryLow, percent as u32, 0, 0);
}

pub fn mark_charging_state_changed(state: u8) {
    record_public_event(PublicEventKind::ChargingStateChanged, state as u32, 0, 0);
}

pub fn mark_buttons_changed(buttons: u8) {
    record_public_event(PublicEventKind::ButtonsChanged, buttons as u32, 0, 0);
}

pub fn mark_ir_changed(ir_byte: u8) {
    record_public_event(PublicEventKind::IrChanged, ir_byte as u32, 0, 0);
}

pub fn mark_wheel_drop_latched() {
    record_public_event(PublicEventKind::WheelDropLatched, 1, 0, 0);
}

pub fn mark_wheel_drop_cleared() {
    record_public_event(PublicEventKind::WheelDropCleared, 0, 0, 0);
}

pub fn mark_heartbeat_expired() {
    record_public_event(PublicEventKind::HeartbeatExpired, 0, 0, 0);
}

pub fn mark_estop_latched() {
    record_public_event(PublicEventKind::EStopLatched, 1, 0, 0);
}

pub fn mark_estop_cleared() {
    record_public_event(PublicEventKind::EStopCleared, 0, 0, 0);
}

pub fn active_service_identity() -> (u32, u32) {
    (
        ACTIVE_SERVICE_SESSION_HASH.load(Ordering::Acquire),
        ACTIVE_SERVICE_LEASE_HASH.load(Ordering::Acquire),
    )
}

pub fn mark_motherbrain_reset_requested(command_id: u32, session_hash: u32, lease_hash: u32) {
    record_public_event(
        PublicEventKind::MotherbrainResetRequested,
        command_id,
        session_hash,
        lease_hash,
    );
}

pub fn mark_motherbrain_reset_asserted(command_id: u32, session_hash: u32, lease_hash: u32) {
    record_public_event(
        PublicEventKind::MotherbrainResetAsserted,
        command_id,
        session_hash,
        lease_hash,
    );
}

pub fn mark_motherbrain_reset_completed(command_id: u32, session_hash: u32, lease_hash: u32) {
    record_public_event(
        PublicEventKind::MotherbrainResetCompleted,
        command_id,
        session_hash,
        lease_hash,
    );
}

pub fn mark_motherbrain_reset_refused(
    reason: MotherbrainResetRefusal,
    session_hash: u32,
    lease_hash: u32,
) {
    record_public_event(
        PublicEventKind::MotherbrainResetRefused,
        reason as u32,
        session_hash,
        lease_hash,
    );
}

pub fn mark_uart_rx_error() {
    UART_RX_HEALTH.store(OFF, Ordering::Relaxed);
}

pub fn mark_uart_rx_error_detail(error: UartReadError) {
    UART_RX_HEALTH.store(OFF, Ordering::Relaxed);
    let code = match error {
        UartReadError::Overrun => {
            increment(&UART_RX_OVERRUNS);
            UartReadErrorCode::Overrun
        }
        UartReadError::Break => {
            increment(&UART_RX_BREAKS);
            UartReadErrorCode::Break
        }
        UartReadError::Parity => {
            increment(&UART_RX_PARITY_ERRORS);
            UartReadErrorCode::Parity
        }
        UartReadError::Framing => {
            increment(&UART_RX_FRAMING_ERRORS);
            UartReadErrorCode::Framing
        }
        UartReadError::Other => {
            increment(&UART_RX_OTHER_ERRORS);
            UartReadErrorCode::Other
        }
    };
    LAST_UART_READ_ERROR.store(code as u8, Ordering::Relaxed);
}

#[cfg(feature = "pico-w")]
pub fn mark_forebrain_uart_rx_byte(uptime_ms: u32) {
    increment(&FOREBRAIN_UART_RX_BYTES);
    FOREBRAIN_UART_LAST_RX_MS.store(uptime_ms, Ordering::Relaxed);
}

#[cfg(feature = "pico-w")]
pub fn mark_forebrain_uart_command(seq: u32, uptime_ms: u32) {
    increment(&FOREBRAIN_UART_RX_LINES);
    FOREBRAIN_UART_LAST_SEQ.store(seq, Ordering::Relaxed);
    FOREBRAIN_UART_LAST_COMMAND_MS.store(uptime_ms, Ordering::Relaxed);
    FOREBRAIN_UART_LAST_ERROR.store(ForebrainUartErrorCode::None as u8, Ordering::Relaxed);
}

#[cfg(feature = "pico-w")]
pub fn mark_forebrain_uart_error(error: ForebrainUartErrorCode) {
    FOREBRAIN_UART_LAST_ERROR.store(error as u8, Ordering::Relaxed);
}

pub fn set_wake_probe_progress(response_bytes: u32, expected_bytes: u32) {
    WAKE_PROBE_RESPONSE_BYTES.store(response_bytes, Ordering::Relaxed);
    WAKE_PROBE_EXPECTED_BYTES.store(expected_bytes, Ordering::Relaxed);
}

pub fn set_error(error: BrainstemError) {
    let code = match error {
        BrainstemError::CreateNoResponse => ErrorCode::CreateNoResponse,
        BrainstemError::UartFraming => ErrorCode::UartFraming,
        BrainstemError::Timeout => ErrorCode::Timeout,
        BrainstemError::InvalidPacket => ErrorCode::InvalidPacket,
    };
    LAST_ERROR.store(code as u8, Ordering::Relaxed);
    LAST_ERROR_UART_READ_ERROR.store(
        LAST_UART_READ_ERROR.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    LAST_ERROR_ACTION.store(
        CURRENT_RUNTIME_ACTION.load(Ordering::Relaxed),
        Ordering::Relaxed,
    );
    set_runtime_state(RuntimeState::Error);
    set_body_state(BodyState::Error);
    request_led_blinks(8);
}

#[cfg(feature = "pico-w")]
pub fn mark_wifi_starting() {
    WIFI_STATE.store(WifiState::Starting as u8, Ordering::Relaxed);
    request_led_blinks(1);
}

#[cfg(feature = "pico-w")]
pub fn mark_wifi_ap_started() {
    WIFI_STATE.store(WifiState::ApStarted as u8, Ordering::Relaxed);
    request_led_blinks(2);
}

#[cfg(feature = "pico-w")]
pub fn mark_wifi_services_started() {
    WIFI_STATE.store(WifiState::ServicesStarted as u8, Ordering::Relaxed);
    request_led_blinks(3);
}

#[cfg(feature = "pico-w")]
#[allow(dead_code)]
pub fn mark_wifi_error() {
    WIFI_STATE.store(WifiState::Error as u8, Ordering::Relaxed);
    request_led_blinks(8);
}

#[cfg(feature = "pico-w")]
pub fn mark_http_request(uptime_ms: u32) {
    increment(&HTTP_REQUESTS);
    LAST_WEB_REQUEST_TIMESTAMP_MS.store(uptime_ms, Ordering::Relaxed);
    request_led_blinks(4);
}

#[cfg(feature = "pico-w")]
pub fn mark_http_response_flushed() {
    request_led_blinks(2);
}

#[cfg(feature = "pico-w")]
pub fn mark_http_response_error() {
    request_led_blinks(8);
}

#[cfg(feature = "pico-w")]
pub fn mark_dhcp_grant() {
    increment(&DHCP_GRANTS);
    request_led_blinks(5);
}

#[cfg(feature = "pico-w")]
pub fn mark_icmp_echo_request() {
    increment(&ICMP_ECHO_REQUESTS);
}

#[cfg(feature = "pico-w")]
pub fn mark_icmp_echo_reply() {
    increment(&ICMP_ECHO_REPLIES);
}

#[cfg(feature = "pico-w")]
pub fn mark_icmp_dropped() {
    increment(&ICMP_DROPPED);
}

#[cfg(feature = "pico-w")]
pub fn mark_icmp_rate_limited() {
    increment(&ICMP_RATE_LIMITED);
}
