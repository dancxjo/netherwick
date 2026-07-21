#[test]
fn careful_mode_opens_held_sensor_gates_for_the_active_possessor() {
    let _guard = status::status_test_guard();
    status::set_careful_mode_until(None);
    status::mark_create_sensor_packet(
        0,
        crate::events::CreateSensorPacket {
            flags: crate::events::CreateSensorFlags {
                bump_right: true,
                cliff_front_right: true,
                wheel_drop: true,
                ..crate::events::CreateSensorFlags::default()
            },
            ..crate::events::CreateSensorPacket::default()
        },
    );
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.create_responsive = true;
    runtime.latch_safety(status::SafetyEventKind::Bump);
    runtime.charging_interlock_latched = true;
    runtime.dock_departure_pending = true;

    assert!(runtime.enter_careful_mode(2_000).is_ok());
    assert!(runtime.enforce_safety_policy().is_ok());
    let now_ms = runtime.now_ms();
    assert!(runtime.start_cmd_vel(120, 0, Some(250), now_ms).is_ok());

    assert!(runtime.careful_mode_active(1_000));
    assert!(!runtime.safety_latched);
    assert!(!runtime.charging_interlock_latched);
    assert!(!runtime.dock_departure_pending);
    assert!(matches!(runtime.active, ActiveAction::Driving { .. }));
    assert_eq!(status::careful_mode_remaining_ms(1_000), 2_000);
    status::set_careful_mode_until(None);
    status::mark_create_sensor_packet(0, crate::events::CreateSensorPacket::default());
}

#[test]
fn careful_mode_expires_stopped_and_relatches_the_live_condition() {
    let _guard = status::status_test_guard();
    status::set_careful_mode_until(None);
    status::mark_create_sensor_packet(
        0,
        crate::events::CreateSensorPacket {
            flags: crate::events::CreateSensorFlags {
                bump_left: true,
                ..crate::events::CreateSensorFlags::default()
            },
            ..crate::events::CreateSensorPacket::default()
        },
    );
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.create_responsive = true;
    assert!(runtime.enter_careful_mode(250).is_ok());
    let now_ms = runtime.now_ms();
    assert!(runtime.start_cmd_vel(120, 0, Some(1_000), now_ms).is_ok());

    runtime.hardware.delay_ms(250);
    assert!(runtime.enforce_careful_mode_timeout().is_ok());

    assert!(!runtime.careful_mode_active(1_250));
    assert!(matches!(runtime.active, ActiveAction::None));
    assert!(runtime.safety_latched);
    assert!(matches!(
        runtime.safety_latch_kind,
        Some(status::SafetyEventKind::Bump)
    ));
    assert_eq!(status::careful_mode_remaining_ms(1_250), 0);
    status::mark_create_sensor_packet(0, crate::events::CreateSensorPacket::default());
}

#[test]
fn contact_edge_preempts_forward_and_starts_reverse_in_one_runtime_tick() {
    let _guard = status::status_test_guard();
    let since_seq = status::event_next_seq().saturating_sub(1);
    status::mark_create_sensor_packet(0, crate::events::CreateSensorPacket::default());
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.create_responsive = true;
    assert!(runtime.enforce_safety_policy().is_ok());
    status::mark_create_sensor_packet(
        0,
        crate::events::CreateSensorPacket {
            flags: crate::events::CreateSensorFlags {
                bump_right: true,
                ..crate::events::CreateSensorFlags::default()
            },
            ..crate::events::CreateSensorPacket::default()
        },
    );
    runtime.active = ActiveAction::Driving { stop_at_ms: 5_000 };
    runtime.active_command_id = Some(71);
    runtime.active_velocity = Some(ActiveVelocity {
        linear_mm_s: 80,
        angular_mrad_s: 0,
    });

    runtime.tick();

    assert!(runtime.active_contact_withdrawal.is_some());
    assert!(matches!(runtime.active, ActiveAction::Driving { .. }));
    assert!(runtime.safety_recovery_motion);
    let mut events = heapless::Vec::<status::PublicEventRecord, 64>::new();
    status::collect_events_since(since_seq, &mut events);
    let interrupted = events
        .iter()
        .position(|event| {
            event.kind == status::PublicEventKind::CommandInterrupted as u8 && event.a == 71
        })
        .unwrap();
    let reflex = events
        .iter()
        .position(|event| event.kind == status::PublicEventKind::ContactWithdrawalStarted as u8)
        .unwrap();
    assert!(interrupted < reflex);
}

#[test]
fn held_or_stationary_bumper_latches_without_reversing() {
    let _guard = status::status_test_guard();
    status::mark_create_sensor_packet(
        0,
        crate::events::CreateSensorPacket {
            flags: crate::events::CreateSensorFlags {
                bump_left: true,
                ..crate::events::CreateSensorFlags::default()
            },
            ..crate::events::CreateSensorPacket::default()
        },
    );
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.create_responsive = true;

    assert!(runtime.enforce_safety_policy().is_ok());
    assert!(runtime.active_contact_withdrawal.is_none());
    assert!(runtime.commands.is_empty());
    assert!(runtime.safety_latched);
    assert!(matches!(
        runtime.safety_latch_kind,
        Some(status::SafetyEventKind::Bump)
    ));

    status::mark_create_sensor_packet(0, crate::events::CreateSensorPacket::default());
    assert!(runtime.enforce_safety_policy().is_ok());
    assert!(runtime
        .enqueue_command(RuntimeCommand::ClearSafetyLatch {
            kind: SafetyLatchKind::Bump,
        })
        .is_ok());
    assert!(runtime.start_next_command().is_ok());
    assert!(!runtime.safety_latched);

    status::mark_create_sensor_packet(
        0,
        crate::events::CreateSensorPacket {
            flags: crate::events::CreateSensorFlags {
                bump_right: true,
                ..crate::events::CreateSensorFlags::default()
            },
            ..crate::events::CreateSensorPacket::default()
        },
    );
    assert!(runtime.enforce_safety_policy().is_ok());
    assert!(runtime.active_contact_withdrawal.is_none());
    assert!(runtime.commands.is_empty());
    assert!(runtime.safety_latched);
}

#[test]
fn cliff_preempts_an_active_contact_withdrawal() {
    let _guard = status::status_test_guard();
    status::mark_create_sensor_packet(0, crate::events::CreateSensorPacket::default());
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.create_responsive = true;
    assert!(runtime.enforce_safety_policy().is_ok());
    status::mark_create_sensor_packet(
        0,
        crate::events::CreateSensorPacket {
            flags: crate::events::CreateSensorFlags {
                bump_left: true,
                ..crate::events::CreateSensorFlags::default()
            },
            ..crate::events::CreateSensorPacket::default()
        },
    );
    runtime.active_velocity = Some(ActiveVelocity {
        linear_mm_s: 80,
        angular_mrad_s: 0,
    });

    assert!(runtime.enforce_safety_policy().is_ok());
    assert!(runtime.start_next_command().is_ok());
    assert!(runtime.active_contact_withdrawal.is_some());

    status::mark_create_sensor_packet(
        0,
        crate::events::CreateSensorPacket {
            flags: crate::events::CreateSensorFlags {
                bump_left: true,
                cliff_front_left: true,
                ..crate::events::CreateSensorFlags::default()
            },
            ..crate::events::CreateSensorPacket::default()
        },
    );
    assert!(runtime.enforce_safety_policy().is_ok());

    assert!(runtime.active_contact_withdrawal.is_none());
    assert!(matches!(runtime.active, ActiveAction::None));
    assert!(matches!(
        runtime.safety_latch_kind,
        Some(status::SafetyEventKind::Cliff)
    ));
}

#[test]
fn wheel_drop_latch_survives_sensor_clear_with_default_policy() {
    let _guard = status::status_test_guard();
    status::mark_create_sensor_packet(
        0,
        crate::events::CreateSensorPacket {
            flags: crate::events::CreateSensorFlags {
                wheel_drop: true,
                ..crate::events::CreateSensorFlags::default()
            },
            ..crate::events::CreateSensorPacket::default()
        },
    );
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.create_responsive = true;

    assert!(runtime.enforce_safety_policy().is_ok());
    status::mark_create_sensor_packet(0, crate::events::CreateSensorPacket::default());
    assert!(runtime.enforce_safety_policy().is_ok());

    assert!(runtime.safety_latched);
    assert!(matches!(
        runtime.safety_latch_kind,
        Some(status::SafetyEventKind::WheelDrop)
    ));
}

#[test]
fn cliff_latch_preserves_sensor_state_and_retrips_after_clear() {
    let _guard = status::status_test_guard();
    status::mark_create_sensor_packet(
        0,
        crate::events::CreateSensorPacket {
            flags: crate::events::CreateSensorFlags {
                cliff_left: true,
                ..crate::events::CreateSensorFlags::default()
            },
            ..crate::events::CreateSensorPacket::default()
        },
    );
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.create_responsive = true;

    assert!(runtime.enforce_safety_policy().is_ok());
    assert!(runtime.safety_latched);
    assert!(matches!(
        runtime.safety_latch_kind,
        Some(status::SafetyEventKind::Cliff)
    ));

    status::mark_create_sensor_packet(0, crate::events::CreateSensorPacket::default());
    assert!(runtime.enforce_safety_policy().is_ok());
    assert!(runtime.safety_latched);
    assert_eq!(status::snapshot(1_000).create_sensor_flags & 0b1111_0000, 0);

    assert!(runtime
        .enqueue_command(RuntimeCommand::ClearSafetyLatch {
            kind: SafetyLatchKind::Cliff,
        })
        .is_ok());
    assert!(runtime.start_next_command().is_ok());
    assert!(!runtime.safety_latched);

    status::mark_create_sensor_packet(
        0,
        crate::events::CreateSensorPacket {
            flags: crate::events::CreateSensorFlags {
                cliff_front_left: true,
                ..crate::events::CreateSensorFlags::default()
            },
            ..crate::events::CreateSensorPacket::default()
        },
    );
    assert!(runtime.enforce_safety_policy().is_ok());
    assert!(runtime.safety_latched);
    assert!(matches!(
        runtime.safety_latch_kind,
        Some(status::SafetyEventKind::Cliff)
    ));
}

#[test]
fn imu_zeroing_keeps_non_tilt_safety_latches_blocked() {
    let _guard = status::status_test_guard();
    status::clear_imu_orientation_calibration();
    status::mark_imu_sample(ImuSample::stationary(1_000));

    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.create_responsive = true;
    runtime.latch_safety(status::SafetyEventKind::Cliff);

    assert!(runtime
        .enqueue_command(RuntimeCommand::ZeroImuOrientation)
        .is_ok());
    assert!(runtime.start_next_command().is_ok());
    assert_eq!(
        status::snapshot(1_000).imu_calibration_state,
        status::ImuCalibrationCode::Ready as u8
    );
    assert!(runtime.safety_latched);

    assert!(runtime
        .enqueue_command(RuntimeCommand::CmdVel {
            linear_mm_s: 80,
            angular_mrad_s: 0,
            duration_ms: Some(300),
        })
        .is_ok());
    assert!(runtime.start_next_command().is_err());
    assert!(runtime.safety_latched);
}

#[test]
fn imu_zeroing_clears_tilt_latch_after_gravity_calibration() {
    let _guard = status::status_test_guard();
    status::clear_imu_orientation_calibration();
    status::mark_imu_sample(ImuSample {
        accel_x_mm_s2: 9_807,
        accel_y_mm_s2: 0,
        accel_z_mm_s2: 0,
        ..ImuSample::stationary(1_000)
    });

    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.create_responsive = true;
    runtime.latch_safety(status::SafetyEventKind::Tilt);

    assert!(runtime
        .enqueue_command(RuntimeCommand::ZeroImuOrientation)
        .is_ok());
    assert!(runtime.start_next_command().is_ok());

    assert_eq!(
        status::snapshot(1_000).imu_calibration_state,
        status::ImuCalibrationCode::Ready as u8
    );
    assert!(!runtime.safety_latched);
    assert!(runtime.safety_latch_kind.is_none());
}

#[test]
fn orientation_probe_clears_tilt_latch_and_starts_spin() {
    let _guard = status::status_test_guard();
    status::clear_imu_orientation_calibration();
    status::mark_imu_sample(ImuSample {
        accel_x_mm_s2: 9_807,
        accel_y_mm_s2: 0,
        accel_z_mm_s2: 0,
        ..ImuSample::stationary(1_000)
    });

    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.create_responsive = true;
    runtime.latch_safety(status::SafetyEventKind::Tilt);

    assert!(runtime
        .enqueue_command(RuntimeCommand::OrientationProbe {
            angular_mrad_s: 250,
            duration_ms: 400,
        })
        .is_ok());
    assert!(runtime.start_next_command().is_ok());

    assert_eq!(
        status::snapshot(1_000).imu_calibration_state,
        status::ImuCalibrationCode::Ready as u8
    );
    assert!(!runtime.safety_latched);
    assert!(runtime.safety_latch_kind.is_none());
    assert!(matches!(runtime.active, ActiveAction::Driving { .. }));
    assert_eq!(status::snapshot(1_000).odometry_reset_count, 1);
    assert!(!runtime.hardware.writes.is_empty());
}

#[test]
fn orientation_probe_does_not_clear_non_tilt_latches() {
    let _guard = status::status_test_guard();
    status::clear_imu_orientation_calibration();
    status::mark_imu_sample(ImuSample::stationary(1_000));

    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.create_responsive = true;
    runtime.latch_safety(status::SafetyEventKind::Cliff);

    assert!(runtime
        .enqueue_command(RuntimeCommand::OrientationProbe {
            angular_mrad_s: 250,
            duration_ms: 400,
        })
        .is_ok());
    assert!(runtime.start_next_command().is_err());

    assert!(runtime.safety_latched);
    assert!(matches!(
        runtime.safety_latch_kind,
        Some(status::SafetyEventKind::Cliff)
    ));
    assert!(matches!(runtime.active, ActiveAction::None));
}
