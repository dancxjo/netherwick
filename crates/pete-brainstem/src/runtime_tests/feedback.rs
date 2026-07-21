#[test]
fn armed_feedback_says_fasolsi() {
    let (tones, count) = default_feedback_tones(FeedbackKind::Armed);

    assert_eq!(count, 3);
    assert_eq!(tones[0].note, 65); // fa
    assert_eq!(tones[1].note, 67); // sol
    assert_eq!(tones[2].note, 71); // si
}

#[test]
fn runtime_poll_imu_sample_updates_status() {
    let _guard = status::status_test_guard();
    status::clear_imu_orientation_calibration();
    let previous_samples = status::snapshot(1_000).imu_sample_count;
    let mut runtime = Runtime::new(FakeHardware::with_imu_sample(
        1_000,
        ImuSample {
            timestamp_ms: 1_000,
            gyro_z_mrad_s: 500,
            ..ImuSample::stationary(1_000)
        },
    ));

    runtime.tick();

    let snapshot = status::snapshot(1_000);
    assert_eq!(snapshot.imu_health, status::ImuHealthCode::Ok as u8);
    assert_eq!(snapshot.imu_sample_count, previous_samples.wrapping_add(1));
    assert_eq!(snapshot.imu_yaw_rate_mrad_s, 500);
}

#[test]
fn missing_imu_does_not_stop_motion() {
    let _guard = status::status_test_guard();
    let mut hardware = FakeHardware::new(1_000);
    hardware.imu_health = Some(crate::drivers::imu::ImuHealth::Absent);
    let mut runtime = Runtime::new(hardware);
    runtime.create_responsive = true;
    runtime.active = ActiveAction::Driving { stop_at_ms: 5_000 };
    runtime.active_command_id = Some(77);

    runtime.tick();

    let snapshot = status::snapshot(1_000);
    assert_eq!(snapshot.imu_health, status::ImuHealthCode::Absent as u8);
    assert!(matches!(runtime.active, ActiveAction::Driving { .. }));
    assert!(!runtime.safety_latched);
}

#[test]
fn persistent_imu_tilt_stops_active_motion_after_hold_window() {
    let _guard = status::status_test_guard();
    status::clear_imu_orientation_calibration();
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.create_responsive = true;
    runtime.active = ActiveAction::Driving { stop_at_ms: 5_000 };
    runtime.active_command_id = Some(77);
    status::mark_imu_sample(ImuSample {
        timestamp_ms: 1_000,
        accel_x_mm_s2: 9_807,
        accel_y_mm_s2: 0,
        accel_z_mm_s2: 1_000,
        ..ImuSample::stationary(1_000)
    });

    assert!(runtime.enforce_safety_policy().is_ok());

    assert!(matches!(runtime.active, ActiveAction::Driving { .. }));
    assert!(!runtime.safety_latched);

    runtime.hardware.delay_ms(IMU_TILT_LATCH_HOLD_MS);
    assert!(runtime.enforce_safety_policy().is_ok());

    assert!(matches!(runtime.active, ActiveAction::None));
    assert!(runtime.safety_latched);
    assert!(runtime
        .events
        .iter()
        .any(|event| matches!(event, BrainstemEvent::DriveStopped)));
    assert!(!runtime.hardware.writes.is_empty());
}

#[test]
fn transient_imu_tilt_does_not_latch_motion_safety() {
    let _guard = status::status_test_guard();
    status::clear_imu_orientation_calibration();
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.create_responsive = true;
    runtime.active = ActiveAction::Driving { stop_at_ms: 5_000 };
    runtime.active_command_id = Some(77);
    status::mark_imu_sample(ImuSample {
        timestamp_ms: 1_000,
        accel_x_mm_s2: 9_807,
        accel_y_mm_s2: 0,
        accel_z_mm_s2: 1_000,
        ..ImuSample::stationary(1_000)
    });

    assert!(runtime.enforce_safety_policy().is_ok());
    runtime.hardware.delay_ms(IMU_TILT_LATCH_HOLD_MS / 2);
    status::mark_imu_sample(ImuSample::stationary(1_050));
    assert!(runtime.enforce_safety_policy().is_ok());

    assert!(matches!(runtime.active, ActiveAction::Driving { .. }));
    assert!(!runtime.safety_latched);
    assert_eq!(runtime.tilt_observed_since_ms, None);
    assert!(!runtime
        .events
        .iter()
        .any(|event| matches!(event, BrainstemEvent::DriveStopped)));
}

#[test]
fn contact_withdrawal_runs_locally_then_stays_latched_until_clear() {
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
    runtime.active = ActiveAction::Driving { stop_at_ms: 5_000 };
    runtime.active_command_id = Some(42);
    runtime.active_velocity = Some(ActiveVelocity {
        linear_mm_s: 80,
        angular_mrad_s: 0,
    });

    assert!(runtime.enforce_safety_policy().is_ok());
    assert!(runtime.safety_latched);
    assert!(runtime.active_contact_withdrawal.is_some());
    assert_eq!(runtime.commands.len(), 1);
    assert!(matches!(
        runtime.safety_latch_kind,
        Some(status::SafetyEventKind::Bump)
    ));

    assert!(runtime.start_next_command().is_ok());
    assert!(matches!(runtime.active, ActiveAction::Driving { .. }));
    assert!(runtime.safety_recovery_motion);

    // Loss of remote authority cannot cancel a brainstem-local reflex.
    status::request_authority_transition(900, 0, 0, 0);
    runtime.poll_authority_transition();
    assert!(matches!(runtime.active, ActiveAction::Driving { .. }));
    assert!(runtime.active_contact_withdrawal.is_some());

    runtime.hardware.now_us += CONTACT_WITHDRAWAL_DURATION_MS * 1_000;
    assert!(runtime.advance_active_action().is_ok());
    assert!(runtime.active_contact_withdrawal.is_none());
    assert!(matches!(runtime.active, ActiveAction::None));

    status::mark_create_sensor_packet(0, crate::events::CreateSensorPacket::default());
    assert!(runtime.enforce_safety_policy().is_ok());
    assert!(runtime.safety_latched);
    assert_eq!(status::snapshot(1_000).create_sensor_flags & 0b11, 0);
    assert!(matches!(
        runtime.safety_latch_kind,
        Some(status::SafetyEventKind::Bump)
    ));

    assert!(runtime
        .enqueue_command(RuntimeCommand::CmdVel {
            linear_mm_s: -80,
            angular_mrad_s: 0,
            duration_ms: Some(300),
        })
        .is_ok());
    assert!(runtime.start_next_command().is_err());
    assert!(runtime.safety_latched);

    assert!(runtime
        .enqueue_command(RuntimeCommand::ClearSafetyLatch {
            kind: SafetyLatchKind::Bump,
        })
        .is_ok());
    assert!(runtime.start_next_command().is_ok());
    assert!(!runtime.safety_latched);
    assert!(runtime.safety_latch_kind.is_none());

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
    runtime.active_velocity = Some(ActiveVelocity {
        linear_mm_s: 80,
        angular_mrad_s: 0,
    });
    assert!(runtime.enforce_safety_policy().is_ok());
    assert!(runtime.safety_latched);
    assert!(matches!(
        runtime.safety_latch_kind,
        Some(status::SafetyEventKind::Bump)
    ));
}

#[test]
fn only_absolute_stop_commands_can_preempt_contact_withdrawal() {
    assert!(command_preempts_contact_withdrawal(BrainstemCommand::Stop));
    assert!(command_preempts_contact_withdrawal(BrainstemCommand::EStop));
    assert!(command_preempts_contact_withdrawal(
        BrainstemCommand::Disarm
    ));
    assert!(!command_preempts_contact_withdrawal(
        BrainstemCommand::CarefulMode {
            ttl_ms: 1_000,
            seq: 7,
        }
    ));
    assert!(!command_preempts_contact_withdrawal(
        BrainstemCommand::EscapeMotion {
            kind: SafetyLatchKind::Bump,
            hazard_generation: 42,
            linear_mm_s: -50,
            angular_mrad_s: 0,
            ttl_ms: 250,
            seq: 8,
        }
    ));
}

#[test]
fn generation_bound_bump_escape_runs_one_segment_and_new_cliff_preempts_it() {
    let _guard = status::status_test_guard();
    status::mark_create_sensor_packet(0, crate::events::CreateSensorPacket::default());
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.create_responsive = true;
    assert!(runtime.enforce_safety_policy().is_ok());
    runtime.active = ActiveAction::Driving { stop_at_ms: 5_000 };
    runtime.active_command_id = Some(41);
    runtime.active_velocity = Some(ActiveVelocity {
        linear_mm_s: 80,
        angular_mrad_s: 0,
    });

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
    assert!(runtime.enforce_safety_policy().is_ok());
    let generation = status::safety_hazard_generation();
    assert_ne!(generation, 0);
    runtime.publish_safety_snapshot();

    assert_eq!(
        status::validate_escape_motion(
            SafetyLatchKind::Bump,
            generation.wrapping_sub(1),
            -50,
            0,
            250,
        ),
        Err(pete_cockpit_protocol::CommandRejectReason::HazardMismatch)
    );
    assert_eq!(
        status::validate_escape_motion(SafetyLatchKind::Bump, generation, 50, 0, 250,),
        Err(pete_cockpit_protocol::CommandRejectReason::EscapeEnvelope)
    );
    assert!(
        status::validate_escape_motion(SafetyLatchKind::Bump, generation, -50, 0, 250,).is_ok()
    );

    assert!(runtime.start_next_command().is_ok());
    runtime.hardware.now_us += CONTACT_WITHDRAWAL_DURATION_MS * 1_000;
    assert!(runtime.advance_active_action().is_ok());
    assert!(runtime.active_contact_withdrawal.is_none());

    assert!(runtime
        .commands
        .push_back(QueuedCommand::safety_recovery(
            55,
            RuntimeCommand::EscapeMotion {
                kind: SafetyLatchKind::Bump,
                hazard_generation: generation,
                linear_mm_s: -50,
                angular_mrad_s: 0,
                ttl_ms: 250,
            },
        ))
        .is_ok());
    assert!(runtime.start_next_command().is_ok());
    assert!(runtime.active_escape.is_some());
    assert!(matches!(runtime.active, ActiveAction::Driving { .. }));

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
    assert!(runtime.active_escape.is_none());
    assert!(matches!(runtime.active, ActiveAction::None));
    assert!(matches!(
        runtime.safety_latch_kind,
        Some(status::SafetyEventKind::Cliff)
    ));
}
