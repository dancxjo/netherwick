#[test]
fn motherbrain_reset_is_nonblocking_timed_and_deduplicated() {
    let _guard = status::status_test_guard();
    let _cleanup = ResetStatusCleanup;
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.motherbrain_reset_hardware_enabled = true;
    status::install_service_authority(11, 22, 61_000, MOTHERBRAIN_RESET_SERVICE_SCOPE);
    status::set_body_state(BodyState::Idle);
    status::set_oi_mode(CreateOiMode::Passive);
    let identity = MotherbrainResetIdentity {
        session_hash: 11,
        lease_hash: 22,
        command_id: 42,
    };

    runtime.request_motherbrain_reset(identity);
    assert_eq!(runtime.hardware.reset_levels.as_slice(), &[true]);
    assert!(runtime.active_motherbrain_reset.is_some());

    runtime.hardware.now_us += 99_000;
    runtime.poll_motherbrain_reset();
    assert_eq!(runtime.hardware.reset_levels.as_slice(), &[true]);
    runtime.hardware.now_us += 1_000;
    runtime.poll_motherbrain_reset();
    assert_eq!(runtime.hardware.reset_levels.as_slice(), &[true, false]);

    runtime.request_motherbrain_reset(identity);
    assert_eq!(runtime.hardware.reset_levels.as_slice(), &[true, false]);
    assert!(runtime
        .motherbrain_reset_history
        .iter()
        .flatten()
        .any(|record| {
            record.identity == identity && record.outcome == MotherbrainResetOutcome::Completed
        }));
}

#[test]
fn motherbrain_reset_refuses_motion_and_cooldown() {
    let _guard = status::status_test_guard();
    let _cleanup = ResetStatusCleanup;
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.motherbrain_reset_hardware_enabled = true;
    status::install_service_authority(11, 22, 61_000, MOTHERBRAIN_RESET_SERVICE_SCOPE);
    status::set_oi_mode(CreateOiMode::Full);
    status::set_body_state(BodyState::Moving);
    runtime.active = ActiveAction::Driving { stop_at_ms: 2_000 };
    runtime.request_motherbrain_reset(MotherbrainResetIdentity {
        session_hash: 11,
        lease_hash: 22,
        command_id: 1,
    });
    assert!(runtime.hardware.reset_levels.is_empty());

    runtime.active = ActiveAction::None;
    status::set_oi_mode(CreateOiMode::Passive);
    status::set_body_state(BodyState::Idle);
    runtime.request_motherbrain_reset(MotherbrainResetIdentity {
        session_hash: 11,
        lease_hash: 22,
        command_id: 2,
    });
    assert_eq!(runtime.hardware.reset_levels.as_slice(), &[true]);
    runtime.hardware.now_us += 100_000;
    runtime.poll_motherbrain_reset();
    runtime.request_motherbrain_reset(MotherbrainResetIdentity {
        session_hash: 11,
        lease_hash: 22,
        command_id: 3,
    });
    assert_eq!(runtime.hardware.reset_levels.as_slice(), &[true, false]);
}

#[test]
fn reset_replay_preserves_refusal_after_state_changes() {
    let _guard = status::status_test_guard();
    let _cleanup = ResetStatusCleanup;
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.motherbrain_reset_hardware_enabled = true;
    status::install_service_authority(31, 32, 61_000, MOTHERBRAIN_RESET_SERVICE_SCOPE);
    status::set_oi_mode(CreateOiMode::Full);
    status::set_body_state(BodyState::Moving);
    runtime.active = ActiveAction::Driving { stop_at_ms: 2_000 };
    let identity = MotherbrainResetIdentity {
        session_hash: 31,
        lease_hash: 32,
        command_id: 7,
    };

    runtime.request_motherbrain_reset(identity);
    runtime.active = ActiveAction::None;
    status::set_oi_mode(CreateOiMode::Passive);
    status::set_body_state(BodyState::Idle);
    runtime.request_motherbrain_reset(identity);

    assert!(runtime.hardware.reset_levels.is_empty());
    assert!(runtime
        .motherbrain_reset_history
        .iter()
        .flatten()
        .any(|record| {
            record.identity == identity
                && record.outcome
                    == MotherbrainResetOutcome::Refused(
                        status::MotherbrainResetRefusal::UnsafeState,
                    )
        }));
}

#[test]
fn reset_identity_includes_service_lease_and_active_pulse_is_immutable() {
    let _guard = status::status_test_guard();
    let _cleanup = ResetStatusCleanup;
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.motherbrain_reset_hardware_enabled = true;
    status::set_oi_mode(CreateOiMode::Passive);
    status::set_body_state(BodyState::Idle);
    status::install_service_authority(41, 42, 61_000, MOTHERBRAIN_RESET_SERVICE_SCOPE);
    let first = MotherbrainResetIdentity {
        session_hash: 41,
        lease_hash: 42,
        command_id: 9,
    };
    let refused_during_pulse = MotherbrainResetIdentity {
        session_hash: 41,
        lease_hash: 42,
        command_id: 10,
    };

    runtime.request_motherbrain_reset(first);
    runtime.request_motherbrain_reset(refused_during_pulse);
    runtime.hardware.now_us += 100_000;
    runtime.poll_motherbrain_reset();

    assert!(runtime
        .motherbrain_reset_history
        .iter()
        .flatten()
        .any(|record| {
            record.identity == first && record.outcome == MotherbrainResetOutcome::Completed
        }));
    assert!(runtime
        .motherbrain_reset_history
        .iter()
        .flatten()
        .any(|record| {
            record.identity == refused_during_pulse
                && record.outcome
                    == MotherbrainResetOutcome::Refused(status::MotherbrainResetRefusal::Cooldown)
        }));

    runtime.hardware.now_us += MOTHERBRAIN_RESET_COOLDOWN_MS * 1_000;
    status::install_service_authority(51, 52, 91_000, MOTHERBRAIN_RESET_SERVICE_SCOPE);
    runtime.request_motherbrain_reset(MotherbrainResetIdentity {
        session_hash: 51,
        lease_hash: 52,
        command_id: 9,
    });
    assert_eq!(
        runtime.hardware.reset_levels.as_slice(),
        &[true, false, true]
    );
}

#[test]
fn reset_command_id_zero_is_never_executed() {
    let _guard = status::status_test_guard();
    let _cleanup = ResetStatusCleanup;
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.motherbrain_reset_hardware_enabled = true;
    runtime.request_motherbrain_reset(MotherbrainResetIdentity {
        session_hash: 1,
        lease_hash: 2,
        command_id: 0,
    });
    assert!(runtime.hardware.reset_levels.is_empty());
}

#[test]
fn battery_charging_and_imu_cues_are_edge_based() {
    let _guard = status::status_test_guard();
    let _cleanup = AudioStatusCleanup;
    status::reset_audio_observability();
    status::mark_create_sensor_packet(
        0,
        crate::events::CreateSensorPacket {
            charge_mah: 1_000,
            capacity_mah: 2_000,
            ..crate::events::CreateSensorPacket::default()
        },
    );
    status::mark_imu_health(crate::drivers::imu::ImuHealth::Ok);
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.observe_audio_transitions();
    assert_eq!(status::snapshot(1_000).audio_last_requested_cue, 0);

    status::mark_create_sensor_packet(
        0,
        crate::events::CreateSensorPacket {
            charge_mah: 200,
            capacity_mah: 2_000,
            ..crate::events::CreateSensorPacket::default()
        },
    );
    runtime.observe_audio_transitions();
    assert_eq!(
        status::snapshot(1_000).audio_last_requested_cue,
        AuditoryCue::LowBattery.code()
    );
    status::reset_audio_observability();
    runtime.observe_audio_transitions();
    assert_eq!(status::snapshot(1_000).audio_last_requested_cue, 0);

    status::mark_create_sensor_packet(
        0,
        crate::events::CreateSensorPacket {
            charge_mah: 420,
            capacity_mah: 2_000,
            ..crate::events::CreateSensorPacket::default()
        },
    );
    runtime.observe_audio_transitions();
    assert_eq!(status::snapshot(1_000).audio_last_requested_cue, 0);
    status::mark_create_sensor_packet(
        0,
        crate::events::CreateSensorPacket {
            charge_mah: 520,
            capacity_mah: 2_000,
            ..crate::events::CreateSensorPacket::default()
        },
    );
    runtime.observe_audio_transitions();
    status::mark_create_sensor_packet(
        0,
        crate::events::CreateSensorPacket {
            charge_mah: 400,
            capacity_mah: 2_000,
            ..crate::events::CreateSensorPacket::default()
        },
    );
    runtime.observe_audio_transitions();
    assert_eq!(
        status::snapshot(1_000).audio_last_requested_cue,
        AuditoryCue::LowBattery.code()
    );

    status::mark_create_sensor_packet(
        0,
        crate::events::CreateSensorPacket {
            charging_state: 2,
            charge_mah: 200,
            capacity_mah: 2_000,
            ..crate::events::CreateSensorPacket::default()
        },
    );
    runtime.observe_audio_transitions();
    assert_eq!(
        status::snapshot(1_000).audio_last_requested_cue,
        AuditoryCue::DockContact.code()
    );

    status::reset_audio_observability();
    status::mark_imu_health(crate::drivers::imu::ImuHealth::Fault);
    runtime.observe_audio_transitions();
    assert_eq!(
        status::snapshot(1_000).audio_last_requested_cue,
        AuditoryCue::ImuFault.code()
    );
    status::mark_imu_health(crate::drivers::imu::ImuHealth::Ok);
    runtime.observe_audio_transitions();
    status::reset_audio_observability();
    runtime.hardware.now_us += 500_000;
    runtime.observe_audio_transitions();
    assert_eq!(
        status::snapshot(1_500).audio_last_requested_cue,
        AuditoryCue::Recovery.code()
    );
}

#[test]
fn heartbeat_and_authority_cues_deduplicate_real_transitions() {
    let _guard = status::status_test_guard();
    let _cleanup = AudioStatusCleanup;
    status::revoke_authority();
    status::reset_audio_observability();
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.heartbeat_stop_at_ms = Some(900);
    assert!(runtime.enforce_heartbeat_stop().is_ok());
    assert_eq!(
        status::snapshot(1_000).audio_last_requested_cue,
        AuditoryCue::HeartbeatLost.code()
    );
    status::reset_audio_observability();
    assert!(runtime.enforce_heartbeat_stop().is_ok());
    assert_eq!(status::snapshot(1_000).audio_last_requested_cue, 0);

    status::request_authority_transition(71, 11, 21, 10_000);
    runtime.poll_authority_transition();
    assert_eq!(
        status::snapshot(1_000).audio_last_requested_cue,
        AuditoryCue::AuthorityAcquired.code()
    );
    status::reset_audio_observability();
    status::request_authority_transition(72, 12, 22, 10_000);
    runtime.poll_authority_transition();
    assert_eq!(
        status::snapshot(1_000).audio_last_requested_cue,
        AuditoryCue::AuthorityReplaced.code()
    );
    status::revoke_authority();
}

#[test]
fn held_safety_sensor_requests_only_one_cue() {
    let _guard = status::status_test_guard();
    let _cleanup = AudioStatusCleanup;
    status::mark_create_sensor_packet(0, crate::events::CreateSensorPacket::default());
    status::reset_audio_observability();
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.create_responsive = true;
    assert!(runtime.enforce_safety_policy().is_ok());

    let cliff = crate::events::CreateSensorPacket {
        flags: crate::events::CreateSensorFlags {
            cliff_front_left: true,
            ..crate::events::CreateSensorFlags::default()
        },
        ..crate::events::CreateSensorPacket::default()
    };
    status::mark_create_sensor_packet(0, cliff);
    assert!(runtime.enforce_safety_policy().is_ok());
    assert_eq!(
        status::snapshot(1_000).audio_last_requested_cue,
        AuditoryCue::Cliff.code()
    );

    status::reset_audio_observability();
    status::mark_create_sensor_packet(0, cliff);
    assert!(runtime.enforce_safety_policy().is_ok());
    assert_eq!(status::snapshot(1_000).audio_last_requested_cue, 0);
}

#[test]
fn silent_mode_suppresses_automatic_and_direct_playback_without_replay() {
    let _guard = status::status_test_guard();
    let _cleanup = AudioStatusCleanup;
    status::reset_audio_observability();
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.create_responsive = true;
    runtime.set_audio_silent(true);
    runtime.hardware.writes.clear();

    assert!(runtime
        .enqueue_command(RuntimeCommand::PlayFeedback {
            kind: FeedbackKind::Danger,
        })
        .is_ok());
    assert!(runtime.start_next_command().is_ok());
    assert!(runtime
        .enqueue_command(RuntimeCommand::SongPlay { id: 1 })
        .is_ok());
    assert!(runtime.start_next_command().is_ok());
    runtime.request_audio(AuditoryCue::EStop);
    runtime.poll_audio();
    assert!(runtime.hardware.writes.is_empty());
    assert_eq!(status::snapshot(1_000).audio_suppressed_by_silent_count, 3);

    runtime.set_audio_silent(false);
    runtime.poll_audio();
    assert!(runtime.hardware.writes.is_empty());
}

#[test]
fn create_loss_cue_and_audio_failure_do_not_enter_runtime_error() {
    let _guard = status::status_test_guard();
    let _cleanup = AudioStatusCleanup;
    status::revoke_authority();
    status::reset_audio_observability();
    status::set_oi_mode(CreateOiMode::Full);
    let mut runtime = Runtime::new(FakeHardware::new(2_000));
    runtime.create_responsive = true;
    runtime.last_create_packet_at_ms = Some(1);
    runtime.poll();
    assert_eq!(
        status::snapshot(2_000).audio_last_requested_cue,
        AuditoryCue::CreateError.code()
    );

    runtime.create_responsive = true;
    runtime.hardware.fail_writes = true;
    runtime.request_audio(AuditoryCue::EStop);
    runtime.poll_audio();
    assert!(runtime.mode == RuntimeMode::Running);
    runtime.hardware.fail_writes = false;
    runtime.active = ActiveAction::Driving { stop_at_ms: 3_000 };
    runtime.stop_sent = false;
    assert!(runtime.stop_drive().is_ok());
    assert!(runtime.stop_sent);
}

#[test]
fn ordinary_observations_do_not_request_audio() {
    let _guard = status::status_test_guard();
    let _cleanup = AudioStatusCleanup;
    status::reset_audio_observability();
    status::mark_create_sensor_packet(0, crate::events::CreateSensorPacket::default());
    status::mark_imu_health(crate::drivers::imu::ImuHealth::Ok);
    let mut runtime = Runtime::new(FakeHardware::new(1_000));
    runtime.observe_audio_transitions();
    assert_eq!(status::snapshot(1_000).audio_last_requested_cue, 0);
}
