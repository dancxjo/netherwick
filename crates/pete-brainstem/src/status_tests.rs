    use super::*;

    #[test]
    fn prior_wire_codes_decode_only_to_the_unsupported_sentinel() {
        for kind in [
            ControlCommandCode::DriveFor,
            ControlCommandCode::BumpEscape,
            ControlCommandCode::CliffGuard,
            ControlCommandCode::SetSafetyPolicy,
        ] {
            assert!(matches!(
                decode_control_command(kind as u8, 1, 2, 3, 4, Some(1_000), 9),
                Some(BrainstemCommand::Unsupported { seq: 9 })
            ));
        }
    }

    fn collect<const N: usize>(since_seq: u32) -> heapless::Vec<PublicEventRecord, N> {
        let mut records = heapless::Vec::<PublicEventRecord, N>::new();
        collect_events_since(since_seq, &mut records);
        records
    }

    #[test]
    fn event_log_reports_oldest_next_and_dropped_before() {
        let _guard = status_test_guard();
        reset_event_log_for_test();
        mark_command_completed(10);
        mark_command_completed(11);

        assert_eq!(event_oldest_seq(), 1);
        assert_eq!(event_next_seq(), 3);
        assert_eq!(event_dropped_before_seq(0), 0);
        let records = collect::<4>(0);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].seq, 1);
        assert_eq!(records[1].seq, 2);
    }

    #[test]
    fn motion_heartbeat_does_not_shorten_control_lease() {
        let _guard = status_test_guard();
        revoke_authority();
        request_authority_transition(77, 22, 11, 5_000);
        acknowledge_authority_transition(77);

        assert!(authority_heartbeat_valid(11, 22, 1_000));
        assert!(active_authority_matches(11, 22, 2_000));
        assert!(!active_authority_matches(11, 22, 5_000));

        revoke_authority();
    }

    #[test]
    fn session_diagnostics_identify_the_active_controller_boot() {
        let _guard = status_test_guard();
        revoke_authority();
        register_diagnostic_session(41, 51, 61, 2, 1, 2);
        request_authority_transition(91, 71, 41, 5_000);
        acknowledge_authority_transition(91);
        let diagnostics = session_diagnostics(1_000);
        assert!(diagnostics.authority_active);
        assert_eq!(diagnostics.authority_generation, 91);
        assert_eq!(diagnostics.authority_session_hash, 41);
        assert_eq!(diagnostics.authority_owner_role, 2);
        assert_eq!(diagnostics.authority_owner_device_hash, 51);
        assert_eq!(diagnostics.authority_owner_boot_hash, 61);
        assert_eq!(diagnostics.authority_lease_remaining_ms, 4_000);
        revoke_authority();
    }

    #[test]
    fn event_log_ring_overwrite_reports_dropped_before_seq() {
        let _guard = status_test_guard();
        reset_event_log_for_test();
        for command_id in 0..(EVENT_LOG_CAPACITY as u32 + 4) {
            mark_command_completed(command_id);
        }

        assert_eq!(event_next_seq(), EVENT_LOG_CAPACITY as u32 + 5);
        assert_eq!(event_oldest_seq(), 5);
        assert_eq!(event_dropped_before_seq(0), 5);
        let records = collect::<EVENT_LOG_CAPACITY>(0);
        assert_eq!(records.len(), EVENT_LOG_CAPACITY);
        assert_eq!(records[0].seq, 5);
        assert_eq!(records.last().unwrap().seq, EVENT_LOG_CAPACITY as u32 + 4);
    }

    #[test]
    fn event_responses_page_through_the_larger_audit_ring() {
        let _guard = status_test_guard();
        reset_event_log_for_test();
        for command_id in 1..=20 {
            mark_command_completed(command_id);
        }

        let mut first = heapless::String::<1024>::new();
        write_compact_events(&mut first, 0).unwrap();
        assert!(first.contains("next=17"));
        assert!(first.contains("count=16"));
        assert!(first.contains("16:command_completed:16,0,0"));
        assert!(!first.contains("17:command_completed:17,0,0"));

        let mut second = heapless::String::<1024>::new();
        write_compact_events(&mut second, 16).unwrap();
        assert!(second.contains("next=21"));
        assert!(second.contains("count=4"));
        assert!(second.contains("20:command_completed:20,0,0"));
    }

    #[test]
    fn safety_preemption_returns_every_accepted_pending_command() {
        let _guard = status_test_guard();
        reset_event_log_for_test();
        PENDING_COMMAND_ID.store(70, Ordering::Relaxed);
        PENDING_COMMAND_KIND.store(ControlCommandCode::HeartbeatStop as u8, Ordering::Relaxed);
        PENDING_VELOCITY_ID.store(71, Ordering::Relaxed);
        PENDING_VELOCITY_KIND.store(ControlCommandCode::CmdVel as u8, Ordering::Relaxed);
        PENDING_VELOCITY_IS_RENEWAL.store(OFF, Ordering::Relaxed);

        let (ordinary, velocity) = preempt_pending_commands_for_safety(72);
        assert_eq!(ordinary, Some(70));
        assert_eq!(velocity, Some(71));
        assert_eq!(
            PENDING_VELOCITY_KIND.load(Ordering::Relaxed),
            ControlCommandCode::None as u8
        );
        for command_id in [ordinary, velocity].into_iter().flatten() {
            mark_command_interrupted(command_id);
        }
        let records = collect::<4>(0);
        assert!(records.iter().any(|record| {
            record.kind == PublicEventKind::CommandInterrupted as u8 && record.a == 70
        }));
        assert!(records.iter().any(|record| {
            record.kind == PublicEventKind::CommandInterrupted as u8 && record.a == 71
        }));

        PENDING_COMMAND_KIND.store(ControlCommandCode::None as u8, Ordering::Relaxed);
    }

    #[test]
    fn generic_event_name_rendering_has_stable_fallback() {
        let _guard = status_test_guard();
        assert_eq!(
            public_event_kind_text(PublicEventKind::MotionRequested as u8),
            "motion_requested"
        );
        assert_eq!(public_event_kind_text(250), "none");
    }

    #[test]
    fn command_lifecycle_event_ordering() {
        let _guard = status_test_guard();
        reset_event_log_for_test();
        mark_command_started(42, ControlCommandCode::CmdVel as u8);
        mark_command_completed(42);

        let records = collect::<4>(0);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].kind, PublicEventKind::CommandStarted as u8);
        assert_eq!(records[0].a, 42);
        assert_eq!(records[0].b, ControlCommandCode::CmdVel as u8 as u32);
        assert_eq!(records[1].kind, PublicEventKind::CommandCompleted as u8);
        assert_eq!(records[1].a, 42);
    }

    #[cfg(feature = "pico-w")]
    #[test]
    fn new_velocity_after_stop_may_restart_with_lower_sequence() {
        let _guard = status_test_guard();
        reset_event_log_for_test();
        PENDING_VELOCITY_KIND.store(ControlCommandCode::None as u8, Ordering::Relaxed);
        PENDING_VELOCITY_SEQ.store(10_000, Ordering::Relaxed);
        assert!(submit_control_command(10_001, BrainstemCommand::Stop).is_ok());
        assert_eq!(
            PENDING_VELOCITY_KIND.load(Ordering::Relaxed),
            ControlCommandCode::None as u8
        );

        assert!(submit_control_command(
            321,
            BrainstemCommand::CmdVel {
                linear_mm_s: 50,
                angular_mrad_s: 0,
                ttl_ms: 300,
                seq: 321
            }
        )
        .is_ok());
        assert_eq!(
            PENDING_VELOCITY_KIND.load(Ordering::Relaxed),
            ControlCommandCode::CmdVel as u8
        );
        assert_eq!(PENDING_VELOCITY_SEQ.load(Ordering::Relaxed), 321);
    }

    #[cfg(feature = "pico-w")]
    #[test]
    fn pending_heartbeat_refresh_is_coalesced() {
        let _guard = status_test_guard();
        reset_event_log_for_test();
        PENDING_COMMAND_KIND.store(ControlCommandCode::None as u8, Ordering::Relaxed);

        assert!(submit_control_command(
            10,
            BrainstemCommand::HeartbeatStop {
                timeout_ms: 750,
                seq: 10,
            },
        )
        .is_ok());
        assert!(submit_control_command(
            11,
            BrainstemCommand::HeartbeatStop {
                timeout_ms: 900,
                seq: 11,
            },
        )
        .is_ok());

        assert_eq!(PENDING_COMMAND_ID.load(Ordering::Relaxed), 11);
        assert_eq!(PENDING_COMMAND_SEQ.load(Ordering::Relaxed), 11);
        assert_eq!(PENDING_COMMAND_DURATION_MS.load(Ordering::Relaxed), 900);
        let records = collect::<8>(0);
        assert!(records.iter().any(|record| {
            record.kind == PublicEventKind::CommandInterrupted as u8 && record.a == 10
        }));
        assert_eq!(
            records
                .iter()
                .filter(|record| record.kind == PublicEventKind::CommandRejected as u8)
                .count(),
            0
        );
        PENDING_COMMAND_KIND.store(ControlCommandCode::None as u8, Ordering::Relaxed);
    }

    #[cfg(feature = "pico-w")]
    #[test]
    fn pending_velocity_refresh_interrupts_the_replaced_command() {
        let _guard = status_test_guard();
        reset_event_log_for_test();
        PENDING_VELOCITY_KIND.store(ControlCommandCode::None as u8, Ordering::Relaxed);

        assert!(submit_control_command(
            41,
            BrainstemCommand::CmdVel {
                linear_mm_s: 50,
                angular_mrad_s: 100,
                ttl_ms: 300,
                seq: 41,
            },
        )
        .is_ok());
        assert!(submit_control_command(
            42,
            BrainstemCommand::CmdVel {
                linear_mm_s: 50,
                angular_mrad_s: 100,
                ttl_ms: 300,
                seq: 42,
            },
        )
        .is_ok());

        assert_eq!(PENDING_VELOCITY_ID.load(Ordering::Relaxed), 42);
        let records = collect::<8>(0);
        assert!(records.iter().any(|record| {
            record.kind == PublicEventKind::CommandInterrupted as u8 && record.a == 41
        }));
        PENDING_VELOCITY_KIND.store(ControlCommandCode::None as u8, Ordering::Relaxed);
    }

    #[cfg(feature = "pico-w")]
    #[test]
    fn active_velocity_refresh_emits_one_compact_renewal_event() {
        let _guard = status_test_guard();
        reset_event_log_for_test();
        PENDING_VELOCITY_KIND.store(ControlCommandCode::None as u8, Ordering::Relaxed);
        mark_velocity_stream_active(41, 50, 100);

        assert!(submit_control_command(
            42,
            BrainstemCommand::CmdVel {
                linear_mm_s: 50,
                angular_mrad_s: 100,
                ttl_ms: 300,
                seq: 42,
            },
        )
        .is_ok());

        let records = collect::<4>(0);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].kind, PublicEventKind::CommandRenewed as u8);
        assert_eq!(records[0].a, 42);
        assert_eq!(records[0].b, 41);
        assert_eq!(records[0].c, 42);
        let _ = take_control_command();
        clear_velocity_stream();
    }

    #[cfg(feature = "pico-w")]
    #[test]
    fn stop_and_estop_interrupt_every_accepted_pending_command() {
        let _guard = status_test_guard();
        for (safety_id, safety_command) in
            [(12, BrainstemCommand::Stop), (22, BrainstemCommand::EStop)]
        {
            reset_event_log_for_test();
            PENDING_COMMAND_KIND.store(ControlCommandCode::None as u8, Ordering::Relaxed);
            PENDING_VELOCITY_KIND.store(ControlCommandCode::None as u8, Ordering::Relaxed);
            let ordinary_id = safety_id - 2;
            let velocity_id = safety_id - 1;
            assert!(submit_control_command(
                ordinary_id,
                BrainstemCommand::HeartbeatStop {
                    timeout_ms: 750,
                    seq: ordinary_id,
                },
            )
            .is_ok());
            assert!(submit_control_command(
                velocity_id,
                BrainstemCommand::CmdVel {
                    linear_mm_s: 50,
                    angular_mrad_s: 0,
                    ttl_ms: 300,
                    seq: velocity_id,
                },
            )
            .is_ok());
            assert!(submit_control_command(safety_id, safety_command).is_ok());

            let records = collect::<12>(0);
            assert!(records.iter().any(|record| {
                record.kind == PublicEventKind::CommandInterrupted as u8 && record.a == ordinary_id
            }));
            assert!(records.iter().any(|record| {
                record.kind == PublicEventKind::CommandInterrupted as u8 && record.a == velocity_id
            }));
            assert_eq!(
                PENDING_VELOCITY_KIND.load(Ordering::Relaxed),
                ControlCommandCode::None as u8
            );
            assert_eq!(PENDING_COMMAND_ID.load(Ordering::Relaxed), safety_id);
            let _ = take_control_command();
        }
    }

    #[cfg(feature = "pico-w")]
    #[test]
    fn command_rejection_event_includes_kind_and_reason() {
        let _guard = status_test_guard();
        reset_event_log_for_test();
        PENDING_COMMAND_KIND.store(ControlCommandCode::HeartbeatStop as u8, Ordering::Relaxed);

        assert_eq!(
            submit_control_command(12, BrainstemCommand::Arm),
            Err(CommandRejectReason::Busy)
        );

        let records = collect::<4>(0);
        let rejected = records
            .iter()
            .find(|record| record.kind == PublicEventKind::CommandRejected as u8)
            .unwrap();
        assert_eq!(rejected.a, 12);
        assert_eq!(rejected.b, 0);
        assert_eq!(rejected.c, ((ControlCommandCode::Arm as u32) << 8) | 1);
        PENDING_COMMAND_KIND.store(ControlCommandCode::None as u8, Ordering::Relaxed);
    }

    #[test]
    fn safety_event_ordering() {
        let _guard = status_test_guard();
        reset_event_log_for_test();
        mark_estop_latched();
        mark_safety_tripped(SafetyEventKind::EStop);
        mark_estop_cleared();
        mark_safety_cleared(SafetyEventKind::EStop);

        let records = collect::<8>(0);
        let kinds: heapless::Vec<u8, 8> = records.iter().map(|record| record.kind).collect();
        assert_eq!(
            kinds.as_slice(),
            &[
                PublicEventKind::EStopLatched as u8,
                PublicEventKind::SafetyTripped as u8,
                PublicEventKind::EStopCleared as u8,
                PublicEventKind::SafetyCleared as u8,
            ]
        );
        assert_eq!(records[1].a, SafetyEventKind::EStop as u32);
        assert_eq!(records[3].a, SafetyEventKind::EStop as u32);
    }

    #[test]
    fn charging_interlock_covers_both_charge_sources() {
        let _guard = status_test_guard();
        mark_create_sensor_packet(0, CreateSensorPacket::default());
        mark_create_charging_indicator(Some(true));
        assert!(charging_interlock_active(&snapshot(0)));

        mark_create_charging_indicator(Some(false));
        mark_create_sensor_packet(
            0,
            CreateSensorPacket {
                charging_state: 3,
                ..CreateSensorPacket::default()
            },
        );
        assert!(charging_interlock_active(&snapshot(0)));
        mark_create_sensor_packet(0, CreateSensorPacket::default());
        assert!(!charging_interlock_active(&snapshot(0)));
    }

    #[test]
    fn charging_source_packet_updates_home_base_source() {
        let _guard = status_test_guard();
        clear_create_sensor_snapshot();

        mark_create_sensor_packet(
            34,
            CreateSensorPacket {
                charging_sources: 0b10,
                ..CreateSensorPacket::default()
            },
        );

        let status = snapshot(0);
        assert_eq!(status.create_sensor_last_packet_id, 34);
        assert_eq!(status.create_sensor_charging_sources, 0b10);
    }

    #[cfg(feature = "pico-w")]
    #[test]
    fn charging_motion_reaches_runtime_for_internal_dock_departure() {
        let _guard = status_test_guard();
        PENDING_VELOCITY_KIND.store(ControlCommandCode::None as u8, Ordering::Relaxed);
        mark_create_charging_indicator(Some(true));

        assert!(submit_control_command(
            41,
            BrainstemCommand::CmdVel {
                linear_mm_s: 50,
                angular_mrad_s: 0,
                ttl_ms: 300,
                seq: 41,
            },
        )
        .is_ok());
        assert!(matches!(
            take_control_command(),
            Some(BrainstemCommand::CmdVel { seq: 41, .. })
        ));

        mark_create_charging_indicator(Some(false));
    }

    #[test]
    fn imu_sample_updates_status_without_per_sample_events() {
        let _guard = status_test_guard();
        clear_imu_orientation_calibration();
        reset_event_log_for_test();
        mark_imu_sample(ImuSample::stationary(100));
        mark_imu_sample(ImuSample {
            timestamp_ms: 120,
            gyro_z_mrad_s: 1_000,
            ..ImuSample::stationary(120)
        });

        let snapshot = snapshot(140);
        assert_eq!(snapshot.imu_present, ON);
        assert_eq!(snapshot.imu_health, ImuHealthCode::Ok as u8);
        assert_eq!(snapshot.imu_sample_age_ms, 20);
        assert_eq!(snapshot.imu_poll_period_ms, body::IMU_POLL_PERIOD_MS);
        assert_eq!(snapshot.imu_yaw_rate_mrad_s, 1_000);
        assert_eq!(snapshot.imu_yaw_mrad, 20);
        assert_eq!(snapshot.imu_accel_magnitude_mm_s2, 9_807);

        let records = collect::<8>(0);
        assert!(!records
            .iter()
            .any(|record| record.kind == PublicEventKind::ImuFrameReceived as u8));
    }

    #[test]
    fn imu_samples_do_not_evict_safety_events() {
        let _guard = status_test_guard();
        clear_imu_orientation_calibration();
        reset_event_log_for_test();
        mark_safety_tripped(SafetyEventKind::Bump);

        for timestamp_ms in 1..=(EVENT_LOG_CAPACITY as u32 * 2) {
            mark_imu_sample(ImuSample::stationary(timestamp_ms));
        }

        let records = collect::<EVENT_LOG_CAPACITY>(0);
        assert!(records.iter().any(|record| {
            record.kind == PublicEventKind::SafetyTripped as u8
                && record.a == SafetyEventKind::Bump as u32
        }));
    }

    #[test]
    fn imu_thresholds_emit_tilt_and_impact_events() {
        let _guard = status_test_guard();
        clear_imu_orientation_calibration();
        reset_event_log_for_test();
        mark_imu_sample(ImuSample::stationary(200));
        mark_imu_sample(ImuSample {
            timestamp_ms: 220,
            accel_x_mm_s2: 9_807,
            accel_y_mm_s2: 0,
            accel_z_mm_s2: 1_000,
            ..ImuSample::stationary(220)
        });
        mark_imu_sample(ImuSample {
            timestamp_ms: 240,
            accel_x_mm_s2: 0,
            accel_y_mm_s2: 0,
            accel_z_mm_s2: 30_000,
            ..ImuSample::stationary(240)
        });

        let records = collect::<12>(0);
        assert!(records
            .iter()
            .any(|record| record.kind == PublicEventKind::TiltChanged as u8 && record.a == 1));
        assert!(records
            .iter()
            .any(|record| record.kind == PublicEventKind::ImpactDetected as u8));
    }

    #[test]
    fn imu_gravity_zero_calibrates_arbitrary_mount_direction() {
        let _guard = status_test_guard();
        clear_imu_orientation_calibration();
        reset_event_log_for_test();

        let mounted_sideways = ImuSample {
            timestamp_ms: 300,
            accel_x_mm_s2: 9_807,
            accel_y_mm_s2: 0,
            accel_z_mm_s2: 0,
            ..ImuSample::stationary(300)
        };
        mark_imu_sample(mounted_sideways);
        assert!(zero_imu_orientation_from_gravity());
        assert_eq!(imu_calibrated_down(), Some(ImuVector::new(-9_807, 0, 0)));

        mark_imu_sample(ImuSample {
            timestamp_ms: 320,
            ..mounted_sideways
        });
        let snapshot = snapshot(320);

        assert_eq!(
            snapshot.imu_calibration_state,
            ImuCalibrationCode::Ready as u8
        );
        assert_eq!(snapshot.imu_pitch_mrad, 0);
        assert_eq!(snapshot.imu_roll_mrad, 0);
        assert_eq!(snapshot.imu_tilt_magnitude_mrad, 0);
    }
