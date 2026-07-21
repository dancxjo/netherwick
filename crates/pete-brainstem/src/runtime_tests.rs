    use super::*;
    use crate::commands::CreateOiMode;
    use crate::drivers::imu::ImuSample;
    use crate::hardware::SerialRead;

    struct ResetStatusCleanup;

    impl Drop for ResetStatusCleanup {
        fn drop(&mut self) {
            status::set_oi_mode_unknown();
            status::set_body_state(BodyState::NotStarted);
            status::revoke_service_authority();
        }
    }

    struct AudioStatusCleanup;

    impl Drop for AudioStatusCleanup {
        fn drop(&mut self) {
            status::reset_audio_observability();
            status::mark_create_sensor_packet(0, crate::events::CreateSensorPacket::default());
            status::mark_imu_health(crate::drivers::imu::ImuHealth::Unknown);
            status::revoke_authority();
            status::set_oi_mode_unknown();
            status::reset_event_log_for_test();
        }
    }

    struct FakeHardware {
        now_us: u32,
        writes: heapless::Vec<u8, 256>,
        imu_sample: Option<ImuSample>,
        imu_health: Option<crate::drivers::imu::ImuHealth>,
        reset_levels: heapless::Vec<bool, 8>,
        power_toggle_levels: heapless::Vec<bool, 8>,
        charging_indicator: Option<bool>,
        fail_writes: bool,
    }

    impl FakeHardware {
        fn new(now_ms: u32) -> Self {
            Self {
                now_us: now_ms * 1_000,
                writes: heapless::Vec::new(),
                imu_sample: None,
                imu_health: None,
                reset_levels: heapless::Vec::new(),
                power_toggle_levels: heapless::Vec::new(),
                charging_indicator: None,
                fail_writes: false,
            }
        }

        fn with_imu_sample(now_ms: u32, imu_sample: ImuSample) -> Self {
            Self {
                now_us: now_ms * 1_000,
                writes: heapless::Vec::new(),
                imu_sample: Some(imu_sample),
                imu_health: None,
                reset_levels: heapless::Vec::new(),
                power_toggle_levels: heapless::Vec::new(),
                charging_indicator: None,
                fail_writes: false,
            }
        }
    }

    impl BrainstemHardware for FakeHardware {
        fn delay_ms(&mut self, ms: u32) {
            self.now_us = self.now_us.wrapping_add(ms * 1_000);
        }

        fn now_us(&mut self) -> u32 {
            self.now_us
        }

        fn feed_watchdog(&mut self) {}

        fn begin_power_toggle_pulse(&mut self) {
            let _ = self.power_toggle_levels.push(false);
            let _ = self.power_toggle_levels.push(true);
        }

        fn end_power_toggle_pulse(&mut self) {
            let _ = self.power_toggle_levels.push(false);
        }

        fn set_indicators(&mut self, _on: bool) {}

        fn set_primary_indicator(&mut self, _on: bool) {}

        fn set_motherbrain_reset(&mut self, asserted: bool) {
            let _ = self.reset_levels.push(asserted);
        }

        fn write_byte(&mut self, byte: u8) -> Result<(), ()> {
            if self.fail_writes {
                return Err(());
            }
            let _ = self.writes.push(byte);
            Ok(())
        }

        fn flush_uart(&mut self) -> Result<(), ()> {
            Ok(())
        }

        fn read_byte(&mut self) -> SerialRead {
            SerialRead::WouldBlock
        }

        fn poll_imu_sample(
            &mut self,
            _now_ms: u32,
        ) -> Result<Option<ImuSample>, crate::drivers::imu::ImuHealth> {
            if let Some(health) = self.imu_health {
                return Err(health);
            }
            Ok(self.imu_sample.take())
        }

        fn charging_indicator_active(&mut self) -> Option<bool> {
            self.charging_indicator
        }
    }

    #[test]
    fn healthy_supervision_lights_keep_power_amber_and_alternate_buttons() {
        assert_eq!(
            healthy_supervision_lights(0),
            (
                CREATE_LED_PLAY,
                CONNECTED_POWER_LED_COLOR,
                CONNECTED_POWER_LED_INTENSITY,
                HEALTHY_LIGHT_STEP_MS
            )
        );
        assert_eq!(
            healthy_supervision_lights(8),
            (
                CREATE_LED_ADVANCE,
                CONNECTED_POWER_LED_COLOR,
                CONNECTED_POWER_LED_INTENSITY,
                HEALTHY_LIGHT_STEP_MS
            )
        );
        assert_eq!(
            healthy_supervision_lights(15),
            (
                CREATE_LED_ADVANCE,
                CONNECTED_POWER_LED_COLOR,
                CONNECTED_POWER_LED_INTENSITY,
                HEALTHY_LIGHT_STEP_MS
            )
        );
        assert_eq!(
            healthy_supervision_lights(16),
            (
                CREATE_LED_PLAY,
                CONNECTED_POWER_LED_COLOR,
                CONNECTED_POWER_LED_INTENSITY,
                HEALTHY_LIGHT_STEP_MS
            )
        );
        assert_eq!(
            healthy_supervision_lights(32),
            healthy_supervision_lights(0)
        );
    }

    #[test]
    fn startup_acquires_create_in_full_mode() {
        let _guard = status::status_test_guard();
        let mut runtime = Runtime::new(FakeHardware::new(0));

        runtime.start();

        assert_eq!(runtime.commands.len(), ACQUIRE_CREATE_SCRIPT.len());
        assert!(runtime
            .commands
            .iter()
            .any(|queued| matches!(queued.command, RuntimeCommand::WakeCreate)));
        assert!(runtime.commands.iter().any(|queued| matches!(
            queued.command,
            RuntimeCommand::SetMode(crate::commands::CreateOiMode::Full)
        )));
        assert!(matches!(runtime.mode, RuntimeMode::Running));
        assert_eq!(
            status::snapshot(0).current_runtime_state,
            RuntimeState::Running as u8
        );
        assert!(
            runtime.hardware.power_toggle_levels.is_empty(),
            "ordinary startup must not toggle Create power"
        );
    }

    #[test]
    fn shutdown_sends_a_final_create_stop_and_clears_runtime_state() {
        let _guard = status::status_test_guard();
        let mut runtime = Runtime::new(FakeHardware::new(1_000));
        runtime.active = ActiveAction::Driving { stop_at_ms: 2_000 };
        status::set_runtime_state(RuntimeState::Running);
        status::set_body_state(BodyState::Moving);

        runtime.shutdown();

        assert!(runtime
            .hardware
            .writes
            .windows(5)
            .any(|bytes| bytes == [137, 0, 0, 0, 0]));
        let snapshot = status::snapshot(1_000);
        assert_eq!(snapshot.current_runtime_state, RuntimeState::Idle as u8);
        assert_eq!(snapshot.body_state, BodyState::Idle as u8);
    }

    #[test]
    fn wake_create_pulses_power_once_when_create_is_known_off() {
        let _guard = status::status_test_guard();
        status::set_create_power_on(false);
        let mut runtime = Runtime::new(FakeHardware::new(1_000));

        assert_eq!(body::POWER_TOGGLE_PULSE_MS, 500);
        assert!(runtime.enqueue_command(RuntimeCommand::WakeCreate).is_ok());
        assert!(runtime.start_next_command().is_ok());

        assert_eq!(
            runtime.hardware.power_toggle_levels.as_slice(),
            &[false, true]
        );
        assert!(matches!(
            runtime.active,
            ActiveAction::PowerPulse {
                power_on: true,
                wake_wait_until_ms: Some(_),
                ..
            }
        ));

        runtime.hardware.delay_ms(body::POWER_TOGGLE_PULSE_MS - 1);
        assert!(runtime.advance_active_action().is_ok());
        assert_eq!(
            runtime.hardware.power_toggle_levels.as_slice(),
            &[false, true],
            "POWER_TOGGLE must remain high for the full 500 ms"
        );

        runtime.hardware.delay_ms(1);
        assert!(runtime.advance_active_action().is_ok());

        assert_eq!(
            runtime.hardware.power_toggle_levels.as_slice(),
            &[false, true, false]
        );
        assert_eq!(
            runtime
                .hardware
                .power_toggle_levels
                .windows(2)
                .filter(|levels| levels == &[false, true])
                .count(),
            1,
            "one request must create exactly one low-to-high edge"
        );
        assert_eq!(
            runtime.hardware.power_toggle_levels.last(),
            Some(&false),
            "POWER_TOGGLE must return low after the pulse"
        );
        assert_eq!(
            runtime
                .events
                .iter()
                .filter(|event| matches!(event, BrainstemEvent::CreatePowerOnRequested))
                .count(),
            1
        );
    }

    #[test]
    fn repeated_sleep_cannot_toggle_a_known_off_create_back_on() {
        let _guard = status::status_test_guard();
        status::set_create_power_on(true);
        let mut runtime = Runtime::new(FakeHardware::new(1_000));

        assert!(runtime.enqueue_command(RuntimeCommand::SleepCreate).is_ok());
        assert!(runtime.start_next_command().is_ok());
        assert_eq!(
            runtime.hardware.power_toggle_levels.as_slice(),
            &[false, true]
        );

        runtime.hardware.delay_ms(body::POWER_TOGGLE_PULSE_MS);
        assert!(runtime.advance_active_action().is_ok());
        assert_eq!(
            runtime.hardware.power_toggle_levels.as_slice(),
            &[false, true, false]
        );
        assert_eq!(
            status::known_create_power_state(status::snapshot(runtime.now_ms()).create_power_state),
            Some(false)
        );

        assert!(runtime.enqueue_command(RuntimeCommand::SleepCreate).is_ok());
        assert!(runtime.start_next_command().is_ok());
        assert_eq!(
            runtime.hardware.power_toggle_levels.as_slice(),
            &[false, true, false],
            "sleeping a known-OFF Create must not produce another edge"
        );
    }

    #[test]
    fn sleep_with_unknown_power_state_refuses_without_pulsing() {
        let _guard = status::status_test_guard();
        status::set_create_power_unknown();
        let mut runtime = Runtime::new(FakeHardware::new(1_000));
        assert!(runtime
            .commands
            .push_back(QueuedCommand::new(55, RuntimeCommand::SleepCreate))
            .is_ok());

        assert!(runtime.start_next_command().is_ok());

        assert!(runtime.hardware.power_toggle_levels.is_empty());
        assert_eq!(
            status::snapshot(runtime.now_ms()).last_interrupted_command_id,
            55
        );
    }

    #[test]
    fn known_on_wake_timeout_never_toggles_create_power() {
        let _guard = status::status_test_guard();
        status::set_create_power_on(true);
        let mut runtime = Runtime::new(FakeHardware::new(1_000));
        assert!(runtime
            .commands
            .push_back(QueuedCommand::new(61, RuntimeCommand::WakeCreate))
            .is_ok());

        assert!(runtime.start_next_command().is_ok());
        let deadline_ms = match runtime.active {
            ActiveAction::WaitForCreate {
                deadline_ms,
                allow_power_toggle_on_timeout: false,
                ..
            } => deadline_ms,
            _ => panic!("known-ON wake must start with a probe-only wait"),
        };
        assert!(runtime.advance_active_action().is_ok());
        assert!(
            runtime.hardware.writes.contains(&128),
            "wake must probe OI first"
        );

        runtime.hardware.now_us = deadline_ms * 1_000;
        assert!(runtime.advance_active_action().is_ok());

        assert!(runtime.hardware.power_toggle_levels.is_empty());
        assert!(runtime.commands.is_empty());
        assert_eq!(
            status::snapshot(runtime.now_ms()).last_timed_out_command_id,
            61
        );
    }

    #[test]
    fn unknown_wake_allows_at_most_one_best_effort_pulse() {
        let _guard = status::status_test_guard();
        status::set_create_power_unknown();
        let mut runtime = Runtime::new(FakeHardware::new(1_000));
        assert!(runtime.enqueue_command(RuntimeCommand::WakeCreate).is_ok());

        assert!(runtime.start_next_command().is_ok());
        let first_deadline_ms = match runtime.active {
            ActiveAction::WaitForCreate {
                deadline_ms,
                allow_power_toggle_on_timeout: true,
                ..
            } => deadline_ms,
            _ => panic!("UNKNOWN wake must probe before its best-effort pulse"),
        };
        assert!(runtime.advance_active_action().is_ok());

        runtime.hardware.now_us = first_deadline_ms * 1_000;
        assert!(runtime.advance_active_action().is_ok());
        assert_eq!(
            runtime.hardware.power_toggle_levels.as_slice(),
            &[false, true]
        );

        runtime.hardware.delay_ms(body::POWER_TOGGLE_PULSE_MS);
        assert!(runtime.advance_active_action().is_ok());
        let settle_until_ms = match runtime.active {
            ActiveAction::WakeSettle { until_ms } => until_ms,
            _ => panic!("best-effort pulse must enter wake settle"),
        };
        runtime.hardware.now_us = settle_until_ms * 1_000;
        assert!(runtime.advance_active_action().is_ok());
        let second_deadline_ms = match runtime.active {
            ActiveAction::WaitForCreate {
                deadline_ms,
                allow_power_toggle_on_timeout: false,
                ..
            } => deadline_ms,
            _ => panic!("post-pulse probe must prohibit another toggle"),
        };
        assert!(runtime.advance_active_action().is_ok());

        runtime.hardware.now_us = second_deadline_ms * 1_000;
        assert!(runtime.advance_active_action().is_ok());

        assert_eq!(
            runtime.hardware.power_toggle_levels.as_slice(),
            &[false, true, false]
        );
        assert!(runtime.commands.is_empty());
    }

    #[test]
    fn interrupting_power_pulse_returns_power_toggle_low() {
        let _guard = status::status_test_guard();
        status::set_create_power_on(false);
        let mut runtime = Runtime::new(FakeHardware::new(1_000));

        assert!(runtime.enqueue_command(RuntimeCommand::WakeCreate).is_ok());
        assert!(runtime.start_next_command().is_ok());
        assert_eq!(
            runtime.hardware.power_toggle_levels.as_slice(),
            &[false, true]
        );

        runtime.shutdown();

        assert_eq!(
            runtime.hardware.power_toggle_levels.as_slice(),
            &[false, true, false]
        );
    }

    #[test]
    fn legacy_disarm_stops_without_sleeping_create() {
        assert_eq!(DISARM_SCRIPT.len(), 1);
        assert!(matches!(DISARM_SCRIPT[0], RuntimeCommand::Stop));
    }

    #[test]
    fn responsive_create_is_refreshed_in_full_mode() {
        let _guard = status::status_test_guard();
        let mut runtime = Runtime::new(FakeHardware::new(1_000));
        runtime.create_responsive = true;
        status::set_oi_mode(crate::commands::CreateOiMode::Full);

        let event_seq = status::event_next_seq();
        assert!(runtime.maintain_full_mode().is_ok());

        assert!(runtime.hardware.writes.contains(&132));
        assert_eq!(status::snapshot(1_000).oi_mode, 3);
        assert_eq!(status::event_next_seq(), event_seq);
    }

    #[test]
    fn stale_create_rx_invalidates_mode_and_stops_active_motor_output() {
        let _guard = status::status_test_guard();
        let mut runtime = Runtime::new(FakeHardware::new(2_001));
        runtime.create_responsive = true;
        runtime.last_create_packet_at_ms = Some(1_000);
        runtime.active = ActiveAction::Driving { stop_at_ms: 5_000 };
        runtime.active_command_id = Some(42);
        status::set_oi_mode(CreateOiMode::Full);

        runtime.poll();

        assert!(!runtime.create_responsive);
        assert_eq!(status::snapshot(2_001).oi_mode, 0);
        assert!(matches!(
            runtime.ensure_create_responsive(),
            Err(BrainstemError::CreateNoResponse)
        ));
        assert!(matches!(
            runtime.maintain_full_mode(),
            Err(BrainstemError::CreateNoResponse)
        ));
        assert!(matches!(runtime.active, ActiveAction::None));
        assert!(runtime
            .hardware
            .writes
            .windows(5)
            .any(|bytes| bytes == [137, 0, 0, 0, 0]));
    }

    #[test]
    fn fresh_create_rx_restores_responsiveness() {
        let _guard = status::status_test_guard();
        let mut runtime = Runtime::new(FakeHardware::new(1_000));
        status::set_oi_mode(CreateOiMode::Full);
        status::mark_uart_rx_byte(35, 1_000);
        status::mark_uart_packet(1);

        runtime.poll();

        assert!(runtime.create_responsive);
        assert_eq!(runtime.last_create_packet_at_ms, Some(1_000));
    }

    #[test]
    fn full_mode_refresh_never_overlays_active_motor_output() {
        let _guard = status::status_test_guard();
        let mut runtime = Runtime::new(FakeHardware::new(1_000));
        runtime.create_responsive = true;
        runtime.active = ActiveAction::DockDeparture { stop_at_ms: 3_500 };
        status::set_oi_mode(CreateOiMode::Full);

        assert!(runtime.maintain_full_mode().is_ok());

        assert!(runtime.hardware.writes.is_empty());
        assert_eq!(runtime.next_full_mode_refresh_ms, 2_000);
        assert!(matches!(
            runtime.active,
            ActiveAction::DockDeparture { stop_at_ms: 3_500 }
        ));
    }

    #[test]
    fn mode_loss_during_active_motor_output_stops_fail_closed() {
        let _guard = status::status_test_guard();
        let mut runtime = Runtime::new(FakeHardware::new(1_000));
        runtime.create_responsive = true;
        runtime.active = ActiveAction::Driving { stop_at_ms: 3_500 };
        runtime.active_command_id = Some(42);
        status::set_oi_mode(CreateOiMode::Passive);

        assert!(matches!(
            runtime.maintain_full_mode(),
            Err(BrainstemError::CreateNoResponse)
        ));

        assert!(matches!(runtime.active, ActiveAction::None));
        assert_eq!(runtime.active_command_id, None);
        assert!(runtime
            .hardware
            .writes
            .windows(5)
            .any(|bytes| bytes == [137, 0, 0, 0, 0]));
    }

    #[test]
    fn completed_motion_disarms_heartbeat_without_revoking_lease() {
        let _guard = status::status_test_guard();
        let mut runtime = Runtime::new(FakeHardware::new(301));
        runtime.heartbeat_stop_at_ms = Some(750);
        runtime.active = ActiveAction::Driving { stop_at_ms: 300 };

        assert!(runtime.advance_active_action().is_ok());

        assert!(matches!(runtime.active, ActiveAction::None));
        assert_eq!(runtime.heartbeat_stop_at_ms, None);
    }

    #[test]
    fn host_absence_expires_authority_and_stops_without_rebooting_body() {
        let _guard = status::status_test_guard();
        status::request_authority_transition(91, 71, 41, 999);
        status::acknowledge_authority_transition(91);
        status::set_oi_mode(CreateOiMode::Full);
        let mut runtime = Runtime::new(FakeHardware::new(1_000));
        runtime.create_responsive = true;
        runtime.next_full_mode_refresh_ms = 2_000;
        runtime.active = ActiveAction::Driving { stop_at_ms: 5_000 };
        runtime.active_command_id = Some(42);
        let _ = runtime.commands.push_back(QueuedCommand::new(
            43,
            RuntimeCommand::CmdVel {
                linear_mm_s: 100,
                angular_mrad_s: 0,
                duration_ms: Some(500),
            },
        ));

        runtime.tick();

        assert!(status::authority_expired(1_000));
        assert!(matches!(runtime.active, ActiveAction::None));
        assert_eq!(runtime.active_command_id, None);
        assert!(runtime.commands.is_empty());
        assert!(runtime
            .hardware
            .writes
            .windows(5)
            .any(|bytes| bytes == [137, 0, 0, 0, 0]));
        assert!(runtime.hardware.power_toggle_levels.is_empty());
        assert!(runtime.hardware.reset_levels.is_empty());
    }

    #[test]
    fn identical_velocity_refresh_renews_stream_without_lifecycle_churn() {
        let _guard = status::status_test_guard();
        let mut runtime = Runtime::new(FakeHardware::new(1_000));
        runtime.create_responsive = true;
        let command = RuntimeCommand::CmdVel {
            linear_mm_s: 50,
            angular_mrad_s: 100,
            duration_ms: Some(300),
        };

        runtime.enqueue_latest_velocity(41, command);
        assert!(runtime.start_next_command().is_ok());
        let drive_writes = runtime.hardware.writes.clone();
        let event_cursor = status::event_next_seq().saturating_sub(1);
        runtime.hardware.now_us = 1_100_000;

        runtime.enqueue_latest_velocity(42, command);

        assert!(runtime.commands.is_empty());
        assert_eq!(runtime.hardware.writes, drive_writes);
        assert_eq!(runtime.active_command_id, Some(41));
        assert!(matches!(
            runtime.active_velocity,
            Some(ActiveVelocity {
                linear_mm_s: 50,
                angular_mrad_s: 100,
            })
        ));
        assert!(matches!(
            runtime.active,
            ActiveAction::Driving { stop_at_ms: 1_400 }
        ));

        let mut lifecycle = heapless::Vec::<status::PublicEventRecord, 8>::new();
        status::collect_events_since(event_cursor, &mut lifecycle);
        let lifecycle = lifecycle
            .iter()
            .filter(|event| {
                matches!(
                    event.kind,
                    x if x == status::PublicEventKind::CommandStarted as u8
                        || x == status::PublicEventKind::CommandCompleted as u8
                        || x == status::PublicEventKind::CommandInterrupted as u8
                        || x == status::PublicEventKind::CommandTimedOut as u8
                )
            })
            .map(|event| (event.kind, event.a))
            .collect::<heapless::Vec<_, 4>>();
        assert!(lifecycle.is_empty());

        runtime.hardware.now_us = 1_400_000;
        assert!(runtime.advance_active_action().is_ok());
        assert_eq!(runtime.active_command_id, None);

        let mut completed = heapless::Vec::<status::PublicEventRecord, 8>::new();
        status::collect_events_since(event_cursor, &mut completed);
        assert!(completed.iter().any(|event| {
            event.kind == status::PublicEventKind::CommandCompleted as u8 && event.a == 41
        }));
    }

    #[test]
    fn unresponsive_create_still_gets_start_and_full() {
        let _guard = status::status_test_guard();
        status::set_oi_mode_unknown();
        let mut runtime = Runtime::new(FakeHardware::new(1_000));

        assert!(runtime.maintain_full_mode().is_ok());

        assert_eq!(runtime.hardware.writes.as_slice(), &[128, 132, 148, 1, 35]);
        assert_eq!(status::snapshot(1_000).oi_mode, 0);
    }

    #[test]
    fn private_status_polls_are_self_sustaining_and_preserve_public_stream() {
        let _guard = status::status_test_guard();
        let mut runtime = Runtime::new(FakeHardware::new(1_000));
        runtime.create_responsive = true;
        runtime.sensor_stream = Some(SensorStream {
            packet_id: 35,
            period_ms: 250,
            next_request_ms: 1_000,
        });

        assert!(runtime.poll_sensor_stream().is_ok());
        assert_eq!(runtime.hardware.writes.as_slice(), &[148, 1, 34]);

        runtime.hardware.writes.clear();
        runtime.hardware.now_us = 1_010_000;
        assert!(runtime.poll_sensor_stream().is_ok());
        assert_eq!(runtime.hardware.writes.as_slice(), &[148, 1, 0]);

        runtime.hardware.writes.clear();
        runtime.hardware.now_us = 1_020_000;
        assert!(runtime.poll_sensor_stream().is_ok());
        assert_eq!(runtime.hardware.writes.as_slice(), &[148, 1, 35]);
        assert_eq!(runtime.sensor_stream.unwrap().packet_id, 35);
    }

    #[test]
    fn low_battery_while_charging_pauses_full_mode_assertion() {
        let _guard = status::status_test_guard();
        status::mark_create_sensor_packet(
            0,
            crate::events::CreateSensorPacket {
                charging_state: 2,
                charge_mah: 200,
                capacity_mah: 2_000,
                ..crate::events::CreateSensorPacket::default()
            },
        );
        status::set_oi_mode(crate::commands::CreateOiMode::Full);
        let mut runtime = Runtime::new(FakeHardware::new(1_000));

        assert!(runtime.maintain_full_mode().is_ok());

        assert!(runtime.hardware.writes.is_empty());
        status::set_oi_mode_unknown();
        assert!(runtime.maintain_full_mode().is_ok());
        assert_eq!(runtime.hardware.writes.as_slice(), &[128, 132, 148, 1, 35]);
        status::mark_create_sensor_packet(
            0,
            crate::events::CreateSensorPacket {
                charging_state: 0,
                charge_mah: 2_000,
                capacity_mah: 2_000,
                ..crate::events::CreateSensorPacket::default()
            },
        );
    }

    #[test]
    fn charging_indicator_stops_motion_then_departure_precedes_next_command() {
        let _guard = status::status_test_guard();
        status::mark_create_sensor_packet(
            34,
            crate::events::CreateSensorPacket {
                charging_sources: 0b10,
                ..crate::events::CreateSensorPacket::default()
            },
        );
        let mut hardware = FakeHardware::new(1_000);
        hardware.charging_indicator = Some(true);
        let mut runtime = Runtime::new(hardware);
        runtime.create_responsive = true;
        runtime.active = ActiveAction::Driving { stop_at_ms: 5_000 };
        runtime.active_command_id = Some(77);
        let _ = runtime.commands.push_back(QueuedCommand::new(
            78,
            RuntimeCommand::CmdVel {
                linear_mm_s: 100,
                angular_mrad_s: 0,
                duration_ms: Some(500),
            },
        ));

        runtime.tick();

        assert!(matches!(runtime.active, ActiveAction::None));
        assert!(runtime.dock_departure_pending);
        assert!(!runtime.charging_interlock_latched);
        assert!(!runtime
            .commands
            .iter()
            .any(|queued| is_motion_command(queued.command)));
        assert!(runtime
            .events
            .iter()
            .any(|event| matches!(event, BrainstemEvent::DriveStopped)));

        runtime.hardware.charging_indicator = Some(false);
        status::mark_create_sensor_packet(
            34,
            crate::events::CreateSensorPacket {
                charging_sources: 0b10,
                ..crate::events::CreateSensorPacket::default()
            },
        );
        runtime.tick();
        assert!(runtime.dock_departure_pending);
        assert!(!status::session_safety_snapshot().2);

        let event_cursor = status::snapshot(1_000).event_next_seq;
        assert!(runtime
            .commands
            .push_back(QueuedCommand::new(
                79,
                RuntimeCommand::CmdVel {
                    linear_mm_s: 100,
                    angular_mrad_s: 0,
                    duration_ms: Some(500),
                },
            ))
            .is_ok());
        assert!(runtime.start_next_command().is_ok());
        assert!(matches!(runtime.active, ActiveAction::DockDeparture { .. }));
        assert!(!runtime.dock_departure_pending);
        assert_eq!(
            &runtime.hardware.writes.as_slice()[runtime.hardware.writes.len() - 5..],
            &[145, 0xff, 0x38, 0xff, 0x38]
        );
        assert_eq!(runtime.commands.len(), 1);

        let mut lifecycle = heapless::Vec::<status::PublicEventRecord, 8>::new();
        status::collect_events_since(event_cursor, &mut lifecycle);
        assert!(!lifecycle.iter().any(|event| {
            event.kind == status::PublicEventKind::CommandStarted as u8 && event.a == 79
        }));

        runtime.hardware.delay_ms(DOCK_DEPARTURE_DURATION_MS);
        assert!(runtime.advance_active_action().is_ok());
        assert!(matches!(runtime.active, ActiveAction::None));
        assert_eq!(runtime.commands.len(), 1);

        assert!(runtime.start_next_command().is_ok());
        assert!(matches!(runtime.active, ActiveAction::Driving { .. }));
        assert_eq!(runtime.active_command_id, Some(79));
        assert!(runtime.events.iter().any(|event| {
            matches!(
                event,
                BrainstemEvent::DriveRequested {
                    left_mm_s: 100,
                    right_mm_s: 100,
                    ..
                }
            )
        }));
    }

    #[test]
    fn home_base_reconciles_cliff_race_through_departure_but_not_wheel_drop() {
        let _guard = status::status_test_guard();
        status::mark_create_sensor_packet(
            0,
            crate::events::CreateSensorPacket {
                flags: crate::events::CreateSensorFlags {
                    bump_left: true,
                    bump_right: true,
                    cliff_left: true,
                    cliff_front_left: true,
                    cliff_front_right: true,
                    cliff_right: true,
                    ..crate::events::CreateSensorFlags::default()
                },
                ..crate::events::CreateSensorPacket::default()
            },
        );
        let mut runtime = Runtime::new(FakeHardware::new(1_000));
        runtime.create_responsive = true;

        assert!(runtime.enforce_safety_policy().is_ok());
        assert!(matches!(
            runtime.safety_latch_kind,
            Some(status::SafetyEventKind::Cliff)
        ));

        let event_cursor = status::snapshot(1_000).event_next_seq.saturating_sub(1);
        status::mark_create_sensor_packet(
            34,
            crate::events::CreateSensorPacket {
                charging_sources: 0b10,
                ..crate::events::CreateSensorPacket::default()
            },
        );
        assert!(runtime.enforce_safety_policy().is_ok());
        assert!(!runtime.safety_latched);
        assert!(runtime.safety_latch_kind.is_none());

        let mut events = heapless::Vec::<status::PublicEventRecord, 8>::new();
        status::collect_events_since(event_cursor, &mut events);
        assert!(events.iter().any(|event| {
            event.kind == status::PublicEventKind::SafetyCleared as u8
                && event.a == status::SafetyEventKind::Cliff as u32
        }));

        runtime.dock_departure_pending = false;
        runtime.heartbeat_stop_at_ms = Some(1_100);
        runtime.active = ActiveAction::DockDeparture { stop_at_ms: 2_000 };
        let departure_cursor = status::snapshot(1_000).event_next_seq.saturating_sub(1);
        assert!(runtime.enforce_safety_policy().is_ok());
        assert!(matches!(runtime.active, ActiveAction::DockDeparture { .. }));
        assert!(!runtime.safety_latched);

        runtime.dock_departure_pending = true;
        assert!(runtime.start_dock_departure(1_000).is_ok());
        assert!(runtime.heartbeat_stop_at_ms.is_none());
        assert!(matches!(
            runtime.active,
            ActiveAction::DockDeparture { stop_at_ms: 2_500 }
        ));

        let mut departure_events = heapless::Vec::<status::PublicEventRecord, 8>::new();
        status::collect_events_since(departure_cursor, &mut departure_events);
        assert!(!departure_events.iter().any(|event| {
            event.kind == status::PublicEventKind::SafetyTripped as u8
                && event.a == status::SafetyEventKind::Cliff as u32
        }));

        status::mark_create_sensor_packet(
            0,
            crate::events::CreateSensorPacket {
                flags: crate::events::CreateSensorFlags {
                    bump_left: true,
                    bump_right: true,
                    wheel_drop: true,
                    cliff_left: true,
                    cliff_front_left: true,
                    cliff_front_right: true,
                    cliff_right: true,
                    ..crate::events::CreateSensorFlags::default()
                },
                ..crate::events::CreateSensorPacket::default()
            },
        );
        assert!(runtime.enforce_safety_policy().is_ok());
        assert!(runtime.safety_latched);
        assert!(matches!(
            runtime.safety_latch_kind,
            Some(status::SafetyEventKind::WheelDrop)
        ));
        assert!(matches!(runtime.active, ActiveAction::None));
    }

    #[test]
    fn fresh_home_base_clear_cancels_unstarted_departure() {
        let _guard = status::status_test_guard();
        let mut runtime = Runtime::new(FakeHardware::new(1_000));
        runtime.create_responsive = true;
        status::set_oi_mode(CreateOiMode::Full);
        status::mark_create_sensor_packet(
            34,
            crate::events::CreateSensorPacket {
                charging_sources: 0b10,
                ..crate::events::CreateSensorPacket::default()
            },
        );

        assert!(runtime.enforce_safety_policy().is_ok());
        assert!(runtime.dock_departure_pending);

        status::mark_create_sensor_packet(34, crate::events::CreateSensorPacket::default());
        assert!(runtime.enforce_safety_policy().is_ok());
        assert!(!runtime.dock_departure_pending);

        assert!(runtime
            .commands
            .push_back(QueuedCommand::new(
                91,
                RuntimeCommand::CmdVel {
                    linear_mm_s: 100,
                    angular_mrad_s: 0,
                    duration_ms: Some(400),
                },
            ))
            .is_ok());
        assert!(runtime.start_next_command().is_ok());
        assert!(matches!(runtime.active, ActiveAction::Driving { .. }));
        assert_eq!(runtime.active_command_id, Some(91));
    }

    #[test]
    fn dock_departure_only_wraps_nonzero_nondocking_motion() {
        assert!(!requires_dock_departure(RuntimeCommand::CmdVel {
            linear_mm_s: 0,
            angular_mrad_s: 0,
            duration_ms: Some(300),
        }));
        assert!(requires_dock_departure(RuntimeCommand::CmdVel {
            linear_mm_s: 50,
            angular_mrad_s: 0,
            duration_ms: Some(300),
        }));
        assert!(!requires_dock_departure(RuntimeCommand::Dock));
    }

    #[test]
    fn zero_velocity_stops_without_consuming_pending_dock_departure() {
        let _guard = status::status_test_guard();
        status::mark_create_sensor_packet(0, crate::events::CreateSensorPacket::default());
        let mut runtime = Runtime::new(FakeHardware::new(1_000));
        runtime.create_responsive = true;
        runtime.dock_departure_pending = true;
        assert!(runtime
            .commands
            .push_back(QueuedCommand::new(
                80,
                RuntimeCommand::CmdVel {
                    linear_mm_s: 0,
                    angular_mrad_s: 0,
                    duration_ms: Some(300),
                },
            ))
            .is_ok());

        assert!(runtime.start_next_command().is_ok());

        assert!(matches!(runtime.active, ActiveAction::None));
        assert!(runtime.dock_departure_pending);
        assert!(runtime.active_velocity.is_none());
        assert!(runtime
            .hardware
            .writes
            .windows(5)
            .any(|bytes| bytes == [137, 0, 0, 0, 0]));
    }

    #[test]
    fn oi_charging_state_stops_active_motion_without_charge_pin() {
        let _guard = status::status_test_guard();
        status::mark_create_sensor_packet(
            0,
            crate::events::CreateSensorPacket {
                charging_state: 2,
                ..crate::events::CreateSensorPacket::default()
            },
        );
        let mut runtime = Runtime::new(FakeHardware::new(1_000));
        runtime.create_responsive = true;
        runtime.active = ActiveAction::Driving { stop_at_ms: 5_000 };

        runtime.tick();

        assert!(matches!(runtime.active, ActiveAction::None));
        assert!(runtime.charging_interlock_latched);

        status::mark_create_sensor_packet(0, crate::events::CreateSensorPacket::default());
    }

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
                        == MotherbrainResetOutcome::Refused(
                            status::MotherbrainResetRefusal::Cooldown,
                        )
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
