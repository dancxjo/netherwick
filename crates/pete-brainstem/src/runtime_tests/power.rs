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
