impl<H> Runtime<H>
where
    H: BrainstemHardware,
{
    pub fn new(hardware: H) -> Self {
        let mut events = Deque::new();
        let observed_uart_rx_packets = status::snapshot(0).uart_rx_packets;
        status::signal_event(&BrainstemEvent::Boot);
        let _ = events.push_back(BrainstemEvent::Boot);
        status::set_runtime_state(RuntimeState::Booting);
        status::set_body_state(BodyState::NotStarted);
        status::set_careful_mode_until(None);
        status::set_audio_silent(false);
        Self {
            hardware,
            events,
            commands: Deque::new(),
            timers: Timers::new(),
            create_uart: CreateUart::new(),
            leds: Leds::new(),
            mode: RuntimeMode::Running,
            active: ActiveAction::None,
            active_command_id: None,
            active_velocity: None,
            active_escape: None,
            stop_sent: false,
            heartbeat_stop_at_ms: None,
            careful_mode_until_ms: None,
            sensor_stream: None,
            next_charging_sources_poll_ms: 0,
            next_complete_sensor_poll_ms: 0,
            next_imu_poll_ms: 0,
            next_full_mode_refresh_ms: 0,
            next_supervision_light_ms: 0,
            supervision_light_phase: 0,
            safety_latched: false,
            safety_latch_kind: None,
            dock_departure_pending: false,
            charging_interlock_latched: false,
            chirps: [[SongTone::default(); MAX_SONG_TONES]; FEEDBACK_KIND_COUNT],
            chirp_counts: [0; FEEDBACK_KIND_COUNT],
            audio: AudioAnnunciator::new(),
            song_durations_ms: [0; 16],
            error_blink_next_ms: 0,
            error_blink_on: false,
            error_blink_count: 0,
            idle_blink_next_ms: 0,
            idle_blink_on: false,
            create_responsive: false,
            estop_latched: false,
            last_bump: false,
            last_cliff: false,
            last_wheel_drop: false,
            safety_observation_initialized: false,
            tilt_observed_since_ms: None,
            active_motherbrain_reset: None,
            motherbrain_reset_cooldown_until_ms: 0,
            motherbrain_reset_hardware_enabled: body::MOTHERBRAIN_RESET_ENABLED,
            motherbrain_reset_history: [None; MOTHERBRAIN_RESET_HISTORY_CAPACITY],
            motherbrain_reset_history_next: 0,
            safety_recovery_motion: false,
            active_contact_withdrawal: None,
            last_contact_withdrawal_at_ms: None,
            repeated_contact_count: 0,
            last_observed_uart_rx_packets: observed_uart_rx_packets,
            last_create_packet_at_ms: None,
            low_battery_active: false,
            charging_active: false,
            imu_recovery_since_ms: None,
            motion_inconsistency_cooldown_until_ms: 0,
            docking_active: false,
            last_dock_ir: 0,
            restart_create_pending: false,
            create_full_ready: false,
            ever_create_full_ready: false,
            imu_fault_active: false,
            last_motion_inconsistent: false,
        }
    }

    pub fn run(mut self) -> ! {
        self.start();
        loop {
            self.tick();
            self.hardware.delay_ms(RUNTIME_TICK_MS);
        }
    }

    pub(crate) fn start(&mut self) {
        self.leds.boot_indicator(&mut self.hardware);
        self.queue_create_acquisition(0);
        status::set_runtime_state(RuntimeState::Running);
    }

    pub(crate) fn shutdown(&mut self) {
        self.interrupt_active_command();
        self.commands.clear();
        self.active = ActiveAction::None;
        self.heartbeat_stop_at_ms = None;
        self.cancel_careful_mode();
        let _ = self.stop_drive();
        status::set_command(None);
        status::set_runtime_state(RuntimeState::Idle);
        status::set_body_state(BodyState::Idle);
    }

    #[allow(dead_code)]
    pub fn enqueue_command(&mut self, command: RuntimeCommand) -> Result<(), RuntimeCommand> {
        self.commands
            .push_back(QueuedCommand::new(0, command))
            .map_err(|queued| queued.command)
    }

    pub fn tick(&mut self) {
        status::set_runtime_action(self.active_action_code());
        self.poll();
        self.poll_motherbrain_reset();
        self.hardware.feed_watchdog();
        self.poll_charging_indicator();
        self.poll_imu();
        self.observe_audio_transitions();
        if let Err(error) = self.poll_sensor_stream() {
            self.enter_error(error);
            return;
        }
        if let Err(error) = self.enforce_careful_mode_timeout() {
            self.enter_error(error);
            return;
        }
        if let Err(error) = self.enforce_safety_policy() {
            self.enter_error(error);
            return;
        }
        if let Err(error) = self.enforce_heartbeat_stop() {
            self.enter_error(error);
            return;
        }
        self.publish_safety_snapshot();
        if let Err(error) = self.maintain_full_mode() {
            self.enter_error(error);
            return;
        }
        if let Err(error) = self.animate_supervision_lights() {
            self.enter_error(error);
            return;
        }
        if status::take_expired_authority(self.now_ms()) {
            self.request_audio(AuditoryCue::AuthorityLost);
            self.heartbeat_stop_at_ms = None;
            self.cancel_careful_mode();
            if self.active_contact_withdrawal.is_none() {
                self.interrupt_active_command();
                self.commands.clear();
                self.active = ActiveAction::None;
                let _ = self.stop_drive();
            }
        }

        match self.mode {
            RuntimeMode::Running => {
                if let Err(error) = self.advance_active_action() {
                    self.enter_error(error);
                    return;
                }

                if self.active == ActiveAction::None {
                    if let Err(error) = self.start_next_command() {
                        self.enter_error(error);
                    } else if self.commands.is_empty() && self.active == ActiveAction::None {
                        self.enter_idle();
                    }
                }
            }
            RuntimeMode::Idle => self.idle_tick(),
            RuntimeMode::Error => self.error_tick(),
        }
        self.poll_audio();
    }

    fn poll(&mut self) {
        self.poll_session_replace();
        self.poll_authority_transition();
        self.timers.poll(&mut self.hardware, &mut self.events);
        self.create_uart.poll(&mut self.hardware, &mut self.events);
        let now_ms = self.now_ms();
        let snapshot = status::snapshot(now_ms);
        let was_create_responsive = self.create_responsive;
        if snapshot.uart_rx_packets != self.last_observed_uart_rx_packets {
            self.last_observed_uart_rx_packets = snapshot.uart_rx_packets;
            self.last_create_packet_at_ms = Some(now_ms);
        }
        let create_link_fresh = self.last_create_packet_at_ms.is_some_and(|last_packet_ms| {
            now_ms.wrapping_sub(last_packet_ms) <= CREATE_LINK_FRESHNESS_TIMEOUT_MS
        });
        if matches!(snapshot.oi_mode, 1..=3) && create_link_fresh {
            self.create_responsive = true;
            status::set_create_power_on(true);
        } else if self.last_create_packet_at_ms.is_some() && !create_link_fresh {
            // OI mode and power state are observations, not durable promises.
            // Once the Create stops answering, invalidate the cached mode so
            // active output is stopped by Full-mode supervision and later
            // motion cannot start against a dead link.
            self.create_responsive = false;
            self.cancel_careful_mode();
            status::set_oi_mode_unknown();
        }
        if was_create_responsive && !self.create_responsive {
            self.create_full_ready = false;
            self.request_audio(AuditoryCue::CreateError);
        }
        self.poll_control_command();
    }

    fn poll_authority_transition(&mut self) {
        let Some(generation) = status::pending_authority_transition() else {
            return;
        };
        let now_ms = self.now_ms();
        if status::pending_authority_continues_owner(now_ms) {
            status::acknowledge_authority_transition(generation);
            return;
        }
        let replacing = status::has_active_authority(now_ms);
        self.heartbeat_stop_at_ms = None;
        self.cancel_careful_mode();
        if self.active_contact_withdrawal.is_none() {
            self.interrupt_active_command();
            self.commands.clear();
            self.active = ActiveAction::None;
            let _ = self.stop_drive();
            status::set_command(None);
            status::set_runtime_state(RuntimeState::Idle);
            status::set_body_state(BodyState::Idle);
        }
        status::acknowledge_authority_transition(generation);
        self.request_audio(if replacing {
            AuditoryCue::AuthorityReplaced
        } else {
            AuditoryCue::AuthorityAcquired
        });
    }

    fn queue_create_acquisition(&mut self, command_id: u32) {
        for command in ACQUIRE_CREATE_SCRIPT {
            let _ = self
                .commands
                .push_back(QueuedCommand::new(command_id, *command));
        }
        self.mode = RuntimeMode::Running;
    }

    fn maintain_full_mode(&mut self) -> Result<(), BrainstemError> {
        let now_ms = self.now_ms();
        let snapshot = status::snapshot(now_ms);
        if !time_reached(now_ms, self.next_full_mode_refresh_ms) {
            return Ok(());
        }
        let motor_output_active = matches!(
            self.active,
            ActiveAction::Driving { .. } | ActiveAction::DockDeparture { .. }
        );
        if motor_output_active {
            if snapshot.oi_mode != 3 {
                self.stop_drive()?;
                self.interrupt_active_command();
                self.active = ActiveAction::None;
                return Err(BrainstemError::CreateNoResponse);
            }
            // Re-sending OI Full zeros wheel output on Create 1 even when the
            // reported mode is already Full. Never overlay that supervision
            // write on a bounded motor program; fresh mode loss still takes
            // the fail-closed branch above.
            self.next_full_mode_refresh_ms = now_ms.wrapping_add(FULL_MODE_REFRESH_PERIOD_MS);
            return Ok(());
        }
        if low_battery_and_charging(&snapshot) && snapshot.oi_mode == 3 {
            return Ok(());
        }

        // RX health is evidence, not permission to transmit. If the Create has
        // rebooted, gone passive, or our wake probe was wrong, START + FULL is
        // the idempotent assertion that lets the brainstem regain control.
        if !self.create_responsive || snapshot.oi_mode != 3 {
            self.create_uart
                .start_oi(&mut self.hardware, &mut self.events)?;
        }
        if snapshot.oi_mode == 3 {
            self.create_uart.refresh_full_mode(&mut self.hardware)?;
        } else {
            self.create_uart.set_mode(
                &mut self.hardware,
                &mut self.events,
                crate::commands::CreateOiMode::Full,
            )?;
        }
        if snapshot.oi_mode == 0 {
            self.create_uart.start_mode_stream(&mut self.hardware)?;
        }
        self.next_full_mode_refresh_ms = now_ms.wrapping_add(FULL_MODE_REFRESH_PERIOD_MS);
        Ok(())
    }

    fn animate_supervision_lights(&mut self) -> Result<(), BrainstemError> {
        let now_ms = self.now_ms();
        if !self.create_responsive
            || status::snapshot(now_ms).oi_mode != 3
            || !time_reached(now_ms, self.next_supervision_light_ms)
        {
            return Ok(());
        }

        let (led_bits, color, intensity, period_ms) = if self.mode == RuntimeMode::Error {
            let on = self.supervision_light_phase & 1 == 0;
            (
                if on { CREATE_BUTTON_LED_MASK } else { 0 },
                255,
                if on { 255 } else { 0 },
                300,
            )
        } else if self.estop_latched || self.safety_latched || self.charging_interlock_latched {
            (CREATE_BUTTON_LED_MASK, 255, 255, 500)
        } else {
            // Keep POWER stable while PLAY and ADVANCE alternate more quickly.
            healthy_supervision_lights(self.supervision_light_phase)
        };
        self.create_uart
            .set_supervision_lights(&mut self.hardware, led_bits, color, intensity)?;
        self.supervision_light_phase = self.supervision_light_phase.wrapping_add(1);
        self.next_supervision_light_ms = now_ms.wrapping_add(period_ms);
        Ok(())
    }

    fn poll_session_replace(&mut self) {
        let Some(generation) = status::pending_session_replace() else {
            return;
        };
        self.heartbeat_stop_at_ms = None;
        self.cancel_careful_mode();
        self.sensor_stream = None;
        network_registry::clear_motherbrain_registration();
        if self.active_contact_withdrawal.is_none() {
            self.interrupt_active_command();
            self.commands.clear();
            self.active = ActiveAction::None;
            let _ = self.stop_drive();
            status::set_command(None);
            status::set_runtime_state(RuntimeState::Idle);
            status::set_body_state(BodyState::Idle);
        }
        self.publish_safety_snapshot();
        // The session module supplies the pending hash before requesting the
        // barrier. Until it is wired, generation itself is a fail-closed token.
        status::acknowledge_session_replace(generation, status::pending_session_hash());
    }

    fn publish_safety_snapshot(&self) {
        status::set_session_safety_snapshot(
            self.estop_latched,
            self.safety_latched,
            self.charging_interlock_latched,
            self.safety_latch_kind,
        );
    }

}
