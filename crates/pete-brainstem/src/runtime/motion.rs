impl<H> Runtime<H>
where
    H: BrainstemHardware,
{
    fn start_drive_direct(
        &mut self,
        left_mm_s: i16,
        right_mm_s: i16,
        duration_ms: Option<u32>,
        now_ms: u32,
    ) -> Result<(), BrainstemError> {
        let Some(duration_ms) = duration_ms else {
            self.stop_drive()?;
            return Err(BrainstemError::Timeout);
        };
        if left_mm_s == 0 && right_mm_s == 0 {
            self.stop_drive()?;
            self.active = ActiveAction::None;
            return Ok(());
        }
        self.ensure_motion_allowed()?;
        self.active_velocity = None;

        status::set_body_state(BodyState::Moving);
        self.stop_sent = false;
        self.create_uart.drive_direct(
            &mut self.hardware,
            &mut self.events,
            left_mm_s,
            right_mm_s,
            duration_ms,
        )?;
        self.active = ActiveAction::Driving {
            stop_at_ms: now_ms.wrapping_add(duration_ms),
        };
        Ok(())
    }

    fn start_drive_arc(
        &mut self,
        velocity_mm_s: i16,
        radius_mm: i16,
        duration_ms: Option<u32>,
        now_ms: u32,
    ) -> Result<(), BrainstemError> {
        let Some(duration_ms) = duration_ms else {
            self.stop_drive()?;
            return Err(BrainstemError::Timeout);
        };
        if velocity_mm_s == 0 {
            self.stop_drive()?;
            self.active = ActiveAction::None;
            return Ok(());
        }
        self.ensure_motion_allowed()?;

        status::set_body_state(BodyState::Moving);
        self.stop_sent = false;
        self.create_uart.drive_arc(
            &mut self.hardware,
            &mut self.events,
            velocity_mm_s,
            radius_mm,
        )?;
        self.active = ActiveAction::Driving {
            stop_at_ms: now_ms.wrapping_add(duration_ms),
        };
        Ok(())
    }

    fn ensure_create_responsive(&mut self) -> Result<(), BrainstemError> {
        if self.last_create_packet_at_ms.is_some_and(|last_packet_ms| {
            self.now_ms().wrapping_sub(last_packet_ms) > CREATE_LINK_FRESHNESS_TIMEOUT_MS
        }) {
            self.create_responsive = false;
            status::set_oi_mode_unknown();
        }
        if !self.create_responsive {
            return Err(BrainstemError::CreateNoResponse);
        }
        Ok(())
    }

    fn ensure_motion_allowed(&mut self) -> Result<(), BrainstemError> {
        if self.estop_latched
            || self.dock_departure_pending
            || self.charging_interlock_latched
            || (self.safety_latched && !self.safety_recovery_latch_allows_motion())
        {
            self.stop_drive()?;
            return Err(BrainstemError::CreateNoResponse);
        }
        self.ensure_create_responsive()?;
        Ok(())
    }

    fn safety_recovery_latch_allows_motion(&self) -> bool {
        self.safety_recovery_motion && recoverable_safety_latch(self.safety_latch_kind)
    }

    fn enforce_heartbeat_stop(&mut self) -> Result<(), BrainstemError> {
        let Some(deadline_ms) = self.heartbeat_stop_at_ms else {
            return Ok(());
        };
        if time_reached(self.now_ms(), deadline_ms) {
            self.heartbeat_stop_at_ms = None;
            self.cancel_careful_mode();
            status::revoke_authority();
            status::mark_heartbeat_expired();
            status::mark_safety_tripped(status::SafetyEventKind::Heartbeat);
            self.request_audio(AuditoryCue::HeartbeatLost);
            if self.active_contact_withdrawal.is_none() {
                self.interrupt_active_command();
                self.commands.clear();
                self.active = ActiveAction::None;
                self.stop_drive()?;
            }
        }
        Ok(())
    }

    fn enter_idle(&mut self) {
        let _ = self.stop_drive();
        self.complete_active_command();
        self.mode = RuntimeMode::Idle;
        self.active = ActiveAction::None;
        status::set_runtime_state(RuntimeState::Idle);
        status::set_body_state(BodyState::Idle);
        status::set_command(None);
        self.hardware.set_indicators(false);
        self.idle_blink_next_ms = self.now_ms();
    }

    fn idle_tick(&mut self) {
        let now_ms = self.now_ms();
        if time_reached(now_ms, self.idle_blink_next_ms) {
            self.idle_blink_on = !self.idle_blink_on;
            self.hardware.set_indicators(self.idle_blink_on);
            self.idle_blink_next_ms = now_ms.wrapping_add(body::IDLE_BLINK_MS);
        }
    }

    fn enter_error(&mut self, error: BrainstemError) {
        status::set_error(error);
        self.push_event(BrainstemEvent::Error(error));
        self.fail_active_command(error);
        let stopped = self.stop_drive().is_ok();
        self.finish_contact_withdrawal(status::ContactWithdrawalOutcome::Failed, None, stopped);
        self.mode = RuntimeMode::Error;
        self.active = ActiveAction::None;
        self.error_blink_next_ms = self.now_ms();
        self.error_blink_count = 0;
        self.error_blink_on = false;
        if self.create_responsive {
            self.request_audio(AuditoryCue::RuntimeError);
        }
    }

    fn error_tick(&mut self) {
        let now_ms = self.now_ms();
        if !time_reached(now_ms, self.error_blink_next_ms) {
            return;
        }

        if self.error_blink_count >= 6 {
            self.hardware.set_indicators(false);
            self.error_blink_on = false;
            self.error_blink_count = 0;
            self.error_blink_next_ms = now_ms.wrapping_add(body::ERROR_PAUSE_MS);
            return;
        }

        self.error_blink_on = !self.error_blink_on;
        self.hardware.set_indicators(self.error_blink_on);
        self.error_blink_count = self.error_blink_count.saturating_add(1);
        self.error_blink_next_ms = now_ms.wrapping_add(body::ERROR_BLINK_MS);
    }

}
