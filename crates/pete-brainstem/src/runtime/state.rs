impl<H> Runtime<H>
where
    H: BrainstemHardware,
{
    fn now_ms(&mut self) -> u32 {
        self.hardware.now_us() / 1_000
    }

    fn active_action_code(&self) -> RuntimeActionCode {
        match self.active {
            ActiveAction::None => RuntimeActionCode::None,
            ActiveAction::PowerPulse { .. } => RuntimeActionCode::PowerPulse,
            ActiveAction::WakeSettle { .. } => RuntimeActionCode::WakeSettle,
            ActiveAction::WaitForCreate { .. } => RuntimeActionCode::WaitForCreate,
            ActiveAction::Settle { .. } => RuntimeActionCode::Settle,
            ActiveAction::Driving { .. } | ActiveAction::DockDeparture { .. } => {
                RuntimeActionCode::Driving
            }
        }
    }

    fn push_event(&mut self, event: BrainstemEvent) {
        status::signal_event(&event);
        let _ = self.events.push_back(event);
    }

    fn complete_active_command(&mut self) {
        self.safety_recovery_motion = false;
        status::clear_velocity_stream();
        if let Some(command_id) = self.active_command_id.take() {
            status::mark_command_completed(command_id);
        }
    }

    fn refuse_active_command(&mut self) {
        self.safety_recovery_motion = false;
        status::clear_velocity_stream();
        if let Some(command_id) = self.active_command_id.take() {
            status::mark_command_interrupted(command_id);
        }
    }

    fn interrupt_active_command(&mut self) {
        if matches!(self.active, ActiveAction::PowerPulse { .. }) {
            self.hardware.end_power_toggle_pulse();
        }
        self.safety_recovery_motion = false;
        self.active_velocity = None;
        self.active_escape = None;
        status::clear_velocity_stream();
        if let Some(command_id) = self.active_command_id.take() {
            status::mark_command_interrupted(command_id);
        }
    }

    fn fail_active_command(&mut self, error: BrainstemError) {
        self.safety_recovery_motion = false;
        status::clear_velocity_stream();
        let Some(command_id) = self.active_command_id.take() else {
            return;
        };
        match error {
            BrainstemError::CreateNoResponse | BrainstemError::Timeout => {
                status::mark_command_timed_out(command_id);
            }
            BrainstemError::UartFraming | BrainstemError::InvalidPacket => {
                status::mark_command_interrupted(command_id);
            }
        }
    }

    fn start_contact_withdrawal(
        &mut self,
        contact_bits: u8,
        preempted_command_id: u32,
        baseline_odometry_mm: i32,
    ) {
        let now_ms = self.now_ms();
        self.repeated_contact_count = match self.last_contact_withdrawal_at_ms {
            Some(previous) if now_ms.wrapping_sub(previous) <= CONTACT_REPEAT_WINDOW_MS => {
                self.repeated_contact_count.saturating_add(1).max(1)
            }
            _ => 1,
        };
        self.last_contact_withdrawal_at_ms = Some(now_ms);
        self.active_contact_withdrawal = Some(ActiveContactWithdrawal {
            started_at_ms: now_ms,
            baseline_odometry_mm,
        });
        status::mark_contact_withdrawal_started(
            contact_bits,
            self.repeated_contact_count,
            preempted_command_id,
            CONTACT_WITHDRAWAL_SPEED_MM_S.unsigned_abs(),
            CONTACT_WITHDRAWAL_DURATION_MS.min(u32::from(u16::MAX)) as u16,
        );
    }

    fn finish_contact_withdrawal(
        &mut self,
        outcome: status::ContactWithdrawalOutcome,
        dominating_safety: Option<status::SafetyEventKind>,
        final_stopped: bool,
    ) {
        let Some(active) = self.active_contact_withdrawal.take() else {
            return;
        };
        let now_ms = self.now_ms();
        let displacement = status::snapshot(now_ms)
            .odometry_distance_mm
            .wrapping_sub(active.baseline_odometry_mm);
        status::mark_contact_withdrawal_completed(
            outcome,
            dominating_safety,
            final_stopped,
            displacement,
            now_ms.wrapping_sub(active.started_at_ms),
        );
        if matches!(
            outcome,
            status::ContactWithdrawalOutcome::SafetyPreempted
                | status::ContactWithdrawalOutcome::Failed
        ) {
            self.request_audio(AuditoryCue::ServiceFailure);
        }
    }

    fn update_safety_edges(&mut self, bump: bool, cliff: bool, wheel_drop: bool) {
        if bump != self.last_bump {
            status::mark_bump_changed(bump);
            self.last_bump = bump;
        }
        if cliff != self.last_cliff {
            status::mark_cliff_changed(cliff);
            self.last_cliff = cliff;
        }
        if wheel_drop != self.last_wheel_drop {
            if !wheel_drop {
                status::mark_wheel_drop_cleared();
            }
            self.last_wheel_drop = wheel_drop;
        }
    }
}
