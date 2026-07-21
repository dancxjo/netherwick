impl<H> Runtime<H>
where
    H: BrainstemHardware,
{
    fn poll_control_command(&mut self) {
        let Some(command) = status::take_control_command() else {
            return;
        };
        let command_id = status::last_dispatched_command_id();
        let (service_session_hash, service_lease_hash) = status::last_dispatched_service_identity();

        if self.active_contact_withdrawal.is_some() && !command_preempts_contact_withdrawal(command)
        {
            // The possessor may lose or replace authority while this runs.
            // Ordinary commands cannot supersede a local reflex.
            status::mark_command_interrupted(command_id);
            return;
        }

        if self.active_contact_withdrawal.is_some() && command_preempts_contact_withdrawal(command)
        {
            let stopped = self.stop_drive().is_ok();
            self.finish_contact_withdrawal(
                status::ContactWithdrawalOutcome::SafetyPreempted,
                matches!(command, BrainstemCommand::EStop)
                    .then_some(status::SafetyEventKind::EStop),
                stopped,
            );
        }

        match command {
            BrainstemCommand::Stop | BrainstemCommand::EStop => {
                self.docking_active = false;
                self.interrupt_active_command();
                self.commands.clear();
                self.active = ActiveAction::None;
                self.heartbeat_stop_at_ms = None;
                if matches!(command, BrainstemCommand::EStop) {
                    self.cancel_careful_mode();
                }
                let command = match command {
                    BrainstemCommand::Stop => RuntimeCommand::Stop,
                    BrainstemCommand::EStop => RuntimeCommand::EStop,
                    _ => unreachable!(),
                };
                let _ = self
                    .commands
                    .push_front(QueuedCommand::new(command_id, command));
                self.mode = RuntimeMode::Running;
            }
            BrainstemCommand::Arm => {
                self.queue_create_acquisition(command_id);
                if self.mode == RuntimeMode::Idle || self.mode == RuntimeMode::Error {
                    self.mode = RuntimeMode::Running;
                    status::set_runtime_state(RuntimeState::Running);
                }
            }
            BrainstemCommand::Disarm => {
                self.interrupt_active_command();
                self.commands.clear();
                self.active = ActiveAction::None;
                self.heartbeat_stop_at_ms = None;
                self.cancel_careful_mode();
                for command in DISARM_SCRIPT.iter().rev() {
                    let _ = self
                        .commands
                        .push_front(QueuedCommand::new(command_id, *command));
                }
                self.mode = RuntimeMode::Running;
            }
            BrainstemCommand::RestartCreate => {
                self.restart_create_pending = true;
                self.interrupt_active_command();
                self.commands.clear();
                self.active = ActiveAction::None;
                self.heartbeat_stop_at_ms = None;
                self.cancel_careful_mode();
                for command in RESTART_CREATE_SCRIPT.iter().rev() {
                    let _ = self
                        .commands
                        .push_front(QueuedCommand::new(command_id, *command));
                }
                self.mode = RuntimeMode::Running;
                status::set_runtime_state(RuntimeState::Running);
            }
            BrainstemCommand::ResetMotherbrain => {
                self.request_motherbrain_reset(MotherbrainResetIdentity {
                    session_hash: service_session_hash,
                    lease_hash: service_lease_hash,
                    command_id,
                });
            }
            BrainstemCommand::CmdVel { .. } => {
                if let Some(command) = runtime_command_from_forebrain(command) {
                    self.enqueue_latest_velocity(command_id, command);
                }
                if self.mode == RuntimeMode::Idle || self.mode == RuntimeMode::Error {
                    self.mode = RuntimeMode::Running;
                    status::set_runtime_state(RuntimeState::Running);
                }
            }
            BrainstemCommand::EscapeMotion { .. } => {
                if let Some(command) = runtime_command_from_forebrain(command) {
                    let pending = self.commands.len();
                    for _ in 0..pending {
                        let Some(existing) = self.commands.pop_front() else {
                            break;
                        };
                        if matches!(existing.command, RuntimeCommand::EscapeMotion { .. }) {
                            status::mark_command_interrupted(existing.command_id);
                        } else {
                            let _ = self.commands.push_back(existing);
                        }
                    }
                    let _ = self
                        .commands
                        .push_back(QueuedCommand::safety_recovery(command_id, command));
                }
                if self.mode == RuntimeMode::Idle || self.mode == RuntimeMode::Error {
                    self.mode = RuntimeMode::Running;
                    status::set_runtime_state(RuntimeState::Running);
                }
            }
            _ => {
                if let Some(command) = runtime_command_from_forebrain(command) {
                    let _ = self
                        .commands
                        .push_back(QueuedCommand::new(command_id, command));
                }
                if self.mode == RuntimeMode::Idle || self.mode == RuntimeMode::Error {
                    self.mode = RuntimeMode::Running;
                    status::set_runtime_state(RuntimeState::Running);
                }
            }
        }
    }

    fn request_motherbrain_reset(&mut self, identity: MotherbrainResetIdentity) {
        let now_ms = self.now_ms();
        status::mark_motherbrain_reset_requested(
            identity.command_id,
            identity.session_hash,
            identity.lease_hash,
        );

        if let Some(record) = self
            .motherbrain_reset_history
            .iter()
            .flatten()
            .find(|record| record.identity == identity)
            .copied()
        {
            Self::replay_motherbrain_reset_outcome(record);
            return;
        }

        let refusal = if identity.command_id == 0 {
            Some(status::MotherbrainResetRefusal::InvalidCommandId)
        } else if !self.motherbrain_reset_hardware_enabled {
            Some(status::MotherbrainResetRefusal::HardwareDisabled)
        } else if !status::active_service_authority_matches(
            identity.session_hash,
            identity.lease_hash,
            now_ms,
            MOTHERBRAIN_RESET_SERVICE_SCOPE,
        ) {
            Some(status::MotherbrainResetRefusal::InvalidServiceAuthority)
        } else if self.active_motherbrain_reset.is_some()
            || !time_reached(now_ms, self.motherbrain_reset_cooldown_until_ms)
        {
            Some(status::MotherbrainResetRefusal::Cooldown)
        } else {
            let snapshot = status::snapshot(now_ms);
            let stopped = snapshot.body_state == BodyState::Idle as u8
                && self.active == ActiveAction::None
                && self.commands.is_empty()
                && self.heartbeat_stop_at_ms.is_none();
            let disarmed = snapshot.oi_mode == 1;
            (!stopped || !disarmed).then_some(status::MotherbrainResetRefusal::UnsafeState)
        };

        if let Some(reason) = refusal {
            let record = MotherbrainResetRecord {
                identity,
                outcome: MotherbrainResetOutcome::Refused(reason),
            };
            self.remember_motherbrain_reset(record);
            Self::replay_motherbrain_reset_outcome(record);
            self.request_audio(AuditoryCue::ServiceFailure);
            return;
        }

        self.hardware.set_motherbrain_reset(true);
        self.active_motherbrain_reset = Some(ActiveMotherbrainReset {
            identity,
            release_at_ms: now_ms.wrapping_add(MOTHERBRAIN_RESET_PULSE_MS),
        });
        self.motherbrain_reset_cooldown_until_ms =
            now_ms.wrapping_add(MOTHERBRAIN_RESET_COOLDOWN_MS);
        let record = MotherbrainResetRecord {
            identity,
            outcome: MotherbrainResetOutcome::Asserted,
        };
        self.remember_motherbrain_reset(record);
        Self::replay_motherbrain_reset_outcome(record);
    }

    fn poll_motherbrain_reset(&mut self) {
        let Some(active) = self.active_motherbrain_reset else {
            return;
        };
        if time_reached(self.now_ms(), active.release_at_ms) {
            self.hardware.set_motherbrain_reset(false);
            self.active_motherbrain_reset = None;
            let record = MotherbrainResetRecord {
                identity: active.identity,
                outcome: MotherbrainResetOutcome::Completed,
            };
            self.remember_motherbrain_reset(record);
            Self::replay_motherbrain_reset_outcome(record);
            self.request_audio(AuditoryCue::ServiceComplete);
        }
    }

    fn remember_motherbrain_reset(&mut self, record: MotherbrainResetRecord) {
        if let Some(existing) = self
            .motherbrain_reset_history
            .iter_mut()
            .flatten()
            .find(|existing| existing.identity == record.identity)
        {
            *existing = record;
            return;
        }
        self.motherbrain_reset_history[self.motherbrain_reset_history_next] = Some(record);
        self.motherbrain_reset_history_next =
            (self.motherbrain_reset_history_next + 1) % MOTHERBRAIN_RESET_HISTORY_CAPACITY;
    }

    fn replay_motherbrain_reset_outcome(record: MotherbrainResetRecord) {
        let identity = record.identity;
        match record.outcome {
            MotherbrainResetOutcome::Refused(reason) => status::mark_motherbrain_reset_refused(
                reason,
                identity.session_hash,
                identity.lease_hash,
            ),
            MotherbrainResetOutcome::Asserted => status::mark_motherbrain_reset_asserted(
                identity.command_id,
                identity.session_hash,
                identity.lease_hash,
            ),
            MotherbrainResetOutcome::Completed => status::mark_motherbrain_reset_completed(
                identity.command_id,
                identity.session_hash,
                identity.lease_hash,
            ),
        }
    }

    fn enqueue_latest_velocity(&mut self, command_id: u32, command: RuntimeCommand) {
        let RuntimeCommand::CmdVel {
            linear_mm_s,
            angular_mrad_s,
            duration_ms: Some(duration_ms),
        } = command
        else {
            let _ = self
                .commands
                .push_back(QueuedCommand::new(command_id, command));
            return;
        };

        let pending = self.commands.len();
        for _ in 0..pending {
            let Some(existing) = self.commands.pop_front() else {
                break;
            };
            if !matches!(existing.command, RuntimeCommand::CmdVel { .. }) {
                let _ = self.commands.push_back(existing);
            } else if existing.command_id != command_id {
                // A newer velocity command has consumed this queued command
                // before it could start. Keep its accepted lifecycle closed
                // even though no motor write was ever issued for it.
                status::mark_command_interrupted(existing.command_id);
            }
        }

        let velocity = ActiveVelocity {
            linear_mm_s,
            angular_mrad_s,
        };
        if matches!(self.active, ActiveAction::Driving { .. })
            && self.active_velocity == Some(velocity)
        {
            // Possession refreshes cmd_vel every control tick.  Restarting the
            // same drive on every refresh makes the Create brake and re-start
            // continuously. Renew its deadline without touching the motor or
            // transferring lifecycle ownership. The ingress lane records the
            // refresh as a compact CommandRenewed event; a changed velocity
            // still preempts immediately below.
            self.active = ActiveAction::Driving {
                stop_at_ms: self.now_ms().wrapping_add(duration_ms),
            };
        } else if matches!(self.active, ActiveAction::Driving { .. }) {
            self.interrupt_active_command();
            self.active = ActiveAction::None;
            let _ = self.commands.push_front(QueuedCommand::new(
                command_id,
                RuntimeCommand::CmdVel {
                    linear_mm_s,
                    angular_mrad_s,
                    duration_ms: Some(duration_ms),
                },
            ));
        } else {
            let _ = self.commands.push_back(QueuedCommand::new(
                command_id,
                RuntimeCommand::CmdVel {
                    linear_mm_s,
                    angular_mrad_s,
                    duration_ms: Some(duration_ms),
                },
            ));
        }
    }

}
