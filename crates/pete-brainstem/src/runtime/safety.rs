impl<H> Runtime<H>
where
    H: BrainstemHardware,
{
    fn enforce_safety_policy(&mut self) -> Result<(), BrainstemError> {
        let now_ms = self.now_ms();
        let snapshot = status::snapshot(now_ms);
        let flags = snapshot.create_sensor_flags;
        let bump = flags & ((1 << 0) | (1 << 1)) != 0;
        let wheel_drop = flags & (1 << 2) != 0;
        let cliff = flags & ((1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)) != 0;
        let imu_ok = body::IMU_ENABLED && snapshot.imu_health == status::ImuHealthCode::Ok as u8;
        let tilt_observed =
            imu_ok && snapshot.imu_tilt_magnitude_mrad as i16 >= body::IMU_TILT_STOP_MRAD;
        let tilt = if tilt_observed {
            match self.tilt_observed_since_ms {
                Some(started_at_ms) => {
                    time_reached(now_ms, started_at_ms.wrapping_add(IMU_TILT_LATCH_HOLD_MS))
                }
                None => {
                    self.tilt_observed_since_ms = Some(now_ms);
                    false
                }
            }
        } else {
            self.tilt_observed_since_ms = None;
            false
        };
        let impact = imu_ok && snapshot.imu_impact_score_mm_s2 >= body::IMU_IMPACT_STOP_MM_S2;
        let charging = status::charging_interlock_active(&snapshot);
        let home_base = snapshot.create_sensor_charging_sources & 0b10 != 0;

        // Packet 34 is the source of truth for Home Base contact. If Pete was
        // physically removed before an unstarted departure consumed the first
        // motion request, do not carry that reverse program into a later,
        // already off-dock command.
        if !home_base && self.dock_departure_pending {
            self.dock_departure_pending = false;
        }

        // The first complete observation establishes the edge baseline. A
        // bumper held through boot is evidence, not permission to move.
        if !self.safety_observation_initialized {
            self.last_bump = bump;
            self.last_cliff = cliff;
            self.last_wheel_drop = wheel_drop;
            self.safety_observation_initialized = true;
        }
        let fresh_bump_edge = bump && !self.last_bump;
        if let Some(active_escape) = self.active_escape {
            let dominating_hazard = if wheel_drop {
                Some(status::SafetyEventKind::WheelDrop)
            } else if tilt {
                Some(status::SafetyEventKind::Tilt)
            } else if impact {
                Some(status::SafetyEventKind::Impact)
            } else if charging || home_base {
                Some(status::SafetyEventKind::Charging)
            } else {
                match active_escape.kind {
                    SafetyLatchKind::Bump if cliff => Some(status::SafetyEventKind::Cliff),
                    SafetyLatchKind::Cliff if bump => Some(status::SafetyEventKind::Bump),
                    _ => None,
                }
            };
            if let Some(kind) = dominating_hazard {
                status::mark_safety_tripped(kind);
                self.apply_safety_response(kind, SafetyResponse::Stop)?;
                self.request_audio(safety_auditory_cue(kind));
                self.update_safety_edges(bump, cliff, wheel_drop);
                return Ok(());
            }
            if status::validate_escape_motion(
                active_escape.kind,
                active_escape.hazard_generation,
                active_escape.linear_mm_s,
                active_escape.angular_mrad_s,
                active_escape.ttl_ms,
            )
            .is_err()
            {
                self.interrupt_active_command();
                self.commands.clear();
                self.active = ActiveAction::None;
                self.stop_drive()?;
                self.update_safety_edges(bump, cliff, wheel_drop);
                return Ok(());
            }
            // The acknowledged latch remains active and visible. Only this
            // exact generation-bound segment may move through it.
            self.update_safety_edges(bump, cliff, wheel_drop);
            return Ok(());
        }

        if self.careful_mode_active(now_ms) {
            // CAREFUL transfers responsibility for these observations to the
            // active possessor. Keep publishing the raw sensors, but do not
            // turn them back into motor gates until the explicit lease ends.
            self.update_safety_edges(bump, cliff, wheel_drop);
            self.clear_sensor_gates_for_careful();
            return Ok(());
        }

        if home_base && !wheel_drop && !tilt && !impact {
            self.clear_dock_contact_latch();
            if !matches!(self.active, ActiveAction::DockDeparture { .. })
                && !self.dock_departure_pending
            {
                self.clear_motion_queue()?;
                self.stop_drive()?;
                self.finish_contact_withdrawal(
                    status::ContactWithdrawalOutcome::SafetyPreempted,
                    Some(status::SafetyEventKind::Charging),
                    true,
                );
                self.dock_departure_pending = true;
            }
            // Packet 34 lets a Home Base contact replace the conservative
            // unknown-source charging interlock with internal dock handling.
            self.charging_interlock_latched = false;
            return Ok(());
        }

        if charging && !wheel_drop && !tilt && !impact {
            if !self.charging_interlock_latched {
                self.clear_motion_queue()?;
                self.stop_drive()?;
                self.finish_contact_withdrawal(
                    status::ContactWithdrawalOutcome::SafetyPreempted,
                    Some(status::SafetyEventKind::Charging),
                    true,
                );
                self.charging_interlock_latched = true;
            }
            return Ok(());
        }

        if !bump && !cliff && !wheel_drop && !tilt && !impact && !charging {
            self.update_safety_edges(bump, cliff, wheel_drop);
            return Ok(());
        }
        self.update_safety_edges(bump, cliff, wheel_drop);

        if wheel_drop {
            if self.safety_latch_kind != Some(status::SafetyEventKind::WheelDrop) {
                status::mark_safety_tripped(status::SafetyEventKind::WheelDrop);
                status::mark_wheel_drop_latched();
                self.latch_safety(status::SafetyEventKind::WheelDrop);
                self.interrupt_active_command();
                self.commands.clear();
                self.stop_drive()?;
                self.active = ActiveAction::None;
                self.finish_contact_withdrawal(
                    status::ContactWithdrawalOutcome::SafetyPreempted,
                    Some(status::SafetyEventKind::WheelDrop),
                    true,
                );
                self.request_audio(AuditoryCue::WheelDrop);
            }
            return Ok(());
        }
        // A bump latch permits only its own bounded reverse. A stronger local
        // safety observation must still preempt that reflex deterministically.
        if self.safety_latched
            && !(self.active_contact_withdrawal.is_some() && (tilt || impact || cliff))
        {
            return Ok(());
        }

        let (kind, response) = if tilt {
            status::mark_safety_tripped(status::SafetyEventKind::Tilt);
            (status::SafetyEventKind::Tilt, SafetyResponse::Stop)
        } else if impact {
            status::mark_safety_tripped(status::SafetyEventKind::Impact);
            (status::SafetyEventKind::Impact, SafetyResponse::Stop)
        } else if cliff {
            status::mark_safety_tripped(status::SafetyEventKind::Cliff);
            (status::SafetyEventKind::Cliff, SafetyResponse::Stop)
        } else if bump && fresh_bump_edge && self.unsafe_forward_output() {
            status::mark_safety_tripped(status::SafetyEventKind::Bump);
            (
                status::SafetyEventKind::Bump,
                SafetyResponse::ContactWithdrawal,
            )
        } else if bump {
            // A stationary press or boot-restored contact remains observable
            // and latched, but cannot initiate authority-independent motion.
            status::mark_safety_tripped(status::SafetyEventKind::Bump);
            (status::SafetyEventKind::Bump, SafetyResponse::Stop)
        } else if wheel_drop {
            status::mark_safety_tripped(status::SafetyEventKind::WheelDrop);
            (status::SafetyEventKind::WheelDrop, SafetyResponse::Stop)
        } else {
            (status::SafetyEventKind::Bump, SafetyResponse::Stop)
        };
        self.apply_safety_response(kind, response)?;
        self.request_audio(safety_auditory_cue(kind));
        Ok(())
    }

    fn apply_safety_response(
        &mut self,
        kind: status::SafetyEventKind,
        response: SafetyResponse,
    ) -> Result<(), BrainstemError> {
        match response {
            SafetyResponse::Stop => {
                self.latch_safety(kind);
                self.interrupt_active_command();
                self.commands.clear();
                self.stop_drive()?;
                self.active = ActiveAction::None;
                self.finish_contact_withdrawal(
                    status::ContactWithdrawalOutcome::SafetyPreempted,
                    Some(kind),
                    true,
                );
                Ok(())
            }
            SafetyResponse::ContactWithdrawal => {
                self.latch_safety(kind);
                let command_id = self.active_command_id.unwrap_or(0);
                self.interrupt_active_command();
                self.commands.clear();
                self.active = ActiveAction::None;
                self.stop_drive()?;
                if kind == status::SafetyEventKind::Bump {
                    let snapshot = status::snapshot(self.now_ms());
                    self.start_contact_withdrawal(
                        (snapshot.create_sensor_flags & 0b11) as u8,
                        command_id,
                        snapshot.odometry_distance_mm,
                    );
                    self.mode = RuntimeMode::Running;
                    status::set_runtime_state(RuntimeState::Running);
                }
                let _ = self.commands.push_front(QueuedCommand::safety_recovery(
                    0,
                    RuntimeCommand::CmdVel {
                        linear_mm_s: -CONTACT_WITHDRAWAL_SPEED_MM_S,
                        angular_mrad_s: 0,
                        duration_ms: Some(CONTACT_WITHDRAWAL_DURATION_MS),
                    },
                ));
                Ok(())
            }
        }
    }

    fn latch_safety(&mut self, kind: status::SafetyEventKind) {
        self.safety_latched = true;
        self.safety_latch_kind = Some(kind);
    }

    fn clear_dock_contact_latch(&mut self) {
        let Some(kind @ (status::SafetyEventKind::Bump | status::SafetyEventKind::Cliff)) =
            self.safety_latch_kind
        else {
            return;
        };
        // Packet 0 can arrive before the private packet-34 poll identifies
        // Home Base, briefly interpreting dock geometry as a bump/cliff
        // incident. Reconcile only those two contact latches once packet 34
        // proves the source; every stronger latch remains untouched.
        status::mark_safety_cleared(kind);
        self.request_audio(AuditoryCue::SafetyClear);
        self.safety_latched = false;
        self.safety_latch_kind = None;
    }

    fn clear_safety_latch(&mut self, expected: Option<status::SafetyEventKind>) {
        if expected == Some(status::SafetyEventKind::Charging) && self.charging_interlock_latched {
            self.charging_interlock_latched = false;
            return;
        }

        let Some(kind) = self.safety_latch_kind else {
            self.safety_latched = false;
            return;
        };
        if expected.is_some_and(|expected| expected != kind) {
            return;
        }
        let snapshot = status::snapshot(self.now_ms());
        let flags = snapshot.create_sensor_flags;
        let physical_condition_active = match kind {
            status::SafetyEventKind::Bump => flags & 0b11 != 0,
            status::SafetyEventKind::WheelDrop => flags & (1 << 2) != 0,
            status::SafetyEventKind::Cliff => flags & 0b1111_0000 != 0,
            status::SafetyEventKind::Tilt => {
                snapshot.imu_health == status::ImuHealthCode::Ok as u8
                    && snapshot.imu_tilt_magnitude_mrad as i16 >= body::IMU_TILT_STOP_MRAD
            }
            status::SafetyEventKind::Impact => {
                snapshot.imu_health == status::ImuHealthCode::Ok as u8
                    && snapshot.imu_impact_score_mm_s2 >= body::IMU_IMPACT_STOP_MM_S2
            }
            status::SafetyEventKind::Charging => status::charging_interlock_active(&snapshot),
            _ => false,
        };
        if physical_condition_active {
            return;
        }
        status::mark_safety_cleared(kind);
        self.request_audio(AuditoryCue::SafetyClear);
        if kind == status::SafetyEventKind::WheelDrop {
            status::mark_wheel_drop_cleared();
        }
        self.safety_latched = false;
        self.safety_latch_kind = None;
    }

    fn enter_careful_mode(&mut self, requested_ttl_ms: u32) -> Result<(), BrainstemError> {
        self.ensure_create_responsive()?;
        if self.estop_latched {
            return Err(BrainstemError::CreateNoResponse);
        }
        let now_ms = self.now_ms();
        let ttl_ms = requested_ttl_ms.clamp(CAREFUL_MODE_MIN_TTL_MS, CAREFUL_MODE_MAX_TTL_MS);
        self.clear_sensor_gates_for_careful();
        self.careful_mode_until_ms = Some(now_ms.wrapping_add(ttl_ms));
        status::set_careful_mode_until(self.careful_mode_until_ms);
        Ok(())
    }

    fn careful_mode_active(&self, now_ms: u32) -> bool {
        self.careful_mode_until_ms
            .is_some_and(|deadline_ms| !time_reached(now_ms, deadline_ms))
    }

    fn cancel_careful_mode(&mut self) {
        self.careful_mode_until_ms = None;
        status::set_careful_mode_until(None);
    }

    fn clear_sensor_gates_for_careful(&mut self) {
        if let Some(kind) = self.safety_latch_kind.take() {
            status::mark_safety_cleared(kind);
            self.request_audio(AuditoryCue::SafetyClear);
            if kind == status::SafetyEventKind::WheelDrop {
                status::mark_wheel_drop_cleared();
            }
        }
        self.safety_latched = false;
        self.charging_interlock_latched = false;
        self.dock_departure_pending = false;
        self.tilt_observed_since_ms = None;
    }

    fn enforce_careful_mode_timeout(&mut self) -> Result<(), BrainstemError> {
        let Some(deadline_ms) = self.careful_mode_until_ms else {
            return Ok(());
        };
        if !time_reached(self.now_ms(), deadline_ms) {
            return Ok(());
        }

        self.cancel_careful_mode();
        self.interrupt_active_command();
        self.commands.clear();
        self.active = ActiveAction::None;
        self.stop_drive()?;
        self.relatch_current_sensor_gate();
        Ok(())
    }

    fn relatch_current_sensor_gate(&mut self) {
        let snapshot = status::snapshot(self.now_ms());
        let flags = snapshot.create_sensor_flags;
        let imu_ok = body::IMU_ENABLED && snapshot.imu_health == status::ImuHealthCode::Ok as u8;
        let kind = if flags & (1 << 2) != 0 {
            Some(status::SafetyEventKind::WheelDrop)
        } else if flags & 0b1111_0000 != 0 {
            Some(status::SafetyEventKind::Cliff)
        } else if imu_ok && snapshot.imu_tilt_magnitude_mrad as i16 >= body::IMU_TILT_STOP_MRAD {
            Some(status::SafetyEventKind::Tilt)
        } else if imu_ok && snapshot.imu_impact_score_mm_s2 >= body::IMU_IMPACT_STOP_MM_S2 {
            Some(status::SafetyEventKind::Impact)
        } else if flags & 0b11 != 0 {
            Some(status::SafetyEventKind::Bump)
        } else {
            None
        };

        if let Some(kind) = kind {
            status::mark_safety_tripped(kind);
            if kind == status::SafetyEventKind::WheelDrop {
                status::mark_wheel_drop_latched();
            }
            self.latch_safety(kind);
        } else if status::charging_interlock_active(&snapshot) {
            self.charging_interlock_latched = true;
        }
    }

    fn unsafe_forward_output(&self) -> bool {
        self.active_velocity
            .is_some_and(|velocity| velocity.linear_mm_s > 0)
    }

    fn feedback_tones(&self, kind: FeedbackKind) -> ([SongTone; MAX_SONG_TONES], u8) {
        let index = feedback_index(kind);
        if self.chirp_counts[index] > 0 {
            return (self.chirps[index], self.chirp_counts[index]);
        }
        default_feedback_tones(kind)
    }

    fn set_audio_silent(&mut self, silent: bool) {
        let dropped = self.audio.set_silent(silent);
        status::increment_audio_dropped_or_replaced(dropped);
        status::set_audio_silent(silent);
    }

    fn request_audio(&mut self, cue: AuditoryCue) {
        let now_ms = self.now_ms();
        status::mark_audio_cue_requested(cue.code());
        match self.audio.request(cue, now_ms) {
            CueRequestResult::Suppressed => status::increment_audio_suppressed_by_silent(),
            CueRequestResult::Dropped => status::increment_audio_dropped_or_replaced(1),
            CueRequestResult::Ready | CueRequestResult::Queued => {}
        }
    }

    fn poll_audio(&mut self) {
        let now_ms = self.now_ms();
        if !self.create_responsive {
            return;
        }
        let Some(cue) = self.audio.take_ready(now_ms) else {
            return;
        };
        let (tones, tone_count) = cue_tones(cue);
        let played = self
            .create_uart
            .define_song(
                &mut self.hardware,
                &mut self.events,
                AUTOMATIC_CUE_SLOT,
                &tones,
                tone_count,
            )
            .and_then(|()| {
                self.create_uart
                    .play_song(&mut self.hardware, &mut self.events, AUTOMATIC_CUE_SLOT)
            })
            .is_ok();
        if played {
            self.audio.mark_played(cue, now_ms);
            status::mark_audio_cue_played(cue.code(), now_ms);
        } else {
            status::increment_audio_dropped_or_replaced(1);
        }
    }

}
