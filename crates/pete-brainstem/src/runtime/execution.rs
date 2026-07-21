impl<H> Runtime<H>
where
    H: BrainstemHardware,
{
    fn start_next_command(&mut self) -> Result<(), BrainstemError> {
        let Some(queued) = self.commands.pop_front() else {
            status::set_command(None);
            return Ok(());
        };
        let command = queued.command;
        let now_ms = self.now_ms();
        if self.dock_departure_pending && requires_dock_departure(command) {
            // Full mode terminates Create 1 charging.  Once the charge signal
            // drops, back off the Home Base before starting the caller's
            // body-neutral motion command.  Keep that command queued and do
            // not give the internal departure its lifecycle identity.
            if status::charging_interlock_active(&status::snapshot(now_ms)) {
                let _ = self.commands.push_front(queued);
                return Ok(());
            }
            let _ = self.commands.push_front(queued);
            self.start_dock_departure(now_ms)?;
            return Ok(());
        }
        let command_code = status::set_command(Some(command));
        self.active_command_id = Some(queued.command_id);
        self.safety_recovery_motion = queued.safety_recovery;
        status::mark_command_started(queued.command_id, command_code);
        match command {
            RuntimeCommand::Stop | RuntimeCommand::StopDrive => {
                self.stop_drive()?;
                self.active = ActiveAction::None;
            }
            RuntimeCommand::EStop => {
                self.stop_drive()?;
                self.estop_latched = true;
                status::mark_estop_latched();
                status::mark_safety_tripped(status::SafetyEventKind::EStop);
                self.request_audio(AuditoryCue::EStop);
                self.active = ActiveAction::None;
            }
            RuntimeCommand::ClearEStop => {
                self.estop_latched = false;
                status::mark_estop_cleared();
                status::mark_safety_cleared(status::SafetyEventKind::EStop);
                self.request_audio(AuditoryCue::SafetyClear);
            }
            RuntimeCommand::WakeCreate => {
                self.create_responsive = false;
                status::set_oi_mode_unknown();
                status::set_body_state(BodyState::WaitingForCreate);
                match status::known_create_power_state(status::snapshot(now_ms).create_power_state)
                {
                    Some(false) => {
                        self.push_event(BrainstemEvent::CreatePowerOnRequested);
                        self.hardware.begin_power_toggle_pulse();
                        self.active = ActiveAction::PowerPulse {
                            release_at_ms: now_ms.wrapping_add(body::POWER_TOGGLE_PULSE_MS),
                            wake_wait_until_ms: Some(
                                now_ms.wrapping_add(body::CREATE_WAKE_WAIT_MS),
                            ),
                            power_on: true,
                        };
                    }
                    known_state => {
                        self.active = ActiveAction::WaitForCreate {
                            deadline_ms: now_ms.wrapping_add(body::CREATE_RESPONSIVE_TIMEOUT_MS),
                            next_probe_ms: now_ms,
                            response_bytes: 0,
                            oi_started: false,
                            // UNKNOWN gets one documented best-effort pulse
                            // after a full probe timeout. Known ON is probe-only:
                            // an RX failure must never toggle a running Create off.
                            allow_power_toggle_on_timeout: known_state.is_none(),
                        };
                    }
                };
            }
            RuntimeCommand::SleepCreate => {
                self.create_responsive = false;
                status::set_oi_mode_unknown();
                self.stop_drive()?;
                match status::known_create_power_state(status::snapshot(now_ms).create_power_state)
                {
                    Some(false) => {
                        // Pin 3 is a toggle, so sleeping an already-OFF Create
                        // succeeds without touching POWER_TOGGLE.
                        status::set_body_state(BodyState::Idle);
                        self.active = ActiveAction::None;
                    }
                    Some(true) => {
                        status::set_body_state(BodyState::PowerCycling);
                        self.push_event(BrainstemEvent::CreatePowerOffRequested);
                        self.hardware.begin_power_toggle_pulse();
                        self.active = ActiveAction::PowerPulse {
                            release_at_ms: now_ms.wrapping_add(body::POWER_TOGGLE_PULSE_MS),
                            wake_wait_until_ms: None,
                            power_on: false,
                        };
                    }
                    None => {
                        // Refuse an ambiguous toggle after stopping output. The
                        // command is reported interrupted, not completed.
                        status::set_body_state(BodyState::Idle);
                        self.refuse_active_command();
                        self.active = ActiveAction::None;
                    }
                }
            }
            RuntimeCommand::StartOi => {
                self.create_uart
                    .start_oi(&mut self.hardware, &mut self.events)?;
                status::set_body_state(BodyState::OiStarted);
                self.active = ActiveAction::Settle {
                    until_ms: now_ms.wrapping_add(body::POST_START_SETTLE_MS),
                };
            }
            RuntimeCommand::SetCreateBaud(baud) => {
                self.create_uart.flush_rx(&mut self.hardware);
                self.hardware
                    .set_create_uart_baud(baud)
                    .map_err(|_| BrainstemError::UartFraming)?;
                self.create_responsive = false;
                status::set_oi_mode_unknown();
                self.next_full_mode_refresh_ms = now_ms;
            }
            RuntimeCommand::SetMode(mode) => {
                self.create_uart
                    .set_mode(&mut self.hardware, &mut self.events, mode)?;
                if mode == crate::commands::CreateOiMode::Full {
                    self.next_full_mode_refresh_ms =
                        now_ms.wrapping_add(FULL_MODE_REFRESH_PERIOD_MS);
                }
                self.active = ActiveAction::Settle {
                    until_ms: now_ms.wrapping_add(body::POST_MODE_SETTLE_MS),
                };
            }
            RuntimeCommand::Drive {
                left_mm_s,
                right_mm_s,
                duration_ms,
            } => self.start_drive_direct(left_mm_s, right_mm_s, Some(duration_ms), now_ms)?,
            RuntimeCommand::DriveDirect {
                left_mm_s,
                right_mm_s,
                duration_ms,
            } => self.start_drive_direct(left_mm_s, right_mm_s, duration_ms, now_ms)?,
            RuntimeCommand::CmdVel {
                linear_mm_s,
                angular_mrad_s,
                duration_ms,
            } => {
                self.start_cmd_vel(linear_mm_s, angular_mrad_s, duration_ms, now_ms)?;
                if linear_mm_s != 0 || angular_mrad_s != 0 {
                    status::mark_velocity_stream_active(
                        queued.command_id,
                        linear_mm_s,
                        angular_mrad_s,
                    );
                }
            }
            RuntimeCommand::ClearSafetyLatch { kind } => {
                self.clear_safety_latch(Some(safety_latch_kind_to_event(kind)));
            }
            RuntimeCommand::CarefulMode { ttl_ms } => {
                self.enter_careful_mode(ttl_ms)?;
            }
            RuntimeCommand::EscapeMotion {
                kind,
                hazard_generation,
                linear_mm_s,
                angular_mrad_s,
                ttl_ms,
            } => {
                status::validate_escape_motion(
                    kind,
                    hazard_generation,
                    linear_mm_s,
                    angular_mrad_s,
                    ttl_ms,
                )
                .map_err(|_| BrainstemError::CreateNoResponse)?;
                self.active_escape = Some(ActiveEscape {
                    kind,
                    hazard_generation,
                    linear_mm_s,
                    angular_mrad_s,
                    ttl_ms,
                });
                self.start_cmd_vel(linear_mm_s, angular_mrad_s, Some(ttl_ms), now_ms)?;
            }
            RuntimeCommand::HeartbeatStop { timeout_ms } => {
                self.heartbeat_stop_at_ms = Some(now_ms.wrapping_add(timeout_ms));
            }
            RuntimeCommand::DriveArc {
                velocity_mm_s,
                radius_mm,
                duration_ms,
            } => self.start_drive_arc(velocity_mm_s, radius_mm, duration_ms, now_ms)?,
            RuntimeCommand::RequestSensors { packet_id } => {
                self.create_uart
                    .request_sensor_packet(&mut self.hardware, packet_id)?;
            }
            RuntimeCommand::StreamSensors {
                enabled,
                packet_id,
                period_ms,
            } => {
                if enabled {
                    self.sensor_stream = Some(SensorStream {
                        packet_id,
                        period_ms: period_ms.max(RUNTIME_TICK_MS),
                        next_request_ms: now_ms,
                    });
                } else {
                    self.sensor_stream = None;
                }
            }
            RuntimeCommand::ClearMotionQueue => {
                self.clear_motion_queue()?;
            }
            RuntimeCommand::DefineChirp {
                kind,
                tones,
                tone_count,
            } => {
                let index = feedback_index(kind);
                self.chirps[index] = tones;
                self.chirp_counts[index] = tone_count.min(MAX_SONG_TONES as u8);
                self.song_durations_ms[feedback_slot(kind) as usize] =
                    tone_duration_ms(&tones, self.chirp_counts[index]);
                if self.create_responsive
                    && self
                        .create_uart
                        .define_song(
                            &mut self.hardware,
                            &mut self.events,
                            feedback_slot(kind),
                            &self.chirps[index],
                            self.chirp_counts[index],
                        )
                        .is_err()
                {
                    status::increment_audio_dropped_or_replaced(1);
                }
            }
            RuntimeCommand::PlayFeedback { kind } => {
                if self.audio.silent() {
                    status::increment_audio_suppressed_by_silent();
                    self.active = ActiveAction::None;
                    self.complete_active_command();
                    return Ok(());
                }
                if !self.create_responsive {
                    self.active = ActiveAction::None;
                    self.complete_active_command();
                    return Ok(());
                }
                let (tones, tone_count) = self.feedback_tones(kind);
                if !self.audio.playback_available(now_ms) {
                    status::increment_audio_dropped_or_replaced(1);
                } else if self
                    .create_uart
                    .define_song(
                        &mut self.hardware,
                        &mut self.events,
                        feedback_slot(kind),
                        &tones,
                        tone_count,
                    )
                    .and_then(|()| {
                        self.create_uart.play_song(
                            &mut self.hardware,
                            &mut self.events,
                            feedback_slot(kind),
                        )
                    })
                    .is_err()
                {
                    status::increment_audio_dropped_or_replaced(1);
                } else {
                    self.audio
                        .mark_manual_played(now_ms, tone_duration_ms(&tones, tone_count));
                }
            }
            RuntimeCommand::SetAudioSilent { silent } => {
                self.set_audio_silent(silent);
            }
            RuntimeCommand::CalibrateTurn {
                angular_mrad_s,
                duration_ms,
            } => self.start_cmd_vel(0, angular_mrad_s, Some(duration_ms), now_ms)?,
            RuntimeCommand::OrientationProbe {
                angular_mrad_s,
                duration_ms,
            } => self.start_orientation_probe(angular_mrad_s, duration_ms, now_ms)?,
            RuntimeCommand::ResetOdometry => {
                status::mark_odometry_reset();
            }
            RuntimeCommand::ZeroImuOrientation => {
                if status::zero_imu_orientation_from_gravity()
                    && self.safety_latch_kind == Some(status::SafetyEventKind::Tilt)
                {
                    self.clear_safety_latch(Some(status::SafetyEventKind::Tilt));
                }
            }
            RuntimeCommand::ClearImuOrientation => {
                status::clear_imu_orientation_calibration();
            }
            RuntimeCommand::SongPlay { id } => {
                if self.audio.silent() {
                    status::increment_audio_suppressed_by_silent();
                } else if !self.audio.playback_available(now_ms) {
                    status::increment_audio_dropped_or_replaced(1);
                } else if self.create_responsive
                    && self
                        .create_uart
                        .play_song(&mut self.hardware, &mut self.events, id)
                        .is_err()
                {
                    status::increment_audio_dropped_or_replaced(1);
                } else if self.create_responsive {
                    let duration_ms = self
                        .song_durations_ms
                        .get(id as usize)
                        .copied()
                        .unwrap_or(0)
                        .max(1_000);
                    self.audio.mark_manual_played(now_ms, duration_ms);
                }
            }
            RuntimeCommand::SongDefine {
                id,
                tones,
                tone_count,
            } => {
                if let Some(duration) = self.song_durations_ms.get_mut(id as usize) {
                    *duration = tone_duration_ms(&tones, tone_count);
                }
                if self.create_responsive
                    && self
                        .create_uart
                        .define_song(&mut self.hardware, &mut self.events, id, &tones, tone_count)
                        .is_err()
                {
                    status::increment_audio_dropped_or_replaced(1);
                }
            }
            RuntimeCommand::Dock => {
                self.ensure_create_responsive()?;
                self.create_uart
                    .seek_dock(&mut self.hardware, &mut self.events)?;
                self.docking_active = true;
                self.last_dock_ir = status::snapshot(now_ms).create_sensor_ir_byte;
            }
            RuntimeCommand::SetLights {
                led_bits,
                color,
                intensity,
            } => {
                self.create_uart.set_lights(
                    &mut self.hardware,
                    &mut self.events,
                    led_bits,
                    color,
                    intensity,
                )?;
            }
        }

        if self.active == ActiveAction::None {
            self.complete_active_command();
        }
        Ok(())
    }

    fn advance_active_action(&mut self) -> Result<(), BrainstemError> {
        let now_ms = self.now_ms();
        match self.active {
            ActiveAction::None => Ok(()),
            ActiveAction::PowerPulse {
                release_at_ms,
                wake_wait_until_ms,
                power_on,
            } => {
                if time_reached(now_ms, release_at_ms) {
                    self.hardware.end_power_toggle_pulse();
                    self.push_event(BrainstemEvent::CreatePowerToggled);
                    if wake_wait_until_ms.is_none() {
                        status::set_create_power_on(power_on);
                    }
                    self.active = match wake_wait_until_ms {
                        Some(until_ms) => ActiveAction::WakeSettle { until_ms },
                        None => ActiveAction::None,
                    };
                    if self.active == ActiveAction::None {
                        self.complete_active_command();
                    }
                }
                Ok(())
            }
            ActiveAction::Settle { until_ms } => {
                if time_reached(now_ms, until_ms) {
                    self.active = ActiveAction::None;
                    self.complete_active_command();
                }
                Ok(())
            }
            ActiveAction::WakeSettle { until_ms } => {
                if time_reached(now_ms, until_ms) {
                    self.active = ActiveAction::WaitForCreate {
                        deadline_ms: now_ms.wrapping_add(body::CREATE_RESPONSIVE_TIMEOUT_MS),
                        next_probe_ms: now_ms,
                        response_bytes: 0,
                        oi_started: false,
                        allow_power_toggle_on_timeout: false,
                    };
                }
                Ok(())
            }
            ActiveAction::WaitForCreate {
                deadline_ms,
                next_probe_ms,
                mut response_bytes,
                oi_started,
                allow_power_toggle_on_timeout,
            } => {
                while let Some(event) = self.events.pop_front() {
                    match event {
                        BrainstemEvent::CreatePacketReceived { bytes, .. } => {
                            response_bytes = response_bytes.saturating_add(bytes.len() as u8);
                        }
                        BrainstemEvent::Error(error) => return Err(error),
                        _ => {}
                    }
                }
                status::set_wake_probe_progress(
                    response_bytes as u32,
                    WAKE_PROBE_RESPONSE_BYTES_REQUIRED as u32,
                );

                if response_bytes >= WAKE_PROBE_RESPONSE_BYTES_REQUIRED {
                    self.create_responsive = true;
                    status::set_create_power_on(true);
                    self.active = ActiveAction::None;
                    self.complete_active_command();
                    return Ok(());
                }

                if time_reached(now_ms, deadline_ms) {
                    if allow_power_toggle_on_timeout {
                        self.push_event(BrainstemEvent::CreatePowerOnRequested);
                        self.hardware.begin_power_toggle_pulse();
                        self.active = ActiveAction::PowerPulse {
                            release_at_ms: now_ms.wrapping_add(body::POWER_TOGGLE_PULSE_MS),
                            wake_wait_until_ms: Some(
                                now_ms.wrapping_add(body::CREATE_WAKE_WAIT_MS),
                            ),
                            power_on: true,
                        };
                        return Ok(());
                    }
                    self.create_responsive = false;
                    status::set_create_power_unknown();
                    status::set_oi_mode_unknown();
                    status::mark_uart_rx_error();
                    // Do not silently escalate a failed probe into another
                    // power cycle. An explicit service-scoped restart remains
                    // available to an attended diagnostic operator.
                    self.active = ActiveAction::None;
                    self.fail_active_command(BrainstemError::Timeout);
                    return Ok(());
                }

                if time_reached(now_ms, next_probe_ms) {
                    if !oi_started {
                        self.create_uart.flush_rx(&mut self.hardware);
                        self.create_uart
                            .start_oi(&mut self.hardware, &mut self.events)?;
                        self.active = ActiveAction::WaitForCreate {
                            deadline_ms,
                            next_probe_ms: now_ms.wrapping_add(body::POST_START_SETTLE_MS),
                            response_bytes: 0,
                            oi_started: true,
                            allow_power_toggle_on_timeout,
                        };
                        return Ok(());
                    }
                    self.create_uart.flush_rx(&mut self.hardware);
                    response_bytes = 0;
                    status::set_wake_probe_progress(
                        response_bytes as u32,
                        WAKE_PROBE_RESPONSE_BYTES_REQUIRED as u32,
                    );
                    self.create_uart.request_sensor_packet(
                        &mut self.hardware,
                        body::CREATE_SENSOR_PROBE_PACKET,
                    )?;
                    self.active = ActiveAction::WaitForCreate {
                        deadline_ms,
                        next_probe_ms: now_ms.wrapping_add(SENSOR_PROBE_PERIOD_MS),
                        response_bytes,
                        oi_started,
                        allow_power_toggle_on_timeout,
                    };
                } else {
                    self.active = ActiveAction::WaitForCreate {
                        deadline_ms,
                        next_probe_ms,
                        response_bytes,
                        oi_started,
                        allow_power_toggle_on_timeout,
                    };
                }
                Ok(())
            }
            ActiveAction::Driving { stop_at_ms } => {
                if time_reached(now_ms, stop_at_ms) {
                    self.stop_drive()?;
                    self.active = ActiveAction::None;
                    self.finish_contact_withdrawal(
                        status::ContactWithdrawalOutcome::Completed,
                        None,
                        true,
                    );
                    self.complete_active_command();
                }
                Ok(())
            }
            ActiveAction::DockDeparture { stop_at_ms } => {
                if time_reached(now_ms, stop_at_ms) {
                    self.stop_drive()?;
                    self.active = ActiveAction::None;
                    status::set_body_state(BodyState::Idle);
                }
                Ok(())
            }
        }
    }

    fn start_dock_departure(&mut self, now_ms: u32) -> Result<(), BrainstemError> {
        self.ensure_create_responsive()?;
        self.dock_departure_pending = false;
        // Dock departure is a fixed, body-local 1.5 second operation. A
        // browser motion heartbeat is shorter (900 ms) and supervises the
        // caller's primitive, not this bounded transition. Clear its deadline
        // before starting so the watchdog cannot cancel the reviewed undock.
        self.heartbeat_stop_at_ms = None;
        self.active_velocity = None;
        status::set_body_state(BodyState::Moving);
        self.stop_sent = false;
        self.create_uart.drive_direct(
            &mut self.hardware,
            &mut self.events,
            DOCK_DEPARTURE_SPEED_MM_S,
            DOCK_DEPARTURE_SPEED_MM_S,
            DOCK_DEPARTURE_DURATION_MS,
        )?;
        self.active = ActiveAction::DockDeparture {
            stop_at_ms: now_ms.wrapping_add(DOCK_DEPARTURE_DURATION_MS),
        };
        Ok(())
    }

    fn stop_drive(&mut self) -> Result<(), BrainstemError> {
        self.create_uart
            .stop(&mut self.hardware, &mut self.events)?;
        self.stop_sent = true;
        self.active_velocity = None;
        self.active_escape = None;
        status::clear_velocity_stream();
        // Once STOP has been sent successfully there is no live motion for the
        // heartbeat watchdog to supervise. Leaving the old deadline armed
        // would later revoke an otherwise valid control lease after a normal
        // TTL-bounded motion pulse had already stopped.
        self.heartbeat_stop_at_ms = None;
        Ok(())
    }

    fn start_cmd_vel(
        &mut self,
        linear_mm_s: i16,
        angular_mrad_s: i16,
        duration_ms: Option<u32>,
        now_ms: u32,
    ) -> Result<(), BrainstemError> {
        if linear_mm_s == 0 && angular_mrad_s == 0 {
            self.stop_drive()?;
            self.active = ActiveAction::None;
            return Ok(());
        }
        let half_delta = angular_mrad_s as i32 * CREATE_AXLE_TRACK_MM / 2_000;
        let left = clamp_i16(linear_mm_s as i32 - half_delta);
        let right = clamp_i16(linear_mm_s as i32 + half_delta);
        self.start_drive_direct(left, right, duration_ms, now_ms)?;
        self.active_velocity = Some(ActiveVelocity {
            linear_mm_s,
            angular_mrad_s,
        });
        Ok(())
    }

    fn start_orientation_probe(
        &mut self,
        angular_mrad_s: i16,
        duration_ms: u32,
        now_ms: u32,
    ) -> Result<(), BrainstemError> {
        let angular_abs = abs_i32(angular_mrad_s as i32);
        if angular_abs == 0 || duration_ms == 0 {
            self.stop_drive()?;
            self.active = ActiveAction::None;
            return Ok(());
        }

        self.ensure_orientation_probe_allowed(now_ms)?;
        if !status::zero_imu_orientation_from_gravity() {
            self.stop_drive()?;
            return Err(BrainstemError::CreateNoResponse);
        }
        if self.safety_latch_kind == Some(status::SafetyEventKind::Tilt) {
            self.clear_safety_latch(Some(status::SafetyEventKind::Tilt));
        }
        self.ensure_orientation_probe_allowed(now_ms)?;

        status::mark_odometry_reset();
        self.start_cmd_vel(0, clamp_i16(angular_abs), Some(duration_ms), now_ms)
    }

    fn ensure_orientation_probe_allowed(&mut self, now_ms: u32) -> Result<(), BrainstemError> {
        if self.estop_latched
            || self.dock_departure_pending
            || self.charging_interlock_latched
            || (self.safety_latched
                && self.safety_latch_kind != Some(status::SafetyEventKind::Tilt))
        {
            self.stop_drive()?;
            return Err(BrainstemError::CreateNoResponse);
        }
        self.ensure_create_responsive()?;

        let snapshot = status::snapshot(now_ms);
        let flags = snapshot.create_sensor_flags;
        let wheel_drop = flags & (1 << 2) != 0;
        let cliff = flags & ((1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)) != 0;
        let imu_ready = body::IMU_ENABLED
            && snapshot.imu_health == status::ImuHealthCode::Ok as u8
            && snapshot.imu_sample_count > 0
            && snapshot.imu_sample_age_ms <= ORIENTATION_PROBE_IMU_MAX_AGE_MS
            && snapshot.imu_accel_magnitude_mm_s2 >= ORIENTATION_PROBE_MIN_ACCEL_MM_S2
            && snapshot.imu_accel_magnitude_mm_s2 <= ORIENTATION_PROBE_MAX_ACCEL_MM_S2;
        let imu_still_tilted = snapshot.imu_tilt_magnitude_mrad as i16 >= body::IMU_TILT_STOP_MRAD;
        let imu_impact = snapshot.imu_impact_score_mm_s2 >= body::IMU_IMPACT_STOP_MM_S2;
        if wheel_drop
            || cliff
            || status::charging_interlock_active(&snapshot)
            || !imu_ready
            || imu_impact
            || (imu_still_tilted && self.safety_latch_kind != Some(status::SafetyEventKind::Tilt))
        {
            self.stop_drive()?;
            return Err(BrainstemError::CreateNoResponse);
        }
        Ok(())
    }

    fn clear_motion_queue(&mut self) -> Result<(), BrainstemError> {
        let pending = self.commands.len();
        for _ in 0..pending {
            let Some(command) = self.commands.pop_front() else {
                break;
            };
            if !is_motion_command(command.command) {
                let _ = self.commands.push_back(command);
            }
        }
        if matches!(
            self.active,
            ActiveAction::Driving { .. } | ActiveAction::DockDeparture { .. }
        ) {
            self.interrupt_active_command();
            self.stop_drive()?;
            self.active = ActiveAction::None;
        }
        Ok(())
    }

}
