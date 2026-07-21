impl<H> Runtime<H>
where
    H: BrainstemHardware,
{
    fn poll_sensor_stream(&mut self) -> Result<(), BrainstemError> {
        let now_ms = self.now_ms();
        if !self.create_responsive {
            return Ok(());
        }

        if time_reached(now_ms, self.next_charging_sources_poll_ms) {
            self.create_uart
                .request_sensor_packet(&mut self.hardware, CREATE_CHARGING_SOURCES_PACKET)?;
            self.next_charging_sources_poll_ms =
                now_ms.wrapping_add(CREATE_CHARGING_SOURCES_POLL_PERIOD_MS);
            return Ok(());
        }

        if time_reached(now_ms, self.next_complete_sensor_poll_ms) {
            self.create_uart
                .request_sensor_packet(&mut self.hardware, CREATE_COMPLETE_SENSOR_PACKET)?;
            self.next_complete_sensor_poll_ms =
                now_ms.wrapping_add(CREATE_COMPLETE_SENSOR_POLL_PERIOD_MS);
            return Ok(());
        }

        let Some(mut stream) = self.sensor_stream else {
            return Ok(());
        };
        if time_reached(now_ms, stream.next_request_ms) {
            self.create_uart
                .request_sensor_packet(&mut self.hardware, stream.packet_id)?;
            stream.next_request_ms = now_ms.wrapping_add(stream.period_ms);
        }
        self.sensor_stream = Some(stream);
        Ok(())
    }

    fn poll_imu(&mut self) {
        if !body::IMU_ENABLED {
            status::mark_imu_health(crate::drivers::imu::ImuHealth::Absent);
            return;
        }

        let now_ms = self.now_ms();
        if !time_reached(now_ms, self.next_imu_poll_ms) {
            return;
        }
        self.next_imu_poll_ms = now_ms.wrapping_add(body::IMU_POLL_PERIOD_MS.max(1));

        match self.hardware.poll_imu_sample(now_ms) {
            Ok(Some(sample)) => status::mark_imu_sample(sample),
            Ok(None) => {}
            Err(health) => status::mark_imu_health(health),
        }
    }

    fn poll_charging_indicator(&mut self) {
        let active = if body::CREATE_CHARGING_INDICATOR_ENABLED {
            self.hardware.charging_indicator_active()
        } else {
            None
        };
        status::mark_create_charging_indicator(active);
    }

    fn observe_audio_transitions(&mut self) {
        const IMU_RECOVERY_STABLE_MS: u32 = 500;
        const MOTION_INCONSISTENCY_COOLDOWN_MS: u32 = 5_000;

        let now_ms = self.now_ms();
        let snapshot = status::snapshot(now_ms);
        let low_battery_threshold = if self.low_battery_active {
            LOW_BATTERY_AUDIO_CLEAR_PERCENT
        } else {
            LOW_BATTERY_PERCENT
        };
        let low_battery = snapshot.create_sensor_capacity_mah > 0
            && u32::from(snapshot.create_sensor_charge_mah) * 100
                <= u32::from(snapshot.create_sensor_capacity_mah) * low_battery_threshold;
        if low_battery && !self.low_battery_active {
            self.request_audio(AuditoryCue::LowBattery);
        }
        self.low_battery_active = low_battery;

        let charging = status::charging_interlock_active(&snapshot)
            || snapshot.create_sensor_charging_sources & 0b10 != 0;
        if charging && !self.charging_active {
            self.request_audio(AuditoryCue::DockContact);
            self.docking_active = false;
        }
        self.charging_active = charging;

        let imu_fault = matches!(
            snapshot.imu_health,
            x if x == status::ImuHealthCode::Fault as u8
                || x == status::ImuHealthCode::Absent as u8
        );
        if imu_fault && !self.imu_fault_active {
            self.request_audio(AuditoryCue::ImuFault);
            self.imu_fault_active = true;
            self.imu_recovery_since_ms = None;
        } else if self.imu_fault_active && snapshot.imu_health == status::ImuHealthCode::Ok as u8 {
            let recovery_since = *self.imu_recovery_since_ms.get_or_insert(now_ms);
            if now_ms.wrapping_sub(recovery_since) >= IMU_RECOVERY_STABLE_MS {
                self.request_audio(AuditoryCue::Recovery);
                self.imu_fault_active = false;
                self.imu_recovery_since_ms = None;
            }
        } else if self.imu_fault_active {
            self.imu_recovery_since_ms = None;
        }
        let inconsistent =
            snapshot.imu_motion_consistency == status::MotionConsistencyCode::Inconsistent as u8;
        if inconsistent
            && !self.last_motion_inconsistent
            && time_reached(now_ms, self.motion_inconsistency_cooldown_until_ms)
        {
            self.request_audio(AuditoryCue::MotionInconsistency);
            self.motion_inconsistency_cooldown_until_ms =
                now_ms.wrapping_add(MOTION_INCONSISTENCY_COOLDOWN_MS);
        }
        self.last_motion_inconsistent = inconsistent;

        if self.docking_active && snapshot.create_sensor_ir_byte != 0 && self.last_dock_ir == 0 {
            self.request_audio(AuditoryCue::DockSeen);
        }
        self.last_dock_ir = snapshot.create_sensor_ir_byte;

        let full_ready = self.create_responsive && snapshot.oi_mode == 3;
        if full_ready && !self.create_full_ready {
            let cue = if self.restart_create_pending {
                self.restart_create_pending = false;
                AuditoryCue::ServiceComplete
            } else if self.ever_create_full_ready {
                AuditoryCue::Recovery
            } else {
                AuditoryCue::Armed
            };
            self.request_audio(cue);
            self.ever_create_full_ready = true;
        }
        self.create_full_ready = full_ready;
    }

}
