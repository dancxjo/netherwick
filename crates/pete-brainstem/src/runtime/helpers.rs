fn time_reached(now_ms: u32, deadline_ms: u32) -> bool {
    now_ms.wrapping_sub(deadline_ms) < u32::MAX / 2
}

fn low_battery_and_charging(snapshot: &status::BrainstemStatus) -> bool {
    create_charging_active(snapshot)
        && snapshot.create_sensor_capacity_mah > 0
        && u32::from(snapshot.create_sensor_charge_mah) * 100
            <= u32::from(snapshot.create_sensor_capacity_mah) * LOW_BATTERY_PERCENT
}

fn create_charging_active(snapshot: &status::BrainstemStatus) -> bool {
    snapshot.create_charging_indicator_state == 2
        || matches!(snapshot.create_sensor_charging_state, 1..=3)
}

fn safety_auditory_cue(kind: status::SafetyEventKind) -> AuditoryCue {
    match kind {
        status::SafetyEventKind::EStop => AuditoryCue::EStop,
        status::SafetyEventKind::Bump => AuditoryCue::BumpContact,
        status::SafetyEventKind::Cliff => AuditoryCue::Cliff,
        status::SafetyEventKind::WheelDrop => AuditoryCue::WheelDrop,
        status::SafetyEventKind::Heartbeat => AuditoryCue::HeartbeatLost,
        status::SafetyEventKind::Tilt => AuditoryCue::Tilt,
        status::SafetyEventKind::Impact => AuditoryCue::Impact,
        status::SafetyEventKind::Charging => AuditoryCue::DockContact,
    }
}

fn command_preempts_contact_withdrawal(command: BrainstemCommand) -> bool {
    matches!(
        command,
        BrainstemCommand::Stop | BrainstemCommand::EStop | BrainstemCommand::Disarm
    )
}

fn runtime_command_from_forebrain(command: BrainstemCommand) -> Option<RuntimeCommand> {
    match command {
        BrainstemCommand::Ping
        | BrainstemCommand::Status
        | BrainstemCommand::Bootsel
        | BrainstemCommand::Arm
        | BrainstemCommand::Disarm
        | BrainstemCommand::RestartCreate => None,
        BrainstemCommand::ResetMotherbrain => None,
        BrainstemCommand::Stop => Some(RuntimeCommand::Stop),
        BrainstemCommand::EStop => Some(RuntimeCommand::EStop),
        BrainstemCommand::ClearEStop => Some(RuntimeCommand::ClearEStop),
        BrainstemCommand::SetMode(mode) => Some(RuntimeCommand::SetMode(mode)),
        BrainstemCommand::CmdVel {
            linear_mm_s,
            angular_mrad_s,
            ttl_ms,
            ..
        } => Some(RuntimeCommand::CmdVel {
            linear_mm_s,
            angular_mrad_s,
            duration_ms: Some(ttl_ms),
        }),
        BrainstemCommand::DriveDirect {
            left_mm_s,
            right_mm_s,
            ttl_ms,
            ..
        } => Some(RuntimeCommand::DriveDirect {
            left_mm_s,
            right_mm_s,
            duration_ms: Some(ttl_ms),
        }),
        BrainstemCommand::DriveArc {
            velocity_mm_s,
            radius_mm,
            ttl_ms,
            ..
        } => Some(RuntimeCommand::DriveArc {
            velocity_mm_s,
            radius_mm,
            duration_ms: Some(ttl_ms),
        }),
        BrainstemCommand::Unsupported { .. } => None,
        BrainstemCommand::ClearSafetyLatch { kind, .. } => {
            Some(RuntimeCommand::ClearSafetyLatch { kind })
        }
        BrainstemCommand::CarefulMode { ttl_ms, .. } => {
            Some(RuntimeCommand::CarefulMode { ttl_ms })
        }
        BrainstemCommand::EscapeMotion {
            kind,
            hazard_generation,
            linear_mm_s,
            angular_mrad_s,
            ttl_ms,
            ..
        } => Some(RuntimeCommand::EscapeMotion {
            kind,
            hazard_generation,
            linear_mm_s,
            angular_mrad_s,
            ttl_ms,
        }),
        BrainstemCommand::HeartbeatStop { timeout_ms, .. } => {
            Some(RuntimeCommand::HeartbeatStop { timeout_ms })
        }
        BrainstemCommand::RequestSensors { packet_id, .. } => {
            Some(RuntimeCommand::RequestSensors { packet_id })
        }
        BrainstemCommand::StreamSensors {
            enabled,
            packet_id,
            period_ms,
            ..
        } => Some(RuntimeCommand::StreamSensors {
            enabled,
            packet_id,
            period_ms,
        }),
        BrainstemCommand::ClearMotionQueue { .. } => Some(RuntimeCommand::ClearMotionQueue),
        BrainstemCommand::DefineChirp {
            kind,
            tones,
            tone_count,
            ..
        } => Some(RuntimeCommand::DefineChirp {
            kind,
            tones,
            tone_count,
        }),
        BrainstemCommand::PlayFeedback { kind, .. } => Some(RuntimeCommand::PlayFeedback { kind }),
        BrainstemCommand::SetAudioSilent { silent, .. } => {
            Some(RuntimeCommand::SetAudioSilent { silent })
        }
        BrainstemCommand::PowerState { request, .. } => match request {
            PowerStateRequest::Wake => Some(RuntimeCommand::WakeCreate),
            PowerStateRequest::Sleep => Some(RuntimeCommand::SleepCreate),
            PowerStateRequest::StartOi => Some(RuntimeCommand::StartOi),
            PowerStateRequest::DebugBaud19200 => Some(RuntimeCommand::SetCreateBaud(19_200)),
            PowerStateRequest::DebugBaud57600 => Some(RuntimeCommand::SetCreateBaud(57_600)),
            PowerStateRequest::DebugBaud115200 => Some(RuntimeCommand::SetCreateBaud(115_200)),
        },
        BrainstemCommand::CalibrateTurn {
            angular_mrad_s,
            duration_ms,
            ..
        } => Some(RuntimeCommand::CalibrateTurn {
            angular_mrad_s,
            duration_ms,
        }),
        BrainstemCommand::OrientationProbe {
            angular_mrad_s,
            duration_ms,
            ..
        } => Some(RuntimeCommand::OrientationProbe {
            angular_mrad_s,
            duration_ms,
        }),
        BrainstemCommand::ResetOdometry { .. } => Some(RuntimeCommand::ResetOdometry),
        BrainstemCommand::ZeroImuOrientation { .. } => Some(RuntimeCommand::ZeroImuOrientation),
        BrainstemCommand::ClearImuOrientation { .. } => Some(RuntimeCommand::ClearImuOrientation),
        BrainstemCommand::SongPlay { id } => Some(RuntimeCommand::SongPlay { id }),
        BrainstemCommand::SongDefine {
            id,
            tones,
            tone_count,
            ..
        } => Some(RuntimeCommand::SongDefine {
            id,
            tones,
            tone_count,
        }),
        BrainstemCommand::Dock => Some(RuntimeCommand::Dock),
        BrainstemCommand::SetLights {
            led_bits,
            color,
            intensity,
        } => Some(RuntimeCommand::SetLights {
            led_bits,
            color,
            intensity,
        }),
        BrainstemCommand::GetCapabilities => None,
        BrainstemCommand::GetEvents { .. } => None,
    }
}

fn clamp_i16(value: i32) -> i16 {
    value.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

fn abs_i32(value: i32) -> i32 {
    value.saturating_abs()
}

fn is_motion_command(command: RuntimeCommand) -> bool {
    matches!(
        command,
        RuntimeCommand::DriveDirect { .. }
            | RuntimeCommand::CmdVel { .. }
            | RuntimeCommand::EscapeMotion { .. }
            | RuntimeCommand::DriveArc { .. }
            | RuntimeCommand::Drive { .. }
            | RuntimeCommand::StopDrive
            | RuntimeCommand::CalibrateTurn { .. }
            | RuntimeCommand::OrientationProbe { .. }
            | RuntimeCommand::Dock
    )
}

fn requires_dock_departure(command: RuntimeCommand) -> bool {
    match command {
        RuntimeCommand::CmdVel {
            linear_mm_s,
            angular_mrad_s,
            ..
        } => linear_mm_s != 0 || angular_mrad_s != 0,
        RuntimeCommand::EscapeMotion {
            linear_mm_s,
            angular_mrad_s,
            ..
        } => linear_mm_s != 0 || angular_mrad_s != 0,
        RuntimeCommand::Drive {
            left_mm_s,
            right_mm_s,
            ..
        }
        | RuntimeCommand::DriveDirect {
            left_mm_s,
            right_mm_s,
            ..
        } => left_mm_s != 0 || right_mm_s != 0,
        RuntimeCommand::DriveArc { velocity_mm_s, .. } => velocity_mm_s != 0,
        RuntimeCommand::Dock | RuntimeCommand::StopDrive => false,
        _ => is_motion_command(command),
    }
}

fn recoverable_safety_latch(kind: Option<status::SafetyEventKind>) -> bool {
    matches!(
        kind,
        Some(status::SafetyEventKind::Bump | status::SafetyEventKind::Cliff)
    )
}

fn safety_latch_kind_to_event(kind: SafetyLatchKind) -> status::SafetyEventKind {
    match kind {
        SafetyLatchKind::Bump => status::SafetyEventKind::Bump,
        SafetyLatchKind::Cliff => status::SafetyEventKind::Cliff,
        SafetyLatchKind::WheelDrop => status::SafetyEventKind::WheelDrop,
        SafetyLatchKind::Heartbeat => status::SafetyEventKind::Heartbeat,
        SafetyLatchKind::Tilt => status::SafetyEventKind::Tilt,
        SafetyLatchKind::Impact => status::SafetyEventKind::Impact,
        SafetyLatchKind::Charging => status::SafetyEventKind::Charging,
    }
}

fn feedback_index(kind: FeedbackKind) -> usize {
    match kind {
        FeedbackKind::Ok => 0,
        FeedbackKind::Error => 1,
        FeedbackKind::Armed => 2,
        FeedbackKind::LostTarget => 3,
        FeedbackKind::DockSeen => 4,
        FeedbackKind::Danger => 5,
    }
}

fn feedback_slot(kind: FeedbackKind) -> u8 {
    FEEDBACK_SLOT_BASE + feedback_index(kind) as u8
}

fn default_feedback_tones(kind: FeedbackKind) -> ([SongTone; MAX_SONG_TONES], u8) {
    let mut tones = [SongTone::default(); MAX_SONG_TONES];
    let notes: &[(u8, u8)] = match kind {
        FeedbackKind::Ok => &[(60, 8), (64, 8), (67, 12)],
        FeedbackKind::Error => &[(64, 8), (62, 8), (60, 12)],
        // Solresol "fasolsi": prepare / make ready.
        FeedbackKind::Armed => &[(65, 8), (67, 8), (71, 12)],
        FeedbackKind::LostTarget => &[(55, 8), (52, 8), (48, 12)],
        FeedbackKind::DockSeen => &[(67, 8), (71, 8), (74, 12)],
        FeedbackKind::Danger => &[(48, 6), (48, 6), (48, 12)],
    };
    for (i, (note, duration_64ths)) in notes.iter().enumerate() {
        tones[i] = SongTone {
            note: *note,
            duration_64ths: *duration_64ths,
        };
    }
    (tones, notes.len() as u8)
}
