pub fn set_runtime_state(state: RuntimeState) {
    RUNTIME_STATE.store(state as u8, Ordering::Relaxed);
}

pub fn set_body_state(state: BodyState) {
    BODY_STATE.store(state as u8, Ordering::Relaxed);
}

pub fn set_command(command: Option<RuntimeCommand>) -> u8 {
    let code = match command {
        None => CommandCode::None,
        Some(RuntimeCommand::WakeCreate) => CommandCode::WakeCreate,
        Some(RuntimeCommand::SleepCreate) => CommandCode::SleepCreate,
        Some(RuntimeCommand::SetMode(CreateOiMode::Passive)) => CommandCode::SetOiPassive,
        Some(RuntimeCommand::SetMode(CreateOiMode::Safe)) => CommandCode::SetOiSafe,
        Some(RuntimeCommand::SetMode(CreateOiMode::Full)) => CommandCode::SetOiFull,
        Some(RuntimeCommand::Stop) => CommandCode::StopDrive,
        Some(RuntimeCommand::EStop) => CommandCode::StopDrive,
        Some(RuntimeCommand::ClearEStop) => CommandCode::None,
        Some(RuntimeCommand::DriveDirect { .. }) => CommandCode::Drive,
        Some(RuntimeCommand::CmdVel { .. }) => CommandCode::Drive,
        Some(RuntimeCommand::DriveArc { .. }) => CommandCode::Drive,
        Some(RuntimeCommand::EscapeMotion { .. }) => CommandCode::Drive,
        Some(RuntimeCommand::ClearSafetyLatch { .. })
        | Some(RuntimeCommand::CarefulMode { .. })
        | Some(RuntimeCommand::HeartbeatStop { .. }) => CommandCode::Behavior,
        Some(RuntimeCommand::StartOi) => CommandCode::StartOi,
        Some(RuntimeCommand::Drive { .. }) => CommandCode::Drive,
        Some(RuntimeCommand::StopDrive) => CommandCode::StopDrive,
        Some(RuntimeCommand::RequestSensors { .. })
        | Some(RuntimeCommand::StreamSensors { .. })
        | Some(RuntimeCommand::ClearMotionQueue)
        | Some(RuntimeCommand::DefineChirp { .. })
        | Some(RuntimeCommand::PlayFeedback { .. })
        | Some(RuntimeCommand::SetAudioSilent { .. })
        | Some(RuntimeCommand::CalibrateTurn { .. })
        | Some(RuntimeCommand::OrientationProbe { .. })
        | Some(RuntimeCommand::ResetOdometry)
        | Some(RuntimeCommand::ZeroImuOrientation)
        | Some(RuntimeCommand::ClearImuOrientation)
        | Some(RuntimeCommand::SetCreateBaud(_))
        | Some(RuntimeCommand::SongDefine { .. })
        | Some(RuntimeCommand::SongPlay { .. })
        | Some(RuntimeCommand::Dock)
        | Some(RuntimeCommand::SetLights { .. }) => CommandCode::None,
    };
    CURRENT_COMMAND.store(code as u8, Ordering::Relaxed);
    code as u8
}

pub fn last_dispatched_command_id() -> u32 {
    LAST_DISPATCHED_COMMAND_ID.load(Ordering::Relaxed)
}

pub fn last_dispatched_service_identity() -> (u32, u32) {
    (
        LAST_DISPATCHED_SERVICE_SESSION_HASH.load(Ordering::Relaxed),
        LAST_DISPATCHED_SERVICE_LEASE_HASH.load(Ordering::Relaxed),
    )
}

pub fn mark_command_started(command_id: u32, command_code: u8) {
    LAST_STARTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
    record_public_event(
        PublicEventKind::CommandStarted,
        command_id,
        command_code as u32,
        0,
    );
}

pub fn mark_velocity_stream_active(command_id: u32, linear_mm_s: i16, angular_mrad_s: i16) {
    ACTIVE_VELOCITY_STREAM_ID.store(command_id, Ordering::Relaxed);
    ACTIVE_VELOCITY_STREAM_A.store(encode_i16(linear_mm_s), Ordering::Relaxed);
    ACTIVE_VELOCITY_STREAM_B.store(encode_i16(angular_mrad_s), Ordering::Relaxed);
    ACTIVE_VELOCITY_STREAM.store(ON, Ordering::Release);
}

pub fn clear_velocity_stream() {
    ACTIVE_VELOCITY_STREAM.store(OFF, Ordering::Release);
    ACTIVE_VELOCITY_STREAM_ID.store(0, Ordering::Relaxed);
}

fn matching_velocity_stream(a: u32, b: u32) -> Option<u32> {
    (ACTIVE_VELOCITY_STREAM.load(Ordering::Acquire) == ON
        && ACTIVE_VELOCITY_STREAM_A.load(Ordering::Relaxed) == a
        && ACTIVE_VELOCITY_STREAM_B.load(Ordering::Relaxed) == b)
        .then(|| ACTIVE_VELOCITY_STREAM_ID.load(Ordering::Relaxed))
        .filter(|command_id| *command_id != 0)
}

fn preempt_pending_commands_for_safety(command_id: u32) -> (Option<u32>, Option<u32>) {
    let velocity_pending =
        PENDING_VELOCITY_KIND.load(Ordering::Relaxed) == ControlCommandCode::CmdVel as u8;
    let velocity_was_renewal = PENDING_VELOCITY_IS_RENEWAL.load(Ordering::Relaxed) == ON;
    let replaced_velocity_id = (velocity_pending && !velocity_was_renewal)
        .then(|| PENDING_VELOCITY_ID.load(Ordering::Relaxed))
        .filter(|pending_id| *pending_id != command_id);
    PENDING_VELOCITY_KIND.store(ControlCommandCode::None as u8, Ordering::Relaxed);
    PENDING_VELOCITY_IS_RENEWAL.store(OFF, Ordering::Relaxed);

    let pending_kind = PENDING_COMMAND_KIND.load(Ordering::Relaxed);
    let replaced_command_id = (pending_kind != ControlCommandCode::None as u8)
        .then(|| PENDING_COMMAND_ID.load(Ordering::Relaxed))
        .filter(|pending_id| *pending_id != command_id);

    (replaced_command_id, replaced_velocity_id)
}

#[cfg(any(feature = "pico-w", feature = "rpi5"))]
pub fn submit_control_command(
    command_id: u32,
    command: BrainstemCommand,
) -> Result<(), CommandRejectReason> {
    submit_control_command_with_service_identity(command_id, command, 0, 0)
}

#[cfg(any(feature = "pico-w", feature = "rpi5"))]
pub fn submit_service_control_command(
    command_id: u32,
    command: BrainstemCommand,
    session_hash: u32,
    lease_hash: u32,
) -> Result<(), CommandRejectReason> {
    submit_control_command_with_service_identity(command_id, command, session_hash, lease_hash)
}

#[cfg(any(feature = "pico-w", feature = "rpi5"))]
fn submit_control_command_with_service_identity(
    command_id: u32,
    command: BrainstemCommand,
    service_session_hash: u32,
    service_lease_hash: u32,
) -> Result<(), CommandRejectReason> {
    if matches!(command, BrainstemCommand::Unsupported { .. }) {
        return reject_control_command(
            command_id,
            command_seq(command),
            ControlCommandCode::None,
            CommandRejectReason::Unsupported,
        );
    }
    if matches!(
        command,
        BrainstemCommand::Status | BrainstemCommand::Ping | BrainstemCommand::GetEvents { .. }
    ) {
        LAST_ACCEPTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
        record_public_event(PublicEventKind::CommandAccepted, command_id, 0, 0);
        return Ok(());
    }
    if matches!(command, BrainstemCommand::GetCapabilities) {
        LAST_ACCEPTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
        record_public_event(PublicEventKind::CommandAccepted, command_id, 0, 0);
        return Ok(());
    }
    if let BrainstemCommand::EscapeMotion {
        kind,
        hazard_generation,
        linear_mm_s,
        angular_mrad_s,
        ttl_ms,
        ..
    } = command
    {
        validate_escape_motion(kind, hazard_generation, linear_mm_s, angular_mrad_s, ttl_ms)?;
    }

    let Some((kind, a, b, c, d, duration_ms)) = encode_control_command(command) else {
        return reject_control_command(
            command_id,
            command_seq(command),
            ControlCommandCode::None,
            CommandRejectReason::Unsupported,
        );
    };

    if kind == ControlCommandCode::CmdVel {
        let seq = command_seq(command);
        let renewed_stream_id = matching_velocity_stream(a, b);
        let velocity_pending =
            PENDING_VELOCITY_KIND.load(Ordering::Relaxed) == ControlCommandCode::CmdVel as u8;
        if velocity_pending
            && !seq_is_current_or_newer(seq, PENDING_VELOCITY_SEQ.load(Ordering::Relaxed))
        {
            return reject_control_command(
                command_id,
                seq,
                kind,
                CommandRejectReason::StaleSequence,
            );
        }
        let replaced_was_renewal =
            velocity_pending && PENDING_VELOCITY_IS_RENEWAL.load(Ordering::Relaxed) == ON;
        let replaced_command_id = velocity_pending
            .then(|| PENDING_VELOCITY_ID.load(Ordering::Relaxed))
            .filter(|pending_id| *pending_id != command_id && !replaced_was_renewal);

        PENDING_VELOCITY_ID.store(command_id, Ordering::Relaxed);
        PENDING_VELOCITY_A.store(a, Ordering::Relaxed);
        PENDING_VELOCITY_B.store(b, Ordering::Relaxed);
        PENDING_VELOCITY_TTL_MS.store(duration_ms.unwrap_or(0), Ordering::Relaxed);
        PENDING_VELOCITY_SEQ.store(seq, Ordering::Relaxed);
        PENDING_VELOCITY_IS_RENEWAL.store(
            if renewed_stream_id.is_some() { ON } else { OFF },
            Ordering::Relaxed,
        );
        PENDING_VELOCITY_KIND.store(ControlCommandCode::CmdVel as u8, Ordering::Relaxed);
        LAST_ACCEPTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
        if let Some(stream_id) = renewed_stream_id {
            record_public_event(PublicEventKind::CommandRenewed, command_id, stream_id, seq);
        } else {
            record_public_event(PublicEventKind::CommandAccepted, command_id, seq, 0);
        }
        if let Some(replaced_command_id) = replaced_command_id {
            mark_command_interrupted(replaced_command_id);
        }
        return Ok(());
    }

    let mut replaced_command_id = None;
    let mut replaced_velocity_id = None;
    if matches!(kind, ControlCommandCode::Stop | ControlCommandCode::EStop) {
        (replaced_command_id, replaced_velocity_id) =
            preempt_pending_commands_for_safety(command_id);
    } else {
        let pending_kind = PENDING_COMMAND_KIND.load(Ordering::Relaxed);
        let replaces_pending_heartbeat = kind == ControlCommandCode::HeartbeatStop
            && pending_kind == ControlCommandCode::HeartbeatStop as u8;
        if pending_kind != ControlCommandCode::None as u8 && !replaces_pending_heartbeat {
            return reject_control_command(
                command_id,
                command_seq(command),
                kind,
                CommandRejectReason::Busy,
            );
        }
        if replaces_pending_heartbeat {
            let pending_id = PENDING_COMMAND_ID.load(Ordering::Relaxed);
            if pending_id != command_id {
                replaced_command_id = Some(pending_id);
            }
        }
    }

    PENDING_COMMAND_ID.store(command_id, Ordering::Relaxed);
    PENDING_COMMAND_A.store(a, Ordering::Relaxed);
    PENDING_COMMAND_B.store(b, Ordering::Relaxed);
    PENDING_COMMAND_C.store(c, Ordering::Relaxed);
    PENDING_COMMAND_D.store(d, Ordering::Relaxed);
    PENDING_COMMAND_DURATION_MS.store(duration_ms.unwrap_or(0), Ordering::Relaxed);
    PENDING_COMMAND_SEQ.store(command_seq(command), Ordering::Relaxed);
    PENDING_COMMAND_SERVICE_SESSION_HASH.store(service_session_hash, Ordering::Relaxed);
    PENDING_COMMAND_SERVICE_LEASE_HASH.store(service_lease_hash, Ordering::Relaxed);
    PENDING_COMMAND_KIND.store(kind as u8, Ordering::Relaxed);
    LAST_ACCEPTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
    record_public_event(
        PublicEventKind::CommandAccepted,
        command_id,
        command_seq(command),
        kind as u32,
    );
    if let Some(replaced_command_id) = replaced_command_id {
        mark_command_interrupted(replaced_command_id);
    }
    if let Some(replaced_velocity_id) =
        replaced_velocity_id.filter(|velocity_id| Some(*velocity_id) != replaced_command_id)
    {
        mark_command_interrupted(replaced_velocity_id);
    }
    Ok(())
}

#[cfg(any(feature = "pico-w", feature = "rpi5"))]
fn reject_control_command(
    command_id: u32,
    command_seq: u32,
    command_kind: ControlCommandCode,
    reason: CommandRejectReason,
) -> Result<(), CommandRejectReason> {
    LAST_REJECTED_COMMAND_ID.store(command_id, Ordering::Relaxed);
    let detail = ((command_kind as u32) << 8) | reason.code() as u32;
    record_public_event(
        PublicEventKind::CommandRejected,
        command_id,
        command_seq,
        detail,
    );
    Err(reason)
}

pub fn take_control_command() -> Option<BrainstemCommand> {
    let kind = PENDING_COMMAND_KIND.load(Ordering::Relaxed);
    if kind != ControlCommandCode::None as u8 {
        let a = PENDING_COMMAND_A.load(Ordering::Relaxed);
        let b = PENDING_COMMAND_B.load(Ordering::Relaxed);
        let c = PENDING_COMMAND_C.load(Ordering::Relaxed);
        let d = PENDING_COMMAND_D.load(Ordering::Relaxed);
        let duration = match PENDING_COMMAND_DURATION_MS.load(Ordering::Relaxed) {
            0 => None,
            duration_ms => Some(duration_ms),
        };
        let seq = PENDING_COMMAND_SEQ.load(Ordering::Relaxed);
        let command_id = PENDING_COMMAND_ID.load(Ordering::Relaxed);
        let service_session_hash = PENDING_COMMAND_SERVICE_SESSION_HASH.load(Ordering::Relaxed);
        let service_lease_hash = PENDING_COMMAND_SERVICE_LEASE_HASH.load(Ordering::Relaxed);
        PENDING_COMMAND_KIND.store(ControlCommandCode::None as u8, Ordering::Relaxed);
        LAST_DISPATCHED_COMMAND_ID.store(command_id, Ordering::Relaxed);
        LAST_DISPATCHED_SERVICE_SESSION_HASH.store(service_session_hash, Ordering::Relaxed);
        LAST_DISPATCHED_SERVICE_LEASE_HASH.store(service_lease_hash, Ordering::Relaxed);

        return decode_control_command(kind, a, b, c, d, duration, seq);
    }

    let kind = PENDING_VELOCITY_KIND.load(Ordering::Relaxed);
    if kind != ControlCommandCode::CmdVel as u8 {
        return None;
    }

    let a = PENDING_VELOCITY_A.load(Ordering::Relaxed);
    let b = PENDING_VELOCITY_B.load(Ordering::Relaxed);
    let ttl_ms = PENDING_VELOCITY_TTL_MS.load(Ordering::Relaxed);
    let seq = PENDING_VELOCITY_SEQ.load(Ordering::Relaxed);
    let command_id = PENDING_VELOCITY_ID.load(Ordering::Relaxed);
    PENDING_VELOCITY_KIND.store(ControlCommandCode::None as u8, Ordering::Relaxed);
    PENDING_VELOCITY_IS_RENEWAL.store(OFF, Ordering::Relaxed);
    LAST_DISPATCHED_COMMAND_ID.store(command_id, Ordering::Relaxed);
    LAST_DISPATCHED_SERVICE_SESSION_HASH.store(0, Ordering::Relaxed);
    LAST_DISPATCHED_SERVICE_LEASE_HASH.store(0, Ordering::Relaxed);

    Some(BrainstemCommand::CmdVel {
        linear_mm_s: decode_i16(a),
        angular_mrad_s: decode_i16(b),
        ttl_ms,
        seq,
    })
}

pub fn request_session_replace(
    generation: u32,
    session_hash: u32,
    peer_device_hash: u32,
    peer_boot_hash: u32,
) {
    PENDING_SESSION_HASH.store(session_hash, Ordering::Release);
    PENDING_PEER_DEVICE_HASH.store(peer_device_hash, Ordering::Release);
    PENDING_PEER_BOOT_HASH.store(peer_boot_hash, Ordering::Release);
    SESSION_REPLACE_REQUEST.store(generation.max(1), Ordering::Release);
}

pub fn pending_session_replace() -> Option<u32> {
    let request = SESSION_REPLACE_REQUEST.load(Ordering::Acquire);
    (request != 0 && request != SESSION_REPLACE_ACK.load(Ordering::Acquire)).then_some(request)
}

pub fn pending_session_hash() -> u32 {
    PENDING_SESSION_HASH.load(Ordering::Acquire)
}

pub fn acknowledge_session_replace(generation: u32, session_hash: u32) {
    // Publish identity only after the runtime lane has synchronously stopped,
    // cleared its queue and revoked heartbeat state.
    let previous = ACTIVE_SESSION_HASH.load(Ordering::Acquire);
    ACTIVE_SESSION_HASH.store(session_hash, Ordering::Release);
    ACTIVE_SESSION_GENERATION.store(generation, Ordering::Release);
    let previous_device = ACTIVE_PEER_DEVICE_HASH.load(Ordering::Acquire);
    let previous_boot = ACTIVE_PEER_BOOT_HASH.load(Ordering::Acquire);
    let peer_device = PENDING_PEER_DEVICE_HASH.load(Ordering::Acquire);
    let peer_boot = PENDING_PEER_BOOT_HASH.load(Ordering::Acquire);
    ACTIVE_PEER_DEVICE_HASH.store(peer_device, Ordering::Release);
    ACTIVE_PEER_BOOT_HASH.store(peer_boot, Ordering::Release);
    SESSION_REPLACE_ACK.store(generation, Ordering::Release);
    ACTIVE_LEASE_HASH.store(0, Ordering::Release);
    ACTIVE_LEASE_SESSION_HASH.store(0, Ordering::Release);
    ACTIVE_LEASE_EXPIRES_MS.store(0, Ordering::Release);
    revoke_service_authority();
    record_public_event(
        if previous == 0 {
            PublicEventKind::SessionOpened
        } else {
            PublicEventKind::SessionReplaced
        },
        generation,
        previous,
        session_hash,
    );
    if previous_device == peer_device && previous_boot != 0 && previous_boot != peer_boot {
        record_public_event(
            PublicEventKind::PeerRebootDetected,
            previous_boot,
            peer_boot,
            0,
        );
    }
}

pub fn mark_session_rejected(reason: u32) {
    record_public_event(PublicEventKind::SessionRejected, reason, 0, 0);
}

pub fn session_replace_acked(generation: u32) -> bool {
    SESSION_REPLACE_ACK.load(Ordering::Acquire) == generation
}

pub fn active_session_matches(session_hash: u32) -> bool {
    session_hash != 0
        && (ACTIVE_SESSION_HASH.load(Ordering::Acquire) == session_hash
            || DIAGNOSTIC_SESSION_HASH
                .iter()
                .any(|entry| entry.load(Ordering::Acquire) == session_hash))
}
pub fn active_peer_matches(device_hash: u32) -> bool {
    device_hash != 0 && ACTIVE_PEER_DEVICE_HASH.load(Ordering::Acquire) == device_hash
}
pub fn session_peer_matches(session_hash: u32, peer_hash: u32) -> bool {
    if ACTIVE_SESSION_HASH.load(Ordering::Acquire) == session_hash {
        return ACTIVE_PEER_DEVICE_HASH.load(Ordering::Acquire) == peer_hash;
    }
    DIAGNOSTIC_SESSION_HASH
        .iter()
        .position(|entry| entry.load(Ordering::Acquire) == session_hash)
        .is_some_and(|slot| DIAGNOSTIC_PEER_HASH[slot].load(Ordering::Acquire) == peer_hash)
}

pub fn register_diagnostic_session(
    session_hash: u32,
    peer_hash: u32,
    peer_boot_hash: u32,
    role: u8,
    purpose: u8,
    transport: u8,
) {
    let slot = DIAGNOSTIC_SESSION_HASH
        .iter()
        .position(|entry| entry.load(Ordering::Acquire) == session_hash)
        .or_else(|| {
            DIAGNOSTIC_SESSION_HASH
                .iter()
                .position(|entry| entry.load(Ordering::Acquire) == 0)
        })
        .unwrap_or((session_hash as usize) % DIAGNOSTIC_SESSION_CAPACITY);
    DIAGNOSTIC_PEER_HASH[slot].store(peer_hash, Ordering::Release);
    DIAGNOSTIC_PEER_BOOT_HASH[slot].store(peer_boot_hash, Ordering::Release);
    DIAGNOSTIC_ROLE[slot].store(role, Ordering::Release);
    DIAGNOSTIC_PURPOSE[slot].store(purpose, Ordering::Release);
    DIAGNOSTIC_TRANSPORT[slot].store(transport, Ordering::Release);
    DIAGNOSTIC_SESSION_HASH[slot].store(session_hash, Ordering::Release);
    record_public_event(
        PublicEventKind::SessionOpened,
        role as u32,
        peer_hash,
        session_hash,
    );
}

#[derive(Clone, Copy)]
pub struct SessionIdentity {
    pub peer_device_hash: u32,
    pub peer_boot_hash: u32,
    pub role: u8,
    pub purpose: u8,
    pub transport: u8,
}

pub fn session_identity(session_hash: u32) -> Option<SessionIdentity> {
    if ACTIVE_SESSION_HASH.load(Ordering::Acquire) == session_hash {
        return Some(SessionIdentity {
            peer_device_hash: ACTIVE_PEER_DEVICE_HASH.load(Ordering::Acquire),
            peer_boot_hash: ACTIVE_PEER_BOOT_HASH.load(Ordering::Acquire),
            role: 1,
            purpose: 1,
            transport: ACTIVE_TRANSPORT.load(Ordering::Acquire),
        });
    }
    DIAGNOSTIC_SESSION_HASH
        .iter()
        .position(|entry| entry.load(Ordering::Acquire) == session_hash)
        .map(|slot| SessionIdentity {
            peer_device_hash: DIAGNOSTIC_PEER_HASH[slot].load(Ordering::Acquire),
            peer_boot_hash: DIAGNOSTIC_PEER_BOOT_HASH[slot].load(Ordering::Acquire),
            role: DIAGNOSTIC_ROLE[slot].load(Ordering::Acquire),
            purpose: DIAGNOSTIC_PURPOSE[slot].load(Ordering::Acquire),
            transport: DIAGNOSTIC_TRANSPORT[slot].load(Ordering::Acquire),
        })
}

pub fn session_role(session_hash: u32) -> Option<u8> {
    if ACTIVE_SESSION_HASH.load(Ordering::Acquire) == session_hash {
        return Some(1);
    }
    DIAGNOSTIC_SESSION_HASH
        .iter()
        .position(|entry| entry.load(Ordering::Acquire) == session_hash)
        .map(|slot| DIAGNOSTIC_ROLE[slot].load(Ordering::Acquire))
}

pub fn set_session_safety_snapshot(
    estop_latched: bool,
    safety_tripped: bool,
    motion_interlock_latched: bool,
    safety_latch_kind: Option<SafetyEventKind>,
) {
    SESSION_SAFETY_FLAGS.store(
        (estop_latched as u32)
            | ((safety_tripped as u32) << 1)
            | ((motion_interlock_latched as u32) << 2),
        Ordering::Release,
    );
    SESSION_SAFETY_LATCH_KIND.store(
        safety_latch_kind.map_or(0, |kind| kind as u8),
        Ordering::Release,
    );
}

pub fn session_safety_snapshot() -> (bool, bool, bool, Option<SafetyEventKind>) {
    let flags = SESSION_SAFETY_FLAGS.load(Ordering::Acquire);
    (
        flags & 1 != 0,
        flags & 2 != 0,
        flags & 4 != 0,
        safety_event_kind(SESSION_SAFETY_LATCH_KIND.load(Ordering::Acquire)),
    )
}

pub fn safety_hazard_generation() -> u32 {
    SAFETY_HAZARD_GENERATION.load(Ordering::Acquire)
}

pub fn set_careful_mode_until(deadline_ms: Option<u32>) {
    CAREFUL_MODE_UNTIL_MS.store(deadline_ms.unwrap_or(0), Ordering::Release);
}

pub fn careful_mode_remaining_ms(now_ms: u32) -> u32 {
    let deadline_ms = CAREFUL_MODE_UNTIL_MS.load(Ordering::Acquire);
    if deadline_ms == 0 || now_ms.wrapping_sub(deadline_ms) < u32::MAX / 2 {
        0
    } else {
        deadline_ms.wrapping_sub(now_ms)
    }
}

pub fn request_authority_transition(
    generation: u32,
    lease_hash: u32,
    session_hash: u32,
    expires_ms: u32,
) {
    PENDING_LEASE_HASH.store(lease_hash, Ordering::Release);
    PENDING_LEASE_SESSION_HASH.store(session_hash, Ordering::Release);
    ACTIVE_LEASE_EXPIRES_MS.store(expires_ms, Ordering::Release);
    AUTHORITY_REQUEST.store(generation.max(1), Ordering::Release);
}
pub fn pending_authority_transition() -> Option<u32> {
    let request = AUTHORITY_REQUEST.load(Ordering::Acquire);
    (request != 0 && request != AUTHORITY_ACK.load(Ordering::Acquire)).then_some(request)
}
pub fn pending_authority_continues_owner(now_ms: u32) -> bool {
    !authority_expired(now_ms)
        && ACTIVE_LEASE_SESSION_HASH.load(Ordering::Acquire) != 0
        && ACTIVE_LEASE_SESSION_HASH.load(Ordering::Acquire)
            == PENDING_LEASE_SESSION_HASH.load(Ordering::Acquire)
}
pub fn acknowledge_authority_transition(generation: u32) {
    revoke_service_authority();
    ACTIVE_LEASE_HASH.store(
        PENDING_LEASE_HASH.load(Ordering::Acquire),
        Ordering::Release,
    );
    ACTIVE_LEASE_SESSION_HASH.store(
        PENDING_LEASE_SESSION_HASH.load(Ordering::Acquire),
        Ordering::Release,
    );
    AUTHORITY_ACK.store(generation, Ordering::Release);
    record_public_event(
        PublicEventKind::AuthorityChanged,
        generation,
        ACTIVE_LEASE_SESSION_HASH.load(Ordering::Acquire),
        ACTIVE_LEASE_HASH.load(Ordering::Acquire),
    );
}
pub fn install_service_authority(session_hash: u32, lease_hash: u32, expires_ms: u32, scope: u8) {
    ACTIVE_SERVICE_SESSION_HASH.store(session_hash, Ordering::Release);
    ACTIVE_SERVICE_LEASE_HASH.store(lease_hash, Ordering::Release);
    ACTIVE_SERVICE_LEASE_EXPIRES_MS.store(expires_ms, Ordering::Release);
    ACTIVE_SERVICE_SCOPE.store(scope, Ordering::Release);
}
pub fn active_service_authority_matches(
    session_hash: u32,
    lease_hash: u32,
    now_ms: u32,
    scope: u8,
) -> bool {
    let deadline = ACTIVE_SERVICE_LEASE_EXPIRES_MS.load(Ordering::Acquire);
    deadline != 0
        && now_ms.wrapping_sub(deadline) >= u32::MAX / 2
        && ACTIVE_SERVICE_SESSION_HASH.load(Ordering::Acquire) == session_hash
        && ACTIVE_SERVICE_LEASE_HASH.load(Ordering::Acquire) == lease_hash
        && ACTIVE_SERVICE_SCOPE.load(Ordering::Acquire) == scope
}
pub fn revoke_service_authority() {
    ACTIVE_SERVICE_LEASE_HASH.store(0, Ordering::Release);
    ACTIVE_SERVICE_SESSION_HASH.store(0, Ordering::Release);
    ACTIVE_SERVICE_LEASE_EXPIRES_MS.store(0, Ordering::Release);
    ACTIVE_SERVICE_SCOPE.store(0, Ordering::Release);
}
pub fn authority_transition_acked(generation: u32) -> bool {
    AUTHORITY_ACK.load(Ordering::Acquire) == generation
}
pub fn authority_expired(now_ms: u32) -> bool {
    let deadline = ACTIVE_LEASE_EXPIRES_MS.load(Ordering::Acquire);
    deadline == 0 || now_ms.wrapping_sub(deadline) < u32::MAX / 2
}
pub fn has_active_authority(now_ms: u32) -> bool {
    ACTIVE_LEASE_HASH.load(Ordering::Acquire) != 0 && !authority_expired(now_ms)
}
pub fn active_authority_matches(session_hash: u32, lease_hash: u32, now_ms: u32) -> bool {
    !authority_expired(now_ms)
        && ACTIVE_LEASE_HASH.load(Ordering::Acquire) == lease_hash
        && ACTIVE_LEASE_SESSION_HASH.load(Ordering::Acquire) == session_hash
}
pub fn authority_heartbeat_valid(session_hash: u32, lease_hash: u32, now_ms: u32) -> bool {
    // HEARTBEAT_STOP has its own runtime deadline. It must validate the
    // negotiated authority, but must not shorten (or extend) the control
    // lease to the motion heartbeat timeout.
    active_authority_matches(session_hash, lease_hash, now_ms)
}
pub fn revoke_authority() {
    let previous = ACTIVE_LEASE_HASH.load(Ordering::Acquire);
    ACTIVE_LEASE_HASH.store(0, Ordering::Release);
    ACTIVE_LEASE_SESSION_HASH.store(0, Ordering::Release);
    ACTIVE_LEASE_EXPIRES_MS.store(0, Ordering::Release);
    if previous != 0 {
        record_public_event(PublicEventKind::AuthorityChanged, 0, previous, 0);
    }
}
pub fn mark_dhcp_lease_changed(identity_hash: u32, ip: u32) {
    record_public_event(PublicEventKind::DhcpLeaseChanged, identity_hash, ip, 0);
}
pub fn mark_dns_registration_changed(generation: u32, ip: u32) {
    record_public_event(PublicEventKind::DnsRegistrationChanged, generation, ip, 0);
}
pub fn take_expired_authority(now_ms: u32) -> bool {
    if ACTIVE_LEASE_HASH.load(Ordering::Acquire) != 0 && authority_expired(now_ms) {
        revoke_authority();
        true
    } else {
        false
    }
}

pub struct SessionDiagnostics {
    pub primary_session_generation: u32,
    pub diagnostic_sessions: u8,
    pub authority_generation: u32,
    pub authority_active: bool,
    pub authority_session_hash: u32,
    pub authority_owner_role: u8,
    pub authority_owner_device_hash: u32,
    pub authority_owner_boot_hash: u32,
    pub authority_lease_remaining_ms: u32,
    pub service_authority_active: bool,
}

pub fn session_diagnostics(now_ms: u32) -> SessionDiagnostics {
    let authority_active = !authority_expired(now_ms);
    let authority_session_hash = if authority_active {
        ACTIVE_LEASE_SESSION_HASH.load(Ordering::Acquire)
    } else {
        0
    };
    let owner = session_identity(authority_session_hash);
    SessionDiagnostics {
        primary_session_generation: ACTIVE_SESSION_GENERATION.load(Ordering::Acquire),
        diagnostic_sessions: DIAGNOSTIC_SESSION_HASH
            .iter()
            .filter(|entry| entry.load(Ordering::Acquire) != 0)
            .count() as u8,
        authority_generation: AUTHORITY_ACK.load(Ordering::Acquire),
        authority_active,
        authority_session_hash,
        authority_owner_role: owner.map_or(0, |identity| identity.role),
        authority_owner_device_hash: owner.map_or(0, |identity| identity.peer_device_hash),
        authority_owner_boot_hash: owner.map_or(0, |identity| identity.peer_boot_hash),
        authority_lease_remaining_ms: if authority_active {
            ACTIVE_LEASE_EXPIRES_MS
                .load(Ordering::Acquire)
                .wrapping_sub(now_ms)
        } else {
            0
        },
        service_authority_active: ACTIVE_SERVICE_LEASE_HASH.load(Ordering::Acquire) != 0
            && active_service_authority_matches(
                ACTIVE_SERVICE_SESSION_HASH.load(Ordering::Acquire),
                ACTIVE_SERVICE_LEASE_HASH.load(Ordering::Acquire),
                now_ms,
                ACTIVE_SERVICE_SCOPE.load(Ordering::Acquire),
            ),
    }
}
pub fn mark_transport_changed(transport: u8) {
    let previous = ACTIVE_TRANSPORT.load(Ordering::Acquire);
    ACTIVE_TRANSPORT.store(transport, Ordering::Release);
    if previous != 0 && previous != transport {
        record_public_event(
            PublicEventKind::TransportChanged,
            previous as u32,
            transport as u32,
            0,
        );
    }
}

#[cfg(any(feature = "pico-w", feature = "rpi5"))]
fn encode_control_command(
    command: BrainstemCommand,
) -> Option<(ControlCommandCode, u32, u32, u32, u32, Option<u32>)> {
    match command {
        BrainstemCommand::Ping => Some((ControlCommandCode::Ping, 0, 0, 0, 0, None)),
        BrainstemCommand::Arm => Some((ControlCommandCode::Arm, 0, 0, 0, 0, None)),
        BrainstemCommand::Disarm => Some((ControlCommandCode::Disarm, 0, 0, 0, 0, None)),
        BrainstemCommand::EStop => Some((ControlCommandCode::EStop, 0, 0, 0, 0, None)),
        BrainstemCommand::ClearEStop => Some((ControlCommandCode::ClearEStop, 0, 0, 0, 0, None)),
        BrainstemCommand::Stop => Some((ControlCommandCode::Stop, 0, 0, 0, 0, None)),
        BrainstemCommand::Status => Some((ControlCommandCode::Status, 0, 0, 0, 0, None)),
        BrainstemCommand::Bootsel => None,
        BrainstemCommand::SetMode(mode) => Some((
            ControlCommandCode::SetMode,
            match mode {
                CreateOiMode::Passive => 1,
                CreateOiMode::Safe => 2,
                CreateOiMode::Full => 3,
            },
            0,
            0,
            0,
            None,
        )),
        BrainstemCommand::CmdVel {
            linear_mm_s,
            angular_mrad_s,
            ttl_ms,
            ..
        } => Some((
            ControlCommandCode::CmdVel,
            encode_i16(linear_mm_s),
            encode_i16(angular_mrad_s),
            0,
            0,
            Some(ttl_ms),
        )),
        BrainstemCommand::DriveDirect {
            left_mm_s,
            right_mm_s,
            ttl_ms,
            ..
        } => Some((
            ControlCommandCode::DriveDirect,
            encode_i16(left_mm_s),
            encode_i16(right_mm_s),
            0,
            0,
            Some(ttl_ms),
        )),
        BrainstemCommand::DriveArc {
            velocity_mm_s,
            radius_mm,
            ttl_ms,
            ..
        } => Some((
            ControlCommandCode::DriveArc,
            encode_i16(velocity_mm_s),
            encode_i16(radius_mm),
            0,
            0,
            Some(ttl_ms),
        )),
        BrainstemCommand::SongPlay { id } => {
            Some((ControlCommandCode::SongPlay, id as u32, 0, 0, 0, None))
        }
        BrainstemCommand::SongDefine {
            id,
            tones,
            tone_count,
            ..
        } => {
            let tone_count = tone_count.min(MAX_SONG_TONES as u8);
            store_pending_song_tones(&tones, tone_count);
            Some((
                ControlCommandCode::SongDefine,
                id as u32,
                tone_count as u32,
                0,
                0,
                None,
            ))
        }
        BrainstemCommand::Dock => Some((ControlCommandCode::Dock, 0, 0, 0, 0, None)),
        BrainstemCommand::SetLights {
            led_bits,
            color,
            intensity,
        } => Some((
            ControlCommandCode::SetLights,
            led_bits as u32,
            color as u32,
            intensity as u32,
            0,
            None,
        )),
        BrainstemCommand::Unsupported { .. } => None,
        BrainstemCommand::HeartbeatStop { timeout_ms, .. } => Some((
            ControlCommandCode::HeartbeatStop,
            0,
            0,
            0,
            0,
            Some(timeout_ms),
        )),
        BrainstemCommand::ClearSafetyLatch { kind, .. } => Some((
            ControlCommandCode::ClearSafetyLatch,
            encode_safety_latch_kind(kind) as u32,
            0,
            0,
            0,
            None,
        )),
        BrainstemCommand::CarefulMode { ttl_ms, .. } => {
            Some((ControlCommandCode::CarefulMode, 0, 0, 0, 0, Some(ttl_ms)))
        }
        BrainstemCommand::EscapeMotion {
            kind,
            hazard_generation,
            linear_mm_s,
            angular_mrad_s,
            ttl_ms,
            ..
        } => Some((
            ControlCommandCode::EscapeMotion,
            encode_safety_latch_kind(kind) as u32,
            hazard_generation,
            encode_i16(linear_mm_s),
            encode_i16(angular_mrad_s),
            Some(ttl_ms),
        )),
        BrainstemCommand::RequestSensors { packet_id, .. } => Some((
            ControlCommandCode::RequestSensors,
            packet_id as u32,
            0,
            0,
            0,
            None,
        )),
        BrainstemCommand::StreamSensors {
            enabled,
            packet_id,
            period_ms,
            ..
        } => Some((
            ControlCommandCode::StreamSensors,
            enabled as u32,
            packet_id as u32,
            0,
            0,
            Some(period_ms),
        )),
        BrainstemCommand::ClearMotionQueue { .. } => {
            Some((ControlCommandCode::ClearMotionQueue, 0, 0, 0, 0, None))
        }
        BrainstemCommand::DefineChirp {
            kind,
            tones,
            tone_count,
            ..
        } => {
            let tone_count = tone_count.min(MAX_SONG_TONES as u8);
            store_pending_song_tones(&tones, tone_count);
            Some((
                ControlCommandCode::DefineChirp,
                encode_feedback_kind(kind) as u32,
                tone_count as u32,
                0,
                0,
                None,
            ))
        }
        BrainstemCommand::PlayFeedback { kind, .. } => Some((
            ControlCommandCode::PlayFeedback,
            encode_feedback_kind(kind) as u32,
            0,
            0,
            0,
            None,
        )),
        BrainstemCommand::SetAudioSilent { silent, .. } => Some((
            ControlCommandCode::SetAudioSilent,
            silent as u32,
            0,
            0,
            0,
            None,
        )),
        BrainstemCommand::PowerState { request, .. } => Some((
            ControlCommandCode::PowerState,
            encode_power_request(request) as u32,
            0,
            0,
            0,
            None,
        )),
        BrainstemCommand::CalibrateTurn {
            angular_mrad_s,
            duration_ms,
            ..
        } => Some((
            ControlCommandCode::CalibrateTurn,
            encode_i16(angular_mrad_s),
            0,
            0,
            0,
            Some(duration_ms),
        )),
        BrainstemCommand::OrientationProbe {
            angular_mrad_s,
            duration_ms,
            ..
        } => Some((
            ControlCommandCode::OrientationProbe,
            encode_i16(angular_mrad_s),
            0,
            0,
            0,
            Some(duration_ms),
        )),
        BrainstemCommand::ResetOdometry { .. } => {
            Some((ControlCommandCode::ResetOdometry, 0, 0, 0, 0, None))
        }
        BrainstemCommand::ZeroImuOrientation { .. } => {
            Some((ControlCommandCode::ZeroImuOrientation, 0, 0, 0, 0, None))
        }
        BrainstemCommand::ClearImuOrientation { .. } => {
            Some((ControlCommandCode::ClearImuOrientation, 0, 0, 0, 0, None))
        }
        BrainstemCommand::RestartCreate => {
            Some((ControlCommandCode::RestartCreate, 0, 0, 0, 0, None))
        }
        BrainstemCommand::ResetMotherbrain => {
            Some((ControlCommandCode::ResetMotherbrain, 0, 0, 0, 0, None))
        }
        BrainstemCommand::GetCapabilities => {
            Some((ControlCommandCode::GetCapabilities, 0, 0, 0, 0, None))
        }
        BrainstemCommand::GetEvents { since_seq } => {
            Some((ControlCommandCode::GetEvents, since_seq, 0, 0, 0, None))
        }
    }
}

fn decode_control_command(
    kind: u8,
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    duration_ms: Option<u32>,
    seq: u32,
) -> Option<BrainstemCommand> {
    if is_retired_control_code(kind) {
        return Some(BrainstemCommand::Unsupported { seq });
    }
    match kind {
        x if x == ControlCommandCode::Ping as u8 => Some(BrainstemCommand::Ping),
        x if x == ControlCommandCode::Arm as u8 => Some(BrainstemCommand::Arm),
        x if x == ControlCommandCode::Disarm as u8 => Some(BrainstemCommand::Disarm),
        x if x == ControlCommandCode::Stop as u8 => Some(BrainstemCommand::Stop),
        x if x == ControlCommandCode::EStop as u8 => Some(BrainstemCommand::EStop),
        x if x == ControlCommandCode::ClearEStop as u8 => Some(BrainstemCommand::ClearEStop),
        x if x == ControlCommandCode::Status as u8 => Some(BrainstemCommand::Status),
        x if x == ControlCommandCode::SetMode as u8 => Some(BrainstemCommand::SetMode(match a {
            1 => CreateOiMode::Passive,
            2 => CreateOiMode::Safe,
            3 => CreateOiMode::Full,
            _ => return None,
        })),
        x if x == ControlCommandCode::CmdVel as u8 => Some(BrainstemCommand::CmdVel {
            linear_mm_s: decode_i16(a),
            angular_mrad_s: decode_i16(b),
            ttl_ms: duration_ms?,
            seq,
        }),
        x if x == ControlCommandCode::DriveDirect as u8 => Some(BrainstemCommand::DriveDirect {
            left_mm_s: decode_i16(a),
            right_mm_s: decode_i16(b),
            ttl_ms: duration_ms?,
            seq,
        }),
        x if x == ControlCommandCode::DriveArc as u8 => Some(BrainstemCommand::DriveArc {
            velocity_mm_s: decode_i16(a),
            radius_mm: decode_i16(b),
            ttl_ms: duration_ms?,
            seq,
        }),
        x if x == ControlCommandCode::SongPlay as u8 => {
            Some(BrainstemCommand::SongPlay { id: a as u8 })
        }
        x if x == ControlCommandCode::SongDefine as u8 => {
            let tone_count = (b as u8).min(MAX_SONG_TONES as u8);
            Some(BrainstemCommand::SongDefine {
                id: a as u8,
                tones: load_pending_song_tones(tone_count),
                tone_count,
                seq,
            })
        }
        x if x == ControlCommandCode::Dock as u8 => Some(BrainstemCommand::Dock),
        x if x == ControlCommandCode::SetLights as u8 => Some(BrainstemCommand::SetLights {
            led_bits: a as u8,
            color: b as u8,
            intensity: c as u8,
        }),
        x if x == ControlCommandCode::HeartbeatStop as u8 => {
            Some(BrainstemCommand::HeartbeatStop {
                timeout_ms: duration_ms?,
                seq,
            })
        }
        x if x == ControlCommandCode::ClearSafetyLatch as u8 => {
            Some(BrainstemCommand::ClearSafetyLatch {
                kind: decode_safety_latch_kind(a as u8)?,
                seq,
            })
        }
        x if x == ControlCommandCode::CarefulMode as u8 => Some(BrainstemCommand::CarefulMode {
            ttl_ms: duration_ms?,
            seq,
        }),
        x if x == ControlCommandCode::EscapeMotion as u8 => Some(BrainstemCommand::EscapeMotion {
            kind: decode_safety_latch_kind(a as u8)?,
            hazard_generation: b,
            linear_mm_s: decode_i16(c),
            angular_mrad_s: decode_i16(d),
            ttl_ms: duration_ms?,
            seq,
        }),
        x if x == ControlCommandCode::RequestSensors as u8 => {
            Some(BrainstemCommand::RequestSensors {
                packet_id: a as u8,
                seq,
            })
        }
        x if x == ControlCommandCode::StreamSensors as u8 => {
            Some(BrainstemCommand::StreamSensors {
                enabled: a != 0,
                packet_id: b as u8,
                period_ms: duration_ms?,
                seq,
            })
        }
        x if x == ControlCommandCode::ClearMotionQueue as u8 => {
            Some(BrainstemCommand::ClearMotionQueue { seq })
        }
        x if x == ControlCommandCode::DefineChirp as u8 => {
            let tone_count = (b as u8).min(MAX_SONG_TONES as u8);
            Some(BrainstemCommand::DefineChirp {
                kind: decode_feedback_kind(a as u8)?,
                tones: load_pending_song_tones(tone_count),
                tone_count,
                seq,
            })
        }
        x if x == ControlCommandCode::PlayFeedback as u8 => Some(BrainstemCommand::PlayFeedback {
            kind: decode_feedback_kind(a as u8)?,
            seq,
        }),
        x if x == ControlCommandCode::SetAudioSilent as u8 => {
            Some(BrainstemCommand::SetAudioSilent {
                silent: a != 0,
                seq,
            })
        }
        x if x == ControlCommandCode::PowerState as u8 => Some(BrainstemCommand::PowerState {
            request: decode_power_request(a as u8)?,
            seq,
        }),
        x if x == ControlCommandCode::CalibrateTurn as u8 => {
            Some(BrainstemCommand::CalibrateTurn {
                angular_mrad_s: decode_i16(a),
                duration_ms: duration_ms?,
                seq,
            })
        }
        x if x == ControlCommandCode::OrientationProbe as u8 => {
            Some(BrainstemCommand::OrientationProbe {
                angular_mrad_s: decode_i16(a),
                duration_ms: duration_ms?,
                seq,
            })
        }
        x if x == ControlCommandCode::ResetOdometry as u8 => {
            Some(BrainstemCommand::ResetOdometry { seq })
        }
        x if x == ControlCommandCode::ZeroImuOrientation as u8 => {
            Some(BrainstemCommand::ZeroImuOrientation { seq })
        }
        x if x == ControlCommandCode::ClearImuOrientation as u8 => {
            Some(BrainstemCommand::ClearImuOrientation { seq })
        }
        x if x == ControlCommandCode::RestartCreate as u8 => Some(BrainstemCommand::RestartCreate),
        x if x == ControlCommandCode::GetCapabilities as u8 => {
            Some(BrainstemCommand::GetCapabilities)
        }
        x if x == ControlCommandCode::GetEvents as u8 => {
            Some(BrainstemCommand::GetEvents { since_seq: a })
        }
        _ => None,
    }
}

fn is_retired_control_code(kind: u8) -> bool {
    matches!(
        kind,
        x if x == ControlCommandCode::FaceBearing as u8
            || x == ControlCommandCode::TrackBearing as u8
            || x == ControlCommandCode::TurnBy as u8
            || x == ControlCommandCode::DriveFor as u8
            || x == ControlCommandCode::BumpEscape as u8
            || x == ControlCommandCode::HoldHeading as u8
            || x == ControlCommandCode::TurnToHeading as u8
            || x == ControlCommandCode::ArcFor as u8
            || x == ControlCommandCode::CreepUntil as u8
            || x == ControlCommandCode::ScanArc as u8
            || x == ControlCommandCode::DockAlign as u8
            || x == ControlCommandCode::WallFollow as u8
            || x == ControlCommandCode::WiggleAlign as u8
            || x == ControlCommandCode::Unstick as u8
            || x == ControlCommandCode::CliffGuard as u8
            || x == ControlCommandCode::SetSafetyPolicy as u8
    )
}

#[cfg(any(feature = "pico-w", feature = "rpi5"))]
fn command_seq(command: BrainstemCommand) -> u32 {
    match command {
        BrainstemCommand::CmdVel { seq, .. }
        | BrainstemCommand::DriveDirect { seq, .. }
        | BrainstemCommand::DriveArc { seq, .. }
        | BrainstemCommand::Unsupported { seq }
        | BrainstemCommand::ClearSafetyLatch { seq, .. }
        | BrainstemCommand::CarefulMode { seq, .. }
        | BrainstemCommand::EscapeMotion { seq, .. }
        | BrainstemCommand::SongDefine { seq, .. }
        | BrainstemCommand::RequestSensors { seq, .. }
        | BrainstemCommand::StreamSensors { seq, .. }
        | BrainstemCommand::ClearMotionQueue { seq, .. }
        | BrainstemCommand::DefineChirp { seq, .. }
        | BrainstemCommand::PlayFeedback { seq, .. }
        | BrainstemCommand::SetAudioSilent { seq, .. }
        | BrainstemCommand::PowerState { seq, .. }
        | BrainstemCommand::CalibrateTurn { seq, .. }
        | BrainstemCommand::OrientationProbe { seq, .. }
        | BrainstemCommand::ResetOdometry { seq, .. }
        | BrainstemCommand::ZeroImuOrientation { seq, .. }
        | BrainstemCommand::ClearImuOrientation { seq, .. }
        | BrainstemCommand::HeartbeatStop { seq, .. } => seq,
        _ => 0,
    }
}

#[cfg(any(feature = "pico-w", feature = "rpi5"))]
fn seq_is_current_or_newer(seq: u32, latest_seq: u32) -> bool {
    seq == latest_seq || seq.wrapping_sub(latest_seq) < u32::MAX / 2
}

fn encode_i16(value: i16) -> u32 {
    value as u16 as u32
}

fn decode_i16(value: u32) -> i16 {
    value as u16 as i16
}

#[cfg(any(feature = "pico-w", feature = "rpi5"))]
fn encode_safety_latch_kind(kind: SafetyLatchKind) -> u8 {
    match kind {
        SafetyLatchKind::Bump => 1,
        SafetyLatchKind::Cliff => 2,
        SafetyLatchKind::WheelDrop => 3,
        SafetyLatchKind::Heartbeat => 5,
        SafetyLatchKind::Tilt => 6,
        SafetyLatchKind::Impact => 7,
        SafetyLatchKind::Charging => 8,
    }
}

fn decode_safety_latch_kind(value: u8) -> Option<SafetyLatchKind> {
    match value {
        1 => Some(SafetyLatchKind::Bump),
        2 => Some(SafetyLatchKind::Cliff),
        3 => Some(SafetyLatchKind::WheelDrop),
        5 => Some(SafetyLatchKind::Heartbeat),
        6 => Some(SafetyLatchKind::Tilt),
        7 => Some(SafetyLatchKind::Impact),
        8 => Some(SafetyLatchKind::Charging),
        _ => None,
    }
}

pub fn validate_escape_motion(
    kind: SafetyLatchKind,
    hazard_generation: u32,
    linear_mm_s: i16,
    angular_mrad_s: i16,
    ttl_ms: u32,
) -> Result<(), CommandRejectReason> {
    const ESCAPE_TTL_MS: u32 = 250;
    const MAX_ESCAPE_LINEAR_MM_S: i16 = 120;
    const MAX_ESCAPE_ANGULAR_MRAD_S: i16 = 500;

    let (estop_latched, safety_tripped, motion_interlock, latched_kind) = session_safety_snapshot();
    let expected_kind = safety_latch_kind_to_event(kind);
    if !safety_tripped
        || latched_kind != Some(expected_kind)
        || hazard_generation == 0
        || hazard_generation != safety_hazard_generation()
    {
        return Err(CommandRejectReason::HazardMismatch);
    }
    if estop_latched || motion_interlock {
        return Err(CommandRejectReason::AbsoluteHazard);
    }
    if ttl_ms != ESCAPE_TTL_MS
        || linear_mm_s > 0
        || linear_mm_s < -MAX_ESCAPE_LINEAR_MM_S
        || angular_mrad_s.unsigned_abs() > MAX_ESCAPE_ANGULAR_MRAD_S as u16
        || (linear_mm_s == 0 && angular_mrad_s == 0)
    {
        return Err(CommandRejectReason::EscapeEnvelope);
    }

    let snapshot = snapshot(0);
    let flags = snapshot.create_sensor_flags;
    let bump_right = flags & (1 << 0) != 0;
    let bump_left = flags & (1 << 1) != 0;
    let wheel_drop = flags & (1 << 2) != 0;
    let cliff = flags & 0b1111_0000 != 0;
    let imu_ok = body::IMU_ENABLED && snapshot.imu_health == ImuHealthCode::Ok as u8;
    let tilt = imu_ok && snapshot.imu_tilt_magnitude_mrad as i16 >= body::IMU_TILT_STOP_MRAD;
    let impact = imu_ok && snapshot.imu_impact_score_mm_s2 >= body::IMU_IMPACT_STOP_MM_S2;
    let charging = charging_interlock_active(&snapshot);

    if wheel_drop || tilt || impact || charging {
        return Err(CommandRejectReason::AbsoluteHazard);
    }
    match kind {
        SafetyLatchKind::Bump => {
            if !bump_left && !bump_right {
                return Err(CommandRejectReason::HazardMismatch);
            }
            if cliff {
                return Err(CommandRejectReason::AbsoluteHazard);
            }
            let turns_toward_contact = (bump_left && !bump_right && angular_mrad_s > 0)
                || (bump_right && !bump_left && angular_mrad_s < 0)
                || (bump_left && bump_right && angular_mrad_s != 0);
            if turns_toward_contact {
                return Err(CommandRejectReason::EscapeEnvelope);
            }
        }
        SafetyLatchKind::Cliff => {
            if !cliff {
                return Err(CommandRejectReason::HazardMismatch);
            }
            if bump_left || bump_right {
                return Err(CommandRejectReason::AbsoluteHazard);
            }
            if linear_mm_s >= 0 || angular_mrad_s != 0 {
                return Err(CommandRejectReason::EscapeEnvelope);
            }
        }
        _ => return Err(CommandRejectReason::HazardMismatch),
    }
    Ok(())
}

fn safety_latch_kind_to_event(kind: SafetyLatchKind) -> SafetyEventKind {
    match kind {
        SafetyLatchKind::Bump => SafetyEventKind::Bump,
        SafetyLatchKind::Cliff => SafetyEventKind::Cliff,
        SafetyLatchKind::WheelDrop => SafetyEventKind::WheelDrop,
        SafetyLatchKind::Heartbeat => SafetyEventKind::Heartbeat,
        SafetyLatchKind::Tilt => SafetyEventKind::Tilt,
        SafetyLatchKind::Impact => SafetyEventKind::Impact,
        SafetyLatchKind::Charging => SafetyEventKind::Charging,
    }
}

#[cfg(any(feature = "pico-w", feature = "rpi5"))]
fn encode_feedback_kind(kind: FeedbackKind) -> u8 {
    match kind {
        FeedbackKind::Ok => 0,
        FeedbackKind::Error => 1,
        FeedbackKind::Armed => 2,
        FeedbackKind::LostTarget => 3,
        FeedbackKind::DockSeen => 4,
        FeedbackKind::Danger => 5,
    }
}

fn decode_feedback_kind(value: u8) -> Option<FeedbackKind> {
    match value {
        0 => Some(FeedbackKind::Ok),
        1 => Some(FeedbackKind::Error),
        2 => Some(FeedbackKind::Armed),
        3 => Some(FeedbackKind::LostTarget),
        4 => Some(FeedbackKind::DockSeen),
        5 => Some(FeedbackKind::Danger),
        _ => None,
    }
}

#[cfg(any(feature = "pico-w", feature = "rpi5"))]
fn encode_power_request(request: PowerStateRequest) -> u8 {
    match request {
        PowerStateRequest::Wake => 1,
        PowerStateRequest::Sleep => 2,
        PowerStateRequest::StartOi => 4,
        PowerStateRequest::DebugBaud19200 => 5,
        PowerStateRequest::DebugBaud57600 => 6,
        PowerStateRequest::DebugBaud115200 => 7,
    }
}

fn decode_power_request(value: u8) -> Option<PowerStateRequest> {
    match value {
        1 => Some(PowerStateRequest::Wake),
        2 => Some(PowerStateRequest::Sleep),
        4 => Some(PowerStateRequest::StartOi),
        5 => Some(PowerStateRequest::DebugBaud19200),
        6 => Some(PowerStateRequest::DebugBaud57600),
        7 => Some(PowerStateRequest::DebugBaud115200),
        _ => None,
    }
}

#[cfg(any(feature = "pico-w", feature = "rpi5"))]
fn store_pending_song_tones(tones: &[SongTone; MAX_SONG_TONES], tone_count: u8) {
    let tone_count = tone_count.min(MAX_SONG_TONES as u8) as usize;
    for i in 0..MAX_SONG_TONES {
        let value = if i < tone_count {
            pack_song_tone(tones[i])
        } else {
            0
        };
        PENDING_SONG_TONES[i].store(value, Ordering::Relaxed);
    }
}

fn load_pending_song_tones(tone_count: u8) -> [SongTone; MAX_SONG_TONES] {
    let mut tones = [SongTone::default(); MAX_SONG_TONES];
    let tone_count = tone_count.min(MAX_SONG_TONES as u8) as usize;
    for i in 0..tone_count {
        tones[i] = unpack_song_tone(PENDING_SONG_TONES[i].load(Ordering::Relaxed));
    }
    tones
}

#[cfg(any(feature = "pico-w", feature = "rpi5"))]
fn pack_song_tone(tone: SongTone) -> u32 {
    ((tone.note as u32) << 8) | tone.duration_64ths as u32
}

fn unpack_song_tone(value: u32) -> SongTone {
    SongTone {
        note: (value >> 8) as u8,
        duration_64ths: value as u8,
    }
}
