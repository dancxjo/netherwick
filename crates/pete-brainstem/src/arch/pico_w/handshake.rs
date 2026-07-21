enum CommandParseError {
    BadRequest,
    Busy(u32, &'static str),
}

fn handle_handshake_json<'a>(body: &str, buffer: &'a mut [u8], transport: u8) -> Option<&'a str> {
    let hello = match session::parse_json(body) {
        Ok(hello) => hello,
        Err(reason) => return render_handshake_reject(buffer, "", reason),
    };
    let mut device_id = heapless::String::<32>::new();
    let mut boot_id = heapless::String::<32>::new();
    let instance = BRAINSTEM_INSTANCE_ID.load(Ordering::Acquire);
    if instance == 0 {
        return render_handshake_reject(
            buffer,
            hello.handshake_nonce.as_str(),
            session::RejectReason::InvalidIdentity,
        );
    }
    let _ = write!(device_id, "pete-brainstem-{instance:04x}");
    let _ = write!(
        boot_id,
        "bsboot-{:08x}",
        BRAINSTEM_BOOT_ID.load(Ordering::Acquire)
    );
    let accepted = match session::validate(&hello, device_id.as_str(), boot_id.as_str()) {
        Ok(accepted) => accepted,
        Err(reason) => {
            return render_handshake_reject(buffer, hello.handshake_nonce.as_str(), reason);
        }
    };
    let session_hash = session::token_hash(accepted.session_id.as_str());
    let peer_hash = session::token_hash(hello.device_id.as_str());
    if hello.role == session::EndpointRole::Motherbrain
        && hello.session_purpose == session::SessionPurpose::Control
    {
        if transport != TransportKind::HardwareUart as u8
            && transport != TransportKind::UsbCdc as u8
            && !status::active_peer_matches(peer_hash)
        {
            return render_handshake_reject(
                buffer,
                hello.handshake_nonce.as_str(),
                session::RejectReason::InvalidIdentity,
            );
        }
        status::request_session_replace(
            accepted.generation,
            session_hash,
            peer_hash,
            session::token_hash(hello.boot_id.as_str()),
        );
        for _ in 0..250 {
            if status::session_replace_acked(accepted.generation) {
                break;
            }
            embassy_time::block_for(Duration::from_millis(1));
        }
        if !status::session_replace_acked(accepted.generation) {
            return None;
        }
        status::mark_transport_changed(transport);
    } else {
        let role = match hello.role {
            session::EndpointRole::Motherbrain => 1,
            session::EndpointRole::Forebrain => 2,
            session::EndpointRole::Operator => 3,
            session::EndpointRole::ServiceTool => 4,
            _ => 0,
        };
        let purpose = match hello.session_purpose {
            session::SessionPurpose::Control => 1,
            session::SessionPurpose::Diagnostic => 2,
        };
        status::register_diagnostic_session(
            session_hash,
            peer_hash,
            session::token_hash(hello.boot_id.as_str()),
            role,
            purpose,
            transport,
        );
    }
    render_handshake_welcome(
        buffer,
        &hello,
        &accepted,
        device_id.as_str(),
        boot_id.as_str(),
    )
}

fn render_handshake_reject<'a>(
    buffer: &'a mut [u8],
    nonce: &str,
    reason: session::RejectReason,
) -> Option<&'a str> {
    status::mark_session_rejected(reason.code());
    let mut response = heapless::String::<512>::new();
    write!(response, "{{\"kind\":\"reject\",\"echoed_handshake_nonce\":\"{nonce}\",\"reason_code\":\"{}\",\"message\":\"handshake rejected\",\"supported_protocol_major\":1,\"supported_minor_min\":0,\"supported_minor_max\":0}}", reason.as_str()).ok()?;
    copy_response(buffer, response.as_str())
}

fn render_handshake_welcome<'a>(
    buffer: &'a mut [u8],
    hello: &session::Hello,
    accepted: &session::AcceptedHello,
    device_id: &str,
    boot_id: &str,
) -> Option<&'a str> {
    let caps = capabilities::current();
    let snapshot = status::snapshot(Instant::now().as_millis() as u32);
    let (estop_latched, safety_tripped, motion_interlock_latched, _) =
        status::session_safety_snapshot();
    let mut response = heapless::String::<4096>::new();
    write!(response, "{{\"kind\":\"welcome\",\"role\":\"brainstem\",\"device_id\":\"{device_id}\",\"boot_id\":\"{boot_id}\",\"echoed_handshake_nonce\":\"{}\",\"session_id\":\"{}\",\"protocol_major\":1,\"protocol_minor\":{},\"supported_features\":[\"session_ids\",\"event_cursor\",\"heartbeat\",\"transport_failover\"],\"required_features\":[\"session_ids\"],\"heartbeat_min_ms\":250,\"heartbeat_max_ms\":2000,\"command_ttl_min_ms\":{},\"command_ttl_max_ms\":{},\"current_event_next_seq\":{},\"capability_contract\":{{\"body_kind\":\"{}\",\"drive\":\"{}\",", hello.handshake_nonce, accepted.session_id, accepted.negotiated_minor, caps.min_ttl_ms, caps.max_ttl_ms, snapshot.event_next_seq, caps.body_kind, caps.drive).ok()?;
    write_json_array(&mut response, "verbs", caps.verbs)?;
    write_json_array(&mut response, "sensors", caps.sensors)?;
    write_json_array(&mut response, "outputs", caps.outputs)?;
    write_json_array(&mut response, "safety", caps.safety)?;
    write_json_array(&mut response, "events", caps.events)?;
    let active_motion = snapshot.body_state == status::BodyState::Moving as u8;
    write!(response, "\"limits\":{{\"max_linear_mm_s\":{},\"max_angular_mrad_s\":{},\"min_ttl_ms\":{},\"max_ttl_ms\":{}}}}},\"software\":{{\"software_name\":\"{}\",", caps.max_linear_mm_s, caps.max_angular_mrad_s, caps.min_ttl_ms, caps.max_ttl_ms, caps.firmware_name).ok()?;
    crate::build_identity::write_json(&mut response, crate::build_identity::CURRENT).ok()?;
    write!(response, "}},\"safety_snapshot\":{{\"armed\":false,\"estop_latched\":{},\"safety_tripped\":{},\"motion_interlock_latched\":{},\"active_motion\":{},\"runtime_state\":\"{}\"}}}}", estop_latched, safety_tripped, motion_interlock_latched, active_motion, if active_motion { "moving" } else { "idle" }).ok()?;
    copy_response(buffer, response.as_str())
}

fn write_json_array<const N: usize>(
    response: &mut heapless::String<N>,
    key: &str,
    values: &[&str],
) -> Option<()> {
    write!(response, "\"{key}\":[").ok()?;
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            response.push(',').ok()?;
        }
        write!(response, "\"{value}\"").ok()?;
    }
    response.push_str("],").ok()?;
    Some(())
}

fn copy_response<'a>(buffer: &'a mut [u8], response: &str) -> Option<&'a str> {
    if response.len() > buffer.len() {
        return None;
    }
    buffer[..response.len()].copy_from_slice(response.as_bytes());
    core::str::from_utf8(&buffer[..response.len()]).ok()
}

fn render_network_diagnostics(buffer: &mut [u8], now_ms: u32) -> Option<&str> {
    let diagnostics = network_registry::diagnostics(now_ms);
    let mut response = heapless::String::<256>::new();
    write!(
        response,
        "{{\"active_leases\":{},\"registration_generation\":{},\"motherbrain_address\":",
        diagnostics.active_leases, diagnostics.registration_generation
    )
    .ok()?;
    if let Some(ip) = diagnostics.motherbrain_ip {
        write!(response, "\"{}.{}.{}.{}\"", ip[0], ip[1], ip[2], ip[3]).ok()?;
    } else {
        response.push_str("null").ok()?;
    }
    response.push('}').ok()?;
    copy_response(buffer, response.as_str())
}

fn render_session_diagnostics(buffer: &mut [u8], now_ms: u32) -> Option<&str> {
    let diagnostics = status::session_diagnostics(now_ms);
    let mut response = heapless::String::<512>::new();
    write!(response, "{{\"primary_session_generation\":{},\"diagnostic_sessions\":{},\"authority_generation\":{},\"authority_active\":{},\"authority_session_hash\":{},\"authority_owner_role\":\"{}\",\"authority_owner_device_hash\":{},\"authority_owner_boot_hash\":{},\"authority_lease_remaining_ms\":{},\"service_authority_active\":{}}}", diagnostics.primary_session_generation, diagnostics.diagnostic_sessions, diagnostics.authority_generation, diagnostics.authority_active, diagnostics.authority_session_hash, role_name(diagnostics.authority_owner_role), diagnostics.authority_owner_device_hash, diagnostics.authority_owner_boot_hash, diagnostics.authority_lease_remaining_ms, diagnostics.service_authority_active).ok()?;
    copy_response(buffer, response.as_str())
}
