fn handle_network_registration_json<'a>(body: &str, buffer: &'a mut [u8]) -> Option<&'a str> {
    let session_id = json_str(body, "session_id")?;
    let session_hash = session::token_hash(session_id);
    let identity = status::session_identity(session_hash)?;
    if !compact_session_valid(session_id)
        || identity.role != 1
        || identity.purpose != 1
        || json_str(body, "hostname") != Some("motherbrain")
    {
        return render_registration_reject(buffer, "invalid_session_or_identity");
    }
    let address = parse_ipv4(json_str(body, "address")?)?;
    let lease_identity = json_str(body, "lease_identity")?;
    let ttl = json_u32(body, "ttl_seconds")
        .unwrap_or(60)
        .clamp(1, DHCP_LEASE_SECONDS);
    let Some((ttl, generation)) = network_registry::register_motherbrain(
        lease_identity.as_bytes(),
        address,
        identity.peer_device_hash,
        identity.peer_boot_hash,
        ttl,
        Instant::now().as_millis() as u32,
    ) else {
        return render_registration_reject(buffer, "lease_mismatch");
    };
    let mut response = heapless::String::<512>::new();
    write!(response, "{{\"accepted\":true,\"session_id\":\"{session_id}\",\"fqdn\":\"motherbrain.pete.internal\",\"address\":\"{}.{}.{}.{}\",\"ttl_seconds\":{ttl},\"registration_generation\":{generation}}}", address[0], address[1], address[2], address[3]).ok()?;
    copy_response(buffer, response.as_str())
}

fn render_registration_reject<'a>(buffer: &'a mut [u8], reason: &str) -> Option<&'a str> {
    let mut response = heapless::String::<192>::new();
    write!(response, "{{\"accepted\":false,\"message\":\"{reason}\"}}").ok()?;
    copy_response(buffer, response.as_str())
}

fn handle_network_registration_compact<const N: usize>(
    line: &str,
    response: &mut heapless::String<N>,
) {
    response.clear();
    let (command, session_id) = compact_session(line);
    let mut fields = command.split_ascii_whitespace();
    let valid = fields.next() == Some("REGISTER_NETWORK_ENDPOINT");
    let seq = fields
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0);
    let _interface_id = fields.next();
    let address_text = fields.next();
    let hostname = fields.next();
    let lease_identity = fields.next();
    let ttl = fields
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(60);
    let Some(address) = address_text.and_then(parse_ipv4) else {
        let _ = writeln!(response, "ERR {seq} malformed_registration");
        return;
    };
    if !valid || hostname != Some("motherbrain") || !session_id.is_some_and(compact_session_valid) {
        let _ = writeln!(response, "ERR {seq} invalid_session_or_identity");
        return;
    }
    let session_identity = status::session_identity(session::token_hash(session_id.unwrap_or("")));
    let Some((ttl, generation)) = lease_identity.and_then(|lease_identity| {
        let peer = session_identity?;
        if peer.role != 1 || peer.purpose != 1 {
            return None;
        }
        network_registry::register_motherbrain(
            lease_identity.as_bytes(),
            address,
            peer.peer_device_hash,
            peer.peer_boot_hash,
            ttl,
            Instant::now().as_millis() as u32,
        )
    }) else {
        let _ = writeln!(response, "ERR {seq} lease_mismatch");
        return;
    };
    let _ = writeln!(
        response,
        "OK {seq} NETWORK_ENDPOINT_REGISTERED session_id={} fqdn=motherbrain.pete.internal address={}.{}.{}.{} ttl={} generation={}",
        session_id.unwrap_or(""),
        address[0],
        address[1],
        address[2],
        address[3],
        ttl,
        generation
    );
}

fn parse_ipv4(value: &str) -> Option<[u8; 4]> {
    let mut octets = [0u8; 4];
    let mut parts = value.split('.');
    for octet in &mut octets {
        *octet = parts.next()?.parse().ok()?;
    }
    if parts.next().is_some() {
        return None;
    }
    Some(octets)
}

fn handle_authority_json<'a>(body: &str, buffer: &'a mut [u8]) -> Option<&'a str> {
    let session_id = json_str(body, "session_id")?;
    let authority = json_str(body, "authority")?;
    let ttl_ms = json_u32(body, "ttl_ms").unwrap_or(2_000).clamp(250, 60_000);
    let session_hash = session::token_hash(session_id);
    if !authority_policy_allows(session_hash, authority, Instant::now().as_millis() as u32) {
        return render_registration_reject(buffer, "authority_policy_rejected");
    }
    let (lease_id, generation) = install_authority(session_hash, ttl_ms)?;
    let mut response = heapless::String::<512>::new();
    write!(response, "{{\"accepted\":true,\"type\":\"control_lease_granted\",\"lease_id\":\"{lease_id}\",\"session_id\":\"{session_id}\",\"owner_role\":\"{}\",\"authority\":\"{authority}\",\"ttl_ms\":{ttl_ms},\"generation\":{generation}}}", role_name(status::session_role(session_hash)?)).ok()?;
    copy_response(buffer, response.as_str())
}

fn handle_service_authority_json<'a>(body: &str, buffer: &'a mut [u8]) -> Option<&'a str> {
    let session_id = json_str(body, "session_id")?;
    let scope = json_str(body, "scope")?;
    let ttl_ms = json_u32(body, "ttl_ms").unwrap_or(2_000).clamp(250, 30_000);
    let session_hash = session::token_hash(session_id);
    let scope_code = service_scope_code(scope)?;
    if !service_policy_allows(session_hash) {
        return render_registration_reject(buffer, "service_policy_rejected");
    }
    let (lease_id, generation) = install_service_authority(session_hash, ttl_ms, scope_code)?;
    let mut response = heapless::String::<512>::new();
    write!(response, "{{\"accepted\":true,\"type\":\"service_lease_granted\",\"lease_id\":\"{lease_id}\",\"session_id\":\"{session_id}\",\"owner_role\":\"{}\",\"scope\":\"{scope}\",\"ttl_ms\":{ttl_ms},\"generation\":{generation}}}", role_name(status::session_role(session_hash)?)).ok()?;
    copy_response(buffer, response.as_str())
}

fn handle_service_authority_compact<const N: usize>(
    line: &str,
    response: &mut heapless::String<N>,
) {
    response.clear();
    let (command, session_id) = compact_session(line);
    let mut fields = command.split_ascii_whitespace();
    let valid = fields.next() == Some("ACQUIRE_SERVICE_LEASE");
    let seq = fields
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0);
    let scope = fields.next().unwrap_or("");
    let ttl_ms = fields
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(2_000)
        .clamp(250, 30_000);
    let Some(session_id) = session_id else {
        let _ = writeln!(response, "ERR {seq} invalid_session");
        return;
    };
    let session_hash = session::token_hash(session_id);
    let Some(scope_code) = service_scope_code(scope) else {
        let _ = writeln!(response, "ERR {seq} service_scope_denied");
        return;
    };
    if !valid || !service_policy_allows(session_hash) {
        let _ = writeln!(response, "ERR {seq} service_policy_rejected");
        return;
    }
    let Some((lease_id, generation)) = install_service_authority(session_hash, ttl_ms, scope_code)
    else {
        let _ = writeln!(response, "ERR {seq} authority_transition_timeout");
        return;
    };
    let _ = writeln!(
        response,
        "OK {seq} SERVICE_LEASE_GRANTED lease_id={lease_id} session_id={session_id} owner_role={} scope={scope} ttl_ms={ttl_ms} generation={generation}",
        role_name(status::session_role(session_hash).unwrap_or(0))
    );
}

fn service_policy_allows(session_hash: u32) -> bool {
    if !cfg!(feature = "service-mode") {
        return false;
    }
    let Some(identity) = status::session_identity(session_hash) else {
        return false;
    };
    let Some(role) = endpoint_role(identity.role) else {
        return false;
    };
    let purpose = if identity.purpose == 1 {
        session::SessionPurpose::Control
    } else {
        session::SessionPurpose::Diagnostic
    };
    let transport = match identity.transport {
        value if value == TransportKind::UsbCdc as u8 => TransportKind::UsbCdc,
        value if value == TransportKind::HardwareUart as u8 => TransportKind::HardwareUart,
        value if value == TransportKind::Http as u8 => TransportKind::Http,
        value if value == TransportKind::WebSocket as u8 => TransportKind::WebSocket,
        value if value == TransportKind::Udp as u8 => TransportKind::Udp,
        _ => return false,
    };
    pete_cockpit_protocol::role_can_request_service(role, purpose, transport)
}

fn install_service_authority(
    session_hash: u32,
    ttl_ms: u32,
    scope: u8,
) -> Option<(heapless::String<40>, u32)> {
    // Entering service authority uses the same synchronous stop/revoke barrier
    // as a controller transition, but installs a separate non-motion lease.
    let barrier_generation = AUTHORITY_GENERATION
        .fetch_add(1, Ordering::Relaxed)
        .wrapping_add(1)
        .max(1);
    status::request_authority_transition(barrier_generation, 0, 0, 0);
    for _ in 0..250 {
        if status::authority_transition_acked(barrier_generation) {
            break;
        }
        embassy_time::block_for(Duration::from_millis(1));
    }
    if !status::authority_transition_acked(barrier_generation) {
        return None;
    }
    let generation = SERVICE_GENERATION
        .fetch_add(1, Ordering::Relaxed)
        .wrapping_add(1)
        .max(1);
    let mut lease_id = heapless::String::<40>::new();
    write!(lease_id, "service-{generation:08x}-{session_hash:08x}").ok()?;
    status::install_service_authority(
        session_hash,
        session::token_hash(lease_id.as_str()),
        (Instant::now().as_millis() as u32).wrapping_add(ttl_ms),
        scope,
    );
    Some((lease_id, generation))
}

fn handle_authority_compact<const N: usize>(line: &str, response: &mut heapless::String<N>) {
    response.clear();
    let (command, session_id) = compact_session(line);
    let mut fields = command.split_ascii_whitespace();
    let valid = fields.next() == Some("ACQUIRE_CONTROL_LEASE");
    let seq = fields
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0);
    let authority = fields.next().unwrap_or("");
    let ttl_ms = fields
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(2_000)
        .clamp(250, 60_000);
    let Some(session_id) = session_id else {
        let _ = writeln!(response, "ERR {seq} invalid_session");
        return;
    };
    let session_hash = session::token_hash(session_id);
    if !valid
        || !authority_policy_allows(session_hash, authority, Instant::now().as_millis() as u32)
    {
        let _ = writeln!(response, "ERR {seq} authority_policy_rejected");
        return;
    }
    let Some((lease_id, generation)) = install_authority(session_hash, ttl_ms) else {
        let _ = writeln!(response, "ERR {seq} authority_transition_timeout");
        return;
    };
    let _ = writeln!(
        response,
        "OK {seq} CONTROL_LEASE_GRANTED lease_id={lease_id} session_id={session_id} owner_role={} authority={authority} ttl_ms={ttl_ms} generation={generation}",
        role_name(status::session_role(session_hash).unwrap_or(0))
    );
}

fn authority_policy_allows(session_hash: u32, authority: &str, now_ms: u32) -> bool {
    let Some(identity) = status::session_identity(session_hash) else {
        return false;
    };
    let Some(role) = endpoint_role(identity.role) else {
        return false;
    };
    let purpose = if identity.purpose == 1 {
        session::SessionPurpose::Control
    } else {
        session::SessionPurpose::Diagnostic
    };
    let requested = match authority {
        "motherbrain" => pete_cockpit_protocol::ControlAuthority::Motherbrain,
        "operator_debug" => pete_cockpit_protocol::ControlAuthority::OperatorDebug,
        "forebrain_recovery" => pete_cockpit_protocol::ControlAuthority::ForebrainRecovery,
        _ => return false,
    };
    if !pete_cockpit_protocol::role_can_request_control(role, purpose, requested) {
        return false;
    }
    match requested {
        pete_cockpit_protocol::ControlAuthority::Motherbrain => true,
        pete_cockpit_protocol::ControlAuthority::OperatorDebug => cfg!(feature = "operator-debug"),
        pete_cockpit_protocol::ControlAuthority::ForebrainRecovery => {
            status::authority_expired(now_ms)
                && option_env!("PETE_RECOVERY_FOREBRAIN_ID").is_some_and(|device_id| {
                    status::session_peer_matches(session_hash, session::token_hash(device_id))
                })
        }
    }
}

fn endpoint_role(role: u8) -> Option<session::EndpointRole> {
    match role {
        1 => Some(session::EndpointRole::Motherbrain),
        2 => Some(session::EndpointRole::Forebrain),
        3 => Some(session::EndpointRole::Operator),
        4 => Some(session::EndpointRole::ServiceTool),
        _ => None,
    }
}

fn install_authority(session_hash: u32, ttl_ms: u32) -> Option<(heapless::String<40>, u32)> {
    let generation = AUTHORITY_GENERATION
        .fetch_add(1, Ordering::Relaxed)
        .wrapping_add(1)
        .max(1);
    let mut lease_id = heapless::String::<40>::new();
    write!(lease_id, "lease-{generation:08x}-{session_hash:08x}").ok()?;
    let now = Instant::now().as_millis() as u32;
    status::request_authority_transition(
        generation,
        session::token_hash(lease_id.as_str()),
        session_hash,
        now.wrapping_add(ttl_ms),
    );
    for _ in 0..250 {
        if status::authority_transition_acked(generation) {
            return Some((lease_id, generation));
        }
        embassy_time::block_for(Duration::from_millis(1));
    }
    None
}

fn service_scope_code(scope: &str) -> Option<u8> {
    match scope {
        "bootsel" => Some(1),
        "restart_create" => Some(3),
        "reset_motherbrain" => Some(4),
        _ => None,
    }
}

fn role_name(role: u8) -> &'static str {
    match role {
        1 => "motherbrain",
        2 => "forebrain",
        3 => "operator",
        4 => "service_tool",
        _ => "unknown",
    }
}

fn parse_u32(value: Option<&str>) -> Option<u32> {
    value?.parse().ok()
}

fn parse_i16(value: Option<&str>) -> Option<i16> {
    value?.parse().ok()
}

fn parse_bool(value: Option<&str>) -> Option<bool> {
    match value? {
        "1" | "true" | "TRUE" | "clear" | "CLEAR" | "on" | "ON" | "enable" | "ENABLE" => Some(true),
        "0" | "false" | "FALSE" | "trip" | "TRIP" | "off" | "OFF" | "disable" | "DISABLE" => {
            Some(false)
        }
        _ => None,
    }
}
