use super::*;

#[test]
fn simulator_rejects_removed_brainstem_convenience_verbs() {
    let mut sim = SimCockpit::new();
    assert!(!sim
        .get_capabilities()
        .unwrap()
        .verbs
        .iter()
        .any(|verb| verb == "drive_for" || verb == "set_safety_policy"));
    assert!(matches!(
        sim.execute(CockpitRequest::DriveFor {
            distance_mm: 100,
            velocity_mm_s: 50,
            timeout_ms: 1_000,
        }),
        Err(CockpitError::Rejected { reason, .. }) if reason == "unsupported"
    ));
}
use std::collections::BTreeSet;
use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::thread;

struct StopRejectingCockpit {
    inner: SimCockpit,
    reject_stop: bool,
    disarm_requests: usize,
}

struct BusyOnceCockpit {
    inner: SimCockpit,
    busy_remaining: usize,
    attempts: usize,
    heartbeat_attempts: usize,
    cmd_vel_attempts: usize,
    last_heartbeat_timeout_ms: Option<u32>,
    last_bump_escape: Option<(EscapeDirection, i16, i16)>,
}

struct StaleLeaseOnceCockpit {
    inner: SimCockpit,
    invalid_remaining: usize,
    attempts: usize,
}

impl Cockpit for BusyOnceCockpit {
    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        self.inner.execute(request)
    }

    fn handshake(&mut self, hello: HandshakeHello) -> Result<HandshakeOutcome> {
        self.inner.handshake(hello)
    }

    fn execute_in_session(
        &mut self,
        session: &CockpitSession,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        self.inner.execute_in_session(session, request)
    }

    fn execute_with_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ControlLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        self.attempts += 1;
        match &request {
            CockpitRequest::HeartbeatStop { timeout_ms } => {
                self.heartbeat_attempts += 1;
                self.last_heartbeat_timeout_ms = Some(*timeout_ms);
            }
            CockpitRequest::CmdVel { .. } => self.cmd_vel_attempts += 1,
            CockpitRequest::BumpEscape {
                direction,
                backoff_mm_s,
                turn_angular_mrad_s,
            } => {
                self.last_bump_escape = Some((*direction, *backoff_mm_s, *turn_angular_mrad_s));
            }
            _ => {}
        }
        if self.busy_remaining > 0 {
            self.busy_remaining -= 1;
            return Err(CockpitError::Rejected {
                command_id: self.attempts as u32,
                reason: "busy".into(),
            });
        }
        self.inner.execute_with_lease(session, lease, request)
    }

    fn execute_with_service_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ServiceLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        self.inner
            .execute_with_service_lease(session, lease, request)
    }
}

impl Cockpit for StaleLeaseOnceCockpit {
    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        self.inner.execute(request)
    }

    fn handshake(&mut self, hello: HandshakeHello) -> Result<HandshakeOutcome> {
        self.inner.handshake(hello)
    }

    fn execute_in_session(
        &mut self,
        session: &CockpitSession,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        self.inner.execute_in_session(session, request)
    }

    fn execute_with_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ControlLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        self.attempts += 1;
        if self.invalid_remaining > 0 {
            self.invalid_remaining -= 1;
            return Err(CockpitError::Rejected {
                command_id: self.attempts as u32,
                reason: "invalid_control_lease".into(),
            });
        }
        self.inner.execute_with_lease(session, lease, request)
    }

    fn execute_with_service_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ServiceLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        self.inner
            .execute_with_service_lease(session, lease, request)
    }
}

impl Cockpit for StopRejectingCockpit {
    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        self.inner.execute(request)
    }

    fn handshake(&mut self, hello: HandshakeHello) -> Result<HandshakeOutcome> {
        self.inner.handshake(hello)
    }

    fn execute_in_session(
        &mut self,
        session: &CockpitSession,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        if matches!(&request, CockpitRequest::Stop) && self.reject_stop {
            return Ok(CockpitResponse::Rejected {
                message: "stop not acknowledged".into(),
            });
        }
        if matches!(&request, CockpitRequest::Disarm) {
            self.disarm_requests += 1;
        }
        self.inner.execute_in_session(session, request)
    }

    fn execute_with_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ControlLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        self.inner.execute_with_lease(session, lease, request)
    }

    fn execute_with_service_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ServiceLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        self.inner
            .execute_with_service_lease(session, lease, request)
    }
}

fn hello() -> HandshakeHello {
    HandshakeHello::motherbrain("pete-motherbrain-test")
}

fn conformance_caps() -> CockpitCapabilities {
    SimCockpit::new().get_capabilities().unwrap()
}

fn conformance_welcome(hello: &HandshakeHello) -> HandshakeResponse {
    negotiate(
        hello,
        "pete-brainstem-wire-test",
        "bsboot-wire-test",
        conformance_caps(),
        SafetySnapshot {
            armed: false,
            estop_latched: false,
            safety_tripped: false,
            active_motion: false,
            runtime_state: "idle".into(),
        },
        1,
        SoftwareInfo {
            software_name: "wire-test".into(),
            software_version: "1".into(),
            build_id: "test".into(),
        },
        1,
    )
}

fn compact_emulator_response(line: &str) -> String {
    if line.starts_with("HELLO ") {
        let hello = decode_compact_hello(line).unwrap();
        return encode_compact_response(&conformance_welcome(&hello)).unwrap();
    }
    let seq = line
        .split_ascii_whitespace()
        .nth(1)
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0);
    if line.starts_with("REGISTER_NETWORK_ENDPOINT ") {
        return format!("OK {seq} NETWORK_ENDPOINT_REGISTERED session_id=sess-wire fqdn=motherbrain.pete.internal address=192.168.4.2 ttl=60 generation=1\n");
    }
    if line.starts_with("ACQUIRE_CONTROL_LEASE ") {
        return format!("OK {seq} CONTROL_LEASE_GRANTED lease_id=lease-wire session_id=sess-wire owner_role=motherbrain authority=motherbrain ttl_ms=1000 generation=1\n");
    }
    if line.starts_with("BOOTSEL ") {
        return format!("ERR {seq} service_authorization_required\n");
    }
    if line.starts_with("CMD_VEL ") {
        return if line.contains(" session_id=") && line.contains(" lease_id=") {
            format!("OK {seq}\n")
        } else if line.contains(" session_id=") {
            format!("ERR {seq} invalid_control_lease\n")
        } else {
            format!("ERR {seq} invalid_session\n")
        };
    }
    format!("ERR {seq} unsupported\n")
}

fn json_emulator_response(body: &str) -> String {
    let value: serde_json::Value = serde_json::from_str(body).unwrap();
    if value.get("kind").and_then(serde_json::Value::as_str) == Some("hello")
        || value.get("handshake_nonce").is_some() && value.get("command_id").is_none()
    {
        let hello: HandshakeHello = serde_json::from_value(value).unwrap();
        return serde_json::to_string(&conformance_welcome(&hello)).unwrap();
    }
    let kind = value
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    match kind {
        "register_network_endpoint" => serde_json::json!({
            "accepted": true,
            "session_id": "sess-wire",
            "fqdn": "motherbrain.pete.internal",
            "address": "192.168.4.2",
            "ttl_seconds": 60,
            "registration_generation": 1
        })
        .to_string(),
        "acquire_control_lease" => serde_json::json!({
            "accepted": true,
            "type": "control_lease_granted",
            "lease_id": "lease-wire",
            "session_id": "sess-wire",
            "owner_role": "motherbrain",
            "authority": "motherbrain",
            "ttl_ms": 1000,
            "generation": 1
        })
        .to_string(),
        "bootsel" => serde_json::json!({
            "accepted": false,
            "message": "service_authorization_required"
        })
        .to_string(),
        "cmd_vel" => {
            let accepted = value.get("session_id").is_some() && value.get("lease_id").is_some();
            if accepted {
                serde_json::json!({"accepted": true}).to_string()
            } else {
                serde_json::json!({
                    "accepted": false,
                    "message": if value.get("session_id").is_some() {
                        "invalid_control_lease"
                    } else {
                        "invalid_session"
                    }
                })
                .to_string()
            }
        }
        _ => serde_json::json!({"accepted": false, "message": "unsupported"}).to_string(),
    }
}

fn run_physical_connector_conformance<C: Cockpit>(connector: &mut C) {
    let hello = hello();
    let outcome = connector.handshake(hello).unwrap();
    assert_eq!(outcome.session.peer_device_id, "pete-brainstem-wire-test");
    let motion = CockpitRequest::CmdVel {
        linear_mm_s: 40,
        angular_mrad_s: 0,
        ttl_ms: 300,
    };
    assert!(connector.execute(motion.clone()).is_err());
    assert!(connector
        .execute_in_session(&outcome.session, motion.clone())
        .is_err());
    let registration = connector
        .execute_in_session(
            &outcome.session,
            CockpitRequest::RegisterNetworkEndpoint(RegisterNetworkEndpoint {
                interface_id: "wlan1".into(),
                address_family: AddressFamily::Ipv4,
                address: "192.168.4.2".into(),
                hostname: "motherbrain".into(),
                lease_identity: "010203".into(),
                ttl_seconds: 60,
            }),
        )
        .unwrap();
    assert!(matches!(
        registration,
        CockpitResponse::NetworkEndpointRegistered(_)
    ));
    let lease = match connector
        .execute_in_session(
            &outcome.session,
            CockpitRequest::AcquireControlLease {
                authority: ControlAuthority::Motherbrain,
                ttl_ms: 1_000,
            },
        )
        .unwrap()
    {
        CockpitResponse::ControlLeaseGranted(lease) => lease,
        other => panic!("{other:?}"),
    };
    assert!(connector
        .execute_with_lease(&outcome.session, &lease, motion)
        .is_ok());
    assert!(connector
        .execute_with_lease(&outcome.session, &lease, CockpitRequest::Bootsel)
        .is_err());
}

#[test]
fn uart_connector_runs_shared_session_conformance() {
    let (host, mut device) = serialport::TTYPort::pair().unwrap();
    let server = thread::spawn(move || {
        let mut reader = BufReader::new(device.try_clone().unwrap());
        for _ in 0..7 {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            device
                .write_all(compact_emulator_response(&line).as_bytes())
                .unwrap();
            device.flush().unwrap();
        }
    });
    let mut connector = UartCockpit::from_port(Box::new(host));
    run_physical_connector_conformance(&mut connector);
    server.join().unwrap();
}

#[test]
fn udp_connector_runs_shared_session_conformance() {
    let server_socket = UdpSocket::bind("127.0.0.1:0").unwrap();
    let address = server_socket.local_addr().unwrap();
    let server = thread::spawn(move || {
        let mut buffer = [0u8; MAX_COMPACT_HANDSHAKE_FRAME_LEN];
        for _ in 0..7 {
            let (len, peer) = server_socket.recv_from(&mut buffer).unwrap();
            let line = std::str::from_utf8(&buffer[..len]).unwrap();
            server_socket
                .send_to(compact_emulator_response(line).as_bytes(), peer)
                .unwrap();
        }
    });
    let mut connector = UdpCockpit::connect(address).unwrap();
    run_physical_connector_conformance(&mut connector);
    server.join().unwrap();
}

fn read_http_test_body(stream: &mut TcpStream) -> String {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut bytes = Vec::new();
    let header_end = loop {
        let mut byte = [0u8; 1];
        stream.read_exact(&mut byte).unwrap();
        bytes.push(byte[0]);
        if bytes.ends_with(b"\r\n\r\n") {
            break bytes.len();
        }
    };
    let header = std::str::from_utf8(&bytes[..header_end]).unwrap();
    let content_length = header
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap();
    let mut body = vec![0u8; content_length];
    stream.read_exact(&mut body).unwrap();
    String::from_utf8(body).unwrap()
}

#[test]
fn http_connector_runs_shared_session_conformance() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        for _ in 0..7 {
            let (mut stream, _) = listener.accept().unwrap();
            let body = read_http_test_body(&mut stream);
            let response = json_emulator_response(&body);
            write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    response.len(),
                    response
                )
                .unwrap();
            stream.flush().unwrap();
        }
    });
    let mut connector = HttpCockpit::connect(address.to_string());
    run_physical_connector_conformance(&mut connector);
    server.join().unwrap();
}

#[test]
fn websocket_connector_runs_shared_session_conformance() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut socket = tungstenite::accept(stream).unwrap();
        for _ in 0..7 {
            let message = socket.read().unwrap();
            let body = message.into_text().unwrap();
            socket
                .send(Message::Text(json_emulator_response(&body).into()))
                .unwrap();
        }
    });
    let mut connector = WebSocketCockpit::connect_url(&format!("ws://{address}/control")).unwrap();
    run_physical_connector_conformance(&mut connector);
    server.join().unwrap();
}

#[test]
fn process_boot_identity_is_stable_while_handshake_nonces_are_fresh() {
    let first = hello();
    let second = hello();
    assert_eq!(first.boot_id, second.boot_id);
    assert_ne!(first.handshake_nonce, second.handshake_nonce);
    assert_eq!(first.boot_id, first.new_attempt().boot_id);
    assert_ne!(first.handshake_nonce, first.new_attempt().handshake_nonce);
}

#[test]
fn handshake_conformance_success_minor_and_json_compact_parity() {
    let mut sim = SimCockpit::new();
    let hello = hello();
    let outcome = sim.handshake(hello.clone()).unwrap();
    assert_eq!(outcome.welcome.protocol_minor, PROTOCOL_MINOR_MAX);
    assert_eq!(
        outcome.welcome.echoed_handshake_nonce,
        hello.handshake_nonce
    );
    assert!(!outcome.welcome.safety_snapshot.armed);
    assert!(!outcome.welcome.safety_snapshot.active_motion);
    assert_eq!(
        outcome.contract.capabilities(),
        &outcome.welcome.capability_contract
    );

    let hello_line = encode_compact_hello(&hello).unwrap();
    assert_eq!(decode_compact_hello(&hello_line).unwrap(), hello);
    let response = HandshakeResponse::Welcome(outcome.welcome.clone());
    let line = encode_compact_response(&response).unwrap();
    assert_eq!(decode_compact_response(&line).unwrap(), response);
    let json = serde_json::to_string(&response).unwrap();
    assert_eq!(
        serde_json::from_str::<HandshakeResponse>(&json).unwrap(),
        response
    );
}

#[test]
fn handshake_structured_rejections_and_stale_nonce() {
    let caps = SimCockpit::new().get_capabilities().unwrap();
    let safety = SafetySnapshot {
        armed: false,
        estop_latched: false,
        safety_tripped: false,
        active_motion: false,
        runtime_state: "idle".into(),
    };
    let software = SoftwareInfo {
        software_name: "test".into(),
        software_version: "1".into(),
        build_id: "x".into(),
    };
    let mut major = hello();
    major.protocol_major += 1;
    assert!(matches!(
        negotiate(
            &major,
            "brain",
            "boot",
            caps.clone(),
            safety.clone(),
            1,
            software.clone(),
            1
        ),
        HandshakeResponse::Reject(HandshakeReject {
            reason_code: HandshakeRejectReason::ProtocolMajorMismatch,
            ..
        })
    ));
    let mut wrong_role = hello();
    wrong_role.role = EndpointRole::Brainstem;
    assert!(matches!(
        negotiate(
            &wrong_role,
            "brain",
            "boot",
            caps.clone(),
            safety.clone(),
            1,
            software.clone(),
            1
        ),
        HandshakeResponse::Reject(HandshakeReject {
            reason_code: HandshakeRejectReason::WrongRole,
            ..
        })
    ));
    let mut feature = hello();
    feature
        .required_features
        .push(HandshakeFeature::Unknown("future_required".into()));
    assert!(matches!(
        negotiate(&feature, "brain", "boot", caps, safety, 1, software, 1),
        HandshakeResponse::Reject(HandshakeReject {
            reason_code: HandshakeRejectReason::MissingRequiredFeature,
            ..
        })
    ));

    let mut sim = SimCockpit::new();
    let hello = hello();
    let mut response = match negotiate(
        &hello,
        "brain",
        "boot",
        sim.get_capabilities().unwrap(),
        SafetySnapshot {
            armed: false,
            estop_latched: false,
            safety_tripped: false,
            active_motion: false,
            runtime_state: "idle".into(),
        },
        1,
        SoftwareInfo {
            software_name: "x".into(),
            software_version: "1".into(),
            build_id: "x".into(),
        },
        1,
    ) {
        HandshakeResponse::Welcome(w) => w,
        _ => unreachable!(),
    };
    response.echoed_handshake_nonce = "old-nonce".into();
    assert!(matches!(
        HandshakeOutcome::validate(&hello, HandshakeResponse::Welcome(response)),
        Err(CockpitError::StaleHandshake { .. })
    ));
}

#[test]
fn compact_handshake_rejects_malformed_and_oversized_frames() {
    assert!(decode_compact_hello("HELLO !!!\n").is_err());
    let oversized = format!("HELLO {}\n", "a".repeat(MAX_COMPACT_HANDSHAKE_FRAME_LEN));
    assert!(matches!(
        decode_compact_hello(&oversized),
        Err(CockpitError::FrameTooLarge { .. })
    ));
}

#[test]
fn compact_line_decoder_handles_fragmentation_and_multiple_frames() {
    let mut decoder = CompactLineDecoder::new(32);
    assert!(decoder.push(b"HEL").unwrap().is_empty());
    assert_eq!(
        decoder.push(b"LO one\nPING two\npart").unwrap(),
        vec!["HELLO one", "PING two"]
    );
    assert_eq!(decoder.push(b"ial\n").unwrap(), vec!["partial"]);
    assert!(matches!(
        CompactLineDecoder::new(2).push(b"abc"),
        Err(CockpitError::FrameTooLarge { .. })
    ));
}

#[test]
fn session_replacement_stops_disarms_and_rejects_old_session_without_clearing_estop() {
    let mut sim = SimCockpit::new();
    let first = sim.handshake(hello()).unwrap();
    sim.execute_in_session(
        &first.session,
        CockpitRequest::AcquireControlLease {
            authority: ControlAuthority::Motherbrain,
            ttl_ms: 1_000,
        },
    )
    .unwrap();
    sim.execute_in_session(&first.session, CockpitRequest::Arm)
        .unwrap();
    sim.execute_in_session(
        &first.session,
        CockpitRequest::CmdVel {
            linear_mm_s: 100,
            angular_mrad_s: 0,
            ttl_ms: 1_000,
        },
    )
    .unwrap();
    sim.execute_in_session(&first.session, CockpitRequest::EStop)
        .unwrap();

    let second = sim.handshake(hello()).unwrap();
    let cockpit_status = sim.get_status().unwrap();
    let status = cockpit_status.summary();
    assert_eq!(status.armed, Some(false));
    assert!(cockpit_status.raw.contains("active_cmd_vel=false"));
    assert_eq!(status.estop_latched, Some(true));
    assert_ne!(first.session.session_id, second.session.session_id);
    assert!(matches!(
        sim.execute_in_session(&first.session, CockpitRequest::Arm),
        Err(CockpitError::InvalidSession { .. })
    ));
}

#[test]
fn reconnect_classifies_transport_reboot_and_replacement() {
    let mut sim = SimCockpit::new().with_identity("brain-a", "boot-a");
    let first = sim.handshake(hello()).unwrap();
    let reconnect = sim
        .handshake(hello())
        .unwrap()
        .classify_against(Some(&first.session));
    assert_eq!(
        reconnect.classification,
        ReconnectClassification::TransportReconnect
    );
    let mut rebooted = SimCockpit::new().with_identity("brain-a", "boot-b");
    let reboot = rebooted
        .handshake(hello())
        .unwrap()
        .classify_against(Some(&first.session));
    assert_eq!(
        reboot.classification,
        ReconnectClassification::BrainstemReboot
    );
    let mut replacement = SimCockpit::new().with_identity("brain-b", "boot-c");
    let replacement = replacement
        .handshake(hello())
        .unwrap()
        .classify_against(Some(&first.session));
    assert_eq!(
        replacement.classification,
        ReconnectClassification::ReplacementBrainstem
    );
}

#[test]
fn duplicate_hello_is_idempotent() {
    let mut sim = SimCockpit::new();
    let hello = hello();
    let first = sim.handshake(hello.clone()).unwrap();
    let duplicate = sim.handshake(hello).unwrap();
    assert_eq!(first.session.session_id, duplicate.session.session_id);
}

#[test]
fn failover_same_device_creates_fresh_single_authority_session() {
    let primary: Box<dyn Cockpit> = Box::new(SimCockpit::new().with_identity("brain-a", "boot-a"));
    let ready = establish_session(primary, hello(), None).unwrap();
    let old_session = ready.session().clone();
    let mut failover = FailoverCockpit::new(ready);
    let backup: Box<dyn Cockpit> = Box::new(SimCockpit::new().with_identity("brain-a", "boot-a"));
    failover
        .failover(backup, hello(), ReplacementPolicy::Reject)
        .unwrap();
    assert_ne!(
        old_session.session_id,
        failover.active().session().session_id
    );
    assert_eq!(
        failover.active().outcome().classification,
        ReconnectClassification::TransportReconnect
    );
    failover
        .active_mut()
        .acquire_control(ControlAuthority::Motherbrain, 1_000)
        .unwrap();
    assert!(failover.active_mut().execute(CockpitRequest::Arm).is_ok());
}

#[test]
fn operator_debug_and_forebrain_recovery_transitions_stop_and_revoke() {
    let mut sim = SimCockpit::new().with_takeover_policy(true, Some("forebrain-alpha".into()));
    let mother = sim.handshake(hello()).unwrap();
    let mother_lease = match sim
        .execute_in_session(
            &mother.session,
            CockpitRequest::AcquireControlLease {
                authority: ControlAuthority::Motherbrain,
                ttl_ms: 500,
            },
        )
        .unwrap()
    {
        CockpitResponse::ControlLeaseGranted(lease) => lease,
        other => panic!("{other:?}"),
    };
    sim.execute_with_lease(&mother.session, &mother_lease, CockpitRequest::Arm)
        .unwrap();
    sim.execute_with_lease(
        &mother.session,
        &mother_lease,
        CockpitRequest::CmdVel {
            linear_mm_s: 100,
            angular_mrad_s: 0,
            ttl_ms: 1_000,
        },
    )
    .unwrap();
    assert!(sim
        .execute_with_lease(
            &mother.session,
            &mother_lease,
            CockpitRequest::CarefulMode { ttl_ms: 500 },
        )
        .is_err());

    let mut operator_hello = HandshakeHello::motherbrain("operator-laptop");
    operator_hello.role = EndpointRole::Operator;
    operator_hello.session_purpose = SessionPurpose::Control;
    let operator = sim.handshake(operator_hello).unwrap();
    assert!(sim
        .get_status()
        .unwrap()
        .raw
        .contains("active_cmd_vel=true"));
    let debug_lease = match sim
        .execute_in_session(
            &operator.session,
            CockpitRequest::AcquireControlLease {
                authority: ControlAuthority::OperatorDebug,
                ttl_ms: 500,
            },
        )
        .unwrap()
    {
        CockpitResponse::ControlLeaseGranted(lease) => lease,
        other => panic!("{other:?}"),
    };
    assert!(sim
        .get_status()
        .unwrap()
        .raw
        .contains("active_cmd_vel=false"));
    assert!(sim
        .execute_with_lease(
            &operator.session,
            &debug_lease,
            CockpitRequest::CarefulMode { ttl_ms: 500 },
        )
        .is_ok());
    assert!(sim
        .execute_with_lease(&mother.session, &mother_lease, CockpitRequest::Arm)
        .is_err());

    let mut forebrain_hello = HandshakeHello::forebrain("forebrain-alpha");
    forebrain_hello.session_purpose = SessionPurpose::Control;
    forebrain_hello.handshake_nonce = "forebrain-recovery".into();
    let forebrain = sim.handshake(forebrain_hello).unwrap();
    assert!(sim
        .execute_in_session(
            &forebrain.session,
            CockpitRequest::AcquireControlLease {
                authority: ControlAuthority::ForebrainRecovery,
                ttl_ms: 500
            }
        )
        .is_err());
    sim.advance_ms(debug_lease.ttl_ms);
    let recovery = sim
        .execute_in_session(
            &forebrain.session,
            CockpitRequest::AcquireControlLease {
                authority: ControlAuthority::ForebrainRecovery,
                ttl_ms: 500,
            },
        )
        .unwrap();
    assert!(matches!(recovery, CockpitResponse::ControlLeaseGranted(_)));
    assert!(sim
        .execute_with_lease(&operator.session, &debug_lease, CockpitRequest::Arm)
        .is_err());
}

#[test]
fn takeover_roles_are_default_deny() {
    let mut sim = SimCockpit::new();
    let operator = sim
        .handshake(HandshakeHello::operator("operator-laptop"))
        .unwrap();
    assert!(sim
        .execute_in_session(
            &operator.session,
            CockpitRequest::AcquireControlLease {
                authority: ControlAuthority::OperatorDebug,
                ttl_ms: 500,
            }
        )
        .is_err());
    let forebrain = sim
        .handshake(HandshakeHello::forebrain("forebrain-alpha"))
        .unwrap();
    assert!(sim
        .execute_in_session(
            &forebrain.session,
            CockpitRequest::AcquireControlLease {
                authority: ControlAuthority::ForebrainRecovery,
                ttl_ms: 500,
            }
        )
        .is_err());
}

#[test]
fn authorization_classes_separate_emergency_control_and_service() {
    assert_eq!(
        CockpitRequest::GetStatus.authorization_class(),
        AuthorizationClass::ReadOnly
    );
    assert_eq!(
        CockpitRequest::EStop.authorization_class(),
        AuthorizationClass::Emergency
    );
    assert_eq!(
        CockpitRequest::Arm.authorization_class(),
        AuthorizationClass::ControlLease
    );
    assert_eq!(
        CockpitRequest::StreamSensors {
            enabled: true,
            packet_id: 0,
            period_ms: 250,
        }
        .authorization_class(),
        AuthorizationClass::Session
    );
    for request in [
        CockpitRequest::Bootsel,
        CockpitRequest::RestartCreate,
        CockpitRequest::ResetMotherbrain,
    ] {
        assert_eq!(
            request.authorization_class(),
            AuthorizationClass::ServiceLease
        );
    }
    let mut sim = SimCockpit::new();
    assert!(Cockpit::arm(&mut sim).is_err());
    assert!(Cockpit::cmd_vel(&mut sim, 10, 0, 100).is_err());
    assert!(Cockpit::stop(&mut sim).is_ok());
}

#[test]
fn diagnostic_sessions_cannot_acquire_motion_authority_or_service_access() {
    let mut sim = SimCockpit::new().with_takeover_policy(true, Some("forebrain-alpha".into()));
    let operator = sim
        .handshake(HandshakeHello::operator("operator-laptop"))
        .unwrap();
    assert!(sim
        .execute_in_session(
            &operator.session,
            CockpitRequest::AcquireControlLease {
                authority: ControlAuthority::OperatorDebug,
                ttl_ms: 500,
            },
        )
        .is_err());
    let mother = sim.handshake(hello()).unwrap();
    let control = match sim
        .execute_in_session(
            &mother.session,
            CockpitRequest::AcquireControlLease {
                authority: ControlAuthority::Motherbrain,
                ttl_ms: 500,
            },
        )
        .unwrap()
    {
        CockpitResponse::ControlLeaseGranted(lease) => lease,
        other => panic!("{other:?}"),
    };
    assert!(sim
        .execute_with_lease(&mother.session, &control, CockpitRequest::Bootsel)
        .is_err());
    assert!(sim
        .execute_in_session(
            &mother.session,
            CockpitRequest::AcquireServiceLease {
                scope: ServiceScope::Bootsel,
                ttl_ms: 500,
            },
        )
        .is_err());
}

#[test]
fn ready_connector_initializes_cursor_at_welcome_without_dropping_session_event() {
    let mut ready = establish_session(SimCockpit::new(), hello(), None).unwrap();
    let batch = ready.poll_events().unwrap();
    assert_eq!(batch.dropped_before_seq, 0);
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::SessionOpened));
}

fn motherbrain_registration(address: &str, lease: &str) -> RegisterNetworkEndpoint {
    RegisterNetworkEndpoint {
        interface_id: "wlan0".into(),
        address_family: AddressFamily::Ipv4,
        address: address.into(),
        hostname: "motherbrain".into(),
        lease_identity: lease.into(),
        ttl_seconds: 600,
    }
}

#[test]
fn authenticated_motherbrain_registers_lease_and_reserved_dhcp_name_cannot() {
    let mut sim = SimCockpit::new();
    sim.add_network_lease(NetworkLease {
        leased_ip: "192.168.4.2".into(),
        client_mac: "02:00:00:00:00:02".into(),
        dhcp_client_identifier: "mb-client".into(),
        requested_hostname: Some("motherbrain".into()),
        lease_start: 0,
        lease_expiry: 3_600,
    });
    assert_eq!(sim.resolve_internal_name("motherbrain"), None);
    let outcome = sim.handshake(hello()).unwrap();
    let registration = motherbrain_registration("192.168.4.2", "mb-client");
    let response = sim
        .execute_in_session(
            &outcome.session,
            CockpitRequest::RegisterNetworkEndpoint(registration.clone()),
        )
        .unwrap();
    let CockpitResponse::NetworkEndpointRegistered(registered) = response else {
        panic!("registration response")
    };
    assert_eq!(
        registered.fqdn,
        format!("motherbrain.{DEFAULT_INTERNAL_DOMAIN}")
    );
    assert_eq!(
        sim.resolve_internal_name("motherbrain"),
        Some("192.168.4.2".into())
    );
    assert_eq!(
        sim.resolve_internal_name("pete"),
        Some("192.168.4.1".into())
    );
    assert_eq!(
        sim.resolve_internal_name("brainstem"),
        Some("192.168.4.1".into())
    );
    let duplicate = sim
        .execute_in_session(
            &outcome.session,
            CockpitRequest::RegisterNetworkEndpoint(registration.clone()),
        )
        .unwrap();
    let CockpitResponse::NetworkEndpointRegistered(duplicate) = duplicate else {
        panic!("duplicate registration response")
    };
    assert_eq!(
        registered.registration_generation,
        duplicate.registration_generation
    );

    let mut rebooted_hello = hello();
    rebooted_hello.boot_id = "mbboot-restarted".into();
    rebooted_hello.handshake_nonce = "hello-after-restart".into();
    let rebooted = sim.handshake(rebooted_hello).unwrap();
    let after_reboot = sim
        .execute_in_session(
            &rebooted.session,
            CockpitRequest::RegisterNetworkEndpoint(registration),
        )
        .unwrap();
    let CockpitResponse::NetworkEndpointRegistered(after_reboot) = after_reboot else {
        panic!("reboot registration response")
    };
    assert_ne!(
        duplicate.registration_generation,
        after_reboot.registration_generation
    );
}

#[test]
fn network_registration_rejects_no_session_forebrain_and_lease_mismatch() {
    let mut sim = SimCockpit::new();
    let registration = motherbrain_registration("192.168.4.2", "mb-client");
    assert!(matches!(
        sim.execute(CockpitRequest::RegisterNetworkEndpoint(
            registration.clone()
        )),
        Err(CockpitError::SessionRequired)
    ));

    let mut forebrain_hello = HandshakeHello::forebrain("pete-forebrain-lab");
    forebrain_hello.handshake_nonce = "forebrain-attempt".into();
    let forebrain = sim.handshake(forebrain_hello).unwrap();
    assert!(matches!(
        sim.execute_in_session(
            &forebrain.session,
            CockpitRequest::RegisterNetworkEndpoint(registration.clone())
        ),
        Err(CockpitError::Policy(_))
    ));

    let outcome = sim.handshake(hello()).unwrap();
    assert!(matches!(
        sim.execute_in_session(
            &outcome.session,
            CockpitRequest::RegisterNetworkEndpoint(registration)
        ),
        Err(CockpitError::Policy(_))
    ));
}

#[test]
fn expired_lease_removes_registered_dns_record() {
    let mut sim = SimCockpit::new();
    sim.add_network_lease(NetworkLease {
        leased_ip: "192.168.4.2".into(),
        client_mac: "mac".into(),
        dhcp_client_identifier: "lease".into(),
        requested_hostname: None,
        lease_start: 0,
        lease_expiry: 2,
    });
    let outcome = sim.handshake(hello()).unwrap();
    sim.execute_in_session(
        &outcome.session,
        CockpitRequest::RegisterNetworkEndpoint(motherbrain_registration("192.168.4.2", "lease")),
    )
    .unwrap();
    assert!(sim.resolve_internal_name("motherbrain").is_some());
    sim.advance_ms(2_000);
    assert_eq!(sim.resolve_internal_name("motherbrain"), None);
}

#[test]
fn simulator_capabilities_round_trip() {
    let mut sim = SimCockpit::new();
    let caps = sim.get_capabilities().unwrap();
    assert_eq!(caps.body_kind, "sim_create_oi");
    assert_eq!(caps.drive, "differential");
    assert!(caps.verbs.contains(&"cmd_vel".to_owned()));
    assert!(caps.events.contains(&"safety_tripped".to_owned()));
    assert_eq!(caps.limits.max_linear_mm_s, 500);
}

#[test]
fn simulator_audio_state_round_trips_through_status_and_events() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    assert_eq!(
        sim.execute(CockpitRequest::SetAudioSilent { silent: true })
            .unwrap(),
        CockpitResponse::Accepted
    );
    let status = sim.get_status().unwrap().summary();
    assert_eq!(status.audio_silent, Some(true));
    let events = sim.get_events_since(0).unwrap();
    assert!(events
        .events
        .iter()
        .any(|event| { event.kind == CockpitEventKind::AudioStateChanged && event.a == 1 }));
    sim.execute(CockpitRequest::SetAudioSilent { silent: false })
        .unwrap();
    assert_eq!(
        sim.get_status().unwrap().summary().audio_silent,
        Some(false)
    );
}

#[test]
fn cockpit_request_covers_public_firmware_verbs_from_body_toml() {
    let cockpit_verbs: BTreeSet<_> = sample_cockpit_requests()
        .into_iter()
        .map(|(verb, _, _)| verb)
        .filter(|verb| *verb != "bootsel")
        .collect();
    let firmware_verbs: BTreeSet<_> = body_toml_array("verbs").into_iter().collect();
    assert!(
        firmware_verbs.is_subset(&cockpit_verbs),
        "public firmware verbs must all be modeled by CockpitRequest"
    );
}

#[test]
fn cockpit_event_kind_covers_public_firmware_events_from_body_toml() {
    for event in body_toml_array("events") {
        assert!(
            !matches!(CockpitEventKind::from(event), CockpitEventKind::Unknown(_)),
            "body.toml event {event} is not modeled by CockpitEventKind"
        );
    }
}

#[test]
fn body_toml_capabilities_validate_local_cockpit_model() {
    let contract = CockpitContract::new(body_toml_capabilities());
    let report = contract.validate_local_model();
    assert!(
        report.is_clean(),
        "missing={:?} extra={:?} unknown_events={:?}",
        report.missing_verbs,
        report.extra_verbs,
        report.unknown_events
    );
}

#[test]
fn live_service_verbs_do_not_block_maintenance_handshake() {
    let mut capabilities = body_toml_capabilities();
    capabilities.verbs.push("bootsel".to_owned());
    capabilities.verbs.push("restart_mpu".to_owned());
    let contract = CockpitContract::new(capabilities);
    let report = contract.validate_local_model();

    assert!(
        report.missing_verbs.is_empty(),
        "missing={:?}",
        report.missing_verbs
    );
}

#[test]
fn previous_brainstem_contract_does_not_block_bootsel_handshake() {
    let mut capabilities = body_toml_capabilities();
    capabilities.verbs.extend(
        legacy_brainstem_convenience_verbs()
            .into_iter()
            .map(ToOwned::to_owned),
    );
    establish_session(
        SimCockpit::new().with_capabilities(capabilities),
        HandshakeHello::default_motherbrain(),
        None,
    )
    .expect("older firmware must remain flashable");
}

#[test]
fn pre_careful_brainstem_contract_remains_accepted_for_migration() {
    let mut capabilities = body_toml_capabilities();
    capabilities.verbs.retain(|verb| verb != "careful_mode");

    establish_session(
        SimCockpit::new().with_capabilities(capabilities),
        HandshakeHello::default_motherbrain(),
        None,
    )
    .expect("pre-CAREFUL firmware must remain flashable");
}

#[test]
fn pre_escape_motion_brainstem_contract_remains_accepted_for_migration() {
    let mut capabilities = body_toml_capabilities();
    capabilities.verbs.retain(|verb| verb != "escape_motion");

    establish_session(
        SimCockpit::new().with_capabilities(capabilities),
        HandshakeHello::default_motherbrain(),
        None,
    )
    .expect("pre-escape-motion firmware must remain flashable");
}

#[test]
fn cockpit_requests_serialize_to_firmware_json_kinds() {
    for (verb, expected_json_kind, _) in sample_cockpit_requests() {
        let request = sample_request_for(verb);
        let json = request.to_firmware_json(7).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            value.get("kind").and_then(serde_json::Value::as_str),
            Some(expected_json_kind),
            "{verb} serialized as {json}"
        );
        assert_eq!(
            value.get("command_id").and_then(serde_json::Value::as_u64),
            Some(7)
        );
    }
}

#[test]
fn cockpit_requests_serialize_to_compact_command_names() {
    for (verb, _, expected_compact_name) in sample_cockpit_requests() {
        let request = sample_request_for(verb);
        let line = request.to_compact_line(9);
        let first = line.split_ascii_whitespace().next().unwrap();
        assert_eq!(first, expected_compact_name, "{verb} serialized as {line}");
    }
}

#[test]
fn firmware_json_rewrites_policy_and_tones() {
    let policy = CockpitRequest::SetSafetyPolicy {
        policy: SafetyPolicy {
            bump: SafetyAction::BumpEscape,
            cliff: SafetyAction::Backoff,
            wheel_drop_latch: true,
        },
    };
    let value: serde_json::Value =
        serde_json::from_str(&policy.to_firmware_json(1).unwrap()).unwrap();
    assert!(value.get("policy").is_none());
    assert_eq!(value["bump_action"], "bump_escape");
    assert_eq!(value["cliff_action"], "backoff");
    assert_eq!(value["wheel_drop_latch"], true);

    let song = CockpitRequest::SongDefine {
        id: 2,
        tones: vec![SongTone {
            note: 72,
            duration_64ths: 8,
        }],
    };
    let value: serde_json::Value =
        serde_json::from_str(&song.to_firmware_json(2).unwrap()).unwrap();
    assert_eq!(value["tones"], "72:8");
}

#[test]
fn parses_json_accepted_and_rejected_command_responses() {
    let accepted = parse_json_cockpit_response(
        4,
        &CockpitRequest::Arm,
        r#"{"accepted":true,"command_id":4,"message":"accepted"}"#,
    )
    .unwrap();
    assert_eq!(accepted, CockpitResponse::Accepted);

    let rejected = parse_json_cockpit_response(
        5,
        &CockpitRequest::Arm,
        r#"{"accepted":false,"command_id":5,"message":"busy"}"#,
    )
    .unwrap_err();
    assert!(matches!(
        rejected,
        CockpitError::Rejected {
            command_id: 5,
            reason
        } if reason == "busy"
    ));
}

#[test]
fn parses_json_status_capabilities_and_events() {
    let status = parse_json_cockpit_response(
            1,
            &CockpitRequest::GetStatus,
            r#"{"type":"status","current_runtime_state":"idle","oi_mode":"safe","estop_latched":false,"safety_tripped":true,"safety_latch_kind":"tilt","event_next_seq":8,"audio_silent":true,"audio":{"silent":true,"last_requested_cue":"cliff","last_played_cue":"authority_acquired","last_playback_timestamp_ms":700,"suppressed_by_silent_count":2,"dropped_or_replaced_count":3},"create_sensors":{"charging_sources":2,"charging_indicator":"on"}}"#,
        )
        .unwrap();
    let CockpitResponse::Status(status) = status else {
        panic!("expected status response");
    };
    let summary = status.summary();
    assert_eq!(summary.runtime_state.as_deref(), Some("idle"));
    assert_eq!(summary.armed, Some(true));
    assert_eq!(summary.estop_latched, Some(false));
    assert_eq!(summary.safety_tripped, Some(true));
    assert_eq!(summary.safety_latch_kind, Some(SafetyLatchKind::Tilt));
    assert_eq!(summary.event_next_seq, Some(8));
    assert_eq!(summary.audio_silent, Some(true));
    assert_eq!(summary.audio_last_requested_cue.as_deref(), Some("cliff"));
    assert_eq!(
        summary.audio_last_played_cue.as_deref(),
        Some("authority_acquired")
    );
    assert_eq!(summary.audio_last_playback_timestamp_ms, Some(700));
    assert_eq!(summary.audio_suppressed_by_silent_count, Some(2));
    assert_eq!(summary.audio_dropped_or_replaced_count, Some(3));
    assert_eq!(summary.battery.charging_indicator, Some(true));
    assert!(summary.battery.home_base());

    let caps = parse_json_cockpit_response(
            2,
            &CockpitRequest::GetCapabilities,
            r#"{"accepted":true,"command_id":2,"body_kind":"create_oi","drive":"differential","verbs":["arm","cmd_vel"],"sensors":["bump"],"outputs":["drive"],"safety":["estop"],"events":["boot","safety_tripped"]}"#,
        )
        .unwrap();
    let CockpitResponse::Capabilities(caps) = caps else {
        panic!("expected capabilities response");
    };
    assert_eq!(caps.body_kind, "create_oi");
    assert_eq!(caps.verbs, ["arm", "cmd_vel"]);
    assert_eq!(caps.limits.max_linear_mm_s, i16::MAX);

    let events = parse_json_cockpit_response(
            3,
            &CockpitRequest::GetEvents { since_seq: 6 },
            r#"{"type":"events","since_seq":6,"oldest_seq":4,"next_seq":9,"dropped_before_seq":0,"events":[{"seq":7,"kind":"safety_tripped","a":1,"b":0,"c":0},{"seq":8,"kind":"motion_stopped","a":0,"b":0,"c":0}]}"#,
        )
        .unwrap();
    let CockpitResponse::Events(events) = events else {
        panic!("expected events response");
    };
    assert_eq!(events.since_seq, 6);
    assert_eq!(events.next_seq, 9);
    assert!(events.has_stop_reason());
}

#[test]
fn compact_status_infers_imu_tilt_safety_latch() {
    let summary = StatusSummary::from_raw(
            "OK 1 STATUS uptime_ms=1000 runtime=3 body=6 command=0 pending=0 power=2 oi=3 create_body_packets=1 create_last_body_packet_ms=900 charging_sources=2 imu_health=1 imu_tilt_mrad=2269 imu_impact_mm_s2=96",
        );

    assert_eq!(summary.safety_tripped, Some(true));
    assert_eq!(summary.safety_latch_kind, Some(SafetyLatchKind::Tilt));
    assert_eq!(summary.imu.tilt_magnitude_mrad, Some(2269));
    assert!(summary.battery.home_base());
}

#[test]
fn dock_ir_cue_decodes_and_steers_toward_both_buoys() {
    let green = DockIrCue::from_character(246).unwrap();
    assert_eq!(green.steering_mrad_s(400), -400);
    assert_eq!(green.bearing_hint_rad(), -0.35);
    assert!(green.force_field);

    let red = DockIrCue::from_character(250).unwrap();
    assert_eq!(red.steering_mrad_s(400), 400);
    assert_eq!(red.bearing_hint_rad(), 0.35);

    let centered = DockIrCue::from_character(254).unwrap();
    assert_eq!(centered.steering_mrad_s(400), 0);
    assert_eq!(centered.bearing_hint_rad(), 0.0);
    assert_eq!(centered.visible_score(), 0.85);
    assert_eq!(centered.near_score(), 0.55);

    assert_eq!(DockIrCue::from_character(255), None);
    assert_eq!(DockIrCue::from_character(0), None);
}

#[test]
fn status_summary_reports_complete_body_packet_age() {
    let summary = CockpitStatus {
        raw: serde_json::json!({
            "uptime_ms": 2_000,
            "current_runtime_state": "idle",
            "create_sensors": {
                "last_packet_id": 0,
                "complete_packet_count": 7,
                "last_complete_packet_timestamp_ms": 1_650,
                "bump_left": false
            }
        })
        .to_string(),
    }
    .summary();

    assert_eq!(summary.body_packet_count, Some(7));
    assert_eq!(summary.body_packet_age_ms, Some(350));
    assert_eq!(summary.body_packet_complete, Some(true));
    assert!(summary.has_fresh_complete_body_packet(500));
    assert!(!summary.has_fresh_complete_body_packet(250));
}

#[test]
fn status_summary_preserves_create_ir_from_json_and_compact_status() {
    let json = StatusSummary::from_raw(
        &serde_json::json!({
            "create_sensors": {
                "complete_packet_count": 1,
                "ir_byte": 248
            }
        })
        .to_string(),
    );
    let compact =
        StatusSummary::from_raw("OK 1 STATUS create_body_packets=1 ir_byte=137 bump_left=false");

    assert_eq!(json.infrared_character, Some(248));
    assert_eq!(compact.infrared_character, Some(137));
}

#[test]
fn compact_status_requires_a_complete_body_packet() {
    let summary = CockpitStatus {
            raw: "OK 1 STATUS uptime_ms=2000 create_rx_packets=7 create_last_packet_ms=1900 create_sensor_packet_id=35 create_body_packets=0 create_last_body_packet_ms=0 bump_left=false".into(),
        }
        .summary();

    assert_eq!(summary.body_packet_age_ms, None);
    assert_eq!(summary.body_packet_complete, Some(false));
    assert!(!summary.has_fresh_complete_body_packet(500));
}

#[test]
fn malformed_json_response_maps_to_json_error() {
    let err = parse_json_cockpit_response(1, &CockpitRequest::Arm, "{not-json").unwrap_err();
    assert!(matches!(err, CockpitError::Json(_)));
}

#[test]
fn simulator_event_cursor_happy_path() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    let mut cursor = EventCursor::new();
    let boot = cursor.poll(&mut sim).unwrap();
    assert_eq!(boot.events[0].kind, CockpitEventKind::Boot);
    sim.arm().unwrap();
    let batch = cursor.poll(&mut sim).unwrap();
    assert_eq!(cursor.next_seq(), batch.next_seq - 1);
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::CommandCompleted));
}

#[test]
fn simulator_detects_missed_events_through_dropped_before_seq() {
    let mut sim = SimCockpit::new()
        .with_unscoped_bench_mode()
        .with_event_capacity(3);
    for _ in 0..4 {
        sim.arm().unwrap();
    }
    let batch = sim.get_events_since(0).unwrap();
    assert!(batch.dropped_before_seq > 0);
    assert!(matches!(
        batch.ensure_no_missed_events(),
        Err(CockpitError::MissedEvents { .. })
    ));
}

#[test]
fn simulator_arm_stop_disarm_lifecycle() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.arm().unwrap();
    sim.cmd_vel(50, 0, 100).unwrap();
    sim.stop().unwrap();
    sim.disarm().unwrap();
    let batch = sim.get_events_since(0).unwrap();
    let kinds: Vec<_> = batch.events.iter().map(|event| &event.kind).collect();
    assert!(kinds.contains(&&CockpitEventKind::CommandInterrupted));
    assert!(kinds.contains(&&CockpitEventKind::MotionStopped));
    assert!(kinds.contains(&&CockpitEventKind::CommandCompleted));
}

#[test]
fn simulator_cmd_vel_completes_after_ttl() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.cmd_vel(70, 10, 300).unwrap();
    sim.advance_ms(299);
    assert!(!sim
        .get_events_since(0)
        .unwrap()
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::MotionStopped));
    sim.advance_ms(1);
    let batch = sim.get_events_since(0).unwrap();
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::MotionStopped));
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::CommandCompleted));
}

#[test]
fn simulator_estop_and_clear_estop() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.estop().unwrap();
    sim.clear_estop().unwrap();
    let batch = sim.get_events_since(0).unwrap();
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::EStopLatched));
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::EStopCleared));
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::SafetyCleared));
}

#[test]
fn simulator_heartbeat_expiry_is_stop_reason() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.cmd_vel(70, 0, 1_000).unwrap();
    sim.heartbeat_stop(100).unwrap();
    sim.advance_ms(100);
    let batch = sim.get_events_since(0).unwrap();
    assert!(batch.has_stop_reason());
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::HeartbeatExpired));
}

#[test]
fn simulator_command_rejection_alone_is_diagnostic() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.push_event(CockpitEventKind::CommandRejected, 7, 11, (6 << 8) | 1);
    let batch = sim.get_events_since(0).unwrap();

    assert!(!batch.has_stop_reason());
    let rejected = batch
        .events
        .iter()
        .find(|event| event.kind == CockpitEventKind::CommandRejected)
        .unwrap();
    assert_eq!(
        rejected.command_rejection(),
        Some(CommandRejection {
            command_id: 7,
            command_seq: 11,
            command_code: 6,
            reason: CommandRejectReason::Busy,
        })
    );
}

#[test]
fn simulator_safety_tripped_stops_motion_and_rejects_motion() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.cmd_vel(70, 0, 1_000).unwrap();
    sim.trip_safety();
    sim.cmd_vel(10, 0, 100).unwrap();
    let batch = sim.get_events_since(0).unwrap();
    assert!(batch.has_stop_reason());
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::CommandRejected));
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::MotionStopped));
}

#[test]
fn simulator_reset_odometry() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.reset_odometry().unwrap();
    assert_eq!(sim.odometry_reset_count(), 1);
    let status = sim.get_status().unwrap();
    assert!(status.raw.contains("odometry_resets=1"));
    assert_eq!(status.summary().odometry.reset_count, Some(1));
    assert_eq!(status.summary().odometry.distance_mm, Some(0));
}

#[test]
fn simulator_builtin_sensor_edges_trip_and_clear() {
    let mut sim = SimCockpit::new();
    sim.set_bump(true, false);
    sim.set_bump(false, false);
    sim.set_cliff(true);
    sim.set_cliff(false);
    sim.set_wall(true);
    sim.set_wall(false);
    sim.set_virtual_wall(true);
    sim.set_virtual_wall(false);

    let batch = sim.get_events_since(0).unwrap();
    assert_eq!(
        batch
            .events
            .iter()
            .filter(|event| event.kind == CockpitEventKind::BumpChanged)
            .count(),
        2
    );
    assert_eq!(
        batch
            .events
            .iter()
            .filter(|event| event.kind == CockpitEventKind::CliffChanged)
            .count(),
        2
    );
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::WallChanged && event.a == 1));
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::VirtualWallChanged && event.a == 0));
}

#[test]
fn simulator_stationary_bump_latches_without_starting_withdrawal() {
    let mut sim = SimCockpit::new();
    sim.set_bump(true, false);

    let status = sim.get_status().unwrap().summary();
    assert_eq!(status.safety_tripped, Some(true));
    assert_eq!(status.safety_latch_kind, Some(SafetyLatchKind::Bump));
    assert!(sim.active_contact_withdrawal.is_none());
    assert!(!sim
        .get_events_since(0)
        .unwrap()
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::ContactWithdrawalStarted));
}

#[test]
fn simulator_contact_withdrawal_is_typed_and_authority_independent() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.cmd_vel(80, 0, 1_000).unwrap();
    sim.set_bump(true, false);
    sim.control_lease = None;
    sim.control_lease_expires_at_ms = None;
    sim.advance_ms(300);

    let events = sim.get_events_since(0).unwrap().events;
    let lifecycle: Vec<_> = events
        .iter()
        .filter_map(CockpitEvent::contact_withdrawal)
        .collect();
    assert_eq!(lifecycle.len(), 2);
    assert!(matches!(
        lifecycle[0],
        ContactWithdrawalEvent::Started {
            contact_bits: 1,
            repeated_count: 1,
            preempted_command_id: 1,
            reverse_speed_mm_s: 80,
            maximum_duration_ms: 300,
        }
    ));
    assert!(matches!(
        lifecycle[1],
        ContactWithdrawalEvent::Completed {
            outcome: ContactWithdrawalOutcome::Completed,
            final_stopped: true,
            observed_displacement_mm: -24,
            elapsed_ms: 300,
            ..
        }
    ));
}

#[test]
fn simulator_escape_motion_is_generation_bound_and_reflex_ordered() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.cmd_vel(80, 0, 1_000).unwrap();
    sim.set_bump(true, false);
    let generation = sim
        .get_status()
        .unwrap()
        .summary()
        .safety_hazard_generation
        .unwrap();

    assert!(matches!(
        sim.escape_motion(SafetyLatchKind::Bump, generation, -50, 0, 250),
        Err(CockpitError::Rejected { ref reason, .. }) if reason == "busy"
    ));
    sim.advance_ms(CONTACT_WITHDRAWAL_DURATION_MS);
    sim.escape_motion(SafetyLatchKind::Bump, generation, -50, 0, 250)
        .unwrap();
    assert!(sim
        .get_status()
        .unwrap()
        .raw
        .contains("active_cmd_vel=true"));

    sim.set_cliff(true);
    let status = sim.get_status().unwrap().summary();
    assert!(status.raw.contains("active_cmd_vel=false"));
    assert_eq!(status.safety_latch_kind, Some(SafetyLatchKind::Cliff));
    assert!(matches!(
        sim.escape_motion(SafetyLatchKind::Bump, generation, -50, 0, 250),
        Err(CockpitError::Rejected { ref reason, .. }) if reason == "hazard_mismatch"
    ));
}

#[test]
fn simulator_wheel_drop_latches_and_clears() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.cmd_vel(70, 0, 1_000).unwrap();
    sim.set_wheel_drop(true);
    sim.set_wheel_drop(false);
    sim.clear_safety_latch(SafetyLatchKind::WheelDrop).unwrap();

    let batch = sim.get_events_since(0).unwrap();
    assert!(batch.has_stop_reason());
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::WheelDropLatched));
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::WheelDropCleared));
}

#[test]
fn simulator_low_battery_and_charging_state_change() {
    let mut sim = SimCockpit::new();
    sim.set_battery(400, 2600);
    sim.set_charging_state(2);

    let status = sim.get_status().unwrap().summary();
    assert_eq!(status.battery.percent, Some(15));
    assert_eq!(status.battery.low, Some(true));
    assert_eq!(status.battery.charging_state, Some(2));

    let batch = sim.get_events_since(0).unwrap();
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::BatteryLow && event.a == 15));
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::ChargingStateChanged && event.a == 2));
}

#[test]
fn diagnostic_session_can_establish_sensor_stream_through_cockpit_trait() {
    let connector = SimCockpit::new();
    let mut cockpit =
        establish_diagnostic_session(connector, HandshakeHello::default_motherbrain(), None)
            .unwrap();

    Cockpit::stream_sensors(&mut cockpit, true, 0, 250).unwrap();
}

#[test]
fn simulator_buttons_and_ir_changes_are_events() {
    let mut sim = SimCockpit::new();
    sim.set_buttons(0b0000_0011);
    sim.set_ir_byte(248);

    let status = sim.get_status().unwrap().summary();
    assert_eq!(status.contact.any_contact(), Some(false));
    let batch = sim.get_events_since(0).unwrap();
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::ButtonsChanged && event.a == 3));
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::IrChanged && event.a == 248));
}

#[test]
fn parses_ok_and_err_responses() {
    assert!(expect_ok(2, "OK 2").is_ok());
    assert!(matches!(
        expect_ok(2, "ERR 2 busy"),
        Err(CockpitError::Rejected {
            command_id: 2,
            reason
        }) if reason == "busy"
    ));
}

#[test]
fn parses_status_response_as_raw_status() {
    expect_ok(9, "OK 9 STATUS runtime=idle demo=idle").unwrap();
    let status = CockpitStatus {
        raw: "OK 9 STATUS runtime=idle demo=idle".to_owned(),
    };
    assert!(status.raw.contains("runtime=idle"));
}

#[test]
fn parses_compact_events() {
    let batch = parse_events(
            7,
            12,
            "OK 7 EVENTS since=12 oldest=4 next=15 dropped_before=0 count=2 | 13:motion_requested:1,2,3 | 14:safety_tripped:2,0,0",
        )
        .unwrap();
    assert_eq!(batch.next_seq, 15);
    assert_eq!(batch.dropped_before_seq, 0);
    assert_eq!(batch.events.len(), 2);
    assert_eq!(batch.events[1].kind, CockpitEventKind::SafetyTripped);
    assert!(batch.has_stop_reason());
}

#[test]
fn parses_unknown_event_kinds() {
    let batch = parse_events(
            7,
            12,
            "OK 7 EVENTS since=12 oldest=13 next=14 dropped_before=0 count=1 | 13:new_future_event:1,2,3",
        )
        .unwrap();
    assert_eq!(
        batch.events[0].kind,
        CockpitEventKind::Unknown("new_future_event".to_owned())
    );
    assert_eq!(batch.events[0].kind.as_str(), "new_future_event");
}

#[test]
fn rejects_malformed_or_truncated_event_lines() {
    for line in [
            "OK 7 EVENTS since=12 oldest=13 next=14 dropped_before=0 count=1 | malformed",
            "OK 7 EVENTS since=12 oldest=13 next=14 dropped_before=0 count=1 | 13:motion_requested:1,2",
            "OK 7 EVENTS since=12 oldest=13 next=14 dropped_before=0 count=2 | 13:motion_requested:1,2,3",
        ] {
            assert!(matches!(
                parse_events(7, 12, line),
                Err(CockpitError::BadResponse(_))
            ));
        }
}

#[test]
fn parses_large_event_lists_near_response_buffer_limits() {
    let mut line = String::from("OK 7 EVENTS since=0 oldest=1 next=29 dropped_before=0 count=28");
    for seq in 1..29 {
        line.push_str(&format!(
            " | {seq}:motion_requested:{seq},{},{}",
            seq + 1,
            seq + 2
        ));
    }
    assert!(line.len() < DEFAULT_UART_MAX_RESPONSE_LEN);
    let batch = parse_events(7, 0, &line).unwrap();
    assert_eq!(batch.events.len(), 28);
    assert_eq!(batch.events.last().unwrap().seq, 28);
}

#[test]
fn detects_missed_events() {
    let batch = parse_events(
        1,
        0,
        "OK 1 EVENTS since=0 oldest=20 next=52 dropped_before=20 count=0",
    )
    .unwrap();
    assert!(matches!(
        batch.ensure_no_missed_events(),
        Err(CockpitError::MissedEvents {
            dropped_before_seq: 20
        })
    ));
}

#[test]
fn parses_capabilities_without_body_specific_api() {
    let caps = parse_capabilities(
            3,
            "OK 3 CAPABILITIES body_kind=create_oi drive=differential verbs=arm,stop,cmd_vel sensors=bump,battery outputs=lights,song safety=bump,estop events=boot,safety_tripped limits=max_linear_mm_s:500 max_tones=16 song_slots=16 feedback_slots=6 sensor_packets=0,7-31",
        )
        .unwrap();
    assert_eq!(caps.drive, "differential");
    assert_eq!(caps.verbs, ["arm", "stop", "cmd_vel"]);
    assert_eq!(caps.events, ["boot", "safety_tripped"]);
    assert_eq!(caps.limits.max_linear_mm_s, 500);
}

#[test]
fn parses_json_capability_limits() {
    let caps = parse_json_capabilities(&serde_json::json!({
        "body_kind":"create_oi",
        "drive":"differential",
        "verbs":["cmd_vel"],
        "events":["boot"],
        "limits":{
            "max_linear_mm_s":120,
            "max_angular_mrad_s":800,
            "min_ttl_ms":20,
            "max_ttl_ms":900
        }
    }))
    .unwrap();
    assert_eq!(
        caps.limits,
        CockpitLimits {
            max_linear_mm_s: 120,
            max_angular_mrad_s: 800,
            min_ttl_ms: 20,
            max_ttl_ms: 900,
        }
    );
}

#[test]
fn contract_rejects_unsupported_lights_music_and_step_verbs() {
    let contract =
        CockpitContract::new(sim_caps_without(&["set_lights", "song_play", "dock_align"]));
    assert!(matches!(
        contract.validate_request(&CockpitRequest::SetLights {
            pattern: LightPattern::Status
        }),
        Err(CockpitError::Policy(message)) if message.contains("set_lights")
    ));
    assert!(matches!(
        contract.validate_request(&CockpitRequest::SongPlay { id: 0 }),
        Err(CockpitError::Policy(message)) if message.contains("song_play")
    ));
    assert!(matches!(
        contract.validate_request(&CockpitRequest::DockAlign {
            bearing_mrad: 0,
            range_mm: 400,
            max_linear_mm_s: 80,
            max_angular_mrad_s: 500,
            stop_range_mm: 200,
            ttl_ms: 300,
        }),
        Err(CockpitError::Policy(message)) if message.contains("dock_align")
    ));
}

#[test]
fn safe_cockpit_clamps_motion_to_body_limits() {
    let mut caps = sim_caps_with_all_verbs();
    caps.limits.max_linear_mm_s = 40;
    caps.limits.max_angular_mrad_s = 100;
    caps.limits.min_ttl_ms = 50;
    caps.limits.max_ttl_ms = 200;
    let sim = SimCockpit::new()
        .with_unscoped_bench_mode()
        .with_capabilities(caps);
    let mut safe = SafeCockpit::with_policy(
        sim,
        AgentPolicy {
            motion_ttl_ms: 500,
            heartbeat_timeout_ms: 500,
        },
    );
    safe.pulse_motion(120, 300).unwrap();
    let batch = safe.client_mut().get_events_since(0).unwrap();
    let motion = batch
        .events
        .iter()
        .find(|event| event.kind == CockpitEventKind::MotionRequested)
        .unwrap();
    assert_eq!(motion.a, pack_i16_pair(40, 100));
    assert_eq!(motion.b, 200);
}

#[test]
fn safe_cockpit_reports_preexisting_bump_latch_as_typed_motion_stop() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.set_bump(true, false);
    let mut safe = SafeCockpit::with_policy(
        sim,
        AgentPolicy {
            motion_ttl_ms: 100,
            heartbeat_timeout_ms: 0,
        },
    );

    assert!(matches!(
        safe.pulse_motion(20, 0),
        Err(CockpitError::MotionStopped { reasons })
            if reasons == vec![SafeStopReason::SafetyTripped {
                latch: Some(SafetyLatchKind::Bump),
            }]
    ));
}

#[test]
fn safe_cockpit_requires_heartbeat_only_when_policy_uses_it() {
    let caps = sim_caps_without(&["heartbeat_stop"]);
    let sim = SimCockpit::new()
        .with_unscoped_bench_mode()
        .with_capabilities(caps.clone());
    let mut safe = SafeCockpit::new(sim);
    assert!(matches!(
        safe.pulse_motion(20, 0),
        Err(CockpitError::Policy(message)) if message.contains("heartbeat_stop")
    ));

    let sim = SimCockpit::new()
        .with_unscoped_bench_mode()
        .with_capabilities(caps);
    let mut safe = SafeCockpit::with_policy(
        sim,
        AgentPolicy {
            motion_ttl_ms: 100,
            heartbeat_timeout_ms: 0,
        },
    );
    safe.pulse_motion(20, 0).unwrap();
}

#[test]
fn safe_cockpit_does_not_treat_historical_command_rejection_as_motion_stop() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.push_event(CockpitEventKind::CommandRejected, 7, 0, 0);
    let mut safe = SafeCockpit::with_policy(
        sim,
        AgentPolicy {
            motion_ttl_ms: 100,
            heartbeat_timeout_ms: 0,
        },
    );

    safe.pulse_motion(20, 0).unwrap();
}

#[test]
fn uart_config_defaults_to_forebrain_baud() {
    let config = UartCockpitConfig::new("/dev/ttyTEST0");
    assert_eq!(config.baud_rate, DEFAULT_UART_BAUD_RATE);
    assert_eq!(config.timeout, DEFAULT_UART_TIMEOUT);
    assert_eq!(config.max_response_len, DEFAULT_UART_MAX_RESPONSE_LEN);
}

#[test]
fn malformed_response_maps_to_bad_response() {
    let err = expect_ok(2, "ERR 2 parse").unwrap_err();
    assert!(matches!(err, CockpitError::BadResponse(_)));
}

#[test]
fn mismatched_sequence_maps_to_bad_response() {
    let err = expect_ok(1, "OK 12").unwrap_err();
    assert!(matches!(err, CockpitError::BadResponse(_)));
}

#[test]
fn non_utf8_response_maps_to_bad_response() {
    let err = response_from_bytes(&[0xff]).unwrap_err();
    assert!(matches!(err, CockpitError::BadResponse(_)));
}

fn sample_cockpit_requests() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        ("ping", "ping", "PING"),
        ("bootsel", "bootsel", "BOOTSEL"),
        ("restart_create", "restart_create", "RESTART_CREATE"),
        ("status", "status", "STATUS"),
        ("get_capabilities", "get_capabilities", "GET_CAPABILITIES"),
        ("get_events", "get_events", "GET_EVENTS"),
        ("arm", "arm", "ARM"),
        ("disarm", "disarm", "DISARM"),
        ("stop", "stop", "STOP"),
        ("estop", "estop", "ESTOP"),
        ("clear_estop", "clear_estop", "CLEAR_ESTOP"),
        (
            "clear_safety_latch",
            "clear_safety_latch",
            "CLEAR_SAFETY_LATCH",
        ),
        ("careful_mode", "careful_mode", "CAREFUL_MODE"),
        ("escape_motion", "escape_motion", "ESCAPE_MOTION"),
        (
            "clear_motion_queue",
            "clear_motion_queue",
            "CLEAR_MOTION_QUEUE",
        ),
        ("cmd_vel", "cmd_vel", "CMD_VEL"),
        ("drive_direct", "drive_direct", "DRIVE_DIRECT"),
        ("drive_arc", "drive_arc", "DRIVE_ARC"),
        ("drive_for", "drive_for", "DRIVE_FOR"),
        ("turn_by", "turn_by", "TURN_BY"),
        ("arc_for", "arc_for", "ARC_FOR"),
        ("creep_until", "creep_until", "CREEP_UNTIL"),
        ("scan_arc", "scan_arc", "SCAN_ARC"),
        ("face_bearing", "face_bearing", "FACE_BEARING"),
        ("track_bearing", "track_bearing", "TRACK_BEARING"),
        ("hold_heading", "hold_heading", "HOLD_HEADING"),
        ("turn_to_heading", "turn_to_heading", "TURN_TO_HEADING"),
        ("dock_align", "dock_align", "DOCK_ALIGN"),
        ("wall_follow", "wall_follow", "WALL_FOLLOW"),
        ("wiggle_align", "wiggle_align", "WIGGLE_ALIGN"),
        ("bump_escape", "bump_escape", "BUMP_ESCAPE"),
        ("unstick", "unstick", "UNSTICK"),
        ("cliff_guard", "cliff_guard", "CLIFF_GUARD"),
        ("heartbeat_stop", "heartbeat_stop", "HEARTBEAT_STOP"),
        ("request_sensors", "request_sensors", "REQUEST_SENSORS"),
        ("stream_sensors", "stream_sensors", "STREAM_SENSORS"),
        (
            "set_safety_policy",
            "set_safety_policy",
            "SET_SAFETY_POLICY",
        ),
        ("song_define", "song_define", "SONG_DEFINE"),
        ("song_play", "song_play", "SONG_PLAY"),
        ("define_chirp", "define_chirp", "DEFINE_CHIRP"),
        ("play_feedback", "play_feedback", "PLAY_FEEDBACK"),
        ("set_silent", "set_silent", "SET_SILENT"),
        ("power_state", "power_state", "POWER_STATE"),
        ("create_power_on", "create_power_on", "CREATE_POWER_ON"),
        ("create_power_off", "create_power_off", "CREATE_POWER_OFF"),
        ("calibrate_turn", "calibrate_turn", "CALIBRATE_TURN"),
        (
            "orientation_probe",
            "orientation_probe",
            "ORIENTATION_PROBE",
        ),
        ("reset_odometry", "reset_odometry", "RESET_ODOMETRY"),
        (
            "zero_imu_orientation",
            "zero_imu_orientation",
            "ZERO_IMU_ORIENTATION",
        ),
        (
            "clear_imu_orientation",
            "clear_imu_orientation",
            "CLEAR_IMU_ORIENTATION",
        ),
        ("dock", "dock", "DOCK"),
        ("set_lights", "set_lights", "SET_LIGHTS"),
        ("set_mode", "set_mode", "SET_MODE"),
    ]
}

fn body_toml() -> toml::Value {
    include_str!("../../pete-brainstem/body.toml")
        .parse()
        .unwrap()
}

fn body_toml_array(key: &str) -> Vec<&'static str> {
    let body = body_toml();
    let values = body["capabilities"][key].as_array().unwrap();
    values
        .iter()
        .map(|value| {
            let value = value.as_str().unwrap().to_owned();
            Box::leak(value.into_boxed_str()) as &'static str
        })
        .collect()
}

fn body_toml_capabilities() -> CockpitCapabilities {
    let body = body_toml();
    let limits = &body["limits"];
    CockpitCapabilities {
        body_kind: body["body"]["kind"].as_str().unwrap().to_owned(),
        drive: body["body"]["drive"].as_str().unwrap().to_owned(),
        verbs: body_toml_array("verbs")
            .into_iter()
            .map(ToOwned::to_owned)
            .collect(),
        sensors: body_toml_array("sensors")
            .into_iter()
            .map(ToOwned::to_owned)
            .collect(),
        outputs: body_toml_array("outputs")
            .into_iter()
            .map(ToOwned::to_owned)
            .collect(),
        safety: body_toml_array("safety")
            .into_iter()
            .map(ToOwned::to_owned)
            .collect(),
        events: body_toml_array("events")
            .into_iter()
            .map(ToOwned::to_owned)
            .collect(),
        limits: CockpitLimits {
            max_linear_mm_s: limits["max_linear_mm_s"]
                .as_integer()
                .unwrap()
                .try_into()
                .unwrap(),
            max_angular_mrad_s: limits["max_angular_mrad_s"]
                .as_integer()
                .unwrap()
                .try_into()
                .unwrap(),
            min_ttl_ms: limits["min_ttl_ms"]
                .as_integer()
                .unwrap()
                .try_into()
                .unwrap(),
            max_ttl_ms: limits["max_ttl_ms"]
                .as_integer()
                .unwrap()
                .try_into()
                .unwrap(),
        },
    }
}

fn sim_caps_with_all_verbs() -> CockpitCapabilities {
    let mut caps = SimCockpit::new().get_capabilities().unwrap();
    caps.verbs = CockpitRequest::capability_verbs()
        .into_iter()
        .map(ToOwned::to_owned)
        .collect();
    caps.events = body_toml_array("events")
        .into_iter()
        .map(ToOwned::to_owned)
        .collect();
    caps
}

fn sim_caps_without(without: &[&str]) -> CockpitCapabilities {
    let mut caps = sim_caps_with_all_verbs();
    caps.verbs.retain(|verb| !without.contains(&verb.as_str()));
    caps
}

fn sample_request_for(verb: &str) -> CockpitRequest {
    match verb {
        "ping" => CockpitRequest::Ping,
        "bootsel" => CockpitRequest::Bootsel,
        "restart_create" => CockpitRequest::RestartCreate,
        "reset_motherbrain" => CockpitRequest::ResetMotherbrain,
        "status" => CockpitRequest::GetStatus,
        "get_capabilities" => CockpitRequest::GetCapabilities,
        "get_events" => CockpitRequest::GetEvents { since_seq: 3 },
        "arm" => CockpitRequest::Arm,
        "disarm" => CockpitRequest::Disarm,
        "stop" => CockpitRequest::Stop,
        "estop" => CockpitRequest::EStop,
        "clear_estop" => CockpitRequest::ClearEStop,
        "clear_safety_latch" => CockpitRequest::ClearSafetyLatch {
            latch: SafetyLatchKind::Bump,
        },
        "careful_mode" => CockpitRequest::CarefulMode { ttl_ms: 5_000 },
        "escape_motion" => CockpitRequest::EscapeMotion {
            hazard: SafetyLatchKind::Bump,
            hazard_generation: 42,
            linear_mm_s: -50,
            angular_mrad_s: 0,
            ttl_ms: 250,
        },
        "clear_motion_queue" => CockpitRequest::ClearMotionQueue,
        "cmd_vel" => CockpitRequest::CmdVel {
            linear_mm_s: 10,
            angular_mrad_s: 20,
            ttl_ms: 300,
        },
        "drive_direct" => CockpitRequest::DriveDirect {
            left_mm_s: 10,
            right_mm_s: 11,
            ttl_ms: 300,
        },
        "drive_arc" => CockpitRequest::DriveArc {
            velocity_mm_s: 10,
            radius_mm: 200,
            ttl_ms: 300,
        },
        "drive_for" => CockpitRequest::DriveFor {
            distance_mm: 300,
            velocity_mm_s: 80,
            timeout_ms: 2_000,
        },
        "turn_by" => CockpitRequest::TurnBy {
            angle_mrad: 1_570,
            angular_mrad_s: 800,
            timeout_ms: 2_000,
        },
        "arc_for" => CockpitRequest::ArcFor {
            velocity_mm_s: 80,
            radius_mm: 250,
            duration_ms: 1_000,
        },
        "creep_until" => CockpitRequest::CreepUntil {
            velocity_mm_s: 40,
            angular_mrad_s: 0,
            timeout_ms: 1_000,
        },
        "scan_arc" => CockpitRequest::ScanArc {
            angle_mrad: 3_140,
            angular_mrad_s: 500,
            timeout_ms: 4_000,
        },
        "face_bearing" => CockpitRequest::FaceBearing {
            bearing_mrad: 100,
            max_angular_mrad_s: 500,
            tolerance_mrad: 35,
            ttl_ms: 300,
        },
        "track_bearing" => CockpitRequest::TrackBearing {
            bearing_mrad: 100,
            range_mm: 900,
            max_linear_mm_s: 120,
            max_angular_mrad_s: 500,
            stop_range_mm: 250,
            ttl_ms: 300,
        },
        "hold_heading" => CockpitRequest::HoldHeading {
            heading_error_mrad: 100,
            velocity_mm_s: 80,
            max_angular_mrad_s: 500,
            ttl_ms: 300,
        },
        "turn_to_heading" => CockpitRequest::TurnToHeading {
            heading_error_mrad: 100,
            angular_mrad_s: 500,
            tolerance_mrad: 35,
            timeout_ms: 2_000,
        },
        "dock_align" => CockpitRequest::DockAlign {
            bearing_mrad: 50,
            range_mm: 600,
            max_linear_mm_s: 80,
            max_angular_mrad_s: 500,
            stop_range_mm: 250,
            ttl_ms: 300,
        },
        "wall_follow" => CockpitRequest::WallFollow {
            distance_error_mm: 20,
            velocity_mm_s: 80,
            max_angular_mrad_s: 400,
            ttl_ms: 300,
        },
        "wiggle_align" => CockpitRequest::WiggleAlign {
            amplitude_mrad: 200,
            angular_mrad_s: 500,
            cycles: 2,
        },
        "bump_escape" => CockpitRequest::BumpEscape {
            direction: EscapeDirection::Either,
            backoff_mm_s: 80,
            turn_angular_mrad_s: 900,
        },
        "unstick" => CockpitRequest::Unstick {
            direction: EscapeDirection::Either,
            backoff_mm_s: 90,
            turn_angular_mrad_s: 900,
        },
        "cliff_guard" => CockpitRequest::CliffGuard { clear: false },
        "heartbeat_stop" => CockpitRequest::HeartbeatStop { timeout_ms: 900 },
        "request_sensors" => CockpitRequest::RequestSensors { packet_id: 0 },
        "stream_sensors" => CockpitRequest::StreamSensors {
            enabled: true,
            packet_id: 0,
            period_ms: 250,
        },
        "set_safety_policy" => CockpitRequest::SetSafetyPolicy {
            policy: SafetyPolicy {
                bump: SafetyAction::Stop,
                cliff: SafetyAction::Stop,
                wheel_drop_latch: true,
            },
        },
        "song_define" => CockpitRequest::SongDefine {
            id: 1,
            tones: sample_tones(),
        },
        "song_play" => CockpitRequest::SongPlay { id: 1 },
        "define_chirp" => CockpitRequest::DefineChirp {
            feedback: FeedbackKind::Ok,
            tones: sample_tones(),
        },
        "play_feedback" => CockpitRequest::PlayFeedback {
            feedback: FeedbackKind::Ok,
        },
        "set_silent" => CockpitRequest::SetAudioSilent { silent: true },
        "power_state" => CockpitRequest::PowerState {
            request: PowerStateRequest::Wake,
        },
        "create_power_on" => CockpitRequest::CreatePowerOn,
        "create_power_off" => CockpitRequest::CreatePowerOff,
        "calibrate_turn" => CockpitRequest::CalibrateTurn {
            angular_mrad_s: 500,
            duration_ms: 1_000,
        },
        "orientation_probe" => CockpitRequest::OrientationProbe {
            angular_mrad_s: 250,
            duration_ms: 400,
        },
        "reset_odometry" => CockpitRequest::ResetOdometry,
        "zero_imu_orientation" => CockpitRequest::ZeroImuOrientation,
        "clear_imu_orientation" => CockpitRequest::ClearImuOrientation,
        "dock" => CockpitRequest::Dock,
        "set_lights" => CockpitRequest::SetLights {
            pattern: LightPattern::Status,
        },
        "set_mode" => CockpitRequest::SetMode {
            mode: CreateOiMode::Safe,
        },
        other => panic!("missing sample for {other}"),
    }
}

fn sample_tones() -> Vec<SongTone> {
    vec![SongTone {
        note: 72,
        duration_64ths: 8,
    }]
}

#[test]
fn production_possession_renews_and_replaces_lease() {
    let ready = establish_session(SimCockpit::new(), hello(), None).unwrap();
    let mut possession = MotherbrainPossession::acquire(ready, 1_000).unwrap();
    let first = possession.snapshot();
    possession.lease_acquired_at = Instant::now() - Duration::from_millis(800);
    possession.maintain().unwrap();
    let renewed = possession.snapshot();
    assert!(renewed.possessed);
    assert!(renewed.lease_generation > first.lease_generation);
    assert_ne!(renewed.lease_id, first.lease_id);
}

#[test]
fn production_possession_renews_long_lease_on_short_cadence() {
    let ready = establish_session(SimCockpit::new(), hello(), None).unwrap();
    let mut possession = MotherbrainPossession::acquire(ready, 60_000).unwrap();
    let first = possession.snapshot();
    possession.lease_acquired_at =
        Instant::now() - Duration::from_millis(POSSESSION_LEASE_RENEW_INTERVAL_MS as u64 + 1);

    possession.maintain().unwrap();

    let renewed = possession.snapshot();
    assert!(renewed.possessed);
    assert!(renewed.lease_generation > first.lease_generation);
    assert_ne!(renewed.lease_id, first.lease_id);
}

#[test]
fn production_possession_retries_transient_busy_commands() {
    let cockpit = BusyOnceCockpit {
        inner: SimCockpit::new(),
        busy_remaining: 0,
        attempts: 0,
        heartbeat_attempts: 0,
        cmd_vel_attempts: 0,
        last_heartbeat_timeout_ms: None,
        last_bump_escape: None,
    };
    let ready = establish_session(cockpit, hello(), None).unwrap();
    let mut possession = MotherbrainPossession::acquire(ready, 1_000).unwrap();
    possession.session.connector_mut().busy_remaining = 1;
    let attempts_before = possession.session.connector_mut().attempts;

    possession
        .execute(CockpitRequest::PlayFeedback {
            feedback: FeedbackKind::Ok,
        })
        .unwrap();

    assert_eq!(
        possession.session.connector_mut().attempts,
        attempts_before + 2
    );
    assert!(possession.snapshot().possessed);
}

#[test]
fn safe_cockpit_uses_possessions_single_motion_heartbeat() {
    let cockpit = BusyOnceCockpit {
        inner: SimCockpit::new(),
        busy_remaining: 0,
        attempts: 0,
        heartbeat_attempts: 0,
        cmd_vel_attempts: 0,
        last_heartbeat_timeout_ms: None,
        last_bump_escape: None,
    };
    let ready = establish_session(cockpit, hello(), None).unwrap();
    let possession = MotherbrainPossession::acquire(ready, 60_000).unwrap();
    let mut safe = SafeCockpit::new(possession);

    safe.pulse_motion(20, 0).unwrap();

    let connector = safe.client_mut().session.connector_mut();
    assert_eq!(connector.heartbeat_attempts, 1);
    assert_eq!(connector.cmd_vel_attempts, 1);
}

#[test]
fn production_possession_renews_and_retries_stale_control_lease() {
    let cockpit = StaleLeaseOnceCockpit {
        inner: SimCockpit::new(),
        invalid_remaining: 0,
        attempts: 0,
    };
    let ready = establish_session(cockpit, hello(), None).unwrap();
    let mut possession = MotherbrainPossession::acquire(ready, 60_000).unwrap();
    possession.session.connector_mut().invalid_remaining = 1;
    let first = possession.snapshot();

    possession
        .execute(CockpitRequest::PlayFeedback {
            feedback: FeedbackKind::Ok,
        })
        .unwrap();

    let renewed = possession.snapshot();
    assert_eq!(possession.session.connector_mut().attempts, 2);
    assert!(renewed.possessed);
    assert!(renewed.lease_generation > first.lease_generation);
    assert_ne!(renewed.lease_id, first.lease_id);
}

#[test]
fn closed_possession_motor_gate_allows_estop_reset_and_imu_zeroing() {
    let ready = establish_session(SimCockpit::new(), hello(), None).unwrap();
    let mut possession = MotherbrainPossession::acquire(ready, 60_000).unwrap();

    expect_accepted(possession.execute(CockpitRequest::EStop).unwrap()).unwrap();
    let status = match possession.execute(CockpitRequest::GetStatus).unwrap() {
        CockpitResponse::Status(status) => status.summary(),
        other => panic!("{other:?}"),
    };
    assert_eq!(status.estop_latched, Some(true));
    assert!(!possession.snapshot().possessed);

    assert!(matches!(
        possession.execute(CockpitRequest::CmdVel {
            linear_mm_s: 10,
            angular_mrad_s: 0,
            ttl_ms: 100,
        }),
        Err(CockpitError::Policy(_))
    ));

    expect_accepted(
        possession
            .execute(CockpitRequest::ZeroImuOrientation)
            .unwrap(),
    )
    .unwrap();
    let status = match possession.execute(CockpitRequest::GetStatus).unwrap() {
        CockpitResponse::Status(status) => status.summary(),
        other => panic!("{other:?}"),
    };
    assert_eq!(status.imu.calibration.as_deref(), Some("3"));

    expect_accepted(
        possession
            .execute(CockpitRequest::ClearImuOrientation)
            .unwrap(),
    )
    .unwrap();
    expect_accepted(possession.execute(CockpitRequest::ClearEStop).unwrap()).unwrap();
    let status = match possession.execute(CockpitRequest::GetStatus).unwrap() {
        CockpitResponse::Status(status) => status.summary(),
        other => panic!("{other:?}"),
    };
    assert_eq!(status.imu.calibration.as_deref(), Some("0"));
    assert_eq!(status.estop_latched, Some(false));
    assert!(possession.snapshot().possessed);
}

#[test]
fn exorcize_closes_gate_only_after_stop_is_acknowledged() {
    let cockpit = StopRejectingCockpit {
        inner: SimCockpit::new(),
        reject_stop: false,
        disarm_requests: 0,
    };
    let ready = establish_session(cockpit, hello(), None).unwrap();
    let mut possession = MotherbrainPossession::acquire(ready, 1_000).unwrap();
    possession.session.connector_mut().reject_stop = true;

    assert!(possession.exorcize().is_err());
    assert!(!possession.snapshot().possessed);
    assert_eq!(possession.session.connector_mut().disarm_requests, 0);
}

#[test]
fn exorcize_stops_without_disarming_create_oi() {
    let cockpit = StopRejectingCockpit {
        inner: SimCockpit::new(),
        reject_stop: false,
        disarm_requests: 0,
    };
    let ready = establish_session(cockpit, hello(), None).unwrap();
    let mut possession = MotherbrainPossession::acquire(ready, 1_000).unwrap();

    possession.exorcize().unwrap();

    assert!(!possession.snapshot().possessed);
    assert_eq!(possession.session.connector_mut().disarm_requests, 0);
}

#[test]
fn renewal_failure_closes_motor_gate() {
    let ready = establish_session(SimCockpit::new(), hello(), None).unwrap();
    let mut possession = MotherbrainPossession::acquire(ready, 1_000).unwrap();
    possession.lease_acquired_at = Instant::now() - Duration::from_millis(800);
    possession
        .session
        .connector_mut()
        .handshake(hello().new_attempt())
        .unwrap();
    assert!(possession.maintain().is_err());
    assert!(!possession.snapshot().possessed);
    assert!(possession
        .execute(CockpitRequest::CmdVel {
            linear_mm_s: 1,
            angular_mrad_s: 0,
            ttl_ms: 100,
        })
        .is_err());
}

#[test]
fn production_possession_clamps_motion_and_hides_oi() {
    let ready = establish_session(SimCockpit::new(), hello(), None).unwrap();
    let mut possession = MotherbrainPossession::acquire(ready, 1_000).unwrap();
    possession
        .execute(CockpitRequest::CmdVel {
            linear_mm_s: 500,
            angular_mrad_s: 5_000,
            ttl_ms: 10_000,
        })
        .unwrap();
    let events = possession.session.poll_events().unwrap();
    let motion = events
        .events
        .iter()
        .find(|event| event.kind == CockpitEventKind::MotionRequested)
        .unwrap();
    assert_eq!(motion.a, pack_i16_pair(50, 500));
    assert_eq!(motion.b, 300);
    assert!(possession
        .execute(CockpitRequest::SetMode {
            mode: CreateOiMode::Full,
        })
        .is_err());
}
