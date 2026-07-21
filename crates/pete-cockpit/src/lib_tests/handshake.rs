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
    assert_eq!(
        CockpitRequest::SetAudioSilent { silent: true }.authorization_class(),
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
