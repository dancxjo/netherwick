pub struct HttpCockpit {
    host: String,
    next_command_id: u32,
    timeout: Duration,
    active_session_id: Option<String>,
}

impl HttpCockpit {
    pub fn connect(host: impl Into<String>) -> Self {
        Self {
            host: host.into(),
            next_command_id: 1,
            timeout: Duration::from_millis(750),
            active_session_id: None,
        }
    }

    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    fn command_id(&mut self) -> u32 {
        let command_id = self.next_command_id;
        self.next_command_id = self.next_command_id.wrapping_add(1).max(1);
        command_id
    }

    fn post(&mut self, path: &str, body: &str) -> Result<String> {
        let addr = self
            .host
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| CockpitError::BadResponse("http host did not resolve".to_owned()))?;
        let mut stream = TcpStream::connect_timeout(&addr, self.timeout)?;
        stream.set_read_timeout(Some(self.timeout))?;
        stream.set_write_timeout(Some(self.timeout))?;
        write!(
            stream,
            "POST {path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            self.host,
            body.len(),
            body
        )?;
        stream.flush()?;
        let mut response = String::new();
        let mut bytes = [0u8; 1024];
        loop {
            match stream.read(&mut bytes) {
                Ok(0) => break,
                Ok(len) => response.push_str(&String::from_utf8_lossy(&bytes[..len])),
                Err(error)
                    if !response.is_empty()
                        && matches!(
                            error.kind(),
                            std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                        ) =>
                {
                    break;
                }
                Err(error) => return Err(error.into()),
            }
        }
        http_body(&response)
    }
}

impl Cockpit for HttpCockpit {
    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        let command_id = self.command_id();
        let body = request.to_firmware_json(command_id)?;
        let response = self.post("/command", &body)?;
        parse_json_cockpit_response(command_id, &request, &response)
    }

    fn handshake(&mut self, hello: HandshakeHello) -> Result<HandshakeOutcome> {
        let body = serde_json::to_string(&hello)?;
        let response = self.post("/handshake", &body)?;
        let outcome = HandshakeOutcome::validate(&hello, serde_json::from_str(&response)?)?;
        self.active_session_id = Some(outcome.session.session_id.clone());
        Ok(outcome)
    }

    fn execute_in_session(
        &mut self,
        session: &CockpitSession,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let command_id = self.command_id();
        let body = request.to_firmware_json_with_session(command_id, &session.session_id)?;
        let response = self.post("/command", &body)?;
        parse_json_cockpit_response(command_id, &request, &response)
    }

    fn execute_with_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ControlLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let command_id = self.command_id();
        let body = request.to_firmware_json_with_authority(
            command_id,
            &session.session_id,
            &lease.lease_id,
        )?;
        let response = self.post("/command", &body)?;
        parse_json_cockpit_response(command_id, &request, &response)
    }

    fn execute_with_service_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ServiceLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let command_id = self.command_id();
        let body = request.to_firmware_json_with_service_authority(
            command_id,
            &session.session_id,
            &lease.lease_id,
        )?;
        let response = self.post("/command", &body)?;
        parse_json_cockpit_response(command_id, &request, &response)
    }
}

pub struct WebSocketCockpit {
    socket: WebSocket<MaybeTlsStream<TcpStream>>,
    next_command_id: u32,
    active_session_id: Option<String>,
}

impl WebSocketCockpit {
    pub fn connect_url(url: &str) -> Result<Self> {
        let (socket, _) = connect(url)?;
        Ok(Self {
            socket,
            next_command_id: 1,
            active_session_id: None,
        })
    }

    pub fn connect_pico_w(host: &str) -> Result<Self> {
        Self::connect_url(&format!("ws://{host}:81/control"))
    }

    fn command_id(&mut self) -> u32 {
        let command_id = self.next_command_id;
        self.next_command_id = self.next_command_id.wrapping_add(1).max(1);
        command_id
    }
}

impl Cockpit for WebSocketCockpit {
    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        let command_id = self.command_id();
        let body = request.to_firmware_json(command_id)?;
        self.socket.send(Message::Text(body.into()))?;
        loop {
            let message = self.socket.read()?;
            match message {
                Message::Text(text) => {
                    return parse_json_cockpit_response(command_id, &request, text.as_str());
                }
                Message::Binary(bytes) => {
                    let text = response_from_bytes(&bytes)?;
                    return parse_json_cockpit_response(command_id, &request, &text);
                }
                Message::Ping(bytes) => self.socket.send(Message::Pong(bytes))?,
                Message::Close(_) => {
                    return Err(CockpitError::BadResponse(
                        "websocket closed before response".to_owned(),
                    ));
                }
                _ => {}
            }
        }
    }

    fn handshake(&mut self, hello: HandshakeHello) -> Result<HandshakeOutcome> {
        let mut value = serde_json::to_value(&hello)?;
        value
            .as_object_mut()
            .expect("hello serializes as object")
            .insert("kind".into(), serde_json::Value::String("hello".into()));
        self.socket
            .send(Message::Text(serde_json::to_string(&value)?.into()))?;
        loop {
            match self.socket.read()? {
                Message::Text(text) => {
                    let outcome =
                        HandshakeOutcome::validate(&hello, serde_json::from_str(text.as_str())?)?;
                    self.active_session_id = Some(outcome.session.session_id.clone());
                    return Ok(outcome);
                }
                Message::Binary(bytes) => {
                    let outcome =
                        HandshakeOutcome::validate(&hello, serde_json::from_slice(&bytes)?)?;
                    self.active_session_id = Some(outcome.session.session_id.clone());
                    return Ok(outcome);
                }
                Message::Ping(bytes) => self.socket.send(Message::Pong(bytes))?,
                Message::Close(_) => {
                    return Err(CockpitError::BadResponse(
                        "websocket closed during handshake".into(),
                    ))
                }
                _ => {}
            }
        }
    }

    fn execute_in_session(
        &mut self,
        session: &CockpitSession,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let command_id = self.command_id();
        let body = request.to_firmware_json_with_session(command_id, &session.session_id)?;
        self.socket.send(Message::Text(body.into()))?;
        loop {
            match self.socket.read()? {
                Message::Text(text) => {
                    return parse_json_cockpit_response(command_id, &request, text.as_str())
                }
                Message::Binary(bytes) => {
                    return parse_json_cockpit_response(
                        command_id,
                        &request,
                        &response_from_bytes(&bytes)?,
                    )
                }
                Message::Ping(bytes) => self.socket.send(Message::Pong(bytes))?,
                Message::Close(_) => {
                    return Err(CockpitError::BadResponse(
                        "websocket closed before response".into(),
                    ))
                }
                _ => {}
            }
        }
    }

    fn execute_with_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ControlLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let command_id = self.command_id();
        let body = request.to_firmware_json_with_authority(
            command_id,
            &session.session_id,
            &lease.lease_id,
        )?;
        self.socket.send(Message::Text(body.into()))?;
        loop {
            match self.socket.read()? {
                Message::Text(text) => {
                    return parse_json_cockpit_response(command_id, &request, text.as_str())
                }
                Message::Binary(bytes) => {
                    return parse_json_cockpit_response(
                        command_id,
                        &request,
                        &response_from_bytes(&bytes)?,
                    )
                }
                Message::Ping(bytes) => self.socket.send(Message::Pong(bytes))?,
                Message::Close(_) => {
                    return Err(CockpitError::BadResponse(
                        "websocket closed before response".into(),
                    ))
                }
                _ => {}
            }
        }
    }

    fn execute_with_service_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ServiceLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let command_id = self.command_id();
        let body = request.to_firmware_json_with_service_authority(
            command_id,
            &session.session_id,
            &lease.lease_id,
        )?;
        self.socket.send(Message::Text(body.into()))?;
        loop {
            match self.socket.read()? {
                Message::Text(text) => {
                    return parse_json_cockpit_response(command_id, &request, text.as_str())
                }
                Message::Binary(bytes) => {
                    return parse_json_cockpit_response(
                        command_id,
                        &request,
                        &response_from_bytes(&bytes)?,
                    )
                }
                Message::Ping(bytes) => self.socket.send(Message::Pong(bytes))?,
                Message::Close(_) => {
                    return Err(CockpitError::BadResponse(
                        "websocket closed before response".into(),
                    ))
                }
                _ => {}
            }
        }
    }
}

pub struct UdpCockpit {
    socket: UdpSocket,
    next_seq: u32,
    timeout: Duration,
    active_session_id: Option<String>,
}

impl UdpCockpit {
    pub fn connect(brainstem: SocketAddr) -> Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        let timeout = Duration::from_millis(750);
        socket.connect(brainstem)?;
        socket.set_read_timeout(Some(timeout))?;
        socket.set_write_timeout(Some(timeout))?;
        Ok(Self {
            socket,
            next_seq: 1,
            timeout,
            active_session_id: None,
        })
    }

    pub fn set_timeout(&mut self, timeout: Duration) -> Result<()> {
        self.timeout = timeout;
        self.socket.set_read_timeout(Some(timeout))?;
        self.socket.set_write_timeout(Some(timeout))?;
        Ok(())
    }

    fn request(&mut self, line: String) -> Result<String> {
        self.socket.send(line.as_bytes())?;
        let mut buf = [0u8; MAX_COMPACT_HANDSHAKE_FRAME_LEN];
        let len = self.socket.recv(&mut buf)?;
        response_from_bytes(&buf[..len])
    }

    fn seq(&mut self) -> u32 {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1).max(1);
        seq
    }

    fn command(&mut self, kind: &str) -> Result<()> {
        let seq = self.seq();
        expect_ok(seq, &self.request(format!("{kind} {seq}\n"))?)
    }
}

impl Cockpit for UdpCockpit {
    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        let seq = self.seq();
        let response = self.request(request.to_compact_line(seq))?;
        parse_compact_cockpit_response(seq, &request, &response)
    }

    fn handshake(&mut self, hello: HandshakeHello) -> Result<HandshakeOutcome> {
        let response = self.request(encode_compact_hello(&hello)?)?;
        let outcome = HandshakeOutcome::validate(&hello, decode_compact_response(&response)?)?;
        self.active_session_id = Some(outcome.session.session_id.clone());
        Ok(outcome)
    }

    fn execute_in_session(
        &mut self,
        session: &CockpitSession,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let seq = self.seq();
        let response =
            self.request(request.to_compact_line_with_session(seq, &session.session_id))?;
        parse_compact_cockpit_response(seq, &request, &response)
    }

    fn execute_with_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ControlLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let seq = self.seq();
        let response = self.request(request.to_compact_line_with_authority(
            seq,
            &session.session_id,
            &lease.lease_id,
        ))?;
        parse_compact_cockpit_response(seq, &request, &response)
    }

    fn execute_with_service_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ServiceLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        ensure_connector_session(&self.active_session_id, session)?;
        let seq = self.seq();
        let response = self.request(request.to_compact_line_with_service_authority(
            seq,
            &session.session_id,
            &lease.lease_id,
        ))?;
        parse_compact_cockpit_response(seq, &request, &response)
    }

    fn get_status(&mut self) -> Result<CockpitStatus> {
        let seq = self.seq();
        let response = self.request(format!("STATUS {seq}\n"))?;
        expect_ok(seq, &response)?;
        Ok(CockpitStatus { raw: response })
    }

    fn get_capabilities(&mut self) -> Result<CockpitCapabilities> {
        let seq = self.seq();
        let response = self.request(format!("GET_CAPABILITIES {seq}\n"))?;
        parse_capabilities(seq, &response)
    }

    fn get_events_since(&mut self, since_seq: u32) -> Result<EventBatch> {
        let seq = self.seq();
        let response = self.request(format!("GET_EVENTS {seq} {since_seq}\n"))?;
        parse_events(seq, since_seq, &response)
    }

    fn arm(&mut self) -> Result<()> {
        self.command("ARM")
    }

    fn disarm(&mut self) -> Result<()> {
        self.command("DISARM")
    }

    fn stop(&mut self) -> Result<()> {
        self.command("STOP")
    }

    fn estop(&mut self) -> Result<()> {
        self.command("ESTOP")
    }

    fn clear_estop(&mut self) -> Result<()> {
        self.command("CLEAR_ESTOP")
    }

    fn cmd_vel(&mut self, linear_mm_s: i16, angular_mrad_s: i16, ttl_ms: u32) -> Result<()> {
        let seq = self.seq();
        expect_ok(
            seq,
            &self.request(format!(
                "CMD_VEL {seq} {linear_mm_s} {angular_mrad_s} {ttl_ms}\n"
            ))?,
        )
    }

    fn heartbeat_stop(&mut self, timeout_ms: u32) -> Result<()> {
        let seq = self.seq();
        expect_ok(
            seq,
            &self.request(format!("HEARTBEAT_STOP {seq} {timeout_ms}\n"))?,
        )
    }

    fn stream_sensors(&mut self, enabled: bool, packet_id: u8, period_ms: u32) -> Result<()> {
        let seq = self.seq();
        let enabled = if enabled { "true" } else { "false" };
        expect_ok(
            seq,
            &self.request(format!(
                "STREAM_SENSORS {seq} {enabled} {packet_id} {period_ms}\n"
            ))?,
        )
    }

    fn reset_odometry(&mut self) -> Result<()> {
        self.command("RESET_ODOMETRY")
    }

    fn zero_imu_orientation(&mut self) -> Result<()> {
        self.command("ZERO_IMU_ORIENTATION")
    }

    fn clear_imu_orientation(&mut self) -> Result<()> {
        self.command("CLEAR_IMU_ORIENTATION")
    }
}
