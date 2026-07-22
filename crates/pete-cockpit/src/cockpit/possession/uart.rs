pub const DEFAULT_UART_BAUD_RATE: u32 = 115_200;
pub const DEFAULT_UART_TIMEOUT: Duration = Duration::from_millis(750);
pub const DEFAULT_UART_MAX_RESPONSE_LEN: usize = MAX_COMPACT_HANDSHAKE_FRAME_LEN;

/// Enumerate stable Linux serial symlinks. Callers may further filter by USB
/// VID/PID or a configured identity hint, but must confirm identity by
/// handshake rather than trusting the filename.
pub fn discover_usb_serial_by_id() -> Result<Vec<PathBuf>> {
    let directory = Path::new("/dev/serial/by-id");
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut paths = std::fs::read_dir(directory)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct UartCockpitConfig {
    pub path: PathBuf,
    pub baud_rate: u32,
    pub timeout: Duration,
    pub max_response_len: usize,
    /// Assert DTR after opening. USB CDC firmware commonly uses DTR as its
    /// indication that a host is ready; hardware UART adapters need not.
    pub data_terminal_ready: bool,
}

impl UartCockpitConfig {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            baud_rate: DEFAULT_UART_BAUD_RATE,
            timeout: DEFAULT_UART_TIMEOUT,
            max_response_len: DEFAULT_UART_MAX_RESPONSE_LEN,
            data_terminal_ready: false,
        }
    }

    pub fn with_baud_rate(mut self, baud_rate: u32) -> Self {
        self.baud_rate = baud_rate;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_max_response_len(mut self, max_response_len: usize) -> Self {
        self.max_response_len = max_response_len;
        self
    }

    pub fn with_data_terminal_ready(mut self, ready: bool) -> Self {
        self.data_terminal_ready = ready;
        self
    }
}

pub struct UartCockpit {
    port: Box<dyn SerialPort>,
    next_seq: u32,
    timeout: Duration,
    max_response_len: usize,
    active_session_id: Option<String>,
}

impl UartCockpit {
    pub fn connect(path: impl AsRef<Path>) -> Result<Self> {
        Self::connect_with_config(UartCockpitConfig::new(path.as_ref()))
    }

    pub fn connect_with_config(config: UartCockpitConfig) -> Result<Self> {
        ensure_physical_actuator_transport_allowed("uart")?;
        let mut port = serialport::new(config.path.to_string_lossy(), config.baud_rate)
            .timeout(config.timeout)
            .open()?;
        if config.data_terminal_ready {
            port.write_data_terminal_ready(true)?;
        }
        Ok(Self {
            port,
            next_seq: 1,
            timeout: config.timeout,
            max_response_len: config.max_response_len,
            active_session_id: None,
        })
    }

    pub fn from_port(port: Box<dyn SerialPort>) -> Self {
        Self {
            port,
            next_seq: 1,
            timeout: DEFAULT_UART_TIMEOUT,
            max_response_len: DEFAULT_UART_MAX_RESPONSE_LEN,
            active_session_id: None,
        }
    }

    pub fn set_timeout(&mut self, timeout: Duration) -> Result<()> {
        self.timeout = timeout;
        self.port.set_timeout(timeout)?;
        Ok(())
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    fn request(&mut self, line: String) -> Result<String> {
        self.port.write_all(line.as_bytes())?;
        self.port.flush()?;
        read_line_response(&mut self.port, self.max_response_len)
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

impl Cockpit for UartCockpit {
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

fn read_line_response(port: &mut Box<dyn SerialPort>, max_len: usize) -> Result<String> {
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        match port.read(&mut byte) {
            Ok(0) => continue,
            Ok(_) if byte[0] == b'\n' => return response_from_bytes(&buf),
            Ok(_) if byte[0] == b'\r' => continue,
            Ok(_) => {
                if buf.len() >= max_len {
                    return Err(CockpitError::BadResponse(
                        "response line exceeded maximum length".into(),
                    ));
                }
                buf.push(byte[0]);
            }
            Err(error) => return Err(error.into()),
        }
    }
}

fn response_from_bytes(bytes: &[u8]) -> Result<String> {
    let response = std::str::from_utf8(bytes)
        .map_err(|_| CockpitError::BadResponse("response was not utf-8".into()))?
        .trim()
        .to_owned();
    Ok(response)
}

fn ensure_connector_session(active: &Option<String>, session: &CockpitSession) -> Result<()> {
    if active.as_deref() == Some(session.session_id.as_str()) {
        Ok(())
    } else {
        Err(CockpitError::InvalidSession {
            session_id: session.session_id.clone(),
        })
    }
}

fn expect_ok(seq: u32, response: &str) -> Result<()> {
    let mut parts = response.split_ascii_whitespace();
    match (
        parts.next(),
        parts.next().and_then(|value| value.parse::<u32>().ok()),
        parts.next(),
    ) {
        (Some("OK"), Some(response_seq), _) if response_seq == seq => Ok(()),
        (Some("ERR"), Some(response_seq), Some(reason))
            if response_seq == seq && is_compact_rejection_reason(reason) =>
        {
            Err(CockpitError::Rejected {
                command_id: seq,
                reason: reason.to_owned(),
            })
        }
        _ => Err(CockpitError::BadResponse(response.to_owned())),
    }
}

fn is_compact_rejection_reason(reason: &str) -> bool {
    matches!(
        reason,
        "busy"
            | "charging_busy"
            | "stale_sequence"
            | "unsupported"
            | "invalid_session"
            | "session_required"
            | "invalid_control_lease"
            | "control_lease_required"
            | "invalid_service_lease"
            | "service_authorization_required"
            | "service_operation_disabled"
    )
}

fn parse_capabilities(seq: u32, response: &str) -> Result<CockpitCapabilities> {
    expect_ok(seq, response)?;
    let rest = response
        .strip_prefix(&format!("OK {seq} CAPABILITIES "))
        .ok_or_else(|| CockpitError::BadResponse(response.to_owned()))?;
    Ok(CockpitCapabilities {
        body_kind: value_for(rest, "body_kind").unwrap_or_default().to_owned(),
        drive: value_for(rest, "drive").unwrap_or_default().to_owned(),
        verbs: csv_for(rest, "verbs"),
        sensors: csv_for(rest, "sensors"),
        outputs: csv_for(rest, "outputs"),
        safety: csv_for(rest, "safety"),
        events: csv_for(rest, "events"),
        independent_watchdog: value_for(rest, "independent_watchdog")
            .and_then(|value| value.parse().ok()),
        limits: parse_compact_limits(rest),
    })
}

fn parse_events(seq: u32, since_seq: u32, response: &str) -> Result<EventBatch> {
    expect_ok(seq, response)?;
    let rest = response
        .strip_prefix(&format!("OK {seq} EVENTS "))
        .ok_or_else(|| CockpitError::BadResponse(response.to_owned()))?;
    let header = rest.split('|').next().unwrap_or(rest);
    let dropped_before_seq = number_for(header, "dropped_before").unwrap_or(0);
    let mut batch = EventBatch {
        since_seq,
        oldest_seq: number_for(header, "oldest").unwrap_or(0),
        next_seq: number_for(header, "next").unwrap_or(since_seq),
        dropped_before_seq,
        events: Vec::new(),
    };
    let mut parsed_count = 0usize;
    for chunk in rest.split('|').skip(1) {
        let chunk = chunk.trim();
        let Some((seq_text, tail)) = chunk.split_once(':') else {
            return Err(CockpitError::BadResponse(response.to_owned()));
        };
        let Some((kind_text, fields)) = tail.split_once(':') else {
            return Err(CockpitError::BadResponse(response.to_owned()));
        };
        let mut nums = fields.split(',');
        let event_seq = seq_text
            .parse()
            .map_err(|_| CockpitError::BadResponse(response.to_owned()))?;
        let a = nums
            .next()
            .ok_or_else(|| CockpitError::BadResponse(response.to_owned()))?
            .parse()
            .map_err(|_| CockpitError::BadResponse(response.to_owned()))?;
        let b = nums
            .next()
            .ok_or_else(|| CockpitError::BadResponse(response.to_owned()))?
            .parse()
            .map_err(|_| CockpitError::BadResponse(response.to_owned()))?;
        let c = nums
            .next()
            .ok_or_else(|| CockpitError::BadResponse(response.to_owned()))?
            .parse()
            .map_err(|_| CockpitError::BadResponse(response.to_owned()))?;
        if nums.next().is_some() {
            return Err(CockpitError::BadResponse(response.to_owned()));
        }
        batch.events.push(CockpitEvent {
            seq: event_seq,
            kind: CockpitEventKind::from(kind_text),
            a,
            b,
            c,
        });
        parsed_count += 1;
    }
    if number_for(header, "count").is_some_and(|count| count as usize != parsed_count) {
        return Err(CockpitError::BadResponse(response.to_owned()));
    }
    Ok(batch)
}

fn parse_compact_cockpit_response(
    seq: u32,
    request: &CockpitRequest,
    response: &str,
) -> Result<CockpitResponse> {
    match request {
        CockpitRequest::GetStatus => {
            expect_ok(seq, response)?;
            Ok(CockpitResponse::Status(CockpitStatus {
                raw: response.to_owned(),
            }))
        }
        CockpitRequest::GetCapabilities => Ok(CockpitResponse::Capabilities(parse_capabilities(
            seq, response,
        )?)),
        CockpitRequest::GetEvents { since_seq } => Ok(CockpitResponse::Events(parse_events(
            seq, *since_seq, response,
        )?)),
        CockpitRequest::RegisterNetworkEndpoint(_) => {
            expect_ok(seq, response)?;
            Ok(CockpitResponse::NetworkEndpointRegistered(
                NetworkEndpointRegistered {
                    session_id: value_for(response, "session_id")
                        .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                        .into(),
                    fqdn: value_for(response, "fqdn")
                        .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                        .into(),
                    address: value_for(response, "address")
                        .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                        .into(),
                    ttl_seconds: number_for(response, "ttl")
                        .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                    registration_generation: number_for(response, "generation")
                        .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                },
            ))
        }
        CockpitRequest::AcquireControlLease { .. } => {
            expect_ok(seq, response)?;
            let owner_role = serde_json::from_value(serde_json::Value::String(
                value_for(response, "owner_role")
                    .unwrap_or("unknown")
                    .into(),
            ))?;
            let authority = serde_json::from_value(serde_json::Value::String(
                value_for(response, "authority").unwrap_or("unknown").into(),
            ))?;
            Ok(CockpitResponse::ControlLeaseGranted(ControlLease {
                lease_id: value_for(response, "lease_id")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                session_id: value_for(response, "session_id")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                owner_role,
                authority,
                ttl_ms: number_for(response, "ttl_ms")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                generation: number_for(response, "generation")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
            }))
        }
        CockpitRequest::AcquireServiceLease { .. } => {
            expect_ok(seq, response)?;
            Ok(CockpitResponse::ServiceLeaseGranted(ServiceLease {
                lease_id: value_for(response, "lease_id")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                session_id: value_for(response, "session_id")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                owner_role: serde_json::from_value(serde_json::Value::String(
                    value_for(response, "owner_role")
                        .unwrap_or("unknown")
                        .into(),
                ))?,
                scope: serde_json::from_value(serde_json::Value::String(
                    value_for(response, "scope").unwrap_or("unknown").into(),
                ))?,
                ttl_ms: number_for(response, "ttl_ms")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                generation: number_for(response, "generation")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
            }))
        }
        _ => {
            expect_ok(seq, response)?;
            Ok(CockpitResponse::Accepted)
        }
    }
}

fn parse_json_cockpit_response(
    command_id: u32,
    request: &CockpitRequest,
    response: &str,
) -> Result<CockpitResponse> {
    let value: serde_json::Value = serde_json::from_str(response.trim())?;
    if value.get("accepted").and_then(serde_json::Value::as_bool) == Some(false) {
        let reason = json_str_value(&value, "message")
            .or_else(|| json_str_value(&value, "reason"))
            .unwrap_or("rejected")
            .to_owned();
        return Err(CockpitError::Rejected { command_id, reason });
    }

    match request {
        CockpitRequest::GetStatus => Ok(CockpitResponse::Status(CockpitStatus {
            raw: response.trim().to_owned(),
        })),
        CockpitRequest::GetCapabilities => Ok(CockpitResponse::Capabilities(
            parse_json_capabilities(&value)?,
        )),
        CockpitRequest::GetEvents { since_seq } => Ok(CockpitResponse::Events(parse_json_events(
            *since_seq, &value,
        )?)),
        CockpitRequest::RegisterNetworkEndpoint(_) => {
            let registered = NetworkEndpointRegistered {
                session_id: json_str_value(&value, "session_id")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                fqdn: json_str_value(&value, "fqdn")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                address: json_str_value(&value, "address")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                ttl_seconds: json_u32_value(&value, "ttl_seconds")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                registration_generation: json_u32_value(&value, "registration_generation")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
            };
            Ok(CockpitResponse::NetworkEndpointRegistered(registered))
        }
        CockpitRequest::AcquireControlLease { .. } => {
            Ok(CockpitResponse::ControlLeaseGranted(ControlLease {
                lease_id: json_str_value(&value, "lease_id")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                session_id: json_str_value(&value, "session_id")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                owner_role: serde_json::from_value(
                    value
                        .get("owner_role")
                        .cloned()
                        .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                )?,
                authority: serde_json::from_value(
                    value
                        .get("authority")
                        .cloned()
                        .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                )?,
                ttl_ms: json_u32_value(&value, "ttl_ms")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                generation: json_u32_value(&value, "generation")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
            }))
        }
        CockpitRequest::AcquireServiceLease { .. } => {
            Ok(CockpitResponse::ServiceLeaseGranted(ServiceLease {
                lease_id: json_str_value(&value, "lease_id")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                session_id: json_str_value(&value, "session_id")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?
                    .into(),
                owner_role: serde_json::from_value(
                    value
                        .get("owner_role")
                        .cloned()
                        .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                )?,
                scope: serde_json::from_value(
                    value
                        .get("scope")
                        .cloned()
                        .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                )?,
                ttl_ms: json_u32_value(&value, "ttl_ms")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
                generation: json_u32_value(&value, "generation")
                    .ok_or_else(|| CockpitError::BadResponse(response.into()))?,
            }))
        }
        _ => {
            if value.get("accepted").and_then(serde_json::Value::as_bool) == Some(true) {
                Ok(CockpitResponse::Accepted)
            } else {
                Err(CockpitError::BadResponse(response.to_owned()))
            }
        }
    }
}

fn parse_json_capabilities(value: &serde_json::Value) -> Result<CockpitCapabilities> {
    Ok(CockpitCapabilities {
        body_kind: json_str_value(value, "body_kind")
            .unwrap_or_default()
            .to_owned(),
        drive: json_str_value(value, "drive")
            .unwrap_or_default()
            .to_owned(),
        verbs: json_string_array(value, "verbs"),
        sensors: json_string_array(value, "sensors"),
        outputs: json_string_array(value, "outputs"),
        safety: json_string_array(value, "safety"),
        events: json_string_array(value, "events"),
        independent_watchdog: value
            .get("independent_watchdog")
            .and_then(serde_json::Value::as_bool),
        limits: parse_json_limits(value),
    })
}

fn parse_compact_limits(line: &str) -> CockpitLimits {
    let Some(raw) = value_for(line, "limits") else {
        return CockpitLimits::default();
    };
    let mut limits = CockpitLimits::default();
    for item in raw.split(',') {
        let Some((key, value)) = item.split_once(':') else {
            continue;
        };
        match key {
            "max_linear_mm_s" => {
                if let Ok(value) = value.parse() {
                    limits.max_linear_mm_s = value;
                }
            }
            "max_angular_mrad_s" => {
                if let Ok(value) = value.parse() {
                    limits.max_angular_mrad_s = value;
                }
            }
            "min_ttl_ms" => {
                if let Ok(value) = value.parse() {
                    limits.min_ttl_ms = value;
                }
            }
            "max_ttl_ms" => {
                if let Ok(value) = value.parse() {
                    limits.max_ttl_ms = value;
                }
            }
            _ => {}
        }
    }
    limits
}

fn parse_json_limits(value: &serde_json::Value) -> CockpitLimits {
    let Some(limits) = value.get("limits") else {
        return CockpitLimits::default();
    };
    CockpitLimits {
        max_linear_mm_s: json_i16_value(limits, "max_linear_mm_s").unwrap_or(i16::MAX),
        max_angular_mrad_s: json_i16_value(limits, "max_angular_mrad_s").unwrap_or(i16::MAX),
        min_ttl_ms: json_u32_value(limits, "min_ttl_ms").unwrap_or(1),
        max_ttl_ms: json_u32_value(limits, "max_ttl_ms").unwrap_or(u32::MAX),
    }
}

fn parse_json_events(since_seq: u32, value: &serde_json::Value) -> Result<EventBatch> {
    let events = value
        .get("events")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| CockpitError::BadResponse(value.to_string()))?
        .iter()
        .map(|event| {
            Ok(CockpitEvent {
                seq: json_u32_value(event, "seq")
                    .ok_or_else(|| CockpitError::BadResponse(event.to_string()))?,
                kind: CockpitEventKind::from(json_str_value(event, "kind").unwrap_or("unknown")),
                a: json_u32_value(event, "a").unwrap_or(0),
                b: json_u32_value(event, "b").unwrap_or(0),
                c: json_u32_value(event, "c").unwrap_or(0),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(EventBatch {
        since_seq,
        oldest_seq: json_u32_value(value, "oldest_seq").unwrap_or(0),
        next_seq: json_u32_value(value, "next_seq").unwrap_or(since_seq),
        dropped_before_seq: json_u32_value(value, "dropped_before_seq").unwrap_or(0),
        events,
    })
}

fn json_string_array(value: &serde_json::Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
        .collect()
}

fn http_body(response: &str) -> Result<String> {
    let (head, body) = response
        .split_once("\r\n\r\n")
        .ok_or_else(|| CockpitError::BadResponse(response.to_owned()))?;
    let protocol_rejection = (head.starts_with("HTTP/1.1 409")
        || head.starts_with("HTTP/1.0 409")
        || head.starts_with("HTTP/1.1 422")
        || head.starts_with("HTTP/1.0 422"))
        && body.trim_start().starts_with('{');
    if !head.starts_with("HTTP/1.1 200") && !head.starts_with("HTTP/1.0 200") && !protocol_rejection
    {
        return Err(CockpitError::BadResponse(head.to_owned()));
    }
    Ok(body.trim().to_owned())
}

fn value_for<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{key}=");
    let start = line.find(&prefix)? + prefix.len();
    let tail = &line[start..];
    Some(tail.split_whitespace().next().unwrap_or(tail))
}

fn csv_for(line: &str, key: &str) -> Vec<String> {
    value_for(line, key)
        .unwrap_or("")
        .split(',')
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn number_for(line: &str, key: &str) -> Option<u32> {
    value_for(line, key)?.parse().ok()
}

fn signed_number_for(line: &str, key: &str) -> Option<i32> {
    value_for(line, key)?.parse().ok()
}

fn bool_for(line: &str, key: &str) -> Option<bool> {
    match value_for(line, key)? {
        "true" | "1" | "on" | "yes" => Some(true),
        "false" | "0" | "off" | "no" => Some(false),
        _ => None,
    }
}

fn json_str_value<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    value.get(key)?.as_str()
}

fn json_bool_value(value: &serde_json::Value, key: &str) -> Option<bool> {
    value.get(key)?.as_bool()
}

fn json_tri_state_value(value: &serde_json::Value, key: &str) -> Option<bool> {
    match value.get(key)? {
        serde_json::Value::Bool(value) => Some(*value),
        serde_json::Value::String(value) => match value.as_str() {
            "true" | "1" | "on" | "yes" => Some(true),
            "false" | "0" | "off" | "no" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn json_u32_value(value: &serde_json::Value, key: &str) -> Option<u32> {
    value
        .get(key)?
        .as_u64()
        .and_then(|value| value.try_into().ok())
}

fn json_i32_value(value: &serde_json::Value, key: &str) -> Option<i32> {
    value
        .get(key)?
        .as_i64()
        .and_then(|value| value.try_into().ok())
}

fn json_i16_value(value: &serde_json::Value, key: &str) -> Option<i16> {
    value
        .get(key)?
        .as_i64()
        .and_then(|value| value.try_into().ok())
}

fn compact_tones(tones: &[SongTone]) -> String {
    let mut encoded = String::new();
    for tone in tones {
        encoded.push_str(&format!(" {} {}", tone.note, tone.duration_64ths));
    }
    encoded
}

fn rewrite_for_firmware_json(
    request: &CockpitRequest,
    object: &mut serde_json::Map<String, serde_json::Value>,
) {
    match request {
        CockpitRequest::DefineChirp { tones, .. } | CockpitRequest::SongDefine { tones, .. } => {
            object.insert(
                "tones".to_owned(),
                tones
                    .iter()
                    .map(|tone| format!("{}:{}", tone.note, tone.duration_64ths))
                    .collect::<Vec<_>>()
                    .join(",")
                    .into(),
            );
        }
        CockpitRequest::SetSafetyPolicy { policy } => {
            object.remove("policy");
            object.insert("bump_action".to_owned(), policy.bump.as_str().into());
            object.insert("cliff_action".to_owned(), policy.cliff.as_str().into());
            object.insert(
                "wheel_drop_latch".to_owned(),
                policy.wheel_drop_latch.into(),
            );
        }
        _ => {}
    }
}

fn pack_i16_pair(left: i16, right: i16) -> u32 {
    ((left as u16 as u32) << 16) | right as u16 as u32
}

fn time_reached(now_ms: u32, deadline_ms: u32) -> bool {
    now_ms.wrapping_sub(deadline_ms) < u32::MAX / 2
}

fn fresh_sim_boot_id() -> String {
    format!("simboot-{}", uuid::Uuid::new_v4().simple())
}
