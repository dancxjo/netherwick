/// Narrow handle exposing commands authorized by an installed control lease.
pub struct ControlCockpit<'a, C> {
    session: &'a mut SessionCockpit<C>,
}

impl<C: Cockpit> ControlCockpit<'_, C> {
    pub fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        if !matches!(
            request.authorization_class(),
            AuthorizationClass::ControlLease
                | AuthorizationClass::Emergency
                | AuthorizationClass::Session
        ) || matches!(
            request,
            CockpitRequest::AcquireControlLease { .. }
                | CockpitRequest::AcquireServiceLease { .. }
                | CockpitRequest::RegisterNetworkEndpoint(_)
        ) {
            return Err(CockpitError::Policy(
                "request is outside control-lease authority".into(),
            ));
        }
        self.session.execute(request)
    }

    pub fn cmd_vel(&mut self, linear_mm_s: i16, angular_mrad_s: i16, ttl_ms: u32) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::CmdVel {
            linear_mm_s,
            angular_mrad_s,
            ttl_ms,
        })?)
    }

    pub fn arm(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::Arm)?)
    }

    pub fn disarm(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::Disarm)?)
    }

    pub fn stop(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::Stop)?)
    }
}

/// Narrow handle exposing only separately authorized maintenance operations.
pub struct ServiceCockpit<'a, C> {
    session: &'a mut SessionCockpit<C>,
}

impl<C: Cockpit> ServiceCockpit<'_, C> {
    pub fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        if request.authorization_class() != AuthorizationClass::ServiceLease {
            return Err(CockpitError::Policy(
                "request is outside service-lease authority".into(),
            ));
        }
        self.session.execute(request)
    }

    pub fn restart_create(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::RestartCreate)?)
    }

    pub fn reset_motherbrain(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::ResetMotherbrain)?)
    }

    pub fn bootsel(&mut self) -> Result<()> {
        expect_accepted(self.execute(CockpitRequest::Bootsel)?)
    }
}

pub fn establish_session<C: Cockpit>(
    mut connector: C,
    hello: HandshakeHello,
    prior: Option<&CockpitSession>,
) -> Result<SessionCockpit<C>> {
    let outcome = connector.handshake(hello)?.classify_against(prior);
    outcome.contract.validate_event_vocabulary()?;
    let report = outcome.contract.validate_local_model();
    if !report.missing_verbs.is_empty() {
        return Err(CockpitError::Policy(format!(
            "live capability contract contains unsupported verbs: {}",
            report.missing_verbs.join(",")
        )));
    }
    if outcome.welcome.safety_snapshot.armed || outcome.welcome.safety_snapshot.active_motion {
        return Err(CockpitError::UnsafeHandshake(
            "brainstem is not ready-but-disarmed".into(),
        ));
    }
    let cursor = EventCursor::from_event_next_seq(outcome.event_cursor);
    Ok(SessionCockpit {
        connector,
        outcome,
        cursor,
        control_lease: None,
        service_lease: None,
    })
}

pub fn establish_diagnostic_session<C: Cockpit>(
    mut connector: C,
    mut hello: HandshakeHello,
    prior: Option<&CockpitSession>,
) -> Result<SessionCockpit<C>> {
    hello.session_purpose = SessionPurpose::Diagnostic;
    let outcome = connector.handshake(hello)?.classify_against(prior);
    outcome.contract.validate_event_vocabulary()?;
    let cursor = EventCursor::from_event_next_seq(outcome.event_cursor);
    Ok(SessionCockpit {
        connector,
        outcome,
        cursor,
        control_lease: None,
        service_lease: None,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplacementPolicy {
    Reject,
    Accept,
}

/// Owns exactly one command connector. Switching always performs a fresh
/// handshake; backup status probes should use a separate read-only connector.
pub struct FailoverCockpit {
    active: SessionCockpit<Box<dyn Cockpit>>,
}

impl FailoverCockpit {
    pub fn new(active: SessionCockpit<Box<dyn Cockpit>>) -> Self {
        Self { active }
    }
    pub fn active(&self) -> &SessionCockpit<Box<dyn Cockpit>> {
        &self.active
    }
    pub fn active_mut(&mut self) -> &mut SessionCockpit<Box<dyn Cockpit>> {
        &mut self.active
    }

    pub fn failover(
        &mut self,
        backup: Box<dyn Cockpit>,
        hello: HandshakeHello,
        replacement_policy: ReplacementPolicy,
    ) -> Result<()> {
        let prior = self.active.session().clone();
        let replacement = establish_session(backup, hello.new_attempt(), Some(&prior))?;
        if replacement.outcome().classification == ReconnectClassification::ReplacementBrainstem
            && replacement_policy == ReplacementPolicy::Reject
        {
            return Err(CockpitError::Policy(format!(
                "replacement brainstem {} requires explicit acceptance",
                replacement.session().peer_device_id
            )));
        }
        // Best effort on a failed physical lane. The new handshake already
        // synchronously stopped and disarmed the brainstem before WELCOME.
        let _ = self.active.execute(CockpitRequest::Stop);
        let _ = self.active.execute(CockpitRequest::Disarm);
        self.active = replacement;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MotherbrainBootstrap {
    pub hello: HandshakeHello,
    pub expected_brainstem_device_id: Option<String>,
}

#[derive(Debug, Error)]
pub enum BootstrapError {
    #[error("brainstem not found under /dev/serial/by-id")]
    BrainstemNotFound,
    #[error("serial open failed for {path}: {source}")]
    SerialOpenFailed { path: PathBuf, source: CockpitError },
    #[error("handshake failed on {path}: {source}")]
    HandshakeFailed { path: PathBuf, source: CockpitError },
    #[error("wrong brainstem identity: expected {expected}, received {received}")]
    WrongBrainstem { expected: String, received: String },
    #[error("unsafe welcome: {0}")]
    UnsafeWelcome(String),
    #[error("capability mismatch: {0}")]
    CapabilityMismatch(String),
    #[error("event history missed before sequence {dropped_before_seq}")]
    EventHistoryMissed { dropped_before_seq: u32 },
    #[error("brainstem network is not ready: {0}")]
    NetworkNotReady(String),
    #[error("network registration rejected: {0}")]
    NetworkRegistrationRejected(String),
    #[error("control lease rejected: {0}")]
    ControlLeaseRejected(String),
    #[error("DNS verification failed for {0}")]
    DnsVerificationFailed(String),
    #[error("all USB CDC candidates failed: {failures:?}")]
    CandidateFailures { failures: Vec<CandidateFailure> },
}

#[derive(Debug)]
pub struct CandidateFailure {
    pub path: PathBuf,
    pub cause: String,
}

impl MotherbrainBootstrap {
    pub fn from_host() -> Self {
        Self {
            hello: HandshakeHello::default_motherbrain(),
            expected_brainstem_device_id: None,
        }
    }

    /// USB CDC enumerates as a serial byte stream; `UartCockpit` is the shared
    /// line-protocol implementation, not an assertion that this is GPIO UART.
    pub fn connect_usb(
        &self,
    ) -> std::result::Result<SessionCockpit<Box<dyn Cockpit>>, BootstrapError> {
        let paths =
            discover_usb_serial_by_id().map_err(|error| BootstrapError::HandshakeFailed {
                path: PathBuf::from("/dev/serial/by-id"),
                source: error,
            })?;
        if paths.is_empty() {
            return Err(BootstrapError::BrainstemNotFound);
        }
        let mut errors = Vec::new();
        for path in paths {
            // Opening a Pico USB CDC endpoint and asserting DTR can race the
            // firmware's wait_connection loop, especially immediately after
            // flashing. Give it a short settle period and retry transient
            // open/handshake failures on this same pinned path.
            for attempt in 1..=3 {
                let connector = match UartCockpit::connect_with_config(
                    UartCockpitConfig::new(&path)
                        .with_timeout(Duration::from_secs(2))
                        .with_data_terminal_ready(true),
                ) {
                    Ok(connector) => connector,
                    Err(error) => {
                        errors.push(CandidateFailure {
                            path: path.clone(),
                            cause: format!(
                                "attempt {attempt}: {}",
                                BootstrapError::SerialOpenFailed {
                                    path: path.clone(),
                                    source: error,
                                }
                            ),
                        });
                        std::thread::sleep(Duration::from_millis(250));
                        continue;
                    }
                };
                std::thread::sleep(Duration::from_millis(250));
                match (|| -> Result<_> {
                    let ready = establish_session(
                        Box::new(connector) as Box<dyn Cockpit>,
                        self.hello.new_attempt(),
                        None,
                    )?;
                    self.validate_identity(ready.session())?;
                    Ok(ready)
                })() {
                    Ok(ready) => return Ok(ready),
                    Err(error) => errors.push(CandidateFailure {
                        path: path.clone(),
                        cause: format!("attempt {attempt}: {error}"),
                    }),
                }
                std::thread::sleep(Duration::from_millis(250));
            }
        }
        Err(BootstrapError::CandidateFailures { failures: errors })
    }

    pub fn connect_backup(
        &self,
        connector: Box<dyn Cockpit>,
        prior: &CockpitSession,
    ) -> Result<SessionCockpit<Box<dyn Cockpit>>> {
        let ready = establish_session(connector, self.hello.new_attempt(), Some(prior))?;
        self.validate_identity(ready.session())?;
        Ok(ready)
    }

    pub fn register_network(
        &self,
        ready: &mut SessionCockpit<Box<dyn Cockpit>>,
        endpoint: RegisterNetworkEndpoint,
    ) -> Result<NetworkEndpointRegistered> {
        match ready.execute(CockpitRequest::RegisterNetworkEndpoint(endpoint))? {
            CockpitResponse::NetworkEndpointRegistered(registered) => Ok(registered),
            response => Err(CockpitError::BadResponse(format!("{response:?}"))),
        }
    }

    fn validate_identity(&self, session: &CockpitSession) -> Result<()> {
        if self
            .expected_brainstem_device_id
            .as_deref()
            .is_some_and(|expected| expected != session.peer_device_id)
        {
            return Err(CockpitError::Policy(format!(
                "brainstem identity {} did not match configured {}",
                session.peer_device_id,
                self.expected_brainstem_device_id.as_deref().unwrap_or("")
            )));
        }
        Ok(())
    }
}
