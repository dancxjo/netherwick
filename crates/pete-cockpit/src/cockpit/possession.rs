pub struct EventCursor {
    next_seq: u32,
}

impl EventCursor {
    pub fn new() -> Self {
        Self { next_seq: 0 }
    }

    pub fn next_seq(&self) -> u32 {
        self.next_seq
    }

    pub fn poll<C: Cockpit>(&mut self, client: &mut C) -> Result<EventBatch> {
        let batch = client.get_events_since(self.next_seq)?;
        batch.ensure_no_missed_events()?;
        self.next_seq = batch.next_seq.saturating_sub(1);
        Ok(batch)
    }
}

impl Default for EventCursor {
    fn default() -> Self {
        Self::new()
    }
}

impl EventCursor {
    pub fn from_event_next_seq(next_seq: u32) -> Self {
        Self {
            next_seq: next_seq.saturating_sub(1),
        }
    }
}

/// A transport-neutral, validated command session. Construction is only
/// possible through `establish_session`, so capabilities and safety state came
/// from the live WELCOME rather than a cache.
pub struct SessionCockpit<C> {
    connector: C,
    outcome: HandshakeOutcome,
    cursor: EventCursor,
    control_lease: Option<ControlLease>,
    service_lease: Option<ServiceLease>,
}

/// Production control boundary for a motherbrain-owned brainstem session.
///
/// Possession is the live `Motherbrain` lease itself. There is deliberately no
/// second local "armed" state and no Create OI surface here: callers can send
/// bounded body intents, STOP/DISARM, and inspect brainstem telemetry only.
pub struct MotherbrainPossession<C: Cockpit> {
    session: SessionCockpit<C>,
    lease_acquired_at: Instant,
    lease_ttl_ms: u32,
    renew_margin_ms: u32,
    motion_ttl_ms: u32,
    heartbeat_timeout_ms: u32,
    max_linear_mm_s: i16,
    max_angular_mrad_s: i16,
    motor_gate_open: bool,
    refusal_reason: Option<String>,
    last_applied_at: Option<Instant>,
    last_applied_command: Option<MotorCommand>,
    last_status: Option<StatusSummary>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PossessionSnapshot {
    pub brainstem_device_id: String,
    pub brainstem_boot_id: String,
    pub session_id: String,
    pub lease_id: String,
    pub lease_generation: u32,
    pub lease_remaining_ms: u32,
    pub possessed: bool,
    pub moving: Option<bool>,
    pub brainstem_armed: Option<bool>,
    pub body_health: Option<String>,
    pub uart_health: Option<String>,
    pub safety_tripped: Option<bool>,
    pub estop_latched: Option<bool>,
    pub refusal_reason: Option<String>,
    pub last_applied_command_age_ms: Option<u64>,
    pub last_applied_command: Option<MotorCommand>,
}

impl<C: Cockpit> MotherbrainPossession<C> {
    pub fn acquire(mut session: SessionCockpit<C>, lease_ttl_ms: u32) -> Result<Self> {
        let lease_ttl_ms = lease_ttl_ms.max(1_000);
        session.acquire_control(ControlAuthority::Motherbrain, lease_ttl_ms)?;
        // A newly acquired authority always begins stopped. Do not select or
        // otherwise expose Create OI modes from the motherbrain.
        expect_accepted(session.execute(CockpitRequest::Stop)?)?;
        let heartbeat_timeout_ms = 750;
        Ok(Self {
            session,
            lease_acquired_at: Instant::now(),
            lease_ttl_ms,
            renew_margin_ms: (lease_ttl_ms / 3).max(500),
            motion_ttl_ms: 300,
            heartbeat_timeout_ms,
            max_linear_mm_s: 50,
            max_angular_mrad_s: 500,
            motor_gate_open: true,
            refusal_reason: None,
            last_applied_at: None,
            last_applied_command: None,
            last_status: None,
        })
    }

    pub fn with_limits(mut self, linear_mm_s: i16, angular_mrad_s: i16) -> Self {
        self.max_linear_mm_s = linear_mm_s.abs().min(50);
        self.max_angular_mrad_s = angular_mrad_s.abs().min(500);
        self
    }

    pub fn snapshot(&self) -> PossessionSnapshot {
        let lease = self.session.control_lease.as_ref();
        let elapsed = self.lease_acquired_at.elapsed().as_millis() as u64;
        PossessionSnapshot {
            brainstem_device_id: self.session.session().peer_device_id.clone(),
            brainstem_boot_id: self.session.session().peer_boot_id.clone(),
            session_id: self.session.session().session_id.clone(),
            lease_id: lease
                .map(|lease| lease.lease_id.clone())
                .unwrap_or_default(),
            lease_generation: lease.map(|lease| lease.generation).unwrap_or_default(),
            lease_remaining_ms: u64::from(self.lease_ttl_ms)
                .saturating_sub(elapsed)
                .min(u64::from(u32::MAX)) as u32,
            possessed: self.motor_gate_open && lease.is_some(),
            moving: self
                .last_status
                .as_ref()
                .and_then(|status| status.active_motion),
            brainstem_armed: self.last_status.as_ref().and_then(|status| status.armed),
            body_health: self
                .last_status
                .as_ref()
                .and_then(|status| status.runtime_state.clone()),
            uart_health: self.last_status.as_ref().and_then(|status| {
                value_for(&status.raw, "create_uart_health")
                    .or_else(|| value_for(&status.raw, "uart_health"))
                    .map(ToOwned::to_owned)
            }),
            safety_tripped: self
                .last_status
                .as_ref()
                .and_then(|status| status.safety_tripped),
            estop_latched: self
                .last_status
                .as_ref()
                .and_then(|status| status.estop_latched),
            refusal_reason: self.refusal_reason.clone(),
            last_applied_command_age_ms: self
                .last_applied_at
                .map(|instant| instant.elapsed().as_millis().min(u128::from(u64::MAX)) as u64),
            last_applied_command: self.last_applied_command,
        }
    }

    pub fn maintain(&mut self) -> Result<()> {
        if !self.motor_gate_open {
            return Err(CockpitError::Policy(
                self.refusal_reason
                    .clone()
                    .unwrap_or_else(|| "not possessed".into()),
            ));
        }
        let renew_at = self
            .lease_ttl_ms
            .saturating_sub(self.renew_margin_ms)
            .min(POSSESSION_LEASE_RENEW_INTERVAL_MS);
        if self.lease_acquired_at.elapsed() < Duration::from_millis(u64::from(renew_at)) {
            return Ok(());
        }
        self.renew_control_lease()
    }

    fn renew_control_lease(&mut self) -> Result<()> {
        if let Err(error) = self
            .session
            .acquire_control(ControlAuthority::Motherbrain, self.lease_ttl_ms)
        {
            self.close_gate(format!("control lease renewal failed: {error}"));
            return Err(error);
        }
        self.lease_acquired_at = Instant::now();
        Ok(())
    }

    fn close_gate(&mut self, reason: String) {
        self.motor_gate_open = false;
        self.refusal_reason = Some(reason);
    }

    /// A safety clear is an explicit request to resume only after the
    /// brainstem confirms that no e-stop or safety latch remains. Refresh the
    /// control lease before reopening the local motor gate so a recovery never
    /// revives stale authority.
    fn reopen_gate_if_safety_clear(&mut self) -> Result<()> {
        let status = match self.session.execute(CockpitRequest::GetStatus) {
            Ok(CockpitResponse::Status(status)) => status.summary(),
            Ok(other) => return Err(CockpitError::BadResponse(format!("{other:?}"))),
            Err(error) => return Err(error),
        };
        self.last_status = Some(status.clone());
        if status.estop_latched == Some(true) || status.safety_tripped == Some(true) {
            return Ok(());
        }
        if let Err(error) = self
            .session
            .acquire_control(ControlAuthority::Motherbrain, self.lease_ttl_ms)
        {
            self.close_gate(format!("control lease recovery failed: {error}"));
            return Err(error);
        }
        self.lease_acquired_at = Instant::now();
        self.motor_gate_open = true;
        self.refusal_reason = None;
        Ok(())
    }

    fn execute_with_busy_retry(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        let mut retried_after_lease_renewal = false;
        for attempt in 0..POSSESSION_BUSY_RETRY_ATTEMPTS {
            match self.session.execute(request.clone()) {
                Err(CockpitError::Rejected { reason, .. })
                    if reason == "busy" && attempt + 1 < POSSESSION_BUSY_RETRY_ATTEMPTS =>
                {
                    std::thread::sleep(POSSESSION_BUSY_RETRY_DELAY);
                }
                Err(CockpitError::Rejected { command_id, reason }) if reason == "busy" => {
                    return Err(self.annotate_busy_rejection(command_id, request.verb()));
                }
                Err(error)
                    if request.authorization_class() == AuthorizationClass::ControlLease
                        && !retried_after_lease_renewal
                        && is_control_lease_rejection(&error) =>
                {
                    self.renew_control_lease()?;
                    retried_after_lease_renewal = true;
                }
                result => return result,
            }
        }
        unreachable!("bounded busy retry always returns on its final attempt")
    }

    fn annotate_busy_rejection(&mut self, command_id: u32, request_name: &str) -> CockpitError {
        let mut reason = format!("busy while submitting {request_name}");
        if let Ok(CockpitResponse::Status(status)) = self.session.execute(CockpitRequest::GetStatus)
        {
            let summary = status.summary();
            let current = value_for(&summary.raw, "command")
                .or_else(|| value_for(&summary.raw, "current_command"))
                .unwrap_or("unknown");
            let pending = value_for(&summary.raw, "pending")
                .or_else(|| value_for(&summary.raw, "pending_command"))
                .unwrap_or("unknown");
            let pending_id = value_for(&summary.raw, "pending_command_id").unwrap_or("unknown");
            let runtime = value_for(&summary.raw, "runtime")
                .or_else(|| value_for(&summary.raw, "current_runtime_state"))
                .unwrap_or("unknown");
            let body = value_for(&summary.raw, "body").unwrap_or("unknown");
            reason = format!(
                "{reason}; status current={current} pending={pending} pending_id={pending_id} runtime={runtime} body={body}"
            );
        }
        CockpitError::Rejected { command_id, reason }
    }

    pub fn exorcize(&mut self) -> Result<()> {
        let stop = self
            .execute_with_busy_retry(CockpitRequest::Stop)
            .and_then(expect_accepted);
        if let Err(error) = stop {
            self.close_gate("possession exorcized".into());
            return Err(error);
        }
        self.close_gate("possession exorcized".into());
        Ok(())
    }

    fn execute_scoped(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        if !request.bypasses_closed_motor_gate() {
            self.maintain()?;
        }
        if matches!(request, CockpitRequest::SetMode { .. }) {
            return Err(CockpitError::Policy(
                "Create OI is private to the brainstem".into(),
            ));
        }
        if matches!(request, CockpitRequest::Arm) {
            return Err(CockpitError::Policy(
                "the motherbrain lease is the possession gate; no second arm layer exists".into(),
            ));
        }
        if matches!(request, CockpitRequest::PowerState { .. }) {
            return Err(CockpitError::Policy(
                "body power control is outside production possession".into(),
            ));
        }
        let request = match request {
            CockpitRequest::CmdVel {
                linear_mm_s,
                angular_mrad_s,
                ..
            } => CockpitRequest::CmdVel {
                linear_mm_s: linear_mm_s.clamp(-self.max_linear_mm_s, self.max_linear_mm_s),
                angular_mrad_s: angular_mrad_s
                    .clamp(-self.max_angular_mrad_s, self.max_angular_mrad_s),
                ttl_ms: self.motion_ttl_ms,
            },
            CockpitRequest::BumpEscape {
                direction,
                backoff_mm_s,
                turn_angular_mrad_s,
            } => CockpitRequest::BumpEscape {
                direction,
                backoff_mm_s: backoff_mm_s.clamp(0, self.max_linear_mm_s),
                turn_angular_mrad_s: turn_angular_mrad_s.clamp(0, self.max_angular_mrad_s),
            },
            other => other,
        };
        let heartbeat_timeout_ms = match &request {
            CockpitRequest::CmdVel { .. } => Some(self.heartbeat_timeout_ms),
            CockpitRequest::BumpEscape {
                turn_angular_mrad_s,
                ..
            } => Some(
                legacy_bump_escape_duration_ms(*turn_angular_mrad_s)
                    .saturating_add(POSSESSION_BUMP_ESCAPE_HEARTBEAT_MARGIN_MS)
                    .max(self.heartbeat_timeout_ms),
            ),
            _ => None,
        };
        if let Some(timeout_ms) = heartbeat_timeout_ms {
            expect_accepted(
                self.execute_with_busy_retry(CockpitRequest::HeartbeatStop { timeout_ms })?,
            )?;
        }
        let response = self.execute_with_busy_retry(request.clone());
        match response {
            Ok(response) => {
                match request {
                    CockpitRequest::CmdVel {
                        linear_mm_s,
                        angular_mrad_s,
                        ..
                    } => {
                        self.last_applied_at = Some(Instant::now());
                        self.last_applied_command = Some(MotorCommand {
                            forward: mm_s_to_meters_per_second(linear_mm_s),
                            turn: mrad_s_to_radians_per_second(angular_mrad_s),
                        });
                    }
                    CockpitRequest::Stop => {
                        self.last_applied_at = Some(Instant::now());
                        self.last_applied_command = Some(MotorCommand::stop());
                    }
                    _ => {}
                }
                Ok(response)
            }
            Err(error) => {
                if request.authorization_class() == AuthorizationClass::ControlLease {
                    self.close_gate(format!("scoped command failed: {error}"));
                }
                Err(error)
            }
        }
    }
}

fn is_control_lease_rejection(error: &CockpitError) -> bool {
    matches!(
        error,
        CockpitError::Rejected { reason, .. }
            if reason.contains("invalid_control_lease")
                || reason.contains("control_lease_required")
    )
}

impl<C: Cockpit> Drop for MotherbrainPossession<C> {
    fn drop(&mut self) {
        let _ = self.execute_with_busy_retry(CockpitRequest::Stop);
    }
}

impl<C: Cockpit> Cockpit for MotherbrainPossession<C> {
    fn possession_snapshot(&self) -> Option<PossessionSnapshot> {
        Some(self.snapshot())
    }

    fn event_cursor_hint(&self) -> Option<u32> {
        Some(self.session.cursor.next_seq())
    }

    fn manages_motion_heartbeat(&self) -> bool {
        true
    }

    fn exorcize(&mut self) -> Result<()> {
        MotherbrainPossession::exorcize(self)
    }

    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        match request.authorization_class() {
            AuthorizationClass::ReadOnly => {
                if self.motor_gate_open {
                    self.maintain()?;
                }
                let response = match self.session.execute(request) {
                    Ok(response) => response,
                    Err(error) => {
                        self.close_gate(format!("brainstem status/event failure: {error}"));
                        return Err(error);
                    }
                };
                if let CockpitResponse::Status(status) = &response {
                    let summary = status.summary();
                    self.last_status = Some(summary.clone());
                    if summary.estop_latched == Some(true) {
                        self.close_gate("brainstem safety refusal".into());
                    }
                }
                Ok(response)
            }
            AuthorizationClass::Emergency
            | AuthorizationClass::Session
            | AuthorizationClass::ControlLease => {
                let reopen_after_clear = matches!(
                    request,
                    CockpitRequest::ClearEStop
                        | CockpitRequest::ClearSafetyLatch { .. }
                        | CockpitRequest::CarefulMode { .. }
                );
                let response = self.execute_scoped(request)?;
                if reopen_after_clear {
                    self.reopen_gate_if_safety_clear()?;
                }
                Ok(response)
            }
            AuthorizationClass::ServiceLease => Err(CockpitError::Policy(
                "service operations are outside motherbrain possession".into(),
            )),
        }
    }

    fn handshake(&mut self, _hello: HandshakeHello) -> Result<HandshakeOutcome> {
        Err(CockpitError::Policy(
            "possession session is already established".into(),
        ))
    }

    fn execute_in_session(
        &mut self,
        session: &CockpitSession,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        if session.session_id != self.session.session().session_id {
            return Err(CockpitError::Policy("session replacement detected".into()));
        }
        self.execute(request)
    }

    fn execute_with_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ControlLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        let current = self.session.control_lease.as_ref();
        if session.session_id != self.session.session().session_id
            || current.map(|value| (&value.lease_id, value.generation))
                != Some((&lease.lease_id, lease.generation))
        {
            return Err(CockpitError::Policy("superseded possession lease".into()));
        }
        self.execute(request)
    }

    fn execute_with_service_lease(
        &mut self,
        _session: &CockpitSession,
        _lease: &ServiceLease,
        _request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        Err(CockpitError::Policy(
            "service operations are outside motherbrain possession".into(),
        ))
    }
}

impl<C: Cockpit> SessionCockpit<C> {
    pub fn session(&self) -> &CockpitSession {
        &self.outcome.session
    }
    pub fn contract(&self) -> &CockpitContract {
        &self.outcome.contract
    }
    pub fn outcome(&self) -> &HandshakeOutcome {
        &self.outcome
    }
    pub fn connector_mut(&mut self) -> &mut C {
        &mut self.connector
    }

    pub fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        self.outcome.contract.validate_request(&request)?;
        match request.authorization_class() {
            AuthorizationClass::ControlLease => {
                let lease = self.control_lease.as_ref().ok_or_else(|| {
                    CockpitError::Policy("no control lease has been acquired".into())
                })?;
                if request.requires_operator_debug()
                    && lease.authority != ControlAuthority::OperatorDebug
                {
                    return Err(CockpitError::Policy(
                        "attended operator-debug authority required".into(),
                    ));
                }
                self.connector
                    .execute_with_lease(&self.outcome.session, lease, request)
            }
            AuthorizationClass::ServiceLease => {
                let lease = self.service_lease.as_ref().ok_or_else(|| {
                    CockpitError::Policy("no service lease has been acquired".into())
                })?;
                if request.required_service_scope() != Some(lease.scope) {
                    return Err(CockpitError::Policy("service_scope_denied".into()));
                }
                self.connector
                    .execute_with_service_lease(&self.outcome.session, lease, request)
            }
            _ => self
                .connector
                .execute_in_session(&self.outcome.session, request),
        }
    }

    pub fn acquire_service(&mut self, scope: ServiceScope, ttl_ms: u32) -> Result<&ServiceLease> {
        let response = self.connector.execute_in_session(
            &self.outcome.session,
            CockpitRequest::AcquireServiceLease { scope, ttl_ms },
        )?;
        let CockpitResponse::ServiceLeaseGranted(lease) = response else {
            return Err(CockpitError::BadResponse(format!("{response:?}")));
        };
        self.control_lease = None;
        self.service_lease = Some(lease);
        Ok(self.service_lease.as_ref().expect("lease was installed"))
    }

    pub fn control(&mut self) -> Result<ControlCockpit<'_, C>> {
        if self.control_lease.is_none() {
            return Err(CockpitError::Policy(
                "no control lease has been acquired".into(),
            ));
        }
        Ok(ControlCockpit { session: self })
    }

    pub fn service(&mut self) -> Result<ServiceCockpit<'_, C>> {
        if self.service_lease.is_none() {
            return Err(CockpitError::Policy(
                "no service lease has been acquired".into(),
            ));
        }
        Ok(ServiceCockpit { session: self })
    }

    pub fn acquire_control(
        &mut self,
        authority: ControlAuthority,
        ttl_ms: u32,
    ) -> Result<&ControlLease> {
        let response = self.connector.execute_in_session(
            &self.outcome.session,
            CockpitRequest::AcquireControlLease { authority, ttl_ms },
        )?;
        let CockpitResponse::ControlLeaseGranted(lease) = response else {
            return Err(CockpitError::BadResponse(format!("{response:?}")));
        };
        self.service_lease = None;
        self.control_lease = Some(lease);
        Ok(self.control_lease.as_ref().expect("lease was installed"))
    }

    pub fn read_only(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        if request.requires_session() {
            return Err(CockpitError::Policy(
                "read_only accepts recovery requests only".into(),
            ));
        }
        self.connector.execute(request)
    }

    pub fn poll_events(&mut self) -> Result<EventBatch> {
        self.cursor.poll(&mut self.connector)
    }

    pub fn into_parts(self) -> (C, HandshakeOutcome) {
        (self.connector, self.outcome)
    }
}

impl<C: Cockpit> Cockpit for SessionCockpit<C> {
    fn event_cursor_hint(&self) -> Option<u32> {
        Some(self.cursor.next_seq())
    }

    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        SessionCockpit::execute(self, request)
    }

    fn handshake(&mut self, _hello: HandshakeHello) -> Result<HandshakeOutcome> {
        Err(CockpitError::Policy(
            "cockpit session is already established".into(),
        ))
    }

    fn execute_in_session(
        &mut self,
        session: &CockpitSession,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        if session.session_id != self.session().session_id {
            return Err(CockpitError::Policy("session replacement detected".into()));
        }
        SessionCockpit::execute(self, request)
    }

    fn execute_with_lease(
        &mut self,
        _session: &CockpitSession,
        _lease: &ControlLease,
        _request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        Err(CockpitError::Policy(
            "nested control authority is not available".into(),
        ))
    }

    fn execute_with_service_lease(
        &mut self,
        _session: &CockpitSession,
        _lease: &ServiceLease,
        _request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        Err(CockpitError::Policy(
            "nested service authority is not available".into(),
        ))
    }
}

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
    brainstem: SocketAddr,
    next_seq: u32,
    timeout: Duration,
    active_session_id: Option<String>,
}

impl UdpCockpit {
    pub fn connect(brainstem: SocketAddr) -> Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        let timeout = Duration::from_millis(750);
        socket.set_read_timeout(Some(timeout))?;
        socket.set_write_timeout(Some(timeout))?;
        Ok(Self {
            socket,
            brainstem,
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
        self.socket.send_to(line.as_bytes(), self.brainstem)?;
        let mut buf = [0u8; MAX_COMPACT_HANDSHAKE_FRAME_LEN];
        let (len, _) = self.socket.recv_from(&mut buf)?;
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

