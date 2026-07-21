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
