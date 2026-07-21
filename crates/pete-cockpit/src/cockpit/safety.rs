#[derive(Debug, Clone)]
pub struct AgentPolicy {
    pub motion_ttl_ms: u32,
    pub heartbeat_timeout_ms: u32,
}

impl Default for AgentPolicy {
    fn default() -> Self {
        Self {
            motion_ttl_ms: 300,
            heartbeat_timeout_ms: 900,
        }
    }
}

pub struct SafeCockpit<C> {
    client: C,
    cursor: EventCursor,
    policy: AgentPolicy,
    contract: Option<CockpitContract>,
}

impl<C: Cockpit> SafeCockpit<C> {
    pub fn new(client: C) -> Self {
        Self::with_policy(client, AgentPolicy::default())
    }

    pub fn with_policy(client: C, policy: AgentPolicy) -> Self {
        let cursor = EventCursor {
            next_seq: client.event_cursor_hint().unwrap_or(0),
        };
        Self {
            client,
            cursor,
            policy,
            contract: None,
        }
    }

    pub fn client_mut(&mut self) -> &mut C {
        &mut self.client
    }

    pub fn replace_client(&mut self, client: C) {
        self.cursor = EventCursor {
            next_seq: client.event_cursor_hint().unwrap_or(0),
        };
        self.client = client;
        self.contract = None;
    }

    pub fn refresh_status(&mut self) -> Result<StatusSummary> {
        let status = self.client.get_status()?.summary();
        if self.cursor.next_seq == 0 {
            if let Some(event_next_seq) = status.event_next_seq {
                self.cursor = EventCursor::from_event_next_seq(event_next_seq);
            }
        }
        Ok(status)
    }

    pub fn resync_event_cursor_from_status(&mut self) -> Result<StatusSummary> {
        let status = self.client.get_status()?.summary();
        if let Some(event_next_seq) = status.event_next_seq {
            self.cursor = EventCursor::from_event_next_seq(event_next_seq);
        }
        Ok(status)
    }

    pub fn poll_events_allowing_history_gap(&mut self) -> Result<EventBatch> {
        let batch = self.client.get_events_since(self.cursor.next_seq)?;
        self.cursor = EventCursor::from_event_next_seq(batch.next_seq);
        Ok(batch)
    }

    /// Consume the next cursor-bounded batch from the brainstem interface.
    /// A reported history gap is an error; callers never silently skip body
    /// events and pretend their view is continuous.
    pub fn poll_events(&mut self) -> Result<EventBatch> {
        self.cursor.poll(&mut self.client)
    }

    pub fn refresh_contract(&mut self) -> Result<&CockpitContract> {
        let capabilities = self.client.get_capabilities()?;
        let contract = CockpitContract::new(capabilities);
        contract.validate_event_vocabulary()?;
        self.contract = Some(contract);
        Ok(self.contract.as_ref().expect("contract was just set"))
    }

    fn ensure_contract(&mut self) -> Result<&CockpitContract> {
        if self.contract.is_none() {
            self.refresh_contract()?;
        }
        Ok(self.contract.as_ref().expect("contract is present"))
    }

    pub fn poll_safety_events(&mut self) -> Result<Vec<SafeStopReason>> {
        let batch = self.cursor.poll(&mut self.client)?;
        Ok(batch
            .events
            .iter()
            .filter_map(SafeStopReason::from_event)
            .collect())
    }

    pub fn pulse_motion(&mut self, linear_mm_s: i16, angular_mrad_s: i16) -> Result<()> {
        let status = self.refresh_status()?;
        if status.estop_latched == Some(true) || status.safety_tripped == Some(true) {
            let mut reasons = Vec::new();
            if status.estop_latched == Some(true) {
                reasons.push(SafeStopReason::EStopLatched);
            }
            if status.safety_tripped == Some(true) {
                reasons.push(SafeStopReason::SafetyTripped {
                    latch: status.safety_latch_kind,
                });
            }
            return Err(CockpitError::MotionStopped { reasons });
        }
        let heartbeat_timeout_ms = self.policy.heartbeat_timeout_ms;
        let motion_ttl_ms = self.policy.motion_ttl_ms;
        let heartbeat_managed = self.client.manages_motion_heartbeat();
        let contract = self.ensure_contract()?;
        if !contract.supports("cmd_vel") {
            return Err(CockpitError::Policy(
                "refusing motion because cmd_vel is unsupported".to_owned(),
            ));
        }
        if heartbeat_timeout_ms > 0 && !heartbeat_managed && !contract.supports("heartbeat_stop") {
            return Err(CockpitError::Policy(
                "heartbeat policy requires heartbeat_stop capability".to_owned(),
            ));
        }
        let request = CockpitRequest::CmdVel {
            linear_mm_s,
            angular_mrad_s,
            ttl_ms: motion_ttl_ms,
        };
        let request = contract.clamp_motion_request(&request);
        contract.validate_request(&request)?;
        if heartbeat_timeout_ms > 0 && !heartbeat_managed {
            let heartbeat = CockpitRequest::HeartbeatStop {
                timeout_ms: heartbeat_timeout_ms,
            };
            let heartbeat = contract.clamp_motion_request(&heartbeat);
            contract.validate_request(&heartbeat)?;
            if let CockpitRequest::HeartbeatStop { timeout_ms } = heartbeat {
                self.client.heartbeat_stop(timeout_ms)?;
            }
        }
        let CockpitRequest::CmdVel {
            linear_mm_s,
            angular_mrad_s,
            ttl_ms,
        } = request
        else {
            unreachable!("request was constructed as cmd_vel")
        };
        self.client.cmd_vel(linear_mm_s, angular_mrad_s, ttl_ms)?;
        let stops = self.poll_safety_events()?;
        if !stops.is_empty() {
            let _ = self.client.stop();
            return Err(CockpitError::MotionStopped { reasons: stops });
        }
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        self.client.stop()?;
        let _ = self.poll_safety_events()?;
        Ok(())
    }
}

