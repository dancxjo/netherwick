#[derive(Debug, Clone)]
struct SimTimedAction {
    command_id: u32,
    complete_at_ms: u32,
    linear_mm_s: i16,
}

#[derive(Debug, Clone)]
struct SimContactWithdrawal {
    started_at_ms: u32,
    complete_at_ms: u32,
    baseline_odometry_mm: i32,
}

#[derive(Debug, Clone)]
pub struct SimCockpit {
    capabilities: CockpitCapabilities,
    events: Vec<CockpitEvent>,
    next_event_seq: u32,
    event_capacity: usize,
    now_ms: u32,
    next_command_id: u32,
    armed: bool,
    estop_latched: bool,
    safety_tripped: bool,
    safety_latch_kind: Option<SafetyLatchKind>,
    safety_hazard_generation: u32,
    bump_left: bool,
    bump_right: bool,
    cliff: bool,
    wheel_drop: bool,
    wall: bool,
    virtual_wall: bool,
    buttons: u8,
    ir_byte: u8,
    charging_state: u8,
    battery_charge_mah: u32,
    battery_capacity_mah: u32,
    odometry_distance_mm: i32,
    odometry_heading_mrad: i32,
    active_cmd_vel: Option<SimTimedAction>,
    active_contact_withdrawal: Option<SimContactWithdrawal>,
    last_contact_withdrawal_at_ms: Option<u32>,
    repeated_contact_count: u8,
    heartbeat_stop_at_ms: Option<u32>,
    audio_silent: bool,
    odometry_reset_count: u32,
    imu_calibration: u8,
    device_id: String,
    boot_id: String,
    active_session: Option<CockpitSession>,
    sessions: Vec<CockpitSession>,
    control_lease: Option<ControlLease>,
    control_lease_expires_at_ms: Option<u32>,
    lease_generation: u32,
    session_serial: u64,
    last_hello: Option<(HandshakeHello, HandshakeResponse)>,
    leases: HashMap<String, NetworkLease>,
    dns_records: HashMap<String, (String, u64, u32)>,
    registration_generation: u32,
    registration_boot_id: Option<String>,
    internal_domain: String,
    operator_debug_allowed: bool,
    recovery_forebrain_device_id: Option<String>,
    allow_unscoped_bench_mode: bool,
    scoped_dispatch: bool,
}

impl SimCockpit {
    pub fn new() -> Self {
        let mut sim = Self {
            capabilities: CockpitCapabilities {
                body_kind: "sim_create_oi".to_owned(),
                drive: "differential".to_owned(),
                verbs: CockpitRequest::capability_verbs()
                    .into_iter()
                    .map(ToOwned::to_owned)
                    .collect(),
                sensors: [
                    "bump",
                    "cliff",
                    "wheel_drop",
                    "wall",
                    "virtual_wall",
                    "ir",
                    "buttons",
                    "battery",
                    "odometry_delta",
                    "imu",
                ]
                .into_iter()
                .map(ToOwned::to_owned)
                .collect(),
                outputs: ["drive", "lights", "song"]
                    .into_iter()
                    .map(ToOwned::to_owned)
                    .collect(),
                safety: [
                    "estop",
                    "heartbeat",
                    "bump",
                    "cliff",
                    "wheel_drop",
                    "tilt",
                    "impact",
                    "contact_withdrawal_reflex_v1",
                ]
                .into_iter()
                .map(ToOwned::to_owned)
                .collect(),
                events: [
                    "boot",
                    "command_accepted",
                    "command_started",
                    "command_completed",
                    "command_interrupted",
                    "command_renewed",
                    "motion_requested",
                    "motion_stopped",
                    "safety_tripped",
                    "safety_cleared",
                    "bump_changed",
                    "cliff_changed",
                    "wheel_drop_latched",
                    "wheel_drop_cleared",
                    "wall_changed",
                    "virtual_wall_changed",
                    "battery_low",
                    "charging_state_changed",
                    "buttons_changed",
                    "ir_changed",
                    "heartbeat_expired",
                    "estop_latched",
                    "estop_cleared",
                    "imu_frame_received",
                    "imu_fault",
                    "tilt_changed",
                    "imu_calibration_changed",
                    "motion_inconsistency_detected",
                    "impact_detected",
                    "contact_withdrawal_started",
                    "contact_withdrawal_completed",
                    "audio_state_changed",
                ]
                .into_iter()
                .map(ToOwned::to_owned)
                .collect(),
                limits: CockpitLimits {
                    max_linear_mm_s: 500,
                    max_angular_mrad_s: 4_000,
                    min_ttl_ms: 10,
                    max_ttl_ms: 60_000,
                },
            },
            events: Vec::new(),
            next_event_seq: 1,
            event_capacity: DEFAULT_SIM_EVENT_CAPACITY,
            now_ms: 0,
            next_command_id: 1,
            armed: false,
            estop_latched: false,
            safety_tripped: false,
            safety_latch_kind: None,
            safety_hazard_generation: 0,
            bump_left: false,
            bump_right: false,
            cliff: false,
            wheel_drop: false,
            wall: false,
            virtual_wall: false,
            buttons: 0,
            ir_byte: 0,
            charging_state: 0,
            battery_charge_mah: 2600,
            battery_capacity_mah: 2600,
            odometry_distance_mm: 0,
            odometry_heading_mrad: 0,
            active_cmd_vel: None,
            active_contact_withdrawal: None,
            last_contact_withdrawal_at_ms: None,
            repeated_contact_count: 0,
            heartbeat_stop_at_ms: None,
            audio_silent: false,
            odometry_reset_count: 0,
            imu_calibration: 3,
            device_id: "pete-brainstem-sim".into(),
            boot_id: fresh_sim_boot_id(),
            active_session: None,
            sessions: Vec::new(),
            control_lease: None,
            control_lease_expires_at_ms: None,
            lease_generation: 0,
            session_serial: 0,
            last_hello: None,
            leases: HashMap::new(),
            dns_records: HashMap::new(),
            registration_generation: 0,
            registration_boot_id: None,
            internal_domain: DEFAULT_INTERNAL_DOMAIN.into(),
            operator_debug_allowed: false,
            recovery_forebrain_device_id: None,
            allow_unscoped_bench_mode: false,
            scoped_dispatch: false,
        };
        sim.push_event(CockpitEventKind::Boot, 0, 0, 0);
        sim
    }

    pub fn with_event_capacity(mut self, event_capacity: usize) -> Self {
        self.event_capacity = event_capacity.max(1);
        self.enforce_event_capacity();
        self
    }

    pub fn with_capabilities(mut self, capabilities: CockpitCapabilities) -> Self {
        self.capabilities = capabilities;
        self
    }

    pub fn with_identity(
        mut self,
        device_id: impl Into<String>,
        boot_id: impl Into<String>,
    ) -> Self {
        self.device_id = device_id.into();
        self.boot_id = boot_id.into();
        self
    }

    pub fn with_takeover_policy(
        mut self,
        operator_debug_allowed: bool,
        recovery_forebrain_device_id: Option<String>,
    ) -> Self {
        self.operator_debug_allowed = operator_debug_allowed;
        self.recovery_forebrain_device_id = recovery_forebrain_device_id;
        self
    }

    /// Explicit compatibility escape hatch for local bench code that has not
    /// yet migrated to `SessionCockpit`. Never enabled by production
    /// connectors or by the simulator default.
    pub fn with_unscoped_bench_mode(mut self) -> Self {
        self.allow_unscoped_bench_mode = true;
        self
    }

    pub fn reboot(&mut self) {
        self.boot_id = fresh_sim_boot_id();
        self.active_session = None;
        self.sessions.clear();
        self.control_lease = None;
        self.control_lease_expires_at_ms = None;
        self.armed = false;
        self.interrupt_active_motion();
        self.heartbeat_stop_at_ms = None;
        self.audio_silent = false;
        self.last_hello = None;
        self.push_event(CockpitEventKind::Boot, 0, 0, 0);
    }

    pub fn active_session(&self) -> Option<&CockpitSession> {
        self.active_session.as_ref()
    }

    pub fn add_network_lease(&mut self, lease: NetworkLease) {
        let identity = lease.dhcp_client_identifier.clone();
        if let Some(hostname) = lease.requested_hostname.as_deref() {
            if !RESERVED_NETWORK_NAMES.contains(&hostname) {
                let fqdn = format!("{hostname}.{}", self.internal_domain);
                self.dns_records
                    .insert(fqdn, (lease.leased_ip.clone(), lease.lease_expiry, 0));
            }
        }
        self.leases.insert(identity, lease);
    }

    pub fn resolve_internal_name(&mut self, name: &str) -> Option<String> {
        self.expire_network_records();
        let canonical = match name {
            "pete" | "brainstem" => format!("brainstem.{}", self.internal_domain),
            other if !other.contains('.') => format!("{other}.{}", self.internal_domain),
            other => other.to_owned(),
        };
        if canonical == format!("brainstem.{}", self.internal_domain) {
            return Some("192.168.4.1".into());
        }
        self.dns_records
            .get(&canonical)
            .map(|record| record.0.clone())
    }

    fn expire_network_records(&mut self) {
        let now = (self.now_ms / 1_000) as u64;
        self.leases.retain(|_, lease| lease.lease_expiry > now);
        self.dns_records.retain(|_, (_, expiry, _)| *expiry > now);
    }

    fn register_network_endpoint(
        &mut self,
        session: &CockpitSession,
        registration: RegisterNetworkEndpoint,
    ) -> Result<CockpitResponse> {
        if session.local_role != EndpointRole::Motherbrain
            || session.local_purpose != SessionPurpose::Control
            || !self
                .active_session
                .as_ref()
                .is_some_and(|active| active.session_id == session.session_id)
            || registration.hostname != "motherbrain"
        {
            return Err(CockpitError::Policy(
                "reserved motherbrain registration requires the active motherbrain identity".into(),
            ));
        }
        self.expire_network_records();
        let lease = self
            .leases
            .get(&registration.lease_identity)
            .ok_or_else(|| {
                CockpitError::Policy(
                    "network registration does not match an active DHCP lease".into(),
                )
            })?;
        if lease.leased_ip != registration.address {
            return Err(CockpitError::Policy(
                "registered address does not match the DHCP lease".into(),
            ));
        }
        let fqdn = format!("motherbrain.{}", self.internal_domain);
        let duplicate = self.registration_boot_id.as_deref()
            == Some(session.local_boot_id.as_str())
            && self
                .dns_records
                .get(&fqdn)
                .is_some_and(|record| record.0 == registration.address);
        if !duplicate {
            self.registration_generation = self.registration_generation.wrapping_add(1).max(1);
        }
        self.registration_boot_id = Some(session.local_boot_id.clone());
        let ttl = registration
            .ttl_seconds
            .min(
                (lease
                    .lease_expiry
                    .saturating_sub((self.now_ms / 1_000) as u64)) as u32,
            )
            .max(1);
        self.dns_records.insert(
            fqdn.clone(),
            (
                registration.address.clone(),
                (self.now_ms / 1_000) as u64 + ttl as u64,
                self.registration_generation,
            ),
        );
        Ok(CockpitResponse::NetworkEndpointRegistered(
            NetworkEndpointRegistered {
                session_id: session.session_id.clone(),
                fqdn,
                address: registration.address,
                ttl_seconds: ttl,
                registration_generation: self.registration_generation,
            },
        ))
    }

    fn acquire_control_lease(
        &mut self,
        session: &CockpitSession,
        authority: ControlAuthority,
        ttl_ms: u32,
    ) -> Result<CockpitResponse> {
        let allowed = pete_cockpit_protocol::role_can_request_control(
            session.local_role,
            session.local_purpose,
            authority,
        );
        if !allowed {
            return Err(CockpitError::Policy(
                "role is not eligible for requested control authority".into(),
            ));
        }
        if authority == ControlAuthority::OperatorDebug && !self.operator_debug_allowed {
            return Err(CockpitError::Policy(
                "operator debug policy is disabled".into(),
            ));
        }
        if authority == ControlAuthority::ForebrainRecovery
            && self.recovery_forebrain_device_id.as_deref()
                != Some(session.local_device_id.as_str())
        {
            return Err(CockpitError::Policy(
                "forebrain is not the configured recovery identity".into(),
            ));
        }
        let lease_alive = self
            .control_lease_expires_at_ms
            .is_some_and(|deadline| !time_reached(self.now_ms, deadline));
        let continuing_owner = lease_alive
            && self.control_lease.as_ref().is_some_and(|lease| {
                lease.session_id == session.session_id && lease.authority == authority
            });
        if authority == ControlAuthority::ForebrainRecovery && lease_alive {
            return Err(CockpitError::Policy(
                "current controller lease has not expired".into(),
            ));
        }
        // A renewal by the same live owner atomically replaces only the lease.
        // A true ownership transition stops and clears inherited state.
        if !continuing_owner {
            self.interrupt_active_motion();
            self.heartbeat_stop_at_ms = None;
            self.armed = authority == ControlAuthority::Motherbrain;
        }
        self.control_lease = None;
        self.control_lease_expires_at_ms = None;
        self.lease_generation = self.lease_generation.wrapping_add(1).max(1);
        let lease = ControlLease {
            lease_id: format!(
                "lease-{}-{}",
                self.lease_generation,
                uuid::Uuid::new_v4().simple()
            ),
            session_id: session.session_id.clone(),
            owner_role: session.local_role,
            authority,
            ttl_ms: ttl_ms.clamp(250, 60_000),
            generation: self.lease_generation,
        };
        self.control_lease = Some(lease.clone());
        self.control_lease_expires_at_ms = Some(self.now_ms.wrapping_add(lease.ttl_ms));
        Ok(CockpitResponse::ControlLeaseGranted(lease))
    }

    fn require_scoped_dispatch(&self) -> Result<()> {
        if self.allow_unscoped_bench_mode || self.scoped_dispatch {
            Ok(())
        } else {
            Err(CockpitError::SessionRequired)
        }
    }

    pub fn advance_ms(&mut self, ms: u32) {
        self.now_ms = self.now_ms.wrapping_add(ms);
        if self
            .control_lease_expires_at_ms
            .is_some_and(|deadline| time_reached(self.now_ms, deadline))
        {
            self.interrupt_active_motion();
            self.heartbeat_stop_at_ms = None;
            self.armed = false;
            self.control_lease = None;
            self.control_lease_expires_at_ms = None;
        }
        self.complete_due_cmd_vel();
        self.complete_due_contact_withdrawal();
        self.expire_heartbeat_if_due();
    }

    pub fn trip_safety(&mut self) {
        if self.safety_tripped {
            return;
        }
        self.safety_tripped = true;
        self.safety_latch_kind = Some(SafetyLatchKind::Bump);
        self.interrupt_active_motion();
        self.preempt_contact_withdrawal(1);
        self.safety_hazard_generation = self.push_event(CockpitEventKind::SafetyTripped, 1, 0, 0);
        self.push_event(CockpitEventKind::MotionStopped, 0, 0, 0);
    }

    pub fn set_bump(&mut self, left: bool, right: bool) {
        if self.bump_left == left && self.bump_right == right {
            return;
        }
        let was_active = self.bump_left || self.bump_right;
        let active = left || right;
        self.bump_left = left;
        self.bump_right = right;
        self.push_event(CockpitEventKind::BumpChanged, active as u32, 0, 0);
        if active && !was_active {
            // Match the firmware split: every fresh contact latches and stops,
            // but only contact during forward output starts the bounded,
            // authority-independent withdrawal.
            let unsafe_forward_output = self
                .active_cmd_vel
                .as_ref()
                .is_some_and(|motion| motion.linear_mm_s > 0);
            self.safety_tripped = true;
            self.safety_latch_kind = Some(SafetyLatchKind::Bump);
            let preempted_command_id = self
                .active_cmd_vel
                .as_ref()
                .map_or(0, |active| active.command_id);
            self.interrupt_active_motion();
            self.safety_hazard_generation =
                self.push_event(CockpitEventKind::SafetyTripped, 1, 0, 0);
            self.push_event(CockpitEventKind::MotionStopped, 0, 0, 0);
            if unsafe_forward_output {
                self.repeated_contact_count = match self.last_contact_withdrawal_at_ms {
                    Some(previous) if self.now_ms.wrapping_sub(previous) <= 2_000 => {
                        self.repeated_contact_count.saturating_add(1).max(1)
                    }
                    _ => 1,
                };
                self.last_contact_withdrawal_at_ms = Some(self.now_ms);
                self.push_event(
                    CockpitEventKind::ContactWithdrawalStarted,
                    u32::from(left)
                        | (u32::from(right) << 1)
                        | (u32::from(self.repeated_contact_count) << 8),
                    preempted_command_id,
                    u32::from(CONTACT_WITHDRAWAL_SPEED_MM_S.unsigned_abs())
                        | (CONTACT_WITHDRAWAL_DURATION_MS << 16),
                );
                self.active_contact_withdrawal = Some(SimContactWithdrawal {
                    started_at_ms: self.now_ms,
                    complete_at_ms: self.now_ms.wrapping_add(CONTACT_WITHDRAWAL_DURATION_MS),
                    baseline_odometry_mm: self.odometry_distance_mm,
                });
            }
        }
    }

    pub fn set_cliff(&mut self, active: bool) {
        if self.cliff == active {
            return;
        }
        self.cliff = active;
        self.push_event(CockpitEventKind::CliffChanged, active as u32, 0, 0);
        if active {
            self.safety_tripped = true;
            self.safety_latch_kind = Some(SafetyLatchKind::Cliff);
            self.interrupt_active_motion();
            self.preempt_contact_withdrawal(2);
            self.safety_hazard_generation =
                self.push_event(CockpitEventKind::SafetyTripped, 2, 0, 0);
            self.push_event(CockpitEventKind::MotionStopped, 0, 0, 0);
        }
    }

    pub fn set_wheel_drop(&mut self, active: bool) {
        if self.wheel_drop == active {
            return;
        }
        self.wheel_drop = active;
        if active {
            self.safety_tripped = true;
            self.safety_latch_kind = Some(SafetyLatchKind::WheelDrop);
            self.interrupt_active_motion();
            self.push_event(CockpitEventKind::WheelDropLatched, 1, 0, 0);
            self.safety_hazard_generation =
                self.push_event(CockpitEventKind::SafetyTripped, 3, 0, 0);
            self.push_event(CockpitEventKind::MotionStopped, 0, 0, 0);
        }
    }

    pub fn set_wall(&mut self, active: bool) {
        if self.wall == active {
            return;
        }
        self.wall = active;
        self.push_event(CockpitEventKind::WallChanged, active as u32, 0, 0);
    }

    pub fn set_virtual_wall(&mut self, active: bool) {
        if self.virtual_wall == active {
            return;
        }
        self.virtual_wall = active;
        self.push_event(CockpitEventKind::VirtualWallChanged, active as u32, 0, 0);
    }

    pub fn set_battery(&mut self, charge_mah: u32, capacity_mah: u32) {
        self.battery_charge_mah = charge_mah;
        self.battery_capacity_mah = capacity_mah;
        if self.battery_percent().is_some_and(|percent| percent <= 20) {
            self.push_event(
                CockpitEventKind::BatteryLow,
                self.battery_percent().unwrap_or(0) as u32,
                0,
                0,
            );
        }
    }

    pub fn set_charging_state(&mut self, state: u8) {
        if self.charging_state == state {
            return;
        }
        self.charging_state = state;
        self.push_event(CockpitEventKind::ChargingStateChanged, state as u32, 0, 0);
    }

    pub fn set_buttons(&mut self, buttons: u8) {
        if self.buttons == buttons {
            return;
        }
        self.buttons = buttons;
        self.push_event(CockpitEventKind::ButtonsChanged, buttons as u32, 0, 0);
    }

    pub fn set_ir_byte(&mut self, ir_byte: u8) {
        if self.ir_byte == ir_byte {
            return;
        }
        self.ir_byte = ir_byte;
        self.push_event(CockpitEventKind::IrChanged, ir_byte as u32, 0, 0);
    }

    pub fn odometry_reset_count(&self) -> u32 {
        self.odometry_reset_count
    }

    fn battery_percent(&self) -> Option<u8> {
        battery_percent(
            Some(self.battery_charge_mah),
            Some(self.battery_capacity_mah),
        )
    }

    fn accept_command(&mut self) -> u32 {
        let id = self.next_command_id;
        self.next_command_id = self.next_command_id.wrapping_add(1).max(1);
        self.push_event(CockpitEventKind::CommandAccepted, id, 0, 0);
        self.push_event(CockpitEventKind::CommandStarted, id, 0, 0);
        id
    }

    fn complete_command(&mut self, id: u32) {
        self.push_event(CockpitEventKind::CommandCompleted, id, 0, 0);
    }

    fn push_event(&mut self, kind: CockpitEventKind, a: u32, b: u32, c: u32) -> u32 {
        let seq = self.next_event_seq;
        self.next_event_seq = self.next_event_seq.wrapping_add(1).max(1);
        self.events.push(CockpitEvent { seq, kind, a, b, c });
        self.enforce_event_capacity();
        seq
    }

    fn enforce_event_capacity(&mut self) {
        let overflow = self.events.len().saturating_sub(self.event_capacity);
        if overflow > 0 {
            self.events.drain(0..overflow);
        }
    }

    fn interrupt_active_motion(&mut self) {
        if let Some(active) = self.active_cmd_vel.take() {
            self.push_event(
                CockpitEventKind::CommandInterrupted,
                active.command_id,
                0,
                0,
            );
        }
    }

    fn complete_due_cmd_vel(&mut self) {
        let Some(active) = self.active_cmd_vel.clone() else {
            return;
        };
        if !time_reached(self.now_ms, active.complete_at_ms) {
            return;
        }
        self.active_cmd_vel = None;
        self.push_event(CockpitEventKind::MotionStopped, 0, 0, 0);
        self.complete_command(active.command_id);
    }

    fn complete_due_contact_withdrawal(&mut self) {
        let Some(active) = self.active_contact_withdrawal.clone() else {
            return;
        };
        if !time_reached(self.now_ms, active.complete_at_ms) {
            return;
        }
        self.active_contact_withdrawal = None;
        // Shared bounds keep simulator and firmware transcripts on the same
        // bounded displacement contract.
        let displacement_mm = i32::from(CONTACT_WITHDRAWAL_SPEED_MM_S)
            * CONTACT_WITHDRAWAL_DURATION_MS as i32
            / 1_000;
        self.odometry_distance_mm = self.odometry_distance_mm.saturating_sub(displacement_mm);
        self.push_event(CockpitEventKind::MotionStopped, 0, 0, 0);
        self.push_event(
            CockpitEventKind::ContactWithdrawalCompleted,
            1 | (1 << 16),
            self.odometry_distance_mm
                .wrapping_sub(active.baseline_odometry_mm) as u32,
            self.now_ms.wrapping_sub(active.started_at_ms),
        );
    }

    fn preempt_contact_withdrawal(&mut self, safety_code: u32) {
        let Some(active) = self.active_contact_withdrawal.take() else {
            return;
        };
        self.push_event(CockpitEventKind::MotionStopped, 0, 0, 0);
        self.push_event(
            CockpitEventKind::ContactWithdrawalCompleted,
            2 | (safety_code << 8) | (1 << 16),
            self.odometry_distance_mm
                .wrapping_sub(active.baseline_odometry_mm) as u32,
            self.now_ms.wrapping_sub(active.started_at_ms),
        );
    }

    fn expire_heartbeat_if_due(&mut self) {
        let Some(deadline_ms) = self.heartbeat_stop_at_ms else {
            return;
        };
        if !time_reached(self.now_ms, deadline_ms) {
            return;
        }
        self.heartbeat_stop_at_ms = None;
        self.interrupt_active_motion();
        self.safety_tripped = true;
        self.push_event(CockpitEventKind::HeartbeatExpired, 0, 0, 0);
        self.push_event(CockpitEventKind::SafetyTripped, 5, 0, 0);
        self.push_event(CockpitEventKind::MotionStopped, 0, 0, 0);
    }

    fn oldest_seq(&self) -> u32 {
        self.events
            .first()
            .map(|event| event.seq)
            .unwrap_or(self.next_event_seq)
    }
}

impl Default for SimCockpit {
    fn default() -> Self {
        Self::new()
    }
}

impl Cockpit for SimCockpit {
    fn execute(&mut self, request: CockpitRequest) -> Result<CockpitResponse> {
        if request_is_removed_brainstem_convenience(&request) {
            return Err(CockpitError::Rejected {
                command_id: 0,
                reason: "unsupported".into(),
            });
        }
        if !self.allow_unscoped_bench_mode
            && !self.scoped_dispatch
            && !matches!(
                request.authorization_class(),
                AuthorizationClass::ReadOnly | AuthorizationClass::Emergency
            )
        {
            return Err(CockpitError::SessionRequired);
        }
        match request {
            CockpitRequest::GetStatus => Ok(CockpitResponse::Status(self.get_status()?)),
            CockpitRequest::GetCapabilities => {
                Ok(CockpitResponse::Capabilities(self.get_capabilities()?))
            }
            CockpitRequest::GetEvents { since_seq } => {
                Ok(CockpitResponse::Events(self.get_events_since(since_seq)?))
            }
            CockpitRequest::RegisterNetworkEndpoint(_) => Err(CockpitError::SessionRequired),
            CockpitRequest::Ping => {
                let id = self.accept_command();
                self.complete_command(id);
                Ok(CockpitResponse::Accepted)
            }
            CockpitRequest::Bootsel => {
                let id = self.accept_command();
                self.complete_command(id);
                Ok(CockpitResponse::Accepted)
            }
            CockpitRequest::Arm => self.arm().map(|()| CockpitResponse::Accepted),
            CockpitRequest::Disarm => self.disarm().map(|()| CockpitResponse::Accepted),
            CockpitRequest::Stop => self.stop().map(|()| CockpitResponse::Accepted),
            CockpitRequest::EStop => self.estop().map(|()| CockpitResponse::Accepted),
            CockpitRequest::ClearEStop => self.clear_estop().map(|()| CockpitResponse::Accepted),
            CockpitRequest::ClearSafetyLatch { latch } => self
                .clear_safety_latch(latch)
                .map(|()| CockpitResponse::Accepted),
            CockpitRequest::EscapeMotion {
                hazard,
                hazard_generation,
                linear_mm_s,
                angular_mrad_s,
                ttl_ms,
            } => self
                .escape_motion(
                    hazard,
                    hazard_generation,
                    linear_mm_s,
                    angular_mrad_s,
                    ttl_ms,
                )
                .map(|()| CockpitResponse::Accepted),
            CockpitRequest::CmdVel {
                linear_mm_s,
                angular_mrad_s,
                ttl_ms,
            } => self
                .cmd_vel(linear_mm_s, angular_mrad_s, ttl_ms)
                .map(|()| CockpitResponse::Accepted),
            CockpitRequest::HeartbeatStop { timeout_ms } => self
                .heartbeat_stop(timeout_ms)
                .map(|()| CockpitResponse::Accepted),
            CockpitRequest::StreamSensors {
                enabled,
                packet_id,
                period_ms,
            } => self
                .stream_sensors(enabled, packet_id, period_ms)
                .map(|()| CockpitResponse::Accepted),
            CockpitRequest::ResetOdometry => {
                self.reset_odometry().map(|()| CockpitResponse::Accepted)
            }
            CockpitRequest::ZeroImuOrientation => self
                .zero_imu_orientation()
                .map(|()| CockpitResponse::Accepted),
            CockpitRequest::ClearImuOrientation => self
                .clear_imu_orientation()
                .map(|()| CockpitResponse::Accepted),
            CockpitRequest::SetAudioSilent { silent } => {
                let id = self.accept_command();
                if self.audio_silent != silent {
                    self.audio_silent = silent;
                    self.push_event(CockpitEventKind::AudioStateChanged, silent as u32, 0, 0);
                }
                self.complete_command(id);
                Ok(CockpitResponse::Accepted)
            }
            _ => {
                let id = self.accept_command();
                self.complete_command(id);
                Ok(CockpitResponse::Accepted)
            }
        }
    }

    fn handshake(&mut self, hello: HandshakeHello) -> Result<HandshakeOutcome> {
        if let Some((cached_hello, cached_response)) = &self.last_hello {
            if cached_hello == &hello {
                return HandshakeOutcome::validate(&hello, cached_response.clone());
            }
        }
        self.session_serial = self.session_serial.wrapping_add(1).max(1);
        let establishes_primary = hello.role == EndpointRole::Motherbrain
            && hello.session_purpose == SessionPurpose::Control;
        let had_session = establishes_primary && self.active_session.is_some();
        let peer_reboot = self.active_session.as_ref().is_some_and(|session| {
            session.local_device_id == hello.device_id && session.local_boot_id != hello.boot_id
        });
        let response = negotiate(
            &hello,
            &self.device_id,
            &self.boot_id,
            self.capabilities.clone(),
            SafetySnapshot {
                armed: if establishes_primary {
                    false
                } else {
                    self.armed
                },
                estop_latched: self.estop_latched,
                safety_tripped: self.safety_tripped,
                active_motion: if establishes_primary {
                    false
                } else {
                    self.active_cmd_vel.is_some()
                },
                runtime_state: if self.active_cmd_vel.is_some() && !establishes_primary {
                    "moving".into()
                } else {
                    "idle".into()
                },
            },
            self.next_event_seq,
            SoftwareInfo {
                software_name: "pete-brainstem-sim".into(),
                software_version: env!("CARGO_PKG_VERSION").into(),
                build_id: "sim".into(),
            },
            self.session_serial,
        );
        if matches!(response, HandshakeResponse::Reject(_)) {
            self.push_event(CockpitEventKind::SessionRejected, 0, 0, 0);
        }
        let outcome = HandshakeOutcome::validate(&hello, response.clone())?;
        // Installing a valid session is a synchronous safety operation: old
        // motion and leases are revoked before WELCOME becomes observable.
        // Rejected/malformed hellos never mutate command state.
        if establishes_primary {
            self.sessions
                .retain(|session| session.local_role != EndpointRole::Motherbrain);
            self.dns_records
                .remove(&format!("motherbrain.{}", self.internal_domain));
            self.registration_boot_id = None;
            self.interrupt_active_motion();
            self.heartbeat_stop_at_ms = None;
            self.armed = false;
            self.control_lease = None;
            self.control_lease_expires_at_ms = None;
            self.active_session = Some(outcome.session.clone());
            self.push_event(
                if had_session {
                    CockpitEventKind::SessionReplaced
                } else {
                    CockpitEventKind::SessionOpened
                },
                0,
                0,
                0,
            );
        } else {
            self.push_event(CockpitEventKind::SessionOpened, 0, 0, 0);
        }
        self.sessions
            .retain(|session| session.session_id != outcome.session.session_id);
        self.sessions.push(outcome.session.clone());
        if peer_reboot {
            self.push_event(CockpitEventKind::PeerRebootDetected, 0, 0, 0);
        }
        self.last_hello = Some((hello, response));
        Ok(outcome)
    }

    fn execute_in_session(
        &mut self,
        session: &CockpitSession,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        if request.requires_session() {
            let valid = self
                .sessions
                .iter()
                .any(|active| active.session_id == session.session_id);
            if !valid {
                return Err(CockpitError::InvalidSession {
                    session_id: session.session_id.clone(),
                });
            }
        }
        if let CockpitRequest::RegisterNetworkEndpoint(registration) = request {
            return self.register_network_endpoint(session, registration);
        }
        if let CockpitRequest::AcquireControlLease { authority, ttl_ms } = request {
            return self.acquire_control_lease(session, authority, ttl_ms);
        }
        if matches!(request, CockpitRequest::AcquireServiceLease { .. }) {
            return Err(CockpitError::Policy(
                "service mode is disabled in this simulator".into(),
            ));
        }
        if request.authorization_class() == AuthorizationClass::ServiceLease {
            return Err(CockpitError::Policy(
                "request requires a separate service lease".into(),
            ));
        }
        if request.requires_control_authority() {
            let valid = self
                .control_lease
                .as_ref()
                .is_some_and(|lease| lease.session_id == session.session_id)
                && self
                    .control_lease_expires_at_ms
                    .is_some_and(|deadline| !time_reached(self.now_ms, deadline));
            if !valid {
                return Err(CockpitError::Policy(
                    "request requires the active control lease".into(),
                ));
            }
            if request.requires_operator_debug()
                && !self
                    .control_lease
                    .as_ref()
                    .is_some_and(|lease| lease.authority == ControlAuthority::OperatorDebug)
            {
                return Err(CockpitError::Policy(
                    "attended operator-debug authority required".into(),
                ));
            }
            if let CockpitRequest::HeartbeatStop { timeout_ms } = &request {
                self.control_lease_expires_at_ms =
                    Some(self.now_ms.wrapping_add((*timeout_ms).clamp(250, 60_000)));
            }
        }
        self.scoped_dispatch = true;
        let result = self.execute(request);
        self.scoped_dispatch = false;
        result
    }

    fn execute_with_lease(
        &mut self,
        session: &CockpitSession,
        lease: &ControlLease,
        request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        let valid = self.control_lease.as_ref().is_some_and(|active| {
            active.lease_id == lease.lease_id && active.session_id == session.session_id
        });
        if !valid {
            return Err(CockpitError::Policy(
                "unknown or replaced control lease".into(),
            ));
        }
        self.execute_in_session(session, request)
    }

    fn execute_with_service_lease(
        &mut self,
        _session: &CockpitSession,
        _lease: &ServiceLease,
        _request: CockpitRequest,
    ) -> Result<CockpitResponse> {
        Err(CockpitError::Policy(
            "service mode is disabled in this simulator".into(),
        ))
    }

    fn get_status(&mut self) -> Result<CockpitStatus> {
        self.complete_due_cmd_vel();
        self.complete_due_contact_withdrawal();
        self.expire_heartbeat_if_due();
        Ok(CockpitStatus {
            raw: format!(
                "OK 0 STATUS sim=true now_ms={} uptime_ms={} create_body_packets=1 create_last_body_packet_ms={} armed={} estop={} safety_tripped={} safety_latch_kind={} safety_hazard_generation={} event_next_seq={} active_cmd_vel={} bump_left={} bump_right={} cliff_left={} cliff_front_left={} cliff_front_right={} cliff_right={} wheel_drop={} wall={} virtual_wall={} ir_byte={} buttons={} charging_state={} charge_mah={} capacity_mah={} voltage_mv={} current_ma={} odometry_resets={} odometry_distance_mm={} odometry_heading_mrad={} imu_present=2 imu_health=1 imu_samples=1 imu_age_ms=0 imu_poll_ms=20 imu_yaw_mrad=0 imu_pitch_mrad=0 imu_roll_mrad=0 imu_yaw_rate_mrad_s=0 imu_gyro_x_mrad_s=0 imu_gyro_y_mrad_s=0 imu_gyro_z_mrad_s=0 imu_accel_x_mm_s2=0 imu_accel_y_mm_s2=0 imu_accel_z_mm_s2=9807 imu_accel_mag_mm_s2=9807 imu_tilt_mrad=0 imu_roughness_mm_s2=0 imu_impact_mm_s2=0 imu_motion_consistency=1 imu_calibration={} audio_silent={} audio_last_requested=none audio_last_played=none audio_last_playback_ms=0 audio_suppressed=0 audio_dropped=0",
                self.now_ms,
                self.now_ms,
                self.now_ms,
                self.armed,
                self.estop_latched,
                self.safety_tripped,
                self.safety_latch_kind
                    .map_or("none", SafetyLatchKind::as_str),
                self.safety_hazard_generation,
                self.next_event_seq,
                self.active_cmd_vel.is_some() || self.active_contact_withdrawal.is_some(),
                self.bump_left,
                self.bump_right,
                self.cliff,
                self.cliff,
                self.cliff,
                self.cliff,
                self.wheel_drop,
                self.wall,
                self.virtual_wall,
                self.ir_byte,
                self.buttons,
                self.charging_state,
                self.battery_charge_mah,
                self.battery_capacity_mah,
                if self.battery_capacity_mah == 0 {
                    0
                } else {
                    14_400
                },
                0,
                self.odometry_reset_count,
                self.odometry_distance_mm,
                self.odometry_heading_mrad,
                self.imu_calibration,
                self.audio_silent
            ),
        })
    }

    fn get_capabilities(&mut self) -> Result<CockpitCapabilities> {
        Ok(self.capabilities.clone())
    }

    fn get_events_since(&mut self, since_seq: u32) -> Result<EventBatch> {
        self.complete_due_cmd_vel();
        self.complete_due_contact_withdrawal();
        self.expire_heartbeat_if_due();
        let oldest_seq = self.oldest_seq();
        let dropped_before_seq = if since_seq.saturating_add(1) < oldest_seq {
            oldest_seq
        } else {
            0
        };
        Ok(EventBatch {
            since_seq,
            oldest_seq,
            next_seq: self.next_event_seq,
            dropped_before_seq,
            events: self
                .events
                .iter()
                .filter(|event| event.seq > since_seq)
                .cloned()
                .collect(),
        })
    }

    fn arm(&mut self) -> Result<()> {
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        self.armed = true;
        self.complete_command(id);
        Ok(())
    }

    fn disarm(&mut self) -> Result<()> {
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        self.interrupt_active_motion();
        self.preempt_contact_withdrawal(0);
        self.armed = false;
        self.complete_command(id);
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        let id = self.accept_command();
        self.interrupt_active_motion();
        self.preempt_contact_withdrawal(0);
        self.push_event(CockpitEventKind::MotionStopped, 0, 0, 0);
        self.complete_command(id);
        Ok(())
    }

    fn estop(&mut self) -> Result<()> {
        let id = self.accept_command();
        self.interrupt_active_motion();
        self.preempt_contact_withdrawal(4);
        self.estop_latched = true;
        self.safety_tripped = true;
        self.safety_latch_kind = None;
        self.push_event(CockpitEventKind::EStopLatched, 1, 0, 0);
        self.safety_hazard_generation = self.push_event(CockpitEventKind::SafetyTripped, 4, 0, 0);
        self.push_event(CockpitEventKind::MotionStopped, 0, 0, 0);
        self.complete_command(id);
        Ok(())
    }

    fn clear_estop(&mut self) -> Result<()> {
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        self.estop_latched = false;
        self.safety_tripped = false;
        self.safety_latch_kind = None;
        self.safety_hazard_generation = 0;
        self.push_event(CockpitEventKind::EStopCleared, 0, 0, 0);
        self.push_event(CockpitEventKind::SafetyCleared, 4, 0, 0);
        self.complete_command(id);
        Ok(())
    }

    fn clear_safety_latch(&mut self, kind: SafetyLatchKind) -> Result<()> {
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        self.safety_tripped = false;
        self.safety_latch_kind = None;
        self.safety_hazard_generation = 0;
        if kind == SafetyLatchKind::WheelDrop {
            self.push_event(CockpitEventKind::WheelDropCleared, 0, 0, 0);
        }
        self.push_event(
            CockpitEventKind::SafetyCleared,
            safety_latch_kind_code(kind),
            0,
            0,
        );
        self.complete_command(id);
        Ok(())
    }

    fn cmd_vel(&mut self, linear_mm_s: i16, angular_mrad_s: i16, ttl_ms: u32) -> Result<()> {
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        if self.estop_latched || self.safety_tripped {
            self.push_event(CockpitEventKind::CommandRejected, id, 0, 0);
            return Ok(());
        }
        self.interrupt_active_motion();
        self.push_event(
            CockpitEventKind::MotionRequested,
            pack_i16_pair(linear_mm_s, angular_mrad_s),
            ttl_ms,
            0,
        );
        self.active_cmd_vel = Some(SimTimedAction {
            command_id: id,
            complete_at_ms: self.now_ms.wrapping_add(ttl_ms.max(1)),
            linear_mm_s,
        });
        Ok(())
    }

    fn escape_motion(
        &mut self,
        hazard: SafetyLatchKind,
        hazard_generation: u32,
        linear_mm_s: i16,
        angular_mrad_s: i16,
        ttl_ms: u32,
    ) -> Result<()> {
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        let absolute_hazard = self.estop_latched || self.wheel_drop || self.charging_state != 0;
        let matching_hazard = self.safety_tripped
            && self.safety_latch_kind == Some(hazard)
            && self.safety_hazard_generation == hazard_generation
            && hazard_generation != 0;
        let bounded = ttl_ms == 250
            && (-120..=0).contains(&linear_mm_s)
            && angular_mrad_s.unsigned_abs() <= 500
            && (linear_mm_s != 0 || angular_mrad_s != 0);
        let compatible = match hazard {
            SafetyLatchKind::Bump => {
                (self.bump_left || self.bump_right)
                    && !self.cliff
                    && !(self.bump_left && !self.bump_right && angular_mrad_s > 0)
                    && !(!self.bump_left && self.bump_right && angular_mrad_s < 0)
                    && !(self.bump_left && self.bump_right && angular_mrad_s != 0)
            }
            SafetyLatchKind::Cliff => {
                self.cliff
                    && !self.bump_left
                    && !self.bump_right
                    && linear_mm_s < 0
                    && angular_mrad_s == 0
            }
            _ => false,
        };
        if self.active_contact_withdrawal.is_some() {
            self.push_event(
                CockpitEventKind::CommandRejected,
                id,
                CommandRejectReason::Busy.code() as u32,
                0,
            );
            return Err(CockpitError::Rejected {
                command_id: id,
                reason: CommandRejectReason::Busy.as_str().into(),
            });
        }
        if absolute_hazard || !matching_hazard || !bounded || !compatible {
            let reason = if absolute_hazard {
                CommandRejectReason::AbsoluteHazard
            } else if !matching_hazard {
                CommandRejectReason::HazardMismatch
            } else {
                CommandRejectReason::EscapeEnvelope
            };
            self.push_event(
                CockpitEventKind::CommandRejected,
                id,
                reason.code() as u32,
                0,
            );
            return Err(CockpitError::Rejected {
                command_id: id,
                reason: reason.as_str().into(),
            });
        }
        self.interrupt_active_motion();
        self.push_event(
            CockpitEventKind::MotionRequested,
            pack_i16_pair(linear_mm_s, angular_mrad_s),
            ttl_ms,
            0,
        );
        self.active_cmd_vel = Some(SimTimedAction {
            command_id: id,
            complete_at_ms: self.now_ms.wrapping_add(ttl_ms),
            linear_mm_s,
        });
        Ok(())
    }

    fn heartbeat_stop(&mut self, timeout_ms: u32) -> Result<()> {
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        self.heartbeat_stop_at_ms = Some(self.now_ms.wrapping_add(timeout_ms.max(1)));
        self.complete_command(id);
        Ok(())
    }

    fn stream_sensors(&mut self, _enabled: bool, _packet_id: u8, _period_ms: u32) -> Result<()> {
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        self.complete_command(id);
        Ok(())
    }

    fn reset_odometry(&mut self) -> Result<()> {
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        self.odometry_reset_count = self.odometry_reset_count.saturating_add(1);
        self.odometry_distance_mm = 0;
        self.odometry_heading_mrad = 0;
        self.complete_command(id);
        Ok(())
    }

    fn zero_imu_orientation(&mut self) -> Result<()> {
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        self.imu_calibration = 3;
        self.push_event(CockpitEventKind::ImuCalibrationChanged, 3, 9807, 1);
        self.complete_command(id);
        Ok(())
    }

    fn clear_imu_orientation(&mut self) -> Result<()> {
        self.require_scoped_dispatch()?;
        let id = self.accept_command();
        self.imu_calibration = 0;
        self.push_event(CockpitEventKind::ImuCalibrationChanged, 0, 0, 1);
        self.complete_command(id);
        Ok(())
    }
}

fn request_is_removed_brainstem_convenience(request: &CockpitRequest) -> bool {
    matches!(
        request,
        CockpitRequest::FaceBearing { .. }
            | CockpitRequest::TrackBearing { .. }
            | CockpitRequest::TurnBy { .. }
            | CockpitRequest::DriveFor { .. }
            | CockpitRequest::BumpEscape { .. }
            | CockpitRequest::HoldHeading { .. }
            | CockpitRequest::TurnToHeading { .. }
            | CockpitRequest::ArcFor { .. }
            | CockpitRequest::CreepUntil { .. }
            | CockpitRequest::ScanArc { .. }
            | CockpitRequest::DockAlign { .. }
            | CockpitRequest::WallFollow { .. }
            | CockpitRequest::WiggleAlign { .. }
            | CockpitRequest::Unstick { .. }
            | CockpitRequest::CliffGuard { .. }
            | CockpitRequest::SetSafetyPolicy { .. }
    )
}

