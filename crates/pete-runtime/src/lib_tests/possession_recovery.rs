fn possessor_test_step(
    runtime: &mut PossessorSkillRuntime,
    cockpit: &mut SimCockpit,
    request: &SkillRequest,
    body: &BodySense,
    home_base_contact: bool,
    events: &pete_cockpit::EventBatch,
    now_ms: u64,
) -> (SkillStatus, bool) {
    let now = Now::blank(now_ms, body.clone());
    runtime.step(
        cockpit,
        request,
        &now,
        &StatusSummary::from_raw(""),
        home_base_contact,
        events,
        now_ms,
    )
}

struct StubRuntime;

#[async_trait::async_trait]
impl RuntimeLoop for StubRuntime {
    async fn tick(
        &mut self,
        now: Now,
        _latent: ExperienceLatent,
        _futures: Vec<FuturePrediction>,
    ) -> Result<RuntimeTick> {
        let reign_input = now.reign.latest.clone();
        let action = ActionPrimitive::Go {
            intensity: 0.2,
            duration_ms: 100,
        };
        let experience =
            Experience::new("test", "test", Vec::new(), Vec::new(), now.t_ms, now.t_ms);
        Ok(RuntimeTick {
            frame: ExperienceFrame {
                id: Uuid::new_v4(),
                t_ms: now.t_ms,
                now,
                sensations: Vec::new(),
                impressions: Vec::new(),
                experiences: vec![experience.clone()],
                z: Some(ExperienceLatent::default()),
                chosen_action: Some(action.clone()),
                conscious_command: None,
                reign_input,
                reign_outcome: None,
                predicted_futures: Vec::new(),
                behavior_runs: Vec::new(),
                actual_next: None,
                reward: Reward::default(),
                surprise: SurpriseSense::default(),
                memory_recall: Vec::new(),
                recollections: Vec::new(),
                llm_teaching: Vec::new(),
                counterfactuals: Vec::new(),
                notes: Vec::new(),
            },
            experience,
            chosen_action: Some(action),
            skill_request: None,
            skill_status: None,
            recall: RecallBundle::default(),
            llm: LlmTickResult::default(),
            combobulation: None,
            inline_learning: InlineLearningTickStatus::default(),
            brain_events: Vec::new(),
        })
    }
}

struct SlowRuntime {
    tick_attempts: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl RuntimeLoop for SlowRuntime {
    async fn tick(
        &mut self,
        now: Now,
        _latent: ExperienceLatent,
        _futures: Vec<FuturePrediction>,
    ) -> Result<RuntimeTick> {
        self.tick_attempts.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let experience =
            Experience::new("test", "test", Vec::new(), Vec::new(), now.t_ms, now.t_ms);
        Ok(RuntimeTick {
            frame: ExperienceFrame {
                id: Uuid::new_v4(),
                t_ms: now.t_ms,
                now,
                sensations: Vec::new(),
                impressions: Vec::new(),
                experiences: vec![experience.clone()],
                z: Some(ExperienceLatent::default()),
                chosen_action: Some(ActionPrimitive::Stop),
                conscious_command: None,
                reign_input: None,
                reign_outcome: None,
                predicted_futures: Vec::new(),
                behavior_runs: Vec::new(),
                actual_next: None,
                reward: Reward::default(),
                surprise: SurpriseSense::default(),
                memory_recall: Vec::new(),
                recollections: Vec::new(),
                llm_teaching: Vec::new(),
                counterfactuals: Vec::new(),
                notes: Vec::new(),
            },
            experience,
            chosen_action: Some(ActionPrimitive::Stop),
            skill_request: None,
            skill_status: None,
            recall: RecallBundle::default(),
            llm: LlmTickResult::default(),
            combobulation: None,
            inline_learning: InlineLearningTickStatus::default(),
            brain_events: Vec::new(),
        })
    }
}

#[derive(Clone)]
struct SharedSimCockpit(Arc<Mutex<SimCockpit>>);

impl Cockpit for SharedSimCockpit {
    fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
        self.0.lock().unwrap().execute(request)
    }

    fn handshake(
        &mut self,
        hello: pete_cockpit::HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        self.0.lock().unwrap().handshake(hello)
    }

    fn execute_in_session(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.0.lock().unwrap().execute_in_session(session, request)
    }

    fn execute_with_lease(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        lease: &pete_cockpit::ControlLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.0
            .lock()
            .unwrap()
            .execute_with_lease(session, lease, request)
    }

    fn execute_with_service_lease(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        lease: &pete_cockpit::ServiceLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.0
            .lock()
            .unwrap()
            .execute_with_service_lease(session, lease, request)
    }
}

#[test]
fn production_possession_composes_bump_stop_and_conductor_recovery() {
    let sim = Arc::new(Mutex::new(SimCockpit::new().with_event_capacity(256)));
    let session = establish_session(
        SharedSimCockpit(Arc::clone(&sim)),
        HandshakeHello::motherbrain("pete-runtime-recovery-test"),
        None,
    )
    .unwrap();
    let possession = MotherbrainPossession::acquire(session, 5_000).unwrap();
    let mut cockpit = SafeCockpit::new(possession);

    cockpit.pulse_motion(40, 0).unwrap();
    sim.lock().unwrap().set_bump(true, false);

    let stop_events = cockpit.poll_events().unwrap();
    assert!(stop_events.has_stop_reason());
    let status = cockpit.refresh_status().unwrap();
    let body = body_sense_from_cockpit_status(status, 1_000);
    assert!(body.flags.bump_left);

    let mut conductor = SimpleConductor::default();
    let mut input = test_conductor_input(ActionPrimitive::Stop);
    input.body = body;
    let first_recovery = conductor.choose(input).unwrap();
    assert!(matches!(
        first_recovery,
        ActionPrimitive::Go {
            intensity,
            duration_ms: 500
        } if intensity < 0.0
    ));

    sim.lock().unwrap().set_bump(false, false);
    cockpit
        .client_mut()
        .clear_safety_latch(SafetyLatchKind::Bump)
        .unwrap();
    let clear_events = cockpit.poll_events().unwrap();
    assert!(clear_events
        .events
        .iter()
        .any(|event| event.kind == pete_cockpit::CockpitEventKind::SafetyCleared));
    let cleared_body = body_sense_from_cockpit_status(cockpit.refresh_status().unwrap(), 1_100);
    assert!(!cleared_body.flags.bump_left);

    let mut cleared_input = test_conductor_input(ActionPrimitive::Stop);
    cleared_input.body = cleared_body.clone();
    let reverse = conductor.choose(cleared_input.clone()).unwrap();
    let reverse_motor = action_to_motor_command(Some(&reverse)).clamped(0.05, 0.5);
    assert!(reverse_motor.forward < 0.0);
    apply_slow_possession_motor(&mut cockpit, reverse_motor).unwrap();

    cleared_input.body.odometry.x_m -= 0.08;
    let turn = conductor.choose(cleared_input).unwrap();
    assert!(matches!(
        turn,
        ActionPrimitive::Turn {
            direction: TurnDir::Right,
            ..
        }
    ));
    let turn_motor = action_to_motor_command(Some(&turn)).clamped(0.05, 0.5);
    assert!(turn_motor.turn.abs() > 0.0);
    apply_slow_possession_motor(&mut cockpit, turn_motor).unwrap();

    assert!(cockpit.client_mut().snapshot().possessed);
    let events = sim.lock().unwrap().get_events_since(0).unwrap();
    let stop_index = events
        .events
        .iter()
        .position(|event| event.kind == pete_cockpit::CockpitEventKind::MotionStopped)
        .unwrap();
    assert!(events.events[stop_index + 1..]
        .iter()
        .any(|event| event.kind == pete_cockpit::CockpitEventKind::MotionRequested));
}

#[test]
fn slow_possession_treats_preflight_bump_latch_as_recoverable() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.set_bump(true, false);
    let mut cockpit = SafeCockpit::with_policy(
        sim,
        pete_cockpit::AgentPolicy {
            motion_ttl_ms: 100,
            heartbeat_timeout_ms: 0,
        },
    );

    let block = apply_slow_possession_motor(
        &mut cockpit,
        MotorCommand {
            forward: 0.2,
            turn: 0.1,
        },
    )
    .unwrap();

    assert!(matches!(
        block,
        Some(SlowPossessionMotionBlock::SafetyLatch(
            SafetyLatchKind::Bump
        ))
    ));
    let status = cockpit.refresh_status().unwrap();
    assert_eq!(status.safety_tripped, Some(true));
    assert_eq!(status.safety_latch_kind, Some(SafetyLatchKind::Bump));
}

#[tokio::test]
async fn operator_estop_during_bump_recovery_requires_explicit_operator_clear() {
    let sim = Arc::new(Mutex::new(SimCockpit::new().with_event_capacity(256)));
    let session = establish_session(
        SharedSimCockpit(Arc::clone(&sim)),
        HandshakeHello::motherbrain("pete-runtime-normal-bump-test"),
        None,
    )
    .unwrap();
    let possession = MotherbrainPossession::acquire(session, 5_000).unwrap();
    let ledger_root = test_ledger_root("normal-possession-bump-recovery");
    let ledger = JsonlLedger::new(&ledger_root);
    let memory = InMemoryExperienceStore::new();
    let runtime = MinimalRuntime::new(
        ledger,
        memory.clone(),
        memory,
        SimpleConductor::default(),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    );
    let mut runner = RealRobotRunner::new(RobotMode::Slow, possession, Vec::new(), runtime)
        .with_autonomous_motion(true);
    runner.cockpit.resync_event_cursor_from_status().unwrap();

    let (_first_snapshot, _first_tick) = runner.tick_slow_manual().await.unwrap();

    sim.lock().unwrap().set_bump(true, false);
    // An E-stop received during the local reflex has no origin metadata,
    // so it must be treated as an operator stop and remain latched.
    runner.cockpit.client_mut().estop().unwrap();
    let (bump_snapshot, bump_tick) = runner.tick_slow_manual().await.unwrap();
    assert!(bump_tick.frame.now.body.flags.bump_left);
    assert_eq!(
        bump_snapshot
            .action_debug
            .as_ref()
            .and_then(|debug| debug.get("possession_recovery"))
            .and_then(|debug| debug.get("phase")),
        Some(&serde_json::json!("Idle"))
    );
    assert_eq!(bump_tick.chosen_action, Some(ActionPrimitive::Stop));
    let latched_status = runner.cockpit.refresh_status().unwrap();
    assert_eq!(latched_status.estop_latched, Some(true));

    let events_while_latched = sim.lock().unwrap().get_events_since(0).unwrap();
    let estop_index = events_while_latched
        .events
        .iter()
        .position(|event| event.kind == CockpitEventKind::EStopLatched)
        .unwrap();
    assert!(!events_while_latched.events[estop_index + 1..]
        .iter()
        .any(|event| event.kind == CockpitEventKind::EStopCleared));
    assert!(!events_while_latched.events[estop_index + 1..]
        .iter()
        .any(|event| event.kind == CockpitEventKind::MotionRequested));

    sim.lock().unwrap().set_bump(false, false);
    runner.cockpit.client_mut().clear_estop().unwrap();
    let (_completed_snapshot, _completed_tick) = runner.tick_slow_manual().await.unwrap();
    assert!(runner.possession_recovery.latch.is_none());
    let status = runner.cockpit.refresh_status().unwrap();
    assert_eq!(status.estop_latched, Some(false));
    assert_eq!(status.safety_tripped, Some(false));

    let events = sim.lock().unwrap().get_events_since(0).unwrap();
    let bump_index = events
        .events
        .iter()
        .position(|event| event.kind == pete_cockpit::CockpitEventKind::BumpChanged)
        .unwrap();
    let stop_index = events.events[bump_index..]
        .iter()
        .position(|event| event.kind == pete_cockpit::CockpitEventKind::MotionStopped)
        .map(|index| bump_index + index)
        .unwrap();
    let safety_index = events.events[bump_index..]
        .iter()
        .position(|event| event.kind == pete_cockpit::CockpitEventKind::SafetyTripped)
        .map(|index| bump_index + index)
        .unwrap();
    assert!(bump_index < safety_index && safety_index < stop_index);
    assert!(events
        .events
        .iter()
        .any(|event| { event.kind == CockpitEventKind::EStopCleared }));
    assert!(runner.cockpit.client_mut().snapshot().possessed);
    let _ = fs::remove_dir_all(ledger_root);
}

struct CountingCockpit {
    motor_attempts: Arc<AtomicUsize>,
    motors: Arc<Mutex<Vec<MotorCommand>>>,
    body: BodySense,
}

impl Cockpit for CountingCockpit {
    fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
        match request {
            CockpitRequest::GetStatus => Ok(CockpitResponse::Status(self.get_status()?)),
            CockpitRequest::GetCapabilities => {
                Ok(CockpitResponse::Capabilities(self.get_capabilities()?))
            }
            CockpitRequest::GetEvents { since_seq } => {
                Ok(CockpitResponse::Events(self.get_events_since(since_seq)?))
            }
            CockpitRequest::Stop => {
                self.motor_attempts.fetch_add(1, Ordering::SeqCst);
                self.motors.lock().unwrap().push(MotorCommand::stop());
                Ok(CockpitResponse::Accepted)
            }
            CockpitRequest::CmdVel {
                linear_mm_s,
                angular_mrad_s,
                ..
            } => {
                self.motor_attempts.fetch_add(1, Ordering::SeqCst);
                self.motors.lock().unwrap().push(MotorCommand {
                    forward: linear_mm_s as f32 / 1000.0,
                    turn: angular_mrad_s as f32 / 1000.0,
                });
                Ok(CockpitResponse::Accepted)
            }
            CockpitRequest::HeartbeatStop { .. } => Ok(CockpitResponse::Accepted),
            _ => Ok(CockpitResponse::Accepted),
        }
    }

    fn handshake(
        &mut self,
        _hello: pete_cockpit::HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        Err(pete_cockpit::CockpitError::Policy(
            "test cockpit has no handshake peer".into(),
        ))
    }

    fn execute_in_session(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ControlLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_service_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ServiceLease,
        _request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        Err(pete_cockpit::CockpitError::Policy(
            "test cockpit has no service mode".into(),
        ))
    }

    fn get_status(&mut self) -> pete_cockpit::Result<CockpitStatus> {
        Ok(CockpitStatus {
            raw: serde_json::json!({
                "uptime_ms": 1_000,
                "current_runtime_state": "test",
                "oi_mode": "safe",
                "current_command": "stop",
                "create_sensors": {
                    "last_packet_id": 0,
                    "complete_packet_count": 1,
                    "last_complete_packet_timestamp_ms": 1_000,
                    "bump_left": self.body.flags.bump_left,
                    "bump_right": self.body.flags.bump_right,
                    "wheel_drop": self.body.flags.wheel_drop,
                    "wall": self.body.flags.wall,
                    "virtual_wall": self.body.flags.virtual_wall,
                    "cliff_left": self.body.flags.cliff_left,
                    "cliff_front_left": self.body.flags.cliff_front_left,
                    "cliff_front_right": self.body.flags.cliff_front_right,
                    "cliff_right": self.body.flags.cliff_right,
                    "charge_mah": (self.body.battery_level.clamp(0.0, 1.0) * 2600.0).round() as u32,
                    "capacity_mah": 2600,
                    "charging_state": if self.body.charging { 1 } else { 0 },
                },
                "odometry": {
                    "distance_mm": (self.body.odometry.x_m * 1000.0).round() as i32,
                    "x_mm": (self.body.odometry.x_m * 1000.0).round() as i32,
                    "y_mm": (self.body.odometry.y_m * 1000.0).round() as i32,
                    "heading_mrad": (self.body.odometry.heading_rad * 1000.0).round() as i32,
                    "reset_count": 0,
                }
            })
            .to_string(),
        })
    }

    fn get_capabilities(&mut self) -> pete_cockpit::Result<CockpitCapabilities> {
        Ok(CockpitCapabilities {
            body_kind: "test".to_string(),
            drive: "differential".to_string(),
            verbs: [
                "status",
                "get_capabilities",
                "get_events",
                "stop",
                "cmd_vel",
                "heartbeat_stop",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            sensors: Vec::new(),
            outputs: Vec::new(),
            safety: Vec::new(),
            events: Vec::new(),
            independent_watchdog: Some(true),
            limits: pete_cockpit::CockpitLimits {
                max_linear_mm_s: 500,
                max_angular_mrad_s: 4_000,
                min_ttl_ms: 1,
                max_ttl_ms: 60_000,
            },
        })
    }

    fn get_events_since(&mut self, since_seq: u32) -> pete_cockpit::Result<EventBatch> {
        Ok(EventBatch {
            since_seq,
            oldest_seq: 1,
            next_seq: since_seq.saturating_add(1),
            dropped_before_seq: 0,
            events: Vec::new(),
        })
    }
}

struct LatchedStatusCockpit {
    clear_attempts: Arc<Mutex<Vec<SafetyLatchKind>>>,
    latch: SafetyLatchKind,
    safety_tripped: bool,
}

impl Cockpit for LatchedStatusCockpit {
    fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
        match request {
            CockpitRequest::GetStatus => Ok(CockpitResponse::Status(self.get_status()?)),
            CockpitRequest::GetCapabilities => {
                Ok(CockpitResponse::Capabilities(self.get_capabilities()?))
            }
            CockpitRequest::GetEvents { since_seq } => {
                Ok(CockpitResponse::Events(self.get_events_since(since_seq)?))
            }
            CockpitRequest::ClearSafetyLatch { latch } => {
                self.clear_attempts.lock().unwrap().push(latch);
                self.safety_tripped = false;
                Ok(CockpitResponse::Accepted)
            }
            CockpitRequest::Stop | CockpitRequest::HeartbeatStop { .. } => {
                Ok(CockpitResponse::Accepted)
            }
            _ => Ok(CockpitResponse::Accepted),
        }
    }

    fn handshake(
        &mut self,
        _hello: pete_cockpit::HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        Err(pete_cockpit::CockpitError::Policy(
            "test cockpit has no handshake peer".into(),
        ))
    }

    fn execute_in_session(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ControlLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_service_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ServiceLease,
        _request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        Err(pete_cockpit::CockpitError::Policy(
            "test cockpit has no service mode".into(),
        ))
    }

    fn get_status(&mut self) -> pete_cockpit::Result<CockpitStatus> {
        Ok(CockpitStatus {
            raw: serde_json::json!({
                "uptime_ms": 1_000,
                "current_runtime_state": "idle",
                "oi_mode": "safe",
                "current_command": "stop",
                "estop_latched": false,
                "safety_tripped": self.safety_tripped,
                "safety_latch_kind": self.latch,
                "create_sensors": {
                    "last_packet_id": 0,
                    "complete_packet_count": 1,
                    "last_complete_packet_timestamp_ms": 1_000,
                    "charging_state": 0,
                },
                "imu": {
                    "health": "ok",
                    "tilt_magnitude_mrad": 0,
                    "impact_score_mm_s2": 0,
                }
            })
            .to_string(),
        })
    }

    fn get_capabilities(&mut self) -> pete_cockpit::Result<CockpitCapabilities> {
        Ok(CockpitCapabilities {
            body_kind: "test".to_string(),
            drive: "differential".to_string(),
            verbs: [
                "status",
                "get_capabilities",
                "get_events",
                "stop",
                "cmd_vel",
                "heartbeat_stop",
                "clear_safety_latch",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            sensors: Vec::new(),
            outputs: Vec::new(),
            safety: Vec::new(),
            events: Vec::new(),
            independent_watchdog: Some(true),
            limits: pete_cockpit::CockpitLimits {
                max_linear_mm_s: 500,
                max_angular_mrad_s: 4_000,
                min_ttl_ms: 1,
                max_ttl_ms: 60_000,
            },
        })
    }

    fn get_events_since(&mut self, since_seq: u32) -> pete_cockpit::Result<EventBatch> {
        Ok(EventBatch {
            since_seq,
            oldest_seq: 1,
            next_seq: since_seq.saturating_add(1),
            dropped_before_seq: 0,
            events: Vec::new(),
        })
    }
}

struct ActiveBumpRecoveryCockpit {
    bump_escape_attempts: Arc<AtomicUsize>,
    careful_mode_attempts: Arc<AtomicUsize>,
    bump_escape_commands: Arc<Mutex<Vec<(SafetyLatchKind, u32, i16, i16, u32)>>>,
    stop_attempts: Arc<AtomicUsize>,
    clear_attempts: Arc<AtomicUsize>,
    bump_active: bool,
}

impl Cockpit for ActiveBumpRecoveryCockpit {
    fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
        match request {
            CockpitRequest::GetStatus => Ok(CockpitResponse::Status(self.get_status()?)),
            CockpitRequest::GetCapabilities => {
                Ok(CockpitResponse::Capabilities(self.get_capabilities()?))
            }
            CockpitRequest::GetEvents { since_seq } => {
                Ok(CockpitResponse::Events(self.get_events_since(since_seq)?))
            }
            CockpitRequest::EscapeMotion {
                hazard,
                hazard_generation,
                linear_mm_s,
                angular_mrad_s,
                ttl_ms,
            } => {
                self.bump_escape_attempts.fetch_add(1, Ordering::SeqCst);
                self.bump_escape_commands.lock().unwrap().push((
                    hazard,
                    hazard_generation,
                    linear_mm_s,
                    angular_mrad_s,
                    ttl_ms,
                ));
                Ok(CockpitResponse::Accepted)
            }
            CockpitRequest::CarefulMode { .. } => {
                self.careful_mode_attempts.fetch_add(1, Ordering::SeqCst);
                Ok(CockpitResponse::Accepted)
            }
            CockpitRequest::Stop => {
                self.stop_attempts.fetch_add(1, Ordering::SeqCst);
                Ok(CockpitResponse::Accepted)
            }
            CockpitRequest::ClearSafetyLatch {
                latch: SafetyLatchKind::Bump,
            } => {
                self.clear_attempts.fetch_add(1, Ordering::SeqCst);
                Ok(CockpitResponse::Accepted)
            }
            CockpitRequest::HeartbeatStop { .. } => Ok(CockpitResponse::Accepted),
            _ => Ok(CockpitResponse::Accepted),
        }
    }

    fn handshake(
        &mut self,
        _hello: pete_cockpit::HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        Err(pete_cockpit::CockpitError::Policy(
            "test cockpit has no handshake peer".into(),
        ))
    }

    fn execute_in_session(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ControlLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_service_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ServiceLease,
        _request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        Err(pete_cockpit::CockpitError::Policy(
            "test cockpit has no service mode".into(),
        ))
    }

    fn get_status(&mut self) -> pete_cockpit::Result<CockpitStatus> {
        Ok(CockpitStatus {
            raw: serde_json::json!({
                "uptime_ms": 1_000,
                "current_runtime_state": "idle",
                "oi_mode": "safe",
                "current_command": "stop",
                "estop_latched": false,
                "safety_tripped": true,
                "safety_latch_kind": "bump",
                "safety_hazard_generation": 42,
                "create_sensors": {
                    "last_packet_id": 0,
                    "complete_packet_count": 1,
                    "last_complete_packet_timestamp_ms": 1_000,
                    "bump_left": self.bump_active,
                    "bump_right": false,
                    "charging_state": 0,
                },
                "imu": {
                    "health": "ok",
                    "tilt_magnitude_mrad": 0,
                    "impact_score_mm_s2": 0,
                }
            })
            .to_string(),
        })
    }

    fn get_capabilities(&mut self) -> pete_cockpit::Result<CockpitCapabilities> {
        Ok(CockpitCapabilities {
            body_kind: "test".to_string(),
            drive: "differential".to_string(),
            verbs: [
                "status",
                "get_capabilities",
                "get_events",
                "stop",
                "cmd_vel",
                "escape_motion",
                "heartbeat_stop",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            sensors: Vec::new(),
            outputs: Vec::new(),
            safety: Vec::new(),
            events: Vec::new(),
            independent_watchdog: Some(true),
            limits: pete_cockpit::CockpitLimits {
                max_linear_mm_s: 500,
                max_angular_mrad_s: 4_000,
                min_ttl_ms: 1,
                max_ttl_ms: 60_000,
            },
        })
    }

    fn get_events_since(&mut self, since_seq: u32) -> pete_cockpit::Result<EventBatch> {
        Ok(EventBatch {
            since_seq,
            oldest_seq: 1,
            next_seq: since_seq.saturating_add(1),
            dropped_before_seq: 0,
            events: Vec::new(),
        })
    }
}

struct HistoryGapCockpit {
    inner: CountingCockpit,
    event_polls: Arc<AtomicUsize>,
    gap_poll: usize,
}

impl Cockpit for HistoryGapCockpit {
    fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
        self.inner.execute(request)
    }

    fn handshake(
        &mut self,
        hello: pete_cockpit::HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        self.inner.handshake(hello)
    }

    fn execute_in_session(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.inner.execute_in_session(session, request)
    }

    fn execute_with_lease(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        lease: &pete_cockpit::ControlLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.inner.execute_with_lease(session, lease, request)
    }

    fn execute_with_service_lease(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        lease: &pete_cockpit::ServiceLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.inner
            .execute_with_service_lease(session, lease, request)
    }

    fn get_status(&mut self) -> pete_cockpit::Result<CockpitStatus> {
        self.inner.get_status()
    }

    fn get_capabilities(&mut self) -> pete_cockpit::Result<CockpitCapabilities> {
        self.inner.get_capabilities()
    }

    fn get_events_since(&mut self, since_seq: u32) -> pete_cockpit::Result<EventBatch> {
        let poll = self.event_polls.fetch_add(1, Ordering::SeqCst);
        let inject_gap = poll == self.gap_poll;
        Ok(EventBatch {
            since_seq,
            oldest_seq: if inject_gap {
                since_seq.saturating_add(2)
            } else {
                1
            },
            next_seq: since_seq.saturating_add(2),
            dropped_before_seq: if inject_gap {
                since_seq.saturating_add(2)
            } else {
                0
            },
            events: Vec::new(),
        })
    }
}

struct MotionStopEventsCockpit {
    inner: CountingCockpit,
    event_polls: Arc<AtomicUsize>,
}

impl Cockpit for MotionStopEventsCockpit {
    fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
        self.inner.execute(request)
    }

    fn handshake(
        &mut self,
        hello: pete_cockpit::HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        self.inner.handshake(hello)
    }

    fn execute_in_session(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.inner.execute_in_session(session, request)
    }

    fn execute_with_lease(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        lease: &pete_cockpit::ControlLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.inner.execute_with_lease(session, lease, request)
    }

    fn execute_with_service_lease(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        lease: &pete_cockpit::ServiceLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.inner
            .execute_with_service_lease(session, lease, request)
    }

    fn get_status(&mut self) -> pete_cockpit::Result<CockpitStatus> {
        self.inner.get_status()
    }

    fn get_capabilities(&mut self) -> pete_cockpit::Result<CockpitCapabilities> {
        self.inner.get_capabilities()
    }

    fn get_events_since(&mut self, since_seq: u32) -> pete_cockpit::Result<EventBatch> {
        let poll = self.event_polls.fetch_add(1, Ordering::SeqCst);
        let events = if poll == 1 {
            vec![
                pete_cockpit::CockpitEvent {
                    seq: since_seq.saturating_add(1),
                    kind: pete_cockpit::CockpitEventKind::HeartbeatExpired,
                    a: 0,
                    b: 0,
                    c: 0,
                },
                pete_cockpit::CockpitEvent {
                    seq: since_seq.saturating_add(2),
                    kind: pete_cockpit::CockpitEventKind::SafetyTripped,
                    a: 1,
                    b: 0,
                    c: 0,
                },
            ]
        } else {
            Vec::new()
        };
        Ok(EventBatch {
            since_seq,
            oldest_seq: 1,
            next_seq: since_seq.saturating_add(events.len() as u32 + 1),
            dropped_before_seq: 0,
            events,
        })
    }
}

struct RejectingMotionCockpit {
    inner: CountingCockpit,
    rejection_attempts: Arc<AtomicUsize>,
}

impl Cockpit for RejectingMotionCockpit {
    fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
        if matches!(&request, CockpitRequest::CmdVel { .. }) {
            self.rejection_attempts.fetch_add(1, Ordering::SeqCst);
            return Err(pete_cockpit::CockpitError::Rejected {
                command_id: 42,
                reason: "stale_sequence".to_string(),
            });
        }
        self.inner.execute(request)
    }

    fn handshake(
        &mut self,
        hello: pete_cockpit::HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        self.inner.handshake(hello)
    }

    fn execute_in_session(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.inner.execute_in_session(session, request)
    }

    fn execute_with_lease(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        lease: &pete_cockpit::ControlLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.inner.execute_with_lease(session, lease, request)
    }

    fn execute_with_service_lease(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        lease: &pete_cockpit::ServiceLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.inner
            .execute_with_service_lease(session, lease, request)
    }

    fn get_status(&mut self) -> pete_cockpit::Result<CockpitStatus> {
        self.inner.get_status()
    }

    fn get_capabilities(&mut self) -> pete_cockpit::Result<CockpitCapabilities> {
        self.inner.get_capabilities()
    }

    fn get_events_since(&mut self, since_seq: u32) -> pete_cockpit::Result<EventBatch> {
        Ok(EventBatch {
            since_seq,
            oldest_seq: 1,
            next_seq: since_seq.saturating_add(1),
            dropped_before_seq: 0,
            events: Vec::new(),
        })
    }
}

struct FailingSensor;

#[async_trait::async_trait]
impl SenseProducer for FailingSensor {
    fn source_name(&self) -> &'static str {
        "kinect-depth"
    }

    async fn poll(&mut self) -> Result<pete_sensors::SensePacket> {
        anyhow::bail!("simulated sensor timeout")
    }
}
