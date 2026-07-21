#[tokio::test]
async fn real_robot_read_only_runner_never_applies_motor() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let mut runner =
        RealRobotRunner::new(RobotMode::ReadOnly, Box::new(body), Vec::new(), StubRuntime);

    let (_snapshot, tick) = runner.tick_read_only().await.unwrap();

    assert!(matches!(
        tick.chosen_action,
        Some(ActionPrimitive::Go { .. })
    ));
    assert_eq!(motor_attempts.load(Ordering::SeqCst), 0);
    assert!(motors.lock().unwrap().is_empty());
    assert_eq!(
        tick.frame
            .now
            .extensions
            .get("safety/read_only_veto")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
}

#[tokio::test]
async fn real_robot_read_only_runner_publishes_snapshot_when_optional_sensor_fails() {
    let body = CountingCockpit {
        motor_attempts: Arc::new(AtomicUsize::new(0)),
        motors: Arc::new(Mutex::new(Vec::new())),
        body: BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let sensors: Vec<Box<dyn SenseProducer + Send>> = vec![Box::new(FailingSensor)];
    let mut runner =
        RealRobotRunner::new(RobotMode::ReadOnly, Box::new(body), sensors, StubRuntime);

    let (snapshot, _tick) = runner.tick_read_only().await.unwrap();

    assert!(snapshot.body.last_update_ms >= 100);
    assert_eq!(runner.tick_count, 1);
    assert_eq!(snapshot.body.odometry.x_m, 0.0);
    assert_eq!(
        _tick
            .frame
            .now
            .extensions
            .get("sensor.health")
            .and_then(|health| health.get(0))
            .and_then(|health| health.get("name")),
        Some(&serde_json::json!("kinect-depth"))
    );
    assert_eq!(
        _tick
            .frame
            .now
            .extensions
            .get("sensor.health")
            .and_then(|health| health.get(0))
            .and_then(|health| health.get("body_evidence_independent")),
        Some(&serde_json::json!(true))
    );
}

#[tokio::test]
async fn real_robot_slow_runner_keeps_body_evidence_when_kinect_fails() {
    let body = BodySense {
        battery_level: 0.61,
        charging: true,
        flags: pete_body::BodyFlags {
            wheel_drop: true,
            ..pete_body::BodyFlags::default()
        },
        odometry: Pose2 {
            x_m: 1.234,
            heading_rad: 0.875,
            ..Pose2::default()
        },
        last_update_ms: 100,
        ..BodySense::default()
    };
    let cockpit = CountingCockpit {
        motor_attempts: Arc::new(AtomicUsize::new(0)),
        motors: Arc::new(Mutex::new(Vec::new())),
        body,
    };
    let sensors: Vec<Box<dyn SenseProducer + Send>> = vec![Box::new(FailingSensor)];
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(cockpit), sensors, StubRuntime);

    let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(snapshot.body.battery_level, 0.61);
    assert!(snapshot.body.charging);
    assert!(snapshot.body.flags.wheel_drop);
    assert_eq!(snapshot.body.odometry.x_m, 1.234);
    assert_eq!(snapshot.body.odometry.heading_rad, 0.875);
    let health = tick.frame.now.extensions["sensor.health"][0].clone();
    assert_eq!(health["name"], "kinect-depth");
    assert_eq!(health["available"], false);
    assert_eq!(health["body_evidence_independent"], true);
}

#[test]
fn optional_sensor_failures_are_reported_once_per_interval() {
    let mut health = SensorPollHealth {
        name: "kinect-depth".to_string(),
        ..SensorPollHealth::default()
    };

    record_optional_sensor_failure(&mut health, "offline".to_string(), 1_000);
    let first_report = health.last_report_ms;
    record_optional_sensor_failure(&mut health, "offline".to_string(), 2_000);
    assert_eq!(health.last_report_ms, first_report);
    assert_eq!(health.consecutive_failures, 2);
    record_optional_sensor_failure(&mut health, "offline".to_string(), 31_001);
    assert_eq!(health.last_report_ms, 31_001);
}

#[tokio::test]
async fn real_robot_slow_runner_without_webremote_direct_sends_stop() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), StubRuntime);

    let (_snapshot, _tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(motor_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(motors.lock().unwrap().as_slice(), &[MotorCommand::stop()]);
}

#[tokio::test]
async fn real_robot_slow_runner_clears_latch_reported_by_status() {
    let clear_attempts = Arc::new(Mutex::new(Vec::new()));
    let body = LatchedStatusCockpit {
        clear_attempts: Arc::clone(&clear_attempts),
        latch: SafetyLatchKind::Tilt,
        safety_tripped: true,
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), StubRuntime)
        .with_autonomous_motion(true);

    let (snapshot, _tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(
        clear_attempts.lock().unwrap().as_slice(),
        &[SafetyLatchKind::Tilt]
    );
    assert_eq!(
        snapshot
            .action_debug
            .as_ref()
            .and_then(|debug| debug.get("possession_recovery"))
            .and_then(|debug| debug.get("latched")),
        Some(&serde_json::json!("Tilt"))
    );
}

#[tokio::test]
async fn real_robot_slow_runner_reports_active_bump_recovery_as_chosen_action() {
    let bump_escape_attempts = Arc::new(AtomicUsize::new(0));
    let careful_mode_attempts = Arc::new(AtomicUsize::new(0));
    let bump_escape_commands = Arc::new(Mutex::new(Vec::new()));
    let stop_attempts = Arc::new(AtomicUsize::new(0));
    let body = ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc::clone(&bump_escape_attempts),
        careful_mode_attempts: Arc::clone(&careful_mode_attempts),
        bump_escape_commands: Arc::clone(&bump_escape_commands),
        stop_attempts,
        clear_attempts: Arc::new(AtomicUsize::new(0)),
        bump_active: true,
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);

    let _ = runner.tick_slow_manual().await.unwrap();
    let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(bump_escape_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(careful_mode_attempts.load(Ordering::SeqCst), 0);
    assert_eq!(
        bump_escape_commands.lock().unwrap().as_slice(),
        &[(SafetyLatchKind::Bump, 42, -100, 0, 250)]
    );
    assert_eq!(
        tick.chosen_action,
        Some(ActionPrimitive::Go {
            intensity: -0.25,
            duration_ms: POSSESSION_ESCAPE_TTL_MS as TimeMs,
        })
    );
    let debug = snapshot.action_debug.as_ref().unwrap();
    assert_eq!(
        debug.get("runtime_chosen_action"),
        Some(
            &serde_json::to_value(ActionPrimitive::Go {
                intensity: 0.2,
                duration_ms: 100,
            })
            .unwrap()
        )
    );
    assert_eq!(
        debug.get("motion_sent_to_robot"),
        Some(
            &serde_json::to_value(motor_command_to_motion(MotorCommand {
                forward: -0.10,
                turn: 0.0,
            }))
            .unwrap()
        )
    );
    assert_eq!(debug.get("motor_applied"), Some(&serde_json::json!(true)));
    assert_eq!(
        debug
            .get("possession_recovery")
            .and_then(|debug| debug.get("latched")),
        Some(&serde_json::json!("Bump"))
    );
    assert_eq!(
        debug
            .get("possession_recovery")
            .and_then(|debug| debug.get("intended_motion"))
            .and_then(|motion| motion.get("linear")),
        Some(&serde_json::json!("reverse"))
    );
    assert_eq!(
        debug
            .get("possessor_skill_status")
            .and_then(|status| status.get("script"))
            .and_then(|script| script.get("skill_id")),
        Some(&serde_json::json!("motherbrain.releasePersistentBumper"))
    );
    assert_eq!(
        debug
            .get("possession_recovery")
            .and_then(|debug| debug.get("observed_motion"))
            .and_then(|motion| motion.get("linear_displacement_m")),
        Some(&serde_json::json!(0.0))
    );
}

#[tokio::test]
async fn possessor_submits_atomic_escape_when_local_withdrawal_ends_still_bumped() {
    let careful_mode_attempts = Arc::new(AtomicUsize::new(0));
    let motion_attempts = Arc::new(AtomicUsize::new(0));
    let body = ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc::clone(&motion_attempts),
        careful_mode_attempts: Arc::clone(&careful_mode_attempts),
        bump_escape_commands: Arc::new(Mutex::new(Vec::new())),
        stop_attempts: Arc::new(AtomicUsize::new(0)),
        clear_attempts: Arc::new(AtomicUsize::new(0)),
        bump_active: true,
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);
    runner.possession_recovery.latch = Some(SafetyLatchKind::Bump);
    runner.possession_recovery.hazard_generation = 42;
    runner.possession_recovery.phase = PossessionRecoveryPhase::WaitingForSensorClear;
    runner.possession_recovery.active_since_ms = wall_time_ms();
    runner.possession_recovery.last_command_ms = 0;
    runner.possession_recovery.brainstem_reflex_observed = true;
    runner.possession_recovery.last_reflex_outcome = Some(ContactWithdrawalOutcome::Completed);

    let _ = runner.tick_slow_manual().await.unwrap();
    let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(careful_mode_attempts.load(Ordering::SeqCst), 0);
    assert_eq!(motion_attempts.load(Ordering::SeqCst), 1);
    assert!(matches!(
        tick.chosen_action,
        Some(ActionPrimitive::Go { .. })
    ));
    assert!(snapshot
        .action_debug
        .as_ref()
        .and_then(|debug| debug.get("why_not_moving"))
        .and_then(|reason| reason.as_str())
        .is_some_and(|reason| reason.contains("foreground Lua")));
}

#[test]
fn lua_cliff_recovery_emits_only_generation_bound_reverse_escape() {
    let commands = Arc::new(Mutex::new(Vec::new()));
    let mut cockpit = ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc::new(AtomicUsize::new(0)),
        careful_mode_attempts: Arc::new(AtomicUsize::new(0)),
        bump_escape_commands: Arc::clone(&commands),
        stop_attempts: Arc::new(AtomicUsize::new(0)),
        clear_attempts: Arc::new(AtomicUsize::new(0)),
        bump_active: false,
    };
    let status = CockpitStatus {
        raw: serde_json::json!({
            "uptime_ms": 1_000,
            "current_runtime_state": "idle",
            "oi_mode": "safe",
            "safety_tripped": true,
            "safety_latch_kind": "cliff",
            "safety_hazard_generation": 77,
            "create_sensors": {
                "complete_packet_count": 1,
                "last_complete_packet_timestamp_ms": 1_000,
                "cliff_front_left": true,
                "charging_state": 0
            }
        })
        .to_string(),
    }
    .summary();
    let request = SkillRequest {
        skill_id: SkillId::RetreatFromCliff,
        ..SkillRequest::default()
    };
    let mut state = EmbodiedLuaDriverState::default();
    let mut driver = RealLuaOrganDriver {
        cockpit: &mut cockpit,
        request: &request,
        status: &status,
        home_base_contact: false,
        state: &mut state,
        command_sent: false,
    };
    let mut now = Now::blank(1_000, BodySense::default());
    now.body.flags.cliff_front_left = true;
    let result = driver.poll(
        &HostOperation::Retreat {
            hazard: HazardKind::Cliff,
            distance_m: 0.1,
        },
        OperationContext {
            operation_id: 1,
            child_id: 0,
            first_poll: true,
            elapsed_ms: 0,
            now_ms: 1_000,
            primitive_ttl_ms: 250,
        },
        &now,
        &EventBatch {
            since_seq: 0,
            oldest_seq: 0,
            next_seq: 0,
            dropped_before_seq: 0,
            events: Vec::new(),
        },
    );
    assert!(matches!(result, OrganPoll::Pending { .. }));
    assert_eq!(
        commands.lock().unwrap().as_slice(),
        &[(SafetyLatchKind::Cliff, 77, -100, 0, 250)]
    );
}

#[test]
fn lua_bump_recovery_cannot_suppress_imu_absolute_hazard() {
    let commands = Arc::new(Mutex::new(Vec::new()));
    let mut cockpit = ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc::new(AtomicUsize::new(0)),
        careful_mode_attempts: Arc::new(AtomicUsize::new(0)),
        bump_escape_commands: Arc::clone(&commands),
        stop_attempts: Arc::new(AtomicUsize::new(0)),
        clear_attempts: Arc::new(AtomicUsize::new(0)),
        bump_active: true,
    };
    let status = CockpitStatus {
        raw: serde_json::json!({
            "safety_tripped": true,
            "safety_latch_kind": "bump",
            "safety_hazard_generation": 78,
            "imu": {"health": "ok", "impact_score_mm_s2": 20_000}
        })
        .to_string(),
    }
    .summary();
    let request = SkillRequest {
        skill_id: SkillId::ReleasePersistentBumper,
        ..SkillRequest::default()
    };
    let mut state = EmbodiedLuaDriverState::default();
    let mut driver = RealLuaOrganDriver {
        cockpit: &mut cockpit,
        request: &request,
        status: &status,
        home_base_contact: false,
        state: &mut state,
        command_sent: false,
    };
    let mut now = Now::blank(1_000, BodySense::default());
    now.body.flags.bump_left = true;
    let result = driver.poll(
        &HostOperation::Retreat {
            hazard: HazardKind::BumperFront,
            distance_m: 0.1,
        },
        OperationContext {
            operation_id: 1,
            child_id: 0,
            first_poll: true,
            elapsed_ms: 0,
            now_ms: 1_000,
            primitive_ttl_ms: 250,
        },
        &now,
        &EventBatch {
            since_seq: 0,
            oldest_seq: 0,
            next_seq: 0,
            dropped_before_seq: 0,
            events: Vec::new(),
        },
    );
    assert!(matches!(
        result,
        OrganPoll::Failed(SkillFailure {
            outcome: SkillOutcome::SafetyPreempted,
            ..
        })
    ));
    assert!(commands.lock().unwrap().is_empty());
}

#[tokio::test]
async fn real_robot_slow_runner_renews_bounded_bump_escape_each_observation_tick() {
    let bump_escape_attempts = Arc::new(AtomicUsize::new(0));
    let bump_escape_commands = Arc::new(Mutex::new(Vec::new()));
    let stop_attempts = Arc::new(AtomicUsize::new(0));
    let body = ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc::clone(&bump_escape_attempts),
        careful_mode_attempts: Arc::new(AtomicUsize::new(0)),
        bump_escape_commands,
        stop_attempts,
        clear_attempts: Arc::new(AtomicUsize::new(0)),
        bump_active: true,
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);

    let (_first_snapshot, _first_tick) = runner.tick_slow_manual().await.unwrap();
    let (_second_snapshot, _second_tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(bump_escape_attempts.load(Ordering::SeqCst), 1);
    std::thread::sleep(Duration::from_millis(260));
    let (second_snapshot, second_tick) = runner.tick_slow_manual().await.unwrap();
    assert_eq!(bump_escape_attempts.load(Ordering::SeqCst), 2);
    assert_eq!(
        second_tick.chosen_action,
        Some(ActionPrimitive::Go {
            intensity: -0.25,
            duration_ms: POSSESSION_ESCAPE_TTL_MS as TimeMs,
        })
    );
    assert!(second_snapshot
        .action_debug
        .as_ref()
        .and_then(|debug| debug.get("why_not_moving"))
        .and_then(|reason| reason.as_str())
        .is_some_and(|reason| reason.contains("foreground Lua")));
}

#[tokio::test]
async fn real_robot_slow_runner_bounds_lua_bump_recovery_instead_of_eagerly_stopping() {
    let bump_escape_attempts = Arc::new(AtomicUsize::new(0));
    let bump_escape_commands = Arc::new(Mutex::new(Vec::new()));
    let stop_attempts = Arc::new(AtomicUsize::new(0));
    let body = ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc::clone(&bump_escape_attempts),
        careful_mode_attempts: Arc::new(AtomicUsize::new(0)),
        bump_escape_commands,
        stop_attempts: Arc::clone(&stop_attempts),
        clear_attempts: Arc::new(AtomicUsize::new(0)),
        bump_active: true,
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);
    runner.possession_recovery.latch = Some(SafetyLatchKind::Bump);
    runner.possession_recovery.hazard_generation = 42;
    runner.possession_recovery.phase = PossessionRecoveryPhase::WaitingForSensorClear;
    runner.possession_recovery.active_since_ms =
        wall_time_ms().saturating_sub(POSSESSION_RECOVERY_STUCK_AFTER_MS + 1);
    runner.possession_recovery.command_attempts = 12;

    let request = runner
        .possession_recovery_skill_request(&EventBatch {
            since_seq: 0,
            oldest_seq: 0,
            next_seq: 0,
            dropped_before_seq: 0,
            events: Vec::new(),
        })
        .expect("Lua recovery request");
    assert_eq!(request.skill_id, SkillId::ReleasePersistentBumper);
    assert_eq!(
        request.maximum_duration_ms,
        POSSESSION_RECOVERY_STUCK_AFTER_MS
    );
    assert_eq!(bump_escape_attempts.load(Ordering::SeqCst), 0);
    assert_eq!(stop_attempts.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn real_robot_slow_runner_does_not_escape_after_momentary_bump_clears() {
    let bump_escape_attempts = Arc::new(AtomicUsize::new(0));
    let bump_escape_commands = Arc::new(Mutex::new(Vec::new()));
    let body = ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc::clone(&bump_escape_attempts),
        careful_mode_attempts: Arc::new(AtomicUsize::new(0)),
        bump_escape_commands: Arc::clone(&bump_escape_commands),
        stop_attempts: Arc::new(AtomicUsize::new(0)),
        clear_attempts: Arc::new(AtomicUsize::new(0)),
        bump_active: false,
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);

    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(bump_escape_attempts.load(Ordering::SeqCst), 0);
    assert!(bump_escape_commands.lock().unwrap().is_empty());
    assert_ne!(
        tick.chosen_action,
        Some(ActionPrimitive::Go {
            intensity: 0.3,
            duration_ms: 700,
        })
    );
}

#[tokio::test]
async fn real_robot_slow_runner_never_imagines_turn_without_submitting_it() {
    let bump_escape_attempts = Arc::new(AtomicUsize::new(0));
    let body = ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc::clone(&bump_escape_attempts),
        careful_mode_attempts: Arc::new(AtomicUsize::new(0)),
        bump_escape_commands: Arc::new(Mutex::new(Vec::new())),
        stop_attempts: Arc::new(AtomicUsize::new(0)),
        clear_attempts: Arc::new(AtomicUsize::new(0)),
        bump_active: true,
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);
    runner.possession_recovery.latch = Some(SafetyLatchKind::Bump);
    runner.possession_recovery.hazard_generation = 42;
    runner.possession_recovery.phase = PossessionRecoveryPhase::Escaping;
    runner.possession_recovery.command_attempts = 1;

    let _ = runner.tick_slow_manual().await.unwrap();
    let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(bump_escape_attempts.load(Ordering::SeqCst), 1);
    assert!(matches!(
        tick.chosen_action,
        Some(ActionPrimitive::Go { .. })
    ));
    assert!(snapshot
        .action_debug
        .as_ref()
        .and_then(|debug| debug.get("why_not_moving"))
        .and_then(|reason| reason.as_str())
        .is_some_and(|reason| reason.contains("foreground Lua")));
}

#[tokio::test]
async fn real_robot_slow_runner_clears_bump_only_after_escape_finishes() {
    let stop_attempts = Arc::new(AtomicUsize::new(0));
    let clear_attempts = Arc::new(AtomicUsize::new(0));
    let body = ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc::new(AtomicUsize::new(0)),
        careful_mode_attempts: Arc::new(AtomicUsize::new(0)),
        bump_escape_commands: Arc::new(Mutex::new(Vec::new())),
        stop_attempts: Arc::clone(&stop_attempts),
        clear_attempts: Arc::clone(&clear_attempts),
        bump_active: false,
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);
    runner.possession_recovery.latch = Some(SafetyLatchKind::Bump);
    runner.possession_recovery.hazard_generation = 42;
    runner.possession_recovery.phase = PossessionRecoveryPhase::Escaping;
    runner.possession_recovery.command_attempts = 1;

    let _ = runner.tick_slow_manual().await.unwrap();
    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(stop_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(clear_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(runner.possession_recovery.latch, None);
    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
}

#[tokio::test]
async fn real_robot_slow_runner_applies_executive_motion_when_explicitly_authorized() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), StubRuntime)
        .with_brainstem_interface(serde_json::json!({
            "verbs": ["status", "get_events", "cmd_vel"]
        }))
        .with_autonomous_motion(true);

    let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(motor_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(
        motors.lock().unwrap().as_slice(),
        &[MotorCommand {
            forward: 0.05,
            turn: 0.0,
        }]
    );
    assert_eq!(
        snapshot
            .action_debug
            .as_ref()
            .and_then(|debug| debug.get("autonomous_hardware_gate"))
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        tick.frame
            .now
            .extensions
            .get("brainstem.events")
            .and_then(|extension| extension.get("events"))
            .and_then(|events| events.as_array())
            .map(Vec::len),
        Some(0)
    );
    assert_eq!(
        tick.frame.now.extensions["brainstem.interface"]["underlying_body_private"],
        serde_json::json!(true)
    );
}

#[tokio::test]
async fn real_robot_slow_runner_waits_for_runtime_tick_without_backoff() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let tick_attempts = Arc::new(AtomicUsize::new(0));
    let runtime = SlowRuntime {
        tick_attempts: Arc::clone(&tick_attempts),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), runtime);
    runner.tick_ms = 25;

    let (_first_snapshot, first_tick) = runner.tick_slow_manual().await.unwrap();
    let (_second_snapshot, second_tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(tick_attempts.load(Ordering::SeqCst), 2);
    assert_eq!(motor_attempts.load(Ordering::SeqCst), 2);
    assert_eq!(
        motors.lock().unwrap().as_slice(),
        &[MotorCommand::stop(), MotorCommand::stop()]
    );
    assert!(first_tick.frame.notes.is_empty());
    assert!(second_tick.frame.notes.is_empty());
}

#[tokio::test]
async fn real_robot_slow_runner_recovers_history_gap_by_stopping() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let event_polls = Arc::new(AtomicUsize::new(0));
    let body = HistoryGapCockpit {
        inner: CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        },
        event_polls: Arc::clone(&event_polls),
        gap_poll: 1,
    };
    let runtime = SlowRuntime {
        tick_attempts: Arc::new(AtomicUsize::new(0)),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), runtime);

    runner.tick_slow_manual().await.unwrap();
    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(event_polls.load(Ordering::SeqCst), 2);
    assert_eq!(
        tick.frame.now.extensions["brainstem.events"]["dropped_before_seq"],
        serde_json::json!(3)
    );
    assert!(motor_attempts.load(Ordering::SeqCst) >= 2);
    assert_eq!(motors.lock().unwrap().last(), Some(&MotorCommand::stop()));
}

#[tokio::test]
async fn real_robot_slow_runner_recovers_motion_safety_poll_history_gap_by_stopping() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let event_polls = Arc::new(AtomicUsize::new(0));
    let body = HistoryGapCockpit {
        inner: CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        },
        event_polls: Arc::clone(&event_polls),
        gap_poll: 1,
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);

    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(
        tick.chosen_action,
        Some(ActionPrimitive::Go {
            intensity: 0.2,
            duration_ms: 100,
        })
    );
    assert_eq!(event_polls.load(Ordering::SeqCst), 2);
    assert!(motor_attempts.load(Ordering::SeqCst) >= 2);
    assert_eq!(motors.lock().unwrap().last(), Some(&MotorCommand::stop()));
}

#[tokio::test]
async fn real_robot_slow_runner_recovers_motion_stop_events_by_stopping() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let event_polls = Arc::new(AtomicUsize::new(0));
    let body = MotionStopEventsCockpit {
        inner: CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        },
        event_polls: Arc::clone(&event_polls),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);

    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(
        tick.chosen_action,
        Some(ActionPrimitive::Go {
            intensity: 0.2,
            duration_ms: 100,
        })
    );
    assert_eq!(event_polls.load(Ordering::SeqCst), 2);
    assert!(motor_attempts.load(Ordering::SeqCst) >= 2);
    assert_eq!(motors.lock().unwrap().last(), Some(&MotorCommand::stop()));

    let (recovery_snapshot, _recovery_tick) = runner.tick_slow_manual().await.unwrap();
    assert_eq!(
        recovery_snapshot
            .action_debug
            .as_ref()
            .and_then(|debug| debug.get("possession_recovery"))
            .and_then(|debug| debug.get("latched")),
        Some(&serde_json::json!("Bump"))
    );
}

#[tokio::test]
async fn real_robot_slow_runner_treats_command_rejected_as_motion_feedback() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let rejection_attempts = Arc::new(AtomicUsize::new(0));
    let body = RejectingMotionCockpit {
        inner: CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        },
        rejection_attempts: Arc::clone(&rejection_attempts),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);

    let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
    assert_eq!(rejection_attempts.load(Ordering::SeqCst), 1);
    let motors = motors.lock().unwrap();
    assert_eq!(motors.last(), Some(&MotorCommand::stop()));
    assert_eq!(
            snapshot
                .action_debug
                .as_ref()
                .and_then(|debug| debug.get("why_not_moving"))
                .and_then(|reason| reason.as_str()),
            Some(
                "brainstem rejected motion command #42: stale_sequence; pausing motion retries for 1000 ms"
            )
        );
    assert_eq!(
        snapshot
            .action_debug
            .as_ref()
            .and_then(|debug| debug.get("motor_applied"))
            .and_then(|value| value.as_bool()),
        Some(false)
    );
}

#[tokio::test]
async fn real_robot_slow_runner_pauses_motion_after_command_rejection() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let rejection_attempts = Arc::new(AtomicUsize::new(0));
    let body = RejectingMotionCockpit {
        inner: CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        },
        rejection_attempts: Arc::clone(&rejection_attempts),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);

    let (_first_snapshot, _first_tick) = runner.tick_slow_manual().await.unwrap();
    let (second_snapshot, second_tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(second_tick.chosen_action, Some(ActionPrimitive::Stop));
    assert_eq!(rejection_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(motors.lock().unwrap().last(), Some(&MotorCommand::stop()));
    assert!(second_snapshot
        .action_debug
        .as_ref()
        .and_then(|debug| debug.get("why_not_moving"))
        .and_then(|reason| reason.as_str())
        .is_some_and(|reason| reason.contains("pausing motion retries")));
    assert_eq!(
        second_snapshot
            .action_debug
            .as_ref()
            .and_then(|debug| debug.get("motion_rejection"))
            .and_then(|debug| debug.get("count")),
        Some(&serde_json::json!(1))
    );
}

#[tokio::test]
async fn real_robot_slow_runner_latches_stuck_after_repeated_command_rejections() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let rejection_attempts = Arc::new(AtomicUsize::new(0));
    let body = RejectingMotionCockpit {
        inner: CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        },
        rejection_attempts: Arc::clone(&rejection_attempts),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);
    let now_ms = wall_time_ms();
    runner.motion_rejection = MotionRejectionState {
        first_ms: now_ms,
        last_ms: now_ms,
        latest_command_id: 41,
        latest_reason: Some("busy".to_string()),
        count: MOTION_REJECTION_STUCK_AFTER - 1,
        ..MotionRejectionState::default()
    };

    let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
    assert!(snapshot
        .action_debug
        .as_ref()
        .and_then(|debug| debug.get("why_not_moving"))
        .and_then(|reason| reason.as_str())
        .is_some_and(|reason| reason.contains("operator intervention needed")));
    assert_eq!(
        snapshot
            .action_debug
            .as_ref()
            .and_then(|debug| debug.get("motion_rejection"))
            .and_then(|debug| debug.get("stuck")),
        Some(&serde_json::json!(true))
    );
}

struct ManualRuntime;

#[async_trait::async_trait]
impl RuntimeLoop for ManualRuntime {
    async fn tick(
        &mut self,
        mut now: Now,
        _latent: ExperienceLatent,
        _futures: Vec<FuturePrediction>,
    ) -> Result<RuntimeTick> {
        let input = ReignInput {
            id: Uuid::new_v4(),
            issued_at_ms: now.t_ms,
            expires_at_ms: now.t_ms + 300,
            source: ReignSource::WebRemote,
            mode: ReignMode::Direct,
            command: pete_actions::ReignCommand::Go {
                intensity: 0.50,
                duration_ms: 300,
            },
            priority: 1.0,
            note: None,
        };
        now.reign.latest = Some(input.clone());
        let action = input.command.to_action().unwrap();
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
                reign_input: Some(input),
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
        })
    }
}

struct QueueOnlyRuntime {
    queue: Arc<Mutex<ReignQueue>>,
    tick_attempts: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl RuntimeLoop for QueueOnlyRuntime {
    async fn tick(
        &mut self,
        _now: Now,
        _latent: ExperienceLatent,
        _futures: Vec<FuturePrediction>,
    ) -> Result<RuntimeTick> {
        self.tick_attempts.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("slow direct hardware should bypass runtime tick")
    }

    fn reign_sense(&self, now_ms: TimeMs) -> Result<ReignSense> {
        Ok(self.queue.lock().unwrap().sense(now_ms))
    }
}

#[tokio::test]
async fn real_robot_slow_runner_applies_only_clamped_webremote_direct_motor() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let mut runner =
        RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), ManualRuntime);

    let (_snapshot, _tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(motor_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(
        motors.lock().unwrap().as_slice(),
        &[MotorCommand {
            forward: 0.05,
            turn: 0.0
        }]
    );
}

#[tokio::test]
async fn real_robot_slow_direct_webremote_bypasses_slow_runtime_tick() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let mut body_sense = BodySense {
        last_update_ms: 100,
        ..BodySense::default()
    };
    body_sense.cliff_sensors.front_left = 0.96;
    body_sense.cliff_sensors.front_right = 0.82;
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: body_sense,
    };
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    queue.lock().unwrap().push(ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: 100,
        expires_at_ms: wall_time_ms().saturating_add(500),
        source: ReignSource::WebRemote,
        mode: ReignMode::Direct,
        command: ReignCommand::Go {
            intensity: 0.50,
            duration_ms: 300,
        },
        priority: 1.0,
        note: None,
    });
    let tick_attempts = Arc::new(AtomicUsize::new(0));
    let runtime = QueueOnlyRuntime {
        queue,
        tick_attempts: Arc::clone(&tick_attempts),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), runtime);

    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(tick_attempts.load(Ordering::SeqCst), 0);
    assert_eq!(motor_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(
        motors.lock().unwrap().as_slice(),
        &[MotorCommand {
            forward: 0.05,
            turn: 0.0
        }]
    );
    assert_eq!(
        tick.frame
            .now
            .extensions
            .get("action.motion_bridge")
            .and_then(|value| value.get("runtime_bypassed"))
            .and_then(|value| value.as_bool()),
        Some(true)
    );
}

#[tokio::test]
async fn real_robot_slow_direct_webremote_stops_locally_while_charging() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: BodySense {
            charging: true,
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    queue.lock().unwrap().push(ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: 100,
        expires_at_ms: wall_time_ms().saturating_add(500),
        source: ReignSource::WebRemote,
        mode: ReignMode::Direct,
        command: ReignCommand::Go {
            intensity: 0.50,
            duration_ms: 300,
        },
        priority: 1.0,
        note: None,
    });
    let tick_attempts = Arc::new(AtomicUsize::new(0));
    let runtime = QueueOnlyRuntime {
        queue,
        tick_attempts: Arc::clone(&tick_attempts),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), runtime);

    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(tick_attempts.load(Ordering::SeqCst), 0);
    assert_eq!(motor_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(motors.lock().unwrap().as_slice(), &[MotorCommand::stop()]);
    assert_eq!(
        tick.frame
            .now
            .extensions
            .get("action.motion_bridge")
            .and_then(|value| value.get("why_not_moving"))
            .and_then(|value| value.as_str()),
        Some("charging active")
    );
}

#[tokio::test]
async fn real_robot_slow_direct_gamepad_bypasses_slow_runtime_tick() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    queue.lock().unwrap().push(ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: 100,
        expires_at_ms: wall_time_ms().saturating_add(500),
        source: ReignSource::Gamepad,
        mode: ReignMode::Direct,
        command: ReignCommand::Drive {
            forward: 0.50,
            turn: -0.50,
            duration_ms: 300,
        },
        priority: 1.0,
        note: None,
    });
    let tick_attempts = Arc::new(AtomicUsize::new(0));
    let runtime = QueueOnlyRuntime {
        queue,
        tick_attempts: Arc::clone(&tick_attempts),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), runtime);

    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(tick_attempts.load(Ordering::SeqCst), 0);
    assert_eq!(motor_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(
        motors.lock().unwrap().as_slice(),
        &[MotorCommand {
            forward: 0.05,
            turn: -0.5
        }]
    );
    assert!(matches!(
        tick.frame.reign_input.as_ref().map(|input| &input.source),
        Some(ReignSource::Gamepad)
    ));
}

#[tokio::test]
async fn real_robot_slow_direct_webremote_chirp_bypasses_runtime_without_motor() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    queue.lock().unwrap().push(ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: 100,
        expires_at_ms: wall_time_ms().saturating_add(500),
        source: ReignSource::WebRemote,
        mode: ReignMode::Direct,
        command: ReignCommand::Chirp {
            pattern: ChirpPattern::Confirm,
        },
        priority: 1.0,
        note: None,
    });
    let tick_attempts = Arc::new(AtomicUsize::new(0));
    let runtime = QueueOnlyRuntime {
        queue,
        tick_attempts: Arc::clone(&tick_attempts),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), runtime);

    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(tick_attempts.load(Ordering::SeqCst), 0);
    assert_eq!(motor_attempts.load(Ordering::SeqCst), 0);
    assert!(motors.lock().unwrap().is_empty());
    assert!(matches!(
        tick.chosen_action,
        Some(ActionPrimitive::Chirp {
            pattern: ChirpPattern::Confirm
        })
    ));
    assert!(matches!(
        tick.frame.reign_input.as_ref().map(|input| &input.command),
        Some(ReignCommand::Chirp {
            pattern: ChirpPattern::Confirm
        })
    ));
}

#[tokio::test]
async fn real_robot_slow_direct_webremote_speak_bypasses_runtime_without_motor() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    queue.lock().unwrap().push(ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: 100,
        expires_at_ms: wall_time_ms().saturating_add(500),
        source: ReignSource::WebRemote,
        mode: ReignMode::Direct,
        command: ReignCommand::Speak {
            text: "hello from reign".to_string(),
        },
        priority: 1.0,
        note: None,
    });
    let tick_attempts = Arc::new(AtomicUsize::new(0));
    let runtime = QueueOnlyRuntime {
        queue,
        tick_attempts: Arc::clone(&tick_attempts),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), runtime);

    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(tick_attempts.load(Ordering::SeqCst), 0);
    assert_eq!(motor_attempts.load(Ordering::SeqCst), 0);
    assert!(motors.lock().unwrap().is_empty());
    assert!(matches!(
        tick.chosen_action,
        Some(ActionPrimitive::Speak { ref text }) if text == "hello from reign"
    ));
    assert!(matches!(
        tick.frame.reign_input.as_ref().map(|input| &input.command),
        Some(ReignCommand::Speak { text }) if text == "hello from reign"
    ));
}
