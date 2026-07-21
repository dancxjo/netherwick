fn arena() -> ArenaConfig {
    ArenaConfig {
        width_m: 4.0,
        height_m: 4.0,
    }
}

fn test_body(x_m: f32, y_m: f32, battery_level: f32, last_update_ms: u64) -> BodySense {
    let mut body = BodySense::default();
    body.odometry.x_m = x_m;
    body.odometry.y_m = y_m;
    body.battery_level = battery_level;
    body.last_update_ms = last_update_ms;
    body
}

fn stuck_test_snapshot(x_m: f32, y_m: f32, battery_level: f32) -> WorldSnapshot {
    let mut snapshot = WorldSnapshot::default();
    snapshot.body = test_body(x_m, y_m, battery_level, 100);
    snapshot.range.nearest_m = Some(0.12);
    snapshot.range.beams = vec![0.05, 0.08, 0.10, 0.09, 0.05];
    snapshot.extensions.push(ExtensionSense {
        schema_version: 1,
        name: "sim.world".to_string(),
        values: vec![4.0, 4.0, 0.0],
    });
    snapshot
}

#[test]
fn stuck_detector_uses_rolling_low_displacement_window() {
    let mut detector = StuckRecoveryController::default();
    let action = ActionPrimitive::Explore {
        style: ExploreStyle::RandomWalk,
        duration_ms: 1_000,
    };

    for _ in 0..=STUCK_LOW_DISPLACEMENT_TICKS {
        detector.observe(&stuck_test_snapshot(0.2, 0.2, 1.0), Some(&action));
    }

    let status = detector.status();
    assert!(status.active);
    assert!(status.corner_trap);
    assert_eq!(status.stuck_ticks, STUCK_LOW_DISPLACEMENT_TICKS);
    assert!(status.event_started);
    assert_eq!(status.recovery_attempts, 1);
    assert_eq!(status.duration_ticks, 1);
    assert!(!status.reset_due);
}

#[test]
fn recovered_stuck_event_reports_attempt_and_duration() {
    let mut detector = StuckRecoveryController::default();
    let action = ActionPrimitive::Explore {
        style: ExploreStyle::RandomWalk,
        duration_ms: 1_000,
    };

    for _ in 0..=STUCK_LOW_DISPLACEMENT_TICKS {
        detector.observe(&stuck_test_snapshot(0.2, 0.2, 1.0), Some(&action));
    }
    detector.observe(&stuck_test_snapshot(0.2, 0.2, 1.0), Some(&action));
    detector.observe(&stuck_test_snapshot(0.3, 0.2, 1.0), Some(&action));
    let status = detector.status();
    assert!(!status.active);
    assert!(status.recovered);
    assert_eq!(status.recovery_attempts, 1);
    assert!(status.duration_ticks >= 2);

    let extension = detector.extension(100);
    assert_eq!(extension.values.get(7).copied(), Some(1.0));
    assert_eq!(extension.values.get(11).copied(), Some(1.0));
    assert!(extension.values.get(3).copied().unwrap_or_default() >= 200.0);
}

#[test]
fn repeated_stuck_escalates_recovery_instead_of_resetting() {
    let mut detector = StuckRecoveryController::default();
    let action = ActionPrimitive::Explore {
        style: ExploreStyle::RandomWalk,
        duration_ms: 1_000,
    };

    for _ in 0..=STUCK_LOW_DISPLACEMENT_TICKS {
        detector.observe(&stuck_test_snapshot(0.2, 0.2, 1.0), Some(&action));
    }
    detector.finish_recovery_success();
    detector.clearance_m = Some(0.10);
    detector.recovery_attempts = 1;
    detector.trap_anchor = Some((0.2, 0.2));

    for _ in 0..=STUCK_LOW_DISPLACEMENT_TICKS {
        detector.observe(&stuck_test_snapshot(0.2, 0.2, 1.0), Some(&action));
    }
    let mut snapshot = stuck_test_snapshot(0.2, 0.2, 1.0);
    detector.annotate_snapshot(&mut snapshot, 100);

    let status = detector.status();
    assert!(!status.reset_due);
    assert!(status.active);
    assert_eq!(status.repeated_trap_count, 1);
    let values = &snapshot
        .extensions
        .iter()
        .find(|extension| extension.name == "sim.stuck")
        .unwrap()
        .values;
    assert_eq!(values.get(9).copied(), Some(0.0));
    assert_eq!(values.get(12).copied(), Some(1.0));
}

#[test]
fn dead_battery_state_is_reported_without_starting_recovery() {
    let mut detector = StuckRecoveryController::default();
    let action = ActionPrimitive::Explore {
        style: ExploreStyle::RandomWalk,
        duration_ms: 1_000,
    };

    for _ in 0..=STUCK_LOW_DISPLACEMENT_TICKS {
        detector.observe(&stuck_test_snapshot(0.2, 0.2, 0.0), Some(&action));
    }
    let mut snapshot = stuck_test_snapshot(0.2, 0.2, 0.0);
    detector.annotate_snapshot(&mut snapshot, 100);

    let status = detector.status();
    assert!(status.dead_battery);
    assert!(!status.active);
    let values = &snapshot
        .extensions
        .iter()
        .find(|extension| extension.name == "sim.stuck")
        .unwrap()
        .values;
    assert_eq!(values.get(8).copied(), Some(1.0));
}

#[test]
fn stopped_column_trap_still_triggers_stuck_recovery() {
    let mut detector = StuckRecoveryController::default();
    let action = ActionPrimitive::Stop;

    for _ in 0..=STUCK_LOW_DISPLACEMENT_TICKS {
        detector.observe(&stuck_test_snapshot(2.0, 2.0, 1.0), Some(&action));
    }

    let status = detector.status();
    assert!(status.active);
    assert_eq!(status.trap_kind, TrapKind::Column);
    assert_eq!(status.stuck_ticks, STUCK_LOW_DISPLACEMENT_TICKS);
}

#[test]
fn bump_left_chooses_rightward_escape() {
    let mut body = test_body(1.0, 1.0, 1.0, 100);
    body.flags.bump_left = true;
    let now = Now::blank(100, body);

    assert_eq!(
        hard_safety_action(&now),
        Some(ActionPrimitive::Turn {
            direction: TurnDir::Right,
            intensity: 0.7,
            duration_ms: 1_200
        })
    );
}

#[test]
fn bump_right_chooses_leftward_escape() {
    let mut body = test_body(1.0, 1.0, 1.0, 100);
    body.flags.bump_right = true;
    let now = Now::blank(100, body);

    assert_eq!(
        hard_safety_action(&now),
        Some(ActionPrimitive::Turn {
            direction: TurnDir::Left,
            intensity: 0.7,
            duration_ms: 1_200
        })
    );
}

#[test]
fn every_cliff_sensor_selects_stop_before_hardware_gate() {
    for sensor in ["left", "front_left", "front_right", "right"] {
        let mut body = test_body(1.0, 1.0, 1.0, 100);
        match sensor {
            "left" => body.flags.cliff_left = true,
            "front_left" => body.flags.cliff_front_left = true,
            "front_right" => body.flags.cliff_front_right = true,
            "right" => body.flags.cliff_right = true,
            _ => unreachable!(),
        }
        let now = Now::blank(100, body.clone());

        assert_eq!(
            hard_safety_action(&now),
            Some(ActionPrimitive::Stop),
            "{sensor}"
        );
        assert_eq!(
            real_slow_body_block_reason(&body).as_deref(),
            Some("cliff sensor active"),
            "{sensor}"
        );
    }
}

fn test_ledger_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("pete-{name}-{}", Uuid::new_v4()));
    let _ = fs::remove_dir_all(&root);
    root
}

fn danger_checkpoint_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("pete-{name}-checkpoint-{}", Uuid::new_v4()));
    let _ = fs::remove_dir_all(&root);
    root
}

fn write_test_danger_checkpoint(root: &Path, action: ActionPrimitive) {
    let mut body = test_body(1.0, 1.0, 0.8, 7);
    body.velocity.forward_m_s = 0.05;
    let now = Now::blank(100, body);
    let input = DangerInput::from_parts(Vec::new(), Some(&action), &now);
    let mut trainer = DangerNetTrainer::new(input.flat_features().len());
    trainer
        .train_step(
            &input,
            &pete_experience::DangerTarget {
                bump: 0.2,
                ..pete_experience::DangerTarget::default()
            },
        )
        .unwrap();
    trainer.save_checkpoint(root).unwrap();
}

fn write_test_charge_checkpoint(root: &Path, action: ActionPrimitive) {
    let mut body = test_body(1.0, 1.0, 0.2, 7);
    body.charging = false;
    let now = Now::blank(100, body);
    let input = ChargeInput::from_parts(Vec::new(), Some(&action), &now);
    let mut trainer = ChargeNetTrainer::new(input.flat_features().len());
    trainer
        .train_step(
            &input,
            &pete_experience::ChargeTarget {
                charging_started: 1.0,
                battery_delta: 0.03,
                charging_after: 1.0,
            },
        )
        .unwrap();
    trainer.save_checkpoint(root).unwrap();
}

fn write_test_action_value_checkpoint(root: &Path, action: ActionPrimitive) {
    let mut body = test_body(1.0, 1.0, 0.2, 7);
    body.charging = false;
    let now = Now::blank(100, body);
    let input = ActionValueInput::from_parts(Vec::new(), Some(&action), &now);
    let mut trainer = ActionValueNetTrainer::new(input.flat_features().len());
    trainer
        .train_step(&input, &pete_experience::ActionValueTarget { value: 0.25 })
        .unwrap();
    trainer.save_checkpoint(root).unwrap();
}

fn write_test_future_checkpoint(root: &Path, action: ActionPrimitive) {
    let now = Now::blank(100, test_body(1.0, 1.0, 0.8, 100));
    let latent = ExperienceLatent {
        t_ms: now.t_ms,
        z: Vec::new(),
        reconstruction_error: 0.0,
        prediction_error: 0.0,
        confidence: 0.0,
    };
    let input = FutureInput {
        latent: latent.clone(),
        action,
        offset_ms: 100,
    };
    let mut trainer = FutureNetTrainer::new(input.flat_features().len(), 1);
    trainer.train_step(&input, &[0.0]).unwrap();
    trainer.save_checkpoint(root).unwrap();
}

fn write_test_ear_next_checkpoint(root: &Path, action: ActionPrimitive) {
    let body = test_body(1.0, 1.0, 0.8, 7);
    let mut now = Now::blank(100, body);
    now.ear.features = vec![vec![0.2, 0.4, 0.6, 0.8]];
    let input = EarNextInput::from_parts(Vec::new(), Some(&action), &now, 100);
    let mut trainer = EarNextNetTrainer::new(input.flat_features().len(), 4);
    trainer
        .train_step(
            &input,
            &pete_experience::EarNextTarget {
                features: vec![0.2, 0.4, 0.6, 0.8],
                ..pete_experience::EarNextTarget::default()
            },
        )
        .unwrap();
    trainer.save_checkpoint(root).unwrap();
}

fn write_test_experience_checkpoint(root: &Path) {
    let mut body = test_body(1.0, 1.0, 0.8, 7);
    body.velocity.forward_m_s = 0.1;
    let mut now = Now::blank(100, body);
    now.eye.frames = vec![vec![0.2, 0.4, 0.6, 0.8]];
    now.ear.features = vec![vec![0.2, 0.4, 0.6, 0.8]];
    now.memory.place_familiarity = 0.6;
    now.drives.curiosity = 0.4;
    let input = experience_encode_input_from_now(&now);
    let target = experience_decode_target_from_now(&now);
    let mut trainer =
        ExperienceAutoencoderTrainer::new(input.flat_features().len(), 8, target.feature_lengths());
    trainer.train_step(&input, &target).unwrap();
    trainer.save_checkpoint(root).unwrap();
}

fn read_transitions(root: &Path) -> Vec<ExperienceTransition> {
    let mut out = Vec::new();
    read_transition_paths(root, &mut out);
    out
}

fn read_transition_paths(path: &Path, out: &mut Vec<ExperienceTransition>) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            read_transition_paths(&path, out);
        } else if path.file_name().and_then(|name| name.to_str()) == Some("transitions.jsonl") {
            let Ok(contents) = fs::read_to_string(path) else {
                continue;
            };
            out.extend(
                contents
                    .lines()
                    .filter(|line| !line.trim().is_empty())
                    .filter_map(|line| serde_json::from_str(line).ok()),
            );
        }
    }
}

#[test]
fn robot_initialized_typescript_behavior_emits_bringup_mouth_sequence() {
    let mut behavior = RobotInitializedScriptBehavior;
    let input = RobotInitializedEventInput {
        t_ms: 42,
        mode: "read-only".to_string(),
        body: "mock Create body connected".to_string(),
        battery_percent: Some(100),
        charging: Some(false),
        active_sensors: 2,
        requested_sensors: 3,
        ledger: "data/ledger/test".to_string(),
        tick_ms: 100,
        dashboard: Some("127.0.0.1:3000".to_string()),
        capture: None,
    };

    let output = behavior.infer(&input).unwrap();

    assert!(matches!(
        output.actions.first(),
        Some(EventScriptAction::Song { name }) if name == "bring_up"
    ));
    assert!(output.actions.iter().any(|action| matches!(
        action,
        EventScriptAction::Chirp {
            pattern: ChirpPattern::Confirm
        }
    )));
    assert!(output.actions.iter().any(|action| matches!(
        action,
        EventScriptAction::Say { text }
            if text.contains("Pete robot initialization complete")
    )));
}

fn test_reign_input(
    issued_at_ms: u64,
    mode: ReignMode,
    command: ReignCommand,
    ttl_ms: u64,
) -> ReignInput {
    ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms,
        expires_at_ms: issued_at_ms + ttl_ms,
        source: ReignSource::WebRemote,
        mode,
        command,
        priority: 1.0,
        note: None,
    }
}
