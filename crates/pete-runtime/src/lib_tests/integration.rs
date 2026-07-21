#[tokio::test]
async fn sim_runner_writes_frames_and_transitions() {
    let root = test_ledger_root("sim-runner-writes");
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), SimpleConductor::default());
    let (world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(10).await.unwrap();

    let frames = ledger.recent(20).await.unwrap();
    let transitions = read_transitions(&root);
    assert!(frames.len() >= 10);
    assert!(transitions.len() >= 9);
    assert!(transitions.iter().any(|transition| {
        transition.before.body.odometry.x_m != transition.after.body.odometry.x_m
            || transition.before.body.odometry.y_m != transition.after.body.odometry.y_m
            || transition.before.body.odometry.heading_rad
                != transition.after.body.odometry.heading_rad
    }));
}

#[tokio::test]
async fn tick_records_erased_behavior_runs() {
    let root = test_ledger_root("runtime-behavior-runs");
    let ledger = JsonlLedger::new(&root);
    let mut runtime = test_runtime(ledger, FixedConductor::new(ActionPrimitive::Stop));

    let tick = runtime
        .tick(
            Now::blank(100, test_body(1.0, 1.0, 0.8, 100)),
            ExperienceLatent::default(),
            Vec::new(),
        )
        .await
        .unwrap();

    for behavior_id in [
        "danger",
        "charge",
        "future",
        "action_value",
        "eye_next",
        "ear_next",
    ] {
        assert!(
            tick.frame
                .behavior_runs
                .iter()
                .any(|run| run.behavior_id == behavior_id),
            "missing behavior run for {behavior_id}"
        );
    }

    let _ = fs::remove_dir_all(root);
}

#[tokio::test]
async fn tick_runs_bump_event_script_and_records_safety_trace() {
    let root = test_ledger_root("runtime-bump-event-script");
    let ledger = JsonlLedger::new(&root);
    let mut runtime = test_runtime(
        ledger,
        FixedConductor::new(ActionPrimitive::Go {
            intensity: 0.3,
            duration_ms: 500,
        }),
    );
    let mut body = test_body(1.0, 1.0, 0.05, 100);
    body.flags.bump_left = true;
    let now = Now::blank(100, body);

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    assert_eq!(
        tick.chosen_action,
        Some(ActionPrimitive::Go {
            intensity: 0.3,
            duration_ms: 500
        })
    );
    assert!(tick
        .frame
        .now
        .extensions
        .get("safety.vetoed")
        .and_then(|value| value.as_bool())
        .unwrap_or(false));
    assert!(tick
        .frame
        .behavior_runs
        .iter()
        .any(|run| run.behavior_id == "event_bump" && run.regime == BehaviorRegime::ShadowTrain));
    let sequence = tick
        .frame
        .now
        .extensions
        .get("event_scripts")
        .and_then(|value| value.get("bump"))
        .cloned()
        .and_then(|value| serde_json::from_value::<SafeScriptSequence>(value).ok())
        .unwrap();
    assert_eq!(sequence.actions.len(), 5);
    assert!(matches!(
        sequence.actions.first().map(|action| &action.requested),
        Some(EventScriptAction::Chirp {
            pattern: ChirpPattern::Warning
        })
    ));
    assert!(matches!(
        sequence.actions.get(1).map(|action| &action.requested),
        Some(EventScriptAction::Say { .. } | EventScriptAction::Song { .. })
    ));
    assert!(sequence.actions.last().unwrap().vetoed);

    let _ = fs::remove_dir_all(root);
}

#[test]
fn target_extractors_create_danger_charge_future_and_action_value_samples() {
    let action = ActionPrimitive::Go {
        intensity: 0.4,
        duration_ms: 1_000,
    };
    let mut before = Now::blank(100, test_body(1.0, 1.0, 0.0, 100));
    before.body.battery_level = 0.5;
    let mut after = before.clone();
    after.t_ms = 200;
    after.body.last_update_ms = 200;
    after.body.flags.bump_left = true;
    after.body.battery_level = 0.55;
    after.body.charging = true;
    after.eye.frames = vec![vec![0.25, 0.5, 0.75]];
    after.ear.features = vec![vec![0.2, 0.4], vec![0.6, 0.8]];
    let transition = ExperienceTransition {
        id: Uuid::new_v4(),
        before_frame_id: Uuid::new_v4(),
        before,
        before_z: ExperienceLatent {
            t_ms: 100,
            z: vec![0.1, 0.2],
            reconstruction_error: 0.0,
            prediction_error: 0.0,
            confidence: 0.8,
        },
        action: Some(action.clone()),
        predicted_futures: Vec::new(),
        after,
        after_z: ExperienceLatent {
            t_ms: 200,
            z: vec![0.3, 0.4],
            reconstruction_error: 0.0,
            prediction_error: 0.0,
            confidence: 0.9,
        },
        reward: Reward { value: 0.25 },
        surprise: SurpriseSense {
            total: 0.6,
            prediction_error: 0.1,
            ..SurpriseSense::default()
        },
        created_at_ms: 200,
    };

    let danger = DangerTargetExtractor.extract(&transition).unwrap().unwrap();
    assert_eq!(danger.source, TrainingSource::WorldOutcome);
    assert_eq!(danger.expected.bump_risk, 1.0);

    let charge = ChargeTargetExtractor.extract(&transition).unwrap().unwrap();
    assert_eq!(charge.expected.charge_probability, 1.0);
    assert!(charge.expected.expected_battery_delta > 0.0);

    let future = FutureTargetExtractor { offset_ms: 1_000 }
        .extract(&transition)
        .unwrap()
        .unwrap();
    assert_eq!(future.input.action, action);
    assert_eq!(future.expected.predicted_z, vec![0.3, 0.4]);

    let action_value = ActionValueTargetExtractor
        .extract(&transition)
        .unwrap()
        .unwrap();
    assert_eq!(action_value.source, TrainingSource::WorldOutcome);
    assert!((action_value.expected.value - 0.18).abs() < 0.0001);
    assert_eq!(action_value.expected.confidence, 1.0);

    let eye_next = EyeNextTargetExtractor { offset_ms: 100 }
        .extract(&transition)
        .unwrap()
        .unwrap();
    assert_eq!(eye_next.source, TrainingSource::WorldOutcome);
    assert_eq!(eye_next.expected.width, 64);
    assert_eq!(eye_next.expected.height, 48);
    assert_eq!(eye_next.expected.rgb.len(), 64 * 48 * 3);

    let ear_next = EarNextTargetExtractor { offset_ms: 100 }
        .extract(&transition)
        .unwrap()
        .unwrap();
    assert_eq!(ear_next.source, TrainingSource::WorldOutcome);
    assert_eq!(ear_next.expected.features, vec![0.2, 0.4, 0.6, 0.8]);
    assert!(ear_next.expected.pcm.is_empty());
}

#[test]
fn ear_next_target_extractor_skips_missing_ear_frame() {
    let before = Now::blank(100, test_body(1.0, 1.0, 0.0, 100));
    let mut after = before.clone();
    after.t_ms = 200;
    let transition = ExperienceTransition {
        id: Uuid::new_v4(),
        before_frame_id: Uuid::new_v4(),
        before,
        before_z: ExperienceLatent {
            t_ms: 100,
            z: vec![0.1, 0.2],
            reconstruction_error: 0.0,
            prediction_error: 0.0,
            confidence: 0.8,
        },
        action: Some(ActionPrimitive::Stop),
        predicted_futures: Vec::new(),
        after,
        after_z: ExperienceLatent::default(),
        reward: Reward { value: 0.0 },
        surprise: SurpriseSense::default(),
        created_at_ms: 200,
    };

    let sample = EarNextTargetExtractor { offset_ms: 100 }
        .extract(&transition)
        .unwrap();

    assert!(sample.is_none());
}

#[test]
fn behavior_registry_default_has_all_replaceable_slots() {
    let mut registry = BehaviorRegistry::default();
    let now = Now::blank(100, test_body(1.0, 1.0, 0.0, 100));
    let latent = ExperienceLatent {
        t_ms: 100,
        z: vec![0.0; 4],
        reconstruction_error: 0.0,
        prediction_error: 0.0,
        confidence: 0.8,
    };
    let action = ActionPrimitive::Dock;

    let locomotion = registry
        .locomotion
        .infer(&LocomotionInput::default(), 100)
        .unwrap();
    let danger = registry
        .danger
        .infer(&danger_behavior_input(&now, &latent, Some(&action)), 100)
        .unwrap();
    let charge = registry
        .charge
        .infer(&charge_behavior_input(&now, &latent, Some(&action)), 100)
        .unwrap();
    let future = registry
        .future
        .infer(
            &FutureInput {
                latent: latent.clone(),
                action: action.clone(),
                offset_ms: 1_000,
            },
            100,
        )
        .unwrap();
    let action_value = registry
        .action_value
        .infer(
            &action_value_behavior_input(&now, &latent, Some(&action), None, None),
            100,
        )
        .unwrap();
    let eye_next = registry
        .eye_next
        .infer(
            &eye_next_behavior_input(&now, &latent, Some(&action), 100),
            100,
        )
        .unwrap();
    let ear_next = registry
        .ear_next
        .infer(
            &ear_next_behavior_input(&now, &latent, Some(&action), 100),
            100,
        )
        .unwrap();
    let experience = registry
        .experience
        .infer(&ExperienceBehaviorInput::from_now(&now), 100)
        .unwrap();

    assert_eq!(locomotion.record.behavior_id, "locomotion");
    assert_eq!(experience.record.behavior_id, "experience");
    assert_eq!(danger.record.behavior_id, "danger");
    assert_eq!(charge.record.behavior_id, "charge");
    assert_eq!(future.record.behavior_id, "future");
    assert_eq!(action_value.record.behavior_id, "action_value");
    assert_eq!(eye_next.record.behavior_id, "eye_next");
    assert_eq!(ear_next.record.behavior_id, "ear_next");
    assert!(locomotion.record.hardcoded_output.is_some());
    assert!(experience.record.hardcoded_output.is_some());
    assert!(danger.record.hardcoded_output.is_some());
    assert!(charge.record.hardcoded_output.is_some());
    assert!(future.record.hardcoded_output.is_some());
    assert!(action_value.record.hardcoded_output.is_some());
    assert!(eye_next.record.hardcoded_output.is_some());
    assert!(ear_next.record.hardcoded_output.is_some());
}

#[test]
fn action_value_hardcoded_regime_returns_hardcoded_output() {
    let now = Now::blank(100, test_body(1.0, 1.0, 0.2, 100));
    let latent = ExperienceLatent {
        t_ms: 100,
        z: vec![0.0; 4],
        confidence: 0.8,
        ..ExperienceLatent::default()
    };
    let input =
        action_value_behavior_input(&now, &latent, Some(&ActionPrimitive::Dock), None, None);
    let mut behavior = action_value_behavior(
        BehaviorRegime::Hardcoded,
        None,
        FallbackPolicy::UseHardcoded,
    );

    let run = behavior.infer(&input, 100).unwrap();

    assert!(run.record.hardcoded_output.is_some());
    assert!(run.record.model_output.is_none());
    assert_eq!(run.record.selected_output, run.record.hardcoded_output);
}

#[test]
fn action_value_shadow_infer_records_model_and_selects_hardcoded() {
    let now = Now::blank(100, test_body(1.0, 1.0, 0.2, 100));
    let latent = ExperienceLatent {
        t_ms: 100,
        z: vec![0.0; 4],
        confidence: 0.8,
        ..ExperienceLatent::default()
    };
    let input =
        action_value_behavior_input(&now, &latent, Some(&ActionPrimitive::Dock), None, None);
    let trainer = ActionValueNetTrainer::new(input.input.flat_features().len());
    let mut behavior = action_value_behavior(
        BehaviorRegime::ShadowInfer,
        Some(trainer),
        FallbackPolicy::UseHardcoded,
    );

    let run = behavior.infer(&input, 100).unwrap();

    assert!(run.record.hardcoded_output.is_some());
    assert!(run.record.model_output.is_some());
    assert_eq!(run.record.selected_output, run.record.hardcoded_output);
}

#[test]
fn action_value_config_with_missing_checkpoint_falls_back_cleanly() {
    let config: BehaviorRegistryConfig = toml::from_str(
        r#"
            [behavior.action_value]
            regime = "shadow_infer"
            hardcoded = "action_value.handcoded"
            model = "action_value.burn.v0"
            checkpoint = "/tmp/pete-missing-action-value-checkpoint"
            fallback = "use_hardcoded"
            "#,
    )
    .unwrap();
    let mut stack = RuntimeModelStack::from_behavior_config(&config).unwrap();
    assert_eq!(
        stack.behaviors.action_value.regime,
        BehaviorRegime::ShadowInfer
    );

    let now = Now::blank(100, test_body(1.0, 1.0, 0.2, 100));
    let latent = ExperienceLatent {
        t_ms: 100,
        z: vec![0.0; 4],
        confidence: 0.8,
        ..ExperienceLatent::default()
    };
    let input =
        action_value_behavior_input(&now, &latent, Some(&ActionPrimitive::Dock), None, None);
    let run = stack
        .behaviors
        .action_value
        .infer(&input, now.t_ms)
        .unwrap();

    assert!(run.record.hardcoded_output.is_some());
    assert!(run.record.model_output.is_none());
}

#[tokio::test]
async fn sim_runner_applies_chosen_action_to_world() {
    let ledger = JsonlLedger::new(test_ledger_root("sim-runner-action-world"));
    let runtime = test_runtime(
        ledger,
        FixedConductor::new(ActionPrimitive::Go {
            intensity: 0.4,
            duration_ms: 1_000,
        }),
    );
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.5, 7));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let snapshot = runner.world.snapshot().await.unwrap();

    assert!(snapshot.body.odometry.x_m > 1.0);
    assert_eq!(runner.tick_count, 1);
}

#[tokio::test]
async fn sim_runner_go_and_explore_send_non_stop_motion_and_change_pose() {
    for (name, action) in [
        (
            "go",
            ActionPrimitive::Go {
                intensity: 0.4,
                duration_ms: 1_000,
            },
        ),
        (
            "explore",
            ActionPrimitive::Explore {
                style: ExploreStyle::RandomWalk,
                duration_ms: 1_000,
            },
        ),
    ] {
        let ledger = JsonlLedger::new(test_ledger_root(&format!("sim-runner-{name}-motor-bridge")));
        let runtime = test_runtime(ledger, FixedConductor::new(action.clone()));
        let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.5, 7));
        let mut runner = SimRunner::new(runtime, world, motors);
        let start = runner.world.body();
        let mut saw_non_zero_final_motor = false;
        let expected_selected_action = match &action {
            ActionPrimitive::Explore { duration_ms, .. } => ActionPrimitive::Drive {
                forward: 0.2,
                turn: 0.1,
                duration_ms: *duration_ms,
            },
            _ => action.clone(),
        };

        runner
            .run_steps_observing_ticks(5, |snapshot, tick| {
                let final_motor = final_motor_from_tick(tick);
                if !is_near_zero_motor(final_motor) {
                    saw_non_zero_final_motor = true;
                }
                assert_eq!(
                    snapshot.final_selected_action,
                    Some(expected_selected_action.clone())
                );
            })
            .await
            .unwrap();

        let end = runner.world.body();
        let delta = movement_delta_m(&start, &end);
        assert!(
            delta > 0.005,
            "{name} should move the simulated body, delta was {delta}"
        );
        assert!(saw_non_zero_final_motor, "{name} final motor was zero");
        assert!(
            !matches!(
                runner.world.last_motion_sent(),
                Some(MotionCommand::Stop) | None
            ),
            "{name} did not send non-stop motion to sim"
        );
    }
}

#[tokio::test]
async fn sim_runner_reaches_charger_gets_positive_reward() {
    let root = test_ledger_root("sim-runner-charger-reward");
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(
        ledger,
        FixedConductor::new(ActionPrimitive::Go {
            intensity: 0.4,
            duration_ms: 1_000,
        }),
    );
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    let mut body = test_body(1.0, 1.0, 0.2, 7);
    body.battery_level = 0.2;
    world.set_body(body);
    world.add_object(SimObject::charger("charger", "charger", 1.38, 1.0, 0.18));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(2).await.unwrap();
    let transitions = read_transitions(&root);

    let transition = transitions.last().unwrap();
    assert!(transition.after.body.charging);
    assert!(transition.reward.value > 0.0);
    assert!(transition.surprise.total > 0.0);
}

#[tokio::test]
async fn sim_runner_collision_sets_bump_and_negative_reward() {
    let root = test_ledger_root("sim-runner-collision-reward");
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(
        ledger,
        FixedConductor::new(ActionPrimitive::Go {
            intensity: 0.4,
            duration_ms: 1_000,
        }),
    );
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.8, 7));
    world.add_object(SimObject::obstacle("box", "box", 1.31, 1.0, 0.1));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(2).await.unwrap();
    let transitions = read_transitions(&root);

    let transition = transitions.last().unwrap();
    assert!(transition.after.body.flags.bump_left || transition.after.body.flags.bump_right);
    assert!(transition.reward.value < 0.0);
    assert!(transition.surprise.total > 0.0);
}

#[tokio::test]
async fn sim_runner_resets_dead_uncharging_battery_and_records_critique() {
    let root = test_ledger_root("sim-runner-dead-battery-reset");
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(
        ledger.clone(),
        FixedConductor::new(ActionPrimitive::Go {
            intensity: 0.4,
            duration_ms: 1_000,
        }),
    );
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.0, 7));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let snapshot = runner.world.snapshot().await.unwrap();
    let frames = ledger.recent(5).await.unwrap();
    let frame = frames.last().unwrap();

    assert_eq!(snapshot.body.battery_level, 1.0);
    assert!(!snapshot.body.charging);
    assert_eq!(snapshot.body.odometry.x_m, 2.0);
    assert_eq!(snapshot.body.odometry.y_m, 2.0);
    assert!(frame.llm_teaching.iter().any(|teaching| teaching
        .critique
        .as_deref()
        .is_some_and(|critique| { critique.contains("Dead battery away from the charger") })));
    assert!(frame
        .notes
        .iter()
        .any(|note| note.contains("VirtualDeadBattery")));

    let _ = fs::remove_dir_all(root);
}

#[tokio::test]
async fn sim_runner_gives_stuck_body_recovery_time_before_reset() {
    let root = test_ledger_root("sim-runner-stuck-recovery-time");
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(
        ledger.clone(),
        FixedConductor::new(ActionPrimitive::Go {
            intensity: 0.4,
            duration_ms: 1_000,
        }),
    );
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    let mut body = test_body(0.2, 0.2, 1.0, 7);
    body.odometry.heading_rad = std::f32::consts::PI;
    world.set_body(body);
    let mut runner = SimRunner::new(runtime, world, motors);

    runner
        .run_steps(STUCK_LOW_DISPLACEMENT_TICKS + 2)
        .await
        .unwrap();
    let snapshot = runner.world.snapshot().await.unwrap();
    let frames = ledger.recent(10).await.unwrap();

    assert_ne!(snapshot.body.odometry.x_m, 2.0);
    assert_ne!(snapshot.body.odometry.y_m, 2.0);
    assert!(!frames
        .iter()
        .any(|frame| frame.notes.iter().any(|note| note.contains("VirtualStuck"))));

    let _ = fs::remove_dir_all(root);
}

#[tokio::test]
async fn sim_with_danger_checkpoint_writes_shadow_predictions() {
    let root = test_ledger_root("sim-runner-danger-shadow");
    let checkpoint = danger_checkpoint_root("sim-runner-danger-shadow");
    let action = ActionPrimitive::Go {
        intensity: 0.4,
        duration_ms: 1_000,
    };
    write_test_danger_checkpoint(&checkpoint, action.clone());
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), FixedConductor::new(action))
        .with_models(RuntimeModelStack::with_danger_shadow_checkpoint(&checkpoint).unwrap());
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.8, 7));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let frames = ledger.recent(5).await.unwrap();
    let frame = frames.last().unwrap();

    assert!(frame.now.predictions.danger_model.is_some());
    assert!(frame.now.predictions.danger_hardcoded.is_some());

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[tokio::test]
async fn sim_attaches_fallback_predictions_to_embodied_experience() {
    let root = test_ledger_root("sim-runner-embodied-predictions");
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), FixedConductor::new(ActionPrimitive::Stop));
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.8, 7));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let frames = ledger.recent(5).await.unwrap();
    let experience = frames.last().unwrap().experiences.last().unwrap();

    assert!(frames.last().unwrap().z.is_some());
    assert!(experience
        .predictions
        .iter()
        .any(|prediction| prediction.text.starts_with("hazard:")));
    assert!(experience
        .predictions
        .iter()
        .any(|prediction| prediction.text.starts_with("uncertainty:")));

    let _ = fs::remove_dir_all(root);
}

#[tokio::test]
async fn danger_shadow_prediction_does_not_bypass_safety() {
    let root = test_ledger_root("sim-runner-danger-shadow-safety");
    let checkpoint = danger_checkpoint_root("sim-runner-danger-shadow-safety");
    let action = ActionPrimitive::Go {
        intensity: 0.5,
        duration_ms: 500,
    };
    write_test_danger_checkpoint(&checkpoint, action.clone());
    let ledger = JsonlLedger::new(&root);
    let mut runtime = test_runtime(ledger, FixedConductor::new(action.clone()))
        .with_models(RuntimeModelStack::with_danger_shadow_checkpoint(&checkpoint).unwrap());
    let mut body = BodySense::default();
    body.flags.cliff_left = true;
    body.last_update_ms = 100;

    let tick = runtime
        .tick(
            Now::blank(100, body),
            ExperienceLatent::default(),
            Vec::new(),
        )
        .await
        .unwrap();

    assert_eq!(tick.chosen_action, Some(action));
    assert!(tick.frame.now.predictions.danger_model.is_some());
    assert!(tick
        .frame
        .notes
        .iter()
        .any(|note| note.contains("Safety vetoed")));

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[tokio::test]
async fn sim_with_charge_checkpoint_writes_shadow_predictions() {
    let root = test_ledger_root("sim-runner-charge-shadow");
    let checkpoint = danger_checkpoint_root("sim-runner-charge-shadow");
    let action = ActionPrimitive::Dock;
    write_test_charge_checkpoint(&checkpoint, action.clone());
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), FixedConductor::new(action))
        .with_models(RuntimeModelStack::with_charge_shadow_checkpoint(&checkpoint).unwrap());
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.2, 7));
    world.add_object(SimObject::charger("charger", "charger", 1.2, 1.0, 0.18));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let frames = ledger.recent(5).await.unwrap();
    let frame = frames.last().unwrap();

    assert!(frame.now.predictions.charge_model.is_some());
    assert!(frame.now.predictions.charge_hardcoded.is_some());

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[tokio::test]
async fn sim_with_action_value_checkpoint_writes_shadow_predictions() {
    let root = test_ledger_root("sim-runner-action-value-shadow");
    let checkpoint = danger_checkpoint_root("sim-runner-action-value-shadow");
    let action = ActionPrimitive::Dock;
    write_test_action_value_checkpoint(&checkpoint, action.clone());
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), FixedConductor::new(action))
        .with_models(RuntimeModelStack::with_action_value_shadow_checkpoint(&checkpoint).unwrap());
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.2, 7));
    world.add_object(SimObject::charger("charger", "charger", 1.2, 1.0, 0.18));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let frames = ledger.recent(5).await.unwrap();
    let frame = frames.last().unwrap();

    assert!(!frame.now.predictions.action_values_model.is_empty());
    assert!(!frame.now.predictions.action_values_hardcoded.is_empty());

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[tokio::test]
async fn action_value_shadow_mode_does_not_override_conductor() {
    let root = test_ledger_root("sim-runner-action-value-shadow-choice");
    let checkpoint = danger_checkpoint_root("sim-runner-action-value-shadow-choice");
    write_test_action_value_checkpoint(&checkpoint, ActionPrimitive::Dock);
    let chosen = ActionPrimitive::Stop;
    let ledger = JsonlLedger::new(&root);
    let mut runtime = test_runtime(ledger, FixedConductor::new(chosen.clone()))
        .with_models(RuntimeModelStack::with_action_value_shadow_checkpoint(&checkpoint).unwrap());

    let tick = runtime
        .tick(
            Now::blank(100, test_body(1.0, 1.0, 0.8, 100)),
            ExperienceLatent::default(),
            Vec::new(),
        )
        .await
        .unwrap();

    assert_eq!(tick.chosen_action, Some(chosen));
    assert!(!tick.frame.now.predictions.action_values_model.is_empty());

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[tokio::test]
async fn sim_with_future_checkpoint_records_shadow_future_runs() {
    let root = test_ledger_root("sim-runner-future-shadow");
    let checkpoint = danger_checkpoint_root("sim-runner-future-shadow");
    write_test_future_checkpoint(&checkpoint, ActionPrimitive::Stop);
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), FixedConductor::new(ActionPrimitive::Stop))
        .with_models(RuntimeModelStack::with_future_shadow_checkpoint(&checkpoint).unwrap());
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.8, 7));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let frames = ledger.recent(5).await.unwrap();
    let frame = frames.last().unwrap();
    let run = frame
        .behavior_runs
        .iter()
        .find(|run| run.behavior_id == "future" && run.model_json.is_some())
        .unwrap();

    assert_eq!(run.regime, BehaviorRegime::ShadowInfer);
    assert!(run.hardcoded_json.is_some());
    assert!(run.selected_json.is_some());
    assert!(!frame.predicted_futures.is_empty());

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[tokio::test]
async fn inline_world_outcome_learning_observes_transition_sample() {
    let root = test_ledger_root("inline-world-outcome");
    let checkpoint = danger_checkpoint_root("inline-world-outcome");
    write_test_future_checkpoint(&checkpoint, ActionPrimitive::Stop);
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), FixedConductor::new(ActionPrimitive::Stop))
        .with_models(RuntimeModelStack::with_future_shadow_checkpoint(&checkpoint).unwrap())
        .with_inline_learning(InlineLearningConfig {
            mode: InlineLearningMode::WorldOutcome,
            behaviors: InlineLearningBehaviors {
                danger: false,
                charge: false,
                future: true,
                action_value: false,
                eye_next: false,
                ear_next: false,
                experience: false,
            },
            max_train_steps_per_tick: 1,
        });
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.8, 7));
    let mut runner = SimRunner::new(runtime, world, motors);
    let mut observed_samples = 0usize;

    runner
        .run_steps_observing_ticks(3, |_snapshot, tick| {
            observed_samples =
                observed_samples.saturating_add(tick.inline_learning.samples_observed);
        })
        .await
        .unwrap();

    assert!(observed_samples > 0);
    assert!(!read_transitions(&root).is_empty());

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[tokio::test]
async fn disabled_inline_learning_reports_no_weight_updates() {
    let root = test_ledger_root("inline-disabled");
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), FixedConductor::new(ActionPrimitive::Stop));
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.8, 7));
    let mut runner = SimRunner::new(runtime, world, motors);
    let mut statuses = Vec::new();

    runner
        .run_steps_observing_ticks(3, |_snapshot, tick| {
            statuses.push(tick.inline_learning.clone());
        })
        .await
        .unwrap();

    assert!(statuses.iter().all(|status| !status.enabled));
    assert!(statuses
        .iter()
        .all(|status| status.samples_observed == 0 && status.train_steps_used == 0));

    let _ = fs::remove_dir_all(root);
}

#[tokio::test]
async fn sim_with_ear_next_checkpoint_writes_shadow_prediction() {
    let root = test_ledger_root("sim-runner-ear-next-shadow");
    let checkpoint = danger_checkpoint_root("sim-runner-ear-next-shadow");
    let action = ActionPrimitive::Stop;
    write_test_ear_next_checkpoint(&checkpoint, action.clone());
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), FixedConductor::new(action))
        .with_models(RuntimeModelStack::with_ear_next_shadow_checkpoint(&checkpoint).unwrap());
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.8, 7));
    world.add_object(SimObject {
        id: "speaker".to_string(),
        label: "speaker".to_string(),
        kind: pete_sim::SimObjectKind::SoundSource {
            label: "speaker".to_string(),
        },
        x_m: 1.5,
        y_m: 1.2,
        radius_m: 0.12,
        color_rgb: [80, 80, 220],
        emits_sound: true,
        spoken_text: Some("listen to the room".to_string()),
        charge_rate: 0.0,
    });
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let frames = ledger.recent(5).await.unwrap();
    let frame = frames.last().unwrap();

    assert!(frame.now.predictions.ear_next_model.is_some());
    assert!(frame.now.predictions.ear_next_hardcoded.is_some());
    assert!(frame
        .behavior_runs
        .iter()
        .any(|run| run.behavior_id == "ear_next"));

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[tokio::test]
async fn ear_next_shadow_mode_does_not_override_safety_or_action() {
    let root = test_ledger_root("sim-runner-ear-next-shadow-safety");
    let checkpoint = danger_checkpoint_root("sim-runner-ear-next-shadow-safety");
    let action = ActionPrimitive::Go {
        intensity: 0.5,
        duration_ms: 500,
    };
    write_test_ear_next_checkpoint(&checkpoint, action.clone());
    let ledger = JsonlLedger::new(&root);
    let mut runtime = test_runtime(ledger, FixedConductor::new(action.clone()))
        .with_models(RuntimeModelStack::with_ear_next_shadow_checkpoint(&checkpoint).unwrap());
    let mut body = BodySense::default();
    body.flags.cliff_left = true;
    body.last_update_ms = 100;
    let mut now = Now::blank(100, body);
    now.ear.features = vec![vec![0.2, 0.4, 0.6, 0.8]];

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    assert_eq!(tick.chosen_action, Some(action));
    assert!(tick.frame.now.predictions.ear_next_model.is_some());
    assert!(tick
        .frame
        .notes
        .iter()
        .any(|note| note.contains("Safety vetoed")));

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[tokio::test]
async fn sim_with_experience_checkpoint_records_autoencoder_behavior_run() {
    let root = test_ledger_root("sim-runner-experience-shadow");
    let checkpoint = danger_checkpoint_root("sim-runner-experience-shadow");
    write_test_experience_checkpoint(&checkpoint);
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), FixedConductor::new(ActionPrimitive::Stop))
        .with_models(RuntimeModelStack::with_experience_shadow_checkpoint(&checkpoint).unwrap());
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    let mut body = test_body(1.0, 1.0, 0.8, 7);
    body.velocity.forward_m_s = 0.1;
    world.set_body(body);
    world.add_object(SimObject {
        id: "speaker".to_string(),
        label: "speaker".to_string(),
        kind: pete_sim::SimObjectKind::SoundSource {
            label: "speaker".to_string(),
        },
        x_m: 1.5,
        y_m: 1.2,
        radius_m: 0.12,
        color_rgb: [80, 80, 220],
        emits_sound: true,
        spoken_text: Some("the walls are awake".to_string()),
        charge_rate: 0.0,
    });
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let frames = ledger.recent(5).await.unwrap();
    let frame = frames.last().unwrap();
    let run = frame
        .behavior_runs
        .iter()
        .find(|run| run.behavior_id == "experience")
        .unwrap();

    assert_eq!(run.regime, BehaviorRegime::ShadowInfer);
    assert!(run.hardcoded_json.is_some());
    assert!(run.model_json.is_some());
    assert!(run.disagreement.unwrap_or_default().is_finite());
    assert!(frame.now.extensions.contains_key("experience.autoencoder"));

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[test]
fn missing_experience_checkpoint_returns_no_latent_yet() {
    let config: BehaviorRegistryConfig = toml::from_str(
        r#"
            [behavior.experience]
            regime = "shadow_infer"
            hardcoded = "experience.no_latent_yet"
            model = "experience.autoencoder.v0"
            checkpoint = "/tmp/pete-missing-experience-checkpoint"
            fallback = "use_hardcoded"
            "#,
    )
    .unwrap();
    let mut stack = RuntimeModelStack::from_behavior_config(&config).unwrap();
    let now = Now::blank(100, test_body(1.0, 1.0, 0.8, 100));
    let run = stack
        .behaviors
        .experience
        .infer(&ExperienceBehaviorInput::from_now(&now), now.t_ms)
        .unwrap();

    assert_eq!(run.record.regime, BehaviorRegime::ShadowInfer);
    assert!(run.record.hardcoded_output.is_some());
    assert!(run.record.model_output.is_none());
    assert_eq!(run.chosen, run.record.hardcoded_output.unwrap());
    assert!(run.chosen.latent.z.is_empty());
    assert_eq!(run.chosen.confidence, 0.0);
}

#[tokio::test]
async fn shared_reign_queue_controls_next_sim_tick() {
    let root = test_ledger_root("sim-runner-shared-reign");
    let ledger = JsonlLedger::new(&root);
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    queue.lock().unwrap().push(test_reign_input(
        7,
        ReignMode::Direct,
        ReignCommand::Turn {
            direction: pete_actions::TurnDir::Left,
            intensity: 0.5,
            duration_ms: 500,
        },
        2_000,
    ));
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let runtime = MinimalRuntime::with_reign_queue(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
        queue,
    );
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.8, 7));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let snapshot = runner.world.snapshot().await.unwrap();
    let frames = JsonlLedger::new(&root).recent(5).await.unwrap();
    let frame = frames.last().unwrap();

    assert!(snapshot.body.odometry.heading_rad > 0.0);
    assert!(frame.now.reign.active);
    assert!(frame
        .sensations
        .iter()
        .any(|sensation| sensation.kind == "reign.command"));
    assert!(frame.reign_input.is_some());
    assert!(frame
        .reign_outcome
        .as_ref()
        .map(|outcome| outcome.accepted_by_conductor)
        .unwrap_or(false));
}

#[tokio::test]
async fn direct_reign_reverse_drives_sim_while_stuck_active() {
    let root = test_ledger_root("sim-runner-reign-reverse-interrupts-stuck");
    let ledger = JsonlLedger::new(&root);
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    queue.lock().unwrap().push(test_reign_input(
        7,
        ReignMode::Direct,
        ReignCommand::Reverse {
            intensity: 0.5,
            duration_ms: 500,
        },
        2_000,
    ));
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let runtime = MinimalRuntime::with_reign_queue(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
        queue,
    );
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 1.0, 7));
    let mut runner = SimRunner::new(runtime, world, motors);
    runner.stuck.active = true;
    runner.stuck.phase = RecoveryPhase::Stop;
    runner.stuck.phase_ticks_remaining = 1;
    runner.stuck.turn_sign = 1.0;

    let mut observed_debug = None;
    runner
        .run_steps_observing(1, |snapshot| {
            observed_debug = snapshot.action_debug.clone();
        })
        .await
        .unwrap();
    let debug = observed_debug.unwrap();
    let motion = debug.get("motion_sent_to_sim").cloned().unwrap();

    let motion = serde_json::from_value::<MotionCommand>(motion.clone())
        .unwrap_or_else(|error| panic!("motion decode failed: {error}; debug={debug}"));
    assert_eq!(motion, MotionCommand::Forward { speed_m_s: -0.5 });
}

#[tokio::test]
async fn column_trap_scenario_recovers_within_budget() {
    let root = test_ledger_root("sim-runner-column-trap-recovery");
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger, SimpleConductor::default());
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ColumnTrap, 7));
    let start = (
        scenario.metadata.body.odometry.x_m,
        scenario.metadata.body.odometry.y_m,
    );
    let mut runner = SimRunner::new(runtime, scenario.world, scenario.motors);
    let mut saw_column = false;
    let mut recovered = false;
    let mut last_skill_status = None;

    runner
        .run_steps_observing_ticks(90, |snapshot, tick| {
            if tick.skill_status.is_some() {
                last_skill_status = tick.skill_status.clone();
            }
            if let Some(stuck) = snapshot
                .extensions
                .iter()
                .find(|extension| extension.name == "sim.stuck")
            {
                saw_column |= stuck.values.get(10).copied() == Some(3.0);
                recovered |= stuck.values.get(7).copied() == Some(1.0);
            }
        })
        .await
        .unwrap();
    let end = runner.world.body();
    let distance = distance_between_points(start, (end.odometry.x_m, end.odometry.y_m));

    assert!(saw_column);
    assert!(recovered, "last Lua skill status was {last_skill_status:?}");
    assert!(distance > 0.10, "distance after recovery was {distance}");
}

#[derive(Clone, Copy, Debug, Default)]
struct TrapRunMetrics {
    collision_frames: usize,
    stuck_frames: usize,
    recovered: bool,
    distance_m: f32,
}

async fn run_column_trap_metrics<C>(ledger_name: &str, conductor: C, steps: usize) -> TrapRunMetrics
where
    C: Conductor + Send + 'static,
{
    let root = test_ledger_root(ledger_name);
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger, conductor);
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ColumnTrap, 7));
    let start = (
        scenario.metadata.body.odometry.x_m,
        scenario.metadata.body.odometry.y_m,
    );
    let mut runner = SimRunner::new(runtime, scenario.world, scenario.motors);
    let mut metrics = TrapRunMetrics::default();

    runner
        .run_steps_observing(steps, |snapshot| {
            let flags = &snapshot.body.flags;
            if flags.wall
                || flags.bump_left
                || flags.bump_right
                || flags.cliff_front_left
                || flags.cliff_front_right
            {
                metrics.collision_frames += 1;
            }
            if let Some(stuck) = snapshot
                .extensions
                .iter()
                .find(|extension| extension.name == "sim.stuck")
            {
                metrics.stuck_frames +=
                    (stuck.values.first().copied().unwrap_or_default() > 0.0) as usize;
                metrics.recovered |= stuck.values.get(7).copied() == Some(1.0);
            }
        })
        .await
        .unwrap();
    let end = runner.world.body();
    metrics.distance_m = distance_between_points(start, (end.odometry.x_m, end.odometry.y_m));
    metrics
}

#[tokio::test]
async fn column_trap_recovery_beats_plain_explore_baseline() {
    let plain = run_column_trap_metrics(
        "sim-runner-column-trap-plain-explore",
        FixedConductor::new(ActionPrimitive::Explore {
            style: ExploreStyle::RandomWalk,
            duration_ms: 1_000,
        }),
        120,
    )
    .await;
    let recovered = run_column_trap_metrics(
        "sim-runner-column-trap-simple-recovery-comparison",
        SimpleConductor::default(),
        120,
    )
    .await;

    assert!(
        recovered.recovered,
        "expected recovery event, got {recovered:?}"
    );
    assert!(
        recovered.collision_frames < plain.collision_frames / 2,
        "recovery should reduce repeated collision frames; plain={plain:?} recovered={recovered:?}"
    );
    assert!(
            recovered.distance_m > plain.distance_m,
            "recovery should make more progress than plain explore; plain={plain:?} recovered={recovered:?}"
        );
    assert!(
        recovered.stuck_frames < plain.stuck_frames,
        "recovery should reduce repeated stuck frames; plain={plain:?} recovered={recovered:?}"
    );
}

#[derive(Clone, Debug)]
struct FixedConductor {
    action: ActionPrimitive,
}

#[derive(Clone, Debug, Default)]
struct FixedRecall {
    bundle: RecallBundle,
}

#[async_trait::async_trait]
impl Recall for FixedRecall {
    async fn recall(&self, _query: RecallQuery) -> Result<RecallBundle> {
        Ok(self.bundle.clone())
    }
}

impl FixedConductor {
    fn new(action: ActionPrimitive) -> Self {
        Self { action }
    }
}

impl Conductor for FixedConductor {
    fn choose(&mut self, _input: ConductorInput) -> Result<ActionPrimitive> {
        Ok(self.action.clone())
    }
}

#[derive(Clone, Debug)]
struct FixedLlmAgent {
    action: ActionPrimitive,
}

#[async_trait::async_trait]
impl LlmAgent for FixedLlmAgent {
    async fn combobulate(
        &mut self,
        _now: &Now,
        _impressions: &[Impression],
        _embodied: Option<&EmbodiedContext>,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
    ) -> Result<Option<Combobulation>> {
        Ok(None)
    }

    async fn maybe_tick(
        &mut self,
        _now: &Now,
        _embodied: Option<&EmbodiedContext>,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
        _awareness_summary: Option<&str>,
    ) -> Result<LlmTickResult> {
        Ok(LlmTickResult {
            sense: pete_now::LlmSense {
                schema_version: 1,
                command_summary: Some("test command".to_string()),
                critique: None,
                confidence: 1.0,
            },
            conscious_command: Some(ConsciousCommand {
                summary: "test command".to_string(),
                action: Some(self.action.clone()),
            }),
            decision: Some(LlmDecision {
                summary: "test command".to_string(),
                action: Some(self.action.clone()),
                confidence: 1.0,
                ..LlmDecision::default()
            }),
            teaching: Vec::new(),
        })
    }

    async fn scientific_review(
        &mut self,
        _request: &LlmReviewRequest,
    ) -> Result<Option<LlmScientificReview>> {
        Ok(None)
    }
}

#[derive(Debug, Default)]
struct SlowAdvisoryAgent;

#[async_trait::async_trait]
impl LlmAgent for SlowAdvisoryAgent {
    async fn combobulate(
        &mut self,
        _now: &Now,
        _impressions: &[Impression],
        _embodied: Option<&EmbodiedContext>,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
    ) -> Result<Option<Combobulation>> {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        Ok(Some(Combobulation {
            summary: "historical doorway hypothesis".to_string(),
            confidence: 0.8,
        }))
    }

    async fn maybe_tick(
        &mut self,
        _now: &Now,
        _embodied: Option<&EmbodiedContext>,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
        _awareness_summary: Option<&str>,
    ) -> Result<LlmTickResult> {
        Ok(LlmTickResult {
            sense: pete_now::LlmSense {
                schema_version: 1,
                critique: Some("test the doorway hypothesis".to_string()),
                confidence: 0.8,
                ..pete_now::LlmSense::default()
            },
            decision: Some(LlmDecision {
                action: Some(ActionPrimitive::Go {
                    intensity: 1.0,
                    duration_ms: 5_000,
                }),
                ..LlmDecision::default()
            }),
            ..LlmTickResult::default()
        })
    }

    async fn scientific_review(
        &mut self,
        _request: &LlmReviewRequest,
    ) -> Result<Option<LlmScientificReview>> {
        Ok(None)
    }
}

fn test_runtime<C>(
    ledger: JsonlLedger,
    conductor: C,
) -> MinimalRuntime<
    JsonlLedger,
    InMemoryExperienceStore,
    InMemoryExperienceStore,
    C,
    SimpleSafety,
    pete_llm::NoopLlmAgent,
>
where
    C: Conductor,
{
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    MinimalRuntime::new(
        ledger,
        memory,
        recall,
        conductor,
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    )
}

async fn finished_cognition_task() -> JoinHandle<Result<(Option<Combobulation>, LlmTickResult)>> {
    let task = tokio::spawn(async {
        Ok((
            None,
            LlmTickResult {
                sense: pete_now::LlmSense {
                    schema_version: 1,
                    command_summary: Some("completed cognition".to_string()),
                    confidence: 1.0,
                    ..pete_now::LlmSense::default()
                },
                ..LlmTickResult::default()
            },
        ))
    });
    tokio::task::yield_now().await;
    assert!(task.is_finished(), "fixture cognition task should be ready");
    task
}

fn cognition_test_inputs() -> (
    EmbodiedContext,
    ExperienceLatent,
    Vec<FuturePrediction>,
    Vec<String>,
) {
    (
        EmbodiedContext::default(),
        ExperienceLatent::default(),
        Vec::new(),
        Vec::new(),
    )
}

#[tokio::test]
async fn llm_command_is_never_granted_control_authority() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-llm-command-action-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let llm_action = ActionPrimitive::Go {
        intensity: 0.3,
        duration_ms: 700,
    };
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        FixedConductor::new(ActionPrimitive::Stop),
        SimpleSafety::default(),
        FixedLlmAgent {
            action: llm_action.clone(),
        },
    );
    let mut now = idle_now(100);
    now.drives.curiosity = 1.0;

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
    assert!(tick.frame.conscious_command.is_none());
    let decision = tick
        .frame
        .now
        .extensions
        .get("action_selector")
        .cloned()
        .and_then(|value| serde_json::from_value::<ActionSelectionDecision>(value).ok())
        .unwrap();
    assert_eq!(decision.selected_action, Some(ActionPrimitive::Stop));
}

#[tokio::test]
async fn accepted_cognition_enters_cooldown_before_scheduling_again() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-cognition-service-health-test");
    let memory = InMemoryExperienceStore::new();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory.clone(),
        memory,
        FixedConductor::new(ActionPrimitive::Stop),
        SimpleSafety::default(),
        FixedLlmAgent {
            action: ActionPrimitive::Stop,
        },
    );

    let first = runtime
        .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    let first_service = &first.frame.now.world.self_model.service_state.services["rich_language"];
    assert!(first_service.available);
    assert!(first_service.busy);
    assert_eq!(first_service.unavailable_reason, None);

    tokio::task::yield_now().await;
    let accepted = runtime
        .tick(idle_now(200), ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    assert!(matches!(
        runtime.cognition.last_outcome.as_ref(),
        Some(CognitionOutcome::Accepted)
    ));
    assert!(runtime.cognition.pending.is_none());
    assert_eq!(runtime.cognition.next_request_at_ms, 2_200);
    let accepted_service =
        &accepted.frame.now.world.self_model.service_state.services["rich_language"];
    assert!(accepted_service.available);
    assert!(!accepted_service.busy);
    assert_eq!(accepted_service.unavailable_reason, None);
    assert!(accepted
        .frame
        .notes
        .iter()
        .any(|note| note == "LlmProviderOutcome: accepted"));

    let cooling_down = runtime
        .tick(idle_now(2_199), ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    assert!(runtime.cognition.pending.is_none());
    assert!(
        !cooling_down
            .frame
            .now
            .world
            .self_model
            .service_state
            .services["rich_language"]
            .busy
    );

    let eligible = runtime
        .tick(idle_now(2_200), ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    assert!(runtime.cognition.pending.is_some());
    assert!(eligible.frame.now.world.self_model.service_state.services["rich_language"].busy);
}

#[tokio::test]
async fn disabled_cognition_is_unavailable_without_scheduling_provider_work() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-disabled-cognition-service-test");
    let memory = InMemoryExperienceStore::new();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory.clone(),
        memory,
        FixedConductor::new(ActionPrimitive::Stop),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    );

    let tick = runtime
        .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    let service = &tick.frame.now.world.self_model.service_state.services["rich_language"];

    assert!(!service.available);
    assert!(!service.busy);
    assert_eq!(
        service.unavailable_reason.as_deref(),
        Some("enhanced language service is disabled")
    );
    assert!(runtime.cognition.pending.is_none());
    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
}

#[tokio::test]
async fn paused_runtime_clock_does_not_expire_completed_cognition() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-cognition-paused-clock-test");
    let mut runtime = test_runtime(ledger, FixedConductor::new(ActionPrimitive::Stop));
    runtime.cognition.pending = Some(PendingLlmCognition {
        snapshot_ref: "paused-frame".to_string(),
        requested_at_ms: 1_000,
        deadline_ms: 1_000 + COGNITION_DEADLINE_MS,
        task: finished_cognition_task().await,
    });
    let now = idle_now(1_000);
    let (embodied, latent, futures, mut notes) = cognition_test_inputs();

    let accepted = runtime
        .advance_cognition(&now, &[], &embodied, &latent, &futures, "", &mut notes)
        .await
        .expect("paused deterministic time should accept a completed provider result");

    assert_eq!(accepted.requested_at_ms, 1_000);
    assert_eq!(accepted.observed_at_ms, 1_000);
    assert!(matches!(
        runtime.cognition.last_outcome.as_ref(),
        Some(CognitionOutcome::Accepted)
    ));
}

#[tokio::test]
async fn replayed_earlier_now_does_not_expire_completed_cognition() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-cognition-replay-clock-test");
    let mut runtime = test_runtime(ledger, FixedConductor::new(ActionPrimitive::Stop));
    runtime.cognition.pending = Some(PendingLlmCognition {
        snapshot_ref: "future-replay-frame".to_string(),
        requested_at_ms: 5_000,
        deadline_ms: 5_000 + COGNITION_DEADLINE_MS,
        task: finished_cognition_task().await,
    });
    let replayed_now = idle_now(500);
    let (embodied, latent, futures, mut notes) = cognition_test_inputs();

    let accepted = runtime
        .advance_cognition(
            &replayed_now,
            &[],
            &embodied,
            &latent,
            &futures,
            "",
            &mut notes,
        )
        .await
        .expect("a backwards replay clock should not invent elapsed runtime time");

    assert_eq!(accepted.requested_at_ms, 5_000);
    assert_eq!(accepted.observed_at_ms, 500);
    assert!(matches!(
        runtime.cognition.last_outcome.as_ref(),
        Some(CognitionOutcome::Accepted)
    ));
}

#[tokio::test]
async fn forward_clock_jump_expires_in_flight_cognition() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-cognition-forward-jump-test");
    let mut runtime = test_runtime(ledger, FixedConductor::new(ActionPrimitive::Stop));
    let (completion_tx, completion_rx) = tokio::sync::oneshot::channel::<()>();
    runtime.cognition.pending = Some(PendingLlmCognition {
        snapshot_ref: "pre-jump-frame".to_string(),
        requested_at_ms: 1_000,
        deadline_ms: 1_000 + COGNITION_DEADLINE_MS,
        task: tokio::spawn(async move {
            completion_rx.await.expect("completion sender retained");
            Ok((None, LlmTickResult::default()))
        }),
    });
    let jumped_now = idle_now(10_000);
    let (embodied, latent, futures, mut notes) = cognition_test_inputs();

    let accepted = runtime
        .advance_cognition(
            &jumped_now,
            &[],
            &embodied,
            &latent,
            &futures,
            "",
            &mut notes,
        )
        .await;

    assert!(accepted.is_none());
    assert!(matches!(
        runtime.cognition.last_outcome.as_ref(),
        Some(CognitionOutcome::Expired)
    ));
    tokio::task::yield_now().await;
    assert!(
        completion_tx.send(()).is_err(),
        "expired task should be aborted"
    );
}

#[tokio::test]
async fn very_slow_cognition_tick_rejects_result_completed_before_late_poll() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-cognition-slow-tick-test");
    let mut runtime = test_runtime(ledger, FixedConductor::new(ActionPrimitive::Stop));
    runtime.cognition.pending = Some(PendingLlmCognition {
        snapshot_ref: "slow-tick-frame".to_string(),
        requested_at_ms: 1_000,
        deadline_ms: 1_000 + COGNITION_DEADLINE_MS,
        task: finished_cognition_task().await,
    });
    let late_now = idle_now(1_000 + COGNITION_DEADLINE_MS + 1);
    let (embodied, latent, futures, mut notes) = cognition_test_inputs();

    let accepted = runtime
        .advance_cognition(&late_now, &[], &embodied, &latent, &futures, "", &mut notes)
        .await;

    assert!(accepted.is_none());
    assert!(matches!(
        runtime.cognition.last_outcome.as_ref(),
        Some(CognitionOutcome::Expired)
    ));
}

#[tokio::test]
async fn cognition_provider_completion_exactly_at_deadline_is_accepted() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-cognition-deadline-boundary-test");
    let mut runtime = test_runtime(ledger, FixedConductor::new(ActionPrimitive::Stop));
    let requested_at_ms = 1_000;
    let deadline_ms = requested_at_ms + COGNITION_DEADLINE_MS;
    runtime.cognition.pending = Some(PendingLlmCognition {
        snapshot_ref: "deadline-frame".to_string(),
        requested_at_ms,
        deadline_ms,
        task: finished_cognition_task().await,
    });
    let deadline_now = idle_now(deadline_ms);
    let (embodied, latent, futures, mut notes) = cognition_test_inputs();

    let accepted = runtime
        .advance_cognition(
            &deadline_now,
            &[],
            &embodied,
            &latent,
            &futures,
            "",
            &mut notes,
        )
        .await
        .expect("the deadline is inclusive");

    assert_eq!(accepted.observed_at_ms, deadline_ms);
    assert!(matches!(
        runtime.cognition.last_outcome.as_ref(),
        Some(CognitionOutcome::Accepted)
    ));
    assert_eq!(
        runtime.cognition.last_sense.command_summary.as_deref(),
        Some("completed cognition")
    );
}

#[tokio::test]
async fn slow_advice_is_retained_as_historical_evidence_only() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-slow-advice-test");
    let memory = InMemoryExperienceStore::new();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory.clone(),
        memory,
        FixedConductor::new(ActionPrimitive::Stop),
        SimpleSafety::default(),
        SlowAdvisoryAgent,
    );
    let mut accepted_tick = None;

    for step in 0..8 {
        let t_ms = 100 + step * 100;
        let mut now = idle_now(t_ms);
        now.body.last_update_ms = t_ms;
        let tick = runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();
        assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
        if tick
            .frame
            .experiences
            .iter()
            .any(|experience| experience.kind == "llm.combobulation")
        {
            accepted_tick = Some(tick);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    let tick = accepted_tick.expect("500 ms advice should be retained on a later tick");
    let evidence = tick
        .frame
        .sensations
        .iter()
        .find(|sensation| sensation.kind == "llm.combobulation")
        .expect("provenance-bearing advisory sensation");
    assert_eq!(evidence.occurred_at_ms, 100);
    assert!(evidence.observed_at_ms >= 600);
    assert!(evidence
        .payload
        .get("input_snapshot_ref")
        .and_then(serde_json::Value::as_str)
        .is_some());
    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
    assert!(tick.frame.conscious_command.is_none());
}

#[tokio::test]
async fn active_safe_reign_wins_over_llm_action() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-llm-reign-wins-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    let reign_command = ReignCommand::Turn {
        direction: TurnDir::Left,
        intensity: 0.4,
        duration_ms: 500,
    };
    queue.lock().unwrap().push(test_reign_input(
        100,
        ReignMode::Direct,
        reign_command.clone(),
        1_000,
    ));
    let mut runtime = MinimalRuntime::with_reign_queue(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        FixedLlmAgent {
            action: ActionPrimitive::Explore {
                style: ExploreStyle::RandomWalk,
                duration_ms: 1_000,
            },
        },
        queue,
    );
    let now = idle_now(100);

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    let proposal = tick
        .frame
        .now
        .extensions
        .get("llm.action_proposal")
        .cloned()
        .and_then(|value| serde_json::from_value::<LlmActionProposal>(value).ok())
        .unwrap();

    assert_eq!(tick.chosen_action, reign_command.to_action());
    assert!(!proposal.accepted);
    assert_eq!(proposal.ignored_reason.as_deref(), None);
}

#[tokio::test]
async fn llm_action_is_discarded_before_safety_and_cockpit() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-llm-safety-veto-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        FixedConductor::new(ActionPrimitive::Stop),
        SimpleSafety::default(),
        FixedLlmAgent {
            action: ActionPrimitive::Go {
                intensity: 0.3,
                duration_ms: 700,
            },
        },
    );
    runtime
        .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    let provider_input_ref = runtime
        .cognition
        .pending
        .as_ref()
        .expect("provider request in flight")
        .snapshot_ref
        .clone();
    tokio::task::yield_now().await;

    let mut now = idle_now(200);
    now.body.flags.cliff_left = true;

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    let proposal = tick
        .frame
        .now
        .extensions
        .get("llm.action_proposal")
        .cloned()
        .and_then(|value| serde_json::from_value::<LlmActionProposal>(value).ok())
        .unwrap();

    assert!(proposal.proposed_action.is_none());
    assert_eq!(
        proposal.advisory_action,
        Some(LlmAdvisoryAction {
            action: ActionPrimitive::Go {
                intensity: 0.3,
                duration_ms: 700,
            },
            source: LlmAdvisoryActionSource::ProviderDecision,
            input_snapshot_ref: provider_input_ref,
            disposition: LlmAdvisoryActionDisposition::DiscardedAtAdvisoryBoundary,
        })
    );
    assert!(!proposal.accepted);
    assert!(!proposal.safety_vetoed);
    assert_eq!(
            proposal.ignored_reason.as_deref(),
            Some(
                "provider suggested Go { intensity: 0.3, duration_ms: 700 }; discarded at advisory boundary"
            )
        );
    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
    let bridge = tick
        .frame
        .now
        .extensions
        .get("action.motion_bridge")
        .expect("motion bridge telemetry");
    assert!(bridge["llm_action"].is_null());
    assert_eq!(
        bridge["llm_advisory_action"]["disposition"],
        "discarded_at_advisory_boundary"
    );
    assert!(tick.frame.notes.iter().any(|note| {
            note.contains(
                "LlmAdvisoryAction: provider suggested Go { intensity: 0.3, duration_ms: 700 }; discarded at advisory boundary",
            )
        }));
}

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
