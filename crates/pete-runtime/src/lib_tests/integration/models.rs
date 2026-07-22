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
                let outcome = tick
                    .brain_events
                    .iter()
                    .find(|event| event.kind == "actuator.dispatch_outcome")
                    .expect("simulator records dispatch only after applying the command");
                assert_eq!(outcome.producer.brain, Brain::Simulator);
                assert_eq!(outcome.disposition, EventDisposition::Accepted);
                assert!(tick.brain_events.iter().any(|event| {
                    event.event_id == outcome.links.parents[0].event_id
                        && event.event_type == BrainEventType::Command
                }));
                assert_eq!(
                    outcome.payload,
                    BrainEventPayload::inline(snapshot.action_debug.clone().unwrap())
                );
                let response = tick
                    .brain_events
                    .iter()
                    .find(|event| event.kind == "motion.response")
                    .expect("simulator records measured motion separately from dispatch");
                assert_eq!(response.producer.brain, Brain::Simulator);
                assert_eq!(response.disposition, EventDisposition::Accepted);
                assert_eq!(
                    response.links.parents[0].event_id,
                    outcome.event_id,
                    "measured response must correlate to dispatch"
                );
                assert_unique_locally_ordered_brain_events(&tick.brain_events);
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
