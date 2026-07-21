#[tokio::test]
async fn goal_shadow_records_evaluation_without_replacing_baseline() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-goal-shadow-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        FixedConductor::new(ActionPrimitive::Stop),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    )
    .with_action_selector_mode(ActionSelectorMode::GoalShadow);
    let tick = runtime
        .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
    let decision = serde_json::from_value::<ActionSelectionDecision>(
        tick.frame.now.extensions["action_selector"].clone(),
    )
    .unwrap();
    assert_eq!(decision.mode, ActionSelectorMode::GoalShadow);
    assert!(decision.selected_goal.is_none());
    assert!(decision.shadow_selected_goal.is_some());
    assert!(tick.frame.now.extensions.contains_key("goal_system"));
}

#[tokio::test]
async fn goal_mode_executes_goal_behavior_and_publishes_homeostatic_drives() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-goal-mode-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        FixedConductor::new(ActionPrimitive::Stop),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    )
    .with_action_selector_mode(ActionSelectorMode::Goal);
    let mut now = idle_now(100);
    now.body.battery_level = 0.05;
    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    let decision = serde_json::from_value::<ActionSelectionDecision>(
        tick.frame.now.extensions["action_selector"].clone(),
    )
    .unwrap();
    assert_eq!(decision.selected_goal.as_deref(), Some("seek_charger"));
    assert!(matches!(
        decision.selected_behavior.as_deref(),
        Some("inspect_for_charger" | "systematic_charger_search")
    ));
    assert_ne!(tick.chosen_action, Some(ActionPrimitive::Dock));
    assert!(tick.frame.now.drives.battery_hunger > 0.5);
}

#[tokio::test]
async fn sleep_quiesces_possessor_goals_and_emits_a_durable_snapshot() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-sleep-quiescence-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        FixedConductor::new(ActionPrimitive::Explore {
            style: ExploreStyle::Wander,
            duration_ms: 1_000,
        }),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    )
    .with_action_selector_mode(ActionSelectorMode::Goal);
    let mut now = idle_now(100);
    now.body.charging = true;
    now.extensions
        .insert("sleep.request".to_string(), serde_json::Value::Bool(true));

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
    assert!(tick.skill_request.is_none());
    let sleep: SleepSnapshot =
        serde_json::from_value(tick.frame.now.extensions["sleep"].clone()).unwrap();
    assert_eq!(sleep.phase, SleepPhase::Preparing);
    let goals: pete_conductor::GoalCycle =
        serde_json::from_value(tick.frame.now.extensions["goal_system"].clone()).unwrap();
    assert!(goals.selection.selected_goal.is_none());
    assert_eq!(
        goals.selection.reason,
        "deliberative goals quiesced for sleep"
    );
}

#[test]
fn executed_goal_behavior_strengthens_approach_progress_from_canonical_target() {
    std::thread::Builder::new()
        .name("semantic-outcome-test".to_string())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async {
                    let ledger = JsonlLedger::new("/tmp/pete-runtime-semantic-outcome-test");
                    let memory = InMemoryExperienceStore::new();
                    let recall = memory.clone();
                    let mut runtime = Box::new(
                        MinimalRuntime::new(
                            ledger,
                            memory,
                            recall,
                            FixedConductor::new(ActionPrimitive::Stop),
                            SimpleSafety::default(),
                            pete_llm::NoopLlmAgent,
                        )
                        .with_action_selector_mode(ActionSelectorMode::Goal),
                    );
                    let charger_now = |t_ms: u64, distance_m: f32| {
                        let mut now = idle_now(t_ms);
                        now.body.battery_level = 0.12;
                        now.objects.observations.push(pete_now::ObjectObservation {
                            label: "dock".to_string(),
                            class: ObjectClass::Charger,
                            bearing_rad: 0.0,
                            distance_m: Some(distance_m),
                            confidence: 0.95,
                            source: pete_now::ObjectObservationSource::Sim,
                        });
                        now
                    };
                    let first = runtime
                        .tick(
                            charger_now(100, 1.0),
                            ExperienceLatent::default(),
                            Vec::new(),
                        )
                        .await
                        .unwrap();
                    assert_eq!(
                        first.frame.now.extensions["action_selector"]["selected_behavior"],
                        serde_json::Value::String("approach_charger".to_string())
                    );
                    drop(first);
                    let second = runtime
                        .tick(
                            charger_now(200, 0.7),
                            ExperienceLatent::default(),
                            Vec::new(),
                        )
                        .await
                        .unwrap();
                    assert_eq!(
                        second.frame.now.extensions["goal_system.outcome"]
                            ["executed_goal_behavior"],
                        serde_json::Value::String("approach_charger".to_string())
                    );
                    drop(second);
                    let third = runtime
                        .tick(
                            charger_now(300, 0.7),
                            ExperienceLatent::default(),
                            Vec::new(),
                        )
                        .await
                        .unwrap();
                    assert!(third
                        .frame
                        .now
                        .world
                        .semantic
                        .relations
                        .values()
                        .any(|relation| {
                            relation.subject
                                == SemanticNodeRef::Behavior(SemanticBehaviorId(
                                    "approach_charger".to_string(),
                                ))
                                && relation.predicate == SemanticPredicate::Predicts
                                && relation
                                    .supporting_evidence
                                    .iter()
                                    .any(|evidence| evidence.source == "runtime.action_outcome")
                        }));
                });
        })
        .unwrap()
        .join()
        .unwrap();
}

#[tokio::test]
async fn shadow_goal_behavior_cannot_claim_the_executed_baseline_outcome() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-semantic-shadow-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        FixedConductor::new(ActionPrimitive::Stop),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    )
    .with_action_selector_mode(ActionSelectorMode::GoalShadow);
    let charger_now = |t_ms: u64, distance_m: f32| {
        let mut now = idle_now(t_ms);
        now.body.battery_level = 0.12;
        now.objects.observations.push(pete_now::ObjectObservation {
            label: "dock".to_string(),
            class: ObjectClass::Charger,
            bearing_rad: 0.0,
            distance_m: Some(distance_m),
            confidence: 0.95,
            source: pete_now::ObjectObservationSource::Sim,
        });
        now
    };
    for (t_ms, distance_m) in [(100, 1.0), (200, 0.7), (300, 0.7)] {
        let tick = runtime
            .tick(
                charger_now(t_ms, distance_m),
                ExperienceLatent::default(),
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
        assert!(
            tick.frame.now.extensions["goal_system.outcome"]["executed_goal_behavior"].is_null()
        );
        assert!(!tick
            .frame
            .now
            .world
            .semantic
            .relations
            .values()
            .any(|relation| relation
                .supporting_evidence
                .iter()
                .any(|evidence| evidence.source == "runtime.action_outcome")));
    }
}

#[test]
fn semantic_outcomes_retain_canonical_target_identity() {
    let world_with_distances = |t_ms: u64, first: f32, second: f32| {
        let mut world = WorldModelSnapshot {
            t_ms,
            ..WorldModelSnapshot::default()
        };
        for (id, distance_m) in [("charger:a", first), ("charger:b", second)] {
            world.entities.insert(
                EntityId(id.to_string()),
                pete_now::WorldEntity {
                    id: EntityId(id.to_string()),
                    kind: pete_now::WorldEntityKind::Charger,
                    distance_m: Some(distance_m),
                    distance_meta: Some(BeliefMeta {
                        confidence: 1.0,
                        observed_at_ms: t_ms,
                        valid_at_ms: t_ms,
                        freshness: Freshness::Current,
                        ..BeliefMeta::default()
                    }),
                    ..pete_now::WorldEntity::default()
                },
            );
        }
        world
    };
    let behavior = pete_conductor::BehaviorDecision {
        goal_id: pete_conductor::GoalId::new("seek_charger"),
        behavior_id: "approach_charger".to_string(),
        action: ActionPrimitive::Approach {
            target: ApproachTarget::Charger,
        },
        affordance: pete_conductor::Affordance {
            target: Some(EntityId("charger:a".to_string())),
            ..pete_conductor::Affordance::default()
        },
    };
    let mut tracker = SemanticOutcomeTracker::default();
    tracker.remember(&world_with_distances(100, 1.0, 2.0), Some(&behavior));

    // Charger B becoming closer is not evidence that the action advanced A.
    tracker.observe_outcome(&world_with_distances(200, 1.0, 0.5));
    assert!(tracker.take_pending().is_empty());

    tracker.observe_outcome(&world_with_distances(300, 0.7, 0.4));
    let evidence = tracker.take_pending();
    assert_eq!(evidence.len(), 1);
    assert_eq!(
        evidence[0].subject,
        SemanticNodeRef::Behavior(SemanticBehaviorId("approach_charger".to_string()))
    );
}

#[tokio::test]
async fn goal_mode_assist_is_only_an_affordance_bias_but_direct_still_overrides() {
    let build_runtime = |path: &'static str| {
        let ledger = JsonlLedger::new(path);
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        MinimalRuntime::new(
            ledger,
            memory,
            recall,
            FixedConductor::new(ActionPrimitive::Stop),
            SimpleSafety::default(),
            pete_llm::NoopLlmAgent,
        )
        .with_action_selector_mode(ActionSelectorMode::Goal)
    };
    let mut assisted = build_runtime("/tmp/pete-runtime-goal-assist-test");
    assisted.reign_queue.lock().unwrap().push(test_reign_input(
        100,
        ReignMode::Assist,
        ReignCommand::Dock,
        2_000,
    ));
    let assisted_tick = assisted
        .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    assert_ne!(assisted_tick.chosen_action, Some(ActionPrimitive::Dock));

    let mut direct = build_runtime("/tmp/pete-runtime-goal-direct-test");
    direct.reign_queue.lock().unwrap().push(test_reign_input(
        100,
        ReignMode::Direct,
        ReignCommand::Dock,
        2_000,
    ));
    let direct_tick = direct
        .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    assert_eq!(direct_tick.chosen_action, Some(ActionPrimitive::Dock));
}

#[test]
fn memory_backed_baseline_action_is_a_selector_candidate_context() {
    let mut now = idle_now(100);
    mark_corrected_map_trusted(&mut now);
    now.memory.place_danger = 0.9;
    now.memory.nearby_best_safe_direction_rad = Some(-0.8);
    let memory_action = ActionPrimitive::Turn {
        direction: TurnDir::Right,
        intensity: 0.5,
        duration_ms: 1_000,
    };
    let default_action = ActionPrimitive::Go {
        intensity: 0.15,
        duration_ms: 1_000,
    };

    assert!(memory_navigation_candidate_context(&now, &memory_action));
    assert!(!memory_navigation_candidate_context(&now, &default_action));
}

#[tokio::test]
async fn direct_reign_overrides_model_assisted_selector() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-reign-model-assisted-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    )
    .with_action_selector_mode(ActionSelectorMode::ModelAssisted);
    let command = ReignCommand::Turn {
        direction: TurnDir::Right,
        intensity: 0.5,
        duration_ms: 500,
    };
    runtime.reign_queue.lock().unwrap().push(test_reign_input(
        100,
        ReignMode::Direct,
        command.clone(),
        2_000,
    ));
    let mut now = idle_now(100);
    now.drives.curiosity = 1.0;

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    assert_eq!(tick.chosen_action, command.to_action());
    let decision = tick
        .frame
        .now
        .extensions
        .get("action_selector")
        .cloned()
        .and_then(|value| serde_json::from_value::<ActionSelectionDecision>(value).ok())
        .unwrap();
    assert_eq!(decision.selected_action, command.to_action());
    assert!(tick
        .frame
        .reign_outcome
        .as_ref()
        .map(|outcome| outcome.accepted_by_conductor)
        .unwrap_or(false));
}

#[tokio::test]
async fn assist_reign_overrides_model_assisted_selector_immediately() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-assist-reign-model-assisted-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    )
    .with_action_selector_mode(ActionSelectorMode::ModelAssisted);
    let command = ReignCommand::Turn {
        direction: TurnDir::Right,
        intensity: 0.5,
        duration_ms: 500,
    };
    runtime.reign_queue.lock().unwrap().push(test_reign_input(
        100,
        ReignMode::Assist,
        command.clone(),
        2_000,
    ));
    let mut now = idle_now(100);
    now.drives.curiosity = 1.0;

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    assert_eq!(tick.chosen_action, command.to_action());
    let decision = tick
        .frame
        .now
        .extensions
        .get("action_selector")
        .cloned()
        .and_then(|value| serde_json::from_value::<ActionSelectionDecision>(value).ok())
        .unwrap();
    assert_eq!(decision.selected_action, command.to_action());
    assert!(tick
        .frame
        .reign_outcome
        .as_ref()
        .map(|outcome| outcome.accepted_by_conductor)
        .unwrap_or(false));
}

#[tokio::test]
async fn observe_or_suggest_reign_does_not_mechanically_override_selector() {
    for mode in [ReignMode::ObserveOnly, ReignMode::Suggest] {
        let ledger = JsonlLedger::new(format!("/tmp/pete-runtime-non-driving-reign-{mode:?}"));
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            FixedConductor::new(ActionPrimitive::Stop),
            SimpleSafety::default(),
            pete_llm::NoopLlmAgent,
        );
        let command = ReignCommand::Turn {
            direction: TurnDir::Right,
            intensity: 0.5,
            duration_ms: 500,
        };
        runtime
            .reign_queue
            .lock()
            .unwrap()
            .push(test_reign_input(100, mode, command, 2_000));

        let tick = runtime
            .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();

        assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
        assert!(tick.frame.reign_input.is_some());
        assert!(!tick
            .frame
            .reign_outcome
            .as_ref()
            .map(|outcome| outcome.accepted_by_conductor)
            .unwrap_or(true));
    }
}

#[tokio::test]
async fn stop_reign_becomes_now_event_and_chosen_action() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-reign-stop-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    );
    runtime.reign_queue.lock().unwrap().push(test_reign_input(
        100,
        ReignMode::Direct,
        ReignCommand::Stop,
        2_000,
    ));
    let mut body = BodySense::default();
    body.last_update_ms = 100;
    let now = Now::blank(100, body);

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    assert!(tick.frame.now.reign.active);
    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
    assert!(tick
        .frame
        .sensations
        .iter()
        .any(|sensation| sensation.kind == "reign.command"));
    assert!(tick
        .frame
        .reign_input
        .as_ref()
        .map(|input| matches!(input.command, ReignCommand::Stop))
        .unwrap_or(false));
    assert!(tick
        .frame
        .reign_outcome
        .as_ref()
        .map(|outcome| outcome.accepted_by_conductor)
        .unwrap_or(false));
}

#[test]
fn expired_reign_disappears_from_sense() {
    let mut queue = ReignQueue::default();
    queue.push(test_reign_input(
        100,
        ReignMode::Direct,
        ReignCommand::Stop,
        100,
    ));

    queue.drain_expired(250);
    let sense = queue.sense(250);

    assert!(!sense.active);
    assert!(sense.latest.is_none());
    assert_eq!(sense.pending_count, 0);
}

#[test]
fn clear_marks_reign_sense_for_event_extraction() {
    let mut queue = ReignQueue::default();
    queue.push(test_reign_input(
        100,
        ReignMode::Direct,
        ReignCommand::Stop,
        1_000,
    ));

    queue.clear();
    let sense = queue.sense(150);

    assert!(!sense.active);
    assert!(sense.latest.is_none());
    assert_eq!(sense.clear_sequence, 1);
}

#[tokio::test]
async fn safety_veto_beats_direct_go_reign_at_cliff() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-reign-safety-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    );
    runtime.reign_queue.lock().unwrap().push(test_reign_input(
        100,
        ReignMode::Direct,
        ReignCommand::Go {
            intensity: 0.5,
            duration_ms: 500,
        },
        2_000,
    ));
    let mut body = BodySense::default();
    body.flags.cliff_front_left = true;
    body.last_update_ms = 100;
    let now = Now::blank(100, body);

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    assert_eq!(
        tick.chosen_action,
        Some(ActionPrimitive::Go {
            intensity: 0.5,
            duration_ms: 500,
        })
    );
    assert!(tick
        .frame
        .reign_outcome
        .as_ref()
        .map(|outcome| outcome.vetoed_by_safety)
        .unwrap_or(false));
    assert!(tick
        .frame
        .notes
        .iter()
        .any(|note| note.contains("Safety vetoed")));
    let motor_gate = tick.frame.now.extensions.get("motor_gate").unwrap();
    assert_eq!(
        serde_json::from_value::<MotorCommand>(motor_gate["final_motor"].clone()).unwrap(),
        MotorCommand::stop()
    );
    assert_eq!(motor_gate["safety_reason"], "cliff");
}
