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
