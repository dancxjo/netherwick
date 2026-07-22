#[test]
fn observatory_page_exposes_timeline_now_and_accessible_navigation() {
    for marker in [
        "Brain Observatory",
        "Event timeline",
        "Canonical Now",
        "Occurred",
        "Observed",
        "Latency",
        "Time scrubber",
        "Raw JSON",
        "aria-label=\"Pause live updates\"",
        "aria-keyshortcuts=\"Space\"",
        "role=\"tree\"",
        "role=\"listbox\"",
    ] {
        assert!(OBSERVATORY_PAGE.contains(marker), "missing {marker}");
    }
}

#[test]
fn observatory_page_virtualizes_and_keeps_ingesting_while_paused() {
    for behavior in [
        "MAX_BROWSER_EVENTS",
        "TIMELINE_ROW_HEIGHT",
        "renderVirtualTimeline",
        "renderVirtualTree",
        "if(state.paused)",
        "ws.onmessage=message=>ingest",
        "transport_gap",
        "inline(e).reason === 'coalesced'",
        "unavailable gap",
        "event.replaced",
        "sessionStorage",
    ] {
        assert!(OBSERVATORY_PAGE.contains(behavior), "missing {behavior}");
    }
}

#[test]
fn observatory_query_filters_extended_dimensions() {
    let mut event = observatory_event(
        10,
        BrainEventType::Evidence,
        LossPolicy::Coalescible { key: "eye".into() },
    );
    event.kind = "vision.object".into();
    event.producer.brain = Brain::Forebrain;
    event.calibration_epochs.push("camera-v4".into());
    event.artifacts.push(pete_events::ArtifactIdentity {
        kind: ArtifactKind::Model,
        id: "detector-v9".into(),
        ..Default::default()
    });
    event.payload = BrainEventPayload::inline(serde_json::json!({"modality": "vision"}));

    assert!(BrainEventQuery {
        kind: Some("vision.object".into()),
        brain: Some(Brain::Forebrain),
        modality: Some("vision".into()),
        model: Some("detector-v9".into()),
        calibration_epoch: Some("camera-v4".into()),
        ..Default::default()
    }
    .matches(&event));
    assert!(!BrainEventQuery {
        modality: Some("audio".into()),
        ..Default::default()
    }
    .matches(&event));
}

#[tokio::test]
async fn live_updates_publish_and_retain_the_exact_selected_now_snapshot() {
    let state = LiveViewState::new();
    assert!(state.observatory().start());
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.last_update_ms = 4242;
    snapshot.body.battery_level = 0.73;

    state.update(snapshot);
    wait_for_observatory_sequence(&state.observatory(), 1).await;
    let response = state
        .observatory()
        .query(&BrainEventQuery::default())
        .unwrap();
    let BrainEventStreamRecord::Event { envelope } = &response.records[0] else {
        panic!("expected snapshot event");
    };
    let snapshot_id = envelope.event.references.snapshot_id.as_ref().unwrap();
    let retained = state.observatory_now_snapshot(snapshot_id).unwrap();

    assert_eq!(retained.selected.snapshot_id, *snapshot_id);
    assert_eq!(retained.selected.now.t_ms, 4242);
    assert_eq!(retained.selected.now.body.battery_level, 0.73);
    assert!(retained.previous.is_none());
}

#[tokio::test]
async fn snapshot_seek_returns_previous_tick_without_leaking_future_state() {
    let state = LiveViewState::new();
    assert!(state.observatory().start());
    for (t_ms, battery) in [(100, 0.9), (200, 0.7), (300, 0.5)] {
        let mut snapshot = WorldSnapshot::default();
        snapshot.body.last_update_ms = t_ms;
        snapshot.body.battery_level = battery;
        state.update(snapshot);
    }

    let selected = state.observatory_now_at_or_before(250).unwrap();
    assert_eq!(selected.selected.now.t_ms, 200);
    assert_eq!(selected.previous.unwrap().now.t_ms, 100);
}

fn observable_runtime_tick() -> (WorldSnapshot, RuntimeTick) {
    let t_ms = 500;
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.last_update_ms = t_ms;
    snapshot.action_debug = Some(serde_json::json!({
        "motor_applied": true,
        "motion_sent_to_sim": {"forward": 0.2, "turn": 0.0},
        "movement_delta": 0.02,
    }));
    let sensation = pete_experience::Sensation::new(
        "range.obstacle",
        "range.front",
        t_ms,
        t_ms,
        serde_json::json!({"range_m": 0.4}),
    );
    let impression = pete_experience::Impression::new(
        "obstacle.near",
        "obstacle is near",
        vec![sensation.id],
        t_ms,
        t_ms,
    );
    let experience = pete_experience::Experience::new(
        "world.obstacle",
        "near obstacle belief",
        vec![impression.id],
        vec![sensation.id],
        t_ms,
        t_ms,
    );
    let action = ActionPrimitive::Stop;
    let mut now = snapshot.to_now(t_ms);
    now.self_sense.active_goal = Some("goal-safe".into());
    now.extensions.insert(
        "motor_gate".into(),
        serde_json::json!({
            "desired_motor": {"forward": 0.0, "turn": 0.0},
            "final_motor": {"forward": 0.0, "turn": 0.0},
            "motor_applied": false,
            "vetoed": false,
            "safety_reason": null,
        }),
    );
    let frame = pete_ledger::ExperienceFrame {
        id: Uuid::new_v4(),
        t_ms,
        now,
        sensations: vec![sensation],
        impressions: vec![impression],
        experiences: vec![experience.clone()],
        z: Some(pete_experience::ExperienceLatent::default()),
        chosen_action: Some(action.clone()),
        conscious_command: None,
        reign_input: None,
        reign_outcome: None,
        predicted_futures: Vec::new(),
        behavior_runs: Vec::new(),
        actual_next: None,
        reward: pete_core::Reward::default(),
        surprise: pete_now::SurpriseSense::default(),
        memory_recall: Vec::new(),
        recollections: Vec::new(),
        llm_teaching: Vec::new(),
        counterfactuals: Vec::new(),
        notes: Vec::new(),
    };
    let mut runtime_boundary_event = BrainEvent::historical(
        BrainEventId::from_domain("runtime-boundary", frame.id),
        BrainEventType::BeliefUpdate,
        ProducerIdentity::new(Brain::Motherbrain, "runtime.test_boundary"),
        EventTimes::observed(t_ms, t_ms),
    );
    runtime_boundary_event.kind = "runtime.boundary.sentinel".into();
    runtime_boundary_event.references.frame_id = Some(frame.id.to_string());
    let tick = RuntimeTick {
        frame,
        experience,
        chosen_action: Some(action),
        skill_request: None,
        skill_status: None,
        recall: pete_memory::RecallBundle::default(),
        llm: pete_llm::LlmTickResult::default(),
        combobulation: None,
        inline_learning: pete_runtime::InlineLearningTickStatus::default(),
        brain_events: vec![runtime_boundary_event],
    };
    (snapshot, tick)
}

#[tokio::test]
async fn server_forwards_runtime_boundary_events_without_reconstructing_causality() {
    let (snapshot, tick) = observable_runtime_tick();
    let events = LiveViewState::runtime_tick_brain_events(&snapshot, &tick);
    for event in &events {
        event.validate().unwrap();
    }
    assert!(events
        .iter()
        .any(|event| event.kind == "runtime.boundary.sentinel"));
    for event_type in [
        BrainEventType::ProviderState,
        BrainEventType::JobState,
        BrainEventType::QueueState,
        BrainEventType::ResourceState,
    ] {
        assert!(events.iter().any(|event| event.event_type == event_type));
    }
    assert!(events
        .iter()
        .filter(|event| {
            matches!(
                event.event_type,
                BrainEventType::ProviderState
                    | BrainEventType::JobState
                    | BrainEventType::QueueState
                    | BrainEventType::ResourceState
            )
        })
        .all(|event| event.record_kind == BrainEventRecordKind::StateProjection));
    assert!(!events.iter().any(|event| matches!(
        event.kind.as_str(),
        "conductor.proposal"
            | "safety.decision"
            | "actuator.command.accepted_by_runtime"
            | "actuator.outcome"
    )));

    let state = LiveViewState::new();
    assert!(state.observatory().start());
    state.update(snapshot.clone());
    let published = state.publish_runtime_tick(&snapshot, &tick);
    wait_for_observatory_sequence(&state.observatory(), published as u64 + 1).await;
    let response = state
        .observatory()
        .query(&BrainEventQuery {
            limit: Some(100),
            ..BrainEventQuery::default()
        })
        .unwrap();
    let snapshot_ids = response
        .records
        .iter()
        .filter_map(|record| match record {
            BrainEventStreamRecord::Event { envelope }
                if envelope.event.event_type != BrainEventType::Snapshot =>
            {
                envelope.event.references.snapshot_id.as_deref()
            }
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(snapshot_ids.len(), 1);
    state.observatory().shutdown().await;
}

#[tokio::test]
async fn scene_calibration_updates_publish_first_class_epoch_transitions() {
    let state = LiveViewState::new();
    assert!(state.observatory().start());
    state.update_scene_metadata(LiveSceneMetadata {
        sensor_calibration: Some(SceneSensorCalibration::sim_default()),
        ..LiveSceneMetadata::default()
    });
    wait_for_observatory_sequence(&state.observatory(), 1).await;

    let response = state
        .observatory()
        .query(&BrainEventQuery::default())
        .unwrap();
    let event = response
        .records
        .iter()
        .find_map(|record| match record {
            BrainEventStreamRecord::Event { envelope }
                if envelope.event.event_type == BrainEventType::CalibrationTransition =>
            {
                Some(&envelope.event)
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(event.calibration_epochs.len(), 1);
    assert_eq!(event.artifacts[0].kind, ArtifactKind::Calibration);
    state.observatory().shutdown().await;
}
