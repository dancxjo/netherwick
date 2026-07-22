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
