fn source_snapshot(sequence: u64, t_ms: u64, epoch: &str) -> SequencedBrainEvent {
    let now = pete_now::Now::blank(t_ms, pete_body::BodySense::default());
    SequencedBrainEvent {
        sequence,
        event: BrainEvent::from_now_snapshot(
            format!("snapshot-{sequence}"),
            &now,
            t_ms + 1,
            Some(epoch.to_string()),
        ),
    }
}

fn replay_source(events: Vec<SequencedBrainEvent>) -> ReplayBrainEventSource {
    ReplayBrainEventSource::from_recorded(
        ObservatorySourceIdentity {
            id: "capture:test".to_string(),
            kind: ObservatorySourceKind::Capture,
            label: "test capture".to_string(),
        },
        events,
    )
}

#[test]
fn recorded_chain_queries_identically_after_replay() {
    let recorded = vec![
        source_snapshot(1, 100, "boot-a"),
        source_snapshot(2, 200, "boot-a"),
    ];
    let source = replay_source(recorded.clone());
    let response = source.query(&BrainEventQuery::default()).unwrap();
    let replayed = response
        .records
        .into_iter()
        .filter_map(|record| match record {
            BrainEventStreamRecord::Event { envelope } => Some(envelope),
            BrainEventStreamRecord::Gap { .. } => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(replayed, recorded);
}

#[test]
fn seek_never_selects_a_future_snapshot() {
    let source = replay_source(vec![
        source_snapshot(1, 100, "boot-a"),
        source_snapshot(2, 200, "boot-a"),
        source_snapshot(3, 300, "boot-a"),
    ]);
    let mut navigation = ObservatoryNavigationState::new("capture:test");
    navigation.seek(&source, 250);
    assert_eq!(navigation.selected_time_ms, Some(250));
    assert_eq!(
        navigation.selected_event_id,
        Some(BrainEventId::from_domain("now", "snapshot-2"))
    );
    navigation.seek(&source, 50);
    assert_eq!(navigation.selected_event_id, None);
}

#[test]
fn clock_epoch_resets_remain_recorded_not_reordered() {
    let recorded = vec![
        source_snapshot(1, 900, "boot-a"),
        source_snapshot(2, 10, "boot-b"),
    ];
    let source = replay_source(recorded.clone());
    let response = source.query(&BrainEventQuery::default()).unwrap();
    let events = response
        .records
        .into_iter()
        .filter_map(|record| match record {
            BrainEventStreamRecord::Event { envelope } => Some(envelope),
            BrainEventStreamRecord::Gap { .. } => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(events[0].sequence, 1);
    assert_eq!(events[1].sequence, 2);
    assert_eq!(events[0].event.times.occurred.clock_epoch.as_deref(), Some("boot-a"));
    assert_eq!(events[1].event.times.occurred.clock_epoch.as_deref(), Some("boot-b"));
}

#[test]
fn candidate_model_events_stay_in_a_separate_lane() {
    let recorded = source_snapshot(1, 100, "boot-a");
    let mut candidate = source_snapshot(2, 100, "boot-a");
    candidate.event.event_id = BrainEventId::from_domain("candidate", 2);
    let mut source = replay_source(vec![recorded.clone()]);
    source.add_reprocessed_lane("candidate-v2", vec![candidate.clone()]);

    assert_eq!(source.events()[0].lane, ObservatoryEventLane::Recorded);
    assert_eq!(
        source.events()[1].lane,
        ObservatoryEventLane::Reprocessed {
            model_id: "candidate-v2".to_string()
        }
    );
    let query = source.query(&BrainEventQuery::default()).unwrap();
    assert_eq!(query.records.len(), 1);
    assert!(matches!(
        &query.records[0],
        BrainEventStreamRecord::Event { envelope } if envelope == &recorded
    ));
}

#[test]
fn incomplete_source_reports_gaps_without_hiding_available_history() {
    let mut source = replay_source(vec![source_snapshot(3, 300, "boot-a")]);
    source.gaps.push(BrainEventSequenceGap::new(
        1,
        2,
        SequenceGapReason::RetentionExpired,
    ));
    source.warnings.push("missing depth stream".to_string());
    let health = source.health();
    assert!(!health.complete);
    assert_eq!(health.gaps[0].from_sequence, 1);
    let response = source.query(&BrainEventQuery::default()).unwrap();
    assert_eq!(
        response
            .records
            .iter()
            .filter(|record| matches!(record, BrainEventStreamRecord::Event { .. }))
            .count(),
        1
    );
}

#[test]
fn navigation_supports_play_pause_step_speed_loop_and_follow_live() {
    let source = replay_source(vec![
        source_snapshot(1, 100, "boot-a"),
        source_snapshot(2, 200, "boot-a"),
    ]);
    let mut navigation = ObservatoryNavigationState::new("capture:test");
    navigation.play();
    assert_eq!(navigation.mode, ObservatoryPlaybackMode::Playing);
    navigation.pause();
    navigation.step(&source, 1);
    assert_eq!(navigation.selected_time_ms, Some(100));
    navigation.step(&source, 1);
    assert_eq!(navigation.selected_time_ms, Some(200));
    navigation.step(&source, -1);
    assert_eq!(navigation.selected_time_ms, Some(100));
    navigation.set_speed(4.0).unwrap();
    navigation.set_loop(Some([100, 200])).unwrap();
    navigation.follow_live();
    assert_eq!(navigation.mode, ObservatoryPlaybackMode::FollowLive);
    assert_eq!(navigation.selected_time_ms, None);
}

#[test]
fn deep_link_state_round_trips_source_selection_panel_and_filters() {
    let mut navigation = ObservatoryNavigationState::new("capture:test");
    navigation.selected_time_ms = Some(123);
    navigation.selected_event_id = Some(BrainEventId::from_domain("event", 7));
    navigation.panel = "provenance".to_string();
    navigation.filters.event_type = Some(BrainEventType::BeliefUpdate);
    navigation.filters.trust = Some(TrustState::Conditional);
    let encoded = serde_json::to_vec(&navigation).unwrap();
    let decoded: ObservatoryNavigationState = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, navigation);
}
