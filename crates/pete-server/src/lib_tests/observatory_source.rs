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
fn capture_gap_ends_before_the_present_frame() {
    let gap = capture_frame_gap(1, 3).expect("indices 1 and 2 are missing");
    assert_eq!((gap.from_sequence, gap.to_sequence), (1, 2));
    assert!(capture_frame_gap(3, 3).is_none());
}

#[tokio::test]
async fn live_snapshot_discovery_paginates_through_the_newest_snapshot() {
    let hub = BrainEventHub::new(BrainEventHubConfig {
        ingress_capacity: 4_096,
        history_capacity: 4_096,
        broadcast_capacity: 4,
        ..BrainEventHubConfig::default()
    });
    assert!(hub.start());
    for sequence in 1..=2_048 {
        let mut event = source_snapshot(sequence, sequence, "boot-a").event;
        event.loss_policy = LossPolicy::LossIntolerant;
        hub.publish(event).unwrap();
    }
    wait_for_observatory_sequence(&hub, 2_048).await;

    let source = LiveBrainEventSource::new("live:test", "live test", hub.clone());
    let snapshots = source.snapshots();
    assert_eq!(snapshots.len(), 2_048);
    assert_eq!(snapshots.last().map(|snapshot| snapshot.t_ms), Some(2_048));
    assert_eq!(
        source
            .snapshot_at_or_before(2_048)
            .map(|snapshot| snapshot.t_ms),
        Some(2_048)
    );
    hub.shutdown().await;
}

#[test]
fn recorded_chain_queries_identically_after_replay() {
    let recorded = vec![
        source_snapshot(1, 100, "boot-a"),
        source_snapshot(2, 200, "boot-a"),
    ];
    let source = replay_source(recorded.clone());
    let response = source.query(&ObservatorySourceQuery::default()).unwrap();
    let replayed = response
        .events
        .into_iter()
        .map(|event| event.envelope)
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
    let response = source.query(&ObservatorySourceQuery::default()).unwrap();
    let events = response
        .events
        .into_iter()
        .map(|event| event.envelope)
        .collect::<Vec<_>>();
    assert_eq!(events[0].sequence, 1);
    assert_eq!(events[1].sequence, 2);
    assert_eq!(
        events[0].event.times.occurred.clock_epoch.as_deref(),
        Some("boot-a")
    );
    assert_eq!(
        events[1].event.times.occurred.clock_epoch.as_deref(),
        Some("boot-b")
    );
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
    let recorded_query = source.query(&ObservatorySourceQuery::default()).unwrap();
    assert_eq!(recorded_query.events.len(), 1);
    assert_eq!(recorded_query.events[0].envelope, recorded);
    assert_eq!(
        recorded_query.events[0].lane,
        ObservatoryEventLane::Recorded
    );

    let candidate_query = source
        .query(&ObservatorySourceQuery {
            lane: ObservatoryEventLaneSelector::Reprocessed,
            lane_model: Some("candidate-v2".into()),
            ..ObservatorySourceQuery::default()
        })
        .unwrap();
    assert_eq!(candidate_query.events.len(), 1);
    assert_eq!(candidate_query.events[0].envelope, candidate);
    assert_eq!(
        candidate_query.events[0].lane,
        ObservatoryEventLane::Reprocessed {
            model_id: "candidate-v2".into()
        }
    );
}

#[test]
fn all_lane_pagination_keeps_events_that_share_a_domain_sequence() {
    let recorded = source_snapshot(42, 100, "boot-a");
    let mut candidate_a = source_snapshot(42, 100, "boot-a");
    candidate_a.event.event_id = BrainEventId::from_domain("candidate-a", 42);
    let mut candidate_b = source_snapshot(42, 100, "boot-a");
    candidate_b.event.event_id = BrainEventId::from_domain("candidate-b", 42);
    let mut source = replay_source(vec![recorded.clone()]);
    source.add_reprocessed_lane("model-b", vec![candidate_b.clone()]);
    source.add_reprocessed_lane("model-a", vec![candidate_a.clone()]);

    let mut after = None;
    let mut pages = Vec::new();
    loop {
        let response = source
            .query(&ObservatorySourceQuery {
                event: BrainEventQuery {
                    after_sequence: after,
                    limit: Some(1),
                    ..BrainEventQuery::default()
                },
                lane: ObservatoryEventLaneSelector::All,
                ..ObservatorySourceQuery::default()
            })
            .unwrap();
        if response.events.is_empty() {
            break;
        }
        assert_eq!(response.events.len(), 1);
        assert!(after.is_none_or(|cursor| response.next_cursor > cursor));
        after = Some(response.next_cursor);
        pages.push(response.events[0].clone());
    }

    assert_eq!(pages.len(), 3);
    assert_eq!(
        pages
            .iter()
            .map(|event| event.source_order)
            .collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert!(pages.iter().all(|event| event.envelope.sequence == 42));
    assert_eq!(pages[0].lane, ObservatoryEventLane::Recorded);
    assert_eq!(
        pages[1].lane,
        ObservatoryEventLane::Reprocessed {
            model_id: "model-a".to_string()
        }
    );
    assert_eq!(
        pages[2].lane,
        ObservatoryEventLane::Reprocessed {
            model_id: "model-b".to_string()
        }
    );
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
    let response = source.query(&ObservatorySourceQuery::default()).unwrap();
    assert_eq!(response.events.len(), 1);
    assert_eq!(response.gaps.len(), 1);
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
