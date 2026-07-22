fn observatory_event(id: u64, event_type: BrainEventType, loss_policy: LossPolicy) -> BrainEvent {
    let mut event = BrainEvent::historical(
        BrainEventId::from_domain("test", id),
        event_type,
        ProducerIdentity::new(Brain::Motherbrain, "test.component"),
        EventTimes::observed(id * 10, id * 10 + 2),
    );
    event.kind = format!("test.{id}");
    event.loss_policy = loss_policy;
    event
}

async fn wait_for_observatory_sequence(hub: &BrainEventHub, sequence: u64) {
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if hub
                .health()
                .newest_sequence
                .is_some_and(|value| value >= sequence)
            {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("observatory worker did not drain in time");
}

#[tokio::test]
async fn observatory_assigns_monotonic_sequences_and_reconnects_exactly() {
    let hub = BrainEventHub::new(BrainEventHubConfig {
        ingress_capacity: 16,
        history_capacity: 16,
        broadcast_capacity: 4,
        ..BrainEventHubConfig::default()
    });
    assert!(hub.start());
    for id in 1..=5 {
        hub.publish(observatory_event(
            id,
            BrainEventType::Command,
            LossPolicy::LossIntolerant,
        ))
        .unwrap();
    }
    wait_for_observatory_sequence(&hub, 5).await;

    let response = hub
        .query(&BrainEventQuery {
            after_sequence: Some(3),
            ..BrainEventQuery::default()
        })
        .unwrap();
    let sequences = response
        .records
        .iter()
        .filter_map(|record| match record {
            BrainEventStreamRecord::Event { envelope } => Some(envelope.sequence),
            BrainEventStreamRecord::Gap { .. } => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(sequences, vec![4, 5]);
    assert!(!response
        .records
        .iter()
        .any(|record| matches!(record, BrainEventStreamRecord::Gap { .. })));
    assert_eq!(response.next_cursor, 5);
    hub.shutdown().await;
}

#[tokio::test]
async fn cursor_expiry_emits_a_loss_intolerant_typed_gap() {
    let hub = BrainEventHub::new(BrainEventHubConfig {
        ingress_capacity: 16,
        history_capacity: 3,
        broadcast_capacity: 4,
        ..BrainEventHubConfig::default()
    });
    assert!(hub.start());
    for id in 1..=5 {
        hub.publish(observatory_event(
            id,
            BrainEventType::Command,
            LossPolicy::LossIntolerant,
        ))
        .unwrap();
    }
    wait_for_observatory_sequence(&hub, 5).await;

    let response = hub.query(&BrainEventQuery::default()).unwrap();
    let gap = response
        .records
        .iter()
        .find_map(|record| match record {
            BrainEventStreamRecord::Gap { gap } => Some(gap),
            BrainEventStreamRecord::Event { .. } => None,
        })
        .expect("expired cursor should emit a gap");
    assert_eq!((gap.from_sequence, gap.to_sequence), (1, 2));
    assert_eq!(gap.reason, SequenceGapReason::RetentionExpired);
    assert_eq!(gap.event.event_type, BrainEventType::TransportGap);
    assert!(matches!(gap.event.loss_policy, LossPolicy::LossIntolerant));
    assert_eq!(response.health.history_expired_critical, 2);
    hub.shutdown().await;
}

#[tokio::test]
async fn telemetry_overflow_cannot_displace_a_loss_intolerant_event() {
    let hub = BrainEventHub::new(BrainEventHubConfig {
        ingress_capacity: 2,
        history_capacity: 8,
        broadcast_capacity: 4,
        ..BrainEventHubConfig::default()
    });
    for id in 1..=2 {
        hub.publish(observatory_event(
            id,
            BrainEventType::Evidence,
            LossPolicy::Coalescible {
                key: format!("telemetry.{id}"),
            },
        ))
        .unwrap();
    }
    let command = observatory_event(3, BrainEventType::Command, LossPolicy::LossIntolerant);
    let command_id = command.event_id.clone();
    assert_eq!(hub.publish(command).unwrap(), PublishOutcome::Queued);
    assert_eq!(hub.health().ingress_dropped_telemetry, 1);

    assert!(hub.start());
    wait_for_observatory_sequence(&hub, 2).await;
    let response = hub.query(&BrainEventQuery::default()).unwrap();
    assert!(response.records.iter().any(|record| matches!(
        record,
        BrainEventStreamRecord::Event { envelope } if envelope.event.event_id == command_id
    )));
    assert_eq!(hub.health().ingress_rejected_critical, 0);
    hub.shutdown().await;
}

#[tokio::test]
async fn pending_and_retained_telemetry_coalesce_by_stable_key() {
    let hub = BrainEventHub::new(BrainEventHubConfig {
        ingress_capacity: 4,
        history_capacity: 4,
        broadcast_capacity: 2,
        ..BrainEventHubConfig::default()
    });
    for id in 1..=100 {
        let outcome = hub
            .publish(observatory_event(
                id,
                BrainEventType::Evidence,
                LossPolicy::Coalescible {
                    key: "body.battery".to_string(),
                },
            ))
            .unwrap();
        if id > 1 {
            assert_eq!(outcome, PublishOutcome::CoalescedPendingTelemetry);
        }
    }
    assert_eq!(hub.health().ingress_depth, 1);
    assert_eq!(hub.health().ingress_coalesced, 99);
    assert!(hub.start());
    wait_for_observatory_sequence(&hub, 1).await;
    let response = hub.query(&BrainEventQuery::default()).unwrap();
    let events = response
        .records
        .iter()
        .filter(|record| matches!(record, BrainEventStreamRecord::Event { .. }))
        .count();
    assert_eq!(events, 1);
    hub.shutdown().await;
}

#[tokio::test]
async fn retained_telemetry_replacement_is_not_reported_as_data_loss() {
    let hub = BrainEventHub::new(BrainEventHubConfig {
        ingress_capacity: 4,
        history_capacity: 4,
        broadcast_capacity: 2,
        ..BrainEventHubConfig::default()
    });
    assert!(hub.start());
    for id in 1..=2 {
        hub.publish(observatory_event(
            id,
            BrainEventType::Snapshot,
            LossPolicy::Coalescible {
                key: "now.current".to_string(),
            },
        ))
        .unwrap();
        wait_for_observatory_sequence(&hub, id).await;
    }

    let response = hub.query(&BrainEventQuery::default()).unwrap();
    let gap = response
        .records
        .iter()
        .find_map(|record| match record {
            BrainEventStreamRecord::Gap { gap } => Some(gap),
            BrainEventStreamRecord::Event { .. } => None,
        })
        .expect("replacement tombstone should remain queryable");
    assert_eq!(gap.reason, SequenceGapReason::Coalesced);
    assert_eq!(gap.replacement_sequence, Some(2));
    assert_eq!(gap.event.disposition, EventDisposition::Superseded);
    assert_eq!(response.health.history_coalesced, 1);
    assert_eq!(response.health.history_expired, 0);
    assert!(!response.records.iter().any(|record| matches!(
        record,
        BrainEventStreamRecord::Gap { gap }
            if gap.reason == SequenceGapReason::RetentionExpired
    )));
    hub.shutdown().await;
}

#[test]
fn history_discontinuity_reconstruction_is_bounded_by_replacement_tombstones() {
    let replacements = BTreeMap::from([(10, 11), (20, 21)]);
    let mut records = Vec::new();

    append_history_discontinuities(&mut records, &replacements, 1, u64::MAX);

    let gaps = records
        .into_iter()
        .map(|record| match record {
            BrainEventStreamRecord::Gap { gap } => gap,
            BrainEventStreamRecord::Event { .. } => panic!("expected only discontinuities"),
        })
        .collect::<Vec<_>>();
    assert_eq!(gaps.len(), 5);
    assert_eq!(
        (gaps[0].from_sequence, gaps[0].to_sequence, gaps[0].reason),
        (1, 9, SequenceGapReason::RetentionExpired)
    );
    assert_eq!(
        (
            gaps[1].from_sequence,
            gaps[1].to_sequence,
            gaps[1].reason,
            gaps[1].replacement_sequence,
        ),
        (10, 10, SequenceGapReason::Coalesced, Some(11))
    );
    assert_eq!(
        (gaps[2].from_sequence, gaps[2].to_sequence, gaps[2].reason),
        (11, 19, SequenceGapReason::RetentionExpired)
    );
    assert_eq!(
        (
            gaps[3].from_sequence,
            gaps[3].to_sequence,
            gaps[3].reason,
            gaps[3].replacement_sequence,
        ),
        (20, 20, SequenceGapReason::Coalesced, Some(21))
    );
    assert_eq!(
        (gaps[4].from_sequence, gaps[4].to_sequence, gaps[4].reason),
        (21, u64::MAX, SequenceGapReason::RetentionExpired)
    );
}

#[tokio::test]
async fn stalled_broadcast_client_never_backpressures_history_ingestion() {
    let hub = BrainEventHub::new(BrainEventHubConfig {
        ingress_capacity: 256,
        history_capacity: 64,
        broadcast_capacity: 2,
        ..BrainEventHubConfig::default()
    });
    assert!(hub.start());
    let mut stalled = hub.subscribe();
    for id in 1..=100 {
        let _ = hub.publish(observatory_event(
            id,
            BrainEventType::Evidence,
            LossPolicy::Coalescible {
                key: format!("sensor.{id}"),
            },
        ));
    }
    wait_for_observatory_sequence(&hub, 64).await;
    assert_eq!(hub.health().history_depth, 64);
    assert!(matches!(
        stalled.recv().await,
        Err(tokio::sync::broadcast::error::RecvError::Lagged(_))
    ));
    hub.shutdown().await;
}

#[tokio::test]
async fn long_synthetic_run_remains_bounded() {
    let hub = BrainEventHub::new(BrainEventHubConfig {
        ingress_capacity: 32,
        history_capacity: 24,
        broadcast_capacity: 8,
        ..BrainEventHubConfig::default()
    });
    assert!(hub.start());
    for id in 1..=10_000 {
        let _ = hub.publish(observatory_event(
            id,
            BrainEventType::Evidence,
            LossPolicy::Coalescible {
                key: format!("field.{}", id % 64),
            },
        ));
        if id % 128 == 0 {
            tokio::task::yield_now().await;
        }
    }
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    let health = hub.health();
    assert!(health.ingress_depth <= 32);
    assert!(health.history_depth <= 24);
    assert!(health.ingress_coalesced + health.ingress_dropped_telemetry > 0);
    hub.shutdown().await;
}

#[tokio::test]
async fn history_filters_cover_every_query_dimension() {
    let hub = BrainEventHub::new(BrainEventHubConfig::default());
    assert!(hub.start());
    let mut matching = observatory_event(
        50,
        BrainEventType::BeliefUpdate,
        LossPolicy::Coalescible {
            key: "belief.person".to_string(),
        },
    );
    matching.producer.component = "world.model".to_string();
    matching.references.snapshot_id = Some("snapshot-50".to_string());
    matching.references.entity_ids.push("person-1".to_string());
    matching.references.goal_ids.push("greet-1".to_string());
    matching
        .references
        .command_ids
        .push("command-1".to_string());
    matching.quality.trust = TrustState::Trusted;
    matching.disposition = EventDisposition::Accepted;
    hub.publish(matching).unwrap();
    hub.publish(observatory_event(
        60,
        BrainEventType::Evidence,
        LossPolicy::Coalescible {
            key: "body.range".to_string(),
        },
    ))
    .unwrap();
    wait_for_observatory_sequence(&hub, 2).await;

    let response = hub
        .query(&BrainEventQuery {
            occurred_from_ms: Some(490),
            occurred_to_ms: Some(510),
            observed_from_ms: Some(500),
            observed_to_ms: Some(505),
            event_type: Some(BrainEventType::BeliefUpdate),
            component: Some("world.model".to_string()),
            snapshot: Some("snapshot-50".to_string()),
            entity: Some("person-1".to_string()),
            goal: Some("greet-1".to_string()),
            command: Some("command-1".to_string()),
            trust: Some(TrustState::Trusted),
            disposition: Some(EventDisposition::Accepted),
            ..BrainEventQuery::default()
        })
        .unwrap();
    let matching_events = response
        .records
        .iter()
        .filter(|record| matches!(record, BrainEventStreamRecord::Event { .. }))
        .count();
    assert_eq!(matching_events, 1);
    hub.shutdown().await;
}

#[test]
fn malformed_history_filters_are_rejected() {
    assert!(BrainEventQuery {
        occurred_from_ms: Some(20),
        occurred_to_ms: Some(10),
        ..BrainEventQuery::default()
    }
    .validate(100)
    .is_err());
    assert!(BrainEventQuery {
        limit: Some(0),
        ..BrainEventQuery::default()
    }
    .validate(100)
    .is_err());
    assert!(BrainEventQuery {
        component: Some(" ".to_string()),
        ..BrainEventQuery::default()
    }
    .validate(100)
    .is_err());
}

#[test]
fn oversized_inline_payload_is_rejected_at_ingress() {
    let hub = BrainEventHub::new(BrainEventHubConfig::default());
    let mut event = observatory_event(
        1,
        BrainEventType::Evidence,
        LossPolicy::Coalescible {
            key: "oversized".to_string(),
        },
    );
    event.payload = BrainEventPayload::inline(serde_json::json!({
        "raw": "x".repeat(pete_events::MAX_INLINE_BRAIN_EVENT_PAYLOAD_BYTES + 1),
    }));
    assert!(matches!(
        hub.publish(event),
        Err(BrainEventPublishError::InvalidEvent(_))
    ));
}

#[tokio::test]
async fn observatory_shutdown_drains_and_rejects_new_publication() {
    let hub = BrainEventHub::new(BrainEventHubConfig::default());
    assert!(hub.start());
    hub.publish(observatory_event(
        1,
        BrainEventType::Evidence,
        LossPolicy::Coalescible {
            key: "test".to_string(),
        },
    ))
    .unwrap();
    hub.shutdown().await;
    assert!(hub.health().closed);
    assert!(matches!(
        hub.publish(observatory_event(
            2,
            BrainEventType::Evidence,
            LossPolicy::Coalescible {
                key: "test".to_string(),
            },
        )),
        Err(BrainEventPublishError::Closed)
    ));
}
