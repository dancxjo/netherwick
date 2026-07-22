#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObservatoryStressConfig {
    pub events: u64,
    pub critical_every: u64,
    pub telemetry_keys: u64,
    pub ingress_capacity: usize,
    pub history_capacity: usize,
    pub broadcast_capacity: usize,
    pub durable_directory: PathBuf,
    pub sync_data: bool,
    pub publish_interval_us: u64,
    pub max_publish_p99_us: f64,
    pub max_rss_growth_bytes: u64,
}

impl ObservatoryStressConfig {
    pub fn ci(durable_directory: impl Into<PathBuf>) -> Self {
        Self {
            events: 20_000,
            critical_every: 64,
            telemetry_keys: 64,
            ingress_capacity: 512,
            history_capacity: 512,
            broadcast_capacity: 32,
            durable_directory: durable_directory.into(),
            sync_data: false,
            publish_interval_us: 0,
            max_publish_p99_us: 1_000.0,
            max_rss_growth_bytes: 128 * 1024 * 1024,
        }
    }

    pub fn pi5_soak(durable_directory: impl Into<PathBuf>) -> Self {
        Self {
            events: 360_000,
            sync_data: true,
            publish_interval_us: 20_000,
            max_publish_p99_us: 500.0,
            max_rss_growth_bytes: 192 * 1024 * 1024,
            ..Self::ci(durable_directory)
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ObservatoryLatencySummary {
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub max_us: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ObservatoryStressMetrics {
    pub attempted_events: u64,
    pub queued_critical_events: u64,
    pub critical_rejections: u64,
    pub telemetry_coalesced: u64,
    pub telemetry_dropped: u64,
    pub max_ingress_depth: usize,
    pub max_history_depth: usize,
    pub max_durable_backlog: usize,
    pub final_sequence: u64,
    pub durable_sequence: u64,
    pub durable_write_failures: u64,
    pub declared_cursor_gaps: usize,
    pub serialized_history_bytes: u64,
    pub recovered_critical_events: usize,
    pub replay_order_matches: bool,
    pub clock_reset_preserved: bool,
    pub stalled_client_lagged: bool,
    pub reconnect_received_live_event: bool,
    pub injected_writer_failures: u64,
    pub publish_deadline_misses: u64,
    pub durable_bytes: u64,
    #[serde(alias = "baseline")]
    pub construct_only: ObservatoryLatencySummary,
    #[serde(alias = "enabled")]
    pub construct_and_publish: ObservatoryLatencySummary,
    pub added_publish_p99_us: f64,
    pub rss_before_bytes: Option<u64>,
    pub rss_after_bytes: Option<u64>,
    pub rss_growth_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObservatoryStressReport {
    pub schema_version: u32,
    pub generated_at_ms: u64,
    pub host_arch: String,
    pub config: ObservatoryStressConfig,
    pub artifact_directory: PathBuf,
    pub metrics: ObservatoryStressMetrics,
    pub checks: BTreeMap<String, bool>,
    pub passed: bool,
    pub physical_pi_soak_performed: bool,
    pub physical_pi_soak_command: String,
    pub coverage: Vec<String>,
}

fn stress_event(id: u64, critical: bool, telemetry_keys: u64) -> BrainEvent {
    let reset = id >= 10_000;
    let occurred_ms = if reset {
        id.saturating_sub(10_000)
    } else {
        1_000_000_u64.saturating_add(id)
    };
    let epoch = if reset { "sensor-epoch-b" } else { "sensor-epoch-a" };
    let mut event = BrainEvent::historical(
        BrainEventId::from_domain("observatory-stress", id),
        if critical {
            BrainEventType::Command
        } else {
            BrainEventType::Evidence
        },
        ProducerIdentity::new(Brain::Motherbrain, "observatory.stress.fixture"),
        EventTimes {
            occurred: ClockedTime::in_epoch(occurred_ms, epoch),
            observed: ClockedTime::in_epoch(2_000_000_u64.saturating_add(id), "host-epoch-a"),
            valid_from: None,
            expires_at: None,
        },
    );
    event.kind = if critical {
        "stress.critical_transition"
    } else {
        "stress.telemetry_projection"
    }
    .into();
    event.payload = BrainEventPayload::inline(serde_json::json!({
        "fixture_sequence": id,
        "clock_reset": reset,
    }));
    event.loss_policy = if critical {
        LossPolicy::LossIntolerant
    } else {
        LossPolicy::Coalescible {
            key: format!("stress.field.{}", id % telemetry_keys.max(1)),
        }
    };
    event
}

fn stress_latency_summary(samples_ns: &mut [u64]) -> ObservatoryLatencySummary {
    if samples_ns.is_empty() {
        return ObservatoryLatencySummary::default();
    }
    samples_ns.sort_unstable();
    let percentile = |percent: usize| {
        let index = (samples_ns.len().saturating_sub(1) * percent) / 100;
        samples_ns[index] as f64 / 1_000.0
    };
    ObservatoryLatencySummary {
        p50_us: percentile(50),
        p95_us: percentile(95),
        p99_us: percentile(99),
        max_us: *samples_ns.last().unwrap_or(&0) as f64 / 1_000.0,
    }
}

fn stress_rss_bytes() -> Option<u64> {
    let resident_pages = fs::read_to_string("/proc/self/statm")
        .ok()?
        .split_ascii_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()?;
    Some(resident_pages.saturating_mul(4_096))
}

fn stress_query_all(hub: &BrainEventHub) -> Result<Vec<SequencedBrainEvent>, BrainEventQueryError> {
    let mut after = 0;
    let mut events = Vec::new();
    loop {
        let page = hub.query(&BrainEventQuery {
            after_sequence: Some(after),
            limit: Some(2_000),
            ..Default::default()
        })?;
        let next = page.next_cursor;
        events.extend(page.records.into_iter().filter_map(|record| match record {
            BrainEventStreamRecord::Event { envelope } => Some(envelope),
            BrainEventStreamRecord::Gap { .. } => None,
        }));
        if next <= after || page.health.newest_sequence.is_none_or(|newest| next >= newest) {
            break;
        }
        after = next;
    }
    Ok(events)
}

async fn stress_wait_drained(hub: &BrainEventHub) -> bool {
    tokio::time::timeout(std::time::Duration::from_secs(15), async {
        loop {
            let health = hub.health();
            if health.ingress_depth == 0
                && health.ingress_inflight == 0
                && health.durable_writer_backlog == 0
            {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .is_ok()
}

pub async fn run_observatory_stress(
    config: ObservatoryStressConfig,
) -> io::Result<ObservatoryStressReport> {
    fs::create_dir_all(&config.durable_directory)?;
    let artifact_directory = config
        .durable_directory
        .join(format!("run-{}", Uuid::new_v4()));
    fs::create_dir_all(&artifact_directory)?;
    let durable_path = artifact_directory.join("critical-events.jsonl");
    let mut durability = BrainEventDurabilityConfig::new(&durable_path);
    durability.sync_data = config.sync_data;
    durability.max_segment_bytes = 64 * 1024 * 1024;
    durability.retained_segments = 4;
    let hub_config = BrainEventHubConfig {
        ingress_capacity: config.ingress_capacity,
        history_capacity: config.history_capacity,
        broadcast_capacity: config.broadcast_capacity,
        ..Default::default()
    };
    let hub = BrainEventHub::new_with_durability(hub_config, durability.clone())?;
    if !hub.start() {
        return Err(io::Error::other("Tokio runtime unavailable for stress harness"));
    }
    let mut stalled_client = hub.subscribe();
    let rss_before_bytes = stress_rss_bytes();
    const MAX_LATENCY_SAMPLES: usize = 100_000;
    let sample_capacity = (config.events as usize).min(MAX_LATENCY_SAMPLES);
    let mut baseline_samples = Vec::with_capacity(sample_capacity);
    let mut enabled_samples = Vec::with_capacity(sample_capacity);
    let mut metrics = ObservatoryStressMetrics {
        attempted_events: config.events,
        rss_before_bytes,
        ..Default::default()
    };
    for id in 0..config.events {
        let critical = id % config.critical_every.max(1) == 0;
        let baseline_started = std::time::Instant::now();
        std::hint::black_box(stress_event(id, critical, config.telemetry_keys));
        let baseline_ns = baseline_started.elapsed().as_nanos() as u64;

        let enabled_started = std::time::Instant::now();
        let event = stress_event(id, critical, config.telemetry_keys);
        match hub.publish(event) {
            Ok(PublishOutcome::Queued) if critical => {
                metrics.queued_critical_events = metrics.queued_critical_events.saturating_add(1);
            }
            Ok(PublishOutcome::CoalescedPendingTelemetry) => {
                metrics.telemetry_coalesced = metrics.telemetry_coalesced.saturating_add(1);
            }
            Ok(PublishOutcome::DroppedTelemetry) => {
                metrics.telemetry_dropped = metrics.telemetry_dropped.saturating_add(1);
            }
            Ok(PublishOutcome::Duplicate | PublishOutcome::Queued) => {}
            Err(BrainEventPublishError::CriticalQueueFull) => {
                metrics.critical_rejections = metrics.critical_rejections.saturating_add(1);
            }
            Err(error) => return Err(io::Error::other(error)),
        }
        let enabled_elapsed = enabled_started.elapsed();
        let enabled_ns = enabled_elapsed.as_nanos() as u64;
        if baseline_samples.len() < MAX_LATENCY_SAMPLES {
            baseline_samples.push(baseline_ns);
            enabled_samples.push(enabled_ns);
        } else {
            let index = id as usize % MAX_LATENCY_SAMPLES;
            baseline_samples[index] = baseline_ns;
            enabled_samples[index] = enabled_ns;
        }
        if config.publish_interval_us > 0 {
            let interval = std::time::Duration::from_micros(config.publish_interval_us);
            if enabled_elapsed > interval {
                metrics.publish_deadline_misses =
                    metrics.publish_deadline_misses.saturating_add(1);
            } else {
                tokio::time::sleep(interval - enabled_elapsed).await;
            }
        }
        if id % 64 == 0 {
            let health = hub.health();
            metrics.max_ingress_depth = metrics.max_ingress_depth.max(health.ingress_depth);
            metrics.max_history_depth = metrics.max_history_depth.max(health.history_depth);
            metrics.max_durable_backlog = metrics
                .max_durable_backlog
                .max(health.durable_writer_backlog);
            tokio::task::yield_now().await;
        }
    }
    let drained = stress_wait_drained(&hub).await;
    let mut reconnect = hub.subscribe();
    let reconnect_event_id = config.events.saturating_add(1);
    hub.publish(stress_event(reconnect_event_id, true, config.telemetry_keys))
        .map_err(io::Error::other)?;
    let reconnect_received_live_event = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        reconnect.recv(),
    )
    .await
    .is_ok_and(|result| {
        result.is_ok_and(|event| event.event.event_id.0
            == format!("observatory-stress:{reconnect_event_id}"))
    });
    let _ = stress_wait_drained(&hub).await;
    let stalled_client_lagged = matches!(
        stalled_client.try_recv(),
        Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_))
    );
    let health = hub.health();
    metrics.max_ingress_depth = metrics.max_ingress_depth.max(health.ingress_depth);
    metrics.max_history_depth = metrics.max_history_depth.max(health.history_depth);
    metrics.max_durable_backlog = metrics
        .max_durable_backlog
        .max(health.durable_writer_backlog);
    metrics.final_sequence = health.newest_sequence.unwrap_or(0);
    metrics.durable_sequence = health.last_durable_sequence.unwrap_or(0);
    metrics.durable_write_failures = health.durable_write_failures;
    metrics.durable_bytes = (0..=4)
        .map(|index| {
            let path = if index == 0 {
                durable_path.clone()
            } else {
                durable_segment_path(&durable_path, index)
            };
            fs::metadata(path).map_or(0, |metadata| metadata.len())
        })
        .sum();
    metrics.telemetry_coalesced = metrics
        .telemetry_coalesced
        .max(health.ingress_coalesced + health.history_coalesced);
    metrics.telemetry_dropped = metrics
        .telemetry_dropped
        .max(health.ingress_dropped_telemetry);
    let old_cursor = hub
        .query(&BrainEventQuery {
            after_sequence: Some(0),
            ..Default::default()
        })
        .map_err(io::Error::other)?;
    metrics.declared_cursor_gaps = old_cursor
        .records
        .iter()
        .filter(|record| matches!(record, BrainEventStreamRecord::Gap { .. }))
        .count();
    metrics.serialized_history_bytes = serde_json::to_vec(&old_cursor)
        .map_err(io::Error::other)?
        .len() as u64;
    let original = stress_query_all(&hub).map_err(io::Error::other)?;
    let original_critical = original
        .iter()
        .filter(|event| matches!(event.event.loss_policy, LossPolicy::LossIntolerant))
        .map(|event| (event.sequence, event.event.event_id.clone()))
        .collect::<Vec<_>>();
    metrics.clock_reset_preserved = original.windows(2).any(|pair| {
        pair[0].event.times.occurred.clock_epoch.as_deref() == Some("sensor-epoch-a")
            && pair[1].event.times.occurred.clock_epoch.as_deref() == Some("sensor-epoch-b")
            && pair[1].event.times.occurred.t_ms < pair[0].event.times.occurred.t_ms
            && pair[1].sequence > pair[0].sequence
    });
    metrics.stalled_client_lagged = stalled_client_lagged;
    metrics.reconnect_received_live_event = reconnect_received_live_event;
    metrics.construct_only = stress_latency_summary(&mut baseline_samples);
    metrics.construct_and_publish = stress_latency_summary(&mut enabled_samples);
    metrics.added_publish_p99_us = (metrics.construct_and_publish.p99_us
        - metrics.construct_only.p99_us)
        .max(0.0);
    metrics.rss_after_bytes = stress_rss_bytes();
    metrics.rss_growth_bytes = metrics
        .rss_before_bytes
        .zip(metrics.rss_after_bytes)
        .map(|(before, after)| after.saturating_sub(before));
    hub.shutdown().await;

    let replay = BrainEventHub::new_with_durability(hub_config, durability)?;
    let recovered = stress_query_all(&replay).map_err(io::Error::other)?;
    let recovered_critical = recovered
        .iter()
        .filter(|event| matches!(event.event.loss_policy, LossPolicy::LossIntolerant))
        .map(|event| (event.sequence, event.event.event_id.clone()))
        .collect::<Vec<_>>();
    metrics.recovered_critical_events = recovered_critical.len();
    metrics.replay_order_matches = original_critical == recovered_critical;
    replay.shutdown().await;

    let mut failure_durability = BrainEventDurabilityConfig::new(
        artifact_directory.join("injected-failure.jsonl"),
    );
    failure_durability.sync_data = false;
    failure_durability.injected_failure_after_records = Some(0);
    let failure_hub = BrainEventHub::new_with_durability(hub_config, failure_durability)?;
    failure_hub.start();
    for id in 0..32 {
        let _ = failure_hub.publish(stress_event(
            10_000_000 + id,
            true,
            config.telemetry_keys,
        ));
    }
    let _ = stress_wait_drained(&failure_hub).await;
    metrics.injected_writer_failures = failure_hub.health().durable_write_failures;
    failure_hub.shutdown().await;

    let mut checks = BTreeMap::new();
    checks.insert("drained_within_15s".into(), drained);
    checks.insert(
        "ingress_memory_bounded".into(),
        metrics.max_ingress_depth <= config.ingress_capacity,
    );
    checks.insert(
        "history_memory_bounded".into(),
        metrics.max_history_depth <= config.history_capacity,
    );
    checks.insert(
        "critical_ingress_loss_free".into(),
        metrics.critical_rejections == 0,
    );
    checks.insert(
        "durable_writer_loss_free".into(),
        metrics.durable_write_failures == 0,
    );
    checks.insert("replay_order_matches".into(), metrics.replay_order_matches);
    checks.insert("clock_reset_preserved".into(), metrics.clock_reset_preserved);
    checks.insert("stalled_client_isolated".into(), metrics.stalled_client_lagged);
    checks.insert(
        "reconnect_receives_live_event".into(),
        metrics.reconnect_received_live_event,
    );
    checks.insert(
        "writer_failure_injection_visible".into(),
        metrics.injected_writer_failures > 0,
    );
    checks.insert(
        "publish_p99_within_budget".into(),
        metrics.added_publish_p99_us <= config.max_publish_p99_us,
    );
    checks.insert(
        "publish_deadlines_met".into(),
        metrics.publish_deadline_misses == 0,
    );
    checks.insert(
        "rss_growth_within_budget".into(),
        metrics
            .rss_growth_bytes
            .is_none_or(|growth| growth <= config.max_rss_growth_bytes),
    );
    let passed = checks.values().all(|passed| *passed);
    Ok(ObservatoryStressReport {
        schema_version: 2,
        generated_at_ms: wall_now_ms(),
        host_arch: std::env::consts::ARCH.into(),
        config,
        artifact_directory,
        metrics,
        checks,
        passed,
        physical_pi_soak_performed: false,
        physical_pi_soak_command: "just observatory-stress profile=pi5-soak events=360000 output=data/reports/observatory-stress/pi5-soak.json".into(),
        coverage: vec![
            "mixed coalescible projections and loss-intolerant bursts".into(),
            "stalled and reconnected broadcast clients".into(),
            "old cursor pagination and declared gaps".into(),
            "durable restart replay and serialization".into(),
            "sensor clock epoch reset without causal reordering".into(),
            "writer failure injection outside ingestion".into(),
            "equivalent event construction versus construction plus publication latency".into(),
            "bounded ingress history durable queue and RSS".into(),
        ],
    })
}
