const HEALTH_STALE_AFTER_MS: u64 = 2_000;
const QUEUE_PRESSURE_RATIO: f64 = 0.75;
const THERMAL_PRESSURE_C: f64 = 75.0;
const DISK_PRESSURE_BYTES: u64 = 1_073_741_824;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentAvailability {
    Available,
    Missing,
    Unavailable,
    #[default]
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentHealthState {
    Healthy,
    Degraded,
    Failed,
    Stale,
    #[default]
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentOccupancy {
    Idle,
    Busy,
    Saturated,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ComponentHealthMetrics {
    pub heartbeat_age_ms: Option<u64>,
    pub lease_age_ms: Option<u64>,
    pub tick_period_ms: Option<f64>,
    pub tick_budget_ms: Option<f64>,
    pub stage_timing_ms: BTreeMap<String, f64>,
    pub cpu_percent: Option<f64>,
    pub memory_bytes: Option<u64>,
    pub temperature_c: Option<f64>,
    pub queue_depth: Option<u64>,
    pub queue_capacity: Option<u64>,
    pub dropped: Option<u64>,
    pub replaced: Option<u64>,
    pub deadline_expired: Option<u64>,
    pub inference_p50_ms: Option<f64>,
    pub inference_p95_ms: Option<f64>,
    pub reconnects: Option<u64>,
    pub capture_bytes: Option<u64>,
    pub capture_streams: Option<u64>,
    pub missing_intervals: Option<u64>,
    pub writer_backlog: Option<u64>,
    pub disk_free_bytes: Option<u64>,
    pub reduced_watchdog: Option<bool>,
    pub ingress_dropped_telemetry: Option<u64>,
    pub ingress_rejected_critical: Option<u64>,
    pub history_expired: Option<u64>,
    pub history_coalesced: Option<u64>,
    pub client_lag_gaps: Option<u64>,
    pub durable_write_failures: Option<u64>,
    pub last_durable_sequence: Option<u64>,
    pub durability_gaps: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComponentHealthRow {
    pub id: String,
    pub group: String,
    pub brain: Brain,
    pub component: String,
    pub optional: bool,
    pub availability: ComponentAvailability,
    pub health: ComponentHealthState,
    pub occupancy: ComponentOccupancy,
    pub authority: Option<String>,
    pub age_ms: Option<u64>,
    pub metrics: ComponentHealthMetrics,
    pub artifacts: Vec<ArtifactIdentity>,
    pub candidate_state: Option<String>,
    pub rollback_state: Option<String>,
    pub latest_error: Option<String>,
    pub event_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HealthAlert {
    pub severity: String,
    pub component_id: String,
    pub message: String,
    pub event_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HealthThresholdTransition {
    pub event_id: String,
    pub component_id: String,
    pub metric: String,
    pub value: f64,
    pub threshold: f64,
    pub crossed: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ComponentHealthResponse {
    pub at_ms: u64,
    pub rows: Vec<ComponentHealthRow>,
    pub alerts: Vec<HealthAlert>,
    pub threshold_history: Vec<HealthThresholdTransition>,
    pub observatory_transport: BrainEventTransportHealth,
}

fn build_component_health(
    events: &[BrainEvent],
    at_ms: u64,
    transport: BrainEventTransportHealth,
    training: &LiveTrainingStatus,
    now: Option<&pete_now::Now>,
) -> ComponentHealthResponse {
    let mut rows = baseline_health_rows();
    let mut latest: BTreeMap<String, (usize, &BrainEvent)> = BTreeMap::new();
    for (sequence, event) in events.iter().enumerate().filter(|(_, event)| {
        event.times.observed.t_ms <= at_ms
            && matches!(
                event.event_type,
                BrainEventType::ProviderState
                    | BrainEventType::JobState
                    | BrainEventType::ResourceState
                    | BrainEventType::QueueState
            )
    }) {
        let id = health_component_id(event);
        let candidate = (
            event.times.observed.clock_epoch.as_deref(),
            event.times.observed.t_ms,
            sequence,
        );
        let replace = latest.get(&id).is_none_or(|(prior_sequence, prior)| {
            candidate
                > (
                    prior.times.observed.clock_epoch.as_deref(),
                    prior.times.observed.t_ms,
                    *prior_sequence,
                )
        });
        if replace {
            latest.insert(id, (sequence, event));
        }
    }
    for (id, (_, event)) in latest {
        if let Some(index) = rows.iter().position(|row| row.id == id) {
            let optional = rows[index].optional;
            let group = rows[index].group.clone();
            rows[index] = health_row_from_event(event, at_ms, optional, &group);
        } else {
            rows.push(health_row_from_event(
                event,
                at_ms,
                health_optional(event),
                health_group(event.event_type),
            ));
        }
    }
    if !rows.iter().any(|row| row.id == "capture.writer") {
        rows.push(capture_health_row(training));
    }
    rows.push(transport_health_row(&transport));
    if let Some(vision) = now.and_then(|now| now.objects.vision_health.as_ref()) {
        rows.push(vision_health_row(vision, at_ms));
    }
    rows.sort_by(|left, right| (&left.group, &left.id).cmp(&(&right.group, &right.id)));
    let mut alerts = Vec::new();
    for row in &rows {
        if row.optional && row.availability == ComponentAvailability::Missing {
            continue;
        }
        if matches!(
            row.health,
            ComponentHealthState::Failed
                | ComponentHealthState::Stale
                | ComponentHealthState::Degraded
        ) {
            alerts.push(HealthAlert {
                severity: if row.health == ComponentHealthState::Failed {
                    "critical"
                } else {
                    "warning"
                }
                .into(),
                component_id: row.id.clone(),
                message: row.latest_error.clone().unwrap_or_else(|| {
                    format!("{} is {:?}", row.component, row.health).to_lowercase()
                }),
                event_id: row.event_id.clone(),
            });
        }
    }
    let threshold_history = events
        .iter()
        .filter(|event| event.times.observed.t_ms <= at_ms)
        .flat_map(health_thresholds)
        .collect();
    ComponentHealthResponse {
        at_ms,
        rows,
        alerts,
        threshold_history,
        observatory_transport: transport,
    }
}

fn baseline_health_rows() -> Vec<ComponentHealthRow> {
    [
        ("brainstem", "brains", Brain::Brainstem, false),
        ("motherbrain.runtime", "brains", Brain::Motherbrain, false),
        ("forebrain.workers", "brains", Brain::Forebrain, true),
        (
            "higher_brain.providers",
            "providers",
            Brain::HigherBrain,
            true,
        ),
        ("sensors", "sensors", Brain::Motherbrain, false),
    ]
    .into_iter()
    .map(|(id, group, brain, optional)| ComponentHealthRow {
        id: id.into(),
        group: group.into(),
        brain,
        component: id.into(),
        optional,
        availability: if optional {
            ComponentAvailability::Missing
        } else {
            ComponentAvailability::Unknown
        },
        health: ComponentHealthState::Unknown,
        occupancy: ComponentOccupancy::Unknown,
        authority: None,
        age_ms: None,
        metrics: Default::default(),
        artifacts: Vec::new(),
        candidate_state: None,
        rollback_state: None,
        latest_error: None,
        event_id: None,
    })
    .collect()
}

fn health_row_from_event(
    event: &BrainEvent,
    at_ms: u64,
    optional: bool,
    group: &str,
) -> ComponentHealthRow {
    let payload = inline_object(event);
    let age_ms = at_ms.saturating_sub(event.times.observed.t_ms);
    let mut metrics = ComponentHealthMetrics {
        heartbeat_age_ms: health_u64(payload, "heartbeat_age_ms"),
        lease_age_ms: health_u64(payload, "lease_age_ms"),
        tick_period_ms: health_f64(payload, "tick_period_ms"),
        tick_budget_ms: health_f64(payload, "tick_budget_ms"),
        cpu_percent: health_f64(payload, "cpu_percent"),
        memory_bytes: health_u64(payload, "memory_bytes"),
        temperature_c: health_f64(payload, "temperature_c"),
        queue_depth: health_u64(payload, "queue_depth"),
        queue_capacity: health_u64(payload, "queue_capacity"),
        dropped: health_u64(payload, "dropped"),
        replaced: health_u64(payload, "replaced"),
        deadline_expired: health_u64(payload, "deadline_expired"),
        inference_p50_ms: health_f64(payload, "inference_p50_ms"),
        inference_p95_ms: health_f64(payload, "inference_p95_ms"),
        reconnects: health_u64(payload, "reconnects"),
        capture_bytes: health_u64(payload, "capture_bytes"),
        capture_streams: health_u64(payload, "capture_streams"),
        missing_intervals: health_u64(payload, "missing_intervals"),
        writer_backlog: health_u64(payload, "writer_backlog"),
        disk_free_bytes: health_u64(payload, "disk_free_bytes"),
        reduced_watchdog: payload
            .and_then(|payload| payload.get("reduced_watchdog"))
            .and_then(serde_json::Value::as_bool),
        ..Default::default()
    };
    if let Some(stages) = payload
        .and_then(|payload| payload.get("stage_timing_ms"))
        .and_then(serde_json::Value::as_object)
    {
        metrics.stage_timing_ms = stages
            .iter()
            .filter_map(|(key, value)| value.as_f64().map(|value| (key.clone(), value)))
            .collect();
    }
    let availability = match health_str(payload, "availability").as_deref() {
        Some("available") => ComponentAvailability::Available,
        Some("missing") => ComponentAvailability::Missing,
        Some("unavailable") => ComponentAvailability::Unavailable,
        _ => ComponentAvailability::Unknown,
    };
    let mut health = match health_str(payload, "health").as_deref() {
        Some("healthy") => ComponentHealthState::Healthy,
        Some("degraded") => ComponentHealthState::Degraded,
        Some("failed") => ComponentHealthState::Failed,
        Some("stale") => ComponentHealthState::Stale,
        _ => ComponentHealthState::Unknown,
    };
    let queue_saturated = matches!((metrics.queue_depth,metrics.queue_capacity),(Some(depth),Some(capacity)) if capacity>0 && depth as f64/capacity as f64>=QUEUE_PRESSURE_RATIO);
    let thermal = metrics
        .temperature_c
        .is_some_and(|temperature| temperature >= THERMAL_PRESSURE_C);
    let heartbeat_stale = metrics
        .heartbeat_age_ms
        .is_some_and(|heartbeat_age_ms| heartbeat_age_ms >= HEALTH_STALE_AFTER_MS);
    let disk_pressure = metrics
        .disk_free_bytes
        .is_some_and(|disk_free_bytes| disk_free_bytes <= DISK_PRESSURE_BYTES);
    let tick_over_budget = matches!(
        (metrics.tick_period_ms, metrics.tick_budget_ms),
        (Some(period), Some(budget)) if budget > 0.0 && period > budget
    );
    let reduced_watchdog = metrics.reduced_watchdog == Some(true);
    if (age_ms > HEALTH_STALE_AFTER_MS || heartbeat_stale)
        && availability == ComponentAvailability::Available
    {
        health = ComponentHealthState::Stale;
    } else if (queue_saturated || thermal || disk_pressure || tick_over_budget || reduced_watchdog)
        && health == ComponentHealthState::Healthy
    {
        health = ComponentHealthState::Degraded;
    }
    let occupancy = match health_str(payload, "occupancy").as_deref() {
        Some("idle") => ComponentOccupancy::Idle,
        Some("busy") => ComponentOccupancy::Busy,
        Some("saturated") => ComponentOccupancy::Saturated,
        _ if queue_saturated => ComponentOccupancy::Saturated,
        _ => ComponentOccupancy::Unknown,
    };
    ComponentHealthRow {
        id: health_component_id(event),
        group: group.into(),
        brain: event.producer.brain,
        component: event.producer.component.clone(),
        optional,
        availability,
        health,
        occupancy,
        authority: health_str(payload, "authority"),
        age_ms: Some(age_ms),
        metrics,
        artifacts: event.artifacts.clone(),
        candidate_state: health_str(payload, "candidate_state"),
        rollback_state: health_str(payload, "rollback_state"),
        latest_error: health_str(payload, "latest_error")
            .or_else(|| health_str(payload, "error"))
            .or_else(|| reduced_watchdog.then(|| "watchdog coverage is reduced".into())),
        event_id: Some(event.event_id.0.clone()),
    }
}

fn capture_health_row(training: &LiveTrainingStatus) -> ComponentHealthRow {
    ComponentHealthRow {
        id: "capture.writer".into(),
        group: "capture".into(),
        brain: Brain::Motherbrain,
        component: "capture.writer".into(),
        optional: true,
        availability: if training.ledger_path.is_some() {
            ComponentAvailability::Available
        } else {
            ComponentAvailability::Missing
        },
        health: if training.ledger_path.is_some() {
            ComponentHealthState::Healthy
        } else {
            ComponentHealthState::Unknown
        },
        occupancy: if training.ledger_path.is_some() {
            ComponentOccupancy::Busy
        } else {
            ComponentOccupancy::Idle
        },
        authority: Some("none".into()),
        metrics: ComponentHealthMetrics {
            capture_streams: Some(
                (training.frames_written > 0) as u64 + (training.transitions_written > 0) as u64,
            ),
            ..Default::default()
        },
        artifacts: training
            .models_loaded
            .iter()
            .map(|id| ArtifactIdentity {
                kind: ArtifactKind::Model,
                id: id.clone(),
                version: None,
                checksum: None,
            })
            .collect(),
        candidate_state: Some(training.action_selector_mode.clone()),
        rollback_state: None,
        latest_error: None,
        event_id: None,
        age_ms: None,
    }
}

fn transport_health_row(health: &BrainEventTransportHealth) -> ComponentHealthRow {
    let server_loss = health.ingress_dropped_telemetry + health.ingress_rejected_critical;
    let durability_failed = health.durable_write_failures > 0 || health.durability_gaps > 0;
    ComponentHealthRow {
        id: "observatory.transport".into(),
        group: "ui transport".into(),
        brain: Brain::Motherbrain,
        component: "observatory.transport".into(),
        optional: false,
        availability: if health.running {
            ComponentAvailability::Available
        } else {
            ComponentAvailability::Unavailable
        },
        health: if !health.running || health.ingress_rejected_critical > 0 || durability_failed {
            ComponentHealthState::Failed
        } else if health.ingress_dropped_telemetry > 0 {
            ComponentHealthState::Degraded
        } else {
            ComponentHealthState::Healthy
        },
        occupancy: if health.ingress_capacity > 0
            && health.ingress_depth * 4 >= health.ingress_capacity * 3
        {
            ComponentOccupancy::Saturated
        } else {
            ComponentOccupancy::Idle
        },
        authority: Some("none".into()),
        age_ms: None,
        metrics: ComponentHealthMetrics {
            queue_depth: Some(health.ingress_depth as u64),
            queue_capacity: Some(health.ingress_capacity as u64),
            dropped: Some(health.ingress_dropped_telemetry),
            replaced: Some(health.ingress_coalesced + health.history_coalesced),
            ingress_dropped_telemetry: Some(health.ingress_dropped_telemetry),
            ingress_rejected_critical: Some(health.ingress_rejected_critical),
            history_expired: Some(health.history_expired),
            history_coalesced: Some(health.history_coalesced),
            client_lag_gaps: Some(health.client_lag_gaps),
            writer_backlog: health
                .durability_enabled
                .then_some(health.durable_writer_backlog as u64),
            durable_write_failures: health
                .durability_enabled
                .then_some(health.durable_write_failures),
            last_durable_sequence: health.last_durable_sequence,
            durability_gaps: health
                .durability_enabled
                .then_some(health.durability_gaps),
            ..Default::default()
        },
        artifacts: Vec::new(),
        candidate_state: None,
        rollback_state: None,
        latest_error: if durability_failed {
            Some(format!(
                "durable observatory history has {} write failures and {} declared gaps; last durable sequence {:?}",
                health.durable_write_failures,
                health.durability_gaps,
                health.last_durable_sequence
            ))
        } else {
            (server_loss > 0).then(|| {
                format!(
                    "observatory server rejected {} critical and dropped {} telemetry events",
                    health.ingress_rejected_critical, health.ingress_dropped_telemetry
                )
            })
        },
        event_id: None,
    }
}

fn vision_health_row(vision: &pete_now::VisionPipelineHealth, _at_ms: u64) -> ComponentHealthRow {
    let health = match vision.state {
        pete_now::VisionBackendState::Ready => ComponentHealthState::Healthy,
        pete_now::VisionBackendState::Degraded => ComponentHealthState::Degraded,
        pete_now::VisionBackendState::Failed => ComponentHealthState::Failed,
        pete_now::VisionBackendState::Missing => ComponentHealthState::Unknown,
    };
    ComponentHealthRow {
        id: "vision.pipeline".into(),
        group: "providers".into(),
        brain: Brain::Forebrain,
        component: "vision.pipeline".into(),
        optional: true,
        availability: if matches!(vision.state, pete_now::VisionBackendState::Missing) {
            ComponentAvailability::Missing
        } else {
            ComponentAvailability::Available
        },
        health,
        occupancy: if vision.queue_depth >= vision.queue_capacity && vision.queue_capacity > 0 {
            ComponentOccupancy::Saturated
        } else if vision.queue_depth > 0 {
            ComponentOccupancy::Busy
        } else {
            ComponentOccupancy::Idle
        },
        authority: Some("evidence_only".into()),
        age_ms: None,
        metrics: ComponentHealthMetrics {
            queue_depth: Some(vision.queue_depth as u64),
            queue_capacity: Some(vision.queue_capacity as u64),
            dropped: Some(vision.dropped_frames),
            replaced: Some(vision.replaced_frames),
            deadline_expired: Some(vision.expired_frames),
            inference_p50_ms: vision.p50_inference_ms.map(|value| value as f64),
            inference_p95_ms: vision.p95_inference_ms.map(|value| value as f64),
            ..Default::default()
        },
        artifacts: vec![ArtifactIdentity {
            kind: ArtifactKind::Model,
            id: vision.backend.model_id.clone(),
            version: Some(vision.backend.version.clone()),
            checksum: vision.backend.checksum.clone(),
        }],
        candidate_state: None,
        rollback_state: None,
        latest_error: vision.latest_error.clone(),
        event_id: None,
    }
}

fn health_thresholds(event: &BrainEvent) -> Vec<HealthThresholdTransition> {
    let payload = inline_object(event);
    let id = health_component_id(event);
    let mut out = Vec::new();
    for (metric, value, threshold) in [
        (
            "queue_ratio",
            match (
                health_u64(payload, "queue_depth"),
                health_u64(payload, "queue_capacity"),
            ) {
                (Some(d), Some(c)) if c > 0 => Some(d as f64 / c as f64),
                _ => None,
            },
            QUEUE_PRESSURE_RATIO,
        ),
        (
            "temperature_c",
            health_f64(payload, "temperature_c"),
            THERMAL_PRESSURE_C,
        ),
        (
            "heartbeat_age_ms",
            health_u64(payload, "heartbeat_age_ms").map(|v| v as f64),
            HEALTH_STALE_AFTER_MS as f64,
        ),
        (
            "disk_free_bytes",
            health_u64(payload, "disk_free_bytes").map(|v| v as f64),
            DISK_PRESSURE_BYTES as f64,
        ),
        (
            "tick_budget_ratio",
            match (
                health_f64(payload, "tick_period_ms"),
                health_f64(payload, "tick_budget_ms"),
            ) {
                (Some(period), Some(budget)) if budget > 0.0 => Some(period / budget),
                _ => None,
            },
            1.0,
        ),
    ] {
        if let Some(value) = value {
            out.push(HealthThresholdTransition {
                event_id: event.event_id.0.clone(),
                component_id: id.clone(),
                metric: metric.into(),
                value,
                threshold,
                crossed: if metric == "disk_free_bytes" {
                    value <= threshold
                } else {
                    value >= threshold
                },
            });
        }
    }
    out
}

fn health_component_id(event: &BrainEvent) -> String {
    inline_object(event)
        .and_then(|p| p.get("component_id"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or(&event.producer.component)
        .to_string()
}
fn health_group(kind: BrainEventType) -> &'static str {
    match kind {
        BrainEventType::ProviderState => "providers",
        BrainEventType::JobState => "jobs",
        BrainEventType::ResourceState => "resources",
        BrainEventType::QueueState => "queues",
        _ => "components",
    }
}
fn health_optional(event: &BrainEvent) -> bool {
    inline_object(event)
        .and_then(|p| p.get("optional"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}
fn health_str(
    payload: Option<&serde_json::Map<String, serde_json::Value>>,
    key: &str,
) -> Option<String> {
    payload?.get(key)?.as_str().map(str::to_string)
}
fn health_u64(
    payload: Option<&serde_json::Map<String, serde_json::Value>>,
    key: &str,
) -> Option<u64> {
    payload?.get(key)?.as_u64()
}
fn health_f64(
    payload: Option<&serde_json::Map<String, serde_json::Value>>,
    key: &str,
) -> Option<f64> {
    payload?.get(key)?.as_f64()
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
struct ComponentHealthQuery {
    at_ms: Option<u64>,
}

async fn get_observatory_component_health(
    State(state): State<LiveViewState>,
    Query(query): Query<ComponentHealthQuery>,
) -> Result<Json<ComponentHealthResponse>, ObservatoryHttpError> {
    let history = state
        .observatory()
        .query_async(BrainEventQuery {
            observed_to_ms: query.at_ms,
            limit: Some(MAX_OBSERVATORY_QUERY_LIMIT),
            ..Default::default()
        })
        .await
        .map_err(|error| ObservatoryHttpError::bad_request(error.to_string()))?;
    let events: Vec<BrainEvent> = history
        .records
        .into_iter()
        .map(|record| match record {
            BrainEventStreamRecord::Event { envelope } => envelope.event,
            BrainEventStreamRecord::Gap { gap } => gap.event,
        })
        .collect();
    let at_ms = query.at_ms.unwrap_or_else(|| {
        events
            .last()
            .map(|event| event.times.observed.t_ms)
            .unwrap_or_else(wall_now_ms)
    });
    let now = query
        .at_ms
        .and_then(|at_ms| {
            state
                .observatory_now_at_or_before(at_ms)
                .map(|selection| selection.selected.now)
        })
        .or_else(|| {
            state
                .latest()
                .map(|snapshot| snapshot.to_now(snapshot.body.last_update_ms))
        });
    Ok(Json(build_component_health(
        &events,
        at_ms,
        state.observatory().health(),
        &state.training_status(),
        now.as_ref(),
    )))
}
