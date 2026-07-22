fn component_health_event(
    id: &str,
    event_type: BrainEventType,
    observed_ms: u64,
    payload: serde_json::Value,
) -> BrainEvent {
    let mut event = graph_event(id, event_type, observed_ms.saturating_sub(2));
    event.producer.component = id.into();
    event.payload = BrainEventPayload::inline(payload);
    event
}

fn health_response(events: &[BrainEvent], at_ms: u64) -> ComponentHealthResponse {
    build_component_health(
        events,
        at_ms,
        BrainEventTransportHealth {
            running: true,
            ingress_capacity: 32,
            ..Default::default()
        },
        &LiveTrainingStatus::default(),
        None,
    )
}

fn health_row<'a>(response: &'a ComponentHealthResponse, id: &str) -> &'a ComponentHealthRow {
    response.rows.iter().find(|row| row.id == id).unwrap()
}

#[test]
fn stalled_required_provider_is_stale_and_links_to_its_event() {
    let event = component_health_event(
        "motherbrain.runtime",
        BrainEventType::ProviderState,
        100,
        serde_json::json!({
            "component_id": "motherbrain.runtime",
            "availability": "available",
            "health": "healthy",
            "heartbeat_age_ms": 50
        }),
    );

    let response = health_response(&[event], 3_000);
    let row = health_row(&response, "motherbrain.runtime");

    assert_eq!(row.availability, ComponentAvailability::Available);
    assert_eq!(row.health, ComponentHealthState::Stale);
    assert_eq!(row.event_id.as_deref(), Some("motherbrain.runtime"));
    assert!(response.alerts.iter().any(|alert| {
        alert.component_id == "motherbrain.runtime"
            && alert.event_id.as_deref() == Some("motherbrain.runtime")
    }));
}

#[test]
fn queue_pressure_is_saturated_without_becoming_unavailable() {
    let event = component_health_event(
        "cognition.queue",
        BrainEventType::QueueState,
        1_000,
        serde_json::json!({
            "component_id": "cognition.queue",
            "availability": "available",
            "health": "healthy",
            "queue_depth": 8,
            "queue_capacity": 10,
            "dropped": 2,
            "replaced": 4,
            "deadline_expired": 1
        }),
    );

    let response = health_response(&[event], 1_100);
    let row = health_row(&response, "cognition.queue");

    assert_eq!(row.availability, ComponentAvailability::Available);
    assert_eq!(row.occupancy, ComponentOccupancy::Saturated);
    assert_eq!(row.health, ComponentHealthState::Degraded);
    assert!(response.threshold_history.iter().any(|transition| {
        transition.component_id == "cognition.queue"
            && transition.metric == "queue_ratio"
            && transition.crossed
    }));
}

#[test]
fn capture_failure_preserves_writer_pressure_and_error_evidence() {
    let event = component_health_event(
        "capture.writer",
        BrainEventType::JobState,
        2_000,
        serde_json::json!({
            "component_id": "capture.writer",
            "availability": "available",
            "health": "failed",
            "occupancy": "saturated",
            "capture_bytes": 4096,
            "capture_streams": 3,
            "missing_intervals": 2,
            "writer_backlog": 19,
            "disk_free_bytes": 1024,
            "latest_error": "disk write failed"
        }),
    );

    let response = health_response(&[event], 2_100);
    let row = health_row(&response, "capture.writer");

    assert_eq!(row.health, ComponentHealthState::Failed);
    assert_eq!(row.metrics.writer_backlog, Some(19));
    assert_eq!(row.metrics.missing_intervals, Some(2));
    assert_eq!(row.latest_error.as_deref(), Some("disk write failed"));
    assert_eq!(
        response
            .rows
            .iter()
            .filter(|row| row.id == "capture.writer")
            .count(),
        1
    );
}

#[test]
fn reconnects_are_component_evidence_not_transport_drop_counters() {
    let event = component_health_event(
        "observatory.client",
        BrainEventType::ResourceState,
        3_000,
        serde_json::json!({
            "component_id": "observatory.client",
            "availability": "available",
            "health": "degraded",
            "reconnects": 3,
            "latest_error": "websocket reconnected"
        }),
    );

    let response = health_response(&[event], 3_100);

    assert_eq!(
        health_row(&response, "observatory.client")
            .metrics
            .reconnects,
        Some(3)
    );
    assert_eq!(
        health_row(&response, "observatory.transport")
            .metrics
            .dropped,
        Some(0)
    );
}

#[test]
fn thermal_pressure_and_reduced_watchdog_degrade_health_without_faking_authority() {
    let thermal = component_health_event(
        "brainstem",
        BrainEventType::ResourceState,
        4_000,
        serde_json::json!({
            "component_id": "brainstem",
            "availability": "available",
            "health": "healthy",
            "temperature_c": 82.0,
            "authority": "reflex_only"
        }),
    );
    let watchdog = component_health_event(
        "motherbrain.runtime",
        BrainEventType::ProviderState,
        4_000,
        serde_json::json!({
            "component_id": "motherbrain.runtime",
            "availability": "available",
            "health": "healthy",
            "reduced_watchdog": true,
            "authority": "runtime"
        }),
    );

    let response = health_response(&[thermal, watchdog], 4_100);

    assert_eq!(
        health_row(&response, "brainstem").health,
        ComponentHealthState::Degraded
    );
    assert_eq!(
        health_row(&response, "brainstem").authority.as_deref(),
        Some("reflex_only")
    );
    let runtime = health_row(&response, "motherbrain.runtime");
    assert_eq!(runtime.health, ComponentHealthState::Degraded);
    assert_eq!(runtime.metrics.reduced_watchdog, Some(true));
    assert!(runtime
        .latest_error
        .as_deref()
        .unwrap()
        .contains("watchdog"));
}

#[test]
fn missing_optional_components_do_not_raise_organism_health_alerts() {
    let response = health_response(&[], 5_000);

    assert!(response.rows.iter().any(|row| {
        row.id == "forebrain.workers"
            && row.optional
            && row.availability == ComponentAvailability::Missing
    }));
    assert!(!response
        .alerts
        .iter()
        .any(|alert| alert.component_id == "forebrain.workers"));
}

#[test]
fn observatory_health_ui_separates_truth_dimensions_and_links_alerts() {
    for marker in [
        "Brain / provider / resource health",
        "availability",
        "health / busy",
        "authority",
        "heartbeat / lease / age",
        "queue / loss / deadlines",
        "model / rollout",
        "capture / disk",
        "No recorded health threshold alert at this time.",
        "browser reconnects in this page session",
        "/api/observatory/component-health?at_ms=",
        "button.onclick=()=>selectEvent(alert.event_id)",
    ] {
        assert!(OBSERVATORY_PAGE.contains(marker), "missing {marker}");
    }
}
