fn graph_event(id: &str, event_type: BrainEventType, t_ms: u64) -> BrainEvent {
    let mut event = BrainEvent::historical(
        BrainEventId(id.to_string()),
        event_type,
        ProducerIdentity::new(Brain::Motherbrain, "graph.test"),
        EventTimes::observed(t_ms, t_ms + 2),
    );
    event.kind = format!("graph.{id}");
    event
}

fn graph_query(max_nodes: usize) -> ProvenanceGraphQuery {
    ProvenanceGraphQuery {
        mode: ProvenancePathMode::Full,
        max_nodes: Some(max_nodes),
        include_artifacts: true,
        include_assets: true,
    }
}

#[test]
fn provenance_cycles_are_deduplicated_and_bounded() {
    let mut a = graph_event("a", BrainEventType::BeliefUpdate, 10);
    let mut b = graph_event("b", BrainEventType::Interpretation, 9);
    a.links.parents.push(pete_events::TypedEventRef::new(
        BrainEventId("b".into()),
        BrainEventType::Interpretation,
    ));
    b.links.parents.push(pete_events::TypedEventRef::new(
        BrainEventId("a".into()),
        BrainEventType::BeliefUpdate,
    ));

    let graph = build_provenance_graph(&[a, b], "a", &graph_query(2)).unwrap();

    assert_eq!(graph.nodes.len(), 2);
    assert_eq!(graph.nodes.iter().filter(|node| node.id == "a").count(), 1);
    assert_eq!(graph.nodes.iter().filter(|node| node.id == "b").count(), 1);
}

#[test]
fn provenance_missing_and_contradicting_references_remain_visible() {
    let evidence = graph_event("evidence", BrainEventType::Evidence, 5);
    let mut belief = graph_event("belief", BrainEventType::BeliefUpdate, 10);
    belief.links.supports.push(pete_events::TypedEventRef::new(
        BrainEventId("evidence".into()),
        BrainEventType::Evidence,
    ));
    belief
        .links
        .contradicts
        .push(pete_events::TypedEventRef::new(
            BrainEventId("missing-evidence".into()),
            BrainEventType::Evidence,
        ));

    let graph = build_provenance_graph(&[belief, evidence], "belief", &graph_query(20)).unwrap();

    assert_eq!(graph.missing_references, 1);
    assert!(graph.nodes.iter().any(|node| {
        node.id == "missing-evidence" && node.node_kind == ProvenanceNodeKind::MissingReference
    }));
    assert!(graph.edges.iter().any(|edge| {
        edge.to == "missing-evidence" && edge.relation == ProvenanceEdgeKind::Contradicts
    }));
}

#[test]
fn expired_evidence_and_recorded_quality_are_not_reinterpreted() {
    let mut evidence = graph_event("old-range", BrainEventType::Evidence, 1);
    evidence.disposition = EventDisposition::Expired;
    evidence.quality.trust = TrustState::Conditional;
    evidence.quality.confidence = Some(0.42);
    evidence.quality.freshness.state = pete_events::FreshnessState::Expired;

    let graph = build_provenance_graph(&[evidence], "old-range", &graph_query(8)).unwrap();
    let node = &graph.nodes[0];

    assert_eq!(node.disposition, Some(EventDisposition::Expired));
    assert_eq!(node.trust, Some(TrustState::Conditional));
    assert_eq!(node.confidence, Some(0.42));
    assert_eq!(
        node.freshness.as_ref().map(|freshness| freshness.state),
        Some(pete_events::FreshnessState::Expired)
    );
}

#[test]
fn large_high_degree_graphs_report_omitted_nodes() {
    let evidence: Vec<BrainEvent> = (0..300)
        .map(|index| graph_event(&format!("e{index}"), BrainEventType::Evidence, index))
        .collect();
    let mut center = graph_event("center", BrainEventType::BeliefUpdate, 500);
    center.links.supports = evidence
        .iter()
        .map(|event| {
            pete_events::TypedEventRef::new(event.event_id.clone(), BrainEventType::Evidence)
        })
        .collect();
    let mut events = vec![center];
    events.extend(evidence);

    let graph = build_provenance_graph(&events, "center", &graph_query(25)).unwrap();

    assert_eq!(graph.nodes.len(), 25);
    assert!(graph.truncated);
    assert!(graph.omitted_nodes > 0);
    assert!(graph.edges.len() <= 2 * (graph.nodes.len() - 1));
}

#[test]
fn graph_includes_model_calibration_asset_and_capture_gap_nodes_on_demand() {
    let mut center = graph_event("center", BrainEventType::Interpretation, 20);
    center.artifacts.push(pete_events::ArtifactIdentity {
        kind: ArtifactKind::Model,
        id: "vision-v5".into(),
        ..Default::default()
    });
    center.calibration_epochs.push("camera-e3".into());
    center.payload = BrainEventPayload::referenced("rgb-7", "capture://rgb/7", "image/png");
    let mut gap = graph_event("gap", BrainEventType::TransportGap, 20);
    gap.kind = "transport.gap".into();

    let graph = build_provenance_graph(&[center, gap], "center", &graph_query(20)).unwrap();

    assert!(graph.nodes.iter().any(|node| node.node_kind == ProvenanceNodeKind::ModelArtifact));
    assert!(graph.nodes.iter().any(|node| node.node_kind == ProvenanceNodeKind::CalibrationEpoch));
    assert!(graph.nodes.iter().any(|node| node.node_kind == ProvenanceNodeKind::RawAsset));
    assert_eq!(graph.capture_gaps, 1);
}
