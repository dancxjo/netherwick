fn authority_event(
    id: &str,
    event_type: BrainEventType,
    t_ms: u64,
    disposition: EventDisposition,
) -> BrainEvent {
    let mut event = graph_event(id, event_type, t_ms);
    event.disposition = disposition;
    event.authority = match event_type {
        BrainEventType::Proposal => AuthoritySignificance::Proposal,
        BrainEventType::GateDecision => AuthoritySignificance::Gate,
        BrainEventType::Command => AuthoritySignificance::Command,
        BrainEventType::Outcome => AuthoritySignificance::Outcome,
        _ => AuthoritySignificance::None,
    };
    event.references.command_ids.push("cmd-7".into());
    event
}

fn authority_response(events: &[BrainEvent]) -> AuthorityFlowResponse {
    build_authority_flow(events, 1_000, 1_000, ObservableAuthorityState::default())
}

#[test]
fn normal_selection_preserves_request_gate_command_ack_and_outcome_identity() {
    let mut proposal = authority_event("proposal", BrainEventType::Proposal, 10, EventDisposition::Unknown);
    proposal.quality.confidence = Some(0.8);
    proposal.payload = BrainEventPayload::inline(serde_json::json!({"score": 0.8}));
    let gate = authority_event("gate", BrainEventType::GateDecision, 11, EventDisposition::Accepted);
    let command = authority_event("command", BrainEventType::Command, 12, EventDisposition::Accepted);
    let mut ack = authority_event("ack", BrainEventType::Outcome, 13, EventDisposition::Accepted);
    ack.kind = "brainstem.ack".into();
    let outcome = authority_event("outcome", BrainEventType::Outcome, 14, EventDisposition::Accepted);

    let response = authority_response(&[proposal, gate, command, ack, outcome]);

    assert_eq!(response.events.len(), 5);
    assert!(response.events.iter().all(|event| event.command_ids == ["cmd-7"]));
    assert_eq!(response.events[0].score, Some(0.8));
    assert_eq!(response.events.last().unwrap().status, AuthorityFlowStatus::Completed);
}

#[test]
fn competing_goals_and_operator_override_keep_every_proposal_and_lane() {
    let mut explore = authority_event("explore", BrainEventType::Proposal, 10, EventDisposition::Rejected);
    explore.references.goal_ids.push("explore".into());
    let mut charge = authority_event("charge", BrainEventType::Proposal, 10, EventDisposition::Accepted);
    charge.references.goal_ids.push("charge".into());
    let mut operator = authority_event("operator", BrainEventType::Proposal, 11, EventDisposition::Accepted);
    operator.producer.brain = Brain::Human;
    operator.kind = "reign.direct".into();
    operator.payload = BrainEventPayload::inline(serde_json::json!({"mode": "direct", "priority": 1.0}));

    let response = authority_response(&[explore, charge, operator]);

    assert_eq!(response.competing_proposals, 3);
    assert!(response.events.iter().any(|event| event.goal_ids == ["explore"]));
    assert!(response.events.iter().any(|event| event.goal_ids == ["charge"]));
    assert!(response.events.iter().any(|event| event.lane == AuthorityLane::HumanDirect));
}

#[test]
fn bump_cliff_veto_and_reflex_preemption_are_distinct() {
    let mut veto = authority_event("veto", BrainEventType::GateDecision, 20, EventDisposition::Vetoed);
    veto.kind = "autonomic.safety".into();
    veto.payload = BrainEventPayload::inline(serde_json::json!({"reason": "bump and cliff active"}));
    let mut reflex = authority_event("reflex", BrainEventType::Command, 21, EventDisposition::Accepted);
    reflex.producer.brain = Brain::Brainstem;
    reflex.kind = "brainstem.reflex.preempt".into();
    reflex.payload = BrainEventPayload::inline(serde_json::json!({"reason": "preempted motherbrain command"}));

    let response = authority_response(&[veto, reflex]);

    assert_eq!(response.vetoes, 1);
    assert_eq!(response.preemptions, 1);
    assert_eq!(response.events[0].lane, AuthorityLane::AutonomicSafety);
    assert_eq!(response.events[1].lane, AuthorityLane::BrainstemReflex);
}

#[test]
fn lease_expiry_and_stop_failure_remain_failed_or_expired_not_completed() {
    let mut lease = authority_event("lease", BrainEventType::GateDecision, 30, EventDisposition::Expired);
    lease.kind = "possession.lease.expired".into();
    lease.payload = BrainEventPayload::inline(serde_json::json!({"reason": "control lease expired", "ttl_ms": 500}));
    let mut stop = authority_event("stop", BrainEventType::Outcome, 31, EventDisposition::Unavailable);
    stop.kind = "brainstem.stop.outcome".into();
    stop.payload = BrainEventPayload::inline(serde_json::json!({"reason": "STOP acknowledgement failed"}));

    let response = authority_response(&[lease, stop]);

    assert_eq!(response.events[0].stage, AuthorityFlowStage::PossessionLease);
    assert_eq!(response.events[0].status, AuthorityFlowStatus::Expired);
    assert_eq!(response.events[0].ttl_ms, Some(500));
    assert_eq!(response.events[1].status, AuthorityFlowStatus::Failed);
}

#[test]
fn advisory_and_shadow_outputs_are_never_labeled_as_runtime_authority() {
    let mut llm = authority_event("llm", BrainEventType::Proposal, 40, EventDisposition::Rejected);
    llm.kind = "llm.advisory".into();
    llm.authority = AuthoritySignificance::Advisory;
    let mut shadow = authority_event("shadow", BrainEventType::Proposal, 41, EventDisposition::Superseded);
    shadow.kind = "locomotion.learned.shadow".into();

    let response = authority_response(&[llm, shadow]);

    assert_eq!(response.advisory_events, 2);
    assert_eq!(response.events[0].lane, AuthorityLane::LlmAdvisory);
    assert_eq!(response.events[1].lane, AuthorityLane::LearnedShadow);
}

#[test]
fn observatory_authority_view_is_read_only_and_surfaces_unknowns() {
    for marker in [
        "Authority / safety flow",
        "Read-only. Advisory and shadow lanes have no implied execution authority.",
        "lease not observed",
        "STOP unknown",
        "heartbeat unknown",
        "/api/observatory/authority",
    ] {
        assert!(OBSERVATORY_PAGE.contains(marker), "missing {marker}");
    }
    assert!(!OBSERVATORY_PAGE.contains("/api/observatory/control"));
}
