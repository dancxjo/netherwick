fn frame_domain_brain_events(frame: &ExperienceFrame) -> Vec<BrainEvent> {
    let frame_id = frame.id.to_string();
    let mut events = Vec::new();
    events.extend(frame.sensations.iter().map(BrainEvent::from));
    events.extend(frame.impressions.iter().map(BrainEvent::from));
    events.extend(frame.experiences.iter().map(BrainEvent::from));
    let mut seen_ids = HashSet::new();
    events.retain(|event| seen_ids.insert(event.event_id.clone()));
    for event in &mut events {
        if event.references.frame_id.is_none() {
            event.references.frame_id = Some(frame_id.clone());
        }
        // These are canonical domain facts in a causal chain, not replaceable
        // current-state projections.
        event.loss_policy = LossPolicy::LossIntolerant;
    }
    events
}

fn forebrain_ingress_event(now: &Now, frame_id: Uuid) -> BrainEvent {
    let observations = now.objects.observations.len();
    let detections = now.objects.detections.len();
    let vectors = now.objects.vectors.len();
    let provider_reports_available = now.objects.vision_health.as_ref().is_some_and(|health| {
        !matches!(
            &health.state,
            VisionBackendState::Missing | VisionBackendState::Failed
        )
    });
    let available = observations > 0 || detections > 0 || vectors > 0 || provider_reports_available;
    let mut event = BrainEvent::historical(
        BrainEventId::from_domain("forebrain-response", frame_id),
        BrainEventType::Interpretation,
        ProducerIdentity::new(Brain::Forebrain, "vision.exchange"),
        EventTimes::observed(now.t_ms, now.t_ms),
    );
    event.kind = if available {
        "brain.exchange.fore_to_mother.response"
    } else {
        "brain.exchange.fore_to_mother.unavailable"
    }
    .to_string();
    event.references.frame_id = Some(frame_id.to_string());
    event.disposition = if available {
        EventDisposition::Accepted
    } else {
        EventDisposition::Unavailable
    };
    event.calibration_epochs = now
        .objects
        .detections
        .iter()
        .filter_map(|detection| {
            detection
                .calibration_epoch
                .map(|epoch| format!("vision:{epoch}"))
        })
        .collect();
    event.calibration_epochs.sort();
    event.calibration_epochs.dedup();
    event.payload = BrainEventPayload::inline(serde_json::json!({
        "observations": observations,
        "detections": detections,
        "vectors": vectors,
        "vision_health": now.objects.vision_health,
    }));
    event.authority = AuthoritySignificance::Advisory;
    event
}

fn higher_brain_request_event(frame_id: Uuid, snapshot_ref: &str, t_ms: TimeMs) -> BrainEvent {
    let mut event = BrainEvent::historical(
        BrainEventId::from_domain("higher-brain-request", frame_id),
        BrainEventType::Proposal,
        ProducerIdentity::new(Brain::Motherbrain, "cognition.scheduler"),
        EventTimes::observed(t_ms, t_ms),
    );
    event.kind = "brain.exchange.mother_to_higher.request".to_string();
    event.references.frame_id = Some(frame_id.to_string());
    event.payload = BrainEventPayload::inline(serde_json::json!({
        "input_snapshot_ref": snapshot_ref,
        "deadline_ms": t_ms.saturating_add(COGNITION_DEADLINE_MS),
    }));
    event.authority = AuthoritySignificance::Advisory;
    event
}

fn higher_brain_response_event(
    request_event_id: &BrainEventId,
    snapshot_ref: &str,
    requested_at_ms: TimeMs,
    observed_at_ms: TimeMs,
    disposition: EventDisposition,
    payload: serde_json::Value,
) -> BrainEvent {
    let mut event = BrainEvent::historical(
        BrainEventId::new(),
        BrainEventType::Interpretation,
        ProducerIdentity::new(Brain::HigherBrain, "llm.provider"),
        EventTimes::observed(requested_at_ms, observed_at_ms),
    );
    event.kind = match disposition {
        EventDisposition::Accepted => "brain.exchange.higher_to_mother.response",
        EventDisposition::Expired => "brain.exchange.higher_to_mother.expired",
        EventDisposition::Rejected => "brain.exchange.higher_to_mother.cancelled",
        _ => "brain.exchange.higher_to_mother.unavailable",
    }
    .to_string();
    event.links.parents.push(TypedEventRef::new(
        request_event_id.clone(),
        BrainEventType::Proposal,
    ));
    event.references.frame_id = Some(snapshot_ref.to_string());
    event.disposition = disposition;
    event.payload = bounded_runtime_payload(
        &event.event_id,
        payload,
        format!("frame://{snapshot_ref}/higher-cognition"),
    );
    event.authority = AuthoritySignificance::Advisory;
    event
}

fn bounded_runtime_payload(
    event_id: &BrainEventId,
    payload: serde_json::Value,
    locator: String,
) -> BrainEventPayload {
    if serde_json::to_vec(&payload)
        .map(|bytes| bytes.len() <= MAX_INLINE_BRAIN_EVENT_PAYLOAD_BYTES)
        .unwrap_or(false)
    {
        BrainEventPayload::inline(payload)
    } else {
        BrainEventPayload::referenced(
            format!("payload:{event_id}"),
            locator,
            "application/json",
        )
    }
}

fn higher_brain_unavailable_event(
    frame_id: Uuid,
    t_ms: TimeMs,
    reason: Option<&str>,
) -> BrainEvent {
    let mut event = BrainEvent::historical(
        BrainEventId::from_domain("higher-brain-unavailable", frame_id),
        BrainEventType::ProviderState,
        ProducerIdentity::new(Brain::HigherBrain, "llm.provider"),
        EventTimes::observed(t_ms, t_ms),
    );
    event.kind = "brain.exchange.higher.unavailable".to_string();
    event.references.frame_id = Some(frame_id.to_string());
    event.disposition = EventDisposition::Unavailable;
    event.payload = BrainEventPayload::inline(serde_json::json!({
        "reason": reason.unwrap_or("provider did not declare availability"),
    }));
    event.authority = AuthoritySignificance::Advisory;
    event.loss_policy = LossPolicy::Coalescible {
        key: "brain.exchange.higher.unavailable".to_string(),
    };
    event.record_kind = BrainEventRecordKind::StateProjection;
    event
}

fn reign_input_boundary_event(input: &ReignInput, frame_id: Uuid, t_ms: TimeMs) -> BrainEvent {
    let mut event = BrainEvent::from_reign_input(input, t_ms);
    event.event_id = BrainEventId::from_domain(
        "reign-input-considered",
        format!("{}:{frame_id}", input.id),
    );
    event.kind = "reign.input.considered".to_string();
    event.references.frame_id = Some(frame_id.to_string());
    event
}

fn reign_outcome_boundary_event(
    input: &ReignInput,
    outcome: &ReignOutcome,
    frame_id: Uuid,
    t_ms: TimeMs,
) -> BrainEvent {
    let parent_id = BrainEventId::from_domain(
        "reign-input-considered",
        format!("{}:{frame_id}", input.id),
    );
    let mut event = BrainEvent::from_reign_outcome(
        BrainEventId::from_domain(
            "reign-outcome",
            format!("{}:{frame_id}", outcome.input_id),
        ),
        outcome,
        t_ms,
    );
    event.references.frame_id = Some(frame_id.to_string());
    event.links.parents.clear();
    event
        .links
        .parents
        .push(TypedEventRef::new(parent_id, BrainEventType::Proposal));
    event
}

fn conductor_proposal_event(
    frame_id: Uuid,
    t_ms: TimeMs,
    chosen_action: &ActionPrimitive,
    goal_id: Option<&str>,
    parent_experience_id: Option<Uuid>,
    reign_input_event_id: Option<&BrainEventId>,
) -> BrainEvent {
    let mut event = BrainEvent::historical(
        BrainEventId::from_domain("conductor-proposal", frame_id),
        BrainEventType::Proposal,
        ProducerIdentity::new(Brain::Motherbrain, "conductor.selection"),
        EventTimes::observed(t_ms, t_ms),
    );
    event.kind = "conductor.proposal".to_string();
    event.references.frame_id = Some(frame_id.to_string());
    event.references.command_ids.push(frame_id.to_string());
    if let Some(goal_id) = goal_id {
        event.references.goal_ids.push(goal_id.to_string());
    }
    if let Some(experience_id) = parent_experience_id {
        event.links.parents.push(TypedEventRef::new(
            BrainEventId::experience(experience_id),
            BrainEventType::BeliefUpdate,
        ));
    }
    if let Some(reign_input_event_id) = reign_input_event_id {
        event.links.parents.push(TypedEventRef::new(
            reign_input_event_id.clone(),
            BrainEventType::Proposal,
        ));
    }
    event.payload = BrainEventPayload::inline(serde_json::json!({
        "chosen_action": chosen_action,
    }));
    event.authority = AuthoritySignificance::Proposal;
    event
}

fn safety_boundary_event(
    frame_id: Uuid,
    t_ms: TimeMs,
    decision: &pete_autonomic::SafetyDecision,
) -> BrainEvent {
    let proposal = TypedEventRef::new(
        BrainEventId::from_domain("conductor-proposal", frame_id),
        BrainEventType::Proposal,
    );
    let mut event = BrainEvent::from_safety_decision(
        BrainEventId::from_domain("safety-decision", frame_id),
        decision,
        format!("frame:{frame_id}"),
        proposal,
        t_ms,
    );
    event.references.snapshot_id = None;
    event.references.frame_id = Some(frame_id.to_string());
    event.references.command_ids.push(frame_id.to_string());
    event
}

fn accepted_runtime_command_event(
    frame_id: Uuid,
    t_ms: TimeMs,
    action: &ActionPrimitive,
    decision: &pete_autonomic::SafetyDecision,
) -> BrainEvent {
    let mut event = BrainEvent::historical(
        BrainEventId::from_domain("actuator-command", frame_id),
        BrainEventType::Command,
        ProducerIdentity::new(Brain::Motherbrain, "actuator.command"),
        EventTimes::observed(t_ms, t_ms),
    );
    event.kind = "actuator.command.accepted_by_runtime".to_string();
    event.references.frame_id = Some(frame_id.to_string());
    event.references.command_ids.push(frame_id.to_string());
    event.links.parents.push(TypedEventRef::new(
        BrainEventId::from_domain("safety-decision", frame_id),
        BrainEventType::GateDecision,
    ));
    event.disposition = EventDisposition::Accepted;
    event.payload = BrainEventPayload::inline(serde_json::json!({
        "action": action,
        "final_motor": decision.command,
        "safety_vetoed": decision.vetoed,
        "safety_reason": decision.reason,
    }));
    event.authority = AuthoritySignificance::Command;
    event
}

/// Record what happened after a runtime-accepted command crossed the actual
/// simulator or brainstem boundary. This must be called only after the result
/// payload is final.
pub fn append_actuator_outcome(
    tick: &mut RuntimeTick,
    producer_brain: Brain,
    component: &str,
    observed_at_ms: TimeMs,
    payload: serde_json::Value,
    disposition: EventDisposition,
) {
    let frame_id = tick.frame.id;
    let mut event = BrainEvent::historical(
        BrainEventId::from_domain("actuator-outcome", frame_id),
        BrainEventType::Outcome,
        ProducerIdentity::new(producer_brain, component),
        EventTimes::observed(tick.frame.t_ms, observed_at_ms),
    );
    event.kind = "actuator.outcome".to_string();
    event.references.frame_id = Some(frame_id.to_string());
    event.references.command_ids.push(frame_id.to_string());
    event.links.parents.push(TypedEventRef::new(
        BrainEventId::from_domain("actuator-command", frame_id),
        BrainEventType::Command,
    ));
    event.disposition = disposition;
    event.payload = bounded_runtime_payload(
        &event.event_id,
        payload,
        format!("frame://{frame_id}/actuator-outcome"),
    );
    event.authority = AuthoritySignificance::Outcome;
    tick.brain_events.push(event);
}

fn append_real_robot_outcome(
    tick: &mut RuntimeTick,
    snapshot: &WorldSnapshot,
    disposition: EventDisposition,
) {
    append_actuator_outcome(
        tick,
        Brain::Brainstem,
        "cockpit.brainstem",
        snapshot.body.last_update_ms,
        snapshot
            .action_debug
            .clone()
            .unwrap_or_else(|| serde_json::json!({"outcome": "not reported"})),
        disposition,
    );
}

fn no_higher_brain_motion_authority(events: &[BrainEvent]) -> bool {
    events.iter().all(|event| {
        !matches!(event.producer.brain, Brain::Forebrain | Brain::HigherBrain)
            || matches!(
                event.authority,
                AuthoritySignificance::None | AuthoritySignificance::Advisory
            )
    })
}
