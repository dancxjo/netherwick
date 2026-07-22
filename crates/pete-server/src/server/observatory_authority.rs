const DEFAULT_AUTHORITY_WINDOW_MS: u64 = 10_000;
const MAX_AUTHORITY_EVENTS: usize = 250;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthorityFlowStage {
    DriveOrGoal,
    GoalEvaluation,
    CommitmentOrArbitration,
    Behavior,
    Skill,
    Proposal,
    OperatorContext,
    AutonomicGate,
    FinalMotorGate,
    PossessionLease,
    BrainstemCommand,
    Acknowledgement,
    Outcome,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthorityLane {
    HumanDirect,
    ReignAssist,
    ReignSuggest,
    LlmAdvisory,
    LearnedShadow,
    DeterministicFallback,
    BrainstemReflex,
    MotherbrainRecovery,
    AutonomicSafety,
    Runtime,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthorityFlowStatus {
    Proposed,
    Accepted,
    Vetoed,
    Rejected,
    Expired,
    Superseded,
    Preempted,
    Completed,
    Failed,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuthorityFlowEvent {
    pub event_id: String,
    pub event_type: BrainEventType,
    pub stage: AuthorityFlowStage,
    pub lane: AuthorityLane,
    pub status: AuthorityFlowStatus,
    pub occurred_at_ms: u64,
    pub observed_at_ms: u64,
    pub command_ids: Vec<String>,
    pub goal_ids: Vec<String>,
    pub parent_ids: Vec<String>,
    pub score: Option<f32>,
    pub ttl_ms: Option<u64>,
    pub reason: Option<String>,
    pub authority: AuthoritySignificance,
    pub trust: TrustState,
    pub disposition: EventDisposition,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ObservableAuthorityState {
    pub control_available: bool,
    pub control_armed: bool,
    pub control_mode: Option<String>,
    pub control_source: Option<String>,
    pub control_block_reason: Option<String>,
    pub session_mode: Option<String>,
    pub session_control_state: Option<String>,
    pub session_control_detail: Option<String>,
    pub possession_lease_id: Option<String>,
    pub possession_lease_ttl_ms: Option<u64>,
    pub heartbeat_age_ms: Option<u64>,
    pub stop_state: Option<String>,
    pub estop_latched: Option<bool>,
    pub safety_latch: Option<String>,
    pub bump_active: Option<bool>,
    pub cliff_active: Option<bool>,
    pub wheel_drop_active: Option<bool>,
    pub brainstem_authority: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuthorityFlowResponse {
    pub at_ms: u64,
    pub window_from_ms: u64,
    pub context: ObservableAuthorityState,
    pub events: Vec<AuthorityFlowEvent>,
    pub truncated: bool,
    pub competing_proposals: usize,
    pub advisory_events: usize,
    pub vetoes: usize,
    pub preemptions: usize,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
struct AuthorityFlowQuery {
    at_ms: Option<u64>,
    window_ms: Option<u64>,
}

fn build_authority_flow(
    events: &[BrainEvent],
    at_ms: u64,
    window_ms: u64,
    context: ObservableAuthorityState,
) -> AuthorityFlowResponse {
    let window_from_ms = at_ms.saturating_sub(window_ms);
    let mut flow: Vec<AuthorityFlowEvent> = events
        .iter()
        .filter(|event| {
            event.times.occurred.t_ms >= window_from_ms
                && event.times.occurred.t_ms <= at_ms
                && is_authority_event(event)
        })
        .map(authority_flow_event)
        .collect();
    flow.sort_by_key(|event| (event.occurred_at_ms, authority_stage_rank(event.stage)));
    let truncated = flow.len() > MAX_AUTHORITY_EVENTS;
    if truncated {
        flow.drain(..flow.len() - MAX_AUTHORITY_EVENTS);
    }
    let competing_proposals = flow
        .iter()
        .filter(|event| event.event_type == BrainEventType::Proposal)
        .count();
    let advisory_events = flow
        .iter()
        .filter(|event| matches!(event.lane, AuthorityLane::LlmAdvisory | AuthorityLane::LearnedShadow))
        .count();
    let vetoes = flow
        .iter()
        .filter(|event| event.status == AuthorityFlowStatus::Vetoed)
        .count();
    let preemptions = flow
        .iter()
        .filter(|event| event.status == AuthorityFlowStatus::Preempted)
        .count();
    AuthorityFlowResponse {
        at_ms,
        window_from_ms,
        context,
        events: flow,
        truncated,
        competing_proposals,
        advisory_events,
        vetoes,
        preemptions,
    }
}

fn is_authority_event(event: &BrainEvent) -> bool {
    event.authority != AuthoritySignificance::None
        || matches!(
            event.event_type,
            BrainEventType::Proposal
                | BrainEventType::GateDecision
                | BrainEventType::Command
                | BrainEventType::Outcome
        )
        || authority_stage(event) != AuthorityFlowStage::Unknown
}

fn authority_flow_event(event: &BrainEvent) -> AuthorityFlowEvent {
    let payload = inline_object(event);
    AuthorityFlowEvent {
        event_id: event.event_id.0.clone(),
        event_type: event.event_type,
        stage: authority_stage(event),
        lane: authority_lane(event, payload),
        status: authority_status(event, payload),
        occurred_at_ms: event.times.occurred.t_ms,
        observed_at_ms: event.times.observed.t_ms,
        command_ids: event.references.command_ids.clone(),
        goal_ids: event.references.goal_ids.clone(),
        parent_ids: event
            .links
            .parents
            .iter()
            .map(|parent| parent.event_id.0.clone())
            .collect(),
        score: authority_json_f32(payload, &["score", "priority", "confidence"])
            .or(event.quality.confidence),
        ttl_ms: authority_json_u64(payload, &["ttl_ms", "lease_ttl_ms", "heartbeat_timeout_ms"])
            .or_else(|| {
                event
                    .times
                    .expires_at
                    .as_ref()
                    .map(|expires| expires.t_ms.saturating_sub(event.times.occurred.t_ms))
            }),
        reason: json_string(payload, &["reason", "ignored_reason", "safety_reason", "detail"]),
        authority: event.authority,
        trust: event.quality.trust,
        disposition: event.disposition,
    }
}

fn authority_stage(event: &BrainEvent) -> AuthorityFlowStage {
    let key = format!("{} {}", event.kind, event.producer.component).to_ascii_lowercase();
    for (needle, stage) in [
        ("goal.evaluation", AuthorityFlowStage::GoalEvaluation),
        ("arbitr", AuthorityFlowStage::CommitmentOrArbitration),
        ("commit", AuthorityFlowStage::CommitmentOrArbitration),
        ("behavior", AuthorityFlowStage::Behavior),
        ("skill", AuthorityFlowStage::Skill),
        ("operator", AuthorityFlowStage::OperatorContext),
        ("reign", AuthorityFlowStage::OperatorContext),
        ("autonomic", AuthorityFlowStage::AutonomicGate),
        ("safety", AuthorityFlowStage::AutonomicGate),
        ("motor.gate", AuthorityFlowStage::FinalMotorGate),
        ("lease", AuthorityFlowStage::PossessionLease),
        ("brainstem.command", AuthorityFlowStage::BrainstemCommand),
        ("ack", AuthorityFlowStage::Acknowledgement),
        ("drive", AuthorityFlowStage::DriveOrGoal),
        ("goal", AuthorityFlowStage::DriveOrGoal),
    ] {
        if key.contains(needle) {
            return stage;
        }
    }
    match event.event_type {
        BrainEventType::Proposal => AuthorityFlowStage::Proposal,
        BrainEventType::GateDecision => AuthorityFlowStage::AutonomicGate,
        BrainEventType::Command => AuthorityFlowStage::BrainstemCommand,
        BrainEventType::Outcome => AuthorityFlowStage::Outcome,
        _ => AuthorityFlowStage::Unknown,
    }
}

fn authority_lane(event: &BrainEvent, payload: Option<&serde_json::Map<String, serde_json::Value>>) -> AuthorityLane {
    let key = format!("{} {}", event.kind, event.producer.component).to_ascii_lowercase();
    let mode = json_string(payload, &["mode"]).unwrap_or_default().to_ascii_lowercase();
    if event.producer.brain == Brain::Human && (mode == "direct" || key.contains("direct")) {
        AuthorityLane::HumanDirect
    } else if key.contains("llm") || event.authority == AuthoritySignificance::Advisory {
        AuthorityLane::LlmAdvisory
    } else if key.contains("shadow") {
        AuthorityLane::LearnedShadow
    } else if key.contains("fallback") {
        AuthorityLane::DeterministicFallback
    } else if event.producer.brain == Brain::Brainstem && (key.contains("reflex") || key.contains("safety")) {
        AuthorityLane::BrainstemReflex
    } else if event.producer.brain == Brain::Motherbrain && key.contains("recovery") {
        AuthorityLane::MotherbrainRecovery
    } else if key.contains("safety") || key.contains("autonomic") {
        AuthorityLane::AutonomicSafety
    } else if mode == "assist" {
        AuthorityLane::ReignAssist
    } else if mode == "suggest" || mode == "observe_only" {
        AuthorityLane::ReignSuggest
    } else {
        AuthorityLane::Runtime
    }
}

fn authority_status(
    event: &BrainEvent,
    payload: Option<&serde_json::Map<String, serde_json::Value>>,
) -> AuthorityFlowStatus {
    let reason = json_string(payload, &["reason", "ignored_reason", "safety_reason"])
        .unwrap_or_default()
        .to_ascii_lowercase();
    if reason.contains("preempt") || event.kind.to_ascii_lowercase().contains("preempt") {
        AuthorityFlowStatus::Preempted
    } else if reason.contains("fail") || reason.contains("timeout") {
        AuthorityFlowStatus::Failed
    } else {
        match event.disposition {
            EventDisposition::Accepted => {
                if event.event_type == BrainEventType::Outcome {
                    AuthorityFlowStatus::Completed
                } else {
                    AuthorityFlowStatus::Accepted
                }
            }
            EventDisposition::Vetoed => AuthorityFlowStatus::Vetoed,
            EventDisposition::Rejected => AuthorityFlowStatus::Rejected,
            EventDisposition::Expired => AuthorityFlowStatus::Expired,
            EventDisposition::Superseded => AuthorityFlowStatus::Superseded,
            EventDisposition::Unavailable => AuthorityFlowStatus::Failed,
            EventDisposition::Unknown if event.event_type == BrainEventType::Proposal => {
                AuthorityFlowStatus::Proposed
            }
            EventDisposition::Unknown => AuthorityFlowStatus::Unknown,
        }
    }
}

fn inline_object(event: &BrainEvent) -> Option<&serde_json::Map<String, serde_json::Value>> {
    match &event.payload {
        BrainEventPayload::Inline { data, .. } => data.as_object(),
        _ => None,
    }
}

fn json_string(
    object: Option<&serde_json::Map<String, serde_json::Value>>,
    keys: &[&str],
) -> Option<String> {
    let object = object?;
    keys.iter()
        .find_map(|key| object.get(*key).and_then(serde_json::Value::as_str))
        .map(str::to_string)
}

fn authority_json_u64(
    object: Option<&serde_json::Map<String, serde_json::Value>>,
    keys: &[&str],
) -> Option<u64> {
    let object = object?;
    keys.iter().find_map(|key| object.get(*key).and_then(serde_json::Value::as_u64))
}

fn authority_json_f32(
    object: Option<&serde_json::Map<String, serde_json::Value>>,
    keys: &[&str],
) -> Option<f32> {
    let object = object?;
    keys.iter()
        .find_map(|key| object.get(*key).and_then(serde_json::Value::as_f64))
        .map(|value| value as f32)
}

fn authority_stage_rank(stage: AuthorityFlowStage) -> u8 {
    stage as u8
}

fn observable_authority_state(state: &LiveViewState, events: &[BrainEvent]) -> ObservableAuthorityState {
    let hardware = state.hardware_control_status();
    let session = state.session();
    let snapshot = state.latest();
    let latest_payload = |keys: &[&str]| {
        events
            .iter()
            .rev()
            .find_map(|event| json_string(inline_object(event), keys))
    };
    let latest_u64 = |keys: &[&str]| {
        events
            .iter()
            .rev()
            .find_map(|event| authority_json_u64(inline_object(event), keys))
    };
    let latest_bool = |keys: &[&str]| {
        events.iter().rev().find_map(|event| {
            let object = inline_object(event)?;
            keys.iter()
                .find_map(|key| object.get(*key).and_then(serde_json::Value::as_bool))
        })
    };
    ObservableAuthorityState {
        control_available: hardware.available,
        control_armed: hardware.armed,
        control_mode: hardware.mode,
        control_source: hardware.source,
        control_block_reason: hardware.reason,
        session_mode: session.as_ref().map(|session| session.mode.clone()),
        session_control_state: session.as_ref().and_then(|session| session.control_state.clone()),
        session_control_detail: session.and_then(|session| session.control_detail),
        possession_lease_id: latest_payload(&["lease_id", "possession_lease_id"]),
        possession_lease_ttl_ms: latest_u64(&["lease_ttl_ms", "ttl_ms"]),
        heartbeat_age_ms: latest_u64(&["heartbeat_age_ms"]),
        stop_state: latest_payload(&["stop_state", "stop_status"]),
        estop_latched: latest_bool(&["estop_latched"]),
        safety_latch: latest_payload(&["safety_latch", "safety_latch_kind"]),
        bump_active: snapshot
            .as_ref()
            .map(|snapshot| snapshot.body.flags.bump_left || snapshot.body.flags.bump_right),
        cliff_active: snapshot.as_ref().map(|snapshot| {
            snapshot.body.flags.cliff_left
                || snapshot.body.flags.cliff_front_left
                || snapshot.body.flags.cliff_front_right
                || snapshot.body.flags.cliff_right
        }),
        wheel_drop_active: snapshot.as_ref().map(|snapshot| snapshot.body.flags.wheel_drop),
        brainstem_authority: latest_payload(&["brainstem_authority", "authority"]),
    }
}

async fn get_observatory_authority(
    State(state): State<LiveViewState>,
    Query(query): Query<AuthorityFlowQuery>,
) -> Result<Json<AuthorityFlowResponse>, ObservatoryHttpError> {
    let history = state
        .observatory()
        .query(&BrainEventQuery {
            limit: Some(MAX_OBSERVATORY_QUERY_LIMIT),
            ..BrainEventQuery::default()
        })
        .map_err(|error| ObservatoryHttpError::bad_request(error.to_string()))?;
    let events: Vec<BrainEvent> = history
        .records
        .into_iter()
        .map(|record| match record {
            BrainEventStreamRecord::Event { envelope } => envelope.event,
            BrainEventStreamRecord::Gap { gap } => gap.event,
        })
        .collect();
    let at_ms = query
        .at_ms
        .or_else(|| events.last().map(|event| event.times.occurred.t_ms))
        .unwrap_or_else(wall_now_ms);
    let window_ms = query.window_ms.unwrap_or(DEFAULT_AUTHORITY_WINDOW_MS).clamp(1, 600_000);
    let context = observable_authority_state(&state, &events);
    Ok(Json(build_authority_flow(&events, at_ms, window_ms, context)))
}
