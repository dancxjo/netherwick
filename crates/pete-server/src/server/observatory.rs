const DEFAULT_OBSERVATORY_INGRESS_CAPACITY: usize = 1_024;
const DEFAULT_OBSERVATORY_HISTORY_CAPACITY: usize = 16_384;
const DEFAULT_OBSERVATORY_BROADCAST_CAPACITY: usize = 2_048;
const DEFAULT_OBSERVATORY_QUERY_LIMIT: usize = 500;
const MAX_OBSERVATORY_QUERY_LIMIT: usize = 2_000;
const OBSERVATORY_NOW_HISTORY_CAPACITY: usize = 2_048;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservatoryNowSnapshot {
    pub snapshot_id: String,
    pub now: pete_now::Now,
    #[serde(default)]
    pub observed_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservatoryNowSelection {
    pub selected: ObservatoryNowSnapshot,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous: Option<ObservatoryNowSnapshot>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct ObservatoryNowSeek {
    at_or_before_ms: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrainEventHubConfig {
    pub ingress_capacity: usize,
    pub history_capacity: usize,
    pub broadcast_capacity: usize,
    pub default_query_limit: usize,
    pub max_query_limit: usize,
}

impl Default for BrainEventHubConfig {
    fn default() -> Self {
        Self {
            ingress_capacity: DEFAULT_OBSERVATORY_INGRESS_CAPACITY,
            history_capacity: DEFAULT_OBSERVATORY_HISTORY_CAPACITY,
            broadcast_capacity: DEFAULT_OBSERVATORY_BROADCAST_CAPACITY,
            default_query_limit: DEFAULT_OBSERVATORY_QUERY_LIMIT,
            max_query_limit: MAX_OBSERVATORY_QUERY_LIMIT,
        }
    }
}

impl BrainEventHubConfig {
    fn normalized(self) -> Self {
        Self {
            ingress_capacity: self.ingress_capacity.max(1),
            history_capacity: self.history_capacity.max(1),
            broadcast_capacity: self.broadcast_capacity.max(1),
            default_query_limit: self.default_query_limit.max(1),
            max_query_limit: self.max_query_limit.max(self.default_query_limit.max(1)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublishOutcome {
    Queued,
    CoalescedPendingTelemetry,
    DroppedTelemetry,
    Duplicate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BrainEventPublishError {
    Closed,
    CriticalQueueFull,
    InvalidEvent(String),
    Internal(String),
}

impl std::fmt::Display for BrainEventPublishError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => write!(formatter, "BrainEvent observatory is closed"),
            Self::CriticalQueueFull => write!(
                formatter,
                "BrainEvent critical ingress is full; event was explicitly rejected"
            ),
            Self::InvalidEvent(message) => write!(formatter, "invalid BrainEvent: {message}"),
            Self::Internal(message) => write!(formatter, "BrainEvent hub failure: {message}"),
        }
    }
}

impl std::error::Error for BrainEventPublishError {}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SequencedBrainEvent {
    pub sequence: u64,
    pub event: BrainEvent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SequenceGapReason {
    RetentionExpired,
    ClientLagged,
    Coalesced,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrainEventSequenceGap {
    pub from_sequence: u64,
    pub to_sequence: u64,
    pub reason: SequenceGapReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replacement_sequence: Option<u64>,
    pub event: BrainEvent,
}

impl BrainEventSequenceGap {
    fn new(from_sequence: u64, to_sequence: u64, reason: SequenceGapReason) -> Self {
        let t_ms = wall_now_ms();
        let mut event = BrainEvent::historical(
            BrainEventId::from_domain("transport-gap", format!("{from_sequence}-{to_sequence}")),
            BrainEventType::TransportGap,
            ProducerIdentity::new(Brain::Motherbrain, "observatory.transport"),
            EventTimes::observed(t_ms, t_ms),
        );
        event.kind = "transport.gap".to_string();
        event.disposition = EventDisposition::Unavailable;
        event.payload = BrainEventPayload::inline(serde_json::json!({
            "from_sequence": from_sequence,
            "to_sequence": to_sequence,
            "reason": reason,
        }));
        event.authority = AuthoritySignificance::None;
        event.loss_policy = LossPolicy::LossIntolerant;
        Self {
            from_sequence,
            to_sequence,
            reason,
            replacement_sequence: None,
            event,
        }
    }

    fn coalesced(sequence: u64, replacement_sequence: u64) -> Self {
        let mut gap = Self::new(sequence, sequence, SequenceGapReason::Coalesced);
        gap.replacement_sequence = Some(replacement_sequence);
        gap.event.kind = "transport.replaced".to_string();
        gap.event.disposition = EventDisposition::Superseded;
        gap.event.payload = BrainEventPayload::inline(serde_json::json!({
            "from_sequence": sequence,
            "to_sequence": sequence,
            "reason": SequenceGapReason::Coalesced,
            "replacement_sequence": replacement_sequence,
        }));
        gap
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "record", rename_all = "snake_case")]
pub enum BrainEventStreamRecord {
    Event { envelope: SequencedBrainEvent },
    Gap { gap: BrainEventSequenceGap },
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrainEventTransportHealth {
    pub running: bool,
    pub closed: bool,
    pub ingress_capacity: usize,
    pub ingress_depth: usize,
    pub history_capacity: usize,
    pub history_depth: usize,
    pub oldest_sequence: Option<u64>,
    pub newest_sequence: Option<u64>,
    pub connected_clients: usize,
    pub max_client_lag_events: u64,
    pub ingress_coalesced: u64,
    pub ingress_dropped_telemetry: u64,
    pub ingress_rejected_critical: u64,
    pub history_coalesced: u64,
    pub history_expired: u64,
    pub history_expired_critical: u64,
    pub client_lag_gaps: u64,
    pub durability_enabled: bool,
    pub durable_writer_backlog: usize,
    pub durable_write_failures: u64,
    pub last_durable_sequence: Option<u64>,
    pub durability_gaps: u64,
    pub durable_recovered_records: u64,
    pub durable_rotations: u64,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct BrainEventQuery {
    pub after_sequence: Option<u64>,
    pub occurred_from_ms: Option<u64>,
    pub occurred_to_ms: Option<u64>,
    pub observed_from_ms: Option<u64>,
    pub observed_to_ms: Option<u64>,
    pub event_type: Option<BrainEventType>,
    pub kind: Option<String>,
    pub brain: Option<Brain>,
    pub component: Option<String>,
    pub modality: Option<String>,
    pub model: Option<String>,
    pub calibration_epoch: Option<String>,
    pub snapshot: Option<String>,
    pub entity: Option<String>,
    pub goal: Option<String>,
    pub command: Option<String>,
    pub trust: Option<TrustState>,
    pub disposition: Option<EventDisposition>,
    pub limit: Option<usize>,
}

impl BrainEventQuery {
    pub fn validate(&self, max_limit: usize) -> Result<(), BrainEventQueryError> {
        validate_time_range("occurred", self.occurred_from_ms, self.occurred_to_ms)?;
        validate_time_range("observed", self.observed_from_ms, self.observed_to_ms)?;
        if self.limit == Some(0) {
            return Err(BrainEventQueryError::new("limit must be greater than zero"));
        }
        if self.limit.is_some_and(|limit| limit > max_limit) {
            return Err(BrainEventQueryError::new(format!(
                "limit exceeds maximum {max_limit}"
            )));
        }
        for (name, value) in [
            ("kind", self.kind.as_deref()),
            ("component", self.component.as_deref()),
            ("modality", self.modality.as_deref()),
            ("model", self.model.as_deref()),
            ("calibration_epoch", self.calibration_epoch.as_deref()),
            ("snapshot", self.snapshot.as_deref()),
            ("entity", self.entity.as_deref()),
            ("goal", self.goal.as_deref()),
            ("command", self.command.as_deref()),
        ] {
            if value.is_some_and(|value| value.trim().is_empty()) {
                return Err(BrainEventQueryError::new(format!(
                    "{name} filter cannot be empty"
                )));
            }
        }
        Ok(())
    }

    fn matches(&self, event: &BrainEvent) -> bool {
        within_range(
            event.times.occurred.t_ms,
            self.occurred_from_ms,
            self.occurred_to_ms,
        ) && within_range(
            event.times.observed.t_ms,
            self.observed_from_ms,
            self.observed_to_ms,
        ) && self
            .event_type
            .is_none_or(|event_type| event.event_type == event_type)
            && self.kind.as_ref().is_none_or(|kind| event.kind == *kind)
            && self.brain.is_none_or(|brain| event.producer.brain == brain)
            && self
                .component
                .as_ref()
                .is_none_or(|component| event.producer.component == *component)
            && self.modality.as_ref().is_none_or(|modality| {
                matches!(&event.payload, BrainEventPayload::Inline { data, .. }
                    if data.get("modality").and_then(serde_json::Value::as_str) == Some(modality))
            })
            && self.model.as_ref().is_none_or(|model| {
                event
                    .artifacts
                    .iter()
                    .any(|artifact| artifact.kind == ArtifactKind::Model && artifact.id == *model)
            })
            && contains_filter(&event.calibration_epochs, self.calibration_epoch.as_ref())
            && self
                .snapshot
                .as_ref()
                .is_none_or(|snapshot| event.references.snapshot_id.as_ref() == Some(snapshot))
            && contains_filter(&event.references.entity_ids, self.entity.as_ref())
            && contains_filter(&event.references.goal_ids, self.goal.as_ref())
            && contains_filter(&event.references.command_ids, self.command.as_ref())
            && self.trust.is_none_or(|trust| event.quality.trust == trust)
            && self
                .disposition
                .is_none_or(|disposition| event.disposition == disposition)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrainEventQueryError {
    message: String,
}

impl BrainEventQueryError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for BrainEventQueryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(formatter)
    }
}

impl std::error::Error for BrainEventQueryError {}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrainEventHistoryResponse {
    pub records: Vec<BrainEventStreamRecord>,
    pub next_cursor: u64,
    pub health: BrainEventTransportHealth,
}

#[derive(Default)]
struct IngressState {
    queue: VecDeque<BrainEvent>,
}

#[derive(Default)]
struct HistoryState {
    events: VecDeque<SequencedBrainEvent>,
    replacements: BTreeMap<u64, u64>,
}

#[derive(Default)]
struct ObservatoryCounters {
    next_sequence: AtomicU64,
    ingress_coalesced: AtomicU64,
    ingress_dropped_telemetry: AtomicU64,
    ingress_rejected_critical: AtomicU64,
    history_coalesced: AtomicU64,
    history_expired: AtomicU64,
    history_expired_critical: AtomicU64,
    connected_clients: AtomicUsize,
    max_client_lag_events: AtomicU64,
    client_lag_gaps: AtomicU64,
}

struct ObservatoryShared {
    config: BrainEventHubConfig,
    ingress: Mutex<IngressState>,
    history: Mutex<HistoryState>,
    events_tx: tokio::sync::broadcast::Sender<SequencedBrainEvent>,
    notify: tokio::sync::Notify,
    closed: AtomicBool,
    worker: Mutex<Option<tokio::task::JoinHandle<()>>>,
    external_handles: AtomicUsize,
    counters: ObservatoryCounters,
    durable: Option<Arc<DurableBrainEventStore>>,
}

pub struct BrainEventHub {
    shared: Arc<ObservatoryShared>,
}

impl std::fmt::Debug for BrainEventHub {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BrainEventHub")
            .field("health", &self.health())
            .finish()
    }
}

impl Clone for BrainEventHub {
    fn clone(&self) -> Self {
        self.shared.external_handles.fetch_add(1, Ordering::Relaxed);
        Self {
            shared: Arc::clone(&self.shared),
        }
    }
}

impl Drop for BrainEventHub {
    fn drop(&mut self) {
        if self.shared.external_handles.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.close();
        }
    }
}

impl BrainEventHub {
    pub fn new(config: BrainEventHubConfig) -> Self {
        Self::from_parts(config, None, Vec::new())
    }

    pub fn new_with_durability(
        config: BrainEventHubConfig,
        durability: BrainEventDurabilityConfig,
    ) -> io::Result<Self> {
        let (durable, recovered) = DurableBrainEventStore::open(durability)?;
        Ok(Self::from_parts(config, Some(Arc::new(durable)), recovered))
    }

    fn from_parts(
        config: BrainEventHubConfig,
        durable: Option<Arc<DurableBrainEventStore>>,
        recovered: Vec<SequencedBrainEvent>,
    ) -> Self {
        let config = config.normalized();
        let (events_tx, _) = tokio::sync::broadcast::channel(config.broadcast_capacity);
        let newest_sequence = recovered
            .iter()
            .map(|event| event.sequence)
            .max()
            .unwrap_or(0);
        let retained_start = recovered.len().saturating_sub(config.history_capacity);
        let shared = Arc::new(ObservatoryShared {
                config,
                ingress: Mutex::new(IngressState::default()),
                history: Mutex::new(HistoryState {
                    events: recovered[retained_start..].iter().cloned().collect(),
                    replacements: BTreeMap::new(),
                }),
                events_tx,
                notify: tokio::sync::Notify::new(),
                closed: AtomicBool::new(false),
                worker: Mutex::new(None),
                external_handles: AtomicUsize::new(1),
                counters: ObservatoryCounters::default(),
                durable,
            });
        shared
            .counters
            .next_sequence
            .store(newest_sequence, Ordering::Release);
        Self { shared }
    }

    pub fn start(&self) -> bool {
        if self.shared.closed.load(Ordering::Acquire) {
            return false;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return false;
        };
        let mut worker = self
            .shared
            .worker
            .lock()
            .expect("observatory worker mutex poisoned");
        if worker.as_ref().is_some_and(|worker| !worker.is_finished()) {
            return true;
        }
        let shared = Arc::clone(&self.shared);
        *worker = Some(runtime.spawn(run_observatory_worker(shared)));
        true
    }

    pub fn publish(&self, event: BrainEvent) -> Result<PublishOutcome, BrainEventPublishError> {
        if self.shared.closed.load(Ordering::Acquire) {
            return Err(BrainEventPublishError::Closed);
        }
        event
            .validate()
            .map_err(|error| BrainEventPublishError::InvalidEvent(error.to_string()))?;
        let loss_intolerant = event.requires_loss_intolerant_delivery()
            || matches!(event.loss_policy, LossPolicy::LossIntolerant);
        let durable_claimed = if loss_intolerant {
            if let Some(durable) = &self.shared.durable {
                if !durable.claim_event_id(&event.event_id) {
                    return Ok(PublishOutcome::Duplicate);
                }
                true
            } else {
                false
            }
        } else {
            false
        };
        let coalescing_key = match &event.loss_policy {
            LossPolicy::Coalescible { key } => Some(key.clone()),
            LossPolicy::LossIntolerant => None,
        };
        let outcome = {
            let mut ingress =
                self.shared.ingress.lock().map_err(|_| {
                    BrainEventPublishError::Internal("ingress lock poisoned".into())
                })?;
            if let Some(key) = coalescing_key.as_deref() {
                if let Some(pending) = ingress.queue.iter_mut().find(|pending| {
                    matches!(&pending.loss_policy, LossPolicy::Coalescible { key: pending_key } if pending_key == key)
                }) {
                    *pending = event;
                    self.shared
                        .counters
                        .ingress_coalesced
                        .fetch_add(1, Ordering::Relaxed);
                    PublishOutcome::CoalescedPendingTelemetry
                } else if ingress.queue.len() >= self.shared.config.ingress_capacity {
                    self.shared
                        .counters
                        .ingress_dropped_telemetry
                        .fetch_add(1, Ordering::Relaxed);
                    PublishOutcome::DroppedTelemetry
                } else {
                    ingress.queue.push_back(event);
                    PublishOutcome::Queued
                }
            } else {
                if ingress.queue.len() >= self.shared.config.ingress_capacity {
                    if let Some(position) = ingress.queue.iter().position(|pending| {
                        matches!(pending.loss_policy, LossPolicy::Coalescible { .. })
                    }) {
                        ingress.queue.remove(position);
                        self.shared
                            .counters
                            .ingress_dropped_telemetry
                            .fetch_add(1, Ordering::Relaxed);
                    } else {
                        self.shared
                            .counters
                            .ingress_rejected_critical
                            .fetch_add(1, Ordering::Relaxed);
                        if durable_claimed {
                            if let Some(durable) = &self.shared.durable {
                                durable.release_event_id(&event.event_id);
                            }
                        }
                        return Err(BrainEventPublishError::CriticalQueueFull);
                    }
                }
                debug_assert!(loss_intolerant);
                ingress.queue.push_back(event);
                PublishOutcome::Queued
            }
        };
        self.shared.notify.notify_one();
        Ok(outcome)
    }

    pub fn query(
        &self,
        query: &BrainEventQuery,
    ) -> Result<BrainEventHistoryResponse, BrainEventQueryError> {
        query.validate(self.shared.config.max_query_limit)?;
        let limit = query
            .limit
            .unwrap_or(self.shared.config.default_query_limit);
        let after = query.after_sequence.unwrap_or(0);
        let history = self
            .shared
            .history
            .lock()
            .map_err(|_| BrainEventQueryError::new("history lock poisoned"))?;
        let replacements = history.replacements.clone();
        let mut combined: BTreeMap<u64, SequencedBrainEvent> = history
            .events
            .iter()
            .cloned()
            .map(|event| (event.sequence, event))
            .collect();
        drop(history);
        if let Some(durable) = &self.shared.durable {
            for event in durable.read_events().map_err(|error| {
                BrainEventQueryError::new(format!("durable history unavailable: {error}"))
            })? {
                combined.entry(event.sequence).or_insert(event);
            }
        }
        let mut records = Vec::new();
        let mut expected = after.saturating_add(1);
        let mut next_cursor = after;
        let mut matched = 0;
        for envelope in combined.values().filter(|event| event.sequence > after) {
            if envelope.sequence > expected {
                append_history_discontinuities(
                    &mut records,
                    &replacements,
                    expected,
                    envelope.sequence - 1,
                );
            }
            expected = envelope.sequence.saturating_add(1);
            next_cursor = envelope.sequence;
            if query.matches(&envelope.event) {
                records.push(BrainEventStreamRecord::Event {
                    envelope: envelope.clone(),
                });
                matched += 1;
                if matched >= limit {
                    break;
                }
            }
        }
        Ok(BrainEventHistoryResponse {
            records,
            next_cursor,
            health: self.health(),
        })
    }

    pub fn health(&self) -> BrainEventTransportHealth {
        let ingress_depth = self
            .shared
            .ingress
            .lock()
            .map(|ingress| ingress.queue.len())
            .unwrap_or_default();
        let (history_depth, oldest_sequence, newest_sequence) = self
            .shared
            .history
            .lock()
            .map(|history| {
                (
                    history.events.len(),
                    history.events.front().map(|event| event.sequence),
                    history.events.back().map(|event| event.sequence),
                )
            })
            .unwrap_or_default();
        let running = self
            .shared
            .worker
            .lock()
            .ok()
            .and_then(|worker| worker.as_ref().map(|worker| !worker.is_finished()))
            .unwrap_or(false);
        BrainEventTransportHealth {
            running,
            closed: self.shared.closed.load(Ordering::Acquire),
            ingress_capacity: self.shared.config.ingress_capacity,
            ingress_depth,
            history_capacity: self.shared.config.history_capacity,
            history_depth,
            oldest_sequence,
            newest_sequence,
            connected_clients: self
                .shared
                .counters
                .connected_clients
                .load(Ordering::Relaxed),
            max_client_lag_events: self
                .shared
                .counters
                .max_client_lag_events
                .load(Ordering::Relaxed),
            ingress_coalesced: self
                .shared
                .counters
                .ingress_coalesced
                .load(Ordering::Relaxed),
            ingress_dropped_telemetry: self
                .shared
                .counters
                .ingress_dropped_telemetry
                .load(Ordering::Relaxed),
            ingress_rejected_critical: self
                .shared
                .counters
                .ingress_rejected_critical
                .load(Ordering::Relaxed),
            history_coalesced: self
                .shared
                .counters
                .history_coalesced
                .load(Ordering::Relaxed),
            history_expired: self.shared.counters.history_expired.load(Ordering::Relaxed),
            history_expired_critical: self
                .shared
                .counters
                .history_expired_critical
                .load(Ordering::Relaxed),
            client_lag_gaps: self.shared.counters.client_lag_gaps.load(Ordering::Relaxed),
            durability_enabled: self.shared.durable.is_some(),
            durable_writer_backlog: self
                .shared
                .durable
                .as_ref()
                .map_or(0, |durable| durable.backlog()),
            durable_write_failures: self
                .shared
                .durable
                .as_ref()
                .map_or(0, |durable| durable.write_failures()),
            last_durable_sequence: self
                .shared
                .durable
                .as_ref()
                .and_then(|durable| durable.last_durable_sequence()),
            durability_gaps: self
                .shared
                .durable
                .as_ref()
                .map_or(0, |durable| durable.gaps()),
            durable_recovered_records: self
                .shared
                .durable
                .as_ref()
                .map_or(0, |durable| durable.recovered_records()),
            durable_rotations: self
                .shared
                .durable
                .as_ref()
                .map_or(0, |durable| durable.rotations()),
        }
    }

    pub fn close(&self) {
        self.shared.closed.store(true, Ordering::Release);
        // There is one worker. `notify_one` retains a permit if close races the
        // worker between its closed check and `notified().await`.
        self.shared.notify.notify_one();
    }

    pub async fn shutdown(&self) {
        self.close();
        let worker = self
            .shared
            .worker
            .lock()
            .ok()
            .and_then(|mut worker| worker.take());
        if let Some(worker) = worker {
            let _ = worker.await;
        }
        if let Some(durable) = &self.shared.durable {
            durable.shutdown();
        }
    }

    fn subscribe(&self) -> tokio::sync::broadcast::Receiver<SequencedBrainEvent> {
        self.shared.events_tx.subscribe()
    }
}

async fn run_observatory_worker(shared: Arc<ObservatoryShared>) {
    loop {
        let event = shared
            .ingress
            .lock()
            .ok()
            .and_then(|mut ingress| ingress.queue.pop_front());
        if let Some(event) = event {
            retain_brain_event(&shared, event);
            continue;
        }
        if shared.closed.load(Ordering::Acquire) {
            break;
        }
        shared.notify.notified().await;
    }
    if let Some(durable) = &shared.durable {
        durable.close();
    }
}

fn retain_brain_event(shared: &ObservatoryShared, event: BrainEvent) {
    let sequence = shared
        .counters
        .next_sequence
        .fetch_add(1, Ordering::Relaxed)
        .saturating_add(1);
    let envelope = SequencedBrainEvent { sequence, event };
    if let Some(durable) = &shared.durable {
        durable.enqueue(envelope.clone());
    }
    if let Ok(mut history) = shared.history.lock() {
        if let LossPolicy::Coalescible { key } = &envelope.event.loss_policy {
            if let Some(position) = history.events.iter().position(|prior| {
                matches!(&prior.event.loss_policy, LossPolicy::Coalescible { key: prior_key } if prior_key == key)
            }) {
                if let Some(replaced) = history.events.remove(position) {
                    history.replacements.insert(replaced.sequence, sequence);
                    while history.replacements.len() > shared.config.history_capacity {
                        let Some(oldest) = history.replacements.keys().next().copied() else {
                            break;
                        };
                        history.replacements.remove(&oldest);
                    }
                }
                shared
                    .counters
                    .history_coalesced
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        while history.events.len() >= shared.config.history_capacity {
            let position = history
                .events
                .iter()
                .position(|prior| matches!(prior.event.loss_policy, LossPolicy::Coalescible { .. }))
                .unwrap_or(0);
            if let Some(expired) = history.events.remove(position) {
                shared
                    .counters
                    .history_expired
                    .fetch_add(1, Ordering::Relaxed);
                if expired.event.requires_loss_intolerant_delivery()
                    || matches!(expired.event.loss_policy, LossPolicy::LossIntolerant)
                {
                    shared
                        .counters
                        .history_expired_critical
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        history.events.push_back(envelope.clone());
    }
    let _ = shared.events_tx.send(envelope);
}

fn append_history_discontinuities(
    records: &mut Vec<BrainEventStreamRecord>,
    replacements: &BTreeMap<u64, u64>,
    from_sequence: u64,
    to_sequence: u64,
) {
    if from_sequence > to_sequence {
        return;
    }
    let mut retention_start = Some(from_sequence);
    for (&sequence, &replacement_sequence) in replacements.range(from_sequence..=to_sequence) {
        if let Some(start) = retention_start.take() {
            if start < sequence {
                records.push(BrainEventStreamRecord::Gap {
                    gap: BrainEventSequenceGap::new(
                        start,
                        sequence - 1,
                        SequenceGapReason::RetentionExpired,
                    ),
                });
            }
        }
        records.push(BrainEventStreamRecord::Gap {
            gap: BrainEventSequenceGap::coalesced(sequence, replacement_sequence),
        });
        retention_start = sequence.checked_add(1);
    }
    if let Some(start) = retention_start.filter(|start| *start <= to_sequence) {
        records.push(BrainEventStreamRecord::Gap {
            gap: BrainEventSequenceGap::new(
                start,
                to_sequence,
                SequenceGapReason::RetentionExpired,
            ),
        });
    }
}

#[derive(Debug)]
struct ObservatoryHttpError {
    status: StatusCode,
    message: String,
}

impl ObservatoryHttpError {
    fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
        }
    }

    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }
}

impl IntoResponse for ObservatoryHttpError {
    fn into_response(self) -> axum::response::Response {
        (
            self.status,
            Json(serde_json::json!({ "error": self.message })),
        )
            .into_response()
    }
}

async fn get_observatory_history(
    State(state): State<LiveViewState>,
    Query(query): Query<BrainEventQuery>,
) -> Result<Json<BrainEventHistoryResponse>, ObservatoryHttpError> {
    state
        .observatory()
        .query(&query)
        .map(Json)
        .map_err(|error| ObservatoryHttpError::bad_request(error.to_string()))
}

async fn get_observatory_health(
    State(state): State<LiveViewState>,
) -> Json<BrainEventTransportHealth> {
    Json(state.observatory().health())
}

async fn get_observatory_now_snapshot(
    State(state): State<LiveViewState>,
    AxumPath(snapshot_id): AxumPath<String>,
) -> Result<Json<ObservatoryNowSelection>, ObservatoryHttpError> {
    state
        .observatory_now_snapshot(&snapshot_id)
        .map(Json)
        .ok_or_else(|| {
            ObservatoryHttpError::unavailable(format!(
                "Now snapshot {snapshot_id} is not retained by this source"
            ))
        })
}

async fn get_observatory_now_at_or_before(
    State(state): State<LiveViewState>,
    Query(seek): Query<ObservatoryNowSeek>,
) -> Result<Json<ObservatoryNowSelection>, ObservatoryHttpError> {
    state
        .observatory_now_at_or_before(seek.at_or_before_ms)
        .map(Json)
        .ok_or_else(|| {
            ObservatoryHttpError::unavailable(format!(
                "no retained Now snapshot exists at or before {}",
                seek.at_or_before_ms
            ))
        })
}

async fn get_observatory_events_ws(
    ws: WebSocketUpgrade,
    State(state): State<LiveViewState>,
    Query(query): Query<BrainEventQuery>,
) -> Result<impl IntoResponse, ObservatoryHttpError> {
    query
        .validate(state.observatory().shared.config.max_query_limit)
        .map_err(|error| ObservatoryHttpError::bad_request(error.to_string()))?;
    if !state.observatory().start() {
        return Err(ObservatoryHttpError::unavailable(
            "BrainEvent observatory is not running",
        ));
    }
    Ok(ws.on_upgrade(move |socket| stream_observatory_events(socket, state, query)))
}

async fn stream_observatory_events(
    mut socket: WebSocket,
    state: LiveViewState,
    query: BrainEventQuery,
) {
    let hub = state.observatory();
    let mut receiver = hub.subscribe();
    let Ok(history) = hub.query(&query) else {
        return;
    };
    let mut last_sequence = query.after_sequence.unwrap_or(0);
    for record in history.records {
        if let BrainEventStreamRecord::Event { envelope } = &record {
            last_sequence = last_sequence.max(envelope.sequence);
        }
        if send_observatory_record(&mut socket, &record).await.is_err() {
            return;
        }
    }
    last_sequence = last_sequence.max(history.next_cursor);
    hub.shared
        .counters
        .connected_clients
        .fetch_add(1, Ordering::Relaxed);
    let _client_guard = ObservatoryClientGuard {
        shared: Arc::clone(&hub.shared),
    };
    loop {
        let lag = receiver.len() as u64;
        hub.shared
            .counters
            .max_client_lag_events
            .fetch_max(lag, Ordering::Relaxed);
        tokio::select! {
            received = receiver.recv() => match received {
                Ok(envelope) => {
                    if envelope.sequence <= last_sequence || !query.matches(&envelope.event) {
                        last_sequence = last_sequence.max(envelope.sequence);
                        continue;
                    }
                    if envelope.sequence > last_sequence.saturating_add(1) {
                        let record = BrainEventStreamRecord::Gap {
                            gap: BrainEventSequenceGap::new(
                                last_sequence.saturating_add(1),
                                envelope.sequence - 1,
                                SequenceGapReason::ClientLagged,
                            ),
                        };
                        hub.shared.counters.client_lag_gaps.fetch_add(1, Ordering::Relaxed);
                        if send_observatory_record(&mut socket, &record).await.is_err() {
                            break;
                        }
                    }
                    last_sequence = envelope.sequence;
                    let record = BrainEventStreamRecord::Event { envelope };
                    if send_observatory_record(&mut socket, &record).await.is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                    let from = last_sequence.saturating_add(1);
                    let to = last_sequence.saturating_add(skipped);
                    let record = BrainEventStreamRecord::Gap {
                        gap: BrainEventSequenceGap::new(from, to, SequenceGapReason::ClientLagged),
                    };
                    hub.shared.counters.client_lag_gaps.fetch_add(1, Ordering::Relaxed);
                    last_sequence = to;
                    if send_observatory_record(&mut socket, &record).await.is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            },
            inbound = socket.recv() => match inbound {
                Some(Ok(Message::Close(_))) | None => break,
                Some(Err(_)) => break,
                Some(Ok(Message::Ping(payload))) => {
                    if socket.send(Message::Pong(payload)).await.is_err() {
                        break;
                    }
                }
                Some(Ok(_)) => {}
            }
        }
    }
}

struct ObservatoryClientGuard {
    shared: Arc<ObservatoryShared>,
}

impl Drop for ObservatoryClientGuard {
    fn drop(&mut self) {
        self.shared
            .counters
            .connected_clients
            .fetch_sub(1, Ordering::Relaxed);
    }
}

async fn send_observatory_record(
    socket: &mut WebSocket,
    record: &BrainEventStreamRecord,
) -> Result<(), axum::Error> {
    let payload = serde_json::to_string(record).map_err(axum::Error::new)?;
    socket.send(Message::Text(payload.into())).await
}

fn validate_time_range(
    name: &str,
    from: Option<u64>,
    to: Option<u64>,
) -> Result<(), BrainEventQueryError> {
    if from.zip(to).is_some_and(|(from, to)| from > to) {
        return Err(BrainEventQueryError::new(format!(
            "{name}_from_ms cannot exceed {name}_to_ms"
        )));
    }
    Ok(())
}

fn within_range(value: u64, from: Option<u64>, to: Option<u64>) -> bool {
    from.is_none_or(|from| value >= from) && to.is_none_or(|to| value <= to)
}

fn contains_filter(values: &[String], filter: Option<&String>) -> bool {
    filter.is_none_or(|filter| values.contains(filter))
}
