#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservatorySourceKind {
    Live,
    Capture,
    Ledger,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservatorySourceIdentity {
    pub id: String,
    pub kind: ObservatorySourceKind,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "lane", rename_all = "snake_case")]
pub enum ObservatoryEventLane {
    Recorded,
    Reprocessed { model_id: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservatorySourceEvent {
    pub envelope: SequencedBrainEvent,
    pub lane: ObservatoryEventLane,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservatorySnapshotRef {
    pub snapshot_id: String,
    pub event_id: BrainEventId,
    pub t_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservatoryAssetRef {
    pub reference: PayloadReference,
    pub available: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ObservatorySourceHealth {
    pub available: bool,
    pub complete: bool,
    pub event_count: usize,
    pub snapshot_count: usize,
    pub asset_count: usize,
    pub gaps: Vec<BrainEventSequenceGap>,
    pub warnings: Vec<String>,
}

pub trait BrainEventSource: Send + Sync {
    fn identity(&self) -> ObservatorySourceIdentity;
    fn query(
        &self,
        query: &BrainEventQuery,
    ) -> Result<BrainEventHistoryResponse, BrainEventQueryError>;
    fn snapshots(&self) -> Vec<ObservatorySnapshotRef>;
    fn snapshot_at_or_before(&self, t_ms: u64) -> Option<ObservatorySnapshotRef>;
    fn asset(&self, id: &str) -> Option<ObservatoryAssetRef>;
    fn health(&self) -> ObservatorySourceHealth;
    fn subscribe(&self) -> Option<tokio::sync::broadcast::Receiver<SequencedBrainEvent>>;
}

#[derive(Clone, Debug)]
pub struct LiveBrainEventSource {
    identity: ObservatorySourceIdentity,
    hub: BrainEventHub,
}

impl LiveBrainEventSource {
    pub fn new(id: impl Into<String>, label: impl Into<String>, hub: BrainEventHub) -> Self {
        Self {
            identity: ObservatorySourceIdentity {
                id: id.into(),
                kind: ObservatorySourceKind::Live,
                label: label.into(),
            },
            hub,
        }
    }
}

impl BrainEventSource for LiveBrainEventSource {
    fn identity(&self) -> ObservatorySourceIdentity {
        self.identity.clone()
    }

    fn query(
        &self,
        query: &BrainEventQuery,
    ) -> Result<BrainEventHistoryResponse, BrainEventQueryError> {
        self.hub.query(query)
    }

    fn snapshots(&self) -> Vec<ObservatorySnapshotRef> {
        source_snapshots(
            &self
                .hub
                .query(&BrainEventQuery {
                    event_type: Some(BrainEventType::Snapshot),
                    limit: Some(self.hub.shared.config.max_query_limit),
                    ..BrainEventQuery::default()
                })
                .ok(),
        )
    }

    fn snapshot_at_or_before(&self, t_ms: u64) -> Option<ObservatorySnapshotRef> {
        self.snapshots()
            .into_iter()
            .filter(|snapshot| snapshot.t_ms <= t_ms)
            .max_by_key(|snapshot| snapshot.t_ms)
    }

    fn asset(&self, _id: &str) -> Option<ObservatoryAssetRef> {
        None
    }

    fn health(&self) -> ObservatorySourceHealth {
        let health = self.hub.health();
        ObservatorySourceHealth {
            available: health.running && !health.closed,
            complete: false,
            event_count: health.history_depth,
            snapshot_count: self.snapshots().len(),
            asset_count: 0,
            gaps: Vec::new(),
            warnings: vec!["live history is bounded; older events may expire".to_string()],
        }
    }

    fn subscribe(&self) -> Option<tokio::sync::broadcast::Receiver<SequencedBrainEvent>> {
        Some(self.hub.subscribe())
    }
}

#[derive(Clone, Debug)]
pub struct ReplayBrainEventSource {
    identity: ObservatorySourceIdentity,
    events: Vec<ObservatorySourceEvent>,
    snapshots: Vec<ObservatorySnapshotRef>,
    assets: BTreeMap<String, ObservatoryAssetRef>,
    gaps: Vec<BrainEventSequenceGap>,
    warnings: Vec<String>,
}

impl ReplayBrainEventSource {
    pub fn from_recorded(
        identity: ObservatorySourceIdentity,
        events: Vec<SequencedBrainEvent>,
    ) -> Self {
        let events = events
            .into_iter()
            .map(|envelope| ObservatorySourceEvent {
                envelope,
                lane: ObservatoryEventLane::Recorded,
            })
            .collect::<Vec<_>>();
        let snapshots = snapshots_from_source_events(&events);
        Self {
            identity,
            events,
            snapshots,
            assets: BTreeMap::new(),
            gaps: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub async fn from_capture(reader: &CaptureReader) -> anyhow::Result<Self> {
        let manifest = reader.manifest();
        let source_id = format!("capture:{}", manifest.id);
        let frames = reader.read_frames().await?;
        let mut events = Vec::new();
        let mut snapshots = Vec::new();
        let mut assets = BTreeMap::new();
        let mut gaps = Vec::new();
        let mut expected_index = 0_u64;
        let mut clock_number = 0_u64;
        let mut previous_t_ms = None;
        let mut sequence = 0_u64;
        for frame in frames {
            if frame.index > expected_index {
                gaps.push(BrainEventSequenceGap::new(
                    expected_index.saturating_add(1),
                    frame.index,
                    SequenceGapReason::RetentionExpired,
                ));
            }
            expected_index = frame.index.saturating_add(1);
            if previous_t_ms.is_some_and(|previous| frame.t_ms < previous) {
                clock_number = clock_number.saturating_add(1);
            }
            previous_t_ms = Some(frame.t_ms);
            let clock_epoch = format!("{source_id}:clock:{clock_number}");
            let snapshot_id = format!("{source_id}:frame:{}", frame.index);
            let now = frame.snapshot.to_now(frame.t_ms);
            let snapshot_event = BrainEvent::from_now_snapshot(
                snapshot_id.clone(),
                &now,
                frame.t_ms,
                Some(clock_epoch.clone()),
            );
            sequence = sequence.saturating_add(1);
            snapshots.push(ObservatorySnapshotRef {
                snapshot_id: snapshot_id.clone(),
                event_id: snapshot_event.event_id.clone(),
                t_ms: frame.t_ms,
            });
            events.push(ObservatorySourceEvent {
                envelope: SequencedBrainEvent {
                    sequence,
                    event: snapshot_event,
                },
                lane: ObservatoryEventLane::Recorded,
            });
            for (event_index, recorded) in frame.events.into_iter().enumerate() {
                let mut event = BrainEvent::historical(
                    BrainEventId::from_domain(
                        "capture-event",
                        format!("{}:{}:{event_index}", manifest.id, frame.index),
                    ),
                    BrainEventType::Unknown,
                    ProducerIdentity::new(Brain::Unknown, "capture.recorded_event"),
                    EventTimes {
                        occurred: pete_events::ClockedTime::in_epoch(
                            recorded.t_ms,
                            clock_epoch.clone(),
                        ),
                        observed: pete_events::ClockedTime::in_epoch(
                            frame.t_ms,
                            clock_epoch.clone(),
                        ),
                        valid_from: None,
                        expires_at: None,
                    },
                );
                event.kind = recorded.kind;
                event.references.snapshot_id = Some(snapshot_id.clone());
                event.payload = BrainEventPayload::inline(recorded.payload);
                event.loss_policy = LossPolicy::LossIntolerant;
                sequence = sequence.saturating_add(1);
                events.push(ObservatorySourceEvent {
                    envelope: SequencedBrainEvent { sequence, event },
                    lane: ObservatoryEventLane::Recorded,
                });
            }
            add_capture_assets(&mut assets, &source_id, frame.index, &frame.assets);
        }
        let mut warnings = manifest.warnings.clone();
        if !manifest.streams.missing.is_empty() {
            warnings.push(format!(
                "capture is missing streams: {}",
                manifest.streams.missing.join(", ")
            ));
        }
        Ok(Self {
            identity: ObservatorySourceIdentity {
                id: source_id,
                kind: ObservatorySourceKind::Capture,
                label: manifest.id.clone(),
            },
            events,
            snapshots,
            assets,
            gaps,
            warnings,
        })
    }

    pub fn add_reprocessed_lane(
        &mut self,
        model_id: impl Into<String>,
        events: Vec<SequencedBrainEvent>,
    ) {
        let model_id = model_id.into();
        self.events
            .extend(events.into_iter().map(|envelope| ObservatorySourceEvent {
                envelope,
                lane: ObservatoryEventLane::Reprocessed {
                    model_id: model_id.clone(),
                },
            }));
        self.events.sort_by_key(|event| event.envelope.sequence);
    }

    pub fn events(&self) -> &[ObservatorySourceEvent] {
        &self.events
    }
}

impl BrainEventSource for ReplayBrainEventSource {
    fn identity(&self) -> ObservatorySourceIdentity {
        self.identity.clone()
    }

    fn query(
        &self,
        query: &BrainEventQuery,
    ) -> Result<BrainEventHistoryResponse, BrainEventQueryError> {
        query.validate(MAX_OBSERVATORY_QUERY_LIMIT)?;
        let after = query.after_sequence.unwrap_or(0);
        let limit = query.limit.unwrap_or(DEFAULT_OBSERVATORY_QUERY_LIMIT);
        let mut records = Vec::new();
        let mut next_cursor = after;
        let mut matches = 0;
        records.extend(
            self.gaps
                .iter()
                .filter(|gap| gap.to_sequence > after)
                .cloned()
                .map(|gap| BrainEventStreamRecord::Gap { gap }),
        );
        for event in self
            .events
            .iter()
            .filter(|event| matches!(event.lane, ObservatoryEventLane::Recorded))
            .filter(|event| event.envelope.sequence > after)
        {
            next_cursor = event.envelope.sequence;
            if query.matches(&event.envelope.event) {
                records.push(BrainEventStreamRecord::Event {
                    envelope: event.envelope.clone(),
                });
                matches += 1;
                if matches >= limit {
                    break;
                }
            }
        }
        Ok(BrainEventHistoryResponse {
            records,
            next_cursor,
            health: BrainEventTransportHealth {
                running: false,
                closed: false,
                history_capacity: self.events.len(),
                history_depth: self.events.len(),
                oldest_sequence: self.events.first().map(|event| event.envelope.sequence),
                newest_sequence: self.events.last().map(|event| event.envelope.sequence),
                ..BrainEventTransportHealth::default()
            },
        })
    }

    fn snapshots(&self) -> Vec<ObservatorySnapshotRef> {
        self.snapshots.clone()
    }

    fn snapshot_at_or_before(&self, t_ms: u64) -> Option<ObservatorySnapshotRef> {
        self.snapshots
            .iter()
            .filter(|snapshot| snapshot.t_ms <= t_ms)
            .max_by_key(|snapshot| snapshot.t_ms)
            .cloned()
    }

    fn asset(&self, id: &str) -> Option<ObservatoryAssetRef> {
        self.assets.get(id).cloned()
    }

    fn health(&self) -> ObservatorySourceHealth {
        ObservatorySourceHealth {
            available: true,
            complete: self.gaps.is_empty() && self.warnings.is_empty(),
            event_count: self.events.len(),
            snapshot_count: self.snapshots.len(),
            asset_count: self.assets.len(),
            gaps: self.gaps.clone(),
            warnings: self.warnings.clone(),
        }
    }

    fn subscribe(&self) -> Option<tokio::sync::broadcast::Receiver<SequencedBrainEvent>> {
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservatoryPlaybackMode {
    Paused,
    Playing,
    FollowLive,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservatoryNavigationState {
    pub source_id: String,
    pub mode: ObservatoryPlaybackMode,
    pub selected_time_ms: Option<u64>,
    pub selected_event_id: Option<BrainEventId>,
    pub panel: String,
    pub filters: BrainEventNavigationFilters,
    pub speed: f32,
    pub loop_range_ms: Option<[u64; 2]>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrainEventNavigationFilters {
    pub event_type: Option<BrainEventType>,
    pub component: Option<String>,
    pub trust: Option<TrustState>,
    pub disposition: Option<EventDisposition>,
}

impl ObservatoryNavigationState {
    pub fn new(source_id: impl Into<String>) -> Self {
        Self {
            source_id: source_id.into(),
            mode: ObservatoryPlaybackMode::Paused,
            selected_time_ms: None,
            selected_event_id: None,
            panel: "timeline".to_string(),
            filters: BrainEventNavigationFilters::default(),
            speed: 1.0,
            loop_range_ms: None,
        }
    }

    pub fn play(&mut self) {
        self.mode = ObservatoryPlaybackMode::Playing;
    }

    pub fn pause(&mut self) {
        self.mode = ObservatoryPlaybackMode::Paused;
    }

    pub fn follow_live(&mut self) {
        self.mode = ObservatoryPlaybackMode::FollowLive;
        self.selected_time_ms = None;
    }

    pub fn seek<S: BrainEventSource + ?Sized>(&mut self, source: &S, t_ms: u64) {
        self.mode = ObservatoryPlaybackMode::Paused;
        self.selected_time_ms = Some(t_ms);
        self.selected_event_id = source
            .snapshot_at_or_before(t_ms)
            .map(|snapshot| snapshot.event_id);
    }

    pub fn step<S: BrainEventSource + ?Sized>(&mut self, source: &S, direction: i8) {
        let snapshots = source.snapshots();
        if snapshots.is_empty() {
            return;
        }
        let selected = self.selected_time_ms.unwrap_or_else(|| {
            if direction < 0 {
                u64::MAX
            } else {
                0
            }
        });
        let target = if direction < 0 {
            snapshots
                .iter()
                .filter(|snapshot| snapshot.t_ms < selected)
                .max_by_key(|snapshot| snapshot.t_ms)
        } else {
            snapshots
                .iter()
                .filter(|snapshot| snapshot.t_ms > selected)
                .min_by_key(|snapshot| snapshot.t_ms)
        };
        if let Some(target) = target {
            self.selected_time_ms = Some(target.t_ms);
            self.selected_event_id = Some(target.event_id.clone());
        }
    }

    pub fn set_speed(&mut self, speed: f32) -> Result<(), &'static str> {
        if !speed.is_finite() || !(0.1..=16.0).contains(&speed) {
            return Err("playback speed must be finite and between 0.1 and 16");
        }
        self.speed = speed;
        Ok(())
    }

    pub fn set_loop(&mut self, range: Option<[u64; 2]>) -> Result<(), &'static str> {
        if range.is_some_and(|range| range[0] > range[1]) {
            return Err("loop start cannot exceed loop end");
        }
        self.loop_range_ms = range;
        Ok(())
    }
}

fn source_snapshots(
    response: &Option<BrainEventHistoryResponse>,
) -> Vec<ObservatorySnapshotRef> {
    response
        .iter()
        .flat_map(|response| &response.records)
        .filter_map(|record| match record {
            BrainEventStreamRecord::Event { envelope } => snapshot_from_event(&envelope.event),
            BrainEventStreamRecord::Gap { .. } => None,
        })
        .collect()
}

fn snapshots_from_source_events(events: &[ObservatorySourceEvent]) -> Vec<ObservatorySnapshotRef> {
    events
        .iter()
        .filter(|event| matches!(event.lane, ObservatoryEventLane::Recorded))
        .filter_map(|event| snapshot_from_event(&event.envelope.event))
        .collect()
}

fn snapshot_from_event(event: &BrainEvent) -> Option<ObservatorySnapshotRef> {
    if event.event_type != BrainEventType::Snapshot {
        return None;
    }
    Some(ObservatorySnapshotRef {
        snapshot_id: event.references.snapshot_id.clone()?,
        event_id: event.event_id.clone(),
        t_ms: event.times.occurred.t_ms,
    })
}

fn add_capture_assets(
    assets: &mut BTreeMap<String, ObservatoryAssetRef>,
    source_id: &str,
    frame_index: u64,
    frame: &pete_worldlab::CaptureFrameAssets,
) {
    for (kind, path) in [
        ("rgb", frame.rgb.as_ref()),
        ("depth", frame.depth.as_ref()),
        ("audio", frame.audio.as_ref()),
        ("pointcloud", frame.pointcloud.as_ref()),
        ("perception", frame.perception.as_ref()),
        ("vision", frame.vision.as_ref()),
        ("camera", frame.camera.as_ref()),
        ("lidar", frame.lidar.as_ref()),
        ("imu", frame.imu.as_ref()),
        ("transcript", frame.transcript.as_ref()),
        ("calibration", frame.calibration.as_ref()),
    ] {
        if let Some(path) = path {
            let id = format!("{source_id}:frame:{frame_index}:{kind}");
            assets.insert(
                id.clone(),
                ObservatoryAssetRef {
                    reference: PayloadReference {
                        id,
                        locator: path.clone(),
                        media_type: format!("application/vnd.netherwick.{kind}"),
                        byte_len: None,
                        checksum: None,
                        redacted: false,
                    },
                    available: true,
                },
            );
        }
    }
}
