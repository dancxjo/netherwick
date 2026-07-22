const DURABLE_BRAIN_EVENT_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrainEventDurabilityConfig {
    pub path: PathBuf,
    pub max_segment_bytes: u64,
    pub retained_segments: usize,
    pub writer_queue_capacity: usize,
    pub sync_data: bool,
    #[doc(hidden)]
    pub injected_failure_after_records: Option<u64>,
}

impl BrainEventDurabilityConfig {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            max_segment_bytes: 64 * 1024 * 1024,
            retained_segments: 8,
            writer_queue_capacity: 4_096,
            sync_data: true,
            injected_failure_after_records: None,
        }
    }

    fn normalized(mut self) -> Self {
        self.max_segment_bytes = self.max_segment_bytes.max(1);
        self.retained_segments = self.retained_segments.max(1);
        self.writer_queue_capacity = self.writer_queue_capacity.max(1);
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DurableBrainEventRecord {
    format_version: u32,
    envelope: SequencedBrainEvent,
    envelope_sha256: String,
}

impl DurableBrainEventRecord {
    fn new(envelope: SequencedBrainEvent) -> io::Result<Self> {
        let canonical = serde_json::to_vec(&envelope).map_err(io::Error::other)?;
        Ok(Self {
            format_version: DURABLE_BRAIN_EVENT_FORMAT_VERSION,
            envelope,
            envelope_sha256: format!("{:x}", Sha256::digest(canonical)),
        })
    }

    fn validate(&self) -> bool {
        self.format_version == DURABLE_BRAIN_EVENT_FORMAT_VERSION
            && serde_json::to_vec(&self.envelope).is_ok_and(|canonical| {
                format!("{:x}", Sha256::digest(canonical)) == self.envelope_sha256
            })
    }
}

#[derive(Default)]
struct DurableBrainEventStats {
    backlog: AtomicUsize,
    write_failures: AtomicU64,
    last_durable_sequence: AtomicU64,
    gaps: AtomicU64,
    recovered_records: AtomicU64,
    rotations: AtomicU64,
}

enum DurableWriterCommand {
    Append(SequencedBrainEvent),
}

struct DurableBrainEventStore {
    config: BrainEventDurabilityConfig,
    sender: Mutex<Option<std::sync::mpsc::SyncSender<DurableWriterCommand>>>,
    writer: Mutex<Option<std::thread::JoinHandle<()>>>,
    stats: Arc<DurableBrainEventStats>,
    seen_ids: Mutex<DurableSeenEventIds>,
}

struct DurableSeenEventIds {
    ids: HashSet<BrainEventId>,
    order: VecDeque<BrainEventId>,
    capacity: usize,
}

impl DurableSeenEventIds {
    fn from_recovered(config: &BrainEventDurabilityConfig, events: &[SequencedBrainEvent]) -> Self {
        // Every framed record consumes at least one byte, so the configured
        // on-disk byte ceiling is also a conservative upper bound on IDs that
        // can still be present in retained segments.
        let capacity = config
            .max_segment_bytes
            .saturating_mul(config.retained_segments.saturating_add(1) as u64)
            .min(usize::MAX as u64) as usize;
        let mut seen = Self {
            ids: HashSet::new(),
            order: VecDeque::new(),
            capacity: capacity.max(1),
        };
        for event in events {
            seen.insert(event.event.event_id.clone());
        }
        seen
    }

    fn insert(&mut self, event_id: BrainEventId) -> bool {
        if !self.ids.insert(event_id.clone()) {
            return false;
        }
        self.order.push_back(event_id);
        while self.order.len() > self.capacity {
            if let Some(expired) = self.order.pop_front() {
                self.ids.remove(&expired);
            }
        }
        true
    }

    fn remove(&mut self, event_id: &BrainEventId) {
        self.ids.remove(event_id);
        if let Some(index) = self.order.iter().position(|seen| seen == event_id) {
            self.order.remove(index);
        }
    }
}

impl DurableBrainEventStore {
    fn open(
        config: BrainEventDurabilityConfig,
    ) -> io::Result<(Self, Vec<SequencedBrainEvent>)> {
        let config = config.normalized();
        if let Some(parent) = config.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let scan = scan_durable_history(&config, true)?;
        let stats = Arc::new(DurableBrainEventStats::default());
        stats
            .last_durable_sequence
            .store(scan.newest_sequence, Ordering::Release);
        stats
            .gaps
            .store(scan.recovery_gaps, Ordering::Release);
        stats
            .recovered_records
            .store(scan.events.len() as u64, Ordering::Release);
        let seen_ids = DurableSeenEventIds::from_recovered(&config, &scan.events);
        let (sender, receiver) =
            std::sync::mpsc::sync_channel(config.writer_queue_capacity);
        let writer_config = config.clone();
        let writer_stats = Arc::clone(&stats);
        let writer = std::thread::Builder::new()
            .name("observatory-durable-writer".into())
            .spawn(move || run_durable_writer(writer_config, writer_stats, receiver))?;
        Ok((
            Self {
                config,
                sender: Mutex::new(Some(sender)),
                writer: Mutex::new(Some(writer)),
                stats,
                seen_ids: Mutex::new(seen_ids),
            },
            scan.events,
        ))
    }

    fn claim_event_id(&self, event_id: &BrainEventId) -> bool {
        self.seen_ids
            .lock()
            .map(|mut ids| ids.insert(event_id.clone()))
            .unwrap_or(false)
    }

    fn release_event_id(&self, event_id: &BrainEventId) {
        if let Ok(mut ids) = self.seen_ids.lock() {
            ids.remove(event_id);
        }
    }

    fn enqueue(&self, envelope: SequencedBrainEvent) {
        self.stats.backlog.fetch_add(1, Ordering::Relaxed);
        let result = self
            .sender
            .lock()
            .ok()
            .and_then(|sender| sender.as_ref().cloned())
            .ok_or(std::sync::mpsc::TrySendError::Disconnected(
                DurableWriterCommand::Append(envelope.clone()),
            ))
            .and_then(|sender| sender.try_send(DurableWriterCommand::Append(envelope)));
        if result.is_err() {
            self.stats.backlog.fetch_sub(1, Ordering::Relaxed);
            self.stats.write_failures.fetch_add(1, Ordering::Relaxed);
            self.stats.gaps.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn read_events(&self) -> io::Result<Vec<SequencedBrainEvent>> {
        Ok(scan_durable_history(&self.config, false)?.events)
    }

    fn close(&self) {
        if let Ok(mut sender) = self.sender.lock() {
            sender.take();
        }
    }

    fn shutdown(&self) {
        self.close();
        if let Some(writer) = self.writer.lock().ok().and_then(|mut writer| writer.take()) {
            let _ = writer.join();
        }
    }

    fn backlog(&self) -> usize {
        self.stats.backlog.load(Ordering::Relaxed)
    }

    fn write_failures(&self) -> u64 {
        self.stats.write_failures.load(Ordering::Relaxed)
    }

    fn last_durable_sequence(&self) -> Option<u64> {
        let sequence = self.stats.last_durable_sequence.load(Ordering::Acquire);
        (sequence > 0).then_some(sequence)
    }

    fn gaps(&self) -> u64 {
        self.stats.gaps.load(Ordering::Relaxed)
    }

    fn recovered_records(&self) -> u64 {
        self.stats.recovered_records.load(Ordering::Relaxed)
    }

    fn rotations(&self) -> u64 {
        self.stats.rotations.load(Ordering::Relaxed)
    }
}

impl Drop for DurableBrainEventStore {
    fn drop(&mut self) {
        self.close();
        if let Some(writer) = self.writer.get_mut().ok().and_then(Option::take) {
            let _ = writer.join();
        }
    }
}

fn run_durable_writer(
    config: BrainEventDurabilityConfig,
    stats: Arc<DurableBrainEventStats>,
    receiver: std::sync::mpsc::Receiver<DurableWriterCommand>,
) {
    let mut successful_records = 0_u64;
    while let Ok(command) = receiver.recv() {
        stats.backlog.fetch_sub(1, Ordering::Relaxed);
        let DurableWriterCommand::Append(envelope) = command;
        let injected_failure = config
            .injected_failure_after_records
            .is_some_and(|limit| successful_records >= limit);
        let result = if injected_failure {
            Err(io::Error::other("injected durable writer failure"))
        } else {
            append_durable_record(&config, &stats, envelope.clone())
        };
        match result {
            Ok(()) => {
                successful_records = successful_records.saturating_add(1);
                stats
                    .last_durable_sequence
                    .store(envelope.sequence, Ordering::Release);
            }
            Err(_) => {
                stats.write_failures.fetch_add(1, Ordering::Relaxed);
                stats.gaps.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

fn append_durable_record(
    config: &BrainEventDurabilityConfig,
    stats: &DurableBrainEventStats,
    envelope: SequencedBrainEvent,
) -> io::Result<()> {
    use std::io::Write;

    let record = DurableBrainEventRecord::new(envelope)?;
    let mut bytes = serde_json::to_vec(&record).map_err(io::Error::other)?;
    bytes.push(b'\n');
    let current_len = fs::metadata(&config.path).map_or(0, |metadata| metadata.len());
    if current_len > 0 && current_len.saturating_add(bytes.len() as u64) > config.max_segment_bytes
    {
        rotate_durable_history(config)?;
        stats.rotations.fetch_add(1, Ordering::Relaxed);
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&config.path)?;
    file.write_all(&bytes)?;
    file.flush()?;
    if config.sync_data {
        file.sync_data()?;
    }
    Ok(())
}

fn durable_segment_path(path: &Path, index: usize) -> PathBuf {
    PathBuf::from(format!("{}.{}", path.display(), index))
}

fn rotate_durable_history(config: &BrainEventDurabilityConfig) -> io::Result<()> {
    let oldest = durable_segment_path(&config.path, config.retained_segments);
    if oldest.exists() {
        fs::remove_file(oldest)?;
    }
    for index in (2..=config.retained_segments).rev() {
        let from = durable_segment_path(&config.path, index - 1);
        if from.exists() {
            fs::rename(from, durable_segment_path(&config.path, index))?;
        }
    }
    if config.path.exists() {
        fs::rename(&config.path, durable_segment_path(&config.path, 1))?;
    }
    Ok(())
}

#[derive(Default)]
struct DurableHistoryScan {
    events: Vec<SequencedBrainEvent>,
    newest_sequence: u64,
    recovery_gaps: u64,
}

fn scan_durable_history(
    config: &BrainEventDurabilityConfig,
    repair_active_tail: bool,
) -> io::Result<DurableHistoryScan> {
    let mut scan = DurableHistoryScan::default();
    let mut seen_ids = HashSet::new();
    let mut by_sequence = BTreeMap::new();
    let paths = (1..=config.retained_segments)
        .rev()
        .map(|index| durable_segment_path(&config.path, index))
        .chain(std::iter::once(config.path.clone()));
    for path in paths.filter(|path| path.exists()) {
        let bytes = fs::read(&path)?;
        let mut valid_len = 0_usize;
        let mut invalid_tail = false;
        for line in bytes.split_inclusive(|byte| *byte == b'\n') {
            if line.last() != Some(&b'\n') {
                invalid_tail = true;
                break;
            }
            let record = serde_json::from_slice::<DurableBrainEventRecord>(
                &line[..line.len().saturating_sub(1)],
            );
            let Ok(record) = record else {
                invalid_tail = true;
                break;
            };
            if !record.validate() {
                invalid_tail = true;
                break;
            }
            valid_len = valid_len.saturating_add(line.len());
            let event_id = record.envelope.event.event_id.clone();
            if seen_ids.insert(event_id) {
                by_sequence
                    .entry(record.envelope.sequence)
                    .or_insert(record.envelope);
            }
        }
        if invalid_tail {
            scan.recovery_gaps = scan.recovery_gaps.saturating_add(1);
            if repair_active_tail && path == config.path {
                fs::OpenOptions::new()
                    .write(true)
                    .open(&path)?
                    .set_len(valid_len as u64)?;
            }
        }
    }
    scan.events = by_sequence.into_values().collect();
    scan.newest_sequence = scan
        .events
        .last()
        .map_or(0, |event| event.sequence);
    Ok(scan)
}
