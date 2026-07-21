const IMU_FRESH_MS: TimeMs = 250;
const IMU_FUTURE_TOLERANCE_MS: TimeMs = 50;
const IMU_MIN_CLOCK_CONFIDENCE: f32 = 0.5;
const IMU_MIN_ORIENTATION_CONFIDENCE: f32 = 0.5;
const IMU_SWITCH_CONFIRMATIONS: u8 = 3;
const IMU_CONFIDENCE_SWITCH_MARGIN: f32 = 0.15;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImuSourceOverride {
    #[default]
    Auto,
    Force(String),
    Disabled,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ImuCandidateMetadata {
    pub source_id: String,
    pub healthy: bool,
    pub clock_confidence: f32,
    pub clock_source: Option<String>,
    pub source_epoch: u64,
    pub reported_sample_age_ms: Option<u32>,
    pub supported_axes: Vec<String>,
    pub provenance: String,
}

#[derive(Clone, Debug, Default)]
struct ImuCandidateState {
    sample: Option<ImuSense>,
    metadata: Option<ImuCandidateMetadata>,
    history: VecDeque<ImuSense>,
    available: bool,
    last_seen_ms: TimeMs,
    upstream_epoch: u64,
    rejection: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ImuSelection {
    pub selected: Option<ImuSense>,
    pub selected_source: Option<String>,
    pub source_epoch: u64,
    pub source_changed: bool,
    pub switch_reason: Option<String>,
    pub diagnostics: serde_json::Value,
}

/// Deterministic Motherbrain IMU discovery and arbitration.
///
/// Every producer retains an isolated history. Only the selected producer is
/// copied into `NowBuilder`'s canonical interpolation history. Mandatory trust
/// gates are never bypassed by an override. A healthy current source is held
/// unless a challenger is materially better for three consecutive evaluations;
/// otherwise equivalent sources deterministically prefer the brainstem.
#[derive(Clone, Debug, Default)]
pub struct ImuArbiter {
    candidates: BTreeMap<String, ImuCandidateState>,
    selected_source: Option<String>,
    previous_source: Option<String>,
    source_epoch: u64,
    last_switch_ms: TimeMs,
    last_switch_reason: Option<String>,
    pending_source: Option<String>,
    pending_count: u8,
    last_reported_source_epoch: u64,
    source_override: ImuSourceOverride,
}

impl ImuArbiter {
    pub fn new(source_override: ImuSourceOverride) -> Self {
        Self {
            source_override,
            ..Self::default()
        }
    }

    pub fn set_override(&mut self, source_override: ImuSourceOverride) {
        self.source_override = source_override;
    }

    pub fn observe(&mut self, sample: ImuSense, host_now_ms: TimeMs) {
        let Some(source_id) = sample.source_id().map(ToOwned::to_owned) else {
            return;
        };
        let source_epoch = sample.source_epoch();
        let supported_axes = supported_axes(&sample);
        let provenance = sample.orientation_source.clone().unwrap_or_default();
        self.observe_with_metadata(
            sample,
            ImuCandidateMetadata {
                source_id,
                healthy: true,
                clock_confidence: 1.0,
                clock_source: Some("motherbrain_host_clock".to_string()),
                source_epoch,
                reported_sample_age_ms: None,
                supported_axes,
                provenance,
            },
            host_now_ms,
        );
    }

    pub fn observe_with_metadata(
        &mut self,
        sample: ImuSense,
        metadata: ImuCandidateMetadata,
        host_now_ms: TimeMs,
    ) {
        let source_id = metadata.source_id.clone();
        if source_id.is_empty() || sample.source_id() != Some(source_id.as_str()) {
            return;
        }
        let sample_source_epoch = metadata.source_epoch;
        let selected_epoch_changed = {
            let state = self.candidates.entry(source_id.clone()).or_default();
            state.available = true;
            state.last_seen_ms = host_now_ms;
            let epoch_changed =
                state.sample.is_some() && state.upstream_epoch != sample_source_epoch;
            if epoch_changed {
                state.history.clear();
                state.rejection =
                    Some("source clock epoch changed; history rebuilding".to_string());
            }
            state.upstream_epoch = sample_source_epoch;
            if state
                .sample
                .as_ref()
                .is_some_and(|last| sample.captured_at_ms < last.captured_at_ms)
            {
                state.rejection = Some("out-of-order sample rejected".to_string());
                return;
            }
            if state
                .history
                .back()
                .is_some_and(|last| last.captured_at_ms == sample.captured_at_ms)
            {
                state.history.pop_back();
            }
            state.history.push_back(sample.clone());
            while state.history.len() > FUSION_HISTORY_LIMIT {
                state.history.pop_front();
            }
            state.sample = Some(sample);
            state.metadata = Some(metadata);
            state.rejection = None;
            epoch_changed && self.selected_source.as_deref() == Some(source_id.as_str())
        };
        if selected_epoch_changed {
            self.previous_source = self.selected_source.take();
            self.advance_epoch(
                host_now_ms,
                format!("{source_id} clock epoch changed; canonical history invalidated"),
            );
        }
    }

    pub fn observe_unavailable(
        &mut self,
        source_id: impl Into<String>,
        reason: impl Into<String>,
        host_now_ms: TimeMs,
    ) {
        let source_id = source_id.into();
        let state = self.candidates.entry(source_id.clone()).or_default();
        state.available = false;
        state.last_seen_ms = host_now_ms;
        state.rejection = Some(reason.into());
        if self.selected_source.as_deref() == Some(source_id.as_str()) {
            self.previous_source = self.selected_source.take();
            self.advance_epoch(
                host_now_ms,
                format!("selected source {source_id} disappeared; history invalidated"),
            );
        }
    }

    pub fn arbitrate_packets(
        &mut self,
        packets: &mut Vec<SensePacket>,
        host_now_ms: TimeMs,
    ) -> ImuSelection {
        let mut non_imu = Vec::with_capacity(packets.len());
        for packet in packets.drain(..) {
            match packet {
                SensePacket::Imu(sample) => self.observe(sample, host_now_ms),
                packet => non_imu.push(packet),
            }
        }
        *packets = non_imu;
        let selection = self.select(host_now_ms);
        if let Some(sample) = selection.selected.clone() {
            packets.push(SensePacket::Imu(sample));
        }
        selection
    }

    pub fn select(&mut self, host_now_ms: TimeMs) -> ImuSelection {
        let evaluations = self
            .candidates
            .iter()
            .map(|(source, state)| {
                (
                    source.clone(),
                    candidate_rejection(source, state, host_now_ms, &self.source_override),
                )
            })
            .collect::<BTreeMap<_, _>>();

        if matches!(self.source_override, ImuSourceOverride::Disabled) {
            if self.selected_source.take().is_some() {
                self.advance_epoch(host_now_ms, "fusion IMU disabled by override".to_string());
            }
        } else if self
            .selected_source
            .as_ref()
            .is_some_and(|source| evaluations.get(source).is_some_and(Option::is_some))
        {
            let source = self.selected_source.take().unwrap_or_default();
            self.previous_source = Some(source.clone());
            self.advance_epoch(
                host_now_ms,
                format!("selected source {source} no longer clears trust gates"),
            );
        }

        let best = evaluations
            .iter()
            .filter(|(_, rejection)| rejection.is_none())
            .filter_map(|(source, _)| {
                let sample = self.candidates.get(source)?.sample.as_ref()?;
                Some((source.clone(), sample.clone()))
            })
            .max_by(|(left_source, left), (right_source, right)| {
                left.orientation_confidence
                    .total_cmp(&right.orientation_confidence)
                    .then_with(|| {
                        source_preference(left_source).cmp(&source_preference(right_source))
                    })
            });

        match (self.selected_source.clone(), best) {
            (None, Some((source, _))) => {
                if self.previous_source.is_some() {
                    if self.pending_source.as_deref() == Some(source.as_str()) {
                        self.pending_count = self.pending_count.saturating_add(1);
                    } else {
                        self.pending_source = Some(source.clone());
                        self.pending_count = 1;
                    }
                    if self.pending_count >= IMU_SWITCH_CONFIRMATIONS {
                        self.select_source(
                            host_now_ms,
                            source,
                            format!(
                                "replacement cleared trust gates for {IMU_SWITCH_CONFIRMATIONS} evaluations"
                            ),
                        );
                    }
                } else {
                    self.select_source(
                        host_now_ms,
                        source,
                        "best trustworthy discovered candidate",
                    );
                }
            }
            (Some(current), Some((challenger, challenger_sample))) if current != challenger => {
                let current_sample = self
                    .candidates
                    .get(&current)
                    .and_then(|state| state.sample.as_ref());
                let materially_better = current_sample.is_none_or(|current_sample| {
                    challenger_sample.orientation_confidence
                        >= current_sample.orientation_confidence + IMU_CONFIDENCE_SWITCH_MARGIN
                        || (challenger.starts_with("brainstem")
                            && !current.starts_with("brainstem")
                            && (challenger_sample.orientation_confidence
                                - current_sample.orientation_confidence)
                                .abs()
                                < IMU_CONFIDENCE_SWITCH_MARGIN)
                });
                if materially_better {
                    if self.pending_source.as_deref() == Some(challenger.as_str()) {
                        self.pending_count = self.pending_count.saturating_add(1);
                    } else {
                        self.pending_source = Some(challenger.clone());
                        self.pending_count = 1;
                    }
                    if self.pending_count >= IMU_SWITCH_CONFIRMATIONS {
                        self.select_source(
                            host_now_ms,
                            challenger,
                            format!("candidate remained materially better for {IMU_SWITCH_CONFIRMATIONS} evaluations"),
                        );
                    }
                } else {
                    self.pending_source = None;
                    self.pending_count = 0;
                }
            }
            _ => {
                self.pending_source = None;
                self.pending_count = 0;
            }
        }

        let selected = self.selected_source.as_ref().and_then(|source| {
            self.candidates
                .get(source)
                .filter(|state| state.history.len() >= 2)
                .and_then(|state| state.sample.clone())
        });
        let diagnostics = self.diagnostics(host_now_ms, &evaluations);
        let source_changed = self.source_epoch != self.last_reported_source_epoch;
        self.last_reported_source_epoch = self.source_epoch;
        ImuSelection {
            selected,
            selected_source: self.selected_source.clone(),
            source_epoch: self.source_epoch,
            source_changed,
            switch_reason: self.last_switch_reason.clone(),
            diagnostics,
        }
    }

    fn select_source(&mut self, now_ms: TimeMs, source: String, reason: impl Into<String>) {
        if self.selected_source.as_deref() == Some(source.as_str()) {
            return;
        }
        if let Some(previous) = self.selected_source.replace(source.clone()) {
            self.previous_source = Some(previous);
        }
        self.advance_epoch(now_ms, format!("selected {source}: {}", reason.into()));
        self.pending_source = None;
        self.pending_count = 0;
    }

    fn advance_epoch(&mut self, now_ms: TimeMs, reason: String) {
        self.source_epoch = self.source_epoch.saturating_add(1);
        self.last_switch_ms = now_ms;
        self.last_switch_reason = Some(reason);
    }

    fn diagnostics(
        &self,
        now_ms: TimeMs,
        evaluations: &BTreeMap<String, Option<String>>,
    ) -> serde_json::Value {
        let candidates = self
            .candidates
            .iter()
            .map(|(source, state)| {
                let sample = state.sample.as_ref();
                let metadata = state.metadata.as_ref();
                serde_json::json!({
                    "source_id": source,
                    "available": state.available,
                    "healthy": metadata.is_some_and(|metadata| metadata.healthy),
                    "last_sample_timestamp_ms": sample.map(|sample| sample.captured_at_ms),
                    "sample": sample,
                    "sample_age_ms": sample.map(|sample| now_ms.saturating_sub(sample.captured_at_ms)),
                    "producer_reported_sample_age_ms": metadata.and_then(|metadata| metadata.reported_sample_age_ms),
                    "fresh": sample.is_some_and(|sample| sample.captured_at_ms <= now_ms.saturating_add(IMU_FUTURE_TOLERANCE_MS) && now_ms.saturating_sub(sample.captured_at_ms) <= IMU_FRESH_MS),
                    "clock_confidence": metadata.map(|metadata| metadata.clock_confidence),
                    "clock_source": metadata.and_then(|metadata| metadata.clock_source.as_deref()),
                    "orientation_confidence": sample.map(|sample| sample.orientation_confidence),
                    "mounting_calibrated": sample.is_some_and(|sample| sample.mounting_calibrated),
                    "gyro_bias_calibrated": sample.is_some_and(|sample| sample.gyro_bias_calibrated),
                    "supported_axes": metadata.map(|metadata| &metadata.supported_axes),
                    "provenance": metadata.map(|metadata| metadata.provenance.as_str()),
                    "history_samples": state.history.len(),
                    "rejection_reason": evaluations.get(source).cloned().flatten().or_else(|| state.rejection.clone()),
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "policy": self.source_override,
            "candidates": candidates,
            "selected_source": self.selected_source,
            "selected_because": self.selected_source.as_ref().map(|_| "clears mandatory trust gates and wins deterministic ranking"),
            "source_epoch": self.source_epoch,
            "last_switch_ms": self.last_switch_ms,
            "last_switch_reason": self.last_switch_reason,
            "previous_source": self.previous_source,
            "pending_source": self.pending_source,
            "pending_confirmations": self.pending_count,
            "kinect_history_ready": self.selected_source.as_ref().and_then(|source| self.candidates.get(source)).is_some_and(|state| state.history.len() >= 2),
        })
    }
}

fn candidate_rejection(
    source: &str,
    state: &ImuCandidateState,
    now_ms: TimeMs,
    source_override: &ImuSourceOverride,
) -> Option<String> {
    if matches!(source_override, ImuSourceOverride::Disabled) {
        return Some("fusion IMU disabled".to_string());
    }
    if let ImuSourceOverride::Force(forced) = source_override {
        if forced != source {
            return Some(format!("diagnostic override forces {forced}"));
        }
    }
    if !state.available {
        return Some("candidate unavailable".to_string());
    }
    let Some(sample) = state.sample.as_ref() else {
        return Some("no complete sample".to_string());
    };
    let Some(metadata) = state.metadata.as_ref() else {
        return Some("producer discovery metadata is missing".to_string());
    };
    if !metadata.healthy {
        return Some("producer reports unhealthy sample".to_string());
    }
    if sample.captured_at_ms > now_ms.saturating_add(IMU_FUTURE_TOLERANCE_MS) {
        return Some("sample timestamp is in the future".to_string());
    }
    if now_ms.saturating_sub(sample.captured_at_ms) > IMU_FRESH_MS {
        return Some("sample is stale".to_string());
    }
    if metadata.clock_confidence < IMU_MIN_CLOCK_CONFIDENCE {
        return Some("clock alignment confidence is inadequate".to_string());
    }
    if !sample.mounting_calibrated {
        return Some("IMU-to-base mounting is uncalibrated".to_string());
    }
    if !sample.gyro_bias_calibrated {
        return Some("stationary gyro bias is uncalibrated".to_string());
    }
    if sample.orientation_confidence < IMU_MIN_ORIENTATION_CONFIDENCE {
        return Some("orientation confidence is inadequate".to_string());
    }
    if sample.orientation.len() != 2 && sample.orientation.len() != 3 {
        return Some("roll/pitch orientation fields are incomplete".to_string());
    }
    if sample.acceleration.len() != 3 || sample.angular_velocity.len() != 3 {
        return Some("acceleration or gyro axes are incomplete".to_string());
    }
    if !pete_now::trusted_imu_orientation(sample).trusted {
        return Some("roll/pitch are implausible or untrusted".to_string());
    }
    None
}

fn source_preference(source: &str) -> u8 {
    u8::from(source.starts_with("brainstem"))
}

fn supported_axes(sample: &ImuSense) -> Vec<String> {
    let mut axes = Vec::new();
    if sample.orientation.len() >= 2 {
        axes.extend(["roll".to_string(), "pitch".to_string()]);
    }
    if sample.orientation.len() == 3 {
        axes.push("yaw".to_string());
    }
    if sample.angular_velocity.len() == 3 {
        axes.push("gyro_xyz".to_string());
    }
    if sample.acceleration.len() == 3 {
        axes.push("accel_xyz".to_string());
    }
    axes
}
