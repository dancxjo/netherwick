use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatencyTrustState {
    #[default]
    Unobservable,
    Estimating,
    Trusted,
    Degraded,
    Invalidated,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LatencyEventFeature {
    pub name: String,
    pub value: f32,
    pub occurred_at_ms: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SensorTimingObservation {
    pub producer_time_ms: Option<u64>,
    pub receive_time_ms: u64,
    pub canonical_frame_time_ms: u64,
    pub clock_epoch: u64,
    pub clock_confidence: f32,
    pub event_features: Vec<LatencyEventFeature>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LatencyDistribution {
    pub median_ms: f32,
    pub p95_ms: f32,
    pub jitter_ms: f32,
    pub uncertainty_ms: f32,
    pub sample_count: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamLatencyCalibration {
    pub stream: String,
    pub trust_state: LatencyTrustState,
    pub epoch: u64,
    pub epoch_started_at_ms: u64,
    pub invalidated_at_ms: Option<u64>,
    pub invalidation_reason: Option<String>,
    pub transport_latency: Option<LatencyDistribution>,
    pub correlated_offset: Option<LatencyDistribution>,
    pub confidence: f32,
    pub evidence_count: u64,
    pub correlated_event_count: u64,
    pub rejected_count: u64,
    pub last_observed_at_ms: Option<u64>,
    pub last_observation: Option<SensorTimingObservation>,
    pub rejection_reasons: Vec<String>,
    pub optional: bool,
    pub enabled: bool,
}

impl StreamLatencyCalibration {
    fn new(stream: String, optional: bool, timestamp_ms: u64) -> Self {
        Self {
            stream,
            trust_state: LatencyTrustState::Unobservable,
            epoch: 0,
            epoch_started_at_ms: timestamp_ms,
            invalidated_at_ms: None,
            invalidation_reason: None,
            transport_latency: None,
            correlated_offset: None,
            confidence: 0.0,
            evidence_count: 0,
            correlated_event_count: 0,
            rejected_count: 0,
            last_observed_at_ms: None,
            last_observation: None,
            rejection_reasons: vec!["producer timing evidence is unavailable".to_string()],
            optional,
            enabled: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LatencyCalibrationConfig {
    pub window_size: usize,
    pub minimum_samples: usize,
    pub minimum_correlated_events: usize,
    pub minimum_clock_confidence: f32,
    pub maximum_uncertainty_ms: f32,
    pub drift_threshold_ms: f32,
    pub stale_after_ms: u64,
    pub correlation_window_ms: u64,
}

impl Default for LatencyCalibrationConfig {
    fn default() -> Self {
        Self {
            window_size: 96,
            minimum_samples: 12,
            minimum_correlated_events: 4,
            minimum_clock_confidence: 0.5,
            maximum_uncertainty_ms: 25.0,
            drift_threshold_ms: 40.0,
            stale_after_ms: 2_000,
            correlation_window_ms: 750,
        }
    }
}

#[derive(Clone, Debug)]
struct StreamEstimator {
    state: StreamLatencyCalibration,
    transport_samples_ms: VecDeque<f32>,
    correlated_offsets_ms: VecDeque<f32>,
    baseline_median_ms: Option<f32>,
    last_clock_epoch: Option<u64>,
}

impl StreamEstimator {
    fn new(stream: String, optional: bool, timestamp_ms: u64) -> Self {
        Self {
            state: StreamLatencyCalibration::new(stream, optional, timestamp_ms),
            transport_samples_ms: VecDeque::new(),
            correlated_offsets_ms: VecDeque::new(),
            baseline_median_ms: None,
            last_clock_epoch: None,
        }
    }

    fn invalidate(&mut self, timestamp_ms: u64, reason: String) {
        self.state.epoch = self.state.epoch.saturating_add(1);
        self.state.epoch_started_at_ms = timestamp_ms;
        self.state.invalidated_at_ms = Some(timestamp_ms);
        self.state.invalidation_reason = Some(reason.clone());
        self.state.trust_state = LatencyTrustState::Invalidated;
        self.state.confidence = 0.0;
        self.state.rejection_reasons = vec![reason];
        self.transport_samples_ms.clear();
        self.correlated_offsets_ms.clear();
        self.baseline_median_ms = None;
    }

    fn observe(
        &mut self,
        observation: SensorTimingObservation,
        reference_events: &BTreeMap<String, VecDeque<u64>>,
        config: &LatencyCalibrationConfig,
    ) {
        self.state.enabled = true;
        self.state.last_observed_at_ms = Some(observation.receive_time_ms);
        self.state.last_observation = Some(observation.clone());
        if self
            .last_clock_epoch
            .is_some_and(|epoch| epoch != observation.clock_epoch)
        {
            self.invalidate(
                observation.receive_time_ms,
                format!(
                    "producer clock epoch changed from {} to {}",
                    self.last_clock_epoch.unwrap_or_default(),
                    observation.clock_epoch
                ),
            );
        }
        self.last_clock_epoch = Some(observation.clock_epoch);

        let Some(producer_time_ms) = observation.producer_time_ms else {
            self.state.rejected_count = self.state.rejected_count.saturating_add(1);
            self.refresh(observation.receive_time_ms, config);
            return;
        };
        if observation.clock_confidence < config.minimum_clock_confidence {
            self.state.rejected_count = self.state.rejected_count.saturating_add(1);
            self.refresh(observation.receive_time_ms, config);
            return;
        }
        let latency_ms = observation.receive_time_ms as i128 - producer_time_ms as i128;
        if !(0..=60_000).contains(&latency_ms) {
            self.state.rejected_count = self.state.rejected_count.saturating_add(1);
            self.refresh(observation.receive_time_ms, config);
            return;
        }
        push_bounded(
            &mut self.transport_samples_ms,
            latency_ms as f32,
            config.window_size,
        );
        self.state.evidence_count = self.state.evidence_count.saturating_add(1);

        for feature in &observation.event_features {
            let Some(references) = reference_events.get(&feature.name) else {
                continue;
            };
            let Some(reference_time_ms) = references.iter().rev().copied().find(|reference| {
                reference.abs_diff(feature.occurred_at_ms) <= config.correlation_window_ms
            }) else {
                continue;
            };
            push_bounded(
                &mut self.correlated_offsets_ms,
                (feature.occurred_at_ms as i128 - reference_time_ms as i128) as f32,
                config.window_size,
            );
            self.state.correlated_event_count = self.state.correlated_event_count.saturating_add(1);
        }

        let current = distribution(&self.transport_samples_ms);
        if self.transport_samples_ms.len() >= config.minimum_samples {
            if let Some(baseline) = self.baseline_median_ms {
                let recent = recent_median(&self.transport_samples_ms, 8);
                if (recent - baseline).abs() > config.drift_threshold_ms {
                    self.invalidate(
                        observation.receive_time_ms,
                        format!(
                            "latency median shifted by {:.1} ms (threshold {:.1} ms)",
                            (recent - baseline).abs(),
                            config.drift_threshold_ms
                        ),
                    );
                    push_bounded(
                        &mut self.transport_samples_ms,
                        latency_ms as f32,
                        config.window_size,
                    );
                    self.state.evidence_count = self.state.evidence_count.saturating_add(1);
                    self.refresh(observation.receive_time_ms, config);
                    return;
                }
            } else {
                self.baseline_median_ms = current.as_ref().map(|value| value.median_ms);
            }
        }
        self.refresh(observation.receive_time_ms, config);
    }

    fn refresh(&mut self, now_ms: u64, config: &LatencyCalibrationConfig) {
        self.state.transport_latency = distribution(&self.transport_samples_ms);
        self.state.correlated_offset = distribution(&self.correlated_offsets_ms);
        if self.state.trust_state == LatencyTrustState::Invalidated
            && self.state.invalidated_at_ms == Some(now_ms)
        {
            return;
        }
        self.state.rejection_reasons.clear();
        let fresh = self
            .state
            .last_observed_at_ms
            .is_some_and(|last| now_ms.saturating_sub(last) <= config.stale_after_ms);
        let enough_samples = self.transport_samples_ms.len() >= config.minimum_samples;
        let enough_events = self.correlated_offsets_ms.len() >= config.minimum_correlated_events;
        let uncertainty = self
            .state
            .transport_latency
            .as_ref()
            .map(|value| value.uncertainty_ms)
            .unwrap_or(f32::INFINITY);
        let clock_confidence = self
            .state
            .last_observation
            .as_ref()
            .map(|value| value.clock_confidence.clamp(0.0, 1.0))
            .unwrap_or(0.0);
        let sample_confidence = (self.transport_samples_ms.len() as f32
            / config.minimum_samples.max(1) as f32)
            .clamp(0.0, 1.0);
        let event_confidence = (self.correlated_offsets_ms.len() as f32
            / config.minimum_correlated_events.max(1) as f32)
            .clamp(0.0, 1.0);
        self.state.confidence = sample_confidence
            * clock_confidence
            * if enough_events {
                1.0
            } else {
                0.7 + 0.3 * event_confidence
            };
        self.state.trust_state = if !fresh && self.state.last_observed_at_ms.is_some() {
            LatencyTrustState::Degraded
        } else if enough_samples && uncertainty <= config.maximum_uncertainty_ms {
            LatencyTrustState::Trusted
        } else if self.transport_samples_ms.is_empty() {
            LatencyTrustState::Unobservable
        } else {
            LatencyTrustState::Estimating
        };
        if !fresh {
            self.state
                .rejection_reasons
                .push("timing evidence is stale".to_string());
        }
        if self.transport_samples_ms.is_empty() {
            self.state.rejection_reasons.push(
                "producer time is missing, future, implausible, or low-confidence".to_string(),
            );
        } else if !enough_samples {
            self.state
                .rejection_reasons
                .push("latency distribution is still estimating".to_string());
        }
        if !enough_events {
            self.state
                .rejection_reasons
                .push("correlated event evidence is sparse".to_string());
        }
        if uncertainty > config.maximum_uncertainty_ms {
            self.state.rejection_reasons.push(format!(
                "latency uncertainty {:.1} ms exceeds {:.1} ms",
                uncertainty, config.maximum_uncertainty_ms
            ));
        }
    }
}

#[derive(Clone, Debug)]
pub struct SensorLatencyRegistry {
    pub config: LatencyCalibrationConfig,
    streams: BTreeMap<String, StreamEstimator>,
    reference_events: BTreeMap<String, VecDeque<u64>>,
}

impl Default for SensorLatencyRegistry {
    fn default() -> Self {
        let mut registry = Self {
            config: LatencyCalibrationConfig::default(),
            streams: BTreeMap::new(),
            reference_events: BTreeMap::new(),
        };
        registry.declare_stream("lidar", true, 0);
        registry
    }
}

impl SensorLatencyRegistry {
    pub fn declare_stream(&mut self, stream: &str, optional: bool, now_ms: u64) {
        self.streams
            .entry(stream.to_string())
            .or_insert_with(|| StreamEstimator::new(stream.to_string(), optional, now_ms));
    }

    pub fn observe_reference_event(&mut self, name: &str, occurred_at_ms: u64) {
        let events = self.reference_events.entry(name.to_string()).or_default();
        events.push_back(occurred_at_ms);
        while events.len() > self.config.window_size {
            events.pop_front();
        }
    }

    pub fn observe(&mut self, stream: &str, observation: SensorTimingObservation) {
        self.declare_stream(stream, stream == "lidar", observation.receive_time_ms);
        if let Some(estimator) = self.streams.get_mut(stream) {
            estimator.observe(observation, &self.reference_events, &self.config);
        }
    }

    pub fn snapshot(&mut self, now_ms: u64) -> BTreeMap<String, StreamLatencyCalibration> {
        self.streams
            .iter_mut()
            .map(|(name, estimator)| {
                estimator.refresh(now_ms, &self.config);
                (name.clone(), estimator.state.clone())
            })
            .collect()
    }

    pub fn validate_held_out_event(
        &self,
        stream: &str,
        producer_event_time_ms: u64,
        reference_event_time_ms: u64,
        tolerance_ms: f32,
    ) -> Result<f32, String> {
        let state = self
            .streams
            .get(stream)
            .map(|estimator| &estimator.state)
            .ok_or_else(|| format!("unknown stream {stream}"))?;
        let estimated_offset = state
            .correlated_offset
            .as_ref()
            .map(|value| value.median_ms)
            .ok_or_else(|| format!("stream {stream} has no correlated offset"))?;
        let observed_offset = producer_event_time_ms as i128 - reference_event_time_ms as i128;
        let error_ms = (observed_offset as f32 - estimated_offset).abs();
        if error_ms <= tolerance_ms {
            Ok(error_ms)
        } else {
            Err(format!(
                "held-out offset error {error_ms:.1} ms exceeds tolerance {tolerance_ms:.1} ms"
            ))
        }
    }
}

fn push_bounded(samples: &mut VecDeque<f32>, value: f32, limit: usize) {
    samples.push_back(value);
    while samples.len() > limit.max(1) {
        samples.pop_front();
    }
}

fn recent_median(samples: &VecDeque<f32>, count: usize) -> f32 {
    let mut values = samples
        .iter()
        .rev()
        .take(count)
        .copied()
        .collect::<Vec<_>>();
    percentile(&mut values, 0.5)
}

fn distribution(samples: &VecDeque<f32>) -> Option<LatencyDistribution> {
    if samples.is_empty() {
        return None;
    }
    let mut values = samples.iter().copied().collect::<Vec<_>>();
    let median_ms = percentile(&mut values.clone(), 0.5);
    let p95_ms = percentile(&mut values, 0.95);
    let mut deviations = samples
        .iter()
        .map(|value| (*value - median_ms).abs())
        .collect::<Vec<_>>();
    let jitter_ms = percentile(&mut deviations, 0.5);
    let uncertainty_ms = (jitter_ms * 1.4826)
        .max(1.0 / (samples.len() as f32).sqrt())
        .max((p95_ms - median_ms).abs() * 0.25);
    Some(LatencyDistribution {
        median_ms,
        p95_ms,
        jitter_ms,
        uncertainty_ms,
        sample_count: samples.len() as u64,
    })
}

fn percentile(values: &mut [f32], quantile: f32) -> f32 {
    values.sort_by(f32::total_cmp);
    let index =
        ((values.len().saturating_sub(1)) as f32 * quantile.clamp(0.0, 1.0)).ceil() as usize;
    values[index.min(values.len().saturating_sub(1))]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(producer_ms: u64, receive_ms: u64, epoch: u64) -> SensorTimingObservation {
        SensorTimingObservation {
            producer_time_ms: Some(producer_ms),
            receive_time_ms: receive_ms,
            canonical_frame_time_ms: receive_ms,
            clock_epoch: epoch,
            clock_confidence: 0.95,
            event_features: Vec::new(),
        }
    }

    #[test]
    fn stable_latency_reports_distribution_jitter_and_confidence() {
        let mut registry = SensorLatencyRegistry::default();
        for index in 0..20 {
            let latency = [18, 20, 21, 19][index % 4];
            let receive = 1_000 + index as u64 * 100;
            registry.observe("camera", observation(receive - latency, receive, 0));
        }
        let state = registry.snapshot(3_000)["camera"].clone();
        assert_eq!(state.trust_state, LatencyTrustState::Trusted);
        let distribution = state.transport_latency.unwrap();
        assert_eq!(distribution.median_ms, 20.0);
        assert_eq!(distribution.p95_ms, 21.0);
        assert!(distribution.jitter_ms <= 1.0);
        assert!(state.confidence > 0.6);
    }

    #[test]
    fn correlated_events_replay_and_validate_held_out_offset() {
        let mut registry = SensorLatencyRegistry::default();
        for index in 0..16 {
            let reference = 1_000 + index * 100;
            registry.observe_reference_event("rotation_cw", reference);
            let mut sample = observation(reference + 30, reference + 50, 2);
            sample.event_features.push(LatencyEventFeature {
                name: "rotation_cw".to_string(),
                value: 0.5,
                occurred_at_ms: reference + 30,
            });
            registry.observe("imu", sample);
        }
        let state = registry.snapshot(2_600)["imu"].clone();
        assert_eq!(state.correlated_offset.as_ref().unwrap().median_ms, 30.0);
        assert!(registry
            .validate_held_out_event("imu", 5_032, 5_000, 3.0)
            .is_ok());
        let encoded = serde_json::to_string(&state).unwrap();
        let replayed: StreamLatencyCalibration = serde_json::from_str(&encoded).unwrap();
        assert_eq!(replayed.epoch, 0);
    }

    #[test]
    fn drift_and_clock_epoch_changes_invalidate_only_affected_stream() {
        let mut registry = SensorLatencyRegistry::default();
        registry.config.minimum_samples = 8;
        for index in 0..12 {
            let receive = 1_000 + index * 100;
            registry.observe("kinect", observation(receive - 20, receive, 4));
            registry.observe("camera", observation(receive - 20, receive, 4));
        }
        for index in 0..8 {
            let receive = 3_000 + index * 100;
            registry.observe("kinect", observation(receive - 100, receive, 4));
        }
        let states = registry.snapshot(3_700);
        assert!(states["kinect"].epoch > 0);
        assert_eq!(states["camera"].epoch, 0);
        registry.observe("camera", observation(3_790, 3_800, 5));
        let camera = registry.snapshot(3_800)["camera"].clone();
        assert_eq!(camera.epoch, 1);
        assert_eq!(camera.trust_state, LatencyTrustState::Invalidated);
    }

    #[test]
    fn missing_optional_lidar_is_not_a_failure() {
        let mut registry = SensorLatencyRegistry::default();
        let lidar = registry.snapshot(10_000)["lidar"].clone();
        assert!(lidar.optional);
        assert!(!lidar.enabled);
        assert_eq!(lidar.trust_state, LatencyTrustState::Unobservable);
    }

    #[test]
    fn stale_timing_degrades_without_blocking_observation() {
        let mut registry = SensorLatencyRegistry::default();
        registry.config.minimum_samples = 2;
        registry.observe("audio", observation(990, 1_000, 0));
        registry.observe("audio", observation(1_090, 1_100, 0));
        assert_eq!(
            registry.snapshot(1_100)["audio"].trust_state,
            LatencyTrustState::Trusted
        );
        assert_eq!(
            registry.snapshot(4_000)["audio"].trust_state,
            LatencyTrustState::Degraded
        );
    }
}
