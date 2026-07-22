use crate::calibration_transition::{consumer_impacts, CalibrationDofState};
use crate::{
    calibration_state, CalibrationClockedTime, CalibrationEvidenceWindow, CalibrationTransition,
    CalibrationTransitionKind, CalibrationTransitionState, RigidTransform3,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

pub const TRANSFORM_DOF_COUNT: usize = 6;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationTrustState {
    #[default]
    Configured,
    Estimating,
    Trusted,
    Degraded,
    Invalidated,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationEvidenceSource {
    #[default]
    FloorPlane,
    Gravity,
    WheelOdometry,
    PersistentSurface,
    MapConsistency,
    LoopClosure,
    /// Optional corroboration only. No trust rule requires this source.
    Lidar,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationResiduals {
    pub floor_m: Option<f32>,
    pub wall_m: Option<f32>,
    pub gravity_rad: Option<f32>,
    pub reprojection_px: Option<f32>,
    pub map_consistency_m: Option<f32>,
    pub loop_closure_m: Option<f32>,
}

impl CalibrationResiduals {
    fn merge(&mut self, observation: &Self) {
        for (slot, value) in [
            (&mut self.floor_m, observation.floor_m),
            (&mut self.wall_m, observation.wall_m),
            (&mut self.gravity_rad, observation.gravity_rad),
            (&mut self.reprojection_px, observation.reprojection_px),
            (&mut self.map_consistency_m, observation.map_consistency_m),
            (&mut self.loop_closure_m, observation.loop_closure_m),
        ] {
            if value.is_some() {
                *slot = value;
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationEpoch {
    pub id: u64,
    pub started_at_ms: u64,
    pub invalidated_at_ms: Option<u64>,
    pub invalidation_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LiveCalibrationEstimate {
    pub transform: RigidTransform3,
    /// Variance for `[x, y, z, roll, pitch, yaw]`.
    pub covariance: [f32; TRANSFORM_DOF_COUNT],
    pub confidence: f32,
    pub trust_state: CalibrationTrustState,
    pub observable_dofs: [bool; TRANSFORM_DOF_COUNT],
    pub evidence_counts: BTreeMap<CalibrationEvidenceSource, u32>,
    pub dof_evidence_counts: [u32; TRANSFORM_DOF_COUNT],
    pub residuals: CalibrationResiduals,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_started_at_ms: Option<u64>,
    pub updated_at_ms: u64,
    pub epoch: CalibrationEpoch,
    pub rejection_reasons: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransformEstimateEvidence {
    pub source: CalibrationEvidenceSource,
    pub captured_at_ms: u64,
    pub transform: RigidTransform3,
    pub observable_dofs: [bool; TRANSFORM_DOF_COUNT],
    /// Observation variance for `[x, y, z, roll, pitch, yaw]`.
    pub covariance: [f32; TRANSFORM_DOF_COUNT],
    #[serde(default)]
    pub residuals: CalibrationResiduals,
}

/// Interface for floor, gravity, odometry, surface, map, loop-closure, and
/// optional lidar adapters. Estimators produce observations; the state machine
/// owns trust and epoch transitions.
pub trait TransformEstimator<Input> {
    fn source(&self) -> CalibrationEvidenceSource;
    fn estimate(
        &mut self,
        input: &Input,
        current: &LiveCalibrationEstimate,
    ) -> Option<TransformEstimateEvidence>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationStateConfig {
    pub minimum_evidence_per_dof: u32,
    pub trusted_covariance: [f32; TRANSFORM_DOF_COUNT],
    pub reset_covariance: [f32; TRANSFORM_DOF_COUNT],
    pub translation_shift_m: f32,
    pub rotation_shift_rad: f32,
    pub floor_shift_m: f32,
    pub gravity_shift_rad: f32,
    pub reprojection_shift_px: f32,
    pub map_shift_m: f32,
    pub maximum_evidence_age_ms: u64,
    #[serde(default = "default_minimum_independent_sources")]
    pub minimum_independent_sources: usize,
    #[serde(default = "default_minimum_trust_span_ms")]
    pub minimum_trust_span_ms: u64,
}

const fn default_minimum_independent_sources() -> usize {
    2
}

const fn default_minimum_trust_span_ms() -> u64 {
    1_000
}

impl Default for CalibrationStateConfig {
    fn default() -> Self {
        Self {
            minimum_evidence_per_dof: 3,
            trusted_covariance: [0.0025, 0.0025, 0.0025, 0.0009, 0.0009, 0.0009],
            reset_covariance: [0.25, 0.25, 0.25, 0.09, 0.09, 0.09],
            translation_shift_m: 0.08,
            rotation_shift_rad: 7.0_f32.to_radians(),
            floor_shift_m: 0.05,
            gravity_shift_rad: 5.0_f32.to_radians(),
            reprojection_shift_px: 6.0,
            map_shift_m: 0.15,
            maximum_evidence_age_ms: 2_000,
            minimum_independent_sources: default_minimum_independent_sources(),
            minimum_trust_span_ms: default_minimum_trust_span_ms(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CalibrationStateMachine {
    configured_guess: RigidTransform3,
    pub config: CalibrationStateConfig,
    estimate: LiveCalibrationEstimate,
    dof_covariance_sources: [BTreeSet<CalibrationEvidenceSource>; TRANSFORM_DOF_COUNT],
    transitions: Vec<CalibrationTransition>,
    transition_sequence: u64,
}

impl CalibrationStateMachine {
    pub fn new(
        configured_guess: RigidTransform3,
        started_at_ms: u64,
        config: CalibrationStateConfig,
    ) -> Self {
        Self {
            configured_guess,
            estimate: LiveCalibrationEstimate {
                transform: configured_guess,
                covariance: config.reset_covariance,
                confidence: 0.0,
                trust_state: CalibrationTrustState::Configured,
                observable_dofs: [false; TRANSFORM_DOF_COUNT],
                evidence_counts: BTreeMap::new(),
                dof_evidence_counts: [0; TRANSFORM_DOF_COUNT],
                residuals: CalibrationResiduals::default(),
                evidence_started_at_ms: None,
                updated_at_ms: started_at_ms,
                epoch: CalibrationEpoch {
                    id: 0,
                    started_at_ms,
                    invalidated_at_ms: None,
                    invalidation_reason: None,
                },
                rejection_reasons: vec!["configured transform is only an initial guess".to_string()],
            },
            dof_covariance_sources: std::array::from_fn(|_| BTreeSet::new()),
            transitions: Vec::new(),
            transition_sequence: 0,
            config,
        }
    }

    pub fn estimate(&self) -> &LiveCalibrationEstimate {
        &self.estimate
    }

    pub fn take_transitions(&mut self) -> Vec<CalibrationTransition> {
        std::mem::take(&mut self.transitions)
    }

    pub fn observe(
        &mut self,
        evidence: TransformEstimateEvidence,
        now_ms: u64,
    ) -> &LiveCalibrationEstimate {
        let prior = self.estimate.clone();
        self.estimate.rejection_reasons.clear();
        if let Err(reason) = self.validate_evidence(&evidence, now_ms) {
            self.refresh_trust_state(now_ms);
            self.estimate.rejection_reasons.insert(0, reason.clone());
            self.record_transition(
                &prior,
                CalibrationTransitionKind::Rejected,
                &evidence,
                now_ms,
                Some(reason),
            );
            return &self.estimate;
        }
        if self.shift_detection_ready(&evidence) {
            if let Some(reason) = self.shift_reason(&evidence) {
                self.invalidate_epoch(&evidence, now_ms, reason.clone());
                self.record_transition(
                    &prior,
                    CalibrationTransitionKind::Remounted,
                    &evidence,
                    now_ms,
                    Some(reason),
                );
                return &self.estimate;
            }
        }

        if self.estimate.trust_state == CalibrationTrustState::Invalidated {
            self.estimate.trust_state = CalibrationTrustState::Estimating;
            self.estimate.epoch.invalidated_at_ms = None;
            self.estimate.epoch.invalidation_reason = None;
        }
        self.fuse(&evidence);
        self.estimate
            .evidence_started_at_ms
            .get_or_insert(evidence.captured_at_ms);
        self.estimate.updated_at_ms = evidence.captured_at_ms;
        *self
            .estimate
            .evidence_counts
            .entry(evidence.source)
            .or_default() += 1;
        self.estimate.residuals.merge(&evidence.residuals);
        self.refresh_trust_state(now_ms);
        let kind = transition_kind(prior.trust_state, self.estimate.trust_state);
        self.record_transition(&prior, kind, &evidence, now_ms, None);
        &self.estimate
    }

    pub fn refresh(&mut self, now_ms: u64) -> &LiveCalibrationEstimate {
        let prior = self.estimate.clone();
        self.refresh_trust_state(now_ms);
        if prior.trust_state != self.estimate.trust_state {
            let evidence = TransformEstimateEvidence {
                source: CalibrationEvidenceSource::MapConsistency,
                captured_at_ms: now_ms,
                transform: self.estimate.transform,
                observable_dofs: [false; TRANSFORM_DOF_COUNT],
                covariance: self.estimate.covariance,
                residuals: self.estimate.residuals.clone(),
            };
            let kind = transition_kind(prior.trust_state, self.estimate.trust_state);
            self.record_transition(
                &prior,
                kind,
                &evidence,
                now_ms,
                self.estimate.rejection_reasons.first().cloned(),
            );
        }
        &self.estimate
    }

    pub fn invalidate(&mut self, now_ms: u64, reason: impl Into<String>) {
        let prior = self.estimate.clone();
        let reason = reason.into();
        let seed = TransformEstimateEvidence {
            source: CalibrationEvidenceSource::MapConsistency,
            captured_at_ms: now_ms,
            transform: self.estimate.transform,
            observable_dofs: [false; TRANSFORM_DOF_COUNT],
            covariance: self.config.reset_covariance,
            residuals: CalibrationResiduals::default(),
        };
        self.invalidate_epoch(&seed, now_ms, reason.clone());
        self.record_transition(
            &prior,
            CalibrationTransitionKind::Invalidated,
            &seed,
            now_ms,
            Some(reason),
        );
    }

    fn validate_evidence(
        &self,
        evidence: &TransformEstimateEvidence,
        now_ms: u64,
    ) -> Result<(), String> {
        if now_ms.saturating_sub(evidence.captured_at_ms) > self.config.maximum_evidence_age_ms {
            return Err("calibration evidence is stale".to_string());
        }
        if evidence.captured_at_ms < self.estimate.updated_at_ms {
            return Err("calibration evidence predates the active estimate".to_string());
        }
        if !evidence.observable_dofs.iter().any(|value| *value) {
            return Err("calibration evidence observes no transform degree of freedom".to_string());
        }
        let values = transform_values(evidence.transform);
        for index in 0..TRANSFORM_DOF_COUNT {
            if evidence.observable_dofs[index]
                && (!values[index].is_finite()
                    || !evidence.covariance[index].is_finite()
                    || evidence.covariance[index] <= 0.0)
            {
                return Err(format!("calibration evidence has invalid dof {index}"));
            }
        }
        Ok(())
    }

    fn fuse(&mut self, evidence: &TransformEstimateEvidence) {
        let mut current = transform_values(self.estimate.transform);
        let observed = transform_values(evidence.transform);
        for index in 0..TRANSFORM_DOF_COUNT {
            if !evidence.observable_dofs[index] {
                continue;
            }
            let prior_variance = self.estimate.covariance[index].max(f32::EPSILON);
            let observation_variance = evidence.covariance[index].max(f32::EPSILON);
            let prior_weight = 1.0 / prior_variance;
            let observation_weight = 1.0 / observation_variance;
            let observation = if index >= 3 {
                current[index] + angle_delta(observed[index], current[index])
            } else {
                observed[index]
            };
            current[index] = (current[index] * prior_weight + observation * observation_weight)
                / (prior_weight + observation_weight);
            if index >= 3 {
                current[index] = normalize_angle(current[index]);
            }
            // Repeated frames from one estimator are correlated. Only a newly
            // independent source gets an inverse-variance covariance update;
            // repetition may report a better bound but cannot compound itself.
            self.estimate.covariance[index] =
                if self.dof_covariance_sources[index].insert(evidence.source) {
                    1.0 / (prior_weight + observation_weight)
                } else {
                    prior_variance.min(observation_variance)
                };
            self.estimate.dof_evidence_counts[index] =
                self.estimate.dof_evidence_counts[index].saturating_add(1);
            self.estimate.observable_dofs[index] = true;
        }
        self.estimate.transform = transform_from_values(current);
    }

    fn shift_reason(&self, evidence: &TransformEstimateEvidence) -> Option<String> {
        let current = transform_values(self.estimate.transform);
        let observed = transform_values(evidence.transform);
        for index in 0..TRANSFORM_DOF_COUNT {
            if !evidence.observable_dofs[index] {
                continue;
            }
            let delta = if index >= 3 {
                angle_delta(observed[index], current[index]).abs()
            } else {
                (observed[index] - current[index]).abs()
            };
            let threshold = if index >= 3 {
                self.config.rotation_shift_rad
            } else {
                self.config.translation_shift_m
            };
            if delta > threshold {
                return Some(format!(
                    "sudden transform change on dof {index}: {delta:.4}"
                ));
            }
        }
        let residuals = &evidence.residuals;
        if residuals
            .floor_m
            .is_some_and(|value| value.abs() > self.config.floor_shift_m)
        {
            return Some("sudden floor-plane residual change".to_string());
        }
        if residuals
            .gravity_rad
            .is_some_and(|value| value.abs() > self.config.gravity_shift_rad)
        {
            return Some("sudden gravity residual change".to_string());
        }
        if residuals
            .reprojection_px
            .is_some_and(|value| value.abs() > self.config.reprojection_shift_px)
        {
            return Some("sudden reprojection residual change".to_string());
        }
        if residuals
            .map_consistency_m
            .is_some_and(|value| value.abs() > self.config.map_shift_m)
        {
            return Some("sudden map-consistency residual change".to_string());
        }
        None
    }

    fn shift_detection_ready(&self, evidence: &TransformEstimateEvidence) -> bool {
        matches!(
            self.estimate.trust_state,
            CalibrationTrustState::Trusted | CalibrationTrustState::Degraded
        ) || evidence
            .observable_dofs
            .iter()
            .zip(self.estimate.dof_evidence_counts)
            .any(|(observable, count)| *observable && count >= self.config.minimum_evidence_per_dof)
    }

    fn invalidate_epoch(
        &mut self,
        evidence: &TransformEstimateEvidence,
        now_ms: u64,
        reason: String,
    ) {
        let mut seed = transform_values(self.configured_guess);
        let observed = transform_values(evidence.transform);
        for index in 0..TRANSFORM_DOF_COUNT {
            if evidence.observable_dofs[index] {
                seed[index] = observed[index];
            }
        }
        self.estimate.transform = transform_from_values(seed);
        self.estimate.covariance = self.config.reset_covariance;
        self.estimate.confidence = 0.0;
        self.estimate.trust_state = CalibrationTrustState::Invalidated;
        self.estimate.observable_dofs = [false; TRANSFORM_DOF_COUNT];
        self.estimate.evidence_counts.clear();
        self.estimate.dof_evidence_counts = [0; TRANSFORM_DOF_COUNT];
        self.dof_covariance_sources = std::array::from_fn(|_| BTreeSet::new());
        self.estimate.residuals = evidence.residuals.clone();
        self.estimate.evidence_started_at_ms = None;
        self.estimate.updated_at_ms = now_ms;
        self.estimate.epoch = CalibrationEpoch {
            id: self.estimate.epoch.id.saturating_add(1),
            started_at_ms: now_ms,
            invalidated_at_ms: Some(now_ms),
            invalidation_reason: Some(reason.clone()),
        };
        self.estimate.rejection_reasons = vec![reason];
    }

    fn refresh_trust_state(&mut self, now_ms: u64) {
        if self.estimate.trust_state == CalibrationTrustState::Invalidated {
            return;
        }
        let all_observable = self
            .estimate
            .dof_evidence_counts
            .iter()
            .all(|count| *count >= self.config.minimum_evidence_per_dof);
        let covariance_ready = self
            .estimate
            .covariance
            .iter()
            .zip(self.config.trusted_covariance)
            .all(|(actual, limit)| *actual <= limit);
        let source_diversity_ready =
            self.estimate.evidence_counts.len() >= self.config.minimum_independent_sources;
        let temporal_separation_ready = self.estimate.evidence_started_at_ms.is_some_and(|first| {
            self.estimate.updated_at_ms.saturating_sub(first) >= self.config.minimum_trust_span_ms
        });
        let evidence_fresh = !self.estimate.evidence_counts.is_empty()
            && now_ms.saturating_sub(self.estimate.updated_at_ms)
                <= self.config.maximum_evidence_age_ms;
        let evidence_stale = !self.estimate.evidence_counts.is_empty() && !evidence_fresh;
        let residual_degraded = self
            .estimate
            .residuals
            .floor_m
            .is_some_and(|v| v.abs() > 0.03)
            || self
                .estimate
                .residuals
                .wall_m
                .is_some_and(|v| v.abs() > 0.05)
            || self
                .estimate
                .residuals
                .reprojection_px
                .is_some_and(|v| v.abs() > 3.0)
            || self
                .estimate
                .residuals
                .map_consistency_m
                .is_some_and(|v| v.abs() > 0.08);
        let observable_fraction = self
            .estimate
            .observable_dofs
            .iter()
            .filter(|value| **value)
            .count() as f32
            / TRANSFORM_DOF_COUNT as f32;
        let covariance_fraction = self
            .estimate
            .covariance
            .iter()
            .zip(self.config.reset_covariance)
            .map(|(actual, reset)| 1.0 - (*actual / reset.max(f32::EPSILON)).clamp(0.0, 1.0))
            .sum::<f32>()
            / TRANSFORM_DOF_COUNT as f32;
        self.estimate.confidence = (observable_fraction * covariance_fraction).clamp(0.0, 1.0);
        self.estimate.trust_state = if residual_degraded || evidence_stale {
            CalibrationTrustState::Degraded
        } else if all_observable
            && covariance_ready
            && source_diversity_ready
            && temporal_separation_ready
        {
            CalibrationTrustState::Trusted
        } else if self.estimate.evidence_counts.is_empty() {
            CalibrationTrustState::Configured
        } else {
            CalibrationTrustState::Estimating
        };
        self.estimate.rejection_reasons.clear();
        if !all_observable {
            let missing = self
                .estimate
                .dof_evidence_counts
                .iter()
                .enumerate()
                .filter(|(_, count)| **count < self.config.minimum_evidence_per_dof)
                .map(|(index, _)| index.to_string())
                .collect::<Vec<_>>();
            self.estimate.rejection_reasons.push(format!(
                "unobservable transform degrees of freedom: {}",
                missing.join(",")
            ));
        }
        if !covariance_ready {
            self.estimate
                .rejection_reasons
                .push("transform covariance remains above trust limits".to_string());
        }
        if !source_diversity_ready {
            self.estimate.rejection_reasons.push(format!(
                "only {} independent calibration source(s); {} required",
                self.estimate.evidence_counts.len(),
                self.config.minimum_independent_sources
            ));
        }
        if !temporal_separation_ready {
            self.estimate
                .rejection_reasons
                .push("calibration evidence lacks temporal separation".to_string());
        }
        if !evidence_fresh && !self.estimate.evidence_counts.is_empty() {
            self.estimate
                .rejection_reasons
                .push("calibration evidence has aged past the freshness limit".to_string());
        }
        if residual_degraded {
            self.estimate
                .rejection_reasons
                .push("geometry residuals exceed trusted limits".to_string());
        }
    }

    fn record_transition(
        &mut self,
        prior: &LiveCalibrationEstimate,
        kind: CalibrationTransitionKind,
        evidence: &TransformEstimateEvidence,
        now_ms: u64,
        reason: Option<String>,
    ) {
        let prior_state = transform_transition_state(prior);
        let new_state = transform_transition_state(&self.estimate);
        if prior_state == new_state && kind != CalibrationTransitionKind::Rejected {
            return;
        }
        self.transition_sequence = self.transition_sequence.saturating_add(1);
        let allowed_before = prior.trust_state == CalibrationTrustState::Trusted;
        let allowed_after = self.estimate.trust_state == CalibrationTrustState::Trusted;
        let occurred = CalibrationClockedTime::new(evidence.captured_at_ms, "sensor:0");
        let observed = CalibrationClockedTime::new(now_ms, "motherbrain:0");
        let evidence_started_at_ms = self
            .estimate
            .evidence_started_at_ms
            .unwrap_or(evidence.captured_at_ms);
        self.transitions.push(CalibrationTransition::author(
            "kinect.extrinsics",
            self.estimate.epoch.id,
            self.transition_sequence,
            kind,
            prior_state,
            new_state,
            format!("{:?}", evidence.source).to_lowercase(),
            serde_json::to_value(evidence).unwrap_or_else(|_| json!({"unavailable": true})),
            CalibrationEvidenceWindow {
                started_at: CalibrationClockedTime::new(evidence_started_at_ms, "sensor:0"),
                ended_at: occurred.clone(),
                sample_count: self
                    .estimate
                    .evidence_counts
                    .values()
                    .copied()
                    .map(u64::from)
                    .sum(),
            },
            consumer_impacts(
                &[
                    "map.3d_fusion",
                    "navigation.geometry",
                    "perception.depth_projection",
                ],
                allowed_before,
                allowed_after,
                "full calibrated Kinect extrinsics are required",
            ),
            reason,
            occurred,
            observed,
        ));
    }
}

fn transition_kind(
    prior: CalibrationTrustState,
    new: CalibrationTrustState,
) -> CalibrationTransitionKind {
    match (prior, new) {
        (_, CalibrationTrustState::Trusted) if prior != CalibrationTrustState::Trusted => {
            CalibrationTransitionKind::NewlyTrusted
        }
        (_, CalibrationTrustState::Degraded) if prior != CalibrationTrustState::Degraded => {
            CalibrationTransitionKind::Degraded
        }
        _ => CalibrationTransitionKind::Accepted,
    }
}

fn transform_transition_state(estimate: &LiveCalibrationEstimate) -> CalibrationTransitionState {
    const NAMES: [&str; TRANSFORM_DOF_COUNT] = ["x", "y", "z", "roll", "pitch", "yaw"];
    let values = transform_values(estimate.transform);
    calibration_state(
        format!("{:?}", estimate.trust_state).to_lowercase(),
        estimate.confidence,
        serde_json::to_value(estimate).unwrap_or(Value::Null),
        NAMES.into_iter().enumerate().map(|(index, name)| {
            (
                name.to_string(),
                CalibrationDofState {
                    value: json!(values[index]),
                    observable: estimate.observable_dofs[index],
                    uncertainty: estimate.covariance[index]
                        .is_finite()
                        .then_some(estimate.covariance[index]),
                },
            )
        }),
    )
}

fn transform_values(transform: RigidTransform3) -> [f32; TRANSFORM_DOF_COUNT] {
    [
        transform.translation_m[0],
        transform.translation_m[1],
        transform.translation_m[2],
        transform.rotation_rpy_rad[0],
        transform.rotation_rpy_rad[1],
        transform.rotation_rpy_rad[2],
    ]
}

fn transform_from_values(values: [f32; TRANSFORM_DOF_COUNT]) -> RigidTransform3 {
    RigidTransform3 {
        translation_m: [values[0], values[1], values[2]],
        rotation_rpy_rad: [values[3], values[4], values[5]],
    }
}

fn normalize_angle(angle: f32) -> f32 {
    (angle + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

fn angle_delta(to: f32, from: f32) -> f32 {
    normalize_angle(to - from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(
        timestamp: u64,
        transform: RigidTransform3,
        observable_dofs: [bool; TRANSFORM_DOF_COUNT],
    ) -> TransformEstimateEvidence {
        TransformEstimateEvidence {
            source: CalibrationEvidenceSource::PersistentSurface,
            captured_at_ms: timestamp,
            transform,
            observable_dofs,
            covariance: [0.001; TRANSFORM_DOF_COUNT],
            residuals: CalibrationResiduals::default(),
        }
    }

    fn converge_full_transform(
        machine: &mut CalibrationStateMachine,
        transform: RigidTransform3,
        start_ms: u64,
    ) {
        for index in 0..4 {
            let timestamp = start_ms + index * 500;
            let mut item = evidence(timestamp, transform, [true; 6]);
            item.source = if index % 2 == 0 {
                CalibrationEvidenceSource::PersistentSurface
            } else {
                CalibrationEvidenceSource::MapConsistency
            };
            machine.observe(item, timestamp);
        }
    }

    #[test]
    fn trust_is_withheld_until_every_transform_dof_is_observable() {
        let mut machine = CalibrationStateMachine::new(
            RigidTransform3::default(),
            0,
            CalibrationStateConfig::default(),
        );
        let partial = [false, false, true, true, true, false];
        for timestamp in 1..=6 {
            machine.observe(
                evidence(timestamp, RigidTransform3::default(), partial),
                timestamp,
            );
        }
        assert_eq!(
            machine.estimate().trust_state,
            CalibrationTrustState::Estimating
        );
        assert!(machine.estimate().rejection_reasons[0].contains("0,1,5"));
    }

    #[test]
    fn full_evidence_converges_then_mount_shift_creates_a_new_epoch() {
        let mut machine = CalibrationStateMachine::new(
            RigidTransform3::default(),
            0,
            CalibrationStateConfig::default(),
        );
        converge_full_transform(&mut machine, RigidTransform3::default(), 0);
        assert_eq!(
            machine.estimate().trust_state,
            CalibrationTrustState::Trusted
        );
        let shifted = RigidTransform3 {
            translation_m: [0.12, 0.0, 0.0],
            ..RigidTransform3::default()
        };
        machine.observe(evidence(2_000, shifted, [true; 6]), 2_000);
        assert_eq!(
            machine.estimate().trust_state,
            CalibrationTrustState::Invalidated
        );
        assert_eq!(machine.estimate().epoch.id, 1);
        assert!(machine.estimate().epoch.invalidation_reason.is_some());

        converge_full_transform(&mut machine, shifted, 2_500);
        assert_eq!(
            machine.estimate().trust_state,
            CalibrationTrustState::Trusted
        );
        assert!((machine.estimate().transform.translation_m[0] - 0.12).abs() < 0.01);
    }

    #[test]
    fn lidar_is_never_required_for_trust() {
        let mut machine = CalibrationStateMachine::new(
            RigidTransform3::default(),
            0,
            CalibrationStateConfig::default(),
        );
        converge_full_transform(&mut machine, RigidTransform3::default(), 0);
        assert_eq!(
            machine.estimate().trust_state,
            CalibrationTrustState::Trusted
        );
        assert!(!machine
            .estimate()
            .evidence_counts
            .contains_key(&CalibrationEvidenceSource::Lidar));
    }

    #[test]
    fn stale_evidence_is_rejected_without_changing_epoch() {
        let mut machine = CalibrationStateMachine::new(
            RigidTransform3::default(),
            5_000,
            CalibrationStateConfig::default(),
        );
        machine.observe(
            evidence(1_000, RigidTransform3::default(), [true; 6]),
            5_000,
        );
        assert_eq!(machine.estimate().epoch.id, 0);
        assert!(machine.estimate().rejection_reasons[0].contains("stale"));
    }

    #[test]
    fn residuals_degrade_then_large_discontinuity_invalidates() {
        let mut machine = CalibrationStateMachine::new(
            RigidTransform3::default(),
            0,
            CalibrationStateConfig::default(),
        );
        converge_full_transform(&mut machine, RigidTransform3::default(), 0);
        let mut degraded = evidence(2_000, RigidTransform3::default(), [true; 6]);
        degraded.residuals.wall_m = Some(0.06);
        machine.observe(degraded, 2_000);
        assert_eq!(
            machine.estimate().trust_state,
            CalibrationTrustState::Degraded
        );
        let mut shifted = evidence(2_500, RigidTransform3::default(), [true; 6]);
        shifted.residuals.map_consistency_m = Some(0.20);
        machine.observe(shifted, 2_500);
        assert_eq!(
            machine.estimate().trust_state,
            CalibrationTrustState::Invalidated
        );
        assert_eq!(machine.estimate().epoch.id, 1);
    }

    #[test]
    fn partially_observable_floor_evidence_can_detect_a_mount_shift() {
        let mut machine = CalibrationStateMachine::new(
            RigidTransform3::default(),
            0,
            CalibrationStateConfig::default(),
        );
        let partial = [false, false, true, true, true, false];
        for timestamp in [0, 500, 1_000] {
            machine.observe(
                evidence(timestamp, RigidTransform3::default(), partial),
                timestamp,
            );
        }
        assert_eq!(
            machine.estimate().trust_state,
            CalibrationTrustState::Estimating
        );
        let shifted = RigidTransform3 {
            translation_m: [0.0, 0.0, 0.12],
            ..RigidTransform3::default()
        };
        machine.observe(evidence(1_500, shifted, partial), 1_500);
        assert_eq!(
            machine.estimate().trust_state,
            CalibrationTrustState::Invalidated
        );
        assert_eq!(machine.estimate().epoch.id, 1);
    }

    #[test]
    fn trusted_transform_degrades_when_calibration_evidence_ages_out() {
        let mut machine = CalibrationStateMachine::new(
            RigidTransform3::default(),
            0,
            CalibrationStateConfig::default(),
        );
        converge_full_transform(&mut machine, RigidTransform3::default(), 0);
        assert_eq!(
            machine.estimate().trust_state,
            CalibrationTrustState::Trusted
        );
        machine.refresh(4_000);
        assert_eq!(
            machine.estimate().trust_state,
            CalibrationTrustState::Degraded
        );
        assert!(machine
            .estimate()
            .rejection_reasons
            .iter()
            .any(|reason| reason.contains("aged past")));
    }
}
