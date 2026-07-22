use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationTransitionKind {
    Accepted,
    Rejected,
    Degraded,
    Invalidated,
    Remounted,
    RolledBack,
    NewlyTrusted,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationClockedTime {
    pub t_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock_epoch: Option<String>,
}

impl CalibrationClockedTime {
    pub fn new(t_ms: u64) -> Self {
        Self {
            t_ms,
            clock_epoch: None,
        }
    }

    pub fn in_epoch(t_ms: u64, clock_epoch: impl Into<String>) -> Self {
        Self {
            t_ms,
            clock_epoch: Some(clock_epoch.into()),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationArtifactIdentity {
    pub id: String,
    pub checksum: String,
}

impl CalibrationArtifactIdentity {
    pub fn from_value(prefix: &str, value: &Value) -> Self {
        let checksum = checksum(value);
        Self {
            id: format!("{prefix}:{checksum}"),
            checksum,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationDofState {
    pub value: Value,
    pub observable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertainty: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationTransitionState {
    pub trust: String,
    pub confidence: f32,
    pub degrees_of_freedom: BTreeMap<String, CalibrationDofState>,
    pub state: Value,
}

pub fn calibration_state(
    trust: impl Into<String>,
    confidence: f32,
    state: Value,
    degrees_of_freedom: impl IntoIterator<Item = (String, CalibrationDofState)>,
) -> CalibrationTransitionState {
    CalibrationTransitionState {
        trust: trust.into(),
        confidence: confidence.clamp(0.0, 1.0),
        degrees_of_freedom: degrees_of_freedom.into_iter().collect(),
        state,
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationEvidenceWindow {
    pub started_at: CalibrationClockedTime,
    pub ended_at: CalibrationClockedTime,
    pub sample_count: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationEvidenceRecord {
    pub event_id: String,
    pub source: String,
    pub occurred: CalibrationClockedTime,
    pub observed: CalibrationClockedTime,
    pub artifact: CalibrationArtifactIdentity,
    pub payload: Value,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationConsumerImpact {
    pub consumer: String,
    pub allowed_before: bool,
    pub allowed_after: bool,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationTransition {
    pub id: String,
    pub estimator: String,
    pub epoch: u64,
    pub sequence: u64,
    pub kind: CalibrationTransitionKind,
    pub prior: CalibrationTransitionState,
    pub new: CalibrationTransitionState,
    pub changed_degrees_of_freedom: Vec<String>,
    pub evidence_window: CalibrationEvidenceWindow,
    pub evidence: Vec<CalibrationEvidenceRecord>,
    pub prior_artifact: CalibrationArtifactIdentity,
    pub candidate_artifact: CalibrationArtifactIdentity,
    pub accepted_artifact: CalibrationArtifactIdentity,
    pub affected_consumers: Vec<CalibrationConsumerImpact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub occurred: CalibrationClockedTime,
    pub observed: CalibrationClockedTime,
}

impl CalibrationTransition {
    #[allow(clippy::too_many_arguments)]
    pub fn author(
        estimator: impl Into<String>,
        epoch: u64,
        sequence: u64,
        kind: CalibrationTransitionKind,
        prior: CalibrationTransitionState,
        new: CalibrationTransitionState,
        evidence_source: impl Into<String>,
        evidence_payload: Value,
        evidence_window: CalibrationEvidenceWindow,
        affected_consumers: Vec<CalibrationConsumerImpact>,
        reason: Option<String>,
        occurred: CalibrationClockedTime,
        observed: CalibrationClockedTime,
    ) -> Self {
        let estimator = estimator.into();
        let prior_value = serde_json::to_value(&prior).unwrap_or(Value::Null);
        let new_value = serde_json::to_value(&new).unwrap_or(Value::Null);
        let prior_artifact = CalibrationArtifactIdentity::from_value(
            &format!("calibration:{estimator}:accepted"),
            &prior_value,
        );
        let new_artifact = CalibrationArtifactIdentity::from_value(
            &format!("calibration:{estimator}:accepted"),
            &new_value,
        );
        let candidate_artifact = CalibrationArtifactIdentity::from_value(
            &format!("calibration:{estimator}:candidate"),
            &evidence_payload,
        );
        let accepted_artifact = if kind == CalibrationTransitionKind::Rejected {
            prior_artifact.clone()
        } else {
            new_artifact
        };
        let evidence_source = evidence_source.into();
        let evidence_artifact = CalibrationArtifactIdentity::from_value(
            &format!("calibration:{estimator}:evidence"),
            &evidence_payload,
        );
        let evidence_id = format!(
            "calibration-evidence:{estimator}:{epoch}:{sequence}:{}",
            evidence_artifact.checksum
        );
        let changed_degrees_of_freedom = changed_dofs(&prior, &new);
        Self {
            id: format!("calibration-transition:{estimator}:{epoch}:{sequence}"),
            estimator,
            epoch,
            sequence,
            kind,
            prior,
            new,
            changed_degrees_of_freedom,
            evidence_window,
            evidence: vec![CalibrationEvidenceRecord {
                event_id: evidence_id,
                source: evidence_source,
                occurred: occurred.clone(),
                observed: observed.clone(),
                artifact: evidence_artifact,
                payload: evidence_payload,
            }],
            prior_artifact,
            candidate_artifact,
            accepted_artifact,
            affected_consumers,
            reason,
            occurred,
            observed,
        }
    }
}

fn changed_dofs(
    prior: &CalibrationTransitionState,
    new: &CalibrationTransitionState,
) -> Vec<String> {
    let mut names = prior
        .degrees_of_freedom
        .keys()
        .chain(new.degrees_of_freedom.keys())
        .cloned()
        .collect::<Vec<_>>();
    names.sort();
    names.dedup();
    names.retain(|name| prior.degrees_of_freedom.get(name) != new.degrees_of_freedom.get(name));
    names
}

fn checksum(value: &Value) -> String {
    let canonical = canonical_json(value);
    format!("sha256:{:x}", Sha256::digest(canonical.as_bytes()))
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value).unwrap_or_else(|_| "\"\"".into()),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}:{}",
                    serde_json::to_string(key).unwrap_or_else(|_| "\"\"".into()),
                    canonical_json(value)
                ))
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationReplay {
    pub estimators: BTreeMap<String, CalibrationTransitionState>,
    pub epochs: BTreeMap<String, u64>,
    pub accepted_artifacts: BTreeMap<String, CalibrationArtifactIdentity>,
    pub transitions: Vec<(String, u64, String, String)>,
}

impl CalibrationReplay {
    pub fn apply(
        &mut self,
        transition: &CalibrationTransition,
    ) -> Result<(), CalibrationReplayError> {
        if let Some(current) = self.estimators.get(&transition.estimator) {
            if current != &transition.prior {
                return Err(CalibrationReplayError::PriorStateMismatch(
                    transition.estimator.clone(),
                ));
            }
        }
        self.estimators
            .insert(transition.estimator.clone(), transition.new.clone());
        self.epochs
            .insert(transition.estimator.clone(), transition.epoch);
        self.accepted_artifacts.insert(
            transition.estimator.clone(),
            transition.accepted_artifact.clone(),
        );
        self.transitions.push((
            transition.estimator.clone(),
            transition.epoch,
            transition.prior.trust.clone(),
            transition.new.trust.clone(),
        ));
        Ok(())
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CalibrationReplayError {
    #[error("calibration replay prior state mismatch for {0}")]
    PriorStateMismatch(String),
}

pub fn consumer_impacts(
    consumers: &[&str],
    allowed_before: bool,
    allowed_after: bool,
    reason: impl Into<String>,
) -> Vec<CalibrationConsumerImpact> {
    let reason = reason.into();
    consumers
        .iter()
        .map(|consumer| CalibrationConsumerImpact {
            consumer: (*consumer).to_string(),
            allowed_before,
            allowed_after,
            reason: reason.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CalibrationEvidenceSource, CalibrationResiduals, CalibrationStateConfig,
        CalibrationStateMachine, ImuCalibrationConfig, ImuCalibrationEstimator, ImuMotionContext,
        LatencyEventFeature, LocomotionCalibrationEstimator, LocomotionConditions, RigidTransform3,
        RotationCalibrationEpisode, RotationDirection, SensorLatencyRegistry,
        SensorTimingObservation, StraightCalibrationEpisode, TransformEstimateEvidence,
        TravelDirection, TRANSFORM_DOF_COUNT,
    };

    fn transform_evidence(
        timestamp: u64,
        x: f32,
        observable: [bool; TRANSFORM_DOF_COUNT],
    ) -> TransformEstimateEvidence {
        TransformEstimateEvidence {
            source: CalibrationEvidenceSource::PersistentSurface,
            captured_at_ms: timestamp,
            transform: RigidTransform3 {
                translation_m: [x, 0.0, 0.0],
                ..RigidTransform3::default()
            },
            observable_dofs: observable,
            covariance: [0.001; TRANSFORM_DOF_COUNT],
            residuals: CalibrationResiduals::default(),
        }
    }

    #[test]
    fn every_estimator_class_authors_only_real_transitions_with_partial_observability() {
        let mut all = Vec::new();

        let mut geometry = CalibrationStateMachine::new(
            RigidTransform3::default(),
            0,
            CalibrationStateConfig {
                minimum_evidence_per_dof: 1,
                minimum_independent_sources: 1,
                minimum_trust_span_ms: 0,
                ..CalibrationStateConfig::default()
            },
        );
        geometry.refresh(1);
        assert!(
            geometry.take_transitions().is_empty(),
            "unchanged refresh must stay silent"
        );
        geometry.observe(
            transform_evidence(10, 0.01, [true, false, false, false, false, false]),
            10,
        );
        let partial = geometry.take_transitions();
        assert_eq!(partial.len(), 1);
        assert_eq!(partial[0].kind, CalibrationTransitionKind::Accepted);
        assert_eq!(partial[0].changed_degrees_of_freedom, vec!["x"]);
        assert!(partial[0].new.degrees_of_freedom["x"].observable);
        assert!(!partial[0].new.degrees_of_freedom["yaw"].observable);
        assert_eq!(partial[0].occurred.clock_epoch, None);
        assert_eq!(partial[0].observed.clock_epoch, None);
        assert_eq!(partial[0].evidence[0].occurred.clock_epoch, None);
        all.extend(partial);
        geometry.observe(
            transform_evidence(5, 0.02, [true, false, false, false, false, false]),
            20,
        );
        let rejected = geometry.take_transitions();
        assert_eq!(rejected[0].kind, CalibrationTransitionKind::Rejected);
        all.extend(rejected);
        geometry.observe(
            transform_evidence(30, 0.25, [true, false, false, false, false, false]),
            30,
        );
        let remounted = geometry.take_transitions();
        assert_eq!(remounted[0].kind, CalibrationTransitionKind::Remounted);
        assert_eq!(remounted[0].epoch, 1);
        all.extend(remounted);

        let mut imu = ImuCalibrationEstimator::new(
            RigidTransform3::default(),
            true,
            0,
            ImuCalibrationConfig {
                minimum_stationary_samples: 8,
                minimum_rotation_samples: 1,
                minimum_warmup_ms: 0,
                ..ImuCalibrationConfig::default()
            },
        );
        for timestamp in 1..=8 {
            imu.observe(
                [0.0, 0.0, 1.0],
                [0.01, 0.0, 0.0],
                None,
                ImuMotionContext::default(),
                timestamp,
            );
        }
        imu.observe(
            [0.0, 0.0, 1.0],
            [0.01, 0.0, 0.2],
            None,
            ImuMotionContext {
                commanded_angular_rps: 0.2,
                odometry_angular_rps: 0.2,
                ..ImuMotionContext::default()
            },
            9,
        );
        let imu_transitions = imu.take_transitions();
        assert!(imu_transitions.iter().any(|transition| {
            transition
                .changed_degrees_of_freedom
                .iter()
                .any(|dof| dof == "gyro_bias_x")
        }));
        assert!(imu_transitions.iter().any(|transition| {
            transition.kind == CalibrationTransitionKind::NewlyTrusted
                && transition.new.degrees_of_freedom["yaw_rate_scale"].observable
        }));
        all.extend(imu_transitions);

        let conditions = LocomotionConditions {
            surface: Some("tile".into()),
            tire_condition: Some("clean".into()),
        };
        let mut locomotion = LocomotionCalibrationEstimator::default();
        locomotion.config.minimum_evidence = 1;
        locomotion.observe_straight(StraightCalibrationEpisode {
            captured_at_ms: 100,
            direction: TravelDirection::Forward,
            reported_distance_m: 1.0,
            actual_distance_m: 1.05,
            lateral_drift_m: 0.0,
            endpoint_heading_error_rad: 0.0,
            environmental_alignment_residual_m: 0.01,
            confidence: 0.95,
            repeated_traversal: true,
            loop_supported: false,
            conditions: conditions.clone(),
        });
        locomotion.observe_rotation(RotationCalibrationEpisode {
            captured_at_ms: 110,
            direction: RotationDirection::Clockwise,
            commanded_angle_rad: 1.0,
            wheel_odometry_angle_rad: 1.0,
            imu_angle_rad: None,
            imu_trusted: false,
            environmental_angle_rad: Some(1.02),
            environmental_alignment_residual_m: 0.01,
            loop_angle_rad: Some(1.02),
            axle_center_displacement_m: 0.0,
            confidence: 0.95,
            conditions: conditions.clone(),
        });
        let locomotion_transitions = locomotion.take_transitions();
        assert!(locomotion_transitions.iter().any(|transition| {
            transition
                .changed_degrees_of_freedom
                .iter()
                .any(|dof| dof == "global_distance_scale")
        }));
        assert!(locomotion_transitions.iter().any(|transition| {
            transition.kind == CalibrationTransitionKind::NewlyTrusted
                && transition
                    .changed_degrees_of_freedom
                    .iter()
                    .any(|dof| dof == "effective_wheelbase_m")
        }));
        all.extend(locomotion_transitions);
        locomotion.observe_straight(StraightCalibrationEpisode {
            captured_at_ms: 120,
            direction: TravelDirection::Forward,
            reported_distance_m: 1.0,
            actual_distance_m: 1.02,
            lateral_drift_m: 0.0,
            endpoint_heading_error_rad: 0.0,
            environmental_alignment_residual_m: 0.01,
            confidence: 0.95,
            repeated_traversal: true,
            loop_supported: false,
            conditions: LocomotionConditions {
                surface: Some("carpet".into()),
                tire_condition: Some("clean".into()),
            },
        });
        let rollback = locomotion.take_transitions();
        assert!(rollback.iter().any(|transition| {
            transition.kind == CalibrationTransitionKind::RolledBack && transition.epoch == 1
        }));
        all.extend(rollback);

        let mut registry = SensorLatencyRegistry::default();
        registry.config.minimum_samples = 2;
        registry.config.minimum_correlated_events = 1;
        registry.observe_reference_event("rotation_cw_started", 200);
        for receive in [220, 240] {
            registry.observe(
                "imu",
                SensorTimingObservation {
                    producer_time_ms: Some(receive - 20),
                    receive_time_ms: receive,
                    canonical_frame_time_ms: receive,
                    clock_epoch: Some(7),
                    clock_confidence: 1.0,
                    event_features: vec![LatencyEventFeature {
                        name: "rotation_cw_started".into(),
                        value: -0.2,
                        occurred_at_ms: 200,
                    }],
                },
            );
        }
        registry.observe(
            "imu",
            SensorTimingObservation {
                producer_time_ms: None,
                receive_time_ms: 260,
                canonical_frame_time_ms: 260,
                clock_epoch: Some(7),
                clock_confidence: 1.0,
                event_features: Vec::new(),
            },
        );
        let latency_transitions = registry.take_transitions();
        assert!(latency_transitions
            .iter()
            .any(|transition| { transition.kind == CalibrationTransitionKind::NewlyTrusted }));
        assert!(latency_transitions
            .iter()
            .any(|transition| { transition.kind == CalibrationTransitionKind::Rejected }));
        assert!(latency_transitions.iter().all(|transition| {
            transition.occurred.clock_epoch.as_deref() == Some("imu:7")
                && transition.observed.clock_epoch.is_none()
        }));
        all.extend(latency_transitions);
        registry.snapshot(5_000);
        registry.observe(
            "imu",
            SensorTimingObservation {
                producer_time_ms: Some(4_990),
                receive_time_ms: 5_010,
                canonical_frame_time_ms: 5_010,
                clock_epoch: Some(8),
                clock_confidence: 1.0,
                event_features: Vec::new(),
            },
        );
        let invalidated = registry.take_transitions();
        assert!(invalidated
            .iter()
            .any(|transition| transition.kind == CalibrationTransitionKind::Degraded));
        assert!(invalidated
            .iter()
            .any(|transition| transition.kind == CalibrationTransitionKind::Invalidated));
        all.extend(invalidated);

        assert!(all.iter().all(|transition| {
            !transition.evidence.is_empty()
                && !transition.prior_artifact.checksum.is_empty()
                && !transition.candidate_artifact.checksum.is_empty()
                && !transition.accepted_artifact.checksum.is_empty()
                && transition
                    .occurred
                    .clock_epoch
                    .as_ref()
                    .is_none_or(|epoch| !epoch.is_empty())
                && transition
                    .observed
                    .clock_epoch
                    .as_ref()
                    .is_none_or(|epoch| !epoch.is_empty())
        }));

        let encoded = serde_json::to_vec(&all).unwrap();
        let decoded: Vec<CalibrationTransition> = serde_json::from_slice(&encoded).unwrap();
        let mut live_replay = CalibrationReplay::default();
        let mut recorded_replay = CalibrationReplay::default();
        for transition in &all {
            live_replay.apply(transition).unwrap();
        }
        for transition in &decoded {
            recorded_replay.apply(transition).unwrap();
        }
        assert_eq!(live_replay.transitions, recorded_replay.transitions);
        assert_eq!(live_replay.epochs, recorded_replay.epochs);
        assert_eq!(
            live_replay.accepted_artifacts,
            recorded_replay.accepted_artifacts
        );
        assert_eq!(
            live_replay.transitions,
            all.iter()
                .map(|transition| (
                    transition.estimator.clone(),
                    transition.epoch,
                    transition.prior.trust.clone(),
                    transition.new.trust.clone(),
                ))
                .collect::<Vec<_>>()
        );
    }
}
