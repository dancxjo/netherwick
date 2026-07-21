use crate::{CalibrationEpoch, RigidTransform3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImuCalibrationTrustState {
    #[default]
    WarmingUp,
    Estimating,
    Trusted,
    Degraded,
    Invalidated,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ImuMotionContext {
    pub commanded_linear_mps: f32,
    pub commanded_angular_rps: f32,
    pub odometry_linear_mps: f32,
    pub odometry_angular_rps: f32,
    pub environmental_angular_rps: Option<f32>,
}

impl ImuMotionContext {
    pub fn confidently_stationary(self) -> bool {
        self.commanded_linear_mps.abs() <= 0.005
            && self.commanded_angular_rps.abs() <= 0.01
            && self.odometry_linear_mps.abs() <= 0.005
            && self.odometry_angular_rps.abs() <= 0.01
            && self
                .environmental_angular_rps
                .is_none_or(|rate| rate.abs() <= 0.01)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImuCalibrationState {
    pub trust_state: ImuCalibrationTrustState,
    pub epoch: CalibrationEpoch,
    pub gyro_bias_rad_s: [f32; 3],
    pub gyro_variance: [f32; 3],
    pub gravity_unit: [f32; 3],
    pub gravity_variance: [f32; 3],
    pub sensor_to_base: RigidTransform3,
    pub roll_pitch_observable: bool,
    pub yaw_axis_observable: bool,
    pub yaw_rate_scale: Option<f32>,
    pub confidence: f32,
    pub total_samples: u64,
    pub stationary_candidates: u64,
    pub stationary_samples: u64,
    pub rejected_motion_samples: u64,
    pub rotation_evidence_samples: u64,
    pub warmup_elapsed_ms: u64,
    pub temperature_c: Option<f32>,
    pub reference_temperature_c: Option<f32>,
    pub bias_temperature_slope_rad_s_per_c: [f32; 3],
    pub updated_at_ms: u64,
    pub rejection_reasons: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImuCalibrationConfig {
    pub minimum_stationary_samples: u64,
    pub minimum_rotation_samples: u64,
    pub minimum_warmup_ms: u64,
    pub maximum_gyro_stationary_rps: f32,
    pub maximum_accel_norm_error_g: f32,
    pub maximum_bias_variance: f32,
    pub gravity_shift_rad: f32,
    pub bias_shift_rad_s: f32,
}

impl Default for ImuCalibrationConfig {
    fn default() -> Self {
        Self {
            minimum_stationary_samples: 50,
            minimum_rotation_samples: 12,
            minimum_warmup_ms: 400,
            maximum_gyro_stationary_rps: 0.08,
            maximum_accel_norm_error_g: 0.04,
            maximum_bias_variance: 0.0004,
            gravity_shift_rad: 12.0_f32.to_radians(),
            bias_shift_rad_s: 0.08,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct RunningVectorStats {
    count: u64,
    mean: [f32; 3],
    m2: [f32; 3],
}

impl RunningVectorStats {
    fn push(&mut self, sample: [f32; 3]) {
        self.count = self.count.saturating_add(1);
        let count = self.count as f32;
        for axis in 0..3 {
            let delta = sample[axis] - self.mean[axis];
            self.mean[axis] += delta / count;
            let delta_after = sample[axis] - self.mean[axis];
            self.m2[axis] += delta * delta_after;
        }
    }

    fn variance(self) -> [f32; 3] {
        if self.count < 2 {
            return [f32::INFINITY; 3];
        }
        self.m2.map(|value| value / (self.count - 1) as f32)
    }
}

#[derive(Clone, Debug)]
pub struct ImuCalibrationEstimator {
    pub config: ImuCalibrationConfig,
    configured_prior: RigidTransform3,
    prior_supplied: bool,
    gyro: RunningVectorStats,
    gravity: RunningVectorStats,
    temperature_bias_anchor: Option<([f32; 3], f32)>,
    state: ImuCalibrationState,
}

impl ImuCalibrationEstimator {
    pub fn new(
        configured_prior: RigidTransform3,
        prior_supplied: bool,
        started_at_ms: u64,
        config: ImuCalibrationConfig,
    ) -> Self {
        Self {
            configured_prior,
            prior_supplied,
            gyro: RunningVectorStats::default(),
            gravity: RunningVectorStats::default(),
            temperature_bias_anchor: None,
            state: ImuCalibrationState {
                trust_state: ImuCalibrationTrustState::WarmingUp,
                epoch: CalibrationEpoch {
                    id: 0,
                    started_at_ms,
                    invalidated_at_ms: None,
                    invalidation_reason: None,
                },
                gyro_bias_rad_s: [0.0; 3],
                gyro_variance: [f32::INFINITY; 3],
                gravity_unit: [0.0, 0.0, 1.0],
                gravity_variance: [f32::INFINITY; 3],
                sensor_to_base: configured_prior,
                roll_pitch_observable: false,
                yaw_axis_observable: false,
                yaw_rate_scale: None,
                confidence: 0.0,
                total_samples: 0,
                stationary_candidates: 0,
                stationary_samples: 0,
                rejected_motion_samples: 0,
                rotation_evidence_samples: 0,
                warmup_elapsed_ms: 0,
                temperature_c: None,
                reference_temperature_c: None,
                bias_temperature_slope_rad_s_per_c: [0.0; 3],
                updated_at_ms: started_at_ms,
                rejection_reasons: vec!["stationary warm-up is incomplete".to_string()],
            },
            config,
        }
    }

    pub fn state(&self) -> &ImuCalibrationState {
        &self.state
    }

    pub fn observe(
        &mut self,
        acceleration_g: [f32; 3],
        angular_velocity_rad_s: [f32; 3],
        temperature_c: Option<f32>,
        motion: ImuMotionContext,
        captured_at_ms: u64,
    ) -> &ImuCalibrationState {
        self.state.total_samples = self.state.total_samples.saturating_add(1);
        self.state.updated_at_ms = captured_at_ms;
        self.state.warmup_elapsed_ms =
            captured_at_ms.saturating_sub(self.state.epoch.started_at_ms);
        self.state.temperature_c = temperature_c.filter(|value| value.is_finite());
        if self.state.trust_state == ImuCalibrationTrustState::Invalidated {
            self.state.trust_state = ImuCalibrationTrustState::WarmingUp;
            self.state.epoch.invalidated_at_ms = None;
            self.state.epoch.invalidation_reason = None;
        }
        let accel_norm = norm(acceleration_g);
        let gyro_norm = norm(angular_velocity_rad_s);
        let inertial_candidate = (accel_norm - 1.0).abs() <= self.config.maximum_accel_norm_error_g
            && gyro_norm <= self.config.maximum_gyro_stationary_rps;
        if inertial_candidate {
            self.state.stationary_candidates = self.state.stationary_candidates.saturating_add(1);
        }
        if inertial_candidate && motion.confidently_stationary() {
            if self.detect_remount(acceleration_g, angular_velocity_rad_s) {
                self.invalidate(
                    captured_at_ms,
                    "gravity or stationary bias changed abruptly",
                );
                return &self.state;
            }
            self.gyro.push(angular_velocity_rad_s);
            self.gravity.push(normalize(acceleration_g));
            self.state.stationary_samples = self.state.stationary_samples.saturating_add(1);
            self.update_stationary_estimate();
        } else if inertial_candidate {
            self.state.rejected_motion_samples =
                self.state.rejected_motion_samples.saturating_add(1);
        }
        self.observe_rotation(angular_velocity_rad_s, motion);
        self.refresh_trust();
        &self.state
    }

    pub fn corrected_gyro(&self, angular_velocity_rad_s: [f32; 3]) -> [f32; 3] {
        let temperature_delta = self
            .state
            .temperature_c
            .zip(self.state.reference_temperature_c)
            .map(|(current, reference)| current - reference)
            .unwrap_or(0.0);
        let corrected = [0, 1, 2].map(|axis| {
            angular_velocity_rad_s[axis]
                - self.state.gyro_bias_rad_s[axis]
                - self.state.bias_temperature_slope_rad_s_per_c[axis] * temperature_delta
        });
        self.state.sensor_to_base.rotate_vector(corrected)
    }

    pub fn acceleration_in_base(&self, acceleration_g: [f32; 3]) -> [f32; 3] {
        self.state.sensor_to_base.rotate_vector(acceleration_g)
    }

    fn update_stationary_estimate(&mut self) {
        self.state.gyro_bias_rad_s = self.gyro.mean;
        self.state.gyro_variance = self.gyro.variance();
        self.state.gravity_unit = normalize(self.gravity.mean);
        self.state.gravity_variance = self.gravity.variance();
        if self.gravity.count >= 8 {
            let gravity = self.state.gravity_unit;
            let roll = gravity[1].atan2(gravity[2]);
            let pitch = (-gravity[0]).atan2(gravity[1].hypot(gravity[2]));
            self.state.sensor_to_base.rotation_rpy_rad = [roll, pitch, 0.0];
            if self.prior_supplied {
                self.state.sensor_to_base.rotation_rpy_rad[2] =
                    self.configured_prior.rotation_rpy_rad[2];
            }
            self.state.roll_pitch_observable = true;
        }
        if let Some(temperature) = self.state.temperature_c {
            match self.temperature_bias_anchor {
                None if self.gyro.count >= self.config.minimum_stationary_samples => {
                    self.temperature_bias_anchor = Some((self.gyro.mean, temperature));
                    self.state.reference_temperature_c = Some(temperature);
                }
                Some((anchor_bias, anchor_temperature))
                    if (temperature - anchor_temperature).abs() >= 2.0 =>
                {
                    let delta = temperature - anchor_temperature;
                    self.state.bias_temperature_slope_rad_s_per_c =
                        [0, 1, 2].map(|axis| (self.gyro.mean[axis] - anchor_bias[axis]) / delta);
                }
                _ => {}
            }
        }
    }

    fn observe_rotation(&mut self, angular_velocity_rad_s: [f32; 3], motion: ImuMotionContext) {
        let reference = motion
            .environmental_angular_rps
            .filter(|value| value.abs() > 0.05)
            .or_else(|| {
                (motion.odometry_angular_rps.abs() > 0.05).then_some(motion.odometry_angular_rps)
            });
        let Some(reference) = reference else {
            return;
        };
        let corrected = self.corrected_gyro(angular_velocity_rad_s);
        if corrected[2].abs() <= 0.02 {
            return;
        }
        let ratio = reference / corrected[2];
        if !ratio.is_finite() || !(0.5..=1.5).contains(&ratio.abs()) {
            return;
        }
        self.state.rotation_evidence_samples =
            self.state.rotation_evidence_samples.saturating_add(1);
        let previous = self.state.yaw_rate_scale.unwrap_or(ratio);
        self.state.yaw_rate_scale = Some(previous * 0.9 + ratio * 0.1);
        if self.state.rotation_evidence_samples >= self.config.minimum_rotation_samples {
            self.state.yaw_axis_observable = true;
        }
    }

    fn detect_remount(&self, acceleration_g: [f32; 3], angular_velocity_rad_s: [f32; 3]) -> bool {
        if !matches!(
            self.state.trust_state,
            ImuCalibrationTrustState::Trusted | ImuCalibrationTrustState::Degraded
        ) {
            return false;
        }
        let gravity = normalize(acceleration_g);
        let dot = gravity
            .iter()
            .zip(self.state.gravity_unit)
            .map(|(left, right)| *left * right)
            .sum::<f32>()
            .clamp(-1.0, 1.0);
        let gravity_shift = dot.acos();
        let bias_shift = angular_velocity_rad_s
            .iter()
            .zip(self.state.gyro_bias_rad_s)
            .map(|(sample, bias)| (*sample - bias).abs())
            .fold(0.0, f32::max);
        gravity_shift > self.config.gravity_shift_rad || bias_shift > self.config.bias_shift_rad_s
    }

    fn invalidate(&mut self, timestamp_ms: u64, reason: &str) {
        let next_epoch = self.state.epoch.id.saturating_add(1);
        self.gyro = RunningVectorStats::default();
        self.gravity = RunningVectorStats::default();
        self.temperature_bias_anchor = None;
        self.state = ImuCalibrationState {
            trust_state: ImuCalibrationTrustState::Invalidated,
            epoch: CalibrationEpoch {
                id: next_epoch,
                started_at_ms: timestamp_ms,
                invalidated_at_ms: Some(timestamp_ms),
                invalidation_reason: Some(reason.to_string()),
            },
            sensor_to_base: self.configured_prior,
            updated_at_ms: timestamp_ms,
            rejection_reasons: vec![reason.to_string()],
            ..ImuCalibrationState {
                trust_state: ImuCalibrationTrustState::WarmingUp,
                epoch: CalibrationEpoch {
                    id: next_epoch,
                    started_at_ms: timestamp_ms,
                    invalidated_at_ms: None,
                    invalidation_reason: None,
                },
                gyro_bias_rad_s: [0.0; 3],
                gyro_variance: [f32::INFINITY; 3],
                gravity_unit: [0.0, 0.0, 1.0],
                gravity_variance: [f32::INFINITY; 3],
                sensor_to_base: self.configured_prior,
                roll_pitch_observable: false,
                yaw_axis_observable: false,
                yaw_rate_scale: None,
                confidence: 0.0,
                total_samples: 0,
                stationary_candidates: 0,
                stationary_samples: 0,
                rejected_motion_samples: 0,
                rotation_evidence_samples: 0,
                warmup_elapsed_ms: 0,
                temperature_c: None,
                reference_temperature_c: None,
                bias_temperature_slope_rad_s_per_c: [0.0; 3],
                updated_at_ms: timestamp_ms,
                rejection_reasons: Vec::new(),
            }
        };
    }

    fn refresh_trust(&mut self) {
        if self.state.trust_state == ImuCalibrationTrustState::Invalidated {
            return;
        }
        let warmed = self.state.warmup_elapsed_ms >= self.config.minimum_warmup_ms;
        let bias_ready = self.state.stationary_samples >= self.config.minimum_stationary_samples
            && self
                .state
                .gyro_variance
                .iter()
                .all(|variance| *variance <= self.config.maximum_bias_variance);
        let rejected_ratio = self.state.rejected_motion_samples as f32
            / self.state.stationary_candidates.max(1) as f32;
        self.state.confidence = ((self.state.stationary_samples as f32
            / self.config.minimum_stationary_samples.max(1) as f32)
            .clamp(0.0, 1.0)
            * if self.state.roll_pitch_observable {
                1.0
            } else {
                0.4
            }
            * (1.0 - rejected_ratio.clamp(0.0, 0.8)))
        .clamp(0.0, 1.0);
        self.state.trust_state = if !warmed {
            ImuCalibrationTrustState::WarmingUp
        } else if bias_ready && self.state.roll_pitch_observable {
            ImuCalibrationTrustState::Trusted
        } else if rejected_ratio > 0.5 {
            ImuCalibrationTrustState::Degraded
        } else {
            ImuCalibrationTrustState::Estimating
        };
        self.state.rejection_reasons.clear();
        if !warmed {
            self.state
                .rejection_reasons
                .push("stationary warm-up is incomplete".to_string());
        }
        if !bias_ready {
            self.state
                .rejection_reasons
                .push("gyro bias/noise has not converged".to_string());
        }
        if !self.state.roll_pitch_observable {
            self.state
                .rejection_reasons
                .push("gravity axis is not observable".to_string());
        }
        if !self.state.yaw_axis_observable {
            self.state
                .rejection_reasons
                .push("yaw mounting remains unobservable without rotation evidence".to_string());
        }
    }
}

fn norm(vector: [f32; 3]) -> f32 {
    vector.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn normalize(vector: [f32; 3]) -> [f32; 3] {
    let length = norm(vector);
    if !length.is_finite() || length <= f32::EPSILON {
        [0.0, 0.0, 1.0]
    } else {
        vector.map(|value| value / length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stationary_bias_converges_while_yaw_stays_explicitly_uncertain() {
        let mut estimator = ImuCalibrationEstimator::new(
            RigidTransform3::default(),
            false,
            0,
            ImuCalibrationConfig::default(),
        );
        for index in 0..60 {
            estimator.observe(
                [0.0, 0.0, 1.0],
                [0.01, -0.02, 0.005],
                Some(30.0),
                ImuMotionContext::default(),
                index * 10,
            );
        }
        assert_eq!(
            estimator.state().trust_state,
            ImuCalibrationTrustState::Trusted
        );
        assert!(estimator.state().roll_pitch_observable);
        assert!(!estimator.state().yaw_axis_observable);
        assert!(estimator
            .state()
            .rejection_reasons
            .iter()
            .any(|reason| reason.contains("yaw")));
        assert!((estimator.state().gyro_bias_rad_s[0] - 0.01).abs() < 0.001);
    }

    #[test]
    fn commanded_motion_rejects_false_stationary_candidates() {
        let mut estimator = ImuCalibrationEstimator::new(
            RigidTransform3::default(),
            false,
            0,
            ImuCalibrationConfig::default(),
        );
        estimator.observe(
            [0.0, 0.0, 1.0],
            [0.0; 3],
            None,
            ImuMotionContext {
                commanded_linear_mps: 0.2,
                ..ImuMotionContext::default()
            },
            10,
        );
        assert_eq!(estimator.state().stationary_candidates, 1);
        assert_eq!(estimator.state().stationary_samples, 0);
        assert_eq!(estimator.state().rejected_motion_samples, 1);
    }

    #[test]
    fn known_rotation_resolves_yaw_axis_sign_and_scale() {
        let mut estimator = ImuCalibrationEstimator::new(
            RigidTransform3::default(),
            false,
            0,
            ImuCalibrationConfig::default(),
        );
        for index in 0..60 {
            estimator.observe(
                [0.0, 0.0, 1.0],
                [0.0; 3],
                None,
                ImuMotionContext::default(),
                index * 10,
            );
        }
        for index in 0..15 {
            estimator.observe(
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -0.5],
                None,
                ImuMotionContext {
                    commanded_angular_rps: 0.5,
                    odometry_angular_rps: 0.5,
                    ..ImuMotionContext::default()
                },
                700 + index * 10,
            );
        }
        assert!(estimator.state().yaw_axis_observable);
        assert!(estimator
            .state()
            .yaw_rate_scale
            .is_some_and(|scale| scale < -0.9));
    }

    #[test]
    fn simulated_remount_invalidates_and_reconverges_new_epoch() {
        let mut estimator = ImuCalibrationEstimator::new(
            RigidTransform3::default(),
            false,
            0,
            ImuCalibrationConfig::default(),
        );
        for index in 0..60 {
            estimator.observe(
                [0.0, 0.0, 1.0],
                [0.0; 3],
                None,
                ImuMotionContext::default(),
                index * 10,
            );
        }
        estimator.observe(
            [0.0, 1.0, 0.0],
            [0.0; 3],
            None,
            ImuMotionContext::default(),
            700,
        );
        assert_eq!(
            estimator.state().trust_state,
            ImuCalibrationTrustState::Invalidated
        );
        assert_eq!(estimator.state().epoch.id, 1);
        for index in 0..60 {
            estimator.observe(
                [0.0, 1.0, 0.0],
                [0.0; 3],
                None,
                ImuMotionContext::default(),
                710 + index * 10,
            );
        }
        assert_eq!(
            estimator.state().trust_state,
            ImuCalibrationTrustState::Trusted
        );
        assert_eq!(estimator.state().epoch.id, 1);
    }
}
