use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocomotionCalibrationTrustState {
    #[default]
    Nominal,
    Estimating,
    Trusted,
    Degraded,
    Invalidated,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TravelDirection {
    #[default]
    Forward,
    Reverse,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RotationDirection {
    #[default]
    CounterClockwise,
    Clockwise,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocomotionConditions {
    pub surface: Option<String>,
    pub tire_condition: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StraightCalibrationEpisode {
    pub captured_at_ms: u64,
    pub direction: TravelDirection,
    pub reported_distance_m: f32,
    pub actual_distance_m: f32,
    pub lateral_drift_m: f32,
    pub endpoint_heading_error_rad: f32,
    pub environmental_alignment_residual_m: f32,
    pub confidence: f32,
    pub repeated_traversal: bool,
    pub loop_supported: bool,
    pub conditions: LocomotionConditions,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RotationCalibrationEpisode {
    pub captured_at_ms: u64,
    pub direction: RotationDirection,
    pub commanded_angle_rad: f32,
    pub wheel_odometry_angle_rad: f32,
    pub imu_angle_rad: Option<f32>,
    pub imu_trusted: bool,
    pub environmental_angle_rad: Option<f32>,
    pub environmental_alignment_residual_m: f32,
    pub loop_angle_rad: Option<f32>,
    pub axle_center_displacement_m: f32,
    pub confidence: f32,
    pub conditions: LocomotionConditions,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BoundedEstimate {
    pub value: f32,
    pub uncertainty: f32,
    pub minimum: f32,
    pub maximum: f32,
    pub evidence_count: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocomotionCalibrationState {
    pub trust_state: LocomotionCalibrationTrustState,
    pub epoch: u64,
    pub epoch_started_at_ms: u64,
    pub global_distance_scale: BoundedEstimate,
    pub left_distance_scale: BoundedEstimate,
    pub right_distance_scale: BoundedEstimate,
    pub forward_distance_ratio: BoundedEstimate,
    pub reverse_distance_ratio: BoundedEstimate,
    pub counter_clockwise_rotation_scale: BoundedEstimate,
    pub clockwise_rotation_scale: BoundedEstimate,
    pub effective_wheelbase_m: BoundedEstimate,
    pub straight_evidence_count: u64,
    pub rotation_evidence_count: u64,
    pub rejected_straight_count: u64,
    pub rejected_rotation_count: u64,
    pub confidence: f32,
    pub straight_line_consistent: bool,
    pub conditions: LocomotionConditions,
    pub rejection_reasons: Vec<String>,
    pub recent_straight_episodes: Vec<StraightCalibrationEpisode>,
    pub recent_rotation_episodes: Vec<RotationCalibrationEpisode>,
    /// Learned parameters are advisory only and never mutate motor/safety authority.
    pub authority: String,
}

impl Default for LocomotionCalibrationState {
    fn default() -> Self {
        let distance = BoundedEstimate {
            value: 1.0,
            uncertainty: 0.15,
            minimum: 0.85,
            maximum: 1.15,
            evidence_count: 0,
        };
        let rotation = BoundedEstimate {
            value: 1.0,
            uncertainty: 0.20,
            minimum: 0.80,
            maximum: 1.20,
            evidence_count: 0,
        };
        Self {
            trust_state: LocomotionCalibrationTrustState::Nominal,
            epoch: 0,
            epoch_started_at_ms: 0,
            global_distance_scale: distance,
            left_distance_scale: distance,
            right_distance_scale: distance,
            forward_distance_ratio: distance,
            reverse_distance_ratio: distance,
            counter_clockwise_rotation_scale: rotation,
            clockwise_rotation_scale: rotation,
            effective_wheelbase_m: BoundedEstimate {
                value: 0.235,
                uncertainty: 0.035,
                minimum: 0.20,
                maximum: 0.30,
                evidence_count: 0,
            },
            straight_evidence_count: 0,
            rotation_evidence_count: 0,
            rejected_straight_count: 0,
            rejected_rotation_count: 0,
            confidence: 0.0,
            straight_line_consistent: true,
            conditions: LocomotionConditions::default(),
            rejection_reasons: vec!["using conservative nominal locomotion parameters".to_string()],
            recent_straight_episodes: Vec::new(),
            recent_rotation_episodes: Vec::new(),
            authority: "advisory_only_brainstem_motion_and_safety_unchanged".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocomotionCalibrationConfig {
    pub update_rate: f32,
    pub minimum_confidence: f32,
    pub minimum_straight_distance_m: f32,
    pub maximum_alignment_residual_m: f32,
    pub maximum_lateral_drift_m: f32,
    pub maximum_straight_heading_error_rad: f32,
    pub minimum_rotation_rad: f32,
    pub maximum_axle_displacement_m: f32,
    pub minimum_evidence: u64,
    pub maximum_straight_conflict: f32,
}

impl Default for LocomotionCalibrationConfig {
    fn default() -> Self {
        Self {
            update_rate: 0.08,
            minimum_confidence: 0.8,
            minimum_straight_distance_m: 0.5,
            maximum_alignment_residual_m: 0.10,
            maximum_lateral_drift_m: 0.25,
            maximum_straight_heading_error_rad: 0.20,
            minimum_rotation_rad: 45.0_f32.to_radians(),
            maximum_axle_displacement_m: 0.08,
            minimum_evidence: 5,
            maximum_straight_conflict: 0.04,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LocomotionCalibrationEstimator {
    pub config: LocomotionCalibrationConfig,
    state: LocomotionCalibrationState,
}

impl LocomotionCalibrationEstimator {
    pub fn state(&self) -> &LocomotionCalibrationState {
        &self.state
    }

    pub fn observe_straight(&mut self, episode: StraightCalibrationEpisode) -> bool {
        push_recent(&mut self.state.recent_straight_episodes, episode.clone());
        if let Some(reason) = self.reject_straight(&episode) {
            self.state.rejected_straight_count =
                self.state.rejected_straight_count.saturating_add(1);
            self.degrade(reason);
            return false;
        }
        self.update_conditions(&episode.conditions, episode.captured_at_ms);
        let ratio = episode.actual_distance_m.abs() / episode.reported_distance_m.abs();
        update_estimate(
            &mut self.state.global_distance_scale,
            ratio,
            self.config.update_rate,
        );
        let directional = match episode.direction {
            TravelDirection::Forward => &mut self.state.forward_distance_ratio,
            TravelDirection::Reverse => &mut self.state.reverse_distance_ratio,
        };
        update_estimate(directional, ratio, self.config.update_rate);

        let curvature = episode.endpoint_heading_error_rad / episode.actual_distance_m.abs();
        let asymmetry = (curvature * 0.5).clamp(-0.08, 0.08);
        update_estimate(
            &mut self.state.left_distance_scale,
            ratio * (1.0 + asymmetry),
            self.config.update_rate,
        );
        update_estimate(
            &mut self.state.right_distance_scale,
            ratio * (1.0 - asymmetry),
            self.config.update_rate,
        );
        self.state.straight_evidence_count = self.state.straight_evidence_count.saturating_add(1);
        self.state.straight_line_consistent =
            (self.state.left_distance_scale.value - self.state.right_distance_scale.value).abs()
                <= 0.12;
        self.refresh_trust();
        true
    }

    pub fn observe_rotation(&mut self, episode: RotationCalibrationEpisode) -> bool {
        push_recent(&mut self.state.recent_rotation_episodes, episode.clone());
        let Some(reference_angle) = fused_rotation_reference(&episode) else {
            self.reject_rotation("rotation has no trusted external angle reference");
            return false;
        };
        if let Some(reason) = self.reject_rotation_episode(&episode, reference_angle) {
            self.reject_rotation(&reason);
            return false;
        }
        self.update_conditions(&episode.conditions, episode.captured_at_ms);
        let ratio = reference_angle.abs() / episode.wheel_odometry_angle_rad.abs();
        let estimate = match episode.direction {
            RotationDirection::CounterClockwise => &mut self.state.counter_clockwise_rotation_scale,
            RotationDirection::Clockwise => &mut self.state.clockwise_rotation_scale,
        };
        update_estimate(estimate, ratio, self.config.update_rate);
        let effective_wheelbase = 0.235 / ratio;
        update_estimate(
            &mut self.state.effective_wheelbase_m,
            effective_wheelbase,
            self.config.update_rate,
        );
        self.state.rotation_evidence_count = self.state.rotation_evidence_count.saturating_add(1);
        self.refresh_trust();
        true
    }

    pub fn validate_straight_held_out(
        &self,
        reported_distance_m: f32,
        actual_distance_m: f32,
        tolerance_m: f32,
    ) -> Result<f32, String> {
        let predicted = reported_distance_m * self.state.global_distance_scale.value;
        bounded_error(predicted, actual_distance_m, tolerance_m, "distance")
    }

    pub fn validate_rotation_held_out(
        &self,
        direction: RotationDirection,
        odometry_angle_rad: f32,
        actual_angle_rad: f32,
        tolerance_rad: f32,
    ) -> Result<f32, String> {
        let scale = match direction {
            RotationDirection::CounterClockwise => {
                self.state.counter_clockwise_rotation_scale.value
            }
            RotationDirection::Clockwise => self.state.clockwise_rotation_scale.value,
        };
        bounded_error(
            odometry_angle_rad * scale,
            actual_angle_rad,
            tolerance_rad,
            "heading",
        )
    }

    fn reject_straight(&self, episode: &StraightCalibrationEpisode) -> Option<String> {
        if episode.confidence < self.config.minimum_confidence {
            return Some("straight episode confidence is below the update gate".to_string());
        }
        if !episode.repeated_traversal && !episode.loop_supported {
            return Some("straight episode lacks repeated-traversal or loop evidence".to_string());
        }
        if episode.actual_distance_m.abs() < self.config.minimum_straight_distance_m
            || episode.reported_distance_m.abs() < self.config.minimum_straight_distance_m
        {
            return Some("straight episode is too short to be observable".to_string());
        }
        if episode.environmental_alignment_residual_m > self.config.maximum_alignment_residual_m {
            return Some("environmental alignment residual is too large".to_string());
        }
        if episode.lateral_drift_m.abs() > self.config.maximum_lateral_drift_m
            || episode.endpoint_heading_error_rad.abs()
                > self.config.maximum_straight_heading_error_rad
        {
            return Some("straight drift or endpoint heading is outside bounds".to_string());
        }
        let ratio = episode.actual_distance_m.abs() / episode.reported_distance_m.abs();
        if !(self.state.global_distance_scale.minimum..=self.state.global_distance_scale.maximum)
            .contains(&ratio)
        {
            return Some("distance ratio conflicts with bounded safe priors".to_string());
        }
        None
    }

    fn reject_rotation_episode(
        &self,
        episode: &RotationCalibrationEpisode,
        reference_angle: f32,
    ) -> Option<String> {
        if episode.confidence < self.config.minimum_confidence {
            return Some("rotation episode confidence is below the update gate".to_string());
        }
        if episode.wheel_odometry_angle_rad.abs() < self.config.minimum_rotation_rad
            || reference_angle.abs() < self.config.minimum_rotation_rad
        {
            return Some("rotation episode is too small to be observable".to_string());
        }
        if episode.environmental_alignment_residual_m > self.config.maximum_alignment_residual_m
            || episode.axle_center_displacement_m.abs() > self.config.maximum_axle_displacement_m
        {
            return Some("rotation alignment or axle displacement is outside bounds".to_string());
        }
        let ratio = reference_angle.abs() / episode.wheel_odometry_angle_rad.abs();
        let bound = &self.state.clockwise_rotation_scale;
        if !(bound.minimum..=bound.maximum).contains(&ratio) {
            return Some("rotation ratio conflicts with bounded safe priors".to_string());
        }
        let straight_asymmetry =
            (self.state.left_distance_scale.value - self.state.right_distance_scale.value).abs();
        if self.state.straight_evidence_count >= self.config.minimum_evidence
            && straight_asymmetry > self.config.maximum_straight_conflict
            && (ratio - 1.0).signum()
                != (self.state.left_distance_scale.value - self.state.right_distance_scale.value)
                    .signum()
        {
            return Some("rotation update conflicts with straight-line asymmetry".to_string());
        }
        None
    }

    fn reject_rotation(&mut self, reason: &str) {
        self.state.rejected_rotation_count = self.state.rejected_rotation_count.saturating_add(1);
        self.degrade(reason.to_string());
    }

    fn degrade(&mut self, reason: String) {
        self.state.rejection_reasons = vec![reason];
        if self.state.straight_evidence_count + self.state.rotation_evidence_count > 0 {
            self.state.trust_state = LocomotionCalibrationTrustState::Degraded;
        }
    }

    fn update_conditions(&mut self, conditions: &LocomotionConditions, timestamp_ms: u64) {
        if self.state.conditions != *conditions
            && (self.state.conditions.surface.is_some()
                || self.state.conditions.tire_condition.is_some())
        {
            let next_epoch = self.state.epoch.saturating_add(1);
            self.state = LocomotionCalibrationState {
                trust_state: LocomotionCalibrationTrustState::Estimating,
                epoch: next_epoch,
                epoch_started_at_ms: timestamp_ms,
                conditions: conditions.clone(),
                rejection_reasons: vec![
                    "locomotion conditions changed; fresh evidence is required".to_string(),
                ],
                ..LocomotionCalibrationState::default()
            };
            return;
        }
        self.state.conditions = conditions.clone();
    }

    fn refresh_trust(&mut self) {
        let total = self.state.straight_evidence_count + self.state.rotation_evidence_count;
        self.state.confidence =
            (total as f32 / (self.config.minimum_evidence * 2).max(1) as f32).clamp(0.0, 1.0);
        self.state.trust_state = if self.state.straight_evidence_count
            >= self.config.minimum_evidence
            && self.state.rotation_evidence_count >= self.config.minimum_evidence
            && self.state.straight_line_consistent
        {
            LocomotionCalibrationTrustState::Trusted
        } else {
            LocomotionCalibrationTrustState::Estimating
        };
        self.state.rejection_reasons.clear();
        if self.state.straight_evidence_count < self.config.minimum_evidence {
            self.state
                .rejection_reasons
                .push("straight-run evidence is incomplete".to_string());
        }
        if self.state.rotation_evidence_count < self.config.minimum_evidence {
            self.state
                .rejection_reasons
                .push("rotation evidence is incomplete".to_string());
        }
        if !self.state.straight_line_consistent {
            self.state
                .rejection_reasons
                .push("left/right estimates conflict with straight-line validation".to_string());
        }
    }
}

fn fused_rotation_reference(episode: &RotationCalibrationEpisode) -> Option<f32> {
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;
    if episode.imu_trusted {
        if let Some(angle) = episode.imu_angle_rad.filter(|value| value.is_finite()) {
            weighted_sum += angle * 0.4;
            total_weight += 0.4;
        }
    }
    if let Some(angle) = episode
        .environmental_angle_rad
        .filter(|value| value.is_finite())
    {
        weighted_sum += angle * 0.4;
        total_weight += 0.4;
    }
    if let Some(angle) = episode.loop_angle_rad.filter(|value| value.is_finite()) {
        weighted_sum += angle * 0.2;
        total_weight += 0.2;
    }
    (total_weight > 0.0).then_some(weighted_sum / total_weight)
}

fn update_estimate(estimate: &mut BoundedEstimate, observation: f32, update_rate: f32) {
    let observation = observation.clamp(estimate.minimum, estimate.maximum);
    let residual = (observation - estimate.value).abs();
    estimate.value = (estimate.value * (1.0 - update_rate) + observation * update_rate)
        .clamp(estimate.minimum, estimate.maximum);
    estimate.evidence_count = estimate.evidence_count.saturating_add(1);
    estimate.uncertainty = (estimate.uncertainty * 0.9 + residual * 0.1)
        .max(0.002)
        .min(estimate.maximum - estimate.minimum);
}

fn bounded_error(predicted: f32, actual: f32, tolerance: f32, label: &str) -> Result<f32, String> {
    let error = (predicted - actual).abs();
    if error <= tolerance {
        Ok(error)
    } else {
        Err(format!(
            "held-out {label} error {error:.4} exceeds tolerance {tolerance:.4}"
        ))
    }
}

fn push_recent<T>(episodes: &mut Vec<T>, episode: T) {
    const HISTORY_LIMIT: usize = 32;
    if episodes.len() == HISTORY_LIMIT {
        episodes.remove(0);
    }
    episodes.push(episode);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn straight(direction: TravelDirection, ratio: f32) -> StraightCalibrationEpisode {
        StraightCalibrationEpisode {
            captured_at_ms: 1_000,
            direction,
            reported_distance_m: 2.0,
            actual_distance_m: 2.0 * ratio,
            lateral_drift_m: 0.02,
            endpoint_heading_error_rad: 0.04,
            environmental_alignment_residual_m: 0.02,
            confidence: 0.95,
            repeated_traversal: true,
            loop_supported: true,
            conditions: LocomotionConditions {
                surface: Some("sealed_concrete".to_string()),
                tire_condition: Some("dry".to_string()),
            },
        }
    }

    fn rotation(direction: RotationDirection, ratio: f32) -> RotationCalibrationEpisode {
        let odometry = std::f32::consts::PI;
        RotationCalibrationEpisode {
            captured_at_ms: 2_000,
            direction,
            commanded_angle_rad: odometry,
            wheel_odometry_angle_rad: odometry,
            imu_angle_rad: Some(odometry * ratio),
            imu_trusted: true,
            environmental_angle_rad: Some(odometry * ratio),
            environmental_alignment_residual_m: 0.02,
            loop_angle_rad: Some(odometry * ratio),
            axle_center_displacement_m: 0.01,
            confidence: 0.95,
            conditions: straight(TravelDirection::Forward, 1.0).conditions,
        }
    }

    #[test]
    fn nominal_parameters_are_safe_fallback_and_advisory_only() {
        let estimator = LocomotionCalibrationEstimator::default();
        assert_eq!(
            estimator.state().trust_state,
            LocomotionCalibrationTrustState::Nominal
        );
        assert_eq!(estimator.state().global_distance_scale.value, 1.0);
        assert!(estimator.state().authority.contains("brainstem"));
    }

    #[test]
    fn repeated_straight_runs_learn_scale_asymmetry_and_direction() {
        let mut estimator = LocomotionCalibrationEstimator::default();
        for _ in 0..10 {
            assert!(estimator.observe_straight(straight(TravelDirection::Forward, 1.05)));
            let mut reverse = straight(TravelDirection::Reverse, 0.98);
            reverse.endpoint_heading_error_rad = -0.03;
            assert!(estimator.observe_straight(reverse));
        }
        assert!(estimator.state().global_distance_scale.value > 1.0);
        assert!(estimator.state().forward_distance_ratio.value > 1.0);
        assert!(estimator.state().reverse_distance_ratio.value < 1.0);
        assert_ne!(
            estimator.state().left_distance_scale.value,
            estimator.state().right_distance_scale.value
        );
    }

    #[test]
    fn low_observability_and_contradictory_straight_runs_are_rejected() {
        let mut estimator = LocomotionCalibrationEstimator::default();
        let mut weak = straight(TravelDirection::Forward, 1.0);
        weak.confidence = 0.2;
        assert!(!estimator.observe_straight(weak));
        assert!(!estimator.observe_straight(straight(TravelDirection::Forward, 1.4)));
        assert_eq!(estimator.state().straight_evidence_count, 0);
        assert_eq!(estimator.state().rejected_straight_count, 2);
    }

    #[test]
    fn cw_and_ccw_rotation_biases_remain_separate_and_bounded() {
        let mut estimator = LocomotionCalibrationEstimator::default();
        for _ in 0..10 {
            assert!(estimator.observe_rotation(rotation(RotationDirection::CounterClockwise, 1.06)));
            assert!(estimator.observe_rotation(rotation(RotationDirection::Clockwise, 0.95)));
        }
        assert!(estimator.state().counter_clockwise_rotation_scale.value > 1.0);
        assert!(estimator.state().clockwise_rotation_scale.value < 1.0);
        assert!((0.20..=0.30).contains(&estimator.state().effective_wheelbase_m.value));
    }

    #[test]
    fn conditions_advance_epoch_and_held_out_validation_is_explicit() {
        let mut estimator = LocomotionCalibrationEstimator::default();
        for _ in 0..20 {
            estimator.observe_straight(straight(TravelDirection::Forward, 1.05));
            estimator.observe_rotation(rotation(RotationDirection::Clockwise, 1.03));
        }
        assert!(estimator.validate_straight_held_out(2.0, 2.1, 0.08).is_ok());
        assert!(estimator
            .validate_rotation_held_out(
                RotationDirection::Clockwise,
                std::f32::consts::PI,
                std::f32::consts::PI * 1.03,
                0.08,
            )
            .is_ok());
        let mut changed = straight(TravelDirection::Forward, 1.02);
        changed.captured_at_ms = 9_000;
        changed.conditions.surface = Some("low_pile_carpet".to_string());
        estimator.observe_straight(changed);
        assert_eq!(estimator.state().epoch, 1);
        assert_eq!(estimator.state().straight_evidence_count, 1);
        assert_eq!(estimator.state().rotation_evidence_count, 0);
        assert_eq!(
            estimator.state().trust_state,
            LocomotionCalibrationTrustState::Estimating
        );
        assert_eq!(estimator.state().global_distance_scale.evidence_count, 1);
        assert_eq!(
            estimator
                .state()
                .counter_clockwise_rotation_scale
                .evidence_count,
            0
        );
    }
}
