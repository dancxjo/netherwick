use anyhow::Result;
use netherwick_actions::{action_to_motor_command, ActionPrimitive, ExploreStyle, TurnDir};
use netherwick_body::BodySense;
use netherwick_core::{ExperienceId, ImpressionId, Provenance, Reward, SensationId, TimeMs};
use netherwick_now::{DriveSense, MemorySense, Now, SenseVectorizer, SurpriseSense};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExperienceLatent {
    pub t_ms: TimeMs,
    pub z: Vec<f32>,
    pub reconstruction_error: f32,
    pub prediction_error: f32,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FuturePrediction {
    pub offset_ms: TimeMs,
    pub predicted_z: Vec<f32>,
    pub confidence: f32,
    pub summary: Option<String>,
}

pub trait ExperienceEncoder {
    fn encode(&mut self, now: &Now) -> Result<ExperienceLatent>;
}

pub trait ExperienceDecoder {
    fn decode(&mut self, latent: &ExperienceLatent) -> Result<NowReconstruction>;
}

pub trait FuturePredictor {
    fn predict(
        &mut self,
        latent: &ExperienceLatent,
        action: &ActionPrimitive,
        offset_ms: TimeMs,
    ) -> Result<FuturePrediction>;
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DangerInput {
    pub z: Vec<f32>,
    pub action_features: Vec<f32>,
    pub body_features: Vec<f32>,
}

impl DangerInput {
    pub fn from_parts(z: Vec<f32>, action: Option<&ActionPrimitive>, now: &Now) -> Self {
        Self {
            z,
            action_features: danger_action_features(action),
            body_features: danger_body_features(now),
        }
    }

    pub fn flat_features(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(
            self.z.len() + self.action_features.len() + self.body_features.len(),
        );
        out.extend(self.z.iter().copied().map(sanitize_feature));
        out.extend(self.action_features.iter().copied().map(sanitize_feature));
        out.extend(self.body_features.iter().copied().map(sanitize_feature));
        out
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DangerOutput {
    pub bump_risk: f32,
    pub cliff_risk: f32,
    pub wheel_drop_risk: f32,
    pub stuck_risk: f32,
    pub confidence: f32,
}

impl DangerOutput {
    pub fn risks(&self) -> [f32; 4] {
        [
            self.bump_risk,
            self.cliff_risk,
            self.wheel_drop_risk,
            self.stuck_risk,
        ]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DangerTarget {
    pub bump: f32,
    pub cliff: f32,
    pub wheel_drop: f32,
    pub stuck: f32,
}

impl DangerTarget {
    pub fn risks(&self) -> [f32; 4] {
        [self.bump, self.cliff, self.wheel_drop, self.stuck]
    }
}

pub fn danger_input_from_transition_like(
    before_z: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
    before: &Now,
) -> DangerInput {
    DangerInput::from_parts(before_z.z.clone(), action, before)
}

pub fn danger_target_from_transition_like(
    before: &Now,
    action: Option<&ActionPrimitive>,
    after: &Now,
) -> DangerTarget {
    let commanded_motion = matches!(
        action,
        Some(ActionPrimitive::Go { .. } | ActionPrimitive::Explore { .. })
    );
    let odom_delta = ((after.body.odometry.x_m - before.body.odometry.x_m).powi(2)
        + (after.body.odometry.y_m - before.body.odometry.y_m).powi(2))
    .sqrt();
    let no_forward_velocity = after.body.velocity.forward_m_s.abs() < 0.01;
    let no_odometry = odom_delta < 0.005;
    DangerTarget {
        bump: bool01(after.body.flags.bump_left || after.body.flags.bump_right),
        cliff: bool01(after.body.flags.cliff_left || after.body.flags.cliff_right),
        wheel_drop: bool01(after.body.flags.wheel_drop),
        stuck: bool01(commanded_motion && no_forward_velocity && no_odometry),
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NowReconstruction {
    pub t_ms: TimeMs,
    pub body: Option<BodySense>,
    pub memory: Option<MemorySense>,
    pub drives: Option<DriveSense>,
    pub prediction_summary: Option<String>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default)]
pub struct FeatureExperienceEncoder {
    vectorizers: Vec<BaselineSenseVectorizer>,
}

impl FeatureExperienceEncoder {
    pub fn new() -> Self {
        Self {
            vectorizers: vec![
                BaselineSenseVectorizer::Body,
                BaselineSenseVectorizer::Memory,
                BaselineSenseVectorizer::Drives,
                BaselineSenseVectorizer::Predictions,
                BaselineSenseVectorizer::Surprise,
                BaselineSenseVectorizer::Safety,
                BaselineSenseVectorizer::Reign,
                BaselineSenseVectorizer::Audio,
                BaselineSenseVectorizer::Range,
            ],
        }
    }
}

impl ExperienceEncoder for FeatureExperienceEncoder {
    fn encode(&mut self, now: &Now) -> Result<ExperienceLatent> {
        let mut z = Vec::new();
        for vectorizer in &self.vectorizers {
            z.extend(vectorizer.encode(now));
        }
        if z.is_empty() {
            z.push(0.0);
        }
        Ok(ExperienceLatent {
            t_ms: now.t_ms,
            z,
            reconstruction_error: 0.0,
            prediction_error: now.surprise.prediction_error,
            confidence: 0.65,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct PartialExperienceDecoder;

impl ExperienceDecoder for PartialExperienceDecoder {
    fn decode(&mut self, latent: &ExperienceLatent) -> Result<NowReconstruction> {
        let body = (!latent.z.is_empty()).then(|| BodySense {
            battery_level: latent.z.first().copied().unwrap_or(1.0).clamp(0.0, 1.0),
            charging: latent.z.get(1).copied().unwrap_or(0.0) >= 0.5,
            ..BodySense::default()
        });
        let drives = (latent.z.len() >= 23).then(|| DriveSense {
            battery_hunger: latent
                .z
                .get(17)
                .copied()
                .unwrap_or_default()
                .clamp(0.0, 1.0),
            danger_avoidance: latent
                .z
                .get(18)
                .copied()
                .unwrap_or_default()
                .clamp(0.0, 1.0),
            curiosity: latent
                .z
                .get(19)
                .copied()
                .unwrap_or_default()
                .clamp(0.0, 1.0),
            social_interest: latent
                .z
                .get(20)
                .copied()
                .unwrap_or_default()
                .clamp(0.0, 1.0),
            fatigue: latent
                .z
                .get(21)
                .copied()
                .unwrap_or_default()
                .clamp(0.0, 1.0),
            uncertainty_pressure: latent
                .z
                .get(22)
                .copied()
                .unwrap_or_default()
                .clamp(0.0, 1.0),
        });
        Ok(NowReconstruction {
            t_ms: latent.t_ms,
            body,
            memory: None,
            drives,
            prediction_summary: Some(format!(
                "Partial reconstruction from {} latent features.",
                latent.z.len()
            )),
            confidence: latent.confidence * (1.0 - latent.reconstruction_error).clamp(0.0, 1.0),
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct StasisFuturePredictor;

impl FuturePredictor for StasisFuturePredictor {
    fn predict(
        &mut self,
        latent: &ExperienceLatent,
        _action: &ActionPrimitive,
        offset_ms: TimeMs,
    ) -> Result<FuturePrediction> {
        let seconds = offset_ms as f32 / 1_000.0;
        Ok(FuturePrediction {
            offset_ms,
            predicted_z: latent.z.clone(),
            confidence: (latent.confidence * (-0.18 * seconds).exp()).clamp(0.05, 1.0),
            summary: Some("I expect the situation to remain mostly stable.".to_string()),
        })
    }
}

pub trait SurpriseComputer {
    fn compute(
        &self,
        predicted: &[FuturePrediction],
        actual_z: &ExperienceLatent,
        actual_now: &Now,
    ) -> SurpriseSense;
}

#[derive(Clone, Debug, Default)]
pub struct BaselineSurpriseComputer;

impl SurpriseComputer for BaselineSurpriseComputer {
    fn compute(
        &self,
        predicted: &[FuturePrediction],
        actual_z: &ExperienceLatent,
        actual_now: &Now,
    ) -> SurpriseSense {
        let nearest = predicted
            .iter()
            .min_by_key(|prediction| prediction.offset_ms);
        let prediction_error = nearest
            .map(|prediction| normalized_distance(&prediction.predicted_z, &actual_z.z))
            .unwrap_or(0.0);
        let mut total = prediction_error;
        if actual_now.body.flags.bump_left || actual_now.body.flags.bump_right {
            total += 0.25;
        }
        if actual_now.body.flags.cliff_left
            || actual_now.body.flags.cliff_right
            || actual_now.body.flags.wheel_drop
        {
            total += 0.45;
        }
        if actual_now.body.charging
            && !predicted
                .iter()
                .any(|prediction| prediction.summary_contains("charge"))
        {
            total += 0.12;
        }
        if actual_now
            .extensions
            .get("safety.vetoed")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            total += 0.2;
        }
        SurpriseSense {
            schema_version: 1,
            total: total.clamp(0.0, 1.0),
            prediction_error,
        }
    }
}

pub trait RewardComputer {
    fn compute(
        &self,
        before: &Now,
        action: Option<&ActionPrimitive>,
        after: &Now,
        surprise: &SurpriseSense,
    ) -> Reward;
}

#[derive(Clone, Debug, Default)]
pub struct BaselineRewardComputer;

impl RewardComputer for BaselineRewardComputer {
    fn compute(
        &self,
        before: &Now,
        action: Option<&ActionPrimitive>,
        after: &Now,
        surprise: &SurpriseSense,
    ) -> Reward {
        let battery_delta = after.body.battery_level - before.body.battery_level;
        let mut value = battery_delta.max(0.0) * (1.0 - before.body.battery_level).clamp(0.0, 1.0);
        if !before.body.charging && after.body.charging && before.body.battery_level < 0.35 {
            value += 0.35;
        }
        if !after.body.flags.bump_left
            && !after.body.flags.bump_right
            && !after.body.flags.cliff_left
            && !after.body.flags.cliff_right
            && !after.body.flags.wheel_drop
        {
            value += 0.01;
        }
        if after.body.flags.bump_left || after.body.flags.bump_right {
            value -= 0.25;
        }
        if after.body.flags.cliff_left
            || after.body.flags.cliff_right
            || after.body.flags.wheel_drop
        {
            value -= 0.8;
        }
        if battery_delta < 0.0 {
            value += battery_delta * 0.2;
        }
        if matches!(
            action,
            Some(ActionPrimitive::Go { .. } | ActionPrimitive::Explore { .. })
        ) && after.body.velocity.forward_m_s.abs() < 0.01
            && before.body.velocity.forward_m_s.abs() < 0.01
        {
            value -= 0.08;
        }
        if after
            .extensions
            .get("safety.vetoed")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            value -= 0.12;
        }
        value -= surprise.total * 0.02;
        Reward { value }
    }
}

#[derive(Clone, Debug)]
enum BaselineSenseVectorizer {
    Body,
    Memory,
    Drives,
    Predictions,
    Surprise,
    Safety,
    Reign,
    Audio,
    Range,
}

impl SenseVectorizer for BaselineSenseVectorizer {
    fn sense_name(&self) -> &'static str {
        match self {
            Self::Body => "body",
            Self::Memory => "memory",
            Self::Drives => "drives",
            Self::Predictions => "predictions",
            Self::Surprise => "surprise",
            Self::Safety => "safety",
            Self::Reign => "reign",
            Self::Audio => "audio",
            Self::Range => "range",
        }
    }

    fn schema_version(&self) -> u32 {
        1
    }

    fn encode(&self, now: &Now) -> Vec<f32> {
        match self {
            Self::Body => vec![
                now.body.battery_level.clamp(0.0, 1.0),
                bool01(now.body.charging),
                bool01(now.body.flags.bump_left || now.body.flags.bump_right),
                bool01(now.body.flags.cliff_left || now.body.flags.cliff_right),
                bool01(now.body.flags.wheel_drop),
                bool01(now.body.flags.wall || now.body.flags.virtual_wall),
                now.body.velocity.forward_m_s.clamp(-1.0, 1.0),
                now.body.velocity.turn_rad_s.clamp(-1.0, 1.0),
                now.body.health.strain.clamp(0.0, 1.0),
                now.body.health.health.clamp(0.0, 1.0),
            ],
            Self::Memory => vec![
                now.memory.place_familiarity.clamp(0.0, 1.0),
                now.memory.place_danger.clamp(0.0, 1.0),
                now.memory.place_charge_value.clamp(0.0, 1.0),
                now.memory.face_familiarity.clamp(0.0, 1.0),
                now.memory.voice_familiarity.clamp(0.0, 1.0),
                (now.memory.similar_situation_count as f32 / 32.0).clamp(0.0, 1.0),
                bool01(now.memory.remembered_warning.is_some()),
            ],
            Self::Drives => vec![
                now.drives.battery_hunger.clamp(0.0, 1.0),
                now.drives.danger_avoidance.clamp(0.0, 1.0),
                now.drives.curiosity.clamp(0.0, 1.0),
                now.drives.social_interest.clamp(0.0, 1.0),
                now.drives.fatigue.clamp(0.0, 1.0),
                now.drives.uncertainty_pressure.clamp(0.0, 1.0),
            ],
            Self::Predictions => vec![
                now.predictions.uncertainty.clamp(0.0, 1.0),
                (now.predictions.expected_events.len() as f32 / 8.0).clamp(0.0, 1.0),
            ],
            Self::Surprise => vec![
                now.surprise.total.clamp(0.0, 1.0),
                now.surprise.prediction_error.clamp(0.0, 1.0),
            ],
            Self::Safety => vec![bool01(
                now.extensions
                    .get("safety.vetoed")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false),
            )],
            Self::Reign => vec![
                bool01(now.reign.active),
                now.reign.human_override_pressure.clamp(0.0, 1.0),
                (now.reign.pending_count as f32 / 8.0).clamp(0.0, 1.0),
            ],
            Self::Audio => vec![bool01(
                now.ear
                    .transcript
                    .as_ref()
                    .map(|text| !text.trim().is_empty())
                    .unwrap_or(false),
            )],
            Self::Range => vec![now
                .range
                .nearest_m
                .map(|m| (1.0 / (1.0 + m)).clamp(0.0, 1.0))
                .unwrap_or(0.0)],
        }
    }
}

fn normalized_distance(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().max(b.len());
    if len == 0 {
        return 0.0;
    }
    let sum = (0..len)
        .map(|idx| {
            let delta =
                a.get(idx).copied().unwrap_or_default() - b.get(idx).copied().unwrap_or_default();
            delta * delta
        })
        .sum::<f32>();
    (sum.sqrt() / (len as f32).sqrt()).clamp(0.0, 1.0)
}

fn bool01(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

fn danger_action_features(action: Option<&ActionPrimitive>) -> Vec<f32> {
    let motor = action_to_motor_command(action);
    let mut out = vec![
        bool01(action.is_none()),
        bool01(matches!(action, Some(ActionPrimitive::Stop))),
        bool01(matches!(action, Some(ActionPrimitive::Go { .. }))),
        bool01(matches!(action, Some(ActionPrimitive::Turn { .. }))),
        bool01(matches!(action, Some(ActionPrimitive::Explore { .. }))),
        bool01(matches!(action, Some(ActionPrimitive::Approach { .. }))),
        bool01(matches!(action, Some(ActionPrimitive::Dock))),
        bool01(matches!(action, Some(ActionPrimitive::Inspect { .. }))),
        bool01(matches!(action, Some(ActionPrimitive::Speak { .. }))),
        bool01(matches!(action, Some(ActionPrimitive::Chirp { .. }))),
        motor.forward.clamp(-1.0, 1.0),
        motor.turn.clamp(-1.0, 1.0),
    ];
    match action {
        Some(ActionPrimitive::Go {
            intensity,
            duration_ms,
        }) => {
            out.push(intensity.clamp(0.0, 1.0));
            out.push((*duration_ms as f32 / 5_000.0).clamp(0.0, 1.0));
            out.push(0.0);
        }
        Some(ActionPrimitive::Turn {
            direction,
            intensity,
            duration_ms,
        }) => {
            out.push(intensity.clamp(0.0, 1.0));
            out.push((*duration_ms as f32 / 5_000.0).clamp(0.0, 1.0));
            out.push(match direction {
                TurnDir::Left => 1.0,
                TurnDir::Right => -1.0,
            });
        }
        Some(ActionPrimitive::Explore { style, duration_ms }) => {
            out.push(match style {
                ExploreStyle::Wander => 0.25,
                ExploreStyle::RandomWalk => 0.5,
                ExploreStyle::WallFollow => 0.9,
            });
            out.push((*duration_ms as f32 / 5_000.0).clamp(0.0, 1.0));
            out.push(0.0);
        }
        _ => {
            out.push(0.0);
            out.push(0.0);
            out.push(0.0);
        }
    }
    out
}

fn danger_body_features(now: &Now) -> Vec<f32> {
    vec![
        now.body.battery_level.clamp(0.0, 1.0),
        bool01(now.body.charging),
        bool01(now.body.flags.bump_left || now.body.flags.bump_right),
        bool01(now.body.flags.cliff_left || now.body.flags.cliff_right),
        bool01(now.body.flags.wheel_drop),
        bool01(now.body.flags.wall || now.body.flags.virtual_wall),
        now.body.velocity.forward_m_s.clamp(-1.0, 1.0),
        now.body.velocity.turn_rad_s.clamp(-1.0, 1.0),
        now.body.health.strain.clamp(0.0, 1.0),
        now.body.health.health.clamp(0.0, 1.0),
        now.range
            .nearest_m
            .map(|m| (1.0 / (1.0 + m)).clamp(0.0, 1.0))
            .unwrap_or(0.0),
        now.memory.place_danger.clamp(0.0, 1.0),
        now.surprise.total.clamp(0.0, 1.0),
        now.surprise.prediction_error.clamp(0.0, 1.0),
        bool01(
            now.extensions
                .get("safety.vetoed")
                .and_then(|value| value.as_bool())
                .unwrap_or(false),
        ),
    ]
}

fn sanitize_feature(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

trait PredictionSummaryExt {
    fn summary_contains(&self, needle: &str) -> bool;
}

impl PredictionSummaryExt for FuturePrediction {
    fn summary_contains(&self, needle: &str) -> bool {
        self.summary
            .as_ref()
            .map(|summary| summary.to_lowercase().contains(needle))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use netherwick_body::BodySense;
    use netherwick_now::Now;

    #[test]
    fn feature_encoder_produces_non_empty_latent() {
        let mut encoder = FeatureExperienceEncoder::new();
        let mut now = Now::blank(42, BodySense::default());
        now.memory.place_familiarity = 0.7;
        now.drives.curiosity = 0.5;

        let latent = encoder.encode(&now).unwrap();

        assert_eq!(latent.t_ms, 42);
        assert!(!latent.z.is_empty());
        assert!(latent.z.iter().any(|value| *value > 0.0));
    }

    #[test]
    fn stasis_predictor_clones_latent_and_decays_confidence() {
        let mut predictor = StasisFuturePredictor;
        let latent = ExperienceLatent {
            t_ms: 10,
            z: vec![0.1, 0.2],
            confidence: 0.8,
            ..ExperienceLatent::default()
        };

        let near = predictor
            .predict(&latent, &ActionPrimitive::Stop, 1_000)
            .unwrap();
        let far = predictor
            .predict(&latent, &ActionPrimitive::Stop, 5_000)
            .unwrap();

        assert_eq!(near.predicted_z, latent.z);
        assert!(near.confidence > far.confidence);
        assert!(near
            .summary
            .as_deref()
            .unwrap_or_default()
            .contains("stable"));
    }

    #[test]
    fn reward_tastes_low_battery_charging_as_good() {
        let computer = BaselineRewardComputer;
        let mut before = Now::blank(1, BodySense::default());
        before.body.battery_level = 0.2;
        before.body.charging = false;
        let mut after = before.clone();
        after.t_ms = 2;
        after.body.battery_level = 0.24;
        after.body.charging = true;

        let reward = computer.compute(
            &before,
            Some(&ActionPrimitive::Dock),
            &after,
            &SurpriseSense::default(),
        );

        assert!(reward.value > 0.0);
    }

    #[test]
    fn danger_target_marks_bump_and_cliff_labels() {
        let before = Now::blank(1, BodySense::default());
        let mut after = before.clone();
        after.body.flags.bump_left = true;
        after.body.flags.cliff_right = true;

        let target =
            danger_target_from_transition_like(&before, Some(&ActionPrimitive::Stop), &after);

        assert_eq!(target.bump, 1.0);
        assert_eq!(target.cliff, 1.0);
        assert_eq!(target.wheel_drop, 0.0);
    }

    #[test]
    fn danger_target_marks_go_with_no_movement_as_stuck() {
        let before = Now::blank(1, BodySense::default());
        let mut after = before.clone();
        after.t_ms = 2;

        let target = danger_target_from_transition_like(
            &before,
            Some(&ActionPrimitive::Go {
                intensity: 0.4,
                duration_ms: 1_000,
            }),
            &after,
        );

        assert_eq!(target.stuck, 1.0);
    }

    #[test]
    fn danger_action_features_are_fixed_width() {
        let now = Now::blank(1, BodySense::default());
        let stop = DangerInput::from_parts(vec![0.0], Some(&ActionPrimitive::Stop), &now);
        let go = DangerInput::from_parts(
            vec![0.0],
            Some(&ActionPrimitive::Go {
                intensity: 0.4,
                duration_ms: 1_000,
            }),
            &now,
        );
        let turn = DangerInput::from_parts(
            vec![0.0],
            Some(&ActionPrimitive::Turn {
                direction: netherwick_actions::TurnDir::Left,
                intensity: 0.4,
                duration_ms: 1_000,
            }),
            &now,
        );

        assert_eq!(stop.action_features.len(), go.action_features.len());
        assert_eq!(go.action_features.len(), turn.action_features.len());
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sensation {
    pub id: SensationId,
    pub kind: String,
    pub source: String,
    pub occurred_at_ms: TimeMs,
    pub observed_at_ms: TimeMs,
    pub summary: Option<String>,
    pub provenance: Provenance,
    pub payload: Value,
}

impl Sensation {
    pub fn new(
        kind: impl Into<String>,
        source: impl Into<String>,
        occurred_at_ms: TimeMs,
        observed_at_ms: TimeMs,
        payload: Value,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            kind: kind.into(),
            source: source.into(),
            occurred_at_ms,
            observed_at_ms,
            summary: None,
            provenance: Provenance::direct(),
            payload,
        }
    }

    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    pub fn with_provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = provenance;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Impression {
    pub id: ImpressionId,
    pub kind: String,
    pub text: String,
    pub about: Vec<SensationId>,
    pub occurred_at_ms: TimeMs,
    pub observed_at_ms: TimeMs,
    pub confidence: f32,
    pub payload: Value,
}

impl Impression {
    pub fn new(
        kind: impl Into<String>,
        text: impl Into<String>,
        about: Vec<SensationId>,
        occurred_at_ms: TimeMs,
        observed_at_ms: TimeMs,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            kind: kind.into(),
            text: text.into(),
            about,
            occurred_at_ms,
            observed_at_ms,
            confidence: 0.5,
            payload: Value::Null,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Experience {
    pub id: ExperienceId,
    pub kind: String,
    pub text: String,
    pub impression_ids: Vec<ImpressionId>,
    pub sensation_ids: Vec<SensationId>,
    pub occurred_at_ms: TimeMs,
    pub observed_at_ms: TimeMs,
    pub salience: f32,
    pub tags: Vec<String>,
    pub payload: Value,
}

impl Experience {
    pub fn new(
        kind: impl Into<String>,
        text: impl Into<String>,
        impression_ids: Vec<ImpressionId>,
        sensation_ids: Vec<SensationId>,
        occurred_at_ms: TimeMs,
        observed_at_ms: TimeMs,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            kind: kind.into(),
            text: text.into(),
            impression_ids,
            sensation_ids,
            occurred_at_ms,
            observed_at_ms,
            salience: 0.5,
            tags: Vec::new(),
            payload: Value::Null,
        }
    }

    pub fn to_recall_sensation(
        &self,
        recall_at_ms: TimeMs,
        score: f32,
        stage: impl Into<String>,
    ) -> Sensation {
        let payload = json!({
            "experience": self,
            "original_experience_id": self.id,
            "original_occurred_at_ms": self.occurred_at_ms,
            "original_observed_at_ms": self.observed_at_ms,
            "score": score,
        });
        Sensation::new(
            "memory.related_experience",
            "memory",
            recall_at_ms,
            recall_at_ms,
            payload,
        )
        .with_summary(format!("I remember: {}", self.text))
        .with_provenance(Provenance::memory_recall(self.id).with_stage(stage))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecalledExperience {
    pub score: f32,
    pub experience: Experience,
    pub sensation: Sensation,
}
