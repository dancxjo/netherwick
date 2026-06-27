use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use base64::Engine;
use netherwick_actions::{action_to_motor_command, ActionPrimitive, ExploreStyle, TurnDir};
use netherwick_body::BodySense;
use netherwick_core::{ExperienceId, ImpressionId, Provenance, Reward, SensationId, TimeMs};
use netherwick_now::{DriveSense, MemorySense, Now, SenseVectorizer, SurpriseSense};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

const DEFAULT_WINDOW_MS: TimeMs = 750;
const PLACEHOLDER_VECTOR_DIM: usize = 16;

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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FutureInput {
    pub latent: ExperienceLatent,
    pub action: ActionPrimitive,
    pub offset_ms: TimeMs,
}

impl FutureInput {
    pub fn flat_features(&self) -> Vec<f32> {
        let mut out =
            Vec::with_capacity(self.latent.z.len() + action_features(Some(&self.action)).len() + 1);
        out.extend(self.latent.z.iter().copied().map(sanitize_feature));
        out.extend(
            action_features(Some(&self.action))
                .into_iter()
                .map(sanitize_feature),
        );
        out.push((self.offset_ms as f32 / 1_000.0).clamp(0.0, 60.0));
        out
    }
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
        cliff: bool01(cliff_detected(after)),
        wheel_drop: bool01(after.body.flags.wheel_drop),
        stuck: bool01(commanded_motion && no_forward_velocity && no_odometry),
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChargeInput {
    pub z: Vec<f32>,
    pub action_features: Vec<f32>,
    pub body_features: Vec<f32>,
    pub memory_features: Vec<f32>,
}

impl ChargeInput {
    pub fn from_parts(z: Vec<f32>, action: Option<&ActionPrimitive>, now: &Now) -> Self {
        Self {
            z,
            action_features: danger_action_features(action),
            body_features: charge_body_features(now),
            memory_features: charge_memory_features(now),
        }
    }

    pub fn flat_features(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(
            self.z.len()
                + self.action_features.len()
                + self.body_features.len()
                + self.memory_features.len(),
        );
        out.extend(self.z.iter().copied().map(sanitize_feature));
        out.extend(self.action_features.iter().copied().map(sanitize_feature));
        out.extend(self.body_features.iter().copied().map(sanitize_feature));
        out.extend(self.memory_features.iter().copied().map(sanitize_feature));
        out
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChargeOutput {
    pub charge_probability: f32,
    pub expected_battery_delta: f32,
    pub dock_likelihood: f32,
    pub confidence: f32,
}

impl ChargeOutput {
    pub fn values(&self) -> [f32; 3] {
        [
            self.charge_probability,
            self.expected_battery_delta,
            self.dock_likelihood,
        ]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChargeTarget {
    pub charging_started: f32,
    pub battery_delta: f32,
    pub charging_after: f32,
}

impl ChargeTarget {
    pub fn values(&self) -> [f32; 3] {
        [
            self.charging_started,
            self.battery_delta,
            self.charging_after,
        ]
    }
}

pub fn charge_input_from_transition_like(
    before_z: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
    before: &Now,
) -> ChargeInput {
    ChargeInput::from_parts(before_z.z.clone(), action, before)
}

pub fn charge_target_from_transition_like(
    before: &Now,
    _action: Option<&ActionPrimitive>,
    after: &Now,
) -> ChargeTarget {
    ChargeTarget {
        charging_started: bool01(!before.body.charging && after.body.charging),
        battery_delta: after.body.battery_level - before.body.battery_level,
        charging_after: bool01(after.body.charging),
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ActionValueInput {
    pub z: Vec<f32>,
    pub action_features: Vec<f32>,
    pub body_features: Vec<f32>,
    pub drive_features: Vec<f32>,
    pub memory_features: Vec<f32>,
    pub prediction_features: Vec<f32>,
}

impl ActionValueInput {
    pub fn from_parts(z: Vec<f32>, action: Option<&ActionPrimitive>, now: &Now) -> Self {
        Self {
            z,
            action_features: danger_action_features(action),
            body_features: action_value_body_features(now),
            drive_features: action_value_drive_features(now),
            memory_features: action_value_memory_features(now, action),
            prediction_features: action_value_prediction_features(now),
        }
    }

    pub fn from_parts_with_predictions(
        z: Vec<f32>,
        action: Option<&ActionPrimitive>,
        now: &Now,
        danger: Option<DangerOutput>,
        charge: Option<ChargeOutput>,
    ) -> Self {
        Self {
            z,
            action_features: danger_action_features(action),
            body_features: action_value_body_features(now),
            drive_features: action_value_drive_features(now),
            memory_features: action_value_memory_features(now, action),
            prediction_features: action_value_prediction_features_from_outputs(now, danger, charge),
        }
    }

    pub fn flat_features(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(
            self.z.len()
                + self.action_features.len()
                + self.body_features.len()
                + self.drive_features.len()
                + self.memory_features.len()
                + self.prediction_features.len(),
        );
        out.extend(self.z.iter().copied().map(sanitize_feature));
        out.extend(self.action_features.iter().copied().map(sanitize_feature));
        out.extend(self.body_features.iter().copied().map(sanitize_feature));
        out.extend(self.drive_features.iter().copied().map(sanitize_feature));
        out.extend(self.memory_features.iter().copied().map(sanitize_feature));
        out.extend(
            self.prediction_features
                .iter()
                .copied()
                .map(sanitize_feature),
        );
        out
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ActionValueOutput {
    pub value: f32,
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ActionValueTarget {
    pub value: f32,
}

pub const EYE_NEXT_WIDTH: u32 = 64;
pub const EYE_NEXT_HEIGHT: u32 = 48;
pub const EYE_NEXT_RGB_LEN: usize = EYE_NEXT_WIDTH as usize * EYE_NEXT_HEIGHT as usize * 3;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EyeNextInput {
    pub z: Vec<f32>,
    pub action_features: Vec<f32>,
    pub eye_features: Vec<f32>,
    pub body_features: Vec<f32>,
    pub offset_ms: TimeMs,
}

impl EyeNextInput {
    pub fn from_parts(
        z: Vec<f32>,
        action: Option<&ActionPrimitive>,
        now: &Now,
        offset_ms: TimeMs,
    ) -> Self {
        Self {
            z,
            action_features: danger_action_features(action),
            eye_features: eye_next_features(now),
            body_features: action_value_body_features(now),
            offset_ms,
        }
    }

    pub fn flat_features(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(
            self.z.len()
                + self.action_features.len()
                + self.eye_features.len()
                + self.body_features.len()
                + 1,
        );
        out.extend(self.z.iter().copied().map(sanitize_feature));
        out.extend(self.action_features.iter().copied().map(sanitize_feature));
        out.extend(self.eye_features.iter().copied().map(sanitize_feature));
        out.extend(self.body_features.iter().copied().map(sanitize_feature));
        out.push((self.offset_ms as f32 / 5_000.0).clamp(0.0, 1.0));
        out
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EyeNextOutput {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EyeNextTarget {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EarNextInput {
    pub z: Vec<f32>,
    pub action_features: Vec<f32>,
    pub ear_features: Vec<f32>,
    pub body_features: Vec<f32>,
    pub offset_ms: TimeMs,
}

impl EarNextInput {
    pub fn from_parts(
        z: Vec<f32>,
        action: Option<&ActionPrimitive>,
        now: &Now,
        offset_ms: TimeMs,
    ) -> Self {
        Self {
            z,
            action_features: danger_action_features(action),
            ear_features: ear_next_features(now),
            body_features: action_value_body_features(now),
            offset_ms,
        }
    }

    pub fn flat_features(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(
            self.z.len()
                + self.action_features.len()
                + self.ear_features.len()
                + self.body_features.len()
                + 1,
        );
        out.extend(self.z.iter().copied().map(sanitize_feature));
        out.extend(self.action_features.iter().copied().map(sanitize_feature));
        out.extend(self.ear_features.iter().copied().map(sanitize_feature));
        out.extend(self.body_features.iter().copied().map(sanitize_feature));
        out.push((self.offset_ms as f32 / 5_000.0).clamp(0.0, 1.0));
        out
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EarNextOutput {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub pcm: Vec<i16>,
    pub features: Vec<f32>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EarNextTarget {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub pcm: Vec<i16>,
    pub features: Vec<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExperienceEncodeInput {
    pub sense_vectors: Vec<Vec<f32>>,
}

impl ExperienceEncodeInput {
    pub fn flat_features(&self) -> Vec<f32> {
        self.sense_vectors
            .iter()
            .flat_map(|sense| sense.iter().copied().map(sanitize_feature))
            .collect()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExperienceEncodeOutput {
    pub z: Vec<f32>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExperienceDecodeOutput {
    pub body_features: Vec<f32>,
    pub memory_features: Vec<f32>,
    pub drive_features: Vec<f32>,
    pub prediction_features: Vec<f32>,
    pub eye_features: Vec<f32>,
    pub ear_features: Vec<f32>,
}

impl ExperienceDecodeOutput {
    pub fn flat_features(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(
            self.body_features.len()
                + self.memory_features.len()
                + self.drive_features.len()
                + self.prediction_features.len()
                + self.eye_features.len()
                + self.ear_features.len(),
        );
        out.extend(self.body_features.iter().copied().map(sanitize_feature));
        out.extend(self.memory_features.iter().copied().map(sanitize_feature));
        out.extend(self.drive_features.iter().copied().map(sanitize_feature));
        out.extend(
            self.prediction_features
                .iter()
                .copied()
                .map(sanitize_feature),
        );
        out.extend(self.eye_features.iter().copied().map(sanitize_feature));
        out.extend(self.ear_features.iter().copied().map(sanitize_feature));
        out
    }

    pub fn feature_lengths(&self) -> ExperienceDecodeFeatureLengths {
        ExperienceDecodeFeatureLengths {
            body: self.body_features.len(),
            memory: self.memory_features.len(),
            drive: self.drive_features.len(),
            prediction: self.prediction_features.len(),
            eye: self.eye_features.len(),
            ear: self.ear_features.len(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExperienceDecodeFeatureLengths {
    pub body: usize,
    pub memory: usize,
    pub drive: usize,
    pub prediction: usize,
    pub eye: usize,
    pub ear: usize,
}

pub fn experience_encode_input_from_now(now: &Now) -> ExperienceEncodeInput {
    let target = experience_decode_target_from_now(now);
    ExperienceEncodeInput {
        sense_vectors: vec![
            target.body_features,
            target.memory_features,
            target.drive_features,
            target.prediction_features,
            target.eye_features,
            target.ear_features,
        ],
    }
}

pub fn experience_decode_target_from_now(now: &Now) -> ExperienceDecodeOutput {
    ExperienceDecodeOutput {
        body_features: action_value_body_features(now),
        memory_features: action_value_memory_features(now, None),
        drive_features: action_value_drive_features(now),
        prediction_features: action_value_prediction_features(now),
        eye_features: eye_next_features(now),
        ear_features: ear_next_features(now),
    }
}

pub fn eye_next_input_from_transition_like(
    before_z: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
    before: &Now,
    offset_ms: TimeMs,
) -> EyeNextInput {
    EyeNextInput::from_parts(before_z.z.clone(), action, before, offset_ms)
}

pub fn eye_next_target_from_now(after: &Now) -> Option<EyeNextTarget> {
    eye_frame_rgb(after).map(|(width, height, rgb)| EyeNextTarget { width, height, rgb })
}

pub fn ear_next_input_from_transition_like(
    before_z: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
    before: &Now,
    offset_ms: TimeMs,
) -> EarNextInput {
    EarNextInput::from_parts(before_z.z.clone(), action, before, offset_ms)
}

pub fn ear_next_target_from_now(after: &Now) -> Option<EarNextTarget> {
    let features = ear_frame_features(after)?;
    Some(EarNextTarget {
        sample_rate_hz: 0,
        channels: 0,
        pcm: Vec::new(),
        features,
    })
}

pub fn eye_frame_rgb(now: &Now) -> Option<(u32, u32, Vec<u8>)> {
    let frame = now.eye.frames.last()?;
    let mut rgb = Vec::with_capacity(EYE_NEXT_RGB_LEN);
    if frame.len() >= EYE_NEXT_RGB_LEN {
        rgb.extend(
            frame
                .iter()
                .take(EYE_NEXT_RGB_LEN)
                .map(|value| unit_to_u8(*value)),
        );
    } else {
        for index in 0..(EYE_NEXT_WIDTH as usize * EYE_NEXT_HEIGHT as usize) {
            let value = frame
                .get(index % frame.len().max(1))
                .copied()
                .unwrap_or_default();
            let byte = unit_to_u8(value / 3.0);
            rgb.extend([byte, byte, byte]);
        }
    }
    Some((EYE_NEXT_WIDTH, EYE_NEXT_HEIGHT, rgb))
}

pub fn ear_frame_features(now: &Now) -> Option<Vec<f32>> {
    if now.ear.features.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    for feature in &now.ear.features {
        out.extend(feature.iter().copied().map(sanitize_feature));
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

pub fn action_value_input_from_transition_like(
    before_z: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
    before: &Now,
) -> ActionValueInput {
    ActionValueInput::from_parts(before_z.z.clone(), action, before)
}

pub fn action_value_target_from_reward_surprise(
    reward: &Reward,
    surprise: &SurpriseSense,
) -> ActionValueTarget {
    let hazard_surprise = (surprise.total - surprise.prediction_error).max(0.0);
    ActionValueTarget {
        value: reward.value + surprise.prediction_error * 0.05 - hazard_surprise * 0.15,
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
                BaselineSenseVectorizer::Asr,
                BaselineSenseVectorizer::Range,
                BaselineSenseVectorizer::KinectIr,
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
        let drives = (latent.z.len() >= 27).then(|| DriveSense {
            battery_hunger: latent
                .z
                .get(21)
                .copied()
                .unwrap_or_default()
                .clamp(0.0, 1.0),
            danger_avoidance: latent
                .z
                .get(22)
                .copied()
                .unwrap_or_default()
                .clamp(0.0, 1.0),
            curiosity: latent
                .z
                .get(23)
                .copied()
                .unwrap_or_default()
                .clamp(0.0, 1.0),
            social_interest: latent
                .z
                .get(24)
                .copied()
                .unwrap_or_default()
                .clamp(0.0, 1.0),
            fatigue: latent
                .z
                .get(25)
                .copied()
                .unwrap_or_default()
                .clamp(0.0, 1.0),
            uncertainty_pressure: latent
                .z
                .get(26)
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
        if cliff_detected(actual_now) || actual_now.body.flags.wheel_drop {
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
        let hazard = after.body.flags.bump_left
            || after.body.flags.bump_right
            || cliff_detected(after)
            || after.body.flags.wheel_drop;
        if !before.body.charging && after.body.charging && before.body.battery_level < 0.35 {
            value += 0.35;
        }
        if !hazard {
            value += 0.01;
        }
        if after.body.flags.bump_left || after.body.flags.bump_right {
            value -= 0.25;
        }
        if cliff_detected(after) || after.body.flags.wheel_drop {
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
        let hazard_surprise = (surprise.total - surprise.prediction_error).max(0.0);
        value -= hazard_surprise * 0.03;
        if !hazard {
            value += discovery_bonus(before, action, after, surprise);
        }
        Reward { value }
    }
}

fn discovery_bonus(
    before: &Now,
    action: Option<&ActionPrimitive>,
    after: &Now,
    surprise: &SurpriseSense,
) -> f32 {
    let novelty = after.memory.place_novelty.clamp(0.0, 1.0);
    let novelty_delta = (after.memory.place_novelty - before.memory.place_novelty)
        .max(0.0)
        .clamp(0.0, 1.0);
    let newly_visited_places = after
        .memory
        .places_visited
        .saturating_sub(before.memory.places_visited)
        .min(3) as f32;
    let dx = after.body.odometry.x_m - before.body.odometry.x_m;
    let dy = after.body.odometry.y_m - before.body.odometry.y_m;
    let distance_m = (dx * dx + dy * dy).sqrt().clamp(0.0, 0.25);
    let motion_bonus = if matches!(
        action,
        Some(
            ActionPrimitive::Go { .. }
                | ActionPrimitive::Turn { .. }
                | ActionPrimitive::Inspect { .. }
                | ActionPrimitive::Explore { .. }
        )
    ) {
        distance_m * 0.16
    } else {
        0.0
    };
    let prediction_bonus = if matches!(
        action,
        Some(
            ActionPrimitive::Go { .. }
                | ActionPrimitive::Turn { .. }
                | ActionPrimitive::Inspect { .. }
                | ActionPrimitive::Explore { .. }
        )
    ) {
        surprise.prediction_error.clamp(0.0, 1.0) * 0.03
    } else {
        0.0
    };

    (novelty * 0.04
        + novelty_delta * 0.06
        + newly_visited_places * 0.03
        + motion_bonus
        + prediction_bonus)
        .clamp(0.0, 0.12)
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
    Asr,
    Range,
    KinectIr,
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
            Self::Asr => "asr",
            Self::Range => "range",
            Self::KinectIr => "kinect_ir",
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
                bool01(cliff_detected(now)),
                bool01(now.body.flags.wheel_drop),
                bool01(now.body.flags.wall || now.body.flags.virtual_wall),
                now.body.velocity.forward_m_s.clamp(-1.0, 1.0),
                now.body.velocity.turn_rad_s.clamp(-1.0, 1.0),
                now.body.health.strain.clamp(0.0, 1.0),
                now.body.health.health.clamp(0.0, 1.0),
                now.body.cliff_sensors.left.clamp(0.0, 1.0),
                now.body.cliff_sensors.front_left.clamp(0.0, 1.0),
                now.body.cliff_sensors.front_right.clamp(0.0, 1.0),
                now.body.cliff_sensors.right.clamp(0.0, 1.0),
            ],
            Self::Memory => vec![
                now.memory.place_familiarity.clamp(0.0, 1.0),
                now.memory.place_danger.clamp(0.0, 1.0),
                now.memory.place_charge_value.clamp(0.0, 1.0),
                now.memory.face_familiarity.clamp(0.0, 1.0),
                now.memory.voice_familiarity.clamp(0.0, 1.0),
                (now.memory.similar_situation_count as f32 / 32.0).clamp(0.0, 1.0),
                bool01(now.memory.remembered_warning.is_some()),
                graph_label_pressure(now, "Person"),
                graph_label_pressure(now, "Place"),
                graph_label_pressure(now, "Experience"),
                (now.memory.remembered_relationships.len() as f32 / 32.0).clamp(0.0, 1.0),
                bool01(now.memory.graph_context_summary.is_some()),
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
            Self::Asr => asr_features(now),
            Self::Range => vec![now
                .range
                .nearest_m
                .map(|m| (1.0 / (1.0 + m)).clamp(0.0, 1.0))
                .unwrap_or(0.0)],
            Self::KinectIr => kinect_ir_features(now),
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

fn graph_label_pressure(now: &Now, label: &str) -> f32 {
    now.memory
        .remembered_entities
        .iter()
        .filter(|entity| entity.has_label(label))
        .map(|entity| entity.score.clamp(0.0, 1.0))
        .fold(0.0f32, f32::max)
}

pub fn action_features(action: Option<&ActionPrimitive>) -> Vec<f32> {
    danger_action_features(action)
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
        bool01(cliff_detected(now)),
        bool01(now.body.flags.wheel_drop),
        bool01(now.body.flags.wall || now.body.flags.virtual_wall),
        now.body.velocity.forward_m_s.clamp(-1.0, 1.0),
        now.body.velocity.turn_rad_s.clamp(-1.0, 1.0),
        now.body.health.strain.clamp(0.0, 1.0),
        now.body.health.health.clamp(0.0, 1.0),
        now.body.cliff_sensors.left.clamp(0.0, 1.0),
        now.body.cliff_sensors.front_left.clamp(0.0, 1.0),
        now.body.cliff_sensors.front_right.clamp(0.0, 1.0),
        now.body.cliff_sensors.right.clamp(0.0, 1.0),
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

fn charge_body_features(now: &Now) -> Vec<f32> {
    let mut out = vec![
        now.body.battery_level.clamp(0.0, 1.0),
        bool01(now.body.charging),
        now.drives.battery_hunger.clamp(0.0, 1.0),
        now.body.velocity.forward_m_s.clamp(-1.0, 1.0),
        now.body.velocity.turn_rad_s.clamp(-1.0, 1.0),
        now.range
            .nearest_m
            .map(|m| (1.0 / (1.0 + m)).clamp(0.0, 1.0))
            .unwrap_or(0.0),
        extension_value(now, "sim.world", 3)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0),
        extension_value(now, "sim.world", 4)
            .unwrap_or(0.0)
            .clamp(0.0, 1.0),
    ];
    out.extend(kinect_ir_features(now));
    out
}

fn charge_memory_features(now: &Now) -> Vec<f32> {
    vec![
        now.memory.place_charge_value.clamp(0.0, 1.0),
        now.memory.place_familiarity.clamp(0.0, 1.0),
        now.memory.place_danger.clamp(0.0, 1.0),
        (now.memory.similar_situation_count as f32 / 32.0).clamp(0.0, 1.0),
        bool01(matches!(
            now.memory.best_remembered_action,
            Some(ActionPrimitive::Dock)
        )),
    ]
}

fn action_value_body_features(now: &Now) -> Vec<f32> {
    let mut out = danger_body_features(now);
    out.extend(charge_body_features(now));
    out
}

fn action_value_drive_features(now: &Now) -> Vec<f32> {
    vec![
        now.drives.battery_hunger.clamp(0.0, 1.0),
        now.drives.danger_avoidance.clamp(0.0, 1.0),
        now.drives.curiosity.clamp(0.0, 1.0),
        now.drives.social_interest.clamp(0.0, 1.0),
        now.drives.fatigue.clamp(0.0, 1.0),
        now.drives.uncertainty_pressure.clamp(0.0, 1.0),
    ]
}

fn action_value_memory_features(now: &Now, action: Option<&ActionPrimitive>) -> Vec<f32> {
    let best_matches = match (&now.memory.best_remembered_action, action) {
        (Some(best), Some(action)) => best == action,
        _ => false,
    };
    vec![
        now.memory.place_familiarity.clamp(0.0, 1.0),
        now.memory.place_danger.clamp(0.0, 1.0),
        now.memory.place_charge_value.clamp(0.0, 1.0),
        now.memory.face_familiarity.clamp(0.0, 1.0),
        now.memory.voice_familiarity.clamp(0.0, 1.0),
        (now.memory.similar_situation_count as f32 / 32.0).clamp(0.0, 1.0),
        bool01(best_matches),
        bool01(now.memory.remembered_warning.is_some()),
    ]
}

fn action_value_prediction_features(now: &Now) -> Vec<f32> {
    let danger = now.predictions.danger_model.map(|value| DangerOutput {
        bump_risk: value.bump_risk,
        cliff_risk: value.cliff_risk,
        wheel_drop_risk: value.wheel_drop_risk,
        stuck_risk: value.stuck_risk,
        confidence: value.confidence,
    });
    let charge = now.predictions.charge_model.map(|value| ChargeOutput {
        charge_probability: value.charge_probability,
        expected_battery_delta: value.expected_battery_delta,
        dock_likelihood: value.dock_likelihood,
        confidence: value.confidence,
    });
    action_value_prediction_features_from_outputs(now, danger, charge)
}

fn action_value_prediction_features_from_outputs(
    now: &Now,
    danger: Option<DangerOutput>,
    charge: Option<ChargeOutput>,
) -> Vec<f32> {
    let danger = danger.unwrap_or_default();
    let charge = charge.unwrap_or_default();
    vec![
        danger.bump_risk.clamp(0.0, 1.0),
        danger.cliff_risk.clamp(0.0, 1.0),
        danger.wheel_drop_risk.clamp(0.0, 1.0),
        danger.stuck_risk.clamp(0.0, 1.0),
        danger.confidence.clamp(0.0, 1.0),
        charge.charge_probability.clamp(0.0, 1.0),
        charge.expected_battery_delta.clamp(-1.0, 1.0),
        charge.dock_likelihood.clamp(0.0, 1.0),
        charge.confidence.clamp(0.0, 1.0),
        now.predictions.uncertainty.clamp(0.0, 1.0),
        (now.predictions.expected_events.len() as f32 / 8.0).clamp(0.0, 1.0),
        bool01(
            now.extensions
                .get("safety.vetoed")
                .and_then(|value| value.as_bool())
                .unwrap_or(false),
        ),
    ]
}

fn eye_next_features(now: &Now) -> Vec<f32> {
    let Some(frame) = now.eye.frames.last() else {
        return vec![0.0; 16];
    };
    if frame.is_empty() {
        return vec![0.0; 16];
    }
    let mut out = Vec::with_capacity(16);
    let chunk = (frame.len() / 16).max(1);
    for part in frame.chunks(chunk).take(16) {
        let avg = part.iter().copied().map(sanitize_feature).sum::<f32>() / part.len() as f32;
        out.push(avg.clamp(0.0, 1.0));
    }
    out.resize(16, 0.0);
    out
}

fn ear_next_features(now: &Now) -> Vec<f32> {
    let mut out = Vec::with_capacity(16);
    if let Some(features) = ear_frame_features(now) {
        let chunk = (features.len() / 8).max(1);
        for part in features.chunks(chunk).take(8) {
            let avg = part.iter().copied().map(sanitize_feature).sum::<f32>() / part.len() as f32;
            out.push(avg.clamp(-1.0, 1.0));
        }
    }
    out.resize(8, 0.0);
    out.extend(asr_features(now));
    out
}

fn asr_features(now: &Now) -> Vec<f32> {
    let transcript = now
        .ear
        .asr
        .transcript
        .as_deref()
        .or(now.ear.transcript.as_deref())
        .map(str::trim)
        .filter(|text| !text.is_empty());
    let Some(transcript) = transcript else {
        return vec![0.0; 8];
    };

    let word_count = now
        .ear
        .asr
        .word_count
        .map(f32::from)
        .unwrap_or_else(|| count_transcript_words(transcript) as f32);
    let char_count = transcript.chars().count() as f32;
    let punctuation_count = transcript
        .chars()
        .filter(|ch| matches!(ch, '.' | ',' | '?' | '!' | ';' | ':'))
        .count() as f32;
    let duration_ms = now
        .ear
        .asr
        .duration_ms
        .or_else(|| Some(now.ear.asr.end_ms?.saturating_sub(now.ear.asr.start_ms?)))
        .unwrap_or_default();
    let sequence_span = match (now.ear.asr.sequence_start, now.ear.asr.sequence_end) {
        (Some(start), Some(end)) => end.saturating_sub(start).saturating_add(1),
        _ => 0,
    };

    vec![
        1.0,
        bool01(now.ear.asr.is_final),
        now.ear.asr.confidence.clamp(0.0, 1.0),
        (word_count / 32.0).clamp(0.0, 1.0),
        (char_count / 160.0).clamp(0.0, 1.0),
        (duration_ms as f32 / 20_000.0).clamp(0.0, 1.0),
        (sequence_span as f32 / 128.0).clamp(0.0, 1.0),
        (punctuation_count / 8.0).clamp(0.0, 1.0),
    ]
}

fn count_transcript_words(transcript: &str) -> usize {
    transcript
        .split_whitespace()
        .filter(|word| word.chars().any(|ch| ch.is_alphanumeric()))
        .count()
}

fn kinect_ir_features(now: &Now) -> Vec<f32> {
    if now.kinect.ir.is_empty() {
        return vec![0.0, 0.0, 0.0, 0.0];
    }
    let len = now.kinect.ir.len() as f32;
    let sum = now.kinect.ir.iter().copied().sum::<f32>();
    let max = now
        .kinect
        .ir
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let bright = now.kinect.ir.iter().filter(|value| **value >= 0.7).count() as f32 / len;
    vec![
        (sum / len).clamp(0.0, 1.0),
        max.clamp(0.0, 1.0),
        bright.clamp(0.0, 1.0),
        (len / 1024.0).clamp(0.0, 1.0),
    ]
}

fn extension_value(now: &Now, name: &str, index: usize) -> Option<f32> {
    now.extensions
        .get(name)?
        .get("values")?
        .as_array()?
        .get(index)?
        .as_f64()
        .map(|value| value as f32)
}

fn cliff_detected(now: &Now) -> bool {
    now.body.flags.cliff_left
        || now.body.flags.cliff_front_left
        || now.body.flags.cliff_front_right
        || now.body.flags.cliff_right
        || now.body.cliff_sensors.max() >= 0.5
}

fn sanitize_feature(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn unit_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
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
    use netherwick_now::{AsrSense, Now};

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
    fn now_with_sensors_produces_non_empty_experience_encoder_input() {
        let mut now = Now::blank(42, BodySense::default());
        now.eye.frames = vec![vec![0.2, 0.4, 0.6, 0.8]];
        now.ear.features = vec![vec![0.1, 0.3, 0.5, 0.7]];
        now.memory.place_familiarity = 0.7;
        now.drives.curiosity = 0.5;

        let input = experience_encode_input_from_now(&now);
        let target = experience_decode_target_from_now(&now);

        assert_eq!(input.sense_vectors.len(), 6);
        assert!(!input.flat_features().is_empty());
        assert_eq!(input.flat_features().len(), target.flat_features().len());
        assert_eq!(target.eye_features.len(), 16);
        assert_eq!(target.ear_features.len(), 16);
    }

    #[test]
    fn ear_features_include_finalized_asr_metadata() {
        let mut now = Now::blank(42, BodySense::default());
        now.ear.features = vec![vec![0.1, 0.3, 0.5, 0.7]];
        now.ear.asr = AsrSense {
            transcript: Some("hello world again".to_string()),
            is_final: true,
            confidence: 0.72,
            sequence_start: Some(10),
            sequence_end: Some(13),
            start_ms: Some(100),
            end_ms: Some(1_100),
            duration_ms: Some(1_000),
            sample_rate_hz: Some(16_000),
            word_count: Some(3),
            speaker_confidence: Some(0.6),
        };

        let features = ear_next_features(&now);

        assert_eq!(features.len(), 16);
        assert_eq!(&features[..4], &[0.1, 0.3, 0.5, 0.7]);
        assert_eq!(features[8], 1.0);
        assert_eq!(features[9], 1.0);
        assert_eq!(features[10], 0.72);
        assert!(features[11] > 0.0);
        assert!(features[13] > 0.0);
        assert!(features[14] > 0.0);
    }

    #[test]
    fn transcript_only_asr_still_reaches_now_vector() {
        let mut now = Now::blank(42, BodySense::default());
        now.ear.transcript = Some("come over here".to_string());

        let target = experience_decode_target_from_now(&now);
        let asr = asr_features(&now);

        assert_eq!(target.ear_features.len(), 16);
        assert_eq!(target.ear_features[8], 1.0);
        assert_eq!(asr[0], 1.0);
        assert!(asr[3] > 0.0);
        assert!(target.flat_features().iter().any(|value| *value > 0.0));
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
    fn reward_tastes_safe_discovery_as_good() {
        let computer = BaselineRewardComputer;
        let mut before = Now::blank(1, BodySense::default());
        before.memory.place_novelty = 0.1;
        before.memory.places_visited = 1;
        let mut after = before.clone();
        after.t_ms = 2;
        after.memory.place_novelty = 0.8;
        after.memory.places_visited = 2;
        after.body.odometry.x_m = 0.12;
        after.body.velocity.forward_m_s = 0.12;

        let reward = computer.compute(
            &before,
            Some(&ActionPrimitive::Explore {
                style: ExploreStyle::Wander,
                duration_ms: 1_000,
            }),
            &after,
            &SurpriseSense {
                total: 0.4,
                prediction_error: 0.4,
                ..SurpriseSense::default()
            },
        );

        assert!(reward.value > 0.08);
    }

    #[test]
    fn reward_keeps_hazard_surprise_negative() {
        let computer = BaselineRewardComputer;
        let before = Now::blank(1, BodySense::default());
        let mut after = before.clone();
        after.t_ms = 2;
        after.body.flags.bump_left = true;

        let reward = computer.compute(
            &before,
            Some(&ActionPrimitive::Go {
                intensity: 0.2,
                duration_ms: 1_000,
            }),
            &after,
            &SurpriseSense {
                total: 0.6,
                prediction_error: 0.1,
                ..SurpriseSense::default()
            },
        );

        assert!(reward.value < -0.2);
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

    #[test]
    fn danger_input_includes_cliff_sensor_channels() {
        let mut now = Now::blank(1, BodySense::default());
        now.body.cliff_sensors.front_left = 0.8;

        let input = DangerInput::from_parts(vec![0.0], Some(&ActionPrimitive::Stop), &now);

        assert!(input.body_features.contains(&0.8));
        assert_eq!(input.body_features[3], 1.0);
    }

    #[test]
    fn charge_target_marks_transition_onto_charger() {
        let mut before = Now::blank(1, BodySense::default());
        before.body.charging = false;
        before.body.battery_level = 0.2;
        let mut after = before.clone();
        after.body.charging = true;
        after.body.battery_level = 0.24;

        let target =
            charge_target_from_transition_like(&before, Some(&ActionPrimitive::Dock), &after);

        assert_eq!(target.charging_started, 1.0);
        assert_eq!(target.charging_after, 1.0);
        assert!(target.battery_delta > 0.0);
    }

    #[test]
    fn charge_target_marks_transition_off_charger() {
        let mut before = Now::blank(1, BodySense::default());
        before.body.charging = true;
        let mut after = before.clone();
        after.body.charging = false;

        let target =
            charge_target_from_transition_like(&before, Some(&ActionPrimitive::Stop), &after);

        assert_eq!(target.charging_started, 0.0);
        assert_eq!(target.charging_after, 0.0);
    }

    #[test]
    fn charge_input_includes_ir_sensor_summary() {
        let mut now = Now::blank(1, BodySense::default());
        now.kinect.ir = vec![0.1, 0.8, 0.9, 0.2];

        let input = ChargeInput::from_parts(vec![0.0], Some(&ActionPrimitive::Dock), &now);

        assert!(input.body_features.iter().any(|value| *value >= 0.8));
    }

    #[test]
    fn action_value_input_includes_input_sensor_channels() {
        let mut now = Now::blank(1, BodySense::default());
        now.body.cliff_sensors.front_right = 0.7;
        now.kinect.ir = vec![0.1, 0.8, 0.9, 0.2];

        let input = ActionValueInput::from_parts(vec![0.0], Some(&ActionPrimitive::Dock), &now);

        assert!(input.body_features.contains(&0.7));
        assert!(input.body_features.iter().any(|value| *value >= 0.8));
    }

    #[test]
    fn action_value_target_positive_for_charging_reward() {
        let reward = Reward { value: 0.35 };
        let surprise = SurpriseSense {
            total: 0.2,
            ..SurpriseSense::default()
        };

        let target = action_value_target_from_reward_surprise(&reward, &surprise);

        assert!(target.value > 0.0);
    }

    #[test]
    fn action_value_target_values_safe_prediction_error() {
        let reward = Reward { value: 0.0 };
        let surprise = SurpriseSense {
            total: 0.5,
            prediction_error: 0.5,
            ..SurpriseSense::default()
        };

        let target = action_value_target_from_reward_surprise(&reward, &surprise);

        assert!(target.value > 0.0);
    }

    #[test]
    fn action_value_target_negative_for_bump_or_cliff_transition() {
        let reward = Reward { value: -0.8 };
        let surprise = SurpriseSense {
            total: 0.4,
            ..SurpriseSense::default()
        };

        let target = action_value_target_from_reward_surprise(&reward, &surprise);

        assert!(target.value < 0.0);
    }

    #[test]
    fn action_value_input_uses_prediction_channels() {
        let mut now = Now::blank(1, BodySense::default());
        now.predictions.danger_model = Some(netherwick_now::DangerPrediction {
            bump_risk: 0.7,
            confidence: 0.8,
            ..netherwick_now::DangerPrediction::default()
        });
        now.predictions.charge_model = Some(netherwick_now::ChargePrediction {
            charge_probability: 0.6,
            expected_battery_delta: 0.1,
            dock_likelihood: 0.5,
            confidence: 0.9,
        });

        let input = ActionValueInput::from_parts(vec![0.0], Some(&ActionPrimitive::Dock), &now);

        assert!(input.prediction_features.contains(&0.7));
        assert!(input.prediction_features.contains(&0.6));
    }

    #[test]
    fn deterministic_extractor_preserves_visual_lineage() {
        let primary = Sensation::primary(
            Modality::Vision,
            SensationSource::new("test-camera"),
            100,
            100,
            SensationPayload::image_metadata(64, 48, "rgb8", 64 * 48 * 3),
        );

        let descendants = DeterministicDescendantExtractor.extract(&primary).unwrap();

        assert_eq!(descendants.len(), 1);
        assert_eq!(descendants[0].parent_id, Some(primary.id));
        assert_eq!(descendants[0].payload_kind, SensationPayloadKind::Crop);
        assert!(matches!(
            descendants[0].provenance.kind,
            netherwick_core::ProvenanceKind::DerivedFromSensations { .. }
        ));
    }

    #[test]
    fn audio_extractor_derives_asr_voice_speech_and_transcript_spans() {
        let mut primary = Sensation::primary(
            Modality::Audio,
            SensationSource::new("test-ear"),
            1_000,
            1_900,
            SensationPayload {
                kind: SensationPayloadKind::AudioPcm,
                value: json!({
                    "feature_sets": 4,
                    "transcript": "hello there",
                    "asr": {
                        "transcript": "hello there",
                        "is_final": true,
                        "confidence": 0.82,
                        "start_ms": 1_000,
                        "end_ms": 1_900,
                        "duration_ms": 900,
                        "sample_rate_hz": 16_000,
                        "word_count": 2,
                        "speaker_confidence": 0.61,
                    },
                }),
            },
        );
        primary.metadata.duration_ms = Some(900);
        primary.metadata.confidence = Some(0.82);

        let descendants = DeterministicDescendantExtractor.extract(&primary).unwrap();

        assert!(descendants
            .iter()
            .any(|sensation| sensation.payload_kind == SensationPayloadKind::VoiceSegment));
        let speech = descendants
            .iter()
            .find(|sensation| sensation.payload_kind == SensationPayloadKind::SpeechSegment)
            .expect("speech span");
        assert_eq!(speech.parent_id, Some(primary.id));
        assert_eq!(speech.occurred_at_ms, 1_000);
        assert_eq!(speech.metadata.duration_ms, Some(900));
        assert_eq!(
            speech.payload.get("text").and_then(Value::as_str),
            Some("hello there")
        );
        assert!(speech
            .provenance
            .stage_chain
            .contains(&"descendant.audio_speech_span".to_string()));
        assert!(descendants
            .iter()
            .any(|sensation| sensation.payload_kind == SensationPayloadKind::TranscriptSpan));
        assert!(descendants.iter().any(|sensation| {
            sensation
                .summary
                .as_deref()
                .is_some_and(|summary| summary == "I hear someone say \"hello there\".")
        }));
    }

    #[test]
    fn audio_extractor_falls_back_to_deterministic_voice_windows() {
        let mut primary = Sensation::primary(
            Modality::Audio,
            SensationSource::new("test-ear"),
            2_000,
            4_600,
            SensationPayload {
                kind: SensationPayloadKind::AudioPcm,
                value: json!({
                    "feature_sets": 130,
                    "transcript": null,
                    "asr": {},
                }),
            },
        );
        primary.metadata.duration_ms = Some(2_600);
        primary.metadata.confidence = Some(0.35);

        let descendants = AudioDescendantExtractor.extract(&primary).unwrap();

        assert_eq!(descendants.len(), 3);
        assert!(descendants
            .iter()
            .all(|sensation| sensation.payload_kind == SensationPayloadKind::VoiceSegment));
        assert!(descendants
            .iter()
            .all(|sensation| sensation.parent_id == Some(primary.id)));
        assert_eq!(
            descendants[0].payload.get("method").and_then(Value::as_str),
            Some("deterministic_audio_features")
        );
        assert_eq!(
            descendants[0].summary.as_deref(),
            Some("I hear a voice nearby.")
        );
    }

    fn visual_primary_with_rgb(width: u32, height: u32, rgb: Vec<u8>) -> Sensation {
        let mut payload = SensationPayload::image_metadata(width, height, "rgb8", rgb.len());
        payload.value["raw_bytes_b64"] =
            Value::String(base64::engine::general_purpose::STANDARD.encode(rgb));
        Sensation::primary(
            Modality::Vision,
            SensationSource::new("test-camera"),
            100,
            110,
            payload,
        )
    }

    #[test]
    fn visual_detector_creates_face_crop_with_bbox_metadata() {
        let mut rgb = vec![8_u8; 64 * 48 * 3];
        for y in 12..32 {
            for x in 22..42 {
                let idx = (y * 64 + x) * 3;
                rgb[idx] = 225;
                rgb[idx + 1] = 168;
                rgb[idx + 2] = 115;
            }
        }
        let primary = visual_primary_with_rgb(64, 48, rgb);

        let descendants = VisualDescendantExtractor.extract(&primary).unwrap();

        assert_eq!(descendants.len(), 1);
        let crop = &descendants[0];
        assert_eq!(crop.parent_id, Some(primary.id));
        assert_eq!(crop.modality, Modality::Vision);
        assert_eq!(crop.payload_kind, SensationPayloadKind::Crop);
        assert_eq!(crop.kind, "vision.face_crop");
        assert_eq!(crop.metadata.bbox.unwrap().x, 22);
        assert!(crop.metadata.confidence.unwrap() > 0.4);
        assert!(crop.metadata.labels.contains(&"face".to_string()));
        assert_eq!(
            crop.metadata
                .properties
                .get("detection_kind")
                .and_then(Value::as_str),
            Some("face")
        );
        assert_eq!(
            crop.provenance.stage_chain,
            vec!["descendant.face_crop".to_string()]
        );
        assert!(crop.payload.get("raw_bytes_b64").is_some());
        assert!(crop.payload.get("crop_content_id").is_some());
    }

    #[test]
    fn visual_extractor_falls_back_to_center_crop_without_detector_output() {
        let primary = Sensation::primary(
            Modality::Vision,
            SensationSource::new("test-camera"),
            100,
            100,
            SensationPayload::image_metadata(64, 48, "rgb8", 64 * 48 * 3),
        );

        let descendants = VisualDescendantExtractor.extract(&primary).unwrap();

        assert_eq!(descendants.len(), 1);
        assert_eq!(descendants[0].kind, "vision.crop");
        assert_eq!(descendants[0].parent_id, Some(primary.id));
        assert_eq!(
            descendants[0].payload.get("method").and_then(Value::as_str),
            Some("deterministic_center_crop")
        );
        assert_eq!(descendants[0].metadata.bbox.unwrap().x, 16);
    }

    #[tokio::test]
    async fn embodied_pipeline_vectorizes_visual_crop_and_impression_text() {
        let mut rgb = vec![5_u8; 64 * 48 * 3];
        for y in 10..34 {
            for x in 20..44 {
                let idx = (y * 64 + x) * 3;
                rgb[idx] = 230;
                rgb[idx + 1] = 172;
                rgb[idx + 2] = 120;
            }
        }
        let primary = visual_primary_with_rgb(64, 48, rgb);

        let batch = EmbodiedPipeline::new()
            .ingest_primary(primary)
            .await
            .unwrap();

        let crop = batch
            .sensations
            .iter()
            .find(|sensation| sensation.kind == "vision.face_crop")
            .expect("face crop sensation");
        assert!(crop.vector.is_some(), "crop should be vectorized");
        assert!(batch
            .impressions
            .iter()
            .any(|impression| impression.sensation_id == Some(crop.id)
                && impression.text == "I see a face close to me."));
    }

    #[test]
    fn primary_sensation_from_now_preserves_raw_visual_bytes() {
        let mut now = Now::blank(200, BodySense::default());
        now.eye_frame = Some(netherwick_now::EyeFrame {
            captured_at_ms: 190,
            width: 2,
            height: 2,
            format: netherwick_now::EyeFrameFormat::Rgb8,
            bytes: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            source: Some("unit-camera".to_string()),
        });

        let sensations = primary_sensations_from_now(&now);

        let vision = sensations
            .iter()
            .find(|sensation| sensation.payload_kind == SensationPayloadKind::ImageBytes)
            .expect("vision primary");
        assert_eq!(
            vision
                .metadata
                .properties
                .get("raw_bytes_present")
                .and_then(Value::as_bool),
            Some(true)
        );
        let encoded = vision
            .payload
            .get("raw_bytes_b64")
            .and_then(Value::as_str)
            .expect("raw bytes payload");
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap();
        assert_eq!(decoded, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn experience_fuser_links_sensations_impressions_and_summary() {
        let mut sensation = Sensation::primary(
            Modality::Vision,
            SensationSource::new("test-camera"),
            100,
            110,
            SensationPayload::image_metadata(64, 48, "rgb8", 64 * 48 * 3),
        );
        sensation.vector = Some(VectorEmbedding::new(
            vec![0.1, 0.2, 0.3],
            "test-vectorizer",
            Modality::Vision,
            SensationPayloadKind::ImageBytes,
            sensation.id,
            110,
        ));
        let impression = TemplateImpressionGenerator.generate_for_sensation(&sensation);

        let experience = ExperienceFuser::new(750)
            .fuse(&[sensation.clone()], &[impression.clone()])
            .unwrap();

        assert_eq!(experience.sensation_ids, vec![sensation.id]);
        assert_eq!(experience.impression_ids, vec![impression.id]);
        assert_eq!(experience.window_start_ms, 100);
        assert_eq!(experience.window_end_ms, 110);
        assert!(experience.fused_vector.is_some());
        assert_eq!(
            experience
                .summary_impression
                .as_ref()
                .and_then(|summary| summary.experience_id),
            Some(experience.id)
        );
        assert!(experience.text.starts_with("I see"));
    }

    #[test]
    fn primary_sensations_from_now_lifts_live_sensor_surfaces() {
        let mut now = Now::blank(200, BodySense::default());
        now.eye_frame = Some(netherwick_now::EyeFrame {
            captured_at_ms: 190,
            width: 32,
            height: 24,
            format: netherwick_now::EyeFrameFormat::Rgb8,
            bytes: vec![0; 32 * 24 * 3],
            source: Some("unit-camera".to_string()),
        });
        now.ear.asr.transcript = Some("hello".to_string());
        now.ear.asr.confidence = 0.8;
        now.range.nearest_m = Some(0.4);
        now.kinect.depth_m = vec![1.0, 1.2, 1.4, 1.6];
        now.kinect.depth_width = 2;
        now.kinect.depth_height = 2;

        let sensations = primary_sensations_from_now(&now);

        assert!(sensations
            .iter()
            .any(|sensation| sensation.payload_kind == SensationPayloadKind::ImageBytes));
        assert!(sensations
            .iter()
            .any(|sensation| sensation.payload_kind == SensationPayloadKind::AudioPcm));
        assert!(sensations
            .iter()
            .any(|sensation| sensation.payload_kind == SensationPayloadKind::LidarScan));
        assert!(sensations
            .iter()
            .any(|sensation| sensation.payload_kind == SensationPayloadKind::DepthFrame));
    }

    #[tokio::test]
    async fn embodied_now_vectorizes_asr_audio_descendants() {
        let mut now = Now::blank(200, BodySense::default());
        now.ear.asr = AsrSense {
            transcript: Some("come closer".to_string()),
            is_final: true,
            confidence: 0.77,
            start_ms: Some(120),
            end_ms: Some(920),
            duration_ms: Some(800),
            word_count: Some(2),
            ..AsrSense::default()
        };

        let embodied = embody_now(&now).await.unwrap();

        let speech = embodied
            .sensations
            .iter()
            .find(|sensation| sensation.payload_kind == SensationPayloadKind::SpeechSegment)
            .expect("speech child sensation");
        assert!(speech.parent_id.is_some());
        assert!(speech.vector.is_some());
        assert_eq!(
            speech
                .impression
                .as_ref()
                .map(|impression| impression.text.as_str()),
            Some("I hear someone say \"come closer\".")
        );
        assert!(embodied
            .sensations
            .iter()
            .any(
                |sensation| sensation.payload_kind == SensationPayloadKind::TranscriptSpan
                    && sensation.vector.is_some()
            ));
    }

    #[test]
    fn embodied_context_from_current_experience_uses_traceable_sensation_lineage() {
        let primary = Sensation::primary(
            Modality::Vision,
            SensationSource::new("unit-camera"),
            100,
            105,
            SensationPayload::image_metadata(32, 24, "rgb8", 32 * 24 * 3),
        )
        .with_summary("I receive a visual frame.");
        let child = Sensation::descendant(
            &primary,
            "vision.crop.focus",
            SensationPayloadKind::Crop,
            json!({"x": 4, "y": 3, "width": 12, "height": 9}),
            SensationMetadata::default(),
            "focus",
        )
        .with_summary("I focus on a patch in the frame.")
        .with_vector(VectorEmbedding::new(
            vec![0.1, 0.2, 0.3],
            "unit.crop.v0",
            Modality::Vision,
            SensationPayloadKind::Crop,
            primary.id,
            106,
        ));
        let impression = Impression::new(
            "vision.focus.impression",
            "I see a frame and focus on part of it.",
            vec![primary.id, child.id],
            100,
            106,
        );
        let mut experience = Experience::new(
            "embodied.now",
            "I see a frame and focus on part of it.",
            vec![impression.id],
            vec![primary.id, child.id],
            100,
            106,
        );
        experience.fused_vector = Some(VectorEmbedding::new(
            vec![0.5, 0.6, 0.7, 0.8],
            "unit.fuser.v0",
            Modality::Other,
            SensationPayloadKind::Structured,
            child.id,
            106,
        ));
        experience.predictions.push(Prediction {
            offset_ms: 750,
            text: "I expect the focused view to remain similar.".to_string(),
            confidence: 0.4,
            vector: experience.fused_vector.clone(),
        });
        experience.memory_links.push(MemoryLink {
            target_id: "memory-1".to_string(),
            relation: "similar".to_string(),
            score: 0.7,
            payload: json!({"text": "A previous focused camera moment."}),
        });

        let context = EmbodiedContext::from_current_experience(
            Some(&experience),
            &[primary.clone(), child.clone()],
            &[impression],
            &[],
            &[],
        );

        assert_eq!(context.experience_id, Some(experience.id));
        assert_eq!(context.summary, experience.text);
        assert_eq!(context.sensations.len(), 2);
        assert_eq!(context.derived_sensation_count(), 1);
        assert_eq!(
            context.lineage,
            vec![EmbodiedLineageEdge {
                parent_id: primary.id,
                child_id: child.id,
            }]
        );
        assert_eq!(
            context
                .fused_vector
                .as_ref()
                .map(|vector| (vector.model_id.as_str(), vector.dim)),
            Some(("unit.fuser.v0", 4))
        );
        assert_eq!(
            context
                .sensation_vectors
                .iter()
                .map(|vector| (vector.model_id.as_str(), vector.dim))
                .collect::<Vec<_>>(),
            vec![("unit.crop.v0", 3)]
        );
        assert_eq!(context.predictions.len(), 1);
        assert_eq!(context.memory_links.len(), 1);
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sensation {
    pub id: SensationId,
    #[serde(default)]
    pub parent_id: Option<SensationId>,
    #[serde(default)]
    pub modality: Modality,
    #[serde(default)]
    pub payload_kind: SensationPayloadKind,
    pub kind: String,
    pub source: String,
    pub occurred_at_ms: TimeMs,
    pub observed_at_ms: TimeMs,
    pub summary: Option<String>,
    pub provenance: Provenance,
    pub payload: Value,
    #[serde(default)]
    pub metadata: SensationMetadata,
    #[serde(default)]
    pub vector: Option<VectorEmbedding>,
    #[serde(default)]
    pub impression: Option<Impression>,
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
            parent_id: None,
            modality: Modality::Other,
            payload_kind: SensationPayloadKind::Structured,
            kind: kind.into(),
            source: source.into(),
            occurred_at_ms,
            observed_at_ms,
            summary: None,
            provenance: Provenance::direct(),
            payload,
            metadata: SensationMetadata::default(),
            vector: None,
            impression: None,
        }
    }

    pub fn primary(
        modality: Modality,
        source: SensationSource,
        occurred_at_ms: TimeMs,
        observed_at_ms: TimeMs,
        payload: SensationPayload,
    ) -> Self {
        let kind = format!("{}.{}", modality.as_str(), payload.kind().as_str());
        let mut sensation = Self::new(
            kind,
            source.name.clone(),
            occurred_at_ms,
            observed_at_ms,
            payload.value,
        );
        sensation.modality = modality;
        sensation.payload_kind = payload.kind;
        sensation.metadata.source = source;
        sensation
    }

    pub fn descendant(
        parent: &Sensation,
        kind: impl Into<String>,
        payload_kind: SensationPayloadKind,
        payload: Value,
        metadata: SensationMetadata,
        stage: impl Into<String>,
    ) -> Self {
        let mut sensation = Self::new(
            kind,
            parent.source.clone(),
            parent.occurred_at_ms,
            parent.observed_at_ms,
            payload,
        );
        sensation.parent_id = Some(parent.id);
        sensation.modality = parent.modality.clone();
        sensation.payload_kind = payload_kind;
        sensation.metadata = metadata;
        sensation.provenance = Provenance::derived_from_sensations([parent.id]).with_stage(stage);
        sensation
    }

    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    pub fn with_provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = provenance;
        self
    }

    pub fn with_vector(mut self, vector: VectorEmbedding) -> Self {
        self.vector = Some(vector);
        self
    }

    pub fn with_impression(mut self, impression: Impression) -> Self {
        self.impression = Some(impression);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Impression {
    pub id: ImpressionId,
    pub kind: String,
    pub text: String,
    pub about: Vec<SensationId>,
    #[serde(default)]
    pub sensation_id: Option<SensationId>,
    #[serde(default)]
    pub experience_id: Option<ExperienceId>,
    pub occurred_at_ms: TimeMs,
    pub observed_at_ms: TimeMs,
    pub confidence: f32,
    #[serde(default)]
    pub generator: ImpressionGenerator,
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
            sensation_id: about.first().copied(),
            experience_id: None,
            about,
            occurred_at_ms,
            observed_at_ms,
            confidence: 0.5,
            generator: ImpressionGenerator::Template,
            payload: Value::Null,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn with_payload(mut self, payload: Value) -> Self {
        self.payload = payload;
        self
    }

    pub fn for_experience(mut self, experience_id: ExperienceId) -> Self {
        self.experience_id = Some(experience_id);
        self.sensation_id = None;
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
    #[serde(default)]
    pub window_start_ms: TimeMs,
    #[serde(default)]
    pub window_end_ms: TimeMs,
    #[serde(default)]
    pub fused_vector: Option<VectorEmbedding>,
    #[serde(default)]
    pub summary_impression: Option<Impression>,
    #[serde(default)]
    pub predictions: Vec<Prediction>,
    #[serde(default)]
    pub memory_links: Vec<MemoryLink>,
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
            window_start_ms: occurred_at_ms,
            window_end_ms: observed_at_ms,
            fused_vector: None,
            summary_impression: None,
            predictions: Vec::new(),
            memory_links: Vec::new(),
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

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Vision,
    Audio,
    Depth,
    Lidar,
    Touch,
    Odometry,
    Memory,
    Language,
    #[default]
    Other,
}

impl Modality {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Vision => "vision",
            Self::Audio => "audio",
            Self::Depth => "depth",
            Self::Lidar => "lidar",
            Self::Touch => "touch",
            Self::Odometry => "odometry",
            Self::Memory => "memory",
            Self::Language => "language",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensationPayloadKind {
    ImageBytes,
    AudioPcm,
    VoiceSegment,
    DepthFrame,
    PointCloud,
    LidarScan,
    ContactEvent,
    OdometryEvent,
    Crop,
    SpeechSegment,
    TranscriptSpan,
    PhonemeSpan,
    MemoryRecall,
    #[default]
    Structured,
}

impl SensationPayloadKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ImageBytes => "image_bytes",
            Self::AudioPcm => "audio_pcm",
            Self::VoiceSegment => "voice_segment",
            Self::DepthFrame => "depth_frame",
            Self::PointCloud => "point_cloud",
            Self::LidarScan => "lidar_scan",
            Self::ContactEvent => "contact_event",
            Self::OdometryEvent => "odometry_event",
            Self::Crop => "crop",
            Self::SpeechSegment => "speech_segment",
            Self::TranscriptSpan => "transcript_span",
            Self::PhonemeSpan => "phoneme_span",
            Self::MemoryRecall => "memory_recall",
            Self::Structured => "structured",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SensationSource {
    pub name: String,
    pub device_id: Option<String>,
    pub frame_id: Option<String>,
}

impl SensationSource {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            device_id: None,
            frame_id: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensationPayload {
    pub kind: SensationPayloadKind,
    pub value: Value,
}

impl SensationPayload {
    pub fn image_metadata(
        width: u32,
        height: u32,
        format: impl Into<String>,
        byte_len: usize,
    ) -> Self {
        Self {
            kind: SensationPayloadKind::ImageBytes,
            value: json!({
                "width": width,
                "height": height,
                "format": format.into(),
                "byte_len": byte_len,
            }),
        }
    }

    pub fn structured(value: Value) -> Self {
        Self {
            kind: SensationPayloadKind::Structured,
            value,
        }
    }

    pub fn kind(&self) -> SensationPayloadKind {
        self.kind.clone()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SensationMetadata {
    pub source: SensationSource,
    pub labels: Vec<String>,
    pub bbox: Option<BoundingBox>,
    pub duration_ms: Option<TimeMs>,
    pub confidence: Option<f32>,
    pub properties: BTreeMap<String, Value>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VectorEmbedding {
    pub vector: Vec<f32>,
    pub dim: usize,
    pub model_id: String,
    pub modality: Modality,
    pub payload_kind: SensationPayloadKind,
    pub source_sensation_id: SensationId,
    pub generated_at_ms: TimeMs,
}

impl VectorEmbedding {
    pub fn new(
        vector: Vec<f32>,
        model_id: impl Into<String>,
        modality: Modality,
        payload_kind: SensationPayloadKind,
        source_sensation_id: SensationId,
        generated_at_ms: TimeMs,
    ) -> Self {
        let dim = vector.len();
        Self {
            vector,
            dim,
            model_id: model_id.into(),
            modality,
            payload_kind,
            source_sensation_id,
            generated_at_ms,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImpressionGenerator {
    #[default]
    Template,
    Llm,
    Human,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    pub offset_ms: TimeMs,
    pub text: String,
    pub confidence: f32,
    pub vector: Option<VectorEmbedding>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MemoryLink {
    pub target_id: String,
    pub relation: String,
    pub score: f32,
    pub payload: Value,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedContext {
    pub experience_id: Option<ExperienceId>,
    pub summary: String,
    pub sensations: Vec<EmbodiedSensationRef>,
    pub impressions: Vec<EmbodiedImpressionRef>,
    pub lineage: Vec<EmbodiedLineageEdge>,
    pub fused_vector: Option<EmbodiedVectorRef>,
    pub sensation_vectors: Vec<EmbodiedVectorRef>,
    pub predictions: Vec<EmbodiedPredictionRef>,
    pub memory_links: Vec<EmbodiedMemoryLinkRef>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedSensationRef {
    pub id: SensationId,
    pub parent_id: Option<SensationId>,
    pub modality: Modality,
    pub payload_kind: SensationPayloadKind,
    pub kind: String,
    pub source: String,
    pub summary: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedImpressionRef {
    pub id: ImpressionId,
    pub sensation_id: Option<SensationId>,
    pub experience_id: Option<ExperienceId>,
    pub kind: String,
    pub text: String,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbodiedLineageEdge {
    pub parent_id: SensationId,
    pub child_id: SensationId,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedVectorRef {
    pub model_id: String,
    pub dim: usize,
    pub modality: Modality,
    pub payload_kind: SensationPayloadKind,
    pub source_sensation_id: SensationId,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedPredictionRef {
    pub offset_ms: TimeMs,
    pub text: String,
    pub confidence: f32,
    pub vector: Option<EmbodiedVectorRef>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedMemoryLinkRef {
    pub target_id: String,
    pub relation: String,
    pub score: f32,
    pub text: Option<String>,
}

impl EmbodiedContext {
    pub fn derived_sensation_count(&self) -> usize {
        self.sensations
            .iter()
            .filter(|sensation| sensation.parent_id.is_some())
            .count()
    }

    pub fn from_current_experience(
        experience: Option<&Experience>,
        sensations: &[Sensation],
        impressions: &[Impression],
        futures: &[FuturePrediction],
        recollections: &[RecalledExperience],
    ) -> Self {
        let sensation_scope = experience
            .map(|experience| {
                experience
                    .sensation_ids
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_else(|| sensations.iter().map(|sensation| sensation.id).collect());
        let impression_scope = experience
            .map(|experience| {
                experience
                    .impression_ids
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();

        let sensation_refs = sensations
            .iter()
            .filter(|sensation| sensation_scope.contains(&sensation.id))
            .map(|sensation| EmbodiedSensationRef {
                id: sensation.id,
                parent_id: sensation.parent_id,
                modality: sensation.modality.clone(),
                payload_kind: sensation.payload_kind.clone(),
                kind: sensation.kind.clone(),
                source: sensation.source.clone(),
                summary: sensation.summary.clone(),
            })
            .collect::<Vec<_>>();
        let scoped_sensation_ids = sensation_refs
            .iter()
            .map(|sensation| sensation.id)
            .collect::<BTreeSet<_>>();
        let impression_refs = impressions
            .iter()
            .filter(|impression| {
                impression_scope.contains(&impression.id)
                    || impression
                        .sensation_id
                        .map(|id| scoped_sensation_ids.contains(&id))
                        .unwrap_or(false)
                    || impression
                        .about
                        .iter()
                        .any(|id| scoped_sensation_ids.contains(id))
            })
            .map(|impression| EmbodiedImpressionRef {
                id: impression.id,
                sensation_id: impression.sensation_id,
                experience_id: impression.experience_id,
                kind: impression.kind.clone(),
                text: impression.text.clone(),
                confidence: impression.confidence,
            })
            .collect::<Vec<_>>();
        let lineage = sensation_refs
            .iter()
            .filter_map(|sensation| {
                let parent_id = sensation.parent_id?;
                if scoped_sensation_ids.contains(&parent_id) {
                    Some(EmbodiedLineageEdge {
                        parent_id,
                        child_id: sensation.id,
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let sensation_vectors = sensations
            .iter()
            .filter(|sensation| scoped_sensation_ids.contains(&sensation.id))
            .filter_map(|sensation| sensation.vector.as_ref().map(vector_ref))
            .collect::<Vec<_>>();
        let fused_vector = experience
            .and_then(|experience| experience.fused_vector.as_ref())
            .map(vector_ref);
        let mut predictions = experience
            .map(|experience| {
                experience
                    .predictions
                    .iter()
                    .map(|prediction| EmbodiedPredictionRef {
                        offset_ms: prediction.offset_ms,
                        text: prediction.text.clone(),
                        confidence: prediction.confidence,
                        vector: prediction.vector.as_ref().map(vector_ref),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        predictions.extend(futures.iter().filter_map(|future| {
            future
                .summary
                .as_ref()
                .map(|summary| EmbodiedPredictionRef {
                    offset_ms: future.offset_ms,
                    text: summary.clone(),
                    confidence: future.confidence,
                    vector: None,
                })
        }));
        let mut memory_links = experience
            .map(|experience| {
                experience
                    .memory_links
                    .iter()
                    .map(|link| EmbodiedMemoryLinkRef {
                        target_id: link.target_id.clone(),
                        relation: link.relation.clone(),
                        score: link.score,
                        text: link
                            .payload
                            .get("text")
                            .and_then(|value| value.as_str())
                            .map(str::to_string),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        memory_links.extend(
            recollections
                .iter()
                .map(|recollection| EmbodiedMemoryLinkRef {
                    target_id: recollection.experience.id.to_string(),
                    relation: "recalled_experience".to_string(),
                    score: recollection.score,
                    text: Some(recollection.experience.text.clone()),
                }),
        );

        let summary = experience
            .map(|experience| experience.text.clone())
            .or_else(|| {
                impression_refs
                    .last()
                    .map(|impression| impression.text.clone())
            })
            .unwrap_or_default();
        Self {
            experience_id: experience.map(|experience| experience.id),
            summary,
            sensations: sensation_refs,
            impressions: impression_refs,
            lineage,
            fused_vector,
            sensation_vectors,
            predictions,
            memory_links,
        }
    }
}

fn vector_ref(vector: &VectorEmbedding) -> EmbodiedVectorRef {
    EmbodiedVectorRef {
        model_id: vector.model_id.clone(),
        dim: vector.dim,
        modality: vector.modality.clone(),
        payload_kind: vector.payload_kind.clone(),
        source_sensation_id: vector.source_sensation_id,
    }
}

#[async_trait]
pub trait SensationVectorizer: Send + Sync {
    fn modality(&self) -> Modality;
    fn payload_kind(&self) -> SensationPayloadKind;
    fn model_id(&self) -> &'static str;
    fn output_dim(&self) -> usize;
    async fn vectorize(&self, sensation: &Sensation) -> Result<VectorEmbedding>;
}

#[derive(Clone, Default)]
pub struct SensationVectorizerRegistry {
    vectorizers: BTreeMap<(Modality, SensationPayloadKind), Arc<dyn SensationVectorizer>>,
}

impl SensationVectorizerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        for (modality, payload_kind) in [
            (Modality::Vision, SensationPayloadKind::ImageBytes),
            (Modality::Vision, SensationPayloadKind::Crop),
            (Modality::Audio, SensationPayloadKind::AudioPcm),
            (Modality::Audio, SensationPayloadKind::VoiceSegment),
            (Modality::Audio, SensationPayloadKind::SpeechSegment),
            (Modality::Audio, SensationPayloadKind::TranscriptSpan),
            (Modality::Audio, SensationPayloadKind::PhonemeSpan),
            (Modality::Depth, SensationPayloadKind::DepthFrame),
            (Modality::Touch, SensationPayloadKind::ContactEvent),
            (Modality::Odometry, SensationPayloadKind::OdometryEvent),
            (Modality::Memory, SensationPayloadKind::MemoryRecall),
            (Modality::Other, SensationPayloadKind::Structured),
        ] {
            registry.register(PlaceholderSensationVectorizer::new(modality, payload_kind));
        }
        registry
    }

    pub fn register<V>(&mut self, vectorizer: V)
    where
        V: SensationVectorizer + 'static,
    {
        self.vectorizers.insert(
            (vectorizer.modality(), vectorizer.payload_kind()),
            Arc::new(vectorizer),
        );
    }

    pub fn get(
        &self,
        modality: &Modality,
        payload_kind: &SensationPayloadKind,
    ) -> Option<Arc<dyn SensationVectorizer>> {
        self.vectorizers
            .get(&(modality.clone(), payload_kind.clone()))
            .cloned()
    }

    pub async fn vectorize(&self, sensation: &Sensation) -> Result<Option<VectorEmbedding>> {
        let Some(vectorizer) = self.get(&sensation.modality, &sensation.payload_kind) else {
            return Ok(None);
        };
        vectorizer.vectorize(sensation).await.map(Some)
    }
}

#[derive(Clone, Debug)]
pub struct PlaceholderSensationVectorizer {
    modality: Modality,
    payload_kind: SensationPayloadKind,
}

impl PlaceholderSensationVectorizer {
    pub fn new(modality: Modality, payload_kind: SensationPayloadKind) -> Self {
        Self {
            modality,
            payload_kind,
        }
    }
}

#[async_trait]
impl SensationVectorizer for PlaceholderSensationVectorizer {
    fn modality(&self) -> Modality {
        self.modality.clone()
    }

    fn payload_kind(&self) -> SensationPayloadKind {
        self.payload_kind.clone()
    }

    fn model_id(&self) -> &'static str {
        "netherwick.placeholder.v0"
    }

    fn output_dim(&self) -> usize {
        PLACEHOLDER_VECTOR_DIM
    }

    async fn vectorize(&self, sensation: &Sensation) -> Result<VectorEmbedding> {
        let mut vector = vec![0.0; self.output_dim()];
        vector[0] = stable_unit(&sensation.kind);
        vector[1] = stable_unit(&sensation.source);
        vector[2] = (sensation.occurred_at_ms % 10_000) as f32 / 10_000.0;
        vector[3] = sensation.metadata.confidence.unwrap_or(0.5).clamp(0.0, 1.0);
        vector[4] = sensation
            .payload
            .get("width")
            .and_then(Value::as_u64)
            .map(|value| (value as f32 / 1920.0).clamp(0.0, 1.0))
            .unwrap_or_default();
        vector[5] = sensation
            .payload
            .get("height")
            .and_then(Value::as_u64)
            .map(|value| (value as f32 / 1080.0).clamp(0.0, 1.0))
            .unwrap_or_default();
        vector[6] = sensation
            .metadata
            .duration_ms
            .map(|value| (value as f32 / 5_000.0).clamp(0.0, 1.0))
            .unwrap_or_default();
        if sensation.parent_id.is_some() {
            vector[7] = 1.0;
        }
        for (idx, label) in sensation.metadata.labels.iter().take(4).enumerate() {
            vector[8 + idx] = stable_unit(label);
        }
        Ok(VectorEmbedding::new(
            vector,
            self.model_id(),
            self.modality.clone(),
            self.payload_kind.clone(),
            sensation.id,
            sensation.observed_at_ms,
        ))
    }
}

pub trait DescendantExtractor {
    fn extract(&self, sensation: &Sensation) -> Result<Vec<Sensation>>;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisualDetectionKind {
    Face,
    Object,
    SalientRegion,
}

impl VisualDetectionKind {
    fn label(&self) -> &'static str {
        match self {
            Self::Face => "face",
            Self::Object => "object-shaped region",
            Self::SalientRegion => "salient visual region",
        }
    }

    fn stage(&self) -> &'static str {
        match self {
            Self::Face => "descendant.face_crop",
            Self::Object => "descendant.object_crop",
            Self::SalientRegion => "descendant.salient_region_crop",
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Face => "vision.face_crop",
            Self::Object => "vision.object_crop",
            Self::SalientRegion => "vision.salient_region_crop",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DetectedRegion {
    pub kind: VisualDetectionKind,
    pub bbox: BoundingBox,
    pub confidence: f32,
    pub labels: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct VisualDescendantExtractor;

pub trait VisualDetector {
    fn detect(&self, sensation: &Sensation) -> Result<Vec<DetectedRegion>>;
}

impl VisualDescendantExtractor {
    pub fn detect_regions(&self, sensation: &Sensation) -> Vec<DetectedRegion> {
        self.detect(sensation).unwrap_or_default()
    }

    fn extract_visual(&self, sensation: &Sensation) -> Vec<Sensation> {
        let frame = VisualFrame::from_sensation(sensation);
        let regions = frame
            .as_ref()
            .map(detect_salient_regions)
            .unwrap_or_default();
        let mut descendants = regions
            .iter()
            .map(|region| visual_crop_sensation(sensation, frame.as_ref(), region))
            .collect::<Vec<_>>();
        if descendants.is_empty() {
            if let Some(crop) = deterministic_center_crop(sensation, frame.as_ref()) {
                descendants.push(crop);
            }
        }
        descendants
    }
}

impl VisualDetector for VisualDescendantExtractor {
    fn detect(&self, sensation: &Sensation) -> Result<Vec<DetectedRegion>> {
        let Some(frame) = VisualFrame::from_sensation(sensation) else {
            return Ok(Vec::new());
        };
        Ok(detect_salient_regions(&frame))
    }
}

impl DescendantExtractor for VisualDescendantExtractor {
    fn extract(&self, sensation: &Sensation) -> Result<Vec<Sensation>> {
        if sensation.modality == Modality::Vision
            && sensation.payload_kind == SensationPayloadKind::ImageBytes
        {
            Ok(self.extract_visual(sensation))
        } else {
            Ok(Vec::new())
        }
    }
}

#[derive(Clone, Debug)]
struct VisualFrame {
    width: u32,
    height: u32,
    format: String,
    rgb: Vec<u8>,
}

impl VisualFrame {
    fn from_sensation(sensation: &Sensation) -> Option<Self> {
        let width = payload_u32(&sensation.payload, "width")?;
        let height = payload_u32(&sensation.payload, "height")?;
        if width == 0 || height == 0 {
            return None;
        }
        let bytes = sensation
            .payload
            .get("raw_bytes_b64")
            .and_then(Value::as_str)
            .and_then(|encoded| {
                base64::engine::general_purpose::STANDARD
                    .decode(encoded)
                    .ok()
            })?;
        let format = sensation
            .payload
            .get("format")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let pixel_count = width as usize * height as usize;
        let rgb = match normalized_visual_format(&format).as_str() {
            "rgb8" if bytes.len() >= pixel_count * 3 => bytes[..pixel_count * 3].to_vec(),
            "bgr8" if bytes.len() >= pixel_count * 3 => {
                let mut rgb = Vec::with_capacity(pixel_count * 3);
                for pixel in bytes.chunks_exact(3).take(pixel_count) {
                    rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
                }
                rgb
            }
            "gray8" | "grey8" if bytes.len() >= pixel_count => {
                let mut rgb = Vec::with_capacity(pixel_count * 3);
                for value in bytes.iter().take(pixel_count) {
                    rgb.extend_from_slice(&[*value, *value, *value]);
                }
                rgb
            }
            _ if bytes.len() >= pixel_count * 3 => bytes[..pixel_count * 3].to_vec(),
            _ => return None,
        };
        Some(Self {
            width,
            height,
            format,
            rgb,
        })
    }
}

fn normalized_visual_format(format: &str) -> String {
    format
        .trim_matches('"')
        .trim()
        .trim_start_matches("EyeFrameFormat::")
        .to_ascii_lowercase()
}

fn payload_u32(payload: &Value, key: &str) -> Option<u32> {
    payload
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn detect_salient_regions(frame: &VisualFrame) -> Vec<DetectedRegion> {
    let width = frame.width as usize;
    let height = frame.height as usize;
    let pixels = width.saturating_mul(height);
    if pixels < 16 || frame.rgb.len() < pixels * 3 {
        return Vec::new();
    }

    let mut luma = Vec::with_capacity(pixels);
    let mut mean = 0.0_f32;
    for pixel in frame.rgb.chunks_exact(3).take(pixels) {
        let value =
            (0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32)
                / 255.0;
        mean += value;
        luma.push(value);
    }
    mean /= pixels as f32;
    let threshold = (mean + 0.18).clamp(0.12, 0.82);
    let mut visited = vec![false; pixels];
    let mut regions = Vec::new();

    for start in 0..pixels {
        if visited[start] || luma[start] < threshold {
            continue;
        }
        let mut stack = vec![start];
        visited[start] = true;
        let mut min_x = width;
        let mut max_x = 0_usize;
        let mut min_y = height;
        let mut max_y = 0_usize;
        let mut count = 0_usize;
        let mut luma_sum = 0.0_f32;
        let mut skin_like = 0_usize;

        while let Some(index) = stack.pop() {
            let x = index % width;
            let y = index / width;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            count += 1;
            luma_sum += luma[index];
            let base = index * 3;
            let r = frame.rgb[base];
            let g = frame.rgb[base + 1];
            let b = frame.rgb[base + 2];
            if is_skin_like_rgb(r, g, b) {
                skin_like += 1;
            }

            for neighbor in neighbors4(index, x, y, width, height) {
                if !visited[neighbor] && luma[neighbor] >= threshold {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }

        let bbox_width = max_x.saturating_sub(min_x) + 1;
        let bbox_height = max_y.saturating_sub(min_y) + 1;
        let area_ratio = count as f32 / pixels as f32;
        if count < 8 || area_ratio < 0.01 || bbox_width < 3 || bbox_height < 3 {
            continue;
        }
        let fill_ratio = count as f32 / (bbox_width * bbox_height) as f32;
        let mean_region_luma = luma_sum / count as f32;
        let aspect = bbox_width as f32 / bbox_height as f32;
        let skin_ratio = skin_like as f32 / count as f32;
        let kind = if skin_ratio > 0.45 && (0.55..=1.45).contains(&aspect) {
            VisualDetectionKind::Face
        } else if fill_ratio > 0.25 && area_ratio > 0.025 {
            VisualDetectionKind::Object
        } else {
            VisualDetectionKind::SalientRegion
        };
        let confidence =
            (0.28 + area_ratio.sqrt() * 0.55 + (mean_region_luma - mean).max(0.0) * 0.4)
                .clamp(0.05, 0.92);
        let mut labels = vec![kind.label().to_string()];
        labels.push("visual crop".to_string());
        regions.push(DetectedRegion {
            kind,
            bbox: BoundingBox {
                x: min_x as u32,
                y: min_y as u32,
                width: bbox_width as u32,
                height: bbox_height as u32,
            },
            confidence,
            labels,
        });
    }

    regions.sort_by(|left, right| {
        right
            .confidence
            .partial_cmp(&left.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    regions.truncate(3);
    regions
}

fn neighbors4(index: usize, x: usize, y: usize, width: usize, height: usize) -> [usize; 4] {
    [
        if x > 0 { index - 1 } else { index },
        if x + 1 < width { index + 1 } else { index },
        if y > 0 { index - width } else { index },
        if y + 1 < height { index + width } else { index },
    ]
}

fn is_skin_like_rgb(r: u8, g: u8, b: u8) -> bool {
    r > 95 && g > 40 && b > 20 && r > g && g >= b && r.saturating_sub(b) > 35
}

fn visual_crop_sensation(
    parent: &Sensation,
    frame: Option<&VisualFrame>,
    region: &DetectedRegion,
) -> Sensation {
    let mut metadata = parent.metadata.clone();
    metadata.bbox = Some(region.bbox);
    metadata.confidence = Some(region.confidence);
    for label in &region.labels {
        if !metadata.labels.contains(label) {
            metadata.labels.push(label.clone());
        }
    }
    metadata.properties.insert(
        "detection_kind".to_string(),
        serde_json::to_value(&region.kind).unwrap_or(Value::Null),
    );
    if let Some(frame) = frame {
        metadata.properties.insert(
            "source_format".to_string(),
            Value::String(frame.format.clone()),
        );
    }
    let crop_bytes_b64 = frame
        .and_then(|frame| crop_rgb_bytes(frame, region.bbox))
        .map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes));
    let crop_content_id = crop_bytes_b64
        .as_deref()
        .map(|encoded| format!("crop:{:04}", (stable_unit(encoded) * 10_000.0) as u32));
    let mut payload = json!({
        "parent_image": parent.id,
        "bbox": region.bbox,
        "width": region.bbox.width,
        "height": region.bbox.height,
        "method": "visual_region_proposal_v0",
        "detection_kind": &region.kind,
        "confidence": region.confidence,
        "labels": &region.labels,
    });
    if let Some(content_id) = crop_content_id {
        payload["crop_content_id"] = Value::String(content_id);
    }
    if let Some(encoded) = crop_bytes_b64 {
        payload["raw_bytes_b64"] = Value::String(encoded);
        payload["format"] = Value::String("rgb8".to_string());
    }
    Sensation::descendant(
        parent,
        region.kind.kind(),
        SensationPayloadKind::Crop,
        payload,
        metadata,
        region.kind.stage(),
    )
    .with_summary(match &region.kind {
        VisualDetectionKind::Face => "I see a face close to me.",
        VisualDetectionKind::Object => "I notice an object-shaped region ahead.",
        VisualDetectionKind::SalientRegion => "I notice a salient patch of the scene.",
    })
}

fn crop_rgb_bytes(frame: &VisualFrame, bbox: BoundingBox) -> Option<Vec<u8>> {
    let frame_width = frame.width as usize;
    let frame_height = frame.height as usize;
    let x0 = bbox.x as usize;
    let y0 = bbox.y as usize;
    let width = bbox.width as usize;
    let height = bbox.height as usize;
    if width == 0 || height == 0 || x0 >= frame_width || y0 >= frame_height {
        return None;
    }
    let x1 = (x0 + width).min(frame_width);
    let y1 = (y0 + height).min(frame_height);
    let mut crop = Vec::with_capacity((x1 - x0) * (y1 - y0) * 3);
    for y in y0..y1 {
        let start = (y * frame_width + x0) * 3;
        let end = (y * frame_width + x1) * 3;
        crop.extend_from_slice(&frame.rgb[start..end]);
    }
    Some(crop)
}

fn deterministic_center_crop(parent: &Sensation, frame: Option<&VisualFrame>) -> Option<Sensation> {
    let width = payload_u32(&parent.payload, "width").unwrap_or(0);
    let height = payload_u32(&parent.payload, "height").unwrap_or(0);
    if width < 16 || height < 16 {
        return None;
    }
    let bbox = BoundingBox {
        x: width / 4,
        y: height / 4,
        width: (width / 2).max(1),
        height: (height / 2).max(1),
    };
    let mut metadata = parent.metadata.clone();
    metadata.bbox = Some(bbox);
    metadata.labels.push("central visual crop".to_string());
    metadata.confidence = Some(0.35);
    let crop_bytes_b64 = frame
        .and_then(|frame| crop_rgb_bytes(frame, bbox))
        .map(|bytes| base64::engine::general_purpose::STANDARD.encode(bytes));
    let mut payload = json!({
        "parent_image": parent.id,
        "bbox": bbox,
        "width": bbox.width,
        "height": bbox.height,
        "method": "deterministic_center_crop",
    });
    if let Some(encoded) = crop_bytes_b64 {
        payload["raw_bytes_b64"] = Value::String(encoded);
        payload["format"] = Value::String("rgb8".to_string());
    }
    Some(
        Sensation::descendant(
            parent,
            "vision.crop",
            SensationPayloadKind::Crop,
            payload,
            metadata,
            "descendant.center_crop",
        )
        .with_summary("I narrow my sight toward the middle of the frame."),
    )
}

#[derive(Clone, Debug, Default)]
pub struct AudioDescendantExtractor;

impl AudioDescendantExtractor {
    fn extract_audio(&self, sensation: &Sensation) -> Vec<Sensation> {
        let Some(window) = AudioWindow::from_sensation(sensation) else {
            return Vec::new();
        };
        let mut descendants = Vec::new();
        let transcript = window
            .transcript
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty());

        descendants.push(audio_voice_segment(
            sensation,
            &window,
            0,
            window.duration_ms,
            "asr_or_vad_window",
        ));
        if let Some(text) = transcript {
            descendants.push(audio_speech_segment(sensation, &window, text));
            descendants.push(audio_transcript_span(sensation, &window, text));
        } else if !window.has_asr_timing {
            descendants = fallback_audio_voice_segments(sensation, &window);
        }
        descendants
    }
}

impl DescendantExtractor for AudioDescendantExtractor {
    fn extract(&self, sensation: &Sensation) -> Result<Vec<Sensation>> {
        if sensation.modality == Modality::Audio
            && sensation.payload_kind == SensationPayloadKind::AudioPcm
        {
            Ok(self.extract_audio(sensation))
        } else {
            Ok(Vec::new())
        }
    }
}

#[derive(Clone, Debug)]
struct AudioWindow {
    start_ms: TimeMs,
    end_ms: TimeMs,
    duration_ms: TimeMs,
    confidence: f32,
    transcript: Option<String>,
    is_final: bool,
    word_count: Option<u64>,
    speaker_confidence: Option<f32>,
    sample_rate_hz: Option<u64>,
    feature_sets: u64,
    has_asr_timing: bool,
}

impl AudioWindow {
    fn from_sensation(sensation: &Sensation) -> Option<Self> {
        let asr = sensation.payload.get("asr").unwrap_or(&Value::Null);
        let feature_sets = sensation
            .payload
            .get("feature_sets")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let transcript = sensation
            .payload
            .get("transcript")
            .and_then(Value::as_str)
            .or_else(|| asr.get("transcript").and_then(Value::as_str))
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(ToOwned::to_owned);
        let asr_start = asr.get("start_ms").and_then(Value::as_u64);
        let asr_end = asr.get("end_ms").and_then(Value::as_u64);
        let duration = sensation
            .metadata
            .duration_ms
            .or_else(|| asr.get("duration_ms").and_then(Value::as_u64))
            .or_else(|| Some(asr_end?.saturating_sub(asr_start?)))
            .or_else(|| (feature_sets > 0).then_some(feature_sets.saturating_mul(20)))
            .or_else(|| {
                (sensation.observed_at_ms > sensation.occurred_at_ms)
                    .then_some(sensation.observed_at_ms - sensation.occurred_at_ms)
            })
            .unwrap_or_default();
        if duration == 0 && transcript.is_none() {
            return None;
        }
        let end_ms = asr_end.unwrap_or(sensation.observed_at_ms.max(sensation.occurred_at_ms));
        let start_ms = asr_start.unwrap_or_else(|| end_ms.saturating_sub(duration));
        let duration_ms = duration.max(end_ms.saturating_sub(start_ms)).max(1);
        Some(Self {
            start_ms,
            end_ms: start_ms.saturating_add(duration_ms),
            duration_ms,
            confidence: sensation
                .metadata
                .confidence
                .or_else(|| {
                    asr.get("confidence")
                        .and_then(Value::as_f64)
                        .map(|value| value as f32)
                })
                .unwrap_or(0.45)
                .clamp(0.0, 1.0),
            transcript,
            is_final: asr
                .get("is_final")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            word_count: asr.get("word_count").and_then(Value::as_u64),
            speaker_confidence: asr
                .get("speaker_confidence")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            sample_rate_hz: asr.get("sample_rate_hz").and_then(Value::as_u64),
            feature_sets,
            has_asr_timing: asr_start.is_some() || asr_end.is_some(),
        })
    }
}

fn fallback_audio_voice_segments(parent: &Sensation, window: &AudioWindow) -> Vec<Sensation> {
    let segment_count = if window.duration_ms >= 2_400 {
        3
    } else if window.duration_ms >= 1_200 {
        2
    } else {
        1
    };
    let segment_duration = (window.duration_ms / segment_count).max(1);
    (0..segment_count)
        .map(|index| {
            let start_offset = segment_duration.saturating_mul(index);
            let end_offset = if index + 1 == segment_count {
                window.duration_ms
            } else {
                segment_duration.saturating_mul(index + 1)
            };
            audio_voice_segment(
                parent,
                window,
                start_offset,
                end_offset,
                "deterministic_audio_features",
            )
        })
        .collect()
}

fn audio_voice_segment(
    parent: &Sensation,
    window: &AudioWindow,
    start_offset_ms: TimeMs,
    end_offset_ms: TimeMs,
    method: &str,
) -> Sensation {
    let start_ms = window.start_ms.saturating_add(start_offset_ms);
    let end_ms = window
        .start_ms
        .saturating_add(end_offset_ms.max(start_offset_ms + 1));
    let mut metadata = parent.metadata.clone();
    metadata.duration_ms = Some(end_ms.saturating_sub(start_ms));
    metadata.confidence = Some(if window.transcript.is_some() {
        window.confidence.max(0.55)
    } else {
        window.confidence.min(0.55).max(0.25)
    });
    push_label(&mut metadata, "voice-like audio");
    if window.transcript.is_some() {
        push_label(&mut metadata, "asr voice activity");
    } else {
        push_label(&mut metadata, "fallback voice activity");
    }
    metadata
        .properties
        .insert("start_ms".to_string(), json!(start_ms));
    metadata
        .properties
        .insert("end_ms".to_string(), json!(end_ms));
    metadata
        .properties
        .insert("method".to_string(), json!(method));
    if let Some(sample_rate_hz) = window.sample_rate_hz {
        metadata
            .properties
            .insert("sample_rate_hz".to_string(), json!(sample_rate_hz));
    }
    let mut sensation = Sensation::descendant(
        parent,
        "audio.voice_segment",
        SensationPayloadKind::VoiceSegment,
        json!({
            "parent_audio": parent.id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_offset_ms": start_offset_ms,
            "end_offset_ms": end_offset_ms,
            "duration_ms": end_ms.saturating_sub(start_ms),
            "confidence": metadata.confidence,
            "feature_sets": window.feature_sets,
            "method": method,
        }),
        metadata,
        "descendant.audio_voice_activity",
    )
    .with_summary("I hear a voice nearby.");
    sensation.occurred_at_ms = start_ms;
    sensation
}

fn audio_speech_segment(parent: &Sensation, window: &AudioWindow, transcript: &str) -> Sensation {
    let mut metadata = parent.metadata.clone();
    metadata.duration_ms = Some(window.duration_ms);
    metadata.confidence = Some(window.confidence.max(0.35));
    push_label(&mut metadata, "speech");
    push_label(&mut metadata, "asr speech span");
    metadata
        .properties
        .insert("start_ms".to_string(), json!(window.start_ms));
    metadata
        .properties
        .insert("end_ms".to_string(), json!(window.end_ms));
    metadata
        .properties
        .insert("is_final".to_string(), json!(window.is_final));
    let mut sensation = Sensation::descendant(
        parent,
        "audio.speech_segment",
        SensationPayloadKind::SpeechSegment,
        json!({
            "parent_audio": parent.id,
            "start_ms": window.start_ms,
            "end_ms": window.end_ms,
            "duration_ms": window.duration_ms,
            "text": transcript,
            "is_final": window.is_final,
            "confidence": window.confidence,
            "word_count": window.word_count,
            "speaker_confidence": window.speaker_confidence,
            "method": "asr_timed_speech_span",
        }),
        metadata,
        "descendant.audio_speech_span",
    )
    .with_summary(format!("I hear someone say \"{transcript}\"."));
    sensation.occurred_at_ms = window.start_ms;
    sensation
}

fn audio_transcript_span(parent: &Sensation, window: &AudioWindow, transcript: &str) -> Sensation {
    let mut metadata = parent.metadata.clone();
    metadata.duration_ms = Some(window.duration_ms);
    metadata.confidence = Some(window.confidence.max(0.35));
    push_label(&mut metadata, "transcript");
    push_label(&mut metadata, "asr transcript span");
    metadata
        .properties
        .insert("start_ms".to_string(), json!(window.start_ms));
    metadata
        .properties
        .insert("end_ms".to_string(), json!(window.end_ms));
    let mut sensation = Sensation::descendant(
        parent,
        "audio.transcript_span",
        SensationPayloadKind::TranscriptSpan,
        json!({
            "parent_audio": parent.id,
            "start_ms": window.start_ms,
            "end_ms": window.end_ms,
            "duration_ms": window.duration_ms,
            "text": transcript,
            "is_final": window.is_final,
            "confidence": window.confidence,
            "word_count": window.word_count,
            "method": "asr_transcript_span",
        }),
        metadata,
        "descendant.audio_transcript_span",
    )
    .with_summary(format!("I hear someone say \"{transcript}\"."));
    sensation.occurred_at_ms = window.start_ms;
    sensation
}

fn push_label(metadata: &mut SensationMetadata, label: &str) {
    if !metadata.labels.iter().any(|existing| existing == label) {
        metadata.labels.push(label.to_string());
    }
}

#[derive(Clone, Debug, Default)]
pub struct DeterministicDescendantExtractor;

impl DescendantExtractor for DeterministicDescendantExtractor {
    fn extract(&self, sensation: &Sensation) -> Result<Vec<Sensation>> {
        match (&sensation.modality, &sensation.payload_kind) {
            (Modality::Vision, SensationPayloadKind::ImageBytes) => {
                VisualDescendantExtractor.extract(sensation)
            }
            (Modality::Audio, SensationPayloadKind::AudioPcm) => {
                AudioDescendantExtractor.extract(sensation)
            }
            _ => Ok(Vec::new()),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TemplateImpressionGenerator;

impl TemplateImpressionGenerator {
    pub fn generate_for_sensation(&self, sensation: &Sensation) -> Impression {
        let text = match (&sensation.modality, &sensation.payload_kind) {
            (Modality::Vision, SensationPayloadKind::ImageBytes) => {
                let width = sensation
                    .payload
                    .get("width")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let height = sensation
                    .payload
                    .get("height")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                if width > 0 && height > 0 {
                    format!("I see a {} by {} frame in front of me.", width, height)
                } else {
                    "I see light and shape in front of me.".to_string()
                }
            }
            (Modality::Vision, SensationPayloadKind::Crop) => {
                match sensation
                    .metadata
                    .properties
                    .get("detection_kind")
                    .and_then(Value::as_str)
                {
                    Some("face") => "I see a face close to me.".to_string(),
                    Some("object") => "I notice an object-shaped region ahead.".to_string(),
                    Some("salient_region") => "I notice a salient patch of the scene.".to_string(),
                    _ => "I focus on a smaller part of what I see.".to_string(),
                }
            }
            (Modality::Audio, SensationPayloadKind::AudioPcm) => {
                "I hear a short sound nearby.".to_string()
            }
            (Modality::Audio, SensationPayloadKind::VoiceSegment) => {
                "I hear a voice nearby.".to_string()
            }
            (Modality::Audio, SensationPayloadKind::SpeechSegment) => sensation
                .payload
                .get("text")
                .and_then(Value::as_str)
                .map(|text| format!("I hear someone say \"{}\".", text.trim()))
                .unwrap_or_else(|| "I hear a voice nearby.".to_string()),
            (Modality::Audio, SensationPayloadKind::TranscriptSpan) => sensation
                .payload
                .get("text")
                .and_then(Value::as_str)
                .map(|text| format!("I hear someone say \"{}\".", text.trim()))
                .unwrap_or_else(|| "I hear speech nearby.".to_string()),
            (Modality::Audio, SensationPayloadKind::PhonemeSpan) => {
                "I hear a small piece of speech sound.".to_string()
            }
            (Modality::Touch, SensationPayloadKind::ContactEvent) => {
                "I feel contact against my body.".to_string()
            }
            (Modality::Odometry, SensationPayloadKind::OdometryEvent) => {
                "I feel my position changing through the room.".to_string()
            }
            (Modality::Depth, _) => "I sense distance and surface in front of me.".to_string(),
            (Modality::Memory, _) => sensation
                .summary
                .clone()
                .unwrap_or_else(|| "I remember something related to now.".to_string()),
            _ => sensation
                .summary
                .clone()
                .unwrap_or_else(|| "I notice something happening now.".to_string()),
        };
        Impression::new(
            "sensation.template",
            text,
            vec![sensation.id],
            sensation.occurred_at_ms,
            sensation.observed_at_ms,
        )
        .with_confidence(
            sensation
                .metadata
                .confidence
                .unwrap_or(0.55)
                .clamp(0.0, 1.0),
        )
        .with_payload(json!({
            "modality": sensation.modality,
            "payload_kind": sensation.payload_kind,
            "source": sensation.source,
        }))
    }

    pub fn generate_for_experience(
        &self,
        experience_id: ExperienceId,
        window_start_ms: TimeMs,
        window_end_ms: TimeMs,
        impressions: &[Impression],
    ) -> Impression {
        let mut parts = impressions
            .iter()
            .map(|impression| impression.text.trim().trim_end_matches('.').to_string())
            .filter(|text| !text.is_empty())
            .take(3)
            .collect::<Vec<_>>();
        let text = if parts.is_empty() {
            "I am here in a quiet moment.".to_string()
        } else if parts.len() == 1 {
            format!("{}.", parts.remove(0))
        } else {
            format!("{}.", parts.join(", and "))
        };
        Impression::new(
            "experience.template",
            text,
            Vec::new(),
            window_start_ms,
            window_end_ms,
        )
        .for_experience(experience_id)
        .with_confidence(0.6)
    }
}

#[derive(Clone)]
pub struct EmbodiedPipeline {
    extractor: DeterministicDescendantExtractor,
    vectorizers: SensationVectorizerRegistry,
    impressions: TemplateImpressionGenerator,
}

impl Default for EmbodiedPipeline {
    fn default() -> Self {
        Self {
            extractor: DeterministicDescendantExtractor,
            vectorizers: SensationVectorizerRegistry::with_defaults(),
            impressions: TemplateImpressionGenerator,
        }
    }
}

impl EmbodiedPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn ingest_primary(&self, primary: Sensation) -> Result<EmbodiedBatch> {
        let mut sensations = vec![primary];
        let descendants = self.extractor.extract(&sensations[0])?;
        sensations.extend(descendants);
        let mut impressions = Vec::with_capacity(sensations.len());
        for sensation in &mut sensations {
            if let Some(vector) = self.vectorizers.vectorize(sensation).await? {
                sensation.vector = Some(vector);
            }
            let impression = self.impressions.generate_for_sensation(sensation);
            sensation.impression = Some(impression.clone());
            impressions.push(impression);
        }
        Ok(EmbodiedBatch {
            sensations,
            impressions,
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedBatch {
    pub sensations: Vec<Sensation>,
    pub impressions: Vec<Impression>,
}

#[derive(Clone, Debug)]
pub struct ExperienceFuser {
    window_ms: TimeMs,
    impressions: TemplateImpressionGenerator,
}

impl Default for ExperienceFuser {
    fn default() -> Self {
        Self {
            window_ms: DEFAULT_WINDOW_MS,
            impressions: TemplateImpressionGenerator,
        }
    }
}

impl ExperienceFuser {
    pub fn new(window_ms: TimeMs) -> Self {
        Self {
            window_ms: window_ms.max(1),
            ..Self::default()
        }
    }

    pub fn fuse(&self, sensations: &[Sensation], impressions: &[Impression]) -> Result<Experience> {
        if sensations.is_empty() && impressions.is_empty() {
            return Err(anyhow!("cannot fuse an empty embodied window"));
        }
        let window_start_ms = sensations
            .iter()
            .map(|sensation| sensation.occurred_at_ms)
            .chain(
                impressions
                    .iter()
                    .map(|impression| impression.occurred_at_ms),
            )
            .min()
            .unwrap_or_default();
        let window_end_ms = sensations
            .iter()
            .map(|sensation| sensation.observed_at_ms)
            .chain(
                impressions
                    .iter()
                    .map(|impression| impression.observed_at_ms),
            )
            .max()
            .unwrap_or(window_start_ms + self.window_ms);
        let sensation_ids = sensations
            .iter()
            .map(|sensation| sensation.id)
            .collect::<Vec<_>>();
        let impression_ids = impressions
            .iter()
            .map(|impression| impression.id)
            .collect::<Vec<_>>();
        let fused_vector = fuse_vectors(sensations, window_end_ms);
        let experience_id = Uuid::new_v4();
        let summary = self.impressions.generate_for_experience(
            experience_id,
            window_start_ms,
            window_end_ms,
            impressions,
        );
        let mut experience = Experience::new(
            "embodied.now",
            summary.text.clone(),
            impression_ids,
            sensation_ids,
            window_start_ms,
            window_end_ms,
        );
        experience.id = experience_id;
        experience.window_start_ms = window_start_ms;
        experience.window_end_ms = window_end_ms;
        experience.fused_vector = fused_vector;
        experience.summary_impression = Some(summary);
        experience.predictions = vec![Prediction {
            offset_ms: self.window_ms,
            text:
                "I expect the next moment to resemble this one unless I move or something changes."
                    .to_string(),
            confidence: 0.35,
            vector: experience.fused_vector.clone(),
        }];
        experience.tags = embodied_tags(sensations);
        experience.payload = json!({
            "pipeline": "embodied.v0",
            "sensation_count": sensations.len(),
            "impression_count": impressions.len(),
            "window_ms": window_end_ms.saturating_sub(window_start_ms),
        });
        Ok(experience)
    }
}

#[derive(Clone, Debug)]
pub struct RollingExperienceWindow {
    window_ms: TimeMs,
    sensations: VecDeque<Sensation>,
    impressions: VecDeque<Impression>,
    fuser: ExperienceFuser,
}

impl RollingExperienceWindow {
    pub fn new(window_ms: TimeMs) -> Self {
        Self {
            window_ms: window_ms.max(1),
            sensations: VecDeque::new(),
            impressions: VecDeque::new(),
            fuser: ExperienceFuser::new(window_ms),
        }
    }

    pub fn push(&mut self, batch: EmbodiedBatch) {
        let newest = batch
            .sensations
            .iter()
            .map(|sensation| sensation.observed_at_ms)
            .chain(
                batch
                    .impressions
                    .iter()
                    .map(|impression| impression.observed_at_ms),
            )
            .max()
            .unwrap_or_default();
        self.sensations.extend(batch.sensations);
        self.impressions.extend(batch.impressions);
        self.prune(newest);
    }

    pub fn fuse_current(&self) -> Result<Experience> {
        let sensations = self.sensations.iter().cloned().collect::<Vec<_>>();
        let impressions = self.impressions.iter().cloned().collect::<Vec<_>>();
        self.fuser.fuse(&sensations, &impressions)
    }

    fn prune(&mut self, newest_observed_at_ms: TimeMs) {
        let cutoff = newest_observed_at_ms.saturating_sub(self.window_ms);
        while self
            .sensations
            .front()
            .map(|sensation| sensation.observed_at_ms < cutoff)
            .unwrap_or(false)
        {
            self.sensations.pop_front();
        }
        while self
            .impressions
            .front()
            .map(|impression| impression.observed_at_ms < cutoff)
            .unwrap_or(false)
        {
            self.impressions.pop_front();
        }
    }
}

pub async fn demo_embodied_experience(now_ms: TimeMs) -> Result<EmbodiedDemo> {
    let mut rgb = vec![12_u8; 64 * 48 * 3];
    for y in 14..34 {
        for x in 24..42 {
            let idx = (y * 64 + x) * 3;
            rgb[idx] = 220;
            rgb[idx + 1] = 170;
            rgb[idx + 2] = 120;
        }
    }
    let primary = Sensation::primary(
        Modality::Vision,
        SensationSource::new("demo.synthetic_camera"),
        now_ms,
        now_ms,
        {
            let mut payload = SensationPayload::image_metadata(64, 48, "rgb8", rgb.len());
            payload.value["raw_bytes_b64"] =
                Value::String(base64::engine::general_purpose::STANDARD.encode(rgb));
            payload
        },
    )
    .with_summary("I receive a synthetic visual frame.");
    let pipeline = EmbodiedPipeline::new();
    let batch = pipeline.ingest_primary(primary).await?;
    let mut window = RollingExperienceWindow::new(DEFAULT_WINDOW_MS);
    window.push(batch.clone());
    let experience = window.fuse_current()?;
    Ok(EmbodiedDemo {
        sensations: batch.sensations,
        impressions: batch.impressions,
        experience,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedDemo {
    pub sensations: Vec<Sensation>,
    pub impressions: Vec<Impression>,
    pub experience: Experience,
}

pub async fn embody_now(now: &Now) -> Result<EmbodiedNow> {
    let pipeline = EmbodiedPipeline::new();
    let mut sensations = Vec::new();
    let mut impressions = Vec::new();

    for primary in primary_sensations_from_now(now) {
        let batch = pipeline.ingest_primary(primary).await?;
        sensations.extend(batch.sensations);
        impressions.extend(batch.impressions);
    }

    let experience = ExperienceFuser::new(DEFAULT_WINDOW_MS).fuse(&sensations, &impressions)?;
    Ok(EmbodiedNow {
        sensations,
        impressions,
        experience,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedNow {
    pub sensations: Vec<Sensation>,
    pub impressions: Vec<Impression>,
    pub experience: Experience,
}

pub fn primary_sensations_from_now(now: &Now) -> Vec<Sensation> {
    let mut sensations = Vec::new();

    let mut body = Sensation::primary(
        if now.body.flags.bump_left
            || now.body.flags.bump_right
            || now.body.flags.wall
            || now.body.flags.virtual_wall
            || now.body.flags.wheel_drop
        {
            Modality::Touch
        } else {
            Modality::Odometry
        },
        SensationSource::new("body"),
        now.t_ms,
        now.t_ms,
        SensationPayload {
            kind: if now.body.flags.bump_left
                || now.body.flags.bump_right
                || now.body.flags.wall
                || now.body.flags.virtual_wall
                || now.body.flags.wheel_drop
            {
                SensationPayloadKind::ContactEvent
            } else {
                SensationPayloadKind::OdometryEvent
            },
            value: json!({
                "battery_level": now.body.battery_level,
                "charging": now.body.charging,
                "flags": now.body.flags,
                "odometry": now.body.odometry,
                "velocity": now.body.velocity,
                "cliff_sensors": now.body.cliff_sensors,
            }),
        },
    )
    .with_summary("I feel the state and motion of my body.");
    body.metadata.confidence = Some(0.9);
    sensations.push(body);

    if let Some(frame) = &now.eye_frame {
        let mut source = SensationSource::new("eye.frame");
        source.device_id = frame.source.clone();
        source.frame_id = Some(frame.captured_at_ms.to_string());
        let mut sensation =
            Sensation::primary(Modality::Vision, source, frame.captured_at_ms, now.t_ms, {
                let mut payload = SensationPayload::image_metadata(
                    frame.width,
                    frame.height,
                    format!("{:?}", frame.format),
                    frame.bytes.len(),
                );
                if !frame.bytes.is_empty() {
                    payload.value["raw_bytes_b64"] = Value::String(
                        base64::engine::general_purpose::STANDARD.encode(&frame.bytes),
                    );
                }
                payload
            })
            .with_summary("I receive a camera frame.");
        sensation.metadata.confidence = Some(0.65);
        sensation.metadata.properties.insert(
            "raw_bytes_present".to_string(),
            json!(!frame.bytes.is_empty()),
        );
        sensations.push(sensation);
    } else if !now.eye.frames.is_empty()
        || !now.eye.image_vectors.is_empty()
        || !now.eye.scene_vectors.is_empty()
    {
        let mut sensation = Sensation::primary(
            Modality::Vision,
            SensationSource::new("eye.features"),
            now.t_ms,
            now.t_ms,
            SensationPayload::structured(json!({
                "frame_feature_sets": now.eye.frames.len(),
                "image_vectors": now.eye.image_vectors.len(),
                "image_description_vectors": now.eye.image_description_vectors.len(),
                "scene_vectors": now.eye.scene_vectors.len(),
            })),
        )
        .with_summary("I have visual features from my eye.");
        sensation.metadata.confidence = Some(0.55);
        sensations.push(sensation);
    }

    if !now.ear.features.is_empty()
        || now
            .ear
            .transcript
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty())
        || now
            .ear
            .asr
            .transcript
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty())
    {
        let transcript = now
            .ear
            .asr
            .transcript
            .as_deref()
            .or(now.ear.transcript.as_deref())
            .map(str::trim)
            .filter(|text| !text.is_empty());
        let duration_ms = now
            .ear
            .asr
            .duration_ms
            .or_else(|| Some(now.ear.asr.end_ms?.saturating_sub(now.ear.asr.start_ms?)))
            .or_else(|| {
                (!now.ear.features.is_empty()).then_some(now.ear.features.len() as u64 * 20)
            });
        let observed_at_ms = now.ear.asr.end_ms.unwrap_or(now.t_ms);
        let occurred_at_ms = now
            .ear
            .asr
            .start_ms
            .or_else(|| duration_ms.map(|duration| observed_at_ms.saturating_sub(duration)))
            .unwrap_or(now.t_ms);
        let mut sensation = Sensation::primary(
            Modality::Audio,
            SensationSource::new("ear"),
            occurred_at_ms,
            observed_at_ms,
            SensationPayload {
                kind: SensationPayloadKind::AudioPcm,
                value: json!({
                    "feature_sets": now.ear.features.len(),
                    "transcript": transcript,
                    "asr": now.ear.asr,
                }),
            },
        )
        .with_summary("I hear sound through my ear.");
        sensation.metadata.duration_ms = duration_ms;
        sensation.metadata.confidence = Some(now.ear.asr.confidence.max(0.35).clamp(0.0, 1.0));
        sensation.metadata.labels.push("audio window".to_string());
        if transcript.is_some() {
            sensation.metadata.labels.push("asr available".to_string());
        }
        sensations.push(sensation);
    }

    if !now.range.beams.is_empty() || now.range.nearest_m.is_some() {
        let mut sensation = Sensation::primary(
            Modality::Lidar,
            SensationSource::new("range"),
            now.t_ms,
            now.t_ms,
            SensationPayload {
                kind: SensationPayloadKind::LidarScan,
                value: json!({
                    "beam_count": now.range.beams.len(),
                    "nearest_m": now.range.nearest_m,
                }),
            },
        )
        .with_summary("I sense nearby distance around me.");
        sensation.metadata.confidence = Some(0.7);
        sensations.push(sensation);
    }

    if !now.kinect.depth_m.is_empty() {
        let mut sensation = Sensation::primary(
            Modality::Depth,
            SensationSource::new("kinect.depth"),
            now.t_ms,
            now.t_ms,
            SensationPayload {
                kind: SensationPayloadKind::DepthFrame,
                value: json!({
                    "sample_count": now.kinect.depth_m.len(),
                    "width": now.kinect.depth_width,
                    "height": now.kinect.depth_height,
                    "min_depth_m": now.kinect.min_depth_m,
                    "max_depth_m": now.kinect.max_depth_m,
                    "coordinate_system": now.kinect.depth_coordinate_system,
                    "skeleton_count": now.kinect.skeletons.len(),
                }),
            },
        )
        .with_summary("I sense depth and surfaces ahead of me.");
        sensation.metadata.confidence = Some(0.65);
        sensations.push(sensation);
    }

    if now.memory.similar_situation_count > 0
        || now.memory.remembered_warning.is_some()
        || now.memory.graph_context_summary.is_some()
    {
        let mut sensation = Sensation::primary(
            Modality::Memory,
            SensationSource::new("memory"),
            now.t_ms,
            now.t_ms,
            SensationPayload {
                kind: SensationPayloadKind::MemoryRecall,
                value: json!({
                    "similar_situation_count": now.memory.similar_situation_count,
                    "remembered_warning": now.memory.remembered_warning,
                    "graph_context_summary": now.memory.graph_context_summary,
                    "remembered_entities": now.memory.remembered_entities,
                }),
            },
        )
        .with_summary("I remember related context for this moment.");
        sensation.metadata.confidence = Some(0.6);
        sensations.push(sensation);
    }

    sensations
}

fn fuse_vectors(sensations: &[Sensation], generated_at_ms: TimeMs) -> Option<VectorEmbedding> {
    let vectors = sensations
        .iter()
        .filter_map(|sensation| sensation.vector.as_ref())
        .collect::<Vec<_>>();
    let first = vectors.first()?;
    let dim = first.dim;
    let source_sensation_id = first.source_sensation_id;
    let mut pooled = vec![0.0; dim];
    let mut count = 0.0_f32;
    for embedding in vectors {
        if embedding.dim != dim {
            continue;
        }
        for (slot, value) in pooled.iter_mut().zip(embedding.vector.iter()) {
            *slot += *value;
        }
        count += 1.0;
    }
    if count == 0.0 {
        return None;
    }
    for value in &mut pooled {
        *value /= count;
    }
    Some(VectorEmbedding::new(
        pooled,
        "netherwick.fusion.mean_pool.v0",
        Modality::Other,
        SensationPayloadKind::Structured,
        source_sensation_id,
        generated_at_ms,
    ))
}

fn embodied_tags(sensations: &[Sensation]) -> Vec<String> {
    let mut tags = sensations
        .iter()
        .map(|sensation| sensation.modality.as_str().to_string())
        .collect::<Vec<_>>();
    tags.sort();
    tags.dedup();
    tags
}

fn stable_unit(text: &str) -> f32 {
    let mut hash = 0_u32;
    for byte in text.as_bytes() {
        hash = hash.wrapping_mul(16777619) ^ u32::from(*byte);
    }
    (hash % 10_000) as f32 / 10_000.0
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecalledExperience {
    pub score: f32,
    pub experience: Experience,
    pub sensation: Sensation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceBehaviorInput {
    pub now: Now,
    pub sense_vectors: Vec<Vec<f32>>,
}

impl ExperienceBehaviorInput {
    pub fn from_now(now: &Now) -> Self {
        let encode_input = experience_encode_input_from_now(now);
        Self {
            now: now.clone(),
            sense_vectors: encode_input.sense_vectors,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceBehaviorOutput {
    pub latent: ExperienceLatent,
    pub reconstruction: Option<ExperienceDecodeOutput>,
    pub reconstruction_loss: Option<f32>,
    pub confidence: f32,
}
