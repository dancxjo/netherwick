use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use base64::Engine;
use pete_actions::{action_to_motor_command, ActionPrimitive, ExploreStyle, TurnDir};
use pete_body::BodySense;
use pete_core::{ExperienceId, ImpressionId, Provenance, Reward, SensationId, TimeMs};
use pete_now::{DriveSense, MemorySense, Now, SenseVectorizer, SurpriseSense};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

const DEFAULT_WINDOW_MS: TimeMs = 750;
const PLACEHOLDER_VECTOR_DIM: usize = 16;
const EMBODIED_FEATURE_VECTOR_DIM: usize = 32;
const TEXT_HASH_VECTOR_DIM: usize = 64;
const TEXT_HASH_MODEL_ID: &str = "pete.text.hashing.v1";
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

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExperiencePrediction {
    pub t_ms: TimeMs,
    pub offset_ms: TimeMs,
    pub source_latent: ExperienceLatent,
    pub predicted_latent: ExperienceLatent,
    pub action_features: Vec<f32>,
    pub predictor_id: String,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExperienceSurprise {
    pub t_ms: TimeMs,
    pub reconstruction_loss: f32,
    pub prediction_loss: f32,
    pub combined_surprise: f32,
    pub confidence: f32,
    pub reconstruction_weight: f32,
    pub prediction_weight: f32,
}

pub trait ExperienceEncoder {
    fn encode(&mut self, now: &Now) -> Result<ExperienceLatent>;
}

pub trait LatentEncoder {
    fn encoder_kind(&self) -> &'static str;
    fn encode_input(
        &mut self,
        input: &ExperienceEncodeInput,
        t_ms: TimeMs,
    ) -> Result<ExperienceLatent>;
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
    let instant = ExperienceInstant::from_now_features(now, None);
    ExperienceEncodeInput {
        sense_vectors: instant
            .teacher_vectors
            .iter()
            .map(|vector| vector.vector.clone())
            .collect(),
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
                BaselineSenseVectorizer::CreateIr,
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

impl LatentEncoder for FeatureExperienceEncoder {
    fn encoder_kind(&self) -> &'static str {
        "online-evolved-filters"
    }

    fn encode_input(
        &mut self,
        input: &ExperienceEncodeInput,
        t_ms: TimeMs,
    ) -> Result<ExperienceLatent> {
        let z = input
            .flat_features()
            .into_iter()
            .map(sanitize_feature)
            .collect::<Vec<_>>();
        Ok(ExperienceLatent {
            t_ms,
            z,
            reconstruction_error: 0.0,
            prediction_error: 0.0,
            confidence: 0.55,
        })
    }
}

#[derive(Clone, Debug)]
pub struct RandomProjectionExperienceEncoder {
    z_dim: usize,
    seed: u64,
}

impl RandomProjectionExperienceEncoder {
    pub fn new(z_dim: usize, seed: u64) -> Self {
        Self {
            z_dim: z_dim.max(1),
            seed,
        }
    }

    fn weight(&self, output_index: usize, input_index: usize) -> f32 {
        let mut x = self.seed
            ^ ((output_index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            ^ ((input_index as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        match x % 6 {
            0 | 1 => -1.0,
            2 | 3 => 0.0,
            _ => 1.0,
        }
    }
}

impl LatentEncoder for RandomProjectionExperienceEncoder {
    fn encoder_kind(&self) -> &'static str {
        "random-projection"
    }

    fn encode_input(
        &mut self,
        input: &ExperienceEncodeInput,
        t_ms: TimeMs,
    ) -> Result<ExperienceLatent> {
        let features = input.flat_features();
        let scale = (features.len().max(1) as f32).sqrt();
        let mut z = vec![0.0; self.z_dim];
        for (out_index, out) in z.iter_mut().enumerate() {
            let sum = features
                .iter()
                .enumerate()
                .map(|(in_index, value)| {
                    sanitize_feature(*value) * self.weight(out_index, in_index)
                })
                .sum::<f32>();
            *out = (sum / scale).tanh();
        }
        Ok(ExperienceLatent {
            t_ms,
            z,
            reconstruction_error: 1.0,
            prediction_error: 0.0,
            confidence: 0.35,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodebookQuantizer {
    pub codes: Vec<Vec<f32>>,
    pub usage: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodebookUsageReport {
    pub code_count: usize,
    pub used_codes: usize,
    pub dead_codes: usize,
    pub usage: Vec<u64>,
}

impl CodebookQuantizer {
    pub fn from_latents(latents: &[Vec<f32>], code_count: usize) -> Self {
        let code_count = code_count.max(1);
        let mut codes = Vec::new();
        for index in 0..code_count {
            let code = latents
                .get(index.saturating_mul(latents.len().max(1)) / code_count)
                .cloned()
                .unwrap_or_default();
            codes.push(code);
        }
        Self {
            codes,
            usage: vec![0; code_count],
        }
    }

    pub fn encode(&mut self, latent: &[f32]) -> usize {
        let (index, _) = self
            .codes
            .iter()
            .enumerate()
            .map(|(index, code)| (index, normalized_distance(code, latent)))
            .min_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((0, 0.0));
        if let Some(count) = self.usage.get_mut(index) {
            *count = count.saturating_add(1);
        }
        index
    }

    pub fn decode(&self, code_id: usize) -> Vec<f32> {
        self.codes.get(code_id).cloned().unwrap_or_default()
    }

    pub fn report(&self) -> CodebookUsageReport {
        let used_codes = self.usage.iter().filter(|count| **count > 0).count();
        CodebookUsageReport {
            code_count: self.codes.len(),
            used_codes,
            dead_codes: self.codes.len().saturating_sub(used_codes),
            usage: self.usage.clone(),
        }
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
    CreateIr,
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
            Self::CreateIr => "create_ir",
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
            Self::CreateIr => vec![
                bool01(now.body.infrared_character != 0),
                f32::from(now.body.infrared_character) / f32::from(u8::MAX),
            ],
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
        .committed_transcript
        .as_deref()
        .or(now.ear.asr.transcript.as_deref())
        .or(now.ear.asr.possible_transcript.as_deref())
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
        bool01(now.ear.asr.is_final || now.ear.asr.committed_transcript.is_some()),
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
#[path = "encoding_tests.rs"]
mod encoding_tests;

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
    #[serde(default)]
    pub vector: Option<VectorEmbedding>,
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
            vector: None,
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

    pub fn with_vector(mut self, vector: VectorEmbedding) -> Self {
        self.vector = Some(vector);
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
        self.to_recall_sensation_with_lineage(recall_at_ms, score, stage, None, Vec::new())
    }

    pub fn to_recall_sensation_with_lineage(
        &self,
        recall_at_ms: TimeMs,
        score: f32,
        stage: impl Into<String>,
        original_frame_id: Option<Uuid>,
        original_vector_ids: Vec<String>,
    ) -> Sensation {
        let stage = stage.into();
        let payload = json!({
            "experience": self,
            "recall_kind": "recalled_experience",
            "original_frame_id": original_frame_id,
            "original_experience_id": self.id,
            "original_sensation_ids": self.sensation_ids,
            "original_impression_ids": self.impression_ids,
            "original_vector_ids": original_vector_ids,
            "original_occurred_at_ms": self.occurred_at_ms,
            "original_observed_at_ms": self.observed_at_ms,
            "score": score,
        });
        let mut provenance = Provenance::memory_recall(self.id).with_stage(stage);
        provenance.metadata = json!({
            "original_frame_id": original_frame_id,
            "original_experience_id": self.id,
            "original_vector_ids": payload.get("original_vector_ids").cloned().unwrap_or(Value::Null),
        });
        let mut sensation = Sensation::primary(
            Modality::Memory,
            SensationSource::new("memory.recall"),
            recall_at_ms,
            recall_at_ms,
            SensationPayload {
                kind: SensationPayloadKind::MemoryRecall,
                value: payload,
            },
        )
        .with_summary(format!(
            "I remember a similar moment near here: {}",
            self.text
        ))
        .with_provenance(provenance);
        sensation.kind = "memory.recall.experience".to_string();
        sensation.metadata.confidence = Some(score.clamp(0.0, 1.0));
        sensation.metadata.labels.push("memory_recall".to_string());
        sensation
            .metadata
            .labels
            .push("recalled_experience".to_string());
        if let Some(frame_id) = original_frame_id {
            sensation.metadata.properties.insert(
                "original_frame_id".to_string(),
                Value::String(frame_id.to_string()),
            );
        }
        sensation.metadata.properties.insert(
            "original_experience_id".to_string(),
            Value::String(self.id.to_string()),
        );
        sensation.metadata.properties.insert(
            "original_vector_count".to_string(),
            json!(original_vector_ids.len()),
        );
        sensation
    }

    pub fn to_recall_impression(&self, sensation: &Sensation, score: f32) -> Impression {
        Impression::new(
            "memory.recall.impression",
            format!("I remember a similar moment near here: {}", self.text),
            vec![sensation.id],
            sensation.occurred_at_ms,
            sensation.observed_at_ms,
        )
        .with_confidence(score.clamp(0.0, 1.0))
        .with_payload(json!({
            "generator": "memory_recall",
            "original_experience_id": self.id,
            "score": score,
        }))
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
    #[serde(default = "default_vectorizer_id")]
    pub vectorizer_id: String,
    pub model_id: String,
    #[serde(default)]
    pub model_label: String,
    pub modality: Modality,
    pub payload_kind: SensationPayloadKind,
    #[serde(default = "default_vector_source_kind")]
    pub source_kind: String,
    pub source_sensation_id: SensationId,
    #[serde(default = "default_vector_purpose")]
    pub purpose: String,
    #[serde(default = "default_vector_collection")]
    pub collection: String,
    #[serde(default)]
    pub input_summary: String,
    #[serde(default)]
    pub is_fallback: bool,
    #[serde(default)]
    pub provenance: String,
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
        let model_id = model_id.into();
        Self {
            vector,
            dim,
            vectorizer_id: model_id.clone(),
            model_label: model_id.clone(),
            model_id,
            modality,
            payload_kind,
            source_kind: default_vector_source_kind(),
            source_sensation_id,
            purpose: default_vector_purpose(),
            collection: default_vector_collection(),
            input_summary: String::new(),
            is_fallback: false,
            provenance: String::new(),
            generated_at_ms,
        }
    }

    pub fn with_metadata(
        mut self,
        vectorizer_id: impl Into<String>,
        model_label: impl Into<String>,
        purpose: impl Into<String>,
        collection: impl Into<String>,
        input_summary: impl Into<String>,
        is_fallback: bool,
        provenance: impl Into<String>,
    ) -> Self {
        self.vectorizer_id = vectorizer_id.into();
        self.model_label = model_label.into();
        self.purpose = purpose.into();
        self.collection = collection.into();
        self.input_summary = input_summary.into();
        self.is_fallback = is_fallback;
        self.provenance = provenance.into();
        self
    }

    pub fn with_source_kind(mut self, source_kind: impl Into<String>) -> Self {
        self.source_kind = source_kind.into();
        self
    }
}

fn default_vectorizer_id() -> String {
    "unknown".to_string()
}

fn default_vector_source_kind() -> String {
    "sensation".to_string()
}

fn default_vector_purpose() -> String {
    "unspecified".to_string()
}

fn default_vector_collection() -> String {
    "embodied_vectors".to_string()
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
    pub sensation_vectors: Vec<EmbodiedVectorRef>,
    #[serde(default)]
    pub impression_vectors: Vec<EmbodiedVectorRef>,
    pub predictions: Vec<EmbodiedPredictionRef>,
    pub memory_links: Vec<EmbodiedMemoryLinkRef>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceInstant {
    pub schema_version: u32,
    pub t_ms: TimeMs,
    pub window_start_ms: TimeMs,
    pub window_end_ms: TimeMs,
    pub experience_id: Option<ExperienceId>,
    pub summary: String,
    pub primary_sensations: Vec<EmbodiedSensationRef>,
    pub descendant_sensations: Vec<EmbodiedSensationRef>,
    pub impressions: Vec<EmbodiedImpressionRef>,
    pub summary_impression: Option<EmbodiedImpressionRef>,
    pub teacher_vectors: Vec<InstantTeacherVector>,
    pub body_context: InstantBodyContext,
    pub action_context: InstantActionContext,
    pub lineage: Vec<EmbodiedLineageEdge>,
    #[serde(default)]
    pub memory_links: Vec<EmbodiedMemoryLinkRef>,
    #[serde(default)]
    pub predictions: Vec<EmbodiedPredictionRef>,
    pub provenance: InstantProvenance,
    pub missing_modalities: Vec<MissingModality>,
}

pub type Instant = ExperienceInstant;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InstantCoverage {
    pub present_modalities: Vec<String>,
    pub missing_modalities: Vec<String>,
    pub sensation_count: usize,
    pub descendant_count: usize,
    pub vector_count: usize,
    pub impression_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InstantTeacherVector {
    pub vector: Vec<f32>,
    pub metadata: EmbodiedVectorRef,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InstantBodyContext {
    pub battery_level: f32,
    pub charging: bool,
    pub bump: bool,
    pub cliff: bool,
    pub wheel_drop: bool,
    pub wall: bool,
    pub x_m: f32,
    pub y_m: f32,
    pub heading_rad: f32,
    pub forward_m_s: f32,
    pub turn_rad_s: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InstantActionContext {
    pub action: Option<ActionPrimitive>,
    pub action_features: Vec<f32>,
    pub source: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InstantProvenance {
    pub source: String,
    pub source_frame_id: Option<String>,
    pub sensation_count: usize,
    pub impression_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissingModality {
    pub modality: Modality,
    pub reason: String,
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
    pub vector: Option<EmbodiedVectorRef>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbodiedLineageEdge {
    pub parent_id: SensationId,
    pub child_id: SensationId,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedVectorRef {
    pub vectorizer_id: String,
    pub model_id: String,
    pub model_label: String,
    pub dim: usize,
    pub modality: Modality,
    pub payload_kind: SensationPayloadKind,
    pub source_kind: String,
    pub source_sensation_id: SensationId,
    pub purpose: String,
    pub collection: String,
    pub input_summary: String,
    pub is_fallback: bool,
    pub provenance: String,
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
                vector: impression.vector.as_ref().map(vector_ref),
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
        let impression_vectors = impressions
            .iter()
            .filter(|impression| {
                impression_scope.contains(&impression.id)
                    || impression
                        .sensation_id
                        .map(|id| scoped_sensation_ids.contains(&id))
                        .unwrap_or(false)
            })
            .filter_map(|impression| impression.vector.as_ref().map(vector_ref))
            .collect::<Vec<_>>();
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
            sensation_vectors,
            impression_vectors,
            predictions,
            memory_links,
        }
    }
}

impl ExperienceInstant {
    pub async fn from_now(now: &Now, action: Option<ActionPrimitive>) -> Result<Self> {
        let embodied = embody_now(now).await?;
        Ok(Self::from_embodied_now(
            &embodied, now, action, None, "live-now",
        ))
    }

    pub fn from_embodied_now(
        embodied: &EmbodiedNow,
        now: &Now,
        action: Option<ActionPrimitive>,
        source_frame_id: Option<String>,
        source: impl Into<String>,
    ) -> Self {
        Self::from_parts(
            Some(&embodied.experience),
            &embodied.sensations,
            &embodied.impressions,
            &[],
            &[],
            now,
            action,
            source_frame_id,
            source,
        )
    }

    pub fn from_now_features(now: &Now, action: Option<ActionPrimitive>) -> Self {
        let target = experience_decode_target_from_now(now);
        let teacher_vectors = [
            (
                "now.body",
                Modality::Odometry,
                SensationPayloadKind::OdometryEvent,
                target.body_features,
            ),
            (
                "now.memory",
                Modality::Memory,
                SensationPayloadKind::MemoryRecall,
                target.memory_features,
            ),
            (
                "now.drive",
                Modality::Other,
                SensationPayloadKind::Structured,
                target.drive_features,
            ),
            (
                "now.prediction",
                Modality::Other,
                SensationPayloadKind::Structured,
                target.prediction_features,
            ),
            (
                "now.eye",
                Modality::Vision,
                SensationPayloadKind::ImageBytes,
                target.eye_features,
            ),
            (
                "now.ear",
                Modality::Audio,
                SensationPayloadKind::AudioPcm,
                target.ear_features,
            ),
        ]
        .into_iter()
        .map(
            |(source, modality, payload_kind, vector)| InstantTeacherVector {
                metadata: EmbodiedVectorRef {
                    vectorizer_id: "pete.now.features.v1".to_string(),
                    model_id: "pete.now.features.v1".to_string(),
                    model_label: "Now deterministic feature vector".to_string(),
                    dim: vector.len(),
                    modality,
                    payload_kind,
                    source_kind: "now_feature".to_string(),
                    source_sensation_id: Uuid::nil(),
                    purpose: "experience_encode_feature".to_string(),
                    collection: "experience_encode_inputs".to_string(),
                    input_summary: source.to_string(),
                    is_fallback: true,
                    provenance: "now-feature-conversion".to_string(),
                },
                vector,
            },
        )
        .collect::<Vec<_>>();
        let present_modalities = teacher_vectors
            .iter()
            .map(|vector| vector.metadata.modality.clone())
            .collect::<BTreeSet<_>>();
        Self {
            schema_version: 1,
            t_ms: now.t_ms,
            window_start_ms: now.t_ms,
            window_end_ms: now.t_ms,
            experience_id: None,
            summary: format!(
                "I am at t={}ms with battery {:.2}.",
                now.t_ms, now.body.battery_level
            ),
            primary_sensations: Vec::new(),
            descendant_sensations: Vec::new(),
            impressions: Vec::new(),
            summary_impression: None,
            teacher_vectors,
            body_context: InstantBodyContext::from_now(now),
            action_context: InstantActionContext::from_action(action),
            lineage: Vec::new(),
            memory_links: Vec::new(),
            predictions: Vec::new(),
            provenance: InstantProvenance {
                source: "now-feature-conversion".to_string(),
                source_frame_id: None,
                sensation_count: 0,
                impression_count: 0,
            },
            missing_modalities: expected_instant_modalities()
                .into_iter()
                .filter(|modality| !present_modalities.contains(modality))
                .map(|modality| MissingModality {
                    modality,
                    reason: "no feature vector for modality in this Now conversion".to_string(),
                })
                .collect(),
        }
    }

    pub fn from_parts(
        experience: Option<&Experience>,
        sensations: &[Sensation],
        impressions: &[Impression],
        futures: &[FuturePrediction],
        recollections: &[RecalledExperience],
        now: &Now,
        action: Option<ActionPrimitive>,
        source_frame_id: Option<String>,
        source: impl Into<String>,
    ) -> Self {
        let context = EmbodiedContext::from_current_experience(
            experience,
            sensations,
            impressions,
            futures,
            recollections,
        );
        let primary_sensations = context
            .sensations
            .iter()
            .filter(|sensation| sensation.parent_id.is_none())
            .cloned()
            .collect::<Vec<_>>();
        let descendant_sensations = context
            .sensations
            .iter()
            .filter(|sensation| sensation.parent_id.is_some())
            .cloned()
            .collect::<Vec<_>>();
        let scoped_sensation_ids = context
            .sensations
            .iter()
            .map(|sensation| sensation.id)
            .collect::<BTreeSet<_>>();
        let scoped_impression_ids = context
            .impressions
            .iter()
            .map(|impression| impression.id)
            .collect::<BTreeSet<_>>();
        let teacher_vectors = sensations
            .iter()
            .filter(|sensation| scoped_sensation_ids.contains(&sensation.id))
            .filter_map(|sensation| sensation.vector.as_ref().map(instant_teacher_vector))
            .chain(
                impressions
                    .iter()
                    .filter(|impression| scoped_impression_ids.contains(&impression.id))
                    .filter_map(|impression| {
                        impression.vector.as_ref().map(instant_teacher_vector)
                    }),
            )
            .collect::<Vec<_>>();
        let summary_impression = experience
            .and_then(|experience| experience.summary_impression.as_ref())
            .map(|impression| EmbodiedImpressionRef {
                id: impression.id,
                sensation_id: impression.sensation_id,
                experience_id: impression.experience_id,
                kind: impression.kind.clone(),
                text: impression.text.clone(),
                confidence: impression.confidence,
                vector: impression.vector.as_ref().map(vector_ref),
            });
        let mut present_modalities = context
            .sensations
            .iter()
            .map(|sensation| sensation.modality.clone())
            .collect::<BTreeSet<_>>();
        present_modalities.extend(
            teacher_vectors
                .iter()
                .map(|vector| vector.metadata.modality.clone()),
        );

        Self {
            schema_version: 1,
            t_ms: now.t_ms,
            window_start_ms: experience
                .map(|experience| experience.window_start_ms)
                .unwrap_or(now.t_ms),
            window_end_ms: experience
                .map(|experience| experience.window_end_ms)
                .unwrap_or(now.t_ms),
            experience_id: experience.map(|experience| experience.id),
            summary: context.summary,
            primary_sensations,
            descendant_sensations,
            impressions: context.impressions,
            summary_impression,
            teacher_vectors,
            body_context: InstantBodyContext::from_now(now),
            action_context: InstantActionContext::from_action(action),
            lineage: context.lineage,
            memory_links: context.memory_links,
            predictions: context.predictions,
            provenance: InstantProvenance {
                source: source.into(),
                source_frame_id,
                sensation_count: sensations.len(),
                impression_count: impressions.len(),
            },
            missing_modalities: expected_instant_modalities()
                .into_iter()
                .filter(|modality| !present_modalities.contains(modality))
                .map(|modality| MissingModality {
                    modality,
                    reason: "no sensation or teacher vector for modality in this Instant"
                        .to_string(),
                })
                .collect(),
        }
    }

    pub fn encode_input(&self) -> ExperienceEncodeInput {
        let mut sense_vectors = self
            .teacher_vectors
            .iter()
            .map(|vector| {
                vector
                    .vector
                    .iter()
                    .copied()
                    .map(sanitize_feature)
                    .collect()
            })
            .collect::<Vec<Vec<f32>>>();
        sense_vectors.push(self.modality_mask());
        sense_vectors.push(self.body_features());
        sense_vectors.push(self.action_context.action_features.clone());
        ExperienceEncodeInput { sense_vectors }
    }

    pub fn coverage(&self) -> InstantCoverage {
        let missing = self
            .missing_modalities
            .iter()
            .map(|missing| missing.modality.clone())
            .collect::<BTreeSet<_>>();
        let present_modalities = expected_instant_modalities()
            .into_iter()
            .filter(|modality| !missing.contains(modality))
            .map(|modality| modality.as_str().to_string())
            .collect();
        let missing_modalities = self
            .missing_modalities
            .iter()
            .map(|missing| missing.modality.as_str().to_string())
            .collect();
        InstantCoverage {
            present_modalities,
            missing_modalities,
            sensation_count: self.primary_sensations.len() + self.descendant_sensations.len(),
            descendant_count: self.descendant_sensations.len(),
            vector_count: self.teacher_vectors.len(),
            impression_count: self.impressions.len()
                + usize::from(self.summary_impression.is_some()),
        }
    }

    pub fn embodied_context(&self) -> EmbodiedContext {
        let mut sensations = self.primary_sensations.clone();
        sensations.extend(self.descendant_sensations.clone());
        let sensation_vectors = self
            .teacher_vectors
            .iter()
            .filter(|vector| vector.metadata.source_kind == "sensation")
            .map(|vector| vector.metadata.clone())
            .collect();
        let impression_vectors = self
            .teacher_vectors
            .iter()
            .filter(|vector| vector.metadata.source_kind == "impression")
            .map(|vector| vector.metadata.clone())
            .collect();
        EmbodiedContext {
            experience_id: self.experience_id,
            summary: self.summary.clone(),
            sensations,
            impressions: self.impressions.clone(),
            lineage: self.lineage.clone(),
            sensation_vectors,
            impression_vectors,
            predictions: self.predictions.clone(),
            memory_links: self.memory_links.clone(),
        }
    }

    pub fn modality_mask(&self) -> Vec<f32> {
        let missing = self
            .missing_modalities
            .iter()
            .map(|missing| missing.modality.clone())
            .collect::<BTreeSet<_>>();
        expected_instant_modalities()
            .into_iter()
            .map(|modality| {
                if missing.contains(&modality) {
                    0.0
                } else {
                    1.0
                }
            })
            .collect()
    }

    fn body_features(&self) -> Vec<f32> {
        vec![
            self.body_context.battery_level,
            bool01(self.body_context.charging),
            bool01(self.body_context.bump),
            bool01(self.body_context.cliff),
            bool01(self.body_context.wheel_drop),
            bool01(self.body_context.wall),
            self.body_context.x_m.tanh(),
            self.body_context.y_m.tanh(),
            self.body_context.heading_rad.sin(),
            self.body_context.heading_rad.cos(),
            self.body_context.forward_m_s.clamp(-1.0, 1.0),
            self.body_context.turn_rad_s.clamp(-1.0, 1.0),
        ]
    }
}

impl ExperienceEncodeInput {
    pub fn from_instant(instant: &ExperienceInstant) -> Self {
        instant.encode_input()
    }
}

impl InstantBodyContext {
    pub fn from_now(now: &Now) -> Self {
        Self {
            battery_level: now.body.battery_level.clamp(0.0, 1.0),
            charging: now.body.charging,
            bump: now.body.flags.bump_left || now.body.flags.bump_right,
            cliff: cliff_detected(now),
            wheel_drop: now.body.flags.wheel_drop,
            wall: now.body.flags.wall || now.body.flags.virtual_wall,
            x_m: now.body.odometry.x_m,
            y_m: now.body.odometry.y_m,
            heading_rad: now.body.odometry.heading_rad,
            forward_m_s: now.body.velocity.forward_m_s,
            turn_rad_s: now.body.velocity.turn_rad_s,
        }
    }
}

impl InstantActionContext {
    pub fn from_action(action: Option<ActionPrimitive>) -> Self {
        Self {
            action_features: action_features(action.as_ref()),
            action,
            source: Some("action_primitive".to_string()),
        }
    }
}

fn expected_instant_modalities() -> Vec<Modality> {
    vec![
        Modality::Vision,
        Modality::Audio,
        Modality::Depth,
        Modality::Lidar,
        Modality::Touch,
        Modality::Odometry,
        Modality::Memory,
        Modality::Language,
    ]
}

fn instant_teacher_vector(vector: &VectorEmbedding) -> InstantTeacherVector {
    InstantTeacherVector {
        vector: vector
            .vector
            .iter()
            .copied()
            .map(sanitize_feature)
            .collect(),
        metadata: vector_ref(vector),
    }
}

fn vector_ref(vector: &VectorEmbedding) -> EmbodiedVectorRef {
    EmbodiedVectorRef {
        vectorizer_id: vector.vectorizer_id.clone(),
        model_id: vector.model_id.clone(),
        model_label: vector.model_label.clone(),
        dim: vector.dim,
        modality: vector.modality.clone(),
        payload_kind: vector.payload_kind.clone(),
        source_kind: vector.source_kind.clone(),
        source_sensation_id: vector.source_sensation_id,
        purpose: vector.purpose.clone(),
        collection: vector.collection.clone(),
        input_summary: vector.input_summary.clone(),
        is_fallback: vector.is_fallback,
        provenance: vector.provenance.clone(),
    }
}

#[async_trait]
pub trait SensationVectorizer: Send + Sync {
    fn vectorizer_id(&self) -> &str;
    fn modality(&self) -> Modality;
    fn payload_kind(&self) -> SensationPayloadKind;
    fn model_id(&self) -> &str;
    fn model_label(&self) -> &str {
        self.model_id()
    }
    fn output_dim(&self) -> usize;
    fn purpose(&self) -> &str;
    fn collection(&self) -> &str {
        self.purpose()
    }
    fn is_fallback(&self) -> bool {
        false
    }
    async fn vectorize(&self, sensation: &Sensation) -> Result<VectorEmbedding>;
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedVectorizerRegistryConfig {
    #[serde(default)]
    pub vectorizer: BTreeMap<String, EmbodiedVectorizerConfig>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedVectorizerConfig {
    #[serde(default = "default_vectorizer_enabled")]
    pub enabled: bool,
    pub model: Option<String>,
    pub model_label: Option<String>,
    pub model_path: Option<String>,
    pub purpose: Option<String>,
    pub collection: Option<String>,
    pub fallback: Option<String>,
}

fn default_vectorizer_enabled() -> bool {
    true
}

#[derive(Clone, Default)]
pub struct SensationVectorizerRegistry {
    vectorizers: BTreeMap<(Modality, SensationPayloadKind), Arc<dyn SensationVectorizer>>,
    duplicate_state: Arc<Mutex<BTreeMap<(Modality, SensationPayloadKind), Vec<f32>>>>,
}

impl SensationVectorizerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_defaults() -> Self {
        Self::from_config(&EmbodiedVectorizerRegistryConfig::default())
    }

    pub fn from_config(config: &EmbodiedVectorizerRegistryConfig) -> Self {
        let mut registry = Self::new();

        registry.register_configured(
            config,
            "vision_image",
            EmbodiedFeatureSensationVectorizer::image(
                "pete.vectorizer.vision_image.frame_stats.v1",
                SensationPayloadKind::ImageBytes,
                "scene_similarity",
            ),
        );
        registry.register_configured(
            config,
            "vision_crop",
            EmbodiedFeatureSensationVectorizer::image(
                "pete.vectorizer.vision_crop.frame_stats.v1",
                SensationPayloadKind::Crop,
                "face_identity",
            ),
        );
        registry.register_configured(
            config,
            "vision_features",
            EmbodiedFeatureSensationVectorizer {
                vectorizer_id: "pete.vectorizer.vision_features.artifact.v1".to_string(),
                modality: Modality::Vision,
                payload_kind: SensationPayloadKind::Structured,
                model_id: "pete.image.feature_artifact.v1".to_string(),
                model_label: "pete.image.feature_artifact.v1".to_string(),
                purpose: "visual_similarity".to_string(),
                collection: "visual_similarity".to_string(),
                kind: EmbodiedFeatureKind::Image,
            },
        );
        registry.register_configured(
            config,
            "audio_pcm",
            EmbodiedFeatureSensationVectorizer::audio(
                "pete.vectorizer.audio_pcm.window_stats.v1",
                SensationPayloadKind::AudioPcm,
                "voice_identity",
            ),
        );
        registry.register_configured(
            config,
            "audio_voice",
            EmbodiedFeatureSensationVectorizer::audio(
                "pete.vectorizer.audio_voice.window_stats.v1",
                SensationPayloadKind::VoiceSegment,
                "voice_identity",
            ),
        );
        registry.register_configured(
            config,
            "audio_speech",
            EmbodiedFeatureSensationVectorizer::text(
                "pete.vectorizer.audio_speech.text_hashing.v1",
                SensationPayloadKind::SpeechSegment,
                "transcript_semantic",
            ),
        );
        registry.register_configured(
            config,
            "audio_transcript",
            EmbodiedFeatureSensationVectorizer::text(
                "pete.vectorizer.audio_transcript.text_hashing.v1",
                SensationPayloadKind::TranscriptSpan,
                "transcript_semantic",
            ),
        );
        registry.register_configured(
            config,
            "audio_phoneme",
            EmbodiedFeatureSensationVectorizer::text(
                "pete.vectorizer.audio_phoneme.text_hashing.v1",
                SensationPayloadKind::PhonemeSpan,
                "transcript_semantic",
            ),
        );
        registry.register_configured(
            config,
            "depth_scene",
            EmbodiedFeatureSensationVectorizer::depth(
                "pete.vectorizer.depth_scene.scene_stats.v1",
                SensationPayloadKind::DepthFrame,
                "scene_similarity",
            ),
        );

        for (modality, payload_kind) in default_vectorizer_keys() {
            if registry.get(&modality, &payload_kind).is_none() {
                registry.register(PlaceholderSensationVectorizer::new(modality, payload_kind));
            }
        }
        registry
    }

    pub fn from_models_toml(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|error| anyhow!("read vectorizer config {}: {error}", path.display()))?;
        let config: EmbodiedVectorizerRegistryConfig = toml::from_str(&text)
            .map_err(|error| anyhow!("parse vectorizer config {}: {error}", path.display()))?;
        Ok(Self::from_config(&config))
    }

    fn register_configured(
        &mut self,
        config: &EmbodiedVectorizerRegistryConfig,
        key: &str,
        mut vectorizer: EmbodiedFeatureSensationVectorizer,
    ) {
        let entry = config.vectorizer.get(key);
        if entry.is_some_and(|entry| !entry.enabled) {
            self.register(PlaceholderSensationVectorizer::new(
                vectorizer.modality(),
                vectorizer.payload_kind(),
            ));
            return;
        }
        if let Some(path) = entry.and_then(|entry| entry.model_path.as_deref()) {
            if !Path::new(path).exists() {
                eprintln!(
                    "warning: vectorizer {key} model_path {path} is missing; using deterministic placeholder fallback"
                );
                self.register(PlaceholderSensationVectorizer::new(
                    vectorizer.modality(),
                    vectorizer.payload_kind(),
                ));
                return;
            }
        }
        if let Some(model) = entry.and_then(|entry| entry.model.clone()) {
            vectorizer.model_id = model.clone();
            vectorizer.model_label = model;
        }
        if let Some(label) = entry.and_then(|entry| entry.model_label.clone()) {
            vectorizer.model_label = label;
        }
        if let Some(purpose) = entry.and_then(|entry| entry.purpose.clone()) {
            vectorizer.purpose = purpose;
        }
        if let Some(collection) = entry.and_then(|entry| entry.collection.clone()) {
            vectorizer.collection = collection;
        }
        self.register(vectorizer);
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
        let embedding = vectorizer.vectorize(sensation).await?;
        if should_suppress_duplicate_embedding(sensation, &embedding) {
            let key = (embedding.modality.clone(), embedding.payload_kind.clone());
            let mut duplicate_state = self
                .duplicate_state
                .lock()
                .map_err(|_| anyhow!("vectorizer duplicate suppression lock poisoned"))?;
            let duplicate = duplicate_state
                .get(&key)
                .is_some_and(|previous| cosine_similarity(previous, &embedding.vector) > 0.999);
            if duplicate {
                return Ok(None);
            }
            duplicate_state.insert(key, embedding.vector.clone());
        }
        Ok(Some(embedding))
    }
}

fn default_vectorizer_keys() -> [(Modality, SensationPayloadKind); 13] {
    [
        (Modality::Vision, SensationPayloadKind::ImageBytes),
        (Modality::Vision, SensationPayloadKind::Crop),
        (Modality::Vision, SensationPayloadKind::Structured),
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
    ]
}

#[derive(Clone, Debug)]
enum EmbodiedFeatureKind {
    Image,
    Audio,
    Text,
    Depth,
}

#[derive(Clone, Debug)]
pub struct EmbodiedFeatureSensationVectorizer {
    vectorizer_id: String,
    modality: Modality,
    payload_kind: SensationPayloadKind,
    model_id: String,
    model_label: String,
    purpose: String,
    collection: String,
    kind: EmbodiedFeatureKind,
}

impl EmbodiedFeatureSensationVectorizer {
    pub fn image(
        vectorizer_id: impl Into<String>,
        payload_kind: SensationPayloadKind,
        purpose: impl Into<String>,
    ) -> Self {
        let model_id = "pete.image.frame_stats.v1".to_string();
        let purpose = purpose.into();
        Self {
            vectorizer_id: vectorizer_id.into(),
            modality: Modality::Vision,
            payload_kind,
            model_id: model_id.clone(),
            model_label: model_id,
            collection: purpose.clone(),
            purpose,
            kind: EmbodiedFeatureKind::Image,
        }
    }

    pub fn audio(
        vectorizer_id: impl Into<String>,
        payload_kind: SensationPayloadKind,
        purpose: impl Into<String>,
    ) -> Self {
        let model_id = "pete.audio.window_stats.v1".to_string();
        let purpose = purpose.into();
        Self {
            vectorizer_id: vectorizer_id.into(),
            modality: Modality::Audio,
            payload_kind,
            model_id: model_id.clone(),
            model_label: model_id,
            collection: purpose.clone(),
            purpose,
            kind: EmbodiedFeatureKind::Audio,
        }
    }

    pub fn text(
        vectorizer_id: impl Into<String>,
        payload_kind: SensationPayloadKind,
        purpose: impl Into<String>,
    ) -> Self {
        let model_id = TEXT_HASH_MODEL_ID.to_string();
        let purpose = purpose.into();
        Self {
            vectorizer_id: vectorizer_id.into(),
            modality: Modality::Audio,
            payload_kind,
            model_id: model_id.clone(),
            model_label: model_id,
            collection: purpose.clone(),
            purpose,
            kind: EmbodiedFeatureKind::Text,
        }
    }

    pub fn depth(
        vectorizer_id: impl Into<String>,
        payload_kind: SensationPayloadKind,
        purpose: impl Into<String>,
    ) -> Self {
        let model_id = "pete.depth.scene_stats.v1".to_string();
        let purpose = purpose.into();
        Self {
            vectorizer_id: vectorizer_id.into(),
            modality: Modality::Depth,
            payload_kind,
            model_id: model_id.clone(),
            model_label: model_id,
            collection: purpose.clone(),
            purpose,
            kind: EmbodiedFeatureKind::Depth,
        }
    }
}

#[async_trait]
impl SensationVectorizer for EmbodiedFeatureSensationVectorizer {
    fn vectorizer_id(&self) -> &str {
        &self.vectorizer_id
    }

    fn modality(&self) -> Modality {
        self.modality.clone()
    }

    fn payload_kind(&self) -> SensationPayloadKind {
        self.payload_kind.clone()
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_label(&self) -> &str {
        &self.model_label
    }

    fn output_dim(&self) -> usize {
        EMBODIED_FEATURE_VECTOR_DIM
    }

    fn purpose(&self) -> &str {
        &self.purpose
    }

    fn collection(&self) -> &str {
        &self.collection
    }

    async fn vectorize(&self, sensation: &Sensation) -> Result<VectorEmbedding> {
        if let Some(artifact) = precomputed_payload_embedding(sensation) {
            return Ok(VectorEmbedding::new(
                sanitize_vector(artifact.vector),
                artifact.model_id.clone(),
                self.modality.clone(),
                self.payload_kind.clone(),
                sensation.id,
                sensation.observed_at_ms,
            )
            .with_metadata(
                artifact.vectorizer_id,
                artifact.model_label,
                artifact.purpose,
                artifact.collection,
                artifact.input_summary,
                false,
                "precomputed_vector_artifact",
            ));
        }

        let vector = match self.kind {
            EmbodiedFeatureKind::Image => image_feature_vector(sensation),
            EmbodiedFeatureKind::Audio => audio_feature_vector(sensation),
            EmbodiedFeatureKind::Text => text_feature_vector(sensation),
            EmbodiedFeatureKind::Depth => depth_feature_vector(sensation),
        };
        Ok(VectorEmbedding::new(
            vector,
            self.model_id.clone(),
            self.modality.clone(),
            self.payload_kind.clone(),
            sensation.id,
            sensation.observed_at_ms,
        )
        .with_metadata(
            self.vectorizer_id.clone(),
            self.model_label.clone(),
            self.purpose.clone(),
            self.collection.clone(),
            input_summary_for_sensation(sensation),
            false,
            "pete_embodied_feature_vectorizer",
        ))
    }
}

#[derive(Clone, Debug)]
struct PrecomputedPayloadEmbedding {
    vector: Vec<f32>,
    model_id: String,
    model_label: String,
    vectorizer_id: String,
    purpose: String,
    collection: String,
    input_summary: String,
}

fn precomputed_payload_embedding(sensation: &Sensation) -> Option<PrecomputedPayloadEmbedding> {
    let artifacts = sensation.payload.get("vector_artifacts")?.as_array()?;
    for artifact in artifacts {
        let vector = artifact
            .get("vector")
            .and_then(Value::as_array)?
            .iter()
            .filter_map(|value| value.as_f64().map(|value| value as f32))
            .collect::<Vec<_>>();
        if vector.is_empty() {
            continue;
        }
        let model_id = artifact
            .get("model")
            .and_then(Value::as_str)
            .filter(|model| !model.trim().is_empty())
            .unwrap_or("pete.precomputed_vector.v0")
            .to_string();
        let collection = artifact
            .get("collection")
            .and_then(Value::as_str)
            .filter(|collection| !collection.trim().is_empty())
            .unwrap_or("precomputed_vectors")
            .to_string();
        let purpose = purpose_for_collection(&collection);
        let point_id = artifact
            .get("point_id")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let source_id = artifact
            .get("source_id")
            .and_then(Value::as_str)
            .or_else(|| artifact.get("source_frame_id").and_then(Value::as_str))
            .unwrap_or("unknown");
        return Some(PrecomputedPayloadEmbedding {
            vector,
            model_id: model_id.clone(),
            model_label: model_id.clone(),
            vectorizer_id: format!("precomputed.{collection}.{model_id}"),
            purpose,
            collection,
            input_summary: format!("vector_artifact point_id={point_id} source_id={source_id}"),
        });
    }
    None
}

fn image_feature_vector(sensation: &Sensation) -> Vec<f32> {
    let mut vector = base_sensation_features(sensation);
    let Some(frame) = VisualFrame::from_sensation(sensation) else {
        return pad_feature_vector(vector);
    };
    let pixels = (frame.width as usize).saturating_mul(frame.height as usize);
    if pixels == 0 {
        return pad_feature_vector(vector);
    }
    let mut sums = [0.0_f32; 3];
    let mut sq_sums = [0.0_f32; 3];
    let mut mins = [1.0_f32; 3];
    let mut maxs = [0.0_f32; 3];
    let mut luma_sum = 0.0_f32;
    let mut skin_like = 0_usize;
    for pixel in frame.rgb.chunks_exact(3).take(pixels) {
        let rgb = [
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
        ];
        for channel in 0..3 {
            sums[channel] += rgb[channel];
            sq_sums[channel] += rgb[channel] * rgb[channel];
            mins[channel] = mins[channel].min(rgb[channel]);
            maxs[channel] = maxs[channel].max(rgb[channel]);
        }
        luma_sum += 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        if is_skin_like_rgb(pixel[0], pixel[1], pixel[2]) {
            skin_like += 1;
        }
    }
    for channel in 0..3 {
        let mean = sums[channel] / pixels as f32;
        let variance = (sq_sums[channel] / pixels as f32) - mean * mean;
        vector.push(mean);
        vector.push(variance.max(0.0).sqrt());
        vector.push(mins[channel]);
        vector.push(maxs[channel]);
    }
    vector.push(luma_sum / pixels as f32);
    vector.push(skin_like as f32 / pixels as f32);
    if let Some(bbox) = sensation.metadata.bbox {
        vector.push((bbox.x as f32 / frame.width.max(1) as f32).clamp(0.0, 1.0));
        vector.push((bbox.y as f32 / frame.height.max(1) as f32).clamp(0.0, 1.0));
        vector.push((bbox.width as f32 / frame.width.max(1) as f32).clamp(0.0, 1.0));
        vector.push((bbox.height as f32 / frame.height.max(1) as f32).clamp(0.0, 1.0));
    }
    push_grid_luma(&mut vector, &frame);
    pad_feature_vector(vector)
}

fn push_grid_luma(vector: &mut Vec<f32>, frame: &VisualFrame) {
    let width = frame.width as usize;
    let height = frame.height as usize;
    if width == 0 || height == 0 {
        return;
    }
    for gy in 0..2 {
        for gx in 0..2 {
            let x0 = gx * width / 2;
            let x1 = ((gx + 1) * width / 2).max(x0 + 1).min(width);
            let y0 = gy * height / 2;
            let y1 = ((gy + 1) * height / 2).max(y0 + 1).min(height);
            let mut sum = 0.0_f32;
            let mut count = 0_usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = (y * width + x) * 3;
                    let r = frame.rgb[idx] as f32 / 255.0;
                    let g = frame.rgb[idx + 1] as f32 / 255.0;
                    let b = frame.rgb[idx + 2] as f32 / 255.0;
                    sum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    count += 1;
                }
            }
            vector.push(if count > 0 { sum / count as f32 } else { 0.0 });
        }
    }
}

fn audio_feature_vector(sensation: &Sensation) -> Vec<f32> {
    let mut vector = base_sensation_features(sensation);
    vector.push(
        sensation
            .metadata
            .duration_ms
            .map(|value| (value as f32 / 10_000.0).clamp(0.0, 1.0))
            .unwrap_or_default(),
    );
    for key in [
        "feature_sets",
        "duration_ms",
        "start_offset_ms",
        "end_offset_ms",
        "confidence",
    ] {
        vector.push(payload_number_unit(&sensation.payload, key));
    }
    if let Some(asr) = sensation.payload.get("asr") {
        vector.push(payload_number_unit(asr, "confidence"));
        vector.push(
            asr.get("is_final")
                .and_then(Value::as_bool)
                .map(bool01)
                .unwrap_or_default(),
        );
        vector.push(
            asr.get("word_count")
                .and_then(Value::as_u64)
                .map(|value| (value as f32 / 32.0).clamp(0.0, 1.0))
                .unwrap_or_default(),
        );
    }
    if let Some(text) = sensation
        .payload
        .get("transcript")
        .and_then(Value::as_str)
        .or_else(|| sensation.payload.get("text").and_then(Value::as_str))
    {
        push_text_hash_features(&mut vector, text, 8);
    }
    pad_feature_vector(vector)
}

fn text_feature_vector(sensation: &Sensation) -> Vec<f32> {
    let mut vector = base_sensation_features(sensation);
    let text = sensation
        .payload
        .get("text")
        .and_then(Value::as_str)
        .or(sensation.summary.as_deref())
        .unwrap_or_default();
    let chars = text.chars().count();
    let words = text.split_whitespace().count();
    vector.push((chars as f32 / 280.0).clamp(0.0, 1.0));
    vector.push((words as f32 / 48.0).clamp(0.0, 1.0));
    vector.push(bool01(
        text.chars()
            .last()
            .is_some_and(|ch| matches!(ch, '?' | '!')),
    ));
    push_text_hash_features(&mut vector, text, EMBODIED_FEATURE_VECTOR_DIM);
    pad_feature_vector(vector)
}

fn depth_feature_vector(sensation: &Sensation) -> Vec<f32> {
    let mut vector = base_sensation_features(sensation);
    for key in [
        "sample_count",
        "width",
        "height",
        "min_depth_m",
        "max_depth_m",
        "skeleton_count",
    ] {
        vector.push(payload_number_unit(&sensation.payload, key));
    }
    let min_depth = sensation
        .payload
        .get("min_depth_m")
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .unwrap_or_default();
    let max_depth = sensation
        .payload
        .get("max_depth_m")
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .unwrap_or_default();
    vector.push((max_depth - min_depth).max(0.0).min(10.0) / 10.0);
    pad_feature_vector(vector)
}

fn base_sensation_features(sensation: &Sensation) -> Vec<f32> {
    let mut vector = Vec::with_capacity(EMBODIED_FEATURE_VECTOR_DIM);
    vector.push(stable_unit(&sensation.kind));
    vector.push(stable_unit(&sensation.source));
    vector.push((sensation.occurred_at_ms % 10_000) as f32 / 10_000.0);
    vector.push(sensation.metadata.confidence.unwrap_or(0.5).clamp(0.0, 1.0));
    vector.push(bool01(sensation.parent_id.is_some()));
    for label in sensation.metadata.labels.iter().take(4) {
        vector.push(stable_unit(label));
    }
    vector
}

fn push_text_hash_features(vector: &mut Vec<f32>, text: &str, max_dim: usize) {
    let reserve = max_dim.saturating_sub(vector.len());
    if reserve == 0 {
        return;
    }
    let mut buckets = vec![0.0_f32; reserve.min(16)];
    for token in text.split_whitespace() {
        let mut hash = 0_u32;
        for byte in token.bytes() {
            hash = hash.wrapping_mul(16777619) ^ u32::from(byte.to_ascii_lowercase());
        }
        let idx = (hash as usize) % buckets.len();
        buckets[idx] += 1.0;
    }
    let norm = buckets
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    for bucket in buckets {
        vector.push(if norm > 0.0 { bucket / norm } else { 0.0 });
    }
}

fn payload_number_unit(payload: &Value, key: &str) -> f32 {
    payload
        .get(key)
        .and_then(Value::as_f64)
        .map(|value| (value as f32).abs())
        .map(|value| (value / (value + 1.0)).clamp(0.0, 1.0))
        .unwrap_or_default()
}

fn pad_feature_vector(mut vector: Vec<f32>) -> Vec<f32> {
    vector = sanitize_vector(vector);
    vector.truncate(EMBODIED_FEATURE_VECTOR_DIM);
    while vector.len() < EMBODIED_FEATURE_VECTOR_DIM {
        vector.push(0.0);
    }
    vector
}

fn sanitize_vector(vector: Vec<f32>) -> Vec<f32> {
    vector
        .into_iter()
        .map(|value| {
            if value.is_finite() {
                value.clamp(-1.0, 1.0)
            } else {
                0.0
            }
        })
        .collect()
}

fn semantic_text_vector(
    text: &str,
    source_id: Uuid,
    generated_at_ms: TimeMs,
    source_kind: impl Into<String>,
    purpose: impl Into<String>,
    collection: impl Into<String>,
    input_summary: impl Into<String>,
) -> VectorEmbedding {
    let purpose = purpose.into();
    let collection = collection.into();
    VectorEmbedding::new(
        text_hash_vector(text, TEXT_HASH_VECTOR_DIM),
        TEXT_HASH_MODEL_ID,
        Modality::Other,
        SensationPayloadKind::Structured,
        source_id,
        generated_at_ms,
    )
    .with_metadata(
        format!("pete.vectorizer.{purpose}.text_hashing.v1"),
        "Pete deterministic text hashing baseline",
        purpose,
        collection,
        input_summary,
        false,
        "pete_text_hashing_vectorizer",
    )
    .with_source_kind(source_kind)
}

fn text_hash_vector(text: &str, dim: usize) -> Vec<f32> {
    let dim = dim.max(1);
    let mut vector = vec![0.0_f32; dim];
    let mut token_count = 0.0_f32;
    for token in text
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| !token.is_empty())
    {
        token_count += 1.0;
        let normalized = token.to_ascii_lowercase();
        for ngram in token_ngrams(&normalized) {
            let mut hash = 2166136261_u32;
            for byte in ngram.bytes() {
                hash = hash.wrapping_mul(16777619) ^ u32::from(byte);
            }
            let index = (hash as usize) % dim;
            let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
            vector[index] += sign;
        }
    }
    vector[0] += (text.chars().count() as f32 / 512.0).clamp(0.0, 1.0);
    if dim > 1 {
        vector[1] += (token_count / 96.0).clamp(0.0, 1.0);
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in &mut vector {
            *value = (*value / norm).clamp(-1.0, 1.0);
        }
    }
    vector
}

fn token_ngrams(token: &str) -> Vec<String> {
    let chars = token.chars().collect::<Vec<_>>();
    if chars.len() <= 3 {
        return vec![token.to_string()];
    }
    let mut ngrams = Vec::new();
    for window in chars.windows(3) {
        ngrams.push(window.iter().collect());
    }
    ngrams.push(token.to_string());
    ngrams
}

fn purpose_for_sensation(modality: &Modality, payload_kind: &SensationPayloadKind) -> String {
    match (modality, payload_kind) {
        (Modality::Vision, SensationPayloadKind::ImageBytes) => "scene_similarity",
        (Modality::Vision, SensationPayloadKind::Crop) => "face_identity",
        (Modality::Vision, SensationPayloadKind::Structured) => "visual_similarity",
        (Modality::Audio, SensationPayloadKind::TranscriptSpan)
        | (Modality::Audio, SensationPayloadKind::SpeechSegment)
        | (Modality::Audio, SensationPayloadKind::PhonemeSpan) => "transcript_semantic",
        (Modality::Audio, SensationPayloadKind::VoiceSegment)
        | (Modality::Audio, SensationPayloadKind::AudioPcm) => "voice_identity",
        (Modality::Depth, SensationPayloadKind::DepthFrame) => "scene_similarity",
        (Modality::Other, SensationPayloadKind::Structured) => "experience_semantic",
        _ => "embodied_similarity",
    }
    .to_string()
}

fn purpose_for_collection(collection: &str) -> String {
    match collection {
        "faces" => "face_identity",
        "objects" => "object_identity",
        "voices" => "voice_identity",
        "scene_vectors" | "images" => "scene_similarity",
        "image_descriptions" | "memories" | "transcripts" => "transcript_semantic",
        "impressions" => "impression_semantic",
        "experiences" => "experience_semantic",
        _ => collection,
    }
    .to_string()
}

fn input_summary_for_sensation(sensation: &Sensation) -> String {
    let mut parts = vec![
        format!("kind={}", sensation.kind),
        format!("payload_kind={}", sensation.payload_kind.as_str()),
    ];
    if let Some(summary) = sensation
        .summary
        .as_deref()
        .filter(|summary| !summary.is_empty())
    {
        parts.push(format!(
            "summary={}",
            summary.chars().take(96).collect::<String>()
        ));
    }
    if let Some(width) = sensation.payload.get("width").and_then(Value::as_u64) {
        if let Some(height) = sensation.payload.get("height").and_then(Value::as_u64) {
            parts.push(format!("size={}x{}", width, height));
        }
    }
    if let Some(format) = sensation.payload.get("format").and_then(Value::as_str) {
        parts.push(format!("format={format}"));
    }
    parts.join(" ")
}

fn should_suppress_duplicate_embedding(sensation: &Sensation, embedding: &VectorEmbedding) -> bool {
    !embedding.is_fallback
        && matches!(embedding.modality, Modality::Vision)
        && matches!(
            embedding.payload_kind,
            SensationPayloadKind::ImageBytes | SensationPayloadKind::Crop
        )
        && VisualFrame::from_sensation(sensation).is_some()
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut left_norm = 0.0_f32;
    let mut right_norm = 0.0_f32;
    for (left, right) in left.iter().zip(right.iter()) {
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    let denom = left_norm.sqrt() * right_norm.sqrt();
    if denom > 0.0 {
        dot / denom
    } else {
        0.0
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
    fn vectorizer_id(&self) -> &str {
        "pete.vectorizer.placeholder.v0"
    }

    fn modality(&self) -> Modality {
        self.modality.clone()
    }

    fn payload_kind(&self) -> SensationPayloadKind {
        self.payload_kind.clone()
    }

    fn model_id(&self) -> &str {
        "pete.placeholder.v0"
    }

    fn output_dim(&self) -> usize {
        PLACEHOLDER_VECTOR_DIM
    }

    fn purpose(&self) -> &str {
        "fallback_deterministic"
    }

    fn collection(&self) -> &str {
        "fallback_vectors"
    }

    fn is_fallback(&self) -> bool {
        true
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
        )
        .with_metadata(
            self.vectorizer_id(),
            self.model_id(),
            purpose_for_sensation(&self.modality, &self.payload_kind),
            self.collection(),
            input_summary_for_sensation(sensation),
            true,
            "deterministic_placeholder_fallback",
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
        if let Some(text) = window.possible_transcript.as_deref() {
            descendants.push(audio_possible_speech(sensation, &window, text));
        }
        if let Some(text) = window.committed_transcript.as_deref().or_else(|| {
            window
                .is_final
                .then_some(window.transcript.as_deref())
                .flatten()
        }) {
            descendants.push(audio_committed_speech(sensation, &window, text));
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
    possible_transcript: Option<String>,
    committed_transcript: Option<String>,
    candidate_id: Option<u64>,
    stable_text: Option<String>,
    unstable_text: Option<String>,
    stable_word_prefix: Option<String>,
    stable_word_count: Option<u64>,
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
            possible_transcript: asr
                .get("possible_transcript")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .map(ToOwned::to_owned),
            committed_transcript: asr
                .get("committed_transcript")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .map(ToOwned::to_owned),
            candidate_id: asr.get("candidate_id").and_then(Value::as_u64),
            stable_text: asr
                .get("stable_text")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            unstable_text: asr
                .get("unstable_text")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            stable_word_prefix: asr
                .get("stable_word_prefix")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            stable_word_count: asr.get("stable_word_count").and_then(Value::as_u64),
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

fn audio_possible_speech(parent: &Sensation, window: &AudioWindow, transcript: &str) -> Sensation {
    let mut metadata = parent.metadata.clone();
    metadata.duration_ms = Some(window.duration_ms);
    metadata.confidence = Some(window.confidence.max(0.25));
    push_label(&mut metadata, "speech");
    push_label(&mut metadata, "possible speech");
    metadata
        .properties
        .insert("start_ms".to_string(), json!(window.start_ms));
    metadata
        .properties
        .insert("end_ms".to_string(), json!(window.end_ms));
    metadata
        .properties
        .insert("commitment".to_string(), json!("possible"));
    let mut sensation = Sensation::descendant(
        parent,
        "audio.possible_speech",
        SensationPayloadKind::SpeechSegment,
        json!({
            "parent_audio": parent.id,
            "start_ms": window.start_ms,
            "end_ms": window.end_ms,
            "duration_ms": window.duration_ms,
            "text": transcript,
            "commitment": "possible",
            "is_final": false,
            "confidence": window.confidence,
            "candidate_id": window.candidate_id,
            "stable_text": window.stable_text,
            "unstable_text": window.unstable_text,
            "stable_word_prefix": window.stable_word_prefix,
            "stable_word_count": window.stable_word_count,
            "method": "asr_tool_transcript_candidate",
        }),
        metadata,
        "descendant.audio_possible_speech",
    )
    .with_summary(format!("I may be hearing someone say \"{transcript}\"."));
    sensation.occurred_at_ms = window.start_ms;
    sensation
}

fn audio_committed_speech(parent: &Sensation, window: &AudioWindow, transcript: &str) -> Sensation {
    let mut metadata = parent.metadata.clone();
    metadata.duration_ms = Some(window.duration_ms);
    metadata.confidence = Some(window.confidence.max(0.35));
    push_label(&mut metadata, "speech");
    push_label(&mut metadata, "committed speech");
    metadata
        .properties
        .insert("start_ms".to_string(), json!(window.start_ms));
    metadata
        .properties
        .insert("end_ms".to_string(), json!(window.end_ms));
    metadata
        .properties
        .insert("commitment".to_string(), json!("committed"));
    let mut sensation = Sensation::descendant(
        parent,
        "audio.committed_speech",
        SensationPayloadKind::TranscriptSpan,
        json!({
            "parent_audio": parent.id,
            "start_ms": window.start_ms,
            "end_ms": window.end_ms,
            "duration_ms": window.duration_ms,
            "text": transcript,
            "commitment": "committed",
            "is_final": true,
            "confidence": window.confidence,
            "candidate_id": window.candidate_id,
            "stable_text": window.stable_text,
            "stable_word_prefix": window.stable_word_prefix,
            "stable_word_count": window.stable_word_count,
            "method": "asr_tool_transcript_commit",
        }),
        metadata,
        "descendant.audio_committed_speech",
    )
    .with_summary(format!(
        "I commit that I heard someone say \"{transcript}\"."
    ));
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
        let mut impression = Impression::new(
            "sensation.template",
            text.clone(),
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
        }));
        impression.vector = Some(semantic_text_vector(
            &text,
            impression.id,
            sensation.observed_at_ms,
            "impression",
            "impression_semantic",
            "impressions",
            format!(
                "impression kind={} about_sensation={} text={}",
                impression.kind,
                sensation.id,
                text.chars().take(96).collect::<String>()
            ),
        ));
        impression
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
        let mut impression = Impression::new(
            "experience.template",
            text.clone(),
            Vec::new(),
            window_start_ms,
            window_end_ms,
        )
        .for_experience(experience_id)
        .with_confidence(0.6);
        impression.vector = Some(semantic_text_vector(
            &text,
            experience_id,
            window_end_ms,
            "experience",
            "experience_semantic",
            "experiences",
            format!(
                "experience_summary id={} text={}",
                experience_id,
                text.chars().take(96).collect::<String>()
            ),
        ));
        impression
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

    pub fn with_vectorizers(vectorizers: SensationVectorizerRegistry) -> Self {
        Self {
            vectorizers,
            ..Self::default()
        }
    }

    pub fn from_models_toml(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::with_vectorizers(
            SensationVectorizerRegistry::from_models_toml(path)?,
        ))
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
        experience.summary_impression = Some(summary);
        experience.predictions = vec![Prediction {
            offset_ms: self.window_ms,
            text:
                "I expect the next moment to resemble this one unless I move or something changes."
                    .to_string(),
            confidence: 0.35,
            vector: None,
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
    let mut now = Now::blank(now_ms, BodySense::default());
    now.eye_frame = Some(pete_now::EyeFrame {
        captured_at_ms: now_ms,
        width: 64,
        height: 48,
        format: pete_now::EyeFrameFormat::Rgb8,
        bytes: rgb,
        source: Some("demo.synthetic_camera".to_string()),
    });
    now.face.vectors.push(
        pete_now::VectorArtifact::new("faces", "demo-face-vector", vec![0.17, 0.41, 0.73, 0.29])
            .with_model("face_id/0.4.1")
            .with_source_id("demo-face")
            .with_source_frame_id("demo-synthetic-frame")
            .with_occurred_at_ms(now_ms),
    );
    now.ear.transcript = Some("hello pete, this is a transcript vector test".to_string());
    now.ear.asr.transcript = now.ear.transcript.clone();
    now.ear.asr.is_final = true;
    now.ear.asr.confidence = 0.82;
    now.ear.asr.start_ms = Some(now_ms.saturating_sub(320));
    now.ear.asr.end_ms = Some(now_ms);
    now.ear.asr.duration_ms = Some(320);
    now.ear.asr.word_count = Some(8);
    now.voice.vectors.push(
        pete_now::VectorArtifact::new(
            "voices",
            "demo-voice-vector",
            vec![0.11, 0.05, 0.33, 0.78, 0.21],
        )
        .with_model("pete/voice_vector/16d")
        .with_source_id("demo-voice")
        .with_occurred_at_ms(now_ms),
    );
    let pipeline = EmbodiedPipeline::from_models_toml("configs/models.toml").unwrap_or_else(|error| {
        eprintln!(
            "warning: embodied demo could not load configs/models.toml ({error}); using built-in vectorizer defaults"
        );
        EmbodiedPipeline::new()
    });
    let mut sensations = Vec::new();
    let mut impressions = Vec::new();
    for primary in primary_sensations_from_now(&now) {
        let batch = pipeline.ingest_primary(primary).await?;
        sensations.extend(batch.sensations);
        impressions.extend(batch.impressions);
    }
    let batch = EmbodiedBatch {
        sensations,
        impressions,
    };
    let mut window = RollingExperienceWindow::new(DEFAULT_WINDOW_MS);
    window.push(batch.clone());
    let experience = window.fuse_current()?;
    let coverage = EmbodiedVectorCoverage::from_parts(
        &batch.sensations,
        &batch.impressions,
        Some(&experience),
    );
    Ok(EmbodiedDemo {
        sensations: batch.sensations,
        impressions: batch.impressions,
        experience,
        coverage,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedDemo {
    pub sensations: Vec<Sensation>,
    pub impressions: Vec<Impression>,
    pub experience: Experience,
    pub coverage: EmbodiedVectorCoverage,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedVectorCoverage {
    pub image: usize,
    pub face: usize,
    pub voice: usize,
    pub transcript: usize,
    pub impression: usize,
    pub experience: usize,
    pub fallback_count: usize,
}

impl EmbodiedVectorCoverage {
    pub fn from_parts(
        sensations: &[Sensation],
        impressions: &[Impression],
        experience: Option<&Experience>,
    ) -> Self {
        let mut coverage = Self::default();
        for vector in sensations
            .iter()
            .filter_map(|sensation| sensation.vector.as_ref())
            .chain(
                impressions
                    .iter()
                    .filter_map(|impression| impression.vector.as_ref()),
            )
            .chain(
                experience
                    .and_then(|experience| experience.summary_impression.as_ref())
                    .and_then(|impression| impression.vector.as_ref())
                    .into_iter(),
            )
        {
            coverage.record(vector);
        }
        coverage
    }

    fn record(&mut self, vector: &VectorEmbedding) {
        if vector.is_fallback {
            self.fallback_count += 1;
        }
        match vector.purpose.as_str() {
            "scene_similarity" | "visual_similarity" => self.image += 1,
            "face_identity" => self.face += 1,
            "voice_identity" => self.voice += 1,
            "transcript_semantic" => self.transcript += 1,
            "impression_semantic" => self.impression += 1,
            "experience_semantic" => self.experience += 1,
            _ => {}
        }
    }
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
        let mut vector_artifacts = now.eye.image_vectors.clone();
        vector_artifacts.extend(now.eye.scene_vectors.clone());
        vector_artifacts.extend(now.eye.image_description_vectors.clone());
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
                "vector_artifacts": vector_artifacts,
            })),
        )
        .with_summary("I have visual features from my eye.");
        sensation.metadata.confidence = Some(0.55);
        sensations.push(sensation);
    }

    if !now.face.vectors.is_empty() {
        let vector_artifacts = now.face.vectors.clone();
        let mut sensation = Sensation::primary(
            Modality::Vision,
            SensationSource::new("face.features"),
            now.t_ms,
            now.t_ms,
            SensationPayload {
                kind: SensationPayloadKind::Crop,
                value: json!({
                    "face_vectors": now.face.vectors.len(),
                    "vector_artifacts": vector_artifacts,
                }),
            },
        )
        .with_summary("I have a face embedding from vision.");
        sensation.metadata.confidence = Some(0.6);
        sensation.metadata.labels.push("face".to_string());
        sensations.push(sensation);
    }

    if !now.objects.vectors.is_empty() {
        let vector_artifacts = now.objects.vectors.clone();
        let mut sensation = Sensation::primary(
            Modality::Vision,
            SensationSource::new("object.features"),
            now.t_ms,
            now.t_ms,
            SensationPayload {
                kind: SensationPayloadKind::Crop,
                value: json!({
                    "object_observations": now.objects.observations.len(),
                    "object_vectors": now.objects.vectors.len(),
                    "vector_artifacts": vector_artifacts,
                }),
            },
        )
        .with_summary("I have object visual vectors from vision.");
        sensation.metadata.confidence = Some(0.6);
        sensation.metadata.labels.push("object".to_string());
        sensations.push(sensation);
    }

    if !now.ear.features.is_empty()
        || !now.ear.transcript_vectors.is_empty()
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
        || now
            .ear
            .asr
            .possible_transcript
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty())
        || now
            .ear
            .asr
            .committed_transcript
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty())
    {
        let transcript = now
            .ear
            .asr
            .committed_transcript
            .as_deref()
            .or(now.ear.asr.transcript.as_deref())
            .or(now.ear.asr.possible_transcript.as_deref())
            .or(now.ear.transcript.as_deref())
            .map(str::trim)
            .filter(|text| !text.is_empty());
        let legacy_transcript = now
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
                    "transcript_vectors": now.ear.transcript_vectors.len(),
                    "transcript": legacy_transcript.or(transcript),
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

    if !now.ear.transcript_vectors.is_empty() {
        let transcript = now
            .ear
            .asr
            .committed_transcript
            .as_deref()
            .or(now.ear.asr.transcript.as_deref())
            .or(now.ear.transcript.as_deref())
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .unwrap_or("speech transcript");
        let observed_at_ms = now.ear.asr.end_ms.unwrap_or(now.t_ms);
        let occurred_at_ms = now.ear.asr.start_ms.unwrap_or(observed_at_ms);
        let mut sensation = Sensation::primary(
            Modality::Audio,
            SensationSource::new("ear.transcript_vectors"),
            occurred_at_ms,
            observed_at_ms,
            SensationPayload {
                kind: SensationPayloadKind::TranscriptSpan,
                value: json!({
                    "text": transcript,
                    "transcript_vectors": now.ear.transcript_vectors.len(),
                    "vector_artifacts": now.ear.transcript_vectors.clone(),
                }),
            },
        )
        .with_summary(format!("I have a transcript vector for \"{transcript}\"."));
        sensation.metadata.confidence = Some(now.ear.asr.confidence.max(0.35).clamp(0.0, 1.0));
        sensation
            .metadata
            .labels
            .push("transcript vector".to_string());
        sensations.push(sensation);
    }

    if !now.voice.vectors.is_empty() {
        let vector_artifacts = now.voice.vectors.clone();
        let mut sensation = Sensation::primary(
            Modality::Audio,
            SensationSource::new("voice.features"),
            now.t_ms,
            now.t_ms,
            SensationPayload {
                kind: SensationPayloadKind::VoiceSegment,
                value: json!({
                    "voice_vectors": now.voice.vectors.len(),
                    "vector_artifacts": vector_artifacts,
                }),
            },
        )
        .with_summary("I have a voice embedding from hearing.");
        sensation.metadata.confidence = Some(0.6);
        sensation.metadata.labels.push("voice identity".to_string());
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
    #[serde(default)]
    pub original_frame_id: Option<Uuid>,
    #[serde(default)]
    pub original_vector_ids: Vec<String>,
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

    pub fn from_instant(now: &Now, instant: &ExperienceInstant) -> Self {
        let encode_input = ExperienceEncodeInput::from_instant(instant);
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
