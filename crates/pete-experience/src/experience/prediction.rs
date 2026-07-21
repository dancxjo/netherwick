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
