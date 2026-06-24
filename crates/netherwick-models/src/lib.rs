use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::{loss::MseLoss, loss::Reduction, Linear, LinearConfig};
use burn::optim::{adaptor::OptimizerAdaptor, GradientsParams, Optimizer, Sgd, SgdConfig};
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::{activation, backend::AutodiffBackend, backend::Backend, Tensor, TensorData};
use netherwick_behaviors::TrainingSample;
use netherwick_experience::{
    ActionValueInput, ActionValueOutput, ActionValueTarget, ChargeInput, ChargeOutput,
    ChargeTarget, DangerInput, DangerOutput, DangerTarget, EarNextInput, EarNextOutput,
    EarNextTarget, ExperienceDecodeFeatureLengths, ExperienceDecodeOutput, ExperienceEncodeInput,
    ExperienceEncodeOutput, EyeNextInput, EyeNextOutput, EyeNextTarget, EYE_NEXT_HEIGHT,
    EYE_NEXT_RGB_LEN, EYE_NEXT_WIDTH,
};
use netherwick_now::Now;
use serde::{Deserialize, Serialize};

pub trait NeuralModel<I, O> {
    fn predict(&self, input: I) -> Result<O>;
}

pub trait OnlineTrainer<I, O> {
    fn train_step(&mut self, sample: TrainingSample<I, O>) -> Result<TrainStats>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TrainStats {
    pub loss: f32,
    pub samples_seen: u64,
    pub improved: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DangerTrainStats {
    pub loss: f32,
    pub samples_seen: u64,
    pub improved: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DangerShadowMetric {
    pub observed_at_ms: u64,
    pub hardcoded: DangerOutput,
    pub model: DangerOutput,
    pub target: DangerTarget,
    pub loss: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChargeTrainStats {
    pub loss: f32,
    pub samples_seen: u64,
    pub improved: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChargeShadowMetric {
    pub observed_at_ms: u64,
    pub hardcoded: ChargeOutput,
    pub model: ChargeOutput,
    pub target: ChargeTarget,
    pub loss: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionValueTrainStats {
    pub loss: f32,
    pub samples_seen: u64,
    pub improved: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionValueShadowMetric {
    pub observed_at_ms: u64,
    pub hardcoded: ActionValueOutput,
    pub model: ActionValueOutput,
    pub target: ActionValueTarget,
    pub loss: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EyeNextTrainStats {
    pub loss: f32,
    pub samples_seen: u64,
    pub improved: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EyeNextShadowMetric {
    pub observed_at_ms: u64,
    pub hardcoded: EyeNextOutput,
    pub model: EyeNextOutput,
    pub target: EyeNextTarget,
    pub loss: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EarNextTrainStats {
    pub loss: f32,
    pub samples_seen: u64,
    pub improved: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EarNextShadowMetric {
    pub observed_at_ms: u64,
    pub hardcoded: EarNextOutput,
    pub model: EarNextOutput,
    pub target: EarNextTarget,
    pub loss: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceAutoencoderTrainStats {
    pub loss: f32,
    pub samples_seen: u64,
    pub improved: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceAutoencoderShadowMetric {
    pub observed_at_ms: u64,
    pub hardcoded: ExperienceDecodeOutput,
    pub model: ExperienceDecodeOutput,
    pub target: ExperienceDecodeOutput,
    pub z: ExperienceEncodeOutput,
    pub loss: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceAutoencoderPrediction {
    pub encoded: ExperienceEncodeOutput,
    pub decoded: ExperienceDecodeOutput,
}

#[derive(Clone, Debug, Default)]
pub struct HardcodedDangerPredictor;

impl HardcodedDangerPredictor {
    pub fn predict_from_now(&self, now: &Now, input: &DangerInput) -> DangerOutput {
        let mut bump_risk = 0.0_f32;
        let nearest_m = now.range.nearest_m;
        if let Some(distance) = nearest_m {
            if distance <= 0.12 {
                bump_risk = bump_risk.max(0.9);
            } else if distance <= 0.25 {
                bump_risk = bump_risk.max(0.65);
            } else if distance <= 0.45 {
                bump_risk = bump_risk.max(0.35);
            }
        }
        if now.body.flags.bump_left || now.body.flags.bump_right {
            bump_risk = 1.0;
        }

        let cliff_risk = if now.body.flags.cliff_left
            || now.body.flags.cliff_front_left
            || now.body.flags.cliff_front_right
            || now.body.flags.cliff_right
            || now.body.cliff_sensors.max() >= 0.5
        {
            1.0
        } else {
            0.0
        };
        let wheel_drop_risk = if now.body.flags.wheel_drop { 1.0 } else { 0.0 };
        let commanded_forward = input.action_features.get(10).copied().unwrap_or(0.0) > 0.01;
        let explore = input.action_features.get(4).copied().unwrap_or(0.0) > 0.5;
        let stuck_risk =
            if (commanded_forward || explore) && now.body.velocity.forward_m_s.abs() < 0.01 {
                0.85
            } else {
                0.0
            };
        let safety_vetoed = now
            .extensions
            .get("safety.vetoed")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);

        DangerOutput {
            bump_risk,
            cliff_risk,
            wheel_drop_risk,
            stuck_risk,
            confidence: if safety_vetoed || cliff_risk >= 1.0 || wheel_drop_risk >= 1.0 {
                1.0
            } else if nearest_m.is_some() || bump_risk > 0.0 || stuck_risk > 0.0 {
                0.75
            } else {
                0.6
            },
        }
    }
}

impl NeuralModel<(DangerInput, Now), DangerOutput> for HardcodedDangerPredictor {
    fn predict(&self, input: (DangerInput, Now)) -> Result<DangerOutput> {
        Ok(self.predict_from_now(&input.1, &input.0))
    }
}

#[derive(Clone, Debug, Default)]
pub struct HardcodedChargePredictor;

impl HardcodedChargePredictor {
    pub fn predict_from_now(&self, now: &Now, input: &ChargeInput) -> ChargeOutput {
        let already_charging = now.body.charging;
        let low_battery = now.body.battery_level < 0.35;
        let dock_action = input.action_features.get(6).copied().unwrap_or(0.0) > 0.5;
        let memory_charge = now.memory.place_charge_value.clamp(0.0, 1.0);
        let charger_near = input.body_features.get(6).copied().unwrap_or(0.0);
        let charger_visible = input.body_features.get(7).copied().unwrap_or(0.0);
        let ir_bright = input.body_features.get(9).copied().unwrap_or(0.0);

        let mut charge_probability = 0.05_f32;
        let mut expected_battery_delta = -0.002_f32;
        let mut dock_likelihood = 0.05_f32;
        let mut confidence = 0.55_f32;

        if already_charging {
            charge_probability = 0.95;
            expected_battery_delta = 0.03;
            dock_likelihood = 0.9;
            confidence = 0.95;
        }
        if memory_charge >= 0.7 {
            charge_probability = charge_probability.max(0.65);
            dock_likelihood = dock_likelihood.max(0.65);
            confidence = confidence.max(0.7);
        } else if memory_charge >= 0.4 {
            charge_probability = charge_probability.max(0.45);
            dock_likelihood = dock_likelihood.max(0.45);
        }
        if dock_action && low_battery {
            charge_probability = charge_probability.max(0.55);
            expected_battery_delta = expected_battery_delta.max(0.015);
            dock_likelihood = dock_likelihood.max(0.7);
            confidence = confidence.max(0.7);
        }
        if charger_near >= 0.5 || charger_visible >= 0.5 {
            charge_probability = charge_probability.max(0.85);
            expected_battery_delta = expected_battery_delta.max(0.025);
            dock_likelihood = dock_likelihood.max(0.9);
            confidence = confidence.max(0.85);
        } else if ir_bright >= 0.7 {
            charge_probability = charge_probability.max(0.35);
            dock_likelihood = dock_likelihood.max(0.35);
        }

        ChargeOutput {
            charge_probability: charge_probability.clamp(0.0, 1.0),
            expected_battery_delta: expected_battery_delta.clamp(-1.0, 1.0),
            dock_likelihood: dock_likelihood.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

impl NeuralModel<(ChargeInput, Now), ChargeOutput> for HardcodedChargePredictor {
    fn predict(&self, input: (ChargeInput, Now)) -> Result<ChargeOutput> {
        Ok(self.predict_from_now(&input.1, &input.0))
    }
}

#[derive(Clone, Debug, Default)]
pub struct HardcodedActionValuePredictor;

impl HardcodedActionValuePredictor {
    pub fn predict_from_now(&self, now: &Now, input: &ActionValueInput) -> ActionValueOutput {
        let battery_low = now.body.battery_level < 0.35;
        let dock_action = input.action_features.get(6).copied().unwrap_or(0.0) > 0.5;
        let go_action = input.action_features.get(2).copied().unwrap_or(0.0) > 0.5;
        let turn_action = input.action_features.get(3).copied().unwrap_or(0.0) > 0.5;
        let explore_action = input.action_features.get(4).copied().unwrap_or(0.0) > 0.5;
        let approach_action = input.action_features.get(5).copied().unwrap_or(0.0) > 0.5;
        let stop_action = input.action_features.get(1).copied().unwrap_or(0.0) > 0.5;
        let forward = input
            .action_features
            .get(10)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        let bump_risk = input.prediction_features.first().copied().unwrap_or(0.0);
        let cliff_risk = input.prediction_features.get(1).copied().unwrap_or(0.0);
        let wheel_drop_risk = input.prediction_features.get(2).copied().unwrap_or(0.0);
        let stuck_risk = input.prediction_features.get(3).copied().unwrap_or(0.0);
        let danger_confidence = input.prediction_features.get(4).copied().unwrap_or(0.0);
        let charge_probability = input.prediction_features.get(5).copied().unwrap_or(0.0);
        let expected_battery_delta = input.prediction_features.get(6).copied().unwrap_or(0.0);
        let dock_likelihood = input.prediction_features.get(7).copied().unwrap_or(0.0);
        let prediction_uncertainty = input.prediction_features.get(9).copied().unwrap_or(0.0);
        let safety_veto_likely = input.prediction_features.get(11).copied().unwrap_or(0.0);
        let memory_danger = input.memory_features.get(1).copied().unwrap_or(0.0);
        let memory_charge = input.memory_features.get(2).copied().unwrap_or(0.0);
        let remembered_best = input.memory_features.get(6).copied().unwrap_or(0.0);
        let has_warning = input.memory_features.get(7).copied().unwrap_or(0.0);

        let mut value = 0.0_f32;
        value += charge_probability * if battery_low { 0.75 } else { 0.25 };
        value += expected_battery_delta * if battery_low { 0.85 } else { 0.35 };
        value += dock_likelihood * if dock_action { 0.35 } else { 0.12 };
        if dock_action && (battery_low || memory_charge > 0.5) {
            value += 0.25;
        }
        if (explore_action || turn_action) && !battery_low {
            value += 0.12 + prediction_uncertainty * 0.08;
        }
        if approach_action && memory_charge > 0.5 {
            value += 0.12;
        }
        if remembered_best > 0.5 {
            value += 0.12;
        }
        if battery_low && (go_action || explore_action) && charge_probability < 0.25 {
            value -= 0.35;
        }
        value -= bump_risk * (0.45 + forward * 0.25);
        value -= cliff_risk * 0.9;
        value -= wheel_drop_risk * 0.95;
        value -= stuck_risk * 0.25;
        value -= memory_danger * 0.2;
        value -= has_warning * 0.15;
        value -= safety_veto_likely * 0.55;
        if stop_action && (cliff_risk > 0.5 || wheel_drop_risk > 0.5 || safety_veto_likely > 0.5) {
            value += 0.18;
        }

        ActionValueOutput {
            value: value.clamp(-1.0, 1.0),
            confidence: (0.45 + danger_confidence * 0.25 + charge_probability * 0.1)
                .clamp(0.0, 1.0),
        }
    }
}

impl NeuralModel<(ActionValueInput, Now), ActionValueOutput> for HardcodedActionValuePredictor {
    fn predict(&self, input: (ActionValueInput, Now)) -> Result<ActionValueOutput> {
        Ok(self.predict_from_now(&input.1, &input.0))
    }
}

#[derive(Clone, Debug, Default)]
pub struct CopyCurrentEyePredictor;

impl CopyCurrentEyePredictor {
    pub fn predict_from_now(&self, now: &Now, _input: &EyeNextInput) -> EyeNextOutput {
        match netherwick_experience::eye_frame_rgb(now) {
            Some((width, height, rgb)) => EyeNextOutput {
                width,
                height,
                rgb,
                confidence: 0.75,
            },
            None => EyeNextOutput {
                width: EYE_NEXT_WIDTH,
                height: EYE_NEXT_HEIGHT,
                rgb: vec![0; EYE_NEXT_RGB_LEN],
                confidence: 0.0,
            },
        }
    }
}

impl NeuralModel<(EyeNextInput, Now), EyeNextOutput> for CopyCurrentEyePredictor {
    fn predict(&self, input: (EyeNextInput, Now)) -> Result<EyeNextOutput> {
        Ok(self.predict_from_now(&input.1, &input.0))
    }
}

#[derive(Clone, Debug, Default)]
pub struct CopyCurrentEarPredictor;

impl CopyCurrentEarPredictor {
    pub fn predict_from_now(&self, now: &Now, input: &EarNextInput) -> EarNextOutput {
        let Some(features) = netherwick_experience::ear_frame_features(now) else {
            return EarNextOutput {
                sample_rate_hz: 0,
                channels: 0,
                pcm: Vec::new(),
                features: vec![0.0; input.ear_features.len()],
                confidence: 0.0,
            };
        };
        EarNextOutput {
            sample_rate_hz: 0,
            channels: 0,
            pcm: Vec::new(),
            features,
            confidence: 0.75,
        }
    }
}

impl NeuralModel<(EarNextInput, Now), EarNextOutput> for CopyCurrentEarPredictor {
    fn predict(&self, input: (EarNextInput, Now)) -> Result<EarNextOutput> {
        Ok(self.predict_from_now(&input.1, &input.0))
    }
}

#[derive(Module, Debug)]
pub struct DangerNet<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
}

#[derive(Module, Debug)]
pub struct ChargeNet<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
}

#[derive(Module, Debug)]
pub struct ActionValueNet<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
}

#[derive(Module, Debug)]
pub struct EyeNextNet<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
}

#[derive(Module, Debug)]
pub struct EarNextNet<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
}

#[derive(Module, Debug)]
pub struct ExperienceAutoencoderNet<B: Backend> {
    encoder_input: Linear<B>,
    encoder_hidden: Linear<B>,
    z: Linear<B>,
    decoder_hidden: Linear<B>,
    decoder_output: Linear<B>,
}

impl<B: Backend> ChargeNet<B> {
    pub fn init(input_dim: usize, device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(input_dim, 32).init(device),
            hidden: LinearConfig::new(32, 16).init(device),
            output: LinearConfig::new(16, 3).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.input.forward(input));
        let x = activation::relu(self.hidden.forward(x));
        activation::sigmoid(self.output.forward(x))
    }
}

impl<B: Backend> DangerNet<B> {
    pub fn init(input_dim: usize, device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(input_dim, 32).init(device),
            hidden: LinearConfig::new(32, 16).init(device),
            output: LinearConfig::new(16, 4).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.input.forward(input));
        let x = activation::relu(self.hidden.forward(x));
        activation::sigmoid(self.output.forward(x))
    }
}

impl<B: Backend> ActionValueNet<B> {
    pub fn init(input_dim: usize, device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(input_dim, 64).init(device),
            hidden: LinearConfig::new(64, 32).init(device),
            output: LinearConfig::new(32, 2).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.input.forward(input));
        let x = activation::relu(self.hidden.forward(x));
        activation::sigmoid(self.output.forward(x))
    }
}

impl<B: Backend> EyeNextNet<B> {
    pub fn init(input_dim: usize, output_dim: usize, device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(input_dim, 128).init(device),
            hidden: LinearConfig::new(128, 128).init(device),
            output: LinearConfig::new(128, output_dim).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.input.forward(input));
        let x = activation::relu(self.hidden.forward(x));
        activation::sigmoid(self.output.forward(x))
    }
}

impl<B: Backend> EarNextNet<B> {
    pub fn init(input_dim: usize, output_dim: usize, device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(input_dim, 64).init(device),
            hidden: LinearConfig::new(64, 32).init(device),
            output: LinearConfig::new(32, output_dim).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.input.forward(input));
        let x = activation::relu(self.hidden.forward(x));
        activation::sigmoid(self.output.forward(x))
    }
}

impl<B: Backend> ExperienceAutoencoderNet<B> {
    pub fn init(input_dim: usize, z_dim: usize, output_dim: usize, device: &B::Device) -> Self {
        Self {
            encoder_input: LinearConfig::new(input_dim, 96).init(device),
            encoder_hidden: LinearConfig::new(96, 48).init(device),
            z: LinearConfig::new(48, z_dim).init(device),
            decoder_hidden: LinearConfig::new(z_dim, 48).init(device),
            decoder_output: LinearConfig::new(48, output_dim).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = activation::relu(self.encoder_input.forward(input));
        let x = activation::relu(self.encoder_hidden.forward(x));
        let z = activation::sigmoid(self.z.forward(x));
        let decoded = activation::relu(self.decoder_hidden.forward(z.clone()));
        let decoded = activation::sigmoid(self.decoder_output.forward(decoded));
        (z, decoded)
    }
}

pub type DangerBackend = NdArray<f32>;
pub type DangerAutodiffBackend = Autodiff<DangerBackend>;
pub type ChargeBackend = NdArray<f32>;
pub type ChargeAutodiffBackend = Autodiff<ChargeBackend>;
pub type ActionValueBackend = NdArray<f32>;
pub type ActionValueAutodiffBackend = Autodiff<ActionValueBackend>;
pub type EyeNextBackend = NdArray<f32>;
pub type EyeNextAutodiffBackend = Autodiff<EyeNextBackend>;
pub type EarNextBackend = NdArray<f32>;
pub type EarNextAutodiffBackend = Autodiff<EarNextBackend>;
pub type ExperienceAutoencoderBackend = NdArray<f32>;
pub type ExperienceAutoencoderAutodiffBackend = Autodiff<ExperienceAutoencoderBackend>;

pub struct DangerNetTrainer<B: AutodiffBackend = DangerAutodiffBackend> {
    model: DangerNet<B>,
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, DangerNet<B>, B>,
    device: B::Device,
    input_dim: usize,
    learning_rate: f64,
    samples_seen: u64,
    best_loss: Option<f32>,
}

pub struct ChargeNetTrainer<B: AutodiffBackend = ChargeAutodiffBackend> {
    model: ChargeNet<B>,
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, ChargeNet<B>, B>,
    device: B::Device,
    input_dim: usize,
    learning_rate: f64,
    samples_seen: u64,
    best_loss: Option<f32>,
}

pub struct ActionValueNetTrainer<B: AutodiffBackend = ActionValueAutodiffBackend> {
    model: ActionValueNet<B>,
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, ActionValueNet<B>, B>,
    device: B::Device,
    input_dim: usize,
    learning_rate: f64,
    samples_seen: u64,
    best_loss: Option<f32>,
}

pub struct EyeNextNetTrainer<B: AutodiffBackend = EyeNextAutodiffBackend> {
    model: EyeNextNet<B>,
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, EyeNextNet<B>, B>,
    device: B::Device,
    input_dim: usize,
    output_dim: usize,
    width: u32,
    height: u32,
    learning_rate: f64,
    samples_seen: u64,
    best_loss: Option<f32>,
}

pub struct EarNextNetTrainer<B: AutodiffBackend = EarNextAutodiffBackend> {
    model: EarNextNet<B>,
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, EarNextNet<B>, B>,
    device: B::Device,
    input_dim: usize,
    output_dim: usize,
    sample_rate_hz: u32,
    channels: u16,
    learning_rate: f64,
    samples_seen: u64,
    best_loss: Option<f32>,
}

pub struct ExperienceAutoencoderTrainer<B: AutodiffBackend = ExperienceAutoencoderAutodiffBackend> {
    model: ExperienceAutoencoderNet<B>,
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, ExperienceAutoencoderNet<B>, B>,
    device: B::Device,
    input_dim: usize,
    z_dim: usize,
    output_dim: usize,
    decode_lengths: ExperienceDecodeFeatureLengths,
    learning_rate: f64,
    samples_seen: u64,
    best_loss: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DangerModelMetadata {
    pub input_dim: usize,
    pub samples_seen: u64,
    pub best_loss: Option<f32>,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChargeModelMetadata {
    pub input_dim: usize,
    pub samples_seen: u64,
    pub best_loss: Option<f32>,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionValueModelMetadata {
    pub input_dim: usize,
    pub samples_seen: u64,
    pub best_loss: Option<f32>,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EyeNextModelMetadata {
    pub input_dim: usize,
    pub output_dim: usize,
    pub width: u32,
    pub height: u32,
    pub samples_seen: u64,
    pub best_loss: Option<f32>,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EarNextModelMetadata {
    pub input_dim: usize,
    pub output_dim: usize,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples_seen: u64,
    pub best_loss: Option<f32>,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceAutoencoderMetadata {
    pub input_dim: usize,
    pub z_dim: usize,
    pub output_dim: usize,
    pub decode_lengths: ExperienceDecodeFeatureLengths,
    pub samples_seen: u64,
    pub best_loss: Option<f32>,
    pub created_at_ms: u64,
}

impl DangerNetTrainer<DangerAutodiffBackend> {
    pub fn new(input_dim: usize) -> Self {
        Self::with_device(input_dim, Default::default())
    }

    pub fn load_checkpoint(path: impl AsRef<Path>, input_dim: usize) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_danger_metadata(path)?;
        if metadata.input_dim != input_dim {
            return Err(anyhow!(
                "danger checkpoint input dimension mismatch at {}: metadata has {}, runtime expected {}",
                path.display(),
                metadata.input_dim,
                input_dim
            ));
        }

        let device = Default::default();
        let model = DangerNet::init(input_dim, &device).load_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
            &device,
        )?;
        Ok(Self {
            model,
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            learning_rate: 0.03,
            samples_seen: metadata.samples_seen,
            best_loss: metadata.best_loss,
        })
    }
}

impl<B: AutodiffBackend> DangerNetTrainer<B> {
    pub fn with_device(input_dim: usize, device: B::Device) -> Self {
        Self {
            model: DangerNet::init(input_dim, &device),
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            learning_rate: 0.03,
            samples_seen: 0,
            best_loss: None,
        }
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn samples_seen(&self) -> u64 {
        self.samples_seen
    }

    pub fn best_loss(&self) -> Option<f32> {
        self.best_loss
    }

    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)
            .with_context(|| format!("create danger checkpoint dir {}", path.display()))?;
        self.model.clone().save_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
        )?;
        let metadata = DangerModelMetadata {
            input_dim: self.input_dim,
            samples_seen: self.samples_seen,
            best_loss: self.best_loss,
            created_at_ms: now_ms(),
        };
        std::fs::write(
            path.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )
        .with_context(|| format!("write danger checkpoint metadata {}", path.display()))?;
        Ok(())
    }

    pub fn predict(&self, input: &DangerInput) -> Result<DangerOutput> {
        let features = self.checked_features(input)?;
        let tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let output = self.model.forward(tensor).inner();
        tensor_to_danger_output(output)
    }

    pub fn train_step(
        &mut self,
        input: &DangerInput,
        target: &DangerTarget,
    ) -> Result<DangerTrainStats> {
        let features = self.checked_features(input)?;
        let target_values = target.risks();
        let input_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let target_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(target_values.to_vec(), [1, 4]),
            &self.device,
        );
        let output = self.model.forward(input_tensor);
        let loss = MseLoss::new().forward(output, target_tensor, Reduction::Mean);
        let loss_value = loss.clone().inner().into_data().to_vec::<f32>()?[0];
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = self
            .optimizer
            .step(self.learning_rate, self.model.clone(), grads);
        self.samples_seen = self.samples_seen.saturating_add(1);
        let improved = self.best_loss.map(|best| loss_value < best).unwrap_or(true);
        if improved {
            self.best_loss = Some(loss_value);
        }
        Ok(DangerTrainStats {
            loss: loss_value,
            samples_seen: self.samples_seen,
            improved,
        })
    }

    pub fn shadow_compare(
        &mut self,
        observed_at_ms: u64,
        now: &Now,
        input: &DangerInput,
        target: &DangerTarget,
    ) -> Result<DangerShadowMetric> {
        let hardcoded = HardcodedDangerPredictor.predict_from_now(now, input);
        let model = self.predict(input)?;
        let loss = mse_output_target(model, *target);
        Ok(DangerShadowMetric {
            observed_at_ms,
            hardcoded,
            model,
            target: *target,
            loss,
        })
    }

    fn checked_features(&self, input: &DangerInput) -> Result<Vec<f32>> {
        let mut features = input.flat_features();
        if features.len() != self.input_dim {
            return Err(anyhow!(
                "danger input dimension mismatch: got {}, expected {}",
                features.len(),
                self.input_dim
            ));
        }
        for value in &mut features {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
        Ok(features)
    }
}

impl ChargeNetTrainer<ChargeAutodiffBackend> {
    pub fn new(input_dim: usize) -> Self {
        Self::with_device(input_dim, Default::default())
    }

    pub fn load_checkpoint(path: impl AsRef<Path>, input_dim: usize) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_charge_metadata(path)?;
        if metadata.input_dim != input_dim {
            return Err(anyhow!(
                "charge checkpoint input dimension mismatch at {}: metadata has {}, runtime expected {}",
                path.display(),
                metadata.input_dim,
                input_dim
            ));
        }

        let device = Default::default();
        let model = ChargeNet::init(input_dim, &device).load_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
            &device,
        )?;
        Ok(Self {
            model,
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            learning_rate: 0.03,
            samples_seen: metadata.samples_seen,
            best_loss: metadata.best_loss,
        })
    }
}

impl<B: AutodiffBackend> ChargeNetTrainer<B> {
    pub fn with_device(input_dim: usize, device: B::Device) -> Self {
        Self {
            model: ChargeNet::init(input_dim, &device),
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            learning_rate: 0.03,
            samples_seen: 0,
            best_loss: None,
        }
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn samples_seen(&self) -> u64 {
        self.samples_seen
    }

    pub fn best_loss(&self) -> Option<f32> {
        self.best_loss
    }

    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)
            .with_context(|| format!("create charge checkpoint dir {}", path.display()))?;
        self.model.clone().save_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
        )?;
        let metadata = ChargeModelMetadata {
            input_dim: self.input_dim,
            samples_seen: self.samples_seen,
            best_loss: self.best_loss,
            created_at_ms: now_ms(),
        };
        std::fs::write(
            path.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )
        .with_context(|| format!("write charge checkpoint metadata {}", path.display()))?;
        Ok(())
    }

    pub fn predict(&self, input: &ChargeInput) -> Result<ChargeOutput> {
        let features = self.checked_features(input)?;
        let tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let output = self.model.forward(tensor).inner();
        tensor_to_charge_output(output)
    }

    pub fn train_step(
        &mut self,
        input: &ChargeInput,
        target: &ChargeTarget,
    ) -> Result<ChargeTrainStats> {
        let features = self.checked_features(input)?;
        let target_values = charge_target_train_values(target);
        let input_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let target_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(target_values.to_vec(), [1, 3]),
            &self.device,
        );
        let output = self.model.forward(input_tensor);
        let loss = MseLoss::new().forward(output, target_tensor, Reduction::Mean);
        let loss_value = loss.clone().inner().into_data().to_vec::<f32>()?[0];
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = self
            .optimizer
            .step(self.learning_rate, self.model.clone(), grads);
        self.samples_seen = self.samples_seen.saturating_add(1);
        let improved = self.best_loss.map(|best| loss_value < best).unwrap_or(true);
        if improved {
            self.best_loss = Some(loss_value);
        }
        Ok(ChargeTrainStats {
            loss: loss_value,
            samples_seen: self.samples_seen,
            improved,
        })
    }

    pub fn shadow_compare(
        &mut self,
        observed_at_ms: u64,
        now: &Now,
        input: &ChargeInput,
        target: &ChargeTarget,
    ) -> Result<ChargeShadowMetric> {
        let hardcoded = HardcodedChargePredictor.predict_from_now(now, input);
        let model = self.predict(input)?;
        let loss = mse_charge_output_target(model, *target);
        Ok(ChargeShadowMetric {
            observed_at_ms,
            hardcoded,
            model,
            target: *target,
            loss,
        })
    }

    fn checked_features(&self, input: &ChargeInput) -> Result<Vec<f32>> {
        let mut features = input.flat_features();
        if features.len() != self.input_dim {
            return Err(anyhow!(
                "charge input dimension mismatch: got {}, expected {}",
                features.len(),
                self.input_dim
            ));
        }
        for value in &mut features {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
        Ok(features)
    }
}

impl ActionValueNetTrainer<ActionValueAutodiffBackend> {
    pub fn new(input_dim: usize) -> Self {
        Self::with_device(input_dim, Default::default())
    }

    pub fn load_checkpoint(path: impl AsRef<Path>, input_dim: usize) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_action_value_metadata(path)?;
        if metadata.input_dim != input_dim {
            return Err(anyhow!(
                "action-value checkpoint input dimension mismatch at {}: metadata has {}, runtime expected {}",
                path.display(),
                metadata.input_dim,
                input_dim
            ));
        }

        let device = Default::default();
        let model = ActionValueNet::init(input_dim, &device).load_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
            &device,
        )?;
        Ok(Self {
            model,
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            learning_rate: 0.03,
            samples_seen: metadata.samples_seen,
            best_loss: metadata.best_loss,
        })
    }
}

impl<B: AutodiffBackend> ActionValueNetTrainer<B> {
    pub fn with_device(input_dim: usize, device: B::Device) -> Self {
        Self {
            model: ActionValueNet::init(input_dim, &device),
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            learning_rate: 0.03,
            samples_seen: 0,
            best_loss: None,
        }
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn samples_seen(&self) -> u64 {
        self.samples_seen
    }

    pub fn best_loss(&self) -> Option<f32> {
        self.best_loss
    }

    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)
            .with_context(|| format!("create action-value checkpoint dir {}", path.display()))?;
        self.model.clone().save_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
        )?;
        let metadata = ActionValueModelMetadata {
            input_dim: self.input_dim,
            samples_seen: self.samples_seen,
            best_loss: self.best_loss,
            created_at_ms: now_ms(),
        };
        std::fs::write(
            path.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )
        .with_context(|| format!("write action-value checkpoint metadata {}", path.display()))?;
        Ok(())
    }

    pub fn predict(&self, input: &ActionValueInput) -> Result<ActionValueOutput> {
        let features = self.checked_features(input)?;
        let tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let output = self.model.forward(tensor).inner();
        tensor_to_action_value_output(output)
    }

    pub fn train_step(
        &mut self,
        input: &ActionValueInput,
        target: &ActionValueTarget,
    ) -> Result<ActionValueTrainStats> {
        let features = self.checked_features(input)?;
        let target_values = action_value_target_train_values(target);
        let input_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let target_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(target_values.to_vec(), [1, 2]),
            &self.device,
        );
        let output = self.model.forward(input_tensor);
        let loss = MseLoss::new().forward(output, target_tensor, Reduction::Mean);
        let loss_value = loss.clone().inner().into_data().to_vec::<f32>()?[0];
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = self
            .optimizer
            .step(self.learning_rate, self.model.clone(), grads);
        self.samples_seen = self.samples_seen.saturating_add(1);
        let improved = self.best_loss.map(|best| loss_value < best).unwrap_or(true);
        if improved {
            self.best_loss = Some(loss_value);
        }
        Ok(ActionValueTrainStats {
            loss: loss_value,
            samples_seen: self.samples_seen,
            improved,
        })
    }

    pub fn shadow_compare(
        &mut self,
        observed_at_ms: u64,
        now: &Now,
        input: &ActionValueInput,
        target: &ActionValueTarget,
    ) -> Result<ActionValueShadowMetric> {
        let hardcoded = HardcodedActionValuePredictor.predict_from_now(now, input);
        let model = self.predict(input)?;
        let loss = mse_action_value_output_target(model, *target);
        Ok(ActionValueShadowMetric {
            observed_at_ms,
            hardcoded,
            model,
            target: *target,
            loss,
        })
    }

    fn checked_features(&self, input: &ActionValueInput) -> Result<Vec<f32>> {
        let mut features = input.flat_features();
        if features.len() != self.input_dim {
            return Err(anyhow!(
                "action-value input dimension mismatch: got {}, expected {}",
                features.len(),
                self.input_dim
            ));
        }
        for value in &mut features {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
        Ok(features)
    }
}

impl EyeNextNetTrainer<EyeNextAutodiffBackend> {
    pub fn new(input_dim: usize, width: u32, height: u32) -> Self {
        Self::with_device(input_dim, width, height, Default::default())
    }

    pub fn load_checkpoint(path: impl AsRef<Path>, input_dim: usize) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_eye_next_metadata(path)?;
        if metadata.input_dim != input_dim {
            return Err(anyhow!(
                "eye-next checkpoint input dimension mismatch at {}: metadata has {}, runtime expected {}",
                path.display(),
                metadata.input_dim,
                input_dim
            ));
        }

        let device = Default::default();
        let model = EyeNextNet::init(input_dim, metadata.output_dim, &device).load_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
            &device,
        )?;
        Ok(Self {
            model,
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            output_dim: metadata.output_dim,
            width: metadata.width,
            height: metadata.height,
            learning_rate: 0.01,
            samples_seen: metadata.samples_seen,
            best_loss: metadata.best_loss,
        })
    }
}

impl<B: AutodiffBackend> EyeNextNetTrainer<B> {
    pub fn with_device(input_dim: usize, width: u32, height: u32, device: B::Device) -> Self {
        let output_dim = width as usize * height as usize * 3;
        Self {
            model: EyeNextNet::init(input_dim, output_dim, &device),
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            output_dim,
            width,
            height,
            learning_rate: 0.01,
            samples_seen: 0,
            best_loss: None,
        }
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn samples_seen(&self) -> u64 {
        self.samples_seen
    }

    pub fn best_loss(&self) -> Option<f32> {
        self.best_loss
    }

    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)
            .with_context(|| format!("create eye-next checkpoint dir {}", path.display()))?;
        self.model.clone().save_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
        )?;
        let metadata = EyeNextModelMetadata {
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            width: self.width,
            height: self.height,
            samples_seen: self.samples_seen,
            best_loss: self.best_loss,
            created_at_ms: now_ms(),
        };
        std::fs::write(
            path.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )
        .with_context(|| format!("write eye-next checkpoint metadata {}", path.display()))?;
        Ok(())
    }

    pub fn predict(&self, input: &EyeNextInput) -> Result<EyeNextOutput> {
        let features = self.checked_features(input)?;
        let tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let output = self.model.forward(tensor).inner();
        tensor_to_eye_next_output(output, self.width, self.height)
    }

    pub fn train_step(
        &mut self,
        input: &EyeNextInput,
        target: &EyeNextTarget,
    ) -> Result<EyeNextTrainStats> {
        let features = self.checked_features(input)?;
        let target_values = eye_target_train_values(target, self.output_dim);
        let input_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let target_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(target_values, [1, self.output_dim]),
            &self.device,
        );
        let output = self.model.forward(input_tensor);
        let loss = MseLoss::new().forward(output, target_tensor, Reduction::Mean);
        let loss_value = loss.clone().inner().into_data().to_vec::<f32>()?[0];
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = self
            .optimizer
            .step(self.learning_rate, self.model.clone(), grads);
        self.samples_seen = self.samples_seen.saturating_add(1);
        let improved = self.best_loss.map(|best| loss_value < best).unwrap_or(true);
        if improved {
            self.best_loss = Some(loss_value);
        }
        Ok(EyeNextTrainStats {
            loss: loss_value,
            samples_seen: self.samples_seen,
            improved,
        })
    }

    pub fn shadow_compare(
        &mut self,
        observed_at_ms: u64,
        now: &Now,
        input: &EyeNextInput,
        target: &EyeNextTarget,
    ) -> Result<EyeNextShadowMetric> {
        let hardcoded = CopyCurrentEyePredictor.predict_from_now(now, input);
        let model = self.predict(input)?;
        let loss = mse_eye_next_output_target(&model, target);
        Ok(EyeNextShadowMetric {
            observed_at_ms,
            hardcoded,
            model,
            target: target.clone(),
            loss,
        })
    }

    fn checked_features(&self, input: &EyeNextInput) -> Result<Vec<f32>> {
        let mut features = input.flat_features();
        if features.len() != self.input_dim {
            return Err(anyhow!(
                "eye-next input dimension mismatch: got {}, expected {}",
                features.len(),
                self.input_dim
            ));
        }
        for value in &mut features {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
        Ok(features)
    }
}

impl EarNextNetTrainer<EarNextAutodiffBackend> {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self::with_device(input_dim, output_dim, 0, 0, Default::default())
    }

    pub fn with_audio_shape(
        input_dim: usize,
        output_dim: usize,
        sample_rate_hz: u32,
        channels: u16,
    ) -> Self {
        Self::with_device(
            input_dim,
            output_dim,
            sample_rate_hz,
            channels,
            Default::default(),
        )
    }

    pub fn load_checkpoint(path: impl AsRef<Path>, input_dim: usize) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_ear_next_metadata(path)?;
        if metadata.input_dim != input_dim {
            return Err(anyhow!(
                "ear-next checkpoint input dimension mismatch at {}: metadata has {}, runtime expected {}",
                path.display(),
                metadata.input_dim,
                input_dim
            ));
        }

        let device = Default::default();
        let model = EarNextNet::init(input_dim, metadata.output_dim, &device).load_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
            &device,
        )?;
        Ok(Self {
            model,
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            output_dim: metadata.output_dim,
            sample_rate_hz: metadata.sample_rate_hz,
            channels: metadata.channels,
            learning_rate: 0.01,
            samples_seen: metadata.samples_seen,
            best_loss: metadata.best_loss,
        })
    }
}

impl<B: AutodiffBackend> EarNextNetTrainer<B> {
    pub fn with_device(
        input_dim: usize,
        output_dim: usize,
        sample_rate_hz: u32,
        channels: u16,
        device: B::Device,
    ) -> Self {
        Self {
            model: EarNextNet::init(input_dim, output_dim, &device),
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            output_dim,
            sample_rate_hz,
            channels,
            learning_rate: 0.01,
            samples_seen: 0,
            best_loss: None,
        }
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn samples_seen(&self) -> u64 {
        self.samples_seen
    }

    pub fn best_loss(&self) -> Option<f32> {
        self.best_loss
    }

    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)
            .with_context(|| format!("create ear-next checkpoint dir {}", path.display()))?;
        self.model.clone().save_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
        )?;
        let metadata = EarNextModelMetadata {
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            sample_rate_hz: self.sample_rate_hz,
            channels: self.channels,
            samples_seen: self.samples_seen,
            best_loss: self.best_loss,
            created_at_ms: now_ms(),
        };
        std::fs::write(
            path.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )
        .with_context(|| format!("write ear-next checkpoint metadata {}", path.display()))?;
        Ok(())
    }

    pub fn predict(&self, input: &EarNextInput) -> Result<EarNextOutput> {
        let features = self.checked_features(input)?;
        let tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let output = self.model.forward(tensor).inner();
        tensor_to_ear_next_output(output, self.output_dim, self.sample_rate_hz, self.channels)
    }

    pub fn train_step(
        &mut self,
        input: &EarNextInput,
        target: &EarNextTarget,
    ) -> Result<EarNextTrainStats> {
        let features = self.checked_features(input)?;
        let target_values = ear_target_train_values(target, self.output_dim);
        let input_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let target_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(target_values, [1, self.output_dim]),
            &self.device,
        );
        let output = self.model.forward(input_tensor);
        let loss = MseLoss::new().forward(output, target_tensor, Reduction::Mean);
        let loss_value = loss.clone().inner().into_data().to_vec::<f32>()?[0];
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = self
            .optimizer
            .step(self.learning_rate, self.model.clone(), grads);
        self.samples_seen = self.samples_seen.saturating_add(1);
        let improved = self.best_loss.map(|best| loss_value < best).unwrap_or(true);
        if improved {
            self.best_loss = Some(loss_value);
        }
        Ok(EarNextTrainStats {
            loss: loss_value,
            samples_seen: self.samples_seen,
            improved,
        })
    }

    pub fn shadow_compare(
        &mut self,
        observed_at_ms: u64,
        now: &Now,
        input: &EarNextInput,
        target: &EarNextTarget,
    ) -> Result<EarNextShadowMetric> {
        let hardcoded = CopyCurrentEarPredictor.predict_from_now(now, input);
        let model = self.predict(input)?;
        let loss = mse_ear_next_output_target(&model, target);
        Ok(EarNextShadowMetric {
            observed_at_ms,
            hardcoded,
            model,
            target: target.clone(),
            loss,
        })
    }

    fn checked_features(&self, input: &EarNextInput) -> Result<Vec<f32>> {
        let mut features = input.flat_features();
        if features.len() != self.input_dim {
            return Err(anyhow!(
                "ear-next input dimension mismatch: got {}, expected {}",
                features.len(),
                self.input_dim
            ));
        }
        for value in &mut features {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
        Ok(features)
    }
}

impl ExperienceAutoencoderTrainer<ExperienceAutoencoderAutodiffBackend> {
    pub fn new(
        input_dim: usize,
        z_dim: usize,
        decode_lengths: ExperienceDecodeFeatureLengths,
    ) -> Self {
        Self::with_device(input_dim, z_dim, decode_lengths, Default::default())
    }

    pub fn load_checkpoint(path: impl AsRef<Path>, input_dim: usize) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_experience_autoencoder_metadata(path)?;
        if metadata.input_dim != input_dim {
            return Err(anyhow!(
                "experience autoencoder checkpoint input dimension mismatch at {}: metadata has {}, runtime expected {}",
                path.display(),
                metadata.input_dim,
                input_dim
            ));
        }

        let device = Default::default();
        let model =
            ExperienceAutoencoderNet::init(input_dim, metadata.z_dim, metadata.output_dim, &device)
                .load_file(
                    path.join("model"),
                    &BinFileRecorder::<FullPrecisionSettings>::default(),
                    &device,
                )?;
        Ok(Self {
            model,
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            z_dim: metadata.z_dim,
            output_dim: metadata.output_dim,
            decode_lengths: metadata.decode_lengths,
            learning_rate: 0.01,
            samples_seen: metadata.samples_seen,
            best_loss: metadata.best_loss,
        })
    }
}

impl<B: AutodiffBackend> ExperienceAutoencoderTrainer<B> {
    pub fn with_device(
        input_dim: usize,
        z_dim: usize,
        decode_lengths: ExperienceDecodeFeatureLengths,
        device: B::Device,
    ) -> Self {
        let output_dim = decode_lengths.body
            + decode_lengths.memory
            + decode_lengths.drive
            + decode_lengths.prediction
            + decode_lengths.eye
            + decode_lengths.ear;
        Self {
            model: ExperienceAutoencoderNet::init(input_dim, z_dim, output_dim, &device),
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            z_dim,
            output_dim,
            decode_lengths,
            learning_rate: 0.01,
            samples_seen: 0,
            best_loss: None,
        }
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn z_dim(&self) -> usize {
        self.z_dim
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn samples_seen(&self) -> u64 {
        self.samples_seen
    }

    pub fn best_loss(&self) -> Option<f32> {
        self.best_loss
    }

    pub fn decode_lengths(&self) -> ExperienceDecodeFeatureLengths {
        self.decode_lengths
    }

    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::create_dir_all(path).with_context(|| {
            format!(
                "create experience autoencoder checkpoint dir {}",
                path.display()
            )
        })?;
        self.model.clone().save_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
        )?;
        let metadata = ExperienceAutoencoderMetadata {
            input_dim: self.input_dim,
            z_dim: self.z_dim,
            output_dim: self.output_dim,
            decode_lengths: self.decode_lengths,
            samples_seen: self.samples_seen,
            best_loss: self.best_loss,
            created_at_ms: now_ms(),
        };
        std::fs::write(
            path.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )
        .with_context(|| {
            format!(
                "write experience autoencoder checkpoint metadata {}",
                path.display()
            )
        })?;
        Ok(())
    }

    pub fn predict(
        &self,
        input: &ExperienceEncodeInput,
    ) -> Result<ExperienceAutoencoderPrediction> {
        let features = self.checked_features(input)?;
        let tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let (z, decoded) = self.model.forward(tensor);
        Ok(ExperienceAutoencoderPrediction {
            encoded: tensor_to_experience_encode_output(z.inner(), self.z_dim)?,
            decoded: tensor_to_experience_decode_output(
                decoded.inner(),
                self.output_dim,
                self.decode_lengths,
            )?,
        })
    }

    pub fn encode(&self, input: &ExperienceEncodeInput) -> Result<ExperienceEncodeOutput> {
        Ok(self.predict(input)?.encoded)
    }

    pub fn train_step(
        &mut self,
        input: &ExperienceEncodeInput,
        target: &ExperienceDecodeOutput,
    ) -> Result<ExperienceAutoencoderTrainStats> {
        let features = self.checked_features(input)?;
        let target_values = experience_decode_target_values(target, self.output_dim);
        let input_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let target_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(target_values, [1, self.output_dim]),
            &self.device,
        );
        let (_z, decoded) = self.model.forward(input_tensor);
        let loss = MseLoss::new().forward(decoded, target_tensor, Reduction::Mean);
        let loss_value = loss.clone().inner().into_data().to_vec::<f32>()?[0];
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = self
            .optimizer
            .step(self.learning_rate, self.model.clone(), grads);
        self.samples_seen = self.samples_seen.saturating_add(1);
        let improved = self.best_loss.map(|best| loss_value < best).unwrap_or(true);
        if improved {
            self.best_loss = Some(loss_value);
        }
        Ok(ExperienceAutoencoderTrainStats {
            loss: loss_value,
            samples_seen: self.samples_seen,
            improved,
        })
    }

    pub fn shadow_compare(
        &mut self,
        observed_at_ms: u64,
        input: &ExperienceEncodeInput,
        target: &ExperienceDecodeOutput,
    ) -> Result<ExperienceAutoencoderShadowMetric> {
        let prediction = self.predict(input)?;
        let loss = mse_experience_decode_output_target(&prediction.decoded, target);
        Ok(ExperienceAutoencoderShadowMetric {
            observed_at_ms,
            hardcoded: target.clone(),
            model: prediction.decoded,
            target: target.clone(),
            z: prediction.encoded,
            loss,
        })
    }

    fn checked_features(&self, input: &ExperienceEncodeInput) -> Result<Vec<f32>> {
        let mut features = input.flat_features();
        if features.len() != self.input_dim {
            return Err(anyhow!(
                "experience autoencoder input dimension mismatch: got {}, expected {}",
                features.len(),
                self.input_dim
            ));
        }
        for value in &mut features {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
        Ok(features)
    }
}

pub fn read_danger_metadata(path: impl AsRef<Path>) -> Result<DangerModelMetadata> {
    let path = path.as_ref();
    let bytes = std::fs::read(path.join("metadata.json"))
        .with_context(|| format!("read danger checkpoint metadata {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse danger checkpoint metadata {}", path.display()))
}

pub fn read_charge_metadata(path: impl AsRef<Path>) -> Result<ChargeModelMetadata> {
    let path = path.as_ref();
    let bytes = std::fs::read(path.join("metadata.json"))
        .with_context(|| format!("read charge checkpoint metadata {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse charge checkpoint metadata {}", path.display()))
}

pub fn read_action_value_metadata(path: impl AsRef<Path>) -> Result<ActionValueModelMetadata> {
    let path = path.as_ref();
    let bytes = std::fs::read(path.join("metadata.json"))
        .with_context(|| format!("read action-value checkpoint metadata {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse action-value checkpoint metadata {}", path.display()))
}

pub fn read_eye_next_metadata(path: impl AsRef<Path>) -> Result<EyeNextModelMetadata> {
    let path = path.as_ref();
    let bytes = std::fs::read(path.join("metadata.json"))
        .with_context(|| format!("read eye-next checkpoint metadata {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse eye-next checkpoint metadata {}", path.display()))
}

pub fn read_ear_next_metadata(path: impl AsRef<Path>) -> Result<EarNextModelMetadata> {
    let path = path.as_ref();
    let bytes = std::fs::read(path.join("metadata.json"))
        .with_context(|| format!("read ear-next checkpoint metadata {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse ear-next checkpoint metadata {}", path.display()))
}

pub fn read_experience_autoencoder_metadata(
    path: impl AsRef<Path>,
) -> Result<ExperienceAutoencoderMetadata> {
    let path = path.as_ref();
    let bytes = std::fs::read(path.join("metadata.json")).with_context(|| {
        format!(
            "read experience autoencoder checkpoint metadata {}",
            path.display()
        )
    })?;
    serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "parse experience autoencoder checkpoint metadata {}",
            path.display()
        )
    })
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

impl<B: AutodiffBackend> OnlineTrainer<DangerInput, DangerTarget> for DangerNetTrainer<B> {
    fn train_step(
        &mut self,
        sample: TrainingSample<DangerInput, DangerTarget>,
    ) -> Result<TrainStats> {
        let stats = DangerNetTrainer::train_step(self, &sample.input, &sample.expected)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }
}

impl<B: AutodiffBackend> OnlineTrainer<ChargeInput, ChargeTarget> for ChargeNetTrainer<B> {
    fn train_step(
        &mut self,
        sample: TrainingSample<ChargeInput, ChargeTarget>,
    ) -> Result<TrainStats> {
        let stats = ChargeNetTrainer::train_step(self, &sample.input, &sample.expected)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }
}

impl<B: AutodiffBackend> OnlineTrainer<ActionValueInput, ActionValueTarget>
    for ActionValueNetTrainer<B>
{
    fn train_step(
        &mut self,
        sample: TrainingSample<ActionValueInput, ActionValueTarget>,
    ) -> Result<TrainStats> {
        let stats = ActionValueNetTrainer::train_step(self, &sample.input, &sample.expected)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }
}

impl<B: AutodiffBackend> OnlineTrainer<EyeNextInput, EyeNextTarget> for EyeNextNetTrainer<B> {
    fn train_step(
        &mut self,
        sample: TrainingSample<EyeNextInput, EyeNextTarget>,
    ) -> Result<TrainStats> {
        let stats = EyeNextNetTrainer::train_step(self, &sample.input, &sample.expected)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }
}

impl<B: AutodiffBackend> OnlineTrainer<EarNextInput, EarNextTarget> for EarNextNetTrainer<B> {
    fn train_step(
        &mut self,
        sample: TrainingSample<EarNextInput, EarNextTarget>,
    ) -> Result<TrainStats> {
        let stats = EarNextNetTrainer::train_step(self, &sample.input, &sample.expected)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }
}

impl<B: AutodiffBackend> OnlineTrainer<ExperienceEncodeInput, ExperienceDecodeOutput>
    for ExperienceAutoencoderTrainer<B>
{
    fn train_step(
        &mut self,
        sample: TrainingSample<ExperienceEncodeInput, ExperienceDecodeOutput>,
    ) -> Result<TrainStats> {
        let stats =
            ExperienceAutoencoderTrainer::train_step(self, &sample.input, &sample.expected)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }
}

fn tensor_to_danger_output<B: Backend>(tensor: Tensor<B, 2>) -> Result<DangerOutput> {
    let values = tensor.into_data().to_vec::<f32>()?;
    if values.len() != 4 {
        return Err(anyhow!(
            "danger net emitted {} outputs, expected 4",
            values.len()
        ));
    }
    Ok(DangerOutput {
        bump_risk: values[0].clamp(0.0, 1.0),
        cliff_risk: values[1].clamp(0.0, 1.0),
        wheel_drop_risk: values[2].clamp(0.0, 1.0),
        stuck_risk: values[3].clamp(0.0, 1.0),
        confidence: 0.5,
    })
}

fn tensor_to_charge_output<B: Backend>(tensor: Tensor<B, 2>) -> Result<ChargeOutput> {
    let values = tensor.into_data().to_vec::<f32>()?;
    if values.len() != 3 {
        return Err(anyhow!(
            "charge net emitted {} outputs, expected 3",
            values.len()
        ));
    }
    Ok(ChargeOutput {
        charge_probability: values[0].clamp(0.0, 1.0),
        expected_battery_delta: (values[1] * 2.0 - 1.0).clamp(-1.0, 1.0),
        dock_likelihood: values[2].clamp(0.0, 1.0),
        confidence: 0.5,
    })
}

fn tensor_to_action_value_output<B: Backend>(tensor: Tensor<B, 2>) -> Result<ActionValueOutput> {
    let values = tensor.into_data().to_vec::<f32>()?;
    if values.len() != 2 {
        return Err(anyhow!(
            "action-value net emitted {} outputs, expected 2",
            values.len()
        ));
    }
    Ok(ActionValueOutput {
        value: (values[0] * 2.0 - 1.0).clamp(-1.0, 1.0),
        confidence: values[1].clamp(0.0, 1.0),
    })
}

fn tensor_to_eye_next_output<B: Backend>(
    tensor: Tensor<B, 2>,
    width: u32,
    height: u32,
) -> Result<EyeNextOutput> {
    let values = tensor.into_data().to_vec::<f32>()?;
    let expected_len = width as usize * height as usize * 3;
    if values.len() != expected_len {
        return Err(anyhow!(
            "eye-next net emitted {} outputs, expected {}",
            values.len(),
            expected_len
        ));
    }
    Ok(EyeNextOutput {
        width,
        height,
        rgb: values
            .into_iter()
            .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect(),
        confidence: 0.5,
    })
}

fn tensor_to_ear_next_output<B: Backend>(
    tensor: Tensor<B, 2>,
    output_dim: usize,
    sample_rate_hz: u32,
    channels: u16,
) -> Result<EarNextOutput> {
    let values = tensor.into_data().to_vec::<f32>()?;
    if values.len() != output_dim {
        return Err(anyhow!(
            "ear-next net emitted {} outputs, expected {}",
            values.len(),
            output_dim
        ));
    }
    Ok(EarNextOutput {
        sample_rate_hz,
        channels,
        pcm: Vec::new(),
        features: values
            .into_iter()
            .map(|value| value.clamp(0.0, 1.0))
            .collect(),
        confidence: 0.5,
    })
}

fn tensor_to_experience_encode_output<B: Backend>(
    tensor: Tensor<B, 2>,
    z_dim: usize,
) -> Result<ExperienceEncodeOutput> {
    let values = tensor.into_data().to_vec::<f32>()?;
    if values.len() != z_dim {
        return Err(anyhow!(
            "experience autoencoder emitted {} z outputs, expected {}",
            values.len(),
            z_dim
        ));
    }
    Ok(ExperienceEncodeOutput {
        z: values
            .into_iter()
            .map(|value| value.clamp(0.0, 1.0))
            .collect(),
        confidence: 0.5,
    })
}

fn tensor_to_experience_decode_output<B: Backend>(
    tensor: Tensor<B, 2>,
    output_dim: usize,
    lengths: ExperienceDecodeFeatureLengths,
) -> Result<ExperienceDecodeOutput> {
    let values = tensor.into_data().to_vec::<f32>()?;
    if values.len() != output_dim {
        return Err(anyhow!(
            "experience autoencoder emitted {} reconstruction outputs, expected {}",
            values.len(),
            output_dim
        ));
    }
    Ok(split_experience_decode_values(
        values
            .into_iter()
            .map(|value| value.clamp(0.0, 1.0))
            .collect(),
        lengths,
    ))
}

fn charge_target_train_values(target: &ChargeTarget) -> [f32; 3] {
    [
        target.charging_started.clamp(0.0, 1.0),
        ((target.battery_delta.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0),
        target.charging_after.clamp(0.0, 1.0),
    ]
}

fn action_value_target_train_values(target: &ActionValueTarget) -> [f32; 2] {
    [
        ((target.value.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0),
        1.0,
    ]
}

fn eye_target_train_values(target: &EyeNextTarget, output_dim: usize) -> Vec<f32> {
    let mut values = target
        .rgb
        .iter()
        .take(output_dim)
        .map(|byte| *byte as f32 / 255.0)
        .collect::<Vec<_>>();
    values.resize(output_dim, 0.0);
    values
}

fn ear_target_train_values(target: &EarNextTarget, output_dim: usize) -> Vec<f32> {
    let mut values = target
        .features
        .iter()
        .take(output_dim)
        .map(|value| value.clamp(0.0, 1.0))
        .collect::<Vec<_>>();
    values.resize(output_dim, 0.0);
    values
}

fn experience_decode_target_values(target: &ExperienceDecodeOutput, output_dim: usize) -> Vec<f32> {
    let mut values = target.flat_features();
    values.resize(output_dim, 0.0);
    values.truncate(output_dim);
    values
        .into_iter()
        .map(|value| value.clamp(0.0, 1.0))
        .collect()
}

fn split_experience_decode_values(
    values: Vec<f32>,
    lengths: ExperienceDecodeFeatureLengths,
) -> ExperienceDecodeOutput {
    let mut cursor = 0;
    let mut take = |len: usize| {
        let end = (cursor + len).min(values.len());
        let mut out = values[cursor..end].to_vec();
        out.resize(len, 0.0);
        cursor = cursor.saturating_add(len);
        out
    };
    ExperienceDecodeOutput {
        body_features: take(lengths.body),
        memory_features: take(lengths.memory),
        drive_features: take(lengths.drive),
        prediction_features: take(lengths.prediction),
        eye_features: take(lengths.eye),
        ear_features: take(lengths.ear),
    }
}

fn mse_output_target(output: DangerOutput, target: DangerTarget) -> f32 {
    let output = output.risks();
    let target = target.risks();
    output
        .iter()
        .zip(target.iter())
        .map(|(actual, expected)| {
            let delta = actual - expected;
            delta * delta
        })
        .sum::<f32>()
        / 4.0
}

fn mse_charge_output_target(output: ChargeOutput, target: ChargeTarget) -> f32 {
    let output = output.values();
    let target = target.values();
    output
        .iter()
        .zip(target.iter())
        .map(|(actual, expected)| {
            let delta = actual - expected;
            delta * delta
        })
        .sum::<f32>()
        / 3.0
}

fn mse_action_value_output_target(output: ActionValueOutput, target: ActionValueTarget) -> f32 {
    let delta = output.value - target.value;
    delta * delta
}

fn mse_eye_next_output_target(output: &EyeNextOutput, target: &EyeNextTarget) -> f32 {
    let len = output.rgb.len().max(target.rgb.len());
    if len == 0 {
        return 0.0;
    }
    (0..len)
        .map(|idx| {
            let actual = output.rgb.get(idx).copied().unwrap_or_default() as f32 / 255.0;
            let expected = target.rgb.get(idx).copied().unwrap_or_default() as f32 / 255.0;
            let delta = actual - expected;
            delta * delta
        })
        .sum::<f32>()
        / len as f32
}

fn mse_ear_next_output_target(output: &EarNextOutput, target: &EarNextTarget) -> f32 {
    let len = output.features.len().max(target.features.len());
    if len == 0 {
        return 0.0;
    }
    (0..len)
        .map(|idx| {
            let actual = output.features.get(idx).copied().unwrap_or_default();
            let expected = target.features.get(idx).copied().unwrap_or_default();
            let delta = actual - expected;
            delta * delta
        })
        .sum::<f32>()
        / len as f32
}

fn mse_experience_decode_output_target(
    output: &ExperienceDecodeOutput,
    target: &ExperienceDecodeOutput,
) -> f32 {
    let output = output.flat_features();
    let target = target.flat_features();
    let len = output.len().max(target.len());
    if len == 0 {
        return 0.0;
    }
    (0..len)
        .map(|idx| {
            let actual = output.get(idx).copied().unwrap_or_default();
            let expected = target.get(idx).copied().unwrap_or_default();
            let delta = actual - expected;
            delta * delta
        })
        .sum::<f32>()
        / len as f32
}

pub const MODEL_REGISTRY: &[&str] = &[
    "ExperienceEncoder",
    "ExperienceDecoder",
    "ExperienceAutoencoder",
    "FuturePredictor",
    "EyeNextPredictor",
    "EarNextPredictor",
    "DangerPredictor",
    "ChargePredictor",
    "ActionValueNet",
    "SalienceNet",
    "GoalArbiterNet",
    "MemoryConsolidationNet",
    "FaceFamiliarityNet",
    "VoiceFamiliarityNet",
    "ConductorNet",
];

#[cfg(test)]
mod tests {
    use super::*;
    use netherwick_actions::ActionPrimitive;
    use netherwick_body::BodySense;
    use netherwick_experience::{
        experience_decode_target_from_now, experience_encode_input_from_now, ActionValueInput,
        ActionValueTarget, ChargeInput, ChargeTarget, DangerInput, EarNextInput, EarNextTarget,
        EyeNextInput, EyeNextTarget,
    };

    #[test]
    fn hardcoded_uses_current_now_for_body_danger() {
        let mut now = Now::blank(1, BodySense::default());
        now.body.flags.bump_left = true;
        let input = DangerInput::from_parts(vec![0.0], Some(&ActionPrimitive::Stop), &now);

        let output = HardcodedDangerPredictor.predict_from_now(&now, &input);

        assert_eq!(output.bump_risk, 1.0);
        assert!(output.confidence > 0.0);
    }

    #[test]
    fn danger_net_forward_returns_unit_risks() {
        let now = Now::blank(1, BodySense::default());
        let input = DangerInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Stop), &now);
        let trainer = DangerNetTrainer::new(input.flat_features().len());

        let output = trainer.predict(&input).unwrap();

        for risk in output.risks() {
            assert!((0.0..=1.0).contains(&risk));
        }
    }

    #[test]
    fn one_train_step_records_loss() {
        let now = Now::blank(1, BodySense::default());
        let input = DangerInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Stop), &now);
        let mut trainer = DangerNetTrainer::new(input.flat_features().len());
        let target = DangerTarget {
            bump: 1.0,
            ..DangerTarget::default()
        };

        let stats = trainer.train_step(&input, &target).unwrap();

        assert_eq!(stats.samples_seen, 1);
        assert!(stats.loss.is_finite());
    }

    #[test]
    fn shadow_comparison_writes_metric_shape() {
        let now = Now::blank(10, BodySense::default());
        let input = DangerInput::from_parts(vec![0.1], Some(&ActionPrimitive::Stop), &now);
        let mut trainer = DangerNetTrainer::new(input.flat_features().len());
        let target = DangerTarget::default();

        let metric = trainer.shadow_compare(10, &now, &input, &target).unwrap();

        assert_eq!(metric.observed_at_ms, 10);
        assert!(metric.loss.is_finite());
    }

    #[test]
    fn danger_checkpoint_round_trips_prediction_shape() {
        let dir = std::env::temp_dir().join(format!("netherwick-danger-checkpoint-{}", now_ms()));
        let now = Now::blank(1, BodySense::default());
        let input = DangerInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Stop), &now);
        let mut trainer = DangerNetTrainer::new(input.flat_features().len());
        trainer
            .train_step(
                &input,
                &DangerTarget {
                    bump: 1.0,
                    ..DangerTarget::default()
                },
            )
            .unwrap();

        trainer.save_checkpoint(&dir).unwrap();
        let loaded = DangerNetTrainer::load_checkpoint(&dir, input.flat_features().len()).unwrap();
        let output = loaded.predict(&input).unwrap();

        assert!(dir.join("model.bin").exists());
        assert!(dir.join("metadata.json").exists());
        assert_eq!(loaded.samples_seen(), 1);
        for risk in output.risks() {
            assert!((0.0..=1.0).contains(&risk));
        }

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn danger_checkpoint_rejects_dimension_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "netherwick-danger-checkpoint-mismatch-{}",
            now_ms()
        ));
        let trainer = DangerNetTrainer::new(3);

        trainer.save_checkpoint(&dir).unwrap();
        let err = match DangerNetTrainer::load_checkpoint(&dir, 4) {
            Ok(_) => panic!("expected dimension mismatch"),
            Err(err) => err,
        };

        assert!(err
            .to_string()
            .contains("danger checkpoint input dimension mismatch"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn hardcoded_charge_uses_current_charging_state() {
        let mut now = Now::blank(1, BodySense::default());
        now.body.charging = true;
        let input = ChargeInput::from_parts(vec![0.0], Some(&ActionPrimitive::Stop), &now);

        let output = HardcodedChargePredictor.predict_from_now(&now, &input);

        assert!(output.charge_probability > 0.9);
        assert!(output.expected_battery_delta > 0.0);
    }

    #[test]
    fn charge_net_forward_returns_bounded_outputs() {
        let now = Now::blank(1, BodySense::default());
        let input = ChargeInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Dock), &now);
        let trainer = ChargeNetTrainer::new(input.flat_features().len());

        let output = trainer.predict(&input).unwrap();

        assert!((0.0..=1.0).contains(&output.charge_probability));
        assert!((-1.0..=1.0).contains(&output.expected_battery_delta));
        assert!((0.0..=1.0).contains(&output.dock_likelihood));
    }

    #[test]
    fn charge_checkpoint_round_trips_prediction_shape() {
        let dir = std::env::temp_dir().join(format!("netherwick-charge-checkpoint-{}", now_ms()));
        let now = Now::blank(1, BodySense::default());
        let input = ChargeInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Dock), &now);
        let mut trainer = ChargeNetTrainer::new(input.flat_features().len());
        trainer
            .train_step(
                &input,
                &ChargeTarget {
                    charging_started: 1.0,
                    battery_delta: 0.03,
                    charging_after: 1.0,
                },
            )
            .unwrap();

        trainer.save_checkpoint(&dir).unwrap();
        let loaded = ChargeNetTrainer::load_checkpoint(&dir, input.flat_features().len()).unwrap();
        let output = loaded.predict(&input).unwrap();

        assert!(dir.join("model.bin").exists());
        assert!(dir.join("metadata.json").exists());
        assert_eq!(loaded.samples_seen(), 1);
        assert!((0.0..=1.0).contains(&output.charge_probability));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn action_value_net_forward_returns_finite_value() {
        let now = Now::blank(1, BodySense::default());
        let input =
            ActionValueInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Dock), &now);
        let trainer = ActionValueNetTrainer::new(input.flat_features().len());

        let output = trainer.predict(&input).unwrap();

        assert!(output.value.is_finite());
        assert!(output.confidence.is_finite());
        assert!((-1.0..=1.0).contains(&output.value));
        assert!((0.0..=1.0).contains(&output.confidence));
    }

    #[test]
    fn action_value_checkpoint_round_trips_prediction_shape() {
        let dir =
            std::env::temp_dir().join(format!("netherwick-action-value-checkpoint-{}", now_ms()));
        let now = Now::blank(1, BodySense::default());
        let input =
            ActionValueInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Dock), &now);
        let mut trainer = ActionValueNetTrainer::new(input.flat_features().len());
        trainer
            .train_step(&input, &ActionValueTarget { value: 0.4 })
            .unwrap();

        trainer.save_checkpoint(&dir).unwrap();
        let loaded =
            ActionValueNetTrainer::load_checkpoint(&dir, input.flat_features().len()).unwrap();
        let output = loaded.predict(&input).unwrap();

        assert!(dir.join("model.bin").exists());
        assert!(dir.join("metadata.json").exists());
        assert_eq!(loaded.samples_seen(), 1);
        assert!(output.value.is_finite());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn eye_next_checkpoint_round_trips_prediction_shape() {
        let dir = std::env::temp_dir().join(format!("netherwick-eye-next-checkpoint-{}", now_ms()));
        let mut now = Now::blank(1, BodySense::default());
        now.eye.frames = vec![vec![0.2, 0.4, 0.6, 0.8]];
        let input =
            EyeNextInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Stop), &now, 100);
        let mut trainer = EyeNextNetTrainer::new(input.flat_features().len(), 4, 4);
        let target = EyeNextTarget {
            width: 4,
            height: 4,
            rgb: vec![128; 4 * 4 * 3],
        };
        trainer.train_step(&input, &target).unwrap();

        trainer.save_checkpoint(&dir).unwrap();
        let loaded = EyeNextNetTrainer::load_checkpoint(&dir, input.flat_features().len()).unwrap();
        let output = loaded.predict(&input).unwrap();

        assert!(dir.join("model.bin").exists());
        assert!(dir.join("metadata.json").exists());
        assert_eq!(loaded.samples_seen(), 1);
        assert_eq!(output.width, 4);
        assert_eq!(output.height, 4);
        assert_eq!(output.rgb.len(), 4 * 4 * 3);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn copy_current_ear_predictor_returns_current_features() {
        let mut now = Now::blank(1, BodySense::default());
        now.ear.features = vec![vec![0.2, 0.4], vec![0.6, 0.8]];
        let input =
            EarNextInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Stop), &now, 100);

        let output = CopyCurrentEarPredictor.predict_from_now(&now, &input);

        assert_eq!(output.features, vec![0.2, 0.4, 0.6, 0.8]);
        assert!(output.pcm.is_empty());
        assert!(output.confidence > 0.0);
    }

    #[test]
    fn copy_current_ear_predictor_returns_zero_features_without_audio() {
        let now = Now::blank(1, BodySense::default());
        let input =
            EarNextInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Stop), &now, 100);

        let output = CopyCurrentEarPredictor.predict_from_now(&now, &input);

        assert_eq!(output.features, vec![0.0; input.ear_features.len()]);
        assert!(output.pcm.is_empty());
        assert_eq!(output.confidence, 0.0);
    }

    #[test]
    fn ear_next_net_forward_returns_bounded_features() {
        let mut now = Now::blank(1, BodySense::default());
        now.ear.features = vec![vec![0.2, 0.4, 0.6, 0.8]];
        let input =
            EarNextInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Stop), &now, 100);
        let trainer = EarNextNetTrainer::new(input.flat_features().len(), 4);

        let output = trainer.predict(&input).unwrap();

        assert_eq!(output.features.len(), 4);
        assert!(output
            .features
            .iter()
            .all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn ear_next_checkpoint_round_trips_prediction_shape() {
        let dir = std::env::temp_dir().join(format!("netherwick-ear-next-checkpoint-{}", now_ms()));
        let mut now = Now::blank(1, BodySense::default());
        now.ear.features = vec![vec![0.2, 0.4, 0.6, 0.8]];
        let input =
            EarNextInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Stop), &now, 100);
        let mut trainer = EarNextNetTrainer::new(input.flat_features().len(), 4);
        let target = EarNextTarget {
            features: vec![0.1, 0.3, 0.5, 0.7],
            ..EarNextTarget::default()
        };
        trainer.train_step(&input, &target).unwrap();

        trainer.save_checkpoint(&dir).unwrap();
        let loaded = EarNextNetTrainer::load_checkpoint(&dir, input.flat_features().len()).unwrap();
        let output = loaded.predict(&input).unwrap();

        assert!(dir.join("model.bin").exists());
        assert!(dir.join("metadata.json").exists());
        assert_eq!(loaded.samples_seen(), 1);
        assert_eq!(output.features.len(), 4);
        assert!(output.pcm.is_empty());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn experience_autoencoder_forward_returns_fixed_size_z_and_decode_lengths() {
        let mut now = Now::blank(1, BodySense::default());
        now.eye.frames = vec![vec![0.2, 0.4, 0.6, 0.8]];
        now.ear.features = vec![vec![0.2, 0.4, 0.6, 0.8]];
        let input = experience_encode_input_from_now(&now);
        let target = experience_decode_target_from_now(&now);
        let trainer = ExperienceAutoencoderTrainer::new(
            input.flat_features().len(),
            12,
            target.feature_lengths(),
        );

        let prediction = trainer.predict(&input).unwrap();

        assert_eq!(prediction.encoded.z.len(), 12);
        assert_eq!(
            prediction.decoded.feature_lengths(),
            target.feature_lengths()
        );
        assert_eq!(
            prediction.decoded.eye_features.len(),
            target.eye_features.len()
        );
        assert_eq!(
            prediction.decoded.ear_features.len(),
            target.ear_features.len()
        );
    }

    #[test]
    fn experience_autoencoder_train_step_records_loss() {
        let mut now = Now::blank(1, BodySense::default());
        now.memory.place_familiarity = 0.7;
        now.drives.curiosity = 0.5;
        let input = experience_encode_input_from_now(&now);
        let target = experience_decode_target_from_now(&now);
        let mut trainer = ExperienceAutoencoderTrainer::new(
            input.flat_features().len(),
            8,
            target.feature_lengths(),
        );

        let stats = trainer.train_step(&input, &target).unwrap();

        assert_eq!(stats.samples_seen, 1);
        assert!(stats.loss.is_finite());
    }

    #[test]
    fn experience_autoencoder_checkpoint_round_trips_prediction_shape() {
        let dir =
            std::env::temp_dir().join(format!("netherwick-experience-checkpoint-{}", now_ms()));
        let mut now = Now::blank(1, BodySense::default());
        now.eye.frames = vec![vec![0.2, 0.4, 0.6, 0.8]];
        now.ear.features = vec![vec![0.2, 0.4, 0.6, 0.8]];
        let input = experience_encode_input_from_now(&now);
        let target = experience_decode_target_from_now(&now);
        let mut trainer = ExperienceAutoencoderTrainer::new(
            input.flat_features().len(),
            10,
            target.feature_lengths(),
        );
        trainer.train_step(&input, &target).unwrap();

        trainer.save_checkpoint(&dir).unwrap();
        let loaded =
            ExperienceAutoencoderTrainer::load_checkpoint(&dir, input.flat_features().len())
                .unwrap();
        let prediction = loaded.predict(&input).unwrap();

        assert!(dir.join("model.bin").exists());
        assert!(dir.join("metadata.json").exists());
        assert_eq!(loaded.samples_seen(), 1);
        assert_eq!(prediction.encoded.z.len(), 10);
        assert_eq!(
            prediction.decoded.feature_lengths(),
            target.feature_lengths()
        );

        let _ = std::fs::remove_dir_all(dir);
    }
}
