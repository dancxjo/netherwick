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
    ChargeInput, ChargeOutput, ChargeTarget, DangerInput, DangerOutput, DangerTarget,
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

pub type DangerBackend = NdArray<f32>;
pub type DangerAutodiffBackend = Autodiff<DangerBackend>;
pub type ChargeBackend = NdArray<f32>;
pub type ChargeAutodiffBackend = Autodiff<ChargeBackend>;

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

fn charge_target_train_values(target: &ChargeTarget) -> [f32; 3] {
    [
        target.charging_started.clamp(0.0, 1.0),
        ((target.battery_delta.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0),
        target.charging_after.clamp(0.0, 1.0),
    ]
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

pub const MODEL_REGISTRY: &[&str] = &[
    "ExperienceEncoder",
    "ExperienceDecoder",
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
    use netherwick_experience::{ChargeInput, ChargeTarget, DangerInput};

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
}
