use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use netherwick_behaviors::{
    BehaviorConfig, BehaviorRegime, BehaviorRegistryConfig, FallbackPolicy,
};
use netherwick_core::TimeMs;
use netherwick_experience::{
    action_value_input_from_transition_like, action_value_target_from_reward_surprise,
    charge_input_from_transition_like, charge_target_from_transition_like,
    danger_input_from_transition_like, danger_target_from_transition_like,
    ear_next_input_from_transition_like, ear_next_target_from_now,
    experience_decode_target_from_now, experience_encode_input_from_now,
    eye_next_input_from_transition_like, eye_next_target_from_now, ActionValueInput,
    ActionValueTarget, ChargeInput, ChargeTarget, DangerInput, DangerTarget, EarNextInput,
    EarNextTarget, ExperienceDecodeOutput, ExperienceEncodeInput, EyeNextInput, EyeNextTarget,
    FutureInput, FuturePredictor, StasisFuturePredictor,
};
use netherwick_ledger::{
    future_input_from_transition, future_target_from_transition, ExperienceTransition, JsonlLedger,
};
use netherwick_models::{
    ActionValueNetTrainer, ChargeNetTrainer, DangerNetTrainer, EarNextNetTrainer,
    ExperienceAutoencoderTrainer, EyeNextNetTrainer, FutureNetTrainer,
    HardcodedActionValuePredictor, HardcodedChargePredictor, HardcodedDangerPredictor, TrainStats,
};
use netherwick_now::Now;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TrainableBehavior {
    Danger,
    Charge,
    ActionValue,
    EyeNext,
    EarNext,
    Experience,
    Future,
}

impl TrainableBehavior {
    pub fn config_key(&self) -> &'static str {
        match self {
            Self::Danger => "danger",
            Self::Charge => "charge",
            Self::ActionValue => "action_value",
            Self::EyeNext => "eye_next",
            Self::EarNext => "ear_next",
            Self::Experience => "experience",
            Self::Future => "future",
        }
    }

    pub fn cli_name(&self) -> &'static str {
        match self {
            Self::Danger => "danger",
            Self::Charge => "charge",
            Self::ActionValue => "action-value",
            Self::EyeNext => "eye-next",
            Self::EarNext => "ear-next",
            Self::Experience => "experience",
            Self::Future => "future",
        }
    }

    pub fn default_model_id(&self) -> &'static str {
        match self {
            Self::Danger => "danger.burn.v0",
            Self::Charge => "charge.burn.v0",
            Self::ActionValue => "action_value.burn.v0",
            Self::EyeNext => "eye.burn.next_v0",
            Self::EarNext => "ear.burn.next_v0",
            Self::Experience => "experience.autoencoder.v0",
            Self::Future => "future.burn.v0",
        }
    }

    pub fn default_hardcoded_id(&self) -> &'static str {
        match self {
            Self::Danger => "danger.range_bumper",
            Self::Charge => "charge.sensor_battery_delta",
            Self::ActionValue => "action_value.handcoded",
            Self::EyeNext => "eye.copy_current",
            Self::EarNext => "ear.copy_current",
            Self::Experience => "experience.feature_encoder",
            Self::Future => "future.stasis",
        }
    }

    fn is_safety_critical(&self) -> bool {
        matches!(
            self,
            Self::Danger | Self::ActionValue | Self::Experience | Self::Future
        )
    }
}

impl fmt::Display for TrainableBehavior {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.cli_name())
    }
}

impl FromStr for TrainableBehavior {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "danger" => Ok(Self::Danger),
            "charge" => Ok(Self::Charge),
            "action-value" | "action_value" => Ok(Self::ActionValue),
            "eye-next" | "eye_next" => Ok(Self::EyeNext),
            "ear-next" | "ear_next" => Ok(Self::EarNext),
            "experience" => Ok(Self::Experience),
            "future" => Ok(Self::Future),
            other => bail!("unknown trainable behavior {other:?}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainBehaviorRequest {
    pub behavior: TrainableBehavior,
    pub ledger_path: PathBuf,
    pub checkpoint_path: PathBuf,
    pub epochs: usize,
    pub validation_split: f32,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct EvaluateBehaviorRequest {
    pub behavior: TrainableBehavior,
    pub ledger_path: PathBuf,
    pub checkpoint_path: PathBuf,
    pub max_samples: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BehaviorEvaluationReport {
    pub behavior: TrainableBehavior,
    pub checkpoint_path: PathBuf,
    pub sample_count: usize,
    pub model_loss_mean: f32,
    pub hardcoded_loss_mean: Option<f32>,
    pub selected_loss_mean: Option<f32>,
    pub model_better_than_hardcoded: Option<bool>,
    pub improvement_ratio: Option<f32>,
    pub warnings: Vec<String>,
    pub recommendation: PromotionRecommendation,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PromotionRecommendation {
    KeepHardcoded,
    ShadowInfer,
    ShadowTrain,
    PromoteToModelInfer,
    RejectCheckpoint,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BehaviorMetricRecord {
    pub t_ms: TimeMs,
    pub behavior: TrainableBehavior,
    pub epoch: usize,
    pub sample_index: usize,
    pub train_loss: Option<f32>,
    pub eval_loss: Option<f32>,
    pub hardcoded_loss: Option<f32>,
    pub model_loss: Option<f32>,
    pub selected_loss: Option<f32>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainSummary {
    pub behavior: TrainableBehavior,
    pub transition_count: usize,
    pub train_sample_count: usize,
    pub eval_sample_count: usize,
    pub epochs: usize,
    pub samples_seen: u64,
    pub last_loss: Option<f32>,
    pub best_loss: Option<f32>,
    pub metrics_path: PathBuf,
    pub checkpoint_path: PathBuf,
    pub evaluation: BehaviorEvaluationReport,
}

pub trait BehaviorTrainer {
    type Input;
    type Output;
    type Target;

    fn behavior(&self) -> TrainableBehavior;

    fn train_step(&mut self, input: &Self::Input, target: &Self::Target) -> Result<TrainStats>;

    fn evaluate_sample(&self, input: &Self::Input, target: &Self::Target) -> Result<f32>;

    fn hardcoded_loss(&self, input: &Self::Input, target: &Self::Target) -> Result<Option<f32>>;

    fn save_checkpoint(&self, path: &Path) -> Result<()>;
}

pub async fn load_transitions(path: impl AsRef<Path>) -> Result<Vec<ExperienceTransition>> {
    let path = path.as_ref();
    if !path.exists() {
        bail!("missing ledger at {}", path.display());
    }
    if !path.is_dir() {
        bail!(
            "wrong ledger path shape: {} is not a directory",
            path.display()
        );
    }

    let transitions = JsonlLedger::new(path)
        .transitions()
        .await
        .with_context(|| {
            format!(
                "invalid JSONL while reading transitions below {}",
                path.display()
            )
        })?;
    if transitions.is_empty() {
        bail!("no transitions found below {}", path.display());
    }
    let without_z = transitions
        .iter()
        .filter(|transition| transition.before_z.z.is_empty() || transition.after_z.z.is_empty())
        .count();
    if without_z == transitions.len() {
        bail!(
            "transitions below {} do not contain usable before_z/after_z vectors",
            path.display()
        );
    }
    Ok(transitions)
}

pub fn split_transitions(
    mut transitions: Vec<ExperienceTransition>,
    validation_split: f32,
    seed: u64,
) -> (Vec<ExperienceTransition>, Vec<ExperienceTransition>) {
    let validation_split = validation_split.clamp(0.0, 0.9);
    let mut rng = StdRng::seed_from_u64(seed);
    transitions.shuffle(&mut rng);
    let eval_len = ((transitions.len() as f32) * validation_split).round() as usize;
    let eval_len = eval_len.min(transitions.len().saturating_sub(1));
    let eval = transitions.split_off(transitions.len().saturating_sub(eval_len));
    (transitions, eval)
}

pub async fn train_behavior(request: TrainBehaviorRequest) -> Result<TrainSummary> {
    let transitions = load_transitions(&request.ledger_path).await?;
    let transition_count = transitions.len();
    let (train, eval) = split_transitions(transitions, request.validation_split, request.seed);
    tokio::fs::create_dir_all(&request.checkpoint_path).await?;
    let metrics_path = request.checkpoint_path.join("metrics.jsonl");

    let mut writer = MetricWriter::open(&metrics_path).await?;
    let mut last_loss = None;
    let mut samples_seen = 0;
    let mut best_loss = None;
    let train_sample_count;

    match request.behavior {
        TrainableBehavior::Danger => {
            let samples = danger_samples(&train);
            train_sample_count = samples.len();
            let mut trainer = DangerNetTrainer::new(first_dim(&samples, |(_, _, input, _)| {
                input.flat_features().len()
            })?);
            for epoch in 0..request.epochs {
                for (sample_index, (t_ms, before, input, target)) in samples.iter().enumerate() {
                    if input.flat_features().len() != trainer.input_dim() {
                        continue;
                    }
                    let hardcoded_loss = Some(mse(
                        &HardcodedDangerPredictor
                            .predict_from_now(before, input)
                            .risks(),
                        &target.risks(),
                    ));
                    let model_loss = trainer.evaluate_sample(input, target)?;
                    let stats = trainer.train_step(input, target)?;
                    last_loss = Some(stats.loss);
                    samples_seen = stats.samples_seen;
                    best_loss = trainer.best_loss();
                    writer
                        .write(train_metric(
                            *t_ms,
                            TrainableBehavior::Danger,
                            epoch,
                            sample_index,
                            stats.loss,
                            hardcoded_loss,
                            model_loss,
                        ))
                        .await?;
                }
            }
            trainer.save_checkpoint(&request.checkpoint_path)?;
        }
        TrainableBehavior::Charge => {
            let samples = charge_samples(&train);
            train_sample_count = samples.len();
            let mut trainer = ChargeNetTrainer::new(first_dim(&samples, |(_, _, input, _)| {
                input.flat_features().len()
            })?);
            for epoch in 0..request.epochs {
                for (sample_index, (t_ms, before, input, target)) in samples.iter().enumerate() {
                    if input.flat_features().len() != trainer.input_dim() {
                        continue;
                    }
                    let hardcoded_loss = Some(mse(
                        &HardcodedChargePredictor
                            .predict_from_now(before, input)
                            .values(),
                        &target.values(),
                    ));
                    let model_loss = trainer.evaluate_sample(input, target)?;
                    let stats = trainer.train_step(input, target)?;
                    last_loss = Some(stats.loss);
                    samples_seen = stats.samples_seen;
                    best_loss = trainer.best_loss();
                    writer
                        .write(train_metric(
                            *t_ms,
                            TrainableBehavior::Charge,
                            epoch,
                            sample_index,
                            stats.loss,
                            hardcoded_loss,
                            model_loss,
                        ))
                        .await?;
                }
            }
            trainer.save_checkpoint(&request.checkpoint_path)?;
        }
        TrainableBehavior::ActionValue => {
            let samples = action_value_samples(&train);
            train_sample_count = samples.len();
            let mut trainer =
                ActionValueNetTrainer::new(first_dim(&samples, |(_, _, input, _)| {
                    input.flat_features().len()
                })?);
            for epoch in 0..request.epochs {
                for (sample_index, (t_ms, before, input, target)) in samples.iter().enumerate() {
                    if input.flat_features().len() != trainer.input_dim() {
                        continue;
                    }
                    let hardcoded = HardcodedActionValuePredictor.predict_from_now(before, input);
                    let hardcoded_loss = Some((hardcoded.value - target.value).powi(2));
                    let model_loss = trainer.evaluate_sample(input, target)?;
                    let stats = trainer.train_step(input, target)?;
                    last_loss = Some(stats.loss);
                    samples_seen = stats.samples_seen;
                    best_loss = trainer.best_loss();
                    writer
                        .write(train_metric(
                            *t_ms,
                            TrainableBehavior::ActionValue,
                            epoch,
                            sample_index,
                            stats.loss,
                            hardcoded_loss,
                            model_loss,
                        ))
                        .await?;
                }
            }
            trainer.save_checkpoint(&request.checkpoint_path)?;
        }
        TrainableBehavior::Future => {
            let samples = future_samples(&train);
            train_sample_count = samples.len();
            let input_dim = first_dim(&samples, |(_, input, _)| input.flat_features().len())?;
            let latent_dim = first_dim(&samples, |(_, _, target)| target.len())?;
            let mut trainer = FutureNetTrainer::new(input_dim, latent_dim);
            let mut stasis = StasisFuturePredictor;
            for epoch in 0..request.epochs {
                for (sample_index, (t_ms, input, target)) in samples.iter().enumerate() {
                    if input.flat_features().len() != trainer.input_dim()
                        || target.len() != trainer.latent_dim()
                    {
                        continue;
                    }
                    let hardcoded =
                        stasis.predict(&input.latent, &input.action, input.offset_ms)?;
                    let hardcoded_loss = Some(mse_vec(&hardcoded.predicted_z, target));
                    let model_loss = trainer.evaluate_sample(input, target)?;
                    let stats = trainer.train_step(input, target)?;
                    last_loss = Some(stats.loss);
                    samples_seen = stats.samples_seen;
                    best_loss = trainer.best_loss();
                    writer
                        .write(train_metric(
                            *t_ms,
                            TrainableBehavior::Future,
                            epoch,
                            sample_index,
                            stats.loss,
                            hardcoded_loss,
                            model_loss,
                        ))
                        .await?;
                }
            }
            trainer.save_checkpoint(&request.checkpoint_path)?;
        }
        TrainableBehavior::EyeNext => {
            let samples = eye_next_samples(&train);
            train_sample_count = samples.len();
            let (input_dim, width, height) = samples
                .first()
                .map(|(_, _, input, target)| {
                    (input.flat_features().len(), target.width, target.height)
                })
                .ok_or_else(|| anyhow!("no usable eye-next samples"))?;
            let mut trainer = EyeNextNetTrainer::new(input_dim, width, height);
            for epoch in 0..request.epochs {
                for (sample_index, (t_ms, before, input, target)) in samples.iter().enumerate() {
                    if input.flat_features().len() != trainer.input_dim() {
                        continue;
                    }
                    let hardcoded_loss = eye_current_loss(before, target);
                    let model_loss = trainer.evaluate_sample(input, target)?;
                    let stats = trainer.train_step(input, target)?;
                    last_loss = Some(stats.loss);
                    samples_seen = stats.samples_seen;
                    best_loss = trainer.best_loss();
                    writer
                        .write(train_metric(
                            *t_ms,
                            TrainableBehavior::EyeNext,
                            epoch,
                            sample_index,
                            stats.loss,
                            hardcoded_loss,
                            model_loss,
                        ))
                        .await?;
                }
            }
            trainer.save_checkpoint(&request.checkpoint_path)?;
        }
        TrainableBehavior::EarNext => {
            let samples = ear_next_samples(&train);
            train_sample_count = samples.len();
            let (input_dim, output_dim, sample_rate_hz, channels) = samples
                .first()
                .map(|(_, _, input, target)| {
                    (
                        input.flat_features().len(),
                        target.features.len(),
                        target.sample_rate_hz,
                        target.channels,
                    )
                })
                .ok_or_else(|| anyhow!("no usable ear-next samples"))?;
            let mut trainer = EarNextNetTrainer::with_audio_shape(
                input_dim,
                output_dim,
                sample_rate_hz,
                channels,
            );
            for epoch in 0..request.epochs {
                for (sample_index, (t_ms, before, input, target)) in samples.iter().enumerate() {
                    if input.flat_features().len() != trainer.input_dim()
                        || target.features.len() != trainer.output_dim()
                    {
                        continue;
                    }
                    let hardcoded_loss = ear_current_loss(before, target);
                    let model_loss = trainer.evaluate_sample(input, target)?;
                    let stats = trainer.train_step(input, target)?;
                    last_loss = Some(stats.loss);
                    samples_seen = stats.samples_seen;
                    best_loss = trainer.best_loss();
                    writer
                        .write(train_metric(
                            *t_ms,
                            TrainableBehavior::EarNext,
                            epoch,
                            sample_index,
                            stats.loss,
                            hardcoded_loss,
                            model_loss,
                        ))
                        .await?;
                }
            }
            trainer.save_checkpoint(&request.checkpoint_path)?;
        }
        TrainableBehavior::Experience => {
            let samples = experience_samples(&train);
            train_sample_count = samples.len();
            let (input_dim, decode_lengths) = samples
                .first()
                .map(|(_, input, target, _)| {
                    (input.flat_features().len(), target.feature_lengths())
                })
                .ok_or_else(|| anyhow!("no usable experience samples"))?;
            let z_dim = input_dim.clamp(8, 32);
            let mut trainer = ExperienceAutoencoderTrainer::new(input_dim, z_dim, decode_lengths);
            for epoch in 0..request.epochs {
                for (sample_index, (t_ms, input, target, _baseline_z)) in samples.iter().enumerate()
                {
                    if input.flat_features().len() != trainer.input_dim()
                        || target.feature_lengths() != trainer.decode_lengths()
                    {
                        continue;
                    }
                    let model_loss = trainer.evaluate_sample(input, target)?;
                    let stats = trainer.train_step(input, target)?;
                    last_loss = Some(stats.loss);
                    samples_seen = stats.samples_seen;
                    best_loss = trainer.best_loss();
                    writer
                        .write(train_metric(
                            *t_ms,
                            TrainableBehavior::Experience,
                            epoch,
                            sample_index,
                            stats.loss,
                            None,
                            model_loss,
                        ))
                        .await?;
                }
            }
            trainer.save_checkpoint(&request.checkpoint_path)?;
        }
    }

    let eval_transitions = if eval.is_empty() { &train } else { &eval };
    let eval_request = EvaluateBehaviorRequest {
        behavior: request.behavior.clone(),
        ledger_path: request.ledger_path,
        checkpoint_path: request.checkpoint_path.clone(),
        max_samples: None,
    };
    let mut evaluation = evaluate_behavior_on_transitions(eval_request, eval_transitions)?;
    evaluation.checkpoint_path = request.checkpoint_path.clone();

    let evaluation_path = request.checkpoint_path.join("evaluation.json");
    if let Ok(json) = serde_json::to_string_pretty(&evaluation) {
        let _ = std::fs::write(&evaluation_path, json);
    }

    Ok(TrainSummary {
        behavior: request.behavior,
        transition_count,
        train_sample_count,
        eval_sample_count: evaluation.sample_count,
        epochs: request.epochs,
        samples_seen,
        last_loss,
        best_loss,
        metrics_path,
        checkpoint_path: request.checkpoint_path,
        evaluation,
    })
}

pub async fn evaluate_behavior(
    request: EvaluateBehaviorRequest,
) -> Result<BehaviorEvaluationReport> {
    let transitions = load_transitions(&request.ledger_path).await?;
    evaluate_behavior_on_transitions(request, &transitions)
}

pub fn load_models_config(path: &Path) -> Result<BehaviorRegistryConfig> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

pub fn set_behavior_checkpoint(
    config: &mut BehaviorRegistryConfig,
    behavior: TrainableBehavior,
    checkpoint: PathBuf,
) -> Result<()> {
    let entry = behavior_config_entry(config, &behavior);
    entry.checkpoint = Some(checkpoint.to_string_lossy().to_string());
    Ok(())
}

pub fn set_behavior_regime(
    config: &mut BehaviorRegistryConfig,
    behavior: TrainableBehavior,
    regime: BehaviorRegime,
) -> Result<()> {
    let entry = behavior_config_entry(config, &behavior);
    entry.regime = regime;
    entry.fallback = FallbackPolicy::UseHardcoded;
    Ok(())
}

pub fn write_models_config(path: &Path, config: &BehaviorRegistryConfig) -> Result<()> {
    let text = toml::to_string_pretty(config)?;
    std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

pub fn promote_behavior_config(
    behavior: TrainableBehavior,
    checkpoint: PathBuf,
    config_path: &Path,
    regime: BehaviorRegime,
) -> Result<()> {
    if regime == BehaviorRegime::ModelInfer && behavior.is_safety_critical() {
        eprintln!(
            "warning: {} is safety-critical; model-infer was explicitly requested",
            behavior
        );
    }
    let mut config = load_models_config(config_path)?;
    set_behavior_checkpoint(&mut config, behavior.clone(), checkpoint)?;
    set_behavior_regime(&mut config, behavior, regime)?;
    write_models_config(config_path, &config)
}

fn behavior_config_entry<'a>(
    config: &'a mut BehaviorRegistryConfig,
    behavior: &TrainableBehavior,
) -> &'a mut BehaviorConfig {
    config
        .behavior
        .entry(behavior.config_key().to_string())
        .or_insert_with(|| BehaviorConfig {
            regime: BehaviorRegime::Hardcoded,
            hardcoded: behavior.default_hardcoded_id().to_string(),
            model: Some(behavior.default_model_id().to_string()),
            checkpoint: None,
            fallback: FallbackPolicy::UseHardcoded,
        })
}

fn evaluate_behavior_on_transitions(
    request: EvaluateBehaviorRequest,
    transitions: &[ExperienceTransition],
) -> Result<BehaviorEvaluationReport> {
    let max_samples = request.max_samples.unwrap_or(usize::MAX);
    let mut warnings = Vec::new();
    let (model_losses, hardcoded_losses): (Vec<f32>, Vec<Option<f32>>) = match request.behavior {
        TrainableBehavior::Danger => {
            let samples = danger_samples(transitions);
            let input_dim = first_dim(&samples, |(_, _, input, _)| input.flat_features().len())?;
            let trainer = DangerNetTrainer::load_checkpoint(&request.checkpoint_path, input_dim)?;
            samples
                .into_iter()
                .take(max_samples)
                .filter(|(_, _, input, _)| input.flat_features().len() == trainer.input_dim())
                .map(|(_, before, input, target)| {
                    let model = trainer.predict(&input)?;
                    let hard = HardcodedDangerPredictor.predict_from_now(&before, &input);
                    Ok((
                        mse(&model.risks(), &target.risks()),
                        Some(mse(&hard.risks(), &target.risks())),
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip()
        }
        TrainableBehavior::Charge => {
            let samples = charge_samples(transitions);
            let input_dim = first_dim(&samples, |(_, _, input, _)| input.flat_features().len())?;
            let trainer = ChargeNetTrainer::load_checkpoint(&request.checkpoint_path, input_dim)?;
            samples
                .into_iter()
                .take(max_samples)
                .filter(|(_, _, input, _)| input.flat_features().len() == trainer.input_dim())
                .map(|(_, before, input, target)| {
                    let model = trainer.predict(&input)?;
                    let hard = HardcodedChargePredictor.predict_from_now(&before, &input);
                    Ok((
                        mse(&model.values(), &target.values()),
                        Some(mse(&hard.values(), &target.values())),
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip()
        }
        TrainableBehavior::ActionValue => {
            let samples = action_value_samples(transitions);
            let input_dim = first_dim(&samples, |(_, _, input, _)| input.flat_features().len())?;
            let trainer =
                ActionValueNetTrainer::load_checkpoint(&request.checkpoint_path, input_dim)?;
            samples
                .into_iter()
                .take(max_samples)
                .filter(|(_, _, input, _)| input.flat_features().len() == trainer.input_dim())
                .map(|(_, before, input, target)| {
                    let model = trainer.predict(&input)?;
                    let hard = HardcodedActionValuePredictor.predict_from_now(&before, &input);
                    Ok((
                        (model.value - target.value).powi(2),
                        Some((hard.value - target.value).powi(2)),
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip()
        }
        TrainableBehavior::Future => {
            let samples = future_samples(transitions);
            let input_dim = first_dim(&samples, |(_, input, _)| input.flat_features().len())?;
            let latent_dim = first_dim(&samples, |(_, _, target)| target.len())?;
            let trainer =
                FutureNetTrainer::load_checkpoint(&request.checkpoint_path, input_dim, latent_dim)?;
            let mut stasis = StasisFuturePredictor;
            samples
                .into_iter()
                .take(max_samples)
                .filter(|(_, input, target)| {
                    input.flat_features().len() == trainer.input_dim()
                        && target.len() == trainer.latent_dim()
                })
                .map(|(_, input, target)| {
                    let model = trainer.predict(&input)?;
                    let hard = stasis.predict(&input.latent, &input.action, input.offset_ms)?;
                    Ok((
                        mse_vec(&model.predicted_z, &target),
                        Some(mse_vec(&hard.predicted_z, &target)),
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip()
        }
        TrainableBehavior::EyeNext => {
            let samples = eye_next_samples(transitions);
            let input_dim = first_dim(&samples, |(_, _, input, _)| input.flat_features().len())?;
            let trainer = EyeNextNetTrainer::load_checkpoint(&request.checkpoint_path, input_dim)?;
            samples
                .into_iter()
                .take(max_samples)
                .filter(|(_, _, input, _)| input.flat_features().len() == trainer.input_dim())
                .map(|(_, before, input, target)| {
                    let model = trainer.predict(&input)?;
                    Ok((
                        mse_bytes(&model.rgb, &target.rgb),
                        eye_current_loss(&before, &target),
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip()
        }
        TrainableBehavior::EarNext => {
            let samples = ear_next_samples(transitions);
            let input_dim = first_dim(&samples, |(_, _, input, _)| input.flat_features().len())?;
            let trainer = EarNextNetTrainer::load_checkpoint(&request.checkpoint_path, input_dim)?;
            samples
                .into_iter()
                .take(max_samples)
                .filter(|(_, _, input, target)| {
                    input.flat_features().len() == trainer.input_dim()
                        && target.features.len() == trainer.output_dim()
                })
                .map(|(_, before, input, target)| {
                    let model = trainer.predict(&input)?;
                    Ok((
                        mse_vec(&model.features, &target.features),
                        ear_current_loss(&before, &target),
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip()
        }
        TrainableBehavior::Experience => {
            let samples = experience_samples(transitions);
            let input_dim = first_dim(&samples, |(_, input, _, _)| input.flat_features().len())?;
            let trainer =
                ExperienceAutoencoderTrainer::load_checkpoint(&request.checkpoint_path, input_dim)?;
            samples
                .into_iter()
                .take(max_samples)
                .filter(|(_, input, target, _)| {
                    input.flat_features().len() == trainer.input_dim()
                        && target.feature_lengths() == trainer.decode_lengths()
                })
                .map(|(_, input, target, _)| {
                    let prediction = trainer.predict(&input)?;
                    Ok((
                        mse_vec(&prediction.decoded.flat_features(), &target.flat_features()),
                        None,
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip()
        }
    };

    let sample_count = model_losses.len();
    if sample_count == 0 {
        bail!("no usable {} samples for evaluation", request.behavior);
    }
    if sample_count < 50 {
        warnings.push(format!(
            "insufficient data: {sample_count} samples is below the conservative 50-sample floor"
        ));
    }
    let model_loss_mean = mean(&model_losses);
    let hardcoded_values: Vec<f32> = hardcoded_losses.into_iter().flatten().collect();
    let hardcoded_loss_mean = (!hardcoded_values.is_empty()).then(|| mean(&hardcoded_values));
    let selected_loss_mean = hardcoded_loss_mean.or(Some(model_loss_mean));
    let model_better_than_hardcoded = hardcoded_loss_mean.map(|hard| model_loss_mean < hard);
    let improvement_ratio =
        hardcoded_loss_mean.and_then(|hard| (hard > 0.0).then(|| (hard - model_loss_mean) / hard));
    let recommendation = recommend(
        &request.behavior,
        sample_count,
        model_loss_mean,
        hardcoded_loss_mean,
        improvement_ratio,
    );

    Ok(BehaviorEvaluationReport {
        behavior: request.behavior,
        checkpoint_path: request.checkpoint_path,
        sample_count,
        model_loss_mean,
        hardcoded_loss_mean,
        selected_loss_mean,
        model_better_than_hardcoded,
        improvement_ratio,
        warnings,
        recommendation,
    })
}

fn recommend(
    behavior: &TrainableBehavior,
    sample_count: usize,
    model_loss_mean: f32,
    hardcoded_loss_mean: Option<f32>,
    improvement_ratio: Option<f32>,
) -> PromotionRecommendation {
    if sample_count < 50 {
        return PromotionRecommendation::KeepHardcoded;
    }
    if !model_loss_mean.is_finite() || model_loss_mean > 1.0e6 {
        return PromotionRecommendation::RejectCheckpoint;
    }
    match (hardcoded_loss_mean, improvement_ratio) {
        (Some(_), Some(ratio))
            if ratio > 0.25 && sample_count > 500 && !behavior.is_safety_critical() =>
        {
            PromotionRecommendation::PromoteToModelInfer
        }
        (Some(_), Some(ratio)) if ratio > 0.10 => PromotionRecommendation::ShadowInfer,
        (Some(_), _) => PromotionRecommendation::KeepHardcoded,
        (None, _) if sample_count >= 50 => PromotionRecommendation::ShadowInfer,
        _ => PromotionRecommendation::KeepHardcoded,
    }
}

impl BehaviorTrainer for DangerNetTrainer {
    type Input = DangerInput;
    type Output = ();
    type Target = DangerTarget;

    fn behavior(&self) -> TrainableBehavior {
        TrainableBehavior::Danger
    }

    fn train_step(&mut self, input: &Self::Input, target: &Self::Target) -> Result<TrainStats> {
        let stats = DangerNetTrainer::train_step(self, input, target)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }

    fn evaluate_sample(&self, input: &Self::Input, target: &Self::Target) -> Result<f32> {
        Ok(mse(&self.predict(input)?.risks(), &target.risks()))
    }

    fn hardcoded_loss(&self, _input: &Self::Input, _target: &Self::Target) -> Result<Option<f32>> {
        Ok(None)
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        DangerNetTrainer::save_checkpoint(self, path)
    }
}

impl BehaviorTrainer for ChargeNetTrainer {
    type Input = ChargeInput;
    type Output = ();
    type Target = ChargeTarget;

    fn behavior(&self) -> TrainableBehavior {
        TrainableBehavior::Charge
    }

    fn train_step(&mut self, input: &Self::Input, target: &Self::Target) -> Result<TrainStats> {
        let stats = ChargeNetTrainer::train_step(self, input, target)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }

    fn evaluate_sample(&self, input: &Self::Input, target: &Self::Target) -> Result<f32> {
        Ok(mse(&self.predict(input)?.values(), &target.values()))
    }

    fn hardcoded_loss(&self, _input: &Self::Input, _target: &Self::Target) -> Result<Option<f32>> {
        Ok(None)
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        ChargeNetTrainer::save_checkpoint(self, path)
    }
}

impl BehaviorTrainer for ActionValueNetTrainer {
    type Input = ActionValueInput;
    type Output = ();
    type Target = ActionValueTarget;

    fn behavior(&self) -> TrainableBehavior {
        TrainableBehavior::ActionValue
    }

    fn train_step(&mut self, input: &Self::Input, target: &Self::Target) -> Result<TrainStats> {
        let stats = ActionValueNetTrainer::train_step(self, input, target)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }

    fn evaluate_sample(&self, input: &Self::Input, target: &Self::Target) -> Result<f32> {
        Ok((self.predict(input)?.value - target.value).powi(2))
    }

    fn hardcoded_loss(&self, _input: &Self::Input, _target: &Self::Target) -> Result<Option<f32>> {
        Ok(None)
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        ActionValueNetTrainer::save_checkpoint(self, path)
    }
}

impl BehaviorTrainer for FutureNetTrainer {
    type Input = FutureInput;
    type Output = ();
    type Target = Vec<f32>;

    fn behavior(&self) -> TrainableBehavior {
        TrainableBehavior::Future
    }

    fn train_step(&mut self, input: &Self::Input, target: &Self::Target) -> Result<TrainStats> {
        FutureNetTrainer::train_step(self, input, target)
    }

    fn evaluate_sample(&self, input: &Self::Input, target: &Self::Target) -> Result<f32> {
        Ok(mse_vec(&self.predict(input)?.predicted_z, target))
    }

    fn hardcoded_loss(&self, _input: &Self::Input, _target: &Self::Target) -> Result<Option<f32>> {
        Ok(None)
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        FutureNetTrainer::save_checkpoint(self, path)
    }
}

impl BehaviorTrainer for EyeNextNetTrainer {
    type Input = EyeNextInput;
    type Output = ();
    type Target = EyeNextTarget;

    fn behavior(&self) -> TrainableBehavior {
        TrainableBehavior::EyeNext
    }

    fn train_step(&mut self, input: &Self::Input, target: &Self::Target) -> Result<TrainStats> {
        let stats = EyeNextNetTrainer::train_step(self, input, target)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }

    fn evaluate_sample(&self, input: &Self::Input, target: &Self::Target) -> Result<f32> {
        Ok(mse_bytes(&self.predict(input)?.rgb, &target.rgb))
    }

    fn hardcoded_loss(&self, _input: &Self::Input, _target: &Self::Target) -> Result<Option<f32>> {
        Ok(None)
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        EyeNextNetTrainer::save_checkpoint(self, path)
    }
}

impl BehaviorTrainer for EarNextNetTrainer {
    type Input = EarNextInput;
    type Output = ();
    type Target = EarNextTarget;

    fn behavior(&self) -> TrainableBehavior {
        TrainableBehavior::EarNext
    }

    fn train_step(&mut self, input: &Self::Input, target: &Self::Target) -> Result<TrainStats> {
        let stats = EarNextNetTrainer::train_step(self, input, target)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }

    fn evaluate_sample(&self, input: &Self::Input, target: &Self::Target) -> Result<f32> {
        Ok(mse_vec(&self.predict(input)?.features, &target.features))
    }

    fn hardcoded_loss(&self, _input: &Self::Input, _target: &Self::Target) -> Result<Option<f32>> {
        Ok(None)
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        EarNextNetTrainer::save_checkpoint(self, path)
    }
}

impl BehaviorTrainer for ExperienceAutoencoderTrainer {
    type Input = ExperienceEncodeInput;
    type Output = ();
    type Target = ExperienceDecodeOutput;

    fn behavior(&self) -> TrainableBehavior {
        TrainableBehavior::Experience
    }

    fn train_step(&mut self, input: &Self::Input, target: &Self::Target) -> Result<TrainStats> {
        let stats = ExperienceAutoencoderTrainer::train_step(self, input, target)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }

    fn evaluate_sample(&self, input: &Self::Input, target: &Self::Target) -> Result<f32> {
        Ok(mse_vec(
            &self.predict(input)?.decoded.flat_features(),
            &target.flat_features(),
        ))
    }

    fn hardcoded_loss(&self, _input: &Self::Input, _target: &Self::Target) -> Result<Option<f32>> {
        Ok(None)
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        ExperienceAutoencoderTrainer::save_checkpoint(self, path)
    }
}

struct MetricWriter {
    file: tokio::fs::File,
}

impl MetricWriter {
    async fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
            .with_context(|| format!("open metrics {}", path.display()))?;
        Ok(Self { file })
    }

    async fn write(&mut self, record: BehaviorMetricRecord) -> Result<()> {
        let line = serde_json::to_string(&record)?;
        self.file.write_all(line.as_bytes()).await?;
        self.file.write_all(b"\n").await?;
        Ok(())
    }
}

fn train_metric(
    t_ms: TimeMs,
    behavior: TrainableBehavior,
    epoch: usize,
    sample_index: usize,
    train_loss: f32,
    hardcoded_loss: Option<f32>,
    model_loss: f32,
) -> BehaviorMetricRecord {
    BehaviorMetricRecord {
        t_ms: if t_ms == 0 { now_ms() } else { t_ms },
        behavior,
        epoch,
        sample_index,
        train_loss: Some(train_loss),
        eval_loss: None,
        hardcoded_loss,
        model_loss: Some(model_loss),
        selected_loss: hardcoded_loss,
        notes: Vec::new(),
    }
}

fn danger_samples(
    transitions: &[ExperienceTransition],
) -> Vec<(TimeMs, Now, DangerInput, DangerTarget)> {
    transitions
        .iter()
        .filter(|transition| !transition.before_z.z.is_empty() && !transition.after_z.z.is_empty())
        .map(|transition| {
            (
                transition.created_at_ms,
                transition.before.clone(),
                danger_input_from_transition_like(
                    &transition.before_z,
                    transition.action.as_ref(),
                    &transition.before,
                ),
                danger_target_from_transition_like(
                    &transition.before,
                    transition.action.as_ref(),
                    &transition.after,
                ),
            )
        })
        .collect()
}

fn charge_samples(
    transitions: &[ExperienceTransition],
) -> Vec<(TimeMs, Now, ChargeInput, ChargeTarget)> {
    transitions
        .iter()
        .filter(|transition| !transition.before_z.z.is_empty() && !transition.after_z.z.is_empty())
        .map(|transition| {
            (
                transition.created_at_ms,
                transition.before.clone(),
                charge_input_from_transition_like(
                    &transition.before_z,
                    transition.action.as_ref(),
                    &transition.before,
                ),
                charge_target_from_transition_like(
                    &transition.before,
                    transition.action.as_ref(),
                    &transition.after,
                ),
            )
        })
        .collect()
}

fn action_value_samples(
    transitions: &[ExperienceTransition],
) -> Vec<(TimeMs, Now, ActionValueInput, ActionValueTarget)> {
    transitions
        .iter()
        .filter(|transition| !transition.before_z.z.is_empty() && !transition.after_z.z.is_empty())
        .map(|transition| {
            (
                transition.created_at_ms,
                transition.before.clone(),
                action_value_input_from_transition_like(
                    &transition.before_z,
                    transition.action.as_ref(),
                    &transition.before,
                ),
                action_value_target_from_reward_surprise(&transition.reward, &transition.surprise),
            )
        })
        .collect()
}

fn future_samples(transitions: &[ExperienceTransition]) -> Vec<(TimeMs, FutureInput, Vec<f32>)> {
    transitions
        .iter()
        .filter_map(|transition| {
            let input = future_input_from_transition(transition, 1_000)?;
            let target = future_target_from_transition(transition);
            (!target.is_empty()).then_some((transition.created_at_ms, input, target))
        })
        .collect()
}

fn eye_next_samples(
    transitions: &[ExperienceTransition],
) -> Vec<(TimeMs, Now, EyeNextInput, EyeNextTarget)> {
    transitions
        .iter()
        .filter_map(|transition| {
            let target = eye_next_target_from_now(&transition.after)?;
            let input = eye_next_input_from_transition_like(
                &transition.before_z,
                transition.action.as_ref(),
                &transition.before,
                100,
            );
            Some((
                transition.created_at_ms,
                transition.before.clone(),
                input,
                target,
            ))
        })
        .collect()
}

fn ear_next_samples(
    transitions: &[ExperienceTransition],
) -> Vec<(TimeMs, Now, EarNextInput, EarNextTarget)> {
    transitions
        .iter()
        .filter_map(|transition| {
            let target = ear_next_target_from_now(&transition.after)?;
            let input = ear_next_input_from_transition_like(
                &transition.before_z,
                transition.action.as_ref(),
                &transition.before,
                100,
            );
            Some((
                transition.created_at_ms,
                transition.before.clone(),
                input,
                target,
            ))
        })
        .collect()
}

fn experience_samples(
    transitions: &[ExperienceTransition],
) -> Vec<(
    TimeMs,
    ExperienceEncodeInput,
    ExperienceDecodeOutput,
    Vec<f32>,
)> {
    let mut samples = Vec::new();
    for transition in transitions {
        for (t_ms, now, baseline_z) in [
            (
                transition.created_at_ms,
                &transition.before,
                transition.before_z.z.clone(),
            ),
            (
                transition.created_at_ms,
                &transition.after,
                transition.after_z.z.clone(),
            ),
        ] {
            let input = experience_encode_input_from_now(now);
            let target = experience_decode_target_from_now(now);
            if input.flat_features().is_empty() || target.flat_features().is_empty() {
                continue;
            }
            samples.push((t_ms, input, target, baseline_z));
        }
    }
    samples
}

fn first_dim<T>(samples: &[T], f: impl Fn(&T) -> usize) -> Result<usize> {
    samples
        .first()
        .map(f)
        .filter(|dim| *dim > 0)
        .ok_or_else(|| anyhow!("no usable samples"))
}

fn mse<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f32>()
        / N.max(1) as f32
}

fn mse_vec(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    a.iter()
        .take(len)
        .zip(b.iter().take(len))
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f32>()
        / len as f32
}

fn mse_bytes(a: &[u8], b: &[u8]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    a.iter()
        .take(len)
        .zip(b.iter().take(len))
        .map(|(left, right)| ((*left as f32 / 255.0) - (*right as f32 / 255.0)).powi(2))
        .sum::<f32>()
        / len as f32
}

fn eye_current_loss(now: &Now, target: &EyeNextTarget) -> Option<f32> {
    eye_next_target_from_now(now).map(|current| mse_bytes(&current.rgb, &target.rgb))
}

fn ear_current_loss(now: &Now, target: &EarNextTarget) -> Option<f32> {
    ear_next_target_from_now(now).map(|current| mse_vec(&current.features, &target.features))
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len().max(1) as f32
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use netherwick_actions::ActionPrimitive;
    use netherwick_body::BodySense;
    use netherwick_core::Reward;
    use netherwick_experience::ExperienceLatent;
    use netherwick_now::SurpriseSense;
    use std::fs;

    #[tokio::test]
    async fn test_train_behavior_writes_evaluation_json() {
        let temp_dir = std::env::temp_dir().join(format!("netherwick_train_test_{}", now_ms()));
        let ledger_dir = temp_dir.join("ledger");
        let session_dir = ledger_dir.join("2026-06-24");
        fs::create_dir_all(&session_dir).unwrap();

        let checkpoint_dir = temp_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).unwrap();

        // Construct 5 mock transitions to have enough data for training and validation splits
        let mut transitions = Vec::new();
        for i in 0..5 {
            let transition = ExperienceTransition {
                id: uuid::Uuid::new_v4(),
                before_frame_id: uuid::Uuid::new_v4(),
                before: Now::blank(100 + i * 100, BodySense::default()),
                before_z: ExperienceLatent {
                    t_ms: 100 + i * 100,
                    z: vec![0.1; 4],
                    ..ExperienceLatent::default()
                },
                action: Some(ActionPrimitive::Stop),
                predicted_futures: Vec::new(),
                after: Now::blank(200 + i * 100, BodySense::default()),
                after_z: ExperienceLatent {
                    t_ms: 200 + i * 100,
                    z: vec![0.2; 4],
                    ..ExperienceLatent::default()
                },
                reward: Reward { value: 0.0 },
                surprise: SurpriseSense::default(),
                created_at_ms: 200 + i * 100,
            };
            transitions.push(transition);
        }

        let transitions_file = session_dir.join("transitions.jsonl");
        let mut content = String::new();
        for t in &transitions {
            content.push_str(&serde_json::to_string(t).unwrap());
            content.push('\n');
        }
        fs::write(&transitions_file, content).unwrap();

        let request = TrainBehaviorRequest {
            behavior: TrainableBehavior::Danger,
            ledger_path: ledger_dir,
            checkpoint_path: checkpoint_dir.clone(),
            epochs: 1,
            validation_split: 0.2,
            seed: 42,
        };

        let summary = train_behavior(request).await.unwrap();
        assert_eq!(summary.behavior, TrainableBehavior::Danger);

        // Verify that evaluation.json was created
        let eval_json_path = checkpoint_dir.join("evaluation.json");
        assert!(eval_json_path.exists());

        let eval_content = fs::read_to_string(&eval_json_path).unwrap();
        let report: BehaviorEvaluationReport = serde_json::from_str(&eval_content).unwrap();
        assert_eq!(report.behavior, TrainableBehavior::Danger);

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
