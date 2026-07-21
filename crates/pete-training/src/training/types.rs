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
            Self::Experience => "experience.no_latent_yet",
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

#[derive(Clone, Debug)]
pub struct TrainLatentRoundTripRequest {
    pub ledger_path: PathBuf,
    pub checkpoint_path: PathBuf,
    pub report_path: PathBuf,
    pub epochs: usize,
    pub validation_split: f32,
    pub seed: u64,
    pub z_dim: usize,
    pub codebook_size: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct TrainUnifiedExperienceRequest {
    pub ledger_path: PathBuf,
    pub checkpoint_path: PathBuf,
    pub report_path: PathBuf,
    pub epochs: usize,
    pub validation_split: f32,
    pub seed: u64,
    pub z_dim: usize,
    pub teacher_dim: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainLatentRoundTripReport {
    pub schema_version: u32,
    pub input_source: String,
    pub architecture: LatentRoundTripArchitectureReport,
    pub transition_count: usize,
    pub train_transition_count: usize,
    pub eval_transition_count: usize,
    pub epochs: usize,
    pub z_dim: usize,
    pub checkpoints: LatentRoundTripCheckpoints,
    pub reconstruction: LatentReconstructionReport,
    pub predictors: Vec<LatentPredictorReport>,
    pub baseline_comparisons: LatentBaselineComparisons,
    pub codebook: Option<CodebookUsageReport>,
    pub verdict: String,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentRoundTripCheckpoints {
    pub experience: PathBuf,
    pub future_trained: PathBuf,
    pub future_random: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentRoundTripArchitectureReport {
    pub pipeline: Vec<String>,
    pub teacher_vectors: Vec<TeacherVectorReport>,
    pub instant: MechanicalInstantReport,
    pub encoder: ExperienceEncoderReport,
    pub owned_latent: OwnedExperienceLatentReport,
    pub heads: Vec<LatentHeadReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeacherVectorReport {
    pub name: String,
    pub source: String,
    pub purpose: String,
    pub dim: usize,
    pub sample_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MechanicalInstantReport {
    pub representation: String,
    pub assembly: String,
    pub sample_count: usize,
    pub input_dim: usize,
    pub decode_target_dim: usize,
    pub decode_target_kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperienceEncoderReport {
    pub name: String,
    pub input_dim: usize,
    pub z_dim: usize,
    pub checkpoint_path: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OwnedExperienceLatentReport {
    pub name: String,
    pub owner: String,
    pub dim: usize,
    pub teacher_independent: bool,
    pub evidence: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentHeadReport {
    pub name: String,
    pub target: String,
    pub checkpoint_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentReconstructionReport {
    pub sample_count: usize,
    pub trained_decoder_loss_mean: f32,
    pub zero_decoder_loss_mean: f32,
    pub target_kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentPredictorReport {
    pub encoder: String,
    pub target_kind: String,
    pub train_sample_count: usize,
    pub eval_sample_count: usize,
    pub latent_dim: usize,
    pub target_dim: usize,
    pub model_loss_mean: f32,
    pub stasis_loss_mean: f32,
    pub improvement_ratio: Option<f32>,
    pub predictive: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentBaselineComparisons {
    pub trained_encoder: String,
    pub copy_current_loss_mean: Option<f32>,
    pub random_projection_loss_mean: Option<f32>,
    pub evolved_vector_loss_mean: Option<f32>,
    pub trained_loss_mean: Option<f32>,
    pub trained_beats_copy_current: bool,
    pub trained_beats_random_projection: bool,
    pub trained_beats_evolved_vector: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainUnifiedExperienceReport {
    pub schema_version: u32,
    pub input_source: String,
    pub example_count: usize,
    pub train_example_count: usize,
    pub eval_example_count: usize,
    pub transition_count: usize,
    pub epochs: usize,
    pub teacher_dim: usize,
    pub latent_dim: usize,
    pub checkpoint_path: PathBuf,
    pub future_checkpoint_path: PathBuf,
    pub instant: UnifiedInstantReport,
    pub modality_coverage: Vec<UnifiedModalityCoverage>,
    pub reconstruction: UnifiedReconstructionReport,
    pub predictors: Vec<LatentPredictorReport>,
    pub learned_loop: UnifiedLearnedLoopReport,
    pub baselines: UnifiedBaselineReport,
    pub verdict: String,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedInstantReport {
    pub representation: String,
    pub teacher_slots: Vec<String>,
    pub input_dim: usize,
    pub mask_dim: usize,
    pub target_dim: usize,
    pub assembly: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedModalityCoverage {
    pub slot: String,
    pub source: String,
    pub purpose: String,
    pub dim: usize,
    pub placeholder: bool,
    pub present_count: usize,
    pub missing_count: usize,
    pub coverage: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedReconstructionReport {
    pub sample_count: usize,
    pub total_loss_mean: f32,
    pub zero_loss_mean: f32,
    pub head_losses: BTreeMap<String, f32>,
    pub zero_head_losses: BTreeMap<String, f32>,
    pub reconstructive: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedBaselineReport {
    pub copy_current_loss_mean: Option<f32>,
    pub random_projection_loss_mean: Option<f32>,
    pub mechanical_instant_loss_mean: Option<f32>,
    pub trained_loss_mean: Option<f32>,
    pub trained_beats_copy_current: bool,
    pub trained_beats_random_projection: bool,
    pub trained_beats_mechanical_instant: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedLearnedLoopReport {
    pub canonical_instant: String,
    pub canonical_latent: String,
    pub prediction: String,
    pub surprise: String,
    pub sample_count: usize,
    pub reconstruction_loss_mean: f32,
    pub prediction_loss_mean: f32,
    pub combined_surprise_mean: f32,
    pub confidence_mean: f32,
    pub records: Vec<UnifiedExperienceLoopRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedExperienceLoopRecord {
    pub t_ms: TimeMs,
    pub offset_ms: TimeMs,
    pub encoded_latent: Vec<f32>,
    pub predicted_next_latent: Vec<f32>,
    pub actual_next_latent: Vec<f32>,
    pub reconstruction_loss: f32,
    pub prediction_loss: f32,
    pub combined_surprise: f32,
    pub confidence: f32,
    pub teacher_coverage: f32,
    pub missing_modality_mask: Vec<f32>,
    pub baseline_comparisons: UnifiedExperienceLoopBaselines,
    pub surprise: ExperienceSurprise,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedExperienceLoopBaselines {
    pub copy_current_prediction_loss: f32,
    pub random_projection_prediction_loss: Option<f32>,
    pub mechanical_instant_prediction_loss: Option<f32>,
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
