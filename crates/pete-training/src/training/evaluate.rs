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
