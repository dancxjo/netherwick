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

pub async fn train_latent_round_trip(
    request: TrainLatentRoundTripRequest,
) -> Result<TrainLatentRoundTripReport> {
    let transitions = load_transitions(&request.ledger_path).await?;
    let transition_count = transitions.len();
    let (train, eval) = split_transitions(transitions, request.validation_split, request.seed);
    let eval_transitions = if eval.is_empty() { &train } else { &eval };
    let checkpoints = LatentRoundTripCheckpoints {
        experience: request.checkpoint_path.join("experience"),
        future_trained: request.checkpoint_path.join("future-trained"),
        future_random: request.checkpoint_path.join("future-random"),
    };

    let experience_train = experience_samples(&train);
    let (input_dim, decode_lengths) = experience_train
        .first()
        .map(|(_, input, target, _)| (input.flat_features().len(), target.feature_lengths()))
        .ok_or_else(|| anyhow!("no usable experience samples for latent round-trip training"))?;
    let z_dim = request.z_dim.clamp(2, input_dim.max(2));
    let mut autoencoder = ExperienceAutoencoderTrainer::new(input_dim, z_dim, decode_lengths);
    for _epoch in 0..request.epochs {
        for (_, input, target, _) in &experience_train {
            if input.flat_features().len() == autoencoder.input_dim()
                && target.feature_lengths() == autoencoder.decode_lengths()
            {
                autoencoder.train_step(input, target)?;
            }
        }
    }
    autoencoder.save_checkpoint(&checkpoints.experience)?;

    let reconstruction = evaluate_trained_reconstruction(&autoencoder, eval_transitions)?;
    let architecture = latent_architecture_report(
        "ledger-replay",
        "trainable-autoencoder",
        &checkpoints.experience,
        &checkpoints.future_trained,
        "compact body/memory/drive/prediction/range-depth/audio-summary features",
        &experience_train
            .iter()
            .map(|(_, input, target, _)| (input, target))
            .collect::<Vec<_>>(),
        z_dim,
    )?;
    let trained_train = trained_latent_future_samples(&autoencoder, &train)?;
    let trained_eval = trained_latent_future_samples(&autoencoder, eval_transitions)?;
    let trained_report = train_and_evaluate_future_latents(
        "trainable-autoencoder",
        trained_train.clone(),
        trained_eval,
        request.epochs,
        &checkpoints.future_trained,
        "next trained latent",
    )?;

    let mut random_train_encoder = RandomProjectionExperienceEncoder::new(z_dim, request.seed);
    let mut random_eval_encoder = RandomProjectionExperienceEncoder::new(z_dim, request.seed);
    let random_train = encoded_future_samples(&mut random_train_encoder, &train)?;
    let random_eval = encoded_future_samples(&mut random_eval_encoder, eval_transitions)?;
    let random_report = train_and_evaluate_future_latents(
        "random-projection",
        random_train,
        random_eval,
        request.epochs,
        &checkpoints.future_random,
        "next random-projected latent",
    )?;

    let codebook = if let Some(codebook_size) = request.codebook_size {
        let mut quantizer = CodebookQuantizer::from_latents(
            &trained_train
                .iter()
                .map(|(_, input, _)| input.latent.z.clone())
                .collect::<Vec<_>>(),
            codebook_size,
        );
        for (_, input, target) in &trained_train {
            let code_id = quantizer.encode(&input.latent.z);
            let decoded = quantizer.decode(code_id);
            let _ = mse_vec(&decoded, target);
        }
        Some(quantizer.report())
    } else {
        None
    };

    let predictors = vec![trained_report, random_report];
    let baseline_comparisons = latent_baseline_comparisons(
        &predictors,
        "trainable-autoencoder",
        "random-projection",
        None,
    );
    let mut warnings = Vec::new();
    if transition_count < 50 {
        warnings.push(format!(
            "insufficient data: {transition_count} transitions is below the conservative 50-transition floor"
        ));
    }
    if predictors.iter().all(|report| !report.predictive) {
        warnings.push("no encoder beat the stasis baseline on held-out prediction".to_string());
    }
    let verdict = if predictors
        .iter()
        .any(|report| report.encoder == "trainable-autoencoder" && report.predictive)
    {
        "trained latent is predictive on held-out replay".to_string()
    } else if predictors.iter().any(|report| report.predictive) {
        "a latent is predictive, but the trained encoder is not yet strongest".to_string()
    } else {
        "latent remains compact but not proven predictive".to_string()
    };

    let report = TrainLatentRoundTripReport {
        schema_version: 2,
        input_source: format!("ledger:{}", request.ledger_path.display()),
        architecture,
        transition_count,
        train_transition_count: train.len(),
        eval_transition_count: eval_transitions.len(),
        epochs: request.epochs,
        z_dim,
        checkpoints,
        reconstruction,
        predictors,
        baseline_comparisons,
        codebook,
        verdict,
        warnings,
    };
    if let Some(parent) = request.report_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&request.report_path, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub async fn train_unified_experience(
    request: TrainUnifiedExperienceRequest,
) -> Result<TrainUnifiedExperienceReport> {
    let transitions = load_transitions(&request.ledger_path).await?;
    let transition_count = transitions.len();
    let mut examples = unified_examples_from_transitions(&transitions, request.teacher_dim)?;
    let example_count = examples.len();
    if example_count == 0 {
        bail!("no usable unified Experience teacher-vector examples");
    }
    let coverage = unified_modality_coverage(&examples);
    let (train, eval) = split_samples(
        std::mem::take(&mut examples),
        request.validation_split,
        request.seed,
    );
    let eval_examples = if eval.is_empty() { &train } else { &eval };
    let first = train
        .first()
        .ok_or_else(|| anyhow!("no training examples for unified Experience"))?;
    let input_dim = first.input.flat_features().len();
    let decode_lengths = first.target.feature_lengths();
    let z_dim = request.z_dim.clamp(2, input_dim.max(2));
    let mut autoencoder = ExperienceAutoencoderTrainer::new(input_dim, z_dim, decode_lengths);
    for _epoch in 0..request.epochs {
        for sample in &train {
            if sample.input.flat_features().len() == autoencoder.input_dim()
                && sample.target.feature_lengths() == autoencoder.decode_lengths()
            {
                autoencoder.train_step(&sample.input, &sample.target)?;
            }
        }
    }
    autoencoder.save_checkpoint(&request.checkpoint_path)?;

    let reconstruction = evaluate_unified_reconstruction(&autoencoder, eval_examples)?;
    let trained_train = unified_trained_future_samples(&autoencoder, &train)?;
    let trained_eval = unified_trained_future_samples(&autoencoder, eval_examples)?;
    let future_checkpoint_path = request.checkpoint_path.join("future-trained");
    let trained_report = train_and_evaluate_future_latents(
        "unified-experience-latent",
        trained_train.clone(),
        trained_eval,
        request.epochs,
        &future_checkpoint_path,
        "next unified Experience latent",
    )?;
    let future_input_dim = first_dim(&trained_train, |(_, input, _)| input.flat_features().len())?;
    let future_trainer =
        FutureNetTrainer::load_checkpoint(&future_checkpoint_path, future_input_dim, z_dim)?;
    let mut random_train_encoder = RandomProjectionExperienceEncoder::new(z_dim, request.seed);
    let mut random_eval_encoder = RandomProjectionExperienceEncoder::new(z_dim, request.seed);
    let random_report = train_and_evaluate_future_latents(
        "random-projection",
        unified_encoded_future_samples(&mut random_train_encoder, &train)?,
        unified_encoded_future_samples(&mut random_eval_encoder, eval_examples)?,
        request.epochs,
        &request.checkpoint_path.join("future-random"),
        "next random-projected unified latent",
    )?;
    let mechanical_report = train_and_evaluate_future_latents(
        "mechanical-instant",
        unified_mechanical_future_samples(&train),
        unified_mechanical_future_samples(eval_examples),
        request.epochs,
        &request.checkpoint_path.join("future-mechanical-instant"),
        "next mechanical Instant",
    )?;
    let predictors = vec![
        trained_report.clone(),
        random_report.clone(),
        mechanical_report.clone(),
    ];
    let baselines = UnifiedBaselineReport {
        copy_current_loss_mean: Some(trained_report.stasis_loss_mean),
        random_projection_loss_mean: Some(random_report.model_loss_mean),
        mechanical_instant_loss_mean: Some(mechanical_report.model_loss_mean),
        trained_loss_mean: Some(trained_report.model_loss_mean),
        trained_beats_copy_current: trained_report.model_loss_mean
            < trained_report.stasis_loss_mean,
        trained_beats_random_projection: trained_report.model_loss_mean
            < random_report.model_loss_mean,
        trained_beats_mechanical_instant: trained_report.model_loss_mean
            < mechanical_report.model_loss_mean,
    };
    let learned_loop = unified_learned_loop_report(
        &autoencoder,
        &future_trainer,
        eval_examples,
        &baselines,
        random_report.model_loss_mean,
        mechanical_report.model_loss_mean,
    )?;
    let mut warnings = Vec::new();
    if example_count < 50 {
        warnings.push(format!(
            "insufficient data: {example_count} examples is below the conservative 50-example floor"
        ));
    }
    for slot in &coverage {
        if slot.present_count == 0 {
            warnings.push(format!(
                "teacher slot {} was explicitly masked as missing for every example",
                slot.slot
            ));
        }
    }
    let collapsed_latent = unified_latent_variance(&autoencoder, eval_examples)? < 1.0e-6;
    if collapsed_latent {
        warnings.push(
            "learned unified Experience latent appears collapsed on held-out examples".to_string(),
        );
    }
    let verdict = if reconstruction.reconstructive && trained_report.predictive && !collapsed_latent
    {
        "unified Experience latent is reconstructive and predictive".to_string()
    } else if reconstruction.reconstructive {
        "unified Experience latent reconstructs teacher/sensor heads but is not yet predictive"
            .to_string()
    } else {
        "unified Experience latent is not yet proven reconstructive or predictive".to_string()
    };
    let report = TrainUnifiedExperienceReport {
        schema_version: 1,
        input_source: format!("ledger:{}", request.ledger_path.display()),
        example_count,
        train_example_count: train.len(),
        eval_example_count: eval_examples.len(),
        transition_count,
        epochs: request.epochs,
        teacher_dim: request.teacher_dim,
        latent_dim: z_dim,
        checkpoint_path: request.checkpoint_path.clone(),
        future_checkpoint_path,
        instant: UnifiedInstantReport {
            representation: "UnifiedExperienceInstant".to_string(),
            teacher_slots: UNIFIED_TEACHER_SLOTS.iter().map(|slot| slot.name.to_string()).collect(),
            input_dim,
            mask_dim: UNIFIED_TEACHER_SLOTS.len(),
            target_dim: first.target.flat_features().len(),
            assembly: "fixed teacher-vector slots plus explicit presence mask; missing modalities stay masked instead of disappearing".to_string(),
        },
        modality_coverage: coverage,
        reconstruction,
        predictors,
        learned_loop,
        baselines,
        verdict,
        warnings,
    };
    if let Some(parent) = request.report_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&request.report_path, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
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
