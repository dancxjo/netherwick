#[derive(Clone, Debug)]
struct RuntimeModelFlags<'a> {
    danger_checkpoint: Option<&'a str>,
    danger_mode: DangerMode,
    charge_checkpoint: Option<&'a str>,
    charge_mode: ChargeMode,
    action_value_checkpoint: Option<&'a str>,
    action_value_mode: ActionValueMode,
    future_checkpoint: Option<&'a str>,
    future_mode: FutureMode,
    eye_next_checkpoint: Option<&'a str>,
    eye_next_mode: EyeNextMode,
    ear_next_checkpoint: Option<&'a str>,
    ear_next_mode: EarNextMode,
    experience_checkpoint: Option<&'a str>,
    experience_mode: ExperienceMode,
}

impl<'a> From<&'a SimArgs> for RuntimeModelFlags<'a> {
    fn from(args: &'a SimArgs) -> Self {
        Self {
            danger_checkpoint: args.danger_checkpoint.as_deref(),
            danger_mode: args.danger_mode,
            charge_checkpoint: args.charge_checkpoint.as_deref(),
            charge_mode: args.charge_mode,
            action_value_checkpoint: args.action_value_checkpoint.as_deref(),
            action_value_mode: args.action_value_mode,
            future_checkpoint: args.future_checkpoint.as_deref(),
            future_mode: args.future_mode,
            eye_next_checkpoint: args.eye_next_checkpoint.as_deref(),
            eye_next_mode: args.eye_next_mode,
            ear_next_checkpoint: args.ear_next_checkpoint.as_deref(),
            ear_next_mode: args.ear_next_mode,
            experience_checkpoint: args.experience_checkpoint.as_deref(),
            experience_mode: args.experience_mode,
        }
    }
}

impl<'a> From<&'a EvalScenarioArgs> for RuntimeModelFlags<'a> {
    fn from(args: &'a EvalScenarioArgs) -> Self {
        Self {
            danger_checkpoint: args.danger_checkpoint.as_deref(),
            danger_mode: args.danger_mode,
            charge_checkpoint: args.charge_checkpoint.as_deref(),
            charge_mode: args.charge_mode,
            action_value_checkpoint: args.action_value_checkpoint.as_deref(),
            action_value_mode: args.action_value_mode,
            future_checkpoint: args.future_checkpoint.as_deref(),
            future_mode: args.future_mode,
            eye_next_checkpoint: args.eye_next_checkpoint.as_deref(),
            eye_next_mode: args.eye_next_mode,
            ear_next_checkpoint: args.ear_next_checkpoint.as_deref(),
            ear_next_mode: args.ear_next_mode,
            experience_checkpoint: args.experience_checkpoint.as_deref(),
            experience_mode: args.experience_mode,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct RuntimeModelLoadReport {
    requested_checkpoints: HashMap<String, Option<String>>,
    loaded_checkpoints: HashMap<String, String>,
    active_modes: HashMap<String, String>,
    blocked_model_infer: Vec<String>,
    warnings: Vec<String>,
}

fn load_runtime_models_from_flags(
    flags: &RuntimeModelFlags<'_>,
) -> Result<(Option<RuntimeModelStack>, RuntimeModelLoadReport)> {
    let mut report = RuntimeModelLoadReport::default();
    report.active_modes = model_modes_from_flags(flags);
    report.requested_checkpoints.insert(
        "danger".to_string(),
        flags.danger_checkpoint.map(ToOwned::to_owned),
    );
    report.requested_checkpoints.insert(
        "charge".to_string(),
        flags.charge_checkpoint.map(ToOwned::to_owned),
    );
    report.requested_checkpoints.insert(
        "action_value".to_string(),
        flags.action_value_checkpoint.map(ToOwned::to_owned),
    );
    report.requested_checkpoints.insert(
        "future".to_string(),
        flags.future_checkpoint.map(ToOwned::to_owned),
    );
    report.requested_checkpoints.insert(
        "eye_next".to_string(),
        flags.eye_next_checkpoint.map(ToOwned::to_owned),
    );
    report.requested_checkpoints.insert(
        "ear_next".to_string(),
        flags.ear_next_checkpoint.map(ToOwned::to_owned),
    );
    report.requested_checkpoints.insert(
        "experience".to_string(),
        flags.experience_checkpoint.map(ToOwned::to_owned),
    );

    if flags.danger_mode != DangerMode::ShadowInfer
        && flags.charge_mode != ChargeMode::ShadowInfer
        && flags.action_value_mode != ActionValueMode::ShadowInfer
        && flags.future_mode == FutureMode::Hardcoded
        && flags.eye_next_mode != EyeNextMode::ShadowInfer
        && flags.ear_next_mode != EarNextMode::ShadowInfer
        && flags.experience_mode == ExperienceMode::Off
    {
        return Ok((None, report));
    }
    let mut checkpoint_path = |behavior: &str, checkpoint: Option<&str>, enabled: bool| {
        if !enabled {
            return None;
        }
        match checkpoint {
            Some(checkpoint) if Path::new(checkpoint).exists() => {
                let path = PathBuf::from(checkpoint);
                println!("loaded {behavior} checkpoint: {}", path.display());
                report
                    .loaded_checkpoints
                    .insert(behavior.to_string(), checkpoint.to_string());
                Some(path)
            }
            Some(checkpoint) => {
                let warning =
                    format!("{behavior} inference disabled: checkpoint not found at {checkpoint}");
                println!("{warning}");
                report.warnings.push(warning);
                None
            }
            None => {
                let warning =
                    format!("{behavior} inference disabled: no --{behavior}-checkpoint provided");
                println!("{warning}");
                report.warnings.push(warning);
                None
            }
        }
    };
    let danger_path = checkpoint_path(
        "danger",
        flags.danger_checkpoint,
        flags.danger_mode == DangerMode::ShadowInfer,
    );
    let charge_path = checkpoint_path(
        "charge",
        flags.charge_checkpoint,
        flags.charge_mode == ChargeMode::ShadowInfer,
    );
    let action_value_path = checkpoint_path(
        "action_value",
        flags.action_value_checkpoint,
        flags.action_value_mode == ActionValueMode::ShadowInfer,
    );
    let future_path = checkpoint_path(
        "future",
        flags.future_checkpoint,
        flags.future_mode != FutureMode::Hardcoded,
    );
    let eye_next_path = checkpoint_path(
        "eye_next",
        flags.eye_next_checkpoint,
        flags.eye_next_mode == EyeNextMode::ShadowInfer,
    );
    let ear_next_path = checkpoint_path(
        "ear_next",
        flags.ear_next_checkpoint,
        flags.ear_next_mode == EarNextMode::ShadowInfer,
    );
    let experience_path = checkpoint_path(
        "experience",
        flags.experience_checkpoint,
        flags.experience_mode != ExperienceMode::Off,
    );
    if danger_path.is_none()
        && charge_path.is_none()
        && action_value_path.is_none()
        && future_path.is_none()
        && eye_next_path.is_none()
        && ear_next_path.is_none()
        && experience_path.is_none()
    {
        return Ok((None, report));
    }

    let mut models = RuntimeModelStack::with_shadow_checkpoints(
        danger_path.as_deref(),
        charge_path.as_deref(),
        action_value_path.as_deref(),
        future_path.as_deref(),
        eye_next_path.as_deref(),
        ear_next_path.as_deref(),
        experience_path.as_deref(),
    )?;
    if future_path.is_some() && flags.future_mode == FutureMode::ModelInfer {
        models.behaviors.future.regime = BehaviorRegime::ModelInfer;
    }
    if experience_path.is_some() && flags.experience_mode == ExperienceMode::ModelInfer {
        models.behaviors.experience.regime = BehaviorRegime::ModelInfer;
    }
    Ok((Some(models), report))
}

fn model_modes_from_flags(flags: &RuntimeModelFlags<'_>) -> HashMap<String, String> {
    HashMap::from([
        (
            "danger".to_string(),
            mode_name(flags.danger_mode).to_string(),
        ),
        (
            "charge".to_string(),
            mode_name(flags.charge_mode).to_string(),
        ),
        (
            "action_value".to_string(),
            mode_name(flags.action_value_mode).to_string(),
        ),
        (
            "future".to_string(),
            mode_name(flags.future_mode).to_string(),
        ),
        (
            "eye_next".to_string(),
            mode_name(flags.eye_next_mode).to_string(),
        ),
        (
            "ear_next".to_string(),
            mode_name(flags.ear_next_mode).to_string(),
        ),
        (
            "experience".to_string(),
            mode_name(flags.experience_mode).to_string(),
        ),
    ])
}

fn mode_name<T: std::fmt::Debug>(mode: T) -> &'static str {
    match format!("{mode:?}").as_str() {
        "Off" => "off",
        "Hardcoded" => "hardcoded",
        "ShadowInfer" => "shadow-infer",
        "ModelInfer" => "model-infer",
        _ => "unknown",
    }
}

async fn inspect_ledger(args: InspectLedgerArgs) -> Result<()> {
    let ledger = JsonlLedger::new(&args.ledger);
    let frames = ledger.recent(10).await?;
    if frames.is_empty() {
        println!("ledger is empty");
        return Ok(());
    }

    for frame in frames {
        print_frame(&frame);
    }
    Ok(())
}

fn is_transient_readonly_timeout(error: &AnyhowError) -> bool {
    is_transient_robot_timeout(error)
}

fn is_charging_busy_error(error: &AnyhowError) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<CockpitError>()
            .is_some_and(|cockpit| matches!(cockpit, CockpitError::Rejected { reason, .. } if reason == "charging_busy"))
    })
}

fn is_plain_busy_cockpit_error(error: &CockpitError) -> bool {
    matches!(error, CockpitError::Rejected { reason, .. } if reason == "busy")
}

fn is_reconnectable_cockpit_error(error: &AnyhowError) -> bool {
    if is_transient_robot_timeout(error) {
        return true;
    }
    error.chain().any(|cause| {
        cause
            .downcast_ref::<CockpitError>()
            .is_some_and(|cockpit| match cockpit {
                CockpitError::Io(_)
                | CockpitError::Serial(_)
                | CockpitError::WebSocket(_)
                | CockpitError::BadResponse(_)
                | CockpitError::Json(_)
                | CockpitError::FrameTooLarge { .. }
                | CockpitError::InvalidSession { .. }
                | CockpitError::SessionRequired => true,
                CockpitError::Rejected { reason, .. } => {
                    reason.contains("invalid_session")
                        || reason.contains("invalid_control_lease")
                        || reason.contains("control_lease_required")
                }
                CockpitError::Policy(_)
                | CockpitError::MotionStopped { .. }
                | CockpitError::MissedEvents { .. }
                | CockpitError::HandshakeRejected(_)
                | CockpitError::StaleHandshake { .. }
                | CockpitError::UnsafeHandshake(_) => false,
            })
    })
}

fn is_transient_robot_timeout(error: &AnyhowError) -> bool {
    let message = error.to_string().to_lowercase();
    if message.contains("timed out") || message.contains("operation timed out") {
        return true;
    }

    if let Some(io_error) = error.downcast_ref::<std::io::Error>() {
        return io_error.kind() == std::io::ErrorKind::TimedOut;
    }

    let mut current = error.source();
    while let Some(err) = current {
        if err.to_string().to_lowercase().contains("timed out")
            || err
                .to_string()
                .to_lowercase()
                .contains("operation timed out")
        {
            return true;
        }
        if let Some(io_error) = err.downcast_ref::<std::io::Error>() {
            if io_error.kind() == std::io::ErrorKind::TimedOut {
                return true;
            }
        }
        current = err.source();
    }
    false
}

async fn memory_inspect(args: MemoryInspectArgs) -> Result<()> {
    let ledger = JsonlLedger::new(&args.ledger);
    let frames = ledger.range(0, u64::MAX).await?;
    let report = place_memory_report_from_frames(&frames);
    print_memory_report(&args.ledger, frames.len(), &report);
    Ok(())
}

fn print_memory_report(source: &str, frame_count: usize, report: &PlaceMemoryReport) {
    println!("memory report: {source}");
    println!("  frames: {frame_count}");
    println!("  places_visited: {}", report.places_visited);
    println!("  coverage_m2: {:.2}", report.coverage_m2);
    println!("  novelty_mean: {:.3}", report.novelty_mean);
    print_place_cells("top danger cells", &report.top_danger_cells);
    print_place_cells("top charge cells", &report.top_charge_cells);
    print_place_cells("top social cells", &report.top_social_cells);
    if report.warnings.is_empty() {
        println!("  warnings: none");
    } else {
        println!("  warnings:");
        for warning in &report.warnings {
            println!("    - {warning}");
        }
    }
}

fn print_place_cells(label: &str, cells: &[pete_memory::PlaceCellSummary]) {
    println!("  {label}:");
    if cells.is_empty() {
        println!("    none");
        return;
    }
    for cell in cells {
        println!(
            "    - cell=({}, {}) center=({:.2}, {:.2}) score={:.3} visits={} confidence={:.3}",
            cell.x,
            cell.y,
            cell.center_x_m,
            cell.center_y_m,
            cell.score,
            cell.visit_count,
            cell.confidence
        );
    }
}

async fn run_train(command: TrainCommand) -> Result<()> {
    match command.model {
        TrainModel::Behavior(args) => train_behavior_command(args).await,
        TrainModel::Danger(args) => {
            train_behavior_command(TrainBehaviorArgs {
                behavior: "danger".to_string(),
                ledger: args.ledger,
                epochs: args.epochs,
                checkpoint: Some(args.checkpoint),
                validation_split: 0.2,
                seed: 7,
            })
            .await
        }
        TrainModel::Charge(args) => {
            train_behavior_command(TrainBehaviorArgs {
                behavior: "charge".to_string(),
                ledger: args.ledger,
                epochs: args.epochs,
                checkpoint: Some(args.checkpoint),
                validation_split: 0.2,
                seed: 7,
            })
            .await
        }
        TrainModel::ActionValue(args) => {
            train_behavior_command(TrainBehaviorArgs {
                behavior: "action-value".to_string(),
                ledger: args.ledger,
                epochs: args.epochs,
                checkpoint: Some(args.checkpoint),
                validation_split: 0.2,
                seed: 7,
            })
            .await
        }
        TrainModel::Future(args) => {
            train_behavior_command(TrainBehaviorArgs {
                behavior: "future".to_string(),
                ledger: args.ledger,
                epochs: args.epochs,
                checkpoint: Some(args.checkpoint),
                validation_split: 0.2,
                seed: 7,
            })
            .await
        }
        TrainModel::EyeNext(args) => {
            train_behavior_command(TrainBehaviorArgs {
                behavior: "eye-next".to_string(),
                ledger: args.ledger,
                epochs: args.epochs,
                checkpoint: Some(args.checkpoint),
                validation_split: 0.2,
                seed: 7,
            })
            .await
        }
        TrainModel::EarNext(args) => {
            train_behavior_command(TrainBehaviorArgs {
                behavior: "ear-next".to_string(),
                ledger: args.ledger,
                epochs: args.epochs,
                checkpoint: Some(args.checkpoint),
                validation_split: 0.2,
                seed: 7,
            })
            .await
        }
        TrainModel::Experience(args) => {
            train_behavior_command(TrainBehaviorArgs {
                behavior: "experience".to_string(),
                ledger: args.ledger,
                epochs: args.epochs,
                checkpoint: Some(args.checkpoint),
                validation_split: 0.2,
                seed: 7,
            })
            .await
        }
        TrainModel::LatentRoundTrip(args) => run_train_latent_round_trip(args).await,
        TrainModel::UnifiedExperience(args) => run_train_unified_experience(args).await,
        TrainModel::Virtual(args) => run_train_virtual(args).await,
    }
}

async fn train_behavior_command(args: TrainBehaviorArgs) -> Result<()> {
    let behavior: TrainableBehavior = args.behavior.parse()?;
    let checkpoint = args
        .checkpoint
        .unwrap_or_else(|| default_checkpoint(&behavior).to_string());
    let summary = train_behavior(TrainBehaviorRequest {
        behavior: behavior.clone(),
        ledger_path: args.ledger.into(),
        checkpoint_path: checkpoint.clone().into(),
        epochs: args.epochs,
        validation_split: args.validation_split,
        seed: args.seed,
    })
    .await?;
    println!(
        "{} training complete: {} transitions, {} train samples, {} eval samples, {} epochs, {} samples seen, metrics {}",
        behavior,
        summary.transition_count,
        summary.train_sample_count,
        summary.eval_sample_count,
        summary.epochs,
        summary.samples_seen,
        summary.metrics_path.display()
    );
    println!(
        "saved {} checkpoint: {}",
        behavior,
        summary.checkpoint_path.display()
    );
    if let Some(last_loss) = summary.last_loss {
        println!("last_loss: {:.6}", last_loss);
    }
    println!("best_loss: {:?}", summary.best_loss);
    print_evaluation_report(&summary.evaluation)?;
    Ok(())
}

async fn run_train_latent_round_trip(args: TrainLatentRoundTripArgs) -> Result<()> {
    let report = train_latent_round_trip(TrainLatentRoundTripRequest {
        ledger_path: args.ledger.into(),
        checkpoint_path: args.checkpoint.clone().into(),
        report_path: args.report.clone().into(),
        epochs: args.epochs,
        validation_split: args.validation_split,
        seed: args.seed,
        z_dim: args.z_dim,
        codebook_size: args.codebook_size,
    })
    .await?;
    println!(
        "latent round-trip training complete: {} transitions, {} epochs, report {}",
        report.transition_count, report.epochs, args.report
    );
    println!(
        "reconstruction_loss_mean: {:.6}",
        report.reconstruction.trained_decoder_loss_mean
    );
    for predictor in &report.predictors {
        println!(
            "{} predictor: model_loss={:.6} stasis_loss={:.6} improvement={:?} predictive={}",
            predictor.encoder,
            predictor.model_loss_mean,
            predictor.stasis_loss_mean,
            predictor.improvement_ratio,
            predictor.predictive
        );
    }
    if let Some(codebook) = &report.codebook {
        println!(
            "codebook: {} codes, {} used, {} dead",
            codebook.code_count, codebook.used_codes, codebook.dead_codes
        );
    }
    println!("verdict: {}", report.verdict);
    for warning in &report.warnings {
        println!("warning: {warning}");
    }
    Ok(())
}

async fn run_train_unified_experience(args: TrainUnifiedExperienceArgs) -> Result<()> {
    let report = train_unified_experience(TrainUnifiedExperienceRequest {
        ledger_path: args.ledger.into(),
        checkpoint_path: args.checkpoint.clone().into(),
        report_path: args.report.clone().into(),
        epochs: args.epochs,
        validation_split: args.validation_split,
        seed: args.seed,
        z_dim: args.z_dim,
        teacher_dim: args.teacher_dim,
    })
    .await?;
    println!(
        "unified Experience training complete: {} examples, {} transitions, {} epochs, checkpoint {}, report {}",
        report.example_count,
        report.transition_count,
        report.epochs,
        args.checkpoint,
        args.report
    );
    println!(
        "reconstruction_loss={:.6} zero_loss={:.6} reconstructive={}",
        report.reconstruction.total_loss_mean,
        report.reconstruction.zero_loss_mean,
        report.reconstruction.reconstructive
    );
    println!(
        "trained_loss={:?} copy_current_loss={:?} random_loss={:?} mechanical_loss={:?}",
        report.baselines.trained_loss_mean,
        report.baselines.copy_current_loss_mean,
        report.baselines.random_projection_loss_mean,
        report.baselines.mechanical_instant_loss_mean
    );
    println!("verdict: {}", report.verdict);
    for warning in &report.warnings {
        println!("warning: {warning}");
    }
    Ok(())
}

async fn run_evaluate(command: EvaluateCommand) -> Result<()> {
    match command.model {
        EvaluateModel::Behavior(args) => {
            let behavior: TrainableBehavior = args.behavior.parse()?;
            let checkpoint = args
                .checkpoint
                .unwrap_or_else(|| default_checkpoint(&behavior).to_string());
            let report = evaluate_behavior(EvaluateBehaviorRequest {
                behavior,
                ledger_path: args.ledger.into(),
                checkpoint_path: checkpoint.clone().into(),
                max_samples: args.max_samples,
            })
            .await?;
            let checkpoint_evaluation_path = Path::new(&checkpoint).join("evaluation.json");
            let json = serde_json::to_string_pretty(&report)?;
            std::fs::write(&checkpoint_evaluation_path, &json)?;
            println!(
                "Saved checkpoint evaluation report to {}",
                checkpoint_evaluation_path.display()
            );
            if let Some(out_path) = &args.out {
                std::fs::write(out_path, &json)?;
                println!("Saved evaluation report to {}", out_path);
            }
            print_evaluation_report(&report)
        }
    }
}

fn run_promote(command: PromoteCommand) -> Result<()> {
    match command.model {
        PromoteModel::Behavior(args) => {
            let behavior: TrainableBehavior = args.behavior.parse()?;
            let checkpoint = args
                .checkpoint
                .unwrap_or_else(|| default_checkpoint(&behavior).to_string());
            let regime = match args.mode {
                PromoteMode::ShadowInfer => BehaviorRegime::ShadowInfer,
                PromoteMode::ModelInfer => BehaviorRegime::ModelInfer,
                PromoteMode::ShadowTrain => BehaviorRegime::ShadowTrain,
            };
            promote_behavior_config(
                behavior.clone(),
                checkpoint.clone().into(),
                Path::new(&args.config),
                regime,
            )?;
            println!(
                "promoted {} in {}: regime {:?}, checkpoint {}",
                behavior, args.config, regime, checkpoint
            );
            Ok(())
        }
    }
}

fn default_checkpoint(behavior: &TrainableBehavior) -> &'static str {
    match behavior {
        TrainableBehavior::Danger => "data/models/danger_v0",
        TrainableBehavior::Charge => "data/models/charge_v0",
        TrainableBehavior::ActionValue => "data/models/action_value_v0",
        TrainableBehavior::EyeNext => "data/models/eye_next_v0",
        TrainableBehavior::EarNext => "data/models/ear_next_v0",
        TrainableBehavior::Experience => "data/models/experience_v0",
        TrainableBehavior::Future => "data/models/future_v0",
    }
}

fn print_evaluation_report(report: &pete_training::BehaviorEvaluationReport) -> Result<()> {
    println!("evaluation behavior: {}", report.behavior);
    println!("checkpoint: {}", report.checkpoint_path.display());
    println!("sample_count: {}", report.sample_count);
    println!("model_loss_mean: {:.6}", report.model_loss_mean);
    println!("hardcoded_loss_mean: {:?}", report.hardcoded_loss_mean);
    println!("selected_loss_mean: {:?}", report.selected_loss_mean);
    println!(
        "model_better_than_hardcoded: {:?}",
        report.model_better_than_hardcoded
    );
    println!("improvement_ratio: {:?}", report.improvement_ratio);
    println!("recommendation: {:?}", report.recommendation);
    for warning in &report.warnings {
        println!("warning: {warning}");
    }
    Ok(())
}

const DEFAULT_REGISTRY_PATH: &str = "data/models/registry.json";
