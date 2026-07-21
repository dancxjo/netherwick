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

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ModelRegistry {
    schema_version: u32,
    entries: Vec<ModelRegistryEntry>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self {
            schema_version: 1,
            entries: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ModelRegistryEntry {
    schema_version: u32,
    name: String,
    behavior: TrainableBehavior,
    checkpoint: String,
    created_at: Option<String>,
    training: ModelTrainingRecord,
    reports: ModelReportRecord,
    scenario_names: Vec<String>,
    metrics: ModelMetricsSummary,
    allowed_modes: Vec<String>,
    status: ModelStatus,
    warnings: Vec<String>,
    notes: Vec<String>,
    parent_model: Option<String>,
    git_commit: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ModelTrainingRecord {
    ledger: Option<String>,
    command: Option<String>,
    epochs: Option<usize>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ModelReportRecord {
    behavior: Option<String>,
    scenario: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    comparison: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct ModelMetricsSummary {
    behavior_loss: Option<f32>,
    scenario_success_rate: Option<f32>,
    collision_rate: Option<f32>,
    battery_delta: Option<f32>,
    fallback_count: Option<usize>,
    episodes: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum ModelStatus {
    Registered,
    Shadow,
    Inference,
    Retired,
    Rejected,
}

impl ModelStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Registered => "registered",
            Self::Shadow => "shadow",
            Self::Inference => "inference",
            Self::Retired => "retired",
            Self::Rejected => "rejected",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ScenarioComparisonReport {
    schema_version: u32,
    baseline_report_path: String,
    candidate_report_path: String,
    baseline_scenario: String,
    candidate_scenario: String,
    baseline_episodes: usize,
    candidate_episodes: usize,
    compared_metrics: ScenarioComparisonMetrics,
    deltas: HashMap<String, f32>,
    recommendation: ScenarioComparisonRecommendation,
    warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct ScenarioComparisonMetrics {
    success_rate: MetricComparison,
    collision_rate: MetricComparison,
    mean_collisions_per_episode: MetricComparison,
    mean_safety_interventions: MetricComparison,
    model_fallbacks: MetricComparison,
    action_selector_fallbacks: MetricComparison,
    action_selector_guard_yields: MetricComparison,
    mean_battery_delta: MetricComparison,
    stuck_count: MetricComparison,
    recovery_attempts: MetricComparison,
    repeated_trap_count: MetricComparison,
    recovery_success_rate: MetricComparison,
    mean_recovery_ticks: MetricComparison,
    mean_stuck_duration: MetricComparison,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct MetricComparison {
    baseline: Option<f32>,
    candidate: Option<f32>,
    delta: Option<f32>,
    regression: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ScenarioComparisonRecommendation {
    PassCandidate,
    NeedsMoreEval,
    RegressionDetected,
    InsufficientData,
}

impl ScenarioComparisonRecommendation {
    fn as_str(self) -> &'static str {
        match self {
            Self::PassCandidate => "pass_candidate",
            Self::NeedsMoreEval => "needs_more_eval",
            Self::RegressionDetected => "regression_detected",
            Self::InsufficientData => "insufficient_data",
        }
    }
}

fn model_register(args: ModelRegisterArgs) -> Result<()> {
    let behavior: TrainableBehavior = args.behavior.parse()?;
    let mut registry = load_model_registry(Path::new(&args.registry))?;
    if registry
        .entries
        .iter()
        .any(|entry| entry.name == args.name && entry.behavior == behavior)
        && !args.overwrite
    {
        anyhow::bail!(
            "model {} for {} already exists; pass --overwrite to replace it",
            args.name,
            behavior
        );
    }

    let behavior_report = args
        .behavior_report
        .as_deref()
        .map(load_behavior_report)
        .transpose()?;
    let scenario_report = args
        .scenario_report
        .as_deref()
        .map(load_scenario_report)
        .transpose()?;
    let mut warnings = Vec::new();
    if !Path::new(&args.checkpoint).exists() {
        warnings.push(format!("checkpoint missing: {}", args.checkpoint));
    }
    if let Some(path) = &args.behavior_report {
        if !Path::new(path).exists() {
            warnings.push(format!("behavior report missing: {path}"));
        }
    } else if behavior == TrainableBehavior::Danger {
        warnings.push("danger registration lacks a behavior evaluation report".to_string());
    }
    if let Some(path) = &args.scenario_report {
        if !Path::new(path).exists() {
            warnings.push(format!("scenario report missing: {path}"));
        }
    }

    let metrics = ModelMetricsSummary {
        behavior_loss: behavior_report
            .as_ref()
            .map(|report| report.model_loss_mean),
        scenario_success_rate: scenario_report
            .as_ref()
            .map(|report| report.summary.success_rate),
        collision_rate: scenario_report
            .as_ref()
            .map(|report| report.summary.collision_rate),
        battery_delta: scenario_report
            .as_ref()
            .map(|report| report.summary.mean_battery_delta),
        fallback_count: scenario_report
            .as_ref()
            .map(|report| report.summary.model_fallbacks),
        episodes: scenario_report.as_ref().map(|report| report.episodes),
    };
    let entry = ModelRegistryEntry {
        schema_version: 1,
        name: args.name.clone(),
        behavior: behavior.clone(),
        checkpoint: args.checkpoint,
        created_at: Some(Utc::now().to_rfc3339()),
        training: ModelTrainingRecord {
            ledger: args.training_ledger,
            command: args.training_command.or_else(|| Some(command_summary())),
            epochs: None,
        },
        reports: ModelReportRecord {
            behavior: args.behavior_report,
            scenario: args.scenario_report,
            comparison: args.comparison_report,
        },
        scenario_names: scenario_report
            .as_ref()
            .map(|report| vec![report.scenario.clone()])
            .unwrap_or_default(),
        metrics,
        allowed_modes: allowed_modes_for_status(&behavior, ModelStatus::Registered),
        status: ModelStatus::Registered,
        warnings,
        notes: args.notes,
        parent_model: args.parent,
        git_commit: current_git_commit(),
    };

    registry
        .entries
        .retain(|entry| !(entry.name == args.name && entry.behavior == behavior));
    registry.entries.push(entry);
    write_model_registry(Path::new(&args.registry), &registry)?;
    println!(
        "registered {} model {} in {}",
        behavior, args.name, args.registry
    );
    Ok(())
}

fn model_promote(args: ModelPromoteArgs) -> Result<()> {
    let behavior: TrainableBehavior = args.behavior.parse()?;
    let path = Path::new(&args.registry);
    let mut registry = load_model_registry(path)?;
    let Some(index) = registry
        .entries
        .iter()
        .position(|entry| entry.name == args.name && entry.behavior == behavior)
    else {
        anyhow::bail!("model {} for {} is not registered", args.name, behavior);
    };

    let candidate_path = args
        .candidate_report
        .clone()
        .or_else(|| registry.entries[index].reports.scenario.clone());
    let baseline_report = args
        .baseline_report
        .as_deref()
        .map(load_scenario_report)
        .transpose()?;
    let candidate_report = candidate_path
        .as_deref()
        .map(load_scenario_report)
        .transpose()?;
    let comparison_report = args
        .comparison_report
        .as_deref()
        .or_else(|| registry.entries[index].reports.comparison.as_deref())
        .map(load_scenario_comparison_report)
        .transpose()?;
    let comparison = match (&comparison_report, &baseline_report, &candidate_report) {
        (Some(comparison), _, _) => Some(comparison.clone()),
        (None, Some(baseline), Some(candidate)) => Some(compare_scenario_reports(
            args.baseline_report.as_deref().unwrap_or("baseline"),
            candidate_path.as_deref().unwrap_or("candidate"),
            baseline,
            candidate,
        )),
        _ => None,
    };
    let decision = promotion_gate(
        &registry.entries[index],
        args.target,
        baseline_report.as_ref(),
        candidate_report.as_ref(),
        comparison.as_ref(),
        args.allow_safety_critical_inference,
    );

    if !decision.allowed {
        for warning in decision.warnings {
            println!("warning: {warning}");
        }
        anyhow::bail!(
            "promotion refused: {} {} -> {}",
            behavior,
            args.name,
            args.target.as_str()
        );
    }

    {
        let entry = &mut registry.entries[index];
        entry.status = args.target;
        entry.allowed_modes = allowed_modes_for_status(&behavior, args.target);
        entry.warnings = merge_warnings(&entry.warnings, &decision.warnings);
        entry.notes.extend(args.notes);
        if let Some(path) = args.candidate_report {
            entry.reports.scenario = Some(path);
        }
        if let Some(path) = args.comparison_report {
            entry.reports.comparison = Some(path);
        }
        if let Some(report) = candidate_report {
            entry.scenario_names = vec![report.scenario.clone()];
            entry.metrics.scenario_success_rate = Some(report.summary.success_rate);
            entry.metrics.collision_rate = Some(report.summary.collision_rate);
            entry.metrics.battery_delta = Some(report.summary.mean_battery_delta);
            entry.metrics.fallback_count = Some(report.summary.model_fallbacks);
            entry.metrics.episodes = Some(report.episodes);
        }
    }
    write_model_registry(path, &registry)?;
    println!(
        "promoted {} model {} to {}",
        behavior,
        args.name,
        args.target.as_str()
    );
    if let Some(comparison) = comparison {
        print_scenario_comparison(&comparison);
    }
    Ok(())
}

fn compare_scenario_reports_command(args: CompareScenarioReportsArgs) -> Result<()> {
    let baseline = load_scenario_report(&args.baseline)?;
    let candidate = load_scenario_report(&args.candidate)?;
    let comparison =
        compare_scenario_reports(&args.baseline, &args.candidate, &baseline, &candidate);
    let out = args.out.unwrap_or_else(|| {
        default_comparison_report_path(args.name.as_deref(), &baseline, &candidate)
    });
    write_scenario_comparison_report(Path::new(&out), &comparison)?;
    print_scenario_comparison(&comparison);
    println!("comparison report written: {out}");
    Ok(())
}

fn load_model_registry(path: &Path) -> Result<ModelRegistry> {
    if !path.exists() {
        return Ok(ModelRegistry::default());
    }
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn write_model_registry(path: &Path, registry: &ModelRegistry) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let temp_path = path.with_extension("json.tmp");
    fs::write(&temp_path, serde_json::to_vec_pretty(registry)?)?;
    fs::rename(temp_path, path)?;
    Ok(())
}

fn load_behavior_report(path: &str) -> Result<pete_training::BehaviorEvaluationReport> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn load_scenario_report(path: &str) -> Result<ScenarioEvaluationReport> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn load_scenario_comparison_report(path: &str) -> Result<ScenarioComparisonReport> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn write_scenario_comparison_report(path: &Path, report: &ScenarioComparisonReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(path, serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

fn compare_scenario_reports(
    baseline_path: &str,
    candidate_path: &str,
    baseline: &ScenarioEvaluationReport,
    candidate: &ScenarioEvaluationReport,
) -> ScenarioComparisonReport {
    let metrics = ScenarioComparisonMetrics {
        success_rate: metric_cmp(
            Some(baseline.summary.success_rate),
            Some(candidate.summary.success_rate),
            RegressionDirection::LowerIsWorse,
            0.01,
        ),
        collision_rate: metric_cmp(
            Some(baseline.summary.collision_rate),
            Some(candidate.summary.collision_rate),
            RegressionDirection::HigherIsWorse,
            0.005,
        ),
        mean_collisions_per_episode: metric_cmp(
            Some(baseline.summary.mean_collisions_per_episode),
            Some(candidate.summary.mean_collisions_per_episode),
            RegressionDirection::HigherIsWorse,
            0.05,
        ),
        mean_safety_interventions: metric_cmp(
            Some(baseline.summary.mean_safety_interventions),
            Some(candidate.summary.mean_safety_interventions),
            RegressionDirection::HigherIsWorse,
            0.05,
        ),
        model_fallbacks: metric_cmp(
            Some(baseline.summary.model_fallbacks as f32),
            Some(candidate.summary.model_fallbacks as f32),
            RegressionDirection::HigherIsWorse,
            0.0,
        ),
        action_selector_fallbacks: metric_cmp(
            Some(baseline.summary.action_selector_fallbacks as f32),
            Some(candidate.summary.action_selector_fallbacks as f32),
            RegressionDirection::HigherIsWorse,
            0.0,
        ),
        action_selector_guard_yields: metric_cmp(
            Some(baseline.summary.action_selector_guard_yields as f32),
            Some(candidate.summary.action_selector_guard_yields as f32),
            RegressionDirection::HigherIsWorse,
            0.0,
        ),
        mean_battery_delta: metric_cmp(
            Some(baseline.summary.mean_battery_delta),
            Some(candidate.summary.mean_battery_delta),
            RegressionDirection::LowerIsWorse,
            0.02,
        ),
        stuck_count: metric_cmp(
            Some(baseline.summary.stuck_count as f32),
            Some(candidate.summary.stuck_count as f32),
            RegressionDirection::HigherIsWorse,
            0.0,
        ),
        recovery_attempts: metric_cmp(
            Some(baseline.summary.recovery_attempts as f32),
            Some(candidate.summary.recovery_attempts as f32),
            RegressionDirection::HigherIsWorse,
            0.0,
        ),
        repeated_trap_count: metric_cmp(
            Some(baseline.summary.repeated_trap_count as f32),
            Some(candidate.summary.repeated_trap_count as f32),
            RegressionDirection::HigherIsWorse,
            0.0,
        ),
        recovery_success_rate: metric_cmp(
            baseline.summary.recovery_success_rate,
            candidate.summary.recovery_success_rate,
            RegressionDirection::LowerIsWorse,
            0.05,
        ),
        mean_recovery_ticks: metric_cmp(
            baseline.summary.mean_recovery_ticks,
            candidate.summary.mean_recovery_ticks,
            RegressionDirection::HigherIsWorse,
            1.0,
        ),
        mean_stuck_duration: metric_cmp(
            baseline.summary.mean_stuck_duration,
            candidate.summary.mean_stuck_duration,
            RegressionDirection::HigherIsWorse,
            50.0,
        ),
    };
    let mut warnings = comparison_warnings(baseline, candidate, &metrics);
    let recommendation = comparison_recommendation(baseline, candidate, &metrics, &warnings);
    if matches!(
        recommendation,
        ScenarioComparisonRecommendation::NeedsMoreEval
    ) && baseline.episodes < 10
    {
        warnings.push(format!(
            "candidate has only {} episodes; run at least 10 for promotion confidence",
            candidate.episodes
        ));
    }
    let deltas = comparison_deltas(&metrics);
    ScenarioComparisonReport {
        schema_version: 1,
        baseline_report_path: baseline_path.to_string(),
        candidate_report_path: candidate_path.to_string(),
        baseline_scenario: baseline.scenario.clone(),
        candidate_scenario: candidate.scenario.clone(),
        baseline_episodes: baseline.episodes,
        candidate_episodes: candidate.episodes,
        compared_metrics: metrics,
        deltas,
        recommendation,
        warnings,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RegressionDirection {
    HigherIsWorse,
    LowerIsWorse,
}

fn metric_cmp(
    baseline: Option<f32>,
    candidate: Option<f32>,
    direction: RegressionDirection,
    tolerance: f32,
) -> MetricComparison {
    let delta = baseline
        .zip(candidate)
        .map(|(baseline, candidate)| candidate - baseline);
    let regression = delta
        .map(|delta| match direction {
            RegressionDirection::HigherIsWorse => delta > tolerance,
            RegressionDirection::LowerIsWorse => delta < -tolerance,
        })
        .unwrap_or(false);
    MetricComparison {
        baseline,
        candidate,
        delta,
        regression,
    }
}

fn comparison_warnings(
    baseline: &ScenarioEvaluationReport,
    candidate: &ScenarioEvaluationReport,
    metrics: &ScenarioComparisonMetrics,
) -> Vec<String> {
    let mut warnings = Vec::new();
    if baseline.scenario != candidate.scenario {
        warnings.push(format!(
            "scenario mismatch: baseline={} candidate={}",
            baseline.scenario, candidate.scenario
        ));
    }
    if baseline.episodes != candidate.episodes {
        warnings.push(format!(
            "episode count mismatch: baseline={} candidate={}",
            baseline.episodes, candidate.episodes
        ));
    }
    for (name, metric) in [
        ("success_rate", &metrics.success_rate),
        ("collision_rate", &metrics.collision_rate),
        (
            "mean_collisions_per_episode",
            &metrics.mean_collisions_per_episode,
        ),
        (
            "mean_safety_interventions",
            &metrics.mean_safety_interventions,
        ),
        ("model_fallbacks", &metrics.model_fallbacks),
        (
            "action_selector_fallbacks",
            &metrics.action_selector_fallbacks,
        ),
        (
            "action_selector_guard_yields",
            &metrics.action_selector_guard_yields,
        ),
        ("mean_battery_delta", &metrics.mean_battery_delta),
        ("stuck_count", &metrics.stuck_count),
        ("recovery_attempts", &metrics.recovery_attempts),
        ("repeated_trap_count", &metrics.repeated_trap_count),
        ("recovery_success_rate", &metrics.recovery_success_rate),
        ("mean_recovery_ticks", &metrics.mean_recovery_ticks),
        ("mean_stuck_duration", &metrics.mean_stuck_duration),
    ] {
        if metric.regression {
            warnings.push(format!(
                "{name} regressed by {:.4}",
                metric.delta.unwrap_or_default()
            ));
        }
    }
    warnings
}

fn comparison_recommendation(
    baseline: &ScenarioEvaluationReport,
    candidate: &ScenarioEvaluationReport,
    metrics: &ScenarioComparisonMetrics,
    warnings: &[String],
) -> ScenarioComparisonRecommendation {
    if baseline.episodes < 3 || candidate.episodes < 3 {
        return ScenarioComparisonRecommendation::InsufficientData;
    }
    if warnings.iter().any(|warning| warning.contains("regressed")) {
        return ScenarioComparisonRecommendation::RegressionDetected;
    }
    if baseline.scenario != candidate.scenario {
        return ScenarioComparisonRecommendation::NeedsMoreEval;
    }
    if candidate.episodes < 10 {
        return ScenarioComparisonRecommendation::NeedsMoreEval;
    }
    if metrics.success_rate.delta.unwrap_or_default() >= -0.01
        && metrics.collision_rate.delta.unwrap_or_default() <= 0.005
        && metrics.mean_battery_delta.delta.unwrap_or_default() >= -0.02
    {
        ScenarioComparisonRecommendation::PassCandidate
    } else {
        ScenarioComparisonRecommendation::NeedsMoreEval
    }
}

fn comparison_deltas(metrics: &ScenarioComparisonMetrics) -> HashMap<String, f32> {
    let mut deltas = HashMap::new();
    for (name, metric) in [
        ("success_rate", &metrics.success_rate),
        ("collision_rate", &metrics.collision_rate),
        (
            "mean_collisions_per_episode",
            &metrics.mean_collisions_per_episode,
        ),
        (
            "mean_safety_interventions",
            &metrics.mean_safety_interventions,
        ),
        ("model_fallbacks", &metrics.model_fallbacks),
        (
            "action_selector_fallbacks",
            &metrics.action_selector_fallbacks,
        ),
        (
            "action_selector_guard_yields",
            &metrics.action_selector_guard_yields,
        ),
        ("mean_battery_delta", &metrics.mean_battery_delta),
        ("stuck_count", &metrics.stuck_count),
        ("recovery_attempts", &metrics.recovery_attempts),
        ("repeated_trap_count", &metrics.repeated_trap_count),
        ("recovery_success_rate", &metrics.recovery_success_rate),
        ("mean_recovery_ticks", &metrics.mean_recovery_ticks),
        ("mean_stuck_duration", &metrics.mean_stuck_duration),
    ] {
        if let Some(delta) = metric.delta {
            deltas.insert(name.to_string(), delta);
        }
    }
    deltas
}

fn default_comparison_report_path(
    name: Option<&str>,
    baseline: &ScenarioEvaluationReport,
    candidate: &ScenarioEvaluationReport,
) -> String {
    let name = name
        .map(safe_report_name)
        .unwrap_or_else(|| format!("{}-{}-candidate", baseline.scenario, candidate.scenario));
    format!("data/reports/comparisons/{name}.json")
}

fn safe_report_name(name: &str) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

fn print_scenario_comparison(comparison: &ScenarioComparisonReport) {
    println!("recommendation: {}", comparison.recommendation.as_str());
    for (name, delta) in &comparison.deltas {
        println!("{name}_delta: {delta:.6}");
    }
    for warning in &comparison.warnings {
        println!("warning: {warning}");
    }
}

#[derive(Clone, Debug)]
struct PromotionGateDecision {
    allowed: bool,
    warnings: Vec<String>,
}

fn promotion_gate(
    entry: &ModelRegistryEntry,
    target: ModelStatus,
    baseline: Option<&ScenarioEvaluationReport>,
    candidate: Option<&ScenarioEvaluationReport>,
    comparison: Option<&ScenarioComparisonReport>,
    allow_safety_critical_inference: bool,
) -> PromotionGateDecision {
    let mut warnings = Vec::new();
    if !Path::new(&entry.checkpoint).exists() {
        warnings.push(format!("checkpoint missing: {}", entry.checkpoint));
    }
    if matches!(
        target,
        ModelStatus::Retired | ModelStatus::Rejected | ModelStatus::Registered
    ) {
        return PromotionGateDecision {
            allowed: true,
            warnings,
        };
    }
    if target == ModelStatus::Shadow {
        if entry.reports.scenario.is_none() {
            warnings.push("shadow requires a scenario evaluation report".to_string());
        }
        return PromotionGateDecision {
            allowed: warnings.is_empty(),
            warnings,
        };
    }
    if target != ModelStatus::Inference {
        return PromotionGateDecision {
            allowed: false,
            warnings: vec!["unknown promotion target".to_string()],
        };
    }
    if is_safety_critical_behavior(&entry.behavior) && !allow_safety_critical_inference {
        warnings.push(
            "safety-critical inference requires --allow-safety-critical-inference".to_string(),
        );
    }
    let Some(candidate) = candidate else {
        warnings.push("inference promotion requires a candidate scenario report".to_string());
        return PromotionGateDecision {
            allowed: false,
            warnings,
        };
    };
    if candidate.episodes < 10 {
        warnings.push(format!(
            "not enough scenario episodes for inference: {} < 10",
            candidate.episodes
        ));
    }
    if candidate.summary.model_fallbacks > 0 {
        warnings.push(format!(
            "model fallback count is not zero: {}",
            candidate.summary.model_fallbacks
        ));
    }
    if candidate.summary.collision_rate > 0.05 {
        warnings.push(format!(
            "candidate collision rate too high: {:.4}",
            candidate.summary.collision_rate
        ));
    }
    if let Some(comparison) = comparison {
        if comparison.recommendation == ScenarioComparisonRecommendation::RegressionDetected {
            warnings.push("candidate scenario report regressed against baseline".to_string());
        }
    } else if baseline.is_none() && is_safety_critical_behavior(&entry.behavior) {
        warnings
            .push("safety-critical inference requires baseline comparison evidence".to_string());
    }
    match entry.behavior {
        TrainableBehavior::Danger => {
            if let Some(comparison) = comparison {
                if comparison
                    .compared_metrics
                    .collision_rate
                    .delta
                    .unwrap_or_default()
                    > 0.002
                {
                    warnings.push(format!(
                        "danger collision rate worse than baseline by {:.4}",
                        comparison
                            .compared_metrics
                            .collision_rate
                            .delta
                            .unwrap_or_default()
                    ));
                }
            }
        }
        TrainableBehavior::Charge => {
            if candidate.summary.success_rate < 0.70 {
                warnings.push(format!(
                    "charger success rate below threshold: {:.3}",
                    candidate.summary.success_rate
                ));
            }
            if candidate.summary.mean_battery_delta < -0.05 {
                warnings.push(format!(
                    "charger battery delta unacceptable: {:.3}",
                    candidate.summary.mean_battery_delta
                ));
            }
        }
        TrainableBehavior::ActionValue => {
            if candidate.scenario != "mixed-room" {
                warnings
                    .push("action-value inference requires mixed-room scenario eval".to_string());
            }
        }
        TrainableBehavior::Future => {
            warnings.push(
                "future inference is not a direct motor-control promotion; keep hardcoded fallback available"
                    .to_string(),
            );
        }
        TrainableBehavior::EyeNext | TrainableBehavior::EarNext | TrainableBehavior::Experience => {
            if entry.behavior == TrainableBehavior::Experience {
                warnings.push(
                    "experience inference changes latent encoding; only use where it cannot directly command motors"
                        .to_string(),
                );
            }
        }
    }
    PromotionGateDecision {
        allowed: warnings.is_empty(),
        warnings,
    }
}

fn allowed_modes_for_status(behavior: &TrainableBehavior, status: ModelStatus) -> Vec<String> {
    let mut modes = vec!["off".to_string(), "hardcoded".to_string()];
    if matches!(status, ModelStatus::Shadow | ModelStatus::Inference) {
        modes.push("shadow-infer".to_string());
    }
    if status == ModelStatus::Inference
        && matches!(
            behavior,
            TrainableBehavior::Future
                | TrainableBehavior::EyeNext
                | TrainableBehavior::EarNext
                | TrainableBehavior::Experience
        )
    {
        modes.push("model-infer".to_string());
    }
    modes
}

fn merge_warnings(left: &[String], right: &[String]) -> Vec<String> {
    let mut warnings = left.to_vec();
    for warning in right {
        if !warnings.contains(warning) {
            warnings.push(warning.clone());
        }
    }
    warnings
}

fn command_summary() -> String {
    std::env::args().collect::<Vec<_>>().join(" ")
}

fn current_git_commit() -> Option<String> {
    let output = ProcessCommand::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|text| !text.is_empty())
}

fn model_status() -> Result<()> {
    print_registry_status(Path::new(DEFAULT_REGISTRY_PATH))?;
    println!();
    println!("registered models:");
    for model in MODEL_REGISTRY {
        println!("  - {model}");
    }

    let config_path = Path::new("configs/models.toml");
    println!();
    println!("models config: {}", config_path.display());
    let config = match load_models_config(config_path) {
        Ok(config) => Some(config),
        Err(error) => {
            println!("  unavailable: {error}");
            None
        }
    };

    println!();
    println!("behavior instrument panel:");
    for behavior in trainable_behaviors() {
        print_behavior_status(behavior, config.as_ref())?;
    }

    println!();
    println!("checkpoint directories:");
    print_model_directories(Path::new("data/models"))?;
    Ok(())
}

fn print_registry_status(path: &Path) -> Result<()> {
    let registry = load_model_registry(path)?;
    println!("model registry: {}", path.display());
    if registry.entries.is_empty() {
        println!("  no registry entries");
        return Ok(());
    }
    println!(
        "{:<14} {:<24} {:<10} {:<32} {:<32} recommendation/warnings",
        "behavior", "name", "status", "checkpoint", "scenario report"
    );
    for entry in registry.entries {
        let report = entry.reports.scenario.as_deref().unwrap_or("-");
        let recommendation = registry_recommendation(&entry);
        println!(
            "{:<14} {:<24} {:<10} {:<32} {:<32} {}",
            entry.behavior,
            entry.name,
            entry.status.as_str(),
            entry.checkpoint,
            report,
            recommendation
        );
    }
    Ok(())
}

fn registry_recommendation(entry: &ModelRegistryEntry) -> String {
    if !entry.warnings.is_empty() {
        return entry.warnings.join("; ");
    }
    match entry.status {
        ModelStatus::Registered => "run scenario eval, then promote to shadow".to_string(),
        ModelStatus::Shadow => {
            if is_safety_critical_behavior(&entry.behavior) {
                "collect baseline comparison before inference".to_string()
            } else {
                "eligible for cautious inference review".to_string()
            }
        }
        ModelStatus::Inference => "allowed for configured inference surfaces".to_string(),
        ModelStatus::Retired => "retired".to_string(),
        ModelStatus::Rejected => "rejected".to_string(),
    }
}

fn trainable_behaviors() -> &'static [TrainableBehavior] {
    &[
        TrainableBehavior::Danger,
        TrainableBehavior::Charge,
        TrainableBehavior::ActionValue,
        TrainableBehavior::Future,
        TrainableBehavior::EyeNext,
        TrainableBehavior::EarNext,
        TrainableBehavior::Experience,
    ]
}

fn print_behavior_status(
    behavior: &TrainableBehavior,
    config: Option<&pete_behaviors::BehaviorRegistryConfig>,
) -> Result<()> {
    let key = behavior.config_key();
    let configured = config.and_then(|config| config.behavior.get(key));
    let checkpoint = configured
        .and_then(|entry| entry.checkpoint.as_deref())
        .unwrap_or_else(|| default_checkpoint(behavior));
    let checkpoint_path = Path::new(checkpoint);
    let checkpoint_present = checkpoint_path.is_dir();
    let metadata = read_json_optional(&checkpoint_path.join("metadata.json"))?;
    let evaluation = read_json_optional(&checkpoint_path.join("evaluation.json"))?;
    let latest_metric = read_latest_metric(&checkpoint_path.join("metrics.jsonl"))?;

    println!("  - {}", behavior);
    println!(
        "      checkpoint: {} ({})",
        checkpoint_path.display(),
        if checkpoint_present {
            "present"
        } else {
            "missing"
        }
    );
    println!(
        "      samples_seen: {}",
        json_field(&metadata, "samples_seen").unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "      best_loss: {}",
        json_field(&metadata, "best_loss").unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "      latest_eval_loss: {}",
        json_field(&evaluation, "model_loss_mean").unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "      hardcoded_loss: {}",
        json_field(&evaluation, "hardcoded_loss_mean").unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "      improvement_ratio: {}",
        json_field(&evaluation, "improvement_ratio").unwrap_or_else(|| "unknown".to_string())
    );
    println!(
        "      current regime: {}",
        configured
            .map(|entry| format!("{:?}", entry.regime))
            .unwrap_or_else(|| "unconfigured".to_string())
    );
    println!(
        "      recommended regime: {}",
        recommended_regime(behavior, evaluation.as_ref())
    );
    println!(
        "      safety-critical? {}",
        if is_safety_critical_behavior(behavior) {
            "yes"
        } else {
            "no"
        }
    );
    println!(
        "      last metrics timestamp: {}",
        latest_metric
            .as_ref()
            .and_then(|metric| json_field(&Some(metric.clone()), "t_ms"))
            .unwrap_or_else(|| "unknown".to_string())
    );

    if let Some(entry) = configured {
        println!("      hardcoded: {}", entry.hardcoded);
        println!("      model: {}", entry.model.as_deref().unwrap_or("none"));
        println!("      fallback: {:?}", entry.fallback);
    } else {
        println!("      hardcoded: {}", behavior.default_hardcoded_id());
        println!("      model: {}", behavior.default_model_id());
        println!("      fallback: UseHardcoded");
    }
    if let Some(warnings) = evaluation
        .as_ref()
        .and_then(|json| json.get("warnings"))
        .and_then(Value::as_array)
    {
        for warning in warnings.iter().filter_map(Value::as_str) {
            println!("      warning: {warning}");
        }
    }
    Ok(())
}

fn read_json_optional(path: &Path) -> Result<Option<Value>> {
    if !path.exists() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_slice(&fs::read(path)?)?))
}

fn read_latest_metric(path: &Path) -> Result<Option<Value>> {
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(path)?;
    let Some(line) = text.lines().rev().find(|line| !line.trim().is_empty()) else {
        return Ok(None);
    };
    Ok(Some(serde_json::from_str(line)?))
}

fn json_field(json: &Option<Value>, field: &str) -> Option<String> {
    json.as_ref().and_then(|json| {
        json.get(field).map(|value| match value {
            Value::String(text) => text.clone(),
            Value::Null => "null".to_string(),
            other => other.to_string(),
        })
    })
}

fn recommended_regime(behavior: &TrainableBehavior, evaluation: Option<&Value>) -> String {
    let recommendation = evaluation
        .and_then(|json| json.get("recommendation"))
        .and_then(Value::as_str);
    match recommendation {
        Some("promote_to_model_infer") if is_safety_critical_behavior(behavior) => {
            "shadow_infer (model_infer blocked for safety-critical behavior)".to_string()
        }
        Some("promote_to_model_infer") => "model_infer".to_string(),
        Some("shadow_infer") => "shadow_infer".to_string(),
        Some("shadow_train") => "shadow_train".to_string(),
        Some("keep_hardcoded") => "hardcoded".to_string(),
        Some("reject_checkpoint") => "hardcoded (reject checkpoint)".to_string(),
        Some(other) => format!("unknown ({other})"),
        None => "unknown".to_string(),
    }
}

fn is_safety_critical_behavior(behavior: &TrainableBehavior) -> bool {
    matches!(
        behavior,
        TrainableBehavior::Danger
            | TrainableBehavior::ActionValue
            | TrainableBehavior::Experience
            | TrainableBehavior::Future
    )
}

fn print_model_directories(path: &Path) -> Result<()> {
    if !path.exists() {
        println!("  missing {}", path.display());
        return Ok(());
    }
    let mut directories = fs::read_dir(path)?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            entry
                .file_type()
                .ok()
                .filter(|file_type| file_type.is_dir())
                .map(|_| entry.path())
        })
        .collect::<Vec<_>>();
    directories.sort();
    if directories.is_empty() {
        println!("  none below {}", path.display());
        return Ok(());
    }
    for directory in directories {
        let metadata = directory.join("metadata.json").exists();
        let evaluation = directory.join("evaluation.json").exists();
        let metrics = directory.join("metrics.jsonl").exists();
        println!(
            "  - {} (metadata={}, evaluation={}, metrics={})",
            directory.display(),
            metadata,
            evaluation,
            metrics
        );
    }
    Ok(())
}

fn print_frame(frame: &ExperienceFrame) {
    println!("frame {} @ {}ms", frame.id, frame.t_ms);
    println!("  summary: {}", frame.summary_text());
    println!("  action: {:?}", frame.chosen_action);
    println!("  recalls: {}", frame.memory_recall.len());
    println!("  recollections: {}", frame.recollections.len());
    let embodied = frame.embodied_context();
    println!(
        "  embodied_summary: {}",
        if embodied.summary.trim().is_empty() {
            "none"
        } else {
            embodied.summary.trim()
        }
    );
    println!(
        "  embodied_experience_id: {}",
        embodied
            .experience_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "none".to_string())
    );
    println!("  sensation_count: {}", embodied.sensations.len());
    println!("  impression_count: {}", embodied.impressions.len());
    println!(
        "  derived_sensation_count: {}",
        embodied.derived_sensation_count()
    );
    println!("  lineage_edge_count: {}", embodied.lineage.len());
    if embodied.lineage.is_empty() {
        println!("  lineage_graph: none");
    } else {
        println!("  lineage_graph:");
        for edge in embodied.lineage.iter().take(8) {
            println!("    - {} -> {}", edge.parent_id, edge.child_id);
        }
    }
    println!(
        "  embodied_prediction_count: {}",
        embodied.predictions.len()
    );
    println!(
        "  embodied_memory_link_count: {}",
        embodied.memory_links.len()
    );
    if let Some(experience) = frame.experiences.last() {
        println!("  experience: {}", experience.text);
    }
    if let Some(transcript) = &frame.now.ear.transcript {
        println!("  heard: {}", transcript);
    }
    if let Some(eye_frame) = &frame.now.eye_frame {
        println!(
            "  eye_frame: {}x{} ({:?}) source={:?}",
            eye_frame.width,
            eye_frame.height,
            eye_frame.format,
            eye_frame.source.as_deref().unwrap_or("none")
        );
    }
}

