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
