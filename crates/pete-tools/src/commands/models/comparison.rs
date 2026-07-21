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
