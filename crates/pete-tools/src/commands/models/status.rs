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
