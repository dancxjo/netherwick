async fn run_train_virtual(args: TrainVirtualArgs) -> Result<()> {
    println!("Starting virtual training pipeline...");
    println!("Ledger: {}", args.ledger);
    println!("Out Dir: {}", args.out_dir);

    // 1. Generate run report
    let run_report = generate_virtual_report(&args.ledger).await?;
    println!("Run report generated successfully.");

    // Create out_dir
    fs::create_dir_all(&args.out_dir)?;

    // 2. Train selected behaviors
    let behaviors = vec![
        TrainableBehavior::Danger,
        TrainableBehavior::Charge,
        TrainableBehavior::EyeNext,
        TrainableBehavior::EarNext,
        TrainableBehavior::Future,
    ];

    let mut trained_summaries = HashMap::new();
    for behavior in &behaviors {
        let checkpoint_path = Path::new(&args.out_dir).join(behavior.config_key());
        println!("Training behavior model: {:?}", behavior);
        let summary = train_behavior(TrainBehaviorRequest {
            behavior: behavior.clone(),
            ledger_path: PathBuf::from(&args.ledger),
            checkpoint_path,
            epochs: args.epochs,
            validation_split: 0.2,
            seed: 7,
        })
        .await?;
        trained_summaries.insert(behavior.clone(), summary);
    }

    // 3. Run scenario evaluations
    println!("Running baseline scenario evaluation (all models Off)...");
    let baseline_report_path = Path::new(&args.out_dir).join("baseline-scenario.json");
    let baseline_args = EvalScenarioArgs {
        scenario: ScenarioArg::MixedRoom,
        episodes: 10,
        steps: 100,
        seed: 7,
        tick_ms: 100,
        out: Some(baseline_report_path.to_string_lossy().to_string()),
        ledger: None,
        capture_root: None,
        memory_report: false,
        danger_checkpoint: None,
        danger_mode: DangerMode::Off,
        charge_checkpoint: None,
        charge_mode: ChargeMode::Off,
        action_value_checkpoint: None,
        action_value_mode: ActionValueMode::Off,
        future_checkpoint: None,
        future_mode: FutureMode::Hardcoded,
        eye_next_checkpoint: None,
        eye_next_mode: EyeNextMode::Off,
        ear_next_checkpoint: None,
        ear_next_mode: EarNextMode::Off,
        experience_checkpoint: None,
        experience_mode: ExperienceMode::Off,
        action_selector: CliActionSelectorMode::Baseline,
        llm: LlmArgs::default(),
    };
    run_eval_scenario(baseline_args).await?;
    let baseline_report = load_scenario_report(&baseline_report_path.to_string_lossy())?;

    println!("Running candidate scenario evaluation (new models ShadowInfer)...");
    let candidate_report_path = Path::new(&args.out_dir).join("candidate-scenario.json");
    let candidate_args = EvalScenarioArgs {
        scenario: ScenarioArg::MixedRoom,
        episodes: 10,
        steps: 100,
        seed: 7,
        tick_ms: 100,
        out: Some(candidate_report_path.to_string_lossy().to_string()),
        ledger: None,
        capture_root: None,
        memory_report: false,
        danger_checkpoint: Some(
            Path::new(&args.out_dir)
                .join("danger")
                .to_string_lossy()
                .to_string(),
        ),
        danger_mode: DangerMode::ShadowInfer,
        charge_checkpoint: Some(
            Path::new(&args.out_dir)
                .join("charge")
                .to_string_lossy()
                .to_string(),
        ),
        charge_mode: ChargeMode::ShadowInfer,
        action_value_checkpoint: None,
        action_value_mode: ActionValueMode::Off,
        future_checkpoint: Some(
            Path::new(&args.out_dir)
                .join("future")
                .to_string_lossy()
                .to_string(),
        ),
        future_mode: FutureMode::ShadowInfer,
        eye_next_checkpoint: Some(
            Path::new(&args.out_dir)
                .join("eye_next")
                .to_string_lossy()
                .to_string(),
        ),
        eye_next_mode: EyeNextMode::ShadowInfer,
        ear_next_checkpoint: Some(
            Path::new(&args.out_dir)
                .join("ear_next")
                .to_string_lossy()
                .to_string(),
        ),
        ear_next_mode: EarNextMode::ShadowInfer,
        experience_checkpoint: None,
        experience_mode: ExperienceMode::Off,
        action_selector: CliActionSelectorMode::Baseline,
        llm: LlmArgs::default(),
    };
    run_eval_scenario(candidate_args).await?;
    let candidate_report = load_scenario_report(&candidate_report_path.to_string_lossy())?;

    // Compare scenario reports
    let comparison_report_path = Path::new(&args.out_dir).join("comparison-scenario.json");
    let comparison = compare_scenario_reports(
        &baseline_report_path.to_string_lossy(),
        &candidate_report_path.to_string_lossy(),
        &baseline_report,
        &candidate_report,
    );
    write_scenario_comparison_report(&comparison_report_path, &comparison)?;
    println!(
        "Evaluation comparison recommendation: {}",
        comparison.recommendation.as_str()
    );

    // 4. Update/register models and run promotion gates
    let registry_path = Path::new("data/models/registry.json");
    let mut model_statuses = HashMap::new();
    let timestamp = Utc::now().format("%Y%m%d_%H%M").to_string();

    for behavior in &behaviors {
        let name = format!("{}_virtual_{}", behavior.config_key(), timestamp);
        let checkpoint = Path::new(&args.out_dir)
            .join(behavior.config_key())
            .to_string_lossy()
            .to_string();
        let behavior_report = Path::new(&checkpoint).join("evaluation.json");

        println!("Registering candidate model {}...", name);
        model_register(ModelRegisterArgs {
            behavior: behavior.cli_name().to_string(),
            checkpoint: checkpoint.clone(),
            training_ledger: Some(args.ledger.clone()),
            training_command: Some("just train virtual".to_string()),
            behavior_report: Some(behavior_report.to_string_lossy().to_string()),
            scenario_report: Some(candidate_report_path.to_string_lossy().to_string()),
            comparison_report: Some(comparison_report_path.to_string_lossy().to_string()),
            name: name.clone(),
            notes: vec!["Automatically trained via virtual pipeline".to_string()],
            parent: None,
            registry: registry_path.to_string_lossy().to_string(),
            overwrite: true,
        })?;

        // Load the registry to get the entry we just registered
        let registry = load_model_registry(registry_path)?;
        let entry = registry
            .entries
            .iter()
            .find(|e| e.name == name && e.behavior == *behavior)
            .unwrap()
            .clone();

        // Determine recommended promotion status
        // First test Inference promotion
        let inference_decision = promotion_gate(
            &entry,
            ModelStatus::Inference,
            Some(&baseline_report),
            Some(&candidate_report),
            Some(&comparison),
            args.allow_safety_critical_inference,
        );

        let mut new_status = ModelStatus::Registered;
        let mut recommended_action = "keep hardcoded".to_string();
        let mut warnings = Vec::new();

        if inference_decision.allowed {
            new_status = ModelStatus::Inference;
            recommended_action = "inference".to_string();
        } else {
            // Test Shadow promotion
            let shadow_decision = promotion_gate(
                &entry,
                ModelStatus::Shadow,
                Some(&baseline_report),
                Some(&candidate_report),
                Some(&comparison),
                args.allow_safety_critical_inference,
            );
            if shadow_decision.allowed {
                new_status = ModelStatus::Shadow;
                recommended_action = "shadow".to_string();
            } else {
                // Collect warnings for why promotion failed
                warnings.extend(inference_decision.warnings);
                warnings.extend(shadow_decision.warnings);
            }
        }

        // Apply promotion if recommended status is higher than Registered
        if new_status != ModelStatus::Registered {
            println!("Promoting model {} to {}...", name, new_status.as_str());
            model_promote(ModelPromoteArgs {
                behavior: behavior.cli_name().to_string(),
                name: name.clone(),
                target: new_status,
                baseline_report: Some(baseline_report_path.to_string_lossy().to_string()),
                candidate_report: Some(candidate_report_path.to_string_lossy().to_string()),
                comparison_report: Some(comparison_report_path.to_string_lossy().to_string()),
                registry: registry_path.to_string_lossy().to_string(),
                allow_safety_critical_inference: args.allow_safety_critical_inference,
                notes: vec!["Automatically promoted via virtual pipeline".to_string()],
            })?;
        }

        let loss = trained_summaries.get(behavior).and_then(|s| s.last_loss);

        model_statuses.insert(
            behavior.config_key().to_string(),
            ModelTrainingStatus {
                name,
                trained: true,
                previous_status: "registered".to_string(),
                new_status: new_status.as_str().to_string(),
                recommended_action,
                warnings,
                loss,
                baseline_collision_rate: Some(baseline_report.summary.collision_rate),
                candidate_collision_rate: Some(candidate_report.summary.collision_rate),
                baseline_success_rate: Some(baseline_report.summary.success_rate),
                candidate_success_rate: Some(candidate_report.summary.success_rate),
            },
        );
    }

    // 5. Write final consolidated training report
    let final_report = VirtualTrainingReport {
        timestamp: Utc::now().to_rfc3339(),
        run_report,
        models: model_statuses,
        warnings: if comparison.recommendation
            == ScenarioComparisonRecommendation::RegressionDetected
        {
            vec![
                "Candidate models overall regressed on MixedRoom scenario against baseline"
                    .to_string(),
            ]
        } else {
            Vec::new()
        },
    };

    let parent = Path::new(&args.report_out).parent();
    if let Some(p) = parent {
        fs::create_dir_all(p)?;
    }
    fs::write(
        &args.report_out,
        serde_json::to_string_pretty(&final_report)?,
    )?;
    println!(
        "Consolidated training report written to {}",
        args.report_out
    );

    Ok(())
}

#[derive(Debug, Parser)]
struct RetinaMockSendArgs {
    /// Server URL
    #[arg(long, default_value = "https://localhost:8443")]
    url: String,

    /// Frame rate (FPS)
    #[arg(long, default_value = "5")]
    fps: u64,

    /// Width of mock image
    #[arg(long, default_value = "160")]
    width: u32,

    /// Height of mock image
    #[arg(long, default_value = "90")]
    height: u32,

    /// Color pattern: "solid-red", "solid-green", "solid-blue", "gradient", or "noise"
    #[arg(long, default_value = "gradient")]
    pattern: String,
}

fn generate_mock_image_base64(
    width: u32,
    height: u32,
    pattern: &str,
    frame_index: usize,
) -> Result<String> {
    use base64::Engine;
    use image::codecs::png::PngEncoder;
    use image::ImageEncoder;
    use image::{Rgb, RgbImage};

    let mut img = RgbImage::new(width, height);

    match pattern {
        "solid-red" => {
            for pixel in img.pixels_mut() {
                *pixel = Rgb([255, 0, 0]);
            }
        }
        "solid-green" => {
            for pixel in img.pixels_mut() {
                *pixel = Rgb([0, 255, 0]);
            }
        }
        "solid-blue" => {
            for pixel in img.pixels_mut() {
                *pixel = Rgb([0, 0, 255]);
            }
        }
        "gradient" => {
            for (x, y, pixel) in img.enumerate_pixels_mut() {
                let r = ((x as f32 / width as f32) * 255.0) as u8;
                let g = ((y as f32 / height as f32) * 255.0) as u8;
                let b = ((frame_index * 10) % 256) as u8;
                *pixel = Rgb([r, g, b]);
            }
        }
        "noise" => {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for pixel in img.pixels_mut() {
                *pixel = Rgb([rng.gen(), rng.gen(), rng.gen()]);
            }
        }
        _ => {
            for (x, y, pixel) in img.enumerate_pixels_mut() {
                let g = ((x as f32 / width as f32) * 255.0) as u8;
                let b = ((y as f32 / height as f32) * 255.0) as u8;
                *pixel = Rgb([0, g, b]);
            }
        }
    }

    let mut png_bytes = Vec::new();
    PngEncoder::new(&mut png_bytes)
        .write_image(&img, width, height, image::ColorType::Rgb8.into())
        .context("failed to encode mock image as PNG")?;

    let encoded = base64::engine::general_purpose::STANDARD.encode(&png_bytes);
    Ok(encoded)
}

async fn run_retina_mock_send(args: RetinaMockSendArgs) -> Result<()> {
    let client = reqwest::Client::builder()
        .danger_accept_invalid_certs(true)
        .build()
        .context("failed to build reqwest client")?;

    let url = format!("{}/view/retina-frame", args.url.trim_end_matches('/'));
    println!(
        "Starting mock retina stream to {url} at {} FPS ({}x{})...",
        args.fps, args.width, args.height
    );

    let interval = Duration::from_millis(1000 / args.fps.max(1));
    let mut interval_timer = tokio::time::interval(interval);
    let mut frame_index = 0;

    let start_time = std::time::Instant::now();

    loop {
        interval_timer.tick().await;

        let t_ms = start_time.elapsed().as_millis() as u64;
        let base64_str =
            match generate_mock_image_base64(args.width, args.height, &args.pattern, frame_index) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error generating mock image: {e}");
                    continue;
                }
            };

        let payload = serde_json::json!({
            "schema_version": 1,
            "source": "babylon-robot-eye",
            "t_ms": t_ms,
            "frame_index": frame_index,
            "width": args.width,
            "height": args.height,
            "format": "Rgb8",
            "encoding": "base64",
            "data": format!("data:image/png;base64,{base64_str}")
        });

        let res = client.post(&url).json(&payload).send().await;

        match res {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    println!(
                        "[frame {}] Sent successfully (t_ms = {})",
                        frame_index, t_ms
                    );
                } else {
                    let err_text = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "unknown error".to_string());
                    eprintln!(
                        "[frame {}] FAILED with status {}: {}",
                        frame_index, status, err_text
                    );
                }
            }
            Err(e) => {
                eprintln!("[frame {}] Request error: {}", frame_index, e);
            }
        }

        frame_index += 1;
    }
}
