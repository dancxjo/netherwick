async fn run_embodied_demo(args: EmbodiedDemoArgs) -> Result<()> {
    let now_ms = Utc::now().timestamp_millis().max(0) as u64;
    let demo = pete_experience::demo_embodied_experience(now_ms).await?;
    let mut impressions = demo.impressions.clone();
    if let Some(summary) = demo.experience.summary_impression.clone() {
        impressions.push(summary);
    }

    if let Some(root) = args.ledger.as_deref() {
        let ledger = JsonlLedger::new(root);
        let frame = ExperienceFrame {
            id: uuid::Uuid::new_v4(),
            t_ms: now_ms,
            now: Now::blank(now_ms, BodySense::default()),
            sensations: demo.sensations.clone(),
            impressions: impressions.clone(),
            experiences: vec![demo.experience.clone()],
            z: None,
            chosen_action: None,
            conscious_command: None,
            reign_input: None,
            reign_outcome: None,
            predicted_futures: Vec::new(),
            behavior_runs: Vec::new(),
            actual_next: None,
            reward: Default::default(),
            surprise: SurpriseSense::default(),
            memory_recall: Vec::new(),
            recollections: Vec::new(),
            llm_teaching: Vec::new(),
            counterfactuals: Vec::new(),
            notes: vec!["embodied demo pipeline".to_string()],
        };
        ledger.append(&frame).await?;
        println!("wrote embodied demo frame to {}", root);
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&demo)?);
        return Ok(());
    }

    println!("embodied experience {}", demo.experience.id);
    println!("  summary: {}", demo.experience.text);
    println!(
        "  vector coverage: image={} face={} voice={} transcript={} impression={} experience={} fallback_count={}",
        demo.coverage.image,
        demo.coverage.face,
        demo.coverage.voice,
        demo.coverage.transcript,
        demo.coverage.impression,
        demo.coverage.experience,
        demo.coverage.fallback_count
    );
    println!("  sensations: {}", demo.sensations.len());
    for sensation in &demo.sensations {
        let vector = sensation
            .vector
            .as_ref()
            .map(|embedding| {
                format!(
                    "{}d {} purpose={} vectorizer={} fallback={}",
                    embedding.dim,
                    embedding.model_id,
                    embedding.purpose,
                    embedding.vectorizer_id,
                    embedding.is_fallback
                )
            })
            .unwrap_or_else(|| "none".to_string());
        println!(
            "    - {} {:?}/{:?} parent={:?} vector={}",
            sensation.kind, sensation.modality, sensation.payload_kind, sensation.parent_id, vector
        );
    }
    println!("  impressions:");
    for impression in &impressions {
        let vector = impression
            .vector
            .as_ref()
            .map(|embedding| {
                format!(
                    "{}d {} purpose={} vectorizer={} fallback={}",
                    embedding.dim,
                    embedding.model_id,
                    embedding.purpose,
                    embedding.vectorizer_id,
                    embedding.is_fallback
                )
            })
            .unwrap_or_else(|| "none".to_string());
        println!("    - {} vector={}", impression.text, vector);
    }
    Ok(())
}

async fn run_embodied_eval(args: EmbodiedEvalArgs) -> Result<()> {
    match args.fixture {
        EmbodiedEvalFixtureArg::Deterministic => {}
    }
    let omissions = args
        .omit
        .into_iter()
        .map(EmbodiedEvalOmission::from)
        .collect::<Vec<_>>();
    let report = deterministic_embodied_eval_report_with_omissions(&omissions).await?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("embodied eval fixture={}", report.fixture);
        println!("  frames: {}", report.frame_count);
        println!("  instants: {}", report.instant_count);
        println!(
            "  instant teacher vectors: {}",
            report.instant_teacher_vector_count
        );
        println!(
            "  instant missing modalities: {}",
            report.instant_missing_modality_count
        );
        println!("  primary sensations: {}", report.primary_sensation_count);
        println!(
            "  descendant sensations: {}",
            report.descendant_sensation_count
        );
        println!(
            "  vectorized sensations: {}",
            report.vectorized_sensation_count
        );
        println!("  impressions: {}", report.impression_count);
        println!("  summary impressions: {}", report.summary_impression_count);
        println!(
            "  learned experience latents: {}",
            report.experience_latent_count
        );
        println!("  predictions: {}", report.prediction_count);
        println!("  memory links: {}", report.memory_link_count);
        println!("  recall sensations: {}", report.recall_sensation_count);
        println!("  recall impressions: {}", report.recall_impression_count);
        println!("  lineage edges: {}", report.lineage_edge_count);
        if !report.warnings.is_empty() {
            println!("  warnings:");
            for warning in &report.warnings {
                println!("    - {warning}");
            }
        }
        if !report.failures.is_empty() {
            println!("  failures:");
            for failure in &report.failures {
                println!("    - {failure}");
            }
        }
    }

    if report.passed() {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "embodied eval failed: {}",
            report.failures.join(", ")
        ))
    }
}

async fn generate_virtual_report(ledger_path: &str) -> Result<VirtualRunReport> {
    let ledger = JsonlLedger::new(ledger_path);
    let frames = ledger.frames().await?;
    let transitions = ledger.transitions().await?;

    let total_frames = frames.len();
    let total_transitions = transitions.len();

    let mut total_eye_frames = 0;
    let mut total_ear_frames = 0;
    let mut total_stuck_trap_events = 0;

    let mut eye_sources = HashMap::new();
    let mut babylon_eye_frames = 0;
    let mut collisions = 0;
    let mut charging_ticks = 0;
    let mut charger_contacts = 0;
    let mut was_charging = false;
    let mut stuck_recovery_attempts = 0;
    let mut stuck_recovery_successes = 0;
    let mut trap_kinds = HashMap::new();
    let mut ledger_gaps = Vec::new();
    let mut warnings = Vec::new();

    let mut min_battery = 1.0f32;
    let mut max_after_min = 0.0f32;
    let mut prev_t_ms = None;

    for frame in &frames {
        if !frame.now.eye.frames.is_empty() || !frame.now.eye.image_vectors.is_empty() {
            total_eye_frames += 1;
        }
        if !frame.now.ear.features.is_empty() || frame.now.ear.transcript.is_some() {
            total_ear_frames += 1;
        }

        // 1. Eye source tracking
        if let Some(eye_frame) = &frame.now.eye_frame {
            let src = eye_frame
                .source
                .clone()
                .unwrap_or_else(|| "none".to_string());
            *eye_sources.entry(src.clone()).or_insert(0) += 1;
            if src == "babylon-robot-eye" {
                babylon_eye_frames += 1;
            }
        }

        // 2. Collision tracking
        if frame.now.body.flags.bump_left || frame.now.body.flags.bump_right {
            collisions += 1;
        }

        // 3. Charger & Battery tracking
        if frame.now.body.charging {
            charging_ticks += 1;
            if !was_charging {
                charger_contacts += 1;
            }
            was_charging = true;
        } else {
            was_charging = false;
        }

        let bat = frame.now.body.battery_level;
        if bat < min_battery {
            min_battery = bat;
            max_after_min = bat;
        } else if bat > max_after_min {
            max_after_min = bat;
        }

        // 4. Stuck recovery / Trap tracking
        if let Some(val) = frame.now.extensions.get("sim.stuck") {
            if let Ok(values) = serde_json::from_value::<Vec<f32>>(val.clone()) {
                let event_started = values.get(6).copied().unwrap_or(0.0) > 0.0;
                let recovered = values.get(7).copied().unwrap_or(0.0) > 0.0;
                let trap_code = values.get(10).copied().unwrap_or(0.0);

                if event_started {
                    total_stuck_trap_events += 1;
                    stuck_recovery_attempts += 1;
                    let trap_name = match trap_code {
                        1.0 => "Wall",
                        2.0 => "Corner",
                        3.0 => "Column",
                        _ => "Unknown",
                    }
                    .to_string();
                    *trap_kinds.entry(trap_name).or_insert(0) += 1;
                }
                if recovered {
                    stuck_recovery_successes += 1;
                }
            }
        }

        // 5. Gap tracking
        if let Some(prev) = prev_t_ms {
            let diff = frame.t_ms.saturating_sub(prev);
            if diff > 500 {
                ledger_gaps.push(format!(
                    "gap of {}ms between {}ms and {}ms",
                    diff, prev, frame.t_ms
                ));
            }
        }
        prev_t_ms = Some(frame.t_ms);
    }

    let battery_delta = if let (Some(first), Some(last)) = (frames.first(), frames.last()) {
        first.now.body.battery_level - last.now.body.battery_level
    } else {
        0.0
    };

    let battery_recovery_success = max_after_min - min_battery >= 0.05;

    let duration_seconds = if let (Some(first), Some(last)) = (frames.first(), frames.last()) {
        (last.t_ms.saturating_sub(first.t_ms) as f64) / 1000.0
    } else {
        0.0
    };

    let collision_rate = if total_frames > 0 {
        collisions as f32 / total_frames as f32
    } else {
        0.0
    };

    let retina_coverage = if total_frames > 0 {
        babylon_eye_frames as f32 / total_frames as f32
    } else {
        0.0
    };

    if total_frames == 0 {
        warnings.push("ledger is empty".to_string());
    } else if babylon_eye_frames == 0 {
        warnings.push("no retina frames from babylon-robot-eye found in ledger".to_string());
    }

    Ok(VirtualRunReport {
        total_frames,
        total_transitions,
        total_eye_frames,
        total_ear_frames,
        total_stuck_trap_events,
        battery_delta,
        duration_seconds,
        eye_sources,
        retina_coverage,
        collisions,
        collision_rate,
        charger_contacts,
        charging_ticks,
        battery_recovery_success,
        stuck_recovery_attempts,
        stuck_recovery_successes,
        trap_kinds,
        ledger_gaps,
        warnings,
    })
}
