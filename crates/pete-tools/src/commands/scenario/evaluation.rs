async fn run_social_exam_command(args: SocialExamArgs) -> Result<()> {
    let report = pete_runtime::run_social_exam().await?;
    for case in &report.cases {
        println!(
            "{:<28} {}",
            case.case,
            if case.passed { "PASS" } else { "FAIL" }
        );
        for failure in &case.failures {
            println!("  {failure}");
        }
    }
    if let Some(out) = args.out.as_deref() {
        let path = Path::new(out);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, serde_json::to_vec_pretty(&report)?)?;
        println!("social exam report written: {out}");
    }
    if !report.passed {
        anyhow::bail!("social exam failed");
    }
    println!("social exam passed: {} cases", report.cases.len());
    Ok(())
}

async fn run_eval_scenario(args: EvalScenarioArgs) -> Result<()> {
    let kind = ScenarioKind::from(args.scenario);
    let flags = RuntimeModelFlags::from(&args);
    let mut model_loading = load_runtime_models_from_flags(&flags)?.1;
    if args.future_mode == FutureMode::ModelInfer {
        model_loading.blocked_model_infer.push(
            "future model-infer is limited to prediction behavior; motor safety remains hardcoded"
                .to_string(),
        );
    }
    if args.experience_mode == ExperienceMode::ModelInfer {
        model_loading.blocked_model_infer.push(
            "experience model-infer changes latent encoding only; motor safety remains hardcoded"
                .to_string(),
        );
    }

    let mut episodes_detail = Vec::with_capacity(args.episodes);
    for episode_index in 0..args.episodes {
        let episode_seed = args.seed.saturating_add(episode_index as u64);
        let scenario = build_scenario(ScenarioConfig::new(kind, episode_seed));
        let capture = args.capture_root.as_ref().map(|root| {
            Path::new(root)
                .join(format!("episode-{episode_index:03}"))
                .to_string_lossy()
                .to_string()
        });
        let builder = EpisodeMetricBuilder::new(
            kind,
            scenario.metadata.clone(),
            episode_index,
            episode_seed,
            args.ledger.clone(),
            capture.clone(),
        );
        let (episode, warnings) = if let Some(ledger_path) = &args.ledger {
            let mut runtime = default_runtime(JsonlLedger::new(ledger_path), &args.llm)?;
            runtime = runtime.with_action_selector_mode(args.action_selector.into());
            if let Some(models) = load_runtime_models_from_flags(&flags)?.0 {
                runtime = runtime.with_models(models);
            }
            run_eval_episode(runtime, scenario.world, scenario.motors, &args, builder).await?
        } else {
            let mut runtime = default_noop_runtime(&args.llm)?;
            runtime = runtime.with_action_selector_mode(args.action_selector.into());
            if let Some(models) = load_runtime_models_from_flags(&flags)?.0 {
                runtime = runtime.with_models(models);
            }
            run_eval_episode(runtime, scenario.world, scenario.motors, &args, builder).await?
        };
        model_loading.warnings.extend(warnings);
        println!(
            "eval episode {} complete: scenario {}, seed {}, ticks {}, success {}, collisions {}",
            episode.index,
            kind.slug(),
            episode.seed,
            episode.ticks,
            episode.success,
            episode.collisions
        );
        episodes_detail.push(episode);
    }

    let summary = summarize_episodes(&episodes_detail);
    let memory = args
        .memory_report
        .then(|| summarize_episode_memory(&episodes_detail));
    let recommendation = scenario_recommendation(args.episodes, &summary);
    let report = ScenarioEvaluationReport {
        schema_version: 1,
        scenario: kind.slug().to_string(),
        base_seed: args.seed,
        episodes: args.episodes,
        steps_per_episode: args.steps,
        tick_ms: args.tick_ms,
        action_selector_mode: ActionSelectorMode::from(args.action_selector)
            .as_str()
            .to_string(),
        model_modes: model_modes_from_flags(&flags),
        model_loading: model_loading.clone(),
        ledger: args.ledger.clone(),
        capture_root: args.capture_root.clone(),
        summary,
        memory,
        episodes_detail,
        recommendation,
        warnings: model_loading.warnings.clone(),
    };

    let bytes = serde_json::to_vec_pretty(&report)?;
    if let Some(out) = &args.out {
        if let Some(parent) = Path::new(out).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(out, &bytes)?;
        println!("scenario evaluation report written: {out}");
    } else {
        println!("{}", String::from_utf8_lossy(&bytes));
    }
    Ok(())
}

async fn run_eval_episode<R>(
    runtime: R,
    world: pete_sim::VirtualWorld,
    motors: pete_sim::SimCockpit,
    args: &EvalScenarioArgs,
    mut metrics: EpisodeMetricBuilder,
) -> Result<(ScenarioEpisodeReport, Vec<String>)>
where
    R: RuntimeLoop + Send,
{
    let mut warnings = Vec::new();
    let mut runner = SimRunner::new(runtime, world, motors);
    runner.tick_ms = args.tick_ms;
    let mut snapshots = Vec::new();
    runner
        .run_steps_observing_ticks(args.steps, |snapshot, tick| {
            if metrics.capture.is_some() {
                snapshots.push(snapshot.clone());
            }
            metrics.observe(snapshot, tick);
        })
        .await?;

    if let Some(capture_path) = &metrics.capture {
        let mut writer =
            CaptureWriter::create(capture_path, CaptureSource::Sim, Some(args.tick_ms)).await?;
        writer.manifest_mut().scenario = Some(metrics.metadata.clone());
        for snapshot in snapshots {
            writer
                .append_snapshot(snapshot.body.last_update_ms, snapshot, Vec::new())
                .await?;
        }
        writer.finish().await?;
    }

    if runner.tick_count < args.steps {
        warnings.push(format!(
            "episode {} stopped after {} ticks before requested {} steps",
            metrics.index, runner.tick_count, args.steps
        ));
    }
    Ok((metrics.finish(), warnings))
}

fn configured_llm_config(args: &LlmArgs) -> Result<LlmConfig> {
    let mut config = match &args.llm_config {
        Some(path) => LlmConfig::load(path)?,
        None => LlmConfig::default(),
    };
    if let Some(provider) = args.llm_provider {
        config.provider = provider.into();
    }
    Ok(config)
}

fn configured_llm_agent(args: &LlmArgs) -> Result<ConfiguredLlmAgent> {
    let config = configured_llm_config(args)?;
    ConfiguredLlmAgent::from_config(config)
}

fn configured_llm_config_for_sim(args: &LlmArgs, live: bool) -> Result<LlmConfig> {
    let mut config = configured_llm_config(args)?;
    if live && args.llm_config.is_none() {
        let live_timeout_ms = std::env::var("PETE_LIVE_LLM_TIMEOUT_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_LIVE_LLM_TIMEOUT_MS);
        config.timeout_ms = config.timeout_ms.min(live_timeout_ms.max(1));
    }
    Ok(config)
}

fn configured_llm_agent_for_sim(args: &LlmArgs, live: bool) -> Result<ConfiguredLlmAgent> {
    let config = configured_llm_config_for_sim(args, live)?;
    ConfiguredLlmAgent::from_config(config)
}

fn default_noop_runtime(
    llm_args: &LlmArgs,
) -> Result<
    MinimalRuntime<
        NoopLedger,
        InMemoryExperienceStore,
        InMemoryExperienceStore,
        SimpleConductor,
        SimpleSafety,
        ConfiguredLlmAgent,
    >,
> {
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    Ok(MinimalRuntime::with_default_events(
        NoopLedger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        configured_llm_agent(llm_args)?,
    ))
}

fn episode_success(kind: ScenarioKind, episode: &ScenarioEpisodeReport) -> bool {
    match kind {
        ScenarioKind::EmptyRoom => episode.ticks > 0 && episode.collisions == 0,
        ScenarioKind::ObstacleAvoidance => {
            episode.ticks > 0
                && episode.collisions <= (episode.ticks / 50).max(1)
                && episode.stuck_ticks < episode.ticks / 2
                && episode.distance_traveled_m > 0.05
        }
        ScenarioKind::CornerTrap | ScenarioKind::ConcaveTrap => {
            episode.stuck_count > 0
                && episode.recovery_success_rate.unwrap_or(0.0) > 0.0
                && episode.distance_traveled_m > 0.10
                && episode.collisions <= (episode.ticks / 20).max(1)
        }
        ScenarioKind::ColumnTrap => {
            episode.stuck_count > 0
                && episode.recovery_success_rate.unwrap_or(0.0) > 0.0
                && episode.distance_traveled_m > 0.25
                && episode.collisions <= (episode.ticks / 20).max(1)
        }
        ScenarioKind::ChargerSeeking => {
            episode.charging_ticks > 0 && episode.dead_battery_tick.is_none()
        }
        ScenarioKind::PersonAndSpeaker => {
            episode.ticks > 0
                && episode.collisions == 0
                && (episode.ticks_with_face_embeddings > 0
                    || episode.ticks_with_voice_embeddings > 0
                    || episode.ticks_with_kinect_skeletons > 0
                    || episode.ticks_with_ear_features > 0)
        }
        ScenarioKind::MixedRoom => {
            episode.ticks > 0
                && episode.collisions <= (episode.ticks / 40).max(1)
                && (episode.charging_ticks > 0
                    || episode.ticks_with_face_embeddings > 0
                    || episode.ticks_with_voice_embeddings > 0)
        }
        ScenarioKind::Dream => {
            episode.ticks > 0
                && episode.collisions <= (episode.ticks / 30).max(1)
                && (episode.charging_ticks > 0
                    || episode.ticks_with_face_embeddings > 0
                    || episode.ticks_with_voice_embeddings > 0
                    || episode.ticks_with_ear_features > 0)
        }
    }
}

fn summarize_episodes(episodes: &[ScenarioEpisodeReport]) -> ScenarioEvaluationSummary {
    if episodes.is_empty() {
        return ScenarioEvaluationSummary::default();
    }
    let count = episodes.len() as f32;
    let total_ticks: usize = episodes.iter().map(|episode| episode.ticks).sum();
    let total_collisions: usize = episodes.iter().map(|episode| episode.collisions).sum();
    let mut trap_kind_counts = HashMap::new();
    for episode in episodes {
        for (kind, count) in &episode.trap_kind_counts {
            *trap_kind_counts.entry(kind.clone()).or_default() += count;
        }
    }
    let goal_progress_samples = episodes
        .iter()
        .map(|episode| episode.goal_progress_samples)
        .sum::<usize>();
    let goal_progress_sum = episodes
        .iter()
        .map(|episode| {
            episode.mean_goal_progress.unwrap_or(0.0) * episode.goal_progress_samples as f32
        })
        .sum::<f32>();
    let stall_responses = episodes
        .iter()
        .map(|episode| episode.stall_responses)
        .sum::<usize>();
    let false_stall_count = episodes
        .iter()
        .map(|episode| episode.false_stall_count)
        .sum::<usize>();
    ScenarioEvaluationSummary {
        success_rate: episodes.iter().filter(|episode| episode.success).count() as f32 / count,
        collision_rate: if total_ticks == 0 {
            0.0
        } else {
            total_collisions as f32 / total_ticks as f32
        },
        mean_collisions_per_episode: total_collisions as f32 / count,
        mean_battery_delta: mean(episodes.iter().map(|episode| episode.battery_delta)),
        mean_final_battery: mean(episodes.iter().map(|episode| episode.final_battery)),
        mean_distance_to_charger_final_m: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.final_distance_to_charger_m),
        ),
        ticks_with_charger_visible: episodes
            .iter()
            .map(|episode| episode.ticks_with_charger_visible)
            .sum(),
        ticks_with_charger_near: episodes
            .iter()
            .map(|episode| episode.ticks_with_charger_near)
            .sum(),
        ticks_approaching_charger: episodes
            .iter()
            .map(|episode| episode.ticks_approaching_charger)
            .sum(),
        ticks_docking_from_too_far: episodes
            .iter()
            .map(|episode| episode.ticks_docking_from_too_far)
            .sum(),
        mean_nearest_obstacle_m: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.mean_nearest_obstacle_m),
        ),
        mean_distance_traveled_m: mean(episodes.iter().map(|episode| episode.distance_traveled_m)),
        action_histogram: summarize_action_histogram(episodes),
        wall_cliff_veto_count: episodes
            .iter()
            .map(|episode| episode.wall_cliff_veto_count)
            .sum(),
        escape_progress_score: mean(episodes.iter().map(|episode| episode.escape_progress_score)),
        mean_ticks_survived: mean(episodes.iter().map(|episode| episode.ticks as f32)),
        stuck_count: episodes.iter().map(|episode| episode.stuck_count).sum(),
        trap_kind_counts,
        recovery_attempts: episodes
            .iter()
            .map(|episode| episode.recovery_attempts)
            .sum(),
        stuck_duration: mean_optional(episodes.iter().filter_map(|episode| episode.stuck_duration)),
        mean_stuck_duration: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.mean_stuck_duration),
        ),
        recovery_success_rate: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.recovery_success_rate),
        ),
        mean_recovery_ticks: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.mean_recovery_ticks),
        ),
        repeated_trap_count: episodes
            .iter()
            .map(|episode| episode.repeated_trap_count)
            .sum(),
        dead_battery_tick: episodes
            .iter()
            .filter_map(|episode| episode.dead_battery_tick)
            .min(),
        distance_after_recovery_m: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.distance_after_recovery_m),
        ),
        mean_safety_interventions: mean(
            episodes
                .iter()
                .map(|episode| episode.safety_interventions as f32),
        ),
        behavior_run_records: episodes
            .iter()
            .map(|episode| episode.behavior_run_records)
            .sum(),
        model_fallbacks: episodes.iter().map(|episode| episode.model_fallbacks).sum(),
        action_selector_fallbacks: episodes
            .iter()
            .map(|episode| episode.action_selector_fallbacks)
            .sum(),
        action_selector_guard_yields: episodes
            .iter()
            .map(|episode| episode.action_selector_guard_yields)
            .sum(),
        map_memory_decisions: episodes
            .iter()
            .map(|episode| episode.map_memory_decisions)
            .sum(),
        danger_memory_decisions: episodes
            .iter()
            .map(|episode| episode.danger_memory_decisions)
            .sum(),
        charge_memory_decisions: episodes
            .iter()
            .map(|episode| episode.charge_memory_decisions)
            .sum(),
        novelty_memory_decisions: episodes
            .iter()
            .map(|episode| episode.novelty_memory_decisions)
            .sum(),
        frontier_memory_decisions: episodes
            .iter()
            .map(|episode| episode.frontier_memory_decisions)
            .sum(),
        trap_memory_decisions: episodes
            .iter()
            .map(|episode| episode.trap_memory_decisions)
            .sum(),
        memory_navigation_intents: summarize_memory_navigation_intents(episodes),
        memory_navigation_reasons: summarize_memory_navigation_reasons(episodes),
        map_memory_signals: summarize_map_memory_signals(episodes),
        map_memory_safety_overrides: episodes
            .iter()
            .map(|episode| episode.map_memory_safety_overrides)
            .sum(),
        low_confidence_navigation_fallbacks: episodes
            .iter()
            .map(|episode| episode.low_confidence_navigation_fallbacks)
            .sum(),
        model_assisted_decisions: episodes
            .iter()
            .map(|episode| episode.model_assisted_decisions)
            .sum(),
        action_selector_safety_overrides: episodes
            .iter()
            .map(|episode| episode.action_selector_safety_overrides)
            .sum(),
        goal_switches: episodes.iter().map(|episode| episode.goal_switches).sum(),
        goal_commitment_retained_ticks: episodes
            .iter()
            .map(|episode| episode.goal_commitment_retained_ticks)
            .sum(),
        goal_behavior_transitions: episodes
            .iter()
            .map(|episode| episode.goal_behavior_transitions)
            .sum(),
        goal_shadow_divergences: episodes
            .iter()
            .map(|episode| episode.goal_shadow_divergences)
            .sum(),
        mean_goal_dwell_ms: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.mean_goal_dwell_ms),
        ),
        goal_histogram: summarize_string_histogram(
            episodes.iter().map(|episode| &episode.goal_histogram),
        ),
        goal_behavior_histogram: summarize_string_histogram(
            episodes
                .iter()
                .map(|episode| &episode.goal_behavior_histogram),
        ),
        goal_progress_samples,
        mean_goal_progress: (goal_progress_samples > 0)
            .then_some(goal_progress_sum / goal_progress_samples as f32),
        goal_no_progress_dwell_ticks: episodes
            .iter()
            .map(|episode| episode.goal_no_progress_dwell_ticks)
            .sum(),
        goal_failed_attempts: episodes
            .iter()
            .map(|episode| episode.goal_failed_attempts)
            .sum(),
        strategy_switches_within_goal: episodes
            .iter()
            .map(|episode| episode.strategy_switches_within_goal)
            .sum(),
        goal_help_requests: episodes
            .iter()
            .map(|episode| episode.goal_help_requests)
            .sum(),
        unmeasurable_progress_ticks: episodes
            .iter()
            .map(|episode| episode.unmeasurable_progress_ticks)
            .sum(),
        false_stall_rate: (stall_responses > 0)
            .then_some(false_stall_count as f32 / stall_responses as f32),
        mean_chosen_score: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.mean_chosen_score),
        ),
        mean_candidate_score: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.mean_candidate_score),
        ),
    }
}

fn summarize_action_histogram(episodes: &[ScenarioEpisodeReport]) -> HashMap<String, usize> {
    let mut histogram = HashMap::new();
    for episode in episodes {
        for (action, count) in &episode.action_histogram {
            *histogram.entry(action.clone()).or_default() += count;
        }
    }
    histogram
}

fn summarize_string_histogram<'a>(
    histograms: impl IntoIterator<Item = &'a HashMap<String, usize>>,
) -> HashMap<String, usize> {
    let mut combined = HashMap::new();
    for histogram in histograms {
        for (key, count) in histogram {
            *combined.entry(key.clone()).or_default() += count;
        }
    }
    combined
}

fn summarize_memory_navigation_intents(
    episodes: &[ScenarioEpisodeReport],
) -> HashMap<String, usize> {
    let mut histogram = HashMap::new();
    for episode in episodes {
        for (intent, count) in &episode.memory_navigation_intents {
            *histogram.entry(intent.clone()).or_default() += count;
        }
    }
    histogram
}

fn summarize_memory_navigation_reasons(
    episodes: &[ScenarioEpisodeReport],
) -> HashMap<String, usize> {
    let mut histogram = HashMap::new();
    for episode in episodes {
        for (reason, count) in &episode.memory_navigation_reasons {
            *histogram.entry(reason.clone()).or_default() += count;
        }
    }
    histogram
}

fn summarize_map_memory_signals(episodes: &[ScenarioEpisodeReport]) -> HashMap<String, usize> {
    let mut histogram = HashMap::new();
    for episode in episodes {
        for (signal, count) in &episode.map_memory_signals {
            *histogram.entry(signal.clone()).or_default() += count;
        }
    }
    histogram
}

fn action_histogram_label(action: &ActionPrimitive) -> &'static str {
    match action {
        ActionPrimitive::Stop => "Stop",
        ActionPrimitive::Go { intensity, .. } if *intensity < 0.0 => "Reverse",
        ActionPrimitive::Go { .. } => "Go",
        ActionPrimitive::Drive { .. } => "Drive",
        ActionPrimitive::Turn {
            direction: TurnDir::Left,
            ..
        } => "TurnLeft",
        ActionPrimitive::Turn {
            direction: TurnDir::Right,
            ..
        } => "TurnRight",
        ActionPrimitive::Inspect { .. } => "Inspect",
        ActionPrimitive::Approach { .. } => "Approach",
        ActionPrimitive::Dock => "Dock",
        ActionPrimitive::Explore { .. } => "Explore",
        ActionPrimitive::Speak { .. } => "Speak",
        ActionPrimitive::Chirp { .. } => "Chirp",
    }
}

fn wall_or_cliff_veto(tick: &RuntimeTick) -> bool {
    tick.frame
        .now
        .extensions
        .get("motor_gate")
        .and_then(|value| value.get("safety_reason"))
        .and_then(|value| value.as_str())
        .map(|reason| reason == "cliff")
        .unwrap_or(false)
        || tick.frame.now.body.flags.wall
        || tick.frame.now.body.flags.cliff_left
        || tick.frame.now.body.flags.cliff_front_left
        || tick.frame.now.body.flags.cliff_front_right
        || tick.frame.now.body.flags.cliff_right
}

fn escape_progress_score(
    kind: ScenarioKind,
    distance_traveled_m: f32,
    distance_at_last_recovery_m: Option<f32>,
    collisions: usize,
    stuck_ticks: usize,
    ticks: usize,
) -> f32 {
    let progress = match kind {
        ScenarioKind::ColumnTrap | ScenarioKind::CornerTrap | ScenarioKind::ConcaveTrap => {
            distance_at_last_recovery_m
                .map(|distance| (distance_traveled_m - distance).max(0.0))
                .filter(|distance| *distance >= 0.08)
                .unwrap_or(distance_traveled_m)
        }
        _ => distance_traveled_m,
    };
    let collision_penalty = collisions as f32 * 0.05;
    let stuck_penalty = if ticks == 0 {
        0.0
    } else {
        stuck_ticks as f32 / ticks as f32 * 0.25
    };
    (progress - collision_penalty - stuck_penalty).max(0.0)
}

fn trap_kind_label(code: f32) -> Option<&'static str> {
    match code.round() as i32 {
        1 => Some("wall"),
        2 => Some("corner"),
        3 => Some("column"),
        _ => None,
    }
}

fn summarize_episode_memory(episodes: &[ScenarioEpisodeReport]) -> ScenarioMemorySummary {
    let memory_reports = episodes
        .iter()
        .filter_map(|episode| episode.memory.as_ref())
        .collect::<Vec<_>>();
    if memory_reports.is_empty() {
        return ScenarioMemorySummary {
            novelty_decay_sane: false,
            warnings: vec!["no episode memory reports".to_string()],
            ..ScenarioMemorySummary::default()
        };
    }
    let places_visited = memory_reports
        .iter()
        .map(|memory| memory.places_visited)
        .max()
        .unwrap_or(0);
    let mut warnings = Vec::new();
    if places_visited == 0 {
        warnings.push("memory observed zero places".to_string());
    }
    let novelty_decay_sane = memory_reports.iter().any(|memory| memory.novelty_decayed);
    if !novelty_decay_sane {
        warnings.push("novelty did not decay in any episode".to_string());
    }
    ScenarioMemorySummary {
        places_visited,
        mean_places_visited_per_episode: mean(
            memory_reports
                .iter()
                .map(|memory| memory.places_visited as f32),
        ),
        charge_memory_hit_rate: aggregate_hit_rate(
            memory_reports
                .iter()
                .map(|memory| (memory.charge_memory_ticks, memory.charge_opportunity_ticks)),
        ),
        danger_memory_hit_rate: aggregate_hit_rate(
            memory_reports
                .iter()
                .map(|memory| (memory.danger_memory_ticks, memory.danger_opportunity_ticks)),
        ),
        social_memory_hit_rate: aggregate_hit_rate(
            memory_reports
                .iter()
                .map(|memory| (memory.social_memory_ticks, memory.social_opportunity_ticks)),
        ),
        novelty_decay_sane,
        warnings,
    }
}
