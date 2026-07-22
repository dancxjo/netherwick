async fn run_sim(args: SimArgs) -> Result<()> {
    let ledger = JsonlLedger::new(&args.ledger);
    let flags = RuntimeModelFlags::from(&args);
    let (models, model_loading) = load_runtime_models_from_flags(&flags)?;
    let action_selector_mode = ActionSelectorMode::from(args.action_selector);
    let inline_learning = inline_learning_config_from_sim_args(&args)?;
    let models_loaded = loaded_model_names(&model_loading);
    let conductor = SimpleConductor::default();
    let mut live_action_selector_label = action_selector_mode.as_str().to_string();
    if let Some(checkpoint) = args.dream_policy_checkpoint.as_deref() {
        println!(
            "Dream NEAT controller ignored for live control: {checkpoint}. Reign mechanics drive directly; models remain shadow observers."
        );
        live_action_selector_label = format!("reign-passthrough+{}", action_selector_mode.as_str());
    }
    let memory = DurableExperienceStore::from_env();
    let recall = memory.clone();
    let llm = configured_llm_agent_for_sim(&args.llm, args.live)?;
    let mut runtime = MinimalRuntime::with_default_events(
        ledger,
        memory,
        recall,
        conductor,
        SimpleSafety::default(),
        llm,
    )
    .with_action_selector_mode(action_selector_mode)
    .with_inline_learning(inline_learning.clone())
    .with_nudge_policy(NudgePolicy::virtual_default());
    if let Some(models) = models {
        runtime = runtime.with_models(models);
    }

    let scenario_kind: ScenarioKind = args.scenario.into();
    let scenario = build_scenario(ScenarioConfig::new(scenario_kind, args.seed));
    let live_metadata = live_scene_metadata_from_scenario(&scenario.metadata);
    let world = scenario.world;
    let motors = scenario.motors;
    let mut runner = SimRunner::new(runtime, world, motors);
    if args.live {
        let live_state = LiveViewState::new().with_virtual_retina(true);
        let initial_snapshot = runner.world.snapshot().await?;
        live_state.update(initial_snapshot);
        live_state.update_inline_learning(inline_learning.clone());
        live_state.update_scene_metadata(live_metadata);
        live_state.update_session(SceneSession {
            mode: "virtual-live".to_string(),
            scenario: Some(scenario_kind.slug().to_string()),
            seed: Some(args.seed),
            source: "sim".to_string(),
            tick_ms: Some(args.tick_delay_ms),
            control_state: None,
            control_detail: None,
            safety_class: None,
            independent_watchdog: None,
            motion_surface: None,
        });
        live_state.update_training_status(pete_server::LiveTrainingStatus {
            training_mode: inline_learning.training_mode_label().to_string(),
            ledger_path: Some(args.ledger.clone()),
            frames_written: 0,
            transitions_written: 0,
            models_loaded: models_loaded.clone(),
            model_modes: model_modes_from_flags(&flags),
            action_selector_mode: live_action_selector_label.clone(),
            weights_updating: inline_learning.is_enabled(),
        });
        live_state.update_behavior_nodes(runner.runtime.behavior_node_states());
        let server_state = live_state.clone();
        let reign_state = pete_server::ReignServerState::with_live_view(
            runner.runtime.reign_queue.clone(),
            &live_state,
        );
        let live_addr = args.live_addr;
        if args.live_tls {
            let cert_path = args.live_tls_cert.clone();
            let key_path = args.live_tls_key.clone();
            tokio::spawn(async move {
                if let Err(error) = pete_server::serve_live_view_with_reign_tls(
                    live_addr,
                    server_state,
                    reign_state,
                    cert_path,
                    key_path,
                )
                .await
                {
                    eprintln!("live robot HTTPS view server stopped: {error}");
                }
            });
        } else {
            tokio::spawn(async move {
                if let Err(error) =
                    pete_server::serve_live_view_with_reign(live_addr, server_state, reign_state)
                        .await
                {
                    eprintln!("live robot view server stopped: {error}");
                }
            });
        }
        let scheme = if args.live_tls { "https" } else { "http" };
        println!();
        println!("Pete Dream World is running.");
        if inline_learning.is_enabled() {
            println!(
                "Dream World is collecting experience and running {} inline learning.",
                inline_learning.mode.as_str()
            );
        } else {
            println!("Dream World is collecting experience.");
            println!("Models are not updated online in this run.");
            println!("Train later with `cargo run --bin pete -- train behavior ...`");
        }
        println!();
        println!("Desktop:");
        println!("  {scheme}://127.0.0.1:{}/view/3d", args.live_addr.port());
        println!();
        println!("Bound address:");
        println!("  {scheme}://{}/view/3d", args.live_addr);
        println!();
        println!("Scene JSON:");
        println!("  {scheme}://{}/view/scene", args.live_addr);
        if args.live_tls {
            println!();
            println!("If your headset warns about the certificate, trust the local dev certificate or install the generated CA/cert.");
            println!("This serves robot/dream-world sensor data on the LAN. Use only on trusted networks.");
        }
        for _ in 0..args.steps {
            let current_inline_learning = live_state.inline_learning();
            runner.runtime.inline_learning = current_inline_learning.clone();
            for node in live_state.behavior_nodes() {
                runner.runtime.apply_behavior_node_update(
                    &node.behavior_id,
                    &pete_behaviors::BehaviorNodeUpdate {
                        selected_regime: Some(node.selected_regime),
                        selected_hardcoded: Some(node.selected_hardcoded.clone()),
                        selected_model: node.selected_model.clone(),
                        checkpoint_path: node.checkpoint_path.clone(),
                        fallback_policy: Some(node.fallback_policy),
                        training_enabled: Some(node.training_enabled),
                    },
                );
            }
            let eye_frame = live_state.take_pending_retina_frame();
            if let Some(mut frame) = eye_frame {
                frame.source = Some("babylon-robot-eye".to_string());
                runner.world.set_retina_frame(Some(frame));
                live_state.record_ledger_write();
            } else {
                runner.world.set_retina_frame(None);
            }
            let mut live_snapshot = None;
            let mut runtime_events = Vec::new();
            runner
                .run_steps_observing_ticks(1, |snapshot, tick| {
                    live_snapshot = Some(snapshot.clone());
                    runtime_events = LiveViewState::runtime_tick_brain_events(snapshot, tick);
                    live_state.update_embodied_context(tick.frame.embodied_context());
                })
                .await?;
            live_state.update_behavior_nodes(runner.runtime.behavior_node_states());
            let runtime_map = runner.runtime.canonical_map();
            live_state.update_with_runtime_map(
                live_snapshot.unwrap_or(runner.world.snapshot().await?),
                runtime_map.as_ref(),
            );
            live_state.publish_brain_events(runtime_events);
            live_state.update_prod_state(runner.runtime.nudge_status());
            live_state.update_training_status(pete_server::LiveTrainingStatus {
                training_mode: current_inline_learning.training_mode_label().to_string(),
                ledger_path: Some(args.ledger.clone()),
                frames_written: runner.tick_count,
                transitions_written: runner.tick_count.saturating_sub(1),
                models_loaded: models_loaded.clone(),
                model_modes: model_modes_from_flags(&flags),
                action_selector_mode: live_action_selector_label.clone(),
                weights_updating: current_inline_learning.is_enabled(),
            });
            tokio::time::sleep(Duration::from_millis(args.tick_delay_ms)).await;
        }
    } else {
        runner.run_steps(args.steps).await?;
    }
    println!(
        "sim complete: {} ticks, seed {}, ledger {}, action_selector {:?}, danger_mode {:?}, charge_mode {:?}, action_value_mode {:?}, eye_next_mode {:?}, ear_next_mode {:?}, experience_mode {:?}",
        runner.tick_count,
        args.seed,
        args.ledger,
        args.action_selector,
        args.danger_mode,
        args.charge_mode,
        args.action_value_mode,
        args.eye_next_mode,
        args.ear_next_mode,
        args.experience_mode
    );
    Ok(())
}

fn loaded_model_names(report: &RuntimeModelLoadReport) -> Vec<String> {
    let mut names = report
        .loaded_checkpoints
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    names.sort();
    names
}

fn inline_learning_config_from_sim_args(args: &SimArgs) -> Result<InlineLearningConfig> {
    let mut mode = InlineLearningMode::from(args.inline_learning_mode);
    if args.inline_learning && mode == InlineLearningMode::Off {
        mode = InlineLearningMode::WorldOutcome;
    }
    Ok(InlineLearningConfig {
        mode,
        behaviors: inline_learning_behaviors(args.inline_behaviors.as_deref())?,
        max_train_steps_per_tick: args.inline_train_steps_per_tick,
    })
}

fn inline_learning_behaviors(list: Option<&str>) -> Result<InlineLearningBehaviors> {
    let Some(list) = list else {
        return Ok(InlineLearningBehaviors::default());
    };
    if list.trim().is_empty() || list.trim().eq_ignore_ascii_case("all") {
        return Ok(InlineLearningBehaviors::default());
    }
    let mut behaviors = InlineLearningBehaviors {
        danger: false,
        charge: false,
        future: false,
        action_value: false,
        eye_next: false,
        ear_next: false,
        experience: false,
    };
    for raw in list.split(',') {
        match raw.trim().replace('-', "_").as_str() {
            "" => {}
            "danger" => behaviors.danger = true,
            "charge" => behaviors.charge = true,
            "future" => behaviors.future = true,
            "action_value" => behaviors.action_value = true,
            "eye_next" => behaviors.eye_next = true,
            "ear_next" => behaviors.ear_next = true,
            "experience" => behaviors.experience = true,
            other => anyhow::bail!(
                "unknown inline behavior '{other}', expected one of danger,charge,future,action_value,eye_next,ear_next,experience"
            ),
        }
    }
    Ok(behaviors)
}

fn live_scene_metadata_from_scenario(metadata: &pete_sim::ScenarioMetadata) -> LiveSceneMetadata {
    LiveSceneMetadata {
        arena: Some(SceneArena {
            width_m: metadata.arena.width_m,
            height_m: metadata.arena.height_m,
        }),
        objects: metadata
            .objects
            .iter()
            .map(|object| SceneObject {
                id: object.id.clone(),
                kind: match &object.kind {
                    SimObjectKind::Obstacle => "obstacle",
                    SimObjectKind::Charger => "charger",
                    SimObjectKind::Person { .. } => "person",
                    SimObjectKind::SoundSource { .. } => "speaker",
                    SimObjectKind::Landmark { .. } => "landmark",
                }
                .to_string(),
                x_m: object.x_m,
                y_m: object.y_m,
                radius_m: object.radius_m,
                label: Some(object.label.clone()),
                color_rgb: Some(object.color_rgb),
            })
            .collect(),
        sensor_calibration: Some(SceneSensorCalibration {
            compact_depth_fov_rad: std::f32::consts::PI * 0.68,
            ..SceneSensorCalibration::sim_default()
        }),
    }
}

const DEFAULT_REAL_DEPTH_CAMERA_YAW_DEG: f32 = -90.0;

fn real_robot_depth_calibration_from_env() -> SceneSensorCalibration {
    let height_m = env_f32("PETE_DEPTH_CAMERA_HEIGHT_M", 0.46);
    let forward_m = env_f32("PETE_DEPTH_CAMERA_FORWARD_M", 0.0);
    let left_m = env_f32("PETE_DEPTH_CAMERA_LEFT_M", 0.0);
    let pitch_rad = env_f32("PETE_DEPTH_CAMERA_PITCH_DEG", 0.0).to_radians();
    let roll_rad = env_f32("PETE_DEPTH_CAMERA_ROLL_DEG", 0.0).to_radians();
    let yaw_rad = env_f32(
        "PETE_DEPTH_CAMERA_YAW_DEG",
        DEFAULT_REAL_DEPTH_CAMERA_YAW_DEG,
    )
    .to_radians();
    let color_offset_x_px = env_i32("PETE_DEPTH_COLOR_OFFSET_X_PX", 3);
    let color_offset_y_px = env_i32("PETE_DEPTH_COLOR_OFFSET_Y_PX", 7);
    SceneSensorCalibration {
        compact_depth_beam_count: env_usize("PETE_COMPACT_DEPTH_BEAM_COUNT", 32),
        compact_depth_fov_rad: env_f32("PETE_DEPTH_FOV_DEG", 122.0).to_radians(),
        depth_scale: env_f32("PETE_DEPTH_SCALE", 1.0),
        point_y_m: height_m,
        depth_forward_offset_m: forward_m,
        depth_pitch_down_rad: pitch_rad,
        camera_forward_m: forward_m,
        camera_left_m: left_m,
        camera_height_m: height_m,
        camera_pitch_rad: pitch_rad,
        camera_roll_rad: roll_rad,
        camera_yaw_rad: yaw_rad,
        color_offset_x_px,
        color_offset_y_px,
    }
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(default)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_i32(name: &str, default: i32) -> i32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .unwrap_or(default)
}

async fn run_sim_curriculum(args: SimCurriculumArgs) -> Result<()> {
    if args.validation_ratio < 0.0 || args.test_ratio < 0.0 {
        anyhow::bail!("validation and test ratios must be non-negative");
    }
    if args.validation_ratio + args.test_ratio >= 1.0 {
        anyhow::bail!("validation_ratio + test_ratio must be less than 1.0");
    }

    let kind = ScenarioKind::from(args.scenario);
    let ledger = JsonlLedger::new(&args.out);
    let mut total_ticks = 0usize;
    let mut capture_count = 0usize;
    let mut manifest_episodes = Vec::with_capacity(args.episodes);

    for episode_index in 0..args.episodes {
        let episode_seed = args.seed.saturating_add(episode_index as u64);
        let scenario = build_scenario(ScenarioConfig::new(kind, episode_seed));
        let object_count = scenario.metadata.objects.len();
        let object_summary = scenario_object_summary(&scenario.metadata.objects);
        let runtime = default_runtime(ledger.clone(), &args.llm)?;
        let mut runner = SimRunner::new(runtime, scenario.world, scenario.motors);
        runner.tick_ms = args.tick_ms;
        let mut capture_path_for_manifest = None;

        if let Some(root) = &args.capture_root {
            let mut snapshots = Vec::with_capacity(args.steps);
            runner
                .run_steps_observing(args.steps, |snapshot| snapshots.push(snapshot.clone()))
                .await?;
            let capture_path = Path::new(root).join(format!("episode-{episode_index:03}"));
            capture_path_for_manifest = Some(capture_path.to_string_lossy().to_string());
            let mut writer =
                CaptureWriter::create(&capture_path, CaptureSource::Sim, Some(args.tick_ms))
                    .await?;
            writer.manifest_mut().scenario = Some(scenario.metadata.clone());
            for snapshot in snapshots {
                writer
                    .append_snapshot(snapshot.body.last_update_ms, snapshot, Vec::new())
                    .await?;
            }
            writer.finish().await?;
            capture_count = capture_count.saturating_add(1);
        } else {
            runner.run_steps(args.steps).await?;
        }
        total_ticks = total_ticks.saturating_add(runner.tick_count);
        manifest_episodes.push(serde_json::json!({
            "index": episode_index,
            "split": curriculum_split(
                episode_index,
                args.episodes,
                args.validation_ratio,
                args.test_ratio,
            ),
            "scenario": kind.slug(),
            "seed": episode_seed,
            "steps": args.steps,
            "ticks": runner.tick_count,
            "arena": scenario.metadata.arena,
            "spawn": {
                "x_m": scenario.metadata.body.odometry.x_m,
                "y_m": scenario.metadata.body.odometry.y_m,
                "heading_rad": scenario.metadata.body.odometry.heading_rad,
                "battery_level": scenario.metadata.body.battery_level,
            },
            "object_count": object_count,
            "objects": object_summary,
            "capture": capture_path_for_manifest,
        }));
        println!(
            "episode {} complete: scenario {}, seed {}, ticks {}, objects {}",
            episode_index,
            kind.slug(),
            episode_seed,
            runner.tick_count,
            object_count
        );
    }

    let manifest = serde_json::json!({
        "schema_version": 1,
        "scenario": kind.slug(),
        "base_seed": args.seed,
        "episodes": args.episodes,
        "steps_per_episode": args.steps,
        "tick_ms": args.tick_ms,
        "ledger": args.out,
        "capture_root": args.capture_root,
        "splits": {
            "train": manifest_episodes.iter().filter(|episode| episode["split"] == "train").count(),
            "validation": manifest_episodes.iter().filter(|episode| episode["split"] == "validation").count(),
            "test": manifest_episodes.iter().filter(|episode| episode["split"] == "test").count(),
            "validation_ratio": args.validation_ratio,
            "test_ratio": args.test_ratio,
        },
        "episodes_detail": manifest_episodes,
    });
    fs::create_dir_all(&args.out)?;
    let manifest_path = Path::new(&args.out).join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;

    let transitions = ledger.transitions().await?;
    println!(
        "sim curriculum complete: scenario {}, episodes {}, ticks {}, ledger {}, transitions {}, captures {}, manifest {}",
        kind.slug(),
        args.episodes,
        total_ticks,
        args.out,
        transitions.len(),
        capture_count,
        manifest_path.display()
    );
    Ok(())
}
