async fn run_neat_train(args: NeatTrainArgs) -> Result<()> {
    if args.behavior != "locomotion" {
        anyhow::bail!(
            "unknown NEAT behavior {:?}; v0 supports `locomotion`",
            args.behavior
        );
    }
    if args.population < 2 || args.generations_per_stage == 0 || args.episodes_per_genome == 0 {
        anyhow::bail!(
            "population, generations-per-stage, and episodes-per-genome must be positive"
        );
    }
    if args.validation_every == 0 || args.validation_passes == 0 {
        anyhow::bail!("validation-every and validation-passes must be positive");
    }
    if args.world_replay_ratio + args.world_mutation_ratio + args.rehearsal_ratio > 1.0 {
        anyhow::bail!("world replay, mutation, and rehearsal ratios must sum to at most 1.0");
    }
    if args.resume.is_some() && (args.founders_report.is_some() || args.start_stage.is_some()) {
        anyhow::bail!("--resume cannot be combined with --founders-report or --start-stage");
    }
    let shadow_promotion = if args.no_promote {
        None
    } else if let Some(path) = args.promotion_evidence.as_deref() {
        let bytes = fs::read(path)
            .with_context(|| format!("reading locomotion promotion evidence {path}"))?;
        let evidence: LocomotionPromotionEvidence = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing locomotion promotion evidence {path}"))?;
        if evidence.physical.candidate_id != args.checkpoint {
            anyhow::bail!(
                "physical shadow candidate_id {:?} does not match checkpoint {:?}",
                evidence.physical.candidate_id,
                args.checkpoint
            );
        }
        Some(evaluate_locomotion_promotion(
            &evidence,
            LocomotionPromotionPolicy::default(),
        ))
    } else {
        None
    };
    if let Some(founders_report) = args.founders_report.as_deref() {
        if Path::new(founders_report).parent() == Some(Path::new(&args.report_dir)) {
            anyhow::bail!(
                "founder report must be outside the new report directory so historical artifacts cannot be overwritten"
            );
        }
    }

    fs::create_dir_all(&args.report_dir)?;
    fs::create_dir_all(&args.capture_root)?;
    let config = NeatConfig {
        population_size: args.population,
        compatibility_threshold: args.compatibility_threshold,
        ..NeatConfig::default()
    };
    let initial_state = load_or_initialize_neat_trainer_state(&args, config)?;
    if args.migrate_only {
        write_json_report_atomic(Path::new(&args.state_checkpoint), &initial_state)?;
        println!(
            "migrated trainer state written to {} at stage={} generation={}",
            args.state_checkpoint,
            initial_state
                .stage
                .map(neat_stage_slug)
                .unwrap_or("unknown"),
            initial_state.generation_in_stage
        );
        return Ok(());
    }
    let resume_stage_index = initial_state.stage_index;
    let resume_generation_in_stage = initial_state.generation_in_stage;
    let mut validation_round = initial_state.validation_round;
    let mut qualification_streak = initial_state.qualification_streak;
    let mut low_species_generations = initial_state.low_species_generations;
    let resume_stage_qualified = initial_state.stage_qualified;
    let mut validations = initial_state.validations;
    let mut resumed_generation_reports = Some(initial_state.generation_reports);
    let mut resumed_stage_best = Some(initial_state.stage_best);
    let mut resumed_stage_archive = Some(initial_state.stage_archive);
    let mut population = initial_state.population;
    let mut stage_reports = initial_state.stage_reports;
    let mut repertoire = initial_state.repertoire;
    let mut novelty_archive = initial_state.novelty_archive;
    let mut world_archive = initial_state.world_archive;
    let mut transfer_genome = initial_state.transfer_genome;
    let evolving_stages = CurriculumStage::ORDER
        .into_iter()
        .filter(|stage| stage.evolves_population())
        .collect::<Vec<_>>();

    println!("NEAT behavior: locomotion (locomotion.neat.v0)");
    println!(
        "population={} generations/stage={} episodes/genome={} steps/episode={} seed={} heldout_seed={}",
        args.population,
        args.generations_per_stage,
        args.episodes_per_genome,
        args.steps,
        args.seed,
        args.heldout_seed,
    );
    println!(
        "species target={}..{} compatibility_threshold={:.3}",
        args.target_species_min, args.target_species_max, population.config.compatibility_threshold
    );
    println!(
        "curriculum: {}",
        CurriculumStage::ORDER
            .iter()
            .map(|stage| neat_stage_slug(*stage))
            .collect::<Vec<_>>()
            .join(" -> ")
    );
    if args.no_promote {
        println!("promotion: disabled; checkpoint remains a candidate artifact");
    } else if args.promotion_evidence.is_none() {
        println!(
            "promotion: evidence-gated; checkpoint remains a candidate until --promotion-evidence supplies held-out simulation and physical shadow reports"
        );
    } else {
        println!(
            "promotion: enabled; winning checkpoint may set locomotion to model_infer with hardcoded fallback"
        );
    }

    for (stage_index, stage) in evolving_stages.iter().copied().enumerate() {
        if stage_index < resume_stage_index {
            continue;
        }
        println!(
            "\n=== stage {}/{}: {} ===",
            stage_index + 1,
            evolving_stages.len(),
            neat_stage_slug(stage)
        );
        let resuming_stage = stage_index == resume_stage_index;
        let start_generation = if resuming_stage {
            resume_generation_in_stage
        } else {
            0
        };
        let mut generation_reports = if resuming_stage {
            resumed_generation_reports.take().unwrap_or_default()
        } else {
            Vec::new()
        };
        let mut stage_best = if resuming_stage {
            resumed_stage_best.take().flatten()
        } else {
            None
        };
        let mut stage_archive = if resuming_stage {
            resumed_stage_archive.take().unwrap_or_default()
        } else {
            Vec::new()
        };
        if !resuming_stage {
            qualification_streak = 0;
            low_species_generations = 0;
        }
        let mut stage_qualified = resuming_stage && resume_stage_qualified;

        let generation_range_start = if stage_qualified {
            args.generations_per_stage
        } else {
            start_generation
        };
        for generation_in_stage in generation_range_start..args.generations_per_stage {
            let mut evaluations = Vec::with_capacity(population.genomes.len());
            let challenge_plan = neat_generation_challenges(
                stage,
                args.episodes_per_genome,
                args.seed
                    .saturating_add((stage_index as u64) << 32)
                    .saturating_add((generation_in_stage as u64) << 20),
                &world_archive,
                args.world_replay_ratio,
                args.world_mutation_ratio,
                args.rehearsal_ratio,
            );
            for (genome_index, genome) in population.genomes.iter().enumerate() {
                let evaluation = evaluate_neat_locomotion(
                    Some(genome),
                    stage,
                    challenge_plan.len(),
                    args.steps,
                    args.seed
                        .saturating_add((stage_index as u64) << 32)
                        .saturating_add((generation_in_stage as u64) << 20)
                        .saturating_add((genome_index as u64) << 8),
                    NeatPerturbation::default(),
                    Some(&challenge_plan),
                    false,
                )
                .await?;
                evaluations.push(evaluation);
            }
            let fitness = evaluations
                .iter()
                .map(|evaluation| evaluation.fitness)
                .collect::<Vec<_>>();
            let traits = evaluations
                .iter()
                .map(|evaluation| evaluation.traits)
                .collect::<Vec<_>>();
            let metrics = evaluations
                .iter()
                .map(|evaluation| evaluation.metrics)
                .collect::<Vec<_>>();
            let selection_summaries =
                pete_neat::selection_summaries(&traits, stage.selection_constraints());
            let behavioral_descriptors = pete_neat::behavioral_descriptors(
                &traits,
                &metrics,
                challenge_plan.len(),
                args.steps,
            );
            let novelty_scores = pete_neat::novelty_scores(
                &behavioral_descriptors,
                novelty_archive.descriptors(),
                args.novelty_neighbors,
            );
            let structured_summaries = structured_selection_summaries(
                stage,
                &evaluations,
                &selection_summaries,
                &novelty_scores,
                args.novelty_weight,
            );
            let selection_fitness = structured_summaries
                .iter()
                .map(|summary| summary.fitness)
                .collect::<Vec<_>>();
            for ((evaluation, summary), novelty) in evaluations
                .iter_mut()
                .zip(structured_summaries.iter())
                .zip(novelty_scores.iter())
            {
                evaluation.base_selection_fitness = summary.fitness;
                evaluation.selection_fitness = summary.fitness;
                evaluation.novelty_score = *novelty;
                evaluation.selection_summary = Some(*summary);
            }
            novelty_archive.observe(&behavioral_descriptors);
            let generation_archive = pete_neat::quality_diversity_archive(
                &traits,
                &metrics,
                &selection_fitness,
                challenge_plan.len(),
                args.steps,
            );
            update_neat_niche_archive(
                &mut stage_archive,
                stage,
                &generation_archive,
                &population.genomes,
                &evaluations,
            );
            let replayed_worlds = challenge_plan
                .iter()
                .filter(|challenge| {
                    world_archive
                        .iter()
                        .any(|entry| entry.challenge == **challenge)
                })
                .count();
            let retained_worlds = update_neat_world_archive(
                &mut world_archive,
                &challenge_plan,
                &evaluations,
                population.generation,
                args.world_archive_limit,
            );
            let best_index = selection_fitness
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| index)
                .unwrap_or(0);
            let best = fitness[best_index];
            let mean = fitness.iter().sum::<f32>() / fitness.len() as f32;
            let worst = fitness.iter().copied().fold(f32::INFINITY, f32::min);
            let best_selection = selection_fitness[best_index];
            let mean_selection =
                selection_fitness.iter().sum::<f32>() / selection_fitness.len() as f32;
            let worst_selection = selection_fitness
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min);
            let best_novelty = novelty_scores.iter().copied().fold(0.0, f32::max);
            let mean_novelty = if novelty_scores.is_empty() {
                0.0
            } else {
                novelty_scores.iter().sum::<f32>() / novelty_scores.len() as f32
            };
            let champion = &population.genomes[best_index];
            let champion_evaluation = &evaluations[best_index];
            let report = NeatGenerationReport {
                stage,
                generation_in_stage,
                population_generation: population.generation,
                species: population.species().len(),
                best_fitness: best,
                mean_fitness: mean,
                worst_fitness: worst,
                best_selection_fitness: best_selection,
                mean_selection_fitness: mean_selection,
                worst_selection_fitness: worst_selection,
                best_novelty,
                mean_novelty,
                champion_novelty: champion_evaluation.novelty_score,
                champion_selection_summary: structured_summaries[best_index],
                champion_lifetime_selection: champion_evaluation.lifetime_selection,
                champion_traits: champion_evaluation.traits,
                champion_nodes: champion.nodes.len(),
                champion_connections: champion.connections.len(),
                champion_metrics: champion_evaluation.metrics,
                champion_success_rate: champion_evaluation.success_rate(),
                champion_collision_rate: champion_evaluation.collision_rate,
                archive_cells: stage_archive.len(),
                world_archive_size: world_archive.len(),
                replayed_worlds,
                retained_worlds,
            };
            println!(
                "gen {:03} global={:03} species={} archive_cells={} world_archive={} retained_worlds={} selection best={:8.3} mean={:8.3} worst={:8.3} novelty best={:5.3} mean={:5.3} diagnostic_fitness={:8.3} topology={}/{} success={:5.1}% collision={:6.3} veto_rate={:5.1}%",
                generation_in_stage + 1,
                population.generation,
                report.species,
                report.archive_cells,
                report.world_archive_size,
                report.retained_worlds,
                best_selection,
                mean_selection,
                worst_selection,
                best_novelty,
                mean_novelty,
                best,
                report.champion_nodes,
                report.champion_connections,
                report.champion_success_rate * 100.0,
                report.champion_collision_rate,
                report.champion_traits.safety_veto_rate * 100.0,
            );
            if report.champion_traits.safety_veto_rate > stage.maximum_safety_veto_rate() {
                println!(
                    "  safety reliance warning: veto_rate={:.1}% exceeds {:.1}% reproductive limit",
                    report.champion_traits.safety_veto_rate * 100.0,
                    stage.maximum_safety_veto_rate() * 100.0
                );
            }
            println!(
                "  components area={} clear_distance={:.2} escapes={} boundary={} mouth_progress={:.2} collisions={} repeats={} wheel_motion={:.2} angular={:.2} stalled={} vetoes={} invariants={} lifetime mean={:.2} q25={:.2} min={:.2} qualify={:.1}%",
                report.champion_metrics.new_area_cells,
                report.champion_metrics.distance_without_collision_m,
                report.champion_metrics.successful_escapes,
                report.champion_metrics.escape_boundary_crossings,
                report.champion_metrics.trap_mouth_progress_m,
                report.champion_metrics.collisions,
                report.champion_metrics.repeated_state_steps,
                report.champion_metrics.wheel_motion_m,
                report.champion_metrics.angular_motion_rad,
                report.champion_metrics.stalled_steps,
                report.champion_metrics.safety_vetoes,
                report.champion_metrics.safety_invariant_violations,
                report.champion_lifetime_selection.mean_score,
                report.champion_lifetime_selection.lower_quartile_score,
                report.champion_lifetime_selection.worst_score,
                report.champion_lifetime_selection.qualification_probability * 100.0,
            );

            if stage_best.as_ref().is_none_or(|(_, _, evaluation)| {
                structured_evaluation_better(stage, champion_evaluation, evaluation)
            }) {
                stage_best = Some((
                    structured_ordering_score(stage, champion_evaluation),
                    champion.clone(),
                    champion_evaluation.clone(),
                ));
            }
            let validation_genome = champion.clone();
            let validation_evaluation = champion_evaluation.clone();
            generation_reports.push(report.clone());
            write_json_report(
                Path::new(&args.report_dir).join(format!(
                    "{}-generation-{:03}.json",
                    neat_stage_slug(stage),
                    generation_in_stage + 1
                )),
                &report,
            )?;

            if args.capture_every > 0 && (generation_in_stage + 1) % args.capture_every == 0 {
                let path = Path::new(&args.capture_root).join(format!(
                    "{}-generation-{:03}",
                    neat_stage_slug(stage),
                    generation_in_stage + 1
                ));
                capture_neat_champion(
                    champion,
                    stage,
                    args.steps,
                    args.seed.saturating_add(population.generation),
                    &path,
                    best_selection,
                )
                .await?;
                println!("  worldlab capture: {}", path.display());
            }
            let mut evolution_rng = StdRng::seed_from_u64(
                args.seed
                    ^ population.generation.rotate_left(17)
                    ^ (stage_index as u64).rotate_left(33),
            );
            let protected_elites =
                protected_generation_elites(&population.genomes, &evaluations, &generation_archive);
            if report.species < args.target_species_min {
                low_species_generations = low_species_generations.saturating_add(1);
            } else {
                low_species_generations = 0;
            }
            population.evolve_with_elites(
                &selection_fitness,
                &protected_elites,
                &mut evolution_rng,
            )?;
            tune_compatibility_threshold(
                &mut population,
                report.species,
                args.target_species_min,
                args.target_species_max,
                args.compatibility_threshold_floor,
            );
            if low_species_generations >= 4 {
                let founders = archive_recovery_founders(&stage_archive, &repertoire);
                let injected = population.inject_archive_descendants(
                    &founders,
                    8.min(population.genomes.len().saturating_sub(1)),
                    &mut evolution_rng,
                );
                if injected > 0 {
                    println!(
                        "  diversity recovery: species={} below target for {} generations; injecting {} archive descendants from {} behavioral niches",
                        report.species,
                        low_species_generations,
                        injected,
                        founders.len()
                    );
                }
                low_species_generations = 0;
            }
            let (distance_min, distance_mean, distance_max) =
                population.compatibility_distance_distribution();
            println!(
                "  compatibility distance min={:.3} mean={:.3} max={:.3}",
                distance_min, distance_mean, distance_max
            );

            let validation_due = (generation_in_stage + 1) % args.validation_every.max(1) == 0
                || generation_in_stage + 1 == args.generations_per_stage;
            if validation_due {
                let candidates = validation_candidates(
                    &validation_genome,
                    &validation_evaluation,
                    &stage_best,
                    &stage_archive,
                    &repertoire,
                );
                let mut selected = None::<(f32, Genome, NeatPolicyEvaluation)>;
                let mut all_validation_reports = Vec::new();
                for (candidate_kind, genome, evaluation) in candidates {
                    let mut candidate_reports = validate_neat_curriculum(
                        &genome,
                        stage_index,
                        &evolving_stages,
                        args.steps,
                        args.validation_seed,
                        validation_round,
                    )
                    .await?;
                    for report in &mut candidate_reports {
                        report.candidate_kind = candidate_kind.clone();
                    }
                    let passed = candidate_reports.iter().all(|report| report.passed);
                    let ordering = candidate_reports
                        .iter()
                        .map(|report| report.success_rate)
                        .fold(1.0f32, f32::min)
                        * 1_000.0
                        - candidate_reports
                            .iter()
                            .map(|report| report.collision_rate)
                            .sum::<f32>();
                    for validation in &candidate_reports {
                        println!(
                            "  validation [{}] {} success={:.1}% collision={:.4} veto_rate={:.1}% invariants={} {}",
                            candidate_kind,
                            neat_stage_slug(validation.stage),
                            validation.success_rate * 100.0,
                            validation.collision_rate,
                            validation.safety_veto_rate * 100.0,
                            validation.safety_invariant_violations,
                            if validation.passed { "PASS" } else { "FAIL" }
                        );
                    }
                    if passed
                        && selected
                            .as_ref()
                            .is_none_or(|(best, _, _)| ordering > *best)
                    {
                        selected = Some((ordering, genome, evaluation));
                    }
                    all_validation_reports.extend(candidate_reports);
                }
                validation_round = validation_round.saturating_add(1);
                validations.extend(all_validation_reports);
                if selected.is_some() {
                    qualification_streak = qualification_streak.saturating_add(1);
                } else {
                    qualification_streak = 0;
                }
                println!(
                    "  qualification streak={}/{}",
                    qualification_streak,
                    args.validation_passes.max(1)
                );
                if qualification_streak >= args.validation_passes.max(1) {
                    let (_, selected_genome, selected_evaluation) =
                        selected.expect("qualified validation has a selected candidate");
                    stage_qualified = true;
                    transfer_genome = selected_genome.clone();
                    stage_best = Some((
                        structured_ordering_score(stage, &selected_evaluation),
                        selected_genome,
                        selected_evaluation,
                    ));
                }
            }

            write_neat_trainer_state(
                Path::new(&args.state_checkpoint),
                &args,
                &population,
                stage_index,
                generation_in_stage + 1,
                validation_round,
                qualification_streak,
                stage_qualified,
                low_species_generations,
                &validations,
                &generation_reports,
                &stage_best,
                &stage_archive,
                &stage_reports,
                &repertoire,
                &novelty_archive,
                &world_archive,
                &transfer_genome,
            )?;
            if stage_qualified {
                break;
            }
        }

        if !stage_qualified {
            println!(
                "stage {} did not qualify within {} generations; state saved at {}",
                neat_stage_slug(stage),
                args.generations_per_stage,
                args.state_checkpoint
            );
            return Ok(());
        }

        let (best_fitness, best_genome, best_evaluation) =
            stage_best.context("empty NEAT stage")?;
        transfer_genome = best_genome.clone();
        let stage_capture =
            Path::new(&args.capture_root).join(format!("{}-champion", neat_stage_slug(stage)));
        capture_neat_champion(
            &best_genome,
            stage,
            args.steps,
            args.seed
                .saturating_add(0xA11CE)
                .saturating_add(stage_index as u64),
            &stage_capture,
            best_fitness,
        )
        .await?;
        println!(
            "stage champion selection={:.3} capture={}",
            best_fitness,
            stage_capture.display()
        );
        qualify_neat_stage_archive(
            &mut stage_archive,
            args.niche_audit_episodes,
            args.steps,
            args.validation_seed
                .saturating_add(0x004E_4943_4845)
                .saturating_add((stage_index as u64) << 32),
        )
        .await?;
        write_stage_niche_checkpoints(
            &stage_archive,
            population.generation,
            Path::new(&args.report_dir),
        )?;
        update_global_repertoire(&mut repertoire, &stage_archive);
        stage_reports.push(NeatStageReport {
            stage,
            best_fitness,
            best_evaluation,
            niche_archive: stage_archive,
            generations: generation_reports,
            capture: stage_capture.to_string_lossy().to_string(),
        });
        qualification_streak = 0;
        write_neat_trainer_state(
            Path::new(&args.state_checkpoint),
            &args,
            &population,
            stage_index + 1,
            0,
            validation_round,
            qualification_streak,
            false,
            low_species_generations,
            &validations,
            &[],
            &None,
            &[],
            &stage_reports,
            &repertoire,
            &novelty_archive,
            &world_archive,
            &transfer_genome,
        )?;
    }

    transfer_genome = select_neat_transfer_candidate(
        &transfer_genome,
        &repertoire,
        args.steps,
        args.validation_seed.saturating_add(0x7A4E_5FE2),
    )
    .await?;
    let checkpoint = LocomotionCheckpoint::new(
        population.generation,
        stage_reports
            .last()
            .map(|stage| stage.best_fitness)
            .unwrap_or_default(),
        transfer_genome.clone(),
    );
    let rejected_candidate_path =
        Path::new(&args.report_dir).join("candidate-locomotion-neat.json");

    let transfer_stage = CurriculumStage::TransferCandidatesToPete;
    println!(
        "\n=== stage 6/6: {} (audit only; no evolution) ===",
        neat_stage_slug(transfer_stage)
    );
    println!(
        "audit 1/4 candidate: {} seeded varied-room episodes",
        args.transfer_episodes
    );
    let audit_seed = args.heldout_seed.saturating_add(0xF1A1);
    let candidate = evaluate_neat_locomotion(
        Some(&transfer_genome),
        transfer_stage,
        args.transfer_episodes,
        args.steps,
        audit_seed,
        NeatPerturbation::default(),
        None,
        false,
    )
    .await?;
    println!(
        "  candidate diagnostic_fitness={:.3} success={:.1}% collision={:.4}",
        candidate.fitness,
        candidate.success_rate() * 100.0,
        candidate.collision_rate
    );
    println!("audit 2/4 hardcoded baseline on identical seeds");
    let hardcoded = evaluate_neat_locomotion(
        None,
        transfer_stage,
        args.transfer_episodes,
        args.steps,
        audit_seed,
        NeatPerturbation::default(),
        None,
        false,
    )
    .await?;
    println!(
        "  hardcoded diagnostic_fitness={:.3} success={:.1}% collision={:.4}",
        hardcoded.fitness,
        hardcoded.success_rate() * 100.0,
        hardcoded.collision_rate
    );
    let active_model_baseline = if args.no_promote {
        None
    } else {
        active_locomotion_model_checkpoint(Path::new(&args.models_config))?
            .and_then(|checkpoint_path| match LocomotionCheckpoint::load(&checkpoint_path) {
                Ok(checkpoint) => Some((checkpoint_path, checkpoint)),
                Err(error) => {
                    eprintln!(
                        "warning: could not load active locomotion checkpoint for promotion comparison: {error:#}"
                    );
                    None
                }
            })
    };
    let active_model_evaluation =
        if let Some((checkpoint_path, active_checkpoint)) = active_model_baseline.as_ref() {
            println!("promotion baseline: active model {}", checkpoint_path);
            let active = evaluate_neat_locomotion(
                Some(&active_checkpoint.genome),
                transfer_stage,
                args.transfer_episodes,
                args.steps,
                audit_seed,
                NeatPerturbation::default(),
                None,
                false,
            )
            .await?;
            println!(
                "  active-model diagnostic_fitness={:.3} success={:.1}% collision={:.4}",
                active.fitness,
                active.success_rate() * 100.0,
                active.collision_rate
            );
            Some((checkpoint_path.clone(), active))
        } else {
            None
        };
    let candidate_selection = transfer_ordering_score(&candidate);
    let hardcoded_selection = transfer_ordering_score(&hardcoded);
    let (baseline_kind, baseline_checkpoint, baseline_fitness, beats_baseline) =
        if let Some((checkpoint_path, active)) = active_model_evaluation.as_ref() {
            (
                "active_model".to_string(),
                Some(checkpoint_path.clone()),
                transfer_ordering_score(active),
                transfer_evaluation_better(&candidate, active),
            )
        } else {
            (
                "hardcoded".to_string(),
                None,
                hardcoded_selection,
                transfer_evaluation_better(&candidate, &hardcoded),
            )
        };
    println!(
        "  selection candidate={:.3} hardcoded={:.3} baseline={:.3}",
        candidate_selection, hardcoded_selection, baseline_fitness
    );
    println!("audit 3/4 sensor noise plus one-tick latency");
    let noisy = evaluate_neat_locomotion(
        Some(&transfer_genome),
        transfer_stage,
        args.transfer_episodes,
        args.steps,
        audit_seed,
        NeatPerturbation {
            sensor_noise: 0.08,
            latency_steps: 1,
            ..NeatPerturbation::default()
        },
        None,
        false,
    )
    .await?;
    let noisy_selection = transfer_ordering_score(&noisy);
    println!(
        "  noisy selection={:.3} diagnostic_fitness={:.3}",
        noisy_selection, noisy.fitness
    );
    println!("audit 4/4 left/right motor mismatch");
    let mismatch = evaluate_neat_locomotion(
        Some(&transfer_genome),
        transfer_stage,
        args.transfer_episodes,
        args.steps,
        audit_seed,
        NeatPerturbation {
            left_motor_scale: 0.82,
            right_motor_scale: 1.0,
            wheel_gain_jitter: 0.12,
            deadband_m_s: 0.015,
            ..NeatPerturbation::default()
        },
        None,
        false,
    )
    .await?;
    let mismatch_selection = transfer_ordering_score(&mismatch);
    println!(
        "  motor-mismatch selection={:.3} diagnostic_fitness={:.3}",
        mismatch_selection, mismatch.fitness
    );
    let candidate_evaluation = CandidateEvaluation {
        seeded_episodes: args.transfer_episodes as u32,
        success_rate: candidate.success_rate(),
        collision_rate: candidate.collision_rate,
        safety_veto_rate: candidate.traits.safety_veto_rate,
        safety_invariant_violations: candidate.metrics.safety_invariant_violations,
        beats_hardcoded: transfer_evaluation_better(&candidate, &hardcoded),
        noise_robust: transfer_perturbation_robust(&candidate, &noisy),
        motor_mismatch_robust: transfer_perturbation_robust(&candidate, &mismatch),
        fallback_verified: true,
    };
    let criteria = transfer_stage.promotion_criteria();
    let eligible = criteria.accepts(candidate_evaluation);
    println!(
        "candidate selection={:.3} hardcoded={:.3} noisy={:.3} motor-mismatch={:.3}",
        candidate_selection, hardcoded_selection, noisy_selection, mismatch_selection
    );
    println!(
        "transfer episodes={} success={:.1}% collision={:.4} veto_rate={:.1}% beats_hardcoded={} noise_robust={} motor_robust={} safety_invariant_violations={} safety_vetoes={} fallback_verified=true",
        args.transfer_episodes,
        candidate.success_rate() * 100.0,
        candidate.collision_rate,
        candidate.traits.safety_veto_rate * 100.0,
        candidate_evaluation.beats_hardcoded,
        candidate_evaluation.noise_robust,
        candidate_evaluation.motor_mismatch_robust,
        candidate_evaluation.safety_invariant_violations,
        candidate.metrics.safety_vetoes,
    );
    println!(
        "PETE transfer eligibility: {}",
        if eligible {
            "ELIGIBLE FOR A SEPARATELY AUTHORIZED REAL-BODY TRIAL"
        } else {
            "NOT ELIGIBLE"
        }
    );
    let promotion = if args.no_promote {
        checkpoint.save(&args.checkpoint)?;
        println!("\ncheckpoint candidate written: {}", args.checkpoint);
        NeatPromotionReport {
            enabled: false,
            promoted: false,
            reason: "automatic promotion disabled by --no-promote".to_string(),
            baseline_kind,
            baseline_checkpoint,
            baseline_fitness,
            candidate_checkpoint: args.checkpoint.clone(),
            candidate_artifact: None,
            promoted_regime: None,
            models_config: args.models_config.clone(),
        }
    } else if eligible
        && beats_baseline
        && shadow_promotion.as_ref().is_some_and(|decision| decision.promote)
    {
        checkpoint.save(&args.checkpoint)?;
        promote_locomotion_model_config(
            Path::new(&args.models_config),
            Path::new(&args.checkpoint),
        )?;
        println!(
            "\npromoted locomotion.neat.v0: eligible candidate score {:.3} beat {} score {:.3}",
            candidate_selection, baseline_kind, baseline_fitness
        );
        println!(
            "active checkpoint: {}; config: {} regime=model_infer",
            args.checkpoint, args.models_config
        );
        NeatPromotionReport {
            enabled: true,
            promoted: true,
            reason: format!(
                "eligible candidate score {:.3} beat {} score {:.3}",
                candidate_selection, baseline_kind, baseline_fitness
            ),
            baseline_kind,
            baseline_checkpoint,
            baseline_fitness,
            candidate_checkpoint: args.checkpoint.clone(),
            candidate_artifact: None,
            promoted_regime: Some(BehaviorRegime::ModelInfer),
            models_config: args.models_config.clone(),
        }
    } else {
        checkpoint.save(&rejected_candidate_path)?;
        let reason = if eligible && beats_baseline {
            match shadow_promotion.as_ref() {
                None => "held-out simulation and physical shadow evidence is required before promotion"
                    .to_string(),
                Some(decision) => format!(
                    "shadow promotion gate rejected candidate: {}",
                    decision.reasons.join("; ")
                ),
            }
        } else {
            transfer_rejection_reason(
                candidate_evaluation,
                criteria,
                beats_baseline,
                &baseline_kind,
            )
        };
        println!("\nnot promoting locomotion.neat.v0: {reason}");
        println!(
            "candidate artifact kept for inspection: {}",
            rejected_candidate_path.display()
        );
        NeatPromotionReport {
            enabled: true,
            promoted: false,
            reason,
            baseline_kind,
            baseline_checkpoint,
            baseline_fitness,
            candidate_checkpoint: args.checkpoint.clone(),
            candidate_artifact: Some(rejected_candidate_path.to_string_lossy().to_string()),
            promoted_regime: None,
            models_config: args.models_config.clone(),
        }
    };

    let report = NeatTrainingReport {
        behavior: args.behavior,
        seed: args.seed,
        heldout_seed_root: args.heldout_seed,
        validation_seed_root: args.validation_seed,
        checkpoint: args.checkpoint,
        novelty_weight: args.novelty_weight,
        novelty_neighbors: args.novelty_neighbors,
        novelty_archive_limit: args.novelty_archive_limit,
        world_archive_limit: args.world_archive_limit,
        world_replay_ratio: args.world_replay_ratio,
        world_mutation_ratio: args.world_mutation_ratio,
        stages: stage_reports,
        repertoire,
        world_archive,
        validations,
        trainer_state: Some(args.state_checkpoint.clone()),
        transfer_candidate: candidate_evaluation,
        transfer_eligible: eligible,
        transfer_criteria: criteria,
        hardcoded_transfer_fitness: hardcoded_selection,
        candidate_transfer_fitness: candidate_selection,
        noisy_transfer_fitness: noisy_selection,
        motor_mismatch_transfer_fitness: mismatch_selection,
        promotion,
    };
    let report_path = Path::new(&args.report_dir).join("training-report.json");
    write_json_report(&report_path, &report)?;
    println!("training report: {}", report_path.display());
    Ok(())
}

fn active_locomotion_model_checkpoint(config_path: &Path) -> Result<Option<String>> {
    let config = load_models_config(config_path)?;
    let Some(entry) = config.behavior.get("locomotion") else {
        return Ok(None);
    };
    if matches!(
        entry.regime,
        BehaviorRegime::ModelInfer | BehaviorRegime::ModelTrainAndInfer
    ) {
        Ok(entry.checkpoint.clone())
    } else {
        Ok(None)
    }
}

fn promote_locomotion_model_config(config_path: &Path, checkpoint: &Path) -> Result<()> {
    let mut config = load_models_config(config_path)?;
    let entry = config
        .behavior
        .entry("locomotion".to_string())
        .or_insert_with(|| BehaviorConfig {
            regime: BehaviorRegime::Hardcoded,
            hardcoded: "locomotion.hardcoded_wander.v0".to_string(),
            model: Some("locomotion.neat.v0".to_string()),
            checkpoint: None,
            fallback: FallbackPolicy::UseHardcoded,
        });
    if entry.hardcoded.is_empty() {
        entry.hardcoded = "locomotion.hardcoded_wander.v0".to_string();
    }
    if entry.model.is_none() {
        entry.model = Some("locomotion.neat.v0".to_string());
    }
    entry.regime = BehaviorRegime::ModelInfer;
    entry.checkpoint = Some(checkpoint.to_string_lossy().to_string());
    entry.fallback = FallbackPolicy::UseHardcoded;
    write_models_config(config_path, &config)
}

fn tune_compatibility_threshold(
    population: &mut Population,
    species_count: usize,
    target_min: usize,
    target_max: usize,
    threshold_floor: f32,
) {
    if target_min == 0 || target_max < target_min {
        return;
    }
    let before = population.config.compatibility_threshold;
    if species_count < target_min {
        population.config.compatibility_threshold =
            (population.config.compatibility_threshold * 0.85).max(threshold_floor.max(0.001));
        population.config.interspecies_mating_rate =
            population.config.interspecies_mating_rate.max(0.15);
    } else if species_count > target_max {
        population.config.compatibility_threshold =
            (population.config.compatibility_threshold * 1.08).min(8.0);
    }
    if (population.config.compatibility_threshold - before).abs() > f32::EPSILON {
        println!(
            "  compatibility threshold adjusted {:.3} -> {:.3} for species target {}..{}",
            before, population.config.compatibility_threshold, target_min, target_max
        );
    }
}

fn protected_generation_elites(
    genomes: &[Genome],
    evaluations: &[NeatPolicyEvaluation],
    generation_archive: &[QualityDiversityEntry],
) -> Vec<Genome> {
    let mut indices = Vec::<usize>::new();
    let feasible = evaluations
        .iter()
        .enumerate()
        .filter(|(_, evaluation)| {
            evaluation
                .selection_summary
                .is_some_and(|summary| summary.constraint_violations == 0)
        })
        .collect::<Vec<_>>();
    let mut add = |index: usize| {
        if index < genomes.len() && !indices.contains(&index) {
            indices.push(index);
        }
    };
    if let Some((index, _)) = feasible.iter().copied().max_by(|left, right| {
        left.1
            .selection_summary
            .map(|summary| summary.stage_success_rate)
            .unwrap_or_default()
            .total_cmp(
                &right
                    .1
                    .selection_summary
                    .map(|summary| summary.stage_success_rate)
                    .unwrap_or_default(),
            )
    }) {
        add(index);
    }
    if let Some((index, _)) = feasible.iter().copied().max_by(|left, right| {
        left.1
            .selection_summary
            .map(|summary| summary.stage_score)
            .unwrap_or(f32::NEG_INFINITY)
            .total_cmp(
                &right
                    .1
                    .selection_summary
                    .map(|summary| summary.stage_score)
                    .unwrap_or(f32::NEG_INFINITY),
            )
    }) {
        add(index);
    }
    if let Some((index, _)) = feasible.iter().copied().max_by(|left, right| {
        left.1
            .selection_summary
            .map(|summary| summary.prerequisite_floor)
            .unwrap_or_default()
            .total_cmp(
                &right
                    .1
                    .selection_summary
                    .map(|summary| summary.prerequisite_floor)
                    .unwrap_or_default(),
            )
    }) {
        add(index);
    }
    if let Some((index, _)) = feasible
        .iter()
        .copied()
        .filter(|(_, evaluation)| {
            evaluation
                .selection_summary
                .is_some_and(|summary| summary.stage_success_rate > 0.0)
        })
        .min_by_key(|(_, evaluation)| evaluation.metrics.short_cycle_steps)
    {
        add(index);
    }
    let mut niches = HashSet::new();
    for entry in generation_archive {
        if evaluations
            .get(entry.genome_index)
            .is_some_and(|evaluation| {
                evaluation
                    .selection_summary
                    .is_some_and(|summary| summary.constraint_violations == 0)
            })
            && niches.insert(entry.niche)
        {
            add(entry.genome_index);
        }
    }
    indices.truncate(8.min(genomes.len()));
    indices
        .into_iter()
        .map(|index| genomes[index].clone())
        .collect()
}

fn archive_recovery_founders(
    stage_archive: &[NeatNicheArchiveEntry],
    repertoire: &[NeatNicheArchiveEntry],
) -> Vec<Genome> {
    let mut entries = stage_archive
        .iter()
        .chain(repertoire.iter())
        .filter(|entry| archived_evaluation_is_safe(entry))
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| {
        right
            .evaluation
            .success_rate()
            .total_cmp(&left.evaluation.success_rate())
            .then_with(|| right.evaluation.fitness.total_cmp(&left.evaluation.fitness))
    });
    let mut niches = HashSet::new();
    let mut founders = Vec::new();
    for entry in entries {
        if niches.insert((entry.stage, entry.niche))
            && !founders.iter().any(|genome| genome == &entry.genome)
        {
            founders.push(entry.genome.clone());
        }
        if founders.len() >= 8 {
            break;
        }
    }
    founders
}

fn archived_evaluation_is_safe(entry: &NeatNicheArchiveEntry) -> bool {
    entry.evaluation.metrics.safety_invariant_violations == 0
        && entry.evaluation.traits.safety_veto_rate <= entry.stage.maximum_safety_veto_rate()
        && entry.evaluation.collision_rate
            <= entry.stage.promotion_criteria().maximum_collision_rate
}

#[derive(Clone, Copy, Debug)]
struct TrapEscapeTarget {
    mouth: (f32, f32),
    initial_distance_m: f32,
    boundary_distance_m: f32,
}

fn trap_escape_target(
    kind: ScenarioKind,
    metadata: &pete_sim::ScenarioMetadata,
) -> Option<TrapEscapeTarget> {
    let start = (metadata.body.odometry.x_m, metadata.body.odometry.y_m);
    let mouth = match kind {
        ScenarioKind::CornerTrap => (1.35, 1.35),
        ScenarioKind::ColumnTrap => (start.0 + 0.85, start.1),
        ScenarioKind::ConcaveTrap => {
            let obstacle_center = obstacle_centroid(&metadata.objects)?;
            let dx = start.0 - obstacle_center.0;
            let dy = start.1 - obstacle_center.1;
            let length = dx.hypot(dy).max(0.001);
            (start.0 + dx / length * 0.85, start.1 + dy / length * 0.85)
        }
        _ => return None,
    };
    let initial_distance_m = distance_between(start, mouth);
    Some(TrapEscapeTarget {
        mouth,
        initial_distance_m,
        boundary_distance_m: initial_distance_m + 0.75,
    })
}

fn obstacle_centroid(objects: &[pete_sim::SimObject]) -> Option<(f32, f32)> {
    let mut count = 0usize;
    let mut x = 0.0f32;
    let mut y = 0.0f32;
    for object in objects
        .iter()
        .filter(|object| matches!(object.kind, SimObjectKind::Obstacle))
    {
        count += 1;
        x += object.x_m;
        y += object.y_m;
    }
    (count > 0).then_some((x / count as f32, y / count as f32))
}
