fn apply_deadband(value: f32, deadband: f32) -> f32 {
    if value.abs() < deadband {
        0.0
    } else {
        value
    }
}

async fn evaluate_neat_locomotion(
    genome: Option<&Genome>,
    stage: CurriculumStage,
    episodes: usize,
    steps: usize,
    seed: u64,
    perturbation: NeatPerturbation,
    challenges: Option<&[NeatEnvironmentChallenge]>,
    capture: bool,
) -> Result<NeatPolicyEvaluation> {
    let mut total = NeatEpisodeMetrics::default();
    let mut fitness = 0.0;
    let mut successful_episodes = 0usize;
    let mut worst_environment_score = f32::INFINITY;
    let mut environment_scores = Vec::with_capacity(episodes);
    let mut captured_snapshots = Vec::new();
    let mut captured_scenario = None;
    let mut stage_competence = BTreeMap::<CurriculumStage, StageCompetence>::new();

    for episode in 0..episodes {
        let challenge = challenges
            .and_then(|challenges| challenges.get(episode))
            .copied();
        let episode_stage = challenge.map(|challenge| challenge.stage).unwrap_or(stage);
        let episode_weights = episode_stage.weights();
        let episode_seed = challenge
            .map(|challenge| challenge.seed)
            .unwrap_or_else(|| seed.saturating_add(episode as u64));
        let kind = challenge
            .map(|challenge| challenge.kind)
            .unwrap_or_else(|| neat_stage_scenario(episode_stage, episode_seed));
        let mut scenario_config = ScenarioConfig::new(kind, episode_seed);
        if let Some((width_m, height_m)) =
            challenge.and_then(|challenge| challenge.arena_override_m)
        {
            scenario_config.arena = pete_sim::ArenaConfig {
                width_m: width_m.max(2.0),
                height_m: height_m.max(2.0),
            };
        }
        if challenge.is_some_and(|challenge| challenge.disable_chargers) {
            scenario_config.object_count = scenario_config
                .object_count
                .saturating_sub(scenario_config.charger_count);
            scenario_config.charger_count = 0;
        }
        let mut scenario = build_scenario(scenario_config);
        if let Some(initial_battery) = challenge.and_then(|challenge| challenge.initial_battery) {
            scenario.metadata.body.battery_level = initial_battery.clamp(0.0, 1.0);
            scenario.world.set_body(scenario.metadata.body.clone());
        }
        if capture && episode == 0 {
            captured_scenario = Some(scenario.metadata.clone());
        }
        let mut world = scenario.world;
        let mut motors = scenario.motors;
        let mut tracker = LocomotionTracker::default();
        let mut safety = SimpleSafety::default();
        let mut rng = StdRng::seed_from_u64(episode_seed ^ 0x5E1150);
        let mut metrics = NeatEpisodeMetrics::default();
        let mut cells: HashMap<(i32, i32), u32> = HashMap::new();
        let mut recent_cells = VecDeque::<(i32, i32)>::new();
        let mut visited_sectors = BTreeSet::<u8>::new();
        let mut reached_radius_bands = BTreeSet::<u32>::new();
        let mut start_position: Option<(f32, f32)> = None;
        let mut last_position: Option<(f32, f32)> = None;
        let mut collision_active = false;
        let mut escape_anchor: Option<(f32, f32)> = None;
        let mut delayed_commands = VecDeque::new();
        let mut genome_state = GenomeState::default();
        let mut resource_battery = scenario.metadata.body.battery_level.clamp(0.10, 1.0);
        let mut resource_health = 1.0f32;
        metrics.minimum_resource_battery = resource_battery;
        metrics.final_resource_battery = resource_battery;
        metrics.minimum_resource_health = resource_health;
        metrics.final_resource_health = resource_health;
        let topology_cost_per_tick = genome.map(neat_topology_metabolic_cost).unwrap_or(0.0);
        let escape_target = trap_escape_target(kind, &scenario.metadata);
        let mut best_escape_progress = 0.0f32;
        let mut crossed_escape_boundary = false;
        let episode_left_gain = if perturbation.wheel_gain_jitter > 0.0 {
            rng.gen_range(
                1.0 - perturbation.wheel_gain_jitter..1.0 + perturbation.wheel_gain_jitter,
            )
        } else {
            1.0
        };
        let episode_right_gain = if perturbation.wheel_gain_jitter > 0.0 {
            rng.gen_range(
                1.0 - perturbation.wheel_gain_jitter..1.0 + perturbation.wheel_gain_jitter,
            )
        } else {
            1.0
        };
        let episode_deadband = if perturbation.deadband_m_s > 0.0 {
            rng.gen_range(0.0..perturbation.deadband_m_s)
        } else {
            0.0
        };

        for _ in 0..steps {
            let snapshot = world.snapshot().await?;
            if capture && episode == 0 {
                captured_snapshots.push(snapshot.clone());
            }
            let body = &snapshot.body;
            let position = (body.odometry.x_m, body.odometry.y_m);
            let start = *start_position.get_or_insert(position);
            let displacement = distance_between(start, position);
            metrics.maximum_displacement_m = metrics.maximum_displacement_m.max(displacement);
            if displacement >= 0.5 {
                reached_radius_bands.insert((displacement / 0.5).floor() as u32);
                metrics.radius_bands_reached = reached_radius_bands.len() as u32;
            }
            if displacement >= 0.25 {
                let angle = (position.1 - start.1).atan2(position.0 - start.0);
                let sector = (((angle + std::f32::consts::PI) / (std::f32::consts::PI / 2.0))
                    .floor() as i32)
                    .rem_euclid(4) as u8;
                visited_sectors.insert(sector);
                metrics.arena_sectors_visited = visited_sectors.len() as u32;
            }
            let step_distance = last_position
                .map(|last| distance_between(last, position))
                .unwrap_or(0.0);
            let collision = body.flags.bump_left || body.flags.bump_right || body.flags.wall;
            let impact_energy = if collision && !collision_active {
                0.03
            } else {
                0.0
            };
            if collision && !collision_active {
                metrics.collisions = metrics.collisions.saturating_add(1);
                metrics.collision_energy_cost += impact_energy;
                resource_health = (resource_health - 0.15).max(0.0);
                escape_anchor = Some(position);
            }
            if !collision {
                metrics.distance_without_collision_m += step_distance;
            }
            if let Some(anchor) = escape_anchor {
                if !collision && distance_between(anchor, position) >= 0.35 {
                    metrics.successful_escapes = metrics.successful_escapes.saturating_add(1);
                    escape_anchor = None;
                }
            }
            if let Some(target) = escape_target {
                let distance_from_mouth = distance_between(position, target.mouth);
                best_escape_progress = best_escape_progress.max(distance_from_mouth);
                metrics.trap_mouth_progress_m = metrics
                    .trap_mouth_progress_m
                    .max((best_escape_progress - target.initial_distance_m).max(0.0));
                if !crossed_escape_boundary
                    && !collision
                    && distance_from_mouth >= target.boundary_distance_m
                {
                    crossed_escape_boundary = true;
                    metrics.escape_boundary_crossings =
                        metrics.escape_boundary_crossings.saturating_add(1);
                }
            }
            collision_active = collision;

            let cell = (
                (position.0 / 0.5).floor() as i32,
                (position.1 / 0.5).floor() as i32,
            );
            let visits = cells.entry(cell).or_insert(0);
            if *visits == 0 {
                metrics.new_area_cells = metrics.new_area_cells.saturating_add(1);
            } else if *visits >= 5 {
                metrics.repeated_state_steps = metrics.repeated_state_steps.saturating_add(1);
            }
            *visits = visits.saturating_add(1);
            observe_short_cycles(&mut recent_cells, cell, &mut metrics);

            let input = tracker.observe(body.last_update_ms, body, &snapshot.range);
            let mut output = if let Some(genome) = genome {
                let mut features = input.features();
                if perturbation.sensor_noise > 0.0 {
                    for feature in &mut features {
                        *feature = (*feature
                            + rng.gen_range(-perturbation.sensor_noise..perturbation.sensor_noise))
                        .clamp(-1.0, 1.0);
                    }
                }
                let raw = genome.activate_stateful(&features, &mut genome_state)?;
                LocomotionOutput {
                    forward_velocity_m_s: raw[0] * 0.6,
                    angular_velocity_rad_s: raw[1],
                    recovery_activation: (raw[2] + 1.0) * 0.5,
                }
                .bounded(0.6, 1.0)
                .with_recovery_intent(0.5)
            } else {
                LocomotionOutput {
                    forward_velocity_m_s: 0.2,
                    angular_velocity_rad_s: 0.1,
                    recovery_activation: 0.0,
                }
            };

            let left_scale = if perturbation.left_motor_scale == 0.0 {
                1.0
            } else {
                perturbation.left_motor_scale
            } * episode_left_gain;
            let right_scale = if perturbation.right_motor_scale == 0.0 {
                1.0
            } else {
                perturbation.right_motor_scale
            } * episode_right_gain;
            let left = output.forward_velocity_m_s - output.angular_velocity_rad_s * 0.235 * 0.5;
            let right = output.forward_velocity_m_s + output.angular_velocity_rad_s * 0.235 * 0.5;
            let mismatched_left = apply_deadband(left * left_scale, episode_deadband);
            let mismatched_right = apply_deadband(right * right_scale, episode_deadband);
            output.forward_velocity_m_s = (mismatched_left + mismatched_right) * 0.5;
            output.angular_velocity_rad_s = (mismatched_right - mismatched_left) / 0.235;
            output = output.bounded(0.6, 1.0);

            delayed_commands.push_back(output);
            let applied = if delayed_commands.len() > perturbation.latency_steps {
                delayed_commands.pop_front().unwrap_or_default()
            } else {
                LocomotionOutput::default()
            };
            metrics.recovery_activation_sum += applied.recovery_activation;
            let now = snapshot.to_now(body.last_update_ms);
            let decision = safety.filter(
                &now,
                pete_cockpit::MotorCommand {
                    forward: applied.forward_velocity_m_s,
                    turn: applied.angular_velocity_rad_s,
                },
            );
            if decision.vetoed {
                metrics.safety_vetoes = metrics.safety_vetoes.saturating_add(1);
            }
            if neat_safety_invariant_violated(&decision) {
                metrics.safety_invariant_violations =
                    metrics.safety_invariant_violations.saturating_add(1);
            }
            tracker.observe_command(LocomotionOutput {
                forward_velocity_m_s: decision.command.forward,
                angular_velocity_rad_s: decision.command.turn,
                recovery_activation: applied.recovery_activation,
            });
            metrics.wheel_motion_m += (mismatched_left.abs() + mismatched_right.abs()) * 0.5 * 0.1;
            metrics.angular_motion_rad += decision.command.turn.abs() * 0.1;
            if decision.command.forward.abs() > 0.02 && step_distance < 0.002 {
                metrics.stalled_steps = metrics.stalled_steps.saturating_add(1);
            }
            let motor_energy = (mismatched_left.abs() + mismatched_right.abs()) * 0.5 * 0.003;
            let sensor_energy = neat_sensor_energy_cost(&snapshot.range);
            metrics.sensor_energy_cost += sensor_energy;
            metrics.computation_energy_cost += topology_cost_per_tick;
            let tick_energy = motor_energy + sensor_energy + topology_cost_per_tick + impact_energy;
            metrics.resource_energy_used += tick_energy;
            resource_battery = (resource_battery - tick_energy).max(0.0);
            if body.charging {
                resource_battery = (resource_battery + 0.02).min(1.0);
            }
            metrics.minimum_resource_battery =
                metrics.minimum_resource_battery.min(resource_battery);
            metrics.minimum_resource_health = metrics.minimum_resource_health.min(resource_health);
            metrics.final_resource_battery = resource_battery;
            metrics.final_resource_health = resource_health;
            if resource_battery <= f32::EPSILON {
                metrics.battery_depleted = metrics.battery_depleted.saturating_add(1);
                break;
            }
            if resource_health <= f32::EPSILON {
                metrics.health_depleted = metrics.health_depleted.saturating_add(1);
                break;
            }
            if let Some(genome) = genome {
                let plasticity_reward =
                    neat_plasticity_reward(step_distance, collision, decision.vetoed);
                genome.apply_plasticity(&mut genome_state, plasticity_reward);
            }
            motors.apply_motion(MotionCommand::Drive {
                forward_m_s: decision.command.forward,
                turn_rad_s: decision.command.turn,
            })?;
            last_position = Some(position);
        }

        let functional = metrics.battery_depleted == 0
            && metrics.health_depleted == 0
            && metrics.final_resource_battery > f32::EPSILON
            && metrics.final_resource_health > f32::EPSILON;
        let succeeded = neat_episode_succeeded(
            episode_stage,
            metrics,
            steps,
            escape_target.is_some(),
            functional,
        );
        let episode_score = constrained_episode_score(metrics, succeeded, steps);
        environment_scores.push(episode_score);
        worst_environment_score = worst_environment_score.min(episode_score);
        successful_episodes += succeeded as usize;
        let weighted_score = stage_weighted_score(episode_stage, episode_weights, metrics, steps);
        fitness += weighted_score;
        let competence = stage_competence.entry(episode_stage).or_default();
        competence.episodes += 1;
        competence.successes += succeeded as usize;
        competence.weighted_score += weighted_score;
        competence.collision_rate += metrics.collisions as f32 / steps.max(1) as f32;
        competence.invariant_violations = competence
            .invariant_violations
            .saturating_add(metrics.safety_invariant_violations);
        add_neat_metrics(&mut total, metrics);
    }

    for evidence in stage_competence.values_mut() {
        let episodes = evidence.episodes.max(1) as f32;
        evidence.weighted_score /= episodes;
        evidence.collision_rate /= episodes;
    }

    let episode_count = episodes.max(1) as f32;
    let traits = FitnessTraits::from_metrics(
        total,
        episodes,
        steps,
        successful_episodes,
        worst_environment_score,
    );
    let lifetime_selection =
        lifetime_selection_summary(&environment_scores, successful_episodes, episodes);
    Ok(NeatPolicyEvaluation {
        fitness: fitness / episode_count,
        selection_fitness: 0.0,
        base_selection_fitness: 0.0,
        novelty_score: 0.0,
        lifetime_selection,
        traits,
        selection_summary: None,
        metrics: total,
        successful_episodes,
        episodes,
        collision_rate: total.collisions as f32 / (episodes.max(1) * steps.max(1)) as f32,
        environment_scores,
        stage_competence,
        snapshots: captured_snapshots,
        scenario: captured_scenario,
    })
}

fn neat_episode_succeeded(
    stage: CurriculumStage,
    metrics: NeatEpisodeMetrics,
    steps: usize,
    escape_required: bool,
    functional: bool,
) -> bool {
    functional
        && match stage {
            CurriculumStage::BackAwayReliably
            | CurriculumStage::ChooseUsefulTurn
            | CurriculumStage::EscapeCorners => {
                metrics.successful_escapes > 0 || metrics.escape_boundary_crossings > 0
            }
            CurriculumStage::LeaveStartRegion => {
                metrics.maximum_displacement_m >= 1.5
                    && metrics.new_area_cells >= 4
                    && metrics.collisions as usize <= (steps / 20).max(1)
            }
            CurriculumStage::ExpandLocalCoverage => {
                metrics.new_area_cells >= 6
                    && metrics.radius_bands_reached >= 2
                    && metrics.arena_sectors_visited >= 2
            }
            CurriculumStage::BreakShortCycles => {
                metrics.new_area_cells >= 8
                    && metrics.short_cycle_count <= 4
                    && metrics.recent_repetition_steps as usize * 100 < steps.max(1) * 35
                    && metrics.maximum_displacement_m >= 0.75
            }
            CurriculumStage::ExploreWithoutLooping => {
                metrics.new_area_cells >= 8
                    && metrics.short_cycle_count <= 4
                    && metrics.recent_repetition_steps as usize * 100 < steps.max(1) * 35
            }
            CurriculumStage::NavigateVariedRooms | CurriculumStage::TransferCandidatesToPete => {
                metrics.new_area_cells >= 10
                    && metrics.collisions <= 3
                    && (!escape_required || metrics.escape_boundary_crossings > 0)
            }
        }
}

fn stage_weighted_score(
    stage: CurriculumStage,
    weights: pete_neat::FitnessWeights,
    metrics: NeatEpisodeMetrics,
    steps: usize,
) -> f32 {
    let base = weights.score(metrics);
    match stage {
        CurriculumStage::LeaveStartRegion => {
            base + metrics.maximum_displacement_m * 20.0 + metrics.new_area_cells as f32 * 2.0
                - metrics.short_cycle_steps as f32 * 0.5
        }
        CurriculumStage::ExpandLocalCoverage => {
            base + metrics.radius_bands_reached as f32 * 12.0
                + metrics.arena_sectors_visited as f32 * 10.0
                + metrics.maximum_displacement_m * 8.0
                - (metrics.maximum_displacement_m < 0.5) as u8 as f32 * 30.0
        }
        CurriculumStage::BreakShortCycles => {
            let recent_rate = metrics.recent_repetition_steps as f32 / steps.max(1) as f32;
            base + metrics.new_area_cells as f32 * 3.0 + metrics.maximum_displacement_m * 5.0
                - metrics.short_cycle_count as f32 * 12.0
                - metrics.short_cycle_steps as f32 * 2.0
                - recent_rate * 100.0
        }
        CurriculumStage::ExploreWithoutLooping => {
            base - metrics.short_cycle_count as f32 * 6.0 - metrics.short_cycle_steps as f32
        }
        _ => base,
    }
}

fn lifetime_selection_summary(
    episode_scores: &[f32],
    successful_episodes: usize,
    episodes: usize,
) -> NeatLifetimeSelection {
    let mut scores = episode_scores
        .iter()
        .copied()
        .filter(|score| score.is_finite())
        .collect::<Vec<_>>();
    let qualification_probability = if episodes == 0 {
        0.0
    } else {
        successful_episodes as f32 / episodes as f32
    };
    if scores.is_empty() {
        return NeatLifetimeSelection {
            qualification_probability,
            ..NeatLifetimeSelection::default()
        };
    }
    scores.sort_by(|left, right| left.total_cmp(right));
    let mean_score = scores.iter().sum::<f32>() / scores.len() as f32;
    let lower_quartile_index = ((scores.len() - 1) as f32 * 0.25).floor() as usize;
    let lower_quartile_score = scores[lower_quartile_index];
    let worst_score = scores[0];
    let robustness_score = 0.5 * mean_score + 0.3 * lower_quartile_score + 0.2 * worst_score;
    NeatLifetimeSelection {
        mean_score,
        lower_quartile_score,
        worst_score,
        qualification_probability,
        robustness_score,
    }
}

fn lifetime_selection_adjustment(selection: NeatLifetimeSelection) -> f32 {
    selection.robustness_score + selection.qualification_probability * 10.0
}

fn structured_selection_summaries(
    stage: CurriculumStage,
    evaluations: &[NeatPolicyEvaluation],
    pareto: &[SelectionSummary],
    novelty: &[f32],
    novelty_weight: f32,
) -> Vec<SelectionSummary> {
    let stage_index = CurriculumStage::ORDER
        .iter()
        .position(|candidate| *candidate == stage)
        .unwrap_or_default();
    evaluations
        .iter()
        .zip(pareto.iter().copied())
        .zip(novelty.iter().copied())
        .map(|((evaluation, mut summary), novelty)| {
            let current = evaluation
                .stage_competence
                .get(&stage)
                .copied()
                .unwrap_or_default();
            let prerequisite_floor = if stage_index == 0 {
                1.0
            } else {
                CurriculumStage::ORDER[..stage_index]
                    .iter()
                    .map(|prerequisite| {
                        evaluation
                            .stage_competence
                            .get(prerequisite)
                            .copied()
                            .unwrap_or_default()
                            .success_rate()
                    })
                    .fold(1.0f32, f32::min)
            };
            let stage_success_rate = current.success_rate();
            let stage_score = current.weighted_score;
            let fitness = if summary.constraint_violations > 0 {
                -1_000_000_000.0 - summary.constraint_violations as f32 * 1_000_000.0
            } else {
                // Each term is bounded below the smallest meaningful increment
                // of the preceding tier. Novelty can only settle a near tie.
                let success_tier = stage_success_rate * 1_000_000.0;
                let retention_tier = prerequisite_floor * 10_000.0;
                let stage_tier = (stage_score / 100.0).tanh() * 1_000.0;
                let crowding = if summary.crowding_distance.is_finite() {
                    summary.crowding_distance.clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let pareto_tier = 100.0 / (summary.pareto_front as f32 + 1.0) + crowding;
                let novelty_tiebreak = (novelty_weight.max(0.0) * novelty).clamp(0.0, 0.99);
                success_tier
                    + retention_tier
                    + stage_tier
                    + pareto_tier
                    + novelty_tiebreak
                    + lifetime_selection_adjustment(evaluation.lifetime_selection)
                        .clamp(-0.49, 0.49)
            };
            summary.fitness = fitness;
            summary.stage_success_rate = stage_success_rate;
            summary.prerequisite_floor = prerequisite_floor;
            summary.stage_score = stage_score;
            summary
        })
        .collect()
}

fn structured_evaluation_better(
    stage: CurriculumStage,
    candidate: &NeatPolicyEvaluation,
    baseline: &NeatPolicyEvaluation,
) -> bool {
    let candidate_summary = stable_selection_summary(stage, candidate);
    let baseline_summary = stable_selection_summary(stage, baseline);
    candidate_summary
        .constraint_violations
        .cmp(&baseline_summary.constraint_violations)
        .reverse()
        .then_with(|| {
            candidate_summary
                .stage_success_rate
                .total_cmp(&baseline_summary.stage_success_rate)
        })
        .then_with(|| {
            candidate_summary
                .prerequisite_floor
                .total_cmp(&baseline_summary.prerequisite_floor)
        })
        .then_with(|| {
            candidate_summary
                .stage_score
                .total_cmp(&baseline_summary.stage_score)
        })
        .then_with(|| baseline.collision_rate.total_cmp(&candidate.collision_rate))
        .then_with(|| {
            baseline
                .metrics
                .short_cycle_steps
                .cmp(&candidate.metrics.short_cycle_steps)
        })
        .is_gt()
}

fn structured_ordering_score(stage: CurriculumStage, evaluation: &NeatPolicyEvaluation) -> f32 {
    let summary = stable_selection_summary(stage, evaluation);
    summary.fitness
}

fn stable_selection_summary(
    stage: CurriculumStage,
    evaluation: &NeatPolicyEvaluation,
) -> SelectionSummary {
    let criteria = stage.promotion_criteria();
    let constraint_violations = evaluation.metrics.safety_invariant_violations
        + (evaluation.collision_rate > criteria.maximum_collision_rate) as u32
        + (evaluation.traits.safety_veto_rate > stage.maximum_safety_veto_rate()) as u32;
    structured_selection_summaries(
        stage,
        std::slice::from_ref(evaluation),
        &[SelectionSummary {
            fitness: 0.0,
            constraint_violations,
            stage_success_rate: 0.0,
            prerequisite_floor: 0.0,
            stage_score: 0.0,
            pareto_front: 0,
            crowding_distance: 0.0,
        }],
        &[0.0],
        0.0,
    )[0]
}

fn transfer_ordering_score(evaluation: &NeatPolicyEvaluation) -> f32 {
    if evaluation.metrics.safety_invariant_violations > 0 {
        return -10_000.0 - evaluation.metrics.safety_invariant_violations as f32;
    }
    evaluation.success_rate() * 1_000.0 - evaluation.collision_rate * 100.0
        + (evaluation.fitness / 1_000.0).tanh() * 10.0
}

fn transfer_evaluation_better(
    candidate: &NeatPolicyEvaluation,
    baseline: &NeatPolicyEvaluation,
) -> bool {
    candidate
        .metrics
        .safety_invariant_violations
        .cmp(&baseline.metrics.safety_invariant_violations)
        .reverse()
        .then_with(|| candidate.success_rate().total_cmp(&baseline.success_rate()))
        .then_with(|| baseline.collision_rate.total_cmp(&candidate.collision_rate))
        .then_with(|| candidate.fitness.total_cmp(&baseline.fitness))
        .is_gt()
}

fn transfer_perturbation_robust(
    nominal: &NeatPolicyEvaluation,
    perturbed: &NeatPolicyEvaluation,
) -> bool {
    let retained_success = if nominal.success_rate() <= f32::EPSILON {
        perturbed.success_rate() >= nominal.success_rate()
    } else {
        perturbed.success_rate() >= nominal.success_rate() * 0.75
    };
    perturbed.metrics.safety_invariant_violations == 0
        && retained_success
        && perturbed.collision_rate <= nominal.collision_rate + 0.02
        && perturbed.traits.safety_veto_rate <= 0.05
}

fn transfer_rejection_reason(
    evaluation: CandidateEvaluation,
    criteria: pete_neat::PromotionCriteria,
    beats_baseline: bool,
    baseline_kind: &str,
) -> String {
    let mut failures = Vec::new();
    if evaluation.seeded_episodes < criteria.minimum_seeded_episodes {
        failures.push(format!(
            "episodes {} < {}",
            evaluation.seeded_episodes, criteria.minimum_seeded_episodes
        ));
    }
    if evaluation.success_rate < criteria.minimum_success_rate {
        failures.push(format!(
            "success {:.1}% < {:.1}%",
            evaluation.success_rate * 100.0,
            criteria.minimum_success_rate * 100.0
        ));
    }
    if evaluation.collision_rate > criteria.maximum_collision_rate {
        failures.push(format!(
            "collision {:.4} > {:.4}",
            evaluation.collision_rate, criteria.maximum_collision_rate
        ));
    }
    if evaluation.safety_veto_rate > criteria.maximum_safety_veto_rate {
        failures.push(format!(
            "safety veto rate {:.1}% > {:.1}%",
            evaluation.safety_veto_rate * 100.0,
            criteria.maximum_safety_veto_rate * 100.0
        ));
    }
    if evaluation.safety_invariant_violations > criteria.maximum_safety_invariant_violations {
        failures.push(format!(
            "safety invariant violations {} > {}",
            evaluation.safety_invariant_violations, criteria.maximum_safety_invariant_violations
        ));
    }
    if !evaluation.beats_hardcoded {
        failures.push("did not beat hardcoded".to_string());
    }
    if !evaluation.noise_robust {
        failures.push("sensor perturbation retention failed".to_string());
    }
    if !evaluation.motor_mismatch_robust {
        failures.push("motor perturbation retention failed".to_string());
    }
    if !evaluation.fallback_verified {
        failures.push("fallback verification failed".to_string());
    }
    if !beats_baseline {
        failures.push(format!("did not beat {baseline_kind}"));
    }
    if failures.is_empty() {
        "candidate did not satisfy promotion policy".to_string()
    } else {
        failures.join("; ")
    }
}

async fn validate_neat_curriculum(
    genome: &Genome,
    current_stage_index: usize,
    evolving_stages: &[CurriculumStage],
    steps: usize,
    validation_seed: u64,
    validation_round: u64,
) -> Result<Vec<NeatStageValidationReport>> {
    let mut reports = Vec::with_capacity(current_stage_index + 1);
    for (stage_index, stage) in evolving_stages
        .iter()
        .copied()
        .enumerate()
        .take(current_stage_index + 1)
    {
        let criteria = stage.promotion_criteria();
        let episodes = criteria.minimum_seeded_episodes as usize;
        let seed = validation_seed
            .saturating_add(validation_round << 32)
            .saturating_add((stage_index as u64) << 20);
        let evaluation = evaluate_neat_locomotion(
            Some(genome),
            stage,
            episodes,
            steps,
            seed,
            NeatPerturbation::default(),
            None,
            false,
        )
        .await?;
        let passed = evaluation.success_rate() >= criteria.minimum_success_rate
            && evaluation.collision_rate <= criteria.maximum_collision_rate
            && evaluation.traits.safety_veto_rate <= stage.maximum_safety_veto_rate()
            && evaluation.metrics.safety_invariant_violations
                <= criteria.maximum_safety_invariant_violations;
        reports.push(NeatStageValidationReport {
            candidate_kind: String::new(),
            stage,
            validation_round,
            seeded_episodes: episodes,
            success_rate: evaluation.success_rate(),
            collision_rate: evaluation.collision_rate,
            safety_veto_rate: evaluation.traits.safety_veto_rate,
            safety_invariant_violations: evaluation.metrics.safety_invariant_violations,
            passed,
        });
    }
    Ok(reports)
}

fn validation_candidates(
    current_genome: &Genome,
    current_evaluation: &NeatPolicyEvaluation,
    stage_best: &Option<(f32, Genome, NeatPolicyEvaluation)>,
    stage_archive: &[NeatNicheArchiveEntry],
    repertoire: &[NeatNicheArchiveEntry],
) -> Vec<(String, Genome, NeatPolicyEvaluation)> {
    let mut candidates = vec![(
        "reproductive-champion".to_string(),
        current_genome.clone(),
        current_evaluation.clone(),
    )];
    if let Some((_, genome, evaluation)) = stage_best.as_ref().filter(|(_, _, evaluation)| {
        evaluation.metrics.safety_invariant_violations == 0
            && evaluation.traits.safety_veto_rate <= 0.05
    }) {
        candidates.push((
            "structured-stage-best".to_string(),
            genome.clone(),
            evaluation.clone(),
        ));
    }
    if let Some(entry) = stage_archive
        .iter()
        .chain(repertoire.iter())
        .filter(|entry| {
            entry.niche == pete_neat::NicheLabel::Generalist && archived_evaluation_is_safe(entry)
        })
        .max_by(|left, right| {
            left.evaluation
                .success_rate()
                .total_cmp(&right.evaluation.success_rate())
                .then_with(|| left.evaluation.fitness.total_cmp(&right.evaluation.fitness))
        })
    {
        candidates.push((
            "archived-generalist".to_string(),
            entry.genome.clone(),
            entry.evaluation.clone(),
        ));
    }
    if let Some(entry) = stage_archive
        .iter()
        .filter(|entry| archived_evaluation_is_safe(entry))
        .max_by(|left, right| {
            left.evaluation
                .success_rate()
                .total_cmp(&right.evaluation.success_rate())
                .then_with(|| left.evaluation.fitness.total_cmp(&right.evaluation.fitness))
        })
    {
        candidates.push((
            "current-stage-specialist".to_string(),
            entry.genome.clone(),
            entry.evaluation.clone(),
        ));
    }
    let mut unique = Vec::<(String, Genome, NeatPolicyEvaluation)>::new();
    for candidate in candidates {
        if !unique.iter().any(|(_, genome, _)| genome == &candidate.1) {
            unique.push(candidate);
        }
    }
    unique
}

async fn select_neat_transfer_candidate(
    stage_champion: &Genome,
    repertoire: &[NeatNicheArchiveEntry],
    steps: usize,
    seed: u64,
) -> Result<Genome> {
    let mut candidates = vec![stage_champion.clone()];
    let mut ranked = repertoire.iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.selection_fitness.total_cmp(&left.selection_fitness));
    for entry in ranked {
        if candidates.len() >= 8 {
            break;
        }
        if !candidates.contains(&entry.genome) {
            candidates.push(entry.genome.clone());
        }
    }
    let candidate_count = candidates.len();
    let mut best: Option<(Genome, NeatPolicyEvaluation)> = None;
    for (index, genome) in candidates.into_iter().enumerate() {
        let evaluation = evaluate_neat_locomotion(
            Some(&genome),
            CurriculumStage::TransferCandidatesToPete,
            64,
            steps,
            seed,
            NeatPerturbation::default(),
            None,
            false,
        )
        .await?;
        println!(
            "transfer shortlist {}/{} success={:.1}% collision={:.4} invariants={}",
            index + 1,
            candidate_count,
            evaluation.success_rate() * 100.0,
            evaluation.collision_rate,
            evaluation.metrics.safety_invariant_violations
        );
        if best
            .as_ref()
            .is_none_or(|(_, current)| transfer_evaluation_better(&evaluation, current))
        {
            best = Some((genome, evaluation));
        }
    }
    best.map(|(genome, _)| genome)
        .context("empty transfer candidate shortlist")
}

async fn capture_neat_champion(
    genome: &Genome,
    stage: CurriculumStage,
    steps: usize,
    seed: u64,
    path: &Path,
    fitness: f32,
) -> Result<()> {
    let evaluation = evaluate_neat_locomotion(
        Some(genome),
        stage,
        1,
        steps,
        seed,
        NeatPerturbation::default(),
        None,
        true,
    )
    .await?;
    let mut writer = CaptureWriter::create(path, CaptureSource::Sim, Some(100)).await?;
    writer.manifest_mut().scenario = evaluation.scenario;
    writer.manifest_mut().notes.push(format!(
        "NEAT locomotion champion stage={} selection={fitness:.4} nodes={} connections={}",
        neat_stage_slug(stage),
        genome.nodes.len(),
        genome.connections.len()
    ));
    writer.manifest_mut().command_args = std::env::args().collect();
    for snapshot in evaluation.snapshots {
        writer
            .append_snapshot(snapshot.body.last_update_ms, snapshot, Vec::new())
            .await?;
    }
    writer.finish().await?;
    Ok(())
}

fn neat_stage_scenario(stage: CurriculumStage, seed: u64) -> ScenarioKind {
    match stage {
        CurriculumStage::BackAwayReliably => ScenarioKind::ObstacleAvoidance,
        CurriculumStage::ChooseUsefulTurn => match seed % 3 {
            0 => ScenarioKind::ObstacleAvoidance,
            1 => ScenarioKind::ColumnTrap,
            _ => ScenarioKind::ConcaveTrap,
        },
        CurriculumStage::EscapeCorners => match seed % 3 {
            0 => ScenarioKind::CornerTrap,
            1 => ScenarioKind::ColumnTrap,
            _ => ScenarioKind::ConcaveTrap,
        },
        CurriculumStage::LeaveStartRegion => match seed % 2 {
            0 => ScenarioKind::EmptyRoom,
            _ => ScenarioKind::ObstacleAvoidance,
        },
        CurriculumStage::ExpandLocalCoverage => match seed % 3 {
            0 => ScenarioKind::EmptyRoom,
            1 => ScenarioKind::ObstacleAvoidance,
            _ => ScenarioKind::MixedRoom,
        },
        CurriculumStage::BreakShortCycles => match seed % 3 {
            0 => ScenarioKind::EmptyRoom,
            1 => ScenarioKind::ObstacleAvoidance,
            _ => ScenarioKind::MixedRoom,
        },
        CurriculumStage::ExploreWithoutLooping => {
            if seed % 4 == 0 {
                ScenarioKind::ConcaveTrap
            } else {
                ScenarioKind::MixedRoom
            }
        }
        CurriculumStage::NavigateVariedRooms | CurriculumStage::TransferCandidatesToPete => {
            match seed % 6 {
                0 => ScenarioKind::EmptyRoom,
                1 => ScenarioKind::ObstacleAvoidance,
                2 => ScenarioKind::CornerTrap,
                3 => ScenarioKind::ColumnTrap,
                4 => ScenarioKind::ConcaveTrap,
                _ => ScenarioKind::Dream,
            }
        }
    }
}

fn neat_stage_slug(stage: CurriculumStage) -> &'static str {
    match stage {
        CurriculumStage::BackAwayReliably => "back-away-reliably",
        CurriculumStage::ChooseUsefulTurn => "choose-a-useful-turn",
        CurriculumStage::EscapeCorners => "escape-corners",
        CurriculumStage::LeaveStartRegion => "leave-start-region",
        CurriculumStage::ExpandLocalCoverage => "expand-local-coverage",
        CurriculumStage::BreakShortCycles => "break-short-cycles",
        CurriculumStage::ExploreWithoutLooping => "explore-without-looping",
        CurriculumStage::NavigateVariedRooms => "navigate-varied-rooms",
        CurriculumStage::TransferCandidatesToPete => "transfer-candidates-to-pete",
    }
}

fn constrained_episode_score(metrics: NeatEpisodeMetrics, succeeded: bool, steps: usize) -> f32 {
    let step_count = steps.max(1) as f32;
    let collision_rate = metrics.collisions as f32 / step_count;
    let repetition_rate = metrics.repeated_state_steps as f32 / step_count;
    let efficiency = (metrics.new_area_cells as f32 + metrics.distance_without_collision_m)
        / (1.0 + metrics.wheel_motion_m);
    (succeeded as u8 as f32) * 20.0
        + metrics.escape_boundary_crossings as f32 * 8.0
        + metrics.successful_escapes as f32 * 6.0
        + efficiency
        + metrics.trap_mouth_progress_m
        - collision_rate * 20.0
        - repetition_rate * 5.0
        - metrics.safety_vetoes as f32 * 100.0
        - metrics.resource_energy_used * 4.0
        - metrics.battery_depleted as f32 * 50.0
        - metrics.health_depleted as f32 * 75.0
}

const RECENT_CELL_WINDOW: usize = 24;

fn observe_short_cycles(
    recent: &mut VecDeque<(i32, i32)>,
    cell: (i32, i32),
    metrics: &mut NeatEpisodeMetrics,
) {
    if recent.iter().any(|candidate| *candidate == cell) {
        metrics.recent_repetition_steps = metrics.recent_repetition_steps.saturating_add(1);
    }
    let cycle_length = (2..=4).find(|length| {
        recent
            .len()
            .checked_sub(*length)
            .and_then(|index| recent.get(index))
            .is_some_and(|candidate| *candidate == cell)
    });
    if let Some(length) = cycle_length {
        metrics.short_cycle_count = metrics.short_cycle_count.saturating_add(1);
        metrics.short_cycle_steps = metrics.short_cycle_steps.saturating_add(length as u32);
    }
    recent.push_back(cell);
    if recent.len() > RECENT_CELL_WINDOW {
        recent.pop_front();
    }
}

fn neat_plasticity_reward(step_distance: f32, collision: bool, safety_vetoed: bool) -> f32 {
    let mut reward = (step_distance * 5.0).clamp(0.0, 0.5);
    if collision {
        reward -= 0.5;
    }
    if safety_vetoed {
        reward -= 0.5;
    }
    reward.clamp(-1.0, 1.0)
}

fn neat_safety_invariant_violated(decision: &pete_autonomic::SafetyDecision) -> bool {
    let command = decision.command;
    let finite = command.forward.is_finite() && command.turn.is_finite();
    let bounded =
        command.forward.abs() <= 0.6 + f32::EPSILON && command.turn.abs() <= 1.0 + f32::EPSILON;
    let stopped_when_vetoed = !decision.vetoed
        || (command.forward.abs() <= f32::EPSILON && command.turn.abs() <= f32::EPSILON);
    !finite || !bounded || !stopped_when_vetoed
}

fn neat_sensor_energy_cost(range: &RangeSense) -> f32 {
    0.00025 + range.beams.len() as f32 * 0.000002
}

fn neat_topology_metabolic_cost(genome: &Genome) -> f32 {
    (genome.nodes.len() as f32 * 0.000015 + genome.connections.len() as f32 * 0.000008).min(0.01)
}

fn neat_generation_challenges(
    stage: CurriculumStage,
    episodes: usize,
    seed: u64,
    archive: &[NeatWorldArchiveEntry],
    replay_ratio: f32,
    mutation_ratio: f32,
    rehearsal_ratio: f32,
) -> Vec<NeatEnvironmentChallenge> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0xA11CE_5757);
    let current_stage_index = CurriculumStage::ORDER
        .iter()
        .position(|candidate| *candidate == stage)
        .unwrap_or_default();
    let prior_stages = &CurriculumStage::ORDER[..current_stage_index];
    let retention_enabled = rehearsal_ratio > 0.0 && !prior_stages.is_empty();
    // A small configured batch is expanded only as far as necessary to give
    // every retained prerequisite one episode while keeping >= 50% fresh
    // current-stage signal.
    let total_episodes = if retention_enabled {
        episodes.max(prior_stages.len().saturating_mul(2))
    } else {
        episodes
    };
    let fresh_count = (total_episodes + 1) / 2;
    let rehearsal_count = if retention_enabled {
        prior_stages.len()
    } else {
        0
    };
    let auxiliary_budget = total_episodes.saturating_sub(fresh_count + rehearsal_count);
    let wants_replay = replay_ratio > 0.0;
    let wants_mutation = mutation_ratio > 0.0;
    let replay_count = auxiliary_budget.min(wants_replay as usize);
    let mutation_count = auxiliary_budget
        .saturating_sub(replay_count)
        .min(wants_mutation as usize);
    let extra_fresh = auxiliary_budget.saturating_sub(replay_count + mutation_count);
    let mut ranked_archive = archive
        .iter()
        .filter(|entry| entry.challenge.stage == stage)
        .collect::<Vec<_>>();
    ranked_archive.sort_by(|left, right| {
        right
            .distinction
            .total_cmp(&left.distinction)
            .then_with(|| right.difficulty.total_cmp(&left.difficulty))
    });

    let mut challenges = Vec::with_capacity(total_episodes);
    for current_index in 0..fresh_count + extra_fresh {
        let episode_seed = seed
            .saturating_add((current_index as u64) << 8)
            .saturating_add(rng.gen_range(0..=u16::MAX) as u64);
        challenges.push(NeatEnvironmentChallenge {
            stage,
            kind: neat_stage_scenario(stage, episode_seed),
            seed: episode_seed,
            arena_override_m: None,
            initial_battery: None,
            disable_chargers: false,
        });
    }
    for entry in ranked_archive.iter().take(replay_count) {
        challenges.push(entry.challenge);
    }
    for entry in ranked_archive.iter().cycle().take(mutation_count) {
        challenges.push(mutate_neat_environment(entry.challenge, &mut rng));
    }
    // If no current-stage archive exists yet, auxiliary slots become fresh
    // current-stage worlds rather than leaking prior-stage archive material.
    while challenges.len() < fresh_count + extra_fresh + replay_count + mutation_count {
        let episode_seed = seed
            .saturating_add((challenges.len() as u64) << 8)
            .saturating_add(rng.gen_range(0..=u16::MAX) as u64);
        challenges.push(NeatEnvironmentChallenge {
            stage,
            kind: neat_stage_scenario(stage, episode_seed),
            seed: episode_seed,
            arena_override_m: None,
            initial_battery: None,
            disable_chargers: false,
        });
    }
    for rehearsal_index in 0..rehearsal_count {
        let rehearsal_stage = prior_stages[rehearsal_index];
        let archived = archive
            .iter()
            .filter(|entry| entry.challenge.stage == rehearsal_stage)
            .max_by(|left, right| {
                left.difficulty
                    .total_cmp(&right.difficulty)
                    .then_with(|| left.distinction.total_cmp(&right.distinction))
            });
        if let Some(entry) = archived {
            challenges.push(entry.challenge);
            continue;
        }
        let episode_seed = seed
            .saturating_add((challenges.len() as u64) << 8)
            .saturating_add(rng.gen_range(0..=u16::MAX) as u64);
        challenges.push(NeatEnvironmentChallenge {
            stage: rehearsal_stage,
            kind: neat_stage_scenario(rehearsal_stage, episode_seed),
            seed: episode_seed,
            arena_override_m: None,
            initial_battery: None,
            disable_chargers: false,
        });
    }
    while challenges.len() < total_episodes {
        let episode_seed = seed
            .saturating_add((challenges.len() as u64) << 8)
            .saturating_add(rng.gen_range(0..=u16::MAX) as u64);
        challenges.push(NeatEnvironmentChallenge {
            stage,
            kind: neat_stage_scenario(stage, episode_seed),
            seed: episode_seed,
            arena_override_m: None,
            initial_battery: None,
            disable_chargers: false,
        });
    }
    challenges
}

fn mutate_neat_environment<R: Rng + ?Sized>(
    challenge: NeatEnvironmentChallenge,
    rng: &mut R,
) -> NeatEnvironmentChallenge {
    let kinds = match challenge.stage {
        CurriculumStage::BackAwayReliably => &[ScenarioKind::ObstacleAvoidance][..],
        CurriculumStage::ChooseUsefulTurn => &[
            ScenarioKind::ObstacleAvoidance,
            ScenarioKind::ColumnTrap,
            ScenarioKind::ConcaveTrap,
        ],
        CurriculumStage::EscapeCorners => &[
            ScenarioKind::CornerTrap,
            ScenarioKind::ColumnTrap,
            ScenarioKind::ConcaveTrap,
        ],
        CurriculumStage::LeaveStartRegion => {
            &[ScenarioKind::EmptyRoom, ScenarioKind::ObstacleAvoidance]
        }
        CurriculumStage::ExpandLocalCoverage | CurriculumStage::BreakShortCycles => &[
            ScenarioKind::EmptyRoom,
            ScenarioKind::ObstacleAvoidance,
            ScenarioKind::MixedRoom,
        ],
        CurriculumStage::ExploreWithoutLooping => {
            &[ScenarioKind::MixedRoom, ScenarioKind::ConcaveTrap]
        }
        CurriculumStage::NavigateVariedRooms | CurriculumStage::TransferCandidatesToPete => &[
            ScenarioKind::EmptyRoom,
            ScenarioKind::ObstacleAvoidance,
            ScenarioKind::CornerTrap,
            ScenarioKind::ColumnTrap,
            ScenarioKind::ConcaveTrap,
            ScenarioKind::Dream,
        ],
    };
    NeatEnvironmentChallenge {
        stage: challenge.stage,
        kind: kinds.choose(rng).copied().unwrap_or(challenge.kind),
        seed: challenge.seed.wrapping_add(rng.gen_range(1..=10_000)) ^ rng.gen::<u64>(),
        arena_override_m: challenge.arena_override_m,
        initial_battery: challenge.initial_battery,
        disable_chargers: challenge.disable_chargers,
    }
}

fn update_neat_world_archive(
    archive: &mut Vec<NeatWorldArchiveEntry>,
    challenges: &[NeatEnvironmentChallenge],
    evaluations: &[NeatPolicyEvaluation],
    generation: u64,
    archive_limit: usize,
) -> usize {
    let mut retained = 0usize;
    for (episode_index, challenge) in challenges.iter().copied().enumerate() {
        let scores = evaluations
            .iter()
            .filter_map(|evaluation| evaluation.environment_scores.get(episode_index).copied())
            .filter(|score| score.is_finite())
            .collect::<Vec<_>>();
        if scores.len() < 2 {
            continue;
        }
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores
            .iter()
            .map(|score| {
                let delta = score - mean;
                delta * delta
            })
            .sum::<f32>()
            / scores.len() as f32;
        let distinction = variance.sqrt();
        let difficulty = (1.0 - ((mean + 20.0) / 80.0)).clamp(0.0, 1.0);
        let discriminates = distinction >= 0.25 && (0.10..=0.90).contains(&difficulty);
        if !discriminates {
            continue;
        }
        let candidate = NeatWorldArchiveEntry {
            challenge,
            difficulty,
            distinction,
            retained_generation: generation,
        };
        if let Some(existing) = archive
            .iter_mut()
            .find(|entry| entry.challenge == candidate.challenge)
        {
            if candidate.distinction > existing.distinction {
                *existing = candidate;
                retained = retained.saturating_add(1);
            }
        } else {
            archive.push(candidate);
            retained = retained.saturating_add(1);
        }
    }
    archive.sort_by(|left, right| {
        let left_score = left.distinction * (1.0 - (left.difficulty - 0.5).abs());
        let right_score = right.distinction * (1.0 - (right.difficulty - 0.5).abs());
        right_score
            .total_cmp(&left_score)
            .then_with(|| right.retained_generation.cmp(&left.retained_generation))
    });
    if archive.len() > archive_limit {
        archive.truncate(archive_limit);
    }
    retained
}

fn update_neat_niche_archive(
    archive: &mut Vec<NeatNicheArchiveEntry>,
    stage: CurriculumStage,
    generation_archive: &[QualityDiversityEntry],
    genomes: &[Genome],
    evaluations: &[NeatPolicyEvaluation],
) {
    for entry in generation_archive {
        let Some(genome) = genomes.get(entry.genome_index) else {
            continue;
        };
        let Some(evaluation) = evaluations.get(entry.genome_index) else {
            continue;
        };
        let candidate = NeatNicheArchiveEntry {
            stage,
            niche: entry.niche,
            descriptor: entry.descriptor,
            selection_fitness: entry.selection_fitness,
            base_selection_fitness: evaluation.base_selection_fitness,
            novelty_score: evaluation.novelty_score,
            diagnostic_fitness: evaluation.fitness,
            qualification_evidence: None,
            genome: genome.clone(),
            evaluation: evaluation.clone(),
        };
        upsert_niche_archive_entry(archive, candidate);
    }
}

async fn qualify_neat_stage_archive(
    archive: &mut [NeatNicheArchiveEntry],
    episodes: usize,
    steps: usize,
    seed: u64,
) -> Result<()> {
    if archive.is_empty() || episodes == 0 {
        return Ok(());
    }
    let mut indices = (0..archive.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        archive[*right]
            .selection_fitness
            .total_cmp(&archive[*left].selection_fitness)
    });
    indices.truncate(16);
    let trap_challenges = niche_challenges(
        CurriculumStage::EscapeCorners,
        ScenarioKind::ConcaveTrap,
        episodes,
        seed ^ 0x7A_A9,
        None,
        None,
        false,
    );
    let low_battery_challenges = niche_challenges(
        CurriculumStage::NavigateVariedRooms,
        ScenarioKind::MixedRoom,
        episodes,
        seed ^ 0xBA77,
        None,
        Some(0.18),
        true,
    );
    let corridor_challenges = niche_challenges(
        CurriculumStage::NavigateVariedRooms,
        ScenarioKind::EmptyRoom,
        episodes,
        seed ^ 0x0C07_71D0,
        Some((12.0, 3.5)),
        None,
        false,
    );

    for index in indices {
        let genome = archive[index].genome.clone();
        let nominal = evaluate_neat_locomotion(
            Some(&genome),
            CurriculumStage::TransferCandidatesToPete,
            episodes,
            steps,
            seed.saturating_add(index as u64 * 0x1000),
            NeatPerturbation::default(),
            None,
            false,
        )
        .await?;
        let degraded = evaluate_neat_locomotion(
            Some(&genome),
            CurriculumStage::TransferCandidatesToPete,
            episodes,
            steps,
            seed.saturating_add(index as u64 * 0x1000),
            NeatPerturbation {
                sensor_noise: 0.08,
                latency_steps: 1,
                ..NeatPerturbation::default()
            },
            None,
            false,
        )
        .await?;
        let mismatch = evaluate_neat_locomotion(
            Some(&genome),
            CurriculumStage::TransferCandidatesToPete,
            episodes,
            steps,
            seed.saturating_add(index as u64 * 0x1000),
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
        let traps = evaluate_neat_locomotion(
            Some(&genome),
            CurriculumStage::EscapeCorners,
            episodes,
            steps,
            seed ^ 0x7A_A9,
            NeatPerturbation::default(),
            Some(&trap_challenges),
            false,
        )
        .await?;
        let low_battery = evaluate_neat_locomotion(
            Some(&genome),
            CurriculumStage::NavigateVariedRooms,
            episodes,
            steps,
            seed ^ 0xBA77,
            NeatPerturbation::default(),
            Some(&low_battery_challenges),
            false,
        )
        .await?;
        let corridor = evaluate_neat_locomotion(
            Some(&genome),
            CurriculumStage::NavigateVariedRooms,
            episodes,
            steps,
            seed ^ 0x0C07_71D0,
            NeatPerturbation::default(),
            Some(&corridor_challenges),
            false,
        )
        .await?;
        let low_battery_progress = if low_battery.metrics.battery_depleted == 0 {
            low_battery.metrics.distance_without_collision_m / episodes as f32
        } else {
            0.0
        };
        let evidence = NicheQualificationEvidence {
            degraded_sensor_retention: success_retention(&nominal, &degraded),
            motor_mismatch_retention: success_retention(&nominal, &mismatch),
            heldout_trap_success_rate: traps.success_rate(),
            low_battery_progress_m: low_battery_progress,
            corridor_success_rate: corridor.success_rate(),
        };
        archive[index].niche = archive[index]
            .descriptor
            .evidence_based_niche_label(archive[index].evaluation.traits, evidence);
        archive[index].qualification_evidence = Some(evidence);
    }
    Ok(())
}

fn success_retention(nominal: &NeatPolicyEvaluation, perturbed: &NeatPolicyEvaluation) -> f32 {
    if nominal.success_rate() <= f32::EPSILON {
        0.0
    } else {
        (perturbed.success_rate() / nominal.success_rate()).clamp(0.0, 1.0)
    }
}

fn niche_challenges(
    stage: CurriculumStage,
    kind: ScenarioKind,
    episodes: usize,
    seed: u64,
    arena_override_m: Option<(f32, f32)>,
    initial_battery: Option<f32>,
    disable_chargers: bool,
) -> Vec<NeatEnvironmentChallenge> {
    (0..episodes)
        .map(|episode| NeatEnvironmentChallenge {
            stage,
            kind,
            seed: seed.saturating_add(episode as u64),
            arena_override_m,
            initial_battery,
            disable_chargers,
        })
        .collect()
}

fn update_global_repertoire(
    repertoire: &mut Vec<NeatNicheArchiveEntry>,
    stage_archive: &[NeatNicheArchiveEntry],
) {
    for entry in stage_archive {
        upsert_niche_archive_entry(repertoire, entry.clone());
    }
}

fn upsert_niche_archive_entry(
    archive: &mut Vec<NeatNicheArchiveEntry>,
    candidate: NeatNicheArchiveEntry,
) {
    if let Some(existing) = archive
        .iter_mut()
        .find(|entry| entry.descriptor == candidate.descriptor)
    {
        if candidate.selection_fitness > existing.selection_fitness {
            *existing = candidate;
        }
    } else {
        archive.push(candidate);
    }
}

fn write_stage_niche_checkpoints(
    archive: &[NeatNicheArchiveEntry],
    generation: u64,
    report_dir: &Path,
) -> Result<()> {
    let archive_dir = report_dir.join("niche-archive");
    fs::create_dir_all(&archive_dir)?;
    for entry in archive {
        let file_name = format!(
            "{}-{}-c{}-t{}-a{}-e{}-r{}.json",
            neat_stage_slug(entry.stage),
            niche_slug(entry.niche),
            entry.descriptor.collision_frequency_bin,
            entry.descriptor.turning_intensity_bin,
            entry.descriptor.area_coverage_bin,
            entry.descriptor.energy_consumption_bin,
            entry.descriptor.recovery_aggressiveness_bin
        );
        let checkpoint =
            LocomotionCheckpoint::new(generation, entry.selection_fitness, entry.genome.clone());
        checkpoint.save(archive_dir.join(file_name))?;
    }
    Ok(())
}

fn niche_slug(niche: pete_neat::NicheLabel) -> &'static str {
    match niche {
        pete_neat::NicheLabel::OpenRoomExplorer => "open-room-explorer",
        pete_neat::NicheLabel::NarrowCorridorNavigator => "narrow-corridor-navigator",
        pete_neat::NicheLabel::ConcaveTrapEscapeSpecialist => "concave-trap-escape-specialist",
        pete_neat::NicheLabel::ClutterSpecialist => "clutter-specialist",
        pete_neat::NicheLabel::LowBatteryConservativeMover => "low-battery-conservative-mover",
        pete_neat::NicheLabel::DegradedSensorNavigator => "degraded-sensor-navigator",
        pete_neat::NicheLabel::AsymmetricMotorCompensator => "asymmetric-motor-compensator",
        pete_neat::NicheLabel::Generalist => "generalist",
    }
}

fn add_neat_metrics(total: &mut NeatEpisodeMetrics, episode: NeatEpisodeMetrics) {
    total.new_area_cells = total.new_area_cells.saturating_add(episode.new_area_cells);
    total.distance_without_collision_m += episode.distance_without_collision_m;
    total.successful_escapes = total
        .successful_escapes
        .saturating_add(episode.successful_escapes);
    total.escape_boundary_crossings = total
        .escape_boundary_crossings
        .saturating_add(episode.escape_boundary_crossings);
    total.trap_mouth_progress_m += episode.trap_mouth_progress_m;
    total.collisions = total.collisions.saturating_add(episode.collisions);
    total.repeated_state_steps = total
        .repeated_state_steps
        .saturating_add(episode.repeated_state_steps);
    total.short_cycle_count = total
        .short_cycle_count
        .saturating_add(episode.short_cycle_count);
    total.short_cycle_steps = total
        .short_cycle_steps
        .saturating_add(episode.short_cycle_steps);
    total.recent_repetition_steps = total
        .recent_repetition_steps
        .saturating_add(episode.recent_repetition_steps);
    total.maximum_displacement_m += episode.maximum_displacement_m;
    total.radius_bands_reached = total
        .radius_bands_reached
        .saturating_add(episode.radius_bands_reached);
    total.arena_sectors_visited = total
        .arena_sectors_visited
        .saturating_add(episode.arena_sectors_visited);
    total.wheel_motion_m += episode.wheel_motion_m;
    total.angular_motion_rad += episode.angular_motion_rad;
    total.recovery_activation_sum += episode.recovery_activation_sum;
    total.stalled_steps = total.stalled_steps.saturating_add(episode.stalled_steps);
    total.safety_vetoes = total.safety_vetoes.saturating_add(episode.safety_vetoes);
    total.safety_invariant_violations = total
        .safety_invariant_violations
        .saturating_add(episode.safety_invariant_violations);
    total.resource_energy_used += episode.resource_energy_used;
    total.sensor_energy_cost += episode.sensor_energy_cost;
    total.computation_energy_cost += episode.computation_energy_cost;
    total.collision_energy_cost += episode.collision_energy_cost;
    total.minimum_resource_battery = if total.minimum_resource_battery == 0.0 {
        episode.minimum_resource_battery
    } else {
        total
            .minimum_resource_battery
            .min(episode.minimum_resource_battery)
    };
    total.final_resource_battery += episode.final_resource_battery;
    total.minimum_resource_health = if total.minimum_resource_health == 0.0 {
        episode.minimum_resource_health
    } else {
        total
            .minimum_resource_health
            .min(episode.minimum_resource_health)
    };
    total.final_resource_health += episode.final_resource_health;
    total.battery_depleted = total
        .battery_depleted
        .saturating_add(episode.battery_depleted);
    total.health_depleted = total
        .health_depleted
        .saturating_add(episode.health_depleted);
}

fn write_json_report(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("writing {}", path.display()))
}

fn curriculum_split(
    episode_index: usize,
    episode_count: usize,
    validation_ratio: f32,
    test_ratio: f32,
) -> &'static str {
    let validation_count = ((episode_count as f32) * validation_ratio).round() as usize;
    let test_count = ((episode_count as f32) * test_ratio).round() as usize;
    let train_count = episode_count.saturating_sub(validation_count + test_count);
    if episode_index < train_count {
        "train"
    } else if episode_index < train_count + validation_count {
        "validation"
    } else {
        "test"
    }
}

fn scenario_object_summary(objects: &[pete_sim::SimObject]) -> serde_json::Value {
    let mut chargers = 0usize;
    let mut obstacles = 0usize;
    let mut people = 0usize;
    let mut speakers = 0usize;
    let mut landmarks = 0usize;

    for object in objects {
        match &object.kind {
            pete_sim::SimObjectKind::Charger => chargers += 1,
            pete_sim::SimObjectKind::Obstacle => obstacles += 1,
            pete_sim::SimObjectKind::Person { .. } => people += 1,
            pete_sim::SimObjectKind::SoundSource { .. } => speakers += 1,
            pete_sim::SimObjectKind::Landmark { .. } => landmarks += 1,
        }
    }

    serde_json::json!({
        "chargers": chargers,
        "obstacles": obstacles,
        "people": people,
        "speakers": speakers,
        "landmarks": landmarks,
    })
}

