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
