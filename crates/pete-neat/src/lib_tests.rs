use super::*;
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn hardcoded_policy_preserves_ancestral_wander_during_contact() {
    let mut behavior = HardcodedLocomotionBehavior::default();
    let left = behavior
        .infer(&LocomotionInput {
            bump_left: 1.0,
            ..LocomotionInput::default()
        })
        .unwrap();
    assert_eq!(left.forward_velocity_m_s, 0.2);
    assert_eq!(left.angular_velocity_rad_s, 0.1);
    assert_eq!(left.recovery_activation, 0.0);
}

#[test]
fn tracker_resets_collision_distance_and_derives_wheels() {
    let mut tracker = LocomotionTracker::default();
    let range = RangeSense::default();
    let mut body = BodySense::default();
    tracker.observe(100, &body, &range);
    body.odometry.x_m = 1.0;
    let moving = tracker.observe(1_100, &body, &range);
    assert!((moving.distance_since_collision_m - 1.0).abs() < 0.001);
    body.flags.bump_left = true;
    let collided = tracker.observe(1_200, &body, &range);
    assert_eq!(collided.distance_since_collision_m, 0.0);
    assert!(collided.left_wheel_travel_m > 0.9);
}

#[test]
fn minimal_genome_activates_and_population_evolves() {
    let mut rng = StdRng::seed_from_u64(7);
    let config = NeatConfig {
        population_size: 12,
        ..NeatConfig::default()
    };
    let mut population = Population::seeded(config, &mut rng);
    let output = population.genomes[0]
        .activate(&[0.0; LOCOMOTION_INPUT_COUNT])
        .unwrap();
    assert_eq!(output.len(), LOCOMOTION_OUTPUT_COUNT);
    let before = population.genomes.len();
    population
        .evolve(
            &(0..before).map(|index| index as f32).collect::<Vec<_>>(),
            &mut rng,
        )
        .unwrap();
    assert_eq!(population.generation, 1);
    assert_eq!(population.genomes.len(), before);
}

#[test]
fn recurrent_connections_use_previous_activation_state() {
    let mut rng = StdRng::seed_from_u64(11);
    let mut innovations =
        InnovationTracker::new((LOCOMOTION_INPUT_COUNT + LOCOMOTION_OUTPUT_COUNT + 1) as u32);
    let mut genome = Genome::minimal(
        LOCOMOTION_INPUT_COUNT,
        LOCOMOTION_OUTPUT_COUNT,
        &mut innovations,
        &mut rng,
    );
    let output_id = LOCOMOTION_INPUT_COUNT as u32 + 1;
    for edge in &mut genome.connections {
        edge.enabled = false;
    }
    genome.connections.push(ConnectionGene {
        innovation: innovations.connection(output_id, output_id, true),
        from: output_id,
        to: output_id,
        weight: 1.0,
        enabled: true,
        recurrent: true,
        plasticity: PlasticityMode::Fixed,
        plasticity_rate: 0.0,
    });
    genome.connections.push(ConnectionGene {
        innovation: innovations.connection(0, output_id, false),
        from: 0,
        to: output_id,
        weight: 1.0,
        enabled: true,
        recurrent: false,
        plasticity: PlasticityMode::Fixed,
        plasticity_rate: 0.0,
    });
    let mut state = GenomeState::default();
    let mut inputs = [0.0; LOCOMOTION_INPUT_COUNT];
    inputs[0] = 1.0;
    let first = genome.activate_stateful(&inputs, &mut state).unwrap()[0];
    inputs[0] = 0.0;
    let second = genome.activate_stateful(&inputs, &mut state).unwrap()[0];
    let stateless_second = genome.activate(&inputs).unwrap()[0];

    assert!(first > 0.7);
    assert!(second > stateless_second);
    assert!(second > 0.5);
}

#[test]
fn plasticity_changes_effective_weights_only_within_lifetime() {
    let mut rng = StdRng::seed_from_u64(13);
    let mut innovations =
        InnovationTracker::new((LOCOMOTION_INPUT_COUNT + LOCOMOTION_OUTPUT_COUNT + 1) as u32);
    let mut genome = Genome::minimal(
        LOCOMOTION_INPUT_COUNT,
        LOCOMOTION_OUTPUT_COUNT,
        &mut innovations,
        &mut rng,
    );
    let output_id = LOCOMOTION_INPUT_COUNT as u32 + 1;
    let innovation = innovations.connection(0, output_id, false);
    genome.connections.clear();
    genome.connections.push(ConnectionGene {
        innovation,
        from: 0,
        to: output_id,
        weight: 0.5,
        enabled: true,
        recurrent: false,
        plasticity: PlasticityMode::Hebbian,
        plasticity_rate: 0.05,
    });
    let mut state = GenomeState::default();
    let mut inputs = [0.0; LOCOMOTION_INPUT_COUNT];
    inputs[0] = 1.0;

    let before = genome.activate_stateful(&inputs, &mut state).unwrap()[0];
    genome.apply_plasticity(&mut state, 0.0);
    let learned_weight = state.effective_weights[&innovation];
    state.reset();
    let after_reset = genome.activate_stateful(&inputs, &mut state).unwrap()[0];

    assert!(learned_weight > 0.5);
    assert_eq!(genome.connections[0].weight, 0.5);
    assert!((before - after_reset).abs() < 1.0e-6);
}

#[test]
fn reward_modulated_locomotion_learns_from_runtime_motion_reward() {
    let mut rng = StdRng::seed_from_u64(14);
    let mut innovations =
        InnovationTracker::new((LOCOMOTION_INPUT_COUNT + LOCOMOTION_OUTPUT_COUNT + 1) as u32);
    let mut genome = Genome::minimal(
        LOCOMOTION_INPUT_COUNT,
        LOCOMOTION_OUTPUT_COUNT,
        &mut innovations,
        &mut rng,
    );
    let output_id = LOCOMOTION_INPUT_COUNT as u32 + 1;
    let innovation = innovations.connection(0, output_id, false);
    genome.connections.clear();
    genome.connections.push(ConnectionGene {
        innovation,
        from: 0,
        to: output_id,
        weight: 0.5,
        enabled: true,
        recurrent: false,
        plasticity: PlasticityMode::RewardModulated,
        plasticity_rate: 0.05,
    });
    let mut behavior = NeatLocomotionBehavior {
        checkpoint: LocomotionCheckpoint::new(0, 0.0, genome),
        max_forward_m_s: 0.6,
        max_turn_rad_s: 1.0,
        state: GenomeState::default(),
        last_input: None,
    };
    let first = LocomotionInput {
        bump_left: 1.0,
        ..LocomotionInput::default()
    };
    behavior.infer(&first).unwrap();
    let mut second = first.clone();
    second.left_wheel_travel_m = 0.2;
    second.right_wheel_travel_m = 0.2;
    behavior.infer(&second).unwrap();

    assert!(behavior.state.effective_weights[&innovation] > 0.5);
}

#[test]
fn recurrent_and_feed_forward_connections_get_distinct_innovations() {
    let mut innovations = InnovationTracker::new(100);
    let feed_forward = innovations.connection(1, 2, false);
    let recurrent = innovations.connection(1, 2, true);

    assert_ne!(feed_forward, recurrent);
    assert_eq!(feed_forward, innovations.connection(1, 2, false));
    assert_eq!(recurrent, innovations.connection(1, 2, true));
}

#[test]
fn safety_vetoes_are_costs_not_invariant_violations() {
    let metrics = EpisodeMetrics {
        safety_vetoes: 40,
        safety_invariant_violations: 0,
        ..EpisodeMetrics::default()
    };
    let traits = FitnessTraits::from_metrics(metrics, 2, 100, 0, 0.0);
    let summary = selection_summaries(&[traits], SelectionConstraints::new(0, 0.05, 0.0))[0];

    assert_eq!(traits.safety_veto_rate, 0.2);
    assert_eq!(summary.constraint_violations, 0);
    assert!(summary.fitness > 0.0);
}

#[test]
fn founder_population_round_trips_complete_innovation_state() {
    let mut rng = StdRng::seed_from_u64(22);
    let mut innovations =
        InnovationTracker::new((LOCOMOTION_INPUT_COUNT + LOCOMOTION_OUTPUT_COUNT + 1) as u32);
    let first = Genome::minimal(
        LOCOMOTION_INPUT_COUNT,
        LOCOMOTION_OUTPUT_COUNT,
        &mut innovations,
        &mut rng,
    );
    let mut second = first.clone();
    second.mutate(NeatConfig::default(), &mut innovations, &mut rng);
    let config = NeatConfig {
        population_size: 8,
        ..NeatConfig::default()
    };
    let population =
        Population::from_founders(config, &[first.clone(), second.clone()], &mut rng).unwrap();

    assert_eq!(population.genomes.len(), 8);
    assert_eq!(population.genomes[0], first);
    assert_eq!(population.genomes[1], second);
    let encoded = serde_json::to_vec(&population).unwrap();
    let decoded: Population = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded.genomes, population.genomes);
    assert_eq!(
        decoded.innovations.connections,
        population.innovations.connections
    );
}

#[test]
fn serialized_population_resumes_deterministically() {
    let config = NeatConfig {
        population_size: 8,
        ..NeatConfig::default()
    };
    let mut seed_rng = StdRng::seed_from_u64(31);
    let population = Population::seeded(config, &mut seed_rng);
    let encoded = serde_json::to_vec(&population).unwrap();
    let mut uninterrupted = population;
    let mut resumed: Population = serde_json::from_slice(&encoded).unwrap();
    let fitness = (0..config.population_size)
        .map(|index| index as f32)
        .collect::<Vec<_>>();
    let mut left_rng = StdRng::seed_from_u64(32);
    let mut right_rng = StdRng::seed_from_u64(32);

    uninterrupted.evolve(&fitness, &mut left_rng).unwrap();
    resumed.evolve(&fitness, &mut right_rng).unwrap();

    assert_eq!(uninterrupted.generation, resumed.generation);
    assert_eq!(uninterrupted.genomes, resumed.genomes);
    assert_eq!(uninterrupted.genome_species, resumed.genome_species);
    assert_eq!(uninterrupted.species_records, resumed.species_records);
    assert_eq!(
        uninterrupted.innovations.connections,
        resumed.innovations.connections
    );
}

#[test]
fn founder_reconstruction_repairs_legacy_recurrent_innovation_collisions() {
    let mut rng = StdRng::seed_from_u64(40);
    let mut innovations =
        InnovationTracker::new((LOCOMOTION_INPUT_COUNT + LOCOMOTION_OUTPUT_COUNT + 1) as u32);
    let first = Genome::minimal(
        LOCOMOTION_INPUT_COUNT,
        LOCOMOTION_OUTPUT_COUNT,
        &mut innovations,
        &mut rng,
    );
    let mut second = first.clone();
    second.connections[0].recurrent = true;
    second.connections[0].innovation = first.connections[0].innovation;
    let population = Population::from_founders(
        NeatConfig {
            population_size: 2,
            ..NeatConfig::default()
        },
        &[first, second],
        &mut rng,
    )
    .unwrap();

    assert_ne!(
        population.genomes[0].connections[0].innovation,
        population.genomes[1].connections[0].innovation
    );
}

#[test]
fn evolution_preserves_a_champion_per_surviving_species() {
    let (mut population, mut rng) = two_species_population();
    let first_species_champion = population.genomes[1].clone();
    let second_species_champion = population.genomes[4].clone();

    population
        .evolve(&[1.0, 5.0, 2.0, 3.0, 4.0, 0.0], &mut rng)
        .unwrap();

    assert!(population
        .genomes
        .iter()
        .any(|genome| genome == &first_species_champion));
    assert!(population
        .genomes
        .iter()
        .any(|genome| genome == &second_species_champion));
    let species_ids = population
        .species()
        .into_iter()
        .map(|species| species.id)
        .collect::<BTreeSet<_>>();
    assert!(species_ids.contains(&10));
    assert!(species_ids.contains(&20));
}

#[test]
fn evolve_with_elites_preserves_a_low_fitness_protected_genome() {
    let mut rng = StdRng::seed_from_u64(1701);
    let mut population = Population::seeded(
        NeatConfig {
            population_size: 8,
            compatibility_threshold: 100.0,
            ..NeatConfig::default()
        },
        &mut rng,
    );
    let protected = population.genomes[7].clone();
    population
        .evolve_with_elites(
            &[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, -100.0],
            &[protected.clone()],
            &mut rng,
        )
        .unwrap();
    assert!(population.genomes.contains(&protected));
}

#[test]
fn archive_injection_recovers_multiple_species_from_a_collapsed_population() {
    let mut rng = StdRng::seed_from_u64(1702);
    let mut population = Population::seeded(
        NeatConfig {
            population_size: 12,
            compatibility_threshold: 0.000_001,
            add_connection_rate: 1.0,
            add_node_rate: 1.0,
            weight_mutation_rate: 1.0,
            ..NeatConfig::default()
        },
        &mut rng,
    );
    let collapsed = population.genomes[0].clone();
    population.genomes.fill(collapsed.clone());
    population.assign_species_members();
    assert_eq!(population.species().len(), 1);
    let injected = population.inject_archive_descendants(&[collapsed], 8, &mut rng);
    assert_eq!(injected, 8);
    assert!(population.species().len() > 1);
}

#[test]
fn stagnant_species_go_extinct_except_global_champion_species() {
    let (mut stagnant_loser, mut rng) = two_species_population();
    stagnant_loser.species_records[1].best_fitness = 100.0;
    stagnant_loser.species_records[1].generations_without_improvement =
        stagnant_loser.config.species_stagnation_generations - 1;
    stagnant_loser
        .evolve(&[1.0, 5.0, 2.0, 3.0, 4.0, 0.0], &mut rng)
        .unwrap();
    assert!(!stagnant_loser
        .species()
        .iter()
        .any(|species| species.id == 20));

    let (mut stagnant_global_champion, mut rng) = two_species_population();
    stagnant_global_champion.species_records[1].best_fitness = 100.0;
    stagnant_global_champion.species_records[1].generations_without_improvement =
        stagnant_global_champion
            .config
            .species_stagnation_generations
            - 1;
    stagnant_global_champion
        .evolve(&[1.0, 2.0, 0.0, 3.0, 6.0, 0.0], &mut rng)
        .unwrap();
    assert!(stagnant_global_champion
        .species()
        .iter()
        .any(|species| species.id == 20));
}

#[test]
fn checkpoint_round_trip_validates_feature_order() {
    let mut rng = StdRng::seed_from_u64(9);
    let mut innovations =
        InnovationTracker::new((LOCOMOTION_INPUT_COUNT + LOCOMOTION_OUTPUT_COUNT + 1) as u32);
    let genome = Genome::minimal(
        LOCOMOTION_INPUT_COUNT,
        LOCOMOTION_OUTPUT_COUNT,
        &mut innovations,
        &mut rng,
    );
    let checkpoint = LocomotionCheckpoint::new(3, 12.5, genome);
    let directory = tempfile::tempdir().unwrap();
    checkpoint.save(directory.path()).unwrap();
    assert_eq!(
        LocomotionCheckpoint::load(directory.path()).unwrap(),
        checkpoint
    );
}

#[test]
fn curriculum_scores_reward_escape_and_penalize_vetoes() {
    let weights = FitnessWeights::collision_recovery();
    let good = weights.score(EpisodeMetrics {
        successful_escapes: 2,
        ..EpisodeMetrics::default()
    });
    let unsafe_run = weights.score(EpisodeMetrics {
        safety_vetoes: 2,
        ..EpisodeMetrics::default()
    });
    assert!(good > unsafe_run);
}

#[test]
fn curriculum_scores_make_resource_budgets_consequential() {
    let weights = FitnessWeights::efficient_wandering();
    let competent = weights.score(EpisodeMetrics {
        new_area_cells: 10,
        distance_without_collision_m: 4.0,
        resource_energy_used: 0.5,
        final_resource_battery: 0.5,
        final_resource_health: 1.0,
        ..EpisodeMetrics::default()
    });
    let depleted = weights.score(EpisodeMetrics {
        new_area_cells: 10,
        distance_without_collision_m: 4.0,
        resource_energy_used: 12.0,
        battery_depleted: 1,
        health_depleted: 1,
        ..EpisodeMetrics::default()
    });
    assert!(competent > depleted);
}

#[test]
fn selection_constraints_prevent_compensation_for_fatal_weaknesses() {
    let reckless = FitnessTraits {
        exploration: 100.0,
        escape_rate: 1.0,
        collision_rate: 0.50,
        energy_use: 1.0,
        forward_progress: 50.0,
        repetition_rate: 0.0,
        worst_environment_score: 10.0,
        safety_veto_rate: 0.0,
        safety_invariant_violations: 0,
    };
    let competent = FitnessTraits {
        exploration: 10.0,
        escape_rate: 0.9,
        collision_rate: 0.01,
        energy_use: 2.0,
        forward_progress: 5.0,
        repetition_rate: 0.1,
        worst_environment_score: 5.0,
        safety_veto_rate: 0.0,
        safety_invariant_violations: 0,
    };
    let scores = rank_fitness(
        &[reckless, competent],
        SelectionConstraints::new(0, 0.05, 0.8),
    );
    assert!(scores[1] > scores[0]);
    assert!(scores[0] < 0.0);
}

#[test]
fn excessive_safety_veto_reliance_is_reproductively_infeasible() {
    let high_success_veto_dependent = FitnessTraits {
        exploration: 20.0,
        escape_rate: 1.0,
        collision_rate: 0.0,
        energy_use: 2.0,
        forward_progress: 8.0,
        repetition_rate: 0.0,
        worst_environment_score: 10.0,
        safety_veto_rate: 0.23,
        safety_invariant_violations: 0,
    };
    let clean_partial_success = FitnessTraits {
        escape_rate: 0.75,
        safety_veto_rate: 0.0,
        ..high_success_veto_dependent
    };
    let scores = rank_fitness(
        &[high_success_veto_dependent, clean_partial_success],
        SelectionConstraints::new(0, 0.10, 0.0).with_maximum_safety_veto_rate(0.05),
    );
    assert!(scores[0] < 0.0);
    assert!(scores[1] > scores[0]);
}

#[test]
fn pareto_ranking_preserves_distinct_feasible_strategies() {
    let explorer = FitnessTraits {
        exploration: 40.0,
        escape_rate: 0.9,
        collision_rate: 0.02,
        energy_use: 8.0,
        forward_progress: 15.0,
        repetition_rate: 0.2,
        worst_environment_score: 7.0,
        safety_veto_rate: 0.0,
        safety_invariant_violations: 0,
    };
    let efficient = FitnessTraits {
        exploration: 15.0,
        escape_rate: 0.9,
        collision_rate: 0.01,
        energy_use: 1.0,
        forward_progress: 8.0,
        repetition_rate: 0.02,
        worst_environment_score: 7.0,
        safety_veto_rate: 0.0,
        safety_invariant_violations: 0,
    };
    let dominated = FitnessTraits {
        exploration: 10.0,
        escape_rate: 0.9,
        collision_rate: 0.03,
        energy_use: 10.0,
        forward_progress: 4.0,
        repetition_rate: 0.3,
        worst_environment_score: 2.0,
        safety_veto_rate: 0.0,
        safety_invariant_violations: 0,
    };
    let summaries = selection_summaries(
        &[explorer, efficient, dominated],
        SelectionConstraints::new(0, 0.05, 0.8),
    );
    assert_eq!(summaries[0].pareto_front, 0);
    assert_eq!(summaries[1].pareto_front, 0);
    assert!(summaries[2].pareto_front > 0);
}

#[test]
fn quality_diversity_archive_retains_distinct_behavior_cells() {
    let traits = [
        FitnessTraits {
            exploration: 40.0,
            escape_rate: 0.9,
            collision_rate: 0.002,
            energy_use: 3.0,
            forward_progress: 15.0,
            repetition_rate: 0.02,
            worst_environment_score: 10.0,
            safety_veto_rate: 0.0,
            safety_invariant_violations: 0,
        },
        FitnessTraits {
            exploration: 8.0,
            escape_rate: 0.8,
            collision_rate: 0.03,
            energy_use: 2.0,
            forward_progress: 3.0,
            repetition_rate: 0.1,
            worst_environment_score: 5.0,
            safety_veto_rate: 0.0,
            safety_invariant_violations: 0,
        },
        FitnessTraits {
            exploration: 40.0,
            escape_rate: 0.9,
            collision_rate: 0.002,
            energy_use: 3.0,
            forward_progress: 15.0,
            repetition_rate: 0.02,
            worst_environment_score: 10.0,
            safety_veto_rate: 0.0,
            safety_invariant_violations: 0,
        },
    ];
    let metrics = [
        EpisodeMetrics {
            new_area_cells: 80,
            angular_motion_rad: 6.0,
            wheel_motion_m: 6.0,
            recovery_activation_sum: 1.0,
            ..EpisodeMetrics::default()
        },
        EpisodeMetrics {
            new_area_cells: 16,
            angular_motion_rad: 60.0,
            wheel_motion_m: 4.0,
            recovery_activation_sum: 80.0,
            ..EpisodeMetrics::default()
        },
        EpisodeMetrics {
            new_area_cells: 80,
            angular_motion_rad: 6.0,
            wheel_motion_m: 6.0,
            recovery_activation_sum: 1.0,
            ..EpisodeMetrics::default()
        },
    ];
    let archive = quality_diversity_archive(&traits, &metrics, &[10.0, 20.0, 30.0], 2, 100);
    assert_eq!(archive.len(), 2);
    assert!(archive.iter().any(|entry| entry.genome_index == 1));
    assert!(archive.iter().any(|entry| entry.genome_index == 2));
}

#[test]
fn specialist_niches_require_explicit_qualification_evidence() {
    let traits = [FitnessTraits {
        exploration: 12.0,
        escape_rate: 0.9,
        collision_rate: 0.002,
        energy_use: 4.0,
        forward_progress: 8.0,
        repetition_rate: 0.02,
        worst_environment_score: 10.0,
        safety_veto_rate: 0.0,
        safety_invariant_violations: 0,
    }];
    let metrics = [EpisodeMetrics {
        new_area_cells: 24,
        angular_motion_rad: 30.0,
        wheel_motion_m: 8.0,
        recovery_activation_sum: 1.0,
        ..EpisodeMetrics::default()
    }];
    let descriptor =
        QualityDiversityDescriptor::from_traits_and_metrics(traits[0], metrics[0], 2, 100);

    assert_eq!(descriptor.niche_label(traits[0]), NicheLabel::Generalist);

    let archive = quality_diversity_archive_with_evidence(
        &traits,
        &metrics,
        &[10.0],
        &[NicheQualificationEvidence::from_selection_retention(
            0.0, 0.9,
        )],
        2,
        100,
    );
    assert_eq!(archive[0].niche, NicheLabel::AsymmetricMotorCompensator);
}

#[test]
fn pareto_boundary_bonus_is_bounded() {
    assert_eq!(normalized_crowding_bonus(f32::INFINITY), 1.0);
    assert_eq!(normalized_crowding_bonus(5.0), 1.0);
    assert_eq!(normalized_crowding_bonus(0.25), 0.25);
}

#[test]
fn novelty_pressure_rewards_behavioral_distance() {
    let familiar = BehavioralDescriptor {
        coverage: 0.2,
        collision_rate: 0.1,
        mean_curvature: 0.1,
        escape_style: 0.1,
        energy: 0.2,
    };
    let strange = BehavioralDescriptor {
        coverage: 0.9,
        collision_rate: 0.1,
        mean_curvature: 0.8,
        escape_style: 0.7,
        energy: 0.3,
    };
    let near_familiar = BehavioralDescriptor {
        coverage: 0.22,
        collision_rate: 0.1,
        mean_curvature: 0.12,
        escape_style: 0.1,
        energy: 0.2,
    };
    let novelty = novelty_scores(&[near_familiar, strange], &[familiar, near_familiar], 2);
    assert!(novelty[1] > novelty[0]);

    let pressured = apply_novelty_pressure(&[10.0, 10.0], &novelty, 5.0);
    assert!(pressured[1].selection_fitness > pressured[0].selection_fitness);
}

#[test]
fn curriculum_has_the_required_order_and_transfer_is_a_gate() {
    assert_eq!(
        CurriculumStage::BackAwayReliably.next(),
        Some(CurriculumStage::ChooseUsefulTurn)
    );
    assert_eq!(
        CurriculumStage::NavigateVariedRooms.next(),
        Some(CurriculumStage::TransferCandidatesToPete)
    );
    assert!(!CurriculumStage::TransferCandidatesToPete.evolves_population());
    assert_eq!(CurriculumStage::TransferCandidatesToPete.next(), None);
}

#[test]
fn transfer_gate_requires_robustness_fallback_and_zero_violations() {
    let criteria = CurriculumStage::TransferCandidatesToPete.promotion_criteria();
    let passing = CandidateEvaluation {
        seeded_episodes: 500,
        success_rate: 0.95,
        collision_rate: 0.02,
        safety_veto_rate: 0.01,
        safety_invariant_violations: 0,
        beats_hardcoded: true,
        noise_robust: true,
        motor_mismatch_robust: true,
        fallback_verified: true,
    };
    assert!(criteria.accepts(passing));
    assert!(!criteria.accepts(CandidateEvaluation {
        safety_invariant_violations: 1,
        ..passing
    }));
    assert!(!criteria.accepts(CandidateEvaluation {
        safety_veto_rate: 0.25,
        ..passing
    }));
    assert!(!criteria.accepts(CandidateEvaluation {
        fallback_verified: false,
        ..passing
    }));
}

fn shadow_report(
    environment: ShadowEnvironment,
    candidate: LocomotionPolicyMetrics,
) -> LocomotionShadowReport {
    LocomotionShadowReport {
        schema_version: LOCOMOTION_SHADOW_SCHEMA_VERSION,
        environment,
        baseline_id: "locomotion.hardcoded_wander.v0".into(),
        candidate_id: "candidate-good".into(),
        capture_ids: vec!["capture-a".into()],
        episodes: 20,
        total_frames: 2_000,
        aligned_input_frames: 2_000,
        baseline_executed_only: true,
        proposal_only: true,
        conductor_gate_observed: true,
        autonomic_gate_observed: true,
        final_motor_gate_observed: true,
        possession_lease_observed: true,
        brainstem_gate_observed: true,
        safety_invariant_violations: 0,
        hardcoded_fallback_verified: true,
        atomic_activation_verified: true,
        rollback_verified: true,
        baseline: LocomotionPolicyMetrics {
            collision_rate: 0.10,
            progress_m: 10.0,
            oscillations_per_m: 0.4,
            energy_per_m: 1.0,
            recovery_success_rate: 0.70,
            command_instability: 0.30,
        },
        candidate,
    }
}

#[test]
fn shadow_frame_proves_identical_input_and_baseline_only_execution() {
    let input = LocomotionInput {
        clearance_front_m: 0.42,
        ..LocomotionInput::default()
    };
    let baseline = LocomotionOutput {
        forward_velocity_m_s: 0.2,
        angular_velocity_rad_s: 0.1,
        recovery_activation: 0.0,
    };
    let candidate = LocomotionOutput {
        forward_velocity_m_s: 0.25,
        angular_velocity_rad_s: 0.05,
        recovery_activation: 0.0,
    };
    let frame = LocomotionShadowFrame::new(
        "frame-1",
        123,
        input.clone(),
        baseline,
        candidate,
        baseline,
        "locomotion.hardcoded_wander.v0",
        "candidate-good",
        8,
        12,
        Some(0.9),
        None,
    );
    assert_eq!(frame.input_id, locomotion_input_id(&input));
    assert!(frame.baseline_executed_only);
    assert!(frame.disagreement > 0.0);
}

#[test]
fn promotion_requires_consistent_simulation_and_physical_shadow_evidence() {
    let candidate = LocomotionPolicyMetrics {
        collision_rate: 0.08,
        progress_m: 11.0,
        oscillations_per_m: 0.3,
        energy_per_m: 1.02,
        recovery_success_rate: 0.75,
        command_instability: 0.25,
    };
    let evidence = LocomotionPromotionEvidence {
        schema_version: LOCOMOTION_SHADOW_SCHEMA_VERSION,
        simulation: shadow_report(ShadowEnvironment::HeldOutSimulation, candidate),
        physical: shadow_report(ShadowEnvironment::Physical, candidate),
    };
    assert!(evaluate_locomotion_promotion(&evidence, Default::default()).promote);
}

#[test]
fn deliberately_poor_candidate_is_rejected_and_hardcoded_remains_fallback() {
    let poor = LocomotionPolicyMetrics {
        collision_rate: 0.35,
        progress_m: 4.0,
        oscillations_per_m: 1.2,
        energy_per_m: 1.8,
        recovery_success_rate: 0.25,
        command_instability: 0.9,
    };
    let mut physical = shadow_report(ShadowEnvironment::Physical, poor);
    physical.hardcoded_fallback_verified = false;
    physical.rollback_verified = false;
    let evidence = LocomotionPromotionEvidence {
        schema_version: LOCOMOTION_SHADOW_SCHEMA_VERSION,
        simulation: shadow_report(ShadowEnvironment::HeldOutSimulation, poor),
        physical,
    };
    let decision = evaluate_locomotion_promotion(&evidence, Default::default());
    assert!(!decision.promote);
    assert!(decision
        .reasons
        .iter()
        .any(|reason| reason.contains("fallback")));
    assert!(decision
        .reasons
        .iter()
        .any(|reason| reason.contains("rollback")));
    assert!(decision
        .reasons
        .iter()
        .any(|reason| reason.contains("collision")));
}

fn two_species_population() -> (Population, StdRng) {
    let mut rng = StdRng::seed_from_u64(17);
    let config = NeatConfig {
        population_size: 6,
        compatibility_threshold: 0.02,
        weight_mutation_rate: 0.0,
        add_connection_rate: 0.0,
        add_node_rate: 0.0,
        crossover_rate: 0.0,
        species_stagnation_generations: 3,
        ..NeatConfig::default()
    };
    let mut population = Population::seeded(config, &mut rng);
    let first_representative = population.genomes[0].clone();
    let mut second_representative = first_representative.clone();
    second_representative.mutate_add_node(&mut population.innovations, &mut rng);
    for genome in population.genomes.iter_mut().skip(3) {
        *genome = second_representative.clone();
    }
    population.species_records = vec![
        SpeciesRecord {
            id: 10,
            representative: first_representative,
            age: 0,
            best_fitness: f32::NEG_INFINITY,
            generations_without_improvement: 0,
        },
        SpeciesRecord {
            id: 20,
            representative: second_representative,
            age: 0,
            best_fitness: f32::NEG_INFINITY,
            generations_without_improvement: 0,
        },
    ];
    population.genome_species = vec![10, 10, 10, 20, 20, 20];
    population.next_species_id = 21;
    (population, rng)
}
