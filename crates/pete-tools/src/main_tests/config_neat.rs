use super::*;
use pete_actions::ActionPrimitive;
use pete_body::BodySense;
use pete_core::Reward;
use pete_experience::{Experience, ExperienceLatent};
use pete_now::{ExtensionSense, Now, SurpriseSense, VectorArtifact, SCENE_VECTOR_COLLECTION};
use pete_sensors::World;
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_path(prefix: &str) -> std::path::PathBuf {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    std::env::temp_dir().join(format!("{prefix}_{now_ms}"))
}

#[test]
fn robot_tick_delay_counts_work_inside_the_requested_period() {
    assert_eq!(
        remaining_tick_delay(20, Duration::from_millis(7)),
        Duration::from_millis(13)
    );
    assert_eq!(
        remaining_tick_delay(20, Duration::from_millis(25)),
        Duration::ZERO
    );
}

#[test]
fn lua_skill_speech_intents_are_execution_and_operation_scoped() {
    let record = serde_json::json!({
        "execution_id": 17,
        "trace": [
            {
                "kind": "primitive",
                "operation_id": 3,
                "operation": "say",
                "detail": {"text": "Hello Alex."},
            },
            {
                "kind": "primitive",
                "operation_id": 4,
                "operation": "drive",
                "detail": {"linear_mm_s": 50},
            },
        ],
    });
    assert_eq!(
        lua_skill_speech_intents(&record),
        vec![("lua:17:3".to_string(), "Hello Alex.".to_string())]
    );
}

#[test]
fn bump_recovery_cleanup_clears_the_bump_latch_after_contact_releases() {
    let mut cockpit = SafeCockpit::new(LocalSimCockpit::new().with_unscoped_bench_mode());
    cockpit.client_mut().set_bump(true, false);
    cockpit.client_mut().set_bump(false, false);

    clear_bump_recovery_latches(&mut cockpit, false).unwrap();

    let status = cockpit.refresh_status().unwrap();
    assert_eq!(status.safety_tripped, Some(false));
    assert_eq!(status.estop_latched, Some(false));
}

#[test]
fn bump_recovery_cleanup_clears_an_estop_observed_during_the_bump_incident() {
    let mut cockpit = SafeCockpit::new(LocalSimCockpit::new().with_unscoped_bench_mode());
    cockpit.client_mut().set_bump(true, false);
    cockpit.client_mut().estop().unwrap();
    cockpit.client_mut().set_bump(false, false);

    clear_bump_recovery_latches(&mut cockpit, true).unwrap();

    let status = cockpit.refresh_status().unwrap();
    assert_eq!(status.estop_latched, Some(false));
    assert_eq!(status.safety_tripped, Some(false));
}

#[test]
fn bump_recovery_cleanup_preserves_an_unrelated_estop() {
    let mut cockpit = SafeCockpit::new(LocalSimCockpit::new().with_unscoped_bench_mode());
    cockpit.client_mut().estop().unwrap();

    let error = clear_bump_recovery_latches(&mut cockpit, false).unwrap_err();
    assert!(error.to_string().contains("already latched"));
    assert_eq!(cockpit.refresh_status().unwrap().estop_latched, Some(true));
}

#[test]
fn executable_action_selector_defaults_to_goal() {
    let Cli {
        command: Command::Sim(args),
    } = Cli::try_parse_from(["pete", "sim"])
        .expect("sim command should parse with its production defaults")
    else {
        panic!("expected sim command");
    };
    assert_eq!(args.action_selector, CliActionSelectorMode::Goal);

    let Cli {
        command: Command::EvalScenario(args),
    } = Cli::try_parse_from(["pete", "eval-scenario", "--scenario", "empty-room"])
        .expect("eval-scenario command should parse with its production defaults")
    else {
        panic!("expected eval-scenario command");
    };
    assert_eq!(args.action_selector, CliActionSelectorMode::Goal);
}

fn evaluation_with_competence(
    stage: CurriculumStage,
    episodes: usize,
    successes: usize,
    weighted_score: f32,
) -> NeatPolicyEvaluation {
    let mut evaluation = NeatPolicyEvaluation::default();
    evaluation.traits = FitnessTraits {
        exploration: 0.0,
        escape_rate: 0.0,
        collision_rate: 0.0,
        energy_use: 0.0,
        forward_progress: 0.0,
        repetition_rate: 0.0,
        worst_environment_score: 0.0,
        safety_veto_rate: 0.0,
        safety_invariant_violations: 0,
    };
    evaluation.stage_competence.insert(
        stage,
        StageCompetence {
            episodes,
            successes,
            weighted_score,
            collision_rate: 0.0,
            invariant_violations: 0,
        },
    );
    evaluation
}

fn feasible_pareto(front: u32) -> SelectionSummary {
    SelectionSummary {
        fitness: 1_000.0,
        constraint_violations: 0,
        stage_success_rate: 0.0,
        prerequisite_floor: 0.0,
        stage_score: 0.0,
        pareto_front: front,
        crowding_distance: 1.0,
    }
}

#[test]
fn stage_success_outranks_novelty_and_pareto_when_other_candidate_has_zero_success() {
    let stage = CurriculumStage::LeaveStartRegion;
    let successful = evaluation_with_competence(stage, 4, 1, 5.0);
    let novel_failure = evaluation_with_competence(stage, 4, 0, 500.0);
    let summaries = structured_selection_summaries(
        stage,
        &[successful, novel_failure],
        &[feasible_pareto(8), feasible_pareto(0)],
        &[0.0, 100_000.0],
        25.0,
    );
    assert!(summaries[0].fitness > summaries[1].fitness);
    assert_eq!(summaries[1].stage_success_rate, 0.0);
}

#[test]
fn excessive_veto_reliance_cannot_dominate_through_stage_success() {
    let stage = CurriculumStage::LeaveStartRegion;
    let mut veto_dependent = evaluation_with_competence(stage, 4, 4, 100.0);
    veto_dependent.traits.safety_veto_rate = 0.23;
    let clean = evaluation_with_competence(stage, 4, 3, 10.0);
    let evaluations = [veto_dependent, clean];
    let traits = evaluations
        .iter()
        .map(|evaluation| evaluation.traits)
        .collect::<Vec<_>>();
    let pareto = pete_neat::selection_summaries(&traits, stage.selection_constraints());
    let summaries =
        structured_selection_summaries(stage, &evaluations, &pareto, &[100.0, 0.0], 25.0);
    assert!(summaries[0].constraint_violations > 0);
    assert!(summaries[1].fitness > summaries[0].fitness);
}

#[tokio::test]
async fn ancestral_challenge_uses_ancestral_stage_evidence_and_weights() {
    let challenge = NeatEnvironmentChallenge {
        stage: CurriculumStage::EscapeCorners,
        kind: ScenarioKind::CornerTrap,
        seed: 44,
        arena_override_m: None,
        initial_battery: None,
        disable_chargers: false,
    };
    let evaluation = evaluate_neat_locomotion(
        None,
        CurriculumStage::ExploreWithoutLooping,
        1,
        2,
        44,
        NeatPerturbation::default(),
        Some(&[challenge]),
        false,
    )
    .await
    .unwrap();
    assert!(evaluation
        .stage_competence
        .contains_key(&CurriculumStage::EscapeCorners));
    assert!(!evaluation
        .stage_competence
        .contains_key(&CurriculumStage::ExploreWithoutLooping));
    let evidence = evaluation.stage_competence[&CurriculumStage::EscapeCorners];
    assert_eq!(evidence.episodes, 1);
    assert_eq!(evaluation.fitness, evidence.weighted_score);
}

#[test]
fn challenge_plan_keeps_half_fresh_current_stage_and_covers_prerequisites() {
    let plan = neat_generation_challenges(
        CurriculumStage::LeaveStartRegion,
        8,
        7,
        &[],
        0.25,
        0.25,
        0.20,
    );
    let current = plan
        .iter()
        .filter(|challenge| challenge.stage == CurriculumStage::LeaveStartRegion)
        .count();
    assert!(current * 2 >= plan.len());
    for prerequisite in [
        CurriculumStage::BackAwayReliably,
        CurriculumStage::ChooseUsefulTurn,
        CurriculumStage::EscapeCorners,
    ] {
        assert!(plan.iter().any(|challenge| challenge.stage == prerequisite));
    }
}

#[test]
fn short_cycle_detection_separates_abab_from_long_route_revisit() {
    let mut recent = VecDeque::new();
    let mut looping = NeatEpisodeMetrics::default();
    for cell in [(0, 0), (1, 0), (0, 0), (1, 0), (0, 0)] {
        observe_short_cycles(&mut recent, cell, &mut looping);
    }
    assert!(looping.short_cycle_count >= 2);

    let mut recent = VecDeque::new();
    let mut useful = NeatEpisodeMetrics::default();
    observe_short_cycles(&mut recent, (0, 0), &mut useful);
    for x in 1..=30 {
        observe_short_cycles(&mut recent, (x, 0), &mut useful);
    }
    observe_short_cycles(&mut recent, (0, 0), &mut useful);
    assert_eq!(useful.short_cycle_count, 0);
}

#[test]
fn schema2_numeric_stage_is_resolved_before_new_order_is_applied() {
    assert_eq!(
        schema2_curriculum_stage(3),
        Some(CurriculumStage::ExploreWithoutLooping)
    );
    assert_eq!(CurriculumStage::ORDER[3], CurriculumStage::LeaveStartRegion);
}

#[test]
fn validation_candidate_set_includes_archived_generalist() {
    let mut rng = StdRng::seed_from_u64(91);
    let population = Population::seeded(
        NeatConfig {
            population_size: 3,
            ..NeatConfig::default()
        },
        &mut rng,
    );
    let archived = NeatNicheArchiveEntry {
        stage: CurriculumStage::EscapeCorners,
        niche: pete_neat::NicheLabel::Generalist,
        descriptor: QualityDiversityDescriptor {
            collision_frequency_bin: 0,
            turning_intensity_bin: 0,
            area_coverage_bin: 0,
            energy_consumption_bin: 0,
            recovery_aggressiveness_bin: 0,
        },
        selection_fitness: 1.0,
        base_selection_fitness: 1.0,
        novelty_score: 0.0,
        diagnostic_fitness: 1.0,
        qualification_evidence: None,
        genome: population.genomes[1].clone(),
        evaluation: evaluation_with_competence(CurriculumStage::EscapeCorners, 1, 1, 1.0),
    };
    let candidates = validation_candidates(
        &population.genomes[0],
        &NeatPolicyEvaluation::default(),
        &None,
        &[],
        &[archived],
    );
    assert!(candidates
        .iter()
        .any(|(kind, _, _)| kind == "archived-generalist"));
}

#[test]
fn exploratory_discovery_remains_in_protected_recovery_repertoire() {
    let mut rng = StdRng::seed_from_u64(238);
    let population = Population::seeded(
        NeatConfig {
            population_size: 2,
            ..NeatConfig::default()
        },
        &mut rng,
    );
    let discovery = population.genomes[1].clone();
    let entry = NeatNicheArchiveEntry {
        stage: CurriculumStage::ExploreWithoutLooping,
        niche: pete_neat::NicheLabel::OpenRoomExplorer,
        descriptor: QualityDiversityDescriptor {
            collision_frequency_bin: 0,
            turning_intensity_bin: 0,
            area_coverage_bin: 4,
            energy_consumption_bin: 1,
            recovery_aggressiveness_bin: 0,
        },
        selection_fitness: 1.0,
        base_selection_fitness: 1.0,
        novelty_score: 0.35,
        diagnostic_fitness: -42.385,
        qualification_evidence: None,
        genome: discovery.clone(),
        evaluation: NeatPolicyEvaluation {
            traits: evaluation_with_competence(CurriculumStage::ExploreWithoutLooping, 1, 1, 1.0)
                .traits,
            metrics: NeatEpisodeMetrics {
                new_area_cells: 113,
                distance_without_collision_m: 90.34,
                repeated_state_steps: 1892,
                ..NeatEpisodeMetrics::default()
            },
            ..NeatPolicyEvaluation::default()
        },
    };
    let founders = archive_recovery_founders(&[entry], &[]);
    assert!(founders.contains(&discovery));
}

#[test]
fn lifetime_selection_uses_mean_lower_quartile_and_worst_episode() {
    let summary = lifetime_selection_summary(&[10.0, 20.0, -10.0, 30.0], 3, 4);
    assert_eq!(summary.mean_score, 12.5);
    assert_eq!(summary.lower_quartile_score, -10.0);
    assert_eq!(summary.worst_score, -10.0);
    assert_eq!(summary.qualification_probability, 0.75);
    assert_eq!(summary.robustness_score, 1.25);
    assert_eq!(lifetime_selection_adjustment(summary), 8.75);
}

#[test]
fn safety_veto_is_not_an_invariant_violation_when_it_stops_motion() {
    let stopped = pete_autonomic::SafetyDecision {
        command: pete_cockpit::MotorCommand::stop(),
        vetoed: true,
        reason: Some(pete_autonomic::SafetyReason::Cliff),
        events: Vec::new(),
    };
    let moving = pete_autonomic::SafetyDecision {
        command: pete_cockpit::MotorCommand {
            forward: 0.1,
            turn: 0.0,
        },
        ..stopped.clone()
    };

    assert!(!neat_safety_invariant_violated(&stopped));
    assert!(neat_safety_invariant_violated(&moving));
}

#[test]
fn transfer_robustness_uses_success_collision_and_invariants() {
    let safe_traits =
        evaluation_with_competence(CurriculumStage::TransferCandidatesToPete, 1, 1, 0.0).traits;
    let nominal = NeatPolicyEvaluation {
        successful_episodes: 80,
        episodes: 100,
        collision_rate: 0.01,
        traits: safe_traits,
        ..NeatPolicyEvaluation::default()
    };
    let retained = NeatPolicyEvaluation {
        successful_episodes: 60,
        episodes: 100,
        collision_rate: 0.03,
        traits: safe_traits,
        ..NeatPolicyEvaluation::default()
    };
    let regressed = NeatPolicyEvaluation {
        successful_episodes: 59,
        episodes: 100,
        collision_rate: 0.01,
        traits: safe_traits,
        ..NeatPolicyEvaluation::default()
    };
    let veto_dependent = NeatPolicyEvaluation {
        traits: FitnessTraits {
            safety_veto_rate: 0.25,
            ..safe_traits
        },
        ..retained.clone()
    };

    assert!(transfer_perturbation_robust(&nominal, &retained));
    assert!(!transfer_perturbation_robust(&nominal, &regressed));
    assert!(!transfer_perturbation_robust(&nominal, &veto_dependent));
}

#[test]
fn chirp_patterns_use_expected_note_sequences() {
    for (pattern, expected) in [
        ("Confirm", vec![79, 84, 79]),
        ("Warning", vec![79, 75]),
        ("Hello", vec![72, 76, 79]),
        ("Goodbye", vec![79, 76, 72]),
        ("Curious", vec![72, 76, 74]),
        ("Idea", vec![76, 81, 84]),
        ("GoalAcquired", vec![72, 79, 84, 91]),
        ("goal-acquired", vec![72, 79, 84, 91]),
        ("Searching", vec![72, 74, 76, 74]),
        ("SawSomething", vec![84, 91]),
        ("Surprise", vec![72, 84]),
        ("Learned", vec![74, 79, 83]),
        ("PersonRecognized", vec![76, 79, 84, 79]),
        ("ObjectRecognized", vec![79, 84, 76]),
        ("PlaceRecognized", vec![79, 84, 72]),
        ("DidntUnderstand", vec![79, 81, 78]),
        ("didn't-understand", vec![79, 81, 78]),
        ("Docking", vec![67, 72, 76, 79]),
        ("ChargingStarted", vec![60, 67, 72]),
        ("Sleep", vec![79, 76, 72, 67]),
        ("Wake", vec![67, 72, 79]),
    ] {
        let notes: Vec<u8> = chirp_pattern_song(pattern)
            .tones
            .into_iter()
            .map(|tone| tone.note)
            .collect();
        assert_eq!(notes, expected, "pattern {pattern}");
    }
}

#[test]
fn live_sim_default_llm_timeout_allows_slow_ollama_responses() {
    let previous = std::env::var("PETE_LIVE_LLM_TIMEOUT_MS").ok();
    std::env::remove_var("PETE_LIVE_LLM_TIMEOUT_MS");

    let config = configured_llm_config_for_sim(&LlmArgs::default(), true).unwrap();

    match previous {
        Some(value) => std::env::set_var("PETE_LIVE_LLM_TIMEOUT_MS", value),
        None => std::env::remove_var("PETE_LIVE_LLM_TIMEOUT_MS"),
    }

    assert_eq!(config.timeout_ms, DEFAULT_LIVE_LLM_TIMEOUT_MS);
}

#[test]
fn background_sensor_keeps_asr_packet_ahead_of_latest_pcm() {
    let mut state = BackgroundSenseState::default();
    let asr = SensePacket::Ear(EarSense {
        transcript: Some("hello robot".to_string()),
        ..EarSense::default()
    });
    let pcm = SensePacket::EarPcm(PcmAudioFrame {
        captured_at_ms: 42,
        sample_rate_hz: 16_000,
        channels: 1,
        samples: vec![0; 160],
    });

    state.record_packet("microphone", asr);
    state.record_packet("microphone", pcm);

    let first = state.next_packet().expect("queued ASR packet");
    assert!(matches!(
        first,
        SensePacket::Ear(EarSense {
            transcript: Some(text),
            ..
        }) if text == "hello robot"
    ));
    assert!(matches!(state.next_packet(), Some(SensePacket::EarPcm(_))));
}

#[test]
fn sim_args_parse_virtual_live_tls_flags() {
    let cli = Cli::try_parse_from([
        "pete",
        "sim",
        "--live",
        "--live-tls",
        "--live-addr",
        "0.0.0.0:9443",
        "--live-tls-cert",
        "certs/test.crt",
        "--live-tls-key",
        "certs/test.key",
        "--scenario",
        "charger-seeking",
        "--steps",
        "123",
    ])
    .unwrap();

    let Command::Sim(args) = cli.command else {
        panic!("expected sim command");
    };
    assert!(args.live);
    assert!(args.live_tls);
    assert_eq!(args.live_addr.port(), 9443);
    assert_eq!(args.live_tls_cert, "certs/test.crt");
    assert_eq!(args.live_tls_key, "certs/test.key");
    assert_eq!(args.scenario, ScenarioArg::ChargerSeeking);
    assert_eq!(args.steps, 123);
}

#[test]
fn scenario_arg_parses_all_public_slugs() {
    for (slug, expected) in [
        ("empty-room", ScenarioArg::EmptyRoom),
        ("obstacle-avoidance", ScenarioArg::ObstacleAvoidance),
        ("corner-trap", ScenarioArg::CornerTrap),
        ("column-trap", ScenarioArg::ColumnTrap),
        ("charger-seeking", ScenarioArg::ChargerSeeking),
        ("person-speaker-room", ScenarioArg::PersonSpeakerRoom),
        ("mixed-room", ScenarioArg::MixedRoom),
        ("dream", ScenarioArg::Dream),
    ] {
        let cli = Cli::try_parse_from(["pete", "sim", "--scenario", slug]).unwrap();
        let Command::Sim(args) = cli.command else {
            panic!("expected sim command");
        };
        assert_eq!(args.scenario, expected);
    }
}

