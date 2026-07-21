
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

fn eval_args(
    scenario: ScenarioArg,
    episodes: usize,
    steps: usize,
    out: Option<String>,
) -> EvalScenarioArgs {
    EvalScenarioArgs {
        scenario,
        episodes,
        steps,
        seed: 7,
        tick_ms: 100,
        out,
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
    }
}

fn replay_counterfactual_args(
    capture: String,
    out_ledger: Option<String>,
    out_report: Option<String>,
) -> ReplayCounterfactualArgs {
    ReplayCounterfactualArgs {
        capture,
        edit: Vec::new(),
        policy: "baseline".to_string(),
        actions: None,
        steps: Some(4),
        out_ledger,
        out_report,
        llm: LlmArgs::default(),
    }
}

#[test]
fn counterfactual_edit_parser_parses_supported_edits() {
    assert_eq!(
        parse_counterfactual_edit("move-charger:x=1.0,y=2.0").unwrap(),
        CounterfactualEdit::MoveObject {
            kind: CounterfactualObjectKind::Charger,
            id: None,
            x_m: 1.0,
            y_m: 2.0,
        }
    );
    assert_eq!(
        parse_counterfactual_edit("set-battery:value=0.42").unwrap(),
        CounterfactualEdit::SetBattery { value: 0.42 }
    );
    assert!(parse_counterfactual_edit("move-moon:x=1,y=2")
        .unwrap_err()
        .to_string()
        .contains("unknown counterfactual edit"));
}

#[test]
fn counterfactual_edits_move_charger_and_set_battery() {
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ChargerSeeking, 9));
    let mut metadata = scenario.metadata;
    let edits = vec![
        parse_counterfactual_edit("move-charger:x=1.0,y=1.0").unwrap(),
        parse_counterfactual_edit("set-battery:value=0.75").unwrap(),
    ];
    let mut warnings = Vec::new();

    apply_counterfactual_edits(&mut metadata, &edits, &mut warnings).unwrap();

    let charger = metadata
        .objects
        .iter()
        .find(|object| matches!(object.kind, pete_sim::SimObjectKind::Charger))
        .unwrap();
    assert_eq!((charger.x_m, charger.y_m), (1.0, 1.0));
    assert_eq!(metadata.body.battery_level, 0.75);
    assert!(warnings
        .iter()
        .any(|warning| warning.contains("first matching object")));
}

#[test]
fn counterfactual_report_serializes_schema() {
    let report = CounterfactualReport {
        schema_version: 1,
        source_capture: "capture".to_string(),
        reconstructable: true,
        edits: vec!["set-battery:value=0.5".to_string()],
        policy: "stop".to_string(),
        steps: 3,
        summary: CounterfactualSummary {
            collisions: 0,
            charging_ticks: 1,
            battery_delta: 0.1,
            distance_traveled: 0.2,
            final_distance_to_charger_m: Some(0.3),
        },
        warnings: Vec::new(),
    };

    let value: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&report).unwrap()).unwrap();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["summary"]["charging_ticks"], 1);
}

#[tokio::test]
async fn replay_counterfactual_baseline_writes_ledger_and_report() {
    let temp_dir = temp_path("pete_counterfactual_baseline");
    let capture_dir = temp_dir.join("capture");
    let ledger_dir = temp_dir.join("ledger");
    let report_path = temp_dir.join("report.json");
    let mut scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ChargerSeeking, 77));
    let snapshot = scenario.world.snapshot().await.unwrap();
    let mut writer = CaptureWriter::create(&capture_dir, CaptureSource::Sim, Some(100))
        .await
        .unwrap();
    writer.manifest_mut().scenario = Some(scenario.metadata);
    writer
        .append_snapshot(snapshot.body.last_update_ms, snapshot, Vec::new())
        .await
        .unwrap();
    writer.finish().await.unwrap();

    let args = replay_counterfactual_args(
        capture_dir.to_string_lossy().to_string(),
        Some(ledger_dir.to_string_lossy().to_string()),
        Some(report_path.to_string_lossy().to_string()),
    );
    replay_counterfactual(args).await.unwrap();

    let transitions = JsonlLedger::new(&ledger_dir).transitions().await.unwrap();
    assert!(!transitions.is_empty());
    let report: CounterfactualReport =
        serde_json::from_slice(&fs::read(&report_path).unwrap()).unwrap();
    assert!(report.reconstructable);
    assert_eq!(report.steps, 4);
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn replay_counterfactual_with_moved_charger_writes_report() {
    let temp_dir = temp_path("pete_counterfactual_moved_charger");
    let capture_dir = temp_dir.join("capture");
    let report_path = temp_dir.join("report.json");
    let mut scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ChargerSeeking, 78));
    let snapshot = scenario.world.snapshot().await.unwrap();
    let mut writer = CaptureWriter::create(&capture_dir, CaptureSource::Sim, Some(100))
        .await
        .unwrap();
    writer.manifest_mut().scenario = Some(scenario.metadata);
    writer
        .append_snapshot(snapshot.body.last_update_ms, snapshot, Vec::new())
        .await
        .unwrap();
    writer.finish().await.unwrap();

    let mut args = replay_counterfactual_args(
        capture_dir.to_string_lossy().to_string(),
        None,
        Some(report_path.to_string_lossy().to_string()),
    );
    args.edit = vec!["move-charger:x=1.0,y=1.0".to_string()];
    args.policy = "seek-charge".to_string();
    replay_counterfactual(args).await.unwrap();

    let report: CounterfactualReport =
        serde_json::from_slice(&fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(report.edits, vec!["move-charger:x=1.0,y=1.0"]);
    assert_eq!(report.policy, "seek-charge");
    assert!(report
        .warnings
        .iter()
        .any(|warning| warning.contains("first matching object")));
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn replay_counterfactual_passive_capture_fails_clearly() {
    let temp_dir = temp_path("pete_counterfactual_passive");
    let capture_dir = temp_dir.join("capture");
    let mut writer = CaptureWriter::create(&capture_dir, CaptureSource::Replay, Some(100))
        .await
        .unwrap();
    writer
        .append_snapshot(0, WorldSnapshot::default(), Vec::new())
        .await
        .unwrap();
    writer.finish().await.unwrap();

    let err = replay_counterfactual(replay_counterfactual_args(
        capture_dir.to_string_lossy().to_string(),
        None,
        None,
    ))
    .await
    .unwrap_err()
    .to_string();
    assert!(err.contains(
            "passive captures without reconstructable sim metadata cannot yet be counterfactually replayed"
        ));
    let _ = fs::remove_dir_all(&temp_dir);
}

fn tick_with_action(action: ActionPrimitive) -> RuntimeTick {
    let now = Now::blank(100, BodySense::default());
    RuntimeTick {
        frame: ExperienceFrame {
            id: uuid::Uuid::new_v4(),
            t_ms: 100,
            now,
            sensations: Vec::new(),
            impressions: Vec::new(),
            experiences: Vec::new(),
            z: None,
            chosen_action: Some(action.clone()),
            conscious_command: None,
            reign_input: None,
            reign_outcome: None,
            predicted_futures: Vec::new(),
            behavior_runs: Vec::new(),
            actual_next: None,
            reward: Reward::default(),
            surprise: SurpriseSense::default(),
            memory_recall: Vec::new(),
            recollections: Vec::new(),
            llm_teaching: Vec::new(),
            counterfactuals: Vec::new(),
            notes: Vec::new(),
        },
        experience: pete_experience::Experience::new(
            "test",
            "test",
            Vec::new(),
            Vec::new(),
            100,
            100,
        ),
        chosen_action: Some(action),
        skill_request: None,
        skill_status: None,
        recall: Default::default(),
        llm: Default::default(),
        combobulation: None,
        inline_learning: Default::default(),
    }
}

fn scenario_report_for_comparison(
    scenario: &str,
    episodes: usize,
    success_rate: f32,
    collision_rate: f32,
    mean_battery_delta: f32,
    model_fallbacks: usize,
) -> ScenarioEvaluationReport {
    ScenarioEvaluationReport {
        schema_version: 1,
        scenario: scenario.to_string(),
        base_seed: 7,
        episodes,
        steps_per_episode: 100,
        tick_ms: 100,
        action_selector_mode: "baseline".to_string(),
        model_modes: HashMap::new(),
        model_loading: RuntimeModelLoadReport::default(),
        ledger: None,
        capture_root: None,
        summary: ScenarioEvaluationSummary {
            success_rate,
            collision_rate,
            mean_collisions_per_episode: collision_rate * episodes as f32,
            mean_battery_delta,
            mean_final_battery: 0.5,
            mean_distance_to_charger_final_m: None,
            mean_nearest_obstacle_m: None,
            mean_distance_traveled_m: 1.0,
            mean_ticks_survived: 100.0,
            mean_safety_interventions: 0.0,
            behavior_run_records: 0,
            model_fallbacks,
            model_assisted_decisions: 0,
            action_selector_safety_overrides: 0,
            mean_chosen_score: None,
            mean_candidate_score: None,
            ..ScenarioEvaluationSummary::default()
        },
        memory: None,
        episodes_detail: Vec::new(),
        recommendation: "pass".to_string(),
        warnings: Vec::new(),
    }
}

#[test]
fn scenario_report_round_trips_json() {
    let report = ScenarioEvaluationReport {
        schema_version: 1,
        scenario: "empty-room".to_string(),
        base_seed: 7,
        episodes: 1,
        steps_per_episode: 2,
        tick_ms: 100,
        action_selector_mode: "baseline".to_string(),
        model_modes: HashMap::new(),
        model_loading: RuntimeModelLoadReport::default(),
        ledger: None,
        capture_root: None,
        summary: ScenarioEvaluationSummary::default(),
        memory: None,
        episodes_detail: Vec::new(),
        recommendation: "insufficient_data".to_string(),
        warnings: Vec::new(),
    };
    let encoded = serde_json::to_string(&report).unwrap();
    let decoded: ScenarioEvaluationReport = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded.scenario, "empty-room");
    assert_eq!(decoded.schema_version, 1);
}

#[test]
fn scenario_comparison_recommends_pass_candidate() {
    let baseline = scenario_report_for_comparison("column-trap", 10, 0.7, 0.2, -0.02, 0);
    let candidate = scenario_report_for_comparison("column-trap", 10, 0.8, 0.15, 0.01, 0);

    let comparison =
        compare_scenario_reports("baseline.json", "candidate.json", &baseline, &candidate);

    assert_eq!(
        comparison.recommendation,
        ScenarioComparisonRecommendation::PassCandidate
    );
    assert_eq!(comparison.baseline_report_path, "baseline.json");
    assert!(
        comparison
            .deltas
            .get("success_rate")
            .copied()
            .unwrap_or_default()
            > 0.0
    );
    assert!(comparison.warnings.is_empty());
}

#[test]
fn scenario_comparison_detects_regression() {
    let baseline = scenario_report_for_comparison("column-trap", 10, 0.8, 0.1, 0.02, 0);
    let candidate = scenario_report_for_comparison("column-trap", 10, 0.7, 0.3, -0.05, 2);

    let comparison =
        compare_scenario_reports("baseline.json", "candidate.json", &baseline, &candidate);

    assert_eq!(
        comparison.recommendation,
        ScenarioComparisonRecommendation::RegressionDetected
    );
    assert!(comparison.compared_metrics.collision_rate.regression);
    assert!(comparison.compared_metrics.model_fallbacks.regression);
    assert!(comparison
        .warnings
        .iter()
        .any(|warning| warning.contains("collision_rate regressed")));
}

#[test]
fn scenario_comparison_reports_insufficient_data() {
    let baseline = scenario_report_for_comparison("column-trap", 2, 0.8, 0.1, 0.0, 0);
    let candidate = scenario_report_for_comparison("column-trap", 2, 0.8, 0.1, 0.0, 0);

    let comparison =
        compare_scenario_reports("baseline.json", "candidate.json", &baseline, &candidate);

    assert_eq!(
        comparison.recommendation,
        ScenarioComparisonRecommendation::InsufficientData
    );
}

#[test]
fn scenario_comparison_report_writes_json_artifact() {
    let baseline = scenario_report_for_comparison("column-trap", 10, 0.7, 0.2, -0.02, 0);
    let candidate = scenario_report_for_comparison("column-trap", 10, 0.8, 0.15, 0.01, 0);
    let comparison =
        compare_scenario_reports("baseline.json", "candidate.json", &baseline, &candidate);
    let root = temp_path("pete_comparison_report");
    let out = root.join("data/reports/comparisons/column-trap.json");

    write_scenario_comparison_report(&out, &comparison).unwrap();
    let decoded = load_scenario_comparison_report(&out.to_string_lossy()).unwrap();

    assert_eq!(
        decoded.recommendation,
        ScenarioComparisonRecommendation::PassCandidate
    );
    assert!(
        (decoded
            .compared_metrics
            .success_rate
            .delta
            .unwrap_or_default()
            - 0.1)
            .abs()
            < 0.001
    );
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn obstacle_metrics_count_collision_flags() {
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ObstacleAvoidance, 11));
    let mut metrics = EpisodeMetricBuilder::new(
        ScenarioKind::ObstacleAvoidance,
        scenario.metadata,
        0,
        11,
        None,
        None,
    );
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.flags.bump_left = true;
    snapshot.body.flags.wall = true;
    snapshot.body.flags.cliff_front_left = true;
    snapshot.range.nearest_m = Some(0.2);
    metrics.observe(
        &snapshot,
        &tick_with_action(ActionPrimitive::Go {
            intensity: 0.2,
            duration_ms: 100,
        }),
    );
    let report = metrics.finish();
    assert_eq!(report.collisions, 1);
    assert_eq!(report.wall_hits, 1);
    assert_eq!(report.bumper_hits, 1);
    assert_eq!(report.cliff_hits, 1);
    assert_eq!(report.min_nearest_obstacle_m, Some(0.2));
}

#[test]
fn action_selector_fallbacks_do_not_count_as_model_fallbacks() {
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::EmptyRoom, 14));
    let mut metrics = EpisodeMetricBuilder::new(
        ScenarioKind::EmptyRoom,
        scenario.metadata,
        0,
        14,
        None,
        None,
    );
    let mut tick = tick_with_action(ActionPrimitive::Stop);
    tick.frame.now.extensions.insert(
        "action_selector".to_string(),
        serde_json::to_value(ActionSelectionDecision {
            mode: ActionSelectorMode::ModelAssisted,
            candidates: vec![pete_runtime::ActionSelectionCandidateScore {
                action: ActionPrimitive::Stop,
                score: -0.5,
                fallback_used: true,
                ..Default::default()
            }],
            selected_action: Some(ActionPrimitive::Stop),
            baseline_action: Some(ActionPrimitive::Stop),
            selected_score: Some(-0.5),
            safety_overrode: true,
            fallback_warnings: vec!["action selector used hardcoded score".to_string()],
            ..Default::default()
        })
        .unwrap(),
    );
    metrics.observe(&WorldSnapshot::default(), &tick);

    let episode = metrics.finish();
    assert_eq!(episode.model_fallbacks, 0);
    assert_eq!(episode.action_selector_fallbacks, 1);
    assert_eq!(episode.action_selector_safety_overrides, 1);
    assert_eq!(episode.model_assisted_decisions, 1);

    let summary = summarize_episodes(&[episode]);
    assert_eq!(summary.model_fallbacks, 0);
    assert_eq!(summary.action_selector_fallbacks, 1);
}

#[test]
fn baseline_recovery_yields_are_reported_separately_from_selector_fallbacks() {
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::EmptyRoom, 16));
    let mut metrics = EpisodeMetricBuilder::new(
        ScenarioKind::EmptyRoom,
        scenario.metadata,
        0,
        16,
        None,
        None,
    );
    let turn_right = ActionPrimitive::Turn {
        direction: TurnDir::Right,
        intensity: 0.75,
        duration_ms: 500,
    };
    let mut tick = tick_with_action(turn_right.clone());
    tick.frame.now.extensions.insert(
        "action_selector".to_string(),
        serde_json::to_value(ActionSelectionDecision {
            mode: ActionSelectorMode::ModelAssisted,
            candidates: vec![pete_runtime::ActionSelectionCandidateScore {
                action: turn_right.clone(),
                score: 1.0,
                fallback_used: false,
                ..Default::default()
            }],
            selected_action: Some(turn_right.clone()),
            baseline_action: Some(turn_right.clone()),
            selected_score: None,
            safety_overrode: false,
            fallback_warnings: vec![
                "model-assisted selector yielded to baseline trap recovery".to_string()
            ],
            ..Default::default()
        })
        .unwrap(),
    );
    metrics.observe(&WorldSnapshot::default(), &tick);

    let episode = metrics.finish();
    assert_eq!(episode.action_selector_fallbacks, 0);
    assert_eq!(episode.action_selector_guard_yields, 1);

    let summary = summarize_episodes(&[episode]);
    assert_eq!(summary.action_selector_fallbacks, 0);
    assert_eq!(summary.action_selector_guard_yields, 1);
}

#[test]
fn goal_progress_metrics_are_counted_in_scenario_reports() {
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::EmptyRoom, 18));
    let mut metrics = EpisodeMetricBuilder::new(
        ScenarioKind::EmptyRoom,
        scenario.metadata,
        0,
        18,
        None,
        None,
    );
    let mut tick = tick_with_action(ActionPrimitive::Stop);
    tick.frame.now.extensions.insert(
        "goal_system".to_string(),
        serde_json::json!({
            "progress": [{
                "goal_id": "explore",
                "selected_behavior": "wall_follow_exploration",
                "previous_behavior": "random_walk_exploration",
                "expectation": {
                    "behavior_id": "random_walk_exploration",
                    "baseline": 0.0,
                    "expected_progress": 0.6,
                    "horizon_ms": 1000,
                    "tolerance": 0.1,
                    "deadline_ms": 100,
                    "metric": "frontier_coverage"
                },
                "observation": {
                    "observed_at_ms": 100,
                    "progress": 0.1,
                    "source": "canonical_world_model",
                    "outcome": null
                },
                "attempts": 3,
                "failed_attempts": 2,
                "recent_progress": 0.1,
                "progress_trend": -0.2,
                "last_progress_at_ms": null,
                "strategy_failure": 0.7,
                "response": "changed",
                "reason": "coverage stalled"
            }]
        }),
    );
    metrics.observe(&WorldSnapshot::default(), &tick);

    let mut help_tick = tick_with_action(ActionPrimitive::Stop);
    help_tick.frame.t_ms = 200;
    help_tick.frame.now.t_ms = 200;
    help_tick.frame.now.extensions.insert(
        "goal_system".to_string(),
        serde_json::json!({
            "progress": [{
                "goal_id": "seek_charger",
                "selected_behavior": "request_charge_help",
                "previous_behavior": "systematic_charger_search",
                "expectation": null,
                "observation": {
                    "observed_at_ms": 200,
                    "progress": null,
                    "source": "canonical_world_model_unmeasurable",
                    "outcome": null
                },
                "attempts": 4,
                "failed_attempts": 4,
                "recent_progress": 0.0,
                "progress_trend": 0.0,
                "last_progress_at_ms": null,
                "strategy_failure": 0.8,
                "response": "help_requested",
                "reason": "bounded escalation"
            }]
        }),
    );
    metrics.observe(&WorldSnapshot::default(), &help_tick);

    let episode = metrics.finish();
    assert_eq!(episode.goal_progress_samples, 1);
    assert_eq!(episode.mean_goal_progress, Some(0.1));
    assert_eq!(episode.goal_no_progress_dwell_ticks, 1);
    assert_eq!(episode.goal_failed_attempts, 6);
    assert_eq!(episode.strategy_switches_within_goal, 1);
    assert_eq!(episode.goal_help_requests, 1);
    assert_eq!(episode.unmeasurable_progress_ticks, 1);
    assert_eq!(episode.false_stall_rate, Some(0.0));

    let summary = summarize_episodes(&[episode]);
    assert_eq!(summary.mean_goal_progress, Some(0.1));
    assert_eq!(summary.goal_no_progress_dwell_ticks, 1);
    assert_eq!(summary.goal_failed_attempts, 6);
    assert_eq!(summary.strategy_switches_within_goal, 1);
    assert_eq!(summary.goal_help_requests, 1);
    assert_eq!(summary.unmeasurable_progress_ticks, 1);
    assert_eq!(summary.false_stall_rate, Some(0.0));
}

#[test]
fn map_memory_decisions_are_counted_in_scenario_reports() {
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::EmptyRoom, 17));
    let mut metrics = EpisodeMetricBuilder::new(
        ScenarioKind::EmptyRoom,
        scenario.metadata,
        0,
        17,
        None,
        None,
    );
    let mut tick = tick_with_action(ActionPrimitive::Turn {
        direction: TurnDir::Right,
        intensity: 0.5,
        duration_ms: 1_000,
    });
    tick.frame.now.extensions.insert(
        "action.motion_bridge".to_string(),
        serde_json::json!({
            "map_memory_decision": {
                "influenced": true,
                "navigation_intent": "avoid_known_danger_cell",
                "reason": "danger_safe_direction",
                "reason_string": "avoiding remembered danger using safe bearing",
                "signal": "memory.nearby_best_safe_direction_rad",
                "signal_value": -0.8,
                "signal_confidence": 0.9,
                "confidence": 0.9,
                "place_danger": 0.9,
                "place_charge_value": 0.0,
                "place_novelty": 0.2,
                "safe_direction_rad": -0.8,
                "charge_direction_rad": null,
                "selected_action": tick.chosen_action.clone(),
                "chosen_action": tick.chosen_action.clone(),
                "safety_overrode": false,
            }
        }),
    );
    metrics.observe(&WorldSnapshot::default(), &tick);

    let episode = metrics.finish();
    assert_eq!(episode.map_memory_decisions, 1);
    assert_eq!(episode.danger_memory_decisions, 1);
    assert_eq!(episode.charge_memory_decisions, 0);
    assert_eq!(episode.novelty_memory_decisions, 0);
    assert_eq!(
        episode
            .memory_navigation_intents
            .get("avoid_known_danger_cell"),
        Some(&1)
    );
    assert_eq!(
        episode
            .memory_navigation_reasons
            .get("danger_safe_direction"),
        Some(&1)
    );
    assert_eq!(
        episode
            .map_memory_signals
            .get("memory.nearby_best_safe_direction_rad"),
        Some(&1)
    );
    assert_eq!(episode.map_memory_safety_overrides, 0);
    assert_eq!(episode.map_memory_decision_samples.len(), 1);
    assert_eq!(
        episode.map_memory_decision_samples[0].chosen_action,
        tick.chosen_action
    );
    assert_eq!(
        episode.map_memory_decision_samples[0].signal_value,
        Some(-0.8)
    );

    let summary = summarize_episodes(&[episode]);
    assert_eq!(summary.map_memory_decisions, 1);
    assert_eq!(summary.danger_memory_decisions, 1);
    assert_eq!(
        summary
            .memory_navigation_intents
            .get("avoid_known_danger_cell"),
        Some(&1)
    );
    assert_eq!(
        summary
            .map_memory_signals
            .get("memory.nearby_best_safe_direction_rad"),
        Some(&1)
    );
}

#[test]
fn low_confidence_memory_navigation_fallbacks_are_counted_in_reports() {
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ChargerSeeking, 19));
    let mut metrics = EpisodeMetricBuilder::new(
        ScenarioKind::ChargerSeeking,
        scenario.metadata,
        0,
        19,
        None,
        None,
    );
    let mut tick = tick_with_action(ActionPrimitive::Stop);
    tick.frame.now.extensions.insert(
        "action.motion_bridge".to_string(),
        serde_json::json!({
            "map_memory_decision": {
                "influenced": true,
                "navigation_intent": "stop_ask_for_help_when_uncertain",
                "reason": "charge_low_confidence_fallback",
                "reason_string": "critical battery but charger memory is too weak",
                "signal": "memory.nearby_best_charge_direction_rad",
                "signal_value": null,
                "signal_confidence": 0.1,
                "confidence": 0.1,
                "place_danger": 0.0,
                "place_charge_value": 0.1,
                "place_novelty": 0.2,
                "safe_direction_rad": null,
                "charge_direction_rad": null,
                "selected_action": tick.chosen_action.clone(),
                "chosen_action": tick.chosen_action.clone(),
                "safety_overrode": true,
            }
        }),
    );
    metrics.observe(&WorldSnapshot::default(), &tick);

    let episode = metrics.finish();
    assert_eq!(episode.map_memory_decisions, 1);
    assert_eq!(episode.charge_memory_decisions, 1);
    assert_eq!(episode.low_confidence_navigation_fallbacks, 1);
    assert_eq!(episode.map_memory_safety_overrides, 1);
    assert_eq!(
        episode
            .memory_navigation_reasons
            .get("charge_low_confidence_fallback"),
        Some(&1)
    );
    assert_eq!(
        episode.map_memory_decision_samples[0].signal,
        "memory.nearby_best_charge_direction_rad"
    );
    assert!(episode.map_memory_decision_samples[0].safety_overrode);

    let summary = summarize_episodes(&[episode]);
    assert_eq!(summary.low_confidence_navigation_fallbacks, 1);
    assert_eq!(summary.map_memory_safety_overrides, 1);
}

#[test]
fn shadow_train_script_errors_do_not_count_as_model_fallbacks() {
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::EmptyRoom, 15));
    let mut metrics = EpisodeMetricBuilder::new(
        ScenarioKind::EmptyRoom,
        scenario.metadata,
        0,
        15,
        None,
        None,
    );
    metrics.observe_behavior_runs(&[ErasedBehaviorRunRecord {
        behavior_id: "event_bump".to_string(),
        regime: BehaviorRegime::ShadowTrain,
        t_ms: 100,
        input_json: Value::Null,
        hardcoded_json: Some(Value::Null),
        model_json: None,
        selected_json: Some(Value::Null),
        error: Some("event.bump.shadow.v0 has no observed script samples".to_string()),
        disagreement: None,
    }]);

    let episode = metrics.finish();
    assert_eq!(episode.behavior_run_records, 1);
    assert_eq!(episode.model_fallbacks, 0);
}

#[test]
fn stuck_metrics_count_recovery_events() {
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::CornerTrap, 11));
    let mut metrics = EpisodeMetricBuilder::new(
        ScenarioKind::CornerTrap,
        scenario.metadata,
        0,
        11,
        None,
        None,
    );
    let mut started = WorldSnapshot::default();
    started.extensions.push(ExtensionSense {
        schema_version: 1,
        name: "sim.stuck".to_string(),
        values: vec![1.0, 1.0, 6.0, 100.0, 1.0, -1.0, 1.0, 0.0],
    });
    metrics.observe(
        &started,
        &tick_with_action(ActionPrimitive::Explore {
            style: pete_actions::ExploreStyle::RandomWalk,
            duration_ms: 100,
        }),
    );
    let mut recovered = started.clone();
    recovered.body.odometry.x_m = 0.1;
    recovered.extensions[0].values = vec![0.0, 0.0, 0.0, 900.0, 0.0, -1.0, 0.0, 1.0];
    metrics.observe(
        &recovered,
        &tick_with_action(ActionPrimitive::Explore {
            style: pete_actions::ExploreStyle::RandomWalk,
            duration_ms: 100,
        }),
    );

    let report = metrics.finish();
    assert_eq!(report.stuck_count, 1);
    assert_eq!(report.stuck_ticks, 1);
    assert_eq!(report.recovery_attempts, 1);
    assert_eq!(report.stuck_duration, Some(900.0));
    assert_eq!(report.mean_stuck_duration, Some(900.0));
    assert_eq!(report.recovery_success_rate, Some(1.0));
    assert_eq!(report.mean_recovery_ticks, Some(9.0));
}

#[test]
fn metrics_record_dead_battery_tick() {
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::EmptyRoom, 12));
    let mut metrics = EpisodeMetricBuilder::new(
        ScenarioKind::EmptyRoom,
        scenario.metadata,
        0,
        12,
        None,
        None,
    );
    let mut alive = WorldSnapshot::default();
    alive.body.battery_level = 0.01;
    metrics.observe(&alive, &tick_with_action(ActionPrimitive::Stop));
    let mut dead = alive.clone();
    dead.body.battery_level = 0.0;
    metrics.observe(&dead, &tick_with_action(ActionPrimitive::Stop));

    let report = metrics.finish();
    assert_eq!(report.dead_battery_tick, Some(1));
}

#[test]
fn charger_metrics_detect_success_and_battery_delta() {
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ChargerSeeking, 12));
    let mut metrics = EpisodeMetricBuilder::new(
        ScenarioKind::ChargerSeeking,
        scenario.metadata,
        0,
        12,
        None,
        None,
    );
    let mut start = WorldSnapshot::default();
    start.body.battery_level = 0.2;
    metrics.observe(&start, &tick_with_action(ActionPrimitive::Stop));
    let mut charged = start.clone();
    charged.body.battery_level = 0.26;
    charged.body.charging = true;
    metrics.observe(&charged, &tick_with_action(ActionPrimitive::Stop));
    let report = metrics.finish();
    assert_eq!(report.charging_ticks, 1);
    assert!(report.battery_delta > 0.05);
    assert!(report.success);
}

#[test]
fn charger_metrics_report_visibility_approach_and_bad_dock_boundaries() {
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ChargerSeeking, 12));
    let mut metrics = EpisodeMetricBuilder::new(
        ScenarioKind::ChargerSeeking,
        scenario.metadata,
        0,
        12,
        None,
        None,
    );
    let mut visible = WorldSnapshot::default();
    visible.body.odometry.heading_rad = 0.25;
    visible.extensions.push(pete_now::ExtensionSense {
        schema_version: 1,
        name: "sim.world".to_string(),
        values: vec![4.0, 4.0, 1.0, 0.4, 0.6],
    });
    metrics.observe(
        &visible,
        &tick_with_action(ActionPrimitive::Approach {
            target: ApproachTarget::Charger,
        }),
    );

    let mut far = visible.clone();
    far.extensions[0].values[3] = 0.1;
    far.extensions[0].values[4] = 0.0;
    metrics.observe(&far, &tick_with_action(ActionPrimitive::Dock));

    let report = metrics.finish();
    assert_eq!(report.ticks_with_charger_visible, 1);
    assert_eq!(report.ticks_with_charger_near, 1);
    assert_eq!(report.ticks_approaching_charger, 1);
    assert_eq!(report.ticks_docking_from_too_far, 1);
    assert_eq!(report.final_heading_rad, Some(0.25));
    assert!(report.final_bearing_to_charger_rad.is_some());
}

#[test]
fn social_metrics_detect_projected_senses() {
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::PersonAndSpeaker, 13));
    let mut metrics = EpisodeMetricBuilder::new(
        ScenarioKind::PersonAndSpeaker,
        scenario.metadata,
        0,
        13,
        None,
        None,
    );
    let mut snapshot = WorldSnapshot::default();
    snapshot.eye.frames.push(vec![0.1, 0.2]);
    snapshot.ear.features.push(vec![0.3]);
    snapshot.voice.vectors.push(pete_now::VectorArtifact::new(
        "voices",
        "test-voice",
        vec![0.4],
    ));
    snapshot.face.vectors.push(pete_now::VectorArtifact::new(
        "faces",
        "test-face",
        vec![0.5],
    ));
    snapshot.kinect.skeletons.push(Default::default());
    metrics.observe(&snapshot, &tick_with_action(ActionPrimitive::Stop));
    let report = metrics.finish();
    assert_eq!(report.ticks_with_eye_frames, 1);
    assert_eq!(report.ticks_with_ear_features, 1);
    assert_eq!(report.ticks_with_voice_embeddings, 1);
    assert_eq!(report.ticks_with_face_embeddings, 1);
    assert_eq!(report.ticks_with_kinect_skeletons, 1);
    assert!(report.success);
}

#[test]
fn recommendation_logic_classifies_common_outcomes() {
    let strong = ScenarioEvaluationSummary {
        success_rate: 0.9,
        collision_rate: 0.01,
        ..ScenarioEvaluationSummary::default()
    };
    assert_eq!(
        scenario_recommendation(10, &strong),
        "candidate_for_more_eval"
    );
    assert_eq!(scenario_recommendation(2, &strong), "insufficient_data");
    let risky = ScenarioEvaluationSummary {
        success_rate: 0.9,
        collision_rate: 0.2,
        ..ScenarioEvaluationSummary::default()
    };
    assert_eq!(
        scenario_recommendation(10, &risky),
        "reject_or_continue_training"
    );
}

#[test]
fn memory_hit_rates_are_bounded() {
    assert_eq!(hit_rate(3, 2), Some(1.0));
    assert_eq!(hit_rate(1, 4), Some(0.25));
    assert_eq!(hit_rate(1, 0), None);
    assert_eq!(aggregate_hit_rate([(4, 2), (3, 1)].into_iter()), Some(1.0));
}

#[tokio::test]
async fn hardware_env_report_has_expected_shape() {
    let report = collect_hardware_env_report().await;
    assert!(report.get("os").is_some());
    assert!(report.get("architecture").is_some());
    assert!(report.get("serial_devices").unwrap().is_array());
    assert!(report.get("gps_serial_candidates").unwrap().is_array());
    assert_eq!(report["default_gps"]["baud"].as_u64(), Some(9600));
    assert!(report.get("lidar_serial_candidates").unwrap().is_array());
    assert_eq!(
        report["default_lidar"]["baud"].as_u64(),
        Some(u64::from(Lfcd2SenseProvider::BAUD_RATE))
    );
    assert!(report.get("i2c_devices").unwrap().is_array());
    assert_eq!(
        report["default_imu"]["device"].as_str(),
        Some(DEFAULT_MPU6050_IMU_DEVICE)
    );
    assert!(report.get("camera_devices").unwrap().is_array());
    assert!(report.get("audio_input_devices").unwrap().is_array());
    assert!(report.get("kinect").unwrap().is_object());
    assert!(report.get("data_dirs_writable").unwrap().is_object());
}

#[test]
fn imu_device_defaults_can_be_overridden_or_disabled() {
    assert_eq!(
        selected_imu_device(None, false),
        Some(DEFAULT_MPU6050_IMU_DEVICE)
    );
    assert_eq!(selected_imu_device(None, true), None);
    assert_eq!(
        selected_imu_device(Some("/dev/i2c-1@0x69"), true),
        Some("/dev/i2c-1@0x69")
    );
    assert_eq!(selected_imu_device(Some("none"), false), None);
}

#[test]
fn serial_auto_selection_keeps_lidar_gps_and_create_separate() {
    let report = serde_json::json!({
        "serial_devices": [
            "/dev/serial/by-id/usb-ROBOTIS_USB2LDS_LDS-01",
            "/dev/serial/by-id/usb-u-blox_AG_-_www.u-blox.com_u-blox_7-if00",
            "/dev/ttyACM0",
            "/dev/ttyUSB0"
        ]
    });

    let lidar = selected_lidar_device(None, false, &report);
    assert_eq!(
        lidar.as_deref(),
        Some("/dev/serial/by-id/usb-ROBOTIS_USB2LDS_LDS-01")
    );

    assert_eq!(
        selected_create_port("auto", &report, lidar.as_deref()),
        Some("/dev/ttyUSB0".to_string())
    );
    assert_eq!(
        selected_gps_device(None, false, &report, Some("/dev/ttyUSB0")),
        Some("/dev/serial/by-id/usb-u-blox_AG_-_www.u-blox.com_u-blox_7-if00".to_string())
    );
    assert_eq!(
        selected_gps_device(Some("/dev/ttyACM1"), false, &report, Some("/dev/ttyUSB0")),
        Some("/dev/ttyACM1".to_string())
    );
    assert_eq!(
        selected_gps_device(Some("none"), false, &report, Some("/dev/ttyUSB0")),
        None
    );
    assert_eq!(selected_lidar_device(Some("none"), false, &report), None);
    assert_eq!(
        selected_lidar_device(Some("/dev/ttyUSB9"), true, &report),
        Some("/dev/ttyUSB9".to_string())
    );
}

#[test]
fn local_cockpit_uses_the_rpi5_brainstem_address_not_a_serial_device() {
    let report = serde_json::json!({
        "serial_devices": ["/dev/ttyUSB0"]
    });
    let address = "127.0.0.1:9876".parse().unwrap();
    assert_eq!(
        selected_cockpit_endpoint(
            CockpitBackendArg::Local,
            "auto",
            "192.168.4.1:80",
            address,
            &report,
            None,
        ),
        Some(address.to_string())
    );
}

#[tokio::test]
async fn possession_mode_never_falls_back_when_brainstem_is_missing() {
    let result = open_robot_cockpit_or_fallback(
        CockpitBackendArg::Uart,
        None,
        RobotMode::Slow,
        None,
        None,
        50,
        500,
    );
    let error = match result {
        Ok(_) => panic!("possession unexpectedly fell back"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("stable brainstem USB CDC"));
}

#[test]
fn possession_reconnect_backoff_is_exponential_and_bounded() {
    assert_eq!(next_reconnect_backoff_ms(250, 5_000), 500);
    assert_eq!(next_reconnect_backoff_ms(4_000, 5_000), 5_000);
    assert_eq!(next_reconnect_backoff_ms(5_000, 5_000), 5_000);
}

struct FreshPacketCockpit {
    status_reads: usize,
    stopped: bool,
    stream_requested: bool,
}

impl Cockpit for FreshPacketCockpit {
    fn execute(
        &mut self,
        request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        match request {
            pete_cockpit::CockpitRequest::Stop => {
                self.stopped = true;
                Ok(pete_cockpit::CockpitResponse::Accepted)
            }
            pete_cockpit::CockpitRequest::StreamSensors {
                enabled: true,
                packet_id: 0,
                ..
            } => {
                assert!(self.stopped, "stream requested before STOP");
                self.stream_requested = true;
                Ok(pete_cockpit::CockpitResponse::Accepted)
            }
            pete_cockpit::CockpitRequest::GetStatus => {
                self.status_reads += 1;
                let fresh = self.stream_requested && self.status_reads >= 3;
                let count = if fresh { 2 } else { 1 };
                let packet_ms = if fresh { 995 } else { 100 };
                Ok(pete_cockpit::CockpitResponse::Status(
                    pete_cockpit::CockpitStatus {
                        raw: serde_json::json!({
                            "uptime_ms": 1_000,
                            "current_runtime_state": "idle",
                            "current_command": "stop",
                            "create_sensors": {
                                "last_packet_id": 0,
                                "complete_packet_count": count,
                                "last_complete_packet_timestamp_ms": packet_ms
                            }
                        })
                        .to_string(),
                    },
                ))
            }
            other => panic!("unexpected readiness request: {other:?}"),
        }
    }

    fn handshake(
        &mut self,
        _hello: HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        Err(CockpitError::Policy("not used by readiness test".into()))
    }

    fn execute_in_session(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ControlLease,
        request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_service_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ServiceLease,
        _request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        Err(CockpitError::Policy("service mode unavailable".into()))
    }
}

#[test]
fn reconnect_readiness_requires_stop_and_new_complete_packet() {
    let mut cockpit = FreshPacketCockpit {
        status_reads: 0,
        stopped: false,
        stream_requested: false,
    };

    establish_create_sensor_stream(&mut cockpit, true).unwrap();

    assert!(cockpit.stopped);
    assert!(cockpit.stream_requested);
    assert!(cockpit.status_reads >= 3);
}

struct DropTrackedCockpit {
    drops: Arc<AtomicUsize>,
}

impl Drop for DropTrackedCockpit {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

impl Cockpit for DropTrackedCockpit {
    fn execute(
        &mut self,
        _request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        Err(CockpitError::Policy("test cockpit is closed".into()))
    }

    fn handshake(
        &mut self,
        _hello: HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        Err(CockpitError::Policy("test cockpit is closed".into()))
    }

    fn execute_in_session(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        Err(CockpitError::Policy("test cockpit is closed".into()))
    }

    fn execute_with_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ControlLease,
        _request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        Err(CockpitError::Policy("test cockpit is closed".into()))
    }

    fn execute_with_service_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ServiceLease,
        _request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        Err(CockpitError::Policy("test cockpit is closed".into()))
    }
}

#[test]
fn possession_reconnect_drops_existing_cockpit_before_opening_replacement() {
    let drops = Arc::new(AtomicUsize::new(0));
    let cockpit: Box<dyn Cockpit + Send> = Box::new(DropTrackedCockpit {
        drops: Arc::clone(&drops),
    });
    let mut safe = SafeCockpit::new(cockpit);

    disconnect_possession_cockpit_for_reconnect(&mut safe);

    assert_eq!(drops.load(Ordering::SeqCst), 1);
    let error = safe.client_mut().stop().unwrap_err();
    assert!(error.to_string().contains("reconnect in progress"));
}

struct BusyShutdownCockpit {
    stop_busy_remaining: usize,
    exorcize_busy_remaining: usize,
    stop_attempts: usize,
    exorcize_attempts: usize,
    stopped: bool,
    exorcized: bool,
}

impl BusyShutdownCockpit {
    fn busy(command_id: u32) -> CockpitError {
        CockpitError::Rejected {
            command_id,
            reason: "busy".into(),
        }
    }
}

impl Cockpit for BusyShutdownCockpit {
    fn exorcize(&mut self) -> pete_cockpit::Result<()> {
        self.exorcize_attempts += 1;
        if self.exorcize_busy_remaining > 0 {
            self.exorcize_busy_remaining -= 1;
            return Err(Self::busy(100 + self.exorcize_attempts as u32));
        }
        self.exorcized = true;
        Ok(())
    }

    fn execute(
        &mut self,
        request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        match request {
            pete_cockpit::CockpitRequest::Stop => {
                self.stop_attempts += 1;
                if self.stop_busy_remaining > 0 {
                    self.stop_busy_remaining -= 1;
                    return Err(Self::busy(self.stop_attempts as u32));
                }
                self.stopped = true;
                Ok(pete_cockpit::CockpitResponse::Accepted)
            }
            other => panic!("unexpected shutdown request: {other:?}"),
        }
    }

    fn handshake(
        &mut self,
        _hello: HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        Err(CockpitError::Policy("not used by shutdown test".into()))
    }

    fn execute_in_session(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ControlLease,
        request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_service_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ServiceLease,
        _request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        Err(CockpitError::Policy("service mode unavailable".into()))
    }
}

#[test]
fn possession_shutdown_retries_plain_busy_stop_and_exorcize() {
    let mut cockpit = BusyShutdownCockpit {
        stop_busy_remaining: 2,
        exorcize_busy_remaining: 1,
        stop_attempts: 0,
        exorcize_attempts: 0,
        stopped: false,
        exorcized: false,
    };

    run_possession_shutdown_with_retry(&mut cockpit, 5, Duration::ZERO).unwrap();

    assert!(cockpit.stopped);
    assert!(cockpit.exorcized);
    assert_eq!(cockpit.stop_attempts, 3);
    assert_eq!(cockpit.exorcize_attempts, 2);
}

#[test]
fn simulated_possession_reconnect_gets_fresh_session_and_lease() {
    let (mut first, _, _) = open_robot_cockpit_or_fallback(
        CockpitBackendArg::Sim,
        Some("mock"),
        RobotMode::Slow,
        None,
        None,
        50,
        500,
    )
    .unwrap();
    let first_snapshot = first.possession_snapshot().unwrap();
    assert!(first_snapshot.lease_remaining_ms > 59_000);
    first
        .cmd_vel(50, 0, 30_000)
        .expect("first lease applies bounded motion");
    drop(first);

    let (mut second, _, _) = open_robot_cockpit_or_fallback(
        CockpitBackendArg::Sim,
        Some("mock"),
        RobotMode::Slow,
        None,
        None,
        50,
        500,
    )
    .unwrap();
    let second_snapshot = second.possession_snapshot().unwrap();
    assert_ne!(first_snapshot.session_id, second_snapshot.session_id);
    assert_ne!(first_snapshot.lease_id, second_snapshot.lease_id);
    assert!(second_snapshot.possessed);
    assert_eq!(
        second.get_status().unwrap().summary().active_motion,
        Some(false)
    );
}

#[test]
fn sensor_only_live_publish_does_not_refresh_body_timestamp() {
    let live_state = LiveViewState::new().with_real_slow_hardware_control();
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.last_update_ms = 1_234;
    live_state.update(snapshot);

    publish_live_sensor_only_snapshot(
        &live_state,
        &SensePacket::EyeFrame(EyeFrame {
            captured_at_ms: 9_999,
            width: 1,
            height: 1,
            format: EyeFrameFormat::Rgb8,
            bytes: vec![0, 0, 0],
            source: Some("test".to_string()),
        }),
    );

    let latest = live_state.latest().unwrap();
    assert_eq!(latest.body.last_update_ms, 1_234);
    assert_eq!(
        latest.eye_frame.as_ref().map(|frame| frame.captured_at_ms),
        Some(9_999)
    );
}

#[test]
fn missing_streams_generate_warnings() {
    let mut counts = StreamCounts::default();
    counts.observe(&WorldSnapshot::default());
    let streams = counts.streams();
    assert!(streams.present.contains(&"body".to_string()));
    assert!(streams.missing.contains(&"rgb".to_string()));
    assert!(counts
        .warnings()
        .iter()
        .any(|warning| warning == "rgb stream missing"));
}

#[tokio::test]
async fn inspect_capture_reads_tiny_fake_capture() {
    let temp_dir = temp_path("pete_inspect_capture");
    let mut writer = CaptureWriter::create(&temp_dir, CaptureSource::RealRobot, Some(100))
        .await
        .unwrap();
    let mut snapshot = WorldSnapshot::default();
    snapshot.eye.frames.push(vec![0.1, 0.2]);
    writer
        .append_snapshot(100, snapshot, Vec::new())
        .await
        .unwrap();
    writer.finish().await.unwrap();

    let report = inspect_capture_report(&temp_dir).await.unwrap();
    assert_eq!(report.frame_count, 1);
    assert!(report.streams_present.contains(&"rgb".to_string()));
    assert!(report.streams_missing.contains(&"audio".to_string()));
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn pose_graph_report_reads_capture_frames_and_gates_loop_candidates() {
    let temp_dir = temp_path("pete_pose_graph_capture");
    let mut writer = CaptureWriter::create(&temp_dir, CaptureSource::Sim, Some(100))
        .await
        .unwrap();
    let mut first = WorldSnapshot::default();
    first.body.odometry.x_m = 0.0;
    first.eye.scene_vectors.push(
        VectorArtifact::new(SCENE_VECTOR_COLLECTION, "scene-first", vec![1.0, 0.0])
            .with_source_frame_id("capture-frame-0"),
    );
    writer
        .append_snapshot(100, first, Vec::new())
        .await
        .unwrap();

    let mut second = WorldSnapshot::default();
    second.body.odometry.x_m = 1.0;
    second.eye.scene_vectors.push(VectorArtifact::new(
        SCENE_VECTOR_COLLECTION,
        "scene-query-strong",
        vec![1.0, 0.0],
    ));
    second.eye.scene_vectors.push(VectorArtifact::new(
        SCENE_VECTOR_COLLECTION,
        "scene-query-weak",
        vec![0.0, 1.0],
    ));
    writer
        .append_snapshot(200, second, Vec::new())
        .await
        .unwrap();
    writer.finish().await.unwrap();

    let args = PoseGraphReportArgs {
        ledger: "unused-when-capture-is-set".to_string(),
        capture: Some(temp_dir.to_string_lossy().to_string()),
        out: temp_dir
            .join("pose-graph.json")
            .to_string_lossy()
            .to_string(),
        min_node_distance_m: 0.5,
        min_node_degrees: 15.0,
        max_ticks_between_nodes: 10,
        min_loop_confidence: 0.55,
    };
    let report = generate_pose_graph_report(&args).await.unwrap();

    assert_eq!(report.nodes, 2);
    assert_eq!(report.odometry_edges, 1);
    assert_eq!(report.loop_candidate_edges, 2);
    assert_eq!(report.active_loop_candidate_edges, 1);
    assert_eq!(report.rejected_loop_candidates, 1);
    assert_eq!(
        report.rejected_candidates[0].reason,
        "confidence 0.000 below gate 0.550"
    );
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn pose_graph_report_uses_ledger_place_recognition_latents() {
    let temp_dir = temp_path("pete_pose_graph_ledger");
    let ledger = JsonlLedger::new(&temp_dir);

    let first = pose_graph_test_frame(100, 0.0, vec![1.0, 0.0, 0.0]);
    let first_experience_id = first.experiences[0].id.to_string();
    let first_frame_id = first.id.to_string();
    let second = pose_graph_test_frame(200, 1.0, vec![0.99, 0.01, 0.0]);
    let second_experience_id = second.experiences[0].id.to_string();
    ledger.append(&first).await.unwrap();
    ledger.append(&second).await.unwrap();

    let args = PoseGraphReportArgs {
        ledger: temp_dir.to_string_lossy().to_string(),
        capture: None,
        out: temp_dir
            .join("pose-graph.json")
            .to_string_lossy()
            .to_string(),
        min_node_distance_m: 0.5,
        min_node_degrees: 15.0,
        max_ticks_between_nodes: 10,
        min_loop_confidence: 0.55,
    };
    let report = generate_pose_graph_report(&args).await.unwrap();

    assert_eq!(report.nodes, 2);
    assert_eq!(report.odometry_edges, 1);
    assert_eq!(report.loop_candidate_edges, 1);
    assert_eq!(report.active_loop_candidate_edges, 1);
    assert_eq!(report.rejected_loop_candidates, 0);
    let loop_edge = report
        .graph
        .edges
        .iter()
        .find(|edge| {
            matches!(
                edge.source,
                pete_map::PoseEdgeSource::LoopClosureCandidate { .. }
            )
        })
        .expect("loop edge");
    match &loop_edge.source {
        pete_map::PoseEdgeSource::LoopClosureCandidate {
            target_frame_id,
            source_frame_id,
            source_experience_id,
            source_instant_frame_id,
            query_experience_id,
            ..
        } => {
            assert_eq!(target_frame_id.as_deref(), Some(first_frame_id.as_str()));
            assert_eq!(
                source_frame_id.as_deref(),
                Some(second.id.to_string().as_str())
            );
            assert_eq!(
                source_experience_id.as_deref(),
                Some(first_experience_id.as_str())
            );
            assert_eq!(
                source_instant_frame_id.as_deref(),
                Some(first_frame_id.as_str())
            );
            assert_eq!(
                query_experience_id.as_deref(),
                Some(second_experience_id.as_str())
            );
        }
        pete_map::PoseEdgeSource::Odometry => panic!("expected loop edge"),
        pete_map::PoseEdgeSource::ScanMatch { .. } => panic!("expected loop edge"),
    }
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn representation_report_writes_json_from_capture_fixture() {
    let temp_dir = temp_path("pete_representation_report_capture");
    let mut writer = CaptureWriter::create(&temp_dir, CaptureSource::Sim, Some(100))
        .await
        .unwrap();

    let mut first = WorldSnapshot::default();
    first.body.odometry.x_m = 0.0;
    first.range.nearest_m = Some(0.5);
    first.eye.scene_vectors.push(
        VectorArtifact::new(SCENE_VECTOR_COLLECTION, "scene-a", vec![1.0, 0.0])
            .with_source_frame_id("capture-frame-0"),
    );
    writer
        .append_snapshot(100, first, Vec::new())
        .await
        .unwrap();

    let mut second = WorldSnapshot::default();
    second.body.odometry.x_m = 0.8;
    second.range.nearest_m = Some(0.45);
    second.eye.scene_vectors.push(VectorArtifact::new(
        SCENE_VECTOR_COLLECTION,
        "scene-b",
        vec![0.98, 0.02],
    ));
    writer
        .append_snapshot(200, second, Vec::new())
        .await
        .unwrap();
    writer.finish().await.unwrap();

    let out = temp_dir.join("reports/representation/report.json");
    run_representation_report(RepresentationReportArgs {
        ledger: "unused-when-capture-is-set".to_string(),
        capture: Some(temp_dir.to_string_lossy().to_string()),
        out: out.to_string_lossy().to_string(),
    })
    .await
    .unwrap();

    let value: serde_json::Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(value["frame_count"], 2);
    assert_eq!(value["input"]["source_type"], "capture");
    assert!(value["entity_memory"].is_object());
    assert!(value["map"].is_object());
    assert!(value["pose_graph"].is_object());
    assert!(value["place_recognition"].is_object());
    let _ = fs::remove_dir_all(&temp_dir);
}

fn pose_graph_test_frame(t_ms: u64, x_m: f32, latent_vector: Vec<f32>) -> ExperienceFrame {
    let mut now = Now::blank(t_ms, BodySense::default());
    now.body.odometry.x_m = x_m;
    let sensation_id = uuid::Uuid::new_v4();
    let experience = Experience::new(
        "test.place",
        format!("test place at {x_m:.1}m"),
        Vec::new(),
        vec![sensation_id],
        t_ms,
        t_ms,
    );
    ExperienceFrame {
        id: uuid::Uuid::new_v4(),
        t_ms,
        now,
        sensations: Vec::new(),
        impressions: Vec::new(),
        experiences: vec![experience],
        z: Some(ExperienceLatent {
            t_ms,
            z: latent_vector,
            confidence: 1.0,
            ..ExperienceLatent::default()
        }),
        chosen_action: None,
        conscious_command: None,
        reign_input: None,
        reign_outcome: None,
        predicted_futures: Vec::new(),
        behavior_runs: Vec::new(),
        actual_next: None,
        reward: Reward::default(),
        surprise: SurpriseSense::default(),
        memory_recall: Vec::new(),
        recollections: Vec::new(),
        llm_teaching: Vec::new(),
        counterfactuals: Vec::new(),
        notes: Vec::new(),
    }
}

#[tokio::test]
#[ignore = "slow capture-real mock path can stall workspace test runs"]
async fn capture_real_mock_writes_manifest_and_frames() {
    let temp_dir = temp_path("pete_capture_real_mock");
    let args = CaptureRealArgs {
        duration_seconds: 1,
        out: temp_dir.to_string_lossy().to_string(),
        ledger: None,
        tick_ms: 1000,
        cockpit: CockpitBackendArg::Sim,
        create_port: "mock".to_string(),
        brainstem_host: "192.168.4.1:80".to_string(),
        brainstem_local: "127.0.0.1:8787".parse().unwrap(),
        create_baud: 57_600,
        camera: None,
        kinect_depth: false,
        kinect_index: 0,
        kinect_rgb_target_luma: 0.32,
        kinect_rgb_auto_gain_max: 3.0,
        kinect_rgb_gain: 1.0,
        kinect_rgb_gamma: 0.80,
        kinect_rgb_brightness: 0.0,
        kinect_rgb_raw: false,
        mic: None,
        imu: None,
        gps: None,
        lidar: None,
        lidar_yaw_deg: 0.0,
        lidar_pitch_deg: 0.0,
        lidar_roll_deg: 0.0,
        lidar_height_m: 0.0,
        lidar_forward_m: 0.0,
        lidar_left_m: 0.0,
        mock: true,
        export_rgb: false,
        export_depth: false,
        export_audio: false,
        export_pointcloud: false,
        pointcloud_stride: 4,
        llm: LlmArgs::default(),
    };

    capture_real(args).await.unwrap();
    assert!(temp_dir.join("manifest.json").exists());
    assert!(temp_dir.join("frames.jsonl").exists());
    let report = inspect_capture_report(&temp_dir).await.unwrap();
    assert_eq!(report.frame_count, 1);
    assert!(report.streams_present.contains(&"body".to_string()));
    assert!(report.streams_present.contains(&"audio".to_string()));
    assert!(report.streams_present.contains(&"depth".to_string()));
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
#[ignore = "slow capture-real mock path can stall workspace test runs"]
async fn capture_real_mock_exports_assets_and_pointclouds() {
    let temp_dir = temp_path("pete_capture_real_mock_assets");
    let args = CaptureRealArgs {
        duration_seconds: 1,
        out: temp_dir.to_string_lossy().to_string(),
        ledger: None,
        tick_ms: 1000,
        cockpit: CockpitBackendArg::Sim,
        create_port: "mock".to_string(),
        brainstem_host: "192.168.4.1:80".to_string(),
        brainstem_local: "127.0.0.1:8787".parse().unwrap(),
        create_baud: 57_600,
        camera: None,
        kinect_depth: false,
        kinect_index: 0,
        kinect_rgb_target_luma: 0.32,
        kinect_rgb_auto_gain_max: 3.0,
        kinect_rgb_gain: 1.0,
        kinect_rgb_gamma: 0.80,
        kinect_rgb_brightness: 0.0,
        kinect_rgb_raw: false,
        mic: None,
        imu: None,
        gps: None,
        lidar: None,
        lidar_yaw_deg: 0.0,
        lidar_pitch_deg: 0.0,
        lidar_roll_deg: 0.0,
        lidar_height_m: 0.0,
        lidar_forward_m: 0.0,
        lidar_left_m: 0.0,
        mock: true,
        export_rgb: true,
        export_depth: true,
        export_audio: true,
        export_pointcloud: false,
        pointcloud_stride: 4,
        llm: LlmArgs::default(),
    };

    capture_real(args).await.unwrap();
    capture_assets(CaptureAssetsArgs {
        capture: temp_dir.to_string_lossy().to_string(),
        pointcloud: true,
        world_pointcloud: true,
        stride: 1,
        max_depth_m: 8.0,
    })
    .await
    .unwrap();

    let report = inspect_capture_report(&temp_dir).await.unwrap();
    assert_eq!(
        report.asset_counts,
        vec![
            ("rgb".to_string(), 1),
            ("depth".to_string(), 1),
            ("audio".to_string(), 1),
            ("pointcloud".to_string(), 2)
        ]
    );
    assert!(report
        .asset_details
        .iter()
        .any(|detail| detail.contains("rgb metadata: 2x2")));
    assert!(report
        .warnings
        .iter()
        .any(|warning| warning.contains("uncalibrated point cloud")));
    let world_ply = temp_dir.join("assets/pointcloud/world-accumulated.ply");
    let world_ply_text = fs::read_to_string(&world_ply).unwrap();
    assert!(world_ply_text.contains("property float confidence"));
    assert!(world_ply_text.contains("property uchar stable"));
    let replay = replay_capture(ReplayCaptureArgs {
        capture: temp_dir.to_string_lossy().to_string(),
        ledger: temp_dir.join("ledger").to_string_lossy().to_string(),
        llm: LlmArgs::default(),
    })
    .await;
    assert!(replay.is_ok());
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn eval_scenario_empty_room_smoke_runs() {
    let args = eval_args(ScenarioArg::EmptyRoom, 1, 3, None);
    run_eval_scenario(args).await.unwrap();
}

#[tokio::test]
async fn eval_scenario_obstacle_writes_report() {
    let temp_dir = temp_path("pete_eval_scenario_obstacle");
    let out = temp_dir.join("obstacle.json");
    let mut args = eval_args(
        ScenarioArg::ObstacleAvoidance,
        3,
        5,
        Some(out.to_string_lossy().to_string()),
    );
    args.memory_report = true;
    run_eval_scenario(args).await.unwrap();
    let report: serde_json::Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(report["scenario"], "obstacle-avoidance");
    assert_eq!(report["action_selector_mode"], "baseline");
    assert_eq!(report["episodes_detail"].as_array().unwrap().len(), 3);
    assert!(report["memory"]["places_visited"].as_u64().unwrap_or(0) > 0);
    assert!(
        report["episodes_detail"][0]["memory"]["danger_memory_ticks"]
            .as_u64()
            .unwrap_or(0)
            > 0
    );
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn eval_scenario_model_assisted_empty_room_runs_and_reports_stats() {
    let temp_dir = temp_path("pete_eval_scenario_model_assisted_empty");
    let out = temp_dir.join("empty-model-assisted.json");
    let mut args = eval_args(
        ScenarioArg::EmptyRoom,
        1,
        3,
        Some(out.to_string_lossy().to_string()),
    );
    args.action_selector = CliActionSelectorMode::ModelAssisted;
    run_eval_scenario(args).await.unwrap();
    let report: serde_json::Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(report["action_selector_mode"], "model-assisted");
    assert_eq!(report["summary"]["model_assisted_decisions"], 3);
    assert!(report["summary"]["mean_candidate_score"].is_number());
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn eval_scenario_model_assisted_charger_seeking_runs() {
    let mut args = eval_args(ScenarioArg::ChargerSeeking, 1, 3, None);
    args.action_selector = CliActionSelectorMode::ModelAssisted;
    run_eval_scenario(args).await.unwrap();
}

#[tokio::test]
async fn eval_scenario_optional_ledger_writes_transitions() {
    let temp_dir = temp_path("pete_eval_scenario_ledger");
    let ledger_dir = temp_dir.join("ledger");
    let out = temp_dir.join("empty.json");
    let mut args = eval_args(
        ScenarioArg::EmptyRoom,
        1,
        4,
        Some(out.to_string_lossy().to_string()),
    );
    args.ledger = Some(ledger_dir.to_string_lossy().to_string());
    run_eval_scenario(args).await.unwrap();
    let transitions = JsonlLedger::new(&ledger_dir).transitions().await.unwrap();
    assert!(!transitions.is_empty());
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn sim_curriculum_writes_one_capture_per_episode() {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let temp_dir = std::env::temp_dir().join(format!("pete_curriculum_test_{now_ms}"));
    let ledger_dir = temp_dir.join("ledger");
    let capture_root = temp_dir.join("captures");

    let args = SimCurriculumArgs {
        scenario: ScenarioArg::PersonSpeakerRoom,
        episodes: 2,
        steps: 3,
        seed: 7,
        out: ledger_dir.to_str().unwrap().to_string(),
        capture_root: Some(capture_root.to_str().unwrap().to_string()),
        tick_ms: 100,
        validation_ratio: 0.25,
        test_ratio: 0.25,
        llm: LlmArgs::default(),
    };

    run_sim_curriculum(args).await.unwrap();

    let manifest_path = ledger_dir.join("manifest.json");
    assert!(manifest_path.exists());
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["scenario"], "person-speaker-room");
    assert_eq!(manifest["episodes"], 2);
    assert_eq!(manifest["splits"]["train"], 0);
    assert_eq!(manifest["splits"]["validation"], 1);
    assert_eq!(manifest["splits"]["test"], 1);
    assert_eq!(
        manifest["episodes_detail"][0]["capture"],
        capture_root
            .join("episode-000")
            .to_string_lossy()
            .to_string()
    );
    assert!(capture_root
        .join("episode-000")
        .join("manifest.json")
        .exists());
    assert!(capture_root
        .join("episode-001")
        .join("manifest.json")
        .exists());
    let transitions = JsonlLedger::new(&ledger_dir).transitions().await.unwrap();
    assert!(!transitions.is_empty());

    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn test_evaluate_behavior_command_writes_json_to_out() {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let temp_dir = std::env::temp_dir().join(format!("pete_eval_test_{}", now_ms));
    let ledger_dir = temp_dir.join("ledger");
    let session_dir = ledger_dir.join("2026-06-24");
    fs::create_dir_all(&session_dir).unwrap();

    let checkpoint_dir = temp_dir.join("checkpoint");
    fs::create_dir_all(&checkpoint_dir).unwrap();

    // Write 5 mock transitions to have enough data for training and validation splits
    let mut transitions = Vec::new();
    for i in 0..5 {
        let transition = ExperienceTransition {
            id: uuid::Uuid::new_v4(),
            before_frame_id: uuid::Uuid::new_v4(),
            before: Now::blank(100 + i * 100, BodySense::default()),
            before_z: ExperienceLatent {
                t_ms: 100 + i * 100,
                z: vec![0.1; 4],
                ..ExperienceLatent::default()
            },
            action: Some(ActionPrimitive::Stop),
            predicted_futures: Vec::new(),
            after: Now::blank(200 + i * 100, BodySense::default()),
            after_z: ExperienceLatent {
                t_ms: 200 + i * 100,
                z: vec![0.2; 4],
                ..ExperienceLatent::default()
            },
            reward: Reward { value: 0.0 },
            surprise: SurpriseSense::default(),
            created_at_ms: 200 + i * 100,
        };
        transitions.push(transition);
    }

    let transitions_file = session_dir.join("transitions.jsonl");
    let mut content = String::new();
    for t in &transitions {
        content.push_str(&serde_json::to_string(t).unwrap());
        content.push('\n');
    }
    fs::write(&transitions_file, content).unwrap();

    // Train first to create the checkpoint and metadata
    pete_training::train_behavior(pete_training::TrainBehaviorRequest {
        behavior: pete_training::TrainableBehavior::Danger,
        ledger_path: ledger_dir.clone(),
        checkpoint_path: checkpoint_dir.clone(),
        epochs: 1,
        validation_split: 0.2,
        seed: 42,
    })
    .await
    .unwrap();

    // Prepare output path
    let out_json_path = temp_dir.join("report.json");

    let args = EvaluateBehaviorArgs {
        behavior: "danger".to_string(),
        ledger: ledger_dir.to_str().unwrap().to_string(),
        checkpoint: Some(checkpoint_dir.to_str().unwrap().to_string()),
        max_samples: None,
        out: Some(out_json_path.to_str().unwrap().to_string()),
    };

    let cmd = EvaluateCommand {
        model: EvaluateModel::Behavior(args),
    };

    let res = run_evaluate(cmd).await;
    assert!(res.is_ok(), "run_evaluate failed: {:?}", res.err());

    // Verify report file exists and has correct behavior name
    assert!(out_json_path.exists());
    let report_content = fs::read_to_string(&out_json_path).unwrap();
    let report: serde_json::Value = serde_json::from_str(&report_content).unwrap();
    assert_eq!(report["behavior"], "danger");

    // Clean up
    let _ = fs::remove_dir_all(&temp_dir);
}
