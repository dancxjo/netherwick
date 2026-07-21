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
        hardcoded_inference_us: Some(1),
        model_inference_us: None,
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
