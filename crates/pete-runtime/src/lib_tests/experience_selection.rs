#[tokio::test]
async fn tick_adds_combobulated_experience() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    );
    let mut now = Now::blank(100, BodySense::default());
    now.ear.transcript = Some("hello world".to_string());

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    assert!(tick
        .frame
        .experiences
        .iter()
        .any(|experience| experience.text.contains("hello world")));
}

#[tokio::test]
async fn tick_persists_recalled_experiences_as_memory_sensations() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-memory-recall-sensations-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    );
    let mut first = Now::blank(100, BodySense::default());
    first.ear.transcript = Some("charger alcove".to_string());
    runtime
        .tick(first, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    let mut second = Now::blank(200, BodySense::default());
    second.ear.transcript = Some("charger alcove".to_string());
    let tick = runtime
        .tick(second, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    let recall_sensation = tick
        .frame
        .sensations
        .iter()
        .find(|sensation| {
            sensation.modality == Modality::Memory
                && sensation.payload_kind == SensationPayloadKind::MemoryRecall
                && sensation.kind == "memory.recall.experience"
        })
        .expect("memory recall sensation");
    assert!(recall_sensation
        .payload
        .get("original_frame_id")
        .and_then(Value::as_str)
        .is_some());
    assert!(tick.frame.impressions.iter().any(|impression| {
        impression.sensation_id == Some(recall_sensation.id)
            && impression.text.starts_with("I remember")
    }));
    let context = tick.frame.embodied_context();
    assert!(context.sensations.iter().any(|sensation| {
        sensation.id == recall_sensation.id
            && sensation.modality == Modality::Memory
            && sensation.payload_kind == SensationPayloadKind::MemoryRecall
    }));
}

#[tokio::test]
async fn tick_feeds_memory_loop_candidates_into_live_map() {
    let root = test_ledger_root("runtime-live-loop-closure");
    let ledger = JsonlLedger::new(&root);
    let config = MapConfig {
        resolution_m: 0.25,
        pose_graph_min_node_distance_m: 0.01,
        pose_graph_max_ticks_between_nodes: 1,
        ..MapConfig::default()
    };
    let mut runtime = test_runtime(ledger, FixedConductor::new(ActionPrimitive::Stop))
        .with_local_map(LocalMap::new(config));

    for step in 0..5 {
        runtime
            .tick(
                mapped_scene_now(100 + step * 100, 0.0, &format!("seed-{step}")),
                ExperienceLatent::default(),
                Vec::new(),
            )
            .await
            .unwrap();
    }

    let tick = runtime
        .tick(
            mapped_scene_now(700, 0.05, "return"),
            ExperienceLatent::default(),
            Vec::new(),
        )
        .await
        .unwrap();
    let frame_id = tick.frame.id.to_string();

    assert_eq!(
        tick.frame
            .now
            .extensions
            .get("frame_id")
            .and_then(Value::as_str),
        Some(frame_id.as_str())
    );
    let summary = runtime.local_map.summary();
    assert!(
        summary.loop_closures_accepted > 0,
        "expected live map to accept a memory loop closure, got {summary:?}"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn analog_cliff_risk_alone_does_not_say_floor_falls_away() {
    let mut now = Now::blank(100, BodySense::default());
    now.body.cliff_sensors.front_left = 0.96;
    now.body.cliff_sensors.front_right = 0.82;

    let (_sensations, impressions) = derive_direct_impressions_from_now(&now);
    let body_text = impressions
        .iter()
        .find(|impression| impression.kind == "body.state.impression")
        .map(|impression| impression.text.as_str())
        .unwrap();

    assert!(!body_text.contains("floor feels like it falls away near me"));
    assert!(body_text.contains("cliff IR signal is uncertain"));
}

#[test]
fn cockpit_charging_indicator_sets_body_charging() {
    let status = StatusSummary::from_raw(
        r#"{"create_sensors":{"charging_state":0,"charging_indicator":"on","charge_mah":1300,"capacity_mah":2600}}"#,
    );

    let body = body_sense_from_cockpit_status(status, 42);

    assert!(body.charging);
    assert_eq!(body.battery_level, 0.5);
    assert_eq!(body.last_update_ms, 42);
}

#[test]
fn real_slow_blocks_charging_body() {
    let mut body = BodySense::default();
    body.charging = true;

    assert_eq!(
        real_slow_body_block_reason(&body).as_deref(),
        Some("charging active")
    );
}

#[test]
fn direct_now_impressions_are_first_person_present() {
    let mut now = Now::blank(100, BodySense::default());
    now.ear.transcript = Some("hello world".to_string());
    now.body.flags.cliff_front_left = true;
    now.body.cliff_sensors.front_left = 0.8;
    now.extensions.insert(
        "test.context".to_string(),
        serde_json::json!({ "ok": true }),
    );

    let (_sensations, impressions) = derive_direct_impressions_from_now(&now);
    let body_text = impressions
        .iter()
        .find(|impression| impression.kind == "body.state.impression")
        .map(|impression| impression.text.as_str())
        .unwrap();
    assert!(body_text.contains("floor feels like it falls away near me"));
    assert!(!body_text.contains("cliffs L/FL/FR/R"));
    assert!(!body_text.contains("cliff levels"));

    assert!(!impressions.is_empty());
    for impression in impressions {
        assert!(
            impression.text.starts_with("I ")
                || impression.text.starts_with("I'm ")
                || impression.text.starts_with("My "),
            "impression should manifest embodiment in first person: {}",
            impression.text
        );
        assert!(
            impression.text.contains("confident")
                || impression.text.contains("pretty sure")
                || impression.text.contains("I think")
                || impression.text.contains("may have")
                || impression.text.contains("not sure"),
            "impression should express confidence in natural language: {}",
            impression.text
        );
        assert_eq!(
            impression
                .payload
                .get("generator")
                .and_then(|value| value.as_str()),
            Some("mechanical")
        );
    }
}

#[test]
fn surface_scene_graph_becomes_spatial_impression() {
    let mut now = Now::blank(100, BodySense::default());
    now.extensions.insert(
        "surface.scene_graph".to_string(),
        serde_json::json!({
            "floor": {"confidence": 0.82},
            "surfaces": [{"id": "floor"}, {"id": "wall_1"}],
            "clusters": [{"id": "cluster_1"}],
            "navigation": {
                "front_clear_m": 0.6,
                "left_clear_m": 1.4,
                "right_clear_m": 0.3
            }
        }),
    );

    let (_sensations, impressions) = derive_direct_impressions_from_now(&now);
    let surface_text = impressions
        .iter()
        .find(|impression| impression.kind == "surface.scene_graph.impression")
        .map(|impression| impression.text.as_str())
        .unwrap();

    assert!(surface_text.contains("persistent geometry"));
    assert!(surface_text.contains("2 stable surfaces"));
    assert!(surface_text.contains("1 leftover clusters"));
    assert!(surface_text.contains("front 0.60m"));
}

#[test]
fn asr_impressions_phrase_partial_and_final_confidence_naturally() {
    let mut partial = Now::blank(100, BodySense::default());
    partial.ear.asr = pete_now::AsrSense {
        transcript: Some("come over here".to_string()),
        is_final: false,
        confidence: 0.52,
        ..pete_now::AsrSense::default()
    };
    let (_sensations, partial_impressions) = derive_direct_impressions_from_now(&partial);
    let partial_text = partial_impressions
        .iter()
        .find(|impression| impression.kind == "audio.transcript.impression")
        .map(|impression| impression.text.as_str())
        .unwrap();
    assert_eq!(partial_text, "I think I heard \"come over here\".");

    let mut final_now = Now::blank(100, BodySense::default());
    final_now.ear.asr = pete_now::AsrSense {
        transcript: Some("come over here".to_string()),
        is_final: true,
        confidence: 0.93,
        ..pete_now::AsrSense::default()
    };
    let (_sensations, final_impressions) = derive_direct_impressions_from_now(&final_now);
    let final_text = final_impressions
        .iter()
        .find(|impression| impression.kind == "audio.transcript.impression")
        .map(|impression| impression.text.as_str())
        .unwrap();
    assert_eq!(
        final_text,
        "I'm confident I finally heard \"come over here\"."
    );
}

#[test]
fn asr_possible_and_committed_speech_become_direct_impressions() {
    let mut now = Now::blank(100, BodySense::default());
    now.ear.asr = pete_now::AsrSense {
        transcript: Some("open the door".to_string()),
        possible_transcript: Some("open the".to_string()),
        committed_transcript: Some("open the door".to_string()),
        is_final: true,
        confidence: 0.72,
        ..pete_now::AsrSense::default()
    };

    let (sensations, impressions) = derive_direct_impressions_from_now(&now);

    assert!(sensations
        .iter()
        .any(|sensation| sensation.kind == "audio.possible_speech"));
    assert!(sensations
        .iter()
        .any(|sensation| sensation.kind == "audio.committed_speech"));
    assert!(impressions.iter().any(|impression| {
        impression.kind == "audio.possible_speech.impression"
            && impression.text.contains("possible speech")
            && impression.text.contains("open the")
    }));
    assert!(impressions.iter().any(|impression| {
        impression.kind == "audio.committed_speech.impression"
            && impression.text.contains("commit")
            && impression.text.contains("open the door")
    }));
}

#[test]
fn model_assisted_safety_override_beats_high_score_candidate() {
    let mut body = BodySense::default();
    body.flags.wheel_drop = true;
    let now = Now::blank(100, body);
    let baseline = ActionPrimitive::Go {
        intensity: 0.15,
        duration_ms: 1_000,
    };
    let decision = select_action_from_scores(
        ActionSelectorMode::ModelAssisted,
        &now,
        baseline,
        vec![ActionSelectionCandidateScore {
            action: ActionPrimitive::Go {
                intensity: 0.15,
                duration_ms: 1_000,
            },
            score: 10.0,
            ..ActionSelectionCandidateScore::default()
        }],
    );

    assert_eq!(decision.selected_action, Some(ActionPrimitive::Stop));
    assert!(decision.safety_overrode);
}

#[test]
fn model_assisted_does_not_yield_to_close_range_alone() {
    let body = BodySense::default();
    let mut now = Now::blank(100, body);
    now.range.nearest_m = Some(0.12);
    let baseline = ActionPrimitive::Go {
        intensity: -0.18,
        duration_ms: 300,
    };
    let decision = select_action_from_scores(
        ActionSelectorMode::ModelAssisted,
        &now,
        baseline.clone(),
        vec![ActionSelectionCandidateScore {
            action: ActionPrimitive::Turn {
                direction: TurnDir::Right,
                intensity: 0.25,
                duration_ms: 750,
            },
            score: 10.0,
            ..ActionSelectionCandidateScore::default()
        }],
    );

    assert_ne!(decision.selected_action, Some(baseline));
    assert!(!decision.safety_overrode);
    assert!(decision.fallback_warnings.is_empty());
}

#[test]
fn close_range_scores_baseline_recovery_candidate() {
    let body = BodySense::default();
    let mut now = Now::blank(100, body);
    now.range.nearest_m = Some(0.12);
    let baseline = ActionPrimitive::Turn {
        direction: TurnDir::Left,
        intensity: 0.75,
        duration_ms: 500,
    };
    let model_signals = CandidateModelSignals {
        danger: Some(DangerOutput {
            confidence: 1.0,
            ..Default::default()
        }),
        charge: Some(ChargeOutput {
            confidence: 1.0,
            ..Default::default()
        }),
        action_value: Some(ActionValueOutput {
            confidence: 1.0,
            ..Default::default()
        }),
    };

    let recovery = score_action_candidate(&now, &baseline, model_signals, Some(&baseline));
    let default_turn = score_action_candidate(
        &now,
        &ActionPrimitive::Turn {
            direction: TurnDir::Right,
            intensity: 0.25,
            duration_ms: 750,
        },
        model_signals,
        Some(&baseline),
    );

    assert!(recovery.score > default_turn.score);
    assert!(!recovery.fallback_used);
}

#[test]
fn model_assisted_scores_active_stuck_recovery_candidate() {
    let body = BodySense::default();
    let mut now = Now::blank(100, body);
    now.extensions.insert(
        "sim.stuck".to_string(),
        serde_json::json!({
            "schema_version": 1,
            "values": [1.0, 0.0, 6.0, 100.0, 1.0, -1.0, 0.0, 0.0]
        }),
    );
    let baseline = ActionPrimitive::Go {
        intensity: -0.18,
        duration_ms: 300,
    };
    let model_signals = CandidateModelSignals {
        danger: Some(DangerOutput {
            confidence: 1.0,
            ..Default::default()
        }),
        charge: Some(ChargeOutput {
            confidence: 1.0,
            ..Default::default()
        }),
        action_value: Some(ActionValueOutput {
            confidence: 1.0,
            ..Default::default()
        }),
    };
    let recovery = score_action_candidate(&now, &baseline, model_signals, Some(&baseline));
    let turn = score_action_candidate(
        &now,
        &ActionPrimitive::Turn {
            direction: TurnDir::Right,
            intensity: 0.25,
            duration_ms: 750,
        },
        model_signals,
        Some(&baseline),
    );
    let decision = select_action_from_scores(
        ActionSelectorMode::ModelAssisted,
        &now,
        baseline.clone(),
        vec![turn, recovery],
    );

    assert_eq!(decision.selected_action, Some(baseline));
    assert!(decision.selected_score.unwrap_or_default() > 0.0);
    assert!(!decision.safety_overrode);
    assert!(decision.fallback_warnings.is_empty());
}

#[test]
fn sim_stuck_extension_sets_recent_trap_memory_hints() {
    let mut now = Now::blank(100, BodySense::default());
    now.extensions.insert(
        "sim.stuck".to_string(),
        serde_json::json!({
            "schema_version": 1,
            "values": [1.0, 1.0, 6.0, 600.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.0]
        }),
    );

    apply_recent_trap_memory_hints(&mut now);

    assert!(now.memory.recent_trap_confidence >= 0.6);
    assert!(now.memory.recent_trap_direction_rad.unwrap() < 0.0);
}

#[test]
fn scoring_prefers_charger_when_charge_value_is_high() {
    let now = Now::blank(100, BodySense::default());
    let stop = score_action_candidate(
        &now,
        &ActionPrimitive::Stop,
        CandidateModelSignals::default(),
        None,
    );
    let charger = score_action_candidate(
        &now,
        &ActionPrimitive::Approach {
            target: ApproachTarget::Charger,
        },
        CandidateModelSignals {
            charge: Some(ChargeOutput {
                charge_probability: 0.8,
                expected_battery_delta: 0.1,
                dock_likelihood: 0.7,
                confidence: 1.0,
            }),
            ..CandidateModelSignals::default()
        },
        None,
    );

    assert!(charger.score > stop.score);
}

#[test]
fn charger_approach_is_a_default_action_value_candidate() {
    let candidates = action_value_candidate_actions(&[], None, &LlmTickResult::default());

    assert!(candidates.contains(&ActionPrimitive::Approach {
        target: ApproachTarget::Charger
    }));
}

#[test]
fn scoring_prefers_approach_over_dock_when_charger_visible_but_not_contacted() {
    let mut now = Now::blank(100, BodySense::default());
    now.body.battery_level = 0.15;
    now.memory.place_charge_value = 0.7;
    now.extensions.insert(
        "sim.world".to_string(),
        serde_json::json!({
            "schema_version": 1,
            "values": [4.0, 4.0, 1.0, 0.35, 0.65]
        }),
    );
    let signals = CandidateModelSignals {
        charge: Some(ChargeOutput {
            charge_probability: 0.85,
            expected_battery_delta: 0.02,
            dock_likelihood: 0.35,
            confidence: 1.0,
        }),
        action_value: Some(ActionValueOutput {
            value: 0.1,
            confidence: 1.0,
        }),
        ..CandidateModelSignals::default()
    };

    let approach = score_action_candidate(
        &now,
        &ActionPrimitive::Approach {
            target: ApproachTarget::Charger,
        },
        signals,
        None,
    );
    let dock = score_action_candidate(&now, &ActionPrimitive::Dock, signals, None);

    assert!(approach.score > dock.score);
}

#[test]
fn scoring_avoids_high_danger_candidate() {
    let now = Now::blank(100, BodySense::default());
    let safe = score_action_candidate(
        &now,
        &ActionPrimitive::Stop,
        CandidateModelSignals::default(),
        None,
    );
    let dangerous = score_action_candidate(
        &now,
        &ActionPrimitive::Go {
            intensity: 0.15,
            duration_ms: 1_000,
        },
        CandidateModelSignals {
            danger: Some(DangerOutput {
                bump_risk: 0.95,
                confidence: 1.0,
                ..DangerOutput::default()
            }),
            ..CandidateModelSignals::default()
        },
        None,
    );

    assert!(safe.score > dangerous.score);
}

#[test]
fn missing_model_signals_fall_back_with_warning() {
    let now = Now::blank(100, BodySense::default());
    let candidate = score_action_candidate(
        &now,
        &ActionPrimitive::Stop,
        CandidateModelSignals::default(),
        None,
    );
    let decision = select_action_from_scores(
        ActionSelectorMode::ModelAssisted,
        &now,
        ActionPrimitive::Stop,
        vec![candidate],
    );

    assert!(!decision.fallback_warnings.is_empty());
    assert!(decision.candidates[0].fallback_used);
}

#[tokio::test]
async fn model_assisted_tick_logs_compact_decision_info() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-action-selector-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    )
    .with_action_selector_mode(ActionSelectorMode::ModelAssisted);

    let tick = runtime
        .tick(
            Now::blank(100, BodySense::default()),
            ExperienceLatent::default(),
            Vec::new(),
        )
        .await
        .unwrap();
    let decision = tick
        .frame
        .now
        .extensions
        .get("action_selector")
        .cloned()
        .and_then(|value| serde_json::from_value::<ActionSelectionDecision>(value).ok())
        .unwrap();

    assert_eq!(decision.mode, ActionSelectorMode::ModelAssisted);
    assert!(!decision.candidates.is_empty());
    assert!(decision.selected_action.is_some());
    assert!(
        tick.frame.now.extensions["conductor.navigation_goal"]["reason"]
            .as_str()
            .is_some_and(|reason| !reason.is_empty())
    );
    assert!(
        !tick.frame.now.extensions["action.motion_bridge"]["conductor_navigation_goal"]["action"]
            .is_null()
    );
}
