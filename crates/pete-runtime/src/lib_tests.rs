use super::*;
use pete_actions::{ChirpPattern, ReignCommand, ReignMode, ReignSource};
use pete_autonomic::SimpleSafety;
use pete_body::BodySense;
use pete_cockpit::{
    establish_session, CockpitCapabilities, CockpitRequest, CockpitResponse, CockpitStatus,
    EventBatch, HandshakeHello, MotherbrainPossession, SimCockpit,
};
use pete_conductor::{Conductor, ConductorInput, GoalId, SimpleConductor};
use pete_experience::{
    embody_now, experience_encode_input_from_now, EmbodiedContext, Modality, SensationPayloadKind,
};
use pete_ledger::{ExperienceFrame, ExperienceTransition, JsonlLedger, LedgerReader};
use pete_llm::{
    ConsciousCommand, LlmDecision, LlmReviewRequest, LlmScientificReview, LlmTickResult,
};
use pete_map::MapConfig;
use pete_memory::{InMemoryExperienceStore, Recall, RecallQuery};
use pete_models::{
    ActionValueNetTrainer, ChargeNetTrainer, DangerNetTrainer, EarNextNetTrainer,
    ExperienceAutoencoderTrainer, FutureNetTrainer,
};
use pete_now::{
    EyeFrame, EyeFrameFormat, Now, ObjectClass, ObjectObservation, ObjectObservationSource,
    SurpriseSense, VectorArtifact, SCENE_VECTOR_COLLECTION,
};
use pete_sensors::World;
use pete_sim::{
    build_scenario, ArenaConfig, ScenarioConfig, ScenarioKind, SimObject, VirtualWorld,
};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

fn mark_corrected_map_trusted(now: &mut Now) {
    now.extensions.insert(
        MAP_EXTENSION_NAME.to_string(),
        serde_json::json!({
            "slam_status": {
                "mode": "loop_closed_pose_graph",
                "local_scan_matching_active": true,
                "loop_closure_active": true,
                "pose_graph_optimized": true,
                "occupancy_remapped_from_pose_graph": true,
                "reasons": []
            },
            "loop_closures_accepted": 1,
            "pose_graph_optimization": {
                "optimized_nodes": 2,
                "active_edges": 2
            },
            "remap": {
                "generation": 1,
                "submaps": 1
            }
        }),
    );
}

#[tokio::test]
async fn no_accelerator_preserves_local_beliefs_and_never_blocks_tick() {
    let mut cognition = LiveImageCognition::new(None);
    let mut now = Now::blank(10, BodySense::default());
    now.range.nearest_m = Some(0.4);
    now.objects.observations.push(ObjectObservation {
        label: "local obstacle".to_string(),
        class: ObjectClass::Obstacle,
        bearing_rad: 0.0,
        distance_m: Some(0.4),
        confidence: 0.9,
        source: ObjectObservationSource::Sim,
    });
    now.eye_frame = Some(EyeFrame {
        captured_at_ms: 10,
        width: 1,
        height: 1,
        format: EyeFrameFormat::Rgb8,
        bytes: vec![1, 2, 3],
        source: Some("fixture".to_string()),
    });
    tokio::time::timeout(
        Duration::from_millis(50),
        enrich_now_latest_image(&mut cognition, &mut now),
    )
    .await
    .expect("optional cognition must not block the organism tick");
    assert_eq!(now.range.nearest_m, Some(0.4));
    assert_eq!(now.objects.observations[0].label, "local obstacle");
    assert!(now.extensions.contains_key("cognition.registry"));
    assert!(!now
        .extensions
        .contains_key("vision.latest_image_description"));
}

#[test]
fn physical_charging_indicator_populates_body_charging_without_oi_state() {
    let status = CockpitStatus {
        raw: serde_json::json!({
            "uptime_ms": 1_000,
            "current_runtime_state": "idle",
            "oi_mode": "safe",
            "create_sensors": {
                "last_packet_id": 0,
                "complete_packet_count": 1,
                "last_complete_packet_timestamp_ms": 1_000,
                "charging_state": 0,
                "charging_indicator": "on"
            }
        })
        .to_string(),
    }
    .summary();

    let body = body_sense_from_cockpit_status(status, 123);

    assert!(body.charging);
    assert_eq!(body.last_update_ms, 123);
}

#[test]
fn create_ir_reaches_motherbrain_body_sense() {
    let status = CockpitStatus {
        raw: serde_json::json!({
            "uptime_ms": 1_000,
            "create_sensors": {
                "last_packet_id": 0,
                "complete_packet_count": 1,
                "last_complete_packet_timestamp_ms": 1_000,
                "ir_byte": 248
            }
        })
        .to_string(),
    }
    .summary();

    let body = body_sense_from_cockpit_status(status, 123);

    assert_eq!(body.infrared_character, 248);
}

#[test]
fn home_base_contacts_do_not_become_upstream_collision_evidence() {
    let status = CockpitStatus {
        raw: serde_json::json!({
            "uptime_ms": 1_000,
            "current_runtime_state": "idle",
            "create_sensors": {
                "last_packet_id": 0,
                "complete_packet_count": 1,
                "last_complete_packet_timestamp_ms": 1_000,
                "charging_sources": 2,
                "bump_left": true,
                "bump_right": true,
                "cliff_left": true,
                "cliff_front_left": true,
                "cliff_front_right": true,
                "cliff_right": true,
                "wheel_drop": true
            }
        })
        .to_string(),
    }
    .summary();

    let body = body_sense_from_cockpit_status(status, 123);

    assert!(!body.flags.bump_left);
    assert!(!body.flags.bump_right);
    assert!(!body.flags.cliff_left);
    assert!(!body.flags.cliff_front_left);
    assert!(!body.flags.cliff_front_right);
    assert!(!body.flags.cliff_right);
    assert!(body.flags.wheel_drop);
}

#[test]
fn identical_contacts_off_home_base_remain_collision_evidence() {
    let status = CockpitStatus {
        raw: serde_json::json!({
            "uptime_ms": 1_000,
            "current_runtime_state": "idle",
            "create_sensors": {
                "last_packet_id": 0,
                "complete_packet_count": 1,
                "last_complete_packet_timestamp_ms": 1_000,
                "charging_sources": 0,
                "bump_left": true,
                "cliff_front_left": true
            }
        })
        .to_string(),
    }
    .summary();

    let body = body_sense_from_cockpit_status(status, 123);

    assert!(body.flags.bump_left);
    assert!(body.flags.cliff_front_left);
}

#[test]
fn body_timestamp_tracks_complete_create_packet_age() {
    let status = CockpitStatus {
        raw: serde_json::json!({
            "uptime_ms": 2_000,
            "current_runtime_state": "idle",
            "oi_mode": "safe",
            "create_sensors": {
                "last_packet_id": 0,
                "complete_packet_count": 4,
                "last_complete_packet_timestamp_ms": 1_250,
                "bump_left": false
            }
        })
        .to_string(),
    }
    .summary();

    let body = body_sense_from_cockpit_status(status, 10_000);

    assert_eq!(body.last_update_ms, 9_250);
    let decision = SimpleSafety::default().filter(
        &Now::blank(10_000, body),
        MotorCommand {
            forward: 0.2,
            turn: 0.0,
        },
    );
    assert_eq!(decision.reason, Some(SafetyReason::StaleSensors));
    assert_eq!(decision.command, MotorCommand::stop());
}

#[test]
fn incomplete_create_packet_never_refreshes_body_timestamp() {
    let status = CockpitStatus {
        raw: serde_json::json!({
            "uptime_ms": 2_000,
            "last_uart_packet_timestamp_ms": 1_990,
            "uart_rx_packets": 4,
            "current_runtime_state": "idle",
            "create_sensors": {
                "last_packet_id": 35,
                "complete_packet_count": 0
            }
        })
        .to_string(),
    }
    .summary();

    assert_eq!(
        body_sense_from_cockpit_status(status, 10_000).last_update_ms,
        0
    );
}

#[test]
fn map_memory_debug_records_intent_confidence_and_signal() {
    let mut now = Now::blank(100, BodySense::default());
    mark_corrected_map_trusted(&mut now);
    now.memory.place_danger = 0.9;
    now.memory.nearby_best_safe_direction_rad = Some(-0.8);
    let action = ActionPrimitive::Turn {
        direction: TurnDir::Right,
        intensity: 0.5,
        duration_ms: 1_000,
    };

    let debug = map_memory_decision_debug(&now, &action, Some(&action), false);

    assert!(debug.influenced);
    assert_eq!(
        debug.navigation_intent,
        Some(NavigationIntent::AvoidKnownDangerCell)
    );
    assert_eq!(debug.reason.as_deref(), Some("danger_safe_direction"));
    assert_eq!(
        debug.signal.as_deref(),
        Some("memory.nearby_best_safe_direction_rad")
    );
    assert_eq!(debug.signal_value, Some(-0.8));
    assert_eq!(debug.signal_confidence, 0.9);
    assert_eq!(debug.confidence, 0.9);
    assert_eq!(debug.chosen_action.as_ref(), Some(&action));
    assert!(!debug.safety_overrode);
    assert!(debug.reason_string.unwrap().contains("remembered danger"));
}

#[test]
fn map_memory_debug_records_low_confidence_charge_fallback() {
    let mut body = BodySense::default();
    body.battery_level = 0.05;
    let mut now = Now::blank(100, body);
    mark_corrected_map_trusted(&mut now);
    now.memory.place_charge_value = 0.1;
    let action = ActionPrimitive::Stop;

    let debug = map_memory_decision_debug(&now, &action, Some(&action), false);

    assert!(debug.influenced);
    assert_eq!(
        debug.navigation_intent,
        Some(NavigationIntent::StopAskForHelpWhenUncertain)
    );
    assert_eq!(
        debug.reason.as_deref(),
        Some("charge_low_confidence_fallback")
    );
    assert_eq!(
        debug.signal.as_deref(),
        Some("memory.nearby_best_charge_direction_rad")
    );
    assert_eq!(debug.signal_value, None);
    assert!(debug.signal_confidence < 0.35);
    assert_eq!(debug.chosen_action, Some(ActionPrimitive::Stop));
    assert!(debug
        .reason_string
        .as_deref()
        .unwrap_or_default()
        .contains("too weak"));
}

#[test]
fn map_memory_debug_rejects_navigation_when_corrected_map_is_untrusted() {
    let mut now = Now::blank(100, BodySense::default());
    now.memory.place_danger = 0.9;
    now.memory.nearby_best_safe_direction_rad = Some(-0.8);
    let action = ActionPrimitive::Turn {
        direction: TurnDir::Right,
        intensity: 0.5,
        duration_ms: 1_000,
    };

    let debug = map_memory_decision_debug(&now, &action, Some(&action), false);

    assert!(!debug.influenced);
    assert!(!debug.corrected_map_trusted);
    assert!(debug
        .corrected_map_untrusted_reason
        .as_deref()
        .unwrap_or_default()
        .contains("summary is missing"));
    assert!(!memory_navigation_candidate_context(&now, &action));
}

#[test]
fn map_memory_debug_rejects_local_scan_match_without_loop_closed_slam() {
    let mut now = Now::blank(100, BodySense::default());
    now.extensions.insert(
        MAP_EXTENSION_NAME.to_string(),
        serde_json::json!({
            "slam_status": {
                "mode": "local_scan_matched",
                "local_scan_matching_active": true,
                "loop_closure_active": false,
                "pose_graph_optimized": true,
                "occupancy_remapped_from_pose_graph": true,
                "reasons": ["no loop-closure candidate has been accepted yet"]
            }
        }),
    );
    now.memory.place_danger = 0.9;
    now.memory.nearby_best_safe_direction_rad = Some(-0.8);
    let action = ActionPrimitive::Turn {
        direction: TurnDir::Right,
        intensity: 0.5,
        duration_ms: 1_000,
    };

    let debug = map_memory_decision_debug(&now, &action, Some(&action), false);

    assert!(!debug.influenced);
    assert!(!debug.corrected_map_trusted);
    assert!(debug
        .corrected_map_untrusted_reason
        .as_deref()
        .unwrap_or_default()
        .contains("slam_status.mode is local_scan_matched"));
    assert!(!memory_navigation_candidate_context(&now, &action));
}

fn idle_now(t_ms: u64) -> Now {
    let mut body = BodySense::default();
    body.last_update_ms = t_ms;
    let mut now = Now::blank(t_ms, body);
    now.range.nearest_m = Some(1.0);
    now.range.beams = vec![1.0, 1.0, 1.0];
    now
}

fn mapped_scene_now(t_ms: u64, x_m: f32, point_id: &str) -> Now {
    let mut body = test_body(x_m, 0.0, 0.8, t_ms);
    body.odometry.heading_rad = 0.0;
    let mut now = Now::blank(t_ms, body);
    now.range.nearest_m = Some(1.0);
    now.range.beams = vec![1.0];
    now.eye.scene_vectors =
        vec![
            VectorArtifact::new(SCENE_VECTOR_COLLECTION, point_id, vec![1.0, 0.0, 0.0])
                .with_occurred_at_ms(t_ms),
        ];
    now
}

#[tokio::test]
async fn ledger_frame_with_asr_metadata_shows_audio_child_sensations() {
    let root = test_ledger_root("asr-audio-child-sensations");
    let ledger = JsonlLedger::new(&root);
    let mut now = Now::blank(1_000, BodySense::default());
    now.ear.asr = pete_now::AsrSense {
        transcript: Some("hello from replay".to_string()),
        is_final: true,
        confidence: 0.84,
        start_ms: Some(250),
        end_ms: Some(950),
        duration_ms: Some(700),
        word_count: Some(3),
        ..pete_now::AsrSense::default()
    };

    let embodied = embody_now(&now).await.unwrap();
    let frame = ExperienceFrame {
        id: Uuid::new_v4(),
        t_ms: now.t_ms,
        now,
        sensations: embodied.sensations,
        impressions: embodied.impressions,
        experiences: vec![embodied.experience],
        z: None,
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
        notes: vec!["asr ledger smoke".to_string()],
    };
    ledger.append(&frame).await.unwrap();

    let frames = ledger.recent(1).await.unwrap();
    let readback = frames.first().expect("ledger frame");
    assert!(readback.sensations.iter().any(|sensation| {
        sensation.payload_kind == SensationPayloadKind::SpeechSegment
            && sensation.parent_id.is_some()
            && sensation.payload.get("text").and_then(Value::as_str) == Some("hello from replay")
    }));
    assert!(readback.sensations.iter().any(|sensation| {
        sensation.payload_kind == SensationPayloadKind::TranscriptSpan
            && sensation.parent_id.is_some()
    }));
}

#[tokio::test]
async fn embodied_eval_coverage_contract_reports_memory_recall() {
    let report = pete_memory::deterministic_embodied_eval_report()
        .await
        .unwrap();

    assert!(report.passed(), "{:?}", report.failures);
    assert!(report.experience_latent_count > 0);
    assert!(report.summary_impression_count > 0);
    assert!(report.prediction_count > 0);
    assert!(report.memory_link_count > 0);
    assert!(report.recall_sensation_count > 0);
    assert!(report.recall_impression_count > 0);
    assert!(report.lineage_edge_count > 0);
}

fn test_conductor_input(action: ActionPrimitive) -> ConductorInput {
    ConductorInput {
        latent: ExperienceLatent::default(),
        drives: DriveSense::default(),
        memory: pete_now::MemorySense::default(),
        predictions: pete_now::PredictionSense::default(),
        surprise: SurpriseSense::default(),
        llm: pete_now::LlmSense::default(),
        safety: SafetySense::default(),
        reign: ReignSense::default(),
        range: pete_now::RangeSense::default(),
        body: BodySense::default(),
        charger_near_score: 0.0,
        charger_visible_score: 0.0,
        proposals: vec![action],
    }
}

#[test]
fn conductor_shadow_train_returns_reign_teacher_action_and_observes_model() {
    let teacher_action = ActionPrimitive::Turn {
        direction: TurnDir::Right,
        intensity: 0.4,
        duration_ms: 500,
    };
    let mut input = test_conductor_input(ActionPrimitive::Stop);
    input.reign.active = true;
    input.reign.latest = Some(ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: 10,
        expires_at_ms: 500,
        source: ReignSource::WebRemote,
        mode: ReignMode::Direct,
        command: ReignCommand::Turn {
            direction: TurnDir::Right,
            intensity: 0.4,
            duration_ms: 500,
        },
        priority: 1.0,
        note: None,
    });
    let mut behavior = conductor_behavior(
        BehaviorRegime::ShadowTrain,
        "reign.teacher",
        Some("conductor.burn.v0".to_string()),
        FallbackPolicy::UseHardcoded,
    );

    let run = behavior
        .infer_with_teacher_source(&input, 10, TrainingSource::HumanReign)
        .unwrap();

    assert_eq!(run.chosen, teacher_action);
    assert!(run.training_sample_emitted);
    assert_eq!(run.record.model_output, None);
}

#[test]
fn conductor_model_infer_falls_back_to_hardcoded_when_model_has_no_sample() {
    let teacher_action = ActionPrimitive::Dock;
    let input = test_conductor_input(teacher_action.clone());
    let mut behavior = conductor_behavior(
        BehaviorRegime::ModelInfer,
        "action_selector.baseline",
        Some("conductor.burn.v0".to_string()),
        FallbackPolicy::UseHardcoded,
    );

    let run = behavior.infer(&input, 10).unwrap();

    assert_eq!(run.chosen, teacher_action);
    assert!(run.fallback_used);
    assert!(run.record.error.is_some());
}

#[test]
fn bump_script_hardcoded_returns_escape_sequence() {
    let mut behavior = bump_event_behavior(
        BehaviorRegime::Hardcoded,
        Some("event.bump.shadow.v0".to_string()),
        FallbackPolicy::UseHardcoded,
    );

    let run = behavior.infer(&BumpEventInput::default(), 10).unwrap();

    assert!(run.chosen.actions.iter().any(is_bump_lament_action));
    let recovery_actions = run
        .chosen
        .actions
        .iter()
        .filter(|action| {
            matches!(
                action,
                EventScriptAction::Stop | EventScriptAction::Rotate { .. } | EventScriptAction::Go
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        recovery_actions,
        vec![
            EventScriptAction::Stop,
            EventScriptAction::Rotate { deg: 180 },
            EventScriptAction::Go,
        ]
    );
}

#[test]
fn bump_script_shadow_train_returns_teacher_and_observes_model() {
    let mut behavior = bump_event_behavior(
        BehaviorRegime::ShadowTrain,
        Some("event.bump.shadow.v0".to_string()),
        FallbackPolicy::UseHardcoded,
    );

    let run = behavior.infer(&BumpEventInput::default(), 10).unwrap();

    assert!(run.chosen.actions.iter().any(is_bump_lament_action));
    let recovery_actions = run
        .chosen
        .actions
        .iter()
        .filter(|action| {
            matches!(
                action,
                EventScriptAction::Stop | EventScriptAction::Rotate { .. } | EventScriptAction::Go
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        recovery_actions,
        vec![
            EventScriptAction::Stop,
            EventScriptAction::Rotate { deg: 180 },
            EventScriptAction::Go,
        ]
    );
    assert!(run.training_sample_emitted);
    assert_eq!(run.record.hardcoded_output, Some(run.chosen));
}

fn is_bump_lament_action(action: &EventScriptAction) -> bool {
    match action {
        EventScriptAction::Say { text } => {
            matches!(text.as_str(), "Uh-oh" | "Oh no!" | "Oopsie!" | "Oh dear!")
        }
        EventScriptAction::Song { name } => name == "mournful_bump",
        _ => false,
    }
}

#[test]
fn safety_veto_prevents_unsafe_script_movement_and_records_context() {
    let mut now = idle_now(10);
    now.body.battery_level = 0.05;
    let mut safety = SimpleSafety::default();
    let output = EventScriptOutput {
        actions: vec![
            EventScriptAction::Stop,
            EventScriptAction::Rotate { deg: 180 },
            EventScriptAction::Go,
        ],
    };

    let sequence = safety_trace_script_actions(&mut safety, &now, &output);

    let go = sequence.actions.last().unwrap();
    assert_eq!(go.requested, EventScriptAction::Go);
    assert!(go.vetoed);
    assert_eq!(go.final_motor, MotorCommand::stop());
    assert_eq!(go.safety_reason.as_deref(), Some("critical battery"));
}

fn prime_idle(controller: &mut NudgeController, now: &Now, policy: NudgePolicy) {
    let mut first = now.clone();
    first.t_ms = now.t_ms.saturating_sub(policy.idle_after_ms);
    first.body.last_update_ms = first.t_ms;
    assert!(controller.propose(&first, policy).is_none());
}

#[test]
fn nudge_refuses_wheel_drop() {
    let policy = NudgePolicy::virtual_default();
    let mut controller = NudgeController::default();
    let mut now = idle_now(5_000);
    now.body.flags.wheel_drop = true;
    prime_idle(&mut controller, &now, policy);

    assert!(controller.propose(&now, policy).is_none());
    assert_eq!(
        controller.status.nudge_blocked_reason.as_deref(),
        Some("wheel drop detected")
    );
}

#[test]
fn nudge_refuses_critical_battery() {
    let policy = NudgePolicy::virtual_default();
    let mut controller = NudgeController::default();
    let mut now = idle_now(5_000);
    now.body.battery_level = 0.05;
    prime_idle(&mut controller, &now, policy);

    assert!(controller.propose(&now, policy).is_none());
    assert_eq!(
        controller.status.nudge_blocked_reason.as_deref(),
        Some("battery is critical")
    );
}

#[test]
fn nudge_avoids_forward_when_obstacle_too_close() {
    let policy = NudgePolicy::virtual_default();
    let mut now = idle_now(5_000);
    now.range.nearest_m = Some(0.2);
    let action = ActionPrimitive::Go {
        intensity: 0.12,
        duration_ms: 500,
    };

    assert!(nudge_action_block_reason(&now, &action, policy)
        .unwrap()
        .contains("clearance"));
}

#[test]
fn turn_nudge_allowed_when_forward_path_blocked() {
    let policy = NudgePolicy::virtual_default();
    let mut controller = NudgeController::default();
    let mut now = idle_now(5_000);
    now.range.nearest_m = Some(0.2);
    now.range.beams = vec![0.2, 0.2, 0.8];
    prime_idle(&mut controller, &now, policy);

    let action = controller.propose(&now, policy).unwrap();
    assert!(matches!(
        action,
        ActionPrimitive::Turn {
            direction: TurnDir::Right,
            ..
        }
    ));
}

#[test]
fn nudge_cooldown_prevents_repeated_twitching() {
    let policy = NudgePolicy::virtual_default();
    let mut controller = NudgeController::default();
    let now = idle_now(5_000);
    prime_idle(&mut controller, &now, policy);
    assert!(controller.propose(&now, policy).is_some());

    let later = idle_now(6_000);
    assert!(controller.propose(&later, policy).is_none());
    assert_eq!(
        controller.status.nudge_blocked_reason.as_deref(),
        Some("prod cooldown active")
    );
}

#[test]
fn default_candidates_include_novelty_inspection() {
    assert!(default_candidate_actions().iter().any(|action| {
        matches!(
            action,
            ActionPrimitive::Inspect {
                target: InspectTarget::Novelty
            }
        )
    }));
}

#[test]
fn curiosity_scores_inspection_above_stopping() {
    let mut now = idle_now(1_000);
    now.drives.curiosity = 0.8;
    now.memory.place_novelty = 0.7;

    let stop = score_action_candidate(
        &now,
        &ActionPrimitive::Stop,
        CandidateModelSignals::default(),
        None,
    );
    let inspect = score_action_candidate(
        &now,
        &ActionPrimitive::Inspect {
            target: InspectTarget::Novelty,
        },
        CandidateModelSignals::default(),
        None,
    );

    assert!(inspect.score > stop.score);
    assert!(inspect.curiosity > 0.0);
}

#[test]
fn possessor_skill_runtime_turns_and_completes_from_updated_target_error() {
    let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
    let events = cockpit.get_events_since(0).unwrap();
    let mut runtime = PossessorSkillRuntime::default();
    let request = SkillRequest {
        skill_id: SkillId::TurnTowardTarget,
        target: Some(pete_now::EntityId("charger:17".to_string())),
        bearing_rad: Some(0.5),
        maximum_duration_ms: 1_000,
        expected_progress: 0.8,
        ..SkillRequest::default()
    };

    let body = BodySense::default();
    let _ = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        1,
    );
    let (running, command_sent) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        101,
    );
    assert!(command_sent);
    assert_eq!(running.phase, SkillPhase::Running);
    assert_eq!(running.attempts, 1);
    assert_eq!(running.dispatch_count, 1);
    assert_ne!(running.execution_id, 0);

    let aligned = SkillRequest {
        bearing_rad: Some(0.05),
        ..request
    };
    let (completed, stop_sent) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &aligned,
        &body,
        false,
        &events,
        201,
    );
    assert!(stop_sent);
    assert_eq!(completed.phase, SkillPhase::Terminal);
    assert_eq!(completed.outcome, Some(SkillOutcome::Completed));
    assert_eq!(completed.progress, Some(1.0));
    assert_eq!(completed.execution_id, running.execution_id);
    assert_eq!(completed.attempts, 1);
}

#[test]
fn possessor_terminal_status_is_consumed_once_before_a_real_retry() {
    let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
    let events = cockpit.get_events_since(0).unwrap();
    let mut runtime = PossessorSkillRuntime::default();
    let request = SkillRequest {
        skill_id: SkillId::ApproachTarget,
        goal_id: Some(pete_conductor::GoalId::new("seek_charger")),
        behavior_id: Some("approach_charger".to_string()),
        target: Some(pete_now::EntityId("charger:17".to_string())),
        bearing_rad: Some(0.0),
        range_m: Some(2.0),
        stop_range_m: Some(0.3),
        maximum_duration_ms: 100,
        ..SkillRequest::default()
    };

    let body = BodySense::default();
    let _ = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        1,
    );
    let (running, first_dispatch) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        51,
    );
    assert!(first_dispatch);
    assert_eq!(running.attempts, 1);
    assert_eq!(running.dispatch_count, 1);

    let (timed_out, stop_sent) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        101,
    );
    assert!(stop_sent);
    assert_eq!(timed_out.phase, SkillPhase::Terminal);
    assert_eq!(timed_out.outcome, Some(SkillOutcome::TimedOut));
    assert_eq!(timed_out.execution_id, running.execution_id);
    assert_eq!(timed_out.attempts, 1);

    let _ = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        102,
    );
    let (retry, retry_dispatched) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        152,
    );
    assert!(retry_dispatched);
    assert_eq!(retry.phase, SkillPhase::Running);
    assert_ne!(retry.execution_id, timed_out.execution_id);
    assert_eq!(retry.attempts, 2);
    assert_eq!(retry.dispatch_count, 1);
}

#[test]
fn possessor_motor_refreshes_do_not_increment_attempts() {
    let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
    let events = cockpit.get_events_since(0).unwrap();
    let mut runtime = PossessorSkillRuntime::default();
    let request = SkillRequest {
        skill_id: SkillId::FollowBearing,
        bearing_rad: Some(0.5),
        maximum_duration_ms: 5_000,
        ..SkillRequest::default()
    };

    let body = BodySense::default();
    let (first, _) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        1,
    );
    let (second, _) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        101,
    );
    let (third, _) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        201,
    );

    assert_eq!(first.execution_id, second.execution_id);
    assert_eq!(second.execution_id, third.execution_id);
    assert_eq!(third.attempts, 1);
    assert_eq!(third.dispatch_count, 2);
}

#[test]
fn lua_skill_progress_preserves_goal_metric_and_normalizes_from_baseline() {
    let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
    let events = cockpit.get_events_since(0).unwrap();
    let mut runtime = PossessorSkillRuntime::default();
    let request = SkillRequest {
        skill_id: SkillId::ApproachTarget,
        goal_id: Some(pete_conductor::GoalId::new("seek_charger")),
        target: Some(pete_now::EntityId("charger:17".to_string())),
        bearing_rad: Some(0.0),
        range_m: Some(2.0),
        stop_range_m: Some(0.3),
        maximum_duration_ms: 5_000,
        progress_metric: "target_distance".to_string(),
        progress_baseline: Some(2.0),
        progress_tolerance: 0.1,
        ..SkillRequest::default()
    };
    let body = BodySense::default();
    let _ = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        1,
    );
    let (started, _) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        101,
    );
    assert_eq!(started.progress, Some(0.0));
    assert_eq!(started.request.goal_id, request.goal_id);
    assert_eq!(started.request.progress_metric, "target_distance");

    let halfway = SkillRequest {
        range_m: Some(1.0),
        ..request
    };
    let (progressed, _) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &halfway,
        &body,
        false,
        &events,
        201,
    );
    assert_eq!(progressed.progress, Some(0.5));
    assert_eq!(
        progressed
            .script
            .as_ref()
            .map(|script| script.skill_id.as_str()),
        Some("motherbrain.approachTarget")
    );
    let provenance = runtime
        .provenance
        .as_ref()
        .expect("running skill provenance");
    assert_eq!(
        provenance
            .pointer("/request/goal_id")
            .and_then(serde_json::Value::as_str),
        Some("seek_charger")
    );
    assert_eq!(
        provenance
            .pointer("/diagnostics/progress/target_distance")
            .and_then(serde_json::Value::as_f64),
        Some(1.0)
    );
    let mut ledger_now = Now::blank(250, body);
    runtime.annotate_now(&mut ledger_now);
    assert!(ledger_now
        .extensions
        .contains_key("motherbrain.skill_execution"));
}

#[test]
fn recognized_person_greeting_runs_through_goal_lua_and_acknowledges_encounter() {
    std::thread::Builder::new()
        .name("lua-greet-goal-test".to_string())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async {
                    let ledger = JsonlLedger::new("/tmp/pete-runtime-lua-greet-goal-test");
                    let memory = InMemoryExperienceStore::new();
                    let recall = FixedRecall {
                        bundle: RecallBundle {
                            sense: pete_now::MemorySense {
                                face_familiarity: 0.92,
                                remembered_entities: vec![pete_now::GraphEntity {
                                    id: "person:alex".to_string(),
                                    labels: vec!["Person".to_string()],
                                    summary: "Alex".to_string(),
                                    score: 0.90,
                                }],
                                ..pete_now::MemorySense::default()
                            },
                            ..RecallBundle::default()
                        },
                    };
                    let mut runtime = MinimalRuntime::new(
                        ledger,
                        memory,
                        recall,
                        FixedConductor::new(ActionPrimitive::Stop),
                        SimpleSafety::default(),
                        pete_llm::NoopLlmAgent,
                    )
                    .with_action_selector_mode(ActionSelectorMode::Goal);
                    let person_now = |t_ms| {
                        let mut now = idle_now(t_ms);
                        now.face.vectors.push(
                            VectorArtifact::new(
                                pete_now::FACE_VECTOR_COLLECTION,
                                format!("face-alex-{t_ms}"),
                                vec![0.1, 0.2],
                            )
                            .with_model("test-face-model"),
                        );
                        now
                    };

                    let first = runtime
                        .tick(person_now(100), ExperienceLatent::default(), Vec::new())
                        .await
                        .unwrap();
                    let request = first
                        .skill_request
                        .clone()
                        .expect("recognized encounter should propose greeting skill");
                    assert_eq!(request.goal_id, Some(GoalId::new("greet_person")));
                    assert_eq!(request.skill_id, SkillId::RuntimeLoaded);
                    assert_eq!(
                        request.implementation_id.as_deref(),
                        Some("motherbrain.greet")
                    );
                    assert!(!first
                        .frame
                        .now
                        .extensions
                        .get("event_scripts")
                        .is_some_and(|scripts| scripts.get("face-detected").is_some()));

                    let mut skills = PossessorSkillRuntime::default();
                    let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
                    let events = cockpit.get_events_since(0).unwrap();
                    let status_summary = cockpit.get_status().unwrap().summary();
                    let mut skill_now = first.frame.now.clone();
                    let (started, _) = skills.step(
                        &mut cockpit,
                        &request,
                        &skill_now,
                        &status_summary,
                        false,
                        &events,
                        100,
                    );
                    assert_ne!(started.phase, SkillPhase::Terminal);
                    let mut completed = started;
                    for now_ms in [200, 300, 400] {
                        skill_now.t_ms = now_ms;
                        skill_now.body.last_update_ms = now_ms;
                        (completed, _) = skills.step(
                            &mut cockpit,
                            &request,
                            &skill_now,
                            &status_summary,
                            false,
                            &events,
                            now_ms,
                        );
                        if completed.phase == SkillPhase::Terminal {
                            break;
                        }
                    }
                    assert_eq!(completed.outcome, Some(SkillOutcome::Completed));
                    assert_eq!(completed.progress, Some(1.0));
                    let execution = skills.provenance.as_ref().unwrap();
                    assert!(execution
                        .get("observations")
                        .and_then(serde_json::Value::as_array)
                        .is_some_and(|observations| observations.iter().any(|observation| {
                            observation.get("kind").and_then(serde_json::Value::as_str)
                                == Some("social_acknowledgment")
                        })));
                    assert!(execution
                        .get("trace")
                        .and_then(serde_json::Value::as_array)
                        .is_some_and(|trace| trace.iter().any(|event| {
                            event.get("kind").and_then(serde_json::Value::as_str)
                                == Some("primitive")
                                && event.get("operation").and_then(serde_json::Value::as_str)
                                    == Some("say")
                                && event
                                    .pointer("/detail/text")
                                    .and_then(serde_json::Value::as_str)
                                    == Some("Hello Alex.")
                        })));

                    let mut next_now = person_now(300);
                    skills.annotate_now(&mut next_now);
                    runtime.observe_skill_status(&completed);
                    let next = runtime
                        .tick(next_now, ExperienceLatent::default(), Vec::new())
                        .await
                        .unwrap();
                    let interaction = next
                        .frame
                        .now
                        .world
                        .social
                        .active_interaction
                        .as_ref()
                        .unwrap();
                    assert!(interaction.has_acknowledgment(
                        &pete_now::PersonId("person:alex".to_string()),
                        pete_now::SocialAcknowledgmentKind::GreetingAttempted,
                    ));
                    assert_ne!(
                        next.frame.now.extensions["goal_system"]["selection"]["selected_goal"]
                            .as_str(),
                        Some("greet_person")
                    );
                    assert!(next.skill_request.as_ref().is_none_or(|request| {
                        request.goal_id != Some(GoalId::new("greet_person"))
                    }));
                    let mut following_now = person_now(400);
                    skills.annotate_now(&mut following_now);
                    let following = runtime
                        .tick(following_now, ExperienceLatent::default(), Vec::new())
                        .await
                        .unwrap();
                    assert_ne!(
                        following.frame.now.world.self_model.active_goal.as_deref(),
                        Some("greet_person")
                    );
                });
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn dock_alignment_skill_follows_ir_and_fails_stopped_when_the_gradient_disappears() {
    let request = SkillRequest {
        skill_id: SkillId::AlignWithDock,
        maximum_duration_ms: 2_000,
        ..SkillRequest::default()
    };
    let mut body = BodySense {
        infrared_character: 254,
        ..BodySense::default()
    };
    let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
    let events = cockpit.get_events_since(0).unwrap();
    let mut runtime = PossessorSkillRuntime::default();

    let _ = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        1,
    );
    let (running, command_sent) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        101,
    );
    assert!(command_sent);
    assert_eq!(running.phase, SkillPhase::Running);

    body.infrared_character = 0;
    let (lost, stop_sent) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        201,
    );
    assert!(stop_sent);
    assert_eq!(lost.phase, SkillPhase::Terminal);
    assert_eq!(lost.outcome, Some(SkillOutcome::Failed));
    assert!(lost.reason.unwrap().contains("IR gradient"));
}

#[test]
fn charging_or_home_base_contact_completes_dock_alignment() {
    let request = SkillRequest {
        skill_id: SkillId::AlignWithDock,
        maximum_duration_ms: 2_000,
        ..SkillRequest::default()
    };
    let body = BodySense {
        charging: true,
        ..BodySense::default()
    };
    let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
    let events = cockpit.get_events_since(0).unwrap();
    let mut runtime = PossessorSkillRuntime::default();

    let _ = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        1,
    );
    let (completed, stop_sent) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &events,
        101,
    );
    assert!(stop_sent);
    assert_eq!(completed.outcome, Some(SkillOutcome::Completed));

    let body = BodySense::default();
    let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
    let events = cockpit.get_events_since(0).unwrap();
    let mut runtime = PossessorSkillRuntime::default();
    let mut completed = SkillStatus::default();
    let mut stop_sent = false;
    for now_ms in [1, 101, 201, 301] {
        (completed, stop_sent) = possessor_test_step(
            &mut runtime,
            &mut cockpit,
            &request,
            &body,
            true,
            &events,
            now_ms,
        );
    }
    let _ = stop_sent;
    assert_eq!(completed.outcome, Some(SkillOutcome::Completed));
}

#[test]
fn create_ir_adds_directional_charger_evidence_and_bias_scores() {
    let mut now = Now::blank(
        100,
        BodySense {
            battery_level: 0.15,
            infrared_character: 246,
            ..BodySense::default()
        },
    );

    apply_create_ir_charger_cue(&mut now);
    let observation = now.objects.observations.last().unwrap();
    assert_eq!(observation.class, ObjectClass::Charger);
    assert_eq!(observation.source, ObjectObservationSource::CreateIr);
    assert_eq!(observation.bearing_rad, -0.35);
    assert_eq!(observation.distance_m, None);
    assert_eq!(charger_signal_scores(&now), (0.55, 0.85));

    let mut input = test_conductor_input(ActionPrimitive::Stop);
    input.body = now.body.clone();
    input.charger_near_score = charger_signal_scores(&now).0;
    input.charger_visible_score = charger_signal_scores(&now).1;
    assert_eq!(
        SimpleConductor::default().choose(input).unwrap(),
        ActionPrimitive::Approach {
            target: ApproachTarget::Charger,
        }
    );
}

#[test]
fn brainstem_contact_reflex_safety_preempts_a_possessor_skill() {
    let mut cockpit = SimCockpit::new().with_unscoped_bench_mode();
    let initial_events = cockpit.get_events_since(0).unwrap();
    let mut runtime = PossessorSkillRuntime::default();
    let request = SkillRequest {
        skill_id: SkillId::ApproachTarget,
        goal_id: None,
        behavior_id: None,
        target: Some(pete_now::EntityId("charger:17".to_string())),
        bearing_rad: Some(0.0),
        range_m: Some(2.0),
        stop_range_m: Some(0.3),
        maximum_duration_ms: 2_000,
        expected_progress: 0.9,
        ..SkillRequest::default()
    };
    let body = BodySense::default();
    let _ = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &initial_events,
        1,
    );
    let (_, command_sent) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &initial_events,
        51,
    );
    assert!(command_sent);

    let cursor = initial_events.next_seq.saturating_sub(1);
    cockpit.set_bump(true, false);
    let reflex_events = cockpit.get_events_since(cursor).unwrap();
    let (preempted, command_sent) = possessor_test_step(
        &mut runtime,
        &mut cockpit,
        &request,
        &body,
        false,
        &reflex_events,
        100,
    );

    assert!(command_sent, "preemption must send a safe stop");
    assert_eq!(preempted.phase, SkillPhase::Terminal);
    assert_eq!(preempted.outcome, Some(SkillOutcome::SafetyPreempted));
}

fn possessor_test_step(
    runtime: &mut PossessorSkillRuntime,
    cockpit: &mut SimCockpit,
    request: &SkillRequest,
    body: &BodySense,
    home_base_contact: bool,
    events: &pete_cockpit::EventBatch,
    now_ms: u64,
) -> (SkillStatus, bool) {
    let now = Now::blank(now_ms, body.clone());
    runtime.step(
        cockpit,
        request,
        &now,
        &StatusSummary::from_raw(""),
        home_base_contact,
        events,
        now_ms,
    )
}

struct StubRuntime;

#[async_trait::async_trait]
impl RuntimeLoop for StubRuntime {
    async fn tick(
        &mut self,
        now: Now,
        _latent: ExperienceLatent,
        _futures: Vec<FuturePrediction>,
    ) -> Result<RuntimeTick> {
        let reign_input = now.reign.latest.clone();
        let action = ActionPrimitive::Go {
            intensity: 0.2,
            duration_ms: 100,
        };
        let experience =
            Experience::new("test", "test", Vec::new(), Vec::new(), now.t_ms, now.t_ms);
        Ok(RuntimeTick {
            frame: ExperienceFrame {
                id: Uuid::new_v4(),
                t_ms: now.t_ms,
                now,
                sensations: Vec::new(),
                impressions: Vec::new(),
                experiences: vec![experience.clone()],
                z: Some(ExperienceLatent::default()),
                chosen_action: Some(action.clone()),
                conscious_command: None,
                reign_input,
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
            experience,
            chosen_action: Some(action),
            skill_request: None,
            skill_status: None,
            recall: RecallBundle::default(),
            llm: LlmTickResult::default(),
            combobulation: None,
            inline_learning: InlineLearningTickStatus::default(),
        })
    }
}

struct SlowRuntime {
    tick_attempts: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl RuntimeLoop for SlowRuntime {
    async fn tick(
        &mut self,
        now: Now,
        _latent: ExperienceLatent,
        _futures: Vec<FuturePrediction>,
    ) -> Result<RuntimeTick> {
        self.tick_attempts.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        let experience =
            Experience::new("test", "test", Vec::new(), Vec::new(), now.t_ms, now.t_ms);
        Ok(RuntimeTick {
            frame: ExperienceFrame {
                id: Uuid::new_v4(),
                t_ms: now.t_ms,
                now,
                sensations: Vec::new(),
                impressions: Vec::new(),
                experiences: vec![experience.clone()],
                z: Some(ExperienceLatent::default()),
                chosen_action: Some(ActionPrimitive::Stop),
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
            experience,
            chosen_action: Some(ActionPrimitive::Stop),
            skill_request: None,
            skill_status: None,
            recall: RecallBundle::default(),
            llm: LlmTickResult::default(),
            combobulation: None,
            inline_learning: InlineLearningTickStatus::default(),
        })
    }
}

#[derive(Clone)]
struct SharedSimCockpit(Arc<Mutex<SimCockpit>>);

impl Cockpit for SharedSimCockpit {
    fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
        self.0.lock().unwrap().execute(request)
    }

    fn handshake(
        &mut self,
        hello: pete_cockpit::HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        self.0.lock().unwrap().handshake(hello)
    }

    fn execute_in_session(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.0.lock().unwrap().execute_in_session(session, request)
    }

    fn execute_with_lease(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        lease: &pete_cockpit::ControlLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.0
            .lock()
            .unwrap()
            .execute_with_lease(session, lease, request)
    }

    fn execute_with_service_lease(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        lease: &pete_cockpit::ServiceLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.0
            .lock()
            .unwrap()
            .execute_with_service_lease(session, lease, request)
    }
}

#[test]
fn production_possession_composes_bump_stop_and_conductor_recovery() {
    let sim = Arc::new(Mutex::new(SimCockpit::new().with_event_capacity(256)));
    let session = establish_session(
        SharedSimCockpit(Arc::clone(&sim)),
        HandshakeHello::motherbrain("pete-runtime-recovery-test"),
        None,
    )
    .unwrap();
    let possession = MotherbrainPossession::acquire(session, 5_000).unwrap();
    let mut cockpit = SafeCockpit::new(possession);

    cockpit.pulse_motion(40, 0).unwrap();
    sim.lock().unwrap().set_bump(true, false);

    let stop_events = cockpit.poll_events().unwrap();
    assert!(stop_events.has_stop_reason());
    let status = cockpit.refresh_status().unwrap();
    let body = body_sense_from_cockpit_status(status, 1_000);
    assert!(body.flags.bump_left);

    let mut conductor = SimpleConductor::default();
    let mut input = test_conductor_input(ActionPrimitive::Stop);
    input.body = body;
    let first_recovery = conductor.choose(input).unwrap();
    assert!(matches!(
        first_recovery,
        ActionPrimitive::Go {
            intensity,
            duration_ms: 500
        } if intensity < 0.0
    ));

    sim.lock().unwrap().set_bump(false, false);
    cockpit
        .client_mut()
        .clear_safety_latch(SafetyLatchKind::Bump)
        .unwrap();
    let clear_events = cockpit.poll_events().unwrap();
    assert!(clear_events
        .events
        .iter()
        .any(|event| event.kind == pete_cockpit::CockpitEventKind::SafetyCleared));
    let cleared_body = body_sense_from_cockpit_status(cockpit.refresh_status().unwrap(), 1_100);
    assert!(!cleared_body.flags.bump_left);

    let mut cleared_input = test_conductor_input(ActionPrimitive::Stop);
    cleared_input.body = cleared_body.clone();
    let reverse = conductor.choose(cleared_input.clone()).unwrap();
    let reverse_motor = action_to_motor_command(Some(&reverse)).clamped(0.05, 0.5);
    assert!(reverse_motor.forward < 0.0);
    apply_slow_possession_motor(&mut cockpit, reverse_motor).unwrap();

    cleared_input.body.odometry.x_m -= 0.08;
    let turn = conductor.choose(cleared_input).unwrap();
    assert!(matches!(
        turn,
        ActionPrimitive::Turn {
            direction: TurnDir::Right,
            ..
        }
    ));
    let turn_motor = action_to_motor_command(Some(&turn)).clamped(0.05, 0.5);
    assert!(turn_motor.turn.abs() > 0.0);
    apply_slow_possession_motor(&mut cockpit, turn_motor).unwrap();

    assert!(cockpit.client_mut().snapshot().possessed);
    let events = sim.lock().unwrap().get_events_since(0).unwrap();
    let stop_index = events
        .events
        .iter()
        .position(|event| event.kind == pete_cockpit::CockpitEventKind::MotionStopped)
        .unwrap();
    assert!(events.events[stop_index + 1..]
        .iter()
        .any(|event| event.kind == pete_cockpit::CockpitEventKind::MotionRequested));
}

#[test]
fn slow_possession_treats_preflight_bump_latch_as_recoverable() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.set_bump(true, false);
    let mut cockpit = SafeCockpit::with_policy(
        sim,
        pete_cockpit::AgentPolicy {
            motion_ttl_ms: 100,
            heartbeat_timeout_ms: 0,
        },
    );

    let block = apply_slow_possession_motor(
        &mut cockpit,
        MotorCommand {
            forward: 0.2,
            turn: 0.1,
        },
    )
    .unwrap();

    assert!(matches!(
        block,
        Some(SlowPossessionMotionBlock::SafetyLatch(
            SafetyLatchKind::Bump
        ))
    ));
    let status = cockpit.refresh_status().unwrap();
    assert_eq!(status.safety_tripped, Some(true));
    assert_eq!(status.safety_latch_kind, Some(SafetyLatchKind::Bump));
}

#[tokio::test]
async fn operator_estop_during_bump_recovery_requires_explicit_operator_clear() {
    let sim = Arc::new(Mutex::new(SimCockpit::new().with_event_capacity(256)));
    let session = establish_session(
        SharedSimCockpit(Arc::clone(&sim)),
        HandshakeHello::motherbrain("pete-runtime-normal-bump-test"),
        None,
    )
    .unwrap();
    let possession = MotherbrainPossession::acquire(session, 5_000).unwrap();
    let ledger_root = test_ledger_root("normal-possession-bump-recovery");
    let ledger = JsonlLedger::new(&ledger_root);
    let memory = InMemoryExperienceStore::new();
    let runtime = MinimalRuntime::new(
        ledger,
        memory.clone(),
        memory,
        SimpleConductor::default(),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    );
    let mut runner = RealRobotRunner::new(RobotMode::Slow, possession, Vec::new(), runtime)
        .with_autonomous_motion(true);
    runner.cockpit.resync_event_cursor_from_status().unwrap();

    let (_first_snapshot, _first_tick) = runner.tick_slow_manual().await.unwrap();

    sim.lock().unwrap().set_bump(true, false);
    // An E-stop received during the local reflex has no origin metadata,
    // so it must be treated as an operator stop and remain latched.
    runner.cockpit.client_mut().estop().unwrap();
    let (bump_snapshot, bump_tick) = runner.tick_slow_manual().await.unwrap();
    assert!(bump_tick.frame.now.body.flags.bump_left);
    assert_eq!(
        bump_snapshot
            .action_debug
            .as_ref()
            .and_then(|debug| debug.get("possession_recovery"))
            .and_then(|debug| debug.get("phase")),
        Some(&serde_json::json!("Idle"))
    );
    assert_eq!(bump_tick.chosen_action, Some(ActionPrimitive::Stop));
    let latched_status = runner.cockpit.refresh_status().unwrap();
    assert_eq!(latched_status.estop_latched, Some(true));

    let events_while_latched = sim.lock().unwrap().get_events_since(0).unwrap();
    let estop_index = events_while_latched
        .events
        .iter()
        .position(|event| event.kind == CockpitEventKind::EStopLatched)
        .unwrap();
    assert!(!events_while_latched.events[estop_index + 1..]
        .iter()
        .any(|event| event.kind == CockpitEventKind::EStopCleared));
    assert!(!events_while_latched.events[estop_index + 1..]
        .iter()
        .any(|event| event.kind == CockpitEventKind::MotionRequested));

    sim.lock().unwrap().set_bump(false, false);
    runner.cockpit.client_mut().clear_estop().unwrap();
    let (_completed_snapshot, _completed_tick) = runner.tick_slow_manual().await.unwrap();
    assert!(runner.possession_recovery.latch.is_none());
    let status = runner.cockpit.refresh_status().unwrap();
    assert_eq!(status.estop_latched, Some(false));
    assert_eq!(status.safety_tripped, Some(false));

    let events = sim.lock().unwrap().get_events_since(0).unwrap();
    let bump_index = events
        .events
        .iter()
        .position(|event| event.kind == pete_cockpit::CockpitEventKind::BumpChanged)
        .unwrap();
    let stop_index = events.events[bump_index..]
        .iter()
        .position(|event| event.kind == pete_cockpit::CockpitEventKind::MotionStopped)
        .map(|index| bump_index + index)
        .unwrap();
    let safety_index = events.events[bump_index..]
        .iter()
        .position(|event| event.kind == pete_cockpit::CockpitEventKind::SafetyTripped)
        .map(|index| bump_index + index)
        .unwrap();
    assert!(bump_index < safety_index && safety_index < stop_index);
    assert!(events
        .events
        .iter()
        .any(|event| { event.kind == CockpitEventKind::EStopCleared }));
    assert!(runner.cockpit.client_mut().snapshot().possessed);
    let _ = fs::remove_dir_all(ledger_root);
}

struct CountingCockpit {
    motor_attempts: Arc<AtomicUsize>,
    motors: Arc<Mutex<Vec<MotorCommand>>>,
    body: BodySense,
}

impl Cockpit for CountingCockpit {
    fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
        match request {
            CockpitRequest::GetStatus => Ok(CockpitResponse::Status(self.get_status()?)),
            CockpitRequest::GetCapabilities => {
                Ok(CockpitResponse::Capabilities(self.get_capabilities()?))
            }
            CockpitRequest::GetEvents { since_seq } => {
                Ok(CockpitResponse::Events(self.get_events_since(since_seq)?))
            }
            CockpitRequest::Stop => {
                self.motor_attempts.fetch_add(1, Ordering::SeqCst);
                self.motors.lock().unwrap().push(MotorCommand::stop());
                Ok(CockpitResponse::Accepted)
            }
            CockpitRequest::CmdVel {
                linear_mm_s,
                angular_mrad_s,
                ..
            } => {
                self.motor_attempts.fetch_add(1, Ordering::SeqCst);
                self.motors.lock().unwrap().push(MotorCommand {
                    forward: linear_mm_s as f32 / 1000.0,
                    turn: angular_mrad_s as f32 / 1000.0,
                });
                Ok(CockpitResponse::Accepted)
            }
            CockpitRequest::HeartbeatStop { .. } => Ok(CockpitResponse::Accepted),
            _ => Ok(CockpitResponse::Accepted),
        }
    }

    fn handshake(
        &mut self,
        _hello: pete_cockpit::HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        Err(pete_cockpit::CockpitError::Policy(
            "test cockpit has no handshake peer".into(),
        ))
    }

    fn execute_in_session(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ControlLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_service_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ServiceLease,
        _request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        Err(pete_cockpit::CockpitError::Policy(
            "test cockpit has no service mode".into(),
        ))
    }

    fn get_status(&mut self) -> pete_cockpit::Result<CockpitStatus> {
        Ok(CockpitStatus {
            raw: serde_json::json!({
                "uptime_ms": 1_000,
                "current_runtime_state": "test",
                "oi_mode": "safe",
                "current_command": "stop",
                "create_sensors": {
                    "last_packet_id": 0,
                    "complete_packet_count": 1,
                    "last_complete_packet_timestamp_ms": 1_000,
                    "bump_left": self.body.flags.bump_left,
                    "bump_right": self.body.flags.bump_right,
                    "wheel_drop": self.body.flags.wheel_drop,
                    "wall": self.body.flags.wall,
                    "virtual_wall": self.body.flags.virtual_wall,
                    "cliff_left": self.body.flags.cliff_left,
                    "cliff_front_left": self.body.flags.cliff_front_left,
                    "cliff_front_right": self.body.flags.cliff_front_right,
                    "cliff_right": self.body.flags.cliff_right,
                    "charge_mah": (self.body.battery_level.clamp(0.0, 1.0) * 2600.0).round() as u32,
                    "capacity_mah": 2600,
                    "charging_state": if self.body.charging { 1 } else { 0 },
                },
                "odometry": {
                    "distance_mm": (self.body.odometry.x_m * 1000.0).round() as i32,
                    "heading_mrad": (self.body.odometry.heading_rad * 1000.0).round() as i32,
                    "reset_count": 0,
                }
            })
            .to_string(),
        })
    }

    fn get_capabilities(&mut self) -> pete_cockpit::Result<CockpitCapabilities> {
        Ok(CockpitCapabilities {
            body_kind: "test".to_string(),
            drive: "differential".to_string(),
            verbs: [
                "status",
                "get_capabilities",
                "get_events",
                "stop",
                "cmd_vel",
                "heartbeat_stop",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            sensors: Vec::new(),
            outputs: Vec::new(),
            safety: Vec::new(),
            events: Vec::new(),
            limits: pete_cockpit::CockpitLimits {
                max_linear_mm_s: 500,
                max_angular_mrad_s: 4_000,
                min_ttl_ms: 1,
                max_ttl_ms: 60_000,
            },
        })
    }

    fn get_events_since(&mut self, since_seq: u32) -> pete_cockpit::Result<EventBatch> {
        Ok(EventBatch {
            since_seq,
            oldest_seq: 1,
            next_seq: since_seq.saturating_add(1),
            dropped_before_seq: 0,
            events: Vec::new(),
        })
    }
}

struct LatchedStatusCockpit {
    clear_attempts: Arc<Mutex<Vec<SafetyLatchKind>>>,
    latch: SafetyLatchKind,
    safety_tripped: bool,
}

impl Cockpit for LatchedStatusCockpit {
    fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
        match request {
            CockpitRequest::GetStatus => Ok(CockpitResponse::Status(self.get_status()?)),
            CockpitRequest::GetCapabilities => {
                Ok(CockpitResponse::Capabilities(self.get_capabilities()?))
            }
            CockpitRequest::GetEvents { since_seq } => {
                Ok(CockpitResponse::Events(self.get_events_since(since_seq)?))
            }
            CockpitRequest::ClearSafetyLatch { latch } => {
                self.clear_attempts.lock().unwrap().push(latch);
                self.safety_tripped = false;
                Ok(CockpitResponse::Accepted)
            }
            CockpitRequest::Stop | CockpitRequest::HeartbeatStop { .. } => {
                Ok(CockpitResponse::Accepted)
            }
            _ => Ok(CockpitResponse::Accepted),
        }
    }

    fn handshake(
        &mut self,
        _hello: pete_cockpit::HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        Err(pete_cockpit::CockpitError::Policy(
            "test cockpit has no handshake peer".into(),
        ))
    }

    fn execute_in_session(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ControlLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_service_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ServiceLease,
        _request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        Err(pete_cockpit::CockpitError::Policy(
            "test cockpit has no service mode".into(),
        ))
    }

    fn get_status(&mut self) -> pete_cockpit::Result<CockpitStatus> {
        Ok(CockpitStatus {
            raw: serde_json::json!({
                "uptime_ms": 1_000,
                "current_runtime_state": "idle",
                "oi_mode": "safe",
                "current_command": "stop",
                "estop_latched": false,
                "safety_tripped": self.safety_tripped,
                "safety_latch_kind": self.latch,
                "create_sensors": {
                    "last_packet_id": 0,
                    "complete_packet_count": 1,
                    "last_complete_packet_timestamp_ms": 1_000,
                    "charging_state": 0,
                },
                "imu": {
                    "health": "ok",
                    "tilt_magnitude_mrad": 0,
                    "impact_score_mm_s2": 0,
                }
            })
            .to_string(),
        })
    }

    fn get_capabilities(&mut self) -> pete_cockpit::Result<CockpitCapabilities> {
        Ok(CockpitCapabilities {
            body_kind: "test".to_string(),
            drive: "differential".to_string(),
            verbs: [
                "status",
                "get_capabilities",
                "get_events",
                "stop",
                "cmd_vel",
                "heartbeat_stop",
                "clear_safety_latch",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            sensors: Vec::new(),
            outputs: Vec::new(),
            safety: Vec::new(),
            events: Vec::new(),
            limits: pete_cockpit::CockpitLimits {
                max_linear_mm_s: 500,
                max_angular_mrad_s: 4_000,
                min_ttl_ms: 1,
                max_ttl_ms: 60_000,
            },
        })
    }

    fn get_events_since(&mut self, since_seq: u32) -> pete_cockpit::Result<EventBatch> {
        Ok(EventBatch {
            since_seq,
            oldest_seq: 1,
            next_seq: since_seq.saturating_add(1),
            dropped_before_seq: 0,
            events: Vec::new(),
        })
    }
}

struct ActiveBumpRecoveryCockpit {
    bump_escape_attempts: Arc<AtomicUsize>,
    careful_mode_attempts: Arc<AtomicUsize>,
    bump_escape_commands: Arc<Mutex<Vec<(SafetyLatchKind, u32, i16, i16, u32)>>>,
    stop_attempts: Arc<AtomicUsize>,
    clear_attempts: Arc<AtomicUsize>,
    bump_active: bool,
}

impl Cockpit for ActiveBumpRecoveryCockpit {
    fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
        match request {
            CockpitRequest::GetStatus => Ok(CockpitResponse::Status(self.get_status()?)),
            CockpitRequest::GetCapabilities => {
                Ok(CockpitResponse::Capabilities(self.get_capabilities()?))
            }
            CockpitRequest::GetEvents { since_seq } => {
                Ok(CockpitResponse::Events(self.get_events_since(since_seq)?))
            }
            CockpitRequest::EscapeMotion {
                hazard,
                hazard_generation,
                linear_mm_s,
                angular_mrad_s,
                ttl_ms,
            } => {
                self.bump_escape_attempts.fetch_add(1, Ordering::SeqCst);
                self.bump_escape_commands.lock().unwrap().push((
                    hazard,
                    hazard_generation,
                    linear_mm_s,
                    angular_mrad_s,
                    ttl_ms,
                ));
                Ok(CockpitResponse::Accepted)
            }
            CockpitRequest::CarefulMode { .. } => {
                self.careful_mode_attempts.fetch_add(1, Ordering::SeqCst);
                Ok(CockpitResponse::Accepted)
            }
            CockpitRequest::Stop => {
                self.stop_attempts.fetch_add(1, Ordering::SeqCst);
                Ok(CockpitResponse::Accepted)
            }
            CockpitRequest::ClearSafetyLatch {
                latch: SafetyLatchKind::Bump,
            } => {
                self.clear_attempts.fetch_add(1, Ordering::SeqCst);
                Ok(CockpitResponse::Accepted)
            }
            CockpitRequest::HeartbeatStop { .. } => Ok(CockpitResponse::Accepted),
            _ => Ok(CockpitResponse::Accepted),
        }
    }

    fn handshake(
        &mut self,
        _hello: pete_cockpit::HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        Err(pete_cockpit::CockpitError::Policy(
            "test cockpit has no handshake peer".into(),
        ))
    }

    fn execute_in_session(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ControlLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.execute(request)
    }

    fn execute_with_service_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ServiceLease,
        _request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        Err(pete_cockpit::CockpitError::Policy(
            "test cockpit has no service mode".into(),
        ))
    }

    fn get_status(&mut self) -> pete_cockpit::Result<CockpitStatus> {
        Ok(CockpitStatus {
            raw: serde_json::json!({
                "uptime_ms": 1_000,
                "current_runtime_state": "idle",
                "oi_mode": "safe",
                "current_command": "stop",
                "estop_latched": false,
                "safety_tripped": true,
                "safety_latch_kind": "bump",
                "safety_hazard_generation": 42,
                "create_sensors": {
                    "last_packet_id": 0,
                    "complete_packet_count": 1,
                    "last_complete_packet_timestamp_ms": 1_000,
                    "bump_left": self.bump_active,
                    "bump_right": false,
                    "charging_state": 0,
                },
                "imu": {
                    "health": "ok",
                    "tilt_magnitude_mrad": 0,
                    "impact_score_mm_s2": 0,
                }
            })
            .to_string(),
        })
    }

    fn get_capabilities(&mut self) -> pete_cockpit::Result<CockpitCapabilities> {
        Ok(CockpitCapabilities {
            body_kind: "test".to_string(),
            drive: "differential".to_string(),
            verbs: [
                "status",
                "get_capabilities",
                "get_events",
                "stop",
                "cmd_vel",
                "escape_motion",
                "heartbeat_stop",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            sensors: Vec::new(),
            outputs: Vec::new(),
            safety: Vec::new(),
            events: Vec::new(),
            limits: pete_cockpit::CockpitLimits {
                max_linear_mm_s: 500,
                max_angular_mrad_s: 4_000,
                min_ttl_ms: 1,
                max_ttl_ms: 60_000,
            },
        })
    }

    fn get_events_since(&mut self, since_seq: u32) -> pete_cockpit::Result<EventBatch> {
        Ok(EventBatch {
            since_seq,
            oldest_seq: 1,
            next_seq: since_seq.saturating_add(1),
            dropped_before_seq: 0,
            events: Vec::new(),
        })
    }
}

struct HistoryGapCockpit {
    inner: CountingCockpit,
    event_polls: Arc<AtomicUsize>,
    gap_poll: usize,
}

impl Cockpit for HistoryGapCockpit {
    fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
        self.inner.execute(request)
    }

    fn handshake(
        &mut self,
        hello: pete_cockpit::HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        self.inner.handshake(hello)
    }

    fn execute_in_session(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.inner.execute_in_session(session, request)
    }

    fn execute_with_lease(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        lease: &pete_cockpit::ControlLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.inner.execute_with_lease(session, lease, request)
    }

    fn execute_with_service_lease(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        lease: &pete_cockpit::ServiceLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.inner
            .execute_with_service_lease(session, lease, request)
    }

    fn get_status(&mut self) -> pete_cockpit::Result<CockpitStatus> {
        self.inner.get_status()
    }

    fn get_capabilities(&mut self) -> pete_cockpit::Result<CockpitCapabilities> {
        self.inner.get_capabilities()
    }

    fn get_events_since(&mut self, since_seq: u32) -> pete_cockpit::Result<EventBatch> {
        let poll = self.event_polls.fetch_add(1, Ordering::SeqCst);
        let inject_gap = poll == self.gap_poll;
        Ok(EventBatch {
            since_seq,
            oldest_seq: if inject_gap {
                since_seq.saturating_add(2)
            } else {
                1
            },
            next_seq: since_seq.saturating_add(2),
            dropped_before_seq: if inject_gap {
                since_seq.saturating_add(2)
            } else {
                0
            },
            events: Vec::new(),
        })
    }
}

struct MotionStopEventsCockpit {
    inner: CountingCockpit,
    event_polls: Arc<AtomicUsize>,
}

impl Cockpit for MotionStopEventsCockpit {
    fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
        self.inner.execute(request)
    }

    fn handshake(
        &mut self,
        hello: pete_cockpit::HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        self.inner.handshake(hello)
    }

    fn execute_in_session(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.inner.execute_in_session(session, request)
    }

    fn execute_with_lease(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        lease: &pete_cockpit::ControlLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.inner.execute_with_lease(session, lease, request)
    }

    fn execute_with_service_lease(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        lease: &pete_cockpit::ServiceLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.inner
            .execute_with_service_lease(session, lease, request)
    }

    fn get_status(&mut self) -> pete_cockpit::Result<CockpitStatus> {
        self.inner.get_status()
    }

    fn get_capabilities(&mut self) -> pete_cockpit::Result<CockpitCapabilities> {
        self.inner.get_capabilities()
    }

    fn get_events_since(&mut self, since_seq: u32) -> pete_cockpit::Result<EventBatch> {
        let poll = self.event_polls.fetch_add(1, Ordering::SeqCst);
        let events = if poll == 1 {
            vec![
                pete_cockpit::CockpitEvent {
                    seq: since_seq.saturating_add(1),
                    kind: pete_cockpit::CockpitEventKind::HeartbeatExpired,
                    a: 0,
                    b: 0,
                    c: 0,
                },
                pete_cockpit::CockpitEvent {
                    seq: since_seq.saturating_add(2),
                    kind: pete_cockpit::CockpitEventKind::SafetyTripped,
                    a: 1,
                    b: 0,
                    c: 0,
                },
            ]
        } else {
            Vec::new()
        };
        Ok(EventBatch {
            since_seq,
            oldest_seq: 1,
            next_seq: since_seq.saturating_add(events.len() as u32 + 1),
            dropped_before_seq: 0,
            events,
        })
    }
}

struct RejectingMotionCockpit {
    inner: CountingCockpit,
    rejection_attempts: Arc<AtomicUsize>,
}

impl Cockpit for RejectingMotionCockpit {
    fn execute(&mut self, request: CockpitRequest) -> pete_cockpit::Result<CockpitResponse> {
        if matches!(&request, CockpitRequest::CmdVel { .. }) {
            self.rejection_attempts.fetch_add(1, Ordering::SeqCst);
            return Err(pete_cockpit::CockpitError::Rejected {
                command_id: 42,
                reason: "stale_sequence".to_string(),
            });
        }
        self.inner.execute(request)
    }

    fn handshake(
        &mut self,
        hello: pete_cockpit::HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        self.inner.handshake(hello)
    }

    fn execute_in_session(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.inner.execute_in_session(session, request)
    }

    fn execute_with_lease(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        lease: &pete_cockpit::ControlLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.inner.execute_with_lease(session, lease, request)
    }

    fn execute_with_service_lease(
        &mut self,
        session: &pete_cockpit::CockpitSession,
        lease: &pete_cockpit::ServiceLease,
        request: CockpitRequest,
    ) -> pete_cockpit::Result<CockpitResponse> {
        self.inner
            .execute_with_service_lease(session, lease, request)
    }

    fn get_status(&mut self) -> pete_cockpit::Result<CockpitStatus> {
        self.inner.get_status()
    }

    fn get_capabilities(&mut self) -> pete_cockpit::Result<CockpitCapabilities> {
        self.inner.get_capabilities()
    }

    fn get_events_since(&mut self, since_seq: u32) -> pete_cockpit::Result<EventBatch> {
        Ok(EventBatch {
            since_seq,
            oldest_seq: 1,
            next_seq: since_seq.saturating_add(1),
            dropped_before_seq: 0,
            events: Vec::new(),
        })
    }
}

struct FailingSensor;

#[async_trait::async_trait]
impl SenseProducer for FailingSensor {
    fn source_name(&self) -> &'static str {
        "kinect-depth"
    }

    async fn poll(&mut self) -> Result<pete_sensors::SensePacket> {
        anyhow::bail!("simulated sensor timeout")
    }
}

#[tokio::test]
async fn real_robot_read_only_runner_never_applies_motor() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let mut runner =
        RealRobotRunner::new(RobotMode::ReadOnly, Box::new(body), Vec::new(), StubRuntime);

    let (_snapshot, tick) = runner.tick_read_only().await.unwrap();

    assert!(matches!(
        tick.chosen_action,
        Some(ActionPrimitive::Go { .. })
    ));
    assert_eq!(motor_attempts.load(Ordering::SeqCst), 0);
    assert!(motors.lock().unwrap().is_empty());
    assert_eq!(
        tick.frame
            .now
            .extensions
            .get("safety/read_only_veto")
            .and_then(|value| value.as_bool()),
        Some(true)
    );
}

#[tokio::test]
async fn real_robot_read_only_runner_publishes_snapshot_when_optional_sensor_fails() {
    let body = CountingCockpit {
        motor_attempts: Arc::new(AtomicUsize::new(0)),
        motors: Arc::new(Mutex::new(Vec::new())),
        body: BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let sensors: Vec<Box<dyn SenseProducer + Send>> = vec![Box::new(FailingSensor)];
    let mut runner =
        RealRobotRunner::new(RobotMode::ReadOnly, Box::new(body), sensors, StubRuntime);

    let (snapshot, _tick) = runner.tick_read_only().await.unwrap();

    assert!(snapshot.body.last_update_ms >= 100);
    assert_eq!(runner.tick_count, 1);
    assert_eq!(snapshot.body.odometry.x_m, 0.0);
    assert_eq!(
        _tick
            .frame
            .now
            .extensions
            .get("sensor.health")
            .and_then(|health| health.get(0))
            .and_then(|health| health.get("name")),
        Some(&serde_json::json!("kinect-depth"))
    );
    assert_eq!(
        _tick
            .frame
            .now
            .extensions
            .get("sensor.health")
            .and_then(|health| health.get(0))
            .and_then(|health| health.get("body_evidence_independent")),
        Some(&serde_json::json!(true))
    );
}

#[tokio::test]
async fn real_robot_slow_runner_keeps_body_evidence_when_kinect_fails() {
    let body = BodySense {
        battery_level: 0.61,
        charging: true,
        flags: pete_body::BodyFlags {
            wheel_drop: true,
            ..pete_body::BodyFlags::default()
        },
        odometry: Pose2 {
            x_m: 1.234,
            heading_rad: 0.875,
            ..Pose2::default()
        },
        last_update_ms: 100,
        ..BodySense::default()
    };
    let cockpit = CountingCockpit {
        motor_attempts: Arc::new(AtomicUsize::new(0)),
        motors: Arc::new(Mutex::new(Vec::new())),
        body,
    };
    let sensors: Vec<Box<dyn SenseProducer + Send>> = vec![Box::new(FailingSensor)];
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(cockpit), sensors, StubRuntime);

    let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(snapshot.body.battery_level, 0.61);
    assert!(snapshot.body.charging);
    assert!(snapshot.body.flags.wheel_drop);
    assert_eq!(snapshot.body.odometry.x_m, 1.234);
    assert_eq!(snapshot.body.odometry.heading_rad, 0.875);
    let health = tick.frame.now.extensions["sensor.health"][0].clone();
    assert_eq!(health["name"], "kinect-depth");
    assert_eq!(health["available"], false);
    assert_eq!(health["body_evidence_independent"], true);
}

#[test]
fn optional_sensor_failures_are_reported_once_per_interval() {
    let mut health = SensorPollHealth {
        name: "kinect-depth".to_string(),
        ..SensorPollHealth::default()
    };

    record_optional_sensor_failure(&mut health, "offline".to_string(), 1_000);
    let first_report = health.last_report_ms;
    record_optional_sensor_failure(&mut health, "offline".to_string(), 2_000);
    assert_eq!(health.last_report_ms, first_report);
    assert_eq!(health.consecutive_failures, 2);
    record_optional_sensor_failure(&mut health, "offline".to_string(), 31_001);
    assert_eq!(health.last_report_ms, 31_001);
}

#[tokio::test]
async fn real_robot_slow_runner_without_webremote_direct_sends_stop() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), StubRuntime);

    let (_snapshot, _tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(motor_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(motors.lock().unwrap().as_slice(), &[MotorCommand::stop()]);
}

#[tokio::test]
async fn real_robot_slow_runner_clears_latch_reported_by_status() {
    let clear_attempts = Arc::new(Mutex::new(Vec::new()));
    let body = LatchedStatusCockpit {
        clear_attempts: Arc::clone(&clear_attempts),
        latch: SafetyLatchKind::Tilt,
        safety_tripped: true,
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), StubRuntime)
        .with_autonomous_motion(true);

    let (snapshot, _tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(
        clear_attempts.lock().unwrap().as_slice(),
        &[SafetyLatchKind::Tilt]
    );
    assert_eq!(
        snapshot
            .action_debug
            .as_ref()
            .and_then(|debug| debug.get("possession_recovery"))
            .and_then(|debug| debug.get("latched")),
        Some(&serde_json::json!("Tilt"))
    );
}

#[tokio::test]
async fn real_robot_slow_runner_reports_active_bump_recovery_as_chosen_action() {
    let bump_escape_attempts = Arc::new(AtomicUsize::new(0));
    let careful_mode_attempts = Arc::new(AtomicUsize::new(0));
    let bump_escape_commands = Arc::new(Mutex::new(Vec::new()));
    let stop_attempts = Arc::new(AtomicUsize::new(0));
    let body = ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc::clone(&bump_escape_attempts),
        careful_mode_attempts: Arc::clone(&careful_mode_attempts),
        bump_escape_commands: Arc::clone(&bump_escape_commands),
        stop_attempts,
        clear_attempts: Arc::new(AtomicUsize::new(0)),
        bump_active: true,
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);

    let _ = runner.tick_slow_manual().await.unwrap();
    let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(bump_escape_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(careful_mode_attempts.load(Ordering::SeqCst), 0);
    assert_eq!(
        bump_escape_commands.lock().unwrap().as_slice(),
        &[(SafetyLatchKind::Bump, 42, -100, 0, 250)]
    );
    assert_eq!(
        tick.chosen_action,
        Some(ActionPrimitive::Go {
            intensity: -0.25,
            duration_ms: POSSESSION_ESCAPE_TTL_MS as TimeMs,
        })
    );
    let debug = snapshot.action_debug.as_ref().unwrap();
    assert_eq!(
        debug.get("runtime_chosen_action"),
        Some(
            &serde_json::to_value(ActionPrimitive::Go {
                intensity: 0.2,
                duration_ms: 100,
            })
            .unwrap()
        )
    );
    assert_eq!(
        debug.get("motion_sent_to_robot"),
        Some(
            &serde_json::to_value(motor_command_to_motion(MotorCommand {
                forward: -0.10,
                turn: 0.0,
            }))
            .unwrap()
        )
    );
    assert_eq!(debug.get("motor_applied"), Some(&serde_json::json!(true)));
    assert_eq!(
        debug
            .get("possession_recovery")
            .and_then(|debug| debug.get("latched")),
        Some(&serde_json::json!("Bump"))
    );
    assert_eq!(
        debug
            .get("possession_recovery")
            .and_then(|debug| debug.get("intended_motion"))
            .and_then(|motion| motion.get("linear")),
        Some(&serde_json::json!("reverse"))
    );
    assert_eq!(
        debug
            .get("possessor_skill_status")
            .and_then(|status| status.get("script"))
            .and_then(|script| script.get("skill_id")),
        Some(&serde_json::json!("motherbrain.releasePersistentBumper"))
    );
    assert_eq!(
        debug
            .get("possession_recovery")
            .and_then(|debug| debug.get("observed_motion"))
            .and_then(|motion| motion.get("linear_displacement_m")),
        Some(&serde_json::json!(0.0))
    );
}

#[tokio::test]
async fn possessor_submits_atomic_escape_when_local_withdrawal_ends_still_bumped() {
    let careful_mode_attempts = Arc::new(AtomicUsize::new(0));
    let motion_attempts = Arc::new(AtomicUsize::new(0));
    let body = ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc::clone(&motion_attempts),
        careful_mode_attempts: Arc::clone(&careful_mode_attempts),
        bump_escape_commands: Arc::new(Mutex::new(Vec::new())),
        stop_attempts: Arc::new(AtomicUsize::new(0)),
        clear_attempts: Arc::new(AtomicUsize::new(0)),
        bump_active: true,
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);
    runner.possession_recovery.latch = Some(SafetyLatchKind::Bump);
    runner.possession_recovery.hazard_generation = 42;
    runner.possession_recovery.phase = PossessionRecoveryPhase::WaitingForSensorClear;
    runner.possession_recovery.active_since_ms = wall_time_ms();
    runner.possession_recovery.last_command_ms = 0;
    runner.possession_recovery.brainstem_reflex_observed = true;
    runner.possession_recovery.last_reflex_outcome = Some(ContactWithdrawalOutcome::Completed);

    let _ = runner.tick_slow_manual().await.unwrap();
    let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(careful_mode_attempts.load(Ordering::SeqCst), 0);
    assert_eq!(motion_attempts.load(Ordering::SeqCst), 1);
    assert!(matches!(
        tick.chosen_action,
        Some(ActionPrimitive::Go { .. })
    ));
    assert!(snapshot
        .action_debug
        .as_ref()
        .and_then(|debug| debug.get("why_not_moving"))
        .and_then(|reason| reason.as_str())
        .is_some_and(|reason| reason.contains("foreground Lua")));
}

#[test]
fn lua_cliff_recovery_emits_only_generation_bound_reverse_escape() {
    let commands = Arc::new(Mutex::new(Vec::new()));
    let mut cockpit = ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc::new(AtomicUsize::new(0)),
        careful_mode_attempts: Arc::new(AtomicUsize::new(0)),
        bump_escape_commands: Arc::clone(&commands),
        stop_attempts: Arc::new(AtomicUsize::new(0)),
        clear_attempts: Arc::new(AtomicUsize::new(0)),
        bump_active: false,
    };
    let status = CockpitStatus {
        raw: serde_json::json!({
            "uptime_ms": 1_000,
            "current_runtime_state": "idle",
            "oi_mode": "safe",
            "safety_tripped": true,
            "safety_latch_kind": "cliff",
            "safety_hazard_generation": 77,
            "create_sensors": {
                "complete_packet_count": 1,
                "last_complete_packet_timestamp_ms": 1_000,
                "cliff_front_left": true,
                "charging_state": 0
            }
        })
        .to_string(),
    }
    .summary();
    let request = SkillRequest {
        skill_id: SkillId::RetreatFromCliff,
        ..SkillRequest::default()
    };
    let mut state = EmbodiedLuaDriverState::default();
    let mut driver = RealLuaOrganDriver {
        cockpit: &mut cockpit,
        request: &request,
        status: &status,
        home_base_contact: false,
        state: &mut state,
        command_sent: false,
    };
    let mut now = Now::blank(1_000, BodySense::default());
    now.body.flags.cliff_front_left = true;
    let result = driver.poll(
        &HostOperation::Retreat {
            hazard: HazardKind::Cliff,
            distance_m: 0.1,
        },
        OperationContext {
            operation_id: 1,
            child_id: 0,
            first_poll: true,
            elapsed_ms: 0,
            now_ms: 1_000,
            primitive_ttl_ms: 250,
        },
        &now,
        &EventBatch {
            since_seq: 0,
            oldest_seq: 0,
            next_seq: 0,
            dropped_before_seq: 0,
            events: Vec::new(),
        },
    );
    assert!(matches!(result, OrganPoll::Pending { .. }));
    assert_eq!(
        commands.lock().unwrap().as_slice(),
        &[(SafetyLatchKind::Cliff, 77, -100, 0, 250)]
    );
}

#[test]
fn lua_bump_recovery_cannot_suppress_imu_absolute_hazard() {
    let commands = Arc::new(Mutex::new(Vec::new()));
    let mut cockpit = ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc::new(AtomicUsize::new(0)),
        careful_mode_attempts: Arc::new(AtomicUsize::new(0)),
        bump_escape_commands: Arc::clone(&commands),
        stop_attempts: Arc::new(AtomicUsize::new(0)),
        clear_attempts: Arc::new(AtomicUsize::new(0)),
        bump_active: true,
    };
    let status = CockpitStatus {
        raw: serde_json::json!({
            "safety_tripped": true,
            "safety_latch_kind": "bump",
            "safety_hazard_generation": 78,
            "imu": {"health": "ok", "impact_score_mm_s2": 20_000}
        })
        .to_string(),
    }
    .summary();
    let request = SkillRequest {
        skill_id: SkillId::ReleasePersistentBumper,
        ..SkillRequest::default()
    };
    let mut state = EmbodiedLuaDriverState::default();
    let mut driver = RealLuaOrganDriver {
        cockpit: &mut cockpit,
        request: &request,
        status: &status,
        home_base_contact: false,
        state: &mut state,
        command_sent: false,
    };
    let mut now = Now::blank(1_000, BodySense::default());
    now.body.flags.bump_left = true;
    let result = driver.poll(
        &HostOperation::Retreat {
            hazard: HazardKind::BumperFront,
            distance_m: 0.1,
        },
        OperationContext {
            operation_id: 1,
            child_id: 0,
            first_poll: true,
            elapsed_ms: 0,
            now_ms: 1_000,
            primitive_ttl_ms: 250,
        },
        &now,
        &EventBatch {
            since_seq: 0,
            oldest_seq: 0,
            next_seq: 0,
            dropped_before_seq: 0,
            events: Vec::new(),
        },
    );
    assert!(matches!(
        result,
        OrganPoll::Failed(SkillFailure {
            outcome: SkillOutcome::SafetyPreempted,
            ..
        })
    ));
    assert!(commands.lock().unwrap().is_empty());
}

#[tokio::test]
async fn real_robot_slow_runner_renews_bounded_bump_escape_each_observation_tick() {
    let bump_escape_attempts = Arc::new(AtomicUsize::new(0));
    let bump_escape_commands = Arc::new(Mutex::new(Vec::new()));
    let stop_attempts = Arc::new(AtomicUsize::new(0));
    let body = ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc::clone(&bump_escape_attempts),
        careful_mode_attempts: Arc::new(AtomicUsize::new(0)),
        bump_escape_commands,
        stop_attempts,
        clear_attempts: Arc::new(AtomicUsize::new(0)),
        bump_active: true,
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);

    let (_first_snapshot, _first_tick) = runner.tick_slow_manual().await.unwrap();
    let (_second_snapshot, _second_tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(bump_escape_attempts.load(Ordering::SeqCst), 1);
    std::thread::sleep(Duration::from_millis(260));
    let (second_snapshot, second_tick) = runner.tick_slow_manual().await.unwrap();
    assert_eq!(bump_escape_attempts.load(Ordering::SeqCst), 2);
    assert_eq!(
        second_tick.chosen_action,
        Some(ActionPrimitive::Go {
            intensity: -0.25,
            duration_ms: POSSESSION_ESCAPE_TTL_MS as TimeMs,
        })
    );
    assert!(second_snapshot
        .action_debug
        .as_ref()
        .and_then(|debug| debug.get("why_not_moving"))
        .and_then(|reason| reason.as_str())
        .is_some_and(|reason| reason.contains("foreground Lua")));
}

#[tokio::test]
async fn real_robot_slow_runner_bounds_lua_bump_recovery_instead_of_eagerly_stopping() {
    let bump_escape_attempts = Arc::new(AtomicUsize::new(0));
    let bump_escape_commands = Arc::new(Mutex::new(Vec::new()));
    let stop_attempts = Arc::new(AtomicUsize::new(0));
    let body = ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc::clone(&bump_escape_attempts),
        careful_mode_attempts: Arc::new(AtomicUsize::new(0)),
        bump_escape_commands,
        stop_attempts: Arc::clone(&stop_attempts),
        clear_attempts: Arc::new(AtomicUsize::new(0)),
        bump_active: true,
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);
    runner.possession_recovery.latch = Some(SafetyLatchKind::Bump);
    runner.possession_recovery.hazard_generation = 42;
    runner.possession_recovery.phase = PossessionRecoveryPhase::WaitingForSensorClear;
    runner.possession_recovery.active_since_ms =
        wall_time_ms().saturating_sub(POSSESSION_RECOVERY_STUCK_AFTER_MS + 1);
    runner.possession_recovery.command_attempts = 12;

    let request = runner
        .possession_recovery_skill_request(&EventBatch {
            since_seq: 0,
            oldest_seq: 0,
            next_seq: 0,
            dropped_before_seq: 0,
            events: Vec::new(),
        })
        .expect("Lua recovery request");
    assert_eq!(request.skill_id, SkillId::ReleasePersistentBumper);
    assert_eq!(
        request.maximum_duration_ms,
        POSSESSION_RECOVERY_STUCK_AFTER_MS
    );
    assert_eq!(bump_escape_attempts.load(Ordering::SeqCst), 0);
    assert_eq!(stop_attempts.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn real_robot_slow_runner_does_not_escape_after_momentary_bump_clears() {
    let bump_escape_attempts = Arc::new(AtomicUsize::new(0));
    let bump_escape_commands = Arc::new(Mutex::new(Vec::new()));
    let body = ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc::clone(&bump_escape_attempts),
        careful_mode_attempts: Arc::new(AtomicUsize::new(0)),
        bump_escape_commands: Arc::clone(&bump_escape_commands),
        stop_attempts: Arc::new(AtomicUsize::new(0)),
        clear_attempts: Arc::new(AtomicUsize::new(0)),
        bump_active: false,
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);

    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(bump_escape_attempts.load(Ordering::SeqCst), 0);
    assert!(bump_escape_commands.lock().unwrap().is_empty());
    assert_ne!(
        tick.chosen_action,
        Some(ActionPrimitive::Go {
            intensity: 0.3,
            duration_ms: 700,
        })
    );
}

#[tokio::test]
async fn real_robot_slow_runner_never_imagines_turn_without_submitting_it() {
    let bump_escape_attempts = Arc::new(AtomicUsize::new(0));
    let body = ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc::clone(&bump_escape_attempts),
        careful_mode_attempts: Arc::new(AtomicUsize::new(0)),
        bump_escape_commands: Arc::new(Mutex::new(Vec::new())),
        stop_attempts: Arc::new(AtomicUsize::new(0)),
        clear_attempts: Arc::new(AtomicUsize::new(0)),
        bump_active: true,
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);
    runner.possession_recovery.latch = Some(SafetyLatchKind::Bump);
    runner.possession_recovery.hazard_generation = 42;
    runner.possession_recovery.phase = PossessionRecoveryPhase::Escaping;
    runner.possession_recovery.command_attempts = 1;

    let _ = runner.tick_slow_manual().await.unwrap();
    let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(bump_escape_attempts.load(Ordering::SeqCst), 1);
    assert!(matches!(
        tick.chosen_action,
        Some(ActionPrimitive::Go { .. })
    ));
    assert!(snapshot
        .action_debug
        .as_ref()
        .and_then(|debug| debug.get("why_not_moving"))
        .and_then(|reason| reason.as_str())
        .is_some_and(|reason| reason.contains("foreground Lua")));
}

#[tokio::test]
async fn real_robot_slow_runner_clears_bump_only_after_escape_finishes() {
    let stop_attempts = Arc::new(AtomicUsize::new(0));
    let clear_attempts = Arc::new(AtomicUsize::new(0));
    let body = ActiveBumpRecoveryCockpit {
        bump_escape_attempts: Arc::new(AtomicUsize::new(0)),
        careful_mode_attempts: Arc::new(AtomicUsize::new(0)),
        bump_escape_commands: Arc::new(Mutex::new(Vec::new())),
        stop_attempts: Arc::clone(&stop_attempts),
        clear_attempts: Arc::clone(&clear_attempts),
        bump_active: false,
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);
    runner.possession_recovery.latch = Some(SafetyLatchKind::Bump);
    runner.possession_recovery.hazard_generation = 42;
    runner.possession_recovery.phase = PossessionRecoveryPhase::Escaping;
    runner.possession_recovery.command_attempts = 1;

    let _ = runner.tick_slow_manual().await.unwrap();
    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(stop_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(clear_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(runner.possession_recovery.latch, None);
    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
}

#[tokio::test]
async fn real_robot_slow_runner_applies_executive_motion_when_explicitly_authorized() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), StubRuntime)
        .with_brainstem_interface(serde_json::json!({
            "verbs": ["status", "get_events", "cmd_vel"]
        }))
        .with_autonomous_motion(true);

    let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(motor_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(
        motors.lock().unwrap().as_slice(),
        &[MotorCommand {
            forward: 0.05,
            turn: 0.0,
        }]
    );
    assert_eq!(
        snapshot
            .action_debug
            .as_ref()
            .and_then(|debug| debug.get("autonomous_hardware_gate"))
            .and_then(|value| value.as_bool()),
        Some(true)
    );
    assert_eq!(
        tick.frame
            .now
            .extensions
            .get("brainstem.events")
            .and_then(|extension| extension.get("events"))
            .and_then(|events| events.as_array())
            .map(Vec::len),
        Some(0)
    );
    assert_eq!(
        tick.frame.now.extensions["brainstem.interface"]["underlying_body_private"],
        serde_json::json!(true)
    );
}

#[tokio::test]
async fn real_robot_slow_runner_waits_for_runtime_tick_without_backoff() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let tick_attempts = Arc::new(AtomicUsize::new(0));
    let runtime = SlowRuntime {
        tick_attempts: Arc::clone(&tick_attempts),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), runtime);
    runner.tick_ms = 25;

    let (_first_snapshot, first_tick) = runner.tick_slow_manual().await.unwrap();
    let (_second_snapshot, second_tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(tick_attempts.load(Ordering::SeqCst), 2);
    assert_eq!(motor_attempts.load(Ordering::SeqCst), 2);
    assert_eq!(
        motors.lock().unwrap().as_slice(),
        &[MotorCommand::stop(), MotorCommand::stop()]
    );
    assert!(first_tick.frame.notes.is_empty());
    assert!(second_tick.frame.notes.is_empty());
}

#[tokio::test]
async fn real_robot_slow_runner_recovers_history_gap_by_stopping() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let event_polls = Arc::new(AtomicUsize::new(0));
    let body = HistoryGapCockpit {
        inner: CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        },
        event_polls: Arc::clone(&event_polls),
        gap_poll: 1,
    };
    let runtime = SlowRuntime {
        tick_attempts: Arc::new(AtomicUsize::new(0)),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), runtime);

    runner.tick_slow_manual().await.unwrap();
    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(event_polls.load(Ordering::SeqCst), 2);
    assert_eq!(
        tick.frame.now.extensions["brainstem.events"]["dropped_before_seq"],
        serde_json::json!(3)
    );
    assert!(motor_attempts.load(Ordering::SeqCst) >= 2);
    assert_eq!(motors.lock().unwrap().last(), Some(&MotorCommand::stop()));
}

#[tokio::test]
async fn real_robot_slow_runner_recovers_motion_safety_poll_history_gap_by_stopping() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let event_polls = Arc::new(AtomicUsize::new(0));
    let body = HistoryGapCockpit {
        inner: CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        },
        event_polls: Arc::clone(&event_polls),
        gap_poll: 1,
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);

    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(
        tick.chosen_action,
        Some(ActionPrimitive::Go {
            intensity: 0.2,
            duration_ms: 100,
        })
    );
    assert_eq!(event_polls.load(Ordering::SeqCst), 2);
    assert!(motor_attempts.load(Ordering::SeqCst) >= 2);
    assert_eq!(motors.lock().unwrap().last(), Some(&MotorCommand::stop()));
}

#[tokio::test]
async fn real_robot_slow_runner_recovers_motion_stop_events_by_stopping() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let event_polls = Arc::new(AtomicUsize::new(0));
    let body = MotionStopEventsCockpit {
        inner: CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        },
        event_polls: Arc::clone(&event_polls),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);

    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(
        tick.chosen_action,
        Some(ActionPrimitive::Go {
            intensity: 0.2,
            duration_ms: 100,
        })
    );
    assert_eq!(event_polls.load(Ordering::SeqCst), 2);
    assert!(motor_attempts.load(Ordering::SeqCst) >= 2);
    assert_eq!(motors.lock().unwrap().last(), Some(&MotorCommand::stop()));

    let (recovery_snapshot, _recovery_tick) = runner.tick_slow_manual().await.unwrap();
    assert_eq!(
        recovery_snapshot
            .action_debug
            .as_ref()
            .and_then(|debug| debug.get("possession_recovery"))
            .and_then(|debug| debug.get("latched")),
        Some(&serde_json::json!("Bump"))
    );
}

#[tokio::test]
async fn real_robot_slow_runner_treats_command_rejected_as_motion_feedback() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let rejection_attempts = Arc::new(AtomicUsize::new(0));
    let body = RejectingMotionCockpit {
        inner: CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        },
        rejection_attempts: Arc::clone(&rejection_attempts),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);

    let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
    assert_eq!(rejection_attempts.load(Ordering::SeqCst), 1);
    let motors = motors.lock().unwrap();
    assert_eq!(motors.last(), Some(&MotorCommand::stop()));
    assert_eq!(
            snapshot
                .action_debug
                .as_ref()
                .and_then(|debug| debug.get("why_not_moving"))
                .and_then(|reason| reason.as_str()),
            Some(
                "brainstem rejected motion command #42: stale_sequence; pausing motion retries for 1000 ms"
            )
        );
    assert_eq!(
        snapshot
            .action_debug
            .as_ref()
            .and_then(|debug| debug.get("motor_applied"))
            .and_then(|value| value.as_bool()),
        Some(false)
    );
}

#[tokio::test]
async fn real_robot_slow_runner_pauses_motion_after_command_rejection() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let rejection_attempts = Arc::new(AtomicUsize::new(0));
    let body = RejectingMotionCockpit {
        inner: CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        },
        rejection_attempts: Arc::clone(&rejection_attempts),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);

    let (_first_snapshot, _first_tick) = runner.tick_slow_manual().await.unwrap();
    let (second_snapshot, second_tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(second_tick.chosen_action, Some(ActionPrimitive::Stop));
    assert_eq!(rejection_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(motors.lock().unwrap().last(), Some(&MotorCommand::stop()));
    assert!(second_snapshot
        .action_debug
        .as_ref()
        .and_then(|debug| debug.get("why_not_moving"))
        .and_then(|reason| reason.as_str())
        .is_some_and(|reason| reason.contains("pausing motion retries")));
    assert_eq!(
        second_snapshot
            .action_debug
            .as_ref()
            .and_then(|debug| debug.get("motion_rejection"))
            .and_then(|debug| debug.get("count")),
        Some(&serde_json::json!(1))
    );
}

#[tokio::test]
async fn real_robot_slow_runner_latches_stuck_after_repeated_command_rejections() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let rejection_attempts = Arc::new(AtomicUsize::new(0));
    let body = RejectingMotionCockpit {
        inner: CountingCockpit {
            motor_attempts: Arc::clone(&motor_attempts),
            motors: Arc::clone(&motors),
            body: BodySense {
                last_update_ms: 100,
                ..BodySense::default()
            },
        },
        rejection_attempts: Arc::clone(&rejection_attempts),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, body, Vec::new(), StubRuntime)
        .with_autonomous_motion(true);
    let now_ms = wall_time_ms();
    runner.motion_rejection = MotionRejectionState {
        first_ms: now_ms,
        last_ms: now_ms,
        latest_command_id: 41,
        latest_reason: Some("busy".to_string()),
        count: MOTION_REJECTION_STUCK_AFTER - 1,
        ..MotionRejectionState::default()
    };

    let (snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
    assert!(snapshot
        .action_debug
        .as_ref()
        .and_then(|debug| debug.get("why_not_moving"))
        .and_then(|reason| reason.as_str())
        .is_some_and(|reason| reason.contains("operator intervention needed")));
    assert_eq!(
        snapshot
            .action_debug
            .as_ref()
            .and_then(|debug| debug.get("motion_rejection"))
            .and_then(|debug| debug.get("stuck")),
        Some(&serde_json::json!(true))
    );
}

struct ManualRuntime;

#[async_trait::async_trait]
impl RuntimeLoop for ManualRuntime {
    async fn tick(
        &mut self,
        mut now: Now,
        _latent: ExperienceLatent,
        _futures: Vec<FuturePrediction>,
    ) -> Result<RuntimeTick> {
        let input = ReignInput {
            id: Uuid::new_v4(),
            issued_at_ms: now.t_ms,
            expires_at_ms: now.t_ms + 300,
            source: ReignSource::WebRemote,
            mode: ReignMode::Direct,
            command: pete_actions::ReignCommand::Go {
                intensity: 0.50,
                duration_ms: 300,
            },
            priority: 1.0,
            note: None,
        };
        now.reign.latest = Some(input.clone());
        let action = input.command.to_action().unwrap();
        let experience =
            Experience::new("test", "test", Vec::new(), Vec::new(), now.t_ms, now.t_ms);
        Ok(RuntimeTick {
            frame: ExperienceFrame {
                id: Uuid::new_v4(),
                t_ms: now.t_ms,
                now,
                sensations: Vec::new(),
                impressions: Vec::new(),
                experiences: vec![experience.clone()],
                z: Some(ExperienceLatent::default()),
                chosen_action: Some(action.clone()),
                conscious_command: None,
                reign_input: Some(input),
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
            experience,
            chosen_action: Some(action),
            skill_request: None,
            skill_status: None,
            recall: RecallBundle::default(),
            llm: LlmTickResult::default(),
            combobulation: None,
            inline_learning: InlineLearningTickStatus::default(),
        })
    }
}

struct QueueOnlyRuntime {
    queue: Arc<Mutex<ReignQueue>>,
    tick_attempts: Arc<AtomicUsize>,
}

#[async_trait::async_trait]
impl RuntimeLoop for QueueOnlyRuntime {
    async fn tick(
        &mut self,
        _now: Now,
        _latent: ExperienceLatent,
        _futures: Vec<FuturePrediction>,
    ) -> Result<RuntimeTick> {
        self.tick_attempts.fetch_add(1, Ordering::SeqCst);
        anyhow::bail!("slow direct hardware should bypass runtime tick")
    }

    fn reign_sense(&self, now_ms: TimeMs) -> Result<ReignSense> {
        Ok(self.queue.lock().unwrap().sense(now_ms))
    }
}

#[tokio::test]
async fn real_robot_slow_runner_applies_only_clamped_webremote_direct_motor() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let mut runner =
        RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), ManualRuntime);

    let (_snapshot, _tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(motor_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(
        motors.lock().unwrap().as_slice(),
        &[MotorCommand {
            forward: 0.05,
            turn: 0.0
        }]
    );
}

#[tokio::test]
async fn real_robot_slow_direct_webremote_bypasses_slow_runtime_tick() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let mut body_sense = BodySense {
        last_update_ms: 100,
        ..BodySense::default()
    };
    body_sense.cliff_sensors.front_left = 0.96;
    body_sense.cliff_sensors.front_right = 0.82;
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: body_sense,
    };
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    queue.lock().unwrap().push(ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: 100,
        expires_at_ms: wall_time_ms().saturating_add(500),
        source: ReignSource::WebRemote,
        mode: ReignMode::Direct,
        command: ReignCommand::Go {
            intensity: 0.50,
            duration_ms: 300,
        },
        priority: 1.0,
        note: None,
    });
    let tick_attempts = Arc::new(AtomicUsize::new(0));
    let runtime = QueueOnlyRuntime {
        queue,
        tick_attempts: Arc::clone(&tick_attempts),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), runtime);

    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(tick_attempts.load(Ordering::SeqCst), 0);
    assert_eq!(motor_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(
        motors.lock().unwrap().as_slice(),
        &[MotorCommand {
            forward: 0.05,
            turn: 0.0
        }]
    );
    assert_eq!(
        tick.frame
            .now
            .extensions
            .get("action.motion_bridge")
            .and_then(|value| value.get("runtime_bypassed"))
            .and_then(|value| value.as_bool()),
        Some(true)
    );
}

#[tokio::test]
async fn real_robot_slow_direct_webremote_stops_locally_while_charging() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: BodySense {
            charging: true,
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    queue.lock().unwrap().push(ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: 100,
        expires_at_ms: wall_time_ms().saturating_add(500),
        source: ReignSource::WebRemote,
        mode: ReignMode::Direct,
        command: ReignCommand::Go {
            intensity: 0.50,
            duration_ms: 300,
        },
        priority: 1.0,
        note: None,
    });
    let tick_attempts = Arc::new(AtomicUsize::new(0));
    let runtime = QueueOnlyRuntime {
        queue,
        tick_attempts: Arc::clone(&tick_attempts),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), runtime);

    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(tick_attempts.load(Ordering::SeqCst), 0);
    assert_eq!(motor_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(motors.lock().unwrap().as_slice(), &[MotorCommand::stop()]);
    assert_eq!(
        tick.frame
            .now
            .extensions
            .get("action.motion_bridge")
            .and_then(|value| value.get("why_not_moving"))
            .and_then(|value| value.as_str()),
        Some("charging active")
    );
}

#[tokio::test]
async fn real_robot_slow_direct_gamepad_bypasses_slow_runtime_tick() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    queue.lock().unwrap().push(ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: 100,
        expires_at_ms: wall_time_ms().saturating_add(500),
        source: ReignSource::Gamepad,
        mode: ReignMode::Direct,
        command: ReignCommand::Drive {
            forward: 0.50,
            turn: -0.50,
            duration_ms: 300,
        },
        priority: 1.0,
        note: None,
    });
    let tick_attempts = Arc::new(AtomicUsize::new(0));
    let runtime = QueueOnlyRuntime {
        queue,
        tick_attempts: Arc::clone(&tick_attempts),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), runtime);

    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(tick_attempts.load(Ordering::SeqCst), 0);
    assert_eq!(motor_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(
        motors.lock().unwrap().as_slice(),
        &[MotorCommand {
            forward: 0.05,
            turn: -0.5
        }]
    );
    assert!(matches!(
        tick.frame.reign_input.as_ref().map(|input| &input.source),
        Some(ReignSource::Gamepad)
    ));
}

#[tokio::test]
async fn real_robot_slow_direct_webremote_chirp_bypasses_runtime_without_motor() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    queue.lock().unwrap().push(ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: 100,
        expires_at_ms: wall_time_ms().saturating_add(500),
        source: ReignSource::WebRemote,
        mode: ReignMode::Direct,
        command: ReignCommand::Chirp {
            pattern: ChirpPattern::Confirm,
        },
        priority: 1.0,
        note: None,
    });
    let tick_attempts = Arc::new(AtomicUsize::new(0));
    let runtime = QueueOnlyRuntime {
        queue,
        tick_attempts: Arc::clone(&tick_attempts),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), runtime);

    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(tick_attempts.load(Ordering::SeqCst), 0);
    assert_eq!(motor_attempts.load(Ordering::SeqCst), 0);
    assert!(motors.lock().unwrap().is_empty());
    assert!(matches!(
        tick.chosen_action,
        Some(ActionPrimitive::Chirp {
            pattern: ChirpPattern::Confirm
        })
    ));
    assert!(matches!(
        tick.frame.reign_input.as_ref().map(|input| &input.command),
        Some(ReignCommand::Chirp {
            pattern: ChirpPattern::Confirm
        })
    ));
}

#[tokio::test]
async fn real_robot_slow_direct_webremote_speak_bypasses_runtime_without_motor() {
    let motor_attempts = Arc::new(AtomicUsize::new(0));
    let motors = Arc::new(Mutex::new(Vec::new()));
    let body = CountingCockpit {
        motor_attempts: Arc::clone(&motor_attempts),
        motors: Arc::clone(&motors),
        body: BodySense {
            last_update_ms: 100,
            ..BodySense::default()
        },
    };
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    queue.lock().unwrap().push(ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: 100,
        expires_at_ms: wall_time_ms().saturating_add(500),
        source: ReignSource::WebRemote,
        mode: ReignMode::Direct,
        command: ReignCommand::Speak {
            text: "hello from reign".to_string(),
        },
        priority: 1.0,
        note: None,
    });
    let tick_attempts = Arc::new(AtomicUsize::new(0));
    let runtime = QueueOnlyRuntime {
        queue,
        tick_attempts: Arc::clone(&tick_attempts),
    };
    let mut runner = RealRobotRunner::new(RobotMode::Slow, Box::new(body), Vec::new(), runtime);

    let (_snapshot, tick) = runner.tick_slow_manual().await.unwrap();

    assert_eq!(tick_attempts.load(Ordering::SeqCst), 0);
    assert_eq!(motor_attempts.load(Ordering::SeqCst), 0);
    assert!(motors.lock().unwrap().is_empty());
    assert!(matches!(
        tick.chosen_action,
        Some(ActionPrimitive::Speak { ref text }) if text == "hello from reign"
    ));
    assert!(matches!(
        tick.frame.reign_input.as_ref().map(|input| &input.command),
        Some(ReignCommand::Speak { text }) if text == "hello from reign"
    ));
}

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

#[tokio::test]
async fn goal_shadow_records_evaluation_without_replacing_baseline() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-goal-shadow-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        FixedConductor::new(ActionPrimitive::Stop),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    )
    .with_action_selector_mode(ActionSelectorMode::GoalShadow);
    let tick = runtime
        .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
    let decision = serde_json::from_value::<ActionSelectionDecision>(
        tick.frame.now.extensions["action_selector"].clone(),
    )
    .unwrap();
    assert_eq!(decision.mode, ActionSelectorMode::GoalShadow);
    assert!(decision.selected_goal.is_none());
    assert!(decision.shadow_selected_goal.is_some());
    assert!(tick.frame.now.extensions.contains_key("goal_system"));
}

#[tokio::test]
async fn goal_mode_executes_goal_behavior_and_publishes_homeostatic_drives() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-goal-mode-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        FixedConductor::new(ActionPrimitive::Stop),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    )
    .with_action_selector_mode(ActionSelectorMode::Goal);
    let mut now = idle_now(100);
    now.body.battery_level = 0.05;
    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    let decision = serde_json::from_value::<ActionSelectionDecision>(
        tick.frame.now.extensions["action_selector"].clone(),
    )
    .unwrap();
    assert_eq!(decision.selected_goal.as_deref(), Some("seek_charger"));
    assert!(matches!(
        decision.selected_behavior.as_deref(),
        Some("inspect_for_charger" | "systematic_charger_search")
    ));
    assert_ne!(tick.chosen_action, Some(ActionPrimitive::Dock));
    assert!(tick.frame.now.drives.battery_hunger > 0.5);
}

#[tokio::test]
async fn sleep_quiesces_possessor_goals_and_emits_a_durable_snapshot() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-sleep-quiescence-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        FixedConductor::new(ActionPrimitive::Explore {
            style: ExploreStyle::Wander,
            duration_ms: 1_000,
        }),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    )
    .with_action_selector_mode(ActionSelectorMode::Goal);
    let mut now = idle_now(100);
    now.body.charging = true;
    now.extensions
        .insert("sleep.request".to_string(), serde_json::Value::Bool(true));

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
    assert!(tick.skill_request.is_none());
    let sleep: SleepSnapshot =
        serde_json::from_value(tick.frame.now.extensions["sleep"].clone()).unwrap();
    assert_eq!(sleep.phase, SleepPhase::Preparing);
    let goals: pete_conductor::GoalCycle =
        serde_json::from_value(tick.frame.now.extensions["goal_system"].clone()).unwrap();
    assert!(goals.selection.selected_goal.is_none());
    assert_eq!(
        goals.selection.reason,
        "deliberative goals quiesced for sleep"
    );
}

#[test]
fn executed_goal_behavior_strengthens_approach_progress_from_canonical_target() {
    std::thread::Builder::new()
        .name("semantic-outcome-test".to_string())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async {
                    let ledger = JsonlLedger::new("/tmp/pete-runtime-semantic-outcome-test");
                    let memory = InMemoryExperienceStore::new();
                    let recall = memory.clone();
                    let mut runtime = Box::new(
                        MinimalRuntime::new(
                            ledger,
                            memory,
                            recall,
                            FixedConductor::new(ActionPrimitive::Stop),
                            SimpleSafety::default(),
                            pete_llm::NoopLlmAgent,
                        )
                        .with_action_selector_mode(ActionSelectorMode::Goal),
                    );
                    let charger_now = |t_ms: u64, distance_m: f32| {
                        let mut now = idle_now(t_ms);
                        now.body.battery_level = 0.12;
                        now.objects.observations.push(pete_now::ObjectObservation {
                            label: "dock".to_string(),
                            class: ObjectClass::Charger,
                            bearing_rad: 0.0,
                            distance_m: Some(distance_m),
                            confidence: 0.95,
                            source: pete_now::ObjectObservationSource::Sim,
                        });
                        now
                    };
                    let first = runtime
                        .tick(
                            charger_now(100, 1.0),
                            ExperienceLatent::default(),
                            Vec::new(),
                        )
                        .await
                        .unwrap();
                    assert_eq!(
                        first.frame.now.extensions["action_selector"]["selected_behavior"],
                        serde_json::Value::String("approach_charger".to_string())
                    );
                    drop(first);
                    let second = runtime
                        .tick(
                            charger_now(200, 0.7),
                            ExperienceLatent::default(),
                            Vec::new(),
                        )
                        .await
                        .unwrap();
                    assert_eq!(
                        second.frame.now.extensions["goal_system.outcome"]
                            ["executed_goal_behavior"],
                        serde_json::Value::String("approach_charger".to_string())
                    );
                    drop(second);
                    let third = runtime
                        .tick(
                            charger_now(300, 0.7),
                            ExperienceLatent::default(),
                            Vec::new(),
                        )
                        .await
                        .unwrap();
                    assert!(third
                        .frame
                        .now
                        .world
                        .semantic
                        .relations
                        .values()
                        .any(|relation| {
                            relation.subject
                                == SemanticNodeRef::Behavior(SemanticBehaviorId(
                                    "approach_charger".to_string(),
                                ))
                                && relation.predicate == SemanticPredicate::Predicts
                                && relation
                                    .supporting_evidence
                                    .iter()
                                    .any(|evidence| evidence.source == "runtime.action_outcome")
                        }));
                });
        })
        .unwrap()
        .join()
        .unwrap();
}

#[tokio::test]
async fn shadow_goal_behavior_cannot_claim_the_executed_baseline_outcome() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-semantic-shadow-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        FixedConductor::new(ActionPrimitive::Stop),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    )
    .with_action_selector_mode(ActionSelectorMode::GoalShadow);
    let charger_now = |t_ms: u64, distance_m: f32| {
        let mut now = idle_now(t_ms);
        now.body.battery_level = 0.12;
        now.objects.observations.push(pete_now::ObjectObservation {
            label: "dock".to_string(),
            class: ObjectClass::Charger,
            bearing_rad: 0.0,
            distance_m: Some(distance_m),
            confidence: 0.95,
            source: pete_now::ObjectObservationSource::Sim,
        });
        now
    };
    for (t_ms, distance_m) in [(100, 1.0), (200, 0.7), (300, 0.7)] {
        let tick = runtime
            .tick(
                charger_now(t_ms, distance_m),
                ExperienceLatent::default(),
                Vec::new(),
            )
            .await
            .unwrap();
        assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
        assert!(
            tick.frame.now.extensions["goal_system.outcome"]["executed_goal_behavior"].is_null()
        );
        assert!(!tick
            .frame
            .now
            .world
            .semantic
            .relations
            .values()
            .any(|relation| relation
                .supporting_evidence
                .iter()
                .any(|evidence| evidence.source == "runtime.action_outcome")));
    }
}

#[test]
fn semantic_outcomes_retain_canonical_target_identity() {
    let world_with_distances = |t_ms: u64, first: f32, second: f32| {
        let mut world = WorldModelSnapshot {
            t_ms,
            ..WorldModelSnapshot::default()
        };
        for (id, distance_m) in [("charger:a", first), ("charger:b", second)] {
            world.entities.insert(
                EntityId(id.to_string()),
                pete_now::WorldEntity {
                    id: EntityId(id.to_string()),
                    kind: pete_now::WorldEntityKind::Charger,
                    distance_m: Some(distance_m),
                    distance_meta: Some(BeliefMeta {
                        confidence: 1.0,
                        observed_at_ms: t_ms,
                        valid_at_ms: t_ms,
                        freshness: Freshness::Current,
                        ..BeliefMeta::default()
                    }),
                    ..pete_now::WorldEntity::default()
                },
            );
        }
        world
    };
    let behavior = pete_conductor::BehaviorDecision {
        goal_id: pete_conductor::GoalId::new("seek_charger"),
        behavior_id: "approach_charger".to_string(),
        action: ActionPrimitive::Approach {
            target: ApproachTarget::Charger,
        },
        affordance: pete_conductor::Affordance {
            target: Some(EntityId("charger:a".to_string())),
            ..pete_conductor::Affordance::default()
        },
    };
    let mut tracker = SemanticOutcomeTracker::default();
    tracker.remember(&world_with_distances(100, 1.0, 2.0), Some(&behavior));

    // Charger B becoming closer is not evidence that the action advanced A.
    tracker.observe_outcome(&world_with_distances(200, 1.0, 0.5));
    assert!(tracker.take_pending().is_empty());

    tracker.observe_outcome(&world_with_distances(300, 0.7, 0.4));
    let evidence = tracker.take_pending();
    assert_eq!(evidence.len(), 1);
    assert_eq!(
        evidence[0].subject,
        SemanticNodeRef::Behavior(SemanticBehaviorId("approach_charger".to_string()))
    );
}

#[tokio::test]
async fn goal_mode_assist_is_only_an_affordance_bias_but_direct_still_overrides() {
    let build_runtime = |path: &'static str| {
        let ledger = JsonlLedger::new(path);
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        MinimalRuntime::new(
            ledger,
            memory,
            recall,
            FixedConductor::new(ActionPrimitive::Stop),
            SimpleSafety::default(),
            pete_llm::NoopLlmAgent,
        )
        .with_action_selector_mode(ActionSelectorMode::Goal)
    };
    let mut assisted = build_runtime("/tmp/pete-runtime-goal-assist-test");
    assisted.reign_queue.lock().unwrap().push(test_reign_input(
        100,
        ReignMode::Assist,
        ReignCommand::Dock,
        2_000,
    ));
    let assisted_tick = assisted
        .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    assert_ne!(assisted_tick.chosen_action, Some(ActionPrimitive::Dock));

    let mut direct = build_runtime("/tmp/pete-runtime-goal-direct-test");
    direct.reign_queue.lock().unwrap().push(test_reign_input(
        100,
        ReignMode::Direct,
        ReignCommand::Dock,
        2_000,
    ));
    let direct_tick = direct
        .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    assert_eq!(direct_tick.chosen_action, Some(ActionPrimitive::Dock));
}

#[test]
fn memory_backed_baseline_action_is_a_selector_candidate_context() {
    let mut now = idle_now(100);
    mark_corrected_map_trusted(&mut now);
    now.memory.place_danger = 0.9;
    now.memory.nearby_best_safe_direction_rad = Some(-0.8);
    let memory_action = ActionPrimitive::Turn {
        direction: TurnDir::Right,
        intensity: 0.5,
        duration_ms: 1_000,
    };
    let default_action = ActionPrimitive::Go {
        intensity: 0.15,
        duration_ms: 1_000,
    };

    assert!(memory_navigation_candidate_context(&now, &memory_action));
    assert!(!memory_navigation_candidate_context(&now, &default_action));
}

#[tokio::test]
async fn direct_reign_overrides_model_assisted_selector() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-reign-model-assisted-test");
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
    let command = ReignCommand::Turn {
        direction: TurnDir::Right,
        intensity: 0.5,
        duration_ms: 500,
    };
    runtime.reign_queue.lock().unwrap().push(test_reign_input(
        100,
        ReignMode::Direct,
        command.clone(),
        2_000,
    ));
    let mut now = idle_now(100);
    now.drives.curiosity = 1.0;

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    assert_eq!(tick.chosen_action, command.to_action());
    let decision = tick
        .frame
        .now
        .extensions
        .get("action_selector")
        .cloned()
        .and_then(|value| serde_json::from_value::<ActionSelectionDecision>(value).ok())
        .unwrap();
    assert_eq!(decision.selected_action, command.to_action());
    assert!(tick
        .frame
        .reign_outcome
        .as_ref()
        .map(|outcome| outcome.accepted_by_conductor)
        .unwrap_or(false));
}

#[tokio::test]
async fn assist_reign_overrides_model_assisted_selector_immediately() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-assist-reign-model-assisted-test");
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
    let command = ReignCommand::Turn {
        direction: TurnDir::Right,
        intensity: 0.5,
        duration_ms: 500,
    };
    runtime.reign_queue.lock().unwrap().push(test_reign_input(
        100,
        ReignMode::Assist,
        command.clone(),
        2_000,
    ));
    let mut now = idle_now(100);
    now.drives.curiosity = 1.0;

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    assert_eq!(tick.chosen_action, command.to_action());
    let decision = tick
        .frame
        .now
        .extensions
        .get("action_selector")
        .cloned()
        .and_then(|value| serde_json::from_value::<ActionSelectionDecision>(value).ok())
        .unwrap();
    assert_eq!(decision.selected_action, command.to_action());
    assert!(tick
        .frame
        .reign_outcome
        .as_ref()
        .map(|outcome| outcome.accepted_by_conductor)
        .unwrap_or(false));
}

#[tokio::test]
async fn observe_or_suggest_reign_does_not_mechanically_override_selector() {
    for mode in [ReignMode::ObserveOnly, ReignMode::Suggest] {
        let ledger = JsonlLedger::new(format!("/tmp/pete-runtime-non-driving-reign-{mode:?}"));
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            FixedConductor::new(ActionPrimitive::Stop),
            SimpleSafety::default(),
            pete_llm::NoopLlmAgent,
        );
        let command = ReignCommand::Turn {
            direction: TurnDir::Right,
            intensity: 0.5,
            duration_ms: 500,
        };
        runtime
            .reign_queue
            .lock()
            .unwrap()
            .push(test_reign_input(100, mode, command, 2_000));

        let tick = runtime
            .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();

        assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
        assert!(tick.frame.reign_input.is_some());
        assert!(!tick
            .frame
            .reign_outcome
            .as_ref()
            .map(|outcome| outcome.accepted_by_conductor)
            .unwrap_or(true));
    }
}

#[tokio::test]
async fn stop_reign_becomes_now_event_and_chosen_action() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-reign-stop-test");
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
    runtime.reign_queue.lock().unwrap().push(test_reign_input(
        100,
        ReignMode::Direct,
        ReignCommand::Stop,
        2_000,
    ));
    let mut body = BodySense::default();
    body.last_update_ms = 100;
    let now = Now::blank(100, body);

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    assert!(tick.frame.now.reign.active);
    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
    assert!(tick
        .frame
        .sensations
        .iter()
        .any(|sensation| sensation.kind == "reign.command"));
    assert!(tick
        .frame
        .reign_input
        .as_ref()
        .map(|input| matches!(input.command, ReignCommand::Stop))
        .unwrap_or(false));
    assert!(tick
        .frame
        .reign_outcome
        .as_ref()
        .map(|outcome| outcome.accepted_by_conductor)
        .unwrap_or(false));
}

#[test]
fn expired_reign_disappears_from_sense() {
    let mut queue = ReignQueue::default();
    queue.push(test_reign_input(
        100,
        ReignMode::Direct,
        ReignCommand::Stop,
        100,
    ));

    queue.drain_expired(250);
    let sense = queue.sense(250);

    assert!(!sense.active);
    assert!(sense.latest.is_none());
    assert_eq!(sense.pending_count, 0);
}

#[test]
fn clear_marks_reign_sense_for_event_extraction() {
    let mut queue = ReignQueue::default();
    queue.push(test_reign_input(
        100,
        ReignMode::Direct,
        ReignCommand::Stop,
        1_000,
    ));

    queue.clear();
    let sense = queue.sense(150);

    assert!(!sense.active);
    assert!(sense.latest.is_none());
    assert_eq!(sense.clear_sequence, 1);
}

#[tokio::test]
async fn safety_veto_beats_direct_go_reign_at_cliff() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-reign-safety-test");
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
    runtime.reign_queue.lock().unwrap().push(test_reign_input(
        100,
        ReignMode::Direct,
        ReignCommand::Go {
            intensity: 0.5,
            duration_ms: 500,
        },
        2_000,
    ));
    let mut body = BodySense::default();
    body.flags.cliff_front_left = true;
    body.last_update_ms = 100;
    let now = Now::blank(100, body);

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    assert_eq!(
        tick.chosen_action,
        Some(ActionPrimitive::Go {
            intensity: 0.5,
            duration_ms: 500,
        })
    );
    assert!(tick
        .frame
        .reign_outcome
        .as_ref()
        .map(|outcome| outcome.vetoed_by_safety)
        .unwrap_or(false));
    assert!(tick
        .frame
        .notes
        .iter()
        .any(|note| note.contains("Safety vetoed")));
    let motor_gate = tick.frame.now.extensions.get("motor_gate").unwrap();
    assert_eq!(
        serde_json::from_value::<MotorCommand>(motor_gate["final_motor"].clone()).unwrap(),
        MotorCommand::stop()
    );
    assert_eq!(motor_gate["safety_reason"], "cliff");
}

#[tokio::test]
async fn sim_runner_writes_frames_and_transitions() {
    let root = test_ledger_root("sim-runner-writes");
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), SimpleConductor::default());
    let (world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(10).await.unwrap();

    let frames = ledger.recent(20).await.unwrap();
    let transitions = read_transitions(&root);
    assert!(frames.len() >= 10);
    assert!(transitions.len() >= 9);
    assert!(transitions.iter().any(|transition| {
        transition.before.body.odometry.x_m != transition.after.body.odometry.x_m
            || transition.before.body.odometry.y_m != transition.after.body.odometry.y_m
            || transition.before.body.odometry.heading_rad
                != transition.after.body.odometry.heading_rad
    }));
}

#[tokio::test]
async fn tick_records_erased_behavior_runs() {
    let root = test_ledger_root("runtime-behavior-runs");
    let ledger = JsonlLedger::new(&root);
    let mut runtime = test_runtime(ledger, FixedConductor::new(ActionPrimitive::Stop));

    let tick = runtime
        .tick(
            Now::blank(100, test_body(1.0, 1.0, 0.8, 100)),
            ExperienceLatent::default(),
            Vec::new(),
        )
        .await
        .unwrap();

    for behavior_id in [
        "danger",
        "charge",
        "future",
        "action_value",
        "eye_next",
        "ear_next",
    ] {
        assert!(
            tick.frame
                .behavior_runs
                .iter()
                .any(|run| run.behavior_id == behavior_id),
            "missing behavior run for {behavior_id}"
        );
    }

    let _ = fs::remove_dir_all(root);
}

#[tokio::test]
async fn tick_runs_bump_event_script_and_records_safety_trace() {
    let root = test_ledger_root("runtime-bump-event-script");
    let ledger = JsonlLedger::new(&root);
    let mut runtime = test_runtime(
        ledger,
        FixedConductor::new(ActionPrimitive::Go {
            intensity: 0.3,
            duration_ms: 500,
        }),
    );
    let mut body = test_body(1.0, 1.0, 0.05, 100);
    body.flags.bump_left = true;
    let now = Now::blank(100, body);

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    assert_eq!(
        tick.chosen_action,
        Some(ActionPrimitive::Go {
            intensity: 0.3,
            duration_ms: 500
        })
    );
    assert!(tick
        .frame
        .now
        .extensions
        .get("safety.vetoed")
        .and_then(|value| value.as_bool())
        .unwrap_or(false));
    assert!(tick
        .frame
        .behavior_runs
        .iter()
        .any(|run| run.behavior_id == "event_bump" && run.regime == BehaviorRegime::ShadowTrain));
    let sequence = tick
        .frame
        .now
        .extensions
        .get("event_scripts")
        .and_then(|value| value.get("bump"))
        .cloned()
        .and_then(|value| serde_json::from_value::<SafeScriptSequence>(value).ok())
        .unwrap();
    assert_eq!(sequence.actions.len(), 5);
    assert!(matches!(
        sequence.actions.first().map(|action| &action.requested),
        Some(EventScriptAction::Chirp {
            pattern: ChirpPattern::Warning
        })
    ));
    assert!(matches!(
        sequence.actions.get(1).map(|action| &action.requested),
        Some(EventScriptAction::Say { .. } | EventScriptAction::Song { .. })
    ));
    assert!(sequence.actions.last().unwrap().vetoed);

    let _ = fs::remove_dir_all(root);
}

#[test]
fn target_extractors_create_danger_charge_future_and_action_value_samples() {
    let action = ActionPrimitive::Go {
        intensity: 0.4,
        duration_ms: 1_000,
    };
    let mut before = Now::blank(100, test_body(1.0, 1.0, 0.0, 100));
    before.body.battery_level = 0.5;
    let mut after = before.clone();
    after.t_ms = 200;
    after.body.last_update_ms = 200;
    after.body.flags.bump_left = true;
    after.body.battery_level = 0.55;
    after.body.charging = true;
    after.eye.frames = vec![vec![0.25, 0.5, 0.75]];
    after.ear.features = vec![vec![0.2, 0.4], vec![0.6, 0.8]];
    let transition = ExperienceTransition {
        id: Uuid::new_v4(),
        before_frame_id: Uuid::new_v4(),
        before,
        before_z: ExperienceLatent {
            t_ms: 100,
            z: vec![0.1, 0.2],
            reconstruction_error: 0.0,
            prediction_error: 0.0,
            confidence: 0.8,
        },
        action: Some(action.clone()),
        predicted_futures: Vec::new(),
        after,
        after_z: ExperienceLatent {
            t_ms: 200,
            z: vec![0.3, 0.4],
            reconstruction_error: 0.0,
            prediction_error: 0.0,
            confidence: 0.9,
        },
        reward: Reward { value: 0.25 },
        surprise: SurpriseSense {
            total: 0.6,
            prediction_error: 0.1,
            ..SurpriseSense::default()
        },
        created_at_ms: 200,
    };

    let danger = DangerTargetExtractor.extract(&transition).unwrap().unwrap();
    assert_eq!(danger.source, TrainingSource::WorldOutcome);
    assert_eq!(danger.expected.bump_risk, 1.0);

    let charge = ChargeTargetExtractor.extract(&transition).unwrap().unwrap();
    assert_eq!(charge.expected.charge_probability, 1.0);
    assert!(charge.expected.expected_battery_delta > 0.0);

    let future = FutureTargetExtractor { offset_ms: 1_000 }
        .extract(&transition)
        .unwrap()
        .unwrap();
    assert_eq!(future.input.action, action);
    assert_eq!(future.expected.predicted_z, vec![0.3, 0.4]);

    let action_value = ActionValueTargetExtractor
        .extract(&transition)
        .unwrap()
        .unwrap();
    assert_eq!(action_value.source, TrainingSource::WorldOutcome);
    assert!((action_value.expected.value - 0.18).abs() < 0.0001);
    assert_eq!(action_value.expected.confidence, 1.0);

    let eye_next = EyeNextTargetExtractor { offset_ms: 100 }
        .extract(&transition)
        .unwrap()
        .unwrap();
    assert_eq!(eye_next.source, TrainingSource::WorldOutcome);
    assert_eq!(eye_next.expected.width, 64);
    assert_eq!(eye_next.expected.height, 48);
    assert_eq!(eye_next.expected.rgb.len(), 64 * 48 * 3);

    let ear_next = EarNextTargetExtractor { offset_ms: 100 }
        .extract(&transition)
        .unwrap()
        .unwrap();
    assert_eq!(ear_next.source, TrainingSource::WorldOutcome);
    assert_eq!(ear_next.expected.features, vec![0.2, 0.4, 0.6, 0.8]);
    assert!(ear_next.expected.pcm.is_empty());
}

#[test]
fn ear_next_target_extractor_skips_missing_ear_frame() {
    let before = Now::blank(100, test_body(1.0, 1.0, 0.0, 100));
    let mut after = before.clone();
    after.t_ms = 200;
    let transition = ExperienceTransition {
        id: Uuid::new_v4(),
        before_frame_id: Uuid::new_v4(),
        before,
        before_z: ExperienceLatent {
            t_ms: 100,
            z: vec![0.1, 0.2],
            reconstruction_error: 0.0,
            prediction_error: 0.0,
            confidence: 0.8,
        },
        action: Some(ActionPrimitive::Stop),
        predicted_futures: Vec::new(),
        after,
        after_z: ExperienceLatent::default(),
        reward: Reward { value: 0.0 },
        surprise: SurpriseSense::default(),
        created_at_ms: 200,
    };

    let sample = EarNextTargetExtractor { offset_ms: 100 }
        .extract(&transition)
        .unwrap();

    assert!(sample.is_none());
}

#[test]
fn behavior_registry_default_has_all_replaceable_slots() {
    let mut registry = BehaviorRegistry::default();
    let now = Now::blank(100, test_body(1.0, 1.0, 0.0, 100));
    let latent = ExperienceLatent {
        t_ms: 100,
        z: vec![0.0; 4],
        reconstruction_error: 0.0,
        prediction_error: 0.0,
        confidence: 0.8,
    };
    let action = ActionPrimitive::Dock;

    let locomotion = registry
        .locomotion
        .infer(&LocomotionInput::default(), 100)
        .unwrap();
    let danger = registry
        .danger
        .infer(&danger_behavior_input(&now, &latent, Some(&action)), 100)
        .unwrap();
    let charge = registry
        .charge
        .infer(&charge_behavior_input(&now, &latent, Some(&action)), 100)
        .unwrap();
    let future = registry
        .future
        .infer(
            &FutureInput {
                latent: latent.clone(),
                action: action.clone(),
                offset_ms: 1_000,
            },
            100,
        )
        .unwrap();
    let action_value = registry
        .action_value
        .infer(
            &action_value_behavior_input(&now, &latent, Some(&action), None, None),
            100,
        )
        .unwrap();
    let eye_next = registry
        .eye_next
        .infer(
            &eye_next_behavior_input(&now, &latent, Some(&action), 100),
            100,
        )
        .unwrap();
    let ear_next = registry
        .ear_next
        .infer(
            &ear_next_behavior_input(&now, &latent, Some(&action), 100),
            100,
        )
        .unwrap();
    let experience = registry
        .experience
        .infer(&ExperienceBehaviorInput::from_now(&now), 100)
        .unwrap();

    assert_eq!(locomotion.record.behavior_id, "locomotion");
    assert_eq!(experience.record.behavior_id, "experience");
    assert_eq!(danger.record.behavior_id, "danger");
    assert_eq!(charge.record.behavior_id, "charge");
    assert_eq!(future.record.behavior_id, "future");
    assert_eq!(action_value.record.behavior_id, "action_value");
    assert_eq!(eye_next.record.behavior_id, "eye_next");
    assert_eq!(ear_next.record.behavior_id, "ear_next");
    assert!(locomotion.record.hardcoded_output.is_some());
    assert!(experience.record.hardcoded_output.is_some());
    assert!(danger.record.hardcoded_output.is_some());
    assert!(charge.record.hardcoded_output.is_some());
    assert!(future.record.hardcoded_output.is_some());
    assert!(action_value.record.hardcoded_output.is_some());
    assert!(eye_next.record.hardcoded_output.is_some());
    assert!(ear_next.record.hardcoded_output.is_some());
}

#[test]
fn action_value_hardcoded_regime_returns_hardcoded_output() {
    let now = Now::blank(100, test_body(1.0, 1.0, 0.2, 100));
    let latent = ExperienceLatent {
        t_ms: 100,
        z: vec![0.0; 4],
        confidence: 0.8,
        ..ExperienceLatent::default()
    };
    let input =
        action_value_behavior_input(&now, &latent, Some(&ActionPrimitive::Dock), None, None);
    let mut behavior = action_value_behavior(
        BehaviorRegime::Hardcoded,
        None,
        FallbackPolicy::UseHardcoded,
    );

    let run = behavior.infer(&input, 100).unwrap();

    assert!(run.record.hardcoded_output.is_some());
    assert!(run.record.model_output.is_none());
    assert_eq!(run.record.selected_output, run.record.hardcoded_output);
}

#[test]
fn action_value_shadow_infer_records_model_and_selects_hardcoded() {
    let now = Now::blank(100, test_body(1.0, 1.0, 0.2, 100));
    let latent = ExperienceLatent {
        t_ms: 100,
        z: vec![0.0; 4],
        confidence: 0.8,
        ..ExperienceLatent::default()
    };
    let input =
        action_value_behavior_input(&now, &latent, Some(&ActionPrimitive::Dock), None, None);
    let trainer = ActionValueNetTrainer::new(input.input.flat_features().len());
    let mut behavior = action_value_behavior(
        BehaviorRegime::ShadowInfer,
        Some(trainer),
        FallbackPolicy::UseHardcoded,
    );

    let run = behavior.infer(&input, 100).unwrap();

    assert!(run.record.hardcoded_output.is_some());
    assert!(run.record.model_output.is_some());
    assert_eq!(run.record.selected_output, run.record.hardcoded_output);
}

#[test]
fn action_value_config_with_missing_checkpoint_falls_back_cleanly() {
    let config: BehaviorRegistryConfig = toml::from_str(
        r#"
            [behavior.action_value]
            regime = "shadow_infer"
            hardcoded = "action_value.handcoded"
            model = "action_value.burn.v0"
            checkpoint = "/tmp/pete-missing-action-value-checkpoint"
            fallback = "use_hardcoded"
            "#,
    )
    .unwrap();
    let mut stack = RuntimeModelStack::from_behavior_config(&config).unwrap();
    assert_eq!(
        stack.behaviors.action_value.regime,
        BehaviorRegime::ShadowInfer
    );

    let now = Now::blank(100, test_body(1.0, 1.0, 0.2, 100));
    let latent = ExperienceLatent {
        t_ms: 100,
        z: vec![0.0; 4],
        confidence: 0.8,
        ..ExperienceLatent::default()
    };
    let input =
        action_value_behavior_input(&now, &latent, Some(&ActionPrimitive::Dock), None, None);
    let run = stack
        .behaviors
        .action_value
        .infer(&input, now.t_ms)
        .unwrap();

    assert!(run.record.hardcoded_output.is_some());
    assert!(run.record.model_output.is_none());
}

#[tokio::test]
async fn sim_runner_applies_chosen_action_to_world() {
    let ledger = JsonlLedger::new(test_ledger_root("sim-runner-action-world"));
    let runtime = test_runtime(
        ledger,
        FixedConductor::new(ActionPrimitive::Go {
            intensity: 0.4,
            duration_ms: 1_000,
        }),
    );
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.5, 7));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let snapshot = runner.world.snapshot().await.unwrap();

    assert!(snapshot.body.odometry.x_m > 1.0);
    assert_eq!(runner.tick_count, 1);
}

#[tokio::test]
async fn sim_runner_go_and_explore_send_non_stop_motion_and_change_pose() {
    for (name, action) in [
        (
            "go",
            ActionPrimitive::Go {
                intensity: 0.4,
                duration_ms: 1_000,
            },
        ),
        (
            "explore",
            ActionPrimitive::Explore {
                style: ExploreStyle::RandomWalk,
                duration_ms: 1_000,
            },
        ),
    ] {
        let ledger = JsonlLedger::new(test_ledger_root(&format!("sim-runner-{name}-motor-bridge")));
        let runtime = test_runtime(ledger, FixedConductor::new(action.clone()));
        let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.5, 7));
        let mut runner = SimRunner::new(runtime, world, motors);
        let start = runner.world.body();
        let mut saw_non_zero_final_motor = false;
        let expected_selected_action = match &action {
            ActionPrimitive::Explore { duration_ms, .. } => ActionPrimitive::Drive {
                forward: 0.2,
                turn: 0.1,
                duration_ms: *duration_ms,
            },
            _ => action.clone(),
        };

        runner
            .run_steps_observing_ticks(5, |snapshot, tick| {
                let final_motor = final_motor_from_tick(tick);
                if !is_near_zero_motor(final_motor) {
                    saw_non_zero_final_motor = true;
                }
                assert_eq!(
                    snapshot.final_selected_action,
                    Some(expected_selected_action.clone())
                );
            })
            .await
            .unwrap();

        let end = runner.world.body();
        let delta = movement_delta_m(&start, &end);
        assert!(
            delta > 0.005,
            "{name} should move the simulated body, delta was {delta}"
        );
        assert!(saw_non_zero_final_motor, "{name} final motor was zero");
        assert!(
            !matches!(
                runner.world.last_motion_sent(),
                Some(MotionCommand::Stop) | None
            ),
            "{name} did not send non-stop motion to sim"
        );
    }
}

#[tokio::test]
async fn sim_runner_reaches_charger_gets_positive_reward() {
    let root = test_ledger_root("sim-runner-charger-reward");
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(
        ledger,
        FixedConductor::new(ActionPrimitive::Go {
            intensity: 0.4,
            duration_ms: 1_000,
        }),
    );
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    let mut body = test_body(1.0, 1.0, 0.2, 7);
    body.battery_level = 0.2;
    world.set_body(body);
    world.add_object(SimObject::charger("charger", "charger", 1.38, 1.0, 0.18));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(2).await.unwrap();
    let transitions = read_transitions(&root);

    let transition = transitions.last().unwrap();
    assert!(transition.after.body.charging);
    assert!(transition.reward.value > 0.0);
    assert!(transition.surprise.total > 0.0);
}

#[tokio::test]
async fn sim_runner_collision_sets_bump_and_negative_reward() {
    let root = test_ledger_root("sim-runner-collision-reward");
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(
        ledger,
        FixedConductor::new(ActionPrimitive::Go {
            intensity: 0.4,
            duration_ms: 1_000,
        }),
    );
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.8, 7));
    world.add_object(SimObject::obstacle("box", "box", 1.31, 1.0, 0.1));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(2).await.unwrap();
    let transitions = read_transitions(&root);

    let transition = transitions.last().unwrap();
    assert!(transition.after.body.flags.bump_left || transition.after.body.flags.bump_right);
    assert!(transition.reward.value < 0.0);
    assert!(transition.surprise.total > 0.0);
}

#[tokio::test]
async fn sim_runner_resets_dead_uncharging_battery_and_records_critique() {
    let root = test_ledger_root("sim-runner-dead-battery-reset");
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(
        ledger.clone(),
        FixedConductor::new(ActionPrimitive::Go {
            intensity: 0.4,
            duration_ms: 1_000,
        }),
    );
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.0, 7));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let snapshot = runner.world.snapshot().await.unwrap();
    let frames = ledger.recent(5).await.unwrap();
    let frame = frames.last().unwrap();

    assert_eq!(snapshot.body.battery_level, 1.0);
    assert!(!snapshot.body.charging);
    assert_eq!(snapshot.body.odometry.x_m, 2.0);
    assert_eq!(snapshot.body.odometry.y_m, 2.0);
    assert!(frame.llm_teaching.iter().any(|teaching| teaching
        .critique
        .as_deref()
        .is_some_and(|critique| { critique.contains("Dead battery away from the charger") })));
    assert!(frame
        .notes
        .iter()
        .any(|note| note.contains("VirtualDeadBattery")));

    let _ = fs::remove_dir_all(root);
}

#[tokio::test]
async fn sim_runner_gives_stuck_body_recovery_time_before_reset() {
    let root = test_ledger_root("sim-runner-stuck-recovery-time");
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(
        ledger.clone(),
        FixedConductor::new(ActionPrimitive::Go {
            intensity: 0.4,
            duration_ms: 1_000,
        }),
    );
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    let mut body = test_body(0.2, 0.2, 1.0, 7);
    body.odometry.heading_rad = std::f32::consts::PI;
    world.set_body(body);
    let mut runner = SimRunner::new(runtime, world, motors);

    runner
        .run_steps(STUCK_LOW_DISPLACEMENT_TICKS + 2)
        .await
        .unwrap();
    let snapshot = runner.world.snapshot().await.unwrap();
    let frames = ledger.recent(10).await.unwrap();

    assert_ne!(snapshot.body.odometry.x_m, 2.0);
    assert_ne!(snapshot.body.odometry.y_m, 2.0);
    assert!(!frames
        .iter()
        .any(|frame| frame.notes.iter().any(|note| note.contains("VirtualStuck"))));

    let _ = fs::remove_dir_all(root);
}

#[tokio::test]
async fn sim_with_danger_checkpoint_writes_shadow_predictions() {
    let root = test_ledger_root("sim-runner-danger-shadow");
    let checkpoint = danger_checkpoint_root("sim-runner-danger-shadow");
    let action = ActionPrimitive::Go {
        intensity: 0.4,
        duration_ms: 1_000,
    };
    write_test_danger_checkpoint(&checkpoint, action.clone());
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), FixedConductor::new(action))
        .with_models(RuntimeModelStack::with_danger_shadow_checkpoint(&checkpoint).unwrap());
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.8, 7));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let frames = ledger.recent(5).await.unwrap();
    let frame = frames.last().unwrap();

    assert!(frame.now.predictions.danger_model.is_some());
    assert!(frame.now.predictions.danger_hardcoded.is_some());

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[tokio::test]
async fn sim_attaches_fallback_predictions_to_embodied_experience() {
    let root = test_ledger_root("sim-runner-embodied-predictions");
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), FixedConductor::new(ActionPrimitive::Stop));
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.8, 7));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let frames = ledger.recent(5).await.unwrap();
    let experience = frames.last().unwrap().experiences.last().unwrap();

    assert!(frames.last().unwrap().z.is_some());
    assert!(experience
        .predictions
        .iter()
        .any(|prediction| prediction.text.starts_with("hazard:")));
    assert!(experience
        .predictions
        .iter()
        .any(|prediction| prediction.text.starts_with("uncertainty:")));

    let _ = fs::remove_dir_all(root);
}

#[tokio::test]
async fn danger_shadow_prediction_does_not_bypass_safety() {
    let root = test_ledger_root("sim-runner-danger-shadow-safety");
    let checkpoint = danger_checkpoint_root("sim-runner-danger-shadow-safety");
    let action = ActionPrimitive::Go {
        intensity: 0.5,
        duration_ms: 500,
    };
    write_test_danger_checkpoint(&checkpoint, action.clone());
    let ledger = JsonlLedger::new(&root);
    let mut runtime = test_runtime(ledger, FixedConductor::new(action.clone()))
        .with_models(RuntimeModelStack::with_danger_shadow_checkpoint(&checkpoint).unwrap());
    let mut body = BodySense::default();
    body.flags.cliff_left = true;
    body.last_update_ms = 100;

    let tick = runtime
        .tick(
            Now::blank(100, body),
            ExperienceLatent::default(),
            Vec::new(),
        )
        .await
        .unwrap();

    assert_eq!(tick.chosen_action, Some(action));
    assert!(tick.frame.now.predictions.danger_model.is_some());
    assert!(tick
        .frame
        .notes
        .iter()
        .any(|note| note.contains("Safety vetoed")));

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[tokio::test]
async fn sim_with_charge_checkpoint_writes_shadow_predictions() {
    let root = test_ledger_root("sim-runner-charge-shadow");
    let checkpoint = danger_checkpoint_root("sim-runner-charge-shadow");
    let action = ActionPrimitive::Dock;
    write_test_charge_checkpoint(&checkpoint, action.clone());
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), FixedConductor::new(action))
        .with_models(RuntimeModelStack::with_charge_shadow_checkpoint(&checkpoint).unwrap());
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.2, 7));
    world.add_object(SimObject::charger("charger", "charger", 1.2, 1.0, 0.18));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let frames = ledger.recent(5).await.unwrap();
    let frame = frames.last().unwrap();

    assert!(frame.now.predictions.charge_model.is_some());
    assert!(frame.now.predictions.charge_hardcoded.is_some());

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[tokio::test]
async fn sim_with_action_value_checkpoint_writes_shadow_predictions() {
    let root = test_ledger_root("sim-runner-action-value-shadow");
    let checkpoint = danger_checkpoint_root("sim-runner-action-value-shadow");
    let action = ActionPrimitive::Dock;
    write_test_action_value_checkpoint(&checkpoint, action.clone());
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), FixedConductor::new(action))
        .with_models(RuntimeModelStack::with_action_value_shadow_checkpoint(&checkpoint).unwrap());
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.2, 7));
    world.add_object(SimObject::charger("charger", "charger", 1.2, 1.0, 0.18));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let frames = ledger.recent(5).await.unwrap();
    let frame = frames.last().unwrap();

    assert!(!frame.now.predictions.action_values_model.is_empty());
    assert!(!frame.now.predictions.action_values_hardcoded.is_empty());

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[tokio::test]
async fn action_value_shadow_mode_does_not_override_conductor() {
    let root = test_ledger_root("sim-runner-action-value-shadow-choice");
    let checkpoint = danger_checkpoint_root("sim-runner-action-value-shadow-choice");
    write_test_action_value_checkpoint(&checkpoint, ActionPrimitive::Dock);
    let chosen = ActionPrimitive::Stop;
    let ledger = JsonlLedger::new(&root);
    let mut runtime = test_runtime(ledger, FixedConductor::new(chosen.clone()))
        .with_models(RuntimeModelStack::with_action_value_shadow_checkpoint(&checkpoint).unwrap());

    let tick = runtime
        .tick(
            Now::blank(100, test_body(1.0, 1.0, 0.8, 100)),
            ExperienceLatent::default(),
            Vec::new(),
        )
        .await
        .unwrap();

    assert_eq!(tick.chosen_action, Some(chosen));
    assert!(!tick.frame.now.predictions.action_values_model.is_empty());

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[tokio::test]
async fn sim_with_future_checkpoint_records_shadow_future_runs() {
    let root = test_ledger_root("sim-runner-future-shadow");
    let checkpoint = danger_checkpoint_root("sim-runner-future-shadow");
    write_test_future_checkpoint(&checkpoint, ActionPrimitive::Stop);
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), FixedConductor::new(ActionPrimitive::Stop))
        .with_models(RuntimeModelStack::with_future_shadow_checkpoint(&checkpoint).unwrap());
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.8, 7));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let frames = ledger.recent(5).await.unwrap();
    let frame = frames.last().unwrap();
    let run = frame
        .behavior_runs
        .iter()
        .find(|run| run.behavior_id == "future" && run.model_json.is_some())
        .unwrap();

    assert_eq!(run.regime, BehaviorRegime::ShadowInfer);
    assert!(run.hardcoded_json.is_some());
    assert!(run.selected_json.is_some());
    assert!(!frame.predicted_futures.is_empty());

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[tokio::test]
async fn inline_world_outcome_learning_observes_transition_sample() {
    let root = test_ledger_root("inline-world-outcome");
    let checkpoint = danger_checkpoint_root("inline-world-outcome");
    write_test_future_checkpoint(&checkpoint, ActionPrimitive::Stop);
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), FixedConductor::new(ActionPrimitive::Stop))
        .with_models(RuntimeModelStack::with_future_shadow_checkpoint(&checkpoint).unwrap())
        .with_inline_learning(InlineLearningConfig {
            mode: InlineLearningMode::WorldOutcome,
            behaviors: InlineLearningBehaviors {
                danger: false,
                charge: false,
                future: true,
                action_value: false,
                eye_next: false,
                ear_next: false,
                experience: false,
            },
            max_train_steps_per_tick: 1,
        });
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.8, 7));
    let mut runner = SimRunner::new(runtime, world, motors);
    let mut observed_samples = 0usize;

    runner
        .run_steps_observing_ticks(3, |_snapshot, tick| {
            observed_samples =
                observed_samples.saturating_add(tick.inline_learning.samples_observed);
        })
        .await
        .unwrap();

    assert!(observed_samples > 0);
    assert!(!read_transitions(&root).is_empty());

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[tokio::test]
async fn disabled_inline_learning_reports_no_weight_updates() {
    let root = test_ledger_root("inline-disabled");
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), FixedConductor::new(ActionPrimitive::Stop));
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.8, 7));
    let mut runner = SimRunner::new(runtime, world, motors);
    let mut statuses = Vec::new();

    runner
        .run_steps_observing_ticks(3, |_snapshot, tick| {
            statuses.push(tick.inline_learning.clone());
        })
        .await
        .unwrap();

    assert!(statuses.iter().all(|status| !status.enabled));
    assert!(statuses
        .iter()
        .all(|status| status.samples_observed == 0 && status.train_steps_used == 0));

    let _ = fs::remove_dir_all(root);
}

#[tokio::test]
async fn sim_with_ear_next_checkpoint_writes_shadow_prediction() {
    let root = test_ledger_root("sim-runner-ear-next-shadow");
    let checkpoint = danger_checkpoint_root("sim-runner-ear-next-shadow");
    let action = ActionPrimitive::Stop;
    write_test_ear_next_checkpoint(&checkpoint, action.clone());
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), FixedConductor::new(action))
        .with_models(RuntimeModelStack::with_ear_next_shadow_checkpoint(&checkpoint).unwrap());
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.8, 7));
    world.add_object(SimObject {
        id: "speaker".to_string(),
        label: "speaker".to_string(),
        kind: pete_sim::SimObjectKind::SoundSource {
            label: "speaker".to_string(),
        },
        x_m: 1.5,
        y_m: 1.2,
        radius_m: 0.12,
        color_rgb: [80, 80, 220],
        emits_sound: true,
        spoken_text: Some("listen to the room".to_string()),
        charge_rate: 0.0,
    });
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let frames = ledger.recent(5).await.unwrap();
    let frame = frames.last().unwrap();

    assert!(frame.now.predictions.ear_next_model.is_some());
    assert!(frame.now.predictions.ear_next_hardcoded.is_some());
    assert!(frame
        .behavior_runs
        .iter()
        .any(|run| run.behavior_id == "ear_next"));

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[tokio::test]
async fn ear_next_shadow_mode_does_not_override_safety_or_action() {
    let root = test_ledger_root("sim-runner-ear-next-shadow-safety");
    let checkpoint = danger_checkpoint_root("sim-runner-ear-next-shadow-safety");
    let action = ActionPrimitive::Go {
        intensity: 0.5,
        duration_ms: 500,
    };
    write_test_ear_next_checkpoint(&checkpoint, action.clone());
    let ledger = JsonlLedger::new(&root);
    let mut runtime = test_runtime(ledger, FixedConductor::new(action.clone()))
        .with_models(RuntimeModelStack::with_ear_next_shadow_checkpoint(&checkpoint).unwrap());
    let mut body = BodySense::default();
    body.flags.cliff_left = true;
    body.last_update_ms = 100;
    let mut now = Now::blank(100, body);
    now.ear.features = vec![vec![0.2, 0.4, 0.6, 0.8]];

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    assert_eq!(tick.chosen_action, Some(action));
    assert!(tick.frame.now.predictions.ear_next_model.is_some());
    assert!(tick
        .frame
        .notes
        .iter()
        .any(|note| note.contains("Safety vetoed")));

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[tokio::test]
async fn sim_with_experience_checkpoint_records_autoencoder_behavior_run() {
    let root = test_ledger_root("sim-runner-experience-shadow");
    let checkpoint = danger_checkpoint_root("sim-runner-experience-shadow");
    write_test_experience_checkpoint(&checkpoint);
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger.clone(), FixedConductor::new(ActionPrimitive::Stop))
        .with_models(RuntimeModelStack::with_experience_shadow_checkpoint(&checkpoint).unwrap());
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    let mut body = test_body(1.0, 1.0, 0.8, 7);
    body.velocity.forward_m_s = 0.1;
    world.set_body(body);
    world.add_object(SimObject {
        id: "speaker".to_string(),
        label: "speaker".to_string(),
        kind: pete_sim::SimObjectKind::SoundSource {
            label: "speaker".to_string(),
        },
        x_m: 1.5,
        y_m: 1.2,
        radius_m: 0.12,
        color_rgb: [80, 80, 220],
        emits_sound: true,
        spoken_text: Some("the walls are awake".to_string()),
        charge_rate: 0.0,
    });
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let frames = ledger.recent(5).await.unwrap();
    let frame = frames.last().unwrap();
    let run = frame
        .behavior_runs
        .iter()
        .find(|run| run.behavior_id == "experience")
        .unwrap();

    assert_eq!(run.regime, BehaviorRegime::ShadowInfer);
    assert!(run.hardcoded_json.is_some());
    assert!(run.model_json.is_some());
    assert!(run.disagreement.unwrap_or_default().is_finite());
    assert!(frame.now.extensions.contains_key("experience.autoencoder"));

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(checkpoint);
}

#[test]
fn missing_experience_checkpoint_returns_no_latent_yet() {
    let config: BehaviorRegistryConfig = toml::from_str(
        r#"
            [behavior.experience]
            regime = "shadow_infer"
            hardcoded = "experience.no_latent_yet"
            model = "experience.autoencoder.v0"
            checkpoint = "/tmp/pete-missing-experience-checkpoint"
            fallback = "use_hardcoded"
            "#,
    )
    .unwrap();
    let mut stack = RuntimeModelStack::from_behavior_config(&config).unwrap();
    let now = Now::blank(100, test_body(1.0, 1.0, 0.8, 100));
    let run = stack
        .behaviors
        .experience
        .infer(&ExperienceBehaviorInput::from_now(&now), now.t_ms)
        .unwrap();

    assert_eq!(run.record.regime, BehaviorRegime::ShadowInfer);
    assert!(run.record.hardcoded_output.is_some());
    assert!(run.record.model_output.is_none());
    assert_eq!(run.chosen, run.record.hardcoded_output.unwrap());
    assert!(run.chosen.latent.z.is_empty());
    assert_eq!(run.chosen.confidence, 0.0);
}

#[tokio::test]
async fn shared_reign_queue_controls_next_sim_tick() {
    let root = test_ledger_root("sim-runner-shared-reign");
    let ledger = JsonlLedger::new(&root);
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    queue.lock().unwrap().push(test_reign_input(
        7,
        ReignMode::Direct,
        ReignCommand::Turn {
            direction: pete_actions::TurnDir::Left,
            intensity: 0.5,
            duration_ms: 500,
        },
        2_000,
    ));
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let runtime = MinimalRuntime::with_reign_queue(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
        queue,
    );
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 0.8, 7));
    let mut runner = SimRunner::new(runtime, world, motors);

    runner.run_steps(1).await.unwrap();
    let snapshot = runner.world.snapshot().await.unwrap();
    let frames = JsonlLedger::new(&root).recent(5).await.unwrap();
    let frame = frames.last().unwrap();

    assert!(snapshot.body.odometry.heading_rad > 0.0);
    assert!(frame.now.reign.active);
    assert!(frame
        .sensations
        .iter()
        .any(|sensation| sensation.kind == "reign.command"));
    assert!(frame.reign_input.is_some());
    assert!(frame
        .reign_outcome
        .as_ref()
        .map(|outcome| outcome.accepted_by_conductor)
        .unwrap_or(false));
}

#[tokio::test]
async fn direct_reign_reverse_drives_sim_while_stuck_active() {
    let root = test_ledger_root("sim-runner-reign-reverse-interrupts-stuck");
    let ledger = JsonlLedger::new(&root);
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    queue.lock().unwrap().push(test_reign_input(
        7,
        ReignMode::Direct,
        ReignCommand::Reverse {
            intensity: 0.5,
            duration_ms: 500,
        },
        2_000,
    ));
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let runtime = MinimalRuntime::with_reign_queue(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
        queue,
    );
    let (mut world, motors) = VirtualWorld::new_with_cockpit(7, arena());
    world.set_body(test_body(1.0, 1.0, 1.0, 7));
    let mut runner = SimRunner::new(runtime, world, motors);
    runner.stuck.active = true;
    runner.stuck.phase = RecoveryPhase::Stop;
    runner.stuck.phase_ticks_remaining = 1;
    runner.stuck.turn_sign = 1.0;

    let mut observed_debug = None;
    runner
        .run_steps_observing(1, |snapshot| {
            observed_debug = snapshot.action_debug.clone();
        })
        .await
        .unwrap();
    let debug = observed_debug.unwrap();
    let motion = debug.get("motion_sent_to_sim").cloned().unwrap();

    let motion = serde_json::from_value::<MotionCommand>(motion.clone())
        .unwrap_or_else(|error| panic!("motion decode failed: {error}; debug={debug}"));
    assert_eq!(motion, MotionCommand::Forward { speed_m_s: -0.5 });
}

#[tokio::test]
async fn column_trap_scenario_recovers_within_budget() {
    let root = test_ledger_root("sim-runner-column-trap-recovery");
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger, SimpleConductor::default());
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ColumnTrap, 7));
    let start = (
        scenario.metadata.body.odometry.x_m,
        scenario.metadata.body.odometry.y_m,
    );
    let mut runner = SimRunner::new(runtime, scenario.world, scenario.motors);
    let mut saw_column = false;
    let mut recovered = false;
    let mut last_skill_status = None;

    runner
        .run_steps_observing_ticks(90, |snapshot, tick| {
            if tick.skill_status.is_some() {
                last_skill_status = tick.skill_status.clone();
            }
            if let Some(stuck) = snapshot
                .extensions
                .iter()
                .find(|extension| extension.name == "sim.stuck")
            {
                saw_column |= stuck.values.get(10).copied() == Some(3.0);
                recovered |= stuck.values.get(7).copied() == Some(1.0);
            }
        })
        .await
        .unwrap();
    let end = runner.world.body();
    let distance = distance_between_points(start, (end.odometry.x_m, end.odometry.y_m));

    assert!(saw_column);
    assert!(recovered, "last Lua skill status was {last_skill_status:?}");
    assert!(distance > 0.10, "distance after recovery was {distance}");
}

#[derive(Clone, Copy, Debug, Default)]
struct TrapRunMetrics {
    collision_frames: usize,
    stuck_frames: usize,
    recovered: bool,
    distance_m: f32,
}

async fn run_column_trap_metrics<C>(ledger_name: &str, conductor: C, steps: usize) -> TrapRunMetrics
where
    C: Conductor + Send + 'static,
{
    let root = test_ledger_root(ledger_name);
    let ledger = JsonlLedger::new(&root);
    let runtime = test_runtime(ledger, conductor);
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ColumnTrap, 7));
    let start = (
        scenario.metadata.body.odometry.x_m,
        scenario.metadata.body.odometry.y_m,
    );
    let mut runner = SimRunner::new(runtime, scenario.world, scenario.motors);
    let mut metrics = TrapRunMetrics::default();

    runner
        .run_steps_observing(steps, |snapshot| {
            let flags = &snapshot.body.flags;
            if flags.wall
                || flags.bump_left
                || flags.bump_right
                || flags.cliff_front_left
                || flags.cliff_front_right
            {
                metrics.collision_frames += 1;
            }
            if let Some(stuck) = snapshot
                .extensions
                .iter()
                .find(|extension| extension.name == "sim.stuck")
            {
                metrics.stuck_frames +=
                    (stuck.values.first().copied().unwrap_or_default() > 0.0) as usize;
                metrics.recovered |= stuck.values.get(7).copied() == Some(1.0);
            }
        })
        .await
        .unwrap();
    let end = runner.world.body();
    metrics.distance_m = distance_between_points(start, (end.odometry.x_m, end.odometry.y_m));
    metrics
}

#[tokio::test]
async fn column_trap_recovery_beats_plain_explore_baseline() {
    let plain = run_column_trap_metrics(
        "sim-runner-column-trap-plain-explore",
        FixedConductor::new(ActionPrimitive::Explore {
            style: ExploreStyle::RandomWalk,
            duration_ms: 1_000,
        }),
        120,
    )
    .await;
    let recovered = run_column_trap_metrics(
        "sim-runner-column-trap-simple-recovery-comparison",
        SimpleConductor::default(),
        120,
    )
    .await;

    assert!(
        recovered.recovered,
        "expected recovery event, got {recovered:?}"
    );
    assert!(
        recovered.collision_frames < plain.collision_frames / 2,
        "recovery should reduce repeated collision frames; plain={plain:?} recovered={recovered:?}"
    );
    assert!(
            recovered.distance_m > plain.distance_m,
            "recovery should make more progress than plain explore; plain={plain:?} recovered={recovered:?}"
        );
    assert!(
        recovered.stuck_frames < plain.stuck_frames,
        "recovery should reduce repeated stuck frames; plain={plain:?} recovered={recovered:?}"
    );
}

#[derive(Clone, Debug)]
struct FixedConductor {
    action: ActionPrimitive,
}

#[derive(Clone, Debug, Default)]
struct FixedRecall {
    bundle: RecallBundle,
}

#[async_trait::async_trait]
impl Recall for FixedRecall {
    async fn recall(&self, _query: RecallQuery) -> Result<RecallBundle> {
        Ok(self.bundle.clone())
    }
}

impl FixedConductor {
    fn new(action: ActionPrimitive) -> Self {
        Self { action }
    }
}

impl Conductor for FixedConductor {
    fn choose(&mut self, _input: ConductorInput) -> Result<ActionPrimitive> {
        Ok(self.action.clone())
    }
}

#[derive(Clone, Debug)]
struct FixedLlmAgent {
    action: ActionPrimitive,
}

#[async_trait::async_trait]
impl LlmAgent for FixedLlmAgent {
    async fn combobulate(
        &mut self,
        _now: &Now,
        _impressions: &[Impression],
        _embodied: Option<&EmbodiedContext>,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
    ) -> Result<Option<Combobulation>> {
        Ok(None)
    }

    async fn maybe_tick(
        &mut self,
        _now: &Now,
        _embodied: Option<&EmbodiedContext>,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
        _awareness_summary: Option<&str>,
    ) -> Result<LlmTickResult> {
        Ok(LlmTickResult {
            sense: pete_now::LlmSense {
                schema_version: 1,
                command_summary: Some("test command".to_string()),
                critique: None,
                confidence: 1.0,
            },
            conscious_command: Some(ConsciousCommand {
                summary: "test command".to_string(),
                action: Some(self.action.clone()),
            }),
            decision: Some(LlmDecision {
                summary: "test command".to_string(),
                action: Some(self.action.clone()),
                confidence: 1.0,
                ..LlmDecision::default()
            }),
            teaching: Vec::new(),
        })
    }

    async fn scientific_review(
        &mut self,
        _request: &LlmReviewRequest,
    ) -> Result<Option<LlmScientificReview>> {
        Ok(None)
    }
}

#[derive(Debug, Default)]
struct SlowAdvisoryAgent;

#[async_trait::async_trait]
impl LlmAgent for SlowAdvisoryAgent {
    async fn combobulate(
        &mut self,
        _now: &Now,
        _impressions: &[Impression],
        _embodied: Option<&EmbodiedContext>,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
    ) -> Result<Option<Combobulation>> {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        Ok(Some(Combobulation {
            summary: "historical doorway hypothesis".to_string(),
            confidence: 0.8,
        }))
    }

    async fn maybe_tick(
        &mut self,
        _now: &Now,
        _embodied: Option<&EmbodiedContext>,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
        _awareness_summary: Option<&str>,
    ) -> Result<LlmTickResult> {
        Ok(LlmTickResult {
            sense: pete_now::LlmSense {
                schema_version: 1,
                critique: Some("test the doorway hypothesis".to_string()),
                confidence: 0.8,
                ..pete_now::LlmSense::default()
            },
            decision: Some(LlmDecision {
                action: Some(ActionPrimitive::Go {
                    intensity: 1.0,
                    duration_ms: 5_000,
                }),
                ..LlmDecision::default()
            }),
            ..LlmTickResult::default()
        })
    }

    async fn scientific_review(
        &mut self,
        _request: &LlmReviewRequest,
    ) -> Result<Option<LlmScientificReview>> {
        Ok(None)
    }
}

fn test_runtime<C>(
    ledger: JsonlLedger,
    conductor: C,
) -> MinimalRuntime<
    JsonlLedger,
    InMemoryExperienceStore,
    InMemoryExperienceStore,
    C,
    SimpleSafety,
    pete_llm::NoopLlmAgent,
>
where
    C: Conductor,
{
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    MinimalRuntime::new(
        ledger,
        memory,
        recall,
        conductor,
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    )
}

async fn finished_cognition_task() -> JoinHandle<Result<(Option<Combobulation>, LlmTickResult)>> {
    let task = tokio::spawn(async {
        Ok((
            None,
            LlmTickResult {
                sense: pete_now::LlmSense {
                    schema_version: 1,
                    command_summary: Some("completed cognition".to_string()),
                    confidence: 1.0,
                    ..pete_now::LlmSense::default()
                },
                ..LlmTickResult::default()
            },
        ))
    });
    tokio::task::yield_now().await;
    assert!(task.is_finished(), "fixture cognition task should be ready");
    task
}

fn cognition_test_inputs() -> (
    EmbodiedContext,
    ExperienceLatent,
    Vec<FuturePrediction>,
    Vec<String>,
) {
    (
        EmbodiedContext::default(),
        ExperienceLatent::default(),
        Vec::new(),
        Vec::new(),
    )
}

#[tokio::test]
async fn llm_command_is_never_granted_control_authority() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-llm-command-action-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let llm_action = ActionPrimitive::Go {
        intensity: 0.3,
        duration_ms: 700,
    };
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        FixedConductor::new(ActionPrimitive::Stop),
        SimpleSafety::default(),
        FixedLlmAgent {
            action: llm_action.clone(),
        },
    );
    let mut now = idle_now(100);
    now.drives.curiosity = 1.0;

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
    assert!(tick.frame.conscious_command.is_none());
    let decision = tick
        .frame
        .now
        .extensions
        .get("action_selector")
        .cloned()
        .and_then(|value| serde_json::from_value::<ActionSelectionDecision>(value).ok())
        .unwrap();
    assert_eq!(decision.selected_action, Some(ActionPrimitive::Stop));
}

#[tokio::test]
async fn accepted_cognition_enters_cooldown_before_scheduling_again() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-cognition-service-health-test");
    let memory = InMemoryExperienceStore::new();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory.clone(),
        memory,
        FixedConductor::new(ActionPrimitive::Stop),
        SimpleSafety::default(),
        FixedLlmAgent {
            action: ActionPrimitive::Stop,
        },
    );

    let first = runtime
        .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    let first_service = &first.frame.now.world.self_model.service_state.services["rich_language"];
    assert!(first_service.available);
    assert!(first_service.busy);
    assert_eq!(first_service.unavailable_reason, None);

    tokio::task::yield_now().await;
    let accepted = runtime
        .tick(idle_now(200), ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    assert!(matches!(
        runtime.cognition.last_outcome.as_ref(),
        Some(CognitionOutcome::Accepted)
    ));
    assert!(runtime.cognition.pending.is_none());
    assert_eq!(runtime.cognition.next_request_at_ms, 2_200);
    let accepted_service =
        &accepted.frame.now.world.self_model.service_state.services["rich_language"];
    assert!(accepted_service.available);
    assert!(!accepted_service.busy);
    assert_eq!(accepted_service.unavailable_reason, None);
    assert!(accepted
        .frame
        .notes
        .iter()
        .any(|note| note == "LlmProviderOutcome: accepted"));

    let cooling_down = runtime
        .tick(idle_now(2_199), ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    assert!(runtime.cognition.pending.is_none());
    assert!(
        !cooling_down
            .frame
            .now
            .world
            .self_model
            .service_state
            .services["rich_language"]
            .busy
    );

    let eligible = runtime
        .tick(idle_now(2_200), ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    assert!(runtime.cognition.pending.is_some());
    assert!(eligible.frame.now.world.self_model.service_state.services["rich_language"].busy);
}

#[tokio::test]
async fn disabled_cognition_is_unavailable_without_scheduling_provider_work() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-disabled-cognition-service-test");
    let memory = InMemoryExperienceStore::new();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory.clone(),
        memory,
        FixedConductor::new(ActionPrimitive::Stop),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    );

    let tick = runtime
        .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    let service = &tick.frame.now.world.self_model.service_state.services["rich_language"];

    assert!(!service.available);
    assert!(!service.busy);
    assert_eq!(
        service.unavailable_reason.as_deref(),
        Some("enhanced language service is disabled")
    );
    assert!(runtime.cognition.pending.is_none());
    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
}

#[tokio::test]
async fn paused_runtime_clock_does_not_expire_completed_cognition() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-cognition-paused-clock-test");
    let mut runtime = test_runtime(ledger, FixedConductor::new(ActionPrimitive::Stop));
    runtime.cognition.pending = Some(PendingLlmCognition {
        snapshot_ref: "paused-frame".to_string(),
        requested_at_ms: 1_000,
        deadline_ms: 1_000 + COGNITION_DEADLINE_MS,
        task: finished_cognition_task().await,
    });
    let now = idle_now(1_000);
    let (embodied, latent, futures, mut notes) = cognition_test_inputs();

    let accepted = runtime
        .advance_cognition(&now, &[], &embodied, &latent, &futures, "", &mut notes)
        .await
        .expect("paused deterministic time should accept a completed provider result");

    assert_eq!(accepted.requested_at_ms, 1_000);
    assert_eq!(accepted.observed_at_ms, 1_000);
    assert!(matches!(
        runtime.cognition.last_outcome.as_ref(),
        Some(CognitionOutcome::Accepted)
    ));
}

#[tokio::test]
async fn replayed_earlier_now_does_not_expire_completed_cognition() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-cognition-replay-clock-test");
    let mut runtime = test_runtime(ledger, FixedConductor::new(ActionPrimitive::Stop));
    runtime.cognition.pending = Some(PendingLlmCognition {
        snapshot_ref: "future-replay-frame".to_string(),
        requested_at_ms: 5_000,
        deadline_ms: 5_000 + COGNITION_DEADLINE_MS,
        task: finished_cognition_task().await,
    });
    let replayed_now = idle_now(500);
    let (embodied, latent, futures, mut notes) = cognition_test_inputs();

    let accepted = runtime
        .advance_cognition(
            &replayed_now,
            &[],
            &embodied,
            &latent,
            &futures,
            "",
            &mut notes,
        )
        .await
        .expect("a backwards replay clock should not invent elapsed runtime time");

    assert_eq!(accepted.requested_at_ms, 5_000);
    assert_eq!(accepted.observed_at_ms, 500);
    assert!(matches!(
        runtime.cognition.last_outcome.as_ref(),
        Some(CognitionOutcome::Accepted)
    ));
}

#[tokio::test]
async fn forward_clock_jump_expires_in_flight_cognition() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-cognition-forward-jump-test");
    let mut runtime = test_runtime(ledger, FixedConductor::new(ActionPrimitive::Stop));
    let (completion_tx, completion_rx) = tokio::sync::oneshot::channel::<()>();
    runtime.cognition.pending = Some(PendingLlmCognition {
        snapshot_ref: "pre-jump-frame".to_string(),
        requested_at_ms: 1_000,
        deadline_ms: 1_000 + COGNITION_DEADLINE_MS,
        task: tokio::spawn(async move {
            completion_rx.await.expect("completion sender retained");
            Ok((None, LlmTickResult::default()))
        }),
    });
    let jumped_now = idle_now(10_000);
    let (embodied, latent, futures, mut notes) = cognition_test_inputs();

    let accepted = runtime
        .advance_cognition(
            &jumped_now,
            &[],
            &embodied,
            &latent,
            &futures,
            "",
            &mut notes,
        )
        .await;

    assert!(accepted.is_none());
    assert!(matches!(
        runtime.cognition.last_outcome.as_ref(),
        Some(CognitionOutcome::Expired)
    ));
    tokio::task::yield_now().await;
    assert!(
        completion_tx.send(()).is_err(),
        "expired task should be aborted"
    );
}

#[tokio::test]
async fn very_slow_cognition_tick_rejects_result_completed_before_late_poll() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-cognition-slow-tick-test");
    let mut runtime = test_runtime(ledger, FixedConductor::new(ActionPrimitive::Stop));
    runtime.cognition.pending = Some(PendingLlmCognition {
        snapshot_ref: "slow-tick-frame".to_string(),
        requested_at_ms: 1_000,
        deadline_ms: 1_000 + COGNITION_DEADLINE_MS,
        task: finished_cognition_task().await,
    });
    let late_now = idle_now(1_000 + COGNITION_DEADLINE_MS + 1);
    let (embodied, latent, futures, mut notes) = cognition_test_inputs();

    let accepted = runtime
        .advance_cognition(&late_now, &[], &embodied, &latent, &futures, "", &mut notes)
        .await;

    assert!(accepted.is_none());
    assert!(matches!(
        runtime.cognition.last_outcome.as_ref(),
        Some(CognitionOutcome::Expired)
    ));
}

#[tokio::test]
async fn cognition_provider_completion_exactly_at_deadline_is_accepted() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-cognition-deadline-boundary-test");
    let mut runtime = test_runtime(ledger, FixedConductor::new(ActionPrimitive::Stop));
    let requested_at_ms = 1_000;
    let deadline_ms = requested_at_ms + COGNITION_DEADLINE_MS;
    runtime.cognition.pending = Some(PendingLlmCognition {
        snapshot_ref: "deadline-frame".to_string(),
        requested_at_ms,
        deadline_ms,
        task: finished_cognition_task().await,
    });
    let deadline_now = idle_now(deadline_ms);
    let (embodied, latent, futures, mut notes) = cognition_test_inputs();

    let accepted = runtime
        .advance_cognition(
            &deadline_now,
            &[],
            &embodied,
            &latent,
            &futures,
            "",
            &mut notes,
        )
        .await
        .expect("the deadline is inclusive");

    assert_eq!(accepted.observed_at_ms, deadline_ms);
    assert!(matches!(
        runtime.cognition.last_outcome.as_ref(),
        Some(CognitionOutcome::Accepted)
    ));
    assert_eq!(
        runtime.cognition.last_sense.command_summary.as_deref(),
        Some("completed cognition")
    );
}

#[tokio::test]
async fn slow_advice_is_retained_as_historical_evidence_only() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-slow-advice-test");
    let memory = InMemoryExperienceStore::new();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory.clone(),
        memory,
        FixedConductor::new(ActionPrimitive::Stop),
        SimpleSafety::default(),
        SlowAdvisoryAgent,
    );
    let mut accepted_tick = None;

    for step in 0..8 {
        let t_ms = 100 + step * 100;
        let mut now = idle_now(t_ms);
        now.body.last_update_ms = t_ms;
        let tick = runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();
        assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
        if tick
            .frame
            .experiences
            .iter()
            .any(|experience| experience.kind == "llm.combobulation")
        {
            accepted_tick = Some(tick);
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    let tick = accepted_tick.expect("500 ms advice should be retained on a later tick");
    let evidence = tick
        .frame
        .sensations
        .iter()
        .find(|sensation| sensation.kind == "llm.combobulation")
        .expect("provenance-bearing advisory sensation");
    assert_eq!(evidence.occurred_at_ms, 100);
    assert!(evidence.observed_at_ms >= 600);
    assert!(evidence
        .payload
        .get("input_snapshot_ref")
        .and_then(serde_json::Value::as_str)
        .is_some());
    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
    assert!(tick.frame.conscious_command.is_none());
}

#[tokio::test]
async fn active_safe_reign_wins_over_llm_action() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-llm-reign-wins-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    let reign_command = ReignCommand::Turn {
        direction: TurnDir::Left,
        intensity: 0.4,
        duration_ms: 500,
    };
    queue.lock().unwrap().push(test_reign_input(
        100,
        ReignMode::Direct,
        reign_command.clone(),
        1_000,
    ));
    let mut runtime = MinimalRuntime::with_reign_queue(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        FixedLlmAgent {
            action: ActionPrimitive::Explore {
                style: ExploreStyle::RandomWalk,
                duration_ms: 1_000,
            },
        },
        queue,
    );
    let now = idle_now(100);

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    let proposal = tick
        .frame
        .now
        .extensions
        .get("llm.action_proposal")
        .cloned()
        .and_then(|value| serde_json::from_value::<LlmActionProposal>(value).ok())
        .unwrap();

    assert_eq!(tick.chosen_action, reign_command.to_action());
    assert!(!proposal.accepted);
    assert_eq!(proposal.ignored_reason.as_deref(), None);
}

#[tokio::test]
async fn llm_action_is_discarded_before_safety_and_cockpit() {
    let ledger = JsonlLedger::new("/tmp/pete-runtime-llm-safety-veto-test");
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let mut runtime = MinimalRuntime::new(
        ledger,
        memory,
        recall,
        FixedConductor::new(ActionPrimitive::Stop),
        SimpleSafety::default(),
        FixedLlmAgent {
            action: ActionPrimitive::Go {
                intensity: 0.3,
                duration_ms: 700,
            },
        },
    );
    runtime
        .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    let provider_input_ref = runtime
        .cognition
        .pending
        .as_ref()
        .expect("provider request in flight")
        .snapshot_ref
        .clone();
    tokio::task::yield_now().await;

    let mut now = idle_now(200);
    now.body.flags.cliff_left = true;

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();
    let proposal = tick
        .frame
        .now
        .extensions
        .get("llm.action_proposal")
        .cloned()
        .and_then(|value| serde_json::from_value::<LlmActionProposal>(value).ok())
        .unwrap();

    assert!(proposal.proposed_action.is_none());
    assert_eq!(
        proposal.advisory_action,
        Some(LlmAdvisoryAction {
            action: ActionPrimitive::Go {
                intensity: 0.3,
                duration_ms: 700,
            },
            source: LlmAdvisoryActionSource::ProviderDecision,
            input_snapshot_ref: provider_input_ref,
            disposition: LlmAdvisoryActionDisposition::DiscardedAtAdvisoryBoundary,
        })
    );
    assert!(!proposal.accepted);
    assert!(!proposal.safety_vetoed);
    assert_eq!(
            proposal.ignored_reason.as_deref(),
            Some(
                "provider suggested Go { intensity: 0.3, duration_ms: 700 }; discarded at advisory boundary"
            )
        );
    assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
    let bridge = tick
        .frame
        .now
        .extensions
        .get("action.motion_bridge")
        .expect("motion bridge telemetry");
    assert!(bridge["llm_action"].is_null());
    assert_eq!(
        bridge["llm_advisory_action"]["disposition"],
        "discarded_at_advisory_boundary"
    );
    assert!(tick.frame.notes.iter().any(|note| {
            note.contains(
                "LlmAdvisoryAction: provider suggested Go { intensity: 0.3, duration_ms: 700 }; discarded at advisory boundary",
            )
        }));
}

fn arena() -> ArenaConfig {
    ArenaConfig {
        width_m: 4.0,
        height_m: 4.0,
    }
}

fn test_body(x_m: f32, y_m: f32, battery_level: f32, last_update_ms: u64) -> BodySense {
    let mut body = BodySense::default();
    body.odometry.x_m = x_m;
    body.odometry.y_m = y_m;
    body.battery_level = battery_level;
    body.last_update_ms = last_update_ms;
    body
}

fn stuck_test_snapshot(x_m: f32, y_m: f32, battery_level: f32) -> WorldSnapshot {
    let mut snapshot = WorldSnapshot::default();
    snapshot.body = test_body(x_m, y_m, battery_level, 100);
    snapshot.range.nearest_m = Some(0.12);
    snapshot.range.beams = vec![0.05, 0.08, 0.10, 0.09, 0.05];
    snapshot.extensions.push(ExtensionSense {
        schema_version: 1,
        name: "sim.world".to_string(),
        values: vec![4.0, 4.0, 0.0],
    });
    snapshot
}

#[test]
fn stuck_detector_uses_rolling_low_displacement_window() {
    let mut detector = StuckRecoveryController::default();
    let action = ActionPrimitive::Explore {
        style: ExploreStyle::RandomWalk,
        duration_ms: 1_000,
    };

    for _ in 0..=STUCK_LOW_DISPLACEMENT_TICKS {
        detector.observe(&stuck_test_snapshot(0.2, 0.2, 1.0), Some(&action));
    }

    let status = detector.status();
    assert!(status.active);
    assert!(status.corner_trap);
    assert_eq!(status.stuck_ticks, STUCK_LOW_DISPLACEMENT_TICKS);
    assert!(status.event_started);
    assert_eq!(status.recovery_attempts, 1);
    assert_eq!(status.duration_ticks, 1);
    assert!(!status.reset_due);
}

#[test]
fn recovered_stuck_event_reports_attempt_and_duration() {
    let mut detector = StuckRecoveryController::default();
    let action = ActionPrimitive::Explore {
        style: ExploreStyle::RandomWalk,
        duration_ms: 1_000,
    };

    for _ in 0..=STUCK_LOW_DISPLACEMENT_TICKS {
        detector.observe(&stuck_test_snapshot(0.2, 0.2, 1.0), Some(&action));
    }
    detector.observe(&stuck_test_snapshot(0.2, 0.2, 1.0), Some(&action));
    detector.observe(&stuck_test_snapshot(0.3, 0.2, 1.0), Some(&action));
    let status = detector.status();
    assert!(!status.active);
    assert!(status.recovered);
    assert_eq!(status.recovery_attempts, 1);
    assert!(status.duration_ticks >= 2);

    let extension = detector.extension(100);
    assert_eq!(extension.values.get(7).copied(), Some(1.0));
    assert_eq!(extension.values.get(11).copied(), Some(1.0));
    assert!(extension.values.get(3).copied().unwrap_or_default() >= 200.0);
}

#[test]
fn repeated_stuck_escalates_recovery_instead_of_resetting() {
    let mut detector = StuckRecoveryController::default();
    let action = ActionPrimitive::Explore {
        style: ExploreStyle::RandomWalk,
        duration_ms: 1_000,
    };

    for _ in 0..=STUCK_LOW_DISPLACEMENT_TICKS {
        detector.observe(&stuck_test_snapshot(0.2, 0.2, 1.0), Some(&action));
    }
    detector.finish_recovery_success();
    detector.clearance_m = Some(0.10);
    detector.recovery_attempts = 1;
    detector.trap_anchor = Some((0.2, 0.2));

    for _ in 0..=STUCK_LOW_DISPLACEMENT_TICKS {
        detector.observe(&stuck_test_snapshot(0.2, 0.2, 1.0), Some(&action));
    }
    let mut snapshot = stuck_test_snapshot(0.2, 0.2, 1.0);
    detector.annotate_snapshot(&mut snapshot, 100);

    let status = detector.status();
    assert!(!status.reset_due);
    assert!(status.active);
    assert_eq!(status.repeated_trap_count, 1);
    let values = &snapshot
        .extensions
        .iter()
        .find(|extension| extension.name == "sim.stuck")
        .unwrap()
        .values;
    assert_eq!(values.get(9).copied(), Some(0.0));
    assert_eq!(values.get(12).copied(), Some(1.0));
}

#[test]
fn dead_battery_state_is_reported_without_starting_recovery() {
    let mut detector = StuckRecoveryController::default();
    let action = ActionPrimitive::Explore {
        style: ExploreStyle::RandomWalk,
        duration_ms: 1_000,
    };

    for _ in 0..=STUCK_LOW_DISPLACEMENT_TICKS {
        detector.observe(&stuck_test_snapshot(0.2, 0.2, 0.0), Some(&action));
    }
    let mut snapshot = stuck_test_snapshot(0.2, 0.2, 0.0);
    detector.annotate_snapshot(&mut snapshot, 100);

    let status = detector.status();
    assert!(status.dead_battery);
    assert!(!status.active);
    let values = &snapshot
        .extensions
        .iter()
        .find(|extension| extension.name == "sim.stuck")
        .unwrap()
        .values;
    assert_eq!(values.get(8).copied(), Some(1.0));
}

#[test]
fn stopped_column_trap_still_triggers_stuck_recovery() {
    let mut detector = StuckRecoveryController::default();
    let action = ActionPrimitive::Stop;

    for _ in 0..=STUCK_LOW_DISPLACEMENT_TICKS {
        detector.observe(&stuck_test_snapshot(2.0, 2.0, 1.0), Some(&action));
    }

    let status = detector.status();
    assert!(status.active);
    assert_eq!(status.trap_kind, TrapKind::Column);
    assert_eq!(status.stuck_ticks, STUCK_LOW_DISPLACEMENT_TICKS);
}

#[test]
fn bump_left_chooses_rightward_escape() {
    let mut body = test_body(1.0, 1.0, 1.0, 100);
    body.flags.bump_left = true;
    let now = Now::blank(100, body);

    assert_eq!(
        hard_safety_action(&now),
        Some(ActionPrimitive::Turn {
            direction: TurnDir::Right,
            intensity: 0.7,
            duration_ms: 1_200
        })
    );
}

#[test]
fn bump_right_chooses_leftward_escape() {
    let mut body = test_body(1.0, 1.0, 1.0, 100);
    body.flags.bump_right = true;
    let now = Now::blank(100, body);

    assert_eq!(
        hard_safety_action(&now),
        Some(ActionPrimitive::Turn {
            direction: TurnDir::Left,
            intensity: 0.7,
            duration_ms: 1_200
        })
    );
}

#[test]
fn every_cliff_sensor_selects_stop_before_hardware_gate() {
    for sensor in ["left", "front_left", "front_right", "right"] {
        let mut body = test_body(1.0, 1.0, 1.0, 100);
        match sensor {
            "left" => body.flags.cliff_left = true,
            "front_left" => body.flags.cliff_front_left = true,
            "front_right" => body.flags.cliff_front_right = true,
            "right" => body.flags.cliff_right = true,
            _ => unreachable!(),
        }
        let now = Now::blank(100, body.clone());

        assert_eq!(
            hard_safety_action(&now),
            Some(ActionPrimitive::Stop),
            "{sensor}"
        );
        assert_eq!(
            real_slow_body_block_reason(&body).as_deref(),
            Some("cliff sensor active"),
            "{sensor}"
        );
    }
}

fn test_ledger_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("pete-{name}-{}", Uuid::new_v4()));
    let _ = fs::remove_dir_all(&root);
    root
}

fn danger_checkpoint_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("pete-{name}-checkpoint-{}", Uuid::new_v4()));
    let _ = fs::remove_dir_all(&root);
    root
}

fn write_test_danger_checkpoint(root: &Path, action: ActionPrimitive) {
    let mut body = test_body(1.0, 1.0, 0.8, 7);
    body.velocity.forward_m_s = 0.05;
    let now = Now::blank(100, body);
    let input = DangerInput::from_parts(Vec::new(), Some(&action), &now);
    let mut trainer = DangerNetTrainer::new(input.flat_features().len());
    trainer
        .train_step(
            &input,
            &pete_experience::DangerTarget {
                bump: 0.2,
                ..pete_experience::DangerTarget::default()
            },
        )
        .unwrap();
    trainer.save_checkpoint(root).unwrap();
}

fn write_test_charge_checkpoint(root: &Path, action: ActionPrimitive) {
    let mut body = test_body(1.0, 1.0, 0.2, 7);
    body.charging = false;
    let now = Now::blank(100, body);
    let input = ChargeInput::from_parts(Vec::new(), Some(&action), &now);
    let mut trainer = ChargeNetTrainer::new(input.flat_features().len());
    trainer
        .train_step(
            &input,
            &pete_experience::ChargeTarget {
                charging_started: 1.0,
                battery_delta: 0.03,
                charging_after: 1.0,
            },
        )
        .unwrap();
    trainer.save_checkpoint(root).unwrap();
}

fn write_test_action_value_checkpoint(root: &Path, action: ActionPrimitive) {
    let mut body = test_body(1.0, 1.0, 0.2, 7);
    body.charging = false;
    let now = Now::blank(100, body);
    let input = ActionValueInput::from_parts(Vec::new(), Some(&action), &now);
    let mut trainer = ActionValueNetTrainer::new(input.flat_features().len());
    trainer
        .train_step(&input, &pete_experience::ActionValueTarget { value: 0.25 })
        .unwrap();
    trainer.save_checkpoint(root).unwrap();
}

fn write_test_future_checkpoint(root: &Path, action: ActionPrimitive) {
    let now = Now::blank(100, test_body(1.0, 1.0, 0.8, 100));
    let latent = ExperienceLatent {
        t_ms: now.t_ms,
        z: Vec::new(),
        reconstruction_error: 0.0,
        prediction_error: 0.0,
        confidence: 0.0,
    };
    let input = FutureInput {
        latent: latent.clone(),
        action,
        offset_ms: 100,
    };
    let mut trainer = FutureNetTrainer::new(input.flat_features().len(), 1);
    trainer.train_step(&input, &[0.0]).unwrap();
    trainer.save_checkpoint(root).unwrap();
}

fn write_test_ear_next_checkpoint(root: &Path, action: ActionPrimitive) {
    let body = test_body(1.0, 1.0, 0.8, 7);
    let mut now = Now::blank(100, body);
    now.ear.features = vec![vec![0.2, 0.4, 0.6, 0.8]];
    let input = EarNextInput::from_parts(Vec::new(), Some(&action), &now, 100);
    let mut trainer = EarNextNetTrainer::new(input.flat_features().len(), 4);
    trainer
        .train_step(
            &input,
            &pete_experience::EarNextTarget {
                features: vec![0.2, 0.4, 0.6, 0.8],
                ..pete_experience::EarNextTarget::default()
            },
        )
        .unwrap();
    trainer.save_checkpoint(root).unwrap();
}

fn write_test_experience_checkpoint(root: &Path) {
    let mut body = test_body(1.0, 1.0, 0.8, 7);
    body.velocity.forward_m_s = 0.1;
    let mut now = Now::blank(100, body);
    now.eye.frames = vec![vec![0.2, 0.4, 0.6, 0.8]];
    now.ear.features = vec![vec![0.2, 0.4, 0.6, 0.8]];
    now.memory.place_familiarity = 0.6;
    now.drives.curiosity = 0.4;
    let input = experience_encode_input_from_now(&now);
    let target = experience_decode_target_from_now(&now);
    let mut trainer =
        ExperienceAutoencoderTrainer::new(input.flat_features().len(), 8, target.feature_lengths());
    trainer.train_step(&input, &target).unwrap();
    trainer.save_checkpoint(root).unwrap();
}

fn read_transitions(root: &Path) -> Vec<ExperienceTransition> {
    let mut out = Vec::new();
    read_transition_paths(root, &mut out);
    out
}

fn read_transition_paths(path: &Path, out: &mut Vec<ExperienceTransition>) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            read_transition_paths(&path, out);
        } else if path.file_name().and_then(|name| name.to_str()) == Some("transitions.jsonl") {
            let Ok(contents) = fs::read_to_string(path) else {
                continue;
            };
            out.extend(
                contents
                    .lines()
                    .filter(|line| !line.trim().is_empty())
                    .filter_map(|line| serde_json::from_str(line).ok()),
            );
        }
    }
}

#[test]
fn robot_initialized_typescript_behavior_emits_bringup_mouth_sequence() {
    let mut behavior = RobotInitializedScriptBehavior;
    let input = RobotInitializedEventInput {
        t_ms: 42,
        mode: "read-only".to_string(),
        body: "mock Create body connected".to_string(),
        battery_percent: Some(100),
        charging: Some(false),
        active_sensors: 2,
        requested_sensors: 3,
        ledger: "data/ledger/test".to_string(),
        tick_ms: 100,
        dashboard: Some("127.0.0.1:3000".to_string()),
        capture: None,
    };

    let output = behavior.infer(&input).unwrap();

    assert!(matches!(
        output.actions.first(),
        Some(EventScriptAction::Song { name }) if name == "bring_up"
    ));
    assert!(output.actions.iter().any(|action| matches!(
        action,
        EventScriptAction::Chirp {
            pattern: ChirpPattern::Confirm
        }
    )));
    assert!(output.actions.iter().any(|action| matches!(
        action,
        EventScriptAction::Say { text }
            if text.contains("Pete robot initialization complete")
    )));
}

fn test_reign_input(
    issued_at_ms: u64,
    mode: ReignMode,
    command: ReignCommand,
    ttl_ms: u64,
) -> ReignInput {
    ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms,
        expires_at_ms: issued_at_ms + ttl_ms,
        source: ReignSource::WebRemote,
        mode,
        command,
        priority: 1.0,
        note: None,
    }
}
