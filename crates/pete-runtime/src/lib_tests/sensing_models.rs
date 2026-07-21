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
fn brainstem_pose_odometry_reaches_body_sense_without_distance_as_x() {
    let status = CockpitStatus {
        raw: serde_json::json!({
            "uptime_ms": 1_000,
            "create_sensors": {
                "last_packet_id": 0,
                "complete_packet_count": 1,
                "last_complete_packet_timestamp_ms": 1_000
            },
            "odometry": {
                "distance_mm": 900,
                "x_mm": 300,
                "y_mm": -400,
                "heading_mrad": 1_250
            }
        })
        .to_string(),
    }
    .summary();

    let body = body_sense_from_cockpit_status(status, 1_000);

    assert_eq!(body.odometry.x_m, 0.3);
    assert_eq!(body.odometry.y_m, -0.4);
    assert_eq!(body.odometry.heading_rad, 1.25);
}

#[test]
fn pre_pose_brainstem_odometry_contract_remains_accepted() {
    let status = CockpitStatus {
        raw: "OK 1 STATUS odometry_distance_mm=425 odometry_heading_mrad=-250".to_string(),
    }
    .summary();

    let body = body_sense_from_cockpit_status(status, 1_000);

    // A stateless conversion cannot recover the path behind a cumulative
    // distance value, so it must not invent global X displacement.
    assert_eq!(body.odometry.x_m, 0.0);
    assert_eq!(body.odometry.y_m, 0.0);
    assert_eq!(body.odometry.heading_rad, -0.25);
}

#[test]
fn legacy_physical_pose_adapter_integrates_distance_after_turn_in_world_y() {
    let mut adapter = PhysicalPoseAdapter::default();
    let samples = [
        (0, 0),
        (1_000, 0),
        (1_000, 1_571),
        (2_000, 1_571),
    ];
    let mut body = BodySense::default();
    for (distance_mm, heading_mrad) in samples {
        let status = CockpitStatus {
            raw: format!(
                "OK 1 STATUS odometry_resets=0 odometry_distance_mm={distance_mm} odometry_heading_mrad={heading_mrad}"
            ),
        }
        .summary();
        body = body_sense_from_cockpit_status_with_pose_adapter(status, 1_000, &mut adapter);
    }

    assert!((body.odometry.x_m - 1.0).abs() < 0.002);
    assert!((body.odometry.y_m - 1.0).abs() < 0.002);
    assert!((body.odometry.heading_rad - 1.571).abs() < 0.001);
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
