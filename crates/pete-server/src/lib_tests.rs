use super::*;
use pete_core::Pose2;
use pete_map::{
    OrientationEstimate, Point3D, PointCloudFrame, PointCloudObservation, PointCloudPoint,
    PoseEdge, PoseEstimate, PoseGraph, PoseNode, YawSource,
};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use pete_actions::{ActionPrimitive, TurnDir};
use pete_autonomic::SimpleSafety;
use pete_body::BodySense;
use pete_conductor::{Conductor, ConductorInput};
use pete_experience::{
    Experience, ExperienceLatent, Impression, MemoryLink, Modality, Prediction, Sensation,
    SensationMetadata, SensationPayload, SensationPayloadKind, SensationSource, VectorEmbedding,
};
use pete_ledger::JsonlLedger;
use pete_memory::InMemoryExperienceStore;
use pete_now::Now;
use pete_runtime::{MinimalRuntime, ReignQueue};
use pete_sensors::{EyeFrame, EyeFrameFormat};
use pete_worldlab::{CaptureSource, CaptureWriter};

#[derive(Clone, Debug, Default)]
struct RecordingConductor {
    last_input: Arc<Mutex<Option<ConductorInput>>>,
}

impl RecordingConductor {
    fn last_input(&self) -> Option<ConductorInput> {
        self.last_input.lock().unwrap().clone()
    }
}

impl Conductor for RecordingConductor {
    fn choose(&mut self, input: ConductorInput) -> Result<ActionPrimitive> {
        let chosen = input
            .proposals
            .last()
            .cloned()
            .unwrap_or(ActionPrimitive::Stop);
        *self.last_input.lock().unwrap() = Some(input);
        Ok(chosen)
    }
}

#[tokio::test]
async fn post_stop_pushes_active_reign_state() {
    let state = ReignServerState::standalone();
    let request = ReignCommandRequest {
        mode: ReignMode::Direct,
        command: ReignCommand::Stop,
        priority: 1.0,
        ttl_ms: Some(2_000),
        note: None,
        source: None,
    };

    let Json(input) = post_reign_command(State(state.clone()), Json(request))
        .await
        .unwrap();
    let sense = state.queue().lock().unwrap().sense(input.issued_at_ms);

    assert!(sense.active);
    assert_eq!(sense.latest.as_ref().map(|value| value.id), Some(input.id));
    assert!(matches!(
        sense.latest.as_ref().map(|value| &value.command),
        Some(ReignCommand::Stop)
    ));
}

#[tokio::test]
async fn posted_reign_command_defaults_to_direct_mode() {
    let state = ReignServerState::standalone();
    let request: ReignCommandRequest = serde_json::from_value(serde_json::json!({
        "command": {
            "type": "Turn",
            "direction": "Left",
            "intensity": 0.5,
            "duration_ms": 500
        },
        "ttl_ms": 2_000
    }))
    .unwrap();

    let Json(input) = post_reign_command(State(state.clone()), Json(request))
        .await
        .unwrap();
    let sense = state.queue().lock().unwrap().sense(input.issued_at_ms);

    assert_eq!(input.mode, ReignMode::Direct);
    assert!(sense.active);
    assert_eq!(
        sense.latest.as_ref().map(|input| &input.mode),
        Some(&ReignMode::Direct)
    );
}

#[tokio::test]
async fn manual_prod_enqueues_bounded_assist_command() {
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    let latest = Arc::new(Mutex::new(Some(WorldSnapshot::default())));
    let state = ReignServerState::with_latest_snapshot(Arc::clone(&queue), latest);

    let Json(response) = post_reign_prod(
        State(state),
        Some(Json(ProdRequest {
            kind: Some("go".to_string()),
            intensity: Some(0.9),
            duration_ms: Some(10_000),
        })),
    )
    .await
    .unwrap();

    assert!(response.accepted);
    assert_eq!(response.input.mode, ReignMode::Assist);
    assert!(matches!(
        response.input.command,
        ReignCommand::Go {
            intensity,
            duration_ms
        } if intensity <= 0.15 && duration_ms <= 1_000
    ));
    let sense = queue.lock().unwrap().sense(response.input.issued_at_ms);
    assert_eq!(
        sense.latest.as_ref().map(|input| input.id),
        Some(response.input.id)
    );
}

fn hardware_reign_state_with_snapshot(snapshot: WorldSnapshot) -> ReignServerState {
    let live = LiveViewState::new().with_real_slow_hardware_control();
    live.update(snapshot);
    ReignServerState::with_live_view(Arc::new(Mutex::new(ReignQueue::default())), &live)
}

fn recent_snapshot() -> WorldSnapshot {
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.last_update_ms = wall_now_ms();
    snapshot.body.battery_level = 1.0;
    snapshot
}

#[tokio::test]
async fn hardware_drive_command_rejected_when_disarmed() {
    let state = hardware_reign_state_with_snapshot(recent_snapshot());
    let request = ReignCommandRequest {
        mode: ReignMode::Direct,
        command: ReignCommand::Go {
            intensity: 0.12,
            duration_ms: 300,
        },
        priority: 1.0,
        ttl_ms: Some(300),
        note: None,
        source: Some(ReignSource::WebRemote),
    };

    let error = post_reign_command(State(state), Json(request))
        .await
        .unwrap_err();

    assert_eq!(error.status, StatusCode::FORBIDDEN);
    assert!(error.message.contains("disarmed"));
}

#[tokio::test]
async fn hardware_stop_accepted_when_disarmed() {
    let state = hardware_reign_state_with_snapshot(recent_snapshot());
    let request = ReignCommandRequest {
        mode: ReignMode::Direct,
        command: ReignCommand::Stop,
        priority: 1.0,
        ttl_ms: Some(300),
        note: None,
        source: Some(ReignSource::WebRemote),
    };

    let Json(input) = post_reign_command(State(state.clone()), Json(request))
        .await
        .unwrap();

    assert!(matches!(input.command, ReignCommand::Stop));
    assert!(
        state
            .queue()
            .lock()
            .unwrap()
            .sense(input.issued_at_ms)
            .active
    );
}

#[tokio::test]
async fn hardware_arm_rejected_in_non_hardware_session() {
    let state = ReignServerState::standalone();

    let error = post_hardware_arm(State(state), Json(HardwareArmRequest { armed: true }))
        .await
        .unwrap_err();

    assert_eq!(error.status, StatusCode::FORBIDDEN);
    assert!(error.message.contains("not available"));
}

#[tokio::test]
async fn hardware_disarm_enqueues_stop() {
    let state = hardware_reign_state_with_snapshot(recent_snapshot());
    let Json(armed) = post_hardware_arm(
        State(state.clone()),
        Json(HardwareArmRequest { armed: true }),
    )
    .await
    .unwrap();
    assert!(armed.armed);

    let Json(disarmed) = post_hardware_arm(
        State(state.clone()),
        Json(HardwareArmRequest { armed: false }),
    )
    .await
    .unwrap();
    let sense = state.queue().lock().unwrap().sense(wall_now_ms());

    assert!(!disarmed.armed);
    assert!(matches!(
        sense.latest.as_ref().map(|input| &input.command),
        Some(ReignCommand::Stop)
    ));
}

#[tokio::test]
async fn hardware_armed_drive_uses_cautious_ttl_and_clamps_command() {
    let state = hardware_reign_state_with_snapshot(recent_snapshot());
    let _ = post_hardware_arm(
        State(state.clone()),
        Json(HardwareArmRequest { armed: true }),
    )
    .await
    .unwrap();
    let request = ReignCommandRequest {
        mode: ReignMode::Direct,
        command: ReignCommand::Go {
            intensity: 0.9,
            duration_ms: 5_000,
        },
        priority: 1.0,
        ttl_ms: Some(5_000),
        note: None,
        source: Some(ReignSource::WebRemote),
    };

    let Json(input) = post_reign_command(State(state), Json(request))
        .await
        .unwrap();

    assert_eq!(input.source, ReignSource::WebRemote);
    assert_eq!(input.mode, ReignMode::Direct);
    assert_eq!(
        input.expires_at_ms - input.issued_at_ms,
        HARDWARE_TTL_MAX_MS
    );
    assert!(matches!(
        input.command,
        ReignCommand::Go {
            intensity,
            duration_ms
        } if intensity <= HARDWARE_MAX_FORWARD_INTENSITY && duration_ms == HARDWARE_TTL_MAX_MS
    ));
}

#[tokio::test]
async fn hardware_armed_analog_drive_clamps_both_axes() {
    let state = hardware_reign_state_with_snapshot(recent_snapshot());
    let _ = post_hardware_arm(
        State(state.clone()),
        Json(HardwareArmRequest { armed: true }),
    )
    .await
    .unwrap();
    let request = ReignCommandRequest {
        mode: ReignMode::Direct,
        command: ReignCommand::Drive {
            forward: 0.9,
            turn: -0.8,
            duration_ms: 5_000,
        },
        priority: 1.0,
        ttl_ms: Some(5_000),
        note: None,
        source: Some(ReignSource::WebRemote),
    };

    let Json(input) = post_reign_command(State(state), Json(request))
        .await
        .unwrap();

    assert_eq!(
        input.expires_at_ms - input.issued_at_ms,
        HARDWARE_TTL_MAX_MS
    );
    assert!(matches!(
        input.command,
        ReignCommand::Drive {
            forward,
            turn,
            duration_ms
        } if forward == HARDWARE_MAX_FORWARD_INTENSITY
            && turn == -HARDWARE_MAX_TURN_INTENSITY
            && duration_ms == HARDWARE_TTL_MAX_MS
    ));
}

#[tokio::test]
async fn hardware_armed_gamepad_drive_is_accepted_and_clamped() {
    let state = hardware_reign_state_with_snapshot(recent_snapshot());
    let _ = post_hardware_arm(
        State(state.clone()),
        Json(HardwareArmRequest { armed: true }),
    )
    .await
    .unwrap();
    let request = ReignCommandRequest {
        mode: ReignMode::Direct,
        command: ReignCommand::Drive {
            forward: 0.9,
            turn: -0.8,
            duration_ms: 5_000,
        },
        priority: 1.0,
        ttl_ms: Some(5_000),
        note: None,
        source: Some(ReignSource::Gamepad),
    };

    let Json(input) = post_reign_command(State(state), Json(request))
        .await
        .unwrap();

    assert_eq!(input.source, ReignSource::Gamepad);
    assert_eq!(
        input.expires_at_ms - input.issued_at_ms,
        HARDWARE_TTL_MAX_MS
    );
    assert!(matches!(
        input.command,
        ReignCommand::Drive {
            forward,
            turn,
            duration_ms
        } if forward == HARDWARE_MAX_FORWARD_INTENSITY
            && turn == -HARDWARE_MAX_TURN_INTENSITY
            && duration_ms == HARDWARE_TTL_MAX_MS
    ));
}

#[tokio::test]
async fn hardware_drive_allows_sensor_only_snapshot_with_non_wall_clock_body_time() {
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.last_update_ms = 0;
    snapshot.body.battery_level = 1.0;
    let state = hardware_reign_state_with_snapshot(snapshot);
    let Json(status) = post_hardware_arm(
        State(state.clone()),
        Json(HardwareArmRequest { armed: true }),
    )
    .await
    .unwrap();
    assert!(status.armed);
    assert_eq!(status.body_age_ms, None);
    assert_eq!(status.reason, None);

    let request = ReignCommandRequest {
        mode: ReignMode::Direct,
        command: ReignCommand::Drive {
            forward: 0.10,
            turn: 0.0,
            duration_ms: 300,
        },
        priority: 1.0,
        ttl_ms: Some(300),
        note: None,
        source: Some(ReignSource::WebRemote),
    };

    let Json(input) = post_reign_command(State(state), Json(request))
        .await
        .unwrap();

    assert!(matches!(input.command, ReignCommand::Drive { .. }));
}

#[tokio::test]
async fn hardware_drive_rejected_on_cliff_even_when_armed() {
    let mut snapshot = recent_snapshot();
    snapshot.body.flags.cliff_front_left = true;
    let state = hardware_reign_state_with_snapshot(snapshot);
    let _ = post_hardware_arm(
        State(state.clone()),
        Json(HardwareArmRequest { armed: true }),
    )
    .await
    .unwrap();
    let request = ReignCommandRequest {
        mode: ReignMode::Direct,
        command: ReignCommand::Turn {
            direction: TurnDir::Left,
            intensity: 0.20,
            duration_ms: 300,
        },
        priority: 1.0,
        ttl_ms: Some(300),
        note: None,
        source: Some(ReignSource::WebRemote),
    };

    let error = post_reign_command(State(state), Json(request))
        .await
        .unwrap_err();

    assert_eq!(error.status, StatusCode::FORBIDDEN);
    assert!(error.message.contains("cliff"));
}

#[tokio::test]
async fn hardware_drive_allows_analog_cliff_risk_without_create_cliff_flag() {
    let mut snapshot = recent_snapshot();
    snapshot.body.cliff_sensors.front_left = 0.96;
    snapshot.body.cliff_sensors.front_right = 0.82;
    let state = hardware_reign_state_with_snapshot(snapshot);
    let _ = post_hardware_arm(
        State(state.clone()),
        Json(HardwareArmRequest { armed: true }),
    )
    .await
    .unwrap();
    let request = ReignCommandRequest {
        mode: ReignMode::Direct,
        command: ReignCommand::Turn {
            direction: TurnDir::Left,
            intensity: 0.20,
            duration_ms: 300,
        },
        priority: 1.0,
        ttl_ms: Some(300),
        note: None,
        source: Some(ReignSource::WebRemote),
    };

    let Json(input) = post_reign_command(State(state), Json(request))
        .await
        .unwrap();

    assert!(matches!(input.command, ReignCommand::Turn { .. }));
}

#[tokio::test]
async fn posted_reign_command_is_seen_by_runtime_tick() {
    let queue = Arc::new(Mutex::new(ReignQueue::default()));
    let state = ReignServerState::new(Arc::clone(&queue));
    let request = ReignCommandRequest {
        mode: ReignMode::Direct,
        command: ReignCommand::Turn {
            direction: TurnDir::Left,
            intensity: 0.5,
            duration_ms: 500,
        },
        priority: 1.0,
        ttl_ms: Some(2_000),
        note: None,
        source: None,
    };

    let Json(input) = post_reign_command(State(state), Json(request))
        .await
        .unwrap();
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    let conductor = RecordingConductor::default();
    let conductor_probe = conductor.clone();
    let mut runtime = MinimalRuntime::with_reign_queue(
        JsonlLedger::new("/tmp/pete-server-runtime-shared-reign-test"),
        memory,
        recall,
        conductor,
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
        Arc::clone(&queue),
    );
    let now = Now::blank(input.issued_at_ms, BodySense::default());
    let expected_action = ActionPrimitive::Turn {
        direction: TurnDir::Left,
        intensity: 0.5,
        duration_ms: 500,
    };

    let tick = runtime
        .tick(now, ExperienceLatent::default(), Vec::new())
        .await
        .unwrap();

    assert!(tick.frame.now.reign.active);
    assert_eq!(
        tick.frame.now.reign.latest.as_ref().map(|value| value.id),
        Some(input.id)
    );
    assert_eq!(
        tick.frame.reign_input.as_ref().map(|value| value.id),
        Some(input.id)
    );
    assert!(tick
        .frame
        .sensations
        .iter()
        .any(|sensation| sensation.kind == "reign.command"));
    let conductor_input = conductor_probe.last_input().unwrap();
    assert_eq!(
        conductor_input.reign.latest.as_ref().map(|value| value.id),
        Some(input.id)
    );
    assert!(
        !conductor_input.proposals.contains(&expected_action),
        "direct operator authority is visible as Reign, not demoted into an ordinary proposal"
    );
    assert_eq!(tick.chosen_action, Some(expected_action.clone()));
    assert_eq!(tick.frame.chosen_action, Some(expected_action));
    assert!(tick
        .frame
        .reign_outcome
        .as_ref()
        .map(|outcome| outcome.accepted_by_conductor)
        .unwrap_or(false));
}

#[tokio::test]
async fn live_scene_returns_503_before_first_snapshot() {
    let err = get_live_scene(State(LiveViewState::new()))
        .await
        .unwrap_err();

    assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn inline_learning_control_updates_live_state() {
    let state = LiveViewState::new();
    let config = InlineLearningConfig {
        mode: InlineLearningMode::WorldOutcome,
        behaviors: pete_runtime::InlineLearningBehaviors {
            danger: false,
            charge: false,
            future: true,
            action_value: true,
            eye_next: false,
            ear_next: false,
            experience: false,
        },
        max_train_steps_per_tick: 2,
    };

    let Json(response) = post_inline_learning(State(state.clone()), Json(config.clone()))
        .await
        .unwrap();
    let Json(readback) = get_inline_learning(State(state)).await;

    assert!(response.enabled);
    assert_eq!(response.training_mode, "inline-world-outcome");
    assert_eq!(readback.config, config);
    assert!(readback.weights_updating);
}

#[tokio::test]
async fn behavior_node_endpoints_round_trip_config() {
    let state = LiveViewState::new();

    let Json(initial) = get_behavior_nodes(State(state.clone())).await;
    assert!(initial.nodes.iter().any(|node| node.node_id == "Conductor"));

    let Json(updated) = post_behavior_node(
        State(state.clone()),
        AxumPath("Conductor".to_string()),
        Json(BehaviorNodeUpdate {
            selected_regime: Some(BehaviorRegime::ShadowTrain),
            selected_hardcoded: Some("reign.teacher".to_string()),
            selected_model: Some("conductor.burn.v0".to_string()),
            checkpoint_path: Some("data/models/conductor_v0".to_string()),
            fallback_policy: Some(pete_behaviors::FallbackPolicy::UseHardcoded),
            training_enabled: Some(true),
        }),
    )
    .await
    .unwrap();
    let Json(readback) = get_behavior_nodes(State(state.clone())).await;
    let conductor = readback
        .nodes
        .iter()
        .find(|node| node.node_id == "Conductor")
        .unwrap();

    assert_eq!(updated.selected_regime, BehaviorRegime::ShadowTrain);
    assert_eq!(conductor.selected_hardcoded, "reign.teacher");
    assert_eq!(
        conductor.checkpoint_path.as_deref(),
        Some("data/models/conductor_v0")
    );
    assert!(conductor.training_enabled);

    let Json(updated_event) = post_behavior_node(
        State(state.clone()),
        AxumPath("EventBump".to_string()),
        Json(BehaviorNodeUpdate {
            selected_regime: Some(BehaviorRegime::ShadowTrain),
            selected_hardcoded: Some("script.on_bump.v0".to_string()),
            selected_model: Some("event.bump.shadow.v0".to_string()),
            checkpoint_path: Some("data/models/event_bump_v0".to_string()),
            fallback_policy: Some(pete_behaviors::FallbackPolicy::UseHardcoded),
            training_enabled: Some(true),
        }),
    )
    .await
    .unwrap();

    assert_eq!(updated_event.selected_regime, BehaviorRegime::ShadowTrain);
    assert_eq!(updated_event.selected_hardcoded, "script.on_bump.v0");
    assert_eq!(
        updated_event.selected_model.as_deref(),
        Some("event.bump.shadow.v0")
    );
}

#[tokio::test]
async fn live_scene_returns_body_pose_and_range_beams() {
    let state = LiveViewState::new();
    state.update_scene_metadata(LiveSceneMetadata {
        arena: Some(SceneArena {
            width_m: 4.0,
            height_m: 3.0,
        }),
        objects: vec![SceneObject {
            id: "charger-0".to_string(),
            kind: "charger".to_string(),
            x_m: 1.2,
            y_m: 0.4,
            radius_m: 0.25,
            label: Some("charger".to_string()),
            color_rgb: Some([80, 220, 130]),
        }],
        sensor_calibration: Some(SceneSensorCalibration::sim_default()),
    });
    state.update_session(SceneSession {
        mode: "virtual-live".to_string(),
        scenario: Some("charger-seeking".to_string()),
        seed: Some(99),
        source: "sim".to_string(),
        tick_ms: Some(100),
    });
    state.update_training_status(LiveTrainingStatus {
        training_mode: "collecting".to_string(),
        ledger_path: Some("data/ledger/virtual-live".to_string()),
        frames_written: 12,
        transitions_written: 11,
        models_loaded: vec!["danger".to_string()],
        model_modes: HashMap::from([("danger".to_string(), "shadow-infer".to_string())]),
        action_selector_mode: "baseline".to_string(),
        weights_updating: false,
    });
    state.update_prod_state(NudgeStatus {
        idle_ms: 4_200,
        last_nudge_ms: Some(1_000),
        nudge_count_recent: 1,
        nudge_blocked_reason: Some("prod cooldown active".to_string()),
        active_nudge: false,
    });
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.odometry.x_m = 0.5;
    snapshot.body.odometry.y_m = 0.75;
    snapshot.body.odometry.heading_rad = 1.25;
    snapshot.body.battery_level = 0.82;
    snapshot.body.last_update_ms = 1234;
    snapshot.final_selected_action = Some(ActionPrimitive::Explore {
        style: pete_actions::ExploreStyle::RandomWalk,
        duration_ms: 1_000,
    });
    snapshot.llm_action_proposal = Some(pete_actions::LlmActionProposal {
        proposed_action: None,
        advisory_action: Some(pete_actions::LlmAdvisoryAction {
            action: ActionPrimitive::Go {
                intensity: 0.4,
                duration_ms: 800,
            },
            source: pete_actions::LlmAdvisoryActionSource::ProviderDecision,
            input_snapshot_ref: "provider-input-1200".to_string(),
            disposition: pete_actions::LlmAdvisoryActionDisposition::DiscardedAtAdvisoryBoundary,
        }),
        accepted: false,
        safety_vetoed: false,
        final_action: snapshot.final_selected_action.clone(),
        ignored_reason: Some("provider suggested Go; discarded at advisory boundary".to_string()),
        safety_reason: None,
    });
    snapshot.range.beams = vec![1.0, 2.0, 3.0];
    snapshot.range.nearest_m = Some(1.0);
    snapshot.extensions.push(pete_now::ExtensionSense {
        schema_version: 1,
        name: "sim.stuck".to_string(),
        values: vec![
            1.0, 1.0, 6.0, 300.0, 3.0, -1.0, 1.0, 0.0, 1.0, 0.0, 3.0, 2.0, 1.0, 0.2,
        ],
    });
    snapshot.eye_frame = Some(EyeFrame {
        captured_at_ms: 1200,
        width: 1,
        height: 1,
        format: EyeFrameFormat::Rgb8,
        bytes: vec![255, 0, 0],
        source: None,
    });
    let expected_llm_action = snapshot
        .llm_action_proposal
        .as_ref()
        .and_then(|proposal| proposal.proposed_action.clone());
    let expected_llm_advisory_action = snapshot
        .llm_action_proposal
        .as_ref()
        .and_then(|proposal| proposal.advisory_action.clone());
    let expected_final_action = snapshot.final_selected_action.clone();
    state.update(snapshot);

    let Json(scene) = get_live_scene(State(state)).await.unwrap();

    assert_eq!(scene.schema_version, 1);
    assert_eq!(scene.session.as_ref().unwrap().mode, "virtual-live");
    assert_eq!(
        scene.session.as_ref().unwrap().scenario.as_deref(),
        Some("charger-seeking")
    );
    assert_eq!(scene.t_ms, 1234);
    assert_eq!(scene.body.x_m, 0.5);
    assert_eq!(scene.body.y_m, 0.75);
    assert_eq!(scene.body.heading_rad, 1.25);
    assert_eq!(scene.action.latest_llm_proposed_action, expected_llm_action);
    assert_eq!(
        scene.action.latest_llm_advisory_action,
        expected_llm_advisory_action
    );
    assert_eq!(scene.action.llm_action_accepted, Some(false));
    assert_eq!(scene.action.llm_action_safety_vetoed, Some(false));
    assert_eq!(
        scene.action.llm_action_ignored_reason.as_deref(),
        Some("provider suggested Go; discarded at advisory boundary")
    );
    assert_eq!(scene.action.final_selected_action, expected_final_action);
    assert_eq!(scene.range.nearest_m, Some(1.0));
    assert_eq!(scene.range.beams.len(), 3);
    assert_eq!(scene.training_mode, "collecting");
    assert_eq!(
        scene.ledger_path.as_deref(),
        Some("data/ledger/virtual-live")
    );
    assert_eq!(scene.frames_written, 12);
    assert_eq!(scene.transitions_written, 11);
    assert_eq!(scene.models_loaded, vec!["danger"]);
    assert!(!scene.weights_updating);
    assert!(scene.stuck);
    assert_eq!(scene.idle_ms, 4_200);
    assert_eq!(scene.last_nudge_ms, Some(1_000));
    assert_eq!(scene.nudge_count_recent, 1);
    assert_eq!(
        scene.nudge_blocked_reason.as_deref(),
        Some("prod cooldown active")
    );
    assert_eq!(scene.prod.idle_ms, 4_200);
    assert!(scene.dead_battery);
    assert_eq!(scene.recovery_mode.as_deref(), Some("turn-away"));
    assert_eq!(scene.stuck_ticks, 6);
    assert_eq!(scene.stuck_detail.class.as_deref(), Some("column-trap"));
    assert_eq!(scene.stuck_detail.trap_kind.as_deref(), Some("column"));
    assert_eq!(scene.stuck_detail.recovery_attempts, 2);
    assert_eq!(scene.stuck_detail.repeated_trap_count, 1);
    assert_eq!(
        scene.stuck_detail.recovery_phase.as_deref(),
        Some("turn-away")
    );
    assert_eq!(scene.objects[0].kind, "charger");
    assert!(scene
        .eye
        .unwrap()
        .data_url
        .unwrap()
        .starts_with("data:image/png;base64,"));
}

#[tokio::test]
async fn live_map_endpoint_returns_pose_projected_beams_and_overlays() {
    let state = LiveViewState::new();
    state.update_scene_metadata(LiveSceneMetadata {
        arena: None,
        objects: vec![
            SceneObject {
                id: "charger-0".to_string(),
                kind: "charger".to_string(),
                x_m: 2.0,
                y_m: 0.5,
                radius_m: 0.2,
                label: Some("charger".to_string()),
                color_rgb: None,
            },
            SceneObject {
                id: "person-0".to_string(),
                kind: "person".to_string(),
                x_m: 0.2,
                y_m: 1.8,
                radius_m: 0.25,
                label: Some("person".to_string()),
                color_rgb: None,
            },
            SceneObject {
                id: "speaker-0".to_string(),
                kind: "speaker".to_string(),
                x_m: -0.4,
                y_m: 1.2,
                radius_m: 0.15,
                label: Some("speaker".to_string()),
                color_rgb: None,
            },
        ],
        sensor_calibration: None,
    });

    let mut snapshot = WorldSnapshot::default();
    snapshot.body.odometry.x_m = 0.5;
    snapshot.body.odometry.y_m = 0.75;
    snapshot.body.odometry.heading_rad = std::f32::consts::FRAC_PI_2;
    snapshot.body.last_update_ms = 1234;
    snapshot.body.flags.bump_left = true;
    snapshot.body.flags.cliff_front_left = true;
    snapshot.range.beams = vec![2.0, 1.0, 2.0];
    snapshot.range.nearest_m = Some(1.0);
    snapshot
        .objects
        .observations
        .push(pete_now::ObjectObservation {
            label: "person-nearby".to_string(),
            class: pete_now::ObjectClass::Person,
            bearing_rad: 0.1,
            distance_m: Some(1.2),
            confidence: 0.82,
            source: pete_now::ObjectObservationSource::Sim,
        });
    snapshot.ear.transcript = Some("Travis".to_string());
    snapshot.llm_action_proposal = Some(pete_actions::LlmActionProposal {
        safety_vetoed: true,
        ..pete_actions::LlmActionProposal::default()
    });
    snapshot.extensions.push(pete_now::ExtensionSense {
        schema_version: 1,
        name: "sim.stuck".to_string(),
        values: vec![1.0, 0.0, 4.0, 200.0, 1.0],
    });
    state.update(snapshot.clone());
    snapshot.body.last_update_ms = 1334;
    state.update(snapshot);

    let Json(map) = get_live_map(State(state)).await.unwrap();

    assert_eq!(map.schema_version, 1);
    assert_eq!(map.label, MAP_LABEL);
    assert!(map.summary.label.contains("scan-matched"));
    assert!(map.summary.label.contains("occupancy"));
    assert_eq!(map.pose_trail.len(), 2);
    assert_eq!(map.current_pose.as_ref().unwrap().x_m, 0.5);
    assert_eq!(map.pose_graph.nodes, map.summary.pose_graph_nodes);
    assert_eq!(map.pose_graph.edges, map.summary.pose_graph_edges);
    assert_eq!(map.pose_graph.nodes, 1);
    assert_eq!(map.remap.submaps, map.summary.remap.submaps);
    assert_eq!(map.remap.cells, map.summary.remap.cells);
    assert!(map.remap.submaps >= 1);
    assert_eq!(
        map.overlays,
        vec![
            "occupancy",
            "rays",
            "raw point cloud",
            "accumulated occupancy",
            "stable wall candidates",
            "danger",
            "charger/charge",
            "social",
            "novelty",
            "events"
        ]
    );
    assert_eq!(map.range_beams.len(), 3);
    assert!(!map.cells.is_empty());
    assert!(map
        .cells
        .iter()
        .any(|cell| cell.occupied_score > cell.free_score));
    assert!(map
        .cells
        .iter()
        .any(|cell| cell.free_score >= cell.occupied_score));
    assert!(map
        .semantic_cells
        .iter()
        .any(|cell| cell.kind == "charger/charge" && cell.label.as_deref() == Some("charger")));
    assert!(map
        .semantic_cells
        .iter()
        .any(|cell| cell.kind == "social" && cell.label.as_deref() == Some("person")));
    assert!(map
        .semantic_cells
        .iter()
        .any(|cell| cell.kind == "social" && cell.label.as_deref() == Some("speaker")));
    assert!(map.events.iter().any(|event| event.kind == "bump"));
    assert!(map.events.iter().any(|event| event.kind == "cliff"));
    assert!(map.events.iter().any(|event| event.kind == "stuck"));
    assert!(map
        .events
        .iter()
        .any(|event| event.kind == "safety_override"));
    assert!(map.events.iter().any(|event| event.kind == "charger"));
    assert!(map.events.iter().any(|event| event.kind == "person"));
    assert!(map.events.iter().any(|event| event.kind == "speaker"));
    assert_eq!(map.entity_graph.schema_version, 1);
    assert!(map
        .entity_graph
        .nodes
        .iter()
        .any(|node| node.node_type == "entity"));
    assert!(map
        .entity_graph
        .nodes
        .iter()
        .any(|node| node.node_type == "text_label"));
    assert!(map
        .entity_graph
        .nodes
        .iter()
        .any(|node| node.node_type == "observation" && node.source_channel.is_some()));
    assert!(map
        .entity_graph
        .edges
        .iter()
        .any(|edge| edge.edge_type == "named_by"));
    assert!(map
        .entity_graph
        .edges
        .iter()
        .any(|edge| edge.observed_at_ms.is_some()));
    assert!(!map.entity_graph.events.is_empty());
    assert!(map
        .entity_graph
        .events
        .iter()
        .any(|event| event.event_type == "create"));
    assert!(map
        .entity_graph
        .events
        .iter()
        .any(|event| event.event_type == "strengthen"));
    assert!(map
        .entity_graph
        .events
        .iter()
        .all(|event| event.timestamp_ms.is_some()));

    let forward_hit = map
        .range_beams
        .iter()
        .find(|beam| beam.hit && beam.angle_rad.abs() < 0.001)
        .expect("center beam should be marked as a hit");
    assert!((forward_hit.origin_x_m - 0.5).abs() < 0.001);
    assert!((forward_hit.origin_y_m - 0.75).abs() < 0.001);
    assert!((forward_hit.end_x_m - 0.5).abs() < 0.001);
    assert!((forward_hit.end_y_m - 1.75).abs() < 0.001);
    assert!((forward_hit.angle_rad - 0.0).abs() < 0.001);
}

#[test]
fn map_pose_graph_summary_exposes_loop_acceptance_and_rejection_reasons() {
    let mut map = LocalMap::default();
    map.pose_graph = PoseGraph {
        nodes: vec![
            PoseNode {
                id: "live-pose-0".to_string(),
                pose_estimate: PoseEstimate {
                    pose: Pose2::default(),
                    confidence: 0.9,
                    covariance: [0.05, 0.05, 0.1],
                    source: "test".to_string(),
                    t_ms: 100,
                },
                t_ms: 100,
                source_frame_id: Some("seed".to_string()),
            },
            PoseNode {
                id: "live-pose-1".to_string(),
                pose_estimate: PoseEstimate {
                    pose: Pose2 {
                        x_m: 0.05,
                        y_m: 0.0,
                        heading_rad: 0.0,
                    },
                    confidence: 0.9,
                    covariance: [0.05, 0.05, 0.1],
                    source: "test".to_string(),
                    t_ms: 200,
                },
                t_ms: 200,
                source_frame_id: Some("return".to_string()),
            },
        ],
        edges: vec![
            PoseEdge {
                from: "live-pose-1".to_string(),
                to: "live-pose-0".to_string(),
                transform: Pose2::default(),
                covariance: [0.04, 0.04, 0.08],
                confidence: 0.94,
                source: PoseEdgeSource::LoopClosureCandidate {
                    kind: "entity_constellation".to_string(),
                    target_frame_id: Some("seed".to_string()),
                    source_frame_id: Some("return".to_string()),
                    source_experience_id: None,
                    source_instant_frame_id: None,
                    source_vector_refs: Vec::new(),
                    source_vector_id: Some("constellation-seed".to_string()),
                    query_vector_id: Some("constellation-return".to_string()),
                    query_experience_id: None,
                },
                active: true,
                rejection_reason: None,
            },
            PoseEdge {
                from: "live-pose-1".to_string(),
                to: "unresolved".to_string(),
                transform: Pose2::default(),
                covariance: [0.2, 0.2, 0.4],
                confidence: 0.6,
                source: PoseEdgeSource::LoopClosureCandidate {
                    kind: "same_place".to_string(),
                    target_frame_id: Some("far-away".to_string()),
                    source_frame_id: Some("return".to_string()),
                    source_experience_id: None,
                    source_instant_frame_id: None,
                    source_vector_refs: Vec::new(),
                    source_vector_id: Some("place-seed".to_string()),
                    query_vector_id: Some("place-return".to_string()),
                    query_experience_id: None,
                },
                active: false,
                rejection_reason: Some("confidence 0.600 below gate 0.850".to_string()),
            },
        ],
    };

    let summary = map_pose_graph_summary(&map);

    assert_eq!(summary.loop_candidate_edges, 2);
    assert_eq!(summary.loop_candidate_active_edges, 1);
    assert_eq!(summary.loop_candidate_rejected_edges, 1);
    assert_eq!(
        summary.loop_candidate_rejection_reasons,
        vec!["confidence 0.600 below gate 0.850".to_string()]
    );
    assert_eq!(
        summary.latest_edge_source.as_deref(),
        Some("loop_closure_candidate")
    );
}

#[test]
fn missing_eye_kinect_and_audio_serialize_as_empty_or_null() {
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.battery_level = 0.0;
    let scene = snapshot_to_scene(
        &snapshot,
        None,
        None,
        LiveTrainingStatus::default(),
        NudgeStatus::default(),
        default_behavior_nodes(),
        None,
        None,
        HardwareControlStatus::unavailable("unit test"),
    );
    let value = serde_json::to_value(scene).unwrap();

    assert!(value["eye"].is_null());
    assert_eq!(value["dead_battery"].as_bool(), Some(true));
    assert_eq!(value["kinect"]["points"].as_array().unwrap().len(), 0);
    assert!(value["audio"].is_null());
    assert!(value["warnings"].as_array().unwrap().len() >= 3);
}

#[test]
fn mono_pcm_audio_reports_energy_without_bearing() {
    let mut snapshot = WorldSnapshot::default();
    snapshot.ear_pcm = Some(pete_sensors::PcmAudioFrame {
        captured_at_ms: 100,
        sample_rate_hz: 44_100,
        channels: 1,
        samples: vec![0, i16::MAX / 2, -(i16::MAX / 2)],
    });
    let scene = snapshot_to_scene(
        &snapshot,
        None,
        None,
        LiveTrainingStatus::default(),
        NudgeStatus::default(),
        default_behavior_nodes(),
        None,
        None,
        HardwareControlStatus::unavailable("unit test"),
    );
    let value = serde_json::to_value(scene).unwrap();

    assert!(value["audio"]["bearing_rad"].is_null());
    assert!(value["audio"]["energy"].as_f64().unwrap() > 0.0);
    assert!(value["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|warning| warning == "no audio bearing stream"));
}

#[test]
fn scene_packet_exposes_persistent_stable_world_belief_points() {
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.last_update_ms = 300;
    let mut cloud = VoxelPointCloud::default();
    for t_ms in [100, 200, 300] {
        cloud.integrate_observation(PointCloudObservation {
            frame: PointCloudFrame::OdometryWorld,
            pose: PoseEstimate {
                pose: Pose2::default(),
                confidence: 0.9,
                covariance: [0.01, 0.01, 0.02],
                source: "unit-test".to_string(),
                t_ms,
            },
            orientation: OrientationEstimate {
                roll_rad: Some(0.01),
                pitch_rad: Some(-0.02),
                yaw_rad: Some(0.0),
                roll_pitch_from_imu: true,
                yaw_source: YawSource::OdometryHeading,
            },
            points: vec![PointCloudPoint {
                position: Point3D {
                    x_m: 1.0,
                    y_m: 0.5,
                    z_m: 0.2,
                },
                color_rgb: None,
                confidence: 1.0,
                depth_index: None,
                depth_uv: None,
                depth_image_size: None,
                source_frame_id: None,
            }],
            source: "unit-test".to_string(),
            t_ms,
            metadata: serde_json::json!({}),
        });
    }

    let scene = snapshot_to_scene(
        &snapshot,
        None,
        None,
        LiveTrainingStatus::default(),
        NudgeStatus::default(),
        default_behavior_nodes(),
        Some(&cloud),
        None,
        HardwareControlStatus::unavailable("unit test"),
    );

    assert_eq!(
        scene.world_belief_layers,
        vec![
            "current rays",
            "raw point cloud",
            "raw camera-frame points",
            "robot-frame points",
            "world-frame points",
            "accumulated occupancy",
            "floor plane",
            "axes gizmo",
            "stable wall candidates"
        ]
    );
    assert_eq!(
        scene
            .kinect
            .accumulated_summary
            .as_ref()
            .unwrap()
            .stable_voxels,
        1
    );
    assert_eq!(scene.kinect.accumulated_points.len(), 1);
    assert!(scene.kinect.accumulated_points[0].stable);
    let belief = scene.kinect.local_world_belief.as_ref().unwrap();
    assert_eq!(belief.stable_voxels, 1);
    assert!(belief.orientation_status.roll_pitch_corrected);
    assert_eq!(scene.kinect.coordinate_system.as_deref(), Some("world"));
}

#[test]
fn live_state_does_not_accumulate_uncalibrated_real_depth_images() {
    let state = LiveViewState::new();
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.last_update_ms = 100;
    snapshot.kinect = KinectSense {
        depth_m: vec![1.0],
        depth_width: 1,
        depth_height: 1,
        depth_fx: 1.0,
        depth_fy: 1.0,
        min_depth_m: 0.1,
        max_depth_m: 3.0,
        depth_coordinate_system: Some("kinect_camera".to_string()),
        ..KinectSense::default()
    };

    state.update(snapshot.clone());

    assert_eq!(state.point_cloud_snapshot().summary().observations, 0);
    let scene = snapshot_to_scene(
        &snapshot,
        None,
        None,
        LiveTrainingStatus::default(),
        NudgeStatus::default(),
        default_behavior_nodes(),
        Some(&state.point_cloud_snapshot()),
        None,
        HardwareControlStatus::unavailable("unit test"),
    );
    assert!(scene
        .warnings
        .iter()
        .any(|warning| warning.contains("accumulated world cloud is disabled")));
}

#[test]
fn live_state_applies_scene_calibration_to_accumulated_point_cloud() {
    let state = LiveViewState::new();
    let calibration = SceneSensorCalibration {
        camera_height_m: 0.50,
        camera_forward_m: 0.10,
        camera_pitch_rad: 12.0_f32.to_radians(),
        ..SceneSensorCalibration::sim_default()
    };
    state.update_scene_metadata(LiveSceneMetadata {
        arena: None,
        objects: Vec::new(),
        sensor_calibration: Some(calibration),
    });
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.last_update_ms = 100;
    snapshot.kinect = KinectSense {
        depth_m: vec![1.0],
        depth_width: 1,
        depth_height: 1,
        depth_fx: 1.0,
        depth_fy: 1.0,
        min_depth_m: 0.1,
        max_depth_m: 3.0,
        depth_coordinate_system: Some("kinect_camera".to_string()),
        ..KinectSense::default()
    };

    state.update(snapshot);

    let cloud = state.point_cloud_snapshot();
    assert_eq!(cloud.summary().observations, 1);
    assert!((cloud.config.camera_height_m - 0.50).abs() < 0.001);
    assert!((cloud.config.camera_forward_m - 0.10).abs() < 0.001);
    assert!((cloud.config.camera_pitch_rad - 12.0_f32.to_radians()).abs() < 0.001);
}

#[test]
fn scene_body_cliff_uses_create_flags_not_analog_risk() {
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.cliff_sensors.front_left = 0.96;
    snapshot.body.cliff_sensors.front_right = 0.82;
    let scene = snapshot_to_scene(
        &snapshot,
        None,
        None,
        LiveTrainingStatus::default(),
        NudgeStatus::default(),
        default_behavior_nodes(),
        None,
        None,
        HardwareControlStatus::unavailable("unit test"),
    );

    assert!(!scene.body.cliff);

    snapshot.body.flags.cliff_front_left = true;
    let scene = snapshot_to_scene(
        &snapshot,
        None,
        None,
        LiveTrainingStatus::default(),
        NudgeStatus::default(),
        default_behavior_nodes(),
        None,
        None,
        HardwareControlStatus::unavailable("unit test"),
    );

    assert!(scene.body.cliff);
}

#[test]
fn compact_kinect_depth_projects_as_meter_range_fan() {
    let depths = vec![2.0; 32];
    let kinect = KinectSense {
        depth_m: depths,
        ..KinectSense::default()
    };
    let (points, diagnostics) =
        depth_points(&kinect, Some(SceneSensorCalibration::sim_default()), None);

    assert_eq!(points.len(), 32);
    assert_eq!(diagnostics.coordinate_system, "scene_robot_render");
    assert_eq!(
        diagnostics.render_frame.as_deref(),
        Some("scene: +x left, +y up, +z forward")
    );
    assert!((points[0].x + 1.847759).abs() < 0.001);
    assert!((points[0].z - 0.765367).abs() < 0.001);
    assert!((points[31].x - 1.847759).abs() < 0.001);
    assert!((points[31].z - 0.765367).abs() < 0.001);

    let near_center = &points[15];
    assert!(near_center.z > 1.99);
    assert!(near_center.x.abs() < 0.11);
}

#[test]
fn kinect_depth_image_projects_with_pinhole_intrinsics() {
    let kinect = KinectSense {
        depth_m: vec![1.0, 2.0, 0.0, 4.0],
        depth_width: 2,
        depth_height: 2,
        depth_fx: 2.0,
        depth_fy: 2.0,
        depth_cx: 0.5,
        depth_cy: 0.5,
        min_depth_m: 0.4,
        max_depth_m: 3.0,
        depth_coordinate_system: Some("kinect_camera".to_string()),
        ..KinectSense::default()
    };

    let (points, diagnostics) = depth_points(&kinect, None, None);

    assert_eq!(points.len(), 2);
    assert_eq!(diagnostics.depth_width, 2);
    assert_eq!(diagnostics.depth_height, 2);
    assert_eq!(diagnostics.valid_depth_count, 2);
    assert_eq!(diagnostics.skipped_depth_count, 1);
    assert_eq!(diagnostics.clipped_depth_count, 1);
    assert_eq!(diagnostics.coordinate_system, "kinect_camera");
    assert!((points[0].x + 0.25).abs() < 0.001);
    assert!((points[0].y + 0.25).abs() < 0.001);
    assert!((points[0].z - 1.0).abs() < 0.001);
    assert!((points[1].x - 0.5).abs() < 0.001);
    assert!((points[1].y + 0.5).abs() < 0.001);
    assert!((points[1].z - 2.0).abs() < 0.001);
}

#[test]
fn kinect_depth_image_projects_rgb_frame_colors_onto_points() {
    let kinect = KinectSense {
        depth_m: vec![1.0, 1.0, 1.0, 1.0],
        depth_width: 2,
        depth_height: 2,
        depth_fx: 2.0,
        depth_fy: 2.0,
        depth_cx: 0.5,
        depth_cy: 0.5,
        min_depth_m: 0.1,
        max_depth_m: 3.0,
        depth_coordinate_system: Some("kinect_camera".to_string()),
        ..KinectSense::default()
    };
    let eye = EyeFrame {
        captured_at_ms: 10,
        width: 2,
        height: 2,
        format: EyeFrameFormat::Rgb8,
        bytes: vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0],
        source: Some("kinect-freenect-rgb".to_string()),
    };
    let color = DepthColorImage::from_eye_frame(&eye).unwrap();

    let (points, diagnostics) = depth_points(&kinect, None, Some(&color));

    assert_eq!(diagnostics.coordinate_system, "kinect_camera");
    assert_eq!(points.len(), 4);
    assert_eq!((points[0].r, points[0].g, points[0].b), (255, 0, 0));
    assert_eq!((points[1].r, points[1].g, points[1].b), (0, 255, 0));
    assert_eq!((points[2].r, points[2].g, points[2].b), (0, 0, 255));
    assert_eq!((points[3].r, points[3].g, points[3].b), (255, 255, 0));
}

#[test]
fn depth_color_sampling_applies_calibrated_pixel_offsets() {
    let width = 80;
    let height = 80;
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            rgb.extend_from_slice(&[x as u8, y as u8, 0]);
        }
    }
    let color = DepthColorImage { width, height, rgb };

    assert_eq!(
        color.sample_depth_pixel_with_offset(10, 10, width, height, 0, 0),
        Some([10, 10, 0])
    );
    assert_eq!(
        color.sample_depth_pixel_with_offset(10, 10, width, height, 3, 7),
        Some([13, 17, 0])
    );
    assert_eq!(
        color.sample_depth_pixel_with_offset(0, 0, width, height, -3, -7),
        Some([0, 0, 0])
    );
}

#[test]
fn calibrated_kinect_depth_image_reports_below_floor_points() {
    let kinect = KinectSense {
        depth_m: vec![1.0],
        depth_width: 1,
        depth_height: 1,
        depth_fx: 1.0,
        depth_fy: 1.0,
        depth_cx: 0.0,
        depth_cy: 0.0,
        min_depth_m: 0.1,
        max_depth_m: 3.0,
        depth_coordinate_system: Some("kinect_camera".to_string()),
        ..KinectSense::default()
    };
    let calibration = SceneSensorCalibration {
        camera_height_m: 0.1,
        camera_pitch_rad: 0.25,
        ..SceneSensorCalibration::sim_default()
    };

    let (points, diagnostics) = depth_points(&kinect, Some(calibration), None);

    assert_eq!(points.len(), 1);
    assert_eq!(diagnostics.coordinate_system, "scene_robot_render");
    assert_eq!(
        diagnostics.point_coordinate_system.as_deref(),
        Some("scene axes derived from robot math frame")
    );
    assert_eq!(diagnostics.below_floor_count, 1);
    assert_eq!(diagnostics.below_floor_ratio, 1.0);
    assert!(diagnostics.min_z_m.unwrap() < 0.0);
    assert!(diagnostics.median_z_m.unwrap() < 0.0);
}

#[tokio::test]
async fn capture_frame_to_scene_conversion_works_for_tiny_capture() {
    let root = unique_test_dir("capture-scene");
    let mut writer = CaptureWriter::create(&root, CaptureSource::Sim, Some(100))
        .await
        .unwrap();
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.last_update_ms = 99;
    snapshot.body.odometry.x_m = 1.0;
    snapshot.range.beams = vec![0.5];
    writer
        .append_snapshot(99, snapshot, Vec::new())
        .await
        .unwrap();
    writer.finish().await.unwrap();

    let Json(scene) = get_capture_scene(Query(CaptureSceneQuery {
        capture: root.clone(),
        frame: 0,
    }))
    .await
    .unwrap();

    assert_eq!(scene.t_ms, 99);
    assert_eq!(scene.body.x_m, 1.0);
    assert_eq!(scene.range.beams.len(), 1);
    std::fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn live_routes_include_3d_and_scene_endpoints() {
    assert!(HTTP_ENDPOINTS.contains(&"/view"));
    assert!(HTTP_ENDPOINTS.contains(&"/view/3d"));
    assert!(HTTP_ENDPOINTS.contains(&"/view/scene"));
    assert!(HTTP_ENDPOINTS.contains(&"/view/map"));
    assert!(HTTP_ENDPOINTS.contains(&"/view/embodied"));
    assert!(HTTP_ENDPOINTS.contains(&"/view/embodied/graph"));
    assert!(HTTP_ENDPOINTS.contains(&"/api/experience/lineage"));
    assert!(HTTP_ENDPOINTS.contains(&"/debug/embodied"));
    assert!(HTTP_ENDPOINTS.contains(&"/debug/embodied/graph"));
    assert!(HTTP_ENDPOINTS.contains(&"/models"));
    assert!(HTTP_ENDPOINTS.contains(&"/stream/llm"));
    assert!(HTTP_ENDPOINTS.contains(&"/reign/command/ws"));
    let Html(live_page) = live_view_page().await;
    assert!(live_page.contains("Embodied lineage"));
    assert!(live_page.contains("/api/experience/lineage"));
    assert!(live_page.contains("graph_query"));
    assert!(live_page.contains("graph_modality"));
    let Html(page) = live_view_3d_page().await;
    assert!(page.contains("Instant 3D"));
    assert!(page.contains("/view/snapshot"));
    assert!(page.contains("/models"));
    assert!(page.contains("Training stats"));
    assert!(page.contains("Connections"));
    assert!(page.contains("/stream/llm"));
    assert!(page.contains("LLM streams"));
    assert!(page.contains("navigator.xr"));
    assert!(page.contains("window.isSecureContext"));
    assert!(page.contains("/reign/command"));
    assert!(page.contains("/reign/command/ws"));
    assert!(page.contains("/view/behavior-nodes"));
    assert!(page.contains("behaviorInspector"));
    assert!(page.contains("packet.behavior_nodes"));
    assert!(page.contains("const nodes = {...graphLayout}"));
    assert!(page.contains("source='Gamepad'"));
    assert!(page.contains("type:'Drive'"));
    assert!(page.contains("data-reign-turn=\"Left\""));
    assert!(page.contains("data-reign-turn=\"Right\""));
    assert!(page.contains("function postTurnOnly"));
    assert!(page.contains("type:'Turn', direction"));
    assert!(page.contains("type:'Speak'"));
    assert!(page.contains("type:'Chirp'"));
    assert!(page.contains("data-chirp=\"Confirm\""));
    assert!(page.contains("data-chirp=\"GoalAcquired\""));
    assert!(page.contains("data-chirp=\"PersonRecognized\""));
    assert!(page.contains("notes 72,79,84,91"));
    assert!(page.contains("WASD / arrow keys"));
    assert!(page.contains("keyboardReignCodes"));
    assert!(page.contains("KeyW"));
    assert!(page.contains("ArrowUp"));
    assert!(page.contains("function syncVisualFloor"));
    assert!(page.contains("const center = world(centerX, centerY, 0);"));
    assert!(page.contains("viewerCamera.setTarget(center);"));
    assert!(!page.contains("selectProjectedFloor"));
    assert!(!page.contains("quaternionFromFloorNormal"));
    assert!(page.contains("Real hardware armed"));
    assert!(page.contains("id=\"reign-voice-panel\""));
    assert!(page.contains("id=\"reign-hardware\""));
    assert!(page.contains("id=\"reign-map\""));
    assert!(page.contains("id=\"reign-constellation\""));
    assert!(page.contains("'reign-voice-panel'"));
    assert!(page.contains("'reign-hardware'"));
    assert!(page.contains("'reign-map'"));
    assert!(page.contains("'reign-constellation'"));
    assert!(page.contains("/reign/hardware-arm"));
    let Html(reign_page) = reign_page().await;
    assert!(reign_page.contains("chirp('Confirm')"));
    assert!(reign_page.contains("chirp('GoalAcquired')"));
    assert!(reign_page.contains("72,79,84,91; little fanfare"));
    assert!(reign_page.contains("chirp('DidntUnderstand')"));
    assert!(page.contains("window.addEventListener('pagehide'"));
    assert!(page.contains("navigator.sendBeacon"));
    assert!(!page.contains("id=\"reign-dock\""));
    assert!(!page.contains("id=\"reign-explore\""));
    assert!(!page.contains("data-cockpit"));
    assert!(!page.contains("data-heading-nudge"));
    assert!(!page.contains("function startCockpitHold"));
    assert!(!page.contains("function commandForHeadingTarget"));
    assert!(page.contains("'WebRemote'"));
    assert!(page.contains("function projectRangeBeam"));
    assert!(page.contains("/view/map"));
    assert!(page.contains("data-map-layer=\"occupancy\""));
    assert!(page.contains("data-map-layer=\"rays\""));
    assert!(page.contains("data-map-layer=\"raw point cloud\""));
    assert!(page.contains("data-map-layer=\"accumulated occupancy\""));
    assert!(page.contains("data-map-layer=\"flat image\""));
    assert!(page.contains("data-map-layer=\"hypotheses\""));
    assert!(page.contains("function syncDisplayToggles"));
    assert!(page.contains("data-map-layer=\"stable wall candidates\""));
    assert!(page.contains("function renderWorldBeliefPoints"));
    assert!(page.contains("function packetHeadingRad"));
    assert!(page.contains("function robotYawToBabylon"));
    assert!(page.contains("function renderRobotMotion"));
    assert!(page.contains("motionState.connections"));
    assert!(page.contains("scanConnections"));
    assert!(page.contains("function pointCloudFrameKind"));
    assert!(page.contains("function robotRenderPointToBabylonLocal"));
    assert!(page.contains("function kinectCameraPointToBabylonLocal"));
    assert!(page.contains("function worldMathPointToBabylonWorld"));
    assert!(page.contains("new BABYLON.Vector3(p.x, -p.y, -p.z)"));
    assert!(page.contains("new BABYLON.Vector3(-p.y, p.z, p.x)"));
    assert!(page.contains("TransformCoordinates(kinectCameraPointToBabylonLocal(p), robotMatrix)"));
    assert!(page.contains("return worldMathPointToBabylonWorld(p);"));
    assert!(!page.contains("eyePanel.scaling.x = -1"));
    assert!(page.contains("function drawMirroredImageToEyeCanvas"));
    assert!(page.contains("mirroredImageTargetOffset"));
    assert!(page.contains("function renderPersistentWorldBelief"));
    assert!(page.contains("local_world_belief"));
    assert!(page.contains("roll_pitch_corrected"));
    assert!(page
        .contains("Scan-matched occupancy map: range scans correct odometry before integration."));
    assert!(page.contains("const traceLocal = (x, y) =>"));
    assert!(page.contains("forward: dx * headingCos + dy * headingSin"));
    assert!(page.contains("function occupancyGridCellCenter"));
    assert!(page.contains("forward: (Number(cell.x) + .5) * res"));
    assert!(page.contains("left: (Number(cell.y) + .5) * res"));
    assert!(page.contains("const center = occupancyGridCellCenter(cell, grid);"));
    assert!(page.contains("const gridPose = packet.body || latest;"));
    assert!(page.contains("occupancyGridCellToWorld(gridPose, cell, grid)"));
    assert!(page.contains("traceCtx.rotate(-Math.PI / 2);"));
    assert!(page.contains("id=\"entity-graph\""));
    assert!(page.contains("drawEntityGraph"));
    assert!(page.contains("createDefaultXRExperienceAsync"));
    assert!(page.contains("if(!eye?.data_url)"));
    assert!(page.contains(
            ".panel-window.is-shaded{height:32px!important;min-height:32px!important;max-height:32px!important;"
        ));
}

#[tokio::test]
async fn live_embodied_endpoint_returns_latest_context() {
    let state = LiveViewState::new();
    let context = EmbodiedContext {
        experience_id: Some(uuid::Uuid::new_v4()),
        summary: "I see a frame.".to_string(),
        ..EmbodiedContext::default()
    };
    state.update_embodied_context(context.clone());

    let Json(response) = get_live_embodied(State(state)).await.unwrap();

    assert_eq!(response, context);
}

#[test]
fn embodied_lineage_graph_traces_current_experience() {
    let primary = Sensation::primary(
        Modality::Vision,
        SensationSource::new("unit-camera"),
        100,
        101,
        SensationPayload::image_metadata(32, 24, "rgb8", 32 * 24 * 3),
    )
    .with_summary("I receive a visual frame.");
    let child = Sensation::descendant(
        &primary,
        "vision.crop.focus",
        SensationPayloadKind::Crop,
        serde_json::json!({"x": 4, "y": 3, "width": 12, "height": 9}),
        SensationMetadata::default(),
        "focus",
    )
    .with_summary("I focus on a patch.")
    .with_vector(VectorEmbedding::new(
        vec![0.1, 0.2, 0.3],
        "unit.crop.v0",
        Modality::Vision,
        SensationPayloadKind::Crop,
        primary.id,
        102,
    ));
    let impression = Impression::new(
        "vision.focus.impression",
        "I see and focus.",
        vec![primary.id, child.id],
        100,
        102,
    );
    let summary_impression = Impression::new(
        "experience.summary",
        "I see and focus.",
        vec![primary.id, child.id],
        100,
        102,
    )
    .with_vector(VectorEmbedding::new(
        vec![0.1, 0.2, 0.3, 0.4],
        "unit.fuser.v0",
        Modality::Other,
        SensationPayloadKind::Structured,
        child.id,
        103,
    ));
    let mut experience = Experience::new(
        "embodied.now",
        "I see and focus.",
        vec![impression.id, summary_impression.id],
        vec![primary.id, child.id],
        100,
        102,
    );
    experience.summary_impression = Some(summary_impression.clone().for_experience(experience.id));
    experience.predictions.push(Prediction {
        offset_ms: 750,
        text: "The focused view should remain stable.".to_string(),
        confidence: 0.6,
        vector: None,
    });
    experience.memory_links.push(MemoryLink {
        target_id: "memory-1".to_string(),
        relation: "similar".to_string(),
        score: 0.8,
        payload: serde_json::json!({"text": "A previous focused camera moment."}),
    });
    let context = EmbodiedContext::from_current_experience(
        Some(&experience),
        &[primary.clone(), child.clone()],
        &[impression.clone(), summary_impression.clone()],
        &[],
        &[],
    );

    let graph = EmbodiedLineageGraph::from_context(&context);

    assert_eq!(graph.schema_version, 1);
    assert_eq!(graph.experience_id, Some(experience.id.to_string()));
    assert!(graph.nodes.iter().any(|node| {
        node.id == format!("sensation:{}", child.id)
            && node.derived
            && node.modality.as_deref() == Some("vision")
    }));
    assert!(graph.edges.iter().any(|edge| {
        edge.from == format!("sensation:{}", primary.id)
            && edge.to == format!("sensation:{}", child.id)
            && edge.relation == EmbodiedGraphEdgeType::ParentSensation
    }));
    assert!(graph.edges.iter().any(|edge| {
        edge.from == format!("sensation:{}", child.id)
            && edge.to == format!("experience:{}", experience.id)
            && edge.relation == EmbodiedGraphEdgeType::SensationMember
    }));
    assert!(graph.edges.iter().any(|edge| {
        edge.from == format!("impression:{}", impression.id)
            && edge.to == format!("experience:{}", experience.id)
            && edge.relation == EmbodiedGraphEdgeType::ImpressionMember
    }));
    assert!(graph
        .nodes
        .iter()
        .any(|node| node.node_type == EmbodiedGraphNodeType::Prediction));
    assert!(graph
        .vector_metadata
        .iter()
        .any(|vector| { vector.model_id == "unit.fuser.v0" && vector.dim == 4 }));
    assert!(graph
        .vector_metadata
        .iter()
        .any(|vector| { vector.model_id == "unit.crop.v0" && vector.dim == 3 }));
    assert_eq!(graph.recent_memories.len(), 1);
    assert_eq!(graph.recent_memories[0].target_id, "memory-1");
}

#[test]
fn model_summary_reads_training_artifacts() {
    let model_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/models");
    let packet = read_models_response(&model_root);

    assert_eq!(packet.schema_version, 1);
    assert!(packet
        .connections
        .iter()
        .any(|edge| edge.from == "Ledger" && edge.to == "Training"));
    assert!(packet
        .behavior_nodes
        .iter()
        .any(|node| node.node_id == "ActionValue"));
    assert!(packet
        .registry
        .iter()
        .any(|entry| entry.behavior == "danger"));
    for (name, behavior_report, scenario_report) in [
        (
            "danger_golden_column_v0",
            "data/reports/danger-golden-column-heldout-eval.json",
            "data/reports/golden-column-trap-model-assisted-full-shadow.json",
        ),
        (
            "action_value_golden_column_v0",
            "data/reports/action-value-golden-column-heldout-eval.json",
            "data/reports/golden-column-trap-model-assisted-full-shadow.json",
        ),
        (
            "charge_golden_charger_v0",
            "data/reports/charge-golden-charger-heldout-eval.json",
            "data/reports/golden-charger-seeking-heldout.json",
        ),
    ] {
        let entry = packet
            .registry
            .iter()
            .find(|entry| entry.name == name)
            .unwrap_or_else(|| panic!("missing registry entry {name}"));
        assert_eq!(entry.status, "shadow");
        assert_eq!(entry.behavior_report_path.as_deref(), Some(behavior_report));
        assert_eq!(entry.scenario_report_path.as_deref(), Some(scenario_report));
        assert!(entry
            .allowed_modes
            .iter()
            .any(|mode| mode == "shadow-infer"));
        assert!(!entry.allowed_modes.iter().any(|mode| mode == "model-infer"));
    }
}

#[tokio::test]
async fn live_view_page_draws_common_camera_formats() {
    let Html(page) = live_view_page().await;

    assert!(page.contains("isBgr"));
    assert!(page.contains("isGray"));
    assert!(page.contains("isYuyv"));
    assert!(page.contains("writeYuvPixel"));
    assert!(page.contains("drawGeneratedEye"));
    assert!(page.contains("/view/scene"));
    assert!(page.contains("session.source === 'sim'"));
    assert!(page.contains("includes('virtual')"));
}

#[test]
fn yuyv_eye_frame_encodes_to_png_data_url() {
    let frame = EyeFrame {
        captured_at_ms: 1,
        width: 2,
        height: 1,
        format: EyeFrameFormat::Yuyv422,
        bytes: vec![82, 90, 145, 240],
        source: None,
    };

    let (eye, warnings) = scene_eye_from_frame(&frame, None, 1);

    assert!(warnings.is_empty());
    assert_eq!(eye.width, 2);
    assert_eq!(eye.height, 1);
    assert!(eye
        .data_url
        .as_deref()
        .unwrap_or_default()
        .starts_with("data:image/png;base64,"));
}

#[test]
fn grbg_bayer_eye_frame_encodes_to_png_data_url() {
    let frame = EyeFrame {
        captured_at_ms: 1,
        width: 2,
        height: 2,
        format: EyeFrameFormat::BayerGrbg8,
        bytes: vec![90, 220, 40, 110],
        source: None,
    };

    let (eye, warnings) = scene_eye_from_frame(&frame, None, 1);

    assert!(warnings.is_empty());
    assert_eq!(eye.width, 2);
    assert_eq!(eye.height, 2);
    assert!(eye
        .data_url
        .as_deref()
        .unwrap_or_default()
        .starts_with("data:image/png;base64,"));
}

#[test]
fn grbg_bayer_eye_frame_encodes_latest_png_bytes() {
    let frame = EyeFrame {
        captured_at_ms: 1,
        width: 2,
        height: 2,
        format: EyeFrameFormat::BayerGrbg8,
        bytes: vec![90, 220, 40, 110],
        source: None,
    };

    let bytes = encode_eye_png_bytes(&frame).unwrap();

    assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
}

#[tokio::test]
async fn cognitive_summary_endpoint_returns_valid_json() {
    let state = LiveViewState::new();
    let Json(report) = get_cognitive_summary(State(state)).await;
    let value = serde_json::to_value(&report).expect("cognitive report serializes");

    assert!(value.get("summary").is_some());
    assert!(value.get("bindings").is_some());
    assert_eq!(value["summary"]["feature_count"], 0);
    assert!(HTTP_ENDPOINTS.contains(&"/api/cognitive/summary"));
    assert!(HTTP_ENDPOINTS.contains(&"/api/cognitive/bindings"));
    let Html(page) = cognitive_view_page().await;
    assert!(page.contains("Cognitive Inspector"));
    assert!(page.contains("/api/cognitive/summary"));
}

fn unique_test_dir(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "pete-server-{name}-{}-{}",
        std::process::id(),
        wall_now_ms()
    ));
    std::fs::remove_dir_all(&path).ok();
    path
}
