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
