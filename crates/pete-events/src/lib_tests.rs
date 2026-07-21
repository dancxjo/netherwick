use super::*;
use pete_actions::TurnDir;
use pete_body::BodySense;
use pete_memory::RecallBundle;
use pete_now::{ObjectObservation, ObjectObservationSource, ObjectSense};

#[test]
fn extractor_emits_memory_recalled_when_hits_exist() {
    let mut extractor = EventExtractor::default();
    let now = Now::blank(7, BodySense::default());
    let recall = RecallBundle {
        hits: vec![pete_now::RecallHit {
            frame_id: None,
            score: 0.9,
            summary: "danger".to_string(),
            warning: None,
            graph_context: Vec::new(),
        }],
        ..RecallBundle::default()
    };

    let events = extractor.events_from_now(&now, Some(&recall));

    assert!(events
        .iter()
        .any(|event| event.kind == EventKind::MemoryRecalled));
}

#[test]
fn extractor_emits_reign_commanded_once_then_expired() {
    let mut extractor = EventExtractor::default();
    let mut now = Now::blank(20, BodySense::default());
    let input = test_reign_input(20);
    now.reign.active = true;
    now.reign.latest = Some(input.clone());

    let first = extractor.events_from_now(&now, None);
    let second = extractor.events_from_now(&now, None);
    now.t_ms = 1_200;
    now.reign.active = false;
    now.reign.latest = None;
    let expired = extractor.events_from_now(&now, None);

    assert!(first
        .iter()
        .any(|event| event.kind == EventKind::ReignCommanded));
    assert!(!second
        .iter()
        .any(|event| event.kind == EventKind::ReignCommanded));
    assert!(expired.iter().any(|event| {
        matches!(
            &event.payload,
            EventPayload::ReignExpired { input: expired_input }
                if expired_input.id == input.id
        )
    }));
}

#[test]
fn extractor_emits_reign_cleared_without_expired() {
    let mut extractor = EventExtractor::default();
    let mut now = Now::blank(20, BodySense::default());
    now.reign.active = true;
    now.reign.latest = Some(test_reign_input(20));
    let _ = extractor.events_from_now(&now, None);

    now.t_ms = 30;
    now.reign.active = false;
    now.reign.latest = None;
    now.reign.clear_sequence = 1;
    let events = extractor.events_from_now(&now, None);

    assert!(events
        .iter()
        .any(|event| event.kind == EventKind::ReignCleared));
    assert!(!events
        .iter()
        .any(|event| event.kind == EventKind::ReignExpired));
}

#[test]
fn battery_low_responder_publishes_drive_impulse_without_action() {
    let mut bus = EventBus::new();
    bus.on(responders::BatteryLowResponder);
    let mut body = BodySense::default();
    body.battery_level = 0.1;
    let now = Now::blank(5, body);
    let ctx = EventContext {
        now: &now,
        latent: None,
        recall: None,
        predicted_futures: &[],
        llm: None,
        safety: None,
    };
    let event = Event::new(5, EventKind::BatteryLow)
        .with_payload(EventPayload::BatteryLow { battery_level: 0.1 });

    let output = bus.dispatch(&ctx, &event).unwrap();

    assert!(output.iter().any(|response| matches!(
        response,
        Response::AddDriveImpulse {
            name: DriveName::BatteryHunger,
            value: 1.0,
        }
    )));
}

#[test]
fn extractor_emits_near_wall_face_and_sound_events() {
    let mut extractor = EventExtractor::default();
    let mut now = Now::blank(12, BodySense::default());
    now.range.beams = vec![0.12, 0.8];
    now.ear.features = vec![vec![0.7]];
    now.objects = ObjectSense {
        schema_version: 1,
        observations: vec![ObjectObservation {
            label: "charger dock".to_string(),
            class: ObjectClass::Charger,
            bearing_rad: 0.0,
            distance_m: Some(0.8),
            confidence: 0.9,
            source: ObjectObservationSource::Sim,
        }],
        ..ObjectSense::default()
    };
    now.face.vectors = vec![pete_now::VectorArtifact::new(
        "faces",
        "test-face",
        vec![0.1, 0.2, 0.3],
    )];
    now.memory.face_familiarity = 0.9;

    let events = extractor.events_from_now(&now, None);

    assert!(events.iter().any(|event| event.kind == EventKind::NearWall));
    assert!(events
        .iter()
        .any(|event| event.kind == EventKind::SoundHeard));
    assert!(events.iter().any(|event| {
        matches!(
            &event.payload,
            EventPayload::ObjectSeen { labels, classes }
                if labels == &vec!["charger dock".to_string()]
                    && classes == &vec![ObjectClass::Charger]
        )
    }));
    assert!(events
        .iter()
        .any(|event| event.kind == EventKind::FaceDetected));
    assert!(events
        .iter()
        .any(|event| event.kind == EventKind::FaceRecognized));
}

#[test]
fn extractor_emits_region_transitions_from_odometry_cells() {
    let mut extractor = EventExtractor::default();
    let mut body = BodySense::default();
    body.odometry.x_m = 0.2;
    body.odometry.y_m = 0.2;
    let first = Now::blank(1, body.clone());
    let _ = extractor.events_from_now(&first, None);
    body.odometry.x_m = 1.2;
    body.odometry.y_m = 0.2;
    let second = Now::blank(2, body);

    let events = extractor.events_from_now(&second, None);

    assert!(events.iter().any(|event| matches!(
        &event.payload,
        EventPayload::ExitedRegion { region_id } if region_id == "cell:0:0"
    )));
    assert!(events.iter().any(|event| matches!(
        &event.payload,
        EventPayload::EnteredRegion { region_id } if region_id == "cell:1:0"
    )));
}

#[test]
fn extractor_emits_bump_and_charging_transition_events() {
    let mut extractor = EventExtractor::default();
    let mut body = BodySense::default();
    let first = Now::blank(1, body.clone());
    let _ = extractor.events_from_now(&first, None);
    body.charging = true;
    body.flags.bump_left = true;
    let charging = Now::blank(2, body.clone());
    let charging_events = extractor.events_from_now(&charging, None);
    body.charging = false;
    body.flags.bump_left = false;
    let stopped = Now::blank(3, body);
    let stopped_events = extractor.events_from_now(&stopped, None);

    assert!(charging_events
        .iter()
        .any(|event| event.kind == EventKind::ChargingStarted));
    assert!(charging_events
        .iter()
        .any(|event| matches!(&event.payload, EventPayload::Bump { side } if side == "left")));
    assert!(stopped_events
        .iter()
        .any(|event| event.kind == EventKind::ChargingStopped));
}

#[test]
fn charging_started_responder_adds_sweetness_and_memory_note() {
    let mut bus = EventBus::new();
    bus.on(responders::ChargingStartedResponder);
    let mut body = BodySense::default();
    body.charging = true;
    let now = Now::blank(8, body);
    let ctx = EventContext {
        now: &now,
        latent: None,
        recall: None,
        predicted_futures: &[],
        llm: None,
        safety: None,
    };
    let event = Event::new(8, EventKind::ChargingStarted)
        .with_payload(EventPayload::ChargingStarted { battery_level: 0.8 });

    let output = bus.dispatch(&ctx, &event).unwrap();

    assert!(output.iter().any(|response| match response {
        Response::AddSensation(sensation) => sensation
            .summary
            .as_deref()
            .unwrap_or_default()
            .contains("sweet"),
        _ => false,
    }));
    assert!(output.iter().any(
        |response| matches!(response, Response::AddMemoryNote(note) if note.contains("sweet"))
    ));
}

#[test]
fn bump_responder_marks_danger_without_selecting_an_action() {
    let mut bus = EventBus::new();
    bus.on(responders::BumpResponder);
    let now = Now::blank(9, BodySense::default());
    let ctx = EventContext {
        now: &now,
        latent: None,
        recall: None,
        predicted_futures: &[],
        llm: None,
        safety: None,
    };
    let event = Event::new(9, EventKind::Bump).with_payload(EventPayload::Bump {
        side: "left".to_string(),
    });

    let output = bus.dispatch(&ctx, &event).unwrap();

    assert!(output.iter().any(|response| match response {
        Response::AddSensation(sensation) => sensation.kind == "body.bump",
        _ => false,
    }));
    assert!(output.iter().any(|response| matches!(
        response,
        Response::AddDriveImpulse {
            name: DriveName::DangerAvoidance,
            value: 1.0,
        }
    )));
}

#[test]
fn memory_recalled_responder_updates_memory_sense() {
    let mut bus = EventBus::new();
    bus.on(responders::MemoryRecalledResponder);
    let now = Now::blank(10, BodySense::default());
    let recall = RecallBundle {
        hits: vec![pete_now::RecallHit {
            frame_id: None,
            score: 0.7,
            summary: "remembered danger".to_string(),
            warning: Some("danger".to_string()),
            graph_context: Vec::new(),
        }],
        sense: MemorySense {
            place_danger: 0.9,
            similar_situation_count: 1,
            ..MemorySense::default()
        },
        first_person_summary: "I remember danger here.".to_string(),
        recollections: Vec::new(),
        semantic_map: None,
        place_recognition_candidates: Vec::new(),
    };
    let ctx = EventContext {
        now: &now,
        latent: None,
        recall: Some(&recall),
        predicted_futures: &[],
        llm: None,
        safety: None,
    };
    let event = Event::new(10, EventKind::MemoryRecalled)
        .with_payload(EventPayload::MemoryRecalled { hits: 1 });

    let output = bus.dispatch(&ctx, &event).unwrap();

    assert!(output.iter().any(|response| matches!(
            response,
            Response::SetMemorySense(MemorySense { place_danger, .. }) if (*place_danger - 0.9).abs() < f32::EPSILON
        )));
}

#[test]
fn safety_veto_responder_adds_experience_note() {
    let mut bus = EventBus::new();
    bus.on(responders::SafetyVetoResponder);
    let now = Now::blank(11, BodySense::default());
    let ctx = EventContext {
        now: &now,
        latent: None,
        recall: None,
        predicted_futures: &[],
        llm: None,
        safety: None,
    };
    let event = Event::new(11, EventKind::SafetyVetoed).with_payload(EventPayload::SafetyVetoed {
        desired_action: ActionPrimitive::Dock,
        reason: "Cliff".to_string(),
    });

    let output = bus.dispatch(&ctx, &event).unwrap();

    assert!(output.iter().any(|response| match response {
        Response::AddExperience(experience) => experience.text.contains("Cliff"),
        _ => false,
    }));
    assert!(output.iter().any(|response| matches!(
        response,
        Response::AddMemoryNote(note) if note.contains("Cliff")
    )));
}

#[test]
fn reign_responder_publishes_human_evidence_without_selecting_action() {
    let mut bus = EventBus::new();
    bus.on(responders::ReignResponder);
    let now = Now::blank(10, BodySense::default());
    let ctx = EventContext {
        now: &now,
        latent: None,
        recall: None,
        predicted_futures: &[],
        llm: None,
        safety: None,
    };
    let input = ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: 10,
        expires_at_ms: 1_010,
        source: pete_actions::ReignSource::WebRemote,
        mode: pete_actions::ReignMode::Direct,
        command: pete_actions::ReignCommand::Turn {
            direction: TurnDir::Left,
            intensity: 0.5,
            duration_ms: 500,
        },
        priority: 1.0,
        note: None,
    };
    let event = Event::new(10, EventKind::ReignCommanded)
        .with_payload(EventPayload::ReignCommanded { input });

    let output = bus.dispatch(&ctx, &event).unwrap();

    assert!(output.iter().any(|response| match response {
        Response::AddSensation(sensation) => sensation.kind == "reign.command",
        _ => false,
    }));
}

fn test_reign_input(issued_at_ms: TimeMs) -> ReignInput {
    ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms,
        expires_at_ms: issued_at_ms + 1_000,
        source: pete_actions::ReignSource::WebRemote,
        mode: pete_actions::ReignMode::Direct,
        command: pete_actions::ReignCommand::Stop,
        priority: 1.0,
        note: None,
    }
}
