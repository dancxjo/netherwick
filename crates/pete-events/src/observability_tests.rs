use pete_actions::{
    action_to_motor_command, ActionPrimitive, ReignCommand, ReignInput, ReignMode, ReignOutcome,
    ReignSource,
};
use pete_autonomic::{SafetyLayer, SimpleSafety};
use pete_body::BodySense;
use pete_core::Provenance;
use pete_experience::{
    Experience, Impression, Modality, Sensation, SensationPayload, SensationPayloadKind,
    SensationSource, VectorEmbedding,
};
use pete_now::Now;
use serde_json::json;
use uuid::Uuid;

use super::*;

fn test_id(value: u128) -> Uuid {
    Uuid::from_u128(value)
}

fn accepted_event(id: &str, event_type: BrainEventType, component: &str, t_ms: u64) -> BrainEvent {
    let mut event = BrainEvent::historical(
        BrainEventId(id.to_string()),
        event_type,
        ProducerIdentity::new(Brain::Motherbrain, component),
        EventTimes::observed(t_ms, t_ms),
    );
    event.disposition = EventDisposition::Accepted;
    event
}

#[test]
fn representative_sensor_to_outcome_chain_round_trips_with_typed_links() {
    let mut sensation = Sensation::primary(
        Modality::Touch,
        SensationSource::new("create.bump"),
        100,
        103,
        SensationPayload::structured(json!({ "left": true })),
    );
    sensation.id = test_id(1);
    sensation.metadata.confidence = Some(1.0);

    let mut impression = Impression::new(
        "hazard.contact",
        "Left bumper is pressed.",
        vec![sensation.id],
        100,
        104,
    )
    .with_confidence(0.99);
    impression.id = test_id(2);

    let mut experience = Experience::new(
        "belief.hazard",
        "Contact hazard on the left.",
        vec![impression.id],
        vec![sensation.id],
        100,
        105,
    );
    experience.id = test_id(3);

    let input = ReignInput {
        id: test_id(4),
        issued_at_ms: 106,
        expires_at_ms: 2_106,
        source: ReignSource::HumanSupervisor,
        mode: ReignMode::Direct,
        command: ReignCommand::Go {
            intensity: 0.4,
            duration_ms: 1_000,
        },
        priority: 1.0,
        note: Some("test chain".to_string()),
    };
    let proposal = BrainEvent::from_reign_input(&input, 107);

    let mut now = Now::blank(108, BodySense::default());
    now.body.flags.bump_left = true;
    let mut safety = SimpleSafety::default();
    let proposed_action = ActionPrimitive::Go {
        intensity: 0.4,
        duration_ms: 1_000,
    };
    let safety_decision = safety.filter(&now, action_to_motor_command(Some(&proposed_action)));
    let gate = BrainEvent::from_safety_decision(
        BrainEventId("gate:1".to_string()),
        &safety_decision,
        "snapshot:108",
        TypedEventRef::new(proposal.event_id.clone(), BrainEventType::Proposal),
        108,
    );

    let mut command = accepted_event("command:1", BrainEventType::Command, "motor.gate", 109);
    command.authority = AuthoritySignificance::Command;
    command.links.parents.push(TypedEventRef::new(
        gate.event_id.clone(),
        BrainEventType::GateDecision,
    ));
    command.references.command_ids.push("create:42".to_string());
    command.payload = BrainEventPayload::inline(json!({ "forward": 0.0, "turn": 0.0 }));

    let mut outcome = accepted_event("outcome:1", BrainEventType::Outcome, "brainstem", 112);
    outcome.producer.brain = Brain::Brainstem;
    outcome.authority = AuthoritySignificance::Outcome;
    outcome.links.parents.push(TypedEventRef::new(
        command.event_id.clone(),
        BrainEventType::Command,
    ));
    outcome.references.command_ids.push("create:42".to_string());

    let events = vec![
        BrainEvent::from(&sensation),
        BrainEvent::from(&impression),
        BrainEvent::from(&experience),
        proposal,
        gate,
        command,
        outcome,
    ];
    for event in &events {
        event.validate().unwrap();
        let encoded = serde_json::to_vec(event).unwrap();
        let decoded = BrainEvent::decode_json(&encoded).unwrap();
        assert_eq!(&decoded, event);
    }

    let final_event = events.last().unwrap();
    let link = final_event.causal_references().next().unwrap();
    assert_eq!(link.relation, CausalRelationKind::Parent);
    assert_eq!(link.target.event_type, BrainEventType::Command);
    assert_eq!(link.target.event_id, BrainEventId("command:1".to_string()));
}

#[test]
fn adapters_preserve_missing_and_contradictory_evidence_as_typed_references() {
    let missing = test_id(10);
    let contradicting = test_id(11);
    let mut experience = Experience::new(
        "belief.person",
        "A person may be nearby.",
        Vec::new(),
        vec![missing],
        200,
        205,
    );
    experience.id = test_id(12);
    let mut event = BrainEvent::from(&experience);
    event.links.contradicts.push(TypedEventRef::new(
        BrainEventId::sensation(contradicting),
        BrainEventType::Evidence,
    ));

    event.validate().unwrap();
    let references = event.causal_references().collect::<Vec<_>>();
    assert!(references.iter().any(|reference| {
        reference.relation == CausalRelationKind::Supports
            && reference.target.event_id == BrainEventId::sensation(missing)
    }));
    assert!(references.iter().any(|reference| {
        reference.relation == CausalRelationKind::Contradicts
            && reference.target.event_id == BrainEventId::sensation(contradicting)
    }));
}

#[test]
fn raw_assets_and_vectors_are_referenced_instead_of_embedded() {
    let mut sensation = Sensation::primary(
        Modality::Vision,
        SensationSource::new("kinect.rgb"),
        300,
        305,
        SensationPayload {
            kind: SensationPayloadKind::ImageBytes,
            value: json!({ "bytes": vec![7_u8; 32_000] }),
        },
    );
    sensation.id = test_id(20);
    sensation.vector = Some(VectorEmbedding::new(
        vec![0.123_456; 4_096],
        "vision-model-v1",
        Modality::Vision,
        SensationPayloadKind::ImageBytes,
        sensation.id,
        306,
    ));

    let event = BrainEvent::from(&sensation);
    event.validate().unwrap();
    assert!(matches!(event.payload, BrainEventPayload::Reference { .. }));
    assert_eq!(event.artifacts.len(), 1);
    assert_eq!(
        event.artifacts[0].version.as_deref(),
        Some("vision-model-v1")
    );
    let encoded = serde_json::to_string(&event).unwrap();
    assert!(!encoded.contains("0.123456"));
}

#[test]
fn clock_and_calibration_epochs_survive_round_trip_and_expiry_is_epoch_aware() {
    let mut event = accepted_event(
        "calibration:imu:8",
        BrainEventType::CalibrationTransition,
        "imu.calibration",
        50,
    );
    event.times = EventTimes {
        occurred: ClockedTime::in_epoch(900, "brainstem-boot-a"),
        observed: ClockedTime::in_epoch(50, "motherbrain-boot-b"),
        valid_from: Some(ClockedTime::in_epoch(50, "motherbrain-boot-b")),
        expires_at: Some(ClockedTime::in_epoch(150, "motherbrain-boot-b")),
    };
    event.calibration_epochs = vec!["imu:8".to_string(), "mount:3".to_string()];
    event.quality.uncertainty = Some(Uncertainty {
        value: 1.8,
        measure: "standard_deviation".to_string(),
        unit: Some("degrees".to_string()),
    });
    event.artifacts.push(ArtifactIdentity {
        kind: ArtifactKind::Calibration,
        id: "imu-calibration".to_string(),
        version: Some("8".to_string()),
        checksum: Some("sha256:test".to_string()),
    });

    event.validate().unwrap();
    assert_eq!(
        event.disposition_at(&ClockedTime::in_epoch(149, "motherbrain-boot-b")),
        EventDisposition::Accepted
    );
    assert_eq!(
        event.disposition_at(&ClockedTime::in_epoch(150, "motherbrain-boot-b")),
        EventDisposition::Expired
    );
    assert_eq!(
        event.disposition_at(&ClockedTime::in_epoch(999, "motherbrain-boot-c")),
        EventDisposition::Accepted
    );

    let decoded = BrainEvent::decode_json(&serde_json::to_vec(&event).unwrap()).unwrap();
    assert_eq!(decoded.times, event.times);
    assert_eq!(decoded.calibration_epochs, event.calibration_epochs);
}

#[test]
fn deduplication_accepts_exact_replay_and_rejects_id_reuse() {
    let event = accepted_event(
        "belief:stable",
        BrainEventType::BeliefUpdate,
        "world-model",
        10,
    );
    let mut index = BrainEventIndex::default();

    assert_eq!(
        index.insert(event.clone()).unwrap(),
        DeduplicationOutcome::Inserted
    );
    assert_eq!(
        index.insert(event.clone()).unwrap(),
        DeduplicationOutcome::Duplicate
    );
    let mut conflict = event;
    conflict.disposition = EventDisposition::Rejected;
    assert!(matches!(
        index.insert(conflict),
        Err(BrainEventError::ConflictingDuplicate(_))
    ));
    assert_eq!(index.len(), 1);
}

#[test]
fn v0_fixture_migrates_and_future_schema_is_rejected() {
    let migrated =
        BrainEvent::decode_json(include_bytes!("../tests/fixtures/brain_event_v0.json")).unwrap();
    assert_eq!(migrated.schema_version, BRAIN_EVENT_SCHEMA_VERSION);
    assert_eq!(
        migrated.event_id,
        BrainEventId("legacy:evidence:1".to_string())
    );
    assert_eq!(migrated.event_type, BrainEventType::Evidence);
    assert_eq!(migrated.producer.component, "legacy.capture");

    let mut future = serde_json::to_value(&migrated).unwrap();
    future["schema_version"] = json!(BRAIN_EVENT_SCHEMA_VERSION + 1);
    assert!(matches!(
        BrainEvent::decode_json(&serde_json::to_vec(&future).unwrap()),
        Err(BrainEventError::UnsupportedSchema(2))
    ));
}

#[test]
fn safety_authority_and_calibration_events_cannot_be_coalesced() {
    for event_type in [
        BrainEventType::GateDecision,
        BrainEventType::Command,
        BrainEventType::Outcome,
        BrainEventType::CalibrationTransition,
        BrainEventType::TransportGap,
    ] {
        let event = accepted_event("critical:1", event_type, "critical", 10).coalescible("bad");
        assert!(matches!(
            event.validate(),
            Err(BrainEventError::LossIntolerantClassCoalesced)
        ));
    }
}

#[test]
fn estimator_authored_calibration_transition_preserves_evidence_provenance_and_artifacts() {
    let mut estimator = pete_now::CalibrationStateMachine::new(
        pete_now::RigidTransform3::default(),
        0,
        pete_now::CalibrationStateConfig {
            minimum_evidence_per_dof: 1,
            minimum_independent_sources: 1,
            minimum_trust_span_ms: 0,
            ..pete_now::CalibrationStateConfig::default()
        },
    );
    estimator.observe(
        pete_now::TransformEstimateEvidence {
            source: pete_now::CalibrationEvidenceSource::FloorPlane,
            captured_at_ms: 10,
            transform: pete_now::RigidTransform3 {
                translation_m: [0.0, 0.0, 0.4],
                ..pete_now::RigidTransform3::default()
            },
            observable_dofs: [false, false, true, true, true, false],
            covariance: [0.001; pete_now::TRANSFORM_DOF_COUNT],
            residuals: pete_now::CalibrationResiduals::default(),
        },
        12,
    );
    let transition = estimator.take_transitions().pop().unwrap();
    let events = BrainEvent::from_calibration_transition(&transition);
    assert_eq!(events.len(), 2);
    let evidence = &events[0];
    let canonical = &events[1];
    assert_eq!(evidence.event_type, BrainEventType::Evidence);
    assert_eq!(canonical.event_type, BrainEventType::CalibrationTransition);
    assert_eq!(canonical.links.supports.len(), 1);
    assert_eq!(canonical.links.supports[0].event_id, evidence.event_id);
    assert_eq!(evidence.artifacts[0].kind, ArtifactKind::Calibration);
    assert_eq!(canonical.artifacts.len(), 3);
    assert!(canonical
        .artifacts
        .iter()
        .all(|artifact| artifact.checksum.is_some()));
    assert_ne!(
        canonical.times.occurred.clock_epoch,
        canonical.times.observed.clock_epoch
    );
    for event in events {
        event.validate().unwrap();
    }
}

#[test]
fn live_and_replay_now_adaptation_is_identical_and_marks_projection() {
    let now = Now::blank(777, BodySense::default());
    let live =
        BrainEvent::from_now_snapshot("ledger-frame:abc", &now, 780, Some("boot:1".to_string()));
    let replay =
        BrainEvent::from_now_snapshot("ledger-frame:abc", &now, 780, Some("boot:1".to_string()));

    assert_eq!(live, replay);
    assert_eq!(live.record_kind, BrainEventRecordKind::StateProjection);
    assert_eq!(live.event_type, BrainEventType::Snapshot);
    assert!(matches!(live.payload, BrainEventPayload::Reference { .. }));
}

#[test]
fn reign_outcome_keeps_proposal_identity_and_disposition() {
    let outcome = ReignOutcome {
        input_id: test_id(30),
        accepted_by_conductor: false,
        vetoed_by_safety: true,
        final_action: Some(ActionPrimitive::Stop),
        reason: Some("cliff".to_string()),
    };
    let event =
        BrainEvent::from_reign_outcome(BrainEventId("reign-gate:30".to_string()), &outcome, 900);

    assert_eq!(event.disposition, EventDisposition::Vetoed);
    assert_eq!(event.authority, AuthoritySignificance::Gate);
    assert_eq!(
        event.links.parents[0],
        TypedEventRef::new(
            BrainEventId::from_domain("reign-input", outcome.input_id),
            BrainEventType::Proposal,
        )
    );
}

#[test]
fn legacy_provenance_maps_to_typed_supporting_links() {
    let source_id = test_id(40);
    let source = Event::new(1_000, EventKind::ObjectSeen)
        .with_provenance(Provenance::derived_from_sensations([source_id]).with_stage("vision"));
    let event = BrainEvent::from(&source);

    assert_eq!(event.event_type, BrainEventType::Evidence);
    assert_eq!(event.producer.component, "vision");
    assert_eq!(
        event.links.supports,
        vec![TypedEventRef::new(
            BrainEventId::sensation(source_id),
            BrainEventType::Evidence,
        )]
    );
}
