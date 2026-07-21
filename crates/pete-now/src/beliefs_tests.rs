use super::*;
use crate::{ObjectObservation, ObjectSense, TypedTimestamp, VectorArtifact};
use pete_actions::{ReignCommand, ReignMode, ReignSource};
use pete_body::BodySense;
use uuid::Uuid;

fn cognition_registry_now(t_ms: u64, host: &str, process: &str, state: ProviderHealthState) -> Now {
    let mut now = Now::blank(t_ms, BodySense::default());
    let provider = pete_cognition::CognitiveProviderDescriptor {
        provider_id: pete_cognition::ProviderId("scene-provider".to_string()),
        role: pete_cognition::CognitiveRole::CognitiveAccelerator,
        host_id: Some(pete_cognition::HostId(host.to_string())),
        process_id: Some(pete_cognition::ProcessId(process.to_string())),
        implementation: "fixture".to_string(),
        implementation_version: "1".to_string(),
        capabilities: vec![pete_cognition::CapabilityDescriptor {
            capability: pete_cognition::CognitiveCapability::DescribeScene,
            version: "1".to_string(),
            performance_confidence: 0.9,
            ..pete_cognition::CapabilityDescriptor::default()
        }],
        health: pete_cognition::ProviderHealth {
            state,
            confidence: 1.0,
            observed_at_ms: t_ms,
            valid_until_ms: t_ms + 1_000,
            reason: (state == ProviderHealthState::Disconnected)
                .then_some("provider disconnected".to_string()),
        },
        locality: pete_cognition::Locality::LocalNetwork,
        ..pete_cognition::CognitiveProviderDescriptor::default()
    };
    now.extensions.insert(
        "cognition.registry".to_string(),
        serde_json::to_value(ProviderRegistrySnapshot {
            schema_version: 1,
            revision: t_ms,
            observed_at_ms: t_ms,
            providers: BTreeMap::from([(provider.provider_id.clone(), provider)]),
        })
        .unwrap(),
    );
    now
}

fn observed_now(t_ms: u64, class: ObjectClass, label: &str) -> Now {
    let mut now = Now::blank(t_ms, BodySense::default());
    now.objects = ObjectSense {
        schema_version: 1,
        observations: vec![ObjectObservation {
            label: label.to_string(),
            class,
            bearing_rad: 0.2,
            distance_m: Some(1.0),
            confidence: 0.9,
            source: ObjectObservationSource::Sim,
        }],
        ..ObjectSense::default()
    };
    now
}

#[test]
fn stale_target_loses_bearing_without_erasing_identity() {
    let mut updater = WorldModelUpdater::default();
    let first = updater.update(
        observed_now(0, ObjectClass::Charger, "dock"),
        WorldModelUpdateContext::default(),
    );
    let id = first.world.entities.keys().next().unwrap().clone();
    let stale = updater.update(
        Now::blank(2_100, BodySense::default()),
        WorldModelUpdateContext::default(),
    );
    assert!(stale.world.entities.contains_key(&id));
    assert!(stale.world.entities[&id].bearing_rad.is_none());
}

#[test]
fn contradictory_claims_coexist_and_are_explicit() {
    let mut updater = WorldModelUpdater::default();
    updater.update(
        observed_now(0, ObjectClass::Person, "Alex"),
        WorldModelUpdateContext::default(),
    );
    let next = updater.update(
        observed_now(1, ObjectClass::Charger, "Alex"),
        WorldModelUpdateContext::default(),
    );
    assert_eq!(next.world.entities.len(), 2);
    assert!(next
        .world
        .entities
        .values()
        .all(|entity| !entity.meta.contradiction_refs.is_empty()));
}

#[test]
fn fixed_evidence_sequence_is_deterministic() {
    let sequence = || {
        let mut updater = WorldModelUpdater::default();
        updater
            .update(
                observed_now(10, ObjectClass::Charger, "dock"),
                WorldModelUpdateContext::default(),
            )
            .world
    };
    assert_eq!(sequence(), sequence());
}

#[test]
fn memory_and_direct_observation_remain_distinguishable() {
    let mut updater = WorldModelUpdater::default();
    let mut now = observed_now(10, ObjectClass::Charger, "dock");
    now.memory.nearby_best_charge_direction_rad = Some(-0.4);
    now.memory.place_charge_value = 0.8;
    now.memory.map_confidence = 0.7;
    let snapshot = updater
        .update(now, WorldModelUpdateContext::default())
        .world;
    let sources = snapshot
        .entities
        .values()
        .map(|entity| entity.meta.source_kind.clone())
        .collect::<BTreeSet<_>>();
    assert!(sources.contains(&BeliefSourceKind::DirectObservation));
    assert!(sources.contains(&BeliefSourceKind::MemoryRecall));
}

#[test]
fn missing_modalities_remain_missing_beliefs() {
    let mut updater = WorldModelUpdater::default();
    let snapshot = updater
        .update(
            Now::blank(10, BodySense::default()),
            WorldModelUpdateContext::default(),
        )
        .world;
    assert!(snapshot.entities.is_empty());
    assert!(snapshot.context.prediction_uncertainty.is_none());
    assert!(snapshot.context.llm_confidence.is_none());
    assert!(snapshot.local_geometry.nearest_m.is_none());
    assert!(snapshot.local_geometry.center_clearance_m.is_none());
}

#[test]
fn entity_belief_is_traceable_to_input_evidence() {
    let mut updater = WorldModelUpdater::default();
    let snapshot = updater
        .update(
            observed_now(10, ObjectClass::Charger, "dock"),
            WorldModelUpdateContext::default(),
        )
        .world;
    let charger = snapshot.entities.values().next().unwrap();
    let evidence_id = &charger.meta.provenance[0].id;
    assert!(snapshot
        .update_trace
        .input_evidence_ids
        .contains(evidence_id));
    assert!(charger.meta.provenance[0]
        .transformation_lineage
        .contains(&"object-observation-v1".to_string()));
}

#[test]
fn learned_latent_extension_cannot_erase_contact_belief() {
    let mut updater = WorldModelUpdater::default();
    let mut body = BodySense::default();
    body.flags.bump_left = true;
    let mut now = Now::blank(10, body);
    now.extensions.insert(
        "experience.latent".to_string(),
        serde_json::json!({"danger": 0.0, "contact": false}),
    );
    let snapshot = updater
        .update(now, WorldModelUpdateContext::default())
        .world;
    assert!(snapshot.self_model.contact);
    assert_eq!(
        snapshot
            .hazards
            .immediate_risk
            .as_ref()
            .map(|belief| belief.value),
        Some(1.0)
    );
    assert_eq!(
        snapshot.self_model.battery_meta.source_kind,
        BeliefSourceKind::DirectObservation
    );
}

#[test]
fn local_geometry_is_typed_derived_belief() {
    let mut updater = WorldModelUpdater::default();
    let mut now = Now::blank(10, BodySense::default());
    now.range.nearest_m = Some(0.2);
    now.range.beams = vec![0.9, 0.8, 0.7, 0.5, 0.4, 0.6, 0.3, 0.2, 0.1];
    let snapshot = updater
        .update(now, WorldModelUpdateContext::default())
        .world;
    assert_eq!(
        snapshot
            .local_geometry
            .center_clearance_m
            .as_ref()
            .map(|belief| belief.value),
        Some(0.3)
    );
    assert_eq!(
        snapshot
            .local_geometry
            .center_clearance_m
            .as_ref()
            .map(|belief| &belief.meta.source_kind),
        Some(&BeliefSourceKind::DirectObservation)
    );
}

#[test]
fn higher_brain_loss_removes_enhanced_capability_not_organism_identity() {
    let service = CognitiveServiceBelief {
        available: true,
        busy: true,
        confidence: 1.0,
        meta: simple_meta(10, BeliefSourceKind::Map, "service.rich_language"),
        ..CognitiveServiceBelief::default()
    };
    let mut updater = WorldModelUpdater::default();
    let first = updater.update(
        Now::blank(10, BodySense::default()),
        WorldModelUpdateContext {
            cognitive_services: BTreeMap::from([("rich_language".to_string(), service)]),
            ..WorldModelUpdateContext::default()
        },
    );
    let second = updater.update(
        Now::blank(20, BodySense::default()),
        WorldModelUpdateContext::default(),
    );
    assert_eq!(
        first.world.self_model.organism_id.value,
        second.world.self_model.organism_id.value
    );
    assert!(first
        .world
        .self_model
        .capabilities
        .is_available("service:rich_language"));
    assert!(first.world.self_model.service_state.services["rich_language"].busy);
    assert!(!second
        .world
        .self_model
        .capabilities
        .is_available("service:rich_language"));
}

#[test]
fn cognitive_service_busy_defaults_false_for_older_snapshots() {
    let mut value = serde_json::to_value(CognitiveServiceBelief::default()).unwrap();
    value
        .as_object_mut()
        .expect("service belief object")
        .remove("busy");

    let service: CognitiveServiceBelief = serde_json::from_value(value).unwrap();

    assert!(!service.busy);
}

#[test]
fn provider_disconnect_and_restart_change_capability_not_organism_identity() {
    let mut updater = WorldModelUpdater::default();
    let available = updater.update(
        cognition_registry_now(
            10,
            "accelerator-a",
            "process-1",
            ProviderHealthState::Available,
        ),
        WorldModelUpdateContext::default(),
    );
    let disconnected = updater.update(
        cognition_registry_now(
            20,
            "accelerator-a",
            "process-1",
            ProviderHealthState::Disconnected,
        ),
        WorldModelUpdateContext::default(),
    );
    let restarted = updater.update(
        cognition_registry_now(
            30,
            "accelerator-b",
            "process-2",
            ProviderHealthState::Available,
        ),
        WorldModelUpdateContext::default(),
    );
    assert_eq!(
        available.world.self_model.organism_id.value,
        disconnected.world.self_model.organism_id.value
    );
    assert_eq!(
        available.world.self_model.organism_id.value,
        restarted.world.self_model.organism_id.value
    );
    assert!(available
        .world
        .self_model
        .capabilities
        .is_available("service:describe_scene"));
    assert!(!disconnected
        .world
        .self_model
        .capabilities
        .is_available("service:describe_scene"));
    assert!(restarted
        .world
        .self_model
        .capabilities
        .is_available("service:describe_scene"));
    assert_eq!(
        restarted.world.self_model.service_state.services["describe_scene"]
            .host_id
            .as_ref()
            .map(|id| id.0.as_str()),
        Some("accelerator-b")
    );
    let service = &restarted.world.self_model.service_state.services["describe_scene"];
    assert_eq!(service.role.as_deref(), Some("cognitive_accelerator"));
    assert_eq!(service.locality.as_deref(), Some("local_network"));
    assert_eq!(service.resource_class.as_deref(), Some("unknown"));
}

#[test]
fn brainstem_reboot_invalidates_authority_but_not_identity() {
    let with_boot = |t_ms, boot: &str, lease: &str| {
        let mut now = Now::blank(t_ms, BodySense::default());
        now.extensions.insert(
            "brainstem.possession".to_string(),
            serde_json::json!({
                "brainstem_device_id": "device-7",
                "brainstem_boot_id": boot,
                "session_id": "session-1",
                "lease_id": lease,
                "possessed": true,
                "brainstem_armed": true
            }),
        );
        now
    };
    let mut updater = WorldModelUpdater::default();
    let first = updater.update(
        with_boot(10, "boot-a", "lease-1"),
        WorldModelUpdateContext::default(),
    );
    let rebooted = updater.update(
        with_boot(20, "boot-b", "lease-1"),
        WorldModelUpdateContext::default(),
    );
    assert_eq!(
        first.world.self_model.organism_id.value,
        rebooted.world.self_model.organism_id.value
    );
    assert!(first.world.self_model.agency.possessed.value);
    assert!(!rebooted.world.self_model.agency.possessed.value);
    assert!(!rebooted.world.self_model.agency.armed.value);
    assert!(!rebooted
        .world
        .self_model
        .capabilities
        .is_authorized("actuator:drive"));
    assert!(!rebooted
        .world
        .self_model
        .agency
        .authority_conflicts
        .is_empty());
    let still_invalid = updater.update(
        with_boot(30, "boot-b", "lease-1"),
        WorldModelUpdateContext::default(),
    );
    assert!(!still_invalid.world.self_model.agency.possessed.value);
    let reacquired = updater.update(
        with_boot(40, "boot-b", "lease-2"),
        WorldModelUpdateContext::default(),
    );
    assert!(reacquired.world.self_model.agency.possessed.value);
    assert!(reacquired
        .world
        .self_model
        .capabilities
        .is_authorized("actuator:drive"));
}

#[test]
fn direct_reign_is_attributed_to_operator() {
    let mut now = Now::blank(10, BodySense::default());
    now.reign.latest = Some(ReignInput {
        id: Uuid::nil(),
        issued_at_ms: 0,
        expires_at_ms: 100,
        source: ReignSource::HumanSupervisor,
        mode: ReignMode::Direct,
        command: ReignCommand::Stop,
        priority: 1.0,
        note: None,
    });
    let mut updater = WorldModelUpdater::default();
    let snapshot = updater
        .update(now, WorldModelUpdateContext::default())
        .world;
    assert_eq!(
        snapshot.self_model.agency.controller,
        ControlProvenance::HumanDirect
    );
    assert_eq!(
        snapshot.self_model.active_control.provenance,
        ControlProvenance::HumanDirect
    );
    assert!(snapshot.self_model.agency.pending_direct_override);
}

#[test]
fn missing_camera_removes_visual_capability_but_preserves_memory() {
    let mut updater = WorldModelUpdater::default();
    updater.update(
        observed_now(10, ObjectClass::Person, "Alex"),
        WorldModelUpdateContext::default(),
    );
    let snapshot = updater
        .update(
            Now::blank(20, BodySense::default()),
            WorldModelUpdateContext::default(),
        )
        .world;
    assert!(!snapshot
        .self_model
        .capabilities
        .is_available("sensor:vision"));
    assert!(snapshot
        .entities
        .values()
        .any(|entity| entity.label == "Alex"));
}

#[test]
fn autonomic_preemption_and_history_have_separate_typed_regions() {
    let context = WorldModelUpdateContext {
        active_control: Some(ActiveControlSummary {
            provenance: ControlProvenance::AutonomicReflex,
            safety_preempted: true,
            veto_reasons: vec!["contact".to_string()],
            ..ActiveControlSummary::default()
        }),
        continuity: ContinuitySummary {
            recent_experience_refs: vec!["experience:old-bump".to_string()],
            recent_self_action_refs: vec!["action:reverse".to_string()],
            ..ContinuitySummary::default()
        },
        ..WorldModelUpdateContext::default()
    };
    let mut updater = WorldModelUpdater::default();
    let snapshot = updater
        .update(Now::blank(10, BodySense::default()), context)
        .world;
    assert_eq!(
        snapshot.self_model.active_control.provenance,
        ControlProvenance::AutonomicReflex
    );
    assert!(snapshot.self_model.active_control.safety_preempted);
    assert_eq!(
        snapshot.self_model.continuity.recent_experience_refs,
        vec!["experience:old-bump"]
    );
    assert!(!snapshot.self_model.contact);
}

#[test]
fn delayed_evidence_keeps_event_observation_and_replay_times_distinct() {
    let mut now = Now::blank(500, BodySense::default());
    now.face
        .vectors
        .push(VectorArtifact::new("faces", "face:delayed", vec![0.1]).with_occurred_at_ms(100));
    let mut updater = WorldModelUpdater::default();
    let snapshot = updater
        .update(
            now,
            WorldModelUpdateContext {
                wall_clock_unix_ms: Some(50_000),
                replay_now_ms: Some(40),
                ..WorldModelUpdateContext::default()
            },
        )
        .world;
    assert!(snapshot
        .temporal
        .current_temporal_beliefs
        .iter()
        .any(|belief| {
            belief.interval.domain == ClockDomain::Event && belief.interval.start_ms == 100
        }));
    assert!(snapshot
        .temporal
        .current_temporal_beliefs
        .iter()
        .any(|belief| {
            belief.interval.domain == ClockDomain::Observation && belief.interval.start_ms == 500
        }));
    assert_eq!(
        snapshot.temporal.replay_now,
        Some(TypedTimestamp {
            domain: ClockDomain::Replay,
            ms: 40,
        })
    );
}
