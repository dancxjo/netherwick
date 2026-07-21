
use super::*;
use crate::{
    GraphEntity, MemorySense, ObjectClass, ObjectObservation, ObjectObservationSource, ObjectSense,
    VectorArtifact,
};
use pete_body::BodySense;

fn person_now(t_ms: u64, label: &str, confidence: f32) -> Now {
    let mut now = Now::blank(t_ms, BodySense::default());
    now.objects = ObjectSense {
        schema_version: 1,
        observations: vec![ObjectObservation {
            label: label.to_string(),
            class: ObjectClass::Person,
            bearing_rad: 0.1,
            distance_m: Some(1.0),
            confidence,
            source: ObjectObservationSource::Kinect,
        }],
        ..ObjectSense::default()
    };
    now
}

#[test]
fn multimodal_repetition_strengthens_identity_without_granting_authority() {
    let mut builder = SocialWorldModelBuilder::default();
    let mut now = person_now(10, "Alex", 0.55);
    now.face
        .vectors
        .push(VectorArtifact::new("faces", "face:alex", vec![0.1]).with_source_id("Alex"));
    now.voice
        .vectors
        .push(VectorArtifact::new("voices", "voice:alex", vec![0.2]).with_source_id("Alex"));
    let mut entity = WorldEntity {
        id: EntityId("person:alex".to_string()),
        kind: WorldEntityKind::Person,
        label: "Alex".to_string(),
        confidence: 0.55,
        last_observed_at_ms: 10,
        ..WorldEntity::default()
    };
    let entities = BTreeMap::from([(entity.id.clone(), entity.clone())]);
    let first = builder.update(&now, &entities);
    now.t_ms = 20;
    entity.last_observed_at_ms = 20;
    let entities = BTreeMap::from([(entity.id.clone(), entity)]);
    let second = builder.update(&now, &entities);
    let person = second.people.values().next().unwrap();
    assert!(
        person.best_identity().unwrap().confidence
            > first
                .people
                .values()
                .next()
                .unwrap()
                .best_identity()
                .unwrap()
                .confidence
    );
    assert!(person
        .best_identity()
        .unwrap()
        .modalities
        .contains(&IdentityModality::Face));
    assert!(person
        .best_identity()
        .unwrap()
        .modalities
        .contains(&IdentityModality::Voice));
    assert!(second
        .relationships
        .values()
        .all(|relationship| relationship.caregiving_or_authority.is_none()));
}

#[test]
fn stale_presence_preserves_durable_identity_and_closes_interaction() {
    let mut builder = SocialWorldModelBuilder::default();
    let now = person_now(0, "Alex", 0.9);
    let entity = WorldEntity {
        id: EntityId("person:alex".to_string()),
        kind: WorldEntityKind::Person,
        label: "Alex".to_string(),
        confidence: 0.9,
        last_observed_at_ms: 0,
        ..WorldEntity::default()
    };
    builder.update(&now, &BTreeMap::from([(entity.id.clone(), entity)]));
    let stale = builder.update(&Now::blank(2_000, BodySense::default()), &BTreeMap::new());
    assert_eq!(stale.people.len(), 1);
    assert!(!stale.people.values().next().unwrap().presence.present);
    assert!(stale.active_interaction.is_none());
    assert_eq!(stale.recent_interactions.len(), 1);
}

#[test]
fn completed_lua_greeting_acknowledges_only_the_current_encounter() {
    let mut builder = SocialWorldModelBuilder::default();
    let first_now = person_now(0, "Alex", 0.9);
    let mut entity = WorldEntity {
        id: EntityId("person:alex".to_string()),
        kind: WorldEntityKind::Person,
        label: "Alex".to_string(),
        confidence: 0.9,
        last_observed_at_ms: 0,
        ..WorldEntity::default()
    };
    let first = builder.update(
        &first_now,
        &BTreeMap::from([(entity.id.clone(), entity.clone())]),
    );
    let encounter_id = first
        .active_interaction
        .as_ref()
        .unwrap()
        .interaction_id
        .0
        .clone();

    let mut forged_now = person_now(50, "Alex", 0.9);
    forged_now.extensions.insert(
        "motherbrain.skill_execution".to_string(),
        serde_json::json!({
            "execution_id": 40,
            "skill": {
                "skill_id": "motherbrain.untrusted",
                "source_hash": "forged",
            },
            "diagnostics": {"terminal_outcome": "completed"},
            "observations": [{
                "kind": "social_acknowledgment",
                "value": {
                    "interaction_id": encounter_id,
                    "person_id": "person:alex",
                    "occurred_at_ms": 50,
                },
                "provenance": "lua_skill",
            }],
        }),
    );
    entity.last_observed_at_ms = 50;
    let forged = builder.update(
        &forged_now,
        &BTreeMap::from([(entity.id.clone(), entity.clone())]),
    );
    assert!(forged
        .active_interaction
        .as_ref()
        .unwrap()
        .acknowledgments
        .is_empty());

    let mut acknowledged_now = person_now(100, "Alex", 0.9);
    acknowledged_now.extensions.insert(
        "motherbrain.skill_execution".to_string(),
        serde_json::json!({
            "execution_id": 41,
            "skill": {
                "skill_id": "motherbrain.greet",
                "source_hash": "abc123",
            },
            "diagnostics": {"terminal_outcome": "completed"},
            "observations": [{
                "kind": "social_acknowledgment",
                "contract": "host_validated_social_acknowledgment_v1",
                "value": {
                    "acknowledgment_id": format!("greet:{encounter_id}:person:alex:41"),
                    "interaction_id": encounter_id,
                    "person_id": "person:alex",
                    "occurred_at_ms": 100,
                },
                "provenance": "lua_skill",
            }],
        }),
    );
    entity.last_observed_at_ms = 100;
    let acknowledged = builder.update(
        &acknowledged_now,
        &BTreeMap::from([(entity.id.clone(), entity.clone())]),
    );
    let interaction = acknowledged.active_interaction.as_ref().unwrap();
    assert_eq!(interaction.acknowledgments.len(), 1);
    assert!(interaction.has_acknowledgment(
        &PersonId("person:alex".to_string()),
        SocialAcknowledgmentKind::GreetingAttempted,
    ));
    assert_eq!(interaction.phase, InteractionPhase::Engaged);
    assert_eq!(interaction.acknowledgments[0].skill_execution_id, 41);
    assert_eq!(
        interaction.acknowledgments[0].provenance[0]
            .implementation_version
            .as_deref(),
        Some("abc123")
    );

    entity.last_observed_at_ms = 200;
    let duplicate = builder.update(
        &acknowledged_now,
        &BTreeMap::from([(entity.id.clone(), entity.clone())]),
    );
    assert_eq!(
        duplicate
            .active_interaction
            .as_ref()
            .unwrap()
            .acknowledgments
            .len(),
        1
    );

    builder.update(&Now::blank(2_000, BodySense::default()), &BTreeMap::new());
    let returned_now = person_now(3_000, "Alex", 0.9);
    entity.last_observed_at_ms = 3_000;
    let returned = builder.update(
        &returned_now,
        &BTreeMap::from([(entity.id.clone(), entity)]),
    );
    let new_interaction = returned.active_interaction.as_ref().unwrap();
    assert_ne!(new_interaction.interaction_id.0, encounter_id);
    assert!(new_interaction.acknowledgments.is_empty());
}

#[test]
fn recognized_face_without_object_detection_opens_a_stable_named_encounter() {
    let mut builder = SocialWorldModelBuilder::default();
    let mut first_now = Now::blank(100, BodySense::default());
    first_now.face.vectors.push(
        VectorArtifact::new("faces", "face-crop-1", vec![0.1, 0.2])
            .with_model("face-model")
            .with_source_id("Alex"),
    );
    first_now.memory.face_familiarity = 0.92;
    first_now.memory.remembered_entities.push(GraphEntity {
        id: "person:alex".to_string(),
        labels: vec!["Person".to_string()],
        summary: "Alex".to_string(),
        score: 0.90,
    });

    let first = builder.update(&first_now, &BTreeMap::new());
    let interaction = first.active_interaction.as_ref().unwrap();
    let encounter_id = interaction.interaction_id.clone();
    assert_eq!(
        interaction.participants,
        vec![PersonId("person:alex".to_string())]
    );
    let alex = first
        .people
        .get(&PersonId("person:alex".to_string()))
        .unwrap();
    assert_eq!(
        alex.preferred_name.as_ref().map(|name| name.value.as_str()),
        Some("Alex")
    );
    assert!(!alex.identity_is_uncertain());
    assert_eq!(
        alex.location
            .as_ref()
            .and_then(|location| location.bearing_rad),
        Some(0.0)
    );

    let mut repeated_now = first_now.clone();
    repeated_now.t_ms = 200;
    repeated_now.face.vectors[0].point_id = "face-crop-2".to_string();
    let repeated = builder.update(&repeated_now, &BTreeMap::new());
    assert_eq!(
        repeated.active_interaction.as_ref().unwrap().interaction_id,
        encounter_id
    );
}

#[test]
fn face_voice_mismatch_remains_an_explicit_identity_contradiction() {
    let mut builder = SocialWorldModelBuilder::default();
    let mut now = person_now(10, "Person", 0.7);
    now.face
        .vectors
        .push(VectorArtifact::new("faces", "face:alex", vec![0.1]).with_source_id("person:alex"));
    now.voice
        .vectors
        .push(VectorArtifact::new("voices", "voice:bob", vec![0.2]).with_source_id("speaker:bob"));
    let entity = WorldEntity {
        id: EntityId("person:person".to_string()),
        kind: WorldEntityKind::Person,
        label: "Person".to_string(),
        confidence: 0.7,
        last_observed_at_ms: 10,
        ..WorldEntity::default()
    };
    let snapshot = builder.update(&now, &BTreeMap::from([(entity.id.clone(), entity)]));
    let person = snapshot.people.values().next().unwrap();
    assert!(person
        .identity_hypotheses
        .iter()
        .filter(|identity| identity.identity_key.starts_with("biometric:"))
        .all(|identity| !identity.contradiction_refs.is_empty()));
}

#[test]
fn recalled_person_is_durable_history_not_current_presence() {
    let mut builder = SocialWorldModelBuilder::default();
    let mut now = Now::blank(100, BodySense::default());
    now.memory = MemorySense {
        remembered_entities: vec![GraphEntity {
            id: "person:alex".to_string(),
            labels: vec!["Person".to_string()],
            summary: "Alex".to_string(),
            score: 0.8,
        }],
        ..MemorySense::default()
    };
    let snapshot = builder.update(&now, &BTreeMap::new());
    let person = snapshot
        .people
        .get(&PersonId("person:alex".to_string()))
        .unwrap();
    assert!(!person.presence.present);
    assert_eq!(person.meta.source_kind, BeliefSourceKind::MemoryRecall);
}
