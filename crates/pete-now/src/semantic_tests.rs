
use super::*;
use crate::{BeliefMeta, BeliefSourceKind, ReachabilityEstimate};
use pete_body::BodySense;

fn charger(id: &str, confidence: f32) -> WorldEntity {
    WorldEntity {
        id: EntityId(id.to_string()),
        kind: WorldEntityKind::Charger,
        label: "charger".to_string(),
        confidence,
        last_observed_at_ms: 10,
        provenance: vec![semantic_evidence(10, "vision", id)],
        meta: BeliefMeta {
            source_kind: BeliefSourceKind::DirectObservation,
            ..BeliefMeta::default()
        },
        ..WorldEntity::default()
    }
}

#[test]
fn charger_meaning_is_stable_but_docking_remains_conditional() {
    let mut builder = SemanticGraphBuilder::default();
    let now = Now::blank(10, BodySense::default());
    let charger = charger("charger:17", 0.9);
    let snapshot = builder.update(
        &now,
        &BTreeMap::from([(charger.id.clone(), charger.clone())]),
        &[],
    );
    assert!(snapshot.supports(
        &SemanticNodeRef::Entity(charger.id.clone()),
        SemanticPredicate::IsA,
        &SemanticNodeRef::Concept(SemanticConceptId("charger".to_string())),
        0.8,
    ));
    let dock_relation = snapshot
        .relations
        .values()
        .find(|relation| {
            relation.subject == SemanticNodeRef::Concept(SemanticConceptId("charger".to_string()))
                && relation.predicate == SemanticPredicate::Affords
                && relation.object
                    == SemanticNodeRef::Behavior(SemanticBehaviorId("dock".to_string()))
        })
        .unwrap();
    assert_eq!(
        dock_relation.context.conditions.get("requires"),
        Some(&"near_aligned_compatible".to_string())
    );
}

#[test]
fn successful_charging_strengthens_instance_restoration_evidence() {
    let mut builder = SemanticGraphBuilder::default();
    let mut now = Now::blank(10, BodySense::default());
    let charger = charger("charger:17", 0.7);
    let entities = BTreeMap::from([(charger.id.clone(), charger.clone())]);
    builder.update(&now, &entities, &[]);
    now.t_ms = 20;
    now.body.charging = true;
    let first = builder.update(&now, &entities, &[]);
    let relation = first
        .relations
        .values()
        .find(|relation| {
            relation.subject == SemanticNodeRef::Entity(charger.id.clone())
                && relation.predicate == SemanticPredicate::Restores
        })
        .unwrap();
    assert!(relation.confidence >= 0.7);
    assert!(relation
        .supporting_evidence
        .iter()
        .any(|evidence| { evidence.source == "body.charging" }));
}

#[test]
fn contradicted_false_charger_does_not_keep_strong_charger_semantics() {
    let mut builder = SemanticGraphBuilder::default();
    let now = Now::blank(10, BodySense::default());
    let charger = charger("entity:false-dock", 0.6);
    let entities = BTreeMap::from([(charger.id.clone(), charger.clone())]);
    builder.update(&now, &entities, &[]);
    let contradiction = SemanticEvidenceObservation {
        subject: SemanticNodeRef::Entity(charger.id.clone()),
        predicate: SemanticPredicate::IsA,
        object: SemanticNodeRef::Concept(SemanticConceptId("charger".to_string())),
        context: organism_context(),
        confidence: 1.0,
        grounding: SemanticGroundingKind::ActionOutcome,
        evidence: semantic_evidence(20, "dock.outcome", "not_a_charger"),
        contradicts: true,
    };
    let snapshot = builder.update(
        &Now::blank(20, BodySense::default()),
        &entities,
        &[contradiction],
    );
    assert!(!snapshot.supports(
        &SemanticNodeRef::Entity(charger.id),
        SemanticPredicate::IsA,
        &SemanticNodeRef::Concept(SemanticConceptId("charger".to_string())),
        0.5,
    ));
}

#[test]
fn repeated_approach_progress_strengthens_expected_effect() {
    let mut builder = SemanticGraphBuilder::default();
    let now = Now::blank(10, BodySense::default());
    let observation = |time| {
        SemanticEvidenceObservation::supported(
            SemanticNodeRef::Behavior(SemanticBehaviorId("approach_charger".to_string())),
            SemanticPredicate::Predicts,
            SemanticNodeRef::Outcome(SemanticOutcomeId("target_distance_decreases".to_string())),
            0.8,
            SemanticGroundingKind::ActionOutcome,
            semantic_evidence(time, "goal.progress", "target_distance_decreased"),
        )
    };
    let first = builder.update(&now, &BTreeMap::new(), &[observation(10)]);
    let first_confidence = first
        .relations
        .values()
        .filter(|relation| {
            relation.subject
                == SemanticNodeRef::Behavior(SemanticBehaviorId("approach_charger".to_string()))
                && relation.predicate == SemanticPredicate::Predicts
        })
        .map(|relation| relation.confidence)
        .fold(0.0f32, f32::max);
    let second = builder.update(
        &Now::blank(20, BodySense::default()),
        &BTreeMap::new(),
        &[observation(20)],
    );
    let second_confidence = second
        .relations
        .values()
        .filter(|relation| {
            relation.subject
                == SemanticNodeRef::Behavior(SemanticBehaviorId("approach_charger".to_string()))
                && relation.predicate == SemanticPredicate::Predicts
        })
        .map(|relation| relation.confidence)
        .fold(0.0f32, f32::max);
    assert!(second_confidence > first_confidence);
}

#[test]
fn temporal_sequence_alone_remains_predictive_not_causal() {
    let mut builder = SemanticGraphBuilder::default();
    let observation = SemanticEvidenceObservation::supported(
        SemanticNodeRef::Action(SemanticActionId("turn".to_string())),
        SemanticPredicate::Causes,
        SemanticNodeRef::Outcome(SemanticOutcomeId("sound_changed".to_string())),
        0.7,
        SemanticGroundingKind::TemporalSequence,
        semantic_evidence(10, "sequence", "turn_then_sound"),
    );
    let snapshot = builder.update(
        &Now::blank(10, BodySense::default()),
        &BTreeMap::new(),
        &[observation],
    );
    assert!(snapshot
        .relations
        .values()
        .any(|relation| relation.predicate == SemanticPredicate::Predicts));
    assert!(!snapshot
        .relations
        .values()
        .any(|relation| relation.predicate == SemanticPredicate::Causes));
}

#[test]
fn human_naming_is_sourced_without_overwriting_identity() {
    let mut builder = SemanticGraphBuilder::default();
    let observation = SemanticEvidenceObservation::supported(
        SemanticNodeRef::Entity(EntityId("entity:unknown".to_string())),
        SemanticPredicate::NamedBy,
        SemanticNodeRef::Concept(SemanticConceptId("home".to_string())),
        0.8,
        SemanticGroundingKind::HumanInstruction,
        semantic_evidence(10, "human.claim", "call_this_home"),
    );
    let snapshot = builder.update(
        &Now::blank(10, BodySense::default()),
        &BTreeMap::new(),
        &[observation],
    );
    let relation = snapshot
        .relations
        .values()
        .find(|relation| relation.predicate == SemanticPredicate::NamedBy)
        .unwrap();
    assert!(relation
        .supporting_evidence
        .iter()
        .any(|evidence| evidence.source == "human.claim"));
    assert!(!snapshot.relations.values().any(|relation| {
        relation.subject == SemanticNodeRef::Entity(EntityId("entity:unknown".to_string()))
            && relation.predicate == SemanticPredicate::SameEntityAs
    }));
}

#[test]
fn blocked_route_relation_is_contextual_and_preserves_charger_meaning() {
    let mut builder = SemanticGraphBuilder::default();
    let now = Now::blank(10, BodySense::default());
    let mut charger = charger("charger:17", 0.9);
    charger.reachability = ReachabilityEstimate {
        reachable: false,
        confidence: 0.9,
    };
    let obstacle = WorldEntity {
        id: EntityId("obstacle:1".to_string()),
        kind: WorldEntityKind::Obstacle,
        label: "box".to_string(),
        confidence: 0.9,
        distance_m: Some(0.5),
        last_observed_at_ms: 10,
        ..WorldEntity::default()
    };
    let snapshot = builder.update(
        &now,
        &BTreeMap::from([
            (charger.id.clone(), charger.clone()),
            (obstacle.id.clone(), obstacle.clone()),
        ]),
        &[],
    );
    assert!(snapshot.relations.values().any(|relation| {
        relation.subject == SemanticNodeRef::Entity(obstacle.id.clone())
            && relation.predicate == SemanticPredicate::Blocks
            && relation.object == SemanticNodeRef::Entity(charger.id.clone())
            && relation.context.conditions.get("route_state")
                == Some(&"currently_blocked".to_string())
    }));
    assert!(!snapshot
        .charger_explanation(&charger.id)
        .relation_ids
        .is_empty());
}
