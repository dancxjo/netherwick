use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{EntityId, EvidenceRef, Now, WorldEntity, WorldEntityKind};

macro_rules! semantic_id {
    ($name:ident) => {
        #[derive(
            Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub String);
    };
}

semantic_id!(SemanticRelationId);
semantic_id!(SemanticConceptId);
semantic_id!(SemanticActionId);
semantic_id!(SemanticSkillId);
semantic_id!(SemanticBehaviorId);
semantic_id!(SemanticGoalId);
semantic_id!(SemanticDriveId);
semantic_id!(SemanticOutcomeId);
semantic_id!(SemanticPropertyId);
semantic_id!(SemanticPlaceId);
semantic_id!(SemanticEpisodeId);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum SemanticNodeRef {
    Entity(EntityId),
    Place(SemanticPlaceId),
    Person(crate::PersonId),
    Action(SemanticActionId),
    Skill(SemanticSkillId),
    Behavior(SemanticBehaviorId),
    Goal(SemanticGoalId),
    Drive(SemanticDriveId),
    Outcome(SemanticOutcomeId),
    Property(SemanticPropertyId),
    Concept(SemanticConceptId),
    Episode(SemanticEpisodeId),
}

impl Default for SemanticNodeRef {
    fn default() -> Self {
        Self::Concept(SemanticConceptId::default())
    }
}

impl SemanticNodeRef {
    pub fn stable_key(&self) -> String {
        match self {
            Self::Entity(id) => format!("entity:{}", id.0),
            Self::Place(id) => format!("place:{}", id.0),
            Self::Person(id) => format!("person:{}", id.0),
            Self::Action(id) => format!("action:{}", id.0),
            Self::Skill(id) => format!("skill:{}", id.0),
            Self::Behavior(id) => format!("behavior:{}", id.0),
            Self::Goal(id) => format!("goal:{}", id.0),
            Self::Drive(id) => format!("drive:{}", id.0),
            Self::Outcome(id) => format!("outcome:{}", id.0),
            Self::Property(id) => format!("property:{}", id.0),
            Self::Concept(id) => format!("concept:{}", id.0),
            Self::Episode(id) => format!("episode:{}", id.0),
        }
    }
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum SemanticPredicate {
    Affords,
    CanBeUsedBy,
    RequiresSkill,
    RequiresCapability,
    ReachableFrom,
    Blocks,
    Contains,
    Supports,
    SatisfiesDrive,
    ReducesDriveError,
    HelpsGoal,
    HindersGoal,
    Restores,
    Depletes,
    Causes,
    ContributesTo,
    Prevents,
    Predicts,
    UsuallyFollowedBy,
    ChangesProperty,
    IsA,
    InstanceOf,
    PartOf,
    SameEntityAs,
    SimilarTo,
    NamedBy,
    LocatedAt,
    Near,
    BelongsToEpisode,
    AssociatedWithPerson,
    RequestedBy,
    OwnedOrMaintainedBy,
    UsedFor,
    ServesGoal,
    DesignedFor,
    #[default]
    RelatedTo,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticRelationStatus {
    #[default]
    Hypothesized,
    Supported,
    Strong,
    Contradicted,
    ContextLimited,
    Deprecated,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticGroundingKind {
    ActionOutcome,
    Intervention,
    PredictionConfirmation,
    Cooccurrence,
    HumanInstruction,
    LlmClaim,
    SimulatorTeacher,
    Consolidation,
    Configuration,
    #[default]
    TemporalSequence,
}

impl SemanticGroundingKind {
    fn supports_causality(self) -> bool {
        matches!(
            self,
            Self::ActionOutcome
                | Self::Intervention
                | Self::PredictionConfirmation
                | Self::SimulatorTeacher
                | Self::Configuration
        )
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticContext {
    pub agent: Option<String>,
    pub goal: Option<SemanticGoalId>,
    pub place: Option<SemanticPlaceId>,
    pub episode: Option<SemanticEpisodeId>,
    #[serde(default)]
    pub conditions: BTreeMap<String, String>,
}

impl SemanticContext {
    fn stable_key(&self) -> String {
        let conditions = self
            .conditions
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "agent={};goal={};place={};episode={};conditions={conditions}",
            self.agent.as_deref().unwrap_or(""),
            self.goal.as_ref().map(|id| id.0.as_str()).unwrap_or(""),
            self.place.as_ref().map(|id| id.0.as_str()).unwrap_or(""),
            self.episode.as_ref().map(|id| id.0.as_str()).unwrap_or("")
        )
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SemanticRelation {
    pub id: SemanticRelationId,
    pub subject: SemanticNodeRef,
    pub predicate: SemanticPredicate,
    pub object: SemanticNodeRef,
    pub context: SemanticContext,
    pub confidence: f32,
    pub evidence_count: u32,
    #[serde(default)]
    pub supporting_evidence: Vec<EvidenceRef>,
    #[serde(default)]
    pub contradicting_evidence: Vec<EvidenceRef>,
    pub learned_at_ms: u64,
    pub last_confirmed_ms: u64,
    pub status: SemanticRelationStatus,
    #[serde(default)]
    pub grounding_sources: BTreeSet<SemanticGroundingKind>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SemanticEvidenceObservation {
    pub subject: SemanticNodeRef,
    pub predicate: SemanticPredicate,
    pub object: SemanticNodeRef,
    pub context: SemanticContext,
    pub confidence: f32,
    pub grounding: SemanticGroundingKind,
    pub evidence: EvidenceRef,
    pub contradicts: bool,
}

impl SemanticEvidenceObservation {
    pub fn supported(
        subject: SemanticNodeRef,
        predicate: SemanticPredicate,
        object: SemanticNodeRef,
        confidence: f32,
        grounding: SemanticGroundingKind,
        evidence: EvidenceRef,
    ) -> Self {
        Self {
            subject,
            predicate,
            object,
            confidence,
            grounding,
            evidence,
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SemanticRevision {
    pub relation_id: SemanticRelationId,
    pub previous_confidence: f32,
    pub current_confidence: f32,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SemanticExplanation {
    pub summary: String,
    #[serde(default)]
    pub relation_ids: Vec<SemanticRelationId>,
    #[serde(default)]
    pub evidence_refs: Vec<EvidenceRef>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SemanticGraphSnapshot {
    pub schema_version: u32,
    pub revision: u64,
    pub t_ms: u64,
    #[serde(default)]
    pub relations: BTreeMap<SemanticRelationId, SemanticRelation>,
    #[serde(default)]
    pub recent_revisions: Vec<SemanticRevision>,
}

impl SemanticGraphSnapshot {
    pub fn relations_from<'a>(
        &'a self,
        subject: &'a SemanticNodeRef,
    ) -> impl Iterator<Item = &'a SemanticRelation> {
        self.relations
            .values()
            .filter(move |relation| &relation.subject == subject)
    }

    pub fn relations_to<'a>(
        &'a self,
        object: &'a SemanticNodeRef,
    ) -> impl Iterator<Item = &'a SemanticRelation> {
        self.relations
            .values()
            .filter(move |relation| &relation.object == object)
    }

    pub fn supports(
        &self,
        subject: &SemanticNodeRef,
        predicate: SemanticPredicate,
        object: &SemanticNodeRef,
        minimum_confidence: f32,
    ) -> bool {
        self.relations.values().any(|relation| {
            &relation.subject == subject
                && relation.predicate == predicate
                && &relation.object == object
                && relation.confidence >= minimum_confidence
                && !matches!(
                    relation.status,
                    SemanticRelationStatus::Contradicted | SemanticRelationStatus::Deprecated
                )
        })
    }

    pub fn relation_ids_supporting(
        &self,
        subject: &SemanticNodeRef,
        predicate: SemanticPredicate,
    ) -> Vec<SemanticRelationId> {
        self.relations
            .values()
            .filter(|relation| {
                &relation.subject == subject
                    && relation.predicate == predicate
                    && relation.confidence >= 0.40
                    && !matches!(
                        relation.status,
                        SemanticRelationStatus::Contradicted | SemanticRelationStatus::Deprecated
                    )
            })
            .map(|relation| relation.id.clone())
            .collect()
    }

    pub fn charger_explanation(&self, entity_id: &EntityId) -> SemanticExplanation {
        let entity = SemanticNodeRef::Entity(entity_id.clone());
        let charger = SemanticNodeRef::Concept(SemanticConceptId("charger".to_string()));
        let energy = SemanticNodeRef::Drive(SemanticDriveId("energy".to_string()));
        let relations = self
            .relations
            .values()
            .filter(|relation| {
                (relation.subject == entity
                    && relation.predicate == SemanticPredicate::IsA
                    && relation.object == charger)
                    || (relation.subject == charger
                        && matches!(
                            relation.predicate,
                            SemanticPredicate::Restores | SemanticPredicate::SatisfiesDrive
                        )
                        && relation.object == energy)
            })
            .collect::<Vec<_>>();
        SemanticExplanation {
            summary: if relations.len() >= 2 {
                format!(
                    "{} is believed to be a charger; grounded charger meaning says it can restore and satisfy PETE's energy drive",
                    entity_id.0
                )
            } else {
                format!("grounded charger meaning for {} is incomplete", entity_id.0)
            },
            relation_ids: relations
                .iter()
                .map(|relation| relation.id.clone())
                .collect(),
            evidence_refs: relations
                .iter()
                .flat_map(|relation| relation.supporting_evidence.clone())
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct SemanticGraphBuilder {
    revision: u64,
    relations: BTreeMap<SemanticRelationId, SemanticRelation>,
}

impl SemanticGraphBuilder {
    pub(crate) fn update(
        &mut self,
        now: &Now,
        entities: &BTreeMap<EntityId, WorldEntity>,
        observations: &[SemanticEvidenceObservation],
    ) -> SemanticGraphSnapshot {
        let mut revisions = Vec::new();
        self.ensure_foundational_relations(now.t_ms, &mut revisions);
        for entity in entities.values() {
            self.integrate_entity(now, entity, &mut revisions);
        }
        self.integrate_situated_blocking(now, entities, &mut revisions);
        for observation in observations {
            self.integrate_observation(now.t_ms, observation.clone(), &mut revisions);
        }
        self.revision = self.revision.saturating_add(1);
        SemanticGraphSnapshot {
            schema_version: 1,
            revision: self.revision,
            t_ms: now.t_ms,
            relations: self.relations.clone(),
            recent_revisions: revisions,
        }
    }

    fn ensure_foundational_relations(
        &mut self,
        now_ms: u64,
        revisions: &mut Vec<SemanticRevision>,
    ) {
        let charger = SemanticNodeRef::Concept(SemanticConceptId("charger".to_string()));
        let energy = SemanticNodeRef::Drive(SemanticDriveId("energy".to_string()));
        let seek_charger = SemanticNodeRef::Goal(SemanticGoalId("seek_charger".to_string()));
        let approach =
            SemanticNodeRef::Behavior(SemanticBehaviorId("approach_charger".to_string()));
        let dock = SemanticNodeRef::Behavior(SemanticBehaviorId("dock".to_string()));
        let approach_skill = SemanticNodeRef::Skill(SemanticSkillId("approach_target".to_string()));
        let dock_skill = SemanticNodeRef::Skill(SemanticSkillId("align_with_dock".to_string()));
        let distance_decreases =
            SemanticNodeRef::Outcome(SemanticOutcomeId("target_distance_decreases".to_string()));
        let charging_started =
            SemanticNodeRef::Outcome(SemanticOutcomeId("charging_started".to_string()));
        let obstacle = SemanticNodeRef::Concept(SemanticConceptId("obstacle".to_string()));
        let back_away = SemanticNodeRef::Behavior(SemanticBehaviorId("back_away".to_string()));
        let clearance_increases =
            SemanticNodeRef::Outcome(SemanticOutcomeId("clearance_increases".to_string()));
        let escape = SemanticNodeRef::Goal(SemanticGoalId("escape_danger".to_string()));
        for (subject, predicate, object, confidence, context) in [
            (
                charger.clone(),
                SemanticPredicate::Restores,
                energy.clone(),
                0.90,
                organism_context(),
            ),
            (
                charger.clone(),
                SemanticPredicate::SatisfiesDrive,
                energy,
                0.90,
                organism_context(),
            ),
            (
                charger.clone(),
                SemanticPredicate::Affords,
                approach.clone(),
                0.85,
                conditional_context(&[("requires", "localized_reachable_target")]),
            ),
            (
                charger.clone(),
                SemanticPredicate::Affords,
                dock.clone(),
                0.85,
                conditional_context(&[("requires", "near_aligned_compatible")]),
            ),
            (
                charger,
                SemanticPredicate::HelpsGoal,
                seek_charger,
                0.90,
                organism_context(),
            ),
            (
                approach.clone(),
                SemanticPredicate::RequiresSkill,
                approach_skill,
                0.95,
                organism_context(),
            ),
            (
                approach,
                SemanticPredicate::Predicts,
                distance_decreases,
                0.65,
                conditional_context(&[("when", "target_visible_and_route_open")]),
            ),
            (
                dock.clone(),
                SemanticPredicate::RequiresSkill,
                dock_skill,
                0.95,
                organism_context(),
            ),
            (
                dock,
                SemanticPredicate::Predicts,
                charging_started,
                0.65,
                conditional_context(&[("when", "near_aligned_compatible")]),
            ),
            (
                obstacle,
                SemanticPredicate::Blocks,
                SemanticNodeRef::Concept(SemanticConceptId("path".to_string())),
                0.80,
                conditional_context(&[("when", "occupies_route")]),
            ),
            (
                back_away.clone(),
                SemanticPredicate::Predicts,
                clearance_increases,
                0.70,
                conditional_context(&[("when", "rear_clear")]),
            ),
            (
                back_away,
                SemanticPredicate::HelpsGoal,
                escape,
                0.85,
                organism_context(),
            ),
        ] {
            let evidence = EvidenceRef {
                id: format!(
                    "semantic:foundation:{}:{:?}:{}:{}",
                    subject.stable_key(),
                    predicate,
                    object.stable_key(),
                    context.stable_key()
                ),
                source: "semantic.foundation".to_string(),
                key: "configured_operational_meaning".to_string(),
                observed_at_ms: 0,
                transformation_lineage: vec!["pete_now::SemanticGraphBuilder".to_string()],
                implementation_version: Some("1".to_string()),
            };
            self.integrate_observation(
                now_ms,
                SemanticEvidenceObservation {
                    subject,
                    predicate,
                    object,
                    context,
                    confidence,
                    grounding: SemanticGroundingKind::Configuration,
                    evidence,
                    contradicts: false,
                },
                revisions,
            );
        }
    }

    fn integrate_entity(
        &mut self,
        now: &Now,
        entity: &WorldEntity,
        revisions: &mut Vec<SemanticRevision>,
    ) {
        let concept = match entity.kind {
            WorldEntityKind::Charger => Some("charger"),
            WorldEntityKind::Obstacle => Some("obstacle"),
            WorldEntityKind::Person => Some("person"),
            WorldEntityKind::Door => Some("door"),
            WorldEntityKind::SoundSource => Some("sound_source"),
            WorldEntityKind::Landmark => Some("landmark"),
            WorldEntityKind::Region => Some("region"),
            WorldEntityKind::Unknown => None,
        };
        let Some(concept) = concept else {
            return;
        };
        self.integrate_observation(
            now.t_ms,
            SemanticEvidenceObservation {
                subject: SemanticNodeRef::Entity(entity.id.clone()),
                predicate: SemanticPredicate::IsA,
                object: SemanticNodeRef::Concept(SemanticConceptId(concept.to_string())),
                context: organism_context(),
                confidence: entity.confidence,
                grounding: SemanticGroundingKind::Cooccurrence,
                evidence: entity
                    .provenance
                    .first()
                    .cloned()
                    .unwrap_or_else(|| semantic_evidence(now.t_ms, "world.entity", &entity.id.0)),
                contradicts: false,
            },
            revisions,
        );
        if entity.kind == WorldEntityKind::Charger && now.body.charging {
            self.integrate_observation(
                now.t_ms,
                SemanticEvidenceObservation {
                    subject: SemanticNodeRef::Entity(entity.id.clone()),
                    predicate: SemanticPredicate::Restores,
                    object: SemanticNodeRef::Drive(SemanticDriveId("energy".to_string())),
                    context: conditional_context(&[("body_state", "charging")]),
                    confidence: entity.confidence,
                    grounding: SemanticGroundingKind::ActionOutcome,
                    evidence: semantic_evidence(now.t_ms, "body.charging", "charging_confirmed"),
                    contradicts: false,
                },
                revisions,
            );
        }
    }

    fn integrate_situated_blocking(
        &mut self,
        now: &Now,
        entities: &BTreeMap<EntityId, WorldEntity>,
        revisions: &mut Vec<SemanticRevision>,
    ) {
        let Some(charger) = entities
            .values()
            .filter(|entity| entity.kind == WorldEntityKind::Charger)
            .max_by(|left, right| left.confidence.total_cmp(&right.confidence))
        else {
            return;
        };
        if charger.reachability.reachable || charger.reachability.confidence < 0.50 {
            return;
        }
        if let Some(obstacle) = entities
            .values()
            .filter(|entity| entity.kind == WorldEntityKind::Obstacle)
            .min_by(|left, right| {
                left.distance_m
                    .unwrap_or(f32::INFINITY)
                    .total_cmp(&right.distance_m.unwrap_or(f32::INFINITY))
            })
        {
            self.integrate_observation(
                now.t_ms,
                SemanticEvidenceObservation {
                    subject: SemanticNodeRef::Entity(obstacle.id.clone()),
                    predicate: SemanticPredicate::Blocks,
                    object: SemanticNodeRef::Entity(charger.id.clone()),
                    context: conditional_context(&[("route_state", "currently_blocked")]),
                    confidence: obstacle.confidence.min(charger.confidence),
                    grounding: SemanticGroundingKind::Cooccurrence,
                    evidence: obstacle.provenance.first().cloned().unwrap_or_else(|| {
                        semantic_evidence(now.t_ms, "world.geometry", "route_blocked")
                    }),
                    contradicts: false,
                },
                revisions,
            );
        }
    }

    fn integrate_observation(
        &mut self,
        now_ms: u64,
        mut observation: SemanticEvidenceObservation,
        revisions: &mut Vec<SemanticRevision>,
    ) {
        if observation.predicate == SemanticPredicate::Causes
            && !observation.grounding.supports_causality()
        {
            observation.predicate = SemanticPredicate::Predicts;
        }
        let id = relation_id(
            &observation.subject,
            observation.predicate,
            &observation.object,
            &observation.context,
        );
        let previous_confidence = self
            .relations
            .get(&id)
            .map(|relation| relation.confidence)
            .unwrap_or(0.0);
        let relation = self
            .relations
            .entry(id.clone())
            .or_insert_with(|| SemanticRelation {
                id: id.clone(),
                subject: observation.subject.clone(),
                predicate: observation.predicate,
                object: observation.object.clone(),
                context: observation.context.clone(),
                confidence: 0.0,
                learned_at_ms: now_ms,
                last_confirmed_ms: now_ms,
                status: SemanticRelationStatus::Hypothesized,
                ..SemanticRelation::default()
            });
        relation.grounding_sources.insert(observation.grounding);
        if observation.contradicts {
            if push_unique_evidence(&mut relation.contradicting_evidence, observation.evidence) {
                relation.confidence = (relation.confidence
                    * (1.0 - 0.65 * observation.confidence.clamp(0.0, 1.0)))
                .clamp(0.0, 1.0);
                relation.status = SemanticRelationStatus::Contradicted;
            }
        } else {
            if push_unique_evidence(&mut relation.supporting_evidence, observation.evidence) {
                let evidence_scale = if relation.supporting_evidence.len() <= 1 {
                    1.0
                } else {
                    0.35
                };
                relation.confidence = (1.0
                    - (1.0 - relation.confidence)
                        * (1.0 - observation.confidence.clamp(0.0, 1.0) * evidence_scale))
                    .clamp(0.0, 1.0);
                relation.last_confirmed_ms = now_ms;
                relation.status = if !relation.context.conditions.is_empty() {
                    SemanticRelationStatus::ContextLimited
                } else if relation.confidence >= 0.85 && relation.supporting_evidence.len() >= 2 {
                    SemanticRelationStatus::Strong
                } else if relation.confidence >= 0.50 {
                    SemanticRelationStatus::Supported
                } else {
                    SemanticRelationStatus::Hypothesized
                };
            }
        }
        relation.evidence_count = relation.supporting_evidence.len() as u32;
        if (relation.confidence - previous_confidence).abs() > f32::EPSILON {
            revisions.push(SemanticRevision {
                relation_id: id,
                previous_confidence,
                current_confidence: relation.confidence,
                reason: if observation.contradicts {
                    "contradicting grounded evidence".to_string()
                } else {
                    format!("support from {:?}", observation.grounding)
                },
            });
        }
    }
}

fn relation_id(
    subject: &SemanticNodeRef,
    predicate: SemanticPredicate,
    object: &SemanticNodeRef,
    context: &SemanticContext,
) -> SemanticRelationId {
    SemanticRelationId(format!(
        "semantic:{}:{:?}:{}:{}",
        subject.stable_key(),
        predicate,
        object.stable_key(),
        context.stable_key()
    ))
}

fn organism_context() -> SemanticContext {
    SemanticContext {
        agent: Some("pete".to_string()),
        ..SemanticContext::default()
    }
}

fn conditional_context(conditions: &[(&str, &str)]) -> SemanticContext {
    SemanticContext {
        agent: Some("pete".to_string()),
        conditions: conditions
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect(),
        ..SemanticContext::default()
    }
}

fn semantic_evidence(now_ms: u64, source: &str, key: &str) -> EvidenceRef {
    EvidenceRef {
        id: format!("semantic:{source}:{key}:{now_ms}"),
        source: source.to_string(),
        key: key.to_string(),
        observed_at_ms: now_ms,
        transformation_lineage: vec!["pete_now::SemanticGraphBuilder".to_string()],
        implementation_version: Some("1".to_string()),
    }
}

fn push_unique_evidence(target: &mut Vec<EvidenceRef>, evidence: EvidenceRef) -> bool {
    if !target.iter().any(|existing| existing.id == evidence.id) {
        target.push(evidence);
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
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
                relation.subject
                    == SemanticNodeRef::Concept(SemanticConceptId("charger".to_string()))
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
                SemanticNodeRef::Outcome(SemanticOutcomeId(
                    "target_distance_decreases".to_string(),
                )),
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
}
