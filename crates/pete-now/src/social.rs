use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{
    Belief, BeliefMeta, BeliefSourceKind, EntityId, EvidenceRef, Freshness, Now, WorldEntity,
    WorldEntityKind, WorldPose,
};

const PRESENCE_CURRENT_MS: u64 = 1_000;
const PRESENCE_STALE_MS: u64 = 15_000;
const RECENT_INTERACTION_LIMIT: usize = 8;

macro_rules! social_id {
    ($name:ident) => {
        #[derive(
            Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub String);
    };
}

social_id!(PersonId);
social_id!(RelationshipId);
social_id!(InteractionId);
social_id!(RequestRef);
social_id!(ConversationTurnRef);
social_id!(ContextRef);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityModality {
    Face,
    Voice,
    TextSelfIdentification,
    HumanLabel,
    CoLocation,
    Memory,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct IdentityHypothesis {
    pub identity_key: String,
    pub display_name: Option<String>,
    pub confidence: f32,
    #[serde(default)]
    pub modalities: Vec<IdentityModality>,
    #[serde(default)]
    pub evidence: Vec<EvidenceRef>,
    #[serde(default)]
    pub contradiction_refs: Vec<EvidenceRef>,
    pub human_confirmed: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PresenceBelief {
    pub present: bool,
    pub last_seen_at_ms: u64,
    pub confidence: f32,
    pub freshness: Freshness,
    pub meta: BeliefMeta,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SpatialBelief {
    pub pose: Option<WorldPose>,
    pub bearing_rad: Option<f32>,
    pub distance_m: Option<f32>,
    pub meta: BeliefMeta,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AttentionBelief {
    pub attending_to_pete: bool,
    pub pete_attending: bool,
    pub expected_response: bool,
    pub confidence: f32,
    pub meta: BeliefMeta,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CommunicationProfile {
    pub can_use_speech: bool,
    pub can_use_gesture: bool,
    pub preferred_channel: Option<String>,
    #[serde(default)]
    pub known_preferences: Vec<Belief<String>>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InteractionRef {
    pub interaction_id: InteractionId,
    pub occurred_at_ms: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PersonModel {
    pub person_id: PersonId,
    #[serde(default)]
    pub identity_hypotheses: Vec<IdentityHypothesis>,
    pub preferred_name: Option<Belief<String>>,
    pub presence: PresenceBelief,
    pub location: Option<SpatialBelief>,
    pub attention: AttentionBelief,
    pub familiarity: f32,
    pub current_identity_confidence: f32,
    #[serde(default)]
    pub relationship_refs: Vec<RelationshipId>,
    pub communication: CommunicationProfile,
    #[serde(default)]
    pub recent_interactions: Vec<InteractionRef>,
    pub meta: BeliefMeta,
}

impl PersonModel {
    pub fn best_identity(&self) -> Option<&IdentityHypothesis> {
        self.identity_hypotheses
            .iter()
            .max_by(|left, right| left.confidence.total_cmp(&right.confidence))
    }

    pub fn identity_is_uncertain(&self) -> bool {
        self.best_identity()
            .is_none_or(|identity| identity.confidence < 0.75 || identity.display_name.is_none())
            || self
                .identity_hypotheses
                .iter()
                .any(|identity| !identity.contradiction_refs.is_empty())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationshipKind {
    KnownPerson,
    Stranger,
    HouseholdMember,
    Caregiver,
    Maintainer,
    TrustedOperator,
    Friend,
    FrequentCompanion,
    TaskOwner,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RelationshipKindBelief {
    pub kind: RelationshipKind,
    pub confidence: f32,
    #[serde(default)]
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ScopedSocialAuthorityBelief {
    pub scope: String,
    pub confidence: f32,
    pub grants_reign_authority: bool,
    #[serde(default)]
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SocialCommitment {
    pub commitment_id: String,
    pub owner: PersonId,
    pub summary: String,
    pub created_at_ms: u64,
    pub due_at_ms: Option<u64>,
    pub fulfilled: bool,
    #[serde(default)]
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RelationshipModel {
    pub relationship_id: RelationshipId,
    pub person_id: PersonId,
    #[serde(default)]
    pub relationship_kinds: Vec<RelationshipKindBelief>,
    pub trust: f32,
    pub affiliation: f32,
    pub caregiving_or_authority: Option<ScopedSocialAuthorityBelief>,
    #[serde(default)]
    pub interaction_preferences: Vec<Belief<String>>,
    #[serde(default)]
    pub commitments: Vec<SocialCommitment>,
    #[serde(default)]
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionPhase {
    Orienting,
    Greeting,
    Engaged,
    AwaitingResponse,
    Closing,
    Ended,
    #[default]
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SocialAcknowledgmentKind {
    #[default]
    GreetingAttempted,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SocialAcknowledgment {
    pub acknowledgment_id: String,
    pub kind: SocialAcknowledgmentKind,
    pub person_id: PersonId,
    pub occurred_at_ms: u64,
    pub skill_id: String,
    pub skill_execution_id: u64,
    #[serde(default)]
    pub provenance: Vec<EvidenceRef>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InteractionState {
    pub interaction_id: InteractionId,
    #[serde(default)]
    pub participants: Vec<PersonId>,
    pub started_at_ms: u64,
    pub last_activity_ms: u64,
    pub ended_at_ms: Option<u64>,
    pub phase: InteractionPhase,
    pub attention_target: Option<PersonId>,
    #[serde(default)]
    pub pending_turns: Vec<ConversationTurnRef>,
    #[serde(default)]
    pub unresolved_requests: Vec<RequestRef>,
    #[serde(default)]
    pub shared_context: Vec<ContextRef>,
    #[serde(default)]
    pub acknowledgments: Vec<SocialAcknowledgment>,
    #[serde(default)]
    pub provenance: Vec<EvidenceRef>,
}

impl InteractionState {
    pub fn has_acknowledgment(&self, person_id: &PersonId, kind: SocialAcknowledgmentKind) -> bool {
        self.acknowledgments.iter().any(|acknowledgment| {
            acknowledgment.person_id == *person_id && acknowledgment.kind == kind
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SocialWorldSnapshot {
    pub schema_version: u32,
    pub t_ms: u64,
    #[serde(default)]
    pub people: BTreeMap<PersonId, PersonModel>,
    #[serde(default)]
    pub relationships: BTreeMap<RelationshipId, RelationshipModel>,
    pub active_interaction: Option<InteractionState>,
    #[serde(default)]
    pub recent_interactions: Vec<InteractionState>,
}

impl SocialWorldSnapshot {
    pub fn present_people(&self) -> impl Iterator<Item = &PersonModel> {
        self.people
            .values()
            .filter(|person| person.presence.present)
    }

    pub fn most_relevant_person(&self) -> Option<&PersonModel> {
        self.present_people().max_by(|left, right| {
            let left_score =
                left.presence.confidence + left.attention.confidence + left.familiarity;
            let right_score =
                right.presence.confidence + right.attention.confidence + right.familiarity;
            left_score.total_cmp(&right_score)
        })
    }

    pub fn last_interaction_with(&self, person_id: &PersonId) -> Option<&InteractionState> {
        self.recent_interactions
            .iter()
            .rev()
            .find(|interaction| interaction.participants.contains(person_id))
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct SocialWorldModelBuilder {
    people: BTreeMap<PersonId, PersonModel>,
    relationships: BTreeMap<RelationshipId, RelationshipModel>,
    observations: BTreeMap<PersonId, u32>,
    active_interaction: Option<InteractionState>,
    recent_interactions: Vec<InteractionState>,
    interaction_sequence: u64,
}

impl SocialWorldModelBuilder {
    pub(crate) fn update(
        &mut self,
        now: &Now,
        entities: &BTreeMap<EntityId, WorldEntity>,
    ) -> SocialWorldSnapshot {
        self.age_presence(now.t_ms);
        self.observe_remembered_people(now);
        let person_entities = entities
            .values()
            .filter(|entity| {
                entity.kind == WorldEntityKind::Person && entity.last_observed_at_ms == now.t_ms
            })
            .collect::<Vec<_>>();
        if person_entities.is_empty() {
            let face_entities = face_presence_entities(now);
            let only_person = (face_entities.len() == 1).then(|| face_entities[0].id.clone());
            for entity in &face_entities {
                self.observe_person(now, entity, only_person.as_ref());
            }
        } else {
            let only_person = (person_entities.len() == 1).then(|| person_entities[0].id.clone());
            for entity in person_entities {
                self.observe_person(now, entity, only_person.as_ref());
            }
        }
        self.observe_skill_acknowledgments(now);
        self.update_interaction(now);
        SocialWorldSnapshot {
            schema_version: 1,
            t_ms: now.t_ms,
            people: self.people.clone(),
            relationships: self.relationships.clone(),
            active_interaction: self.active_interaction.clone(),
            recent_interactions: self.recent_interactions.clone(),
        }
    }

    fn age_presence(&mut self, now_ms: u64) {
        for person in self.people.values_mut() {
            let age_ms = now_ms.saturating_sub(person.presence.last_seen_at_ms);
            person.presence.present = age_ms <= PRESENCE_CURRENT_MS;
            person.presence.freshness = if age_ms <= PRESENCE_CURRENT_MS {
                Freshness::Current
            } else if age_ms <= PRESENCE_STALE_MS {
                Freshness::Stale
            } else {
                Freshness::Invalidated
            };
            person.presence.meta.freshness = person.presence.freshness.clone();
            person.attention.pete_attending &= person.presence.present;
            person.attention.expected_response &= person.presence.present;
        }
    }

    fn observe_person(&mut self, now: &Now, entity: &WorldEntity, only_person: Option<&EntityId>) {
        let person_id = PersonId(entity.id.0.clone());
        let observation_count = self
            .observations
            .entry(person_id.clone())
            .and_modify(|count| *count = count.saturating_add(1))
            .or_insert(1);
        let normalized_label = normalize_identity(&entity.label);
        let is_generic = normalized_label.is_empty() || normalized_label == "person";
        let mut modalities = entity
            .provenance
            .iter()
            .map(|evidence| modality_from_source(&evidence.source))
            .collect::<BTreeSet<_>>();
        let vector_matches = |source_id: Option<&String>| {
            only_person.is_some_and(|id| id == &entity.id)
                || source_id.is_some_and(|source| {
                    let normalized_source = normalize_identity(source);
                    !normalized_label.is_empty() && normalized_source.contains(&normalized_label)
                })
        };
        if now
            .face
            .vectors
            .iter()
            .any(|vector| vector_matches(vector.source_id.as_ref()))
        {
            modalities.insert(IdentityModality::Face);
        }
        if now
            .voice
            .vectors
            .iter()
            .any(|vector| vector_matches(vector.source_id.as_ref()))
        {
            modalities.insert(IdentityModality::Voice);
        }
        let multimodal_bonus = 0.12 * modalities.len().saturating_sub(1) as f32;
        let repeated_bonus = 0.05 * observation_count.saturating_sub(1).min(5) as f32;
        let identity_confidence =
            (entity.confidence + multimodal_bonus + repeated_bonus).clamp(0.0, 0.99);
        let mut model = self
            .people
            .remove(&person_id)
            .unwrap_or_else(|| PersonModel {
                person_id: person_id.clone(),
                communication: CommunicationProfile {
                    can_use_speech: true,
                    can_use_gesture: true,
                    preferred_channel: Some("speech".to_string()),
                    ..CommunicationProfile::default()
                },
                ..PersonModel::default()
            });
        let identity_key = if is_generic {
            person_id.0.clone()
        } else {
            format!("name:{normalized_label}")
        };
        upsert_identity(
            &mut model.identity_hypotheses,
            IdentityHypothesis {
                identity_key,
                display_name: (!is_generic).then(|| entity.label.clone()),
                confidence: identity_confidence,
                modalities: modalities.iter().copied().collect(),
                evidence: entity.provenance.clone(),
                human_confirmed: entity.meta.source_kind == BeliefSourceKind::HumanClaim,
                ..IdentityHypothesis::default()
            },
        );
        for (modality, vector) in now
            .face
            .vectors
            .iter()
            .map(|vector| (IdentityModality::Face, vector))
            .chain(
                now.voice
                    .vectors
                    .iter()
                    .map(|vector| (IdentityModality::Voice, vector)),
            )
        {
            let Some(source_id) = vector.source_id.as_ref() else {
                continue;
            };
            if !vector_matches(Some(source_id)) {
                continue;
            }
            let identity_key = biometric_identity_key(source_id);
            if identity_key.is_empty() {
                continue;
            }
            let evidence = EvidenceRef {
                id: format!(
                    "social:{:?}:{}:{}",
                    modality,
                    vector.point_id,
                    vector.occurred_at_ms.unwrap_or(now.t_ms)
                )
                .to_ascii_lowercase(),
                source: format!("{:?}", modality).to_ascii_lowercase(),
                key: source_id.clone(),
                observed_at_ms: now.t_ms,
                transformation_lineage: vec!["pete_now::SocialWorldModelBuilder".to_string()],
                implementation_version: vector.model.clone(),
            };
            upsert_identity(
                &mut model.identity_hypotheses,
                IdentityHypothesis {
                    identity_key: format!("biometric:{identity_key}"),
                    display_name: None,
                    confidence: 0.65,
                    modalities: vec![modality],
                    evidence: vec![evidence],
                    ..IdentityHypothesis::default()
                },
            );
        }

        if let Some(claimed_name) = self_identified_name(now.ear.transcript.as_deref()) {
            let claim_evidence = EvidenceRef {
                id: format!("social:self_identification:{}:{}", claimed_name, now.t_ms),
                source: "speech.self_identification".to_string(),
                key: claimed_name.clone(),
                observed_at_ms: now.t_ms,
                transformation_lineage: vec!["pete_now::SocialWorldModelBuilder".to_string()],
                implementation_version: Some("1".to_string()),
            };
            let contradiction_refs = if !is_generic
                && normalize_identity(&claimed_name) != normalized_label
                && modalities.iter().any(|modality| {
                    matches!(modality, IdentityModality::Face | IdentityModality::Voice)
                }) {
                entity.provenance.clone()
            } else {
                Vec::new()
            };
            upsert_identity(
                &mut model.identity_hypotheses,
                IdentityHypothesis {
                    identity_key: format!("name:{}", normalize_identity(&claimed_name)),
                    display_name: Some(claimed_name.clone()),
                    confidence: 0.55,
                    modalities: vec![IdentityModality::TextSelfIdentification],
                    evidence: vec![claim_evidence.clone()],
                    contradiction_refs: contradiction_refs.clone(),
                    human_confirmed: false,
                },
            );
            if !contradiction_refs.is_empty() {
                for hypothesis in &mut model.identity_hypotheses {
                    if hypothesis.identity_key == format!("name:{normalized_label}") {
                        hypothesis.contradiction_refs.push(claim_evidence.clone());
                    }
                }
            }
        }

        let best_named = model
            .identity_hypotheses
            .iter()
            .filter_map(|identity| identity.display_name.as_ref().map(|name| (name, identity)))
            .max_by(|(_, left), (_, right)| left.confidence.total_cmp(&right.confidence));
        model.preferred_name = best_named.map(|(name, identity)| Belief {
            value: name.clone(),
            meta: BeliefMeta {
                confidence: identity.confidence,
                observed_at_ms: now.t_ms,
                valid_at_ms: now.t_ms,
                freshness: Freshness::Current,
                provenance: identity.evidence.clone(),
                contradiction_refs: identity.contradiction_refs.clone(),
                source_kind: if identity.human_confirmed {
                    BeliefSourceKind::HumanClaim
                } else {
                    BeliefSourceKind::DerivedPerception
                },
                ..BeliefMeta::default()
            },
        });
        model.presence = PresenceBelief {
            present: true,
            last_seen_at_ms: entity.last_observed_at_ms,
            confidence: entity.confidence,
            freshness: Freshness::Current,
            meta: entity.meta.clone(),
        };
        model.location = Some(SpatialBelief {
            pose: entity.pose,
            bearing_rad: entity.bearing_rad,
            distance_m: entity.distance_m,
            meta: entity.meta.clone(),
        });
        model.attention = AttentionBelief {
            attending_to_pete: now.ear.transcript.is_some(),
            pete_attending: true,
            expected_response: now
                .ear
                .transcript
                .as_deref()
                .is_some_and(is_addressed_to_pete),
            confidence: if now.ear.transcript.is_some() {
                0.7
            } else {
                0.4
            },
            meta: entity.meta.clone(),
        };
        model.familiarity = (1.0 - (-(*observation_count as f32) / 3.0).exp()).clamp(0.0, 1.0);
        model.current_identity_confidence = identity_confidence;
        mark_biometric_contradictions(&mut model.identity_hypotheses);
        model.meta = entity.meta.clone();

        let relationship_id = RelationshipId(format!("relationship:{}", person_id.0));
        if !model.relationship_refs.contains(&relationship_id) {
            model.relationship_refs.push(relationship_id.clone());
        }
        self.relationships.insert(
            relationship_id.clone(),
            RelationshipModel {
                relationship_id,
                person_id: person_id.clone(),
                relationship_kinds: vec![RelationshipKindBelief {
                    kind: if model.familiarity >= 0.55 {
                        RelationshipKind::KnownPerson
                    } else {
                        RelationshipKind::Stranger
                    },
                    confidence: model.familiarity.max(0.5),
                    evidence: entity.provenance.clone(),
                }],
                trust: (model.familiarity * 0.5).clamp(0.0, 0.5),
                affiliation: model.familiarity,
                // Social familiarity never manufactures motor authority.
                caregiving_or_authority: None,
                evidence: entity.provenance.clone(),
                ..RelationshipModel::default()
            },
        );
        self.people.insert(person_id, model);
    }

    fn observe_remembered_people(&mut self, now: &Now) {
        for remembered in &now.memory.remembered_entities {
            if !remembered.has_label("Person") && !remembered.has_label("person") {
                continue;
            }
            let person_id = PersonId(if remembered.id.starts_with("person:") {
                remembered.id.clone()
            } else {
                format!("person:memory:{}", remembered.id)
            });
            if self.people.contains_key(&person_id) {
                continue;
            }
            let evidence = EvidenceRef {
                id: format!("social:memory:{}:{}", remembered.id, now.t_ms),
                source: "memory.recall".to_string(),
                key: remembered.id.clone(),
                observed_at_ms: now.t_ms,
                transformation_lineage: vec!["pete_now::SocialWorldModelBuilder".to_string()],
                implementation_version: Some("1".to_string()),
            };
            let meta = BeliefMeta {
                confidence: remembered.score.clamp(0.0, 1.0),
                observed_at_ms: now.t_ms,
                valid_at_ms: now.t_ms,
                freshness: Freshness::Stale,
                provenance: vec![evidence.clone()],
                source_kind: BeliefSourceKind::MemoryRecall,
                ..BeliefMeta::default()
            };
            self.people.insert(
                person_id.clone(),
                PersonModel {
                    person_id,
                    identity_hypotheses: vec![IdentityHypothesis {
                        identity_key: format!("memory:{}", remembered.id),
                        display_name: (!remembered.summary.trim().is_empty())
                            .then(|| remembered.summary.clone()),
                        confidence: remembered.score.clamp(0.0, 1.0),
                        modalities: vec![IdentityModality::Memory],
                        evidence: vec![evidence],
                        ..IdentityHypothesis::default()
                    }],
                    presence: PresenceBelief {
                        present: false,
                        confidence: 0.0,
                        freshness: Freshness::Stale,
                        meta: meta.clone(),
                        ..PresenceBelief::default()
                    },
                    familiarity: remembered.score.clamp(0.0, 1.0),
                    current_identity_confidence: 0.0,
                    meta,
                    ..PersonModel::default()
                },
            );
        }
    }

    fn update_interaction(&mut self, now: &Now) {
        let mut participants = self
            .people
            .values()
            .filter(|person| person.presence.present)
            .map(|person| person.person_id.clone())
            .collect::<Vec<_>>();
        participants.sort();
        let active_participants = self
            .active_interaction
            .as_ref()
            .map(|interaction| interaction.participants.clone())
            .unwrap_or_default();
        if participants.is_empty()
            || (!active_participants.is_empty() && active_participants != participants)
        {
            self.close_interaction(now.t_ms);
        }
        if participants.is_empty() {
            return;
        }
        if self.active_interaction.is_none() {
            self.interaction_sequence = self.interaction_sequence.saturating_add(1);
            self.active_interaction = Some(InteractionState {
                interaction_id: InteractionId(format!(
                    "interaction:{}:{}",
                    now.t_ms, self.interaction_sequence
                )),
                participants: participants.clone(),
                started_at_ms: now.t_ms,
                last_activity_ms: now.t_ms,
                phase: InteractionPhase::Greeting,
                attention_target: participants.first().cloned(),
                ..InteractionState::default()
            });
        }
        let Some(interaction) = self.active_interaction.as_mut() else {
            return;
        };
        interaction.participants = participants.clone();
        if let Some(transcript) = now.ear.transcript.as_ref() {
            interaction.last_activity_ms = now.t_ms;
            interaction.phase = if is_addressed_to_pete(transcript) {
                InteractionPhase::AwaitingResponse
            } else {
                InteractionPhase::Engaged
            };
            interaction
                .pending_turns
                .push(ConversationTurnRef(format!("turn:{}", now.t_ms)));
            if participants.len() == 1 && looks_like_request(transcript) {
                interaction.unresolved_requests.push(RequestRef(format!(
                    "request:{}:{}",
                    participants[0].0, now.t_ms
                )));
            }
        }
        let interaction_ref = InteractionRef {
            interaction_id: interaction.interaction_id.clone(),
            occurred_at_ms: interaction.started_at_ms,
        };
        for participant in participants {
            if let Some(person) = self.people.get_mut(&participant) {
                if !person
                    .recent_interactions
                    .iter()
                    .any(|existing| existing.interaction_id == interaction_ref.interaction_id)
                {
                    person.recent_interactions.push(interaction_ref.clone());
                }
            }
        }
    }

    fn observe_skill_acknowledgments(&mut self, now: &Now) {
        let Some(record) = now.extensions.get("motherbrain.skill_execution") else {
            return;
        };
        if record
            .pointer("/diagnostics/terminal_outcome")
            .and_then(serde_json::Value::as_str)
            != Some("completed")
        {
            return;
        }
        let skill_id = record
            .pointer("/skill/skill_id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let source_hash = record
            .pointer("/skill/source_hash")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let execution_id = record
            .get("execution_id")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_default();
        let Some(observations) = record
            .get("observations")
            .and_then(serde_json::Value::as_array)
        else {
            return;
        };
        let Some(interaction) = self.active_interaction.as_mut() else {
            return;
        };
        for observation in observations {
            if observation.get("kind").and_then(serde_json::Value::as_str)
                != Some("social_acknowledgment")
                || observation
                    .get("contract")
                    .and_then(serde_json::Value::as_str)
                    != Some("host_validated_social_acknowledgment_v1")
            {
                continue;
            }
            let Some(value) = observation.get("value") else {
                continue;
            };
            let Some(interaction_id) = value
                .get("interaction_id")
                .and_then(serde_json::Value::as_str)
            else {
                continue;
            };
            let Some(person_id) = value.get("person_id").and_then(serde_json::Value::as_str) else {
                continue;
            };
            if interaction.interaction_id.0 != interaction_id {
                continue;
            }
            let Some(participant) = interaction
                .participants
                .iter()
                .find(|participant| participant.0.eq_ignore_ascii_case(person_id))
                .cloned()
            else {
                continue;
            };
            let acknowledgment_id = value
                .get("acknowledgment_id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| {
                    format!(
                        "greet:{}:{}:{}",
                        interaction.interaction_id.0, participant.0, execution_id
                    )
                });
            if interaction
                .acknowledgments
                .iter()
                .any(|existing| existing.acknowledgment_id == acknowledgment_id)
            {
                continue;
            }
            let occurred_at_ms = value
                .get("occurred_at_ms")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(now.t_ms);
            interaction.acknowledgments.push(SocialAcknowledgment {
                acknowledgment_id: acknowledgment_id.clone(),
                kind: SocialAcknowledgmentKind::GreetingAttempted,
                person_id: participant,
                occurred_at_ms,
                skill_id: skill_id.to_string(),
                skill_execution_id: execution_id,
                provenance: vec![EvidenceRef {
                    id: format!("social:acknowledgment:{acknowledgment_id}"),
                    source: "lua.skill.acknowledge".to_string(),
                    key: skill_id.to_string(),
                    observed_at_ms: now.t_ms,
                    transformation_lineage: vec!["pete_now::SocialWorldModelBuilder".to_string()],
                    implementation_version: (!source_hash.is_empty())
                        .then(|| source_hash.to_string()),
                }],
            });
            interaction.last_activity_ms = now.t_ms;
            interaction.phase = InteractionPhase::Engaged;
        }
    }

    fn close_interaction(&mut self, now_ms: u64) {
        let Some(mut interaction) = self.active_interaction.take() else {
            return;
        };
        interaction.phase = InteractionPhase::Ended;
        interaction.ended_at_ms = Some(now_ms);
        self.recent_interactions.push(interaction);
        if self.recent_interactions.len() > RECENT_INTERACTION_LIMIT {
            self.recent_interactions
                .drain(..self.recent_interactions.len() - RECENT_INTERACTION_LIMIT);
        }
    }
}

fn upsert_identity(identities: &mut Vec<IdentityHypothesis>, incoming: IdentityHypothesis) {
    if let Some(existing) = identities
        .iter_mut()
        .find(|identity| identity.identity_key == incoming.identity_key)
    {
        existing.confidence = existing.confidence.max(incoming.confidence);
        existing.human_confirmed |= incoming.human_confirmed;
        existing.evidence.extend(incoming.evidence);
        existing
            .evidence
            .sort_by(|left, right| left.id.cmp(&right.id));
        existing
            .evidence
            .dedup_by(|left, right| left.id == right.id);
        existing.modalities.extend(incoming.modalities);
        existing.modalities.sort();
        existing.modalities.dedup();
        existing
            .contradiction_refs
            .extend(incoming.contradiction_refs);
        existing
            .contradiction_refs
            .sort_by(|left, right| left.id.cmp(&right.id));
        existing
            .contradiction_refs
            .dedup_by(|left, right| left.id == right.id);
    } else {
        identities.push(incoming);
    }
}

fn mark_biometric_contradictions(identities: &mut [IdentityHypothesis]) {
    let biometric = identities
        .iter()
        .enumerate()
        .filter(|(_, identity)| identity.identity_key.starts_with("biometric:"))
        .map(|(index, identity)| {
            (
                index,
                identity.identity_key.clone(),
                identity.evidence.clone(),
            )
        })
        .collect::<Vec<_>>();
    let distinct = biometric
        .iter()
        .map(|(_, key, _)| key)
        .collect::<BTreeSet<_>>();
    if distinct.len() <= 1 {
        return;
    }
    for (index, key, _) in &biometric {
        let contradictions = biometric
            .iter()
            .filter(|(_, other_key, _)| other_key != key)
            .flat_map(|(_, _, evidence)| evidence.clone())
            .collect::<Vec<_>>();
        identities[*index].contradiction_refs.extend(contradictions);
        identities[*index]
            .contradiction_refs
            .sort_by(|left, right| left.id.cmp(&right.id));
        identities[*index]
            .contradiction_refs
            .dedup_by(|left, right| left.id == right.id);
    }
}

fn face_presence_entities(now: &Now) -> Vec<WorldEntity> {
    let remembered_people = now
        .memory
        .remembered_entities
        .iter()
        .filter(|entity| entity.has_label("Person") || entity.has_label("person"))
        .collect::<Vec<_>>();
    let recognized = (now.face.vectors.len() == 1
        && now.memory.face_familiarity >= 0.70
        && remembered_people.len() == 1)
        .then(|| remembered_people[0]);

    now.face
        .vectors
        .iter()
        .enumerate()
        .map(|(index, face)| {
            let (id, label, confidence) = if let Some(person) = recognized {
                (
                    PersonId(if person.id.starts_with("person:") {
                        person.id.clone()
                    } else {
                        format!("person:memory:{}", person.id)
                    }),
                    person.summary.clone(),
                    now.memory
                        .face_familiarity
                        .min(person.score.max(0.70))
                        .clamp(0.0, 0.99),
                )
            } else {
                (
                    PersonId(format!("person:face:{index}")),
                    "person".to_string(),
                    0.65,
                )
            };
            let evidence = EvidenceRef {
                id: format!("social:face:{}:{}", face.point_id, now.t_ms),
                source: "face.embedding".to_string(),
                key: face.point_id.clone(),
                observed_at_ms: now.t_ms,
                transformation_lineage: vec![
                    "pete_now::SocialWorldModelBuilder::face_presence_entities".to_string(),
                ],
                implementation_version: face.model.clone(),
            };
            let meta = BeliefMeta {
                confidence,
                observed_at_ms: now.t_ms,
                valid_at_ms: now.t_ms,
                freshness: Freshness::Current,
                provenance: vec![evidence.clone()],
                source_kind: BeliefSourceKind::DerivedPerception,
                ..BeliefMeta::default()
            };
            WorldEntity {
                id: EntityId(id.0),
                kind: WorldEntityKind::Person,
                label,
                last_observed_at_ms: now.t_ms,
                confidence,
                meta: meta.clone(),
                // A detected face is already inside the camera field of view.
                // Zero is a conservative orientation target when no separate
                // object detector supplied a more precise bearing.
                bearing_rad: Some(0.0),
                bearing_meta: Some(meta.clone()),
                reachability_meta: Some(meta),
                attributes: BTreeMap::from([("observed_confidence".to_string(), confidence)]),
                provenance: vec![evidence],
                ..WorldEntity::default()
            }
        })
        .collect()
}

fn modality_from_source(source: &str) -> IdentityModality {
    if source.contains("human") {
        IdentityModality::HumanLabel
    } else if source.contains("memory") {
        IdentityModality::Memory
    } else {
        IdentityModality::CoLocation
    }
}

fn normalize_identity(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .collect()
}

fn biometric_identity_key(source_id: &str) -> String {
    normalize_identity(source_id.rsplit(':').next().unwrap_or(source_id))
}

fn self_identified_name(transcript: Option<&str>) -> Option<String> {
    let transcript = transcript?.trim();
    let lower = transcript.to_ascii_lowercase();
    for marker in ["my name is ", "i am ", "i'm "] {
        let Some(index) = lower.find(marker) else {
            continue;
        };
        let start = index + marker.len();
        let name = transcript[start..]
            .split(|character: char| {
                !character.is_alphanumeric() && character != '-' && character != '\''
            })
            .next()
            .unwrap_or_default()
            .trim();
        if !name.is_empty() {
            return Some(name.to_string());
        }
    }
    None
}

fn is_addressed_to_pete(transcript: &str) -> bool {
    let lower = transcript.to_ascii_lowercase();
    lower.contains("pete") || lower.ends_with('?')
}

fn looks_like_request(transcript: &str) -> bool {
    let lower = transcript.to_ascii_lowercase();
    lower.contains("please")
        || lower.starts_with("pete ")
        || lower.contains("could you")
        || lower.contains("would you")
}

#[cfg(test)]
#[path = "social_tests.rs"]
mod tests;
