use std::collections::{BTreeMap, BTreeSet};

use pete_actions::ReignInput;
use pete_cognition::{ProviderHealthState, ProviderRegistrySnapshot};
use pete_core::{FrameId, Pose2};
use serde::{Deserialize, Serialize};

use crate::epistemic::{EpistemicAttempt, EpistemicModelBuilder, EpistemicSnapshot};
use crate::social::{SocialWorldModelBuilder, SocialWorldSnapshot};
use crate::temporal::{
    ClockDomain, PendingTemporalExpectation, TemporalBelief, TemporalContext, TemporalIntegrator,
    TemporalRelation, TemporalUpdateInput, TimeInterval,
};
use crate::{Now, ObjectClass, ObjectObservationSource};

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EntityId(pub String);

macro_rules! string_id {
    ($name:ident) => {
        #[derive(
            Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub String);
    };
}

string_id!(OrganismId);
string_id!(BodyId);
string_id!(BrainstemDeviceId);
string_id!(BootId);
string_id!(HostId);
string_id!(ProcessId);
string_id!(SessionId);
string_id!(CapabilityId);

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeliefSourceKind {
    DirectObservation,
    DerivedPerception,
    MemoryRecall,
    LearnedPrediction,
    Map,
    ActionOutcome,
    HumanClaim,
    LlmClaim,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Freshness {
    Current,
    Aging,
    Stale,
    Invalidated,
    #[default]
    Missing,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRef {
    pub id: String,
    pub source: String,
    pub key: String,
    pub observed_at_ms: u64,
    #[serde(default)]
    pub transformation_lineage: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implementation_version: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BeliefMeta {
    pub confidence: f32,
    pub observed_at_ms: u64,
    pub valid_at_ms: u64,
    pub freshness: Freshness,
    #[serde(default)]
    pub provenance: Vec<EvidenceRef>,
    #[serde(default)]
    pub contradiction_refs: Vec<EvidenceRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coordinate_frame: Option<FrameId>,
    pub source_kind: BeliefSourceKind,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Belief<T> {
    pub value: T,
    pub meta: BeliefMeta,
}

impl<T: Default> Default for Belief<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
            meta: BeliefMeta::default(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorldEntityKind {
    Charger,
    Person,
    Obstacle,
    SoundSource,
    Landmark,
    Door,
    Region,
    #[default]
    Unknown,
}

impl From<&ObjectClass> for WorldEntityKind {
    fn from(value: &ObjectClass) -> Self {
        match value {
            ObjectClass::Charger => Self::Charger,
            ObjectClass::Person => Self::Person,
            ObjectClass::Obstacle => Self::Obstacle,
            ObjectClass::SoundSource => Self::SoundSource,
            ObjectClass::Landmark => Self::Landmark,
            ObjectClass::Unknown => Self::Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorldPose {
    pub x_m: f32,
    pub y_m: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReachabilityEstimate {
    pub reachable: bool,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorldEntity {
    pub id: EntityId,
    pub kind: WorldEntityKind,
    pub label: String,
    pub last_observed_at_ms: u64,
    pub confidence: f32,
    pub meta: BeliefMeta,
    pub pose: Option<WorldPose>,
    pub bearing_rad: Option<f32>,
    pub bearing_meta: Option<BeliefMeta>,
    pub distance_m: Option<f32>,
    pub distance_meta: Option<BeliefMeta>,
    pub reachability: ReachabilityEstimate,
    pub reachability_meta: Option<BeliefMeta>,
    #[serde(default)]
    pub attributes: BTreeMap<String, f32>,
    #[serde(default)]
    pub provenance: Vec<EvidenceRef>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GoalStatusBelief {
    pub meta: BeliefMeta,
    pub active: bool,
    pub elapsed_time_ms: u64,
    #[serde(default)]
    pub attempts: u32,
    pub failed_attempts: u32,
    pub recent_progress: f32,
    #[serde(default)]
    pub progress_trend: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_progress_at_ms: Option<u64>,
    pub confidence_trend: f32,
    pub frustration: f32,
    pub last_exit_reason: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StuckTrapKind {
    Wall,
    Corner,
    Column,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityKind {
    Sensor,
    Actuator,
    Goal,
    Behavior,
    Skill,
    CognitiveService,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityAvailability {
    Available,
    Degraded,
    Unavailable,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CapabilityBelief {
    pub id: CapabilityId,
    pub kind: CapabilityKind,
    pub availability: CapabilityAvailability,
    pub confidence: f32,
    pub unavailable_reason: Option<String>,
    pub authorized: bool,
    pub authority_reason: Option<String>,
    #[serde(default)]
    pub dependencies: Vec<CapabilityId>,
    pub performance_summary: Option<String>,
    pub meta: BeliefMeta,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CapabilitySelfModel {
    #[serde(default)]
    pub capabilities: BTreeMap<CapabilityId, CapabilityBelief>,
}

impl CapabilitySelfModel {
    pub fn is_available(&self, id: &str) -> bool {
        self.capabilities
            .get(&CapabilityId(id.to_string()))
            .is_some_and(|capability| {
                matches!(
                    capability.availability,
                    CapabilityAvailability::Available | CapabilityAvailability::Degraded
                )
            })
    }

    pub fn unavailable_reason(&self, id: &str) -> Option<&str> {
        self.capabilities
            .get(&CapabilityId(id.to_string()))
            .and_then(|capability| capability.unavailable_reason.as_deref())
    }

    pub fn is_authorized(&self, id: &str) -> bool {
        self.capabilities
            .get(&CapabilityId(id.to_string()))
            .is_some_and(|capability| capability.authorized)
    }

    pub fn execution_block_reason(&self, id: &str) -> Option<&str> {
        self.capabilities
            .get(&CapabilityId(id.to_string()))
            .and_then(|capability| {
                capability
                    .unavailable_reason
                    .as_deref()
                    .or(capability.authority_reason.as_deref())
            })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BodyEnvelope {
    pub radius_m: f32,
    pub height_m: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SelfBodyBelief {
    pub body_id: Belief<BodyId>,
    pub implementation: Belief<String>,
    pub implementation_version: Belief<String>,
    pub brainstem_device_id: Option<Belief<BrainstemDeviceId>>,
    pub brainstem_boot_id: Option<Belief<BootId>>,
    pub pose: Belief<Pose2>,
    pub envelope: Belief<BodyEnvelope>,
    pub energy: Belief<f32>,
    pub charging: Belief<bool>,
    pub health: Belief<f32>,
    #[serde(default)]
    pub faults: Vec<Belief<String>>,
    pub being_moved: Option<Belief<bool>>,
    pub tilted: Option<Belief<bool>>,
    pub blocked: Option<Belief<bool>>,
    pub carried: Option<Belief<bool>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlProvenance {
    Autonomous,
    HumanDirect,
    HumanAssist,
    HumanSuggestion,
    AutonomicReflex,
    SafetyVeto,
    #[default]
    None,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AgencyState {
    pub controller: ControlProvenance,
    pub reign_mode: Option<String>,
    pub reign_source: Option<String>,
    pub session_id: Option<Belief<SessionId>>,
    pub lease_id: Option<Belief<String>>,
    pub possessed: Belief<bool>,
    pub armed: Belief<bool>,
    pub stopped: bool,
    pub moving: bool,
    pub pending_direct_override: bool,
    #[serde(default)]
    pub authority_conflicts: Vec<EvidenceRef>,
    pub meta: BeliefMeta,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DriveSelfSummary {
    pub desired: f32,
    pub actual: f32,
    pub predicted: f32,
    pub error: f32,
    pub satisfaction: f32,
    pub activation: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MotivationSummary {
    #[serde(default)]
    pub drives: BTreeMap<String, DriveSelfSummary>,
    pub selected_goal: Option<String>,
    pub commitment_age_ms: u64,
    pub expected_progress: Option<f32>,
    pub recent_progress: Option<f32>,
    pub uncertainty: f32,
    pub strategy_failure_pressure: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ActiveControlSummary {
    pub goal_id: Option<String>,
    pub behavior_id: Option<String>,
    pub skill_id: Option<String>,
    pub action_kind: Option<String>,
    pub provenance: ControlProvenance,
    pub safety_preempted: bool,
    #[serde(default)]
    pub veto_reasons: Vec<String>,
    pub unable_to_act_reason: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ContinuitySummary {
    pub episode_id: Option<String>,
    pub session_id: Option<SessionId>,
    #[serde(default)]
    pub recent_experience_refs: Vec<String>,
    #[serde(default)]
    pub important_relationship_refs: Vec<EntityId>,
    #[serde(default)]
    pub important_place_refs: Vec<String>,
    #[serde(default)]
    pub recent_self_action_refs: Vec<String>,
    #[serde(default)]
    pub recent_outcome_refs: Vec<String>,
    #[serde(default)]
    pub capability_change_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CognitiveServiceBelief {
    pub provider_id: Option<String>,
    pub role: Option<String>,
    pub capability: Option<String>,
    pub capability_version: Option<String>,
    pub available: bool,
    pub confidence: f32,
    pub unavailable_reason: Option<String>,
    pub host_id: Option<HostId>,
    pub process_id: Option<ProcessId>,
    pub implementation: Option<String>,
    pub implementation_version: Option<String>,
    pub model_version: Option<String>,
    pub locality: Option<String>,
    pub resource_class: Option<String>,
    pub meta: BeliefMeta,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CognitiveServiceSummary {
    #[serde(default)]
    pub services: BTreeMap<String, CognitiveServiceBelief>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SelfModelSnapshot {
    pub organism_id: Belief<OrganismId>,
    pub body: SelfBodyBelief,
    pub capabilities: CapabilitySelfModel,
    pub agency: AgencyState,
    pub motivation: MotivationSummary,
    pub active_control: ActiveControlSummary,
    pub continuity: ContinuitySummary,
    pub service_state: CognitiveServiceSummary,
    pub meta: BeliefMeta,
    // Stable compatibility projections for existing goal and safety consumers.
    pub battery_level: f32,
    pub battery_meta: BeliefMeta,
    pub charging: bool,
    pub charging_meta: BeliefMeta,
    pub stuck: bool,
    pub stuck_meta: BeliefMeta,
    pub stuck_trap_kind: Option<Belief<StuckTrapKind>>,
    pub pose: Pose2,
    pub pose_meta: BeliefMeta,
    pub contact: bool,
    pub bump_left: bool,
    pub moving: bool,
    pub range_nearest_m: Option<f32>,
    pub active_goal: Option<String>,
    #[serde(default)]
    pub goal_status: BTreeMap<String, GoalStatusBelief>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LocalGeometrySnapshot {
    pub nearest_m: Option<Belief<f32>>,
    pub left_clearance_m: Option<Belief<f32>>,
    pub center_clearance_m: Option<Belief<f32>>,
    pub right_clearance_m: Option<Belief<f32>>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ContextBeliefs {
    pub novelty: Option<Belief<f32>>,
    pub surprise: Option<Belief<f32>>,
    pub prediction_uncertainty: Option<Belief<f32>>,
    pub map_confidence: Option<Belief<f32>>,
    pub safe_bearing_rad: Option<Belief<f32>>,
    pub frontier_bearing_rad: Option<Belief<f32>>,
    pub llm_confidence: Option<Belief<f32>>,
    pub expected_battery_delta: Option<Belief<f32>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuthorityBelief {
    pub input: ReignInput,
    pub meta: BeliefMeta,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HazardBeliefs {
    pub immediate_risk: Option<Belief<f32>>,
    pub remembered_risk: Option<Belief<f32>>,
    pub predicted_risk: Option<Belief<f32>>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BeliefUpdateTrace {
    #[serde(default)]
    pub input_evidence_ids: Vec<String>,
    #[serde(default)]
    pub added: Vec<String>,
    #[serde(default)]
    pub updated: Vec<String>,
    #[serde(default)]
    pub removed: Vec<String>,
    #[serde(default)]
    pub freshness_changes: Vec<String>,
    #[serde(default)]
    pub confidence_changes: Vec<String>,
    #[serde(default)]
    pub contradiction_resolutions: Vec<String>,
    pub builder_implementation: String,
    pub builder_version: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorldModelSnapshot {
    pub schema_version: u32,
    pub revision: u64,
    pub t_ms: u64,
    #[serde(default)]
    pub entities: BTreeMap<EntityId, WorldEntity>,
    pub self_model: SelfModelSnapshot,
    pub local_geometry: LocalGeometrySnapshot,
    pub hazards: HazardBeliefs,
    pub context: ContextBeliefs,
    pub temporal: TemporalContext,
    pub social: SocialWorldSnapshot,
    pub epistemic: EpistemicSnapshot,
    #[serde(default)]
    pub semantic: crate::SemanticGraphSnapshot,
    pub authority: Option<AuthorityBelief>,
    pub update_trace: BeliefUpdateTrace,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorldModelUpdateContext {
    pub active_goal: Option<String>,
    #[serde(default)]
    pub goal_status: BTreeMap<String, GoalStatusBelief>,
    #[serde(default)]
    pub registered_goals: Vec<String>,
    #[serde(default)]
    pub registered_behaviors: Vec<String>,
    #[serde(default)]
    pub registered_skills: Vec<String>,
    #[serde(default)]
    pub drive_summaries: BTreeMap<String, DriveSelfSummary>,
    pub commitment_age_ms: u64,
    pub active_behavior: Option<String>,
    pub active_skill: Option<String>,
    pub expected_progress: Option<f32>,
    pub recent_progress: Option<f32>,
    pub uncertainty: f32,
    pub strategy_failure_pressure: f32,
    #[serde(default)]
    pub capability_evidence: Vec<CapabilityBelief>,
    #[serde(default)]
    pub cognitive_services: BTreeMap<String, CognitiveServiceBelief>,
    pub active_control: Option<ActiveControlSummary>,
    pub continuity: ContinuitySummary,
    pub wall_clock_unix_ms: Option<u64>,
    pub replay_now_ms: Option<u64>,
    #[serde(default)]
    pub temporal_expectations: Vec<PendingTemporalExpectation>,
    pub epistemic_attempt: Option<EpistemicAttempt>,
    #[serde(default)]
    pub semantic_observations: Vec<crate::SemanticEvidenceObservation>,
}

#[derive(Clone, Debug, Default)]
pub struct WorldModelUpdater {
    revision: u64,
    entities: BTreeMap<EntityId, WorldEntity>,
    last_brainstem_boot_id: Option<BootId>,
    invalidated_authority_lease: Option<String>,
    social: SocialWorldModelBuilder,
    temporal: TemporalIntegrator,
    epistemic: EpistemicModelBuilder,
    semantic: crate::semantic::SemanticGraphBuilder,
}

impl WorldModelUpdater {
    pub fn update(&mut self, mut now: Now, context: WorldModelUpdateContext) -> Now {
        let previous = self.entities.clone();
        let mut trace = BeliefUpdateTrace {
            builder_implementation: "pete_now::WorldModelUpdater".to_string(),
            builder_version: "1".to_string(),
            ..BeliefUpdateTrace::default()
        };
        self.age_entities(now.t_ms, &mut trace);
        self.integrate_objects(&now, &mut trace);
        self.integrate_sound(&now, &mut trace);
        self.integrate_memory(&now, &mut trace);
        self.mark_contradictions(&mut trace);
        self.remove_expired(now.t_ms, &mut trace);

        for id in self.entities.keys() {
            if previous.contains_key(id) {
                if previous.get(id) != self.entities.get(id) {
                    trace.updated.push(id.0.clone());
                }
            } else {
                trace.added.push(id.0.clone());
            }
        }
        let local_geometry = local_geometry(&now);
        let hazards = hazard_beliefs(&now);
        let world_context = context_beliefs(&now);
        let authority = authority_belief(&now);
        let social = self.social.update(&now, &self.entities);
        let active_interaction = social.active_interaction.as_ref();
        let temporal = self.temporal.update(TemporalUpdateInput {
            monotonic_now_ms: now.t_ms,
            wall_clock_unix_ms: context.wall_clock_unix_ms,
            replay_now_ms: context.replay_now_ms,
            charging: now.body.charging,
            contact_or_recovery: now.body.flags.bump_left
                || now.body.flags.bump_right
                || context.active_goal.as_deref() == Some("escape_danger"),
            active_goal: context.active_goal.clone(),
            interaction_id: active_interaction
                .map(|interaction| interaction.interaction_id.0.clone()),
            interaction_participants: active_interaction
                .map(|interaction| {
                    interaction
                        .participants
                        .iter()
                        .map(|person| EntityId(person.0.clone()))
                        .collect()
                })
                .unwrap_or_default(),
            expectations: context.temporal_expectations.clone(),
            temporal_beliefs: temporal_beliefs(&now),
        });
        let epistemic = self.epistemic.update(
            &now,
            &self.entities,
            &local_geometry,
            &social,
            context.strategy_failure_pressure,
            context.epistemic_attempt.as_ref(),
        );
        let semantic = self
            .semantic
            .update(&now, &self.entities, &context.semantic_observations);
        let mut self_model = self.self_model(&now, &context);
        self_model.continuity.episode_id = temporal
            .current_episode
            .as_ref()
            .map(|episode_id| episode_id.0.clone())
            .or(self_model.continuity.episode_id);
        record_meta_evidence(&mut trace, &self_model.battery_meta);
        record_meta_evidence(&mut trace, &self_model.charging_meta);
        record_meta_evidence(&mut trace, &self_model.stuck_meta);
        record_meta_evidence(&mut trace, &self_model.pose_meta);
        if let Some(trap_kind) = &self_model.stuck_trap_kind {
            record_meta_evidence(&mut trace, &trap_kind.meta);
        }
        record_meta_evidence(&mut trace, &self_model.organism_id.meta);
        record_meta_evidence(&mut trace, &self_model.body.body_id.meta);
        record_meta_evidence(&mut trace, &self_model.body.pose.meta);
        record_meta_evidence(&mut trace, &self_model.body.energy.meta);
        record_meta_evidence(&mut trace, &self_model.body.charging.meta);
        record_meta_evidence(&mut trace, &self_model.body.health.meta);
        record_meta_evidence(&mut trace, &self_model.agency.meta);
        for capability in self_model.capabilities.capabilities.values() {
            record_meta_evidence(&mut trace, &capability.meta);
        }
        for service in self_model.service_state.services.values() {
            record_meta_evidence(&mut trace, &service.meta);
        }
        for status in self_model.goal_status.values() {
            record_meta_evidence(&mut trace, &status.meta);
        }
        for belief in [
            local_geometry.nearest_m.as_ref(),
            local_geometry.left_clearance_m.as_ref(),
            local_geometry.center_clearance_m.as_ref(),
            local_geometry.right_clearance_m.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            record_meta_evidence(&mut trace, &belief.meta);
        }
        for belief in [
            hazards.immediate_risk.as_ref(),
            hazards.remembered_risk.as_ref(),
            hazards.predicted_risk.as_ref(),
            world_context.novelty.as_ref(),
            world_context.surprise.as_ref(),
            world_context.prediction_uncertainty.as_ref(),
            world_context.map_confidence.as_ref(),
            world_context.safe_bearing_rad.as_ref(),
            world_context.frontier_bearing_rad.as_ref(),
            world_context.llm_confidence.as_ref(),
            world_context.expected_battery_delta.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            record_meta_evidence(&mut trace, &belief.meta);
        }
        if let Some(authority) = &authority {
            record_meta_evidence(&mut trace, &authority.meta);
        }
        for person in social.people.values() {
            record_meta_evidence(&mut trace, &person.meta);
            for identity in &person.identity_hypotheses {
                for evidence in &identity.evidence {
                    trace.input_evidence_ids.push(evidence.id.clone());
                }
            }
        }
        for question in &epistemic.active_questions {
            for evidence in &question.provenance {
                trace.input_evidence_ids.push(evidence.id.clone());
            }
        }
        for relation in semantic.relations.values() {
            for evidence in relation
                .supporting_evidence
                .iter()
                .chain(relation.contradicting_evidence.iter())
            {
                trace.input_evidence_ids.push(evidence.id.clone());
            }
        }
        trace.input_evidence_ids.sort();
        trace.input_evidence_ids.dedup();
        trace.added.sort();
        trace.updated.sort();
        trace.removed.sort();
        trace.freshness_changes.sort();
        trace.confidence_changes.sort();
        trace.contradiction_resolutions.sort();

        self.revision = self.revision.saturating_add(1);
        now.world = WorldModelSnapshot {
            schema_version: 3,
            revision: self.revision,
            t_ms: now.t_ms,
            entities: self.entities.clone(),
            self_model,
            local_geometry,
            hazards,
            context: world_context,
            temporal,
            social,
            epistemic,
            semantic,
            authority,
            update_trace: trace,
        };
        now
    }

    fn age_entities(&mut self, now_ms: u64, trace: &mut BeliefUpdateTrace) {
        for entity in self.entities.values_mut() {
            let age_ms = now_ms.saturating_sub(entity.last_observed_at_ms);
            let previous = entity.meta.freshness.clone();
            let previous_confidence = entity.confidence;
            entity.meta.freshness = freshness(age_ms, identity_policy(&entity.kind));
            entity.meta.valid_at_ms = now_ms;
            let base_confidence = entity
                .attributes
                .get("observed_confidence")
                .copied()
                .unwrap_or(entity.confidence);
            entity.confidence =
                decayed_confidence(base_confidence, age_ms, identity_policy(&entity.kind));
            entity.meta.confidence = entity.confidence;
            if (entity.confidence - previous_confidence).abs() > f32::EPSILON {
                trace.confidence_changes.push(format!(
                    "{}:{previous_confidence:.6}->{:.6}",
                    entity.id.0, entity.confidence
                ));
            }
            if previous != entity.meta.freshness {
                trace.freshness_changes.push(format!(
                    "{}:identity:{previous:?}->{:?}",
                    entity.id.0, entity.meta.freshness
                ));
            }
            if age_ms > bearing_policy(&entity.kind).aging_after_ms {
                if entity.bearing_rad.take().is_some() {
                    trace
                        .freshness_changes
                        .push(format!("{}:bearing:stale", entity.id.0));
                }
                entity.bearing_meta = None;
            }
            if age_ms > distance_policy(&entity.kind).aging_after_ms {
                entity.distance_m = None;
                entity.distance_meta = None;
                entity.reachability = ReachabilityEstimate::default();
                entity.reachability_meta = None;
            }
        }
    }

    fn integrate_objects(&mut self, now: &Now, trace: &mut BeliefUpdateTrace) {
        for (index, observation) in now.objects.observations.iter().enumerate() {
            let kind = WorldEntityKind::from(&observation.class);
            let id = EntityId(format!(
                "{}:{}",
                entity_kind_key(&kind),
                normalized_label(&observation.label)
            ));
            let source_kind = match observation.source {
                ObjectObservationSource::Sim | ObjectObservationSource::CreateIr => {
                    BeliefSourceKind::DirectObservation
                }
                ObjectObservationSource::HumanLabel => BeliefSourceKind::HumanClaim,
                ObjectObservationSource::Kinect | ObjectObservationSource::Captioner => {
                    BeliefSourceKind::DerivedPerception
                }
                ObjectObservationSource::Unknown => BeliefSourceKind::Unknown,
            };
            let evidence = evidence_ref(
                &format!("object.{:?}", observation.source).to_lowercase(),
                &format!("{}:{index}", observation.label),
                now.t_ms,
                "object-observation-v1",
            );
            trace.input_evidence_ids.push(evidence.id.clone());
            let meta = belief_meta(
                observation.confidence,
                now.t_ms,
                source_kind,
                evidence.clone(),
                Some("base_link".to_string()),
            );
            let pose = observation.distance_m.map(|distance| {
                let heading = now.body.odometry.heading_rad + observation.bearing_rad;
                WorldPose {
                    x_m: now.body.odometry.x_m + heading.cos() * distance,
                    y_m: now.body.odometry.y_m + heading.sin() * distance,
                }
            });
            let occluded = observation.distance_m.is_some_and(|target_distance| {
                now.objects.observations.iter().any(|other| {
                    other.class == ObjectClass::Obstacle
                        && other.distance_m.is_some_and(|obstacle_distance| {
                            obstacle_distance + 0.10 < target_distance
                                && normalize_angle(other.bearing_rad - observation.bearing_rad)
                                    .abs()
                                    < 0.28
                        })
                })
            });
            let reachable = observation.distance_m.is_some() && !occluded;
            self.entities.insert(
                id.clone(),
                WorldEntity {
                    id,
                    kind,
                    label: observation.label.clone(),
                    last_observed_at_ms: now.t_ms,
                    confidence: observation.confidence.clamp(0.0, 1.0),
                    meta: meta.clone(),
                    pose,
                    bearing_rad: Some(observation.bearing_rad),
                    bearing_meta: Some(meta.clone()),
                    distance_m: observation.distance_m,
                    distance_meta: observation.distance_m.map(|_| meta.clone()),
                    reachability: ReachabilityEstimate {
                        reachable,
                        confidence: observation.confidence.clamp(0.0, 1.0),
                    },
                    reachability_meta: Some(meta.clone()),
                    attributes: BTreeMap::from([(
                        "observed_confidence".to_string(),
                        observation.confidence.clamp(0.0, 1.0),
                    )]),
                    provenance: vec![evidence],
                },
            );
        }
    }

    fn integrate_sound(&mut self, now: &Now, trace: &mut BeliefUpdateTrace) {
        let Some(label) = now
            .ear
            .transcript
            .clone()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| (!now.ear.features.is_empty()).then(|| "unidentified sound".to_string()))
        else {
            return;
        };
        let evidence = evidence_ref("ear", "sound_source", now.t_ms, "sound-hypothesis-v1");
        trace.input_evidence_ids.push(evidence.id.clone());
        let confidence = now.ear.asr.confidence.clamp(0.2, 1.0);
        let meta = belief_meta(
            confidence,
            now.t_ms,
            BeliefSourceKind::DerivedPerception,
            evidence.clone(),
            Some("base_link".to_string()),
        );
        let id = EntityId("sound_source:current".to_string());
        self.entities.insert(
            id.clone(),
            WorldEntity {
                id,
                kind: WorldEntityKind::SoundSource,
                label,
                last_observed_at_ms: now.t_ms,
                confidence,
                meta,
                provenance: vec![evidence],
                ..WorldEntity::default()
            },
        );
    }

    fn integrate_memory(&mut self, now: &Now, trace: &mut BeliefUpdateTrace) {
        let Some(bearing) = now.memory.nearby_best_charge_direction_rad else {
            return;
        };
        let confidence =
            (now.memory.place_charge_value * now.memory.map_confidence).clamp(0.0, 1.0);
        if confidence <= 0.01 {
            return;
        }
        let evidence = evidence_ref(
            "memory.recall",
            "charger_direction",
            now.t_ms,
            "memory-belief-v1",
        );
        trace.input_evidence_ids.push(evidence.id.clone());
        let meta = belief_meta(
            confidence,
            now.t_ms,
            BeliefSourceKind::MemoryRecall,
            evidence.clone(),
            Some("base_link".to_string()),
        );
        let id = EntityId("charger:remembered_home".to_string());
        self.entities.insert(
            id.clone(),
            WorldEntity {
                id,
                kind: WorldEntityKind::Charger,
                label: "remembered charger".to_string(),
                last_observed_at_ms: now.t_ms,
                confidence,
                meta: meta.clone(),
                bearing_rad: Some(bearing),
                bearing_meta: Some(meta.clone()),
                reachability: ReachabilityEstimate {
                    reachable: false,
                    confidence,
                },
                reachability_meta: Some(meta),
                attributes: BTreeMap::from([("observed_confidence".to_string(), confidence)]),
                provenance: vec![evidence],
                ..WorldEntity::default()
            },
        );
    }

    fn mark_contradictions(&mut self, trace: &mut BeliefUpdateTrace) {
        let mut by_label: BTreeMap<String, Vec<EntityId>> = BTreeMap::new();
        for entity in self.entities.values() {
            by_label
                .entry(normalized_label(&entity.label))
                .or_default()
                .push(entity.id.clone());
        }
        for ids in by_label.values().filter(|ids| ids.len() > 1) {
            let kinds = ids
                .iter()
                .filter_map(|id| self.entities.get(id))
                .map(|entity| entity.kind.clone())
                .collect::<BTreeSet<_>>();
            if kinds.len() <= 1 {
                continue;
            }
            for id in ids {
                let contradictions = ids
                    .iter()
                    .filter(|other| *other != id)
                    .filter_map(|other| self.entities.get(other))
                    .flat_map(|entity| entity.provenance.clone())
                    .collect::<Vec<_>>();
                if let Some(entity) = self.entities.get_mut(id) {
                    entity.meta.contradiction_refs = contradictions;
                }
            }
            trace.contradiction_resolutions.push(format!(
                "preserved:{}",
                ids.iter()
                    .map(|id| id.0.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
    }

    fn remove_expired(&mut self, now_ms: u64, trace: &mut BeliefUpdateTrace) {
        self.entities.retain(|id, entity| {
            let keep = now_ms.saturating_sub(entity.last_observed_at_ms)
                <= identity_policy(&entity.kind).invalidate_after_ms
                && entity.confidence > 0.01;
            if !keep {
                trace.removed.push(id.0.clone());
            }
            keep
        });
    }

    fn self_model(&mut self, now: &Now, context: &WorldModelUpdateContext) -> SelfModelSnapshot {
        let body_evidence = evidence_ref("body", "state", now.t_ms, "body-belief-v1");
        let body_meta = belief_meta(
            1.0,
            now.t_ms,
            BeliefSourceKind::DirectObservation,
            body_evidence,
            Some("base_link".to_string()),
        );
        let stuck = now
            .extensions
            .get("sim.stuck")
            .and_then(|value| value.get("values"))
            .and_then(|value| value.as_array())
            .and_then(|values| values.first())
            .and_then(|value| value.as_f64())
            .is_some_and(|active| active > 0.0);
        let stuck_trap_kind = stuck.then(|| {
            let value = now
                .extensions
                .get("sim.stuck")
                .and_then(|value| value.get("values"))
                .and_then(|value| value.as_array())
                .and_then(|values| values.get(10))
                .and_then(|value| value.as_f64())
                .map(|code| match code.round() as i32 {
                    1 => StuckTrapKind::Wall,
                    2 => StuckTrapKind::Corner,
                    3 => StuckTrapKind::Column,
                    _ => StuckTrapKind::Unknown,
                })
                .unwrap_or_default();
            Belief {
                value,
                meta: simple_meta(
                    now.t_ms,
                    BeliefSourceKind::DerivedPerception,
                    "recovery.trap_kind",
                ),
            }
        });
        let mut goal_status = context.goal_status.clone();
        for (goal_id, status) in &mut goal_status {
            status.meta = simple_meta(
                now.t_ms,
                BeliefSourceKind::ActionOutcome,
                &format!("goal_outcome.{goal_id}"),
            );
        }
        let pose_meta = belief_meta(
            1.0,
            now.t_ms,
            BeliefSourceKind::DirectObservation,
            evidence_ref("body", "odometry", now.t_ms, "body-belief-v1"),
            Some("map".to_string()),
        );
        let identity_meta = simple_meta(
            now.t_ms,
            BeliefSourceKind::Map,
            "self.identity.configuration",
        );
        let possession = now.extensions.get("brainstem.possession");
        let device_id = possession
            .and_then(|value| value.get("brainstem_device_id"))
            .and_then(|value| value.as_str())
            .map(|value| BrainstemDeviceId(value.to_string()));
        let boot_id = possession
            .and_then(|value| value.get("brainstem_boot_id"))
            .and_then(|value| value.as_str())
            .map(|value| BootId(value.to_string()));
        let boot_changed = self
            .last_brainstem_boot_id
            .as_ref()
            .zip(boot_id.as_ref())
            .is_some_and(|(previous, current)| previous != current);
        if boot_id.is_some() {
            self.last_brainstem_boot_id = boot_id.clone();
        }
        let session_id = possession
            .and_then(|value| value.get("session_id"))
            .and_then(|value| value.as_str())
            .map(|value| SessionId(value.to_string()));
        let lease_id = possession
            .and_then(|value| value.get("lease_id"))
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned);
        if boot_changed {
            self.invalidated_authority_lease = Some(
                lease_id
                    .clone()
                    .unwrap_or_else(|| "<missing-lease>".to_string()),
            );
        } else if self
            .invalidated_authority_lease
            .as_ref()
            .zip(lease_id.as_ref())
            .is_some_and(|(invalidated, current)| invalidated != current)
        {
            self.invalidated_authority_lease = None;
        }
        let authority_invalidated = self.invalidated_authority_lease.as_ref().is_some_and(|id| {
            lease_id.as_deref() == Some(id.as_str())
                || (id == "<missing-lease>" && lease_id.is_none())
        });
        let reported_possessed = possession
            .and_then(|value| value.get("possessed"))
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let reported_armed = possession
            .and_then(|value| value.get("brainstem_armed"))
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let possessed = reported_possessed && !authority_invalidated;
        let armed = reported_armed && !authority_invalidated;
        let moving =
            now.body.velocity.forward_m_s.abs() > 0.01 || now.body.velocity.turn_rad_s.abs() > 0.01;
        let reign = now.reign.latest.as_ref();
        let controller = reign
            .map(|input| match input.mode {
                pete_actions::ReignMode::Direct => ControlProvenance::HumanDirect,
                pete_actions::ReignMode::Assist => ControlProvenance::HumanAssist,
                pete_actions::ReignMode::Suggest => ControlProvenance::HumanSuggestion,
                pete_actions::ReignMode::ObserveOnly => ControlProvenance::Autonomous,
            })
            .unwrap_or_else(|| {
                if context.active_goal.is_some() {
                    ControlProvenance::Autonomous
                } else {
                    ControlProvenance::None
                }
            });
        let agency_meta = if let Some(authority) = reign {
            simple_meta(
                now.t_ms,
                BeliefSourceKind::HumanClaim,
                &format!("reign.{}", authority.id),
            )
        } else {
            simple_meta(
                now.t_ms,
                BeliefSourceKind::ActionOutcome,
                "control.autonomous",
            )
        };

        let mut capabilities = CapabilitySelfModel::default();
        let body_current = now.t_ms.saturating_sub(now.body.last_update_ms) <= 1_000;
        insert_capability(
            &mut capabilities,
            "sensor:body",
            CapabilityKind::Sensor,
            body_current,
            (!body_current).then_some("body telemetry is stale"),
            now.t_ms,
        );
        let range_current = (!now.range.beams.is_empty() || now.range.nearest_m.is_some())
            && now.t_ms.saturating_sub(now.range.captured_at_ms) <= 1_000;
        insert_capability(
            &mut capabilities,
            "sensor:range",
            CapabilityKind::Sensor,
            range_current,
            (!range_current).then_some("range observations are missing or stale"),
            now.t_ms,
        );
        let visual_current = now.objects.observations.iter().any(|observation| {
            matches!(
                observation.source,
                ObjectObservationSource::Kinect | ObjectObservationSource::Captioner
            )
        });
        insert_capability(
            &mut capabilities,
            "sensor:vision",
            CapabilityKind::Sensor,
            visual_current,
            (!visual_current).then_some("camera evidence is unavailable this tick"),
            now.t_ms,
        );
        let drive_available = now.body.health.health > 0.2 && !now.body.flags.wheel_drop;
        insert_capability(
            &mut capabilities,
            "actuator:drive",
            CapabilityKind::Actuator,
            drive_available,
            (!drive_available).then_some("drive is unsafe or body health is degraded"),
            now.t_ms,
        );
        insert_capability(
            &mut capabilities,
            "actuator:speaker",
            CapabilityKind::Actuator,
            true,
            None,
            now.t_ms,
        );
        for goal in &context.registered_goals {
            insert_capability(
                &mut capabilities,
                &format!("goal:{goal}"),
                CapabilityKind::Goal,
                true,
                None,
                now.t_ms,
            );
        }
        for behavior in &context.registered_behaviors {
            insert_capability(
                &mut capabilities,
                &format!("behavior:{behavior}"),
                CapabilityKind::Behavior,
                true,
                None,
                now.t_ms,
            );
        }
        for skill in &context.registered_skills {
            insert_capability(
                &mut capabilities,
                &format!("skill:{skill}"),
                CapabilityKind::Skill,
                true,
                None,
                now.t_ms,
            );
        }
        for capability in &context.capability_evidence {
            capabilities
                .capabilities
                .insert(capability.id.clone(), capability.clone());
        }
        let hardware_authority_available = possession.is_none() || (possessed && armed);
        for capability in capabilities.capabilities.values_mut() {
            if matches!(
                capability.kind,
                CapabilityKind::Actuator | CapabilityKind::Behavior | CapabilityKind::Skill
            ) {
                capability.authorized = capability.authorized && hardware_authority_available;
                if !capability.authorized && capability.authority_reason.is_none() {
                    capability.authority_reason = Some(if authority_invalidated {
                        "brainstem reboot invalidated the control lease".to_string()
                    } else {
                        "no current actuation authority".to_string()
                    });
                }
            }
        }
        let mut service_state = CognitiveServiceSummary {
            services: context.cognitive_services.clone(),
        };
        integrate_cognitive_registry(now, &mut service_state);
        service_state
            .services
            .entry("local_language".to_string())
            .or_insert_with(|| CognitiveServiceBelief {
                available: true,
                confidence: 1.0,
                meta: simple_meta(now.t_ms, BeliefSourceKind::Map, "service.local_language"),
                ..CognitiveServiceBelief::default()
            });
        service_state
            .services
            .entry("rich_language".to_string())
            .or_insert_with(|| CognitiveServiceBelief {
                available: false,
                confidence: 1.0,
                unavailable_reason: Some("enhanced cognition was not reported available".into()),
                meta: simple_meta(
                    now.t_ms,
                    BeliefSourceKind::Map,
                    "service.rich_language.missing",
                ),
                ..CognitiveServiceBelief::default()
            });
        for (service, state) in &service_state.services {
            insert_capability(
                &mut capabilities,
                &format!("service:{service}"),
                CapabilityKind::CognitiveService,
                state.available,
                state.unavailable_reason.as_deref(),
                now.t_ms,
            );
        }
        let faults = [
            now.body.flags.wheel_drop.then_some("wheel_drop"),
            (now.body.health.health <= 0.2).then_some("body_health_critical"),
        ]
        .into_iter()
        .flatten()
        .map(|fault| Belief {
            value: fault.to_string(),
            meta: body_meta.clone(),
        })
        .collect::<Vec<_>>();
        let tilt_known = now.imu.orientation.len() >= 2;
        let tilted = tilt_known
            .then(|| now.imu.orientation[0].abs() > 0.35 || now.imu.orientation[1].abs() > 0.35);
        let body = SelfBodyBelief {
            body_id: Belief {
                value: BodyId("pete.primary_body".to_string()),
                meta: identity_meta.clone(),
            },
            implementation: Belief {
                value: "mobile_robot".to_string(),
                meta: identity_meta.clone(),
            },
            implementation_version: Belief {
                value: "1".to_string(),
                meta: identity_meta.clone(),
            },
            brainstem_device_id: device_id.map(|value| Belief {
                value,
                meta: agency_meta.clone(),
            }),
            brainstem_boot_id: boot_id.map(|value| Belief {
                value,
                meta: agency_meta.clone(),
            }),
            pose: Belief {
                value: now.body.odometry,
                meta: pose_meta.clone(),
            },
            envelope: Belief {
                value: BodyEnvelope {
                    radius_m: 0.18,
                    height_m: 0.10,
                },
                meta: identity_meta.clone(),
            },
            energy: Belief {
                value: now.body.battery_level,
                meta: body_meta.clone(),
            },
            charging: Belief {
                value: now.body.charging,
                meta: body_meta.clone(),
            },
            health: Belief {
                value: now.body.health.health,
                meta: body_meta.clone(),
            },
            faults,
            being_moved: Some(Belief {
                value: moving && context.active_behavior.is_none() && reign.is_none(),
                meta: body_meta.clone(),
            }),
            tilted: tilted.map(|value| Belief {
                value,
                meta: simple_meta(
                    now.t_ms,
                    BeliefSourceKind::DirectObservation,
                    "body.imu.tilt",
                ),
            }),
            blocked: Some(Belief {
                value: stuck || now.body.flags.bump_left || now.body.flags.bump_right,
                meta: body_meta.clone(),
            }),
            carried: now.body.flags.wheel_drop.then(|| Belief {
                value: true,
                meta: body_meta.clone(),
            }),
        };
        let motivation = MotivationSummary {
            drives: context.drive_summaries.clone(),
            selected_goal: context.active_goal.clone(),
            commitment_age_ms: context.commitment_age_ms,
            expected_progress: context.expected_progress,
            recent_progress: context.recent_progress,
            uncertainty: context.uncertainty,
            strategy_failure_pressure: context.strategy_failure_pressure,
        };
        let mut continuity = context.continuity.clone();
        continuity.session_id = continuity.session_id.or_else(|| session_id.clone());
        continuity.important_relationship_refs.extend(
            self.entities
                .values()
                .filter(|entity| entity.kind == WorldEntityKind::Person)
                .map(|entity| entity.id.clone()),
        );
        for entity in &now.memory.remembered_entities {
            if entity.has_label("Person") || entity.has_label("person") {
                continuity
                    .important_relationship_refs
                    .push(EntityId(entity.id.clone()));
            }
            if entity.has_label("Place") || entity.has_label("place") {
                continuity.important_place_refs.push(entity.id.clone());
            }
        }
        continuity.important_relationship_refs.sort();
        continuity.important_relationship_refs.dedup();
        continuity.important_place_refs.sort();
        continuity.important_place_refs.dedup();
        let mut active_control =
            context
                .active_control
                .clone()
                .unwrap_or_else(|| ActiveControlSummary {
                    goal_id: context.active_goal.clone(),
                    behavior_id: context.active_behavior.clone(),
                    skill_id: context.active_skill.clone(),
                    provenance: controller.clone(),
                    unable_to_act_reason: (!drive_available)
                        .then_some("drive capability is unavailable".to_string()),
                    ..ActiveControlSummary::default()
                });
        if reign.is_some_and(|input| input.mode == pete_actions::ReignMode::Direct) {
            active_control.provenance = ControlProvenance::HumanDirect;
        }
        SelfModelSnapshot {
            organism_id: Belief {
                value: OrganismId("pete".to_string()),
                meta: identity_meta,
            },
            body,
            capabilities,
            agency: AgencyState {
                controller,
                reign_mode: reign.map(|input| format!("{:?}", input.mode).to_ascii_lowercase()),
                reign_source: reign.map(|input| format!("{:?}", input.source).to_ascii_lowercase()),
                session_id: session_id.map(|value| Belief {
                    value,
                    meta: agency_meta.clone(),
                }),
                lease_id: lease_id.map(|value| Belief {
                    value,
                    meta: agency_meta.clone(),
                }),
                possessed: Belief {
                    value: possessed,
                    meta: agency_meta.clone(),
                },
                armed: Belief {
                    value: armed,
                    meta: agency_meta.clone(),
                },
                stopped: !moving,
                moving,
                pending_direct_override: reign
                    .is_some_and(|input| input.mode == pete_actions::ReignMode::Direct),
                authority_conflicts: if authority_invalidated {
                    agency_meta.provenance.clone()
                } else {
                    Vec::new()
                },
                meta: agency_meta,
            },
            motivation,
            active_control,
            continuity,
            service_state,
            meta: body_meta.clone(),
            battery_level: now.body.battery_level,
            battery_meta: body_meta.clone(),
            charging: now.body.charging,
            charging_meta: body_meta.clone(),
            stuck,
            stuck_meta: body_meta,
            stuck_trap_kind,
            pose: now.body.odometry,
            pose_meta,
            contact: now.body.flags.bump_left || now.body.flags.bump_right || now.body.flags.wall,
            bump_left: now.body.flags.bump_left,
            moving,
            range_nearest_m: now.range.nearest_m,
            active_goal: context.active_goal.clone(),
            goal_status,
        }
    }
}

fn integrate_cognitive_registry(now: &Now, services: &mut CognitiveServiceSummary) {
    let Some(value) = now.extensions.get("cognition.registry") else {
        return;
    };
    let Ok(registry) = serde_json::from_value::<ProviderRegistrySnapshot>(value.clone()) else {
        return;
    };
    for provider in registry.providers.values() {
        for capability in &provider.capabilities {
            let key = capability.capability.as_str().to_string();
            let available = matches!(
                provider.health.state,
                ProviderHealthState::Available | ProviderHealthState::Degraded
            ) && now.t_ms <= provider.health.valid_until_ms;
            let candidate = CognitiveServiceBelief {
                provider_id: Some(provider.provider_id.0.clone()),
                role: Some(provider.role.as_str().to_string()),
                capability: Some(key.clone()),
                capability_version: Some(capability.version.clone()),
                available,
                confidence: (provider.health.confidence * capability.performance_confidence)
                    .clamp(0.0, 1.0),
                unavailable_reason: (!available).then(|| {
                    provider.health.reason.clone().unwrap_or_else(|| {
                        format!("provider health is {:?}", provider.health.state)
                            .to_ascii_lowercase()
                    })
                }),
                host_id: provider.host_id.as_ref().map(|id| HostId(id.0.clone())),
                process_id: provider
                    .process_id
                    .as_ref()
                    .map(|id| ProcessId(id.0.clone())),
                implementation: Some(provider.implementation.clone()),
                implementation_version: Some(provider.implementation_version.clone()),
                model_version: provider.model_version.clone(),
                locality: Some(provider.locality.as_str().to_string()),
                resource_class: Some(provider.resource_class.as_str().to_string()),
                meta: simple_meta(
                    now.t_ms,
                    BeliefSourceKind::DerivedPerception,
                    &format!("cognition.registry.{}", provider.provider_id.0),
                ),
            };
            let replace = services.services.get(&key).map_or(true, |incumbent| {
                (candidate.available && !incumbent.available)
                    || (candidate.available == incumbent.available
                        && (candidate.confidence > incumbent.confidence
                            || (candidate.confidence == incumbent.confidence
                                && candidate.provider_id < incumbent.provider_id)))
            });
            if replace {
                services.services.insert(key, candidate);
            }
        }
    }
}

fn insert_capability(
    model: &mut CapabilitySelfModel,
    id: &str,
    kind: CapabilityKind,
    available: bool,
    unavailable_reason: Option<&str>,
    now_ms: u64,
) {
    let id = CapabilityId(id.to_string());
    model.capabilities.insert(
        id.clone(),
        CapabilityBelief {
            id,
            kind,
            availability: if available {
                CapabilityAvailability::Available
            } else {
                CapabilityAvailability::Unavailable
            },
            confidence: 1.0,
            unavailable_reason: unavailable_reason.map(ToOwned::to_owned),
            authorized: available,
            meta: simple_meta(now_ms, BeliefSourceKind::Map, "self.capability.registry"),
            ..CapabilityBelief::default()
        },
    );
}

fn record_meta_evidence(trace: &mut BeliefUpdateTrace, meta: &BeliefMeta) {
    trace
        .input_evidence_ids
        .extend(meta.provenance.iter().map(|evidence| evidence.id.clone()));
    trace.input_evidence_ids.extend(
        meta.contradiction_refs
            .iter()
            .map(|evidence| evidence.id.clone()),
    );
}

fn context_beliefs(now: &Now) -> ContextBeliefs {
    let memory_present = now.memory.map_confidence > 0.0
        || now.memory.places_visited > 0
        || !now.memory.remembered_entities.is_empty();
    let predictions_present = !now.predictions.expected_events.is_empty()
        || now.predictions.danger_model.is_some()
        || now.predictions.danger_hardcoded.is_some()
        || now.predictions.charge_model.is_some()
        || now.predictions.charge_hardcoded.is_some();
    ContextBeliefs {
        novelty: memory_present.then(|| Belief {
            value: now.memory.place_novelty.clamp(0.0, 1.0),
            meta: simple_meta(now.t_ms, BeliefSourceKind::MemoryRecall, "memory.novelty"),
        }),
        surprise: Some(Belief {
            value: now.surprise.total.clamp(0.0, 1.0),
            meta: simple_meta(
                now.t_ms,
                BeliefSourceKind::DerivedPerception,
                "surprise.total",
            ),
        }),
        prediction_uncertainty: predictions_present.then(|| Belief {
            value: now.predictions.uncertainty.clamp(0.0, 1.0),
            meta: simple_meta(
                now.t_ms,
                BeliefSourceKind::LearnedPrediction,
                "prediction.uncertainty",
            ),
        }),
        map_confidence: memory_present.then(|| Belief {
            value: now.memory.map_confidence.clamp(0.0, 1.0),
            meta: simple_meta(now.t_ms, BeliefSourceKind::Map, "memory.map_confidence"),
        }),
        safe_bearing_rad: now
            .memory
            .nearby_best_safe_direction_rad
            .map(|value| Belief {
                value,
                meta: simple_meta(
                    now.t_ms,
                    BeliefSourceKind::MemoryRecall,
                    "memory.safe_bearing",
                ),
            }),
        frontier_bearing_rad: now
            .memory
            .nearby_frontier_direction_rad
            .map(|value| Belief {
                value,
                meta: simple_meta(
                    now.t_ms,
                    BeliefSourceKind::MemoryRecall,
                    "memory.frontier_bearing",
                ),
            }),
        llm_confidence: (now.llm.command_summary.is_some() || now.llm.critique.is_some()).then(
            || Belief {
                value: now.llm.confidence.clamp(0.0, 1.0),
                meta: simple_meta(now.t_ms, BeliefSourceKind::LlmClaim, "llm.confidence"),
            },
        ),
        expected_battery_delta: now
            .predictions
            .charge_model
            .or(now.predictions.charge_hardcoded)
            .map(|prediction| Belief {
                value: prediction.expected_battery_delta,
                meta: simple_meta(
                    now.t_ms,
                    BeliefSourceKind::LearnedPrediction,
                    "prediction.expected_battery_delta",
                ),
            }),
    }
}

fn local_geometry(now: &Now) -> LocalGeometrySnapshot {
    let range_belief = |key: &str, value: f32| Belief {
        value,
        meta: simple_meta(
            now.t_ms,
            BeliefSourceKind::DirectObservation,
            &format!("range.{key}"),
        ),
    };
    let (left, center, right) = clearance_buckets(&now.range.beams);
    LocalGeometrySnapshot {
        nearest_m: now
            .range
            .nearest_m
            .map(|value| range_belief("nearest", value)),
        left_clearance_m: left.map(|value| range_belief("left_clearance", value)),
        center_clearance_m: center.map(|value| range_belief("center_clearance", value)),
        right_clearance_m: right.map(|value| range_belief("right_clearance", value)),
    }
}

fn clearance_buckets(beams: &[f32]) -> (Option<f32>, Option<f32>, Option<f32>) {
    if beams.is_empty() {
        return (None, None, None);
    }
    let third = (beams.len() / 3).max(1);
    let left_end = third.min(beams.len());
    let right_start = beams.len().saturating_sub(third);
    let center_start = left_end.saturating_sub(1).min(beams.len());
    let center_end = (right_start + 1).min(beams.len()).max(center_start + 1);
    let nearest = |slice: &[f32]| slice.iter().copied().reduce(f32::min);
    (
        nearest(&beams[..left_end]),
        nearest(&beams[center_start..center_end]),
        nearest(&beams[right_start..]),
    )
}

fn authority_belief(now: &Now) -> Option<AuthorityBelief> {
    now.reign.latest.clone().map(|input| AuthorityBelief {
        meta: simple_meta(now.t_ms, BeliefSourceKind::HumanClaim, "authority.reign"),
        input,
    })
}

#[derive(Clone, Copy)]
struct FreshnessPolicy {
    current_for_ms: u64,
    aging_after_ms: u64,
    invalidate_after_ms: u64,
}

fn identity_policy(kind: &WorldEntityKind) -> FreshnessPolicy {
    match kind {
        WorldEntityKind::Person | WorldEntityKind::SoundSource => FreshnessPolicy {
            current_for_ms: 1_000,
            aging_after_ms: 5_000,
            invalidate_after_ms: 15_000,
        },
        _ => FreshnessPolicy {
            current_for_ms: 2_000,
            aging_after_ms: 15_000,
            invalidate_after_ms: 60_000,
        },
    }
}

fn bearing_policy(_kind: &WorldEntityKind) -> FreshnessPolicy {
    FreshnessPolicy {
        current_for_ms: 500,
        aging_after_ms: 2_000,
        invalidate_after_ms: 3_000,
    }
}

fn distance_policy(_kind: &WorldEntityKind) -> FreshnessPolicy {
    FreshnessPolicy {
        current_for_ms: 500,
        aging_after_ms: 2_000,
        invalidate_after_ms: 3_000,
    }
}

fn freshness(age_ms: u64, policy: FreshnessPolicy) -> Freshness {
    if age_ms <= policy.current_for_ms {
        Freshness::Current
    } else if age_ms <= policy.aging_after_ms {
        Freshness::Aging
    } else if age_ms <= policy.invalidate_after_ms {
        Freshness::Stale
    } else {
        Freshness::Invalidated
    }
}

fn decayed_confidence(base: f32, age_ms: u64, policy: FreshnessPolicy) -> f32 {
    if age_ms <= policy.current_for_ms {
        base.clamp(0.0, 1.0)
    } else {
        let span = policy
            .invalidate_after_ms
            .saturating_sub(policy.current_for_ms)
            .max(1);
        let elapsed = age_ms.saturating_sub(policy.current_for_ms);
        (base * (1.0 - elapsed as f32 / span as f32)).clamp(0.0, 1.0)
    }
}

fn hazard_beliefs(now: &Now) -> HazardBeliefs {
    let contact = now.body.flags.bump_left
        || now.body.flags.bump_right
        || now.body.flags.wall
        || now.body.flags.wheel_drop;
    let range_risk = now
        .range
        .nearest_m
        .map(|distance| ((0.35 - distance) / 0.35).clamp(0.0, 1.0));
    let immediate = if contact { Some(1.0) } else { range_risk };
    let predicted = now
        .predictions
        .danger_model
        .or(now.predictions.danger_hardcoded)
        .map(|prediction| {
            prediction
                .bump_risk
                .max(prediction.cliff_risk)
                .max(prediction.wheel_drop_risk)
                .max(prediction.stuck_risk)
        });
    HazardBeliefs {
        immediate_risk: immediate.map(|value| Belief {
            value,
            meta: simple_meta(now.t_ms, BeliefSourceKind::DirectObservation, "range/body"),
        }),
        remembered_risk: (now.memory.map_confidence > 0.0).then(|| Belief {
            value: now.memory.place_danger.clamp(0.0, 1.0),
            meta: simple_meta(
                now.t_ms,
                BeliefSourceKind::MemoryRecall,
                "memory.place_danger",
            ),
        }),
        predicted_risk: predicted.map(|value| Belief {
            value,
            meta: simple_meta(
                now.t_ms,
                BeliefSourceKind::LearnedPrediction,
                "prediction.danger",
            ),
        }),
    }
}

fn temporal_beliefs(now: &Now) -> Vec<TemporalBelief> {
    let vectors = now
        .eye
        .image_vectors
        .iter()
        .chain(now.eye.image_description_vectors.iter())
        .chain(now.eye.scene_vectors.iter())
        .chain(now.face.vectors.iter())
        .chain(now.voice.vectors.iter())
        .chain(now.objects.vectors.iter())
        .chain(now.ear.transcript_vectors.iter());
    let mut beliefs = Vec::new();
    for vector in vectors {
        let Some(occurred_at_ms) = vector.occurred_at_ms else {
            continue;
        };
        let evidence = evidence_ref(
            &format!("vector.{}", vector.collection),
            &vector.point_id,
            now.t_ms,
            "temporal-evidence-v1",
        );
        let subject = format!("vector:{}:{}", vector.collection, vector.point_id);
        beliefs.push(TemporalBelief {
            interval: TimeInterval {
                domain: ClockDomain::Event,
                start_ms: occurred_at_ms,
                end_ms: Some(occurred_at_ms),
                uncertainty_ms: 0,
            },
            relation: TemporalRelation::OccurredDuring,
            subject: subject.clone(),
            confidence: 1.0,
            provenance: vec![evidence.clone()],
        });
        beliefs.push(TemporalBelief {
            interval: TimeInterval {
                domain: ClockDomain::Observation,
                start_ms: now.t_ms,
                end_ms: Some(now.t_ms),
                uncertainty_ms: 0,
            },
            relation: if occurred_at_ms < now.t_ms {
                TemporalRelation::After
            } else {
                TemporalRelation::Overlaps
            },
            subject,
            confidence: 1.0,
            provenance: vec![evidence],
        });
    }
    beliefs
}

fn simple_meta(now_ms: u64, source_kind: BeliefSourceKind, key: &str) -> BeliefMeta {
    let evidence = evidence_ref(key, key, now_ms, "world-model-v1");
    belief_meta(1.0, now_ms, source_kind, evidence, None)
}

fn belief_meta(
    confidence: f32,
    now_ms: u64,
    source_kind: BeliefSourceKind,
    evidence: EvidenceRef,
    coordinate_frame: Option<FrameId>,
) -> BeliefMeta {
    BeliefMeta {
        confidence: confidence.clamp(0.0, 1.0),
        observed_at_ms: now_ms,
        valid_at_ms: now_ms,
        freshness: Freshness::Current,
        provenance: vec![evidence],
        contradiction_refs: Vec::new(),
        coordinate_frame,
        source_kind,
    }
}

fn evidence_ref(source: &str, key: &str, now_ms: u64, implementation: &str) -> EvidenceRef {
    EvidenceRef {
        id: format!("{source}:{key}:{now_ms}"),
        source: source.to_string(),
        key: key.to_string(),
        observed_at_ms: now_ms,
        transformation_lineage: vec![implementation.to_string()],
        implementation_version: Some("1".to_string()),
    }
}

fn entity_kind_key(kind: &WorldEntityKind) -> &'static str {
    match kind {
        WorldEntityKind::Charger => "charger",
        WorldEntityKind::Person => "person",
        WorldEntityKind::Obstacle => "obstacle",
        WorldEntityKind::SoundSource => "sound_source",
        WorldEntityKind::Landmark => "landmark",
        WorldEntityKind::Door => "door",
        WorldEntityKind::Region => "region",
        WorldEntityKind::Unknown => "unknown",
    }
}

fn normalized_label(label: &str) -> String {
    let normalized = label
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    if normalized.is_empty() {
        "unlabeled".to_string()
    } else {
        normalized
    }
}

fn normalize_angle(mut angle: f32) -> f32 {
    while angle > std::f32::consts::PI {
        angle -= std::f32::consts::TAU;
    }
    while angle < -std::f32::consts::PI {
        angle += std::f32::consts::TAU;
    }
    angle
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ObjectObservation, ObjectSense, TypedTimestamp, VectorArtifact};
    use pete_actions::{ReignCommand, ReignMode, ReignSource};
    use pete_body::BodySense;
    use uuid::Uuid;

    fn cognition_registry_now(
        t_ms: u64,
        host: &str,
        process: &str,
        state: ProviderHealthState,
    ) -> Now {
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
        assert!(!second
            .world
            .self_model
            .capabilities
            .is_available("service:rich_language"));
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
                belief.interval.domain == ClockDomain::Observation
                    && belief.interval.start_ms == 500
            }));
        assert_eq!(
            snapshot.temporal.replay_now,
            Some(TypedTimestamp {
                domain: ClockDomain::Replay,
                ms: 40,
            })
        );
    }
}
