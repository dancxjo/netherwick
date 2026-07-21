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
    /// The service is healthy but currently processing a request.
    ///
    /// Busy services remain available: callers may rely on the capability even
    /// when they cannot expect a new request to begin immediately.
    #[serde(default)]
    pub busy: bool,
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
