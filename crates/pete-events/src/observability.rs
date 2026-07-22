use std::collections::BTreeMap;

use pete_actions::{ReignInput, ReignOutcome};
use pete_autonomic::SafetyDecision;
use pete_core::{Provenance, ProvenanceKind};
use pete_experience::{Experience, Impression, Sensation, SensationPayloadKind, VectorEmbedding};
use pete_now::Now;
use pete_now::{CalibrationArtifactIdentity, CalibrationTransition, CalibrationTransitionKind};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;
use uuid::Uuid;

use crate::{Event, EventKind};

pub const BRAIN_EVENT_SCHEMA_VERSION: u32 = 1;
pub const MAX_INLINE_BRAIN_EVENT_PAYLOAD_BYTES: usize = 16 * 1024;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BrainEventId(pub String);

impl BrainEventId {
    pub fn new() -> Self {
        Self(format!("event:{}", Uuid::new_v4()))
    }

    pub fn from_domain(domain: &str, id: impl std::fmt::Display) -> Self {
        Self(format!("{domain}:{id}"))
    }

    pub fn sensation(id: Uuid) -> Self {
        Self::from_domain("sensation", id)
    }

    pub fn impression(id: Uuid) -> Self {
        Self::from_domain("impression", id)
    }

    pub fn experience(id: Uuid) -> Self {
        Self::from_domain("experience", id)
    }
}

impl std::fmt::Display for BrainEventId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrainEventType {
    Evidence,
    Interpretation,
    BeliefUpdate,
    Proposal,
    GateDecision,
    Command,
    Outcome,
    CalibrationTransition,
    ProviderState,
    JobState,
    ResourceState,
    QueueState,
    Snapshot,
    TransportGap,
    #[default]
    Unknown,
}

impl BrainEventType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Evidence => "evidence",
            Self::Interpretation => "interpretation",
            Self::BeliefUpdate => "belief_update",
            Self::Proposal => "proposal",
            Self::GateDecision => "gate_decision",
            Self::Command => "command",
            Self::Outcome => "outcome",
            Self::CalibrationTransition => "calibration_transition",
            Self::ProviderState => "provider_state",
            Self::JobState => "job_state",
            Self::ResourceState => "resource_state",
            Self::QueueState => "queue_state",
            Self::Snapshot => "snapshot",
            Self::TransportGap => "transport_gap",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Brain {
    Brainstem,
    Motherbrain,
    Forebrain,
    HigherBrain,
    Human,
    Simulator,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProducerIdentity {
    pub brain: Brain,
    pub component: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance: Option<String>,
}

impl ProducerIdentity {
    pub fn new(brain: Brain, component: impl Into<String>) -> Self {
        Self {
            brain,
            component: component.into(),
            instance: None,
        }
    }

    pub fn with_instance(mut self, instance: impl Into<String>) -> Self {
        self.instance = Some(instance.into());
        self
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockedTime {
    pub t_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clock_epoch: Option<String>,
}

impl ClockedTime {
    pub fn new(t_ms: u64) -> Self {
        Self {
            t_ms,
            clock_epoch: None,
        }
    }

    pub fn in_epoch(t_ms: u64, clock_epoch: impl Into<String>) -> Self {
        Self {
            t_ms,
            clock_epoch: Some(clock_epoch.into()),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventTimes {
    pub occurred: ClockedTime,
    pub observed: ClockedTime,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<ClockedTime>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<ClockedTime>,
}

impl EventTimes {
    pub fn observed(occurred_at_ms: u64, observed_at_ms: u64) -> Self {
        Self {
            occurred: ClockedTime::new(occurred_at_ms),
            observed: ClockedTime::new(observed_at_ms),
            valid_from: None,
            expires_at: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrainEventRecordKind {
    #[default]
    HistoricalEvent,
    StateProjection,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainReferences {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experience_id: Option<String>,
    #[serde(default)]
    pub entity_ids: Vec<String>,
    #[serde(default)]
    pub goal_ids: Vec<String>,
    #[serde(default)]
    pub command_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypedEventRef {
    pub event_id: BrainEventId,
    pub event_type: BrainEventType,
}

impl TypedEventRef {
    pub fn new(event_id: BrainEventId, event_type: BrainEventType) -> Self {
        Self {
            event_id,
            event_type,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalLinks {
    #[serde(default)]
    pub parents: Vec<TypedEventRef>,
    #[serde(default)]
    pub supports: Vec<TypedEventRef>,
    #[serde(default)]
    pub contradicts: Vec<TypedEventRef>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CausalRelationKind {
    Parent,
    Supports,
    Contradicts,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CausalReference<'a> {
    pub relation: CausalRelationKind,
    pub target: &'a TypedEventRef,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FreshnessState {
    Current,
    Aging,
    Stale,
    Expired,
    Invalidated,
    Missing,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Freshness {
    pub state: FreshnessState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub age_ms: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustState {
    Trusted,
    Conditional,
    Untrusted,
    Rejected,
    #[default]
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventDisposition {
    Unavailable,
    Rejected,
    Expired,
    Superseded,
    Vetoed,
    Accepted,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EventQuality {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertainty: Option<Uncertainty>,
    pub freshness: Freshness,
    pub trust: TrustState,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Uncertainty {
    pub value: f32,
    pub measure: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Calibration,
    Model,
    Configuration,
    Capture,
    Vector,
    #[default]
    Other,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactIdentity {
    pub kind: ArtifactKind,
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PayloadReference {
    pub id: String,
    pub locator: String,
    pub media_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub byte_len: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
    #[serde(default)]
    pub redacted: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "storage", rename_all = "snake_case")]
pub enum BrainEventPayload {
    Inline {
        #[serde(default = "default_json_media_type")]
        media_type: String,
        data: Value,
    },
    Reference {
        reference: PayloadReference,
    },
    #[default]
    Empty,
}

fn default_json_media_type() -> String {
    "application/json".to_string()
}

impl BrainEventPayload {
    pub fn inline(data: Value) -> Self {
        Self::Inline {
            media_type: default_json_media_type(),
            data,
        }
    }

    pub fn referenced(
        id: impl Into<String>,
        locator: impl Into<String>,
        media_type: impl Into<String>,
    ) -> Self {
        Self::Reference {
            reference: PayloadReference {
                id: id.into(),
                locator: locator.into(),
                media_type: media_type.into(),
                ..PayloadReference::default()
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthoritySignificance {
    Advisory,
    Proposal,
    Gate,
    Command,
    Outcome,
    SafetyTransition,
    AuthorityTransition,
    #[default]
    None,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case")]
pub enum LossPolicy {
    #[default]
    LossIntolerant,
    Coalescible {
        key: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrainEvent {
    pub schema_version: u32,
    pub event_id: BrainEventId,
    pub event_type: BrainEventType,
    pub kind: String,
    pub record_kind: BrainEventRecordKind,
    pub producer: ProducerIdentity,
    pub times: EventTimes,
    #[serde(default)]
    pub references: DomainReferences,
    #[serde(default)]
    pub links: CausalLinks,
    #[serde(default)]
    pub quality: EventQuality,
    #[serde(default)]
    pub disposition: EventDisposition,
    #[serde(default)]
    pub calibration_epochs: Vec<String>,
    #[serde(default)]
    pub artifacts: Vec<ArtifactIdentity>,
    #[serde(default)]
    pub payload: BrainEventPayload,
    #[serde(default)]
    pub authority: AuthoritySignificance,
    #[serde(default)]
    pub loss_policy: LossPolicy,
}

impl BrainEvent {
    pub fn historical(
        event_id: BrainEventId,
        event_type: BrainEventType,
        producer: ProducerIdentity,
        times: EventTimes,
    ) -> Self {
        Self {
            schema_version: BRAIN_EVENT_SCHEMA_VERSION,
            event_id,
            event_type,
            kind: event_type.as_str().to_string(),
            record_kind: BrainEventRecordKind::HistoricalEvent,
            producer,
            times,
            references: DomainReferences::default(),
            links: CausalLinks::default(),
            quality: EventQuality::default(),
            disposition: EventDisposition::Unknown,
            calibration_epochs: Vec::new(),
            artifacts: Vec::new(),
            payload: BrainEventPayload::Empty,
            authority: AuthoritySignificance::None,
            loss_policy: LossPolicy::LossIntolerant,
        }
    }

    pub fn coalescible(mut self, key: impl Into<String>) -> Self {
        self.loss_policy = LossPolicy::Coalescible { key: key.into() };
        self
    }

    pub fn causal_references(&self) -> impl Iterator<Item = CausalReference<'_>> {
        self.links
            .parents
            .iter()
            .map(|target| CausalReference {
                relation: CausalRelationKind::Parent,
                target,
            })
            .chain(self.links.supports.iter().map(|target| CausalReference {
                relation: CausalRelationKind::Supports,
                target,
            }))
            .chain(self.links.contradicts.iter().map(|target| CausalReference {
                relation: CausalRelationKind::Contradicts,
                target,
            }))
    }

    pub fn requires_loss_intolerant_delivery(&self) -> bool {
        matches!(
            self.event_type,
            BrainEventType::GateDecision
                | BrainEventType::Command
                | BrainEventType::Outcome
                | BrainEventType::CalibrationTransition
                | BrainEventType::TransportGap
        ) || matches!(
            self.authority,
            AuthoritySignificance::Gate
                | AuthoritySignificance::Command
                | AuthoritySignificance::Outcome
                | AuthoritySignificance::SafetyTransition
                | AuthoritySignificance::AuthorityTransition
        )
    }

    pub fn disposition_at(&self, now: &ClockedTime) -> EventDisposition {
        if self.times.expires_at.as_ref().is_some_and(|expires_at| {
            expires_at.clock_epoch == now.clock_epoch && now.t_ms >= expires_at.t_ms
        }) {
            EventDisposition::Expired
        } else {
            self.disposition
        }
    }

    pub fn validate(&self) -> Result<(), BrainEventError> {
        if self.schema_version != BRAIN_EVENT_SCHEMA_VERSION {
            return Err(BrainEventError::UnsupportedSchema(self.schema_version));
        }
        if self.event_id.0.trim().is_empty() {
            return Err(BrainEventError::Invalid("event_id is empty".to_string()));
        }
        if self.producer.component.trim().is_empty() {
            return Err(BrainEventError::Invalid(
                "producer.component is empty".to_string(),
            ));
        }
        if self.kind.trim().is_empty() {
            return Err(BrainEventError::Invalid("kind is empty".to_string()));
        }
        validate_unit_interval("quality.confidence", self.quality.confidence)?;
        if let Some(uncertainty) = &self.quality.uncertainty {
            if !uncertainty.value.is_finite() || uncertainty.value < 0.0 {
                return Err(BrainEventError::Invalid(
                    "quality.uncertainty.value must be finite and non-negative".to_string(),
                ));
            }
            if uncertainty.measure.trim().is_empty() {
                return Err(BrainEventError::Invalid(
                    "quality.uncertainty.measure is empty".to_string(),
                ));
            }
        }
        if let (Some(valid_from), Some(expires_at)) =
            (&self.times.valid_from, &self.times.expires_at)
        {
            if valid_from.clock_epoch == expires_at.clock_epoch && expires_at.t_ms < valid_from.t_ms
            {
                return Err(BrainEventError::Invalid(
                    "expires_at precedes valid_from in the same clock epoch".to_string(),
                ));
            }
        }
        if self
            .causal_references()
            .any(|reference| reference.target.event_id == self.event_id)
        {
            return Err(BrainEventError::Invalid(
                "event cannot causally reference itself".to_string(),
            ));
        }
        if let BrainEventPayload::Inline { data, .. } = &self.payload {
            let size = serde_json::to_vec(data)?.len();
            if size > MAX_INLINE_BRAIN_EVENT_PAYLOAD_BYTES {
                return Err(BrainEventError::InlinePayloadTooLarge(size));
            }
        }
        if self.requires_loss_intolerant_delivery()
            && !matches!(self.loss_policy, LossPolicy::LossIntolerant)
        {
            return Err(BrainEventError::LossIntolerantClassCoalesced);
        }
        Ok(())
    }

    pub fn decode_json(bytes: &[u8]) -> Result<Self, BrainEventError> {
        let value: Value = serde_json::from_slice(bytes)?;
        let version = value
            .get("schema_version")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let event = match version {
            0 => LegacyBrainEventV0::from_value(value)?.migrate(),
            1 => serde_json::from_value(value)?,
            other => return Err(BrainEventError::UnsupportedSchema(other as u32)),
        };
        event.validate()?;
        Ok(event)
    }

    pub fn from_now_snapshot(
        snapshot_id: impl Into<String>,
        now: &Now,
        observed_at_ms: u64,
        clock_epoch: Option<String>,
    ) -> Self {
        let snapshot_id = snapshot_id.into();
        let mut event = Self::historical(
            BrainEventId::from_domain("now", &snapshot_id),
            BrainEventType::Snapshot,
            ProducerIdentity::new(Brain::Motherbrain, "now"),
            EventTimes {
                occurred: ClockedTime {
                    t_ms: now.t_ms,
                    clock_epoch: clock_epoch.clone(),
                },
                observed: ClockedTime {
                    t_ms: observed_at_ms,
                    clock_epoch: clock_epoch.clone(),
                },
                valid_from: Some(ClockedTime {
                    t_ms: now.t_ms,
                    clock_epoch: clock_epoch.clone(),
                }),
                expires_at: None,
            },
        );
        event.record_kind = BrainEventRecordKind::StateProjection;
        event.kind = "now.snapshot".to_string();
        event.references.snapshot_id = Some(snapshot_id.clone());
        event.payload = BrainEventPayload::referenced(
            format!("now:{snapshot_id}"),
            format!("snapshot://{snapshot_id}"),
            "application/vnd.netherwick.now+json",
        );
        event.loss_policy = LossPolicy::Coalescible {
            key: "now.current".to_string(),
        };
        event
    }

    /// Converts an estimator-authored calibration decision into its supporting
    /// evidence records followed by the canonical loss-intolerant transition.
    /// No state comparison occurs here: the estimator already decided whether
    /// a real accepted/rejected/trust/epoch transition happened.
    pub fn from_calibration_transition(transition: &CalibrationTransition) -> Vec<BrainEvent> {
        let mut events = transition
            .evidence
            .iter()
            .map(|evidence| {
                let mut event = BrainEvent::historical(
                    BrainEventId(evidence.event_id.clone()),
                    BrainEventType::Evidence,
                    ProducerIdentity::new(Brain::Motherbrain, transition.estimator.clone()),
                    EventTimes {
                        occurred: ClockedTime::in_epoch(
                            evidence.occurred.t_ms,
                            evidence.occurred.clock_epoch.clone(),
                        ),
                        observed: ClockedTime::in_epoch(
                            evidence.observed.t_ms,
                            evidence.observed.clock_epoch.clone(),
                        ),
                        valid_from: None,
                        expires_at: None,
                    },
                );
                event.kind = format!("calibration.evidence.{}", evidence.source);
                event
                    .artifacts
                    .push(calibration_artifact(&evidence.artifact));
                event.payload =
                    compact_payload_or_reference(&event.event_id, evidence.payload.clone());
                event
            })
            .collect::<Vec<_>>();

        let mut event = BrainEvent::historical(
            BrainEventId(transition.id.clone()),
            BrainEventType::CalibrationTransition,
            ProducerIdentity::new(Brain::Motherbrain, transition.estimator.clone()),
            EventTimes {
                occurred: ClockedTime::in_epoch(
                    transition.occurred.t_ms,
                    transition.occurred.clock_epoch.clone(),
                ),
                observed: ClockedTime::in_epoch(
                    transition.observed.t_ms,
                    transition.observed.clock_epoch.clone(),
                ),
                valid_from: Some(ClockedTime::in_epoch(
                    transition.observed.t_ms,
                    transition.observed.clock_epoch.clone(),
                )),
                expires_at: None,
            },
        );
        event.kind = format!(
            "calibration.{}.{}",
            transition.estimator,
            transition_kind_name(transition.kind)
        );
        event
            .calibration_epochs
            .push(format!("{}:{}", transition.estimator, transition.epoch));
        event.artifacts.extend([
            calibration_artifact(&transition.prior_artifact),
            calibration_artifact(&transition.candidate_artifact),
            calibration_artifact(&transition.accepted_artifact),
        ]);
        event
            .links
            .supports
            .extend(transition.evidence.iter().map(|evidence| {
                TypedEventRef::new(
                    BrainEventId(evidence.event_id.clone()),
                    BrainEventType::Evidence,
                )
            }));
        event.quality.confidence = Some(transition.new.confidence.clamp(0.0, 1.0));
        event.quality.trust = match transition.new.trust.as_str() {
            "trusted" => TrustState::Trusted,
            "estimating" | "configured" | "warming_up" | "nominal" => TrustState::Conditional,
            "invalidated" | "degraded" | "unobservable" => TrustState::Untrusted,
            _ => TrustState::Unknown,
        };
        event.disposition = match transition.kind {
            CalibrationTransitionKind::Accepted | CalibrationTransitionKind::NewlyTrusted => {
                EventDisposition::Accepted
            }
            CalibrationTransitionKind::Rejected => EventDisposition::Rejected,
            CalibrationTransitionKind::Invalidated
            | CalibrationTransitionKind::Remounted
            | CalibrationTransitionKind::RolledBack => EventDisposition::Superseded,
            CalibrationTransitionKind::Degraded => EventDisposition::Unavailable,
        };
        event.payload = compact_payload_or_reference(
            &event.event_id,
            serde_json::to_value(transition).unwrap_or(Value::Null),
        );
        event.loss_policy = LossPolicy::LossIntolerant;
        events.push(event);
        events
    }

    pub fn from_reign_input(input: &ReignInput, observed_at_ms: u64) -> Self {
        let mut event = Self::historical(
            BrainEventId::from_domain("reign-input", input.id),
            BrainEventType::Proposal,
            ProducerIdentity::new(Brain::Human, format!("reign.{:?}", input.source)),
            EventTimes {
                occurred: ClockedTime::new(input.issued_at_ms),
                observed: ClockedTime::new(observed_at_ms),
                valid_from: Some(ClockedTime::new(input.issued_at_ms)),
                expires_at: Some(ClockedTime::new(input.expires_at_ms)),
            },
        );
        event.references.command_ids.push(input.id.to_string());
        event.kind = "reign.input".to_string();
        event.payload = compact_payload_or_reference(
            &event.event_id,
            json!({
                "source": input.source,
                "mode": input.mode,
                "command": input.command,
                "priority": input.priority,
                "note": input.note,
            }),
        );
        event.authority = AuthoritySignificance::Proposal;
        event.loss_policy = LossPolicy::LossIntolerant;
        event
    }

    pub fn from_reign_outcome(event_id: BrainEventId, outcome: &ReignOutcome, t_ms: u64) -> Self {
        let mut event = Self::historical(
            event_id,
            BrainEventType::GateDecision,
            ProducerIdentity::new(Brain::Motherbrain, "reign.gate"),
            EventTimes::observed(t_ms, t_ms),
        );
        event.links.parents.push(TypedEventRef::new(
            BrainEventId::from_domain("reign-input", outcome.input_id),
            BrainEventType::Proposal,
        ));
        event.kind = "reign.outcome".to_string();
        event
            .references
            .command_ids
            .push(outcome.input_id.to_string());
        event.disposition = if outcome.vetoed_by_safety {
            EventDisposition::Vetoed
        } else if outcome.accepted_by_conductor {
            EventDisposition::Accepted
        } else {
            EventDisposition::Rejected
        };
        event.payload = compact_payload_or_reference(&event.event_id, json!(outcome));
        event.authority = AuthoritySignificance::Gate;
        event
    }

    pub fn from_safety_decision(
        event_id: BrainEventId,
        decision: &SafetyDecision,
        snapshot_id: impl Into<String>,
        proposal: TypedEventRef,
        t_ms: u64,
    ) -> Self {
        let snapshot_id = snapshot_id.into();
        let mut event = Self::historical(
            event_id,
            BrainEventType::GateDecision,
            ProducerIdentity::new(Brain::Motherbrain, "autonomic.safety"),
            EventTimes::observed(t_ms, t_ms),
        );
        event.references.snapshot_id = Some(snapshot_id);
        event.kind = "safety.decision".to_string();
        event.links.parents.push(proposal);
        event.disposition = if decision.vetoed {
            EventDisposition::Vetoed
        } else {
            EventDisposition::Accepted
        };
        event.payload = compact_payload_or_reference(&event.event_id, json!(decision));
        event.authority = AuthoritySignificance::SafetyTransition;
        event
    }
}

fn calibration_artifact(artifact: &CalibrationArtifactIdentity) -> ArtifactIdentity {
    ArtifactIdentity {
        kind: ArtifactKind::Calibration,
        id: artifact.id.clone(),
        version: None,
        checksum: Some(artifact.checksum.clone()),
    }
}

fn transition_kind_name(kind: CalibrationTransitionKind) -> &'static str {
    match kind {
        CalibrationTransitionKind::Accepted => "accepted",
        CalibrationTransitionKind::Rejected => "rejected",
        CalibrationTransitionKind::Degraded => "degraded",
        CalibrationTransitionKind::Invalidated => "invalidated",
        CalibrationTransitionKind::Remounted => "remounted",
        CalibrationTransitionKind::RolledBack => "rolled_back",
        CalibrationTransitionKind::NewlyTrusted => "newly_trusted",
    }
}

impl From<&Sensation> for BrainEvent {
    fn from(sensation: &Sensation) -> Self {
        let event_id = BrainEventId::sensation(sensation.id);
        let mut event = Self::historical(
            event_id.clone(),
            BrainEventType::Evidence,
            ProducerIdentity::new(Brain::Motherbrain, sensation.source.clone()),
            EventTimes::observed(sensation.occurred_at_ms, sensation.observed_at_ms),
        );
        event.references.frame_id = sensation.metadata.source.frame_id.clone();
        event.kind = sensation.kind.clone();
        event.quality.confidence = sensation.metadata.confidence;
        event.quality.freshness = Freshness {
            state: FreshnessState::Current,
            age_ms: Some(
                sensation
                    .observed_at_ms
                    .saturating_sub(sensation.occurred_at_ms),
            ),
        };
        event.links = links_from_provenance(&sensation.provenance);
        if let Some(parent_id) = sensation.parent_id {
            event.links.parents.push(TypedEventRef::new(
                BrainEventId::sensation(parent_id),
                BrainEventType::Evidence,
            ));
        }
        if is_heavy_sensation_payload(&sensation.payload_kind) {
            event.payload = BrainEventPayload::referenced(
                format!("sensation-payload:{}", sensation.id),
                format!("sensation://{}/payload", sensation.id),
                sensation_media_type(&sensation.payload_kind),
            );
        } else {
            event.payload = compact_payload_or_reference(
                &event_id,
                json!({
                    "kind": sensation.kind,
                    "modality": sensation.modality,
                    "payload_kind": sensation.payload_kind,
                    "summary": sensation.summary,
                    "payload": sensation.payload,
                    "metadata": sensation.metadata,
                }),
            );
        }
        if let Some(vector) = &sensation.vector {
            event.artifacts.push(vector_identity(vector));
        }
        event.coalescible(format!("evidence.{}.{}", sensation.source, sensation.kind))
    }
}

impl From<&Impression> for BrainEvent {
    fn from(impression: &Impression) -> Self {
        let event_id = BrainEventId::impression(impression.id);
        let mut event = Self::historical(
            event_id.clone(),
            BrainEventType::Interpretation,
            ProducerIdentity::new(Brain::Motherbrain, "experience.impression"),
            EventTimes::observed(impression.occurred_at_ms, impression.observed_at_ms),
        );
        event.quality.confidence = Some(impression.confidence);
        event.kind = impression.kind.clone();
        event.links.supports = impression
            .about
            .iter()
            .map(|id| TypedEventRef::new(BrainEventId::sensation(*id), BrainEventType::Evidence))
            .collect();
        if let Some(sensation_id) = impression.sensation_id {
            event.links.parents.push(TypedEventRef::new(
                BrainEventId::sensation(sensation_id),
                BrainEventType::Evidence,
            ));
        }
        event.references.experience_id = impression.experience_id.map(|id| id.to_string());
        event.payload = compact_payload_or_reference(
            &event_id,
            json!({
                "kind": impression.kind,
                "text": impression.text,
                "generator": impression.generator,
                "payload": impression.payload,
            }),
        );
        if let Some(vector) = &impression.vector {
            event.artifacts.push(vector_identity(vector));
        }
        event.coalescible(format!("interpretation.{}", impression.kind))
    }
}

impl From<&Experience> for BrainEvent {
    fn from(experience: &Experience) -> Self {
        let event_id = BrainEventId::experience(experience.id);
        let mut event = Self::historical(
            event_id.clone(),
            BrainEventType::BeliefUpdate,
            ProducerIdentity::new(Brain::Motherbrain, "experience.fuser"),
            EventTimes::observed(experience.occurred_at_ms, experience.observed_at_ms),
        );
        event.references.experience_id = Some(experience.id.to_string());
        event.kind = experience.kind.clone();
        event.links.parents = experience
            .impression_ids
            .iter()
            .map(|id| {
                TypedEventRef::new(
                    BrainEventId::impression(*id),
                    BrainEventType::Interpretation,
                )
            })
            .collect();
        event.links.supports = experience
            .sensation_ids
            .iter()
            .map(|id| TypedEventRef::new(BrainEventId::sensation(*id), BrainEventType::Evidence))
            .collect();
        event.quality.confidence = Some(experience.salience);
        event.payload = compact_payload_or_reference(
            &event_id,
            json!({
                "kind": experience.kind,
                "text": experience.text,
                "window_start_ms": experience.window_start_ms,
                "window_end_ms": experience.window_end_ms,
                "tags": experience.tags,
                "payload": experience.payload,
            }),
        );
        event.coalescible(format!("belief.{}", experience.kind))
    }
}

impl From<&Event> for BrainEvent {
    fn from(source: &Event) -> Self {
        let (event_type, authority, loss_policy) = legacy_event_semantics(source.kind);
        let component = source
            .provenance
            .stage_chain
            .last()
            .cloned()
            .unwrap_or_else(|| "event".to_string());
        let mut event = Self::historical(
            BrainEventId::from_domain("event", source.id),
            event_type,
            ProducerIdentity::new(Brain::Motherbrain, component),
            EventTimes::observed(source.t_ms, source.t_ms),
        );
        event.links = links_from_provenance(&source.provenance);
        event.kind = format!("legacy.{:?}", source.kind).to_lowercase();
        event.payload = compact_payload_or_reference(
            &event.event_id,
            json!({
                "kind": source.kind,
                "summary": source.summary,
                "payload": source.payload,
            }),
        );
        event.authority = authority;
        event.loss_policy = loss_policy;
        event
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeduplicationOutcome {
    Inserted,
    Duplicate,
}

#[derive(Clone, Debug, Default)]
pub struct BrainEventIndex {
    events: BTreeMap<BrainEventId, BrainEvent>,
}

impl BrainEventIndex {
    pub fn insert(&mut self, event: BrainEvent) -> Result<DeduplicationOutcome, BrainEventError> {
        event.validate()?;
        match self.events.get(&event.event_id) {
            Some(existing) if existing == &event => Ok(DeduplicationOutcome::Duplicate),
            Some(_) => Err(BrainEventError::ConflictingDuplicate(event.event_id)),
            None => {
                self.events.insert(event.event_id.clone(), event);
                Ok(DeduplicationOutcome::Inserted)
            }
        }
    }

    pub fn get(&self, event_id: &BrainEventId) -> Option<&BrainEvent> {
        self.events.get(event_id)
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

#[derive(Debug, Error)]
pub enum BrainEventError {
    #[error("unsupported BrainEvent schema version {0}")]
    UnsupportedSchema(u32),
    #[error("invalid BrainEvent: {0}")]
    Invalid(String),
    #[error("inline BrainEvent payload is {0} bytes; use a payload reference")]
    InlinePayloadTooLarge(usize),
    #[error("a loss-intolerant BrainEvent class cannot be coalesced")]
    LossIntolerantClassCoalesced,
    #[error("event id {0} was reused with different content")]
    ConflictingDuplicate(BrainEventId),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Deserialize)]
struct LegacyBrainEventV0 {
    #[serde(alias = "event_id")]
    id: String,
    #[serde(alias = "event_type")]
    kind: BrainEventType,
    component: String,
    occurred_at_ms: u64,
    observed_at_ms: u64,
    #[serde(default)]
    payload: Value,
}

impl LegacyBrainEventV0 {
    fn from_value(value: Value) -> Result<Self, BrainEventError> {
        Ok(serde_json::from_value(value)?)
    }

    fn migrate(self) -> BrainEvent {
        let mut event = BrainEvent::historical(
            BrainEventId(self.id),
            self.kind,
            ProducerIdentity::new(Brain::Unknown, self.component),
            EventTimes::observed(self.occurred_at_ms, self.observed_at_ms),
        );
        event.payload = compact_payload_or_reference(&event.event_id, self.payload);
        if event.event_type == BrainEventType::CalibrationTransition {
            event.loss_policy = LossPolicy::LossIntolerant;
        }
        event
    }
}

fn validate_unit_interval(name: &str, value: Option<f32>) -> Result<(), BrainEventError> {
    if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        return Err(BrainEventError::Invalid(format!(
            "{name} must be finite and between 0 and 1"
        )));
    }
    Ok(())
}

fn compact_payload_or_reference(event_id: &BrainEventId, data: Value) -> BrainEventPayload {
    match serde_json::to_vec(&data) {
        Ok(bytes) if bytes.len() <= MAX_INLINE_BRAIN_EVENT_PAYLOAD_BYTES => {
            BrainEventPayload::inline(data)
        }
        _ => BrainEventPayload::referenced(
            format!("payload:{event_id}"),
            format!("event://{event_id}/payload"),
            "application/json",
        ),
    }
}

fn links_from_provenance(provenance: &Provenance) -> CausalLinks {
    let supports = match &provenance.kind {
        ProvenanceKind::Direct => Vec::new(),
        ProvenanceKind::DerivedFromSensations { sensation_ids } => sensation_ids
            .iter()
            .map(|id| TypedEventRef::new(BrainEventId::sensation(*id), BrainEventType::Evidence))
            .collect(),
        ProvenanceKind::DerivedFromImpressions { impression_ids } => impression_ids
            .iter()
            .map(|id| {
                TypedEventRef::new(
                    BrainEventId::impression(*id),
                    BrainEventType::Interpretation,
                )
            })
            .collect(),
        ProvenanceKind::MemoryRecall { experience_id } => vec![TypedEventRef::new(
            BrainEventId::experience(*experience_id),
            BrainEventType::BeliefUpdate,
        )],
    };
    CausalLinks {
        supports,
        ..CausalLinks::default()
    }
}

fn vector_identity(vector: &VectorEmbedding) -> ArtifactIdentity {
    ArtifactIdentity {
        kind: ArtifactKind::Vector,
        id: format!("{}:{}", vector.collection, vector.source_sensation_id),
        version: Some(vector.model_id.clone()),
        checksum: None,
    }
}

fn is_heavy_sensation_payload(kind: &SensationPayloadKind) -> bool {
    matches!(
        kind,
        SensationPayloadKind::ImageBytes
            | SensationPayloadKind::AudioPcm
            | SensationPayloadKind::VoiceSegment
            | SensationPayloadKind::DepthFrame
            | SensationPayloadKind::PointCloud
            | SensationPayloadKind::LidarScan
            | SensationPayloadKind::Crop
    )
}

fn sensation_media_type(kind: &SensationPayloadKind) -> &'static str {
    match kind {
        SensationPayloadKind::ImageBytes | SensationPayloadKind::Crop => {
            "application/vnd.netherwick.image"
        }
        SensationPayloadKind::AudioPcm | SensationPayloadKind::VoiceSegment => "audio/L16",
        SensationPayloadKind::DepthFrame => "application/vnd.netherwick.depth",
        SensationPayloadKind::PointCloud => "application/vnd.netherwick.point-cloud",
        SensationPayloadKind::LidarScan => "application/vnd.netherwick.lidar-scan",
        _ => "application/json",
    }
}

fn legacy_event_semantics(kind: EventKind) -> (BrainEventType, AuthoritySignificance, LossPolicy) {
    match kind {
        EventKind::SafetyVetoed => (
            BrainEventType::GateDecision,
            AuthoritySignificance::SafetyTransition,
            LossPolicy::LossIntolerant,
        ),
        EventKind::ReignCommanded => (
            BrainEventType::Proposal,
            AuthoritySignificance::Proposal,
            LossPolicy::LossIntolerant,
        ),
        EventKind::ReignExpired | EventKind::ReignCleared => (
            BrainEventType::GateDecision,
            AuthoritySignificance::AuthorityTransition,
            LossPolicy::LossIntolerant,
        ),
        EventKind::LlmCommanded => (
            BrainEventType::Proposal,
            AuthoritySignificance::Advisory,
            LossPolicy::Coalescible {
                key: "llm.advisory".to_string(),
            },
        ),
        EventKind::LlmCritiqued
        | EventKind::PredictionMade
        | EventKind::MemoryRecalled
        | EventKind::SurpriseHigh => (
            BrainEventType::Interpretation,
            AuthoritySignificance::None,
            LossPolicy::Coalescible {
                key: format!("interpretation.{kind:?}"),
            },
        ),
        EventKind::SimDeadBatteryReset | EventKind::SimStuckReset => (
            BrainEventType::Outcome,
            AuthoritySignificance::Outcome,
            LossPolicy::LossIntolerant,
        ),
        _ => (
            BrainEventType::Evidence,
            AuthoritySignificance::None,
            LossPolicy::Coalescible {
                key: format!("evidence.{kind:?}"),
            },
        ),
    }
}
