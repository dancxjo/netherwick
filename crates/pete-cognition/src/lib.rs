//! Role-neutral contracts for optional cognitive services.
//!
//! These types deliberately contain no body command or cockpit authority.
//! Providers return evidence, predictions, suggestions, or artifacts to the
//! organism runtime; they cannot mutate `Now` or acquire motor control.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::task::JoinHandle;
use uuid::Uuid;

macro_rules! string_id {
    ($name:ident) => {
        #[derive(
            Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub String);
    };
}

string_id!(ProviderId);
string_id!(HostId);
string_id!(ProcessId);
string_id!(RequestId);
string_id!(CancellationTokenId);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveRole {
    BodyController,
    #[default]
    OrganismRuntime,
    CognitiveAccelerator,
    RemoteAdvisor,
}

impl CognitiveRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BodyController => "body_controller",
            Self::OrganismRuntime => "organism_runtime",
            Self::CognitiveAccelerator => "cognitive_accelerator",
            Self::RemoteAdvisor => "remote_advisor",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveCapability {
    DescribeScene,
    RecognizeEntity,
    InterpretSpeech,
    GenerateSpeech,
    PredictOutcome,
    ProposePlan,
    ReviewFailure,
    SuggestAlternativeSkill,
    TrainModel,
    ConsolidateMemory,
    RunCounterfactual,
    #[default]
    Unknown,
}

impl CognitiveCapability {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DescribeScene => "describe_scene",
            Self::RecognizeEntity => "recognize_entity",
            Self::InterpretSpeech => "interpret_speech",
            Self::GenerateSpeech => "generate_speech",
            Self::PredictOutcome => "predict_outcome",
            Self::ProposePlan => "propose_plan",
            Self::ReviewFailure => "review_failure",
            Self::SuggestAlternativeSkill => "suggest_alternative_skill",
            Self::TrainModel => "train_model",
            Self::ConsolidateMemory => "consolidate_memory",
            Self::RunCounterfactual => "run_counterfactual",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Locality {
    #[default]
    OnOrganism,
    LocalNetwork,
    Remote,
}

impl Locality {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OnOrganism => "on_organism",
            Self::LocalNetwork => "local_network",
            Self::Remote => "remote",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceClass {
    Embedded,
    GeneralPurpose,
    Accelerated,
    Cloud,
    #[default]
    Unknown,
}

impl ResourceClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Embedded => "embedded",
            Self::GeneralPurpose => "general_purpose",
            Self::Accelerated => "accelerated",
            Self::Cloud => "cloud",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustPolicy {
    LocalDeterministic,
    TrustedProvider,
    AdvisoryOnly,
    #[default]
    Untrusted,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderHealthState {
    Available,
    Degraded,
    Busy,
    Disconnected,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderHealth {
    pub state: ProviderHealthState,
    pub confidence: f32,
    pub observed_at_ms: u64,
    pub valid_until_ms: u64,
    pub reason: Option<String>,
}

impl ProviderHealth {
    pub fn usable_at(&self, now_ms: u64) -> bool {
        matches!(
            self.state,
            ProviderHealthState::Available | ProviderHealthState::Degraded
        ) && now_ms <= self.valid_until_ms
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LatencyEstimate {
    pub expected_ms: u64,
    pub p95_ms: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CapabilityDescriptor {
    pub capability: CognitiveCapability,
    pub version: String,
    pub supports_partial: bool,
    pub performance_confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CognitiveProviderDescriptor {
    pub provider_id: ProviderId,
    pub role: CognitiveRole,
    pub host_id: Option<HostId>,
    pub process_id: Option<ProcessId>,
    pub implementation: String,
    pub implementation_version: String,
    pub model_version: Option<String>,
    #[serde(default)]
    pub capabilities: Vec<CapabilityDescriptor>,
    pub health: ProviderHealth,
    pub latency: LatencyEstimate,
    pub resource_class: ResourceClass,
    pub locality: Locality,
    pub trust: TrustPolicy,
    pub energy_cost: f32,
    pub network_cost: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderRegistrySnapshot {
    pub schema_version: u32,
    pub revision: u64,
    pub observed_at_ms: u64,
    #[serde(default)]
    pub providers: BTreeMap<ProviderId, CognitiveProviderDescriptor>,
}

#[derive(Clone, Debug, Default)]
pub struct CognitiveProviderRegistry {
    revision: u64,
    providers: BTreeMap<ProviderId, CognitiveProviderDescriptor>,
}

impl CognitiveProviderRegistry {
    pub fn register(&mut self, descriptor: CognitiveProviderDescriptor) {
        self.providers
            .insert(descriptor.provider_id.clone(), descriptor);
        self.revision = self.revision.saturating_add(1);
    }

    pub fn remove(&mut self, provider_id: &ProviderId) {
        if self.providers.remove(provider_id).is_some() {
            self.revision = self.revision.saturating_add(1);
        }
    }

    pub fn update_health(&mut self, provider_id: &ProviderId, health: ProviderHealth) {
        if let Some(provider) = self.providers.get_mut(provider_id) {
            provider.health = health;
            self.revision = self.revision.saturating_add(1);
        }
    }

    pub fn get(&self, provider_id: &ProviderId) -> Option<&CognitiveProviderDescriptor> {
        self.providers.get(provider_id)
    }

    pub fn snapshot(&self, now_ms: u64) -> ProviderRegistrySnapshot {
        ProviderRegistrySnapshot {
            schema_version: 1,
            revision: self.revision,
            observed_at_ms: now_ms,
            providers: self.providers.clone(),
        }
    }

    pub fn route(&self, request: &CognitiveRequest, now_ms: u64) -> RouteDecision {
        let remaining_ms = request.deadline_ms.saturating_sub(now_ms);
        let mut eligible = self
            .providers
            .values()
            .filter_map(|provider| {
                let capability = provider.capabilities.iter().find(|candidate| {
                    candidate.capability == request.requirement.capability
                        && request
                            .requirement
                            .version
                            .as_ref()
                            .map(|required| required == &candidate.version)
                            .unwrap_or(true)
                        && (!request.allow_partial || candidate.supports_partial)
                })?;
                (provider.health.usable_at(now_ms)
                    && provider.locality <= request.privacy.maximum_locality
                    && provider.latency.expected_ms <= remaining_ms)
                    .then_some((provider, capability))
            })
            .collect::<Vec<_>>();
        eligible.sort_by(|(left, left_cap), (right, right_cap)| {
            left.locality
                .cmp(&right.locality)
                .then_with(|| health_rank(left.health.state).cmp(&health_rank(right.health.state)))
                .then_with(|| left.latency.expected_ms.cmp(&right.latency.expected_ms))
                .then_with(|| {
                    right_cap
                        .performance_confidence
                        .total_cmp(&left_cap.performance_confidence)
                })
                .then_with(|| left.provider_id.cmp(&right.provider_id))
        });
        match eligible.first() {
            Some((provider, _)) => RouteDecision {
                provider_id: Some(provider.provider_id.clone()),
                reason: "selected deterministic best eligible provider".to_string(),
            },
            None => RouteDecision {
                provider_id: None,
                reason: "no healthy provider satisfies capability, deadline, locality, and partial-response policy".to_string(),
            },
        }
    }
}

fn health_rank(state: ProviderHealthState) -> u8 {
    match state {
        ProviderHealthState::Available => 0,
        ProviderHealthState::Degraded => 1,
        ProviderHealthState::Busy => 2,
        ProviderHealthState::Disconnected => 3,
        ProviderHealthState::Unknown => 4,
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityRequirement {
    pub capability: CognitiveCapability,
    pub version: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotRef {
    pub snapshot_id: String,
    pub schema_version: u32,
    pub revision: u64,
    pub captured_at_ms: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundedInputRef {
    pub id: String,
    pub kind: String,
    pub byte_len: usize,
    pub content_hash: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundedImageInput {
    pub reference: BoundedInputRef,
    pub content_type: String,
    pub width: u32,
    pub height: u32,
    pub captured_at_ms: u64,
    #[serde(with = "serde_bytes_compat")]
    pub bytes: Vec<u8>,
}

mod serde_bytes_compat {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        bytes.serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<u8>, D::Error> {
        Vec::<u8>::deserialize(deserializer)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BoundedTaskInput {
    #[serde(default)]
    pub references: Vec<BoundedInputRef>,
    #[serde(default)]
    pub facts: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "input", rename_all = "snake_case")]
pub enum CognitiveRequestPayload {
    DescribeScene(BoundedImageInput),
    RecognizeEntity(BoundedTaskInput),
    InterpretSpeech(BoundedTaskInput),
    GenerateSpeech(BoundedTaskInput),
    PredictOutcome(BoundedTaskInput),
    ProposePlan(BoundedTaskInput),
    ReviewFailure(BoundedTaskInput),
    SuggestAlternativeSkill(BoundedTaskInput),
    TrainModel(BoundedTaskInput),
    ConsolidateMemory(BoundedTaskInput),
    RunCounterfactual(BoundedTaskInput),
}

impl CognitiveRequestPayload {
    pub fn capability(&self) -> CognitiveCapability {
        match self {
            Self::DescribeScene(_) => CognitiveCapability::DescribeScene,
            Self::RecognizeEntity(_) => CognitiveCapability::RecognizeEntity,
            Self::InterpretSpeech(_) => CognitiveCapability::InterpretSpeech,
            Self::GenerateSpeech(_) => CognitiveCapability::GenerateSpeech,
            Self::PredictOutcome(_) => CognitiveCapability::PredictOutcome,
            Self::ProposePlan(_) => CognitiveCapability::ProposePlan,
            Self::ReviewFailure(_) => CognitiveCapability::ReviewFailure,
            Self::SuggestAlternativeSkill(_) => CognitiveCapability::SuggestAlternativeSkill,
            Self::TrainModel(_) => CognitiveCapability::TrainModel,
            Self::ConsolidateMemory(_) => CognitiveCapability::ConsolidateMemory,
            Self::RunCounterfactual(_) => CognitiveCapability::RunCounterfactual,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallerRole {
    #[default]
    OrganismRuntime,
    Goal,
    Skill,
    Memory,
    Operator,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestProvenance {
    pub caller: CallerRole,
    pub caller_id: String,
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrivacyPolicy {
    pub maximum_locality: Locality,
    pub allow_raw_image: bool,
    pub allow_persistence: bool,
}

impl Default for PrivacyPolicy {
    fn default() -> Self {
        Self {
            maximum_locality: Locality::OnOrganism,
            allow_raw_image: false,
            allow_persistence: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CognitiveRequest {
    pub schema_version: u32,
    pub request_id: RequestId,
    pub requirement: CapabilityRequirement,
    pub input_snapshot: SnapshotRef,
    pub input_refs: Vec<BoundedInputRef>,
    pub created_at_ms: u64,
    pub deadline_ms: u64,
    pub privacy: PrivacyPolicy,
    pub allow_partial: bool,
    pub cancellation_token: CancellationTokenId,
    pub provenance: RequestProvenance,
    pub payload: CognitiveRequestPayload,
}

impl CognitiveRequest {
    pub fn new(
        input_snapshot: SnapshotRef,
        created_at_ms: u64,
        deadline_ms: u64,
        privacy: PrivacyPolicy,
        provenance: RequestProvenance,
        payload: CognitiveRequestPayload,
    ) -> Self {
        let requirement = CapabilityRequirement {
            capability: payload.capability(),
            version: None,
        };
        Self {
            schema_version: 1,
            request_id: RequestId(Uuid::new_v4().to_string()),
            requirement,
            input_snapshot,
            input_refs: Vec::new(),
            created_at_ms,
            deadline_ms,
            privacy,
            allow_partial: false,
            cancellation_token: CancellationTokenId(Uuid::new_v4().to_string()),
            provenance,
            payload,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.requirement.capability != self.payload.capability() {
            return Err(anyhow!("request capability does not match typed payload"));
        }
        if self.deadline_ms <= self.created_at_ms {
            return Err(anyhow!("request deadline must be after creation"));
        }
        if let CognitiveRequestPayload::DescribeScene(image) = &self.payload {
            if image.bytes.len() != image.reference.byte_len {
                return Err(anyhow!(
                    "bounded image byte length does not match reference"
                ));
            }
            if !self.privacy.allow_raw_image {
                return Err(anyhow!("privacy policy forbids attached image bytes"));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveResponseStatus {
    Completed,
    Partial,
    Unavailable,
    TimedOut,
    Cancelled,
    Failed,
    Stale,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "output", rename_all = "snake_case")]
pub enum CognitiveResponsePayload {
    SceneDescription { text: String, embedding: Vec<f32> },
    Evidence(BoundedTaskInput),
    Suggestion(BoundedTaskInput),
    ModelArtifact(BoundedTaskInput),
    None,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceCost {
    pub elapsed_ms: u64,
    pub energy_estimate: f32,
    pub network_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CognitiveResponse {
    pub schema_version: u32,
    pub request_id: RequestId,
    pub provider_id: ProviderId,
    pub provider_role: CognitiveRole,
    pub implementation: String,
    pub implementation_version: String,
    pub model_version: Option<String>,
    pub status: CognitiveResponseStatus,
    pub confidence: f32,
    pub uncertainty: f32,
    pub input_snapshot: SnapshotRef,
    pub completed_at_ms: u64,
    pub resource_cost: ResourceCost,
    pub provenance: Vec<String>,
    pub payload: CognitiveResponsePayload,
    pub failure: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteDecision {
    pub provider_id: Option<ProviderId>,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseDisposition {
    Accepted,
    Stale,
    Rejected,
    #[default]
    Failed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoutedResponse {
    pub route: RouteDecision,
    pub disposition: ResponseDisposition,
    pub response: CognitiveResponse,
}

pub fn validate_response(
    request: &CognitiveRequest,
    response: &CognitiveResponse,
) -> ResponseDisposition {
    if response.request_id != request.request_id
        || response.input_snapshot != request.input_snapshot
    {
        return ResponseDisposition::Rejected;
    }
    if response.completed_at_ms > request.deadline_ms
        || response.status == CognitiveResponseStatus::Stale
    {
        return ResponseDisposition::Stale;
    }
    if matches!(
        response.status,
        CognitiveResponseStatus::Completed | CognitiveResponseStatus::Partial
    ) {
        ResponseDisposition::Accepted
    } else {
        ResponseDisposition::Failed
    }
}

#[async_trait]
pub trait CognitiveProvider: Send {
    fn descriptor(&self) -> CognitiveProviderDescriptor;
    async fn execute(&mut self, request: &CognitiveRequest) -> Result<CognitiveResponse>;
}

#[derive(Default)]
pub struct CognitiveRouter {
    registry: CognitiveProviderRegistry,
    providers: BTreeMap<ProviderId, Box<dyn CognitiveProvider>>,
    cancelled: BTreeSet<CancellationTokenId>,
}

/// Runtime-owned, single-flight boundary for optional cognitive I/O.
///
/// `submit` only moves an immutable, bounded request into a background task;
/// [`poll`](Self::poll) never waits for provider I/O. A response is useful only
/// when it still refers to the caller's current snapshot and arrives before its
/// deadline. This deliberately keeps providers outside the control-loop
/// authority boundary: their only output is evidence or advice.
pub struct AsyncCognitionSupervisor {
    router: Option<CognitiveRouter>,
    pending: Option<PendingCognition>,
    cancelled: BTreeSet<CancellationTokenId>,
    last_registry: ProviderRegistrySnapshot,
}

struct PendingCognition {
    request: CognitiveRequest,
    task: JoinHandle<(CognitiveRouter, RoutedResponse)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmissionDisposition {
    Submitted,
    Busy,
    Invalid,
}

impl AsyncCognitionSupervisor {
    pub fn new(router: CognitiveRouter, now_ms: u64) -> Self {
        let last_registry = router.registry_snapshot(now_ms);
        Self {
            router: Some(router),
            pending: None,
            cancelled: BTreeSet::new(),
            last_registry,
        }
    }

    pub fn registry_snapshot(&self) -> &ProviderRegistrySnapshot {
        &self.last_registry
    }

    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Enqueue one bounded request without performing provider I/O inline.
    pub fn submit(&mut self, request: CognitiveRequest, now_ms: u64) -> SubmissionDisposition {
        if request.validate().is_err() || request.deadline_ms <= now_ms {
            return SubmissionDisposition::Invalid;
        }
        if self.pending.is_some() {
            return SubmissionDisposition::Busy;
        }
        let Some(mut router) = self.router.take() else {
            return SubmissionDisposition::Busy;
        };
        self.last_registry = router.registry_snapshot(now_ms);
        let dispatched = request.clone();
        let task = tokio::spawn(async move {
            let response = router.dispatch(dispatched, now_ms).await;
            (router, response)
        });
        self.pending = Some(PendingCognition { request, task });
        SubmissionDisposition::Submitted
    }

    /// Mark a request as cancelled. Its provider task may finish normally, but
    /// its result can no longer cross back into the organism model.
    pub fn cancel(&mut self, token: CancellationTokenId) {
        self.cancelled.insert(token);
    }

    /// Poll once for a finished response, returning immediately when I/O is
    /// still pending. `current_snapshot` must come from the current immutable
    /// `Now` view, not from provider-controlled state.
    pub async fn poll(
        &mut self,
        current_snapshot: &SnapshotRef,
        now_ms: u64,
    ) -> Option<RoutedResponse> {
        if !self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.task.is_finished())
        {
            return None;
        }
        let pending = self.pending.take().expect("finished task exists");
        let cancelled = self.cancelled.remove(&pending.request.cancellation_token);
        let (router, mut routed) = match pending.task.await {
            Ok(result) => result,
            Err(_) => {
                // A panicking provider task cannot safely return its provider
                // registry. Keep a truthful empty/degraded boundary rather than
                // ever blocking the caller while attempting recovery.
                self.router = Some(CognitiveRouter::default());
                self.last_registry = ProviderRegistrySnapshot {
                    schema_version: 1,
                    observed_at_ms: now_ms,
                    ..ProviderRegistrySnapshot::default()
                };
                return None;
            }
        };
        self.last_registry = router.registry_snapshot(now_ms);
        self.router = Some(router);

        if cancelled {
            routed.disposition = ResponseDisposition::Rejected;
            routed.response.status = CognitiveResponseStatus::Cancelled;
            routed.response.failure = Some("request was cancelled".to_string());
        } else if now_ms > pending.request.deadline_ms {
            routed.disposition = ResponseDisposition::Stale;
            routed.response.status = CognitiveResponseStatus::Stale;
            routed.response.failure = Some("response expired before polling".to_string());
        } else if routed.response.input_snapshot != *current_snapshot {
            routed.disposition = ResponseDisposition::Stale;
            routed.response.status = CognitiveResponseStatus::Stale;
            routed.response.failure = Some("response belongs to an obsolete snapshot".to_string());
        }
        Some(routed)
    }
}

impl CognitiveRouter {
    pub fn register(&mut self, provider: Box<dyn CognitiveProvider>) {
        let descriptor = provider.descriptor();
        self.registry.register(descriptor.clone());
        self.providers.insert(descriptor.provider_id, provider);
    }

    pub fn update_health(&mut self, provider_id: &ProviderId, health: ProviderHealth) {
        self.registry.update_health(provider_id, health);
    }

    pub fn cancel(&mut self, token: CancellationTokenId) {
        self.cancelled.insert(token);
    }

    pub fn registry_snapshot(&self, now_ms: u64) -> ProviderRegistrySnapshot {
        self.registry.snapshot(now_ms)
    }

    pub async fn dispatch(&mut self, request: CognitiveRequest, now_ms: u64) -> RoutedResponse {
        let route = self.registry.route(&request, now_ms);
        if let Err(error) = request.validate() {
            return failed_routed_response(
                route,
                &request,
                ProviderId::default(),
                now_ms,
                CognitiveResponseStatus::Failed,
                error.to_string(),
            );
        }
        if self.cancelled.remove(&request.cancellation_token) {
            return failed_routed_response(
                route,
                &request,
                ProviderId::default(),
                now_ms,
                CognitiveResponseStatus::Cancelled,
                "request was cancelled".to_string(),
            );
        }
        let Some(provider_id) = route.provider_id.clone() else {
            return failed_routed_response(
                route,
                &request,
                ProviderId::default(),
                now_ms,
                CognitiveResponseStatus::Unavailable,
                "no eligible cognitive provider".to_string(),
            );
        };
        let remaining_ms = request.deadline_ms.saturating_sub(now_ms);
        let Some(provider) = self.providers.get_mut(&provider_id) else {
            return failed_routed_response(
                route,
                &request,
                provider_id,
                now_ms,
                CognitiveResponseStatus::Unavailable,
                "selected provider implementation is not registered".to_string(),
            );
        };
        match tokio::time::timeout(
            Duration::from_millis(remaining_ms),
            provider.execute(&request),
        )
        .await
        {
            Ok(Ok(response)) => {
                self.registry.update_health(
                    &provider_id,
                    ProviderHealth {
                        state: ProviderHealthState::Available,
                        confidence: response.confidence,
                        observed_at_ms: response.completed_at_ms,
                        valid_until_ms: request.deadline_ms,
                        reason: None,
                    },
                );
                RoutedResponse {
                    disposition: validate_response(&request, &response),
                    route,
                    response,
                }
            }
            Ok(Err(error)) => {
                self.registry.update_health(
                    &provider_id,
                    ProviderHealth {
                        state: ProviderHealthState::Degraded,
                        confidence: 1.0,
                        observed_at_ms: now_ms,
                        valid_until_ms: request.deadline_ms,
                        reason: Some(error.to_string()),
                    },
                );
                failed_routed_response(
                    route,
                    &request,
                    provider_id,
                    now_ms,
                    CognitiveResponseStatus::Failed,
                    error.to_string(),
                )
            }
            Err(_) => {
                self.registry.update_health(
                    &provider_id,
                    ProviderHealth {
                        state: ProviderHealthState::Degraded,
                        confidence: 1.0,
                        observed_at_ms: request.deadline_ms,
                        valid_until_ms: request.deadline_ms,
                        reason: Some("provider exceeded request deadline".to_string()),
                    },
                );
                failed_routed_response(
                    route,
                    &request,
                    provider_id,
                    request.deadline_ms,
                    CognitiveResponseStatus::TimedOut,
                    "provider exceeded request deadline".to_string(),
                )
            }
        }
    }
}

fn failed_routed_response(
    route: RouteDecision,
    request: &CognitiveRequest,
    provider_id: ProviderId,
    completed_at_ms: u64,
    status: CognitiveResponseStatus,
    failure: String,
) -> RoutedResponse {
    let response = CognitiveResponse {
        schema_version: 1,
        request_id: request.request_id.clone(),
        provider_id,
        provider_role: CognitiveRole::RemoteAdvisor,
        implementation: "none".to_string(),
        implementation_version: "0".to_string(),
        model_version: None,
        status,
        confidence: 0.0,
        uncertainty: 1.0,
        input_snapshot: request.input_snapshot.clone(),
        completed_at_ms,
        resource_cost: ResourceCost::default(),
        provenance: Vec::new(),
        payload: CognitiveResponsePayload::None,
        failure: Some(failure),
    };
    RoutedResponse {
        disposition: validate_response(request, &response),
        route,
        response,
    }
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
