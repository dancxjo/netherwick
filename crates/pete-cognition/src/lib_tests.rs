use super::*;
use std::sync::Arc;
use tokio::sync::Notify;

fn provider(id: &str, locality: Locality, latency: u64) -> CognitiveProviderDescriptor {
    CognitiveProviderDescriptor {
        provider_id: ProviderId(id.to_string()),
        role: CognitiveRole::CognitiveAccelerator,
        implementation: "fixture".to_string(),
        implementation_version: "1".to_string(),
        capabilities: vec![CapabilityDescriptor {
            capability: CognitiveCapability::DescribeScene,
            version: "1".to_string(),
            performance_confidence: 0.8,
            ..CapabilityDescriptor::default()
        }],
        health: ProviderHealth {
            state: ProviderHealthState::Available,
            confidence: 1.0,
            observed_at_ms: 0,
            valid_until_ms: 10_000,
            ..ProviderHealth::default()
        },
        latency: LatencyEstimate {
            expected_ms: latency,
            p95_ms: latency,
        },
        locality,
        trust: TrustPolicy::TrustedProvider,
        ..CognitiveProviderDescriptor::default()
    }
}

fn request() -> CognitiveRequest {
    let bytes = vec![1, 2, 3];
    CognitiveRequest::new(
        SnapshotRef {
            snapshot_id: "now:7".to_string(),
            schema_version: 1,
            revision: 7,
            captured_at_ms: 10,
        },
        10,
        1_010,
        PrivacyPolicy {
            maximum_locality: Locality::LocalNetwork,
            allow_raw_image: true,
            allow_persistence: false,
        },
        RequestProvenance {
            caller: CallerRole::OrganismRuntime,
            caller_id: "vision".to_string(),
            evidence_refs: vec!["frame:7".to_string()],
        },
        CognitiveRequestPayload::DescribeScene(BoundedImageInput {
            reference: BoundedInputRef {
                id: "frame:7".to_string(),
                kind: "image".to_string(),
                byte_len: bytes.len(),
                content_hash: None,
            },
            content_type: "image/png".to_string(),
            width: 1,
            height: 1,
            captured_at_ms: 10,
            bytes,
        }),
    )
}

#[test]
fn two_providers_route_deterministically_by_policy() {
    let mut registry = CognitiveProviderRegistry::default();
    registry.register(provider("remote-fast", Locality::Remote, 10));
    registry.register(provider("local-b", Locality::LocalNetwork, 50));
    registry.register(provider("local-a", Locality::LocalNetwork, 50));
    assert_eq!(
        registry.route(&request(), 10).provider_id,
        Some(ProviderId("local-a".to_string()))
    );
}

#[test]
fn disconnected_or_too_slow_provider_is_not_selected() {
    let mut registry = CognitiveProviderRegistry::default();
    let mut disconnected = provider("gone", Locality::OnOrganism, 1);
    disconnected.health.state = ProviderHealthState::Disconnected;
    registry.register(disconnected);
    registry.register(provider("too-slow", Locality::LocalNetwork, 5_000));
    assert!(registry.route(&request(), 10).provider_id.is_none());
}

#[test]
fn late_or_wrong_snapshot_response_is_not_accepted() {
    let request = request();
    let mut response = CognitiveResponse {
        schema_version: 1,
        request_id: request.request_id.clone(),
        provider_id: ProviderId("fixture".to_string()),
        provider_role: CognitiveRole::CognitiveAccelerator,
        implementation: "fixture".to_string(),
        implementation_version: "1".to_string(),
        model_version: None,
        status: CognitiveResponseStatus::Completed,
        confidence: 0.8,
        uncertainty: 0.2,
        input_snapshot: request.input_snapshot.clone(),
        completed_at_ms: request.deadline_ms + 1,
        resource_cost: ResourceCost::default(),
        provenance: Vec::new(),
        payload: CognitiveResponsePayload::None,
        failure: None,
    };
    assert_eq!(
        validate_response(&request, &response),
        ResponseDisposition::Stale
    );
    response.completed_at_ms = request.deadline_ms;
    response.input_snapshot.revision += 1;
    assert_eq!(
        validate_response(&request, &response),
        ResponseDisposition::Rejected
    );
}

#[test]
fn request_contract_has_no_action_authority_surface() {
    let serialized = serde_json::to_string(&request()).unwrap();
    assert!(!serialized.contains("motor"));
    assert!(!serialized.contains("authority"));
    assert!(!serialized.contains("wheel"));
}

struct StalledProvider {
    descriptor: CognitiveProviderDescriptor,
    release: Arc<Notify>,
}

#[async_trait]
impl CognitiveProvider for StalledProvider {
    fn descriptor(&self) -> CognitiveProviderDescriptor {
        self.descriptor.clone()
    }

    async fn execute(&mut self, request: &CognitiveRequest) -> Result<CognitiveResponse> {
        self.release.notified().await;
        Ok(CognitiveResponse {
            schema_version: 1,
            request_id: request.request_id.clone(),
            provider_id: self.descriptor.provider_id.clone(),
            provider_role: self.descriptor.role,
            implementation: self.descriptor.implementation.clone(),
            implementation_version: "1".to_string(),
            model_version: None,
            status: CognitiveResponseStatus::Completed,
            confidence: 0.9,
            uncertainty: 0.1,
            input_snapshot: request.input_snapshot.clone(),
            completed_at_ms: 20,
            resource_cost: ResourceCost::default(),
            provenance: vec!["stalled-provider-fixture".to_string()],
            payload: CognitiveResponsePayload::SceneDescription {
                text: "a clear path".to_string(),
                embedding: Vec::new(),
            },
            failure: None,
        })
    }
}

#[tokio::test]
async fn stalled_provider_never_delays_poll_and_obsolete_answer_is_rejected() {
    let release = Arc::new(Notify::new());
    let mut router = CognitiveRouter::default();
    router.register(Box::new(StalledProvider {
        descriptor: provider("forebrain", Locality::LocalNetwork, 10),
        release: release.clone(),
    }));
    let mut supervisor = AsyncCognitionSupervisor::new(router, 10);
    let submitted = request();
    assert_eq!(
        supervisor.submit(submitted.clone(), 10),
        SubmissionDisposition::Submitted
    );

    // This is the control-loop guarantee: polling pending network/model
    // work completes locally, without awaiting the provider.
    tokio::time::timeout(
        Duration::from_millis(10),
        supervisor.poll(&submitted.input_snapshot, 20),
    )
    .await
    .expect("poll must not delay the control tick");

    release.notify_one();
    tokio::task::yield_now().await;
    let mut newer = submitted.input_snapshot.clone();
    newer.revision += 1;
    let response = loop {
        if let Some(response) = supervisor.poll(&newer, 30).await {
            break response;
        }
        tokio::task::yield_now().await;
    };
    assert_eq!(response.disposition, ResponseDisposition::Stale);
    assert_eq!(response.response.status, CognitiveResponseStatus::Stale);
    assert_eq!(
        response.response.failure.as_deref(),
        Some("response belongs to an obsolete snapshot")
    );
}

#[tokio::test]
async fn cancelled_answer_cannot_reenter_the_runtime() {
    let release = Arc::new(Notify::new());
    let mut router = CognitiveRouter::default();
    router.register(Box::new(StalledProvider {
        descriptor: provider("forebrain", Locality::LocalNetwork, 10),
        release: release.clone(),
    }));
    let mut supervisor = AsyncCognitionSupervisor::new(router, 10);
    let submitted = request();
    supervisor.submit(submitted.clone(), 10);
    supervisor.cancel(submitted.cancellation_token.clone());
    release.notify_one();
    let response = loop {
        if let Some(response) = supervisor.poll(&submitted.input_snapshot, 30).await {
            break response;
        }
        tokio::task::yield_now().await;
    };
    assert_eq!(response.disposition, ResponseDisposition::Rejected);
    assert_eq!(response.response.status, CognitiveResponseStatus::Cancelled);
}
