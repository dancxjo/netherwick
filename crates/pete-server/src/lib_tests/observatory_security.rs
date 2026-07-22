use tower::ServiceExt;

fn security_request(
    method: axum::http::Method,
    uri: &str,
    token: Option<&str>,
) -> axum::http::Request<axum::body::Body> {
    let mut builder = axum::http::Request::builder().method(method).uri(uri);
    if let Some(token) = token {
        builder = builder.header(
            axum::http::header::AUTHORIZATION,
            format!("Bearer {token}"),
        );
    }
    builder.body(axum::body::Body::empty()).unwrap()
}

#[tokio::test]
async fn remote_observatory_requires_read_auth_and_rejects_cross_origin_websocket() {
    let state = LiveViewState::new().with_test_observatory_security(
        "read-secret",
        Some("control-secret"),
        &["https://engineer.example"],
    );
    let router = live_view_router(state);

    let unauthenticated = router
        .clone()
        .oneshot(security_request(
            axum::http::Method::GET,
            "/view/observatory",
            None,
        ))
        .await
        .unwrap();
    assert_eq!(unauthenticated.status(), StatusCode::UNAUTHORIZED);

    let authenticated = router
        .clone()
        .oneshot(security_request(
            axum::http::Method::GET,
            "/view/observatory",
            Some("read-secret"),
        ))
        .await
        .unwrap();
    assert_eq!(authenticated.status(), StatusCode::OK);

    let cross_origin = axum::http::Request::builder()
        .uri("/api/observatory/events/ws")
        .header(axum::http::header::AUTHORIZATION, "Bearer read-secret")
        .header(axum::http::header::HOST, "robot.example")
        .header(axum::http::header::ORIGIN, "https://evil.example")
        .body(axum::body::Body::empty())
        .unwrap();
    let response = router.clone().oneshot(cross_origin).await.unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);

    let allowed_origin = axum::http::Request::builder()
        .uri("/api/observatory/events/ws")
        .header(axum::http::header::AUTHORIZATION, "Bearer read-secret")
        .header(axum::http::header::HOST, "robot.example")
        .header(axum::http::header::ORIGIN, "https://engineer.example")
        .body(axum::body::Body::empty())
        .unwrap();
    let response = router.oneshot(allowed_origin).await.unwrap();
    assert_ne!(response.status(), StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn read_only_token_cannot_discover_or_invoke_control_routes() {
    let state = LiveViewState::new().with_test_observatory_security(
        "read-secret",
        Some("control-secret"),
        &[],
    );
    let router = secured_live_view_router(
        live_view_routes(state.clone()).merge(reign_router(ReignServerState::standalone())),
        state,
    );
    for path in [
        "/reign/command",
        "/reign/command/ws",
        "/reign/hardware-arm",
        "/view/calibration",
        "/view/inline-learning",
    ] {
        let method = if path.ends_with("/ws") {
            axum::http::Method::GET
        } else {
            axum::http::Method::POST
        };
        let response = router
            .clone()
            .oneshot(security_request(method, path, Some("read-secret")))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
    }
}

#[tokio::test]
async fn malicious_verification_body_and_expensive_request_rate_are_bounded() {
    let state = LiveViewState::new().with_test_observatory_security("read-secret", None, &[]);
    let router = live_view_router(state);
    let oversized = axum::http::Request::builder()
        .method(axum::http::Method::POST)
        .uri("/api/observatory/diagnostic-verify")
        .header(axum::http::header::AUTHORIZATION, "Bearer read-secret")
        .header(
            axum::http::header::CONTENT_LENGTH,
            (MAX_DIAGNOSTIC_VERIFY_BYTES + 1).to_string(),
        )
        .body(axum::body::Body::empty())
        .unwrap();
    assert_eq!(
        router.clone().oneshot(oversized).await.unwrap().status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );

    for attempt in 0..9 {
        let response = router
            .clone()
            .oneshot(security_request(
                axum::http::Method::GET,
                "/api/observatory/diagnostic-export",
                Some("read-secret"),
            ))
            .await
            .unwrap();
        if attempt < 8 {
            assert_ne!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        } else {
            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        }
    }
}

#[tokio::test]
async fn security_failures_are_loss_intolerant_and_never_record_credentials() {
    let state = LiveViewState::new().with_test_observatory_security("read-secret", None, &[]);
    let hub = state.observatory();
    let router = live_view_router(state);
    let response = router
        .oneshot(security_request(
            axum::http::Method::GET,
            "/api/observatory/history",
            Some("do-not-record-me"),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    wait_for_observatory_sequence(&hub, 1).await;
    let response = hub.query(&BrainEventQuery::default()).unwrap();
    let event = response
        .records
        .iter()
        .find_map(|record| match record {
            BrainEventStreamRecord::Event { envelope }
                if envelope.event.kind == "observatory.access_rejected" =>
            {
                Some(&envelope.event)
            }
            _ => None,
        })
        .unwrap();
    assert!(matches!(event.loss_policy, LossPolicy::LossIntolerant));
    let serialized = serde_json::to_string(event).unwrap();
    assert!(!serialized.contains("do-not-record-me"));
    assert!(!serialized.contains("read-secret"));
    hub.shutdown().await;
}

#[test]
fn endpoint_inventory_covers_every_observatory_route() {
    let inventory = OBSERVATORY_ENDPOINT_INVENTORY
        .iter()
        .map(|endpoint| endpoint.path)
        .collect::<BTreeSet<_>>();
    let declared = HTTP_ENDPOINTS
        .iter()
        .copied()
        .filter(|path| path.starts_with("/api/observatory/") || *path == "/view/observatory")
        .collect::<BTreeSet<_>>();
    assert_eq!(inventory, declared);
    assert!(OBSERVATORY_ENDPOINT_INVENTORY.iter().all(|endpoint| {
        !endpoint.data_class.is_empty() && !endpoint.resource_limit.is_empty()
    }));
}

#[test]
fn non_loopback_binding_fails_closed_without_a_read_token() {
    let policy = ObservatorySecurityPolicy {
        require_authentication: AtomicBool::new(false),
        read_token_sha256: None,
        control_token_sha256: None,
        allowed_origins: BTreeSet::new(),
        rate_windows: Mutex::new(HashMap::new()),
        rejected_requests: AtomicU64::new(0),
    };
    assert!(policy
        .configure_binding("0.0.0.0:8787".parse().unwrap(), true)
        .is_err());
    assert!(policy
        .configure_binding("127.0.0.1:8787".parse().unwrap(), false)
        .is_ok());
}
