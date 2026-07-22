const MAX_DIAGNOSTIC_VERIFY_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservatoryEndpointSecurity {
    pub method: &'static str,
    pub path: &'static str,
    pub data_class: &'static str,
    pub resource_limit: &'static str,
}

pub const OBSERVATORY_ENDPOINT_INVENTORY: &[ObservatoryEndpointSecurity] = &[
    ObservatoryEndpointSecurity { method: "GET", path: "/view/observatory", data_class: "live diagnostic UI", resource_limit: "static page" },
    ObservatoryEndpointSecurity { method: "GET", path: "/api/observatory/history", data_class: "retained and durable robot events", resource_limit: "maximum 2000 events per page" },
    ObservatoryEndpointSecurity { method: "GET", path: "/api/observatory/health", data_class: "transport and durability health", resource_limit: "constant size" },
    ObservatoryEndpointSecurity { method: "GET", path: "/api/observatory/snapshots/{id}", data_class: "historical robot state", resource_limit: "one retained snapshot" },
    ObservatoryEndpointSecurity { method: "GET", path: "/api/observatory/snapshot", data_class: "historical robot state", resource_limit: "one retained snapshot" },
    ObservatoryEndpointSecurity { method: "GET", path: "/api/observatory/events/ws", data_class: "live robot event stream", resource_limit: "30 upgrades per minute plus bounded broadcast" },
    ObservatoryEndpointSecurity { method: "GET", path: "/api/observatory/provenance/{id}", data_class: "evidence and artifact graph", resource_limit: "bounded graph and 60 expensive queries per minute" },
    ObservatoryEndpointSecurity { method: "GET", path: "/api/observatory/authority", data_class: "command and safety authority history", resource_limit: "bounded history and 60 expensive queries per minute" },
    ObservatoryEndpointSecurity { method: "GET", path: "/api/observatory/calibration", data_class: "calibration evidence and artifacts", resource_limit: "bounded history and 60 expensive queries per minute" },
    ObservatoryEndpointSecurity { method: "GET", path: "/api/observatory/spatial", data_class: "imagery map and spatial evidence", resource_limit: "bounded retained assets and 60 expensive queries per minute" },
    ObservatoryEndpointSecurity { method: "GET", path: "/api/observatory/component-health", data_class: "robot component and resource health", resource_limit: "bounded history" },
    ObservatoryEndpointSecurity { method: "GET", path: "/api/observatory/diagnostic-export", data_class: "portable robot diagnostic evidence", resource_limit: "50000 events and 8 exports per minute" },
    ObservatoryEndpointSecurity { method: "POST", path: "/api/observatory/diagnostic-verify", data_class: "untrusted diagnostic bundle", resource_limit: "16 MiB body and 8 verifications per minute" },
    ObservatoryEndpointSecurity { method: "GET", path: "/api/observatory/compare", data_class: "multi-lane robot state comparison", resource_limit: "bounded history and 60 expensive queries per minute" },
];

#[derive(Debug)]
struct ObservatorySecurityPolicy {
    require_authentication: AtomicBool,
    read_token_sha256: Option<[u8; 32]>,
    control_token_sha256: Option<[u8; 32]>,
    allowed_origins: BTreeSet<String>,
    rate_windows: Mutex<HashMap<&'static str, VecDeque<std::time::Instant>>>,
    rejected_requests: AtomicU64,
}

impl ObservatorySecurityPolicy {
    fn from_env() -> Self {
        let token_hash = |name| {
            std::env::var(name)
                .ok()
                .filter(|token| !token.is_empty())
                .map(|token| Sha256::digest(token.as_bytes()).into())
        };
        let allowed_origins = std::env::var("PETE_OBSERVATORY_ALLOWED_ORIGINS")
            .ok()
            .into_iter()
            .flat_map(|origins| {
                origins
                    .split(',')
                    .map(str::trim)
                    .filter(|origin| !origin.is_empty())
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .collect();
        Self {
            require_authentication: AtomicBool::new(false),
            read_token_sha256: token_hash("PETE_OBSERVATORY_TOKEN"),
            control_token_sha256: token_hash("PETE_CONTROL_TOKEN"),
            allowed_origins,
            rate_windows: Mutex::new(HashMap::new()),
            rejected_requests: AtomicU64::new(0),
        }
    }

    #[cfg(test)]
    fn for_test(read_token: &str, control_token: Option<&str>, origins: &[&str]) -> Self {
        Self {
            require_authentication: AtomicBool::new(true),
            read_token_sha256: Some(Sha256::digest(read_token.as_bytes()).into()),
            control_token_sha256: control_token
                .map(|token| Sha256::digest(token.as_bytes()).into()),
            allowed_origins: origins.iter().map(|origin| (*origin).to_string()).collect(),
            rate_windows: Mutex::new(HashMap::new()),
            rejected_requests: AtomicU64::new(0),
        }
    }

    fn configure_binding(&self, addr: SocketAddr, tls: bool) -> io::Result<()> {
        if addr.ip().is_loopback() {
            return Ok(());
        }
        if self.read_token_sha256.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "non-loopback Observatory bind requires PETE_OBSERVATORY_TOKEN",
            ));
        }
        let trusted_proxy = std::env::var("PETE_OBSERVATORY_BEHIND_TLS_PROXY")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "yes"));
        if !tls && !trusted_proxy {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "non-loopback Observatory bind requires TLS or PETE_OBSERVATORY_BEHIND_TLS_PROXY=1",
            ));
        }
        self.require_authentication.store(true, Ordering::Release);
        Ok(())
    }

    fn origin_allowed(&self, headers: &axum::http::HeaderMap) -> bool {
        let Some(origin) = headers
            .get(axum::http::header::ORIGIN)
            .and_then(|value| value.to_str().ok())
        else {
            return true;
        };
        if self.allowed_origins.contains(origin) {
            return true;
        }
        let Some(host) = headers
            .get(axum::http::header::HOST)
            .and_then(|value| value.to_str().ok())
        else {
            return false;
        };
        origin
            .split_once("://")
            .is_some_and(|(_, authority)| authority == host)
    }

    fn token_allowed(&self, headers: &axum::http::HeaderMap, control: bool) -> bool {
        if !self.require_authentication.load(Ordering::Acquire) {
            return true;
        }
        let expected = if control {
            self.control_token_sha256.as_ref()
        } else {
            self.read_token_sha256.as_ref()
        };
        let Some(expected) = expected else {
            return false;
        };
        let Some(token) = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
        else {
            return false;
        };
        let actual: [u8; 32] = Sha256::digest(token.as_bytes()).into();
        actual
            .iter()
            .zip(expected)
            .fold(0_u8, |difference, (actual, expected)| {
                difference | (actual ^ expected)
            })
            == 0
    }

    fn within_rate_limit(&self, path: &str) -> bool {
        let Some((class, limit)) = security_rate_class(path) else {
            return true;
        };
        let now = std::time::Instant::now();
        let cutoff = now - std::time::Duration::from_secs(60);
        let Ok(mut windows) = self.rate_windows.lock() else {
            return false;
        };
        let window = windows.entry(class).or_default();
        while window.front().is_some_and(|request| *request < cutoff) {
            window.pop_front();
        }
        if window.len() >= limit {
            return false;
        }
        window.push_back(now);
        true
    }
}

fn security_rate_class(path: &str) -> Option<(&'static str, usize)> {
    match path {
        "/api/observatory/diagnostic-export" => Some(("export", 8)),
        "/api/observatory/diagnostic-verify" => Some(("verify", 8)),
        "/api/observatory/events/ws" => Some(("websocket", 30)),
        "/api/observatory/provenance" | "/api/observatory/authority"
        | "/api/observatory/calibration" | "/api/observatory/spatial"
        | "/api/observatory/compare" => Some(("expensive-query", 60)),
        _ if path.starts_with("/api/observatory/provenance/") => {
            Some(("expensive-query", 60))
        }
        _ => None,
    }
}

fn security_control_path(method: &axum::http::Method, path: &str) -> bool {
    path == "/command"
        || path == "/mode"
        || path == "/reign/command"
        || path == "/reign/command/ws"
        || path == "/reign/prod"
        || path == "/reign/clear"
        || path == "/reign/hardware-arm"
        || (*method != axum::http::Method::GET && *method != axum::http::Method::HEAD)
            && (path == "/view/calibration"
                || path == "/view/inline-learning"
                || path == "/view/retina-frame"
                || path.starts_with("/view/behavior-nodes/"))
}

fn record_observatory_security_failure(state: &LiveViewState, path: &str, reason: &str) {
    state
        .observatory_security
        .rejected_requests
        .fetch_add(1, Ordering::Relaxed);
    let t_ms = wall_now_ms();
    let mut event = BrainEvent::historical(
        BrainEventId::from_domain("observatory-security", Uuid::new_v4()),
        BrainEventType::GateDecision,
        ProducerIdentity::new(Brain::Motherbrain, "observatory.security"),
        EventTimes::observed(t_ms, t_ms),
    );
    event.kind = "observatory.access_rejected".into();
    event.disposition = EventDisposition::Rejected;
    event.loss_policy = LossPolicy::LossIntolerant;
    event.payload = BrainEventPayload::inline(serde_json::json!({
        "path": path,
        "reason": reason,
        "credentials_recorded": false,
    }));
    let _ = state.publish_brain_event(event);
}

async fn observatory_security_middleware(
    State(state): State<LiveViewState>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let path = request.uri().path().to_string();
    let control = security_control_path(request.method(), &path);
    if !state
        .observatory_security
        .origin_allowed(request.headers())
    {
        record_observatory_security_failure(&state, &path, "origin_rejected");
        return (StatusCode::FORBIDDEN, "request origin is not allowed").into_response();
    }
    if !state
        .observatory_security
        .token_allowed(request.headers(), control)
    {
        record_observatory_security_failure(
            &state,
            &path,
            if control {
                "control_authorization_rejected"
            } else {
                "read_authorization_rejected"
            },
        );
        return if control {
            StatusCode::NOT_FOUND.into_response()
        } else {
            (StatusCode::UNAUTHORIZED, "Observatory authorization required").into_response()
        };
    }
    if !state.observatory_security.within_rate_limit(&path) {
        record_observatory_security_failure(&state, &path, "rate_limited");
        return (StatusCode::TOO_MANY_REQUESTS, "request rate limit exceeded").into_response();
    }
    if path == "/api/observatory/diagnostic-verify"
        && request
            .headers()
            .get(axum::http::header::CONTENT_LENGTH)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<usize>().ok())
            .is_some_and(|length| length > MAX_DIAGNOSTIC_VERIFY_BYTES)
    {
        record_observatory_security_failure(&state, &path, "body_too_large");
        return (StatusCode::PAYLOAD_TOO_LARGE, "diagnostic bundle is too large").into_response();
    }
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-store"),
    );
    response.headers_mut().insert(
        axum::http::header::X_CONTENT_TYPE_OPTIONS,
        axum::http::HeaderValue::from_static("nosniff"),
    );
    response
}

impl LiveViewState {
    fn configure_observatory_binding(&self, addr: SocketAddr, tls: bool) -> io::Result<()> {
        self.observatory_security.configure_binding(addr, tls)
    }

    #[cfg(test)]
    fn with_test_observatory_security(
        mut self,
        read_token: &str,
        control_token: Option<&str>,
        origins: &[&str],
    ) -> Self {
        self.observatory_security = Arc::new(ObservatorySecurityPolicy::for_test(
            read_token,
            control_token,
            origins,
        ));
        self
    }
}
