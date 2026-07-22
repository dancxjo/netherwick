pub const HTTP_ENDPOINTS: &[&str] = &[
    "/health",
    "/now",
    "/models",
    "/ledger/recent",
    "/command",
    "/mode",
    "/reign",
    "/reign/command",
    "/reign/command/ws",
    "/reign/prod",
    "/reign/state",
    "/reign/clear",
    "/reign/hardware-arm",
    "/debug/embodied",
    "/debug/embodied/graph",
    "/stream/now",
    "/stream/mind",
    "/stream/logs",
    "/stream/llm",
    "/view",
    "/view/snapshot",
    "/view/embodied",
    "/view/embodied/graph",
    "/api/experience/lineage",
    "/view/vision",
    "/view/scene",
    "/view/map",
    "/view/behavior-nodes",
    "/view/cognitive",
    "/api/cognitive/features",
    "/api/cognitive/clusters",
    "/api/cognitive/bindings",
    "/api/cognitive/hypotheses",
    "/api/cognitive/constellations",
    "/api/cognitive/associations",
    "/api/cognitive/predictions",
    "/api/cognitive/questions",
    "/api/cognitive/summary",
    "/api/observatory/history",
    "/api/observatory/health",
    "/api/observatory/snapshots/{id}",
    "/api/observatory/snapshot",
    "/api/observatory/events/ws",
    "/api/observatory/provenance/{id}",
    "/api/observatory/authority",
    "/api/observatory/calibration",
    "/view/observatory",
    "/view/3d",
    "/view/capture-scene",
    "/view/training/latest",
];

#[derive(Clone, Debug)]
pub struct ReignServerState {
    queue: Arc<Mutex<ReignQueue>>,
    latest_snapshot: Option<Arc<Mutex<Option<WorldSnapshot>>>>,
    hardware_control: Option<Arc<Mutex<HardwareControlState>>>,
}

impl ReignServerState {
    pub fn new(queue: Arc<Mutex<ReignQueue>>) -> Self {
        Self {
            queue,
            latest_snapshot: None,
            hardware_control: None,
        }
    }

    pub fn with_live_view(queue: Arc<Mutex<ReignQueue>>, live_view: &LiveViewState) -> Self {
        Self {
            queue,
            latest_snapshot: Some(Arc::clone(&live_view.latest)),
            hardware_control: Some(Arc::clone(&live_view.hardware_control)),
        }
    }

    pub fn with_latest_snapshot(
        queue: Arc<Mutex<ReignQueue>>,
        latest_snapshot: Arc<Mutex<Option<WorldSnapshot>>>,
    ) -> Self {
        Self {
            queue,
            latest_snapshot: Some(latest_snapshot),
            hardware_control: None,
        }
    }

    pub fn standalone() -> Self {
        Self::new(Arc::new(Mutex::new(ReignQueue::default())))
    }

    pub fn queue(&self) -> Arc<Mutex<ReignQueue>> {
        Arc::clone(&self.queue)
    }

    fn latest_snapshot(&self) -> Option<WorldSnapshot> {
        self.latest_snapshot
            .as_ref()
            .and_then(|latest| latest.lock().expect("live snapshot mutex poisoned").clone())
    }

    fn hardware_control_status(&self, now_ms: TimeMs) -> HardwareControlStatus {
        self.hardware_control
            .as_ref()
            .map(|state| {
                let state = state.lock().expect("hardware control mutex poisoned");
                state.status(self.latest_snapshot().as_ref(), now_ms)
            })
            .unwrap_or_else(|| HardwareControlStatus::unavailable("not a hardware cockpit session"))
    }

    fn set_hardware_armed(&self, armed: bool) -> Result<HardwareControlStatus, ReignApiError> {
        let hardware = self
            .hardware_control
            .as_ref()
            .ok_or_else(|| ReignApiError::forbidden("hardware cockpit is not available"))?;
        let now_ms = wall_now_ms();
        {
            let mut state = hardware
                .lock()
                .map_err(|_| ReignApiError::internal("hardware control lock poisoned"))?;
            if armed && !state.available {
                return Err(ReignApiError::forbidden(
                    "hardware cockpit is not available in this session",
                ));
            }
            state.armed = armed;
            state.last_changed_ms = Some(now_ms);
        }
        if !armed {
            self.queue
                .lock()
                .map_err(|_| ReignApiError::internal("reign queue lock poisoned"))?
                .push(ReignInput {
                    id: Uuid::new_v4(),
                    issued_at_ms: now_ms,
                    expires_at_ms: now_ms.saturating_add(HARDWARE_TTL_MAX_MS),
                    source: ReignSource::WebRemote,
                    mode: ReignMode::Direct,
                    command: ReignCommand::Stop,
                    priority: 1.0,
                    note: Some("hardware cockpit disarmed".to_string()),
                });
        }
        Ok(self.hardware_control_status(now_ms))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReignCommandRequest {
    #[serde(default = "default_reign_mode")]
    pub mode: ReignMode,
    pub command: ReignCommand,
    #[serde(default = "default_priority")]
    pub priority: f32,
    pub ttl_ms: Option<TimeMs>,
    pub note: Option<String>,
    pub source: Option<ReignSource>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReignCommandWsResponse {
    pub accepted: bool,
    pub input: Option<ReignInput>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct HardwareArmRequest {
    pub armed: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HardwareControlStatus {
    pub available: bool,
    pub armed: bool,
    pub mode: Option<String>,
    pub source: Option<String>,
    pub reason: Option<String>,
    pub ttl_min_ms: TimeMs,
    pub ttl_max_ms: TimeMs,
    pub max_forward_intensity: f32,
    pub max_turn_intensity: f32,
    pub body_age_ms: Option<TimeMs>,
}

impl HardwareControlStatus {
    fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            available: false,
            armed: false,
            mode: None,
            source: None,
            reason: Some(reason.into()),
            ttl_min_ms: HARDWARE_TTL_MIN_MS,
            ttl_max_ms: HARDWARE_TTL_MAX_MS,
            max_forward_intensity: HARDWARE_MAX_FORWARD_INTENSITY,
            max_turn_intensity: HARDWARE_MAX_TURN_INTENSITY,
            body_age_ms: None,
        }
    }
}
#[derive(Clone, Debug)]
struct HardwareControlState {
    available: bool,
    armed: bool,
    mode: Option<String>,
    source: Option<String>,
    last_changed_ms: Option<TimeMs>,
}

impl Default for HardwareControlState {
    fn default() -> Self {
        Self {
            available: false,
            armed: false,
            mode: None,
            source: None,
            last_changed_ms: None,
        }
    }
}

impl HardwareControlState {
    fn real_slow() -> Self {
        Self {
            available: true,
            armed: false,
            mode: Some("slow".to_string()),
            source: Some("real_robot".to_string()),
            last_changed_ms: None,
        }
    }

    fn status(&self, snapshot: Option<&WorldSnapshot>, now_ms: TimeMs) -> HardwareControlStatus {
        let reason = if !self.available {
            Some("not a real slow robot session".to_string())
        } else if self.mode.as_deref() != Some("slow")
            || self.source.as_deref() != Some("real_robot")
        {
            Some("robot runner is not in real slow mode".to_string())
        } else if snapshot.is_none() {
            Some("no body snapshot yet".to_string())
        } else if let Some(snapshot) = snapshot {
            hardware_snapshot_block_reason(snapshot, now_ms)
        } else {
            None
        };
        HardwareControlStatus {
            available: self.available,
            armed: self.armed,
            mode: self.mode.clone(),
            source: self.source.clone(),
            reason,
            ttl_min_ms: HARDWARE_TTL_MIN_MS,
            ttl_max_ms: HARDWARE_TTL_MAX_MS,
            max_forward_intensity: HARDWARE_MAX_FORWARD_INTENSITY,
            max_turn_intensity: HARDWARE_MAX_TURN_INTENSITY,
            body_age_ms: snapshot.and_then(|snapshot| hardware_body_age_ms(snapshot, now_ms)),
        }
    }
}

const HARDWARE_TTL_MIN_MS: TimeMs = 250;
const HARDWARE_TTL_MAX_MS: TimeMs = 500;
const REIGN_TTL_MIN_MS: TimeMs = 100;
const REIGN_TTL_MAX_MS: TimeMs = 300_000;
const HARDWARE_MAX_FORWARD_INTENSITY: f32 = 0.15;
const HARDWARE_MAX_TURN_INTENSITY: f32 = 0.25;
const HARDWARE_BODY_STALE_MS: TimeMs = 1_000;
const HARDWARE_CRITICAL_BATTERY: f32 = 0.10;
const HARDWARE_WALL_CLOCK_FLOOR_MS: TimeMs = 1_600_000_000_000;

fn hardware_body_age_ms(snapshot: &WorldSnapshot, now_ms: TimeMs) -> Option<TimeMs> {
    let body_time = snapshot.body.last_update_ms;
    if body_time < HARDWARE_WALL_CLOCK_FLOOR_MS || body_time > now_ms {
        return None;
    }
    Some(now_ms.saturating_sub(body_time))
}

fn hardware_snapshot_block_reason(snapshot: &WorldSnapshot, now_ms: TimeMs) -> Option<String> {
    if let Some(age_ms) = hardware_body_age_ms(snapshot, now_ms) {
        if age_ms <= HARDWARE_BODY_STALE_MS {
            return hardware_snapshot_body_block_reason(snapshot);
        }
        return Some(format!("body snapshot stale: {age_ms} ms old"));
    }
    hardware_snapshot_body_block_reason(snapshot)
}

fn hardware_snapshot_body_block_reason(snapshot: &WorldSnapshot) -> Option<String> {
    if snapshot.body.flags.wheel_drop {
        return Some("wheel drop active".to_string());
    }
    if snapshot.body.flags.cliff_left
        || snapshot.body.flags.cliff_front_left
        || snapshot.body.flags.cliff_front_right
        || snapshot.body.flags.cliff_right
    {
        return Some("cliff sensor active".to_string());
    }
    if snapshot.body.battery_level <= HARDWARE_CRITICAL_BATTERY && !snapshot.body.charging {
        return Some("battery is critical".to_string());
    }
    None
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ProdRequest {
    pub kind: Option<String>,
    pub intensity: Option<f32>,
    pub duration_ms: Option<TimeMs>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProdResponse {
    pub accepted: bool,
    pub input: ReignInput,
}

pub fn reign_router(state: ReignServerState) -> Router {
    Router::new()
        .route("/reign", get(reign_page))
        .route("/reign/command", post(post_reign_command))
        .route("/reign/command/ws", get(get_reign_command_ws))
        .route("/reign/prod", post(post_reign_prod))
        .route("/reign/state", get(get_reign_state))
        .route("/reign/clear", post(post_reign_clear))
        .route("/reign/hardware-arm", post(post_hardware_arm))
        .with_state(state)
}
