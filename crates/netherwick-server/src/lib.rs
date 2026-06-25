use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{fs, io};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as AxumPath, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use axum_server::tls_rustls::RustlsConfig;
use base64::Engine;
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageEncoder};
use netherwick_actions::{
    ActionPrimitive, ReignCommand, ReignInput, ReignMode, ReignSource, TurnDir,
};
use netherwick_behaviors::{BehaviorNodeState, BehaviorNodeUpdate, BehaviorRegime};
use netherwick_body::{MotionCommand, MotorCommand};
use netherwick_core::TimeMs;
use netherwick_now::{KinectSkeletonSense, ReignSense};
use netherwick_runtime::{
    nudge_action_block_reason_for_snapshot, InlineLearningConfig, InlineLearningMode, NudgePolicy,
    NudgeStatus, ReignQueue, RuntimeModelStack,
};
use netherwick_sensors::{EyeFrameFormat, WorldSnapshot};
use netherwick_worldlab::CaptureReader;
use serde::{Deserialize, Serialize};
use tower_http::services::ServeDir;
use uuid::Uuid;

pub const HTTP_ENDPOINTS: &[&str] = &[
    "/health",
    "/now",
    "/models",
    "/ledger/recent",
    "/command",
    "/mode",
    "/reign",
    "/reign/command",
    "/reign/prod",
    "/reign/state",
    "/reign/clear",
    "/stream/now",
    "/stream/mind",
    "/stream/logs",
    "/stream/llm",
    "/view",
    "/view/snapshot",
    "/view/scene",
    "/view/behavior-nodes",
    "/view/3d",
    "/view/capture-scene",
    "/view/training/latest",
];

#[derive(Clone, Debug)]
pub struct ReignServerState {
    queue: Arc<Mutex<ReignQueue>>,
    latest_snapshot: Option<Arc<Mutex<Option<WorldSnapshot>>>>,
}

impl ReignServerState {
    pub fn new(queue: Arc<Mutex<ReignQueue>>) -> Self {
        Self {
            queue,
            latest_snapshot: None,
        }
    }

    pub fn with_live_view(queue: Arc<Mutex<ReignQueue>>, live_view: &LiveViewState) -> Self {
        Self {
            queue,
            latest_snapshot: Some(Arc::clone(&live_view.latest)),
        }
    }

    pub fn with_latest_snapshot(
        queue: Arc<Mutex<ReignQueue>>,
        latest_snapshot: Arc<Mutex<Option<WorldSnapshot>>>,
    ) -> Self {
        Self {
            queue,
            latest_snapshot: Some(latest_snapshot),
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
        .route("/reign/prod", post(post_reign_prod))
        .route("/reign/state", get(get_reign_state))
        .route("/reign/clear", post(post_reign_clear))
        .with_state(state)
}

#[derive(Clone, Debug)]
pub struct LiveViewState {
    latest: Arc<Mutex<Option<WorldSnapshot>>>,
    scene_metadata: Arc<Mutex<Option<LiveSceneMetadata>>>,
    session: Arc<Mutex<Option<SceneSession>>>,
    training_status: Arc<Mutex<LiveTrainingStatus>>,
    inline_learning: Arc<Mutex<InlineLearningConfig>>,
    prod_state: Arc<Mutex<NudgeStatus>>,
    behavior_nodes: Arc<Mutex<Vec<BehaviorNodeState>>>,
    pub virtual_retina: bool,
    pub retina_width: u32,
    pub retina_height: u32,
    pub retina_fps: f32,
    retina_state: Arc<Mutex<RetinaState>>,
}

#[derive(Clone, Debug, Default)]
struct RetinaState {
    latest_frame: Option<netherwick_sensors::EyeFrame>,
    has_new_frame: bool,
    last_received_at: Option<std::time::Instant>,
    frames_received: usize,
    frames_attached_to_snapshots: usize,
    frames_written_to_ledger: usize,
    warnings: Vec<String>,
}

impl Default for LiveViewState {
    fn default() -> Self {
        Self {
            latest: Arc::new(Mutex::new(None)),
            scene_metadata: Arc::new(Mutex::new(None)),
            session: Arc::new(Mutex::new(None)),
            training_status: Arc::new(Mutex::new(LiveTrainingStatus::default())),
            inline_learning: Arc::new(Mutex::new(InlineLearningConfig::default())),
            prod_state: Arc::new(Mutex::new(NudgeStatus::default())),
            behavior_nodes: Arc::new(Mutex::new(default_behavior_nodes())),
            virtual_retina: false,
            retina_width: 160,
            retina_height: 90,
            retina_fps: 5.0,
            retina_state: Arc::new(Mutex::new(RetinaState::default())),
        }
    }
}

impl LiveViewState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_virtual_retina(mut self, enabled: bool) -> Self {
        self.virtual_retina = enabled;
        self
    }

    pub fn with_retina_dimensions(mut self, width: u32, height: u32) -> Self {
        self.retina_width = width;
        self.retina_height = height;
        self
    }

    pub fn with_retina_fps(mut self, fps: f32) -> Self {
        self.retina_fps = fps;
        self
    }

    pub fn take_pending_retina_frame(&self) -> Option<netherwick_sensors::EyeFrame> {
        let mut state = self
            .retina_state
            .lock()
            .expect("retina state mutex poisoned");
        if state.has_new_frame {
            state.has_new_frame = false;
            state.frames_attached_to_snapshots += 1;
            state.latest_frame.clone()
        } else {
            None
        }
    }

    pub fn record_ledger_write(&self) {
        let mut state = self
            .retina_state
            .lock()
            .expect("retina state mutex poisoned");
        state.frames_written_to_ledger += 1;
    }

    pub fn update(&self, snapshot: WorldSnapshot) {
        *self
            .latest
            .lock()
            .expect("live view snapshot mutex poisoned") = Some(snapshot);
    }

    pub fn latest(&self) -> Option<WorldSnapshot> {
        self.latest
            .lock()
            .expect("live view snapshot mutex poisoned")
            .clone()
    }

    pub fn update_scene_metadata(&self, metadata: LiveSceneMetadata) {
        *self
            .scene_metadata
            .lock()
            .expect("live view scene metadata mutex poisoned") = Some(metadata);
    }

    pub fn scene_metadata(&self) -> Option<LiveSceneMetadata> {
        self.scene_metadata
            .lock()
            .expect("live view scene metadata mutex poisoned")
            .clone()
    }

    pub fn update_session(&self, session: SceneSession) {
        *self
            .session
            .lock()
            .expect("live view session mutex poisoned") = Some(session);
    }

    pub fn session(&self) -> Option<SceneSession> {
        self.session
            .lock()
            .expect("live view session mutex poisoned")
            .clone()
    }

    pub fn update_training_status(&self, status: LiveTrainingStatus) {
        *self
            .training_status
            .lock()
            .expect("live view training status mutex poisoned") = status;
    }

    pub fn training_status(&self) -> LiveTrainingStatus {
        self.training_status
            .lock()
            .expect("live view training status mutex poisoned")
            .clone()
    }

    pub fn update_inline_learning(&self, config: InlineLearningConfig) {
        *self
            .inline_learning
            .lock()
            .expect("inline learning mutex poisoned") = config;
    }

    pub fn inline_learning(&self) -> InlineLearningConfig {
        self.inline_learning
            .lock()
            .expect("inline learning mutex poisoned")
            .clone()
    }

    pub fn update_prod_state(&self, status: NudgeStatus) {
        *self.prod_state.lock().expect("prod state mutex poisoned") = status;
    }

    pub fn prod_state(&self) -> NudgeStatus {
        self.prod_state
            .lock()
            .expect("prod state mutex poisoned")
            .clone()
    }

    pub fn behavior_nodes(&self) -> Vec<BehaviorNodeState> {
        self.behavior_nodes
            .lock()
            .expect("behavior nodes mutex poisoned")
            .clone()
    }

    pub fn update_behavior_nodes(&self, nodes: Vec<BehaviorNodeState>) {
        let mut current = self
            .behavior_nodes
            .lock()
            .expect("behavior nodes mutex poisoned");
        let merged = nodes
            .into_iter()
            .map(|mut node| {
                if let Some(previous) = current
                    .iter()
                    .find(|old| same_behavior_node(&old.node_id, &node.node_id))
                {
                    if node.checkpoint_path.is_none() {
                        node.checkpoint_path = previous.checkpoint_path.clone();
                    }
                    node.training_enabled = previous.training_enabled
                        || matches!(
                            node.selected_regime,
                            BehaviorRegime::ShadowTrain | BehaviorRegime::ModelTrainAndInfer
                        );
                }
                node.missing_model_or_checkpoint =
                    !matches!(node.selected_regime, BehaviorRegime::Hardcoded)
                        && (node.selected_model.is_none()
                            || node
                                .checkpoint_path
                                .as_ref()
                                .map(|path| path.trim().is_empty())
                                .unwrap_or(true));
                node
            })
            .collect();
        *current = merged;
    }

    pub fn update_behavior_node(
        &self,
        id: &str,
        update: BehaviorNodeUpdate,
    ) -> Option<BehaviorNodeState> {
        let mut nodes = self
            .behavior_nodes
            .lock()
            .expect("behavior nodes mutex poisoned");
        let node = nodes.iter_mut().find(|node| {
            same_behavior_node(&node.node_id, id) || same_behavior_node(&node.behavior_id, id)
        })?;
        if let Some(regime) = update.selected_regime {
            node.selected_regime = regime;
            node.training_enabled = update.training_enabled.unwrap_or(matches!(
                regime,
                BehaviorRegime::ShadowTrain | BehaviorRegime::ModelTrainAndInfer
            ));
        }
        if let Some(hardcoded) = update.selected_hardcoded {
            node.selected_hardcoded = hardcoded;
        }
        if let Some(model) = update.selected_model {
            node.selected_model = Some(model);
        }
        if let Some(checkpoint) = update.checkpoint_path {
            node.checkpoint_path = (!checkpoint.trim().is_empty()).then_some(checkpoint);
        }
        if let Some(fallback) = update.fallback_policy {
            node.fallback_policy = fallback;
        }
        if let Some(training_enabled) = update.training_enabled {
            node.training_enabled = training_enabled;
        }
        node.missing_model_or_checkpoint =
            !matches!(node.selected_regime, BehaviorRegime::Hardcoded)
                && (node.selected_model.is_none()
                    || node
                        .checkpoint_path
                        .as_ref()
                        .map(|path| path.trim().is_empty())
                        .unwrap_or(true));
        Some(node.clone())
    }
}

fn default_behavior_nodes() -> Vec<BehaviorNodeState> {
    RuntimeModelStack::default().behavior_node_states(&[])
}

fn same_behavior_node(left: &str, right: &str) -> bool {
    normalize_behavior_node_id(left) == normalize_behavior_node_id(right)
}

fn normalize_behavior_node_id(id: &str) -> String {
    match id {
        "ActionValue" => "action_value".to_string(),
        "EyeNext" => "eye_next".to_string(),
        "EarNext" => "ear_next".to_string(),
        "EventBump" => "event_bump".to_string(),
        "EventFaceDetected" => "event_face_detected".to_string(),
        other => other.to_ascii_lowercase().replace('-', "_"),
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct LiveSnapshotResponse {
    pub t_ms: TimeMs,
    pub body: netherwick_body::BodySense,
    pub range: netherwick_now::RangeSense,
    pub eye_frame: Option<netherwick_sensors::EyeFrame>,
    pub gps: Option<netherwick_now::GpsSense>,
    pub ear_pcm: Option<netherwick_sensors::PcmAudioFrame>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LiveSceneMetadata {
    pub arena: Option<SceneArena>,
    #[serde(default)]
    pub objects: Vec<SceneObject>,
    pub sensor_calibration: Option<SceneSensorCalibration>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneSensorCalibration {
    pub compact_depth_beam_count: usize,
    pub compact_depth_fov_rad: f32,
    pub depth_scale: f32,
    pub point_y_m: f32,
}

impl SceneSensorCalibration {
    pub fn sim_default() -> Self {
        Self {
            compact_depth_beam_count: 32,
            compact_depth_fov_rad: std::f32::consts::PI * 0.75,
            depth_scale: 1.0,
            point_y_m: 0.18,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneArena {
    pub width_m: f32,
    pub height_m: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneObject {
    pub id: String,
    pub kind: String,
    pub x_m: f32,
    pub y_m: f32,
    pub radius_m: f32,
    pub label: Option<String>,
    pub color_rgb: Option<[u8; 3]>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LiveSceneResponse {
    pub schema_version: u32,
    pub session: Option<SceneSession>,
    pub training: LiveTrainingStatus,
    pub training_mode: String,
    pub ledger_path: Option<String>,
    pub frames_written: usize,
    pub transitions_written: usize,
    pub models_loaded: Vec<String>,
    pub model_modes: HashMap<String, String>,
    pub behavior_nodes: Vec<BehaviorNodeState>,
    pub action_selector_mode: String,
    pub weights_updating: bool,
    pub t_ms: TimeMs,
    pub body: SceneBody,
    pub range: SceneRange,
    pub eye: Option<SceneEye>,
    pub kinect: SceneKinect,
    pub audio: Option<SceneAudio>,
    pub objects: Vec<SceneObject>,
    pub arena: Option<SceneArena>,
    pub sensor_calibration: Option<SceneSensorCalibration>,
    pub action: SceneAction,
    pub prod: SceneProd,
    pub idle_ms: u64,
    pub last_nudge_ms: Option<u64>,
    pub nudge_count_recent: u32,
    pub nudge_blocked_reason: Option<String>,
    pub active_nudge: bool,
    pub stuck: bool,
    pub dead_battery: bool,
    pub recovery_mode: Option<String>,
    pub stuck_ticks: usize,
    pub stuck_detail: SceneStuck,
    pub mind: SceneMind,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneSession {
    pub mode: String,
    pub scenario: Option<String>,
    pub seed: Option<u64>,
    pub source: String,
    pub tick_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LiveTrainingStatus {
    pub training_mode: String,
    pub ledger_path: Option<String>,
    pub frames_written: usize,
    pub transitions_written: usize,
    pub models_loaded: Vec<String>,
    pub model_modes: HashMap<String, String>,
    pub action_selector_mode: String,
    pub weights_updating: bool,
}

impl Default for LiveTrainingStatus {
    fn default() -> Self {
        Self {
            training_mode: "none".to_string(),
            ledger_path: None,
            frames_written: 0,
            transitions_written: 0,
            models_loaded: Vec::new(),
            model_modes: HashMap::new(),
            action_selector_mode: "baseline".to_string(),
            weights_updating: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SceneBody {
    pub x_m: f32,
    pub y_m: f32,
    pub heading_rad: f32,
    pub battery_level: f32,
    pub charging: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SceneRange {
    pub nearest_m: Option<f32>,
    pub beams: Vec<SceneRangeBeam>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SceneRangeBeam {
    pub angle_rad: f32,
    pub distance_m: f32,
    pub hit: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SceneEye {
    pub width: u32,
    pub height: u32,
    pub format: String,
    pub data_url: Option<String>,
    pub mean_luma: f32,
    pub non_background_ratio: f32,
    pub source: String,
    pub authoritative: bool,
    pub retina_connected: bool,
    pub retina_last_frame_age_ms: Option<u64>,
    pub frames_received: usize,
    pub frames_written_to_ledger: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SceneKinect {
    pub points: Vec<ScenePoint>,
    pub skeletons: Vec<KinectSkeletonSense>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coordinate_system: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ScenePoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SceneAudio {
    pub bearing_rad: Option<f32>,
    pub energy: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SceneAction {
    pub latest: Option<String>,
    pub desired_motor: Option<MotorCommand>,
    pub final_motor: Option<MotorCommand>,
    pub motion_sent: Option<MotionCommand>,
    pub motor_applied: Option<bool>,
    pub movement_delta: Option<f32>,
    pub safety_override: bool,
    pub not_moving_reason: Option<String>,
    pub latest_llm_proposed_action: Option<ActionPrimitive>,
    pub llm_action_accepted: Option<bool>,
    pub llm_action_safety_vetoed: Option<bool>,
    pub final_selected_action: Option<ActionPrimitive>,
    pub llm_action_ignored_reason: Option<String>,
    pub llm_action_safety_reason: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SceneProd {
    pub idle_ms: u64,
    pub last_nudge_ms: Option<u64>,
    pub nudge_count_recent: u32,
    pub nudge_blocked_reason: Option<String>,
    pub active_nudge: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SceneStuck {
    pub active: bool,
    pub class: Option<String>,
    pub trap_kind: Option<String>,
    pub stuck_ticks: usize,
    pub duration_ms: u64,
    pub recovery_phase: Option<String>,
    pub turn_direction: Option<String>,
    pub recovery_attempts: usize,
    pub repeated_trap_count: usize,
    pub clearance_m: Option<f32>,
    pub event_started: bool,
    pub recovered: bool,
    pub dead_battery: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SceneMind {
    pub combobulation: Option<String>,
    pub surprise: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ModelsResponse {
    pub schema_version: u32,
    pub root: String,
    pub models: Vec<ModelSummary>,
    pub registry: Vec<ModelRegistrySummary>,
    pub behavior_nodes: Vec<BehaviorNodeState>,
    pub connections: Vec<ModelConnection>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ModelSummary {
    pub name: String,
    pub behavior: Option<String>,
    pub checkpoint_path: String,
    pub samples_seen: Option<u64>,
    pub best_loss: Option<f32>,
    pub input_dim: Option<u64>,
    pub output_dim: Option<u64>,
    pub latent_dim: Option<u64>,
    pub width: Option<u64>,
    pub height: Option<u64>,
    pub evaluation: Option<ModelEvaluationSummary>,
    pub metrics: Option<ModelTrainingMetricSummary>,
    pub registered_status: Option<String>,
    pub allowed_modes: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ModelEvaluationSummary {
    pub sample_count: Option<u64>,
    pub model_loss_mean: Option<f32>,
    pub hardcoded_loss_mean: Option<f32>,
    pub selected_loss_mean: Option<f32>,
    pub model_better_than_hardcoded: Option<bool>,
    pub improvement_ratio: Option<f32>,
    pub recommendation: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ModelTrainingMetricSummary {
    pub record_count: usize,
    pub last_epoch: Option<u64>,
    pub last_sample_index: Option<u64>,
    pub last_train_loss: Option<f32>,
    pub last_model_loss: Option<f32>,
    pub last_hardcoded_loss: Option<f32>,
    pub last_selected_loss: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ModelRegistrySummary {
    pub name: String,
    pub behavior: String,
    pub checkpoint_path: String,
    pub status: String,
    pub allowed_modes: Vec<String>,
    pub scenario_success_rate: Option<f32>,
    pub collision_rate: Option<f32>,
    pub episodes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelConnection {
    pub from: String,
    pub to: String,
    pub label: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CaptureSceneQuery {
    pub capture: PathBuf,
    #[serde(default)]
    pub frame: usize,
}

pub fn live_view_router(state: LiveViewState) -> Router {
    Router::new()
        .route("/", get(live_view_page))
        .route("/now", get(get_live_now))
        .route("/models", get(get_models))
        .route("/view", get(live_view_page))
        .route("/view/snapshot", get(get_live_snapshot))
        .route("/view/scene", get(get_live_scene))
        .route("/view/behavior-nodes", get(get_behavior_nodes))
        .route("/view/behavior-nodes/{id}", post(post_behavior_node))
        .route(
            "/view/behavior-nodes/{id}/promote",
            post(post_promote_behavior_node),
        )
        .route("/view/3d", get(live_view_3d_page))
        .route("/view/capture-scene", get(get_capture_scene))
        .route("/stream/llm", get(get_llm_stream))
        .route("/view/retina-frame", post(post_retina_frame))
        .route("/view/retina/status", get(get_retina_status))
        .route("/view/retina/latest.png", get(get_retina_latest))
        .route("/view/training/latest", get(get_latest_training))
        .route("/view/inline-learning", get(get_inline_learning))
        .route("/view/inline-learning", post(post_inline_learning))
        .route("/view/calibration", post(post_calibration))
        .nest_service(
            "/static",
            ServeDir::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("static")),
        )
        .with_state(state)
}

pub async fn serve_live_view(addr: SocketAddr, state: LiveViewState) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, live_view_router(state)).await
}

pub async fn serve_live_view_with_reign(
    addr: SocketAddr,
    live_state: LiveViewState,
    reign_state: ReignServerState,
) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let router = live_view_router(live_state).merge(reign_router(reign_state));
    axum::serve(listener, router).await
}

pub async fn serve_live_view_tls(
    addr: SocketAddr,
    state: LiveViewState,
    cert_path: impl AsRef<Path>,
    key_path: impl AsRef<Path>,
) -> std::io::Result<()> {
    let config = RustlsConfig::from_pem_file(cert_path, key_path).await?;
    axum_server::bind_rustls(addr, config)
        .serve(live_view_router(state).into_make_service())
        .await
        .map_err(std::io::Error::other)
}

pub async fn serve_live_view_with_reign_tls(
    addr: SocketAddr,
    live_state: LiveViewState,
    reign_state: ReignServerState,
    cert_path: impl AsRef<Path>,
    key_path: impl AsRef<Path>,
) -> std::io::Result<()> {
    let config = RustlsConfig::from_pem_file(cert_path, key_path).await?;
    let router = live_view_router(live_state).merge(reign_router(reign_state));
    axum_server::bind_rustls(addr, config)
        .serve(router.into_make_service())
        .await
        .map_err(std::io::Error::other)
}

async fn get_live_snapshot(
    State(state): State<LiveViewState>,
) -> Result<Json<LiveSnapshotResponse>, LiveViewError> {
    let snapshot = state
        .latest()
        .ok_or_else(|| LiveViewError::unavailable("no live world snapshot has arrived yet"))?;
    Ok(Json(LiveSnapshotResponse {
        t_ms: snapshot.body.last_update_ms,
        body: snapshot.body,
        range: snapshot.range,
        eye_frame: snapshot.eye_frame,
        gps: snapshot.gps,
        ear_pcm: snapshot.ear_pcm,
    }))
}

async fn get_live_scene(
    State(state): State<LiveViewState>,
) -> Result<Json<LiveSceneResponse>, LiveViewError> {
    let snapshot = state
        .latest()
        .ok_or_else(|| LiveViewError::unavailable("no live world snapshot has arrived yet"))?;

    let rstate = state
        .retina_state
        .lock()
        .expect("retina state mutex poisoned");
    let connected = state.virtual_retina
        && rstate
            .last_received_at
            .map(|t| t.elapsed() < std::time::Duration::from_millis(1500))
            .unwrap_or(false);
    let retina_status = Some(RetinaStatusInfo {
        enabled: state.virtual_retina,
        connected,
        frames_received: rstate.frames_received,
        frames_written_to_ledger: rstate.frames_written_to_ledger,
    });

    Ok(Json(snapshot_to_scene(
        &snapshot,
        state.scene_metadata().as_ref(),
        state.session(),
        state.training_status(),
        state.prod_state(),
        state.behavior_nodes(),
        retina_status,
    )))
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BehaviorNodesResponse {
    pub schema_version: u32,
    pub nodes: Vec<BehaviorNodeState>,
    pub connections: Vec<ModelConnection>,
}

async fn get_behavior_nodes(State(state): State<LiveViewState>) -> Json<BehaviorNodesResponse> {
    Json(BehaviorNodesResponse {
        schema_version: 1,
        nodes: state.behavior_nodes(),
        connections: model_connections(),
    })
}

async fn post_behavior_node(
    State(state): State<LiveViewState>,
    AxumPath(id): AxumPath<String>,
    Json(update): Json<BehaviorNodeUpdate>,
) -> Result<Json<BehaviorNodeState>, LiveViewError> {
    state
        .update_behavior_node(&id, update)
        .map(Json)
        .ok_or_else(|| LiveViewError::not_found(format!("behavior node {id} was not found")))
}

async fn post_promote_behavior_node(
    State(state): State<LiveViewState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<BehaviorNodeState>, LiveViewError> {
    let current = state
        .behavior_nodes()
        .into_iter()
        .find(|node| {
            same_behavior_node(&node.node_id, &id) || same_behavior_node(&node.behavior_id, &id)
        })
        .ok_or_else(|| LiveViewError::not_found(format!("behavior node {id} was not found")))?;
    let gates_pass = current
        .last_run
        .as_ref()
        .and_then(|run| run.error.as_ref())
        .is_none()
        && current
            .last_run
            .as_ref()
            .and_then(|run| run.disagreement)
            .map(|value| value <= 0.25)
            .unwrap_or(true)
        && !current.missing_model_or_checkpoint;
    if !gates_pass {
        return Err(LiveViewError::bad_request(
            "promotion safety gates failed: resolve errors, disagreement, or missing model/checkpoint",
        ));
    }
    let update = BehaviorNodeUpdate {
        selected_regime: Some(BehaviorRegime::ModelInfer),
        ..BehaviorNodeUpdate::default()
    };
    state
        .update_behavior_node(&id, update)
        .map(Json)
        .ok_or_else(|| LiveViewError::not_found(format!("behavior node {id} was not found")))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetinaStatusInfo {
    pub enabled: bool,
    pub connected: bool,
    pub frames_received: usize,
    pub frames_written_to_ledger: usize,
}

#[derive(Deserialize)]
pub struct RetinaFramePayload {
    pub schema_version: u32,
    pub source: String,
    pub t_ms: u64,
    pub frame_index: usize,
    pub width: u32,
    pub height: u32,
    pub format: String,
    pub encoding: String,
    pub data: String,
}

async fn post_retina_frame(
    State(state): State<LiveViewState>,
    Json(payload): Json<RetinaFramePayload>,
) -> Result<StatusCode, LiveViewError> {
    let session = state.session();
    let is_virtual_live = session
        .as_ref()
        .map(|s| s.mode == "virtual-live")
        .unwrap_or(false);
    if !is_virtual_live {
        return Err(LiveViewError::forbidden(
            "retina frames are only accepted in virtual-live mode",
        ));
    }

    let mut base64_str = payload.data.as_str();
    if let Some(pos) = base64_str.find(',') {
        base64_str = &base64_str[pos + 1..];
    }
    let decoded_bytes = base64::engine::general_purpose::STANDARD
        .decode(base64_str)
        .map_err(|e| LiveViewError::bad_request(format!("invalid base64 image data: {e}")))?;

    let img = image::load_from_memory(&decoded_bytes)
        .map_err(|e| LiveViewError::bad_request(format!("failed to decode image format: {e}")))?;

    let width = img.width();
    let height = img.height();

    if width > state.retina_width || height > state.retina_height {
        return Err(LiveViewError::bad_request(format!(
            "retina frame dimensions ({}x{}) exceed configured limits ({}x{})",
            width, height, state.retina_width, state.retina_height
        )));
    }

    let sim_t_ms = state
        .latest()
        .map(|snap| snap.body.last_update_ms)
        .unwrap_or(0);
    if sim_t_ms > 0 && payload.t_ms + 1500 < sim_t_ms {
        return Err(LiveViewError::bad_request("retina frame is too stale"));
    }

    let raw_rgb = img.to_rgb8().into_raw();

    let format = match payload.format.as_str() {
        "Rgb8" => netherwick_sensors::EyeFrameFormat::Rgb8,
        "Bgr8" => netherwick_sensors::EyeFrameFormat::Bgr8,
        "Gray8" => netherwick_sensors::EyeFrameFormat::Gray8,
        "Mjpeg" => netherwick_sensors::EyeFrameFormat::Mjpeg,
        other => netherwick_sensors::EyeFrameFormat::Unknown(other.to_string()),
    };

    let eye_frame = netherwick_sensors::EyeFrame {
        captured_at_ms: payload.t_ms,
        width,
        height,
        format,
        bytes: raw_rgb,
        source: Some(payload.source.clone()),
    };

    {
        let mut rstate = state
            .retina_state
            .lock()
            .expect("retina state mutex poisoned");
        rstate.latest_frame = Some(eye_frame);
        rstate.has_new_frame = true;
        rstate.last_received_at = Some(std::time::Instant::now());
        rstate.frames_received += 1;
    }

    Ok(StatusCode::OK)
}

async fn post_calibration(
    State(state): State<LiveViewState>,
    Json(calibration): Json<SceneSensorCalibration>,
) -> Result<StatusCode, LiveViewError> {
    let mut metadata = state.scene_metadata().unwrap_or_default();
    metadata.sensor_calibration = Some(calibration);
    state.update_scene_metadata(metadata);
    Ok(StatusCode::OK)
}

#[derive(Clone, Debug, Serialize)]
pub struct InlineLearningControlResponse {
    pub config: InlineLearningConfig,
    pub enabled: bool,
    pub training_mode: String,
    pub weights_updating: bool,
}

fn inline_learning_response(config: InlineLearningConfig) -> InlineLearningControlResponse {
    InlineLearningControlResponse {
        enabled: config.is_enabled(),
        training_mode: config.training_mode_label().to_string(),
        weights_updating: config.is_enabled(),
        config,
    }
}

async fn get_inline_learning(
    State(state): State<LiveViewState>,
) -> Json<InlineLearningControlResponse> {
    Json(inline_learning_response(state.inline_learning()))
}

async fn post_inline_learning(
    State(state): State<LiveViewState>,
    Json(mut config): Json<InlineLearningConfig>,
) -> Result<Json<InlineLearningControlResponse>, LiveViewError> {
    if config.max_train_steps_per_tick > 64 {
        return Err(LiveViewError::bad_request(
            "max_train_steps_per_tick must be 64 or less",
        ));
    }
    if config.mode == InlineLearningMode::Off {
        config.max_train_steps_per_tick = config.max_train_steps_per_tick.max(1);
    }
    state.update_inline_learning(config.clone());
    Ok(Json(inline_learning_response(config)))
}

#[derive(Serialize)]
pub struct RetinaStatusResponse {
    pub enabled: bool,
    pub connected: bool,
    pub source: String,
    pub width: u32,
    pub height: u32,
    pub fps: f32,
    pub frames_received: usize,
    pub frames_attached_to_snapshots: usize,
    pub frames_written_to_ledger: usize,
    pub latest_age_ms: Option<u64>,
    pub latest_luma: Option<f32>,
    pub warnings: Vec<String>,
}

async fn get_retina_status(State(state): State<LiveViewState>) -> Json<RetinaStatusResponse> {
    let rstate = state
        .retina_state
        .lock()
        .expect("retina state mutex poisoned");
    let connected = state.virtual_retina
        && rstate
            .last_received_at
            .map(|t| t.elapsed() < std::time::Duration::from_millis(1500))
            .unwrap_or(false);
    let latest_age_ms = rstate
        .last_received_at
        .map(|t| t.elapsed().as_millis() as u64);
    let latest_luma = rstate.latest_frame.as_ref().map(|frame| {
        let stats = eye_frame_stats(frame);
        stats.mean_luma
    });
    let source = rstate
        .latest_frame
        .as_ref()
        .and_then(|f| f.source.clone())
        .unwrap_or_else(|| "none".to_string());

    let mut warnings = rstate.warnings.clone();
    if state.virtual_retina {
        if rstate.frames_received == 0 {
            warnings.push("no retina frame received yet".to_string());
        } else if !connected {
            warnings.push("retina frame stream is stale/disconnected".to_string());
        }
    }

    Json(RetinaStatusResponse {
        enabled: state.virtual_retina,
        connected,
        source,
        width: state.retina_width,
        height: state.retina_height,
        fps: state.retina_fps,
        frames_received: rstate.frames_received,
        frames_attached_to_snapshots: rstate.frames_attached_to_snapshots,
        frames_written_to_ledger: rstate.frames_written_to_ledger,
        latest_age_ms,
        latest_luma,
        warnings,
    })
}

fn encode_eye_png_bytes(frame: &netherwick_sensors::EyeFrame) -> Result<Vec<u8>, String> {
    match frame.format {
        EyeFrameFormat::Mjpeg => Ok(frame.bytes.clone()),
        EyeFrameFormat::Rgb8
        | EyeFrameFormat::Bgr8
        | EyeFrameFormat::Gray8
        | EyeFrameFormat::Yuyv422 => {
            let expected_len = match frame.format {
                EyeFrameFormat::Gray8 => frame.width as usize * frame.height as usize,
                EyeFrameFormat::Yuyv422 => frame.width as usize * frame.height as usize * 2,
                _ => frame.width as usize * frame.height as usize * 3,
            };
            if frame.bytes.len() < expected_len {
                return Err(format!(
                    "eye frame has {} bytes, expected at least {}",
                    frame.bytes.len(),
                    expected_len
                ));
            }
            let mut rgb = Vec::with_capacity(frame.width as usize * frame.height as usize * 3);
            match frame.format {
                EyeFrameFormat::Rgb8 => rgb.extend_from_slice(&frame.bytes[..expected_len]),
                EyeFrameFormat::Bgr8 => {
                    for pixel in frame.bytes[..expected_len].chunks_exact(3) {
                        rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
                    }
                }
                EyeFrameFormat::Gray8 => {
                    for value in &frame.bytes[..expected_len] {
                        rgb.extend_from_slice(&[*value, *value, *value]);
                    }
                }
                EyeFrameFormat::Yuyv422 => {
                    rgb.extend(yuyv422_to_rgb(&frame.bytes[..expected_len]));
                }
                _ => {}
            }
            let mut png = Vec::new();
            let result = PngEncoder::new(&mut png).write_image(
                &rgb,
                frame.width,
                frame.height,
                ColorType::Rgb8.into(),
            );
            match result {
                Ok(()) => Ok(png),
                Err(error) => Err(format!("failed to encode eye PNG: {error}")),
            }
        }
        EyeFrameFormat::Unknown(ref format) => {
            Err(format!("unsupported eye frame format {format}"))
        }
    }
}

async fn get_retina_latest(State(state): State<LiveViewState>) -> impl IntoResponse {
    let rstate = state
        .retina_state
        .lock()
        .expect("retina state mutex poisoned");
    if let Some(ref frame) = rstate.latest_frame {
        match encode_eye_png_bytes(frame) {
            Ok(bytes) => {
                let content_type = match frame.format {
                    EyeFrameFormat::Mjpeg => "image/jpeg",
                    _ => "image/png",
                };
                (
                    StatusCode::OK,
                    [(axum::http::header::CONTENT_TYPE, content_type)],
                    bytes,
                )
                    .into_response()
            }
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to encode frame: {err}"),
            )
                .into_response(),
        }
    } else {
        (StatusCode::NOT_FOUND, "no retina frame received yet").into_response()
    }
}

async fn get_models() -> Json<ModelsResponse> {
    Json(read_models_response(&default_model_root()))
}

async fn get_latest_training() -> impl IntoResponse {
    let report_path = Path::new("data/reports/virtual/latest.json");
    if report_path.exists() {
        if let Ok(content) = std::fs::read_to_string(report_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                return (StatusCode::OK, Json(json)).into_response();
            }
        }
    }
    let alt_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/reports/virtual/latest.json");
    if alt_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&alt_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                return (StatusCode::OK, Json(json)).into_response();
            }
        }
    }
    (
        StatusCode::OK,
        Json(serde_json::json!({ "status": "none" })),
    )
        .into_response()
}

fn default_model_root() -> PathBuf {
    let root = PathBuf::from("data/models");
    if root.exists() {
        root
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/models")
    }
}

fn read_models_response(root: &Path) -> ModelsResponse {
    let mut response = ModelsResponse {
        schema_version: 1,
        root: root.display().to_string(),
        connections: model_connections(),
        behavior_nodes: default_behavior_nodes(),
        ..ModelsResponse::default()
    };
    let registry = read_model_registry_summary(&root.join("registry.json"), &mut response.warnings);
    response.registry = registry.clone();

    let mut models = Vec::new();
    match fs::read_dir(root) {
        Ok(entries) => {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                if let Some(summary) = read_model_summary(&path, &registry) {
                    models.push(summary);
                }
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => response
            .warnings
            .push(format!("model root {} was not found", root.display())),
        Err(error) => response.warnings.push(format!(
            "failed to read model root {}: {error}",
            root.display()
        )),
    }

    for registered in &registry {
        if models
            .iter()
            .all(|model| model.checkpoint_path != registered.checkpoint_path)
        {
            models.push(ModelSummary {
                name: registered.name.clone(),
                behavior: Some(registered.behavior.clone()),
                checkpoint_path: registered.checkpoint_path.clone(),
                registered_status: Some(registered.status.clone()),
                allowed_modes: registered.allowed_modes.clone(),
                warnings: vec!["registered checkpoint has no local metadata summary".to_string()],
                ..ModelSummary::default()
            });
        }
    }

    models.sort_by(|left, right| {
        left.behavior
            .cmp(&right.behavior)
            .then_with(|| left.name.cmp(&right.name))
    });
    response.models = models;
    response
}

fn read_model_summary(path: &Path, registry: &[ModelRegistrySummary]) -> Option<ModelSummary> {
    let name = path.file_name()?.to_string_lossy().to_string();
    let checkpoint_path = path.display().to_string();
    let metadata = read_json_value(&path.join("metadata.json")).ok();
    let evaluation = read_json_value(&path.join("evaluation.json")).ok();
    let registry_entry = registry
        .iter()
        .find(|entry| same_path_text(&entry.checkpoint_path, &checkpoint_path));

    let behavior = evaluation
        .as_ref()
        .and_then(|value| value.get("behavior"))
        .and_then(|value| value.as_str())
        .or_else(|| registry_entry.map(|entry| entry.behavior.as_str()))
        .or_else(|| behavior_from_checkpoint_name(&name))
        .map(str::to_string);

    let mut warnings = Vec::new();
    if metadata.is_none() {
        warnings.push("metadata.json missing".to_string());
    }
    if evaluation.is_none() {
        warnings.push("evaluation.json missing".to_string());
    }

    Some(ModelSummary {
        name,
        behavior,
        checkpoint_path,
        samples_seen: metadata
            .as_ref()
            .and_then(|value| json_u64(value, "samples_seen")),
        best_loss: metadata
            .as_ref()
            .and_then(|value| json_f32(value, "best_loss")),
        input_dim: metadata
            .as_ref()
            .and_then(|value| json_u64(value, "input_dim")),
        output_dim: metadata
            .as_ref()
            .and_then(|value| json_u64(value, "output_dim")),
        latent_dim: metadata
            .as_ref()
            .and_then(|value| json_u64(value, "latent_dim")),
        width: metadata.as_ref().and_then(|value| json_u64(value, "width")),
        height: metadata
            .as_ref()
            .and_then(|value| json_u64(value, "height")),
        evaluation: evaluation.as_ref().map(model_evaluation_summary),
        metrics: read_metric_summary(&path.join("metrics.jsonl")),
        registered_status: registry_entry.map(|entry| entry.status.clone()),
        allowed_modes: registry_entry
            .map(|entry| entry.allowed_modes.clone())
            .unwrap_or_default(),
        warnings,
    })
}

fn model_evaluation_summary(value: &serde_json::Value) -> ModelEvaluationSummary {
    ModelEvaluationSummary {
        sample_count: json_u64(value, "sample_count"),
        model_loss_mean: json_f32(value, "model_loss_mean"),
        hardcoded_loss_mean: json_f32(value, "hardcoded_loss_mean"),
        selected_loss_mean: json_f32(value, "selected_loss_mean"),
        model_better_than_hardcoded: value
            .get("model_better_than_hardcoded")
            .and_then(|value| value.as_bool()),
        improvement_ratio: json_f32(value, "improvement_ratio"),
        recommendation: value
            .get("recommendation")
            .and_then(|value| value.as_str())
            .map(str::to_string),
    }
}

fn read_metric_summary(path: &Path) -> Option<ModelTrainingMetricSummary> {
    let text = fs::read_to_string(path).ok()?;
    let mut summary = ModelTrainingMetricSummary::default();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        summary.record_count += 1;
        summary.last_epoch = json_u64(&value, "epoch");
        summary.last_sample_index = json_u64(&value, "sample_index");
        summary.last_train_loss = json_f32(&value, "train_loss");
        summary.last_model_loss = json_f32(&value, "model_loss");
        summary.last_hardcoded_loss = json_f32(&value, "hardcoded_loss");
        summary.last_selected_loss = json_f32(&value, "selected_loss");
    }
    (summary.record_count > 0).then_some(summary)
}

fn read_model_registry_summary(
    path: &Path,
    warnings: &mut Vec<String>,
) -> Vec<ModelRegistrySummary> {
    let Ok(value) = read_json_value(path) else {
        warnings.push(format!(
            "model registry {} was not readable",
            path.display()
        ));
        return Vec::new();
    };
    value
        .get("entries")
        .and_then(|value| value.as_array())
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| {
                    Some(ModelRegistrySummary {
                        name: entry.get("name")?.as_str()?.to_string(),
                        behavior: entry.get("behavior")?.as_str()?.to_string(),
                        checkpoint_path: entry.get("checkpoint")?.as_str()?.to_string(),
                        status: entry
                            .get("status")
                            .and_then(|value| value.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                        allowed_modes: entry
                            .get("allowed_modes")
                            .and_then(|value| value.as_array())
                            .map(|values| {
                                values
                                    .iter()
                                    .filter_map(|value| value.as_str().map(str::to_string))
                                    .collect()
                            })
                            .unwrap_or_default(),
                        scenario_success_rate: entry
                            .get("metrics")
                            .and_then(|value| json_f32(value, "scenario_success_rate")),
                        collision_rate: entry
                            .get("metrics")
                            .and_then(|value| json_f32(value, "collision_rate")),
                        episodes: entry
                            .get("metrics")
                            .and_then(|value| json_u64(value, "episodes")),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn model_connections() -> Vec<ModelConnection> {
    [
        ("Sensors", "Now", "snapshot"),
        ("Now", "Experience", "encode"),
        ("Now", "Danger", "range/body/action"),
        ("Now", "Charge", "battery/dock cues"),
        ("Now", "EyeNext", "vision context"),
        ("Now", "EarNext", "audio context"),
        ("Now", "EventBump", "body bump event"),
        ("Now", "EventFaceDetected", "face event"),
        ("Experience", "Future", "latent z"),
        ("EventBump", "Autonomic", "scripted escape"),
        ("EventFaceDetected", "Conductor", "scripted greeting"),
        ("Danger", "Conductor", "risk"),
        ("Charge", "Conductor", "dock value"),
        ("Future", "Conductor", "imagined state"),
        ("Conductor", "ActionValue", "candidate actions"),
        ("ActionValue", "Autonomic", "ranked action"),
        ("Autonomic", "Body", "safe command"),
        ("Body", "Ledger", "transition"),
        ("Ledger", "Training", "replay"),
        ("Training", "Models", "checkpoints"),
    ]
    .into_iter()
    .map(|(from, to, label)| ModelConnection {
        from: from.to_string(),
        to: to.to_string(),
        label: label.to_string(),
    })
    .collect()
}

fn read_json_value(path: &Path) -> Result<serde_json::Value, io::Error> {
    let text = fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn json_f32(value: &serde_json::Value, key: &str) -> Option<f32> {
    value
        .get(key)
        .and_then(|value| value.as_f64())
        .filter(|value| value.is_finite())
        .map(|value| value as f32)
}

fn json_u64(value: &serde_json::Value, key: &str) -> Option<u64> {
    value.get(key).and_then(|value| value.as_u64())
}

fn same_path_text(left: &str, right: &str) -> bool {
    left.trim_end_matches('/') == right.trim_end_matches('/')
}

fn behavior_from_checkpoint_name(name: &str) -> Option<&'static str> {
    if name.starts_with("action_value") {
        Some("action_value")
    } else if name.starts_with("eye") {
        Some("eye_next")
    } else if name.starts_with("ear") {
        Some("ear_next")
    } else if name.starts_with("danger") {
        Some("danger")
    } else if name.starts_with("charge") {
        Some("charge")
    } else if name.starts_with("future") {
        Some("future")
    } else if name.starts_with("experience") {
        Some("experience")
    } else {
        None
    }
}

async fn get_capture_scene(
    Query(query): Query<CaptureSceneQuery>,
) -> Result<Json<LiveSceneResponse>, LiveViewError> {
    let reader = CaptureReader::open(&query.capture)
        .await
        .map_err(|error| LiveViewError::bad_request(format!("failed to open capture: {error}")))?;
    let frames = reader.read_frames().await.map_err(|error| {
        LiveViewError::bad_request(format!("failed to read capture frames: {error}"))
    })?;
    let record = frames.get(query.frame).ok_or_else(|| {
        LiveViewError::not_found(format!("capture frame {} was not found", query.frame))
    })?;
    let metadata = reader
        .manifest()
        .scenario
        .as_ref()
        .map(|scenario| LiveSceneMetadata {
            arena: Some(SceneArena {
                width_m: scenario.arena.width_m,
                height_m: scenario.arena.height_m,
            }),
            objects: scenario
                .objects
                .iter()
                .map(|object| SceneObject {
                    id: object.id.clone(),
                    kind: scene_object_kind(&format!("{:?}", object.kind)),
                    x_m: object.x_m,
                    y_m: object.y_m,
                    radius_m: object.radius_m,
                    label: Some(object.label.clone()),
                    color_rgb: Some(object.color_rgb),
                })
                .collect(),
            sensor_calibration: None,
        });
    let mut scene = snapshot_to_scene(
        &record.snapshot,
        metadata.as_ref(),
        None,
        LiveTrainingStatus::default(),
        NudgeStatus::default(),
        default_behavior_nodes(),
        None,
    );
    scene.t_ms = record.t_ms;
    if let Some(pointcloud) = &record.assets.pointcloud {
        let ply_path = query.capture.join(pointcloud);
        if ply_path.exists() {
            match std::fs::read_to_string(&ply_path) {
                Ok(content) => {
                    let mut points = Vec::new();
                    let mut in_header = true;
                    for line in content.lines() {
                        let line = line.trim();
                        if in_header {
                            if line == "end_header" {
                                in_header = false;
                            }
                            continue;
                        }
                        if line.is_empty() {
                            continue;
                        }
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 3 {
                            if let (Ok(x), Ok(y), Ok(z)) = (
                                parts[0].parse::<f32>(),
                                parts[1].parse::<f32>(),
                                parts[2].parse::<f32>(),
                            ) {
                                points.push(ScenePoint {
                                    x,
                                    y: -y,
                                    z,
                                    r: 180,
                                    g: 180,
                                    b: 240,
                                });
                            }
                        }
                    }
                    scene.kinect.points = points;
                    scene.kinect.coordinate_system = Some("camera".to_string());
                }
                Err(error) => {
                    scene.warnings.push(format!(
                        "failed to read PLY file {}: {error}",
                        ply_path.display()
                    ));
                }
            }
        } else {
            scene
                .warnings
                .push(format!("PLY file not found: {}", ply_path.display()));
        }
    }
    Ok(Json(scene))
}

async fn get_live_now(
    State(state): State<LiveViewState>,
) -> Result<Json<netherwick_now::Now>, LiveViewError> {
    let snapshot = state
        .latest()
        .ok_or_else(|| LiveViewError::unavailable("no live world snapshot has arrived yet"))?;
    Ok(Json(snapshot.to_now(snapshot.body.last_update_ms)))
}

async fn live_view_page() -> Html<&'static str> {
    Html(LIVE_VIEW_PAGE)
}

async fn live_view_3d_page() -> Html<&'static str> {
    Html(LIVE_VIEW_3D_PAGE)
}

async fn get_llm_stream(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(stream_llm_events)
}

async fn stream_llm_events(mut socket: WebSocket) {
    let mut rx = netherwick_llm::subscribe_llm_streams();
    loop {
        match rx.recv().await {
            Ok(event) => {
                let Ok(text) = serde_json::to_string(&event) else {
                    continue;
                };
                if socket.send(Message::Text(text.into())).await.is_err() {
                    break;
                }
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
        }
    }
}

pub fn snapshot_to_scene(
    snapshot: &WorldSnapshot,
    metadata: Option<&LiveSceneMetadata>,
    session: Option<SceneSession>,
    training: LiveTrainingStatus,
    prod_state: NudgeStatus,
    behavior_nodes: Vec<BehaviorNodeState>,
    retina_status: Option<RetinaStatusInfo>,
) -> LiveSceneResponse {
    let mut warnings = Vec::new();
    let body = &snapshot.body;
    let eye = match snapshot.eye_frame.as_ref() {
        Some(f) => {
            let (eye, frame_warnings) =
                scene_eye_from_frame(f, retina_status.as_ref(), snapshot.body.last_update_ms);
            warnings.extend(frame_warnings);
            Some(eye)
        }
        None => {
            if let Some(rs) = retina_status.as_ref() {
                if rs.connected {
                    Some(SceneEye {
                        width: 0,
                        height: 0,
                        format: "None".to_string(),
                        data_url: None,
                        mean_luma: 1.0,
                        non_background_ratio: 1.0,
                        source: "babylon-robot-eye".to_string(),
                        authoritative: false,
                        retina_connected: true,
                        retina_last_frame_age_ms: None,
                        frames_received: rs.frames_received,
                        frames_written_to_ledger: rs.frames_written_to_ledger,
                    })
                } else {
                    warnings.push("no eye frame stream".to_string());
                    None
                }
            } else {
                warnings.push("no eye frame stream".to_string());
                None
            }
        }
    };
    let sensor_calibration = metadata.and_then(|metadata| metadata.sensor_calibration);
    let kinect = scene_kinect_from_snapshot(snapshot, sensor_calibration, &mut warnings);
    let audio = snapshot
        .kinect
        .audio_angle_rad
        .or_else(|| audio_bearing_from_objects(body.odometry.x_m, body.odometry.y_m, metadata))
        .map(|bearing_rad| SceneAudio {
            bearing_rad: Some(bearing_rad),
            energy: snapshot.kinect.audio_confidence.clamp(0.0, 1.0),
        });
    if audio.is_none() {
        warnings.push("no audio bearing stream".to_string());
    }
    let stuck = scene_stuck_from_snapshot(snapshot);
    let training_mode = training.training_mode.clone();
    let ledger_path = training.ledger_path.clone();
    let frames_written = training.frames_written;
    let transitions_written = training.transitions_written;
    let models_loaded = training.models_loaded.clone();
    let model_modes = training.model_modes.clone();
    let action_selector_mode = training.action_selector_mode.clone();
    let weights_updating = training.weights_updating;
    LiveSceneResponse {
        schema_version: 1,
        session,
        training,
        training_mode,
        ledger_path,
        frames_written,
        transitions_written,
        models_loaded,
        model_modes,
        behavior_nodes,
        action_selector_mode,
        weights_updating,
        t_ms: body.last_update_ms,
        body: SceneBody {
            x_m: body.odometry.x_m,
            y_m: body.odometry.y_m,
            heading_rad: body.odometry.heading_rad,
            battery_level: body.battery_level,
            charging: body.charging,
        },
        range: scene_range_from_snapshot(snapshot),
        eye,
        kinect,
        audio,
        objects: metadata
            .map(|metadata| metadata.objects.clone())
            .unwrap_or_default(),
        arena: metadata.and_then(|metadata| metadata.arena),
        sensor_calibration,
        action: scene_action_from_snapshot(snapshot),
        prod: SceneProd {
            idle_ms: prod_state.idle_ms,
            last_nudge_ms: prod_state.last_nudge_ms,
            nudge_count_recent: prod_state.nudge_count_recent,
            nudge_blocked_reason: prod_state.nudge_blocked_reason.clone(),
            active_nudge: prod_state.active_nudge,
        },
        idle_ms: prod_state.idle_ms,
        last_nudge_ms: prod_state.last_nudge_ms,
        nudge_count_recent: prod_state.nudge_count_recent,
        nudge_blocked_reason: prod_state.nudge_blocked_reason.clone(),
        active_nudge: prod_state.active_nudge,
        stuck: stuck.active,
        dead_battery: stuck.dead_battery,
        recovery_mode: stuck.recovery_phase.clone(),
        stuck_ticks: stuck.stuck_ticks,
        stuck_detail: stuck,
        mind: SceneMind {
            combobulation: snapshot
                .extensions
                .iter()
                .find(|extension| extension.name == "vision.frame_summary")
                .map(|extension| {
                    format!(
                        "vision summary vector with {} values",
                        extension.values.len()
                    )
                }),
            surprise: None,
        },
        warnings,
    }
}

fn scene_action_from_snapshot(snapshot: &WorldSnapshot) -> SceneAction {
    let forward = snapshot.body.velocity.forward_m_s;
    let turn = snapshot.body.velocity.turn_rad_s;
    let action_debug = snapshot.action_debug.as_ref();
    let latest = if forward.abs() < 0.01 && turn.abs() < 0.01 {
        Some("stop".to_string())
    } else if turn.abs() < 0.01 {
        Some(format!("forward {:.2} m/s", forward))
    } else if forward.abs() < 0.01 {
        Some(format!("turn {:.2} rad/s", turn))
    } else {
        Some(format!("drive {:.2} m/s, turn {:.2} rad/s", forward, turn))
    };
    SceneAction {
        latest,
        desired_motor: action_debug_value(action_debug, "desired_motor"),
        final_motor: action_debug_value(action_debug, "final_motor"),
        motion_sent: action_debug_value(action_debug, "motion_sent_to_sim"),
        motor_applied: action_debug
            .and_then(|debug| debug.get("motor_applied"))
            .and_then(|value| value.as_bool()),
        movement_delta: action_debug
            .and_then(|debug| debug.get("movement_delta"))
            .and_then(|value| serde_json::from_value(value.clone()).ok()),
        safety_override: action_debug
            .and_then(|debug| debug.get("safety_override"))
            .and_then(|value| value.as_bool())
            .unwrap_or_else(|| {
                snapshot.body.flags.wheel_drop
                    || snapshot.body.flags.cliff_left
                    || snapshot.body.flags.cliff_right
            }),
        not_moving_reason: action_debug
            .and_then(|debug| {
                debug
                    .get("why_not_moving")
                    .or_else(|| debug.get("not_moving_reason"))
            })
            .and_then(|value| value.as_str())
            .map(str::to_string),
        latest_llm_proposed_action: snapshot
            .llm_action_proposal
            .as_ref()
            .and_then(|proposal| proposal.proposed_action.clone()),
        llm_action_accepted: snapshot
            .llm_action_proposal
            .as_ref()
            .map(|proposal| proposal.accepted),
        llm_action_safety_vetoed: snapshot
            .llm_action_proposal
            .as_ref()
            .map(|proposal| proposal.safety_vetoed),
        final_selected_action: snapshot.final_selected_action.clone().or_else(|| {
            snapshot
                .llm_action_proposal
                .as_ref()
                .and_then(|proposal| proposal.final_action.clone())
        }),
        llm_action_ignored_reason: snapshot
            .llm_action_proposal
            .as_ref()
            .and_then(|proposal| proposal.ignored_reason.clone()),
        llm_action_safety_reason: snapshot
            .llm_action_proposal
            .as_ref()
            .and_then(|proposal| proposal.safety_reason.clone()),
    }
}

fn action_debug_value<T>(debug: Option<&serde_json::Value>, key: &str) -> Option<T>
where
    T: serde::de::DeserializeOwned,
{
    debug
        .and_then(|debug| debug.get(key))
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
}

fn scene_stuck_from_snapshot(snapshot: &WorldSnapshot) -> SceneStuck {
    let Some(extension) = snapshot
        .extensions
        .iter()
        .find(|extension| extension.name == "sim.stuck")
    else {
        return SceneStuck {
            dead_battery: snapshot.body.battery_level <= f32::EPSILON && !snapshot.body.charging,
            ..SceneStuck::default()
        };
    };
    let values = &extension.values;
    let active = values.first().copied().unwrap_or(0.0) > 0.0;
    let corner_trap = values.get(1).copied().unwrap_or(0.0) > 0.0;
    let phase_code = values.get(4).copied().unwrap_or(0.0).round() as i32;
    let trap_kind = trap_kind_name(values.get(10).copied().unwrap_or(0.0));
    SceneStuck {
        active,
        class: if active || corner_trap || trap_kind.is_some() {
            Some(
                trap_kind
                    .map(|kind| format!("{kind}-trap"))
                    .unwrap_or_else(|| {
                        if corner_trap {
                            "corner-trap".to_string()
                        } else {
                            "stuck".to_string()
                        }
                    }),
            )
        } else {
            None
        },
        trap_kind: trap_kind.map(str::to_string),
        stuck_ticks: values.get(2).copied().unwrap_or(0.0).max(0.0) as usize,
        duration_ms: values.get(3).copied().unwrap_or(0.0).max(0.0) as u64,
        recovery_phase: match phase_code {
            1 => Some("stop".to_string()),
            2 => Some("reverse".to_string()),
            3 => Some("turn-away".to_string()),
            4 => Some("probe".to_string()),
            _ => None,
        },
        turn_direction: match values.get(5).copied().unwrap_or(0.0) {
            value if value < 0.0 => Some("right".to_string()),
            value if value > 0.0 => Some("left".to_string()),
            _ => None,
        },
        recovery_attempts: values.get(11).copied().unwrap_or(0.0).max(0.0) as usize,
        repeated_trap_count: values.get(12).copied().unwrap_or(0.0).max(0.0) as usize,
        clearance_m: values.get(13).copied().filter(|value| *value >= 0.0),
        event_started: values.get(6).copied().unwrap_or(0.0) > 0.0,
        recovered: values.get(7).copied().unwrap_or(0.0) > 0.0,
        dead_battery: values.get(8).copied().unwrap_or(0.0) > 0.0,
    }
}

fn trap_kind_name(code: f32) -> Option<&'static str> {
    match code.round() as i32 {
        1 => Some("wall"),
        2 => Some("corner"),
        3 => Some("column"),
        _ => None,
    }
}

fn scene_range_from_snapshot(snapshot: &WorldSnapshot) -> SceneRange {
    let beam_count = snapshot.range.beams.len().max(1);
    let fov_rad = std::f32::consts::PI;
    SceneRange {
        nearest_m: snapshot.range.nearest_m,
        beams: snapshot
            .range
            .beams
            .iter()
            .enumerate()
            .map(|(index, distance_m)| {
                let ratio = if beam_count <= 1 {
                    0.5
                } else {
                    index as f32 / (beam_count - 1) as f32
                };
                let angle_rad = -fov_rad * 0.5 + ratio * fov_rad;
                SceneRangeBeam {
                    angle_rad,
                    distance_m: finite_or_zero(*distance_m),
                    hit: snapshot
                        .range
                        .nearest_m
                        .map(|nearest| (*distance_m - nearest).abs() < 0.05)
                        .unwrap_or(false),
                }
            })
            .collect(),
    }
}

fn scene_eye_from_frame(
    frame: &netherwick_sensors::EyeFrame,
    retina_status: Option<&RetinaStatusInfo>,
    current_t_ms: u64,
) -> (SceneEye, Vec<String>) {
    let mut warnings = Vec::new();
    let (data_url, encode_warning) = encode_eye_data_url(frame);
    if let Some(warning) = encode_warning {
        warnings.push(warning);
    }
    let stats = eye_frame_stats(frame);
    if stats.mean_luma < 0.08 {
        warnings.push(format!(
            "eye frame is very dim (mean luma {:.3})",
            stats.mean_luma
        ));
    }

    let source = frame.source.clone().unwrap_or_else(|| "none".to_string());
    let authoritative = source == "babylon-robot-eye" || source == "real-camera";
    let retina_connected = retina_status.map(|s| s.connected).unwrap_or(false);

    let retina_last_frame_age_ms = if source == "babylon-robot-eye" {
        Some(current_t_ms.saturating_sub(frame.captured_at_ms))
    } else {
        None
    };

    let frames_received = retina_status.map(|s| s.frames_received).unwrap_or(0);
    let frames_written_to_ledger = retina_status
        .map(|s| s.frames_written_to_ledger)
        .unwrap_or(0);

    (
        SceneEye {
            width: frame.width,
            height: frame.height,
            format: format!("{:?}", frame.format),
            data_url,
            mean_luma: stats.mean_luma,
            non_background_ratio: stats.non_background_ratio,
            source,
            authoritative,
            retina_connected,
            retina_last_frame_age_ms,
            frames_received,
            frames_written_to_ledger,
        },
        warnings,
    )
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct EyeFrameStats {
    mean_luma: f32,
    non_background_ratio: f32,
}

fn eye_frame_stats(frame: &netherwick_sensors::EyeFrame) -> EyeFrameStats {
    let pixels = frame.width as usize * frame.height as usize;
    if pixels == 0 {
        return EyeFrameStats::default();
    }
    let mut luma_sum = 0.0f32;
    let mut non_background = 0usize;
    match frame.format {
        EyeFrameFormat::Gray8 => {
            for value in frame.bytes.iter().take(pixels) {
                let luma = *value as f32 / 255.0;
                luma_sum += luma;
                if luma > 0.08 {
                    non_background += 1;
                }
            }
        }
        EyeFrameFormat::Rgb8 | EyeFrameFormat::Bgr8 => {
            for pixel in frame.bytes.chunks_exact(3).take(pixels) {
                let (r, g, b) = match frame.format {
                    EyeFrameFormat::Bgr8 => (pixel[2], pixel[1], pixel[0]),
                    _ => (pixel[0], pixel[1], pixel[2]),
                };
                let luma = (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32) / 255.0;
                luma_sum += luma;
                if luma > 0.08 || r.abs_diff(g) > 8 || g.abs_diff(b) > 8 {
                    non_background += 1;
                }
            }
        }
        EyeFrameFormat::Yuyv422 => {
            for pair in frame.bytes.chunks_exact(4).take(pixels.div_ceil(2)) {
                for value in [pair[0], pair[2]] {
                    let luma = value as f32 / 255.0;
                    luma_sum += luma;
                    if luma > 0.08 {
                        non_background += 1;
                    }
                }
            }
        }
        EyeFrameFormat::Mjpeg | EyeFrameFormat::Unknown(_) => {}
    }
    EyeFrameStats {
        mean_luma: luma_sum / pixels as f32,
        non_background_ratio: non_background as f32 / pixels as f32,
    }
}

fn encode_eye_data_url(frame: &netherwick_sensors::EyeFrame) -> (Option<String>, Option<String>) {
    match frame.format {
        EyeFrameFormat::Mjpeg => {
            let encoded = base64::engine::general_purpose::STANDARD.encode(&frame.bytes);
            (Some(format!("data:image/jpeg;base64,{encoded}")), None)
        }
        EyeFrameFormat::Rgb8
        | EyeFrameFormat::Bgr8
        | EyeFrameFormat::Gray8
        | EyeFrameFormat::Yuyv422 => {
            let expected_len = match frame.format {
                EyeFrameFormat::Gray8 => frame.width as usize * frame.height as usize,
                EyeFrameFormat::Yuyv422 => frame.width as usize * frame.height as usize * 2,
                _ => frame.width as usize * frame.height as usize * 3,
            };
            if frame.bytes.len() < expected_len {
                return (
                    None,
                    Some(format!(
                        "eye frame has {} bytes, expected at least {}",
                        frame.bytes.len(),
                        expected_len
                    )),
                );
            }
            let mut rgb = Vec::with_capacity(frame.width as usize * frame.height as usize * 3);
            match frame.format {
                EyeFrameFormat::Rgb8 => rgb.extend_from_slice(&frame.bytes[..expected_len]),
                EyeFrameFormat::Bgr8 => {
                    for pixel in frame.bytes[..expected_len].chunks_exact(3) {
                        rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
                    }
                }
                EyeFrameFormat::Gray8 => {
                    for value in &frame.bytes[..expected_len] {
                        rgb.extend_from_slice(&[*value, *value, *value]);
                    }
                }
                EyeFrameFormat::Yuyv422 => {
                    rgb.extend(yuyv422_to_rgb(&frame.bytes[..expected_len]));
                }
                _ => {}
            }
            let mut png = Vec::new();
            let result = PngEncoder::new(&mut png).write_image(
                &rgb,
                frame.width,
                frame.height,
                ColorType::Rgb8.into(),
            );
            match result {
                Ok(()) => {
                    let encoded = base64::engine::general_purpose::STANDARD.encode(png);
                    (Some(format!("data:image/png;base64,{encoded}")), None)
                }
                Err(error) => (None, Some(format!("failed to encode eye PNG: {error}"))),
            }
        }
        EyeFrameFormat::Unknown(ref format) => {
            (None, Some(format!("unsupported eye frame format {format}")))
        }
    }
}

fn yuyv422_to_rgb(bytes: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(bytes.len() / 2 * 3);
    for pair in bytes.chunks_exact(4) {
        let y0 = pair[0];
        let u = pair[1];
        let y1 = pair[2];
        let v = pair[3];
        push_yuv_rgb(&mut rgb, y0, u, v);
        push_yuv_rgb(&mut rgb, y1, u, v);
    }
    rgb
}

fn push_yuv_rgb(rgb: &mut Vec<u8>, y: u8, u: u8, v: u8) {
    let c = y as i32 - 16;
    let d = u as i32 - 128;
    let e = v as i32 - 128;
    rgb.push(((298 * c + 409 * e + 128) >> 8).clamp(0, 255) as u8);
    rgb.push(((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255) as u8);
    rgb.push(((298 * c + 516 * d + 128) >> 8).clamp(0, 255) as u8);
}

fn scene_kinect_from_snapshot(
    snapshot: &WorldSnapshot,
    calibration: Option<SceneSensorCalibration>,
    warnings: &mut Vec<String>,
) -> SceneKinect {
    let points = depth_points(&snapshot.kinect.depth_m, calibration);
    if points.is_empty() {
        warnings.push("no point cloud stream".to_string());
    }
    let coordinate_system = if snapshot.kinect.depth_m.len()
        == calibration.map_or(32, |c| c.compact_depth_beam_count)
    {
        "robot".to_string()
    } else {
        "camera".to_string()
    };
    SceneKinect {
        points,
        skeletons: snapshot.kinect.skeletons.clone(),
        coordinate_system: Some(coordinate_system),
    }
}

fn depth_points(depth_m: &[f32], calibration: Option<SceneSensorCalibration>) -> Vec<ScenePoint> {
    const MAX_POINTS: usize = 2_000;
    if depth_m.is_empty() {
        return Vec::new();
    }
    if let Some(calibration) = calibration {
        if depth_m.len() == calibration.compact_depth_beam_count {
            return range_beam_points(depth_m, calibration);
        }
    }
    let width = (depth_m.len() as f32).sqrt().ceil().max(1.0) as usize;
    let height = depth_m.len().div_ceil(width).max(1);
    let stride = (depth_m.len().div_ceil(MAX_POINTS)).max(1);
    depth_m
        .iter()
        .enumerate()
        .step_by(stride)
        .filter_map(|(index, depth)| {
            if !depth.is_finite() || *depth <= 0.0 {
                return None;
            }
            let x = index % width;
            let y = index / width;
            let nx = (x as f32 / width as f32) - 0.5;
            let ny = (y as f32 / height as f32) - 0.5;
            let z = (depth * calibration.map_or(1.0, |calibration| calibration.depth_scale))
                .clamp(0.0, 8.0);
            let shade = ((1.0 - (z / 8.0)).clamp(0.15, 1.0) * 255.0) as u8;
            Some(ScenePoint {
                x: nx * z,
                y: ny * z,
                z,
                r: shade,
                g: shade,
                b: shade,
            })
        })
        .collect()
}

fn range_beam_points(depth_m: &[f32], calibration: SceneSensorCalibration) -> Vec<ScenePoint> {
    let beam_count = depth_m.len().max(1);
    let fov_rad = calibration
        .compact_depth_fov_rad
        .clamp(0.01, std::f32::consts::TAU);
    let start = if beam_count == 1 { 0.0 } else { -fov_rad * 0.5 };
    let step = if beam_count == 1 {
        0.0
    } else {
        fov_rad / (beam_count - 1) as f32
    };
    depth_m
        .iter()
        .enumerate()
        .filter_map(|(index, depth)| {
            if !depth.is_finite() || *depth <= 0.0 {
                return None;
            }
            let distance = (depth * calibration.depth_scale).clamp(0.0, 8.0);
            let angle = start + step * index as f32;
            let shade = ((1.0 - (distance / 8.0)).clamp(0.15, 1.0) * 255.0) as u8;
            Some(ScenePoint {
                x: angle.sin() * distance,
                y: calibration.point_y_m,
                z: angle.cos() * distance,
                r: shade,
                g: shade,
                b: shade,
            })
        })
        .collect()
}

fn audio_bearing_from_objects(
    robot_x_m: f32,
    robot_y_m: f32,
    metadata: Option<&LiveSceneMetadata>,
) -> Option<f32> {
    metadata
        .into_iter()
        .flat_map(|metadata| metadata.objects.iter())
        .find(|object| {
            object.kind == "person" || object.kind == "speaker" || object.kind == "sound_source"
        })
        .map(|object| (object.y_m - robot_y_m).atan2(object.x_m - robot_x_m))
}

fn scene_object_kind(debug_kind: &str) -> String {
    let lower = debug_kind.to_ascii_lowercase();
    if lower.contains("charger") {
        "charger"
    } else if lower.contains("person") {
        "person"
    } else if lower.contains("sound") || lower.contains("speaker") {
        "speaker"
    } else if lower.contains("landmark") {
        "landmark"
    } else {
        "obstacle"
    }
    .to_string()
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

#[derive(Debug)]
pub struct LiveViewError {
    status: StatusCode,
    message: String,
}

impl LiveViewError {
    fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
        }
    }

    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
        }
    }

    fn forbidden(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            message: message.into(),
        }
    }
}

impl IntoResponse for LiveViewError {
    fn into_response(self) -> axum::response::Response {
        (
            self.status,
            Json(serde_json::json!({
                "error": self.message,
            })),
        )
            .into_response()
    }
}

async fn post_reign_command(
    State(state): State<ReignServerState>,
    Json(request): Json<ReignCommandRequest>,
) -> Result<Json<ReignInput>, ReignApiError> {
    let now_ms = wall_now_ms();
    let command = sanitize_command(request.command)?;
    let ttl_ms = request.ttl_ms.unwrap_or_else(|| command.default_ttl_ms());
    let input = ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: now_ms,
        expires_at_ms: now_ms.saturating_add(ttl_ms.clamp(100, 30_000)),
        source: request.source.unwrap_or(ReignSource::WebRemote),
        mode: request.mode,
        command,
        priority: request.priority.clamp(0.0, 1.0),
        note: request.note.filter(|note| !note.trim().is_empty()),
    };
    state
        .queue
        .lock()
        .map_err(|_| ReignApiError::internal("reign queue lock poisoned"))?
        .push(input.clone());
    Ok(Json(input))
}

async fn post_reign_prod(
    State(state): State<ReignServerState>,
    request: Option<Json<ProdRequest>>,
) -> Result<Json<ProdResponse>, ReignApiError> {
    let request = request.map(|Json(request)| request).unwrap_or_default();
    let snapshot = state.latest_snapshot().ok_or_else(|| {
        ReignApiError::bad_request("cannot prod before a live scene snapshot exists")
    })?;
    let command = prod_command_from_request(request)?;
    let action = command
        .to_action()
        .ok_or_else(|| ReignApiError::bad_request("prod command cannot be converted to action"))?;
    let policy = NudgePolicy::virtual_default();
    if let Some(reason) = nudge_action_block_reason_for_snapshot(&snapshot, &action, policy) {
        return Err(ReignApiError::forbidden(format!("prod refused: {reason}")));
    }

    let now_ms = wall_now_ms();
    let ttl_ms = command.default_ttl_ms().clamp(500, 5_000);
    let input = ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: now_ms,
        expires_at_ms: now_ms.saturating_add(ttl_ms),
        source: ReignSource::HumanSupervisor,
        mode: ReignMode::Assist,
        command,
        priority: 0.8,
        note: Some("manual prod".to_string()),
    };
    state
        .queue
        .lock()
        .map_err(|_| ReignApiError::internal("reign queue lock poisoned"))?
        .push(input.clone());
    Ok(Json(ProdResponse {
        accepted: true,
        input,
    }))
}

async fn get_reign_state(
    State(state): State<ReignServerState>,
) -> Result<Json<ReignSense>, ReignApiError> {
    let now_ms = wall_now_ms();
    let mut queue = state
        .queue
        .lock()
        .map_err(|_| ReignApiError::internal("reign queue lock poisoned"))?;
    queue.drain_expired(now_ms);
    Ok(Json(queue.sense(now_ms)))
}

fn prod_command_from_request(request: ProdRequest) -> Result<ReignCommand, ReignApiError> {
    let duration_ms = request.duration_ms.unwrap_or(1_500).clamp(250, 3_000);
    let intensity = match request.intensity {
        Some(value) if value.is_finite() => value,
        Some(_) => return Err(ReignApiError::bad_request("intensity must be finite")),
        None => 0.12,
    };
    let intensity = intensity.clamp(0.0, 0.25);
    let kind = request
        .kind
        .as_deref()
        .unwrap_or("explore")
        .trim()
        .to_ascii_lowercase();
    let command = match kind.as_str() {
        "" | "explore" | "random_walk" | "random-walk" => ReignCommand::Explore { duration_ms },
        "go" | "forward" => ReignCommand::Go {
            intensity: intensity.min(0.15),
            duration_ms: duration_ms.min(1_000),
        },
        "reverse" | "back" | "backward" => ReignCommand::Reverse {
            intensity: intensity.min(0.15),
            duration_ms: duration_ms.min(1_000),
        },
        "turn" | "left" => ReignCommand::Turn {
            direction: TurnDir::Left,
            intensity: intensity.min(0.25),
            duration_ms: duration_ms.min(1_000),
        },
        "right" => ReignCommand::Turn {
            direction: TurnDir::Right,
            intensity: intensity.min(0.25),
            duration_ms: duration_ms.min(1_000),
        },
        other => {
            return Err(ReignApiError::bad_request(format!(
                "unsupported prod kind '{other}'"
            )))
        }
    };
    sanitize_command(command)
}

async fn post_reign_clear(
    State(state): State<ReignServerState>,
) -> Result<Json<ReignSense>, ReignApiError> {
    let now_ms = wall_now_ms();
    let mut queue = state
        .queue
        .lock()
        .map_err(|_| ReignApiError::internal("reign queue lock poisoned"))?;
    queue.clear();
    Ok(Json(queue.sense(now_ms)))
}

async fn reign_page() -> Html<&'static str> {
    Html(REIGN_PAGE)
}

#[derive(Debug)]
pub struct ReignApiError {
    status: StatusCode,
    message: String,
}

impl ReignApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }

    fn forbidden(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            message: message.into(),
        }
    }
}

impl IntoResponse for ReignApiError {
    fn into_response(self) -> axum::response::Response {
        (
            self.status,
            Json(serde_json::json!({
                "error": self.message,
            })),
        )
            .into_response()
    }
}

fn sanitize_command(command: ReignCommand) -> Result<ReignCommand, ReignApiError> {
    Ok(match command {
        ReignCommand::Go {
            intensity,
            duration_ms,
        } => ReignCommand::Go {
            intensity: finite_intensity(intensity)?,
            duration_ms: duration_ms.clamp(50, 10_000),
        },
        ReignCommand::Reverse {
            intensity,
            duration_ms,
        } => ReignCommand::Reverse {
            intensity: finite_intensity(intensity)?,
            duration_ms: duration_ms.clamp(50, 10_000),
        },
        ReignCommand::Turn {
            direction,
            intensity,
            duration_ms,
        } => ReignCommand::Turn {
            direction,
            intensity: finite_intensity(intensity)?,
            duration_ms: duration_ms.clamp(50, 10_000),
        },
        ReignCommand::Explore { duration_ms } => ReignCommand::Explore {
            duration_ms: duration_ms.clamp(250, 30_000),
        },
        ReignCommand::Speak { text } => {
            let text = text.trim();
            if text.is_empty() {
                return Err(ReignApiError::bad_request("speak text cannot be empty"));
            }
            ReignCommand::Speak {
                text: text.chars().take(500).collect(),
            }
        }
        other => other,
    })
}

fn finite_intensity(value: f32) -> Result<f32, ReignApiError> {
    if !value.is_finite() {
        return Err(ReignApiError::bad_request("intensity must be finite"));
    }
    Ok(value.clamp(0.0, 1.0))
}

fn default_priority() -> f32 {
    0.8
}

fn default_reign_mode() -> ReignMode {
    ReignMode::Direct
}

fn wall_now_ms() -> TimeMs {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as TimeMs)
        .unwrap_or_default()
}

const REIGN_PAGE: &str = r#"<!doctype html>
<meta charset="utf-8">
<title>Netherwick Reign</title>
<style>
body{font:16px system-ui;margin:2rem;max-width:780px}
button,input,select{font:inherit;margin:.25rem;padding:.5rem}
button.stop{background:#b00020;color:white}
#latest{white-space:pre-wrap;background:#f5f5f5;padding:1rem}
</style>
<h1>Reign Remote</h1>
<select id="mode">
  <option>Direct</option><option>Assist</option><option>Suggest</option><option>ObserveOnly</option>
</select>
<button class="stop" onclick="send({type:'Stop'},2000)">STOP</button>
<button onclick="send({type:'Go',intensity:.4,duration_ms:700},1200)">Forward</button>
<button onclick="send({type:'Reverse',intensity:.4,duration_ms:700},1200)">Reverse</button>
<button onclick="send({type:'Turn',direction:'Left',intensity:.5,duration_ms:500},1000)">Turn Left</button>
<button onclick="send({type:'Turn',direction:'Right',intensity:.5,duration_ms:500},1000)">Turn Right</button>
<button onclick="send({type:'Dock'},5000)">Dock</button>
<button onclick="send({type:'Explore',duration_ms:3000},5000)">Explore</button>
<input id="say" placeholder="Speak text"><button onclick="speak()">Speak</button>
<pre id="latest">loading...</pre>
<script>
async function send(command, ttl_ms){
  await fetch('/reign/command',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({mode:mode.value,priority:1,ttl_ms,command})});
  refresh();
}
function speak(){ const text = say.value.trim(); if(text) send({type:'Speak',text},10000); }
async function refresh(){ latest.textContent = JSON.stringify(await (await fetch('/reign/state')).json(), null, 2); }
addEventListener('keydown', e => {
  if(e.repeat) return;
  if(e.code === 'Space') send({type:'Stop'},2000);
  if(e.key === 'w') send({type:'Go',intensity:.4,duration_ms:700},1200);
  if(e.key === 'a') send({type:'Turn',direction:'Left',intensity:.5,duration_ms:500},1000);
  if(e.key === 'd') send({type:'Turn',direction:'Right',intensity:.5,duration_ms:500},1000);
  if(e.key === 's') send({type:'Stop'},2000);
  if(e.key === 'c') send({type:'Dock'},5000);
  if(e.key === 'e') send({type:'Explore',duration_ms:3000},5000);
});
refresh(); setInterval(refresh, 1000);
</script>"#;

const LIVE_VIEW_PAGE: &str = r#"<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Netherwick Robot View</title>
<style>
:root{color-scheme:dark;background:#101214;color:#eef1f3;font:14px system-ui}
body{margin:0;min-height:100vh;display:grid;grid-template-columns:minmax(0,1fr) 320px}
main{display:grid;place-items:center;background:#070809}
canvas{width:min(100vw,calc((100vh - 28px)*1.333));height:auto;image-rendering:pixelated;background:#111;border:1px solid #34383c}
aside{border-left:1px solid #2f3337;padding:16px;background:#171a1d}
h1{font-size:16px;margin:0 0 14px}
a{color:#8bd3ff}
dl{display:grid;grid-template-columns:auto 1fr;gap:8px 12px;margin:0 0 18px}
dt{color:#9aa4ad}
dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
#status{color:#ffcf7a;margin-bottom:12px;min-height:20px}
.beams{display:flex;align-items:end;gap:5px;height:96px;margin-top:8px}
.beam{flex:1;background:#60d394;min-height:2px}
@media(max-width:760px){body{display:block}aside{border-left:0;border-top:1px solid #2f3337}canvas{width:100vw}}
</style>
<main><canvas id="eye" width="64" height="48"></canvas></main>
<aside>
  <h1>Robot View</h1>
  <p><a href="/view/3d">Open 3D sensorium</a></p>
  <div id="status">waiting for frames...</div>
  <dl>
    <dt>t</dt><dd id="t">-</dd>
    <dt>x</dt><dd id="x">-</dd>
    <dt>y</dt><dd id="y">-</dd>
    <dt>heading</dt><dd id="heading">-</dd>
    <dt>battery</dt><dd id="battery">-</dd>
    <dt>stuck</dt><dd id="stuck">-</dd>
    <dt>nearest</dt><dd id="nearest">-</dd>
    <dt>gps lat</dt><dd id="gps_lat">-</dd>
    <dt>gps lon</dt><dd id="gps_lon">-</dd>
    <dt>gps alt</dt><dd id="gps_alt">-</dd>
    <dt>eye format</dt><dd id="eye_format">-</dd>
    <dt>eye age</dt><dd id="eye_age">-</dd>
    <dt>ear age</dt><dd id="ear_age">-</dd>
  </dl>
  <div>Range</div>
  <div class="beams" id="beams"></div>
</aside>
<script>
const canvas = document.getElementById('eye');
const ctx = canvas.getContext('2d');
const fields = Object.fromEntries(['t','x','y','heading','battery','nearest','gps_lat','gps_lon','gps_alt','eye_format','eye_age','ear_age'].map(id => [id, document.getElementById(id)]));
const status = document.getElementById('status');
const beams = document.getElementById('beams');
function fmt(value, digits = 2){
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '-';
}
function drawEye(frame){
  if(!frame) return;
  const fmt = frame.format;
  const isRgb = fmt === 'Rgb8' || (typeof fmt === 'object' && fmt.Rgb8 !== undefined);
  const isBgr = fmt === 'Bgr8' || (typeof fmt === 'object' && fmt.Bgr8 !== undefined);
  const isGray = fmt === 'Gray8' || (typeof fmt === 'object' && fmt.Gray8 !== undefined);
  const isYuyv = fmt === 'Yuyv422' || (typeof fmt === 'object' && fmt.Yuyv422 !== undefined);
  const isMjpg = fmt === 'Mjpeg' || (typeof fmt === 'object' && fmt.Mjpeg !== undefined) || (typeof fmt === 'object' && JSON.stringify(fmt).includes('MJPG'));
  if(isRgb || isBgr || isGray || isYuyv){
    if(canvas.width !== frame.width || canvas.height !== frame.height){
      canvas.width = frame.width; canvas.height = frame.height;
    }
    const image = ctx.createImageData(frame.width, frame.height);
    if(isGray){
      for(let source = 0, target = 0; target < image.data.length && source < frame.bytes.length; source += 1, target += 4){
        const value = frame.bytes[source];
        image.data[target] = value;
        image.data[target + 1] = value;
        image.data[target + 2] = value;
        image.data[target + 3] = 255;
      }
    }else if(isYuyv){
      for(let source = 0, target = 0; target + 7 < image.data.length && source + 3 < frame.bytes.length; source += 4, target += 8){
        const y0 = frame.bytes[source], u = frame.bytes[source + 1], y1 = frame.bytes[source + 2], v = frame.bytes[source + 3];
        writeYuvPixel(image.data, target, y0, u, v);
        writeYuvPixel(image.data, target + 4, y1, u, v);
      }
    }else{
      for(let source = 0, target = 0; target < image.data.length && source + 2 < frame.bytes.length; source += 3, target += 4){
        image.data[target] = isBgr ? frame.bytes[source + 2] : frame.bytes[source];
        image.data[target + 1] = frame.bytes[source + 1];
        image.data[target + 2] = isBgr ? frame.bytes[source] : frame.bytes[source + 2];
        image.data[target + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  } else if(isMjpg){
    const blob = new Blob([new Uint8Array(frame.bytes)], {type: 'image/jpeg'});
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      if(canvas.width !== img.width || canvas.height !== img.height){
        canvas.width = img.width; canvas.height = img.height;
      }
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
    };
    img.src = url;
  }
}
function writeYuvPixel(data, target, y, u, v){
  const c = y - 16, d = u - 128, e = v - 128;
  data[target] = Math.max(0, Math.min(255, (298 * c + 409 * e + 128) >> 8));
  data[target + 1] = Math.max(0, Math.min(255, (298 * c - 100 * d - 208 * e + 128) >> 8));
  data[target + 2] = Math.max(0, Math.min(255, (298 * c + 516 * d + 128) >> 8));
  data[target + 3] = 255;
}
function shouldGenerateEye(scenePacket){
  const session = scenePacket?.session || {};
  const virtualMode = session.source === 'sim' || String(session.mode || '').includes('virtual');
  if(!virtualMode) return false;
  if(scenePacket?.eye?.source === 'babylon-robot-eye') return false;
  const eye = scenePacket?.eye;
  return !eye || !eye.data_url || eye.mean_luma < .08 || eye.non_background_ratio < .01;
}
function colorForObject(item){
  if(item.color_rgb) return `rgb(${item.color_rgb[0]},${item.color_rgb[1]},${item.color_rgb[2]})`;
  return {charger:'#45d483', person:'#e8c08c', speaker:'#778cff', obstacle:'#d67666', landmark:'#b0b8c0'}[item.kind] || '#b0b8c0';
}
function drawGeneratedEye(scenePacket){
  if(!scenePacket?.body) return false;
  const width = 160, height = 120, horizon = 54;
  if(canvas.width !== width || canvas.height !== height){
    canvas.width = width; canvas.height = height;
  }
  const g = ctx.createLinearGradient(0, 0, 0, horizon);
  g.addColorStop(0, '#253646');
  g.addColorStop(1, '#55636d');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, width, horizon);
  const floor = ctx.createLinearGradient(0, horizon, 0, height);
  floor.addColorStop(0, '#555954');
  floor.addColorStop(1, '#262b29');
  ctx.fillStyle = floor;
  ctx.fillRect(0, horizon, width, height - horizon);
  ctx.strokeStyle = 'rgba(255,255,255,.16)';
  ctx.lineWidth = 1;
  for(let y = horizon + 10; y < height; y += Math.max(5, (y - horizon) * .22)){
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
  }
  const body = scenePacket.body;
  const fov = Math.PI * .62;
  const visible = (scenePacket.objects || []).map(item => {
    const dx = item.x_m - body.x_m, dy = item.y_m - body.y_m;
    const forward = dx * Math.cos(body.heading_rad) + dy * Math.sin(body.heading_rad);
    const lateral = -dx * Math.sin(body.heading_rad) + dy * Math.cos(body.heading_rad);
    const angle = Math.atan2(lateral, forward);
    return {...item, forward, angle};
  }).filter(item => item.forward > .05 && Math.abs(item.angle) < fov * .55)
    .sort((a, b) => b.forward - a.forward);
  for(const item of visible){
    const cx = width * .5 + (item.angle / (fov * .5)) * width * .5;
    const near = Math.max(.18, item.forward);
    const radiusPx = Math.max(5, item.radius_m / near * 58);
    const h = Math.max(8, radiusPx * ({person:4.2, speaker:2.0, charger:.9, landmark:2.4}[item.kind] || 2.2));
    const y = horizon + Math.min(45, 28 / near);
    ctx.fillStyle = colorForObject(item);
    ctx.strokeStyle = 'rgba(0,0,0,.45)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    if(item.kind === 'person'){
      ctx.ellipse(cx, y - h * .45, radiusPx * .8, h * .5, 0, 0, Math.PI * 2);
    }else if(item.kind === 'speaker'){
      ctx.moveTo(cx, y - h);
      ctx.lineTo(cx + radiusPx * 1.3, y);
      ctx.lineTo(cx - radiusPx * 1.3, y);
      ctx.closePath();
    }else{
      ctx.rect(cx - radiusPx, y - h, radiusPx * 2, h);
    }
    ctx.fill();
    ctx.stroke();
  }
  ctx.strokeStyle = 'rgba(255,207,102,.9)';
  ctx.beginPath();
  ctx.moveTo(width / 2 - 8, height / 2); ctx.lineTo(width / 2 + 8, height / 2);
  ctx.moveTo(width / 2, height / 2 - 8); ctx.lineTo(width / 2, height / 2 + 8);
  ctx.stroke();
  return true;
}
function drawBeams(values){
  beams.replaceChildren(...(values || []).map(value => {
    const bar = document.createElement('div');
    bar.className = 'beam';
    bar.style.height = `${Math.max(4, value * 96)}px`;
    bar.style.opacity = `${0.35 + (1 - value) * 0.65}`;
    return bar;
  }));
}
async function refresh(){
  try{
    const [response, sceneResponse] = await Promise.all([
      fetch('/view/snapshot', {cache:'no-store'}),
      fetch('/view/scene', {cache:'no-store'})
    ]);
    if(!response.ok) throw new Error(await response.text());
    const snapshot = await response.json();
    const scenePacket = sceneResponse.ok ? await sceneResponse.json() : null;
    const body = snapshot.body;
    fields.t.textContent = `${snapshot.t_ms} ms`;
    fields.x.textContent = `${fmt(body.odometry.x_m)} m`;
    fields.y.textContent = `${fmt(body.odometry.y_m)} m`;
    fields.heading.textContent = `${fmt(body.odometry.heading_rad)} rad`;
    fields.battery.textContent = `${fmt(body.battery_level * 100, 1)}%`;
    fields.nearest.textContent = snapshot.range.nearest_m == null ? '-' : `${fmt(snapshot.range.nearest_m)} m`;
    const gps = snapshot.gps;
    if(gps){
      fields.gps_lat.textContent = fmt(gps.lat, 6);
      fields.gps_lon.textContent = fmt(gps.lon, 6);
      fields.gps_alt.textContent = gps.altitude_m != null ? `${fmt(gps.altitude_m, 1)} m` : '-';
    }else{
      fields.gps_lat.textContent = '-';
      fields.gps_lon.textContent = '-';
      fields.gps_alt.textContent = '-';
    }
    if(snapshot.eye_frame){
      let fmt_str = typeof snapshot.eye_frame.format === 'string' ? snapshot.eye_frame.format : JSON.stringify(snapshot.eye_frame.format);
      if(scenePacket && scenePacket.eye){
        const eye = scenePacket.eye;
        const isAuth = eye.authoritative ? " (auth)" : " (symbolic)";
        const statusText = eye.retina_connected ? `connected, age ${eye.retina_last_frame_age_ms}ms` : "disconnected";
        fields.eye_format.textContent = `${eye.width}x${eye.height} luma ${fmt(eye.mean_luma, 2)} | source: ${eye.source}${isAuth} | retina: ${statusText} | rx: ${eye.frames_received} tx: ${eye.frames_written_to_ledger}`;
        if (eye.source === 'babylon-robot-eye' && !eye.retina_connected) {
          fields.eye_format.style.color = '#ff4444';
          fields.eye_format.style.fontWeight = 'bold';
        } else {
          fields.eye_format.style.color = '';
          fields.eye_format.style.fontWeight = '';
        }
      } else {
        const src = snapshot.eye_frame.source || 'none';
        fields.eye_format.textContent = `${fmt_str} (src: ${src})`;
        fields.eye_format.style.color = '';
        fields.eye_format.style.fontWeight = '';
      }
      fields.eye_age.textContent = `${snapshot.t_ms - snapshot.eye_frame.captured_at_ms} ms`;
    }else{
      fields.eye_format.textContent = '-';
      fields.eye_format.style.color = '';
      fields.eye_format.style.fontWeight = '';
      fields.eye_age.textContent = '-';
    }
    if(snapshot.ear_pcm){
      fields.ear_age.textContent = `${snapshot.t_ms - snapshot.ear_pcm.captured_at_ms} ms`;
    }else{
      fields.ear_age.textContent = '-';
    }
    drawEye(snapshot.eye_frame);
    const generated = shouldGenerateEye(scenePacket) && drawGeneratedEye(scenePacket);
    drawBeams(snapshot.range.beams);
    status.textContent = generated ? 'live generated eye' : 'live';
  }catch(error){
    status.textContent = 'waiting for frames...';
  }finally{
    setTimeout(refresh, 100);
  }
}
refresh();
</script>"#;

const LIVE_VIEW_3D_PAGE: &str = r#"<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Netherwick Dream World</title>
<style>
:root{color-scheme:dark;background:#0b0d10;color:#eef1f3;font:13px system-ui}
html,body,#scene{width:100%;height:100%;margin:0;overflow:hidden}
.panel-window{position:fixed;box-sizing:border-box;display:flex;flex-direction:column;min-width:220px;min-height:84px;padding:0;border:1px solid #34414d;background:rgba(10,13,17,.86);backdrop-filter:blur(10px);border-radius:7px;color:#e7eef5;resize:both;overflow:hidden;box-shadow:0 16px 40px rgba(0,0,0,.28),inset 0 1px 0 rgba(255,255,255,.05);transition:height .24s ease,min-height .24s ease,border-color .18s ease,box-shadow .18s ease,background .18s ease}
.panel-window:hover{border-color:#4d6275;box-shadow:0 18px 48px rgba(0,0,0,.34),0 0 0 1px rgba(121,190,242,.08),inset 0 1px 0 rgba(255,255,255,.07)}
.panel-titlebar{flex:0 0 auto;display:flex;align-items:center;min-height:31px;padding:0 10px;border-bottom:1px solid rgba(118,144,166,.28);background:linear-gradient(180deg,rgba(38,55,68,.92),rgba(18,27,36,.9));color:#c6e6ff;font-weight:700;font-size:12px;letter-spacing:0;cursor:move;user-select:none;touch-action:none}
.panel-titlebar::before{content:"";width:8px;height:8px;margin-right:8px;border-radius:50%;background:#6fc1ff;box-shadow:0 0 12px rgba(111,193,255,.5)}
.panel-titlebar::after{content:"double-click to shade";margin-left:auto;color:#8fa1b2;font-size:10px;font-weight:500;opacity:0;transition:opacity .16s ease}
.panel-window:hover .panel-titlebar::after{opacity:.72}
.panel-content{flex:1 1 auto;min-height:0;padding:10px;overflow:auto;transition:max-height .24s ease,opacity .2s ease,padding .24s ease}
.panel-window.is-shaded{height:32px!important;min-height:32px!important;max-height:32px!important;resize:none;background:rgba(12,18,24,.92)}
.panel-window.is-shaded .panel-content{max-height:0;opacity:0;padding-top:0;padding-bottom:0;pointer-events:none;overflow:hidden}
.panel-window.is-shaded .panel-titlebar{border-bottom-color:transparent}
.drag-handle{cursor:move;user-select:none}
#hud{left:12px;top:12px;min-width:240px;border-color:#2b3138}
#hud .panel-content{display:grid;gap:6px}
#hud h1{font-size:14px;margin:0}
#hud .panel-content>h1,#reign .panel-content>strong,#virtual-pipeline-section .panel-content>h2,#model-graph-window .panel-content>section>h2{display:none}
#hud dl{display:grid;grid-template-columns:auto 1fr;gap:4px 10px;margin:0}
#hud dt{color:#aab4bd}
#hud dd{margin:0;text-align:right;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
#hud dd.active-learning{color:#8df0b2;font-weight:700}
#status{color:#ffd083}
#reign{right:12px;top:12px;width:236px;min-width:220px;border-color:#36424d;background:rgba(11,16,22,.84);color:#dce8f2}
#reign .panel-content{display:grid;gap:8px}
#reign strong{display:block;font-size:12px;margin-bottom:6px;color:#91d7ff}
#reign div{font-variant-numeric:tabular-nums}
#reign-state{min-height:17px;color:#ffd083}
.reign-pad{display:grid;grid-template-columns:92px 1fr;gap:10px;align-items:center}
#reign-joystick{position:relative;width:92px;aspect-ratio:1;border:1px solid #455565;border-radius:50%;background:radial-gradient(circle at 50% 50%,rgba(79,149,188,.24),rgba(15,23,31,.72) 64%);touch-action:none;cursor:pointer}
#reign-joystick::before,#reign-joystick::after{content:"";position:absolute;background:rgba(170,196,216,.22)}
#reign-joystick::before{left:12px;right:12px;top:50%;height:1px}
#reign-joystick::after{top:12px;bottom:12px;left:50%;width:1px}
#reign-stick{position:absolute;left:50%;top:50%;width:34px;height:34px;border-radius:50%;background:#dce8f2;border:1px solid #8ec9ef;box-shadow:0 0 18px rgba(111,191,242,.42);transform:translate(-50%,-50%)}
.reign-buttons{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}
.reign-buttons button{min-height:34px;border:1px solid #3f607b;background:#1a2b38;color:#dce8f2;border-radius:5px;cursor:pointer}
.reign-buttons button:hover,.reign-buttons button:focus-visible{border-color:#8ec9ef;color:#fff;outline:none}
.reign-buttons .stop{grid-column:1/-1;background:#5b1720;border-color:#9a3443;color:#fff}
.reign-hint{color:#9fb0bf;font-size:11px;line-height:1.25}
#llm{right:12px;top:96px;width:320px;height:300px;min-width:200px;min-height:150px;border-color:#3f4c58;background:rgba(9,12,16,.88)}
#llm .panel-content{display:grid;grid-template-rows:auto minmax(0,1fr);gap:8px;overflow:hidden}
#llm header{display:flex;align-items:center;justify-content:space-between;gap:12px}
#llm h2{font-size:12px;margin:0;color:#b6e0ff}
#llm-status{color:#ffcf7a;font-size:12px;font-variant-numeric:tabular-nums}
#llm-streams{display:grid;gap:8px;min-height:0;overflow:auto;scrollbar-width:thin}
.llm-card{display:grid;gap:6px;padding:8px;border:1px solid #2c3640;background:rgba(19,25,32,.78);border-radius:5px}
.llm-card.live{border-color:#5a7892}
.llm-card.error{border-color:#a65454}
.llm-top{display:flex;align-items:center;justify-content:space-between;gap:10px;font-size:12px}
.llm-title{color:#ffffff;font-weight:700;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.llm-phase{color:#9fb0bf;font-variant-numeric:tabular-nums;white-space:nowrap}
.llm-block{display:grid;gap:3px;min-width:0}
.llm-label{color:#8fa1b2;font-size:11px;text-transform:uppercase}
.llm-text{max-height:112px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere;font:11px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;line-height:1.35;color:#dbe7f1;background:rgba(4,7,10,.38);border-radius:4px;padding:6px}
#virtual-pipeline-section{left:374px;top:318px;width:300px;height:230px;min-width:260px;min-height:140px;border-color:#37424c;background:rgba(8,11,15,.88)}
#virtual-pipeline-section .panel-content{display:grid;gap:8px}
#models{left:12px;bottom:40px;width:310px;height:320px;min-width:260px;min-height:180px;border-color:#37424c;background:rgba(8,11,15,.88)}
#models .panel-content{display:flex;flex-direction:column;gap:8px;overflow:hidden}
#model-graph-window{left:336px;bottom:40px;width:430px;height:420px;min-width:320px;min-height:260px;border-color:#37424c;background:rgba(8,11,15,.88)}
#model-graph-window .panel-content{display:flex;flex-direction:column;gap:8px;overflow:hidden}
#models section,#model-graph-window section{display:flex;flex-direction:column;min-height:0;flex:1}
#models h2,#model-graph-window h2,#virtual-pipeline-section h2{font-size:12px;margin:0 0 8px;color:#c6e6ff}
#model-status{color:#ffcf7a;font-size:12px;font-variant-numeric:tabular-nums}
#model-list{flex:1;display:grid;gap:6px;min-height:0;overflow-y:auto;scrollbar-width:thin}
.model-row{display:grid;grid-template-columns:1fr auto;gap:3px 8px;padding:6px;border:1px solid #27313b;background:rgba(18,24,30,.72);border-radius:5px}
.model-name{font-weight:700;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.model-pill{padding:1px 5px;border-radius:999px;background:#243646;color:#b8e1ff;font-size:11px;white-space:nowrap}
.model-line{grid-column:1/-1;color:#aeb9c3;font-size:11px;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
#model-graph{width:100%;flex:1;min-height:0;border:1px solid #27313b;border-radius:5px;background:rgba(4,7,10,.32)}
#model-graph text{font:10px system-ui;fill:#e7eef5}
#model-graph .edge{stroke:#627384;stroke-width:1.2;fill:none;marker-end:url(#arrow)}
#model-graph .edge-label{fill:#99a8b5;font-size:8px}
#model-graph .node rect{fill:#17222b;stroke:#52687b;stroke-width:1.2;rx:5}
#model-graph .node.model rect{stroke:#81c995}
#model-graph .node.core rect{stroke:#8bd3ff}
#model-graph .node{cursor:pointer}
#model-graph .node.hardcoded rect{stroke:#8bd3ff}
#model-graph .node.shadow_train rect{stroke:#ffcf66}
#model-graph .node.shadow_infer rect{stroke:#d6b5ff}
#model-graph .node.model_infer rect,#model-graph .node.model_train_and_infer rect{stroke:#81c995}
#model-graph .node.compare rect{stroke:#ff9f6e}
#model-graph .node.missing rect{stroke:#f06d6d;stroke-dasharray:4 3}
#model-graph .node.selected rect{stroke-width:2.4}
#behavior-inspector{display:grid;grid-template-columns:1fr 1fr;gap:6px;border:1px solid #27313b;border-radius:5px;background:rgba(4,7,10,.28);padding:8px;min-height:112px}
#behavior-inspector .wide{grid-column:1/-1}
#behavior-inspector label{display:grid;gap:3px;color:#8fa1b2;font-size:10px;text-transform:uppercase}
#behavior-inspector select,#behavior-inspector input{width:100%;min-width:0;background:#111820;border:1px solid #2d3944;color:#e7eef5;border-radius:4px;padding:4px;font-size:11px}
#behavior-inspector input[type="checkbox"]{width:auto;justify-self:start;accent-color:#52a9ff}
#behavior-inspector .title{grid-column:1/-1;color:#fff;font-weight:700;font-size:12px;display:flex;justify-content:space-between;gap:8px}
#behavior-inspector .run{grid-column:1/-1;color:#9fb0bf;font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#calibration{left:12px;top:540px;width:340px;border-color:#3c4854}
#calibration .panel-content{display:grid;gap:6px}
#calibration header{display:flex;align-items:center;justify-content:space-between;gap:12px}
#calibration h2{font-size:12px;margin:0;color:#c6e6ff}
#calibration label{color:#8fa1b2;font-size:11px}
#calibration span{color:#ffd083;font-size:11px;font-variant-numeric:tabular-nums}
#calibration input[type="range"]{width:100%;accent-color:#52a9ff;background:#242a32;height:4px;border-radius:2px;outline:none}
#learning{left:12px;top:318px;width:340px;border-color:#3c4854}
#learning .panel-content{display:grid;gap:8px}
#learning header{display:flex;align-items:center;justify-content:space-between;gap:12px}
#learning h2{font-size:12px;margin:0;color:#c6e6ff}
#learning-status{color:#ffcf7a;font-size:11px;font-variant-numeric:tabular-nums}
#learning .row{display:grid;grid-template-columns:86px 1fr;gap:8px;align-items:center}
#learning label{color:#8fa1b2;font-size:11px}
#learning select,#learning input[type="number"]{width:100%;box-sizing:border-box;background:#111820;border:1px solid #33414d;color:#e7eef5;border-radius:4px;padding:4px 6px}
#learning .checks{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 8px}
#learning .checks label{display:flex;align-items:center;gap:5px;color:#d7e3ee}
#learning button{justify-self:end;font-size:11px;padding:4px 8px;background:#243646;border:1px solid #3f607b;color:#b8e1ff;border-radius:4px;cursor:pointer}
#xr{position:fixed;right:12px;bottom:12px;padding:9px 12px;border:1px solid #405060;background:#15202b;color:#fff;border-radius:6px}
#xr[disabled]{opacity:.55}
#fallback{position:fixed;left:12px;bottom:12px;color:#aab4bd;max-width:min(520px,calc(100vw - 24px))}
@media(max-width:820px){.panel-window{max-width:calc(100vw - 24px)}#llm{left:12px;right:12px;top:auto;bottom:56px;width:auto;max-height:42vh}#reign{top:auto;bottom:12px;right:12px;min-width:0}.llm-text{max-height:76px}#models{bottom:98px;max-height:46vh}#model-graph-window{display:none}#virtual-pipeline-section{left:12px;right:12px;top:128px;width:auto;max-height:28vh}#model-graph{height:220px}#learning{left:12px;right:12px;top:220px;width:auto;max-height:34vh}#calibration{display:none}}
canvas{display:block}
</style>
<canvas id="scene"></canvas>
<aside id="hud" data-window-title="Sensorium 3D">
  <h1>Sensorium 3D</h1>
  <div id="status">connecting...</div>
  <dl>
    <dt>mode</dt><dd id="mode">-</dd>
    <dt>scenario</dt><dd id="scenario">-</dd>
    <dt>training</dt><dd id="training_mode">-</dd>
    <dt>weights</dt><dd id="weights_updating">-</dd>
    <dt>ledger</dt><dd id="ledger_counts">-</dd>
    <dt>selector</dt><dd id="action_selector_mode">-</dd>
    <dt>chosen</dt><dd id="chosen_action">-</dd>
    <dt>motor</dt><dd id="motor_line">-</dd>
    <dt>motion sent</dt><dd id="motion_sent">-</dd>
    <dt>delta</dt><dd id="movement_delta">-</dd>
    <dt>blocked</dt><dd id="blocked_reason">-</dd>
    <dt>seed</dt><dd id="seed">-</dd>
    <dt>tick</dt><dd id="tick">-</dd>
    <dt>t</dt><dd id="t">-</dd>
    <dt>pose</dt><dd id="pose">-</dd>
    <dt>battery</dt><dd id="battery">-</dd>
    <dt>stuck</dt><dd id="stuck">-</dd>
    <dt>trap</dt><dd id="trap_kind">-</dd>
    <dt>dead battery</dt><dd id="dead_battery">-</dd>
    <dt>recovery</dt><dd id="recovery_mode">-</dd>
    <dt>attempts</dt><dd id="recovery_attempts">-</dd>
    <dt>stuck ticks</dt><dd id="stuck_ticks">-</dd>
    <dt>nearest</dt><dd id="nearest">-</dd>
    <dt>eye</dt><dd id="eye">-</dd>
    <dt>points</dt><dd id="points">-</dd>
    <dt>audio</dt><dd id="audio">-</dd>
    <dt>mind</dt><dd id="mind">-</dd>
    <dt>scheme</dt><dd id="scheme">-</dd>
    <dt>secure</dt><dd id="secure">-</dd>
    <dt>WebXR</dt><dd id="webxr">checking...</dd>
  </dl>
</aside>
<aside id="learning" data-window-title="Live learning">
  <header>
    <h2>Live learning</h2>
    <div id="learning-status">loading...</div>
  </header>
  <div class="row">
    <label for="learning-mode">mode</label>
    <select id="learning-mode">
      <option value="off">off</option>
      <option value="shadow-only">shadow-only</option>
      <option value="world-outcome">world-outcome</option>
    </select>
  </div>
  <div class="row">
    <label for="learning-steps">steps/tick</label>
    <input id="learning-steps" type="number" min="1" max="64" step="1" value="1">
  </div>
  <div class="checks" id="learning-behaviors">
    <label><input type="checkbox" data-behavior="danger">danger</label>
    <label><input type="checkbox" data-behavior="charge">charge</label>
    <label><input type="checkbox" data-behavior="future">future</label>
    <label><input type="checkbox" data-behavior="action_value">action value</label>
    <label><input type="checkbox" data-behavior="eye_next">eye next</label>
    <label><input type="checkbox" data-behavior="ear_next">ear next</label>
    <label><input type="checkbox" data-behavior="experience">experience</label>
  </div>
  <button id="learning-apply">Apply</button>
</aside>
<aside id="reign" data-window-title="Reign controls">
  <strong>Reign controls</strong>
  <div class="reign-pad">
    <div id="reign-joystick" role="application" aria-label="Hold and drag to steer">
      <div id="reign-stick"></div>
    </div>
    <div class="reign-buttons">
      <button class="stop" id="reign-stop" type="button">Stop</button>
      <button id="reign-dock" type="button">Dock</button>
      <button id="reign-explore" type="button">Explore</button>
    </div>
  </div>
  <div id="reign-state">web reign ready</div>
  <div class="reign-hint">Drag up to go, left or right to turn. Release to stop.</div>
</aside>
<aside id="llm" data-window-title="LLM streams">
  <header>
    <h2>LLM streams</h2>
    <div id="llm-status">connecting...</div>
  </header>
  <div id="llm-streams"></div>
</aside>
<aside id="virtual-pipeline-section" data-window-title="Dream World Training" style="display:none">
  <h2>Dream World Training</h2>
  <div id="virtual-report-summary" style="font-size: 0.85em; opacity: 0.9; margin-bottom: 8px; line-height: 1.4;"></div>
  <div id="virtual-model-recommendations" style="font-size: 0.85em;"></div>
</aside>
<aside id="models" data-window-title="Training stats">
  <section>
    <h2>Training stats <span id="model-status">loading...</span></h2>
    <div id="model-list"></div>
  </section>
</aside>
<aside id="model-graph-window" data-window-title="Connections">
  <section>
    <h2>Connections</h2>
    <svg id="model-graph" viewBox="0 0 560 260" role="img" aria-label="Model connection diagram"></svg>
    <div id="behavior-inspector" aria-live="polite"></div>
  </section>
</aside>
<aside id="calibration" data-window-title="Sensor Calibration">
  <header>
    <h2>Sensor Calibration</h2>
    <button id="reset-calibration" style="font-size:11px;padding:2px 6px;background:#243646;border:1px solid #36424d;color:#b8e1ff;border-radius:4px;cursor:pointer">Reset</button>
  </header>
  <div style="display:grid;gap:8px;margin-top:8px">
    <div style="border-bottom:1px solid #2b3138;padding-bottom:4px;font-weight:bold;color:#ffd083">Point Cloud</div>
    <div style="display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center">
      <label for="cal-depth-scale" style="width:90px">Depth Scale</label>
      <input type="range" id="cal-depth-scale" min="0.5" max="2.0" step="0.01" value="1.0">
      <span id="val-depth-scale" style="width:40px;text-align:right;font-family:monospace">1.00</span>
    </div>
    <div style="display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center">
      <label for="cal-point-y" style="width:90px">Depth Height</label>
      <input type="range" id="cal-point-y" min="-0.50" max="1.00" step="0.01" value="0.18">
      <span id="val-point-y" style="width:40px;text-align:right;font-family:monospace">0.18m</span>
    </div>
    <div style="display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center">
      <label for="cal-depth-fov" style="width:90px">Depth FOV</label>
      <input type="range" id="cal-depth-fov" min="30" max="180" step="1" value="122">
      <span id="val-depth-fov" style="width:40px;text-align:right;font-family:monospace">122°</span>
    </div>
    
    <div style="border-bottom:1px solid #2b3138;padding-bottom:4px;font-weight:bold;color:#ffd083;margin-top:6px">Vision / Camera</div>
    <div style="display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center">
      <label for="cal-camera-fov" style="width:90px">Camera FOV</label>
      <input type="range" id="cal-camera-fov" min="30" max="120" step="1" value="62">
      <span id="val-camera-fov" style="width:40px;text-align:right;font-family:monospace">62°</span>
    </div>
    <div style="display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center">
      <label for="cal-camera-y" style="width:90px">Cam Height (Y)</label>
      <input type="range" id="cal-camera-y" min="0.10" max="1.00" step="0.01" value="0.46">
      <span id="val-camera-y" style="width:40px;text-align:right;font-family:monospace">0.46m</span>
    </div>
    <div style="display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center">
      <label for="cal-camera-z" style="width:90px">Cam Depth (Z)</label>
      <input type="range" id="cal-camera-z" min="-0.50" max="0.50" step="0.01" value="-0.18">
      <span id="val-camera-z" style="width:40px;text-align:right;font-family:monospace">-0.18m</span>
    </div>
    <div style="display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center">
      <label for="cal-camera-pitch" style="width:90px">Cam Tilt</label>
      <input type="range" id="cal-camera-pitch" min="-45" max="45" step="1" value="0">
      <span id="val-camera-pitch" style="width:40px;text-align:right;font-family:monospace">0°</span>
    </div>
  </div>
  <div id="calibration-status" style="margin-top:6px;font-size:11px;color:#aab4bd;text-align:right">status: synced</div>
</aside>
<button id="xr" disabled>VR unavailable</button>
<div id="fallback">Desktop drag rotates, wheel zooms, right-drag pans. In VR, thumbstick steers, squeeze stops, A/B dock or explore.</div>
<script src="https://cdn.babylonjs.com/babylon.js"></script>
<script type="module">
if (window.trustedTypes && window.trustedTypes.createPolicy) {
  if (!window.trustedTypes.defaultPolicy) {
    window.trustedTypes.createPolicy("default", {
      createHTML: (string) => string,
      createScript: (string) => string,
      createScriptURL: (string) => string,
    });
  }
}

const canvas = document.getElementById('scene');
const statusEl = document.getElementById('status');
const xrButton = document.getElementById('xr');
const reignState = document.getElementById('reign-state');
const reignJoystick = document.getElementById('reign-joystick');
const reignStick = document.getElementById('reign-stick');
const reignStop = document.getElementById('reign-stop');
const reignDock = document.getElementById('reign-dock');
const reignExplore = document.getElementById('reign-explore');
const llmStatus = document.getElementById('llm-status');
const llmStreams = document.getElementById('llm-streams');
const modelStatus = document.getElementById('model-status');
const modelList = document.getElementById('model-list');
const modelGraph = document.getElementById('model-graph');
const behaviorInspector = document.getElementById('behavior-inspector');
const virtualPipelineSection = document.getElementById('virtual-pipeline-section');
const virtualReportSummary = document.getElementById('virtual-report-summary');
const virtualModelRecommendations = document.getElementById('virtual-model-recommendations');
const fields = Object.fromEntries(['mode','scenario','training_mode','weights_updating','ledger_counts','action_selector_mode','chosen_action','motor_line','motion_sent','movement_delta','blocked_reason','seed','tick','t','pose','battery','stuck','trap_kind','dead_battery','recovery_mode','recovery_attempts','stuck_ticks','nearest','eye','points','audio','mind','scheme','secure','webxr'].map(id => [id, document.getElementById(id)]));
const learningMode = document.getElementById('learning-mode');
const learningSteps = document.getElementById('learning-steps');
const learningStatus = document.getElementById('learning-status');
const learningApply = document.getElementById('learning-apply');
const learningChecks = Array.from(document.querySelectorAll('#learning-behaviors input[type="checkbox"]'));

// Initialize Babylon Engine & Scene
const engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true });
const scene = new BABYLON.Scene(engine);
scene.clearColor = new BABYLON.Color4(0.043, 0.051, 0.063, 1.0); // 0x0b0d10

// Viewer Camera (OrbitControls)
const viewerCamera = new BABYLON.ArcRotateCamera("viewerCamera", 0, 0, 10, new BABYLON.Vector3(0, 0, 0), scene);
viewerCamera.setPosition(new BABYLON.Vector3(3.8, 4.0, 5.2));
viewerCamera.attachControl(canvas, true);
viewerCamera.fov = 62 * Math.PI / 180;
viewerCamera.minZ = 0.05;
viewerCamera.maxZ = 80;
viewerCamera.inertia = 0.9;
viewerCamera.layerMask = 0xFFFFFFFF;

// Hemispheric Light (ambient)
const hemiLight = new BABYLON.HemisphericLight("hemiLight", new BABYLON.Vector3(0, 1, 0), scene);
hemiLight.diffuse = new BABYLON.Color3(0.937, 0.965, 1.0); // 0xeef6ff
hemiLight.groundColor = new BABYLON.Color3(0.145, 0.188, 0.251); // 0x253040
hemiLight.intensity = 1.8;

// Directional Light (sun)
const sunLight = new BABYLON.DirectionalLight("sunLight", new BABYLON.Vector3(-3, -6, -2).normalize(), scene);
sunLight.diffuse = new BABYLON.Color3(1.0, 1.0, 1.0);
sunLight.intensity = 1.8;

// Ground Plane
const ground = BABYLON.MeshBuilder.CreateGround("ground", {width: 10, height: 10, subdivisions: 10}, scene);
const groundMat = new BABYLON.PBRMaterial("groundMat", scene);
groundMat.albedoColor = new BABYLON.Color3(0.09, 0.114, 0.133); // 0x171d22
groundMat.roughness = 0.8;
groundMat.metallic = 0.0;
ground.material = groundMat;

// Grid line helper
function createGrid(size, subdivisions, color1, color2) {
  const points = [];
  const colors = [];
  const half = size / 2;
  const step = size / subdivisions;
  const c1 = BABYLON.Color4.FromColor3(color1, 1.0);
  const c2 = BABYLON.Color4.FromColor3(color2, 1.0);
  for (let i = 0; i <= subdivisions; i++) {
    const pos = -half + i * step;
    const isCenter = Math.abs(pos) < 0.001;
    const col = isCenter ? c1 : c2;
    points.push([new BABYLON.Vector3(pos, 0.001, -half), new BABYLON.Vector3(pos, 0.001, half)]);
    colors.push([col, col]);
    points.push([new BABYLON.Vector3(-half, 0.001, pos), new BABYLON.Vector3(half, 0.001, pos)]);
    colors.push([col, col]);
  }
  return BABYLON.MeshBuilder.CreateLineSystem("grid", {lines: points, colors: colors}, scene);
}
const gridMesh = createGrid(10, 20, new BABYLON.Color3(0.208, 0.255, 0.302), new BABYLON.Color3(0.141, 0.188, 0.227));

// Robot node & components
const robot = new BABYLON.TransformNode("robot", scene);

const bodyMesh = BABYLON.MeshBuilder.CreateCylinder("bodyMesh", {height: 0.18, diameter: 0.36, tessellation: 32}, scene);
const bodyMat = new BABYLON.PBRMaterial("bodyMat", scene);
bodyMat.albedoColor = new BABYLON.Color3(0.545, 0.827, 1.0); // 0x8bd3ff
bodyMat.roughness = 0.55;
bodyMesh.material = bodyMat;
bodyMesh.position.y = 0.09;
bodyMesh.parent = robot;

const headingArrow = BABYLON.MeshBuilder.CreateCylinder("headingArrow", {height: 0.16, diameterTop: 0.0, diameterBottom: 0.08, tessellation: 12}, scene);
const headingLine = BABYLON.MeshBuilder.CreateLines("headingLine", {points: [new BABYLON.Vector3(0, 0.22, 0), new BABYLON.Vector3(0, 0.22, -0.55)]}, scene);
const headingMat = new BABYLON.StandardMaterial("headingMat", scene);
headingMat.emissiveColor = new BABYLON.Color3(1.0, 0.812, 0.4); // 0xffcf66
headingLine.color = new BABYLON.Color3(1.0, 0.812, 0.4);
headingArrow.material = headingMat;
headingArrow.position.set(0, 0.22, -0.55);
headingArrow.rotation.x = -Math.PI / 2;
headingArrow.parent = robot;
headingLine.parent = robot;

// Second Camera (Eye Camera)
const eyeCamera = new BABYLON.TargetCamera("eyeCamera", new BABYLON.Vector3(0, 0.46, -0.18), scene);
eyeCamera.parent = robot;
eyeCamera.rotation.set(0, Math.PI, 0); // pointing forward along robot's -Z axis
eyeCamera.fov = 62 * Math.PI / 180;
eyeCamera.minZ = 0.05;
eyeCamera.maxZ = 80;
eyeCamera.layerMask = 0x0FFFFFFF; // hide eye screen/frustum lines

// Render Target Texture (RTT) for eyeCamera
const eyeRTT = new BABYLON.RenderTargetTexture("eyeRTT", 256, scene);
eyeRTT.activeCamera = eyeCamera;
eyeRTT.renderList = null;
eyeRTT.uScale = -1;

// Eye Panel
const eyePanel = BABYLON.MeshBuilder.CreatePlane("eyePanel", {width: 0.96, height: 0.72, sideOrientation: BABYLON.Mesh.DOUBLESIDE}, scene);
eyePanel.position.set(0, 0.65, -0.78);
eyePanel.parent = robot;
eyePanel.layerMask = 0x10000000;

const eyePanelMat = new BABYLON.StandardMaterial("eyePanelMat", scene);
eyePanelMat.disableLighting = true;
eyePanelMat.backFaceCulling = false;
eyePanel.material = eyePanelMat;

// Frustum Helper Lines
let frustum = null;
function updateFrustum() {
  if (frustum) {
    frustum.dispose();
  }
  const camPos = new BABYLON.Vector3(0, eyeCamera.position.y, eyeCamera.position.z);
  const halfW = 0.96 / 2;
  const halfH = 0.72 / 2;
  const panelY = 0.65;
  const panelZ = -0.78;
  const frustumPoints = [
    [camPos, new BABYLON.Vector3(-halfW, panelY - halfH, panelZ)],
    [camPos, new BABYLON.Vector3(halfW, panelY - halfH, panelZ)],
    [camPos, new BABYLON.Vector3(-halfW, panelY + halfH, panelZ)],
    [camPos, new BABYLON.Vector3(halfW, panelY + halfH, panelZ)]
  ];
  frustum = BABYLON.MeshBuilder.CreateLineSystem("frustum", {lines: frustumPoints}, scene);
  frustum.color = new BABYLON.Color3(0.565, 0.643, 0.722);
  frustum.parent = robot;
  frustum.layerMask = 0x10000000;
}
updateFrustum();

// Textures
const eyeCanvas = document.createElement('canvas');
eyeCanvas.width = 2; eyeCanvas.height = 2;
const eyeTexture = new BABYLON.DynamicTexture("eyeTexture", eyeCanvas, scene, true);

function useRTT() {
  eyePanelMat.emissiveTexture = eyeRTT;
  eyePanelMat.diffuseTexture = null;
  if (scene.customRenderTargets.indexOf(eyeRTT) === -1) {
    scene.customRenderTargets.push(eyeRTT);
  }
}

function useDynamicTexture() {
  eyePanelMat.emissiveTexture = eyeTexture;
  eyePanelMat.diffuseTexture = eyeTexture;
  const index = scene.customRenderTargets.indexOf(eyeRTT);
  if (index !== -1) {
    scene.customRenderTargets.splice(index, 1);
  }
}

// Scene Nodes
const beams = new BABYLON.TransformNode("beams", scene);
const objects = new BABYLON.TransformNode("objects", scene);

let pointCloud = null;
let lastScene = null;
let xrSession = null;
let lastReignKey = '';
let lastReignSentAt = 0;
let lastReignText = 'idle';
let webReignPointerId = null;
let webReignVector = {x: 0, y: 0};
const llmCards = new Map();

function fmt(v, d=2){ return Number.isFinite(v) ? v.toFixed(d) : '-'; }
function world(x, y, up=0){ return new BABYLON.Vector3(x, up, y); }
function titleCase(value){
  return String(value || '-').replace(/_/g, ' ').replace(/\b\w/g, ch => ch.toUpperCase());
}
function actionLabel(action){
  if(!action) return '-';
  if(typeof action === 'string') return titleCase(action);
  const kind = action.kind || Object.keys(action)[0];
  return titleCase(kind);
}
function motorLabel(motor){
  if(!motor) return '-';
  return `forward=${fmt(motor.forward)}, turn=${fmt(motor.turn)}`;
}
function motionLabel(motion){
  if(!motion) return '-';
  if(typeof motion === 'string') return motion;
  if(motion.Stop != null) return 'Stop';
  if(motion.Forward) return `Forward ${fmt(motion.Forward.speed_m_s)} m/s`;
  if(motion.Turn) return `Turn ${fmt(motion.Turn.turn_rad_s)} rad/s`;
  if(motion.Drive) return `Drive ${fmt(motion.Drive.forward_m_s)} m/s, turn ${fmt(motion.Drive.turn_rad_s)}`;
  const kind = Object.keys(motion)[0];
  return kind ? titleCase(kind) : '-';
}

function clearChildren(node){
  const children = node.getChildren();
  for (const child of children) {
    child.dispose();
  }
}

function xrReason(message){
  xrButton.textContent = 'VR unavailable';
  xrButton.disabled = true;
  fields.webxr.textContent = message;
  return message;
}

function materialFor(kind, color){
  const hex = color ? new BABYLON.Color3(color[0]/255, color[1]/255, color[2]/255) : ({
    charger: new BABYLON.Color3(0.27, 0.83, 0.51), // 0x45d483
    person: new BABYLON.Color3(0.91, 0.75, 0.55), // 0xe8c08c
    speaker: new BABYLON.Color3(0.47, 0.55, 1.0), // 0x778cff
    obstacle: new BABYLON.Color3(0.84, 0.46, 0.4) // 0xd67666
  }[kind] || new BABYLON.Color3(0.69, 0.72, 0.75));
  
  const mat = new BABYLON.PBRMaterial("mat_" + kind, scene);
  mat.albedoColor = hex;
  mat.roughness = 0.7;
  mat.metallic = 0.0;
  return mat;
}

function renderObjects(scenePacket){
  clearChildren(objects);
  const arena = scenePacket.arena;
  if(arena){
    ground.scaling.set(arena.width_m / 10, 1, arena.height_m / 10);
    ground.position.set(arena.width_m / 2, 0, arena.height_m / 2);
    gridMesh.position.set(arena.width_m / 2, 0, arena.height_m / 2);
    viewerCamera.setTarget(new BABYLON.Vector3(arena.width_m / 2, 0, arena.height_m / 2));
  }
  for(const item of scenePacket.objects || []){
    let mesh;
    const mat = materialFor(item.kind, item.color_rgb);
    if(item.kind === 'person'){
      const radius = item.radius_m;
      const height = 0.8 + 2 * radius;
      mesh = BABYLON.MeshBuilder.CreateCapsule("person", {radius: radius, height: height, subdivisions: 8}, scene);
      mesh.material = mat;
      mesh.position.copyFrom(world(item.x_m, item.y_m, 0.55));
    }else if(item.kind === 'speaker'){
      mesh = BABYLON.MeshBuilder.CreateCylinder("speaker", {
        height: 0.35,
        diameterTop: 0,
        diameterBottom: item.radius_m * 1.8 * 2,
        tessellation: 24
      }, scene);
      mesh.material = mat;
      mesh.rotation.x = Math.PI / 2;
      mesh.position.copyFrom(world(item.x_m, item.y_m, 0.25));
    }else{
      const h = item.kind === 'charger' ? 0.08 : 0.45;
      mesh = BABYLON.MeshBuilder.CreateCylinder("obj", {
        height: h,
        diameterTop: item.radius_m * 2,
        diameterBottom: item.radius_m * 2,
        tessellation: 28
      }, scene);
      mesh.material = mat;
      mesh.position.copyFrom(world(item.x_m, item.y_m, item.kind === 'charger' ? 0.04 : 0.225));
    }
    mesh.metadata = { id: item.id };
    mesh.parent = objects;
  }
}

function renderBeams(packet){
  clearChildren(beams);
  const b = packet.body;
  for(const beam of packet.range?.beams || []){
    const angle = b.heading_rad + beam.angle_rad;
    const start = world(b.x_m, b.y_m, .12);
    const end = world(b.x_m + Math.cos(angle) * beam.distance_m, b.y_m + Math.sin(angle) * beam.distance_m, .12);
    
    const beamLine = BABYLON.MeshBuilder.CreateLines("beamLine", {points: [start, end]}, scene);
    beamLine.color = beam.hit ? new BABYLON.Color3(1.0, 0.43, 0.36) : new BABYLON.Color3(0.38, 0.83, 0.58);
    beamLine.alpha = beam.hit ? 0.95 : 0.55;
    beamLine.parent = beams;
    
    if(beam.hit){
      const dot = BABYLON.MeshBuilder.CreateSphere("dot", {diameter: 0.11, segments: 12}, scene);
      const dotMat = new BABYLON.StandardMaterial("dotMat", scene);
      dotMat.emissiveColor = new BABYLON.Color3(1.0, 0.43, 0.36);
      dotMat.disableLighting = true;
      dot.material = dotMat;
      dot.position.copyFrom(end);
      dot.parent = beams;
    }
  }
}

function renderPoints(points, coordinateSystem){
  if(pointCloud){
    pointCloud.dispose();
    pointCloud = null;
  }
  if(!points || !points.length) return;
  
  const positions = [];
  const colors = [];
  const indices = [];
  
  const isRobot = coordinateSystem === 'robot';
  
  points.forEach((p, i) => {
    if (isRobot) {
      positions.push(p.x, p.y, -p.z);
    } else {
      positions.push(p.x, p.y, p.z);
    }
    colors.push(p.r / 255, p.g / 255, p.b / 255, 1.0);
    indices.push(i);
  });
  
  pointCloud = new BABYLON.Mesh("pointCloud", scene);
  const vertexData = new BABYLON.VertexData();
  vertexData.positions = positions;
  vertexData.colors = colors;
  vertexData.indices = indices;
  vertexData.applyToMesh(pointCloud);
  
  const mat = new BABYLON.StandardMaterial("pointCloudMat", scene);
  mat.pointsCloud = true;
  mat.pointSize = 3.5;
  mat.emissiveColor = new BABYLON.Color3(1, 1, 1);
  pointCloud.material = mat;
  
  if (isRobot) {
    pointCloud.parent = robot;
  } else {
    pointCloud.parent = eyeCamera;
  }
}

function shouldUseGeneratedEye(packet){
  const session = packet?.session || {};
  const virtualMode = session.source === 'sim' || String(session.mode || '').includes('virtual');
  if(!virtualMode) return false;
  if(session.mode === 'virtual-live') return true;
  const eye = packet?.eye;
  return !eye || !eye.data_url || eye.mean_luma < .08 || eye.non_background_ratio < .01;
}

function frameFormatIs(format, name){
  return format === name || (typeof format === 'object' && format?.[name] !== undefined);
}

function frameFormatText(format){
  return typeof format === 'string' ? format : JSON.stringify(format || '');
}

function yuvToRgb(y, u, v){
  const c = y - 16, d = u - 128, e = v - 128;
  return [
    Math.max(0, Math.min(255, (298 * c + 409 * e + 128) >> 8)),
    Math.max(0, Math.min(255, (298 * c - 100 * d - 208 * e + 128) >> 8)),
    Math.max(0, Math.min(255, (298 * c + 516 * d + 128) >> 8))
  ];
}

function writeYuvPixel(data, target, y, u, v){
  const [r, g, b] = yuvToRgb(y, u, v);
  data[target] = r; data[target + 1] = g; data[target + 2] = b; data[target + 3] = 255;
}

function renderRawEyeFrame(frame){
  if(!frame) return false;
  const fmt = frame.format;
  const isRgb = frameFormatIs(fmt, 'Rgb8');
  const isBgr = frameFormatIs(fmt, 'Bgr8');
  const isGray = frameFormatIs(fmt, 'Gray8');
  const isYuyv = frameFormatIs(fmt, 'Yuyv422');
  const isMjpg = frameFormatIs(fmt, 'Mjpeg') || frameFormatText(fmt).includes('MJPG');
  if(isRgb || isBgr || isGray || isYuyv){
    eyeCanvas.width = frame.width; eyeCanvas.height = frame.height;
    const image = eyeCanvas.getContext('2d').createImageData(frame.width, frame.height);
    if(isGray){
      for(let source = 0, target = 0; target < image.data.length && source < frame.bytes.length; source += 1, target += 4){
        const value = frame.bytes[source];
        image.data[target] = value; image.data[target + 1] = value; image.data[target + 2] = value; image.data[target + 3] = 255;
      }
    }else if(isYuyv){
      for(let source = 0, target = 0; target + 7 < image.data.length && source + 3 < frame.bytes.length; source += 4, target += 8){
        const y0 = frame.bytes[source], u = frame.bytes[source + 1], y1 = frame.bytes[source + 2], v = frame.bytes[source + 3];
        writeYuvPixel(image.data, target, y0, u, v);
        writeYuvPixel(image.data, target + 4, y1, u, v);
      }
    }else{
      for(let source = 0, target = 0; target < image.data.length && source + 2 < frame.bytes.length; source += 3, target += 4){
        image.data[target] = isBgr ? frame.bytes[source + 2] : frame.bytes[source];
        image.data[target + 1] = frame.bytes[source + 1];
        image.data[target + 2] = isBgr ? frame.bytes[source] : frame.bytes[source + 2];
        image.data[target + 3] = 255;
      }
    }
    eyeCanvas.getContext('2d').putImageData(image, 0, 0);
    eyeTexture.update();
    useDynamicTexture();
    return true;
  }
  if(isMjpg){
    const blob = new Blob([new Uint8Array(frame.bytes)], {type:'image/jpeg'});
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      eyeCanvas.width = img.width; eyeCanvas.height = img.height;
      eyeCanvas.getContext('2d').drawImage(img, 0, 0);
      eyeTexture.update();
      useDynamicTexture();
      URL.revokeObjectURL(url);
    };
    img.onerror = () => URL.revokeObjectURL(url);
    img.src = url;
    return true;
  }
  return false;
}

async function renderSnapshotEyeFallback(packet){
  try{
    const res = await fetch('/view/snapshot', {cache:'no-store'});
    if(!res.ok) throw new Error(await res.text());
    const snapshot = await res.json();
    if(!renderRawEyeFrame(snapshot.eye_frame)) useRTT();
  }catch(_error){
    useRTT();
  }
}

function renderEye(packet){
  if(shouldUseGeneratedEye(packet)){
    useRTT();
    return;
  }
  const eye = packet.eye;
  if(!eye?.data_url){
    renderSnapshotEyeFallback(packet);
    return;
  }
  const img = new Image();
  img.onload = () => {
    eyeCanvas.width = img.width; eyeCanvas.height = img.height;
    eyeCanvas.getContext('2d').drawImage(img, 0, 0);
    eyeTexture.update();
    useDynamicTexture();
  };
  img.onerror = () => renderSnapshotEyeFallback(packet);
  img.src = eye.data_url;
}

function learningConfigFromControls(){
  const behaviors = {};
  for(const checkbox of learningChecks){
    behaviors[checkbox.dataset.behavior] = checkbox.checked;
  }
  return {
    mode: learningMode.value,
    behaviors,
    max_train_steps_per_tick: Math.max(1, Math.min(64, Number.parseInt(learningSteps.value || '1', 10)))
  };
}

function applyLearningConfigToControls(config){
  learningMode.value = config.mode || 'off';
  learningSteps.value = String(config.max_train_steps_per_tick || 1);
  const behaviors = config.behaviors || {};
  for(const checkbox of learningChecks){
    checkbox.checked = behaviors[checkbox.dataset.behavior] !== false;
  }
}

function renderLearningStatus(packet){
  if(!packet) return;
  const config = packet.config || packet;
  if(config.mode) applyLearningConfigToControls(config);
  const active = !!(packet.enabled || packet.weights_updating || (config.mode && config.mode !== 'off'));
  learningStatus.textContent = active ? (packet.training_mode || config.mode || 'active') : 'off';
  learningStatus.style.color = active ? '#8df0b2' : '#ffcf7a';
}

async function loadLearningConfig(){
  try{
    const res = await fetch('/view/inline-learning', {cache:'no-store'});
    if(!res.ok) throw new Error(await res.text());
    renderLearningStatus(await res.json());
  }catch(error){
    learningStatus.textContent = 'unavailable';
  }
}

async function saveLearningConfig(){
  learningStatus.textContent = 'saving...';
  try{
    const res = await fetch('/view/inline-learning', {
      method:'POST',
      headers:{'content-type':'application/json'},
      body:JSON.stringify(learningConfigFromControls())
    });
    if(!res.ok) throw new Error(await res.text());
    renderLearningStatus(await res.json());
  }catch(error){
    learningStatus.textContent = 'save failed';
    learningStatus.style.color = '#ff8f8f';
  }
}

function updateScene(packet){
  lastScene = packet;
  robot.position.copyFrom(world(packet.body.x_m, packet.body.y_m, 0));
  robot.rotation.y = -packet.body.heading_rad - Math.PI / 2;
  renderObjects(packet);
  renderBeams(packet);
  renderPoints(packet.kinect?.points || [], packet.kinect?.coordinate_system || 'camera');
  renderEye(packet);
  const session = packet.session || {};
  fields.mode.textContent = session.mode || '-';
  fields.scenario.textContent = session.scenario || '-';
  fields.training_mode.textContent = packet.training_mode || packet.training?.training_mode || '-';
  fields.weights_updating.textContent = packet.weights_updating ? 'live learning active' : 'not updating';
  fields.weights_updating.classList.toggle('active-learning', !!packet.weights_updating);
  if(packet.weights_updating){
    learningStatus.textContent = packet.training_mode || 'active';
    learningStatus.style.color = '#8df0b2';
  }
  fields.ledger_counts.textContent = `${packet.frames_written || 0}f / ${packet.transitions_written || 0}t`;
  fields.action_selector_mode.textContent = packet.action_selector_mode || '-';
  const action = packet.action || {};
  fields.chosen_action.textContent = actionLabel(action.final_selected_action);
  fields.motor_line.textContent = motorLabel(action.final_motor || action.desired_motor);
  fields.motion_sent.textContent = motionLabel(action.motion_sent);
  fields.movement_delta.textContent = action.movement_delta == null ? '-' : `${fmt(action.movement_delta, 3)} m`;
  fields.blocked_reason.textContent = action.not_moving_reason || (action.safety_override ? 'safety override active' : '-');
  fields.seed.textContent = session.seed == null ? '-' : String(session.seed);
  fields.tick.textContent = session.tick_ms == null ? '-' : `${session.tick_ms} ms`;
  fields.t.textContent = `${packet.t_ms} ms`;
  fields.pose.textContent = `${fmt(packet.body.x_m)}, ${fmt(packet.body.y_m)}, ${fmt(packet.body.heading_rad)} rad`;
  fields.battery.textContent = `${fmt(packet.body.battery_level * 100, 1)}%${packet.body.charging ? ' charging' : ''}`;
  const detail = packet.stuck_detail || {};
  fields.stuck.textContent = packet.stuck ? (detail.class || 'stuck') : (detail.recovered ? 'recovered' : 'clear');
  fields.trap_kind.textContent = detail.trap_kind || '-';
  fields.dead_battery.textContent = packet.dead_battery ? 'yes' : 'no';
  fields.recovery_mode.textContent = packet.recovery_mode || '-';
  fields.recovery_attempts.textContent = `${detail.recovery_attempts || 0} (${detail.repeated_trap_count || 0} repeat)`;
  fields.stuck_ticks.textContent = String(packet.stuck_ticks || 0);
  fields.nearest.textContent = packet.range?.nearest_m == null ? '-' : `${fmt(packet.range.nearest_m)} m`;
  if (packet.eye) {
    const isAuth = packet.eye.authoritative ? " (auth)" : " (symbolic)";
    const statusText = packet.eye.retina_connected ? `connected, age ${packet.eye.retina_last_frame_age_ms}ms` : "disconnected";
    fields.eye.textContent = `${packet.eye.width}x${packet.eye.height} luma ${fmt(packet.eye.mean_luma, 2)} | source: ${packet.eye.source}${isAuth} | retina: ${statusText} | rx: ${packet.eye.frames_received} tx: ${packet.eye.frames_written_to_ledger}`;
    if (packet.eye.source === 'babylon-robot-eye' && !packet.eye.retina_connected) {
      fields.eye.style.color = '#ff4444';
      fields.eye.style.fontWeight = 'bold';
    } else {
      fields.eye.style.color = '';
      fields.eye.style.fontWeight = '';
    }
  } else {
    fields.eye.textContent = '-';
    fields.eye.style.color = '';
    fields.eye.style.fontWeight = '';
  }
  fields.points.textContent = String(packet.kinect?.points?.length || 0);
  fields.audio.textContent = packet.audio?.heading_rad == null && packet.audio?.bearing_rad == null ? '-' : `${fmt(packet.audio?.bearing_rad ?? packet.audio?.heading_rad)} rad`;
  fields.mind.textContent = packet.mind?.combobulation || packet.warnings?.[0] || '-';
  fields.scheme.textContent = location.protocol.replace(':', '');
  fields.secure.textContent = window.isSecureContext ? 'yes' : 'no';
}

function buttonPressed(gamepad, index){
  const button = gamepad?.buttons?.[index];
  return !!button && (button.pressed || button.value > .6);
}

function stickAxes(gamepad){
  const axes = gamepad?.axes || [];
  if(axes.length >= 4) return {x: axes[2], y: axes[3]};
  if(axes.length >= 2) return {x: axes[0], y: axes[1]};
  return {x: 0, y: 0};
}

async function postReign(command, ttl_ms, label, priority=.95, source='Gamepad', note='WebXR controller reign'){
  const key = JSON.stringify(command);
  const now = performance.now();
  if(key === lastReignKey && now - lastReignSentAt < 220) return;
  lastReignKey = key;
  lastReignSentAt = now;
  lastReignText = label;
  reignState.textContent = `sending ${label}`;
  try{
    const res = await fetch('/reign/command', {
      method:'POST',
      headers:{'content-type':'application/json'},
      body:JSON.stringify({
        mode:'Direct',
        priority,
        ttl_ms,
        source,
        note,
        command
      })
    });
    if(!res.ok) throw new Error(await res.text());
    reignState.textContent = label;
  }catch(error){
    reignState.textContent = 'reign send failed';
  }
}

function postWebReign(command, ttl_ms, label, priority=.95){
  return postReign(command, ttl_ms, label, priority, 'WebRemote', 'Web panel reign');
}

function resetWebJoystick(){
  webReignVector = {x: 0, y: 0};
  reignStick.style.transform = 'translate(-50%,-50%)';
}

function updateWebJoystick(event){
  const rect = reignJoystick.getBoundingClientRect();
  const radius = rect.width / 2;
  const cx = rect.left + radius;
  const cy = rect.top + radius;
  const dx = event.clientX - cx;
  const dy = event.clientY - cy;
  const distance = Math.min(radius - 17, Math.hypot(dx, dy));
  const angle = Math.atan2(dy, dx);
  const knobX = Math.cos(angle) * distance;
  const knobY = Math.sin(angle) * distance;
  reignStick.style.transform = `translate(calc(-50% + ${knobX}px), calc(-50% + ${knobY}px))`;
  webReignVector = {
    x: Math.max(-1, Math.min(1, dx / radius)),
    y: Math.max(-1, Math.min(1, dy / radius))
  };
}

function commandForWebJoystick(){
  const {x, y} = webReignVector;
  if(Math.hypot(x, y) < .28) return null;
  if(y < -.32 && Math.abs(y) >= Math.abs(x) * .85){
    const intensity = Math.min(1, Math.max(.25, -y));
    return {command:{type:'Go', intensity, duration_ms:350}, ttl:700, label:`forward ${fmt(intensity, 2)}`, priority:.95};
  }
  if(y > .32 && Math.abs(y) >= Math.abs(x) * .85){
    const intensity = Math.min(1, Math.max(.25, y));
    return {command:{type:'Reverse', intensity, duration_ms:350}, ttl:700, label:`reverse ${fmt(intensity, 2)}`, priority:.98};
  }
  if(Math.abs(x) > .32){
    const intensity = Math.min(1, Math.max(.25, Math.abs(x)));
    return {command:{type:'Turn', direction:x < 0 ? 'Left' : 'Right', intensity, duration_ms:300}, ttl:650, label:`turn ${x < 0 ? 'left' : 'right'} ${fmt(intensity, 2)}`, priority:.95};
  }
  return null;
}

function pollWebReigns(){
  if(webReignPointerId == null) return false;
  const next = commandForWebJoystick();
  if(next){
    postWebReign(next.command, next.ttl, next.label, next.priority);
    return true;
  }
  if(performance.now() - lastReignSentAt > 900) reignState.textContent = 'hold and drag to reign';
  return true;
}

reignJoystick.addEventListener('pointerdown', event => {
  webReignPointerId = event.pointerId;
  reignJoystick.setPointerCapture(event.pointerId);
  updateWebJoystick(event);
  event.preventDefault();
});

reignJoystick.addEventListener('pointermove', event => {
  if(event.pointerId !== webReignPointerId) return;
  updateWebJoystick(event);
  event.preventDefault();
});

function endWebJoystick(event){
  if(event.pointerId !== webReignPointerId) return;
  webReignPointerId = null;
  resetWebJoystick();
  postWebReign({type:'Stop'}, 900, 'stop', 1);
}

reignJoystick.addEventListener('pointerup', endWebJoystick);
reignJoystick.addEventListener('pointercancel', endWebJoystick);
reignStop.addEventListener('click', () => postWebReign({type:'Stop'}, 2000, 'stop', 1));
reignDock.addEventListener('click', () => postWebReign({type:'Dock'}, 5000, 'dock', .9));
reignExplore.addEventListener('click', () => postWebReign({type:'Explore', duration_ms:3000}, 5000, 'explore', .85));

function commandForInputSource(inputSource){
  const gamepad = inputSource.gamepad;
  if(!gamepad) return null;
  if(buttonPressed(gamepad, 1) || buttonPressed(gamepad, 3)) {
    return {command:{type:'Stop'}, ttl:900, label:'stop', priority:1};
  }
  if(buttonPressed(gamepad, 4)) {
    return {command:{type:'Dock'}, ttl:5000, label:'dock', priority:.9};
  }
  if(buttonPressed(gamepad, 5)) {
    return {command:{type:'Explore', duration_ms:3000}, ttl:5000, label:'explore', priority:.85};
  }
  const {x, y} = stickAxes(gamepad);
  if(y < -.35) {
    const intensity = Math.min(1, Math.max(.25, -y));
    return {command:{type:'Go', intensity, duration_ms:350}, ttl:700, label:`forward ${fmt(intensity, 2)}`, priority:.95};
  }
  if(y > .35) {
    const intensity = Math.min(1, Math.max(.25, y));
    return {command:{type:'Reverse', intensity, duration_ms:350}, ttl:700, label:`reverse ${fmt(intensity, 2)}`, priority:.98};
  }
  if(Math.abs(x) > .35) {
    const intensity = Math.min(1, Math.max(.25, Math.abs(x)));
    return {command:{type:'Turn', direction:x < 0 ? 'Left' : 'Right', intensity, duration_ms:300}, ttl:650, label:`turn ${x < 0 ? 'left' : 'right'} ${fmt(intensity, 2)}`, priority:.95};
  }
  if(buttonPressed(gamepad, 0)) {
    return {command:{type:'Go', intensity:.45, duration_ms:350}, ttl:700, label:'trigger forward', priority:.9};
  }
  return null;
}

function pollXrReigns(){
  if(pollWebReigns()) return;
  if(!xrSession){
    if(performance.now() - lastReignSentAt > 1200){
      lastReignKey = '';
      reignState.textContent = lastReignText === 'idle' ? 'web reign ready' : `ready, last ${lastReignText}`;
    }
    return;
  }
  let activeSources = 0;
  for(const inputSource of xrSession.inputSources){
    if(!inputSource.gamepad) continue;
    activeSources += 1;
    const next = commandForInputSource(inputSource);
    if(next){
      postReign(next.command, next.ttl, next.label, next.priority);
      return;
    }
  }
  if(activeSources === 0){
    reignState.textContent = 'no XR gamepad found';
  }else if(performance.now() - lastReignSentAt > 900){
    lastReignKey = '';
    reignState.textContent = lastReignText === 'idle' ? 'ready' : `ready, last ${lastReignText}`;
  }
}

async function poll(){
  try{
    const res = await fetch('/view/scene', {cache:'no-store'});
    if(!res.ok) throw new Error(await res.text());
    updateScene(await res.json());
    statusEl.textContent = 'live';
  }catch(error){
    statusEl.textContent = 'waiting for scene packets...';
  }finally{
    setTimeout(poll, 100);
  }
}

function llmText(value){
  return value && value.trim() ? value : '-';
}

function phaseLabel(event){
  if(event.phase === 'start') return 'prompt';
  if(event.phase === 'delta') return 'streaming';
  if(event.phase === 'done') return 'done';
  if(event.phase === 'error') return 'error';
  return event.phase || '-';
}

function createLlmCard(event){
  const card = document.createElement('article');
  card.className = 'llm-card live';
  card.innerHTML = `
    <div class="llm-top">
      <div class="llm-title"></div>
      <div class="llm-phase"></div>
    </div>
    <div class="llm-block">
      <div class="llm-label">Prompt</div>
      <div class="llm-text prompt"></div>
    </div>
    <div class="llm-block">
      <div class="llm-label">Live output</div>
      <div class="llm-text output"></div>
    </div>`;
  const state = {event, card, output:''};
  llmCards.set(event.id, state);
  llmStreams.prepend(card);
  while(llmStreams.children.length > 5){
    const last = llmStreams.lastElementChild;
    const id = Number(last?.dataset?.id);
    if(Number.isFinite(id)) llmCards.delete(id);
    last?.remove();
  }
  card.dataset.id = String(event.id);
  return state;
}

function renderLlmEvent(event){
  let state = llmCards.get(event.id);
  if(!state) state = createLlmCard(event);
  state.event = {...state.event, ...event};
  if(event.delta) state.output += event.delta;
  if(event.response != null) state.output = event.response;
  if(event.error) state.output = event.error;
  const card = state.card;
  card.classList.toggle('live', event.phase === 'start' || event.phase === 'delta');
  card.classList.toggle('error', event.phase === 'error');
  card.querySelector('.llm-title').textContent = `${event.purpose || 'llm'} · ${event.model || 'model'} #${event.id}`;
  card.querySelector('.llm-phase').textContent = phaseLabel(event);
  const promptEl = card.querySelector('.prompt');
  if(event.prompt != null) promptEl.textContent = llmText(event.prompt);
  else if(!promptEl.textContent) promptEl.textContent = '-';
  const outputEl = card.querySelector('.output');
  outputEl.textContent = llmText(state.output);
  outputEl.scrollTop = outputEl.scrollHeight;
  llmStatus.textContent = `${phaseLabel(event)} #${event.id}`;
}

function connectLlmStream(){
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socket = new WebSocket(`${protocol}//${location.host}/stream/llm`);
  socket.addEventListener('open', () => { llmStatus.textContent = 'listening'; });
  socket.addEventListener('message', event => {
    try{ renderLlmEvent(JSON.parse(event.data)); }
    catch(_error){ llmStatus.textContent = 'bad stream packet'; }
  });
  socket.addEventListener('close', () => {
    llmStatus.textContent = 'reconnecting...';
    setTimeout(connectLlmStream, 1000);
  });
  socket.addEventListener('error', () => { llmStatus.textContent = 'socket error'; });
}

function compactNumber(value, digits=3){
  if(value == null || !Number.isFinite(Number(value))) return '-';
  const n = Number(value);
  if(Math.abs(n) >= 1000) return n.toExponential(2);
  if(Math.abs(n) > 0 && Math.abs(n) < .001) return n.toExponential(2);
  return n.toFixed(digits).replace(/\.?0+$/, '');
}

function modelLabel(model){
  return model.behavior || model.name.replace(/_v\d+$/, '');
}

function renderModelStats(packet){
  const models = packet.models || [];
  modelStatus.textContent = `${models.length} models`;
  modelList.replaceChildren(...models.map(model => {
    const row = document.createElement('article');
    row.className = 'model-row';
    const evalStats = model.evaluation || {};
    const metrics = model.metrics || {};
    const best = model.best_loss != null ? model.best_loss : evalStats.model_loss_mean;
    const latest = metrics.last_train_loss != null ? metrics.last_train_loss : evalStats.model_loss_mean;
    const status = model.registered_status || evalStats.recommendation || 'local';
    const dims = [
      model.input_dim != null ? `in ${model.input_dim}` : null,
      model.output_dim != null ? `out ${model.output_dim}` : null,
      model.latent_dim != null ? `z ${model.latent_dim}` : null,
      model.width && model.height ? `${model.width}x${model.height}` : null
    ].filter(Boolean).join(' · ');
    row.innerHTML = `
      <div class="model-name"></div>
      <div class="model-pill"></div>
      <div class="model-line"></div>
      <div class="model-line"></div>`;
    row.querySelector('.model-name').textContent = model.name;
    row.querySelector('.model-pill').textContent = status;
    const lines = row.querySelectorAll('.model-line');
    lines[0].textContent = `${modelLabel(model)} · samples ${model.samples_seen ?? evalStats.sample_count ?? '-'} · best ${compactNumber(best)} · latest ${compactNumber(latest)}`;
    lines[1].textContent = `${dims || 'dims -'} · model ${compactNumber(evalStats.model_loss_mean)} · hardcoded ${compactNumber(evalStats.hardcoded_loss_mean)} · ${evalStats.model_better_than_hardcoded === true ? 'beats hardcoded' : evalStats.model_better_than_hardcoded === false ? 'hardcoded ahead' : 'no comparison'}`;
    return row;
  }));
}

let behaviorNodes = [];
let selectedBehaviorNodeId = 'Conductor';
const graphLayout = {
  Sensors:[28,18,'core'], Now:[148,18,'core'], Experience:[268,18,'model'], Future:[396,18,'model'],
  Danger:[28,92,'model'], Charge:[148,92,'model'], EyeNext:[268,92,'model'], EarNext:[396,92,'model'],
  EventBump:[28,166,'model'], EventFaceDetected:[148,166,'model'], Conductor:[268,166,'core'], ActionValue:[416,166,'model'],
  Autonomic:[416,226,'core'], Body:[28,226,'core'], Ledger:[148,226,'core'], Training:[268,226,'core'], Models:[416,286,'model']
};
const staticGraphNodes = ['Sensors','Now','Autonomic','Body','Ledger','Training','Models'];

function nodeStateClass(node){
  if(!node) return 'core';
  const classes = [node.selected_regime || 'hardcoded'];
  if(node.missing_model_or_checkpoint) classes.push('missing');
  if(node.node_id === selectedBehaviorNodeId) classes.push('selected');
  return classes.join(' ');
}

function renderBehaviorInspector(){
  const node = behaviorNodes.find(item => item.node_id === selectedBehaviorNodeId) || behaviorNodes[0];
  if(!node){
    behaviorInspector.innerHTML = '<div class="title">No behavior nodes</div>';
    return;
  }
  selectedBehaviorNodeId = node.node_id;
  const hardcodedOptions = (node.hardcoded_implementations || []).map(item => `<option value="${item.id}" ${item.id === node.selected_hardcoded ? 'selected' : ''}>${item.id}</option>`).join('');
  const modelOptions = (node.model_implementations || []).map(item => `<option value="${item.id}" ${item.id === node.selected_model ? 'selected' : ''}>${item.id}</option>`).join('');
  const regimeOptions = (node.allowed_regimes || []).map(item => `<option value="${item}" ${item === node.selected_regime ? 'selected' : ''}>${item}</option>`).join('');
  const fallbacks = ['use_hardcoded','use_last_good_output','return_error','stop_safely','hardcoded_on_error','stop_on_error'];
  const fallbackOptions = fallbacks.map(item => `<option value="${item}" ${item === node.fallback_policy ? 'selected' : ''}>${item}</option>`).join('');
  const run = node.last_run ? `${node.last_run.regime} · ${node.last_run.error || 'ok'}${node.last_run.disagreement != null ? ` · disagreement ${compactNumber(node.last_run.disagreement)}` : ''}` : 'no run yet';
  behaviorInspector.innerHTML = `
    <div class="title"><span>${node.label || node.node_id}</span><span>${node.behavior_id}</span></div>
    <label>Regime<select data-field="selected_regime">${regimeOptions}</select></label>
    <label>Fallback<select data-field="fallback_policy">${fallbackOptions}</select></label>
    <label class="wide">Hardcoded teacher<select data-field="selected_hardcoded">${hardcodedOptions}</select></label>
    <label class="wide">Model<select data-field="selected_model"><option value="">none</option>${modelOptions}</select></label>
    <label class="wide">Checkpoint<input data-field="checkpoint_path" value="${node.checkpoint_path || ''}" placeholder="data/models/..."></label>
    <label>Training<input type="checkbox" data-field="training_enabled" ${node.training_enabled ? 'checked' : ''}></label>
    <div class="run">${run}</div>`;
  behaviorInspector.querySelectorAll('[data-field]').forEach(control => {
    control.addEventListener('change', () => updateBehaviorNode(node.node_id));
  });
}

async function updateBehaviorNode(nodeId){
  const payload = {};
  behaviorInspector.querySelectorAll('[data-field]').forEach(control => {
    const field = control.dataset.field;
    payload[field] = control.type === 'checkbox' ? control.checked : control.value;
  });
  if(payload.selected_model === '') payload.selected_model = null;
  try{
    const res = await fetch(`/view/behavior-nodes/${encodeURIComponent(nodeId)}`, {
      method:'POST',
      headers:{'content-type':'application/json'},
      body: JSON.stringify(payload)
    });
    if(!res.ok) throw new Error(await res.text());
    const updated = await res.json();
    behaviorNodes = behaviorNodes.map(node => node.node_id === updated.node_id ? updated : node);
    renderBehaviorInspector();
    renderModelGraph({connections: currentModelConnections, behavior_nodes: behaviorNodes});
  }catch(error){
    behaviorInspector.querySelector('.run').textContent = error.message || 'update failed';
  }
}

let currentModelConnections = [];

function renderModelGraph(packet){
  currentModelConnections = packet.connections || currentModelConnections;
  behaviorNodes = packet.behavior_nodes || packet.nodes || behaviorNodes;
  const nodes = {...graphLayout};
  for(const node of behaviorNodes){
    if(!nodes[node.node_id]) nodes[node.node_id] = [282,166,'model'];
  }
  const w = 94, h = 26;
  const cx = name => nodes[name] ? nodes[name][0] + w / 2 : 0;
  const cy = name => nodes[name] ? nodes[name][1] + h / 2 : 0;
  const edges = (packet.connections || []).filter(edge => nodes[edge.from] && nodes[edge.to]);
  modelGraph.innerHTML = `<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill='#627384'></path></marker></defs>`;
  for(const edge of edges){
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    const x1 = cx(edge.from), y1 = cy(edge.from), x2 = cx(edge.to), y2 = cy(edge.to);
    const bend = Math.max(22, Math.abs(x2 - x1) * .35);
    line.setAttribute('class', 'edge');
    line.setAttribute('d', `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`);
    modelGraph.appendChild(line);
    if(edge.label && Math.abs(y2 - y1) < 95){
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('class', 'edge-label');
      label.setAttribute('x', String((x1 + x2) / 2 - 18));
      label.setAttribute('y', String((y1 + y2) / 2 - 4));
      label.textContent = edge.label;
      modelGraph.appendChild(label);
    }
  }
  for(const [name, [x, y, kind]] of Object.entries(nodes)){
    const behaviorNode = behaviorNodes.find(item => item.node_id === name);
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    group.setAttribute('class', `node ${kind} ${nodeStateClass(behaviorNode)}`);
    if(behaviorNode){
      group.addEventListener('click', () => {
        selectedBehaviorNodeId = behaviorNode.node_id;
        renderBehaviorInspector();
        renderModelGraph({connections: currentModelConnections, behavior_nodes: behaviorNodes});
      });
    }
    group.innerHTML = `<rect x="${x}" y="${y}" width="${w}" height="${h}"></rect><text x="${x + w / 2}" y="${y + 17}" text-anchor="middle">${name}</text>`;
    modelGraph.appendChild(group);
  }
  renderBehaviorInspector();
}

async function refreshModels(){
  try{
    const res = await fetch('/models', {cache:'no-store'});
    if(!res.ok) throw new Error(await res.text());
    const packet = await res.json();
    renderModelStats(packet);
    renderModelGraph(packet);
  }catch(error){
    modelStatus.textContent = 'unavailable';
  }finally{
    setTimeout(refreshModels, 5000);
  }
}

async function setupXr(){
  fields.scheme.textContent = location.protocol.replace(':', '');
  fields.secure.textContent = window.isSecureContext ? 'yes' : 'no';
  if(!window.isSecureContext){
    xrReason('HTTPS required');
    return;
  }
  if(!navigator.xr){
    xrReason('navigator.xr missing');
    return;
  }
  const ok = await BABYLON.WebXRSessionManager.IsSessionSupportedAsync('immersive-vr');
  if(!ok){
    xrReason('immersive-vr unsupported');
    return;
  }
  xrButton.textContent = 'Enter VR';
  xrButton.disabled = false;
  fields.webxr.textContent = 'immersive-vr supported';
  xrButton.onclick = async () => {
    const customXRButton = new BABYLON.WebXREnterExitUIButton(xrButton, "immersive-vr", "local-floor");
    const xrHelper = await scene.createDefaultXRExperienceAsync({
      floorMeshes: [ground],
      uiOptions: {
        customButtons: [customXRButton]
      }
    });
    xrSession = xrHelper.baseExperience.sessionManager.session;
    reignState.textContent = 'controller and web reigns ready';
    xrHelper.baseExperience.sessionManager.onXRSessionEnded.add(() => {
      xrSession = null;
      lastReignKey = '';
      lastReignText = 'idle';
      reignState.textContent = 'web reign ready';
    });
  };
}

let isInitialized = false;
let maxZIndex = 100;
const panelIds = ['hud', 'learning', 'reign', 'llm', 'virtual-pipeline-section', 'models', 'model-graph-window', 'calibration'];

function preparePanelWindow(el) {
  if (el.classList.contains('panel-window')) return el.querySelector('.panel-titlebar');
  const title = el.dataset.windowTitle || el.id;
  el.classList.add('panel-window');

  const titlebar = document.createElement('div');
  titlebar.className = 'panel-titlebar drag-handle';
  titlebar.textContent = title;
  titlebar.title = 'Drag to move. Double-click to shade.';

  const content = document.createElement('div');
  content.className = 'panel-content';
  while (el.firstChild) content.appendChild(el.firstChild);
  el.append(titlebar, content);

  titlebar.addEventListener('dblclick', (e) => {
    e.preventDefault();
    el.classList.toggle('is-shaded');
    bringToFront(el);
    savePanelLayouts();
  });

  return titlebar;
}

function savePanelLayouts() {
  if (!isInitialized) return;
  const layouts = {};
  panelIds.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      const rect = el.getBoundingClientRect();
      layouts[id] = {
        left: rect.left,
        top: rect.top,
        width: rect.width,
        height: rect.height,
        zIndex: el.style.zIndex ? parseInt(el.style.zIndex, 10) : 100,
        shaded: el.classList.contains('is-shaded')
      };
    }
  });
  localStorage.setItem('netherwick-hud-layout', JSON.stringify(layouts));
}

function loadPanelLayouts() {
  const data = localStorage.getItem('netherwick-hud-layout');
  if (data) {
    try {
      const layouts = JSON.parse(data);
      let highestZ = 100;
      for (const id in layouts) {
        const el = document.getElementById(id);
        if (el && layouts[id]) {
          const layout = layouts[id];
          
          const left = Math.max(0, Math.min(window.innerWidth - 100, layout.left));
          const top = Math.max(0, Math.min(window.innerHeight - 50, layout.top));
          const width = Math.max(100, Math.min(window.innerWidth, layout.width));
          const height = Math.max(50, Math.min(window.innerHeight, layout.height));
          
          el.style.left = left + 'px';
          el.style.top = top + 'px';
          el.style.width = width + 'px';
          el.style.height = height + 'px';
          el.style.right = 'auto';
          el.style.bottom = 'auto';
          
          if (layout.zIndex) {
            el.style.zIndex = layout.zIndex;
            if (layout.zIndex > highestZ) highestZ = layout.zIndex;
          }
          el.classList.toggle('is-shaded', !!layout.shaded);
        }
      }
      maxZIndex = highestZ;
    } catch (e) {
      console.error("Failed to load HUD layouts:", e);
    }
  }
  isInitialized = true;
}

function applyOrClearLayouts() {
  if (window.innerWidth < 820) {
    panelIds.forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.style.left = '';
        el.style.top = '';
        el.style.width = '';
        el.style.height = '';
        el.style.right = '';
        el.style.bottom = '';
        el.style.zIndex = '';
        el.classList.remove('is-shaded');
      }
    });
  } else {
    if (!isInitialized) {
      loadPanelLayouts();
    } else {
      panelIds.forEach(id => {
        const el = document.getElementById(id);
        if (el && el.style.left) {
          const rect = el.getBoundingClientRect();
          const left = Math.max(0, Math.min(window.innerWidth - 100, rect.left));
          const top = Math.max(0, Math.min(window.innerHeight - 50, rect.top));
          el.style.left = left + 'px';
          el.style.top = top + 'px';
        }
      });
    }
  }
}

function bringToFront(el) {
  maxZIndex += 1;
  el.style.zIndex = maxZIndex;
  savePanelLayouts();
}

function setupDraggableAndResizable() {
  panelIds.forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    const handle = preparePanelWindow(el);

    el.addEventListener('mousedown', () => bringToFront(el));
    el.addEventListener('touchstart', () => bringToFront(el));

    const resizeObserver = new ResizeObserver(() => {
      savePanelLayouts();
    });
    resizeObserver.observe(el);

    handle.addEventListener('mousedown', dragStart);
    handle.addEventListener('touchstart', dragStart, { passive: false });

    function dragStart(e) {
        if (e.target.tagName === 'BUTTON' || e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'A') {
          return;
        }
        
        e.preventDefault();

        const isTouch = e.type === 'touchstart';
        const clientX = isTouch ? e.touches[0].clientX : e.clientX;
        const clientY = isTouch ? e.touches[0].clientY : e.clientY;

        const rect = el.getBoundingClientRect();
        const startLeft = rect.left;
        const startTop = rect.top;

        const onMouseMove = (moveEvent) => {
          const moveTouch = moveEvent.type === 'touchmove';
          const curX = moveTouch ? moveEvent.touches[0].clientX : moveEvent.clientX;
          const curY = moveTouch ? moveEvent.touches[0].clientY : moveEvent.clientY;

          const dx = curX - clientX;
          const dy = curY - clientY;

          let newLeft = startLeft + dx;
          let newTop = startTop + dy;

          newLeft = Math.max(0, Math.min(window.innerWidth - 60, newLeft));
          newTop = Math.max(0, Math.min(window.innerHeight - 40, newTop));

          el.style.left = newLeft + 'px';
          el.style.top = newTop + 'px';
          el.style.right = 'auto';
          el.style.bottom = 'auto';
        };

        const onMouseUp = () => {
          window.removeEventListener('mousemove', onMouseMove);
          window.removeEventListener('mouseup', onMouseUp);
          window.removeEventListener('touchmove', onMouseMove);
          window.removeEventListener('touchend', onMouseUp);
          savePanelLayouts();
        };

        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);
        window.addEventListener('touchmove', onMouseMove, { passive: false });
        window.addEventListener('touchend', onMouseUp);
    }
  });

  applyOrClearLayouts();
}

addEventListener('resize', () => {
  engine.resize();
  applyOrClearLayouts();
});

let retinaFrameIndex = 0;
let lastRetinaSendTime = 0;
let isSendingRetina = false;

async function sendRetinaFrame() {
  if (!lastScene || !lastScene.session || lastScene.session.mode !== 'virtual-live') {
    return;
  }
  if (isSendingRetina) return;

  const fps = lastScene.eye?.fps || 5.0;
  const interval = 1000 / fps;
  const now = performance.now();
  if (now - lastRetinaSendTime < interval) {
    return;
  }
  
  isSendingRetina = true;
  lastRetinaSendTime = now;

  try {
    const rgbaData = await eyeRTT.readPixels();
    if (!rgbaData || rgbaData.length === 0) {
      isSendingRetina = false;
      return;
    }

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 256;
    tempCanvas.height = 256;
    const tempCtx = tempCanvas.getContext('2d');
    const imgData = tempCtx.createImageData(256, 256);
    imgData.data.set(rgbaData);
    tempCtx.putImageData(imgData, 0, 0);

    const retinaWidth = lastScene.eye?.width || 160;
    const retinaHeight = lastScene.eye?.height || 90;

    const retinaCanvas = document.createElement('canvas');
    retinaCanvas.width = retinaWidth;
    retinaCanvas.height = retinaHeight;
    const retinaCtx = retinaCanvas.getContext('2d');

    // Correct left-right mirror so uploaded robot-eye frames match world orientation.
    retinaCtx.translate(retinaWidth, 0);
    retinaCtx.scale(-1, 1);
    retinaCtx.drawImage(tempCanvas, 0, 0, 256, 256, 0, 0, retinaWidth, retinaHeight);
    retinaCtx.setTransform(1, 0, 0, 1, 0, 0);

    const dataUrl = retinaCanvas.toDataURL('image/png');

    const payload = {
      schema_version: 1,
      source: 'babylon-robot-eye',
      t_ms: lastScene.t_ms || Date.now(),
      frame_index: retinaFrameIndex++,
      width: retinaWidth,
      height: retinaHeight,
      format: 'Rgb8',
      encoding: 'base64',
      data: dataUrl
    };

    const res = await fetch('/view/retina-frame', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      console.warn('Failed to post retina frame:', await res.text());
    }
  } catch (err) {
    console.error('Error posting retina frame:', err);
  } finally {
    isSendingRetina = false;
  }
}

engine.runRenderLoop(() => {
  pollXrReigns();
  scene.render();
  sendRetinaFrame();
});

setupXr();
connectLlmStream();
refreshModels();
learningApply.addEventListener('click', saveLearningConfig);
loadLearningConfig();
poll();
async function pollVirtualTraining() {
  try {
    const res = await fetch('/view/training/latest', {cache:'no-store'});
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    if (data.status === 'none') {
      virtualPipelineSection.style.display = 'none';
    } else {
      virtualPipelineSection.style.display = '';
      const rr = data.run_report || {};
      virtualReportSummary.innerHTML = `
        <strong>Latest Session Metrics:</strong><br>
        • Duration: ${rr.duration_seconds ? rr.duration_seconds.toFixed(1) : 0}s (${rr.total_frames || 0} frames)<br>
        • Transitions: ${rr.total_transitions || 0} · Eye frames: ${rr.total_eye_frames || 0}<br>
        • Stuck events: ${rr.total_stuck_trap_events || 0}<br>
        • Battery Delta: ${rr.battery_delta ? (rr.battery_delta * 100).toFixed(1) : 0}%
      `;
      let recsHtml = '<strong>Model Recommendations & status:</strong><br>';
      if (data.models) {
        for (const [behavior, info] of Object.entries(data.models)) {
          const lossStr = info.loss != null ? info.loss.toFixed(4) : 'N/A';
          let warningText = '';
          if (info.warnings && info.warnings.length > 0) {
            warningText = ` <span style="color: #ff6b6b;" title="${info.warnings.join(', ')}">⚠️</span>`;
          }
          recsHtml += `
            <div style="margin-bottom: 4px; padding-left: 6px; border-left: 2px solid #52a9ff;">
              <strong>${behavior}</strong>: ${info.name}<br>
              Status: ${info.new_status} (Rec: ${info.recommended_action})${warningText}<br>
              Loss: ${lossStr} · Collisions: ${info.candidate_collision_rate != null ? info.candidate_collision_rate.toFixed(3) : 'N/A'} (vs ${info.baseline_collision_rate != null ? info.baseline_collision_rate.toFixed(3) : 'N/A'})
            </div>
          `;
        }
      }
      virtualModelRecommendations.innerHTML = recsHtml;
    }
  } catch (error) {
    console.error("Error polling virtual training:", error);
  } finally {
    setTimeout(pollVirtualTraining, 5000);
  }
}
pollVirtualTraining();

// Sensor Calibration Panel Wiring
const defaults = {
  depthScale: 1.0,
  pointY: 0.18,
  depthFov: 122,
  cameraFov: 62,
  cameraY: 0.46,
  cameraZ: -0.18,
  cameraPitch: 0
};

let cal = { ...defaults };

const depthScaleEl = document.getElementById('cal-depth-scale');
const pointYEl = document.getElementById('cal-point-y');
const depthFovEl = document.getElementById('cal-depth-fov');
const cameraFovEl = document.getElementById('cal-camera-fov');
const cameraYEl = document.getElementById('cal-camera-y');
const cameraZEl = document.getElementById('cal-camera-z');
const cameraPitchEl = document.getElementById('cal-camera-pitch');
const resetBtn = document.getElementById('reset-calibration');
const statusEl2 = document.getElementById('calibration-status');

const valDepthScale = document.getElementById('val-depth-scale');
const valPointY = document.getElementById('val-point-y');
const valDepthFov = document.getElementById('val-depth-fov');
const valCameraFov = document.getElementById('val-camera-fov');
const valCameraY = document.getElementById('val-camera-y');
const valCameraZ = document.getElementById('val-camera-z');
const valCameraPitch = document.getElementById('val-camera-pitch');

function loadCalibration() {
  const stored = localStorage.getItem('netherwick-sensor-calibration');
  if (stored) {
    try {
      cal = { ...defaults, ...JSON.parse(stored) };
    } catch (e) {
      console.error('Failed to parse calibration:', e);
    }
  }
}

function saveCalibration() {
  localStorage.setItem('netherwick-sensor-calibration', JSON.stringify(cal));
}

function applyCalibrationToUI() {
  depthScaleEl.value = cal.depthScale;
  pointYEl.value = cal.pointY;
  depthFovEl.value = cal.depthFov;
  cameraFovEl.value = cal.cameraFov;
  cameraYEl.value = cal.cameraY;
  cameraZEl.value = cal.cameraZ;
  cameraPitchEl.value = cal.cameraPitch;

  valDepthScale.textContent = cal.depthScale.toFixed(2);
  valPointY.textContent = cal.pointY.toFixed(2) + 'm';
  valDepthFov.textContent = cal.depthFov + '°';
  valCameraFov.textContent = cal.cameraFov + '°';
  valCameraY.textContent = cal.cameraY.toFixed(2) + 'm';
  valCameraZ.textContent = cal.cameraZ.toFixed(2) + 'm';
  valCameraPitch.textContent = cal.cameraPitch + '°';
}

function applyVisionCalibration() {
  eyeCamera.position.y = cal.cameraY;
  eyeCamera.position.z = cal.cameraZ;
  eyeCamera.fov = cal.cameraFov * Math.PI / 180;
  eyeCamera.rotation.x = cal.cameraPitch * Math.PI / 180;
  updateFrustum();
}

let sendCalTimeout = null;
function sendCalibrationToServer() {
  statusEl2.textContent = 'status: syncing...';
  if (sendCalTimeout) clearTimeout(sendCalTimeout);
  
  sendCalTimeout = setTimeout(async () => {
    try {
      const res = await fetch('/view/calibration', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          compact_depth_beam_count: (lastScene && lastScene.sensor_calibration) ? lastScene.sensor_calibration.compact_depth_beam_count : 32,
          compact_depth_fov_rad: cal.depthFov * Math.PI / 180,
          depth_scale: cal.depthScale,
          point_y_m: cal.pointY
        })
      });
      if (res.ok) {
        statusEl2.textContent = 'status: synced';
      } else {
        statusEl2.textContent = 'status: sync failed';
      }
    } catch (e) {
      statusEl2.textContent = 'status: network error';
    }
  }, 150);
}

function onCalChange() {
  cal.depthScale = parseFloat(depthScaleEl.value);
  cal.pointY = parseFloat(pointYEl.value);
  cal.depthFov = parseInt(depthFovEl.value, 10);
  cal.cameraFov = parseInt(cameraFovEl.value, 10);
  cal.cameraY = parseFloat(cameraYEl.value);
  cal.cameraZ = parseFloat(cameraZEl.value);
  cal.cameraPitch = parseInt(cameraPitchEl.value, 10);

  applyCalibrationToUI();
  applyVisionCalibration();
  saveCalibration();
  sendCalibrationToServer();
}

[depthScaleEl, pointYEl, depthFovEl, cameraFovEl, cameraYEl, cameraZEl, cameraPitchEl].forEach(el => {
  el.addEventListener('input', onCalChange);
});

resetBtn.onclick = () => {
  cal = { ...defaults };
  applyCalibrationToUI();
  applyVisionCalibration();
  saveCalibration();
  sendCalibrationToServer();
};

loadCalibration();
applyCalibrationToUI();
applyVisionCalibration();
sendCalibrationToServer();

setupDraggableAndResizable();
</script>"#;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use anyhow::Result;
    use netherwick_actions::{ActionPrimitive, TurnDir};
    use netherwick_autonomic::SimpleSafety;
    use netherwick_body::BodySense;
    use netherwick_conductor::{Conductor, ConductorInput};
    use netherwick_experience::ExperienceLatent;
    use netherwick_ledger::JsonlLedger;
    use netherwick_memory::InMemoryExperienceStore;
    use netherwick_now::Now;
    use netherwick_runtime::{MinimalRuntime, ReignQueue};
    use netherwick_sensors::{EyeFrame, EyeFrameFormat};
    use netherwick_worldlab::{CaptureSource, CaptureWriter};

    #[derive(Clone, Debug, Default)]
    struct RecordingConductor {
        last_input: Arc<Mutex<Option<ConductorInput>>>,
    }

    impl RecordingConductor {
        fn last_input(&self) -> Option<ConductorInput> {
            self.last_input.lock().unwrap().clone()
        }
    }

    impl Conductor for RecordingConductor {
        fn choose(&mut self, input: ConductorInput) -> Result<ActionPrimitive> {
            let chosen = input
                .proposals
                .last()
                .cloned()
                .unwrap_or(ActionPrimitive::Stop);
            *self.last_input.lock().unwrap() = Some(input);
            Ok(chosen)
        }
    }

    #[tokio::test]
    async fn post_stop_pushes_active_reign_state() {
        let state = ReignServerState::standalone();
        let request = ReignCommandRequest {
            mode: ReignMode::Direct,
            command: ReignCommand::Stop,
            priority: 1.0,
            ttl_ms: Some(2_000),
            note: None,
            source: None,
        };

        let Json(input) = post_reign_command(State(state.clone()), Json(request))
            .await
            .unwrap();
        let sense = state.queue().lock().unwrap().sense(input.issued_at_ms);

        assert!(sense.active);
        assert_eq!(sense.latest.as_ref().map(|value| value.id), Some(input.id));
        assert!(matches!(
            sense.latest.as_ref().map(|value| &value.command),
            Some(ReignCommand::Stop)
        ));
    }

    #[tokio::test]
    async fn posted_reign_command_defaults_to_direct_mode() {
        let state = ReignServerState::standalone();
        let request: ReignCommandRequest = serde_json::from_value(serde_json::json!({
            "command": {
                "type": "Turn",
                "direction": "Left",
                "intensity": 0.5,
                "duration_ms": 500
            },
            "ttl_ms": 2_000
        }))
        .unwrap();

        let Json(input) = post_reign_command(State(state.clone()), Json(request))
            .await
            .unwrap();
        let sense = state.queue().lock().unwrap().sense(input.issued_at_ms);

        assert_eq!(input.mode, ReignMode::Direct);
        assert!(sense.active);
        assert_eq!(
            sense.latest.as_ref().map(|input| &input.mode),
            Some(&ReignMode::Direct)
        );
    }

    #[tokio::test]
    async fn manual_prod_enqueues_bounded_assist_command() {
        let queue = Arc::new(Mutex::new(ReignQueue::default()));
        let latest = Arc::new(Mutex::new(Some(WorldSnapshot::default())));
        let state = ReignServerState::with_latest_snapshot(Arc::clone(&queue), latest);

        let Json(response) = post_reign_prod(
            State(state),
            Some(Json(ProdRequest {
                kind: Some("go".to_string()),
                intensity: Some(0.9),
                duration_ms: Some(10_000),
            })),
        )
        .await
        .unwrap();

        assert!(response.accepted);
        assert_eq!(response.input.mode, ReignMode::Assist);
        assert!(matches!(
            response.input.command,
            ReignCommand::Go {
                intensity,
                duration_ms
            } if intensity <= 0.15 && duration_ms <= 1_000
        ));
        let sense = queue.lock().unwrap().sense(response.input.issued_at_ms);
        assert_eq!(
            sense.latest.as_ref().map(|input| input.id),
            Some(response.input.id)
        );
    }

    #[tokio::test]
    async fn posted_reign_command_is_seen_by_runtime_tick() {
        let queue = Arc::new(Mutex::new(ReignQueue::default()));
        let state = ReignServerState::new(Arc::clone(&queue));
        let request = ReignCommandRequest {
            mode: ReignMode::Direct,
            command: ReignCommand::Turn {
                direction: TurnDir::Left,
                intensity: 0.5,
                duration_ms: 500,
            },
            priority: 1.0,
            ttl_ms: Some(2_000),
            note: None,
            source: None,
        };

        let Json(input) = post_reign_command(State(state), Json(request))
            .await
            .unwrap();
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let conductor = RecordingConductor::default();
        let conductor_probe = conductor.clone();
        let mut runtime = MinimalRuntime::with_reign_queue(
            JsonlLedger::new("/tmp/netherwick-server-runtime-shared-reign-test"),
            memory,
            recall,
            conductor,
            SimpleSafety::default(),
            netherwick_llm::NoopLlmAgent,
            Arc::clone(&queue),
        );
        let now = Now::blank(input.issued_at_ms, BodySense::default());
        let expected_action = ActionPrimitive::Turn {
            direction: TurnDir::Left,
            intensity: 0.5,
            duration_ms: 500,
        };

        let tick = runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();

        assert!(tick.frame.now.reign.active);
        assert_eq!(
            tick.frame.now.reign.latest.as_ref().map(|value| value.id),
            Some(input.id)
        );
        assert_eq!(
            tick.frame.reign_input.as_ref().map(|value| value.id),
            Some(input.id)
        );
        assert!(tick
            .frame
            .sensations
            .iter()
            .any(|sensation| sensation.kind == "reign.command"));
        let conductor_input = conductor_probe.last_input().unwrap();
        assert_eq!(
            conductor_input.reign.latest.as_ref().map(|value| value.id),
            Some(input.id)
        );
        assert!(conductor_input.proposals.contains(&expected_action));
        assert_eq!(tick.chosen_action, Some(expected_action.clone()));
        assert_eq!(tick.frame.chosen_action, Some(expected_action));
        assert!(tick
            .frame
            .reign_outcome
            .as_ref()
            .map(|outcome| outcome.accepted_by_conductor)
            .unwrap_or(false));
    }

    #[tokio::test]
    async fn live_scene_returns_503_before_first_snapshot() {
        let err = get_live_scene(State(LiveViewState::new()))
            .await
            .unwrap_err();

        assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn inline_learning_control_updates_live_state() {
        let state = LiveViewState::new();
        let config = InlineLearningConfig {
            mode: InlineLearningMode::WorldOutcome,
            behaviors: netherwick_runtime::InlineLearningBehaviors {
                danger: false,
                charge: false,
                future: true,
                action_value: true,
                eye_next: false,
                ear_next: false,
                experience: false,
            },
            max_train_steps_per_tick: 2,
        };

        let Json(response) = post_inline_learning(State(state.clone()), Json(config.clone()))
            .await
            .unwrap();
        let Json(readback) = get_inline_learning(State(state)).await;

        assert!(response.enabled);
        assert_eq!(response.training_mode, "inline-world-outcome");
        assert_eq!(readback.config, config);
        assert!(readback.weights_updating);
    }

    #[tokio::test]
    async fn behavior_node_endpoints_round_trip_config() {
        let state = LiveViewState::new();

        let Json(initial) = get_behavior_nodes(State(state.clone())).await;
        assert!(initial.nodes.iter().any(|node| node.node_id == "Conductor"));

        let Json(updated) = post_behavior_node(
            State(state.clone()),
            AxumPath("Conductor".to_string()),
            Json(BehaviorNodeUpdate {
                selected_regime: Some(BehaviorRegime::ShadowTrain),
                selected_hardcoded: Some("reign.teacher".to_string()),
                selected_model: Some("conductor.burn.v0".to_string()),
                checkpoint_path: Some("data/models/conductor_v0".to_string()),
                fallback_policy: Some(netherwick_behaviors::FallbackPolicy::UseHardcoded),
                training_enabled: Some(true),
            }),
        )
        .await
        .unwrap();
        let Json(readback) = get_behavior_nodes(State(state.clone())).await;
        let conductor = readback
            .nodes
            .iter()
            .find(|node| node.node_id == "Conductor")
            .unwrap();

        assert_eq!(updated.selected_regime, BehaviorRegime::ShadowTrain);
        assert_eq!(conductor.selected_hardcoded, "reign.teacher");
        assert_eq!(
            conductor.checkpoint_path.as_deref(),
            Some("data/models/conductor_v0")
        );
        assert!(conductor.training_enabled);

        let Json(updated_event) = post_behavior_node(
            State(state.clone()),
            AxumPath("EventBump".to_string()),
            Json(BehaviorNodeUpdate {
                selected_regime: Some(BehaviorRegime::ShadowTrain),
                selected_hardcoded: Some("script.on_bump.v0".to_string()),
                selected_model: Some("event.bump.shadow.v0".to_string()),
                checkpoint_path: Some("data/models/event_bump_v0".to_string()),
                fallback_policy: Some(netherwick_behaviors::FallbackPolicy::UseHardcoded),
                training_enabled: Some(true),
            }),
        )
        .await
        .unwrap();

        assert_eq!(updated_event.selected_regime, BehaviorRegime::ShadowTrain);
        assert_eq!(updated_event.selected_hardcoded, "script.on_bump.v0");
        assert_eq!(
            updated_event.selected_model.as_deref(),
            Some("event.bump.shadow.v0")
        );
    }

    #[tokio::test]
    async fn live_scene_returns_body_pose_and_range_beams() {
        let state = LiveViewState::new();
        state.update_scene_metadata(LiveSceneMetadata {
            arena: Some(SceneArena {
                width_m: 4.0,
                height_m: 3.0,
            }),
            objects: vec![SceneObject {
                id: "charger-0".to_string(),
                kind: "charger".to_string(),
                x_m: 1.2,
                y_m: 0.4,
                radius_m: 0.25,
                label: Some("charger".to_string()),
                color_rgb: Some([80, 220, 130]),
            }],
            sensor_calibration: Some(SceneSensorCalibration::sim_default()),
        });
        state.update_session(SceneSession {
            mode: "virtual-live".to_string(),
            scenario: Some("charger-seeking".to_string()),
            seed: Some(99),
            source: "sim".to_string(),
            tick_ms: Some(100),
        });
        state.update_training_status(LiveTrainingStatus {
            training_mode: "collecting".to_string(),
            ledger_path: Some("data/ledger/virtual-live".to_string()),
            frames_written: 12,
            transitions_written: 11,
            models_loaded: vec!["danger".to_string()],
            model_modes: HashMap::from([("danger".to_string(), "shadow-infer".to_string())]),
            action_selector_mode: "baseline".to_string(),
            weights_updating: false,
        });
        state.update_prod_state(NudgeStatus {
            idle_ms: 4_200,
            last_nudge_ms: Some(1_000),
            nudge_count_recent: 1,
            nudge_blocked_reason: Some("prod cooldown active".to_string()),
            active_nudge: false,
        });
        let mut snapshot = WorldSnapshot::default();
        snapshot.body.odometry.x_m = 0.5;
        snapshot.body.odometry.y_m = 0.75;
        snapshot.body.odometry.heading_rad = 1.25;
        snapshot.body.battery_level = 0.82;
        snapshot.body.last_update_ms = 1234;
        snapshot.final_selected_action = Some(ActionPrimitive::Explore {
            style: netherwick_actions::ExploreStyle::RandomWalk,
            duration_ms: 1_000,
        });
        snapshot.llm_action_proposal = Some(netherwick_actions::LlmActionProposal {
            proposed_action: Some(ActionPrimitive::Explore {
                style: netherwick_actions::ExploreStyle::RandomWalk,
                duration_ms: 1_000,
            }),
            accepted: true,
            safety_vetoed: false,
            final_action: snapshot.final_selected_action.clone(),
            ignored_reason: None,
            safety_reason: None,
        });
        snapshot.range.beams = vec![1.0, 2.0, 3.0];
        snapshot.range.nearest_m = Some(1.0);
        snapshot.extensions.push(netherwick_now::ExtensionSense {
            schema_version: 1,
            name: "sim.stuck".to_string(),
            values: vec![
                1.0, 1.0, 6.0, 300.0, 3.0, -1.0, 1.0, 0.0, 1.0, 0.0, 3.0, 2.0, 1.0, 0.2,
            ],
        });
        snapshot.eye_frame = Some(EyeFrame {
            captured_at_ms: 1200,
            width: 1,
            height: 1,
            format: EyeFrameFormat::Rgb8,
            bytes: vec![255, 0, 0],
            source: None,
        });
        let expected_llm_action = snapshot
            .llm_action_proposal
            .as_ref()
            .and_then(|proposal| proposal.proposed_action.clone());
        let expected_final_action = snapshot.final_selected_action.clone();
        state.update(snapshot);

        let Json(scene) = get_live_scene(State(state)).await.unwrap();

        assert_eq!(scene.schema_version, 1);
        assert_eq!(scene.session.as_ref().unwrap().mode, "virtual-live");
        assert_eq!(
            scene.session.as_ref().unwrap().scenario.as_deref(),
            Some("charger-seeking")
        );
        assert_eq!(scene.t_ms, 1234);
        assert_eq!(scene.body.x_m, 0.5);
        assert_eq!(scene.body.y_m, 0.75);
        assert_eq!(scene.body.heading_rad, 1.25);
        assert_eq!(scene.action.latest_llm_proposed_action, expected_llm_action);
        assert_eq!(scene.action.llm_action_accepted, Some(true));
        assert_eq!(scene.action.llm_action_safety_vetoed, Some(false));
        assert_eq!(scene.action.final_selected_action, expected_final_action);
        assert_eq!(scene.range.nearest_m, Some(1.0));
        assert_eq!(scene.range.beams.len(), 3);
        assert_eq!(scene.training_mode, "collecting");
        assert_eq!(
            scene.ledger_path.as_deref(),
            Some("data/ledger/virtual-live")
        );
        assert_eq!(scene.frames_written, 12);
        assert_eq!(scene.transitions_written, 11);
        assert_eq!(scene.models_loaded, vec!["danger"]);
        assert!(!scene.weights_updating);
        assert!(scene.stuck);
        assert_eq!(scene.idle_ms, 4_200);
        assert_eq!(scene.last_nudge_ms, Some(1_000));
        assert_eq!(scene.nudge_count_recent, 1);
        assert_eq!(
            scene.nudge_blocked_reason.as_deref(),
            Some("prod cooldown active")
        );
        assert_eq!(scene.prod.idle_ms, 4_200);
        assert!(scene.dead_battery);
        assert_eq!(scene.recovery_mode.as_deref(), Some("turn-away"));
        assert_eq!(scene.stuck_ticks, 6);
        assert_eq!(scene.stuck_detail.class.as_deref(), Some("column-trap"));
        assert_eq!(scene.stuck_detail.trap_kind.as_deref(), Some("column"));
        assert_eq!(scene.stuck_detail.recovery_attempts, 2);
        assert_eq!(scene.stuck_detail.repeated_trap_count, 1);
        assert_eq!(
            scene.stuck_detail.recovery_phase.as_deref(),
            Some("turn-away")
        );
        assert_eq!(scene.objects[0].kind, "charger");
        assert!(scene
            .eye
            .unwrap()
            .data_url
            .unwrap()
            .starts_with("data:image/png;base64,"));
    }

    #[test]
    fn missing_eye_kinect_and_audio_serialize_as_empty_or_null() {
        let mut snapshot = WorldSnapshot::default();
        snapshot.body.battery_level = 0.0;
        let scene = snapshot_to_scene(
            &snapshot,
            None,
            None,
            LiveTrainingStatus::default(),
            NudgeStatus::default(),
            default_behavior_nodes(),
            None,
        );
        let value = serde_json::to_value(scene).unwrap();

        assert!(value["eye"].is_null());
        assert_eq!(value["dead_battery"].as_bool(), Some(true));
        assert_eq!(value["kinect"]["points"].as_array().unwrap().len(), 0);
        assert!(value["audio"].is_null());
        assert!(value["warnings"].as_array().unwrap().len() >= 3);
    }

    #[test]
    fn compact_kinect_depth_projects_as_meter_range_fan() {
        let depths = vec![2.0; 32];
        let points = depth_points(&depths, Some(SceneSensorCalibration::sim_default()));

        assert_eq!(points.len(), 32);
        assert!((points[0].x + 1.847759).abs() < 0.001);
        assert!((points[0].z - 0.765367).abs() < 0.001);
        assert!((points[31].x - 1.847759).abs() < 0.001);
        assert!((points[31].z - 0.765367).abs() < 0.001);

        let near_center = &points[15];
        assert!(near_center.z > 1.99);
        assert!(near_center.x.abs() < 0.11);
    }

    #[tokio::test]
    async fn capture_frame_to_scene_conversion_works_for_tiny_capture() {
        let root = unique_test_dir("capture-scene");
        let mut writer = CaptureWriter::create(&root, CaptureSource::Sim, Some(100))
            .await
            .unwrap();
        let mut snapshot = WorldSnapshot::default();
        snapshot.body.last_update_ms = 99;
        snapshot.body.odometry.x_m = 1.0;
        snapshot.range.beams = vec![0.5];
        writer
            .append_snapshot(99, snapshot, Vec::new())
            .await
            .unwrap();
        writer.finish().await.unwrap();

        let Json(scene) = get_capture_scene(Query(CaptureSceneQuery {
            capture: root.clone(),
            frame: 0,
        }))
        .await
        .unwrap();

        assert_eq!(scene.t_ms, 99);
        assert_eq!(scene.body.x_m, 1.0);
        assert_eq!(scene.range.beams.len(), 1);
        std::fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn live_routes_include_3d_and_scene_endpoints() {
        assert!(HTTP_ENDPOINTS.contains(&"/view"));
        assert!(HTTP_ENDPOINTS.contains(&"/view/3d"));
        assert!(HTTP_ENDPOINTS.contains(&"/view/scene"));
        assert!(HTTP_ENDPOINTS.contains(&"/models"));
        assert!(HTTP_ENDPOINTS.contains(&"/stream/llm"));
        let Html(page) = live_view_3d_page().await;
        assert!(page.contains("Sensorium 3D"));
        assert!(page.contains("/view/scene"));
        assert!(page.contains("/models"));
        assert!(page.contains("Training stats"));
        assert!(page.contains("Connections"));
        assert!(page.contains("/stream/llm"));
        assert!(page.contains("LLM streams"));
        assert!(page.contains("navigator.xr"));
        assert!(page.contains("window.isSecureContext"));
        assert!(page.contains("/reign/command"));
        assert!(page.contains("/view/behavior-nodes"));
        assert!(page.contains("behaviorInspector"));
        assert!(page.contains("packet.behavior_nodes"));
        assert!(page.contains("const nodes = {...graphLayout}"));
        assert!(page.contains("source='Gamepad'"));
        assert!(page.contains("type:'Reverse'"));
        assert!(page.contains("createDefaultXRExperienceAsync"));
        assert!(page.contains("if(!eye?.data_url)"));
        assert!(page.contains(
            ".panel-window.is-shaded{height:32px!important;min-height:32px!important;max-height:32px!important;"
        ));
    }

    #[test]
    fn model_summary_reads_training_artifacts() {
        let model_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/models");
        let packet = read_models_response(&model_root);

        assert_eq!(packet.schema_version, 1);
        assert!(packet
            .connections
            .iter()
            .any(|edge| edge.from == "Ledger" && edge.to == "Training"));
        assert!(packet
            .behavior_nodes
            .iter()
            .any(|node| node.node_id == "ActionValue"));
        assert!(packet
            .registry
            .iter()
            .any(|entry| entry.behavior == "danger"));
    }

    #[tokio::test]
    async fn live_view_page_draws_common_camera_formats() {
        let Html(page) = live_view_page().await;

        assert!(page.contains("isBgr"));
        assert!(page.contains("isGray"));
        assert!(page.contains("isYuyv"));
        assert!(page.contains("writeYuvPixel"));
        assert!(page.contains("drawGeneratedEye"));
        assert!(page.contains("/view/scene"));
        assert!(page.contains("session.source === 'sim'"));
        assert!(page.contains("includes('virtual')"));
    }

    #[test]
    fn yuyv_eye_frame_encodes_to_png_data_url() {
        let frame = EyeFrame {
            captured_at_ms: 1,
            width: 2,
            height: 1,
            format: EyeFrameFormat::Yuyv422,
            bytes: vec![82, 90, 145, 240],
            source: None,
        };

        let (eye, warnings) = scene_eye_from_frame(&frame, None, 1);

        assert!(warnings.is_empty());
        assert_eq!(eye.width, 2);
        assert_eq!(eye.height, 1);
        assert!(eye
            .data_url
            .as_deref()
            .unwrap_or_default()
            .starts_with("data:image/png;base64,"));
    }

    fn unique_test_dir(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "netherwick-server-{name}-{}-{}",
            std::process::id(),
            wall_now_ms()
        ));
        std::fs::remove_dir_all(&path).ok();
        path
    }
}
