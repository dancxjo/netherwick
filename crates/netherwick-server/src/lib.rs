use std::collections::{BTreeSet, HashMap};
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
use netherwick_experience::{
    attach_experience_forge_vector, EmbodiedContext, ExperienceForge, ExperienceForgeSnapshot,
};
use netherwick_map::{
    project_beam_endpoint, LocalMap, LocalWorldBelief, MapObservation, MapSummary,
    OccupancyCell as OdomMapCell, PointCloudSummary, VoxelPoint, VoxelPointCloud, MAP_LABEL,
};
use netherwick_memory::{
    EntityConstellationState, EntityLifecycleState, EntityMemory, EntityMemoryReport,
};
use netherwick_now::{KinectSense, KinectSkeletonSense, ReignSense};
use netherwick_runtime::{
    nudge_action_block_reason_for_snapshot, InlineLearningConfig, InlineLearningMode, NudgePolicy,
    NudgeStatus, ReignQueue, RuntimeModelStack,
};
use netherwick_sensors::{
    ClusterObservation, EyeFrameFormat, OccupancyGrid, PlaneObservation, SceneGraphSummary,
    SurfaceExtractor, SurfaceExtractorDiagnostics, SurfaceHypothesis, SurfaceTrack, WorldSnapshot,
};
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
    "/view/scene",
    "/view/map",
    "/view/experience-forge",
    "/view/behavior-nodes",
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
            body_age_ms: snapshot
                .map(|snapshot| now_ms.saturating_sub(snapshot.body.last_update_ms)),
        }
    }
}

const HARDWARE_TTL_MIN_MS: TimeMs = 250;
const HARDWARE_TTL_MAX_MS: TimeMs = 500;
const HARDWARE_MAX_FORWARD_INTENSITY: f32 = 0.15;
const HARDWARE_MAX_TURN_INTENSITY: f32 = 0.25;
const HARDWARE_BODY_STALE_MS: TimeMs = 1_000;
const HARDWARE_CRITICAL_BATTERY: f32 = 0.10;

fn hardware_snapshot_block_reason(snapshot: &WorldSnapshot, now_ms: TimeMs) -> Option<String> {
    let age_ms = now_ms.saturating_sub(snapshot.body.last_update_ms);
    if age_ms > HARDWARE_BODY_STALE_MS {
        return Some(format!("body snapshot stale: {age_ms} ms old"));
    }
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
        .route("/reign/prod", post(post_reign_prod))
        .route("/reign/state", get(get_reign_state))
        .route("/reign/clear", post(post_reign_clear))
        .route("/reign/hardware-arm", post(post_hardware_arm))
        .with_state(state)
}

#[derive(Clone, Debug)]
pub struct LiveViewState {
    latest: Arc<Mutex<Option<WorldSnapshot>>>,
    map: Arc<Mutex<LocalMap>>,
    point_cloud: Arc<Mutex<VoxelPointCloud>>,
    latest_embodied: Arc<Mutex<Option<EmbodiedContext>>>,
    experience_forge: Arc<Mutex<ExperienceForge>>,
    scene_metadata: Arc<Mutex<Option<LiveSceneMetadata>>>,
    session: Arc<Mutex<Option<SceneSession>>>,
    hardware_control: Arc<Mutex<HardwareControlState>>,
    training_status: Arc<Mutex<LiveTrainingStatus>>,
    inline_learning: Arc<Mutex<InlineLearningConfig>>,
    prod_state: Arc<Mutex<NudgeStatus>>,
    behavior_nodes: Arc<Mutex<Vec<BehaviorNodeState>>>,
    surface_extractor: Arc<Mutex<SurfaceExtractor>>,
    entity_memory: Arc<Mutex<EntityMemory>>,
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
            map: Arc::new(Mutex::new(LocalMap::default())),
            point_cloud: Arc::new(Mutex::new(VoxelPointCloud::default())),
            latest_embodied: Arc::new(Mutex::new(None)),
            experience_forge: Arc::new(Mutex::new(ExperienceForge::default())),
            scene_metadata: Arc::new(Mutex::new(None)),
            session: Arc::new(Mutex::new(None)),
            hardware_control: Arc::new(Mutex::new(HardwareControlState::default())),
            training_status: Arc::new(Mutex::new(LiveTrainingStatus::default())),
            inline_learning: Arc::new(Mutex::new(InlineLearningConfig::default())),
            prod_state: Arc::new(Mutex::new(NudgeStatus::default())),
            behavior_nodes: Arc::new(Mutex::new(default_behavior_nodes())),
            surface_extractor: Arc::new(Mutex::new(SurfaceExtractor::default())),
            entity_memory: Arc::new(Mutex::new(EntityMemory::default())),
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

    pub fn with_real_slow_hardware_control(self) -> Self {
        *self
            .hardware_control
            .lock()
            .expect("hardware control mutex poisoned") = HardwareControlState::real_slow();
        self
    }

    pub fn hardware_control_status(&self) -> HardwareControlStatus {
        let now_ms = wall_now_ms();
        let latest = self.latest();
        self.hardware_control
            .lock()
            .expect("hardware control mutex poisoned")
            .status(latest.as_ref(), now_ms)
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

    pub fn record_live_eye_frame(&self, frame: netherwick_sensors::EyeFrame) {
        {
            let mut state = self
                .retina_state
                .lock()
                .expect("retina state mutex poisoned");
            state.latest_frame = Some(frame.clone());
            state.has_new_frame = true;
            state.last_received_at = Some(std::time::Instant::now());
            state.frames_received += 1;
        }

        if let Some(snapshot) = self
            .latest
            .lock()
            .expect("live view snapshot mutex poisoned")
            .as_mut()
        {
            snapshot.eye_frame = Some(frame);
        }
    }

    pub fn update(&self, snapshot: WorldSnapshot) {
        let mut now = snapshot.to_now(snapshot.body.last_update_ms);
        let forge_snapshot = self
            .experience_forge
            .lock()
            .expect("experience forge mutex poisoned")
            .tick(&now, snapshot.final_selected_action.clone());
        if let Some(compact_vector) = forge_snapshot.compact_vector_artifact.as_ref() {
            attach_experience_forge_vector(&mut now, compact_vector);
        }
        self.map
            .lock()
            .expect("live map mutex poisoned")
            .observe_snapshot(&snapshot, snapshot.body.last_update_ms);
        self.point_cloud
            .lock()
            .expect("live point cloud mutex poisoned")
            .observe_snapshot(&snapshot, snapshot.body.last_update_ms);
        {
            use netherwick_memory::PlaceCellKey;
            const CELL_SIZE: f32 = 0.5;
            let x = now.body.odometry.x_m;
            let y = now.body.odometry.y_m;
            let cell_key = PlaceCellKey {
                x: (x / CELL_SIZE).floor() as i32,
                y: (y / CELL_SIZE).floor() as i32,
            };
            self.entity_memory
                .lock()
                .expect("entity memory mutex poisoned")
                .observe_now(&now, Some(cell_key));
        }
        *self
            .latest
            .lock()
            .expect("live view snapshot mutex poisoned") = Some(snapshot);
    }

    pub fn entity_memory_report(&self) -> EntityMemoryReport {
        self.entity_memory
            .lock()
            .expect("entity memory mutex poisoned")
            .report()
    }

    pub fn latest(&self) -> Option<WorldSnapshot> {
        self.latest
            .lock()
            .expect("live view snapshot mutex poisoned")
            .clone()
    }

    pub fn map_snapshot(&self) -> LocalMap {
        self.map.lock().expect("live map mutex poisoned").clone()
    }

    pub fn point_cloud_snapshot(&self) -> VoxelPointCloud {
        self.point_cloud
            .lock()
            .expect("live point cloud mutex poisoned")
            .clone()
    }

    pub fn update_embodied_context(&self, context: EmbodiedContext) {
        *self
            .latest_embodied
            .lock()
            .expect("live embodied context mutex poisoned") = Some(context);
    }

    pub fn latest_embodied_context(&self) -> Option<EmbodiedContext> {
        self.latest_embodied
            .lock()
            .expect("live embodied context mutex poisoned")
            .clone()
    }

    pub fn experience_forge_snapshot(&self) -> ExperienceForgeSnapshot {
        self.experience_forge
            .lock()
            .expect("experience forge mutex poisoned")
            .snapshot()
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

    pub fn surface_perception(
        &self,
        snapshot: &WorldSnapshot,
        calibration: Option<SceneSensorCalibration>,
        action: Option<&ActionPrimitive>,
    ) -> Option<SceneSurfacePerception> {
        if snapshot.kinect.depth_m.is_empty()
            || snapshot.kinect.depth_width == 0
            || snapshot.kinect.depth_height == 0
        {
            return None;
        }
        let mut extractor = self
            .surface_extractor
            .lock()
            .expect("surface extractor mutex poisoned");
        let calibration = calibration.unwrap_or_else(SceneSensorCalibration::sim_default);
        extractor.set_depth_camera_extrinsics(
            calibration.depth_camera_height_m(),
            calibration.depth_camera_forward_m(),
            calibration.depth_camera_pitch_rad(),
            calibration.camera_roll_rad,
            calibration.camera_yaw_rad,
        );
        let mut perception = SceneSurfacePerception::from(extractor.process(
            &snapshot.kinect,
            snapshot.body.odometry,
            snapshot.body.last_update_ms,
        ));
        if let Some(action) = action {
            let frames = netherwick_sensors::anticipate_surfaces(
                &netherwick_sensors::SurfaceExtractorOutput {
                    plane_observations: perception.plane_observations.clone(),
                    stable_surfaces: perception.stable_surfaces.clone(),
                    floor: perception.floor.clone(),
                    obstacle_grid: perception.obstacle_grid.clone(),
                    clusters: perception.clusters.clone(),
                    scene_graph: perception.scene_graph.clone(),
                    diagnostics: perception.diagnostics.clone(),
                    raw_cloud: Vec::new(),
                    filtered_cloud: Vec::new(),
                },
                snapshot.body.odometry,
                action,
            );
            if let Some(object) = perception.scene_graph.navigation.as_object_mut() {
                object.insert(
                    "anticipation".to_string(),
                    serde_json::json!({
                        "action": action,
                        "frames": frames,
                    }),
                );
            }
        }
        Some(perception)
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
    #[serde(default)]
    pub depth_forward_offset_m: f32,
    #[serde(default)]
    pub depth_pitch_down_rad: f32,
    #[serde(default)]
    pub camera_forward_m: f32,
    #[serde(default)]
    pub camera_height_m: f32,
    #[serde(default)]
    pub camera_pitch_rad: f32,
    #[serde(default)]
    pub camera_roll_rad: f32,
    #[serde(default)]
    pub camera_yaw_rad: f32,
}

impl SceneSensorCalibration {
    pub fn sim_default() -> Self {
        Self {
            compact_depth_beam_count: 32,
            compact_depth_fov_rad: std::f32::consts::PI * 0.75,
            depth_scale: 1.0,
            point_y_m: 0.18,
            depth_forward_offset_m: 0.0,
            depth_pitch_down_rad: 0.0,
            camera_forward_m: 0.0,
            camera_height_m: 0.18,
            camera_pitch_rad: 0.0,
            camera_roll_rad: 0.0,
            camera_yaw_rad: 0.0,
        }
    }

    fn depth_camera_forward_m(self) -> f32 {
        if self.camera_forward_m != 0.0 {
            self.camera_forward_m
        } else {
            self.depth_forward_offset_m
        }
    }

    fn depth_camera_height_m(self) -> f32 {
        if self.camera_height_m != 0.0 {
            self.camera_height_m
        } else {
            self.point_y_m
        }
    }

    fn depth_camera_pitch_rad(self) -> f32 {
        if self.camera_pitch_rad != 0.0 {
            self.camera_pitch_rad
        } else {
            self.depth_pitch_down_rad
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
    pub hardware_control: HardwareControlStatus,
    pub training_mode: String,
    pub ledger_path: Option<String>,
    pub frames_written: usize,
    pub transitions_written: usize,
    pub models_loaded: Vec<String>,
    pub model_modes: HashMap<String, String>,
    pub behavior_nodes: Vec<BehaviorNodeState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experience_forge: Option<ExperienceForgeSnapshot>,
    pub action_selector_mode: String,
    pub weights_updating: bool,
    pub t_ms: TimeMs,
    pub body: SceneBody,
    pub range: SceneRange,
    pub eye: Option<SceneEye>,
    pub kinect: SceneKinect,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub surface_perception: Option<SceneSurfacePerception>,
    pub world_belief_layers: Vec<&'static str>,
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
    pub bump_left: bool,
    pub bump_right: bool,
    pub cliff: bool,
    pub wheel_drop: bool,
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
pub struct LiveMapResponse {
    pub schema_version: u32,
    pub label: &'static str,
    pub summary: MapSummary,
    pub overlays: Vec<&'static str>,
    pub pose_trail: Vec<MapPosePoint>,
    pub current_pose: Option<MapPosePoint>,
    pub range_beams: Vec<MapProjectedBeam>,
    pub cells: Vec<MapViewCell>,
    pub semantic_cells: Vec<MapSemanticCell>,
    pub events: Vec<MapEventMarker>,
    pub entity_graph: MapEntityGraph,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapPosePoint {
    pub x_m: f32,
    pub y_m: f32,
    pub heading_rad: f32,
    pub confidence: f32,
    pub t_ms: TimeMs,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapProjectedBeam {
    pub origin_x_m: f32,
    pub origin_y_m: f32,
    pub end_x_m: f32,
    pub end_y_m: f32,
    pub angle_rad: f32,
    pub distance_m: f32,
    pub hit: bool,
    pub confidence: f32,
    pub age_ms: TimeMs,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapViewCell {
    pub x: i32,
    pub y: i32,
    pub center_x_m: f32,
    pub center_y_m: f32,
    pub occupied_score: f32,
    pub free_score: f32,
    pub confidence: f32,
    pub age_ms: TimeMs,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapSemanticCell {
    pub x_m: f32,
    pub y_m: f32,
    pub kind: String,
    pub score: f32,
    pub confidence: f32,
    pub age_ms: Option<TimeMs>,
    pub label: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapEventMarker {
    pub x_m: f32,
    pub y_m: f32,
    pub kind: String,
    pub confidence: f32,
    pub age_ms: TimeMs,
    pub label: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapEntityGraph {
    pub schema_version: u32,
    pub generated_from: &'static str,
    pub nodes: Vec<MapEntityGraphNode>,
    pub edges: Vec<MapEntityGraphEdge>,
    pub events: Vec<MapEntityGraphEvent>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapEntityGraphNode {
    pub id: String,
    pub node_type: String,
    pub label: String,
    pub modality: Option<String>,
    pub x_m: Option<f32>,
    pub y_m: Option<f32>,
    pub confidence: f32,
    pub age_ms: TimeMs,
    pub source_channel: Option<String>,
    pub observed_at_ms: Option<TimeMs>,
    pub vector_shape: Option<String>,
    pub nearest_cluster: Option<String>,
    pub attached_text: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapEntityGraphEdge {
    pub id: String,
    pub from: String,
    pub to: String,
    pub edge_type: String,
    pub confidence: f32,
    pub observed_at_ms: Option<TimeMs>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapEntityGraphEvent {
    pub t_ms: TimeMs,
    pub node_id: String,
    pub event_type: String,
    pub label: String,
    pub confidence: f32,
    pub timestamp_ms: Option<TimeMs>,
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
    #[serde(default)]
    pub accumulated_points: Vec<SceneAccumulatedPoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accumulated_summary: Option<PointCloudSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_world_belief: Option<LocalWorldBelief>,
    pub skeletons: Vec<KinectSkeletonSense>,
    pub diagnostics: SceneKinectDiagnostics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coordinate_system: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SceneKinectDiagnostics {
    pub depth_width: u32,
    pub depth_height: u32,
    pub valid_depth_count: usize,
    pub skipped_depth_count: usize,
    pub clipped_depth_count: usize,
    pub min_depth_m: Option<f32>,
    pub median_depth_m: Option<f32>,
    pub max_depth_m: Option<f32>,
    pub sample_stride: usize,
    pub coordinate_system: String,
    pub below_floor_count: usize,
    pub below_floor_ratio: f32,
    pub min_z_m: Option<f32>,
    pub median_z_m: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SceneSurfacePerception {
    pub diagnostics: SurfaceExtractorDiagnostics,
    pub plane_observations: Vec<PlaneObservation>,
    pub stable_surfaces: Vec<SurfaceHypothesis>,
    pub floor: Option<SurfaceTrack>,
    pub obstacle_grid: OccupancyGrid,
    pub clusters: Vec<ClusterObservation>,
    pub scene_graph: SceneGraphSummary,
}

impl From<netherwick_sensors::SurfaceExtractorOutput> for SceneSurfacePerception {
    fn from(output: netherwick_sensors::SurfaceExtractorOutput) -> Self {
        Self {
            diagnostics: output.diagnostics,
            plane_observations: output.plane_observations,
            stable_surfaces: output.stable_surfaces,
            floor: output.floor,
            obstacle_grid: output.obstacle_grid,
            clusters: output.clusters,
            scene_graph: output.scene_graph,
        }
    }
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
pub struct SceneAccumulatedPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub confidence: f32,
    pub age_ms: TimeMs,
    pub stable: bool,
    pub transient: bool,
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
    pub training_ledger: Option<String>,
    pub behavior_report_path: Option<String>,
    pub scenario_report_path: Option<String>,
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
        .route("/view/embodied", get(get_live_embodied))
        .route("/view/embodied/graph", get(get_live_embodied_graph))
        .route("/api/experience/lineage", get(get_live_embodied_graph))
        .route("/debug/embodied", get(get_live_embodied))
        .route("/debug/embodied/graph", get(get_live_embodied_graph))
        .route("/view/scene", get(get_live_scene))
        .route("/view/map", get(get_live_map))
        .route("/view/experience-forge", get(get_experience_forge))
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
        .route("/memory/entities", get(get_entity_memory))
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

async fn get_live_embodied(
    State(state): State<LiveViewState>,
) -> Result<Json<EmbodiedContext>, LiveViewError> {
    state
        .latest_embodied_context()
        .map(Json)
        .ok_or_else(|| LiveViewError::unavailable("no embodied experience has arrived yet"))
}

async fn get_live_embodied_graph(
    State(state): State<LiveViewState>,
) -> Result<Json<EmbodiedLineageGraph>, LiveViewError> {
    let context = state
        .latest_embodied_context()
        .ok_or_else(|| LiveViewError::unavailable("no embodied experience has arrived yet"))?;
    Ok(Json(EmbodiedLineageGraph::from_context(&context)))
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EmbodiedLineageGraph {
    pub schema_version: u32,
    pub experience_id: Option<String>,
    pub summary: String,
    pub nodes: Vec<EmbodiedGraphNode>,
    pub edges: Vec<EmbodiedGraphEdge>,
    pub vector_metadata: Vec<EmbodiedGraphVector>,
    pub recent_memories: Vec<EmbodiedGraphMemory>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EmbodiedGraphNode {
    pub id: String,
    pub node_type: EmbodiedGraphNodeType,
    pub label: String,
    pub detail: Option<String>,
    pub entity_id: String,
    pub modality: Option<String>,
    pub payload_kind: Option<String>,
    pub derived: bool,
    pub vector_refs: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbodiedGraphNodeType {
    Sensation,
    Impression,
    Experience,
    Prediction,
    MemoryLink,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EmbodiedGraphEdge {
    pub from: String,
    pub to: String,
    pub relation: EmbodiedGraphEdgeType,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbodiedGraphEdgeType {
    ParentSensation,
    AboutSensation,
    SensationMember,
    ImpressionMember,
    SummarizesExperience,
    Predicts,
    MemoryLink,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EmbodiedGraphVector {
    pub index: usize,
    pub owner_node_id: String,
    pub vectorizer_id: String,
    pub model_id: String,
    pub model_label: String,
    pub dim: usize,
    pub modality: String,
    pub payload_kind: String,
    pub source_kind: String,
    pub source_sensation_id: String,
    pub purpose: String,
    pub collection: String,
    pub input_summary: String,
    pub is_fallback: bool,
    pub provenance: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EmbodiedGraphMemory {
    pub node_id: String,
    pub target_id: String,
    pub relation: String,
    pub score: f32,
    pub text: Option<String>,
}

impl EmbodiedLineageGraph {
    pub fn from_context(context: &EmbodiedContext) -> Self {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut vectors = Vec::new();
        let mut recent_memories = Vec::new();
        let experience_node_id = context.experience_id.map(|id| format!("experience:{id}"));

        if let Some(experience_id) = context.experience_id {
            nodes.push(EmbodiedGraphNode {
                id: format!("experience:{experience_id}"),
                node_type: EmbodiedGraphNodeType::Experience,
                label: "fused experience".to_string(),
                detail: non_empty(context.summary.clone()),
                entity_id: experience_id.to_string(),
                modality: None,
                payload_kind: None,
                derived: false,
                vector_refs: vector_refs_for_node(
                    &mut vectors,
                    format!("experience:{experience_id}"),
                    context.fused_vector.as_ref().into_iter(),
                ),
            });
        }

        let sensation_ids = context
            .sensations
            .iter()
            .map(|sensation| sensation.id)
            .collect::<BTreeSet<_>>();
        let impression_ids = context
            .impressions
            .iter()
            .map(|impression| impression.id)
            .collect::<BTreeSet<_>>();

        for sensation in &context.sensations {
            let node_id = format!("sensation:{}", sensation.id);
            let owned_vectors = context
                .sensation_vectors
                .iter()
                .filter(|vector| vector.source_sensation_id == sensation.id);
            let vector_refs = vector_refs_for_node(&mut vectors, node_id.clone(), owned_vectors);
            nodes.push(EmbodiedGraphNode {
                id: node_id.clone(),
                node_type: EmbodiedGraphNodeType::Sensation,
                label: sensation.kind.clone(),
                detail: sensation.summary.clone(),
                entity_id: sensation.id.to_string(),
                modality: Some(sensation.modality.as_str().to_string()),
                payload_kind: Some(sensation.payload_kind.as_str().to_string()),
                derived: sensation.parent_id.is_some(),
                vector_refs,
            });
            if let Some(experience_node_id) = &experience_node_id {
                edges.push(EmbodiedGraphEdge {
                    from: node_id,
                    to: experience_node_id.clone(),
                    relation: EmbodiedGraphEdgeType::SensationMember,
                });
            }
        }

        for edge in &context.lineage {
            if sensation_ids.contains(&edge.parent_id) && sensation_ids.contains(&edge.child_id) {
                edges.push(EmbodiedGraphEdge {
                    from: format!("sensation:{}", edge.parent_id),
                    to: format!("sensation:{}", edge.child_id),
                    relation: EmbodiedGraphEdgeType::ParentSensation,
                });
            }
        }

        for impression in &context.impressions {
            let node_id = format!("impression:{}", impression.id);
            let vector_refs = vector_refs_for_node(
                &mut vectors,
                node_id.clone(),
                impression.vector.as_ref().into_iter(),
            );
            nodes.push(EmbodiedGraphNode {
                id: node_id.clone(),
                node_type: EmbodiedGraphNodeType::Impression,
                label: impression.kind.clone(),
                detail: Some(impression.text.clone()),
                entity_id: impression.id.to_string(),
                modality: None,
                payload_kind: None,
                derived: false,
                vector_refs,
            });
            if let Some(sensation_id) = impression.sensation_id {
                if sensation_ids.contains(&sensation_id) {
                    edges.push(EmbodiedGraphEdge {
                        from: format!("sensation:{sensation_id}"),
                        to: node_id.clone(),
                        relation: EmbodiedGraphEdgeType::AboutSensation,
                    });
                }
            }
            if let (Some(experience_id), Some(experience_node_id)) =
                (context.experience_id, experience_node_id.as_ref())
            {
                if impression.experience_id.unwrap_or(experience_id) == experience_id
                    && impression_ids.contains(&impression.id)
                {
                    edges.push(EmbodiedGraphEdge {
                        from: node_id.clone(),
                        to: experience_node_id.clone(),
                        relation: EmbodiedGraphEdgeType::ImpressionMember,
                    });
                }
            }
        }

        for (index, prediction) in context.predictions.iter().enumerate() {
            let node_id = format!("prediction:{index}");
            let vector_refs = vector_refs_for_node(
                &mut vectors,
                node_id.clone(),
                prediction.vector.as_ref().into_iter(),
            );
            nodes.push(EmbodiedGraphNode {
                id: node_id.clone(),
                node_type: EmbodiedGraphNodeType::Prediction,
                label: format!("+{}ms", prediction.offset_ms),
                detail: Some(format!(
                    "{} ({:.0}% confidence)",
                    prediction.text,
                    prediction.confidence.clamp(0.0, 1.0) * 100.0
                )),
                entity_id: index.to_string(),
                modality: None,
                payload_kind: None,
                derived: false,
                vector_refs,
            });
            if let Some(experience_node_id) = &experience_node_id {
                edges.push(EmbodiedGraphEdge {
                    from: experience_node_id.clone(),
                    to: node_id,
                    relation: EmbodiedGraphEdgeType::Predicts,
                });
            }
        }

        for (index, link) in context.memory_links.iter().enumerate() {
            let node_id = format!("memory:{index}");
            nodes.push(EmbodiedGraphNode {
                id: node_id.clone(),
                node_type: EmbodiedGraphNodeType::MemoryLink,
                label: format!("{} {:.2}", link.relation, link.score),
                detail: link.text.clone().or_else(|| Some(link.target_id.clone())),
                entity_id: link.target_id.clone(),
                modality: Some("memory".to_string()),
                payload_kind: None,
                derived: false,
                vector_refs: Vec::new(),
            });
            if let Some(experience_node_id) = &experience_node_id {
                edges.push(EmbodiedGraphEdge {
                    from: experience_node_id.clone(),
                    to: node_id.clone(),
                    relation: EmbodiedGraphEdgeType::MemoryLink,
                });
            }
            recent_memories.push(EmbodiedGraphMemory {
                node_id,
                target_id: link.target_id.clone(),
                relation: link.relation.clone(),
                score: link.score,
                text: link.text.clone(),
            });
        }

        Self {
            schema_version: 1,
            experience_id: context.experience_id.map(|id| id.to_string()),
            summary: context.summary.clone(),
            nodes,
            edges,
            vector_metadata: vectors,
            recent_memories,
        }
    }
}

fn vector_refs_for_node<'a>(
    vectors: &mut Vec<EmbodiedGraphVector>,
    owner_node_id: String,
    source: impl Iterator<Item = &'a netherwick_experience::EmbodiedVectorRef>,
) -> Vec<usize> {
    source
        .map(|vector| {
            let index = vectors.len();
            vectors.push(EmbodiedGraphVector {
                index,
                owner_node_id: owner_node_id.clone(),
                vectorizer_id: vector.vectorizer_id.clone(),
                model_id: vector.model_id.clone(),
                model_label: vector.model_label.clone(),
                dim: vector.dim,
                modality: vector.modality.as_str().to_string(),
                payload_kind: vector.payload_kind.as_str().to_string(),
                source_kind: vector.source_kind.clone(),
                source_sensation_id: vector.source_sensation_id.to_string(),
                purpose: vector.purpose.clone(),
                collection: vector.collection.clone(),
                input_summary: vector.input_summary.clone(),
                is_fallback: vector.is_fallback,
                provenance: vector.provenance.clone(),
            });
            index
        })
        .collect()
}

fn non_empty(value: String) -> Option<String> {
    if value.trim().is_empty() {
        None
    } else {
        Some(value)
    }
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

    let metadata = state.scene_metadata();
    let calibration = metadata
        .as_ref()
        .and_then(|metadata| metadata.sensor_calibration);
    let mut scene = snapshot_to_scene(
        &snapshot,
        metadata.as_ref(),
        state.session(),
        state.training_status(),
        state.prod_state(),
        state.behavior_nodes(),
        Some(&state.point_cloud_snapshot()),
        retina_status,
        state.hardware_control_status(),
    );
    scene.surface_perception = state.surface_perception(
        &snapshot,
        calibration,
        scene.action.final_selected_action.as_ref(),
    );
    scene.experience_forge = Some(state.experience_forge_snapshot());
    Ok(Json(scene))
}

async fn get_experience_forge(State(state): State<LiveViewState>) -> Json<ExperienceForgeSnapshot> {
    Json(state.experience_forge_snapshot())
}

async fn get_live_map(
    State(state): State<LiveViewState>,
) -> Result<Json<LiveMapResponse>, LiveViewError> {
    let snapshot = state
        .latest()
        .ok_or_else(|| LiveViewError::unavailable("no live world snapshot has arrived yet"))?;
    let map = state.map_snapshot();
    let entity_report = state.entity_memory_report();
    Ok(Json(map_response_from_parts(
        &map,
        &snapshot,
        state.scene_metadata().as_ref(),
        &entity_report,
    )))
}

async fn get_entity_memory(State(state): State<LiveViewState>) -> Json<EntityMemoryReport> {
    Json(state.entity_memory_report())
}

fn map_response_from_parts(
    map: &LocalMap,
    latest: &WorldSnapshot,
    metadata: Option<&LiveSceneMetadata>,
    entity_report: &EntityMemoryReport,
) -> LiveMapResponse {
    let now_ms = latest.body.last_update_ms;
    let summary = map.summary();
    let pose_trail: Vec<_> = map
        .pose_history
        .iter()
        .map(|pose| MapPosePoint {
            x_m: pose.pose.x_m,
            y_m: pose.pose.y_m,
            heading_rad: pose.pose.heading_rad,
            confidence: pose.confidence,
            t_ms: pose.t_ms,
        })
        .collect();
    let current_pose = pose_trail.last().cloned();
    let range_beams = map
        .observations
        .last()
        .map(|observation| projected_beams_from_observation(observation, now_ms))
        .unwrap_or_default();
    let cells: Vec<_> = map
        .cells
        .values()
        .map(|cell| map_view_cell(cell, map.config.resolution_m, now_ms))
        .collect();
    let semantic_cells = map_semantic_cells(latest, metadata, now_ms);
    let events = map_event_markers(latest, metadata, now_ms);
    let entity_graph = map_entity_graph(
        &pose_trail,
        &range_beams,
        &cells,
        &semantic_cells,
        &events,
        entity_report,
        latest,
        now_ms,
    );

    LiveMapResponse {
        schema_version: 1,
        label: MAP_LABEL,
        summary,
        overlays: vec![
            "occupancy",
            "rays",
            "raw point cloud",
            "accumulated occupancy",
            "stable wall candidates",
            "danger",
            "charger/charge",
            "social",
            "novelty",
            "events",
        ],
        pose_trail,
        current_pose,
        range_beams,
        cells,
        semantic_cells,
        events,
        entity_graph,
    }
}

fn map_entity_graph(
    pose_trail: &[MapPosePoint],
    range_beams: &[MapProjectedBeam],
    cells: &[MapViewCell],
    semantic_cells: &[MapSemanticCell],
    map_events: &[MapEventMarker],
    entity_report: &EntityMemoryReport,
    latest: &WorldSnapshot,
    now_ms: TimeMs,
) -> MapEntityGraph {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut events = Vec::new();
    let current_pose = pose_trail.last();

    if let Some(pose) = current_pose {
        push_graph_node(
            &mut nodes,
            "place:current",
            "place",
            "current place",
            Some("odometry"),
            Some((pose.x_m, pose.y_m)),
            pose.confidence,
            now_ms.saturating_sub(pose.t_ms),
        );
        push_graph_node(
            &mut nodes,
            "cluster:odometry:trail",
            "cluster",
            "odometry trail",
            Some("odometry"),
            Some((pose.x_m, pose.y_m)),
            pose.confidence,
            now_ms.saturating_sub(pose.t_ms),
        );
        push_graph_edge(
            &mut edges,
            "cluster:odometry:trail",
            "place:current",
            "same_place_as",
            pose.confidence,
        );
    }

    let nearest_beams: Vec<_> = range_beams
        .iter()
        .enumerate()
        .filter(|(_, beam)| beam.hit)
        .take(8)
        .collect();
    let has_range_cluster = !nearest_beams.is_empty();
    if has_range_cluster {
        push_graph_node(
            &mut nodes,
            "cluster:range:nearest",
            "cluster",
            "nearest range returns",
            Some("range"),
            current_pose.map(|pose| (pose.x_m, pose.y_m)),
            0.76,
            nearest_beams
                .iter()
                .map(|(_, beam)| beam.age_ms)
                .min()
                .unwrap_or(0),
        );
        if current_pose.is_some() {
            push_graph_edge(
                &mut edges,
                "cluster:range:nearest",
                "place:current",
                "same_time_as",
                0.72,
            );
        }
    }

    for (index, beam) in nearest_beams {
        let id = format!("observation:range:{index}");
        push_graph_node(
            &mut nodes,
            &id,
            "observation",
            &format!("range {:.2}m", beam.distance_m),
            Some("range"),
            Some((beam.end_x_m, beam.end_y_m)),
            beam.confidence,
            beam.age_ms,
        );
        push_graph_edge(
            &mut edges,
            &id,
            "cluster:range:nearest",
            "belongs_to",
            beam.confidence,
        );
        if current_pose.is_some() {
            push_graph_edge(&mut edges, &id, "place:current", "same_place_as", 0.54);
        }
    }

    for (index, cell) in cells
        .iter()
        .filter(|cell| cell.occupied_score > cell.free_score && cell.confidence > 0.18)
        .take(10)
        .enumerate()
    {
        let cluster_id = format!("cluster:occupancy:{}:{}", cell.x, cell.y);
        push_graph_node(
            &mut nodes,
            &cluster_id,
            "cluster",
            "occupied cell cluster",
            Some("range"),
            Some((cell.center_x_m, cell.center_y_m)),
            cell.confidence,
            cell.age_ms,
        );
        if has_range_cluster {
            push_graph_edge(
                &mut edges,
                &cluster_id,
                "cluster:range:nearest",
                "co_occurs_with",
                cell.confidence.min(0.7),
            );
        }
        if index < 4 {
            push_graph_edge(
                &mut edges,
                &cluster_id,
                "place:current",
                "same_place_as",
                cell.confidence.min(0.62),
            );
        }
    }

    for (index, cell) in semantic_cells.iter().take(12).enumerate() {
        let clean_kind = graph_id_fragment(&cell.kind);
        let cluster_id = format!("cluster:semantic:{clean_kind}:{index}");
        let entity_id = format!("entity:{clean_kind}:{index}");
        let label_id = format!("text_label:{clean_kind}:{index}");
        let age_ms = cell.age_ms.unwrap_or(0);
        let label = cell.label.clone().unwrap_or_else(|| cell.kind.clone());
        push_graph_node(
            &mut nodes,
            &cluster_id,
            "cluster",
            &format!("{} cluster", cell.kind),
            Some(cell.kind.as_str()),
            Some((cell.x_m, cell.y_m)),
            cell.confidence,
            age_ms,
        );
        push_graph_node(
            &mut nodes,
            &entity_id,
            "entity",
            &label,
            Some(cell.kind.as_str()),
            Some((cell.x_m, cell.y_m)),
            cell.confidence * cell.score,
            age_ms,
        );
        push_graph_node(
            &mut nodes,
            &label_id,
            "text_label",
            &label,
            Some("language"),
            Some((cell.x_m, cell.y_m)),
            cell.confidence,
            age_ms,
        );
        push_graph_edge(
            &mut edges,
            &cluster_id,
            &entity_id,
            "part_of_entity",
            cell.confidence,
        );
        push_graph_edge(
            &mut edges,
            &label_id,
            &entity_id,
            "named_by",
            cell.confidence,
        );
        push_graph_edge(
            &mut edges,
            &cluster_id,
            "place:current",
            "same_place_as",
            0.58,
        );
        push_graph_edge(
            &mut edges,
            &entity_id,
            "place:current",
            "same_place_as",
            0.62,
        );
        if has_range_cluster {
            push_graph_edge(
                &mut edges,
                "cluster:range:nearest",
                &cluster_id,
                "co_occurs_with",
                0.48,
            );
        }
        events.push(MapEntityGraphEvent {
            t_ms: now_ms.saturating_sub(age_ms),
            node_id: entity_id,
            event_type: "entity_seen".to_string(),
            label,
            confidence: cell.confidence,
            timestamp_ms: Some(now_ms.saturating_sub(age_ms)),
        });
    }

    for (index, event) in map_events.iter().take(8).enumerate() {
        let event_id = format!(
            "observation:event:{}:{index}",
            graph_id_fragment(&event.kind)
        );
        push_graph_node(
            &mut nodes,
            &event_id,
            "observation",
            event.label.as_deref().unwrap_or(&event.kind),
            Some(event.kind.as_str()),
            Some((event.x_m, event.y_m)),
            event.confidence,
            event.age_ms,
        );
        push_graph_edge(
            &mut edges,
            &event_id,
            "place:current",
            "same_place_as",
            event.confidence,
        );
        if has_range_cluster
            && (event.kind == "charger" || event.kind == "person" || event.kind == "speaker")
        {
            push_graph_edge(
                &mut edges,
                &event_id,
                "cluster:range:nearest",
                "co_occurs_with",
                0.44,
            );
        }
        events.push(MapEntityGraphEvent {
            t_ms: now_ms.saturating_sub(event.age_ms),
            node_id: event_id,
            event_type: event.kind.clone(),
            label: event.label.clone().unwrap_or_else(|| event.kind.clone()),
            confidence: event.confidence,
            timestamp_ms: Some(now_ms.saturating_sub(event.age_ms)),
        });
    }

    for entity in entity_report.top_entities.iter().take(8) {
        let entity_id = entity.id.clone();
        let entity_label = entity
            .display_name
            .clone()
            .or_else(|| entity.labels.first().cloned())
            .unwrap_or_else(|| entity.kind.clone());
        let entity_age_ms = now_ms.saturating_sub(entity.last_seen_ms);
        push_graph_node(
            &mut nodes,
            &entity_id,
            "entity",
            &entity_label,
            Some(entity.kind.as_str()),
            current_pose.map(|pose| (pose.x_m, pose.y_m)),
            entity.confidence,
            entity_age_ms,
        );
        if current_pose.is_some() {
            push_graph_edge(
                &mut edges,
                &entity_id,
                "place:current",
                "same_place_as",
                entity.confidence.min(0.7),
            );
        }
        events.push(MapEntityGraphEvent {
            t_ms: now_ms.saturating_sub(entity.first_seen_ms),
            node_id: entity_id.clone(),
            event_type: "create".to_string(),
            label: entity_label.clone(),
            confidence: entity.confidence,
            timestamp_ms: Some(entity.first_seen_ms),
        });
        if entity.observation_count > 1 {
            events.push(MapEntityGraphEvent {
                t_ms: now_ms.saturating_sub(entity.last_seen_ms),
                node_id: entity_id.clone(),
                event_type: "strengthen".to_string(),
                label: entity_label.clone(),
                confidence: entity.confidence,
                timestamp_ms: Some(entity.last_seen_ms),
            });
        }
        match entity.lifecycle {
            EntityLifecycleState::Occluded => events.push(MapEntityGraphEvent {
                t_ms: entity_age_ms,
                node_id: entity_id.clone(),
                event_type: "weaken".to_string(),
                label: entity_label.clone(),
                confidence: entity.confidence,
                timestamp_ms: Some(entity.last_seen_ms),
            }),
            EntityLifecycleState::Vanished => events.push(MapEntityGraphEvent {
                t_ms: entity_age_ms,
                node_id: entity_id.clone(),
                event_type: "vanish".to_string(),
                label: entity_label.clone(),
                confidence: entity.confidence,
                timestamp_ms: Some(entity.last_seen_ms),
            }),
            EntityLifecycleState::Active => {}
        }
        match entity.constellation_state {
            EntityConstellationState::Revived => events.push(MapEntityGraphEvent {
                t_ms: entity_age_ms,
                node_id: entity_id.clone(),
                event_type: "revive".to_string(),
                label: entity_label.clone(),
                confidence: entity.confidence,
                timestamp_ms: Some(entity.last_seen_ms),
            }),
            EntityConstellationState::Split => events.push(MapEntityGraphEvent {
                t_ms: entity_age_ms,
                node_id: entity_id.clone(),
                event_type: "split".to_string(),
                label: entity_label.clone(),
                confidence: entity.confidence,
                timestamp_ms: Some(entity.last_seen_ms),
            }),
            EntityConstellationState::Merged => events.push(MapEntityGraphEvent {
                t_ms: entity_age_ms,
                node_id: entity_id.clone(),
                event_type: "merge".to_string(),
                label: entity_label.clone(),
                confidence: entity.confidence,
                timestamp_ms: Some(entity.last_seen_ms),
            }),
            EntityConstellationState::Weak
            | EntityConstellationState::Strong
            | EntityConstellationState::Vanished => {}
        }
        for (label_index, label) in entity.text_labels.iter().take(3).enumerate() {
            let label_id = format!(
                "text_label:{}:{label_index}",
                graph_id_fragment(entity.id.as_str())
            );
            push_graph_node(
                &mut nodes,
                &label_id,
                "text_label",
                label,
                Some("language"),
                current_pose.map(|pose| (pose.x_m, pose.y_m)),
                entity.confidence,
                entity_age_ms,
            );
            push_graph_edge(
                &mut edges,
                &label_id,
                &entity_id,
                "named_by",
                entity.confidence.min(0.9),
            );
        }
        for cluster in entity.modality_clusters.iter().take(12) {
            let cluster_id = format!(
                "cluster:{}:{}",
                graph_id_fragment(entity.id.as_str()),
                graph_id_fragment(cluster.id.as_str())
            );
            push_graph_node(
                &mut nodes,
                &cluster_id,
                "cluster",
                cluster.id.as_str(),
                Some(cluster.modality.as_str()),
                current_pose.map(|pose| (pose.x_m, pose.y_m)),
                cluster.confidence,
                entity_age_ms,
            );
            push_graph_edge(
                &mut edges,
                &cluster_id,
                &entity_id,
                "part_of_entity",
                cluster.confidence,
            );
            let edge_id = format!("{cluster_id}->part_of_entity->{entity_id}");
            if let Some(edge) = edges.iter_mut().find(|edge| edge.id == edge_id) {
                edge.observed_at_ms = Some(entity.last_seen_ms);
            }
        }
        for point in entity.observation_points.iter().rev().take(24) {
            let node_id = format!(
                "observation:{}:{}",
                graph_id_fragment(entity.id.as_str()),
                graph_id_fragment(point.id.as_str())
            );
            let nearest_cluster = entity
                .modality_clusters
                .iter()
                .find(|cluster| {
                    cluster
                        .observation_point_ids
                        .iter()
                        .any(|id| id == &point.id)
                })
                .map(|cluster| {
                    format!(
                        "cluster:{}:{}",
                        graph_id_fragment(entity.id.as_str()),
                        graph_id_fragment(cluster.id.as_str())
                    )
                });
            if nodes.iter().all(|node| node.id != node_id) {
                nodes.push(MapEntityGraphNode {
                    id: node_id.clone(),
                    node_type: "observation".to_string(),
                    label: point.source.clone(),
                    modality: Some(point.modality.as_str().to_string()),
                    x_m: current_pose.map(|pose| pose.x_m),
                    y_m: current_pose.map(|pose| pose.y_m),
                    confidence: point.confidence.clamp(0.0, 1.0),
                    age_ms: now_ms.saturating_sub(point.observed_at_ms),
                    source_channel: Some(point.source.clone()),
                    observed_at_ms: Some(point.observed_at_ms),
                    vector_shape: graph_vector_shape(
                        latest,
                        point.source.as_str(),
                        point.modality.as_str(),
                    ),
                    nearest_cluster: nearest_cluster.clone(),
                    attached_text: point
                        .source
                        .strip_prefix("text:")
                        .map(str::to_string)
                        .filter(|text| !text.trim().is_empty()),
                });
            }
            if let Some(cluster_id) = nearest_cluster.as_ref() {
                push_graph_edge(
                    &mut edges,
                    &node_id,
                    cluster_id,
                    "belongs_to",
                    point.confidence,
                );
                let edge_id = format!("{node_id}->belongs_to->{cluster_id}");
                if let Some(edge) = edges.iter_mut().find(|edge| edge.id == edge_id) {
                    edge.observed_at_ms = Some(point.observed_at_ms);
                }
            }
            push_graph_edge(
                &mut edges,
                &node_id,
                &entity_id,
                "same_time_as",
                point.confidence.min(entity.confidence),
            );
        }
    }

    if let Some(action) = &latest.final_selected_action {
        push_graph_node(
            &mut nodes,
            "action_context:current",
            "action_context",
            &format!("{action:?}"),
            Some("action"),
            current_pose.map(|pose| (pose.x_m, pose.y_m)),
            0.86,
            0,
        );
        if current_pose.is_some() {
            push_graph_edge(
                &mut edges,
                "action_context:current",
                "place:current",
                "same_time_as",
                0.86,
            );
            push_graph_edge(
                &mut edges,
                "action_context:current",
                "cluster:odometry:trail",
                "predicts",
                0.42,
            );
            if has_range_cluster {
                push_graph_edge(
                    &mut edges,
                    "action_context:current",
                    "cluster:range:nearest",
                    "moves_with",
                    0.38,
                );
            }
        }
    }

    MapEntityGraph {
        schema_version: 1,
        generated_from: "live_map_mvp",
        nodes,
        edges,
        events,
    }
}

fn push_graph_node(
    nodes: &mut Vec<MapEntityGraphNode>,
    id: &str,
    node_type: &str,
    label: &str,
    modality: Option<&str>,
    position: Option<(f32, f32)>,
    confidence: f32,
    age_ms: TimeMs,
) {
    if nodes.iter().any(|node| node.id == id) {
        return;
    }
    nodes.push(MapEntityGraphNode {
        id: id.to_string(),
        node_type: node_type.to_string(),
        label: label.to_string(),
        modality: modality.map(str::to_string),
        x_m: position.map(|(x, _)| x),
        y_m: position.map(|(_, y)| y),
        confidence: confidence.clamp(0.0, 1.0),
        age_ms,
        source_channel: None,
        observed_at_ms: None,
        vector_shape: None,
        nearest_cluster: None,
        attached_text: None,
    });
}

fn push_graph_edge(
    edges: &mut Vec<MapEntityGraphEdge>,
    from: &str,
    to: &str,
    edge_type: &str,
    confidence: f32,
) {
    if from == to {
        return;
    }
    let id = format!("{from}->{edge_type}->{to}");
    if edges.iter().any(|edge| edge.id == id) {
        return;
    }
    edges.push(MapEntityGraphEdge {
        id,
        from: from.to_string(),
        to: to.to_string(),
        edge_type: edge_type.to_string(),
        confidence: confidence.clamp(0.0, 1.0),
        observed_at_ms: None,
    });
}

fn graph_id_fragment(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

fn graph_vector_shape(snapshot: &WorldSnapshot, source: &str, modality: &str) -> Option<String> {
    if source.starts_with("face:") {
        return snapshot
            .face
            .vectors
            .first()
            .map(|vector| format!("[{}]", vector.vector.len()));
    }
    if source.starts_with("voice:") {
        return snapshot
            .voice
            .vectors
            .first()
            .map(|vector| format!("[{}]", vector.vector.len()));
    }
    if source.starts_with("scene:")
        && snapshot.kinect.depth_width > 0
        && snapshot.kinect.depth_height > 0
    {
        return Some(format!(
            "{}x{}",
            snapshot.kinect.depth_width, snapshot.kinect.depth_height
        ));
    }
    if source.starts_with("text:") {
        return Some(format!(
            "tokens:{}",
            source.split_whitespace().count().max(1)
        ));
    }
    match modality {
        "vision" => Some("vision:observation".to_string()),
        "audio" => Some("audio:observation".to_string()),
        "depth" | "lidar" => Some("depth:observation".to_string()),
        "touch" => Some("touch:observation".to_string()),
        "motor" | "action" => Some("action:observation".to_string()),
        "language" => Some("text:observation".to_string()),
        _ => None,
    }
}

fn projected_beams_from_observation(
    observation: &MapObservation,
    now_ms: TimeMs,
) -> Vec<MapProjectedBeam> {
    observation
        .range_beams
        .iter()
        .map(|beam| {
            let end = project_beam_endpoint(observation.pose.pose, beam.angle_rad, beam.distance_m);
            MapProjectedBeam {
                origin_x_m: observation.pose.pose.x_m,
                origin_y_m: observation.pose.pose.y_m,
                end_x_m: end.x_m,
                end_y_m: end.y_m,
                angle_rad: beam.angle_rad,
                distance_m: beam.distance_m,
                hit: beam.hit,
                confidence: beam.confidence,
                age_ms: now_ms.saturating_sub(observation.t_ms),
            }
        })
        .collect()
}

fn map_view_cell(cell: &OdomMapCell, resolution_m: f32, now_ms: TimeMs) -> MapViewCell {
    MapViewCell {
        x: cell.key.x,
        y: cell.key.y,
        center_x_m: (cell.key.x as f32 + 0.5) * resolution_m,
        center_y_m: (cell.key.y as f32 + 0.5) * resolution_m,
        occupied_score: cell.occupied_score,
        free_score: cell.free_score,
        confidence: cell.confidence,
        age_ms: now_ms.saturating_sub(cell.last_seen_ms),
    }
}

fn map_semantic_cells(
    snapshot: &WorldSnapshot,
    metadata: Option<&LiveSceneMetadata>,
    now_ms: TimeMs,
) -> Vec<MapSemanticCell> {
    let mut cells = Vec::new();
    cells.extend(memory_semantic_cells(snapshot, now_ms));
    if let Some(metadata) = metadata {
        cells.extend(metadata.objects.iter().filter_map(|object| {
            let kind = semantic_kind_for_object(&object.kind)?;
            Some(MapSemanticCell {
                x_m: object.x_m,
                y_m: object.y_m,
                kind: kind.to_string(),
                score: 1.0,
                confidence: 1.0,
                age_ms: Some(0),
                label: object.label.clone().or_else(|| Some(object.id.clone())),
            })
        }));
    }
    if snapshot.body.charging {
        cells.push(MapSemanticCell {
            x_m: snapshot.body.odometry.x_m,
            y_m: snapshot.body.odometry.y_m,
            kind: "charger/charge".to_string(),
            score: 1.0,
            confidence: 0.9,
            age_ms: Some(0),
            label: Some("charging contact".to_string()),
        });
    }
    cells
}

fn memory_semantic_cells(snapshot: &WorldSnapshot, now_ms: TimeMs) -> Vec<MapSemanticCell> {
    let Some(value) = snapshot
        .to_now(snapshot.body.last_update_ms)
        .extensions
        .get("memory.semantic_map")
        .cloned()
    else {
        return Vec::new();
    };
    let mut cells = Vec::new();
    for (field, kind) in [
        ("danger_cells", "danger"),
        ("charge_cells", "charger/charge"),
        ("social_cells", "social"),
        ("novelty_cells", "novelty"),
    ] {
        if let Some(items) = value.get(field).and_then(|items| items.as_array()) {
            cells.extend(
                items
                    .iter()
                    .filter_map(|item| semantic_cell_from_value(item, kind, now_ms)),
            );
        }
    }
    if let Some(current) = value.get("current") {
        if let Some(cell) = semantic_cell_from_value(current, "current", now_ms) {
            cells.push(cell);
        }
    }
    cells
}

fn semantic_cell_from_value(
    value: &serde_json::Value,
    kind: &str,
    now_ms: TimeMs,
) -> Option<MapSemanticCell> {
    let x_m = value.get("center_x_m")?.as_f64()? as f32;
    let y_m = value.get("center_y_m")?.as_f64()? as f32;
    let last_seen = value.get("last_seen_tick").and_then(|value| value.as_u64());
    Some(MapSemanticCell {
        x_m,
        y_m,
        kind: kind.to_string(),
        score: value
            .get("score")
            .and_then(|value| value.as_f64())
            .unwrap_or(1.0) as f32,
        confidence: value
            .get("confidence")
            .and_then(|value| value.as_f64())
            .unwrap_or(1.0) as f32,
        age_ms: last_seen.map(|seen| now_ms.saturating_sub(seen)),
        label: value
            .get("last_observed_objects")
            .and_then(|value| value.as_array())
            .and_then(|items| items.first())
            .and_then(|value| value.as_str())
            .map(str::to_string),
    })
}

fn semantic_kind_for_object(kind: &str) -> Option<&'static str> {
    match kind {
        "charger" => Some("charger/charge"),
        "person" | "speaker" | "sound_source" => Some("social"),
        _ => None,
    }
}

fn map_event_markers(
    snapshot: &WorldSnapshot,
    metadata: Option<&LiveSceneMetadata>,
    now_ms: TimeMs,
) -> Vec<MapEventMarker> {
    let pose = snapshot.body.odometry;
    let mut events = Vec::new();
    if snapshot.body.flags.bump_left || snapshot.body.flags.bump_right {
        events.push(map_event_at_pose(pose.x_m, pose.y_m, "bump", 1.0, now_ms));
    }
    if snapshot.body.flags.cliff_left
        || snapshot.body.flags.cliff_front_left
        || snapshot.body.flags.cliff_front_right
        || snapshot.body.flags.cliff_right
        || snapshot.body.flags.wheel_drop
    {
        events.push(map_event_at_pose(pose.x_m, pose.y_m, "cliff", 1.0, now_ms));
    }
    if scene_stuck_from_snapshot(snapshot).active {
        events.push(map_event_at_pose(pose.x_m, pose.y_m, "stuck", 0.9, now_ms));
    }
    if snapshot
        .llm_action_proposal
        .as_ref()
        .and_then(|proposal| proposal.safety_vetoed.then_some(()))
        .is_some()
    {
        events.push(map_event_at_pose(
            pose.x_m,
            pose.y_m,
            "safety_override",
            1.0,
            now_ms,
        ));
    }
    if snapshot.body.charging {
        events.push(map_event_at_pose(
            pose.x_m, pose.y_m, "charger", 1.0, now_ms,
        ));
    }
    if let Some(metadata) = metadata {
        events.extend(metadata.objects.iter().filter_map(|object| {
            matches!(
                object.kind.as_str(),
                "charger" | "person" | "speaker" | "sound_source"
            )
            .then(|| MapEventMarker {
                x_m: object.x_m,
                y_m: object.y_m,
                kind: object.kind.clone(),
                confidence: 1.0,
                age_ms: 0,
                label: object.label.clone().or_else(|| Some(object.id.clone())),
            })
        }));
    }
    events
}

fn map_event_at_pose(
    x_m: f32,
    y_m: f32,
    kind: &str,
    confidence: f32,
    now_ms: TimeMs,
) -> MapEventMarker {
    MapEventMarker {
        x_m,
        y_m,
        kind: kind.to_string(),
        confidence,
        age_ms: now_ms.saturating_sub(now_ms),
        label: None,
    }
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
        | EyeFrameFormat::Yuyv422
        | EyeFrameFormat::Uyvy422
        | EyeFrameFormat::BayerGrbg8
        | EyeFrameFormat::BayerRggb8
        | EyeFrameFormat::BayerBggr8
        | EyeFrameFormat::BayerGbrg8 => {
            let rgb = eye_frame_to_rgb(frame)?;
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

fn eye_frame_to_rgb(frame: &netherwick_sensors::EyeFrame) -> Result<Vec<u8>, String> {
    let expected_len = match frame.format {
        EyeFrameFormat::Gray8
        | EyeFrameFormat::BayerGrbg8
        | EyeFrameFormat::BayerRggb8
        | EyeFrameFormat::BayerBggr8
        | EyeFrameFormat::BayerGbrg8 => frame.width as usize * frame.height as usize,
        EyeFrameFormat::Yuyv422 | EyeFrameFormat::Uyvy422 => {
            frame.width as usize * frame.height as usize * 2
        }
        EyeFrameFormat::Rgb8 | EyeFrameFormat::Bgr8 => {
            frame.width as usize * frame.height as usize * 3
        }
        EyeFrameFormat::Mjpeg | EyeFrameFormat::Unknown(_) => {
            return Err(format!("unsupported eye frame format {:?}", frame.format));
        }
    };
    if frame.bytes.len() < expected_len {
        return Err(format!(
            "eye frame has {} bytes, expected at least {}",
            frame.bytes.len(),
            expected_len
        ));
    }
    let bytes = &frame.bytes[..expected_len];
    let mut rgb = Vec::with_capacity(frame.width as usize * frame.height as usize * 3);
    match frame.format {
        EyeFrameFormat::Rgb8 => rgb.extend_from_slice(bytes),
        EyeFrameFormat::Bgr8 => {
            for pixel in bytes.chunks_exact(3) {
                rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
            }
        }
        EyeFrameFormat::Gray8 => {
            for value in bytes {
                rgb.extend_from_slice(&[*value, *value, *value]);
            }
        }
        EyeFrameFormat::Yuyv422 => {
            rgb.extend(yuyv422_to_rgb(bytes));
        }
        EyeFrameFormat::Uyvy422 => {
            rgb.extend(uyvy422_to_rgb(bytes));
        }
        EyeFrameFormat::BayerGrbg8
        | EyeFrameFormat::BayerRggb8
        | EyeFrameFormat::BayerBggr8
        | EyeFrameFormat::BayerGbrg8 => {
            rgb.extend(bayer8_to_rgb(
                bytes,
                frame.width as usize,
                frame.height as usize,
                &frame.format,
            ));
        }
        EyeFrameFormat::Mjpeg | EyeFrameFormat::Unknown(_) => {}
    }
    Ok(rgb)
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
                        training_ledger: entry
                            .get("training")
                            .and_then(|value| value.get("ledger"))
                            .and_then(|value| value.as_str())
                            .map(str::to_string),
                        behavior_report_path: entry
                            .get("reports")
                            .and_then(|value| value.get("behavior"))
                            .and_then(|value| value.as_str())
                            .map(str::to_string),
                        scenario_report_path: entry
                            .get("reports")
                            .and_then(|value| value.get("scenario"))
                            .and_then(|value| value.as_str())
                            .map(str::to_string),
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
        Some(&accumulated_point_cloud_for_frames(&frames[..=query.frame])),
        None,
        HardwareControlStatus::unavailable("capture replay is not a hardware cockpit session"),
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

fn accumulated_point_cloud_for_frames(
    frames: &[netherwick_worldlab::CaptureFrameRecord],
) -> VoxelPointCloud {
    let mut cloud = VoxelPointCloud::default();
    for frame in frames {
        cloud.observe_snapshot(&frame.snapshot, frame.t_ms);
    }
    cloud
}

fn accumulated_scene_points(
    point_cloud: &VoxelPointCloud,
    now_ms: TimeMs,
) -> Vec<SceneAccumulatedPoint> {
    const MAX_SCENE_POINTS: usize = 8_000;
    let points = point_cloud.points();
    let stride = points.len().div_ceil(MAX_SCENE_POINTS).max(1);
    points
        .into_iter()
        .step_by(stride)
        .map(|point| scene_accumulated_point(point, now_ms))
        .collect()
}

fn scene_accumulated_point(point: VoxelPoint, now_ms: TimeMs) -> SceneAccumulatedPoint {
    let [r, g, b] = point.color_rgb.unwrap_or(if point.stable {
        [124, 230, 174]
    } else {
        [190, 194, 246]
    });
    SceneAccumulatedPoint {
        x: point.position.x_m,
        y: point.position.y_m,
        z: point.position.z_m,
        r,
        g,
        b,
        confidence: point.confidence,
        age_ms: now_ms.saturating_sub(point.last_seen_ms),
        stable: point.stable,
        transient: point.transient,
    }
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
    point_cloud: Option<&VoxelPointCloud>,
    retina_status: Option<RetinaStatusInfo>,
    hardware_control: HardwareControlStatus,
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
    let mut kinect = scene_kinect_from_snapshot(snapshot, sensor_calibration, &mut warnings);
    if let Some(point_cloud) = point_cloud {
        kinect.accumulated_points =
            accumulated_scene_points(point_cloud, snapshot.body.last_update_ms);
        kinect.accumulated_summary = Some(point_cloud.summary());
        kinect.local_world_belief = Some(point_cloud.local_world_belief());
        if !kinect.accumulated_points.is_empty() {
            kinect.coordinate_system = Some("world".to_string());
        }
    }
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
        hardware_control,
        training_mode,
        ledger_path,
        frames_written,
        transitions_written,
        models_loaded,
        model_modes,
        behavior_nodes,
        experience_forge: None,
        action_selector_mode,
        weights_updating,
        t_ms: body.last_update_ms,
        body: SceneBody {
            x_m: body.odometry.x_m,
            y_m: body.odometry.y_m,
            heading_rad: body.odometry.heading_rad,
            battery_level: body.battery_level,
            charging: body.charging,
            bump_left: body.flags.bump_left,
            bump_right: body.flags.bump_right,
            cliff: body.flags.cliff_left
                || body.flags.cliff_front_left
                || body.flags.cliff_front_right
                || body.flags.cliff_right,
            wheel_drop: body.flags.wheel_drop,
        },
        range: scene_range_from_snapshot(snapshot),
        eye,
        kinect,
        surface_perception: None,
        world_belief_layers: vec![
            "current rays",
            "raw point cloud",
            "accumulated occupancy",
            "stable wall candidates",
        ],
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
        EyeFrameFormat::Yuyv422 | EyeFrameFormat::Uyvy422 => {
            for pair in frame.bytes.chunks_exact(4).take(pixels.div_ceil(2)) {
                let values = match frame.format {
                    EyeFrameFormat::Uyvy422 => [pair[1], pair[3]],
                    _ => [pair[0], pair[2]],
                };
                for value in values {
                    let luma = value as f32 / 255.0;
                    luma_sum += luma;
                    if luma > 0.08 {
                        non_background += 1;
                    }
                }
            }
        }
        EyeFrameFormat::BayerGrbg8
        | EyeFrameFormat::BayerRggb8
        | EyeFrameFormat::BayerBggr8
        | EyeFrameFormat::BayerGbrg8 => {
            for value in frame.bytes.iter().take(pixels) {
                let luma = *value as f32 / 255.0;
                luma_sum += luma;
                if luma > 0.08 {
                    non_background += 1;
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
        | EyeFrameFormat::Yuyv422
        | EyeFrameFormat::Uyvy422
        | EyeFrameFormat::BayerGrbg8
        | EyeFrameFormat::BayerRggb8
        | EyeFrameFormat::BayerBggr8
        | EyeFrameFormat::BayerGbrg8 => {
            let rgb = match eye_frame_to_rgb(frame) {
                Ok(rgb) => rgb,
                Err(error) => return (None, Some(error)),
            };
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

fn bayer8_to_rgb(bytes: &[u8], width: usize, height: usize, format: &EyeFrameFormat) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let (r, g, b) = bayer_pixel_to_rgb(bytes, width, height, x, y, format);
            rgb.extend_from_slice(&[r, g, b]);
        }
    }
    rgb
}

fn bayer_pixel_to_rgb(
    bytes: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    format: &EyeFrameFormat,
) -> (u8, u8, u8) {
    let value = bayer_sample(bytes, width, x, y);
    match bayer_color_at(x, y, format) {
        BayerColor::Red => (
            value,
            average_bayer_neighbors(bytes, width, height, x, y, format, &[BayerColor::Green]),
            average_bayer_neighbors(bytes, width, height, x, y, format, &[BayerColor::Blue]),
        ),
        BayerColor::Green => (
            average_bayer_neighbors(bytes, width, height, x, y, format, &[BayerColor::Red]),
            value,
            average_bayer_neighbors(bytes, width, height, x, y, format, &[BayerColor::Blue]),
        ),
        BayerColor::Blue => (
            average_bayer_neighbors(bytes, width, height, x, y, format, &[BayerColor::Red]),
            average_bayer_neighbors(bytes, width, height, x, y, format, &[BayerColor::Green]),
            value,
        ),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BayerColor {
    Red,
    Green,
    Blue,
}

fn bayer_color_at(x: usize, y: usize, format: &EyeFrameFormat) -> BayerColor {
    let even_x = x % 2 == 0;
    let even_y = y % 2 == 0;
    match format {
        EyeFrameFormat::BayerGrbg8 => match (even_y, even_x) {
            (true, true) | (false, false) => BayerColor::Green,
            (true, false) => BayerColor::Red,
            (false, true) => BayerColor::Blue,
        },
        EyeFrameFormat::BayerRggb8 => match (even_y, even_x) {
            (true, true) => BayerColor::Red,
            (true, false) | (false, true) => BayerColor::Green,
            (false, false) => BayerColor::Blue,
        },
        EyeFrameFormat::BayerBggr8 => match (even_y, even_x) {
            (true, true) => BayerColor::Blue,
            (true, false) | (false, true) => BayerColor::Green,
            (false, false) => BayerColor::Red,
        },
        EyeFrameFormat::BayerGbrg8 => match (even_y, even_x) {
            (true, true) | (false, false) => BayerColor::Green,
            (true, false) => BayerColor::Blue,
            (false, true) => BayerColor::Red,
        },
        _ => BayerColor::Green,
    }
}

fn average_bayer_neighbors(
    bytes: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    format: &EyeFrameFormat,
    colors: &[BayerColor],
) -> u8 {
    let mut sum = 0usize;
    let mut count = 0usize;
    let min_y = y.saturating_sub(1);
    let max_y = (y + 1).min(height.saturating_sub(1));
    let min_x = x.saturating_sub(1);
    let max_x = (x + 1).min(width.saturating_sub(1));
    for ny in min_y..=max_y {
        for nx in min_x..=max_x {
            if nx == x && ny == y {
                continue;
            }
            if colors.contains(&bayer_color_at(nx, ny, format)) {
                sum += bayer_sample(bytes, width, nx, ny) as usize;
                count += 1;
            }
        }
    }
    if count == 0 {
        bayer_sample(bytes, width, x, y)
    } else {
        (sum / count) as u8
    }
}

fn bayer_sample(bytes: &[u8], width: usize, x: usize, y: usize) -> u8 {
    bytes[y * width + x]
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

fn uyvy422_to_rgb(bytes: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(bytes.len() / 2 * 3);
    for pair in bytes.chunks_exact(4) {
        let u = pair[0];
        let y0 = pair[1];
        let v = pair[2];
        let y1 = pair[3];
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
    let (points, diagnostics) = depth_points(&snapshot.kinect, calibration);
    if points.is_empty() {
        warnings.push("no point cloud stream".to_string());
    }
    if diagnostics.coordinate_system == "depth_image_unknown" {
        warnings.push(
            "Kinect depth frame has no width/height metadata; using legacy approximate projection"
                .to_string(),
        );
    }
    SceneKinect {
        points,
        accumulated_points: Vec::new(),
        accumulated_summary: None,
        local_world_belief: None,
        skeletons: snapshot.kinect.skeletons.clone(),
        coordinate_system: Some(diagnostics.coordinate_system.clone()),
        diagnostics,
    }
}

fn depth_points(
    kinect: &KinectSense,
    calibration: Option<SceneSensorCalibration>,
) -> (Vec<ScenePoint>, SceneKinectDiagnostics) {
    const MAX_POINTS: usize = 2_000;
    let depth_m = &kinect.depth_m;
    if depth_m.is_empty() {
        return (
            Vec::new(),
            SceneKinectDiagnostics {
                coordinate_system: "none".to_string(),
                sample_stride: 1,
                ..SceneKinectDiagnostics::default()
            },
        );
    }
    if let Some(calibration) = calibration {
        if depth_m.len() == calibration.compact_depth_beam_count {
            let points = range_beam_points(depth_m, calibration);
            let stats = depth_stats(depth_m, 1, 0.0, 8.0, "robot", 0, 0)
                .with_floor_stats(floor_stats_from_scene_points(&points));
            return (points, stats);
        }
    }
    if let Some(frame) = KinectDepthProjection::from_kinect(kinect) {
        return project_depth_image(depth_m, frame, calibration);
    }
    if depth_m.len() == 640 * 480 {
        return project_depth_image(
            depth_m,
            KinectDepthProjection {
                width: 640,
                height: 480,
                fx: 594.0,
                fy: 591.0,
                cx: 339.0,
                cy: 242.0,
                min_depth_m: 0.4,
                max_depth_m: 8.0,
                coordinate_system: "kinect_camera".to_string(),
            },
            calibration,
        );
    }
    let width = (depth_m.len() as f32).sqrt().ceil().max(1.0) as usize;
    let height = depth_m.len().div_ceil(width).max(1);
    let stride = (depth_m.len().div_ceil(MAX_POINTS)).max(1);
    let points = depth_m
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
        .collect();
    (
        points,
        depth_stats(
            depth_m,
            stride,
            0.0,
            8.0,
            "depth_image_unknown",
            width as u32,
            height as u32,
        ),
    )
}

#[derive(Clone, Debug)]
struct KinectDepthProjection {
    width: usize,
    height: usize,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    min_depth_m: f32,
    max_depth_m: f32,
    coordinate_system: String,
}

impl KinectDepthProjection {
    fn from_kinect(kinect: &KinectSense) -> Option<Self> {
        let width = usize::try_from(kinect.depth_width).ok()?;
        let height = usize::try_from(kinect.depth_height).ok()?;
        if width == 0 || height == 0 || width.checked_mul(height)? != kinect.depth_m.len() {
            return None;
        }
        let fx = if kinect.depth_fx > 0.0 {
            kinect.depth_fx
        } else {
            594.0
        };
        let fy = if kinect.depth_fy > 0.0 {
            kinect.depth_fy
        } else {
            591.0
        };
        let cx = if kinect.depth_cx > 0.0 {
            kinect.depth_cx
        } else {
            (width as f32 - 1.0) * 0.5
        };
        let cy = if kinect.depth_cy > 0.0 {
            kinect.depth_cy
        } else {
            (height as f32 - 1.0) * 0.5
        };
        let max_depth_m = if kinect.max_depth_m > 0.0 {
            kinect.max_depth_m
        } else {
            8.0
        };
        Some(Self {
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            min_depth_m: kinect.min_depth_m.max(0.0),
            max_depth_m,
            coordinate_system: kinect
                .depth_coordinate_system
                .clone()
                .filter(|system| system != "kinect_depth_image")
                .unwrap_or_else(|| "kinect_camera".to_string()),
        })
    }
}

fn project_depth_image(
    depth_m: &[f32],
    frame: KinectDepthProjection,
    calibration: Option<SceneSensorCalibration>,
) -> (Vec<ScenePoint>, SceneKinectDiagnostics) {
    const MAX_POINTS: usize = 2_000;
    let stride = (depth_m.len().div_ceil(MAX_POINTS)).max(1);
    let mut points = Vec::with_capacity(MAX_POINTS.min(depth_m.len()));
    let calibrated = calibration.map(|calibration| DepthExtrinsics::from(calibration));
    for (index, depth) in depth_m.iter().enumerate().step_by(stride) {
        if !depth.is_finite() || *depth <= 0.0 {
            continue;
        }
        if *depth < frame.min_depth_m || *depth > frame.max_depth_m {
            continue;
        }
        let u = (index % frame.width) as f32;
        let v = (index / frame.width) as f32;
        let z = *depth;
        let x = (u - frame.cx) * z / frame.fx.max(f32::EPSILON);
        let y = (v - frame.cy) * z / frame.fy.max(f32::EPSILON);
        let shade =
            ((1.0 - (z / frame.max_depth_m.max(f32::EPSILON))).clamp(0.15, 1.0) * 255.0) as u8;
        let point = if let Some(extrinsics) = calibrated {
            let robot = camera_point_to_robot([x, y, z], extrinsics);
            ScenePoint {
                x: robot[1],
                y: robot[2],
                z: robot[0],
                r: shade,
                g: shade,
                b: shade,
            }
        } else {
            ScenePoint {
                x,
                y,
                z,
                r: shade,
                g: shade,
                b: shade,
            }
        };
        points.push(point);
    }
    let coordinate_system = if calibrated.is_some() {
        "robot"
    } else {
        &frame.coordinate_system
    };
    let diagnostics = depth_stats(
        depth_m,
        stride,
        frame.min_depth_m,
        frame.max_depth_m,
        coordinate_system,
        frame.width as u32,
        frame.height as u32,
    )
    .with_floor_stats(floor_stats_from_scene_points(&points));
    (points, diagnostics)
}

fn depth_stats(
    depth_m: &[f32],
    sample_stride: usize,
    min_depth_m: f32,
    max_depth_m: f32,
    coordinate_system: &str,
    depth_width: u32,
    depth_height: u32,
) -> SceneKinectDiagnostics {
    let mut valid = Vec::new();
    let mut skipped = 0usize;
    let mut clipped = 0usize;
    for depth in depth_m {
        if !depth.is_finite() || *depth <= 0.0 {
            skipped = skipped.saturating_add(1);
        } else if *depth < min_depth_m || *depth > max_depth_m {
            clipped = clipped.saturating_add(1);
        } else {
            valid.push(*depth);
        }
    }
    valid.sort_by(|left, right| left.total_cmp(right));
    let median_depth_m = if valid.is_empty() {
        None
    } else {
        Some(valid[valid.len() / 2])
    };
    SceneKinectDiagnostics {
        depth_width,
        depth_height,
        valid_depth_count: valid.len(),
        skipped_depth_count: skipped,
        clipped_depth_count: clipped,
        min_depth_m: valid.first().copied(),
        median_depth_m,
        max_depth_m: valid.last().copied(),
        sample_stride,
        coordinate_system: coordinate_system.to_string(),
        below_floor_count: 0,
        below_floor_ratio: 0.0,
        min_z_m: None,
        median_z_m: None,
    }
}

impl SceneKinectDiagnostics {
    fn with_floor_stats(mut self, stats: FloorPointStats) -> Self {
        self.below_floor_count = stats.below_floor_count;
        self.below_floor_ratio = stats.below_floor_ratio;
        self.min_z_m = stats.min_z_m;
        self.median_z_m = stats.median_z_m;
        self
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FloorPointStats {
    below_floor_count: usize,
    below_floor_ratio: f32,
    min_z_m: Option<f32>,
    median_z_m: Option<f32>,
}

fn floor_stats_from_scene_points(points: &[ScenePoint]) -> FloorPointStats {
    let mut heights = points
        .iter()
        .map(|point| point.y)
        .filter(|height| height.is_finite())
        .collect::<Vec<_>>();
    if heights.is_empty() {
        return FloorPointStats::default();
    }
    heights.sort_by(|left, right| left.total_cmp(right));
    let below_floor_count = heights.iter().filter(|height| **height < 0.0).count();
    FloorPointStats {
        below_floor_count,
        below_floor_ratio: below_floor_count as f32 / heights.len() as f32,
        min_z_m: heights.first().copied(),
        median_z_m: heights.get(heights.len() / 2).copied(),
    }
}

#[derive(Clone, Copy, Debug)]
struct DepthExtrinsics {
    forward_m: f32,
    height_m: f32,
    pitch_rad: f32,
    roll_rad: f32,
    yaw_rad: f32,
}

impl From<SceneSensorCalibration> for DepthExtrinsics {
    fn from(calibration: SceneSensorCalibration) -> Self {
        Self {
            forward_m: calibration.depth_camera_forward_m(),
            height_m: calibration.depth_camera_height_m(),
            pitch_rad: calibration.depth_camera_pitch_rad(),
            roll_rad: calibration.camera_roll_rad,
            yaw_rad: calibration.camera_yaw_rad,
        }
    }
}

fn camera_point_to_robot(camera: [f32; 3], extrinsics: DepthExtrinsics) -> [f32; 3] {
    let base = [camera[2], -camera[0], -camera[1]];
    apply_robot_extrinsics(base, extrinsics)
}

fn apply_robot_extrinsics(base: [f32; 3], extrinsics: DepthExtrinsics) -> [f32; 3] {
    let rotated = rotate_robot_extrinsic(
        base,
        extrinsics.pitch_rad,
        extrinsics.roll_rad,
        extrinsics.yaw_rad,
    );
    [
        rotated[0] + extrinsics.forward_m,
        rotated[1],
        rotated[2] + extrinsics.height_m,
    ]
}

fn rotate_robot_extrinsic(
    point: [f32; 3],
    pitch_rad: f32,
    roll_rad: f32,
    yaw_rad: f32,
) -> [f32; 3] {
    let (pitch_sin, pitch_cos) = pitch_rad.sin_cos();
    let mut x = point[0] * pitch_cos + point[2] * pitch_sin;
    let y = point[1];
    let mut z = -point[0] * pitch_sin + point[2] * pitch_cos;

    let (roll_sin, roll_cos) = roll_rad.sin_cos();
    let rolled_y = y * roll_cos - z * roll_sin;
    z = y * roll_sin + z * roll_cos;

    let (yaw_sin, yaw_cos) = yaw_rad.sin_cos();
    let yawed_x = x * yaw_cos - rolled_y * yaw_sin;
    let yawed_y = x * yaw_sin + rolled_y * yaw_cos;
    x = yawed_x;

    [x, yawed_y, z]
}

fn range_beam_points(depth_m: &[f32], calibration: SceneSensorCalibration) -> Vec<ScenePoint> {
    let beam_count = depth_m.len().max(1);
    let extrinsics = DepthExtrinsics::from(calibration);
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
            let robot = apply_robot_extrinsics(
                [angle.cos() * distance, angle.sin() * distance, 0.0],
                extrinsics,
            );
            Some(ScenePoint {
                x: robot[1],
                y: robot[2],
                z: robot[0],
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
    let source = request.source.unwrap_or(ReignSource::WebRemote);
    let mut command = sanitize_command(request.command)?;
    let mut ttl_ms = request.ttl_ms.unwrap_or_else(|| command.default_ttl_ms());
    if state.hardware_control_status(now_ms).available {
        command = sanitize_hardware_command(command)?;
        ttl_ms = ttl_ms.clamp(HARDWARE_TTL_MIN_MS, HARDWARE_TTL_MAX_MS);
        enforce_hardware_command_gate(&state, &source, &request.mode, &command, now_ms)?;
    }
    let input = ReignInput {
        id: Uuid::new_v4(),
        issued_at_ms: now_ms,
        expires_at_ms: now_ms.saturating_add(ttl_ms.clamp(100, 30_000)),
        source,
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

async fn post_hardware_arm(
    State(state): State<ReignServerState>,
    Json(request): Json<HardwareArmRequest>,
) -> Result<Json<HardwareControlStatus>, ReignApiError> {
    Ok(Json(state.set_hardware_armed(request.armed)?))
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
        ReignCommand::Drive {
            forward,
            turn,
            duration_ms,
        } => ReignCommand::Drive {
            forward: finite_axis(forward)?,
            turn: finite_axis(turn)?,
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

fn sanitize_hardware_command(command: ReignCommand) -> Result<ReignCommand, ReignApiError> {
    Ok(match command {
        ReignCommand::Go {
            intensity,
            duration_ms,
        } => ReignCommand::Go {
            intensity: finite_intensity(intensity)?.min(HARDWARE_MAX_FORWARD_INTENSITY),
            duration_ms: duration_ms.clamp(HARDWARE_TTL_MIN_MS, HARDWARE_TTL_MAX_MS),
        },
        ReignCommand::Reverse {
            intensity,
            duration_ms,
        } => ReignCommand::Reverse {
            intensity: finite_intensity(intensity)?.min(HARDWARE_MAX_FORWARD_INTENSITY),
            duration_ms: duration_ms.clamp(HARDWARE_TTL_MIN_MS, HARDWARE_TTL_MAX_MS),
        },
        ReignCommand::Drive {
            forward,
            turn,
            duration_ms,
        } => ReignCommand::Drive {
            forward: finite_axis(forward)?.clamp(
                -HARDWARE_MAX_FORWARD_INTENSITY,
                HARDWARE_MAX_FORWARD_INTENSITY,
            ),
            turn: finite_axis(turn)?
                .clamp(-HARDWARE_MAX_TURN_INTENSITY, HARDWARE_MAX_TURN_INTENSITY),
            duration_ms: duration_ms.clamp(HARDWARE_TTL_MIN_MS, HARDWARE_TTL_MAX_MS),
        },
        ReignCommand::Turn {
            direction,
            intensity,
            duration_ms,
        } => ReignCommand::Turn {
            direction,
            intensity: finite_intensity(intensity)?.min(HARDWARE_MAX_TURN_INTENSITY),
            duration_ms: duration_ms.clamp(HARDWARE_TTL_MIN_MS, HARDWARE_TTL_MAX_MS),
        },
        ReignCommand::Stop => ReignCommand::Stop,
        _ => {
            return Err(ReignApiError::forbidden(
                "hardware cockpit only accepts Stop, Go, Reverse, Drive, and Turn",
            ))
        }
    })
}

fn enforce_hardware_command_gate(
    state: &ReignServerState,
    source: &ReignSource,
    mode: &ReignMode,
    command: &ReignCommand,
    now_ms: TimeMs,
) -> Result<(), ReignApiError> {
    if matches!(command, ReignCommand::Stop) {
        return Ok(());
    }
    if source != &ReignSource::WebRemote {
        return Err(ReignApiError::forbidden(
            "hardware cockpit only accepts WebRemote drive commands",
        ));
    }
    if mode != &ReignMode::Direct {
        return Err(ReignApiError::forbidden(
            "hardware cockpit drive commands must use Direct mode",
        ));
    }
    let status = state.hardware_control_status(now_ms);
    if !status.armed {
        return Err(ReignApiError::forbidden(
            "hardware cockpit is disarmed; movement rejected",
        ));
    }
    if let Some(reason) = status.reason {
        return Err(ReignApiError::forbidden(format!(
            "hardware cockpit movement rejected: {reason}"
        )));
    }
    Ok(())
}

fn finite_intensity(value: f32) -> Result<f32, ReignApiError> {
    if !value.is_finite() {
        return Err(ReignApiError::bad_request("intensity must be finite"));
    }
    Ok(value.clamp(0.0, 1.0))
}

fn finite_axis(value: f32) -> Result<f32, ReignApiError> {
    if !value.is_finite() {
        return Err(ReignApiError::bad_request("drive axis must be finite"));
    }
    Ok(value.clamp(-1.0, 1.0))
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
body{margin:0;min-height:100vh;display:grid;grid-template-columns:minmax(0,1fr) minmax(360px,460px)}
main{display:grid;place-items:center;background:#070809}
canvas{width:min(100vw,calc((100vh - 28px)*1.333));height:auto;image-rendering:pixelated;background:#111;border:1px solid #34383c}
aside{border-left:1px solid #2f3337;padding:16px;background:#171a1d;overflow:auto;max-height:100vh}
h1{font-size:16px;margin:0 0 14px}
h2{font-size:13px;margin:18px 0 8px;color:#c9d2da}
a{color:#8bd3ff}
dl{display:grid;grid-template-columns:auto 1fr;gap:8px 12px;margin:0 0 18px}
dt{color:#9aa4ad}
dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
#status{color:#ffcf7a;margin-bottom:12px;min-height:20px}
.beams{display:flex;align-items:end;gap:5px;height:96px;margin-top:8px}
.beam{flex:1;background:#60d394;min-height:2px}
.graph-controls{display:grid;grid-template-columns:1fr auto;gap:8px;margin:8px 0}
.graph-controls input,.graph-controls select,.graph-controls button{min-width:0;background:#101316;color:#eef1f3;border:1px solid #394049;border-radius:6px;padding:7px}
.graph-controls select{grid-column:1 / 3}
.graph-summary{color:#9aa4ad;font-size:12px;margin:4px 0 8px}
.graph-list{display:grid;gap:6px;margin:0}
.graph-node,.graph-edge,.graph-memory{border:1px solid #30363d;border-radius:6px;background:#111519;padding:8px;overflow-wrap:anywhere}
.graph-node[data-type="sensation"]{border-left:3px solid #60d394}
.graph-node[data-type="impression"]{border-left:3px solid #8bd3ff}
.graph-node[data-type="experience"]{border-left:3px solid #ffcf7a}
.graph-node[data-type="prediction"]{border-left:3px solid #d7a8ff}
.graph-node[data-type="memory_link"]{border-left:3px solid #f5a97f}
.graph-title{font-weight:650;font-size:12px}
.graph-meta{color:#9aa4ad;font-size:11px;margin-top:3px}
.graph-detail{font-size:12px;margin-top:5px;line-height:1.35}
.graph-edge{font-size:12px;color:#c9d2da}
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
  <h2>Embodied lineage</h2>
  <div class="graph-controls">
    <input id="graph_query" placeholder="Search ids, text, model">
    <button id="graph_refresh" type="button">Refresh</button>
    <select id="graph_modality">
      <option value="">All modalities</option>
      <option value="vision">Vision</option>
      <option value="audio">Audio</option>
      <option value="depth">Depth</option>
      <option value="lidar">Lidar</option>
      <option value="touch">Touch</option>
      <option value="odometry">Odometry</option>
      <option value="memory">Memory</option>
      <option value="language">Language</option>
      <option value="other">Other</option>
    </select>
  </div>
  <div class="graph-summary" id="graph_summary">waiting for embodied experience...</div>
  <div class="graph-list" id="graph_nodes"></div>
  <h2>Edges</h2>
  <div class="graph-list" id="graph_edges"></div>
  <h2>Recent memories</h2>
  <div class="graph-list" id="graph_memories"></div>
</aside>
<script>
const canvas = document.getElementById('eye');
const ctx = canvas.getContext('2d');
const fields = Object.fromEntries(['t','x','y','heading','battery','nearest','gps_lat','gps_lon','gps_alt','eye_format','eye_age','ear_age'].map(id => [id, document.getElementById(id)]));
const status = document.getElementById('status');
const beams = document.getElementById('beams');
const graphQuery = document.getElementById('graph_query');
const graphModality = document.getElementById('graph_modality');
const graphSummary = document.getElementById('graph_summary');
const graphNodes = document.getElementById('graph_nodes');
const graphEdges = document.getElementById('graph_edges');
const graphMemories = document.getElementById('graph_memories');
let latestGraph = null;
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
  const isUyvy = fmt === 'Uyvy422' || (typeof fmt === 'object' && fmt.Uyvy422 !== undefined);
  const isMjpg = fmt === 'Mjpeg' || (typeof fmt === 'object' && fmt.Mjpeg !== undefined) || (typeof fmt === 'object' && JSON.stringify(fmt).includes('MJPG'));
  if(isRgb || isBgr || isGray || isYuyv || isUyvy){
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
    }else if(isYuyv || isUyvy){
      for(let source = 0, target = 0; target + 7 < image.data.length && source + 3 < frame.bytes.length; source += 4, target += 8){
        const y0 = isUyvy ? frame.bytes[source + 1] : frame.bytes[source];
        const u = isUyvy ? frame.bytes[source] : frame.bytes[source + 1];
        const y1 = isUyvy ? frame.bytes[source + 3] : frame.bytes[source + 2];
        const v = isUyvy ? frame.bytes[source + 2] : frame.bytes[source + 3];
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
function drawEyeDataUrl(dataUrl){
  if(!dataUrl) return false;
  const img = new Image();
  img.onload = () => {
    if(canvas.width !== img.width || canvas.height !== img.height){
      canvas.width = img.width; canvas.height = img.height;
    }
    ctx.drawImage(img, 0, 0);
  };
  img.src = dataUrl;
  return true;
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
  const mode = String(session.mode || '').toLowerCase();
  const virtualMode =
    session.source === 'sim' ||
    mode.includes('virtual') ||
    mode.includes('read-only') ||
    mode.includes('readonly');
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
function graphMatches(node, query, modality){
  if(modality && node.modality !== modality) return false;
  if(!query) return true;
  return JSON.stringify(node).toLowerCase().includes(query);
}
function escapeHtml(value){
  return String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}
function renderGraph(){
  if(!latestGraph){
    graphSummary.textContent = 'waiting for embodied experience...';
    graphNodes.replaceChildren();
    graphEdges.replaceChildren();
    graphMemories.replaceChildren();
    return;
  }
  const query = graphQuery.value.trim().toLowerCase();
  const modality = graphModality.value;
  const vectorsByNode = new Map();
  for(const vector of latestGraph.vector_metadata || []){
    const list = vectorsByNode.get(vector.owner_node_id) || [];
    list.push(vector);
    vectorsByNode.set(vector.owner_node_id, list);
  }
  const visibleNodes = (latestGraph.nodes || []).filter(node => graphMatches(node, query, modality));
  const visibleIds = new Set(visibleNodes.map(node => node.id));
  graphSummary.textContent = `${visibleNodes.length}/${(latestGraph.nodes || []).length} nodes, ${(latestGraph.edges || []).length} edges | ${latestGraph.summary || 'no summary'}`;
  graphNodes.replaceChildren(...visibleNodes.map(node => {
    const item = document.createElement('div');
    item.className = 'graph-node';
    item.dataset.type = node.node_type;
    const vectorText = (vectorsByNode.get(node.id) || [])
      .map(vector => `${vector.model_id} dim=${vector.dim} source=${vector.source_sensation_id}`)
      .join(' | ');
    item.innerHTML = `
      <div class="graph-title">${escapeHtml(node.label)} ${node.derived ? '<span class="graph-meta">(derived)</span>' : ''}</div>
      <div class="graph-meta">${escapeHtml(node.node_type)} ${escapeHtml(node.modality || '')} ${escapeHtml(node.payload_kind || '')} | ${escapeHtml(node.entity_id)}</div>
      ${node.detail ? `<div class="graph-detail">${escapeHtml(node.detail)}</div>` : ''}
      ${vectorText ? `<div class="graph-meta">vector: ${escapeHtml(vectorText)}</div>` : ''}
    `;
    return item;
  }));
  const visibleEdges = (latestGraph.edges || []).filter(edge =>
    visibleIds.has(edge.from) || visibleIds.has(edge.to) || (!query && !modality)
  );
  graphEdges.replaceChildren(...visibleEdges.slice(0, 80).map(edge => {
    const item = document.createElement('div');
    item.className = 'graph-edge';
    item.textContent = `${edge.from} -> ${edge.to} (${edge.relation})`;
    return item;
  }));
  const visibleMemories = (latestGraph.recent_memories || []).filter(memory =>
    !query || JSON.stringify(memory).toLowerCase().includes(query)
  );
  graphMemories.replaceChildren(...visibleMemories.map(memory => {
    const item = document.createElement('div');
    item.className = 'graph-memory';
    item.innerHTML = `<div class="graph-title">${escapeHtml(memory.relation)} ${fmt(memory.score, 2)}</div><div class="graph-meta">${escapeHtml(memory.target_id)}</div>${memory.text ? `<div class="graph-detail">${escapeHtml(memory.text)}</div>` : ''}`;
    return item;
  }));
  if(!visibleMemories.length){
    const item = document.createElement('div');
    item.className = 'graph-meta';
    item.textContent = 'none';
    graphMemories.replaceChildren(item);
  }
}
async function refreshGraph(){
  try{
    const response = await fetch('/api/experience/lineage', {cache:'no-store'});
    if(!response.ok) throw new Error(await response.text());
    latestGraph = await response.json();
    renderGraph();
  }catch(error){
    graphSummary.textContent = error.message || String(error);
  }
}
async function refresh(){
  try{
    const sceneResponse = await fetch('/view/scene', {cache:'no-store'});
    if(sceneResponse.ok){
      const scenePacket = await sceneResponse.json();
      const body = scenePacket.body;
      fields.t.textContent = `${scenePacket.t_ms} ms`;
      fields.x.textContent = `${fmt(body.x_m)} m`;
      fields.y.textContent = `${fmt(body.y_m)} m`;
      fields.heading.textContent = `${fmt(body.heading_rad)} rad`;
      fields.battery.textContent = `${fmt(body.battery_level * 100, 1)}%`;
      fields.nearest.textContent = scenePacket.range?.nearest_m == null ? '-' : `${fmt(scenePacket.range.nearest_m)} m`;
      fields.gps_lat.textContent = '-';
      fields.gps_lon.textContent = '-';
      fields.gps_alt.textContent = '-';
      if(scenePacket.eye){
        const eye = scenePacket.eye;
        const isAuth = eye.authoritative ? " (auth)" : " (symbolic)";
        const statusText = eye.retina_connected ? `connected, age ${eye.retina_last_frame_age_ms}ms` : "disconnected";
        fields.eye_format.textContent = `${eye.width}x${eye.height} luma ${fmt(eye.mean_luma, 2)} | source: ${eye.source}${isAuth} | retina: ${statusText} | rx: ${eye.frames_received} tx: ${eye.frames_written_to_ledger}`;
        fields.eye_format.style.color = '';
        fields.eye_format.style.fontWeight = '';
        fields.eye_age.textContent = '-';
        drawEyeDataUrl(eye.data_url);
      }else{
        fields.eye_format.textContent = '-';
        fields.eye_age.textContent = '-';
      }
      fields.ear_age.textContent = '-';
      drawBeams((scenePacket.range?.beams || []).map(beam => beam.distance_m));
      status.textContent = 'live';
      refreshGraph();
      return;
    }
    const response = await fetch('/view/snapshot', {cache:'no-store'});
    if(!response.ok) throw new Error(await response.text());
    const snapshot = await response.json();
    const scenePacket = null;
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
    refreshGraph();
  }catch(error){
    status.textContent = 'waiting for frames...';
  }finally{
    setTimeout(refresh, 250);
  }
}
graphQuery.addEventListener('input', renderGraph);
graphModality.addEventListener('change', renderGraph);
document.getElementById('graph_refresh').addEventListener('click', refreshGraph);
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
#reign{right:12px;top:12px;width:282px;min-width:250px;border-color:#36424d;background:rgba(11,16,22,.84);color:#dce8f2}
#reign .panel-content{display:grid;gap:8px}
#reign strong{display:block;font-size:12px;margin-bottom:6px;color:#91d7ff}
#reign div{font-variant-numeric:tabular-nums}
#reign-state{min-height:17px;color:#ffd083}
.cockpit{border-top:1px solid #344350;padding-top:8px;display:grid;gap:7px}
.cockpit-arm{display:flex;align-items:center;gap:7px;font-size:12px;color:#dce8f2}
.cockpit-arm input{width:16px;height:16px}
#hardware-state{font-size:11px;color:#ffbf7a;min-height:15px}
.trace-label{font-size:11px;color:#9fb0bf}
#trace-map{width:100%;height:auto;display:block;background:#081015;border:1px solid #2d3d4a;border-radius:4px}
#entity-graph{width:100%;height:auto;display:block;background:#070b0f;border:1px solid #2d3d4a;border-radius:4px;margin-top:8px}
#entity-graph-summary{font-size:10px;color:#b7c8d8;min-height:13px;margin-top:4px;font-variant-numeric:tabular-nums}
.reign-pad{display:grid;grid-template-columns:132px 1fr;gap:10px;align-items:center}
#reign-joystick{position:relative;width:132px;aspect-ratio:1;border:1px solid #455565;border-radius:50%;background:radial-gradient(circle at 50% 50%,rgba(79,149,188,.24),rgba(15,23,31,.72) 64%);touch-action:none;cursor:pointer}
#reign-joystick::before,#reign-joystick::after{content:"";position:absolute;background:rgba(170,196,216,.22)}
#reign-joystick::before{left:12px;right:12px;top:50%;height:1px}
#reign-joystick::after{top:12px;bottom:12px;left:50%;width:1px}
#reign-stick{position:absolute;left:50%;top:50%;width:34px;height:34px;border-radius:50%;background:#dce8f2;border:1px solid #8ec9ef;box-shadow:0 0 18px rgba(111,191,242,.42);transform:translate(-50%,-50%)}
.reign-buttons{display:grid;grid-template-columns:1fr;gap:6px}
.reign-buttons button{min-height:34px;border:1px solid #3f607b;background:#1a2b38;color:#dce8f2;border-radius:5px;cursor:pointer}
.reign-buttons button:hover,.reign-buttons button:focus-visible{border-color:#8ec9ef;color:#fff;outline:none}
.reign-buttons .stop{background:#5b1720;border-color:#9a3443;color:#fff}
.reign-readout{font-size:11px;color:#b7c8d8}
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
.map-toggles{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:3px 8px;margin-top:5px;font-size:10px;color:#c8d4df}
.map-toggles label{display:flex;align-items:center;gap:4px;min-width:0}
.map-toggles input{accent-color:#52a9ff}
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
#experience-forge{left:12px;top:470px;width:340px;height:300px;border-color:#3c4854;background:rgba(8,11,15,.88)}
#experience-forge .panel-content{display:grid;grid-template-rows:auto auto minmax(0,1fr);gap:8px;overflow:hidden}
#experience-forge header{display:flex;align-items:center;justify-content:space-between;gap:12px}
#experience-forge h2{font-size:12px;margin:0;color:#c6e6ff}
#forge-status{color:#ffcf7a;font-size:11px;font-variant-numeric:tabular-nums}
#forge-vector{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:3px}
.forge-slot{height:18px;border:1px solid #27313b;border-radius:3px;background:#111820;position:relative;overflow:hidden}
.forge-slot span{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);font-size:9px;color:#e7eef5;font-variant-numeric:tabular-nums}
.forge-fill{position:absolute;left:0;top:0;bottom:0;background:#5aa9e6;opacity:.72}
#forge-filters{display:grid;gap:5px;min-height:0;overflow:auto;scrollbar-width:thin}
.forge-filter{display:grid;gap:3px;padding:6px;border:1px solid #27313b;background:rgba(18,24,30,.72);border-radius:5px;font-size:11px}
.forge-filter-top{display:flex;justify-content:space-between;gap:8px;color:#fff;font-weight:700;font-variant-numeric:tabular-nums}
.forge-filter-line{color:#aeb9c3;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.forge-event{color:#ffd083}
#xr{position:fixed;right:12px;bottom:12px;padding:9px 12px;border:1px solid #405060;background:#15202b;color:#fff;border-radius:6px}
#xr[disabled]{opacity:.55}
#fallback{position:fixed;left:12px;bottom:12px;color:#aab4bd;max-width:min(520px,calc(100vw - 24px))}
@media(max-width:820px){.panel-window{max-width:calc(100vw - 24px)}#llm{left:12px;right:12px;top:auto;bottom:56px;width:auto;max-height:42vh}#reign{top:auto;bottom:12px;right:12px;min-width:0}.llm-text{max-height:76px}#models{bottom:98px;max-height:46vh}#model-graph-window{display:none}#virtual-pipeline-section{left:12px;right:12px;top:128px;width:auto;max-height:28vh}#model-graph{height:220px}#learning{left:12px;right:12px;top:220px;width:auto;max-height:34vh}#experience-forge{left:12px;right:12px;top:380px;width:auto;max-height:34vh}#calibration{display:none}}
canvas{display:block}
</style>
<canvas id="scene"></canvas>
<aside id="hud" data-window-title="Instant 3D">
  <h1>Instant 3D</h1>
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
    <dt>depth</dt><dd id="depth_stats">-</dd>
    <dt>surfaces</dt><dd id="surfaces">-</dd>
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
<aside id="experience-forge" data-window-title="Experience Forge">
  <header>
    <h2>Experience Forge</h2>
    <div id="forge-status">waiting...</div>
  </header>
  <div id="forge-vector"></div>
  <div id="forge-filters"></div>
</aside>
<aside id="reign" data-window-title="Reign controls">
  <strong>Reign controls</strong>
  <div class="reign-pad">
    <div id="reign-joystick" role="application" aria-label="Hold and drag to steer">
      <div id="reign-stick"></div>
    </div>
    <div class="reign-buttons">
      <button class="stop" id="reign-stop" type="button">Stop</button>
      <div class="reign-readout" id="reign-readout">F 0.00 / T 0.00</div>
    </div>
  </div>
  <div id="reign-state">web reign ready</div>
  <div class="reign-hint">Drag the stick for analog forward/reverse and turn together. Release to stop.</div>
  <div class="cockpit">
    <label class="cockpit-arm"><input id="hardware-arm" type="checkbox"> Real hardware armed</label>
    <div id="hardware-state">hardware cockpit unavailable</div>
    <div class="trace-label">Real hardware uses the same analog stick, with cautious speed limits.</div>
    <canvas id="trace-map" width="220" height="180" aria-label="odometry/range trace map"></canvas>
    <div class="map-toggles" aria-label="map overlays">
      <label><input type="checkbox" data-map-layer="occupancy" checked>occupancy</label>
      <label><input type="checkbox" data-map-layer="rays" checked>rays</label>
      <label><input type="checkbox" data-map-layer="raw point cloud">raw cloud</label>
      <label><input type="checkbox" data-map-layer="accumulated occupancy" checked>accum</label>
      <label><input type="checkbox" data-map-layer="stable wall candidates" checked>stable</label>
      <label><input type="checkbox" data-map-layer="danger" checked>danger</label>
      <label><input type="checkbox" data-map-layer="charger/charge" checked>charge</label>
      <label><input type="checkbox" data-map-layer="social" checked>social</label>
      <label><input type="checkbox" data-map-layer="novelty">novelty</label>
      <label><input type="checkbox" data-map-layer="events" checked>events</label>
    </div>
    <div class="trace-label">SLAM-lite / odometry map: odometry/range trace, not corrected SLAM.</div>
    <canvas id="entity-graph" width="220" height="170" aria-label="entity constellation graph"></canvas>
    <div id="entity-graph-summary">sensations/experience/impressions waiting...</div>
  </div>
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
      <label for="cal-depth-z" style="width:90px">Depth Forward</label>
      <input type="range" id="cal-depth-z" min="-0.50" max="0.50" step="0.01" value="0.00">
      <span id="val-depth-z" style="width:40px;text-align:right;font-family:monospace">0.00m</span>
    </div>
    <div style="display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center">
      <label for="cal-depth-pitch" style="width:90px">Depth Pitch</label>
      <input type="range" id="cal-depth-pitch" min="-30" max="30" step="1" value="0">
      <span id="val-depth-pitch" style="width:40px;text-align:right;font-family:monospace">0°</span>
    </div>
    <div style="display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center">
      <label for="cal-depth-roll" style="width:90px">Depth Roll</label>
      <input type="range" id="cal-depth-roll" min="-30" max="30" step="1" value="0">
      <span id="val-depth-roll" style="width:40px;text-align:right;font-family:monospace">0°</span>
    </div>
    <div style="display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center">
      <label for="cal-depth-yaw" style="width:90px">Depth Yaw</label>
      <input type="range" id="cal-depth-yaw" min="-45" max="45" step="1" value="0">
      <span id="val-depth-yaw" style="width:40px;text-align:right;font-family:monospace">0°</span>
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
<div id="fallback">Desktop drag rotates, wheel zooms, right-drag pans. In VR, thumbstick steers and release stops.</div>
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
const reignReadout = document.getElementById('reign-readout');
const hardwareArm = document.getElementById('hardware-arm');
const hardwareState = document.getElementById('hardware-state');
const traceCanvas = document.getElementById('trace-map');
const traceCtx = traceCanvas.getContext('2d');
const entityGraphCanvas = document.getElementById('entity-graph');
const entityGraphCtx = entityGraphCanvas.getContext('2d');
const entityGraphSummary = document.getElementById('entity-graph-summary');
const mapLayerInputs = Array.from(document.querySelectorAll('[data-map-layer]'));
const llmStatus = document.getElementById('llm-status');
const llmStreams = document.getElementById('llm-streams');
const modelStatus = document.getElementById('model-status');
const modelList = document.getElementById('model-list');
const modelGraph = document.getElementById('model-graph');
const behaviorInspector = document.getElementById('behavior-inspector');
const virtualPipelineSection = document.getElementById('virtual-pipeline-section');
const virtualReportSummary = document.getElementById('virtual-report-summary');
const virtualModelRecommendations = document.getElementById('virtual-model-recommendations');
const fields = Object.fromEntries(['mode','scenario','training_mode','weights_updating','ledger_counts','action_selector_mode','chosen_action','motor_line','motion_sent','movement_delta','blocked_reason','seed','tick','t','pose','battery','stuck','trap_kind','dead_battery','recovery_mode','recovery_attempts','stuck_ticks','nearest','eye','points','depth_stats','surfaces','audio','mind','scheme','secure','webxr'].map(id => [id, document.getElementById(id)]));
const learningMode = document.getElementById('learning-mode');
const learningSteps = document.getElementById('learning-steps');
const learningStatus = document.getElementById('learning-status');
const learningApply = document.getElementById('learning-apply');
const learningChecks = Array.from(document.querySelectorAll('#learning-behaviors input[type="checkbox"]'));
const forgeStatus = document.getElementById('forge-status');
const forgeVector = document.getElementById('forge-vector');
const forgeFilters = document.getElementById('forge-filters');
const traceState = {poses: [], events: [], occupied: new Map(), free: new Map()};
const entityGraphState = {positions: new Map(), selectedNodeId: null, lastGraph: null};
let cockpitAvailable = false;
let cockpitArmed = false;

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
eyePanel.scaling.x = -1;
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
const surfaceOverlays = new BABYLON.TransformNode("surfaceOverlays", scene);

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
function clamp(value, min, max){ return Math.max(min, Math.min(max, value)); }
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
    if(child.material){
      const diffuseTexture = child.material.diffuseTexture;
      const emissiveTexture = child.material.emissiveTexture;
      if(diffuseTexture) diffuseTexture.dispose();
      if(emissiveTexture && emissiveTexture !== diffuseTexture) emissiveTexture.dispose();
      child.material.dispose();
    }
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
  const isKinectCamera = coordinateSystem === 'kinect_camera' || coordinateSystem === 'camera' || coordinateSystem === 'depth_image_unknown';
  const robotMatrix = robot.getWorldMatrix();
  const kinectMatrix = eyeCamera.getWorldMatrix();
  
  points.forEach((p, i) => {
    let worldPoint;
    if (isRobot) {
      worldPoint = BABYLON.Vector3.TransformCoordinates(
        new BABYLON.Vector3(p.x, p.y, -p.z),
        robotMatrix
      );
    } else if (isKinectCamera) {
      worldPoint = BABYLON.Vector3.TransformCoordinates(
        new BABYLON.Vector3(p.x, -p.y, p.z),
        kinectMatrix
      );
    } else {
      worldPoint = new BABYLON.Vector3(p.x, p.y, p.z);
    }
    positions.push(worldPoint.x, worldPoint.y, worldPoint.z);
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
  
  pointCloud.parent = null;
}

function renderAccumulatedPoints(points){
  renderPoints(points || [], 'world');
}

function renderWorldBeliefPoints(packet){
  const layers = enabledMapLayers();
  const accumulated = packet.kinect?.accumulated_points || [];
  const raw = packet.kinect?.points || [];
  if(layers.has('stable wall candidates') && accumulated.some(point => point.stable)){
    renderAccumulatedPoints(accumulated.filter(point => point.stable));
  }else if(layers.has('accumulated occupancy') && accumulated.length){
    renderAccumulatedPoints(accumulated);
  }else if(layers.has('raw point cloud')){
    renderPoints(raw, packet.kinect?.coordinate_system || 'camera');
  }else{
    renderPoints([], 'world');
  }
}

function v3(value){
  return {
    x: Number(value?.x) || 0,
    y: Number(value?.y) || 0,
    z: Number(value?.z) || 0
  };
}

function add3(a, b){ return {x:a.x + b.x, y:a.y + b.y, z:a.z + b.z}; }
function sub3(a, b){ return {x:a.x - b.x, y:a.y - b.y, z:a.z - b.z}; }
function mul3(a, s){ return {x:a.x * s, y:a.y * s, z:a.z * s}; }
function dot3(a, b){ return a.x * b.x + a.y * b.y + a.z * b.z; }
function cross3(a, b){
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x
  };
}
function norm3(a){
  const len = Math.sqrt(dot3(a, a));
  return len > 1e-6 ? mul3(a, 1 / len) : {x:1, y:0, z:0};
}
function surfacePoint(value){
  const p = v3(value);
  return world(p.x, p.y, p.z);
}
function localSurfaceToWorld(surface, pose){
  const heading = Number(pose?.heading_rad) || 0;
  const sin = Math.sin(heading), cos = Math.cos(heading);
  const centroid = v3(surface.centroid);
  const normal = v3(surface.normal);
  const worldCentroid = {
    x: (Number(pose?.x_m) || 0) + centroid.x * cos - centroid.y * sin,
    y: (Number(pose?.y_m) || 0) + centroid.x * sin + centroid.y * cos,
    z: centroid.z
  };
  const worldNormal = {
    x: normal.x * cos - normal.y * sin,
    y: normal.x * sin + normal.y * cos,
    z: normal.z
  };
  return {...surface, centroid: worldCentroid, normal: worldNormal};
}
function surfaceColor(kind, confidence=1){
  const alpha = Math.max(.18, Math.min(.56, .16 + confidence * .34));
  if(kind === 'floor') return {color:new BABYLON.Color3(0.32, 0.82, 0.58), alpha};
  if(kind === 'horizontal_plane') return {color:new BABYLON.Color3(0.36, 0.66, 1.0), alpha};
  if(kind === 'vertical_plane') return {color:new BABYLON.Color3(1.0, 0.72, 0.36), alpha};
  return {color:new BABYLON.Color3(0.84, 0.68, 1.0), alpha};
}
function surfaceCandidateLabel(surface){
  const labels = Array.isArray(surface.labels) ? surface.labels : [];
  if(labels.length) return labels[0];
  if(surface.kind === 'floor') return 'floor_candidate';
  if(surface.kind === 'horizontal_plane') return 'table_candidate';
  if(surface.kind === 'vertical_plane') return 'wall_candidate';
  return 'unknown_surface';
}
function planeBasis(normal){
  const n = norm3(normal);
  const anchor = Math.abs(n.z) < .9 ? {x:0, y:0, z:1} : {x:1, y:0, z:0};
  const basisA = norm3(cross3(n, anchor));
  const basisB = norm3(cross3(n, basisA));
  return {basisA, basisB};
}
function planeCorners(surface, boundsName='bounds_2d'){
  const bounds = surface[boundsName] || surface.bounds_2d || {};
  const minU = Number.isFinite(bounds.min_u) ? bounds.min_u : -.25;
  const maxU = Number.isFinite(bounds.max_u) ? bounds.max_u : .25;
  const minV = Number.isFinite(bounds.min_v) ? bounds.min_v : -.25;
  const maxV = Number.isFinite(bounds.max_v) ? bounds.max_v : .25;
  const centroid = v3(surface.centroid);
  const {basisA, basisB} = planeBasis(v3(surface.normal));
  return [
    add3(add3(centroid, mul3(basisA, minU)), mul3(basisB, minV)),
    add3(add3(centroid, mul3(basisA, maxU)), mul3(basisB, minV)),
    add3(add3(centroid, mul3(basisA, maxU)), mul3(basisB, maxV)),
    add3(add3(centroid, mul3(basisA, minU)), mul3(basisB, maxV))
  ];
}
function createPlanePatch(surface, options={}){
  const corners = planeCorners(surface, options.boundsName || 'bounds_2d').map(surfacePoint);
  const mesh = new BABYLON.Mesh(`surface_${surface.id || 'plane'}`, scene);
  const vertexData = new BABYLON.VertexData();
  vertexData.positions = corners.flatMap(p => [p.x, p.y, p.z]);
  vertexData.indices = [0, 1, 2, 0, 2, 3];
  vertexData.applyToMesh(mesh);
  const style = surfaceColor(surface.kind, surface.confidence);
  const mat = new BABYLON.StandardMaterial(`surfaceMat_${surface.id || 'plane'}`, scene);
  mat.diffuseColor = style.color;
  mat.emissiveColor = style.color.scale(.45);
  mat.alpha = options.alpha ?? style.alpha;
  mat.backFaceCulling = false;
  mat.wireframe = !!options.wireframe;
  mesh.material = mat;
  mesh.parent = surfaceOverlays;
  const outline = BABYLON.MeshBuilder.CreateLines(`surfaceOutline_${surface.id || 'plane'}`, {
    points: [corners[0], corners[1], corners[2], corners[3], corners[0]]
  }, scene);
  outline.color = options.outlineColor || style.color;
  outline.alpha = options.outlineAlpha ?? .9;
  outline.parent = surfaceOverlays;
  return mesh;
}
function createLabel(text, position, color){
  const width = 256, height = 64;
  const texture = new BABYLON.DynamicTexture(`label_${text}`, {width, height}, scene, true);
  const ctx = texture.getContext();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = 'rgba(5, 8, 11, .72)';
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = `rgb(${Math.round(color.r * 255)},${Math.round(color.g * 255)},${Math.round(color.b * 255)})`;
  ctx.lineWidth = 4;
  ctx.strokeRect(2, 2, width - 4, height - 4);
  ctx.fillStyle = '#eef6ff';
  ctx.font = 'bold 26px system-ui, sans-serif';
  ctx.textBaseline = 'middle';
  ctx.fillText(text.slice(0, 18), 14, height / 2);
  texture.update();
  const plane = BABYLON.MeshBuilder.CreatePlane(`label_${text}`, {width:.72, height:.18}, scene);
  const mat = new BABYLON.StandardMaterial(`labelMat_${text}`, scene);
  mat.diffuseTexture = texture;
  mat.emissiveTexture = texture;
  mat.disableLighting = true;
  mat.backFaceCulling = false;
  plane.material = mat;
  plane.position.copyFrom(position);
  plane.billboardMode = BABYLON.Mesh.BILLBOARDMODE_ALL;
  plane.parent = surfaceOverlays;
}
function createClusterBox(cluster){
  const size = v3(cluster.size_m);
  const centroid = surfacePoint(cluster.centroid);
  const mesh = BABYLON.MeshBuilder.CreateBox(`cluster_${cluster.id || 'cluster'}`, {
    width: Math.max(size.x, .04),
    height: Math.max(size.z, .04),
    depth: Math.max(size.y, .04)
  }, scene);
  const mat = new BABYLON.StandardMaterial(`clusterMat_${cluster.id || 'cluster'}`, scene);
  const color = cluster.moving ? new BABYLON.Color3(1.0, 0.92, 0.32) : new BABYLON.Color3(1.0, 0.35, 0.48);
  mat.diffuseColor = color;
  mat.emissiveColor = color.scale(.55);
  mat.alpha = cluster.moving ? .48 : .32;
  mat.wireframe = true;
  mesh.material = mat;
  mesh.position.copyFrom(centroid);
  mesh.parent = surfaceOverlays;
  const hint = cluster.semantic_hint || 'transient_cluster';
  const motion = cluster.moving ? ' moving' : '';
  createLabel(`${hint}${motion} ${fmt(cluster.confidence || 0, 2)}`, centroid.add(new BABYLON.Vector3(0, Math.max(size.z, .18) + .14, 0)), mat.diffuseColor);
}
function createBeliefBox(item, kind){
  const size = v3(item.size_m);
  const centroid = surfacePoint(item.centroid);
  const mesh = BABYLON.MeshBuilder.CreateBox(`belief_${item.id || kind}`, {
    width: Math.max(size.x, .06),
    height: Math.max(size.z, .06),
    depth: Math.max(size.y, .06)
  }, scene);
  const mat = new BABYLON.StandardMaterial(`beliefMat_${item.id || kind}`, scene);
  const isSurface = kind === 'surface';
  const color = isSurface ? new BABYLON.Color3(0.49, 0.90, 0.68) : new BABYLON.Color3(0.39, 0.78, 1.0);
  mat.diffuseColor = color;
  mat.emissiveColor = color.scale(.42);
  mat.alpha = isSurface ? .26 : .18;
  mat.wireframe = true;
  mesh.material = mat;
  mesh.position.copyFrom(centroid);
  mesh.parent = surfaceOverlays;
  const label = isSurface ? (item.kind || 'stable_surface') : 'stable_blob';
  createLabel(`${label} ${fmt(item.confidence || 0, 2)}`, centroid.add(new BABYLON.Vector3(0, Math.max(size.z, .16) + .22, 0)), color);
}
function renderPersistentWorldBelief(packet, layers){
  if(!layers.has('stable wall candidates')) return;
  const belief = packet.kinect?.local_world_belief;
  if(!belief) return;
  for(const surface of belief.stable_surfaces || []) createBeliefBox(surface, 'surface');
  for(const blob of belief.stable_blobs || []) createBeliefBox(blob, 'blob');
}
function renderSurfacePerception(packet){
  clearChildren(surfaceOverlays);
  const perception = packet.surface_perception;
  const layers = enabledMapLayers();
  const showStableSurfaces = layers.has('stable wall candidates');
  renderPersistentWorldBelief(packet, layers);
  if(!perception) return;
  if(showStableSurfaces){
    for(const surface of perception.stable_surfaces || []){
      createPlanePatch(surface);
      const style = surfaceColor(surface.kind, surface.confidence);
      const labelPosition = surfacePoint(surface.centroid).add(new BABYLON.Vector3(0, .12, 0));
      createLabel(`${surfaceCandidateLabel(surface)} ${fmt(surface.confidence, 2)}`, labelPosition, style.color);
    }
  }
  for(const cluster of perception.clusters || []){
    createClusterBox(cluster);
  }
  const frames = perception.scene_graph?.navigation?.anticipation?.frames || perception.scene_graph?.anticipation?.frames || [];
  const future = frames[frames.length - 1];
  if(showStableSurfaces && future){
    const pose = packet.body || {};
    for(const projected of future.projected_surfaces || []){
      const surface = localSurfaceToWorld(projected, pose);
      createPlanePatch(surface, {
        boundsName: 'extrapolated_bounds_2d',
        alpha: .08,
        wireframe: true,
        outlineAlpha: .45,
        outlineColor: new BABYLON.Color3(1.0, 0.88, 0.28)
      });
    }
  }
}

function shouldUseGeneratedEye(packet){
  const session = packet?.session || {};
  const mode = String(session.mode || '').toLowerCase();
  const virtualMode =
    session.source === 'sim' ||
    mode.includes('virtual') ||
    mode.includes('read-only') ||
    mode.includes('readonly');
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
  const isUyvy = frameFormatIs(fmt, 'Uyvy422');
  const isMjpg = frameFormatIs(fmt, 'Mjpeg') || frameFormatText(fmt).includes('MJPG');
  if(isRgb || isBgr || isGray || isYuyv || isUyvy){
    eyeCanvas.width = frame.width; eyeCanvas.height = frame.height;
    const image = eyeCanvas.getContext('2d').createImageData(frame.width, frame.height);
    if(isGray){
      for(let source = 0, target = 0; target < image.data.length && source < frame.bytes.length; source += 1, target += 4){
        const value = frame.bytes[source];
        image.data[target] = value; image.data[target + 1] = value; image.data[target + 2] = value; image.data[target + 3] = 255;
      }
    }else if(isYuyv || isUyvy){
      for(let source = 0, target = 0; target + 7 < image.data.length && source + 3 < frame.bytes.length; source += 4, target += 8){
        const y0 = isUyvy ? frame.bytes[source + 1] : frame.bytes[source];
        const u = isUyvy ? frame.bytes[source] : frame.bytes[source + 1];
        const y1 = isUyvy ? frame.bytes[source + 3] : frame.bytes[source + 2];
        const v = isUyvy ? frame.bytes[source + 2] : frame.bytes[source + 3];
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

function renderExperienceForge(forge){
  if(!forge){
    forgeStatus.textContent = 'waiting...';
    forgeVector.replaceChildren();
    forgeFilters.replaceChildren();
    return;
  }
  forgeStatus.textContent = `tick ${forge.ticks || 0} | pop ${forge.population_size || 0} | buf ${forge.buffer_len || 0}`;
  const vector = forge.tiny_now_vector || [];
  forgeVector.innerHTML = vector.map((value, index) => {
    const clamped = Math.max(-1, Math.min(1, Number(value) || 0));
    const width = Math.round(Math.abs(clamped) * 50);
    const left = clamped < 0 ? 50 - width : 50;
    const color = clamped < 0 ? '#f06d6d' : '#5aa9e6';
    return `<div class="forge-slot" title="slot ${index}: ${fmt(clamped, 2)}"><div class="forge-fill" style="left:${left}%;width:${width}%;background:${color}"></div><span>${fmt(clamped, 2)}</span></div>`;
  }).join('');
  forgeFilters.innerHTML = (forge.top_filters || []).slice(0, 8).map(filter => {
    const channels = (filter.channels || []).join(', ') || '-';
    const event = (filter.fired_events || []).slice(-1)[0];
    const labels = event?.labels || {};
    const eventNames = ['bump','stuck','blocked_forward','intervention','action_changed_scene'].filter(name => labels[name]).join(' ');
    return `<div class="forge-filter">
      <div class="forge-filter-top"><span>#${filter.slot ?? '-'} f${filter.id}</span><span>${fmt(filter.score || 0, 3)}</span></div>
      <div class="forge-filter-line">${escapeHtml(channels)}</div>
      <div class="forge-filter-line">out ${fmt(filter.output || 0, 2)} age ${filter.age_ticks || 0}${eventNames ? ` <span class="forge-event">${escapeHtml(eventNames)}</span>` : ''}</div>
    </div>`;
  }).join('') || '<div class="forge-filter-line">no filters yet</div>';
}

function updateScene(packet, liveMap=null){
  lastScene = packet;
  robot.position.copyFrom(world(packet.body.x_m, packet.body.y_m, 0));
  robot.rotation.y = -packet.body.heading_rad - Math.PI / 2;
  renderObjects(packet);
  renderBeams(packet);
  renderWorldBeliefPoints(packet);
  renderSurfacePerception(packet);
  renderEye(packet);
  updateHardwareControl(packet.hardware_control);
  renderExperienceForge(packet.experience_forge);
  drawTraceMap(packet, liveMap);
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
  const depth = packet.kinect?.diagnostics || {};
  const cloudSummary = packet.kinect?.accumulated_summary;
  const accumulated = packet.kinect?.accumulated_points || [];
  fields.points.textContent = cloudSummary
    ? `${accumulated.length} drawn / ${cloudSummary.voxels || 0} voxels / ${cloudSummary.stable_voxels || 0} stable`
    : `${packet.kinect?.points?.length || 0} drawn / ${depth.valid_depth_count || 0} valid`;
  fields.depth_stats.textContent = depth.depth_width
    ? `${depth.depth_width}x${depth.depth_height} ${depth.coordinate_system || '-'} med ${depth.median_depth_m == null ? '-' : fmt(depth.median_depth_m)}m floor min ${depth.min_z_m == null ? '-' : fmt(depth.min_z_m)} med ${depth.median_z_m == null ? '-' : fmt(depth.median_z_m)} below ${depth.below_floor_count || 0}/${fmt(depth.below_floor_ratio || 0, 2)} skip ${depth.skipped_depth_count || 0} clip ${depth.clipped_depth_count || 0} stride ${depth.sample_stride || '-'}`
    : '-';
  const surface = packet.surface_perception;
  if(surface){
    const nav = surface.scene_graph?.navigation || {};
    const movingClusters = (surface.clusters || []).filter(cluster => cluster.moving).length;
    const hint = surface.diagnostics?.calibration_hint;
    const belief = packet.kinect?.local_world_belief;
    const orient = belief?.orientation_status;
    const beliefText = belief ? ` | belief ${belief.stable_surfaces?.length || 0}s/${belief.stable_blobs?.length || 0}b ${orient?.roll_pitch_corrected ? 'imu rp' : 'odom planar'}` : '';
    const calText = hint ? ` | floor z ${fmt(hint.floor_height_error_m, 2)}m tilt ${fmt(hint.floor_tilt_rad * 180 / Math.PI, 1)}°` : '';
    fields.surfaces.textContent = `${surface.stable_surfaces?.length || 0} stable / ${surface.plane_observations?.length || 0} planes / ${surface.clusters?.length || 0} clusters (${movingClusters} moving) | front ${nav.front_clear_m == null ? '-' : fmt(nav.front_clear_m)}m${calText}${beliefText}`;
  }else{
    const belief = packet.kinect?.local_world_belief;
    fields.surfaces.textContent = belief ? `belief ${belief.stable_surfaces?.length || 0} surfaces / ${belief.stable_blobs?.length || 0} blobs | ${belief.orientation_status?.note || '-'}` : '-';
  }
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

function updateHardwareControl(hw){
  cockpitAvailable = !!hw?.available;
  cockpitArmed = !!hw?.armed;
  hardwareArm.disabled = !cockpitAvailable;
  hardwareArm.checked = cockpitArmed;
  const stateText = !cockpitAvailable ? 'unavailable'
    : cockpitArmed ? 'ARMED'
    : 'disarmed';
  const sessionText = hw?.source && hw?.mode ? ` ${hw.source}/${hw.mode}` : '';
  const detail = hw?.reason ? `: ${hw.reason}` : '';
  const age = hw?.body_age_ms == null ? '' : ` body ${hw.body_age_ms}ms`;
  hardwareState.textContent = `${stateText}${sessionText}${detail}${age}`;
  hardwareState.style.color = cockpitArmed ? '#8df0b2' : '#ffbf7a';
}

async function setHardwareArmed(armed){
  try{
    const res = await fetch('/reign/hardware-arm', {
      method:'POST',
      headers:{'content-type':'application/json'},
      body:JSON.stringify({armed})
    });
    if(!res.ok) throw new Error(await res.text());
    updateHardwareControl(await res.json());
    if(!armed) postWebReign({type:'Stop'}, 900, 'hardware disarmed', 1);
  }catch(error){
    hardwareState.textContent = 'arm request rejected';
    hardwareArm.checked = false;
    cockpitArmed = false;
  }
}

function disarmHardwareOnExit(){
  if(navigator.sendBeacon){
    const body = new Blob([JSON.stringify({armed:false})], {type:'application/json'});
    navigator.sendBeacon('/reign/hardware-arm', body);
  } else {
    postWebReign({type:'Stop'}, 900, 'hardware stop', 1);
  }
}

function projectRangeBeam(pose, beam){
  const angle = pose.heading_rad + beam.angle_rad;
  return {
    x: pose.x_m + Math.cos(angle) * beam.distance_m,
    y: pose.y_m + Math.sin(angle) * beam.distance_m,
    hit: !!beam.hit
  };
}

function cellKey(point, size=.12){
  return `${Math.round(point.x / size)},${Math.round(point.y / size)}`;
}

function rememberCell(map, point, value){
  const key = cellKey(point);
  map.set(key, {x: point.x, y: point.y, value});
  if(map.size > 520) map.delete(map.keys().next().value);
}

function robotLocalToWorld(pose, localX, localY){
  const heading = Number(pose.heading_rad) || 0;
  const cos = Math.cos(heading);
  const sin = Math.sin(heading);
  return {
    x: pose.x_m + localX * cos - localY * sin,
    y: pose.y_m + localX * sin + localY * cos
  };
}

function updateTraceState(packet){
  const pose = packet.body;
  traceState.poses.push({x_m: pose.x_m, y_m: pose.y_m, heading_rad: pose.heading_rad});
  if(traceState.poses.length > 900) traceState.poses.shift();
  for(const beam of packet.range?.beams || []){
    const point = projectRangeBeam(pose, beam);
    if(beam.hit) rememberCell(traceState.occupied, point, 1);
    else rememberCell(traceState.free, point, .35);
  }
  if(pose.bump_left || pose.bump_right || pose.cliff || pose.wheel_drop || packet.stuck || packet.action?.safety_override){
    traceState.events.push({
      x: pose.x_m,
      y: pose.y_m,
      kind: pose.cliff || pose.wheel_drop ? 'safety' : packet.stuck ? 'stuck' : 'bump'
    });
    if(traceState.events.length > 160) traceState.events.shift();
  }
}

function enabledMapLayers(){
  return new Set(mapLayerInputs.filter(input => input.checked).map(input => input.dataset.mapLayer));
}

function hashUnit(value){
  let hash = 2166136261;
  for(let i = 0; i < value.length; i++){
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 4294967295;
}

function graphNodeColor(type){
  return {
    observation:'#8bd3ff',
    cluster:'#ffcf66',
    entity:'#60d394',
    text_label:'#f3a6ff',
    place:'#dce8f2',
    action_context:'#ff8b4a'
  }[type] || '#aeb9c3';
}

function graphEdgeColor(type){
  return {
    belongs_to:'rgba(139,211,255,.55)',
    co_occurs_with:'rgba(255,207,102,.48)',
    same_place_as:'rgba(220,232,242,.42)',
    same_time_as:'rgba(168,137,255,.44)',
    moves_with:'rgba(96,211,148,.44)',
    predicts:'rgba(255,139,74,.46)',
    named_by:'rgba(243,166,255,.52)',
    part_of_entity:'rgba(96,211,148,.58)'
  }[type] || 'rgba(174,185,195,.38)';
}

function ensureGraphPosition(node, width, height){
  let pos = entityGraphState.positions.get(node.id);
  if(!pos){
    const a = hashUnit(`${node.id}:a`) * Math.PI * 2;
    const r = 28 + hashUnit(`${node.id}:r`) * Math.min(width, height) * .28;
    pos = {
      x: width / 2 + Math.cos(a) * r,
      y: height / 2 + Math.sin(a) * r,
      vx: 0,
      vy: 0
    };
    entityGraphState.positions.set(node.id, pos);
  }
  return pos;
}

function updateEntityGraphLayout(graph, width, height){
  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];
  const ids = new Set(nodes.map(node => node.id));
  for(const id of Array.from(entityGraphState.positions.keys())){
    if(!ids.has(id)) entityGraphState.positions.delete(id);
  }
  const byId = new Map(nodes.map(node => [node.id, node]));
  for(const node of nodes) ensureGraphPosition(node, width, height);
  for(let step = 0; step < 18; step++){
    for(let i = 0; i < nodes.length; i++){
      const a = nodes[i];
      const pa = entityGraphState.positions.get(a.id);
      for(let j = i + 1; j < nodes.length; j++){
        const b = nodes[j];
        const pb = entityGraphState.positions.get(b.id);
        const dx = pa.x - pb.x;
        const dy = pa.y - pb.y;
        const d2 = Math.max(36, dx * dx + dy * dy);
        const force = 48 / d2;
        pa.vx += dx * force;
        pa.vy += dy * force;
        pb.vx -= dx * force;
        pb.vy -= dy * force;
      }
    }
    for(const edge of edges){
      if(!byId.has(edge.from) || !byId.has(edge.to)) continue;
      const a = entityGraphState.positions.get(edge.from);
      const b = entityGraphState.positions.get(edge.to);
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const distance = Math.max(1, Math.hypot(dx, dy));
      const target = edge.edge_type === 'part_of_entity' || edge.edge_type === 'named_by' ? 34 : 56;
      const force = (distance - target) * .004 * (edge.confidence || .5);
      const fx = dx / distance * force;
      const fy = dy / distance * force;
      a.vx += fx;
      a.vy += fy;
      b.vx -= fx;
      b.vy -= fy;
    }
    for(const node of nodes){
      const p = entityGraphState.positions.get(node.id);
      p.vx += (width / 2 - p.x) * .002;
      p.vy += ((height - 26) / 2 - p.y) * .002;
      p.x = clamp(p.x + p.vx, 10, width - 10);
      p.y = clamp(p.y + p.vy, 10, height - 30);
      p.vx *= .64;
      p.vy *= .64;
    }
  }
}

function nearestEntityGraphNode(clientX, clientY){
  const rect = entityGraphCanvas.getBoundingClientRect();
  const x = (clientX - rect.left) * (entityGraphCanvas.width / Math.max(1, rect.width));
  const y = (clientY - rect.top) * (entityGraphCanvas.height / Math.max(1, rect.height));
  let best = null;
  let bestDistance = 16;
  for(const node of entityGraphState.lastGraph?.nodes || []){
    const pos = entityGraphState.positions.get(node.id);
    if(!pos) continue;
    const distance = Math.hypot(pos.x - x, pos.y - y);
    if(distance < bestDistance){
      bestDistance = distance;
      best = node;
    }
  }
  return best;
}

function drawEntityGraph(graph){
  const width = entityGraphCanvas.width;
  const height = entityGraphCanvas.height;
  entityGraphCtx.clearRect(0, 0, width, height);
  entityGraphCtx.fillStyle = '#070b0f';
  entityGraphCtx.fillRect(0, 0, width, height);
  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];
  const events = graph?.events || [];
  entityGraphState.lastGraph = graph;
  if(entityGraphState.selectedNodeId && !nodes.some(node => node.id === entityGraphState.selectedNodeId)){
    entityGraphState.selectedNodeId = null;
  }
  if(!nodes.length){
    entityGraphSummary.textContent = 'constellation waiting...';
    return;
  }
  updateEntityGraphLayout(graph, width, height);
  const byId = new Map(nodes.map(node => [node.id, node]));
  entityGraphCtx.lineWidth = 1;
  for(const edge of edges){
    if(!byId.has(edge.from) || !byId.has(edge.to)) continue;
    const a = entityGraphState.positions.get(edge.from);
    const b = entityGraphState.positions.get(edge.to);
    entityGraphCtx.strokeStyle = graphEdgeColor(edge.edge_type);
    entityGraphCtx.globalAlpha = clamp(.18 + (edge.confidence || .5) * .72, .18, .9);
    entityGraphCtx.beginPath();
    entityGraphCtx.moveTo(a.x, a.y);
    entityGraphCtx.lineTo(b.x, b.y);
    entityGraphCtx.stroke();
  }
  entityGraphCtx.globalAlpha = 1;
  for(const node of nodes){
    const p = entityGraphState.positions.get(node.id);
    const radius = node.node_type === 'entity' ? 6 : node.node_type === 'cluster' ? 5 : 4;
    const alpha = clamp(.35 + (node.confidence || .5) * .65, .35, 1);
    entityGraphCtx.fillStyle = graphNodeColor(node.node_type);
    entityGraphCtx.globalAlpha = alpha;
    entityGraphCtx.beginPath();
    if(node.node_type === 'text_label'){
      entityGraphCtx.rect(p.x - radius, p.y - radius, radius * 2, radius * 2);
    }else{
      entityGraphCtx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    }
    entityGraphCtx.fill();
    if(node.age_ms > 1200){
      entityGraphCtx.strokeStyle = 'rgba(255,255,255,.38)';
      entityGraphCtx.setLineDash([2, 2]);
      entityGraphCtx.stroke();
      entityGraphCtx.setLineDash([]);
    }
    if(entityGraphState.selectedNodeId === node.id){
      entityGraphCtx.lineWidth = 1.5;
      entityGraphCtx.strokeStyle = '#ffcf66';
      entityGraphCtx.beginPath();
      entityGraphCtx.arc(p.x, p.y, radius + 2, 0, Math.PI * 2);
      entityGraphCtx.stroke();
      entityGraphCtx.lineWidth = 1;
    }
  }
  entityGraphCtx.globalAlpha = 1;
  entityGraphCtx.font = '10px system-ui';
  entityGraphCtx.textBaseline = 'top';
  for(const node of nodes.filter(node => node.node_type === 'entity' || node.node_type === 'place' || node.node_type === 'action_context').slice(0, 6)){
    const p = entityGraphState.positions.get(node.id);
    const text = String(node.label || node.id).slice(0, 22);
    entityGraphCtx.fillStyle = 'rgba(7,11,15,.72)';
    const labelWidth = Math.min(width - p.x - 8, entityGraphCtx.measureText(text).width + 6);
    if(labelWidth > 16) entityGraphCtx.fillRect(p.x + 7, p.y - 1, labelWidth, 13);
    entityGraphCtx.fillStyle = '#dce8f2';
    entityGraphCtx.fillText(text, p.x + 10, p.y);
  }
  const timelineY = height - 17;
  entityGraphCtx.strokeStyle = 'rgba(174,185,195,.3)';
  entityGraphCtx.beginPath();
  entityGraphCtx.moveTo(8, timelineY);
  entityGraphCtx.lineTo(width - 8, timelineY);
  entityGraphCtx.stroke();
  const recent = events.slice(-16);
  recent.forEach((event, index) => {
    const x = 10 + index * ((width - 20) / Math.max(1, recent.length - 1));
    entityGraphCtx.fillStyle = graphNodeColor(byId.get(event.node_id)?.node_type);
    entityGraphCtx.globalAlpha = clamp(.28 + (event.confidence || .5) * .72, .28, 1);
    entityGraphCtx.fillRect(x - 2, timelineY - 5, 4, 10);
  });
  entityGraphCtx.globalAlpha = 1;
  entityGraphCtx.fillStyle = 'rgba(220,232,242,.8)';
  entityGraphCtx.font = '9px monospace';
  entityGraphCtx.textBaseline = 'bottom';
  edges.slice(0, 6).forEach((edge, index) => {
    const a = entityGraphState.positions.get(edge.from);
    const b = entityGraphState.positions.get(edge.to);
    if(!a || !b) return;
    const midX = (a.x + b.x) / 2;
    const midY = (a.y + b.y) / 2;
    entityGraphCtx.fillText((edge.confidence || 0).toFixed(2), midX + 2, midY - 2 - index % 2 * 8);
  });
  const typeCounts = nodes.reduce((counts, node) => {
    counts[node.node_type] = (counts[node.node_type] || 0) + 1;
    return counts;
  }, {});
  const sensations = typeCounts.observation || 0;
  const experiences = typeCounts.cluster || 0;
  const impressions = typeCounts.text_label || 0;
  const entities = typeCounts.entity || 0;
  if(entityGraphState.selectedNodeId){
    const selected = byId.get(entityGraphState.selectedNodeId);
    if(selected?.node_type === 'entity'){
      const clusterIds = edges
        .filter(edge => edge.to === selected.id && edge.edge_type === 'part_of_entity')
        .map(edge => edge.from);
      const observationIds = edges
        .filter(edge => edge.edge_type === 'belongs_to' && clusterIds.includes(edge.to))
        .map(edge => edge.from);
      const observations = observationIds
        .map(id => byId.get(id))
        .filter(Boolean)
        .slice(0, 4)
        .map(node => node.source_channel || node.label);
      entityGraphSummary.textContent = `Sensations ${sensations} · Experience ${experiences} · Impressions ${impressions} | ${selected.label}: clusters ${clusterIds.length}, observations ${observationIds.length}${observations.length ? ` [${observations.join(', ')}]` : ''}`;
      return;
    }
  }
  entityGraphSummary.textContent = `Sensations ${sensations} · Experience ${experiences} · Impressions ${impressions} · Entities ${entities} | ${edges.length} bindings`;
}

entityGraphCanvas.addEventListener('click', (event) => {
  const nearest = nearestEntityGraphNode(event.clientX, event.clientY);
  entityGraphState.selectedNodeId = nearest?.id || null;
  if(entityGraphState.lastGraph){
    drawEntityGraph(entityGraphState.lastGraph);
  }
});

function drawTraceMap(packet, liveMap=null){
  updateTraceState(packet);
  const w = traceCanvas.width, h = traceCanvas.height;
  traceCtx.clearRect(0, 0, w, h);
  traceCtx.fillStyle = '#081015';
  traceCtx.fillRect(0, 0, w, h);
  const layers = enabledMapLayers();
  const poses = liveMap?.pose_trail?.length ? liveMap.pose_trail : traceState.poses;
  const latest = liveMap?.current_pose || poses[poses.length - 1] || {x_m:0, y_m:0, heading_rad:0};
  const scale = 42;
  const sx = x => w / 2 + (x - latest.x_m) * scale;
  const sy = y => h / 2 - (y - latest.y_m) * scale;
  if(layers.has('occupancy')){
    if(liveMap?.cells?.length){
      for(const cell of liveMap.cells){
        const occupied = Number(cell.occupied_score) > Number(cell.free_score);
        const alpha = Math.max(.16, Math.min(.78, Number(cell.confidence) || .2));
        traceCtx.fillStyle = occupied ? `rgba(255,182,86,${alpha})` : `rgba(77,144,185,${alpha * .45})`;
        const size = occupied ? 4 : 2;
        traceCtx.fillRect(sx(cell.center_x_m)-size/2, sy(cell.center_y_m)-size/2, size, size);
      }
    }else{
      traceCtx.fillStyle = 'rgba(77,144,185,.22)';
      for(const cell of traceState.free.values()) traceCtx.fillRect(sx(cell.x)-1, sy(cell.y)-1, 2, 2);
      traceCtx.fillStyle = 'rgba(255,182,86,.62)';
      for(const cell of traceState.occupied.values()) traceCtx.fillRect(sx(cell.x)-2, sy(cell.y)-2, 4, 4);
    }
  }
  if((layers.has('rays') || layers.has('free-space rays')) && liveMap?.range_beams?.length){
    traceCtx.lineWidth = 1;
    for(const beam of liveMap.range_beams){
      traceCtx.strokeStyle = beam.hit ? 'rgba(255,207,102,.46)' : 'rgba(82,169,255,.22)';
      traceCtx.beginPath();
      traceCtx.moveTo(sx(beam.origin_x_m), sy(beam.origin_y_m));
      traceCtx.lineTo(sx(beam.end_x_m), sy(beam.end_y_m));
      traceCtx.stroke();
    }
  }
  const grid = packet.surface_perception?.obstacle_grid;
  if(layers.has('occupancy') && grid?.cells?.length){
    const res = Number(grid.resolution_m) || .1;
    for(const cell of grid.cells){
      const localX = (Number(cell.x) + .5) * res;
      const localY = (Number(cell.y) + .5) * res;
      const p = robotLocalToWorld(latest, localX, localY);
      const occupied = cell.state === 'occupied' || cell.state?.Occupied != null;
      traceCtx.fillStyle = occupied ? 'rgba(255,102,126,.72)' : 'rgba(91,220,159,.26)';
      const size = occupied ? 4 : 2;
      traceCtx.fillRect(sx(p.x)-size/2, sy(p.y)-size/2, size, size);
    }
  }
  if(liveMap?.semantic_cells?.length){
    for(const cell of liveMap.semantic_cells){
      if(!layers.has(cell.kind)) continue;
      const colors = {'danger':'#ff5a70','charger/charge':'#45d483','social':'#e8c08c','novelty':'#a889ff','current':'#dce8f2'};
      traceCtx.fillStyle = colors[cell.kind] || '#dce8f2';
      traceCtx.globalAlpha = Math.max(.24, Math.min(.9, Number(cell.confidence) || .6));
      traceCtx.beginPath();
      traceCtx.arc(sx(cell.x_m), sy(cell.y_m), 4 + 4 * Math.min(1, Number(cell.score) || .5), 0, Math.PI * 2);
      traceCtx.fill();
      traceCtx.globalAlpha = 1;
    }
  }
  traceCtx.strokeStyle = '#8df0b2';
  traceCtx.lineWidth = 1.5;
  traceCtx.beginPath();
  poses.forEach((pose, index) => {
    if(index === 0) traceCtx.moveTo(sx(pose.x_m), sy(pose.y_m));
    else traceCtx.lineTo(sx(pose.x_m), sy(pose.y_m));
  });
  traceCtx.stroke();
  const events = liveMap?.events?.length ? liveMap.events : traceState.events;
  if(layers.has('events')) for(const event of events){
    traceCtx.fillStyle = event.kind === 'safety' ? '#ff4d5f' : event.kind === 'stuck' ? '#ffd166' : '#ff8b4a';
    traceCtx.beginPath();
    traceCtx.arc(sx(event.x_m ?? event.x), sy(event.y_m ?? event.y), 3, 0, Math.PI * 2);
    traceCtx.fill();
  }
  traceCtx.save();
  traceCtx.translate(sx(latest.x_m), sy(latest.y_m));
  traceCtx.rotate(-latest.heading_rad);
  traceCtx.fillStyle = '#dce8f2';
  traceCtx.beginPath();
  traceCtx.moveTo(9, 0);
  traceCtx.lineTo(-6, -5);
  traceCtx.lineTo(-4, 0);
  traceCtx.lineTo(-6, 5);
  traceCtx.closePath();
  traceCtx.fill();
  traceCtx.restore();
  drawEntityGraph(liveMap?.entity_graph);
}

function resetWebJoystick(){
  webReignVector = {x: 0, y: 0};
  reignStick.style.transform = 'translate(-50%,-50%)';
  reignReadout.textContent = 'F 0.00 / T 0.00';
}

function updateWebJoystick(event){
  const rect = reignJoystick.getBoundingClientRect();
  const radius = rect.width / 2;
  const usable = Math.max(1, radius - 17);
  const cx = rect.left + radius;
  const cy = rect.top + radius;
  const dx = event.clientX - cx;
  const dy = event.clientY - cy;
  const distance = Math.min(usable, Math.hypot(dx, dy));
  const angle = Math.atan2(dy, dx);
  const knobX = Math.cos(angle) * distance;
  const knobY = Math.sin(angle) * distance;
  reignStick.style.transform = `translate(calc(-50% + ${knobX}px), calc(-50% + ${knobY}px))`;
  webReignVector = {
    x: clamp(dx / usable, -1, 1),
    y: clamp(dy / usable, -1, 1)
  };
  reignReadout.textContent = `F ${fmt(-webReignVector.y, 2)} / T ${fmt(webReignVector.x, 2)}`;
}

function commandForWebJoystick(){
  const {x, y} = webReignVector;
  const magnitude = Math.hypot(x, y);
  if(magnitude < .18) return null;
  const forward = Math.abs(y) < .12 ? 0 : -Math.sign(y) * clamp(Math.pow(Math.abs(y), 1.25), .08, 1);
  const turn = Math.abs(x) < .12 ? 0 : -Math.sign(x) * clamp(Math.pow(Math.abs(x), 1.35), .08, 1);
  return {
    command:{type:'Drive', forward, turn, duration_ms:320},
    ttl:680,
    label:`drive F ${fmt(forward, 2)} T ${fmt(turn, 2)}`,
    priority:.96
  };
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
reignStop.addEventListener('click', () => {
  postWebReign({type:'Stop'}, 2000, 'stop', 1);
});
hardwareArm.addEventListener('change', () => setHardwareArmed(hardwareArm.checked));
window.addEventListener('pagehide', disarmHardwareOnExit);
window.addEventListener('beforeunload', disarmHardwareOnExit);

function commandForInputSource(inputSource){
  const gamepad = inputSource.gamepad;
  if(!gamepad) return null;
  if(buttonPressed(gamepad, 1) || buttonPressed(gamepad, 3)) {
    return {command:{type:'Stop'}, ttl:900, label:'stop', priority:1};
  }
  const {x, y} = stickAxes(gamepad);
  if(Math.hypot(x, y) > .22) {
    const forward = Math.abs(y) < .14 ? 0 : -Math.sign(y) * clamp(Math.pow(Math.abs(y), 1.25), .08, 1);
    const turn = Math.abs(x) < .14 ? 0 : -Math.sign(x) * clamp(Math.pow(Math.abs(x), 1.35), .08, 1);
    return {
      command:{type:'Drive', forward, turn, duration_ms:320},
      ttl:680,
      label:`drive F ${fmt(forward, 2)} T ${fmt(turn, 2)}`,
      priority:.96
    };
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
    const sceneResponse = await fetch('/view/scene', {cache:'no-store'});
    if(!sceneResponse.ok){
      const statusText = await sceneResponse.text();
      throw new Error(statusText || 'scene unavailable');
    }
    const scene = await sceneResponse.json();
    if(!scene) throw new Error('invalid snapshot payload');
    let liveMap = null;
    try{
      const mapResponse = await fetch('/view/map', {cache:'no-store'});
      if(mapResponse.ok) liveMap = await mapResponse.json();
    }catch(_error){}
    updateScene(scene, liveMap);
    statusEl.textContent = 'live';
  }catch(error){
    statusEl.textContent = 'waiting for scene packets...';
  }finally{
    setTimeout(poll, 250);
  }
}

function sceneFromSnapshot(snapshot){
  if(!snapshot?.body) return null;
  const body = snapshot.body;
  const pose = body.odometry || {};
  const range = snapshot.range || {};
  const beams = (range.beams || []).map((distance, index, allBeams) => {
    const finite = Number(distance);
    const beamCount = allBeams.length || 1;
    const ratio = beamCount <= 1 ? 0.5 : index / (beamCount - 1);
    return {
      angle_rad: -Math.PI * .5 + ratio * Math.PI,
      distance_m: Number.isFinite(finite) ? finite : 0,
      hit: Number.isFinite(range.nearest_m) ? Math.abs(finite - range.nearest_m) < 0.05 : false
    };
  });
  return {
    schema_version: 1,
    t_ms: body.last_update_ms || 0,
    body: {
      x_m: pose.x_m || 0,
      y_m: pose.y_m || 0,
      heading_rad: pose.heading_rad || 0,
      battery_level: body.battery_level == null ? 1 : body.battery_level,
      charging: !!body.charging,
      bump_left: !!body.flags?.bump_left,
      bump_right: !!body.flags?.bump_right,
      cliff: !!(
        body.flags?.cliff_left ||
        body.flags?.cliff_front_left ||
        body.flags?.cliff_front_right ||
        body.flags?.cliff_right
      ),
      wheel_drop: !!body.flags?.wheel_drop
    },
    range: {
      nearest_m: range.nearest_m == null ? null : range.nearest_m,
      beams,
    },
    eye: null,
    kinect: {points: [], coordinate_system: 'camera', skeletons: []},
    objects: [],
    session: snapshot.session || null,
    hardware_control: snapshot.hardware_control || null,
    arena: null,
    action: {},
    prod: {},
    stuck: false,
    dead_battery: false,
    recovery_mode: null,
    stuck_ticks: 0,
    stuck_detail: {},
    training_mode: 'standalone',
    frames_written: 0,
    transitions_written: 0,
    models_loaded: [],
    model_modes: {},
    behavior_nodes: [],
    action_selector_mode: '-',
    weights_updating: false,
    training: {training_mode: 'standalone'}
  };
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
const panelIds = ['hud', 'learning', 'experience-forge', 'reign', 'llm', 'virtual-pipeline-section', 'models', 'model-graph-window', 'calibration'];

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
  depthZ: 0.0,
  depthPitch: 0,
  depthRoll: 0,
  depthYaw: 0,
  depthFov: 122,
  cameraFov: 62,
  cameraY: 0.46,
  cameraZ: -0.18,
  cameraPitch: 0
};

let cal = { ...defaults };

const depthScaleEl = document.getElementById('cal-depth-scale');
const pointYEl = document.getElementById('cal-point-y');
const depthZEl = document.getElementById('cal-depth-z');
const depthPitchEl = document.getElementById('cal-depth-pitch');
const depthRollEl = document.getElementById('cal-depth-roll');
const depthYawEl = document.getElementById('cal-depth-yaw');
const depthFovEl = document.getElementById('cal-depth-fov');
const cameraFovEl = document.getElementById('cal-camera-fov');
const cameraYEl = document.getElementById('cal-camera-y');
const cameraZEl = document.getElementById('cal-camera-z');
const cameraPitchEl = document.getElementById('cal-camera-pitch');
const resetBtn = document.getElementById('reset-calibration');
const statusEl2 = document.getElementById('calibration-status');

const valDepthScale = document.getElementById('val-depth-scale');
const valPointY = document.getElementById('val-point-y');
const valDepthZ = document.getElementById('val-depth-z');
const valDepthPitch = document.getElementById('val-depth-pitch');
const valDepthRoll = document.getElementById('val-depth-roll');
const valDepthYaw = document.getElementById('val-depth-yaw');
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
  depthZEl.value = cal.depthZ;
  depthPitchEl.value = cal.depthPitch;
  depthRollEl.value = cal.depthRoll;
  depthYawEl.value = cal.depthYaw;
  depthFovEl.value = cal.depthFov;
  cameraFovEl.value = cal.cameraFov;
  cameraYEl.value = cal.cameraY;
  cameraZEl.value = cal.cameraZ;
  cameraPitchEl.value = cal.cameraPitch;

  valDepthScale.textContent = cal.depthScale.toFixed(2);
  valPointY.textContent = cal.pointY.toFixed(2) + 'm';
  valDepthZ.textContent = cal.depthZ.toFixed(2) + 'm';
  valDepthPitch.textContent = cal.depthPitch + '°';
  valDepthRoll.textContent = cal.depthRoll + '°';
  valDepthYaw.textContent = cal.depthYaw + '°';
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
          point_y_m: cal.pointY,
          depth_forward_offset_m: cal.depthZ,
          depth_pitch_down_rad: cal.depthPitch * Math.PI / 180,
          camera_forward_m: cal.depthZ,
          camera_height_m: cal.pointY,
          camera_pitch_rad: cal.depthPitch * Math.PI / 180,
          camera_roll_rad: cal.depthRoll * Math.PI / 180,
          camera_yaw_rad: cal.depthYaw * Math.PI / 180
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
  cal.depthZ = parseFloat(depthZEl.value);
  cal.depthPitch = parseInt(depthPitchEl.value, 10);
  cal.depthRoll = parseInt(depthRollEl.value, 10);
  cal.depthYaw = parseInt(depthYawEl.value, 10);
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

[depthScaleEl, pointYEl, depthZEl, depthPitchEl, depthRollEl, depthYawEl, depthFovEl, cameraFovEl, cameraYEl, cameraZEl, cameraPitchEl].forEach(el => {
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
    use netherwick_core::Pose2;
    use netherwick_map::{
        OrientationEstimate, Point3D, PointCloudFrame, PointCloudObservation, PointCloudPoint,
        PoseEstimate, YawSource,
    };
    use std::sync::{Arc, Mutex};

    use anyhow::Result;
    use netherwick_actions::{ActionPrimitive, TurnDir};
    use netherwick_autonomic::SimpleSafety;
    use netherwick_body::BodySense;
    use netherwick_conductor::{Conductor, ConductorInput};
    use netherwick_experience::{
        Experience, ExperienceLatent, Impression, MemoryLink, Modality, Prediction, Sensation,
        SensationMetadata, SensationPayload, SensationPayloadKind, SensationSource,
        VectorEmbedding,
    };
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

    fn hardware_reign_state_with_snapshot(snapshot: WorldSnapshot) -> ReignServerState {
        let live = LiveViewState::new().with_real_slow_hardware_control();
        live.update(snapshot);
        ReignServerState::with_live_view(Arc::new(Mutex::new(ReignQueue::default())), &live)
    }

    fn recent_snapshot() -> WorldSnapshot {
        let mut snapshot = WorldSnapshot::default();
        snapshot.body.last_update_ms = wall_now_ms();
        snapshot.body.battery_level = 1.0;
        snapshot
    }

    #[tokio::test]
    async fn hardware_drive_command_rejected_when_disarmed() {
        let state = hardware_reign_state_with_snapshot(recent_snapshot());
        let request = ReignCommandRequest {
            mode: ReignMode::Direct,
            command: ReignCommand::Go {
                intensity: 0.12,
                duration_ms: 300,
            },
            priority: 1.0,
            ttl_ms: Some(300),
            note: None,
            source: Some(ReignSource::WebRemote),
        };

        let error = post_reign_command(State(state), Json(request))
            .await
            .unwrap_err();

        assert_eq!(error.status, StatusCode::FORBIDDEN);
        assert!(error.message.contains("disarmed"));
    }

    #[tokio::test]
    async fn hardware_stop_accepted_when_disarmed() {
        let state = hardware_reign_state_with_snapshot(recent_snapshot());
        let request = ReignCommandRequest {
            mode: ReignMode::Direct,
            command: ReignCommand::Stop,
            priority: 1.0,
            ttl_ms: Some(300),
            note: None,
            source: Some(ReignSource::WebRemote),
        };

        let Json(input) = post_reign_command(State(state.clone()), Json(request))
            .await
            .unwrap();

        assert!(matches!(input.command, ReignCommand::Stop));
        assert!(
            state
                .queue()
                .lock()
                .unwrap()
                .sense(input.issued_at_ms)
                .active
        );
    }

    #[tokio::test]
    async fn hardware_arm_rejected_in_non_hardware_session() {
        let state = ReignServerState::standalone();

        let error = post_hardware_arm(State(state), Json(HardwareArmRequest { armed: true }))
            .await
            .unwrap_err();

        assert_eq!(error.status, StatusCode::FORBIDDEN);
        assert!(error.message.contains("not available"));
    }

    #[tokio::test]
    async fn hardware_disarm_enqueues_stop() {
        let state = hardware_reign_state_with_snapshot(recent_snapshot());
        let Json(armed) = post_hardware_arm(
            State(state.clone()),
            Json(HardwareArmRequest { armed: true }),
        )
        .await
        .unwrap();
        assert!(armed.armed);

        let Json(disarmed) = post_hardware_arm(
            State(state.clone()),
            Json(HardwareArmRequest { armed: false }),
        )
        .await
        .unwrap();
        let sense = state.queue().lock().unwrap().sense(wall_now_ms());

        assert!(!disarmed.armed);
        assert!(matches!(
            sense.latest.as_ref().map(|input| &input.command),
            Some(ReignCommand::Stop)
        ));
    }

    #[tokio::test]
    async fn hardware_armed_drive_uses_cautious_ttl_and_clamps_command() {
        let state = hardware_reign_state_with_snapshot(recent_snapshot());
        let _ = post_hardware_arm(
            State(state.clone()),
            Json(HardwareArmRequest { armed: true }),
        )
        .await
        .unwrap();
        let request = ReignCommandRequest {
            mode: ReignMode::Direct,
            command: ReignCommand::Go {
                intensity: 0.9,
                duration_ms: 5_000,
            },
            priority: 1.0,
            ttl_ms: Some(5_000),
            note: None,
            source: Some(ReignSource::WebRemote),
        };

        let Json(input) = post_reign_command(State(state), Json(request))
            .await
            .unwrap();

        assert_eq!(input.source, ReignSource::WebRemote);
        assert_eq!(input.mode, ReignMode::Direct);
        assert_eq!(
            input.expires_at_ms - input.issued_at_ms,
            HARDWARE_TTL_MAX_MS
        );
        assert!(matches!(
            input.command,
            ReignCommand::Go {
                intensity,
                duration_ms
            } if intensity <= HARDWARE_MAX_FORWARD_INTENSITY && duration_ms == HARDWARE_TTL_MAX_MS
        ));
    }

    #[tokio::test]
    async fn hardware_armed_analog_drive_clamps_both_axes() {
        let state = hardware_reign_state_with_snapshot(recent_snapshot());
        let _ = post_hardware_arm(
            State(state.clone()),
            Json(HardwareArmRequest { armed: true }),
        )
        .await
        .unwrap();
        let request = ReignCommandRequest {
            mode: ReignMode::Direct,
            command: ReignCommand::Drive {
                forward: 0.9,
                turn: -0.8,
                duration_ms: 5_000,
            },
            priority: 1.0,
            ttl_ms: Some(5_000),
            note: None,
            source: Some(ReignSource::WebRemote),
        };

        let Json(input) = post_reign_command(State(state), Json(request))
            .await
            .unwrap();

        assert_eq!(
            input.expires_at_ms - input.issued_at_ms,
            HARDWARE_TTL_MAX_MS
        );
        assert!(matches!(
            input.command,
            ReignCommand::Drive {
                forward,
                turn,
                duration_ms
            } if forward == HARDWARE_MAX_FORWARD_INTENSITY
                && turn == -HARDWARE_MAX_TURN_INTENSITY
                && duration_ms == HARDWARE_TTL_MAX_MS
        ));
    }

    #[tokio::test]
    async fn hardware_drive_rejected_on_cliff_even_when_armed() {
        let mut snapshot = recent_snapshot();
        snapshot.body.flags.cliff_front_left = true;
        let state = hardware_reign_state_with_snapshot(snapshot);
        let _ = post_hardware_arm(
            State(state.clone()),
            Json(HardwareArmRequest { armed: true }),
        )
        .await
        .unwrap();
        let request = ReignCommandRequest {
            mode: ReignMode::Direct,
            command: ReignCommand::Turn {
                direction: TurnDir::Left,
                intensity: 0.20,
                duration_ms: 300,
            },
            priority: 1.0,
            ttl_ms: Some(300),
            note: None,
            source: Some(ReignSource::WebRemote),
        };

        let error = post_reign_command(State(state), Json(request))
            .await
            .unwrap_err();

        assert_eq!(error.status, StatusCode::FORBIDDEN);
        assert!(error.message.contains("cliff"));
    }

    #[tokio::test]
    async fn hardware_drive_allows_analog_cliff_risk_without_create_cliff_flag() {
        let mut snapshot = recent_snapshot();
        snapshot.body.cliff_sensors.front_left = 0.96;
        snapshot.body.cliff_sensors.front_right = 0.82;
        let state = hardware_reign_state_with_snapshot(snapshot);
        let _ = post_hardware_arm(
            State(state.clone()),
            Json(HardwareArmRequest { armed: true }),
        )
        .await
        .unwrap();
        let request = ReignCommandRequest {
            mode: ReignMode::Direct,
            command: ReignCommand::Turn {
                direction: TurnDir::Left,
                intensity: 0.20,
                duration_ms: 300,
            },
            priority: 1.0,
            ttl_ms: Some(300),
            note: None,
            source: Some(ReignSource::WebRemote),
        };

        let Json(input) = post_reign_command(State(state), Json(request))
            .await
            .unwrap();

        assert!(matches!(input.command, ReignCommand::Turn { .. }));
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

    #[tokio::test]
    async fn live_map_endpoint_returns_pose_projected_beams_and_overlays() {
        let state = LiveViewState::new();
        state.update_scene_metadata(LiveSceneMetadata {
            arena: None,
            objects: vec![
                SceneObject {
                    id: "charger-0".to_string(),
                    kind: "charger".to_string(),
                    x_m: 2.0,
                    y_m: 0.5,
                    radius_m: 0.2,
                    label: Some("charger".to_string()),
                    color_rgb: None,
                },
                SceneObject {
                    id: "person-0".to_string(),
                    kind: "person".to_string(),
                    x_m: 0.2,
                    y_m: 1.8,
                    radius_m: 0.25,
                    label: Some("person".to_string()),
                    color_rgb: None,
                },
                SceneObject {
                    id: "speaker-0".to_string(),
                    kind: "speaker".to_string(),
                    x_m: -0.4,
                    y_m: 1.2,
                    radius_m: 0.15,
                    label: Some("speaker".to_string()),
                    color_rgb: None,
                },
            ],
            sensor_calibration: None,
        });

        let mut snapshot = WorldSnapshot::default();
        snapshot.body.odometry.x_m = 0.5;
        snapshot.body.odometry.y_m = 0.75;
        snapshot.body.odometry.heading_rad = std::f32::consts::FRAC_PI_2;
        snapshot.body.last_update_ms = 1234;
        snapshot.body.flags.bump_left = true;
        snapshot.body.flags.cliff_front_left = true;
        snapshot.range.beams = vec![2.0, 1.0, 2.0];
        snapshot.range.nearest_m = Some(1.0);
        snapshot
            .objects
            .observations
            .push(netherwick_now::ObjectObservation {
                label: "person-nearby".to_string(),
                class: netherwick_now::ObjectClass::Person,
                bearing_rad: 0.1,
                distance_m: Some(1.2),
                confidence: 0.82,
                source: netherwick_now::ObjectObservationSource::Sim,
            });
        snapshot.ear.transcript = Some("Travis".to_string());
        snapshot.llm_action_proposal = Some(netherwick_actions::LlmActionProposal {
            safety_vetoed: true,
            ..netherwick_actions::LlmActionProposal::default()
        });
        snapshot.extensions.push(netherwick_now::ExtensionSense {
            schema_version: 1,
            name: "sim.stuck".to_string(),
            values: vec![1.0, 0.0, 4.0, 200.0, 1.0],
        });
        state.update(snapshot.clone());
        snapshot.body.last_update_ms = 1334;
        state.update(snapshot);

        let Json(map) = get_live_map(State(state)).await.unwrap();

        assert_eq!(map.schema_version, 1);
        assert_eq!(map.label, MAP_LABEL);
        assert!(map.summary.label.contains("SLAM-lite"));
        assert!(map.summary.label.contains("odometry map"));
        assert_eq!(map.pose_trail.len(), 2);
        assert_eq!(map.current_pose.as_ref().unwrap().x_m, 0.5);
        assert_eq!(
            map.overlays,
            vec![
                "occupancy",
                "rays",
                "raw point cloud",
                "accumulated occupancy",
                "stable wall candidates",
                "danger",
                "charger/charge",
                "social",
                "novelty",
                "events"
            ]
        );
        assert_eq!(map.range_beams.len(), 3);
        assert!(!map.cells.is_empty());
        assert!(map
            .cells
            .iter()
            .any(|cell| cell.occupied_score > cell.free_score));
        assert!(map
            .cells
            .iter()
            .any(|cell| cell.free_score >= cell.occupied_score));
        assert!(map
            .semantic_cells
            .iter()
            .any(|cell| cell.kind == "charger/charge" && cell.label.as_deref() == Some("charger")));
        assert!(map
            .semantic_cells
            .iter()
            .any(|cell| cell.kind == "social" && cell.label.as_deref() == Some("person")));
        assert!(map
            .semantic_cells
            .iter()
            .any(|cell| cell.kind == "social" && cell.label.as_deref() == Some("speaker")));
        assert!(map.events.iter().any(|event| event.kind == "bump"));
        assert!(map.events.iter().any(|event| event.kind == "cliff"));
        assert!(map.events.iter().any(|event| event.kind == "stuck"));
        assert!(map
            .events
            .iter()
            .any(|event| event.kind == "safety_override"));
        assert!(map.events.iter().any(|event| event.kind == "charger"));
        assert!(map.events.iter().any(|event| event.kind == "person"));
        assert!(map.events.iter().any(|event| event.kind == "speaker"));
        assert_eq!(map.entity_graph.schema_version, 1);
        assert!(map
            .entity_graph
            .nodes
            .iter()
            .any(|node| node.node_type == "entity"));
        assert!(map
            .entity_graph
            .nodes
            .iter()
            .any(|node| node.node_type == "text_label"));
        assert!(map
            .entity_graph
            .nodes
            .iter()
            .any(|node| node.node_type == "observation" && node.source_channel.is_some()));
        assert!(map
            .entity_graph
            .edges
            .iter()
            .any(|edge| edge.edge_type == "named_by"));
        assert!(map
            .entity_graph
            .edges
            .iter()
            .any(|edge| edge.observed_at_ms.is_some()));
        assert!(!map.entity_graph.events.is_empty());
        assert!(map
            .entity_graph
            .events
            .iter()
            .any(|event| event.event_type == "create"));
        assert!(map
            .entity_graph
            .events
            .iter()
            .any(|event| event.event_type == "strengthen"));
        assert!(map
            .entity_graph
            .events
            .iter()
            .all(|event| event.timestamp_ms.is_some()));

        let forward_hit = map
            .range_beams
            .iter()
            .find(|beam| beam.hit)
            .expect("nearest beam should be marked as hit");
        assert!((forward_hit.origin_x_m - 0.5).abs() < 0.001);
        assert!((forward_hit.origin_y_m - 0.75).abs() < 0.001);
        assert!((forward_hit.end_x_m - 0.5).abs() < 0.001);
        assert!((forward_hit.end_y_m - 1.75).abs() < 0.001);
        assert!((forward_hit.angle_rad - 0.0).abs() < 0.001);
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
            None,
            HardwareControlStatus::unavailable("unit test"),
        );
        let value = serde_json::to_value(scene).unwrap();

        assert!(value["eye"].is_null());
        assert_eq!(value["dead_battery"].as_bool(), Some(true));
        assert_eq!(value["kinect"]["points"].as_array().unwrap().len(), 0);
        assert!(value["audio"].is_null());
        assert!(value["warnings"].as_array().unwrap().len() >= 3);
    }

    #[test]
    fn scene_packet_exposes_persistent_stable_world_belief_points() {
        let mut snapshot = WorldSnapshot::default();
        snapshot.body.last_update_ms = 300;
        let mut cloud = VoxelPointCloud::default();
        for t_ms in [100, 200, 300] {
            cloud.integrate_observation(PointCloudObservation {
                frame: PointCloudFrame::OdometryWorld,
                pose: PoseEstimate {
                    pose: Pose2::default(),
                    confidence: 0.9,
                    covariance: [0.01, 0.01, 0.02],
                    source: "unit-test".to_string(),
                    t_ms,
                },
                orientation: OrientationEstimate {
                    roll_rad: Some(0.01),
                    pitch_rad: Some(-0.02),
                    yaw_rad: Some(0.0),
                    roll_pitch_from_imu: true,
                    yaw_source: YawSource::OdometryHeading,
                },
                points: vec![PointCloudPoint {
                    position: Point3D {
                        x_m: 1.0,
                        y_m: 0.5,
                        z_m: 0.2,
                    },
                    color_rgb: None,
                    confidence: 1.0,
                }],
                source: "unit-test".to_string(),
                t_ms,
                metadata: serde_json::json!({}),
            });
        }

        let scene = snapshot_to_scene(
            &snapshot,
            None,
            None,
            LiveTrainingStatus::default(),
            NudgeStatus::default(),
            default_behavior_nodes(),
            Some(&cloud),
            None,
            HardwareControlStatus::unavailable("unit test"),
        );

        assert_eq!(
            scene.world_belief_layers,
            vec![
                "current rays",
                "raw point cloud",
                "accumulated occupancy",
                "stable wall candidates"
            ]
        );
        assert_eq!(
            scene
                .kinect
                .accumulated_summary
                .as_ref()
                .unwrap()
                .stable_voxels,
            1
        );
        assert_eq!(scene.kinect.accumulated_points.len(), 1);
        assert!(scene.kinect.accumulated_points[0].stable);
        let belief = scene.kinect.local_world_belief.as_ref().unwrap();
        assert_eq!(belief.stable_voxels, 1);
        assert!(belief.orientation_status.roll_pitch_corrected);
        assert_eq!(scene.kinect.coordinate_system.as_deref(), Some("world"));
    }

    #[test]
    fn scene_body_cliff_uses_create_flags_not_analog_risk() {
        let mut snapshot = WorldSnapshot::default();
        snapshot.body.cliff_sensors.front_left = 0.96;
        snapshot.body.cliff_sensors.front_right = 0.82;
        let scene = snapshot_to_scene(
            &snapshot,
            None,
            None,
            LiveTrainingStatus::default(),
            NudgeStatus::default(),
            default_behavior_nodes(),
            None,
            None,
            HardwareControlStatus::unavailable("unit test"),
        );

        assert!(!scene.body.cliff);

        snapshot.body.flags.cliff_front_left = true;
        let scene = snapshot_to_scene(
            &snapshot,
            None,
            None,
            LiveTrainingStatus::default(),
            NudgeStatus::default(),
            default_behavior_nodes(),
            None,
            None,
            HardwareControlStatus::unavailable("unit test"),
        );

        assert!(scene.body.cliff);
    }

    #[test]
    fn compact_kinect_depth_projects_as_meter_range_fan() {
        let depths = vec![2.0; 32];
        let kinect = KinectSense {
            depth_m: depths,
            ..KinectSense::default()
        };
        let (points, diagnostics) =
            depth_points(&kinect, Some(SceneSensorCalibration::sim_default()));

        assert_eq!(points.len(), 32);
        assert_eq!(diagnostics.coordinate_system, "robot");
        assert!((points[0].x + 1.847759).abs() < 0.001);
        assert!((points[0].z - 0.765367).abs() < 0.001);
        assert!((points[31].x - 1.847759).abs() < 0.001);
        assert!((points[31].z - 0.765367).abs() < 0.001);

        let near_center = &points[15];
        assert!(near_center.z > 1.99);
        assert!(near_center.x.abs() < 0.11);
    }

    #[test]
    fn kinect_depth_image_projects_with_pinhole_intrinsics() {
        let kinect = KinectSense {
            depth_m: vec![1.0, 2.0, 0.0, 4.0],
            depth_width: 2,
            depth_height: 2,
            depth_fx: 2.0,
            depth_fy: 2.0,
            depth_cx: 0.5,
            depth_cy: 0.5,
            min_depth_m: 0.4,
            max_depth_m: 3.0,
            depth_coordinate_system: Some("kinect_camera".to_string()),
            ..KinectSense::default()
        };

        let (points, diagnostics) = depth_points(&kinect, None);

        assert_eq!(points.len(), 2);
        assert_eq!(diagnostics.depth_width, 2);
        assert_eq!(diagnostics.depth_height, 2);
        assert_eq!(diagnostics.valid_depth_count, 2);
        assert_eq!(diagnostics.skipped_depth_count, 1);
        assert_eq!(diagnostics.clipped_depth_count, 1);
        assert_eq!(diagnostics.coordinate_system, "kinect_camera");
        assert!((points[0].x + 0.25).abs() < 0.001);
        assert!((points[0].y + 0.25).abs() < 0.001);
        assert!((points[0].z - 1.0).abs() < 0.001);
        assert!((points[1].x - 0.5).abs() < 0.001);
        assert!((points[1].y + 0.5).abs() < 0.001);
        assert!((points[1].z - 2.0).abs() < 0.001);
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
        assert!(HTTP_ENDPOINTS.contains(&"/view/map"));
        assert!(HTTP_ENDPOINTS.contains(&"/view/embodied"));
        assert!(HTTP_ENDPOINTS.contains(&"/view/embodied/graph"));
        assert!(HTTP_ENDPOINTS.contains(&"/api/experience/lineage"));
        assert!(HTTP_ENDPOINTS.contains(&"/debug/embodied"));
        assert!(HTTP_ENDPOINTS.contains(&"/debug/embodied/graph"));
        assert!(HTTP_ENDPOINTS.contains(&"/models"));
        assert!(HTTP_ENDPOINTS.contains(&"/stream/llm"));
        let Html(live_page) = live_view_page().await;
        assert!(live_page.contains("Embodied lineage"));
        assert!(live_page.contains("/api/experience/lineage"));
        assert!(live_page.contains("graph_query"));
        assert!(live_page.contains("graph_modality"));
        let Html(page) = live_view_3d_page().await;
        assert!(page.contains("Instant 3D"));
        assert!(page.contains("/view/snapshot"));
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
        assert!(page.contains("type:'Drive'"));
        assert!(page.contains("Real hardware armed"));
        assert!(page.contains("/reign/hardware-arm"));
        assert!(page.contains("window.addEventListener('pagehide'"));
        assert!(page.contains("navigator.sendBeacon"));
        assert!(!page.contains("id=\"reign-dock\""));
        assert!(!page.contains("id=\"reign-explore\""));
        assert!(!page.contains("data-cockpit"));
        assert!(!page.contains("data-heading-nudge"));
        assert!(!page.contains("function startCockpitHold"));
        assert!(!page.contains("function commandForHeadingTarget"));
        assert!(page.contains("'WebRemote'"));
        assert!(page.contains("function projectRangeBeam"));
        assert!(page.contains("/view/map"));
        assert!(page.contains("data-map-layer=\"occupancy\""));
        assert!(page.contains("data-map-layer=\"rays\""));
        assert!(page.contains("data-map-layer=\"raw point cloud\""));
        assert!(page.contains("data-map-layer=\"accumulated occupancy\""));
        assert!(page.contains("data-map-layer=\"stable wall candidates\""));
        assert!(page.contains("function renderWorldBeliefPoints"));
        assert!(page.contains("function renderPersistentWorldBelief"));
        assert!(page.contains("local_world_belief"));
        assert!(page.contains("roll_pitch_corrected"));
        assert!(
            page.contains("SLAM-lite / odometry map: odometry/range trace, not corrected SLAM.")
        );
        assert!(page.contains("id=\"entity-graph\""));
        assert!(page.contains("drawEntityGraph"));
        assert!(page.contains("createDefaultXRExperienceAsync"));
        assert!(page.contains("if(!eye?.data_url)"));
        assert!(page.contains(
            ".panel-window.is-shaded{height:32px!important;min-height:32px!important;max-height:32px!important;"
        ));
    }

    #[tokio::test]
    async fn live_embodied_endpoint_returns_latest_context() {
        let state = LiveViewState::new();
        let context = EmbodiedContext {
            experience_id: Some(uuid::Uuid::new_v4()),
            summary: "I see a frame.".to_string(),
            ..EmbodiedContext::default()
        };
        state.update_embodied_context(context.clone());

        let Json(response) = get_live_embodied(State(state)).await.unwrap();

        assert_eq!(response, context);
    }

    #[test]
    fn embodied_lineage_graph_traces_current_experience() {
        let primary = Sensation::primary(
            Modality::Vision,
            SensationSource::new("unit-camera"),
            100,
            101,
            SensationPayload::image_metadata(32, 24, "rgb8", 32 * 24 * 3),
        )
        .with_summary("I receive a visual frame.");
        let child = Sensation::descendant(
            &primary,
            "vision.crop.focus",
            SensationPayloadKind::Crop,
            serde_json::json!({"x": 4, "y": 3, "width": 12, "height": 9}),
            SensationMetadata::default(),
            "focus",
        )
        .with_summary("I focus on a patch.")
        .with_vector(VectorEmbedding::new(
            vec![0.1, 0.2, 0.3],
            "unit.crop.v0",
            Modality::Vision,
            SensationPayloadKind::Crop,
            primary.id,
            102,
        ));
        let impression = Impression::new(
            "vision.focus.impression",
            "I see and focus.",
            vec![primary.id, child.id],
            100,
            102,
        );
        let mut experience = Experience::new(
            "embodied.now",
            "I see and focus.",
            vec![impression.id],
            vec![primary.id, child.id],
            100,
            102,
        );
        experience.fused_vector = Some(VectorEmbedding::new(
            vec![0.5, 0.6, 0.7, 0.8],
            "unit.fuser.v0",
            Modality::Other,
            SensationPayloadKind::Structured,
            child.id,
            102,
        ));
        experience.predictions.push(Prediction {
            offset_ms: 750,
            text: "The focused view should remain stable.".to_string(),
            confidence: 0.6,
            vector: None,
        });
        experience.memory_links.push(MemoryLink {
            target_id: "memory-1".to_string(),
            relation: "similar".to_string(),
            score: 0.8,
            payload: serde_json::json!({"text": "A previous focused camera moment."}),
        });
        let context = EmbodiedContext::from_current_experience(
            Some(&experience),
            &[primary.clone(), child.clone()],
            &[impression.clone()],
            &[],
            &[],
        );

        let graph = EmbodiedLineageGraph::from_context(&context);

        assert_eq!(graph.schema_version, 1);
        assert_eq!(graph.experience_id, Some(experience.id.to_string()));
        assert!(graph.nodes.iter().any(|node| {
            node.id == format!("sensation:{}", child.id)
                && node.derived
                && node.modality.as_deref() == Some("vision")
        }));
        assert!(graph.edges.iter().any(|edge| {
            edge.from == format!("sensation:{}", primary.id)
                && edge.to == format!("sensation:{}", child.id)
                && edge.relation == EmbodiedGraphEdgeType::ParentSensation
        }));
        assert!(graph.edges.iter().any(|edge| {
            edge.from == format!("sensation:{}", child.id)
                && edge.to == format!("experience:{}", experience.id)
                && edge.relation == EmbodiedGraphEdgeType::SensationMember
        }));
        assert!(graph.edges.iter().any(|edge| {
            edge.from == format!("impression:{}", impression.id)
                && edge.to == format!("experience:{}", experience.id)
                && edge.relation == EmbodiedGraphEdgeType::ImpressionMember
        }));
        assert!(graph
            .nodes
            .iter()
            .any(|node| node.node_type == EmbodiedGraphNodeType::Prediction));
        assert!(graph
            .vector_metadata
            .iter()
            .any(|vector| { vector.model_id == "unit.fuser.v0" && vector.dim == 4 }));
        assert!(graph
            .vector_metadata
            .iter()
            .any(|vector| { vector.model_id == "unit.crop.v0" && vector.dim == 3 }));
        assert_eq!(graph.recent_memories.len(), 1);
        assert_eq!(graph.recent_memories[0].target_id, "memory-1");
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
        for (name, behavior_report, scenario_report) in [
            (
                "danger_golden_column_v0",
                "data/reports/danger-golden-column-heldout-eval.json",
                "data/reports/golden-column-trap-model-assisted-full-shadow.json",
            ),
            (
                "action_value_golden_column_v0",
                "data/reports/action-value-golden-column-heldout-eval.json",
                "data/reports/golden-column-trap-model-assisted-full-shadow.json",
            ),
            (
                "charge_golden_charger_v0",
                "data/reports/charge-golden-charger-heldout-eval.json",
                "data/reports/golden-charger-seeking-heldout.json",
            ),
        ] {
            let entry = packet
                .registry
                .iter()
                .find(|entry| entry.name == name)
                .unwrap_or_else(|| panic!("missing registry entry {name}"));
            assert_eq!(entry.status, "shadow");
            assert_eq!(entry.behavior_report_path.as_deref(), Some(behavior_report));
            assert_eq!(entry.scenario_report_path.as_deref(), Some(scenario_report));
            assert!(entry
                .allowed_modes
                .iter()
                .any(|mode| mode == "shadow-infer"));
            assert!(!entry.allowed_modes.iter().any(|mode| mode == "model-infer"));
        }
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

    #[test]
    fn grbg_bayer_eye_frame_encodes_to_png_data_url() {
        let frame = EyeFrame {
            captured_at_ms: 1,
            width: 2,
            height: 2,
            format: EyeFrameFormat::BayerGrbg8,
            bytes: vec![90, 220, 40, 110],
            source: None,
        };

        let (eye, warnings) = scene_eye_from_frame(&frame, None, 1);

        assert!(warnings.is_empty());
        assert_eq!(eye.width, 2);
        assert_eq!(eye.height, 2);
        assert!(eye
            .data_url
            .as_deref()
            .unwrap_or_default()
            .starts_with("data:image/png;base64,"));
    }

    #[test]
    fn grbg_bayer_eye_frame_encodes_latest_png_bytes() {
        let frame = EyeFrame {
            captured_at_ms: 1,
            width: 2,
            height: 2,
            format: EyeFrameFormat::BayerGrbg8,
            bytes: vec![90, 220, 40, 110],
            source: None,
        };

        let bytes = encode_eye_png_bytes(&frame).unwrap();

        assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
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
