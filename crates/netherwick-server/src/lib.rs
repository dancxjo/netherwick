use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use axum_server::tls_rustls::RustlsConfig;
use base64::Engine;
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageEncoder};
use netherwick_actions::{ReignCommand, ReignInput, ReignMode, ReignSource};
use netherwick_core::TimeMs;
use netherwick_now::{KinectSkeletonSense, ReignSense};
use netherwick_runtime::ReignQueue;
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
    "/reign/state",
    "/reign/clear",
    "/stream/now",
    "/stream/mind",
    "/stream/logs",
    "/view",
    "/view/snapshot",
    "/view/scene",
    "/view/3d",
    "/view/capture-scene",
];

#[derive(Clone, Debug)]
pub struct ReignServerState {
    queue: Arc<Mutex<ReignQueue>>,
}

impl ReignServerState {
    pub fn new(queue: Arc<Mutex<ReignQueue>>) -> Self {
        Self { queue }
    }

    pub fn standalone() -> Self {
        Self::new(Arc::new(Mutex::new(ReignQueue::default())))
    }

    pub fn queue(&self) -> Arc<Mutex<ReignQueue>> {
        Arc::clone(&self.queue)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReignCommandRequest {
    pub mode: ReignMode,
    pub command: ReignCommand,
    #[serde(default = "default_priority")]
    pub priority: f32,
    pub ttl_ms: Option<TimeMs>,
    pub note: Option<String>,
    pub source: Option<ReignSource>,
}

pub fn reign_router(state: ReignServerState) -> Router {
    Router::new()
        .route("/reign", get(reign_page))
        .route("/reign/command", post(post_reign_command))
        .route("/reign/state", get(get_reign_state))
        .route("/reign/clear", post(post_reign_clear))
        .with_state(state)
}

#[derive(Clone, Debug, Default)]
pub struct LiveViewState {
    latest: Arc<Mutex<Option<WorldSnapshot>>>,
    scene_metadata: Arc<Mutex<Option<LiveSceneMetadata>>>,
    session: Arc<Mutex<Option<SceneSession>>>,
}

impl LiveViewState {
    pub fn new() -> Self {
        Self::default()
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
    pub t_ms: TimeMs,
    pub body: SceneBody,
    pub range: SceneRange,
    pub eye: Option<SceneEye>,
    pub kinect: SceneKinect,
    pub audio: Option<SceneAudio>,
    pub objects: Vec<SceneObject>,
    pub arena: Option<SceneArena>,
    pub action: SceneAction,
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
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SceneKinect {
    pub points: Vec<ScenePoint>,
    pub skeletons: Vec<KinectSkeletonSense>,
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
    pub safety_override: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SceneStuck {
    pub active: bool,
    pub class: Option<String>,
    pub stuck_ticks: usize,
    pub duration_ms: u64,
    pub recovery_phase: Option<String>,
    pub turn_direction: Option<String>,
    pub event_started: bool,
    pub recovered: bool,
    pub dead_battery: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SceneMind {
    pub combobulation: Option<String>,
    pub surprise: Option<f32>,
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
        .route("/view", get(live_view_page))
        .route("/view/snapshot", get(get_live_snapshot))
        .route("/view/scene", get(get_live_scene))
        .route("/view/3d", get(live_view_3d_page))
        .route("/view/capture-scene", get(get_capture_scene))
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
    Ok(Json(snapshot_to_scene(
        &snapshot,
        state.scene_metadata().as_ref(),
        state.session(),
    )))
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
        });
    let mut scene = snapshot_to_scene(&record.snapshot, metadata.as_ref(), None);
    scene.t_ms = record.t_ms;
    if let Some(pointcloud) = &record.assets.pointcloud {
        scene.warnings.push(format!(
            "capture point cloud asset available at {pointcloud}; PLY loading is planned"
        ));
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

pub fn snapshot_to_scene(
    snapshot: &WorldSnapshot,
    metadata: Option<&LiveSceneMetadata>,
    session: Option<SceneSession>,
) -> LiveSceneResponse {
    let mut warnings = Vec::new();
    let body = &snapshot.body;
    let eye = match snapshot.eye_frame.as_ref().map(scene_eye_from_frame) {
        Some((eye, frame_warnings)) => {
            warnings.extend(frame_warnings);
            Some(eye)
        }
        None => {
            warnings.push("no eye frame stream".to_string());
            None
        }
    };
    let kinect = scene_kinect_from_snapshot(snapshot, &mut warnings);
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
    LiveSceneResponse {
        schema_version: 1,
        session,
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
        action: SceneAction::default(),
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
    SceneStuck {
        active,
        class: if active || corner_trap {
            Some(if corner_trap { "corner-trap" } else { "stuck" }.to_string())
        } else {
            None
        },
        stuck_ticks: values.get(2).copied().unwrap_or(0.0).max(0.0) as usize,
        duration_ms: values.get(3).copied().unwrap_or(0.0).max(0.0) as u64,
        recovery_phase: match phase_code {
            1 => Some("stop".to_string()),
            2 => Some("reverse".to_string()),
            3 => Some("turn-away".to_string()),
            _ => None,
        },
        turn_direction: match values.get(5).copied().unwrap_or(0.0) {
            value if value < 0.0 => Some("right".to_string()),
            value if value > 0.0 => Some("left".to_string()),
            _ => None,
        },
        event_started: values.get(6).copied().unwrap_or(0.0) > 0.0,
        recovered: values.get(7).copied().unwrap_or(0.0) > 0.0,
        dead_battery: values.get(8).copied().unwrap_or(0.0) > 0.0,
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

fn scene_eye_from_frame(frame: &netherwick_sensors::EyeFrame) -> (SceneEye, Vec<String>) {
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
    (
        SceneEye {
            width: frame.width,
            height: frame.height,
            format: format!("{:?}", frame.format),
            data_url,
            mean_luma: stats.mean_luma,
            non_background_ratio: stats.non_background_ratio,
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
        EyeFrameFormat::Rgb8 | EyeFrameFormat::Bgr8 | EyeFrameFormat::Gray8 => {
            let expected_len = match frame.format {
                EyeFrameFormat::Gray8 => frame.width as usize * frame.height as usize,
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

fn scene_kinect_from_snapshot(snapshot: &WorldSnapshot, warnings: &mut Vec<String>) -> SceneKinect {
    let points = depth_points(&snapshot.kinect.depth_m);
    if points.is_empty() {
        warnings.push("no point cloud stream".to_string());
    }
    SceneKinect {
        points,
        skeletons: snapshot.kinect.skeletons.clone(),
    }
}

fn depth_points(depth_m: &[f32]) -> Vec<ScenePoint> {
    const MAX_POINTS: usize = 2_000;
    if depth_m.is_empty() {
        return Vec::new();
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
            let z = depth.clamp(0.0, 8.0);
            let shade = ((1.0 - (z / 8.0)).clamp(0.15, 1.0) * 255.0) as u8;
            Some(ScenePoint {
                x: nx * z,
                y: -ny * z + 0.25,
                z,
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
        .find(|object| object.kind == "speaker" || object.kind == "sound_source")
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
  const isMjpg = fmt === 'Mjpeg' || (typeof fmt === 'object' && fmt.Mjpeg !== undefined) || (typeof fmt === 'object' && JSON.stringify(fmt).includes('MJPG'));
  if(isRgb){
    if(canvas.width !== frame.width || canvas.height !== frame.height){
      canvas.width = frame.width; canvas.height = frame.height;
    }
    const image = ctx.createImageData(frame.width, frame.height);
    for(let source = 0, target = 0; source < frame.bytes.length; source += 3, target += 4){
      image.data[target] = frame.bytes[source];
      image.data[target + 1] = frame.bytes[source + 1];
      image.data[target + 2] = frame.bytes[source + 2];
      image.data[target + 3] = 255;
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
    const response = await fetch('/view/snapshot', {cache:'no-store'});
    if(!response.ok) throw new Error(await response.text());
    const snapshot = await response.json();
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
      fields.eye_format.textContent = fmt_str;
      fields.eye_age.textContent = `${snapshot.t_ms - snapshot.eye_frame.captured_at_ms} ms`;
    }else{
      fields.eye_format.textContent = '-';
      fields.eye_age.textContent = '-';
    }
    if(snapshot.ear_pcm){
      fields.ear_age.textContent = `${snapshot.t_ms - snapshot.ear_pcm.captured_at_ms} ms`;
    }else{
      fields.ear_age.textContent = '-';
    }
    drawEye(snapshot.eye_frame);
    drawBeams(snapshot.range.beams);
    status.textContent = 'live';
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
<title>Netherwick Sensorium 3D</title>
<style>
:root{color-scheme:dark;background:#0b0d10;color:#eef1f3;font:13px system-ui}
html,body,#scene{width:100%;height:100%;margin:0;overflow:hidden}
#hud{position:fixed;left:12px;top:12px;display:grid;gap:6px;min-width:240px;max-width:min(360px,calc(100vw - 24px));padding:10px;border:1px solid #2b3138;background:rgba(10,13,17,.86);backdrop-filter:blur(8px);border-radius:6px}
#hud h1{font-size:14px;margin:0}
#hud dl{display:grid;grid-template-columns:auto 1fr;gap:4px 10px;margin:0}
#hud dt{color:#aab4bd}
#hud dd{margin:0;text-align:right;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
#status{color:#ffd083}
#reign{position:fixed;right:12px;top:12px;min-width:220px;max-width:min(320px,calc(100vw - 24px));padding:10px;border:1px solid #36424d;background:rgba(11,16,22,.84);backdrop-filter:blur(8px);border-radius:6px;color:#dce8f2}
#reign strong{display:block;font-size:12px;margin-bottom:6px;color:#91d7ff}
#reign div{font-variant-numeric:tabular-nums}
#xr{position:fixed;right:12px;bottom:12px;padding:9px 12px;border:1px solid #405060;background:#15202b;color:#fff;border-radius:6px}
#xr[disabled]{opacity:.55}
#fallback{position:fixed;left:12px;bottom:12px;color:#aab4bd;max-width:min(520px,calc(100vw - 24px))}
canvas{display:block}
</style>
<div id="scene"></div>
<aside id="hud">
  <h1>Sensorium 3D</h1>
  <div id="status">connecting...</div>
  <dl>
    <dt>mode</dt><dd id="mode">-</dd>
    <dt>scenario</dt><dd id="scenario">-</dd>
    <dt>seed</dt><dd id="seed">-</dd>
    <dt>tick</dt><dd id="tick">-</dd>
    <dt>t</dt><dd id="t">-</dd>
    <dt>pose</dt><dd id="pose">-</dd>
    <dt>battery</dt><dd id="battery">-</dd>
    <dt>stuck</dt><dd id="stuck">-</dd>
    <dt>dead battery</dt><dd id="dead_battery">-</dd>
    <dt>recovery</dt><dd id="recovery_mode">-</dd>
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
<aside id="reign">
  <strong>XR reigns</strong>
  <div id="reign-state">enter VR to enable controller reigns</div>
</aside>
<button id="xr" disabled>VR unavailable</button>
<div id="fallback">Desktop drag rotates, wheel zooms, right-drag pans. In VR, thumbstick steers, squeeze stops, A/B dock or explore.</div>
<script type="module">
import * as THREE from '/static/vendor/three/three.module.js';
import { OrbitControls } from '/static/vendor/three/OrbitControls.js';

const root = document.getElementById('scene');
const statusEl = document.getElementById('status');
const xrButton = document.getElementById('xr');
const reignState = document.getElementById('reign-state');
const fields = Object.fromEntries(['mode','scenario','seed','tick','t','pose','battery','stuck','dead_battery','recovery_mode','stuck_ticks','nearest','eye','points','audio','mind','scheme','secure','webxr'].map(id => [id, document.getElementById(id)]));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d10);
const camera = new THREE.PerspectiveCamera(62, innerWidth / innerHeight, 0.05, 80);
camera.position.set(3.8, 4.0, 5.2);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.xr.enabled = true;
root.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xeef6ff, 0x253040, 1.8));
const sun = new THREE.DirectionalLight(0xffffff, 1.8);
sun.position.set(3, 6, 2);
scene.add(sun);

// Coordinate convention: Netherwick sim/world x maps to Three.js x, sim/world y maps to Three.js z,
// and Three.js y is up. heading_rad rotates around the y axis.
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(10, 10, 10, 10),
  new THREE.MeshStandardMaterial({color:0x171d22, roughness:.8, metalness:0})
);
ground.rotation.x = -Math.PI / 2;
scene.add(ground);
scene.add(new THREE.GridHelper(10, 20, 0x35414d, 0x24303a));

const robot = new THREE.Group();
const bodyMesh = new THREE.Mesh(
  new THREE.CylinderGeometry(.18, .18, .18, 32),
  new THREE.MeshStandardMaterial({color:0x8bd3ff, roughness:.55})
);
bodyMesh.position.y = .09;
robot.add(bodyMesh);
const heading = new THREE.ArrowHelper(new THREE.Vector3(0,0,-1), new THREE.Vector3(0,.22,0), .55, 0xffcf66, .16, .08);
robot.add(heading);
scene.add(robot);

const beams = new THREE.Group();
scene.add(beams);
const objects = new THREE.Group();
scene.add(objects);
const controller0 = renderer.xr.getController(0);
const controller1 = renderer.xr.getController(1);
scene.add(controller0);
scene.add(controller1);

const eyeCanvas = document.createElement('canvas');
eyeCanvas.width = 2; eyeCanvas.height = 2;
const eyeTexture = new THREE.CanvasTexture(eyeCanvas);
const eyePanel = new THREE.Mesh(
  new THREE.PlaneGeometry(.96, .72),
  new THREE.MeshBasicMaterial({map: eyeTexture, side: THREE.DoubleSide})
);
eyePanel.position.set(0, .65, -.78);
robot.add(eyePanel);
const frustum = new THREE.LineSegments(
  new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0,.46,-.18), new THREE.Vector3(-.48,.28,-.78),
    new THREE.Vector3(0,.46,-.18), new THREE.Vector3(.48,.28,-.78),
    new THREE.Vector3(0,.46,-.18), new THREE.Vector3(-.48,1.02,-.78),
    new THREE.Vector3(0,.46,-.18), new THREE.Vector3(.48,1.02,-.78)
  ]),
  new THREE.LineBasicMaterial({color:0x90a4b8})
);
robot.add(frustum);

let pointCloud = null;
let lastScene = null;
let xrSession = null;
let lastReignKey = '';
let lastReignSentAt = 0;
let lastReignText = 'idle';
function fmt(v, d=2){ return Number.isFinite(v) ? v.toFixed(d) : '-'; }
function world(x, y, up=0){ return new THREE.Vector3(x, up, y); }
function clear(group){ while(group.children.length) group.remove(group.children.pop()); }
function xrReason(message){
  xrButton.textContent = 'VR unavailable';
  xrButton.disabled = true;
  fields.webxr.textContent = message;
  return message;
}
function materialFor(kind, color){
  const hex = color ? (color[0]<<16) | (color[1]<<8) | color[2] : ({charger:0x45d483, person:0xe8c08c, speaker:0x778cff, obstacle:0xd67666}[kind] || 0xb0b8c0);
  return new THREE.MeshStandardMaterial({color:hex, roughness:.7});
}
function renderObjects(scenePacket){
  clear(objects);
  const arena = scenePacket.arena;
  if(arena){
    ground.scale.set(arena.width_m / 10, arena.height_m / 10, 1);
    ground.position.set(arena.width_m / 2, 0, arena.height_m / 2);
    controls.target.set(arena.width_m / 2, 0, arena.height_m / 2);
  }
  for(const item of scenePacket.objects || []){
    let mesh;
    if(item.kind === 'person'){
      mesh = new THREE.Mesh(new THREE.CapsuleGeometry(item.radius_m, .8, 8, 16), materialFor(item.kind, item.color_rgb));
      mesh.position.copy(world(item.x_m, item.y_m, .55));
    }else if(item.kind === 'speaker'){
      mesh = new THREE.Mesh(new THREE.ConeGeometry(item.radius_m * 1.8, .35, 24), materialFor(item.kind, item.color_rgb));
      mesh.rotation.x = Math.PI / 2;
      mesh.position.copy(world(item.x_m, item.y_m, .25));
    }else{
      mesh = new THREE.Mesh(new THREE.CylinderGeometry(item.radius_m, item.radius_m, item.kind === 'charger' ? .08 : .45, 28), materialFor(item.kind, item.color_rgb));
      mesh.position.copy(world(item.x_m, item.y_m, item.kind === 'charger' ? .04 : .225));
    }
    mesh.userData.id = item.id;
    objects.add(mesh);
  }
}
function renderBeams(packet){
  clear(beams);
  const b = packet.body;
  for(const beam of packet.range?.beams || []){
    const angle = b.heading_rad + beam.angle_rad;
    const start = world(b.x_m, b.y_m, .12);
    const end = world(b.x_m + Math.cos(angle) * beam.distance_m, b.y_m + Math.sin(angle) * beam.distance_m, .12);
    const geo = new THREE.BufferGeometry().setFromPoints([start, end]);
    const mat = new THREE.LineBasicMaterial({color:beam.hit ? 0xff6e5c : 0x62d394, transparent:true, opacity:beam.hit ? .95 : .55});
    beams.add(new THREE.Line(geo, mat));
    if(beam.hit){
      const dot = new THREE.Mesh(new THREE.SphereGeometry(.055, 12, 12), new THREE.MeshBasicMaterial({color:0xff6e5c}));
      dot.position.copy(end);
      beams.add(dot);
    }
  }
}
function renderPoints(points){
  if(pointCloud){
    scene.remove(pointCloud);
    pointCloud.geometry.dispose();
    pointCloud.material.dispose();
    pointCloud = null;
  }
  if(!points || !points.length) return;
  const positions = new Float32Array(points.length * 3);
  const colors = new Float32Array(points.length * 3);
  points.forEach((p, i) => {
    positions[i*3] = robot.position.x + p.x;
    positions[i*3+1] = p.y;
    positions[i*3+2] = robot.position.z - p.z;
    colors[i*3] = p.r / 255; colors[i*3+1] = p.g / 255; colors[i*3+2] = p.b / 255;
  });
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  pointCloud = new THREE.Points(geo, new THREE.PointsMaterial({size:.035, vertexColors:true}));
  scene.add(pointCloud);
}
function renderEye(eye){
  if(!eye?.data_url) return;
  const img = new Image();
  img.onload = () => {
    eyeCanvas.width = img.width; eyeCanvas.height = img.height;
    eyeCanvas.getContext('2d').drawImage(img, 0, 0);
    eyeTexture.needsUpdate = true;
  };
  img.src = eye.data_url;
}
function updateScene(packet){
  lastScene = packet;
  robot.position.copy(world(packet.body.x_m, packet.body.y_m, 0));
  robot.rotation.y = -packet.body.heading_rad - Math.PI / 2;
  renderObjects(packet);
  renderBeams(packet);
  renderPoints(packet.kinect?.points || []);
  renderEye(packet.eye);
  const session = packet.session || {};
  fields.mode.textContent = session.mode || '-';
  fields.scenario.textContent = session.scenario || '-';
  fields.seed.textContent = session.seed == null ? '-' : String(session.seed);
  fields.tick.textContent = session.tick_ms == null ? '-' : `${session.tick_ms} ms`;
  fields.t.textContent = `${packet.t_ms} ms`;
  fields.pose.textContent = `${fmt(packet.body.x_m)}, ${fmt(packet.body.y_m)}, ${fmt(packet.body.heading_rad)} rad`;
  fields.battery.textContent = `${fmt(packet.body.battery_level * 100, 1)}%${packet.body.charging ? ' charging' : ''}`;
  const detail = packet.stuck_detail || {};
  fields.stuck.textContent = packet.stuck ? (detail.class || 'stuck') : (detail.recovered ? 'recovered' : 'clear');
  fields.dead_battery.textContent = packet.dead_battery ? 'yes' : 'no';
  fields.recovery_mode.textContent = packet.recovery_mode || '-';
  fields.stuck_ticks.textContent = String(packet.stuck_ticks || 0);
  fields.nearest.textContent = packet.range?.nearest_m == null ? '-' : `${fmt(packet.range.nearest_m)} m`;
  fields.eye.textContent = packet.eye ? `${packet.eye.width}x${packet.eye.height} luma ${fmt(packet.eye.mean_luma, 2)}` : '-';
  fields.points.textContent = String(packet.kinect?.points?.length || 0);
  fields.audio.textContent = packet.audio?.bearing_rad == null ? '-' : `${fmt(packet.audio.bearing_rad)} rad`;
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
async function postReign(command, ttl_ms, label, priority=.95){
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
        source:'Gamepad',
        note:'WebXR controller reign',
        command
      })
    });
    if(!res.ok) throw new Error(await res.text());
    reignState.textContent = label;
  }catch(error){
    reignState.textContent = 'reign send failed';
  }
}
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
  if(!xrSession){
    reignState.textContent = 'enter VR to enable controller reigns';
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
  const ok = await navigator.xr.isSessionSupported('immersive-vr').catch(() => false);
  if(!ok){
    xrReason('immersive-vr unsupported');
    return;
  }
  xrButton.textContent = 'Enter VR';
  xrButton.disabled = false;
  fields.webxr.textContent = 'immersive-vr supported';
  xrButton.onclick = async () => {
    const session = await navigator.xr.requestSession('immersive-vr', {optionalFeatures:['local-floor','bounded-floor']});
    xrSession = session;
    reignState.textContent = 'controller reigns ready';
    session.addEventListener('end', () => {
      xrSession = null;
      lastReignKey = '';
      lastReignText = 'idle';
      reignState.textContent = 'enter VR to enable controller reigns';
    });
    await renderer.xr.setSession(session);
  };
}
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
renderer.setAnimationLoop(() => {
  controls.update();
  pollXrReigns();
  renderer.render(scene, camera);
});
setupXr();
poll();
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
        });
        state.update_session(SceneSession {
            mode: "virtual-live".to_string(),
            scenario: Some("charger-seeking".to_string()),
            seed: Some(99),
            source: "sim".to_string(),
            tick_ms: Some(100),
        });
        let mut snapshot = WorldSnapshot::default();
        snapshot.body.odometry.x_m = 0.5;
        snapshot.body.odometry.y_m = 0.75;
        snapshot.body.odometry.heading_rad = 1.25;
        snapshot.body.battery_level = 0.82;
        snapshot.body.last_update_ms = 1234;
        snapshot.range.beams = vec![1.0, 2.0, 3.0];
        snapshot.range.nearest_m = Some(1.0);
        snapshot.extensions.push(netherwick_now::ExtensionSense {
            schema_version: 1,
            name: "sim.stuck".to_string(),
            values: vec![1.0, 1.0, 6.0, 300.0, 3.0, -1.0, 1.0, 0.0, 1.0],
        });
        snapshot.eye_frame = Some(EyeFrame {
            captured_at_ms: 1200,
            width: 1,
            height: 1,
            format: EyeFrameFormat::Rgb8,
            bytes: vec![255, 0, 0],
        });
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
        assert_eq!(scene.range.nearest_m, Some(1.0));
        assert_eq!(scene.range.beams.len(), 3);
        assert!(scene.stuck);
        assert!(scene.dead_battery);
        assert_eq!(scene.recovery_mode.as_deref(), Some("turn-away"));
        assert_eq!(scene.stuck_ticks, 6);
        assert_eq!(scene.stuck_detail.class.as_deref(), Some("corner-trap"));
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
        let scene = snapshot_to_scene(&snapshot, None, None);
        let value = serde_json::to_value(scene).unwrap();

        assert!(value["eye"].is_null());
        assert_eq!(value["dead_battery"].as_bool(), Some(true));
        assert_eq!(value["kinect"]["points"].as_array().unwrap().len(), 0);
        assert!(value["audio"].is_null());
        assert!(value["warnings"].as_array().unwrap().len() >= 3);
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
        let Html(page) = live_view_3d_page().await;
        assert!(page.contains("Sensorium 3D"));
        assert!(page.contains("/view/scene"));
        assert!(page.contains("navigator.xr"));
        assert!(page.contains("window.isSecureContext"));
        assert!(page.contains("/reign/command"));
        assert!(page.contains("source:'Gamepad'"));
        assert!(page.contains("renderer.xr.getController"));
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
