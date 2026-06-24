use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use netherwick_actions::{ReignCommand, ReignInput, ReignMode, ReignSource};
use netherwick_core::TimeMs;
use netherwick_now::ReignSense;
use netherwick_runtime::ReignQueue;
use netherwick_sensors::WorldSnapshot;
use serde::{Deserialize, Serialize};
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

pub fn live_view_router(state: LiveViewState) -> Router {
    Router::new()
        .route("/", get(live_view_page))
        .route("/now", get(get_live_now))
        .route("/view", get(live_view_page))
        .route("/view/snapshot", get(get_live_snapshot))
        .with_state(state)
}

pub async fn serve_live_view(addr: SocketAddr, state: LiveViewState) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, live_view_router(state)).await
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
  <div id="status">waiting for frames...</div>
  <dl>
    <dt>t</dt><dd id="t">-</dd>
    <dt>x</dt><dd id="x">-</dd>
    <dt>y</dt><dd id="y">-</dd>
    <dt>heading</dt><dd id="heading">-</dd>
    <dt>battery</dt><dd id="battery">-</dd>
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
}
