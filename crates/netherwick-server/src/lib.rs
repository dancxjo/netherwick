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
