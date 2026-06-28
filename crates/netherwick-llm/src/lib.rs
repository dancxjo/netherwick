use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{Local, SecondsFormat, TimeZone};
use futures_util::StreamExt;
use netherwick_actions::{
    ActionPrimitive, ApproachTarget, ChirpPattern, ExploreStyle, InspectTarget, TurnDir,
};
use netherwick_experience::{EmbodiedContext, ExperienceLatent, FuturePrediction, Impression};
use netherwick_now::{LlmSense, Now, ReignSense};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;

pub const IMAGE_CAPTION_PROMPT: &str = "The attached visual input is what I am seeing now. Describe only what you see from my viewpoint. Start from the fact that this is my own vision looking out, so first person should mean phrases like \"I see...\" or \"in front of me,\" not that visible people, faces, hands, eyes, or bodies are mine. Prefer concrete scene details over lighting or color summaries. Stay grounded in visible evidence and do not speculate beyond what can be seen. Do not interpret this as an image, screenshot, photo, frame, camera feed, metadata, data URL, or analysis; interpret it as the robot's own live view. When looking out, one does not see oneself: anyone visible is most likely someone I am looking at, not myself, unless I am clearly looking in a mirror or reflection. Describe visible people in third person, as someone in front of me.";

const SENSOR_GROUNDING_RULES: &str = "Describe the real-world scene or event, not the sensor stream. Interpret images, audio, motion, location, body state, memory-derived entries, and other sensor-derived entries as the robot's own vision, hearing, body sense, position sense, and memory sense, not as media files or external sensor artifacts. Convert raw body data into feeling-centered first-person interpretations from inside the robot: say things like \"I feel steady,\" \"I feel the floor fall away ahead,\" or \"my body feels blocked,\" instead of naming raw flags, sensor booleans, channel levels, or detector states. Do not summarize the amount, density, cadence, or mix of input modalities as if that were the situation. Repeated frames, repeated detections, image embeddings, pending audio clips, and heartbeat-like status records are usually evidence to compress or ignore, not events to report. If people are visible, do not assume any visible person is me unless the vision is clearly a mirror or reflection. If the evidence does not reveal what is happening, say that I cannot tell what is happening yet. Do not infer emotional tone or words like chaotic, intense, overwhelming, anxious, or ominous from sensor volume alone.";

const COMBOBULATOR_DISTILLATION_RULES: &str = "Distill what matters, not what the records said. Treat the entries as fragmentary, possibly contradictory, fleeting evidence about the actual situation, not as the topic to describe. Try to infer what is going on in the real world from those fragments. Sort meaning by time: occurred time first, observed time second. Consume the timeline in order; do not group by faculty or source. When related entries describe raw audio and the transcript derived from it, treat them as one real-world event. Some entries may be prior combobulation summaries looping back as impressions; use those only as provisional, possibly stale self-context, not as fresh external evidence. Do not say that you are observing a timeline, records, recordings, sensor streams, previous summaries, or a shift in conversation. Compress repeated low-level records into the real-world gist; do not enumerate ids, hashes, timestamps, edges, or detections unless they are the point.";

const LIVE_EVENT_RULES: &str = "Live events may arrive while generation is happening. Treat them as observations from outside. Do not assume a human is currently present or addressing me; there may be nobody nearby. Clock and status events help track timing, pauses, and elapsed time, but do not narrate every tick, quiet moment, or idle thought.";
const STRICT_JSON_RESPONSE_RULES: &str = "\n\nFINAL OUTPUT REQUIREMENT (MANDATORY):\nReturn exactly one JSON object and nothing else.\nDo not include markdown fences, prose, explanations, preambles, or trailing text.\nOutput must start with '{' and end with '}'.\nIf unsure, still emit the best-effort valid JSON object matching the schema.";
const COMBOBULATOR_CLUSTER_GAP_MS: u64 = 1_000;
const MILLIS_PER_SECOND: f64 = 1_000.0;
const DEFAULT_OLLAMA_TIMEOUT_MS: u64 = 300_000;

static LLM_STREAM_BUS: OnceLock<broadcast::Sender<LlmStreamEvent>> = OnceLock::new();
static LLM_STREAM_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmStreamPhase {
    Start,
    Delta,
    Done,
    Error,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LlmStreamEvent {
    pub id: u64,
    pub t_ms: u64,
    pub phase: LlmStreamPhase,
    pub purpose: String,
    pub provider: String,
    pub model: String,
    pub prompt: Option<String>,
    pub delta: Option<String>,
    pub response: Option<String>,
    pub error: Option<String>,
}

pub fn subscribe_llm_streams() -> broadcast::Receiver<LlmStreamEvent> {
    llm_stream_bus().subscribe()
}

fn llm_stream_bus() -> &'static broadcast::Sender<LlmStreamEvent> {
    LLM_STREAM_BUS.get_or_init(|| {
        let (tx, _rx) = broadcast::channel(256);
        tx
    })
}

fn next_stream_id() -> u64 {
    LLM_STREAM_ID.fetch_add(1, Ordering::Relaxed)
}

fn emit_llm_stream(event: LlmStreamEvent) {
    let _ = llm_stream_bus().send(event);
}

fn wall_now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConsciousCommand {
    pub summary: String,
    pub action: Option<ActionPrimitive>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmDecision {
    pub summary: String,
    pub critique: Option<String>,
    pub confidence: f32,
    pub action: Option<ActionPrimitive>,
    pub counterfactuals: Vec<CounterfactualAction>,
    pub memory_notes: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmTeaching {
    pub t_ms: u64,
    pub summary: String,
    pub critique: Option<String>,
    pub counterfactuals: Vec<CounterfactualAction>,
    pub memory_notes: Vec<String>,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CounterfactualAction {
    pub instead_of: Option<ActionPrimitive>,
    pub proposed: ActionPrimitive,
    pub reason: String,
    pub weight: f32,
}

impl Default for CounterfactualAction {
    fn default() -> Self {
        Self {
            instead_of: None,
            proposed: ActionPrimitive::Stop,
            reason: String::new(),
            weight: 0.0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Combobulation {
    pub summary: String,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmTickResult {
    pub sense: LlmSense,
    pub conscious_command: Option<ConsciousCommand>,
    pub decision: Option<LlmDecision>,
    pub teaching: Vec<LlmTeaching>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmBrief {
    pub now_ms: u64,
    pub senses: Vec<String>,
    pub reign: ReignSense,
}

#[async_trait]
pub trait LlmAgent: Send {
    async fn combobulate(
        &mut self,
        now: &Now,
        impressions: &[Impression],
        embodied: Option<&EmbodiedContext>,
        z: &ExperienceLatent,
        futures: &[FuturePrediction],
        recall_summary: &str,
    ) -> Result<Option<Combobulation>>;

    async fn maybe_tick(
        &mut self,
        now: &Now,
        embodied: Option<&EmbodiedContext>,
        z: &ExperienceLatent,
        futures: &[FuturePrediction],
        recall_summary: &str,
        awareness_summary: Option<&str>,
    ) -> Result<LlmTickResult>;
}

#[derive(Default)]
pub struct NoopLlmAgent;

#[async_trait]
impl LlmAgent for NoopLlmAgent {
    async fn combobulate(
        &mut self,
        _now: &Now,
        _impressions: &[Impression],
        _embodied: Option<&EmbodiedContext>,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
    ) -> Result<Option<Combobulation>> {
        Ok(None)
    }

    async fn maybe_tick(
        &mut self,
        _now: &Now,
        _embodied: Option<&EmbodiedContext>,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
        _awareness_summary: Option<&str>,
    ) -> Result<LlmTickResult> {
        Ok(LlmTickResult::default())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmProvider {
    Disabled,
    Ollama,
}

impl Default for LlmProvider {
    fn default() -> Self {
        Self::Disabled
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    pub provider: LlmProvider,
    pub allow_commands: bool,
    pub allow_teaching: bool,
    pub endpoint: String,
    pub agent_model: String,
    pub combobulator_model: Option<String>,
    pub temperature: f32,
    pub timeout_ms: u64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        let endpoint = std::env::var("NETHERWICK_LLM_ENDPOINT")
            .or_else(|_| std::env::var("OLLAMA_HOST"))
            .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
        let agent_model = std::env::var("NETHERWICK_LLM_MODEL")
            .or_else(|_| std::env::var("OLLAMA_MODEL"))
            .unwrap_or_else(|_| "llama3.2".to_string());
        Self {
            provider: LlmProvider::Ollama,
            allow_commands: true,
            allow_teaching: true,
            endpoint,
            agent_model,
            combobulator_model: None,
            temperature: 0.2,
            timeout_ms: DEFAULT_OLLAMA_TIMEOUT_MS,
        }
    }
}

impl LlmConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let text = fs::read_to_string(path.as_ref())
            .with_context(|| format!("failed to read {}", path.as_ref().display()))?;
        let mut config: Self = toml::from_str(&text).context("failed to parse llm config")?;
        if config.combobulator_model.is_none() {
            config.combobulator_model = Some(config.agent_model.clone());
        }
        Ok(config)
    }
}

pub fn summarized_senses(now: &Now) -> Vec<String> {
    let mut lines = Vec::new();
    lines.push(format!(
        "Battery level is {:.0}%.",
        now.body.battery_level * 100.0
    ));
    if now.body.charging {
        lines.push("I am charging.".to_string());
    }
    if now.body.flags.bump_left || now.body.flags.bump_right {
        lines.push("My body feels blocked by contact.".to_string());
    }
    if now.body.flags.cliff_left
        || now.body.flags.cliff_front_left
        || now.body.flags.cliff_front_right
        || now.body.flags.cliff_right
        || now.body.cliff_sensors.max() >= 0.5
    {
        lines.push("I feel the floor fall away near me.".to_string());
    } else if now.body.cliff_sensors.max() > 0.0 {
        lines.push("I feel steady, with only a faint edge-sense under me.".to_string());
    }
    if now.body.flags.wheel_drop {
        lines.push("My wheel is dropped.".to_string());
    }
    if now.body.flags.wall {
        lines.push("My wall sensor is active.".to_string());
    }
    if now.body.flags.virtual_wall {
        lines.push("I detect a virtual wall.".to_string());
    }
    if let Some(transcript) = &now.ear.transcript {
        let transcript = transcript.trim();
        if !transcript.is_empty() {
            lines.push(format!("I hear: {transcript}"));
        }
    }
    if let Some(nearest_m) = now.range.nearest_m {
        lines.push(format!("Nearest obstacle is {:.2} meters away.", nearest_m));
    }
    if !now.kinect.ir.is_empty() {
        let count = now.kinect.ir.len();
        let mean = now.kinect.ir.iter().copied().sum::<f32>() / count as f32;
        let max = now
            .kinect
            .ir
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let bright = now.kinect.ir.iter().filter(|value| **value >= 0.7).count();
        lines.push(format!(
            "Kinect IR has {count} samples, mean {:.2}, max {:.2}, bright fraction {:.2}.",
            mean,
            max,
            bright as f32 / count as f32
        ));
    }
    if !now.predictions.expected_events.is_empty() {
        lines.push(format!(
            "I expect: {}.",
            now.predictions.expected_events.join(", ")
        ));
    }
    if now.surprise.total > 0.0 {
        lines.push(format!("Surprise is {:.2}.", now.surprise.total));
    }
    if let Some(goal) = &now.self_sense.active_goal {
        lines.push(format!("My active goal is {goal}."));
    }
    if let Some(mode) = &now.self_sense.mode {
        lines.push(format!("My mode is {mode}."));
    }
    if let Some(input) = &now.reign.latest {
        lines.push(format!("Remote control active: {:?}.", input.mode));
        lines.push(format!(
            "Latest remote command: {}.",
            summarize_reign_command(input)
        ));
        if let Some(action) = input.command.to_action() {
            if let Ok(json) = serde_json::to_string(&action_spec_json(&action)) {
                lines.push(format!("Matching executable remote action JSON: {json}."));
            }
        }
        if let Some(note) = &input.note {
            lines.push(format!("Remote note: {note}"));
        }
    }
    if now.memory.similar_situation_count > 0 {
        lines.push(format!(
            "Memory says this feels like {} similar situations.",
            now.memory.similar_situation_count
        ));
    }
    lines
}

pub enum ConfiguredLlmAgent {
    Disabled(NoopLlmAgent),
    Ollama(OllamaLlmAgent),
}

impl ConfiguredLlmAgent {
    pub fn from_config(config: LlmConfig) -> Result<Self> {
        Ok(match config.provider {
            LlmProvider::Disabled => Self::Disabled(NoopLlmAgent),
            LlmProvider::Ollama => Self::Ollama(OllamaLlmAgent::new(config)?),
        })
    }
}

#[async_trait]
impl LlmAgent for ConfiguredLlmAgent {
    async fn combobulate(
        &mut self,
        now: &Now,
        impressions: &[Impression],
        embodied: Option<&EmbodiedContext>,
        z: &ExperienceLatent,
        futures: &[FuturePrediction],
        recall_summary: &str,
    ) -> Result<Option<Combobulation>> {
        match self {
            Self::Disabled(agent) => {
                agent
                    .combobulate(now, impressions, embodied, z, futures, recall_summary)
                    .await
            }
            Self::Ollama(agent) => {
                agent
                    .combobulate(now, impressions, embodied, z, futures, recall_summary)
                    .await
            }
        }
    }

    async fn maybe_tick(
        &mut self,
        now: &Now,
        embodied: Option<&EmbodiedContext>,
        z: &ExperienceLatent,
        futures: &[FuturePrediction],
        recall_summary: &str,
        awareness_summary: Option<&str>,
    ) -> Result<LlmTickResult> {
        match self {
            Self::Disabled(agent) => {
                agent
                    .maybe_tick(now, embodied, z, futures, recall_summary, awareness_summary)
                    .await
            }
            Self::Ollama(agent) => {
                agent
                    .maybe_tick(now, embodied, z, futures, recall_summary, awareness_summary)
                    .await
            }
        }
    }
}

pub struct OllamaLlmAgent {
    config: LlmConfig,
    client: Client,
}

impl OllamaLlmAgent {
    pub fn new(config: LlmConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .build()
            .context("failed to build ollama http client")?;
        Ok(Self { config, client })
    }

    async fn generate_json<T>(&self, purpose: &str, model: &str, prompt: String) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let prompt = append_strict_json_suffix(prompt);
        let stream_id = next_stream_id();
        emit_llm_stream(LlmStreamEvent {
            id: stream_id,
            t_ms: wall_now_ms(),
            phase: LlmStreamPhase::Start,
            purpose: purpose.to_string(),
            provider: "ollama".to_string(),
            model: model.to_string(),
            prompt: Some(prompt.clone()),
            delta: None,
            response: None,
            error: None,
        });
        let request = OllamaGenerateRequest {
            model,
            prompt,
            stream: true,
            options: OllamaGenerateOptions {
                temperature: self.config.temperature,
            },
        };
        let response = self
            .client
            .post(format!(
                "{}/api/generate",
                self.config.endpoint.trim_end_matches('/')
            ))
            .json(&request)
            .send()
            .await
            .map_err(|error| {
                emit_llm_stream_error(
                    stream_id,
                    purpose,
                    model,
                    format!("failed to reach ollama: {error}"),
                );
                error
            })
            .context("failed to reach ollama")?
            .error_for_status()
            .map_err(|error| {
                emit_llm_stream_error(
                    stream_id,
                    purpose,
                    model,
                    format!("ollama returned an error: {error}"),
                );
                error
            })
            .context("ollama returned an error")?;
        let mut body = String::new();
        let mut pending = String::new();
        let mut chunks = response.bytes_stream();
        while let Some(chunk) = chunks.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    emit_llm_stream_error(
                        stream_id,
                        purpose,
                        model,
                        format!("failed while reading ollama stream: {error}"),
                    );
                    return Err(error).context("failed while reading ollama stream");
                }
            };
            pending.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(newline) = pending.find('\n') {
                let line = pending[..newline].trim().to_string();
                pending = pending[newline + 1..].to_string();
                if !line.is_empty() {
                    handle_ollama_stream_line(stream_id, purpose, model, &line, &mut body)?;
                }
            }
        }
        let line = pending.trim();
        if !line.is_empty() {
            handle_ollama_stream_line(stream_id, purpose, model, line, &mut body)?;
        }
        emit_llm_stream(LlmStreamEvent {
            id: stream_id,
            t_ms: wall_now_ms(),
            phase: LlmStreamPhase::Done,
            purpose: purpose.to_string(),
            provider: "ollama".to_string(),
            model: model.to_string(),
            prompt: None,
            delta: None,
            response: Some(body.clone()),
            error: None,
        });
        let json = extract_json_object(&body)
            .with_context(|| format!("ollama returned non-json content: {}", body))?;
        serde_json::from_str(&json).context("failed to parse llm json payload")
    }
}

fn append_strict_json_suffix(mut prompt: String) -> String {
    prompt.push_str(STRICT_JSON_RESPONSE_RULES);
    prompt
}

fn handle_ollama_stream_line(
    stream_id: u64,
    purpose: &str,
    model: &str,
    line: &str,
    body: &mut String,
) -> Result<()> {
    let chunk: OllamaGenerateResponse = serde_json::from_str(line).with_context(|| {
        emit_llm_stream_error(
            stream_id,
            purpose,
            model,
            format!("failed to decode ollama stream line: {line}"),
        );
        "failed to decode ollama stream line"
    })?;
    if !chunk.response.is_empty() {
        body.push_str(&chunk.response);
        emit_llm_stream(LlmStreamEvent {
            id: stream_id,
            t_ms: wall_now_ms(),
            phase: LlmStreamPhase::Delta,
            purpose: purpose.to_string(),
            provider: "ollama".to_string(),
            model: model.to_string(),
            prompt: None,
            delta: Some(chunk.response),
            response: None,
            error: None,
        });
    }
    Ok(())
}

fn emit_llm_stream_error(stream_id: u64, purpose: &str, model: &str, error: String) {
    emit_llm_stream(LlmStreamEvent {
        id: stream_id,
        t_ms: wall_now_ms(),
        phase: LlmStreamPhase::Error,
        purpose: purpose.to_string(),
        provider: "ollama".to_string(),
        model: model.to_string(),
        prompt: None,
        delta: None,
        response: None,
        error: Some(error),
    });
}

#[async_trait]
impl LlmAgent for OllamaLlmAgent {
    async fn combobulate(
        &mut self,
        now: &Now,
        impressions: &[Impression],
        embodied: Option<&EmbodiedContext>,
        z: &ExperienceLatent,
        futures: &[FuturePrediction],
        recall_summary: &str,
    ) -> Result<Option<Combobulation>> {
        let prompt =
            build_combobulator_prompt(now, impressions, embodied, z, futures, recall_summary);
        let model = self
            .config
            .combobulator_model
            .as_deref()
            .unwrap_or(self.config.agent_model.as_str());
        match self
            .generate_json::<CombobulatorReply>("combobulator", model, prompt)
            .await
        {
            Ok(reply) => Ok(Some(Combobulation {
                summary: reply.summary.trim().to_string(),
                confidence: reply.confidence.clamp(0.0, 1.0),
            })),
            Err(_) => Ok(Some(heuristic_combobulation(now, recall_summary))),
        }
    }

    async fn maybe_tick(
        &mut self,
        now: &Now,
        embodied: Option<&EmbodiedContext>,
        z: &ExperienceLatent,
        futures: &[FuturePrediction],
        recall_summary: &str,
        awareness_summary: Option<&str>,
    ) -> Result<LlmTickResult> {
        if !self.config.allow_commands && !self.config.allow_teaching {
            return Ok(LlmTickResult::default());
        }

        let prompt = build_agent_prompt(
            now,
            embodied,
            z,
            futures,
            recall_summary,
            awareness_summary,
            &self.config,
        );
        let reply = match self
            .generate_json::<AgentReply>("agent", &self.config.agent_model, prompt)
            .await
        {
            Ok(reply) => reply,
            Err(_) => return Ok(LlmTickResult::default()),
        };

        let model_action = if self.config.allow_commands {
            reply.action.and_then(parse_action_spec)
        } else {
            None
        };
        let reign_action = if self.config.allow_commands {
            active_reign_action(now)
        } else {
            None
        };
        let command_action = reign_action.clone().or_else(|| model_action.clone());
        let critique = reply
            .critique
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let summary = if reply.summary.trim().is_empty() {
            reign_command_summary(now).unwrap_or_default()
        } else {
            reply.summary.trim().to_string()
        };
        let confidence = reply.confidence.clamp(0.0, 1.0);
        let counterfactuals = reply
            .counterfactuals
            .into_iter()
            .filter_map(parse_counterfactual_spec)
            .collect::<Vec<_>>();
        let decision = LlmDecision {
            summary: summary.clone(),
            critique: critique.clone(),
            confidence,
            action: model_action.clone(),
            counterfactuals,
            memory_notes: reply.memory_notes,
        };

        let conscious_command =
            if self.config.allow_commands && (command_action.is_some() || !summary.is_empty()) {
                Some(ConsciousCommand {
                    summary: summary.clone(),
                    action: command_action.clone(),
                })
            } else {
                None
            };

        let teaching = if self.config.allow_teaching
            && (critique.is_some()
                || !decision.memory_notes.is_empty()
                || !decision.counterfactuals.is_empty())
        {
            vec![LlmTeaching {
                t_ms: now.t_ms,
                summary: if summary.is_empty() {
                    "LLM reflection".to_string()
                } else {
                    summary.clone()
                },
                critique: critique.clone(),
                counterfactuals: decision.counterfactuals.clone(),
                memory_notes: decision.memory_notes.clone(),
                confidence,
            }]
        } else {
            Vec::new()
        };

        Ok(LlmTickResult {
            sense: LlmSense {
                schema_version: 1,
                command_summary: conscious_command.as_ref().map(|cmd| cmd.summary.clone()),
                critique,
                confidence,
            },
            conscious_command,
            decision: Some(decision),
            teaching,
        })
    }
}

#[derive(Serialize)]
struct OllamaGenerateRequest<'a> {
    model: &'a str,
    prompt: String,
    stream: bool,
    options: OllamaGenerateOptions,
}

#[derive(Serialize)]
struct OllamaGenerateOptions {
    temperature: f32,
}

#[derive(Deserialize)]
struct OllamaGenerateResponse {
    response: String,
}

#[derive(Deserialize)]
struct CombobulatorReply {
    summary: String,
    #[serde(default)]
    confidence: f32,
}

#[derive(Deserialize)]
struct AgentReply {
    #[serde(default)]
    summary: String,
    #[serde(default)]
    critique: Option<String>,
    #[serde(default)]
    confidence: f32,
    #[serde(default)]
    action: Option<ActionSpec>,
    #[serde(default)]
    counterfactuals: Vec<CounterfactualSpec>,
    #[serde(default)]
    memory_notes: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct ActionSpec {
    kind: String,
    #[serde(default)]
    direction: Option<String>,
    #[serde(default)]
    intensity: Option<f32>,
    #[serde(default)]
    duration_ms: Option<u64>,
    #[serde(default)]
    target: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    style: Option<String>,
    #[serde(default)]
    pattern: Option<String>,
}

#[derive(Deserialize)]
struct CounterfactualSpec {
    #[serde(default)]
    instead_of: Option<ActionSpec>,
    proposed: ActionSpec,
    reason: String,
    #[serde(default)]
    weight: f32,
}

fn build_combobulator_prompt(
    now: &Now,
    impressions: &[Impression],
    embodied: Option<&EmbodiedContext>,
    z: &ExperienceLatent,
    futures: &[FuturePrediction],
    recall_summary: &str,
) -> String {
    let timeline = render_combobulator_timeline(impressions);
    let embodied = render_embodied_context(embodied);
    let futures = summarize_futures(futures);
    format!(
        "You are the combobulator for an embodied robot.\n\
Given recent impressions and predicted futures in timeline order, distill what appears to be happening right now.\n\
You run continuously over the recent timeline; each pass tries to understand what is going on right now. Write from first-person lived experience from the robot's point of view, using I/my/me naturally.\n\
This summary will be used as a basic understanding of the current situation for a system that may need to act immediately. Think of it as telling someone with amnesia as quickly as possible, but as thoroughly as needed for them to act reasonably.\n\
Use only the evidence below. The impressions are first-person present-tense embodied claims such as \"I see...\", \"I hear...\", or \"My body...\"; preserve that lived point of view. Prefer concrete body facts, nearby people or speech, visible scene details, memory, safety, and immediate context. Explain what appears to be happening right now, not a redundant list of events.\n\
{SENSOR_GROUNDING_RULES}\n\
{COMBOBULATOR_DISTILLATION_RULES}\n\
{LIVE_EVENT_RULES}\n\
Return JSON only with this schema:\n\
{{\"summary\":\"...\",\"confidence\":0.0}}\n\n\
CONTEXT FRAME\n\
WHO\n\
- embodied robot\n\
WHAT\n\
- current awareness synthesis from impressions\n\
WHERE\n\
- current body location if sensors or memory reveal it; otherwise unknown\n\
WHEN\n\
- now at {} ms\n\
WHY\n\
- produce a compact awareness statement useful to the next action decision\n\
HOW\n\
- distill text impressions produced from body, hearing, vision, range, memory, predictions, surprise, and remote controls\n\n\
Latent confidence: {:.2}\n\
Latent prediction error: {:.2}\n\
Recall summary: {}\n\
Current embodied experience:\n{}\n\
Timeline evidence:\n{}\n\
Predicted futures:\n{}\n",
        now.t_ms, z.confidence, z.prediction_error, recall_summary, embodied, timeline, futures
    )
}

fn build_agent_prompt(
    now: &Now,
    embodied: Option<&EmbodiedContext>,
    z: &ExperienceLatent,
    futures: &[FuturePrediction],
    recall_summary: &str,
    awareness_summary: Option<&str>,
    config: &LlmConfig,
) -> String {
    let senses = summarized_senses(now)
        .into_iter()
        .map(|line| format!("- {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    let embodied = render_embodied_context(embodied);
    let futures = summarize_futures(futures);
    format!(
        "You are the conscious LLM layer for an embodied robot.\n\
When commands are enabled, choose a high-level action primitive whenever movement, speech, inspection, docking, or stopping is appropriate.\n\
You are in autonomous discovery mode: safely explore, inspect uncertain or interesting stimuli, and prefer active information-gathering when there is no higher-priority goal or danger.\n\
The action field is an executable command candidate for the robot body, not only a suggestion or note.\n\
Never output raw motor control such as wheel speeds, PWM values, serial bytes, or velocity arrays.\n\
Treat Reign controls as present-tense command input. If a Reign command is active and safe, set action to the matching allowed action; if you choose something else, explain why in critique.\n\
{LIVE_EVENT_RULES}\n\
Allowed action kinds: stop, go, turn, inspect, approach, dock, explore, speak, chirp.\n\
If commands are disabled, leave action null. Commands enabled: {}. Teaching enabled: {}.\n\
Return JSON only with this schema:\n\
{{\n\
  \"summary\":\"short first-person command or reflection\",\n\
  \"critique\":\"optional critique\",\n\
  \"confidence\":0.0,\n\
  \"action\":{{\"kind\":\"dock\"}} or null,\n\
  \"counterfactuals\":[{{\"instead_of\":null,\"proposed\":{{\"kind\":\"turn\",\"direction\":\"left\",\"intensity\":0.4,\"duration_ms\":1000}},\"reason\":\"...\",\"weight\":0.5}}],\n\
  \"memory_notes\":[\"...\"]\n\
}}\n\n\
Current time: {} ms\n\
Awareness summary: {}\n\
Current embodied experience:\n{}\n\
Recall summary: {}\n\
Battery: {:.2}\n\
Surprise: {:.2}\n\
Latent confidence: {:.2}\n\
Predicted futures:\n{}\n\
Summarized senses:\n{}\n",
        config.allow_commands,
        config.allow_teaching,
        now.t_ms,
        awareness_summary.unwrap_or("none"),
        embodied,
        recall_summary,
        now.body.battery_level,
        now.surprise.total,
        z.confidence,
        futures,
        senses
    )
}

fn render_embodied_context(context: Option<&EmbodiedContext>) -> String {
    let Some(context) = context else {
        return "- unavailable".to_string();
    };

    let mut lines = Vec::new();
    if let Some(id) = context.experience_id {
        lines.push(format!("- experience_id: {id}"));
    }
    if !context.summary.trim().is_empty() {
        lines.push(format!(
            "- summary: {}",
            compact_line(&context.summary, 240)
        ));
    }
    lines.push(format!(
        "- counts: sensations={}, derived_sensations={}, impressions={}, lineage_edges={}",
        context.sensations.len(),
        context.derived_sensation_count(),
        context.impressions.len(),
        context.lineage.len()
    ));
    if let Some(vector) = &context.fused_vector {
        lines.push(format!(
            "- fused_vector: model={} dim={} source_sensation={}",
            vector.model_id, vector.dim, vector.source_sensation_id
        ));
    }
    for sensation in context.sensations.iter().take(8) {
        let parent = sensation
            .parent_id
            .map(|id| format!(" parent={id}"))
            .unwrap_or_default();
        let summary = sensation
            .summary
            .as_deref()
            .map(|text| format!(" summary=\"{}\"", compact_line(text, 120)))
            .unwrap_or_default();
        lines.push(format!(
            "- sensation {}: modality={} payload={} kind={}{}{}",
            sensation.id,
            sensation.modality.as_str(),
            sensation.payload_kind.as_str(),
            sensation.kind,
            parent,
            summary
        ));
    }
    for impression in context.impressions.iter().rev().take(6).rev() {
        let target = impression
            .sensation_id
            .map(|id| format!("sensation={id}"))
            .or_else(|| {
                impression
                    .experience_id
                    .map(|id| format!("experience={id}"))
            })
            .unwrap_or_else(|| "target=unknown".to_string());
        lines.push(format!(
            "- impression {}: {} \"{}\"",
            impression.id,
            target,
            compact_line(&impression.text, 160)
        ));
    }
    for edge in context.lineage.iter().take(8) {
        lines.push(format!(
            "- lineage: {} -> {}",
            edge.parent_id, edge.child_id
        ));
    }
    for vector in context.sensation_vectors.iter().take(6) {
        lines.push(format!(
            "- sensation_vector: sensation={} model={} dim={} modality={} payload={}",
            vector.source_sensation_id,
            vector.model_id,
            vector.dim,
            vector.modality.as_str(),
            vector.payload_kind.as_str()
        ));
    }
    for prediction in context.predictions.iter().take(4) {
        let vector = prediction
            .vector
            .as_ref()
            .map(|vector| {
                format!(
                    " vector_model={} vector_dim={}",
                    vector.model_id, vector.dim
                )
            })
            .unwrap_or_default();
        lines.push(format!(
            "- prediction +{}ms confidence={:.2}{}: {}",
            prediction.offset_ms,
            prediction.confidence,
            vector,
            compact_line(&prediction.text, 140)
        ));
    }
    for link in context.memory_links.iter().take(4) {
        let text = link
            .text
            .as_deref()
            .map(|text| format!(" \"{}\"", compact_line(text, 120)))
            .unwrap_or_default();
        lines.push(format!(
            "- memory_link: target={} relation={} score={:.2}{}",
            link.target_id, link.relation, link.score, text
        ));
    }
    lines.join("\n")
}

fn compact_line(text: &str, max_chars: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= max_chars {
        return compact;
    }
    let mut out = compact
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>();
    out.push_str("...");
    out
}

fn summarize_reign_command(input: &netherwick_actions::ReignInput) -> String {
    match &input.command {
        netherwick_actions::ReignCommand::Stop => "Stop".to_string(),
        netherwick_actions::ReignCommand::Go {
            intensity,
            duration_ms,
        } => format!("Go, intensity {:.2}, {}ms", intensity, duration_ms),
        netherwick_actions::ReignCommand::Reverse {
            intensity,
            duration_ms,
        } => format!("Reverse, intensity {:.2}, {}ms", intensity, duration_ms),
        netherwick_actions::ReignCommand::Drive {
            forward,
            turn,
            duration_ms,
        } => format!(
            "Drive, forward {:.2}, turn {:.2}, {}ms",
            forward, turn, duration_ms
        ),
        netherwick_actions::ReignCommand::Turn {
            direction,
            intensity,
            duration_ms,
        } => format!(
            "Turn {:?}, intensity {:.2}, {}ms",
            direction, intensity, duration_ms
        ),
        netherwick_actions::ReignCommand::Inspect { target } => {
            format!("Inspect {:?}", target)
        }
        netherwick_actions::ReignCommand::Approach { target } => {
            format!("Approach {:?}", target)
        }
        netherwick_actions::ReignCommand::Dock => "Dock".to_string(),
        netherwick_actions::ReignCommand::Explore { duration_ms } => {
            format!("Explore for {}ms", duration_ms)
        }
        netherwick_actions::ReignCommand::Speak { text } => {
            format!("Speak {text}")
        }
        netherwick_actions::ReignCommand::SetMode { mode } => {
            format!("Set mode {:?}", mode)
        }
    }
}

fn reign_command_summary(now: &Now) -> Option<String> {
    let input = now.reign.latest.as_ref()?;
    if !reign_command_can_drive_agent(now, input) {
        return None;
    }
    Some(format!(
        "Following Reign command: {}",
        summarize_reign_command(input)
    ))
}

fn active_reign_action(now: &Now) -> Option<ActionPrimitive> {
    let input = now.reign.latest.as_ref()?;
    if !reign_command_can_drive_agent(now, input) {
        return None;
    }
    input.command.to_action()
}

fn action_spec_json(action: &ActionPrimitive) -> serde_json::Value {
    match action {
        ActionPrimitive::Stop => serde_json::json!({ "kind": "stop" }),
        ActionPrimitive::Go {
            intensity,
            duration_ms,
        } => serde_json::json!({
            "kind": "go",
            "intensity": intensity,
            "duration_ms": duration_ms,
        }),
        ActionPrimitive::Drive {
            forward,
            turn,
            duration_ms,
        } => serde_json::json!({
            "kind": "drive",
            "forward": forward,
            "turn": turn,
            "duration_ms": duration_ms,
        }),
        ActionPrimitive::Turn {
            direction,
            intensity,
            duration_ms,
        } => serde_json::json!({
            "kind": "turn",
            "direction": match direction {
                TurnDir::Left => "left",
                TurnDir::Right => "right",
            },
            "intensity": intensity,
            "duration_ms": duration_ms,
        }),
        ActionPrimitive::Inspect { target } => serde_json::json!({
            "kind": "inspect",
            "target": match target {
                InspectTarget::Novelty => "novelty",
                InspectTarget::Charger => "charger",
                InspectTarget::Person => "person",
                InspectTarget::Sound => "sound",
            },
        }),
        ActionPrimitive::Approach { target } => serde_json::json!({
            "kind": "approach",
            "target": match target {
                ApproachTarget::Charger => "charger",
                ApproachTarget::Person => "person",
                ApproachTarget::Sound => "sound",
            },
        }),
        ActionPrimitive::Dock => serde_json::json!({ "kind": "dock" }),
        ActionPrimitive::Explore { style, duration_ms } => serde_json::json!({
            "kind": "explore",
            "style": match style {
                ExploreStyle::Wander => "wander",
                ExploreStyle::RandomWalk => "random_walk",
                ExploreStyle::WallFollow => "wall_follow",
            },
            "duration_ms": duration_ms,
        }),
        ActionPrimitive::Speak { text } => serde_json::json!({
            "kind": "speak",
            "text": text,
        }),
        ActionPrimitive::Chirp { pattern } => serde_json::json!({
            "kind": "chirp",
            "pattern": match pattern {
                ChirpPattern::Confirm => "confirm",
                ChirpPattern::Warning => "warning",
                ChirpPattern::Curious => "curious",
            },
        }),
    }
}

fn reign_command_can_drive_agent(now: &Now, input: &netherwick_actions::ReignInput) -> bool {
    if !now.reign.active {
        return false;
    }
    !matches!(input.mode, netherwick_actions::ReignMode::ObserveOnly)
}

fn summarize_futures(futures: &[FuturePrediction]) -> String {
    if futures.is_empty() {
        return "- none".to_string();
    }
    futures
        .iter()
        .map(|future| {
            format!(
                "- +{}ms confidence {:.2}{}",
                future.offset_ms,
                future.confidence,
                future
                    .summary
                    .as_ref()
                    .map(|summary| format!(": {summary}"))
                    .unwrap_or_default()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_combobulator_timeline(impressions: &[Impression]) -> String {
    if impressions.is_empty() {
        return "- no impressions".to_string();
    }

    let mut ordered = impressions.to_vec();
    ordered.sort_by_key(|impression| (impression.occurred_at_ms, impression.observed_at_ms));
    let start_ms = ordered
        .first()
        .map(|impression| impression.occurred_at_ms)
        .unwrap_or_default();
    let clusters = impression_clusters(&ordered, COMBOBULATOR_CLUSTER_GAP_MS);
    let mut out = String::new();
    for (index, cluster) in clusters.iter().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        out.push_str(&format_impression_cluster(cluster, start_ms));
        for impression in *cluster {
            out.push_str(&format_impression_timeline_entry(impression, start_ms));
        }
    }
    out
}

fn local_iso_ms(t_ms: u64) -> String {
    match Local.timestamp_millis_opt(t_ms as i64).single() {
        Some(value) => value.to_rfc3339_opts(SecondsFormat::Millis, false),
        None => format!("{t_ms}ms"),
    }
}

fn impression_clusters(impressions: &[Impression], max_gap_ms: u64) -> Vec<&[Impression]> {
    if impressions.is_empty() {
        return Vec::new();
    }

    let mut clusters = Vec::new();
    let mut start = 0usize;
    let mut previous_ms = impressions[0].occurred_at_ms;
    for (index, impression) in impressions.iter().enumerate().skip(1) {
        if impression.occurred_at_ms.saturating_sub(previous_ms) > max_gap_ms {
            clusters.push(&impressions[start..index]);
            start = index;
        }
        previous_ms = impression.occurred_at_ms;
    }
    clusters.push(&impressions[start..]);
    clusters
}

fn format_impression_cluster(cluster: &[Impression], start_ms: u64) -> String {
    let first_ms = cluster
        .first()
        .map(|impression| impression.occurred_at_ms)
        .unwrap_or(start_ms);
    let last_ms = cluster
        .last()
        .map(|impression| impression.occurred_at_ms)
        .unwrap_or(first_ms);
    format!(
        "[T+{:06.3} - T+{:06.3} | {} to {}]\n",
        elapsed_seconds(start_ms, first_ms),
        elapsed_seconds(start_ms, last_ms),
        local_iso_ms(first_ms),
        local_iso_ms(last_ms)
    )
}

fn format_impression_timeline_entry(impression: &Impression, start_ms: u64) -> String {
    let generator = impression
        .payload
        .get("generator")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    let faculty = impression
        .payload
        .get("faculty")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    format!(
        "T+{:06.3} occurred_at={}\n  IMPRESSION id={} kind={} generator={} faculty={} observed_at={} confidence={:.3} about=[{}] payload={} text={}\n",
        elapsed_seconds(start_ms, impression.occurred_at_ms),
        local_iso_ms(impression.occurred_at_ms),
        impression.id,
        impression.kind,
        prompt_json_string(generator),
        prompt_json_string(faculty),
        local_iso_ms(impression.observed_at_ms),
        impression.confidence,
        impression
            .about
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(","),
        prompt_json_string(&impression.payload.to_string()),
        prompt_json_string(&impression.text)
    )
}

fn elapsed_seconds(start_ms: u64, t_ms: u64) -> f64 {
    t_ms.saturating_sub(start_ms) as f64 / MILLIS_PER_SECOND
}

fn prompt_json_string(text: &str) -> String {
    serde_json::to_string(text)
        .expect("prompt string fragment is serializable")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
}

fn heuristic_combobulation(now: &Now, recall_summary: &str) -> Combobulation {
    let summary = if let Some(transcript) = now
        .ear
        .transcript
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        format!("I hear: {transcript}")
    } else if now.body.flags.bump_left || now.body.flags.bump_right {
        "My body feels blocked by contact.".to_string()
    } else if now.body.flags.cliff_left
        || now.body.flags.cliff_front_left
        || now.body.flags.cliff_front_right
        || now.body.flags.cliff_right
        || now.body.cliff_sensors.max() >= 0.5
    {
        "I feel the floor fall away near me.".to_string()
    } else if let Some(nearest_m) = now.range.nearest_m {
        format!("Nearest obstacle is {:.2} meters away.", nearest_m)
    } else if !recall_summary.trim().is_empty() {
        recall_summary.trim().to_string()
    } else {
        format!("I am active at t={}ms.", now.t_ms)
    };
    Combobulation {
        summary,
        confidence: 0.35,
    }
}

fn parse_counterfactual_spec(spec: CounterfactualSpec) -> Option<CounterfactualAction> {
    Some(CounterfactualAction {
        instead_of: spec.instead_of.and_then(parse_action_spec),
        proposed: parse_action_spec(spec.proposed)?,
        reason: spec.reason,
        weight: spec.weight.clamp(0.0, 1.0),
    })
}

pub fn parse_llm_decision_json(text: &str, commands_enabled: bool) -> Result<LlmDecision> {
    let json = extract_json_object(text).unwrap_or_else(|| text.trim().to_string());
    let reply: AgentReply = serde_json::from_str(&json).context("failed to parse llm decision")?;
    let summary = reply.summary.trim().to_string();
    Ok(LlmDecision {
        summary,
        critique: reply
            .critique
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
        confidence: reply.confidence.clamp(0.0, 1.0),
        action: if commands_enabled {
            reply.action.and_then(parse_action_spec)
        } else {
            None
        },
        counterfactuals: reply
            .counterfactuals
            .into_iter()
            .filter_map(parse_counterfactual_spec)
            .collect(),
        memory_notes: reply.memory_notes,
    })
}

fn parse_action_spec(spec: ActionSpec) -> Option<ActionPrimitive> {
    let kind = spec.kind.to_ascii_lowercase();
    match kind.as_str() {
        "stop" => Some(ActionPrimitive::Stop),
        "go" => Some(ActionPrimitive::Go {
            intensity: spec.intensity.unwrap_or(0.2).clamp(0.0, 1.0),
            duration_ms: spec.duration_ms.unwrap_or(1_000),
        }),
        "turn" => Some(ActionPrimitive::Turn {
            direction: match spec.direction.as_deref()?.to_ascii_lowercase().as_str() {
                "left" => TurnDir::Left,
                "right" => TurnDir::Right,
                _ => return None,
            },
            intensity: spec.intensity.unwrap_or(0.4).clamp(0.0, 1.0),
            duration_ms: spec.duration_ms.unwrap_or(1_000),
        }),
        "inspect" => Some(ActionPrimitive::Inspect {
            target: match spec.target.as_deref()?.to_ascii_lowercase().as_str() {
                "novelty" => InspectTarget::Novelty,
                "charger" => InspectTarget::Charger,
                "person" => InspectTarget::Person,
                "sound" => InspectTarget::Sound,
                _ => return None,
            },
        }),
        "approach" => Some(ActionPrimitive::Approach {
            target: match spec.target.as_deref()?.to_ascii_lowercase().as_str() {
                "charger" => ApproachTarget::Charger,
                "person" => ApproachTarget::Person,
                "sound" => ApproachTarget::Sound,
                _ => return None,
            },
        }),
        "dock" => Some(ActionPrimitive::Dock),
        "explore" => Some(ActionPrimitive::Explore {
            style: match spec
                .style
                .as_deref()
                .unwrap_or("random_walk")
                .to_ascii_lowercase()
                .as_str()
            {
                "wander" => ExploreStyle::Wander,
                "random_walk" => ExploreStyle::RandomWalk,
                "wall_follow" => ExploreStyle::WallFollow,
                _ => return None,
            },
            duration_ms: spec.duration_ms.unwrap_or(1_000),
        }),
        "speak" => Some(ActionPrimitive::Speak {
            text: spec.text.unwrap_or_default(),
        }),
        "chirp" => Some(ActionPrimitive::Chirp {
            pattern: match spec
                .pattern
                .as_deref()
                .unwrap_or("confirm")
                .to_ascii_lowercase()
                .as_str()
            {
                "confirm" => ChirpPattern::Confirm,
                "warning" => ChirpPattern::Warning,
                "curious" => ChirpPattern::Curious,
                _ => return None,
            },
        }),
        _ => None,
    }
}

fn extract_json_object(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if let Some(json) = normalize_json_candidate(trimmed) {
        return Some(json);
    }

    if let Some(unfenced) = strip_markdown_fence(trimmed) {
        if let Some(json) = normalize_json_candidate(&unfenced) {
            return Some(json);
        }
    }

    let mut start = None;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (index, ch) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            continue;
        }

        if ch == '"' {
            in_string = true;
            continue;
        }

        if ch == '{' {
            if start.is_none() {
                start = Some(index);
            }
            depth += 1;
        } else if ch == '}' {
            if depth == 0 {
                continue;
            }
            depth -= 1;
            if depth == 0 {
                let candidate = &text[start?..=index];
                if let Some(json) = normalize_json_candidate(candidate) {
                    return Some(json);
                }
            }
        }
    }
    None
}

fn normalize_json_candidate(candidate: &str) -> Option<String> {
    let trimmed = candidate.trim();
    if serde_json::from_str::<Value>(trimmed).is_ok() {
        return Some(trimmed.to_string());
    }

    if let Ok(json_text) = serde_json::from_str::<String>(trimmed) {
        let inner = json_text.trim();
        if serde_json::from_str::<Value>(inner).is_ok() {
            return Some(inner.to_string());
        }
    }

    None
}

fn strip_markdown_fence(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if !(trimmed.starts_with("```") && trimmed.ends_with("```")) {
        return None;
    }

    let mut lines = trimmed.lines();
    let first = lines.next()?;
    if !first.starts_with("```") {
        return None;
    }

    let mut content = lines.collect::<Vec<_>>();
    if content.last().copied() != Some("```") {
        return None;
    }
    content.pop();
    Some(content.join("\n").trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use netherwick_actions::{ReignCommand, ReignInput, ReignMode, ReignSource};
    use netherwick_body::{BodySense, CliffSensors};
    use netherwick_now::Now;
    use uuid::Uuid;

    #[test]
    fn default_llm_config_uses_local_ollama() {
        let config = LlmConfig::default();

        assert_eq!(config.provider, LlmProvider::Ollama);
        assert_eq!(config.endpoint, "http://127.0.0.1:11434");
    }

    #[test]
    fn extracts_json_from_fenced_response() {
        let text = "```json\n{\"summary\":\"hi\",\"confidence\":0.9}\n```";
        let json = extract_json_object(text).unwrap();
        assert_eq!(json, "{\"summary\":\"hi\",\"confidence\":0.9}");
    }

    #[test]
    fn extracts_json_from_wrapped_response_text() {
        let text = "Sure, here you go:\n{\"summary\":\"hi\",\"confidence\":0.9}\nThanks";
        let json = extract_json_object(text).unwrap();
        assert_eq!(json, "{\"summary\":\"hi\",\"confidence\":0.9}");
    }

    #[test]
    fn parses_turn_action() {
        let action = parse_action_spec(ActionSpec {
            kind: "turn".to_string(),
            direction: Some("left".to_string()),
            intensity: Some(0.6),
            duration_ms: Some(1200),
            target: None,
            text: None,
            style: None,
            pattern: None,
        })
        .unwrap();
        assert_eq!(
            action,
            ActionPrimitive::Turn {
                direction: TurnDir::Left,
                intensity: 0.6,
                duration_ms: 1200,
            }
        );
    }

    #[test]
    fn parses_llm_json_explore_action_into_decision() {
        let decision = parse_llm_decision_json(r#"{"action":{"kind":"explore"}}"#, true).unwrap();
        assert_eq!(
            decision.action,
            Some(ActionPrimitive::Explore {
                style: ExploreStyle::RandomWalk,
                duration_ms: 1_000,
            })
        );
    }

    #[test]
    fn commands_disabled_ignores_llm_json_action() {
        let decision = parse_llm_decision_json(r#"{"action":{"kind":"explore"}}"#, false).unwrap();
        assert_eq!(decision.action, None);
    }

    #[test]
    fn summarized_senses_include_latest_reign_command() {
        let mut now = Now::blank(100, BodySense::default());
        now.reign.latest = Some(ReignInput {
            id: Uuid::new_v4(),
            issued_at_ms: 100,
            expires_at_ms: 1_100,
            source: ReignSource::WebRemote,
            mode: ReignMode::Direct,
            command: ReignCommand::Turn {
                direction: TurnDir::Left,
                intensity: 0.5,
                duration_ms: 500,
            },
            priority: 1.0,
            note: Some("turn toward charger".to_string()),
        });
        now.reign.active = true;
        now.reign.mode = Some(ReignMode::Direct);

        let senses = summarized_senses(&now).join("\n");

        assert!(senses.contains("Remote control active: Direct"));
        assert!(senses.contains("Latest remote command: Turn Left"));
        assert!(senses.contains(
            "Matching executable remote action JSON: {\"direction\":\"left\",\"duration_ms\":500,\"intensity\":0.5,\"kind\":\"turn\"}."
        ));
        assert!(senses.contains("Remote note: turn toward charger"));
    }

    #[test]
    fn active_reign_action_becomes_llm_command_action() {
        let command = ReignCommand::Go {
            intensity: 0.4,
            duration_ms: 700,
        };
        let mut now = Now::blank(100, BodySense::default());
        now.reign.active = true;
        now.reign.mode = Some(ReignMode::Assist);
        now.reign.latest = Some(ReignInput {
            id: Uuid::new_v4(),
            issued_at_ms: 100,
            expires_at_ms: 1_100,
            source: ReignSource::WebRemote,
            mode: ReignMode::Assist,
            command: command.clone(),
            priority: 1.0,
            note: None,
        });

        assert_eq!(active_reign_action(&now), command.to_action());
        assert_eq!(
            reign_command_summary(&now),
            Some("Following Reign command: Go, intensity 0.40, 700ms".to_string())
        );
    }

    #[test]
    fn observe_only_reign_does_not_become_llm_command_action() {
        let mut now = Now::blank(100, BodySense::default());
        now.reign.active = true;
        now.reign.mode = Some(ReignMode::ObserveOnly);
        now.reign.latest = Some(ReignInput {
            id: Uuid::new_v4(),
            issued_at_ms: 100,
            expires_at_ms: 1_100,
            source: ReignSource::WebRemote,
            mode: ReignMode::ObserveOnly,
            command: ReignCommand::Stop,
            priority: 1.0,
            note: None,
        });
        assert_eq!(active_reign_action(&now), None);
    }

    #[test]
    fn agent_prompt_frames_actions_as_executable_and_reign_as_command_input() {
        let now = Now::blank(100, BodySense::default());
        let prompt = build_agent_prompt(
            &now,
            None,
            &ExperienceLatent::default(),
            &[],
            "none",
            Some("I am awake."),
            &LlmConfig::default(),
        );

        assert!(prompt.contains("choose a high-level action primitive"));
        assert!(prompt.contains("executable command candidate"));
        assert!(prompt.contains("not only a suggestion or note"));
        assert!(prompt.contains("Never output raw motor control such as wheel speeds"));
        assert!(prompt.contains("Treat Reign controls as present-tense command input"));
        assert!(prompt.contains("set action to the matching allowed action"));
        assert!(!prompt.contains("Do not override active Direct reign"));
    }

    #[test]
    fn summarized_senses_include_input_sensor_channels() {
        let mut now = Now::blank(100, BodySense::default());
        now.body.flags.cliff_front_left = true;
        now.body.flags.wall = true;
        now.body.flags.virtual_wall = true;
        now.body.cliff_sensors = CliffSensors {
            left: 0.10,
            front_left: 0.80,
            front_right: 0.40,
            right: 0.20,
        };
        now.kinect.ir = vec![0.1, 0.8, 0.9, 0.2];

        let senses = summarized_senses(&now).join("\n");

        assert!(senses.contains("I feel the floor fall away near me."));
        assert!(!senses.contains("Cliff sensor levels"));
        assert!(senses.contains("My wall sensor is active."));
        assert!(senses.contains("I detect a virtual wall."));
        assert!(
            senses.contains("Kinect IR has 4 samples, mean 0.50, max 0.90, bright fraction 0.50.")
        );
    }

    #[test]
    fn combobulator_prompt_uses_timeline_distillation_rules() {
        let mut now = Now::blank(250, BodySense::default());
        now.ear.transcript = Some("hello there".to_string());

        let impression = Impression::new(
            "audio.transcript.impression",
            "I hear: <hello there>",
            Vec::new(),
            now.t_ms,
            now.t_ms,
        )
        .with_confidence(0.8)
        .with_payload(serde_json::json!({
            "generator": "mechanical",
            "faculty": "ear.mechanical_impression",
        }));
        let prompt = build_combobulator_prompt(
            &now,
            &[impression],
            None,
            &ExperienceLatent::default(),
            &[],
            "I remember Tim.",
        );

        assert!(prompt.contains("Timeline evidence:"));
        assert!(prompt.contains("[T+00.000 - T+00.000 | "));
        assert!(prompt.contains("IMPRESSION id="));
        assert!(prompt.contains("kind=audio.transcript.impression"));
        assert!(prompt.contains("generator=\"mechanical\""));
        assert!(prompt.contains("faculty=\"ear.mechanical_impression\""));
        assert!(prompt.contains("confidence=0.800"));
        assert!(prompt.contains("occurred_at="));
        assert!(prompt.contains("observed_at="));
        assert!(prompt.contains(".250"));
        assert!(prompt.contains(":00 to "));
        assert!(prompt.contains("what is going on right now"));
        assert!(prompt.contains("first-person lived experience"));
        assert!(prompt
            .contains("Convert raw body data into feeling-centered first-person interpretations"));
        assert!(prompt.contains("I feel steady"));
        assert!(prompt.contains("telling someone with amnesia"));
        assert!(prompt.contains("Distill what matters, not what the records said."));
        assert!(prompt.contains("Treat the entries as fragmentary, possibly contradictory"));
        assert!(prompt.contains("not as the topic to describe"));
        assert!(prompt.contains("do not group by faculty or source"));
        assert!(prompt.contains("Do not infer emotional tone"));
        assert!(prompt.contains("do not enumerate ids"));
        assert!(prompt.contains("Do not assume a human is currently present"));
        assert!(prompt.contains("CONTEXT FRAME"));
        assert!(prompt.contains("text=\"I hear: \\u003chello there\\u003e\""));
    }

    #[test]
    fn prompts_include_embodied_context_without_raw_vectors() {
        let sensation_id = Uuid::new_v4();
        let experience_id = Uuid::new_v4();
        let context = EmbodiedContext {
            experience_id: Some(experience_id),
            summary: "I see a frame and focus on part of it.".to_string(),
            sensations: vec![netherwick_experience::EmbodiedSensationRef {
                id: sensation_id,
                parent_id: Some(Uuid::new_v4()),
                modality: netherwick_experience::Modality::Vision,
                payload_kind: netherwick_experience::SensationPayloadKind::ImageBytes,
                kind: "vision.image_bytes".to_string(),
                source: "camera".to_string(),
                summary: Some("A camera frame is visible.".to_string()),
            }],
            impressions: Vec::new(),
            lineage: Vec::new(),
            fused_vector: Some(netherwick_experience::EmbodiedVectorRef {
                model_id: "fuser.v0".to_string(),
                dim: 16,
                modality: netherwick_experience::Modality::Other,
                payload_kind: netherwick_experience::SensationPayloadKind::Structured,
                source_sensation_id: sensation_id,
            }),
            sensation_vectors: Vec::new(),
            predictions: Vec::new(),
            memory_links: Vec::new(),
        };
        let now = Now::blank(100, BodySense::default());

        let prompt = build_agent_prompt(
            &now,
            Some(&context),
            &ExperienceLatent::default(),
            &[],
            "none",
            None,
            &LlmConfig::default(),
        );

        assert!(prompt.contains("Current embodied experience:"));
        assert!(prompt.contains(&format!("experience_id: {experience_id}")));
        assert!(prompt.contains("derived_sensations=1"));
        assert!(prompt.contains("fused_vector: model=fuser.v0 dim=16"));
        assert!(prompt.contains("payload=image_bytes"));
        assert!(!prompt.contains("[0."));
    }

    #[test]
    fn image_caption_prompt_frames_live_vision_viewpoint() {
        assert!(IMAGE_CAPTION_PROMPT.contains("what I am seeing now"));
        assert!(IMAGE_CAPTION_PROMPT.contains("my own vision looking out"));
        assert!(IMAGE_CAPTION_PROMPT.contains("not that visible people"));
        assert!(IMAGE_CAPTION_PROMPT.contains("When looking out, one does not see oneself"));
        assert!(
            IMAGE_CAPTION_PROMPT.contains("unless I am clearly looking in a mirror or reflection")
        );
        assert!(IMAGE_CAPTION_PROMPT.contains("Describe visible people in third person"));
        assert!(!IMAGE_CAPTION_PROMPT.contains("data:image"));
    }

    #[test]
    fn heuristic_combobulation_prefers_concrete_present_evidence() {
        let mut now = Now::blank(500, BodySense::default());
        now.ear.transcript = Some("come over here".to_string());
        now.body.flags.bump_left = true;

        let combobulation = heuristic_combobulation(&now, "A stale memory.");

        assert_eq!(combobulation.summary, "I hear: come over here");
        assert_eq!(combobulation.confidence, 0.35);
    }
}
