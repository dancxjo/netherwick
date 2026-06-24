use anyhow::{Context, Result};
use async_trait::async_trait;
use netherwick_actions::{
    ActionPrimitive, ApproachTarget, ChirpPattern, ExploreStyle, InspectTarget, TurnDir,
};
use netherwick_experience::{ExperienceLatent, FuturePrediction};
use netherwick_now::{LlmSense, Now, ReignSense};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::time::Duration;

const SENSOR_GROUNDING_RULES: &str = "Describe the real-world scene or event, not the sensor stream. Interpret images, audio, motion, location, body state, and memory-derived entries as the robot's own senses, not as media files or external sensor artifacts. Do not summarize the amount, density, cadence, or mix of input modalities as if that were the situation. Repeated frames, repeated detections, embeddings, and heartbeat-like status records are usually evidence to compress or ignore, not events to report. If the evidence does not reveal what is happening, say that I cannot tell what is happening yet.";

const COMBOBULATOR_DISTILLATION_RULES: &str = "Distill what matters, not what the records said. Treat the entries as fragmentary, possibly contradictory, fleeting evidence about the actual situation. Sort meaning by time: occurred time first, observed time second. When related entries describe raw audio and the transcript derived from it, treat them as one real-world event. Some entries may be prior combobulation summaries looping back as sensation; use those only as provisional, possibly stale self-context, not as fresh external evidence. Do not say that you are observing a timeline, records, sensor streams, previous summaries, or a shift in conversation. Compress repeated low-level records into the real-world gist; do not enumerate ids, hashes, timestamps, or each detection unless they are the point.";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConsciousCommand {
    pub summary: String,
    pub action: Option<ActionPrimitive>,
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
        z: &ExperienceLatent,
        futures: &[FuturePrediction],
        recall_summary: &str,
    ) -> Result<Option<Combobulation>>;

    async fn maybe_tick(
        &mut self,
        now: &Now,
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
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
    ) -> Result<Option<Combobulation>> {
        Ok(None)
    }

    async fn maybe_tick(
        &mut self,
        _now: &Now,
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
        Self {
            provider: LlmProvider::Disabled,
            allow_commands: true,
            allow_teaching: true,
            endpoint: "http://127.0.0.1:11434".to_string(),
            agent_model: "qwen2.5:7b-instruct".to_string(),
            combobulator_model: None,
            temperature: 0.2,
            timeout_ms: 20_000,
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
        lines.push("My bump sensors are pressed.".to_string());
    }
    if now.body.flags.cliff_left
        || now.body.flags.cliff_front_left
        || now.body.flags.cliff_front_right
        || now.body.flags.cliff_right
        || now.body.cliff_sensors.max() >= 0.5
    {
        lines.push("I detect a cliff edge.".to_string());
    }
    if now.body.cliff_sensors.max() > 0.0 {
        lines.push(format!(
            "Cliff sensor levels are left {:.2}, front-left {:.2}, front-right {:.2}, right {:.2}.",
            now.body.cliff_sensors.left,
            now.body.cliff_sensors.front_left,
            now.body.cliff_sensors.front_right,
            now.body.cliff_sensors.right
        ));
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
            "Latest human reign command: {}.",
            summarize_reign_command(input)
        ));
        if let Some(note) = &input.note {
            lines.push(format!("Human note: {note}"));
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
        z: &ExperienceLatent,
        futures: &[FuturePrediction],
        recall_summary: &str,
    ) -> Result<Option<Combobulation>> {
        match self {
            Self::Disabled(agent) => agent.combobulate(now, z, futures, recall_summary).await,
            Self::Ollama(agent) => agent.combobulate(now, z, futures, recall_summary).await,
        }
    }

    async fn maybe_tick(
        &mut self,
        now: &Now,
        z: &ExperienceLatent,
        futures: &[FuturePrediction],
        recall_summary: &str,
        awareness_summary: Option<&str>,
    ) -> Result<LlmTickResult> {
        match self {
            Self::Disabled(agent) => {
                agent
                    .maybe_tick(now, z, futures, recall_summary, awareness_summary)
                    .await
            }
            Self::Ollama(agent) => {
                agent
                    .maybe_tick(now, z, futures, recall_summary, awareness_summary)
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

    async fn generate_json<T>(&self, model: &str, prompt: String) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let request = OllamaGenerateRequest {
            model,
            prompt,
            stream: false,
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
            .context("failed to reach ollama")?
            .error_for_status()
            .context("ollama returned an error")?;
        let body: OllamaGenerateResponse = response
            .json()
            .await
            .context("failed to decode ollama response")?;
        let json = extract_json_object(&body.response)
            .with_context(|| format!("ollama returned non-json content: {}", body.response))?;
        serde_json::from_str(&json).context("failed to parse llm json payload")
    }
}

#[async_trait]
impl LlmAgent for OllamaLlmAgent {
    async fn combobulate(
        &mut self,
        now: &Now,
        z: &ExperienceLatent,
        futures: &[FuturePrediction],
        recall_summary: &str,
    ) -> Result<Option<Combobulation>> {
        let prompt = build_combobulator_prompt(now, z, futures, recall_summary);
        let model = self
            .config
            .combobulator_model
            .as_deref()
            .unwrap_or(self.config.agent_model.as_str());
        match self.generate_json::<CombobulatorReply>(model, prompt).await {
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
            z,
            futures,
            recall_summary,
            awareness_summary,
            &self.config,
        );
        let reply = match self
            .generate_json::<AgentReply>(&self.config.agent_model, prompt)
            .await
        {
            Ok(reply) => reply,
            Err(_) => return Ok(LlmTickResult::default()),
        };

        let action = if self.config.allow_commands {
            reply.action.and_then(parse_action_spec)
        } else {
            None
        };
        let critique = reply
            .critique
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        let summary = reply.summary.trim().to_string();
        let confidence = reply.confidence.clamp(0.0, 1.0);

        let conscious_command =
            if self.config.allow_commands && (action.is_some() || !summary.is_empty()) {
                Some(ConsciousCommand {
                    summary: summary.clone(),
                    action: action.clone(),
                })
            } else {
                None
            };

        let teaching = if self.config.allow_teaching
            && (critique.is_some()
                || !reply.memory_notes.is_empty()
                || !reply.counterfactuals.is_empty())
        {
            vec![LlmTeaching {
                t_ms: now.t_ms,
                summary: if summary.is_empty() {
                    "LLM reflection".to_string()
                } else {
                    summary.clone()
                },
                critique: critique.clone(),
                counterfactuals: reply
                    .counterfactuals
                    .into_iter()
                    .filter_map(parse_counterfactual_spec)
                    .collect(),
                memory_notes: reply.memory_notes,
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
    z: &ExperienceLatent,
    futures: &[FuturePrediction],
    recall_summary: &str,
) -> String {
    let timeline = render_combobulator_timeline(now);
    let futures = summarize_futures(futures);
    format!(
        "You are the combobulator for an embodied robot.\n\
Given recent sensations, impressions, memories, and predicted futures in timeline order, distill what appears to be happening right now.\n\
Write one short grounded first-person sentence using I/my/me. Use only the evidence below. Prefer concrete body facts, nearby people or speech, memory, safety, and immediate context.\n\
{SENSOR_GROUNDING_RULES}\n\
{COMBOBULATOR_DISTILLATION_RULES}\n\
Return JSON only with this schema:\n\
{{\"summary\":\"...\",\"confidence\":0.0}}\n\n\
CONTEXT FRAME\n\
WHO\n\
- embodied robot\n\
WHAT\n\
- current awareness synthesis\n\
WHERE\n\
- current body location if sensors or memory reveal it; otherwise unknown\n\
WHEN\n\
- now at {} ms\n\
WHY\n\
- produce a compact awareness statement useful to the next action decision\n\
HOW\n\
- distill body sense, hearing, range, memory, predictions, surprise, and human reign controls\n\n\
Latent confidence: {:.2}\n\
Latent prediction error: {:.2}\n\
Recall summary: {}\n\
Timeline evidence:\n{}\n\
Predicted futures:\n{}\n",
        now.t_ms, z.confidence, z.prediction_error, recall_summary, timeline, futures
    )
}

fn build_agent_prompt(
    now: &Now,
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
    let futures = summarize_futures(futures);
    format!(
        "You are the conscious LLM layer for an embodied robot.\n\
You may suggest a high-level action primitive, critique the situation, and record memory notes.\n\
Never output raw motor control.\n\
A human may be steering you. Treat Reign controls as important present-tense input. Do not override active Direct reign unless there is a safety or coherence reason. You may comment on it, remember it, or learn from it.\n\
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
        recall_summary,
        now.body.battery_level,
        now.surprise.total,
        z.confidence,
        futures,
        senses
    )
}

fn summarize_reign_command(input: &netherwick_actions::ReignInput) -> String {
    match &input.command {
        netherwick_actions::ReignCommand::Stop => "Stop".to_string(),
        netherwick_actions::ReignCommand::Go {
            intensity,
            duration_ms,
        } => format!("Go, intensity {:.2}, {}ms", intensity, duration_ms),
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

fn render_combobulator_timeline(now: &Now) -> String {
    let senses = summarized_senses(now);
    if senses.is_empty() {
        return "- no direct sensory evidence".to_string();
    }
    senses
        .into_iter()
        .map(|line| format!("[{} ms observed]\nSENSE\n  {line}", now.t_ms))
        .collect::<Vec<_>>()
        .join("\n")
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
        "My bump sensors are pressed.".to_string()
    } else if now.body.flags.cliff_left
        || now.body.flags.cliff_front_left
        || now.body.flags.cliff_front_right
        || now.body.flags.cliff_right
        || now.body.cliff_sensors.max() >= 0.5
    {
        "I detect a cliff edge.".to_string()
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
    if serde_json::from_str::<Value>(text).is_ok() {
        return Some(text.to_string());
    }

    let mut start = None;
    let mut depth = 0usize;
    for (index, ch) in text.char_indices() {
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
                if serde_json::from_str::<Value>(candidate).is_ok() {
                    return Some(candidate.to_string());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use netherwick_actions::{ReignCommand, ReignInput, ReignMode, ReignSource};
    use netherwick_body::{BodySense, CliffSensors};
    use netherwick_now::Now;
    use uuid::Uuid;

    #[test]
    fn extracts_json_from_fenced_response() {
        let text = "```json\n{\"summary\":\"hi\",\"confidence\":0.9}\n```";
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
        assert!(senses.contains("Latest human reign command: Turn Left"));
        assert!(senses.contains("Human note: turn toward charger"));
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

        assert!(senses.contains("I detect a cliff edge."));
        assert!(senses.contains(
            "Cliff sensor levels are left 0.10, front-left 0.80, front-right 0.40, right 0.20."
        ));
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

        let prompt =
            build_combobulator_prompt(&now, &ExperienceLatent::default(), &[], "I remember Tim.");

        assert!(prompt.contains("Timeline evidence:"));
        assert!(prompt.contains("[250 ms observed]"));
        assert!(prompt.contains("Distill what matters, not what the records said."));
        assert!(prompt.contains("prior combobulation summaries looping back as sensation"));
        assert!(prompt.contains("CONTEXT FRAME"));
        assert!(prompt.contains("I hear: hello there"));
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
