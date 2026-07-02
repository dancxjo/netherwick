use anyhow::{Context, Result};
use async_trait::async_trait;
use base64::Engine;
use chrono::{Local, SecondsFormat, TimeZone};
use futures_util::StreamExt;
use image::ImageEncoder;
use netherwick_actions::{
    ActionPrimitive, ApproachTarget, ChirpPattern, ExploreStyle, InspectTarget, TurnDir,
};
use netherwick_experience::{EmbodiedContext, ExperienceLatent, FuturePrediction, Impression};
use netherwick_now::{
    EyeFrame, EyeFrameFormat, LlmSense, Now, ReignSense, VectorArtifact,
    IMAGE_DESCRIPTION_VECTOR_COLLECTION, SCENE_VECTOR_COLLECTION,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Cursor;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;
use uuid::Uuid;

pub const IMAGE_CAPTION_PROMPT: &str = "Describe only what you see from your viewpoint. Start from the fact that this is your own vision looking out, so the first person should mean phrases like \"I see...\" or \"in front of me,\" not that visible people, faces, hands, eyes, or bodies are yours. You may use more than one sentence when the visible scene needs fuller description, but stay grounded in visible evidence and do not speculate beyond what can be seen. Do not interpret this as an image; interpret it as the machine's own live view. When looking out, one does not see oneself: anyone you see is most likely someone you're looking at, not yourself, unless you're clearly looking in a mirror or reflection. Describe visible people in third person, as someone in front of you.";

const SENSOR_GROUNDING_RULES: &str = "Describe the real-world scene or event, not the sensor stream. Interpret images, audio, motion, location, body state, memory-derived entries, and other sensor-derived entries as the robot's own vision, hearing, body sense, position sense, and memory sense, not as media files or external sensor artifacts. Convert raw body data into feeling-centered first-person interpretations from inside the robot: say things like \"I feel steady,\" \"I feel the floor fall away ahead,\" or \"my body feels blocked,\" instead of naming raw flags, sensor booleans, channel levels, or detector states. Do not summarize the amount, density, cadence, or mix of input modalities as if that were the situation. Repeated frames, repeated detections, image embeddings, pending audio clips, and heartbeat-like status records are usually evidence to compress or ignore, not events to report. If people are visible, do not assume any visible person is me unless the vision is clearly a mirror or reflection. If the evidence does not reveal what is happening, say that I cannot tell what is happening yet. Do not infer emotional tone or words like chaotic, intense, overwhelming, anxious, or ominous from sensor volume alone.";

const COMBOBULATOR_DISTILLATION_RULES: &str = "Distill what matters, not what the records said. Treat the entries as fragmentary, possibly contradictory, fleeting evidence about the actual situation, not as the topic to describe. Try to infer what is going on in the real world from those fragments. Sort meaning by time: occurred time first, observed time second. Consume the timeline in order; do not group by faculty or source. When related entries describe raw audio and the transcript derived from it, treat them as one real-world event. Some entries may be prior combobulation summaries looping back as impressions; use those only as provisional, possibly stale self-context, not as fresh external evidence. Do not say that you are observing a timeline, records, recordings, sensor streams, previous summaries, or a shift in conversation. Compress repeated low-level records into the real-world gist; do not enumerate ids, hashes, timestamps, edges, or detections unless they are the point.";

const LIVE_EVENT_RULES: &str = "Live events may arrive while generation is happening. Treat them as observations from outside. Do not assume a human is currently present or addressing me; there may be nobody nearby. Clock and status events help track timing, pauses, and elapsed time, but do not narrate every tick, quiet moment, or idle thought.";
const STRICT_JSON_RESPONSE_RULES: &str = "\n\nFINAL OUTPUT REQUIREMENT (MANDATORY):\nReturn exactly one JSON object and nothing else.\nDo not include markdown fences, prose, explanations, preambles, or trailing text.\nOutput must start with '{' and end with '}'.\nIf unsure, still emit the best-effort valid JSON object matching the schema.";
const COMBOBULATOR_CLUSTER_GAP_MS: u64 = 1_000;
const COMBOBULATOR_MIN_INTERVAL_MS: u64 = 2_000;
const AGENT_MIN_INTERVAL_MS: u64 = 1_500;
const MILLIS_PER_SECOND: f64 = 1_000.0;
const DEFAULT_OLLAMA_TIMEOUT_MS: u64 = 300_000;
const PROMPT_UUID_OPTION_COUNT: usize = 5;

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
        let (tx, _rx) = broadcast::channel(8_192);
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewTargetKind {
    Feature,
    Cluster,
    BindingCandidate,
    BindingEdge,
    Hypothesis,
    Constellation,
    Prediction,
    ActionOutcome,
    TrainingExample,
}

impl Default for ReviewTargetKind {
    fn default() -> Self {
        Self::TrainingExample
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmScientificReview {
    pub id: String,
    pub t_ms: u64,
    pub target_id: String,
    pub target_kind: ReviewTargetKind,
    pub critique: Option<String>,
    pub counterfactuals: Vec<CounterfactualAction>,
    pub suggested_tests: Vec<LlmSuggestedTest>,
    pub suspicious_training_examples: Vec<LlmTrainingWarning>,
    pub label_proposals: Vec<LlmLabelProposal>,
    pub human_review_prompts: Vec<String>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmSuggestedTest {
    pub action: Option<ActionPrimitive>,
    pub question: String,
    pub expected_observation: String,
    pub disconfirming_observation: String,
    pub risk_note: Option<String>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmTrainingWarning {
    pub example_id: String,
    pub reason: String,
    pub severity: f32,
    pub suspected_issue: String,
    pub supporting_evidence: Vec<String>,
    pub missing_evidence: Vec<String>,
    pub suggested_fix: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmLabelProposal {
    pub example_id: String,
    pub proposed_label: String,
    pub rationale: String,
    pub confidence: f32,
    pub requires_human_review: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmReviewRequest {
    pub t_ms: u64,
    pub target_id: String,
    pub target_kind: ReviewTargetKind,
    pub observed_evidence: Vec<String>,
    pub candidate_explanation: Option<String>,
    pub current_confidence: f32,
    pub known_contradictions: Vec<String>,
    pub missing_evidence: Vec<String>,
    pub available_actions: Vec<ActionPrimitive>,
    pub safety_state: Option<String>,
    pub training_examples: Vec<LlmTrainingExampleEvidence>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmTrainingExampleEvidence {
    pub example_id: String,
    pub behavior: String,
    pub input_summary: String,
    pub expected_summary: String,
    pub actual_summary: Option<String>,
    pub reward: Option<f32>,
    pub source: Option<String>,
    pub contradictions: Vec<String>,
    pub missing_evidence: Vec<String>,
}

impl LlmReviewRequest {
    pub fn training_example(t_ms: u64, example: LlmTrainingExampleEvidence) -> Self {
        Self {
            t_ms,
            target_id: example.example_id.clone(),
            target_kind: ReviewTargetKind::TrainingExample,
            observed_evidence: vec![
                format!("behavior: {}", example.behavior),
                format!("input: {}", example.input_summary),
                format!("expected label: {}", example.expected_summary),
            ],
            candidate_explanation: Some("candidate training row for model learning".to_string()),
            current_confidence: 0.5,
            known_contradictions: example.contradictions.clone(),
            missing_evidence: example.missing_evidence.clone(),
            available_actions: Vec::new(),
            safety_state: None,
            training_examples: vec![example],
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

    async fn scientific_review(
        &mut self,
        request: &LlmReviewRequest,
    ) -> Result<Option<LlmScientificReview>>;
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

    async fn scientific_review(
        &mut self,
        _request: &LlmReviewRequest,
    ) -> Result<Option<LlmScientificReview>> {
        Ok(None)
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
    pub vision_model: Option<String>,
    pub embedding_model: Option<String>,
    pub enrich_live_images: bool,
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
        let vision_model = std::env::var("NETHERWICK_VISION_MODEL")
            .or_else(|_| std::env::var("OLLAMA_VISION_MODEL"))
            .unwrap_or_else(|_| "gemma4".to_string());
        let embedding_model = std::env::var("NETHERWICK_EMBEDDING_MODEL")
            .or_else(|_| std::env::var("OLLAMA_EMBEDDING_MODEL"))
            .unwrap_or_else(|_| "embeddinggemma".to_string());
        Self {
            provider: LlmProvider::Ollama,
            allow_commands: true,
            allow_teaching: true,
            endpoint,
            agent_model,
            combobulator_model: None,
            vision_model: Some(vision_model),
            embedding_model: Some(embedding_model),
            enrich_live_images: true,
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
        if config.vision_model.is_none() {
            config.vision_model = Some("gemma4".to_string());
        }
        if config.embedding_model.is_none() {
            config.embedding_model = Some("embeddinggemma".to_string());
        }
        Ok(config)
    }
}

#[derive(Clone, Debug)]
pub struct LiveImageEnricher {
    config: LlmConfig,
    client: Client,
    last_frame_key: Option<LiveImageFrameKey>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LiveImageFrameKey {
    captured_at_ms: u64,
    width: u32,
    height: u32,
    format: String,
    byte_len: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LiveImageEnrichment {
    pub description: String,
    pub image_description_vector: VectorArtifact,
    pub scene_vector: VectorArtifact,
}

impl LiveImageEnricher {
    pub fn new(config: LlmConfig) -> Result<Option<Self>> {
        if config.provider != LlmProvider::Ollama || !config.enrich_live_images {
            return Ok(None);
        }
        let client = Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .build()
            .context("failed to build ollama image enricher http client")?;
        Ok(Some(Self {
            config,
            client,
            last_frame_key: None,
        }))
    }

    pub async fn enrich_latest(&mut self, frame: &EyeFrame) -> Result<Option<LiveImageEnrichment>> {
        let key = LiveImageFrameKey::from(frame);
        if self.last_frame_key.as_ref() == Some(&key) {
            return Ok(None);
        }
        self.last_frame_key = Some(key);

        let image_base64 = encode_eye_frame_png_base64(frame)?;
        let vision_model = self
            .config
            .vision_model
            .as_deref()
            .filter(|model| !model.trim().is_empty())
            .context("live image enrichment requires vision_model")?;
        let embedding_model = self
            .config
            .embedding_model
            .as_deref()
            .filter(|model| !model.trim().is_empty())
            .context("live image enrichment requires embedding_model")?;
        let description = self.describe_image(vision_model, image_base64).await?;
        let embedding = self.embed_text(embedding_model, &description).await?;
        let source_frame_id = live_image_source_frame_id(frame);
        let point_id = stable_uuid_for_text(&format!(
            "{}:{vision_model}:{embedding_model}:{description}",
            source_frame_id
        ))
        .to_string();
        let image_description_vector = VectorArtifact::new(
            IMAGE_DESCRIPTION_VECTOR_COLLECTION,
            point_id.clone(),
            embedding.clone(),
        )
        .with_model(embedding_model.to_string())
        .with_source_id(format!("image-description:{source_frame_id}"))
        .with_source_frame_id(source_frame_id.clone())
        .with_occurred_at_ms(frame.captured_at_ms);
        let scene_vector = VectorArtifact::new(SCENE_VECTOR_COLLECTION, point_id, embedding)
            .with_model(embedding_model.to_string())
            .with_source_id(format!("image-description:{source_frame_id}"))
            .with_source_frame_id(source_frame_id)
            .with_occurred_at_ms(frame.captured_at_ms);
        Ok(Some(LiveImageEnrichment {
            description,
            image_description_vector,
            scene_vector,
        }))
    }

    async fn describe_image(&self, model: &str, image_base64: String) -> Result<String> {
        let request = OllamaChatRequest {
            model,
            stream: false,
            messages: vec![OllamaChatMessage {
                role: "user",
                content: IMAGE_CAPTION_PROMPT,
                images: vec![image_base64],
            }],
            options: OllamaGenerateOptions {
                temperature: self.config.temperature,
            },
        };
        let response: OllamaChatResponse = self
            .client
            .post(format!(
                "{}/api/chat",
                self.config.endpoint.trim_end_matches('/')
            ))
            .json(&request)
            .send()
            .await
            .context("failed to reach ollama for image description")?
            .error_for_status()
            .context("ollama image description returned an error")?
            .json()
            .await
            .context("failed to decode ollama image description")?;
        let text = response.message.content.trim().to_string();
        if text.is_empty() {
            anyhow::bail!("ollama image description was empty");
        }
        Ok(text)
    }

    async fn embed_text(&self, model: &str, text: &str) -> Result<Vec<f32>> {
        let request = OllamaEmbedRequest { model, input: text };
        let response: OllamaEmbedResponse = self
            .client
            .post(format!(
                "{}/api/embed",
                self.config.endpoint.trim_end_matches('/')
            ))
            .json(&request)
            .send()
            .await
            .context("failed to reach ollama for text embedding")?
            .error_for_status()
            .context("ollama text embedding returned an error")?
            .json()
            .await
            .context("failed to decode ollama text embedding")?;
        let embedding = response
            .embeddings
            .into_iter()
            .next()
            .context("ollama returned no embeddings")?;
        if embedding.is_empty() {
            anyhow::bail!("ollama returned an empty embedding");
        }
        Ok(embedding)
    }
}

impl From<&EyeFrame> for LiveImageFrameKey {
    fn from(frame: &EyeFrame) -> Self {
        Self {
            captured_at_ms: frame.captured_at_ms,
            width: frame.width,
            height: frame.height,
            format: format!("{:?}", frame.format),
            byte_len: frame.bytes.len(),
        }
    }
}

fn encode_eye_frame_png_base64(frame: &EyeFrame) -> Result<String> {
    let (rgb, color_type) = match &frame.format {
        EyeFrameFormat::Rgb8 => (frame.bytes.clone(), image::ColorType::Rgb8),
        EyeFrameFormat::Bgr8 => {
            let mut rgb = Vec::with_capacity(frame.bytes.len());
            for pixel in frame.bytes.chunks_exact(3) {
                rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
            }
            (rgb, image::ColorType::Rgb8)
        }
        EyeFrameFormat::Gray8 => (frame.bytes.clone(), image::ColorType::L8),
        EyeFrameFormat::Mjpeg => {
            let image = image::load_from_memory(&frame.bytes)
                .context("failed to decode MJPEG eye frame")?
                .to_rgb8();
            (image.into_raw(), image::ColorType::Rgb8)
        }
        EyeFrameFormat::Yuyv422
        | EyeFrameFormat::Uyvy422
        | EyeFrameFormat::BayerGrbg8
        | EyeFrameFormat::BayerRggb8
        | EyeFrameFormat::BayerBggr8
        | EyeFrameFormat::BayerGbrg8
        | EyeFrameFormat::Unknown(_) => {
            anyhow::bail!("unsupported live image frame format {:?}", frame.format)
        }
    };
    let expected_len = match color_type {
        image::ColorType::Rgb8 => frame.width as usize * frame.height as usize * 3,
        image::ColorType::L8 => frame.width as usize * frame.height as usize,
        _ => unreachable!("only rgb8 and gray8 are encoded"),
    };
    if frame.width == 0 || frame.height == 0 || rgb.len() != expected_len {
        anyhow::bail!(
            "invalid live image frame dimensions {}x{} for {} bytes",
            frame.width,
            frame.height,
            rgb.len()
        );
    }

    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(Cursor::new(&mut png))
        .write_image(&rgb, frame.width, frame.height, color_type.into())
        .context("failed to encode live image frame as PNG")?;
    Ok(base64::engine::general_purpose::STANDARD.encode(png))
}

fn live_image_source_frame_id(frame: &EyeFrame) -> String {
    if frame.captured_at_ms > 0 {
        format!("eye-frame-{}", frame.captured_at_ms)
    } else {
        format!(
            "eye-frame-{}",
            stable_uuid_for_text(&format!(
                "{}x{}:{:?}:{}",
                frame.width,
                frame.height,
                frame.format,
                frame.bytes.len()
            ))
        )
    }
}

fn stable_uuid_for_text(text: &str) -> Uuid {
    let mut first = DefaultHasher::new();
    "netherwick-live-image-a".hash(&mut first);
    text.hash(&mut first);
    let mut second = DefaultHasher::new();
    "netherwick-live-image-b".hash(&mut second);
    text.hash(&mut second);
    let mut bytes = [0_u8; 16];
    bytes[..8].copy_from_slice(&first.finish().to_be_bytes());
    bytes[8..].copy_from_slice(&second.finish().to_be_bytes());
    Uuid::from_bytes(bytes)
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

    async fn scientific_review(
        &mut self,
        request: &LlmReviewRequest,
    ) -> Result<Option<LlmScientificReview>> {
        match self {
            Self::Disabled(agent) => agent.scientific_review(request).await,
            Self::Ollama(agent) => agent.scientific_review(request).await,
        }
    }
}

pub struct OllamaLlmAgent {
    config: LlmConfig,
    client: Client,
    last_combobulate_ms: Option<u64>,
    last_combobulation: Option<Combobulation>,
    last_agent_tick_ms: Option<u64>,
    last_tick_result: LlmTickResult,
}

impl OllamaLlmAgent {
    pub fn new(config: LlmConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_millis(config.timeout_ms))
            .build()
            .context("failed to build ollama http client")?;
        Ok(Self {
            config,
            client,
            last_combobulate_ms: None,
            last_combobulation: None,
            last_agent_tick_ms: None,
            last_tick_result: LlmTickResult::default(),
        })
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
            prompt: Some(prompt_preview(&prompt, 1_500)),
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

fn prompt_preview(text: &str, max_chars: usize) -> String {
    let mut iter = text.chars();
    let preview = iter.by_ref().take(max_chars).collect::<String>();
    if iter.next().is_some() {
        format!("{preview}... [truncated]")
    } else {
        preview
    }
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
    let mut live_delta = String::new();
    if !chunk.response.is_empty() {
        body.push_str(&chunk.response);
        if !chunk.response.trim().is_empty() {
            live_delta = chunk.response;
        }
    }
    if live_delta.is_empty() && !chunk.thinking.is_empty() {
        live_delta = chunk.thinking;
    }
    if !live_delta.is_empty() {
        emit_llm_stream(LlmStreamEvent {
            id: stream_id,
            t_ms: wall_now_ms(),
            phase: LlmStreamPhase::Delta,
            purpose: purpose.to_string(),
            provider: "ollama".to_string(),
            model: model.to_string(),
            prompt: None,
            delta: Some(live_delta),
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
        if let Some(previous_ms) = self.last_combobulate_ms {
            if now.t_ms.saturating_sub(previous_ms) < COMBOBULATOR_MIN_INTERVAL_MS {
                return Ok(self.last_combobulation.clone());
            }
        }
        self.last_combobulate_ms = Some(now.t_ms);
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
            Ok(reply) => {
                let combobulation = Combobulation {
                    summary: reply.summary.trim().to_string(),
                    confidence: reply.confidence.clamp(0.0, 1.0),
                };
                self.last_combobulation = Some(combobulation.clone());
                Ok(Some(combobulation))
            }
            Err(_) => {
                let fallback = heuristic_combobulation(now, recall_summary);
                self.last_combobulation = Some(fallback.clone());
                Ok(Some(fallback))
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
        if !self.config.allow_commands && !self.config.allow_teaching {
            return Ok(LlmTickResult::default());
        }
        if let Some(previous_ms) = self.last_agent_tick_ms {
            if now.t_ms.saturating_sub(previous_ms) < AGENT_MIN_INTERVAL_MS {
                return Ok(self.last_tick_result.clone());
            }
        }
        self.last_agent_tick_ms = Some(now.t_ms);

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

        let tick_result = LlmTickResult {
            sense: LlmSense {
                schema_version: 1,
                command_summary: conscious_command.as_ref().map(|cmd| cmd.summary.clone()),
                critique,
                confidence,
            },
            conscious_command,
            decision: Some(decision),
            teaching,
        };
        self.last_tick_result = tick_result.clone();
        Ok(tick_result)
    }

    async fn scientific_review(
        &mut self,
        request: &LlmReviewRequest,
    ) -> Result<Option<LlmScientificReview>> {
        let prompt = build_scientific_review_prompt(request);
        let reply = self
            .generate_json::<ScientificReviewReply>(
                "scientific_review",
                &self.config.agent_model,
                prompt,
            )
            .await?;
        Ok(Some(scientific_review_from_reply(request, reply)))
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
    #[serde(default)]
    response: String,
    #[serde(default)]
    thinking: String,
}

#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: Vec<OllamaChatMessage<'a>>,
    stream: bool,
    options: OllamaGenerateOptions,
}

#[derive(Serialize)]
struct OllamaChatMessage<'a> {
    role: &'a str,
    content: &'a str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    images: Vec<String>,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaChatResponseMessage,
}

#[derive(Deserialize)]
struct OllamaChatResponseMessage {
    #[serde(default)]
    content: String,
}

#[derive(Serialize)]
struct OllamaEmbedRequest<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    #[serde(default)]
    embeddings: Vec<Vec<f32>>,
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

#[derive(Deserialize)]
struct ScientificReviewReply {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    critique: Option<String>,
    #[serde(default)]
    counterfactuals: Vec<CounterfactualSpec>,
    #[serde(default)]
    suggested_tests: Vec<SuggestedTestSpec>,
    #[serde(default)]
    suspicious_training_examples: Vec<TrainingWarningSpec>,
    #[serde(default)]
    label_proposals: Vec<LabelProposalSpec>,
    #[serde(default)]
    human_review_prompts: Vec<String>,
    #[serde(default)]
    confidence: f32,
}

#[derive(Deserialize)]
struct SuggestedTestSpec {
    #[serde(default)]
    action: Option<ActionSpec>,
    #[serde(default)]
    question: String,
    #[serde(default)]
    expected_observation: String,
    #[serde(default)]
    disconfirming_observation: String,
    #[serde(default)]
    risk_note: Option<String>,
    #[serde(default)]
    confidence: f32,
}

#[derive(Deserialize)]
struct TrainingWarningSpec {
    #[serde(default)]
    example_id: String,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    severity: f32,
    #[serde(default)]
    suspected_issue: String,
    #[serde(default)]
    supporting_evidence: Vec<String>,
    #[serde(default)]
    missing_evidence: Vec<String>,
    #[serde(default)]
    suggested_fix: Option<String>,
}

#[derive(Deserialize)]
struct LabelProposalSpec {
    #[serde(default)]
    example_id: String,
    #[serde(default)]
    proposed_label: String,
    #[serde(default)]
    rationale: String,
    #[serde(default)]
    confidence: f32,
    #[serde(default)]
    requires_human_review: bool,
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
    let uuid_options = render_prompt_uuid_options();
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
If any output field calls for a new UUID or id, choose one of these exact UUID options and do not invent your own:\n\
{}\n\n\
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
        uuid_options,
        now.t_ms,
        z.confidence,
        z.prediction_error,
        recall_summary,
        embodied,
        timeline,
        futures
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
    let uuid_options = render_prompt_uuid_options();
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
If any output field calls for a new UUID or id, choose one of these exact UUID options and do not invent your own:\n\
{}\n\n\
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
        uuid_options,
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

pub fn build_scientific_review_prompt(request: &LlmReviewRequest) -> String {
    let available_actions = request
        .available_actions
        .iter()
        .filter_map(|action| serde_json::to_string(&action_spec_json(action)).ok())
        .collect::<Vec<_>>();
    let training_examples = request
        .training_examples
        .iter()
        .map(|example| {
            serde_json::json!({
                "example_id": example.example_id,
                "behavior": example.behavior,
                "input_summary": example.input_summary,
                "expected_summary": example.expected_summary,
                "actual_summary": example.actual_summary,
                "reward": example.reward,
                "source": example.source,
                "contradictions": example.contradictions,
                "missing_evidence": example.missing_evidence,
            })
        })
        .collect::<Vec<_>>();
    format!(
        "You are Netherwick's scientific critic, not its source of truth.\n\
Inspect the target below and produce skeptical, evidence-grounded review JSON only.\n\
You may identify weak evidence, possible fused clusters, suspicious training rows, plausible labels, counterfactual actions, and tests that would reduce uncertainty.\n\
You must not declare identity as certain, override safety, merge entities, accept bindings, invent sensor evidence, pretend movement happened, or mark a training row as true.\n\
Motion actions are suggestions only; downstream safety and admission systems decide what can happen.\n\
Be compact, explicit about uncertainty, and prefer \"plausible but unproven\" language when evidence is incomplete.\n\
Return JSON only with this schema:\n\
{{\n\
  \"id\":\"optional review id\",\n\
  \"critique\":\"optional critique grounded in evidence\",\n\
  \"counterfactuals\":[{{\"instead_of\":null,\"proposed\":{{\"kind\":\"inspect\",\"target\":\"novelty\"}},\"reason\":\"...\",\"weight\":0.5}}],\n\
  \"suggested_tests\":[{{\"action\":{{\"kind\":\"inspect\",\"target\":\"novelty\"}},\"question\":\"...\",\"expected_observation\":\"...\",\"disconfirming_observation\":\"...\",\"risk_note\":null,\"confidence\":0.5}}],\n\
  \"suspicious_training_examples\":[{{\"example_id\":\"...\",\"reason\":\"...\",\"severity\":0.5,\"suspected_issue\":\"unsupported_label\",\"supporting_evidence\":[\"...\"],\"missing_evidence\":[\"...\"],\"suggested_fix\":\"human review\"}}],\n\
  \"label_proposals\":[{{\"example_id\":\"...\",\"proposed_label\":\"...\",\"rationale\":\"...\",\"confidence\":0.5,\"requires_human_review\":true}}],\n\
  \"human_review_prompts\":[\"...\"],\n\
  \"confidence\":0.0\n\
}}\n\n\
REVIEW TARGET\n\
- target_id: {}\n\
- target_kind: {:?}\n\
- review_time_ms: {}\n\
- candidate_explanation: {}\n\
- current_confidence: {:.2}\n\
- safety_state: {}\n\n\
OBSERVED EVIDENCE\n{}\n\n\
KNOWN CONTRADICTIONS\n{}\n\n\
MISSING EVIDENCE\n{}\n\n\
AVAILABLE ACTIONS JSON\n{}\n\n\
TRAINING EXAMPLES JSON\n{}\n",
        prompt_json_string(&request.target_id),
        request.target_kind,
        request.t_ms,
        prompt_json_string(request.candidate_explanation.as_deref().unwrap_or("none")),
        request.current_confidence.clamp(0.0, 1.0),
        prompt_json_string(request.safety_state.as_deref().unwrap_or("unknown")),
        prompt_lines(&request.observed_evidence),
        prompt_lines(&request.known_contradictions),
        prompt_lines(&request.missing_evidence),
        if available_actions.is_empty() {
            "- none".to_string()
        } else {
            available_actions
                .into_iter()
                .map(|line| format!("- {line}"))
                .collect::<Vec<_>>()
                .join("\n")
        },
        serde_json::to_string_pretty(&training_examples)
            .unwrap_or_else(|_| "[]".to_string())
    )
}

fn prompt_lines(lines: &[String]) -> String {
    if lines.is_empty() {
        return "- none".to_string();
    }
    lines
        .iter()
        .map(|line| format!("- {}", prompt_json_string(line)))
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_prompt_uuid_options() -> String {
    (0..PROMPT_UUID_OPTION_COUNT)
        .map(|_| format!("- {}", Uuid::new_v4()))
        .collect::<Vec<_>>()
        .join("\n")
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
        netherwick_actions::ReignCommand::Chirp { pattern } => {
            format!("Chirp {:?}", pattern)
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

pub fn parse_scientific_review_json(
    request: &LlmReviewRequest,
    text: &str,
) -> Result<LlmScientificReview> {
    let json = extract_json_object(text).unwrap_or_else(|| text.trim().to_string());
    let reply: ScientificReviewReply =
        serde_json::from_str(&json).context("failed to parse llm scientific review")?;
    Ok(scientific_review_from_reply(request, reply))
}

fn scientific_review_from_reply(
    request: &LlmReviewRequest,
    reply: ScientificReviewReply,
) -> LlmScientificReview {
    let default_id = stable_uuid_for_text(&format!(
        "llm-review:{}:{:?}:{}",
        request.target_id, request.target_kind, request.t_ms
    ))
    .to_string();
    LlmScientificReview {
        id: reply
            .id
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or(default_id),
        t_ms: request.t_ms,
        target_id: request.target_id.clone(),
        target_kind: request.target_kind.clone(),
        critique: reply
            .critique
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
        counterfactuals: reply
            .counterfactuals
            .into_iter()
            .filter_map(parse_counterfactual_spec)
            .collect(),
        suggested_tests: reply
            .suggested_tests
            .into_iter()
            .map(|test| LlmSuggestedTest {
                action: test.action.and_then(parse_action_spec),
                question: test.question,
                expected_observation: test.expected_observation,
                disconfirming_observation: test.disconfirming_observation,
                risk_note: test
                    .risk_note
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty()),
                confidence: test.confidence.clamp(0.0, 1.0),
            })
            .collect(),
        suspicious_training_examples: reply
            .suspicious_training_examples
            .into_iter()
            .map(|warning| LlmTrainingWarning {
                example_id: if warning.example_id.trim().is_empty() {
                    request.target_id.clone()
                } else {
                    warning.example_id
                },
                reason: warning.reason,
                severity: warning.severity.clamp(0.0, 1.0),
                suspected_issue: warning.suspected_issue,
                supporting_evidence: warning.supporting_evidence,
                missing_evidence: warning.missing_evidence,
                suggested_fix: warning
                    .suggested_fix
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty()),
            })
            .collect(),
        label_proposals: reply
            .label_proposals
            .into_iter()
            .map(|proposal| LlmLabelProposal {
                example_id: if proposal.example_id.trim().is_empty() {
                    request.target_id.clone()
                } else {
                    proposal.example_id
                },
                proposed_label: proposal.proposed_label,
                rationale: proposal.rationale,
                confidence: proposal.confidence.clamp(0.0, 1.0),
                requires_human_review: proposal.requires_human_review,
            })
            .collect(),
        human_review_prompts: reply
            .human_review_prompts
            .into_iter()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .collect(),
        confidence: reply.confidence.clamp(0.0, 1.0),
    }
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
    fn prompts_offer_generated_uuid_options_instead_of_asking_llm_to_invent_ids() {
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

        assert!(prompt.contains("choose one of these exact UUID options"));
        assert!(prompt.contains("do not invent your own"));

        let options = prompt
            .lines()
            .skip_while(|line| !line.contains("choose one of these exact UUID options"))
            .skip(1)
            .take_while(|line| line.starts_with("- "))
            .map(|line| line.trim_start_matches("- "))
            .collect::<Vec<_>>();

        assert_eq!(options.len(), PROMPT_UUID_OPTION_COUNT);
        for option in options {
            Uuid::parse_str(option).expect("prompt UUID option should be valid");
        }
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
    fn scientific_review_prompt_frames_training_rows_as_uncertain_evidence() {
        let request = LlmReviewRequest::training_example(
            42,
            LlmTrainingExampleEvidence {
                example_id: "danger-row-7".to_string(),
                behavior: "danger".to_string(),
                input_summary: "front range says clear; no bump flags".to_string(),
                expected_summary: "bump_risk=1.0".to_string(),
                actual_summary: Some("model predicted low bump risk".to_string()),
                reward: Some(-0.2),
                source: Some("world_outcome".to_string()),
                contradictions: vec!["no contact evidence supports bump label".to_string()],
                missing_evidence: vec!["no post-action body flags".to_string()],
            },
        );

        let prompt = build_scientific_review_prompt(&request);

        assert!(prompt.contains("scientific critic, not its source of truth"));
        assert!(prompt.contains("must not declare identity as certain"));
        assert!(prompt.contains("mark a training row as true"));
        assert!(prompt.contains("suspicious_training_examples"));
        assert!(prompt.contains("label_proposals"));
        assert!(prompt.contains("\"example_id\": \"danger-row-7\""));
        assert!(prompt.contains("no contact evidence supports bump label"));
        assert!(prompt.contains("AVAILABLE ACTIONS JSON\n- none"));
    }

    #[test]
    fn parse_scientific_review_json_clamps_and_reuses_existing_action_types() {
        let request = LlmReviewRequest::training_example(
            99,
            LlmTrainingExampleEvidence {
                example_id: "charge-row-3".to_string(),
                behavior: "charge".to_string(),
                input_summary: "battery dropping, charger not visible".to_string(),
                expected_summary: "charging_started=true".to_string(),
                ..LlmTrainingExampleEvidence::default()
            },
        );
        let review = parse_scientific_review_json(
            &request,
            r#"Here is JSON:
{
  "critique":"Label is plausible but unsupported by visible charger or battery delta.",
  "counterfactuals":[{"instead_of":null,"proposed":{"kind":"inspect","target":"charger"},"reason":"Look for missing charger evidence.","weight":1.5}],
  "suggested_tests":[{"action":{"kind":"inspect","target":"charger"},"question":"Is the charger visible?","expected_observation":"charger appears","disconfirming_observation":"no charger evidence","risk_note":"","confidence":-0.2}],
  "suspicious_training_examples":[{"example_id":"","reason":"Expected charging label lacks support.","severity":2.0,"suspected_issue":"unsupported_label","supporting_evidence":["charger not visible"],"missing_evidence":["battery delta"],"suggested_fix":"send to human review"}],
  "label_proposals":[{"example_id":"","proposed_label":"charging_started=unknown","rationale":"evidence is incomplete","confidence":0.7,"requires_human_review":true}],
  "human_review_prompts":["Check whether charging actually started."],
  "confidence":1.2
}"#,
        )
        .expect("scientific review json should parse");

        assert_eq!(review.t_ms, 99);
        assert_eq!(review.target_id, "charge-row-3");
        assert_eq!(review.target_kind, ReviewTargetKind::TrainingExample);
        assert_eq!(review.confidence, 1.0);
        assert_eq!(review.counterfactuals.len(), 1);
        assert_eq!(review.counterfactuals[0].weight, 1.0);
        assert_eq!(
            review.suggested_tests[0].action,
            Some(ActionPrimitive::Inspect {
                target: InspectTarget::Charger
            })
        );
        assert_eq!(review.suggested_tests[0].confidence, 0.0);
        assert_eq!(
            review.suspicious_training_examples[0].example_id,
            "charge-row-3"
        );
        assert_eq!(review.suspicious_training_examples[0].severity, 1.0);
        assert_eq!(review.label_proposals[0].confidence, 0.7);
        assert!(review.label_proposals[0].requires_human_review);
    }

    #[test]
    fn stream_line_uses_thinking_when_response_is_whitespace() {
        let mut rx = subscribe_llm_streams();
        while rx.try_recv().is_ok() {}

        let mut body = String::new();
        handle_ollama_stream_line(
            7,
            "combobulator",
            "gpt-oss:20b",
            r#"{"response":"\n","thinking":"hello","done":false}"#,
            &mut body,
        )
        .expect("stream line should parse");

        let event = next_stream_event_for_id(&mut rx, 7);
        assert_eq!(event.phase, LlmStreamPhase::Delta);
        assert_eq!(event.delta.as_deref(), Some("hello"));
        assert_eq!(body, "\n");
    }

    #[test]
    fn stream_line_prefers_response_when_non_whitespace() {
        let mut rx = subscribe_llm_streams();
        while rx.try_recv().is_ok() {}

        let mut body = String::new();
        handle_ollama_stream_line(
            8,
            "agent",
            "llama3.2",
            r#"{"response":"ok","thinking":"ignored","done":false}"#,
            &mut body,
        )
        .expect("stream line should parse");

        let event = next_stream_event_for_id(&mut rx, 8);
        assert_eq!(event.phase, LlmStreamPhase::Delta);
        assert_eq!(event.delta.as_deref(), Some("ok"));
        assert_eq!(body, "ok");
    }

    fn next_stream_event_for_id(
        rx: &mut tokio::sync::broadcast::Receiver<LlmStreamEvent>,
        id: u64,
    ) -> LlmStreamEvent {
        for _ in 0..64 {
            let event = rx.try_recv().expect("delta event should be emitted");
            if event.id == id {
                return event;
            }
        }
        panic!("delta event for stream {id} was not emitted");
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
            sensation_vectors: Vec::new(),
            impression_vectors: Vec::new(),
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
        assert!(prompt.contains("payload=image_bytes"));
        assert!(!prompt.contains("[0."));
    }

    #[test]
    fn image_caption_prompt_frames_live_vision_viewpoint() {
        assert!(IMAGE_CAPTION_PROMPT.contains("Describe only what you see from your viewpoint"));
        assert!(IMAGE_CAPTION_PROMPT.contains("your own vision looking out"));
        assert!(IMAGE_CAPTION_PROMPT.contains("not that visible people"));
        assert!(IMAGE_CAPTION_PROMPT.contains("the machine's own live view"));
        assert!(IMAGE_CAPTION_PROMPT.contains("When looking out, one does not see oneself"));
        assert!(
            IMAGE_CAPTION_PROMPT.contains("most likely someone you're looking at, not yourself")
        );
        assert!(IMAGE_CAPTION_PROMPT
            .contains("unless you're clearly looking in a mirror or reflection"));
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
