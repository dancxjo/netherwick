pub const IMAGE_CAPTION_PROMPT: &str = "Describe only what you see from your viewpoint. Start from the fact that this is your own vision looking out, so the first person should mean phrases like \"I see...\" or \"in front of me,\" not that visible people, faces, hands, eyes, or bodies are yours. You may use more than one sentence when the visible scene needs fuller description, but stay grounded in visible evidence and do not speculate beyond what can be seen. Do not interpret this as an image; interpret it as the machine's own live view. When looking out, one does not see oneself: anyone you see is most likely someone you're looking at, not yourself, unless you're clearly looking in a mirror or reflection. Describe visible people in third person, as someone in front of you.";

const SENSOR_GROUNDING_RULES: &str = "Describe the real-world scene or event, not the sensor stream. Interpret images, audio, motion, location, body state, memory-derived entries, and other sensor-derived entries as the robot's own vision, hearing, body sense, position sense, and memory sense, not as media files or external sensor artifacts. Convert raw body data into feeling-centered first-person interpretations from inside the robot: say things like \"I feel steady,\" \"I feel the floor fall away ahead,\" or \"my body feels blocked,\" instead of naming raw flags, sensor booleans, channel levels, or detector states. Do not summarize the amount, density, cadence, or mix of input modalities as if that were the situation. Repeated frames, repeated detections, image embeddings, pending audio clips, and heartbeat-like status records are usually evidence to compress or ignore, not events to report. If people are visible, do not assume any visible person is me unless the vision is clearly a mirror or reflection. If the evidence does not reveal what is happening, say that I cannot tell what is happening yet. Do not infer emotional tone or words like chaotic, intense, overwhelming, anxious, or ominous from sensor volume alone.";

const COMBOBULATOR_DISTILLATION_RULES: &str = "Distill what matters, not what the records said. Treat the entries as fragmentary, possibly contradictory, fleeting evidence about the actual situation, not as the topic to describe. Try to infer what is going on in the real world from those fragments. Sort meaning by time: occurred time first, observed time second. Consume the timeline in order; do not group by faculty or source. When related entries describe raw audio and the transcript derived from it, treat them as one real-world event. Some entries may be prior combobulation summaries looping back as impressions; use those only as provisional, possibly stale self-context, not as fresh external evidence. Do not say that you are observing a timeline, records, recordings, sensor streams, previous summaries, or a shift in conversation. Compress repeated low-level records into the real-world gist; do not enumerate ids, hashes, timestamps, edges, or detections unless they are the point.";

const LIVE_EVENT_RULES: &str = "Live events may arrive while generation is happening. Treat them as observations from outside. Do not assume a human is currently present or addressing me; there may be nobody nearby. Clock and status events help track timing, pauses, and elapsed time, but do not narrate every tick, quiet moment, or idle thought.";
const CHIRP_PATTERN_PROMPT: &str = "\
- confirm: notes 79,84,79; bright, decisive confirmation\n\
- warning: notes 79,75; rejection, warning, or nope\n\
- hello: notes 72,76,79; ascending greeting\n\
- goodbye: notes 79,76,72; descending farewell\n\
- curious: notes 72,76,74; unresolved question or interest\n\
- idea: notes 76,81,84; rising idea or realization\n\
- goal_acquired: notes 72,79,84,91; little fanfare for finding a goal\n\
- searching: notes 72,74,76,74; gentle oscillation while looking\n\
- saw_something: notes 84,91; quick widened-attention signal\n\
- surprise: notes 72,84; big leap for surprise\n\
- learned: notes 74,79,83; settled memory or learning\n\
- person_recognized: notes 76,79,84,79; warm arch for a recognized person\n\
- object_recognized: notes 79,84,76; recognition motif for an object\n\
- place_recognized: notes 79,84,72; recognition motif for a place\n\
- didnt_understand: notes 79,81,78; crooked ending for confusion\n\
- docking: notes 67,72,76,79; climbing toward home\n\
- charging_started: notes 60,67,72; solid foundation for charging contact\n\
- sleep: notes 79,76,72,67; gentle descent into rest\n\
- wake: notes 67,72,79; upward stretch into activity";
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
    fn enhanced_cognition_available(&self) -> bool {
        true
    }

    fn enhanced_cognition_unavailable_reason(&self) -> Option<&str> {
        None
    }

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
    fn enhanced_cognition_available(&self) -> bool {
        false
    }

    fn enhanced_cognition_unavailable_reason(&self) -> Option<&str> {
        Some("enhanced language service is disabled")
    }

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
    /// Ollama context-window bound. `None` leaves the model/server default intact.
    pub num_ctx: Option<u32>,
    /// Ollama generated-token bound. `None` leaves the model/server default intact.
    pub num_predict: Option<i32>,
    /// Ollama CPU-thread bound. `None` leaves scheduling to the model/server.
    pub num_thread: Option<u32>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        let endpoint = std::env::var("PETE_LLM_ENDPOINT")
            .or_else(|_| std::env::var("OLLAMA_HOST"))
            .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
        let agent_model = std::env::var("PETE_LLM_MODEL")
            .or_else(|_| std::env::var("OLLAMA_MODEL"))
            .unwrap_or_else(|_| "llama3.2".to_string());
        let vision_model = std::env::var("PETE_VISION_MODEL")
            .or_else(|_| std::env::var("OLLAMA_VISION_MODEL"))
            .unwrap_or_else(|_| "gemma4".to_string());
        let embedding_model = std::env::var("PETE_EMBEDDING_MODEL")
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
            num_ctx: None,
            num_predict: None,
            num_thread: None,
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
