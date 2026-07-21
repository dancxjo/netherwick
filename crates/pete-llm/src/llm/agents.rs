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
    fn enhanced_cognition_available(&self) -> bool {
        matches!(self, Self::Ollama(_))
    }

    fn enhanced_cognition_unavailable_reason(&self) -> Option<&str> {
        matches!(self, Self::Disabled(_)).then_some("enhanced language service is disabled")
    }

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
            options: OllamaGenerateOptions::from(&self.config),
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
    #[serde(skip_serializing_if = "Option::is_none")]
    num_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_thread: Option<u32>,
}

impl From<&LlmConfig> for OllamaGenerateOptions {
    fn from(config: &LlmConfig) -> Self {
        Self {
            temperature: config.temperature,
            num_ctx: config.num_ctx,
            num_predict: config.num_predict,
            num_thread: config.num_thread,
        }
    }
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
