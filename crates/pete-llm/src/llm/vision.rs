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

const OLLAMA_SCENE_PROVIDER_ID: &str = "ollama.scene_description";

pub struct LiveImageCognition {
    supervisor: AsyncCognitionSupervisor,
    last_frame_key: Option<LiveImageFrameKey>,
    request_timeout_ms: u64,
}

#[derive(Clone, Debug, Default)]
pub struct LiveImageCognitionTick {
    pub registry: ProviderRegistrySnapshot,
    pub enrichment: Option<LiveImageEnrichment>,
    pub response: Option<RoutedResponse>,
}

impl LiveImageCognition {
    pub fn new(enricher: Option<LiveImageEnricher>) -> Self {
        let request_timeout_ms = enricher
            .as_ref()
            .map(|provider| provider.config.timeout_ms)
            .unwrap_or(1_000)
            .max(1);
        let mut router = CognitiveRouter::default();
        if let Some(enricher) = enricher {
            router.register(Box::new(enricher));
        }
        Self::from_router(router, request_timeout_ms)
    }

    pub fn from_router(router: CognitiveRouter, request_timeout_ms: u64) -> Self {
        Self {
            supervisor: AsyncCognitionSupervisor::new(router, wall_now_ms()),
            last_frame_key: None,
            request_timeout_ms: request_timeout_ms.max(1),
        }
    }

    pub fn registry_snapshot(&self) -> &ProviderRegistrySnapshot {
        self.supervisor.registry_snapshot()
    }

    /// Poll a completed request and enqueue at most one new frame. This never
    /// waits for provider I/O: unfinished work remains in the background while
    /// the organism tick proceeds with local structured beliefs.
    pub async fn poll_and_submit(
        &mut self,
        frame: Option<&EyeFrame>,
        world_revision: u64,
        now_ms: u64,
    ) -> LiveImageCognitionTick {
        let mut tick = LiveImageCognitionTick::default();
        let current_snapshot = frame.map_or_else(SnapshotRef::default, |frame| SnapshotRef {
            snapshot_id: live_image_source_frame_id(frame),
            schema_version: 1,
            revision: world_revision,
            captured_at_ms: frame.captured_at_ms,
        });
        if let Some(response) = self.supervisor.poll(&current_snapshot, now_ms).await {
            if response.disposition == ResponseDisposition::Accepted {
                tick.enrichment = enrichment_from_response(&response.response);
            }
            tick.response = Some(response);
        }

        if !self.supervisor.is_pending() {
            if let Some(frame) = frame {
                let key = LiveImageFrameKey::from(frame);
                if self.last_frame_key.as_ref() != Some(&key) {
                    self.last_frame_key = Some(key);
                    if let Ok(request) =
                        scene_request(frame, world_revision, now_ms, self.request_timeout_ms)
                    {
                        let disposition = self.supervisor.submit(request, now_ms);
                        debug_assert_ne!(disposition, SubmissionDisposition::Busy);
                    }
                }
            }
        }
        tick.registry = self.supervisor.registry_snapshot().clone();
        tick
    }
}

fn scene_request(
    frame: &EyeFrame,
    world_revision: u64,
    now_ms: u64,
    timeout_ms: u64,
) -> Result<CognitiveRequest> {
    let png = encode_eye_frame_png(frame)?;
    let source_frame_id = live_image_source_frame_id(frame);
    let reference = BoundedInputRef {
        id: source_frame_id.clone(),
        kind: "eye_frame_png".to_string(),
        byte_len: png.len(),
        content_hash: None,
    };
    let mut request = CognitiveRequest::new(
        SnapshotRef {
            snapshot_id: source_frame_id.clone(),
            schema_version: 1,
            revision: world_revision,
            captured_at_ms: frame.captured_at_ms,
        },
        now_ms,
        now_ms.saturating_add(timeout_ms),
        PrivacyPolicy {
            maximum_locality: Locality::LocalNetwork,
            allow_raw_image: true,
            allow_persistence: false,
        },
        RequestProvenance {
            caller: CallerRole::OrganismRuntime,
            caller_id: "live_vision".to_string(),
            evidence_refs: vec![source_frame_id],
        },
        CognitiveRequestPayload::DescribeScene(BoundedImageInput {
            reference: reference.clone(),
            content_type: "image/png".to_string(),
            width: frame.width,
            height: frame.height,
            captured_at_ms: frame.captured_at_ms,
            bytes: png,
        }),
    );
    request.input_refs.push(reference);
    Ok(request)
}

fn enrichment_from_response(response: &CognitiveResponse) -> Option<LiveImageEnrichment> {
    let CognitiveResponsePayload::SceneDescription { text, embedding } = &response.payload else {
        return None;
    };
    let source_frame_id = response.input_snapshot.snapshot_id.clone();
    let embedding_model = response
        .model_version
        .clone()
        .unwrap_or_else(|| response.implementation_version.clone());
    let point_id = stable_uuid_for_text(&format!(
        "{}:{}:{}",
        source_frame_id, response.provider_id.0, text
    ))
    .to_string();
    let image_description_vector = VectorArtifact::new(
        IMAGE_DESCRIPTION_VECTOR_COLLECTION,
        point_id.clone(),
        embedding.clone(),
    )
    .with_model(embedding_model.clone())
    .with_source_id(format!("image-description:{source_frame_id}"))
    .with_source_frame_id(source_frame_id.clone())
    .with_occurred_at_ms(response.input_snapshot.captured_at_ms);
    let scene_vector = VectorArtifact::new(SCENE_VECTOR_COLLECTION, point_id, embedding.clone())
        .with_model(embedding_model)
        .with_source_id(format!("image-description:{source_frame_id}"))
        .with_source_frame_id(source_frame_id)
        .with_occurred_at_ms(response.input_snapshot.captured_at_ms);
    Some(LiveImageEnrichment {
        description: text.clone(),
        image_description_vector,
        scene_vector,
    })
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
            options: OllamaGenerateOptions::from(&self.config),
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

#[async_trait]
impl CognitiveProvider for LiveImageEnricher {
    fn descriptor(&self) -> CognitiveProviderDescriptor {
        CognitiveProviderDescriptor {
            provider_id: ProviderId(OLLAMA_SCENE_PROVIDER_ID.to_string()),
            role: CognitiveRole::CognitiveAccelerator,
            host_id: None::<HostId>,
            process_id: None::<ProcessId>,
            implementation: "ollama".to_string(),
            implementation_version: "api-v1".to_string(),
            model_version: self.config.vision_model.clone(),
            capabilities: vec![CapabilityDescriptor {
                capability: CognitiveCapability::DescribeScene,
                version: "1".to_string(),
                supports_partial: false,
                performance_confidence: 0.8,
            }],
            health: ProviderHealth {
                state: ProviderHealthState::Available,
                confidence: 0.5,
                observed_at_ms: wall_now_ms(),
                valid_until_ms: u64::MAX,
                reason: None,
            },
            latency: LatencyEstimate {
                expected_ms: self.config.timeout_ms.min(2_000),
                p95_ms: self.config.timeout_ms,
            },
            resource_class: ResourceClass::Accelerated,
            locality: Locality::LocalNetwork,
            trust: TrustPolicy::TrustedProvider,
            energy_cost: 0.2,
            network_cost: 0.2,
        }
    }

    async fn execute(&mut self, request: &CognitiveRequest) -> Result<CognitiveResponse> {
        let CognitiveRequestPayload::DescribeScene(image) = &request.payload else {
            anyhow::bail!("ollama scene provider received incompatible request payload");
        };
        let started = std::time::Instant::now();
        let vision_model = self
            .config
            .vision_model
            .as_deref()
            .filter(|model| !model.trim().is_empty())
            .context("scene description requires vision_model")?;
        let embedding_model = self
            .config
            .embedding_model
            .as_deref()
            .filter(|model| !model.trim().is_empty())
            .context("scene description requires embedding_model")?;
        let image_base64 = base64::engine::general_purpose::STANDARD.encode(&image.bytes);
        let description = self.describe_image(vision_model, image_base64).await?;
        let embedding = self.embed_text(embedding_model, &description).await?;
        let completed_at_ms = wall_now_ms();
        Ok(CognitiveResponse {
            schema_version: 1,
            request_id: request.request_id.clone(),
            provider_id: ProviderId(OLLAMA_SCENE_PROVIDER_ID.to_string()),
            provider_role: CognitiveRole::CognitiveAccelerator,
            implementation: "ollama".to_string(),
            implementation_version: "api-v1".to_string(),
            model_version: Some(format!("{vision_model}+{embedding_model}")),
            status: if completed_at_ms > request.deadline_ms {
                CognitiveResponseStatus::Stale
            } else {
                CognitiveResponseStatus::Completed
            },
            confidence: 0.8,
            uncertainty: 0.2,
            input_snapshot: request.input_snapshot.clone(),
            completed_at_ms,
            resource_cost: ResourceCost {
                elapsed_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
                energy_estimate: 0.2,
                network_bytes: image.bytes.len().try_into().unwrap_or(u64::MAX),
            },
            provenance: request.provenance.evidence_refs.clone(),
            payload: CognitiveResponsePayload::SceneDescription {
                text: description,
                embedding,
            },
            failure: None,
        })
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

fn encode_eye_frame_png(frame: &EyeFrame) -> Result<Vec<u8>> {
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
    Ok(png)
}

fn encode_eye_frame_png_base64(frame: &EyeFrame) -> Result<String> {
    Ok(base64::engine::general_purpose::STANDARD.encode(encode_eye_frame_png(frame)?))
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
    "pete-live-image-a".hash(&mut first);
    text.hash(&mut first);
    let mut second = DefaultHasher::new();
    "pete-live-image-b".hash(&mut second);
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
    if now.body.infrared_character != 0 {
        lines.push(format!(
            "My Create IR receiver reports character {}.",
            now.body.infrared_character
        ));
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
