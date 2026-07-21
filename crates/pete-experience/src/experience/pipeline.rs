#[derive(Clone, Debug, Default)]
pub struct TemplateImpressionGenerator;

impl TemplateImpressionGenerator {
    pub fn generate_for_sensation(&self, sensation: &Sensation) -> Impression {
        let text = match (&sensation.modality, &sensation.payload_kind) {
            (Modality::Vision, SensationPayloadKind::ImageBytes) => {
                let width = sensation
                    .payload
                    .get("width")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let height = sensation
                    .payload
                    .get("height")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                if width > 0 && height > 0 {
                    format!("I see a {} by {} frame in front of me.", width, height)
                } else {
                    "I see light and shape in front of me.".to_string()
                }
            }
            (Modality::Vision, SensationPayloadKind::Crop) => {
                match sensation
                    .metadata
                    .properties
                    .get("detection_kind")
                    .and_then(Value::as_str)
                {
                    Some("face") => "I see a face close to me.".to_string(),
                    Some("object") => "I notice an object-shaped region ahead.".to_string(),
                    Some("salient_region") => "I notice a salient patch of the scene.".to_string(),
                    _ => "I focus on a smaller part of what I see.".to_string(),
                }
            }
            (Modality::Audio, SensationPayloadKind::AudioPcm) => {
                "I hear a short sound nearby.".to_string()
            }
            (Modality::Audio, SensationPayloadKind::VoiceSegment) => {
                "I hear a voice nearby.".to_string()
            }
            (Modality::Audio, SensationPayloadKind::SpeechSegment) => sensation
                .payload
                .get("text")
                .and_then(Value::as_str)
                .map(|text| format!("I hear someone say \"{}\".", text.trim()))
                .unwrap_or_else(|| "I hear a voice nearby.".to_string()),
            (Modality::Audio, SensationPayloadKind::TranscriptSpan) => sensation
                .payload
                .get("text")
                .and_then(Value::as_str)
                .map(|text| format!("I hear someone say \"{}\".", text.trim()))
                .unwrap_or_else(|| "I hear speech nearby.".to_string()),
            (Modality::Audio, SensationPayloadKind::PhonemeSpan) => {
                "I hear a small piece of speech sound.".to_string()
            }
            (Modality::Touch, SensationPayloadKind::ContactEvent) => {
                "I feel contact against my body.".to_string()
            }
            (Modality::Odometry, SensationPayloadKind::OdometryEvent) => {
                "I feel my position changing through the room.".to_string()
            }
            (Modality::Depth, _) => "I sense distance and surface in front of me.".to_string(),
            (Modality::Memory, _) => sensation
                .summary
                .clone()
                .unwrap_or_else(|| "I remember something related to now.".to_string()),
            _ => sensation
                .summary
                .clone()
                .unwrap_or_else(|| "I notice something happening now.".to_string()),
        };
        let mut impression = Impression::new(
            "sensation.template",
            text.clone(),
            vec![sensation.id],
            sensation.occurred_at_ms,
            sensation.observed_at_ms,
        )
        .with_confidence(
            sensation
                .metadata
                .confidence
                .unwrap_or(0.55)
                .clamp(0.0, 1.0),
        )
        .with_payload(json!({
            "modality": sensation.modality,
            "payload_kind": sensation.payload_kind,
            "source": sensation.source,
        }));
        impression.vector = Some(semantic_text_vector(
            &text,
            impression.id,
            sensation.observed_at_ms,
            "impression",
            "impression_semantic",
            "impressions",
            format!(
                "impression kind={} about_sensation={} text={}",
                impression.kind,
                sensation.id,
                text.chars().take(96).collect::<String>()
            ),
        ));
        impression
    }

    pub fn generate_for_experience(
        &self,
        experience_id: ExperienceId,
        window_start_ms: TimeMs,
        window_end_ms: TimeMs,
        impressions: &[Impression],
    ) -> Impression {
        let mut parts = impressions
            .iter()
            .map(|impression| impression.text.trim().trim_end_matches('.').to_string())
            .filter(|text| !text.is_empty())
            .take(3)
            .collect::<Vec<_>>();
        let text = if parts.is_empty() {
            "I am here in a quiet moment.".to_string()
        } else if parts.len() == 1 {
            format!("{}.", parts.remove(0))
        } else {
            format!("{}.", parts.join(", and "))
        };
        let mut impression = Impression::new(
            "experience.template",
            text.clone(),
            Vec::new(),
            window_start_ms,
            window_end_ms,
        )
        .for_experience(experience_id)
        .with_confidence(0.6);
        impression.vector = Some(semantic_text_vector(
            &text,
            experience_id,
            window_end_ms,
            "experience",
            "experience_semantic",
            "experiences",
            format!(
                "experience_summary id={} text={}",
                experience_id,
                text.chars().take(96).collect::<String>()
            ),
        ));
        impression
    }
}

#[derive(Clone)]
pub struct EmbodiedPipeline {
    extractor: DeterministicDescendantExtractor,
    vectorizers: SensationVectorizerRegistry,
    impressions: TemplateImpressionGenerator,
}

impl Default for EmbodiedPipeline {
    fn default() -> Self {
        Self {
            extractor: DeterministicDescendantExtractor,
            vectorizers: SensationVectorizerRegistry::with_defaults(),
            impressions: TemplateImpressionGenerator,
        }
    }
}

impl EmbodiedPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_vectorizers(vectorizers: SensationVectorizerRegistry) -> Self {
        Self {
            vectorizers,
            ..Self::default()
        }
    }

    pub fn from_models_toml(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::with_vectorizers(
            SensationVectorizerRegistry::from_models_toml(path)?,
        ))
    }

    pub async fn ingest_primary(&self, primary: Sensation) -> Result<EmbodiedBatch> {
        let mut sensations = vec![primary];
        let descendants = self.extractor.extract(&sensations[0])?;
        sensations.extend(descendants);
        let mut impressions = Vec::with_capacity(sensations.len());
        for sensation in &mut sensations {
            if let Some(vector) = self.vectorizers.vectorize(sensation).await? {
                sensation.vector = Some(vector);
            }
            let impression = self.impressions.generate_for_sensation(sensation);
            sensation.impression = Some(impression.clone());
            impressions.push(impression);
        }
        Ok(EmbodiedBatch {
            sensations,
            impressions,
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedBatch {
    pub sensations: Vec<Sensation>,
    pub impressions: Vec<Impression>,
}

#[derive(Clone, Debug)]
pub struct ExperienceFuser {
    window_ms: TimeMs,
    impressions: TemplateImpressionGenerator,
}

impl Default for ExperienceFuser {
    fn default() -> Self {
        Self {
            window_ms: DEFAULT_WINDOW_MS,
            impressions: TemplateImpressionGenerator,
        }
    }
}

impl ExperienceFuser {
    pub fn new(window_ms: TimeMs) -> Self {
        Self {
            window_ms: window_ms.max(1),
            ..Self::default()
        }
    }

    pub fn fuse(&self, sensations: &[Sensation], impressions: &[Impression]) -> Result<Experience> {
        if sensations.is_empty() && impressions.is_empty() {
            return Err(anyhow!("cannot fuse an empty embodied window"));
        }
        let window_start_ms = sensations
            .iter()
            .map(|sensation| sensation.occurred_at_ms)
            .chain(
                impressions
                    .iter()
                    .map(|impression| impression.occurred_at_ms),
            )
            .min()
            .unwrap_or_default();
        let window_end_ms = sensations
            .iter()
            .map(|sensation| sensation.observed_at_ms)
            .chain(
                impressions
                    .iter()
                    .map(|impression| impression.observed_at_ms),
            )
            .max()
            .unwrap_or(window_start_ms + self.window_ms);
        let sensation_ids = sensations
            .iter()
            .map(|sensation| sensation.id)
            .collect::<Vec<_>>();
        let impression_ids = impressions
            .iter()
            .map(|impression| impression.id)
            .collect::<Vec<_>>();
        let experience_id = Uuid::new_v4();
        let summary = self.impressions.generate_for_experience(
            experience_id,
            window_start_ms,
            window_end_ms,
            impressions,
        );
        let mut experience = Experience::new(
            "embodied.now",
            summary.text.clone(),
            impression_ids,
            sensation_ids,
            window_start_ms,
            window_end_ms,
        );
        experience.id = experience_id;
        experience.window_start_ms = window_start_ms;
        experience.window_end_ms = window_end_ms;
        experience.summary_impression = Some(summary);
        experience.predictions = vec![Prediction {
            offset_ms: self.window_ms,
            text:
                "I expect the next moment to resemble this one unless I move or something changes."
                    .to_string(),
            confidence: 0.35,
            vector: None,
        }];
        experience.tags = embodied_tags(sensations);
        experience.payload = json!({
            "pipeline": "embodied.v0",
            "sensation_count": sensations.len(),
            "impression_count": impressions.len(),
            "window_ms": window_end_ms.saturating_sub(window_start_ms),
        });
        Ok(experience)
    }
}

#[derive(Clone, Debug)]
pub struct RollingExperienceWindow {
    window_ms: TimeMs,
    sensations: VecDeque<Sensation>,
    impressions: VecDeque<Impression>,
    fuser: ExperienceFuser,
}

impl RollingExperienceWindow {
    pub fn new(window_ms: TimeMs) -> Self {
        Self {
            window_ms: window_ms.max(1),
            sensations: VecDeque::new(),
            impressions: VecDeque::new(),
            fuser: ExperienceFuser::new(window_ms),
        }
    }

    pub fn push(&mut self, batch: EmbodiedBatch) {
        let newest = batch
            .sensations
            .iter()
            .map(|sensation| sensation.observed_at_ms)
            .chain(
                batch
                    .impressions
                    .iter()
                    .map(|impression| impression.observed_at_ms),
            )
            .max()
            .unwrap_or_default();
        self.sensations.extend(batch.sensations);
        self.impressions.extend(batch.impressions);
        self.prune(newest);
    }

    pub fn fuse_current(&self) -> Result<Experience> {
        let sensations = self.sensations.iter().cloned().collect::<Vec<_>>();
        let impressions = self.impressions.iter().cloned().collect::<Vec<_>>();
        self.fuser.fuse(&sensations, &impressions)
    }

    fn prune(&mut self, newest_observed_at_ms: TimeMs) {
        let cutoff = newest_observed_at_ms.saturating_sub(self.window_ms);
        while self
            .sensations
            .front()
            .map(|sensation| sensation.observed_at_ms < cutoff)
            .unwrap_or(false)
        {
            self.sensations.pop_front();
        }
        while self
            .impressions
            .front()
            .map(|impression| impression.observed_at_ms < cutoff)
            .unwrap_or(false)
        {
            self.impressions.pop_front();
        }
    }
}

pub async fn demo_embodied_experience(now_ms: TimeMs) -> Result<EmbodiedDemo> {
    let mut rgb = vec![12_u8; 64 * 48 * 3];
    for y in 14..34 {
        for x in 24..42 {
            let idx = (y * 64 + x) * 3;
            rgb[idx] = 220;
            rgb[idx + 1] = 170;
            rgb[idx + 2] = 120;
        }
    }
    let mut now = Now::blank(now_ms, BodySense::default());
    now.eye_frame = Some(pete_now::EyeFrame {
        rgbd_frame_id: None,
        device_timestamp_ms: None,
        captured_at_ms: now_ms,
        width: 64,
        height: 48,
        format: pete_now::EyeFrameFormat::Rgb8,
        bytes: rgb,
        source: Some("demo.synthetic_camera".to_string()),
    });
    now.face.vectors.push(
        pete_now::VectorArtifact::new("faces", "demo-face-vector", vec![0.17, 0.41, 0.73, 0.29])
            .with_model("face_id/0.4.1")
            .with_source_id("demo-face")
            .with_source_frame_id("demo-synthetic-frame")
            .with_occurred_at_ms(now_ms),
    );
    now.ear.transcript = Some("hello pete, this is a transcript vector test".to_string());
    now.ear.asr.transcript = now.ear.transcript.clone();
    now.ear.asr.is_final = true;
    now.ear.asr.confidence = 0.82;
    now.ear.asr.start_ms = Some(now_ms.saturating_sub(320));
    now.ear.asr.end_ms = Some(now_ms);
    now.ear.asr.duration_ms = Some(320);
    now.ear.asr.word_count = Some(8);
    now.voice.vectors.push(
        pete_now::VectorArtifact::new(
            "voices",
            "demo-voice-vector",
            vec![0.11, 0.05, 0.33, 0.78, 0.21],
        )
        .with_model("pete/voice_vector/16d")
        .with_source_id("demo-voice")
        .with_occurred_at_ms(now_ms),
    );
    let pipeline = EmbodiedPipeline::from_models_toml("configs/models.toml").unwrap_or_else(|error| {
        eprintln!(
            "warning: embodied demo could not load configs/models.toml ({error}); using built-in vectorizer defaults"
        );
        EmbodiedPipeline::new()
    });
    let mut sensations = Vec::new();
    let mut impressions = Vec::new();
    for primary in primary_sensations_from_now(&now) {
        let batch = pipeline.ingest_primary(primary).await?;
        sensations.extend(batch.sensations);
        impressions.extend(batch.impressions);
    }
    let batch = EmbodiedBatch {
        sensations,
        impressions,
    };
    let mut window = RollingExperienceWindow::new(DEFAULT_WINDOW_MS);
    window.push(batch.clone());
    let experience = window.fuse_current()?;
    let coverage = EmbodiedVectorCoverage::from_parts(
        &batch.sensations,
        &batch.impressions,
        Some(&experience),
    );
    Ok(EmbodiedDemo {
        sensations: batch.sensations,
        impressions: batch.impressions,
        experience,
        coverage,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedDemo {
    pub sensations: Vec<Sensation>,
    pub impressions: Vec<Impression>,
    pub experience: Experience,
    pub coverage: EmbodiedVectorCoverage,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedVectorCoverage {
    pub image: usize,
    pub face: usize,
    pub voice: usize,
    pub transcript: usize,
    pub impression: usize,
    pub experience: usize,
    pub fallback_count: usize,
}

impl EmbodiedVectorCoverage {
    pub fn from_parts(
        sensations: &[Sensation],
        impressions: &[Impression],
        experience: Option<&Experience>,
    ) -> Self {
        let mut coverage = Self::default();
        for vector in sensations
            .iter()
            .filter_map(|sensation| sensation.vector.as_ref())
            .chain(
                impressions
                    .iter()
                    .filter_map(|impression| impression.vector.as_ref()),
            )
            .chain(
                experience
                    .and_then(|experience| experience.summary_impression.as_ref())
                    .and_then(|impression| impression.vector.as_ref())
                    .into_iter(),
            )
        {
            coverage.record(vector);
        }
        coverage
    }

    fn record(&mut self, vector: &VectorEmbedding) {
        if vector.is_fallback {
            self.fallback_count += 1;
        }
        match vector.purpose.as_str() {
            "scene_similarity" | "visual_similarity" => self.image += 1,
            "face_identity" => self.face += 1,
            "voice_identity" => self.voice += 1,
            "transcript_semantic" => self.transcript += 1,
            "impression_semantic" => self.impression += 1,
            "experience_semantic" => self.experience += 1,
            _ => {}
        }
    }
}

pub async fn embody_now(now: &Now) -> Result<EmbodiedNow> {
    let pipeline = EmbodiedPipeline::new();
    let mut sensations = Vec::new();
    let mut impressions = Vec::new();

    for primary in primary_sensations_from_now(now) {
        let batch = pipeline.ingest_primary(primary).await?;
        sensations.extend(batch.sensations);
        impressions.extend(batch.impressions);
    }

    let experience = ExperienceFuser::new(DEFAULT_WINDOW_MS).fuse(&sensations, &impressions)?;
    Ok(EmbodiedNow {
        sensations,
        impressions,
        experience,
    })
}
