#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sensation {
    pub id: SensationId,
    #[serde(default)]
    pub parent_id: Option<SensationId>,
    #[serde(default)]
    pub modality: Modality,
    #[serde(default)]
    pub payload_kind: SensationPayloadKind,
    pub kind: String,
    pub source: String,
    pub occurred_at_ms: TimeMs,
    pub observed_at_ms: TimeMs,
    pub summary: Option<String>,
    pub provenance: Provenance,
    pub payload: Value,
    #[serde(default)]
    pub metadata: SensationMetadata,
    #[serde(default)]
    pub vector: Option<VectorEmbedding>,
    #[serde(default)]
    pub impression: Option<Impression>,
}

impl Sensation {
    pub fn new(
        kind: impl Into<String>,
        source: impl Into<String>,
        occurred_at_ms: TimeMs,
        observed_at_ms: TimeMs,
        payload: Value,
    ) -> Self {
        Self {
            id: new_experience_uuid(),
            parent_id: None,
            modality: Modality::Other,
            payload_kind: SensationPayloadKind::Structured,
            kind: kind.into(),
            source: source.into(),
            occurred_at_ms,
            observed_at_ms,
            summary: None,
            provenance: Provenance::direct(),
            payload,
            metadata: SensationMetadata::default(),
            vector: None,
            impression: None,
        }
    }

    pub fn primary(
        modality: Modality,
        source: SensationSource,
        occurred_at_ms: TimeMs,
        observed_at_ms: TimeMs,
        payload: SensationPayload,
    ) -> Self {
        let kind = format!("{}.{}", modality.as_str(), payload.kind().as_str());
        let mut sensation = Self::new(
            kind,
            source.name.clone(),
            occurred_at_ms,
            observed_at_ms,
            payload.value,
        );
        sensation.modality = modality;
        sensation.payload_kind = payload.kind;
        sensation.metadata.source = source;
        sensation
    }

    pub fn descendant(
        parent: &Sensation,
        kind: impl Into<String>,
        payload_kind: SensationPayloadKind,
        payload: Value,
        metadata: SensationMetadata,
        stage: impl Into<String>,
    ) -> Self {
        let mut sensation = Self::new(
            kind,
            parent.source.clone(),
            parent.occurred_at_ms,
            parent.observed_at_ms,
            payload,
        );
        sensation.parent_id = Some(parent.id);
        sensation.modality = parent.modality.clone();
        sensation.payload_kind = payload_kind;
        sensation.metadata = metadata;
        sensation.provenance = Provenance::derived_from_sensations([parent.id]).with_stage(stage);
        sensation
    }

    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    pub fn with_provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = provenance;
        self
    }

    pub fn with_vector(mut self, vector: VectorEmbedding) -> Self {
        self.vector = Some(vector);
        self
    }

    pub fn with_impression(mut self, impression: Impression) -> Self {
        self.impression = Some(impression);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Impression {
    pub id: ImpressionId,
    pub kind: String,
    pub text: String,
    pub about: Vec<SensationId>,
    #[serde(default)]
    pub sensation_id: Option<SensationId>,
    #[serde(default)]
    pub experience_id: Option<ExperienceId>,
    pub occurred_at_ms: TimeMs,
    pub observed_at_ms: TimeMs,
    pub confidence: f32,
    #[serde(default)]
    pub generator: ImpressionGenerator,
    #[serde(default)]
    pub vector: Option<VectorEmbedding>,
    pub payload: Value,
}

impl Impression {
    pub fn new(
        kind: impl Into<String>,
        text: impl Into<String>,
        about: Vec<SensationId>,
        occurred_at_ms: TimeMs,
        observed_at_ms: TimeMs,
    ) -> Self {
        Self {
            id: new_experience_uuid(),
            kind: kind.into(),
            text: text.into(),
            sensation_id: about.first().copied(),
            experience_id: None,
            about,
            occurred_at_ms,
            observed_at_ms,
            confidence: 0.5,
            generator: ImpressionGenerator::Template,
            vector: None,
            payload: Value::Null,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn with_payload(mut self, payload: Value) -> Self {
        self.payload = payload;
        self
    }

    pub fn with_vector(mut self, vector: VectorEmbedding) -> Self {
        self.vector = Some(vector);
        self
    }

    pub fn for_experience(mut self, experience_id: ExperienceId) -> Self {
        self.experience_id = Some(experience_id);
        self.sensation_id = None;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Experience {
    pub id: ExperienceId,
    pub kind: String,
    pub text: String,
    pub impression_ids: Vec<ImpressionId>,
    pub sensation_ids: Vec<SensationId>,
    #[serde(default)]
    pub window_start_ms: TimeMs,
    #[serde(default)]
    pub window_end_ms: TimeMs,
    #[serde(default)]
    pub summary_impression: Option<Impression>,
    #[serde(default)]
    pub predictions: Vec<Prediction>,
    #[serde(default)]
    pub memory_links: Vec<MemoryLink>,
    pub occurred_at_ms: TimeMs,
    pub observed_at_ms: TimeMs,
    pub salience: f32,
    pub tags: Vec<String>,
    pub payload: Value,
}

impl Experience {
    pub fn new(
        kind: impl Into<String>,
        text: impl Into<String>,
        impression_ids: Vec<ImpressionId>,
        sensation_ids: Vec<SensationId>,
        occurred_at_ms: TimeMs,
        observed_at_ms: TimeMs,
    ) -> Self {
        Self {
            id: new_experience_uuid(),
            kind: kind.into(),
            text: text.into(),
            impression_ids,
            sensation_ids,
            window_start_ms: occurred_at_ms,
            window_end_ms: observed_at_ms,
            summary_impression: None,
            predictions: Vec::new(),
            memory_links: Vec::new(),
            occurred_at_ms,
            observed_at_ms,
            salience: 0.5,
            tags: Vec::new(),
            payload: Value::Null,
        }
    }

    pub fn to_recall_sensation(
        &self,
        recall_at_ms: TimeMs,
        score: f32,
        stage: impl Into<String>,
    ) -> Sensation {
        self.to_recall_sensation_with_lineage(recall_at_ms, score, stage, None, Vec::new())
    }

    pub fn to_recall_sensation_with_lineage(
        &self,
        recall_at_ms: TimeMs,
        score: f32,
        stage: impl Into<String>,
        original_frame_id: Option<Uuid>,
        original_vector_ids: Vec<String>,
    ) -> Sensation {
        let stage = stage.into();
        let payload = json!({
            "experience": self,
            "recall_kind": "recalled_experience",
            "original_frame_id": original_frame_id,
            "original_experience_id": self.id,
            "original_sensation_ids": self.sensation_ids,
            "original_impression_ids": self.impression_ids,
            "original_vector_ids": original_vector_ids,
            "original_occurred_at_ms": self.occurred_at_ms,
            "original_observed_at_ms": self.observed_at_ms,
            "score": score,
        });
        let mut provenance = Provenance::memory_recall(self.id).with_stage(stage);
        provenance.metadata = json!({
            "original_frame_id": original_frame_id,
            "original_experience_id": self.id,
            "original_vector_ids": payload.get("original_vector_ids").cloned().unwrap_or(Value::Null),
        });
        let mut sensation = Sensation::primary(
            Modality::Memory,
            SensationSource::new("memory.recall"),
            recall_at_ms,
            recall_at_ms,
            SensationPayload {
                kind: SensationPayloadKind::MemoryRecall,
                value: payload,
            },
        )
        .with_summary(format!(
            "I remember a similar moment near here: {}",
            self.text
        ))
        .with_provenance(provenance);
        sensation.kind = "memory.recall.experience".to_string();
        sensation.metadata.confidence = Some(score.clamp(0.0, 1.0));
        sensation.metadata.labels.push("memory_recall".to_string());
        sensation
            .metadata
            .labels
            .push("recalled_experience".to_string());
        if let Some(frame_id) = original_frame_id {
            sensation.metadata.properties.insert(
                "original_frame_id".to_string(),
                Value::String(frame_id.to_string()),
            );
        }
        sensation.metadata.properties.insert(
            "original_experience_id".to_string(),
            Value::String(self.id.to_string()),
        );
        sensation.metadata.properties.insert(
            "original_vector_count".to_string(),
            json!(original_vector_ids.len()),
        );
        sensation
    }

    pub fn to_recall_impression(&self, sensation: &Sensation, score: f32) -> Impression {
        Impression::new(
            "memory.recall.impression",
            format!("I remember a similar moment near here: {}", self.text),
            vec![sensation.id],
            sensation.occurred_at_ms,
            sensation.observed_at_ms,
        )
        .with_confidence(score.clamp(0.0, 1.0))
        .with_payload(json!({
            "generator": "memory_recall",
            "original_experience_id": self.id,
            "score": score,
        }))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Vision,
    Audio,
    Depth,
    Lidar,
    Touch,
    Odometry,
    Memory,
    Language,
    #[default]
    Other,
}

impl Modality {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Vision => "vision",
            Self::Audio => "audio",
            Self::Depth => "depth",
            Self::Lidar => "lidar",
            Self::Touch => "touch",
            Self::Odometry => "odometry",
            Self::Memory => "memory",
            Self::Language => "language",
            Self::Other => "other",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensationPayloadKind {
    ImageBytes,
    AudioPcm,
    VoiceSegment,
    DepthFrame,
    PointCloud,
    LidarScan,
    ContactEvent,
    OdometryEvent,
    Crop,
    SpeechSegment,
    TranscriptSpan,
    PhonemeSpan,
    MemoryRecall,
    #[default]
    Structured,
}

impl SensationPayloadKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ImageBytes => "image_bytes",
            Self::AudioPcm => "audio_pcm",
            Self::VoiceSegment => "voice_segment",
            Self::DepthFrame => "depth_frame",
            Self::PointCloud => "point_cloud",
            Self::LidarScan => "lidar_scan",
            Self::ContactEvent => "contact_event",
            Self::OdometryEvent => "odometry_event",
            Self::Crop => "crop",
            Self::SpeechSegment => "speech_segment",
            Self::TranscriptSpan => "transcript_span",
            Self::PhonemeSpan => "phoneme_span",
            Self::MemoryRecall => "memory_recall",
            Self::Structured => "structured",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SensationSource {
    pub name: String,
    pub device_id: Option<String>,
    pub frame_id: Option<String>,
}

impl SensationSource {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            device_id: None,
            frame_id: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensationPayload {
    pub kind: SensationPayloadKind,
    pub value: Value,
}

impl SensationPayload {
    pub fn image_metadata(
        width: u32,
        height: u32,
        format: impl Into<String>,
        byte_len: usize,
    ) -> Self {
        Self {
            kind: SensationPayloadKind::ImageBytes,
            value: json!({
                "width": width,
                "height": height,
                "format": format.into(),
                "byte_len": byte_len,
            }),
        }
    }

    pub fn structured(value: Value) -> Self {
        Self {
            kind: SensationPayloadKind::Structured,
            value,
        }
    }

    pub fn kind(&self) -> SensationPayloadKind {
        self.kind.clone()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SensationMetadata {
    pub source: SensationSource,
    pub labels: Vec<String>,
    pub bbox: Option<BoundingBox>,
    pub duration_ms: Option<TimeMs>,
    pub confidence: Option<f32>,
    pub properties: BTreeMap<String, Value>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VectorEmbedding {
    pub vector: Vec<f32>,
    pub dim: usize,
    #[serde(default = "default_vectorizer_id")]
    pub vectorizer_id: String,
    pub model_id: String,
    #[serde(default)]
    pub model_label: String,
    pub modality: Modality,
    pub payload_kind: SensationPayloadKind,
    #[serde(default = "default_vector_source_kind")]
    pub source_kind: String,
    pub source_sensation_id: SensationId,
    #[serde(default = "default_vector_purpose")]
    pub purpose: String,
    #[serde(default = "default_vector_collection")]
    pub collection: String,
    #[serde(default)]
    pub input_summary: String,
    #[serde(default)]
    pub is_fallback: bool,
    #[serde(default)]
    pub provenance: String,
    pub generated_at_ms: TimeMs,
}

impl VectorEmbedding {
    pub fn new(
        vector: Vec<f32>,
        model_id: impl Into<String>,
        modality: Modality,
        payload_kind: SensationPayloadKind,
        source_sensation_id: SensationId,
        generated_at_ms: TimeMs,
    ) -> Self {
        let dim = vector.len();
        let model_id = model_id.into();
        Self {
            vector,
            dim,
            vectorizer_id: model_id.clone(),
            model_label: model_id.clone(),
            model_id,
            modality,
            payload_kind,
            source_kind: default_vector_source_kind(),
            source_sensation_id,
            purpose: default_vector_purpose(),
            collection: default_vector_collection(),
            input_summary: String::new(),
            is_fallback: false,
            provenance: String::new(),
            generated_at_ms,
        }
    }

    pub fn with_metadata(
        mut self,
        vectorizer_id: impl Into<String>,
        model_label: impl Into<String>,
        purpose: impl Into<String>,
        collection: impl Into<String>,
        input_summary: impl Into<String>,
        is_fallback: bool,
        provenance: impl Into<String>,
    ) -> Self {
        self.vectorizer_id = vectorizer_id.into();
        self.model_label = model_label.into();
        self.purpose = purpose.into();
        self.collection = collection.into();
        self.input_summary = input_summary.into();
        self.is_fallback = is_fallback;
        self.provenance = provenance.into();
        self
    }

    pub fn with_source_kind(mut self, source_kind: impl Into<String>) -> Self {
        self.source_kind = source_kind.into();
        self
    }
}

fn default_vectorizer_id() -> String {
    "unknown".to_string()
}

fn default_vector_source_kind() -> String {
    "sensation".to_string()
}

fn default_vector_purpose() -> String {
    "unspecified".to_string()
}

fn default_vector_collection() -> String {
    "embodied_vectors".to_string()
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImpressionGenerator {
    #[default]
    Template,
    Llm,
    Human,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    pub offset_ms: TimeMs,
    pub text: String,
    pub confidence: f32,
    pub vector: Option<VectorEmbedding>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MemoryLink {
    pub target_id: String,
    pub relation: String,
    pub score: f32,
    pub payload: Value,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedContext {
    pub experience_id: Option<ExperienceId>,
    pub summary: String,
    pub sensations: Vec<EmbodiedSensationRef>,
    pub impressions: Vec<EmbodiedImpressionRef>,
    pub lineage: Vec<EmbodiedLineageEdge>,
    pub sensation_vectors: Vec<EmbodiedVectorRef>,
    #[serde(default)]
    pub impression_vectors: Vec<EmbodiedVectorRef>,
    pub predictions: Vec<EmbodiedPredictionRef>,
    pub memory_links: Vec<EmbodiedMemoryLinkRef>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceInstant {
    pub schema_version: u32,
    pub t_ms: TimeMs,
    pub window_start_ms: TimeMs,
    pub window_end_ms: TimeMs,
    pub experience_id: Option<ExperienceId>,
    pub summary: String,
    pub primary_sensations: Vec<EmbodiedSensationRef>,
    pub descendant_sensations: Vec<EmbodiedSensationRef>,
    pub impressions: Vec<EmbodiedImpressionRef>,
    pub summary_impression: Option<EmbodiedImpressionRef>,
    pub teacher_vectors: Vec<InstantTeacherVector>,
    pub body_context: InstantBodyContext,
    pub action_context: InstantActionContext,
    pub lineage: Vec<EmbodiedLineageEdge>,
    #[serde(default)]
    pub memory_links: Vec<EmbodiedMemoryLinkRef>,
    #[serde(default)]
    pub predictions: Vec<EmbodiedPredictionRef>,
    pub provenance: InstantProvenance,
    pub missing_modalities: Vec<MissingModality>,
}

pub type Instant = ExperienceInstant;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InstantCoverage {
    pub present_modalities: Vec<String>,
    pub missing_modalities: Vec<String>,
    pub sensation_count: usize,
    pub descendant_count: usize,
    pub vector_count: usize,
    pub impression_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InstantTeacherVector {
    pub vector: Vec<f32>,
    pub metadata: EmbodiedVectorRef,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InstantBodyContext {
    pub battery_level: f32,
    pub charging: bool,
    pub bump: bool,
    pub cliff: bool,
    pub wheel_drop: bool,
    pub wall: bool,
    pub x_m: f32,
    pub y_m: f32,
    pub heading_rad: f32,
    pub forward_m_s: f32,
    pub turn_rad_s: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InstantActionContext {
    pub action: Option<ActionPrimitive>,
    pub action_features: Vec<f32>,
    pub source: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InstantProvenance {
    pub source: String,
    pub source_frame_id: Option<String>,
    pub sensation_count: usize,
    pub impression_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissingModality {
    pub modality: Modality,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedSensationRef {
    pub id: SensationId,
    pub parent_id: Option<SensationId>,
    pub modality: Modality,
    pub payload_kind: SensationPayloadKind,
    pub kind: String,
    pub source: String,
    pub summary: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedImpressionRef {
    pub id: ImpressionId,
    pub sensation_id: Option<SensationId>,
    pub experience_id: Option<ExperienceId>,
    pub kind: String,
    pub text: String,
    pub confidence: f32,
    pub vector: Option<EmbodiedVectorRef>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbodiedLineageEdge {
    pub parent_id: SensationId,
    pub child_id: SensationId,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedVectorRef {
    pub vectorizer_id: String,
    pub model_id: String,
    pub model_label: String,
    pub dim: usize,
    pub modality: Modality,
    pub payload_kind: SensationPayloadKind,
    pub source_kind: String,
    pub source_sensation_id: SensationId,
    pub purpose: String,
    pub collection: String,
    pub input_summary: String,
    pub is_fallback: bool,
    pub provenance: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedPredictionRef {
    pub offset_ms: TimeMs,
    pub text: String,
    pub confidence: f32,
    pub vector: Option<EmbodiedVectorRef>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedMemoryLinkRef {
    pub target_id: String,
    pub relation: String,
    pub score: f32,
    pub text: Option<String>,
}

impl EmbodiedContext {
    pub fn derived_sensation_count(&self) -> usize {
        self.sensations
            .iter()
            .filter(|sensation| sensation.parent_id.is_some())
            .count()
    }

    pub fn from_current_experience(
        experience: Option<&Experience>,
        sensations: &[Sensation],
        impressions: &[Impression],
        futures: &[FuturePrediction],
        recollections: &[RecalledExperience],
    ) -> Self {
        let sensation_scope = experience
            .map(|experience| {
                experience
                    .sensation_ids
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_else(|| sensations.iter().map(|sensation| sensation.id).collect());
        let impression_scope = experience
            .map(|experience| {
                experience
                    .impression_ids
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();

        let sensation_refs = sensations
            .iter()
            .filter(|sensation| sensation_scope.contains(&sensation.id))
            .map(|sensation| EmbodiedSensationRef {
                id: sensation.id,
                parent_id: sensation.parent_id,
                modality: sensation.modality.clone(),
                payload_kind: sensation.payload_kind.clone(),
                kind: sensation.kind.clone(),
                source: sensation.source.clone(),
                summary: sensation.summary.clone(),
            })
            .collect::<Vec<_>>();
        let scoped_sensation_ids = sensation_refs
            .iter()
            .map(|sensation| sensation.id)
            .collect::<BTreeSet<_>>();
        let impression_refs = impressions
            .iter()
            .filter(|impression| {
                impression_scope.contains(&impression.id)
                    || impression
                        .sensation_id
                        .map(|id| scoped_sensation_ids.contains(&id))
                        .unwrap_or(false)
                    || impression
                        .about
                        .iter()
                        .any(|id| scoped_sensation_ids.contains(id))
            })
            .map(|impression| EmbodiedImpressionRef {
                id: impression.id,
                sensation_id: impression.sensation_id,
                experience_id: impression.experience_id,
                kind: impression.kind.clone(),
                text: impression.text.clone(),
                confidence: impression.confidence,
                vector: impression.vector.as_ref().map(vector_ref),
            })
            .collect::<Vec<_>>();
        let lineage = sensation_refs
            .iter()
            .filter_map(|sensation| {
                let parent_id = sensation.parent_id?;
                if scoped_sensation_ids.contains(&parent_id) {
                    Some(EmbodiedLineageEdge {
                        parent_id,
                        child_id: sensation.id,
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let sensation_vectors = sensations
            .iter()
            .filter(|sensation| scoped_sensation_ids.contains(&sensation.id))
            .filter_map(|sensation| sensation.vector.as_ref().map(vector_ref))
            .collect::<Vec<_>>();
        let impression_vectors = impressions
            .iter()
            .filter(|impression| {
                impression_scope.contains(&impression.id)
                    || impression
                        .sensation_id
                        .map(|id| scoped_sensation_ids.contains(&id))
                        .unwrap_or(false)
            })
            .filter_map(|impression| impression.vector.as_ref().map(vector_ref))
            .collect::<Vec<_>>();
        let mut predictions = experience
            .map(|experience| {
                experience
                    .predictions
                    .iter()
                    .map(|prediction| EmbodiedPredictionRef {
                        offset_ms: prediction.offset_ms,
                        text: prediction.text.clone(),
                        confidence: prediction.confidence,
                        vector: prediction.vector.as_ref().map(vector_ref),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        predictions.extend(futures.iter().filter_map(|future| {
            future
                .summary
                .as_ref()
                .map(|summary| EmbodiedPredictionRef {
                    offset_ms: future.offset_ms,
                    text: summary.clone(),
                    confidence: future.confidence,
                    vector: None,
                })
        }));
        let mut memory_links = experience
            .map(|experience| {
                experience
                    .memory_links
                    .iter()
                    .map(|link| EmbodiedMemoryLinkRef {
                        target_id: link.target_id.clone(),
                        relation: link.relation.clone(),
                        score: link.score,
                        text: link
                            .payload
                            .get("text")
                            .and_then(|value| value.as_str())
                            .map(str::to_string),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        memory_links.extend(
            recollections
                .iter()
                .map(|recollection| EmbodiedMemoryLinkRef {
                    target_id: recollection.experience.id.to_string(),
                    relation: "recalled_experience".to_string(),
                    score: recollection.score,
                    text: Some(recollection.experience.text.clone()),
                }),
        );

        let summary = experience
            .map(|experience| experience.text.clone())
            .or_else(|| {
                impression_refs
                    .last()
                    .map(|impression| impression.text.clone())
            })
            .unwrap_or_default();
        Self {
            experience_id: experience.map(|experience| experience.id),
            summary,
            sensations: sensation_refs,
            impressions: impression_refs,
            lineage,
            sensation_vectors,
            impression_vectors,
            predictions,
            memory_links,
        }
    }
}

impl ExperienceInstant {
    pub async fn from_now(now: &Now, action: Option<ActionPrimitive>) -> Result<Self> {
        let embodied = embody_now(now).await?;
        Ok(Self::from_embodied_now(
            &embodied, now, action, None, "live-now",
        ))
    }

    pub fn from_embodied_now(
        embodied: &EmbodiedNow,
        now: &Now,
        action: Option<ActionPrimitive>,
        source_frame_id: Option<String>,
        source: impl Into<String>,
    ) -> Self {
        Self::from_parts(
            Some(&embodied.experience),
            &embodied.sensations,
            &embodied.impressions,
            &[],
            &[],
            now,
            action,
            source_frame_id,
            source,
        )
    }

    pub fn from_now_features(now: &Now, action: Option<ActionPrimitive>) -> Self {
        let target = experience_decode_target_from_now(now);
        let teacher_vectors = [
            (
                "now.body",
                Modality::Odometry,
                SensationPayloadKind::OdometryEvent,
                target.body_features,
            ),
            (
                "now.memory",
                Modality::Memory,
                SensationPayloadKind::MemoryRecall,
                target.memory_features,
            ),
            (
                "now.drive",
                Modality::Other,
                SensationPayloadKind::Structured,
                target.drive_features,
            ),
            (
                "now.prediction",
                Modality::Other,
                SensationPayloadKind::Structured,
                target.prediction_features,
            ),
            (
                "now.eye",
                Modality::Vision,
                SensationPayloadKind::ImageBytes,
                target.eye_features,
            ),
            (
                "now.ear",
                Modality::Audio,
                SensationPayloadKind::AudioPcm,
                target.ear_features,
            ),
        ]
        .into_iter()
        .map(
            |(source, modality, payload_kind, vector)| InstantTeacherVector {
                metadata: EmbodiedVectorRef {
                    vectorizer_id: "pete.now.features.v1".to_string(),
                    model_id: "pete.now.features.v1".to_string(),
                    model_label: "Now deterministic feature vector".to_string(),
                    dim: vector.len(),
                    modality,
                    payload_kind,
                    source_kind: "now_feature".to_string(),
                    source_sensation_id: Uuid::nil(),
                    purpose: "experience_encode_feature".to_string(),
                    collection: "experience_encode_inputs".to_string(),
                    input_summary: source.to_string(),
                    is_fallback: true,
                    provenance: "now-feature-conversion".to_string(),
                },
                vector,
            },
        )
        .collect::<Vec<_>>();
        let present_modalities = teacher_vectors
            .iter()
            .map(|vector| vector.metadata.modality.clone())
            .collect::<BTreeSet<_>>();
        Self {
            schema_version: 1,
            t_ms: now.t_ms,
            window_start_ms: now.t_ms,
            window_end_ms: now.t_ms,
            experience_id: None,
            summary: format!(
                "I am at t={}ms with battery {:.2}.",
                now.t_ms, now.body.battery_level
            ),
            primary_sensations: Vec::new(),
            descendant_sensations: Vec::new(),
            impressions: Vec::new(),
            summary_impression: None,
            teacher_vectors,
            body_context: InstantBodyContext::from_now(now),
            action_context: InstantActionContext::from_action(action),
            lineage: Vec::new(),
            memory_links: Vec::new(),
            predictions: Vec::new(),
            provenance: InstantProvenance {
                source: "now-feature-conversion".to_string(),
                source_frame_id: None,
                sensation_count: 0,
                impression_count: 0,
            },
            missing_modalities: expected_instant_modalities()
                .into_iter()
                .filter(|modality| !present_modalities.contains(modality))
                .map(|modality| MissingModality {
                    modality,
                    reason: "no feature vector for modality in this Now conversion".to_string(),
                })
                .collect(),
        }
    }

    pub fn from_parts(
        experience: Option<&Experience>,
        sensations: &[Sensation],
        impressions: &[Impression],
        futures: &[FuturePrediction],
        recollections: &[RecalledExperience],
        now: &Now,
        action: Option<ActionPrimitive>,
        source_frame_id: Option<String>,
        source: impl Into<String>,
    ) -> Self {
        let context = EmbodiedContext::from_current_experience(
            experience,
            sensations,
            impressions,
            futures,
            recollections,
        );
        let primary_sensations = context
            .sensations
            .iter()
            .filter(|sensation| sensation.parent_id.is_none())
            .cloned()
            .collect::<Vec<_>>();
        let descendant_sensations = context
            .sensations
            .iter()
            .filter(|sensation| sensation.parent_id.is_some())
            .cloned()
            .collect::<Vec<_>>();
        let scoped_sensation_ids = context
            .sensations
            .iter()
            .map(|sensation| sensation.id)
            .collect::<BTreeSet<_>>();
        let scoped_impression_ids = context
            .impressions
            .iter()
            .map(|impression| impression.id)
            .collect::<BTreeSet<_>>();
        let teacher_vectors = sensations
            .iter()
            .filter(|sensation| scoped_sensation_ids.contains(&sensation.id))
            .filter_map(|sensation| sensation.vector.as_ref().map(instant_teacher_vector))
            .chain(
                impressions
                    .iter()
                    .filter(|impression| scoped_impression_ids.contains(&impression.id))
                    .filter_map(|impression| {
                        impression.vector.as_ref().map(instant_teacher_vector)
                    }),
            )
            .collect::<Vec<_>>();
        let summary_impression = experience
            .and_then(|experience| experience.summary_impression.as_ref())
            .map(|impression| EmbodiedImpressionRef {
                id: impression.id,
                sensation_id: impression.sensation_id,
                experience_id: impression.experience_id,
                kind: impression.kind.clone(),
                text: impression.text.clone(),
                confidence: impression.confidence,
                vector: impression.vector.as_ref().map(vector_ref),
            });
        let mut present_modalities = context
            .sensations
            .iter()
            .map(|sensation| sensation.modality.clone())
            .collect::<BTreeSet<_>>();
        present_modalities.extend(
            teacher_vectors
                .iter()
                .map(|vector| vector.metadata.modality.clone()),
        );

        Self {
            schema_version: 1,
            t_ms: now.t_ms,
            window_start_ms: experience
                .map(|experience| experience.window_start_ms)
                .unwrap_or(now.t_ms),
            window_end_ms: experience
                .map(|experience| experience.window_end_ms)
                .unwrap_or(now.t_ms),
            experience_id: experience.map(|experience| experience.id),
            summary: context.summary,
            primary_sensations,
            descendant_sensations,
            impressions: context.impressions,
            summary_impression,
            teacher_vectors,
            body_context: InstantBodyContext::from_now(now),
            action_context: InstantActionContext::from_action(action),
            lineage: context.lineage,
            memory_links: context.memory_links,
            predictions: context.predictions,
            provenance: InstantProvenance {
                source: source.into(),
                source_frame_id,
                sensation_count: sensations.len(),
                impression_count: impressions.len(),
            },
            missing_modalities: expected_instant_modalities()
                .into_iter()
                .filter(|modality| !present_modalities.contains(modality))
                .map(|modality| MissingModality {
                    modality,
                    reason: "no sensation or teacher vector for modality in this Instant"
                        .to_string(),
                })
                .collect(),
        }
    }

    pub fn encode_input(&self) -> ExperienceEncodeInput {
        let mut sense_vectors = self
            .teacher_vectors
            .iter()
            .map(|vector| {
                vector
                    .vector
                    .iter()
                    .copied()
                    .map(sanitize_feature)
                    .collect()
            })
            .collect::<Vec<Vec<f32>>>();
        sense_vectors.push(self.modality_mask());
        sense_vectors.push(self.body_features());
        sense_vectors.push(self.action_context.action_features.clone());
        ExperienceEncodeInput { sense_vectors }
    }

    pub fn coverage(&self) -> InstantCoverage {
        let missing = self
            .missing_modalities
            .iter()
            .map(|missing| missing.modality.clone())
            .collect::<BTreeSet<_>>();
        let present_modalities = expected_instant_modalities()
            .into_iter()
            .filter(|modality| !missing.contains(modality))
            .map(|modality| modality.as_str().to_string())
            .collect();
        let missing_modalities = self
            .missing_modalities
            .iter()
            .map(|missing| missing.modality.as_str().to_string())
            .collect();
        InstantCoverage {
            present_modalities,
            missing_modalities,
            sensation_count: self.primary_sensations.len() + self.descendant_sensations.len(),
            descendant_count: self.descendant_sensations.len(),
            vector_count: self.teacher_vectors.len(),
            impression_count: self.impressions.len()
                + usize::from(self.summary_impression.is_some()),
        }
    }

    pub fn embodied_context(&self) -> EmbodiedContext {
        let mut sensations = self.primary_sensations.clone();
        sensations.extend(self.descendant_sensations.clone());
        let sensation_vectors = self
            .teacher_vectors
            .iter()
            .filter(|vector| vector.metadata.source_kind == "sensation")
            .map(|vector| vector.metadata.clone())
            .collect();
        let impression_vectors = self
            .teacher_vectors
            .iter()
            .filter(|vector| vector.metadata.source_kind == "impression")
            .map(|vector| vector.metadata.clone())
            .collect();
        EmbodiedContext {
            experience_id: self.experience_id,
            summary: self.summary.clone(),
            sensations,
            impressions: self.impressions.clone(),
            lineage: self.lineage.clone(),
            sensation_vectors,
            impression_vectors,
            predictions: self.predictions.clone(),
            memory_links: self.memory_links.clone(),
        }
    }

    pub fn modality_mask(&self) -> Vec<f32> {
        let missing = self
            .missing_modalities
            .iter()
            .map(|missing| missing.modality.clone())
            .collect::<BTreeSet<_>>();
        expected_instant_modalities()
            .into_iter()
            .map(|modality| {
                if missing.contains(&modality) {
                    0.0
                } else {
                    1.0
                }
            })
            .collect()
    }

    fn body_features(&self) -> Vec<f32> {
        vec![
            self.body_context.battery_level,
            bool01(self.body_context.charging),
            bool01(self.body_context.bump),
            bool01(self.body_context.cliff),
            bool01(self.body_context.wheel_drop),
            bool01(self.body_context.wall),
            self.body_context.x_m.tanh(),
            self.body_context.y_m.tanh(),
            self.body_context.heading_rad.sin(),
            self.body_context.heading_rad.cos(),
            self.body_context.forward_m_s.clamp(-1.0, 1.0),
            self.body_context.turn_rad_s.clamp(-1.0, 1.0),
        ]
    }
}

impl ExperienceEncodeInput {
    pub fn from_instant(instant: &ExperienceInstant) -> Self {
        instant.encode_input()
    }
}

impl InstantBodyContext {
    pub fn from_now(now: &Now) -> Self {
        Self {
            battery_level: now.body.battery_level.clamp(0.0, 1.0),
            charging: now.body.charging,
            bump: now.body.flags.bump_left || now.body.flags.bump_right,
            cliff: cliff_detected(now),
            wheel_drop: now.body.flags.wheel_drop,
            wall: now.body.flags.wall || now.body.flags.virtual_wall,
            x_m: now.body.odometry.x_m,
            y_m: now.body.odometry.y_m,
            heading_rad: now.body.odometry.heading_rad,
            forward_m_s: now.body.velocity.forward_m_s,
            turn_rad_s: now.body.velocity.turn_rad_s,
        }
    }
}

impl InstantActionContext {
    pub fn from_action(action: Option<ActionPrimitive>) -> Self {
        Self {
            action_features: action_features(action.as_ref()),
            action,
            source: Some("action_primitive".to_string()),
        }
    }
}

fn expected_instant_modalities() -> Vec<Modality> {
    vec![
        Modality::Vision,
        Modality::Audio,
        Modality::Depth,
        Modality::Lidar,
        Modality::Touch,
        Modality::Odometry,
        Modality::Memory,
        Modality::Language,
    ]
}

fn instant_teacher_vector(vector: &VectorEmbedding) -> InstantTeacherVector {
    InstantTeacherVector {
        vector: vector
            .vector
            .iter()
            .copied()
            .map(sanitize_feature)
            .collect(),
        metadata: vector_ref(vector),
    }
}

fn vector_ref(vector: &VectorEmbedding) -> EmbodiedVectorRef {
    EmbodiedVectorRef {
        vectorizer_id: vector.vectorizer_id.clone(),
        model_id: vector.model_id.clone(),
        model_label: vector.model_label.clone(),
        dim: vector.dim,
        modality: vector.modality.clone(),
        payload_kind: vector.payload_kind.clone(),
        source_kind: vector.source_kind.clone(),
        source_sensation_id: vector.source_sensation_id,
        purpose: vector.purpose.clone(),
        collection: vector.collection.clone(),
        input_summary: vector.input_summary.clone(),
        is_fallback: vector.is_fallback,
        provenance: vector.provenance.clone(),
    }
}
