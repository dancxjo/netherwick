#[async_trait]
pub trait SensationVectorizer: Send + Sync {
    fn vectorizer_id(&self) -> &str;
    fn modality(&self) -> Modality;
    fn payload_kind(&self) -> SensationPayloadKind;
    fn model_id(&self) -> &str;
    fn model_label(&self) -> &str {
        self.model_id()
    }
    fn output_dim(&self) -> usize;
    fn purpose(&self) -> &str;
    fn collection(&self) -> &str {
        self.purpose()
    }
    fn is_fallback(&self) -> bool {
        false
    }
    async fn vectorize(&self, sensation: &Sensation) -> Result<VectorEmbedding>;
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedVectorizerRegistryConfig {
    #[serde(default)]
    pub vectorizer: BTreeMap<String, EmbodiedVectorizerConfig>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedVectorizerConfig {
    #[serde(default = "default_vectorizer_enabled")]
    pub enabled: bool,
    pub model: Option<String>,
    pub model_label: Option<String>,
    pub model_path: Option<String>,
    pub purpose: Option<String>,
    pub collection: Option<String>,
    pub fallback: Option<String>,
}

fn default_vectorizer_enabled() -> bool {
    true
}

#[derive(Clone, Default)]
pub struct SensationVectorizerRegistry {
    vectorizers: BTreeMap<(Modality, SensationPayloadKind), Arc<dyn SensationVectorizer>>,
    duplicate_state: Arc<Mutex<BTreeMap<(Modality, SensationPayloadKind), Vec<f32>>>>,
}

impl SensationVectorizerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_defaults() -> Self {
        Self::from_config(&EmbodiedVectorizerRegistryConfig::default())
    }

    pub fn from_config(config: &EmbodiedVectorizerRegistryConfig) -> Self {
        let mut registry = Self::new();

        registry.register_configured(
            config,
            "vision_image",
            EmbodiedFeatureSensationVectorizer::image(
                "pete.vectorizer.vision_image.frame_stats.v1",
                SensationPayloadKind::ImageBytes,
                "scene_similarity",
            ),
        );
        registry.register_configured(
            config,
            "vision_crop",
            EmbodiedFeatureSensationVectorizer::image(
                "pete.vectorizer.vision_crop.frame_stats.v1",
                SensationPayloadKind::Crop,
                "face_identity",
            ),
        );
        registry.register_configured(
            config,
            "vision_features",
            EmbodiedFeatureSensationVectorizer {
                vectorizer_id: "pete.vectorizer.vision_features.artifact.v1".to_string(),
                modality: Modality::Vision,
                payload_kind: SensationPayloadKind::Structured,
                model_id: "pete.image.feature_artifact.v1".to_string(),
                model_label: "pete.image.feature_artifact.v1".to_string(),
                purpose: "visual_similarity".to_string(),
                collection: "visual_similarity".to_string(),
                kind: EmbodiedFeatureKind::Image,
            },
        );
        registry.register_configured(
            config,
            "audio_pcm",
            EmbodiedFeatureSensationVectorizer::audio(
                "pete.vectorizer.audio_pcm.window_stats.v1",
                SensationPayloadKind::AudioPcm,
                "voice_identity",
            ),
        );
        registry.register_configured(
            config,
            "audio_voice",
            EmbodiedFeatureSensationVectorizer::audio(
                "pete.vectorizer.audio_voice.window_stats.v1",
                SensationPayloadKind::VoiceSegment,
                "voice_identity",
            ),
        );
        registry.register_configured(
            config,
            "audio_speech",
            EmbodiedFeatureSensationVectorizer::text(
                "pete.vectorizer.audio_speech.text_hashing.v1",
                SensationPayloadKind::SpeechSegment,
                "transcript_semantic",
            ),
        );
        registry.register_configured(
            config,
            "audio_transcript",
            EmbodiedFeatureSensationVectorizer::text(
                "pete.vectorizer.audio_transcript.text_hashing.v1",
                SensationPayloadKind::TranscriptSpan,
                "transcript_semantic",
            ),
        );
        registry.register_configured(
            config,
            "audio_phoneme",
            EmbodiedFeatureSensationVectorizer::text(
                "pete.vectorizer.audio_phoneme.text_hashing.v1",
                SensationPayloadKind::PhonemeSpan,
                "transcript_semantic",
            ),
        );
        registry.register_configured(
            config,
            "depth_scene",
            EmbodiedFeatureSensationVectorizer::depth(
                "pete.vectorizer.depth_scene.scene_stats.v1",
                SensationPayloadKind::DepthFrame,
                "scene_similarity",
            ),
        );

        for (modality, payload_kind) in default_vectorizer_keys() {
            if registry.get(&modality, &payload_kind).is_none() {
                registry.register(PlaceholderSensationVectorizer::new(modality, payload_kind));
            }
        }
        registry
    }

    pub fn from_models_toml(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|error| anyhow!("read vectorizer config {}: {error}", path.display()))?;
        let config: EmbodiedVectorizerRegistryConfig = toml::from_str(&text)
            .map_err(|error| anyhow!("parse vectorizer config {}: {error}", path.display()))?;
        Ok(Self::from_config(&config))
    }

    fn register_configured(
        &mut self,
        config: &EmbodiedVectorizerRegistryConfig,
        key: &str,
        mut vectorizer: EmbodiedFeatureSensationVectorizer,
    ) {
        let entry = config.vectorizer.get(key);
        if entry.is_some_and(|entry| !entry.enabled) {
            self.register(PlaceholderSensationVectorizer::new(
                vectorizer.modality(),
                vectorizer.payload_kind(),
            ));
            return;
        }
        if let Some(path) = entry.and_then(|entry| entry.model_path.as_deref()) {
            if !Path::new(path).exists() {
                eprintln!(
                    "warning: vectorizer {key} model_path {path} is missing; using deterministic placeholder fallback"
                );
                self.register(PlaceholderSensationVectorizer::new(
                    vectorizer.modality(),
                    vectorizer.payload_kind(),
                ));
                return;
            }
        }
        if let Some(model) = entry.and_then(|entry| entry.model.clone()) {
            vectorizer.model_id = model.clone();
            vectorizer.model_label = model;
        }
        if let Some(label) = entry.and_then(|entry| entry.model_label.clone()) {
            vectorizer.model_label = label;
        }
        if let Some(purpose) = entry.and_then(|entry| entry.purpose.clone()) {
            vectorizer.purpose = purpose;
        }
        if let Some(collection) = entry.and_then(|entry| entry.collection.clone()) {
            vectorizer.collection = collection;
        }
        self.register(vectorizer);
    }

    pub fn register<V>(&mut self, vectorizer: V)
    where
        V: SensationVectorizer + 'static,
    {
        self.vectorizers.insert(
            (vectorizer.modality(), vectorizer.payload_kind()),
            Arc::new(vectorizer),
        );
    }

    pub fn get(
        &self,
        modality: &Modality,
        payload_kind: &SensationPayloadKind,
    ) -> Option<Arc<dyn SensationVectorizer>> {
        self.vectorizers
            .get(&(modality.clone(), payload_kind.clone()))
            .cloned()
    }

    pub async fn vectorize(&self, sensation: &Sensation) -> Result<Option<VectorEmbedding>> {
        let Some(vectorizer) = self.get(&sensation.modality, &sensation.payload_kind) else {
            return Ok(None);
        };
        let embedding = vectorizer.vectorize(sensation).await?;
        if should_suppress_duplicate_embedding(sensation, &embedding) {
            let key = (embedding.modality.clone(), embedding.payload_kind.clone());
            let mut duplicate_state = self
                .duplicate_state
                .lock()
                .map_err(|_| anyhow!("vectorizer duplicate suppression lock poisoned"))?;
            let duplicate = duplicate_state
                .get(&key)
                .is_some_and(|previous| cosine_similarity(previous, &embedding.vector) > 0.999);
            if duplicate {
                return Ok(None);
            }
            duplicate_state.insert(key, embedding.vector.clone());
        }
        Ok(Some(embedding))
    }
}

fn default_vectorizer_keys() -> [(Modality, SensationPayloadKind); 13] {
    [
        (Modality::Vision, SensationPayloadKind::ImageBytes),
        (Modality::Vision, SensationPayloadKind::Crop),
        (Modality::Vision, SensationPayloadKind::Structured),
        (Modality::Audio, SensationPayloadKind::AudioPcm),
        (Modality::Audio, SensationPayloadKind::VoiceSegment),
        (Modality::Audio, SensationPayloadKind::SpeechSegment),
        (Modality::Audio, SensationPayloadKind::TranscriptSpan),
        (Modality::Audio, SensationPayloadKind::PhonemeSpan),
        (Modality::Depth, SensationPayloadKind::DepthFrame),
        (Modality::Touch, SensationPayloadKind::ContactEvent),
        (Modality::Odometry, SensationPayloadKind::OdometryEvent),
        (Modality::Memory, SensationPayloadKind::MemoryRecall),
        (Modality::Other, SensationPayloadKind::Structured),
    ]
}

#[derive(Clone, Debug)]
enum EmbodiedFeatureKind {
    Image,
    Audio,
    Text,
    Depth,
}

#[derive(Clone, Debug)]
pub struct EmbodiedFeatureSensationVectorizer {
    vectorizer_id: String,
    modality: Modality,
    payload_kind: SensationPayloadKind,
    model_id: String,
    model_label: String,
    purpose: String,
    collection: String,
    kind: EmbodiedFeatureKind,
}

impl EmbodiedFeatureSensationVectorizer {
    pub fn image(
        vectorizer_id: impl Into<String>,
        payload_kind: SensationPayloadKind,
        purpose: impl Into<String>,
    ) -> Self {
        let model_id = "pete.image.frame_stats.v1".to_string();
        let purpose = purpose.into();
        Self {
            vectorizer_id: vectorizer_id.into(),
            modality: Modality::Vision,
            payload_kind,
            model_id: model_id.clone(),
            model_label: model_id,
            collection: purpose.clone(),
            purpose,
            kind: EmbodiedFeatureKind::Image,
        }
    }

    pub fn audio(
        vectorizer_id: impl Into<String>,
        payload_kind: SensationPayloadKind,
        purpose: impl Into<String>,
    ) -> Self {
        let model_id = "pete.audio.window_stats.v1".to_string();
        let purpose = purpose.into();
        Self {
            vectorizer_id: vectorizer_id.into(),
            modality: Modality::Audio,
            payload_kind,
            model_id: model_id.clone(),
            model_label: model_id,
            collection: purpose.clone(),
            purpose,
            kind: EmbodiedFeatureKind::Audio,
        }
    }

    pub fn text(
        vectorizer_id: impl Into<String>,
        payload_kind: SensationPayloadKind,
        purpose: impl Into<String>,
    ) -> Self {
        let model_id = TEXT_HASH_MODEL_ID.to_string();
        let purpose = purpose.into();
        Self {
            vectorizer_id: vectorizer_id.into(),
            modality: Modality::Audio,
            payload_kind,
            model_id: model_id.clone(),
            model_label: model_id,
            collection: purpose.clone(),
            purpose,
            kind: EmbodiedFeatureKind::Text,
        }
    }

    pub fn depth(
        vectorizer_id: impl Into<String>,
        payload_kind: SensationPayloadKind,
        purpose: impl Into<String>,
    ) -> Self {
        let model_id = "pete.depth.scene_stats.v1".to_string();
        let purpose = purpose.into();
        Self {
            vectorizer_id: vectorizer_id.into(),
            modality: Modality::Depth,
            payload_kind,
            model_id: model_id.clone(),
            model_label: model_id,
            collection: purpose.clone(),
            purpose,
            kind: EmbodiedFeatureKind::Depth,
        }
    }
}

#[async_trait]
impl SensationVectorizer for EmbodiedFeatureSensationVectorizer {
    fn vectorizer_id(&self) -> &str {
        &self.vectorizer_id
    }

    fn modality(&self) -> Modality {
        self.modality.clone()
    }

    fn payload_kind(&self) -> SensationPayloadKind {
        self.payload_kind.clone()
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_label(&self) -> &str {
        &self.model_label
    }

    fn output_dim(&self) -> usize {
        EMBODIED_FEATURE_VECTOR_DIM
    }

    fn purpose(&self) -> &str {
        &self.purpose
    }

    fn collection(&self) -> &str {
        &self.collection
    }

    async fn vectorize(&self, sensation: &Sensation) -> Result<VectorEmbedding> {
        if let Some(artifact) = precomputed_payload_embedding(sensation) {
            return Ok(VectorEmbedding::new(
                sanitize_vector(artifact.vector),
                artifact.model_id.clone(),
                self.modality.clone(),
                self.payload_kind.clone(),
                sensation.id,
                sensation.observed_at_ms,
            )
            .with_metadata(
                artifact.vectorizer_id,
                artifact.model_label,
                artifact.purpose,
                artifact.collection,
                artifact.input_summary,
                false,
                "precomputed_vector_artifact",
            ));
        }

        let vector = match self.kind {
            EmbodiedFeatureKind::Image => image_feature_vector(sensation),
            EmbodiedFeatureKind::Audio => audio_feature_vector(sensation),
            EmbodiedFeatureKind::Text => text_feature_vector(sensation),
            EmbodiedFeatureKind::Depth => depth_feature_vector(sensation),
        };
        Ok(VectorEmbedding::new(
            vector,
            self.model_id.clone(),
            self.modality.clone(),
            self.payload_kind.clone(),
            sensation.id,
            sensation.observed_at_ms,
        )
        .with_metadata(
            self.vectorizer_id.clone(),
            self.model_label.clone(),
            self.purpose.clone(),
            self.collection.clone(),
            input_summary_for_sensation(sensation),
            false,
            "pete_embodied_feature_vectorizer",
        ))
    }
}

#[derive(Clone, Debug)]
struct PrecomputedPayloadEmbedding {
    vector: Vec<f32>,
    model_id: String,
    model_label: String,
    vectorizer_id: String,
    purpose: String,
    collection: String,
    input_summary: String,
}

fn precomputed_payload_embedding(sensation: &Sensation) -> Option<PrecomputedPayloadEmbedding> {
    let artifacts = sensation.payload.get("vector_artifacts")?.as_array()?;
    for artifact in artifacts {
        let vector = artifact
            .get("vector")
            .and_then(Value::as_array)?
            .iter()
            .filter_map(|value| value.as_f64().map(|value| value as f32))
            .collect::<Vec<_>>();
        if vector.is_empty() {
            continue;
        }
        let model_id = artifact
            .get("model")
            .and_then(Value::as_str)
            .filter(|model| !model.trim().is_empty())
            .unwrap_or("pete.precomputed_vector.v0")
            .to_string();
        let collection = artifact
            .get("collection")
            .and_then(Value::as_str)
            .filter(|collection| !collection.trim().is_empty())
            .unwrap_or("precomputed_vectors")
            .to_string();
        let purpose = purpose_for_collection(&collection);
        let point_id = artifact
            .get("point_id")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let source_id = artifact
            .get("source_id")
            .and_then(Value::as_str)
            .or_else(|| artifact.get("source_frame_id").and_then(Value::as_str))
            .unwrap_or("unknown");
        return Some(PrecomputedPayloadEmbedding {
            vector,
            model_id: model_id.clone(),
            model_label: model_id.clone(),
            vectorizer_id: format!("precomputed.{collection}.{model_id}"),
            purpose,
            collection,
            input_summary: format!("vector_artifact point_id={point_id} source_id={source_id}"),
        });
    }
    None
}

fn image_feature_vector(sensation: &Sensation) -> Vec<f32> {
    let mut vector = base_sensation_features(sensation);
    let Some(frame) = VisualFrame::from_sensation(sensation) else {
        return pad_feature_vector(vector);
    };
    let pixels = (frame.width as usize).saturating_mul(frame.height as usize);
    if pixels == 0 {
        return pad_feature_vector(vector);
    }
    let mut sums = [0.0_f32; 3];
    let mut sq_sums = [0.0_f32; 3];
    let mut mins = [1.0_f32; 3];
    let mut maxs = [0.0_f32; 3];
    let mut luma_sum = 0.0_f32;
    let mut skin_like = 0_usize;
    for pixel in frame.rgb.chunks_exact(3).take(pixels) {
        let rgb = [
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
        ];
        for channel in 0..3 {
            sums[channel] += rgb[channel];
            sq_sums[channel] += rgb[channel] * rgb[channel];
            mins[channel] = mins[channel].min(rgb[channel]);
            maxs[channel] = maxs[channel].max(rgb[channel]);
        }
        luma_sum += 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        if is_skin_like_rgb(pixel[0], pixel[1], pixel[2]) {
            skin_like += 1;
        }
    }
    for channel in 0..3 {
        let mean = sums[channel] / pixels as f32;
        let variance = (sq_sums[channel] / pixels as f32) - mean * mean;
        vector.push(mean);
        vector.push(variance.max(0.0).sqrt());
        vector.push(mins[channel]);
        vector.push(maxs[channel]);
    }
    vector.push(luma_sum / pixels as f32);
    vector.push(skin_like as f32 / pixels as f32);
    if let Some(bbox) = sensation.metadata.bbox {
        vector.push((bbox.x as f32 / frame.width.max(1) as f32).clamp(0.0, 1.0));
        vector.push((bbox.y as f32 / frame.height.max(1) as f32).clamp(0.0, 1.0));
        vector.push((bbox.width as f32 / frame.width.max(1) as f32).clamp(0.0, 1.0));
        vector.push((bbox.height as f32 / frame.height.max(1) as f32).clamp(0.0, 1.0));
    }
    push_grid_luma(&mut vector, &frame);
    pad_feature_vector(vector)
}

fn push_grid_luma(vector: &mut Vec<f32>, frame: &VisualFrame) {
    let width = frame.width as usize;
    let height = frame.height as usize;
    if width == 0 || height == 0 {
        return;
    }
    for gy in 0..2 {
        for gx in 0..2 {
            let x0 = gx * width / 2;
            let x1 = ((gx + 1) * width / 2).max(x0 + 1).min(width);
            let y0 = gy * height / 2;
            let y1 = ((gy + 1) * height / 2).max(y0 + 1).min(height);
            let mut sum = 0.0_f32;
            let mut count = 0_usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = (y * width + x) * 3;
                    let r = frame.rgb[idx] as f32 / 255.0;
                    let g = frame.rgb[idx + 1] as f32 / 255.0;
                    let b = frame.rgb[idx + 2] as f32 / 255.0;
                    sum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    count += 1;
                }
            }
            vector.push(if count > 0 { sum / count as f32 } else { 0.0 });
        }
    }
}

fn audio_feature_vector(sensation: &Sensation) -> Vec<f32> {
    let mut vector = base_sensation_features(sensation);
    vector.push(
        sensation
            .metadata
            .duration_ms
            .map(|value| (value as f32 / 10_000.0).clamp(0.0, 1.0))
            .unwrap_or_default(),
    );
    for key in [
        "feature_sets",
        "duration_ms",
        "start_offset_ms",
        "end_offset_ms",
        "confidence",
    ] {
        vector.push(payload_number_unit(&sensation.payload, key));
    }
    if let Some(asr) = sensation.payload.get("asr") {
        vector.push(payload_number_unit(asr, "confidence"));
        vector.push(
            asr.get("is_final")
                .and_then(Value::as_bool)
                .map(bool01)
                .unwrap_or_default(),
        );
        vector.push(
            asr.get("word_count")
                .and_then(Value::as_u64)
                .map(|value| (value as f32 / 32.0).clamp(0.0, 1.0))
                .unwrap_or_default(),
        );
    }
    if let Some(text) = sensation
        .payload
        .get("transcript")
        .and_then(Value::as_str)
        .or_else(|| sensation.payload.get("text").and_then(Value::as_str))
    {
        push_text_hash_features(&mut vector, text, 8);
    }
    pad_feature_vector(vector)
}

fn text_feature_vector(sensation: &Sensation) -> Vec<f32> {
    let mut vector = base_sensation_features(sensation);
    let text = sensation
        .payload
        .get("text")
        .and_then(Value::as_str)
        .or(sensation.summary.as_deref())
        .unwrap_or_default();
    let chars = text.chars().count();
    let words = text.split_whitespace().count();
    vector.push((chars as f32 / 280.0).clamp(0.0, 1.0));
    vector.push((words as f32 / 48.0).clamp(0.0, 1.0));
    vector.push(bool01(
        text.chars()
            .last()
            .is_some_and(|ch| matches!(ch, '?' | '!')),
    ));
    push_text_hash_features(&mut vector, text, EMBODIED_FEATURE_VECTOR_DIM);
    pad_feature_vector(vector)
}

fn depth_feature_vector(sensation: &Sensation) -> Vec<f32> {
    let mut vector = base_sensation_features(sensation);
    for key in [
        "sample_count",
        "width",
        "height",
        "min_depth_m",
        "max_depth_m",
        "skeleton_count",
    ] {
        vector.push(payload_number_unit(&sensation.payload, key));
    }
    let min_depth = sensation
        .payload
        .get("min_depth_m")
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .unwrap_or_default();
    let max_depth = sensation
        .payload
        .get("max_depth_m")
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .unwrap_or_default();
    vector.push((max_depth - min_depth).max(0.0).min(10.0) / 10.0);
    pad_feature_vector(vector)
}

fn base_sensation_features(sensation: &Sensation) -> Vec<f32> {
    let mut vector = Vec::with_capacity(EMBODIED_FEATURE_VECTOR_DIM);
    vector.push(stable_unit(&sensation.kind));
    vector.push(stable_unit(&sensation.source));
    vector.push((sensation.occurred_at_ms % 10_000) as f32 / 10_000.0);
    vector.push(sensation.metadata.confidence.unwrap_or(0.5).clamp(0.0, 1.0));
    vector.push(bool01(sensation.parent_id.is_some()));
    for label in sensation.metadata.labels.iter().take(4) {
        vector.push(stable_unit(label));
    }
    vector
}

fn push_text_hash_features(vector: &mut Vec<f32>, text: &str, max_dim: usize) {
    let reserve = max_dim.saturating_sub(vector.len());
    if reserve == 0 {
        return;
    }
    let mut buckets = vec![0.0_f32; reserve.min(16)];
    for token in text.split_whitespace() {
        let mut hash = 0_u32;
        for byte in token.bytes() {
            hash = hash.wrapping_mul(16777619) ^ u32::from(byte.to_ascii_lowercase());
        }
        let idx = (hash as usize) % buckets.len();
        buckets[idx] += 1.0;
    }
    let norm = buckets
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    for bucket in buckets {
        vector.push(if norm > 0.0 { bucket / norm } else { 0.0 });
    }
}

fn payload_number_unit(payload: &Value, key: &str) -> f32 {
    payload
        .get(key)
        .and_then(Value::as_f64)
        .map(|value| (value as f32).abs())
        .map(|value| (value / (value + 1.0)).clamp(0.0, 1.0))
        .unwrap_or_default()
}

fn pad_feature_vector(mut vector: Vec<f32>) -> Vec<f32> {
    vector = sanitize_vector(vector);
    vector.truncate(EMBODIED_FEATURE_VECTOR_DIM);
    while vector.len() < EMBODIED_FEATURE_VECTOR_DIM {
        vector.push(0.0);
    }
    vector
}

fn sanitize_vector(vector: Vec<f32>) -> Vec<f32> {
    vector
        .into_iter()
        .map(|value| {
            if value.is_finite() {
                value.clamp(-1.0, 1.0)
            } else {
                0.0
            }
        })
        .collect()
}

fn semantic_text_vector(
    text: &str,
    source_id: Uuid,
    generated_at_ms: TimeMs,
    source_kind: impl Into<String>,
    purpose: impl Into<String>,
    collection: impl Into<String>,
    input_summary: impl Into<String>,
) -> VectorEmbedding {
    let purpose = purpose.into();
    let collection = collection.into();
    VectorEmbedding::new(
        text_hash_vector(text, TEXT_HASH_VECTOR_DIM),
        TEXT_HASH_MODEL_ID,
        Modality::Other,
        SensationPayloadKind::Structured,
        source_id,
        generated_at_ms,
    )
    .with_metadata(
        format!("pete.vectorizer.{purpose}.text_hashing.v1"),
        "Pete deterministic text hashing baseline",
        purpose,
        collection,
        input_summary,
        false,
        "pete_text_hashing_vectorizer",
    )
    .with_source_kind(source_kind)
}

fn text_hash_vector(text: &str, dim: usize) -> Vec<f32> {
    let dim = dim.max(1);
    let mut vector = vec![0.0_f32; dim];
    let mut token_count = 0.0_f32;
    for token in text
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| !token.is_empty())
    {
        token_count += 1.0;
        let normalized = token.to_ascii_lowercase();
        for ngram in token_ngrams(&normalized) {
            let mut hash = 2166136261_u32;
            for byte in ngram.bytes() {
                hash = hash.wrapping_mul(16777619) ^ u32::from(byte);
            }
            let index = (hash as usize) % dim;
            let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
            vector[index] += sign;
        }
    }
    vector[0] += (text.chars().count() as f32 / 512.0).clamp(0.0, 1.0);
    if dim > 1 {
        vector[1] += (token_count / 96.0).clamp(0.0, 1.0);
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in &mut vector {
            *value = (*value / norm).clamp(-1.0, 1.0);
        }
    }
    vector
}

fn token_ngrams(token: &str) -> Vec<String> {
    let chars = token.chars().collect::<Vec<_>>();
    if chars.len() <= 3 {
        return vec![token.to_string()];
    }
    let mut ngrams = Vec::new();
    for window in chars.windows(3) {
        ngrams.push(window.iter().collect());
    }
    ngrams.push(token.to_string());
    ngrams
}

fn purpose_for_sensation(modality: &Modality, payload_kind: &SensationPayloadKind) -> String {
    match (modality, payload_kind) {
        (Modality::Vision, SensationPayloadKind::ImageBytes) => "scene_similarity",
        (Modality::Vision, SensationPayloadKind::Crop) => "face_identity",
        (Modality::Vision, SensationPayloadKind::Structured) => "visual_similarity",
        (Modality::Audio, SensationPayloadKind::TranscriptSpan)
        | (Modality::Audio, SensationPayloadKind::SpeechSegment)
        | (Modality::Audio, SensationPayloadKind::PhonemeSpan) => "transcript_semantic",
        (Modality::Audio, SensationPayloadKind::VoiceSegment)
        | (Modality::Audio, SensationPayloadKind::AudioPcm) => "voice_identity",
        (Modality::Depth, SensationPayloadKind::DepthFrame) => "scene_similarity",
        (Modality::Other, SensationPayloadKind::Structured) => "experience_semantic",
        _ => "embodied_similarity",
    }
    .to_string()
}

fn purpose_for_collection(collection: &str) -> String {
    match collection {
        "faces" => "face_identity",
        "objects" => "object_identity",
        "voices" => "voice_identity",
        "scene_vectors" | "images" => "scene_similarity",
        "image_descriptions" | "memories" | "transcripts" => "transcript_semantic",
        "impressions" => "impression_semantic",
        "experiences" => "experience_semantic",
        _ => collection,
    }
    .to_string()
}

fn input_summary_for_sensation(sensation: &Sensation) -> String {
    let mut parts = vec![
        format!("kind={}", sensation.kind),
        format!("payload_kind={}", sensation.payload_kind.as_str()),
    ];
    if let Some(summary) = sensation
        .summary
        .as_deref()
        .filter(|summary| !summary.is_empty())
    {
        parts.push(format!(
            "summary={}",
            summary.chars().take(96).collect::<String>()
        ));
    }
    if let Some(width) = sensation.payload.get("width").and_then(Value::as_u64) {
        if let Some(height) = sensation.payload.get("height").and_then(Value::as_u64) {
            parts.push(format!("size={}x{}", width, height));
        }
    }
    if let Some(format) = sensation.payload.get("format").and_then(Value::as_str) {
        parts.push(format!("format={format}"));
    }
    parts.join(" ")
}

fn should_suppress_duplicate_embedding(sensation: &Sensation, embedding: &VectorEmbedding) -> bool {
    !embedding.is_fallback
        && matches!(embedding.modality, Modality::Vision)
        && matches!(
            embedding.payload_kind,
            SensationPayloadKind::ImageBytes | SensationPayloadKind::Crop
        )
        && VisualFrame::from_sensation(sensation).is_some()
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut left_norm = 0.0_f32;
    let mut right_norm = 0.0_f32;
    for (left, right) in left.iter().zip(right.iter()) {
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    let denom = left_norm.sqrt() * right_norm.sqrt();
    if denom > 0.0 {
        dot / denom
    } else {
        0.0
    }
}

#[derive(Clone, Debug)]
pub struct PlaceholderSensationVectorizer {
    modality: Modality,
    payload_kind: SensationPayloadKind,
}

impl PlaceholderSensationVectorizer {
    pub fn new(modality: Modality, payload_kind: SensationPayloadKind) -> Self {
        Self {
            modality,
            payload_kind,
        }
    }
}

#[async_trait]
impl SensationVectorizer for PlaceholderSensationVectorizer {
    fn vectorizer_id(&self) -> &str {
        "pete.vectorizer.placeholder.v0"
    }

    fn modality(&self) -> Modality {
        self.modality.clone()
    }

    fn payload_kind(&self) -> SensationPayloadKind {
        self.payload_kind.clone()
    }

    fn model_id(&self) -> &str {
        "pete.placeholder.v0"
    }

    fn output_dim(&self) -> usize {
        PLACEHOLDER_VECTOR_DIM
    }

    fn purpose(&self) -> &str {
        "fallback_deterministic"
    }

    fn collection(&self) -> &str {
        "fallback_vectors"
    }

    fn is_fallback(&self) -> bool {
        true
    }

    async fn vectorize(&self, sensation: &Sensation) -> Result<VectorEmbedding> {
        let mut vector = vec![0.0; self.output_dim()];
        vector[0] = stable_unit(&sensation.kind);
        vector[1] = stable_unit(&sensation.source);
        vector[2] = (sensation.occurred_at_ms % 10_000) as f32 / 10_000.0;
        vector[3] = sensation.metadata.confidence.unwrap_or(0.5).clamp(0.0, 1.0);
        vector[4] = sensation
            .payload
            .get("width")
            .and_then(Value::as_u64)
            .map(|value| (value as f32 / 1920.0).clamp(0.0, 1.0))
            .unwrap_or_default();
        vector[5] = sensation
            .payload
            .get("height")
            .and_then(Value::as_u64)
            .map(|value| (value as f32 / 1080.0).clamp(0.0, 1.0))
            .unwrap_or_default();
        vector[6] = sensation
            .metadata
            .duration_ms
            .map(|value| (value as f32 / 5_000.0).clamp(0.0, 1.0))
            .unwrap_or_default();
        if sensation.parent_id.is_some() {
            vector[7] = 1.0;
        }
        for (idx, label) in sensation.metadata.labels.iter().take(4).enumerate() {
            vector[8 + idx] = stable_unit(label);
        }
        Ok(VectorEmbedding::new(
            vector,
            self.model_id(),
            self.modality.clone(),
            self.payload_kind.clone(),
            sensation.id,
            sensation.observed_at_ms,
        )
        .with_metadata(
            self.vectorizer_id(),
            self.model_id(),
            purpose_for_sensation(&self.modality, &self.payload_kind),
            self.collection(),
            input_summary_for_sensation(sensation),
            true,
            "deterministic_placeholder_fallback",
        ))
    }
}
