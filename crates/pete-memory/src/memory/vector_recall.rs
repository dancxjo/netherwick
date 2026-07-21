pub fn place_recognition_input_from_frame(frame: &ExperienceFrame) -> PlaceRecognitionInput {
    let instant = frame.experience_instant();
    let experience_id = instant.experience_id.map(|id| id.to_string());
    let experience_latent_vector = frame.z.as_ref().map(|latent| {
        let artifact = VectorArtifact::new(
            EXPERIENCE_VECTOR_COLLECTION,
            format!("{}:experience-latent", frame.id),
            latent.z.clone(),
        )
        .with_model("pete.experience.latent")
        .with_source_frame_id(frame.id.to_string())
        .with_occurred_at_ms(frame.t_ms);
        if let Some(experience_id) = &experience_id {
            artifact.with_source_id(experience_id.clone())
        } else {
            artifact
        }
    });
    PlaceRecognitionInput {
        experience_id,
        instant_frame_id: Some(frame.id.to_string()),
        experience_latent_vector,
        teacher_vector_refs: instant
            .teacher_vectors
            .iter()
            .map(|vector| embodied_vector_ref_id(&vector.metadata))
            .collect(),
        compact_range_summary: compact_range_summary(&frame.now),
        compact_depth_summary: compact_depth_summary(&frame.now),
        object_labels: object_labels(&frame.now, None),
        person_labels: object_labels(&frame.now, Some(ObjectClass::Person)),
        voice_labels: voice_labels(&frame.now),
        action: frame.chosen_action.clone(),
        pose: Some(frame.now.body.odometry),
        window_start_ms: instant.window_start_ms,
        window_end_ms: instant.window_end_ms,
        provenance: format!(
            "{}:{}",
            instant.provenance.source,
            instant
                .provenance
                .source_frame_id
                .as_deref()
                .unwrap_or("unknown-frame")
        ),
    }
}

pub fn place_recognition_input_from_query_now(
    now: &Now,
    latent: Option<&pete_experience::ExperienceLatent>,
    provenance: impl Into<String>,
) -> PlaceRecognitionInput {
    let experience_latent_vector = latent.map(|latent| {
        VectorArtifact::new(
            EXPERIENCE_VECTOR_COLLECTION,
            format!("query:{}:experience-latent", now.t_ms),
            latent.z.clone(),
        )
        .with_model("pete.experience.latent")
        .with_occurred_at_ms(now.t_ms)
    });
    PlaceRecognitionInput {
        experience_id: None,
        instant_frame_id: None,
        experience_latent_vector,
        teacher_vector_refs: now
            .eye
            .scene_vectors
            .iter()
            .chain(now.face.vectors.iter())
            .chain(now.objects.vectors.iter())
            .chain(now.voice.vectors.iter())
            .map(|artifact| format!("{}:{}", artifact.collection, artifact.point_id))
            .collect(),
        compact_range_summary: compact_range_summary(now),
        compact_depth_summary: compact_depth_summary(now),
        object_labels: object_labels(now, None),
        person_labels: object_labels(now, Some(ObjectClass::Person)),
        voice_labels: voice_labels(now),
        action: now.memory.best_remembered_action.clone(),
        pose: Some(now.body.odometry),
        window_start_ms: now.t_ms,
        window_end_ms: now.t_ms,
        provenance: provenance.into(),
    }
}

pub fn place_recognition_vectors_from_input(input: &PlaceRecognitionInput) -> Vec<VectorArtifact> {
    input
        .experience_latent_vector
        .iter()
        .cloned()
        .collect::<Vec<_>>()
}

fn compact_range_summary(now: &Now) -> Option<CompactRangeSummary> {
    let finite = now
        .range
        .beams
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() && now.range.nearest_m.is_none() {
        return None;
    }
    let mean_m = (!finite.is_empty()).then(|| finite.iter().sum::<f32>() / finite.len() as f32);
    Some(CompactRangeSummary {
        beam_count: now.range.beams.len(),
        nearest_m: now.range.nearest_m,
        mean_m,
    })
}

fn compact_depth_summary(now: &Now) -> Option<CompactDepthSummary> {
    let finite = now
        .kinect
        .depth_m
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return None;
    }
    let min_m = finite.iter().copied().reduce(f32::min);
    let max_m = finite.iter().copied().reduce(f32::max);
    let mean_m = Some(finite.iter().sum::<f32>() / finite.len() as f32);
    Some(CompactDepthSummary {
        sample_count: finite.len(),
        min_m,
        max_m,
        mean_m,
    })
}

fn object_labels(now: &Now, class: Option<ObjectClass>) -> Vec<String> {
    let mut labels = now
        .objects
        .observations
        .iter()
        .filter(|observation| {
            class
                .as_ref()
                .map(|class| observation.class == *class)
                .unwrap_or(true)
        })
        .map(|observation| observation.label.clone())
        .collect::<Vec<_>>();
    labels.sort();
    labels.dedup();
    labels
}

fn voice_labels(now: &Now) -> Vec<String> {
    now.ear
        .transcript
        .as_ref()
        .into_iter()
        .chain(now.ear.asr.transcript.as_ref())
        .map(|text| text.trim())
        .filter(|text| !text.is_empty())
        .map(str::to_string)
        .collect()
}

fn embodied_vector_ref_id(vector: &pete_experience::EmbodiedVectorRef) -> String {
    format!(
        "{}:{}:{}:{}",
        vector.collection, vector.model_id, vector.source_sensation_id, vector.dim
    )
}

fn query_face_vectors(query: &RecallQuery) -> Vec<&[f32]> {
    query
        .face_vectors
        .iter()
        .map(|artifact| artifact.vector.as_slice())
        .collect()
}

fn query_object_vectors(query: &RecallQuery) -> Vec<&[f32]> {
    query
        .object_vectors
        .iter()
        .map(|artifact| artifact.vector.as_slice())
        .collect()
}

fn query_voice_vectors(query: &RecallQuery) -> Vec<&[f32]> {
    query
        .voice_vectors
        .iter()
        .map(|artifact| artifact.vector.as_slice())
        .collect()
}

fn recall_vector_ids(record: &MemoryRecord) -> Vec<String> {
    let mut ids = record
        .experience_vectors
        .iter()
        .chain(record.sensation_vectors.iter())
        .chain(record.scene_vectors.iter())
        .chain(record.face_vectors.iter())
        .chain(record.object_vectors.iter())
        .chain(record.voice_vectors.iter())
        .map(|artifact| format!("{}:{}", artifact.collection, artifact.point_id))
        .collect::<Vec<_>>();
    ids.sort();
    ids.dedup();
    ids
}

fn query_all_vectors(query: &RecallQuery) -> Vec<&[f32]> {
    let mut vectors = query_scene_vectors(query);
    vectors.extend(query_face_vectors(query));
    vectors.extend(query_object_vectors(query));
    vectors.extend(query_voice_vectors(query));
    vectors
}

fn record_all_vectors(record: &MemoryRecord) -> Vec<&VectorArtifact> {
    record
        .scene_vectors
        .iter()
        .chain(record.face_vectors.iter())
        .chain(record.object_vectors.iter())
        .chain(record.voice_vectors.iter())
        .chain(record.sensation_vectors.iter())
        .chain(record.experience_vectors.iter())
        .collect()
}

fn vector_payload_key(artifact: &VectorArtifact) -> String {
    format!("{}:{}", artifact.collection, artifact.point_id)
}

fn merge_json_object(base: &mut serde_json::Value, extra: &serde_json::Value) {
    let (Some(base), Some(extra)) = (base.as_object_mut(), extra.as_object()) else {
        return;
    };
    for (key, value) in extra {
        base.insert(key.clone(), value.clone());
    }
}

fn has_face_query(query: &RecallQuery) -> bool {
    !query.face_vectors.is_empty()
}

fn has_object_query(query: &RecallQuery) -> bool {
    !query.object_vectors.is_empty()
}

fn has_voice_query(query: &RecallQuery) -> bool {
    !query.voice_vectors.is_empty()
}

fn max_vector_similarity(query_vectors: Vec<&[f32]>, record_vectors: Vec<&VectorArtifact>) -> f32 {
    query_vectors
        .into_iter()
        .flat_map(|query| {
            record_vectors
                .iter()
                .map(move |record| cosine_similarity(query, &record.vector))
        })
        .fold(0.0f32, f32::max)
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.is_empty() || right.is_empty() || left.len() != right.len() {
        return 0.0;
    }
    let (mut dot, mut left_norm, mut right_norm) = (0.0f32, 0.0f32, 0.0f32);
    for (left, right) in left.iter().zip(right) {
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm <= f32::EPSILON || right_norm <= f32::EPSILON {
        return 0.0;
    }
    (dot / (left_norm.sqrt() * right_norm.sqrt())).clamp(0.0, 1.0)
}

fn tokenize(text: &str) -> BTreeSet<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect()
}

fn token_overlap(left: &BTreeSet<String>, right: &BTreeSet<String>) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let shared = left.intersection(right).count() as f32;
    let total = left.union(right).count() as f32;
    if total <= f32::EPSILON {
        0.0
    } else {
        shared / total
    }
}

fn query_pose_time_hint(query: &RecallQuery, ordinal: u64) -> u64 {
    let pose_hint = query
        .pose
        .map(|pose| ((pose.x_m.abs() + pose.y_m.abs()) * 100.0) as u64)
        .unwrap_or(0);
    pose_hint.saturating_add(ordinal)
}

fn stable_qdrant_point_id(collection: &str, point_id: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    collection.hash(&mut hasher);
    point_id.hash(&mut hasher);
    hasher.finish()
}

fn neo4j_http_url_from_uri(uri: &str) -> Option<String> {
    let trimmed = uri.trim();
    let rest = trimmed
        .strip_prefix("bolt://")
        .or_else(|| trimmed.strip_prefix("neo4j://"))?;
    let host = rest.split('/').next().unwrap_or(rest);
    let host_without_port = host.split(':').next().unwrap_or(host);
    if host_without_port.is_empty() {
        return None;
    }
    Some(format!("http://{host_without_port}:7474"))
}

