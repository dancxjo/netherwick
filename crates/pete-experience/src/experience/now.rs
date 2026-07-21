#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodiedNow {
    pub sensations: Vec<Sensation>,
    pub impressions: Vec<Impression>,
    pub experience: Experience,
}

pub fn primary_sensations_from_now(now: &Now) -> Vec<Sensation> {
    let mut sensations = Vec::new();

    let mut body = Sensation::primary(
        if now.body.flags.bump_left
            || now.body.flags.bump_right
            || now.body.flags.wall
            || now.body.flags.virtual_wall
            || now.body.flags.wheel_drop
        {
            Modality::Touch
        } else {
            Modality::Odometry
        },
        SensationSource::new("body"),
        now.t_ms,
        now.t_ms,
        SensationPayload {
            kind: if now.body.flags.bump_left
                || now.body.flags.bump_right
                || now.body.flags.wall
                || now.body.flags.virtual_wall
                || now.body.flags.wheel_drop
            {
                SensationPayloadKind::ContactEvent
            } else {
                SensationPayloadKind::OdometryEvent
            },
            value: json!({
                "battery_level": now.body.battery_level,
                "charging": now.body.charging,
                "flags": now.body.flags,
                "odometry": now.body.odometry,
                "velocity": now.body.velocity,
                "cliff_sensors": now.body.cliff_sensors,
            }),
        },
    )
    .with_summary("I feel the state and motion of my body.");
    body.metadata.confidence = Some(0.9);
    sensations.push(body);

    if let Some(frame) = &now.eye_frame {
        let mut source = SensationSource::new("eye.frame");
        source.device_id = frame.source.clone();
        source.frame_id = Some(frame.captured_at_ms.to_string());
        let mut sensation =
            Sensation::primary(Modality::Vision, source, frame.captured_at_ms, now.t_ms, {
                let mut payload = SensationPayload::image_metadata(
                    frame.width,
                    frame.height,
                    format!("{:?}", frame.format),
                    frame.bytes.len(),
                );
                if !frame.bytes.is_empty() {
                    payload.value["raw_bytes_b64"] = Value::String(
                        base64::engine::general_purpose::STANDARD.encode(&frame.bytes),
                    );
                }
                payload
            })
            .with_summary("I receive a camera frame.");
        sensation.metadata.confidence = Some(0.65);
        sensation.metadata.properties.insert(
            "raw_bytes_present".to_string(),
            json!(!frame.bytes.is_empty()),
        );
        sensations.push(sensation);
    } else if !now.eye.frames.is_empty()
        || !now.eye.image_vectors.is_empty()
        || !now.eye.scene_vectors.is_empty()
    {
        let mut vector_artifacts = now.eye.image_vectors.clone();
        vector_artifacts.extend(now.eye.scene_vectors.clone());
        vector_artifacts.extend(now.eye.image_description_vectors.clone());
        let mut sensation = Sensation::primary(
            Modality::Vision,
            SensationSource::new("eye.features"),
            now.t_ms,
            now.t_ms,
            SensationPayload::structured(json!({
                "frame_feature_sets": now.eye.frames.len(),
                "image_vectors": now.eye.image_vectors.len(),
                "image_description_vectors": now.eye.image_description_vectors.len(),
                "scene_vectors": now.eye.scene_vectors.len(),
                "vector_artifacts": vector_artifacts,
            })),
        )
        .with_summary("I have visual features from my eye.");
        sensation.metadata.confidence = Some(0.55);
        sensations.push(sensation);
    }

    if !now.face.vectors.is_empty() {
        let vector_artifacts = now.face.vectors.clone();
        let mut sensation = Sensation::primary(
            Modality::Vision,
            SensationSource::new("face.features"),
            now.t_ms,
            now.t_ms,
            SensationPayload {
                kind: SensationPayloadKind::Crop,
                value: json!({
                    "face_vectors": now.face.vectors.len(),
                    "vector_artifacts": vector_artifacts,
                }),
            },
        )
        .with_summary("I have a face embedding from vision.");
        sensation.metadata.confidence = Some(0.6);
        sensation.metadata.labels.push("face".to_string());
        sensations.push(sensation);
    }

    if !now.objects.vectors.is_empty() {
        let vector_artifacts = now.objects.vectors.clone();
        let mut sensation = Sensation::primary(
            Modality::Vision,
            SensationSource::new("object.features"),
            now.t_ms,
            now.t_ms,
            SensationPayload {
                kind: SensationPayloadKind::Crop,
                value: json!({
                    "object_observations": now.objects.observations.len(),
                    "object_vectors": now.objects.vectors.len(),
                    "vector_artifacts": vector_artifacts,
                }),
            },
        )
        .with_summary("I have object visual vectors from vision.");
        sensation.metadata.confidence = Some(0.6);
        sensation.metadata.labels.push("object".to_string());
        sensations.push(sensation);
    }

    if !now.ear.features.is_empty()
        || !now.ear.transcript_vectors.is_empty()
        || now
            .ear
            .transcript
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty())
        || now
            .ear
            .asr
            .transcript
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty())
        || now
            .ear
            .asr
            .possible_transcript
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty())
        || now
            .ear
            .asr
            .committed_transcript
            .as_deref()
            .is_some_and(|text| !text.trim().is_empty())
    {
        let transcript = now
            .ear
            .asr
            .committed_transcript
            .as_deref()
            .or(now.ear.asr.transcript.as_deref())
            .or(now.ear.asr.possible_transcript.as_deref())
            .or(now.ear.transcript.as_deref())
            .map(str::trim)
            .filter(|text| !text.is_empty());
        let legacy_transcript = now
            .ear
            .asr
            .transcript
            .as_deref()
            .or(now.ear.transcript.as_deref())
            .map(str::trim)
            .filter(|text| !text.is_empty());
        let duration_ms = now
            .ear
            .asr
            .duration_ms
            .or_else(|| Some(now.ear.asr.end_ms?.saturating_sub(now.ear.asr.start_ms?)))
            .or_else(|| {
                (!now.ear.features.is_empty()).then_some(now.ear.features.len() as u64 * 20)
            });
        let observed_at_ms = now.ear.asr.end_ms.unwrap_or(now.t_ms);
        let occurred_at_ms = now
            .ear
            .asr
            .start_ms
            .or_else(|| duration_ms.map(|duration| observed_at_ms.saturating_sub(duration)))
            .unwrap_or(now.t_ms);
        let mut sensation = Sensation::primary(
            Modality::Audio,
            SensationSource::new("ear"),
            occurred_at_ms,
            observed_at_ms,
            SensationPayload {
                kind: SensationPayloadKind::AudioPcm,
                value: json!({
                    "feature_sets": now.ear.features.len(),
                    "transcript_vectors": now.ear.transcript_vectors.len(),
                    "transcript": legacy_transcript.or(transcript),
                    "asr": now.ear.asr,
                }),
            },
        )
        .with_summary("I hear sound through my ear.");
        sensation.metadata.duration_ms = duration_ms;
        sensation.metadata.confidence = Some(now.ear.asr.confidence.max(0.35).clamp(0.0, 1.0));
        sensation.metadata.labels.push("audio window".to_string());
        if transcript.is_some() {
            sensation.metadata.labels.push("asr available".to_string());
        }
        sensations.push(sensation);
    }

    if !now.ear.transcript_vectors.is_empty() {
        let transcript = now
            .ear
            .asr
            .committed_transcript
            .as_deref()
            .or(now.ear.asr.transcript.as_deref())
            .or(now.ear.transcript.as_deref())
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .unwrap_or("speech transcript");
        let observed_at_ms = now.ear.asr.end_ms.unwrap_or(now.t_ms);
        let occurred_at_ms = now.ear.asr.start_ms.unwrap_or(observed_at_ms);
        let mut sensation = Sensation::primary(
            Modality::Audio,
            SensationSource::new("ear.transcript_vectors"),
            occurred_at_ms,
            observed_at_ms,
            SensationPayload {
                kind: SensationPayloadKind::TranscriptSpan,
                value: json!({
                    "text": transcript,
                    "transcript_vectors": now.ear.transcript_vectors.len(),
                    "vector_artifacts": now.ear.transcript_vectors.clone(),
                }),
            },
        )
        .with_summary(format!("I have a transcript vector for \"{transcript}\"."));
        sensation.metadata.confidence = Some(now.ear.asr.confidence.max(0.35).clamp(0.0, 1.0));
        sensation
            .metadata
            .labels
            .push("transcript vector".to_string());
        sensations.push(sensation);
    }

    if !now.voice.vectors.is_empty() {
        let vector_artifacts = now.voice.vectors.clone();
        let mut sensation = Sensation::primary(
            Modality::Audio,
            SensationSource::new("voice.features"),
            now.t_ms,
            now.t_ms,
            SensationPayload {
                kind: SensationPayloadKind::VoiceSegment,
                value: json!({
                    "voice_vectors": now.voice.vectors.len(),
                    "vector_artifacts": vector_artifacts,
                }),
            },
        )
        .with_summary("I have a voice embedding from hearing.");
        sensation.metadata.confidence = Some(0.6);
        sensation.metadata.labels.push("voice identity".to_string());
        sensations.push(sensation);
    }

    if !now.range.beams.is_empty() || now.range.nearest_m.is_some() {
        let mut sensation = Sensation::primary(
            Modality::Lidar,
            SensationSource::new("range"),
            now.t_ms,
            now.t_ms,
            SensationPayload {
                kind: SensationPayloadKind::LidarScan,
                value: json!({
                    "beam_count": now.range.beams.len(),
                    "nearest_m": now.range.nearest_m,
                }),
            },
        )
        .with_summary("I sense nearby distance around me.");
        sensation.metadata.confidence = Some(0.7);
        sensations.push(sensation);
    }

    if !now.kinect.depth_m.is_empty() {
        let mut sensation = Sensation::primary(
            Modality::Depth,
            SensationSource::new("kinect.depth"),
            now.t_ms,
            now.t_ms,
            SensationPayload {
                kind: SensationPayloadKind::DepthFrame,
                value: json!({
                    "sample_count": now.kinect.depth_m.len(),
                    "width": now.kinect.depth_width,
                    "height": now.kinect.depth_height,
                    "min_depth_m": now.kinect.min_depth_m,
                    "max_depth_m": now.kinect.max_depth_m,
                    "coordinate_system": now.kinect.depth_coordinate_system,
                    "skeleton_count": now.kinect.skeletons.len(),
                }),
            },
        )
        .with_summary("I sense depth and surfaces ahead of me.");
        sensation.metadata.confidence = Some(0.65);
        sensations.push(sensation);
    }

    if now.memory.similar_situation_count > 0
        || now.memory.remembered_warning.is_some()
        || now.memory.graph_context_summary.is_some()
    {
        let mut sensation = Sensation::primary(
            Modality::Memory,
            SensationSource::new("memory"),
            now.t_ms,
            now.t_ms,
            SensationPayload {
                kind: SensationPayloadKind::MemoryRecall,
                value: json!({
                    "similar_situation_count": now.memory.similar_situation_count,
                    "remembered_warning": now.memory.remembered_warning,
                    "graph_context_summary": now.memory.graph_context_summary,
                    "remembered_entities": now.memory.remembered_entities,
                }),
            },
        )
        .with_summary("I remember related context for this moment.");
        sensation.metadata.confidence = Some(0.6);
        sensations.push(sensation);
    }

    sensations
}

fn embodied_tags(sensations: &[Sensation]) -> Vec<String> {
    let mut tags = sensations
        .iter()
        .map(|sensation| sensation.modality.as_str().to_string())
        .collect::<Vec<_>>();
    tags.sort();
    tags.dedup();
    tags
}

fn stable_unit(text: &str) -> f32 {
    let mut hash = 0_u32;
    for byte in text.as_bytes() {
        hash = hash.wrapping_mul(16777619) ^ u32::from(*byte);
    }
    (hash % 10_000) as f32 / 10_000.0
}
