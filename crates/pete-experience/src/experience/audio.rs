#[derive(Clone, Debug, Default)]
pub struct AudioDescendantExtractor;

impl AudioDescendantExtractor {
    fn extract_audio(&self, sensation: &Sensation) -> Vec<Sensation> {
        let Some(window) = AudioWindow::from_sensation(sensation) else {
            return Vec::new();
        };
        let mut descendants = Vec::new();
        let transcript = window
            .transcript
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty());

        descendants.push(audio_voice_segment(
            sensation,
            &window,
            0,
            window.duration_ms,
            "asr_or_vad_window",
        ));
        if let Some(text) = transcript {
            descendants.push(audio_speech_segment(sensation, &window, text));
            descendants.push(audio_transcript_span(sensation, &window, text));
        } else if !window.has_asr_timing {
            descendants = fallback_audio_voice_segments(sensation, &window);
        }
        if let Some(text) = window.possible_transcript.as_deref() {
            descendants.push(audio_possible_speech(sensation, &window, text));
        }
        if let Some(text) = window.committed_transcript.as_deref().or_else(|| {
            window
                .is_final
                .then_some(window.transcript.as_deref())
                .flatten()
        }) {
            descendants.push(audio_committed_speech(sensation, &window, text));
        }
        descendants
    }
}

impl DescendantExtractor for AudioDescendantExtractor {
    fn extract(&self, sensation: &Sensation) -> Result<Vec<Sensation>> {
        if sensation.modality == Modality::Audio
            && sensation.payload_kind == SensationPayloadKind::AudioPcm
        {
            Ok(self.extract_audio(sensation))
        } else {
            Ok(Vec::new())
        }
    }
}

#[derive(Clone, Debug)]
struct AudioWindow {
    start_ms: TimeMs,
    end_ms: TimeMs,
    duration_ms: TimeMs,
    confidence: f32,
    transcript: Option<String>,
    is_final: bool,
    word_count: Option<u64>,
    speaker_confidence: Option<f32>,
    sample_rate_hz: Option<u64>,
    feature_sets: u64,
    has_asr_timing: bool,
    possible_transcript: Option<String>,
    committed_transcript: Option<String>,
    candidate_id: Option<u64>,
    stable_text: Option<String>,
    unstable_text: Option<String>,
    stable_word_prefix: Option<String>,
    stable_word_count: Option<u64>,
}

impl AudioWindow {
    fn from_sensation(sensation: &Sensation) -> Option<Self> {
        let asr = sensation.payload.get("asr").unwrap_or(&Value::Null);
        let feature_sets = sensation
            .payload
            .get("feature_sets")
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let transcript = sensation
            .payload
            .get("transcript")
            .and_then(Value::as_str)
            .or_else(|| asr.get("transcript").and_then(Value::as_str))
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(ToOwned::to_owned);
        let asr_start = asr.get("start_ms").and_then(Value::as_u64);
        let asr_end = asr.get("end_ms").and_then(Value::as_u64);
        let duration = sensation
            .metadata
            .duration_ms
            .or_else(|| asr.get("duration_ms").and_then(Value::as_u64))
            .or_else(|| Some(asr_end?.saturating_sub(asr_start?)))
            .or_else(|| (feature_sets > 0).then_some(feature_sets.saturating_mul(20)))
            .or_else(|| {
                (sensation.observed_at_ms > sensation.occurred_at_ms)
                    .then_some(sensation.observed_at_ms - sensation.occurred_at_ms)
            })
            .unwrap_or_default();
        if duration == 0 && transcript.is_none() {
            return None;
        }
        let end_ms = asr_end.unwrap_or(sensation.observed_at_ms.max(sensation.occurred_at_ms));
        let start_ms = asr_start.unwrap_or_else(|| end_ms.saturating_sub(duration));
        let duration_ms = duration.max(end_ms.saturating_sub(start_ms)).max(1);
        Some(Self {
            start_ms,
            end_ms: start_ms.saturating_add(duration_ms),
            duration_ms,
            confidence: sensation
                .metadata
                .confidence
                .or_else(|| {
                    asr.get("confidence")
                        .and_then(Value::as_f64)
                        .map(|value| value as f32)
                })
                .unwrap_or(0.45)
                .clamp(0.0, 1.0),
            transcript,
            is_final: asr
                .get("is_final")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            word_count: asr.get("word_count").and_then(Value::as_u64),
            speaker_confidence: asr
                .get("speaker_confidence")
                .and_then(Value::as_f64)
                .map(|value| value as f32),
            sample_rate_hz: asr.get("sample_rate_hz").and_then(Value::as_u64),
            feature_sets,
            has_asr_timing: asr_start.is_some() || asr_end.is_some(),
            possible_transcript: asr
                .get("possible_transcript")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .map(ToOwned::to_owned),
            committed_transcript: asr
                .get("committed_transcript")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .map(ToOwned::to_owned),
            candidate_id: asr.get("candidate_id").and_then(Value::as_u64),
            stable_text: asr
                .get("stable_text")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            unstable_text: asr
                .get("unstable_text")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            stable_word_prefix: asr
                .get("stable_word_prefix")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            stable_word_count: asr.get("stable_word_count").and_then(Value::as_u64),
        })
    }
}

fn fallback_audio_voice_segments(parent: &Sensation, window: &AudioWindow) -> Vec<Sensation> {
    let segment_count = if window.duration_ms >= 2_400 {
        3
    } else if window.duration_ms >= 1_200 {
        2
    } else {
        1
    };
    let segment_duration = (window.duration_ms / segment_count).max(1);
    (0..segment_count)
        .map(|index| {
            let start_offset = segment_duration.saturating_mul(index);
            let end_offset = if index + 1 == segment_count {
                window.duration_ms
            } else {
                segment_duration.saturating_mul(index + 1)
            };
            audio_voice_segment(
                parent,
                window,
                start_offset,
                end_offset,
                "deterministic_audio_features",
            )
        })
        .collect()
}

fn audio_voice_segment(
    parent: &Sensation,
    window: &AudioWindow,
    start_offset_ms: TimeMs,
    end_offset_ms: TimeMs,
    method: &str,
) -> Sensation {
    let start_ms = window.start_ms.saturating_add(start_offset_ms);
    let end_ms = window
        .start_ms
        .saturating_add(end_offset_ms.max(start_offset_ms + 1));
    let mut metadata = parent.metadata.clone();
    metadata.duration_ms = Some(end_ms.saturating_sub(start_ms));
    metadata.confidence = Some(if window.transcript.is_some() {
        window.confidence.max(0.55)
    } else {
        window.confidence.min(0.55).max(0.25)
    });
    push_label(&mut metadata, "voice-like audio");
    if window.transcript.is_some() {
        push_label(&mut metadata, "asr voice activity");
    } else {
        push_label(&mut metadata, "fallback voice activity");
    }
    metadata
        .properties
        .insert("start_ms".to_string(), json!(start_ms));
    metadata
        .properties
        .insert("end_ms".to_string(), json!(end_ms));
    metadata
        .properties
        .insert("method".to_string(), json!(method));
    if let Some(sample_rate_hz) = window.sample_rate_hz {
        metadata
            .properties
            .insert("sample_rate_hz".to_string(), json!(sample_rate_hz));
    }
    let mut sensation = Sensation::descendant(
        parent,
        "audio.voice_segment",
        SensationPayloadKind::VoiceSegment,
        json!({
            "parent_audio": parent.id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_offset_ms": start_offset_ms,
            "end_offset_ms": end_offset_ms,
            "duration_ms": end_ms.saturating_sub(start_ms),
            "confidence": metadata.confidence,
            "feature_sets": window.feature_sets,
            "method": method,
        }),
        metadata,
        "descendant.audio_voice_activity",
    )
    .with_summary("I hear a voice nearby.");
    sensation.occurred_at_ms = start_ms;
    sensation
}

fn audio_speech_segment(parent: &Sensation, window: &AudioWindow, transcript: &str) -> Sensation {
    let mut metadata = parent.metadata.clone();
    metadata.duration_ms = Some(window.duration_ms);
    metadata.confidence = Some(window.confidence.max(0.35));
    push_label(&mut metadata, "speech");
    push_label(&mut metadata, "asr speech span");
    metadata
        .properties
        .insert("start_ms".to_string(), json!(window.start_ms));
    metadata
        .properties
        .insert("end_ms".to_string(), json!(window.end_ms));
    metadata
        .properties
        .insert("is_final".to_string(), json!(window.is_final));
    let mut sensation = Sensation::descendant(
        parent,
        "audio.speech_segment",
        SensationPayloadKind::SpeechSegment,
        json!({
            "parent_audio": parent.id,
            "start_ms": window.start_ms,
            "end_ms": window.end_ms,
            "duration_ms": window.duration_ms,
            "text": transcript,
            "is_final": window.is_final,
            "confidence": window.confidence,
            "word_count": window.word_count,
            "speaker_confidence": window.speaker_confidence,
            "method": "asr_timed_speech_span",
        }),
        metadata,
        "descendant.audio_speech_span",
    )
    .with_summary(format!("I hear someone say \"{transcript}\"."));
    sensation.occurred_at_ms = window.start_ms;
    sensation
}

fn audio_transcript_span(parent: &Sensation, window: &AudioWindow, transcript: &str) -> Sensation {
    let mut metadata = parent.metadata.clone();
    metadata.duration_ms = Some(window.duration_ms);
    metadata.confidence = Some(window.confidence.max(0.35));
    push_label(&mut metadata, "transcript");
    push_label(&mut metadata, "asr transcript span");
    metadata
        .properties
        .insert("start_ms".to_string(), json!(window.start_ms));
    metadata
        .properties
        .insert("end_ms".to_string(), json!(window.end_ms));
    let mut sensation = Sensation::descendant(
        parent,
        "audio.transcript_span",
        SensationPayloadKind::TranscriptSpan,
        json!({
            "parent_audio": parent.id,
            "start_ms": window.start_ms,
            "end_ms": window.end_ms,
            "duration_ms": window.duration_ms,
            "text": transcript,
            "is_final": window.is_final,
            "confidence": window.confidence,
            "word_count": window.word_count,
            "method": "asr_transcript_span",
        }),
        metadata,
        "descendant.audio_transcript_span",
    )
    .with_summary(format!("I hear someone say \"{transcript}\"."));
    sensation.occurred_at_ms = window.start_ms;
    sensation
}

fn audio_possible_speech(parent: &Sensation, window: &AudioWindow, transcript: &str) -> Sensation {
    let mut metadata = parent.metadata.clone();
    metadata.duration_ms = Some(window.duration_ms);
    metadata.confidence = Some(window.confidence.max(0.25));
    push_label(&mut metadata, "speech");
    push_label(&mut metadata, "possible speech");
    metadata
        .properties
        .insert("start_ms".to_string(), json!(window.start_ms));
    metadata
        .properties
        .insert("end_ms".to_string(), json!(window.end_ms));
    metadata
        .properties
        .insert("commitment".to_string(), json!("possible"));
    let mut sensation = Sensation::descendant(
        parent,
        "audio.possible_speech",
        SensationPayloadKind::SpeechSegment,
        json!({
            "parent_audio": parent.id,
            "start_ms": window.start_ms,
            "end_ms": window.end_ms,
            "duration_ms": window.duration_ms,
            "text": transcript,
            "commitment": "possible",
            "is_final": false,
            "confidence": window.confidence,
            "candidate_id": window.candidate_id,
            "stable_text": window.stable_text,
            "unstable_text": window.unstable_text,
            "stable_word_prefix": window.stable_word_prefix,
            "stable_word_count": window.stable_word_count,
            "method": "asr_tool_transcript_candidate",
        }),
        metadata,
        "descendant.audio_possible_speech",
    )
    .with_summary(format!("I may be hearing someone say \"{transcript}\"."));
    sensation.occurred_at_ms = window.start_ms;
    sensation
}

fn audio_committed_speech(parent: &Sensation, window: &AudioWindow, transcript: &str) -> Sensation {
    let mut metadata = parent.metadata.clone();
    metadata.duration_ms = Some(window.duration_ms);
    metadata.confidence = Some(window.confidence.max(0.35));
    push_label(&mut metadata, "speech");
    push_label(&mut metadata, "committed speech");
    metadata
        .properties
        .insert("start_ms".to_string(), json!(window.start_ms));
    metadata
        .properties
        .insert("end_ms".to_string(), json!(window.end_ms));
    metadata
        .properties
        .insert("commitment".to_string(), json!("committed"));
    let mut sensation = Sensation::descendant(
        parent,
        "audio.committed_speech",
        SensationPayloadKind::TranscriptSpan,
        json!({
            "parent_audio": parent.id,
            "start_ms": window.start_ms,
            "end_ms": window.end_ms,
            "duration_ms": window.duration_ms,
            "text": transcript,
            "commitment": "committed",
            "is_final": true,
            "confidence": window.confidence,
            "candidate_id": window.candidate_id,
            "stable_text": window.stable_text,
            "stable_word_prefix": window.stable_word_prefix,
            "stable_word_count": window.stable_word_count,
            "method": "asr_tool_transcript_commit",
        }),
        metadata,
        "descendant.audio_committed_speech",
    )
    .with_summary(format!(
        "I commit that I heard someone say \"{transcript}\"."
    ));
    sensation.occurred_at_ms = window.start_ms;
    sensation
}

fn push_label(metadata: &mut SensationMetadata, label: &str) {
    if !metadata.labels.iter().any(|existing| existing == label) {
        metadata.labels.push(label.to_string());
    }
}

#[derive(Clone, Debug, Default)]
pub struct DeterministicDescendantExtractor;

impl DescendantExtractor for DeterministicDescendantExtractor {
    fn extract(&self, sensation: &Sensation) -> Result<Vec<Sensation>> {
        match (&sensation.modality, &sensation.payload_kind) {
            (Modality::Vision, SensationPayloadKind::ImageBytes) => {
                VisualDescendantExtractor.extract(sensation)
            }
            (Modality::Audio, SensationPayloadKind::AudioPcm) => {
                AudioDescendantExtractor.extract(sensation)
            }
            _ => Ok(Vec::new()),
        }
    }
}
