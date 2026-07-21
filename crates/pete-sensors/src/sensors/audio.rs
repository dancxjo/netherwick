#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PcmAudioFrame {
    pub captured_at_ms: u64,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples: Vec<i16>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AsrToolConfig {
    pub command: Option<String>,
    pub min_voice_rms: f32,
    pub min_chunk_ms: u64,
    pub max_chunk_ms: u64,
    pub silence_finalize_ms: u64,
}

impl Default for AsrToolConfig {
    fn default() -> Self {
        Self {
            command: std::env::var("PETE_ASR_COMMAND")
                .ok()
                .filter(|value| !value.trim().is_empty()),
            min_voice_rms: std::env::var("PETE_ASR_MIN_RMS")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .unwrap_or(0.012),
            min_chunk_ms: std::env::var("PETE_ASR_MIN_CHUNK_MS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(450),
            max_chunk_ms: std::env::var("PETE_ASR_MAX_CHUNK_MS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(8_000),
            silence_finalize_ms: std::env::var("PETE_ASR_SILENCE_FINALIZE_MS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(700),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AsrTool {
    config: AsrToolConfig,
    chunk: Vec<i16>,
    chunk_start_ms: Option<u64>,
    chunk_end_ms: Option<u64>,
    last_voice_ms: Option<u64>,
    sequence: u64,
    transcript_tracker: TranscriptCandidateTracker,
}

impl AsrTool {
    pub fn new(config: AsrToolConfig) -> Self {
        Self {
            config,
            chunk: Vec::new(),
            chunk_start_ms: None,
            chunk_end_ms: None,
            last_voice_ms: None,
            sequence: 0,
            transcript_tracker: TranscriptCandidateTracker::new(),
        }
    }

    pub fn observe_frame(&mut self, frame: &PcmAudioFrame) -> Option<EarSense> {
        if frame.samples.is_empty() || frame.sample_rate_hz == 0 || frame.channels == 0 {
            return self.try_finalize(frame.captured_at_ms, frame.sample_rate_hz, frame.channels);
        }
        let rms = pcm_rms(&frame.samples);
        let voice = rms >= self.config.min_voice_rms;
        let duration_ms =
            pcm_duration_ms(frame.samples.len(), frame.sample_rate_hz, frame.channels);
        let frame_start = frame.captured_at_ms;
        let frame_end = frame_start.saturating_add(duration_ms);

        if voice {
            if self.chunk_start_ms.is_none() {
                self.chunk_start_ms = Some(frame_start);
            }
            self.chunk.extend_from_slice(&frame.samples);
            self.chunk_end_ms = Some(frame_end);
            self.last_voice_ms = Some(frame_end);
        } else if self.chunk_start_ms.is_some() {
            self.chunk_end_ms = Some(frame_end);
        }

        let chunk_duration = self
            .chunk_start_ms
            .zip(self.chunk_end_ms)
            .map(|(start, end)| end.saturating_sub(start))
            .unwrap_or_default();
        let silence_ms = self
            .last_voice_ms
            .map(|last| frame_end.saturating_sub(last))
            .unwrap_or_default();
        let should_finalize = chunk_duration >= self.config.max_chunk_ms
            || (chunk_duration >= self.config.min_chunk_ms
                && silence_ms >= self.config.silence_finalize_ms);
        if should_finalize {
            self.try_finalize(frame_end, frame.sample_rate_hz, frame.channels)
        } else {
            None
        }
    }

    fn try_finalize(
        &mut self,
        fallback_end_ms: u64,
        sample_rate_hz: u32,
        channels: u16,
    ) -> Option<EarSense> {
        let start_ms = self.chunk_start_ms?;
        let end_ms = self.chunk_end_ms.unwrap_or(fallback_end_ms);
        if end_ms.saturating_sub(start_ms) < self.config.min_chunk_ms || self.chunk.is_empty() {
            self.clear_chunk();
            return None;
        }
        let transcript = self
            .config
            .command
            .as_deref()
            .and_then(|command| {
                transcribe_with_command(command, &self.chunk, sample_rate_hz, channels).ok()
            })
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty())?;

        let sequence = self.sequence;
        self.sequence = self.sequence.saturating_add(1);
        let word_count = transcript.split_whitespace().count().min(u16::MAX as usize) as u16;
        let confidence = command_transcript_confidence(&transcript);
        let candidate_events =
            self.transcript_tracker
                .ingest_candidate(transcript.clone(), Some(confidence), true);
        let stability = transcript_stability_from_events(&candidate_events);
        let transcript_vector = transcript_vector_artifact(&transcript, sequence, start_ms, end_ms);
        self.clear_chunk();
        Some(EarSense {
            schema_version: 1,
            features: Vec::new(),
            transcript: Some(transcript.clone()),
            transcript_vectors: vec![transcript_vector],
            asr: AsrSense {
                transcript: Some(transcript.clone()),
                possible_transcript: None,
                committed_transcript: Some(transcript),
                is_final: true,
                confidence,
                sequence_start: Some(sequence),
                sequence_end: Some(sequence),
                candidate_id: stability.as_ref().map(|state| state.candidate_id.0),
                stable_text: stability.as_ref().map(|state| state.stable_text.clone()),
                unstable_text: stability.as_ref().map(|state| state.unstable_text.clone()),
                stable_word_prefix: stability
                    .as_ref()
                    .and_then(|state| state.stable_word_prefix.clone()),
                stable_word_count: stability
                    .as_ref()
                    .map(|state| state.stable_word_count.min(u16::MAX as usize) as u16),
                start_ms: Some(start_ms),
                end_ms: Some(end_ms),
                duration_ms: Some(end_ms.saturating_sub(start_ms)),
                sample_rate_hz: Some(sample_rate_hz),
                word_count: Some(word_count),
                speaker_confidence: None,
                candidate_events,
            },
        })
    }

    fn clear_chunk(&mut self) {
        self.chunk.clear();
        self.chunk_start_ms = None;
        self.chunk_end_ms = None;
        self.last_voice_ms = None;
    }
}

fn transcript_stability_from_events(
    events: &[TranscriptCandidateEvent],
) -> Option<TranscriptStabilityState> {
    events.iter().rev().find_map(|event| match event {
        TranscriptCandidateEvent::CandidateUpdated {
            id,
            text,
            stable_prefix_len,
            confidence,
        } => Some(TranscriptStabilityState::from_parts(
            *id,
            text,
            *stable_prefix_len,
            *confidence,
        )),
        TranscriptCandidateEvent::CandidateFinalized {
            id,
            text,
            confidence,
        } => Some(TranscriptStabilityState::from_parts(
            *id,
            text,
            text.len(),
            *confidence,
        )),
        TranscriptCandidateEvent::CandidateStarted { .. }
        | TranscriptCandidateEvent::CandidateReplaced { .. }
        | TranscriptCandidateEvent::CandidateCancelled { .. } => None,
    })
}
