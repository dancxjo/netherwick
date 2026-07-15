use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample};
use serde::{Deserialize, Serialize};
use speaking::{
    phonemicizer_for_variety, EvidenceProvenance, EvidenceSource, PhonemicizeOutput,
    PhonemicizeRequest, UtteranceId, UtterancePlan, VarietyId,
};
use tongues_tts::{PiperAudioChunk, PiperOnnxBackend, PiperVoice, PiperVoiceConfig};

const DEFAULT_TTS_VARIETY: &str = "en-US";

pub trait Mouth: Send {
    fn speak(&mut self, text: &str) -> Result<SpeechOutcome>;
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SpeechOutcome {
    pub spoken: bool,
    pub backend: String,
    pub text_len: usize,
    pub sample_rate_hz: Option<u32>,
    pub channels: Option<u16>,
    pub sample_count: usize,
    pub duration_ms: Option<u64>,
    pub device: Option<String>,
}

#[derive(Default)]
pub struct NoopMouth;

impl Mouth for NoopMouth {
    fn speak(&mut self, text: &str) -> Result<SpeechOutcome> {
        Ok(SpeechOutcome {
            spoken: false,
            backend: "noop".to_string(),
            text_len: text.trim().len(),
            ..SpeechOutcome::default()
        })
    }
}

pub fn mouth_from_env() -> Box<dyn Mouth + Send> {
    match PiperCpalMouth::from_env() {
        Ok(Some(mouth)) => Box::new(mouth),
        Ok(None) => Box::<NoopMouth>::default(),
        Err(error) => {
            tracing::warn!(error = %error, "failed to configure speech mouth; using noop mouth");
            Box::<NoopMouth>::default()
        }
    }
}

pub struct QueuedPiperCpalMouth {
    tx: Option<mpsc::Sender<MouthQueueItem>>,
    worker: Option<JoinHandle<()>>,
}

struct MouthQueueItem {
    text: String,
    outcome_tx: Option<mpsc::Sender<std::result::Result<SpeechOutcome, String>>>,
}

impl QueuedPiperCpalMouth {
    pub fn from_env() -> Result<Option<Self>> {
        PiperConfig::from_env()?.map(Self::new).transpose()
    }

    pub fn new(config: PiperConfig) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<MouthQueueItem>();
        let worker = std::thread::Builder::new()
            .name("pete-piper-mouth".to_string())
            .spawn(move || {
                println!(
                    "robot mouth loading Piper voice: {}",
                    config.model_path.display()
                );
                let mut mouth = match PiperCpalMouth::new(config) {
                    Ok(mouth) => mouth,
                    Err(error) => {
                        let message =
                            format!("queued Piper mouth failed to load voice: {error:#}");
                        println!("robot mouth failed: {message}");
                        tracing::warn!(error = %error, "queued Piper mouth failed to load voice");
                        for item in rx {
                            if let Some(outcome_tx) = item.outcome_tx {
                                let _ = outcome_tx.send(Err(message.clone()));
                            }
                        }
                        return;
                    }
                };
                println!("robot mouth Piper voice ready");
                while let Ok(item) = rx.recv() {
                    match mouth.speak(&item.text) {
                        Ok(outcome) => {
                            println!(
                                "robot mouth spoke: device {}, duration {} ms",
                                outcome.device.as_deref().unwrap_or("<unknown>"),
                                outcome.duration_ms.unwrap_or_default()
                            );
                            if let Some(outcome_tx) = item.outcome_tx {
                                let _ = outcome_tx.send(Ok(outcome));
                            }
                        }
                        Err(error) => {
                            let message = error.to_string();
                            println!(
                                "robot mouth failed: {message}; disabling mouth worker; text {:?}",
                                item.text
                            );
                            tracing::warn!(error = %message, text = %item.text, "queued Piper mouth failed");
                            if let Some(outcome_tx) = item.outcome_tx {
                                let _ = outcome_tx.send(Err(message.clone()));
                            }
                            for pending in rx.try_iter() {
                                if let Some(outcome_tx) = pending.outcome_tx {
                                    let _ = outcome_tx.send(Err(message.clone()));
                                }
                            }
                            break;
                        }
                    }
                }
            })
            .context("failed to spawn queued Piper mouth thread")?;
        Ok(Self {
            tx: Some(tx),
            worker: Some(worker),
        })
    }

    pub fn enqueue(&self, text: impl Into<String>) -> Result<()> {
        let text = text.into();
        if text.trim().is_empty() {
            return Ok(());
        }
        self.send_item(MouthQueueItem {
            text,
            outcome_tx: None,
        })
    }

    pub fn enqueue_and_wait(&self, text: impl Into<String>) -> Result<SpeechOutcome> {
        self.enqueue_and_wait_timeout(text, None)
    }

    pub fn enqueue_and_wait_timeout(
        &self,
        text: impl Into<String>,
        timeout: Option<Duration>,
    ) -> Result<SpeechOutcome> {
        let text = text.into();
        if text.trim().is_empty() {
            return Ok(SpeechOutcome {
                spoken: false,
                backend: "queued-piper-cpal".to_string(),
                ..SpeechOutcome::default()
            });
        }
        let (outcome_tx, outcome_rx) = mpsc::channel();
        self.send_item(MouthQueueItem {
            text,
            outcome_tx: Some(outcome_tx),
        })?;
        let result = match timeout {
            Some(timeout) => outcome_rx
                .recv_timeout(timeout)
                .with_context(|| format!("queued Piper mouth did not finish within {timeout:?}"))?,
            None => outcome_rx
                .recv()
                .context("queued Piper mouth worker did not report outcome")?,
        };
        match result {
            Ok(outcome) => Ok(outcome),
            Err(error) => anyhow::bail!(error),
        }
    }

    fn send_item(&self, item: MouthQueueItem) -> Result<()> {
        self.tx
            .as_ref()
            .context("queued Piper mouth is already closed")?
            .send(item)
            .context("queued Piper mouth worker is not running")
    }
}

impl Drop for QueuedPiperCpalMouth {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Some(worker) = self.worker.take() {
            if !worker.is_finished() {
                println!("robot mouth worker still running at shutdown; waiting for it to stop");
            }
            let _ = worker.join();
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiperConfig {
    pub model_path: PathBuf,
    pub config_path: PathBuf,
    pub variety: String,
    pub output_device_name: Option<String>,
}

impl PiperConfig {
    pub fn new(model_path: impl Into<PathBuf>, config_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            config_path: config_path.into(),
            variety: DEFAULT_TTS_VARIETY.to_string(),
            output_device_name: None,
        }
    }

    pub fn from_env() -> Result<Option<Self>> {
        let (model_path, config_path) = match env_path("PETE_TTS_PIPER_VOICE") {
            Some(model_path) => {
                let config_path = env_path("PETE_TTS_PIPER_CONFIG")
                    .unwrap_or_else(|| tongues_tts::piper_voice_config_path(&model_path));
                (model_path, config_path)
            }
            None => {
                let model_path = tongues_tts::default_voice_model_path(PiperVoice::RyanMedium);
                let config_path = tongues_tts::default_voice_config_path(PiperVoice::RyanMedium);
                if !model_path.is_file() || !config_path.is_file() {
                    return Ok(None);
                }
                (model_path, config_path)
            }
        };
        let mut config = Self::new(model_path, config_path);
        config.variety = std::env::var("PETE_TTS_VARIETY")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_TTS_VARIETY.to_string());
        config.output_device_name = std::env::var("PETE_TTS_OUTPUT_DEVICE")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        Ok(Some(config))
    }
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

pub struct PiperCpalMouth {
    config: PiperConfig,
    speech: PiperOnnxBackend,
}

impl PiperCpalMouth {
    pub fn new(config: PiperConfig) -> Result<Self> {
        let voice_config = PiperVoiceConfig::from_json_file(&config.config_path)
            .with_context(|| format!("failed to read {}", config.config_path.display()))?;
        let speech = PiperOnnxBackend::load(&config.model_path, voice_config)
            .context("failed to load Piper ONNX speech backend")?;
        Ok(Self { config, speech })
    }

    pub fn from_env() -> Result<Option<Self>> {
        PiperConfig::from_env()?.map(Self::new).transpose()
    }
}

impl Mouth for PiperCpalMouth {
    fn speak(&mut self, text: &str) -> Result<SpeechOutcome> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(SpeechOutcome {
                spoken: false,
                backend: "piper-cpal".to_string(),
                ..SpeechOutcome::default()
            });
        }

        let plan = utterance_plan_from_text(text, &self.config.variety)?;
        play_tongues_streaming(
            &mut self.speech,
            &plan,
            text.len(),
            self.config.output_device_name.as_deref(),
        )
    }
}

fn play_tongues_streaming(
    speech: &mut PiperOnnxBackend,
    plan: &UtterancePlan,
    text_len: usize,
    output_device_name: Option<&str>,
) -> Result<SpeechOutcome> {
    let host = cpal::default_host();
    let device = select_output_device(&host, output_device_name)?;
    let device_name = device
        .name()
        .unwrap_or_else(|_| "<unknown output device>".to_string());
    println!("robot mouth using output device: {device_name}");
    let output_config = output_config(&device)?;
    let buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
    let cursor = Arc::new(AtomicUsize::new(0));
    let finished = Arc::new(AtomicBool::new(false));
    let stream = build_streaming_output_stream(
        &device,
        &output_config,
        Arc::clone(&buffer),
        Arc::clone(&cursor),
        Arc::clone(&finished),
    )?;
    stream
        .play()
        .with_context(|| format!("failed to start speech playback on {device_name}"))?;

    let source_sample_rate_hz = speech.sample_rate_hz();
    let source_channels = 1u16;
    let mut queued_samples = 0usize;
    println!("robot mouth synthesizing speech");
    speech
        .synthesize_plan_streaming(plan, &mut |audio: PiperAudioChunk| {
            anyhow::ensure!(
                audio.sample_rate_hz > 0,
                "speech sample rate must be positive"
            );
            let converted = convert_interleaved_f32(
                &audio.pcm_mono_f32,
                audio.sample_rate_hz,
                1,
                output_config.sample_rate_hz,
                output_config.channels,
            );
            queued_samples += converted.len();
            buffer
                .lock()
                .expect("speech output buffer poisoned")
                .extend(converted);
            Ok(())
        })
        .context("Tongues Piper ONNX streaming synthesis failed")?;

    anyhow::ensure!(queued_samples > 0, "speech synthesis produced no audio");
    println!("robot mouth draining {queued_samples} output samples");
    finished.store(true, Ordering::Release);
    while cursor.load(Ordering::Acquire) < queued_samples {
        std::thread::sleep(Duration::from_millis(10));
    }
    std::thread::sleep(Duration::from_millis(20));
    drop(stream);

    let duration = playback_duration(
        queued_samples,
        output_config.sample_rate_hz,
        output_config.channels,
    );
    Ok(SpeechOutcome {
        spoken: true,
        backend: "tongues-piper-onnx-cpal".to_string(),
        text_len,
        sample_rate_hz: Some(source_sample_rate_hz),
        channels: Some(source_channels),
        sample_count: queued_samples,
        duration_ms: Some(duration.as_millis() as u64),
        device: Some(device_name),
    })
}

fn utterance_plan_from_text(text: &str, variety: &str) -> Result<UtterancePlan> {
    let variety = VarietyId(variety.to_string());
    let phonemicizer = phonemicizer_for_variety(&variety)
        .map_err(|error| anyhow::anyhow!("failed to load phonemicizer: {error}"))?;
    let phonemicized = phonemicizer
        .phonemicize(&PhonemicizeRequest {
            text: text.to_string(),
            variety,
            style: None,
        })
        .context("failed to phonemicize text into a Piper speech plan")?;
    Ok(utterance_plan_from_phonemicized(&phonemicized))
}

fn utterance_plan_from_phonemicized(output: &PhonemicizeOutput) -> UtterancePlan {
    UtterancePlan {
        id: UtteranceId("pete.mouth.utterance".into()),
        variety: output.variety.clone(),
        speaker: None,
        intended_text: Some(output.text.clone()),
        intended_morphemes: Vec::new(),
        intended_phonemes: output.phonemes.clone(),
        target_phones: output.phones.clone(),
        target_syllables: output.syllables.clone(),
        boundaries: output.boundaries.clone(),
        target_prosody: output.prosody.clone(),
        target_acoustics: Vec::new(),
        style: None,
        provenance: EvidenceProvenance {
            source: EvidenceSource::TtsPlan,
            method: "pete mouth phonemicized Piper plan".into(),
            version: Some("0.1".into()),
        },
    }
}

fn select_output_device(host: &cpal::Host, requested_name: Option<&str>) -> Result<cpal::Device> {
    let Some(requested_name) = requested_name else {
        return host
            .default_output_device()
            .ok_or_else(|| anyhow::anyhow!("no default output device available"));
    };
    let requested_name = requested_name.to_ascii_lowercase();
    let devices = host
        .output_devices()
        .context("failed to enumerate output devices")?;
    let mut available = Vec::new();
    for device in devices {
        let name = device
            .name()
            .unwrap_or_else(|_| "<unknown output device>".to_string());
        if name.to_ascii_lowercase().contains(&requested_name) {
            return Ok(device);
        }
        available.push(name);
    }
    anyhow::bail!(
        "requested speech output device {:?} not found; available output devices: {}",
        requested_name,
        available.join(", ")
    );
}

struct OutputConfig {
    sample_format: cpal::SampleFormat,
    sample_rate_hz: u32,
    channels: u16,
    stream_config: cpal::StreamConfig,
}

fn output_config(device: &cpal::Device) -> Result<OutputConfig> {
    let config = device
        .default_output_config()
        .context("failed to read default output config")?;
    Ok(OutputConfig {
        sample_format: config.sample_format(),
        sample_rate_hz: config.sample_rate().0,
        channels: config.channels(),
        stream_config: config.config(),
    })
}

fn build_streaming_output_stream(
    device: &cpal::Device,
    config: &OutputConfig,
    samples: Arc<Mutex<Vec<f32>>>,
    cursor: Arc<AtomicUsize>,
    finished: Arc<AtomicBool>,
) -> Result<cpal::Stream> {
    let err_fn = |err| tracing::warn!(error = %err, "speech output stream error");
    match config.sample_format {
        cpal::SampleFormat::F32 => build_typed_streaming_output_stream::<f32>(
            device,
            &config.stream_config,
            samples,
            cursor,
            finished,
            err_fn,
        ),
        cpal::SampleFormat::F64 => build_typed_streaming_output_stream::<f64>(
            device,
            &config.stream_config,
            samples,
            cursor,
            finished,
            err_fn,
        ),
        cpal::SampleFormat::I8 => build_typed_streaming_output_stream::<i8>(
            device,
            &config.stream_config,
            samples,
            cursor,
            finished,
            err_fn,
        ),
        cpal::SampleFormat::I16 => build_typed_streaming_output_stream::<i16>(
            device,
            &config.stream_config,
            samples,
            cursor,
            finished,
            err_fn,
        ),
        cpal::SampleFormat::I32 => build_typed_streaming_output_stream::<i32>(
            device,
            &config.stream_config,
            samples,
            cursor,
            finished,
            err_fn,
        ),
        cpal::SampleFormat::I64 => build_typed_streaming_output_stream::<i64>(
            device,
            &config.stream_config,
            samples,
            cursor,
            finished,
            err_fn,
        ),
        cpal::SampleFormat::U8 => build_typed_streaming_output_stream::<u8>(
            device,
            &config.stream_config,
            samples,
            cursor,
            finished,
            err_fn,
        ),
        cpal::SampleFormat::U16 => build_typed_streaming_output_stream::<u16>(
            device,
            &config.stream_config,
            samples,
            cursor,
            finished,
            err_fn,
        ),
        cpal::SampleFormat::U32 => build_typed_streaming_output_stream::<u32>(
            device,
            &config.stream_config,
            samples,
            cursor,
            finished,
            err_fn,
        ),
        cpal::SampleFormat::U64 => build_typed_streaming_output_stream::<u64>(
            device,
            &config.stream_config,
            samples,
            cursor,
            finished,
            err_fn,
        ),
        sample_format => anyhow::bail!("unsupported output sample format: {sample_format:?}"),
    }
}

fn build_typed_streaming_output_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    samples: Arc<Mutex<Vec<f32>>>,
    cursor: Arc<AtomicUsize>,
    finished: Arc<AtomicBool>,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream>
where
    T: Sample + SizedSample + FromSample<f32>,
{
    device
        .build_output_stream(
            config,
            move |output: &mut [T], _| {
                for out in output.iter_mut() {
                    let idx = cursor.load(Ordering::Relaxed);
                    let sample = samples
                        .lock()
                        .expect("speech output buffer poisoned")
                        .get(idx)
                        .copied();
                    if let Some(sample) = sample {
                        cursor.store(idx + 1, Ordering::Relaxed);
                        *out = T::from_sample(sample);
                    } else {
                        let _done = finished.load(Ordering::Relaxed);
                        *out = T::from_sample(0.0);
                    }
                }
            },
            err_fn,
            None,
        )
        .context("failed to build streaming speech output stream")
}

fn convert_interleaved_f32(
    samples: &[f32],
    source_sample_rate_hz: u32,
    source_channels: u16,
    target_sample_rate_hz: u32,
    target_channels: u16,
) -> Vec<f32> {
    let source_channels = usize::from(source_channels);
    let target_channels = usize::from(target_channels);
    let source_frames = samples.len() / source_channels;
    if source_frames == 0 {
        return Vec::new();
    }
    let target_frames = ((source_frames as u128 * target_sample_rate_hz as u128)
        / source_sample_rate_hz as u128)
        .max(1) as usize;
    let mut out = Vec::with_capacity(target_frames * target_channels);
    for frame_idx in 0..target_frames {
        let source_idx = ((frame_idx as u128 * source_sample_rate_hz as u128)
            / target_sample_rate_hz as u128)
            .min(source_frames.saturating_sub(1) as u128) as usize;
        let source_base = source_idx * source_channels;
        for channel in 0..target_channels {
            let sample = if channel < source_channels {
                samples[source_base + channel]
            } else if source_channels == 1 {
                samples[source_base]
            } else {
                0.0
            };
            out.push(sample.clamp(-1.0, 1.0));
        }
    }
    out
}

fn playback_duration(total_samples: usize, sample_rate: u32, channels: u16) -> Duration {
    let sample_frames = total_samples as f64 / f64::from(channels);
    Duration::from_secs_f64(sample_frames / f64::from(sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_mouth_reports_quiet_outcome() {
        let mut mouth = NoopMouth;
        let outcome = mouth.speak("hello").unwrap();
        assert!(!outcome.spoken);
        assert_eq!(outcome.backend, "noop");
        assert_eq!(outcome.text_len, 5);
    }

    #[test]
    fn mono_audio_converts_to_stereo_and_resamples() {
        let converted = convert_interleaved_f32(&[0.25, -0.25], 2, 1, 4, 2);
        assert_eq!(
            converted,
            vec![0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25, -0.25]
        );
    }
}
