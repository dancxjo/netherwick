use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context, Result};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample};
use serde::{Deserialize, Serialize};
use speaking::{
    phonemicizer_for_variety, EvidenceProvenance, EvidenceSource, PhonemicizeOutput,
    PhonemicizeRequest, SpeakerId, UtteranceId, UtterancePlan, VarietyId,
};
use tongues_tts::{
    AudioChunk, BurnHifiganVocoder, BurnSpeedySpeechAcoustic, OnnxSpeechBackend, SpeechPipeline,
    SpeechSynthesisEngine, SpeechSynthesisRequest, SynthesisOptions, VocoderDecoder, VoiceConfig,
};

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
    match CpalSpeechMouth::from_env() {
        Ok(Some(mouth)) => Box::new(mouth),
        Ok(None) => Box::<NoopMouth>::default(),
        Err(error) => {
            tracing::warn!(error = %error, "failed to configure speech mouth; using noop mouth");
            Box::<NoopMouth>::default()
        }
    }
}

pub struct QueuedSpeechMouth {
    tx: Option<mpsc::Sender<MouthQueueItem>>,
    worker: Option<JoinHandle<()>>,
}

struct MouthQueueItem {
    text: String,
    outcome_tx: Option<mpsc::Sender<std::result::Result<SpeechOutcome, String>>>,
}

impl QueuedSpeechMouth {
    pub fn from_env() -> Result<Option<Self>> {
        SpeechConfig::from_env()?.map(Self::new).transpose()
    }

    pub fn new(config: SpeechConfig) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<MouthQueueItem>();
        let worker = std::thread::Builder::new()
            .name("pete-speech-mouth".to_string())
            .spawn(move || {
                println!(
                    "robot mouth loading speech components: {}",
                    config.backend.description()
                );
                let mut mouth = match CpalSpeechMouth::new(config) {
                    Ok(mouth) => mouth,
                    Err(error) => {
                        let message =
                            format!("queued speech mouth failed to load voice: {error:#}");
                        println!("robot mouth failed: {message}");
                        tracing::warn!(error = %error, "queued speech mouth failed to load voice");
                        for item in rx {
                            if let Some(outcome_tx) = item.outcome_tx {
                                let _ = outcome_tx.send(Err(message.clone()));
                            }
                        }
                        return;
                    }
                };
                println!("robot mouth voice model ready");
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
                            tracing::warn!(error = %message, text = %item.text, "queued speech mouth failed");
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
            .context("failed to spawn queued speech mouth thread")?;
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
                backend: "queued-speech".to_string(),
                ..SpeechOutcome::default()
            });
        }
        let (outcome_tx, outcome_rx) = mpsc::channel();
        self.send_item(MouthQueueItem {
            text,
            outcome_tx: Some(outcome_tx),
        })?;
        let result = match timeout {
            Some(timeout) => outcome_rx.recv_timeout(timeout).with_context(|| {
                format!("queued speech mouth did not finish within {timeout:?}")
            })?,
            None => outcome_rx
                .recv()
                .context("queued speech mouth worker did not report outcome")?,
        };
        match result {
            Ok(outcome) => Ok(outcome),
            Err(error) => anyhow::bail!(error),
        }
    }

    fn send_item(&self, item: MouthQueueItem) -> Result<()> {
        self.tx
            .as_ref()
            .context("queued speech mouth is already closed")?
            .send(item)
            .context("queued speech mouth worker is not running")
    }
}

impl Drop for QueuedSpeechMouth {
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
pub struct SpeechConfig {
    pub backend: SpeechBackendConfig,
    pub variety: String,
    pub speaker: Option<String>,
    pub output_device_name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpeechBackendConfig {
    Burn {
        acoustic_model_path: PathBuf,
        acoustic_config_path: PathBuf,
        vocoder_model_path: PathBuf,
        vocoder_config_path: PathBuf,
    },
    OnnxCompatibility {
        model_path: PathBuf,
        config_path: PathBuf,
    },
}

impl SpeechBackendConfig {
    fn description(&self) -> String {
        match self {
            Self::Burn {
                acoustic_model_path,
                vocoder_model_path,
                ..
            } => format!(
                "{} + {}",
                acoustic_model_path.display(),
                vocoder_model_path.display()
            ),
            Self::OnnxCompatibility { model_path, .. } => model_path.display().to_string(),
        }
    }
}

impl SpeechConfig {
    pub fn burn(
        acoustic_model_path: impl Into<PathBuf>,
        acoustic_config_path: impl Into<PathBuf>,
        vocoder_model_path: impl Into<PathBuf>,
        vocoder_config_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            backend: SpeechBackendConfig::Burn {
                acoustic_model_path: acoustic_model_path.into(),
                acoustic_config_path: acoustic_config_path.into(),
                vocoder_model_path: vocoder_model_path.into(),
                vocoder_config_path: vocoder_config_path.into(),
            },
            variety: DEFAULT_TTS_VARIETY.to_string(),
            speaker: None,
            output_device_name: None,
        }
    }

    pub fn onnx_compatibility(
        model_path: impl Into<PathBuf>,
        config_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            backend: SpeechBackendConfig::OnnxCompatibility {
                model_path: model_path.into(),
                config_path: config_path.into(),
            },
            variety: DEFAULT_TTS_VARIETY.to_string(),
            speaker: None,
            output_device_name: None,
        }
    }

    pub fn from_env() -> Result<Option<Self>> {
        let burn_env_requested = [
            "PETE_TTS_ACOUSTIC_MODEL",
            "PETE_TTS_ACOUSTIC_CONFIG",
            "PETE_TTS_VOCODER_MODEL",
            "PETE_TTS_VOCODER_CONFIG",
        ]
        .iter()
        .any(|name| env_path(name).is_some());
        let mut config = if burn_env_requested {
            let acoustic_model_path = env_path("PETE_TTS_ACOUSTIC_MODEL")
                .context("PETE_TTS_ACOUSTIC_MODEL is required for a Burn speech backend")?;
            let acoustic_config_path = env_path("PETE_TTS_ACOUSTIC_CONFIG")
                .unwrap_or_else(|| sibling_config_path(&acoustic_model_path));
            let vocoder_model_path = env_path("PETE_TTS_VOCODER_MODEL")
                .context("PETE_TTS_VOCODER_MODEL is required with PETE_TTS_ACOUSTIC_MODEL")?;
            let vocoder_config_path = env_path("PETE_TTS_VOCODER_CONFIG")
                .unwrap_or_else(|| sibling_config_path(&vocoder_model_path));
            Self::burn(
                acoustic_model_path,
                acoustic_config_path,
                vocoder_model_path,
                vocoder_config_path,
            )
        } else if let Some(model_path) =
            env_path_with_deprecated("PETE_TTS_VOICE", "PETE_TTS_PIPER_VOICE")
        {
            let config_path = env_path_with_deprecated("PETE_TTS_CONFIG", "PETE_TTS_PIPER_CONFIG")
                .unwrap_or_else(|| tongues_tts::voice_config_path(&model_path));
            Self::onnx_compatibility(model_path, config_path)
        } else if let Some(config) = burn_config_from_model_home() {
            config
        } else {
            let default_voice = tongues_tts::default_voice_model();
            let model_path = tongues_tts::default_voice_model_path(default_voice.clone());
            let config_path = tongues_tts::default_voice_config_path(default_voice);
            if !model_path.is_file() || !config_path.is_file() {
                return Ok(None);
            };
            Self::onnx_compatibility(model_path, config_path)
        };
        config.variety = std::env::var("PETE_TTS_VARIETY")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_TTS_VARIETY.to_string());
        config.speaker = std::env::var("PETE_TTS_SPEAKER")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        config.output_device_name = std::env::var("PETE_TTS_OUTPUT_DEVICE")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
        Ok(Some(config))
    }
}

fn sibling_config_path(model_path: &std::path::Path) -> PathBuf {
    model_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join("config.json")
}

fn burn_config_from_model_home() -> Option<SpeechConfig> {
    let home = env_path("MORTAR_SEA_HOME")?;
    let acoustic_dir = home.join("models/speech/coqui/en/ljspeech/speedy-speech");
    let vocoder_dir = home.join("models/speech/coqui/en/ljspeech/hifigan-v2");
    let config = SpeechConfig::burn(
        acoustic_dir.join("model_file.pth"),
        acoustic_dir.join("config.json"),
        vocoder_dir.join("model_file.pth"),
        vocoder_dir.join("config.json"),
    );
    match &config.backend {
        SpeechBackendConfig::Burn {
            acoustic_model_path,
            acoustic_config_path,
            vocoder_model_path,
            vocoder_config_path,
        } if [
            acoustic_model_path,
            acoustic_config_path,
            vocoder_model_path,
            vocoder_config_path,
        ]
        .iter()
        .all(|path| path.is_file()) =>
        {
            Some(config)
        }
        _ => None,
    }
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn env_path_with_deprecated(name: &str, deprecated_name: &str) -> Option<PathBuf> {
    if let Some(path) = env_path(name) {
        if env_path(deprecated_name).is_some() {
            tracing::warn!(
                old = deprecated_name,
                new = name,
                "deprecated speech environment variable ignored because neutral variable is set"
            );
        }
        return Some(path);
    }
    let path = env_path(deprecated_name);
    if path.is_some() {
        tracing::warn!(
            old = deprecated_name,
            new = name,
            "deprecated speech environment variable is still supported temporarily"
        );
    }
    path
}

type BurnSpeech = SpeechPipeline<
    BurnSpeedySpeechAcoustic<NdArray<f32>>,
    VocoderDecoder<BurnHifiganVocoder<NdArray<f32>>>,
>;

enum SpeechBackend {
    Burn(Box<BurnSpeech>),
    OnnxCompatibility(Box<OnnxSpeechBackend>),
}

impl SpeechBackend {
    fn engine_mut(&mut self) -> &mut dyn SpeechSynthesisEngine {
        match self {
            Self::Burn(speech) => speech.as_mut(),
            Self::OnnxCompatibility(speech) => speech.as_mut(),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Burn(_) => "tongues-burn-cpal",
            Self::OnnxCompatibility(_) => "tongues-onnx-compatibility-cpal",
        }
    }
}

pub struct CpalSpeechMouth {
    config: SpeechConfig,
    speech: SpeechBackend,
}

impl CpalSpeechMouth {
    pub fn new(config: SpeechConfig) -> Result<Self> {
        let speech = match &config.backend {
            SpeechBackendConfig::Burn {
                acoustic_model_path,
                acoustic_config_path,
                vocoder_model_path,
                vocoder_config_path,
            } => {
                let device = NdArrayDevice::Cpu;
                let acoustic = BurnSpeedySpeechAcoustic::load(
                    acoustic_config_path,
                    acoustic_model_path,
                    device,
                )
                .context("failed to load Burn acoustic model")?;
                let vocoder =
                    BurnHifiganVocoder::load(vocoder_config_path, vocoder_model_path, device)
                        .context("failed to load Burn neural vocoder")?;
                let pipeline = SpeechPipeline::new(acoustic, VocoderDecoder::new(vocoder))
                    .context("Burn speech components are incompatible")?;
                SpeechBackend::Burn(Box::new(pipeline))
            }
            SpeechBackendConfig::OnnxCompatibility {
                model_path,
                config_path,
            } => {
                let voice_config = VoiceConfig::from_json_file(config_path)
                    .with_context(|| format!("failed to read {}", config_path.display()))?;
                let speech = OnnxSpeechBackend::load(model_path, voice_config)
                    .context("failed to load ONNX compatibility speech backend")?;
                SpeechBackend::OnnxCompatibility(Box::new(speech))
            }
        };
        Ok(Self { config, speech })
    }

    pub fn from_env() -> Result<Option<Self>> {
        SpeechConfig::from_env()?.map(Self::new).transpose()
    }
}

impl Mouth for CpalSpeechMouth {
    fn speak(&mut self, text: &str) -> Result<SpeechOutcome> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(SpeechOutcome {
                spoken: false,
                backend: self.speech.label().to_string(),
                ..SpeechOutcome::default()
            });
        }

        let plan = utterance_plan_from_text(text, &self.config.variety)?;
        let backend_label = self.speech.label();
        play_tongues_streaming(
            self.speech.engine_mut(),
            &plan,
            text.len(),
            self.config.output_device_name.as_deref(),
            self.config.speaker.as_deref(),
            backend_label,
        )
    }
}

fn play_tongues_streaming(
    speech: &mut dyn SpeechSynthesisEngine,
    plan: &UtterancePlan,
    text_len: usize,
    output_device_name: Option<&str>,
    speaker: Option<&str>,
    backend_label: &str,
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
    let mut plan = plan.clone();
    plan.speaker = speaker.map(|name| SpeakerId(name.to_string()));
    println!("robot mouth synthesizing speech");
    speech
        .synthesize_plan_streaming(
            &SpeechSynthesisRequest {
                plan,
                options: SynthesisOptions::default(),
            },
            &mut |audio: AudioChunk| {
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
            },
        )
        .context("Tongues speech synthesis failed")?;

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
        backend: backend_label.to_string(),
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
        .context("failed to phonemicize text into a speech plan")?;
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
        speaker_reference: None,
        style: None,
        provenance: EvidenceProvenance {
            source: EvidenceSource::TtsPlan,
            method: "pete mouth phonemicized speech plan".into(),
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
    use std::sync::Mutex as StdMutex;

    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

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

    #[test]
    fn neutral_tts_env_wins_over_deprecated_voice_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_tts_env();
        std::env::set_var("PETE_TTS_VOICE", "/tmp/neutral.onnx");
        std::env::set_var("PETE_TTS_CONFIG", "/tmp/neutral.onnx.json");
        std::env::set_var("PETE_TTS_PIPER_VOICE", "/tmp/deprecated.onnx");
        std::env::set_var("PETE_TTS_PIPER_CONFIG", "/tmp/deprecated.onnx.json");
        std::env::set_var("PETE_TTS_SPEAKER", "p225");

        let config = SpeechConfig::from_env().unwrap().unwrap();

        assert_eq!(
            config.backend,
            SpeechBackendConfig::OnnxCompatibility {
                model_path: PathBuf::from("/tmp/neutral.onnx"),
                config_path: PathBuf::from("/tmp/neutral.onnx.json"),
            }
        );
        assert_eq!(config.speaker.as_deref(), Some("p225"));
        clear_tts_env();
    }

    #[test]
    fn deprecated_tts_env_bridge_still_loads() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_tts_env();
        std::env::set_var("PETE_TTS_PIPER_VOICE", "/tmp/deprecated.onnx");
        std::env::set_var("PETE_TTS_PIPER_CONFIG", "/tmp/deprecated.onnx.json");

        let config = SpeechConfig::from_env().unwrap().unwrap();

        assert_eq!(
            config.backend,
            SpeechBackendConfig::OnnxCompatibility {
                model_path: PathBuf::from("/tmp/deprecated.onnx"),
                config_path: PathBuf::from("/tmp/deprecated.onnx.json"),
            }
        );
        clear_tts_env();
    }

    #[test]
    fn burn_component_env_selects_native_pipeline() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_tts_env();
        std::env::set_var("PETE_TTS_ACOUSTIC_MODEL", "/tmp/acoustic/model_file.pth");
        std::env::set_var("PETE_TTS_VOCODER_MODEL", "/tmp/vocoder/model_file.pth");

        let config = SpeechConfig::from_env().unwrap().unwrap();

        assert_eq!(
            config.backend,
            SpeechBackendConfig::Burn {
                acoustic_model_path: PathBuf::from("/tmp/acoustic/model_file.pth"),
                acoustic_config_path: PathBuf::from("/tmp/acoustic/config.json"),
                vocoder_model_path: PathBuf::from("/tmp/vocoder/model_file.pth"),
                vocoder_config_path: PathBuf::from("/tmp/vocoder/config.json"),
            }
        );
        clear_tts_env();
    }

    fn clear_tts_env() {
        for name in [
            "PETE_TTS_ACOUSTIC_MODEL",
            "PETE_TTS_ACOUSTIC_CONFIG",
            "PETE_TTS_VOCODER_MODEL",
            "PETE_TTS_VOCODER_CONFIG",
            "PETE_TTS_VOICE",
            "PETE_TTS_CONFIG",
            "PETE_TTS_PIPER_VOICE",
            "PETE_TTS_PIPER_CONFIG",
            "PETE_TTS_SPEAKER",
            "PETE_TTS_VARIETY",
            "PETE_TTS_OUTPUT_DEVICE",
            "MORTAR_SEA_HOME",
        ] {
            std::env::remove_var(name);
        }
    }
}
