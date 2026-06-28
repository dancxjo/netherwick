use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample};
use serde::{Deserialize, Serialize};

const DEFAULT_PIPER_SAMPLE_RATE_HZ: u32 = 22_050;
const DEFAULT_PIPER_CHANNELS: u16 = 1;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiperConfig {
    pub executable: PathBuf,
    pub model_path: PathBuf,
    pub config_path: Option<PathBuf>,
    pub num_threads: Option<usize>,
    pub sample_rate_hz: u32,
    pub channels: u16,
}

impl PiperConfig {
    pub fn new(executable: impl Into<PathBuf>, model_path: impl Into<PathBuf>) -> Self {
        Self {
            executable: executable.into(),
            model_path: model_path.into(),
            config_path: None,
            num_threads: Some(1),
            sample_rate_hz: DEFAULT_PIPER_SAMPLE_RATE_HZ,
            channels: DEFAULT_PIPER_CHANNELS,
        }
    }

    pub fn from_env() -> Result<Option<Self>> {
        let Some(executable) = env_path("NETHERWICK_TTS_PIPER_BIN") else {
            return Ok(None);
        };
        let model_path = env_path("NETHERWICK_TTS_PIPER_VOICE")
            .context("NETHERWICK_TTS_PIPER_BIN is set but NETHERWICK_TTS_PIPER_VOICE is missing")?;
        let mut config = Self::new(executable, model_path);
        config.config_path = env_path("NETHERWICK_TTS_PIPER_CONFIG");
        config.num_threads = std::env::var("NETHERWICK_TTS_PIPER_THREADS")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(|value| value.parse::<usize>())
            .transpose()
            .context("failed to parse NETHERWICK_TTS_PIPER_THREADS")?;
        if let Some(sample_rate) = std::env::var("NETHERWICK_TTS_SAMPLE_RATE_HZ")
            .ok()
            .filter(|value| !value.trim().is_empty())
        {
            config.sample_rate_hz = sample_rate
                .parse()
                .context("failed to parse NETHERWICK_TTS_SAMPLE_RATE_HZ")?;
        }
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
}

impl PiperCpalMouth {
    pub fn new(config: PiperConfig) -> Self {
        Self { config }
    }

    pub fn from_env() -> Result<Option<Self>> {
        Ok(PiperConfig::from_env()?.map(Self::new))
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

        let samples = synthesize_piper_raw(&self.config, text)?;
        play_interleaved_f32(
            samples,
            self.config.sample_rate_hz,
            self.config.channels,
            text.len(),
            "piper-cpal",
        )
    }
}

fn synthesize_piper_raw(config: &PiperConfig, text: &str) -> Result<Vec<f32>> {
    let mut command = Command::new(&config.executable);
    command
        .arg("--model")
        .arg(&config.model_path)
        .arg("--output-raw")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(num_threads) = config.num_threads {
        command.arg("--num-threads").arg(num_threads.to_string());
    }
    if let Some(config_path) = &config.config_path {
        command.arg("--config").arg(config_path);
    }

    let mut child = command
        .spawn()
        .with_context(|| format!("failed to spawn Piper at {}", config.executable.display()))?;
    {
        let mut stdin = child.stdin.take().context("failed to open Piper stdin")?;
        stdin
            .write_all(text.as_bytes())
            .context("failed to write speech text to Piper stdin")?;
        stdin
            .write_all(b"\n")
            .context("failed to finish Piper stdin")?;
    }

    let output = child.wait_with_output().context("failed to read Piper output")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Piper exited with {}: {}", output.status, stderr.trim());
    }
    anyhow::ensure!(
        output.stdout.len() % 2 == 0,
        "Piper returned an odd number of raw PCM bytes"
    );

    Ok(output
        .stdout
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / i16::MAX as f32)
        .collect())
}

fn play_interleaved_f32(
    samples: Vec<f32>,
    source_sample_rate_hz: u32,
    source_channels: u16,
    text_len: usize,
    backend: &str,
) -> Result<SpeechOutcome> {
    anyhow::ensure!(!samples.is_empty(), "speech synthesis produced no audio");
    anyhow::ensure!(source_sample_rate_hz > 0, "speech sample rate must be positive");
    anyhow::ensure!(source_channels > 0, "speech channel count must be positive");

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::anyhow!("no default output device available"))?;
    let device_name = device
        .name()
        .unwrap_or_else(|_| "<unknown output device>".to_string());
    let output_config = output_config(&device)?;
    let converted = convert_interleaved_f32(
        &samples,
        source_sample_rate_hz,
        source_channels,
        output_config.sample_rate_hz,
        output_config.channels,
    );
    let sample_count = converted.len();
    let duration = playback_duration(sample_count, output_config.sample_rate_hz, output_config.channels);
    let samples = Arc::new(converted);
    let cursor = Arc::new(AtomicUsize::new(0));
    let stream = build_output_stream(&device, &output_config, Arc::clone(&samples), Arc::clone(&cursor))?;
    stream
        .play()
        .with_context(|| format!("failed to start speech playback on {device_name}"))?;

    while cursor.load(Ordering::Relaxed) < sample_count {
        std::thread::sleep(Duration::from_millis(10));
    }
    std::thread::sleep(Duration::from_millis(20));
    drop(stream);

    Ok(SpeechOutcome {
        spoken: true,
        backend: backend.to_string(),
        text_len,
        sample_rate_hz: Some(output_config.sample_rate_hz),
        channels: Some(output_config.channels),
        sample_count,
        duration_ms: Some(duration.as_millis() as u64),
        device: Some(device_name),
    })
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

fn build_output_stream(
    device: &cpal::Device,
    config: &OutputConfig,
    samples: Arc<Vec<f32>>,
    cursor: Arc<AtomicUsize>,
) -> Result<cpal::Stream> {
    let err_fn = |err| tracing::warn!(error = %err, "speech output stream error");
    match config.sample_format {
        cpal::SampleFormat::F32 => build_typed_output_stream::<f32>(device, &config.stream_config, samples, cursor, err_fn),
        cpal::SampleFormat::F64 => build_typed_output_stream::<f64>(device, &config.stream_config, samples, cursor, err_fn),
        cpal::SampleFormat::I8 => build_typed_output_stream::<i8>(device, &config.stream_config, samples, cursor, err_fn),
        cpal::SampleFormat::I16 => build_typed_output_stream::<i16>(device, &config.stream_config, samples, cursor, err_fn),
        cpal::SampleFormat::I32 => build_typed_output_stream::<i32>(device, &config.stream_config, samples, cursor, err_fn),
        cpal::SampleFormat::I64 => build_typed_output_stream::<i64>(device, &config.stream_config, samples, cursor, err_fn),
        cpal::SampleFormat::U8 => build_typed_output_stream::<u8>(device, &config.stream_config, samples, cursor, err_fn),
        cpal::SampleFormat::U16 => build_typed_output_stream::<u16>(device, &config.stream_config, samples, cursor, err_fn),
        cpal::SampleFormat::U32 => build_typed_output_stream::<u32>(device, &config.stream_config, samples, cursor, err_fn),
        cpal::SampleFormat::U64 => build_typed_output_stream::<u64>(device, &config.stream_config, samples, cursor, err_fn),
        sample_format => anyhow::bail!("unsupported output sample format: {sample_format:?}"),
    }
}

fn build_typed_output_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    samples: Arc<Vec<f32>>,
    cursor: Arc<AtomicUsize>,
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
                    let idx = cursor.fetch_add(1, Ordering::Relaxed);
                    let sample = samples.get(idx).copied().unwrap_or(0.0);
                    *out = T::from_sample(sample);
                }
            },
            err_fn,
            None,
        )
        .context("failed to build speech output stream")
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
        assert_eq!(converted, vec![0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25, -0.25]);
    }
}

