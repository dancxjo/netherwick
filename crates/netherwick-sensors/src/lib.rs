#[cfg(any(feature = "linux-hardware", test))]
use anyhow::Context;
use anyhow::Result;
use async_trait::async_trait;
use netherwick_actions::{ActionPrimitive, LlmActionProposal};
use netherwick_body::BodySense;
use netherwick_now::{
    EarSense, ExtensionSense, EyeSense, FaceSense, GpsSense, ImuSense, KinectSense, ObjectSense,
    RangeSense, VectorArtifact, VoiceSense, FACE_VECTOR_COLLECTION,
    IMAGE_DESCRIPTION_VECTOR_COLLECTION, IMAGE_VECTOR_COLLECTION, SCENE_VECTOR_COLLECTION,
};
use netherwick_now::{Now, PredictionSense, SurpriseSense};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

type TimeMs = u64;

#[cfg(feature = "linux-hardware")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(feature = "linux-hardware")]
use serialport::SerialPort;
#[cfg(feature = "linux-hardware")]
use std::io::Write;
#[cfg(feature = "linux-hardware")]
use std::io::{ErrorKind, Read};
#[cfg(feature = "linux-hardware")]
use std::os::fd::AsRawFd;
#[cfg(feature = "linux-hardware")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "linux-hardware")]
use std::time::Duration;
#[cfg(feature = "linux-hardware")]
use v4l::buffer::Type;
#[cfg(feature = "linux-hardware")]
use v4l::io::traits::CaptureStream;
#[cfg(feature = "linux-hardware")]
use v4l::prelude::{MmapStream, *};
#[cfg(feature = "linux-hardware")]
use v4l::video::Capture;
#[cfg(feature = "linux-hardware")]
use v4l::{Format, FourCC};

#[async_trait]
pub trait SenseProducer {
    async fn poll(&mut self) -> Result<SensePacket>;
}

#[async_trait]
pub trait World: Send {
    async fn snapshot(&mut self) -> Result<WorldSnapshot>;
    async fn apply_update(&mut self, update: WorldUpdate) -> Result<()>;

    async fn set_body(&mut self, body: BodySense) -> Result<()> {
        self.apply_update(WorldUpdate {
            body: Some(body),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_eye_frame(&mut self, frame: EyeFrame) -> Result<()> {
        self.apply_update(WorldUpdate {
            eye_frame: Some(frame),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_eye_sense(&mut self, eye: EyeSense) -> Result<()> {
        self.apply_update(WorldUpdate {
            eye: Some(eye),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_ear_pcm_frame(&mut self, frame: PcmAudioFrame) -> Result<()> {
        self.apply_update(WorldUpdate {
            ear_pcm: Some(frame),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_ear_sense(&mut self, ear: EarSense) -> Result<()> {
        self.apply_update(WorldUpdate {
            ear: Some(ear),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_range_sense(&mut self, range: RangeSense) -> Result<()> {
        self.apply_update(WorldUpdate {
            range: Some(range),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_imu_sense(&mut self, imu: ImuSense) -> Result<()> {
        self.apply_update(WorldUpdate {
            imu: Some(imu),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_gps_sense(&mut self, gps: Option<GpsSense>) -> Result<()> {
        self.apply_update(WorldUpdate {
            gps,
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_kinect_sense(&mut self, kinect: KinectSense) -> Result<()> {
        self.apply_update(WorldUpdate {
            kinect: Some(kinect),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_object_sense(&mut self, objects: ObjectSense) -> Result<()> {
        self.apply_update(WorldUpdate {
            objects: Some(objects),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_face_sense(&mut self, face: FaceSense) -> Result<()> {
        self.apply_update(WorldUpdate {
            face: Some(face),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_voice_sense(&mut self, voice: VoiceSense) -> Result<()> {
        self.apply_update(WorldUpdate {
            voice: Some(voice),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_extensions(&mut self, extensions: Vec<ExtensionSense>) -> Result<()> {
        self.apply_update(WorldUpdate {
            extensions: Some(extensions),
            ..WorldUpdate::default()
        })
        .await
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SensePacket {
    Eye(EyeSense),
    EyeFrame(EyeFrame),
    Ear(EarSense),
    EarPcm(PcmAudioFrame),
    Range(RangeSense),
    Imu(ImuSense),
    Gps(GpsSense),
    Kinect(KinectSense),
    Face(FaceSense),
    Voice(VoiceSense),
    Objects(ObjectSense),
    Extension(ExtensionSense),
}

#[derive(Clone, Debug, Default)]
pub struct NowBuilder {
    last_snapshot: WorldSnapshot,
    last_updates: SensorUpdateTimes,
}

#[derive(Clone, Debug, Default)]
pub struct FrameProcessor {
    last_processed_frame_key: Option<FrameKey>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FrameKey {
    captured_at_ms: u64,
    width: u32,
    height: u32,
    format: String,
    byte_len: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProcessedFrame {
    pub eye: EyeSense,
    pub face: FaceSense,
    pub summary: String,
    pub source_frame_id: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorUpdateTimes {
    pub body: Option<TimeMs>,
    pub eye: Option<TimeMs>,
    pub ear: Option<TimeMs>,
    pub range: Option<TimeMs>,
    pub imu: Option<TimeMs>,
    pub gps: Option<TimeMs>,
    pub kinect: Option<TimeMs>,
    pub face: Option<TimeMs>,
    pub voice: Option<TimeMs>,
}

impl NowBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(
        &mut self,
        t_ms: TimeMs,
        mut body: BodySense,
        packets: Vec<SensePacket>,
    ) -> Result<Now> {
        body.last_update_ms = body.last_update_ms.max(t_ms);
        self.last_updates.body = Some(body.last_update_ms);
        self.last_snapshot.body = body;
        self.last_snapshot.extensions.clear();

        for packet in packets {
            match packet {
                SensePacket::Eye(eye) => {
                    self.last_snapshot.eye = eye;
                    self.last_updates.eye = Some(t_ms);
                }
                SensePacket::EyeFrame(frame) => {
                    self.last_snapshot.eye.frames = vec![bytes_to_unit_signal(&frame.bytes)];
                    self.last_snapshot.eye_frame = Some(frame);
                    self.last_updates.eye = Some(t_ms);
                }
                SensePacket::Ear(ear) => {
                    self.last_snapshot.ear = ear;
                    self.last_updates.ear = Some(t_ms);
                }
                SensePacket::EarPcm(frame) => {
                    self.last_snapshot.ear.features = vec![pcm_to_unit_signal(&frame.samples)];
                    self.last_snapshot.ear_pcm = Some(frame);
                    self.last_updates.ear = Some(t_ms);
                }
                SensePacket::Range(range) => {
                    self.last_snapshot.range = range;
                    self.last_updates.range = Some(t_ms);
                }
                SensePacket::Imu(imu) => {
                    self.last_snapshot.imu = imu;
                    self.last_updates.imu = Some(t_ms);
                }
                SensePacket::Gps(gps) => {
                    self.last_snapshot.gps = Some(gps);
                    self.last_updates.gps = Some(t_ms);
                }
                SensePacket::Kinect(kinect) => {
                    self.last_snapshot.kinect = kinect;
                    self.last_updates.kinect = Some(t_ms);
                }
                SensePacket::Face(face) => {
                    self.last_snapshot.face = face;
                    self.last_updates.face = Some(t_ms);
                }
                SensePacket::Voice(voice) => {
                    self.last_snapshot.voice = voice;
                    self.last_updates.voice = Some(t_ms);
                }
                SensePacket::Objects(objects) => {
                    self.last_snapshot.objects = objects;
                }
                SensePacket::Extension(extension) => {
                    self.last_snapshot.extensions.push(extension);
                }
            }
        }

        let mut now = self.last_snapshot.to_now(t_ms);
        now.extensions.insert(
            "sensor_status".to_string(),
            serde_json::json!({
                "last_update_ms": self.last_updates,
                "age_ms": self.last_updates.age_ms(t_ms),
            }),
        );
        Ok(now)
    }

    pub fn snapshot(&self) -> WorldSnapshot {
        self.last_snapshot.clone()
    }
}

impl FrameProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn process_packets(&mut self, t_ms: TimeMs, packets: &mut Vec<SensePacket>) {
        let Some(frame) = packets.iter().rev().find_map(|packet| match packet {
            SensePacket::EyeFrame(frame) => Some(frame),
            _ => None,
        }) else {
            return;
        };
        let Some(processed) = self.process_frame(t_ms, frame) else {
            return;
        };
        let summary_values = summary_extension_values(&processed);
        packets.push(SensePacket::Eye(processed.eye));
        if !processed.face.embeddings.is_empty() || !processed.face.vectors.is_empty() {
            packets.push(SensePacket::Face(processed.face));
        }
        packets.push(SensePacket::Extension(ExtensionSense {
            schema_version: 1,
            name: "vision.frame_summary".to_string(),
            values: summary_values,
        }));
    }

    pub fn process_snapshot(&mut self, t_ms: TimeMs, snapshot: &mut WorldSnapshot) {
        let Some(frame) = snapshot.eye_frame.clone() else {
            return;
        };
        let Some(processed) = self.process_frame(t_ms, &frame) else {
            return;
        };
        let summary_values = summary_extension_values(&processed);
        snapshot.eye = processed.eye;
        if !processed.face.embeddings.is_empty() || !processed.face.vectors.is_empty() {
            snapshot.face = processed.face;
        }
        snapshot.extensions.push(ExtensionSense {
            schema_version: 1,
            name: "vision.frame_summary".to_string(),
            values: summary_values,
        });
    }

    pub fn process_frame(&mut self, t_ms: TimeMs, frame: &EyeFrame) -> Option<ProcessedFrame> {
        let key = FrameKey::from(frame);
        if self.last_processed_frame_key.as_ref() == Some(&key) {
            return None;
        }
        self.last_processed_frame_key = Some(key);
        Some(process_eye_frame(t_ms, frame))
    }
}

fn process_eye_frame(t_ms: TimeMs, frame: &EyeFrame) -> ProcessedFrame {
    let source_frame_id = format!(
        "eye-{}-{}x{}-{}",
        frame.captured_at_ms,
        frame.width,
        frame.height,
        frame.bytes.len()
    );
    let signal = bytes_to_unit_signal(&frame.bytes);
    let mut eye = EyeSense {
        schema_version: 1,
        frames: vec![signal.clone()],
        ..EyeSense::default()
    };
    eye.image_vectors.push(
        VectorArtifact::new(
            IMAGE_VECTOR_COLLECTION,
            source_frame_id.clone(),
            signal.clone(),
        )
        .with_model("raw-byte-unit-signal-v0")
        .with_source_frame_id(source_frame_id.clone())
        .with_occurred_at_ms(t_ms),
    );
    eye.image_description_vectors.push(
        VectorArtifact::new(
            IMAGE_DESCRIPTION_VECTOR_COLLECTION,
            format!("{source_frame_id}-summary"),
            frame_summary_vector(frame, &signal),
        )
        .with_model("frame-summary-v0")
        .with_source_frame_id(source_frame_id.clone())
        .with_occurred_at_ms(t_ms),
    );
    eye.scene_vectors.push(
        VectorArtifact::new(
            SCENE_VECTOR_COLLECTION,
            format!("{source_frame_id}-scene"),
            frame_summary_vector(frame, &signal),
        )
        .with_model("scene-summary-v0")
        .with_source_frame_id(source_frame_id.clone())
        .with_occurred_at_ms(t_ms),
    );

    ProcessedFrame {
        eye,
        face: FaceSense {
            schema_version: 1,
            embeddings: Vec::new(),
            vectors: vec![VectorArtifact::new(
                FACE_VECTOR_COLLECTION,
                format!("{source_frame_id}-no-face"),
                Vec::new(),
            )
            .with_model("no-face-detected-v0")
            .with_source_frame_id(source_frame_id.clone())
            .with_occurred_at_ms(t_ms)],
        },
        summary: format!(
            "{:?} frame {}x{}, {} bytes",
            frame.format,
            frame.width,
            frame.height,
            frame.bytes.len()
        ),
        source_frame_id,
    }
}

fn summary_extension_values(processed: &ProcessedFrame) -> Vec<f32> {
    let signal = processed
        .eye
        .frames
        .first()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mean = if signal.is_empty() {
        0.0
    } else {
        signal.iter().sum::<f32>() / signal.len() as f32
    };
    vec![
        signal.len() as f32,
        mean,
        processed.eye.image_vectors.len() as f32,
        processed.face.vectors.len() as f32,
    ]
}

fn frame_summary_vector(frame: &EyeFrame, signal: &[f32]) -> Vec<f32> {
    let mean = if signal.is_empty() {
        0.0
    } else {
        signal.iter().sum::<f32>() / signal.len() as f32
    };
    vec![
        frame.width as f32,
        frame.height as f32,
        frame.bytes.len() as f32,
        mean,
    ]
}

impl SensorUpdateTimes {
    fn age_ms(&self, t_ms: TimeMs) -> serde_json::Value {
        serde_json::json!({
            "body": self.body.map(|value| t_ms.saturating_sub(value)),
            "eye": self.eye.map(|value| t_ms.saturating_sub(value)),
            "ear": self.ear.map(|value| t_ms.saturating_sub(value)),
            "range": self.range.map(|value| t_ms.saturating_sub(value)),
            "imu": self.imu.map(|value| t_ms.saturating_sub(value)),
            "gps": self.gps.map(|value| t_ms.saturating_sub(value)),
            "kinect": self.kinect.map(|value| t_ms.saturating_sub(value)),
            "face": self.face.map(|value| t_ms.saturating_sub(value)),
            "voice": self.voice.map(|value| t_ms.saturating_sub(value)),
        })
    }
}

pub use netherwick_now::{EyeFrame, EyeFrameFormat};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PcmAudioFrame {
    pub captured_at_ms: u64,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples: Vec<i16>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldSnapshot {
    pub body: BodySense,
    pub final_selected_action: Option<ActionPrimitive>,
    pub llm_action_proposal: Option<LlmActionProposal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_debug: Option<serde_json::Value>,
    pub eye_frame: Option<EyeFrame>,
    pub ear_pcm: Option<PcmAudioFrame>,
    pub eye: EyeSense,
    pub ear: EarSense,
    pub range: RangeSense,
    pub imu: ImuSense,
    pub gps: Option<GpsSense>,
    pub kinect: KinectSense,
    pub objects: ObjectSense,
    pub face: FaceSense,
    pub voice: VoiceSense,
    pub extensions: Vec<ExtensionSense>,
}

impl Default for WorldSnapshot {
    fn default() -> Self {
        Self {
            body: BodySense::default(),
            final_selected_action: None,
            llm_action_proposal: None,
            action_debug: None,
            eye_frame: None,
            ear_pcm: None,
            eye: EyeSense {
                schema_version: 1,
                ..EyeSense::default()
            },
            ear: EarSense {
                schema_version: 1,
                ..EarSense::default()
            },
            range: RangeSense {
                schema_version: 1,
                ..RangeSense::default()
            },
            imu: ImuSense {
                schema_version: 1,
                ..ImuSense::default()
            },
            gps: None,
            kinect: KinectSense {
                schema_version: 1,
                ..KinectSense::default()
            },
            objects: ObjectSense {
                schema_version: 1,
                ..ObjectSense::default()
            },
            face: FaceSense {
                schema_version: 1,
                ..FaceSense::default()
            },
            voice: VoiceSense {
                schema_version: 1,
                ..VoiceSense::default()
            },
            extensions: Vec::new(),
        }
    }
}

impl WorldSnapshot {
    pub fn to_now(&self, t_ms: u64) -> Now {
        let mut now = Now::blank(t_ms, self.body.clone());
        now.eye = self.eye.clone();
        now.eye_frame = self.eye_frame.clone();
        now.ear = self.ear.clone();
        now.face = self.face.clone();
        now.voice = self.voice.clone();
        now.range = self.range.clone();
        now.imu = self.imu.clone();
        now.gps = self.gps.clone();
        now.kinect = self.kinect.clone();
        now.objects = self.objects.clone();
        now.predictions = PredictionSense {
            schema_version: 1,
            ..PredictionSense::default()
        };
        now.surprise = SurpriseSense {
            schema_version: 1,
            ..SurpriseSense::default()
        };
        for extension in &self.extensions {
            now.extensions.insert(
                extension.name.clone(),
                serde_json::json!({
                    "schema_version": extension.schema_version,
                    "values": extension.values,
                }),
            );
        }
        now
    }
}

impl From<&EyeFrame> for FrameKey {
    fn from(frame: &EyeFrame) -> Self {
        Self {
            captured_at_ms: frame.captured_at_ms,
            width: frame.width,
            height: frame.height,
            format: format!("{:?}", frame.format),
            byte_len: frame.bytes.len(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorldUpdate {
    pub body: Option<BodySense>,
    pub eye_frame: Option<EyeFrame>,
    pub ear_pcm: Option<PcmAudioFrame>,
    pub eye: Option<EyeSense>,
    pub ear: Option<EarSense>,
    pub range: Option<RangeSense>,
    pub imu: Option<ImuSense>,
    pub gps: Option<GpsSense>,
    pub kinect: Option<KinectSense>,
    pub objects: Option<ObjectSense>,
    pub face: Option<FaceSense>,
    pub voice: Option<VoiceSense>,
    pub extensions: Option<Vec<ExtensionSense>>,
}

impl WorldUpdate {
    pub fn apply_to(self, snapshot: &mut WorldSnapshot) {
        if let Some(body) = self.body {
            snapshot.body = body;
        }
        if let Some(frame) = self.eye_frame {
            snapshot.eye_frame = Some(frame);
        }
        if let Some(frame) = self.ear_pcm {
            snapshot.ear_pcm = Some(frame);
        }
        if let Some(eye) = self.eye {
            snapshot.eye = eye;
        }
        if let Some(ear) = self.ear {
            snapshot.ear = ear;
        }
        if let Some(range) = self.range {
            snapshot.range = range;
        }
        if let Some(imu) = self.imu {
            snapshot.imu = imu;
        }
        if self.gps.is_some() {
            snapshot.gps = self.gps;
        }
        if let Some(kinect) = self.kinect {
            snapshot.kinect = kinect;
        }
        if let Some(objects) = self.objects {
            snapshot.objects = objects;
        }
        if let Some(face) = self.face {
            snapshot.face = face;
        }
        if let Some(voice) = self.voice {
            snapshot.voice = voice;
        }
        if let Some(extensions) = self.extensions {
            snapshot.extensions = extensions;
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LinuxWorldConfig {
    pub camera_device: Option<String>,
    pub gps_serial_port: Option<String>,
    pub gps_baud_rate: u32,
    pub microphone_name: Option<String>,
    pub audio_sample_rate_hz: u32,
    pub audio_channels: u16,
}

#[cfg(feature = "linux-hardware")]
pub struct LinuxWorld {
    snapshot: WorldSnapshot,
    microphone: Option<CpalMicrophone>,
    camera: Option<V4lCamera>,
    gps: Option<Ublox7Gps>,
}

#[cfg(feature = "linux-hardware")]
impl LinuxWorld {
    pub fn new(config: LinuxWorldConfig) -> Result<Self> {
        let microphone = CpalMicrophone::new(
            config.microphone_name.as_deref(),
            config.audio_sample_rate_hz.max(8_000),
            config.audio_channels.max(1),
        )
        .ok();
        let camera = config
            .camera_device
            .as_deref()
            .map(V4lCamera::new)
            .transpose()?;
        let gps = config
            .gps_serial_port
            .as_deref()
            .map(|port| Ublox7Gps::new(port, config.gps_baud_rate.max(9_600)))
            .transpose()?;
        Ok(Self {
            snapshot: WorldSnapshot::default(),
            microphone,
            camera,
            gps,
        })
    }

    pub fn snapshot_ref(&self) -> &WorldSnapshot {
        &self.snapshot
    }

    fn refresh_hardware(&mut self) -> Result<()> {
        if let Some(camera) = self.camera.as_mut() {
            if let Ok(frame) = camera.capture_frame() {
                self.snapshot.eye.frames = vec![bytes_to_unit_signal(&frame.bytes)];
                self.snapshot.eye_frame = Some(frame);
            }
        }

        if let Some(microphone) = self.microphone.as_ref() {
            if let Some(frame) = microphone.latest_frame() {
                self.snapshot.ear.features = vec![pcm_to_unit_signal(&frame.samples)];
                self.snapshot.ear_pcm = Some(frame);
            }
        }

        if let Some(gps) = self.gps.as_mut() {
            if let Some(fix) = gps.try_read_fix()? {
                self.snapshot.gps = Some(fix);
            }
        }

        Ok(())
    }
}

#[cfg(feature = "linux-hardware")]
#[async_trait]
impl World for LinuxWorld {
    async fn snapshot(&mut self) -> Result<WorldSnapshot> {
        self.refresh_hardware()?;
        Ok(self.snapshot.clone())
    }

    async fn apply_update(&mut self, update: WorldUpdate) -> Result<()> {
        update.apply_to(&mut self.snapshot);
        Ok(())
    }
}

#[cfg(feature = "linux-hardware")]
pub struct CpalMicrophone {
    latest: Arc<Mutex<Option<PcmAudioFrame>>>,
    _stream: cpal::Stream,
}

#[cfg(feature = "linux-hardware")]
unsafe impl Send for CpalMicrophone {}
#[cfg(feature = "linux-hardware")]
unsafe impl Sync for CpalMicrophone {}

#[cfg(feature = "linux-hardware")]
impl CpalMicrophone {
    pub fn new(preferred_name: Option<&str>, sample_rate_hz: u32, channels: u16) -> Result<Self> {
        let host = cpal::default_host();
        let device = select_input_device(&host, preferred_name)?;
        let config = cpal::StreamConfig {
            channels,
            sample_rate: cpal::SampleRate(sample_rate_hz),
            buffer_size: cpal::BufferSize::Default,
        };
        let latest = Arc::new(Mutex::new(None));
        let shared = Arc::clone(&latest);
        let err_fn = |err| eprintln!("cpal input stream error: {err}");
        let sample_format = device.default_input_config()?.sample_format();
        let stream = match sample_format {
            cpal::SampleFormat::I16 => device.build_input_stream(
                &config,
                move |data: &[i16], _| store_i16_pcm_frame(&shared, data, sample_rate_hz, channels),
                err_fn,
                None,
            )?,
            cpal::SampleFormat::U16 => device.build_input_stream(
                &config,
                move |data: &[u16], _| {
                    let pcm = data
                        .iter()
                        .map(|sample| (*sample as i32 - 32_768) as i16)
                        .collect::<Vec<_>>();
                    store_i16_pcm_frame(&shared, &pcm, sample_rate_hz, channels);
                },
                err_fn,
                None,
            )?,
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config,
                move |data: &[f32], _| {
                    let pcm = data
                        .iter()
                        .map(|sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
                        .collect::<Vec<_>>();
                    store_i16_pcm_frame(&shared, &pcm, sample_rate_hz, channels);
                },
                err_fn,
                None,
            )?,
            other => {
                anyhow::bail!("unsupported CPAL sample format: {:?}", other);
            }
        };
        stream.play()?;
        Ok(Self {
            latest,
            _stream: stream,
        })
    }

    pub fn latest_frame(&self) -> Option<PcmAudioFrame> {
        self.latest.lock().ok().and_then(|guard| guard.clone())
    }
}

#[cfg(feature = "linux-hardware")]
pub struct V4lCamera {
    stream: MmapStream<'static>,
    format: Format,
}

#[cfg(feature = "linux-hardware")]
impl V4lCamera {
    pub fn new(path: &str) -> Result<Self> {
        let mut device = Device::with_path(path)?;
        configure_camera_format(&mut device);
        let format = device.format()?;
        let device = Box::leak(Box::new(device));
        let stream = MmapStream::with_buffers(device, Type::VideoCapture, 4)?;
        Ok(Self { stream, format })
    }

    pub fn capture_frame(&mut self) -> Result<EyeFrame> {
        let (bytes, _) = self.stream.next()?;
        Ok(EyeFrame {
            captured_at_ms: unix_time_ms(),
            width: self.format.width,
            height: self.format.height,
            format: eye_frame_format_from_fourcc(self.format.fourcc.str().unwrap_or_default()),
            bytes: bytes.to_vec(),
            source: Some("real-camera".to_string()),
        })
    }
}

#[cfg(feature = "linux-hardware")]
fn eye_frame_format_from_fourcc(fourcc: &str) -> EyeFrameFormat {
    match fourcc.trim_end_matches('\0') {
        "GREY" | "Y800" => EyeFrameFormat::Gray8,
        "RGB3" => EyeFrameFormat::Rgb8,
        "BGR3" => EyeFrameFormat::Bgr8,
        "YUYV" | "YUY2" => EyeFrameFormat::Yuyv422,
        "UYVY" => EyeFrameFormat::Uyvy422,
        "GRBG" => EyeFrameFormat::BayerGrbg8,
        "RGGB" => EyeFrameFormat::BayerRggb8,
        "BGGR" => EyeFrameFormat::BayerBggr8,
        "GBRG" => EyeFrameFormat::BayerGbrg8,
        "MJPG" | "JPEG" => EyeFrameFormat::Mjpeg,
        other => EyeFrameFormat::Unknown(other.to_string()),
    }
}

#[cfg(feature = "linux-hardware")]
fn configure_camera_format(device: &mut Device) {
    let candidates = [
        (320, 240, *b"MJPG"),
        (320, 240, *b"YUYV"),
        (640, 480, *b"MJPG"),
        (640, 480, *b"RGB3"),
        (640, 480, *b"BGR3"),
        (640, 480, *b"GREY"),
    ];
    for (width, height, fourcc) in candidates {
        let format = Format::new(width, height, FourCC::new(&fourcc));
        if device.set_format(&format).is_ok() {
            return;
        }
    }
}

#[cfg(feature = "linux-hardware")]
pub struct Ublox7Gps {
    port: Box<dyn SerialPort>,
    buffer: Vec<u8>,
}

#[cfg(feature = "linux-hardware")]
impl Ublox7Gps {
    pub fn new(path: &str, baud_rate: u32) -> Result<Self> {
        let port = serialport::new(path, baud_rate)
            .timeout(Duration::from_millis(25))
            .open()?;
        Ok(Self {
            port,
            buffer: Vec::new(),
        })
    }

    pub fn try_read_fix(&mut self) -> Result<Option<GpsSense>> {
        let mut chunk = [0u8; 512];
        match self.port.read(&mut chunk) {
            Ok(count) => self.buffer.extend_from_slice(&chunk[..count]),
            Err(error) if error.kind() == ErrorKind::TimedOut => {}
            Err(error) => return Err(error.into()),
        }

        while let Some(position) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let line = self.buffer.drain(..=position).collect::<Vec<_>>();
            if let Ok(text) = std::str::from_utf8(&line) {
                if let Some(fix) = parse_nmea_fix(text.trim()) {
                    return Ok(Some(fix));
                }
            }
        }

        Ok(None)
    }
}

#[cfg(feature = "linux-hardware")]
pub struct Mpu6050Imu {
    bus: File,
}

#[cfg(feature = "linux-hardware")]
impl Mpu6050Imu {
    pub fn new(device: &str) -> Result<Self> {
        let spec = parse_mpu6050_device_spec(device)?;
        let bus = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&spec.path)?;
        set_i2c_slave(&bus, spec.address)?;
        let mut imu = Self { bus };
        imu.write_register(MPU6050_PWR_MGMT_1, 0x00)?;
        imu.write_register(MPU6050_ACCEL_CONFIG, 0x00)?;
        imu.write_register(MPU6050_GYRO_CONFIG, 0x00)?;
        Ok(imu)
    }

    pub fn read_sense(&mut self) -> Result<ImuSense> {
        let mut bytes = [0u8; 14];
        self.read_registers(MPU6050_ACCEL_XOUT_H, &mut bytes)?;
        Ok(mpu6050_samples_to_imu(bytes))
    }

    fn write_register(&mut self, register: u8, value: u8) -> Result<()> {
        self.bus.write_all(&[register, value])?;
        Ok(())
    }

    fn read_registers(&mut self, register: u8, buffer: &mut [u8]) -> Result<()> {
        self.bus.write_all(&[register])?;
        self.bus.read_exact(buffer)?;
        Ok(())
    }
}

#[cfg(feature = "linux-hardware")]
fn set_i2c_slave(bus: &File, address: u16) -> Result<()> {
    const I2C_SLAVE: libc::c_ulong = 0x0703;
    let result = unsafe { libc::ioctl(bus.as_raw_fd(), I2C_SLAVE, libc::c_ulong::from(address)) };
    if result < 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("failed to select I2C slave address 0x{address:02x}"));
    }
    Ok(())
}

#[cfg(any(feature = "linux-hardware", test))]
#[derive(Clone, Debug, PartialEq, Eq)]
struct Mpu6050DeviceSpec {
    path: String,
    address: u16,
}

#[cfg(any(feature = "linux-hardware", test))]
const MPU6050_DEFAULT_ADDRESS: u16 = 0x68;
#[cfg(feature = "linux-hardware")]
const MPU6050_PWR_MGMT_1: u8 = 0x6b;
#[cfg(feature = "linux-hardware")]
const MPU6050_ACCEL_CONFIG: u8 = 0x1c;
#[cfg(feature = "linux-hardware")]
const MPU6050_GYRO_CONFIG: u8 = 0x1b;
#[cfg(feature = "linux-hardware")]
const MPU6050_ACCEL_XOUT_H: u8 = 0x3b;

#[cfg(any(feature = "linux-hardware", test))]
fn parse_mpu6050_device_spec(device: &str) -> Result<Mpu6050DeviceSpec> {
    let (path, address) = device
        .rsplit_once('@')
        .or_else(|| device.rsplit_once(':'))
        .map(|(path, address)| (path, Some(address)))
        .unwrap_or((device, None));
    let address = address
        .map(parse_i2c_address)
        .transpose()?
        .unwrap_or(MPU6050_DEFAULT_ADDRESS);
    if path.trim().is_empty() {
        anyhow::bail!("MPU-6050 I2C device path is empty");
    }
    if !(0x03..=0x77).contains(&address) {
        anyhow::bail!("I2C address 0x{address:02x} is outside the 7-bit usable range");
    }
    Ok(Mpu6050DeviceSpec {
        path: path.to_string(),
        address,
    })
}

#[cfg(any(feature = "linux-hardware", test))]
fn parse_i2c_address(value: &str) -> Result<u16> {
    let trimmed = value.trim();
    let digits = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"));
    match digits {
        Some(hex) => u16::from_str_radix(hex, 16).map_err(anyhow::Error::from),
        None => trimmed.parse::<u16>().map_err(anyhow::Error::from),
    }
    .with_context(|| format!("invalid I2C address `{value}`"))
}

#[cfg(any(feature = "linux-hardware", test))]
fn mpu6050_samples_to_imu(bytes: [u8; 14]) -> ImuSense {
    let accel_x = read_i16_be(bytes[0], bytes[1]) as f32 / 16_384.0;
    let accel_y = read_i16_be(bytes[2], bytes[3]) as f32 / 16_384.0;
    let accel_z = read_i16_be(bytes[4], bytes[5]) as f32 / 16_384.0;
    let gyro_x = read_i16_be(bytes[8], bytes[9]) as f32 / 131.0;
    let gyro_y = read_i16_be(bytes[10], bytes[11]) as f32 / 131.0;
    let gyro_z = read_i16_be(bytes[12], bytes[13]) as f32 / 131.0;

    let roll_rad = accel_y.atan2(accel_z);
    let pitch_rad = (-accel_x).atan2((accel_y * accel_y + accel_z * accel_z).sqrt());

    ImuSense {
        schema_version: 1,
        orientation: vec![roll_rad, pitch_rad],
        acceleration: vec![accel_x, accel_y, accel_z],
        angular_velocity: vec![gyro_x, gyro_y, gyro_z],
    }
}

#[cfg(any(feature = "linux-hardware", test))]
fn read_i16_be(high: u8, low: u8) -> i16 {
    i16::from_be_bytes([high, low])
}

#[cfg(feature = "linux-hardware")]
fn select_input_device(host: &cpal::Host, preferred_name: Option<&str>) -> Result<cpal::Device> {
    if let Some(name) = preferred_name {
        for device in host.input_devices()? {
            if device.name().ok().as_deref() == Some(name) {
                return Ok(device);
            }
        }
    }
    host.default_input_device()
        .ok_or_else(|| anyhow::anyhow!("no CPAL input device available"))
}

#[cfg(feature = "linux-hardware")]
fn store_i16_pcm_frame(
    shared: &Arc<Mutex<Option<PcmAudioFrame>>>,
    samples: &[i16],
    sample_rate_hz: u32,
    channels: u16,
) {
    if let Ok(mut guard) = shared.lock() {
        *guard = Some(PcmAudioFrame {
            captured_at_ms: unix_time_ms(),
            sample_rate_hz,
            channels,
            samples: samples.to_vec(),
        });
    }
}

fn bytes_to_unit_signal(bytes: &[u8]) -> Vec<f32> {
    bytes
        .iter()
        .take(256)
        .map(|byte| *byte as f32 / 255.0)
        .collect()
}

fn pcm_to_unit_signal(samples: &[i16]) -> Vec<f32> {
    samples
        .iter()
        .take(256)
        .map(|sample| *sample as f32 / i16::MAX as f32)
        .collect()
}

#[cfg(feature = "linux-hardware")]
fn parse_nmea_fix(line: &str) -> Option<GpsSense> {
    if line.starts_with("$GPGGA") || line.starts_with("$GNGGA") {
        let fields = line.split(',').collect::<Vec<_>>();
        if fields.len() < 10 {
            return None;
        }
        let lat = parse_nmea_coord(fields[2], fields[3])?;
        let lon = parse_nmea_coord(fields[4], fields[5])?;
        let altitude_m = fields[9].parse::<f32>().ok();
        return Some(GpsSense {
            schema_version: 1,
            lat,
            lon,
            altitude_m,
        });
    }
    if line.starts_with("$GPRMC") || line.starts_with("$GNRMC") {
        let fields = line.split(',').collect::<Vec<_>>();
        if fields.len() < 7 || fields[2] != "A" {
            return None;
        }
        let lat = parse_nmea_coord(fields[3], fields[4])?;
        let lon = parse_nmea_coord(fields[5], fields[6])?;
        return Some(GpsSense {
            schema_version: 1,
            lat,
            lon,
            altitude_m: None,
        });
    }
    None
}

#[cfg(feature = "linux-hardware")]
fn parse_nmea_coord(value: &str, hemi: &str) -> Option<f64> {
    let dot = value.find('.')?;
    let degrees_len = if dot > 4 { 3 } else { 2 };
    let (degrees, minutes) = value.split_at(degrees_len);
    let degrees = degrees.parse::<f64>().ok()?;
    let minutes = minutes.parse::<f64>().ok()?;
    let mut decimal = degrees + minutes / 60.0;
    if matches!(hemi, "S" | "W") {
        decimal = -decimal;
    }
    Some(decimal)
}

#[cfg(any(feature = "linux-hardware", test))]
fn unix_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct KinectReplayFrame {
    pub t_ms: u64,
    pub rgb_path: Option<String>,
    pub depth_path: Option<String>,
    pub color_features: Option<Vec<Vec<f32>>>,
    pub depth_m: Option<Vec<f32>>,
    pub audio_angle_rad: Option<f32>,
    pub audio_confidence: Option<f32>,
}

pub struct KinectReplayProvider {
    root: PathBuf,
    frames: Vec<KinectReplayFrame>,
    cursor: usize,
    pending: VecDeque<SensePacket>,
}

impl KinectReplayProvider {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        let manifest_path = root.join("timestamps.jsonl");
        let manifest = File::open(&manifest_path)?;
        let frames = BufReader::new(manifest)
            .lines()
            .filter_map(|line| match line {
                Ok(line) if !line.trim().is_empty() => {
                    Some(serde_json::from_str(&line).map_err(anyhow::Error::from))
                }
                Ok(_) => None,
                Err(error) => Some(Err(error.into())),
            })
            .collect::<Result<Vec<KinectReplayFrame>>>()?;
        Ok(Self {
            root,
            frames,
            cursor: 0,
            pending: VecDeque::new(),
        })
    }

    fn packet_for_frame(
        &self,
        frame: &KinectReplayFrame,
    ) -> Result<(SensePacket, Option<SensePacket>)> {
        let rgb_bytes = frame
            .rgb_path
            .as_ref()
            .map(|path| std::fs::read(self.root.join(path)))
            .transpose()?;
        let depth_m = match &frame.depth_m {
            Some(depth) => depth.clone(),
            None => frame
                .depth_path
                .as_ref()
                .map(|path| read_depth_values(&self.root.join(path)))
                .transpose()?
                .unwrap_or_default(),
        };
        let color_features = frame
            .color_features
            .clone()
            .or_else(|| {
                rgb_bytes
                    .as_deref()
                    .map(|bytes| vec![bytes_to_unit_signal(bytes)])
            })
            .unwrap_or_default();
        let eye = rgb_bytes.map(|bytes| {
            SensePacket::Eye(EyeSense {
                schema_version: 1,
                frames: vec![bytes_to_unit_signal(&bytes)],
                image_vectors: Vec::new(),
                image_description_vectors: Vec::new(),
                scene_vectors: Vec::new(),
            })
        });
        let kinect = KinectSense {
            schema_version: 1,
            color_features,
            depth_m,
            audio_angle_rad: frame.audio_angle_rad,
            audio_confidence: frame.audio_confidence.unwrap_or(0.0),
            ..KinectSense::default()
        };
        Ok((SensePacket::Kinect(kinect), eye))
    }
}

#[async_trait]
impl SenseProducer for KinectReplayProvider {
    async fn poll(&mut self) -> Result<SensePacket> {
        if let Some(packet) = self.pending.pop_front() {
            return Ok(packet);
        }
        if self.frames.is_empty() {
            anyhow::bail!("kinect replay has no frames");
        }
        let frame = &self.frames[self.cursor % self.frames.len()];
        self.cursor += 1;
        let (kinect, eye) = self.packet_for_frame(frame)?;
        if let Some(eye) = eye {
            self.pending.push_back(eye);
        }
        Ok(kinect)
    }
}

#[cfg(feature = "kinect-freenect")]
pub struct FreenectKinectProvider;

#[cfg(feature = "kinect-freenect")]
impl FreenectKinectProvider {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }
}

#[cfg(feature = "kinect-freenect")]
#[async_trait]
impl SenseProducer for FreenectKinectProvider {
    async fn poll(&mut self) -> Result<SensePacket> {
        anyhow::bail!(
            "FreenectKinectProvider is a feature-gated skeleton; wire libfreenect FFI or a freenect subprocess here"
        )
    }
}

fn read_depth_values(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if let Ok(values) = serde_json::from_slice::<Vec<f32>>(&bytes) {
        return Ok(values);
    }
    let text = String::from_utf8(bytes)?;
    text.split_whitespace()
        .map(|value| value.parse::<f32>().map_err(anyhow::Error::from))
        .collect()
}

pub struct CameraSenseProvider {
    #[cfg(feature = "linux-hardware")]
    camera: V4lCamera,
}

impl CameraSenseProvider {
    pub fn new(device: &str) -> Result<Self> {
        #[cfg(feature = "linux-hardware")]
        {
            Ok(Self {
                camera: V4lCamera::new(device)?,
            })
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            let _ = device;
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

#[async_trait]
impl SenseProducer for CameraSenseProvider {
    async fn poll(&mut self) -> Result<SensePacket> {
        #[cfg(feature = "linux-hardware")]
        {
            let frame = self.camera.capture_frame()?;
            Ok(SensePacket::EyeFrame(frame))
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

pub struct MicrophoneSenseProvider {
    #[cfg(feature = "linux-hardware")]
    microphone: CpalMicrophone,
}

impl MicrophoneSenseProvider {
    pub fn new(preferred_name: Option<&str>) -> Result<Self> {
        #[cfg(feature = "linux-hardware")]
        {
            Ok(Self {
                microphone: CpalMicrophone::new(preferred_name, 16000, 1)?,
            })
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            let _ = preferred_name;
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

#[async_trait]
impl SenseProducer for MicrophoneSenseProvider {
    async fn poll(&mut self) -> Result<SensePacket> {
        #[cfg(feature = "linux-hardware")]
        {
            let frame = self
                .microphone
                .latest_frame()
                .unwrap_or_else(|| PcmAudioFrame {
                    captured_at_ms: unix_time_ms(),
                    sample_rate_hz: 16000,
                    channels: 1,
                    samples: Vec::new(),
                });
            Ok(SensePacket::EarPcm(frame))
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

pub struct GpsSenseProvider {
    #[cfg(feature = "linux-hardware")]
    gps: Ublox7Gps,
    #[cfg(feature = "linux-hardware")]
    last_fix: GpsSense,
}

impl GpsSenseProvider {
    pub fn new(port: &str, baud_rate: u32) -> Result<Self> {
        #[cfg(feature = "linux-hardware")]
        {
            Ok(Self {
                gps: Ublox7Gps::new(port, baud_rate)?,
                last_fix: GpsSense::default(),
            })
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            let _ = port;
            let _ = baud_rate;
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

#[async_trait]
impl SenseProducer for GpsSenseProvider {
    async fn poll(&mut self) -> Result<SensePacket> {
        #[cfg(feature = "linux-hardware")]
        {
            if let Some(fix) = self.gps.try_read_fix()? {
                self.last_fix = fix;
            }
            Ok(SensePacket::Gps(self.last_fix.clone()))
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

pub struct ImuSenseProvider {
    #[cfg(feature = "linux-hardware")]
    imu: Mpu6050Imu,
}

impl ImuSenseProvider {
    pub fn new(device: &str) -> Result<Self> {
        #[cfg(feature = "linux-hardware")]
        {
            Ok(Self {
                imu: Mpu6050Imu::new(device)?,
            })
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            let _ = device;
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

#[async_trait]
impl SenseProducer for ImuSenseProvider {
    async fn poll(&mut self) -> Result<SensePacket> {
        #[cfg(feature = "linux-hardware")]
        {
            Ok(SensePacket::Imu(self.imu.read_sense()?))
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[tokio::test]
    async fn kinect_replay_emits_kinect_then_eye_packet() {
        let root =
            std::env::temp_dir().join(format!("netherwick-kinect-replay-{}", unix_time_ms()));
        std::fs::create_dir_all(root.join("rgb")).unwrap();
        std::fs::create_dir_all(root.join("depth")).unwrap();
        std::fs::write(root.join("rgb/frame.raw"), [0u8, 128, 255]).unwrap();
        std::fs::write(root.join("depth/frame.json"), "[1.0,2.0]").unwrap();
        let mut manifest = File::create(root.join("timestamps.jsonl")).unwrap();
        writeln!(
            manifest,
            "{}",
            serde_json::json!({
                "t_ms": 1,
                "rgb_path": "rgb/frame.raw",
                "depth_path": "depth/frame.json"
            })
        )
        .unwrap();

        let mut provider = KinectReplayProvider::new(&root).unwrap();
        let first = provider.poll().await.unwrap();
        let second = provider.poll().await.unwrap();

        assert!(
            matches!(first, SensePacket::Kinect(KinectSense { depth_m, .. }) if depth_m == vec![1.0, 2.0])
        );
        assert!(matches!(second, SensePacket::Eye(EyeSense { frames, .. }) if frames.len() == 1));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn now_builder_maps_packets_and_marks_stale_sensor_ages() {
        let mut builder = NowBuilder::new();
        let first = builder
            .build(
                100,
                BodySense::default(),
                vec![
                    SensePacket::Ear(EarSense {
                        transcript: Some("hello".to_string()),
                        ..EarSense::default()
                    }),
                    SensePacket::Range(RangeSense {
                        beams: vec![0.4],
                        nearest_m: Some(0.4),
                        ..RangeSense::default()
                    }),
                ],
            )
            .unwrap();

        assert_eq!(first.ear.transcript.as_deref(), Some("hello"));
        assert_eq!(first.range.nearest_m, Some(0.4));

        let second = builder
            .build(250, BodySense::default(), Vec::new())
            .unwrap();
        assert_eq!(second.ear.transcript.as_deref(), Some("hello"));
        assert_eq!(second.range.nearest_m, Some(0.4));
        assert_eq!(
            second
                .extensions
                .get("sensor_status")
                .and_then(|status| status.get("age_ms"))
                .and_then(|age| age.get("ear"))
                .and_then(|age| age.as_u64()),
            Some(150)
        );
    }

    #[test]
    fn parses_mpu6050_device_specs() {
        assert_eq!(
            parse_mpu6050_device_spec("/dev/i2c-1").unwrap(),
            Mpu6050DeviceSpec {
                path: "/dev/i2c-1".to_string(),
                address: 0x68,
            }
        );
        assert_eq!(
            parse_mpu6050_device_spec("/dev/i2c-1@0x69").unwrap(),
            Mpu6050DeviceSpec {
                path: "/dev/i2c-1".to_string(),
                address: 0x69,
            }
        );
        assert_eq!(
            parse_mpu6050_device_spec("/dev/i2c-2:105").unwrap(),
            Mpu6050DeviceSpec {
                path: "/dev/i2c-2".to_string(),
                address: 0x69,
            }
        );
    }

    #[test]
    fn converts_mpu6050_raw_samples_to_imu_sense() {
        let imu = mpu6050_samples_to_imu([
            0x00, 0x00, // accel x = 0 g
            0x00, 0x00, // accel y = 0 g
            0x40, 0x00, // accel z = 1 g
            0x00, 0x00, // temperature, ignored
            0x00, 0x83, // gyro x = 1 deg/s
            0xff, 0x7d, // gyro y = -1 deg/s
            0x01, 0x06, // gyro z = 2 deg/s
        ]);

        assert_eq!(imu.schema_version, 1);
        assert_eq!(imu.acceleration, vec![0.0, 0.0, 1.0]);
        assert_eq!(imu.angular_velocity, vec![1.0, -1.0, 2.0]);
        assert_eq!(imu.orientation, vec![0.0, -0.0]);
    }
}
