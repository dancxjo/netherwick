use anyhow::Result;
use async_trait::async_trait;
use netherwick_body::BodySense;
use netherwick_now::{
    EarSense, ExtensionSense, EyeSense, FaceSense, GpsSense, ImuSense, KinectSense, RangeSense,
    VoiceSense,
};
use netherwick_now::{Now, PredictionSense, SurpriseSense};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

#[cfg(feature = "linux-hardware")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(feature = "linux-hardware")]
use serialport::SerialPort;
#[cfg(feature = "linux-hardware")]
use std::io::{ErrorKind, Read};
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
    Ear(EarSense),
    Range(RangeSense),
    Imu(ImuSense),
    Gps(GpsSense),
    Kinect(KinectSense),
    Face(FaceSense),
    Voice(VoiceSense),
    Extension(ExtensionSense),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EyeFrameFormat {
    Gray8,
    Rgb8,
    Bgr8,
    Mjpeg,
    Unknown(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EyeFrame {
    pub captured_at_ms: u64,
    pub width: u32,
    pub height: u32,
    pub format: EyeFrameFormat,
    pub bytes: Vec<u8>,
}

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
    pub eye_frame: Option<EyeFrame>,
    pub ear_pcm: Option<PcmAudioFrame>,
    pub eye: EyeSense,
    pub ear: EarSense,
    pub range: RangeSense,
    pub imu: ImuSense,
    pub gps: Option<GpsSense>,
    pub kinect: KinectSense,
    pub face: FaceSense,
    pub voice: VoiceSense,
    pub extensions: Vec<ExtensionSense>,
}

impl Default for WorldSnapshot {
    fn default() -> Self {
        Self {
            body: BodySense::default(),
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
        now.ear = self.ear.clone();
        now.face = self.face.clone();
        now.voice = self.voice.clone();
        now.range = self.range.clone();
        now.imu = self.imu.clone();
        now.gps = self.gps.clone();
        now.kinect = self.kinect.clone();
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
    device: Device,
}

#[cfg(feature = "linux-hardware")]
impl V4lCamera {
    pub fn new(path: &str) -> Result<Self> {
        Ok(Self {
            device: Device::with_path(path)?,
        })
    }

    pub fn capture_frame(&mut self) -> Result<EyeFrame> {
        let format = self.device.format()?;
        let mut stream = MmapStream::with_buffers(&self.device, Type::VideoCapture, 2)?;
        let (bytes, _) = stream.next()?;
        Ok(EyeFrame {
            captured_at_ms: unix_time_ms(),
            width: format.width,
            height: format.height,
            format: EyeFrameFormat::Unknown(format!("{:?}", format.fourcc)),
            bytes: bytes.to_vec(),
        })
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

#[cfg(feature = "linux-hardware")]
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
}
