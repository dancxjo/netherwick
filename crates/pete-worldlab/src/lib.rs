use std::fs as std_fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use chrono::Utc;
use image::{ImageBuffer, Luma, RgbImage};
use pete_experience::{ExperienceLatent, FuturePrediction};
use pete_perception::PerceptionFrame;
use pete_runtime::{RuntimeLoop, RuntimeTick};
use pete_sensors::{EyeFrameFormat, PcmAudioFrame, WorldSnapshot};
use pete_sim::ScenarioMetadata;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::fs::{self, File, OpenOptions};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio_stream::{self as stream, Stream};
use uuid::Uuid;

pub const CAPTURE_SCHEMA_VERSION: u32 = 2;
pub const CAPTURE_QUEUE_CAPACITY: usize = 4;
const ASSET_LATE_AFTER_MS: u64 = 1_000;
const MANIFEST_FILE: &str = "manifest.json";
const FRAMES_FILE: &str = "frames.jsonl";
const EVENTS_FILE: &str = "events.jsonl";

pub type SerializableWorldSnapshot = WorldSnapshot;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaptureManifest {
    pub id: String,
    pub created_at_ms: u64,
    pub source: CaptureSource,
    pub schema_version: u32,
    pub frame_count: u64,
    pub tick_ms: Option<u64>,
    pub notes: Vec<String>,
    #[serde(default)]
    pub machine: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub firmware_identity: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brainstem_safety: Option<Value>,
    #[serde(default)]
    pub command_args: Vec<String>,
    #[serde(default)]
    pub device_availability: Value,
    #[serde(default)]
    pub streams: CaptureStreams,
    #[serde(default)]
    pub started_at_ms: Option<u64>,
    #[serde(default)]
    pub ended_at_ms: Option<u64>,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default = "default_asset_layout")]
    pub asset_layout: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scenario: Option<ScenarioMetadata>,
    #[serde(default)]
    pub writer_health: CaptureWriterHealth,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CaptureSource {
    Sim,
    RealRobot,
    Replay,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaptureFrameRecord {
    pub index: u64,
    pub t_ms: u64,
    pub snapshot: SerializableWorldSnapshot,
    pub events: Vec<RecordedEvent>,
    pub assets: CaptureFrameAssets,
    pub stream_metadata: Option<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecordedEvent {
    pub t_ms: u64,
    pub kind: String,
    pub payload: Value,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureStreams {
    pub present: Vec<String>,
    pub missing: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureFrameAssets {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgb: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pointcloud: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub perception: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lidar: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imu: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<String>,
}

impl CaptureFrameAssets {
    pub fn is_empty(&self) -> bool {
        self.rgb.is_none()
            && self.depth.is_none()
            && self.audio.is_none()
            && self.pointcloud.is_none()
            && self.perception.is_none()
            && self.camera.is_none()
            && self.lidar.is_none()
            && self.imu.is_none()
            && self.transcript.is_none()
            && self.calibration.is_none()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureWriterHealth {
    pub queue_capacity: usize,
    pub submitted_frames: u64,
    pub written_frames: u64,
    pub dropped_frames: u64,
    #[serde(default)]
    pub dropped_assets: std::collections::BTreeMap<String, u64>,
    #[serde(default)]
    pub write_failures: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct CaptureExportContext {
    pub imu_selection: Option<Value>,
}

struct QueuedCaptureFrame {
    index: u64,
    t_ms: u64,
    snapshot: WorldSnapshot,
    events: Vec<RecordedEvent>,
    export_rgb: bool,
    export_depth: bool,
    export_audio: bool,
    context: CaptureExportContext,
}

enum BackgroundCommand {
    Frame(QueuedCaptureFrame),
    Finish,
}

struct BackgroundCaptureWriter {
    sender: SyncSender<BackgroundCommand>,
    worker: JoinHandle<Result<CaptureWriterHealth>>,
}

pub struct CaptureWriter {
    root: PathBuf,
    manifest: CaptureManifest,
    frames: Option<File>,
    frame_count: u64,
    background: Option<BackgroundCaptureWriter>,
    writer_health: CaptureWriterHealth,
    #[cfg(test)]
    background_write_delay: std::time::Duration,
}

impl CaptureWriter {
    pub async fn create(
        root: impl AsRef<Path>,
        source: CaptureSource,
        tick_ms: Option<u64>,
    ) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("assets").join("rgb")).await?;
        fs::create_dir_all(root.join("assets").join("depth")).await?;
        fs::create_dir_all(root.join("assets").join("audio")).await?;
        fs::create_dir_all(root.join("assets").join("pointcloud")).await?;
        fs::create_dir_all(root.join("assets").join("camera")).await?;
        fs::create_dir_all(root.join("assets").join("lidar")).await?;
        fs::create_dir_all(root.join("assets").join("imu")).await?;
        fs::create_dir_all(root.join("assets").join("transcript")).await?;
        fs::create_dir_all(root.join("assets").join("calibration")).await?;
        File::create(root.join(EVENTS_FILE)).await?;

        let now_ms = Utc::now().timestamp_millis().max(0) as u64;
        let manifest = CaptureManifest {
            id: capture_id_from_path(&root),
            created_at_ms: now_ms,
            source,
            schema_version: CAPTURE_SCHEMA_VERSION,
            frame_count: 0,
            tick_ms,
            notes: Vec::new(),
            machine: None,
            firmware_identity: None,
            brainstem_safety: None,
            command_args: Vec::new(),
            device_availability: Value::Null,
            streams: CaptureStreams::default(),
            started_at_ms: Some(now_ms),
            ended_at_ms: None,
            warnings: Vec::new(),
            asset_layout: default_asset_layout(),
            scenario: None,
            writer_health: CaptureWriterHealth {
                queue_capacity: CAPTURE_QUEUE_CAPACITY,
                ..CaptureWriterHealth::default()
            },
        };
        write_manifest_atomic(&root, &manifest).await?;

        let frames = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(root.join(FRAMES_FILE))
            .await?;

        Ok(Self {
            root,
            manifest,
            frames: Some(frames),
            frame_count: 0,
            background: None,
            writer_health: CaptureWriterHealth {
                queue_capacity: CAPTURE_QUEUE_CAPACITY,
                ..CaptureWriterHealth::default()
            },
            #[cfg(test)]
            background_write_delay: std::time::Duration::ZERO,
        })
    }

    pub async fn append_snapshot(
        &mut self,
        t_ms: u64,
        snapshot: WorldSnapshot,
        events: Vec<RecordedEvent>,
    ) -> Result<()> {
        self.append_snapshot_with_assets(
            t_ms,
            snapshot,
            events,
            CaptureFrameAssets::default(),
            None,
        )
        .await
    }

    pub async fn append_snapshot_with_assets(
        &mut self,
        t_ms: u64,
        snapshot: WorldSnapshot,
        events: Vec<RecordedEvent>,
        assets: CaptureFrameAssets,
        stream_metadata: Option<Value>,
    ) -> Result<()> {
        let record = CaptureFrameRecord {
            index: self.frame_count,
            t_ms,
            snapshot,
            events,
            assets,
            stream_metadata,
        };
        let line = serde_json::to_string(&record)?;
        let frames = self
            .frames
            .as_mut()
            .context("direct capture append attempted after background writer started")?;
        frames.write_all(line.as_bytes()).await?;
        frames.write_all(b"\n").await?;
        self.frame_count = self.frame_count.saturating_add(1);
        self.writer_health.submitted_frames = self.writer_health.submitted_frames.saturating_add(1);
        self.writer_health.written_frames = self.writer_health.written_frames.saturating_add(1);
        Ok(())
    }

    pub async fn append_snapshot_with_exported_assets(
        &mut self,
        t_ms: u64,
        snapshot: WorldSnapshot,
        events: Vec<RecordedEvent>,
        export_rgb: bool,
        export_depth: bool,
        export_audio: bool,
    ) -> Result<()> {
        self.append_snapshot_with_exported_assets_and_context(
            t_ms,
            snapshot,
            events,
            export_rgb,
            export_depth,
            export_audio,
            CaptureExportContext::default(),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn append_snapshot_with_exported_assets_and_context(
        &mut self,
        t_ms: u64,
        snapshot: WorldSnapshot,
        events: Vec<RecordedEvent>,
        export_rgb: bool,
        export_depth: bool,
        export_audio: bool,
        context: CaptureExportContext,
    ) -> Result<()> {
        self.ensure_background_writer().await?;
        let index = self.frame_count;
        self.frame_count = self.frame_count.saturating_add(1);
        self.writer_health.submitted_frames = self.writer_health.submitted_frames.saturating_add(1);
        let frame = QueuedCaptureFrame {
            index,
            t_ms,
            snapshot,
            events,
            export_rgb,
            export_depth,
            export_audio,
            context,
        };
        let background = self
            .background
            .as_ref()
            .context("background capture writer missing")?;
        match background.sender.try_send(BackgroundCommand::Frame(frame)) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(BackgroundCommand::Frame(frame))) => {
                self.writer_health.dropped_frames =
                    self.writer_health.dropped_frames.saturating_add(1);
                for kind in requested_asset_kinds(&frame) {
                    *self.writer_health.dropped_assets.entry(kind).or_default() += 1;
                }
                Ok(())
            }
            Err(TrySendError::Disconnected(_)) => {
                anyhow::bail!("background capture writer stopped unexpectedly")
            }
            Err(TrySendError::Full(BackgroundCommand::Finish)) => unreachable!(),
        }
    }

    async fn ensure_background_writer(&mut self) -> Result<()> {
        if self.background.is_some() {
            return Ok(());
        }
        let frames = self
            .frames
            .take()
            .context("capture frame file is unavailable")?;
        let mut frames = frames.into_std().await;
        let root = self.root.clone();
        #[cfg(test)]
        let background_write_delay = self.background_write_delay;
        let (sender, receiver) = mpsc::sync_channel(CAPTURE_QUEUE_CAPACITY);
        let worker = std::thread::Builder::new()
            .name("pete-capture-writer".to_string())
            .spawn(move || {
                let mut health = CaptureWriterHealth {
                    queue_capacity: CAPTURE_QUEUE_CAPACITY,
                    ..CaptureWriterHealth::default()
                };
                while let Ok(command) = receiver.recv() {
                    match command {
                        BackgroundCommand::Frame(frame) => {
                            #[cfg(test)]
                            std::thread::sleep(background_write_delay);
                            match write_queued_capture_frame(&root, &mut frames, frame) {
                                Ok(()) => {
                                    health.written_frames = health.written_frames.saturating_add(1)
                                }
                                Err(error) => {
                                    health.write_failures.push(format!("{error:#}"));
                                    return Err(error);
                                }
                            }
                        }
                        BackgroundCommand::Finish => break,
                    }
                }
                frames.flush()?;
                Ok(health)
            })?;
        self.background = Some(BackgroundCaptureWriter { sender, worker });
        Ok(())
    }

    pub async fn finish(mut self) -> Result<CaptureManifest> {
        if let Some(background) = self.background.take() {
            background
                .sender
                .send(BackgroundCommand::Finish)
                .map_err(|_| anyhow::anyhow!("background capture writer stopped before finish"))?;
            let worker_health = tokio::task::spawn_blocking(move || background.worker.join())
                .await
                .context("joining background capture writer task")?
                .map_err(|_| anyhow::anyhow!("background capture writer panicked"))??;
            self.writer_health.written_frames = worker_health.written_frames;
            self.writer_health
                .write_failures
                .extend(worker_health.write_failures);
        } else if let Some(frames) = self.frames.as_mut() {
            frames.flush().await?;
        }
        self.manifest.frame_count = self.writer_health.written_frames;
        self.manifest.writer_health = self.writer_health.clone();
        if self.writer_health.dropped_frames > 0 {
            self.manifest.warnings.push(format!(
                "capture writer queue dropped {} frame(s) and their requested assets: {:?}",
                self.writer_health.dropped_frames, self.writer_health.dropped_assets
            ));
        }
        self.manifest.ended_at_ms = Some(Utc::now().timestamp_millis().max(0) as u64);
        write_manifest_atomic(&self.root, &self.manifest).await?;
        Ok(self.manifest)
    }

    pub fn manifest_mut(&mut self) -> &mut CaptureManifest {
        &mut self.manifest
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    #[cfg(test)]
    fn set_background_write_delay(&mut self, delay: std::time::Duration) {
        self.background_write_delay = delay;
    }
}

fn requested_asset_kinds(frame: &QueuedCaptureFrame) -> Vec<String> {
    let mut kinds = vec![
        "calibration".to_string(),
        "lidar".to_string(),
        "imu".to_string(),
    ];
    if frame.export_rgb {
        kinds.extend(["rgb".to_string(), "camera".to_string()]);
    }
    if frame.export_depth {
        kinds.extend(["depth".to_string(), "pointcloud".to_string()]);
    }
    if frame.export_audio {
        kinds.extend(["audio".to_string(), "transcript".to_string()]);
    }
    kinds
}

fn write_queued_capture_frame(
    root: &Path,
    frames: &mut std_fs::File,
    frame: QueuedCaptureFrame,
) -> Result<()> {
    let export = export_snapshot_assets_with_context(
        root,
        frame.index,
        frame.t_ms,
        &frame.snapshot,
        frame.export_rgb,
        frame.export_depth,
        frame.export_audio,
        &frame.context,
    )?;
    let metadata = export
        .metadata
        .as_object()
        .is_some_and(|metadata| !metadata.is_empty())
        .then_some(export.metadata);
    let mut snapshot = frame.snapshot;
    strip_exported_payloads(&mut snapshot, &export.assets);
    let record = CaptureFrameRecord {
        index: frame.index,
        t_ms: frame.t_ms,
        snapshot,
        events: frame.events,
        assets: export.assets,
        stream_metadata: metadata,
    };
    serde_json::to_writer(&mut *frames, &record)?;
    frames.write_all(b"\n")?;
    Ok(())
}

fn strip_exported_payloads(snapshot: &mut WorldSnapshot, assets: &CaptureFrameAssets) {
    if assets.camera.is_some() || assets.rgb.is_some() {
        if let Some(frame) = snapshot.eye_frame.as_mut() {
            frame.bytes.clear();
        }
        if let Some(frame) = snapshot.kinect.color_frame.as_mut() {
            frame.bytes.clear();
        }
    }
    if assets.depth.is_some() {
        snapshot.kinect.depth_m.clear();
    }
    if assets.audio.is_some() {
        if let Some(audio) = snapshot.ear_pcm.as_mut() {
            audio.samples.clear();
        }
    }
    if assets.lidar.is_some() {
        snapshot.range.beams.clear();
        snapshot.range.beam_angles_rad.clear();
        snapshot.range.beam_time_offsets_ms.clear();
        snapshot.range.beam_poses.clear();
    }
}

/// Restore losslessly externalized sensor payloads for replay and offline
/// derivation. Compact frame JSON remains useful on its own, while readers get
/// the same raw arrays that were present at capture time.
pub fn hydrate_frame_assets(root: &Path, frame: &mut CaptureFrameRecord) -> Result<()> {
    if let Some(rel) = frame.assets.depth.as_deref() {
        let path = root.join(rel);
        if path.exists() {
            let image = image::open(&path)
                .with_context(|| format!("reading depth asset {}", path.display()))?
                .to_luma16();
            frame.snapshot.kinect.depth_width = image.width();
            frame.snapshot.kinect.depth_height = image.height();
            frame.snapshot.kinect.depth_m = image
                .into_raw()
                .into_iter()
                .map(|millimeters| millimeters as f32 * 0.001)
                .collect();
        }
    }
    if let Some(rel) = frame.assets.camera.clone() {
        hydrate_rgb_asset(root, &rel, frame, false)?;
    } else if let Some(rel) = frame.assets.rgb.clone() {
        hydrate_rgb_asset(root, &rel, frame, false)?;
    }
    if let Some(rel) = frame.assets.rgb.clone() {
        if frame.snapshot.kinect.color_frame.is_some() {
            hydrate_rgb_asset(root, &rel, frame, true)?;
        }
    }
    if let Some(rel) = frame.assets.audio.as_deref() {
        let path = root.join(rel);
        if path.exists() {
            let mut reader = hound::WavReader::open(&path)
                .with_context(|| format!("reading audio asset {}", path.display()))?;
            let spec = reader.spec();
            let samples = reader
                .samples::<i16>()
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let captured_at_ms = asset_producer_time(frame, "audio").unwrap_or(frame.t_ms);
            frame.snapshot.ear_pcm = Some(PcmAudioFrame {
                captured_at_ms,
                sample_rate_hz: spec.sample_rate,
                channels: spec.channels,
                samples,
            });
        }
    }
    if let Some(rel) = frame.assets.lidar.as_deref() {
        let path = root.join(rel);
        if path.exists() {
            frame.snapshot.range = serde_json::from_slice(&std_fs::read(&path)?)
                .with_context(|| format!("reading lidar asset {}", path.display()))?;
        }
    }
    Ok(())
}

fn hydrate_rgb_asset(
    root: &Path,
    rel: &str,
    frame: &mut CaptureFrameRecord,
    kinect: bool,
) -> Result<()> {
    let path = root.join(rel);
    if !path.exists() {
        return Ok(());
    }
    let image = image::open(&path)
        .with_context(|| format!("reading RGB asset {}", path.display()))?
        .to_rgb8();
    let metadata_kind = if kinect { "rgb" } else { "camera" };
    let metadata = frame
        .stream_metadata
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|metadata| metadata.get(metadata_kind))
        .and_then(Value::as_object);
    let hydrated = pete_now::EyeFrame {
        captured_at_ms: metadata
            .and_then(|value| value.get("producer_t_ms"))
            .and_then(Value::as_u64)
            .unwrap_or(frame.t_ms),
        rgbd_frame_id: metadata
            .and_then(|value| value.get("rgbd_frame_id"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        device_timestamp_ms: metadata
            .and_then(|value| value.get("device_timestamp_ms"))
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok()),
        width: image.width(),
        height: image.height(),
        format: EyeFrameFormat::Rgb8,
        bytes: image.into_raw(),
        source: metadata
            .and_then(|value| value.get("source"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
    };
    if kinect {
        frame.snapshot.kinect.color_frame = Some(hydrated);
    } else {
        frame.snapshot.eye_frame = Some(hydrated);
    }
    Ok(())
}

fn asset_producer_time(frame: &CaptureFrameRecord, kind: &str) -> Option<u64> {
    frame
        .stream_metadata
        .as_ref()?
        .as_object()?
        .get(kind)?
        .get("producer_t_ms")?
        .as_u64()
}

#[derive(Clone, Debug)]
pub struct CaptureReader {
    root: PathBuf,
    manifest: CaptureManifest,
}

impl CaptureReader {
    pub async fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let manifest_path = root.join(MANIFEST_FILE);
        let manifest = serde_json::from_slice(
            &fs::read(&manifest_path)
                .await
                .with_context(|| format!("reading {}", manifest_path.display()))?,
        )?;
        Ok(Self { root, manifest })
    }

    pub fn manifest(&self) -> &CaptureManifest {
        &self.manifest
    }

    pub async fn read_frames(&self) -> Result<Vec<CaptureFrameRecord>> {
        let path = self.root.join(FRAMES_FILE);
        let file = File::open(&path)
            .await
            .with_context(|| format!("opening {}", path.display()))?;
        let mut lines = BufReader::new(file).lines();
        let mut frames = Vec::new();
        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }
            let mut frame: CaptureFrameRecord = serde_json::from_str(&line)?;
            hydrate_frame_assets(&self.root, &mut frame)?;
            frames.push(frame);
        }
        Ok(frames)
    }

    pub async fn stream_frames(&self) -> Result<impl Stream<Item = Result<CaptureFrameRecord>>> {
        let frames = self.read_frames().await?;
        Ok(stream::iter(frames.into_iter().map(Ok)))
    }
}

pub struct CaptureReplayRunner<R> {
    pub runtime: R,
    pub reader: CaptureReader,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CaptureReplaySummary {
    pub frames_replayed: u64,
    pub runtime_ticks: u64,
}

impl<R> CaptureReplayRunner<R>
where
    R: RuntimeLoop + Send,
{
    pub fn new(runtime: R, reader: CaptureReader) -> Self {
        Self { runtime, reader }
    }

    pub async fn replay(&mut self) -> Result<CaptureReplaySummary> {
        self.replay_with_ticks(|_| {}).await
    }

    pub async fn replay_with_ticks<F>(
        &mut self,
        mut observe_tick: F,
    ) -> Result<CaptureReplaySummary>
    where
        F: FnMut(&RuntimeTick),
    {
        let mut summary = CaptureReplaySummary::default();
        for record in self.reader.read_frames().await? {
            let now = record.snapshot.to_now(record.t_ms);
            let tick = self
                .runtime
                .tick(
                    now,
                    ExperienceLatent::default(),
                    Vec::<FuturePrediction>::new(),
                )
                .await?;
            observe_tick(&tick);
            summary.frames_replayed = summary.frames_replayed.saturating_add(1);
            summary.runtime_ticks = summary.runtime_ticks.saturating_add(1);
        }
        Ok(summary)
    }
}

fn capture_id_from_path(root: &Path) -> String {
    root.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.trim().is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| Uuid::new_v4().to_string())
}

async fn write_manifest_atomic(root: &Path, manifest: &CaptureManifest) -> Result<()> {
    let final_path = root.join(MANIFEST_FILE);
    let temp_path = root.join("manifest.json.tmp");
    let bytes = serde_json::to_vec_pretty(manifest)?;
    fs::write(&temp_path, bytes).await?;
    fs::rename(&temp_path, &final_path).await?;
    Ok(())
}

fn default_asset_layout() -> Value {
    serde_json::json!({
        "rgb": "assets/rgb/",
        "depth": "assets/depth/",
        "audio": "assets/audio/",
        "pointcloud": "assets/pointcloud/",
        "perception": "assets/perception/",
        "camera": "assets/camera/",
        "lidar": "assets/lidar/",
        "imu": "assets/imu/",
        "transcript": "assets/transcript/",
        "calibration": "assets/calibration/",
        "paths": "frame asset paths are relative to capture root",
        "rgb_format": "PNG RGB8",
        "depth_format": "PNG Gray16, millimeters",
        "audio_format": "WAV PCM16",
        "pointcloud_format": "PLY ASCII",
        "perception_format": "JSON PerceptionFrame",
        "camera_format": "PNG RGB8",
        "lidar_format": "JSON RangeSense",
        "imu_format": "JSON selected sample plus candidate diagnostics",
        "transcript_format": "JSON EarSense timing and ASR provenance",
        "calibration_format": "JSON sensor geometry/configuration identity"
    })
}

pub fn capture_asset_path(kind: &str, index: u64, extension: &str) -> String {
    format!("assets/{kind}/{index:06}.{extension}")
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaptureAssetExport {
    pub assets: CaptureFrameAssets,
    pub metadata: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DepthImage {
    pub width: u32,
    pub height: u32,
    pub units: DepthUnits,
    pub values_mm: Vec<u16>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthUnits {
    Millimeters,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

impl CameraIntrinsics {
    pub fn approximate_for(width: u32, height: u32) -> Self {
        Self {
            fx: width.max(1) as f32,
            fy: width.max(1) as f32,
            cx: (width.saturating_sub(1)) as f32 * 0.5,
            cy: (height.saturating_sub(1)) as f32 * 0.5,
        }
    }
}

pub fn export_snapshot_assets(
    root: &Path,
    index: u64,
    snapshot: &WorldSnapshot,
    export_rgb: bool,
    export_depth: bool,
    export_audio: bool,
) -> Result<CaptureAssetExport> {
    export_snapshot_assets_with_context(
        root,
        index,
        0,
        snapshot,
        export_rgb,
        export_depth,
        export_audio,
        &CaptureExportContext::default(),
    )
}

#[allow(clippy::too_many_arguments)]
fn export_snapshot_assets_with_context(
    root: &Path,
    index: u64,
    capture_t_ms: u64,
    snapshot: &WorldSnapshot,
    export_rgb: bool,
    export_depth: bool,
    export_audio: bool,
    context: &CaptureExportContext,
) -> Result<CaptureAssetExport> {
    let mut assets = CaptureFrameAssets::default();
    let mut metadata = serde_json::Map::new();

    if export_rgb {
        if let Some(frame) = snapshot.kinect.color_frame.as_ref() {
            if let Some((width, height, rgb)) = eye_frame_rgb8(frame) {
                let rel = capture_asset_path("rgb", index, "png");
                write_rgb_png(&root.join(&rel), width, height, &rgb)?;
                assets.rgb = Some(rel.clone());
                metadata.insert(
                    "rgb".to_string(),
                    written_asset_metadata(
                        root,
                        &rel,
                        capture_t_ms,
                        Some(frame.captured_at_ms),
                        serde_json::json!({
                            "width": width,
                            "height": height,
                            "format": "rgb8_png",
                            "rgbd_frame_id": frame.rgbd_frame_id,
                            "device_timestamp_ms": frame.device_timestamp_ms,
                            "source": frame.source,
                        }),
                    )?,
                );
            } else {
                metadata.insert(
                    "rgb".to_string(),
                    partial_asset_metadata(capture_t_ms, "unsupported or partial Kinect RGB frame"),
                );
            }
        } else {
            metadata.insert(
                "rgb".to_string(),
                unavailable_asset_metadata(capture_t_ms, "Kinect RGB frame unavailable"),
            );
        }

        if let Some(frame) = snapshot.eye_frame.as_ref() {
            if let Some((width, height, rgb)) = eye_frame_rgb8(frame) {
                let rel = capture_asset_path("camera", index, "png");
                write_rgb_png(&root.join(&rel), width, height, &rgb)?;
                assets.camera = Some(rel.clone());
                metadata.insert(
                    "camera".to_string(),
                    written_asset_metadata(
                        root,
                        &rel,
                        capture_t_ms,
                        Some(frame.captured_at_ms),
                        serde_json::json!({
                            "width": width,
                            "height": height,
                            "format": "rgb8_png",
                            "rgbd_frame_id": frame.rgbd_frame_id,
                            "device_timestamp_ms": frame.device_timestamp_ms,
                            "source": frame.source,
                        }),
                    )?,
                );
            } else {
                metadata.insert(
                    "camera".to_string(),
                    partial_asset_metadata(capture_t_ms, "unsupported or partial camera frame"),
                );
            }
        } else if assets.rgb.is_none() {
            // Older single-camera captures used eye_frame for Kinect RGB. Keep
            // the compact fallback useful without claiming a raw camera asset.
            if let Some((width, height, rgb)) = snapshot_rgb8(snapshot) {
                let rel = capture_asset_path("rgb", index, "png");
                write_rgb_png(&root.join(&rel), width, height, &rgb)?;
                assets.rgb = Some(rel.clone());
                metadata.insert(
                    "rgb".to_string(),
                    written_asset_metadata(root, &rel, capture_t_ms, None, serde_json::json!({
                        "width": width, "height": height, "format": "rgb8_png", "source": "compact_eye_features"
                    }))?,
                );
            }
        } else {
            metadata.insert(
                "camera".to_string(),
                unavailable_asset_metadata(capture_t_ms, "camera frame unavailable"),
            );
        }
        if assets.rgb.is_none() {
            if let Some((width, height, rgb)) = snapshot_rgb8(snapshot) {
                let rel = capture_asset_path("rgb", index, "png");
                write_rgb_png(&root.join(&rel), width, height, &rgb)?;
                assets.rgb = Some(rel.clone());
                let producer_t_ms = snapshot
                    .eye_frame
                    .as_ref()
                    .map(|frame| frame.captured_at_ms);
                metadata.insert(
                    "rgb".to_string(),
                    written_asset_metadata(root, &rel, capture_t_ms, producer_t_ms, serde_json::json!({
                        "width": width, "height": height, "format": "rgb8_png",
                        "source": snapshot.eye_frame.as_ref().and_then(|frame| frame.source.as_deref()).unwrap_or("compact_eye_features"),
                        "rgbd_frame_id": snapshot.eye_frame.as_ref().and_then(|frame| frame.rgbd_frame_id.as_deref()),
                        "device_timestamp_ms": snapshot.eye_frame.as_ref().and_then(|frame| frame.device_timestamp_ms),
                    }))?,
                );
            }
        }
    }

    if export_depth {
        if let Some(depth) = snapshot_depth_image(snapshot) {
            let rel = capture_asset_path("depth", index, "depth16.png");
            write_depth_png(&root.join(&rel), &depth)?;
            assets.depth = Some(rel.clone());
            metadata.insert(
                "depth".to_string(),
                written_asset_metadata(
                    root,
                    &rel,
                    capture_t_ms,
                    Some(snapshot.kinect.captured_at_ms),
                    serde_json::json!({
                        "width": depth.width, "height": depth.height, "format": "gray16_png",
                        "units": "millimeters", "scale": 0.001,
                        "rgbd_frame_id": snapshot.kinect.rgbd_frame_id,
                        "device_timestamp_ms": snapshot.kinect.device_timestamp_ms,
                        "coordinate_system": snapshot.kinect.depth_coordinate_system,
                    }),
                )?,
            );
        } else if !snapshot.kinect.depth_m.is_empty() {
            metadata.insert(
                "depth".to_string(),
                partial_asset_metadata(
                    capture_t_ms,
                    "depth samples are present but declared dimensions are incomplete",
                ),
            );
        } else {
            metadata.insert(
                "depth".to_string(),
                unavailable_asset_metadata(
                    capture_t_ms,
                    "depth frame unavailable or dimensions are partial",
                ),
            );
        }
    }

    if export_audio {
        if let Some(audio) = &snapshot.ear_pcm {
            if audio.sample_rate_hz == 0 || audio.channels == 0 {
                metadata.insert(
                    "audio".to_string(),
                    partial_asset_metadata(
                        capture_t_ms,
                        "PCM audio chunk has no valid sample rate or channel count",
                    ),
                );
            } else {
                let rel = capture_asset_path("audio", index, "wav");
                write_wav(&root.join(&rel), audio)?;
                assets.audio = Some(rel.clone());
                metadata.insert(
                    "audio".to_string(),
                    written_asset_metadata(
                        root,
                        &rel,
                        capture_t_ms,
                        Some(audio.captured_at_ms),
                        serde_json::json!({
                            "sample_rate_hz": audio.sample_rate_hz, "channels": audio.channels,
                            "format": "pcm16_wav", "samples": audio.samples.len(),
                        }),
                    )?,
                );
            }
        } else {
            metadata.insert(
                "audio".to_string(),
                unavailable_asset_metadata(capture_t_ms, "PCM audio chunk unavailable"),
            );
        }
    }

    if !snapshot.range.beams.is_empty() || snapshot.range.nearest_m.is_some() {
        let angles_complete = snapshot.range.beam_angles_rad.is_empty()
            || snapshot.range.beam_angles_rad.len() == snapshot.range.beams.len();
        let timing_complete = snapshot.range.beam_time_offsets_ms.is_empty()
            || snapshot.range.beam_time_offsets_ms.len() == snapshot.range.beams.len();
        let poses_complete = snapshot.range.beam_poses.is_empty()
            || snapshot.range.beam_poses.len() == snapshot.range.beams.len();
        let partial_reasons = [
            (!angles_complete).then_some("beam angle count does not match beam count"),
            (!timing_complete).then_some("beam time-offset count does not match beam count"),
            (!poses_complete).then_some("beam pose count does not match beam count"),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        let rel = capture_asset_path("lidar", index, "json");
        write_json_asset(&root.join(&rel), &snapshot.range)?;
        assets.lidar = Some(rel.clone());
        metadata.insert(
            "lidar".to_string(),
            written_asset_metadata(
                root,
                &rel,
                capture_t_ms,
                Some(snapshot.range.captured_at_ms),
                serde_json::json!({
                    "format": "range_sense_json", "beams": snapshot.range.beams.len(),
                    "source": snapshot.range.source, "frame": snapshot.range.frame,
                    "producer_time_offsets": snapshot.range.beam_time_offsets_ms.len(),
                    "interpolated_beam_poses": snapshot.range.beam_poses.len(),
                    "completeness": if angles_complete && timing_complete && poses_complete { "complete" } else { "partial" },
                    "partial_reasons": partial_reasons,
                }),
            )?,
        );
    } else {
        metadata.insert(
            "lidar".to_string(),
            unavailable_asset_metadata(capture_t_ms, "lidar sweep unavailable"),
        );
    }

    let has_imu = !snapshot.imu.orientation.is_empty()
        || !snapshot.imu.acceleration.is_empty()
        || !snapshot.imu.angular_velocity.is_empty()
        || context.imu_selection.is_some();
    if has_imu {
        let rel = capture_asset_path("imu", index, "json");
        let value = serde_json::json!({
            "selected": snapshot.imu,
            "selection": context.imu_selection,
        });
        write_json_asset(&root.join(&rel), &value)?;
        assets.imu = Some(rel.clone());
        metadata.insert(
            "imu".to_string(),
            written_asset_metadata(root, &rel, capture_t_ms, Some(snapshot.imu.captured_at_ms), serde_json::json!({
                "format": "imu_selection_json", "selected_source": snapshot.imu.source_id(),
                "source_epoch": snapshot.imu.source_epoch(),
                "calibration_epoch": snapshot.imu.calibration.as_ref().map(|value| value.epoch.id),
                "calibration_trust": snapshot.imu.calibration.as_ref().map(|value| value.trust_state),
                "temperature_c": snapshot.imu.temperature_c,
                "gyro_bias_rad_s": snapshot.imu.calibration.as_ref().map(|value| value.gyro_bias_rad_s),
                "gyro_variance": snapshot.imu.calibration.as_ref().map(|value| value.gyro_variance),
                "candidate_count": context.imu_selection.as_ref().and_then(|value| value.get("candidates")).and_then(Value::as_array).map(Vec::len).unwrap_or(0),
            }))?,
        );
    } else {
        metadata.insert(
            "imu".to_string(),
            unavailable_asset_metadata(
                capture_t_ms,
                "selected and candidate IMU samples unavailable",
            ),
        );
    }

    if snapshot.ear.transcript.is_some()
        || snapshot.ear.asr.transcript.is_some()
        || !snapshot.ear.asr.candidate_events.is_empty()
    {
        let rel = capture_asset_path("transcript", index, "json");
        write_json_asset(&root.join(&rel), &snapshot.ear)?;
        assets.transcript = Some(rel.clone());
        metadata.insert(
            "transcript".to_string(),
            written_asset_metadata(
                root,
                &rel,
                capture_t_ms,
                snapshot.ear.asr.start_ms,
                serde_json::json!({
                    "format": "ear_sense_json", "start_ms": snapshot.ear.asr.start_ms,
                    "end_ms": snapshot.ear.asr.end_ms, "is_final": snapshot.ear.asr.is_final,
                    "candidate_events": snapshot.ear.asr.candidate_events.len(),
                }),
            )?,
        );
    } else {
        metadata.insert(
            "transcript".to_string(),
            unavailable_asset_metadata(capture_t_ms, "audio transcript unavailable"),
        );
    }

    let calibration = serde_json::json!({
        "kinect_geometry": snapshot.kinect.geometry_calibration,
        "kinect_live_geometry": snapshot.kinect.live_geometry_calibration,
        "depth_intrinsics_fallback": {
            "width": snapshot.kinect.depth_width, "height": snapshot.kinect.depth_height,
            "fx": snapshot.kinect.depth_fx, "fy": snapshot.kinect.depth_fy,
            "cx": snapshot.kinect.depth_cx, "cy": snapshot.kinect.depth_cy,
            "distortion": snapshot.kinect.depth_distortion,
        },
        "depth_coordinate_system": snapshot.kinect.depth_coordinate_system,
        "range_extrinsics": snapshot.range.extrinsics,
        "range_source": snapshot.range.source,
        "sensor_latency": snapshot.latency_calibration,
        "locomotion": snapshot.locomotion_calibration,
    });
    let rel = capture_asset_path("calibration", index, "json");
    write_json_asset(&root.join(&rel), &calibration)?;
    assets.calibration = Some(rel.clone());
    let calibration_metadata = written_asset_metadata(
        root,
        &rel,
        capture_t_ms,
        None,
        serde_json::json!({
            "format": "sensor_calibration_json",
            "identity": sha256_bytes(&serde_json::to_vec(&calibration)?),
            "physically_calibrated": snapshot.kinect.geometry_calibration.is_some_and(|value| value.calibrated),
            "live_trust_state": snapshot.kinect.live_geometry_calibration.as_ref().map(|value| value.trust_state),
            "calibration_epoch": snapshot.kinect.live_geometry_calibration.as_ref().map(|value| value.epoch.id),
        }),
    )?;
    metadata.insert("calibration".to_string(), calibration_metadata.clone());

    if export_depth {
        if let (Some(depth), Some(calibration)) = (
            snapshot_depth_image(snapshot),
            snapshot.kinect.geometry_calibration,
        ) {
            let rel = capture_asset_path("pointcloud", index, "ply");
            let intrinsics = CameraIntrinsics {
                fx: calibration.depth.fx,
                fy: calibration.depth.fy,
                cx: calibration.depth.cx,
                cy: calibration.depth.cy,
            };
            let vertices = write_pointcloud_ply(
                &root.join(&rel),
                &depth,
                intrinsics,
                snapshot.kinect.max_depth_m.max(8.0),
                1,
            )?;
            assets.pointcloud = Some(rel.clone());
            metadata.insert(
                "pointcloud".to_string(),
                written_asset_metadata(
                    root,
                    &rel,
                    capture_t_ms,
                    Some(snapshot.kinect.captured_at_ms),
                    serde_json::json!({
                        "format": "ply_ascii", "vertices": vertices, "stride": 1,
                        "calibration_identity": calibration_metadata["identity"],
                        "derived_from": assets.depth,
                    }),
                )?,
            );
        } else {
            metadata.insert(
                "pointcloud".to_string(),
                unavailable_asset_metadata(
                    capture_t_ms,
                    "point cloud requires a raw depth frame and explicit calibration",
                ),
            );
        }
    }

    Ok(CaptureAssetExport {
        assets,
        metadata: Value::Object(metadata),
    })
}

fn write_json_asset(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        std_fs::create_dir_all(parent)?;
    }
    std_fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn written_asset_metadata(
    root: &Path,
    rel: &str,
    capture_t_ms: u64,
    producer_t_ms: Option<u64>,
    details: Value,
) -> Result<Value> {
    let path = root.join(rel);
    let bytes = std_fs::metadata(&path)?.len();
    let timing = producer_t_ms.map_or("unknown", |producer| {
        if capture_t_ms == 0 {
            "unknown"
        } else if producer > capture_t_ms {
            "future"
        } else if capture_t_ms.saturating_sub(producer) > ASSET_LATE_AFTER_MS {
            "late"
        } else {
            "on_time"
        }
    });
    let mut value = details.as_object().cloned().unwrap_or_default();
    value.insert("status".to_string(), Value::String("written".to_string()));
    value.insert("path".to_string(), Value::String(rel.to_string()));
    value.insert("capture_t_ms".to_string(), Value::from(capture_t_ms));
    value.insert(
        "producer_t_ms".to_string(),
        producer_t_ms.map(Value::from).unwrap_or(Value::Null),
    );
    value.insert(
        "captured_at_ms".to_string(),
        producer_t_ms.map(Value::from).unwrap_or(Value::Null),
    );
    value.insert("timing".to_string(), Value::String(timing.to_string()));
    value.insert("bytes".to_string(), Value::from(bytes));
    value.insert("sha256".to_string(), Value::String(sha256_file(&path)?));
    Ok(Value::Object(value))
}

fn unavailable_asset_metadata(capture_t_ms: u64, reason: &str) -> Value {
    serde_json::json!({
        "status": "unavailable",
        "capture_t_ms": capture_t_ms,
        "reason": reason,
    })
}

fn partial_asset_metadata(capture_t_ms: u64, reason: &str) -> Value {
    serde_json::json!({
        "status": "partial",
        "capture_t_ms": capture_t_ms,
        "reason": reason,
    })
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String> {
    Ok(sha256_bytes(&std_fs::read(path)?))
}

pub fn snapshot_depth_image(snapshot: &WorldSnapshot) -> Option<DepthImage> {
    if snapshot.kinect.depth_m.is_empty() {
        return None;
    }
    let sample_count = snapshot.kinect.depth_m.len();
    let declared_width = usize::try_from(snapshot.kinect.depth_width).unwrap_or(0);
    let declared_height = usize::try_from(snapshot.kinect.depth_height).unwrap_or(0);
    let (width, height) = if declared_width > 0
        && declared_height > 0
        && declared_width.saturating_mul(declared_height) == sample_count
    {
        (snapshot.kinect.depth_width, snapshot.kinect.depth_height)
    } else {
        return None;
    };
    let values_mm = snapshot
        .kinect
        .depth_m
        .iter()
        .map(|meters| {
            if meters.is_finite() && *meters > 0.0 {
                (meters * 1000.0).round().clamp(0.0, u16::MAX as f32) as u16
            } else {
                0
            }
        })
        .collect();
    Some(DepthImage {
        width,
        height,
        units: DepthUnits::Millimeters,
        values_mm,
    })
}

pub fn write_pointcloud_ply(
    path: &Path,
    depth: &DepthImage,
    intrinsics: CameraIntrinsics,
    max_depth_m: f32,
    stride: usize,
) -> Result<usize> {
    let stride = stride.max(1);
    let mut vertices = Vec::new();
    for y in (0..depth.height as usize).step_by(stride) {
        for x in (0..depth.width as usize).step_by(stride) {
            let index = y * depth.width as usize + x;
            let z_m = depth.values_mm.get(index).copied().unwrap_or(0) as f32 * 0.001;
            if z_m <= 0.0 || z_m > max_depth_m {
                continue;
            }
            let x_m = ((x as f32 - intrinsics.cx) * z_m) / intrinsics.fx.max(f32::EPSILON);
            let y_m = ((y as f32 - intrinsics.cy) * z_m) / intrinsics.fy.max(f32::EPSILON);
            vertices.push((x_m, y_m, z_m));
        }
    }

    let mut out = String::new();
    out.push_str("ply\nformat ascii 1.0\n");
    out.push_str(&format!("element vertex {}\n", vertices.len()));
    out.push_str("property float x\nproperty float y\nproperty float z\nend_header\n");
    for (x, y, z) in &vertices {
        out.push_str(&format!("{x:.6} {y:.6} {z:.6}\n"));
    }
    if let Some(parent) = path.parent() {
        std_fs::create_dir_all(parent)?;
    }
    std_fs::write(path, out)?;
    Ok(vertices.len())
}

pub fn export_pointcloud_for_frame(
    root: &Path,
    frame: &mut CaptureFrameRecord,
    max_depth_m: f32,
    stride: usize,
) -> Result<Option<Value>> {
    let Some(depth) = snapshot_depth_image(&frame.snapshot) else {
        return Ok(None);
    };
    let rel = capture_asset_path("pointcloud", frame.index, "ply");
    let recorded_calibration = frame.snapshot.kinect.geometry_calibration;
    let intrinsics = recorded_calibration.map_or_else(
        || CameraIntrinsics::approximate_for(depth.width, depth.height),
        |calibration| CameraIntrinsics {
            fx: calibration.depth.fx,
            fy: calibration.depth.fy,
            cx: calibration.depth.cx,
            cy: calibration.depth.cy,
        },
    );
    let vertices = write_pointcloud_ply(&root.join(&rel), &depth, intrinsics, max_depth_m, stride)?;
    frame.assets.pointcloud = Some(rel);
    let mut metadata = frame
        .stream_metadata
        .take()
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    metadata.insert(
        "pointcloud".to_string(),
        serde_json::json!({
            "format": "ply_ascii",
            "vertices": vertices,
            "stride": stride.max(1),
            "max_depth_m": max_depth_m,
            "intrinsics": {
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.cx,
                "cy": intrinsics.cy
            },
            "calibration": if recorded_calibration.is_some() { "recorded" } else { "uncalibrated" },
            "calibration_identity": recorded_calibration
                .and_then(|calibration| serde_json::to_vec(&calibration).ok())
                .map(|bytes| sha256_bytes(&bytes)),
        }),
    );
    let metadata = Value::Object(metadata);
    frame.stream_metadata = Some(metadata.clone());
    Ok(Some(metadata))
}

pub fn export_perception_for_frame(
    root: &Path,
    frame: &mut CaptureFrameRecord,
) -> Result<Option<Value>> {
    let Some(perception) = PerceptionFrame::from_world_snapshot(&frame.snapshot, frame.t_ms) else {
        return Ok(None);
    };
    let rel = capture_asset_path("perception", frame.index, "json");
    let path = root.join(&rel);
    if let Some(parent) = path.parent() {
        std_fs::create_dir_all(parent)?;
    }
    std_fs::write(&path, serde_json::to_vec_pretty(&perception)?)?;
    frame.assets.perception = Some(rel);
    let mut metadata = frame
        .stream_metadata
        .take()
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    metadata.insert(
        "perception".to_string(),
        serde_json::json!({
            "format": "perception_frame_json",
            "frame_id": perception.frame_id.0,
            "points": perception.points.len()
        }),
    );
    let metadata = Value::Object(metadata);
    frame.stream_metadata = Some(metadata.clone());
    Ok(Some(metadata))
}

pub async fn rewrite_frames(root: &Path, frames: &[CaptureFrameRecord]) -> Result<()> {
    let path = root.join(FRAMES_FILE);
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&path)
        .await?;
    for frame in frames {
        let line = serde_json::to_string(frame)?;
        file.write_all(line.as_bytes()).await?;
        file.write_all(b"\n").await?;
    }
    file.flush().await?;
    Ok(())
}

pub async fn update_manifest(root: &Path, manifest: &CaptureManifest) -> Result<()> {
    write_manifest_atomic(root, manifest).await
}

fn snapshot_rgb8(snapshot: &WorldSnapshot) -> Option<(u32, u32, Vec<u8>)> {
    if let Some(frame) = &snapshot.eye_frame {
        if let Some(rgb) = eye_frame_rgb8(frame) {
            return Some(rgb);
        }
    }

    let features = snapshot.eye.frames.first()?;
    let width = features.len().max(1) as u32;
    let mut rgb = Vec::with_capacity(width as usize * 3);
    for value in features {
        let byte = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
        rgb.extend_from_slice(&[byte, byte, byte]);
    }
    Some((width, 1, rgb))
}

fn eye_frame_rgb8(frame: &pete_now::EyeFrame) -> Option<(u32, u32, Vec<u8>)> {
    match frame.format {
        EyeFrameFormat::Rgb8
            if frame.bytes.len() == frame.width as usize * frame.height as usize * 3 =>
        {
            Some((frame.width, frame.height, frame.bytes.clone()))
        }
        EyeFrameFormat::Bgr8
            if frame.bytes.len() == frame.width as usize * frame.height as usize * 3 =>
        {
            let mut rgb = frame.bytes.clone();
            for pixel in rgb.chunks_exact_mut(3) {
                pixel.swap(0, 2);
            }
            Some((frame.width, frame.height, rgb))
        }
        EyeFrameFormat::Gray8
            if frame.bytes.len() == frame.width as usize * frame.height as usize =>
        {
            let mut rgb = Vec::with_capacity(frame.bytes.len() * 3);
            for value in &frame.bytes {
                rgb.extend_from_slice(&[*value, *value, *value]);
            }
            Some((frame.width, frame.height, rgb))
        }
        EyeFrameFormat::Yuyv422
            if frame.bytes.len() == frame.width as usize * frame.height as usize * 2 =>
        {
            Some((frame.width, frame.height, yuyv422_to_rgb(&frame.bytes)))
        }
        _ => None,
    }
}

fn yuyv422_to_rgb(bytes: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(bytes.len() / 2 * 3);
    for pair in bytes.chunks_exact(4) {
        let y0 = pair[0];
        let u = pair[1];
        let y1 = pair[2];
        let v = pair[3];
        push_yuv_rgb(&mut rgb, y0, u, v);
        push_yuv_rgb(&mut rgb, y1, u, v);
    }
    rgb
}

fn push_yuv_rgb(rgb: &mut Vec<u8>, y: u8, u: u8, v: u8) {
    let c = y as i32 - 16;
    let d = u as i32 - 128;
    let e = v as i32 - 128;
    rgb.push(((298 * c + 409 * e + 128) >> 8).clamp(0, 255) as u8);
    rgb.push(((298 * c - 100 * d - 208 * e + 128) >> 8).clamp(0, 255) as u8);
    rgb.push(((298 * c + 516 * d + 128) >> 8).clamp(0, 255) as u8);
}

fn write_rgb_png(path: &Path, width: u32, height: u32, rgb: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std_fs::create_dir_all(parent)?;
    }
    let image: RgbImage = ImageBuffer::from_vec(width, height, rgb.to_vec())
        .context("RGB bytes did not match image dimensions")?;
    image.save(path)?;
    Ok(())
}

fn write_depth_png(path: &Path, depth: &DepthImage) -> Result<()> {
    if let Some(parent) = path.parent() {
        std_fs::create_dir_all(parent)?;
    }
    let image: ImageBuffer<Luma<u16>, Vec<u16>> =
        ImageBuffer::from_vec(depth.width, depth.height, depth.values_mm.clone())
            .context("depth values did not match image dimensions")?;
    image.save(path)?;
    Ok(())
}

fn write_wav(path: &Path, audio: &PcmAudioFrame) -> Result<()> {
    if let Some(parent) = path.parent() {
        std_fs::create_dir_all(parent)?;
    }
    let spec = hound::WavSpec {
        channels: audio.channels,
        sample_rate: audio.sample_rate_hz,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in &audio.samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
