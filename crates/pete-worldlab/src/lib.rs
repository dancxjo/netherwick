use std::fs as std_fs;
use std::path::{Path, PathBuf};

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
use tokio::fs::{self, File, OpenOptions};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio_stream::{self as stream, Stream};
use uuid::Uuid;

pub const CAPTURE_SCHEMA_VERSION: u32 = 1;
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
}

impl CaptureFrameAssets {
    pub fn is_empty(&self) -> bool {
        self.rgb.is_none()
            && self.depth.is_none()
            && self.audio.is_none()
            && self.pointcloud.is_none()
            && self.perception.is_none()
    }
}

pub struct CaptureWriter {
    root: PathBuf,
    manifest: CaptureManifest,
    frames: File,
    frame_count: u64,
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
            command_args: Vec::new(),
            device_availability: Value::Null,
            streams: CaptureStreams::default(),
            started_at_ms: Some(now_ms),
            ended_at_ms: None,
            warnings: Vec::new(),
            asset_layout: default_asset_layout(),
            scenario: None,
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
            frames,
            frame_count: 0,
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
        self.frames.write_all(line.as_bytes()).await?;
        self.frames.write_all(b"\n").await?;
        self.frame_count = self.frame_count.saturating_add(1);
        Ok(())
    }

    pub async fn finish(mut self) -> Result<CaptureManifest> {
        self.frames.flush().await?;
        self.manifest.frame_count = self.frame_count;
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
            frames.push(serde_json::from_str(&line)?);
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
        "paths": "frame asset paths are relative to capture root",
        "rgb_format": "PNG RGB8",
        "depth_format": "PNG Gray16, millimeters",
        "audio_format": "WAV PCM16",
        "pointcloud_format": "PLY ASCII",
        "perception_format": "JSON PerceptionFrame"
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
    let mut assets = CaptureFrameAssets::default();
    let mut metadata = serde_json::Map::new();

    if export_rgb {
        if let Some((width, height, rgb)) = snapshot_rgb8(snapshot) {
            let rel = capture_asset_path("rgb", index, "png");
            write_rgb_png(&root.join(&rel), width, height, &rgb)?;
            assets.rgb = Some(rel);
            metadata.insert(
                "rgb".to_string(),
                serde_json::json!({"width": width, "height": height, "format": "rgb8_png"}),
            );
        }
    }

    if export_depth {
        if let Some(depth) = snapshot_depth_image(snapshot) {
            let rel = capture_asset_path("depth", index, "depth16.png");
            write_depth_png(&root.join(&rel), &depth)?;
            assets.depth = Some(rel);
            metadata.insert(
                "depth".to_string(),
                serde_json::json!({
                    "width": depth.width,
                    "height": depth.height,
                    "format": "gray16_png",
                    "units": "millimeters",
                    "scale": 0.001
                }),
            );
        }
    }

    if export_audio {
        if let Some(audio) = &snapshot.ear_pcm {
            let rel = capture_asset_path("audio", index, "wav");
            write_wav(&root.join(&rel), audio)?;
            assets.audio = Some(rel);
            metadata.insert(
                "audio".to_string(),
                serde_json::json!({
                    "sample_rate_hz": audio.sample_rate_hz,
                    "channels": audio.channels,
                    "format": "pcm16_wav",
                    "samples": audio.samples.len()
                }),
            );
        }
    }

    Ok(CaptureAssetExport {
        assets,
        metadata: Value::Object(metadata),
    })
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
    let intrinsics = CameraIntrinsics::approximate_for(depth.width, depth.height);
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
            "calibration": "uncalibrated"
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
        match frame.format {
            EyeFrameFormat::Rgb8
                if frame.bytes.len() == frame.width as usize * frame.height as usize * 3 =>
            {
                return Some((frame.width, frame.height, frame.bytes.clone()));
            }
            EyeFrameFormat::Bgr8
                if frame.bytes.len() == frame.width as usize * frame.height as usize * 3 =>
            {
                let mut rgb = frame.bytes.clone();
                for pixel in rgb.chunks_exact_mut(3) {
                    pixel.swap(0, 2);
                }
                return Some((frame.width, frame.height, rgb));
            }
            EyeFrameFormat::Gray8
                if frame.bytes.len() == frame.width as usize * frame.height as usize =>
            {
                let mut rgb = Vec::with_capacity(frame.bytes.len() * 3);
                for value in &frame.bytes {
                    rgb.extend_from_slice(&[*value, *value, *value]);
                }
                return Some((frame.width, frame.height, rgb));
            }
            EyeFrameFormat::Yuyv422
                if frame.bytes.len() == frame.width as usize * frame.height as usize * 2 =>
            {
                return Some((frame.width, frame.height, yuyv422_to_rgb(&frame.bytes)));
            }
            _ => {}
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
mod tests {
    use super::*;
    use pete_autonomic::SimpleSafety;
    use pete_body::BodySense;
    use pete_conductor::SimpleConductor;
    use pete_ledger::JsonlLedger;
    use pete_llm::NoopLlmAgent;
    use pete_memory::InMemoryExperienceStore;
    use pete_runtime::{MinimalRuntime, SimRunner};
    use pete_sim::{ArenaConfig, VirtualWorld};
    use tempfile::tempdir;

    #[test]
    fn capture_manifest_round_trips() {
        let manifest = CaptureManifest {
            id: "round-trip".to_string(),
            created_at_ms: 123,
            source: CaptureSource::Sim,
            schema_version: CAPTURE_SCHEMA_VERSION,
            frame_count: 2,
            tick_ms: Some(100),
            notes: vec!["small and sturdy".to_string()],
            machine: None,
            command_args: Vec::new(),
            device_availability: Value::Null,
            streams: CaptureStreams::default(),
            started_at_ms: Some(123),
            ended_at_ms: Some(456),
            warnings: Vec::new(),
            asset_layout: default_asset_layout(),
            scenario: None,
        };

        let encoded = serde_json::to_string(&manifest).unwrap();
        let decoded: CaptureManifest = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded, manifest);
    }

    #[test]
    fn capture_asset_paths_are_deterministic() {
        assert_eq!(capture_asset_path("rgb", 7, "png"), "assets/rgb/000007.png");
        assert_eq!(
            capture_asset_path("depth", 42, "depth16.png"),
            "assets/depth/000042.depth16.png"
        );
    }

    #[test]
    fn capture_frame_records_require_asset_and_stream_metadata_fields() {
        let encoded = serde_json::json!({
            "index": 0,
            "t_ms": 123,
            "snapshot": WorldSnapshot::default(),
            "events": []
        });

        assert!(serde_json::from_value::<CaptureFrameRecord>(encoded).is_err());
    }

    #[test]
    fn frame_asset_references_serialize() {
        let record = CaptureFrameRecord {
            index: 1,
            t_ms: 1234,
            snapshot: WorldSnapshot::default(),
            events: Vec::new(),
            assets: CaptureFrameAssets {
                rgb: Some("assets/rgb/000001.png".to_string()),
                depth: Some("assets/depth/000001.depth16.png".to_string()),
                audio: None,
                pointcloud: Some("assets/pointcloud/000001.ply".to_string()),
                perception: None,
            },
            stream_metadata: Some(serde_json::json!({"rgb": {"width": 2, "height": 2}})),
        };

        let encoded = serde_json::to_value(&record).unwrap();

        assert_eq!(encoded["assets"]["rgb"], "assets/rgb/000001.png");
        assert_eq!(
            encoded["assets"]["pointcloud"],
            "assets/pointcloud/000001.ply"
        );
        assert_eq!(encoded["stream_metadata"]["rgb"]["width"], 2);
    }

    #[test]
    fn perception_export_writes_frame_json_asset() {
        let dir = tempdir().unwrap();
        let mut frame = CaptureFrameRecord {
            index: 123,
            t_ms: 456,
            snapshot: WorldSnapshot::default(),
            events: Vec::new(),
            assets: CaptureFrameAssets::default(),
            stream_metadata: None,
        };
        frame.snapshot.kinect.depth_width = 2;
        frame.snapshot.kinect.depth_height = 1;
        frame.snapshot.kinect.depth_m = vec![1.0, 2.0];
        frame.snapshot.kinect.depth_fx = 1.0;
        frame.snapshot.kinect.depth_fy = 1.0;
        frame.snapshot.kinect.min_depth_m = 0.1;
        frame.snapshot.kinect.max_depth_m = 8.0;

        let metadata = export_perception_for_frame(dir.path(), &mut frame)
            .unwrap()
            .unwrap();
        let rel = frame.assets.perception.as_deref().unwrap();
        let encoded = std_fs::read_to_string(dir.path().join(rel)).unwrap();
        let decoded: PerceptionFrame = serde_json::from_str(&encoded).unwrap();

        assert_eq!(rel, "assets/perception/000123.json");
        assert_eq!(metadata["perception"]["points"], 2);
        assert_eq!(decoded.points[1].depth.index, 1);
        assert_eq!(decoded.points[1].depth.u, 1);
    }

    #[test]
    fn tiny_depth_map_exports_expected_ply_vertices() {
        let dir = tempdir().unwrap();
        let depth = DepthImage {
            width: 2,
            height: 2,
            units: DepthUnits::Millimeters,
            values_mm: vec![1000, 0, 2000, 9000],
        };
        let path = dir.path().join("tiny.ply");

        let vertices = write_pointcloud_ply(
            &path,
            &depth,
            CameraIntrinsics {
                fx: 2.0,
                fy: 2.0,
                cx: 0.5,
                cy: 0.5,
            },
            3.0,
            1,
        )
        .unwrap();
        let ply = std_fs::read_to_string(path).unwrap();

        assert_eq!(vertices, 2);
        assert!(ply.contains("element vertex 2"));
        let vertex_rows = ply
            .lines()
            .skip_while(|line| *line != "end_header")
            .skip(1)
            .count();
        assert_eq!(vertex_rows, 2);
    }

    #[test]
    fn snapshot_depth_image_preserves_declared_dimensions() {
        let mut snapshot = WorldSnapshot::default();
        snapshot.kinect.depth_width = 2;
        snapshot.kinect.depth_height = 2;
        snapshot.kinect.depth_m = vec![1.0, 0.0, f32::NAN, 65.536];

        let depth = snapshot_depth_image(&snapshot).unwrap();

        assert_eq!(depth.width, 2);
        assert_eq!(depth.height, 2);
        assert_eq!(depth.values_mm, vec![1000, 0, 0, u16::MAX]);
    }

    #[test]
    fn snapshot_depth_image_rejects_invalid_dimensions() {
        let mut snapshot = WorldSnapshot::default();
        snapshot.kinect.depth_width = 2;
        snapshot.kinect.depth_height = 2;
        snapshot.kinect.depth_m = vec![1.0, 2.0, 3.0];

        assert!(snapshot_depth_image(&snapshot).is_none());
    }

    #[tokio::test]
    async fn capture_writer_writes_manifest_and_frames() {
        let dir = tempdir().unwrap();
        let mut writer = CaptureWriter::create(dir.path(), CaptureSource::Sim, Some(100))
            .await
            .unwrap();
        writer
            .append_snapshot(10, WorldSnapshot::default(), Vec::new())
            .await
            .unwrap();
        let manifest = writer.finish().await.unwrap();

        assert_eq!(manifest.frame_count, 1);
        assert!(dir.path().join(MANIFEST_FILE).exists());
        let frames = fs::read_to_string(dir.path().join(FRAMES_FILE))
            .await
            .unwrap();
        assert_eq!(frames.lines().count(), 1);
    }

    #[tokio::test]
    async fn capture_reader_reads_frames_in_order() {
        let dir = tempdir().unwrap();
        let mut writer = CaptureWriter::create(dir.path(), CaptureSource::Sim, Some(100))
            .await
            .unwrap();
        for t_ms in [100, 200, 300] {
            writer
                .append_snapshot(t_ms, WorldSnapshot::default(), Vec::new())
                .await
                .unwrap();
        }
        writer.finish().await.unwrap();

        let reader = CaptureReader::open(dir.path()).await.unwrap();
        let frames = reader.read_frames().await.unwrap();

        assert_eq!(
            frames.iter().map(|frame| frame.index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            frames.iter().map(|frame| frame.t_ms).collect::<Vec<_>>(),
            vec![100, 200, 300]
        );
    }

    #[test]
    fn serializable_snapshot_to_now_preserves_body_battery() {
        let mut snapshot = SerializableWorldSnapshot::default();
        snapshot.body = BodySense {
            battery_level: 0.42,
            last_update_ms: 500,
            ..BodySense::default()
        };

        let now = snapshot.to_now(600);

        assert_eq!(now.t_ms, 600);
        assert_eq!(now.body.battery_level, 0.42);
    }

    #[tokio::test]
    async fn capture_sim_creates_nonempty_session() {
        let capture_dir = tempdir().unwrap();
        let ledger_dir = tempdir().unwrap();
        let runtime = test_runtime(ledger_dir.path());
        let (world, motors) = VirtualWorld::new_with_motor(
            7,
            ArenaConfig {
                width_m: 4.0,
                height_m: 4.0,
            },
        );
        let mut runner = SimRunner::new(runtime, world, motors);
        let mut snapshots = Vec::new();
        runner
            .run_steps_observing(3, |snapshot| snapshots.push(snapshot.clone()))
            .await
            .unwrap();

        let mut writer = CaptureWriter::create(capture_dir.path(), CaptureSource::Sim, Some(100))
            .await
            .unwrap();
        for snapshot in snapshots {
            writer
                .append_snapshot(snapshot.body.last_update_ms, snapshot, Vec::new())
                .await
                .unwrap();
        }
        let manifest = writer.finish().await.unwrap();
        let reader = CaptureReader::open(capture_dir.path()).await.unwrap();

        assert_eq!(manifest.frame_count, 3);
        assert_eq!(reader.read_frames().await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn replay_capture_produces_runtime_ticks() {
        let capture_dir = tempdir().unwrap();
        let ledger_dir = tempdir().unwrap();
        let mut writer = CaptureWriter::create(capture_dir.path(), CaptureSource::Sim, Some(100))
            .await
            .unwrap();
        for t_ms in [100, 200, 300] {
            let mut snapshot = WorldSnapshot::default();
            snapshot.body.last_update_ms = t_ms;
            snapshot.body.battery_level = 1.0 - (t_ms as f32 / 1_000.0);
            writer
                .append_snapshot(t_ms, snapshot, Vec::new())
                .await
                .unwrap();
        }
        writer.finish().await.unwrap();

        let reader = CaptureReader::open(capture_dir.path()).await.unwrap();
        let ledger = JsonlLedger::new(ledger_dir.path());
        let runtime = test_runtime(ledger_dir.path());
        let mut runner = CaptureReplayRunner::new(runtime, reader);
        let summary = runner.replay().await.unwrap();
        let transitions = ledger.transitions().await.unwrap();

        assert_eq!(summary.frames_replayed, 3);
        assert_eq!(summary.runtime_ticks, 3);
        assert_eq!(transitions.len(), 2);
    }

    fn test_runtime(
        ledger_path: impl Into<std::path::PathBuf>,
    ) -> MinimalRuntime<
        JsonlLedger,
        InMemoryExperienceStore,
        InMemoryExperienceStore,
        SimpleConductor,
        SimpleSafety,
        NoopLlmAgent,
    > {
        let memory = InMemoryExperienceStore::new();
        MinimalRuntime::with_default_events(
            JsonlLedger::new(ledger_path),
            memory.clone(),
            memory,
            SimpleConductor::default(),
            SimpleSafety::default(),
            NoopLlmAgent,
        )
    }
}
