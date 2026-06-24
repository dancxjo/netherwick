use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::Utc;
use netherwick_experience::{ExperienceLatent, FuturePrediction};
use netherwick_runtime::{RuntimeLoop, RuntimeTick};
use netherwick_sensors::WorldSnapshot;
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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecordedEvent {
    pub t_ms: u64,
    pub kind: String,
    pub payload: Value,
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
        File::create(root.join(EVENTS_FILE)).await?;

        let manifest = CaptureManifest {
            id: capture_id_from_path(&root),
            created_at_ms: Utc::now().timestamp_millis().max(0) as u64,
            source,
            schema_version: CAPTURE_SCHEMA_VERSION,
            frame_count: 0,
            tick_ms,
            notes: Vec::new(),
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
        let record = CaptureFrameRecord {
            index: self.frame_count,
            t_ms,
            snapshot,
            events,
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
        write_manifest_atomic(&self.root, &self.manifest).await?;
        Ok(self.manifest)
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

    pub async fn stream_frames(
        &self,
    ) -> Result<impl Stream<Item = Result<CaptureFrameRecord>>> {
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

    pub async fn replay_with_ticks<F>(&mut self, mut observe_tick: F) -> Result<CaptureReplaySummary>
    where
        F: FnMut(&RuntimeTick),
    {
        let mut summary = CaptureReplaySummary::default();
        for record in self.reader.read_frames().await? {
            let now = record.snapshot.to_now(record.t_ms);
            let tick = self
                .runtime
                .tick(now, ExperienceLatent::default(), Vec::<FuturePrediction>::new())
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

#[cfg(test)]
mod tests {
    use super::*;
    use netherwick_body::BodySense;
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
        };

        let encoded = serde_json::to_string(&manifest).unwrap();
        let decoded: CaptureManifest = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded, manifest);
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

        assert_eq!(frames.iter().map(|frame| frame.index).collect::<Vec<_>>(), vec![0, 1, 2]);
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
}
