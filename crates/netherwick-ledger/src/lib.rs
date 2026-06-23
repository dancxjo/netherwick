use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use netherwick_actions::ActionPrimitive;
use netherwick_core::Reward;
use netherwick_experience::{ExperienceLatent, FuturePrediction};
use netherwick_llm::{ConsciousCommand, CounterfactualAction, LlmTeaching};
use netherwick_now::{Now, RecallHit, SurpriseSense};
use serde::{Deserialize, Serialize};
use tokio::fs::OpenOptions;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use uuid::Uuid;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceFrame {
    pub id: Uuid,
    pub t_ms: u64,
    pub now: Now,
    pub z: Option<ExperienceLatent>,
    pub chosen_action: Option<ActionPrimitive>,
    pub conscious_command: Option<ConsciousCommand>,
    pub predicted_futures: Vec<FuturePrediction>,
    pub actual_next: Option<Box<Now>>,
    pub reward: Reward,
    pub surprise: SurpriseSense,
    pub memory_recall: Vec<RecallHit>,
    pub llm_teaching: Vec<LlmTeaching>,
    pub counterfactuals: Vec<CounterfactualAction>,
    pub notes: Vec<String>,
}

#[async_trait]
pub trait LedgerWriter {
    async fn append(&self, frame: &ExperienceFrame) -> Result<()>;
}

#[async_trait]
pub trait LedgerReader {
    async fn recent(&self, limit: usize) -> Result<Vec<ExperienceFrame>>;
    async fn range(&self, start_ms: u64, end_ms: u64) -> Result<Vec<ExperienceFrame>>;
}

#[derive(Clone, Debug)]
pub struct JsonlLedger {
    root: PathBuf,
}

impl JsonlLedger {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn session_path(&self) -> PathBuf {
        let date = Utc::now().format("%Y-%m-%d").to_string();
        self.root.join(date).join("session.jsonl")
    }

    fn collect_paths(root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        if !root.exists() {
            return Ok(());
        }
        for entry in fs::read_dir(root)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                Self::collect_paths(&path, out)?;
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("jsonl") {
                out.push(path);
            }
        }
        Ok(())
    }

    async fn read_all(&self) -> Result<Vec<ExperienceFrame>> {
        let mut paths = Vec::new();
        Self::collect_paths(&self.root, &mut paths)?;
        paths.sort();

        let mut frames = Vec::new();
        for path in paths {
            let file = match tokio::fs::File::open(&path).await {
                Ok(file) => file,
                Err(_) => continue,
            };
            let mut lines = BufReader::new(file).lines();
            while let Some(line) = lines.next_line().await? {
                if line.trim().is_empty() {
                    continue;
                }
                frames.push(serde_json::from_str(&line)?);
            }
        }
        Ok(frames)
    }
}

#[async_trait]
impl LedgerWriter for JsonlLedger {
    async fn append(&self, frame: &ExperienceFrame) -> Result<()> {
        let path = self.session_path();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        let line = serde_json::to_string(frame)?;
        file.write_all(line.as_bytes()).await?;
        file.write_all(
            b"
",
        )
        .await?;
        Ok(())
    }
}

#[async_trait]
impl LedgerReader for JsonlLedger {
    async fn recent(&self, limit: usize) -> Result<Vec<ExperienceFrame>> {
        let mut frames = self.read_all().await?;
        if frames.len() > limit {
            frames.drain(0..frames.len() - limit);
        }
        Ok(frames)
    }

    async fn range(&self, start_ms: u64, end_ms: u64) -> Result<Vec<ExperienceFrame>> {
        let frames = self
            .read_all()
            .await?
            .into_iter()
            .filter(|frame| frame.t_ms >= start_ms && frame.t_ms <= end_ms)
            .collect();
        Ok(frames)
    }
}
