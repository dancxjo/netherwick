use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use netherwick_actions::{ActionPrimitive, ReignInput, ReignOutcome};
use netherwick_core::Reward;
use netherwick_experience::{
    Experience, ExperienceLatent, FuturePrediction, Impression, RecalledExperience, Sensation,
};
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
    pub sensations: Vec<Sensation>,
    pub impressions: Vec<Impression>,
    pub experiences: Vec<Experience>,
    pub z: Option<ExperienceLatent>,
    pub chosen_action: Option<ActionPrimitive>,
    pub conscious_command: Option<ConsciousCommand>,
    pub reign_input: Option<ReignInput>,
    pub reign_outcome: Option<ReignOutcome>,
    pub predicted_futures: Vec<FuturePrediction>,
    pub actual_next: Option<Box<Now>>,
    pub reward: Reward,
    pub surprise: SurpriseSense,
    pub memory_recall: Vec<RecallHit>,
    pub recollections: Vec<RecalledExperience>,
    pub llm_teaching: Vec<LlmTeaching>,
    pub counterfactuals: Vec<CounterfactualAction>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceTransition {
    pub id: Uuid,
    pub before_frame_id: Uuid,
    pub before: Now,
    pub before_z: ExperienceLatent,
    pub action: Option<ActionPrimitive>,
    pub predicted_futures: Vec<FuturePrediction>,
    pub after: Now,
    pub after_z: ExperienceLatent,
    pub reward: Reward,
    pub surprise: SurpriseSense,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PendingFrame {
    pub frame_id: Uuid,
    pub now: Now,
    pub z: ExperienceLatent,
    pub action: Option<ActionPrimitive>,
    pub predicted_futures: Vec<FuturePrediction>,
}

#[derive(Clone, Debug, Default)]
pub struct TransitionBuilder {
    previous: Option<PendingFrame>,
}

impl TransitionBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe(
        &mut self,
        current: PendingFrame,
        reward: impl FnOnce(&PendingFrame, &PendingFrame) -> Reward,
        surprise: impl FnOnce(&PendingFrame, &PendingFrame) -> SurpriseSense,
    ) -> Option<ExperienceTransition> {
        let previous = self.previous.replace(current.clone())?;
        Some(ExperienceTransition {
            id: Uuid::new_v4(),
            before_frame_id: previous.frame_id,
            before: previous.now.clone(),
            before_z: previous.z.clone(),
            action: previous.action.clone(),
            predicted_futures: previous.predicted_futures.clone(),
            after: current.now.clone(),
            after_z: current.z.clone(),
            reward: reward(&previous, &current),
            surprise: surprise(&previous, &current),
            created_at_ms: current.now.t_ms,
        })
    }
}

impl ExperienceFrame {
    pub fn summary_text(&self) -> String {
        if let Some(experience) = self.experiences.last() {
            return experience.text.clone();
        }
        if let Some(impression) = self.impressions.last() {
            return impression.text.clone();
        }
        if let Some(transcript) = &self.now.ear.transcript {
            if !transcript.trim().is_empty() {
                return format!("I hear: {}", transcript.trim());
            }
        }
        if let Some(command) = &self.conscious_command {
            if !command.summary.trim().is_empty() {
                return command.summary.clone();
            }
        }
        if !self.notes.is_empty() {
            return self.notes.join(" ");
        }
        format!(
            "I am at t={}ms with battery {:.2}.",
            self.t_ms, self.now.body.battery_level
        )
    }
}

#[async_trait]
pub trait LedgerWriter {
    async fn append(&self, frame: &ExperienceFrame) -> Result<()>;
    async fn append_transition(&self, _transition: &ExperienceTransition) -> Result<()> {
        Ok(())
    }
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

    fn session_dir(&self) -> PathBuf {
        let date = Utc::now().format("%Y-%m-%d").to_string();
        self.root.join(date)
    }

    fn frames_path(&self) -> PathBuf {
        self.session_dir().join("frames.jsonl")
    }

    fn transitions_path(&self) -> PathBuf {
        self.session_dir().join("transitions.jsonl")
    }

    fn collect_frame_paths(root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        if !root.exists() {
            return Ok(());
        }
        for entry in fs::read_dir(root)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                Self::collect_frame_paths(&path, out)?;
            } else if path.file_name().and_then(|name| name.to_str()) == Some("frames.jsonl")
                || path.file_name().and_then(|name| name.to_str()) == Some("session.jsonl")
            {
                out.push(path);
            }
        }
        Ok(())
    }

    fn collect_transition_paths(root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        if !root.exists() {
            return Ok(());
        }
        for entry in fs::read_dir(root)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                Self::collect_transition_paths(&path, out)?;
            } else if path.file_name().and_then(|name| name.to_str()) == Some("transitions.jsonl") {
                out.push(path);
            }
        }
        Ok(())
    }

    async fn read_all(&self) -> Result<Vec<ExperienceFrame>> {
        let mut paths = Vec::new();
        Self::collect_frame_paths(&self.root, &mut paths)?;
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

    pub async fn transitions(&self) -> Result<Vec<ExperienceTransition>> {
        let mut paths = Vec::new();
        Self::collect_transition_paths(&self.root, &mut paths)?;
        paths.sort();

        let mut transitions = Vec::new();
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
                transitions.push(serde_json::from_str(&line)?);
            }
        }
        Ok(transitions)
    }
}

#[async_trait]
impl LedgerWriter for JsonlLedger {
    async fn append(&self, frame: &ExperienceFrame) -> Result<()> {
        let path = self.frames_path();
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
        file.write_all(b"\n").await?;
        Ok(())
    }

    async fn append_transition(&self, transition: &ExperienceTransition) -> Result<()> {
        let path = self.transitions_path();
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        let line = serde_json::to_string(transition)?;
        file.write_all(line.as_bytes()).await?;
        file.write_all(b"\n").await?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use netherwick_body::BodySense;

    #[test]
    fn transition_builder_pairs_second_frame_with_first() {
        let mut builder = TransitionBuilder::new();
        let first = PendingFrame {
            frame_id: Uuid::new_v4(),
            now: Now::blank(100, BodySense::default()),
            z: ExperienceLatent {
                t_ms: 100,
                z: vec![0.1],
                ..ExperienceLatent::default()
            },
            action: Some(ActionPrimitive::Stop),
            predicted_futures: vec![FuturePrediction {
                offset_ms: 1_000,
                predicted_z: vec![0.1],
                confidence: 0.5,
                summary: None,
            }],
        };
        let mut second_now = Now::blank(200, BodySense::default());
        second_now.body.battery_level = 0.9;
        let second = PendingFrame {
            frame_id: Uuid::new_v4(),
            now: second_now,
            z: ExperienceLatent {
                t_ms: 200,
                z: vec![0.2],
                ..ExperienceLatent::default()
            },
            action: Some(ActionPrimitive::Dock),
            predicted_futures: Vec::new(),
        };

        assert!(builder
            .observe(
                first.clone(),
                |_before, _after| Reward { value: 0.0 },
                |_before, _after| SurpriseSense::default()
            )
            .is_none());
        let transition = builder
            .observe(
                second.clone(),
                |_before, _after| Reward { value: 0.25 },
                |_before, _after| SurpriseSense {
                    schema_version: 1,
                    total: 0.5,
                    prediction_error: 0.4,
                },
            )
            .unwrap();

        assert_eq!(transition.before_frame_id, first.frame_id);
        assert_eq!(transition.before_z.z, vec![0.1]);
        assert_eq!(transition.after_z.z, vec![0.2]);
        assert_eq!(transition.action, Some(ActionPrimitive::Stop));
        assert_eq!(transition.reward.value, 0.25);
    }
}
