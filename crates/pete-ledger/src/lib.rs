use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use pete_actions::{ActionPrimitive, ReignInput, ReignOutcome};
use pete_behaviors::ErasedBehaviorRunRecord;
use pete_core::Reward;
use pete_experience::{
    EmbodiedContext, Experience, ExperienceInstant, ExperienceLatent, FutureInput,
    FuturePrediction, Impression, RecalledExperience, Sensation,
};
use pete_llm::{ConsciousCommand, CounterfactualAction, LlmTeaching};
use pete_now::{Now, RecallHit, SurpriseSense};
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
    #[serde(default)]
    pub behavior_runs: Vec<ErasedBehaviorRunRecord>,
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

pub fn future_input_from_transition(
    transition: &ExperienceTransition,
    offset_ms: u64,
) -> Option<FutureInput> {
    Some(FutureInput {
        latent: transition.before_z.clone(),
        action: transition.action.clone()?,
        offset_ms,
    })
}

pub fn future_target_from_transition(transition: &ExperienceTransition) -> Vec<f32> {
    transition.after_z.z.clone()
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
    pub fn embodied_context(&self) -> EmbodiedContext {
        self.experience_instant().embodied_context()
    }

    pub fn experience_instant(&self) -> ExperienceInstant {
        ExperienceInstant::from_parts(
            self.experiences.last(),
            &self.sensations,
            &self.impressions,
            &self.predicted_futures,
            &self.recollections,
            &self.now,
            self.chosen_action.clone(),
            Some(self.id.to_string()),
            "ledger-frame",
        )
    }

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

    pub async fn frames(&self) -> Result<Vec<ExperienceFrame>> {
        self.read_all().await
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

/// A loss-bounded ledger for long-running validation. It uses the production
/// ledger contracts while retaining a declared rolling replay window instead
/// of allowing diagnostic artifacts to consume the host filesystem.
#[derive(Clone, Debug)]
pub struct RollingLedger {
    root: PathBuf,
    max_frames: usize,
    max_transitions: usize,
}

impl RollingLedger {
    pub fn new(root: impl Into<PathBuf>, max_frames: usize, max_transitions: usize) -> Self {
        Self {
            root: root.into(),
            max_frames: max_frames.max(1),
            max_transitions: max_transitions.max(1),
        }
    }

    fn frames_dir(&self) -> PathBuf {
        self.root.join("rolling").join("frames")
    }

    fn transitions_dir(&self) -> PathBuf {
        self.root.join("rolling").join("transitions")
    }

    async fn write_bounded<T: Serialize>(
        directory: &Path,
        filename: String,
        maximum: usize,
        value: &T,
    ) -> Result<()> {
        tokio::fs::create_dir_all(directory).await?;
        let final_path = directory.join(filename);
        let temporary_path = final_path.with_extension("json.tmp");
        tokio::fs::write(&temporary_path, serde_json::to_vec(value)?).await?;
        tokio::fs::rename(temporary_path, final_path).await?;

        let mut entries = tokio::fs::read_dir(directory).await?;
        let mut paths = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) == Some("json") {
                paths.push(path);
            }
        }
        paths.sort();
        let remove_count = paths.len().saturating_sub(maximum);
        for path in paths.into_iter().take(remove_count) {
            tokio::fs::remove_file(path).await?;
        }
        Ok(())
    }

    async fn read_frames(&self) -> Result<Vec<ExperienceFrame>> {
        let directory = self.frames_dir();
        if !directory.exists() {
            return Ok(Vec::new());
        }
        let mut entries = tokio::fs::read_dir(directory).await?;
        let mut paths = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            paths.push(entry.path());
        }
        paths.sort();
        let mut frames = Vec::with_capacity(paths.len());
        for path in paths {
            frames.push(serde_json::from_slice(&tokio::fs::read(path).await?)?);
        }
        Ok(frames)
    }

    pub async fn frames(&self) -> Result<Vec<ExperienceFrame>> {
        self.read_frames().await
    }
}

#[async_trait]
impl LedgerWriter for RollingLedger {
    async fn append(&self, frame: &ExperienceFrame) -> Result<()> {
        Self::write_bounded(
            &self.frames_dir(),
            format!("frame-{:020}-{}.json", frame.t_ms, frame.id),
            self.max_frames,
            frame,
        )
        .await
    }

    async fn append_transition(&self, transition: &ExperienceTransition) -> Result<()> {
        Self::write_bounded(
            &self.transitions_dir(),
            format!(
                "transition-{:020}-{}.json",
                transition.created_at_ms, transition.id
            ),
            self.max_transitions,
            transition,
        )
        .await
    }
}

#[async_trait]
impl LedgerReader for RollingLedger {
    async fn recent(&self, limit: usize) -> Result<Vec<ExperienceFrame>> {
        let mut frames = self.read_frames().await?;
        if frames.len() > limit {
            frames.drain(0..frames.len() - limit);
        }
        Ok(frames)
    }

    async fn range(&self, start_ms: u64, end_ms: u64) -> Result<Vec<ExperienceFrame>> {
        Ok(self
            .read_frames()
            .await?
            .into_iter()
            .filter(|frame| frame.t_ms >= start_ms && frame.t_ms <= end_ms)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pete_body::BodySense;
    use pete_experience::{
        EmbodiedLineageEdge, Modality, SensationPayload, SensationPayloadKind, SensationSource,
        VectorEmbedding,
    };
    use serde_json::json;

    fn blank_frame(t_ms: u64) -> ExperienceFrame {
        ExperienceFrame {
            id: Uuid::new_v4(),
            t_ms,
            now: Now::blank(t_ms, BodySense::default()),
            sensations: Vec::new(),
            impressions: Vec::new(),
            experiences: Vec::new(),
            z: None,
            chosen_action: None,
            conscious_command: None,
            reign_input: None,
            reign_outcome: None,
            predicted_futures: Vec::new(),
            behavior_runs: Vec::new(),
            actual_next: None,
            reward: Reward::default(),
            surprise: SurpriseSense::default(),
            memory_recall: Vec::new(),
            recollections: Vec::new(),
            llm_teaching: Vec::new(),
            counterfactuals: Vec::new(),
            notes: Vec::new(),
        }
    }

    #[tokio::test]
    async fn rolling_ledger_retains_only_the_declared_replay_window() {
        let root = std::env::temp_dir().join(format!("pete-rolling-ledger-{}", Uuid::new_v4()));
        let ledger = RollingLedger::new(&root, 2, 2);
        for t_ms in [100, 200, 300] {
            ledger.append(&blank_frame(t_ms)).await.unwrap();
        }
        let frames = ledger.frames().await.unwrap();
        assert_eq!(
            frames.iter().map(|frame| frame.t_ms).collect::<Vec<_>>(),
            vec![200, 300]
        );
        assert_eq!(std::fs::read_dir(ledger.frames_dir()).unwrap().count(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }

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

    #[test]
    fn future_helpers_use_before_latent_action_offset_and_after_target() {
        let transition = ExperienceTransition {
            id: Uuid::new_v4(),
            before_frame_id: Uuid::new_v4(),
            before: Now::blank(100, BodySense::default()),
            before_z: ExperienceLatent {
                t_ms: 100,
                z: vec![0.1, 0.2],
                confidence: 0.8,
                ..ExperienceLatent::default()
            },
            action: Some(ActionPrimitive::Dock),
            predicted_futures: Vec::new(),
            after: Now::blank(200, BodySense::default()),
            after_z: ExperienceLatent {
                t_ms: 200,
                z: vec![0.3, 0.4],
                confidence: 0.9,
                ..ExperienceLatent::default()
            },
            reward: Reward { value: 0.0 },
            surprise: SurpriseSense::default(),
            created_at_ms: 200,
        };

        let input = future_input_from_transition(&transition, 500).unwrap();
        let target = future_target_from_transition(&transition);

        assert_eq!(input.latent.z, vec![0.1, 0.2]);
        assert_eq!(input.action, ActionPrimitive::Dock);
        assert_eq!(input.offset_ms, 500);
        assert_eq!(target, vec![0.3, 0.4]);
        assert_eq!(
            input.flat_features().len(),
            transition.before_z.z.len() + pete_experience::action_features(None).len() + 1
        );
    }

    #[test]
    fn embodied_context_tracks_primary_and_derived_sensation_lineage() {
        let primary = Sensation::primary(
            Modality::Vision,
            SensationSource::new("camera"),
            100,
            105,
            SensationPayload::image_metadata(64, 48, "rgb", 9_216),
        )
        .with_summary("I see a camera frame.");
        let child = Sensation::descendant(
            &primary,
            "vision.crop.focus",
            SensationPayloadKind::Crop,
            json!({"x": 2, "y": 3, "width": 12, "height": 10}),
            Default::default(),
            "focus",
        )
        .with_summary("I focus on part of the frame.")
        .with_vector(VectorEmbedding::new(
            vec![0.1, 0.2, 0.3],
            "crop-vectorizer.v0",
            Modality::Vision,
            SensationPayloadKind::Crop,
            primary.id,
            110,
        ));
        let impression = Impression::new(
            "vision.focus.impression",
            "I see a frame and focus on part of it.",
            vec![primary.id, child.id],
            100,
            110,
        );
        let experience = Experience::new(
            "embodied.now",
            "I see a frame and focus on part of it.",
            vec![impression.id],
            vec![primary.id, child.id],
            100,
            110,
        );
        let frame = ExperienceFrame {
            id: Uuid::new_v4(),
            t_ms: 110,
            now: Now::blank(110, BodySense::default()),
            sensations: vec![primary.clone(), child.clone()],
            impressions: vec![impression],
            experiences: vec![experience],
            z: None,
            chosen_action: None,
            conscious_command: None,
            reign_input: None,
            reign_outcome: None,
            predicted_futures: Vec::new(),
            behavior_runs: Vec::new(),
            actual_next: None,
            reward: Reward::default(),
            surprise: SurpriseSense::default(),
            memory_recall: Vec::new(),
            recollections: Vec::new(),
            llm_teaching: Vec::new(),
            counterfactuals: Vec::new(),
            notes: Vec::new(),
        };

        let context = frame.embodied_context();

        assert_eq!(context.sensations.len(), 2);
        assert_eq!(context.impressions.len(), 1);
        assert_eq!(
            context.lineage,
            vec![EmbodiedLineageEdge {
                parent_id: primary.id,
                child_id: child.id,
            }]
        );
        assert_eq!(context.sensation_vectors[0].model_id, "crop-vectorizer.v0");
        assert_eq!(context.sensation_vectors[0].dim, 3);
    }
}
