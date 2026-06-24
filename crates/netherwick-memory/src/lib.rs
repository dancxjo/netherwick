use anyhow::Result;
use async_trait::async_trait;
use netherwick_actions::ActionPrimitive;
use netherwick_core::{Goal, Pose2};
use netherwick_experience::{Experience, RecalledExperience};
use netherwick_ledger::ExperienceFrame;
use netherwick_now::{MemorySense, Now, RecallHit};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

#[async_trait]
pub trait MemoryStore {
    async fn store(&self, frame: &ExperienceFrame) -> Result<()>;
}

#[async_trait]
pub trait Recall {
    async fn recall(&self, query: RecallQuery) -> Result<RecallBundle>;
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RecallQuery {
    pub now_text: String,
    pub pose: Option<Pose2>,
    pub scene_vector: Option<Vec<f32>>,
    pub face_vectors: Vec<Vec<f32>>,
    pub voice_vectors: Vec<Vec<f32>>,
    pub battery: f32,
    pub active_goal: Option<Goal>,
    pub proposed_action: Option<ActionPrimitive>,
}

impl RecallQuery {
    pub fn from_now(now: &Now) -> Self {
        Self {
            now_text: format!("t_ms={} battery={:.2}", now.t_ms, now.body.battery_level),
            pose: Some(now.body.odometry),
            scene_vector: None,
            face_vectors: now.face.embeddings.clone(),
            voice_vectors: now.voice.embeddings.clone(),
            battery: now.body.battery_level,
            active_goal: now
                .self_sense
                .active_goal
                .as_ref()
                .map(|goal| match goal.as_str() {
                    "dock" => Goal::Dock,
                    "rest" => Goal::Rest,
                    "escape" => Goal::Escape,
                    "inspect" => Goal::Inspect,
                    "approach" => Goal::Approach,
                    "speak" => Goal::Speak,
                    _ => Goal::Explore,
                }),
            proposed_action: now.memory.best_remembered_action.clone(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RecallBundle {
    pub hits: Vec<RecallHit>,
    pub sense: MemorySense,
    pub first_person_summary: String,
    pub recollections: Vec<RecalledExperience>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub frame_id: uuid::Uuid,
    pub t_ms: u64,
    pub summary: String,
    pub battery: f32,
    pub active_goal: Option<Goal>,
    pub chosen_action: Option<ActionPrimitive>,
    pub warning: Option<String>,
    pub experience: Option<Experience>,
}

#[derive(Clone, Default)]
pub struct InMemoryExperienceStore {
    records: Arc<Mutex<Vec<MemoryRecord>>>,
}

impl InMemoryExperienceStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> Vec<MemoryRecord> {
        self.records.lock().expect("memory mutex poisoned").clone()
    }
}

#[async_trait]
impl MemoryStore for InMemoryExperienceStore {
    async fn store(&self, frame: &ExperienceFrame) -> Result<()> {
        let warning = frame
            .memory_recall
            .iter()
            .find_map(|hit| hit.warning.clone())
            .or_else(|| frame.now.memory.remembered_warning.clone());
        let record = MemoryRecord {
            frame_id: frame.id,
            t_ms: frame.t_ms,
            summary: frame.summary_text(),
            battery: frame.now.body.battery_level,
            active_goal: RecallQuery::from_now(&frame.now).active_goal,
            chosen_action: frame.chosen_action.clone(),
            warning,
            experience: frame.experiences.last().cloned(),
        };
        self.records
            .lock()
            .expect("memory mutex poisoned")
            .push(record);
        Ok(())
    }
}

#[async_trait]
impl Recall for InMemoryExperienceStore {
    async fn recall(&self, query: RecallQuery) -> Result<RecallBundle> {
        let records = self.snapshot();
        let mut scored = records
            .into_iter()
            .filter_map(|record| score_record(&query, record))
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.1.t_ms.cmp(&left.1.t_ms))
        });

        let mut hits = Vec::new();
        let mut recollections = Vec::new();
        let mut seen_actions = BTreeSet::new();
        let mut place_familiarity = 0.0f32;
        let mut place_danger = 0.0f32;
        let mut place_charge_value = 0.0f32;
        let mut face_familiarity = 0.0f32;
        let mut voice_familiarity = 0.0f32;
        let mut remembered_warning = None;
        let mut best_remembered_action = None;

        for (index, (score, record)) in scored.into_iter().take(5).enumerate() {
            hits.push(RecallHit {
                frame_id: Some(record.frame_id),
                score,
                summary: record.summary.clone(),
                warning: record.warning.clone(),
            });
            place_familiarity = place_familiarity.max(score);
            if record.summary.to_ascii_lowercase().contains("danger") || record.warning.is_some() {
                place_danger = place_danger.max(score);
            }
            if matches!(record.active_goal, Some(Goal::Dock)) || record.summary.contains("charge") {
                place_charge_value = place_charge_value.max(score);
            }
            if !query.face_vectors.is_empty() {
                face_familiarity = face_familiarity.max(score);
            }
            if !query.voice_vectors.is_empty() {
                voice_familiarity = voice_familiarity.max(score);
            }
            if remembered_warning.is_none() {
                remembered_warning = record.warning.clone();
            }
            if best_remembered_action.is_none() {
                if let Some(action) = &record.chosen_action {
                    let key = format!("{action:?}");
                    if seen_actions.insert(key) {
                        best_remembered_action = Some(action.clone());
                    }
                }
            }
            if let Some(experience) = record.experience {
                let sensation = experience.to_recall_sensation(
                    query_pose_time_hint(&query, index as u64),
                    score,
                    "memory-recall",
                );
                recollections.push(RecalledExperience {
                    score,
                    experience,
                    sensation,
                });
            }
        }

        let sense = MemorySense {
            place_familiarity,
            place_danger,
            place_charge_value,
            face_familiarity,
            voice_familiarity,
            similar_situation_count: hits.len().try_into().unwrap_or(u16::MAX),
            best_remembered_action,
            remembered_warning,
        };
        let first_person_summary = if hits.is_empty() {
            "I do not remember a similar situation yet.".to_string()
        } else {
            format!(
                "I remember {} similar moments. The closest one was: {}",
                hits.len(),
                hits[0].summary
            )
        };

        Ok(RecallBundle {
            hits,
            sense,
            first_person_summary,
            recollections,
        })
    }
}

fn score_record(query: &RecallQuery, record: MemoryRecord) -> Option<(f32, MemoryRecord)> {
    let query_tokens = tokenize(&query.now_text);
    let summary_tokens = tokenize(&record.summary);
    let overlap = token_overlap(&query_tokens, &summary_tokens);
    let battery_distance = (query.battery - record.battery).abs();
    let battery_score = (1.0 - battery_distance).clamp(0.0, 1.0);
    let goal_score = if query.active_goal.is_some() && query.active_goal == record.active_goal {
        1.0
    } else {
        0.0
    };
    let action_score =
        if query.proposed_action.is_some() && query.proposed_action == record.chosen_action {
            0.8
        } else {
            0.0
        };
    let score = (overlap * 0.5) + (battery_score * 0.2) + (goal_score * 0.2) + (action_score * 0.1);
    if score <= 0.05 {
        return None;
    }
    Some((score, record))
}

fn tokenize(text: &str) -> BTreeSet<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_ascii_lowercase())
        .collect()
}

fn token_overlap(left: &BTreeSet<String>, right: &BTreeSet<String>) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let shared = left.intersection(right).count() as f32;
    let total = left.union(right).count() as f32;
    if total <= f32::EPSILON {
        0.0
    } else {
        shared / total
    }
}

fn query_pose_time_hint(query: &RecallQuery, ordinal: u64) -> u64 {
    let pose_hint = query
        .pose
        .map(|pose| ((pose.x_m.abs() + pose.y_m.abs()) * 100.0) as u64)
        .unwrap_or(0);
    pose_hint.saturating_add(ordinal)
}
