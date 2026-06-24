use anyhow::Result;
use async_trait::async_trait;
use netherwick_actions::ActionPrimitive;
use netherwick_core::{Goal, Pose2};
use netherwick_experience::{Experience, RecalledExperience};
use netherwick_ledger::ExperienceFrame;
use netherwick_now::{
    GraphEdge, GraphEntity, MemorySense, Now, RecallHit, VectorArtifact, FACE_VECTOR_COLLECTION,
    SCENE_VECTOR_COLLECTION, VOICE_VECTOR_COLLECTION,
};
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
    #[serde(default)]
    pub scene_vectors: Vec<VectorArtifact>,
    pub face_vectors: Vec<Vec<f32>>,
    #[serde(default)]
    pub face_vector_artifacts: Vec<VectorArtifact>,
    pub voice_vectors: Vec<Vec<f32>>,
    #[serde(default)]
    pub voice_vector_artifacts: Vec<VectorArtifact>,
    pub battery: f32,
    pub active_goal: Option<Goal>,
    pub proposed_action: Option<ActionPrimitive>,
}

impl RecallQuery {
    pub fn from_now(now: &Now) -> Self {
        Self {
            now_text: format!("t_ms={} battery={:.2}", now.t_ms, now.body.battery_level),
            pose: Some(now.body.odometry),
            scene_vector: now
                .eye
                .scene_vectors
                .last()
                .map(|artifact| artifact.vector.clone()),
            scene_vectors: now.eye.scene_vectors.clone(),
            face_vectors: now.face.embeddings.clone(),
            face_vector_artifacts: now.face.vectors.clone(),
            voice_vectors: now.voice.embeddings.clone(),
            voice_vector_artifacts: now.voice.vectors.clone(),
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
    #[serde(default)]
    pub graph_entities: Vec<GraphEntity>,
    #[serde(default)]
    pub graph_relationships: Vec<GraphEdge>,
    #[serde(default)]
    pub scene_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub face_vectors: Vec<VectorArtifact>,
    #[serde(default)]
    pub voice_vectors: Vec<VectorArtifact>,
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
        let scene_vectors = scene_vectors_from_now(&frame.now, frame.id, frame.t_ms);
        let face_vectors = vector_artifacts_from_now(
            FACE_VECTOR_COLLECTION,
            &frame.now.face.vectors,
            &frame.now.face.embeddings,
            frame.id,
            frame.t_ms,
        );
        let voice_vectors = vector_artifacts_from_now(
            VOICE_VECTOR_COLLECTION,
            &frame.now.voice.vectors,
            &frame.now.voice.embeddings,
            frame.id,
            frame.t_ms,
        );
        let (graph_entities, graph_relationships) =
            graph_context_from_frame(frame, &scene_vectors, &face_vectors, &voice_vectors);
        let record = MemoryRecord {
            frame_id: frame.id,
            t_ms: frame.t_ms,
            summary: frame.summary_text(),
            graph_entities,
            graph_relationships,
            scene_vectors,
            face_vectors,
            voice_vectors,
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
        let mut remembered_entities = Vec::new();
        let mut remembered_relationships = Vec::new();

        for (index, (score, record)) in scored.into_iter().take(5).enumerate() {
            hits.push(RecallHit {
                frame_id: Some(record.frame_id),
                score,
                summary: record.summary.clone(),
                warning: record.warning.clone(),
                graph_context: scored_entities(&record, score),
            });
            let scene_score = max_vector_similarity(
                query_scene_vectors(&query),
                record.scene_vectors.iter().collect(),
            );
            let face_score = max_vector_similarity(
                query_face_vectors(&query),
                record.face_vectors.iter().collect(),
            );
            let voice_score = max_vector_similarity(
                query_voice_vectors(&query),
                record.voice_vectors.iter().collect(),
            );

            place_familiarity = place_familiarity.max(score).max(scene_score);
            if record.summary.to_ascii_lowercase().contains("danger") || record.warning.is_some() {
                place_danger = place_danger.max(score);
            }
            if matches!(record.active_goal, Some(Goal::Dock)) || record.summary.contains("charge") {
                place_charge_value = place_charge_value.max(score);
            }
            if has_face_query(&query) {
                face_familiarity = face_familiarity.max(score).max(face_score);
            }
            if has_voice_query(&query) {
                voice_familiarity = voice_familiarity.max(score).max(voice_score);
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
            remembered_entities.extend(scored_entities(&record, score));
            remembered_relationships.extend(record.graph_relationships.clone());
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
        remembered_entities = dedupe_entities(remembered_entities, 12);
        remembered_relationships = dedupe_relationships(remembered_relationships, 16);
        let graph_context_summary = graph_context_summary(&remembered_entities);

        let sense = MemorySense {
            place_familiarity,
            place_danger,
            place_charge_value,
            face_familiarity,
            voice_familiarity,
            similar_situation_count: hits.len().try_into().unwrap_or(u16::MAX),
            best_remembered_action,
            remembered_warning,
            remembered_entities,
            remembered_relationships,
            graph_context_summary,
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
    let vector_score = max_vector_similarity(query_all_vectors(query), record_all_vectors(&record));
    let score = (overlap * 0.4)
        + (battery_score * 0.15)
        + (goal_score * 0.15)
        + (action_score * 0.1)
        + (vector_score * 0.2);
    if score <= 0.05 {
        return None;
    }
    Some((score, record))
}

fn graph_context_from_frame(
    frame: &ExperienceFrame,
    scene_vectors: &[VectorArtifact],
    face_vectors: &[VectorArtifact],
    voice_vectors: &[VectorArtifact],
) -> (Vec<GraphEntity>, Vec<GraphEdge>) {
    let frame_id = frame.id.to_string();
    let experience_id = format!("experience:{frame_id}");
    let mut entities = vec![GraphEntity {
        id: experience_id.clone(),
        labels: vec!["Experience".to_string(), "Memory".to_string()],
        summary: frame.summary_text(),
        score: 1.0,
    }];
    let mut relationships = Vec::new();

    let pose = frame.now.body.odometry;
    let place_id = place_id_for_pose(pose);
    entities.push(GraphEntity {
        id: place_id.clone(),
        labels: vec!["Place".to_string()],
        summary: format!("place near x={:.1}m y={:.1}m", pose.x_m, pose.y_m),
        score: 1.0,
    });
    relationships.push(GraphEdge {
        from: experience_id.clone(),
        to: place_id,
        relationship: "OCCURRED_AT".to_string(),
        summary: None,
    });

    for artifact in scene_vectors {
        let vector_id = vector_node_id(artifact);
        entities.push(vector_entity(artifact, "scene"));
        relationships.push(GraphEdge {
            from: experience_id.clone(),
            to: vector_id,
            relationship: "HAS_SCENE_VECTOR".to_string(),
            summary: None,
        });
    }

    for (index, artifact) in face_vectors.iter().enumerate() {
        let person_id = artifact
            .source_id
            .clone()
            .unwrap_or_else(|| format!("person:face:{frame_id}:{index}"));
        entities.push(GraphEntity {
            id: person_id.clone(),
            labels: vec![
                "Person".to_string(),
                "FaceInstance".to_string(),
                "Entity".to_string(),
            ],
            summary: "person seen by face vector".to_string(),
            score: 1.0,
        });
        entities.push(vector_entity(artifact, "face"));
        relationships.push(GraphEdge {
            from: experience_id.clone(),
            to: person_id.clone(),
            relationship: "SAW_PERSON".to_string(),
            summary: None,
        });
        relationships.push(GraphEdge {
            from: person_id,
            to: vector_node_id(artifact),
            relationship: "HAS_FACE_VECTOR".to_string(),
            summary: None,
        });
    }

    for (index, artifact) in voice_vectors.iter().enumerate() {
        let person_id = artifact
            .source_id
            .clone()
            .unwrap_or_else(|| format!("person:voice:{frame_id}:{index}"));
        entities.push(GraphEntity {
            id: person_id.clone(),
            labels: vec![
                "Person".to_string(),
                "VoiceSignature".to_string(),
                "Entity".to_string(),
            ],
            summary: "person heard by voice vector".to_string(),
            score: 1.0,
        });
        entities.push(vector_entity(artifact, "voice"));
        relationships.push(GraphEdge {
            from: experience_id.clone(),
            to: person_id.clone(),
            relationship: "HEARD_PERSON".to_string(),
            summary: None,
        });
        relationships.push(GraphEdge {
            from: person_id,
            to: vector_node_id(artifact),
            relationship: "HAS_VOICE_VECTOR".to_string(),
            summary: None,
        });
    }

    (
        dedupe_entities(entities, usize::MAX),
        dedupe_relationships(relationships, usize::MAX),
    )
}

fn place_id_for_pose(pose: Pose2) -> String {
    format!(
        "place:grid:{:.0}:{:.0}",
        (pose.x_m * 2.0).round(),
        (pose.y_m * 2.0).round()
    )
}

fn vector_node_id(artifact: &VectorArtifact) -> String {
    format!("vector:{}:{}", artifact.collection, artifact.point_id)
}

fn vector_entity(artifact: &VectorArtifact, kind: &str) -> GraphEntity {
    GraphEntity {
        id: vector_node_id(artifact),
        labels: vec!["Vector".to_string()],
        summary: format!(
            "{kind} vector in {} with {} dimensions",
            artifact.collection,
            artifact.vector.len()
        ),
        score: 1.0,
    }
}

fn scored_entities(record: &MemoryRecord, score: f32) -> Vec<GraphEntity> {
    record
        .graph_entities
        .iter()
        .filter(|entity| !entity.has_label("Vector"))
        .map(|entity| {
            let mut entity = entity.clone();
            entity.score = score;
            entity
        })
        .collect()
}

fn dedupe_entities(entities: Vec<GraphEntity>, limit: usize) -> Vec<GraphEntity> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for entity in entities {
        if seen.insert(entity.id.clone()) {
            out.push(entity);
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}

fn dedupe_relationships(relationships: Vec<GraphEdge>, limit: usize) -> Vec<GraphEdge> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for relationship in relationships {
        let key = format!(
            "{}:{}:{}",
            relationship.from, relationship.relationship, relationship.to
        );
        if seen.insert(key) {
            out.push(relationship);
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}

fn graph_context_summary(entities: &[GraphEntity]) -> Option<String> {
    let people = entities
        .iter()
        .filter(|entity| entity.has_label("Person"))
        .count();
    let places = entities
        .iter()
        .filter(|entity| entity.has_label("Place"))
        .count();
    let experiences = entities
        .iter()
        .filter(|entity| entity.has_label("Experience"))
        .count();
    if people == 0 && places == 0 && experiences == 0 {
        return None;
    }
    Some(format!(
        "Graph recall: {people} person nodes, {places} place nodes, {experiences} experience nodes."
    ))
}

fn scene_vectors_from_now(now: &Now, frame_id: uuid::Uuid, t_ms: u64) -> Vec<VectorArtifact> {
    if !now.eye.scene_vectors.is_empty() {
        return now.eye.scene_vectors.clone();
    }
    now.eye
        .frames
        .last()
        .map(|vector| {
            VectorArtifact::new(
                SCENE_VECTOR_COLLECTION,
                format!(
                    "{}:scene:{}",
                    frame_id,
                    now.eye.frames.len().saturating_sub(1)
                ),
                vector.clone(),
            )
            .with_source_frame_id(frame_id.to_string())
            .with_occurred_at_ms(t_ms)
        })
        .into_iter()
        .collect()
}

fn vector_artifacts_from_now(
    collection: &str,
    artifacts: &[VectorArtifact],
    legacy_embeddings: &[Vec<f32>],
    frame_id: uuid::Uuid,
    t_ms: u64,
) -> Vec<VectorArtifact> {
    if !artifacts.is_empty() {
        return artifacts.to_vec();
    }
    legacy_embeddings
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            VectorArtifact::new(
                collection,
                format!("{frame_id}:{collection}:{index}"),
                vector.clone(),
            )
            .with_source_frame_id(frame_id.to_string())
            .with_occurred_at_ms(t_ms)
        })
        .collect()
}

fn query_scene_vectors(query: &RecallQuery) -> Vec<&[f32]> {
    let mut vectors = query
        .scene_vectors
        .iter()
        .map(|artifact| artifact.vector.as_slice())
        .collect::<Vec<_>>();
    if let Some(vector) = &query.scene_vector {
        vectors.push(vector.as_slice());
    }
    vectors
}

fn query_face_vectors(query: &RecallQuery) -> Vec<&[f32]> {
    let mut vectors = query
        .face_vector_artifacts
        .iter()
        .map(|artifact| artifact.vector.as_slice())
        .collect::<Vec<_>>();
    vectors.extend(query.face_vectors.iter().map(Vec::as_slice));
    vectors
}

fn query_voice_vectors(query: &RecallQuery) -> Vec<&[f32]> {
    let mut vectors = query
        .voice_vector_artifacts
        .iter()
        .map(|artifact| artifact.vector.as_slice())
        .collect::<Vec<_>>();
    vectors.extend(query.voice_vectors.iter().map(Vec::as_slice));
    vectors
}

fn query_all_vectors(query: &RecallQuery) -> Vec<&[f32]> {
    let mut vectors = query_scene_vectors(query);
    vectors.extend(query_face_vectors(query));
    vectors.extend(query_voice_vectors(query));
    vectors
}

fn record_all_vectors(record: &MemoryRecord) -> Vec<&VectorArtifact> {
    record
        .scene_vectors
        .iter()
        .chain(record.face_vectors.iter())
        .chain(record.voice_vectors.iter())
        .collect()
}

fn has_face_query(query: &RecallQuery) -> bool {
    !query.face_vectors.is_empty() || !query.face_vector_artifacts.is_empty()
}

fn has_voice_query(query: &RecallQuery) -> bool {
    !query.voice_vectors.is_empty() || !query.voice_vector_artifacts.is_empty()
}

fn max_vector_similarity(query_vectors: Vec<&[f32]>, record_vectors: Vec<&VectorArtifact>) -> f32 {
    query_vectors
        .into_iter()
        .flat_map(|query| {
            record_vectors
                .iter()
                .map(move |record| cosine_similarity(query, &record.vector))
        })
        .fold(0.0f32, f32::max)
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.is_empty() || right.is_empty() || left.len() != right.len() {
        return 0.0;
    }
    let (mut dot, mut left_norm, mut right_norm) = (0.0f32, 0.0f32, 0.0f32);
    for (left, right) in left.iter().zip(right) {
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm <= f32::EPSILON || right_norm <= f32::EPSILON {
        return 0.0;
    }
    (dot / (left_norm.sqrt() * right_norm.sqrt())).clamp(0.0, 1.0)
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

#[cfg(test)]
mod tests {
    use super::*;
    use netherwick_body::BodySense;
    use netherwick_ledger::ExperienceFrame;
    use netherwick_now::{
        FaceSense, SurpriseSense, VectorArtifact, FACE_VECTOR_COLLECTION, SCENE_VECTOR_COLLECTION,
    };

    fn empty_frame(now: Now) -> ExperienceFrame {
        ExperienceFrame {
            id: uuid::Uuid::new_v4(),
            t_ms: now.t_ms,
            now,
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
            reward: Default::default(),
            surprise: SurpriseSense::default(),
            memory_recall: Vec::new(),
            recollections: Vec::new(),
            llm_teaching: Vec::new(),
            counterfactuals: Vec::new(),
            notes: vec!["saw a familiar person".to_string()],
        }
    }

    #[tokio::test]
    async fn store_preserves_typed_vector_artifacts() {
        let mut now = Now::blank(123, BodySense::default());
        now.face = FaceSense {
            schema_version: 1,
            embeddings: Vec::new(),
            vectors: vec![
                VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-1", vec![1.0, 0.0])
                    .with_source_id("face-crop-1"),
            ],
        };
        now.eye.scene_vectors.push(VectorArtifact::new(
            SCENE_VECTOR_COLLECTION,
            "scene-1",
            vec![0.0, 1.0],
        ));

        let store = InMemoryExperienceStore::new();
        store.store(&empty_frame(now)).await.unwrap();

        let snapshot = store.snapshot();
        assert_eq!(snapshot[0].face_vectors[0].point_id, "face-1");
        assert_eq!(
            snapshot[0].face_vectors[0].source_id.as_deref(),
            Some("face-crop-1")
        );
        assert_eq!(
            snapshot[0].scene_vectors[0].collection,
            SCENE_VECTOR_COLLECTION
        );
        assert!(snapshot[0]
            .graph_entities
            .iter()
            .any(|entity| entity.has_label("Person")));
        assert!(snapshot[0]
            .graph_entities
            .iter()
            .any(|entity| entity.has_label("Place")));
        assert!(snapshot[0]
            .graph_relationships
            .iter()
            .any(|edge| edge.relationship == "HAS_FACE_VECTOR"));
    }

    #[tokio::test]
    async fn recall_returns_graph_context_as_memory_sense() {
        let mut now = Now::blank(123, BodySense::default());
        now.face.vectors =
            vec![
                VectorArtifact::new(FACE_VECTOR_COLLECTION, "face-1", vec![1.0, 0.0])
                    .with_source_id("person:ada"),
            ];

        let store = InMemoryExperienceStore::new();
        store.store(&empty_frame(now)).await.unwrap();

        let recall = store
            .recall(RecallQuery {
                face_vector_artifacts: vec![VectorArtifact::new(
                    FACE_VECTOR_COLLECTION,
                    "query-face",
                    vec![1.0, 0.0],
                )],
                battery: 1.0,
                ..RecallQuery::default()
            })
            .await
            .unwrap();

        assert!(recall
            .sense
            .remembered_entities
            .iter()
            .any(|entity| entity.id == "person:ada" && entity.has_label("Person")));
        assert!(recall
            .sense
            .remembered_entities
            .iter()
            .any(|entity| entity.has_label("Place")));
        assert!(recall.sense.graph_context_summary.is_some());
        assert!(recall.hits[0]
            .graph_context
            .iter()
            .any(|entity| entity.has_label("Person")));
    }

    #[test]
    fn legacy_embeddings_become_collection_artifacts() {
        let frame_id = uuid::Uuid::new_v4();
        let artifacts = vector_artifacts_from_now(
            FACE_VECTOR_COLLECTION,
            &[],
            &[vec![0.25, 0.75]],
            frame_id,
            99,
        );

        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].collection, FACE_VECTOR_COLLECTION);
        assert_eq!(artifacts[0].vector, vec![0.25, 0.75]);
        let expected_frame_id = frame_id.to_string();
        assert_eq!(
            artifacts[0].source_frame_id.as_deref(),
            Some(expected_frame_id.as_str())
        );
    }

    #[test]
    fn vector_similarity_scores_matching_vectors() {
        let query = vec![vec![1.0, 0.0]];
        let records = vec![VectorArtifact::new(
            FACE_VECTOR_COLLECTION,
            "face-1",
            vec![1.0, 0.0],
        )];
        let score = max_vector_similarity(
            query.iter().map(Vec::as_slice).collect(),
            records.iter().collect(),
        );

        assert!((score - 1.0).abs() < f32::EPSILON);
    }
}
