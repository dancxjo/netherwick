use netherwick_core::{
    ExperienceId, ImpressionId, Provenance, SensationId, TimeMs,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExperienceLatent {
    pub t_ms: TimeMs,
    pub z: Vec<f32>,
    pub reconstruction_error: f32,
    pub prediction_error: f32,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FuturePrediction {
    pub offset_ms: TimeMs,
    pub predicted_z: Vec<f32>,
    pub confidence: f32,
    pub summary: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sensation {
    pub id: SensationId,
    pub kind: String,
    pub source: String,
    pub occurred_at_ms: TimeMs,
    pub observed_at_ms: TimeMs,
    pub summary: Option<String>,
    pub provenance: Provenance,
    pub payload: Value,
}

impl Sensation {
    pub fn new(
        kind: impl Into<String>,
        source: impl Into<String>,
        occurred_at_ms: TimeMs,
        observed_at_ms: TimeMs,
        payload: Value,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            kind: kind.into(),
            source: source.into(),
            occurred_at_ms,
            observed_at_ms,
            summary: None,
            provenance: Provenance::direct(),
            payload,
        }
    }

    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    pub fn with_provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = provenance;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Impression {
    pub id: ImpressionId,
    pub kind: String,
    pub text: String,
    pub about: Vec<SensationId>,
    pub occurred_at_ms: TimeMs,
    pub observed_at_ms: TimeMs,
    pub confidence: f32,
    pub payload: Value,
}

impl Impression {
    pub fn new(
        kind: impl Into<String>,
        text: impl Into<String>,
        about: Vec<SensationId>,
        occurred_at_ms: TimeMs,
        observed_at_ms: TimeMs,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            kind: kind.into(),
            text: text.into(),
            about,
            occurred_at_ms,
            observed_at_ms,
            confidence: 0.5,
            payload: Value::Null,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Experience {
    pub id: ExperienceId,
    pub kind: String,
    pub text: String,
    pub impression_ids: Vec<ImpressionId>,
    pub sensation_ids: Vec<SensationId>,
    pub occurred_at_ms: TimeMs,
    pub observed_at_ms: TimeMs,
    pub salience: f32,
    pub tags: Vec<String>,
    pub payload: Value,
}

impl Experience {
    pub fn new(
        kind: impl Into<String>,
        text: impl Into<String>,
        impression_ids: Vec<ImpressionId>,
        sensation_ids: Vec<SensationId>,
        occurred_at_ms: TimeMs,
        observed_at_ms: TimeMs,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            kind: kind.into(),
            text: text.into(),
            impression_ids,
            sensation_ids,
            occurred_at_ms,
            observed_at_ms,
            salience: 0.5,
            tags: Vec::new(),
            payload: Value::Null,
        }
    }

    pub fn to_recall_sensation(
        &self,
        recall_at_ms: TimeMs,
        score: f32,
        stage: impl Into<String>,
    ) -> Sensation {
        let payload = json!({
            "experience": self,
            "original_experience_id": self.id,
            "original_occurred_at_ms": self.occurred_at_ms,
            "original_observed_at_ms": self.observed_at_ms,
            "score": score,
        });
        Sensation::new(
            "memory.related_experience",
            "memory",
            recall_at_ms,
            recall_at_ms,
            payload,
        )
        .with_summary(format!("I remember: {}", self.text))
        .with_provenance(Provenance::memory_recall(self.id).with_stage(stage))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecalledExperience {
    pub score: f32,
    pub experience: Experience,
    pub sensation: Sensation,
}
