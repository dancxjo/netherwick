use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

pub type TimeMs = u64;
pub type BehaviorId = String;
pub type ModelId = String;
pub type SensorId = String;
pub type SensationId = Uuid;
pub type ImpressionId = Uuid;
pub type ExperienceId = Uuid;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VecF32(pub Vec<f32>);

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Confidence(pub f32);

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Pose2 {
    pub x_m: f32,
    pub y_m: f32,
    pub heading_rad: f32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Goal {
    Explore,
    Dock,
    Rest,
    Escape,
    Inspect,
    Approach,
    Speak,
}

impl Default for Goal {
    fn default() -> Self {
        Self::Explore
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Reward {
    pub value: f32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceKind {
    Direct,
    DerivedFromSensations { sensation_ids: Vec<SensationId> },
    DerivedFromImpressions { impression_ids: Vec<ImpressionId> },
    MemoryRecall { experience_id: ExperienceId },
}

impl Default for ProvenanceKind {
    fn default() -> Self {
        Self::Direct
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub kind: ProvenanceKind,
    pub stage_chain: Vec<String>,
    pub metadata: Value,
}

impl Provenance {
    pub fn direct() -> Self {
        Self::default()
    }

    pub fn derived_from_sensations(sensation_ids: impl IntoIterator<Item = SensationId>) -> Self {
        Self {
            kind: ProvenanceKind::DerivedFromSensations {
                sensation_ids: sensation_ids.into_iter().collect(),
            },
            ..Self::default()
        }
    }

    pub fn derived_from_impressions(
        impression_ids: impl IntoIterator<Item = ImpressionId>,
    ) -> Self {
        Self {
            kind: ProvenanceKind::DerivedFromImpressions {
                impression_ids: impression_ids.into_iter().collect(),
            },
            ..Self::default()
        }
    }

    pub fn memory_recall(experience_id: ExperienceId) -> Self {
        Self {
            kind: ProvenanceKind::MemoryRecall { experience_id },
            ..Self::default()
        }
    }

    pub fn with_stage(mut self, stage: impl Into<String>) -> Self {
        self.stage_chain.push(stage.into());
        self
    }
}
