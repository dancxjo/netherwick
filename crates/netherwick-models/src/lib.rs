use anyhow::Result;
use netherwick_behaviors::TrainingSample;
use serde::{Deserialize, Serialize};

pub trait NeuralModel<I, O> {
    fn predict(&self, input: I) -> Result<O>;
}

pub trait OnlineTrainer<I, O> {
    fn train_step(&mut self, sample: TrainingSample<I, O>) -> Result<TrainStats>;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TrainStats {
    pub loss: f32,
    pub samples_seen: u64,
    pub improved: bool,
}

pub const MODEL_REGISTRY: &[&str] = &[
    "ExperienceEncoder",
    "ExperienceDecoder",
    "FuturePredictor",
    "EyeNextPredictor",
    "EarNextPredictor",
    "DangerPredictor",
    "ChargePredictor",
    "ActionValueNet",
    "SalienceNet",
    "GoalArbiterNet",
    "MemoryConsolidationNet",
    "FaceFamiliarityNet",
    "VoiceFamiliarityNet",
    "ConductorNet",
];
