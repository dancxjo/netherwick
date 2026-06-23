use anyhow::Result;
use netherwick_core::{BehaviorId, TimeMs};
use serde::{Deserialize, Serialize};

pub trait Behavior<I, O>: Send {
    fn id(&self) -> BehaviorId;
    fn mode(&self) -> BehaviorMode;
    fn infer(&mut self, input: &I) -> Result<O>;
    fn observe(&mut self, sample: &TrainingSample<I, O>) -> Result<()>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BehaviorMode {
    Hardcoded,
    ShadowTrain,
    ShadowInfer,
    ModelInfer,
    ModelTrainAndInfer,
    Compare,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FallbackPolicy {
    HardcodedOnError,
    StopOnError,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingSample<I, O> {
    pub observed_at_ms: TimeMs,
    pub input: I,
    pub expected: O,
}

pub struct Replaceable<I, O> {
    pub id: BehaviorId,
    pub mode: BehaviorMode,
    pub hardcoded: Box<dyn Behavior<I, O>>,
    pub model: Option<Box<dyn Behavior<I, O>>>,
    pub fallback: FallbackPolicy,
}
