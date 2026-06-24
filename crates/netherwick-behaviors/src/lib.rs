use anyhow::Result;
use netherwick_core::{BehaviorId, TimeMs};
use serde::{Deserialize, Serialize};

pub trait Behavior<I, O>: Send {
    fn id(&self) -> BehaviorId;
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

impl Default for BehaviorMode {
    fn default() -> Self {
        Self::Hardcoded
    }
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BehaviorComparison<O> {
    pub hardcoded: O,
    pub model: O,
    pub matched: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BehaviorRun<O> {
    pub chosen: O,
    pub comparison: Option<BehaviorComparison<O>>,
    pub fallback_used: bool,
    pub training_sample_emitted: bool,
}

impl<I, O> Replaceable<I, O>
where
    I: Clone,
    O: Clone + PartialEq,
{
    pub fn run(&mut self, input: &I, observed_at_ms: TimeMs) -> Result<BehaviorRun<O>> {
        match self.mode {
            BehaviorMode::Hardcoded => {
                let chosen = self.hardcoded.infer(input)?;
                Ok(BehaviorRun {
                    chosen,
                    comparison: None,
                    fallback_used: false,
                    training_sample_emitted: false,
                })
            }
            BehaviorMode::ShadowTrain => {
                let chosen = self.hardcoded.infer(input)?;
                if let Some(model) = self.model.as_mut() {
                    let sample = TrainingSample {
                        observed_at_ms,
                        input: input.clone(),
                        expected: chosen.clone(),
                    };
                    model.observe(&sample)?;
                }
                Ok(BehaviorRun {
                    chosen,
                    comparison: None,
                    fallback_used: false,
                    training_sample_emitted: self.model.is_some(),
                })
            }
            BehaviorMode::ShadowInfer => {
                let chosen = self.hardcoded.infer(input)?;
                let comparison = self
                    .model
                    .as_mut()
                    .and_then(|model| model.infer(input).ok())
                    .map(|model_output| BehaviorComparison {
                        hardcoded: chosen.clone(),
                        matched: chosen == model_output,
                        model: model_output,
                    });
                Ok(BehaviorRun {
                    chosen,
                    comparison,
                    fallback_used: false,
                    training_sample_emitted: false,
                })
            }
            BehaviorMode::ModelInfer => self.run_model_first(input, observed_at_ms, false),
            BehaviorMode::ModelTrainAndInfer => self.run_model_first(input, observed_at_ms, true),
            BehaviorMode::Compare => {
                let hardcoded = self.hardcoded.infer(input)?;
                let model = match self.model.as_mut() {
                    Some(model) => match model.infer(input) {
                        Ok(output) => output,
                        Err(error) => match self.fallback {
                            FallbackPolicy::HardcodedOnError => {
                                return Ok(BehaviorRun {
                                    chosen: hardcoded,
                                    comparison: None,
                                    fallback_used: true,
                                    training_sample_emitted: false,
                                });
                            }
                            FallbackPolicy::StopOnError => return Err(error),
                        },
                    },
                    None => hardcoded.clone(),
                };
                let matched = hardcoded == model;
                Ok(BehaviorRun {
                    chosen: hardcoded.clone(),
                    comparison: Some(BehaviorComparison {
                        hardcoded,
                        matched,
                        model,
                    }),
                    fallback_used: false,
                    training_sample_emitted: false,
                })
            }
        }
    }

    fn run_model_first(
        &mut self,
        input: &I,
        observed_at_ms: TimeMs,
        train_model: bool,
    ) -> Result<BehaviorRun<O>> {
        if let Some(model) = self.model.as_mut() {
            match model.infer(input) {
                Ok(chosen) => {
                    if train_model {
                        let target = self.hardcoded.infer(input)?;
                        let sample = TrainingSample {
                            observed_at_ms,
                            input: input.clone(),
                            expected: target,
                        };
                        model.observe(&sample)?;
                    }
                    return Ok(BehaviorRun {
                        chosen,
                        comparison: None,
                        fallback_used: false,
                        training_sample_emitted: train_model,
                    });
                }
                Err(error) => match self.fallback {
                    FallbackPolicy::HardcodedOnError => {
                        let chosen = self.hardcoded.infer(input)?;
                        return Ok(BehaviorRun {
                            chosen,
                            comparison: None,
                            fallback_used: true,
                            training_sample_emitted: false,
                        });
                    }
                    FallbackPolicy::StopOnError => return Err(error),
                },
            }
        }

        let chosen = self.hardcoded.infer(input)?;
        Ok(BehaviorRun {
            chosen,
            comparison: None,
            fallback_used: false,
            training_sample_emitted: false,
        })
    }
}
