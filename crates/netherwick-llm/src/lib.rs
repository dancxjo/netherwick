use anyhow::Result;
use async_trait::async_trait;
use netherwick_actions::ActionPrimitive;
use netherwick_experience::{ExperienceLatent, FuturePrediction};
use netherwick_now::{LlmSense, Now};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConsciousCommand {
    pub summary: String,
    pub action: Option<ActionPrimitive>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmTeaching {
    pub t_ms: u64,
    pub summary: String,
    pub critique: Option<String>,
    pub counterfactuals: Vec<CounterfactualAction>,
    pub memory_notes: Vec<String>,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CounterfactualAction {
    pub instead_of: Option<ActionPrimitive>,
    pub proposed: ActionPrimitive,
    pub reason: String,
    pub weight: f32,
}

impl Default for CounterfactualAction {
    fn default() -> Self {
        Self {
            instead_of: None,
            proposed: ActionPrimitive::Stop,
            reason: String::new(),
            weight: 0.0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmTickResult {
    pub sense: LlmSense,
    pub conscious_command: Option<ConsciousCommand>,
    pub teaching: Vec<LlmTeaching>,
}

#[async_trait]
pub trait LlmAgent {
    async fn maybe_tick(
        &mut self,
        now: &Now,
        z: &ExperienceLatent,
        futures: &[FuturePrediction],
        recall_summary: &str,
    ) -> Result<LlmTickResult>;
}

#[derive(Default)]
pub struct NoopLlmAgent;

#[async_trait]
impl LlmAgent for NoopLlmAgent {
    async fn maybe_tick(
        &mut self,
        _now: &Now,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
    ) -> Result<LlmTickResult> {
        Ok(LlmTickResult::default())
    }
}
