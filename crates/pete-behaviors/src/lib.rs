use anyhow::{anyhow, Result};
use pete_core::TimeMs;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Instant;

pub trait FunctionBehavior<I, O>: Send {
    fn id(&self) -> &'static str;

    fn infer(&mut self, input: &I) -> Result<O>;

    fn observe(&mut self, _sample: &TrainingSample<I, O>) -> Result<()> {
        Ok(())
    }
}

pub type Behavior<I, O> = dyn FunctionBehavior<I, O>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingSample<I, O> {
    pub input: I,
    pub expected: O,
    pub actual: Option<O>,
    pub reward: Option<f32>,
    pub weight: f32,
    pub source: TrainingSource,
    pub t_ms: TimeMs,
}

impl<I, O> TrainingSample<I, O> {
    pub fn teacher(input: I, expected: O, actual: Option<O>, t_ms: TimeMs) -> Self {
        Self {
            input,
            expected,
            actual,
            reward: None,
            weight: 1.0,
            source: TrainingSource::HardcodedTeacher,
            t_ms,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingSource {
    WorldOutcome,
    HardcodedTeacher,
    HumanReign,
    LlmCritique,
    SafetyVeto,
    MemoryRecall,
    Replay,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BehaviorRegime {
    Hardcoded,
    ShadowTrain,
    ShadowInfer,
    ModelInfer,
    ModelTrainAndInfer,
    Compare,
}

impl Default for BehaviorRegime {
    fn default() -> Self {
        Self::Hardcoded
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackPolicy {
    UseHardcoded,
    UseLastGoodOutput,
    ReturnError,
    StopSafely,
}

impl FallbackPolicy {
    fn should_use_hardcoded(self) -> bool {
        matches!(self, Self::UseHardcoded)
    }

    fn should_error(self) -> bool {
        matches!(self, Self::ReturnError)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BehaviorRunRecord<I, O> {
    pub behavior_id: String,
    pub regime: BehaviorRegime,
    pub t_ms: TimeMs,
    pub input: I,
    pub hardcoded_output: Option<O>,
    pub model_output: Option<O>,
    pub selected_output: Option<O>,
    pub error: Option<String>,
    pub disagreement: Option<f32>,
    pub confidence: Option<f32>,
    pub hardcoded_inference_us: Option<u64>,
    pub model_inference_us: Option<u64>,
}

impl<I, O> BehaviorRunRecord<I, O>
where
    I: Serialize,
    O: Serialize,
{
    pub fn erase(&self) -> ErasedBehaviorRunRecord {
        ErasedBehaviorRunRecord {
            behavior_id: self.behavior_id.clone(),
            regime: self.regime,
            t_ms: self.t_ms,
            input_json: serde_json::to_value(&self.input).unwrap_or(Value::Null),
            hardcoded_json: self
                .hardcoded_output
                .as_ref()
                .and_then(|value| serde_json::to_value(value).ok()),
            model_json: self
                .model_output
                .as_ref()
                .and_then(|value| serde_json::to_value(value).ok()),
            selected_json: self
                .selected_output
                .as_ref()
                .and_then(|value| serde_json::to_value(value).ok()),
            error: self.error.clone(),
            disagreement: self.disagreement,
            hardcoded_inference_us: self.hardcoded_inference_us,
            model_inference_us: self.model_inference_us,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ErasedBehaviorRunRecord {
    pub behavior_id: String,
    pub regime: BehaviorRegime,
    pub t_ms: TimeMs,
    pub input_json: Value,
    pub hardcoded_json: Option<Value>,
    pub model_json: Option<Value>,
    pub selected_json: Option<Value>,
    pub error: Option<String>,
    pub disagreement: Option<f32>,
    #[serde(default)]
    pub hardcoded_inference_us: Option<u64>,
    #[serde(default)]
    pub model_inference_us: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BehaviorImplementation {
    pub id: String,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BehaviorNodeState {
    pub node_id: String,
    pub behavior_id: String,
    pub label: String,
    pub allowed_regimes: Vec<BehaviorRegime>,
    pub hardcoded_implementations: Vec<BehaviorImplementation>,
    pub model_implementations: Vec<BehaviorImplementation>,
    pub selected_regime: BehaviorRegime,
    pub selected_hardcoded: String,
    pub selected_model: Option<String>,
    pub checkpoint_path: Option<String>,
    pub fallback_policy: FallbackPolicy,
    pub training_enabled: bool,
    pub last_run: Option<ErasedBehaviorRunRecord>,
    pub samples_observed: usize,
    pub train_steps_used: usize,
    pub missing_model_or_checkpoint: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BehaviorNodeUpdate {
    pub selected_regime: Option<BehaviorRegime>,
    pub selected_hardcoded: Option<String>,
    pub selected_model: Option<String>,
    pub checkpoint_path: Option<String>,
    pub fallback_policy: Option<FallbackPolicy>,
    pub training_enabled: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BehaviorRun<I, O> {
    pub chosen: O,
    pub record: BehaviorRunRecord<I, O>,
    pub fallback_used: bool,
    pub training_sample_emitted: bool,
}

pub type Replaceable<I, O> = ReplaceableBehavior<I, O>;

pub struct ReplaceableBehavior<I, O> {
    pub id: String,
    pub regime: BehaviorRegime,
    pub hardcoded: Box<dyn FunctionBehavior<I, O>>,
    pub model: Option<Box<dyn FunctionBehavior<I, O>>>,
    pub fallback: FallbackPolicy,
    last_good_output: Option<O>,
}

impl<I, O> ReplaceableBehavior<I, O>
where
    I: Clone,
    O: Clone,
{
    pub fn new(
        id: impl Into<String>,
        regime: BehaviorRegime,
        hardcoded: Box<dyn FunctionBehavior<I, O>>,
        model: Option<Box<dyn FunctionBehavior<I, O>>>,
        fallback: FallbackPolicy,
    ) -> Self {
        Self {
            id: id.into(),
            regime,
            hardcoded,
            model,
            fallback,
            last_good_output: None,
        }
    }

    pub fn infer(&mut self, input: &I, t_ms: TimeMs) -> Result<BehaviorRun<I, O>> {
        self.infer_with_teacher_source(input, t_ms, TrainingSource::HardcodedTeacher)
    }

    pub fn infer_with_teacher_source(
        &mut self,
        input: &I,
        t_ms: TimeMs,
        teacher_source: TrainingSource,
    ) -> Result<BehaviorRun<I, O>> {
        let mut record = BehaviorRunRecord {
            behavior_id: self.id.clone(),
            regime: self.regime,
            t_ms,
            input: input.clone(),
            hardcoded_output: None,
            model_output: None,
            selected_output: None,
            error: None,
            disagreement: None,
            confidence: None,
            hardcoded_inference_us: None,
            model_inference_us: None,
        };

        let selected = match self.regime {
            BehaviorRegime::Hardcoded => {
                let started = Instant::now();
                let hard = self.hardcoded.infer(input)?;
                record.hardcoded_inference_us = Some(elapsed_us(started));
                record.hardcoded_output = Some(hard.clone());
                (hard, false, false)
            }
            BehaviorRegime::ShadowTrain => {
                let started = Instant::now();
                let hard = self.hardcoded.infer(input)?;
                record.hardcoded_inference_us = Some(elapsed_us(started));
                record.hardcoded_output = Some(hard.clone());
                let mut trained = false;
                if let Some(model) = self.model.as_mut() {
                    let started = Instant::now();
                    let result = model.infer(input);
                    record.model_inference_us = Some(elapsed_us(started));
                    let actual = match result {
                        Ok(model_output) => {
                            record.model_output = Some(model_output.clone());
                            Some(model_output)
                        }
                        Err(error) => {
                            record.error = Some(error.to_string());
                            None
                        }
                    };
                    let mut sample =
                        TrainingSample::teacher(input.clone(), hard.clone(), actual, t_ms);
                    sample.source = teacher_source;
                    model.observe(&sample)?;
                    trained = true;
                }
                (hard, false, trained)
            }
            BehaviorRegime::ShadowInfer => {
                let started = Instant::now();
                let hard = self.hardcoded.infer(input)?;
                record.hardcoded_inference_us = Some(elapsed_us(started));
                record.hardcoded_output = Some(hard.clone());
                if let Some(model) = self.model.as_mut() {
                    let started = Instant::now();
                    let result = model.infer(input);
                    record.model_inference_us = Some(elapsed_us(started));
                    match result {
                        Ok(model_output) => record.model_output = Some(model_output),
                        Err(error) => record.error = Some(error.to_string()),
                    }
                }
                (hard, false, false)
            }
            BehaviorRegime::ModelInfer => self.run_model_controlled(
                input,
                t_ms,
                false,
                TrainingSource::HardcodedTeacher,
                &mut record,
            )?,
            BehaviorRegime::ModelTrainAndInfer => {
                self.run_model_controlled(input, t_ms, true, teacher_source, &mut record)?
            }
            BehaviorRegime::Compare => {
                let started = Instant::now();
                let hard = self.hardcoded.infer(input)?;
                record.hardcoded_inference_us = Some(elapsed_us(started));
                record.hardcoded_output = Some(hard.clone());
                if let Some(model) = self.model.as_mut() {
                    let started = Instant::now();
                    let result = model.infer(input);
                    record.model_inference_us = Some(elapsed_us(started));
                    match result {
                        Ok(model_output) => record.model_output = Some(model_output),
                        Err(error) => record.error = Some(error.to_string()),
                    }
                }
                (hard, false, false)
            }
        };

        let (chosen, fallback_used, training_sample_emitted) = selected;
        record.selected_output = Some(chosen.clone());
        self.last_good_output = Some(chosen.clone());
        Ok(BehaviorRun {
            chosen,
            record,
            fallback_used,
            training_sample_emitted,
        })
    }

    pub fn run(&mut self, input: &I, t_ms: TimeMs) -> Result<BehaviorRun<I, O>> {
        self.infer(input, t_ms)
    }

    pub fn hardcoded_id(&self) -> &'static str {
        self.hardcoded.id()
    }

    pub fn model_id(&self) -> Option<&'static str> {
        self.model.as_ref().map(|model| model.id())
    }

    pub fn observe(&mut self, sample: &TrainingSample<I, O>) -> Result<()> {
        if let Some(model) = self.model.as_mut() {
            model.observe(sample)?;
        }
        Ok(())
    }

    fn run_model_controlled(
        &mut self,
        input: &I,
        t_ms: TimeMs,
        train: bool,
        teacher_source: TrainingSource,
        record: &mut BehaviorRunRecord<I, O>,
    ) -> Result<(O, bool, bool)> {
        if let Some(model) = self.model.as_mut() {
            let started = Instant::now();
            let result = model.infer(input);
            record.model_inference_us = Some(elapsed_us(started));
            match result {
                Ok(model_output) => {
                    record.model_output = Some(model_output.clone());
                    let mut trained = false;
                    if train {
                        let started = Instant::now();
                        let hard = self.hardcoded.infer(input)?;
                        record.hardcoded_inference_us = Some(elapsed_us(started));
                        record.hardcoded_output = Some(hard.clone());
                        let mut sample = TrainingSample::teacher(
                            input.clone(),
                            hard,
                            Some(model_output.clone()),
                            t_ms,
                        );
                        sample.source = teacher_source;
                        model.observe(&sample)?;
                        trained = true;
                    }
                    return Ok((model_output, false, trained));
                }
                Err(error) => {
                    record.error = Some(error.to_string());
                    return self.fallback_output(input, record);
                }
            }
        }
        record.error = Some("model behavior is not configured".to_string());
        self.fallback_output(input, record)
    }

    fn fallback_output(
        &mut self,
        input: &I,
        record: &mut BehaviorRunRecord<I, O>,
    ) -> Result<(O, bool, bool)> {
        if self.fallback.should_use_hardcoded() {
            let started = Instant::now();
            let hard = self.hardcoded.infer(input)?;
            record.hardcoded_inference_us = Some(elapsed_us(started));
            record.hardcoded_output = Some(hard.clone());
            return Ok((hard, true, false));
        }
        if matches!(self.fallback, FallbackPolicy::UseLastGoodOutput) {
            if let Some(output) = self.last_good_output.clone() {
                return Ok((output, true, false));
            }
        }
        if matches!(self.fallback, FallbackPolicy::StopSafely) {
            if let Some(output) = self.last_good_output.clone() {
                return Ok((output, true, false));
            }
        }
        if self.fallback.should_error() || record.error.is_some() {
            return Err(anyhow!(
                "{} failed under {:?}: {}",
                self.id,
                self.regime,
                record
                    .error
                    .clone()
                    .unwrap_or_else(|| "unknown error".to_string())
            ));
        }
        Err(anyhow!("{} failed without fallback", self.id))
    }
}

fn elapsed_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

impl<I, O> ReplaceableBehavior<I, O>
where
    I: Clone,
    O: Clone + OutputDistance,
{
    pub fn infer_with_disagreement(
        &mut self,
        input: &I,
        t_ms: TimeMs,
    ) -> Result<BehaviorRun<I, O>> {
        let mut run = self.infer(input, t_ms)?;
        run.record.disagreement = match (&run.record.hardcoded_output, &run.record.model_output) {
            (Some(hard), Some(model)) => Some(hard.distance(model)),
            _ => None,
        };
        Ok(run)
    }
}

pub trait OutputDistance {
    fn distance(&self, other: &Self) -> f32;
}

impl OutputDistance for f32 {
    fn distance(&self, other: &Self) -> f32 {
        (self - other).abs()
    }
}

impl OutputDistance for bool {
    fn distance(&self, other: &Self) -> f32 {
        if self == other {
            0.0
        } else {
            1.0
        }
    }
}

pub trait TargetExtractor<T, I, O>: Send + Sync {
    fn extract(&self, transition: &T) -> Result<Option<TrainingSample<I, O>>>;
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BehaviorRegistryConfig {
    #[serde(default)]
    pub behavior: std::collections::BTreeMap<String, BehaviorConfig>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BehaviorConfig {
    #[serde(default)]
    pub regime: BehaviorRegime,
    pub hardcoded: String,
    pub model: Option<String>,
    pub checkpoint: Option<String>,
    #[serde(default = "default_fallback")]
    pub fallback: FallbackPolicy,
}

fn default_fallback() -> FallbackPolicy {
    FallbackPolicy::UseHardcoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct ConstBehavior {
        id: &'static str,
        output: i32,
        observes: Arc<Mutex<usize>>,
        fail: bool,
    }

    impl FunctionBehavior<i32, i32> for ConstBehavior {
        fn id(&self) -> &'static str {
            self.id
        }

        fn infer(&mut self, _input: &i32) -> Result<i32> {
            if self.fail {
                Err(anyhow!("boom"))
            } else {
                Ok(self.output)
            }
        }

        fn observe(&mut self, _sample: &TrainingSample<i32, i32>) -> Result<()> {
            *self.observes.lock().unwrap() += 1;
            Ok(())
        }
    }

    impl OutputDistance for i32 {
        fn distance(&self, other: &Self) -> f32 {
            (*self - *other).abs() as f32
        }
    }

    fn behavior(regime: BehaviorRegime) -> ReplaceableBehavior<i32, i32> {
        ReplaceableBehavior::new(
            "test",
            regime,
            Box::new(ConstBehavior {
                id: "hard",
                output: 7,
                observes: Arc::new(Mutex::new(0)),
                fail: false,
            }),
            Some(Box::new(ConstBehavior {
                id: "model",
                output: 9,
                observes: Arc::new(Mutex::new(0)),
                fail: false,
            })),
            FallbackPolicy::UseHardcoded,
        )
    }

    #[test]
    fn hardcoded_returns_hardcoded_output() {
        let mut b = behavior(BehaviorRegime::Hardcoded);
        let run = b.infer(&1, 10).unwrap();
        assert_eq!(run.chosen, 7);
        assert_eq!(run.record.hardcoded_output, Some(7));
        assert_eq!(run.record.model_output, None);
    }

    #[test]
    fn shadow_infer_returns_hardcoded_and_records_model() {
        let mut b = behavior(BehaviorRegime::ShadowInfer);
        let run = b.infer_with_disagreement(&1, 10).unwrap();
        assert_eq!(run.chosen, 7);
        assert_eq!(run.record.model_output, Some(9));
        assert_eq!(run.record.disagreement, Some(2.0));
    }

    #[test]
    fn shadow_train_calls_model_observe_but_returns_hardcoded() {
        let observes = Arc::new(Mutex::new(0));
        let mut b = ReplaceableBehavior::new(
            "train",
            BehaviorRegime::ShadowTrain,
            Box::new(ConstBehavior {
                id: "hard",
                output: 4,
                observes: Arc::new(Mutex::new(0)),
                fail: false,
            }),
            Some(Box::new(ConstBehavior {
                id: "model",
                output: 2,
                observes: observes.clone(),
                fail: false,
            })),
            FallbackPolicy::UseHardcoded,
        );
        let run = b.infer(&1, 10).unwrap();
        assert_eq!(run.chosen, 4);
        assert_eq!(*observes.lock().unwrap(), 1);
    }

    #[test]
    fn model_infer_returns_model_output() {
        let mut b = behavior(BehaviorRegime::ModelInfer);
        assert_eq!(b.infer(&1, 10).unwrap().chosen, 9);
    }

    #[test]
    fn model_infer_falls_back_to_hardcoded_on_model_error() {
        let mut b = ReplaceableBehavior::new(
            "fallback",
            BehaviorRegime::ModelInfer,
            Box::new(ConstBehavior {
                id: "hard",
                output: 7,
                observes: Arc::new(Mutex::new(0)),
                fail: false,
            }),
            Some(Box::new(ConstBehavior {
                id: "model",
                output: 9,
                observes: Arc::new(Mutex::new(0)),
                fail: true,
            })),
            FallbackPolicy::UseHardcoded,
        );
        let run = b.infer(&1, 10).unwrap();
        assert_eq!(run.chosen, 7);
        assert!(run.fallback_used);
    }

    #[test]
    fn compare_records_both_outputs() {
        let mut b = behavior(BehaviorRegime::Compare);
        let run = b.infer(&1, 10).unwrap();
        assert_eq!(run.chosen, 7);
        assert_eq!(run.record.hardcoded_output, Some(7));
        assert_eq!(run.record.model_output, Some(9));
    }

    #[test]
    fn regime_config_loads_from_toml() {
        let config: BehaviorRegistryConfig = toml::from_str(
            r#"
            [behavior.danger]
            regime = "shadow_train"
            hardcoded = "danger.range_bumper"
            model = "danger.burn.v0"
            checkpoint = "data/models/danger_v0"
            fallback = "use_hardcoded"
            "#,
        )
        .unwrap();
        assert_eq!(
            config.behavior["danger"].regime,
            BehaviorRegime::ShadowTrain
        );
        assert_eq!(
            config.behavior["danger"].fallback,
            FallbackPolicy::UseHardcoded
        );
    }

    #[test]
    fn retired_behavior_config_spellings_are_rejected() {
        for config in [
            r#"
            [behavior.danger]
            mode = "shadow_train"
            hardcoded = "danger.range_bumper"
            model = "danger.burn.v0"
            checkpoint = "data/models/danger_v0"
            fallback = "use_hardcoded"
            "#,
            r#"
            [behavior.danger]
            regime = "shadow_train"
            hardcoded = "danger.range_bumper"
            model = "danger.burn.v0"
            checkpoint = "data/models/danger_v0"
            fallback = "hardcoded_on_error"
            "#,
            r#"
            [behavior.danger]
            regime = "shadow_train"
            hardcoded = "danger.range_bumper"
            model = "danger.burn.v0"
            checkpoint = "data/models/danger_v0"
            fallback = "stop_on_error"
            "#,
        ] {
            assert!(toml::from_str::<BehaviorRegistryConfig>(config).is_err());
        }
    }
}
