use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use pete_actions::ActionPrimitive;
use pete_behaviors::{BehaviorConfig, BehaviorRegime, BehaviorRegistryConfig, FallbackPolicy};
use pete_core::TimeMs;
use pete_experience::{
    action_features, action_value_input_from_transition_like,
    action_value_target_from_reward_surprise, charge_input_from_transition_like,
    charge_target_from_transition_like, danger_input_from_transition_like,
    danger_target_from_transition_like, ear_next_input_from_transition_like,
    ear_next_target_from_now, experience_decode_target_from_now, experience_encode_input_from_now,
    eye_next_input_from_transition_like, eye_next_target_from_now, ActionValueInput,
    ActionValueTarget, ChargeInput, ChargeTarget, CodebookQuantizer, CodebookUsageReport,
    DangerInput, DangerTarget, EarNextInput, EarNextTarget, ExperienceDecodeOutput,
    ExperienceEncodeInput, ExperienceLatent, ExperienceSurprise, EyeNextInput, EyeNextTarget,
    FutureInput, FuturePredictor, LatentEncoder, RandomProjectionExperienceEncoder,
    StasisFuturePredictor,
};
use pete_ledger::{
    future_input_from_transition, future_target_from_transition, ExperienceTransition, JsonlLedger,
};
use pete_models::{
    ActionValueNetTrainer, ChargeNetTrainer, DangerNetTrainer, EarNextNetTrainer,
    ExperienceAutoencoderTrainer, EyeNextNetTrainer, FutureNetTrainer,
    HardcodedActionValuePredictor, HardcodedChargePredictor, HardcodedDangerPredictor, TrainStats,
};
use pete_now::Now;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;

pub mod dream_policy;

// Training domains share the crate namespace to preserve its API.
include!("training/types.rs");
include!("training/train.rs");
include!("training/evaluate.rs");
include!("training/unified.rs");
include!("training/samples.rs");

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
