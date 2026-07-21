use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::{loss::MseLoss, loss::Reduction, Linear, LinearConfig};
use burn::optim::{adaptor::OptimizerAdaptor, GradientsParams, Optimizer, Sgd, SgdConfig};
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::{activation, backend::AutodiffBackend, backend::Backend, Tensor, TensorData};
use pete_behaviors::TrainingSample;
use pete_core::TimeMs;
use pete_experience::{
    ActionValueInput, ActionValueOutput, ActionValueTarget, ChargeInput, ChargeOutput,
    ChargeTarget, DangerInput, DangerOutput, DangerTarget, EarNextInput, EarNextOutput,
    EarNextTarget, ExperienceDecodeFeatureLengths, ExperienceDecodeOutput, ExperienceEncodeInput,
    ExperienceEncodeOutput, ExperienceLatent, EyeNextInput, EyeNextOutput, EyeNextTarget,
    FutureInput, FuturePrediction, LatentEncoder, EYE_NEXT_HEIGHT, EYE_NEXT_RGB_LEN,
    EYE_NEXT_WIDTH,
};
use pete_now::Now;
use serde::{Deserialize, Serialize};

include!("models/interfaces.rs");
include!("models/networks.rs");
include!("models/body_trainers.rs");
include!("models/perception_trainers.rs");
include!("models/metadata.rs");
include!("models/tensors.rs");
include!("models/registry.rs");

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
