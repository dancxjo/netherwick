use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use base64::Engine;
use pete_actions::{action_to_motor_command, ActionPrimitive, ExploreStyle, TurnDir};
use pete_body::BodySense;
use pete_core::{ExperienceId, ImpressionId, Provenance, Reward, SensationId, TimeMs};
use pete_now::{DriveSense, MemorySense, Now, SenseVectorizer, SurpriseSense};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

// Experience domains share one namespace to preserve the crate API.
include!("experience/prediction.rs");
include!("experience/baseline.rs");
include!("experience/types.rs");
include!("experience/vectorization.rs");
include!("experience/visual.rs");
include!("experience/audio.rs");
include!("experience/pipeline.rs");
include!("experience/now.rs");
include!("experience/behavior.rs");

#[cfg(test)]
#[path = "encoding_tests.rs"]
mod encoding_tests;
