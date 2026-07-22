use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use base64::Engine;
use pete_actions::{action_to_motor_command, ActionPrimitive, ExploreStyle, TurnDir};
use pete_body::BodySense;
use pete_core::{ExperienceId, ImpressionId, Provenance, Reward, SensationId, TimeMs};
use pete_now::{DriveSense, MemorySense, Now, SenseVectorizer, SurpriseSense, VisionDetection};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

tokio::task_local! {
    static DETERMINISTIC_IDENTITIES: std::cell::RefCell<(u128, u64)>;
}

/// Run a replay/shadow future with deterministic descendant identities rooted
/// in its immutable input frame. Live callers do not enter this scope and keep
/// random v4 identities.
pub async fn with_deterministic_identities<F>(frame_id: Uuid, future: F) -> F::Output
where
    F: std::future::Future,
{
    DETERMINISTIC_IDENTITIES
        .scope(std::cell::RefCell::new((frame_id.as_u128(), 0)), future)
        .await
}

fn new_experience_uuid() -> Uuid {
    DETERMINISTIC_IDENTITIES
        .try_with(|state| {
            let mut state = state.borrow_mut();
            let counter = state.1;
            state.1 = state.1.saturating_add(1);
            let mixed = state.0
                ^ (u128::from(counter).wrapping_mul(0x9e3779b97f4a7c15_u128))
                ^ 0x5a17_0000_0000_0000_8000_0000_0000_0000_u128;
            Uuid::from_u128(mixed)
        })
        .unwrap_or_else(|_| Uuid::new_v4())
}

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
