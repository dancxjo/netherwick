use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use pete_actions::{ActionPrimitive, TurnDir};
use pete_body::{BodyFlags, BodySense, Velocity};
use pete_core::{Feature, FeatureId, Goal, Pose2, Reward};
use pete_experience::{
    EmbodiedContext, EmbodiedPipeline, EmbodiedVectorCoverage, Experience, ExperienceFuser,
    FuturePrediction, Impression, InstantCoverage, MemoryLink, Modality, RecalledExperience,
    SensationPayloadKind, VectorEmbedding,
};
use pete_ledger::{ExperienceFrame, ExperienceTransition};
use pete_now::{
    AsrSense, EarSense, Episode, EpisodeKind, EpistemicSnapshot, EyeFrame, EyeFrameFormat,
    GraphEdge, GraphEntity, InteractionState, KinectJointSense, KinectSense, KinectSkeletonSense,
    MemorySense, Now, ObjectClass, ObjectObservation, ObjectObservationSource, PersonId,
    RangeSense, RecallHit, SemanticGraphSnapshot, SemanticNodeRef, SocialWorldSnapshot,
    SurpriseSense, TemporalContext, VectorArtifact, FACE_VECTOR_COLLECTION,
    SCENE_VECTOR_COLLECTION, VOICE_VECTOR_COLLECTION,
};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

// Memory domains share one namespace to preserve the crate API.
include!("memory/place.rs");
include!("memory/binding.rs");
include!("memory/constellations.rs");
include!("memory/learning.rs");
include!("memory/diagnostics.rs");
include!("memory/intelligence.rs");
include!("memory/hypotheses.rs");
include!("memory/entity.rs");
include!("memory/backends.rs");
include!("memory/stores.rs");
include!("memory/embodied_eval.rs");
include!("memory/graph.rs");
include!("memory/vector_recall.rs");

#[cfg(test)]
#[path = "cognitive_diagnostics_tests.rs"]
mod cognitive_diagnostics_tests;

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
