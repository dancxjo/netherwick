use anyhow::{Context, Result};
use async_trait::async_trait;
use base64::Engine;
use chrono::{Local, SecondsFormat, TimeZone};
use futures_util::StreamExt;
use image::ImageEncoder;
use pete_actions::{
    ActionPrimitive, ApproachTarget, ChirpPattern, ExploreStyle, InspectTarget, TurnDir,
};
use pete_cognition::{
    AsyncCognitionSupervisor, BoundedImageInput, BoundedInputRef, CallerRole, CapabilityDescriptor,
    CognitiveCapability, CognitiveProvider, CognitiveProviderDescriptor, CognitiveRequest,
    CognitiveRequestPayload, CognitiveResponse, CognitiveResponsePayload, CognitiveResponseStatus,
    CognitiveRole, CognitiveRouter, HostId, LatencyEstimate, Locality, PrivacyPolicy, ProcessId,
    ProviderHealth, ProviderHealthState, ProviderId, ProviderRegistrySnapshot, RequestProvenance,
    ResourceClass, ResourceCost, ResponseDisposition, RoutedResponse, SnapshotRef,
    SubmissionDisposition, TrustPolicy,
};
use pete_experience::{EmbodiedContext, ExperienceLatent, FuturePrediction, Impression};
use pete_now::{
    EyeFrame, EyeFrameFormat, LlmSense, Now, ReignSense, VectorArtifact,
    IMAGE_DESCRIPTION_VECTOR_COLLECTION, SCENE_VECTOR_COLLECTION,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Cursor;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;
use uuid::Uuid;

include!("llm/types.rs");
include!("llm/vision.rs");
include!("llm/agents.rs");
include!("llm/prompts.rs");
include!("llm/parsing.rs");

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
