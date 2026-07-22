use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{fs, io};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{DefaultBodyLimit, Path as AxumPath, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use axum_server::tls_rustls::RustlsConfig;
use base64::Engine;
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageEncoder};
use pete_actions::{
    ActionPrimitive, LlmAdvisoryAction, ReignCommand, ReignInput, ReignMode, ReignSource, TurnDir,
};
use pete_behaviors::{BehaviorNodeState, BehaviorNodeUpdate, BehaviorRegime};
use pete_cockpit::{MotionCommand, MotorCommand};
use pete_core::TimeMs;
use pete_events::{
    ArtifactIdentity, ArtifactKind, AuthoritySignificance, Brain, BrainEvent, BrainEventId,
    BrainEventPayload, BrainEventType, EventDisposition, EventTimes, LossPolicy, PayloadReference,
    ProducerIdentity, TrustState, BRAIN_EVENT_SCHEMA_VERSION,
};
use pete_experience::EmbodiedContext;
use pete_map::{
    orientation_from_imu, project_beam_endpoint, LocalMap, LocalWorldBelief, MapObservation,
    MapSummary, OccupancyCell as OdomMapCell, PointCloudSummary, PoseEdgeSource,
    PoseGraphOptimizationSummary, RemapSummary, SlamMode, VoxelPoint, VoxelPointCloud, MAP_LABEL,
    WORLD_POINT_CLOUD_LABEL,
};
use pete_memory::{
    CognitiveDiagnosticsReport, EntityConstellationState, EntityLifecycleState, EntityMemory,
    EntityMemoryReport,
};
use pete_now::{KinectSense, KinectSkeletonSense, ObjectSense, ReignSense};
use pete_runtime::{
    nudge_action_block_reason_for_snapshot, InlineLearningConfig, InlineLearningMode, NudgePolicy,
    NudgeStatus, ReignQueue, RuntimeModelStack,
};
use pete_sensors::{
    ClusterObservation, EyeFrame, EyeFrameFormat, OccupancyGrid, PlaneObservation,
    SceneGraphSummary, SurfaceExtractor, SurfaceExtractorDiagnostics, SurfaceHypothesis,
    SurfaceTrack, WorldSnapshot,
};
use pete_worldlab::CaptureReader;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tower_http::services::ServeDir;
use uuid::Uuid;

// Server domains share one namespace to keep the public API stable.
include!("server/reign_state.rs");
include!("server/live_state.rs");
include!("server/models.rs");
include!("server/router.rs");
include!("server/map.rs");
include!("server/retina.rs");
include!("server/scene.rs");
include!("server/reign.rs");
include!("server/observatory.rs");
include!("server/observatory_source.rs");
include!("server/observatory_graph.rs");
include!("server/observatory_authority.rs");
include!("server/observatory_calibration.rs");
include!("server/observatory_spatial.rs");
include!("server/observatory_health.rs");
include!("server/observatory_diagnostics.rs");

const REIGN_PAGE: &str = include_str!("web/reign.html");
const COGNITIVE_VIEW_PAGE: &str = include_str!("web/cognitive.html");
const LIVE_VIEW_PAGE: &str = include_str!("web/live.html");
const LIVE_VIEW_3D_PAGE: &str = include_str!("web/live_3d.html");
const OBSERVATORY_PAGE: &str = include_str!("web/observatory.html");

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
