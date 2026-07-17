//! Runtime-loaded, sandboxed motherbrain skills.
//!
//! Lua owns semantic sequencing. Rust owns bodily resources, bounded command
//! renewal, authority, numerical controllers, and every physical safety check.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::future::Future;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll, Wake, Waker};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use mlua::{
    AsyncThread, DebugEvent, Error as LuaError, Function, HookTriggers, Lua, LuaOptions,
    LuaSerdeExt, MultiValue, StdLib, Table, UserData, UserDataFields, Value as LuaValue, Variadic,
    VmState,
};
use pete_cockpit::{CockpitEventKind, EventBatch, SafetyLatchKind};
use pete_conductor::{SkillId, SkillOutcome, SkillPhase, SkillRequest, SkillStatus};
use pete_now::{Now, ObjectClass};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const LUA_ERROR_PREFIX: &str = "__netherwick_skill_error__:";
const MAX_CONVERTED_VALUE_BYTES: usize = 64 * 1024;
const MAX_PROGRESS_ENTRIES: usize = 64;
const MAX_TRACE_EVENTS: usize = 512;
const PRIMITIVE_TTL_MS: u32 = 250;

#[derive(Clone, Debug)]
pub struct LuaSkillConfig {
    pub directory: PathBuf,
    pub namespace: String,
    pub instruction_budget: u64,
    pub activation_budget: Duration,
    pub memory_limit_bytes: usize,
    pub maximum_result_bytes: usize,
    pub maximum_operation_ms: u64,
}

impl Default for LuaSkillConfig {
    fn default() -> Self {
        Self {
            directory: PathBuf::from("skills/motherbrain"),
            namespace: "motherbrain".to_string(),
            instruction_budget: 1_000_000,
            activation_budget: Duration::from_millis(20),
            memory_limit_bytes: 8 * 1024 * 1024,
            maximum_result_bytes: MAX_CONVERTED_VALUE_BYTES,
            maximum_operation_ms: 30_000,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LoadedSkill {
    pub skill_id: String,
    pub function_name: String,
    pub source_path: PathBuf,
    pub source_hash: String,
    pub loaded_at_ms: u64,
    pub runtime_version: String,
}

#[derive(Clone, Debug)]
struct SkillSource {
    metadata: LoadedSkill,
    source: Arc<str>,
}

#[derive(Clone, Debug)]
struct LoadedSkillSet {
    generation_hash: String,
    skills: BTreeMap<String, SkillSource>,
    ordered_sources: Vec<SkillSource>,
}

#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum BodyResource {
    #[default]
    Locomotion,
    Gaze,
    Manipulator,
    Voice,
    BodyMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HazardKind {
    BumperFront,
    Cliff,
}

impl HazardKind {
    pub fn latch(self) -> SafetyLatchKind {
        match self {
            Self::BumperFront => SafetyLatchKind::Bump,
            Self::Cliff => SafetyLatchKind::Cliff,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HostOperation {
    Stop,
    FaceBearing {
        bearing_rad: f32,
    },
    TurnBy {
        angle_rad: f32,
    },
    Drive {
        linear_m_s: f32,
        duration_ms: u64,
    },
    DriveDistance {
        distance_m: f32,
        velocity_m_s: f32,
    },
    Approach {
        target: EntityHandle,
        stop_range_m: f32,
    },
    FollowBearing {
        bearing_rad: f32,
        linear_m_s: f32,
    },
    HoldHeading {
        heading_rad: f32,
        linear_m_s: f32,
    },
    FollowWall {
        side: String,
        distance_m: f32,
    },
    Scan,
    LookAt {
        target: EntityHandle,
    },
    Observe {
        target: EntityHandle,
    },
    SearchForDockSignal,
    AlignWithDock,
    VerifyCharging,
    Undock,
    Retreat {
        hazard: HazardKind,
        distance_m: f32,
    },
    CompleteHazardRecovery {
        hazard: HazardKind,
    },
    ReleasePersistentBumper,
    Grasp {
        target: EntityHandle,
    },
    Release {
        target: Option<EntityHandle>,
    },
    BringToMouth {
        target: EntityHandle,
    },
    Chew,
    Swallow,
    Say {
        text: String,
    },
    PlayFeedback {
        pattern: String,
    },
    WaitUntil {
        predicate: String,
        timeout_ms: u64,
    },
}

impl HostOperation {
    pub fn resource(&self) -> Option<BodyResource> {
        match self {
            Self::Stop
            | Self::FaceBearing { .. }
            | Self::TurnBy { .. }
            | Self::Drive { .. }
            | Self::DriveDistance { .. }
            | Self::Approach { .. }
            | Self::FollowBearing { .. }
            | Self::HoldHeading { .. }
            | Self::FollowWall { .. }
            | Self::AlignWithDock
            | Self::Retreat { .. }
            | Self::CompleteHazardRecovery { .. }
            | Self::ReleasePersistentBumper => Some(BodyResource::Locomotion),
            Self::Scan | Self::LookAt { .. } => Some(BodyResource::Gaze),
            Self::Grasp { .. }
            | Self::Release { .. }
            | Self::BringToMouth { .. }
            | Self::Chew
            | Self::Swallow => Some(BodyResource::Manipulator),
            Self::Say { .. } | Self::PlayFeedback { .. } => Some(BodyResource::Voice),
            Self::Undock => Some(BodyResource::BodyMode),
            Self::SearchForDockSignal => Some(BodyResource::Locomotion),
            Self::Observe { .. } | Self::VerifyCharging | Self::WaitUntil { .. } => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::FaceBearing { .. } => "face_bearing",
            Self::TurnBy { .. } => "turn_by",
            Self::Drive { .. } => "drive",
            Self::DriveDistance { .. } => "drive_distance",
            Self::Approach { .. } => "approach",
            Self::FollowBearing { .. } => "follow_bearing",
            Self::HoldHeading { .. } => "hold_heading",
            Self::FollowWall { .. } => "follow_wall",
            Self::Scan => "scan",
            Self::LookAt { .. } => "look_at",
            Self::Observe { .. } => "observe",
            Self::SearchForDockSignal => "search_for_dock_signal",
            Self::AlignWithDock => "align_with_dock",
            Self::VerifyCharging => "verify_charging",
            Self::Undock => "undock",
            Self::Retreat { .. } => "retreat",
            Self::CompleteHazardRecovery { .. } => "complete_hazard_recovery",
            Self::ReleasePersistentBumper => "release_persistent_bumper",
            Self::Grasp { .. } => "grasp",
            Self::Release { .. } => "release",
            Self::BringToMouth { .. } => "bring_to_mouth",
            Self::Chew => "chew",
            Self::Swallow => "swallow",
            Self::Say { .. } => "say",
            Self::PlayFeedback { .. } => "play_feedback",
            Self::WaitUntil { .. } => "wait_until",
        }
    }

    fn default_timeout_ms(&self) -> u64 {
        match self {
            Self::Stop | Self::Observe { .. } | Self::PlayFeedback { .. } => 1_000,
            Self::Retreat { .. } | Self::ReleasePersistentBumper => 15_000,
            Self::WaitUntil { timeout_ms, .. } => *timeout_ms,
            _ => 30_000,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EntityHandle {
    id: String,
    kind: String,
    label: String,
    name: Option<String>,
}

impl EntityHandle {
    pub fn new(id: impl Into<String>, kind: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind: kind.into(),
            label: label.into(),
            name: None,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    fn with_name(mut self, name: Option<String>) -> Self {
        self.name = name;
        self
    }
}

impl UserData for EntityHandle {
    fn add_fields<F: UserDataFields<Self>>(fields: &mut F) {
        fields.add_field_method_get("id", |_, this| Ok(this.id.clone()));
        fields.add_field_method_get("kind", |_, this| Ok(this.kind.clone()));
        fields.add_field_method_get("label", |_, this| Ok(this.label.clone()));
        fields.add_field_method_get("name", |_, this| Ok(this.name.clone()));
        fields.add_field_method_get("recognized", |_, this| Ok(this.name.is_some()));
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SkillFailure {
    pub outcome: SkillOutcome,
    pub kind: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource: Option<BodyResource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brainstem_event: Option<Value>,
    #[serde(default)]
    pub details: BTreeMap<String, Value>,
}

impl SkillFailure {
    pub fn new(outcome: SkillOutcome, kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            outcome,
            kind: kind.into(),
            message: message.into(),
            operation: None,
            resource: None,
            brainstem_event: None,
            details: BTreeMap::new(),
        }
    }

    pub fn capability(operation: &HostOperation) -> Self {
        Self::new(
            SkillOutcome::CapabilityUnavailable,
            "capability_unavailable",
            format!("{} is not present on this body", operation.name()),
        )
        .for_operation(operation)
    }

    pub fn safety(kind: impl Into<String>, message: impl Into<String>, event: Value) -> Self {
        let mut failure = Self::new(SkillOutcome::SafetyPreempted, kind, message);
        failure.brainstem_event = Some(event);
        failure
    }

    pub fn for_operation(mut self, operation: &HostOperation) -> Self {
        self.operation = Some(operation.name().to_string());
        self.resource = operation.resource();
        self
    }

    fn encoded(&self) -> LuaError {
        let json = serde_json::to_string(self).unwrap_or_else(|_| {
            r#"{"outcome":"script_error","kind":"error_encoding","message":"failed to encode typed skill error"}"#.to_string()
        });
        LuaError::RuntimeError(format!("{LUA_ERROR_PREFIX}{json}"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrimitiveIntent {
    pub operation_id: u64,
    pub child_id: u64,
    pub operation: String,
    pub resource: Option<BodyResource>,
    pub emitted_at_ms: u64,
    pub detail: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OrganPoll {
    Pending {
        progress: Option<(String, f32)>,
        primitive: Option<PrimitiveIntent>,
    },
    Completed(Value),
    Failed(SkillFailure),
}

#[derive(Clone, Copy, Debug)]
pub struct OperationContext {
    pub operation_id: u64,
    pub child_id: u64,
    pub first_poll: bool,
    pub elapsed_ms: u64,
    pub now_ms: u64,
    pub primitive_ttl_ms: u32,
}

pub trait OrganDriver {
    fn poll(
        &mut self,
        operation: &HostOperation,
        context: OperationContext,
        now: &Now,
        events: &EventBatch,
    ) -> OrganPoll;

    fn stop(&mut self, resource: BodyResource, reason: &SkillFailure);

    fn shutdown(&mut self) {
        let failure = SkillFailure::new(
            SkillOutcome::Cancelled,
            "motherbrain_shutdown",
            "motherbrain skill runtime shut down",
        );
        for resource in [
            BodyResource::Locomotion,
            BodyResource::Gaze,
            BodyResource::Manipulator,
            BodyResource::Voice,
            BodyResource::BodyMode,
        ] {
            self.stop(resource, &failure);
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChildDiagnostic {
    pub child_id: u64,
    pub parent_child_id: u64,
    pub order: usize,
    pub phase: String,
    pub current_function: Option<String>,
    pub current_operation: Option<String>,
    pub held_resource: Option<BodyResource>,
    pub waiting_resource: Option<BodyResource>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SkillDiagnostic {
    pub foreground_skill_id: Option<String>,
    pub source_hash: Option<String>,
    pub source_path: Option<PathBuf>,
    pub start_time_ms: Option<u64>,
    pub current_lua_function: Option<String>,
    pub current_operation: Option<String>,
    pub phase: String,
    pub held_resources: BTreeMap<BodyResource, u64>,
    pub waiting_resources: BTreeMap<BodyResource, Vec<u64>>,
    pub active_together_children: Vec<ChildDiagnostic>,
    pub progress: BTreeMap<String, f32>,
    pub last_yield_ms: Option<u64>,
    pub last_resume_ms: Option<u64>,
    pub last_preemption: Option<SkillFailure>,
    pub terminal_outcome: Option<SkillOutcome>,
    pub terminal_detail: Option<SkillFailure>,
}

/// Bounded execution evidence carried into `Now` and the experience ledger.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SkillExecutionRecord {
    pub execution_id: u64,
    pub skill: LoadedSkill,
    pub request: SkillRequest,
    pub diagnostics: SkillDiagnostic,
    pub trace: Vec<SkillTraceEvent>,
    pub observations: Vec<Value>,
    pub memories: Vec<Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SkillTraceEvent {
    Started {
        at_ms: u64,
        skill_id: String,
        source_hash: String,
        arguments: Value,
        starting_now: Value,
    },
    ChildStarted {
        at_ms: u64,
        child_id: u64,
        parent_child_id: u64,
        order: usize,
    },
    ResourceWaiting {
        at_ms: u64,
        operation_id: u64,
        child_id: u64,
        resource: BodyResource,
    },
    ResourceAcquired {
        at_ms: u64,
        operation_id: u64,
        child_id: u64,
        resource: BodyResource,
    },
    ResourceReleased {
        at_ms: u64,
        operation_id: u64,
        child_id: u64,
        resource: BodyResource,
        reason: String,
    },
    Primitive(PrimitiveIntent),
    Progress {
        at_ms: u64,
        key: String,
        value: f32,
    },
    Preempted {
        at_ms: u64,
        operation_id: u64,
        failure: SkillFailure,
    },
    Completed {
        at_ms: u64,
        outcome: SkillOutcome,
        detail: Option<SkillFailure>,
        duration_ms: u64,
        result: Option<Value>,
    },
}

#[derive(Clone, Debug)]
struct OperationRequest {
    id: u64,
    child_id: u64,
    operation: HostOperation,
    requested_at_ms: u64,
    timeout_ms: u64,
    slot: Arc<OperationSlot>,
}

#[derive(Debug, Default)]
struct OperationSlot {
    result: Mutex<Option<std::result::Result<Value, SkillFailure>>>,
    waker: Mutex<Option<Waker>>,
}

impl OperationSlot {
    fn finish(&self, result: std::result::Result<Value, SkillFailure>) {
        *self.result.lock().expect("operation result lock") = Some(result);
        if let Some(waker) = self.waker.lock().expect("operation waker lock").take() {
            waker.wake();
        }
    }
}

struct OperationWait {
    slot: Arc<OperationSlot>,
}

impl Future for OperationWait {
    type Output = std::result::Result<Value, SkillFailure>;

    fn poll(self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        if let Some(result) = self
            .slot
            .result
            .lock()
            .expect("operation result lock")
            .take()
        {
            return Poll::Ready(result);
        }
        *self.slot.waker.lock().expect("operation waker lock") = Some(cx.waker().clone());
        Poll::Pending
    }
}

#[derive(Clone, Debug)]
struct ActiveOperation {
    request: OperationRequest,
    granted_at_ms: u64,
    polls: u64,
}

#[derive(Default)]
struct BridgeState {
    requests: VecDeque<OperationRequest>,
    cancelled_children: VecDeque<(HashSet<u64>, SkillFailure)>,
    current_child: u64,
    next_child: u64,
    child_parent: HashMap<u64, (u64, usize)>,
    current_functions: HashMap<u64, String>,
    child_hazard: HashMap<u64, HazardKind>,
    snapshot: Option<Now>,
    /// Normalized, author-reported goal progress in the range 0..=1.
    reported_progress: BTreeMap<String, f32>,
    /// Raw operation measurements such as bearing error or target distance.
    progress: BTreeMap<String, f32>,
    observations: Vec<Value>,
    memories: Vec<Value>,
    trace: VecDeque<SkillTraceEvent>,
    execution_id: u64,
}

struct Bridge {
    state: Mutex<BridgeState>,
    next_operation: AtomicU64,
    cancelled: AtomicBool,
}

impl Default for Bridge {
    fn default() -> Self {
        Self {
            state: Mutex::new(BridgeState {
                next_child: 1,
                ..BridgeState::default()
            }),
            next_operation: AtomicU64::new(1),
            cancelled: AtomicBool::new(false),
        }
    }
}

impl Bridge {
    fn enqueue(&self, operation: HostOperation, maximum_operation_ms: u64) -> OperationWait {
        let id = self.next_operation.fetch_add(1, Ordering::Relaxed);
        let slot = Arc::new(OperationSlot::default());
        let mut state = self.state.lock().expect("skill bridge lock");
        if let Err(failure) = validate_operation_context(&state, &operation) {
            slot.finish(Err(failure.for_operation(&operation)));
            return OperationWait { slot };
        }
        let timeout_ms = operation
            .default_timeout_ms()
            .min(maximum_operation_ms.max(PRIMITIVE_TTL_MS as u64));
        let requested_at_ms = state.snapshot.as_ref().map_or(0, |now| now.t_ms);
        let child_id = state.current_child;
        state.requests.push_back(OperationRequest {
            id,
            child_id,
            operation,
            requested_at_ms,
            timeout_ms,
            slot: slot.clone(),
        });
        OperationWait { slot }
    }

    fn allocate_children(&self, count: usize) -> (u64, Vec<u64>) {
        let mut state = self.state.lock().expect("skill bridge lock");
        let parent = state.current_child;
        let mut ids = Vec::with_capacity(count);
        for order in 0..count {
            let id = state.next_child;
            state.next_child = state.next_child.saturating_add(1);
            state.child_parent.insert(id, (parent, order));
            ids.push(id);
        }
        (parent, ids)
    }

    fn set_current_child(&self, child_id: u64) -> u64 {
        let mut state = self.state.lock().expect("skill bridge lock");
        let previous = state.current_child;
        state.current_child = child_id;
        previous
    }

    fn current_child(&self) -> u64 {
        self.state.lock().expect("skill bridge lock").current_child
    }

    fn cancel_children(&self, children: impl IntoIterator<Item = u64>, failure: SkillFailure) {
        self.state
            .lock()
            .expect("skill bridge lock")
            .cancelled_children
            .push_back((children.into_iter().collect(), failure));
    }

    fn finish_children(&self, children: &[u64]) {
        let mut state = self.state.lock().expect("skill bridge lock");
        for child in children {
            state.child_parent.remove(child);
            state.child_hazard.remove(child);
            state.current_functions.remove(child);
        }
    }

    fn push_trace(&self, event: SkillTraceEvent) {
        let mut state = self.state.lock().expect("skill bridge lock");
        if state.trace.len() == MAX_TRACE_EVENTS {
            state.trace.pop_front();
        }
        state.trace.push_back(event);
    }

    fn snapshot(&self) -> mlua::Result<Now> {
        self.state
            .lock()
            .map_err(|_| LuaError::RuntimeError("skill snapshot lock poisoned".into()))?
            .snapshot
            .clone()
            .ok_or_else(|| LuaError::RuntimeError("Now is not available".into()))
    }
}

fn validate_operation_context(
    state: &BridgeState,
    operation: &HostOperation,
) -> std::result::Result<(), SkillFailure> {
    if let Some(hazard) = state.child_hazard.get(&state.current_child).copied() {
        match operation {
            HostOperation::Retreat {
                hazard: operation_hazard,
                ..
            } if *operation_hazard == hazard => {}
            HostOperation::Stop
            | HostOperation::Observe { .. }
            | HostOperation::VerifyCharging
            | HostOperation::WaitUntil { .. } => {}
            _ => {
                return Err(SkillFailure::new(
                    SkillOutcome::Failed,
                    "careful_envelope_violation",
                    "carefully permits only the acknowledged hazard's bounded retreat",
                ));
            }
        }
    }
    if let HostOperation::Retreat { hazard, .. } = operation {
        let Some(now) = state.snapshot.as_ref() else {
            return Err(SkillFailure::new(
                SkillOutcome::Failed,
                "now_unavailable",
                "cannot validate a careful retreat without Now",
            ));
        };
        let cliff_active = QueryPredicate::Cliff.evaluate(now);
        let contact_active = QueryPredicate::Contact.evaluate(now);
        let incompatible_hazard = match hazard {
            HazardKind::BumperFront => cliff_active,
            HazardKind::Cliff => contact_active,
        };
        if now.body.flags.wheel_drop || now.body.charging || incompatible_hazard {
            return Err(SkillFailure::new(
                SkillOutcome::SafetyPreempted,
                "absolute_hazard",
                "an absolute or incompatible hazard forbids careful retreat",
            ));
        }
        validate_hazard_acknowledged(now, *hazard)?;
    }
    Ok(())
}

struct Invocation {
    _lua: Lua,
    _set: Arc<LoadedSkillSet>,
    metadata: LoadedSkill,
    request: SkillRequest,
    thread: Pin<Box<AsyncThread<MultiValue>>>,
    bridge: Arc<Bridge>,
    started_at_ms: u64,
    metric_baseline: Option<f32>,
    active: HashMap<u64, ActiveOperation>,
    owners: BTreeMap<BodyResource, u64>,
    waiters: BTreeMap<BodyResource, VecDeque<OperationRequest>>,
    result: Option<std::result::Result<Value, SkillFailure>>,
    status: SkillStatus,
    diagnostics: SkillDiagnostic,
    trace: Vec<SkillTraceEvent>,
}

pub struct LuaSkillRuntime {
    config: LuaSkillConfig,
    active_set: Arc<LoadedSkillSet>,
    invocation: Option<Invocation>,
    next_execution_id: u64,
    attempts: BTreeMap<String, u32>,
    last_reload_error: Option<String>,
}

impl LuaSkillRuntime {
    pub fn load(config: LuaSkillConfig) -> Result<Self> {
        let active_set = Arc::new(load_skill_set(&config)?);
        Ok(Self {
            config,
            active_set,
            invocation: None,
            next_execution_id: 1,
            attempts: BTreeMap::new(),
            last_reload_error: None,
        })
    }

    pub fn discoverable_skills(&self) -> Vec<LoadedSkill> {
        self.active_set
            .skills
            .values()
            .map(|source| source.metadata.clone())
            .collect()
    }

    pub fn generation_hash(&self) -> &str {
        &self.active_set.generation_hash
    }

    pub fn last_reload_error(&self) -> Option<&str> {
        self.last_reload_error.as_deref()
    }

    pub fn reload(&mut self) -> Result<bool> {
        match load_skill_set(&self.config) {
            Ok(candidate) => {
                if candidate.generation_hash == self.active_set.generation_hash {
                    self.last_reload_error = None;
                    return Ok(false);
                }
                self.active_set = Arc::new(candidate);
                self.last_reload_error = None;
                Ok(true)
            }
            Err(error) => {
                self.last_reload_error = Some(error.to_string());
                Err(error)
            }
        }
    }

    pub fn is_active(&self) -> bool {
        self.invocation.is_some()
    }

    pub fn active_skill_id(&self) -> Option<&str> {
        self.invocation
            .as_ref()
            .map(|invocation| invocation.metadata.skill_id.as_str())
    }

    pub fn diagnostics(&self) -> SkillDiagnostic {
        self.invocation
            .as_ref()
            .map(|invocation| invocation.diagnostics.clone())
            .unwrap_or_else(|| SkillDiagnostic {
                phase: "idle".to_string(),
                ..SkillDiagnostic::default()
            })
    }

    pub fn trace(&self) -> Vec<SkillTraceEvent> {
        self.invocation
            .as_ref()
            .map(|invocation| invocation.trace.clone())
            .unwrap_or_default()
    }

    pub fn execution_record(&self) -> Option<SkillExecutionRecord> {
        let invocation = self.invocation.as_ref()?;
        let state = invocation.bridge.state.lock().expect("skill bridge lock");
        Some(SkillExecutionRecord {
            execution_id: invocation.status.execution_id,
            skill: invocation.metadata.clone(),
            request: invocation.request.clone(),
            diagnostics: invocation.diagnostics.clone(),
            trace: state.trace.iter().cloned().collect(),
            observations: state.observations.iter().take(128).cloned().collect(),
            memories: state.memories.iter().take(128).cloned().collect(),
        })
    }

    pub fn start(&mut self, request: SkillRequest, now: &Now) -> Result<()> {
        if self.invocation.is_some() {
            anyhow::bail!("a foreground Lua skill is already active");
        }
        let (skill_id, function_name) = if request.skill_id == SkillId::RuntimeLoaded {
            let skill_id = request
                .implementation_id
                .clone()
                .context("runtime-loaded skill request is missing implementation_id")?;
            let prefix = format!("{}.", self.config.namespace);
            let function_name = skill_id
                .strip_prefix(&prefix)
                .with_context(|| {
                    format!(
                        "runtime skill {skill_id} is outside configured namespace {}",
                        self.config.namespace
                    )
                })?
                .to_string();
            validate_identifier(&function_name)?;
            (skill_id, function_name)
        } else {
            let function_name = function_name_for_skill(request.skill_id).to_string();
            (
                format!("{}.{}", self.config.namespace, function_name),
                function_name,
            )
        };
        let source = self
            .active_set
            .skills
            .get(&skill_id)
            .with_context(|| format!("Lua skill {skill_id} is not loaded"))?
            .clone();
        let bridge = Arc::new(Bridge::default());
        bridge.state.lock().expect("skill bridge lock").snapshot = Some(now.clone());
        let lua = build_vm(&self.config, &self.active_set, bridge.clone())?;
        let function: Function = lua
            .globals()
            .get(function_name.as_str())
            .with_context(|| format!("loaded skill {skill_id} did not export {function_name}"))?;
        let arguments = request_arguments(&request, now);
        let lua_arguments = request_arguments_lua(&lua, &request, now)?;
        let thread = lua
            .create_thread(function)?
            .into_async::<MultiValue>(lua_arguments)?;
        let execution_id = self.next_execution_id;
        self.next_execution_id = execution_id.saturating_add(1);
        bridge.state.lock().expect("skill bridge lock").execution_id = execution_id;
        let metric_baseline = request.progress_baseline;
        let intention_key = intention_key(&request);
        let attempts = self.attempts.entry(intention_key).or_insert(0);
        *attempts = attempts.saturating_add(1);
        let status = SkillStatus {
            request: request.clone(),
            execution_id,
            phase: SkillPhase::Requested,
            outcome: None,
            progress: None,
            attempts: *attempts,
            dispatch_count: 0,
            started_at_ms: Some(now.t_ms),
            updated_at_ms: now.t_ms,
            reason: None,
            script: Some(pete_conductor::SkillScriptStatus {
                skill_id: skill_id.clone(),
                source_hash: source.metadata.source_hash.clone(),
                source_path: source.metadata.source_path.display().to_string(),
                current_function: Some(function_name.clone()),
                current_operation: None,
                held_resources: Vec::new(),
                waiting_resources: Vec::new(),
                active_children: 0,
            }),
        };
        let start = SkillTraceEvent::Started {
            at_ms: now.t_ms,
            skill_id: skill_id.clone(),
            source_hash: source.metadata.source_hash.clone(),
            arguments,
            starting_now: bounded_now_for_trace(now),
        };
        bridge.push_trace(start.clone());
        self.invocation = Some(Invocation {
            _lua: lua,
            _set: self.active_set.clone(),
            metadata: source.metadata.clone(),
            request,
            thread: Box::pin(thread),
            bridge,
            started_at_ms: now.t_ms,
            metric_baseline,
            active: HashMap::new(),
            owners: BTreeMap::new(),
            waiters: BTreeMap::new(),
            result: None,
            status,
            diagnostics: SkillDiagnostic {
                foreground_skill_id: Some(skill_id),
                source_hash: Some(source.metadata.source_hash),
                source_path: Some(source.metadata.source_path),
                start_time_ms: Some(now.t_ms),
                current_lua_function: Some(function_name),
                phase: "requested".to_string(),
                ..SkillDiagnostic::default()
            },
            trace: vec![start],
        });
        Ok(())
    }

    pub fn step<D: OrganDriver>(
        &mut self,
        now: &Now,
        events: &EventBatch,
        driver: &mut D,
    ) -> Option<SkillStatus> {
        let invocation = self.invocation.as_mut()?;
        invocation.status.updated_at_ms = now.t_ms;
        invocation.status.phase = SkillPhase::Running;
        invocation.diagnostics.phase = "running".to_string();
        invocation
            .bridge
            .state
            .lock()
            .expect("skill bridge lock")
            .snapshot = Some(now.clone());

        if invocation.request.maximum_duration_ms > 0
            && now.t_ms.saturating_sub(invocation.started_at_ms)
                >= invocation.request.maximum_duration_ms
        {
            cancel_invocation(
                invocation,
                driver,
                SkillFailure::new(
                    SkillOutcome::TimedOut,
                    "skill_timed_out",
                    "foreground skill exceeded its bounded duration",
                ),
            );
        }
        if let Some(failure) = external_preemption(events) {
            cancel_invocation(invocation, driver, failure);
        }

        service_child_cancellations(invocation, driver, now.t_ms);
        expire_resource_waits(invocation, now.t_ms);
        service_active_operations(
            invocation,
            driver,
            now,
            events,
            self.config.maximum_operation_ms,
        );
        grant_waiting_operations(invocation, now.t_ms);

        if invocation.result.is_none() {
            invocation.diagnostics.last_resume_ms = Some(now.t_ms);
            let previous = invocation.bridge.set_current_child(0);
            let poll_started = Instant::now();
            let waker = Waker::from(Arc::new(RuntimeWaker));
            let mut context = TaskContext::from_waker(&waker);
            let polled = catch_unwind(AssertUnwindSafe(|| {
                invocation.thread.as_mut().poll(&mut context)
            }));
            invocation.bridge.set_current_child(previous);
            if polled.is_err() {
                cancel_invocation(
                    invocation,
                    driver,
                    SkillFailure::new(
                        SkillOutcome::ScriptError,
                        "lua_vm_panic",
                        "embedded Lua VM panicked during activation",
                    ),
                );
            } else if poll_started.elapsed() > self.config.activation_budget {
                let failure = SkillFailure::new(
                    SkillOutcome::BudgetExceeded,
                    "wall_clock_budget_exceeded",
                    format!(
                        "Lua activation exceeded {} ms",
                        self.config.activation_budget.as_millis()
                    ),
                );
                cancel_invocation(invocation, driver, failure);
            } else {
                match polled.expect("Lua poll panic handled") {
                    Poll::Ready(Ok(values)) => {
                        let result = lua_values_to_json(&invocation._lua, values)
                            .and_then(|value| {
                                bounded_value(value, self.config.maximum_result_bytes)
                            })
                            .map_err(|error| {
                                SkillFailure::new(
                                    SkillOutcome::ScriptError,
                                    "result_conversion",
                                    error.to_string(),
                                )
                            });
                        invocation.result = Some(result);
                    }
                    Poll::Ready(Err(error)) => {
                        let failure = decode_lua_error(error);
                        cancel_invocation(invocation, driver, failure);
                    }
                    Poll::Pending => {
                        invocation.diagnostics.last_yield_ms = Some(now.t_ms);
                    }
                }
            }
        }

        drain_new_requests(invocation, now.t_ms);
        grant_waiting_operations(invocation, now.t_ms);
        update_diagnostics(invocation);

        if invocation.result.is_some() {
            finish_status(invocation, now.t_ms);
        } else {
            invocation.status.progress = skill_progress(invocation);
        }
        Some(invocation.status.clone())
    }

    pub fn cancel<D: OrganDriver>(
        &mut self,
        driver: &mut D,
        outcome: SkillOutcome,
        kind: impl Into<String>,
        message: impl Into<String>,
        now_ms: u64,
    ) -> Option<SkillStatus> {
        let invocation = self.invocation.as_mut()?;
        cancel_invocation(
            invocation,
            driver,
            SkillFailure::new(outcome, kind, message),
        );
        finish_status(invocation, now_ms);
        Some(invocation.status.clone())
    }

    pub fn take_terminal(&mut self) -> Option<(SkillStatus, Vec<SkillTraceEvent>)> {
        if !self
            .invocation
            .as_ref()
            .is_some_and(|invocation| invocation.status.phase == SkillPhase::Terminal)
        {
            return None;
        }
        let invocation = self.invocation.take()?;
        Some((invocation.status, invocation.trace))
    }

    pub fn shutdown<D: OrganDriver>(&mut self, driver: &mut D, now_ms: u64) {
        let _ = self.cancel(
            driver,
            SkillOutcome::Cancelled,
            "motherbrain_shutdown",
            "motherbrain shut down while the skill was active",
            now_ms,
        );
        driver.shutdown();
    }
}

struct RuntimeWaker;

impl Wake for RuntimeWaker {
    fn wake(self: Arc<Self>) {}
}

fn load_skill_set(config: &LuaSkillConfig) -> Result<LoadedSkillSet> {
    let mut paths = discover_lua_files(&config.directory)?;
    paths.sort();
    anyhow::ensure!(
        !paths.is_empty(),
        "no Lua skills found in {}",
        config.directory.display()
    );
    let loaded_at_ms = wall_time_ms();
    let runtime_version = "Lua 5.4 / mlua 0.11.6".to_string();
    let mut skills = BTreeMap::new();
    let mut ordered_sources = Vec::new();
    let mut generation_hasher = Sha256::new();
    for path in paths {
        let source = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read Lua skill {}", path.display()))?;
        let function_name = path
            .file_stem()
            .and_then(|value| value.to_str())
            .context("Lua skill filename is not valid UTF-8")?
            .to_string();
        validate_identifier(&function_name)?;
        let source_hash = hex_sha256(source.as_bytes());
        generation_hasher.update(path.to_string_lossy().as_bytes());
        generation_hasher.update(source_hash.as_bytes());
        let skill_id = format!("{}.{}", config.namespace, function_name);
        anyhow::ensure!(
            !skills.contains_key(&skill_id),
            "duplicate Lua skill ID {skill_id}"
        );
        let loaded = SkillSource {
            metadata: LoadedSkill {
                skill_id: skill_id.clone(),
                function_name,
                source_path: path,
                source_hash,
                loaded_at_ms,
                runtime_version: runtime_version.clone(),
            },
            source: Arc::from(source),
        };
        skills.insert(skill_id, loaded.clone());
        ordered_sources.push(loaded);
    }
    let set = LoadedSkillSet {
        generation_hash: format!("{:x}", generation_hasher.finalize()),
        skills,
        ordered_sources,
    };
    validate_skill_set(config, &set)?;
    Ok(set)
}

fn discover_lua_files(directory: &Path) -> Result<Vec<PathBuf>> {
    fn visit(path: &Path, found: &mut Vec<PathBuf>) -> Result<()> {
        for entry in std::fs::read_dir(path)
            .with_context(|| format!("failed to read skill directory {}", path.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let kind = entry.file_type()?;
            if kind.is_dir() {
                visit(&path, found)?;
            } else if kind.is_file() && path.extension().is_some_and(|ext| ext == "lua") {
                found.push(path);
            }
        }
        Ok(())
    }
    let mut found = Vec::new();
    visit(directory, &mut found)?;
    Ok(found)
}

fn validate_skill_set(config: &LuaSkillConfig, set: &LoadedSkillSet) -> Result<()> {
    let bridge = Arc::new(Bridge::default());
    bridge.state.lock().expect("skill bridge lock").snapshot =
        Some(Now::blank(0, pete_body::BodySense::default()));
    let lua = build_vm(config, set, bridge)?;
    for source in set.skills.values() {
        let value: LuaValue = lua
            .globals()
            .get(source.metadata.function_name.as_str())
            .with_context(|| {
                format!(
                    "{} did not export {}",
                    source.metadata.source_path.display(),
                    source.metadata.function_name
                )
            })?;
        anyhow::ensure!(
            matches!(value, LuaValue::Function(_)),
            "{} must export function {}",
            source.metadata.source_path.display(),
            source.metadata.function_name
        );
    }
    Ok(())
}

fn build_vm(config: &LuaSkillConfig, set: &LoadedSkillSet, bridge: Arc<Bridge>) -> Result<Lua> {
    let libraries = StdLib::TABLE | StdLib::STRING | StdLib::MATH | StdLib::UTF8;
    let lua = Lua::new_with(libraries, LuaOptions::default())?;
    lua.set_memory_limit(config.memory_limit_bytes)?;
    let remaining = Arc::new(AtomicU64::new(config.instruction_budget));
    let hook_remaining = remaining.clone();
    let hook_bridge = bridge.clone();
    // This must be global: ordinary skills, `try`, `carefully`, and every
    // `together` child execute in newly-created Lua threads.
    lua.set_global_hook(
        HookTriggers::new()
            .every_nth_instruction(1_000)
            .on_calls()
            .on_returns(),
        move |_, debug| {
            match debug.event() {
                DebugEvent::Count => {
                    let prior = hook_remaining.fetch_sub(1_000, Ordering::Relaxed);
                    if prior <= 1_000 {
                        return Err(SkillFailure::new(
                            SkillOutcome::BudgetExceeded,
                            "instruction_budget_exceeded",
                            "Lua instruction budget exhausted",
                        )
                        .encoded());
                    }
                }
                DebugEvent::Call | DebugEvent::TailCall => {
                    if let Some(name) = debug.names().name {
                        let mut state = hook_bridge.state.lock().expect("skill bridge lock");
                        let child = state.current_child;
                        state.current_functions.insert(child, name.into_owned());
                    }
                }
                DebugEvent::Ret => {
                    let mut state = hook_bridge.state.lock().expect("skill bridge lock");
                    let child = state.current_child;
                    state.current_functions.remove(&child);
                }
                _ => {}
            }
            Ok(VmState::Continue)
        },
    )?;
    install_api(&lua, bridge, config.maximum_operation_ms)?;
    for source in &set.ordered_sources {
        lua.load(source.source.as_ref())
            .set_name(source.metadata.source_path.to_string_lossy())
            .exec()
            .with_context(|| {
                format!(
                    "failed to load Lua skill {}",
                    source.metadata.source_path.display()
                )
            })?;
    }
    remove_forbidden_globals(&lua)?;
    Ok(lua)
}

fn remove_forbidden_globals(lua: &Lua) -> mlua::Result<()> {
    let globals = lua.globals();
    for name in [
        "io",
        "os",
        "debug",
        "package",
        "dofile",
        "loadfile",
        "load",
        "collectgarbage",
        "coroutine",
        "rawget",
        "rawset",
        "rawequal",
        "setmetatable",
        "getmetatable",
    ] {
        globals.set(name, LuaValue::Nil)?;
    }
    let math: Table = globals.get("math")?;
    math.set("random", LuaValue::Nil)?;
    math.set("randomseed", LuaValue::Nil)?;
    Ok(())
}

fn install_api(lua: &Lua, bridge: Arc<Bridge>, maximum_operation_ms: u64) -> mlua::Result<()> {
    install_operations(lua, bridge.clone(), maximum_operation_ms)?;
    install_queries(lua, bridge.clone())?;
    install_composition(lua, bridge.clone())?;
    install_provenance(lua, bridge)?;
    Ok(())
}

fn install_operations(
    lua: &Lua,
    bridge: Arc<Bridge>,
    maximum_operation_ms: u64,
) -> mlua::Result<()> {
    macro_rules! async_op {
        ($name:literal, $args:ty, $make:expr) => {{
            let bridge = bridge.clone();
            let make: Arc<dyn Fn($args) -> mlua::Result<HostOperation> + Send + Sync + 'static> =
                Arc::new($make);
            lua.globals().set(
                $name,
                lua.create_async_function(move |lua, args: $args| {
                    let bridge = bridge.clone();
                    let make = make.clone();
                    async move {
                        if bridge.cancelled.load(Ordering::Acquire) {
                            return Err(SkillFailure::new(
                                SkillOutcome::Cancelled,
                                "cancelled",
                                "foreground skill was cancelled",
                            )
                            .encoded());
                        }
                        let operation = make_operation(args, |args| make(args))?;
                        let value = bridge
                            .enqueue(operation, maximum_operation_ms)
                            .await
                            .map_err(|failure| failure.encoded())?;
                        lua.to_value(&value)
                    }
                })?,
            )?;
        }};
    }
    async_op!("stop", (), |_| Ok(HostOperation::Stop));
    async_op!("faceBearing", f32, |bearing| Ok(
        HostOperation::FaceBearing {
            bearing_rad: finite(bearing, "bearing")?,
        }
    ));
    async_op!("face", LuaValue, {
        let bridge = bridge.clone();
        move |target| {
            Ok(HostOperation::FaceBearing {
                bearing_rad: bearing_from_lua_value(&bridge, target)?,
            })
        }
    });
    async_op!("turn", f32, |angle| Ok(HostOperation::TurnBy {
        angle_rad: finite(angle, "angle")?,
    }));
    async_op!("turnBy", f32, |angle| Ok(HostOperation::TurnBy {
        angle_rad: finite(angle, "angle")?,
    }));
    async_op!("turnToward", LuaValue, {
        let bridge = bridge.clone();
        move |target| {
            Ok(HostOperation::FaceBearing {
                bearing_rad: bearing_from_lua_value(&bridge, target)?,
            })
        }
    });
    async_op!("drive", (f32, u64), |(velocity, duration_ms)| Ok(
        HostOperation::Drive {
            linear_m_s: finite(velocity, "velocity")?.clamp(-0.12, 0.12),
            duration_ms: duration_ms.min(30_000),
        }
    ));
    async_op!("driveDistance", (f32, Option<f32>), |(
        distance,
        velocity,
    )| {
        let distance = finite(distance, "distance")?.clamp(-5.0, 5.0);
        let direction = if distance < 0.0 { -1.0 } else { 1.0 };
        Ok(HostOperation::DriveDistance {
            distance_m: distance,
            velocity_m_s: finite(velocity.unwrap_or(0.08), "velocity")?
                .abs()
                .clamp(0.01, 0.12)
                * direction,
        })
    });
    async_op!("approach", (mlua::AnyUserData, Option<f32>), |(
        target,
        stop,
    )| {
        Ok(HostOperation::Approach {
            target: target.borrow::<EntityHandle>()?.clone(),
            stop_range_m: finite(stop.unwrap_or(0.30), "stop range")?.clamp(0.05, 5.0),
        })
    });
    async_op!("followBearing", (f32, Option<f32>), |(
        bearing,
        velocity,
    )| {
        Ok(HostOperation::FollowBearing {
            bearing_rad: finite(bearing, "bearing")?,
            linear_m_s: finite(velocity.unwrap_or(0.045), "velocity")?.clamp(0.0, 0.12),
        })
    });
    async_op!("holdHeading", (f32, Option<f32>), |(heading, velocity)| {
        Ok(HostOperation::HoldHeading {
            heading_rad: finite(heading, "heading")?,
            linear_m_s: finite(velocity.unwrap_or(0.0), "velocity")?.clamp(-0.12, 0.12),
        })
    });
    async_op!("followWall", (Option<String>, Option<f32>), |(
        side,
        distance,
    )| {
        let side = side.unwrap_or_else(|| "left".to_string());
        if side != "left" && side != "right" {
            return Err(LuaError::RuntimeError(
                "wall side must be left or right".into(),
            ));
        }
        Ok(HostOperation::FollowWall {
            side,
            distance_m: finite(distance.unwrap_or(0.25), "wall distance")?.clamp(0.05, 2.0),
        })
    });
    async_op!("scan", (), |_| Ok(HostOperation::Scan));
    async_op!("lookAt", mlua::AnyUserData, |target| {
        Ok(HostOperation::LookAt {
            target: target.borrow::<EntityHandle>()?.clone(),
        })
    });
    async_op!("observe", mlua::AnyUserData, |target| {
        Ok(HostOperation::Observe {
            target: target.borrow::<EntityHandle>()?.clone(),
        })
    });
    async_op!("searchForDockSignal", (), |_| Ok(
        HostOperation::SearchForDockSignal
    ));
    async_op!("alignWithDock", (), |_| Ok(HostOperation::AlignWithDock));
    async_op!("verifyCharging", (), |_| Ok(HostOperation::VerifyCharging));
    async_op!("undock", (), |_| Ok(HostOperation::Undock));
    async_op!("retreatFromContact", Option<f32>, |distance_mm| Ok(
        HostOperation::Retreat {
            hazard: HazardKind::BumperFront,
            distance_m: finite(distance_mm.unwrap_or(100.0), "distance")?.clamp(1.0, 500.0)
                / 1_000.0,
        }
    ));
    async_op!("retreatFromCliff", Option<f32>, |distance_mm| Ok(
        HostOperation::Retreat {
            hazard: HazardKind::Cliff,
            distance_m: finite(distance_mm.unwrap_or(100.0), "distance")?.clamp(1.0, 500.0)
                / 1_000.0,
        }
    ));
    async_op!("retreat", Option<f32>, {
        let bridge = bridge.clone();
        move |distance_mm| {
            let child = bridge.current_child();
            let hazard = bridge
                .state
                .lock()
                .expect("skill bridge lock")
                .child_hazard
                .get(&child)
                .copied()
                .ok_or_else(|| {
                    LuaError::RuntimeError(
                        "retreat must be called inside carefully(hazard, function)".into(),
                    )
                })?;
            Ok(HostOperation::Retreat {
                hazard,
                distance_m: finite(distance_mm.unwrap_or(100.0), "distance")?.clamp(1.0, 500.0)
                    / 1_000.0,
            })
        }
    });
    async_op!("completeHazardRecovery", String, |hazard| {
        let hazard = match hazard.as_str() {
            "bumper_front" | "bump" | "contact" => HazardKind::BumperFront,
            "cliff" => HazardKind::Cliff,
            _ => {
                return Err(LuaError::RuntimeError(
                    "completeHazardRecovery requires bumper_front or cliff".to_string(),
                ))
            }
        };
        Ok(HostOperation::CompleteHazardRecovery { hazard })
    });
    async_op!("releasePersistentBumper", (), |_| Ok(
        HostOperation::ReleasePersistentBumper
    ));
    async_op!("grasp", mlua::AnyUserData, |target| {
        Ok(HostOperation::Grasp {
            target: target.borrow::<EntityHandle>()?.clone(),
        })
    });
    async_op!("release", Option<mlua::AnyUserData>, |target| {
        Ok(HostOperation::Release {
            target: target
                .map(|target| target.borrow::<EntityHandle>().map(|value| value.clone()))
                .transpose()?,
        })
    });
    async_op!("bringToMouth", mlua::AnyUserData, |target| {
        Ok(HostOperation::BringToMouth {
            target: target.borrow::<EntityHandle>()?.clone(),
        })
    });
    async_op!("chew", (), |_| Ok(HostOperation::Chew));
    async_op!("swallow", (), |_| Ok(HostOperation::Swallow));
    async_op!("say", String, |text| {
        anyhow_lua(text.len() <= 4_096, "speech exceeds 4096 bytes")?;
        Ok(HostOperation::Say { text })
    });
    async_op!("playFeedback", String, |pattern| {
        anyhow_lua(pattern.len() <= 128, "feedback pattern is too long")?;
        Ok(HostOperation::PlayFeedback { pattern })
    });
    async_op!("waitUntil", (String, Option<u64>), |(
        predicate,
        timeout,
    )| {
        Ok(HostOperation::WaitUntil {
            predicate,
            timeout_ms: timeout.unwrap_or(5_000).clamp(1, 30_000),
        })
    });
    Ok(())
}

fn install_queries(lua: &Lua, bridge: Arc<Bridge>) -> mlua::Result<()> {
    let query_bridge = bridge.clone();
    lua.globals().set(
        "nearestVisible",
        lua.create_function(move |lua, kind: String| {
            let now = query_bridge.snapshot()?;
            let observation = now
                .objects
                .observations
                .iter()
                .filter(|observation| object_matches(&observation.class, &observation.label, &kind))
                .min_by(|left, right| {
                    left.distance_m
                        .unwrap_or(f32::INFINITY)
                        .total_cmp(&right.distance_m.unwrap_or(f32::INFINITY))
                });
            observation
                .map(|observation| {
                    lua.create_userdata(EntityHandle::new(
                        stable_observation_id(observation),
                        object_class_name(&observation.class),
                        observation.label.clone(),
                    ))
                })
                .transpose()
        })?,
    )?;
    let query_bridge = bridge.clone();
    lua.globals().set(
        "visible",
        lua.create_function(move |lua, kind: String| {
            let now = query_bridge.snapshot()?;
            let values = lua.create_table()?;
            for (index, observation) in now
                .objects
                .observations
                .iter()
                .filter(|observation| object_matches(&observation.class, &observation.label, &kind))
                .take(128)
                .enumerate()
            {
                values.set(
                    index + 1,
                    lua.create_userdata(
                        EntityHandle::new(
                            stable_observation_id(observation),
                            object_class_name(&observation.class),
                            observation.label.clone(),
                        )
                        .with_name(recognized_person_name(
                            &now,
                            &stable_observation_id(observation),
                        )),
                    )?,
                )?;
            }
            Ok(values)
        })?,
    )?;
    for (name, bearing) in [("distanceTo", false), ("bearingTo", true)] {
        let query_bridge = bridge.clone();
        lua.globals().set(
            name,
            lua.create_function(move |_, target: mlua::AnyUserData| {
                let target = target.borrow::<EntityHandle>()?;
                let now = query_bridge.snapshot()?;
                let value = if bearing {
                    current_entity_bearing(&now, &target)
                } else {
                    current_entity_distance(&now, &target)
                };
                value.ok_or_else(|| {
                    if current_entity_exists(&now, &target) {
                        SkillFailure::new(
                            SkillOutcome::Failed,
                            if bearing {
                                "bearing_unavailable"
                            } else {
                                "range_unavailable"
                            },
                            format!(
                                "entity {} has no current {}",
                                target.id,
                                if bearing { "bearing" } else { "range" }
                            ),
                        )
                        .encoded()
                    } else {
                        SkillFailure::new(
                            SkillOutcome::Failed,
                            "target_stale",
                            format!("entity {} is no longer visible", target.id),
                        )
                        .encoded()
                    }
                })
            })?,
        )?;
    }
    for (name, predicate) in [
        ("contactActive", QueryPredicate::Contact),
        ("cliffActive", QueryPredicate::Cliff),
        ("charging", QueryPredicate::Charging),
        ("cliffIsClear", QueryPredicate::CliffClear),
    ] {
        let query_bridge = bridge.clone();
        lua.globals().set(
            name,
            lua.create_function(move |_, ()| {
                let now = query_bridge.snapshot()?;
                Ok(predicate.evaluate(&now))
            })?,
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum QueryPredicate {
    Contact,
    Cliff,
    Charging,
    CliffClear,
}

impl QueryPredicate {
    fn evaluate(self, now: &Now) -> bool {
        match self {
            Self::Contact => now.body.flags.bump_left || now.body.flags.bump_right,
            Self::Cliff => {
                now.body.flags.cliff_left
                    || now.body.flags.cliff_front_left
                    || now.body.flags.cliff_front_right
                    || now.body.flags.cliff_right
            }
            Self::Charging => now.body.charging,
            Self::CliffClear => !Self::Cliff.evaluate(now),
        }
    }
}

fn install_composition(lua: &Lua, bridge: Arc<Bridge>) -> mlua::Result<()> {
    let together_bridge = bridge.clone();
    lua.globals().set(
        "together",
        lua.create_async_function(move |lua, functions: Variadic<Function>| {
            let bridge = together_bridge.clone();
            async move {
                anyhow_lua(
                    !functions.is_empty(),
                    "together requires at least one function",
                )?;
                anyhow_lua(
                    functions.len() <= 32,
                    "together supports at most 32 children",
                )?;
                let (parent, ids) = bridge.allocate_children(functions.len());
                let now_ms = bridge.snapshot()?.t_ms;
                let mut threads = Vec::with_capacity(functions.len());
                for (order, (function, child_id)) in
                    functions.into_iter().zip(ids.iter().copied()).enumerate()
                {
                    bridge.push_trace(SkillTraceEvent::ChildStarted {
                        at_ms: now_ms,
                        child_id,
                        parent_child_id: parent,
                        order,
                    });
                    threads.push(Some(Box::pin(
                        lua.create_thread(function)?.into_async::<MultiValue>(())?,
                    )));
                }
                let mut results: Vec<Option<MultiValue>> = vec![None; threads.len()];
                let outcome = std::future::poll_fn(|cx| {
                    let mut all_done = true;
                    for index in 0..threads.len() {
                        let Some(thread) = threads[index].as_mut() else {
                            continue;
                        };
                        all_done = false;
                        let previous = bridge.set_current_child(ids[index]);
                        let polled = thread.as_mut().poll(cx);
                        bridge.set_current_child(previous);
                        match polled {
                            Poll::Ready(Ok(values)) => {
                                results[index] = Some(values);
                                threads[index] = None;
                            }
                            Poll::Ready(Err(error)) => {
                                let failure = decode_lua_error(error);
                                bridge.cancel_children(
                                    ids.iter().copied().filter(|id| *id != ids[index]),
                                    failure.clone(),
                                );
                                return Poll::Ready(Err(failure.encoded()));
                            }
                            Poll::Pending => {}
                        }
                    }
                    if all_done || threads.iter().all(Option::is_none) {
                        Poll::Ready(Ok(()))
                    } else {
                        Poll::Pending
                    }
                })
                .await;
                bridge.set_current_child(parent);
                bridge.finish_children(&ids);
                outcome?;
                let table = lua.create_table()?;
                for (index, values) in results.into_iter().enumerate() {
                    let child_results = lua.create_table()?;
                    for (value_index, value) in values.unwrap_or_default().into_iter().enumerate() {
                        child_results.set(value_index + 1, value)?;
                    }
                    table.set(index + 1, child_results)?;
                }
                Ok(table)
            }
        })?,
    )?;

    let try_bridge = bridge.clone();
    lua.globals().set(
        "try",
        lua.create_async_function(move |lua, function: Function| {
            let bridge = try_bridge.clone();
            async move {
                let child = bridge.current_child();
                let previous = bridge.set_current_child(child);
                let result = lua
                    .create_thread(function)?
                    .into_async::<MultiValue>(())?
                    .await;
                bridge.set_current_child(previous);
                match result {
                    Ok(values) => {
                        let value = if values.len() == 1 {
                            values.into_iter().next().unwrap_or(LuaValue::Nil)
                        } else {
                            let table = lua.create_table()?;
                            for (index, value) in values.into_iter().enumerate() {
                                table.set(index + 1, value)?;
                            }
                            LuaValue::Table(table)
                        };
                        Ok((true, value))
                    }
                    Err(error) => {
                        let failure = decode_lua_error(error);
                        Ok((false, failure_to_lua(&lua, &failure)?))
                    }
                }
            }
        })?,
    )?;

    lua.globals().set(
        "blocked",
        lua.create_function(|_, value: LuaValue| Ok(value))?,
    )?;
    lua.globals().set(
        "error",
        lua.create_function(|lua, value: LuaValue| -> mlua::Result<()> {
            if let Ok(failure) = lua.from_value::<SkillFailure>(value.clone()) {
                return Err(failure.encoded());
            }
            let message = match value {
                LuaValue::String(value) => value.to_string_lossy(),
                other => format!("{other:?}"),
            };
            Err(LuaError::RuntimeError(message))
        })?,
    )?;
    lua.globals().set(
        "require",
        lua.create_function(|_, (condition, message): (bool, Option<String>)| {
            if condition {
                Ok(true)
            } else {
                Err(SkillFailure::new(
                    SkillOutcome::PostconditionFailed,
                    "postcondition_failed",
                    message.unwrap_or_else(|| "required postcondition was false".to_string()),
                )
                .encoded())
            }
        })?,
    )?;
    let careful_bridge = bridge;
    lua.globals().set(
        "carefully",
        lua.create_async_function(move |lua, (hazard, function): (LuaValue, Function)| {
            let bridge = careful_bridge.clone();
            async move {
                let child = bridge.current_child();
                let hazard = parse_hazard(&hazard)?;
                validate_hazard_acknowledged(&bridge.snapshot()?, hazard)
                    .map_err(|failure| failure.encoded())?;
                bridge
                    .state
                    .lock()
                    .expect("skill bridge lock")
                    .child_hazard
                    .insert(child, hazard);
                let result = lua
                    .create_thread(function)?
                    .into_async::<MultiValue>(())?
                    .await;
                bridge
                    .state
                    .lock()
                    .expect("skill bridge lock")
                    .child_hazard
                    .remove(&child);
                result
            }
        })?,
    )?;
    Ok(())
}

fn install_provenance(lua: &Lua, bridge: Arc<Bridge>) -> mlua::Result<()> {
    let progress_bridge = bridge.clone();
    lua.globals().set(
        "reportProgress",
        lua.create_function(move |_, (key, value): (String, f32)| {
            anyhow_lua(key.len() <= 128, "progress key is too long")?;
            let value = finite(value, "progress")?;
            let mut state = progress_bridge.state.lock().expect("skill bridge lock");
            anyhow_lua(
                state.reported_progress.contains_key(&key)
                    || state.reported_progress.len() < MAX_PROGRESS_ENTRIES,
                "too many progress values",
            )?;
            state
                .reported_progress
                .insert(key.clone(), value.clamp(0.0, 1.0));
            let at_ms = state.snapshot.as_ref().map_or(0, |now| now.t_ms);
            state
                .trace
                .push_back(SkillTraceEvent::Progress { at_ms, key, value });
            Ok(())
        })?,
    )?;
    let observation_bridge = bridge.clone();
    lua.globals().set(
        "hypothesize",
        lua.create_function(move |lua, (kind, value): (String, LuaValue)| {
            let value = bounded_lua_value(&lua, value)?;
            observation_bridge
                .state
                .lock()
                .expect("skill bridge lock")
                .observations
                .push(json!({
                    "kind": kind,
                    "value": value,
                    "provenance": "lua_skill",
                }));
            Ok(())
        })?,
    )?;
    let remember_bridge = bridge.clone();
    lua.globals().set(
        "remember",
        lua.create_function(move |lua, (key, value): (String, LuaValue)| {
            anyhow_lua(key.len() <= 128, "memory key is too long")?;
            let value = bounded_lua_value(&lua, value)?;
            remember_bridge
                .state
                .lock()
                .expect("skill bridge lock")
                .memories
                .push(json!({
                    "key": key,
                    "value": value,
                    "provenance": "lua_skill",
                }));
            Ok(())
        })?,
    )?;
    lua.globals().set(
        "acknowledge",
        lua.create_function(move |_, target: mlua::AnyUserData| {
            let target = target.borrow::<EntityHandle>()?.clone();
            let mut state = bridge.state.lock().expect("skill bridge lock");
            let now = state.snapshot.clone().ok_or_else(|| {
                LuaError::RuntimeError("canonical Now is unavailable".to_string())
            })?;
            let interaction = now
                .world
                .social
                .active_interaction
                .as_ref()
                .ok_or_else(|| {
                    SkillFailure::new(
                        SkillOutcome::PostconditionFailed,
                        "encounter_ended",
                        "cannot acknowledge a person outside an active encounter",
                    )
                    .encoded()
                })?;
            let participant = interaction
                .participants
                .iter()
                .find(|participant| participant.0.eq_ignore_ascii_case(target.id()))
                .ok_or_else(|| {
                    SkillFailure::new(
                        SkillOutcome::PostconditionFailed,
                        "person_not_in_encounter",
                        format!(
                            "{} is not a participant in the active encounter",
                            target.id()
                        ),
                    )
                    .encoded()
                })?;
            let acknowledgment_id = format!(
                "greet:{}:{}:{}",
                interaction.interaction_id.0, participant.0, state.execution_id
            );
            if !state.observations.iter().any(|observation| {
                observation
                    .pointer("/value/acknowledgment_id")
                    .and_then(Value::as_str)
                    == Some(acknowledgment_id.as_str())
            }) {
                state.observations.push(json!({
                    "kind": "social_acknowledgment",
                    "contract": "host_validated_social_acknowledgment_v1",
                    "value": {
                        "acknowledgment_id": acknowledgment_id,
                        "interaction_id": interaction.interaction_id.0,
                        "person_id": participant.0,
                        "occurred_at_ms": now.t_ms,
                    },
                    "provenance": "lua_skill",
                }));
            }
            Ok(true)
        })?,
    )?;
    Ok(())
}

fn drain_new_requests(invocation: &mut Invocation, now_ms: u64) {
    let requests: Vec<_> = invocation
        .bridge
        .state
        .lock()
        .expect("skill bridge lock")
        .requests
        .drain(..)
        .collect();
    for request in requests {
        if let Some(resource) = request.operation.resource() {
            if invocation
                .active
                .values()
                .any(|active| active.request.child_id == request.child_id)
                || invocation.waiters.values().any(|waiters| {
                    waiters
                        .iter()
                        .any(|waiting| waiting.child_id == request.child_id)
                })
            {
                request.slot.finish(Err(SkillFailure::new(
                    SkillOutcome::Failed,
                    "multiple_exclusive_resources",
                    "a child cannot await a second exclusive organ",
                )
                .for_operation(&request.operation)));
                continue;
            }
            if invocation.owners.contains_key(&resource) {
                invocation
                    .waiters
                    .entry(resource)
                    .or_default()
                    .push_back(request.clone());
                push_invocation_trace(
                    invocation,
                    SkillTraceEvent::ResourceWaiting {
                        at_ms: now_ms,
                        operation_id: request.id,
                        child_id: request.child_id,
                        resource,
                    },
                );
            } else {
                acquire(invocation, request, resource, now_ms);
            }
        } else {
            invocation.active.insert(
                request.id,
                ActiveOperation {
                    request,
                    granted_at_ms: now_ms,
                    polls: 0,
                },
            );
        }
    }
}

fn acquire(
    invocation: &mut Invocation,
    request: OperationRequest,
    resource: BodyResource,
    now_ms: u64,
) {
    invocation.owners.insert(resource, request.id);
    push_invocation_trace(
        invocation,
        SkillTraceEvent::ResourceAcquired {
            at_ms: now_ms,
            operation_id: request.id,
            child_id: request.child_id,
            resource,
        },
    );
    invocation.active.insert(
        request.id,
        ActiveOperation {
            request,
            granted_at_ms: now_ms,
            polls: 0,
        },
    );
}

fn grant_waiting_operations(invocation: &mut Invocation, now_ms: u64) {
    let resources: Vec<_> = invocation.waiters.keys().copied().collect();
    for resource in resources {
        if invocation.owners.contains_key(&resource) {
            continue;
        }
        let request = invocation
            .waiters
            .get_mut(&resource)
            .and_then(VecDeque::pop_front);
        if let Some(request) = request {
            acquire(invocation, request, resource, now_ms);
        }
        if invocation
            .waiters
            .get(&resource)
            .is_some_and(VecDeque::is_empty)
        {
            invocation.waiters.remove(&resource);
        }
    }
}

fn expire_resource_waits(invocation: &mut Invocation, now_ms: u64) {
    for waiters in invocation.waiters.values_mut() {
        let mut retained = VecDeque::new();
        while let Some(request) = waiters.pop_front() {
            if now_ms.saturating_sub(request.requested_at_ms) >= request.timeout_ms {
                request.slot.finish(Err(SkillFailure::new(
                    SkillOutcome::TimedOut,
                    "resource_wait_timed_out",
                    format!(
                        "timed out waiting for {:?}",
                        request.operation.resource().unwrap_or_default()
                    ),
                )
                .for_operation(&request.operation)));
            } else {
                retained.push_back(request);
            }
        }
        *waiters = retained;
    }
    invocation.waiters.retain(|_, waiters| !waiters.is_empty());
}

fn service_active_operations<D: OrganDriver>(
    invocation: &mut Invocation,
    driver: &mut D,
    now: &Now,
    events: &EventBatch,
    maximum_operation_ms: u64,
) {
    let ids: Vec<_> = invocation.active.keys().copied().collect();
    let mut finished = Vec::new();
    for id in ids {
        let Some(active) = invocation.active.get_mut(&id) else {
            continue;
        };
        let elapsed_ms = now.t_ms.saturating_sub(active.granted_at_ms);
        let timeout_ms = active.request.timeout_ms.min(maximum_operation_ms);
        if elapsed_ms >= timeout_ms {
            finished.push((
                id,
                Err(SkillFailure::new(
                    SkillOutcome::TimedOut,
                    "operation_timed_out",
                    format!(
                        "{} exceeded {} ms",
                        active.request.operation.name(),
                        timeout_ms
                    ),
                )
                .for_operation(&active.request.operation)),
            ));
            continue;
        }
        let context = OperationContext {
            operation_id: id,
            child_id: active.request.child_id,
            first_poll: active.polls == 0,
            elapsed_ms,
            now_ms: now.t_ms,
            primitive_ttl_ms: PRIMITIVE_TTL_MS,
        };
        active.polls = active.polls.saturating_add(1);
        let polled = catch_unwind(AssertUnwindSafe(|| {
            driver.poll(&active.request.operation, context, now, events)
        }))
        .unwrap_or_else(|_| {
            OrganPoll::Failed(
                SkillFailure::new(
                    SkillOutcome::ScriptError,
                    "organ_driver_panic",
                    "bodily operation driver panicked",
                )
                .for_operation(&active.request.operation),
            )
        });
        match polled {
            OrganPoll::Pending {
                progress,
                primitive,
            } => {
                if let Some((key, value)) = progress {
                    invocation.diagnostics.progress.insert(key.clone(), value);
                    push_invocation_trace(
                        invocation,
                        SkillTraceEvent::Progress {
                            at_ms: now.t_ms,
                            key,
                            value,
                        },
                    );
                }
                if let Some(primitive) = primitive {
                    invocation.status.dispatch_count =
                        invocation.status.dispatch_count.saturating_add(1);
                    push_invocation_trace(invocation, SkillTraceEvent::Primitive(primitive));
                }
            }
            OrganPoll::Completed(value) => finished.push((id, Ok(value))),
            OrganPoll::Failed(failure) => finished.push((id, Err(failure))),
        }
    }
    for (id, result) in finished {
        finish_operation(invocation, driver, id, result, now.t_ms);
    }
}

fn finish_operation<D: OrganDriver>(
    invocation: &mut Invocation,
    driver: &mut D,
    id: u64,
    result: std::result::Result<Value, SkillFailure>,
    now_ms: u64,
) {
    let Some(active) = invocation.active.remove(&id) else {
        return;
    };
    if let Some(resource) = active.request.operation.resource() {
        invocation.owners.remove(&resource);
        if result.is_err() {
            driver.stop(
                resource,
                result.as_ref().err().expect("failed operation has error"),
            );
        }
        push_invocation_trace(
            invocation,
            SkillTraceEvent::ResourceReleased {
                at_ms: now_ms,
                operation_id: id,
                child_id: active.request.child_id,
                resource,
                reason: result
                    .as_ref()
                    .err()
                    .map_or_else(|| "completed".to_string(), |failure| failure.kind.clone()),
            },
        );
    }
    if let Err(failure) = &result {
        push_invocation_trace(
            invocation,
            SkillTraceEvent::Preempted {
                at_ms: now_ms,
                operation_id: id,
                failure: failure.clone(),
            },
        );
        invocation.diagnostics.last_preemption = Some(failure.clone());
    }
    active.request.slot.finish(result);
}

fn service_child_cancellations<D: OrganDriver>(
    invocation: &mut Invocation,
    driver: &mut D,
    now_ms: u64,
) {
    let cancellations: Vec<_> = invocation
        .bridge
        .state
        .lock()
        .expect("skill bridge lock")
        .cancelled_children
        .drain(..)
        .collect();
    for (children, failure) in cancellations {
        let active_ids: Vec<_> = invocation
            .active
            .iter()
            .filter_map(|(id, active)| children.contains(&active.request.child_id).then_some(*id))
            .collect();
        for id in active_ids {
            finish_operation(invocation, driver, id, Err(failure.clone()), now_ms);
        }
        for waiters in invocation.waiters.values_mut() {
            let mut retained = VecDeque::new();
            while let Some(request) = waiters.pop_front() {
                if children.contains(&request.child_id) {
                    request.slot.finish(Err(failure.clone()));
                } else {
                    retained.push_back(request);
                }
            }
            *waiters = retained;
        }
    }
}

fn cancel_invocation<D: OrganDriver>(
    invocation: &mut Invocation,
    driver: &mut D,
    failure: SkillFailure,
) {
    if invocation.result.is_some() {
        return;
    }
    invocation.bridge.cancelled.store(true, Ordering::Release);
    let ids: Vec<_> = invocation.active.keys().copied().collect();
    for id in ids {
        finish_operation(
            invocation,
            driver,
            id,
            Err(failure.clone()),
            invocation.status.updated_at_ms,
        );
    }
    for waiters in invocation.waiters.values_mut() {
        for request in waiters.drain(..) {
            request.slot.finish(Err(failure.clone()));
        }
    }
    invocation.waiters.clear();
    for request in invocation
        .bridge
        .state
        .lock()
        .expect("skill bridge lock")
        .requests
        .drain(..)
    {
        request.slot.finish(Err(failure.clone()));
    }
    invocation.result = Some(Err(failure));
    update_diagnostics(invocation);
}

fn update_diagnostics(invocation: &mut Invocation) {
    invocation.diagnostics.held_resources = invocation
        .owners
        .iter()
        .filter_map(|(resource, operation_id)| {
            invocation
                .active
                .get(operation_id)
                .map(|active| (*resource, active.request.child_id))
        })
        .collect();
    invocation.diagnostics.waiting_resources = invocation
        .waiters
        .iter()
        .map(|(resource, waiters)| {
            (
                *resource,
                waiters.iter().map(|request| request.child_id).collect(),
            )
        })
        .collect();
    invocation.diagnostics.current_operation = invocation
        .active
        .values()
        .min_by_key(|active| active.request.id)
        .map(|active| active.request.operation.name().to_string());
    let bridge = invocation.bridge.state.lock().expect("skill bridge lock");
    invocation.diagnostics.current_lua_function = bridge
        .current_functions
        .get(&0)
        .cloned()
        .or_else(|| Some(invocation.metadata.function_name.clone()));
    for (key, value) in &bridge.progress {
        invocation.diagnostics.progress.insert(key.clone(), *value);
    }
    for (key, value) in &bridge.reported_progress {
        invocation.diagnostics.progress.insert(key.clone(), *value);
    }
    invocation.diagnostics.active_together_children = bridge
        .child_parent
        .iter()
        .map(|(child_id, (parent, order))| {
            let active = invocation
                .active
                .values()
                .find(|active| active.request.child_id == *child_id);
            let waiting = invocation.waiters.iter().find_map(|(resource, waiters)| {
                waiters
                    .iter()
                    .any(|request| request.child_id == *child_id)
                    .then_some(*resource)
            });
            ChildDiagnostic {
                child_id: *child_id,
                parent_child_id: *parent,
                order: *order,
                phase: if active.is_some() {
                    "operating"
                } else if waiting.is_some() {
                    "waiting"
                } else {
                    "lua"
                }
                .to_string(),
                current_function: bridge.current_functions.get(child_id).cloned(),
                current_operation: active.map(|active| active.request.operation.name().to_string()),
                held_resource: active.and_then(|active| active.request.operation.resource()),
                waiting_resource: waiting,
            }
        })
        .collect();
    if let Some(script) = invocation.status.script.as_mut() {
        script.current_operation = invocation.diagnostics.current_operation.clone();
        script.current_function = invocation.diagnostics.current_lua_function.clone();
        script.held_resources = invocation
            .diagnostics
            .held_resources
            .keys()
            .map(|resource| format!("{resource:?}").to_lowercase())
            .collect();
        script.waiting_resources = invocation
            .diagnostics
            .waiting_resources
            .keys()
            .map(|resource| format!("{resource:?}").to_lowercase())
            .collect();
        script.active_children = invocation.diagnostics.active_together_children.len() as u32;
    }
}

fn finish_status(invocation: &mut Invocation, now_ms: u64) {
    if invocation.status.phase == SkillPhase::Terminal {
        return;
    }
    let (outcome, reason, detail, result) = match invocation.result.as_ref() {
        Some(Ok(value)) => (SkillOutcome::Completed, None, None, Some(value.clone())),
        Some(Err(failure)) => (
            failure.outcome,
            Some(failure.message.clone()),
            Some(failure.clone()),
            None,
        ),
        None => return,
    };
    invocation.status.phase = SkillPhase::Terminal;
    invocation.status.outcome = Some(outcome);
    invocation.status.reason = reason;
    invocation.status.updated_at_ms = now_ms;
    invocation.status.progress = (outcome == SkillOutcome::Completed)
        .then_some(1.0)
        .or_else(|| skill_progress(invocation));
    invocation.diagnostics.phase = "terminal".to_string();
    invocation.diagnostics.terminal_outcome = Some(outcome);
    invocation.diagnostics.terminal_detail = detail.clone();
    let completed = SkillTraceEvent::Completed {
        at_ms: now_ms,
        outcome,
        detail,
        duration_ms: now_ms.saturating_sub(invocation.started_at_ms),
        result,
    };
    push_invocation_trace(invocation, completed);
}

fn external_preemption(events: &EventBatch) -> Option<SkillFailure> {
    for event in &events.events {
        let event_json = serde_json::to_value(event).unwrap_or(Value::Null);
        match event.kind {
            CockpitEventKind::ContactWithdrawalStarted => {
                return Some(SkillFailure::safety(
                    "contact_withdrawal",
                    "Brainstem contact-withdrawal reflex preempted the operation",
                    event_json,
                ));
            }
            CockpitEventKind::SafetyTripped | CockpitEventKind::EStopLatched => {
                return Some(SkillFailure::safety(
                    "brainstem_safety",
                    "Brainstem safety preempted the operation",
                    event_json,
                ));
            }
            CockpitEventKind::SessionReplaced | CockpitEventKind::AuthorityChanged => {
                return Some(SkillFailure::new(
                    SkillOutcome::AuthorityLost,
                    "authority_lost",
                    "possession authority changed while the skill was active",
                ));
            }
            CockpitEventKind::SessionClosed
            | CockpitEventKind::PeerRebootDetected
            | CockpitEventKind::HeartbeatExpired => {
                return Some(SkillFailure::new(
                    SkillOutcome::TransportLost,
                    "transport_lost",
                    "body transport or supervised heartbeat was lost",
                ));
            }
            _ => {}
        }
    }
    None
}

fn decode_lua_error(error: LuaError) -> SkillFailure {
    let rendered = error.to_string();
    if let Some(index) = rendered.find(LUA_ERROR_PREFIX) {
        let json = &rendered[index + LUA_ERROR_PREFIX.len()..];
        let json = json.lines().next().unwrap_or(json);
        if let Ok(failure) = serde_json::from_str(json) {
            return failure;
        }
    }
    if rendered.contains("memory error") || rendered.contains("not enough memory") {
        return SkillFailure::new(
            SkillOutcome::BudgetExceeded,
            "memory_budget_exceeded",
            "Lua memory budget exhausted",
        );
    }
    if rendered.contains("stack overflow") {
        return SkillFailure::new(
            SkillOutcome::BudgetExceeded,
            "stack_budget_exceeded",
            "Lua recursion or stack budget exhausted",
        );
    }
    SkillFailure::new(SkillOutcome::ScriptError, "script_error", rendered)
}

fn failure_to_lua(lua: &Lua, failure: &SkillFailure) -> mlua::Result<LuaValue> {
    lua.to_value(failure)
}

fn parse_hazard(value: &LuaValue) -> mlua::Result<HazardKind> {
    match value {
        LuaValue::String(value) => match value.to_str()?.as_ref() {
            "bumper_front" | "bump" | "contact" => Ok(HazardKind::BumperFront),
            "cliff" => Ok(HazardKind::Cliff),
            other => Err(LuaError::RuntimeError(format!(
                "unsupported CAREFUL hazard {other}"
            ))),
        },
        LuaValue::Table(table) => {
            let kind: String = table.get("kind")?;
            match kind.as_str() {
                "bumper_front" | "bump" | "contact" => Ok(HazardKind::BumperFront),
                "cliff" => Ok(HazardKind::Cliff),
                other => Err(LuaError::RuntimeError(format!(
                    "unsupported CAREFUL hazard {other}"
                ))),
            }
        }
        _ => Err(LuaError::RuntimeError(
            "carefully requires a validated hazard name or handle".into(),
        )),
    }
}

fn validate_hazard_acknowledged(
    now: &Now,
    hazard: HazardKind,
) -> std::result::Result<(), SkillFailure> {
    let active = match hazard {
        HazardKind::BumperFront => now.body.flags.bump_left || now.body.flags.bump_right,
        HazardKind::Cliff => QueryPredicate::Cliff.evaluate(now),
    };
    if active {
        Ok(())
    } else {
        Err(SkillFailure::new(
            SkillOutcome::Failed,
            "hazard_not_acknowledged",
            format!("{hazard:?} is not active in the current Now"),
        ))
    }
}

fn request_arguments(request: &SkillRequest, now: &Now) -> Value {
    let target = request.target.as_ref().map(|target| {
        let id = target.0.clone();
        let observation = now
            .objects
            .observations
            .iter()
            .find(|observation| stable_observation_id(observation) == id);
        json!({
            "id": id,
            "kind": observation.map(|value| object_class_name(&value.class)).unwrap_or("unknown"),
            "label": observation.map(|value| value.label.as_str()).unwrap_or("target"),
        })
    });
    json!({
        "goal_id": request.goal_id.as_ref().map(|value| value.as_str()),
        "implementation_id": request.implementation_id,
        "behavior_id": request.behavior_id,
        "target": target,
        "bearing_rad": request.bearing_rad,
        "angle_rad": request.bearing_rad,
        "range_m": request.range_m,
        "distance_m": request.range_m,
        "stop_range_m": request.stop_range_m,
        "maximum_duration_ms": request.maximum_duration_ms,
        "expected_progress": request.expected_progress,
        "progress_metric": request.progress_metric,
    })
}

fn request_arguments_lua(lua: &Lua, request: &SkillRequest, now: &Now) -> mlua::Result<Table> {
    let table = lua.create_table()?;
    table.set(
        "goal_id",
        request.goal_id.as_ref().map(|value| value.as_str()),
    )?;
    table.set("implementation_id", request.implementation_id.clone())?;
    table.set("behavior_id", request.behavior_id.clone())?;
    table.set("bearing_rad", request.bearing_rad)?;
    table.set("angle_rad", request.bearing_rad)?;
    table.set("range_m", request.range_m)?;
    table.set("distance_m", request.range_m)?;
    table.set("stop_range_m", request.stop_range_m)?;
    table.set("maximum_duration_ms", request.maximum_duration_ms)?;
    table.set("expected_progress", request.expected_progress)?;
    table.set("progress_metric", request.progress_metric.clone())?;
    if let Some(target) = request.target.as_ref() {
        let observation =
            now.objects.observations.iter().find(|observation| {
                stable_observation_id(observation).eq_ignore_ascii_case(&target.0)
            });
        let world_entity = now
            .world
            .entities
            .iter()
            .find(|(id, _)| id.0.eq_ignore_ascii_case(&target.0))
            .map(|(_, entity)| entity);
        let social_person = now
            .world
            .social
            .people
            .iter()
            .find(|(person_id, _)| person_id.0.eq_ignore_ascii_case(&target.0))
            .map(|(_, person)| person);
        let kind = observation
            .map(|value| object_class_name(&value.class))
            .or_else(|| world_entity.map(|entity| world_entity_kind_name(&entity.kind)))
            .or_else(|| social_person.map(|_| "person"))
            .unwrap_or("unknown");
        let label = observation
            .map(|value| value.label.as_str())
            .or_else(|| world_entity.map(|entity| entity.label.as_str()))
            .or_else(|| {
                social_person
                    .and_then(|person| person.preferred_name.as_ref())
                    .map(|name| name.value.as_str())
            })
            .unwrap_or("target");
        table.set(
            "target",
            lua.create_userdata(
                EntityHandle::new(target.0.clone(), kind, label)
                    .with_name(recognized_person_name(now, &target.0)),
            )?,
        )?;
    }
    Ok(table)
}

fn function_name_for_skill(skill: SkillId) -> &'static str {
    match skill {
        SkillId::StopAndStabilize => "stopAndStabilize",
        SkillId::TurnTowardTarget => "turnTowardTarget",
        SkillId::FollowBearing => "followBearingSkill",
        SkillId::ApproachTarget => "approachTarget",
        SkillId::BackAway => "driveFor",
        SkillId::InspectTarget => "inspectObject",
        SkillId::WallFollow => "wallFollow",
        SkillId::AlignWithDock => "alignWithDockSkill",
        SkillId::SystematicSearch => "systematicSearch",
        SkillId::HoldHeading => "holdHeadingSkill",
        SkillId::RetreatFromCliff => "retreatFromCliff",
        SkillId::ReleasePersistentBumper => "releasePersistentBumper",
        SkillId::TurnBy => "turnBySkill",
        SkillId::DriveDistance => "driveDistanceSkill",
        SkillId::Undock => "undockSkill",
        SkillId::SearchForDock => "searchForDock",
        SkillId::ReturnToDock => "returnToDock",
        SkillId::RuntimeLoaded => unreachable!("runtime-loaded skills provide implementation_id"),
    }
}

fn intention_key(request: &SkillRequest) -> String {
    format!(
        "{:?}:{:?}:{:?}:{:?}",
        request.skill_id, request.goal_id, request.behavior_id, request.target
    )
}

fn bounded_now_for_trace(now: &Now) -> Value {
    json!({
        "t_ms": now.t_ms,
        "body": {
            "charging": now.body.charging,
            "flags": now.body.flags,
            "odometry": now.body.odometry,
            "infrared_character": now.body.infrared_character,
        },
        "objects": now.objects.observations.iter().take(32).collect::<Vec<_>>(),
        "active_goal": now.self_sense.active_goal,
    })
}

fn bounded_lua_value(lua: &Lua, value: LuaValue) -> mlua::Result<Value> {
    let value: Value = lua.from_value(value)?;
    bounded_value(value, MAX_CONVERTED_VALUE_BYTES)
        .map_err(|error| LuaError::RuntimeError(error.to_string()))
}

fn lua_values_to_json(lua: &Lua, values: MultiValue) -> Result<Value> {
    if values.is_empty() {
        return Ok(Value::Null);
    }
    if values.len() == 1 {
        return Ok(lua.from_value(values.into_iter().next().unwrap_or(LuaValue::Nil))?);
    }
    let mut converted = Vec::new();
    for value in values {
        converted.push(lua.from_value(value)?);
    }
    Ok(Value::Array(converted))
}

fn bounded_value(value: Value, maximum_bytes: usize) -> Result<Value> {
    let encoded = serde_json::to_vec(&value)?;
    anyhow::ensure!(
        encoded.len() <= maximum_bytes,
        "Lua value exceeds {maximum_bytes} bytes"
    );
    Ok(value)
}

fn push_invocation_trace(invocation: &mut Invocation, event: SkillTraceEvent) {
    if invocation.trace.len() == MAX_TRACE_EVENTS {
        invocation.trace.remove(0);
    }
    invocation.bridge.push_trace(event.clone());
    invocation.trace.push(event);
}

fn skill_progress(invocation: &mut Invocation) -> Option<f32> {
    let state = invocation.bridge.state.lock().expect("skill bridge lock");
    let metric = invocation.request.progress_metric.as_str();
    if let Some(reported) = state
        .reported_progress
        .get(metric)
        .copied()
        .or_else(|| state.reported_progress.get("goal_progress").copied())
    {
        return Some(reported.clamp(0.0, 1.0));
    }
    let current = state
        .progress
        .get(metric)
        .copied()
        .or_else(|| invocation.diagnostics.progress.get(metric).copied())?;
    match metric {
        "bearing_error" | "target_distance" => {
            let baseline = *invocation.metric_baseline.get_or_insert(current);
            if baseline <= f32::EPSILON {
                Some(if current <= invocation.request.progress_tolerance {
                    1.0
                } else {
                    0.0
                })
            } else {
                Some(((baseline - current) / baseline).clamp(0.0, 1.0))
            }
        }
        "reverse_displacement"
        | "uncertainty_reduction"
        | "path_progress"
        | "frontier_coverage"
        | "motion_stability" => Some(current.clamp(0.0, 1.0)),
        _ => Some(current.clamp(0.0, 1.0)),
    }
}

fn stable_observation_id(observation: &pete_now::ObjectObservation) -> String {
    format!(
        "{}:{}",
        object_class_name(&observation.class),
        observation.label
    )
}

fn current_entity_exists(now: &Now, target: &EntityHandle) -> bool {
    now.objects
        .observations
        .iter()
        .any(|observation| stable_observation_id(observation).eq_ignore_ascii_case(target.id()))
        || now
            .world
            .entities
            .keys()
            .any(|id| id.0.eq_ignore_ascii_case(target.id()))
        || now.world.social.people.iter().any(|(person_id, person)| {
            person_id.0.eq_ignore_ascii_case(target.id()) && person.presence.present
        })
}

fn current_entity_bearing(now: &Now, target: &EntityHandle) -> Option<f32> {
    now.objects
        .observations
        .iter()
        .find(|observation| stable_observation_id(observation).eq_ignore_ascii_case(target.id()))
        .map(|observation| observation.bearing_rad)
        .or_else(|| {
            now.world
                .entities
                .iter()
                .find(|(id, _)| id.0.eq_ignore_ascii_case(target.id()))
                .and_then(|(_, entity)| entity.bearing_rad)
        })
        .or_else(|| {
            now.world
                .social
                .people
                .iter()
                .find(|(person_id, _)| person_id.0.eq_ignore_ascii_case(target.id()))
                .and_then(|(_, person)| person.location.as_ref())
                .and_then(|location| location.bearing_rad)
        })
}

fn current_entity_distance(now: &Now, target: &EntityHandle) -> Option<f32> {
    now.objects
        .observations
        .iter()
        .find(|observation| stable_observation_id(observation).eq_ignore_ascii_case(target.id()))
        .and_then(|observation| observation.distance_m)
        .or_else(|| {
            now.world
                .entities
                .iter()
                .find(|(id, _)| id.0.eq_ignore_ascii_case(target.id()))
                .and_then(|(_, entity)| entity.distance_m)
        })
        .or_else(|| {
            now.world
                .social
                .people
                .iter()
                .find(|(person_id, _)| person_id.0.eq_ignore_ascii_case(target.id()))
                .and_then(|(_, person)| person.location.as_ref())
                .and_then(|location| location.distance_m)
        })
}

fn recognized_person_name(now: &Now, target_id: &str) -> Option<String> {
    now.world
        .social
        .people
        .iter()
        .find(|(person_id, _)| person_id.0.eq_ignore_ascii_case(target_id))
        .and_then(|(_, person)| {
            (!person.identity_is_uncertain())
                .then(|| {
                    person
                        .preferred_name
                        .as_ref()
                        .map(|name| name.value.clone())
                })
                .flatten()
        })
}

fn object_matches(class: &ObjectClass, label: &str, wanted: &str) -> bool {
    label.eq_ignore_ascii_case(wanted) || object_class_name(class).eq_ignore_ascii_case(wanted)
}

fn object_class_name(class: &ObjectClass) -> &'static str {
    match class {
        ObjectClass::Obstacle => "obstacle",
        ObjectClass::Charger => "dock",
        ObjectClass::Person => "person",
        ObjectClass::SoundSource => "sound",
        ObjectClass::Landmark => "landmark",
        ObjectClass::Unknown => "unknown",
    }
}

fn world_entity_kind_name(kind: &pete_now::WorldEntityKind) -> &'static str {
    match kind {
        pete_now::WorldEntityKind::Obstacle => "obstacle",
        pete_now::WorldEntityKind::Charger => "dock",
        pete_now::WorldEntityKind::Person => "person",
        pete_now::WorldEntityKind::SoundSource => "sound",
        pete_now::WorldEntityKind::Landmark => "landmark",
        pete_now::WorldEntityKind::Door => "door",
        pete_now::WorldEntityKind::Region => "region",
        pete_now::WorldEntityKind::Unknown => "unknown",
    }
}

fn validate_identifier(value: &str) -> Result<()> {
    let mut chars = value.chars();
    anyhow::ensure!(
        chars
            .next()
            .is_some_and(|value| value == '_' || value.is_ascii_alphabetic())
            && chars.all(|value| value == '_' || value.is_ascii_alphanumeric()),
        "{value:?} is not a valid Lua function identifier"
    );
    Ok(())
}

fn finite(value: f32, name: &str) -> mlua::Result<f32> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(LuaError::RuntimeError(format!("{name} must be finite")))
    }
}

fn bearing_from_lua_value(bridge: &Bridge, value: LuaValue) -> mlua::Result<f32> {
    match value {
        LuaValue::Integer(value) => finite(value as f32, "bearing"),
        LuaValue::Number(value) => finite(value as f32, "bearing"),
        LuaValue::UserData(target) => {
            let target = target.borrow::<EntityHandle>()?;
            let now = bridge.snapshot()?;
            current_entity_bearing(&now, &target).ok_or_else(|| {
                if current_entity_exists(&now, &target) {
                    SkillFailure::new(
                        SkillOutcome::Failed,
                        "bearing_unavailable",
                        format!("entity {} has no current bearing", target.id),
                    )
                    .encoded()
                } else {
                    SkillFailure::new(
                        SkillOutcome::Failed,
                        "target_stale",
                        format!("entity {} is no longer visible", target.id),
                    )
                    .encoded()
                }
            })
        }
        _ => Err(LuaError::RuntimeError(
            "face/turnToward requires a bearing or entity handle".to_string(),
        )),
    }
}

fn make_operation<A, F>(args: A, make: F) -> mlua::Result<HostOperation>
where
    F: FnOnce(A) -> mlua::Result<HostOperation>,
{
    make(args)
}

fn anyhow_lua(condition: bool, message: &str) -> mlua::Result<()> {
    if condition {
        Ok(())
    } else {
        Err(LuaError::RuntimeError(message.to_string()))
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn wall_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use pete_body::BodySense;
    use pete_cockpit::{CockpitEvent, EventBatch};
    use pete_conductor::{GoalId, SkillScriptStatus};
    use pete_now::EntityId;
    use std::collections::BTreeSet;
    use std::fs;
    use tempfile::TempDir;

    #[derive(Clone, Debug)]
    struct OperationRecord {
        operation_id: u64,
        child_id: u64,
        name: String,
        resource: Option<BodyResource>,
        started_at_ms: u64,
        ended_at_ms: Option<u64>,
    }

    #[derive(Default)]
    struct FakeDriver {
        records: Vec<OperationRecord>,
        stops: Vec<(BodyResource, String)>,
        fail_operations: HashMap<String, SkillFailure>,
        panic_operations: HashSet<String>,
        duration_ms: HashMap<String, u64>,
    }

    impl FakeDriver {
        fn duration_for(&self, operation: &HostOperation) -> u64 {
            self.duration_ms
                .get(operation.name())
                .copied()
                .unwrap_or(200)
        }
    }

    impl OrganDriver for FakeDriver {
        fn poll(
            &mut self,
            operation: &HostOperation,
            context: OperationContext,
            _now: &Now,
            _events: &EventBatch,
        ) -> OrganPoll {
            if context.first_poll {
                self.records.push(OperationRecord {
                    operation_id: context.operation_id,
                    child_id: context.child_id,
                    name: operation.name().to_string(),
                    resource: operation.resource(),
                    started_at_ms: context.now_ms,
                    ended_at_ms: None,
                });
            }
            assert!(
                !self.panic_operations.contains(operation.name()),
                "simulated organ driver panic"
            );
            if let Some(failure) = self.fail_operations.get(operation.name()).cloned() {
                self.records
                    .iter_mut()
                    .find(|record| record.operation_id == context.operation_id)
                    .unwrap()
                    .ended_at_ms = Some(context.now_ms);
                return OrganPoll::Failed(failure.for_operation(operation));
            }
            if context.elapsed_ms >= self.duration_for(operation) {
                self.records
                    .iter_mut()
                    .find(|record| record.operation_id == context.operation_id)
                    .unwrap()
                    .ended_at_ms = Some(context.now_ms);
                return OrganPoll::Completed(json!({
                    "operation": operation.name(),
                    "child_id": context.child_id,
                }));
            }
            OrganPoll::Pending {
                progress: operation.resource().map(|_| {
                    (
                        "goal_progress".to_string(),
                        context.elapsed_ms as f32 / 200.0,
                    )
                }),
                primitive: operation.resource().map(|resource| PrimitiveIntent {
                    operation_id: context.operation_id,
                    child_id: context.child_id,
                    operation: operation.name().to_string(),
                    resource: Some(resource),
                    emitted_at_ms: context.now_ms,
                    detail: json!({"fake": true}),
                }),
            }
        }

        fn stop(&mut self, resource: BodyResource, reason: &SkillFailure) {
            self.stops.push((resource, reason.kind.clone()));
        }
    }

    fn empty_events() -> EventBatch {
        EventBatch {
            since_seq: 0,
            oldest_seq: 0,
            next_seq: 0,
            dropped_before_seq: 0,
            events: Vec::new(),
        }
    }

    fn event(kind: CockpitEventKind) -> EventBatch {
        EventBatch {
            events: vec![CockpitEvent {
                seq: 1,
                kind,
                a: 0,
                b: 0,
                c: 0,
            }],
            next_seq: 2,
            ..empty_events()
        }
    }

    fn request(skill_id: SkillId) -> SkillRequest {
        SkillRequest {
            skill_id,
            goal_id: Some(GoalId::new("test_goal")),
            target: Some(EntityId("food:apple".to_string())),
            bearing_rad: Some(0.5),
            range_m: Some(2.0),
            stop_range_m: Some(0.2),
            maximum_duration_ms: 10_000,
            progress_metric: "goal_progress".to_string(),
            progress_baseline: Some(0.0),
            ..SkillRequest::default()
        }
    }

    fn write_skill(directory: &Path, function_name: &str, source: &str) {
        fs::write(directory.join(format!("{function_name}.lua")), source).unwrap();
    }

    fn runtime_with(function_name: &str, source: &str) -> (TempDir, LuaSkillRuntime, Now) {
        let directory = TempDir::new().unwrap();
        write_skill(directory.path(), function_name, source);
        let config = LuaSkillConfig {
            directory: directory.path().to_path_buf(),
            namespace: "test".to_string(),
            instruction_budget: 100_000,
            activation_budget: Duration::from_millis(50),
            memory_limit_bytes: 2 * 1024 * 1024,
            maximum_result_bytes: 16 * 1024,
            maximum_operation_ms: 2_000,
        };
        let runtime = LuaSkillRuntime::load(config).unwrap();
        let mut now = Now::blank(0, BodySense::default());
        now.objects.observations.push(pete_now::ObjectObservation {
            label: "apple".to_string(),
            class: ObjectClass::Unknown,
            bearing_rad: 0.5,
            distance_m: Some(2.0),
            confidence: 1.0,
            source: pete_now::ObjectObservationSource::Sim,
        });
        (directory, runtime, now)
    }

    fn advance(
        runtime: &mut LuaSkillRuntime,
        driver: &mut FakeDriver,
        now: &mut Now,
        ticks: usize,
    ) -> SkillStatus {
        let mut status = SkillStatus::default();
        for _ in 0..ticks {
            status = runtime
                .step(now, &empty_events(), driver)
                .expect("foreground invocation");
            if status.phase == SkillPhase::Terminal {
                break;
            }
            now.t_ms += 100;
            now.body.last_update_ms = now.t_ms;
        }
        status
    }

    fn completed_result(runtime: &LuaSkillRuntime) -> Option<Value> {
        runtime.trace().iter().rev().find_map(|event| match event {
            SkillTraceEvent::Completed { result, .. } => result.clone(),
            _ => None,
        })
    }

    #[test]
    fn valid_skills_load_with_hash_path_and_runtime_version() {
        let (_directory, runtime, _now) = runtime_with(
            "stopAndStabilize",
            "function stopAndStabilize(args) return 'ok' end",
        );
        let skills = runtime.discoverable_skills();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].skill_id, "test.stopAndStabilize");
        assert_eq!(skills[0].source_hash.len(), 64);
        assert!(skills[0].source_path.ends_with("stopAndStabilize.lua"));
        assert!(skills[0].runtime_version.contains("Lua 5.4"));
    }

    #[test]
    fn newly_discovered_skill_can_run_by_inferred_runtime_id() {
        let (_directory, mut runtime, mut now) = runtime_with(
            "waveHello",
            "function waveHello(args) return args.implementation_id end",
        );
        let request = SkillRequest::runtime_loaded("test.waveHello");
        runtime.start(request, &now).unwrap();
        let status = advance(&mut runtime, &mut FakeDriver::default(), &mut now, 2);
        assert_eq!(status.outcome, Some(SkillOutcome::Completed));
        assert_eq!(completed_result(&runtime), Some(json!("test.waveHello")));
    }

    #[test]
    fn invalid_reload_leaves_prior_generation_active() {
        let (directory, mut runtime, _now) = runtime_with(
            "stopAndStabilize",
            "function stopAndStabilize(args) return 'valid' end",
        );
        let generation = runtime.generation_hash().to_string();
        write_skill(
            directory.path(),
            "stopAndStabilize",
            "function stopAndStabilize(",
        );
        assert!(runtime.reload().is_err());
        assert_eq!(runtime.generation_hash(), generation);
        assert!(runtime.last_reload_error().is_some());
    }

    #[test]
    fn sandbox_removes_host_access_and_raw_cockpit() {
        let (_directory, mut runtime, mut now) = runtime_with(
            "stopAndStabilize",
            r#"
                function stopAndStabilize(args)
                    assert(io == nil and os == nil and debug == nil)
                    assert(package == nil and dofile == nil and loadfile == nil and load == nil)
                    assert(coroutine == nil and rawget == nil and rawset == nil)
                    assert(Cockpit == nil and cockpit == nil and socket == nil)
                    assert(math.random == nil and math.randomseed == nil)
                    return "sandboxed"
                end
            "#,
        );
        runtime
            .start(request(SkillId::StopAndStabilize), &now)
            .unwrap();
        let status = advance(&mut runtime, &mut FakeDriver::default(), &mut now, 2);
        assert_eq!(status.outcome, Some(SkillOutcome::Completed));
        assert_eq!(completed_result(&runtime), Some(json!("sandboxed")));
    }

    #[test]
    fn active_invocation_is_pinned_across_atomic_hot_reload() {
        let (directory, mut runtime, mut now) = runtime_with(
            "stopAndStabilize",
            "function stopAndStabilize(args) drive(0.05, 400); return 'old' end",
        );
        runtime
            .start(request(SkillId::StopAndStabilize), &now)
            .unwrap();
        let old_hash = runtime.diagnostics().source_hash.unwrap();
        let _ = advance(&mut runtime, &mut FakeDriver::default(), &mut now, 1);
        write_skill(
            directory.path(),
            "stopAndStabilize",
            "function stopAndStabilize(args) return 'new' end",
        );
        assert!(runtime.reload().unwrap());
        assert_eq!(
            runtime.diagnostics().source_hash.as_deref(),
            Some(old_hash.as_str())
        );
        let mut driver = FakeDriver::default();
        let status = advance(&mut runtime, &mut driver, &mut now, 10);
        assert_eq!(status.outcome, Some(SkillOutcome::Completed));
        assert_eq!(completed_result(&runtime), Some(json!("old")));
        runtime.take_terminal();

        now.t_ms += 100;
        runtime
            .start(request(SkillId::StopAndStabilize), &now)
            .unwrap();
        let status = advance(&mut runtime, &mut driver, &mut now, 2);
        assert_eq!(status.outcome, Some(SkillOutcome::Completed));
        assert_eq!(completed_result(&runtime), Some(json!("new")));
        assert_ne!(
            runtime.diagnostics().source_hash.as_deref(),
            Some(old_hash.as_str())
        );
    }

    #[test]
    fn infinite_loop_exhausts_budget_without_dispatching() {
        let (_directory, mut runtime, mut now) = runtime_with(
            "stopAndStabilize",
            "function stopAndStabilize(args) while true do end end",
        );
        runtime
            .start(request(SkillId::StopAndStabilize), &now)
            .unwrap();
        let mut driver = FakeDriver::default();
        let status = advance(&mut runtime, &mut driver, &mut now, 2);
        assert_eq!(status.outcome, Some(SkillOutcome::BudgetExceeded));
        assert!(driver.records.is_empty());
        assert!(runtime.diagnostics().held_resources.is_empty());
    }

    #[test]
    fn child_budget_exhaustion_stops_active_sibling_command() {
        let (_directory, mut runtime, mut now) = runtime_with(
            "approachTarget",
            r#"
                function approachTarget(args)
                    together(
                        function() drive(0.05, 1000) end,
                        function()
                            scan()
                            while true do end
                        end
                    )
                end
            "#,
        );
        runtime
            .start(request(SkillId::ApproachTarget), &now)
            .unwrap();
        let mut driver = FakeDriver::default();
        driver.duration_ms.insert("drive".to_string(), 1_000);
        driver.duration_ms.insert("scan".to_string(), 100);
        let status = advance(&mut runtime, &mut driver, &mut now, 8);
        assert_eq!(status.outcome, Some(SkillOutcome::BudgetExceeded));
        assert!(driver
            .stops
            .iter()
            .any(|(resource, _)| *resource == BodyResource::Locomotion));
        assert!(runtime.diagnostics().held_resources.is_empty());
    }

    #[test]
    fn excessive_recursion_fails_safely() {
        let (_directory, mut runtime, mut now) = runtime_with(
            "stopAndStabilize",
            r#"
                local function recurse(n)
                    return n + recurse(n + 1)
                end
                function stopAndStabilize(args)
                    return recurse(0)
                end
            "#,
        );
        runtime
            .start(request(SkillId::StopAndStabilize), &now)
            .unwrap();
        let status = advance(&mut runtime, &mut FakeDriver::default(), &mut now, 2);
        assert!(matches!(
            status.outcome,
            Some(SkillOutcome::BudgetExceeded | SkillOutcome::ScriptError)
        ));
        assert!(runtime.diagnostics().held_resources.is_empty());
    }

    #[test]
    fn excessive_memory_fails_safely() {
        let (_directory, mut runtime, mut now) = runtime_with(
            "stopAndStabilize",
            r#"
                function stopAndStabilize(args)
                    return string.rep("x", 3 * 1024 * 1024)
                end
            "#,
        );
        runtime
            .start(request(SkillId::StopAndStabilize), &now)
            .unwrap();
        let status = advance(&mut runtime, &mut FakeDriver::default(), &mut now, 2);
        assert_eq!(status.outcome, Some(SkillOutcome::BudgetExceeded));
        assert!(runtime.diagnostics().held_resources.is_empty());
    }

    #[test]
    fn plain_nested_functions_suspend_and_resume_with_return_values() {
        let (_directory, mut runtime, mut now) = runtime_with(
            "stopAndStabilize",
            r#"
                local function inner()
                    drive(0.05, 200)
                    return 41
                end
                function stopAndStabilize(args)
                    return inner() + 1
                end
            "#,
        );
        runtime
            .start(request(SkillId::StopAndStabilize), &now)
            .unwrap();
        let mut driver = FakeDriver::default();
        let first = advance(&mut runtime, &mut driver, &mut now, 1);
        assert_eq!(first.phase, SkillPhase::Running);
        assert!(driver.records.is_empty());
        let status = advance(&mut runtime, &mut driver, &mut now, 8);
        assert_eq!(status.outcome, Some(SkillOutcome::Completed));
        assert_eq!(completed_result(&runtime), Some(json!(42)));
        assert_eq!(driver.records.len(), 1);
        assert!(runtime.diagnostics().held_resources.is_empty());
    }

    #[test]
    fn typed_failure_can_be_handled_with_try() {
        let (_directory, mut runtime, mut now) = runtime_with(
            "approachTarget",
            r#"
                function approachTarget(args)
                    local ok, result = try(function()
                        approach(args.target, 0.2)
                    end)
                    if not ok and result.kind == "contact_withdrawal" then
                        return "handled"
                    end
                    if not ok then error(result) end
                    return "unexpected"
                end
            "#,
        );
        runtime
            .start(request(SkillId::ApproachTarget), &now)
            .unwrap();
        let mut driver = FakeDriver::default();
        driver.fail_operations.insert(
            "approach".to_string(),
            SkillFailure::new(
                SkillOutcome::SafetyPreempted,
                "contact_withdrawal",
                "contact reflex",
            ),
        );
        let status = advance(&mut runtime, &mut driver, &mut now, 6);
        assert_eq!(status.outcome, Some(SkillOutcome::Completed));
        assert_eq!(completed_result(&runtime), Some(json!("handled")));
        assert!(runtime.diagnostics().held_resources.is_empty());
    }

    #[test]
    fn typed_failure_rethrown_from_lua_preserves_original_outcome() {
        let (_directory, mut runtime, mut now) = runtime_with(
            "stopAndStabilize",
            r#"
                function stopAndStabilize(args)
                    local ok, result = try(function() scan() end)
                    if not ok then error(result) end
                end
            "#,
        );
        runtime
            .start(request(SkillId::StopAndStabilize), &now)
            .unwrap();
        let mut driver = FakeDriver::default();
        driver.fail_operations.insert(
            "scan".to_string(),
            SkillFailure::new(
                SkillOutcome::CapabilityUnavailable,
                "capability_unavailable",
                "gaze is absent",
            ),
        );
        let status = advance(&mut runtime, &mut driver, &mut now, 8);
        assert_eq!(status.outcome, Some(SkillOutcome::CapabilityUnavailable));
        assert_eq!(status.reason.as_deref(), Some("gaze is absent"));
    }

    #[test]
    fn organ_driver_panic_stops_and_releases_owned_resource() {
        let (_directory, mut runtime, mut now) = runtime_with(
            "stopAndStabilize",
            "function stopAndStabilize(args) drive(0.05, 200) end",
        );
        runtime
            .start(request(SkillId::StopAndStabilize), &now)
            .unwrap();
        let mut driver = FakeDriver::default();
        driver.panic_operations.insert("drive".to_string());
        let status = advance(&mut runtime, &mut driver, &mut now, 4);
        assert_eq!(status.outcome, Some(SkillOutcome::ScriptError));
        assert_eq!(driver.stops.len(), 1);
        assert!(runtime.diagnostics().held_resources.is_empty());
    }

    #[test]
    fn foreground_is_exclusive_and_explicit_cancellation_unwinds_resources() {
        let (_directory, mut runtime, mut now) = runtime_with(
            "stopAndStabilize",
            "function stopAndStabilize(args) drive(0.05, 1000) end",
        );
        runtime
            .start(request(SkillId::StopAndStabilize), &now)
            .unwrap();
        assert!(runtime
            .start(request(SkillId::StopAndStabilize), &now)
            .is_err());
        let mut driver = FakeDriver::default();
        let _ = advance(&mut runtime, &mut driver, &mut now, 2);
        let status = runtime
            .cancel(
                &mut driver,
                SkillOutcome::Cancelled,
                "operator_preempted",
                "operator selected a stronger intention",
                now.t_ms,
            )
            .unwrap();
        assert_eq!(status.outcome, Some(SkillOutcome::Cancelled));
        assert_eq!(driver.stops.len(), 1);
        assert!(runtime.diagnostics().held_resources.is_empty());
    }

    #[test]
    fn together_overlaps_disjoint_organs_and_preserves_result_order() {
        let (_directory, mut runtime, mut now) = runtime_with(
            "approachTarget",
            r#"
                function approachTarget(args)
                    return together(
                        function() drive(0.05, 200); return "move" end,
                        function() scan(); return "look" end,
                        function() say("hello"); return "speak" end
                    )
                end
            "#,
        );
        runtime
            .start(request(SkillId::ApproachTarget), &now)
            .unwrap();
        let mut driver = FakeDriver::default();
        let status = advance(&mut runtime, &mut driver, &mut now, 10);
        assert_eq!(status.outcome, Some(SkillOutcome::Completed));
        assert_eq!(driver.records.len(), 3);
        let starts: BTreeSet<_> = driver
            .records
            .iter()
            .map(|record| record.started_at_ms)
            .collect();
        assert_eq!(starts.len(), 1, "disjoint organs must overlap");
        assert_eq!(
            driver
                .records
                .iter()
                .map(|record| record.resource)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                Some(BodyResource::Locomotion),
                Some(BodyResource::Gaze),
                Some(BodyResource::Voice),
            ])
        );
        assert_eq!(
            completed_result(&runtime),
            Some(json!([["move"], ["look"], ["speak"]]))
        );
    }

    #[test]
    fn together_serializes_same_resource_in_child_order_without_busy_polling() {
        let (_directory, mut runtime, mut now) = runtime_with(
            "approachTarget",
            r#"
                function approachTarget(args)
                    return together(
                        function() drive(0.04, 200); return 1 end,
                        function() turnBy(0.5); return 2 end
                    )
                end
            "#,
        );
        runtime
            .start(request(SkillId::ApproachTarget), &now)
            .unwrap();
        let mut driver = FakeDriver::default();
        let _ = advance(&mut runtime, &mut driver, &mut now, 2);
        let diagnostics = runtime.diagnostics();
        assert_eq!(
            diagnostics.held_resources.get(&BodyResource::Locomotion),
            Some(&1)
        );
        assert_eq!(
            diagnostics.waiting_resources.get(&BodyResource::Locomotion),
            Some(&vec![2])
        );
        let status = advance(&mut runtime, &mut driver, &mut now, 10);
        assert_eq!(status.outcome, Some(SkillOutcome::Completed));
        assert_eq!(driver.records[0].child_id, 1);
        assert_eq!(driver.records[1].child_id, 2);
        assert!(
            driver.records[1].started_at_ms
                >= driver.records[0]
                    .ended_at_ms
                    .expect("first operation ended")
        );
    }

    #[test]
    fn together_is_fail_fast_and_stops_siblings() {
        let (_directory, mut runtime, mut now) = runtime_with(
            "approachTarget",
            r#"
                function approachTarget(args)
                    together(
                        function() drive(0.04, 1000) end,
                        function() scan() end,
                        function() say("still speaking") end
                    )
                end
            "#,
        );
        runtime
            .start(request(SkillId::ApproachTarget), &now)
            .unwrap();
        let mut driver = FakeDriver::default();
        driver.fail_operations.insert(
            "scan".to_string(),
            SkillFailure::new(
                SkillOutcome::CapabilityUnavailable,
                "capability_unavailable",
                "gaze is absent",
            ),
        );
        let status = advance(&mut runtime, &mut driver, &mut now, 8);
        assert_eq!(status.outcome, Some(SkillOutcome::CapabilityUnavailable));
        assert!(driver
            .stops
            .iter()
            .any(|(resource, _)| *resource == BodyResource::Locomotion));
        assert!(driver
            .stops
            .iter()
            .any(|(resource, _)| *resource == BodyResource::Voice));
        assert!(runtime.diagnostics().held_resources.is_empty());
    }

    #[test]
    fn parent_cancellation_and_reflex_preemption_cancel_together_children() {
        for preempt in [false, true] {
            let (_directory, mut runtime, mut now) = runtime_with(
                "approachTarget",
                r#"
                    function approachTarget(args)
                        together(
                            function() drive(0.04, 1000) end,
                            function() say("working") end
                        )
                    end
                "#,
            );
            runtime
                .start(request(SkillId::ApproachTarget), &now)
                .unwrap();
            let mut driver = FakeDriver::default();
            let _ = advance(&mut runtime, &mut driver, &mut now, 2);
            let status = if preempt {
                runtime
                    .step(
                        &now,
                        &event(CockpitEventKind::ContactWithdrawalStarted),
                        &mut driver,
                    )
                    .unwrap()
            } else {
                runtime
                    .cancel(
                        &mut driver,
                        SkillOutcome::Cancelled,
                        "parent_cancelled",
                        "parent cancelled",
                        now.t_ms,
                    )
                    .unwrap()
            };
            assert_eq!(
                status.outcome,
                Some(if preempt {
                    SkillOutcome::SafetyPreempted
                } else {
                    SkillOutcome::Cancelled
                })
            );
            assert_eq!(driver.stops.len(), 2);
            assert!(runtime.diagnostics().held_resources.is_empty());
        }
    }

    #[test]
    fn authority_and_transport_loss_cancel_active_skill_and_careful_escape() {
        for (kind, expected) in [
            (
                CockpitEventKind::AuthorityChanged,
                SkillOutcome::AuthorityLost,
            ),
            (
                CockpitEventKind::HeartbeatExpired,
                SkillOutcome::TransportLost,
            ),
        ] {
            let (_directory, mut runtime, mut now) = runtime_with(
                "releasePersistentBumper",
                r#"
                    function releasePersistentBumper(args)
                        carefully("bumper_front", function() retreat(100) end)
                    end
                "#,
            );
            now.body.flags.bump_left = true;
            runtime
                .start(request(SkillId::ReleasePersistentBumper), &now)
                .unwrap();
            let mut driver = FakeDriver::default();
            let _ = advance(&mut runtime, &mut driver, &mut now, 2);
            let status = runtime.step(&now, &event(kind), &mut driver).unwrap();
            assert_eq!(status.outcome, Some(expected));
            assert!(driver
                .stops
                .iter()
                .any(|(resource, _)| *resource == BodyResource::Locomotion));
            assert!(runtime.diagnostics().held_resources.is_empty());
        }
    }

    #[test]
    fn nested_together_is_deterministic_and_safe() {
        let (_directory, mut runtime, mut now) = runtime_with(
            "approachTarget",
            r#"
                function approachTarget(args)
                    return together(
                        function()
                            return together(
                                function() drive(0.04, 200); return "a" end,
                                function() say("nested"); return "b" end
                            )
                        end,
                        function() scan(); return "c" end
                    )
                end
            "#,
        );
        runtime
            .start(request(SkillId::ApproachTarget), &now)
            .unwrap();
        let status = advance(&mut runtime, &mut FakeDriver::default(), &mut now, 12);
        assert_eq!(status.outcome, Some(SkillOutcome::Completed));
        assert_eq!(
            completed_result(&runtime),
            Some(json!([[[["a"], ["b"]]], ["c"]]))
        );
    }

    #[test]
    fn careful_allows_only_acknowledged_hazard_retreat() {
        let (_directory, mut runtime, mut now) = runtime_with(
            "stopAndStabilize",
            r#"
                function stopAndStabilize(args)
                    carefully("bumper_front", function()
                        retreat(100)
                    end)
                end
            "#,
        );
        now.body.flags.bump_left = true;
        runtime
            .start(request(SkillId::StopAndStabilize), &now)
            .unwrap();
        let mut driver = FakeDriver::default();
        let status = advance(&mut runtime, &mut driver, &mut now, 8);
        assert_eq!(status.outcome, Some(SkillOutcome::Completed));
        assert!(matches!(driver.records[0].name.as_str(), "retreat"));
    }

    #[test]
    fn careful_cannot_expand_envelope_or_suppress_absolute_hazard() {
        let source = r#"
            function stopAndStabilize(args)
                carefully("bumper_front", function()
                    drive(0.12, 1000)
                end)
            end
        "#;
        let (_directory, mut runtime, mut now) = runtime_with("stopAndStabilize", source);
        now.body.flags.bump_left = true;
        runtime
            .start(request(SkillId::StopAndStabilize), &now)
            .unwrap();
        let status = advance(&mut runtime, &mut FakeDriver::default(), &mut now, 5);
        assert_eq!(status.outcome, Some(SkillOutcome::Failed));
        assert!(status.reason.unwrap().contains("bounded retreat"));

        let (_directory, mut runtime, mut now) = runtime_with(
            "stopAndStabilize",
            r#"
                function stopAndStabilize(args)
                    carefully("bumper_front", function() retreat(100) end)
                end
            "#,
        );
        now.body.flags.bump_left = true;
        now.body.flags.wheel_drop = true;
        runtime
            .start(request(SkillId::StopAndStabilize), &now)
            .unwrap();
        let status = advance(&mut runtime, &mut FakeDriver::default(), &mut now, 5);
        assert_eq!(status.outcome, Some(SkillOutcome::SafetyPreempted));
        assert!(status.reason.unwrap().contains("absolute"));
    }

    #[test]
    fn progress_is_explicit_bounded_and_retains_originating_goal() {
        let (_directory, mut runtime, mut now) = runtime_with(
            "stopAndStabilize",
            r#"
                function stopAndStabilize(args)
                    reportProgress("goal_progress", 0.65)
                    drive(0.04, 200)
                    return true
                end
            "#,
        );
        let request = request(SkillId::StopAndStabilize);
        runtime.start(request.clone(), &now).unwrap();
        let mut driver = FakeDriver::default();
        let running = advance(&mut runtime, &mut driver, &mut now, 2);
        assert_eq!(running.request.goal_id, request.goal_id);
        assert_eq!(running.progress, Some(0.65));
        let terminal = advance(&mut runtime, &mut driver, &mut now, 6);
        assert_eq!(terminal.progress, Some(1.0));
        assert_eq!(
            terminal.script,
            Some(SkillScriptStatus {
                skill_id: "test.stopAndStabilize".to_string(),
                source_hash: runtime.discoverable_skills()[0].source_hash.clone(),
                source_path: runtime.discoverable_skills()[0]
                    .source_path
                    .display()
                    .to_string(),
                current_function: Some("stopAndStabilize".to_string()),
                current_operation: None,
                held_resources: Vec::new(),
                waiting_resources: Vec::new(),
                active_children: 0,
            })
        );
    }

    #[test]
    fn greet_uses_canonical_person_identity_and_records_encounter_acknowledgment() {
        let source = r#"
            function greet(args)
                local person = args.target
                require(person ~= nil, "person required")
                together(
                    function() face(person) end,
                    function() say("Hello " .. person.name .. ".") end
                )
                acknowledge(person)
                reportProgress("social_acknowledgment", 1.0)
                return {
                    person_id = person.id,
                    name = person.name,
                    acknowledged = true,
                }
            end
        "#;
        let (_directory, mut runtime, _) = runtime_with("greet", source);
        let mut raw = Now::blank(0, BodySense::default());
        raw.objects.observations.push(pete_now::ObjectObservation {
            label: "Alex".to_string(),
            class: ObjectClass::Person,
            bearing_rad: 0.25,
            distance_m: Some(0.8),
            confidence: 0.95,
            source: pete_now::ObjectObservationSource::Kinect,
        });
        let mut now = pete_now::WorldModelUpdater::default()
            .update(raw, pete_now::WorldModelUpdateContext::default());
        let encounter_id = now
            .world
            .social
            .active_interaction
            .as_ref()
            .unwrap()
            .interaction_id
            .0
            .clone();
        let mut request = SkillRequest::runtime_loaded("test.greet");
        request.goal_id = Some(GoalId::new("greet_person"));
        request.behavior_id = Some(format!("greet:person:alex:{encounter_id}"));
        request.target = Some(EntityId("person:alex".to_string()));
        request.bearing_rad = Some(0.25);
        request.progress_metric = "social_acknowledgment".to_string();
        runtime.start(request, &now).unwrap();

        let mut driver = FakeDriver::default();
        let status = advance(&mut runtime, &mut driver, &mut now, 8);
        assert_eq!(status.outcome, Some(SkillOutcome::Completed));
        assert_eq!(status.progress, Some(1.0));
        assert_eq!(
            completed_result(&runtime),
            Some(json!({
                "person_id": "person:alex",
                "name": "Alex",
                "acknowledged": true,
            }))
        );
        assert_eq!(
            driver
                .records
                .iter()
                .map(|record| record.name.as_str())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["face_bearing", "say"])
        );
        let record = runtime.execution_record().unwrap();
        assert_eq!(record.execution_id, status.execution_id);
        assert_eq!(
            record.observations,
            vec![json!({
                "kind": "social_acknowledgment",
                "contract": "host_validated_social_acknowledgment_v1",
                "value": {
                    "acknowledgment_id": format!(
                        "greet:{encounter_id}:person:alex:{}",
                        status.execution_id
                    ),
                    "interaction_id": encounter_id,
                    "person_id": "person:alex",
                    "occurred_at_ms": 200,
                },
                "provenance": "lua_skill",
            })]
        );
    }
}
