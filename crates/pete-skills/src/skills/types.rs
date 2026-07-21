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
