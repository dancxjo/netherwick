pub struct RuntimeTick {
    pub frame: ExperienceFrame,
    pub experience: Experience,
    pub chosen_action: Option<ActionPrimitive>,
    pub skill_request: Option<SkillRequest>,
    pub skill_status: Option<SkillStatus>,
    pub recall: RecallBundle,
    pub llm: LlmTickResult,
    pub combobulation: Option<Combobulation>,
    pub inline_learning: InlineLearningTickStatus,
    /// Canonical observability records created at production execution
    /// boundaries. Consumers may attach transport-local snapshot references,
    /// but must not reconstruct causal events from this tick's aggregate state.
    pub brain_events: Vec<BrainEvent>,
}

#[async_trait::async_trait]
pub trait RuntimeLoop {
    async fn tick(
        &mut self,
        now: Now,
        latent: ExperienceLatent,
        futures: Vec<FuturePrediction>,
    ) -> Result<RuntimeTick>;

    fn reign_sense(&self, _now_ms: TimeMs) -> Result<ReignSense> {
        Ok(ReignSense::default())
    }

    fn observe_skill_status(&mut self, _status: &SkillStatus) {}

    fn canonical_map(&self) -> Option<LocalMap> {
        None
    }
}

#[async_trait::async_trait]
impl<L, M, R, C, S, A> RuntimeLoop for MinimalRuntime<L, M, R, C, S, A>
where
    L: LedgerWriter + Sync + Send,
    M: MemoryStore + Send,
    R: Recall + Send + Sync,
    C: Conductor + Send,
    S: SafetyLayer + Send,
    A: LlmAgent + Send + 'static,
{
    fn reign_sense(&self, now_ms: TimeMs) -> Result<ReignSense> {
        let mut reign_queue = self
            .reign_queue
            .lock()
            .map_err(|_| anyhow::anyhow!("reign queue lock poisoned"))?;
        reign_queue.drain_expired(now_ms);
        Ok(reign_queue.sense(now_ms))
    }

    fn observe_skill_status(&mut self, status: &SkillStatus) {
        self.goal_system.observe_skill_status(status);
    }

    fn canonical_map(&self) -> Option<LocalMap> {
        Some(self.local_map.clone())
    }

    async fn tick(
        &mut self,
        now: Now,
        latent: ExperienceLatent,
        futures: Vec<FuturePrediction>,
    ) -> Result<RuntimeTick> {
        MinimalRuntime::tick(self, now, latent, futures).await
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RobotMode {
    ReadOnly,
    Slow,
    Disabled,
}

pub struct RealRobotRunner<R, C> {
    pub mode: RobotMode,
    pub cockpit: SafeCockpit<C>,
    pub sensors: Vec<Box<dyn SenseProducer + Send>>,
    pub runtime: R,
    pub tick_ms: u64,
    pub tick_count: usize,
    /// Allow executive-selected motion to reach real slow hardware. Direct
    /// WebRemote/Gamepad commands remain available regardless of this gate.
    pub autonomous_motion: bool,
    now_builder: NowBuilder,
    frame_processor: FrameProcessor,
    live_image_cognition: LiveImageCognition,
    robot_initialization: Option<serde_json::Value>,
    brainstem_interface: Option<serde_json::Value>,
    possession_recovery: PossessionRecoveryState,
    motion_rejection: MotionRejectionState,
    possessor_skills: PossessorSkillRuntime,
    sensor_poll_health: Vec<SensorPollHealth>,
    physical_pose: PhysicalPoseAdapter,
    brainstem_clock: BrainstemClockMapper,
    imu_arbiter: ImuArbiter,
    last_imu_selection: ImuSelection,
    last_ups_trend_sample: Option<UpsTelemetry>,
}

#[derive(Clone, Debug, Default)]
pub struct PhysicalPoseAdapter {
    pose: Pose2,
    last_distance_mm: Option<i32>,
    last_heading_mrad: Option<i32>,
    last_reset_count: Option<u32>,
}

#[derive(Clone, Debug, Default)]
struct SensorPollHealth {
    name: String,
    available: bool,
    consecutive_failures: u32,
    last_error: Option<String>,
    last_report_ms: TimeMs,
    last_success_ms: TimeMs,
    producer: Option<serde_json::Value>,
}

const SENSOR_FAILURE_REPORT_INTERVAL_MS: TimeMs = 30_000;
const VERIFY_CHARGING_PACKET_MAX_AGE_MS: u32 = 500;

struct PossessorSkillRuntime {
    lua: Option<LuaSkillRuntime>,
    load_error: Option<String>,
    driver: EmbodiedLuaDriverState,
    status: Option<SkillStatus>,
    provenance: Option<serde_json::Value>,
    last_reload_check_ms: TimeMs,
}

impl Default for PossessorSkillRuntime {
    fn default() -> Self {
        let mut config = LuaSkillConfig::default();
        config.directory = std::env::var_os("PETE_MOTHERBRAIN_SKILL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../skills/motherbrain")
            });
        match LuaSkillRuntime::load(config) {
            Ok(lua) => Self {
                lua: Some(lua),
                load_error: None,
                driver: EmbodiedLuaDriverState::default(),
                status: None,
                provenance: None,
                last_reload_check_ms: 0,
            },
            Err(error) => Self {
                lua: None,
                load_error: Some(error.to_string()),
                driver: EmbodiedLuaDriverState::default(),
                status: None,
                provenance: None,
                last_reload_check_ms: 0,
            },
        }
    }
}

#[derive(Clone, Debug, Default)]
struct LuaDriverOperationState {
    start_x_m: f32,
    start_y_m: f32,
    start_heading_rad: f32,
    last_dispatch_ms: TimeMs,
}

#[derive(Clone, Debug, Default)]
struct EmbodiedLuaDriverState {
    operations: std::collections::HashMap<u64, LuaDriverOperationState>,
}

struct RealLuaOrganDriver<'a, C> {
    cockpit: &'a mut C,
    request: &'a SkillRequest,
    status: &'a StatusSummary,
    home_base_contact: bool,
    state: &'a mut EmbodiedLuaDriverState,
    command_sent: bool,
}

impl<C: Cockpit> OrganDriver for RealLuaOrganDriver<'_, C> {
    fn poll(
        &mut self,
        operation: &HostOperation,
        context: OperationContext,
        now: &Now,
        _events: &pete_cockpit::EventBatch,
    ) -> OrganPoll {
        let state = self
            .state
            .operations
            .entry(context.operation_id)
            .or_insert_with(|| LuaDriverOperationState {
                start_x_m: now.body.odometry.x_m,
                start_y_m: now.body.odometry.y_m,
                start_heading_rad: now.body.odometry.heading_rad,
                last_dispatch_ms: 0,
            });
        let dispatch_due = state.last_dispatch_ms == 0
            || context.now_ms.saturating_sub(state.last_dispatch_ms) >= 100;
        let mut primitive = None;
        let outcome = match operation {
            HostOperation::Stop => {
                if context.first_poll {
                    if let Err(error) = self.cockpit.stop() {
                        return OrganPoll::Failed(cockpit_skill_failure(operation, error));
                    }
                    self.command_sent = true;
                }
                OrganPoll::Completed(json!({"stopped": true}))
            }
            HostOperation::FaceBearing { bearing_rad } => {
                let requested_bearing = self.request.bearing_rad.unwrap_or(*bearing_rad);
                let turned = angle_delta(now.body.odometry.heading_rad, state.start_heading_rad);
                let remaining = if self.request.target.is_some() {
                    requested_bearing
                } else {
                    angle_delta(requested_bearing, turned)
                };
                if remaining.abs() <= 0.10 {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"bearing_error": remaining, "turned_rad": turned}),
                        &mut self.command_sent,
                    )
                } else if dispatch_due {
                    let available_s = self
                        .request
                        .maximum_duration_ms
                        .saturating_sub(100)
                        .max(100) as f32
                        / 1_000.0;
                    let angular_rad_s = (remaining.abs() / available_s * 1.25).clamp(0.25, 2.5)
                        * remaining.signum();
                    let angular = radians_to_mrad(angular_rad_s);
                    match self.cockpit.cmd_vel(0, angular, context.primitive_ttl_ms) {
                        Ok(()) => {
                            self.command_sent = true;
                            primitive = Some(primitive_intent(
                                context,
                                operation,
                                json!({"linear_mm_s": 0, "angular_mrad_s": angular, "ttl_ms": context.primitive_ttl_ms, "remaining_rad": remaining}),
                            ));
                            OrganPoll::Pending {
                                progress: Some(("bearing_error".into(), remaining.abs())),
                                primitive: None,
                            }
                        }
                        Err(error) => OrganPoll::Failed(cockpit_skill_failure(operation, error)),
                    }
                } else {
                    OrganPoll::Pending {
                        progress: Some(("bearing_error".into(), remaining.abs())),
                        primitive: None,
                    }
                }
            }
            HostOperation::FollowBearing {
                bearing_rad,
                linear_m_s,
            } => {
                let bearing = self.request.bearing_rad.unwrap_or(*bearing_rad);
                if bearing.abs() <= 0.10 {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"bearing_error": bearing}),
                        &mut self.command_sent,
                    )
                } else if dispatch_due {
                    with_operation_progress(
                        dispatch_velocity(
                            self.cockpit,
                            operation,
                            context,
                            (*linear_m_s * 1_000.0).round() as i16,
                            radians_to_mrad(bearing).clamp(-500, 500),
                            &mut self.command_sent,
                            &mut primitive,
                        ),
                        Some(("bearing_error".into(), bearing.abs())),
                    )
                } else {
                    OrganPoll::Pending {
                        progress: Some(("bearing_error".into(), bearing.abs())),
                        primitive: None,
                    }
                }
            }
            HostOperation::HoldHeading {
                heading_rad,
                linear_m_s,
            } => {
                let error = self.request.bearing_rad.unwrap_or(*heading_rad);
                if error.abs() <= 0.10 {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"heading_error": error}),
                        &mut self.command_sent,
                    )
                } else if dispatch_due {
                    with_operation_progress(
                        dispatch_velocity(
                            self.cockpit,
                            operation,
                            context,
                            (*linear_m_s * 1_000.0).round() as i16,
                            radians_to_mrad(error).clamp(-400, 400),
                            &mut self.command_sent,
                            &mut primitive,
                        ),
                        Some(("bearing_error".into(), error.abs())),
                    )
                } else {
                    OrganPoll::Pending {
                        progress: Some(("heading_error".into(), error.abs())),
                        primitive: None,
                    }
                }
            }
            HostOperation::Approach { stop_range_m, .. } => {
                let Some(range) = self.request.range_m else {
                    return OrganPoll::Failed(
                        SkillFailure::new(
                            SkillOutcome::Failed,
                            "target_stale",
                            "approach target has no current range",
                        )
                        .for_operation(operation),
                    );
                };
                let Some(bearing) = self.request.bearing_rad else {
                    return OrganPoll::Failed(
                        SkillFailure::new(
                            SkillOutcome::Failed,
                            "target_stale",
                            "approach target has no current bearing",
                        )
                        .for_operation(operation),
                    );
                };
                if range <= *stop_range_m {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"range_m": range, "stop_range_m": stop_range_m}),
                        &mut self.command_sent,
                    )
                } else if dispatch_due {
                    with_operation_progress(
                        dispatch_velocity(
                            self.cockpit,
                            operation,
                            context,
                            50,
                            radians_to_mrad(bearing).clamp(-500, 500),
                            &mut self.command_sent,
                            &mut primitive,
                        ),
                        Some(("target_distance".into(), range)),
                    )
                } else {
                    OrganPoll::Pending {
                        progress: Some(("target_distance".into(), range)),
                        primitive: None,
                    }
                }
            }
            HostOperation::AlignWithDock => {
                if now.body.charging || self.home_base_contact {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({
                            "aligned": true,
                            "home_base_contact": self.home_base_contact,
                            "charging_observed": now.body.charging,
                        }),
                        &mut self.command_sent,
                    )
                } else {
                    let Some(cue) = DockIrCue::from_character(now.body.infrared_character) else {
                        return OrganPoll::Failed(
                            SkillFailure::new(
                                SkillOutcome::Failed,
                                "target_stale",
                                "Home Base IR gradient disappeared",
                            )
                            .for_operation(operation),
                        );
                    };
                    if dispatch_due {
                        with_operation_progress(
                            dispatch_velocity(
                                self.cockpit,
                                operation,
                                context,
                                50,
                                cue.steering_mrad_s(400),
                                &mut self.command_sent,
                                &mut primitive,
                            ),
                            self.request
                                .range_m
                                .map(|range| ("target_distance".into(), range)),
                        )
                    } else {
                        OrganPoll::Pending {
                            progress: self
                                .request
                                .range_m
                                .map(|range| ("target_distance".into(), range)),
                            primitive: None,
                        }
                    }
                }
            }
            HostOperation::SearchForDockSignal => {
                if self.home_base_contact
                    || DockIrCue::from_character(now.body.infrared_character).is_some()
                {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"infrared_character": now.body.infrared_character}),
                        &mut self.command_sent,
                    )
                } else if dispatch_due {
                    dispatch_velocity(
                        self.cockpit,
                        operation,
                        context,
                        0,
                        300,
                        &mut self.command_sent,
                        &mut primitive,
                    )
                } else {
                    OrganPoll::Pending {
                        progress: None,
                        primitive: None,
                    }
                }
            }
            HostOperation::VerifyCharging => {
                let charging_state = self.status.battery.charging_state;
                if self
                    .status
                    .has_fresh_complete_body_packet(VERIFY_CHARGING_PACKET_MAX_AGE_MS)
                    && matches!(charging_state, Some(1..=3))
                {
                    OrganPoll::Completed(json!({
                        "charging": true,
                        "source": "create_oi",
                        "charging_state": charging_state,
                        "body_packet_age_ms": self.status.body_packet_age_ms,
                    }))
                } else if context.elapsed_ms >= 1_000 {
                    OrganPoll::Failed(
                        SkillFailure::new(
                            SkillOutcome::PostconditionFailed,
                            "charging_not_verified",
                            "fresh Create OI charging telemetry was not observed",
                        )
                        .for_operation(operation),
                    )
                } else {
                    OrganPoll::Pending {
                        progress: None,
                        primitive: None,
                    }
                }
            }
            HostOperation::Drive {
                linear_m_s,
                duration_ms,
            } => {
                let progress = if self.request.progress_metric == "reverse_displacement" {
                    let expected_distance =
                        linear_m_s.abs() * (*duration_ms as f32 / 1_000.0).max(0.001);
                    (
                        "reverse_displacement".to_string(),
                        distance_from_start(state, &now.body) / expected_distance,
                    )
                } else {
                    (
                        "duration".to_string(),
                        context.elapsed_ms as f32 / (*duration_ms).max(1) as f32,
                    )
                };
                if context.elapsed_ms >= *duration_ms {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"duration_ms": duration_ms}),
                        &mut self.command_sent,
                    )
                } else if dispatch_due {
                    with_operation_progress(
                        dispatch_velocity(
                            self.cockpit,
                            operation,
                            context,
                            (*linear_m_s * 1_000.0).round() as i16,
                            0,
                            &mut self.command_sent,
                            &mut primitive,
                        ),
                        Some(progress),
                    )
                } else {
                    OrganPoll::Pending {
                        progress: Some(progress),
                        primitive: None,
                    }
                }
            }
            HostOperation::DriveDistance {
                distance_m,
                velocity_m_s,
            } => {
                let travelled =
                    distance_from_start(state, &now.body) * velocity_m_s.signum().max(-1.0);
                if travelled.abs() >= distance_m.abs() {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"distance_m": travelled}),
                        &mut self.command_sent,
                    )
                } else if dispatch_due {
                    with_operation_progress(
                        dispatch_velocity(
                            self.cockpit,
                            operation,
                            context,
                            (*velocity_m_s * 1_000.0).round() as i16,
                            0,
                            &mut self.command_sent,
                            &mut primitive,
                        ),
                        Some((
                            "reverse_displacement".into(),
                            travelled.abs() / distance_m.abs().max(0.001),
                        )),
                    )
                } else {
                    OrganPoll::Pending {
                        progress: Some((
                            "reverse_displacement".into(),
                            travelled.abs() / distance_m.abs().max(0.001),
                        )),
                        primitive: None,
                    }
                }
            }
            HostOperation::TurnBy { angle_rad } => {
                let turned = angle_delta(now.body.odometry.heading_rad, state.start_heading_rad);
                if turned.abs() >= angle_rad.abs() {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"turned_rad": turned}),
                        &mut self.command_sent,
                    )
                } else if dispatch_due {
                    with_operation_progress(
                        dispatch_velocity(
                            self.cockpit,
                            operation,
                            context,
                            0,
                            if *angle_rad < 0.0 { -300 } else { 300 },
                            &mut self.command_sent,
                            &mut primitive,
                        ),
                        Some((
                            "frontier_coverage".into(),
                            turned.abs() / angle_rad.abs().max(0.001),
                        )),
                    )
                } else {
                    OrganPoll::Pending {
                        progress: Some((
                            "frontier_coverage".into(),
                            turned.abs() / angle_rad.abs().max(0.001),
                        )),
                        primitive: None,
                    }
                }
            }
            HostOperation::FollowWall { side, .. } => {
                if !now.body.flags.wall {
                    complete_stopped(
                        self.cockpit,
                        operation,
                        json!({"wall_clear": true}),
                        &mut self.command_sent,
                    )
                } else if dispatch_due {
                    with_operation_progress(
                        dispatch_velocity(
                            self.cockpit,
                            operation,
                            context,
                            45,
                            if side == "left" { 120 } else { -120 },
                            &mut self.command_sent,
                            &mut primitive,
                        ),
                        Some((
                            "path_progress".into(),
                            distance_from_start(state, &now.body).clamp(0.0, 1.0),
                        )),
                    )
                } else {
                    OrganPoll::Pending {
                        progress: None,
                        primitive: None,
                    }
                }
            }
            HostOperation::Undock => {
                if context.first_poll {
                    match self.cockpit.cmd_vel(-1, 0, 10) {
                        Ok(()) => {
                            self.command_sent = true;
                            primitive = Some(primitive_intent(
                                context,
                                operation,
                                json!({"linear_mm_s": -1, "angular_mrad_s": 0, "ttl_ms": 10}),
                            ));
                        }
                        Err(error) => {
                            return OrganPoll::Failed(cockpit_skill_failure(operation, error));
                        }
                    }
                }
                if !now.body.charging && context.elapsed_ms >= 1_500 {
                    OrganPoll::Completed(json!({"undocked": true}))
                } else {
                    OrganPoll::Pending {
                        progress: Some((
                            "dock_departure".into(),
                            context.elapsed_ms as f32 / 1_500.0,
                        )),
                        primitive: None,
                    }
                }
            }
            HostOperation::Retreat { hazard, distance_m } => {
                let incompatible_sensor = match hazard {
                    HazardKind::BumperFront => {
                        now.body.flags.cliff_left
                            || now.body.flags.cliff_front_left
                            || now.body.flags.cliff_front_right
                            || now.body.flags.cliff_right
                    }
                    HazardKind::Cliff => now.body.flags.bump_left || now.body.flags.bump_right,
                };
                let imu_absolute = self.status.imu.health.as_deref() == Some("fault")
                    || self
                        .status
                        .imu
                        .tilt_magnitude_mrad
                        .is_some_and(|value| value >= 650)
                    || self
                        .status
                        .imu
                        .impact_score_mm_s2
                        .is_some_and(|value| value >= 18_000);
                if self.status.estop_latched == Some(true)
                    || now.body.flags.wheel_drop
                    || now.body.charging
                    || incompatible_sensor
                    || imu_absolute
                {
                    return OrganPoll::Failed(
                        SkillFailure::new(
                            SkillOutcome::SafetyPreempted,
                            "absolute_hazard",
                            "an absolute or incompatible hazard forbids careful retreat",
                        )
                        .for_operation(operation),
                    );
                }
                if self.status.safety_latch_kind != Some(hazard.latch()) {
                    return OrganPoll::Failed(
                        SkillFailure::new(
                            SkillOutcome::SafetyPreempted,
                            "absolute_or_mismatched_hazard",
                            format!(
                                "Brainstem latch {:?} does not match requested {:?} retreat",
                                self.status.safety_latch_kind, hazard
                            ),
                        )
                        .for_operation(operation),
                    );
                }
                let active = match hazard {
                    HazardKind::BumperFront => {
                        now.body.flags.bump_left || now.body.flags.bump_right
                    }
                    HazardKind::Cliff => {
                        now.body.flags.cliff_left
                            || now.body.flags.cliff_front_left
                            || now.body.flags.cliff_front_right
                            || now.body.flags.cliff_right
                    }
                };
                let travelled = distance_from_start(state, &now.body);
                if !active {
                    if let Err(error) = self.cockpit.stop() {
                        return OrganPoll::Failed(cockpit_skill_failure(operation, error));
                    }
                    self.command_sent = true;
                    OrganPoll::Completed(json!({
                        "hazard": hazard,
                        "distance_m": travelled,
                        "clear": true,
                    }))
                } else if state.last_dispatch_ms == 0
                    || context.now_ms.saturating_sub(state.last_dispatch_ms)
                        >= context.primitive_ttl_ms as u64
                {
                    let Some(generation) = self.status.safety_hazard_generation.filter(|v| *v > 0)
                    else {
                        return OrganPoll::Failed(
                            SkillFailure::new(
                                SkillOutcome::Failed,
                                "hazard_generation_unavailable",
                                "Brainstem did not report an acknowledged hazard generation",
                            )
                            .for_operation(operation),
                        );
                    };
                    match self.cockpit.escape_motion(
                        hazard.latch(),
                        generation,
                        -100,
                        0,
                        context.primitive_ttl_ms,
                    ) {
                        Ok(()) => {
                            self.command_sent = true;
                            primitive = Some(primitive_intent(
                                context,
                                operation,
                                json!({
                                    "hazard": hazard,
                                    "hazard_generation": generation,
                                    "linear_mm_s": -100,
                                    "angular_mrad_s": 0,
                                    "ttl_ms": context.primitive_ttl_ms,
                                }),
                            ));
                            OrganPoll::Pending {
                                progress: Some((
                                    "reverse_displacement".into(),
                                    travelled / distance_m.max(0.001),
                                )),
                                primitive: None,
                            }
                        }
                        Err(error) => OrganPoll::Failed(cockpit_skill_failure(operation, error)),
                    }
                } else {
                    OrganPoll::Pending {
                        progress: Some((
                            "reverse_displacement".into(),
                            travelled / distance_m.max(0.001),
                        )),
                        primitive: None,
                    }
                }
            }
            HostOperation::CompleteHazardRecovery { hazard } => {
                let active = match hazard {
                    HazardKind::BumperFront => {
                        now.body.flags.bump_left || now.body.flags.bump_right
                    }
                    HazardKind::Cliff => {
                        now.body.flags.cliff_left
                            || now.body.flags.cliff_front_left
                            || now.body.flags.cliff_front_right
                            || now.body.flags.cliff_right
                    }
                };
                if self.status.safety_latch_kind != Some(hazard.latch())
                    || active
                    || now.body.flags.wheel_drop
                    || now.body.charging
                {
                    OrganPoll::Failed(
                        SkillFailure::new(
                            SkillOutcome::PostconditionFailed,
                            "hazard_not_clear",
                            "acknowledged hazard may be cleared only after its sensor is clear",
                        )
                        .for_operation(operation),
                    )
                } else if let Err(error) = self.cockpit.stop() {
                    OrganPoll::Failed(cockpit_skill_failure(operation, error))
                } else if let Err(error) = self.cockpit.clear_safety_latch(hazard.latch()) {
                    OrganPoll::Failed(cockpit_skill_failure(operation, error))
                } else {
                    self.command_sent = true;
                    OrganPoll::Completed(json!({
                        "hazard": hazard,
                        "clear": true,
                        "stopped": true,
                    }))
                }
            }
            HostOperation::ReleasePersistentBumper => OrganPoll::Failed(
                SkillFailure::new(
                    SkillOutcome::ScriptError,
                    "invalid_host_sequence",
                    "releasePersistentBumper policy must run through carefully and retreat",
                )
                .for_operation(operation),
            ),
            HostOperation::Observe { target } => OrganPoll::Completed(json!({
                "entity_id": target.id(),
                "observed_at_ms": now.t_ms,
                "provenance": "canonical_now",
            })),
            HostOperation::WaitUntil {
                predicate,
                timeout_ms: _,
            } => match predicate.as_str() {
                "charging" if now.body.charging => OrganPoll::Completed(json!(true)),
                "contact_clear" if !(now.body.flags.bump_left || now.body.flags.bump_right) => {
                    OrganPoll::Completed(json!(true))
                }
                "cliff_clear"
                    if !(now.body.flags.cliff_left
                        || now.body.flags.cliff_front_left
                        || now.body.flags.cliff_front_right
                        || now.body.flags.cliff_right) =>
                {
                    OrganPoll::Completed(json!(true))
                }
                _ => OrganPoll::Pending {
                    progress: None,
                    primitive: None,
                },
            },
            HostOperation::PlayFeedback { pattern } => {
                let kind = match pattern.as_str() {
                    "ok" => pete_cockpit::FeedbackKind::Ok,
                    "error" => pete_cockpit::FeedbackKind::Error,
                    "armed" => pete_cockpit::FeedbackKind::Armed,
                    "lost_target" => pete_cockpit::FeedbackKind::LostTarget,
                    "dock_seen" => pete_cockpit::FeedbackKind::DockSeen,
                    "danger" => pete_cockpit::FeedbackKind::Danger,
                    _ => return OrganPoll::Failed(SkillFailure::capability(operation)),
                };
                match self.cockpit.play_feedback(kind) {
                    Ok(()) => {
                        self.command_sent = true;
                        OrganPoll::Completed(json!({"played": pattern}))
                    }
                    Err(error) => OrganPoll::Failed(cockpit_skill_failure(operation, error)),
                }
            }
            HostOperation::Say { text } => {
                if context.first_poll {
                    primitive = Some(primitive_intent(
                        context,
                        operation,
                        json!({
                            "text": text,
                            "delivery": "motherbrain_mouth_queue",
                        }),
                    ));
                    OrganPoll::Pending {
                        progress: None,
                        primitive: None,
                    }
                } else {
                    OrganPoll::Completed(json!({
                        "text": text,
                        "attempted": true,
                    }))
                }
            }
            HostOperation::Scan
            | HostOperation::LookAt { .. }
            | HostOperation::Grasp { .. }
            | HostOperation::Release { .. }
            | HostOperation::BringToMouth { .. }
            | HostOperation::Chew
            | HostOperation::Swallow => OrganPoll::Failed(SkillFailure::capability(operation)),
        };
        if primitive.is_some() {
            state.last_dispatch_ms = context.now_ms;
        }
        match outcome {
            OrganPoll::Pending { progress, .. } => OrganPoll::Pending {
                progress,
                primitive,
            },
            terminal => {
                self.state.operations.remove(&context.operation_id);
                terminal
            }
        }
    }

    fn stop(&mut self, resource: BodyResource, _reason: &SkillFailure) {
        if resource == BodyResource::Locomotion {
            let _ = self.cockpit.stop();
            self.command_sent = true;
        }
    }
}

struct CockpitStopDriver<'a, C> {
    cockpit: &'a mut C,
}

impl<C: Cockpit> OrganDriver for CockpitStopDriver<'_, C> {
    fn poll(
        &mut self,
        operation: &HostOperation,
        _context: OperationContext,
        _now: &Now,
        _events: &pete_cockpit::EventBatch,
    ) -> OrganPoll {
        OrganPoll::Failed(
            SkillFailure::new(
                SkillOutcome::ResourcePreempted,
                "foreground_replaced",
                "operation was replaced before it could be polled",
            )
            .for_operation(operation),
        )
    }

    fn stop(&mut self, resource: BodyResource, _reason: &SkillFailure) {
        if resource == BodyResource::Locomotion {
            let _ = self.cockpit.stop();
        }
    }
}

impl PossessorSkillRuntime {
    fn active_request(&self) -> Option<SkillRequest> {
        self.status
            .as_ref()
            .filter(|status| status.phase != SkillPhase::Terminal)
            .map(|status| status.request.clone())
    }

    fn request_for_tick(&self, candidate: Option<SkillRequest>) -> Option<SkillRequest> {
        match (self.active_request(), candidate) {
            (Some(active), Some(candidate))
                if active.skill_id == candidate.skill_id
                    && active.implementation_id == candidate.implementation_id
                    && active.goal_id == candidate.goal_id =>
            {
                Some(candidate)
            }
            (Some(active), _) => Some(active),
            (None, candidate) => candidate,
        }
    }

    fn annotate_now(&self, now: &mut Now) {
        if let Some(provenance) = &self.provenance {
            now.extensions.insert(
                "motherbrain.skill_execution".to_string(),
                provenance.clone(),
            );
        }
    }

    fn step<C: Cockpit>(
        &mut self,
        cockpit: &mut C,
        request: &SkillRequest,
        now: &Now,
        status_summary: &StatusSummary,
        home_base_contact: bool,
        events: &pete_cockpit::EventBatch,
        now_ms: TimeMs,
    ) -> (SkillStatus, bool) {
        let Some(lua) = self.lua.as_mut() else {
            let status = SkillStatus {
                request: request.clone(),
                phase: SkillPhase::Terminal,
                outcome: Some(SkillOutcome::ScriptError),
                updated_at_ms: now_ms,
                reason: self.load_error.clone(),
                ..SkillStatus::default()
            };
            self.status = Some(status.clone());
            return (status, false);
        };
        if now_ms.saturating_sub(self.last_reload_check_ms) >= 1_000 {
            let _ = lua.reload();
            self.last_reload_check_ms = now_ms;
        }
        if let Some(current) = lua.active_skill_id() {
            let wanted = if request.skill_id == SkillId::RuntimeLoaded {
                request.implementation_id.clone().unwrap_or_default()
            } else {
                format!(
                    "motherbrain.{}",
                    match request.skill_id {
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
                        SkillId::RuntimeLoaded => unreachable!(),
                    }
                )
            };
            if current != wanted {
                if matches!(
                    request.skill_id,
                    SkillId::RetreatFromCliff | SkillId::ReleasePersistentBumper
                ) {
                    let mut stop_driver = CockpitStopDriver { cockpit };
                    let _ = lua.cancel(
                        &mut stop_driver,
                        SkillOutcome::ResourcePreempted,
                        "safety_recovery_preempted_foreground",
                        "acknowledged bodily hazard replaced the foreground skill",
                        now_ms,
                    );
                    self.provenance = lua
                        .execution_record()
                        .and_then(|record| serde_json::to_value(record).ok());
                    let _ = lua.take_terminal();
                    self.driver.operations.clear();
                } else {
                    // Competing ordinary goals remain pending. Only explicit
                    // higher authority calls the cancellation API.
                    return (
                        self.status.clone().unwrap_or_else(|| SkillStatus {
                            request: request.clone(),
                            phase: SkillPhase::Running,
                            updated_at_ms: now_ms,
                            reason: Some(format!("foreground skill {current} retains commitment")),
                            ..SkillStatus::default()
                        }),
                        false,
                    );
                }
            }
        }
        if !lua.is_active() {
            if let Err(error) = lua.start(request.clone(), now) {
                let status = SkillStatus {
                    request: request.clone(),
                    phase: SkillPhase::Terminal,
                    outcome: Some(SkillOutcome::ScriptError),
                    updated_at_ms: now_ms,
                    reason: Some(error.to_string()),
                    ..SkillStatus::default()
                };
                self.status = Some(status.clone());
                return (status, false);
            }
        }
        let mut driver = RealLuaOrganDriver {
            cockpit,
            request,
            status: status_summary,
            home_base_contact,
            state: &mut self.driver,
            command_sent: false,
        };
        let status = lua
            .step(now, events, &mut driver)
            .unwrap_or_else(|| SkillStatus {
                request: request.clone(),
                phase: SkillPhase::Terminal,
                outcome: Some(SkillOutcome::ScriptError),
                updated_at_ms: now_ms,
                reason: Some("Lua runtime lost the foreground invocation".into()),
                ..SkillStatus::default()
            });
        let command_sent = driver.command_sent;
        self.status = Some(status.clone());
        self.provenance = lua
            .execution_record()
            .and_then(|record| serde_json::to_value(record).ok());
        if status.phase == SkillPhase::Terminal {
            let _ = lua.take_terminal();
        }
        (status, command_sent)
    }
}

fn radians_to_mrad(value: f32) -> i16 {
    (value * 1_000.0)
        .round()
        .clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn distance_from_start(state: &LuaDriverOperationState, body: &BodySense) -> f32 {
    (body.odometry.x_m - state.start_x_m).hypot(body.odometry.y_m - state.start_y_m)
}

fn angle_delta(current: f32, initial: f32) -> f32 {
    let mut delta = current - initial;
    while delta > std::f32::consts::PI {
        delta -= std::f32::consts::TAU;
    }
    while delta < -std::f32::consts::PI {
        delta += std::f32::consts::TAU;
    }
    delta
}

fn primitive_intent(
    context: OperationContext,
    operation: &HostOperation,
    detail: serde_json::Value,
) -> PrimitiveIntent {
    PrimitiveIntent {
        operation_id: context.operation_id,
        child_id: context.child_id,
        operation: operation.name().to_string(),
        resource: operation.resource(),
        emitted_at_ms: context.now_ms,
        detail,
    }
}

fn dispatch_velocity<C: Cockpit>(
    cockpit: &mut C,
    operation: &HostOperation,
    context: OperationContext,
    linear_mm_s: i16,
    angular_mrad_s: i16,
    command_sent: &mut bool,
    primitive: &mut Option<PrimitiveIntent>,
) -> OrganPoll {
    match cockpit.cmd_vel(linear_mm_s, angular_mrad_s, context.primitive_ttl_ms) {
        Ok(()) => {
            *command_sent = true;
            *primitive = Some(primitive_intent(
                context,
                operation,
                json!({
                    "linear_mm_s": linear_mm_s,
                    "angular_mrad_s": angular_mrad_s,
                    "ttl_ms": context.primitive_ttl_ms,
                }),
            ));
            OrganPoll::Pending {
                progress: None,
                primitive: None,
            }
        }
        Err(error) => OrganPoll::Failed(cockpit_skill_failure(operation, error)),
    }
}

fn with_operation_progress(poll: OrganPoll, progress: Option<(String, f32)>) -> OrganPoll {
    match poll {
        OrganPoll::Pending { primitive, .. } => OrganPoll::Pending {
            progress,
            primitive,
        },
        terminal => terminal,
    }
}

fn complete_stopped<C: Cockpit>(
    cockpit: &mut C,
    operation: &HostOperation,
    value: serde_json::Value,
    command_sent: &mut bool,
) -> OrganPoll {
    match cockpit.stop() {
        Ok(()) => {
            *command_sent = true;
            OrganPoll::Completed(value)
        }
        Err(error) => OrganPoll::Failed(cockpit_skill_failure(operation, error)),
    }
}

fn cockpit_skill_failure(
    operation: &HostOperation,
    error: pete_cockpit::CockpitError,
) -> SkillFailure {
    SkillFailure::new(
        SkillOutcome::Failed,
        "body_command_rejected",
        error.to_string(),
    )
    .for_operation(operation)
}
