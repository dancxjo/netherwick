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
}

#[derive(Clone, Debug, Default)]
struct SensorPollHealth {
    name: String,
    available: bool,
    consecutive_failures: u32,
    last_error: Option<String>,
    last_report_ms: TimeMs,
    last_success_ms: TimeMs,
}

const SENSOR_FAILURE_REPORT_INTERVAL_MS: TimeMs = 30_000;

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
                        json!({"charging": true}),
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
                if now.body.charging || self.home_base_contact {
                    OrganPoll::Completed(json!({"charging": true}))
                } else if context.elapsed_ms >= 1_000 {
                    OrganPoll::Failed(
                        SkillFailure::new(
                            SkillOutcome::PostconditionFailed,
                            "charging_not_verified",
                            "Home Base contact did not produce charging",
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PossessionRecoveryPhase {
    Idle,
    BrainstemReflex,
    WaitingForSensorClear,
    Escaping,
}

#[derive(Clone, Debug)]
struct PossessionRecoveryState {
    latch: Option<SafetyLatchKind>,
    hazard_generation: u32,
    phase: PossessionRecoveryPhase,
    turn_direction: TurnDir,
    active_since_ms: TimeMs,
    last_command_ms: TimeMs,
    command_attempts: u32,
    stuck_stop_sent: bool,
    brainstem_reflex_observed: bool,
    last_reflex_outcome: Option<ContactWithdrawalOutcome>,
    last_observed_x_m: f32,
    last_observed_y_m: f32,
    last_observed_heading_rad: f32,
    observed_linear_m: f32,
    observed_turn_rad: f32,
}

impl Default for PossessionRecoveryState {
    fn default() -> Self {
        Self {
            latch: None,
            hazard_generation: 0,
            phase: PossessionRecoveryPhase::Idle,
            turn_direction: TurnDir::Left,
            active_since_ms: 0,
            last_command_ms: 0,
            command_attempts: 0,
            stuck_stop_sent: false,
            brainstem_reflex_observed: false,
            last_reflex_outcome: None,
            last_observed_x_m: 0.0,
            last_observed_y_m: 0.0,
            last_observed_heading_rad: 0.0,
            observed_linear_m: 0.0,
            observed_turn_rad: 0.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct MotionRejectionState {
    first_ms: TimeMs,
    last_ms: TimeMs,
    blocked_until_ms: TimeMs,
    latest_command_id: u32,
    latest_reason: Option<String>,
    count: u32,
    stuck: bool,
    stuck_stop_sent: bool,
}

const POSSESSION_RECOVERY_STUCK_AFTER_MS: TimeMs = 15_000;
const POSSESSION_ESCAPE_TTL_MS: u32 = 250;
const MOTION_REJECTION_WINDOW_MS: TimeMs = 5_000;
const MOTION_REJECTION_BASE_BACKOFF_MS: TimeMs = 1_000;
const MOTION_REJECTION_MAX_BACKOFF_MS: TimeMs = 5_000;
const MOTION_REJECTION_STUCK_AFTER: u32 = 3;

#[derive(Clone, Debug)]
struct PossessionRecoveryDecision {
    block_reason: Option<String>,
    command_sent: bool,
    action: Option<ActionPrimitive>,
    motor: Option<MotorCommand>,
    debug: serde_json::Value,
}

impl<R, C> RealRobotRunner<R, C>
where
    R: RuntimeLoop + Send,
    C: Cockpit + Send,
{
    pub fn new(
        mode: RobotMode,
        cockpit: C,
        sensors: Vec<Box<dyn SenseProducer + Send>>,
        runtime: R,
    ) -> Self {
        let sensor_poll_health = sensors
            .iter()
            .map(|sensor| SensorPollHealth {
                name: sensor.source_name().to_string(),
                ..SensorPollHealth::default()
            })
            .collect();
        Self {
            mode,
            cockpit: SafeCockpit::new(cockpit),
            sensors,
            runtime,
            tick_ms: 100,
            tick_count: 0,
            autonomous_motion: false,
            now_builder: NowBuilder::new(),
            frame_processor: FrameProcessor::new(),
            live_image_cognition: LiveImageCognition::new(None),
            robot_initialization: None,
            brainstem_interface: None,
            possession_recovery: PossessionRecoveryState::default(),
            motion_rejection: MotionRejectionState::default(),
            possessor_skills: PossessorSkillRuntime::default(),
            sensor_poll_health,
        }
    }

    pub fn with_frame_processor(mut self, frame_processor: FrameProcessor) -> Self {
        self.frame_processor = frame_processor;
        self
    }

    pub fn with_live_image_enricher(mut self, enricher: Option<LiveImageEnricher>) -> Self {
        self.live_image_cognition = LiveImageCognition::new(enricher);
        self
    }

    pub fn with_robot_initialization(mut self, initialization: serde_json::Value) -> Self {
        self.robot_initialization = Some(initialization);
        self
    }

    pub fn with_brainstem_interface(mut self, capabilities: serde_json::Value) -> Self {
        self.brainstem_interface = Some(capabilities);
        self
    }

    pub fn with_autonomous_motion(mut self, enabled: bool) -> Self {
        self.autonomous_motion = enabled;
        self
    }

    pub async fn run_read_only(&mut self, steps: Option<usize>) -> Result<()> {
        self.run_read_only_observing(steps, |_, _| {}).await
    }

    pub async fn run_read_only_observing<F>(
        &mut self,
        steps: Option<usize>,
        mut observe: F,
    ) -> Result<()>
    where
        F: FnMut(&WorldSnapshot, &RuntimeTick),
    {
        if self.mode != RobotMode::ReadOnly {
            anyhow::bail!("only read-only robot mode is implemented");
        }

        while steps.map(|limit| self.tick_count < limit).unwrap_or(true) {
            let (snapshot, tick) = self.tick_read_only().await?;
            observe(&snapshot, &tick);
            tokio::time::sleep(std::time::Duration::from_millis(self.tick_ms)).await;
        }
        Ok(())
    }

    pub async fn tick_read_only(&mut self) -> Result<(WorldSnapshot, RuntimeTick)> {
        if self.mode != RobotMode::ReadOnly {
            anyhow::bail!("only read-only robot mode is implemented");
        }

        let body = body_sense_from_cockpit_status(self.cockpit.refresh_status()?, wall_time_ms());
        let brainstem_events = self.cockpit.poll_events()?;
        let mut packets = poll_sensors_lossy(&mut self.sensors, &mut self.sensor_poll_health).await;
        let t_ms = body.last_update_ms.max(wall_time_ms());
        self.frame_processor.process_packets(t_ms, &mut packets);
        let mut now = self.now_builder.build(t_ms, body, packets)?;
        insert_sensor_health(&mut now, &self.sensor_poll_health);
        now.self_sense.mode = Some("read-only".to_string());
        now.extensions.insert(
            "source".to_string(),
            serde_json::Value::String("real_robot_read_only".to_string()),
        );
        now.extensions.insert(
            "mode".to_string(),
            serde_json::Value::String("read_only".to_string()),
        );
        now.extensions.insert(
            "safety/read_only_veto".to_string(),
            serde_json::Value::Bool(true),
        );
        now.extensions.insert(
            "read_only_motor_gate".to_string(),
            serde_json::json!({
                "motor_applied": false,
                "final_motor": MotorCommand::stop(),
                "safety_reason": "ReadOnlyMode",
            }),
        );
        self.insert_robot_initialization(&mut now);
        self.insert_brainstem_interface(&mut now, &brainstem_events);
        self.possessor_skills.annotate_now(&mut now);
        enrich_now_latest_image(&mut self.live_image_cognition, &mut now).await;

        let tick = self
            .runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await?;
        let mut snapshot = self.now_builder.snapshot();
        snapshot.eye = tick.frame.now.eye.clone();
        annotate_snapshot_from_tick(&mut snapshot, &tick);
        self.tick_count = self.tick_count.saturating_add(1);
        Ok((snapshot, tick))
    }

    pub async fn tick_slow_manual(&mut self) -> Result<(WorldSnapshot, RuntimeTick)> {
        if self.mode != RobotMode::Slow {
            anyhow::bail!("slow manual tick requires RobotMode::Slow");
        }

        let pre_body_t_ms = wall_time_ms();
        let brainstem_events = self.poll_slow_possession_events()?;
        if let Some(input) = self.runtime.reign_sense(pre_body_t_ms)?.latest {
            if reign_input_outputs_real_slow_directly(&input) {
                let body_before = pete_body::BodySense {
                    last_update_ms: pre_body_t_ms,
                    ..pete_body::BodySense::default()
                };
                let mut now =
                    self.now_builder
                        .build(pre_body_t_ms, body_before.clone(), Vec::new())?;
                now.self_sense.mode = Some("slow".to_string());
                self.insert_possession_snapshot(&mut now);
                now.extensions.insert(
                    "source".to_string(),
                    serde_json::Value::String("real_robot".to_string()),
                );
                now.extensions.insert(
                    "mode".to_string(),
                    serde_json::Value::String("slow".to_string()),
                );
                self.insert_robot_initialization(&mut now);
                self.insert_brainstem_interface(&mut now, &brainstem_events);
                now.reign.latest = Some(input.clone());
                now.reign.active = true;
                now.reign.mode = Some(input.mode.clone());
                now.reign.human_override_pressure = input.priority.clamp(0.0, 1.0);
                now.reign.last_command_age_ms =
                    Some(pre_body_t_ms.saturating_sub(input.issued_at_ms));
                let tick = synthetic_slow_manual_tick(
                    now,
                    input,
                    MotorCommand::stop(),
                    MotorCommand::stop(),
                    None,
                    &body_before,
                )?;
                let mut snapshot = self.now_builder.snapshot();
                snapshot.eye = tick.frame.now.eye.clone();
                annotate_snapshot_from_tick(&mut snapshot, &tick);
                self.tick_count = self.tick_count.saturating_add(1);
                return Ok((snapshot, tick));
            }
        }

        let status_before = self.cockpit.refresh_status()?;
        let body_before = body_sense_from_cockpit_status(status_before.clone(), wall_time_ms());
        let recovery_decision =
            self.apply_possession_recovery(&body_before, &brainstem_events, &status_before)?;
        let mut packets = poll_sensors_lossy(&mut self.sensors, &mut self.sensor_poll_health).await;
        let t_ms = body_before.last_update_ms.max(wall_time_ms());
        self.frame_processor.process_packets(t_ms, &mut packets);
        let mut now = self.now_builder.build(t_ms, body_before.clone(), packets)?;
        insert_sensor_health(&mut now, &self.sensor_poll_health);
        now.self_sense.mode = Some("slow".to_string());
        self.insert_possession_snapshot(&mut now);
        now.extensions.insert(
            "source".to_string(),
            serde_json::Value::String("real_robot".to_string()),
        );
        now.extensions.insert(
            "mode".to_string(),
            serde_json::Value::String("slow".to_string()),
        );
        self.insert_robot_initialization(&mut now);
        self.insert_brainstem_interface(&mut now, &brainstem_events);
        self.possessor_skills.annotate_now(&mut now);

        now.reign = self.runtime.reign_sense(t_ms)?;
        if let Some(input) = now.reign.latest.clone() {
            if reign_input_outputs_real_slow_directly(&input) {
                let tick = synthetic_slow_manual_tick(
                    now,
                    input,
                    MotorCommand::stop(),
                    MotorCommand::stop(),
                    None,
                    &body_before,
                )?;
                let mut snapshot = self.now_builder.snapshot();
                snapshot.eye = tick.frame.now.eye.clone();
                annotate_snapshot_from_tick(&mut snapshot, &tick);
                self.tick_count = self.tick_count.saturating_add(1);
                return Ok((snapshot, tick));
            }
            if reign_input_drives_real_slow(&input) {
                let desired_motor =
                    action_to_motor_command(input.command.to_action().as_ref()).clamped(0.05, 0.5);
                let mut block_reason = recovery_decision
                    .block_reason
                    .clone()
                    .or_else(|| self.motion_rejection_block_reason(wall_time_ms()))
                    .or_else(|| real_slow_body_block_reason(&body_before));
                let mut final_motor = if block_reason.is_none() {
                    desired_motor
                } else {
                    MotorCommand::stop()
                };
                if !recovery_decision.command_sent {
                    if let Some(block) =
                        apply_slow_possession_motor(&mut self.cockpit, final_motor)?
                    {
                        match block {
                            SlowPossessionMotionBlock::SafetyLatch(latch) => {
                                self.start_possession_recovery(latch, &body_before);
                                block_reason = Some(format!("recovering {latch:?} safety latch"));
                            }
                            SlowPossessionMotionBlock::CommandRejected { command_id, reason } => {
                                block_reason =
                                    Some(self.record_motion_rejection(command_id, &reason));
                            }
                        }
                        final_motor = MotorCommand::stop();
                    } else if !is_near_zero_motor(final_motor) {
                        self.motion_rejection = MotionRejectionState::default();
                    }
                }
                let tick = synthetic_slow_manual_tick(
                    now,
                    input,
                    desired_motor,
                    final_motor,
                    block_reason,
                    &body_before,
                )?;
                let mut snapshot = self.now_builder.snapshot();
                snapshot.eye = tick.frame.now.eye.clone();
                annotate_snapshot_from_tick(&mut snapshot, &tick);
                self.tick_count = self.tick_count.saturating_add(1);
                return Ok((snapshot, tick));
            }
        }

        enrich_now_latest_image(&mut self.live_image_cognition, &mut now).await;

        let mut tick = self
            .runtime
            .tick(now.clone(), ExperienceLatent::default(), Vec::new())
            .await?;
        let original_chosen_action = tick.chosen_action.clone();
        let chosen_motor = final_motor_from_tick(&tick).clamped(0.05, 0.5);
        let manual_drive = tick
            .frame
            .reign_input
            .as_ref()
            .map(reign_input_drives_real_slow)
            .unwrap_or(false);
        let mut block_reason = recovery_decision
            .block_reason
            .clone()
            .or_else(|| self.motion_rejection_block_reason(wall_time_ms()))
            .or_else(|| {
                real_slow_motor_block_reason(
                    &body_before,
                    &tick,
                    manual_drive,
                    self.autonomous_motion,
                )
            });
        let mut final_motor = if block_reason.is_none() {
            chosen_motor
        } else {
            MotorCommand::stop()
        };
        let recovery_request = self.possession_recovery_skill_request(&brainstem_events);
        let reflex_preemption = brainstem_events.events.iter().any(|event| {
            matches!(
                event.kind,
                CockpitEventKind::ContactWithdrawalStarted
                    | CockpitEventKind::SafetyTripped
                    | CockpitEventKind::EStopLatched
            )
        });
        let interrupted_request = reflex_preemption
            .then(|| self.possessor_skills.status.as_ref())
            .flatten()
            .filter(|status| status.phase != SkillPhase::Terminal)
            .map(|status| status.request.clone());
        let selected_skill_request =
            recovery_request
                .clone()
                .or(interrupted_request)
                .or_else(|| {
                    self.possessor_skills
                        .request_for_tick(tick.skill_request.clone())
                });
        let possessor_skill_owns_motion = selected_skill_request.is_some();
        let mut recovery_motion_sent = false;
        if let Some(request) = selected_skill_request {
            tick.skill_request = Some(request.clone());
            final_motor = MotorCommand::stop();
            if (block_reason.is_none() || recovery_request.is_some() || reflex_preemption)
                && !recovery_decision.command_sent
            {
                let (status, command_sent) = self.possessor_skills.step(
                    self.cockpit.client_mut(),
                    &request,
                    &tick.frame.now,
                    &status_before,
                    status_before.battery.home_base(),
                    &brainstem_events,
                    t_ms,
                );
                recovery_motion_sent = command_sent
                    && recovery_request.is_some()
                    && status.script.as_ref().is_some_and(|script| {
                        script.current_operation.as_deref() == Some("retreat")
                    });
                self.runtime.observe_skill_status(&status);
                if recovery_request.is_some()
                    && status.phase == SkillPhase::Terminal
                    && status.outcome == Some(SkillOutcome::Completed)
                {
                    self.finish_possession_recovery();
                }
                tick.skill_status = Some(status);
                self.possessor_skills.annotate_now(&mut tick.frame.now);
            }
        }
        if !recovery_decision.command_sent && !possessor_skill_owns_motion {
            if let Some(block) = apply_slow_possession_motor(&mut self.cockpit, final_motor)? {
                match block {
                    SlowPossessionMotionBlock::SafetyLatch(latch) => {
                        self.start_possession_recovery(latch, &body_before);
                        block_reason = Some(format!("recovering {latch:?} safety latch"));
                    }
                    SlowPossessionMotionBlock::CommandRejected { command_id, reason } => {
                        block_reason = Some(self.record_motion_rejection(command_id, &reason));
                    }
                }
                final_motor = MotorCommand::stop();
            } else if !is_near_zero_motor(final_motor) {
                self.motion_rejection = MotionRejectionState::default();
            }
        }
        if self.motion_rejection_block_reason(wall_time_ms()).is_some() {
            tick.chosen_action = Some(ActionPrimitive::Stop);
            tick.frame.chosen_action = Some(ActionPrimitive::Stop);
            if self.motion_rejection.stuck && !self.motion_rejection.stuck_stop_sent {
                self.cockpit.client_mut().stop()?;
                self.motion_rejection.stuck_stop_sent = true;
            }
        } else if let Some(recovery_action) = recovery_decision.action.clone() {
            tick.chosen_action = Some(recovery_action.clone());
            tick.frame.chosen_action = Some(recovery_action);
        } else if recovery_motion_sent {
            let recovery_action = ActionPrimitive::Go {
                intensity: -0.25,
                duration_ms: POSSESSION_ESCAPE_TTL_MS as TimeMs,
            };
            tick.chosen_action = Some(recovery_action.clone());
            tick.frame.chosen_action = Some(recovery_action);
        } else if possessor_skill_owns_motion {
            tick.chosen_action = Some(ActionPrimitive::Stop);
            tick.frame.chosen_action = Some(ActionPrimitive::Stop);
        }

        let mut snapshot = self.now_builder.snapshot();
        snapshot.eye = tick.frame.now.eye.clone();
        annotate_snapshot_from_tick(&mut snapshot, &tick);
        let mut action_debug = snapshot
            .action_debug
            .take()
            .unwrap_or_else(|| serde_json::json!({}));
        if !action_debug.is_object() {
            action_debug = serde_json::json!({});
        }
        let reported_robot_motor = if recovery_motion_sent {
            MotorCommand {
                forward: -0.10,
                turn: 0.0,
            }
        } else if recovery_decision.command_sent {
            recovery_decision.motor.unwrap_or_else(MotorCommand::stop)
        } else if recovery_decision.block_reason.is_some() {
            MotorCommand::stop()
        } else {
            final_motor
        };
        if let Some(object) = action_debug.as_object_mut() {
            object.insert("body_pose_before".to_string(), pose_json(&body_before));
            object.insert("body_pose_after".to_string(), pose_json(&snapshot.body));
            object.insert(
                "desired_motor".to_string(),
                serde_json::to_value(chosen_motor)?,
            );
            object.insert(
                "final_motor".to_string(),
                serde_json::to_value(final_motor)?,
            );
            object.insert(
                "motion_sent_to_robot".to_string(),
                serde_json::to_value(motor_command_to_motion(reported_robot_motor))?,
            );
            object.insert(
                "motion_sent_to_sim".to_string(),
                serde_json::to_value(motor_command_to_motion(final_motor))?,
            );
            object.insert(
                "motor_applied".to_string(),
                serde_json::json!(!is_near_zero_motor(reported_robot_motor)),
            );
            object.insert(
                "runtime_chosen_action".to_string(),
                serde_json::to_value(original_chosen_action)?,
            );
            object.insert(
                "recovery_action".to_string(),
                serde_json::to_value(recovery_decision.action.clone())?,
            );
            object.insert(
                "recovery_motor".to_string(),
                recovery_decision
                    .motor
                    .map(serde_json::to_value)
                    .transpose()?
                    .unwrap_or(serde_json::Value::Null),
            );
            object.insert(
                "manual_hardware_gate".to_string(),
                serde_json::json!(manual_drive),
            );
            object.insert(
                "autonomous_hardware_gate".to_string(),
                serde_json::json!(self.autonomous_motion),
            );
            object.insert(
                "possession_recovery".to_string(),
                recovery_decision.debug.clone(),
            );
            object.insert(
                "possessor_skill_request".to_string(),
                serde_json::to_value(&tick.skill_request)?,
            );
            object.insert(
                "possessor_skill_status".to_string(),
                serde_json::to_value(&tick.skill_status)?,
            );
            object.insert(
                "possessor_skill_execution".to_string(),
                self.possessor_skills
                    .provenance
                    .clone()
                    .unwrap_or(serde_json::Value::Null),
            );
            object.insert(
                "motion_rejection".to_string(),
                motion_rejection_debug(&self.motion_rejection),
            );
            object.insert(
                "why_not_moving".to_string(),
                block_reason
                    .clone()
                    .map(serde_json::Value::String)
                    .unwrap_or(serde_json::Value::Null),
            );
        }
        snapshot.action_debug = Some(action_debug);
        self.tick_count = self.tick_count.saturating_add(1);
        Ok((snapshot, tick))
    }

    fn insert_robot_initialization(&mut self, now: &mut Now) {
        if self.tick_count == 0 {
            if let Some(initialization) = self.robot_initialization.take() {
                now.extensions
                    .insert("robot.initialization".to_string(), initialization);
            }
        }
    }

    fn insert_possession_snapshot(&mut self, now: &mut Now) {
        if let Some(snapshot) = self.cockpit.client_mut().possession_snapshot() {
            now.extensions.insert(
                "brainstem.possession".to_string(),
                serde_json::to_value(snapshot).unwrap_or_else(|error| {
                    serde_json::json!({"possessed": false, "refusal_reason": error.to_string()})
                }),
            );
        }
    }

    fn poll_slow_possession_events(&mut self) -> Result<pete_cockpit::EventBatch> {
        let events = self.cockpit.poll_events_allowing_history_gap()?;
        if events.dropped_before_seq > 0 {
            if self.tick_count == 0 {
                eprintln!(
                    "slow possession recovered from pre-loop event history gap before sequence {}",
                    events.dropped_before_seq
                );
            } else {
                eprintln!(
                    "slow possession recovered from event history gap before sequence {}; stopping before continuing",
                    events.dropped_before_seq
                );
                self.cockpit.client_mut().stop()?;
            }
        }
        Ok(events)
    }

    fn apply_possession_recovery(
        &mut self,
        body: &BodySense,
        events: &pete_cockpit::EventBatch,
        status: &StatusSummary,
    ) -> Result<PossessionRecoveryDecision> {
        for event in &events.events {
            if let Some(reflex) = event.contact_withdrawal() {
                match reflex {
                    ContactWithdrawalEvent::Started { repeated_count, .. } => {
                        self.start_possession_recovery(SafetyLatchKind::Bump, body);
                        self.possession_recovery.phase = PossessionRecoveryPhase::BrainstemReflex;
                        self.possession_recovery.brainstem_reflex_observed = true;
                        self.possession_recovery.command_attempts = u32::from(repeated_count);
                        self.possession_recovery.last_command_ms = wall_time_ms();
                        self.possession_recovery.last_reflex_outcome = None;
                    }
                    ContactWithdrawalEvent::Completed { outcome, .. } => {
                        self.possession_recovery.brainstem_reflex_observed = true;
                        self.possession_recovery.last_reflex_outcome = Some(outcome);
                        self.possession_recovery.phase =
                            PossessionRecoveryPhase::WaitingForSensorClear;
                    }
                }
                continue;
            }
            match event.kind {
                CockpitEventKind::SafetyTripped => {
                    if let Some(kind) = safety_latch_kind_from_event_code(event.a) {
                        self.start_possession_recovery_generation(kind, event.seq, body);
                    }
                }
                CockpitEventKind::SafetyCleared => {
                    if self.possession_recovery.latch == safety_latch_kind_from_event_code(event.a)
                    {
                        self.possession_recovery = PossessionRecoveryState::default();
                    }
                }
                CockpitEventKind::EStopLatched => {
                    // The event contract does not distinguish an operator
                    // E-stop from any internally generated stop. Without
                    // trustworthy provenance, fail closed and require an
                    // explicit operator clear.
                    self.finish_possession_recovery();
                }
                _ => {}
            }
        }

        if status.estop_latched == Some(true) {
            self.possession_recovery = PossessionRecoveryState::default();
            return Ok(PossessionRecoveryDecision {
                block_reason: Some(
                    "operator E-stop is latched; explicit operator clear required".to_string(),
                ),
                command_sent: false,
                action: Some(ActionPrimitive::Stop),
                motor: Some(MotorCommand::stop()),
                debug: possession_recovery_debug(&self.possession_recovery, None, false),
            });
        }

        if self.possession_recovery.latch.is_none()
            && status.safety_tripped == Some(true)
            && status.estop_latched != Some(true)
        {
            if let Some(kind) = status
                .safety_latch_kind
                .or_else(|| infer_safety_latch_from_sensors(body))
            {
                self.start_possession_recovery_generation(
                    kind,
                    status.safety_hazard_generation.unwrap_or(0),
                    body,
                );
            }
        }

        let Some(latch) = self.possession_recovery.latch else {
            return Ok(PossessionRecoveryDecision {
                block_reason: None,
                command_sent: false,
                action: None,
                motor: None,
                debug: possession_recovery_debug(&self.possession_recovery, None, false),
            });
        };

        self.observe_possession_recovery_motion(body);
        let mut command_sent = false;
        let mut action = None;
        let mut motor = None;
        let mut reason = format!("recovering {latch:?} safety latch");
        match latch {
            SafetyLatchKind::Bump => {
                if self.possession_recovery.phase == PossessionRecoveryPhase::BrainstemReflex {
                    reason =
                        "brainstem contact-withdrawal reflex owns motion; possessor is observing"
                            .to_string();
                } else {
                    reason = "bumper recovery is delegated to the foreground Lua releasePersistentBumper skill"
                        .to_string();
                }
            }
            SafetyLatchKind::Cliff => {
                reason = "cliff recovery is delegated to the foreground Lua retreatFromCliff skill"
                    .to_string();
            }
            SafetyLatchKind::WheelDrop => {
                if body.flags.wheel_drop {
                    action = Some(ActionPrimitive::Stop);
                    motor = Some(MotorCommand::stop());
                    self.cockpit.client_mut().stop()?;
                    command_sent = true;
                } else {
                    self.cockpit.client_mut().clear_safety_latch(latch)?;
                    command_sent = true;
                    self.finish_possession_recovery();
                }
            }
            SafetyLatchKind::Charging => {
                if body.charging {
                    action = Some(ActionPrimitive::Stop);
                    motor = Some(MotorCommand::stop());
                    self.cockpit.client_mut().stop()?;
                    command_sent = true;
                } else {
                    self.cockpit.client_mut().clear_safety_latch(latch)?;
                    command_sent = true;
                    self.finish_possession_recovery();
                }
            }
            SafetyLatchKind::Heartbeat => {
                self.cockpit.client_mut().clear_safety_latch(latch)?;
                self.finish_possession_recovery();
                command_sent = true;
            }
            SafetyLatchKind::Tilt | SafetyLatchKind::Impact => {
                if imu_recovery_clear(status, latch) {
                    self.cockpit.client_mut().clear_safety_latch(latch)?;
                    self.finish_possession_recovery();
                    command_sent = true;
                } else {
                    action = Some(ActionPrimitive::Stop);
                    motor = Some(MotorCommand::stop());
                    self.cockpit.client_mut().stop()?;
                    command_sent = true;
                }
            }
        }

        Ok(PossessionRecoveryDecision {
            block_reason: Some(reason),
            command_sent,
            action,
            motor,
            debug: possession_recovery_debug(&self.possession_recovery, Some(latch), command_sent),
        })
    }

    fn possession_recovery_skill_request(
        &self,
        events: &pete_cockpit::EventBatch,
    ) -> Option<SkillRequest> {
        if self.possession_recovery.phase == PossessionRecoveryPhase::BrainstemReflex
            || self.possession_recovery.hazard_generation == 0
            || events.events.iter().any(|event| {
                matches!(
                    event.kind,
                    CockpitEventKind::ContactWithdrawalStarted | CockpitEventKind::SafetyTripped
                )
            })
        {
            return None;
        }
        let skill_id = match self.possession_recovery.latch? {
            SafetyLatchKind::Bump => SkillId::ReleasePersistentBumper,
            SafetyLatchKind::Cliff => SkillId::RetreatFromCliff,
            _ => return None,
        };
        Some(SkillRequest {
            skill_id,
            goal_id: Some(pete_conductor::GoalId::new("escape_danger")),
            behavior_id: Some("acknowledged_hazard_recovery".to_string()),
            maximum_duration_ms: POSSESSION_RECOVERY_STUCK_AFTER_MS,
            expected_progress: 1.0,
            progress_metric: "reverse_displacement".to_string(),
            progress_baseline: Some(0.0),
            progress_tolerance: 0.1,
            ..SkillRequest::default()
        })
    }

    fn start_possession_recovery(&mut self, latch: SafetyLatchKind, body: &BodySense) {
        self.start_possession_recovery_generation(latch, 0, body);
    }

    fn start_possession_recovery_generation(
        &mut self,
        latch: SafetyLatchKind,
        hazard_generation: u32,
        body: &BodySense,
    ) {
        let now_ms = wall_time_ms();
        let latch_changed = self.possession_recovery.latch != Some(latch);
        self.possession_recovery.latch = Some(latch);
        if hazard_generation != 0 {
            self.possession_recovery.hazard_generation = hazard_generation;
        }
        self.possession_recovery.phase = PossessionRecoveryPhase::WaitingForSensorClear;
        self.possession_recovery.turn_direction = recovery_turn_direction_for_latch(latch, body);
        if latch_changed || self.possession_recovery.active_since_ms == 0 {
            self.possession_recovery.active_since_ms = now_ms;
            self.possession_recovery.last_command_ms = 0;
            self.possession_recovery.command_attempts = 0;
            self.possession_recovery.stuck_stop_sent = false;
            self.possession_recovery.last_observed_x_m = body.odometry.x_m;
            self.possession_recovery.last_observed_y_m = body.odometry.y_m;
            self.possession_recovery.last_observed_heading_rad = body.odometry.heading_rad;
            self.possession_recovery.observed_linear_m = 0.0;
            self.possession_recovery.observed_turn_rad = 0.0;
        }
    }

    fn observe_possession_recovery_motion(&mut self, body: &BodySense) {
        if self.possession_recovery.latch.is_none() {
            return;
        }
        let dx = body.odometry.x_m - self.possession_recovery.last_observed_x_m;
        let dy = body.odometry.y_m - self.possession_recovery.last_observed_y_m;
        let distance = dx.hypot(dy);
        let heading_delta =
            body.odometry.heading_rad - self.possession_recovery.last_observed_heading_rad;
        self.possession_recovery.observed_linear_m += distance;
        self.possession_recovery.observed_turn_rad += heading_delta;
        self.possession_recovery.last_observed_x_m = body.odometry.x_m;
        self.possession_recovery.last_observed_y_m = body.odometry.y_m;
        self.possession_recovery.last_observed_heading_rad = body.odometry.heading_rad;
    }

    fn finish_possession_recovery(&mut self) {
        self.possession_recovery.latch = None;
        self.possession_recovery.hazard_generation = 0;
        self.possession_recovery.phase = PossessionRecoveryPhase::Idle;
        self.possession_recovery.active_since_ms = 0;
        self.possession_recovery.last_command_ms = 0;
        self.possession_recovery.brainstem_reflex_observed = false;
    }

    fn motion_rejection_block_reason(&self, now_ms: TimeMs) -> Option<String> {
        let reason = self
            .motion_rejection
            .latest_reason
            .as_deref()
            .unwrap_or("unknown rejection");
        if self.motion_rejection.stuck {
            return Some(format!(
                "brainstem repeatedly rejected motion; latest command #{}: {}; operator intervention needed",
                self.motion_rejection.latest_command_id, reason
            ));
        }
        if self.motion_rejection.blocked_until_ms > now_ms {
            return Some(format!(
                "brainstem rejected motion command #{}: {}; pausing motion retries for {} ms",
                self.motion_rejection.latest_command_id,
                reason,
                self.motion_rejection
                    .blocked_until_ms
                    .saturating_sub(now_ms)
            ));
        }
        None
    }

    fn record_motion_rejection(&mut self, command_id: u32, reason: &str) -> String {
        let now_ms = wall_time_ms();
        if self.motion_rejection.first_ms == 0
            || now_ms.saturating_sub(self.motion_rejection.first_ms) > MOTION_REJECTION_WINDOW_MS
        {
            self.motion_rejection = MotionRejectionState {
                first_ms: now_ms,
                ..MotionRejectionState::default()
            };
        }
        self.motion_rejection.last_ms = now_ms;
        self.motion_rejection.latest_command_id = command_id;
        self.motion_rejection.latest_reason = Some(reason.to_string());
        self.motion_rejection.count = self.motion_rejection.count.saturating_add(1);
        let backoff = MOTION_REJECTION_BASE_BACKOFF_MS
            .saturating_mul(u64::from(self.motion_rejection.count))
            .min(MOTION_REJECTION_MAX_BACKOFF_MS);
        self.motion_rejection.blocked_until_ms = now_ms.saturating_add(backoff);
        if self.motion_rejection.count >= MOTION_REJECTION_STUCK_AFTER {
            self.motion_rejection.stuck = true;
        }

        if self.motion_rejection.stuck {
            format!(
                "brainstem repeatedly rejected motion; latest command #{command_id}: {reason}; operator intervention needed"
            )
        } else {
            format!(
                "brainstem rejected motion command #{command_id}: {reason}; pausing motion retries for {backoff} ms"
            )
        }
    }

    fn insert_brainstem_interface(&self, now: &mut Now, events: &pete_cockpit::EventBatch) {
        if let Some(capabilities) = &self.brainstem_interface {
            now.extensions.insert(
                "brainstem.interface".to_string(),
                serde_json::json!({
                    "capabilities": capabilities,
                    "source": "brainstem",
                    "underlying_body_private": true,
                }),
            );
        }
        now.extensions.insert(
            "brainstem.events".to_string(),
            serde_json::to_value(events).unwrap_or_else(
                |error| serde_json::json!({"error": error.to_string(), "events": []}),
            ),
        );
    }
}

async fn enrich_now_latest_image(cognition: &mut LiveImageCognition, now: &mut Now) {
    let update = cognition
        .poll_and_submit(now.eye_frame.as_ref(), now.world.revision, now.t_ms)
        .await;
    now.extensions.insert(
        "cognition.registry".to_string(),
        serde_json::to_value(&update.registry)
            .unwrap_or_else(|error| serde_json::json!({"error": error.to_string()})),
    );
    if let Some(response) = update.response {
        now.extensions.insert(
            "cognition.describe_scene.last_response".to_string(),
            serde_json::to_value(&response)
                .unwrap_or_else(|error| serde_json::json!({"error": error.to_string()})),
        );
        if let Some(failure) = response.response.failure {
            now.extensions.insert(
                "vision.image_enrichment_error".to_string(),
                serde_json::json!(failure),
            );
        }
    }
    if let Some(enrichment) = update.enrichment {
        now.eye
            .image_description_vectors
            .push(enrichment.image_description_vector);
        now.eye.scene_vectors.push(enrichment.scene_vector);
        now.extensions.insert(
            "vision.latest_image_description".to_string(),
            serde_json::json!({
                "text": enrichment.description,
                "source_frame_id": now
                    .eye
                    .scene_vectors
                    .last()
                    .and_then(|vector| vector.source_frame_id.clone()),
                "scene_vector_count": now.eye.scene_vectors.len(),
                "image_description_vector_count": now.eye.image_description_vectors.len(),
            }),
        );
    }
}

async fn poll_sensors_lossy(
    sensors: &mut [Box<dyn SenseProducer + Send>],
    health: &mut Vec<SensorPollHealth>,
) -> Vec<pete_sensors::SensePacket> {
    let mut packets = Vec::new();
    if health.len() != sensors.len() {
        health.clear();
        health.extend(sensors.iter().map(|sensor| SensorPollHealth {
            name: sensor.source_name().to_string(),
            ..SensorPollHealth::default()
        }));
    }
    for (sensor, health) in sensors.iter_mut().zip(health.iter_mut()) {
        let now_ms = wall_time_ms();
        match tokio::time::timeout(std::time::Duration::from_millis(25), sensor.poll()).await {
            Ok(Ok(packet)) => {
                if health.consecutive_failures > 0 {
                    eprintln!(
                        "optional sensor {} recovered after {} failed polls; brainstem body evidence remained active",
                        health.name, health.consecutive_failures
                    );
                }
                health.available = true;
                health.consecutive_failures = 0;
                health.last_error = None;
                health.last_success_ms = now_ms;
                packets.push(packet);
            }
            Ok(Err(error)) => record_optional_sensor_failure(health, error.to_string(), now_ms),
            Err(_) => record_optional_sensor_failure(health, "poll timed out".to_string(), now_ms),
        }
    }
    packets
}

fn record_optional_sensor_failure(health: &mut SensorPollHealth, error: String, now_ms: TimeMs) {
    health.available = false;
    health.consecutive_failures = health.consecutive_failures.saturating_add(1);
    health.last_error = Some(error.clone());
    if health.last_report_ms == 0
        || now_ms.saturating_sub(health.last_report_ms) >= SENSOR_FAILURE_REPORT_INTERVAL_MS
    {
        eprintln!(
            "optional sensor {} unavailable; continuing with brainstem body evidence: {} ({} failed polls; repeated reports suppressed for 30s)",
            health.name, error, health.consecutive_failures
        );
        health.last_report_ms = now_ms;
    }
}

fn insert_sensor_health(now: &mut Now, health: &[SensorPollHealth]) {
    now.extensions.insert(
        "sensor.health".to_string(),
        serde_json::Value::Array(
            health
                .iter()
                .map(|health| {
                    serde_json::json!({
                        "name": health.name,
                        "available": health.available,
                        "consecutive_failures": health.consecutive_failures,
                        "last_error": health.last_error,
                        "last_success_ms": health.last_success_ms,
                        "body_evidence_independent": true,
                    })
                })
                .collect(),
        ),
    );
}

fn wall_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn final_motor_from_tick(tick: &RuntimeTick) -> MotorCommand {
    tick.frame
        .now
        .extensions
        .get("motor_gate")
        .and_then(|value| value.get("final_motor"))
        .and_then(|value| serde_json::from_value::<MotorCommand>(value.clone()).ok())
        .unwrap_or_else(|| action_to_motor_command(tick.chosen_action.as_ref()))
}

fn annotate_snapshot_from_tick(snapshot: &mut WorldSnapshot, tick: &RuntimeTick) {
    snapshot.final_selected_action = tick.chosen_action.clone();
    snapshot.llm_action_proposal = tick
        .frame
        .now
        .extensions
        .get("llm.action_proposal")
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok());
    snapshot.action_debug = tick
        .frame
        .now
        .extensions
        .get("action.motion_bridge")
        .cloned();
    if let Some(possession) = tick.frame.now.extensions.get("brainstem.possession") {
        let debug = snapshot
            .action_debug
            .get_or_insert_with(|| serde_json::json!({}));
        if !debug.is_object() {
            *debug = serde_json::json!({});
        }
        debug
            .as_object_mut()
            .expect("action debug was normalized to an object")
            .insert("brainstem_possession".to_string(), possession.clone());
    }
}

fn motor_command_to_motion(motor: MotorCommand) -> MotionCommand {
    if is_near_zero_motor(motor) {
        MotionCommand::Stop
    } else if motor.turn == 0.0 {
        MotionCommand::Forward {
            speed_m_s: motor.forward,
        }
    } else if motor.forward == 0.0 {
        MotionCommand::Turn {
            turn_rad_s: motor.turn,
        }
    } else {
        MotionCommand::Drive {
            forward_m_s: motor.forward,
            turn_rad_s: motor.turn,
        }
    }
}

fn apply_safe_cockpit_motor<C: Cockpit>(
    cockpit: &mut SafeCockpit<C>,
    motor: MotorCommand,
) -> Result<()> {
    if is_near_zero_motor(motor) {
        cockpit.stop().map_err(anyhow::Error::from)
    } else {
        cockpit
            .pulse_motion(
                pete_cockpit::meters_per_second_to_mm_s(motor.forward),
                pete_cockpit::radians_per_second_to_mrad_s(motor.turn),
            )
            .map_err(anyhow::Error::from)
    }
}

fn apply_slow_possession_motor<C: Cockpit>(
    cockpit: &mut SafeCockpit<C>,
    motor: MotorCommand,
) -> Result<Option<SlowPossessionMotionBlock>> {
    if is_near_zero_motor(motor) {
        cockpit.client_mut().stop().map_err(anyhow::Error::from)?;
        return Ok(None);
    }
    match cockpit
        .pulse_motion(
            pete_cockpit::meters_per_second_to_mm_s(motor.forward),
            pete_cockpit::radians_per_second_to_mrad_s(motor.turn),
        )
        .map_err(anyhow::Error::from)
    {
        Ok(()) => Ok(None),
        Err(error) if is_missed_events_error(&error) => {
            eprintln!(
                "slow possession recovered from event history gap during motion safety poll; stopping before continuing"
            );
            cockpit.client_mut().stop().map_err(anyhow::Error::from)?;
            Ok(None)
        }
        Err(error) if motion_stopped_latch(&error).is_some() => {
            let latch = motion_stopped_latch(&error);
            eprintln!("{error}; slow possession stopping before continuing");
            cockpit.client_mut().stop().map_err(anyhow::Error::from)?;
            Ok(latch.map(SlowPossessionMotionBlock::SafetyLatch))
        }
        Err(error) if command_rejection(&error).is_some() => {
            let (command_id, reason) = command_rejection(&error).expect("rejection was present");
            eprintln!("{error}; slow possession stopping before continuing");
            cockpit.client_mut().stop().map_err(anyhow::Error::from)?;
            Ok(Some(SlowPossessionMotionBlock::CommandRejected {
                command_id,
                reason,
            }))
        }
        Err(error) => Err(error),
    }
}

#[derive(Clone, Debug)]
enum SlowPossessionMotionBlock {
    SafetyLatch(SafetyLatchKind),
    CommandRejected { command_id: u32, reason: String },
}

fn is_missed_events_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<pete_cockpit::CockpitError>()
            .is_some_and(|cockpit| {
                matches!(cockpit, pete_cockpit::CockpitError::MissedEvents { .. })
            })
    })
}

fn motion_stopped_latch(error: &anyhow::Error) -> Option<SafetyLatchKind> {
    error.chain().find_map(|cause| {
        let cockpit = cause.downcast_ref::<pete_cockpit::CockpitError>()?;
        match cockpit {
            pete_cockpit::CockpitError::MotionStopped { reasons } => {
                latch_from_stop_reasons(reasons)
            }
            pete_cockpit::CockpitError::Policy(reason)
                if reason.starts_with("motion stopped by ") =>
            {
                Some(SafetyLatchKind::Heartbeat)
            }
            _ => None,
        }
    })
}

fn command_rejection(error: &anyhow::Error) -> Option<(u32, String)> {
    error.chain().find_map(|cause| {
        let cockpit = cause.downcast_ref::<pete_cockpit::CockpitError>()?;
        match cockpit {
            pete_cockpit::CockpitError::Rejected { command_id, reason } => {
                Some((*command_id, reason.clone()))
            }
            _ => None,
        }
    })
}

fn latch_from_stop_reasons(reasons: &[SafeStopReason]) -> Option<SafetyLatchKind> {
    reasons
        .iter()
        .find_map(|reason| match reason {
            SafeStopReason::SafetyTripped { latch } => *latch,
            _ => None,
        })
        .or_else(|| {
            reasons.iter().find_map(|reason| match reason {
                SafeStopReason::HeartbeatExpired => Some(SafetyLatchKind::Heartbeat),
                _ => None,
            })
        })
}

fn apply_safe_cockpit_motion<C: Cockpit>(
    cockpit: &mut SafeCockpit<C>,
    motion: &MotionCommand,
) -> Result<()> {
    apply_safe_cockpit_motor(cockpit, motion.to_motor_command())
}

pub fn body_sense_from_cockpit_status(status: StatusSummary, last_update_ms: TimeMs) -> BodySense {
    let charging = status.battery.charging_state.unwrap_or(0) != 0
        || status.battery.charging_indicator.unwrap_or(false);
    let home_base = status.battery.home_base();
    let packet_update_ms = status
        .body_packet_age_ms
        .filter(|_| status.body_packet_complete == Some(true))
        .map(|age_ms| last_update_ms.saturating_sub(u64::from(age_ms)))
        .unwrap_or(if charging { last_update_ms } else { 0 });
    BodySense {
        battery_level: status
            .battery
            .percent
            .map(|percent| percent as f32 / 100.0)
            .unwrap_or(1.0),
        charging,
        infrared_character: status.infrared_character.unwrap_or(0),
        flags: BodyFlags {
            // Create 1 can report its dock contacts as both bumpers plus all
            // four cliff bits. Packet 34 is the authoritative Home Base
            // discriminator. Keep those raw bits in Cockpit status, but do
            // not promote dock geometry into upstream collision evidence.
            bump_left: !home_base && status.contact.bump_left.unwrap_or(false),
            bump_right: !home_base && status.contact.bump_right.unwrap_or(false),
            cliff_left: !home_base && status.contact.cliff_left.unwrap_or(false),
            cliff_front_left: !home_base && status.contact.cliff_front_left.unwrap_or(false),
            cliff_front_right: !home_base && status.contact.cliff_front_right.unwrap_or(false),
            cliff_right: !home_base && status.contact.cliff_right.unwrap_or(false),
            wheel_drop: status.contact.wheel_drop.unwrap_or(false),
            wall: status.contact.wall.unwrap_or(false),
            virtual_wall: status.contact.virtual_wall.unwrap_or(false),
        },
        odometry: Pose2 {
            x_m: status.odometry.distance_mm.unwrap_or(0) as f32 / 1000.0,
            y_m: 0.0,
            heading_rad: status.odometry.heading_mrad.unwrap_or(0) as f32 / 1000.0,
        },
        last_update_ms: packet_update_ms,
        ..BodySense::default()
    }
}

fn safety_latch_kind_from_event_code(code: u32) -> Option<SafetyLatchKind> {
    match code {
        1 => Some(SafetyLatchKind::Bump),
        2 => Some(SafetyLatchKind::Cliff),
        3 => Some(SafetyLatchKind::WheelDrop),
        5 => Some(SafetyLatchKind::Heartbeat),
        6 => Some(SafetyLatchKind::Tilt),
        7 => Some(SafetyLatchKind::Impact),
        8 => Some(SafetyLatchKind::Charging),
        _ => None,
    }
}

fn infer_safety_latch_from_sensors(body: &BodySense) -> Option<SafetyLatchKind> {
    if bump_active(body) {
        Some(SafetyLatchKind::Bump)
    } else if cliff_active(body) {
        Some(SafetyLatchKind::Cliff)
    } else if body.flags.wheel_drop {
        Some(SafetyLatchKind::WheelDrop)
    } else if body.charging {
        Some(SafetyLatchKind::Charging)
    } else {
        None
    }
}

fn bump_active(body: &BodySense) -> bool {
    body.flags.bump_left || body.flags.bump_right
}

fn cliff_active(body: &BodySense) -> bool {
    body.flags.cliff_left
        || body.flags.cliff_front_left
        || body.flags.cliff_front_right
        || body.flags.cliff_right
}

fn recovery_turn_direction_for_latch(kind: SafetyLatchKind, body: &BodySense) -> TurnDir {
    match kind {
        SafetyLatchKind::Bump => contact_turn_direction(body),
        SafetyLatchKind::Cliff => {
            if body.flags.cliff_left || body.flags.cliff_front_left {
                TurnDir::Right
            } else if body.flags.cliff_right || body.flags.cliff_front_right {
                TurnDir::Left
            } else {
                TurnDir::Left
            }
        }
        _ => TurnDir::Left,
    }
}

fn contact_turn_direction(body: &BodySense) -> TurnDir {
    match (body.flags.bump_left, body.flags.bump_right) {
        (true, false) => TurnDir::Right,
        (false, true) => TurnDir::Left,
        _ => TurnDir::Left,
    }
}

fn imu_recovery_clear(status: &StatusSummary, latch: SafetyLatchKind) -> bool {
    match latch {
        SafetyLatchKind::Tilt => status
            .imu
            .tilt_magnitude_mrad
            .is_none_or(|value| value < 650),
        SafetyLatchKind::Impact => status
            .imu
            .impact_score_mm_s2
            .is_none_or(|value| value < 18_000),
        _ => false,
    }
}

fn possession_recovery_debug(
    state: &PossessionRecoveryState,
    active_latch: Option<SafetyLatchKind>,
    command_sent: bool,
) -> serde_json::Value {
    let latch = active_latch.or(state.latch);
    let intended_motion = match (state.phase, latch) {
        (PossessionRecoveryPhase::BrainstemReflex, _) => {
            serde_json::json!({"owner": "brainstem_contact_withdrawal"})
        }
        (
            PossessionRecoveryPhase::WaitingForSensorClear | PossessionRecoveryPhase::Escaping,
            Some(SafetyLatchKind::Bump | SafetyLatchKind::Cliff),
        ) => serde_json::json!({
            "linear": "reverse",
            "ttl_ms": POSSESSION_ESCAPE_TTL_MS,
        }),
        (_, Some(_)) => serde_json::json!({"linear": "stop"}),
        _ => serde_json::Value::Null,
    };
    serde_json::json!({
        "latched": latch.map(|latch| format!("{latch:?}")),
        "hazard_generation": state.hazard_generation,
        "phase": format!("{:?}", state.phase),
        "turn_direction": format!("{:?}", state.turn_direction),
        "active_since_ms": state.active_since_ms,
        "last_command_ms": state.last_command_ms,
        "command_attempts": state.command_attempts,
        "stuck_stop_sent": state.stuck_stop_sent,
        "brainstem_reflex_observed": state.brainstem_reflex_observed,
        "last_reflex_outcome": state.last_reflex_outcome.map(|outcome| format!("{outcome:?}")),
        "command_sent": command_sent,
        "intended_motion": intended_motion,
        "commanded_motion": if command_sent && state.phase == PossessionRecoveryPhase::Escaping {
            serde_json::json!({
                "linear": "reverse",
                "ttl_ms": POSSESSION_ESCAPE_TTL_MS,
            })
        } else if command_sent {
            serde_json::json!({"linear": "stop"})
        } else {
            serde_json::Value::Null
        },
        "observed_motion": {
            "linear_displacement_m": state.observed_linear_m,
            "heading_change_rad": state.observed_turn_rad,
        },
    })
}

fn motion_rejection_debug(state: &MotionRejectionState) -> serde_json::Value {
    serde_json::json!({
        "first_ms": state.first_ms,
        "last_ms": state.last_ms,
        "blocked_until_ms": state.blocked_until_ms,
        "latest_command_id": state.latest_command_id,
        "latest_reason": state.latest_reason,
        "count": state.count,
        "stuck": state.stuck,
        "stuck_stop_sent": state.stuck_stop_sent,
    })
}

fn synthetic_slow_manual_tick(
    mut now: Now,
    input: ReignInput,
    desired_motor: MotorCommand,
    final_motor: MotorCommand,
    block_reason: Option<String>,
    body_before: &pete_body::BodySense,
) -> Result<RuntimeTick> {
    let action = input.command.to_action();
    let motor_applied = !is_near_zero_motor(final_motor);
    now.extensions.insert(
        "action.motion_bridge".to_string(),
        serde_json::json!({
            "selected_action": action,
            "chosen_action": action,
            "desired_motor": desired_motor,
            "final_motor": final_motor,
            "motor_applied": motor_applied,
            "manual_hardware_gate": true,
            "body_pose_before": pose_json(body_before),
            "body_pose_after": pose_json(&now.body),
            "motion_sent_to_robot": motor_command_to_motion(final_motor),
            "motion_sent_to_sim": motor_command_to_motion(final_motor),
            "why_not_moving": block_reason,
            "runtime_bypassed": true,
        }),
    );
    let experience = Experience::new(
        "real_robot_slow_manual",
        "Direct WebRemote slow hardware command.",
        Vec::new(),
        Vec::new(),
        now.t_ms,
        now.t_ms,
    );
    Ok(RuntimeTick {
        frame: ExperienceFrame {
            id: Uuid::new_v4(),
            t_ms: now.t_ms,
            now,
            sensations: Vec::new(),
            impressions: Vec::new(),
            experiences: vec![experience.clone()],
            z: Some(ExperienceLatent::default()),
            chosen_action: action.clone(),
            conscious_command: None,
            reign_input: Some(input),
            reign_outcome: None,
            predicted_futures: Vec::new(),
            behavior_runs: Vec::new(),
            actual_next: None,
            reward: Reward::default(),
            surprise: SurpriseSense::default(),
            memory_recall: Vec::new(),
            recollections: Vec::new(),
            llm_teaching: Vec::new(),
            counterfactuals: Vec::new(),
            notes: vec!["RealSlowManualRuntimeBypass: direct hardware command".to_string()],
        },
        experience,
        chosen_action: action,
        skill_request: None,
        skill_status: None,
        recall: RecallBundle::default(),
        llm: LlmTickResult::default(),
        combobulation: None,
        inline_learning: InlineLearningTickStatus::default(),
    })
}

fn reign_input_drives_real_slow(input: &ReignInput) -> bool {
    if !matches!(input.source, ReignSource::WebRemote | ReignSource::Gamepad)
        || input.mode != ReignMode::Direct
    {
        return false;
    }
    matches!(
        input.command,
        pete_actions::ReignCommand::Go { .. }
            | pete_actions::ReignCommand::Reverse { .. }
            | pete_actions::ReignCommand::Drive { .. }
            | pete_actions::ReignCommand::Turn { .. }
            | pete_actions::ReignCommand::Stop
    )
}

fn reign_input_outputs_real_slow_directly(input: &ReignInput) -> bool {
    if input.source != ReignSource::WebRemote || input.mode != ReignMode::Direct {
        return false;
    }
    matches!(
        input.command,
        pete_actions::ReignCommand::Speak { .. } | pete_actions::ReignCommand::Chirp { .. }
    )
}

fn real_slow_body_block_reason(body: &pete_body::BodySense) -> Option<String> {
    if body.charging {
        return Some("charging active".to_string());
    }
    if body.flags.wheel_drop {
        return Some("wheel drop active".to_string());
    }
    if body.flags.cliff_left
        || body.flags.cliff_front_left
        || body.flags.cliff_front_right
        || body.flags.cliff_right
    {
        return Some("cliff sensor active".to_string());
    }
    if body.battery_level <= 0.10 && !body.charging {
        return Some("battery is critical".to_string());
    }
    None
}

fn real_slow_motor_block_reason(
    body: &pete_body::BodySense,
    tick: &RuntimeTick,
    manual_drive: bool,
    autonomous_motion: bool,
) -> Option<String> {
    if !manual_drive && !autonomous_motion {
        return Some(
            "real slow mode requires active WebRemote/Gamepad Direct command or explicit autonomous motion authorization"
                .to_string(),
        );
    }
    if let Some(reason) = real_slow_body_block_reason(body) {
        return Some(reason);
    }
    tick.frame
        .now
        .extensions
        .get("motor_gate")
        .and_then(|value| value.get("safety_reason"))
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

fn pose_json(body: &pete_body::BodySense) -> serde_json::Value {
    serde_json::json!({
        "x_m": body.odometry.x_m,
        "y_m": body.odometry.y_m,
        "heading_rad": body.odometry.heading_rad,
    })
}

fn movement_delta_m(before: &pete_body::BodySense, after: &pete_body::BodySense) -> f32 {
    distance_between_points(
        (before.odometry.x_m, before.odometry.y_m),
        (after.odometry.x_m, after.odometry.y_m),
    )
}

fn not_moving_reason(
    final_motor: MotorCommand,
    motion: &MotionCommand,
    before: &pete_body::BodySense,
    after: &pete_body::BodySense,
    movement_delta: f32,
    reset_or_dead: bool,
    tick: &RuntimeTick,
) -> Option<String> {
    if movement_delta >= 0.005 {
        return None;
    }
    if reset_or_dead {
        return Some("dead battery or stuck reset prevented motor application".to_string());
    }
    if is_near_zero_motor(final_motor) || matches!(motion, MotionCommand::Stop) {
        return Some(
            tick.frame
                .now
                .extensions
                .get("motor_gate")
                .and_then(|value| value.get("safety_reason"))
                .and_then(|value| value.as_str())
                .map(|reason| format!("final motor was stop: {reason}"))
                .unwrap_or_else(|| "final motor was stop or near zero".to_string()),
        );
    }
    if after.flags.wall || after.flags.bump_left || after.flags.bump_right {
        return Some("sim collision blocked commanded motion".to_string());
    }
    if before.odometry.x_m == after.odometry.x_m
        && before.odometry.y_m == after.odometry.y_m
        && before.odometry.heading_rad != after.odometry.heading_rad
    {
        return Some("turn-only motion changed heading without translation".to_string());
    }
    Some("non-stop motion was sent but pose delta was near zero".to_string())
}

