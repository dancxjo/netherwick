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
            physical_pose: PhysicalPoseAdapter::default(),
            brainstem_clock: BrainstemClockMapper::default(),
            imu_arbiter: ImuArbiter::default(),
            last_imu_selection: ImuSelection::default(),
        }
    }

    pub fn with_imu_override(mut self, source_override: ImuSourceOverride) -> Self {
        self.imu_arbiter.set_override(source_override);
        self
    }

    pub fn note_brainstem_reconnect(&mut self) {
        self.brainstem_clock.mark_reconnect();
        self.imu_arbiter.observe_unavailable(
            "brainstem_board_imu",
            "brainstem transport reconnected; awaiting new clock epoch",
            wall_time_ms(),
        );
        self.now_builder.clear_imu_history();
    }

    fn refresh_brainstem_observation(&mut self) -> Result<BrainstemObservation> {
        let host_request_started_ms = wall_time_ms();
        let status = self.cockpit.refresh_status()?;
        let host_response_received_ms = wall_time_ms();
        Ok(brainstem_observation_from_cockpit_status(
            status,
            StatusRequestTiming {
                host_request_started_ms,
                host_response_received_ms,
            },
            &mut self.brainstem_clock,
            &mut self.physical_pose,
        ))
    }

    fn arbitrate_imu(
        &mut self,
        observation: &BrainstemObservation,
        packets: &mut Vec<pete_sensors::SensePacket>,
        now_ms: TimeMs,
    ) {
        if let (Some(imu), Some(metadata)) =
            (observation.imu.clone(), observation.imu_metadata.clone())
        {
            self.imu_arbiter
                .observe_with_metadata(imu, metadata, now_ms);
        } else {
            self.imu_arbiter.observe_unavailable(
                "brainstem_board_imu",
                observation
                    .imu_rejection
                    .clone()
                    .unwrap_or_else(|| "brainstem IMU unavailable".to_string()),
                now_ms,
            );
        }
        self.last_imu_selection = self.imu_arbiter.arbitrate_packets(packets, now_ms);
        if observation.clock_epoch_changed || self.last_imu_selection.source_changed {
            self.now_builder.clear_imu_history();
        }
    }

    fn insert_imu_selection(&self, now: &mut Now) {
        now.extensions.insert(
            "sensor.imu_selection".to_string(),
            self.last_imu_selection.diagnostics.clone(),
        );
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

        let observation = self.refresh_brainstem_observation()?;
        let body = observation.body.clone();
        let brainstem_events = self.cockpit.poll_events()?;
        let mut packets = poll_sensors_lossy(
            &mut self.sensors,
            &mut self.sensor_poll_health,
            imu_motion_context(&body),
        )
        .await;
        let t_ms = body.last_update_ms.max(wall_time_ms());
        self.frame_processor.process_packets(t_ms, &mut packets);
        self.arbitrate_imu(&observation, &mut packets, t_ms);
        let mut now = self.now_builder.build(t_ms, body, packets)?;
        insert_sensor_health(&mut now, &self.sensor_poll_health);
        self.insert_imu_selection(&mut now);
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

        let observation = self.refresh_brainstem_observation()?;
        let status_before = &observation.status;
        let body_before = observation.body.clone();
        let recovery_decision =
            self.apply_possession_recovery(&body_before, &brainstem_events, status_before)?;
        let mut packets = poll_sensors_lossy(
            &mut self.sensors,
            &mut self.sensor_poll_health,
            imu_motion_context(&body_before),
        )
        .await;
        let t_ms = body_before.last_update_ms.max(wall_time_ms());
        self.frame_processor.process_packets(t_ms, &mut packets);
        self.arbitrate_imu(&observation, &mut packets, t_ms);
        let mut now = self.now_builder.build(t_ms, body_before.clone(), packets)?;
        insert_sensor_health(&mut now, &self.sensor_poll_health);
        self.insert_imu_selection(&mut now);
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
                    status_before,
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
