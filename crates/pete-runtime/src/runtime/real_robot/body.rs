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
