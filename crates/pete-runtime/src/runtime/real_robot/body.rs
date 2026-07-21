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
    static HOST_CLOCK: std::sync::OnceLock<(std::time::Instant, TimeMs)> =
        std::sync::OnceLock::new();
    let (monotonic_anchor, unix_anchor_ms) = HOST_CLOCK.get_or_init(|| {
        let unix_anchor_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);
        (std::time::Instant::now(), unix_anchor_ms)
    });
    unix_anchor_ms.saturating_add(
        monotonic_anchor
            .elapsed()
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX),
    )
}

const BRAINSTEM_CLOCK_HALF_RANGE: u32 = 1 << 31;
const BRAINSTEM_MAX_STATUS_RTT_MS: u64 = 200;
const STANDARD_GRAVITY_MM_S2: f32 = 9_806.65;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StatusRequestTiming {
    pub host_request_started_ms: TimeMs,
    pub host_response_received_ms: TimeMs,
}

#[derive(Clone, Copy, Debug, Default)]
struct BrainstemClockEstimate {
    uptime_ms: u32,
    host_midpoint_ms: TimeMs,
    confidence: f32,
    source_epoch: u64,
    epoch_changed: bool,
}

#[derive(Clone, Debug, Default)]
pub struct BrainstemClockMapper {
    last_uptime_ms: Option<u32>,
    last_host_midpoint_ms: Option<TimeMs>,
    firmware_epoch: Option<u32>,
    source_epoch: u64,
    reconnect_pending: bool,
}

impl BrainstemClockMapper {
    pub fn mark_reconnect(&mut self) {
        self.reconnect_pending = true;
    }

    fn observe_status(
        &mut self,
        status: &StatusSummary,
        timing: StatusRequestTiming,
    ) -> Result<BrainstemClockEstimate, String> {
        let uptime_ms = status
            .uptime_ms
            .ok_or_else(|| "brainstem uptime is unavailable".to_string())?;
        if timing.host_response_received_ms < timing.host_request_started_ms {
            return Err("host status timing regressed".to_string());
        }
        let round_trip_ms = timing
            .host_response_received_ms
            .saturating_sub(timing.host_request_started_ms);
        let host_midpoint_ms = timing
            .host_request_started_ms
            .saturating_add(round_trip_ms / 2);
        if self
            .last_host_midpoint_ms
            .is_some_and(|last| host_midpoint_ms < last)
        {
            return Err("out-of-order status response rejected".to_string());
        }

        let firmware_epoch_changed = matches!(
            (self.firmware_epoch, status.clock_epoch),
            (Some(previous), Some(current)) if previous != current
        );
        let uptime_regressed = self.last_uptime_ms.is_some_and(|last| {
            uptime_ms < last && last.wrapping_sub(uptime_ms) < BRAINSTEM_CLOCK_HALF_RANGE
        });
        // A large numeric regression is normal 32-bit wrap; a small regression is
        // a reboot because status requests are serialized at this boundary.
        let epoch_changed = self.reconnect_pending || firmware_epoch_changed || uptime_regressed;
        if epoch_changed {
            self.source_epoch = self.source_epoch.saturating_add(1);
        }
        self.reconnect_pending = false;
        self.last_uptime_ms = Some(uptime_ms);
        self.last_host_midpoint_ms = Some(host_midpoint_ms);
        self.firmware_epoch = status.clock_epoch.or(self.firmware_epoch);

        let confidence = if round_trip_ms <= 50 {
            0.95
        } else if round_trip_ms <= BRAINSTEM_MAX_STATUS_RTT_MS {
            0.70
        } else {
            0.20
        };
        Ok(BrainstemClockEstimate {
            uptime_ms,
            host_midpoint_ms,
            confidence,
            source_epoch: self.source_epoch,
            epoch_changed,
        })
    }
}

impl BrainstemClockEstimate {
    fn map_exact(self, brainstem_timestamp_ms: u32) -> Result<TimeMs, String> {
        let age_ms = self.uptime_ms.wrapping_sub(brainstem_timestamp_ms);
        if age_ms >= BRAINSTEM_CLOCK_HALF_RANGE {
            return Err("brainstem sensor timestamp is in the future".to_string());
        }
        self.host_midpoint_ms
            .checked_sub(u64::from(age_ms))
            .ok_or_else(|| "brainstem timestamp predates host clock origin".to_string())
    }
}

#[derive(Clone, Debug)]
pub struct BrainstemObservation {
    pub status: StatusSummary,
    pub body: BodySense,
    pub imu: Option<ImuSense>,
    pub imu_metadata: Option<ImuCandidateMetadata>,
    pub imu_rejection: Option<String>,
    pub clock_epoch_changed: bool,
}

pub fn brainstem_observation_from_cockpit_status(
    status: StatusSummary,
    timing: StatusRequestTiming,
    mapper: &mut BrainstemClockMapper,
    adapter: &mut PhysicalPoseAdapter,
) -> BrainstemObservation {
    let clock = mapper.observe_status(&status, timing);
    let exact_body_timestamp = clock.as_ref().ok().and_then(|estimate| {
        status
            .body_packet_timestamp_ms
            .and_then(|timestamp| estimate.map_exact(timestamp).ok())
    });
    let (imu, imu_metadata, imu_rejection, epoch_changed) = match &clock {
        Ok(estimate) => {
            match brainstem_imu_sense(&status, Some(*estimate), timing.host_response_received_ms) {
                Ok((imu, metadata)) => {
                    (Some(imu), Some(metadata), None, estimate.epoch_changed)
                }
                Err(reason) => (None, None, Some(reason), estimate.epoch_changed),
            }
        }
        Err(_) if status.uptime_ms.is_none() => {
            match brainstem_imu_sense(&status, None, timing.host_response_received_ms) {
                Ok((imu, metadata)) => (Some(imu), Some(metadata), None, false),
                Err(fallback_reason) => (None, None, Some(fallback_reason), false),
            }
        }
        Err(reason) => (None, None, Some(reason.clone()), false),
    };
    let pose = adapter.observe(&status);
    let mut body =
        body_sense_from_cockpit_status_and_pose(&status, timing.host_response_received_ms, pose);
    if status_body_timestamp_is_usable(&body, exact_body_timestamp) {
        body.last_update_ms = exact_body_timestamp.unwrap_or(body.last_update_ms);
    }
    BrainstemObservation {
        status,
        body,
        imu,
        imu_metadata,
        imu_rejection,
        clock_epoch_changed: epoch_changed,
    }
}

fn status_body_timestamp_is_usable(body: &BodySense, timestamp: Option<TimeMs>) -> bool {
    body.last_update_ms > 0 && timestamp.is_some()
}

fn brainstem_imu_sense(
    status: &StatusSummary,
    clock: Option<BrainstemClockEstimate>,
    host_received_ms: TimeMs,
) -> Result<(ImuSense, ImuCandidateMetadata), String> {
    let imu = &status.imu;
    let present = imu
        .present
        .as_deref()
        .is_some_and(|value| matches!(value, "1" | "2" | "true" | "on" | "present"));
    if !present || imu.sample_count.unwrap_or(0) == 0 {
        return Err("brainstem IMU is absent or has no samples".to_string());
    }
    let health = imu
        .health
        .as_deref()
        .ok_or_else(|| "brainstem IMU health is missing".to_string())?;
    let healthy = matches!(health, "1" | "ok");
    let roll = imu
        .roll_mrad
        .ok_or_else(|| "brainstem IMU roll is missing".to_string())? as f32
        / 1_000.0;
    let pitch = imu
        .pitch_mrad
        .ok_or_else(|| "brainstem IMU pitch is missing".to_string())? as f32
        / 1_000.0;
    let gyro =
        complete_axes(&imu.angular_velocity_mrad_s, "gyro")?.map(|value| value as f32 / 1_000.0);
    let acceleration = complete_axes(&imu.linear_acceleration_mm_s2, "acceleration")?
        .map(|value| value as f32 / STANDARD_GRAVITY_MM_S2);
    let (captured_at_ms, clock_confidence, clock_source, source_epoch) =
        match (imu.sample_timestamp_ms, clock) {
        (Some(timestamp_ms), Some(clock)) => (
            clock.map_exact(timestamp_ms)?,
            clock.confidence,
            "brainstem_exact_midpoint".to_string(),
            clock.source_epoch,
        ),
        _ => {
            let age_ms = imu
                .sample_age_ms
                .ok_or_else(|| "brainstem IMU has no exact timestamp or sample age".to_string())?;
            (
                host_received_ms.saturating_sub(u64::from(age_ms)),
                clock.map_or(0.25, |clock| clock.confidence.min(0.35)),
                "brainstem_sample_age_fallback".to_string(),
                clock.map_or(0, |clock| clock.source_epoch),
            )
        }
    };
    let orientation_source = imu
        .orientation_source
        .clone()
        .ok_or_else(|| "brainstem IMU orientation provenance is missing".to_string())?;
    let source_id = "brainstem_board_imu";
    let orientation_confidence = imu
        .orientation_confidence_permille
        .ok_or_else(|| "brainstem IMU orientation confidence is missing".to_string())?;
    let gyro_bias_calibrated = imu
        .gyro_bias_calibrated
        .ok_or_else(|| "brainstem IMU gyro-bias state is missing".to_string())?;
    let mounting_calibrated = imu
        .mounting_calibrated
        .ok_or_else(|| "brainstem IMU mounting state is missing".to_string())?;
    let imu_sense = ImuSense {
        schema_version: 2,
        captured_at_ms,
        // MPU-6050 yaw is gyro-integrated and is deliberately not published as
        // absolute orientation. Wheel odometry remains planar-heading authority.
        orientation: vec![roll, pitch],
        acceleration: acceleration.to_vec(),
        angular_velocity: gyro.to_vec(),
        orientation_confidence: f32::from(orientation_confidence).clamp(0.0, 1_000.0) / 1_000.0,
        gyro_bias_calibrated,
        mounting_calibrated,
        orientation_source: Some(format!(
            "{source_id}@{}:{orientation_source}",
            source_epoch
        )),
    };
    Ok((
        imu_sense,
        ImuCandidateMetadata {
            source_id: source_id.to_string(),
            provenance: orientation_source,
            healthy,
            clock_confidence,
            clock_source: Some(clock_source),
            source_epoch,
            supported_axes: vec![
                "roll".to_string(),
                "pitch".to_string(),
                "gyro_xyz".to_string(),
                "accel_xyz".to_string(),
            ],
        },
    ))
}

fn complete_axes(summary: &pete_cockpit::Axis3Summary, name: &str) -> Result<[i32; 3], String> {
    match (summary.x, summary.y, summary.z) {
        (Some(x), Some(y), Some(z)) => Ok([x, y, z]),
        _ => Err(format!("brainstem IMU {name} axes are incomplete")),
    }
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
    let pose = Pose2 {
        x_m: status.odometry.x_mm.unwrap_or(0) as f32 / 1000.0,
        y_m: status.odometry.y_mm.unwrap_or(0) as f32 / 1000.0,
        heading_rad: status.odometry.heading_mrad.unwrap_or(0) as f32 / 1000.0,
    };
    body_sense_from_cockpit_status_and_pose(&status, last_update_ms, pose)
}

pub fn body_sense_from_cockpit_status_with_pose_adapter(
    status: StatusSummary,
    last_update_ms: TimeMs,
    adapter: &mut PhysicalPoseAdapter,
) -> BodySense {
    let pose = adapter.observe(&status);
    body_sense_from_cockpit_status_and_pose(&status, last_update_ms, pose)
}

impl PhysicalPoseAdapter {
    pub fn observe(&mut self, status: &StatusSummary) -> Pose2 {
        let distance_mm = status.odometry.distance_mm;
        let heading_mrad = status.odometry.heading_mrad;
        let reset_changed = matches!(
            (self.last_reset_count, status.odometry.reset_count),
            (Some(previous), Some(current)) if previous != current
        );

        if reset_changed {
            self.pose = Pose2::default();
            self.last_distance_mm = None;
            self.last_heading_mrad = None;
        }

        if let (Some(x_mm), Some(y_mm)) = (status.odometry.x_mm, status.odometry.y_mm) {
            self.pose = Pose2 {
                x_m: x_mm as f32 / 1000.0,
                y_m: y_mm as f32 / 1000.0,
                heading_rad: heading_mrad.unwrap_or(0) as f32 / 1000.0,
            };
        } else if let (Some(distance_mm), Some(heading_mrad)) = (distance_mm, heading_mrad) {
            if let (Some(previous_distance_mm), Some(previous_heading_mrad)) =
                (self.last_distance_mm, self.last_heading_mrad)
            {
                let distance_delta_m =
                    distance_mm.wrapping_sub(previous_distance_mm) as f32 / 1000.0;
                let heading_delta_rad = normalize_pose_angle(
                    heading_mrad.wrapping_sub(previous_heading_mrad) as f32 / 1000.0,
                );
                let midpoint_heading = self.pose.heading_rad + heading_delta_rad * 0.5;
                self.pose.x_m += distance_delta_m * midpoint_heading.cos();
                self.pose.y_m += distance_delta_m * midpoint_heading.sin();
                self.pose.heading_rad =
                    normalize_pose_angle(self.pose.heading_rad + heading_delta_rad);
            } else {
                // A cumulative legacy stream has no recoverable pre-connection
                // translation history. Establish a truthful local origin and
                // integrate only observed deltas from this sample onward.
                self.pose.heading_rad = heading_mrad as f32 / 1000.0;
            }
        }

        self.last_distance_mm = distance_mm;
        self.last_heading_mrad = heading_mrad;
        self.last_reset_count = status.odometry.reset_count;
        self.pose
    }
}

fn normalize_pose_angle(mut angle: f32) -> f32 {
    while angle > std::f32::consts::PI {
        angle -= std::f32::consts::TAU;
    }
    while angle < -std::f32::consts::PI {
        angle += std::f32::consts::TAU;
    }
    angle
}

fn body_sense_from_cockpit_status_and_pose(
    status: &StatusSummary,
    last_update_ms: TimeMs,
    pose: Pose2,
) -> BodySense {
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
        odometry: pose,
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
