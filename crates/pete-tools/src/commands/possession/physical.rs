fn requested_robot_sensor_count(args: &RobotArgs) -> usize {
    usize::from(args.camera.is_some() || args.kinect_depth)
        + usize::from(args.mic.is_some())
        + usize::from(args.imu.is_some())
        + usize::from(args.gps.is_some())
}

fn establish_create_sensor_stream(
    cockpit: &mut dyn Cockpit,
    require_new_complete_packet: bool,
) -> Result<()> {
    cockpit
        .stop()
        .context("failed to establish stopped state before Create sensor streaming")?;
    let baseline_count = cockpit
        .get_status()
        .context("failed to read pre-stream Create packet counter")?
        .summary()
        .body_packet_count;
    cockpit
        .stream_sensors(
            true,
            CREATE_SENSOR_STREAM_PACKET_ID,
            CREATE_SENSOR_STREAM_PERIOD_MS,
        )
        .context("failed to establish the production Create sensor stream")?;
    if !require_new_complete_packet {
        return Ok(());
    }

    let deadline = Instant::now() + Duration::from_millis(CREATE_SENSOR_READY_TIMEOUT_MS);
    loop {
        let status = cockpit
            .get_status()
            .context("failed while waiting for fresh Create body telemetry")?
            .summary();
        let count_advanced = match (baseline_count, status.body_packet_count) {
            (Some(before), Some(after)) => after != before,
            (None, Some(after)) => after > 0,
            _ => false,
        };
        if count_advanced
            && status.has_fresh_complete_body_packet(CREATE_SENSOR_FRESHNESS_MAX_AGE_MS)
        {
            return Ok(());
        }
        if Instant::now() >= deadline {
            anyhow::bail!(
                "Create sensor stream did not produce a new complete packet within {} ms (before={baseline_count:?}, after={:?}, age_ms={:?}, complete={:?})",
                CREATE_SENSOR_READY_TIMEOUT_MS,
                status.body_packet_count,
                status.body_packet_age_ms,
                status.body_packet_complete,
            );
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

// Ownership may span comparatively expensive perception/runtime ticks. Wheel
// motion remains independently bounded by the 300 ms command TTL and 750 ms
// heartbeat stop, so use the firmware's maximum lease and renew it proactively.
async fn run_physical_possession_recovery_smoke(cockpit: Box<dyn Cockpit + Send>) -> Result<()> {
    let mut cockpit = SafeCockpit::new(cockpit);
    let result = run_physical_possession_recovery_smoke_inner(&mut cockpit).await;
    let stop_result = cockpit
        .client_mut()
        .stop()
        .context("recovery smoke final STOP was not acknowledged");
    let exorcize_result = cockpit
        .client_mut()
        .exorcize()
        .context("recovery smoke could not surrender possession");
    result?;
    stop_result?;
    exorcize_result?;
    println!("physical possession recovery smoke complete: stopped and exorcized");
    Ok(())
}

async fn run_physical_orientation_probe(cockpit: Box<dyn Cockpit + Send>) -> Result<()> {
    let mut cockpit = SafeCockpit::new(cockpit);
    let result = run_physical_orientation_probe_inner(&mut cockpit).await;
    let stop_result = cockpit
        .client_mut()
        .stop()
        .context("orientation probe final STOP was not acknowledged");
    let exorcize_result = cockpit
        .client_mut()
        .exorcize()
        .context("orientation probe could not surrender possession");
    result?;
    stop_result?;
    exorcize_result?;
    println!("orientation probe complete: stopped and exorcized");
    Ok(())
}

async fn run_physical_orientation_probe_inner(
    cockpit: &mut SafeCockpit<Box<dyn Cockpit + Send>>,
) -> Result<()> {
    cockpit.client_mut().stop()?;
    cockpit.resync_event_cursor_from_status()?;
    let before = cockpit.refresh_status()?;
    ensure_orientation_probe_safe(&before, "before orientation probe")?;
    println!(
        "orientation probe before: tilt={} mrad accel={} mm/s^2 rough={} impact={} odom={} mm heading={} mrad calibration={}",
        before.imu.tilt_magnitude_mrad.unwrap_or_default(),
        before.imu.accel_magnitude_mm_s2.unwrap_or_default(),
        before.imu.roughness_mm_s2.unwrap_or_default(),
        before.imu.impact_score_mm_s2.unwrap_or_default(),
        before.odometry.distance_mm.unwrap_or_default(),
        before.odometry.heading_mrad.unwrap_or_default(),
        before.imu.calibration.as_deref().unwrap_or("unknown"),
    );

    cockpit.client_mut().orientation_probe(250, 400)?;
    tokio::time::sleep(Duration::from_millis(650)).await;
    cockpit.client_mut().stop()?;
    tokio::time::sleep(Duration::from_millis(150)).await;
    let end = cockpit.refresh_status()?;
    ensure_orientation_probe_safe(&end, "after orientation probe")?;

    let heading_delta = end.odometry.heading_mrad.unwrap_or_default();
    let yaw_delta = end.imu.yaw_mrad.unwrap_or_default();
    let distance_delta = end.odometry.distance_mm.unwrap_or_default();
    println!(
        "orientation probe firmware spin: heading_delta={} mrad yaw_delta={} mrad distance_delta={} mm yaw_rate={} mrad/s gyro=({},{},{}) mrad/s tilt={} mrad calibration={} motion_consistency={}",
        heading_delta,
        yaw_delta,
        distance_delta,
        end.imu.yaw_rate_mrad_s.unwrap_or_default(),
        end.imu.angular_velocity_mrad_s.x.unwrap_or_default(),
        end.imu.angular_velocity_mrad_s.y.unwrap_or_default(),
        end.imu.angular_velocity_mrad_s.z.unwrap_or_default(),
        end.imu.tilt_magnitude_mrad.unwrap_or_default(),
        end.imu.calibration.as_deref().unwrap_or("unknown"),
        end.imu.motion_consistency.as_deref().unwrap_or("unknown"),
    );
    if heading_delta.abs() < 20 && yaw_delta.abs() < 20 {
        println!("orientation probe warning: spin pulse produced little or no heading/yaw change; ground contact, wheel slip, or Create drive response is uncertain");
    }
    if distance_delta.abs() > 20 {
        println!("orientation probe warning: spin pulse produced translational odometry; wheel slip or uneven ground is possible");
    }
    Ok(())
}

fn ensure_orientation_probe_safe(status: &pete_cockpit::StatusSummary, phase: &str) -> Result<()> {
    if !status.has_fresh_complete_body_packet(CREATE_SENSOR_FRESHNESS_MAX_AGE_MS) {
        anyhow::bail!("{phase}: no fresh complete Create body packet");
    }
    if status.battery.charging_state.unwrap_or(0) != 0
        || status.battery.charging_indicator.unwrap_or(false)
    {
        anyhow::bail!("{phase}: charging is active");
    }
    if status.contact.wheel_drop.unwrap_or(false) {
        anyhow::bail!("{phase}: wheel drop is active");
    }
    if status.contact.any_safety_stop() == Some(true) {
        anyhow::bail!("{phase}: cliff or wheel-drop safety sensor is active");
    }
    if status.imu.health.as_deref() != Some("1") && status.imu.health.as_deref() != Some("ok") {
        anyhow::bail!(
            "{phase}: IMU health is {}",
            status.imu.health.as_deref().unwrap_or("unknown")
        );
    }
    if status.imu.sample_age_ms.is_some_and(|age| age > 100) {
        anyhow::bail!("{phase}: IMU sample is stale");
    }
    if status
        .imu
        .impact_score_mm_s2
        .is_some_and(|impact| impact >= 18_000)
    {
        anyhow::bail!("{phase}: IMU impact score is high");
    }
    Ok(())
}

async fn run_physical_possession_recovery_smoke_inner(
    cockpit: &mut SafeCockpit<Box<dyn Cockpit + Send>>,
) -> Result<()> {
    cockpit.client_mut().stop()?;
    let initial_status = cockpit.resync_event_cursor_from_status()?;
    ensure_recovery_smoke_ready(&initial_status)?;
    println!(
        "recovery smoke armed: wheels must remain off the floor; press and hold either bumper until contact is acknowledged"
    );

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut next_motion_at = Instant::now();
    let mut saw_safety_trip = false;
    let mut saw_motion_stop = false;
    let mut saw_recovery_estop = false;
    let contacted_body = loop {
        if Instant::now() >= deadline {
            anyhow::bail!(
                "recovery smoke timed out waiting for live bump telemetry and safety-stop events"
            );
        }
        let status = cockpit.refresh_status()?;
        let body =
            body_sense_from_cockpit_status(status, Utc::now().timestamp_millis().max(0) as u64);
        let events = cockpit.poll_events()?;
        saw_safety_trip |= events
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::SafetyTripped);
        saw_motion_stop |= events
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::MotionStopped);
        saw_recovery_estop |= events
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::EStopLatched);
        if body.flags.bump_left || body.flags.bump_right {
            if saw_safety_trip && saw_motion_stop {
                break body;
            }
        } else if Instant::now() >= next_motion_at {
            // Keep a bounded motion active so the observed bump proves that
            // firmware interruption, not mere stationary telemetry, occurred.
            cockpit.client_mut().cmd_vel(25, 0, 300)?;
            next_motion_at = Instant::now() + Duration::from_millis(150);
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    };

    let expected_turn = if contacted_body.flags.bump_left {
        TurnDir::Right
    } else {
        TurnDir::Left
    };
    let mut conductor = SimpleConductor::default();
    let first = conductor.choose(recovery_smoke_input(contacted_body))?;
    if !matches!(
        first,
        ActionPrimitive::Go {
            intensity,
            duration_ms: 300
        } if intensity < 0.0
    ) {
        anyhow::bail!("contact did not enter conductor reverse recovery: {first:?}");
    }
    println!("contact observed; brainstem stopped motion; release the bumper");

    let clear_deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if Instant::now() >= clear_deadline {
            anyhow::bail!("recovery smoke timed out waiting for bumper and safety latch to clear");
        }
        let status = cockpit.refresh_status()?;
        let body =
            body_sense_from_cockpit_status(status, Utc::now().timestamp_millis().max(0) as u64);
        let events = cockpit.poll_events()?;
        saw_recovery_estop |= events
            .events
            .iter()
            .any(|event| event.kind == CockpitEventKind::EStopLatched);
        if !body.flags.bump_left && !body.flags.bump_right {
            clear_bump_recovery_latches(cockpit, saw_recovery_estop)?;
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    let mut saw_reverse = false;
    let mut saw_turn = false;
    let mut saw_probe = false;
    let mut saw_inspect = false;
    for _ in 0..20 {
        let status = cockpit.refresh_status()?;
        let body =
            body_sense_from_cockpit_status(status, Utc::now().timestamp_millis().max(0) as u64);
        let action = conductor.choose(recovery_smoke_input(body))?;
        match &action {
            ActionPrimitive::Go { intensity, .. } if *intensity < 0.0 => saw_reverse = true,
            ActionPrimitive::Turn { direction, .. } => {
                if *direction != expected_turn {
                    anyhow::bail!(
                        "conductor turned {direction:?} after contact; expected {expected_turn:?}"
                    );
                }
                saw_turn = true;
            }
            ActionPrimitive::Go { intensity, .. } if *intensity > 0.0 => saw_probe = true,
            ActionPrimitive::Inspect { .. } => {
                saw_inspect = true;
                cockpit.client_mut().stop()?;
                break;
            }
            other => anyhow::bail!("unexpected recovery action during physical smoke: {other:?}"),
        }
        let motor = pete_actions::action_to_motor_command(Some(&action));
        cockpit.pulse_motion(
            pete_cockpit::meters_per_second_to_mm_s(motor.forward),
            pete_cockpit::radians_per_second_to_mrad_s(motor.turn),
        )?;
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    if !(saw_reverse && saw_turn && saw_probe && saw_inspect) {
        anyhow::bail!(
            "incomplete physical recovery sequence: reverse={saw_reverse} turn={saw_turn} probe={saw_probe} inspect={saw_inspect}"
        );
    }
    println!(
        "verified live sequence: contact -> stop -> clear -> reverse -> turn -> probe -> inspect"
    );
    Ok(())
}

fn ensure_recovery_smoke_ready(status: &pete_cockpit::StatusSummary) -> Result<()> {
    if status.estop_latched == Some(true) || status.safety_tripped == Some(true) {
        anyhow::bail!("recovery smoke requires an initially clear e-stop and safety latch");
    }
    if status.contact.any_contact() == Some(true) {
        anyhow::bail!(
            "recovery smoke requires the bumper and contact sensors to be clear initially"
        );
    }
    if status.contact.any_safety_stop() == Some(true) {
        anyhow::bail!("recovery smoke cannot run while a cliff or wheel-drop sensor is active");
    }
    if status.battery.charging_state.unwrap_or(0) != 0
        || status.battery.charging_indicator.unwrap_or(false)
    {
        anyhow::bail!("recovery smoke cannot run while charging is active");
    }
    Ok(())
}

/// Clear only the latches created by this explicitly guarded bump smoke after
/// telemetry proves contact is gone. An e-stop that predates the smoke is left
/// for an operator rather than being treated as a recoverable bump side effect.
fn clear_bump_recovery_latches<C: Cockpit>(
    cockpit: &mut SafeCockpit<C>,
    saw_recovery_estop: bool,
) -> Result<()> {
    let status = cockpit.refresh_status()?;
    if status.contact.bump_left == Some(true) || status.contact.bump_right == Some(true) {
        anyhow::bail!("refusing to clear recovery latches while a bumper is still pressed");
    }
    if status.contact.any_safety_stop() == Some(true) {
        anyhow::bail!(
            "refusing to clear recovery latches while a cliff or wheel-drop sensor is active"
        );
    }
    if status.battery.charging_state.unwrap_or(0) != 0
        || status.battery.charging_indicator.unwrap_or(false)
    {
        anyhow::bail!("refusing to clear recovery latches while charging is active");
    }
    if let Some(kind) = status.safety_latch_kind {
        if kind != SafetyLatchKind::Bump {
            anyhow::bail!("refusing to clear non-bump safety latch during recovery: {kind:?}");
        }
    }
    if status.estop_latched == Some(true) {
        if !saw_recovery_estop {
            anyhow::bail!(
                "e-stop was already latched before bump recovery; leave it latched for operator clearance"
            );
        }
        cockpit
            .client_mut()
            .clear_estop()
            .context("recovery bump e-stop could not be cleared after contact release")?;
    }

    let status = cockpit.refresh_status()?;
    if status.safety_tripped == Some(true)
        || status.safety_latch_kind == Some(SafetyLatchKind::Bump)
    {
        cockpit
            .client_mut()
            .clear_safety_latch(SafetyLatchKind::Bump)
            .context("recovery bump safety latch could not be cleared after contact release")?;
    }

    let status = cockpit.refresh_status()?;
    if status.estop_latched == Some(true) || status.safety_tripped == Some(true) {
        anyhow::bail!("recovery latches remain set after bumper release");
    }
    Ok(())
}

fn recovery_smoke_input(body: BodySense) -> ConductorInput {
    ConductorInput {
        latent: Default::default(),
        drives: Default::default(),
        memory: Default::default(),
        predictions: Default::default(),
        surprise: Default::default(),
        llm: Default::default(),
        safety: Default::default(),
        reign: Default::default(),
        range: Default::default(),
        body,
        charger_near_score: 0.0,
        charger_visible_score: 0.0,
        proposals: Vec::new(),
    }
}
