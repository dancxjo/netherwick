async fn run_robot(args: RobotArgs) -> Result<()> {
    let env_report = collect_hardware_env_report().await;
    let lidar_device = selected_lidar_device(
        args.lidar.as_deref(),
        args.cockpit == CockpitBackendArg::Sim,
        &env_report,
    );
    let create_port = selected_cockpit_endpoint(
        args.cockpit,
        &args.create_port,
        &args.brainstem_host,
        args.brainstem_local,
        &env_report,
        lidar_device.as_deref(),
    );
    let robot_mode = match args.mode {
        RobotModeArg::ReadOnly => RobotMode::ReadOnly,
        RobotModeArg::Regular => RobotMode::Slow,
        RobotModeArg::PossessionSlow => RobotMode::Slow,
        RobotModeArg::Disabled => RobotMode::Disabled,
    };
    if robot_mode == RobotMode::Disabled {
        anyhow::bail!("--mode disabled does not start the real robot runner");
    }

    let (mut cockpit, robot_mode, is_mock_body) = open_robot_cockpit_or_fallback(
        args.cockpit,
        create_port.as_deref(),
        robot_mode,
        args.brainstem_device_id.as_deref(),
        args.brainstem_boot_id.as_deref(),
        args.max_linear_mm_s,
        args.max_angular_mrad_s,
    )?;
    let brainstem_capabilities = cockpit
        .get_capabilities()
        .context("failed to read the brainstem capability contract")?;
    establish_create_sensor_stream(cockpit.as_mut(), !is_mock_body)?;
    if args.recovery_smoke {
        if robot_mode != RobotMode::Slow {
            anyhow::bail!("--recovery-smoke requires --mode regular");
        }
        if is_mock_body || args.cockpit == CockpitBackendArg::Sim {
            anyhow::bail!("--recovery-smoke requires a physical brainstem");
        }
        if !args.wheels_off_floor {
            anyhow::bail!("--recovery-smoke requires --wheels-off-floor");
        }
        return run_physical_possession_recovery_smoke(cockpit).await;
    }
    if args.orientation_probe {
        if robot_mode != RobotMode::Slow {
            anyhow::bail!("--orientation-probe requires --mode regular");
        }
        if is_mock_body || args.cockpit == CockpitBackendArg::Sim {
            anyhow::bail!("--orientation-probe requires a physical brainstem");
        }
        return run_physical_orientation_probe(cockpit).await;
    }

    let reign_queue = std::sync::Arc::new(std::sync::Mutex::new(ReignQueue::default()));
    let live_state = args.dashboard.map(|_| {
        let live_state = if robot_mode == RobotMode::Slow {
            LiveViewState::new().with_real_slow_hardware_control()
        } else {
            LiveViewState::new()
        };
        live_state.update_session(SceneSession {
            mode: match robot_mode {
                RobotMode::ReadOnly => "read-only".to_string(),
                RobotMode::Slow => "regular".to_string(),
                RobotMode::Disabled => "disabled".to_string(),
            },
            scenario: None,
            seed: None,
            source: "real_robot".to_string(),
            tick_ms: Some(args.tick_ms),
        });
        live_state.update_scene_metadata(LiveSceneMetadata {
            arena: None,
            objects: Vec::new(),
            sensor_calibration: Some(real_robot_depth_calibration_from_env()),
        });
        live_state
    });

    if let (Some(addr), Some(live_state)) = (args.dashboard, live_state.clone()) {
        let server_state = live_state.clone();
        let reign_state =
            pete_server::ReignServerState::with_live_view(reign_queue.clone(), &live_state);
        if args.dashboard_tls {
            let cert_path = args.dashboard_tls_cert.clone();
            let key_path = args.dashboard_tls_key.clone();
            tokio::spawn(async move {
                if let Err(error) = pete_server::serve_live_view_with_reign_tls(
                    addr,
                    server_state,
                    reign_state,
                    cert_path,
                    key_path,
                )
                .await
                {
                    eprintln!("live robot HTTPS view server stopped: {error}");
                }
            });
        } else {
            tokio::spawn(async move {
                if let Err(error) =
                    pete_server::serve_live_view_with_reign(addr, server_state, reign_state).await
                {
                    eprintln!("live robot view server stopped: {error}");
                }
            });
        }
        let scheme = if args.dashboard_tls { "https" } else { "http" };
        println!("robot {:?} dashboard: {scheme}://{addr}/view", robot_mode);
    }

    let mut sensors: Vec<Box<dyn SenseProducer + Send>> = Vec::new();
    let lidar_extrinsics = lidar_extrinsics(
        args.lidar_forward_m,
        args.lidar_left_m,
        args.lidar_height_m,
        args.lidar_roll_deg,
        args.lidar_pitch_deg,
        args.lidar_yaw_deg,
    );

    if args.kinect_depth {
        if let Some(device) = &args.camera {
            println!(
                "Kinect depth enabled; using libfreenect for Kinect RGB/depth and not opening {device} through V4L"
            );
        }
    } else if let Some(device) = &args.camera {
        match CameraSenseProvider::new(device) {
            Ok(provider) => {
                let live_state_for_camera = live_state.clone();
                sensors.push(Box::new(BackgroundSenseProducer::spawn_with_callback(
                    "camera",
                    provider,
                    Duration::from_millis(33),
                    move |packet| {
                        if let (Some(live_state), SensePacket::EyeFrame(frame)) =
                            (&live_state_for_camera, packet)
                        {
                            live_state.record_live_eye_frame(frame.clone());
                            publish_live_sensor_only_snapshot(live_state, packet);
                        }
                    },
                )));
            }
            Err(err) => {
                if args.require_camera {
                    anyhow::bail!("failed to initialize camera: {err}");
                } else {
                    println!("failed to initialize camera: {err}; continuing without it");
                }
            }
        }
    }

    if args.kinect_depth {
        #[cfg(feature = "kinect-freenect")]
        match FreenectKinectProvider::with_index(args.kinect_index)
            .map(|provider| provider.with_rgb_adjustment(kinect_rgb_adjustment_for_robot(&args)))
        {
            Ok(provider) => {
                let live_state_for_kinect = live_state.clone();
                sensors.push(Box::new(BackgroundSenseProducer::spawn_with_callback(
                    "kinect-depth",
                    provider,
                    Duration::from_millis(33),
                    move |packet| {
                        if let Some(live_state) = &live_state_for_kinect {
                            if let SensePacket::EyeFrame(frame) = packet {
                                live_state.record_live_eye_frame(frame.clone());
                            }
                            if matches!(packet, SensePacket::EyeFrame(_) | SensePacket::Kinect(_)) {
                                publish_live_sensor_only_snapshot(live_state, packet);
                            }
                        }
                    },
                )));
            }
            Err(err) => {
                println!("failed to initialize Kinect depth: {err}; continuing without it");
            }
        }
        #[cfg(not(feature = "kinect-freenect"))]
        println!(
            "failed to initialize Kinect depth: rebuild with --features kinect-freenect; continuing without it"
        );
    }

    if let Some(device) = &args.mic {
        let pref_name = if device == "default" {
            None
        } else {
            Some(device.as_str())
        };
        let mut asr_config = AsrToolConfig::default();
        if let Some(command) = args.asr_command.clone() {
            asr_config.command = Some(command);
        }
        match MicrophoneSenseProvider::with_asr_config(pref_name, asr_config) {
            Ok(provider) => {
                let live_state_for_mic = live_state.clone();
                sensors.push(Box::new(BackgroundSenseProducer::spawn_with_callback(
                    "microphone",
                    provider,
                    Duration::from_millis(25),
                    move |packet| {
                        if let Some(live_state) = &live_state_for_mic {
                            if matches!(packet, SensePacket::EarPcm(_) | SensePacket::Ear(_)) {
                                publish_live_sensor_only_snapshot(live_state, packet);
                            }
                        }
                    },
                )));
            }
            Err(err) => {
                if args.require_mic {
                    anyhow::bail!("failed to initialize mic: {err}");
                } else {
                    println!("failed to initialize mic: {err}; continuing without it");
                }
            }
        }
    }

    if let Some(device) = lidar_device.as_deref() {
        if create_port
            .as_deref()
            .map(|create| same_serial_device(create, device))
            .unwrap_or(false)
        {
            let error = format!(
                "lidar {device} is also selected as the brainstem cockpit device; pin PETE_COCKPIT_PORT and LIDAR_SERIAL_PORT to distinct devices"
            );
            if args.require_lidar || args.lidar.is_some() {
                anyhow::bail!(error);
            }
            println!("{error}; continuing without lidar");
        } else {
            match Lfcd2SenseProvider::with_extrinsics(device, lidar_extrinsics) {
                Ok(provider) => {
                    println!(
                        "HLS-LFCD2 lidar: {device} at {} baud (position [{}, {}, {}] m, roll/pitch/yaw [{}, {}, {}] deg)",
                        Lfcd2SenseProvider::BAUD_RATE,
                        args.lidar_forward_m,
                        args.lidar_left_m,
                        args.lidar_height_m,
                        args.lidar_roll_deg,
                        args.lidar_pitch_deg,
                        args.lidar_yaw_deg
                    );
                    sensors.push(Box::new(provider));
                }
                Err(err) => {
                    if args.require_lidar {
                        anyhow::bail!("failed to initialize HLS-LFCD2 lidar: {err}");
                    } else {
                        println!(
                            "failed to initialize HLS-LFCD2 lidar: {err}; continuing without it"
                        );
                    }
                }
            }
        }
    } else if args.require_lidar {
        anyhow::bail!(
            "--require-lidar was set but no HLS-LFCD2 device was detected; pass --lidar /dev/serial/by-id/DEVICE"
        );
    }

    if let Some(device) = selected_gps_device(
        args.gps.as_deref(),
        is_mock_body,
        &env_report,
        create_port.as_deref(),
    ) {
        match GpsSenseProvider::new(&device, 9600) {
            Ok(provider) => sensors.push(Box::new(provider)),
            Err(err) => {
                if args.require_gps {
                    anyhow::bail!("failed to initialize gps: {err}");
                } else {
                    println!("failed to initialize gps: {err}; continuing without it");
                }
            }
        }
    }

    if let Some(device) = selected_imu_device(args.imu.as_deref(), is_mock_body) {
        match ImuSenseProvider::new(device) {
            Ok(provider) => {
                let live_state_for_imu = live_state.clone();
                sensors.push(Box::new(BackgroundSenseProducer::spawn_with_callback(
                    "imu",
                    provider,
                    Duration::from_millis(25),
                    move |packet| {
                        if let Some(live_state) = &live_state_for_imu {
                            if matches!(packet, SensePacket::Imu(_)) {
                                publish_live_sensor_only_snapshot(live_state, packet);
                            }
                        }
                    },
                )));
            }
            Err(err) => {
                if args.require_imu {
                    anyhow::bail!("failed to initialize imu: {err}");
                } else {
                    println!("failed to initialize imu: {err}; continuing without it");
                }
            }
        }
    }

    let mut mouth = match QueuedPiperCpalMouth::from_env() {
        Ok(Some(mouth)) => Some(mouth),
        Ok(None) => {
            println!(
                "robot mouth disabled: no Piper voice found; set PETE_TTS_PIPER_VOICE and PETE_TTS_PIPER_CONFIG"
            );
            None
        }
        Err(error) => {
            println!("robot mouth disabled: could not load Piper voice: {error}");
            None
        }
    };
    if robot_mode != RobotMode::Slow {
        if let Some(mouth_ref) = &mouth {
            if !speak_robot_mouth_text_before_status(mouth_ref, "Hello. My name is Pete.") {
                mouth = None;
            }
        }
    }

    let init_body = None;

    let ledger = JsonlLedger::new(&args.ledger);
    let memory = DurableExperienceStore::from_env();
    let recall = memory.clone();
    let runtime = MinimalRuntime::with_reign_queue(
        ledger.clone(),
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        configured_llm_agent(&args.llm)?,
        reign_queue,
    );
    let live_image_enricher = LiveImageEnricher::new(configured_llm_config(&args.llm)?)?;
    let mut frame_processor_warnings = Vec::new();
    let active_sensor_count = sensors.len();
    let initialization = robot_initialization_metadata(
        robot_mode,
        &args,
        is_mock_body,
        create_port.as_deref(),
        active_sensor_count,
        init_body.as_ref(),
        &brainstem_capabilities,
    );
    let mut runner = RealRobotRunner::new(robot_mode, cockpit, sensors, runtime)
        .with_frame_processor(real_robot_frame_processor(&mut frame_processor_warnings).await)
        .with_live_image_enricher(live_image_enricher)
        .with_robot_initialization(initialization.clone())
        .with_brainstem_interface(serde_json::to_value(&brainstem_capabilities)?)
        .with_autonomous_motion(args.autonomous_motion);
    runner.tick_ms = args.tick_ms;
    for warning in frame_processor_warnings {
        println!("{warning}");
    }

    let mut capture = match &args.capture {
        Some(path) => {
            let mut writer =
                CaptureWriter::create(path, CaptureSource::RealRobot, Some(args.tick_ms)).await?;
            writer.manifest_mut().firmware_identity =
                brainstem_firmware_identity(runner.cockpit.client_mut().as_mut());
            Some(writer)
        }
        None => None,
    };

    if robot_mode != RobotMode::Slow {
        enqueue_default_bringup_outputs(
            &mouth,
            runner.cockpit.client_mut().as_mut(),
            &initialization,
        );
    } else {
        let status = runner.cockpit.resync_event_cursor_from_status()?;
        if let Some(event_next_seq) = status.event_next_seq {
            println!(
                "possession event cursor resynced before control loop: next_seq={event_next_seq}"
            );
        }
    }

    let max_steps = args.steps.or_else(|| {
        args.duration_seconds
            .map(|seconds| duration_to_steps(seconds, args.tick_ms))
    });
    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);
    let mut played_reign_audio = HashSet::new();
    let mut played_skill_audio = HashSet::new();
    while max_steps
        .map(|limit| runner.tick_count < limit)
        .unwrap_or(true)
    {
        let tick_started_at = Instant::now();
        let tick_result = tokio::select! {
            signal = &mut shutdown => {
                println!("received {signal}; stopping robot and surrendering possession");
                break;
            }
            result = async {
                match robot_mode {
                    RobotMode::ReadOnly => runner.tick_read_only().await,
                    RobotMode::Slow => runner.tick_slow_manual().await,
                    RobotMode::Disabled => unreachable!("disabled mode bailed before runner start"),
                }
            } => result,
        };
        let (snapshot, tick) = match tick_result {
            Ok(values) => values,
            Err(error)
                if robot_mode == RobotMode::Slow && is_reconnectable_cockpit_error(&error) =>
            {
                eprintln!("possession transport/session lost; motor gate closed: {error}");
                disconnect_possession_cockpit_for_reconnect(&mut runner.cockpit);
                let replacement =
                    reconnect_possession_cockpit(create_port.as_deref(), &args).await?;
                let mut replacement = replacement;
                establish_create_sensor_stream(replacement.as_mut(), true)?;
                runner.cockpit.replace_client(replacement);
                eprintln!(
                    "possession reconnected with fresh session, lease, and complete body packet; stopped=true"
                );
                continue;
            }
            Err(error) if robot_mode == RobotMode::Slow && is_charging_busy_error(&error) => {
                eprintln!(
                    "possession motor gate closed: brainstem reports charging_busy; Create charging indicator is wired to Pico GP17 physical pin 22"
                );
                return Err(error);
            }
            Err(error) if robot_mode == RobotMode::ReadOnly => {
                if is_transient_robot_timeout(&error) {
                    eprintln!("read-only tick timed out; continuing");
                    tokio::time::sleep(remaining_tick_delay(
                        args.tick_ms,
                        tick_started_at.elapsed(),
                    ))
                    .await;
                    continue;
                }
                return Err(error);
            }
            Err(error) => return Err(error),
        };
        if robot_mode != RobotMode::Slow {
            play_event_script_outputs(&mouth, runner.cockpit.client_mut().as_mut(), &tick);
            play_reign_audio_action(
                &mouth,
                runner.cockpit.client_mut().as_mut(),
                &tick,
                &mut played_reign_audio,
            );
        }
        play_lua_skill_audio(&mouth, &tick, &mut played_skill_audio);
        if let Some(live_state) = &live_state {
            live_state.update(snapshot.clone());
            live_state.update_embodied_context(tick.frame.embodied_context());
        }
        if let Some(writer) = capture.as_mut() {
            writer
                .append_snapshot(snapshot.body.last_update_ms, snapshot.clone(), Vec::new())
                .await?;
        }
        let motion_note = slow_motion_note(&snapshot);
        println!(
            "robot {:?} tick {}: battery {:.2}, chosen {:?}{}",
            robot_mode,
            runner.tick_count,
            tick.frame.now.body.battery_level,
            tick.chosen_action,
            motion_note
        );
        tokio::select! {
            signal = &mut shutdown => {
                println!("received {signal}; stopping robot and surrendering possession");
                break;
            }
            _ = tokio::time::sleep(remaining_tick_delay(
                args.tick_ms,
                tick_started_at.elapsed(),
            )) => {}
        }
    }

    if robot_mode == RobotMode::Slow {
        // Preserve acknowledgement semantics: motion must be stopped before
        // surrendering the motherbrain gate. The brainstem continues owning
        // and supervising Create OI in Full mode.
        run_possession_shutdown(runner.cockpit.client_mut().as_mut())?;
        let final_status = runner
            .cockpit
            .client_mut()
            .get_status()
            .context("possession final status was not acknowledged")?
            .summary();
        if final_status.active_motion == Some(true) {
            anyhow::bail!(
                "possession shutdown did not prove stopped: moving={:?} armed={:?}",
                final_status.active_motion,
                final_status.armed
            );
        }
        println!("possession exorcize acknowledged: stopped=true possessed=false; brainstem OI supervision retained");
    }

    let capture_summary = if let Some(writer) = capture {
        let manifest = writer.finish().await?;
        format!(
            ", capture {}, {} frames",
            args.capture.as_deref().unwrap_or_default(),
            manifest.frame_count
        )
    } else {
        String::new()
    };
    let transitions = ledger.transitions().await?;
    println!(
        "robot {:?} complete: {} ticks, ledger {}, {} transitions{}",
        robot_mode,
        runner.tick_count,
        args.ledger,
        transitions.len(),
        capture_summary
    );
    Ok(())
}

async fn shutdown_signal() -> &'static str {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};

        let mut terminate =
            signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");
        tokio::select! {
            result = tokio::signal::ctrl_c() => {
                result.expect("failed to install Ctrl-C handler");
                "SIGINT"
            }
            _ = terminate.recv() => "SIGTERM",
        }
    }
    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
        "Ctrl-C"
    }
}

async fn reconnect_possession_cockpit(
    create_port: Option<&str>,
    args: &RobotArgs,
) -> Result<Box<dyn Cockpit + Send>> {
    let mut backoff_ms = args.reconnect_initial_backoff_ms.max(1);
    let max_backoff_ms = args.reconnect_max_backoff_ms.max(backoff_ms).min(60_000);
    loop {
        match open_robot_cockpit_or_fallback(
            args.cockpit,
            create_port,
            RobotMode::Slow,
            args.brainstem_device_id.as_deref(),
            args.brainstem_boot_id.as_deref(),
            args.max_linear_mm_s,
            args.max_angular_mrad_s,
        ) {
            Ok((cockpit, RobotMode::Slow, _)) => return Ok(cockpit),
            Ok(_) => anyhow::bail!("possession reconnect attempted an invalid fallback"),
            Err(error) if is_identity_acceptance_error(&error) => return Err(error),
            Err(error) => {
                eprintln!("possession reconnect failed: {error}; retrying in {backoff_ms} ms");
                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                backoff_ms = next_reconnect_backoff_ms(backoff_ms, max_backoff_ms);
            }
        }
    }
}

fn disconnect_possession_cockpit_for_reconnect(cockpit: &mut SafeCockpit<Box<dyn Cockpit + Send>>) {
    cockpit.replace_client(Box::new(ClosedCockpit::new(
        "possession reconnect in progress",
    )));
}

fn slow_motion_note(snapshot: &WorldSnapshot) -> String {
    if let Some(reason) = snapshot
        .action_debug
        .as_ref()
        .and_then(|debug| debug.get("why_not_moving"))
        .and_then(|reason| reason.as_str())
        .filter(|reason| !reason.is_empty())
    {
        return format!(", motion blocked: {reason}");
    }
    let Some(goal) = snapshot
        .action_debug
        .as_ref()
        .and_then(|debug| debug.get("conductor_navigation_goal"))
    else {
        return String::new();
    };
    if goal.get("intent").and_then(|intent| intent.as_str()) != Some("recover_from_contact") {
        return String::new();
    }
    let selected_action = snapshot
        .final_selected_action
        .as_ref()
        .and_then(|action| serde_json::to_value(action).ok());
    if selected_action.as_ref() != goal.get("action") {
        return String::new();
    }
    goal.get("reason")
        .and_then(|reason| reason.as_str())
        .filter(|reason| !reason.is_empty())
        .map(|reason| format!(", recovery: {reason}"))
        .unwrap_or_default()
}

fn run_possession_shutdown(cockpit: &mut dyn Cockpit) -> Result<()> {
    run_possession_shutdown_with_retry(
        cockpit,
        POSSESSION_SHUTDOWN_BUSY_RETRY_ATTEMPTS,
        POSSESSION_SHUTDOWN_BUSY_RETRY_DELAY,
    )
}

fn run_possession_shutdown_with_retry<C: Cockpit + ?Sized>(
    cockpit: &mut C,
    attempts: usize,
    delay: Duration,
) -> Result<()> {
    retry_possession_shutdown_command(
        cockpit,
        "possession shutdown STOP",
        attempts,
        delay,
        Cockpit::stop,
    )?;
    retry_possession_shutdown_command(
        cockpit,
        "possession exorcize",
        attempts,
        delay,
        Cockpit::exorcize,
    )
}

fn retry_possession_shutdown_command<C, F>(
    cockpit: &mut C,
    label: &'static str,
    attempts: usize,
    delay: Duration,
    mut command: F,
) -> Result<()>
where
    C: Cockpit + ?Sized,
    F: FnMut(&mut C) -> std::result::Result<(), CockpitError>,
{
    let attempts = attempts.max(1);
    for attempt in 0..attempts {
        match command(cockpit) {
            Ok(()) => return Ok(()),
            Err(error) if is_plain_busy_cockpit_error(&error) && attempt + 1 < attempts => {
                std::thread::sleep(delay);
            }
            Err(error) => {
                return Err(error).with_context(|| format!("{label} was not acknowledged"))
            }
        }
    }
    unreachable!("bounded shutdown retry always returns on its final attempt")
}

fn next_reconnect_backoff_ms(current_ms: u64, maximum_ms: u64) -> u64 {
    current_ms.saturating_mul(2).min(maximum_ms.max(1))
}

fn is_identity_acceptance_error(error: &AnyhowError) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("identity mismatch") || message.contains("requires --brainstem")
}

struct ClosedCockpit {
    reason: String,
}

impl ClosedCockpit {
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    fn error(&self) -> CockpitError {
        CockpitError::Policy(self.reason.clone())
    }
}

impl Cockpit for ClosedCockpit {
    fn execute(
        &mut self,
        _request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        Err(self.error())
    }

    fn handshake(
        &mut self,
        _hello: HandshakeHello,
    ) -> pete_cockpit::Result<pete_cockpit::HandshakeOutcome> {
        Err(self.error())
    }

    fn execute_in_session(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        Err(self.error())
    }

    fn execute_with_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ControlLease,
        _request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        Err(self.error())
    }

    fn execute_with_service_lease(
        &mut self,
        _session: &pete_cockpit::CockpitSession,
        _lease: &pete_cockpit::ServiceLease,
        _request: pete_cockpit::CockpitRequest,
    ) -> pete_cockpit::Result<pete_cockpit::CockpitResponse> {
        Err(self.error())
    }
}

fn run_mouth(args: MouthArgs) -> Result<()> {
    let Some(mouth) = QueuedPiperCpalMouth::from_env()? else {
        anyhow::bail!(
            "robot mouth disabled: no Piper voice found; set PETE_TTS_PIPER_VOICE and PETE_TTS_PIPER_CONFIG"
        );
    };
    let outcome = mouth.enqueue_and_wait_timeout(args.text, Some(Duration::from_secs(60)))?;
    println!(
        "robot mouth diagnostic complete: device {}, duration {} ms",
        outcome.device.as_deref().unwrap_or("<unknown>"),
        outcome.duration_ms.unwrap_or_default()
    );
    Ok(())
}

fn run_whisper_transcribe(args: WhisperTranscribeArgs) -> Result<()> {
    use speaking::{AudioFrame, SpeechRecognizer, WhisperSpeechRecognizer};

    let model = args
        .model
        .or_else(|| env_path("PETE_WHISPER_MODEL"))
        .or_else(default_whisper_model_path)
        .context("missing Whisper model path; run `just setup-whisper`, set PETE_WHISPER_MODEL, or pass --model")?;
    let samples = read_wav_as_16khz_mono_f32(&args.wav)
        .with_context(|| format!("failed to read {}", args.wav.display()))?;
    if samples.is_empty() {
        return Ok(());
    }
    let mut recognizer = WhisperSpeechRecognizer::new_quiet_without_input_padding(&model)
        .with_context(|| format!("loading Whisper model {}", model.display()))?;
    recognizer.push_frame(&AudioFrame {
        sample_rate_hz: 16_000,
        channels: 1,
        samples,
    })?;
    let chunks = recognizer.poll_chunks()?;
    let transcript = chunks
        .into_iter()
        .map(|chunk| chunk.text)
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();
    if !transcript.is_empty() {
        println!("{transcript}");
    }
    Ok(())
}

fn read_wav_as_16khz_mono_f32(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));
    let source_rate = spec.sample_rate.max(1);
    let mut interleaved = Vec::new();
    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => {
            for sample in reader.samples::<f32>() {
                interleaved.push(sample?);
            }
        }
        (hound::SampleFormat::Int, 8) => {
            for sample in reader.samples::<i8>() {
                interleaved.push(sample? as f32 / i8::MAX as f32);
            }
        }
        (hound::SampleFormat::Int, 16) => {
            for sample in reader.samples::<i16>() {
                interleaved.push(sample? as f32 / i16::MAX as f32);
            }
        }
        (hound::SampleFormat::Int, 24 | 32) => {
            let scale = ((1_i64 << (spec.bits_per_sample - 1)) - 1) as f32;
            for sample in reader.samples::<i32>() {
                interleaved.push(sample? as f32 / scale);
            }
        }
        _ => anyhow::bail!(
            "unsupported WAV format: {:?} {} bits",
            spec.sample_format,
            spec.bits_per_sample
        ),
    }
    let mono = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().copied().sum::<f32>() / frame.len().max(1) as f32)
        .collect::<Vec<_>>();
    Ok(resample_mono_linear(&mono, source_rate, 16_000))
}

fn resample_mono_linear(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if samples.is_empty() || source_rate == 0 || target_rate == 0 {
        return Vec::new();
    }
    if source_rate == target_rate {
        return samples.to_vec();
    }
    let output_len = (samples.len() as u64)
        .saturating_mul(u64::from(target_rate))
        .div_ceil(u64::from(source_rate)) as usize;
    let ratio = source_rate as f64 / target_rate as f64;
    let mut output = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let pos = index as f64 * ratio;
        let left = pos.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let fraction = (pos - left as f64) as f32;
        let sample = samples[left] * (1.0 - fraction) + samples[right] * fraction;
        output.push(sample.clamp(-1.0, 1.0));
    }
    output
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn default_whisper_model_path() -> Option<PathBuf> {
    let data_home = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"))
        })?;
    Some(
        data_home
            .join("pete")
            .join("models")
            .join("whisper")
            .join(DEFAULT_WHISPER_MODEL_FILENAME),
    )
}

fn robot_initialization_metadata(
    robot_mode: RobotMode,
    args: &RobotArgs,
    is_mock_body: bool,
    create_port: Option<&str>,
    active_sensor_count: usize,
    init_body: Option<&BodySense>,
    brainstem_capabilities: &pete_cockpit::CockpitCapabilities,
) -> serde_json::Value {
    let body_status = if is_mock_body {
        "mock Create body connected".to_string()
    } else if let Some(port) = create_port {
        format!("Create body connected on {port}")
    } else {
        "Create body connected".to_string()
    };
    let mode = match robot_mode {
        RobotMode::ReadOnly => "read-only",
        RobotMode::Slow => "slow",
        RobotMode::Disabled => "disabled",
    };
    serde_json::json!({
        "mode": mode,
        "body": body_status,
        "brainstem_capabilities": brainstem_capabilities,
        "battery_percent": init_body.map(|body| {
            (body.battery_level.clamp(0.0, 1.0) * 100.0).round() as u32
        }),
        "charging": init_body.map(|body| body.charging),
        "active_sensors": active_sensor_count,
        "requested_sensors": requested_robot_sensor_count(args),
        "ledger": args.ledger.clone(),
        "tick_ms": args.tick_ms,
        "dashboard": args.dashboard.map(|addr| addr.to_string()),
        "capture": args.capture.clone(),
        "brainstem_device_id": args.brainstem_device_id.clone(),
        "brainstem_boot_id": args.brainstem_boot_id.clone(),
        "reconnect_initial_backoff_ms": args.reconnect_initial_backoff_ms,
        "reconnect_max_backoff_ms": args.reconnect_max_backoff_ms,
    })
}

fn brainstem_firmware_identity(cockpit: &mut dyn Cockpit) -> Option<serde_json::Value> {
    let status = cockpit.get_status().ok()?;
    let fields = [
        "firmware_name",
        "firmware_version",
        "git_commit",
        "git_commit_short",
        "git_dirty",
        "build_timestamp",
        "build_profile",
        "build_target",
        "build_backend",
        "build_id",
    ];
    if let Ok(status) = serde_json::from_str::<serde_json::Value>(&status.raw) {
        let mut identity = serde_json::Map::new();
        for field in fields {
            if let Some(value) = status.get(field) {
                identity.insert(field.into(), value.clone());
            }
        }
        return (!identity.is_empty()).then_some(serde_json::Value::Object(identity));
    }
    let mut identity = serde_json::Map::new();
    for item in status.raw.split_ascii_whitespace() {
        let Some((key, value)) = item.split_once('=') else {
            continue;
        };
        if fields.contains(&key) {
            let value = match key {
                "git_dirty" => serde_json::Value::Bool(value == "true"),
                _ => serde_json::Value::String(value.to_string()),
            };
            identity.insert(key.into(), value);
        }
    }
    (!identity.is_empty()).then_some(serde_json::Value::Object(identity))
}

fn enqueue_default_bringup_outputs(
    mouth: &Option<QueuedPiperCpalMouth>,
    cockpit: &mut dyn Cockpit,
    initialization: &serde_json::Value,
) {
    play_robot_song(cockpit, "bring_up");
    play_robot_chirp(cockpit, "Confirm");
    let Some(mouth) = mouth.as_ref() else {
        return;
    };
    if let Some(mode) = initialization.get("mode").and_then(|value| value.as_str()) {
        enqueue_robot_mouth_text(
            mouth,
            &format!("Pete robot initialization complete in {mode} mode."),
        );
    }
    if let Some(body) = initialization.get("body").and_then(|value| value.as_str()) {
        enqueue_robot_mouth_text(mouth, &format!("{body}."));
    }
    match (
        initialization
            .get("battery_percent")
            .and_then(|value| value.as_u64()),
        initialization
            .get("charging")
            .and_then(|value| value.as_bool()),
    ) {
        (Some(percent), Some(charging)) => {
            let charging = if charging { "charging" } else { "not charging" };
            enqueue_robot_mouth_text(
                mouth,
                &format!("Battery is {percent} percent and {charging}."),
            );
        }
        _ => enqueue_robot_mouth_text(mouth, "Battery status is unavailable."),
    }
}

fn speak_robot_mouth_text_before_status(mouth: &QueuedPiperCpalMouth, text: &str) -> bool {
    println!("robot mouth speaking before body status: {text:?}");
    match mouth.enqueue_and_wait_timeout(text.to_string(), Some(Duration::from_secs(20))) {
        Ok(outcome) => {
            println!(
                "robot mouth completed before body status: device {}, duration {} ms",
                outcome.device.as_deref().unwrap_or("<unknown>"),
                outcome.duration_ms.unwrap_or_default()
            );
            true
        }
        Err(error) => {
            println!(
                "robot mouth pre-status speech failed; disabling mouth for this run and continuing: {error}"
            );
            false
        }
    }
}

fn play_event_script_outputs(
    mouth: &Option<QueuedPiperCpalMouth>,
    cockpit: &mut dyn Cockpit,
    tick: &RuntimeTick,
) {
    let Some(scripts) = tick.frame.now.extensions.get("event_scripts") else {
        return;
    };
    let Some(object) = scripts.as_object() else {
        return;
    };
    for sequence in object.values() {
        let Some(actions) = sequence.get("actions").and_then(|value| value.as_array()) else {
            continue;
        };
        for action in actions {
            let requested = action.get("requested").unwrap_or(action);
            if let Some(text) = requested.get("text").and_then(|value| value.as_str()) {
                if let Some(mouth) = mouth.as_ref() {
                    enqueue_robot_mouth_text(mouth, text);
                }
            } else if let Some(pattern) = requested.get("pattern").and_then(|value| value.as_str())
            {
                play_robot_chirp(cockpit, pattern);
            } else if let Some(name) = requested.get("name").and_then(|value| value.as_str()) {
                play_robot_song(cockpit, name);
            }
        }
    }
}

fn play_reign_audio_action(
    mouth: &Option<QueuedPiperCpalMouth>,
    cockpit: &mut dyn Cockpit,
    tick: &RuntimeTick,
    played: &mut HashSet<String>,
) {
    if tick.skill_request.is_some() {
        return;
    }
    let Some(action) = tick.chosen_action.as_ref() else {
        return;
    };
    let action_key = match action {
        ActionPrimitive::Speak { text } => format!("speak:{text}"),
        ActionPrimitive::Chirp { pattern } => format!("chirp:{pattern:?}"),
        _ => return,
    };
    let key = tick
        .frame
        .reign_input
        .as_ref()
        .map(|input| format!("reign:{}:{action_key}", input.id))
        .unwrap_or_else(|| format!("frame:{}:{action_key}", tick.frame.id));
    if !played.insert(key) {
        return;
    }
    match action {
        ActionPrimitive::Speak { text } => {
            if let Some(mouth) = mouth.as_ref() {
                enqueue_robot_mouth_text(mouth, text);
            } else {
                println!("robot mouth unavailable; skipped Reign speech {text:?}");
            }
        }
        ActionPrimitive::Chirp { pattern } => {
            play_robot_chirp(cockpit, &format!("{pattern:?}"));
        }
        _ => {}
    }
}

fn play_lua_skill_audio(
    mouth: &Option<QueuedPiperCpalMouth>,
    tick: &RuntimeTick,
    played: &mut HashSet<String>,
) {
    let Some(record) = tick.frame.now.extensions.get("motherbrain.skill_execution") else {
        return;
    };
    for (key, text) in lua_skill_speech_intents(record) {
        if !played.insert(key) {
            continue;
        }
        if let Some(mouth) = mouth.as_ref() {
            enqueue_robot_mouth_text(mouth, &text);
        } else {
            println!("robot mouth unavailable; skipped Lua skill speech {text:?}");
        }
    }
}

fn lua_skill_speech_intents(record: &serde_json::Value) -> Vec<(String, String)> {
    let execution_id = record
        .get("execution_id")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_default();
    let Some(trace) = record.get("trace").and_then(serde_json::Value::as_array) else {
        return Vec::new();
    };
    let mut intents = Vec::new();
    for event in trace {
        if event.get("kind").and_then(serde_json::Value::as_str) != Some("primitive")
            || event.get("operation").and_then(serde_json::Value::as_str) != Some("say")
        {
            continue;
        }
        let operation_id = event
            .get("operation_id")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or_default();
        let Some(text) = event
            .pointer("/detail/text")
            .and_then(serde_json::Value::as_str)
        else {
            continue;
        };
        let key = format!("lua:{execution_id}:{operation_id}");
        intents.push((key, text.to_string()));
    }
    intents
}

fn enqueue_robot_mouth_text(mouth: &QueuedPiperCpalMouth, text: &str) {
    match mouth.enqueue(text.to_string()) {
        Ok(()) => println!("robot mouth queued: {text:?}"),
        Err(error) => println!("robot mouth queue failed: {error}; text {text:?}"),
    }
}

fn play_robot_chirp(cockpit: &mut dyn Cockpit, pattern: &str) {
    play_body_song(
        cockpit,
        &format!("chirp {pattern}"),
        chirp_pattern_song(pattern),
    );
}

fn play_robot_song(cockpit: &mut dyn Cockpit, name: &str) {
    play_body_song(cockpit, name, robot_song(name));
}

fn play_body_song(cockpit: &mut dyn Cockpit, label: &str, song: BodySong) {
    let tones = song
        .tones
        .iter()
        .map(|tone| SongTone {
            note: tone.note,
            duration_64ths: tone.duration_64ths,
        })
        .collect::<Vec<_>>();
    match cockpit
        .song_define(0, &tones)
        .and_then(|()| cockpit.song_play(0))
    {
        Ok(()) => println!("robot cockpit song played: {label}"),
        Err(error) => println!("robot cockpit song skipped: {error}; song {label}"),
    }
}

fn chirp_pattern_song(pattern: &str) -> BodySong {
    BodySong::new(
        chirp_pattern_notes(pattern)
            .iter()
            .enumerate()
            .map(|(index, note)| {
                tone(
                    *note,
                    if index + 1 == chirp_pattern_notes(pattern).len() {
                        8
                    } else {
                        6
                    },
                )
            })
            .collect::<Vec<_>>(),
    )
}

fn chirp_pattern_notes(pattern: &str) -> &'static [u8] {
    match normalized_chirp_pattern(pattern).as_str() {
        "confirm" => &[79, 84, 79],
        "warning" => &[79, 75],
        "hello" => &[72, 76, 79],
        "goodbye" => &[79, 76, 72],
        "curious" => &[72, 76, 74],
        "idea" => &[76, 81, 84],
        "goalacquired" => &[72, 79, 84, 91],
        "searching" => &[72, 74, 76, 74],
        "sawsomething" => &[84, 91],
        "surprise" => &[72, 84],
        "learned" => &[74, 79, 83],
        "personrecognized" => &[76, 79, 84, 79],
        "objectrecognized" => &[79, 84, 76],
        "placerecognized" => &[79, 84, 72],
        "didntunderstand" => &[79, 81, 78],
        "docking" => &[67, 72, 76, 79],
        "chargingstarted" => &[60, 67, 72],
        "sleep" => &[79, 76, 72, 67],
        "wake" => &[67, 72, 79],
        _ => &[72],
    }
}

fn normalized_chirp_pattern(pattern: &str) -> String {
    pattern
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase()
}

fn robot_song(name: &str) -> BodySong {
    match name {
        "bring_up" => BodySong::new([
            tone(60, 8),
            tone(64, 8),
            tone(67, 8),
            tone(72, 12),
            tone(67, 6),
            tone(72, 14),
        ]),
        "mournful_bump" => BodySong::new([tone(64, 12), tone(63, 12), tone(60, 16), tone(55, 20)]),
        _ => BodySong::new([tone(60, 8), tone(67, 8), tone(72, 12)]),
    }
}

fn tone(note: u8, duration_64ths: u8) -> BodyTone {
    BodyTone::new(note, duration_64ths)
}

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

