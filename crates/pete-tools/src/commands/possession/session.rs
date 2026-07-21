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
            control_state: (robot_mode == RobotMode::Slow).then(|| "active".to_string()),
            control_detail: (robot_mode == RobotMode::Slow)
                .then(|| "brainstem possession active".to_string()),
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

    if let Some(device) = local_imu_provider_allowed(&args)
        .then(|| selected_imu_device(args.imu.as_deref(), is_mock_body))
        .flatten()
    {
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
        .with_imu_override(imu_source_override(&args))
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
            writer.manifest_mut().notes.push(
                "raw RGB, depth, and audio assets are exported when present; paths are relative to the capture root"
                    .to_string(),
            );
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
    let mut possession_connected = true;
    // Keep every fallible control-loop exit inside this outcome. Returning from
    // the async block cannot bypass the common STOP, exorcize, status, and
    // capture finalization path below.
    let control_result: Result<()> = async {
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
                    possession_connected = false;
                    if let Some(live_state) = &live_state {
                        live_state.update_session_control(
                            "stopped-reconnecting",
                            format!("stopped; brainstem unavailable; reconnecting: {error}"),
                        );
                    }
                    match reconnect_possession_cockpit(
                        create_port.as_deref(),
                        &args,
                        shutdown.as_mut(),
                    )
                    .await?
                    {
                        PossessionReconnect::Reconnected(replacement) => {
                            runner.cockpit.replace_client(replacement);
                            possession_connected = true;
                            runner.note_brainstem_reconnect();
                            if let Some(live_state) = &live_state {
                                live_state.update_session_control(
                                    "active",
                                    "brainstem possession active after reconnect",
                                );
                            }
                            eprintln!(
                                "possession reconnected with fresh session, lease, and complete body packet; stopped=true"
                            );
                            continue;
                        }
                        PossessionReconnect::Shutdown(signal) => {
                            if let Some(live_state) = &live_state {
                                live_state.update_session_control(
                                    "stopped-shutdown",
                                    format!(
                                        "stopped; received {signal} while waiting for brainstem"
                                    ),
                                );
                            }
                            println!(
                                "received {signal} while waiting for brainstem; closing capture and ledger"
                            );
                            break;
                        }
                    }
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
                let runtime_map = runner.runtime.canonical_map();
                live_state.update_with_runtime_map(snapshot.clone(), runtime_map.as_ref());
                live_state.update_embodied_context(tick.frame.embodied_context());
            }
            if let Some(writer) = capture.as_mut() {
                append_real_robot_snapshot(writer, &snapshot, &tick).await?;
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
        Ok(())
    }
    .await;

    if robot_mode == RobotMode::Slow && !possession_connected {
        eprintln!(
            "possession shutdown acknowledgement unavailable: brainstem disconnected; motion remains fail-closed under command, heartbeat, and lease expiry"
        );
    }

    let possession_cockpit = (robot_mode == RobotMode::Slow && possession_connected)
        .then(|| runner.cockpit.client_mut().as_mut());
    let finalization = finalize_robot_exit(
        control_result,
        possession_cockpit,
        capture,
        args.capture.as_deref(),
    )
    .await;
    let capture_summary = finalization.capture_summary.clone();
    let transitions_result = ledger.transitions().await;
    let transition_count = transitions_result
        .as_ref()
        .map(|transitions| transitions.len())
        .unwrap_or_default();
    finalization.into_result(transitions_result.map(|_| ()))?;
    println!(
        "robot {:?} complete: {} ticks, ledger {}, {} transitions{}",
        robot_mode,
        runner.tick_count,
        args.ledger,
        transition_count,
        capture_summary
    );
    Ok(())
}

struct RobotExitFinalization {
    control_result: Result<()>,
    shutdown_result: Result<()>,
    capture_result: Result<()>,
    capture_summary: String,
}

impl RobotExitFinalization {
    fn into_result(self, ledger_result: Result<()>) -> Result<()> {
        combine_robot_exit_results(
            self.control_result,
            self.shutdown_result,
            self.capture_result,
            ledger_result,
        )
    }
}

async fn finalize_robot_exit<C: Cockpit + ?Sized>(
    control_result: Result<()>,
    possession_cockpit: Option<&mut C>,
    capture: Option<CaptureWriter>,
    capture_path: Option<&str>,
) -> RobotExitFinalization {
    let shutdown_result = possession_cockpit
        .map(run_acknowledged_possession_shutdown)
        .unwrap_or(Ok(()));
    let capture_result = match capture {
        Some(writer) => writer.finish().await.map(|manifest| {
            format!(
                ", capture {}, {} frames",
                capture_path.unwrap_or_default(),
                manifest.frame_count
            )
        }),
        None => Ok(String::new()),
    };
    let capture_summary = capture_result.as_ref().cloned().unwrap_or_default();
    RobotExitFinalization {
        control_result,
        shutdown_result,
        capture_result: capture_result.map(|_| ()),
        capture_summary,
    }
}

fn run_acknowledged_possession_shutdown<C: Cockpit + ?Sized>(cockpit: &mut C) -> Result<()> {
    // Preserve acknowledgement semantics: motion must be stopped before
    // surrendering the motherbrain gate. The brainstem continues owning and
    // supervising Create OI in Full mode.
    let shutdown_result = run_possession_shutdown(cockpit);
    let final_status_result = cockpit
        .get_status()
        .context("possession final status was not acknowledged")
        .and_then(|status| {
            let final_status = status.summary();
            if final_status.active_motion == Some(true) {
                anyhow::bail!(
                    "possession shutdown did not prove stopped: moving={:?} armed={:?}",
                    final_status.active_motion,
                    final_status.armed
                );
            }
            Ok(())
        });
    match (shutdown_result, final_status_result) {
        (Ok(()), Ok(())) => {
            println!("possession exorcize acknowledged: stopped=true possessed=false; brainstem OI supervision retained");
            Ok(())
        }
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(shutdown_error), Err(status_error)) => anyhow::bail!(
            "possession shutdown and final status both failed: shutdown: {shutdown_error:#}; final status: {status_error:#}"
        ),
    }
}

fn combine_robot_exit_results(
    control_result: Result<()>,
    shutdown_result: Result<()>,
    capture_result: Result<()>,
    ledger_result: Result<()>,
) -> Result<()> {
    let mut failures = Vec::new();
    if let Err(error) = control_result {
        failures.push(("robot control loop", error));
    }
    if let Err(error) = shutdown_result {
        failures.push(("possession shutdown", error));
    }
    if let Err(error) = capture_result {
        failures.push(("capture finalization", error));
    }
    if let Err(error) = ledger_result {
        failures.push(("ledger finalization", error));
    }
    match failures.len() {
        0 => Ok(()),
        1 => {
            let (stage, error) = failures.pop().expect("one exit failure");
            if stage == "robot control loop" {
                Err(error)
            } else {
                Err(error.context(format!("{stage} failed")))
            }
        }
        _ => anyhow::bail!(
            "robot exit had multiple failures: {}",
            failures
                .into_iter()
                .map(|(stage, error)| format!("{stage}: {error:#}"))
                .collect::<Vec<_>>()
                .join("; ")
        ),
    }
}

async fn append_real_robot_snapshot(
    writer: &mut CaptureWriter,
    snapshot: &WorldSnapshot,
    tick: &RuntimeTick,
) -> Result<()> {
    writer
        .append_snapshot_with_exported_assets(
            tick.frame.now.t_ms,
            snapshot.clone(),
            Vec::new(),
            true,
            true,
            true,
        )
        .await
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

enum PossessionReconnect<C> {
    Reconnected(C),
    Shutdown(&'static str),
}

#[derive(Clone)]
struct PossessionReconnectConfig {
    backend: CockpitBackendArg,
    create_port: Option<String>,
    expected_device_id: Option<String>,
    expected_boot_id: Option<String>,
    max_linear_mm_s: i16,
    max_angular_mrad_s: i16,
}

async fn reconnect_possession_cockpit<Shutdown>(
    create_port: Option<&str>,
    args: &RobotArgs,
    shutdown: Pin<&mut Shutdown>,
) -> Result<PossessionReconnect<Box<dyn Cockpit + Send>>>
where
    Shutdown: Future<Output = &'static str> + ?Sized,
{
    let config = PossessionReconnectConfig {
        backend: args.cockpit,
        create_port: create_port.map(str::to_owned),
        expected_device_id: args.brainstem_device_id.clone(),
        expected_boot_id: args.brainstem_boot_id.clone(),
        max_linear_mm_s: args.max_linear_mm_s,
        max_angular_mrad_s: args.max_angular_mrad_s,
    };
    reconnect_possession_cockpit_with(
        args.reconnect_initial_backoff_ms,
        args.reconnect_max_backoff_ms,
        move || open_ready_possession_cockpit(config.clone()),
        shutdown,
    )
    .await
}

async fn open_ready_possession_cockpit(
    config: PossessionReconnectConfig,
) -> Result<Box<dyn Cockpit + Send>> {
    tokio::task::spawn_blocking(move || {
        let (mut cockpit, mode, _) = open_robot_cockpit_or_fallback(
            config.backend,
            config.create_port.as_deref(),
            RobotMode::Slow,
            config.expected_device_id.as_deref(),
            config.expected_boot_id.as_deref(),
            config.max_linear_mm_s,
            config.max_angular_mrad_s,
        )?;
        if mode != RobotMode::Slow {
            anyhow::bail!("possession reconnect attempted an invalid fallback");
        }
        establish_create_sensor_stream(cockpit.as_mut(), true)?;
        Ok(cockpit)
    })
    .await
    .context("possession reconnect worker failed")?
}

async fn reconnect_possession_cockpit_with<C, Connect, ConnectFuture, Shutdown>(
    initial_backoff_ms: u64,
    maximum_backoff_ms: u64,
    mut connect: Connect,
    mut shutdown: Pin<&mut Shutdown>,
) -> Result<PossessionReconnect<C>>
where
    Connect: FnMut() -> ConnectFuture,
    ConnectFuture: Future<Output = Result<C>>,
    Shutdown: Future<Output = &'static str> + ?Sized,
{
    let mut backoff_ms = initial_backoff_ms.max(1);
    let max_backoff_ms = maximum_backoff_ms.max(backoff_ms).min(60_000);
    loop {
        let attempt = tokio::select! {
            biased;
            signal = shutdown.as_mut() => return Ok(PossessionReconnect::Shutdown(signal)),
            result = connect() => result,
        };
        match attempt {
            Ok(cockpit) => return Ok(PossessionReconnect::Reconnected(cockpit)),
            Err(error) if is_identity_acceptance_error(&error) => return Err(error),
            Err(error) => {
                eprintln!("possession reconnect failed: {error}; retrying in {backoff_ms} ms");
                tokio::select! {
                    biased;
                    signal = shutdown.as_mut() => {
                        return Ok(PossessionReconnect::Shutdown(signal));
                    }
                    _ = tokio::time::sleep(Duration::from_millis(backoff_ms)) => {}
                }
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

fn run_possession_shutdown<C: Cockpit + ?Sized>(cockpit: &mut C) -> Result<()> {
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
    let stop_result = retry_possession_shutdown_command(
        cockpit,
        "possession shutdown STOP",
        attempts,
        delay,
        Cockpit::stop,
    );
    let exorcize_result = retry_possession_shutdown_command(
        cockpit,
        "possession exorcize",
        attempts,
        delay,
        Cockpit::exorcize,
    );
    match (stop_result, exorcize_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(stop_error), Err(exorcize_error)) => anyhow::bail!(
            "possession STOP and exorcize both failed: STOP: {stop_error:#}; exorcize: {exorcize_error:#}"
        ),
    }
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
