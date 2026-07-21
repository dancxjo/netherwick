const POSSESSION_CONTROL_LEASE_TTL_MS: u32 = 60_000;

fn open_robot_cockpit_or_fallback(
    backend: CockpitBackendArg,
    create_port: Option<&str>,
    robot_mode: RobotMode,
    expected_device_id: Option<&str>,
    expected_boot_id: Option<&str>,
    max_linear_mm_s: i16,
    max_angular_mrad_s: i16,
) -> Result<(Box<dyn Cockpit + Send>, RobotMode, bool)> {
    if create_port == Some("mock") {
        if robot_mode == RobotMode::Slow {
            let connector: Box<dyn Cockpit + Send> = Box::new(LocalSimCockpit::new());
            let ready = establish_session(connector, HandshakeHello::default_motherbrain(), None)?;
            let possession =
                MotherbrainPossession::acquire(ready, POSSESSION_CONTROL_LEASE_TTL_MS)?
                    .with_limits(max_linear_mm_s, max_angular_mrad_s);
            return Ok((Box::new(possession), robot_mode, true));
        }
        return Ok((
            Box::new(LocalSimCockpit::new().with_unscoped_bench_mode()),
            robot_mode,
            true,
        ));
    }

    let Some(create_port) = create_port else {
        if robot_mode == RobotMode::Slow {
            anyhow::bail!(
                "regular possession requires a stable brainstem USB CDC device; none was found"
            );
        } else {
            println!("warning: no cockpit UART device found; falling back to simulated cockpit");
        }
        return Ok((
            Box::new(LocalSimCockpit::new().with_unscoped_bench_mode()),
            robot_mode,
            true,
        ));
    };

    let opened: pete_cockpit::Result<Box<dyn Cockpit + Send>> = match backend {
        CockpitBackendArg::Wifi => Ok(Box::new(HttpCockpit::connect(create_port))),
        CockpitBackendArg::Uart => UartCockpit::connect(create_port)
            .map(|cockpit| Box::new(cockpit) as Box<dyn Cockpit + Send>),
        CockpitBackendArg::Local => create_port
            .parse()
            .map_err(|error| CockpitError::BadResponse(format!("invalid local address: {error}")))
            .and_then(UdpCockpit::connect)
            .map(|cockpit| Box::new(cockpit) as Box<dyn Cockpit + Send>),
        CockpitBackendArg::Sim => unreachable!("sim resolves to mock"),
    };
    match opened {
        Ok(cockpit) if robot_mode == RobotMode::Slow => {
            if backend == CockpitBackendArg::Uart && !create_port.starts_with("/dev/serial/by-id/")
            {
                anyhow::bail!(
                    "regular possession requires a stable /dev/serial/by-id brainstem path, got {create_port}"
                );
            }
            let expected_device_id = expected_device_id.context(
                "regular possession requires --brainstem-device-id to prevent identity fallback",
            )?;
            let expected_boot_id = expected_boot_id.context(
                "regular possession requires --brainstem-boot-id; a boot change needs explicit acceptance",
            )?;
            let ready = establish_session(cockpit, HandshakeHello::default_motherbrain(), None)?;
            if ready.session().peer_device_id != expected_device_id {
                anyhow::bail!(
                    "brainstem identity mismatch: expected {expected_device_id}, received {}",
                    ready.session().peer_device_id
                );
            }
            if ready.session().peer_boot_id != expected_boot_id {
                anyhow::bail!(
                    "brainstem boot identity mismatch: expected {expected_boot_id}, received {}",
                    ready.session().peer_boot_id
                );
            }
            let possession =
                MotherbrainPossession::acquire(ready, POSSESSION_CONTROL_LEASE_TTL_MS)?
                    .with_limits(max_linear_mm_s, max_angular_mrad_s);
            Ok((Box::new(possession), robot_mode, false))
        }
        Ok(cockpit) => {
            let session =
                establish_diagnostic_session(cockpit, HandshakeHello::default_motherbrain(), None)?;
            Ok((Box::new(session), robot_mode, false))
        }
        Err(error) => {
            if robot_mode == RobotMode::Slow {
                anyhow::bail!("failed to open possession brainstem {create_port}: {error}");
            } else {
                println!(
                    "warning: failed to open cockpit UART device {create_port}: {error}; falling back to simulated cockpit"
                );
            }
            Ok((
                Box::new(LocalSimCockpit::new().with_unscoped_bench_mode()),
                robot_mode,
                true,
            ))
        }
    }
}

fn default_runtime(
    ledger: JsonlLedger,
    llm_args: &LlmArgs,
) -> Result<
    MinimalRuntime<
        JsonlLedger,
        InMemoryExperienceStore,
        InMemoryExperienceStore,
        SimpleConductor,
        SimpleSafety,
        ConfiguredLlmAgent,
    >,
> {
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    Ok(MinimalRuntime::with_default_events(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        configured_llm_agent(llm_args)?,
    ))
}

fn durable_runtime(
    ledger: JsonlLedger,
    llm_args: &LlmArgs,
) -> Result<
    MinimalRuntime<
        JsonlLedger,
        DurableExperienceStore,
        DurableExperienceStore,
        SimpleConductor,
        SimpleSafety,
        ConfiguredLlmAgent,
    >,
> {
    let memory = DurableExperienceStore::from_env();
    let recall = memory.clone();
    Ok(MinimalRuntime::with_default_events(
        ledger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        configured_llm_agent(llm_args)?,
    ))
}

async fn real_robot_frame_processor(warnings: &mut Vec<String>) -> FrameProcessor {
    let processor = FrameProcessor::new()
        .with_kinect_range_projection(real_robot_depth_range_projection_from_env());
    if std::env::var("PETE_FACE_DETECTION")
        .map(|value| matches!(value.as_str(), "0" | "false" | "FALSE" | "off" | "OFF"))
        .unwrap_or(false)
    {
        warnings.push("face detection disabled by PETE_FACE_DETECTION".to_string());
        return processor;
    }

    match pete_sensors::FaceIdDetector::from_hf().await {
        Ok(detector) => processor.with_face_detector(std::sync::Arc::new(detector)),
        Err(error) => {
            warnings.push(format!(
                "face detection unavailable; continuing without face vectors: {error}"
            ));
            processor
        }
    }
}

fn real_robot_depth_range_projection_from_env() -> DepthRangeProjectionConfig {
    let calibration = real_robot_depth_calibration_from_env();
    DepthRangeProjectionConfig {
        compact_depth_beam_count: calibration.compact_depth_beam_count,
        compact_depth_fov_rad: calibration.compact_depth_fov_rad,
        depth_scale: calibration.depth_scale,
        camera_forward_m: calibration.camera_forward_m,
        camera_height_m: calibration.camera_height_m,
        camera_pitch_rad: calibration.camera_pitch_rad,
        camera_roll_rad: calibration.camera_roll_rad,
        camera_yaw_rad: calibration.camera_yaw_rad,
        min_depth_m: 0.35,
        max_depth_m: 8.0,
    }
}

fn duration_to_steps(duration_seconds: u64, tick_ms: u64) -> usize {
    let tick_ms = tick_ms.max(1);
    let total_ms = duration_seconds.saturating_mul(1000);
    total_ms.div_ceil(tick_ms).max(1) as usize
}

fn remaining_tick_delay(tick_ms: u64, elapsed: Duration) -> Duration {
    Duration::from_millis(tick_ms.max(1)).saturating_sub(elapsed)
}

fn add_optional_real_sensors(
    args: &CaptureRealArgs,
    env_report: &Value,
    create_port: Option<&str>,
    lidar_device: Option<&str>,
    sensors: &mut Vec<Box<dyn SenseProducer + Send>>,
    availability: &mut Value,
    warnings: &mut Vec<String>,
) {
    if let Some(device) = lidar_device {
        if create_port
            .map(|create| same_serial_device(create, device))
            .unwrap_or(false)
        {
            availability["lidar"] = serde_json::json!({
                "present": false,
                "device": device,
                "reason": "same serial device selected for the brainstem cockpit"
            });
            warnings.push(format!(
                "lidar {device} conflicts with the selected brainstem cockpit device"
            ));
        } else {
            let extrinsics = lidar_extrinsics(
                args.lidar_forward_m,
                args.lidar_left_m,
                args.lidar_height_m,
                args.lidar_roll_deg,
                args.lidar_pitch_deg,
                args.lidar_yaw_deg,
            );
            match Lfcd2SenseProvider::with_extrinsics(device, extrinsics) {
                Ok(provider) => {
                    sensors.push(Box::new(provider));
                    availability["lidar"] = serde_json::json!({
                        "present": true,
                        "device": device,
                        "kind": "hls-lfcd2",
                        "baud": Lfcd2SenseProvider::BAUD_RATE,
                        "extrinsics": extrinsics
                    });
                }
                Err(error) => {
                    availability["lidar"] = serde_json::json!({
                        "present": false,
                        "device": device,
                        "error": error.to_string()
                    });
                    warnings.push(format!("HLS-LFCD2 lidar unavailable: {error}"));
                }
            }
        }
    } else {
        availability["lidar"] =
            serde_json::json!({"present": false, "reason": "disabled or not detected"});
    }

    if args.kinect_depth {
        if let Some(device) = &args.camera {
            warnings.push(format!(
                "Kinect depth requested; using libfreenect for Kinect RGB/depth instead of opening {device} through V4L"
            ));
        }
    } else if let Some(device) = &args.camera {
        match CameraSenseProvider::new(device) {
            Ok(provider) => {
                sensors.push(Box::new(provider));
                availability["camera"] = serde_json::json!({"present": true, "device": device});
            }
            Err(error) => {
                availability["camera"] = serde_json::json!({"present": false, "device": device, "error": error.to_string()});
                warnings.push(format!("camera unavailable: {error}"));
            }
        }
    } else {
        availability["camera"] = serde_json::json!({"present": false, "reason": "not requested"});
        warnings.push("camera not requested; RGB stream missing".to_string());
    }

    if args.kinect_depth {
        #[cfg(feature = "kinect-freenect")]
        match FreenectKinectProvider::with_index(args.kinect_index)
            .map(|provider| provider.with_rgb_adjustment(kinect_rgb_adjustment_for_capture(&args)))
        {
            Ok(provider) => {
                sensors.push(Box::new(provider));
                availability["kinect"] = serde_json::json!({
                    "present": true,
                    "source": "libfreenect",
                    "index": args.kinect_index,
                    "rgb_adjustment": kinect_rgb_adjustment_json(kinect_rgb_adjustment_for_capture(args))
                });
                availability["camera"] = serde_json::json!({
                    "present": true,
                    "source": "libfreenect",
                    "index": args.kinect_index,
                    "rgb_adjustment": kinect_rgb_adjustment_json(kinect_rgb_adjustment_for_capture(args))
                });
            }
            Err(error) => {
                availability["kinect"] = serde_json::json!({
                    "present": false,
                    "source": "libfreenect",
                    "index": args.kinect_index,
                    "error": error.to_string()
                });
                warnings.push(format!("Kinect depth unavailable: {error}"));
            }
        }
        #[cfg(not(feature = "kinect-freenect"))]
        {
            availability["kinect"] = serde_json::json!({
                "present": false,
                "reason": "pete-tools was built without kinect-freenect"
            });
            warnings.push(
                "Kinect depth requested but binary was built without kinect-freenect".to_string(),
            );
        }
    }

    if let Some(device) = &args.mic {
        let pref_name = (device != "default").then_some(device.as_str());
        match MicrophoneSenseProvider::new(pref_name) {
            Ok(provider) => {
                sensors.push(Box::new(provider));
                availability["microphone"] = serde_json::json!({"present": true, "device": device});
            }
            Err(error) => {
                availability["microphone"] = serde_json::json!({"present": false, "device": device, "error": error.to_string()});
                warnings.push(format!("microphone unavailable: {error}"));
            }
        }
    } else {
        availability["microphone"] =
            serde_json::json!({"present": false, "reason": "not requested"});
        warnings.push("microphone not requested; audio stream missing".to_string());
    }

    if let Some(device) = selected_gps_device(args.gps.as_deref(), false, env_report, create_port) {
        match GpsSenseProvider::new(&device, 9600) {
            Ok(provider) => {
                sensors.push(Box::new(provider));
                availability["gps"] = serde_json::json!({"present": true, "device": device});
            }
            Err(error) => {
                availability["gps"] = serde_json::json!({"present": false, "device": device, "error": error.to_string()});
                warnings.push(format!("gps unavailable: {error}"));
            }
        }
    } else {
        availability["gps"] =
            serde_json::json!({"present": false, "reason": "disabled or not detected"});
    }

    if let Some(device) = selected_imu_device(args.imu.as_deref(), false) {
        match ImuSenseProvider::new(device) {
            Ok(provider) => {
                sensors.push(Box::new(provider));
                availability["imu"] = serde_json::json!({"present": true, "device": device});
            }
            Err(error) => {
                availability["imu"] = serde_json::json!({"present": false, "device": device, "error": error.to_string()});
                warnings.push(format!("imu unavailable: {error}"));
            }
        }
    } else {
        availability["imu"] = serde_json::json!({"present": false, "reason": "disabled"});
    }

    if availability["kinect"]["present"].as_bool() != Some(true)
        && availability["kinect"]["freenect_available"].as_bool() != Some(true)
    {
        warnings.push("Kinect/libfreenect not detected; depth stream missing".to_string());
    }
}

fn selected_create_port(
    requested: &str,
    env_report: &Value,
    reserved_lidar_port: Option<&str>,
) -> Option<String> {
    if requested != "auto" {
        return Some(requested.to_string());
    }
    let mut candidates = serial_device_strings(env_report)
        .into_iter()
        .filter(|device| {
            !looks_like_gps_serial_device(device)
                && !looks_like_lidar_serial_device(device)
                && reserved_lidar_port
                    .map(|reserved| !same_serial_device(device, reserved))
                    .unwrap_or(true)
        })
        .collect::<Vec<_>>();
    candidates.sort_by_key(|device| create_serial_priority(device));
    candidates.into_iter().next()
}

fn selected_cockpit_endpoint(
    backend: CockpitBackendArg,
    requested: &str,
    brainstem_host: &str,
    brainstem_local: SocketAddr,
    env_report: &Value,
    reserved_lidar_port: Option<&str>,
) -> Option<String> {
    match backend {
        CockpitBackendArg::Sim => Some("mock".to_string()),
        CockpitBackendArg::Uart => selected_create_port(requested, env_report, reserved_lidar_port),
        CockpitBackendArg::Wifi => Some(brainstem_host.to_string()),
        CockpitBackendArg::Local => Some(brainstem_local.to_string()),
    }
}

fn create_serial_priority(device: &str) -> u8 {
    if device.contains("/dev/serial/by-id") {
        0
    } else if device.contains("/dev/ttyUSB") {
        1
    } else if device.contains("/dev/ttyACM") {
        2
    } else {
        3
    }
}

fn selected_gps_device(
    requested: Option<&str>,
    suppress_default: bool,
    env_report: &Value,
    create_port: Option<&str>,
) -> Option<String> {
    match requested.map(str::trim) {
        Some(value) if gps_disabled_value(value) => return None,
        Some(value) if !value.is_empty() => return Some(value.to_string()),
        _ if suppress_default => return None,
        _ => {}
    }

    let available = serial_device_strings(env_report)
        .into_iter()
        .filter(|device| {
            create_port
                .map(|port| port != device.as_str())
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();
    available
        .iter()
        .find(|device| looks_like_gps_serial_device(device))
        .cloned()
        .or_else(|| {
            available
                .iter()
                .find(|device| device.contains("/dev/ttyACM"))
                .cloned()
        })
}

fn serial_device_strings(env_report: &Value) -> Vec<String> {
    env_report["serial_devices"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect()
}

fn looks_like_gps_serial_device(device: &str) -> bool {
    let lower = device.to_lowercase();
    lower.contains("u-blox")
        || lower.contains("ublox")
        || lower.contains("gps")
        || lower.contains("gnss")
}

fn selected_lidar_device(
    requested: Option<&str>,
    suppress_default: bool,
    env_report: &Value,
) -> Option<String> {
    match requested.map(str::trim) {
        Some(value) if serial_sensor_disabled_value(value) => return None,
        Some(value) if !value.is_empty() => return Some(value.to_string()),
        _ if suppress_default => return None,
        _ => {}
    }
    serial_device_strings(env_report)
        .into_iter()
        .find(|device| looks_like_lidar_serial_device(device))
}

fn lidar_extrinsics(
    forward_m: f32,
    left_m: f32,
    height_m: f32,
    roll_deg: f32,
    pitch_deg: f32,
    yaw_deg: f32,
) -> RangeExtrinsics {
    RangeExtrinsics {
        forward_m,
        left_m,
        height_m,
        roll_rad: roll_deg.to_radians(),
        pitch_rad: pitch_deg.to_radians(),
        yaw_rad: yaw_deg.to_radians(),
    }
}

fn looks_like_lidar_serial_device(device: &str) -> bool {
    let lower = device.to_lowercase();
    lower.contains("hls-lfcd")
        || lower.contains("hls_lfcd")
        || lower.contains("lfcd2")
        || lower.contains("usb2lds")
        || lower.contains("lidar")
        || lower.contains("lds-01")
        || lower.contains("lds_01")
}

fn same_serial_device(left: &str, right: &str) -> bool {
    if left == right {
        return true;
    }
    std::fs::canonicalize(left)
        .ok()
        .zip(std::fs::canonicalize(right).ok())
        .map(|(left, right)| left == right)
        .unwrap_or(false)
}

fn serial_sensor_disabled_value(value: &str) -> bool {
    value.eq_ignore_ascii_case("none")
        || value.eq_ignore_ascii_case("off")
        || value.eq_ignore_ascii_case("disabled")
}

fn gps_disabled_value(value: &str) -> bool {
    serial_sensor_disabled_value(value)
}

fn selected_imu_device(requested: Option<&str>, suppress_default: bool) -> Option<&str> {
    match requested.map(str::trim) {
        Some(value) if imu_disabled_value(value) => None,
        Some(value) if !value.is_empty() => Some(value),
        _ if suppress_default => None,
        _ => Some(DEFAULT_MPU6050_IMU_DEVICE),
    }
}

#[cfg(feature = "kinect-freenect")]
fn kinect_rgb_adjustment_for_robot(args: &RobotArgs) -> KinectRgbAdjustment {
    KinectRgbAdjustment {
        enabled: !args.kinect_rgb_raw,
        gain: args.kinect_rgb_gain,
        gamma: args.kinect_rgb_gamma,
        target_luma: args.kinect_rgb_target_luma,
        auto_gain_max: args.kinect_rgb_auto_gain_max,
        brightness: args.kinect_rgb_brightness,
    }
}

#[cfg(feature = "kinect-freenect")]
fn kinect_rgb_adjustment_for_capture(args: &CaptureRealArgs) -> KinectRgbAdjustment {
    KinectRgbAdjustment {
        enabled: !args.kinect_rgb_raw,
        gain: args.kinect_rgb_gain,
        gamma: args.kinect_rgb_gamma,
        target_luma: args.kinect_rgb_target_luma,
        auto_gain_max: args.kinect_rgb_auto_gain_max,
        brightness: args.kinect_rgb_brightness,
    }
}

#[cfg(feature = "kinect-freenect")]
fn kinect_rgb_adjustment_json(adjustment: KinectRgbAdjustment) -> serde_json::Value {
    serde_json::json!({
        "enabled": adjustment.enabled,
        "gain": adjustment.gain,
        "gamma": adjustment.gamma,
        "target_luma": adjustment.target_luma,
        "auto_gain_max": adjustment.auto_gain_max,
        "brightness": adjustment.brightness,
    })
}

fn imu_disabled_value(value: &str) -> bool {
    value.eq_ignore_ascii_case("none")
        || value.eq_ignore_ascii_case("off")
        || value.eq_ignore_ascii_case("disabled")
}

#[derive(Default)]
struct MockEyeProducer {
    tick: u64,
}

#[async_trait::async_trait]
impl SenseProducer for MockEyeProducer {
    async fn poll(&mut self) -> Result<SensePacket> {
        self.tick = self.tick.saturating_add(1);
        let base = (self.tick % 16) as f32 / 16.0;
        let b = (base * 255.0).round() as u8;
        Ok(SensePacket::EyeFrame(EyeFrame {
            captured_at_ms: Utc::now().timestamp_millis().max(0) as u64,
            width: 2,
            height: 2,
            format: EyeFrameFormat::Rgb8,
            bytes: vec![b, 64, 128, 128, b, 64, 64, 128, b, 255, 255, 255],
            source: None,
        }))
    }
}

#[derive(Default)]
struct MockEarProducer {
    tick: u64,
}

#[async_trait::async_trait]
impl SenseProducer for MockEarProducer {
    async fn poll(&mut self) -> Result<SensePacket> {
        self.tick = self.tick.saturating_add(1);
        Ok(if self.tick % 2 == 0 {
            SensePacket::Ear(EarSense {
                schema_version: 1,
                features: vec![vec![0.1, 0.2, 0.1]],
                transcript: None,
                ..EarSense::default()
            })
        } else {
            SensePacket::EarPcm(PcmAudioFrame {
                captured_at_ms: Utc::now().timestamp_millis().max(0) as u64,
                sample_rate_hz: 16_000,
                channels: 1,
                samples: vec![0, 128, -128, 64],
            })
        })
    }
}

#[derive(Default)]
struct MockRangeProducer;

#[async_trait::async_trait]
impl SenseProducer for MockRangeProducer {
    async fn poll(&mut self) -> Result<SensePacket> {
        Ok(SensePacket::Range(RangeSense {
            schema_version: 1,
            beams: vec![1.2, 1.0, 0.8],
            nearest_m: Some(0.8),
            ..RangeSense::default()
        }))
    }
}

#[derive(Default)]
struct MockKinectProducer;

#[async_trait::async_trait]
impl SenseProducer for MockKinectProducer {
    async fn poll(&mut self) -> Result<SensePacket> {
        Ok(SensePacket::Kinect(KinectSense {
            schema_version: 1,
            captured_at_ms: Utc::now().timestamp_millis().max(0) as u64,
            color_features: vec![vec![0.2, 0.4, 0.6]],
            depth_m: vec![0.8, 1.0, 1.2],
            audio_angle_rad: Some(0.0),
            audio_confidence: 0.75,
            ..KinectSense::default()
        }))
    }
}

#[derive(Clone, Debug, Default)]
struct NoopLedger;

#[async_trait::async_trait]
impl LedgerWriter for NoopLedger {
    async fn append(&self, _frame: &ExperienceFrame) -> Result<()> {
        Ok(())
    }

    async fn append_transition(&self, _transition: &ExperienceTransition) -> Result<()> {
        Ok(())
    }
}

async fn collect_hardware_env_report() -> Value {
    let serial_devices = list_matching_paths(&["/dev/ttyUSB", "/dev/ttyACM", "/dev/serial/by-id/"]);
    let gps_serial_candidates = serial_devices
        .iter()
        .filter(|device| looks_like_gps_serial_device(device))
        .cloned()
        .collect::<Vec<_>>();
    let default_gps_device = gps_serial_candidates.first().cloned().or_else(|| {
        serial_devices
            .iter()
            .find(|device| device.contains("/dev/ttyACM"))
            .cloned()
    });
    let lidar_serial_candidates = serial_devices
        .iter()
        .filter(|device| looks_like_lidar_serial_device(device))
        .cloned()
        .collect::<Vec<_>>();
    let default_lidar_device = lidar_serial_candidates.first().cloned();
    let i2c_devices = list_matching_paths(&["/dev/i2c-"]);
    let camera_devices = list_matching_paths(&["/dev/video"]);
    let audio_input_devices = audio_input_devices();
    let warnings = hardware_env_warnings(
        &serial_devices,
        &i2c_devices,
        &camera_devices,
        &audio_input_devices,
    );
    serde_json::json!({
        "os": std::env::consts::OS,
        "architecture": std::env::consts::ARCH,
        "cpu_model": cpu_model(),
        "memory_total_kb": memory_total_kb(),
        "serial_devices": serial_devices,
        "gps_serial_candidates": gps_serial_candidates,
        "default_gps": {
            "kind": "u-blox7",
            "device": default_gps_device,
            "baud": 9600,
            "protocol": "nmea"
        },
        "lidar_serial_candidates": lidar_serial_candidates,
        "default_lidar": {
            "kind": "hls-lfcd2",
            "device": default_lidar_device,
            "baud": Lfcd2SenseProvider::BAUD_RATE,
            "protocol": "lfcd2-42-byte-segments"
        },
        "i2c_devices": i2c_devices,
        "default_imu": {
            "kind": "mpu6050",
            "device": DEFAULT_MPU6050_IMU_DEVICE,
            "address": "0x68",
            "raspberry_pi_header_pins": {
                "sda": 3,
                "scl": 5,
                "power_3v3": 1,
                "ground": 6
            },
            "gpio_bcm": {
                "sda": 2,
                "scl": 3
            },
            "present": Path::new(DEFAULT_MPU6050_IMU_DEVICE).exists()
        },
        "camera_devices": camera_devices,
        "audio_input_devices": audio_input_devices,
        "kinect": {
            "freenect_available": command_exists("freenect-glview") || command_exists("freenect-camtest") || pkg_config_exists("libfreenect"),
            "freenect_glview": command_exists("freenect-glview"),
            "freenect_camtest": command_exists("freenect-camtest"),
            "pkg_config_libfreenect": pkg_config_exists("libfreenect"),
        },
        "permissions": {
            "groups": current_groups(),
            "serial_group_hint": "dialout",
            "i2c_group_hint": "i2c",
            "video_group_hint": "video",
            "audio_group_hint": "audio",
        },
        "data_dirs_writable": {
            "data": directory_writable(Path::new("data")),
            "data/captures/real": directory_writable(Path::new("data/captures/real")),
            "data/ledger/real": directory_writable(Path::new("data/ledger/real")),
        },
        "raspberry_pi_like": raspberry_pi_like(),
        "warnings": warnings,
    })
}

fn hardware_env_warnings(
    serial_devices: &[String],
    i2c_devices: &[String],
    camera_devices: &[String],
    audio_input_devices: &[String],
) -> Vec<String> {
    let mut warnings = Vec::new();
    let groups = current_groups();
    if serial_devices.is_empty() {
        warnings.push("no likely Create serial devices found under /dev/ttyUSB*, /dev/ttyACM*, or /dev/serial/by-id".to_string());
    }
    if i2c_devices.is_empty() {
        warnings.push(
            "no /dev/i2c-* buses found; enable Raspberry Pi I2C before using the MPU-6050"
                .to_string(),
        );
    } else if !i2c_devices
        .iter()
        .any(|device| device == DEFAULT_MPU6050_IMU_DEVICE)
    {
        warnings.push(format!(
            "default MPU-6050 bus {DEFAULT_MPU6050_IMU_DEVICE} not found; pass --imu /dev/i2c-N if your Pi exposes a different bus"
        ));
    }
    if camera_devices.is_empty() {
        warnings.push("no /dev/video* camera devices found".to_string());
    }
    if audio_input_devices.is_empty() {
        warnings.push("no audio input devices detected by arecord or /proc/asound".to_string());
    }
    for group in ["dialout", "i2c", "video", "audio"] {
        if !groups.iter().any(|item| item == group) {
            warnings.push(format!(
                "current user is not in `{group}` group; hardware permissions may fail"
            ));
        }
    }
    warnings
}

fn machine_info_from_env(report: &Value) -> Value {
    serde_json::json!({
        "os": report["os"].clone(),
        "architecture": report["architecture"].clone(),
        "cpu_model": report["cpu_model"].clone(),
        "memory_total_kb": report["memory_total_kb"].clone(),
        "raspberry_pi_like": report["raspberry_pi_like"].clone(),
    })
}

fn cpu_model() -> Option<String> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok()?;
    cpuinfo.lines().find_map(|line| {
        line.strip_prefix("Model")
            .or_else(|| line.strip_prefix("model name"))
            .and_then(|line| {
                line.split_once(':')
                    .map(|(_, value)| value.trim().to_string())
            })
            .filter(|value| !value.is_empty())
    })
}

fn memory_total_kb() -> Option<u64> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    meminfo.lines().find_map(|line| {
        line.strip_prefix("MemTotal:")
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|value| value.parse().ok())
    })
}

fn raspberry_pi_like() -> bool {
    let model = fs::read_to_string("/proc/device-tree/model")
        .or_else(|_| fs::read_to_string("/sys/firmware/devicetree/base/model"))
        .unwrap_or_default()
        .to_lowercase();
    model.contains("raspberry pi")
}

fn list_matching_paths(prefixes: &[&str]) -> Vec<String> {
    let mut paths = Vec::new();
    for prefix in prefixes {
        let path = Path::new(prefix);
        if path.is_dir() {
            if let Ok(entries) = fs::read_dir(path) {
                for entry in entries.flatten() {
                    paths.push(entry.path().to_string_lossy().to_string());
                }
            }
            continue;
        }
        let Some(parent) = path.parent() else {
            continue;
        };
        let Some(name_prefix) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if let Ok(entries) = fs::read_dir(parent) {
            for entry in entries.flatten() {
                if entry
                    .file_name()
                    .to_str()
                    .map(|name| name.starts_with(name_prefix))
                    .unwrap_or(false)
                {
                    paths.push(entry.path().to_string_lossy().to_string());
                }
            }
        }
    }
    paths.sort();
    paths.dedup();
    paths
}

fn audio_input_devices() -> Vec<String> {
    if let Ok(output) = ProcessCommand::new("arecord").arg("-l").output() {
        if output.status.success() {
            return String::from_utf8_lossy(&output.stdout)
                .lines()
                .filter(|line| line.trim_start().starts_with("card "))
                .map(|line| line.trim().to_string())
                .collect();
        }
    }
    let proc_asound = Path::new("/proc/asound/cards");
    fs::read_to_string(proc_asound)
        .ok()
        .map(|text| {
            text.lines()
                .filter(|line| line.contains('[') && line.contains(']'))
                .map(|line| line.trim().to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn command_exists(command: &str) -> bool {
    ProcessCommand::new("sh")
        .arg("-c")
        .arg(format!("command -v {command} >/dev/null 2>&1"))
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn pkg_config_exists(package: &str) -> bool {
    ProcessCommand::new("pkg-config")
        .arg("--exists")
        .arg(package)
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn current_groups() -> Vec<String> {
    ProcessCommand::new("id")
        .arg("-nG")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .split_whitespace()
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn directory_writable(path: &Path) -> bool {
    if fs::create_dir_all(path).is_err() {
        return false;
    }
    let probe = path.join(".pete-write-test");
    match fs::write(&probe, b"ok") {
        Ok(()) => {
            let _ = fs::remove_file(probe);
            true
        }
        Err(_) => false,
    }
}

fn print_json_list(label: &str, value: &Value) {
    println!("{label}:");
    let Some(items) = value.as_array() else {
        println!("    none");
        return;
    };
    if items.is_empty() {
        println!("    none");
        return;
    }
    for item in items {
        if let Some(text) = item.as_str() {
            println!("    - {text}");
        } else {
            println!("    - {item}");
        }
    }
}

#[derive(Clone, Debug, Default)]
struct StreamCounts {
    body: usize,
    rgb: usize,
    depth: usize,
    audio: usize,
    range: usize,
    imu: usize,
    gps: usize,
    kinect: usize,
}

impl StreamCounts {
    fn observe(&mut self, snapshot: &WorldSnapshot) {
        self.body = self.body.saturating_add(1);
        if snapshot.eye_frame.is_some() || !snapshot.eye.frames.is_empty() {
            self.rgb = self.rgb.saturating_add(1);
        }
        if snapshot.ear_pcm.is_some() || !snapshot.ear.features.is_empty() {
            self.audio = self.audio.saturating_add(1);
        }
        if !snapshot.kinect.depth_m.is_empty() {
            self.depth = self.depth.saturating_add(1);
        }
        if !snapshot.range.beams.is_empty() || snapshot.range.nearest_m.is_some() {
            self.range = self.range.saturating_add(1);
        }
        if !snapshot.imu.orientation.is_empty()
            || !snapshot.imu.acceleration.is_empty()
            || !snapshot.imu.angular_velocity.is_empty()
        {
            self.imu = self.imu.saturating_add(1);
        }
        if snapshot.gps.is_some() {
            self.gps = self.gps.saturating_add(1);
        }
        if !snapshot.kinect.color_features.is_empty()
            || !snapshot.kinect.depth_m.is_empty()
            || !snapshot.kinect.skeletons.is_empty()
        {
            self.kinect = self.kinect.saturating_add(1);
        }
    }

    fn streams(&self) -> CaptureStreams {
        let all = [
            ("body", self.body),
            ("rgb", self.rgb),
            ("depth", self.depth),
            ("audio", self.audio),
            ("range", self.range),
            ("imu", self.imu),
            ("gps", self.gps),
            ("kinect", self.kinect),
        ];
        CaptureStreams {
            present: all
                .iter()
                .filter(|(_, count)| *count > 0)
                .map(|(name, _)| (*name).to_string())
                .collect(),
            missing: all
                .iter()
                .filter(|(_, count)| *count == 0)
                .map(|(name, _)| (*name).to_string())
                .collect(),
        }
    }

    fn warnings(&self) -> Vec<String> {
        self.streams()
            .missing
            .into_iter()
            .filter(|name| name != "gps" && name != "imu")
            .map(|name| format!("{name} stream missing"))
            .collect()
    }

    fn useful_stream_count(&self) -> usize {
        [
            self.body,
            self.rgb,
            self.depth,
            self.audio,
            self.range,
            self.imu,
            self.gps,
            self.kinect,
        ]
        .into_iter()
        .filter(|count| *count > 0)
        .count()
    }
}

#[derive(Clone, Debug)]
struct CaptureInspectionReport {
    path: PathBuf,
    frame_count: usize,
    duration_ms: Option<u64>,
    streams_present: Vec<String>,
    streams_missing: Vec<String>,
    first_timestamp_ms: Option<u64>,
    last_timestamp_ms: Option<u64>,
    event_count: usize,
    asset_counts: Vec<(String, usize)>,
    asset_details: Vec<String>,
    warnings: Vec<String>,
}

async fn inspect_capture_report(path: impl AsRef<Path>) -> Result<CaptureInspectionReport> {
    let path = path.as_ref().to_path_buf();
    let reader = CaptureReader::open(&path).await?;
    let frames = reader.read_frames().await?;
    let mut stream_counts = StreamCounts::default();
    let mut event_count = 0usize;
    for frame in &frames {
        stream_counts.observe(&frame.snapshot);
        event_count = event_count.saturating_add(frame.events.len());
    }
    event_count = event_count.saturating_add(count_jsonl_lines(&path.join("events.jsonl"))?);
    let first_timestamp_ms = frames.first().map(|frame| frame.t_ms);
    let last_timestamp_ms = frames.last().map(|frame| frame.t_ms);
    let duration_ms = first_timestamp_ms
        .zip(last_timestamp_ms)
        .map(|(first, last)| last.saturating_sub(first));
    let streams = if reader.manifest().streams.present.is_empty()
        && reader.manifest().streams.missing.is_empty()
    {
        stream_counts.streams()
    } else {
        reader.manifest().streams.clone()
    };
    Ok(CaptureInspectionReport {
        path: path.clone(),
        frame_count: frames.len(),
        duration_ms,
        streams_present: streams.present,
        streams_missing: streams.missing,
        first_timestamp_ms,
        last_timestamp_ms,
        event_count,
        asset_counts: asset_counts(&path),
        asset_details: asset_details(&frames),
        warnings: reader.manifest().warnings.clone(),
    })
}

fn asset_details(frames: &[pete_worldlab::CaptureFrameRecord]) -> Vec<String> {
    let mut details = Vec::new();
    let mut seen = BTreeSet::new();
    for frame in frames {
        let Some(metadata) = frame.stream_metadata.as_ref().and_then(Value::as_object) else {
            continue;
        };
        for kind in ["rgb", "depth", "audio", "pointcloud"] {
            if seen.contains(kind) {
                continue;
            }
            let Some(value) = metadata.get(kind).and_then(Value::as_object) else {
                continue;
            };
            let detail = match kind {
                "rgb" | "depth" => format!(
                    "{kind} metadata: {}x{}, {}",
                    value.get("width").and_then(Value::as_u64).unwrap_or(0),
                    value.get("height").and_then(Value::as_u64).unwrap_or(0),
                    value
                        .get("format")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                ),
                "audio" => format!(
                    "audio metadata: {} Hz, {} channel(s), {}",
                    value
                        .get("sample_rate_hz")
                        .and_then(Value::as_u64)
                        .unwrap_or(0),
                    value.get("channels").and_then(Value::as_u64).unwrap_or(0),
                    value
                        .get("format")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                ),
                "pointcloud" => format!(
                    "pointcloud metadata: {} vertices, {}, calibration {}",
                    value.get("vertices").and_then(Value::as_u64).unwrap_or(0),
                    value
                        .get("format")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown"),
                    value
                        .get("calibration")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                ),
                _ => continue,
            };
            seen.insert(kind);
            details.push(detail);
        }
    }
    details
}

fn asset_counts(root: &Path) -> Vec<(String, usize)> {
    ["rgb", "depth", "audio", "pointcloud"]
        .into_iter()
        .map(|kind| {
            let path = root.join("assets").join(kind);
            (kind.to_string(), count_files(&path))
        })
        .collect()
}

fn count_files(path: &Path) -> usize {
    fs::read_dir(path)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.flatten())
        .filter(|entry| {
            entry
                .file_type()
                .map(|kind| kind.is_file())
                .unwrap_or(false)
        })
        .count()
}

fn count_jsonl_lines(path: &Path) -> Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    Ok(fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .count())
}

fn join_or_none(items: &[String]) -> String {
    if items.is_empty() {
        "none".to_string()
    } else {
        items.join(", ")
    }
}

