#[test]
fn safe_cockpit_clamps_motion_to_body_limits() {
    let mut caps = sim_caps_with_all_verbs();
    caps.limits.max_linear_mm_s = 40;
    caps.limits.max_angular_mrad_s = 100;
    caps.limits.min_ttl_ms = 50;
    caps.limits.max_ttl_ms = 200;
    let sim = SimCockpit::new()
        .with_unscoped_bench_mode()
        .with_capabilities(caps);
    let mut safe = SafeCockpit::with_policy(
        sim,
        AgentPolicy {
            motion_ttl_ms: 500,
            heartbeat_timeout_ms: 500,
        },
    );
    safe.pulse_motion(120, 300).unwrap();
    let batch = safe.client_mut().get_events_since(0).unwrap();
    let motion = batch
        .events
        .iter()
        .find(|event| event.kind == CockpitEventKind::MotionRequested)
        .unwrap();
    assert_eq!(motion.a, pack_i16_pair(40, 100));
    assert_eq!(motion.b, 200);
}

#[test]
fn safe_cockpit_reports_preexisting_bump_latch_as_typed_motion_stop() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.set_bump(true, false);
    let mut safe = SafeCockpit::with_policy(
        sim,
        AgentPolicy {
            motion_ttl_ms: 100,
            heartbeat_timeout_ms: 0,
        },
    );

    assert!(matches!(
        safe.pulse_motion(20, 0),
        Err(CockpitError::MotionStopped { reasons })
            if reasons == vec![SafeStopReason::SafetyTripped {
                latch: Some(SafetyLatchKind::Bump),
            }]
    ));
}

#[test]
fn safe_cockpit_requires_heartbeat_only_when_policy_uses_it() {
    let caps = sim_caps_without(&["heartbeat_stop"]);
    let sim = SimCockpit::new()
        .with_unscoped_bench_mode()
        .with_capabilities(caps.clone());
    let mut safe = SafeCockpit::new(sim);
    assert!(matches!(
        safe.pulse_motion(20, 0),
        Err(CockpitError::Policy(message)) if message.contains("heartbeat_stop")
    ));

    let sim = SimCockpit::new()
        .with_unscoped_bench_mode()
        .with_capabilities(caps);
    let mut safe = SafeCockpit::with_policy(
        sim,
        AgentPolicy {
            motion_ttl_ms: 100,
            heartbeat_timeout_ms: 0,
        },
    );
    safe.pulse_motion(20, 0).unwrap();
}

#[test]
fn safe_cockpit_does_not_treat_historical_command_rejection_as_motion_stop() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.push_event(CockpitEventKind::CommandRejected, 7, 0, 0);
    let mut safe = SafeCockpit::with_policy(
        sim,
        AgentPolicy {
            motion_ttl_ms: 100,
            heartbeat_timeout_ms: 0,
        },
    );

    safe.pulse_motion(20, 0).unwrap();
}

#[test]
fn uart_config_defaults_to_forebrain_baud() {
    let config = UartCockpitConfig::new("/dev/ttyTEST0");
    assert_eq!(config.baud_rate, DEFAULT_UART_BAUD_RATE);
    assert_eq!(config.timeout, DEFAULT_UART_TIMEOUT);
    assert_eq!(config.max_response_len, DEFAULT_UART_MAX_RESPONSE_LEN);
}

#[test]
fn malformed_response_maps_to_bad_response() {
    let err = expect_ok(2, "ERR 2 parse").unwrap_err();
    assert!(matches!(err, CockpitError::BadResponse(_)));
}

#[test]
fn mismatched_sequence_maps_to_bad_response() {
    let err = expect_ok(1, "OK 12").unwrap_err();
    assert!(matches!(err, CockpitError::BadResponse(_)));
}

#[test]
fn non_utf8_response_maps_to_bad_response() {
    let err = response_from_bytes(&[0xff]).unwrap_err();
    assert!(matches!(err, CockpitError::BadResponse(_)));
}

fn sample_cockpit_requests() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        ("ping", "ping", "PING"),
        ("bootsel", "bootsel", "BOOTSEL"),
        ("restart_create", "restart_create", "RESTART_CREATE"),
        ("status", "status", "STATUS"),
        ("get_capabilities", "get_capabilities", "GET_CAPABILITIES"),
        ("get_events", "get_events", "GET_EVENTS"),
        ("arm", "arm", "ARM"),
        ("disarm", "disarm", "DISARM"),
        ("stop", "stop", "STOP"),
        ("estop", "estop", "ESTOP"),
        ("clear_estop", "clear_estop", "CLEAR_ESTOP"),
        (
            "clear_safety_latch",
            "clear_safety_latch",
            "CLEAR_SAFETY_LATCH",
        ),
        ("careful_mode", "careful_mode", "CAREFUL_MODE"),
        ("escape_motion", "escape_motion", "ESCAPE_MOTION"),
        (
            "clear_motion_queue",
            "clear_motion_queue",
            "CLEAR_MOTION_QUEUE",
        ),
        ("cmd_vel", "cmd_vel", "CMD_VEL"),
        ("drive_direct", "drive_direct", "DRIVE_DIRECT"),
        ("drive_arc", "drive_arc", "DRIVE_ARC"),
        ("drive_for", "drive_for", "DRIVE_FOR"),
        ("turn_by", "turn_by", "TURN_BY"),
        ("arc_for", "arc_for", "ARC_FOR"),
        ("creep_until", "creep_until", "CREEP_UNTIL"),
        ("scan_arc", "scan_arc", "SCAN_ARC"),
        ("face_bearing", "face_bearing", "FACE_BEARING"),
        ("track_bearing", "track_bearing", "TRACK_BEARING"),
        ("hold_heading", "hold_heading", "HOLD_HEADING"),
        ("turn_to_heading", "turn_to_heading", "TURN_TO_HEADING"),
        ("dock_align", "dock_align", "DOCK_ALIGN"),
        ("wall_follow", "wall_follow", "WALL_FOLLOW"),
        ("wiggle_align", "wiggle_align", "WIGGLE_ALIGN"),
        ("bump_escape", "bump_escape", "BUMP_ESCAPE"),
        ("unstick", "unstick", "UNSTICK"),
        ("cliff_guard", "cliff_guard", "CLIFF_GUARD"),
        ("heartbeat_stop", "heartbeat_stop", "HEARTBEAT_STOP"),
        ("request_sensors", "request_sensors", "REQUEST_SENSORS"),
        ("stream_sensors", "stream_sensors", "STREAM_SENSORS"),
        (
            "set_safety_policy",
            "set_safety_policy",
            "SET_SAFETY_POLICY",
        ),
        ("song_define", "song_define", "SONG_DEFINE"),
        ("song_play", "song_play", "SONG_PLAY"),
        ("define_chirp", "define_chirp", "DEFINE_CHIRP"),
        ("play_feedback", "play_feedback", "PLAY_FEEDBACK"),
        ("set_silent", "set_silent", "SET_SILENT"),
        ("power_state", "power_state", "POWER_STATE"),
        ("create_power_on", "create_power_on", "CREATE_POWER_ON"),
        ("create_power_off", "create_power_off", "CREATE_POWER_OFF"),
        ("calibrate_turn", "calibrate_turn", "CALIBRATE_TURN"),
        (
            "orientation_probe",
            "orientation_probe",
            "ORIENTATION_PROBE",
        ),
        ("reset_odometry", "reset_odometry", "RESET_ODOMETRY"),
        (
            "zero_imu_orientation",
            "zero_imu_orientation",
            "ZERO_IMU_ORIENTATION",
        ),
        (
            "clear_imu_orientation",
            "clear_imu_orientation",
            "CLEAR_IMU_ORIENTATION",
        ),
        ("dock", "dock", "DOCK"),
        ("set_lights", "set_lights", "SET_LIGHTS"),
        ("set_mode", "set_mode", "SET_MODE"),
    ]
}

fn body_toml() -> toml::Value {
    include_str!("../../../pete-brainstem/body.toml")
        .parse()
        .unwrap()
}

fn body_toml_array(key: &str) -> Vec<&'static str> {
    let body = body_toml();
    let values = body["capabilities"][key].as_array().unwrap();
    values
        .iter()
        .map(|value| {
            let value = value.as_str().unwrap().to_owned();
            Box::leak(value.into_boxed_str()) as &'static str
        })
        .collect()
}

fn body_toml_capabilities() -> CockpitCapabilities {
    let body = body_toml();
    let limits = &body["limits"];
    CockpitCapabilities {
        body_kind: body["body"]["kind"].as_str().unwrap().to_owned(),
        drive: body["body"]["drive"].as_str().unwrap().to_owned(),
        verbs: body_toml_array("verbs")
            .into_iter()
            .map(ToOwned::to_owned)
            .collect(),
        sensors: body_toml_array("sensors")
            .into_iter()
            .map(ToOwned::to_owned)
            .collect(),
        outputs: body_toml_array("outputs")
            .into_iter()
            .map(ToOwned::to_owned)
            .collect(),
        safety: body_toml_array("safety")
            .into_iter()
            .map(ToOwned::to_owned)
            .collect(),
        events: body_toml_array("events")
            .into_iter()
            .map(ToOwned::to_owned)
            .collect(),
        independent_watchdog: Some(true),
        limits: CockpitLimits {
            max_linear_mm_s: limits["max_linear_mm_s"]
                .as_integer()
                .unwrap()
                .try_into()
                .unwrap(),
            max_angular_mrad_s: limits["max_angular_mrad_s"]
                .as_integer()
                .unwrap()
                .try_into()
                .unwrap(),
            min_ttl_ms: limits["min_ttl_ms"]
                .as_integer()
                .unwrap()
                .try_into()
                .unwrap(),
            max_ttl_ms: limits["max_ttl_ms"]
                .as_integer()
                .unwrap()
                .try_into()
                .unwrap(),
        },
    }
}

fn sim_caps_with_all_verbs() -> CockpitCapabilities {
    let mut caps = SimCockpit::new().get_capabilities().unwrap();
    caps.verbs = CockpitRequest::capability_verbs()
        .into_iter()
        .map(ToOwned::to_owned)
        .collect();
    caps.events = body_toml_array("events")
        .into_iter()
        .map(ToOwned::to_owned)
        .collect();
    caps
}

fn sim_caps_without(without: &[&str]) -> CockpitCapabilities {
    let mut caps = sim_caps_with_all_verbs();
    caps.verbs.retain(|verb| !without.contains(&verb.as_str()));
    caps
}

fn sample_request_for(verb: &str) -> CockpitRequest {
    match verb {
        "ping" => CockpitRequest::Ping,
        "bootsel" => CockpitRequest::Bootsel,
        "restart_create" => CockpitRequest::RestartCreate,
        "reset_motherbrain" => CockpitRequest::ResetMotherbrain,
        "status" => CockpitRequest::GetStatus,
        "get_capabilities" => CockpitRequest::GetCapabilities,
        "get_events" => CockpitRequest::GetEvents { since_seq: 3 },
        "arm" => CockpitRequest::Arm,
        "disarm" => CockpitRequest::Disarm,
        "stop" => CockpitRequest::Stop,
        "estop" => CockpitRequest::EStop,
        "clear_estop" => CockpitRequest::ClearEStop,
        "clear_safety_latch" => CockpitRequest::ClearSafetyLatch {
            latch: SafetyLatchKind::Bump,
        },
        "careful_mode" => CockpitRequest::CarefulMode { ttl_ms: 5_000 },
        "escape_motion" => CockpitRequest::EscapeMotion {
            hazard: SafetyLatchKind::Bump,
            hazard_generation: 42,
            linear_mm_s: -50,
            angular_mrad_s: 0,
            ttl_ms: 250,
        },
        "clear_motion_queue" => CockpitRequest::ClearMotionQueue,
        "cmd_vel" => CockpitRequest::CmdVel {
            linear_mm_s: 10,
            angular_mrad_s: 20,
            ttl_ms: 300,
        },
        "drive_direct" => CockpitRequest::DriveDirect {
            left_mm_s: 10,
            right_mm_s: 11,
            ttl_ms: 300,
        },
        "drive_arc" => CockpitRequest::DriveArc {
            velocity_mm_s: 10,
            radius_mm: 200,
            ttl_ms: 300,
        },
        "drive_for" => CockpitRequest::DriveFor {
            distance_mm: 300,
            velocity_mm_s: 80,
            timeout_ms: 2_000,
        },
        "turn_by" => CockpitRequest::TurnBy {
            angle_mrad: 1_570,
            angular_mrad_s: 800,
            timeout_ms: 2_000,
        },
        "arc_for" => CockpitRequest::ArcFor {
            velocity_mm_s: 80,
            radius_mm: 250,
            duration_ms: 1_000,
        },
        "creep_until" => CockpitRequest::CreepUntil {
            velocity_mm_s: 40,
            angular_mrad_s: 0,
            timeout_ms: 1_000,
        },
        "scan_arc" => CockpitRequest::ScanArc {
            angle_mrad: 3_140,
            angular_mrad_s: 500,
            timeout_ms: 4_000,
        },
        "face_bearing" => CockpitRequest::FaceBearing {
            bearing_mrad: 100,
            max_angular_mrad_s: 500,
            tolerance_mrad: 35,
            ttl_ms: 300,
        },
        "track_bearing" => CockpitRequest::TrackBearing {
            bearing_mrad: 100,
            range_mm: 900,
            max_linear_mm_s: 120,
            max_angular_mrad_s: 500,
            stop_range_mm: 250,
            ttl_ms: 300,
        },
        "hold_heading" => CockpitRequest::HoldHeading {
            heading_error_mrad: 100,
            velocity_mm_s: 80,
            max_angular_mrad_s: 500,
            ttl_ms: 300,
        },
        "turn_to_heading" => CockpitRequest::TurnToHeading {
            heading_error_mrad: 100,
            angular_mrad_s: 500,
            tolerance_mrad: 35,
            timeout_ms: 2_000,
        },
        "dock_align" => CockpitRequest::DockAlign {
            bearing_mrad: 50,
            range_mm: 600,
            max_linear_mm_s: 80,
            max_angular_mrad_s: 500,
            stop_range_mm: 250,
            ttl_ms: 300,
        },
        "wall_follow" => CockpitRequest::WallFollow {
            distance_error_mm: 20,
            velocity_mm_s: 80,
            max_angular_mrad_s: 400,
            ttl_ms: 300,
        },
        "wiggle_align" => CockpitRequest::WiggleAlign {
            amplitude_mrad: 200,
            angular_mrad_s: 500,
            cycles: 2,
        },
        "bump_escape" => CockpitRequest::BumpEscape {
            direction: EscapeDirection::Either,
            backoff_mm_s: 80,
            turn_angular_mrad_s: 900,
        },
        "unstick" => CockpitRequest::Unstick {
            direction: EscapeDirection::Either,
            backoff_mm_s: 90,
            turn_angular_mrad_s: 900,
        },
        "cliff_guard" => CockpitRequest::CliffGuard { clear: false },
        "heartbeat_stop" => CockpitRequest::HeartbeatStop { timeout_ms: 900 },
        "request_sensors" => CockpitRequest::RequestSensors { packet_id: 0 },
        "stream_sensors" => CockpitRequest::StreamSensors {
            enabled: true,
            packet_id: 0,
            period_ms: 250,
        },
        "set_safety_policy" => CockpitRequest::SetSafetyPolicy {
            policy: SafetyPolicy {
                bump: SafetyAction::Stop,
                cliff: SafetyAction::Stop,
                wheel_drop_latch: true,
            },
        },
        "song_define" => CockpitRequest::SongDefine {
            id: 1,
            tones: sample_tones(),
        },
        "song_play" => CockpitRequest::SongPlay { id: 1 },
        "define_chirp" => CockpitRequest::DefineChirp {
            feedback: FeedbackKind::Ok,
            tones: sample_tones(),
        },
        "play_feedback" => CockpitRequest::PlayFeedback {
            feedback: FeedbackKind::Ok,
        },
        "set_silent" => CockpitRequest::SetAudioSilent { silent: true },
        "power_state" => CockpitRequest::PowerState {
            request: PowerStateRequest::Wake,
        },
        "create_power_on" => CockpitRequest::CreatePowerOn,
        "create_power_off" => CockpitRequest::CreatePowerOff,
        "calibrate_turn" => CockpitRequest::CalibrateTurn {
            angular_mrad_s: 500,
            duration_ms: 1_000,
        },
        "orientation_probe" => CockpitRequest::OrientationProbe {
            angular_mrad_s: 250,
            duration_ms: 400,
        },
        "reset_odometry" => CockpitRequest::ResetOdometry,
        "zero_imu_orientation" => CockpitRequest::ZeroImuOrientation,
        "clear_imu_orientation" => CockpitRequest::ClearImuOrientation,
        "dock" => CockpitRequest::Dock,
        "set_lights" => CockpitRequest::SetLights {
            pattern: LightPattern::Status,
        },
        "set_mode" => CockpitRequest::SetMode {
            mode: CreateOiMode::Safe,
        },
        other => panic!("missing sample for {other}"),
    }
}

fn sample_tones() -> Vec<SongTone> {
    vec![SongTone {
        note: 72,
        duration_64ths: 8,
    }]
}
