#[test]
fn simulator_capabilities_round_trip() {
    let mut sim = SimCockpit::new();
    let caps = sim.get_capabilities().unwrap();
    assert_eq!(caps.body_kind, "sim_create_oi");
    assert_eq!(caps.drive, "differential");
    assert!(caps.verbs.contains(&"cmd_vel".to_owned()));
    assert!(caps.events.contains(&"safety_tripped".to_owned()));
    assert_eq!(caps.limits.max_linear_mm_s, 500);
}

#[test]
fn simulator_audio_state_round_trips_through_status_and_events() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    assert_eq!(
        sim.execute(CockpitRequest::SetAudioSilent { silent: true })
            .unwrap(),
        CockpitResponse::Accepted
    );
    let status = sim.get_status().unwrap().summary();
    assert_eq!(status.audio_silent, Some(true));
    let events = sim.get_events_since(0).unwrap();
    assert!(events
        .events
        .iter()
        .any(|event| { event.kind == CockpitEventKind::AudioStateChanged && event.a == 1 }));
    sim.execute(CockpitRequest::SetAudioSilent { silent: false })
        .unwrap();
    assert_eq!(
        sim.get_status().unwrap().summary().audio_silent,
        Some(false)
    );
}

#[test]
fn audio_session_toggle_preserves_motherbrain_lease_and_active_motion() {
    let mut sim = SimCockpit::new();
    let mother = sim.handshake(hello()).unwrap();
    let mother_lease = match sim
        .execute_in_session(
            &mother.session,
            CockpitRequest::AcquireControlLease {
                authority: ControlAuthority::Motherbrain,
                ttl_ms: 1_000,
            },
        )
        .unwrap()
    {
        CockpitResponse::ControlLeaseGranted(lease) => lease,
        other => panic!("{other:?}"),
    };
    sim.execute_with_lease(&mother.session, &mother_lease, CockpitRequest::Arm)
        .unwrap();
    sim.execute_with_lease(
        &mother.session,
        &mother_lease,
        CockpitRequest::CmdVel {
            linear_mm_s: 100,
            angular_mrad_s: 0,
            ttl_ms: 1_000,
        },
    )
    .unwrap();

    let operator = sim
        .handshake(HandshakeHello::operator("operator-laptop"))
        .unwrap();
    sim.execute_in_session(
        &operator.session,
        CockpitRequest::SetAudioSilent { silent: true },
    )
    .unwrap();

    let status = sim.get_status().unwrap();
    assert_eq!(status.summary().armed, Some(true));
    assert_eq!(status.summary().audio_silent, Some(true));
    assert!(status.raw.contains("active_cmd_vel=true"));
    sim.execute_with_lease(
        &mother.session,
        &mother_lease,
        CockpitRequest::CmdVel {
            linear_mm_s: 80,
            angular_mrad_s: 0,
            ttl_ms: 1_000,
        },
    )
    .expect("audio preference must not replace the motherbrain control lease");
}

#[test]
fn cockpit_request_covers_public_firmware_verbs_from_body_toml() {
    let cockpit_verbs: BTreeSet<_> = sample_cockpit_requests()
        .into_iter()
        .map(|(verb, _, _)| verb)
        .filter(|verb| *verb != "bootsel")
        .collect();
    let firmware_verbs: BTreeSet<_> = body_toml_array("verbs").into_iter().collect();
    assert!(
        firmware_verbs.is_subset(&cockpit_verbs),
        "public firmware verbs must all be modeled by CockpitRequest"
    );
}

#[test]
fn cockpit_event_kind_covers_public_firmware_events_from_body_toml() {
    for event in body_toml_array("events") {
        assert!(
            !matches!(CockpitEventKind::from(event), CockpitEventKind::Unknown(_)),
            "body.toml event {event} is not modeled by CockpitEventKind"
        );
    }
}

#[test]
fn body_toml_capabilities_validate_local_cockpit_model() {
    let contract = CockpitContract::new(body_toml_capabilities());
    let report = contract.validate_local_model();
    assert!(
        report.is_clean(),
        "missing={:?} extra={:?} unknown_events={:?}",
        report.missing_verbs,
        report.extra_verbs,
        report.unknown_events
    );
}

#[test]
fn live_service_verbs_do_not_block_maintenance_handshake() {
    let mut capabilities = body_toml_capabilities();
    capabilities.verbs.push("bootsel".to_owned());
    capabilities.verbs.push("restart_mpu".to_owned());
    let contract = CockpitContract::new(capabilities);
    let report = contract.validate_local_model();

    assert!(
        report.missing_verbs.is_empty(),
        "missing={:?}",
        report.missing_verbs
    );
}

#[test]
fn previous_brainstem_contract_does_not_block_bootsel_handshake() {
    let mut capabilities = body_toml_capabilities();
    capabilities.verbs.extend(
        legacy_brainstem_convenience_verbs()
            .into_iter()
            .map(ToOwned::to_owned),
    );
    establish_session(
        SimCockpit::new().with_capabilities(capabilities),
        HandshakeHello::default_motherbrain(),
        None,
    )
    .expect("older firmware must remain flashable");
}

#[test]
fn pre_careful_brainstem_contract_remains_accepted_for_migration() {
    let mut capabilities = body_toml_capabilities();
    capabilities.verbs.retain(|verb| verb != "careful_mode");

    establish_session(
        SimCockpit::new().with_capabilities(capabilities),
        HandshakeHello::default_motherbrain(),
        None,
    )
    .expect("pre-CAREFUL firmware must remain flashable");
}

#[test]
fn pre_escape_motion_brainstem_contract_remains_accepted_for_migration() {
    let mut capabilities = body_toml_capabilities();
    capabilities.verbs.retain(|verb| verb != "escape_motion");

    establish_session(
        SimCockpit::new().with_capabilities(capabilities),
        HandshakeHello::default_motherbrain(),
        None,
    )
    .expect("pre-escape-motion firmware must remain flashable");
}

#[test]
fn cockpit_requests_serialize_to_firmware_json_kinds() {
    for (verb, expected_json_kind, _) in sample_cockpit_requests() {
        let request = sample_request_for(verb);
        let json = request.to_firmware_json(7).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            value.get("kind").and_then(serde_json::Value::as_str),
            Some(expected_json_kind),
            "{verb} serialized as {json}"
        );
        assert_eq!(
            value.get("command_id").and_then(serde_json::Value::as_u64),
            Some(7)
        );
    }
}

#[test]
fn cockpit_requests_serialize_to_compact_command_names() {
    for (verb, _, expected_compact_name) in sample_cockpit_requests() {
        let request = sample_request_for(verb);
        let line = request.to_compact_line(9);
        let first = line.split_ascii_whitespace().next().unwrap();
        assert_eq!(first, expected_compact_name, "{verb} serialized as {line}");
    }
}

#[test]
fn firmware_json_rewrites_policy_and_tones() {
    let policy = CockpitRequest::SetSafetyPolicy {
        policy: SafetyPolicy {
            bump: SafetyAction::BumpEscape,
            cliff: SafetyAction::Backoff,
            wheel_drop_latch: true,
        },
    };
    let value: serde_json::Value =
        serde_json::from_str(&policy.to_firmware_json(1).unwrap()).unwrap();
    assert!(value.get("policy").is_none());
    assert_eq!(value["bump_action"], "bump_escape");
    assert_eq!(value["cliff_action"], "backoff");
    assert_eq!(value["wheel_drop_latch"], true);

    let song = CockpitRequest::SongDefine {
        id: 2,
        tones: vec![SongTone {
            note: 72,
            duration_64ths: 8,
        }],
    };
    let value: serde_json::Value =
        serde_json::from_str(&song.to_firmware_json(2).unwrap()).unwrap();
    assert_eq!(value["tones"], "72:8");
}

#[test]
fn parses_json_accepted_and_rejected_command_responses() {
    let accepted = parse_json_cockpit_response(
        4,
        &CockpitRequest::Arm,
        r#"{"accepted":true,"command_id":4,"message":"accepted"}"#,
    )
    .unwrap();
    assert_eq!(accepted, CockpitResponse::Accepted);

    let rejected = parse_json_cockpit_response(
        5,
        &CockpitRequest::Arm,
        r#"{"accepted":false,"command_id":5,"message":"busy"}"#,
    )
    .unwrap_err();
    assert!(matches!(
        rejected,
        CockpitError::Rejected {
            command_id: 5,
            reason
        } if reason == "busy"
    ));
}

#[test]
fn parses_json_status_capabilities_and_events() {
    let status = parse_json_cockpit_response(
            1,
            &CockpitRequest::GetStatus,
            r#"{"type":"status","current_runtime_state":"idle","oi_mode":"safe","estop_latched":false,"safety_tripped":true,"safety_latch_kind":"tilt","event_next_seq":8,"audio_silent":true,"audio":{"silent":true,"last_requested_cue":"cliff","last_played_cue":"authority_acquired","last_playback_timestamp_ms":700,"suppressed_by_silent_count":2,"dropped_or_replaced_count":3},"create_sensors":{"charging_sources":2,"charging_indicator":"on"}}"#,
        )
        .unwrap();
    let CockpitResponse::Status(status) = status else {
        panic!("expected status response");
    };
    let summary = status.summary();
    assert_eq!(summary.runtime_state.as_deref(), Some("idle"));
    assert_eq!(summary.armed, Some(true));
    assert_eq!(summary.estop_latched, Some(false));
    assert_eq!(summary.safety_tripped, Some(true));
    assert_eq!(summary.safety_latch_kind, Some(SafetyLatchKind::Tilt));
    assert_eq!(summary.event_next_seq, Some(8));
    assert_eq!(summary.audio_silent, Some(true));
    assert_eq!(summary.audio_last_requested_cue.as_deref(), Some("cliff"));
    assert_eq!(
        summary.audio_last_played_cue.as_deref(),
        Some("authority_acquired")
    );
    assert_eq!(summary.audio_last_playback_timestamp_ms, Some(700));
    assert_eq!(summary.audio_suppressed_by_silent_count, Some(2));
    assert_eq!(summary.audio_dropped_or_replaced_count, Some(3));
    assert_eq!(summary.battery.charging_indicator, Some(true));
    assert!(summary.battery.home_base());

    let caps = parse_json_cockpit_response(
            2,
            &CockpitRequest::GetCapabilities,
            r#"{"accepted":true,"command_id":2,"body_kind":"create_oi","drive":"differential","verbs":["arm","cmd_vel"],"sensors":["bump"],"outputs":["drive"],"safety":["estop"],"events":["boot","safety_tripped"]}"#,
        )
        .unwrap();
    let CockpitResponse::Capabilities(caps) = caps else {
        panic!("expected capabilities response");
    };
    assert_eq!(caps.body_kind, "create_oi");
    assert_eq!(caps.verbs, ["arm", "cmd_vel"]);
    assert_eq!(caps.limits.max_linear_mm_s, i16::MAX);

    let events = parse_json_cockpit_response(
            3,
            &CockpitRequest::GetEvents { since_seq: 6 },
            r#"{"type":"events","since_seq":6,"oldest_seq":4,"next_seq":9,"dropped_before_seq":0,"events":[{"seq":7,"kind":"safety_tripped","a":1,"b":0,"c":0},{"seq":8,"kind":"motion_stopped","a":0,"b":0,"c":0}]}"#,
        )
        .unwrap();
    let CockpitResponse::Events(events) = events else {
        panic!("expected events response");
    };
    assert_eq!(events.since_seq, 6);
    assert_eq!(events.next_seq, 9);
    assert!(events.has_stop_reason());
}

#[test]
fn compact_status_infers_imu_tilt_safety_latch() {
    let summary = StatusSummary::from_raw(
            "OK 1 STATUS uptime_ms=1000 runtime=3 body=6 command=0 pending=0 power=2 oi=3 create_body_packets=1 create_last_body_packet_ms=900 charging_sources=2 imu_health=1 imu_tilt_mrad=2269 imu_impact_mm_s2=96",
        );

    assert_eq!(summary.safety_tripped, Some(true));
    assert_eq!(summary.safety_latch_kind, Some(SafetyLatchKind::Tilt));
    assert_eq!(summary.imu.tilt_magnitude_mrad, Some(2269));
    assert!(summary.battery.home_base());
}

#[test]
fn dock_ir_cue_decodes_and_steers_toward_both_buoys() {
    let green = DockIrCue::from_character(246).unwrap();
    assert_eq!(green.steering_mrad_s(400), -400);
    assert_eq!(green.bearing_hint_rad(), -0.35);
    assert!(green.force_field);

    let red = DockIrCue::from_character(250).unwrap();
    assert_eq!(red.steering_mrad_s(400), 400);
    assert_eq!(red.bearing_hint_rad(), 0.35);

    let centered = DockIrCue::from_character(254).unwrap();
    assert_eq!(centered.steering_mrad_s(400), 0);
    assert_eq!(centered.bearing_hint_rad(), 0.0);
    assert_eq!(centered.visible_score(), 0.85);
    assert_eq!(centered.near_score(), 0.55);

    assert_eq!(DockIrCue::from_character(255), None);
    assert_eq!(DockIrCue::from_character(0), None);
}

#[test]
fn status_summary_reports_complete_body_packet_age() {
    let summary = CockpitStatus {
        raw: serde_json::json!({
            "uptime_ms": 2_000,
            "current_runtime_state": "idle",
            "create_sensors": {
                "last_packet_id": 0,
                "complete_packet_count": 7,
                "last_complete_packet_timestamp_ms": 1_650,
                "bump_left": false
            }
        })
        .to_string(),
    }
    .summary();

    assert_eq!(summary.body_packet_count, Some(7));
    assert_eq!(summary.body_packet_age_ms, Some(350));
    assert_eq!(summary.body_packet_complete, Some(true));
    assert!(summary.has_fresh_complete_body_packet(500));
    assert!(!summary.has_fresh_complete_body_packet(250));
}

#[test]
fn status_summary_preserves_create_ir_from_json_and_compact_status() {
    let json = StatusSummary::from_raw(
        &serde_json::json!({
            "create_sensors": {
                "complete_packet_count": 1,
                "ir_byte": 248
            }
        })
        .to_string(),
    );
    let compact =
        StatusSummary::from_raw("OK 1 STATUS create_body_packets=1 ir_byte=137 bump_left=false");

    assert_eq!(json.infrared_character, Some(248));
    assert_eq!(compact.infrared_character, Some(137));
}

#[test]
fn compact_status_requires_a_complete_body_packet() {
    let summary = CockpitStatus {
            raw: "OK 1 STATUS uptime_ms=2000 create_rx_packets=7 create_last_packet_ms=1900 create_sensor_packet_id=35 create_body_packets=0 create_last_body_packet_ms=0 bump_left=false".into(),
        }
        .summary();

    assert_eq!(summary.body_packet_age_ms, None);
    assert_eq!(summary.body_packet_complete, Some(false));
    assert!(!summary.has_fresh_complete_body_packet(500));
}

#[test]
fn malformed_json_response_maps_to_json_error() {
    let err = parse_json_cockpit_response(1, &CockpitRequest::Arm, "{not-json").unwrap_err();
    assert!(matches!(err, CockpitError::Json(_)));
}
