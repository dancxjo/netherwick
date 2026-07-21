#[test]
fn simulator_event_cursor_happy_path() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    let mut cursor = EventCursor::new();
    let boot = cursor.poll(&mut sim).unwrap();
    assert_eq!(boot.events[0].kind, CockpitEventKind::Boot);
    sim.arm().unwrap();
    let batch = cursor.poll(&mut sim).unwrap();
    assert_eq!(cursor.next_seq(), batch.next_seq - 1);
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::CommandCompleted));
}

#[test]
fn simulator_detects_missed_events_through_dropped_before_seq() {
    let mut sim = SimCockpit::new()
        .with_unscoped_bench_mode()
        .with_event_capacity(3);
    for _ in 0..4 {
        sim.arm().unwrap();
    }
    let batch = sim.get_events_since(0).unwrap();
    assert!(batch.dropped_before_seq > 0);
    assert!(matches!(
        batch.ensure_no_missed_events(),
        Err(CockpitError::MissedEvents { .. })
    ));
}

#[test]
fn simulator_arm_stop_disarm_lifecycle() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.arm().unwrap();
    sim.cmd_vel(50, 0, 100).unwrap();
    sim.stop().unwrap();
    sim.disarm().unwrap();
    let batch = sim.get_events_since(0).unwrap();
    let kinds: Vec<_> = batch.events.iter().map(|event| &event.kind).collect();
    assert!(kinds.contains(&&CockpitEventKind::CommandInterrupted));
    assert!(kinds.contains(&&CockpitEventKind::MotionStopped));
    assert!(kinds.contains(&&CockpitEventKind::CommandCompleted));
}

#[test]
fn simulator_cmd_vel_completes_after_ttl() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.cmd_vel(70, 10, 300).unwrap();
    sim.advance_ms(299);
    assert!(!sim
        .get_events_since(0)
        .unwrap()
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::MotionStopped));
    sim.advance_ms(1);
    let batch = sim.get_events_since(0).unwrap();
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::MotionStopped));
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::CommandCompleted));
}

#[test]
fn simulator_estop_and_clear_estop() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.estop().unwrap();
    sim.clear_estop().unwrap();
    let batch = sim.get_events_since(0).unwrap();
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::EStopLatched));
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::EStopCleared));
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::SafetyCleared));
}

#[test]
fn simulator_heartbeat_expiry_is_stop_reason() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.cmd_vel(70, 0, 1_000).unwrap();
    sim.heartbeat_stop(100).unwrap();
    sim.advance_ms(100);
    let batch = sim.get_events_since(0).unwrap();
    assert!(batch.has_stop_reason());
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::HeartbeatExpired));
}

#[test]
fn simulator_command_rejection_alone_is_diagnostic() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.push_event(CockpitEventKind::CommandRejected, 7, 11, (6 << 8) | 1);
    let batch = sim.get_events_since(0).unwrap();

    assert!(!batch.has_stop_reason());
    let rejected = batch
        .events
        .iter()
        .find(|event| event.kind == CockpitEventKind::CommandRejected)
        .unwrap();
    assert_eq!(
        rejected.command_rejection(),
        Some(CommandRejection {
            command_id: 7,
            command_seq: 11,
            command_code: 6,
            reason: CommandRejectReason::Busy,
        })
    );
}

#[test]
fn simulator_safety_tripped_stops_motion_and_rejects_motion() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.cmd_vel(70, 0, 1_000).unwrap();
    sim.trip_safety();
    sim.cmd_vel(10, 0, 100).unwrap();
    let batch = sim.get_events_since(0).unwrap();
    assert!(batch.has_stop_reason());
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::CommandRejected));
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::MotionStopped));
}

#[test]
fn simulator_reset_odometry() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.reset_odometry().unwrap();
    assert_eq!(sim.odometry_reset_count(), 1);
    let status = sim.get_status().unwrap();
    assert!(status.raw.contains("odometry_resets=1"));
    assert_eq!(status.summary().odometry.reset_count, Some(1));
    assert_eq!(status.summary().odometry.distance_mm, Some(0));
}

#[test]
fn simulator_builtin_sensor_edges_trip_and_clear() {
    let mut sim = SimCockpit::new();
    sim.set_bump(true, false);
    sim.set_bump(false, false);
    sim.set_cliff(true);
    sim.set_cliff(false);
    sim.set_wall(true);
    sim.set_wall(false);
    sim.set_virtual_wall(true);
    sim.set_virtual_wall(false);

    let batch = sim.get_events_since(0).unwrap();
    assert_eq!(
        batch
            .events
            .iter()
            .filter(|event| event.kind == CockpitEventKind::BumpChanged)
            .count(),
        2
    );
    assert_eq!(
        batch
            .events
            .iter()
            .filter(|event| event.kind == CockpitEventKind::CliffChanged)
            .count(),
        2
    );
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::WallChanged && event.a == 1));
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::VirtualWallChanged && event.a == 0));
}

#[test]
fn simulator_stationary_bump_latches_without_starting_withdrawal() {
    let mut sim = SimCockpit::new();
    sim.set_bump(true, false);

    let status = sim.get_status().unwrap().summary();
    assert_eq!(status.safety_tripped, Some(true));
    assert_eq!(status.safety_latch_kind, Some(SafetyLatchKind::Bump));
    assert!(sim.active_contact_withdrawal.is_none());
    assert!(!sim
        .get_events_since(0)
        .unwrap()
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::ContactWithdrawalStarted));
}

#[test]
fn simulator_contact_withdrawal_is_typed_and_authority_independent() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.cmd_vel(80, 0, 1_000).unwrap();
    sim.set_bump(true, false);
    sim.control_lease = None;
    sim.control_lease_expires_at_ms = None;
    sim.advance_ms(300);

    let events = sim.get_events_since(0).unwrap().events;
    let lifecycle: Vec<_> = events
        .iter()
        .filter_map(CockpitEvent::contact_withdrawal)
        .collect();
    assert_eq!(lifecycle.len(), 2);
    assert!(matches!(
        lifecycle[0],
        ContactWithdrawalEvent::Started {
            contact_bits: 1,
            repeated_count: 1,
            preempted_command_id: 1,
            reverse_speed_mm_s: 80,
            maximum_duration_ms: 300,
        }
    ));
    assert!(matches!(
        lifecycle[1],
        ContactWithdrawalEvent::Completed {
            outcome: ContactWithdrawalOutcome::Completed,
            final_stopped: true,
            observed_displacement_mm: -24,
            elapsed_ms: 300,
            ..
        }
    ));
}

#[test]
fn simulator_escape_motion_is_generation_bound_and_reflex_ordered() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.cmd_vel(80, 0, 1_000).unwrap();
    sim.set_bump(true, false);
    let generation = sim
        .get_status()
        .unwrap()
        .summary()
        .safety_hazard_generation
        .unwrap();

    assert!(matches!(
        sim.escape_motion(SafetyLatchKind::Bump, generation, -50, 0, 250),
        Err(CockpitError::Rejected { ref reason, .. }) if reason == "busy"
    ));
    sim.advance_ms(CONTACT_WITHDRAWAL_DURATION_MS);
    sim.escape_motion(SafetyLatchKind::Bump, generation, -50, 0, 250)
        .unwrap();
    assert!(sim
        .get_status()
        .unwrap()
        .raw
        .contains("active_cmd_vel=true"));

    sim.set_cliff(true);
    let status = sim.get_status().unwrap().summary();
    assert!(status.raw.contains("active_cmd_vel=false"));
    assert_eq!(status.safety_latch_kind, Some(SafetyLatchKind::Cliff));
    assert!(matches!(
        sim.escape_motion(SafetyLatchKind::Bump, generation, -50, 0, 250),
        Err(CockpitError::Rejected { ref reason, .. }) if reason == "hazard_mismatch"
    ));
}

#[test]
fn simulator_wheel_drop_latches_and_clears() {
    let mut sim = SimCockpit::new().with_unscoped_bench_mode();
    sim.cmd_vel(70, 0, 1_000).unwrap();
    sim.set_wheel_drop(true);
    sim.set_wheel_drop(false);
    sim.clear_safety_latch(SafetyLatchKind::WheelDrop).unwrap();

    let batch = sim.get_events_since(0).unwrap();
    assert!(batch.has_stop_reason());
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::WheelDropLatched));
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::WheelDropCleared));
}

#[test]
fn simulator_low_battery_and_charging_state_change() {
    let mut sim = SimCockpit::new();
    sim.set_battery(400, 2600);
    sim.set_charging_state(2);

    let status = sim.get_status().unwrap().summary();
    assert_eq!(status.battery.percent, Some(15));
    assert_eq!(status.battery.low, Some(true));
    assert_eq!(status.battery.charging_state, Some(2));

    let batch = sim.get_events_since(0).unwrap();
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::BatteryLow && event.a == 15));
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::ChargingStateChanged && event.a == 2));
}

#[test]
fn diagnostic_session_can_establish_sensor_stream_through_cockpit_trait() {
    let connector = SimCockpit::new();
    let mut cockpit =
        establish_diagnostic_session(connector, HandshakeHello::default_motherbrain(), None)
            .unwrap();

    Cockpit::stream_sensors(&mut cockpit, true, 0, 250).unwrap();
}

#[test]
fn simulator_buttons_and_ir_changes_are_events() {
    let mut sim = SimCockpit::new();
    sim.set_buttons(0b0000_0011);
    sim.set_ir_byte(248);

    let status = sim.get_status().unwrap().summary();
    assert_eq!(status.contact.any_contact(), Some(false));
    let batch = sim.get_events_since(0).unwrap();
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::ButtonsChanged && event.a == 3));
    assert!(batch
        .events
        .iter()
        .any(|event| event.kind == CockpitEventKind::IrChanged && event.a == 248));
}

#[test]
fn parses_ok_and_err_responses() {
    assert!(expect_ok(2, "OK 2").is_ok());
    assert!(matches!(
        expect_ok(2, "ERR 2 busy"),
        Err(CockpitError::Rejected {
            command_id: 2,
            reason
        }) if reason == "busy"
    ));
}

#[test]
fn parses_status_response_as_raw_status() {
    expect_ok(9, "OK 9 STATUS runtime=idle demo=idle").unwrap();
    let status = CockpitStatus {
        raw: "OK 9 STATUS runtime=idle demo=idle".to_owned(),
    };
    assert!(status.raw.contains("runtime=idle"));
}

#[test]
fn parses_compact_events() {
    let batch = parse_events(
            7,
            12,
            "OK 7 EVENTS since=12 oldest=4 next=15 dropped_before=0 count=2 | 13:motion_requested:1,2,3 | 14:safety_tripped:2,0,0",
        )
        .unwrap();
    assert_eq!(batch.next_seq, 15);
    assert_eq!(batch.dropped_before_seq, 0);
    assert_eq!(batch.events.len(), 2);
    assert_eq!(batch.events[1].kind, CockpitEventKind::SafetyTripped);
    assert!(batch.has_stop_reason());
}

#[test]
fn parses_unknown_event_kinds() {
    let batch = parse_events(
            7,
            12,
            "OK 7 EVENTS since=12 oldest=13 next=14 dropped_before=0 count=1 | 13:new_future_event:1,2,3",
        )
        .unwrap();
    assert_eq!(
        batch.events[0].kind,
        CockpitEventKind::Unknown("new_future_event".to_owned())
    );
    assert_eq!(batch.events[0].kind.as_str(), "new_future_event");
}

#[test]
fn rejects_malformed_or_truncated_event_lines() {
    for line in [
            "OK 7 EVENTS since=12 oldest=13 next=14 dropped_before=0 count=1 | malformed",
            "OK 7 EVENTS since=12 oldest=13 next=14 dropped_before=0 count=1 | 13:motion_requested:1,2",
            "OK 7 EVENTS since=12 oldest=13 next=14 dropped_before=0 count=2 | 13:motion_requested:1,2,3",
        ] {
            assert!(matches!(
                parse_events(7, 12, line),
                Err(CockpitError::BadResponse(_))
            ));
        }
}

#[test]
fn parses_large_event_lists_near_response_buffer_limits() {
    let mut line = String::from("OK 7 EVENTS since=0 oldest=1 next=29 dropped_before=0 count=28");
    for seq in 1..29 {
        line.push_str(&format!(
            " | {seq}:motion_requested:{seq},{},{}",
            seq + 1,
            seq + 2
        ));
    }
    assert!(line.len() < DEFAULT_UART_MAX_RESPONSE_LEN);
    let batch = parse_events(7, 0, &line).unwrap();
    assert_eq!(batch.events.len(), 28);
    assert_eq!(batch.events.last().unwrap().seq, 28);
}

#[test]
fn detects_missed_events() {
    let batch = parse_events(
        1,
        0,
        "OK 1 EVENTS since=0 oldest=20 next=52 dropped_before=20 count=0",
    )
    .unwrap();
    assert!(matches!(
        batch.ensure_no_missed_events(),
        Err(CockpitError::MissedEvents {
            dropped_before_seq: 20
        })
    ));
}

#[test]
fn parses_capabilities_without_body_specific_api() {
    let caps = parse_capabilities(
            3,
            "OK 3 CAPABILITIES body_kind=create_oi drive=differential verbs=arm,stop,cmd_vel sensors=bump,battery outputs=lights,song safety=bump,estop events=boot,safety_tripped limits=max_linear_mm_s:500 max_tones=16 song_slots=16 feedback_slots=6 sensor_packets=0,7-31",
        )
        .unwrap();
    assert_eq!(caps.drive, "differential");
    assert_eq!(caps.verbs, ["arm", "stop", "cmd_vel"]);
    assert_eq!(caps.events, ["boot", "safety_tripped"]);
    assert_eq!(caps.limits.max_linear_mm_s, 500);
}

#[test]
fn parses_json_capability_limits() {
    let caps = parse_json_capabilities(&serde_json::json!({
        "body_kind":"create_oi",
        "drive":"differential",
        "verbs":["cmd_vel"],
        "events":["boot"],
        "limits":{
            "max_linear_mm_s":120,
            "max_angular_mrad_s":800,
            "min_ttl_ms":20,
            "max_ttl_ms":900
        }
    }))
    .unwrap();
    assert_eq!(
        caps.limits,
        CockpitLimits {
            max_linear_mm_s: 120,
            max_angular_mrad_s: 800,
            min_ttl_ms: 20,
            max_ttl_ms: 900,
        }
    );
}

#[test]
fn contract_rejects_unsupported_lights_music_and_step_verbs() {
    let contract =
        CockpitContract::new(sim_caps_without(&["set_lights", "song_play", "dock_align"]));
    assert!(matches!(
        contract.validate_request(&CockpitRequest::SetLights {
            pattern: LightPattern::Status
        }),
        Err(CockpitError::Policy(message)) if message.contains("set_lights")
    ));
    assert!(matches!(
        contract.validate_request(&CockpitRequest::SongPlay { id: 0 }),
        Err(CockpitError::Policy(message)) if message.contains("song_play")
    ));
    assert!(matches!(
        contract.validate_request(&CockpitRequest::DockAlign {
            bearing_mrad: 0,
            range_mm: 400,
            max_linear_mm_s: 80,
            max_angular_mrad_s: 500,
            stop_range_mm: 200,
            ttl_ms: 300,
        }),
        Err(CockpitError::Policy(message)) if message.contains("dock_align")
    ));
}
