#[test]
fn production_possession_renews_and_replaces_lease() {
    let ready = establish_session(SimCockpit::new(), hello(), None).unwrap();
    let mut possession = MotherbrainPossession::acquire(ready, 1_000).unwrap();
    let first = possession.snapshot();
    possession.lease_acquired_at = Instant::now() - Duration::from_millis(800);
    possession.maintain().unwrap();
    let renewed = possession.snapshot();
    assert!(renewed.possessed);
    assert!(renewed.lease_generation > first.lease_generation);
    assert_ne!(renewed.lease_id, first.lease_id);
}

#[test]
fn production_possession_renews_long_lease_on_short_cadence() {
    let ready = establish_session(SimCockpit::new(), hello(), None).unwrap();
    let mut possession = MotherbrainPossession::acquire(ready, 60_000).unwrap();
    let first = possession.snapshot();
    possession.lease_acquired_at =
        Instant::now() - Duration::from_millis(POSSESSION_LEASE_RENEW_INTERVAL_MS as u64 + 1);

    possession.maintain().unwrap();

    let renewed = possession.snapshot();
    assert!(renewed.possessed);
    assert!(renewed.lease_generation > first.lease_generation);
    assert_ne!(renewed.lease_id, first.lease_id);
}

#[test]
fn production_possession_retries_transient_busy_commands() {
    let cockpit = BusyOnceCockpit {
        inner: SimCockpit::new(),
        busy_remaining: 0,
        attempts: 0,
        heartbeat_attempts: 0,
        cmd_vel_attempts: 0,
        last_heartbeat_timeout_ms: None,
        last_bump_escape: None,
    };
    let ready = establish_session(cockpit, hello(), None).unwrap();
    let mut possession = MotherbrainPossession::acquire(ready, 1_000).unwrap();
    possession.session.connector_mut().busy_remaining = 1;
    let attempts_before = possession.session.connector_mut().attempts;

    possession
        .execute(CockpitRequest::PlayFeedback {
            feedback: FeedbackKind::Ok,
        })
        .unwrap();

    assert_eq!(
        possession.session.connector_mut().attempts,
        attempts_before + 2
    );
    assert!(possession.snapshot().possessed);
}

#[test]
fn safe_cockpit_uses_possessions_single_motion_heartbeat() {
    let cockpit = BusyOnceCockpit {
        inner: SimCockpit::new(),
        busy_remaining: 0,
        attempts: 0,
        heartbeat_attempts: 0,
        cmd_vel_attempts: 0,
        last_heartbeat_timeout_ms: None,
        last_bump_escape: None,
    };
    let ready = establish_session(cockpit, hello(), None).unwrap();
    let possession = MotherbrainPossession::acquire(ready, 60_000).unwrap();
    let mut safe = SafeCockpit::new(possession);

    safe.pulse_motion(20, 0).unwrap();

    let connector = safe.client_mut().session.connector_mut();
    assert_eq!(connector.heartbeat_attempts, 1);
    assert_eq!(connector.cmd_vel_attempts, 1);
}

#[test]
fn production_possession_renews_and_retries_stale_control_lease() {
    let cockpit = StaleLeaseOnceCockpit {
        inner: SimCockpit::new(),
        invalid_remaining: 0,
        attempts: 0,
    };
    let ready = establish_session(cockpit, hello(), None).unwrap();
    let mut possession = MotherbrainPossession::acquire(ready, 60_000).unwrap();
    possession.session.connector_mut().invalid_remaining = 1;
    let first = possession.snapshot();

    possession
        .execute(CockpitRequest::PlayFeedback {
            feedback: FeedbackKind::Ok,
        })
        .unwrap();

    let renewed = possession.snapshot();
    assert_eq!(possession.session.connector_mut().attempts, 2);
    assert!(renewed.possessed);
    assert!(renewed.lease_generation > first.lease_generation);
    assert_ne!(renewed.lease_id, first.lease_id);
}

#[test]
fn closed_possession_motor_gate_allows_estop_reset_and_imu_zeroing() {
    let ready = establish_session(SimCockpit::new(), hello(), None).unwrap();
    let mut possession = MotherbrainPossession::acquire(ready, 60_000).unwrap();

    expect_accepted(possession.execute(CockpitRequest::EStop).unwrap()).unwrap();
    let status = match possession.execute(CockpitRequest::GetStatus).unwrap() {
        CockpitResponse::Status(status) => status.summary(),
        other => panic!("{other:?}"),
    };
    assert_eq!(status.estop_latched, Some(true));
    assert!(!possession.snapshot().possessed);

    assert!(matches!(
        possession.execute(CockpitRequest::CmdVel {
            linear_mm_s: 10,
            angular_mrad_s: 0,
            ttl_ms: 100,
        }),
        Err(CockpitError::Policy(_))
    ));

    expect_accepted(
        possession
            .execute(CockpitRequest::ZeroImuOrientation)
            .unwrap(),
    )
    .unwrap();
    let status = match possession.execute(CockpitRequest::GetStatus).unwrap() {
        CockpitResponse::Status(status) => status.summary(),
        other => panic!("{other:?}"),
    };
    assert_eq!(status.imu.calibration.as_deref(), Some("3"));

    expect_accepted(
        possession
            .execute(CockpitRequest::ClearImuOrientation)
            .unwrap(),
    )
    .unwrap();
    expect_accepted(possession.execute(CockpitRequest::ClearEStop).unwrap()).unwrap();
    let status = match possession.execute(CockpitRequest::GetStatus).unwrap() {
        CockpitResponse::Status(status) => status.summary(),
        other => panic!("{other:?}"),
    };
    assert_eq!(status.imu.calibration.as_deref(), Some("0"));
    assert_eq!(status.estop_latched, Some(false));
    assert!(possession.snapshot().possessed);
}

#[test]
fn exorcize_closes_gate_only_after_stop_is_acknowledged() {
    let cockpit = StopRejectingCockpit {
        inner: SimCockpit::new(),
        reject_stop: false,
        disarm_requests: 0,
    };
    let ready = establish_session(cockpit, hello(), None).unwrap();
    let mut possession = MotherbrainPossession::acquire(ready, 1_000).unwrap();
    possession.session.connector_mut().reject_stop = true;

    assert!(possession.exorcize().is_err());
    assert!(!possession.snapshot().possessed);
    assert_eq!(possession.session.connector_mut().disarm_requests, 0);
}

#[test]
fn exorcize_stops_without_disarming_create_oi() {
    let cockpit = StopRejectingCockpit {
        inner: SimCockpit::new(),
        reject_stop: false,
        disarm_requests: 0,
    };
    let ready = establish_session(cockpit, hello(), None).unwrap();
    let mut possession = MotherbrainPossession::acquire(ready, 1_000).unwrap();

    possession.exorcize().unwrap();

    assert!(!possession.snapshot().possessed);
    assert_eq!(possession.session.connector_mut().disarm_requests, 0);
}

#[test]
fn renewal_failure_closes_motor_gate() {
    let ready = establish_session(SimCockpit::new(), hello(), None).unwrap();
    let mut possession = MotherbrainPossession::acquire(ready, 1_000).unwrap();
    possession.lease_acquired_at = Instant::now() - Duration::from_millis(800);
    possession
        .session
        .connector_mut()
        .handshake(hello().new_attempt())
        .unwrap();
    assert!(possession.maintain().is_err());
    assert!(!possession.snapshot().possessed);
    assert!(possession
        .execute(CockpitRequest::CmdVel {
            linear_mm_s: 1,
            angular_mrad_s: 0,
            ttl_ms: 100,
        })
        .is_err());
}

#[test]
fn production_possession_clamps_motion_and_hides_oi() {
    let ready = establish_session(SimCockpit::new(), hello(), None).unwrap();
    let mut possession = MotherbrainPossession::acquire(ready, 1_000).unwrap();
    possession
        .execute(CockpitRequest::CmdVel {
            linear_mm_s: 500,
            angular_mrad_s: 5_000,
            ttl_ms: 10_000,
        })
        .unwrap();
    let events = possession.session.poll_events().unwrap();
    let motion = events
        .events
        .iter()
        .find(|event| event.kind == CockpitEventKind::MotionRequested)
        .unwrap();
    assert_eq!(motion.a, pack_i16_pair(50, 500));
    assert_eq!(motion.b, 300);
    assert!(possession
        .execute(CockpitRequest::SetMode {
            mode: CreateOiMode::Full,
        })
        .is_err());
}
