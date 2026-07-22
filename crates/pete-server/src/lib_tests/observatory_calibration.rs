fn calibration_selection(
    previous: Option<pete_now::Now>,
    current: pete_now::Now,
) -> ObservatoryNowSelection {
    ObservatoryNowSelection {
        selected: ObservatoryNowSnapshot {
            snapshot_id: format!("now-{}", current.t_ms),
            now: current,
        },
        previous: previous.map(|now| ObservatoryNowSnapshot {
            snapshot_id: format!("now-{}", now.t_ms),
            now,
        }),
    }
}

fn blank_now(t_ms: u64) -> pete_now::Now {
    pete_now::Now::blank(t_ms, pete_body::BodySense::default())
}

#[test]
fn kinect_remount_epoch_and_partial_observability_are_unmistakable() {
    let config = pete_now::CalibrationStateConfig::default();
    let previous_estimator = pete_now::CalibrationStateMachine::new(
        pete_now::RigidTransform3::default(),
        10,
        config.clone(),
    );
    let mut previous = blank_now(100);
    previous.kinect.live_geometry_calibration = Some(previous_estimator.estimate().clone());
    let current_estimator =
        pete_now::CalibrationStateMachine::new(pete_now::RigidTransform3::default(), 200, config);
    let mut current_estimate = current_estimator.estimate().clone();
    current_estimate.epoch.id = 1;
    current_estimate.epoch.invalidation_reason = Some("Kinect mount moved".into());
    current_estimate.observable_dofs = [true, true, true, true, true, false];
    current_estimate.trust_state = pete_now::CalibrationTrustState::Degraded;
    let mut current = blank_now(250);
    current.kinect.live_geometry_calibration = Some(current_estimate);

    let response = build_calibration_console(&calibration_selection(Some(previous), current), &[]);
    let kinect = &response.estimators[0];

    assert!(kinect.epoch_changed);
    assert_eq!(
        kinect.invalidation_reason.as_deref(),
        Some("Kinect mount moved")
    );
    assert!(!kinect.degrees_of_freedom.last().unwrap().observable);
    assert!(kinect.consumers.iter().all(|consumer| !consumer.allowed));
}

#[test]
fn stale_timing_and_clock_epoch_changes_remain_visible() {
    let timing = |epoch, observed| {
        serde_json::json!({
            "kinect": {
                "stream": "kinect",
                "trust_state": "degraded",
                "epoch": epoch,
                "epoch_started_at_ms": 10,
                "last_observed_at_ms": observed,
                "confidence": 0.4,
                "evidence_count": 12,
                "rejection_reasons": ["evidence is stale"],
                "transport_latency": {"median_ms": 12.0, "p95_ms": 30.0, "jitter_ms": 4.0, "uncertainty_ms": 8.0, "sample_count": 12}
            }
        })
    };
    let mut previous = blank_now(500);
    previous
        .extensions
        .insert("sensor.latency_calibration".into(), timing(3, 450));
    let mut current = blank_now(10_000);
    current
        .extensions
        .insert("sensor.latency_calibration".into(), timing(4, 500));

    let response = build_calibration_console(&calibration_selection(Some(previous), current), &[]);
    let timing = response
        .estimators
        .iter()
        .find(|estimator| estimator.id == "timing:kinect")
        .unwrap();

    assert!(timing.epoch_changed);
    assert_eq!(timing.evidence_age_ms, Some(9_500));
    assert!(timing
        .rejection_reasons
        .iter()
        .any(|reason| reason.contains("stale")));
    assert!(!timing.consumers[0].allowed);
    assert!(timing.plots.iter().any(|plot| plot.metric == "p95 latency"));
}

#[test]
fn missing_lidar_does_not_block_an_otherwise_fully_trusted_transform() {
    let mut estimator = pete_now::CalibrationStateMachine::new(
        pete_now::RigidTransform3::default(),
        1,
        pete_now::CalibrationStateConfig::default(),
    )
    .estimate()
    .clone();
    estimator.trust_state = pete_now::CalibrationTrustState::Trusted;
    estimator.observable_dofs = [true; 6];
    estimator.confidence = 0.95;
    let mut now = blank_now(100);
    now.kinect.live_geometry_calibration = Some(estimator);

    let response = build_calibration_console(&calibration_selection(None, now), &[]);
    let kinect = &response.estimators[0];

    assert!(kinect.consumers.iter().all(|consumer| consumer.allowed));
    assert!(kinect
        .notes
        .iter()
        .any(|note| note.contains("optional corroboration")));
}

#[test]
fn imu_partial_axis_trust_allows_roll_pitch_but_blocks_absolute_yaw() {
    let estimator = pete_now::ImuCalibrationEstimator::new(
        pete_now::RigidTransform3::default(),
        true,
        0,
        pete_now::ImuCalibrationConfig::default(),
    );
    let mut calibration = estimator.state().clone();
    calibration.trust_state = pete_now::ImuCalibrationTrustState::Trusted;
    calibration.roll_pitch_observable = true;
    calibration.yaw_axis_observable = false;
    calibration.confidence = 0.9;
    let mut now = blank_now(100);
    now.imu.calibration = Some(calibration);

    let response = build_calibration_console(&calibration_selection(None, now), &[]);
    let imu = response
        .estimators
        .iter()
        .find(|view| view.id == "imu")
        .unwrap();

    assert!(
        imu.consumers
            .iter()
            .find(|gate| gate.consumer == "roll/pitch correction")
            .unwrap()
            .allowed
    );
    assert!(
        !imu.consumers
            .iter()
            .find(|gate| gate.consumer == "absolute yaw")
            .unwrap()
            .allowed
    );
}

#[test]
fn surface_and_tire_condition_changes_are_reported_with_wheel_scale_plots() {
    let locomotion = |surface: &str, tire: &str| {
        let mut state = pete_now::LocomotionCalibrationState::default();
        state.conditions.surface = Some(surface.into());
        state.conditions.tire_condition = Some(tire.into());
        serde_json::to_value(state).unwrap()
    };
    let mut previous = blank_now(100);
    previous
        .extensions
        .insert("calibration.locomotion".into(), locomotion("wood", "fresh"));
    let mut current = blank_now(200);
    current.extensions.insert(
        "calibration.locomotion".into(),
        locomotion("carpet", "worn"),
    );

    let response = build_calibration_console(&calibration_selection(Some(previous), current), &[]);
    let locomotion = response
        .estimators
        .iter()
        .find(|view| view.id == "locomotion")
        .unwrap();

    assert!(locomotion
        .notes
        .iter()
        .any(|note| note.contains("conditions changed")));
    assert!(locomotion
        .plots
        .iter()
        .any(|plot| plot.metric == "CW rotation scale"));
    assert!(locomotion
        .plots
        .iter()
        .any(|plot| plot.metric == "left wheel scale"));
    assert!(locomotion
        .consumers
        .iter()
        .any(|gate| gate.consumer == "brainstem motor/safety authority" && !gate.allowed));
}
