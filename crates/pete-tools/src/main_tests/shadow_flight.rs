fn shadow_test_args(output: PathBuf) -> ShadowFlightArgs {
    ShadowFlightArgs {
        source: ShadowFlightSource::Fixture,
        input: None,
        seed: 7,
        ticks: 2,
        clock: ShadowClockMode::Accelerated,
        speed: 1_000.0,
        higher_brain: ShadowHigherBrainMode::Disabled,
        allow_substitutions: Vec::new(),
        pause_at: Vec::new(),
        faults: vec!["1:wheel_drop".into()],
        output,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn seeded_shadow_flight_is_reproducible_complete_and_transport_isolated() {
    let root = std::env::temp_dir().join(format!("pete-shadow-flight-{}", Uuid::new_v4()));
    let left_args = shadow_test_args(root.join("left"));
    let right_args = shadow_test_args(root.join("right"));
    let advisory_args = ShadowFlightArgs {
        higher_brain: ShadowHigherBrainMode::AdvisoryStub,
        allow_substitutions: vec!["higher_brain".into()],
        output: root.join("advisory"),
        ..shadow_test_args(root.join("unused"))
    };
    let (left_manifest, left_summary) = run_shadow_flight(&left_args).await.unwrap();
    let (right_manifest, right_summary) = run_shadow_flight(&right_args).await.unwrap();
    let (_, advisory_summary) = run_shadow_flight(&advisory_args).await.unwrap();

    assert_eq!(left_manifest.events_sha256, right_manifest.events_sha256);
    assert_eq!(left_summary.event_type_counts, right_summary.event_type_counts);
    assert!(left_summary.full_causal_chain_observed);
    assert!(left_summary.simulated_outcomes > 0);
    assert!(left_summary.outcome_feedback_frames > 0);
    assert!(left_summary.inline_learning_samples_observed > 0);
    assert!(left_summary.safety_gate_events > 0);
    assert_eq!(left_summary.higher_brain_authority_violations, 0);
    assert_eq!(advisory_summary.higher_brain_authority_violations, 0);
    assert!(advisory_summary.higher_brain_advice_responses > 0);
    assert_eq!(
        left_summary.local_authority_sha256,
        advisory_summary.local_authority_sha256,
        "enabling advisory higher cognition must not change local gate or command authority"
    );
    assert_eq!(left_manifest.actuator_transport, "in_process_simulator_only");
    assert!(!left_manifest.network_required);
    assert!(!left_manifest.physical_hardware_required);
    assert!(!left_manifest.lidar_required);
    assert!(left_manifest
        .production_components
        .iter()
        .any(|component| component.contains("MinimalRuntime")));
    assert!(fs::read_to_string(left_args.output.join("input-frames.jsonl"))
        .unwrap()
        .contains("wheel_drop"));
    let advisory_events = fs::read_to_string(advisory_args.output.join("events.jsonl")).unwrap();
    assert!(advisory_events.contains("shadow higher-brain advice was produced"));
    let canonical = fs::read_to_string(left_args.output.join("events.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<ShadowEventRecord>(line).unwrap().event)
        .collect::<Vec<_>>();
    let certification = pete_worldlab::score_shadow_events(
        pete_worldlab::CertificationRunIdentity::deterministic(
            left_manifest.input_frames_sha256.clone(),
            "test:software",
            "test:config",
            left_manifest.source_identity.clone(),
            left_manifest.seed,
        ),
        &canonical,
        &["test://shadow-flight".into()],
    );
    assert!(certification.passed, "{:#?}", certification.gates);

    fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn required_higher_brain_substitution_fails_without_explicit_authorization() {
    let root = std::env::temp_dir().join(format!("pete-shadow-substitution-{}", Uuid::new_v4()));
    let args = ShadowFlightArgs {
        higher_brain: ShadowHigherBrainMode::AdvisoryStub,
        output: root.clone(),
        ..shadow_test_args(root.clone())
    };
    let error = run_shadow_flight_command(args).await.unwrap_err();
    assert!(error
        .to_string()
        .contains("required production component higher_brain is substituted"));
    let failure: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("failure.json")).unwrap()).unwrap();
    assert_eq!(failure["status"], "failed");
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn capture_shadow_replay_uses_capture_identity_and_simulated_outcomes() {
    let root = std::env::temp_dir().join(format!("pete-shadow-capture-{}", Uuid::new_v4()));
    let capture = root.join("capture");
    let mut writer = CaptureWriter::create(&capture, CaptureSource::Sim, Some(100))
        .await
        .unwrap();
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.last_update_ms = 100;
    writer
        .append_snapshot(100, snapshot, Vec::new())
        .await
        .unwrap();
    let mut next_snapshot = WorldSnapshot::default();
    next_snapshot.body.last_update_ms = 200;
    writer
        .append_snapshot(200, next_snapshot, Vec::new())
        .await
        .unwrap();
    let capture_manifest = writer.finish().await.unwrap();
    let args = ShadowFlightArgs {
        source: ShadowFlightSource::Capture,
        input: Some(capture),
        ticks: 2,
        faults: Vec::new(),
        output: root.join("replay"),
        ..shadow_test_args(root.join("unused"))
    };
    let (manifest, summary) = run_shadow_flight(&args).await.unwrap();
    assert_eq!(manifest.source_identity, format!("capture:{}", capture_manifest.id));
    assert!(summary.simulated_outcomes > 0);
    assert!(summary.full_causal_chain_observed);
    assert_eq!(summary.outcome_feedback_frames, 1);
    assert!(summary.inline_learning_samples_observed > 0);
    let inputs = fs::read_to_string(args.output.join("input-frames.jsonl")).unwrap();
    let provenance = inputs
        .lines()
        .map(|line| serde_json::from_str::<ShadowInputFrameProvenance>(line).unwrap())
        .collect::<Vec<_>>();
    assert!(provenance[0].outcome_feedback_event_ids.is_empty());
    assert_eq!(provenance[1].outcome_feedback_event_ids.len(), 2);
    assert!(provenance[1].inline_learning_samples_observed > 0);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 1)]
async fn ledger_shadow_replay_preserves_original_frame_identity_and_provenance() {
    let root = std::env::temp_dir().join(format!("pete-shadow-ledger-{}", Uuid::new_v4()));
    let fixture_args = shadow_test_args(root.join("fixture"));
    run_shadow_flight(&fixture_args).await.unwrap();
    let ledger_args = ShadowFlightArgs {
        source: ShadowFlightSource::Ledger,
        input: Some(fixture_args.output.join("ledger")),
        ticks: 2,
        output: root.join("replay"),
        ..shadow_test_args(root.join("unused"))
    };
    let (manifest, summary) = run_shadow_flight(&ledger_args).await.unwrap();
    assert_eq!(manifest.source, ShadowFlightSource::Ledger);
    assert_eq!(summary.ticks_completed, 2);
    assert_eq!(summary.outcome_feedback_frames, 1);
    assert!(summary.inline_learning_samples_observed > 0);

    let inputs = fs::read_to_string(ledger_args.output.join("input-frames.jsonl")).unwrap();
    for (index, line) in inputs.lines().enumerate() {
        let provenance: ShadowInputFrameProvenance = serde_json::from_str(line).unwrap();
        assert_eq!(
            provenance.input_frame_id,
            format!("ledger-frame:{}", provenance.runtime_frame_id)
        );
        if index == 1 {
            assert_eq!(provenance.outcome_feedback_event_ids.len(), 2);
            assert!(provenance.inline_learning_samples_observed > 0);
        }
    }
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn unavailable_shadow_source_fails_explicitly_without_opening_a_transport() {
    let root = std::env::temp_dir().join(format!("pete-shadow-failure-{}", Uuid::new_v4()));
    let args = ShadowFlightArgs {
        source: ShadowFlightSource::Capture,
        input: None,
        output: root.clone(),
        ..shadow_test_args(root.clone())
    };
    let error = run_shadow_flight_command(args).await.unwrap_err();
    assert!(error.to_string().contains("--input is required"));
    let failure: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("failure.json")).unwrap()).unwrap();
    assert_eq!(failure["physical_transport_open"], false);
    fs::remove_dir_all(root).unwrap();
}
