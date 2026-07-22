fn shadow_test_args(output: PathBuf) -> ShadowFlightArgs {
    ShadowFlightArgs {
        source: ShadowFlightSource::Fixture,
        input: None,
        seed: 7,
        ticks: 2,
        clock: ShadowClockMode::Accelerated,
        speed: 1_000.0,
        higher_brain: ShadowHigherBrainMode::Disabled,
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
    assert!(left_summary.safety_gate_events > 0);
    assert_eq!(left_summary.higher_brain_authority_violations, 0);
    assert_eq!(advisory_summary.higher_brain_authority_violations, 0);
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
    let capture_manifest = writer.finish().await.unwrap();
    let args = ShadowFlightArgs {
        source: ShadowFlightSource::Capture,
        input: Some(capture),
        ticks: 1,
        faults: Vec::new(),
        output: root.join("replay"),
        ..shadow_test_args(root.join("unused"))
    };
    let (manifest, summary) = run_shadow_flight(&args).await.unwrap();
    assert_eq!(manifest.source_identity, format!("capture:{}", capture_manifest.id));
    assert!(summary.simulated_outcomes > 0);
    assert!(summary.full_causal_chain_observed);
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

    let inputs = fs::read_to_string(ledger_args.output.join("input-frames.jsonl")).unwrap();
    for line in inputs.lines() {
        let provenance: ShadowInputFrameProvenance = serde_json::from_str(line).unwrap();
        assert_eq!(
            provenance.input_frame_id,
            format!("ledger-frame:{}", provenance.runtime_frame_id)
        );
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
