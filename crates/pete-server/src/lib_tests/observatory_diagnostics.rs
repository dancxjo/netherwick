fn diagnostic_snapshot(id: &str, t_ms: u64) -> ObservatoryNowSnapshot {
    ObservatoryNowSnapshot {
        snapshot_id: id.into(),
        now: blank_now(t_ms),
    }
}

fn diagnostic_envelope(sequence: u64, mut event: BrainEvent) -> SequencedBrainEvent {
    event.references.snapshot_id = Some(format!("snapshot-{sequence}"));
    SequencedBrainEvent { sequence, event }
}

fn test_diagnostic_bundle(
    events: Vec<SequencedBrainEvent>,
    snapshots: Vec<ObservatoryNowSnapshot>,
    gaps: Vec<BrainEventSequenceGap>,
    policy: DiagnosticAssetPolicy,
) -> DiagnosticBundle {
    let training = LiveTrainingStatus::default();
    build_diagnostic_bundle(DiagnosticBundleBuild {
        events,
        snapshots,
        gaps,
        transport: BrainEventTransportHealth {
            running: true,
            history_capacity: 100,
            ..Default::default()
        },
        training: &training,
        from_ms: 100,
        to_ms: 300,
        policy,
        partial: false,
    })
}

#[test]
fn diagnostic_export_redacts_sensitive_assets_but_preserves_manifest_identity() {
    let mut event = graph_event("camera", BrainEventType::Evidence, 100);
    event.payload = BrainEventPayload::Reference {
        reference: PayloadReference {
            id: "rgb-7".into(),
            locator: "/private/capture/rgb-7.png".into(),
            media_type: "image/png".into(),
            byte_len: Some(800_000),
            checksum: Some(diagnostic_sha256(b"pixels")),
            redacted: false,
        },
    };

    let bundle = test_diagnostic_bundle(
        vec![diagnostic_envelope(1, event)],
        vec![diagnostic_snapshot("snapshot-1", 100)],
        vec![],
        DiagnosticAssetPolicy::RedactSensitive,
    );

    assert_eq!(bundle.assets[0].disposition, "redacted");
    assert!(bundle.assets[0].reference.redacted);
    assert_eq!(bundle.assets[0].reference.locator, "redacted://rgb-7");
    assert!(bundle.assets[0].locator_sha256.starts_with("sha256:"));
    assert!(verify_diagnostic_bundle(&bundle).bundle_checksum_valid);
}

#[test]
fn export_links_observed_wall_time_events_to_monotonic_snapshots() {
    let now = blank_now(4_200);
    let event = BrainEvent::from_now_snapshot(
        "snapshot-linked",
        &now,
        1_784_683_256_368,
        Some("sim-clock".into()),
    );
    let retained = vec![
        diagnostic_snapshot("snapshot-old", 4_100),
        ObservatoryNowSnapshot {
            snapshot_id: "snapshot-linked".into(),
            now,
        },
    ];

    let selected =
        diagnostic_select_snapshots(&retained, &[SequencedBrainEvent { sequence: 1, event }]);

    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].snapshot_id, "snapshot-linked");
}

#[test]
fn import_verification_reports_missing_asset_manifest_entries() {
    let mut event = graph_event("audio", BrainEventType::Evidence, 100);
    event.payload = BrainEventPayload::Reference {
        reference: PayloadReference {
            id: "audio-2".into(),
            locator: "capture/audio-2.flac".into(),
            media_type: "audio/flac".into(),
            ..Default::default()
        },
    };
    let mut bundle = test_diagnostic_bundle(
        vec![diagnostic_envelope(1, event)],
        vec![diagnostic_snapshot("snapshot-1", 100)],
        vec![],
        DiagnosticAssetPolicy::ManifestOnly,
    );
    bundle.assets.clear();
    bundle.manifest.asset_count = 0;
    bundle = finalize_diagnostic_bundle(bundle);

    let report = verify_diagnostic_bundle(&bundle);

    assert_eq!(report.missing_references, ["audio-2"]);
    assert!(report.bundle_checksum_valid);
    assert!(report.replayable);
}

#[test]
fn import_verification_rejects_bundle_and_embedded_asset_checksum_failures() {
    let mut bundle = test_diagnostic_bundle(
        vec![],
        vec![diagnostic_snapshot("snapshot-1", 100)],
        vec![],
        DiagnosticAssetPolicy::ManifestOnly,
    );
    bundle.manifest.source_id = "tampered".into();
    assert!(!verify_diagnostic_bundle(&bundle).bundle_checksum_valid);

    bundle.assets.push(DiagnosticAssetEntry {
        reference: PayloadReference {
            id: "config".into(),
            media_type: "application/json".into(),
            checksum: Some(diagnostic_sha256(b"expected")),
            ..Default::default()
        },
        disposition: "embedded".into(),
        locator_sha256: diagnostic_sha256(b"config"),
        embedded_base64: Some(base64::engine::general_purpose::STANDARD.encode(b"wrong")),
    });
    bundle.manifest.asset_count = 1;
    bundle = finalize_diagnostic_bundle(bundle);
    let report = verify_diagnostic_bundle(&bundle);
    assert_eq!(report.invalid_asset_checksums, ["config"]);
    assert!(!report.replayable);
}

#[test]
fn bundle_checksum_is_stable_across_json_object_key_order() {
    let mut bundle = test_diagnostic_bundle(
        vec![],
        vec![diagnostic_snapshot("snapshot-1", 100)],
        vec![],
        DiagnosticAssetPolicy::ManifestOnly,
    );
    bundle.snapshots[0].now.extensions.insert(
        "unordered".into(),
        serde_json::from_str(r#"{"z":1,"a":2,"middle":{"y":3,"b":4}}"#).unwrap(),
    );
    bundle = finalize_diagnostic_bundle(bundle);
    let encoded = serde_json::to_string(&bundle).unwrap();
    let reordered = encoded.replace(
        r#""unordered":{"a":2,"middle":{"b":4,"y":3},"z":1}"#,
        r#""unordered":{"z":1,"middle":{"y":3,"b":4},"a":2}"#,
    );
    let decoded: DiagnosticBundle = serde_json::from_str(&reordered).unwrap();

    assert!(verify_diagnostic_bundle(&decoded).bundle_checksum_valid);
}

#[test]
fn partial_capture_keeps_declared_gaps_and_round_trips_for_offline_replay() {
    let snapshot_event =
        BrainEvent::from_now_snapshot("snapshot-1", &blank_now(100), 101, Some("clock-a".into()));
    let bundle = test_diagnostic_bundle(
        vec![diagnostic_envelope(1, snapshot_event)],
        vec![diagnostic_snapshot("snapshot-1", 100)],
        vec![BrainEventSequenceGap::new(
            2,
            5,
            SequenceGapReason::RetentionExpired,
        )],
        DiagnosticAssetPolicy::OmitHeavy,
    );
    let encoded = serde_json::to_vec(&bundle).unwrap();
    let decoded: DiagnosticBundle = serde_json::from_slice(&encoded).unwrap();
    let replay = ReplayBrainEventSource::from_recorded(
        ObservatorySourceIdentity {
            id: decoded.manifest.bundle_id.clone(),
            kind: ObservatorySourceKind::Capture,
            label: "diagnostic bundle".into(),
        },
        decoded.events.clone(),
    );

    let report = verify_diagnostic_bundle(&decoded);
    assert!(report.bundle_checksum_valid);
    assert!(report.partial);
    assert_eq!(report.declared_gaps, 1);
    assert_eq!(
        replay
            .query(&BrainEventQuery::default())
            .unwrap()
            .records
            .len(),
        1
    );
}

#[test]
fn verified_bundle_populates_the_same_server_history_and_snapshot_ui_contract() {
    let first =
        BrainEvent::from_now_snapshot("snapshot-1", &blank_now(100), 1_000, Some("clock-a".into()));
    let second =
        BrainEvent::from_now_snapshot("snapshot-2", &blank_now(200), 1_100, Some("clock-a".into()));
    let bundle = test_diagnostic_bundle(
        vec![
            SequencedBrainEvent {
                sequence: 4,
                event: first,
            },
            SequencedBrainEvent {
                sequence: 8,
                event: second,
            },
        ],
        vec![
            diagnostic_snapshot("snapshot-1", 100),
            diagnostic_snapshot("snapshot-2", 200),
        ],
        vec![],
        DiagnosticAssetPolicy::ManifestOnly,
    );

    let state = LiveViewState::from_diagnostic_bundle(bundle).unwrap();
    let history = state
        .observatory()
        .query(&BrainEventQuery::default())
        .unwrap();

    assert_eq!(history.records.len(), 4);
    assert_eq!(
        state
            .observatory_now_at_or_before(150)
            .unwrap()
            .selected
            .snapshot_id,
        "snapshot-1"
    );
    assert_eq!(state.session().unwrap().mode, "diagnostic-replay");
}

#[test]
fn invalid_bundle_cannot_start_replay_server_state() {
    let mut bundle = test_diagnostic_bundle(
        vec![],
        vec![diagnostic_snapshot("snapshot-1", 100)],
        vec![],
        DiagnosticAssetPolicy::ManifestOnly,
    );
    bundle.manifest.source_id = "tampered".into();

    assert!(LiveViewState::from_diagnostic_bundle(bundle).is_err());
}

#[test]
fn comparison_separates_value_from_trust_and_epoch_changes() {
    let mut left = diagnostic_snapshot("left", 100);
    left.now.extensions.insert(
        "belief".into(),
        serde_json::json!({"value": "open", "meta": {"trust": "conditional"}}),
    );
    let mut right = diagnostic_snapshot("right", 200);
    right.now.extensions.insert(
        "belief".into(),
        serde_json::json!({"value": "closed", "meta": {"trust": "trusted"}}),
    );
    let mut old_epoch = graph_event("epoch-1", BrainEventType::CalibrationTransition, 90);
    old_epoch.artifacts.push(ArtifactIdentity {
        kind: ArtifactKind::Calibration,
        id: "kinect-epoch-1".into(),
        version: Some("1".into()),
        checksum: Some("sha256:old".into()),
    });
    let mut new_epoch = graph_event("epoch-2", BrainEventType::CalibrationTransition, 190);
    new_epoch.artifacts.push(ArtifactIdentity {
        kind: ArtifactKind::Calibration,
        id: "kinect-epoch-2".into(),
        version: Some("2".into()),
        checksum: Some("sha256:new".into()),
    });

    let comparison =
        build_diagnostic_comparison(&left, &right, &[old_epoch, new_epoch], false, vec![]);

    assert!(comparison.fields.iter().any(|change| {
        change.path.ends_with("belief.value") && change.kind == DiagnosticChangeKind::Value
    }));
    assert!(comparison.fields.iter().any(|change| {
        change.path.ends_with("belief.meta.trust")
            && change.kind == DiagnosticChangeKind::ProvenanceTrust
    }));
    assert!(comparison
        .calibration_epochs
        .added
        .contains(&"kinect-epoch-2".into()));
}

#[test]
fn candidate_and_reprocessed_outputs_remain_distinct_from_baseline() {
    let left = diagnostic_snapshot("baseline", 100);
    let right = diagnostic_snapshot("candidate", 200);
    let mut baseline = graph_event("baseline", BrainEventType::Proposal, 90);
    baseline.kind = "locomotion.baseline.recorded".into();
    baseline.artifacts.push(ArtifactIdentity {
        kind: ArtifactKind::Model,
        id: "hardcoded-baseline".into(),
        version: Some("1".into()),
        checksum: Some("sha256:baseline".into()),
    });
    let mut candidate = graph_event("candidate", BrainEventType::Proposal, 190);
    candidate.kind = "locomotion.candidate.reprocessed".into();
    candidate.artifacts.push(ArtifactIdentity {
        kind: ArtifactKind::Model,
        id: "neat-candidate".into(),
        version: Some("7".into()),
        checksum: Some("sha256:candidate".into()),
    });

    let comparison =
        build_diagnostic_comparison(&left, &right, &[baseline, candidate], false, vec![]);

    assert!(comparison
        .model_artifacts
        .added
        .contains(&"neat-candidate".into()));
    assert!(comparison
        .recorded_reprocessed
        .added
        .iter()
        .any(|key| key.contains("candidate")));
    assert!(comparison.event_categories.contains_key("proposal"));
}

#[test]
fn observatory_diagnostic_ui_exports_verifies_and_compares_selected_times() {
    for marker in [
        "Diagnostic interval export and comparison controls",
        "Set interval start",
        "Export interval",
        "redact sensitive",
        "Set comparison A",
        "Compare A ↔ B",
        "Verify diagnostic bundle",
        "Value changes",
        "Provenance / trust changes",
        "Baseline / candidate models",
        "Recorded / reprocessed results",
        "raw/corrected pose/map paths",
        "/api/observatory/diagnostic-export?",
        "/api/observatory/diagnostic-verify",
        "/api/observatory/compare?",
        "name.startsWith('compare')?selectedOccurredMs():selectedObservedMs()",
        "REPLAY · INSPECTING",
    ] {
        assert!(OBSERVATORY_PAGE.contains(marker), "missing {marker}");
    }
}
