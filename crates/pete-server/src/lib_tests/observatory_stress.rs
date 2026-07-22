#[tokio::test]
async fn software_observatory_stress_harness_reports_bounded_truthful_behavior() {
    let directory = std::env::temp_dir().join(format!(
        "pete-observatory-stress-{}",
        Uuid::new_v4()
    ));
    let mut config = ObservatoryStressConfig::ci(&directory);
    config.events = 12_000;
    let report = run_observatory_stress(config).await.unwrap();
    assert_eq!(report.schema_version, 2);
    assert!(report.passed, "failed checks: {:?}", report.checks);
    assert!(!report.physical_pi_soak_performed);
    assert!(report.metrics.replay_order_matches);
    assert!(report.metrics.clock_reset_preserved);
    assert!(report.metrics.stalled_client_lagged);
    assert!(report.metrics.reconnect_received_live_event);
    assert!(report.metrics.injected_writer_failures > 0);
    assert!(report.metrics.max_ingress_depth <= report.config.ingress_capacity);
    assert!(report.metrics.max_history_depth <= report.config.history_capacity);
    let report_json = serde_json::to_value(&report).unwrap();
    assert!(report_json["metrics"].get("construct_only").is_some());
    assert!(report_json["metrics"]
        .get("construct_and_publish")
        .is_some());
    assert!(report_json["metrics"].get("baseline").is_none());
    assert!(report_json["metrics"].get("enabled").is_none());
    fs::remove_dir_all(directory).unwrap();
}
