#[test]
fn fixture_vision_evaluation_is_ordered_and_scores_tracking() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/vision-fixtures/manifest.json");
    let frames = load_vision_fixture_frames(&path).expect("fixture frames");
    let report = evaluate_vision_backend(
        std::sync::Arc::new(pete_sensors::ClassicalSaliencyBackend),
        pete_sensors::VisionPipelineConfig::raspberry_pi_5(),
        &frames,
    );

    assert_eq!(report.metrics.frames, 7);
    assert!(report.metrics.true_positives >= 3);
    assert_eq!(report.metrics.track_fragmentations, 0);
    assert_eq!(report.detections[0].frame_id, "positive-1");
    assert_eq!(report.detections[1].frame_id, "positive-2");
    assert_eq!(report.detections[0].track_id, report.detections[1].track_id);
}

#[test]
fn unavailable_baseline_reports_each_failure_without_panicking() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/vision-fixtures/manifest.json");
    let frames = load_vision_fixture_frames(&path).expect("fixture frames");
    let report = evaluate_vision_backend(
        std::sync::Arc::new(pete_sensors::UnavailableVisionBackend::new(
            "test model absent",
        )),
        pete_sensors::VisionPipelineConfig::raspberry_pi_5(),
        &frames,
    );

    assert_eq!(report.state, pete_now::VisionBackendState::Missing);
    assert_eq!(report.metrics.failures, frames.len());
    assert!(report.detections.is_empty());
}
