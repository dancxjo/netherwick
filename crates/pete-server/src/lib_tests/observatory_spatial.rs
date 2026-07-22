fn valid_depth_calibration() -> pete_now::DepthGeometryCalibration {
    pete_now::DepthGeometryCalibration {
        calibrated: true,
        validation: Some(pete_now::DepthCalibrationValidation {
            distance_sample_count: 4,
            min_test_distance_m: 0.4,
            max_test_distance_m: 3.0,
            max_plane_distance_error_m: 0.01,
            rgb_depth_boundary_error_px: 2.0,
        }),
        ..Default::default()
    }
}

fn spatial_response(
    now: pete_now::Now,
    map: Option<serde_json::Value>,
    rgb: bool,
) -> SpatialViewResponse {
    build_spatial_view(&calibration_selection(None, now), map, rgb, &[])
}

#[test]
fn kinect_only_detection_keeps_frame_model_track_depth_and_optional_lidar_truth() {
    let mut now = blank_now(100);
    now.kinect.geometry_calibration = Some(valid_depth_calibration());
    now.kinect.depth_width = 2;
    now.kinect.depth_height = 2;
    now.kinect.depth_m = vec![1.0, 1.1, 1.2, 1.3];
    now.objects.detections.push(pete_now::VisionDetection {
        source_frame_id: "rgb-1".into(),
        source_snapshot_id: "now-100".into(),
        track_id: Some("track-7".into()),
        labels: vec![pete_now::VisionLabelHypothesis {
            label: "cup".into(),
            confidence: 0.91,
        }],
        model: pete_now::VisionModelIdentity {
            backend: "tract".into(),
            model_id: "detector".into(),
            version: "5".into(),
            checksum: None,
        },
        geometry_trust: "trusted".into(),
        image_width: 640,
        image_height: 480,
        ..Default::default()
    });

    let response = spatial_response(now, None, false);
    let detection = &response.detections[0];

    assert_eq!(detection.source_frame_id, "rgb-1");
    assert_eq!(detection.track_id.as_deref(), Some("track-7"));
    assert_eq!(detection.model.model_id, "detector");
    assert!(response.lidar.contains("optional"));
}

#[test]
fn depth_only_snapshot_is_retained_without_claiming_rgb_or_map_alignment() {
    let mut now = blank_now(200);
    now.kinect.depth_width = 2;
    now.kinect.depth_height = 2;
    now.kinect.depth_m = vec![0.8, 1.0, 1.2, 1.4];

    let response = spatial_response(now, None, false);

    assert_eq!(response.depth.original_sample_count, 4);
    assert!(response.rgb_asset_url.is_none());
    assert_eq!(
        response.map_alignment,
        "unavailable; not substituted with current map"
    );
    assert!(!response.navigation_trusted);
}

#[test]
fn missing_world_position_exposes_reasons_instead_of_fabricating_coordinates() {
    let mut now = blank_now(300);
    now.objects.detections.push(pete_now::VisionDetection {
        source_frame_id: "rgb-3".into(),
        position: None,
        position_unavailable_reasons: vec!["depth association rejected: stale frame".into()],
        ..Default::default()
    });

    let response = spatial_response(now, None, false);

    assert!(response.detections[0].position.is_none());
    assert!(response.detections[0].position_unavailable_reasons[0].contains("stale"));
}

#[test]
fn remount_invalidates_rgb_depth_registration_even_with_measured_intrinsics() {
    let mut now = blank_now(400);
    now.kinect.depth_m = vec![1.0];
    now.kinect.depth_width = 1;
    now.kinect.depth_height = 1;
    now.kinect.geometry_calibration = Some(valid_depth_calibration());
    let mut estimate = pete_now::CalibrationStateMachine::new(
        pete_now::RigidTransform3::default(),
        1,
        pete_now::CalibrationStateConfig::default(),
    )
    .estimate()
    .clone();
    estimate.trust_state = pete_now::CalibrationTrustState::Invalidated;
    estimate.epoch.invalidation_reason = Some("mount moved".into());
    now.kinect.live_geometry_calibration = Some(estimate);

    let response = spatial_response(now, None, true);

    assert!(!response.depth.registration_trusted);
    assert!(response
        .depth
        .registration_reasons
        .iter()
        .any(|reason| reason.contains("not fully trusted")));
    assert!(!response.navigation_trusted);
}

#[test]
fn loop_correction_exposes_raw_corrected_comparison_and_navigation_trust() {
    let now = blank_now(500);
    let map = serde_json::json!({
        "world_projection": {"navigation_trusted": true, "reasons": []},
        "pose_graph": {"optimization": {"max_node_update_m": 0.42}},
        "pose_trail": [{"x_m": 0.0, "y_m": 0.0}, {"x_m": 1.0, "y_m": 0.2}],
        "cells": [], "semantic_cells": []
    });

    let response = spatial_response(now, Some(map), false);

    assert!(response.raw_corrected_comparison_available);
    assert!(response.navigation_trusted);
}

#[test]
fn heavy_assets_are_stripped_and_depth_history_stays_bounded() {
    let mut now = blank_now(600);
    now.eye_frame = Some(EyeFrame {
        rgbd_frame_id: None,
        device_timestamp_ms: None,
        width: 1,
        height: 1,
        format: EyeFrameFormat::Rgb8,
        captured_at_ms: 600,
        bytes: vec![1, 2, 3],
        source: None,
    });
    now.kinect.depth_m = vec![1.0; MAX_RETAINED_DEPTH_SAMPLES * 3];
    now.objects.detections.push(pete_now::VisionDetection {
        crop_rgb8: vec![9; 512],
        ..Default::default()
    });

    strip_observatory_heavy_payloads(&mut now);

    assert!(now.eye_frame.is_none());
    assert!(now.kinect.depth_m.len() <= MAX_RETAINED_DEPTH_SAMPLES);
    assert!(now.objects.detections[0].crop_rgb8.is_empty());
}

#[test]
fn observatory_spatial_ui_lazy_loads_assets_and_exposes_registration_truth() {
    for marker in [
        "Synchronized spatial view",
        "Load selected RGB asset",
        "registration ${data.depth.registration_trusted?'trusted':'REJECTED'}",
        "Point cloud unavailable for this historical snapshot",
        "position_unavailable_reasons",
        "/api/observatory/spatial",
    ] {
        assert!(OBSERVATORY_PAGE.contains(marker), "missing {marker}");
    }
}
