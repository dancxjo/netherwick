use super::*;
use pete_now::{EyeFrame, EyeFrameFormat, KinectSense, RangeSense};

fn snapshot_at(x_m: f32, y_m: f32, heading_rad: f32, beams: Vec<f32>) -> WorldSnapshot {
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.odometry = Pose2 {
        x_m,
        y_m,
        heading_rad,
    };
    snapshot.range = RangeSense {
        schema_version: 1,
        nearest_m: beams.iter().copied().reduce(f32::min),
        beams,
        ..RangeSense::default()
    };
    snapshot
}

fn range_sense(beams: &[f32]) -> RangeSense {
    RangeSense {
        schema_version: 1,
        beams: beams.to_vec(),
        nearest_m: beams.iter().copied().reduce(f32::min),
        ..RangeSense::default()
    }
}

fn kinect_snapshot_at(
    x_m: f32,
    y_m: f32,
    heading_rad: f32,
    depth_m: Vec<f32>,
    width: u32,
    height: u32,
) -> WorldSnapshot {
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.odometry = Pose2 {
        x_m,
        y_m,
        heading_rad,
    };
    snapshot.kinect = KinectSense {
        depth_m,
        depth_width: width,
        depth_height: height,
        depth_fx: 1.0,
        depth_fy: 1.0,
        depth_cx: 0.0,
        depth_cy: 0.0,
        min_depth_m: 0.1,
        max_depth_m: 8.0,
        depth_coordinate_system: Some("kinect_camera".to_string()),
        ..KinectSense::default()
    };
    snapshot
}

#[test]
fn beam_projection_uses_pose_heading_and_relative_angle() {
    let pose = Pose2 {
        x_m: 1.0,
        y_m: 2.0,
        heading_rad: std::f32::consts::FRAC_PI_2,
    };
    let endpoint = project_beam_endpoint(pose, 0.0, 1.5);
    assert!((endpoint.x_m - 1.0).abs() < 0.001);
    assert!((endpoint.y_m - 3.5).abs() < 0.001);
}

#[test]
fn occupancy_update_accumulates_endpoint_hits() {
    let mut map = LocalMap::new(MapConfig {
        resolution_m: 0.5,
        ..MapConfig::default()
    });
    let snapshot = snapshot_at(0.0, 0.0, 0.0, vec![1.0]);
    map.observe_snapshot(&snapshot, 100);
    map.observe_snapshot(&snapshot, 200);

    let key = cell_key(1.0, 0.0, map.config.resolution_m);
    let cell = map.cells.get(&key).expect("endpoint cell should exist");
    assert!(cell.occupied_score > 0.5);
    assert!(cell.occupied_score > cell.free_score);
}

#[test]
fn occupancy_update_uses_explicit_robot_frame_beam_angles() {
    let config = MapConfig {
        resolution_m: 0.5,
        ..MapConfig::default()
    };
    let mut snapshot = snapshot_at(0.0, 0.0, 0.0, vec![1.0]);
    snapshot.range.beam_angles_rad = vec![std::f32::consts::FRAC_PI_2];
    snapshot.range.frame = Some("robot_base".to_string());
    let observation = observation_from_snapshot(&snapshot, 100, config);

    assert_eq!(observation.range_beams.len(), 1);
    assert!((observation.range_beams[0].angle_rad - std::f32::consts::FRAC_PI_2).abs() < 0.001);
}

#[test]
fn free_space_is_marked_along_beam_before_hit() {
    let mut map = LocalMap::new(MapConfig {
        resolution_m: 0.25,
        ..MapConfig::default()
    });
    let snapshot = snapshot_at(0.0, 0.0, 0.0, vec![1.0]);
    map.observe_snapshot(&snapshot, 100);

    let free_key = cell_key(0.5, 0.0, map.config.resolution_m);
    let free = map.cells.get(&free_key).expect("free cell should exist");
    assert!(free.free_score > free.occupied_score);
}

#[test]
fn stale_observations_decay_and_empty_cells_are_removed() {
    let mut map = LocalMap::new(MapConfig {
        resolution_m: 0.5,
        decay_after_ms: 10,
        decay_per_tick: 1.0,
        ..MapConfig::default()
    });
    let snapshot = snapshot_at(0.0, 0.0, 0.0, vec![1.0]);
    map.observe_snapshot(&snapshot, 100);
    assert!(!map.cells.is_empty());

    map.decay_stale(111);
    assert!(map.cells.is_empty());
}

#[test]
fn map_grows_as_odometry_moves_through_sim_like_snapshots() {
    let mut map = LocalMap::new(MapConfig {
        resolution_m: 0.25,
        ..MapConfig::default()
    });

    map.observe_snapshot(&snapshot_at(0.0, 0.0, 0.0, vec![1.0]), 100);
    let first_cells = map.cells.len();
    map.observe_snapshot(&snapshot_at(1.0, 0.0, 0.0, vec![1.0]), 200);

    assert!(first_cells > 0);
    assert!(map.cells.len() > first_cells);
    assert_eq!(map.pose_history.len(), 2);
    assert_eq!(map.pose_graph.nodes.len(), 2);
    assert_eq!(map.pose_graph.edges.len(), 1);
    assert!(matches!(
        map.pose_graph.edges[0].source,
        PoseEdgeSource::Odometry
    ));
    assert_eq!(map.submaps.len(), 2);
    assert_eq!(map.summary().remap.submaps, 2);
    assert_eq!(map.summary().label, MAP_LABEL);
    assert_eq!(map.summary().slam_status.mode, SlamMode::MappingOnly);
    assert!(map
        .summary()
        .slam_status
        .reasons
        .iter()
        .any(|reason| reason.contains("no scan-match correction")));
}

#[test]
fn remap_rebuilds_occupancy_from_optimized_submap_node_poses() {
    let mut map = LocalMap::new(MapConfig {
        resolution_m: 0.1,
        pose_graph_min_node_distance_m: 0.01,
        ..MapConfig::default()
    });
    map.observe_snapshot(&snapshot_at(0.0, 0.0, 0.0, vec![1.0]), 100);

    let original_key = cell_key(1.0, 0.0, map.config.resolution_m);
    assert!(map
        .cells
        .get(&original_key)
        .is_some_and(|cell| cell.occupied_score > cell.free_score));

    map.pose_graph.nodes[0].pose_estimate.pose.x_m = 0.5;
    map.rebuild_occupancy_from_submaps();

    let remapped_key = cell_key(1.5, 0.0, map.config.resolution_m);
    assert!(map
        .cells
        .get(&remapped_key)
        .is_some_and(|cell| cell.occupied_score > cell.free_score));
    assert!(map
        .cells
        .get(&original_key)
        .map_or(true, |cell| cell.occupied_score <= cell.free_score));
    assert_eq!(map.summary().remap.submaps, 1);
    assert!(map.summary().remap.generation >= 2);
}

#[test]
fn scan_matching_corrects_small_odometry_drift_against_existing_occupancy() {
    let config = MapConfig {
        resolution_m: 0.1,
        scan_match_xy_window_m: 0.2,
        scan_match_theta_window_rad: 0.0,
        scan_match_min_occupied_cells: 1,
        scan_match_min_hit_beams: 1,
        pose_graph_min_node_distance_m: 0.01,
        pose_graph_max_ticks_between_nodes: 1,
        ..MapConfig::default()
    };
    let mut map = LocalMap::new(config);
    let observation = observation_from_parts(
        pose(0.0, 0.0, 0.0),
        0.75,
        &range_sense(&[1.0]),
        serde_json::json!({"frame_id":"seed"}),
        150,
        map.config,
    );
    map.integrate_observation(observation);
    for y in [-0.1, 0.0, 0.1] {
        let key = cell_key(1.0, y, map.config.resolution_m);
        map.cells.insert(
            key,
            OccupancyCell {
                key,
                occupied_score: 0.9,
                free_score: 0.0,
                confidence: 0.9,
                last_seen_ms: 100,
            },
        );
    }

    let observation = observation_from_parts(
        pose(0.12, 0.0, 0.0),
        0.75,
        &range_sense(&[1.0]),
        serde_json::json!({"frame_id":"drifted"}),
        200,
        map.config,
    );
    map.integrate_observation(observation);

    let corrected = map.pose_history.last().unwrap();
    assert_eq!(corrected.source, "odometry+occupancy_scan_match");
    assert!(corrected.pose.x_m.abs() < 0.08);
    assert!(corrected.confidence > 0.75);
    assert_eq!(map.pose_graph.nodes.len(), 2);
    assert_eq!(map.pose_graph.edges.len(), 1);
    assert!(matches!(
        map.pose_graph.edges[0].source,
        PoseEdgeSource::ScanMatch { .. }
    ));
    let summary = map.summary();
    assert_eq!(summary.scan_match_edges, 1);
    assert_eq!(summary.slam_status.mode, SlamMode::LocalScanMatched);
    assert!(summary.slam_status.local_scan_matching_active);
}

#[test]
fn live_loop_candidate_empty_path_preserves_scan_matched_behavior() {
    let config = MapConfig {
        resolution_m: 0.25,
        pose_graph_min_node_distance_m: 0.01,
        ..MapConfig::default()
    };
    let mut baseline = LocalMap::new(config);
    let mut candidate_aware = LocalMap::new(config);
    let first = observation_from_parts(
        pose(0.0, 0.0, 0.0),
        0.75,
        &range_sense(&[1.0]),
        serde_json::json!({"frame_id":"seed"}),
        100,
        config,
    );
    let second = observation_from_parts(
        pose(1.0, 0.0, 0.0),
        0.75,
        &range_sense(&[1.0]),
        serde_json::json!({"frame_id":"next"}),
        200,
        config,
    );

    baseline.integrate_observation(first.clone());
    baseline.integrate_observation(second.clone());
    candidate_aware.integrate_observation_with_loop_candidates(first, &[]);
    candidate_aware.integrate_observation_with_loop_candidates(second, &[]);

    assert_eq!(candidate_aware.pose_history, baseline.pose_history);
    assert_eq!(candidate_aware.pose_graph, baseline.pose_graph);
    assert_eq!(candidate_aware.summary().loop_closure_edges, 0);
}

#[test]
fn live_loop_candidate_low_confidence_is_rejected_with_reason() {
    let config = live_loop_test_config();
    let mut map = seeded_live_loop_map(config);
    let weak = live_loop_candidate("entity_constellation", 0.60, "seed", "return");
    let observation = observation_from_parts(
        pose(0.05, 0.0, 0.0),
        0.75,
        &range_sense(&[1.0]),
        serde_json::json!({"frame_id":"return"}),
        300,
        config,
    );

    let summary = map.integrate_observation_with_loop_candidates(observation, &[weak]);

    assert_eq!(summary.loop_closure_edges, 1);
    assert_eq!(summary.loop_closures_accepted, 0);
    assert_eq!(summary.loop_closures_rejected, 1);
    let edge = map.pose_graph.edges.last().unwrap();
    assert!(!edge.active);
    assert!(edge
        .rejection_reason
        .as_deref()
        .is_some_and(|reason| reason.contains("below gate")));
}

#[test]
fn live_entity_constellation_candidate_adds_active_loop_edge() {
    let config = live_loop_test_config();
    let mut map = seeded_live_loop_map(config);
    let candidate = live_loop_candidate("entity_constellation", 0.94, "seed", "return");
    let observation = observation_from_parts(
        pose(0.05, 0.0, 0.0),
        0.75,
        &range_sense(&[1.0]),
        serde_json::json!({"frame_id":"return"}),
        300,
        config,
    );

    let summary = map.integrate_observation_with_loop_candidates(observation, &[candidate]);

    assert_eq!(summary.loop_closure_edges, 1);
    assert_eq!(summary.loop_closures_accepted, 1);
    assert_eq!(summary.loop_closures_rejected, 0);
    assert_eq!(summary.slam_status.mode, SlamMode::LoopClosedPoseGraph);
    assert!(summary.slam_status.loop_closure_active);
    assert!(summary.slam_status.pose_graph_optimized);
    let edge = map.pose_graph.edges.last().unwrap();
    assert!(edge.active);
    assert_eq!(edge.to, "live-pose-0");
    assert!(matches!(
        edge.source,
        PoseEdgeSource::LoopClosureCandidate { ref kind, .. } if kind == "entity_constellation"
    ));
}

#[test]
fn live_loop_rejections_explain_bad_targets_and_weak_geometry() {
    let config = live_loop_test_config();
    let mut map = seeded_live_loop_map(config);
    let current_target = live_loop_candidate("entity_constellation", 0.94, "return", "return");
    let weak_geometry = LoopClosureCandidateInput {
        target_frame_id: Some("seed".to_string()),
        source_frame_id: Some("return".to_string()),
        ..live_loop_candidate("entity_constellation", 0.94, "seed", "return")
    };
    let observation = observation_from_parts(
        pose(0.05, 0.0, 0.0),
        0.75,
        &range_sense(&[3.0]),
        serde_json::json!({"frame_id":"return"}),
        300,
        config,
    );

    map.integrate_observation_with_loop_candidates(observation, &[current_target, weak_geometry]);

    let reasons = map
        .pose_graph
        .edges
        .iter()
        .filter_map(|edge| edge.rejection_reason.as_deref())
        .collect::<Vec<_>>();
    assert!(reasons
        .iter()
        .any(|reason| reason.contains("current/source frame")));
    assert!(reasons
        .iter()
        .any(|reason| reason.contains("geometric occupancy agreement")));
}

#[test]
fn live_loop_candidate_rebuilds_occupancy_after_optimization() {
    let config = live_loop_test_config();
    let mut map = seeded_live_loop_map(config);
    let generation_before = map.remap_summary.generation;
    let candidate = live_loop_candidate("entity_constellation", 0.94, "seed", "return");
    let observation = observation_from_parts(
        pose(0.05, 0.0, 0.0),
        0.75,
        &range_sense(&[1.0]),
        serde_json::json!({"frame_id":"return"}),
        300,
        config,
    );

    let summary = map.integrate_observation_with_loop_candidates(observation, &[candidate]);

    assert!(summary.pose_graph_optimization.active_edges >= 2);
    assert_eq!(summary.remap.submaps, map.submaps.len());
    assert!(summary.remap.generation > generation_before);
    assert!(!map.cells.is_empty());
}

#[test]
fn kinect_point_transforms_into_odometry_world_frame() {
    let config = PointCloudConfig {
        voxel_size_m: 0.1,
        camera_height_m: 0.2,
        ..PointCloudConfig::default()
    };
    let point = Point3D {
        x_m: 0.0,
        y_m: 0.0,
        z_m: 1.0,
    };
    let world = transform_point_to_world(
        point,
        PointCloudFrame::KinectCamera,
        pose(1.0, 2.0, std::f32::consts::FRAC_PI_2),
        OrientationEstimate {
            yaw_rad: Some(std::f32::consts::FRAC_PI_2),
            yaw_source: YawSource::OdometryHeading,
            ..OrientationEstimate::default()
        },
        config,
    );
    assert!((world.x_m - 1.0).abs() < 0.001);
    assert!((world.y_m - 3.0).abs() < 0.001);
    assert!((world.z_m - 0.2).abs() < 0.001);
}

#[test]
fn camera_to_robot_zero_rotation_maps_forward_and_height() {
    let config = PointCloudConfig {
        camera_height_m: 0.25,
        camera_forward_m: 0.10,
        ..PointCloudConfig::default()
    };

    let robot = camera_point_to_robot(
        Point3D {
            x_m: 0.0,
            y_m: 0.0,
            z_m: 1.0,
        },
        config,
    );

    assert!((robot.x_m - 1.10).abs() < 0.001);
    assert!(robot.y_m.abs() < 0.001);
    assert!((robot.z_m - 0.25).abs() < 0.001);
}

#[test]
fn plausible_floor_ray_lands_near_robot_floor() {
    let config = PointCloudConfig {
        camera_height_m: 0.5,
        ..PointCloudConfig::default()
    };
    let robot = camera_point_to_robot(
        Point3D {
            x_m: 0.0,
            y_m: 0.5,
            z_m: 1.0,
        },
        config,
    );

    assert!(robot.x_m > 0.9);
    assert!(robot.y_m.abs() < 0.001);
    assert!(robot.z_m.abs() < 0.001);
}

#[test]
fn kinect_point_transform_applies_camera_pitch_before_world_yaw() {
    let config = PointCloudConfig {
        camera_height_m: 0.5,
        camera_pitch_rad: 0.25,
        ..PointCloudConfig::default()
    };
    let point = Point3D {
        x_m: 0.0,
        y_m: 0.0,
        z_m: 2.0,
    };

    let robot = camera_point_to_robot(point, config);
    let world = transform_point_to_world(
        point,
        PointCloudFrame::KinectCamera,
        pose(0.0, 0.0, std::f32::consts::FRAC_PI_2),
        OrientationEstimate {
            yaw_rad: Some(std::f32::consts::FRAC_PI_2),
            yaw_source: YawSource::OdometryHeading,
            ..OrientationEstimate::default()
        },
        config,
    );

    assert!(robot.z_m < 0.5);
    assert!((world.x_m + robot.y_m).abs() < 0.001);
    assert!((world.y_m - robot.x_m).abs() < 0.001);
    assert!((world.z_m - robot.z_m).abs() < 0.001);
}

#[test]
fn positive_pitch_lowers_straight_ahead_points() {
    let zero = camera_point_to_robot(
        Point3D {
            x_m: 0.0,
            y_m: 0.0,
            z_m: 1.0,
        },
        PointCloudConfig {
            camera_height_m: 0.5,
            ..PointCloudConfig::default()
        },
    );
    let pitched = camera_point_to_robot(
        Point3D {
            x_m: 0.0,
            y_m: 0.0,
            z_m: 1.0,
        },
        PointCloudConfig {
            camera_height_m: 0.5,
            camera_pitch_rad: 10.0_f32.to_radians(),
            ..PointCloudConfig::default()
        },
    );

    assert!(pitched.z_m < zero.z_m);
    assert!(pitched.x_m < zero.x_m);
}

#[test]
fn positive_roll_raises_left_floor_relative_to_right() {
    let config = PointCloudConfig {
        camera_height_m: 0.5,
        camera_roll_rad: 10.0_f32.to_radians(),
        ..PointCloudConfig::default()
    };
    let left = camera_point_to_robot(
        Point3D {
            x_m: -0.25,
            y_m: 0.5,
            z_m: 1.0,
        },
        config,
    );
    let right = camera_point_to_robot(
        Point3D {
            x_m: 0.25,
            y_m: 0.5,
            z_m: 1.0,
        },
        config,
    );

    assert!(left.y_m > 0.0);
    assert!(right.y_m < 0.0);
    assert!(left.z_m > right.z_m);
}

#[test]
fn imu_roll_pitch_correction_changes_world_height_before_yaw() {
    let config = PointCloudConfig {
        camera_height_m: 0.5,
        ..PointCloudConfig::default()
    };
    let point = Point3D {
        x_m: 0.0,
        y_m: 0.0,
        z_m: 1.0,
    };
    let uncorrected = transform_point_to_world(
        point,
        PointCloudFrame::KinectCamera,
        pose(0.0, 0.0, 0.0),
        OrientationEstimate {
            yaw_rad: Some(0.0),
            yaw_source: YawSource::OdometryHeading,
            ..OrientationEstimate::default()
        },
        config,
    );
    let corrected = transform_point_to_world(
        point,
        PointCloudFrame::KinectCamera,
        pose(0.0, 0.0, 0.0),
        OrientationEstimate {
            pitch_rad: Some(10.0_f32.to_radians()),
            yaw_rad: Some(0.0),
            roll_pitch_from_imu: true,
            yaw_source: YawSource::OdometryHeading,
            ..OrientationEstimate::default()
        },
        config,
    );

    assert!(corrected.z_m < uncorrected.z_m);
    assert!(corrected.x_m > uncorrected.x_m);
}

#[test]
fn imu_orientation_contract_ignores_invalid_one_value_shape() {
    let hardware = orientation_from_imu(
        &ImuSense {
            orientation: vec![0.1, -0.2],
            ..ImuSense::default()
        },
        0.7,
    );
    assert_eq!(hardware.roll_rad, Some(0.1));
    assert_eq!(hardware.pitch_rad, Some(-0.2));
    assert_eq!(hardware.yaw_rad, Some(0.7));
    assert_eq!(hardware.yaw_source, YawSource::OdometryHeading);
    assert!(hardware.roll_pitch_from_imu);

    let sim = orientation_from_imu(
        &ImuSense {
            orientation: vec![0.0, 0.0, 1.2],
            ..ImuSense::default()
        },
        0.7,
    );
    assert_eq!(sim.roll_rad, Some(0.0));
    assert_eq!(sim.pitch_rad, Some(0.0));
    assert_eq!(sim.yaw_rad, Some(1.2));
    assert_eq!(sim.yaw_source, YawSource::ImuOrientation);

    let invalid_one_value = orientation_from_imu(
        &ImuSense {
            orientation: vec![1.4],
            ..ImuSense::default()
        },
        0.7,
    );
    assert_eq!(invalid_one_value.roll_rad, None);
    assert_eq!(invalid_one_value.pitch_rad, None);
    assert_eq!(invalid_one_value.yaw_rad, Some(0.7));
    assert_eq!(invalid_one_value.yaw_source, YawSource::OdometryHeading);
    assert!(!invalid_one_value.roll_pitch_from_imu);
}

#[test]
fn implausible_gravity_roll_pitch_is_not_applied_to_world_cloud() {
    let orientation = orientation_from_imu(
        &ImuSense {
            orientation: vec![120.0_f32.to_radians(), 62.0_f32.to_radians()],
            ..ImuSense::default()
        },
        0.3,
    );

    assert_eq!(orientation.roll_rad, None);
    assert_eq!(orientation.pitch_rad, None);
    assert_eq!(orientation.yaw_rad, Some(0.3));
    assert_eq!(orientation.yaw_source, YawSource::OdometryHeading);
    assert!(!orientation.roll_pitch_from_imu);
}

#[test]
fn voxel_cloud_merges_points_and_marks_stable() {
    let mut cloud = VoxelPointCloud::new(PointCloudConfig {
        voxel_size_m: 0.25,
        stable_seen_count: 2,
        stable_confidence: 0.2,
        ..PointCloudConfig::default()
    });
    let snapshot = kinect_snapshot_at(0.0, 0.0, 0.0, vec![1.0], 1, 1);

    cloud.observe_snapshot(&snapshot, 100);
    cloud.observe_snapshot(&snapshot, 200);

    assert_eq!(cloud.voxels.len(), 1);
    let voxel = cloud.voxels.values().next().unwrap();
    assert!(voxel.stable);
    assert_eq!(voxel.seen_count, 2);
    assert!(voxel.confidence > 0.2);
}

#[test]
fn pointcloud_projection_preserves_depth_pixel_provenance() {
    let mut snapshot = kinect_snapshot_at(0.0, 0.0, 0.0, vec![1.0, 2.0, 0.0, 3.0], 2, 2);
    snapshot.kinect.captured_at_ms = 123;

    let observation = pointcloud_observation_from_snapshot(
        &snapshot,
        500,
        PointCloudConfig {
            max_points_per_observation: 16,
            ..PointCloudConfig::default()
        },
    )
    .unwrap();

    assert_eq!(observation.points.len(), 3);
    assert_eq!(observation.points[1].depth_index, Some(1));
    assert_eq!(observation.points[1].depth_uv, Some([1, 0]));
    assert_eq!(observation.points[1].depth_image_size, Some([2, 2]));
    assert_eq!(
        observation.points[1].source_frame_id.as_deref(),
        Some("kinect-depth-123")
    );
}

#[test]
fn tilted_lidar_ground_returns_feed_3d_cloud_but_not_planar_obstacles() {
    let range = RangeSense {
        schema_version: 1,
        captured_at_ms: 123,
        beams: vec![2.0_f32.sqrt()],
        nearest_m: Some(2.0_f32.sqrt()),
        beam_angles_rad: vec![0.0],
        frame: Some("hls_lfcd2".to_string()),
        source: Some("hls_lfcd2".to_string()),
        extrinsics: Some(RangeExtrinsics {
            height_m: 1.0,
            pitch_rad: std::f32::consts::FRAC_PI_4,
            ..RangeExtrinsics::default()
        }),
    };
    let observation = pointcloud_observation_from_range(
        &range,
        pose(0.0, 0.0, 0.0),
        OrientationEstimate::default(),
        0.9,
        500,
        PointCloudConfig::default(),
    )
    .expect("tilted lidar observation");

    assert_eq!(observation.frame, PointCloudFrame::RobotBase);
    assert_eq!(observation.source, "hls_lfcd2");
    assert_eq!(observation.points.len(), 1);
    assert!((observation.points[0].position.x_m - 1.0).abs() < 0.001);
    assert!(observation.points[0].position.y_m.abs() < 0.001);
    assert!(observation.points[0].position.z_m.abs() < 0.001);

    let mut snapshot = WorldSnapshot::default();
    snapshot.range = range;
    let planar = observation_from_snapshot(&snapshot, 500, MapConfig::default());
    assert!(planar.range_beams.is_empty());
}

#[test]
fn robot_spin_accumulates_tilted_lidar_plane_into_world_cloud() {
    let mut cloud = VoxelPointCloud::new(PointCloudConfig {
        voxel_size_m: 0.05,
        ..PointCloudConfig::default()
    });
    let mut snapshot = WorldSnapshot::default();
    snapshot.range = RangeSense {
        schema_version: 1,
        captured_at_ms: 100,
        beams: vec![1.0],
        nearest_m: Some(1.0),
        beam_angles_rad: vec![0.0],
        source: Some("hls_lfcd2".to_string()),
        extrinsics: Some(RangeExtrinsics {
            height_m: 0.5,
            pitch_rad: 30.0_f32.to_radians(),
            ..RangeExtrinsics::default()
        }),
        ..RangeSense::default()
    };

    snapshot.body.odometry.heading_rad = 0.0;
    cloud.observe_snapshot(&snapshot, 100);
    snapshot.body.odometry.heading_rad = std::f32::consts::FRAC_PI_2;
    cloud.observe_snapshot(&snapshot, 150);
    assert_eq!(
        cloud.observations, 1,
        "cached scan must not smear across poses"
    );
    snapshot.range.captured_at_ms = 200;
    cloud.observe_snapshot(&snapshot, 200);

    assert_eq!(cloud.observations, 2);
    assert_eq!(cloud.raw_points_seen, 2);
    assert_eq!(cloud.voxels.len(), 2);
    assert!(cloud
        .voxels
        .values()
        .any(|point| point.position.x_m > 0.8 && point.position.y_m.abs() < 0.05));
    assert!(cloud
        .voxels
        .values()
        .any(|point| point.position.y_m > 0.8 && point.position.x_m.abs() < 0.05));
}

#[test]
fn pointcloud_projection_samples_rgb_eye_frame_by_depth_pixel() {
    let mut snapshot = kinect_snapshot_at(0.0, 0.0, 0.0, vec![1.0, 1.0, 1.0, 1.0], 2, 2);
    snapshot.eye_frame = Some(EyeFrame {
        captured_at_ms: 123,
        width: 2,
        height: 2,
        format: EyeFrameFormat::Rgb8,
        bytes: vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0],
        source: Some("kinect-freenect-rgb".to_string()),
    });

    let observation = pointcloud_observation_from_snapshot(
        &snapshot,
        500,
        PointCloudConfig {
            max_points_per_observation: 16,
            ..PointCloudConfig::default()
        },
    )
    .unwrap();

    assert_eq!(observation.points.len(), 4);
    assert_eq!(observation.points[0].color_rgb, Some([255, 0, 0]));
    assert_eq!(observation.points[1].color_rgb, Some([0, 255, 0]));
    assert_eq!(observation.points[2].color_rgb, Some([0, 0, 255]));
    assert_eq!(observation.points[3].color_rgb, Some([255, 255, 0]));
}

#[test]
fn stationary_rotate_world_frame_observations_merge_into_stable_belief() {
    let mut cloud = VoxelPointCloud::new(PointCloudConfig {
        voxel_size_m: 0.25,
        stable_seen_count: 3,
        stable_confidence: 0.2,
        ..PointCloudConfig::default()
    });

    for (t_ms, heading_rad) in [
        (100, 0.0),
        (200, std::f32::consts::FRAC_PI_2),
        (300, std::f32::consts::PI),
    ] {
        cloud.integrate_observation(PointCloudObservation {
            frame: PointCloudFrame::OdometryWorld,
            pose: PoseEstimate {
                pose: pose(0.0, 0.0, heading_rad),
                confidence: 0.9,
                covariance: [0.01, 0.01, 0.02],
                source: "rotate-test".to_string(),
                t_ms,
            },
            orientation: OrientationEstimate {
                roll_rad: Some(0.02),
                pitch_rad: Some(-0.01),
                yaw_rad: Some(heading_rad),
                roll_pitch_from_imu: true,
                yaw_source: YawSource::OdometryHeading,
            },
            points: vec![PointCloudPoint {
                position: Point3D {
                    x_m: 1.0,
                    y_m: 0.0,
                    z_m: 0.4,
                },
                color_rgb: None,
                confidence: 1.0,
                depth_index: None,
                depth_uv: None,
                depth_image_size: None,
                source_frame_id: None,
            }],
            source: "rotate-test".to_string(),
            t_ms,
            metadata: serde_json::json!({}),
        });
    }

    assert_eq!(cloud.voxels.len(), 1);
    let voxel = cloud.voxels.values().next().unwrap();
    assert!(voxel.stable);
    assert!((voxel.position.x_m - 1.0).abs() < 0.001);
    assert!((voxel.position.y_m - 0.0).abs() < 0.001);
    assert_eq!(
        cloud.orientation_status.yaw_source,
        YawSource::OdometryHeading
    );
    assert!(cloud.orientation_status.roll_pitch_corrected);
}

#[test]
fn local_world_belief_clusters_stable_voxels_into_surface_hypotheses() {
    let mut cloud = VoxelPointCloud::new(PointCloudConfig {
        voxel_size_m: 0.1,
        stable_seen_count: 1,
        stable_confidence: 0.1,
        ..PointCloudConfig::default()
    });
    for y in [0.0, 0.1, 0.2, 0.3] {
        for z in [0.1, 0.2, 0.3, 0.4] {
            cloud.integrate_observation(PointCloudObservation {
                frame: PointCloudFrame::OdometryWorld,
                pose: PoseEstimate {
                    pose: pose(0.0, 0.0, 0.0),
                    confidence: 0.9,
                    covariance: [0.01, 0.01, 0.02],
                    source: "surface-test".to_string(),
                    t_ms: 100,
                },
                orientation: OrientationEstimate {
                    yaw_rad: Some(0.0),
                    yaw_source: YawSource::OdometryHeading,
                    ..OrientationEstimate::default()
                },
                points: vec![PointCloudPoint {
                    position: Point3D {
                        x_m: 1.0,
                        y_m: y,
                        z_m: z,
                    },
                    color_rgb: None,
                    confidence: 1.0,
                    depth_index: None,
                    depth_uv: None,
                    depth_image_size: None,
                    source_frame_id: None,
                }],
                source: "surface-test".to_string(),
                t_ms: 100,
                metadata: serde_json::json!({}),
            });
        }
    }

    let belief = cloud.local_world_belief();
    assert_eq!(belief.stable_voxels, 16);
    assert!(belief
        .stable_surfaces
        .iter()
        .any(|surface| surface.kind == WorldSurfaceKind::WallLike));
    assert!(belief.stable_blobs.is_empty());
}

#[test]
fn voxel_cloud_ages_transient_points_and_bounds_growth() {
    let mut cloud = VoxelPointCloud::new(PointCloudConfig {
        voxel_size_m: 0.1,
        max_voxels: 2,
        decay_after_ms: 10,
        decay_per_tick: 0.05,
        transient_after_ms: 20,
        ..PointCloudConfig::default()
    });
    cloud.observe_snapshot(&kinect_snapshot_at(0.0, 0.0, 0.0, vec![1.0], 1, 1), 100);
    cloud.observe_snapshot(&kinect_snapshot_at(1.0, 0.0, 0.0, vec![1.0], 1, 1), 110);
    cloud.observe_snapshot(&kinect_snapshot_at(2.0, 0.0, 0.0, vec![1.0], 1, 1), 120);

    assert_eq!(cloud.voxels.len(), 2);
    cloud.decay_stale(200);
    assert!(cloud.voxels.values().any(|voxel| voxel.transient));
}

#[test]
fn pose_graph_adds_nodes_by_motion_and_odometry_edges() {
    let mut builder = PoseGraphBuilder::new(PoseGraphConfig {
        min_node_distance_m: 0.5,
        min_node_heading_delta_rad: 0.5,
        max_ticks_between_nodes: 10,
        ..PoseGraphConfig::default()
    });
    builder.observe(pose(0.0, 0.0, 0.0), 100, Some("frame-a".to_string()), &[]);
    builder.observe(pose(0.2, 0.0, 0.0), 200, Some("frame-b".to_string()), &[]);
    builder.observe(pose(0.6, 0.0, 0.0), 300, Some("frame-c".to_string()), &[]);

    let graph = builder.finish();
    assert_eq!(graph.nodes.len(), 2);
    assert_eq!(graph.edges.len(), 1);
    assert!(matches!(graph.edges[0].source, PoseEdgeSource::Odometry));
    assert_eq!(graph.edges[0].from, "pose-0");
    assert_eq!(graph.edges[0].to, "pose-1");
    assert!((graph.edges[0].transform.x_m - 0.6).abs() < 0.001);
}

#[test]
fn pose_graph_gates_loop_candidates_and_reports_rejections() {
    let mut builder = PoseGraphBuilder::new(PoseGraphConfig {
        min_node_distance_m: 0.5,
        min_loop_confidence: 0.85,
        loop_target_max_distance_m: 0.5,
        ..PoseGraphConfig::default()
    });
    builder.observe(pose(0.0, 0.0, 0.0), 100, Some("frame-a".to_string()), &[]);
    builder.observe(pose(1.0, 0.0, 0.0), 200, Some("frame-b".to_string()), &[]);

    let accepted = LoopClosureCandidateInput {
        target_pose: pose(0.0, 0.0, 0.0),
        confidence: 0.93,
        similarity: 0.94,
        kind: "same_place".to_string(),
        target_frame_id: Some("frame-a".to_string()),
        source_frame_id: Some("frame-a".to_string()),
        source_experience_id: Some("experience-a".to_string()),
        source_instant_frame_id: Some("frame-a".to_string()),
        source_vector_refs: vec!["teacher:a".to_string()],
        source_vector_id: Some("scene-a".to_string()),
        query_vector_id: Some("scene-b".to_string()),
        query_experience_id: Some("experience-b".to_string()),
    };
    let rejected = LoopClosureCandidateInput {
        confidence: 0.60,
        query_vector_id: Some("weak-scene".to_string()),
        ..accepted.clone()
    };
    builder.observe(
        pose(1.0, 0.0, 0.0),
        300,
        Some("frame-c".to_string()),
        &[accepted, rejected],
    );

    let report = builder.finish_report();
    assert_eq!(report.nodes, 2);
    assert_eq!(report.odometry_edges, 1);
    assert_eq!(report.loop_candidate_edges, 2);
    assert_eq!(report.active_loop_candidate_edges, 1);
    assert_eq!(report.rejected_loop_candidates, 1);
    assert_eq!(report.confidence_distribution.buckets["0.85-0.94"], 1);
    assert_eq!(report.confidence_distribution.buckets["0.50-0.69"], 1);
    assert!(report.rejected_candidates[0].reason.contains("below gate"));
}

#[test]
fn pose_graph_optimizer_reduces_loop_closure_error() {
    let mut graph = PoseGraph {
        nodes: vec![
            test_pose_node("pose-0", 0.0, 0),
            test_pose_node("pose-1", 1.2, 100),
            test_pose_node("pose-2", 2.4, 200),
        ],
        edges: vec![
            test_edge(
                "pose-0",
                "pose-1",
                pose(1.0, 0.0, 0.0),
                PoseEdgeSource::Odometry,
                0.7,
            ),
            test_edge(
                "pose-1",
                "pose-2",
                pose(1.0, 0.0, 0.0),
                PoseEdgeSource::Odometry,
                0.7,
            ),
            test_edge(
                "pose-0",
                "pose-2",
                pose(2.0, 0.0, 0.0),
                PoseEdgeSource::LoopClosureCandidate {
                    kind: "same_place_geometry".to_string(),
                    target_frame_id: Some("pose-2".to_string()),
                    source_frame_id: Some("pose-0".to_string()),
                    source_experience_id: None,
                    source_instant_frame_id: None,
                    source_vector_refs: Vec::new(),
                    source_vector_id: None,
                    query_vector_id: None,
                    query_experience_id: None,
                },
                0.95,
            ),
        ],
    };

    let summary = graph.optimize_anchored(PoseGraphOptimizationConfig {
        iterations: 30,
        step_size: 0.6,
        ..PoseGraphOptimizationConfig::default()
    });

    assert!(summary.final_mean_error < summary.initial_mean_error);
    assert!(graph.nodes[2].pose_estimate.pose.x_m < 2.4);
    assert_eq!(graph.nodes[0].pose_estimate.pose.x_m, 0.0);
    assert!(summary.active_edges >= 3);
}

fn pose(x_m: f32, y_m: f32, heading_rad: f32) -> Pose2 {
    Pose2 {
        x_m,
        y_m,
        heading_rad,
    }
}

fn test_pose_node(id: &str, x_m: f32, t_ms: TimeMs) -> PoseNode {
    PoseNode {
        id: id.to_string(),
        pose_estimate: PoseEstimate {
            pose: pose(x_m, 0.0, 0.0),
            confidence: 0.8,
            covariance: [0.05, 0.05, 0.1],
            source: "test".to_string(),
            t_ms,
        },
        t_ms,
        source_frame_id: Some(id.to_string()),
    }
}

fn test_edge(
    from: &str,
    to: &str,
    transform: Pose2,
    source: PoseEdgeSource,
    confidence: f32,
) -> PoseEdge {
    PoseEdge {
        from: from.to_string(),
        to: to.to_string(),
        transform,
        covariance: [0.05, 0.05, 0.08],
        confidence,
        source,
        active: true,
        rejection_reason: None,
    }
}

fn live_loop_test_config() -> MapConfig {
    MapConfig {
        resolution_m: 0.1,
        scan_match_enabled: false,
        pose_graph_min_node_distance_m: 0.01,
        pose_graph_max_ticks_between_nodes: 1,
        pose_graph_optimize_iterations: 8,
        pose_graph_min_loop_confidence: 0.85,
        pose_graph_loop_target_max_distance_m: 0.75,
        pose_graph_loop_min_geometric_overlap: 0.40,
        ..MapConfig::default()
    }
}

fn seeded_live_loop_map(config: MapConfig) -> LocalMap {
    let mut map = LocalMap::new(config);
    map.integrate_observation(observation_from_parts(
        pose(0.0, 0.0, 0.0),
        0.75,
        &range_sense(&[1.0]),
        serde_json::json!({"frame_id":"seed"}),
        100,
        config,
    ));
    map.integrate_observation(observation_from_parts(
        pose(1.0, 0.0, 0.0),
        0.75,
        &range_sense(&[1.0]),
        serde_json::json!({"frame_id":"away"}),
        200,
        config,
    ));
    map
}

fn live_loop_candidate(
    kind: &str,
    confidence: f32,
    target_frame_id: &str,
    source_frame_id: &str,
) -> LoopClosureCandidateInput {
    LoopClosureCandidateInput {
        target_pose: pose(0.0, 0.0, 0.0),
        confidence,
        similarity: confidence,
        kind: kind.to_string(),
        target_frame_id: Some(target_frame_id.to_string()),
        source_frame_id: Some(source_frame_id.to_string()),
        source_experience_id: Some("experience-seed".to_string()),
        source_instant_frame_id: Some(target_frame_id.to_string()),
        source_vector_refs: vec!["entity:charger".to_string()],
        source_vector_id: Some("constellation-seed".to_string()),
        query_vector_id: Some("constellation-return".to_string()),
        query_experience_id: Some("experience-return".to_string()),
    }
}
