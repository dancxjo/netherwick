use pete_core::{Pose2, TimeMs};
use pete_map::{
    LocalMap, OccupancySubmap, OrientationEstimate, Point3D, PointCloudConfig, PointCloudFrame,
    PointCloudObservation, PointCloudPoint, PoseEstimate, PoseNode, VoxelPointCloud,
};

#[test]
fn pose_graph_rebuild_preserves_observations_beyond_previous_short_run_limit() {
    const OBSERVATION_COUNT: usize = 300;
    let mut cloud = VoxelPointCloud::new(PointCloudConfig {
        voxel_size_m: 0.05,
        max_voxels: 1_000,
        decay_after_ms: u64::MAX,
        decay_per_tick: 0.0,
        ..PointCloudConfig::default()
    });
    let mut map = LocalMap::default();

    for index in 0..OBSERVATION_COUNT {
        let t_ms = index as TimeMs * 100;
        let x_m = index as f32 * 0.1;
        cloud.integrate_observation(observation_at(x_m, t_ms));

        let node_id = format!("node-{index}");
        map.pose_graph.nodes.push(PoseNode {
            id: node_id.clone(),
            pose_estimate: PoseEstimate {
                pose: pose(x_m + 1.0, 0.0, 0.0),
                confidence: 1.0,
                covariance: [0.0; 3],
                source: "optimized".to_string(),
                t_ms,
            },
            t_ms,
            source_frame_id: Some(format!("depth-{index}")),
        });
        map.submaps.push(OccupancySubmap {
            id: format!("submap-{index}"),
            node_id,
            local_pose: Pose2::default(),
            range_beams: Vec::new(),
            t_ms,
            source_frame_id: Some(format!("depth-{index}")),
        });
    }
    map.pose_graph_optimization.max_node_update_m = 1.0;

    assert!(cloud.rebuild_from_pose_graph(&map));
    assert_eq!(cloud.summary().observations, OBSERVATION_COUNT as u64);
    assert!(contains_point_near(&cloud, 1.0, 0.0));
    assert!(contains_point_near(
        &cloud,
        (OBSERVATION_COUNT - 1) as f32 * 0.1 + 1.0,
        0.0,
    ));
}

fn observation_at(x_m: f32, t_ms: TimeMs) -> PointCloudObservation {
    PointCloudObservation {
        frame: PointCloudFrame::RobotBase,
        pose: PoseEstimate {
            pose: pose(x_m, 0.0, 0.0),
            confidence: 1.0,
            covariance: [0.0; 3],
            source: "raw_odometry".to_string(),
            t_ms,
        },
        orientation: OrientationEstimate::default(),
        points: vec![PointCloudPoint {
            position: Point3D::default(),
            color_rgb: None,
            confidence: 1.0,
            depth_index: None,
            depth_uv: None,
            depth_image_size: None,
            source_frame_id: Some(format!("depth-{t_ms}")),
        }],
        source: "kinect_depth".to_string(),
        t_ms,
        metadata: serde_json::Value::Null,
    }
}

fn pose(x_m: f32, y_m: f32, heading_rad: f32) -> Pose2 {
    Pose2 {
        x_m,
        y_m,
        heading_rad,
    }
}

fn contains_point_near(cloud: &VoxelPointCloud, x_m: f32, y_m: f32) -> bool {
    cloud.voxels.values().any(|voxel| {
        (voxel.position.x_m - x_m).abs() < 0.001 && (voxel.position.y_m - y_m).abs() < 0.001
    })
}
