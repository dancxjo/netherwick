use pete_core::Pose2;
use pete_map::{PoseGraphBuilder, PoseGraphConfig};

#[test]
fn rotated_rectangular_loop_edges_are_expressed_in_each_node_frame() {
    let mut builder = PoseGraphBuilder::new(PoseGraphConfig {
        min_node_distance_m: 0.01,
        min_node_heading_delta_rad: 0.01,
        ..PoseGraphConfig::default()
    });
    let poses = [
        pose(0.0, 0.0, 0.0),
        pose(2.0, 0.0, std::f32::consts::FRAC_PI_2),
        pose(2.0, 1.0, std::f32::consts::PI),
        pose(0.0, 1.0, -std::f32::consts::FRAC_PI_2),
        pose(0.0, 0.0, 0.0),
    ];
    for (index, pose) in poses.into_iter().enumerate() {
        builder.observe(
            pose,
            index as u64 * 100,
            Some(format!("frame-{index}")),
            &[],
        );
    }

    let graph = builder.finish();
    assert_eq!(graph.nodes.len(), poses.len());
    assert_eq!(graph.edges.len(), poses.len() - 1);
    for (index, edge) in graph.edges.iter().enumerate() {
        let expected_length = if index % 2 == 0 { 2.0 } else { 1.0 };
        assert!((edge.transform.x_m - expected_length).abs() < 0.001);
        assert!(edge.transform.y_m.abs() < 0.001);
        assert!(angle_delta(edge.transform.heading_rad, std::f32::consts::FRAC_PI_2) < 0.001);

        let reconstructed = compose(poses[index], edge.transform);
        assert_pose_near(reconstructed, poses[index + 1]);
    }
}

fn pose(x_m: f32, y_m: f32, heading_rad: f32) -> Pose2 {
    Pose2 {
        x_m,
        y_m,
        heading_rad,
    }
}

fn compose(from: Pose2, delta: Pose2) -> Pose2 {
    let (sin, cos) = from.heading_rad.sin_cos();
    pose(
        from.x_m + cos * delta.x_m - sin * delta.y_m,
        from.y_m + sin * delta.x_m + cos * delta.y_m,
        from.heading_rad + delta.heading_rad,
    )
}

fn assert_pose_near(actual: Pose2, expected: Pose2) {
    assert!((actual.x_m - expected.x_m).abs() < 0.001);
    assert!((actual.y_m - expected.y_m).abs() < 0.001);
    assert!(angle_delta(actual.heading_rad, expected.heading_rad) < 0.001);
}

fn angle_delta(left: f32, right: f32) -> f32 {
    ((left - right + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI)
        .abs()
}
