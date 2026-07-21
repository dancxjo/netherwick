use pete_core::Pose2;
use pete_map::{
    PoseEdge, PoseEdgeSource, PoseEstimate, PoseGraph, PoseGraphBuilder, PoseGraphConfig,
    PoseGraphOptimizationConfig, PoseNode,
};

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

#[test]
fn rotated_rectangular_loop_optimization_reduces_closure_drift() {
    let quarter_turn = std::f32::consts::FRAC_PI_2;
    let mut graph = PoseGraph {
        nodes: vec![
            node("pose-0", pose(0.0, 0.0, 0.0), 0),
            node("pose-1", pose(2.1, 0.1, quarter_turn + 0.03), 100),
            node("pose-2", pose(2.15, 1.15, std::f32::consts::PI + 0.05), 200),
            node("pose-3", pose(0.1, 1.2, -quarter_turn + 0.02), 300),
            node("pose-4", pose(0.25, 0.15, 0.04), 400),
        ],
        edges: vec![
            edge("pose-0", "pose-1", pose(2.0, 0.0, quarter_turn)),
            edge("pose-1", "pose-2", pose(1.0, 0.0, quarter_turn)),
            edge("pose-2", "pose-3", pose(2.0, 0.0, quarter_turn)),
            edge("pose-3", "pose-4", pose(1.0, 0.0, quarter_turn)),
            PoseEdge {
                source: PoseEdgeSource::LoopClosureCandidate {
                    kind: "same_place_geometry".to_string(),
                    target_frame_id: Some("pose-0".to_string()),
                    source_frame_id: Some("pose-4".to_string()),
                    source_experience_id: None,
                    source_instant_frame_id: None,
                    source_vector_refs: Vec::new(),
                    source_vector_id: None,
                    query_vector_id: None,
                    query_experience_id: None,
                    registration: None,
                },
                confidence: 0.98,
                ..edge("pose-4", "pose-0", Pose2::default())
            },
        ],
    };
    let initial_closure_drift = graph.nodes[4]
        .pose_estimate
        .pose
        .x_m
        .hypot(graph.nodes[4].pose_estimate.pose.y_m);

    let summary = graph.optimize_anchored(PoseGraphOptimizationConfig {
        iterations: 60,
        step_size: 0.5,
        ..PoseGraphOptimizationConfig::default()
    });
    let final_closure_drift = graph.nodes[4]
        .pose_estimate
        .pose
        .x_m
        .hypot(graph.nodes[4].pose_estimate.pose.y_m);

    assert!(summary.final_mean_error < summary.initial_mean_error);
    assert!(final_closure_drift < initial_closure_drift);
    assert_pose_near(graph.nodes[0].pose_estimate.pose, pose(0.0, 0.0, 0.0));
}

fn pose(x_m: f32, y_m: f32, heading_rad: f32) -> Pose2 {
    Pose2 {
        x_m,
        y_m,
        heading_rad,
    }
}

fn node(id: &str, pose: Pose2, t_ms: u64) -> PoseNode {
    PoseNode {
        id: id.to_string(),
        pose_estimate: PoseEstimate {
            pose,
            confidence: 0.8,
            covariance: [0.05, 0.05, 0.1],
            source: "test".to_string(),
            t_ms,
        },
        t_ms,
        source_frame_id: Some(id.to_string()),
    }
}

fn edge(from: &str, to: &str, transform: Pose2) -> PoseEdge {
    PoseEdge {
        from: from.to_string(),
        to: to.to_string(),
        transform,
        covariance: [0.08, 0.08, 0.15],
        confidence: 0.8,
        source: PoseEdgeSource::Odometry,
        active: true,
        rejection_reason: None,
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
