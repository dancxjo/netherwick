use super::*;

#[test]
fn extracts_and_tracks_a_floor_plane() {
    let mut extractor = SurfaceExtractor::new(SurfaceExtractorConfig {
        min_plane_points: 12,
        outlier_min_neighbors: 1,
        ..SurfaceExtractorConfig::default()
    });
    let kinect = synthetic_floor_depth(24, 18, 1.2);

    let first = extractor.process(&kinect, Pose2::default(), 1_000);
    let second = extractor.process(&kinect, Pose2::default(), 1_100);

    assert!(first.diagnostics.raw_points > 0);
    assert!(second.floor.is_some());
    assert_eq!(second.floor.as_ref().unwrap().id, "floor");
    assert!(second.floor.as_ref().unwrap().confidence > first.floor.unwrap().confidence);
}

#[test]
fn clusters_leftover_points_after_plane_removal() {
    let config = SurfaceExtractorConfig {
        min_plane_points: 8,
        min_cluster_points: 3,
        cluster_distance_m: 0.35,
        ..SurfaceExtractorConfig::default()
    };
    let mut points = Vec::new();
    for x in 0..5 {
        for y in 0..5 {
            points.push(Point3 {
                position: Vec3::new(x as f32 * 0.1, y as f32 * 0.1, 0.0),
            });
        }
    }
    for z in 0..4 {
        points.push(Point3 {
            position: Vec3::new(1.0, 1.0, 0.3 + z as f32 * 0.05),
        });
    }
    let (_planes, leftovers) = extract_planes(&points, &config);
    let clusters = euclidean_clusters(&leftovers, config.cluster_distance_m, 3);
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].point_count, 4);
}

#[test]
fn camera_extrinsics_apply_height_and_downward_pitch() {
    let config = SurfaceExtractorConfig {
        depth_camera_height_m: 0.25,
        depth_camera_forward_offset_m: 0.1,
        depth_camera_pitch_down_rad: 10.0_f32.to_radians(),
        ..SurfaceExtractorConfig::default()
    };

    let center_ray = camera_to_robot(Vec3::new(0.0, 0.0, 1.0), &config);

    assert!(center_ray.x > 1.0);
    assert!(center_ray.z < config.depth_camera_height_m);
}

#[test]
fn compact_depth_obstacle_grid_center_beam_points_forward() {
    let mut extractor = SurfaceExtractor::new(SurfaceExtractorConfig {
        compact_depth_beam_count: 3,
        compact_depth_fov_rad: std::f32::consts::FRAC_PI_2,
        depth_camera_height_m: 0.18,
        min_depth_m: 0.1,
        max_depth_m: 3.0,
        outlier_min_neighbors: 0,
        ..SurfaceExtractorConfig::default()
    });
    let kinect = KinectSense {
        depth_m: vec![1.0, 1.0, 1.0],
        ..KinectSense::default()
    };

    let output = extractor.process(&kinect, Pose2::default(), 1);
    let center_cell = output
        .obstacle_grid
        .cells
        .iter()
        .find(|cell| cell.state == OccupancyState::Occupied && cell.y == 0)
        .expect("center compact beam should create a forward occupied cell");

    assert!(center_cell.x > 0);
}

#[test]
fn calibration_changes_clear_surface_tracking_state() {
    let mut extractor = SurfaceExtractor::new(SurfaceExtractorConfig {
        compact_depth_beam_count: 3,
        compact_depth_fov_rad: std::f32::consts::FRAC_PI_2,
        depth_scale: 1.0,
        depth_camera_height_m: 0.18,
        depth_camera_forward_offset_m: 0.02,
        depth_camera_yaw_rad: 0.0,
        ..SurfaceExtractorConfig::default()
    });
    extractor.temporal_clouds.push_back(vec![Point3 {
        position: Vec3::new(1.0, 0.0, 0.0),
    }]);
    extractor.next_surface_id = 42;
    extractor.next_cluster_id = 24;

    extractor.set_depth_camera_extrinsics(0.18, 0.02, 0.0, 0.0, 0.0);
    assert_eq!(extractor.temporal_clouds.len(), 1);
    assert_eq!(extractor.next_surface_id, 42);
    assert_eq!(extractor.next_cluster_id, 24);

    extractor.set_depth_camera_extrinsics(0.18, 0.02, 0.0, 0.0, 0.25);
    assert!(extractor.temporal_clouds.is_empty());
    assert_eq!(extractor.next_surface_id, 1);
    assert_eq!(extractor.next_cluster_id, 1);

    extractor.temporal_clouds.push_back(vec![Point3 {
        position: Vec3::new(1.0, 0.0, 0.0),
    }]);
    extractor.next_surface_id = 42;
    extractor.next_cluster_id = 24;

    extractor.set_compact_depth_calibration(5, std::f32::consts::FRAC_PI_2, 1.0);
    assert!(extractor.temporal_clouds.is_empty());
    assert_eq!(extractor.next_surface_id, 1);
    assert_eq!(extractor.next_cluster_id, 1);
}

#[test]
fn floor_calibration_hint_reports_height_and_tilt() {
    let config = SurfaceExtractorConfig {
        depth_camera_height_m: 0.3,
        depth_camera_pitch_down_rad: 0.0,
        ..SurfaceExtractorConfig::default()
    };
    let floor = SurfaceTrack {
        id: "floor".to_string(),
        primitive_kind: SurfacePrimitiveKind::Plane,
        kind: SurfaceKind::Floor,
        normal: Vec3::new(0.1, 0.0, 0.995).normalized().unwrap(),
        centroid: Vec3::new(0.0, 0.0, 0.05),
        distance_from_origin_m: 0.0,
        bounds_2d: Bounds2::default(),
        extent_m: Vec3::new(2.0, 2.0, 0.02),
        confidence: 0.8,
        supporting_point_count: 64,
        first_seen_ms: 0,
        last_seen_ms: 0,
        seen_count: 1,
        missing_count: 0,
        labels: surface_labels(SurfaceKind::Floor),
    };

    let hint = calibration_hint(&floor, Pose2::default(), &config);

    assert!(hint.floor_tilt_rad > 0.0);
    assert_eq!(hint.floor_height_error_m, 0.05);
    assert!(hint.suggested_depth_height_m < config.depth_camera_height_m);
}

#[test]
fn cluster_tracks_keep_ids_and_detect_motion() {
    let mut extractor = SurfaceExtractor::new(SurfaceExtractorConfig {
        min_cluster_points: 3,
        cluster_track_match_threshold_m: 1.0,
        cluster_moving_speed_m_s: 0.05,
        ..SurfaceExtractorConfig::default()
    });
    let surfaces = vec![SurfaceTrack {
        id: "floor".to_string(),
        primitive_kind: SurfacePrimitiveKind::Plane,
        kind: SurfaceKind::Floor,
        normal: Vec3::new(0.0, 0.0, 1.0),
        centroid: Vec3::default(),
        distance_from_origin_m: 0.0,
        bounds_2d: Bounds2 {
            min_u: -2.0,
            max_u: 2.0,
            min_v: -2.0,
            max_v: 2.0,
        },
        extent_m: Vec3::new(4.0, 4.0, 0.0),
        confidence: 1.0,
        supporting_point_count: 100,
        first_seen_ms: 0,
        last_seen_ms: 0,
        seen_count: 1,
        missing_count: 0,
        labels: surface_labels(SurfaceKind::Floor),
    }];

    let first = extractor.update_cluster_tracks(
        vec![ClusterObservation {
            id: String::new(),
            centroid: Vec3::new(0.0, 0.0, 0.6),
            size_m: Vec3::new(0.3, 0.3, 0.7),
            point_count: 8,
            confidence: 0.4,
            moving: false,
            velocity_m_s: Vec3::default(),
            last_seen_ms: 0,
            seen_count: 0,
            above_surface_id: None,
            semantic_hint: None,
        }],
        &surfaces,
        1_000,
    );
    let second = extractor.update_cluster_tracks(
        vec![ClusterObservation {
            id: String::new(),
            centroid: Vec3::new(0.2, 0.0, 0.6),
            size_m: Vec3::new(0.3, 0.3, 0.7),
            point_count: 8,
            confidence: 0.4,
            moving: false,
            velocity_m_s: Vec3::default(),
            last_seen_ms: 0,
            seen_count: 0,
            above_surface_id: None,
            semantic_hint: None,
        }],
        &surfaces,
        2_000,
    );

    assert_eq!(first[0].id, second[0].id);
    assert!(second[0].moving);
    assert_eq!(second[0].above_surface_id.as_deref(), Some("floor"));
}

#[test]
fn wall_ahead_forward_anticipation_increases_forward_risk() {
    let output = output_with_wall("wall_1", Vec3::new(0.7, 0.0, 0.6));
    let action = ActionPrimitive::Go {
        intensity: 0.2,
        duration_ms: 2_000,
    };

    let frames = anticipate_surfaces(&output, Pose2::default(), &action);

    assert!(frames[0].navigation.front_clear_m.unwrap() < 0.7);
    assert!(
        frames[2].navigation.front_clear_m.unwrap() < frames[0].navigation.front_clear_m.unwrap()
    );
    assert!(frames[2].navigation.collision_risk > frames[0].navigation.collision_risk);
}

#[test]
fn wall_left_turn_left_anticipation_becomes_risky() {
    let output = output_with_wall("wall_1", Vec3::new(0.0, 0.45, 0.6));
    let action = ActionPrimitive::Turn {
        direction: pete_actions::TurnDir::Left,
        intensity: 0.8,
        duration_ms: 2_000,
    };

    let frames = anticipate_surfaces(&output, Pose2::default(), &action);

    assert!(frames[0].navigation.left_clear_m.unwrap() < 0.6);
    assert!(frames[0].navigation.collision_risk > 0.0);
    assert!(frames[2]
        .projected_surfaces
        .iter()
        .any(|surface| surface.centroid.x > 0.0 && surface.centroid.y.abs() < 0.35));
}

#[test]
fn open_floor_forward_anticipation_stays_low_risk() {
    let output = SurfaceExtractorOutput {
        stable_surfaces: vec![floor_track()],
        floor: Some(floor_track()),
        obstacle_grid: OccupancyGrid {
            resolution_m: 0.1,
            half_extent_m: 3.0,
            cells: Vec::new(),
        },
        ..SurfaceExtractorOutput::default()
    };
    let action = ActionPrimitive::Go {
        intensity: 0.2,
        duration_ms: 2_000,
    };

    let frames = anticipate_surfaces(&output, Pose2::default(), &action);

    assert!(frames
        .iter()
        .all(|frame| frame.navigation.front_clear_m.is_none()
            && frame.navigation.collision_risk <= 0.01));
}

#[test]
fn visible_wall_centroid_shift_keeps_existing_wall_id() {
    let mut extractor = SurfaceExtractor::new(SurfaceExtractorConfig {
        track_centroid_threshold_m: 0.35,
        ..SurfaceExtractorConfig::default()
    });
    let first = wall_observation(Vec3::new(0.8, -0.4, 0.6));
    let second = PlaneObservation {
        centroid: Vec3::new(0.81, 0.55, 0.62),
        bounds_2d: Bounds2 {
            min_u: -0.2,
            max_u: 0.2,
            min_v: -0.45,
            max_v: 0.35,
        },
        ..first
    };

    let tracks = extractor.update_tracks(&[first], 1_000);
    let id = tracks[0].id.clone();
    let tracks = extractor.update_tracks(&[second], 1_100);

    assert_eq!(tracks.len(), 1);
    assert_eq!(tracks[0].id, id);
    assert_eq!(extractor.next_surface_id, 2);
}

#[test]
fn repeated_wall_observations_keep_id_and_raise_confidence() {
    let mut extractor = SurfaceExtractor::new(SurfaceExtractorConfig {
        track_seen_gain: 0.2,
        ..SurfaceExtractorConfig::default()
    });
    let observation = wall_observation(Vec3::new(1.2, 0.0, 0.7));

    let first = extractor.update_tracks(&[observation], 1_000);
    let id = first[0].id.clone();
    let first_confidence = first[0].confidence;
    let second = extractor.update_tracks(&[observation], 1_100);
    let third = extractor.update_tracks(&[observation], 1_200);

    assert_eq!(second[0].id, id);
    assert_eq!(third[0].id, id);
    assert!(third[0].confidence > first_confidence);
    assert_eq!(third[0].primitive_kind, SurfacePrimitiveKind::Plane);
    assert_eq!(third[0].supporting_point_count, observation.point_count);
    assert!(third[0]
        .labels
        .iter()
        .any(|label| label == "wall_candidate"));
}

#[test]
fn confidence_decays_when_plane_disappears() {
    let mut extractor = SurfaceExtractor::new(SurfaceExtractorConfig {
        track_missing_decay: 0.12,
        ..SurfaceExtractorConfig::default()
    });
    let observation = wall_observation(Vec3::new(1.2, 0.0, 0.7));

    let first = extractor.update_tracks(&[observation], 1_000);
    let confidence = first[0].confidence;
    let missing = extractor.update_tracks(&[], 1_100);

    assert_eq!(missing[0].id, first[0].id);
    assert!(missing[0].confidence < confidence);
    assert_eq!(missing[0].missing_count, 1);
}

#[test]
fn tiny_noisy_planar_groups_are_rejected() {
    let config = SurfaceExtractorConfig {
        min_plane_points: 6,
        min_plane_major_extent_m: 0.35,
        min_plane_minor_extent_m: 0.12,
        min_plane_area_m: 0.08,
        max_plane_rms_error_m: 0.03,
        ..SurfaceExtractorConfig::default()
    };
    let mut points = Vec::new();
    for x in 0..3 {
        for z in 0..3 {
            points.push(Point3 {
                position: Vec3::new(
                    1.0 + x as f32 * 0.025,
                    (x + z) as f32 * 0.003,
                    0.4 + z as f32 * 0.025,
                ),
            });
        }
    }

    let (planes, leftovers) = extract_planes(&points, &config);

    assert!(planes.is_empty());
    assert_eq!(leftovers.len(), points.len());
}

#[test]
fn planar_fragments_supported_by_surfaces_do_not_become_clusters() {
    let mut extractor = SurfaceExtractor::new(SurfaceExtractorConfig {
        min_cluster_points: 3,
        cluster_distance_m: 0.2,
        ..SurfaceExtractorConfig::default()
    });
    let wall = SurfaceTrack {
        id: "wall_1".to_string(),
        primitive_kind: SurfacePrimitiveKind::Plane,
        kind: SurfaceKind::VerticalPlane,
        normal: Vec3::new(1.0, 0.0, 0.0),
        centroid: Vec3::new(1.0, 0.0, 0.6),
        distance_from_origin_m: -1.0,
        bounds_2d: Bounds2 {
            min_u: -0.5,
            max_u: 0.5,
            min_v: -0.5,
            max_v: 0.5,
        },
        extent_m: Vec3::new(0.02, 1.0, 1.0),
        confidence: 0.8,
        supporting_point_count: 80,
        first_seen_ms: 1_000,
        last_seen_ms: 1_000,
        seen_count: 1,
        missing_count: 0,
        labels: surface_labels(SurfaceKind::VerticalPlane),
    };
    let fragment = vec![
        Point3 {
            position: Vec3::new(1.01, -0.05, 0.55),
        },
        Point3 {
            position: Vec3::new(1.02, 0.0, 0.6),
        },
        Point3 {
            position: Vec3::new(1.01, 0.05, 0.65),
        },
    ];

    let clusters = extractor.cluster_leftovers(&fragment, &[wall], 1_000);

    assert!(clusters.is_empty());
}

fn output_with_wall(id: &str, centroid: Vec3) -> SurfaceExtractorOutput {
    let normal = if centroid.x.abs() >= centroid.y.abs() {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let wall = SurfaceTrack {
        id: id.to_string(),
        primitive_kind: SurfacePrimitiveKind::Plane,
        kind: SurfaceKind::VerticalPlane,
        normal,
        centroid,
        distance_from_origin_m: -normal.dot(centroid),
        bounds_2d: Bounds2 {
            min_u: -0.45,
            max_u: 0.45,
            min_v: -0.55,
            max_v: 0.55,
        },
        extent_m: Vec3::new(0.02, 0.9, 1.1),
        confidence: 0.85,
        supporting_point_count: 80,
        first_seen_ms: 1_000,
        last_seen_ms: 1_000,
        seen_count: 3,
        missing_count: 0,
        labels: surface_labels(SurfaceKind::VerticalPlane),
    };
    SurfaceExtractorOutput {
        stable_surfaces: vec![floor_track(), wall],
        floor: Some(floor_track()),
        obstacle_grid: OccupancyGrid {
            resolution_m: 0.1,
            half_extent_m: 3.0,
            cells: Vec::new(),
        },
        ..SurfaceExtractorOutput::default()
    }
}

fn wall_observation(centroid: Vec3) -> PlaneObservation {
    PlaneObservation {
        normal: Vec3::new(1.0, 0.0, 0.0),
        centroid,
        distance_from_origin_m: -centroid.x,
        bounds_2d: Bounds2 {
            min_u: -0.4,
            max_u: 0.4,
            min_v: -0.5,
            max_v: 0.5,
        },
        extent_m: Vec3::new(0.02, 0.8, 1.0),
        point_count: 64,
        confidence: 0.7,
        rms_error_m: 0.01,
    }
}

fn floor_track() -> SurfaceTrack {
    SurfaceTrack {
        id: "floor".to_string(),
        primitive_kind: SurfacePrimitiveKind::Plane,
        kind: SurfaceKind::Floor,
        normal: Vec3::new(0.0, 0.0, 1.0),
        centroid: Vec3::default(),
        distance_from_origin_m: 0.0,
        bounds_2d: Bounds2 {
            min_u: -2.0,
            max_u: 2.0,
            min_v: -2.0,
            max_v: 2.0,
        },
        extent_m: Vec3::new(4.0, 4.0, 0.0),
        confidence: 1.0,
        supporting_point_count: 100,
        first_seen_ms: 0,
        last_seen_ms: 0,
        seen_count: 1,
        missing_count: 0,
        labels: surface_labels(SurfaceKind::Floor),
    }
}

fn synthetic_floor_depth(width: u32, height: u32, camera_height_m: f32) -> KinectSense {
    let fx = 80.0;
    let fy = 20.0;
    let cx = (width as f32 - 1.0) * 0.5;
    let cy = (height as f32 - 1.0) * 0.5;
    let mut depth_m = Vec::new();
    for v in 0..height {
        for _u in 0..width {
            let ray_y = (v as f32 - cy) / fy;
            if ray_y <= 0.05 {
                depth_m.push(0.0);
            } else {
                depth_m.push(camera_height_m / ray_y);
            }
        }
    }
    KinectSense {
        depth_m,
        depth_width: width,
        depth_height: height,
        depth_fx: fx,
        depth_fy: fy,
        depth_cx: cx,
        depth_cy: cy,
        min_depth_m: 0.1,
        max_depth_m: 8.0,
        ..KinectSense::default()
    }
}
