pub fn anticipate_surfaces(
    current: &SurfaceExtractorOutput,
    current_pose: Pose2,
    action: &ActionPrimitive,
) -> Vec<SurfaceAnticipationFrame> {
    [500, 1_000, 2_000]
        .into_iter()
        .map(|offset_ms| anticipate_surface_frame(current, current_pose, action, offset_ms))
        .collect()
}

pub fn anticipate_surface_frame(
    current: &SurfaceExtractorOutput,
    current_pose: Pose2,
    action: &ActionPrimitive,
    offset_ms: u64,
) -> SurfaceAnticipationFrame {
    let expected_pose = predict_pose(current_pose, action, offset_ms);
    let projected_surfaces = current
        .stable_surfaces
        .iter()
        .map(|surface| project_surface(surface, expected_pose))
        .collect::<Vec<_>>();
    let projected_clusters = current
        .clusters
        .iter()
        .map(|cluster| project_cluster(cluster, expected_pose))
        .collect::<Vec<_>>();
    let projected_obstacle_grid = projected_obstacle_grid(current, current_pose, expected_pose);
    let navigation = anticipated_navigation(&projected_obstacle_grid, action);
    SurfaceAnticipationFrame {
        offset_ms,
        expected_pose,
        projected_surfaces,
        projected_clusters,
        projected_obstacle_grid,
        navigation,
    }
}

fn surface_camera_extrinsics_changed(
    config: &SurfaceExtractorConfig,
    height_m: f32,
    forward_offset_m: f32,
    pitch_rad: f32,
    roll_rad: f32,
    yaw_rad: f32,
) -> bool {
    const EPS: f32 = 1.0e-4;
    (config.depth_camera_height_m - height_m).abs() > EPS
        || (config.depth_camera_forward_offset_m - forward_offset_m).abs() > EPS
        || (config.depth_camera_pitch_down_rad - pitch_rad).abs() > EPS
        || (config.depth_camera_roll_rad - roll_rad).abs() > EPS
        || (config.depth_camera_yaw_rad - yaw_rad).abs() > EPS
}

fn surface_compact_depth_calibration_changed(
    config: &SurfaceExtractorConfig,
    beam_count: usize,
    fov_rad: f32,
    depth_scale: f32,
) -> bool {
    const EPS: f32 = 1.0e-4;
    config.compact_depth_beam_count != beam_count
        || (config.compact_depth_fov_rad - fov_rad).abs() > EPS
        || (config.depth_scale - depth_scale).abs() > EPS
}

fn depth_to_world_points(
    kinect: &KinectSense,
    robot_pose: Pose2,
    roll_rad: Option<f32>,
    pitch_rad: Option<f32>,
    config: &SurfaceExtractorConfig,
) -> Vec<Point3> {
    if config.compact_depth_beam_count > 0
        && kinect.depth_m.len() == config.compact_depth_beam_count
    {
        return compact_depth_to_world_points(&kinect.depth_m, robot_pose, config);
    }
    let Some(frame) = DepthProjection::from_kinect(kinect, config) else {
        return Vec::new();
    };
    let Some(geometry) = pete_now::DepthGeometry::from_kinect(kinect) else {
        return Vec::new();
    };
    let mut points = Vec::new();
    for (index, depth) in kinect.depth_m.iter().enumerate() {
        if !depth.is_finite() || *depth <= 0.0 {
            continue;
        }
        if *depth < frame.min_depth_m || *depth > frame.max_depth_m {
            continue;
        }
        let u = (index % frame.width) as f32;
        let v = (index / frame.width) as f32;
        let Some(camera_point) = geometry.depth_pixel_to_camera(u, v, *depth) else {
            continue;
        };
        let robot = if kinect.geometry_calibration.is_some() {
            let base = geometry.depth_point_to_base(camera_point);
            Vec3::new(base[0], base[1], base[2])
        } else {
            camera_to_robot(
                Vec3::new(camera_point[0], camera_point[1], camera_point[2]),
                config,
            )
        };
        let world = pete_now::DepthGeometry::base_point_to_world(
            [robot.x, robot.y, robot.z],
            robot_pose,
            roll_rad,
            pitch_rad,
        );
        points.push(Point3 {
            position: Vec3::new(world[0], world[1], world[2]),
        });
    }
    points
}

fn compact_depth_to_world_points(
    depth_m: &[f32],
    robot_pose: Pose2,
    config: &SurfaceExtractorConfig,
) -> Vec<Point3> {
    let beam_count = depth_m.len().max(1);
    let fov_rad = config
        .compact_depth_fov_rad
        .clamp(0.01, std::f32::consts::TAU);
    let start = if beam_count == 1 { 0.0 } else { -fov_rad * 0.5 };
    let step = if beam_count == 1 {
        0.0
    } else {
        fov_rad / (beam_count - 1) as f32
    };
    depth_m
        .iter()
        .enumerate()
        .filter_map(|(index, depth)| {
            if !depth.is_finite() || *depth <= 0.0 {
                return None;
            }
            let distance =
                (*depth * config.depth_scale).clamp(config.min_depth_m, config.max_depth_m);
            let angle = start + step * index as f32;
            let robot = rotate_robot_extrinsic(
                Vec3::new(angle.cos() * distance, angle.sin() * distance, 0.0),
                config.depth_camera_pitch_down_rad,
                config.depth_camera_roll_rad,
                config.depth_camera_yaw_rad,
            );
            let robot = Vec3::new(
                robot.x + config.depth_camera_forward_offset_m,
                robot.y,
                robot.z + config.depth_camera_height_m,
            );
            Some(Point3 {
                position: robot_to_world(robot, robot_pose),
            })
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct DepthProjection {
    width: usize,
    min_depth_m: f32,
    max_depth_m: f32,
}

impl DepthProjection {
    fn from_kinect(kinect: &KinectSense, config: &SurfaceExtractorConfig) -> Option<Self> {
        let width = usize::try_from(kinect.depth_width).ok()?;
        let height = usize::try_from(kinect.depth_height).ok()?;
        if width == 0 || height == 0 || width.checked_mul(height)? != kinect.depth_m.len() {
            return None;
        }
        Some(Self {
            width,
            min_depth_m: positive_or(kinect.min_depth_m, config.min_depth_m),
            max_depth_m: positive_or(kinect.max_depth_m, config.max_depth_m),
        })
    }
}

fn positive_or(value: f32, fallback: f32) -> f32 {
    if value > 0.0 {
        value
    } else {
        fallback
    }
}

fn robot_to_world(point: Vec3, pose: Pose2) -> Vec3 {
    let (sin, cos) = pose.heading_rad.sin_cos();
    Vec3::new(
        pose.x_m + point.x * cos - point.y * sin,
        pose.y_m + point.x * sin + point.y * cos,
        point.z,
    )
}

fn camera_to_robot(camera: Vec3, config: &SurfaceExtractorConfig) -> Vec3 {
    let base = Vec3::new(camera.z, -camera.x, -camera.y);
    let rotated = rotate_robot_extrinsic(
        base,
        config.depth_camera_pitch_down_rad,
        config.depth_camera_roll_rad,
        config.depth_camera_yaw_rad,
    );
    Vec3::new(
        rotated.x + config.depth_camera_forward_offset_m,
        rotated.y,
        rotated.z + config.depth_camera_height_m,
    )
}

fn rotate_robot_extrinsic(point: Vec3, pitch_rad: f32, roll_rad: f32, yaw_rad: f32) -> Vec3 {
    let (pitch_sin, pitch_cos) = pitch_rad.sin_cos();
    let mut x = point.x * pitch_cos + point.z * pitch_sin;
    let y = point.y;
    let mut z = -point.x * pitch_sin + point.z * pitch_cos;

    let (roll_sin, roll_cos) = roll_rad.sin_cos();
    let rolled_y = y * roll_cos - z * roll_sin;
    z = y * roll_sin + z * roll_cos;

    let (yaw_sin, yaw_cos) = yaw_rad.sin_cos();
    let yawed_x = x * yaw_cos - rolled_y * yaw_sin;
    let yawed_y = x * yaw_sin + rolled_y * yaw_cos;
    x = yawed_x;

    Vec3::new(x, yawed_y, z)
}

fn world_to_robot(point: Vec3, pose: Pose2) -> Vec3 {
    let dx = point.x - pose.x_m;
    let dy = point.y - pose.y_m;
    let (sin, cos) = pose.heading_rad.sin_cos();
    Vec3::new(dx * cos + dy * sin, -dx * sin + dy * cos, point.z)
}

fn world_vector_to_robot(vector: Vec3, pose: Pose2) -> Vec3 {
    let (sin, cos) = pose.heading_rad.sin_cos();
    Vec3::new(
        vector.x * cos + vector.y * sin,
        -vector.x * sin + vector.y * cos,
        vector.z,
    )
}

fn calibration_hint(
    floor: &SurfaceTrack,
    robot_pose: Pose2,
    config: &SurfaceExtractorConfig,
) -> SurfaceCalibrationHint {
    let normal_robot = world_vector_to_robot(floor.normal, robot_pose)
        .normalized()
        .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
    let floor_tilt_rad = normal_robot.z.abs().clamp(0.0, 1.0).acos();
    let floor_pitch_error_rad = normal_robot.x.atan2(normal_robot.z.max(f32::EPSILON));
    let floor_roll_error_rad = normal_robot.y.atan2(normal_robot.z.max(f32::EPSILON));
    SurfaceCalibrationHint {
        floor_confidence: floor.confidence,
        floor_height_error_m: floor.centroid.z,
        floor_tilt_rad,
        floor_pitch_error_rad,
        floor_roll_error_rad,
        suggested_depth_height_m: (config.depth_camera_height_m - floor.centroid.z).max(0.0),
        suggested_depth_pitch_down_rad: config.depth_camera_pitch_down_rad + floor_pitch_error_rad,
    }
}

fn voxel_downsample(points: &[Point3], voxel_size_m: f32) -> Vec<Point3> {
    let mut voxels: HashMap<(i32, i32, i32), (Vec3, usize)> = HashMap::new();
    for point in points {
        let key = voxel_key(point.position, voxel_size_m);
        let entry = voxels.entry(key).or_insert((Vec3::default(), 0));
        entry.0 += point.position;
        entry.1 += 1;
    }
    voxels
        .into_values()
        .map(|(sum, count)| Point3 {
            position: sum / count as f32,
        })
        .collect()
}

fn temporal_voxel_average(clouds: &VecDeque<Vec<Point3>>, voxel_size_m: f32) -> Vec<Point3> {
    let mut voxels: HashMap<(i32, i32, i32), (Vec3, usize)> = HashMap::new();
    for cloud in clouds {
        for point in cloud {
            let key = voxel_key(point.position, voxel_size_m);
            let entry = voxels.entry(key).or_insert((Vec3::default(), 0));
            entry.0 += point.position;
            entry.1 += 1;
        }
    }
    voxels
        .into_values()
        .map(|(sum, count)| Point3 {
            position: sum / count as f32,
        })
        .collect()
}

fn voxel_key(point: Vec3, voxel_size_m: f32) -> (i32, i32, i32) {
    let scale = voxel_size_m.max(0.01);
    (
        (point.x / scale).floor() as i32,
        (point.y / scale).floor() as i32,
        (point.z / scale).floor() as i32,
    )
}

fn remove_outliers(points: &[Point3], radius_m: f32, min_neighbors: usize) -> Vec<Point3> {
    if min_neighbors == 0 || points.len() <= min_neighbors {
        return points.to_vec();
    }
    let radius_sq = radius_m * radius_m;
    points
        .iter()
        .enumerate()
        .filter(|(index, point)| {
            points
                .iter()
                .enumerate()
                .filter(|(other_index, other)| {
                    index != other_index
                        && (point.position - other.position).dot(point.position - other.position)
                            <= radius_sq
                })
                .take(min_neighbors)
                .count()
                >= min_neighbors
        })
        .map(|(_, point)| *point)
        .collect()
}
