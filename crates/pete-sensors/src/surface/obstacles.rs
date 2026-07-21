fn project_obstacles(
    points: &[Point3],
    floor: Option<&SurfaceTrack>,
    robot_pose: Pose2,
    config: &SurfaceExtractorConfig,
) -> OccupancyGrid {
    let floor_z = floor.map_or(0.0, |floor| floor.centroid.z);
    let mut cells = HashMap::<(i32, i32), OccupancyState>::new();
    for point in points {
        let local = world_to_robot(point.position, robot_pose);
        if local.x.abs() > config.occupancy_half_extent_m
            || local.y.abs() > config.occupancy_half_extent_m
        {
            continue;
        }
        let point_key = occupancy_key(local, config.occupancy_resolution_m);
        let height = point.position.z - floor_z;
        if height < config.obstacle_min_height_m {
            cells.entry(point_key).or_insert(OccupancyState::Free);
            continue;
        }
        if height > config.obstacle_max_height_m {
            continue;
        }
        mark_free_ray(&mut cells, local, config.occupancy_resolution_m);
        cells.insert(point_key, OccupancyState::Occupied);
    }
    OccupancyGrid {
        resolution_m: config.occupancy_resolution_m,
        half_extent_m: config.occupancy_half_extent_m,
        cells: cells
            .into_iter()
            .map(|((x, y), state)| OccupancyCell { x, y, state })
            .collect(),
    }
}

fn occupancy_key(point: Vec3, resolution_m: f32) -> (i32, i32) {
    (
        (point.x / resolution_m).floor() as i32,
        (point.y / resolution_m).floor() as i32,
    )
}

fn mark_free_ray(cells: &mut HashMap<(i32, i32), OccupancyState>, point: Vec3, resolution_m: f32) {
    let distance = (point.x * point.x + point.y * point.y).sqrt();
    let steps = (distance / resolution_m).floor().max(0.0) as usize;
    if steps == 0 {
        return;
    }
    for step in 0..steps {
        let t = step as f32 / steps as f32;
        let key = occupancy_key(Vec3::new(point.x * t, point.y * t, 0.0), resolution_m);
        cells.entry(key).or_insert(OccupancyState::Free);
    }
}

fn euclidean_clusters(
    points: &[Point3],
    distance_m: f32,
    min_points: usize,
) -> Vec<ClusterObservation> {
    let mut visited = vec![false; points.len()];
    let mut clusters = Vec::new();
    let distance_sq = distance_m * distance_m;
    for seed in 0..points.len() {
        if visited[seed] {
            continue;
        }
        let mut stack = vec![seed];
        let mut members = Vec::new();
        visited[seed] = true;
        while let Some(index) = stack.pop() {
            members.push(index);
            for other in 0..points.len() {
                if visited[other] {
                    continue;
                }
                let delta = points[index].position - points[other].position;
                if delta.dot(delta) <= distance_sq {
                    visited[other] = true;
                    stack.push(other);
                }
            }
        }
        if members.len() >= min_points {
            clusters.push(cluster_from_members(points, &members));
        }
    }
    clusters
}

fn cluster_from_members(points: &[Point3], members: &[usize]) -> ClusterObservation {
    let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    let mut centroid = Vec3::default();
    for index in members {
        let position = points[*index].position;
        centroid += position;
        min.x = min.x.min(position.x);
        min.y = min.y.min(position.y);
        min.z = min.z.min(position.z);
        max.x = max.x.max(position.x);
        max.y = max.y.max(position.y);
        max.z = max.z.max(position.z);
    }
    ClusterObservation {
        id: String::new(),
        centroid: centroid / members.len() as f32,
        size_m: max - min,
        point_count: members.len(),
        confidence: (members.len() as f32 / 24.0).clamp(0.2, 0.8),
        moving: false,
        velocity_m_s: Vec3::default(),
        last_seen_ms: 0,
        seen_count: 0,
        above_surface_id: None,
        semantic_hint: None,
    }
}

fn surface_below_cluster(
    cluster: &ClusterObservation,
    surfaces: &[SurfaceTrack],
) -> Option<String> {
    surfaces
        .iter()
        .filter(|surface| surface.normal.z.abs() > 0.75)
        .filter(|surface| surface.centroid.z <= cluster.centroid.z + 0.05)
        .filter(|surface| point_inside_surface_bounds(cluster.centroid, surface, 0.25))
        .min_by(|left, right| {
            let left_height = (cluster.centroid.z - left.centroid.z).abs();
            let right_height = (cluster.centroid.z - right.centroid.z).abs();
            left_height.total_cmp(&right_height)
        })
        .map(|surface| surface.id.clone())
        .or_else(|| {
            surfaces
                .iter()
                .find(|surface| surface.kind == SurfaceKind::Floor)
                .map(|surface| surface.id.clone())
        })
}

fn point_inside_surface_bounds(point: Vec3, surface: &SurfaceTrack, margin_m: f32) -> bool {
    let normal = surface.normal;
    let basis_a = if normal.z.abs() < 0.9 {
        normal.cross(Vec3::new(0.0, 0.0, 1.0))
    } else {
        normal.cross(Vec3::new(1.0, 0.0, 0.0))
    }
    .normalized()
    .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
    let basis_b = normal
        .cross(basis_a)
        .normalized()
        .unwrap_or(Vec3::new(0.0, 1.0, 0.0));
    let relative = point - surface.centroid;
    let u = relative.dot(basis_a);
    let v = relative.dot(basis_b);
    u >= surface.bounds_2d.min_u - margin_m
        && u <= surface.bounds_2d.max_u + margin_m
        && v >= surface.bounds_2d.min_v - margin_m
        && v <= surface.bounds_2d.max_v + margin_m
}

fn semantic_hint_for_cluster(cluster: &ClusterObservation) -> Option<String> {
    let height = cluster.size_m.z.max(0.0);
    let width = cluster.size_m.x.abs().max(cluster.size_m.y.abs());
    if cluster.moving && height > 0.8 && width > 0.25 {
        Some("moving_human_sized".to_string())
    } else if cluster.moving {
        Some("moving_obstacle".to_string())
    } else if height < 0.18 && width < 0.35 {
        Some("small_clutter".to_string())
    } else if height > 0.5 && width < 0.35 {
        Some("thin_vertical_obstacle".to_string())
    } else if cluster.above_surface_id.is_some() {
        Some("object_on_surface".to_string())
    } else {
        None
    }
}

fn surface_labels(kind: SurfaceKind) -> Vec<String> {
    match kind {
        SurfaceKind::Floor => vec!["floor_candidate".to_string()],
        SurfaceKind::HorizontalPlane => vec!["table_candidate".to_string()],
        SurfaceKind::VerticalPlane => vec!["wall_candidate".to_string()],
        SurfaceKind::UnknownPlane => vec!["unknown_surface".to_string()],
    }
}

fn cluster_is_planar_room_geometry(
    cluster: &ClusterObservation,
    surfaces: &[SurfaceTrack],
) -> bool {
    surfaces
        .iter()
        .filter(|surface| surface.confidence >= 0.2)
        .any(|surface| {
            let normal_delta = (cluster.centroid - surface.centroid)
                .dot(surface.normal)
                .abs();
            normal_delta <= 0.08 && point_inside_surface_bounds(cluster.centroid, surface, 0.2)
        })
}

fn predict_pose(pose: Pose2, action: &ActionPrimitive, offset_ms: u64) -> Pose2 {
    let motor = action_to_motor_command(Some(action));
    let active_ms = match action {
        ActionPrimitive::Go { duration_ms, .. }
        | ActionPrimitive::Turn { duration_ms, .. }
        | ActionPrimitive::Explore { duration_ms, .. } => offset_ms.min(*duration_ms),
        _ => offset_ms,
    };
    let dt_s = active_ms as f32 / 1_000.0;
    let forward = motor.forward;
    let turn = motor.turn;
    if dt_s <= 0.0 || (!forward.is_finite()) || (!turn.is_finite()) {
        return pose;
    }
    if turn.abs() < 1.0e-4 {
        Pose2 {
            x_m: pose.x_m + forward * dt_s * pose.heading_rad.cos(),
            y_m: pose.y_m + forward * dt_s * pose.heading_rad.sin(),
            heading_rad: pose.heading_rad,
        }
    } else {
        let heading = pose.heading_rad + turn * dt_s;
        let radius = forward / turn;
        Pose2 {
            x_m: pose.x_m + radius * (heading.sin() - pose.heading_rad.sin()),
            y_m: pose.y_m - radius * (heading.cos() - pose.heading_rad.cos()),
            heading_rad: wrap_angle(heading),
        }
    }
}

fn wrap_angle(angle: f32) -> f32 {
    let two_pi = std::f32::consts::TAU;
    (angle + std::f32::consts::PI).rem_euclid(two_pi) - std::f32::consts::PI
}

fn project_surface(surface: &SurfaceTrack, expected_pose: Pose2) -> ProjectedSurface {
    ProjectedSurface {
        id: surface.id.clone(),
        kind: surface.kind,
        normal: world_vector_to_robot(surface.normal, expected_pose)
            .normalized()
            .unwrap_or(surface.normal),
        centroid: world_to_robot(surface.centroid, expected_pose),
        bounds_2d: surface.bounds_2d,
        confidence: surface.confidence,
        observed_bounds_2d: surface.bounds_2d,
        extrapolated_bounds_2d: extrapolated_bounds(surface),
    }
}

fn extrapolated_bounds(surface: &SurfaceTrack) -> Bounds2 {
    if surface.kind != SurfaceKind::VerticalPlane {
        return surface.bounds_2d;
    }
    let horizontal = (surface.bounds_2d.max_u - surface.bounds_2d.min_u).abs();
    let vertical = (surface.bounds_2d.max_v - surface.bounds_2d.min_v).abs();
    let grow_u = ((1.2 - horizontal) * 0.5).max(0.0).min(0.4);
    let grow_v = ((0.9 - vertical) * 0.5).max(0.0).min(0.25);
    Bounds2 {
        min_u: surface.bounds_2d.min_u - grow_u,
        max_u: surface.bounds_2d.max_u + grow_u,
        min_v: surface.bounds_2d.min_v - grow_v,
        max_v: surface.bounds_2d.max_v + grow_v,
    }
}

fn project_cluster(cluster: &ClusterObservation, expected_pose: Pose2) -> ProjectedCluster {
    ProjectedCluster {
        id: cluster.id.clone(),
        centroid: world_to_robot(cluster.centroid, expected_pose),
        size_m: cluster.size_m,
        confidence: cluster.confidence,
        moving: cluster.moving,
        semantic_hint: cluster.semantic_hint.clone(),
    }
}

fn projected_obstacle_grid(
    current: &SurfaceExtractorOutput,
    current_pose: Pose2,
    expected_pose: Pose2,
) -> OccupancyGrid {
    let mut cells = HashMap::<(i32, i32), OccupancyState>::new();
    let resolution_m = current.obstacle_grid.resolution_m.max(0.05);
    let half_extent_m = current.obstacle_grid.half_extent_m.max(1.0);
    for cell in &current.obstacle_grid.cells {
        let local = Vec3::new(
            (cell.x as f32 + 0.5) * resolution_m,
            (cell.y as f32 + 0.5) * resolution_m,
            0.0,
        );
        let world = robot_to_world(local, current_pose);
        let future = world_to_robot(world, expected_pose);
        if future.x.abs() <= half_extent_m && future.y.abs() <= half_extent_m {
            let key = occupancy_key(future, resolution_m);
            cells.insert(key, cell.state.clone());
        }
    }
    for surface in &current.stable_surfaces {
        if surface.kind == SurfaceKind::VerticalPlane && surface.confidence >= 0.2 {
            mark_projected_surface_cells(
                &mut cells,
                surface,
                expected_pose,
                resolution_m,
                half_extent_m,
            );
        }
    }
    for cluster in &current.clusters {
        mark_projected_cluster_cells(
            &mut cells,
            cluster,
            expected_pose,
            resolution_m,
            half_extent_m,
        );
    }
    OccupancyGrid {
        resolution_m,
        half_extent_m,
        cells: cells
            .into_iter()
            .map(|((x, y), state)| OccupancyCell { x, y, state })
            .collect(),
    }
}

fn mark_projected_surface_cells(
    cells: &mut HashMap<(i32, i32), OccupancyState>,
    surface: &SurfaceTrack,
    expected_pose: Pose2,
    resolution_m: f32,
    half_extent_m: f32,
) {
    let bounds = extrapolated_bounds(surface);
    let normal = surface.normal;
    let basis_a = if normal.z.abs() < 0.9 {
        normal.cross(Vec3::new(0.0, 0.0, 1.0))
    } else {
        normal.cross(Vec3::new(1.0, 0.0, 0.0))
    }
    .normalized()
    .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
    let basis_b = normal
        .cross(basis_a)
        .normalized()
        .unwrap_or(Vec3::new(0.0, 1.0, 0.0));
    let span_u = (bounds.max_u - bounds.min_u).abs().max(resolution_m);
    let span_v = (bounds.max_v - bounds.min_v).abs().max(resolution_m);
    let steps_u = (span_u / resolution_m).ceil().clamp(1.0, 48.0) as usize;
    let steps_v = (span_v / resolution_m).ceil().clamp(1.0, 24.0) as usize;
    for u_step in 0..=steps_u {
        let u = bounds.min_u + (bounds.max_u - bounds.min_u) * u_step as f32 / steps_u as f32;
        for v_step in 0..=steps_v {
            let v = bounds.min_v + (bounds.max_v - bounds.min_v) * v_step as f32 / steps_v as f32;
            let world = surface.centroid + basis_a * u + basis_b * v;
            let local = world_to_robot(world, expected_pose);
            if local.z >= 0.05
                && local.z <= 1.8
                && local.x.abs() <= half_extent_m
                && local.y.abs() <= half_extent_m
            {
                cells.insert(occupancy_key(local, resolution_m), OccupancyState::Occupied);
            }
        }
    }
}

fn mark_projected_cluster_cells(
    cells: &mut HashMap<(i32, i32), OccupancyState>,
    cluster: &ClusterObservation,
    expected_pose: Pose2,
    resolution_m: f32,
    half_extent_m: f32,
) {
    let local = world_to_robot(cluster.centroid, expected_pose);
    if local.x.abs() > half_extent_m || local.y.abs() > half_extent_m {
        return;
    }
    let radius = cluster
        .size_m
        .x
        .abs()
        .max(cluster.size_m.y.abs())
        .max(resolution_m)
        * 0.5;
    let steps = (radius / resolution_m).ceil().clamp(1.0, 6.0) as i32;
    let center = occupancy_key(local, resolution_m);
    for dx in -steps..=steps {
        for dy in -steps..=steps {
            cells.insert((center.0 + dx, center.1 + dy), OccupancyState::Occupied);
        }
    }
}

fn anticipated_navigation(grid: &OccupancyGrid, action: &ActionPrimitive) -> AnticipatedNavigation {
    let front_clear_m = clear_distance(grid, -0.25, 0.25);
    let left_clear_m = clear_distance(grid, 0.25, 1.1);
    let right_clear_m = clear_distance(grid, -1.1, -0.25);
    let motor = action_to_motor_command(Some(action));
    let mut collision_risk = clearance_risk(front_clear_m, 0.85);
    if motor.turn > 0.02 {
        collision_risk = collision_risk.max(clearance_risk(left_clear_m, 0.65));
    } else if motor.turn < -0.02 {
        collision_risk = collision_risk.max(clearance_risk(right_clear_m, 0.65));
    }
    if motor.forward <= 0.0 && motor.turn.abs() <= 0.02 {
        collision_risk *= 0.25;
    }
    AnticipatedNavigation {
        front_clear_m,
        left_clear_m,
        right_clear_m,
        collision_risk: collision_risk.clamp(0.0, 1.0),
        occupied_cells: grid
            .cells
            .iter()
            .filter(|cell| cell.state == OccupancyState::Occupied)
            .count(),
        free_cells: grid
            .cells
            .iter()
            .filter(|cell| cell.state == OccupancyState::Free)
            .count(),
    }
}

fn clearance_risk(clearance_m: Option<f32>, caution_m: f32) -> f32 {
    clearance_m
        .map(|clearance| ((caution_m - clearance) / caution_m).clamp(0.0, 1.0))
        .unwrap_or(0.0)
}

fn scene_graph(
    surfaces: &[SurfaceTrack],
    floor: Option<SurfaceTrack>,
    clusters: &[ClusterObservation],
    grid: &OccupancyGrid,
) -> SceneGraphSummary {
    SceneGraphSummary {
        floor,
        surfaces: surfaces.to_vec(),
        clusters: clusters.to_vec(),
        navigation: serde_json::json!({
            "front_clear_m": clear_distance(grid, 0.0, 0.4),
            "left_clear_m": clear_distance(grid, 0.6, 1.2),
            "right_clear_m": clear_distance(grid, -1.2, -0.6),
            "occupied_cells": grid.cells.iter().filter(|cell| cell.state == OccupancyState::Occupied).count(),
            "free_cells": grid.cells.iter().filter(|cell| cell.state == OccupancyState::Free).count(),
        }),
    }
}

fn clear_distance(grid: &OccupancyGrid, min_y: f32, max_y: f32) -> Option<f32> {
    grid.cells
        .iter()
        .filter(|cell| cell.state == OccupancyState::Occupied)
        .filter_map(|cell| {
            let x = (cell.x as f32 + 0.5) * grid.resolution_m;
            let y = (cell.y as f32 + 0.5) * grid.resolution_m;
            if x >= 0.0 && y >= min_y && y <= max_y {
                Some(x)
            } else {
                None
            }
        })
        .min_by(|left, right| left.total_cmp(right))
}
