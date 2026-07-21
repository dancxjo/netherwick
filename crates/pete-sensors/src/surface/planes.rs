fn extract_planes(
    points: &[Point3],
    config: &SurfaceExtractorConfig,
) -> (Vec<PlaneObservation>, Vec<Point3>) {
    let mut remaining = points.to_vec();
    let mut planes = Vec::new();
    for _ in 0..config.max_planes {
        if remaining.len() < config.min_plane_points {
            break;
        }
        let Some(model) = best_plane_model(&remaining, config) else {
            break;
        };
        let (inliers, outliers): (Vec<_>, Vec<_>) = remaining.into_iter().partition(|point| {
            plane_distance(model, point.position) <= config.plane_distance_threshold_m
        });
        if inliers.len() < config.min_plane_points {
            remaining = outliers;
            break;
        }
        let observation = plane_observation(model, &inliers, points.len());
        if plane_observation_is_coherent(&observation, config) {
            planes.push(observation);
            remaining = outliers;
        } else {
            remaining = outliers;
            remaining.extend(inliers);
            break;
        }
    }
    (planes, remaining)
}

fn best_plane_model(points: &[Point3], config: &SurfaceExtractorConfig) -> Option<(Vec3, f32)> {
    let mut best: Option<((Vec3, f32), usize)> = None;
    let n = points.len();
    let step_a = (n / 17).max(1);
    let step_b = (n / 11).max(2);
    let step_c = (n / 7).max(3);
    for a in (0..n).step_by(step_a) {
        for b in ((a + step_b)..n).step_by(step_b) {
            for c in ((b + step_c)..n).step_by(step_c) {
                let Some(model) =
                    plane_from_points(points[a].position, points[b].position, points[c].position)
                else {
                    continue;
                };
                let inliers = points
                    .iter()
                    .filter(|point| {
                        plane_distance(model, point.position) <= config.plane_distance_threshold_m
                    })
                    .count();
                if best
                    .as_ref()
                    .map_or(true, |(_, best_count)| inliers > *best_count)
                {
                    best = Some((model, inliers));
                }
            }
        }
    }
    best.map(|(model, _)| model)
}

fn plane_from_points(a: Vec3, b: Vec3, c: Vec3) -> Option<(Vec3, f32)> {
    let normal = canonical_normal((b - a).cross(c - a).normalized()?);
    let distance = -normal.dot(a);
    Some((normal, distance))
}

fn canonical_normal(mut normal: Vec3) -> Vec3 {
    let ax = normal.x.abs();
    let ay = normal.y.abs();
    let az = normal.z.abs();
    let dominant = if ax >= ay && ax >= az {
        normal.x
    } else if ay >= ax && ay >= az {
        normal.y
    } else {
        normal.z
    };
    if dominant < 0.0 {
        normal = normal * -1.0;
    }
    normal
}

fn plane_distance((normal, distance): (Vec3, f32), point: Vec3) -> f32 {
    (normal.dot(point) + distance).abs()
}

fn plane_observation(
    (normal, distance): (Vec3, f32),
    inliers: &[Point3],
    total_points: usize,
) -> PlaneObservation {
    let centroid = inliers
        .iter()
        .fold(Vec3::default(), |sum, point| sum + point.position)
        / inliers.len() as f32;
    PlaneObservation {
        normal,
        centroid,
        distance_from_origin_m: distance,
        bounds_2d: plane_bounds(normal, centroid, inliers),
        extent_m: point_extent(inliers),
        point_count: inliers.len(),
        confidence: (inliers.len() as f32 / total_points.max(1) as f32).clamp(0.0, 1.0),
        rms_error_m: plane_rms_error((normal, distance), inliers),
    }
}

fn plane_observation_is_coherent(
    observation: &PlaneObservation,
    config: &SurfaceExtractorConfig,
) -> bool {
    let span_u = (observation.bounds_2d.max_u - observation.bounds_2d.min_u).abs();
    let span_v = (observation.bounds_2d.max_v - observation.bounds_2d.min_v).abs();
    let major = span_u.max(span_v);
    let minor = span_u.min(span_v);
    let area = span_u * span_v;
    observation.point_count >= config.min_plane_points
        && major >= config.min_plane_major_extent_m
        && minor >= config.min_plane_minor_extent_m
        && area >= config.min_plane_area_m
        && observation.rms_error_m <= config.max_plane_rms_error_m
}

fn point_extent(points: &[Point3]) -> Vec3 {
    let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for point in points {
        let position = point.position;
        min.x = min.x.min(position.x);
        min.y = min.y.min(position.y);
        min.z = min.z.min(position.z);
        max.x = max.x.max(position.x);
        max.y = max.y.max(position.y);
        max.z = max.z.max(position.z);
    }
    max - min
}

fn plane_rms_error((normal, distance): (Vec3, f32), points: &[Point3]) -> f32 {
    if points.is_empty() {
        return 0.0;
    }
    let sum_sq = points
        .iter()
        .map(|point| {
            let error = normal.dot(point.position) + distance;
            error * error
        })
        .sum::<f32>();
    (sum_sq / points.len() as f32).sqrt()
}

fn plane_bounds(normal: Vec3, centroid: Vec3, points: &[Point3]) -> Bounds2 {
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
    let mut bounds = Bounds2 {
        min_u: f32::INFINITY,
        max_u: f32::NEG_INFINITY,
        min_v: f32::INFINITY,
        max_v: f32::NEG_INFINITY,
    };
    for point in points {
        let relative = point.position - centroid;
        let u = relative.dot(basis_a);
        let v = relative.dot(basis_b);
        bounds.min_u = bounds.min_u.min(u);
        bounds.max_u = bounds.max_u.max(u);
        bounds.min_v = bounds.min_v.min(v);
        bounds.max_v = bounds.max_v.max(v);
    }
    bounds
}

fn classify_surface(
    observation: &PlaneObservation,
    observations: &[PlaneObservation],
) -> SurfaceKind {
    if observation.normal.z.abs() > 0.88 {
        let lowest_horizontal = observations
            .iter()
            .filter(|plane| plane.normal.z.abs() > 0.88)
            .map(|plane| plane.centroid.z)
            .fold(f32::INFINITY, f32::min);
        if observation.centroid.z <= lowest_horizontal + 0.08 {
            SurfaceKind::Floor
        } else {
            SurfaceKind::HorizontalPlane
        }
    } else if observation.normal.z.abs() < 0.35 {
        SurfaceKind::VerticalPlane
    } else {
        SurfaceKind::UnknownPlane
    }
}

fn track_match_score(
    track: &SurfaceTrack,
    observation: &PlaneObservation,
    config: &SurfaceExtractorConfig,
) -> Option<f32> {
    let normal_dot = track.normal.dot(observation.normal).abs().clamp(0.0, 1.0);
    let normal_angle = normal_dot.acos();
    let distance_delta = (track.distance_from_origin_m - observation.distance_from_origin_m).abs();
    let centroid_delta = track_centroid_delta(track, observation);
    if normal_angle > config.track_normal_max_angle_rad
        || distance_delta > config.track_distance_threshold_m
        || centroid_delta > track_centroid_threshold(track, config)
    {
        None
    } else {
        let overlap_bonus = bounds_overlap_ratio(track.bounds_2d, observation.bounds_2d) * 0.25;
        Some((normal_angle + distance_delta + centroid_delta - overlap_bonus).max(0.0))
    }
}

fn bounds_overlap_ratio(left: Bounds2, right: Bounds2) -> f32 {
    let overlap_u = (left.max_u.min(right.max_u) - left.min_u.max(right.min_u)).max(0.0);
    let overlap_v = (left.max_v.min(right.max_v) - left.min_v.max(right.min_v)).max(0.0);
    let overlap = overlap_u * overlap_v;
    let left_area = ((left.max_u - left.min_u) * (left.max_v - left.min_v)).abs();
    let right_area = ((right.max_u - right.min_u) * (right.max_v - right.min_v)).abs();
    let union = left_area + right_area - overlap;
    if union <= f32::EPSILON {
        0.0
    } else {
        (overlap / union).clamp(0.0, 1.0)
    }
}

fn track_centroid_delta(track: &SurfaceTrack, observation: &PlaneObservation) -> f32 {
    let delta = observation.centroid - track.centroid;
    if track.kind == SurfaceKind::VerticalPlane && track.normal.z.abs() < 0.35 {
        delta.dot(track.normal).abs()
    } else {
        delta.length()
    }
}

fn track_centroid_threshold(track: &SurfaceTrack, config: &SurfaceExtractorConfig) -> f32 {
    if track.kind == SurfaceKind::VerticalPlane && track.normal.z.abs() < 0.35 {
        (config.track_centroid_threshold_m * 2.5).max(0.9)
    } else {
        config.track_centroid_threshold_m
    }
}

fn smooth_track(
    track: &mut SurfaceTrack,
    observation: &PlaneObservation,
    kind: SurfaceKind,
    alpha: f32,
    seen_gain: f32,
    t_ms: u64,
) {
    let alpha = alpha.clamp(0.0, 1.0);
    track.kind = if track.kind == SurfaceKind::Floor {
        SurfaceKind::Floor
    } else {
        kind
    };
    track.normal = (track.normal * (1.0 - alpha) + observation.normal * alpha)
        .normalized()
        .unwrap_or(observation.normal);
    track.centroid = track.centroid * (1.0 - alpha) + observation.centroid * alpha;
    track.distance_from_origin_m =
        track.distance_from_origin_m * (1.0 - alpha) + observation.distance_from_origin_m * alpha;
    track.bounds_2d = observation.bounds_2d;
    track.extent_m = observation.extent_m;
    track.confidence = (track.confidence + seen_gain + observation.confidence * 0.1).min(1.0);
    track.supporting_point_count = observation.point_count;
    track.last_seen_ms = t_ms;
    track.seen_count += 1;
    track.missing_count = 0;
    track.labels = surface_labels(track.kind);
}
