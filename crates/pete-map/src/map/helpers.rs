pub fn voxel_key(point: Point3D, voxel_size_m: f32) -> VoxelKey {
    VoxelKey {
        x: (point.x_m / voxel_size_m).floor() as i32,
        y: (point.y_m / voxel_size_m).floor() as i32,
        z: (point.z_m / voxel_size_m).floor() as i32,
    }
}

#[derive(Clone, Copy, Debug)]
struct DepthProjection {
    width: usize,
    height: usize,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    frame: PointCloudFrame,
}

impl DepthProjection {
    fn from_kinect(kinect: &KinectSense) -> Option<Self> {
        let width = usize::try_from(kinect.depth_width).ok()?;
        let height = usize::try_from(kinect.depth_height).ok()?;
        if width == 0 || height == 0 || width.checked_mul(height)? != kinect.depth_m.len() {
            return None;
        }
        Some(Self {
            width,
            height,
            fx: positive_or(kinect.depth_fx, 594.0),
            fy: positive_or(kinect.depth_fy, 591.0),
            cx: if kinect.depth_cx > 0.0 {
                kinect.depth_cx
            } else {
                (width as f32 - 1.0) * 0.5
            },
            cy: if kinect.depth_cy > 0.0 {
                kinect.depth_cy
            } else {
                (height as f32 - 1.0) * 0.5
            },
            frame: PointCloudFrame::KinectCamera,
        })
    }
}

fn positive_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

fn depth_shade(depth_m: f32, max_depth_m: f32) -> Option<[u8; 3]> {
    let shade = ((1.0 - (depth_m / max_depth_m.max(f32::EPSILON))).clamp(0.15, 1.0) * 255.0) as u8;
    Some([shade, shade, 255])
}

fn merge_color(
    existing: Option<[u8; 3]>,
    incoming: Option<[u8; 3]>,
    seen_count: u32,
) -> Option<[u8; 3]> {
    match (existing, incoming) {
        (Some(existing), Some(incoming)) => {
            let seen = seen_count as u32;
            let denom = seen.saturating_add(1).max(1);
            Some([
                ((existing[0] as u32 * seen + incoming[0] as u32) / denom) as u8,
                ((existing[1] as u32 * seen + incoming[1] as u32) / denom) as u8,
                ((existing[2] as u32 * seen + incoming[2] as u32) / denom) as u8,
            ])
        }
        (Some(existing), None) => Some(existing),
        (None, Some(incoming)) => Some(incoming),
        (None, None) => None,
    }
}

fn odometry_confidence_from_motion(forward_m_s: f32, turn_rad_s: f32) -> f32 {
    let moving = forward_m_s.abs() + turn_rad_s.abs();
    if moving > 0.001 {
        0.85
    } else {
        0.75
    }
}

fn cap_vec<T>(items: &mut Vec<T>, max_len: usize) {
    if max_len == 0 {
        items.clear();
        return;
    }
    let overflow = items.len().saturating_sub(max_len);
    if overflow > 0 {
        items.drain(0..overflow);
    }
}

fn pose_delta(from: Pose2, to: Pose2) -> Pose2 {
    let world_x = to.x_m - from.x_m;
    let world_y = to.y_m - from.y_m;
    let (from_sin, from_cos) = from.heading_rad.sin_cos();
    Pose2 {
        x_m: from_cos * world_x + from_sin * world_y,
        y_m: -from_sin * world_x + from_cos * world_y,
        heading_rad: normalize_angle(to.heading_rad - from.heading_rad),
    }
}

fn apply_pose_delta(from: Pose2, delta: Pose2) -> Pose2 {
    let (from_sin, from_cos) = from.heading_rad.sin_cos();
    Pose2 {
        x_m: from.x_m + from_cos * delta.x_m - from_sin * delta.y_m,
        y_m: from.y_m + from_sin * delta.x_m + from_cos * delta.y_m,
        heading_rad: normalize_angle(from.heading_rad + delta.heading_rad),
    }
}

fn edge_constraint_weight(edge: &PoseEdge) -> f32 {
    let covariance = edge
        .covariance
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .sum::<f32>()
        / edge.covariance.len().max(1) as f32;
    let covariance_weight = 1.0 / (1.0 + covariance.max(0.001) * 4.0);
    (edge.confidence.clamp(0.05, 1.0) * covariance_weight).clamp(0.01, 1.0)
}

fn clamp_pose_update(mut update: Pose2, config: PoseGraphOptimizationConfig) -> Pose2 {
    let translation = update.x_m.hypot(update.y_m);
    if translation > config.max_translation_update_m && translation > f32::EPSILON {
        let scale = config.max_translation_update_m / translation;
        update.x_m *= scale;
        update.y_m *= scale;
    }
    update.heading_rad = update.heading_rad.clamp(
        -config.max_heading_update_rad,
        config.max_heading_update_rad,
    );
    update
}

fn distance_m(left: Pose2, right: Pose2) -> f32 {
    ((right.x_m - left.x_m).powi(2) + (right.y_m - left.y_m).powi(2)).sqrt()
}

fn heading_delta_rad(left: f32, right: f32) -> f32 {
    normalize_angle(right - left).abs()
}

fn normalize_angle(angle: f32) -> f32 {
    let mut normalized = angle;
    while normalized > std::f32::consts::PI {
        normalized -= std::f32::consts::TAU;
    }
    while normalized < -std::f32::consts::PI {
        normalized += std::f32::consts::TAU;
    }
    normalized
}

fn loop_covariance(confidence: f32) -> [f32; 3] {
    let uncertainty = (1.0 - confidence.clamp(0.0, 1.0)).max(0.05);
    [uncertainty * 0.20, uncertainty * 0.20, uncertainty * 0.35]
}

fn confidence_distribution(confidences: impl Iterator<Item = f32>) -> ConfidenceDistribution {
    let mut values = Vec::new();
    let mut buckets = BTreeMap::new();
    for confidence in confidences {
        let confidence = confidence.clamp(0.0, 1.0);
        values.push(confidence);
        let bucket = match confidence {
            c if c < 0.50 => "0.00-0.49",
            c if c < 0.70 => "0.50-0.69",
            c if c < 0.85 => "0.70-0.84",
            c if c < 0.95 => "0.85-0.94",
            _ => "0.95-1.00",
        };
        *buckets.entry(bucket.to_string()).or_insert(0) += 1;
    }

    if values.is_empty() {
        return ConfidenceDistribution {
            min: None,
            max: None,
            mean: None,
            buckets,
        };
    }

    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    ConfidenceDistribution {
        min: Some(min),
        max: Some(max),
        mean: Some(mean),
        buckets,
    }
}
