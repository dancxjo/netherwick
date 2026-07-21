pub fn observation_from_snapshot(
    snapshot: &WorldSnapshot,
    t_ms: TimeMs,
    config: MapConfig,
) -> MapObservation {
    observation_from_parts(
        snapshot.body.odometry,
        odometry_confidence_from_motion(
            snapshot.body.velocity.forward_m_s,
            snapshot.body.velocity.turn_rad_s,
        ),
        &snapshot.range,
        serde_json::json!({
            "body": {
                "odometry": snapshot.body.odometry,
                "velocity": snapshot.body.velocity,
            },
            "range": snapshot.range,
        }),
        t_ms,
        config,
    )
}

fn source_frame_id_from_observation(observation: &MapObservation) -> Option<String> {
    observation
        .source_snapshot
        .get("frame_id")
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .or_else(|| Some(format!("t:{}", observation.t_ms)))
}

pub fn observation_from_now(now: &Now, config: MapConfig) -> MapObservation {
    observation_from_parts(
        now.body.odometry,
        odometry_confidence_from_motion(
            now.body.velocity.forward_m_s,
            now.body.velocity.turn_rad_s,
        ),
        &now.range,
        serde_json::json!({
            "body": {
                "odometry": now.body.odometry,
                "velocity": now.body.velocity,
            },
            "range": now.range,
            "source": now.extensions.get("source"),
            "mode": now.extensions.get("mode"),
            "frame_id": now.extensions.get("frame_id"),
        }),
        now.t_ms,
        config,
    )
}

fn observation_from_parts(
    odometry: Pose2,
    pose_confidence: f32,
    range: &RangeSense,
    source_snapshot: serde_json::Value,
    t_ms: TimeMs,
    config: MapConfig,
) -> MapObservation {
    let pose = PoseEstimate {
        pose: odometry,
        confidence: pose_confidence,
        covariance: [0.05, 0.05, 0.10],
        source: "odometry".to_string(),
        t_ms,
    };
    let beam_count = range.beams.len();
    let explicit_angles = range.beam_angles_rad.len() == beam_count;
    let range_beams = range
        .beams
        .iter()
        .enumerate()
        .filter_map(|(index, distance)| {
            let distance = *distance;
            if !distance.is_finite() || distance <= 0.0 {
                return None;
            }
            let ratio = if beam_count <= 1 {
                0.5
            } else {
                index as f32 / (beam_count - 1) as f32
            };
            let angle_rad = if explicit_angles {
                range.beam_angles_rad[index]
            } else {
                -config.range_fov_rad * 0.5 + ratio * config.range_fov_rad
            };
            if !angle_rad.is_finite() {
                return None;
            }
            let (angle_rad, planar_distance, endpoint_height, tilted_sensor) = range
                .extrinsics
                .map(|extrinsics| {
                    let endpoint = range_endpoint_in_robot(distance, angle_rad, extrinsics);
                    (
                        endpoint.y_m.atan2(endpoint.x_m),
                        endpoint.x_m.hypot(endpoint.y_m),
                        endpoint.z_m,
                        extrinsics.pitch_rad.abs() > 1.0e-4 || extrinsics.roll_rad.abs() > 1.0e-4,
                    )
                })
                .unwrap_or((angle_rad, distance, 0.0, false));
            // A downward-tilted lidar sees the floor by design. Keep those
            // returns in the 3D cloud, but do not turn the floor into a ring of
            // obstacles in the planar occupancy map.
            if tilted_sensor && endpoint_height <= 0.05 {
                return None;
            }
            if !planar_distance.is_finite() || planar_distance <= 0.0 {
                return None;
            }
            let nearest_hit = range
                .nearest_m
                .filter(|nearest| nearest.is_finite())
                .map(|nearest| (distance - nearest).abs() <= config.hit_epsilon_m)
                .unwrap_or(false);
            let hit = planar_distance <= config.max_range_m
                && (nearest_hit || distance < config.max_range_m - config.hit_epsilon_m);
            Some(RangeBeam {
                angle_rad,
                distance_m: planar_distance,
                hit,
                confidence: if hit { 0.9 } else { 0.65 },
            })
        })
        .collect();

    MapObservation {
        pose,
        range_beams,
        source_snapshot,
        t_ms,
    }
}

pub fn project_beam_endpoint(pose: Pose2, beam_angle_rad: f32, distance_m: f32) -> Pose2 {
    let heading = pose.heading_rad + beam_angle_rad;
    Pose2 {
        x_m: pose.x_m + heading.cos() * distance_m,
        y_m: pose.y_m + heading.sin() * distance_m,
        heading_rad: heading,
    }
}

pub fn cell_key(x_m: f32, y_m: f32, resolution_m: f32) -> CellKey {
    CellKey {
        x: (x_m / resolution_m).floor() as i32,
        y: (y_m / resolution_m).floor() as i32,
    }
}

pub fn trace_cells(
    pose: Pose2,
    beam_angle_rad: f32,
    distance_m: f32,
    resolution_m: f32,
) -> Vec<CellKey> {
    if distance_m <= 0.0 {
        return Vec::new();
    }
    let steps = (distance_m / (resolution_m * 0.5)).ceil().max(1.0) as usize;
    let heading = pose.heading_rad + beam_angle_rad;
    let mut cells = Vec::new();
    let mut last = None;
    for step in 1..=steps {
        let d = (step as f32 / steps as f32) * distance_m;
        let key = cell_key(
            pose.x_m + heading.cos() * d,
            pose.y_m + heading.sin() * d,
            resolution_m,
        );
        if last != Some(key) {
            cells.push(key);
            last = Some(key);
        }
    }
    cells
}
