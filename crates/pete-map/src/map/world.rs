pub fn transform_point_to_world(
    point: Point3D,
    frame: PointCloudFrame,
    pose: Pose2,
    orientation: OrientationEstimate,
    config: PointCloudConfig,
) -> Point3D {
    let robot = match frame {
        PointCloudFrame::OdometryWorld => return point,
        PointCloudFrame::RobotBase => point,
        PointCloudFrame::KinectCamera | PointCloudFrame::DepthImageUnknown => {
            camera_point_to_robot(point, config)
        }
    };
    let yaw = orientation.yaw_rad.unwrap_or(pose.heading_rad);
    let world = pete_now::DepthGeometry::base_point_to_world(
        [robot.x_m, robot.y_m, robot.z_m],
        Pose2 {
            heading_rad: yaw,
            ..pose
        },
        orientation.roll_rad,
        orientation.pitch_rad,
    );
    Point3D {
        x_m: world[0],
        y_m: world[1],
        z_m: world[2],
    }
}

pub fn orientation_from_snapshot(snapshot: &WorldSnapshot) -> OrientationEstimate {
    orientation_from_imu(&snapshot.imu, snapshot.body.odometry.heading_rad)
}

pub fn orientation_from_imu(imu: &ImuSense, odometry_heading_rad: f32) -> OrientationEstimate {
    let trusted = pete_now::trusted_imu_orientation(imu);
    let trusted_roll = trusted.roll_rad;
    let trusted_pitch = trusted.pitch_rad;
    let imu_yaw = trusted.yaw_rad;
    OrientationEstimate {
        roll_rad: trusted_roll,
        pitch_rad: trusted_pitch,
        yaw_rad: imu_yaw.or(Some(odometry_heading_rad)),
        roll_pitch_from_imu: trusted_roll.is_some() || trusted_pitch.is_some(),
        yaw_source: if imu_yaw.is_some() {
            YawSource::ImuOrientation
        } else {
            YawSource::OdometryHeading
        },
    }
}

fn camera_point_to_robot(point: Point3D, config: PointCloudConfig) -> Point3D {
    let base = Point3D {
        x_m: point.z_m,
        y_m: -point.x_m,
        z_m: -point.y_m,
    };
    let rotated = rotate_robot_extrinsic(
        base,
        config.camera_pitch_rad,
        config.camera_roll_rad,
        config.camera_yaw_rad,
    );
    Point3D {
        x_m: rotated.x_m + config.camera_forward_m,
        y_m: rotated.y_m + config.camera_left_m,
        z_m: rotated.z_m + config.camera_height_m,
    }
}

fn rotate_robot_extrinsic(point: Point3D, pitch_rad: f32, roll_rad: f32, yaw_rad: f32) -> Point3D {
    let (pitch_sin, pitch_cos) = pitch_rad.sin_cos();
    let mut x = point.x_m * pitch_cos + point.z_m * pitch_sin;
    let y = point.y_m;
    let mut z = -point.x_m * pitch_sin + point.z_m * pitch_cos;

    let (roll_sin, roll_cos) = roll_rad.sin_cos();
    let rolled_y = y * roll_cos - z * roll_sin;
    z = y * roll_sin + z * roll_cos;

    let (yaw_sin, yaw_cos) = yaw_rad.sin_cos();
    let yawed_x = x * yaw_cos - rolled_y * yaw_sin;
    let yawed_y = x * yaw_sin + rolled_y * yaw_cos;
    x = yawed_x;

    Point3D {
        x_m: x,
        y_m: yawed_y,
        z_m: z,
    }
}

fn range_endpoint_in_robot(
    distance_m: f32,
    angle_rad: f32,
    extrinsics: RangeExtrinsics,
) -> Point3D {
    let sensor = Point3D {
        x_m: distance_m * angle_rad.cos(),
        y_m: distance_m * angle_rad.sin(),
        z_m: 0.0,
    };
    let rotated = rotate_robot_extrinsic(
        sensor,
        extrinsics.pitch_rad,
        extrinsics.roll_rad,
        extrinsics.yaw_rad,
    );
    Point3D {
        x_m: rotated.x_m + extrinsics.forward_m,
        y_m: rotated.y_m + extrinsics.left_m,
        z_m: rotated.z_m + extrinsics.height_m,
    }
}

fn orientation_status(orientation: OrientationEstimate) -> OrientationStatus {
    let roll_pitch_corrected = orientation.roll_pitch_from_imu;
    let note = match (roll_pitch_corrected, orientation.yaw_source) {
        (true, YawSource::ImuOrientation) => {
            "depth cloud uses IMU roll/pitch and IMU yaw before world accumulation"
        }
        (true, YawSource::OdometryHeading) => {
            "depth cloud uses IMU roll/pitch; yaw remains odometry heading because no IMU yaw is available"
        }
        (false, YawSource::ImuOrientation) => {
            "depth cloud uses IMU yaw, but no IMU roll/pitch was available"
        }
        (false, YawSource::OdometryHeading) => {
            "depth cloud is planar odometry-frame only; no IMU roll/pitch is available"
        }
        (_, YawSource::Unavailable) => "depth cloud orientation is unavailable",
    };
    OrientationStatus {
        roll_pitch_corrected,
        yaw_source: orientation.yaw_source,
        note: note.to_string(),
    }
}

fn local_world_belief_from_voxels(cloud: &VoxelPointCloud) -> LocalWorldBelief {
    let components = stable_components(cloud);
    let mut stable_surfaces = Vec::new();
    let mut stable_blobs = Vec::new();

    for (index, component) in components.iter().enumerate() {
        let stats = component_stats(component);
        if component.len() >= 4 {
            if let Some(kind) = surface_kind_from_extent(stats.size_m) {
                stable_surfaces.push(WorldSurfaceHypothesis {
                    id: format!("surface_{}", index + 1),
                    kind,
                    centroid: stats.centroid,
                    normal: normal_for_surface(kind, stats.size_m),
                    size_m: stats.size_m,
                    voxel_count: component.len(),
                    confidence: stats.confidence,
                    first_seen_ms: stats.first_seen_ms,
                    last_seen_ms: stats.last_seen_ms,
                });
                continue;
            }
        }

        stable_blobs.push(WorldBlobHypothesis {
            id: format!("blob_{}", index + 1),
            centroid: stats.centroid,
            size_m: stats.size_m,
            voxel_count: component.len(),
            confidence: stats.confidence,
            first_seen_ms: stats.first_seen_ms,
            last_seen_ms: stats.last_seen_ms,
        });
    }

    let summary = cloud.summary();
    LocalWorldBelief {
        label: "persistent local world belief from accumulated stable voxels, not full SLAM",
        orientation_status: cloud.orientation_status.clone(),
        stable_surfaces,
        stable_blobs,
        stable_voxels: summary.stable_voxels,
        transient_voxels: summary.transient_voxels,
        observations: summary.observations,
        latest_t_ms: summary.latest_t_ms,
    }
}

#[derive(Clone, Debug)]
struct ComponentStats {
    centroid: Point3D,
    size_m: Point3D,
    confidence: f32,
    first_seen_ms: TimeMs,
    last_seen_ms: TimeMs,
}

fn stable_components(cloud: &VoxelPointCloud) -> Vec<Vec<VoxelPoint>> {
    let stable = cloud
        .voxels
        .iter()
        .filter(|(_, voxel)| voxel.stable)
        .map(|(key, voxel)| (*key, voxel.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut remaining = stable.keys().copied().collect::<Vec<_>>();
    let mut components = Vec::new();

    while let Some(seed) = remaining.pop() {
        if !stable.contains_key(&seed) {
            continue;
        }
        let mut stack = vec![seed];
        let mut component = Vec::new();
        while let Some(key) = stack.pop() {
            let Some(voxel) = stable.get(&key) else {
                continue;
            };
            if component
                .iter()
                .any(|existing: &VoxelPoint| existing.key == key)
            {
                continue;
            }
            component.push(voxel.clone());
            remaining.retain(|candidate| *candidate != key);
            for neighbor in voxel_neighbors(key) {
                if stable.contains_key(&neighbor)
                    && !component
                        .iter()
                        .any(|existing: &VoxelPoint| existing.key == neighbor)
                {
                    stack.push(neighbor);
                }
            }
        }
        if !component.is_empty() {
            components.push(component);
        }
    }

    components
}

fn voxel_neighbors(key: VoxelKey) -> impl Iterator<Item = VoxelKey> {
    (-1..=1).flat_map(move |dx| {
        (-1..=1).flat_map(move |dy| {
            (-1..=1).filter_map(move |dz| {
                (dx != 0 || dy != 0 || dz != 0).then_some(VoxelKey {
                    x: key.x + dx,
                    y: key.y + dy,
                    z: key.z + dz,
                })
            })
        })
    })
}

fn component_stats(component: &[VoxelPoint]) -> ComponentStats {
    let mut min = Point3D {
        x_m: f32::INFINITY,
        y_m: f32::INFINITY,
        z_m: f32::INFINITY,
    };
    let mut max = Point3D {
        x_m: f32::NEG_INFINITY,
        y_m: f32::NEG_INFINITY,
        z_m: f32::NEG_INFINITY,
    };
    let mut sum = Point3D::default();
    let mut confidence_sum = 0.0;
    let mut first_seen_ms = TimeMs::MAX;
    let mut last_seen_ms = 0;
    for voxel in component {
        min.x_m = min.x_m.min(voxel.position.x_m);
        min.y_m = min.y_m.min(voxel.position.y_m);
        min.z_m = min.z_m.min(voxel.position.z_m);
        max.x_m = max.x_m.max(voxel.position.x_m);
        max.y_m = max.y_m.max(voxel.position.y_m);
        max.z_m = max.z_m.max(voxel.position.z_m);
        sum.x_m += voxel.position.x_m;
        sum.y_m += voxel.position.y_m;
        sum.z_m += voxel.position.z_m;
        confidence_sum += voxel.confidence;
        first_seen_ms = first_seen_ms.min(voxel.first_seen_ms);
        last_seen_ms = last_seen_ms.max(voxel.last_seen_ms);
    }
    let count = component.len().max(1) as f32;
    ComponentStats {
        centroid: Point3D {
            x_m: sum.x_m / count,
            y_m: sum.y_m / count,
            z_m: sum.z_m / count,
        },
        size_m: Point3D {
            x_m: max.x_m - min.x_m,
            y_m: max.y_m - min.y_m,
            z_m: max.z_m - min.z_m,
        },
        confidence: (confidence_sum / count).clamp(0.0, 1.0),
        first_seen_ms,
        last_seen_ms,
    }
}

fn surface_kind_from_extent(size: Point3D) -> Option<WorldSurfaceKind> {
    let thickness = size.x_m.min(size.y_m).min(size.z_m);
    let span = size.x_m.max(size.y_m).max(size.z_m);
    if span < 0.15 || thickness > 0.16 {
        return None;
    }
    if size.z_m <= 0.12 && size.x_m.max(size.y_m) >= 0.25 {
        Some(WorldSurfaceKind::FloorLike)
    } else if size.x_m.min(size.y_m) <= 0.12 && size.z_m >= 0.20 {
        Some(WorldSurfaceKind::WallLike)
    } else if size.z_m <= 0.16 {
        Some(WorldSurfaceKind::HorizontalSurface)
    } else {
        Some(WorldSurfaceKind::UnknownSurface)
    }
}

fn normal_for_surface(kind: WorldSurfaceKind, size: Point3D) -> Point3D {
    match kind {
        WorldSurfaceKind::FloorLike | WorldSurfaceKind::HorizontalSurface => Point3D {
            x_m: 0.0,
            y_m: 0.0,
            z_m: 1.0,
        },
        WorldSurfaceKind::WallLike if size.x_m <= size.y_m => Point3D {
            x_m: 1.0,
            y_m: 0.0,
            z_m: 0.0,
        },
        WorldSurfaceKind::WallLike => Point3D {
            x_m: 0.0,
            y_m: 1.0,
            z_m: 0.0,
        },
        WorldSurfaceKind::UnknownSurface => Point3D {
            x_m: 0.0,
            y_m: 0.0,
            z_m: 0.0,
        },
    }
}
