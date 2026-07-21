pub fn pointcloud_observation_from_snapshot(
    snapshot: &WorldSnapshot,
    t_ms: TimeMs,
    config: PointCloudConfig,
) -> Option<PointCloudObservation> {
    pointcloud_observations_from_snapshot(snapshot, t_ms, config)
        .into_iter()
        .next()
}

pub fn pointcloud_observations_from_snapshot(
    snapshot: &WorldSnapshot,
    t_ms: TimeMs,
    config: PointCloudConfig,
) -> Vec<PointCloudObservation> {
    let paired_color = snapshot.kinect.color_frame.as_ref().filter(|frame| {
        frame
            .captured_at_ms
            .abs_diff(snapshot.kinect.captured_at_ms)
            <= 50
            && frame.rgbd_frame_id.is_some()
            && frame.rgbd_frame_id == snapshot.kinect.rgbd_frame_id
    });
    let legacy_color = (snapshot.kinect.schema_version < 2)
        .then_some(snapshot.eye_frame.as_ref())
        .flatten();
    let color = paired_color
        .or(legacy_color)
        .and_then(DepthColorImage::from_eye_frame);
    let pose = snapshot.body.odometry;
    let orientation = orientation_from_snapshot(snapshot);
    let pose_confidence = odometry_confidence_from_motion(
        snapshot.body.velocity.forward_m_s,
        snapshot.body.velocity.turn_rad_s,
    );
    let mut observations = Vec::new();
    let aligned = snapshot.kinect.fusion_alignment.as_ref();
    if snapshot.kinect.schema_version < 2 || aligned.is_some() {
        let kinect_pose = aligned.map(|aligned| aligned.pose).unwrap_or(pose);
        let kinect_orientation = aligned
            .map(|aligned| orientation_from_imu(&aligned.imu, aligned.pose.heading_rad))
            .unwrap_or(orientation);
        let kinect_confidence = aligned
            .map(|aligned| pose_confidence * aligned.confidence)
            .unwrap_or(pose_confidence);
        let kinect_t_ms = (snapshot.kinect.captured_at_ms > 0)
            .then_some(snapshot.kinect.captured_at_ms)
            .unwrap_or(t_ms);
        if let Some(observation) = pointcloud_observation_from_kinect_with_color(
            &snapshot.kinect,
            kinect_pose,
            kinect_orientation,
            kinect_confidence,
            kinect_t_ms,
            config,
            color.as_ref(),
        ) {
            observations.push(observation);
        }
    }
    if let Some(observation) = pointcloud_observation_from_range(
        &snapshot.range,
        pose,
        orientation,
        pose_confidence,
        t_ms,
        config,
    ) {
        observations.push(observation);
    }
    observations
}

pub fn pointcloud_observation_from_range(
    range: &RangeSense,
    pose: Pose2,
    orientation: OrientationEstimate,
    pose_confidence: f32,
    t_ms: TimeMs,
    config: PointCloudConfig,
) -> Option<PointCloudObservation> {
    let extrinsics = range.extrinsics?;
    if range.beams.is_empty() || range.beam_angles_rad.len() != range.beams.len() {
        return None;
    }
    let stride = range
        .beams
        .len()
        .div_ceil(config.max_points_per_observation.max(1))
        .max(1);
    let source_frame_id =
        (range.captured_at_ms > 0).then(|| format!("range-{}", range.captured_at_ms));
    let beam_poses = (range.beam_poses.len() == range.beams.len()).then_some(&range.beam_poses);
    let points = range
        .beams
        .iter()
        .zip(&range.beam_angles_rad)
        .enumerate()
        .step_by(stride)
        .filter_map(|(index, (distance_m, angle_rad))| {
            if !distance_m.is_finite()
                || *distance_m <= 0.0
                || *distance_m > config.max_depth_m
                || !angle_rad.is_finite()
            {
                return None;
            }
            let robot_point = range_endpoint_in_robot(*distance_m, *angle_rad, extrinsics);
            let position = beam_poses
                .map(|poses| {
                    transform_point_to_world(
                        robot_point,
                        PointCloudFrame::RobotBase,
                        poses[index],
                        OrientationEstimate {
                            yaw_rad: Some(poses[index].heading_rad),
                            yaw_source: YawSource::OdometryHeading,
                            ..OrientationEstimate::default()
                        },
                        config,
                    )
                })
                .unwrap_or(robot_point);
            Some(PointCloudPoint {
                position,
                color_rgb: None,
                confidence: pose_confidence,
                depth_index: Some(index),
                depth_uv: None,
                depth_image_size: None,
                source_frame_id: source_frame_id.clone(),
            })
        })
        .collect::<Vec<_>>();
    if points.is_empty() {
        return None;
    }
    Some(PointCloudObservation {
        frame: if beam_poses.is_some() {
            PointCloudFrame::OdometryWorld
        } else {
            PointCloudFrame::RobotBase
        },
        pose: PoseEstimate {
            pose,
            confidence: pose_confidence,
            covariance: [0.05, 0.05, 0.10],
            source: "odometry".to_string(),
            t_ms,
        },
        orientation,
        points,
        source: range.source.clone().unwrap_or_else(|| "range".to_string()),
        t_ms,
        metadata: serde_json::json!({
            "beam_count": range.beams.len(),
            "sample_stride": stride,
            "sensor_extrinsics": extrinsics,
            "motion_deskewed": beam_poses.is_some(),
            "beam_pose_max_skew_ms": range.beam_pose_max_skew_ms,
            "orientation": orientation,
        }),
    })
}

pub fn pointcloud_observation_from_kinect(
    kinect: &KinectSense,
    pose: Pose2,
    orientation: OrientationEstimate,
    pose_confidence: f32,
    t_ms: TimeMs,
    config: PointCloudConfig,
) -> Option<PointCloudObservation> {
    pointcloud_observation_from_kinect_with_color(
        kinect,
        pose,
        orientation,
        pose_confidence,
        t_ms,
        config,
        None,
    )
}

fn pointcloud_observation_from_kinect_with_color(
    kinect: &KinectSense,
    pose: Pose2,
    orientation: OrientationEstimate,
    pose_confidence: f32,
    t_ms: TimeMs,
    config: PointCloudConfig,
    color: Option<&DepthColorImage>,
) -> Option<PointCloudObservation> {
    if kinect.depth_m.is_empty() {
        return None;
    }
    let projection = DepthProjection::from_kinect(kinect)?;
    let geometry = pete_now::DepthGeometry::from_kinect(kinect)?;
    let calibrated_base = kinect.geometry_calibration.is_some();
    let stride = kinect
        .depth_m
        .len()
        .div_ceil(config.max_points_per_observation.max(1))
        .max(1);
    let min_depth_m = positive_or(kinect.min_depth_m, config.min_depth_m);
    let max_depth_m = positive_or(kinect.max_depth_m, config.max_depth_m);
    let source_frame_id =
        (kinect.captured_at_ms > 0).then(|| format!("kinect-depth-{}", kinect.captured_at_ms));
    let mut skipped_depth_count = 0usize;
    let mut clipped_depth_count = 0usize;
    let mut points = Vec::new();
    let mut rgb_nearest_depth = BTreeMap::<(u32, u32), f32>::new();
    if color.is_some() && kinect.geometry_calibration.is_some() {
        for (index, depth) in kinect.depth_m.iter().enumerate().step_by(stride) {
            if !depth.is_finite() || *depth < min_depth_m || *depth > max_depth_m {
                continue;
            }
            let u = (index % projection.width) as f32;
            let v = (index / projection.width) as f32;
            let Some(camera_point) = geometry.depth_pixel_to_camera(u, v, *depth) else {
                continue;
            };
            let Some(pixel) = geometry.depth_point_to_rgb_pixel(camera_point) else {
                continue;
            };
            let key = (pixel[0].round() as u32, pixel[1].round() as u32);
            rgb_nearest_depth
                .entry(key)
                .and_modify(|nearest| *nearest = nearest.min(camera_point[2]))
                .or_insert(camera_point[2]);
        }
    }
    for (index, depth) in kinect.depth_m.iter().enumerate().step_by(stride) {
        if !depth.is_finite() || *depth <= 0.0 {
            skipped_depth_count = skipped_depth_count.saturating_add(1);
            continue;
        }
        if *depth < min_depth_m || *depth > max_depth_m {
            clipped_depth_count = clipped_depth_count.saturating_add(1);
            continue;
        }
        let u = (index % projection.width) as f32;
        let v = (index / projection.width) as f32;
        let camera_point = geometry.depth_pixel_to_camera(u, v, *depth)?;
        let [x_m, y_m, z_m] = if calibrated_base {
            geometry.depth_point_to_base(camera_point)
        } else {
            camera_point
        };
        let color_rgb = color
            .and_then(|color| {
                if kinect.geometry_calibration.is_some() {
                    geometry
                        .depth_point_to_rgb_pixel(camera_point)
                        .filter(|pixel| {
                            let key = (pixel[0].round() as u32, pixel[1].round() as u32);
                            rgb_nearest_depth
                                .get(&key)
                                .is_some_and(|nearest| camera_point[2] <= *nearest + 0.02)
                        })
                        .and_then(|pixel| color.sample_projected(pixel))
                } else if kinect.schema_version < 2 {
                    color.sample_depth_pixel(
                        index % projection.width,
                        index / projection.width,
                        projection.width,
                        projection.height,
                    )
                } else {
                    None
                }
            })
            .or_else(|| depth_shade(z_m, max_depth_m));
        points.push(PointCloudPoint {
            position: Point3D { x_m, y_m, z_m },
            color_rgb,
            confidence: pose_confidence,
            depth_index: Some(index),
            depth_uv: Some([u as u32, v as u32]),
            depth_image_size: Some([projection.width as u32, projection.height as u32]),
            source_frame_id: source_frame_id.clone(),
        });
    }
    if points.is_empty() {
        return None;
    }
    Some(PointCloudObservation {
        frame: if calibrated_base {
            PointCloudFrame::RobotBase
        } else {
            projection.frame
        },
        pose: PoseEstimate {
            pose,
            confidence: pose_confidence,
            covariance: [0.05, 0.05, 0.10],
            source: "odometry".to_string(),
            t_ms,
        },
        orientation,
        points,
        source: "kinect_depth".to_string(),
        t_ms,
        metadata: serde_json::json!({
            "depth_width": projection.width,
            "depth_height": projection.height,
            "depth_fx": projection.fx,
            "depth_fy": projection.fy,
            "depth_cx": projection.cx,
            "depth_cy": projection.cy,
            "coordinate_frame": projection.frame,
            "orientation": orientation,
            "sample_stride": stride,
            "min_depth_m": min_depth_m,
            "max_depth_m": max_depth_m,
            "skipped_depth_count": skipped_depth_count,
            "clipped_depth_count": clipped_depth_count,
            "fusion_alignment": kinect.fusion_alignment,
            "rgbd_frame_id": kinect.rgbd_frame_id,
            "geometry_calibrated": kinect.geometry_calibration.is_some_and(|calibration| calibration.calibrated),
        }),
    })
}

#[derive(Clone, Debug)]
struct DepthColorImage {
    width: usize,
    height: usize,
    rgb: Vec<u8>,
}

impl DepthColorImage {
    fn from_eye_frame(frame: &EyeFrame) -> Option<Self> {
        let width = usize::try_from(frame.width).ok()?;
        let height = usize::try_from(frame.height).ok()?;
        if width == 0 || height == 0 {
            return None;
        }
        let rgb = eye_frame_to_rgb(frame)?;
        if rgb.len() < width.checked_mul(height)?.checked_mul(3)? {
            return None;
        }
        Some(Self { width, height, rgb })
    }

    fn sample_depth_pixel(
        &self,
        depth_x: usize,
        depth_y: usize,
        depth_width: usize,
        depth_height: usize,
    ) -> Option<[u8; 3]> {
        if depth_width == 0 || depth_height == 0 {
            return None;
        }
        let color_x = (depth_x.saturating_mul(self.width) / depth_width).min(self.width - 1);
        let color_y = (depth_y.saturating_mul(self.height) / depth_height).min(self.height - 1);
        self.sample(color_x, color_y)
    }

    fn sample(&self, x: usize, y: usize) -> Option<[u8; 3]> {
        let offset = y.checked_mul(self.width)?.checked_add(x)?.checked_mul(3)?;
        Some([
            *self.rgb.get(offset)?,
            *self.rgb.get(offset + 1)?,
            *self.rgb.get(offset + 2)?,
        ])
    }

    fn sample_projected(&self, pixel: [f32; 2]) -> Option<[u8; 3]> {
        if !(pixel[0].is_finite() && pixel[1].is_finite()) {
            return None;
        }
        let x = pixel[0].round() as usize;
        let y = pixel[1].round() as usize;
        (x < self.width && y < self.height)
            .then(|| self.sample(x, y))
            .flatten()
    }
}

fn eye_frame_to_rgb(frame: &EyeFrame) -> Option<Vec<u8>> {
    let pixels = usize::try_from(frame.width)
        .ok()?
        .checked_mul(usize::try_from(frame.height).ok()?)?;
    match frame.format {
        EyeFrameFormat::Rgb8 => {
            (frame.bytes.len() >= pixels.checked_mul(3)?).then(|| frame.bytes.clone())
        }
        EyeFrameFormat::Bgr8 => {
            if frame.bytes.len() < pixels.checked_mul(3)? {
                return None;
            }
            let mut rgb = Vec::with_capacity(pixels * 3);
            for pixel in frame.bytes.chunks_exact(3).take(pixels) {
                rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
            }
            Some(rgb)
        }
        EyeFrameFormat::Gray8 => {
            if frame.bytes.len() < pixels {
                return None;
            }
            let mut rgb = Vec::with_capacity(pixels * 3);
            for value in frame.bytes.iter().take(pixels) {
                rgb.extend_from_slice(&[*value, *value, *value]);
            }
            Some(rgb)
        }
        _ => None,
    }
}
