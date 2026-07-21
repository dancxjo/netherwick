fn scene_kinect_from_snapshot(
    snapshot: &WorldSnapshot,
    calibration: Option<SceneSensorCalibration>,
    warnings: &mut Vec<String>,
) -> SceneKinect {
    let paired_color = snapshot.kinect.color_frame.as_ref().filter(|frame| {
        frame.rgbd_frame_id.is_some()
            && frame.rgbd_frame_id == snapshot.kinect.rgbd_frame_id
            && frame
                .captured_at_ms
                .abs_diff(snapshot.kinect.captured_at_ms)
                <= 50
    });
    let color = paired_color
        .or_else(|| {
            (snapshot.kinect.schema_version < 2)
                .then_some(snapshot.eye_frame.as_ref())
                .flatten()
        })
        .and_then(DepthColorImage::from_eye_frame);
    let (points, diagnostics) = depth_points(&snapshot.kinect, calibration, color.as_ref());
    if points.is_empty() {
        warnings.push("no point cloud stream".to_string());
    }
    if diagnostics.coordinate_system == "depth_image_unknown" {
        warnings.push(
            "Kinect depth frame has no width/height metadata; using legacy approximate projection"
                .to_string(),
        );
    }
    if calibration.is_none() && snapshot.kinect.depth_width > 0 && snapshot.kinect.depth_height > 0
    {
        warnings.push(
            "Kinect depth image has no explicit calibration; accumulated world cloud is disabled"
                .to_string(),
        );
    }
    SceneKinect {
        points,
        accumulated_points: Vec::new(),
        accumulated_summary: None,
        local_world_belief: None,
        skeletons: snapshot.kinect.skeletons.clone(),
        coordinate_system: Some(diagnostics.coordinate_system.clone()),
        diagnostics,
    }
}

fn depth_points(
    kinect: &KinectSense,
    calibration: Option<SceneSensorCalibration>,
    color: Option<&DepthColorImage>,
) -> (Vec<ScenePoint>, SceneKinectDiagnostics) {
    const MAX_POINTS: usize = 12_000;
    let depth_m = &kinect.depth_m;
    if depth_m.is_empty() {
        return (
            Vec::new(),
            SceneKinectDiagnostics {
                coordinate_system: "none".to_string(),
                sample_stride: 1,
                ..SceneKinectDiagnostics::default()
            },
        );
    }
    if let Some(calibration) = calibration {
        if depth_m.len() == calibration.compact_depth_beam_count {
            let points = range_beam_points(depth_m, calibration);
            let mut stats = depth_stats(depth_m, 1, 0.0, 8.0, "scene_robot_render", 0, 0)
                .with_floor_stats(floor_stats_from_scene_points(&points));
            stats.point_coordinate_system =
                Some("scene axes derived from robot math frame".to_string());
            stats.math_frame =
                Some("robot/base: +x forward, +y left, +z up; floor z=0".to_string());
            stats.render_frame = Some("scene: +x left, +y up, +z forward".to_string());
            return (points, stats);
        }
    }
    if kinect.geometry_calibration.is_some() {
        return project_calibrated_depth_image(kinect, color);
    }
    if let Some(frame) = KinectDepthProjection::from_kinect(kinect) {
        return project_depth_image(depth_m, frame, calibration, color);
    }
    if depth_m.len() == 640 * 480 {
        return project_depth_image(
            depth_m,
            KinectDepthProjection {
                width: 640,
                height: 480,
                fx: 594.0,
                fy: 591.0,
                cx: 339.0,
                cy: 242.0,
                min_depth_m: 0.4,
                max_depth_m: 8.0,
                coordinate_system: "kinect_camera".to_string(),
            },
            calibration,
            color,
        );
    }
    let width = (depth_m.len() as f32).sqrt().ceil().max(1.0) as usize;
    let height = depth_m.len().div_ceil(width).max(1);
    let stride = (depth_m.len().div_ceil(MAX_POINTS)).max(1);
    let points = depth_m
        .iter()
        .enumerate()
        .step_by(stride)
        .filter_map(|(index, depth)| {
            if !depth.is_finite() || *depth <= 0.0 {
                return None;
            }
            let x = index % width;
            let y = index / width;
            let nx = (x as f32 / width as f32) - 0.5;
            let ny = (y as f32 / height as f32) - 0.5;
            let z = (depth * calibration.map_or(1.0, |calibration| calibration.depth_scale))
                .clamp(0.0, 8.0);
            let [r, g, b] = color
                .and_then(|color| {
                    let (offset_x, offset_y) = calibration
                        .map(SceneSensorCalibration::color_offset_px)
                        .unwrap_or_default();
                    color.sample_depth_pixel_with_offset(x, y, width, height, offset_x, offset_y)
                })
                .unwrap_or_else(|| depth_shade(z, 8.0));
            Some(ScenePoint {
                x: nx * z,
                y: ny * z,
                z,
                r,
                g,
                b,
            })
        })
        .collect();
    (
        points,
        depth_stats(
            depth_m,
            stride,
            0.0,
            8.0,
            "depth_image_unknown",
            width as u32,
            height as u32,
        ),
    )
}

fn project_calibrated_depth_image(
    kinect: &KinectSense,
    color: Option<&DepthColorImage>,
) -> (Vec<ScenePoint>, SceneKinectDiagnostics) {
    const MAX_POINTS: usize = 2_000;
    let Some(geometry) = pete_now::DepthGeometry::from_kinect(kinect) else {
        return (Vec::new(), SceneKinectDiagnostics::default());
    };
    let width = geometry.calibration.depth.width as usize;
    let height = geometry.calibration.depth.height as usize;
    let stride = kinect.depth_m.len().div_ceil(MAX_POINTS).max(1);
    let mut points = Vec::new();
    for (index, depth) in kinect.depth_m.iter().enumerate().step_by(stride) {
        if !depth.is_finite() || *depth <= 0.0 {
            continue;
        }
        let u = (index % width) as f32;
        let v = (index / width) as f32;
        let Some(camera) = geometry.depth_pixel_to_camera(u, v, *depth) else {
            continue;
        };
        let base = geometry.depth_point_to_base(camera);
        let [r, g, b] = color
            .and_then(|color| {
                geometry.depth_point_to_rgb_pixel(camera).and_then(|pixel| {
                    color.sample(pixel[0].round() as usize, pixel[1].round() as usize)
                })
            })
            .unwrap_or_else(|| depth_shade(camera[2], kinect.max_depth_m.max(8.0)));
        points.push(scene_point_from_robot(base, r, g, b));
    }
    let mut diagnostics = depth_stats(
        &kinect.depth_m,
        stride,
        kinect.min_depth_m,
        kinect.max_depth_m,
        "scene_robot_render",
        width as u32,
        height as u32,
    )
    .with_floor_stats(floor_stats_from_scene_points(&points));
    diagnostics.point_coordinate_system =
        Some("calibrated depth optical to robot base".to_string());
    diagnostics.math_frame = Some("robot/base: +x forward, +y left, +z up; floor z=0".to_string());
    diagnostics.render_frame = Some("scene: +x left, +y up, +z forward".to_string());
    (points, diagnostics)
}

#[derive(Clone, Debug)]
struct KinectDepthProjection {
    width: usize,
    height: usize,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    min_depth_m: f32,
    max_depth_m: f32,
    coordinate_system: String,
}

impl KinectDepthProjection {
    fn from_kinect(kinect: &KinectSense) -> Option<Self> {
        let width = usize::try_from(kinect.depth_width).ok()?;
        let height = usize::try_from(kinect.depth_height).ok()?;
        if width == 0 || height == 0 || width.checked_mul(height)? != kinect.depth_m.len() {
            return None;
        }
        let fx = if kinect.depth_fx > 0.0 {
            kinect.depth_fx
        } else {
            594.0
        };
        let fy = if kinect.depth_fy > 0.0 {
            kinect.depth_fy
        } else {
            591.0
        };
        let cx = if kinect.depth_cx > 0.0 {
            kinect.depth_cx
        } else {
            (width as f32 - 1.0) * 0.5
        };
        let cy = if kinect.depth_cy > 0.0 {
            kinect.depth_cy
        } else {
            (height as f32 - 1.0) * 0.5
        };
        let max_depth_m = if kinect.max_depth_m > 0.0 {
            kinect.max_depth_m
        } else {
            8.0
        };
        Some(Self {
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            min_depth_m: kinect.min_depth_m.max(0.0),
            max_depth_m,
            coordinate_system: kinect
                .depth_coordinate_system
                .clone()
                .filter(|system| system != "kinect_depth_image")
                .unwrap_or_else(|| "kinect_camera".to_string()),
        })
    }
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
        let rgb = eye_frame_to_rgb(frame).ok()?;
        if rgb.len() < width.checked_mul(height)?.checked_mul(3)? {
            return None;
        }
        Some(Self { width, height, rgb })
    }

    fn sample_depth_pixel_with_offset(
        &self,
        depth_x: usize,
        depth_y: usize,
        depth_width: usize,
        depth_height: usize,
        offset_x_px: i32,
        offset_y_px: i32,
    ) -> Option<[u8; 3]> {
        if depth_width == 0 || depth_height == 0 {
            return None;
        }
        let color_x = (depth_x.saturating_mul(self.width) / depth_width).min(self.width - 1);
        let color_y = (depth_y.saturating_mul(self.height) / depth_height).min(self.height - 1);
        self.sample_offset(color_x, color_y, offset_x_px, offset_y_px)
    }

    fn sample_offset(
        &self,
        x: usize,
        y: usize,
        offset_x_px: i32,
        offset_y_px: i32,
    ) -> Option<[u8; 3]> {
        let x = offset_index(x, offset_x_px, self.width)?;
        let y = offset_index(y, offset_y_px, self.height)?;
        self.sample(x, y)
    }

    fn sample(&self, x: usize, y: usize) -> Option<[u8; 3]> {
        let offset = y.checked_mul(self.width)?.checked_add(x)?.checked_mul(3)?;
        Some([
            *self.rgb.get(offset)?,
            *self.rgb.get(offset + 1)?,
            *self.rgb.get(offset + 2)?,
        ])
    }
}

fn offset_index(index: usize, offset: i32, len: usize) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let shifted = index as i64 + offset as i64;
    Some(shifted.clamp(0, len as i64 - 1) as usize)
}

fn depth_shade(depth_m: f32, max_depth_m: f32) -> [u8; 3] {
    let shade = ((1.0 - (depth_m / max_depth_m.max(f32::EPSILON))).clamp(0.15, 1.0) * 255.0) as u8;
    [shade, shade, shade]
}

fn project_depth_image(
    depth_m: &[f32],
    frame: KinectDepthProjection,
    calibration: Option<SceneSensorCalibration>,
    color: Option<&DepthColorImage>,
) -> (Vec<ScenePoint>, SceneKinectDiagnostics) {
    const MAX_POINTS: usize = 2_000;
    let stride = (depth_m.len().div_ceil(MAX_POINTS)).max(1);
    let mut points = Vec::with_capacity(MAX_POINTS.min(depth_m.len()));
    let calibrated = calibration.map(|calibration| DepthExtrinsics::from(calibration));
    let (color_offset_x_px, color_offset_y_px) = calibration
        .map(SceneSensorCalibration::color_offset_px)
        .unwrap_or_default();
    for (index, depth) in depth_m.iter().enumerate().step_by(stride) {
        if !depth.is_finite() || *depth <= 0.0 {
            continue;
        }
        if *depth < frame.min_depth_m || *depth > frame.max_depth_m {
            continue;
        }
        let u = (index % frame.width) as f32;
        let v = (index / frame.width) as f32;
        let z = *depth;
        let x = (u - frame.cx) * z / frame.fx.max(f32::EPSILON);
        let y = (v - frame.cy) * z / frame.fy.max(f32::EPSILON);
        let [r, g, b] = color
            .and_then(|color| {
                color.sample_depth_pixel_with_offset(
                    index % frame.width,
                    index / frame.width,
                    frame.width,
                    frame.height,
                    color_offset_x_px,
                    color_offset_y_px,
                )
            })
            .unwrap_or_else(|| depth_shade(z, frame.max_depth_m));
        let point = if let Some(extrinsics) = calibrated {
            let robot = camera_point_to_robot([x, y, z], extrinsics);
            scene_point_from_robot(robot, r, g, b)
        } else {
            ScenePoint { x, y, z, r, g, b }
        };
        points.push(point);
    }
    let coordinate_system = if calibrated.is_some() {
        "scene_robot_render"
    } else {
        &frame.coordinate_system
    };
    let mut diagnostics = depth_stats(
        depth_m,
        stride,
        frame.min_depth_m,
        frame.max_depth_m,
        coordinate_system,
        frame.width as u32,
        frame.height as u32,
    )
    .with_floor_stats(floor_stats_from_scene_points(&points));
    if calibrated.is_some() {
        diagnostics.point_coordinate_system =
            Some("scene axes derived from robot math frame".to_string());
        diagnostics.math_frame =
            Some("robot/base: +x forward, +y left, +z up; floor z=0".to_string());
        diagnostics.render_frame = Some("scene: +x left, +y up, +z forward".to_string());
    }
    (points, diagnostics)
}
