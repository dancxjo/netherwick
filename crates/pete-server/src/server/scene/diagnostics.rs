fn scene_point_from_robot(robot: [f32; 3], r: u8, g: u8, b: u8) -> ScenePoint {
    ScenePoint {
        x: robot[1],
        y: robot[2],
        z: robot[0],
        r,
        g,
        b,
    }
}

fn depth_stats(
    depth_m: &[f32],
    sample_stride: usize,
    min_depth_m: f32,
    max_depth_m: f32,
    coordinate_system: &str,
    depth_width: u32,
    depth_height: u32,
) -> SceneKinectDiagnostics {
    let mut valid = Vec::new();
    let mut skipped = 0usize;
    let mut clipped = 0usize;
    for depth in depth_m {
        if !depth.is_finite() || *depth <= 0.0 {
            skipped = skipped.saturating_add(1);
        } else if *depth < min_depth_m || *depth > max_depth_m {
            clipped = clipped.saturating_add(1);
        } else {
            valid.push(*depth);
        }
    }
    valid.sort_by(|left, right| left.total_cmp(right));
    let median_depth_m = if valid.is_empty() {
        None
    } else {
        Some(valid[valid.len() / 2])
    };
    SceneKinectDiagnostics {
        depth_width,
        depth_height,
        valid_depth_count: valid.len(),
        skipped_depth_count: skipped,
        clipped_depth_count: clipped,
        min_depth_m: valid.first().copied(),
        median_depth_m,
        max_depth_m: valid.last().copied(),
        sample_stride,
        coordinate_system: coordinate_system.to_string(),
        point_coordinate_system: None,
        math_frame: None,
        render_frame: None,
        below_floor_count: 0,
        below_floor_ratio: 0.0,
        min_z_m: None,
        median_z_m: None,
        min_math_z_m: None,
        median_math_z_m: None,
        min_render_vertical_m: None,
        median_render_vertical_m: None,
        warnings: Vec::new(),
    }
}

impl SceneKinectDiagnostics {
    fn with_floor_stats(mut self, stats: FloorPointStats) -> Self {
        self.below_floor_count = stats.below_floor_count;
        self.below_floor_ratio = stats.below_floor_ratio;
        self.min_z_m = stats.min_z_m;
        self.median_z_m = stats.median_z_m;
        self.min_render_vertical_m = stats.min_z_m;
        self.median_render_vertical_m = stats.median_z_m;
        if self.coordinate_system == "scene_robot_render" {
            self.min_math_z_m = stats.min_z_m;
            self.median_math_z_m = stats.median_z_m;
        }
        self
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FloorPointStats {
    below_floor_count: usize,
    below_floor_ratio: f32,
    min_z_m: Option<f32>,
    median_z_m: Option<f32>,
}

fn floor_stats_from_scene_points(points: &[ScenePoint]) -> FloorPointStats {
    let mut heights = points
        .iter()
        .map(|point| point.y)
        .filter(|height| height.is_finite())
        .collect::<Vec<_>>();
    if heights.is_empty() {
        return FloorPointStats::default();
    }
    heights.sort_by(|left, right| left.total_cmp(right));
    let below_floor_count = heights.iter().filter(|height| **height < 0.0).count();
    FloorPointStats {
        below_floor_count,
        below_floor_ratio: below_floor_count as f32 / heights.len() as f32,
        min_z_m: heights.first().copied(),
        median_z_m: heights.get(heights.len() / 2).copied(),
    }
}

#[derive(Clone, Copy, Debug)]
struct DepthExtrinsics {
    forward_m: f32,
    height_m: f32,
    pitch_rad: f32,
    roll_rad: f32,
    yaw_rad: f32,
}

impl From<SceneSensorCalibration> for DepthExtrinsics {
    fn from(calibration: SceneSensorCalibration) -> Self {
        Self {
            forward_m: calibration.depth_camera_forward_m(),
            height_m: calibration.depth_camera_height_m(),
            pitch_rad: calibration.depth_camera_pitch_rad(),
            roll_rad: calibration.camera_roll_rad,
            yaw_rad: calibration.camera_yaw_rad,
        }
    }
}

fn camera_point_to_robot(camera: [f32; 3], extrinsics: DepthExtrinsics) -> [f32; 3] {
    let base = [camera[2], -camera[0], -camera[1]];
    apply_robot_extrinsics(base, extrinsics)
}

fn apply_robot_extrinsics(base: [f32; 3], extrinsics: DepthExtrinsics) -> [f32; 3] {
    let rotated = rotate_robot_extrinsic(
        base,
        extrinsics.pitch_rad,
        extrinsics.roll_rad,
        extrinsics.yaw_rad,
    );
    [
        rotated[0] + extrinsics.forward_m,
        rotated[1],
        rotated[2] + extrinsics.height_m,
    ]
}

fn rotate_robot_extrinsic(
    point: [f32; 3],
    pitch_rad: f32,
    roll_rad: f32,
    yaw_rad: f32,
) -> [f32; 3] {
    let (pitch_sin, pitch_cos) = pitch_rad.sin_cos();
    let mut x = point[0] * pitch_cos + point[2] * pitch_sin;
    let y = point[1];
    let mut z = -point[0] * pitch_sin + point[2] * pitch_cos;

    let (roll_sin, roll_cos) = roll_rad.sin_cos();
    let rolled_y = y * roll_cos - z * roll_sin;
    z = y * roll_sin + z * roll_cos;

    let (yaw_sin, yaw_cos) = yaw_rad.sin_cos();
    let yawed_x = x * yaw_cos - rolled_y * yaw_sin;
    let yawed_y = x * yaw_sin + rolled_y * yaw_cos;
    x = yawed_x;

    [x, yawed_y, z]
}

fn range_beam_points(depth_m: &[f32], calibration: SceneSensorCalibration) -> Vec<ScenePoint> {
    let beam_count = depth_m.len().max(1);
    let extrinsics = DepthExtrinsics::from(calibration);
    let fov_rad = calibration
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
            let distance = (depth * calibration.depth_scale).clamp(0.0, 8.0);
            let angle = start + step * index as f32;
            let shade = ((1.0 - (distance / 8.0)).clamp(0.15, 1.0) * 255.0) as u8;
            let robot = apply_robot_extrinsics(
                [angle.cos() * distance, angle.sin() * distance, 0.0],
                extrinsics,
            );
            Some(scene_point_from_robot(robot, shade, shade, shade))
        })
        .collect()
}

fn audio_bearing_from_objects(
    robot_x_m: f32,
    robot_y_m: f32,
    metadata: Option<&LiveSceneMetadata>,
) -> Option<f32> {
    metadata
        .into_iter()
        .flat_map(|metadata| metadata.objects.iter())
        .find(|object| {
            object.kind == "person" || object.kind == "speaker" || object.kind == "sound_source"
        })
        .map(|object| (object.y_m - robot_y_m).atan2(object.x_m - robot_x_m))
}

fn pcm_audio_energy(frame: &pete_sensors::PcmAudioFrame) -> f32 {
    if frame.samples.is_empty() {
        return 0.0;
    }
    let mean_square = frame
        .samples
        .iter()
        .map(|sample| {
            let normalized = *sample as f32 / i16::MAX as f32;
            normalized * normalized
        })
        .sum::<f32>()
        / frame.samples.len() as f32;
    mean_square.sqrt().clamp(0.0, 1.0)
}

fn scene_object_kind(debug_kind: &str) -> String {
    let lower = debug_kind.to_ascii_lowercase();
    if lower.contains("charger") {
        "charger"
    } else if lower.contains("person") {
        "person"
    } else if lower.contains("sound") || lower.contains("speaker") {
        "speaker"
    } else if lower.contains("landmark") {
        "landmark"
    } else {
        "obstacle"
    }
    .to_string()
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}
