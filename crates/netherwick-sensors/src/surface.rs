use netherwick_actions::{action_to_motor_command, ActionPrimitive};
use netherwick_core::Pose2;
use netherwick_now::KinectSense;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    fn normalized(self) -> Option<Self> {
        let length = self.length();
        if length <= f32::EPSILON || !length.is_finite() {
            None
        } else {
            Some(self / length)
        }
    }

    fn distance(self, other: Self) -> f32 {
        (self - other).length()
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Point3 {
    pub position: Vec3,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SurfaceExtractorConfig {
    pub min_depth_m: f32,
    pub max_depth_m: f32,
    pub depth_camera_height_m: f32,
    pub depth_camera_forward_offset_m: f32,
    pub depth_camera_pitch_down_rad: f32,
    pub voxel_size_m: f32,
    pub temporal_frames: usize,
    pub outlier_radius_m: f32,
    pub outlier_min_neighbors: usize,
    pub plane_distance_threshold_m: f32,
    pub min_plane_points: usize,
    pub max_planes: usize,
    pub track_normal_max_angle_rad: f32,
    pub track_distance_threshold_m: f32,
    pub track_centroid_threshold_m: f32,
    pub track_seen_gain: f32,
    pub track_missing_decay: f32,
    pub track_smoothing_alpha: f32,
    pub cluster_distance_m: f32,
    pub min_cluster_points: usize,
    pub cluster_track_match_threshold_m: f32,
    pub cluster_moving_speed_m_s: f32,
    pub cluster_seen_gain: f32,
    pub cluster_missing_decay: f32,
    pub occupancy_resolution_m: f32,
    pub occupancy_half_extent_m: f32,
    pub obstacle_min_height_m: f32,
    pub obstacle_max_height_m: f32,
}

impl Default for SurfaceExtractorConfig {
    fn default() -> Self {
        Self {
            min_depth_m: 0.35,
            max_depth_m: 8.0,
            depth_camera_height_m: 0.18,
            depth_camera_forward_offset_m: 0.0,
            depth_camera_pitch_down_rad: 0.0,
            voxel_size_m: 0.075,
            temporal_frames: 4,
            outlier_radius_m: 0.18,
            outlier_min_neighbors: 2,
            plane_distance_threshold_m: 0.05,
            min_plane_points: 24,
            max_planes: 6,
            track_normal_max_angle_rad: 12.0_f32.to_radians(),
            track_distance_threshold_m: 0.12,
            track_centroid_threshold_m: 0.45,
            track_seen_gain: 0.18,
            track_missing_decay: 0.08,
            track_smoothing_alpha: 0.2,
            cluster_distance_m: 0.22,
            min_cluster_points: 4,
            cluster_track_match_threshold_m: 0.4,
            cluster_moving_speed_m_s: 0.08,
            cluster_seen_gain: 0.18,
            cluster_missing_decay: 0.12,
            occupancy_resolution_m: 0.1,
            occupancy_half_extent_m: 3.0,
            obstacle_min_height_m: 0.08,
            obstacle_max_height_m: 1.8,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Bounds2 {
    pub min_u: f32,
    pub max_u: f32,
    pub min_v: f32,
    pub max_v: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlaneObservation {
    pub normal: Vec3,
    pub centroid: Vec3,
    pub distance_from_origin_m: f32,
    pub bounds_2d: Bounds2,
    pub point_count: usize,
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceKind {
    Floor,
    HorizontalPlane,
    VerticalPlane,
    #[default]
    UnknownPlane,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceTrack {
    pub id: String,
    pub kind: SurfaceKind,
    pub normal: Vec3,
    pub centroid: Vec3,
    pub distance_from_origin_m: f32,
    pub bounds_2d: Bounds2,
    pub confidence: f32,
    pub first_seen_ms: u64,
    pub last_seen_ms: u64,
    pub seen_count: u32,
    pub missing_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OccupancyState {
    Free,
    Occupied,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OccupancyCell {
    pub x: i32,
    pub y: i32,
    pub state: OccupancyState,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OccupancyGrid {
    pub resolution_m: f32,
    pub half_extent_m: f32,
    pub cells: Vec<OccupancyCell>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClusterObservation {
    pub id: String,
    pub centroid: Vec3,
    pub size_m: Vec3,
    pub point_count: usize,
    #[serde(default)]
    pub confidence: f32,
    #[serde(default)]
    pub moving: bool,
    #[serde(default)]
    pub velocity_m_s: Vec3,
    #[serde(default)]
    pub last_seen_ms: u64,
    #[serde(default)]
    pub seen_count: u32,
    pub above_surface_id: Option<String>,
    #[serde(default)]
    pub semantic_hint: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SurfaceExtractorDiagnostics {
    pub raw_points: usize,
    pub downsampled_points: usize,
    pub smoothed_points: usize,
    pub filtered_points: usize,
    pub plane_points: usize,
    pub leftover_points: usize,
    pub calibration_hint: Option<SurfaceCalibrationHint>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SurfaceCalibrationHint {
    pub floor_confidence: f32,
    pub floor_height_error_m: f32,
    pub floor_tilt_rad: f32,
    pub floor_pitch_error_rad: f32,
    pub floor_roll_error_rad: f32,
    pub suggested_depth_height_m: f32,
    pub suggested_depth_pitch_down_rad: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SceneGraphSummary {
    pub floor: Option<SurfaceTrack>,
    pub surfaces: Vec<SurfaceTrack>,
    pub clusters: Vec<ClusterObservation>,
    pub navigation: serde_json::Value,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SurfaceExtractorOutput {
    pub raw_cloud: Vec<Point3>,
    pub filtered_cloud: Vec<Point3>,
    pub plane_observations: Vec<PlaneObservation>,
    pub stable_surfaces: Vec<SurfaceTrack>,
    pub floor: Option<SurfaceTrack>,
    pub obstacle_grid: OccupancyGrid,
    pub clusters: Vec<ClusterObservation>,
    pub scene_graph: SceneGraphSummary,
    pub diagnostics: SurfaceExtractorDiagnostics,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AnticipatedNavigation {
    pub front_clear_m: Option<f32>,
    pub left_clear_m: Option<f32>,
    pub right_clear_m: Option<f32>,
    pub collision_risk: f32,
    pub occupied_cells: usize,
    pub free_cells: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectedSurface {
    pub id: String,
    pub kind: SurfaceKind,
    pub normal: Vec3,
    pub centroid: Vec3,
    pub bounds_2d: Bounds2,
    pub confidence: f32,
    pub observed_bounds_2d: Bounds2,
    pub extrapolated_bounds_2d: Bounds2,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectedCluster {
    pub id: String,
    pub centroid: Vec3,
    pub size_m: Vec3,
    pub confidence: f32,
    pub moving: bool,
    pub semantic_hint: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceAnticipationFrame {
    pub offset_ms: u64,
    pub expected_pose: Pose2,
    pub projected_surfaces: Vec<ProjectedSurface>,
    pub projected_clusters: Vec<ProjectedCluster>,
    pub projected_obstacle_grid: OccupancyGrid,
    pub navigation: AnticipatedNavigation,
}

#[derive(Clone, Debug)]
pub struct SurfaceExtractor {
    config: SurfaceExtractorConfig,
    temporal_clouds: VecDeque<Vec<Point3>>,
    tracks: Vec<SurfaceTrack>,
    cluster_tracks: Vec<ClusterTrack>,
    next_surface_id: u64,
    next_cluster_id: u64,
}

#[derive(Clone, Debug, PartialEq)]
struct ClusterTrack {
    id: String,
    centroid: Vec3,
    size_m: Vec3,
    confidence: f32,
    velocity_m_s: Vec3,
    last_seen_ms: u64,
    seen_count: u32,
    missing_count: u32,
    point_count: usize,
}

impl Default for SurfaceExtractor {
    fn default() -> Self {
        Self::new(SurfaceExtractorConfig::default())
    }
}

impl SurfaceExtractor {
    pub fn new(config: SurfaceExtractorConfig) -> Self {
        Self {
            config,
            temporal_clouds: VecDeque::new(),
            tracks: Vec::new(),
            cluster_tracks: Vec::new(),
            next_surface_id: 1,
            next_cluster_id: 1,
        }
    }

    pub fn set_depth_camera_extrinsics(
        &mut self,
        height_m: f32,
        forward_offset_m: f32,
        pitch_down_rad: f32,
    ) {
        self.config.depth_camera_height_m = height_m;
        self.config.depth_camera_forward_offset_m = forward_offset_m;
        self.config.depth_camera_pitch_down_rad = pitch_down_rad;
    }

    pub fn process(
        &mut self,
        kinect: &KinectSense,
        robot_pose: Pose2,
        t_ms: u64,
    ) -> SurfaceExtractorOutput {
        let raw_cloud = depth_to_world_points(kinect, robot_pose, &self.config);
        let downsampled = voxel_downsample(&raw_cloud, self.config.voxel_size_m);
        self.temporal_clouds.push_back(downsampled.clone());
        while self.temporal_clouds.len() > self.config.temporal_frames.max(1) {
            self.temporal_clouds.pop_front();
        }
        let smoothed = temporal_voxel_average(&self.temporal_clouds, self.config.voxel_size_m);
        let filtered = remove_outliers(
            &smoothed,
            self.config.outlier_radius_m,
            self.config.outlier_min_neighbors,
        );
        let (plane_observations, leftover_points) = extract_planes(&filtered, &self.config);
        let stable_surfaces = self.update_tracks(&plane_observations, t_ms);
        let floor = stable_surfaces
            .iter()
            .find(|surface| surface.kind == SurfaceKind::Floor)
            .cloned();
        let obstacle_grid = project_obstacles(&filtered, floor.as_ref(), robot_pose, &self.config);
        let clusters = self.cluster_leftovers(&leftover_points, &stable_surfaces, t_ms);
        let scene_graph = scene_graph(&stable_surfaces, floor.clone(), &clusters, &obstacle_grid);
        let diagnostics = SurfaceExtractorDiagnostics {
            raw_points: raw_cloud.len(),
            downsampled_points: downsampled.len(),
            smoothed_points: smoothed.len(),
            filtered_points: filtered.len(),
            plane_points: plane_observations
                .iter()
                .map(|plane| plane.point_count)
                .sum(),
            leftover_points: leftover_points.len(),
            calibration_hint: floor
                .as_ref()
                .map(|floor| calibration_hint(floor, robot_pose, &self.config)),
        };

        SurfaceExtractorOutput {
            raw_cloud,
            filtered_cloud: filtered,
            plane_observations,
            stable_surfaces,
            floor,
            obstacle_grid,
            clusters,
            scene_graph,
            diagnostics,
        }
    }

    fn update_tracks(&mut self, observations: &[PlaneObservation], t_ms: u64) -> Vec<SurfaceTrack> {
        let mut matched_tracks = vec![false; self.tracks.len()];

        for observation in observations {
            let kind = classify_surface(observation, observations);
            let match_index = self
                .tracks
                .iter()
                .enumerate()
                .filter(|(index, _)| !matched_tracks[*index])
                .filter_map(|(index, track)| {
                    track_match_score(track, observation, &self.config).map(|score| (index, score))
                })
                .min_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index);

            if let Some(index) = match_index {
                matched_tracks[index] = true;
                smooth_track(
                    &mut self.tracks[index],
                    observation,
                    kind,
                    self.config.track_smoothing_alpha,
                    self.config.track_seen_gain,
                    t_ms,
                );
            } else {
                let id = match kind {
                    SurfaceKind::Floor => "floor".to_string(),
                    SurfaceKind::VerticalPlane => format!("wall_{}", self.next_surface_id),
                    SurfaceKind::HorizontalPlane => format!("surface_{}", self.next_surface_id),
                    SurfaceKind::UnknownPlane => format!("plane_{}", self.next_surface_id),
                };
                self.next_surface_id += 1;
                self.tracks.push(SurfaceTrack {
                    id,
                    kind,
                    normal: observation.normal,
                    centroid: observation.centroid,
                    distance_from_origin_m: observation.distance_from_origin_m,
                    bounds_2d: observation.bounds_2d,
                    confidence: observation.confidence.clamp(0.15, 0.55),
                    first_seen_ms: t_ms,
                    last_seen_ms: t_ms,
                    seen_count: 1,
                    missing_count: 0,
                });
                matched_tracks.push(true);
            }
        }

        for (index, track) in self.tracks.iter_mut().enumerate() {
            if !matched_tracks.get(index).copied().unwrap_or(false) {
                track.confidence = (track.confidence - self.config.track_missing_decay).max(0.0);
                track.missing_count += 1;
            }
        }
        self.tracks.retain(|track| track.confidence > 0.02);
        self.tracks.clone()
    }

    fn cluster_leftovers(
        &mut self,
        points: &[Point3],
        surfaces: &[SurfaceTrack],
        t_ms: u64,
    ) -> Vec<ClusterObservation> {
        let clusters = euclidean_clusters(
            points,
            self.config.cluster_distance_m,
            self.config.min_cluster_points,
        );
        self.update_cluster_tracks(clusters, surfaces, t_ms)
    }

    fn update_cluster_tracks(
        &mut self,
        observations: Vec<ClusterObservation>,
        surfaces: &[SurfaceTrack],
        t_ms: u64,
    ) -> Vec<ClusterObservation> {
        let mut matched_tracks = vec![false; self.cluster_tracks.len()];
        let mut output = Vec::new();

        for mut observation in observations {
            let match_index = self
                .cluster_tracks
                .iter()
                .enumerate()
                .filter(|(index, _)| !matched_tracks[*index])
                .filter_map(|(index, track)| {
                    let distance = track.centroid.distance(observation.centroid);
                    if distance <= self.config.cluster_track_match_threshold_m {
                        Some((index, distance))
                    } else {
                        None
                    }
                })
                .min_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index);

            if let Some(index) = match_index {
                matched_tracks[index] = true;
                let track = &mut self.cluster_tracks[index];
                let dt_s = t_ms.saturating_sub(track.last_seen_ms).max(1) as f32 / 1000.0;
                let velocity = (observation.centroid - track.centroid) / dt_s;
                track.velocity_m_s = track.velocity_m_s * 0.6 + velocity * 0.4;
                track.centroid = track.centroid * 0.75 + observation.centroid * 0.25;
                track.size_m = track.size_m * 0.75 + observation.size_m * 0.25;
                track.confidence = (track.confidence + self.config.cluster_seen_gain).min(1.0);
                track.last_seen_ms = t_ms;
                track.seen_count += 1;
                track.missing_count = 0;
                track.point_count = observation.point_count;
                observation.id = track.id.clone();
                observation.centroid = track.centroid;
                observation.size_m = track.size_m;
                observation.confidence = track.confidence;
                observation.velocity_m_s = track.velocity_m_s;
                observation.last_seen_ms = track.last_seen_ms;
                observation.seen_count = track.seen_count;
            } else {
                observation.id = format!("cluster_{}", self.next_cluster_id);
                self.next_cluster_id += 1;
                observation.confidence = observation.confidence.max(0.35);
                observation.last_seen_ms = t_ms;
                observation.seen_count = 1;
                self.cluster_tracks.push(ClusterTrack {
                    id: observation.id.clone(),
                    centroid: observation.centroid,
                    size_m: observation.size_m,
                    confidence: observation.confidence,
                    velocity_m_s: Vec3::default(),
                    last_seen_ms: t_ms,
                    seen_count: 1,
                    missing_count: 0,
                    point_count: observation.point_count,
                });
                matched_tracks.push(true);
            }

            observation.moving =
                observation.velocity_m_s.length() >= self.config.cluster_moving_speed_m_s;
            observation.above_surface_id = surface_below_cluster(&observation, surfaces);
            observation.semantic_hint = semantic_hint_for_cluster(&observation);
            output.push(observation);
        }

        for (index, track) in self.cluster_tracks.iter_mut().enumerate() {
            if !matched_tracks.get(index).copied().unwrap_or(false) {
                track.confidence = (track.confidence - self.config.cluster_missing_decay).max(0.0);
                track.missing_count += 1;
            }
        }
        self.cluster_tracks.retain(|track| track.confidence > 0.04);
        output
    }
}

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

fn depth_to_world_points(
    kinect: &KinectSense,
    robot_pose: Pose2,
    config: &SurfaceExtractorConfig,
) -> Vec<Point3> {
    let Some(frame) = DepthProjection::from_kinect(kinect, config) else {
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
        let z = *depth;
        let camera = Vec3::new(
            (u - frame.cx) * z / frame.fx,
            (v - frame.cy) * z / frame.fy,
            z,
        );
        let robot = camera_to_robot(camera, config);
        points.push(Point3 {
            position: robot_to_world(robot, robot_pose),
        });
    }
    points
}

#[derive(Clone, Copy, Debug)]
struct DepthProjection {
    width: usize,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
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
            fx: positive_or(kinect.depth_fx, 594.0),
            fy: positive_or(kinect.depth_fy, 591.0),
            cx: positive_or(kinect.depth_cx, (width as f32 - 1.0) * 0.5),
            cy: positive_or(kinect.depth_cy, (height as f32 - 1.0) * 0.5),
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
    let base_x = camera.z;
    let base_y = -camera.x;
    let base_z = -camera.y;
    let (sin, cos) = config.depth_camera_pitch_down_rad.sin_cos();
    Vec3::new(
        base_x * cos - base_z * sin + config.depth_camera_forward_offset_m,
        base_y,
        -base_x * sin + base_z * cos + config.depth_camera_height_m,
    )
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
        planes.push(plane_observation(model, &inliers, points.len()));
        remaining = outliers;
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
        point_count: inliers.len(),
        confidence: (inliers.len() as f32 / total_points.max(1) as f32).clamp(0.0, 1.0),
    }
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
        Some(normal_angle + distance_delta + centroid_delta)
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
    track.confidence = (track.confidence + seen_gain + observation.confidence * 0.1).min(1.0);
    track.last_seen_ms = t_ms;
    track.seen_count += 1;
    track.missing_count = 0;
}

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

#[cfg(test)]
mod tests {
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
    fn floor_calibration_hint_reports_height_and_tilt() {
        let config = SurfaceExtractorConfig {
            depth_camera_height_m: 0.3,
            depth_camera_pitch_down_rad: 0.0,
            ..SurfaceExtractorConfig::default()
        };
        let floor = SurfaceTrack {
            id: "floor".to_string(),
            kind: SurfaceKind::Floor,
            normal: Vec3::new(0.1, 0.0, 0.995).normalized().unwrap(),
            centroid: Vec3::new(0.0, 0.0, 0.05),
            distance_from_origin_m: 0.0,
            bounds_2d: Bounds2::default(),
            confidence: 0.8,
            first_seen_ms: 0,
            last_seen_ms: 0,
            seen_count: 1,
            missing_count: 0,
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
            confidence: 1.0,
            first_seen_ms: 0,
            last_seen_ms: 0,
            seen_count: 1,
            missing_count: 0,
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
            frames[2].navigation.front_clear_m.unwrap()
                < frames[0].navigation.front_clear_m.unwrap()
        );
        assert!(frames[2].navigation.collision_risk > frames[0].navigation.collision_risk);
    }

    #[test]
    fn wall_left_turn_left_anticipation_becomes_risky() {
        let output = output_with_wall("wall_1", Vec3::new(0.0, 0.45, 0.6));
        let action = ActionPrimitive::Turn {
            direction: netherwick_actions::TurnDir::Left,
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
        let first = PlaneObservation {
            normal: Vec3::new(1.0, 0.0, 0.0),
            centroid: Vec3::new(0.8, -0.4, 0.6),
            distance_from_origin_m: -0.8,
            bounds_2d: Bounds2 {
                min_u: -0.25,
                max_u: 0.25,
                min_v: -0.4,
                max_v: 0.4,
            },
            point_count: 64,
            confidence: 0.7,
        };
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

    fn output_with_wall(id: &str, centroid: Vec3) -> SurfaceExtractorOutput {
        let normal = if centroid.x.abs() >= centroid.y.abs() {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let wall = SurfaceTrack {
            id: id.to_string(),
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
            confidence: 0.85,
            first_seen_ms: 1_000,
            last_seen_ms: 1_000,
            seen_count: 3,
            missing_count: 0,
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

    fn floor_track() -> SurfaceTrack {
        SurfaceTrack {
            id: "floor".to_string(),
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
            confidence: 1.0,
            first_seen_ms: 0,
            last_seen_ms: 0,
            seen_count: 1,
            missing_count: 0,
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
}
