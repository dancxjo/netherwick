use pete_actions::{action_to_motor_command, ActionPrimitive};
use pete_core::Pose2;
use pete_now::KinectSense;
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
    pub depth_camera_roll_rad: f32,
    pub depth_camera_yaw_rad: f32,
    pub voxel_size_m: f32,
    pub temporal_frames: usize,
    pub outlier_radius_m: f32,
    pub outlier_min_neighbors: usize,
    pub plane_distance_threshold_m: f32,
    pub min_plane_points: usize,
    pub min_plane_major_extent_m: f32,
    pub min_plane_minor_extent_m: f32,
    pub min_plane_area_m: f32,
    pub max_plane_rms_error_m: f32,
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
    pub compact_depth_beam_count: usize,
    pub compact_depth_fov_rad: f32,
    pub depth_scale: f32,
}

impl Default for SurfaceExtractorConfig {
    fn default() -> Self {
        Self {
            min_depth_m: 0.35,
            max_depth_m: 8.0,
            depth_camera_height_m: 0.18,
            depth_camera_forward_offset_m: 0.0,
            depth_camera_pitch_down_rad: 0.0,
            depth_camera_roll_rad: 0.0,
            depth_camera_yaw_rad: 0.0,
            voxel_size_m: 0.04,
            temporal_frames: 4,
            outlier_radius_m: 0.14,
            outlier_min_neighbors: 1,
            plane_distance_threshold_m: 0.05,
            min_plane_points: 18,
            min_plane_major_extent_m: 0.28,
            min_plane_minor_extent_m: 0.08,
            min_plane_area_m: 0.045,
            max_plane_rms_error_m: 0.035,
            max_planes: 6,
            track_normal_max_angle_rad: 12.0_f32.to_radians(),
            track_distance_threshold_m: 0.12,
            track_centroid_threshold_m: 0.45,
            track_seen_gain: 0.18,
            track_missing_decay: 0.08,
            track_smoothing_alpha: 0.2,
            cluster_distance_m: 0.18,
            min_cluster_points: 3,
            cluster_track_match_threshold_m: 0.4,
            cluster_moving_speed_m_s: 0.08,
            cluster_seen_gain: 0.18,
            cluster_missing_decay: 0.12,
            occupancy_resolution_m: 0.075,
            occupancy_half_extent_m: 3.0,
            obstacle_min_height_m: 0.08,
            obstacle_max_height_m: 1.8,
            compact_depth_beam_count: 0,
            compact_depth_fov_rad: 0.0,
            depth_scale: 1.0,
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
    #[serde(default)]
    pub extent_m: Vec3,
    pub point_count: usize,
    pub confidence: f32,
    #[serde(default)]
    pub rms_error_m: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfacePrimitiveKind {
    #[default]
    Plane,
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
    #[serde(default)]
    pub primitive_kind: SurfacePrimitiveKind,
    pub kind: SurfaceKind,
    pub normal: Vec3,
    pub centroid: Vec3,
    pub distance_from_origin_m: f32,
    pub bounds_2d: Bounds2,
    #[serde(default)]
    pub extent_m: Vec3,
    pub confidence: f32,
    #[serde(default)]
    pub supporting_point_count: usize,
    pub first_seen_ms: u64,
    pub last_seen_ms: u64,
    pub seen_count: u32,
    pub missing_count: u32,
    #[serde(default)]
    pub labels: Vec<String>,
}

pub type SurfaceHypothesis = SurfaceTrack;

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
    pub surfaces: Vec<SurfaceHypothesis>,
    pub clusters: Vec<ClusterObservation>,
    pub navigation: serde_json::Value,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SurfaceExtractorOutput {
    pub raw_cloud: Vec<Point3>,
    pub filtered_cloud: Vec<Point3>,
    pub plane_observations: Vec<PlaneObservation>,
    pub stable_surfaces: Vec<SurfaceHypothesis>,
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

    pub fn reset_tracking(&mut self) {
        self.temporal_clouds.clear();
        self.tracks.clear();
        self.cluster_tracks.clear();
        self.next_surface_id = 1;
        self.next_cluster_id = 1;
    }

    pub fn set_depth_camera_extrinsics(
        &mut self,
        height_m: f32,
        forward_offset_m: f32,
        pitch_rad: f32,
        roll_rad: f32,
        yaw_rad: f32,
    ) {
        if surface_camera_extrinsics_changed(
            &self.config,
            height_m,
            forward_offset_m,
            pitch_rad,
            roll_rad,
            yaw_rad,
        ) {
            self.reset_tracking();
        }
        self.config.depth_camera_height_m = height_m;
        self.config.depth_camera_forward_offset_m = forward_offset_m;
        self.config.depth_camera_pitch_down_rad = pitch_rad;
        self.config.depth_camera_roll_rad = roll_rad;
        self.config.depth_camera_yaw_rad = yaw_rad;
    }

    pub fn set_compact_depth_calibration(
        &mut self,
        beam_count: usize,
        fov_rad: f32,
        depth_scale: f32,
    ) {
        if surface_compact_depth_calibration_changed(&self.config, beam_count, fov_rad, depth_scale)
        {
            self.reset_tracking();
        }
        self.config.compact_depth_beam_count = beam_count;
        self.config.compact_depth_fov_rad = fov_rad;
        self.config.depth_scale = depth_scale;
    }

    pub fn process(
        &mut self,
        kinect: &KinectSense,
        robot_pose: Pose2,
        t_ms: u64,
    ) -> SurfaceExtractorOutput {
        self.process_with_orientation(kinect, robot_pose, None, None, t_ms)
    }

    pub fn process_with_orientation(
        &mut self,
        kinect: &KinectSense,
        robot_pose: Pose2,
        roll_rad: Option<f32>,
        pitch_rad: Option<f32>,
        t_ms: u64,
    ) -> SurfaceExtractorOutput {
        let raw_cloud =
            depth_to_world_points(kinect, robot_pose, roll_rad, pitch_rad, &self.config);
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
                    primitive_kind: SurfacePrimitiveKind::Plane,
                    kind,
                    normal: observation.normal,
                    centroid: observation.centroid,
                    distance_from_origin_m: observation.distance_from_origin_m,
                    bounds_2d: observation.bounds_2d,
                    extent_m: observation.extent_m,
                    confidence: observation.confidence.clamp(0.15, 0.55),
                    supporting_point_count: observation.point_count,
                    first_seen_ms: t_ms,
                    last_seen_ms: t_ms,
                    seen_count: 1,
                    missing_count: 0,
                    labels: surface_labels(kind),
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
        let object_clusters = clusters
            .into_iter()
            .filter(|cluster| !cluster_is_planar_room_geometry(cluster, surfaces))
            .collect();
        self.update_cluster_tracks(object_clusters, surfaces, t_ms)
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
