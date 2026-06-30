use std::collections::BTreeMap;

use netherwick_core::{Pose2, TimeMs};
use netherwick_now::{ImuSense, KinectSense, Now};
use netherwick_sensors::WorldSnapshot;
use serde::{Deserialize, Serialize};

pub const MAP_EXTENSION_NAME: &str = "map.odometry";
pub const MAP_LABEL: &str = "scan-matched occupancy SLAM prototype";
pub const POSE_GRAPH_LABEL: &str = "offline pose graph with odometry and gated loop candidates";
pub const WORLD_POINT_CLOUD_LABEL: &str =
    "provisional odometry-frame Kinect/depth point cloud, not full SLAM";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoseEstimate {
    pub pose: Pose2,
    pub confidence: f32,
    pub covariance: [f32; 3],
    pub source: String,
    pub t_ms: TimeMs,
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct CellKey {
    pub x: i32,
    pub y: i32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OccupancyCell {
    pub key: CellKey,
    pub occupied_score: f32,
    pub free_score: f32,
    pub confidence: f32,
    pub last_seen_ms: TimeMs,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RangeBeam {
    pub angle_rad: f32,
    pub distance_m: f32,
    pub hit: bool,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MapObservation {
    pub pose: PoseEstimate,
    pub range_beams: Vec<RangeBeam>,
    pub source_snapshot: serde_json::Value,
    pub t_ms: TimeMs,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Point3D {
    pub x_m: f32,
    pub y_m: f32,
    pub z_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VoxelKey {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointCloudFrame {
    KinectCamera,
    RobotBase,
    OdometryWorld,
    DepthImageUnknown,
}

impl Default for PointCloudFrame {
    fn default() -> Self {
        Self::KinectCamera
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointCloudPoint {
    pub position: Point3D,
    pub color_rgb: Option<[u8; 3]>,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointCloudObservation {
    pub frame: PointCloudFrame,
    pub pose: PoseEstimate,
    #[serde(default)]
    pub orientation: OrientationEstimate,
    pub points: Vec<PointCloudPoint>,
    pub source: String,
    pub t_ms: TimeMs,
    pub metadata: serde_json::Value,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OrientationEstimate {
    pub roll_rad: Option<f32>,
    pub pitch_rad: Option<f32>,
    pub yaw_rad: Option<f32>,
    pub roll_pitch_from_imu: bool,
    pub yaw_source: YawSource,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum YawSource {
    #[default]
    OdometryHeading,
    ImuOrientation,
    Unavailable,
}

const MAX_TRUSTED_GRAVITY_TILT_RAD: f32 = std::f32::consts::FRAC_PI_4;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VoxelPoint {
    pub key: VoxelKey,
    pub position: Point3D,
    pub color_rgb: Option<[u8; 3]>,
    pub confidence: f32,
    pub first_seen_ms: TimeMs,
    pub last_seen_ms: TimeMs,
    pub seen_count: u32,
    pub stable: bool,
    pub transient: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VoxelPointCloud {
    pub voxels: BTreeMap<VoxelKey, VoxelPoint>,
    pub config: PointCloudConfig,
    pub observations: u64,
    pub raw_points_seen: u64,
    pub orientation_status: OrientationStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointCloudConfig {
    pub voxel_size_m: f32,
    pub max_voxels: usize,
    pub max_points_per_observation: usize,
    pub min_depth_m: f32,
    pub max_depth_m: f32,
    pub confidence_increment: f32,
    pub decay_after_ms: TimeMs,
    pub decay_per_tick: f32,
    pub stable_seen_count: u32,
    pub stable_confidence: f32,
    pub transient_after_ms: TimeMs,
    pub camera_height_m: f32,
    pub camera_forward_m: f32,
    #[serde(default)]
    pub camera_pitch_rad: f32,
    #[serde(default)]
    pub camera_roll_rad: f32,
    #[serde(default)]
    pub camera_yaw_rad: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointCloudSummary {
    pub label: &'static str,
    pub voxel_size_m: f32,
    pub voxels: usize,
    pub stable_voxels: usize,
    pub transient_voxels: usize,
    pub observations: u64,
    pub raw_points_seen: u64,
    pub latest_t_ms: Option<TimeMs>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocalWorldBelief {
    pub label: &'static str,
    pub orientation_status: OrientationStatus,
    pub stable_surfaces: Vec<WorldSurfaceHypothesis>,
    pub stable_blobs: Vec<WorldBlobHypothesis>,
    pub stable_voxels: usize,
    pub transient_voxels: usize,
    pub observations: u64,
    pub latest_t_ms: Option<TimeMs>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrientationStatus {
    pub roll_pitch_corrected: bool,
    pub yaw_source: YawSource,
    pub note: String,
}

impl Default for OrientationStatus {
    fn default() -> Self {
        Self {
            roll_pitch_corrected: false,
            yaw_source: YawSource::Unavailable,
            note: "no point-cloud observation has supplied orientation yet".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldSurfaceHypothesis {
    pub id: String,
    pub kind: WorldSurfaceKind,
    pub centroid: Point3D,
    pub normal: Point3D,
    pub size_m: Point3D,
    pub voxel_count: usize,
    pub confidence: f32,
    pub first_seen_ms: TimeMs,
    pub last_seen_ms: TimeMs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorldSurfaceKind {
    FloorLike,
    WallLike,
    HorizontalSurface,
    UnknownSurface,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldBlobHypothesis {
    pub id: String,
    pub centroid: Point3D,
    pub size_m: Point3D,
    pub voxel_count: usize,
    pub confidence: f32,
    pub first_seen_ms: TimeMs,
    pub last_seen_ms: TimeMs,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocalMap {
    pub cells: BTreeMap<CellKey, OccupancyCell>,
    pub pose_history: Vec<PoseEstimate>,
    pub observations: Vec<MapObservation>,
    #[serde(default)]
    pub submaps: Vec<OccupancySubmap>,
    #[serde(default)]
    pub pose_graph: PoseGraph,
    #[serde(default)]
    pub pose_graph_optimization: PoseGraphOptimizationSummary,
    #[serde(default)]
    pub remap_summary: RemapSummary,
    pub config: MapConfig,
    #[serde(default)]
    pose_graph_ticks_since_node: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OccupancySubmap {
    pub id: String,
    pub node_id: String,
    pub local_pose: Pose2,
    pub range_beams: Vec<RangeBeam>,
    pub t_ms: TimeMs,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_frame_id: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MapConfig {
    pub resolution_m: f32,
    pub range_fov_rad: f32,
    pub max_range_m: f32,
    pub hit_epsilon_m: f32,
    pub free_increment: f32,
    pub occupied_increment: f32,
    pub decay_after_ms: TimeMs,
    pub decay_per_tick: f32,
    pub max_pose_history: usize,
    pub max_observations: usize,
    pub max_submaps: usize,
    pub scan_match_enabled: bool,
    pub scan_match_xy_window_m: f32,
    pub scan_match_theta_window_rad: f32,
    pub scan_match_min_occupied_cells: usize,
    pub scan_match_min_hit_beams: usize,
    pub pose_graph_min_node_distance_m: f32,
    pub pose_graph_min_node_heading_delta_rad: f32,
    pub pose_graph_max_ticks_between_nodes: u64,
    pub pose_graph_optimize_enabled: bool,
    pub pose_graph_optimize_iterations: usize,
    pub pose_graph_optimize_step: f32,
    #[serde(default = "default_pose_graph_min_loop_confidence")]
    pub pose_graph_min_loop_confidence: f32,
    #[serde(default = "default_pose_graph_loop_target_max_distance_m")]
    pub pose_graph_loop_target_max_distance_m: f32,
    #[serde(default = "default_pose_graph_loop_min_geometric_overlap")]
    pub pose_graph_loop_min_geometric_overlap: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MapSummary {
    pub label: &'static str,
    pub resolution_m: f32,
    pub cells: usize,
    pub occupied_cells: usize,
    pub free_cells: usize,
    pub observations: usize,
    pub pose_graph_nodes: usize,
    pub pose_graph_edges: usize,
    pub scan_match_edges: usize,
    pub loop_closure_edges: usize,
    pub loop_closures_accepted: usize,
    pub loop_closures_rejected: usize,
    pub pose_graph_optimization: PoseGraphOptimizationSummary,
    pub remap: RemapSummary,
    pub latest_pose: Option<PoseEstimate>,
    pub latest_observation: Option<MapObservationSummary>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RemapSummary {
    pub generation: u64,
    pub submaps: usize,
    pub cells: usize,
    pub occupied_cells: usize,
    pub free_cells: usize,
    pub latest_t_ms: Option<TimeMs>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MapObservationSummary {
    pub t_ms: TimeMs,
    pub beam_count: usize,
    pub hit_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ScanMatchCorrection {
    pose: Pose2,
    odometry_pose: Pose2,
    score: f32,
    odometry_score: f32,
    confidence_boost: f32,
    covariance_scale: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoseNode {
    pub id: String,
    pub pose_estimate: PoseEstimate,
    pub t_ms: TimeMs,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_frame_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoseEdgeSource {
    Odometry,
    ScanMatch {
        algorithm: String,
        score: f32,
        odometry_score: f32,
    },
    LoopClosureCandidate {
        kind: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        target_frame_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_frame_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_experience_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_instant_frame_id: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        source_vector_refs: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_vector_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        query_vector_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        query_experience_id: Option<String>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoseEdge {
    pub from: String,
    pub to: String,
    pub transform: Pose2,
    pub covariance: [f32; 3],
    pub confidence: f32,
    pub source: PoseEdgeSource,
    pub active: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejection_reason: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PoseGraph {
    pub nodes: Vec<PoseNode>,
    pub edges: Vec<PoseEdge>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoseGraphConfig {
    pub min_node_distance_m: f32,
    pub min_node_heading_delta_rad: f32,
    pub max_ticks_between_nodes: u64,
    pub min_loop_confidence: f32,
    pub loop_target_max_distance_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoseGraphOptimizationConfig {
    pub iterations: usize,
    pub step_size: f32,
    pub max_translation_update_m: f32,
    pub max_heading_update_rad: f32,
    pub convergence_epsilon: f32,
}

impl Default for PoseGraphOptimizationConfig {
    fn default() -> Self {
        Self {
            iterations: 12,
            step_size: 0.45,
            max_translation_update_m: 0.12,
            max_heading_update_rad: 8.0_f32.to_radians(),
            convergence_epsilon: 0.0005,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PoseGraphOptimizationSummary {
    pub iterations: usize,
    pub initial_mean_error: f32,
    pub final_mean_error: f32,
    pub max_node_update_m: f32,
    pub optimized_nodes: usize,
    pub active_edges: usize,
    pub converged: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoopClosureCandidateInput {
    pub target_pose: Pose2,
    pub confidence: f32,
    pub similarity: f32,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_frame_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_frame_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_experience_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_instant_frame_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_vector_refs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_vector_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_vector_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_experience_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PoseGraphReport {
    pub label: &'static str,
    pub nodes: usize,
    pub edges: usize,
    pub odometry_edges: usize,
    pub loop_candidate_edges: usize,
    pub active_loop_candidate_edges: usize,
    pub rejected_loop_candidates: usize,
    pub confidence_distribution: ConfidenceDistribution,
    pub rejected_candidates: Vec<PoseGraphRejectedCandidate>,
    pub graph: PoseGraph,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceDistribution {
    pub min: Option<f32>,
    pub max: Option<f32>,
    pub mean: Option<f32>,
    pub buckets: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoseGraphRejectedCandidate {
    pub from: String,
    pub to: String,
    pub confidence: f32,
    pub reason: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_frame_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_frame_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_experience_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_instant_frame_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_vector_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_vector_id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct PoseGraphBuilder {
    graph: PoseGraph,
    config: PoseGraphConfig,
    ticks_since_node: u64,
}

impl Default for MapConfig {
    fn default() -> Self {
        Self {
            resolution_m: 0.10,
            range_fov_rad: std::f32::consts::PI,
            max_range_m: 4.0,
            hit_epsilon_m: 0.05,
            free_increment: 0.18,
            occupied_increment: 0.35,
            decay_after_ms: 10_000,
            decay_per_tick: 0.04,
            max_pose_history: 1_000,
            max_observations: 250,
            max_submaps: 250,
            scan_match_enabled: true,
            scan_match_xy_window_m: 0.18,
            scan_match_theta_window_rad: 8.0_f32.to_radians(),
            scan_match_min_occupied_cells: 4,
            scan_match_min_hit_beams: 2,
            pose_graph_min_node_distance_m: 0.20,
            pose_graph_min_node_heading_delta_rad: 10.0_f32.to_radians(),
            pose_graph_max_ticks_between_nodes: 8,
            pose_graph_optimize_enabled: true,
            pose_graph_optimize_iterations: 12,
            pose_graph_optimize_step: 0.45,
            pose_graph_min_loop_confidence: default_pose_graph_min_loop_confidence(),
            pose_graph_loop_target_max_distance_m: default_pose_graph_loop_target_max_distance_m(),
            pose_graph_loop_min_geometric_overlap: default_pose_graph_loop_min_geometric_overlap(),
        }
    }
}

fn default_pose_graph_min_loop_confidence() -> f32 {
    0.85
}

fn default_pose_graph_loop_target_max_distance_m() -> f32 {
    0.75
}

fn default_pose_graph_loop_min_geometric_overlap() -> f32 {
    0.40
}

impl Default for PointCloudConfig {
    fn default() -> Self {
        Self {
            voxel_size_m: 0.05,
            max_voxels: 20_000,
            max_points_per_observation: 2_500,
            min_depth_m: 0.35,
            max_depth_m: 8.0,
            confidence_increment: 0.18,
            decay_after_ms: 15_000,
            decay_per_tick: 0.02,
            stable_seen_count: 3,
            stable_confidence: 0.45,
            transient_after_ms: 5_000,
            camera_height_m: 0.18,
            camera_forward_m: 0.0,
            camera_pitch_rad: 0.0,
            camera_roll_rad: 0.0,
            camera_yaw_rad: 0.0,
        }
    }
}

impl Default for VoxelPointCloud {
    fn default() -> Self {
        Self::new(PointCloudConfig::default())
    }
}

impl VoxelPointCloud {
    pub fn new(config: PointCloudConfig) -> Self {
        assert!(config.voxel_size_m > 0.0, "voxel size must be positive");
        assert!(config.max_voxels > 0, "max voxels must be positive");
        Self {
            voxels: BTreeMap::new(),
            config,
            observations: 0,
            raw_points_seen: 0,
            orientation_status: OrientationStatus::default(),
        }
    }

    pub fn observe_snapshot(
        &mut self,
        snapshot: &WorldSnapshot,
        t_ms: TimeMs,
    ) -> PointCloudSummary {
        if let Some(observation) = pointcloud_observation_from_snapshot(snapshot, t_ms, self.config)
        {
            self.integrate_observation(observation);
        } else {
            self.decay_stale(t_ms);
        }
        self.summary()
    }

    pub fn integrate_observation(&mut self, observation: PointCloudObservation) {
        self.observations = self.observations.saturating_add(1);
        self.orientation_status = orientation_status(observation.orientation);
        self.raw_points_seen = self
            .raw_points_seen
            .saturating_add(observation.points.len() as u64);

        for point in &observation.points {
            if !point.position.x_m.is_finite()
                || !point.position.y_m.is_finite()
                || !point.position.z_m.is_finite()
            {
                continue;
            }
            let world = transform_point_to_world(
                point.position,
                observation.frame,
                observation.pose.pose,
                observation.orientation,
                self.config,
            );
            self.bump_voxel(world, point.color_rgb, point.confidence, observation.t_ms);
        }
        self.decay_stale(observation.t_ms);
        self.bound_growth();
    }

    pub fn decay_stale(&mut self, now_ms: TimeMs) {
        for voxel in self.voxels.values_mut() {
            let age = now_ms.saturating_sub(voxel.last_seen_ms);
            if age > self.config.decay_after_ms {
                voxel.confidence = (voxel.confidence - self.config.decay_per_tick).max(0.0);
            }
            voxel.transient =
                !voxel.stable && age >= self.config.transient_after_ms && voxel.seen_count <= 1;
        }
        self.voxels.retain(|_, voxel| voxel.confidence > 0.001);
    }

    pub fn points(&self) -> Vec<VoxelPoint> {
        self.voxels.values().cloned().collect()
    }

    pub fn summary(&self) -> PointCloudSummary {
        let stable_voxels = self.voxels.values().filter(|voxel| voxel.stable).count();
        let transient_voxels = self.voxels.values().filter(|voxel| voxel.transient).count();
        let latest_t_ms = self.voxels.values().map(|voxel| voxel.last_seen_ms).max();
        PointCloudSummary {
            label: WORLD_POINT_CLOUD_LABEL,
            voxel_size_m: self.config.voxel_size_m,
            voxels: self.voxels.len(),
            stable_voxels,
            transient_voxels,
            observations: self.observations,
            raw_points_seen: self.raw_points_seen,
            latest_t_ms,
        }
    }

    pub fn local_world_belief(&self) -> LocalWorldBelief {
        local_world_belief_from_voxels(self)
    }

    fn bump_voxel(
        &mut self,
        position: Point3D,
        color_rgb: Option<[u8; 3]>,
        confidence: f32,
        t_ms: TimeMs,
    ) {
        let key = voxel_key(position, self.config.voxel_size_m);
        let increment = self.config.confidence_increment * confidence.clamp(0.0, 1.0);
        let voxel = self.voxels.entry(key).or_insert_with(|| VoxelPoint {
            key,
            position,
            color_rgb,
            confidence: 0.0,
            first_seen_ms: t_ms,
            last_seen_ms: t_ms,
            seen_count: 0,
            stable: false,
            transient: false,
        });
        let seen = voxel.seen_count as f32;
        voxel.position = Point3D {
            x_m: (voxel.position.x_m * seen + position.x_m) / (seen + 1.0),
            y_m: (voxel.position.y_m * seen + position.y_m) / (seen + 1.0),
            z_m: (voxel.position.z_m * seen + position.z_m) / (seen + 1.0),
        };
        voxel.color_rgb = merge_color(voxel.color_rgb, color_rgb, voxel.seen_count);
        voxel.confidence = (voxel.confidence + increment).clamp(0.0, 1.0);
        voxel.last_seen_ms = t_ms;
        voxel.seen_count = voxel.seen_count.saturating_add(1);
        voxel.stable = voxel.seen_count >= self.config.stable_seen_count
            && voxel.confidence >= self.config.stable_confidence;
        voxel.transient = false;
    }

    fn bound_growth(&mut self) {
        if self.voxels.len() <= self.config.max_voxels {
            return;
        }
        let remove_count = self.voxels.len() - self.config.max_voxels;
        let mut candidates = self
            .voxels
            .iter()
            .map(|(key, voxel)| (*key, voxel.last_seen_ms, voxel.confidence))
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            left.1
                .cmp(&right.1)
                .then_with(|| left.2.total_cmp(&right.2))
        });
        for (key, _, _) in candidates.into_iter().take(remove_count) {
            self.voxels.remove(&key);
        }
    }
}

impl Default for PoseGraphConfig {
    fn default() -> Self {
        Self {
            min_node_distance_m: 0.25,
            min_node_heading_delta_rad: 15.0_f32.to_radians(),
            max_ticks_between_nodes: 10,
            min_loop_confidence: 0.85,
            loop_target_max_distance_m: 0.75,
        }
    }
}

impl PoseGraphBuilder {
    pub fn new(config: PoseGraphConfig) -> Self {
        assert!(
            config.min_node_distance_m >= 0.0,
            "node distance threshold cannot be negative"
        );
        assert!(
            config.min_node_heading_delta_rad >= 0.0,
            "heading threshold cannot be negative"
        );
        assert!(
            (0.0..=1.0).contains(&config.min_loop_confidence),
            "loop confidence gate must be between 0 and 1"
        );
        Self {
            graph: PoseGraph::default(),
            config,
            ticks_since_node: 0,
        }
    }

    pub fn observe(
        &mut self,
        pose: Pose2,
        t_ms: TimeMs,
        source_frame_id: Option<String>,
        loop_candidates: &[LoopClosureCandidateInput],
    ) {
        self.ticks_since_node = self.ticks_since_node.saturating_add(1);
        if self.should_add_node(pose) {
            self.push_node(pose, t_ms, source_frame_id);
        }

        for candidate in loop_candidates {
            self.add_loop_candidate(candidate);
        }
    }

    pub fn finish(self) -> PoseGraph {
        self.graph
    }

    pub fn finish_report(self) -> PoseGraphReport {
        self.finish().report()
    }

    fn should_add_node(&self, pose: Pose2) -> bool {
        let Some(last) = self.graph.nodes.last() else {
            return true;
        };
        distance_m(last.pose_estimate.pose, pose) >= self.config.min_node_distance_m
            || heading_delta_rad(last.pose_estimate.pose.heading_rad, pose.heading_rad)
                >= self.config.min_node_heading_delta_rad
            || self.ticks_since_node >= self.config.max_ticks_between_nodes.max(1)
    }

    fn push_node(&mut self, pose: Pose2, t_ms: TimeMs, source_frame_id: Option<String>) {
        let id = format!("pose-{}", self.graph.nodes.len());
        let previous = self.graph.nodes.last().cloned();
        let node = PoseNode {
            id: id.clone(),
            pose_estimate: PoseEstimate {
                pose,
                confidence: 0.80,
                covariance: [0.05, 0.05, 0.10],
                source: "odometry".to_string(),
                t_ms,
            },
            t_ms,
            source_frame_id,
        };
        self.graph.nodes.push(node);
        self.ticks_since_node = 0;

        if let Some(previous) = previous {
            self.graph.edges.push(PoseEdge {
                from: previous.id,
                to: id,
                transform: pose_delta(previous.pose_estimate.pose, pose),
                covariance: [0.08, 0.08, 0.15],
                confidence: 0.80,
                source: PoseEdgeSource::Odometry,
                active: true,
                rejection_reason: None,
            });
        }
    }

    fn add_loop_candidate(&mut self, candidate: &LoopClosureCandidateInput) {
        let Some(current) = self.graph.nodes.last() else {
            return;
        };
        let from = current.id.clone();
        let source = PoseEdgeSource::LoopClosureCandidate {
            kind: candidate.kind.clone(),
            target_frame_id: candidate.target_frame_id.clone(),
            source_frame_id: candidate.source_frame_id.clone(),
            source_experience_id: candidate.source_experience_id.clone(),
            source_instant_frame_id: candidate.source_instant_frame_id.clone(),
            source_vector_refs: candidate.source_vector_refs.clone(),
            source_vector_id: candidate.source_vector_id.clone(),
            query_vector_id: candidate.query_vector_id.clone(),
            query_experience_id: candidate.query_experience_id.clone(),
        };
        let target = self.find_loop_target(candidate, &from);
        let to = target
            .as_ref()
            .map(|node| node.id.clone())
            .unwrap_or_else(|| "unresolved".to_string());
        let transform = target
            .as_ref()
            .map(|node| pose_delta(current.pose_estimate.pose, node.pose_estimate.pose))
            .unwrap_or_else(|| pose_delta(current.pose_estimate.pose, candidate.target_pose));

        let rejection_reason = if candidate.confidence < self.config.min_loop_confidence {
            Some(format!(
                "confidence {:.3} below gate {:.3}",
                candidate.confidence, self.config.min_loop_confidence
            ))
        } else if target.is_none() {
            Some("no prior node close enough to candidate target".to_string())
        } else {
            None
        };

        self.graph.edges.push(PoseEdge {
            from,
            to,
            transform,
            covariance: loop_covariance(candidate.confidence),
            confidence: candidate.confidence.clamp(0.0, 1.0),
            source,
            active: rejection_reason.is_none(),
            rejection_reason,
        });
    }

    fn find_loop_target(
        &self,
        candidate: &LoopClosureCandidateInput,
        current_id: &str,
    ) -> Option<&PoseNode> {
        if let Some(target_frame_id) = candidate.target_frame_id.as_deref() {
            if let Some(node) = self.graph.nodes.iter().find(|node| {
                node.id != current_id && node.source_frame_id.as_deref() == Some(target_frame_id)
            }) {
                return Some(node);
            }
        }

        self.graph
            .nodes
            .iter()
            .filter(|node| node.id != current_id)
            .filter_map(|node| {
                let distance = distance_m(node.pose_estimate.pose, candidate.target_pose);
                (distance <= self.config.loop_target_max_distance_m).then_some((distance, node))
            })
            .min_by(|left, right| {
                left.0
                    .partial_cmp(&right.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, node)| node)
    }
}

impl Default for PoseGraphBuilder {
    fn default() -> Self {
        Self::new(PoseGraphConfig::default())
    }
}

impl PoseGraph {
    pub fn optimize_anchored(
        &mut self,
        config: PoseGraphOptimizationConfig,
    ) -> PoseGraphOptimizationSummary {
        let active_edges = self.edges.iter().filter(|edge| edge.active).count();
        let initial_mean_error = self.mean_edge_error();
        if self.nodes.len() < 2 || active_edges == 0 || config.iterations == 0 {
            return PoseGraphOptimizationSummary {
                iterations: 0,
                initial_mean_error,
                final_mean_error: initial_mean_error,
                max_node_update_m: 0.0,
                optimized_nodes: self.nodes.len(),
                active_edges,
                converged: true,
            };
        }

        let node_indices = self
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.id.clone(), index))
            .collect::<BTreeMap<_, _>>();
        let mut max_node_update_m = 0.0_f32;
        let mut iterations_run = 0usize;
        let mut converged = false;

        for iteration in 0..config.iterations {
            iterations_run = iteration + 1;
            let mut corrections = vec![Pose2::default(); self.nodes.len()];
            let mut weights = vec![0.0_f32; self.nodes.len()];

            for edge in self.edges.iter().filter(|edge| edge.active) {
                let (Some(&from_index), Some(&to_index)) =
                    (node_indices.get(&edge.from), node_indices.get(&edge.to))
                else {
                    continue;
                };
                if from_index == to_index {
                    continue;
                }

                let from_pose = self.nodes[from_index].pose_estimate.pose;
                let to_pose = self.nodes[to_index].pose_estimate.pose;
                let predicted_to = apply_pose_delta(from_pose, edge.transform);
                let residual = pose_delta(predicted_to, to_pose);
                let weight = edge_constraint_weight(edge) * config.step_size;

                if to_index != 0 {
                    corrections[to_index].x_m -= residual.x_m * weight;
                    corrections[to_index].y_m -= residual.y_m * weight;
                    corrections[to_index].heading_rad -= residual.heading_rad * weight;
                    weights[to_index] += weight;
                }
                if from_index != 0 {
                    corrections[from_index].x_m += residual.x_m * weight;
                    corrections[from_index].y_m += residual.y_m * weight;
                    corrections[from_index].heading_rad += residual.heading_rad * weight;
                    weights[from_index] += weight;
                }
            }

            let mut iteration_max_update = 0.0_f32;
            for (index, node) in self.nodes.iter_mut().enumerate().skip(1) {
                if weights[index] <= 0.0 {
                    continue;
                }
                let mut correction = corrections[index];
                correction.x_m /= weights[index];
                correction.y_m /= weights[index];
                correction.heading_rad = normalize_angle(correction.heading_rad / weights[index]);
                correction = clamp_pose_update(correction, config);
                let update_m = (correction.x_m.powi(2) + correction.y_m.powi(2)).sqrt();
                iteration_max_update = iteration_max_update.max(update_m);
                node.pose_estimate.pose.x_m += correction.x_m;
                node.pose_estimate.pose.y_m += correction.y_m;
                node.pose_estimate.pose.heading_rad =
                    normalize_angle(node.pose_estimate.pose.heading_rad + correction.heading_rad);
            }

            max_node_update_m = max_node_update_m.max(iteration_max_update);
            if iteration_max_update < config.convergence_epsilon {
                converged = true;
                break;
            }
        }

        PoseGraphOptimizationSummary {
            iterations: iterations_run,
            initial_mean_error,
            final_mean_error: self.mean_edge_error(),
            max_node_update_m,
            optimized_nodes: self.nodes.len(),
            active_edges,
            converged,
        }
    }

    fn mean_edge_error(&self) -> f32 {
        let node_indices = self
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.id.as_str(), index))
            .collect::<BTreeMap<_, _>>();
        let mut total = 0.0;
        let mut count = 0usize;
        for edge in self.edges.iter().filter(|edge| edge.active) {
            let (Some(&from_index), Some(&to_index)) = (
                node_indices.get(edge.from.as_str()),
                node_indices.get(edge.to.as_str()),
            ) else {
                continue;
            };
            let predicted_to =
                apply_pose_delta(self.nodes[from_index].pose_estimate.pose, edge.transform);
            let residual = pose_delta(predicted_to, self.nodes[to_index].pose_estimate.pose);
            total += residual.x_m.hypot(residual.y_m) + residual.heading_rad.abs() * 0.25;
            count = count.saturating_add(1);
        }
        if count == 0 {
            0.0
        } else {
            total / count as f32
        }
    }

    pub fn report(self) -> PoseGraphReport {
        let odometry_edges = self
            .edges
            .iter()
            .filter(|edge| matches!(edge.source, PoseEdgeSource::Odometry))
            .count();
        let loop_edges: Vec<_> = self
            .edges
            .iter()
            .filter(|edge| matches!(edge.source, PoseEdgeSource::LoopClosureCandidate { .. }))
            .collect();
        let active_loop_candidate_edges = loop_edges.iter().filter(|edge| edge.active).count();
        let rejected_candidates = loop_edges
            .iter()
            .filter_map(|edge| {
                let reason = edge.rejection_reason.clone()?;
                let (
                    kind,
                    target_frame_id,
                    source_frame_id,
                    source_experience_id,
                    source_instant_frame_id,
                    source_vector_id,
                    query_vector_id,
                ) = match &edge.source {
                    PoseEdgeSource::LoopClosureCandidate {
                        kind,
                        target_frame_id,
                        source_frame_id,
                        source_experience_id,
                        source_instant_frame_id,
                        source_vector_id,
                        query_vector_id,
                        ..
                    } => (
                        kind.clone(),
                        target_frame_id.clone(),
                        source_frame_id.clone(),
                        source_experience_id.clone(),
                        source_instant_frame_id.clone(),
                        source_vector_id.clone(),
                        query_vector_id.clone(),
                    ),
                    PoseEdgeSource::Odometry => {
                        ("odometry".to_string(), None, None, None, None, None, None)
                    }
                    PoseEdgeSource::ScanMatch { .. } => {
                        ("scan_match".to_string(), None, None, None, None, None, None)
                    }
                };
                Some(PoseGraphRejectedCandidate {
                    from: edge.from.clone(),
                    to: edge.to.clone(),
                    confidence: edge.confidence,
                    reason,
                    kind,
                    target_frame_id,
                    source_frame_id,
                    source_experience_id,
                    source_instant_frame_id,
                    source_vector_id,
                    query_vector_id,
                })
            })
            .collect::<Vec<_>>();

        PoseGraphReport {
            label: POSE_GRAPH_LABEL,
            nodes: self.nodes.len(),
            edges: self.edges.len(),
            odometry_edges,
            loop_candidate_edges: loop_edges.len(),
            active_loop_candidate_edges,
            rejected_loop_candidates: rejected_candidates.len(),
            confidence_distribution: confidence_distribution(
                loop_edges.iter().map(|edge| edge.confidence),
            ),
            rejected_candidates,
            graph: self,
        }
    }
}

impl Default for LocalMap {
    fn default() -> Self {
        Self::new(MapConfig::default())
    }
}

impl LocalMap {
    pub fn new(config: MapConfig) -> Self {
        assert!(config.resolution_m > 0.0, "map resolution must be positive");
        Self {
            cells: BTreeMap::new(),
            pose_history: Vec::new(),
            observations: Vec::new(),
            submaps: Vec::new(),
            pose_graph: PoseGraph::default(),
            pose_graph_optimization: PoseGraphOptimizationSummary::default(),
            remap_summary: RemapSummary::default(),
            config,
            pose_graph_ticks_since_node: 0,
        }
    }

    pub fn observe_snapshot(&mut self, snapshot: &WorldSnapshot, t_ms: TimeMs) -> MapSummary {
        let observation = observation_from_snapshot(snapshot, t_ms, self.config);
        self.integrate_observation(observation);
        self.decay_stale(t_ms);
        self.summary()
    }

    pub fn observe_now(&mut self, now: &Now) -> MapSummary {
        let observation = observation_from_now(now, self.config);
        self.integrate_observation(observation);
        self.decay_stale(now.t_ms);
        self.summary()
    }

    pub fn integrate_observation(&mut self, observation: MapObservation) -> MapSummary {
        self.integrate_observation_with_loop_candidates(observation, &[])
    }

    pub fn integrate_observation_with_loop_candidates(
        &mut self,
        observation: MapObservation,
        loop_candidates: &[LoopClosureCandidateInput],
    ) -> MapSummary {
        let (mut observation, scan_match) = self.scan_matched_observation(observation);
        let pose_node_id =
            self.update_pose_graph(&observation, scan_match.as_ref(), loop_candidates);
        self.optimize_pose_graph();
        if let Some(latest) = self.pose_graph.nodes.last() {
            if latest.t_ms == observation.t_ms {
                observation.pose.pose = latest.pose_estimate.pose;
                observation.pose.confidence = observation
                    .pose
                    .confidence
                    .max(latest.pose_estimate.confidence);
            }
        }
        self.pose_history.push(observation.pose.clone());
        cap_vec(&mut self.pose_history, self.config.max_pose_history);

        self.store_submap(&observation, pose_node_id);

        self.observations.push(observation);
        cap_vec(&mut self.observations, self.config.max_observations);
        self.rebuild_occupancy_from_submaps();
        self.summary()
    }

    fn scan_matched_observation(
        &self,
        mut observation: MapObservation,
    ) -> (MapObservation, Option<ScanMatchCorrection>) {
        let correction = self.scan_match_pose(&observation);
        let Some(correction) = correction else {
            return (observation, None);
        };
        observation.pose.pose = correction.pose;
        observation.pose.confidence =
            (observation.pose.confidence + correction.confidence_boost).clamp(0.0, 0.98);
        observation.pose.covariance = [
            (observation.pose.covariance[0] * correction.covariance_scale).max(0.01),
            (observation.pose.covariance[1] * correction.covariance_scale).max(0.01),
            (observation.pose.covariance[2] * correction.covariance_scale).max(0.02),
        ];
        observation.pose.source = "odometry+occupancy_scan_match".to_string();
        if let Some(object) = observation.source_snapshot.as_object_mut() {
            object.insert(
                "scan_match".to_string(),
                serde_json::json!({
                    "dx_m": correction.pose.x_m - correction.odometry_pose.x_m,
                    "dy_m": correction.pose.y_m - correction.odometry_pose.y_m,
                    "dtheta_rad": normalize_angle(correction.pose.heading_rad - correction.odometry_pose.heading_rad),
                    "score": correction.score,
                    "odometry_score": correction.odometry_score,
                    "confidence_boost": correction.confidence_boost,
                }),
            );
        }
        (observation, Some(correction))
    }

    fn update_pose_graph(
        &mut self,
        observation: &MapObservation,
        scan_match: Option<&ScanMatchCorrection>,
        loop_candidates: &[LoopClosureCandidateInput],
    ) -> Option<String> {
        self.pose_graph_ticks_since_node = self.pose_graph_ticks_since_node.saturating_add(1);
        if !self.should_add_live_pose_node(observation.pose.pose) {
            return self.pose_graph.nodes.last().map(|node| node.id.clone());
        }

        let id = format!("live-pose-{}", self.pose_graph.nodes.len());
        let previous = self.pose_graph.nodes.last().cloned();
        self.pose_graph.nodes.push(PoseNode {
            id: id.clone(),
            pose_estimate: observation.pose.clone(),
            t_ms: observation.t_ms,
            source_frame_id: source_frame_id_from_observation(observation),
        });
        self.pose_graph_ticks_since_node = 0;

        if let Some(previous) = previous {
            let (source, covariance, confidence) = if let Some(scan_match) = scan_match {
                (
                    PoseEdgeSource::ScanMatch {
                        algorithm: "correlative_occupancy_grid".to_string(),
                        score: scan_match.score,
                        odometry_score: scan_match.odometry_score,
                    },
                    observation.pose.covariance,
                    observation.pose.confidence,
                )
            } else {
                (
                    PoseEdgeSource::Odometry,
                    [0.08, 0.08, 0.15],
                    observation.pose.confidence.min(0.85),
                )
            };
            self.pose_graph.edges.push(PoseEdge {
                from: previous.id,
                to: id.clone(),
                transform: pose_delta(previous.pose_estimate.pose, observation.pose.pose),
                covariance,
                confidence,
                source,
                active: true,
                rejection_reason: None,
            });
        }
        for candidate in loop_candidates {
            self.add_live_loop_candidate(&id, observation, candidate);
        }
        Some(id)
    }

    fn add_live_loop_candidate(
        &mut self,
        current_node_id: &str,
        observation: &MapObservation,
        candidate: &LoopClosureCandidateInput,
    ) {
        let Some(current) = self
            .pose_graph
            .nodes
            .iter()
            .find(|node| node.id == current_node_id)
            .cloned()
        else {
            return;
        };
        let source = PoseEdgeSource::LoopClosureCandidate {
            kind: candidate.kind.clone(),
            target_frame_id: candidate.target_frame_id.clone(),
            source_frame_id: candidate.source_frame_id.clone(),
            source_experience_id: candidate.source_experience_id.clone(),
            source_instant_frame_id: candidate.source_instant_frame_id.clone(),
            source_vector_refs: candidate.source_vector_refs.clone(),
            source_vector_id: candidate.source_vector_id.clone(),
            query_vector_id: candidate.query_vector_id.clone(),
            query_experience_id: candidate.query_experience_id.clone(),
        };
        let target = self.find_live_loop_target(candidate, &current.id).cloned();
        let to = target
            .as_ref()
            .map(|node| node.id.clone())
            .unwrap_or_else(|| "unresolved".to_string());
        let target_pose = target
            .as_ref()
            .map(|node| node.pose_estimate.pose)
            .unwrap_or(candidate.target_pose);
        let rejection_reason =
            self.live_loop_rejection_reason(&current, target.as_ref(), observation, candidate);

        self.pose_graph.edges.push(PoseEdge {
            from: current.id,
            to,
            transform: pose_delta(current.pose_estimate.pose, target_pose),
            covariance: loop_covariance(candidate.confidence),
            confidence: candidate.confidence.clamp(0.0, 1.0),
            source,
            active: rejection_reason.is_none(),
            rejection_reason,
        });
    }

    fn live_loop_rejection_reason(
        &self,
        current: &PoseNode,
        target: Option<&PoseNode>,
        observation: &MapObservation,
        candidate: &LoopClosureCandidateInput,
    ) -> Option<String> {
        let current_source_frame_id = current.source_frame_id.as_deref();
        if candidate.target_frame_id.as_deref() == Some(current.id.as_str())
            || candidate.target_frame_id.as_deref() == current_source_frame_id
        {
            return Some("candidate targets the current/source frame".to_string());
        }
        if candidate.confidence < self.config.pose_graph_min_loop_confidence {
            return Some(format!(
                "confidence {:.3} below gate {:.3}",
                candidate.confidence, self.config.pose_graph_min_loop_confidence
            ));
        }
        let target_distance = distance_m(current.pose_estimate.pose, candidate.target_pose);
        if target_distance > self.config.pose_graph_loop_target_max_distance_m {
            return Some(format!(
                "target pose {:.3}m from current pose exceeds gate {:.3}m",
                target_distance, self.config.pose_graph_loop_target_max_distance_m
            ));
        }
        let Some(target) = target else {
            return Some("no prior node close enough to candidate target".to_string());
        };
        let overlap = self.loop_candidate_geometric_overlap(target.pose_estimate.pose, observation);
        if overlap < self.config.pose_graph_loop_min_geometric_overlap {
            return Some(format!(
                "geometric occupancy agreement {:.3} below gate {:.3}",
                overlap, self.config.pose_graph_loop_min_geometric_overlap
            ));
        }
        None
    }

    fn find_live_loop_target(
        &self,
        candidate: &LoopClosureCandidateInput,
        current_id: &str,
    ) -> Option<&PoseNode> {
        if let Some(target_frame_id) = candidate.target_frame_id.as_deref() {
            if let Some(node) = self.pose_graph.nodes.iter().find(|node| {
                node.id != current_id && node.source_frame_id.as_deref() == Some(target_frame_id)
            }) {
                return Some(node);
            }
        }

        self.pose_graph
            .nodes
            .iter()
            .filter(|node| node.id != current_id)
            .filter_map(|node| {
                let distance = distance_m(node.pose_estimate.pose, candidate.target_pose);
                (distance <= self.config.pose_graph_loop_target_max_distance_m)
                    .then_some((distance, node))
            })
            .min_by(|left, right| {
                left.0
                    .partial_cmp(&right.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, node)| node)
    }

    fn loop_candidate_geometric_overlap(
        &self,
        candidate_pose: Pose2,
        observation: &MapObservation,
    ) -> f32 {
        let mut hits = 0usize;
        let mut matched_hits = 0usize;
        for beam in observation
            .range_beams
            .iter()
            .filter(|beam| beam.hit && beam.confidence > 0.0)
        {
            if !beam.distance_m.is_finite() || beam.distance_m <= 0.0 {
                continue;
            }
            hits = hits.saturating_add(1);
            let end = project_beam_endpoint(
                candidate_pose,
                beam.angle_rad,
                beam.distance_m.min(self.config.max_range_m),
            );
            let key = cell_key(end.x_m, end.y_m, self.config.resolution_m);
            if self.cells.get(&key).is_some_and(|cell| {
                cell.occupied_score > cell.free_score && cell.confidence >= 0.05
            }) {
                matched_hits = matched_hits.saturating_add(1);
            }
        }
        if hits == 0 {
            0.0
        } else {
            matched_hits as f32 / hits as f32
        }
    }

    fn should_add_live_pose_node(&self, pose: Pose2) -> bool {
        let Some(last) = self.pose_graph.nodes.last() else {
            return true;
        };
        distance_m(last.pose_estimate.pose, pose) >= self.config.pose_graph_min_node_distance_m
            || heading_delta_rad(last.pose_estimate.pose.heading_rad, pose.heading_rad)
                >= self.config.pose_graph_min_node_heading_delta_rad
            || self.pose_graph_ticks_since_node
                >= self.config.pose_graph_max_ticks_between_nodes.max(1)
    }

    fn optimize_pose_graph(&mut self) {
        if !self.config.pose_graph_optimize_enabled {
            self.pose_graph_optimization = PoseGraphOptimizationSummary::default();
            return;
        }
        self.pose_graph_optimization =
            self.pose_graph
                .optimize_anchored(PoseGraphOptimizationConfig {
                    iterations: self.config.pose_graph_optimize_iterations,
                    step_size: self.config.pose_graph_optimize_step,
                    ..PoseGraphOptimizationConfig::default()
                });
    }

    fn store_submap(&mut self, observation: &MapObservation, pose_node_id: Option<String>) {
        let Some(node_id) = pose_node_id else {
            return;
        };
        let Some(node) = self.pose_graph.nodes.iter().find(|node| node.id == node_id) else {
            return;
        };
        self.submaps.push(OccupancySubmap {
            id: format!("submap-{}", self.submaps.len()),
            node_id,
            local_pose: pose_delta(node.pose_estimate.pose, observation.pose.pose),
            range_beams: observation.range_beams.clone(),
            t_ms: observation.t_ms,
            source_frame_id: source_frame_id_from_observation(observation),
        });
        cap_vec(&mut self.submaps, self.config.max_submaps);
    }

    fn rebuild_occupancy_from_submaps(&mut self) {
        let submaps = self.submaps.clone();
        self.cells.clear();
        for submap in &submaps {
            let Some(node) = self
                .pose_graph
                .nodes
                .iter()
                .find(|node| node.id == submap.node_id)
            else {
                continue;
            };
            let pose = apply_pose_delta(node.pose_estimate.pose, submap.local_pose);
            for beam in &submap.range_beams {
                self.integrate_beam(pose, beam, submap.t_ms);
            }
        }
        self.update_remap_summary();
    }

    fn update_remap_summary(&mut self) {
        let occupied_cells = self
            .cells
            .values()
            .filter(|cell| cell.occupied_score > cell.free_score && cell.confidence > 0.0)
            .count();
        let free_cells = self
            .cells
            .values()
            .filter(|cell| cell.free_score >= cell.occupied_score && cell.confidence > 0.0)
            .count();
        self.remap_summary = RemapSummary {
            generation: self.remap_summary.generation.saturating_add(1),
            submaps: self.submaps.len(),
            cells: self.cells.len(),
            occupied_cells,
            free_cells,
            latest_t_ms: self.submaps.iter().map(|submap| submap.t_ms).max(),
        };
    }

    fn scan_match_pose(&self, observation: &MapObservation) -> Option<ScanMatchCorrection> {
        if !self.config.scan_match_enabled {
            return None;
        }
        let occupied_cells = self
            .cells
            .values()
            .filter(|cell| cell.occupied_score > cell.free_score && cell.confidence > 0.05)
            .count();
        if occupied_cells < self.config.scan_match_min_occupied_cells {
            return None;
        }
        let hit_beams = observation
            .range_beams
            .iter()
            .filter(|beam| beam.hit && beam.confidence > 0.0)
            .count();
        if hit_beams < self.config.scan_match_min_hit_beams {
            return None;
        }

        let odometry_pose = observation.pose.pose;
        let odometry_score = self.scan_match_score(odometry_pose, &observation.range_beams);
        let mut best_pose = odometry_pose;
        let mut best_score = odometry_score;
        let xy_step = (self.config.resolution_m * 0.5).max(0.025);
        let theta_step = (self.config.scan_match_theta_window_rad / 2.0).max(2.0_f32.to_radians());
        let xy_steps = (self.config.scan_match_xy_window_m / xy_step).ceil() as i32;
        let theta_steps = (self.config.scan_match_theta_window_rad / theta_step).ceil() as i32;

        for ix in -xy_steps..=xy_steps {
            for iy in -xy_steps..=xy_steps {
                for itheta in -theta_steps..=theta_steps {
                    let candidate = Pose2 {
                        x_m: odometry_pose.x_m + ix as f32 * xy_step,
                        y_m: odometry_pose.y_m + iy as f32 * xy_step,
                        heading_rad: normalize_angle(
                            odometry_pose.heading_rad + itheta as f32 * theta_step,
                        ),
                    };
                    let score = self.scan_match_score(candidate, &observation.range_beams);
                    if score > best_score {
                        best_score = score;
                        best_pose = candidate;
                    }
                }
            }
        }

        let improvement = best_score - odometry_score;
        if improvement < 0.20 {
            return None;
        }
        let confidence_boost = (improvement / hit_beams as f32 * 0.20).clamp(0.02, 0.12);
        Some(ScanMatchCorrection {
            pose: best_pose,
            odometry_pose,
            score: best_score,
            odometry_score,
            confidence_boost,
            covariance_scale: (1.0 - confidence_boost).clamp(0.75, 0.98),
        })
    }

    fn scan_match_score(&self, pose: Pose2, beams: &[RangeBeam]) -> f32 {
        let mut score = 0.0;
        let mut evidence = 0usize;
        for beam in beams.iter().filter(|beam| beam.confidence > 0.0) {
            if !beam.distance_m.is_finite() || beam.distance_m <= 0.0 {
                continue;
            }
            let distance = beam.distance_m.min(self.config.max_range_m);
            if beam.hit {
                let end = project_beam_endpoint(pose, beam.angle_rad, distance);
                let end_key = cell_key(end.x_m, end.y_m, self.config.resolution_m);
                score += self.cell_match_score(end_key) * 1.5;
                evidence = evidence.saturating_add(1);
            }
            let free_end = if beam.hit {
                distance - self.config.resolution_m
            } else {
                distance
            };
            for key in trace_cells(
                pose,
                beam.angle_rad,
                free_end.max(0.0),
                self.config.resolution_m,
            )
            .into_iter()
            .step_by(2)
            {
                score += self.free_match_score(key) * 0.18;
                evidence = evidence.saturating_add(1);
            }
        }
        if evidence == 0 {
            0.0
        } else {
            score / evidence as f32
        }
    }

    fn cell_match_score(&self, key: CellKey) -> f32 {
        self.cells
            .get(&key)
            .map(|cell| (cell.occupied_score - cell.free_score) * cell.confidence.clamp(0.0, 1.0))
            .unwrap_or(-0.08)
    }

    fn free_match_score(&self, key: CellKey) -> f32 {
        self.cells
            .get(&key)
            .map(|cell| (cell.free_score - cell.occupied_score) * cell.confidence.clamp(0.0, 1.0))
            .unwrap_or(0.02)
    }

    pub fn decay_stale(&mut self, now_ms: TimeMs) {
        for cell in self.cells.values_mut() {
            if now_ms.saturating_sub(cell.last_seen_ms) <= self.config.decay_after_ms {
                continue;
            }
            cell.occupied_score = (cell.occupied_score - self.config.decay_per_tick).max(0.0);
            cell.free_score = (cell.free_score - self.config.decay_per_tick).max(0.0);
            cell.confidence = cell.occupied_score.max(cell.free_score).clamp(0.0, 1.0);
        }
        self.cells.retain(|_, cell| cell.confidence > 0.001);
    }

    pub fn summary(&self) -> MapSummary {
        let occupied_cells = self
            .cells
            .values()
            .filter(|cell| cell.occupied_score > cell.free_score && cell.confidence > 0.0)
            .count();
        let free_cells = self
            .cells
            .values()
            .filter(|cell| cell.free_score >= cell.occupied_score && cell.confidence > 0.0)
            .count();
        let loop_closure_edges = self
            .pose_graph
            .edges
            .iter()
            .filter(|edge| matches!(edge.source, PoseEdgeSource::LoopClosureCandidate { .. }))
            .count();
        let loop_closures_accepted = self
            .pose_graph
            .edges
            .iter()
            .filter(|edge| {
                matches!(edge.source, PoseEdgeSource::LoopClosureCandidate { .. }) && edge.active
            })
            .count();
        let loop_closures_rejected = loop_closure_edges.saturating_sub(loop_closures_accepted);

        MapSummary {
            label: MAP_LABEL,
            resolution_m: self.config.resolution_m,
            cells: self.cells.len(),
            occupied_cells,
            free_cells,
            observations: self.observations.len(),
            pose_graph_nodes: self.pose_graph.nodes.len(),
            pose_graph_edges: self.pose_graph.edges.len(),
            scan_match_edges: self
                .pose_graph
                .edges
                .iter()
                .filter(|edge| matches!(edge.source, PoseEdgeSource::ScanMatch { .. }))
                .count(),
            loop_closure_edges,
            loop_closures_accepted,
            loop_closures_rejected,
            pose_graph_optimization: self.pose_graph_optimization,
            remap: self.remap_summary,
            latest_pose: self.pose_history.last().cloned(),
            latest_observation: self
                .observations
                .last()
                .map(|observation| MapObservationSummary {
                    t_ms: observation.t_ms,
                    beam_count: observation.range_beams.len(),
                    hit_count: observation
                        .range_beams
                        .iter()
                        .filter(|beam| beam.hit)
                        .count(),
                }),
        }
    }

    fn integrate_beam(&mut self, pose: Pose2, beam: &RangeBeam, t_ms: TimeMs) {
        if !beam.distance_m.is_finite() || beam.distance_m <= 0.0 {
            return;
        }

        let distance = beam.distance_m.min(self.config.max_range_m);
        let end = project_beam_endpoint(pose, beam.angle_rad, distance);
        let origin_key = cell_key(pose.x_m, pose.y_m, self.config.resolution_m);
        let end_key = cell_key(end.x_m, end.y_m, self.config.resolution_m);
        let free_end = if beam.hit {
            distance - self.config.resolution_m
        } else {
            distance
        };
        for key in trace_cells(
            pose,
            beam.angle_rad,
            free_end.max(0.0),
            self.config.resolution_m,
        ) {
            if beam.hit && key == end_key {
                continue;
            }
            if key == origin_key {
                continue;
            }
            self.bump_free(key, t_ms, beam.confidence);
        }

        if beam.hit && beam.confidence > 0.0 && beam.distance_m <= self.config.max_range_m {
            self.bump_occupied(end_key, t_ms, beam.confidence);
        }
    }

    fn bump_free(&mut self, key: CellKey, t_ms: TimeMs, confidence: f32) {
        let increment = self.config.free_increment * confidence.clamp(0.0, 1.0);
        let cell = self.cell_mut(key, t_ms);
        cell.free_score = (cell.free_score + increment).clamp(0.0, 1.0);
        cell.occupied_score = (cell.occupied_score - increment * 0.25).max(0.0);
        cell.confidence = cell.free_score.max(cell.occupied_score).clamp(0.0, 1.0);
        cell.last_seen_ms = t_ms;
    }

    fn bump_occupied(&mut self, key: CellKey, t_ms: TimeMs, confidence: f32) {
        let increment = self.config.occupied_increment * confidence.clamp(0.0, 1.0);
        let cell = self.cell_mut(key, t_ms);
        cell.occupied_score = (cell.occupied_score + increment).clamp(0.0, 1.0);
        cell.free_score = (cell.free_score - increment * 0.20).max(0.0);
        cell.confidence = cell.free_score.max(cell.occupied_score).clamp(0.0, 1.0);
        cell.last_seen_ms = t_ms;
    }

    fn cell_mut(&mut self, key: CellKey, t_ms: TimeMs) -> &mut OccupancyCell {
        self.cells.entry(key).or_insert_with(|| OccupancyCell {
            key,
            occupied_score: 0.0,
            free_score: 0.0,
            confidence: 0.0,
            last_seen_ms: t_ms,
        })
    }
}

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
        &snapshot.range.beams,
        snapshot.range.nearest_m,
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
        &now.range.beams,
        now.range.nearest_m,
        serde_json::json!({
            "body": {
                "odometry": now.body.odometry,
                "velocity": now.body.velocity,
            },
            "range": now.range,
            "source": now.extensions.get("source"),
            "mode": now.extensions.get("mode"),
        }),
        now.t_ms,
        config,
    )
}

fn observation_from_parts(
    odometry: Pose2,
    pose_confidence: f32,
    beams: &[f32],
    nearest_m: Option<f32>,
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
    let beam_count = beams.len();
    let range_beams = beams
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
            let angle_rad = -config.range_fov_rad * 0.5 + ratio * config.range_fov_rad;
            let nearest_hit = nearest_m
                .filter(|nearest| nearest.is_finite())
                .map(|nearest| (distance - nearest).abs() <= config.hit_epsilon_m)
                .unwrap_or(false);
            let hit = distance <= config.max_range_m
                && (nearest_hit || distance < config.max_range_m - config.hit_epsilon_m);
            Some(RangeBeam {
                angle_rad,
                distance_m: distance,
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

pub fn pointcloud_observation_from_snapshot(
    snapshot: &WorldSnapshot,
    t_ms: TimeMs,
    config: PointCloudConfig,
) -> Option<PointCloudObservation> {
    pointcloud_observation_from_kinect(
        &snapshot.kinect,
        snapshot.body.odometry,
        orientation_from_snapshot(snapshot),
        odometry_confidence_from_motion(
            snapshot.body.velocity.forward_m_s,
            snapshot.body.velocity.turn_rad_s,
        ),
        t_ms,
        config,
    )
}

pub fn pointcloud_observation_from_kinect(
    kinect: &KinectSense,
    pose: Pose2,
    orientation: OrientationEstimate,
    pose_confidence: f32,
    t_ms: TimeMs,
    config: PointCloudConfig,
) -> Option<PointCloudObservation> {
    if kinect.depth_m.is_empty() {
        return None;
    }
    let projection = DepthProjection::from_kinect(kinect)
        .unwrap_or_else(|| DepthProjection::legacy(kinect.depth_m.len()));
    let stride = kinect
        .depth_m
        .len()
        .div_ceil(config.max_points_per_observation.max(1))
        .max(1);
    let min_depth_m = positive_or(kinect.min_depth_m, config.min_depth_m);
    let max_depth_m = positive_or(kinect.max_depth_m, config.max_depth_m);
    let mut skipped_depth_count = 0usize;
    let mut clipped_depth_count = 0usize;
    let mut points = Vec::new();
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
        let z_m = *depth;
        let x_m = (u - projection.cx) * z_m / projection.fx.max(f32::EPSILON);
        let y_m = (v - projection.cy) * z_m / projection.fy.max(f32::EPSILON);
        points.push(PointCloudPoint {
            position: Point3D { x_m, y_m, z_m },
            color_rgb: depth_shade(z_m, max_depth_m),
            confidence: pose_confidence,
        });
    }
    if points.is_empty() {
        return None;
    }
    Some(PointCloudObservation {
        frame: projection.frame,
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
        }),
    })
}

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
    let robot = apply_roll_pitch(robot, orientation);
    let yaw = orientation.yaw_rad.unwrap_or(pose.heading_rad);
    let sin = yaw.sin();
    let cos = yaw.cos();
    Point3D {
        x_m: pose.x_m + robot.x_m * cos - robot.y_m * sin,
        y_m: pose.y_m + robot.x_m * sin + robot.y_m * cos,
        z_m: robot.z_m,
    }
}

pub fn orientation_from_snapshot(snapshot: &WorldSnapshot) -> OrientationEstimate {
    orientation_from_imu(&snapshot.imu, snapshot.body.odometry.heading_rad)
}

pub fn orientation_from_imu(imu: &ImuSense, odometry_heading_rad: f32) -> OrientationEstimate {
    let finite = |index: usize| {
        imu.orientation
            .get(index)
            .copied()
            .filter(|value| value.is_finite())
    };
    let (roll, pitch, imu_yaw) = match imu.orientation.len() {
        0 => (None, None, None),
        1 => (None, None, finite(0)),
        2 => (finite(0), finite(1), None),
        _ => (finite(0), finite(1), finite(2)),
    };
    let roll_pitch_plausible = roll
        .map(|value| value.abs() <= MAX_TRUSTED_GRAVITY_TILT_RAD)
        .unwrap_or(true)
        && pitch
            .map(|value| value.abs() <= MAX_TRUSTED_GRAVITY_TILT_RAD)
            .unwrap_or(true);
    let trusted_roll = roll.filter(|_| roll_pitch_plausible);
    let trusted_pitch = pitch.filter(|_| roll_pitch_plausible);
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
        y_m: rotated.y_m,
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

fn apply_roll_pitch(point: Point3D, orientation: OrientationEstimate) -> Point3D {
    let mut rotated = point;
    if let Some(roll) = orientation.roll_rad {
        let (sin, cos) = roll.sin_cos();
        rotated = Point3D {
            x_m: rotated.x_m,
            y_m: rotated.y_m * cos - rotated.z_m * sin,
            z_m: rotated.y_m * sin + rotated.z_m * cos,
        };
    }
    if let Some(pitch) = orientation.pitch_rad {
        let (sin, cos) = pitch.sin_cos();
        rotated = Point3D {
            x_m: rotated.x_m * cos + rotated.z_m * sin,
            y_m: rotated.y_m,
            z_m: -rotated.x_m * sin + rotated.z_m * cos,
        };
    }
    rotated
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

    fn legacy(depth_len: usize) -> Self {
        let width = (depth_len as f32).sqrt().ceil().max(1.0) as usize;
        let height = depth_len.div_ceil(width).max(1);
        Self {
            width,
            height,
            fx: width.max(1) as f32,
            fy: width.max(1) as f32,
            cx: (width.saturating_sub(1)) as f32 * 0.5,
            cy: (height.saturating_sub(1)) as f32 * 0.5,
            frame: PointCloudFrame::DepthImageUnknown,
        }
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
    Pose2 {
        x_m: to.x_m - from.x_m,
        y_m: to.y_m - from.y_m,
        heading_rad: normalize_angle(to.heading_rad - from.heading_rad),
    }
}

fn apply_pose_delta(from: Pose2, delta: Pose2) -> Pose2 {
    Pose2 {
        x_m: from.x_m + delta.x_m,
        y_m: from.y_m + delta.y_m,
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

#[cfg(test)]
mod tests {
    use super::*;
    use netherwick_now::{KinectSense, RangeSense};

    fn snapshot_at(x_m: f32, y_m: f32, heading_rad: f32, beams: Vec<f32>) -> WorldSnapshot {
        let mut snapshot = WorldSnapshot::default();
        snapshot.body.odometry = Pose2 {
            x_m,
            y_m,
            heading_rad,
        };
        snapshot.range = RangeSense {
            schema_version: 1,
            nearest_m: beams.iter().copied().reduce(f32::min),
            beams,
        };
        snapshot
    }

    fn kinect_snapshot_at(
        x_m: f32,
        y_m: f32,
        heading_rad: f32,
        depth_m: Vec<f32>,
        width: u32,
        height: u32,
    ) -> WorldSnapshot {
        let mut snapshot = WorldSnapshot::default();
        snapshot.body.odometry = Pose2 {
            x_m,
            y_m,
            heading_rad,
        };
        snapshot.kinect = KinectSense {
            depth_m,
            depth_width: width,
            depth_height: height,
            depth_fx: 1.0,
            depth_fy: 1.0,
            depth_cx: 0.0,
            depth_cy: 0.0,
            min_depth_m: 0.1,
            max_depth_m: 8.0,
            depth_coordinate_system: Some("kinect_camera".to_string()),
            ..KinectSense::default()
        };
        snapshot
    }

    #[test]
    fn beam_projection_uses_pose_heading_and_relative_angle() {
        let pose = Pose2 {
            x_m: 1.0,
            y_m: 2.0,
            heading_rad: std::f32::consts::FRAC_PI_2,
        };
        let endpoint = project_beam_endpoint(pose, 0.0, 1.5);
        assert!((endpoint.x_m - 1.0).abs() < 0.001);
        assert!((endpoint.y_m - 3.5).abs() < 0.001);
    }

    #[test]
    fn occupancy_update_accumulates_endpoint_hits() {
        let mut map = LocalMap::new(MapConfig {
            resolution_m: 0.5,
            ..MapConfig::default()
        });
        let snapshot = snapshot_at(0.0, 0.0, 0.0, vec![1.0]);
        map.observe_snapshot(&snapshot, 100);
        map.observe_snapshot(&snapshot, 200);

        let key = cell_key(1.0, 0.0, map.config.resolution_m);
        let cell = map.cells.get(&key).expect("endpoint cell should exist");
        assert!(cell.occupied_score > 0.5);
        assert!(cell.occupied_score > cell.free_score);
    }

    #[test]
    fn free_space_is_marked_along_beam_before_hit() {
        let mut map = LocalMap::new(MapConfig {
            resolution_m: 0.25,
            ..MapConfig::default()
        });
        let snapshot = snapshot_at(0.0, 0.0, 0.0, vec![1.0]);
        map.observe_snapshot(&snapshot, 100);

        let free_key = cell_key(0.5, 0.0, map.config.resolution_m);
        let free = map.cells.get(&free_key).expect("free cell should exist");
        assert!(free.free_score > free.occupied_score);
    }

    #[test]
    fn stale_observations_decay_and_empty_cells_are_removed() {
        let mut map = LocalMap::new(MapConfig {
            resolution_m: 0.5,
            decay_after_ms: 10,
            decay_per_tick: 1.0,
            ..MapConfig::default()
        });
        let snapshot = snapshot_at(0.0, 0.0, 0.0, vec![1.0]);
        map.observe_snapshot(&snapshot, 100);
        assert!(!map.cells.is_empty());

        map.decay_stale(111);
        assert!(map.cells.is_empty());
    }

    #[test]
    fn map_grows_as_odometry_moves_through_sim_like_snapshots() {
        let mut map = LocalMap::new(MapConfig {
            resolution_m: 0.25,
            ..MapConfig::default()
        });

        map.observe_snapshot(&snapshot_at(0.0, 0.0, 0.0, vec![1.0]), 100);
        let first_cells = map.cells.len();
        map.observe_snapshot(&snapshot_at(1.0, 0.0, 0.0, vec![1.0]), 200);

        assert!(first_cells > 0);
        assert!(map.cells.len() > first_cells);
        assert_eq!(map.pose_history.len(), 2);
        assert_eq!(map.pose_graph.nodes.len(), 2);
        assert_eq!(map.pose_graph.edges.len(), 1);
        assert!(matches!(
            map.pose_graph.edges[0].source,
            PoseEdgeSource::Odometry
        ));
        assert_eq!(map.submaps.len(), 2);
        assert_eq!(map.summary().remap.submaps, 2);
        assert_eq!(map.summary().label, MAP_LABEL);
    }

    #[test]
    fn remap_rebuilds_occupancy_from_optimized_submap_node_poses() {
        let mut map = LocalMap::new(MapConfig {
            resolution_m: 0.1,
            pose_graph_min_node_distance_m: 0.01,
            ..MapConfig::default()
        });
        map.observe_snapshot(&snapshot_at(0.0, 0.0, 0.0, vec![1.0]), 100);

        let original_key = cell_key(1.0, 0.0, map.config.resolution_m);
        assert!(map
            .cells
            .get(&original_key)
            .is_some_and(|cell| cell.occupied_score > cell.free_score));

        map.pose_graph.nodes[0].pose_estimate.pose.x_m = 0.5;
        map.rebuild_occupancy_from_submaps();

        let remapped_key = cell_key(1.5, 0.0, map.config.resolution_m);
        assert!(map
            .cells
            .get(&remapped_key)
            .is_some_and(|cell| cell.occupied_score > cell.free_score));
        assert!(map
            .cells
            .get(&original_key)
            .map_or(true, |cell| cell.occupied_score <= cell.free_score));
        assert_eq!(map.summary().remap.submaps, 1);
        assert!(map.summary().remap.generation >= 2);
    }

    #[test]
    fn scan_matching_corrects_small_odometry_drift_against_existing_occupancy() {
        let config = MapConfig {
            resolution_m: 0.1,
            scan_match_xy_window_m: 0.2,
            scan_match_theta_window_rad: 0.0,
            scan_match_min_occupied_cells: 1,
            scan_match_min_hit_beams: 1,
            pose_graph_min_node_distance_m: 0.01,
            pose_graph_max_ticks_between_nodes: 1,
            ..MapConfig::default()
        };
        let mut map = LocalMap::new(config);
        let observation = observation_from_parts(
            pose(0.0, 0.0, 0.0),
            0.75,
            &[1.0],
            Some(1.0),
            serde_json::json!({"frame_id":"seed"}),
            150,
            map.config,
        );
        map.integrate_observation(observation);
        for y in [-0.1, 0.0, 0.1] {
            let key = cell_key(1.0, y, map.config.resolution_m);
            map.cells.insert(
                key,
                OccupancyCell {
                    key,
                    occupied_score: 0.9,
                    free_score: 0.0,
                    confidence: 0.9,
                    last_seen_ms: 100,
                },
            );
        }

        let observation = observation_from_parts(
            pose(0.12, 0.0, 0.0),
            0.75,
            &[1.0],
            Some(1.0),
            serde_json::json!({"frame_id":"drifted"}),
            200,
            map.config,
        );
        map.integrate_observation(observation);

        let corrected = map.pose_history.last().unwrap();
        assert_eq!(corrected.source, "odometry+occupancy_scan_match");
        assert!(corrected.pose.x_m.abs() < 0.08);
        assert!(corrected.confidence > 0.75);
        assert_eq!(map.pose_graph.nodes.len(), 2);
        assert_eq!(map.pose_graph.edges.len(), 1);
        assert!(matches!(
            map.pose_graph.edges[0].source,
            PoseEdgeSource::ScanMatch { .. }
        ));
        assert_eq!(map.summary().scan_match_edges, 1);
    }

    #[test]
    fn live_loop_candidate_empty_path_preserves_scan_matched_behavior() {
        let config = MapConfig {
            resolution_m: 0.25,
            pose_graph_min_node_distance_m: 0.01,
            ..MapConfig::default()
        };
        let mut baseline = LocalMap::new(config);
        let mut candidate_aware = LocalMap::new(config);
        let first = observation_from_parts(
            pose(0.0, 0.0, 0.0),
            0.75,
            &[1.0],
            Some(1.0),
            serde_json::json!({"frame_id":"seed"}),
            100,
            config,
        );
        let second = observation_from_parts(
            pose(1.0, 0.0, 0.0),
            0.75,
            &[1.0],
            Some(1.0),
            serde_json::json!({"frame_id":"next"}),
            200,
            config,
        );

        baseline.integrate_observation(first.clone());
        baseline.integrate_observation(second.clone());
        candidate_aware.integrate_observation_with_loop_candidates(first, &[]);
        candidate_aware.integrate_observation_with_loop_candidates(second, &[]);

        assert_eq!(candidate_aware.pose_history, baseline.pose_history);
        assert_eq!(candidate_aware.pose_graph, baseline.pose_graph);
        assert_eq!(candidate_aware.summary().loop_closure_edges, 0);
    }

    #[test]
    fn live_loop_candidate_low_confidence_is_rejected_with_reason() {
        let config = live_loop_test_config();
        let mut map = seeded_live_loop_map(config);
        let weak = live_loop_candidate("entity_constellation", 0.60, "seed", "return");
        let observation = observation_from_parts(
            pose(0.05, 0.0, 0.0),
            0.75,
            &[1.0],
            Some(1.0),
            serde_json::json!({"frame_id":"return"}),
            300,
            config,
        );

        let summary = map.integrate_observation_with_loop_candidates(observation, &[weak]);

        assert_eq!(summary.loop_closure_edges, 1);
        assert_eq!(summary.loop_closures_accepted, 0);
        assert_eq!(summary.loop_closures_rejected, 1);
        let edge = map.pose_graph.edges.last().unwrap();
        assert!(!edge.active);
        assert!(edge
            .rejection_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("below gate")));
    }

    #[test]
    fn live_entity_constellation_candidate_adds_active_loop_edge() {
        let config = live_loop_test_config();
        let mut map = seeded_live_loop_map(config);
        let candidate = live_loop_candidate("entity_constellation", 0.94, "seed", "return");
        let observation = observation_from_parts(
            pose(0.05, 0.0, 0.0),
            0.75,
            &[1.0],
            Some(1.0),
            serde_json::json!({"frame_id":"return"}),
            300,
            config,
        );

        let summary = map.integrate_observation_with_loop_candidates(observation, &[candidate]);

        assert_eq!(summary.loop_closure_edges, 1);
        assert_eq!(summary.loop_closures_accepted, 1);
        assert_eq!(summary.loop_closures_rejected, 0);
        let edge = map.pose_graph.edges.last().unwrap();
        assert!(edge.active);
        assert_eq!(edge.to, "live-pose-0");
        assert!(matches!(
            edge.source,
            PoseEdgeSource::LoopClosureCandidate { ref kind, .. } if kind == "entity_constellation"
        ));
    }

    #[test]
    fn live_loop_rejections_explain_bad_targets_and_weak_geometry() {
        let config = live_loop_test_config();
        let mut map = seeded_live_loop_map(config);
        let current_target = live_loop_candidate("entity_constellation", 0.94, "return", "return");
        let weak_geometry = LoopClosureCandidateInput {
            target_frame_id: Some("seed".to_string()),
            source_frame_id: Some("return".to_string()),
            ..live_loop_candidate("entity_constellation", 0.94, "seed", "return")
        };
        let observation = observation_from_parts(
            pose(0.05, 0.0, 0.0),
            0.75,
            &[3.0],
            Some(3.0),
            serde_json::json!({"frame_id":"return"}),
            300,
            config,
        );

        map.integrate_observation_with_loop_candidates(
            observation,
            &[current_target, weak_geometry],
        );

        let reasons = map
            .pose_graph
            .edges
            .iter()
            .filter_map(|edge| edge.rejection_reason.as_deref())
            .collect::<Vec<_>>();
        assert!(reasons
            .iter()
            .any(|reason| reason.contains("current/source frame")));
        assert!(reasons
            .iter()
            .any(|reason| reason.contains("geometric occupancy agreement")));
    }

    #[test]
    fn live_loop_candidate_rebuilds_occupancy_after_optimization() {
        let config = live_loop_test_config();
        let mut map = seeded_live_loop_map(config);
        let generation_before = map.remap_summary.generation;
        let candidate = live_loop_candidate("entity_constellation", 0.94, "seed", "return");
        let observation = observation_from_parts(
            pose(0.05, 0.0, 0.0),
            0.75,
            &[1.0],
            Some(1.0),
            serde_json::json!({"frame_id":"return"}),
            300,
            config,
        );

        let summary = map.integrate_observation_with_loop_candidates(observation, &[candidate]);

        assert!(summary.pose_graph_optimization.active_edges >= 2);
        assert_eq!(summary.remap.submaps, map.submaps.len());
        assert!(summary.remap.generation > generation_before);
        assert!(!map.cells.is_empty());
    }

    #[test]
    fn kinect_point_transforms_into_odometry_world_frame() {
        let config = PointCloudConfig {
            voxel_size_m: 0.1,
            camera_height_m: 0.2,
            ..PointCloudConfig::default()
        };
        let point = Point3D {
            x_m: 0.0,
            y_m: 0.0,
            z_m: 1.0,
        };
        let world = transform_point_to_world(
            point,
            PointCloudFrame::KinectCamera,
            pose(1.0, 2.0, std::f32::consts::FRAC_PI_2),
            OrientationEstimate {
                yaw_rad: Some(std::f32::consts::FRAC_PI_2),
                yaw_source: YawSource::OdometryHeading,
                ..OrientationEstimate::default()
            },
            config,
        );
        assert!((world.x_m - 1.0).abs() < 0.001);
        assert!((world.y_m - 3.0).abs() < 0.001);
        assert!((world.z_m - 0.2).abs() < 0.001);
    }

    #[test]
    fn camera_to_robot_zero_rotation_maps_forward_and_height() {
        let config = PointCloudConfig {
            camera_height_m: 0.25,
            camera_forward_m: 0.10,
            ..PointCloudConfig::default()
        };

        let robot = camera_point_to_robot(
            Point3D {
                x_m: 0.0,
                y_m: 0.0,
                z_m: 1.0,
            },
            config,
        );

        assert!((robot.x_m - 1.10).abs() < 0.001);
        assert!(robot.y_m.abs() < 0.001);
        assert!((robot.z_m - 0.25).abs() < 0.001);
    }

    #[test]
    fn plausible_floor_ray_lands_near_robot_floor() {
        let config = PointCloudConfig {
            camera_height_m: 0.5,
            ..PointCloudConfig::default()
        };
        let robot = camera_point_to_robot(
            Point3D {
                x_m: 0.0,
                y_m: 0.5,
                z_m: 1.0,
            },
            config,
        );

        assert!(robot.x_m > 0.9);
        assert!(robot.y_m.abs() < 0.001);
        assert!(robot.z_m.abs() < 0.001);
    }

    #[test]
    fn kinect_point_transform_applies_camera_pitch_before_world_yaw() {
        let config = PointCloudConfig {
            camera_height_m: 0.5,
            camera_pitch_rad: 0.25,
            ..PointCloudConfig::default()
        };
        let point = Point3D {
            x_m: 0.0,
            y_m: 0.0,
            z_m: 2.0,
        };

        let robot = camera_point_to_robot(point, config);
        let world = transform_point_to_world(
            point,
            PointCloudFrame::KinectCamera,
            pose(0.0, 0.0, std::f32::consts::FRAC_PI_2),
            OrientationEstimate {
                yaw_rad: Some(std::f32::consts::FRAC_PI_2),
                yaw_source: YawSource::OdometryHeading,
                ..OrientationEstimate::default()
            },
            config,
        );

        assert!(robot.z_m < 0.5);
        assert!((world.x_m + robot.y_m).abs() < 0.001);
        assert!((world.y_m - robot.x_m).abs() < 0.001);
        assert!((world.z_m - robot.z_m).abs() < 0.001);
    }

    #[test]
    fn positive_pitch_lowers_straight_ahead_points() {
        let zero = camera_point_to_robot(
            Point3D {
                x_m: 0.0,
                y_m: 0.0,
                z_m: 1.0,
            },
            PointCloudConfig {
                camera_height_m: 0.5,
                ..PointCloudConfig::default()
            },
        );
        let pitched = camera_point_to_robot(
            Point3D {
                x_m: 0.0,
                y_m: 0.0,
                z_m: 1.0,
            },
            PointCloudConfig {
                camera_height_m: 0.5,
                camera_pitch_rad: 10.0_f32.to_radians(),
                ..PointCloudConfig::default()
            },
        );

        assert!(pitched.z_m < zero.z_m);
        assert!(pitched.x_m < zero.x_m);
    }

    #[test]
    fn positive_roll_raises_left_floor_relative_to_right() {
        let config = PointCloudConfig {
            camera_height_m: 0.5,
            camera_roll_rad: 10.0_f32.to_radians(),
            ..PointCloudConfig::default()
        };
        let left = camera_point_to_robot(
            Point3D {
                x_m: -0.25,
                y_m: 0.5,
                z_m: 1.0,
            },
            config,
        );
        let right = camera_point_to_robot(
            Point3D {
                x_m: 0.25,
                y_m: 0.5,
                z_m: 1.0,
            },
            config,
        );

        assert!(left.y_m > 0.0);
        assert!(right.y_m < 0.0);
        assert!(left.z_m > right.z_m);
    }

    #[test]
    fn imu_roll_pitch_correction_changes_world_height_before_yaw() {
        let config = PointCloudConfig {
            camera_height_m: 0.5,
            ..PointCloudConfig::default()
        };
        let point = Point3D {
            x_m: 0.0,
            y_m: 0.0,
            z_m: 1.0,
        };
        let uncorrected = transform_point_to_world(
            point,
            PointCloudFrame::KinectCamera,
            pose(0.0, 0.0, 0.0),
            OrientationEstimate {
                yaw_rad: Some(0.0),
                yaw_source: YawSource::OdometryHeading,
                ..OrientationEstimate::default()
            },
            config,
        );
        let corrected = transform_point_to_world(
            point,
            PointCloudFrame::KinectCamera,
            pose(0.0, 0.0, 0.0),
            OrientationEstimate {
                pitch_rad: Some(10.0_f32.to_radians()),
                yaw_rad: Some(0.0),
                roll_pitch_from_imu: true,
                yaw_source: YawSource::OdometryHeading,
                ..OrientationEstimate::default()
            },
            config,
        );

        assert!(corrected.z_m < uncorrected.z_m);
        assert!(corrected.x_m > uncorrected.x_m);
    }

    #[test]
    fn imu_orientation_contract_handles_hardware_sim_and_legacy_shapes() {
        let hardware = orientation_from_imu(
            &ImuSense {
                orientation: vec![0.1, -0.2],
                ..ImuSense::default()
            },
            0.7,
        );
        assert_eq!(hardware.roll_rad, Some(0.1));
        assert_eq!(hardware.pitch_rad, Some(-0.2));
        assert_eq!(hardware.yaw_rad, Some(0.7));
        assert_eq!(hardware.yaw_source, YawSource::OdometryHeading);
        assert!(hardware.roll_pitch_from_imu);

        let sim = orientation_from_imu(
            &ImuSense {
                orientation: vec![0.0, 0.0, 1.2],
                ..ImuSense::default()
            },
            0.7,
        );
        assert_eq!(sim.roll_rad, Some(0.0));
        assert_eq!(sim.pitch_rad, Some(0.0));
        assert_eq!(sim.yaw_rad, Some(1.2));
        assert_eq!(sim.yaw_source, YawSource::ImuOrientation);

        let legacy_heading_only = orientation_from_imu(
            &ImuSense {
                orientation: vec![1.4],
                ..ImuSense::default()
            },
            0.7,
        );
        assert_eq!(legacy_heading_only.roll_rad, None);
        assert_eq!(legacy_heading_only.pitch_rad, None);
        assert_eq!(legacy_heading_only.yaw_rad, Some(1.4));
        assert_eq!(legacy_heading_only.yaw_source, YawSource::ImuOrientation);
        assert!(!legacy_heading_only.roll_pitch_from_imu);
    }

    #[test]
    fn implausible_gravity_roll_pitch_is_not_applied_to_world_cloud() {
        let orientation = orientation_from_imu(
            &ImuSense {
                orientation: vec![120.0_f32.to_radians(), 62.0_f32.to_radians()],
                ..ImuSense::default()
            },
            0.3,
        );

        assert_eq!(orientation.roll_rad, None);
        assert_eq!(orientation.pitch_rad, None);
        assert_eq!(orientation.yaw_rad, Some(0.3));
        assert_eq!(orientation.yaw_source, YawSource::OdometryHeading);
        assert!(!orientation.roll_pitch_from_imu);
    }

    #[test]
    fn voxel_cloud_merges_points_and_marks_stable() {
        let mut cloud = VoxelPointCloud::new(PointCloudConfig {
            voxel_size_m: 0.25,
            stable_seen_count: 2,
            stable_confidence: 0.2,
            ..PointCloudConfig::default()
        });
        let snapshot = kinect_snapshot_at(0.0, 0.0, 0.0, vec![1.0], 1, 1);

        cloud.observe_snapshot(&snapshot, 100);
        cloud.observe_snapshot(&snapshot, 200);

        assert_eq!(cloud.voxels.len(), 1);
        let voxel = cloud.voxels.values().next().unwrap();
        assert!(voxel.stable);
        assert_eq!(voxel.seen_count, 2);
        assert!(voxel.confidence > 0.2);
    }

    #[test]
    fn stationary_rotate_world_frame_observations_merge_into_stable_belief() {
        let mut cloud = VoxelPointCloud::new(PointCloudConfig {
            voxel_size_m: 0.25,
            stable_seen_count: 3,
            stable_confidence: 0.2,
            ..PointCloudConfig::default()
        });

        for (t_ms, heading_rad) in [
            (100, 0.0),
            (200, std::f32::consts::FRAC_PI_2),
            (300, std::f32::consts::PI),
        ] {
            cloud.integrate_observation(PointCloudObservation {
                frame: PointCloudFrame::OdometryWorld,
                pose: PoseEstimate {
                    pose: pose(0.0, 0.0, heading_rad),
                    confidence: 0.9,
                    covariance: [0.01, 0.01, 0.02],
                    source: "rotate-test".to_string(),
                    t_ms,
                },
                orientation: OrientationEstimate {
                    roll_rad: Some(0.02),
                    pitch_rad: Some(-0.01),
                    yaw_rad: Some(heading_rad),
                    roll_pitch_from_imu: true,
                    yaw_source: YawSource::OdometryHeading,
                },
                points: vec![PointCloudPoint {
                    position: Point3D {
                        x_m: 1.0,
                        y_m: 0.0,
                        z_m: 0.4,
                    },
                    color_rgb: None,
                    confidence: 1.0,
                }],
                source: "rotate-test".to_string(),
                t_ms,
                metadata: serde_json::json!({}),
            });
        }

        assert_eq!(cloud.voxels.len(), 1);
        let voxel = cloud.voxels.values().next().unwrap();
        assert!(voxel.stable);
        assert!((voxel.position.x_m - 1.0).abs() < 0.001);
        assert!((voxel.position.y_m - 0.0).abs() < 0.001);
        assert_eq!(
            cloud.orientation_status.yaw_source,
            YawSource::OdometryHeading
        );
        assert!(cloud.orientation_status.roll_pitch_corrected);
    }

    #[test]
    fn local_world_belief_clusters_stable_voxels_into_surface_hypotheses() {
        let mut cloud = VoxelPointCloud::new(PointCloudConfig {
            voxel_size_m: 0.1,
            stable_seen_count: 1,
            stable_confidence: 0.1,
            ..PointCloudConfig::default()
        });
        for y in [0.0, 0.1, 0.2, 0.3] {
            for z in [0.1, 0.2, 0.3, 0.4] {
                cloud.integrate_observation(PointCloudObservation {
                    frame: PointCloudFrame::OdometryWorld,
                    pose: PoseEstimate {
                        pose: pose(0.0, 0.0, 0.0),
                        confidence: 0.9,
                        covariance: [0.01, 0.01, 0.02],
                        source: "surface-test".to_string(),
                        t_ms: 100,
                    },
                    orientation: OrientationEstimate {
                        yaw_rad: Some(0.0),
                        yaw_source: YawSource::OdometryHeading,
                        ..OrientationEstimate::default()
                    },
                    points: vec![PointCloudPoint {
                        position: Point3D {
                            x_m: 1.0,
                            y_m: y,
                            z_m: z,
                        },
                        color_rgb: None,
                        confidence: 1.0,
                    }],
                    source: "surface-test".to_string(),
                    t_ms: 100,
                    metadata: serde_json::json!({}),
                });
            }
        }

        let belief = cloud.local_world_belief();
        assert_eq!(belief.stable_voxels, 16);
        assert!(belief
            .stable_surfaces
            .iter()
            .any(|surface| surface.kind == WorldSurfaceKind::WallLike));
        assert!(belief.stable_blobs.is_empty());
    }

    #[test]
    fn voxel_cloud_ages_transient_points_and_bounds_growth() {
        let mut cloud = VoxelPointCloud::new(PointCloudConfig {
            voxel_size_m: 0.1,
            max_voxels: 2,
            decay_after_ms: 10,
            decay_per_tick: 0.05,
            transient_after_ms: 20,
            ..PointCloudConfig::default()
        });
        cloud.observe_snapshot(&kinect_snapshot_at(0.0, 0.0, 0.0, vec![1.0], 1, 1), 100);
        cloud.observe_snapshot(&kinect_snapshot_at(1.0, 0.0, 0.0, vec![1.0], 1, 1), 110);
        cloud.observe_snapshot(&kinect_snapshot_at(2.0, 0.0, 0.0, vec![1.0], 1, 1), 120);

        assert_eq!(cloud.voxels.len(), 2);
        cloud.decay_stale(200);
        assert!(cloud.voxels.values().any(|voxel| voxel.transient));
    }

    #[test]
    fn pose_graph_adds_nodes_by_motion_and_odometry_edges() {
        let mut builder = PoseGraphBuilder::new(PoseGraphConfig {
            min_node_distance_m: 0.5,
            min_node_heading_delta_rad: 0.5,
            max_ticks_between_nodes: 10,
            ..PoseGraphConfig::default()
        });
        builder.observe(pose(0.0, 0.0, 0.0), 100, Some("frame-a".to_string()), &[]);
        builder.observe(pose(0.2, 0.0, 0.0), 200, Some("frame-b".to_string()), &[]);
        builder.observe(pose(0.6, 0.0, 0.0), 300, Some("frame-c".to_string()), &[]);

        let graph = builder.finish();
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
        assert!(matches!(graph.edges[0].source, PoseEdgeSource::Odometry));
        assert_eq!(graph.edges[0].from, "pose-0");
        assert_eq!(graph.edges[0].to, "pose-1");
        assert!((graph.edges[0].transform.x_m - 0.6).abs() < 0.001);
    }

    #[test]
    fn pose_graph_gates_loop_candidates_and_reports_rejections() {
        let mut builder = PoseGraphBuilder::new(PoseGraphConfig {
            min_node_distance_m: 0.5,
            min_loop_confidence: 0.85,
            loop_target_max_distance_m: 0.5,
            ..PoseGraphConfig::default()
        });
        builder.observe(pose(0.0, 0.0, 0.0), 100, Some("frame-a".to_string()), &[]);
        builder.observe(pose(1.0, 0.0, 0.0), 200, Some("frame-b".to_string()), &[]);

        let accepted = LoopClosureCandidateInput {
            target_pose: pose(0.0, 0.0, 0.0),
            confidence: 0.93,
            similarity: 0.94,
            kind: "same_place".to_string(),
            target_frame_id: Some("frame-a".to_string()),
            source_frame_id: Some("frame-a".to_string()),
            source_experience_id: Some("experience-a".to_string()),
            source_instant_frame_id: Some("frame-a".to_string()),
            source_vector_refs: vec!["teacher:a".to_string()],
            source_vector_id: Some("scene-a".to_string()),
            query_vector_id: Some("scene-b".to_string()),
            query_experience_id: Some("experience-b".to_string()),
        };
        let rejected = LoopClosureCandidateInput {
            confidence: 0.60,
            query_vector_id: Some("weak-scene".to_string()),
            ..accepted.clone()
        };
        builder.observe(
            pose(1.0, 0.0, 0.0),
            300,
            Some("frame-c".to_string()),
            &[accepted, rejected],
        );

        let report = builder.finish_report();
        assert_eq!(report.nodes, 2);
        assert_eq!(report.odometry_edges, 1);
        assert_eq!(report.loop_candidate_edges, 2);
        assert_eq!(report.active_loop_candidate_edges, 1);
        assert_eq!(report.rejected_loop_candidates, 1);
        assert_eq!(report.confidence_distribution.buckets["0.85-0.94"], 1);
        assert_eq!(report.confidence_distribution.buckets["0.50-0.69"], 1);
        assert!(report.rejected_candidates[0].reason.contains("below gate"));
    }

    #[test]
    fn pose_graph_optimizer_reduces_loop_closure_error() {
        let mut graph = PoseGraph {
            nodes: vec![
                test_pose_node("pose-0", 0.0, 0),
                test_pose_node("pose-1", 1.2, 100),
                test_pose_node("pose-2", 2.4, 200),
            ],
            edges: vec![
                test_edge(
                    "pose-0",
                    "pose-1",
                    pose(1.0, 0.0, 0.0),
                    PoseEdgeSource::Odometry,
                    0.7,
                ),
                test_edge(
                    "pose-1",
                    "pose-2",
                    pose(1.0, 0.0, 0.0),
                    PoseEdgeSource::Odometry,
                    0.7,
                ),
                test_edge(
                    "pose-0",
                    "pose-2",
                    pose(2.0, 0.0, 0.0),
                    PoseEdgeSource::LoopClosureCandidate {
                        kind: "same_place_geometry".to_string(),
                        target_frame_id: Some("pose-2".to_string()),
                        source_frame_id: Some("pose-0".to_string()),
                        source_experience_id: None,
                        source_instant_frame_id: None,
                        source_vector_refs: Vec::new(),
                        source_vector_id: None,
                        query_vector_id: None,
                        query_experience_id: None,
                    },
                    0.95,
                ),
            ],
        };

        let summary = graph.optimize_anchored(PoseGraphOptimizationConfig {
            iterations: 30,
            step_size: 0.6,
            ..PoseGraphOptimizationConfig::default()
        });

        assert!(summary.final_mean_error < summary.initial_mean_error);
        assert!(graph.nodes[2].pose_estimate.pose.x_m < 2.4);
        assert_eq!(graph.nodes[0].pose_estimate.pose.x_m, 0.0);
        assert!(summary.active_edges >= 3);
    }

    fn pose(x_m: f32, y_m: f32, heading_rad: f32) -> Pose2 {
        Pose2 {
            x_m,
            y_m,
            heading_rad,
        }
    }

    fn test_pose_node(id: &str, x_m: f32, t_ms: TimeMs) -> PoseNode {
        PoseNode {
            id: id.to_string(),
            pose_estimate: PoseEstimate {
                pose: pose(x_m, 0.0, 0.0),
                confidence: 0.8,
                covariance: [0.05, 0.05, 0.1],
                source: "test".to_string(),
                t_ms,
            },
            t_ms,
            source_frame_id: Some(id.to_string()),
        }
    }

    fn test_edge(
        from: &str,
        to: &str,
        transform: Pose2,
        source: PoseEdgeSource,
        confidence: f32,
    ) -> PoseEdge {
        PoseEdge {
            from: from.to_string(),
            to: to.to_string(),
            transform,
            covariance: [0.05, 0.05, 0.08],
            confidence,
            source,
            active: true,
            rejection_reason: None,
        }
    }

    fn live_loop_test_config() -> MapConfig {
        MapConfig {
            resolution_m: 0.1,
            scan_match_enabled: false,
            pose_graph_min_node_distance_m: 0.01,
            pose_graph_max_ticks_between_nodes: 1,
            pose_graph_optimize_iterations: 8,
            pose_graph_min_loop_confidence: 0.85,
            pose_graph_loop_target_max_distance_m: 0.75,
            pose_graph_loop_min_geometric_overlap: 0.40,
            ..MapConfig::default()
        }
    }

    fn seeded_live_loop_map(config: MapConfig) -> LocalMap {
        let mut map = LocalMap::new(config);
        map.integrate_observation(observation_from_parts(
            pose(0.0, 0.0, 0.0),
            0.75,
            &[1.0],
            Some(1.0),
            serde_json::json!({"frame_id":"seed"}),
            100,
            config,
        ));
        map.integrate_observation(observation_from_parts(
            pose(1.0, 0.0, 0.0),
            0.75,
            &[1.0],
            Some(1.0),
            serde_json::json!({"frame_id":"away"}),
            200,
            config,
        ));
        map
    }

    fn live_loop_candidate(
        kind: &str,
        confidence: f32,
        target_frame_id: &str,
        source_frame_id: &str,
    ) -> LoopClosureCandidateInput {
        LoopClosureCandidateInput {
            target_pose: pose(0.0, 0.0, 0.0),
            confidence,
            similarity: confidence,
            kind: kind.to_string(),
            target_frame_id: Some(target_frame_id.to_string()),
            source_frame_id: Some(source_frame_id.to_string()),
            source_experience_id: Some("experience-seed".to_string()),
            source_instant_frame_id: Some(target_frame_id.to_string()),
            source_vector_refs: vec!["entity:charger".to_string()],
            source_vector_id: Some("constellation-seed".to_string()),
            query_vector_id: Some("constellation-return".to_string()),
            query_experience_id: Some("experience-return".to_string()),
        }
    }
}
