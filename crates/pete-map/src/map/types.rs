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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_uv: Option<[u32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_image_size: Option<[u32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_frame_id: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_kinect_capture_ms: Option<TimeMs>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_range_capture_ms: Option<TimeMs>,
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
    #[serde(default)]
    pub slam_status: SlamStatus,
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

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SlamStatus {
    pub mode: SlamMode,
    pub local_scan_matching_active: bool,
    pub loop_closure_active: bool,
    pub pose_graph_optimized: bool,
    pub occupancy_remapped_from_pose_graph: bool,
    pub reasons: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlamMode {
    #[default]
    OdometryOnly,
    MappingOnly,
    LocalScanMatched,
    LoopClosedPoseGraph,
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        registration: Option<LoopRegistrationMeasurement>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LoopRegistrationMeasurement {
    pub algorithm: String,
    pub registered_pose: Pose2,
    pub score: f32,
    pub odometry_score: f32,
    pub geometric_overlap: f32,
    pub odometry_geometric_overlap: f32,
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
            voxel_size_m: 0.02,
            max_voxels: 200_000,
            max_points_per_observation: 12_000,
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
