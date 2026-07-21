#[derive(Clone, Debug, Serialize)]
pub struct LiveSnapshotResponse {
    pub t_ms: TimeMs,
    pub body: pete_body::BodySense,
    pub range: pete_now::RangeSense,
    pub eye_frame: Option<pete_sensors::EyeFrame>,
    pub gps: Option<pete_now::GpsSense>,
    pub ear_pcm: Option<pete_sensors::PcmAudioFrame>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LiveSceneMetadata {
    pub arena: Option<SceneArena>,
    #[serde(default)]
    pub objects: Vec<SceneObject>,
    pub sensor_calibration: Option<SceneSensorCalibration>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneSensorCalibration {
    pub compact_depth_beam_count: usize,
    pub compact_depth_fov_rad: f32,
    pub depth_scale: f32,
    pub point_y_m: f32,
    #[serde(default)]
    pub depth_forward_offset_m: f32,
    #[serde(default)]
    pub depth_pitch_down_rad: f32,
    #[serde(default)]
    pub camera_forward_m: f32,
    #[serde(default)]
    pub camera_height_m: f32,
    #[serde(default)]
    pub camera_pitch_rad: f32,
    #[serde(default)]
    pub camera_roll_rad: f32,
    #[serde(default)]
    pub camera_yaw_rad: f32,
    #[serde(default)]
    pub color_offset_x_px: i32,
    #[serde(default)]
    pub color_offset_y_px: i32,
}

impl SceneSensorCalibration {
    pub fn sim_default() -> Self {
        Self {
            compact_depth_beam_count: 32,
            compact_depth_fov_rad: std::f32::consts::PI * 0.75,
            depth_scale: 1.0,
            point_y_m: 0.18,
            depth_forward_offset_m: 0.0,
            depth_pitch_down_rad: 0.0,
            camera_forward_m: 0.0,
            camera_height_m: 0.18,
            camera_pitch_rad: 0.0,
            camera_roll_rad: 0.0,
            camera_yaw_rad: 0.0,
            color_offset_x_px: 3,
            color_offset_y_px: 7,
        }
    }

    fn depth_camera_forward_m(self) -> f32 {
        if self.camera_forward_m != 0.0 {
            self.camera_forward_m
        } else {
            self.depth_forward_offset_m
        }
    }

    fn depth_camera_height_m(self) -> f32 {
        if self.camera_height_m != 0.0 {
            self.camera_height_m
        } else {
            self.point_y_m
        }
    }

    fn depth_camera_pitch_rad(self) -> f32 {
        if self.camera_pitch_rad != 0.0 {
            self.camera_pitch_rad
        } else {
            self.depth_pitch_down_rad
        }
    }

    fn color_offset_px(self) -> (i32, i32) {
        (self.color_offset_x_px, self.color_offset_y_px)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneArena {
    pub width_m: f32,
    pub height_m: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneObject {
    pub id: String,
    pub kind: String,
    pub x_m: f32,
    pub y_m: f32,
    pub radius_m: f32,
    pub label: Option<String>,
    pub color_rgb: Option<[u8; 3]>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LiveSceneResponse {
    pub schema_version: u32,
    pub session: Option<SceneSession>,
    pub training: LiveTrainingStatus,
    pub hardware_control: HardwareControlStatus,
    pub training_mode: String,
    pub ledger_path: Option<String>,
    pub frames_written: usize,
    pub transitions_written: usize,
    pub models_loaded: Vec<String>,
    pub model_modes: HashMap<String, String>,
    pub behavior_nodes: Vec<BehaviorNodeState>,
    pub action_selector_mode: String,
    pub weights_updating: bool,
    pub t_ms: TimeMs,
    pub body: SceneBody,
    pub range: SceneRange,
    pub eye: Option<SceneEye>,
    pub kinect: SceneKinect,
    pub imu: SceneImuDebug,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub surface_perception: Option<SceneSurfacePerception>,
    pub world_belief_layers: Vec<&'static str>,
    pub audio: Option<SceneAudio>,
    pub objects: Vec<SceneObject>,
    pub arena: Option<SceneArena>,
    pub sensor_calibration: Option<SceneSensorCalibration>,
    pub action: SceneAction,
    pub prod: SceneProd,
    pub idle_ms: u64,
    pub last_nudge_ms: Option<u64>,
    pub nudge_count_recent: u32,
    pub nudge_blocked_reason: Option<String>,
    pub active_nudge: bool,
    pub stuck: bool,
    pub dead_battery: bool,
    pub recovery_mode: Option<String>,
    pub stuck_ticks: usize,
    pub stuck_detail: SceneStuck,
    pub mind: SceneMind,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneSession {
    pub mode: String,
    pub scenario: Option<String>,
    pub seed: Option<u64>,
    pub source: String,
    pub tick_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LiveTrainingStatus {
    pub training_mode: String,
    pub ledger_path: Option<String>,
    pub frames_written: usize,
    pub transitions_written: usize,
    pub models_loaded: Vec<String>,
    pub model_modes: HashMap<String, String>,
    pub action_selector_mode: String,
    pub weights_updating: bool,
}

impl Default for LiveTrainingStatus {
    fn default() -> Self {
        Self {
            training_mode: "none".to_string(),
            ledger_path: None,
            frames_written: 0,
            transitions_written: 0,
            models_loaded: Vec::new(),
            model_modes: HashMap::new(),
            action_selector_mode: "goal".to_string(),
            weights_updating: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SceneBody {
    pub x_m: f32,
    pub y_m: f32,
    pub heading_rad: f32,
    pub battery_level: f32,
    pub charging: bool,
    pub bump_left: bool,
    pub bump_right: bool,
    pub cliff: bool,
    pub wheel_drop: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SceneRange {
    pub nearest_m: Option<f32>,
    pub beams: Vec<SceneRangeBeam>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SceneRangeBeam {
    pub angle_rad: f32,
    pub distance_m: f32,
    pub hit: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LiveMapResponse {
    pub schema_version: u32,
    pub label: &'static str,
    pub summary: MapSummary,
    pub overlays: Vec<&'static str>,
    pub pose_trail: Vec<MapPosePoint>,
    pub current_pose: Option<MapPosePoint>,
    pub range_beams: Vec<MapProjectedBeam>,
    pub cells: Vec<MapViewCell>,
    pub world_projection: MapWorldProjection,
    pub semantic_cells: Vec<MapSemanticCell>,
    pub events: Vec<MapEventMarker>,
    pub pose_graph: MapPoseGraphSummary,
    pub remap: RemapSummary,
    pub entity_graph: MapEntityGraph,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapWorldProjection {
    pub label: &'static str,
    pub source: &'static str,
    pub coordinate_frame: &'static str,
    pub resolution_m: f32,
    pub aligned_with_3d: bool,
    pub geometry_trusted: bool,
    pub navigation_trusted: bool,
    pub reasons: Vec<String>,
    pub source_voxels: usize,
    pub projected_cells: usize,
    pub stable_cells: usize,
    pub cells: Vec<MapWorldProjectionCell>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapWorldProjectionCell {
    pub x: i32,
    pub y: i32,
    pub center_x_m: f32,
    pub center_y_m: f32,
    pub confidence: f32,
    pub age_ms: TimeMs,
    pub voxel_count: usize,
    pub stable: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapPoseGraphSummary {
    pub nodes: usize,
    pub edges: usize,
    pub odometry_edges: usize,
    pub scan_match_edges: usize,
    pub loop_candidate_edges: usize,
    pub loop_candidate_active_edges: usize,
    pub loop_candidate_rejected_edges: usize,
    pub loop_candidate_rejection_reasons: Vec<String>,
    pub latest_node_id: Option<String>,
    pub latest_edge_source: Option<String>,
    pub optimization: PoseGraphOptimizationSummary,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapPosePoint {
    pub x_m: f32,
    pub y_m: f32,
    pub heading_rad: f32,
    pub confidence: f32,
    pub t_ms: TimeMs,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapProjectedBeam {
    pub origin_x_m: f32,
    pub origin_y_m: f32,
    pub end_x_m: f32,
    pub end_y_m: f32,
    pub angle_rad: f32,
    pub distance_m: f32,
    pub hit: bool,
    pub confidence: f32,
    pub age_ms: TimeMs,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapViewCell {
    pub x: i32,
    pub y: i32,
    pub center_x_m: f32,
    pub center_y_m: f32,
    pub occupied_score: f32,
    pub free_score: f32,
    pub confidence: f32,
    pub age_ms: TimeMs,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapSemanticCell {
    pub x_m: f32,
    pub y_m: f32,
    pub kind: String,
    pub score: f32,
    pub confidence: f32,
    pub age_ms: Option<TimeMs>,
    pub label: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapEventMarker {
    pub x_m: f32,
    pub y_m: f32,
    pub kind: String,
    pub confidence: f32,
    pub age_ms: TimeMs,
    pub label: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapEntityGraph {
    pub schema_version: u32,
    pub generated_from: &'static str,
    pub nodes: Vec<MapEntityGraphNode>,
    pub edges: Vec<MapEntityGraphEdge>,
    pub events: Vec<MapEntityGraphEvent>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapEntityGraphNode {
    pub id: String,
    pub node_type: String,
    pub label: String,
    pub modality: Option<String>,
    pub x_m: Option<f32>,
    pub y_m: Option<f32>,
    pub confidence: f32,
    pub age_ms: TimeMs,
    pub source_channel: Option<String>,
    pub observed_at_ms: Option<TimeMs>,
    pub vector_shape: Option<String>,
    pub nearest_cluster: Option<String>,
    pub attached_text: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapEntityGraphEdge {
    pub id: String,
    pub from: String,
    pub to: String,
    pub edge_type: String,
    pub confidence: f32,
    pub observed_at_ms: Option<TimeMs>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapEntityGraphEvent {
    pub t_ms: TimeMs,
    pub node_id: String,
    pub event_type: String,
    pub label: String,
    pub confidence: f32,
    pub timestamp_ms: Option<TimeMs>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SceneEye {
    pub width: u32,
    pub height: u32,
    pub format: String,
    pub data_url: Option<String>,
    pub mean_luma: f32,
    pub non_background_ratio: f32,
    pub source: String,
    pub authoritative: bool,
    pub retina_connected: bool,
    pub retina_last_frame_age_ms: Option<u64>,
    pub frames_received: usize,
    pub frames_written_to_ledger: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SceneKinect {
    pub points: Vec<ScenePoint>,
    #[serde(default)]
    pub accumulated_points: Vec<SceneAccumulatedPoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accumulated_summary: Option<PointCloudSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub local_world_belief: Option<LocalWorldBelief>,
    pub skeletons: Vec<KinectSkeletonSense>,
    pub diagnostics: SceneKinectDiagnostics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coordinate_system: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SceneImuDebug {
    pub raw_orientation: Vec<f32>,
    pub assumed_units: String,
    pub assumed_axis_order: String,
    pub roll_deg: Option<f32>,
    pub pitch_deg: Option<f32>,
    pub yaw_deg: Option<f32>,
    pub roll_pitch_correction_active: bool,
    pub yaw_source: String,
    pub contract_known: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SceneKinectDiagnostics {
    pub depth_width: u32,
    pub depth_height: u32,
    pub valid_depth_count: usize,
    pub skipped_depth_count: usize,
    pub clipped_depth_count: usize,
    pub min_depth_m: Option<f32>,
    pub median_depth_m: Option<f32>,
    pub max_depth_m: Option<f32>,
    pub sample_stride: usize,
    pub coordinate_system: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub point_coordinate_system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub math_frame: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub render_frame: Option<String>,
    pub below_floor_count: usize,
    pub below_floor_ratio: f32,
    pub min_z_m: Option<f32>,
    pub median_z_m: Option<f32>,
    pub min_math_z_m: Option<f32>,
    pub median_math_z_m: Option<f32>,
    pub min_render_vertical_m: Option<f32>,
    pub median_render_vertical_m: Option<f32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SceneSurfacePerception {
    pub diagnostics: SurfaceExtractorDiagnostics,
    pub plane_observations: Vec<PlaneObservation>,
    pub stable_surfaces: Vec<SurfaceHypothesis>,
    pub floor: Option<SurfaceTrack>,
    pub obstacle_grid: OccupancyGrid,
    pub clusters: Vec<ClusterObservation>,
    pub scene_graph: SceneGraphSummary,
}

impl From<pete_sensors::SurfaceExtractorOutput> for SceneSurfacePerception {
    fn from(output: pete_sensors::SurfaceExtractorOutput) -> Self {
        Self {
            diagnostics: output.diagnostics,
            plane_observations: output.plane_observations,
            stable_surfaces: output.stable_surfaces,
            floor: output.floor,
            obstacle_grid: output.obstacle_grid,
            clusters: output.clusters,
            scene_graph: output.scene_graph,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ScenePoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SceneAccumulatedPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub confidence: f32,
    pub age_ms: TimeMs,
    pub stable: bool,
    pub transient: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SceneAudio {
    pub bearing_rad: Option<f32>,
    pub energy: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SceneAction {
    pub latest: Option<String>,
    pub desired_motor: Option<MotorCommand>,
    pub final_motor: Option<MotorCommand>,
    pub motion_sent: Option<MotionCommand>,
    pub motor_applied: Option<bool>,
    pub movement_delta: Option<f32>,
    pub safety_override: bool,
    pub not_moving_reason: Option<String>,
    pub latest_llm_proposed_action: Option<ActionPrimitive>,
    pub latest_llm_advisory_action: Option<LlmAdvisoryAction>,
    pub llm_action_accepted: Option<bool>,
    pub llm_action_safety_vetoed: Option<bool>,
    pub final_selected_action: Option<ActionPrimitive>,
    pub llm_action_ignored_reason: Option<String>,
    pub llm_action_safety_reason: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SceneProd {
    pub idle_ms: u64,
    pub last_nudge_ms: Option<u64>,
    pub nudge_count_recent: u32,
    pub nudge_blocked_reason: Option<String>,
    pub active_nudge: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SceneStuck {
    pub active: bool,
    pub class: Option<String>,
    pub trap_kind: Option<String>,
    pub stuck_ticks: usize,
    pub duration_ms: u64,
    pub recovery_phase: Option<String>,
    pub turn_direction: Option<String>,
    pub recovery_attempts: usize,
    pub repeated_trap_count: usize,
    pub clearance_m: Option<f32>,
    pub event_started: bool,
    pub recovered: bool,
    pub dead_battery: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SceneMind {
    pub combobulation: Option<String>,
    pub surprise: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ModelsResponse {
    pub schema_version: u32,
    pub root: String,
    pub models: Vec<ModelSummary>,
    pub registry: Vec<ModelRegistrySummary>,
    pub behavior_nodes: Vec<BehaviorNodeState>,
    pub connections: Vec<ModelConnection>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ModelSummary {
    pub name: String,
    pub behavior: Option<String>,
    pub checkpoint_path: String,
    pub samples_seen: Option<u64>,
    pub best_loss: Option<f32>,
    pub input_dim: Option<u64>,
    pub output_dim: Option<u64>,
    pub latent_dim: Option<u64>,
    pub width: Option<u64>,
    pub height: Option<u64>,
    pub evaluation: Option<ModelEvaluationSummary>,
    pub metrics: Option<ModelTrainingMetricSummary>,
    pub registered_status: Option<String>,
    pub allowed_modes: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ModelEvaluationSummary {
    pub sample_count: Option<u64>,
    pub model_loss_mean: Option<f32>,
    pub hardcoded_loss_mean: Option<f32>,
    pub selected_loss_mean: Option<f32>,
    pub model_better_than_hardcoded: Option<bool>,
    pub improvement_ratio: Option<f32>,
    pub recommendation: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ModelTrainingMetricSummary {
    pub record_count: usize,
    pub last_epoch: Option<u64>,
    pub last_sample_index: Option<u64>,
    pub last_train_loss: Option<f32>,
    pub last_model_loss: Option<f32>,
    pub last_hardcoded_loss: Option<f32>,
    pub last_selected_loss: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ModelRegistrySummary {
    pub name: String,
    pub behavior: String,
    pub checkpoint_path: String,
    pub training_ledger: Option<String>,
    pub behavior_report_path: Option<String>,
    pub scenario_report_path: Option<String>,
    pub status: String,
    pub allowed_modes: Vec<String>,
    pub scenario_success_rate: Option<f32>,
    pub collision_rate: Option<f32>,
    pub episodes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelConnection {
    pub from: String,
    pub to: String,
    pub label: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CaptureSceneQuery {
    pub capture: PathBuf,
    #[serde(default)]
    pub frame: usize,
}
