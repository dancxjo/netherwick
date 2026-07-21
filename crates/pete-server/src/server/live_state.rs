#[derive(Clone, Debug)]
pub struct LiveViewState {
    latest: Arc<Mutex<Option<WorldSnapshot>>>,
    map: Arc<Mutex<LocalMap>>,
    point_cloud: Arc<Mutex<VoxelPointCloud>>,
    latest_embodied: Arc<Mutex<Option<EmbodiedContext>>>,
    scene_metadata: Arc<Mutex<Option<LiveSceneMetadata>>>,
    session: Arc<Mutex<Option<SceneSession>>>,
    hardware_control: Arc<Mutex<HardwareControlState>>,
    training_status: Arc<Mutex<LiveTrainingStatus>>,
    inline_learning: Arc<Mutex<InlineLearningConfig>>,
    prod_state: Arc<Mutex<NudgeStatus>>,
    behavior_nodes: Arc<Mutex<Vec<BehaviorNodeState>>>,
    surface_extractor: Arc<Mutex<SurfaceExtractor>>,
    entity_memory: Arc<Mutex<EntityMemory>>,
    pub virtual_retina: bool,
    pub retina_width: u32,
    pub retina_height: u32,
    pub retina_fps: f32,
    retina_state: Arc<Mutex<RetinaState>>,
}

#[derive(Clone, Debug, Default)]
struct RetinaState {
    latest_frame: Option<pete_sensors::EyeFrame>,
    has_new_frame: bool,
    last_received_at: Option<std::time::Instant>,
    frames_received: usize,
    frames_attached_to_snapshots: usize,
    frames_written_to_ledger: usize,
    warnings: Vec<String>,
}

impl Default for LiveViewState {
    fn default() -> Self {
        Self {
            latest: Arc::new(Mutex::new(None)),
            map: Arc::new(Mutex::new(LocalMap::default())),
            point_cloud: Arc::new(Mutex::new(VoxelPointCloud::default())),
            latest_embodied: Arc::new(Mutex::new(None)),
            scene_metadata: Arc::new(Mutex::new(None)),
            session: Arc::new(Mutex::new(None)),
            hardware_control: Arc::new(Mutex::new(HardwareControlState::default())),
            training_status: Arc::new(Mutex::new(LiveTrainingStatus::default())),
            inline_learning: Arc::new(Mutex::new(InlineLearningConfig::default())),
            prod_state: Arc::new(Mutex::new(NudgeStatus::default())),
            behavior_nodes: Arc::new(Mutex::new(default_behavior_nodes())),
            surface_extractor: Arc::new(Mutex::new(SurfaceExtractor::default())),
            entity_memory: Arc::new(Mutex::new(EntityMemory::default())),
            virtual_retina: false,
            retina_width: 160,
            retina_height: 90,
            retina_fps: 5.0,
            retina_state: Arc::new(Mutex::new(RetinaState::default())),
        }
    }
}

impl LiveViewState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_real_slow_hardware_control(self) -> Self {
        *self
            .hardware_control
            .lock()
            .expect("hardware control mutex poisoned") = HardwareControlState::real_slow();
        self
    }

    pub fn hardware_control_status(&self) -> HardwareControlStatus {
        let now_ms = wall_now_ms();
        let latest = self.latest();
        self.hardware_control
            .lock()
            .expect("hardware control mutex poisoned")
            .status(latest.as_ref(), now_ms)
    }

    pub fn with_virtual_retina(mut self, enabled: bool) -> Self {
        self.virtual_retina = enabled;
        self
    }

    pub fn with_retina_dimensions(mut self, width: u32, height: u32) -> Self {
        self.retina_width = width;
        self.retina_height = height;
        self
    }

    pub fn with_retina_fps(mut self, fps: f32) -> Self {
        self.retina_fps = fps;
        self
    }

    pub fn take_pending_retina_frame(&self) -> Option<pete_sensors::EyeFrame> {
        let mut state = self
            .retina_state
            .lock()
            .expect("retina state mutex poisoned");
        if state.has_new_frame {
            state.has_new_frame = false;
            state.frames_attached_to_snapshots += 1;
            state.latest_frame.clone()
        } else {
            None
        }
    }

    pub fn record_ledger_write(&self) {
        let mut state = self
            .retina_state
            .lock()
            .expect("retina state mutex poisoned");
        state.frames_written_to_ledger += 1;
    }

    pub fn record_live_eye_frame(&self, frame: pete_sensors::EyeFrame) {
        {
            let mut state = self
                .retina_state
                .lock()
                .expect("retina state mutex poisoned");
            state.latest_frame = Some(frame.clone());
            state.has_new_frame = true;
            state.last_received_at = Some(std::time::Instant::now());
            state.frames_received += 1;
        }

        if let Some(snapshot) = self
            .latest
            .lock()
            .expect("live view snapshot mutex poisoned")
            .as_mut()
        {
            snapshot.eye_frame = Some(frame);
        }
    }

    pub fn update(&self, snapshot: WorldSnapshot) {
        self.update_with_runtime_map(snapshot, None);
    }

    pub fn update_with_runtime_map(
        &self,
        snapshot: WorldSnapshot,
        runtime_map: Option<&LocalMap>,
    ) {
        let now = snapshot.to_now(snapshot.body.last_update_ms);
        let mut map = self.map.lock().expect("live map mutex poisoned");
        if let Some(runtime_map) = runtime_map {
            *map = runtime_map.clone();
        } else {
            map.observe_snapshot(&snapshot, snapshot.body.last_update_ms);
        }
        drop(map);
        {
            let calibration = self
                .scene_metadata()
                .and_then(|metadata| metadata.sensor_calibration);
            let mut cloud = self
                .point_cloud
                .lock()
                .expect("live point cloud mutex poisoned");
            if apply_live_point_cloud_calibration(&mut cloud, calibration, &snapshot) {
                cloud.observe_snapshot(&snapshot, snapshot.body.last_update_ms);
            } else {
                reset_point_cloud(&mut cloud);
            }
        }
        {
            use pete_memory::PlaceCellKey;
            const CELL_SIZE: f32 = 0.5;
            let x = now.body.odometry.x_m;
            let y = now.body.odometry.y_m;
            let cell_key = PlaceCellKey {
                x: (x / CELL_SIZE).floor() as i32,
                y: (y / CELL_SIZE).floor() as i32,
            };
            self.entity_memory
                .lock()
                .expect("entity memory mutex poisoned")
                .observe_now(&now, Some(cell_key));
        }
        *self
            .latest
            .lock()
            .expect("live view snapshot mutex poisoned") = Some(snapshot);
    }

    pub fn entity_memory_report(&self) -> EntityMemoryReport {
        self.entity_memory
            .lock()
            .expect("entity memory mutex poisoned")
            .report()
    }

    pub fn cognitive_diagnostics_report(&self) -> CognitiveDiagnosticsReport {
        let report =
            CognitiveDiagnosticsReport::from_entity_memory_report(&self.entity_memory_report());
        if let Some(context) = self.latest_embodied_context() {
            report.with_embodied_context(&context)
        } else {
            report
        }
    }

    pub fn latest(&self) -> Option<WorldSnapshot> {
        self.latest
            .lock()
            .expect("live view snapshot mutex poisoned")
            .clone()
    }

    pub fn map_snapshot(&self) -> LocalMap {
        self.map.lock().expect("live map mutex poisoned").clone()
    }

    pub fn point_cloud_snapshot(&self) -> VoxelPointCloud {
        self.point_cloud
            .lock()
            .expect("live point cloud mutex poisoned")
            .clone()
    }

    pub fn update_embodied_context(&self, context: EmbodiedContext) {
        *self
            .latest_embodied
            .lock()
            .expect("live embodied context mutex poisoned") = Some(context);
    }

    pub fn latest_embodied_context(&self) -> Option<EmbodiedContext> {
        self.latest_embodied
            .lock()
            .expect("live embodied context mutex poisoned")
            .clone()
    }

    pub fn update_scene_metadata(&self, metadata: LiveSceneMetadata) {
        *self
            .scene_metadata
            .lock()
            .expect("live view scene metadata mutex poisoned") = Some(metadata);
    }

    pub fn scene_metadata(&self) -> Option<LiveSceneMetadata> {
        self.scene_metadata
            .lock()
            .expect("live view scene metadata mutex poisoned")
            .clone()
    }

    pub fn update_session(&self, session: SceneSession) {
        *self
            .session
            .lock()
            .expect("live view session mutex poisoned") = Some(session);
    }

    pub fn session(&self) -> Option<SceneSession> {
        self.session
            .lock()
            .expect("live view session mutex poisoned")
            .clone()
    }

    pub fn update_training_status(&self, status: LiveTrainingStatus) {
        *self
            .training_status
            .lock()
            .expect("live view training status mutex poisoned") = status;
    }

    pub fn training_status(&self) -> LiveTrainingStatus {
        self.training_status
            .lock()
            .expect("live view training status mutex poisoned")
            .clone()
    }

    pub fn update_inline_learning(&self, config: InlineLearningConfig) {
        *self
            .inline_learning
            .lock()
            .expect("inline learning mutex poisoned") = config;
    }

    pub fn inline_learning(&self) -> InlineLearningConfig {
        self.inline_learning
            .lock()
            .expect("inline learning mutex poisoned")
            .clone()
    }

    pub fn update_prod_state(&self, status: NudgeStatus) {
        *self.prod_state.lock().expect("prod state mutex poisoned") = status;
    }

    pub fn prod_state(&self) -> NudgeStatus {
        self.prod_state
            .lock()
            .expect("prod state mutex poisoned")
            .clone()
    }

    pub fn behavior_nodes(&self) -> Vec<BehaviorNodeState> {
        self.behavior_nodes
            .lock()
            .expect("behavior nodes mutex poisoned")
            .clone()
    }

    pub fn surface_perception(
        &self,
        snapshot: &WorldSnapshot,
        calibration: Option<SceneSensorCalibration>,
        action: Option<&ActionPrimitive>,
    ) -> Option<SceneSurfacePerception> {
        if snapshot.kinect.depth_m.is_empty()
            || snapshot.kinect.depth_width == 0
            || snapshot.kinect.depth_height == 0
        {
            return None;
        }
        let mut extractor = self
            .surface_extractor
            .lock()
            .expect("surface extractor mutex poisoned");
        let calibration = calibration.unwrap_or_else(SceneSensorCalibration::sim_default);
        extractor.set_depth_camera_extrinsics(
            calibration.depth_camera_height_m(),
            calibration.depth_camera_forward_m(),
            calibration.depth_camera_pitch_rad(),
            calibration.camera_roll_rad,
            calibration.camera_yaw_rad,
        );
        extractor.set_compact_depth_calibration(
            calibration.compact_depth_beam_count,
            calibration.compact_depth_fov_rad,
            calibration.depth_scale,
        );
        let mut perception = SceneSurfacePerception::from(extractor.process(
            &snapshot.kinect,
            snapshot.body.odometry,
            snapshot.body.last_update_ms,
        ));
        if let Some(action) = action {
            let frames = pete_sensors::anticipate_surfaces(
                &pete_sensors::SurfaceExtractorOutput {
                    plane_observations: perception.plane_observations.clone(),
                    stable_surfaces: perception.stable_surfaces.clone(),
                    floor: perception.floor.clone(),
                    obstacle_grid: perception.obstacle_grid.clone(),
                    clusters: perception.clusters.clone(),
                    scene_graph: perception.scene_graph.clone(),
                    diagnostics: perception.diagnostics.clone(),
                    raw_cloud: Vec::new(),
                    filtered_cloud: Vec::new(),
                },
                snapshot.body.odometry,
                action,
            );
            if let Some(object) = perception.scene_graph.navigation.as_object_mut() {
                object.insert(
                    "anticipation".to_string(),
                    serde_json::json!({
                        "action": action,
                        "frames": frames,
                    }),
                );
            }
        }
        Some(perception)
    }

    pub fn update_behavior_nodes(&self, nodes: Vec<BehaviorNodeState>) {
        let mut current = self
            .behavior_nodes
            .lock()
            .expect("behavior nodes mutex poisoned");
        let merged = nodes
            .into_iter()
            .map(|mut node| {
                if let Some(previous) = current
                    .iter()
                    .find(|old| same_behavior_node(&old.node_id, &node.node_id))
                {
                    if node.checkpoint_path.is_none() {
                        node.checkpoint_path = previous.checkpoint_path.clone();
                    }
                    node.training_enabled = previous.training_enabled
                        || matches!(
                            node.selected_regime,
                            BehaviorRegime::ShadowTrain | BehaviorRegime::ModelTrainAndInfer
                        );
                }
                node.missing_model_or_checkpoint =
                    !matches!(node.selected_regime, BehaviorRegime::Hardcoded)
                        && (node.selected_model.is_none()
                            || node
                                .checkpoint_path
                                .as_ref()
                                .map(|path| path.trim().is_empty())
                                .unwrap_or(true));
                node
            })
            .collect();
        *current = merged;
    }

    pub fn update_behavior_node(
        &self,
        id: &str,
        update: BehaviorNodeUpdate,
    ) -> Option<BehaviorNodeState> {
        let mut nodes = self
            .behavior_nodes
            .lock()
            .expect("behavior nodes mutex poisoned");
        let node = nodes.iter_mut().find(|node| {
            same_behavior_node(&node.node_id, id) || same_behavior_node(&node.behavior_id, id)
        })?;
        if let Some(regime) = update.selected_regime {
            node.selected_regime = regime;
            node.training_enabled = update.training_enabled.unwrap_or(matches!(
                regime,
                BehaviorRegime::ShadowTrain | BehaviorRegime::ModelTrainAndInfer
            ));
        }
        if let Some(hardcoded) = update.selected_hardcoded {
            node.selected_hardcoded = hardcoded;
        }
        if let Some(model) = update.selected_model {
            node.selected_model = Some(model);
        }
        if let Some(checkpoint) = update.checkpoint_path {
            node.checkpoint_path = (!checkpoint.trim().is_empty()).then_some(checkpoint);
        }
        if let Some(fallback) = update.fallback_policy {
            node.fallback_policy = fallback;
        }
        if let Some(training_enabled) = update.training_enabled {
            node.training_enabled = training_enabled;
        }
        node.missing_model_or_checkpoint =
            !matches!(node.selected_regime, BehaviorRegime::Hardcoded)
                && (node.selected_model.is_none()
                    || node
                        .checkpoint_path
                        .as_ref()
                        .map(|path| path.trim().is_empty())
                        .unwrap_or(true));
        Some(node.clone())
    }
}

fn apply_live_point_cloud_calibration(
    cloud: &mut VoxelPointCloud,
    calibration: Option<SceneSensorCalibration>,
    snapshot: &WorldSnapshot,
) -> bool {
    if snapshot.kinect.depth_m.is_empty() {
        return true;
    }
    let full_depth_image = snapshot.kinect.depth_width > 0 && snapshot.kinect.depth_height > 0;
    let Some(calibration) = calibration else {
        return !full_depth_image;
    };

    let previous = cloud.config;
    cloud.config.camera_height_m = calibration.depth_camera_height_m();
    cloud.config.camera_forward_m = calibration.depth_camera_forward_m();
    cloud.config.camera_pitch_rad = calibration.depth_camera_pitch_rad();
    cloud.config.camera_roll_rad = calibration.camera_roll_rad;
    cloud.config.camera_yaw_rad = calibration.camera_yaw_rad;

    if point_cloud_extrinsics_changed(previous, cloud.config) {
        reset_point_cloud(cloud);
    }
    true
}

fn point_cloud_extrinsics_changed(
    previous: pete_map::PointCloudConfig,
    current: pete_map::PointCloudConfig,
) -> bool {
    const EPS: f32 = 1.0e-4;
    (previous.camera_height_m - current.camera_height_m).abs() > EPS
        || (previous.camera_forward_m - current.camera_forward_m).abs() > EPS
        || (previous.camera_pitch_rad - current.camera_pitch_rad).abs() > EPS
        || (previous.camera_roll_rad - current.camera_roll_rad).abs() > EPS
        || (previous.camera_yaw_rad - current.camera_yaw_rad).abs() > EPS
}

fn reset_point_cloud(cloud: &mut VoxelPointCloud) {
    cloud.voxels.clear();
    cloud.observations = 0;
    cloud.raw_points_seen = 0;
    cloud.orientation_status = Default::default();
    cloud.last_kinect_capture_ms = None;
    cloud.last_range_capture_ms = None;
}

fn default_behavior_nodes() -> Vec<BehaviorNodeState> {
    RuntimeModelStack::default().behavior_node_states(&[])
}

fn same_behavior_node(left: &str, right: &str) -> bool {
    normalize_behavior_node_id(left) == normalize_behavior_node_id(right)
}

fn normalize_behavior_node_id(id: &str) -> String {
    match id {
        "ActionValue" => "action_value".to_string(),
        "EyeNext" => "eye_next".to_string(),
        "EarNext" => "ear_next".to_string(),
        "EventBump" => "event_bump".to_string(),
        other => other.to_ascii_lowercase().replace('-', "_"),
    }
}
