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
            pose_graph_corrections_applied: false,
            retained_observations: Vec::new(),
            last_pose_graph_revision: None,
            last_kinect_capture_ms: None,
            last_range_capture_ms: None,
        }
    }

    pub fn observe_snapshot(
        &mut self,
        snapshot: &WorldSnapshot,
        t_ms: TimeMs,
    ) -> PointCloudSummary {
        let new_kinect = snapshot.kinect.captured_at_ms == 0
            || self.last_kinect_capture_ms != Some(snapshot.kinect.captured_at_ms);
        let new_range = snapshot.range.captured_at_ms == 0
            || self.last_range_capture_ms != Some(snapshot.range.captured_at_ms);
        let mut observations = pointcloud_observations_from_snapshot(snapshot, t_ms, self.config);
        observations.retain(|observation| {
            if observation.source == "kinect_depth" {
                new_kinect
            } else {
                new_range
            }
        });
        if new_kinect
            && snapshot.kinect.captured_at_ms > 0
            && observations
                .iter()
                .any(|observation| observation.source == "kinect_depth")
        {
            self.last_kinect_capture_ms = Some(snapshot.kinect.captured_at_ms);
        }
        if new_range
            && snapshot.range.captured_at_ms > 0
            && observations
                .iter()
                .any(|observation| observation.source != "kinect_depth")
        {
            self.last_range_capture_ms = Some(snapshot.range.captured_at_ms);
        }
        if observations.is_empty() {
            self.decay_stale(t_ms);
        } else {
            for observation in observations {
                self.integrate_observation(observation);
            }
        }
        self.summary()
    }

    pub fn integrate_observation(&mut self, observation: PointCloudObservation) {
        self.retained_observations.push(observation.clone());
        if self.retained_observations.len() > 256 {
            self.retained_observations.remove(0);
        }
        self.pose_graph_corrections_applied = false;
        self.last_pose_graph_revision = None;
        self.integrate_observation_points(&observation);
    }

    fn integrate_observation_points(&mut self, observation: &PointCloudObservation) {
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

    /// Rebuild retained sensor-frame observations using optimized graph poses.
    /// Returns true only when every retained observation could be associated
    /// with a corrected graph node.
    pub fn rebuild_from_pose_graph(&mut self, map: &LocalMap) -> bool {
        let revision = pose_graph_revision(map);
        if self.pose_graph_corrections_applied && self.last_pose_graph_revision == Some(revision) {
            return true;
        }
        let correction_required = map.pose_graph_optimization.max_node_update_m > 0.001;
        if !correction_required {
            self.pose_graph_corrections_applied = true;
            self.last_pose_graph_revision = Some(revision);
            return true;
        }
        if self.retained_observations.is_empty() || map.pose_graph.nodes.is_empty() {
            self.pose_graph_corrections_applied = false;
            return false;
        }
        let mut rebuilt = Vec::with_capacity(self.retained_observations.len());
        for original in &self.retained_observations {
            // Submaps retain both the node association chosen at capture time and
            // the observation offset from that node. Re-derive neither from the
            // optimized graph: a later node may now be closer in time or space.
            let Some(submap) = map
                .submaps
                .iter()
                .min_by_key(|submap| submap.t_ms.abs_diff(original.t_ms))
                .filter(|submap| submap.t_ms.abs_diff(original.t_ms) <= 1_000)
            else {
                self.pose_graph_corrections_applied = false;
                return false;
            };
            let Some(node) = map
                .pose_graph
                .nodes
                .iter()
                .find(|node| node.id == submap.node_id)
            else {
                self.pose_graph_corrections_applied = false;
                return false;
            };
            let mut corrected = original.clone();
            corrected.pose.pose = apply_pose_delta(node.pose_estimate.pose, submap.local_pose);
            if corrected.orientation.yaw_source == YawSource::OdometryHeading {
                corrected.orientation.yaw_rad = Some(corrected.pose.pose.heading_rad);
            }
            corrected.pose.source = "pose_graph_corrected".to_string();
            rebuilt.push(corrected);
        }
        self.voxels.clear();
        self.observations = 0;
        self.raw_points_seen = 0;
        self.orientation_status = OrientationStatus::default();
        for observation in &rebuilt {
            self.integrate_observation_points(observation);
        }
        self.pose_graph_corrections_applied = true;
        self.last_pose_graph_revision = Some(revision);
        true
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

fn pose_graph_revision(map: &LocalMap) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    hash ^= u64::from(map.pose_graph_optimization.max_node_update_m.to_bits());
    hash = hash.wrapping_mul(0x100000001b3);
    for node in &map.pose_graph.nodes {
        for value in [
            node.t_ms,
            u64::from(node.pose_estimate.pose.x_m.to_bits()),
            u64::from(node.pose_estimate.pose.y_m.to_bits()),
            u64::from(node.pose_estimate.pose.heading_rad.to_bits()),
        ] {
            hash ^= value;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}
