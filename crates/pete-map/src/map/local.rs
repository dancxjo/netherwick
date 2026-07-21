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
        let target = self.find_live_loop_target(candidate, &current.id).cloned();
        let registration = target.as_ref().and_then(|target| {
            self.measure_loop_registration(target.pose_estimate.pose, observation)
        });
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
            registration: registration.clone(),
        };
        let to = target
            .as_ref()
            .map(|node| node.id.clone())
            .unwrap_or_else(|| "unresolved".to_string());
        let target_pose = target.as_ref().map(|node| node.pose_estimate.pose);
        let transform = match (target_pose, registration.as_ref()) {
            (Some(target_pose), Some(registration)) => {
                pose_delta(registration.registered_pose, target_pose)
            }
            (Some(target_pose), None) => pose_delta(current.pose_estimate.pose, target_pose),
            (None, _) => pose_delta(current.pose_estimate.pose, candidate.target_pose),
        };
        let rejection_reason = self.live_loop_rejection_reason(
            &current,
            target.as_ref(),
            registration.as_ref(),
            candidate,
        );

        self.pose_graph.edges.push(PoseEdge {
            from: current.id,
            to,
            transform,
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
        registration: Option<&LoopRegistrationMeasurement>,
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
        if target.is_none() {
            return Some("no prior node close enough to candidate target".to_string());
        }
        let Some(registration) = registration else {
            return Some("scan/submap registration produced no measured constraint".to_string());
        };
        if registration.geometric_overlap < self.config.pose_graph_loop_min_geometric_overlap {
            return Some(format!(
                "geometric occupancy agreement {:.3} below gate {:.3}",
                registration.geometric_overlap, self.config.pose_graph_loop_min_geometric_overlap
            ));
        }
        None
    }

    fn measure_loop_registration(
        &self,
        target_pose: Pose2,
        observation: &MapObservation,
    ) -> Option<LoopRegistrationMeasurement> {
        if !observation.range_beams.iter().any(|beam| {
            beam.hit
                && beam.confidence > 0.0
                && beam.distance_m.is_finite()
                && beam.distance_m > 0.0
        }) {
            return None;
        }

        let mut best_pose = target_pose;
        let mut best_score = self.scan_match_score(target_pose, &observation.range_beams);
        let xy_step = (self.config.resolution_m * 0.5).max(0.025);
        let xy_window = self
            .config
            .scan_match_xy_window_m
            .max(self.config.resolution_m * 2.0);
        let theta_window = self
            .config
            .scan_match_theta_window_rad
            .max(10.0_f32.to_radians());
        let theta_step = (theta_window / 2.0).max(2.0_f32.to_radians());
        let xy_steps = (xy_window / xy_step).ceil() as i32;
        let theta_steps = (theta_window / theta_step).ceil() as i32;

        for ix in -xy_steps..=xy_steps {
            for iy in -xy_steps..=xy_steps {
                for itheta in -theta_steps..=theta_steps {
                    let pose = Pose2 {
                        x_m: target_pose.x_m + ix as f32 * xy_step,
                        y_m: target_pose.y_m + iy as f32 * xy_step,
                        heading_rad: normalize_angle(
                            target_pose.heading_rad + itheta as f32 * theta_step,
                        ),
                    };
                    let score = self.scan_match_score(pose, &observation.range_beams);
                    if score > best_score + f32::EPSILON {
                        best_score = score;
                        best_pose = pose;
                    }
                }
            }
        }

        Some(LoopRegistrationMeasurement {
            algorithm: "correlative_occupancy_submap_registration".to_string(),
            registered_pose: best_pose,
            score: best_score,
            odometry_score: self.scan_match_score(
                observation.pose.pose,
                &observation.range_beams,
            ),
            geometric_overlap: self.loop_candidate_geometric_overlap(best_pose, observation),
            odometry_geometric_overlap: self
                .loop_candidate_geometric_overlap(observation.pose.pose, observation),
        })
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
        let scan_match_edges = self
            .pose_graph
            .edges
            .iter()
            .filter(|edge| matches!(edge.source, PoseEdgeSource::ScanMatch { .. }))
            .count();
        let slam_status =
            self.slam_status(occupied_cells, scan_match_edges, loop_closures_accepted);

        MapSummary {
            label: MAP_LABEL,
            slam_status,
            resolution_m: self.config.resolution_m,
            cells: self.cells.len(),
            occupied_cells,
            free_cells,
            observations: self.observations.len(),
            pose_graph_nodes: self.pose_graph.nodes.len(),
            pose_graph_edges: self.pose_graph.edges.len(),
            scan_match_edges,
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

    fn slam_status(
        &self,
        occupied_cells: usize,
        scan_match_edges: usize,
        loop_closures_accepted: usize,
    ) -> SlamStatus {
        let local_scan_matching_active = scan_match_edges > 0;
        let loop_closure_active = loop_closures_accepted > 0;
        let pose_graph_optimized = self.pose_graph_optimization.active_edges > 0
            && self.pose_graph_optimization.optimized_nodes >= 2;
        let occupancy_remapped_from_pose_graph =
            self.remap_summary.generation > 0 && self.remap_summary.submaps > 0;
        let mode =
            if loop_closure_active && pose_graph_optimized && occupancy_remapped_from_pose_graph {
                SlamMode::LoopClosedPoseGraph
            } else if local_scan_matching_active && occupancy_remapped_from_pose_graph {
                SlamMode::LocalScanMatched
            } else if occupied_cells > 0 || self.remap_summary.submaps > 0 {
                SlamMode::MappingOnly
            } else {
                SlamMode::OdometryOnly
            };

        let mut reasons = Vec::new();
        if occupied_cells == 0 {
            reasons.push("no occupied map cells from range/depth observations yet".to_string());
        }
        if self.pose_graph.nodes.len() < 2 {
            reasons.push("pose graph has fewer than two nodes".to_string());
        }
        if !local_scan_matching_active {
            reasons.push("no scan-match correction edge has been accepted yet".to_string());
        }
        if !loop_closure_active {
            reasons.push("no loop-closure candidate has been accepted yet".to_string());
        }
        if !pose_graph_optimized {
            reasons.push(
                "pose graph optimization has not constrained multiple active nodes yet".to_string(),
            );
        }
        if !occupancy_remapped_from_pose_graph {
            reasons.push("occupancy has not been rebuilt from pose-graph submaps yet".to_string());
        }

        SlamStatus {
            mode,
            local_scan_matching_active,
            loop_closure_active,
            pose_graph_optimized,
            occupancy_remapped_from_pose_graph,
            reasons,
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
