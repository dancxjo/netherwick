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
