use std::collections::BTreeMap;

use netherwick_core::{Pose2, TimeMs};
use netherwick_now::{KinectSense, Now};
use netherwick_sensors::WorldSnapshot;
use serde::{Deserialize, Serialize};

pub const MAP_EXTENSION_NAME: &str = "map.odometry";
pub const MAP_LABEL: &str = "SLAM-lite / odometry map, not full SLAM";
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
    pub points: Vec<PointCloudPoint>,
    pub source: String,
    pub t_ms: TimeMs,
    pub metadata: serde_json::Value,
}

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
pub struct LocalMap {
    pub cells: BTreeMap<CellKey, OccupancyCell>,
    pub pose_history: Vec<PoseEstimate>,
    pub observations: Vec<MapObservation>,
    pub config: MapConfig,
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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MapSummary {
    pub label: &'static str,
    pub resolution_m: f32,
    pub cells: usize,
    pub occupied_cells: usize,
    pub free_cells: usize,
    pub observations: usize,
    pub latest_pose: Option<PoseEstimate>,
    pub latest_observation: Option<MapObservationSummary>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MapObservationSummary {
    pub t_ms: TimeMs,
    pub beam_count: usize,
    pub hit_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoseNode {
    pub id: String,
    pub pose_estimate: PoseEstimate,
    pub t_ms: TimeMs,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_frame_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoseEdgeSource {
    Odometry,
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
        }
    }
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
            config,
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

    pub fn integrate_observation(&mut self, observation: MapObservation) {
        self.pose_history.push(observation.pose.clone());
        cap_vec(&mut self.pose_history, self.config.max_pose_history);

        for beam in &observation.range_beams {
            self.integrate_beam(observation.pose.pose, beam, observation.t_ms);
        }

        self.observations.push(observation);
        cap_vec(&mut self.observations, self.config.max_observations);
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
        MapSummary {
            label: MAP_LABEL,
            resolution_m: self.config.resolution_m,
            cells: self.cells.len(),
            occupied_cells,
            free_cells,
            observations: self.observations.len(),
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
            let hit = nearest_m
                .filter(|nearest| nearest.is_finite())
                .map(|nearest| (distance - nearest).abs() <= config.hit_epsilon_m)
                .unwrap_or(false)
                && distance <= config.max_range_m;
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
    config: PointCloudConfig,
) -> Point3D {
    let robot = match frame {
        PointCloudFrame::OdometryWorld => return point,
        PointCloudFrame::RobotBase => point,
        PointCloudFrame::KinectCamera | PointCloudFrame::DepthImageUnknown => Point3D {
            x_m: config.camera_forward_m + point.z_m,
            y_m: -point.x_m,
            z_m: config.camera_height_m - point.y_m,
        },
    };
    let sin = pose.heading_rad.sin();
    let cos = pose.heading_rad.cos();
    Point3D {
        x_m: pose.x_m + robot.x_m * cos - robot.y_m * sin,
        y_m: pose.y_m + robot.x_m * sin + robot.y_m * cos,
        z_m: robot.z_m,
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
        assert_eq!(map.summary().label, MAP_LABEL);
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
            config,
        );
        assert!((world.x_m - 1.0).abs() < 0.001);
        assert!((world.y_m - 3.0).abs() < 0.001);
        assert!((world.z_m - 0.2).abs() < 0.001);
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

    fn pose(x_m: f32, y_m: f32, heading_rad: f32) -> Pose2 {
        Pose2 {
            x_m,
            y_m,
            heading_rad,
        }
    }
}
