fn map_response_from_parts(
    map: &LocalMap,
    point_cloud: &VoxelPointCloud,
    latest: &WorldSnapshot,
    metadata: Option<&LiveSceneMetadata>,
    entity_report: &EntityMemoryReport,
) -> LiveMapResponse {
    let now_ms = latest.body.last_update_ms;
    let summary = map.summary();
    let pose_trail: Vec<_> = map
        .pose_history
        .iter()
        .map(|pose| MapPosePoint {
            x_m: pose.pose.x_m,
            y_m: pose.pose.y_m,
            heading_rad: pose.pose.heading_rad,
            confidence: pose.confidence,
            t_ms: pose.t_ms,
        })
        .collect();
    let current_pose = pose_trail.last().cloned();
    let range_beams = map
        .observations
        .last()
        .map(|observation| projected_beams_from_observation(observation, now_ms))
        .unwrap_or_default();
    let cells: Vec<_> = map
        .cells
        .values()
        .map(|cell| map_view_cell(cell, map.config.resolution_m, now_ms))
        .collect();
    let world_projection = map_world_projection(point_cloud, &summary, latest, metadata, now_ms);
    let semantic_cells = map_semantic_cells(latest, metadata, now_ms);
    let events = map_event_markers(latest, metadata, now_ms);
    let pose_graph = map_pose_graph_summary(map);
    let entity_graph = map_entity_graph(
        &pose_trail,
        &range_beams,
        &cells,
        &semantic_cells,
        &events,
        entity_report,
        latest,
        now_ms,
    );

    LiveMapResponse {
        schema_version: 1,
        label: MAP_LABEL,
        summary,
        overlays: vec![
            "occupancy",
            "rays",
            "raw point cloud",
            "accumulated occupancy",
            "stable wall candidates",
            "danger",
            "charger/charge",
            "social",
            "novelty",
            "events",
        ],
        pose_trail,
        current_pose,
        range_beams,
        cells,
        world_projection,
        semantic_cells,
        events,
        pose_graph,
        remap: map.remap_summary,
        entity_graph,
    }
}

const WORLD_PROJECTION_LABEL: &str =
    "2D obstacle projection of the calibrated 3D odometry-world voxel cloud";
const WORLD_PROJECTION_MIN_OBSTACLE_Z_M: f32 = 0.05;
const WORLD_PROJECTION_MAX_OBSTACLE_Z_M: f32 = 2.0;
const MAX_TRUSTED_BELOW_FLOOR_RATIO: f32 = 0.02;

#[derive(Clone, Copy, Debug)]
struct ProjectedCellAccumulator {
    confidence: f32,
    last_seen_ms: TimeMs,
    voxel_count: usize,
    stable: bool,
}

fn map_world_projection(
    point_cloud: &VoxelPointCloud,
    map_summary: &MapSummary,
    latest: &WorldSnapshot,
    metadata: Option<&LiveSceneMetadata>,
    now_ms: TimeMs,
) -> MapWorldProjection {
    let resolution_m = map_summary.resolution_m;
    let points = point_cloud.points();
    let source_voxels = points.len();
    let below_floor_count = points
        .iter()
        .filter(|point| point.position.z_m < -WORLD_PROJECTION_MIN_OBSTACLE_Z_M)
        .count();
    let below_floor_ratio = if source_voxels == 0 {
        0.0
    } else {
        below_floor_count as f32 / source_voxels as f32
    };
    let mut projected = BTreeMap::<(i32, i32), ProjectedCellAccumulator>::new();
    for point in points.iter().filter(|point| {
        point.position.z_m >= WORLD_PROJECTION_MIN_OBSTACLE_Z_M
            && point.position.z_m <= WORLD_PROJECTION_MAX_OBSTACLE_Z_M
            && !point.transient
    }) {
        let key = (
            (point.position.x_m / resolution_m).floor() as i32,
            (point.position.y_m / resolution_m).floor() as i32,
        );
        let cell = projected.entry(key).or_insert(ProjectedCellAccumulator {
            confidence: 0.0,
            last_seen_ms: 0,
            voxel_count: 0,
            stable: false,
        });
        cell.confidence = cell.confidence.max(point.confidence);
        cell.last_seen_ms = cell.last_seen_ms.max(point.last_seen_ms);
        cell.voxel_count = cell.voxel_count.saturating_add(1);
        cell.stable |= point.stable;
    }
    let cells = projected
        .into_iter()
        .map(|((x, y), cell)| MapWorldProjectionCell {
            x,
            y,
            center_x_m: (x as f32 + 0.5) * resolution_m,
            center_y_m: (y as f32 + 0.5) * resolution_m,
            confidence: cell.confidence,
            age_ms: now_ms.saturating_sub(cell.last_seen_ms),
            voxel_count: cell.voxel_count,
            stable: cell.stable,
        })
        .collect::<Vec<_>>();
    let stable_cells = cells.iter().filter(|cell| cell.stable).count();
    let aligned_with_3d = point_cloud.observations > 0 && !cells.is_empty();
    let corrected_slam_ready = map_summary.slam_status.mode == SlamMode::LoopClosedPoseGraph;
    let graph_correction_not_applied_to_voxels = corrected_slam_ready
        && map_summary.pose_graph_optimization.max_node_update_m > 0.001
        && !point_cloud.pose_graph_corrections_applied;
    let has_depth = !latest.kinect.depth_m.is_empty();
    let calibrated_depth = !has_depth
        || if latest.kinect.schema_version >= 2 {
            latest
                .kinect
                .geometry_calibration
                .is_some_and(|calibration| calibration.physical_validation_ready())
        } else {
            metadata
                .and_then(|metadata| metadata.sensor_calibration)
                .is_some()
        };
    let depth_orientation_trusted =
        !has_depth || point_cloud.orientation_status.roll_pitch_corrected;
    let mut reasons = Vec::new();
    if point_cloud.observations == 0 {
        reasons.push("no calibrated 3D world observations have arrived".to_string());
    } else if cells.is_empty() {
        reasons.push("the shared 3D world has no projectable obstacle voxels".to_string());
    }
    if below_floor_ratio > MAX_TRUSTED_BELOW_FLOOR_RATIO {
        reasons.push(format!(
            "below-floor voxel ratio {below_floor_ratio:.3} exceeds {MAX_TRUSTED_BELOW_FLOOR_RATIO:.3}"
        ));
    }
    if stable_cells == 0 && !cells.is_empty() {
        reasons.push("the projection has no repeatedly observed stable cells yet".to_string());
    }
    if !calibrated_depth {
        reasons.push("the depth stream has no explicit camera calibration".to_string());
    }
    if !depth_orientation_trusted {
        reasons.push("the depth stream has no trusted IMU roll/pitch correction".to_string());
    }
    if !corrected_slam_ready {
        reasons.push(format!(
            "navigation remains gated while SLAM mode is {:?}",
            map_summary.slam_status.mode
        ));
    }
    if graph_correction_not_applied_to_voxels {
        reasons.push(
            "pose-graph corrections have not been applied to the accumulated 3D voxels".to_string(),
        );
    }
    let geometry_trusted = aligned_with_3d
        && below_floor_ratio <= MAX_TRUSTED_BELOW_FLOOR_RATIO
        && stable_cells > 0
        && calibrated_depth
        && depth_orientation_trusted
        && !graph_correction_not_applied_to_voxels;
    let navigation_trusted = geometry_trusted && corrected_slam_ready;

    MapWorldProjection {
        label: WORLD_PROJECTION_LABEL,
        source: WORLD_POINT_CLOUD_LABEL,
        coordinate_frame: "odometry_world",
        resolution_m,
        aligned_with_3d,
        geometry_trusted,
        navigation_trusted,
        reasons,
        source_voxels,
        projected_cells: cells.len(),
        stable_cells,
        cells,
    }
}

fn map_pose_graph_summary(map: &LocalMap) -> MapPoseGraphSummary {
    let mut odometry_edges = 0usize;
    let mut scan_match_edges = 0usize;
    let mut loop_candidate_edges = 0usize;
    let mut loop_candidate_active_edges = 0usize;
    let mut loop_candidate_rejection_reasons = Vec::new();
    for edge in &map.pose_graph.edges {
        match &edge.source {
            PoseEdgeSource::Odometry => odometry_edges = odometry_edges.saturating_add(1),
            PoseEdgeSource::ScanMatch { .. } => {
                scan_match_edges = scan_match_edges.saturating_add(1)
            }
            PoseEdgeSource::LoopClosureCandidate { .. } => {
                loop_candidate_edges = loop_candidate_edges.saturating_add(1);
                if edge.active {
                    loop_candidate_active_edges = loop_candidate_active_edges.saturating_add(1);
                } else if let Some(reason) = edge.rejection_reason.as_ref() {
                    loop_candidate_rejection_reasons.push(reason.clone());
                }
            }
        }
    }
    let loop_candidate_rejected_edges =
        loop_candidate_edges.saturating_sub(loop_candidate_active_edges);
    loop_candidate_rejection_reasons.sort();
    loop_candidate_rejection_reasons.dedup();

    MapPoseGraphSummary {
        nodes: map.pose_graph.nodes.len(),
        edges: map.pose_graph.edges.len(),
        odometry_edges,
        scan_match_edges,
        loop_candidate_edges,
        loop_candidate_active_edges,
        loop_candidate_rejected_edges,
        loop_candidate_rejection_reasons,
        latest_node_id: map.pose_graph.nodes.last().map(|node| node.id.clone()),
        latest_edge_source: map.pose_graph.edges.last().map(|edge| match &edge.source {
            PoseEdgeSource::Odometry => "odometry".to_string(),
            PoseEdgeSource::ScanMatch { .. } => "scan_match".to_string(),
            PoseEdgeSource::LoopClosureCandidate { .. } => "loop_closure_candidate".to_string(),
        }),
        optimization: map.pose_graph_optimization,
    }
}

fn map_entity_graph(
    pose_trail: &[MapPosePoint],
    range_beams: &[MapProjectedBeam],
    cells: &[MapViewCell],
    semantic_cells: &[MapSemanticCell],
    map_events: &[MapEventMarker],
    entity_report: &EntityMemoryReport,
    latest: &WorldSnapshot,
    now_ms: TimeMs,
) -> MapEntityGraph {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut events = Vec::new();
    let current_pose = pose_trail.last();

    if let Some(pose) = current_pose {
        push_graph_node(
            &mut nodes,
            "place:current",
            "place",
            "current place",
            Some("odometry"),
            Some((pose.x_m, pose.y_m)),
            pose.confidence,
            now_ms.saturating_sub(pose.t_ms),
        );
        push_graph_node(
            &mut nodes,
            "cluster:odometry:trail",
            "cluster",
            "odometry trail",
            Some("odometry"),
            Some((pose.x_m, pose.y_m)),
            pose.confidence,
            now_ms.saturating_sub(pose.t_ms),
        );
        push_graph_edge(
            &mut edges,
            "cluster:odometry:trail",
            "place:current",
            "same_place_as",
            pose.confidence,
        );
    }

    let nearest_beams: Vec<_> = range_beams
        .iter()
        .enumerate()
        .filter(|(_, beam)| beam.hit)
        .take(8)
        .collect();
    let has_range_cluster = !nearest_beams.is_empty();
    if has_range_cluster {
        push_graph_node(
            &mut nodes,
            "cluster:range:nearest",
            "cluster",
            "nearest range returns",
            Some("range"),
            current_pose.map(|pose| (pose.x_m, pose.y_m)),
            0.76,
            nearest_beams
                .iter()
                .map(|(_, beam)| beam.age_ms)
                .min()
                .unwrap_or(0),
        );
        if current_pose.is_some() {
            push_graph_edge(
                &mut edges,
                "cluster:range:nearest",
                "place:current",
                "same_time_as",
                0.72,
            );
        }
    }

    for (index, beam) in nearest_beams {
        let id = format!("observation:range:{index}");
        push_graph_node(
            &mut nodes,
            &id,
            "observation",
            &format!("range {:.2}m", beam.distance_m),
            Some("range"),
            Some((beam.end_x_m, beam.end_y_m)),
            beam.confidence,
            beam.age_ms,
        );
        push_graph_edge(
            &mut edges,
            &id,
            "cluster:range:nearest",
            "belongs_to",
            beam.confidence,
        );
        if current_pose.is_some() {
            push_graph_edge(&mut edges, &id, "place:current", "same_place_as", 0.54);
        }
    }

    for (index, cell) in cells
        .iter()
        .filter(|cell| cell.occupied_score > cell.free_score && cell.confidence > 0.18)
        .take(10)
        .enumerate()
    {
        let cluster_id = format!("cluster:occupancy:{}:{}", cell.x, cell.y);
        push_graph_node(
            &mut nodes,
            &cluster_id,
            "cluster",
            "occupied cell cluster",
            Some("range"),
            Some((cell.center_x_m, cell.center_y_m)),
            cell.confidence,
            cell.age_ms,
        );
        if has_range_cluster {
            push_graph_edge(
                &mut edges,
                &cluster_id,
                "cluster:range:nearest",
                "co_occurs_with",
                cell.confidence.min(0.7),
            );
        }
        if index < 4 {
            push_graph_edge(
                &mut edges,
                &cluster_id,
                "place:current",
                "same_place_as",
                cell.confidence.min(0.62),
            );
        }
    }

    for (index, cell) in semantic_cells.iter().take(12).enumerate() {
        let clean_kind = graph_id_fragment(&cell.kind);
        let cluster_id = format!("cluster:semantic:{clean_kind}:{index}");
        let entity_id = format!("entity:{clean_kind}:{index}");
        let label_id = format!("text_label:{clean_kind}:{index}");
        let age_ms = cell.age_ms.unwrap_or(0);
        let label = cell.label.clone().unwrap_or_else(|| cell.kind.clone());
        push_graph_node(
            &mut nodes,
            &cluster_id,
            "cluster",
            &format!("{} cluster", cell.kind),
            Some(cell.kind.as_str()),
            Some((cell.x_m, cell.y_m)),
            cell.confidence,
            age_ms,
        );
        push_graph_node(
            &mut nodes,
            &entity_id,
            "entity",
            &label,
            Some(cell.kind.as_str()),
            Some((cell.x_m, cell.y_m)),
            cell.confidence * cell.score,
            age_ms,
        );
        push_graph_node(
            &mut nodes,
            &label_id,
            "text_label",
            &label,
            Some("language"),
            Some((cell.x_m, cell.y_m)),
            cell.confidence,
            age_ms,
        );
        push_graph_edge(
            &mut edges,
            &cluster_id,
            &entity_id,
            "part_of_entity",
            cell.confidence,
        );
        push_graph_edge(
            &mut edges,
            &label_id,
            &entity_id,
            "named_by",
            cell.confidence,
        );
        push_graph_edge(
            &mut edges,
            &cluster_id,
            "place:current",
            "same_place_as",
            0.58,
        );
        push_graph_edge(
            &mut edges,
            &entity_id,
            "place:current",
            "same_place_as",
            0.62,
        );
        if has_range_cluster {
            push_graph_edge(
                &mut edges,
                "cluster:range:nearest",
                &cluster_id,
                "co_occurs_with",
                0.48,
            );
        }
        events.push(MapEntityGraphEvent {
            t_ms: now_ms.saturating_sub(age_ms),
            node_id: entity_id,
            event_type: "entity_seen".to_string(),
            label,
            confidence: cell.confidence,
            timestamp_ms: Some(now_ms.saturating_sub(age_ms)),
        });
    }

    for (index, event) in map_events.iter().take(8).enumerate() {
        let event_id = format!(
            "observation:event:{}:{index}",
            graph_id_fragment(&event.kind)
        );
        push_graph_node(
            &mut nodes,
            &event_id,
            "observation",
            event.label.as_deref().unwrap_or(&event.kind),
            Some(event.kind.as_str()),
            Some((event.x_m, event.y_m)),
            event.confidence,
            event.age_ms,
        );
        push_graph_edge(
            &mut edges,
            &event_id,
            "place:current",
            "same_place_as",
            event.confidence,
        );
        if has_range_cluster
            && (event.kind == "charger" || event.kind == "person" || event.kind == "speaker")
        {
            push_graph_edge(
                &mut edges,
                &event_id,
                "cluster:range:nearest",
                "co_occurs_with",
                0.44,
            );
        }
        events.push(MapEntityGraphEvent {
            t_ms: now_ms.saturating_sub(event.age_ms),
            node_id: event_id,
            event_type: event.kind.clone(),
            label: event.label.clone().unwrap_or_else(|| event.kind.clone()),
            confidence: event.confidence,
            timestamp_ms: Some(now_ms.saturating_sub(event.age_ms)),
        });
    }

    for entity in entity_report.top_entities.iter().take(8) {
        let entity_id = entity.id.clone();
        let entity_label = entity
            .display_name
            .clone()
            .or_else(|| entity.labels.first().cloned())
            .unwrap_or_else(|| entity.kind.clone());
        let entity_age_ms = now_ms.saturating_sub(entity.last_seen_ms);
        push_graph_node(
            &mut nodes,
            &entity_id,
            "entity",
            &entity_label,
            Some(entity.kind.as_str()),
            current_pose.map(|pose| (pose.x_m, pose.y_m)),
            entity.confidence,
            entity_age_ms,
        );
        if current_pose.is_some() {
            push_graph_edge(
                &mut edges,
                &entity_id,
                "place:current",
                "same_place_as",
                entity.confidence.min(0.7),
            );
        }
        events.push(MapEntityGraphEvent {
            t_ms: now_ms.saturating_sub(entity.first_seen_ms),
            node_id: entity_id.clone(),
            event_type: "create".to_string(),
            label: entity_label.clone(),
            confidence: entity.confidence,
            timestamp_ms: Some(entity.first_seen_ms),
        });
        if entity.observation_count > 1 {
            events.push(MapEntityGraphEvent {
                t_ms: now_ms.saturating_sub(entity.last_seen_ms),
                node_id: entity_id.clone(),
                event_type: "strengthen".to_string(),
                label: entity_label.clone(),
                confidence: entity.confidence,
                timestamp_ms: Some(entity.last_seen_ms),
            });
        }
        match entity.lifecycle {
            EntityLifecycleState::Occluded => events.push(MapEntityGraphEvent {
                t_ms: entity_age_ms,
                node_id: entity_id.clone(),
                event_type: "weaken".to_string(),
                label: entity_label.clone(),
                confidence: entity.confidence,
                timestamp_ms: Some(entity.last_seen_ms),
            }),
            EntityLifecycleState::Vanished => events.push(MapEntityGraphEvent {
                t_ms: entity_age_ms,
                node_id: entity_id.clone(),
                event_type: "vanish".to_string(),
                label: entity_label.clone(),
                confidence: entity.confidence,
                timestamp_ms: Some(entity.last_seen_ms),
            }),
            EntityLifecycleState::Active => {}
        }
        match entity.constellation_state {
            EntityConstellationState::Revived => events.push(MapEntityGraphEvent {
                t_ms: entity_age_ms,
                node_id: entity_id.clone(),
                event_type: "revive".to_string(),
                label: entity_label.clone(),
                confidence: entity.confidence,
                timestamp_ms: Some(entity.last_seen_ms),
            }),
            EntityConstellationState::Split => events.push(MapEntityGraphEvent {
                t_ms: entity_age_ms,
                node_id: entity_id.clone(),
                event_type: "split".to_string(),
                label: entity_label.clone(),
                confidence: entity.confidence,
                timestamp_ms: Some(entity.last_seen_ms),
            }),
            EntityConstellationState::Merged => events.push(MapEntityGraphEvent {
                t_ms: entity_age_ms,
                node_id: entity_id.clone(),
                event_type: "merge".to_string(),
                label: entity_label.clone(),
                confidence: entity.confidence,
                timestamp_ms: Some(entity.last_seen_ms),
            }),
            EntityConstellationState::Weak
            | EntityConstellationState::Strong
            | EntityConstellationState::Vanished => {}
        }
        for (label_index, label) in entity.text_labels.iter().take(3).enumerate() {
            let label_id = format!(
                "text_label:{}:{label_index}",
                graph_id_fragment(entity.id.as_str())
            );
            push_graph_node(
                &mut nodes,
                &label_id,
                "text_label",
                label,
                Some("language"),
                current_pose.map(|pose| (pose.x_m, pose.y_m)),
                entity.confidence,
                entity_age_ms,
            );
            push_graph_edge(
                &mut edges,
                &label_id,
                &entity_id,
                "named_by",
                entity.confidence.min(0.9),
            );
        }
        for cluster in entity.modality_clusters.iter().take(12) {
            let cluster_id = format!(
                "cluster:{}:{}",
                graph_id_fragment(entity.id.as_str()),
                graph_id_fragment(cluster.id.as_str())
            );
            push_graph_node(
                &mut nodes,
                &cluster_id,
                "cluster",
                cluster.id.as_str(),
                Some(cluster.modality.as_str()),
                current_pose.map(|pose| (pose.x_m, pose.y_m)),
                cluster.confidence,
                entity_age_ms,
            );
            push_graph_edge(
                &mut edges,
                &cluster_id,
                &entity_id,
                "part_of_entity",
                cluster.confidence,
            );
            let edge_id = format!("{cluster_id}->part_of_entity->{entity_id}");
            if let Some(edge) = edges.iter_mut().find(|edge| edge.id == edge_id) {
                edge.observed_at_ms = Some(entity.last_seen_ms);
            }
        }
        for point in entity.observation_points.iter().rev().take(24) {
            let node_id = format!(
                "observation:{}:{}",
                graph_id_fragment(entity.id.as_str()),
                graph_id_fragment(point.id.as_str())
            );
            let nearest_cluster = entity
                .modality_clusters
                .iter()
                .find(|cluster| {
                    cluster
                        .observation_point_ids
                        .iter()
                        .any(|id| id == &point.id)
                })
                .map(|cluster| {
                    format!(
                        "cluster:{}:{}",
                        graph_id_fragment(entity.id.as_str()),
                        graph_id_fragment(cluster.id.as_str())
                    )
                });
            if nodes.iter().all(|node| node.id != node_id) {
                nodes.push(MapEntityGraphNode {
                    id: node_id.clone(),
                    node_type: "observation".to_string(),
                    label: point.source.clone(),
                    modality: Some(point.modality.as_str().to_string()),
                    x_m: current_pose.map(|pose| pose.x_m),
                    y_m: current_pose.map(|pose| pose.y_m),
                    confidence: point.confidence.clamp(0.0, 1.0),
                    age_ms: now_ms.saturating_sub(point.observed_at_ms),
                    source_channel: Some(point.source.clone()),
                    observed_at_ms: Some(point.observed_at_ms),
                    vector_shape: graph_vector_shape(
                        latest,
                        point.source.as_str(),
                        point.modality.as_str(),
                    ),
                    nearest_cluster: nearest_cluster.clone(),
                    attached_text: point
                        .source
                        .strip_prefix("text:")
                        .map(str::to_string)
                        .filter(|text| !text.trim().is_empty()),
                });
            }
            if let Some(cluster_id) = nearest_cluster.as_ref() {
                push_graph_edge(
                    &mut edges,
                    &node_id,
                    cluster_id,
                    "belongs_to",
                    point.confidence,
                );
                let edge_id = format!("{node_id}->belongs_to->{cluster_id}");
                if let Some(edge) = edges.iter_mut().find(|edge| edge.id == edge_id) {
                    edge.observed_at_ms = Some(point.observed_at_ms);
                }
            }
            push_graph_edge(
                &mut edges,
                &node_id,
                &entity_id,
                "same_time_as",
                point.confidence.min(entity.confidence),
            );
        }
    }

    if let Some(action) = &latest.final_selected_action {
        push_graph_node(
            &mut nodes,
            "action_context:current",
            "action_context",
            &format!("{action:?}"),
            Some("action"),
            current_pose.map(|pose| (pose.x_m, pose.y_m)),
            0.86,
            0,
        );
        if current_pose.is_some() {
            push_graph_edge(
                &mut edges,
                "action_context:current",
                "place:current",
                "same_time_as",
                0.86,
            );
            push_graph_edge(
                &mut edges,
                "action_context:current",
                "cluster:odometry:trail",
                "predicts",
                0.42,
            );
            if has_range_cluster {
                push_graph_edge(
                    &mut edges,
                    "action_context:current",
                    "cluster:range:nearest",
                    "moves_with",
                    0.38,
                );
            }
        }
    }

    MapEntityGraph {
        schema_version: 1,
        generated_from: "live_map_mvp",
        nodes,
        edges,
        events,
    }
}

fn push_graph_node(
    nodes: &mut Vec<MapEntityGraphNode>,
    id: &str,
    node_type: &str,
    label: &str,
    modality: Option<&str>,
    position: Option<(f32, f32)>,
    confidence: f32,
    age_ms: TimeMs,
) {
    if nodes.iter().any(|node| node.id == id) {
        return;
    }
    nodes.push(MapEntityGraphNode {
        id: id.to_string(),
        node_type: node_type.to_string(),
        label: label.to_string(),
        modality: modality.map(str::to_string),
        x_m: position.map(|(x, _)| x),
        y_m: position.map(|(_, y)| y),
        confidence: confidence.clamp(0.0, 1.0),
        age_ms,
        source_channel: None,
        observed_at_ms: None,
        vector_shape: None,
        nearest_cluster: None,
        attached_text: None,
    });
}

fn push_graph_edge(
    edges: &mut Vec<MapEntityGraphEdge>,
    from: &str,
    to: &str,
    edge_type: &str,
    confidence: f32,
) {
    if from == to {
        return;
    }
    let id = format!("{from}->{edge_type}->{to}");
    if edges.iter().any(|edge| edge.id == id) {
        return;
    }
    edges.push(MapEntityGraphEdge {
        id,
        from: from.to_string(),
        to: to.to_string(),
        edge_type: edge_type.to_string(),
        confidence: confidence.clamp(0.0, 1.0),
        observed_at_ms: None,
    });
}

fn graph_id_fragment(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

fn graph_vector_shape(snapshot: &WorldSnapshot, source: &str, modality: &str) -> Option<String> {
    if source.starts_with("face:") {
        return snapshot
            .face
            .vectors
            .first()
            .map(|vector| format!("[{}]", vector.vector.len()));
    }
    if source.starts_with("voice:") {
        return snapshot
            .voice
            .vectors
            .first()
            .map(|vector| format!("[{}]", vector.vector.len()));
    }
    if source.starts_with("scene:")
        && snapshot.kinect.depth_width > 0
        && snapshot.kinect.depth_height > 0
    {
        return Some(format!(
            "{}x{}",
            snapshot.kinect.depth_width, snapshot.kinect.depth_height
        ));
    }
    if source.starts_with("text:") {
        return Some(format!(
            "tokens:{}",
            source.split_whitespace().count().max(1)
        ));
    }
    match modality {
        "vision" => Some("vision:observation".to_string()),
        "audio" => Some("audio:observation".to_string()),
        "depth" | "lidar" => Some("depth:observation".to_string()),
        "touch" => Some("touch:observation".to_string()),
        "motor" | "action" => Some("action:observation".to_string()),
        "language" => Some("text:observation".to_string()),
        _ => None,
    }
}

fn projected_beams_from_observation(
    observation: &MapObservation,
    now_ms: TimeMs,
) -> Vec<MapProjectedBeam> {
    observation
        .range_beams
        .iter()
        .map(|beam| {
            let end = project_beam_endpoint(observation.pose.pose, beam.angle_rad, beam.distance_m);
            MapProjectedBeam {
                origin_x_m: observation.pose.pose.x_m,
                origin_y_m: observation.pose.pose.y_m,
                end_x_m: end.x_m,
                end_y_m: end.y_m,
                angle_rad: beam.angle_rad,
                distance_m: beam.distance_m,
                hit: beam.hit,
                confidence: beam.confidence,
                age_ms: now_ms.saturating_sub(observation.t_ms),
            }
        })
        .collect()
}

fn map_view_cell(cell: &OdomMapCell, resolution_m: f32, now_ms: TimeMs) -> MapViewCell {
    MapViewCell {
        x: cell.key.x,
        y: cell.key.y,
        center_x_m: (cell.key.x as f32 + 0.5) * resolution_m,
        center_y_m: (cell.key.y as f32 + 0.5) * resolution_m,
        occupied_score: cell.occupied_score,
        free_score: cell.free_score,
        confidence: cell.confidence,
        age_ms: now_ms.saturating_sub(cell.last_seen_ms),
    }
}

fn map_semantic_cells(
    snapshot: &WorldSnapshot,
    metadata: Option<&LiveSceneMetadata>,
    now_ms: TimeMs,
) -> Vec<MapSemanticCell> {
    let mut cells = Vec::new();
    cells.extend(memory_semantic_cells(snapshot, now_ms));
    if let Some(metadata) = metadata {
        cells.extend(metadata.objects.iter().filter_map(|object| {
            let kind = semantic_kind_for_object(&object.kind)?;
            Some(MapSemanticCell {
                x_m: object.x_m,
                y_m: object.y_m,
                kind: kind.to_string(),
                score: 1.0,
                confidence: 1.0,
                age_ms: Some(0),
                label: object.label.clone().or_else(|| Some(object.id.clone())),
            })
        }));
    }
    if snapshot.body.charging {
        cells.push(MapSemanticCell {
            x_m: snapshot.body.odometry.x_m,
            y_m: snapshot.body.odometry.y_m,
            kind: "charger/charge".to_string(),
            score: 1.0,
            confidence: 0.9,
            age_ms: Some(0),
            label: Some("charging contact".to_string()),
        });
    }
    cells
}

fn memory_semantic_cells(snapshot: &WorldSnapshot, now_ms: TimeMs) -> Vec<MapSemanticCell> {
    let Some(value) = snapshot
        .to_now(snapshot.body.last_update_ms)
        .extensions
        .get("memory.semantic_map")
        .cloned()
    else {
        return Vec::new();
    };
    let mut cells = Vec::new();
    for (field, kind) in [
        ("danger_cells", "danger"),
        ("charge_cells", "charger/charge"),
        ("social_cells", "social"),
        ("novelty_cells", "novelty"),
    ] {
        if let Some(items) = value.get(field).and_then(|items| items.as_array()) {
            cells.extend(
                items
                    .iter()
                    .filter_map(|item| semantic_cell_from_value(item, kind, now_ms)),
            );
        }
    }
    if let Some(current) = value.get("current") {
        if let Some(cell) = semantic_cell_from_value(current, "current", now_ms) {
            cells.push(cell);
        }
    }
    cells
}

fn semantic_cell_from_value(
    value: &serde_json::Value,
    kind: &str,
    now_ms: TimeMs,
) -> Option<MapSemanticCell> {
    let x_m = value.get("center_x_m")?.as_f64()? as f32;
    let y_m = value.get("center_y_m")?.as_f64()? as f32;
    let last_seen = value.get("last_seen_tick").and_then(|value| value.as_u64());
    Some(MapSemanticCell {
        x_m,
        y_m,
        kind: kind.to_string(),
        score: value
            .get("score")
            .and_then(|value| value.as_f64())
            .unwrap_or(1.0) as f32,
        confidence: value
            .get("confidence")
            .and_then(|value| value.as_f64())
            .unwrap_or(1.0) as f32,
        age_ms: last_seen.map(|seen| now_ms.saturating_sub(seen)),
        label: value
            .get("last_observed_objects")
            .and_then(|value| value.as_array())
            .and_then(|items| items.first())
            .and_then(|value| value.as_str())
            .map(str::to_string),
    })
}

fn semantic_kind_for_object(kind: &str) -> Option<&'static str> {
    match kind {
        "charger" => Some("charger/charge"),
        "person" | "speaker" | "sound_source" => Some("social"),
        _ => None,
    }
}

fn map_event_markers(
    snapshot: &WorldSnapshot,
    metadata: Option<&LiveSceneMetadata>,
    now_ms: TimeMs,
) -> Vec<MapEventMarker> {
    let pose = snapshot.body.odometry;
    let mut events = Vec::new();
    if snapshot.body.flags.bump_left || snapshot.body.flags.bump_right {
        events.push(map_event_at_pose(pose.x_m, pose.y_m, "bump", 1.0, now_ms));
    }
    if snapshot.body.flags.cliff_left
        || snapshot.body.flags.cliff_front_left
        || snapshot.body.flags.cliff_front_right
        || snapshot.body.flags.cliff_right
        || snapshot.body.flags.wheel_drop
    {
        events.push(map_event_at_pose(pose.x_m, pose.y_m, "cliff", 1.0, now_ms));
    }
    if scene_stuck_from_snapshot(snapshot).active {
        events.push(map_event_at_pose(pose.x_m, pose.y_m, "stuck", 0.9, now_ms));
    }
    if snapshot
        .llm_action_proposal
        .as_ref()
        .and_then(|proposal| proposal.safety_vetoed.then_some(()))
        .is_some()
    {
        events.push(map_event_at_pose(
            pose.x_m,
            pose.y_m,
            "safety_override",
            1.0,
            now_ms,
        ));
    }
    if snapshot.body.charging {
        events.push(map_event_at_pose(
            pose.x_m, pose.y_m, "charger", 1.0, now_ms,
        ));
    }
    if let Some(metadata) = metadata {
        events.extend(metadata.objects.iter().filter_map(|object| {
            matches!(
                object.kind.as_str(),
                "charger" | "person" | "speaker" | "sound_source"
            )
            .then(|| MapEventMarker {
                x_m: object.x_m,
                y_m: object.y_m,
                kind: object.kind.clone(),
                confidence: 1.0,
                age_ms: 0,
                label: object.label.clone().or_else(|| Some(object.id.clone())),
            })
        }));
    }
    events
}

fn map_event_at_pose(
    x_m: f32,
    y_m: f32,
    kind: &str,
    confidence: f32,
    now_ms: TimeMs,
) -> MapEventMarker {
    MapEventMarker {
        x_m,
        y_m,
        kind: kind.to_string(),
        confidence,
        age_ms: now_ms.saturating_sub(now_ms),
        label: None,
    }
}
