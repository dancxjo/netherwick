async fn run_representation_report(args: RepresentationReportArgs) -> Result<()> {
    let report = generate_representation_report(&args).await?;
    if let Some(parent) = Path::new(&args.out).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "representation report written to {} (frames={}, entities={}, voxels={})",
        args.out,
        report.frame_count,
        report.entity_memory.total_entities,
        report.map.point_cloud_voxel_count
    );
    Ok(())
}

async fn generate_representation_report(
    args: &RepresentationReportArgs,
) -> Result<RepresentationHealthReport> {
    let mut warnings = BTreeSet::new();
    let mut provenance: HashMap<String, usize> = HashMap::new();
    let mut place_memory = PlaceMemory::new();
    let mut entity_memory = EntityMemory::new();
    let mut local_map = LocalMap::default();
    let mut point_cloud = VoxelPointCloud::default();
    let mut pose_graph = PoseGraphBuilder::new(PoseGraphConfig::default());
    let mut place_candidates = Vec::new();
    let mut place_recognition_warnings = BTreeSet::new();
    let mut frame_count = 0usize;

    let mut saw_range = false;
    let mut saw_scene_vectors = false;
    let mut saw_objects = false;
    let mut saw_depth = false;
    let mut saw_audio = false;
    let mut calibration_evidence = ReturnToStartCalibrationEvidence::default();
    let physical_reference = args
        .physical_reference
        .as_deref()
        .map(|path| -> Result<ReturnToStartPhysicalReference> {
            let bytes = fs::read(path)
                .with_context(|| format!("reading physical reference sidecar {path}"))?;
            serde_json::from_slice(&bytes)
                .with_context(|| format!("decoding physical reference sidecar {path}"))
        })
        .transpose()?;

    let input = if let Some(capture) = args.capture.as_deref() {
        let reader = CaptureReader::open(capture).await?;
        let mut records = reader.read_frames().await?;
        records.sort_by_key(|record| record.t_ms);
        for record in records {
            frame_count += 1;
            *provenance
                .entry("capture_snapshot".to_string())
                .or_default() += 1;
            let frame_id = format!("capture-frame-{}", record.index);
            let mut now = record.snapshot.to_now(record.t_ms);
            observe_return_to_start_calibration(&mut calibration_evidence, &now);
            set_now_frame_id(&mut now, &frame_id);
            saw_range |= !now.range.beams.is_empty() || now.range.nearest_m.is_some();
            saw_scene_vectors |= !now.eye.scene_vectors.is_empty();
            saw_objects |= !now.objects.observations.is_empty();
            saw_depth |= !record.snapshot.kinect.depth_m.is_empty();
            saw_audio |= !now.ear.features.is_empty() || now.ear.transcript.is_some();

            let current_key =
                Some(place_memory.quantize(now.body.odometry.x_m, now.body.odometry.y_m));
            let live_loop_candidates = live_loop_candidates_from_now(
                &place_memory,
                &now,
                current_key,
                Some(frame_id.clone()),
            );
            place_memory.observe_now(&now);
            entity_memory.observe_now(&now, current_key);
            let map_observation = observation_from_now(&now, local_map.config);
            local_map
                .integrate_observation_with_loop_candidates(map_observation, &live_loop_candidates);
            point_cloud.observe_snapshot(&record.snapshot, record.t_ms);
            observe_pose_graph_now(&mut pose_graph, &mut place_memory, &now, Some(frame_id));

            let output =
                place_memory.recognize_places_report(current_key, &now.eye.scene_vectors, 0.0, 20);
            if let Some(reason) = output.not_enough_evidence {
                place_recognition_warnings.insert(reason);
            }
            place_candidates.extend(output.candidates);
        }
        RepresentationInputSummary {
            source_type: "capture".to_string(),
            source_path: capture.to_string(),
            provenance,
        }
    } else {
        let ledger = JsonlLedger::new(&args.ledger);
        let mut frames = ledger.range(0, u64::MAX).await?;
        frames.sort_by_key(|frame| frame.t_ms);
        for frame in &frames {
            frame_count += 1;
            let place_input = place_recognition_input_from_frame(frame);
            *provenance
                .entry(place_input.provenance.clone())
                .or_default() += 1;

            let now = &frame.now;
            observe_return_to_start_calibration(&mut calibration_evidence, now);
            saw_range |= !now.range.beams.is_empty() || now.range.nearest_m.is_some();
            saw_scene_vectors |= !now.eye.scene_vectors.is_empty();
            saw_objects |= !now.objects.observations.is_empty();
            saw_depth |= !now.kinect.depth_m.is_empty();
            saw_audio |= !now.ear.features.is_empty() || now.ear.transcript.is_some();

            let current_key =
                Some(place_memory.quantize(now.body.odometry.x_m, now.body.odometry.y_m));
            let live_loop_candidates =
                live_loop_candidates_from_frame(&place_memory, frame, current_key);
            place_memory.observe_frame(frame);
            entity_memory.observe_now(now, current_key);
            let map_now = now_with_frame_id(now, &frame.id.to_string());
            let map_observation = observation_from_now(&map_now, local_map.config);
            local_map
                .integrate_observation_with_loop_candidates(map_observation, &live_loop_candidates);
            point_cloud.decay_stale(now.t_ms);
            observe_pose_graph_frame(&mut pose_graph, &mut place_memory, frame);

            let mut query_vectors = now.eye.scene_vectors.clone();
            query_vectors.extend(place_recognition_vectors_from_input(&place_input));
            let output = place_memory.recognize_places_report(current_key, &query_vectors, 0.0, 20);
            if let Some(reason) = output.not_enough_evidence {
                place_recognition_warnings.insert(reason);
            }
            place_candidates.extend(output.candidates);
        }
        RepresentationInputSummary {
            source_type: "ledger".to_string(),
            source_path: args.ledger.clone(),
            provenance,
        }
    };

    if frame_count == 0 {
        warnings.insert("no frames found in input".to_string());
    }
    if !saw_range {
        warnings.insert("range sensor data missing across all frames".to_string());
    }
    if !saw_scene_vectors {
        warnings.insert("scene vectors missing across all frames".to_string());
    }
    if !saw_objects {
        warnings.insert("object observations missing across all frames".to_string());
    }
    if !saw_depth {
        warnings.insert("depth channel missing across all frames".to_string());
    }
    if !saw_audio {
        warnings.insert("audio/transcript channel missing across all frames".to_string());
    }

    let entity_report = entity_memory.report();
    let revived_entities = entity_memory
        .entities
        .values()
        .filter(|entity| entity.constellation.state == EntityConstellationState::Revived)
        .count();
    let mut modality_support_counts = HashMap::new();
    let mut constellation_edges_by_relation = HashMap::new();
    for entity in entity_memory.entities.values() {
        if !entity.modality_support.face_vector_ids.is_empty() {
            *modality_support_counts
                .entry("face".to_string())
                .or_default() += 1;
        }
        if !entity.modality_support.voice_vector_ids.is_empty() {
            *modality_support_counts
                .entry("voice".to_string())
                .or_default() += 1;
        }
        if !entity.modality_support.scene_vector_ids.is_empty() {
            *modality_support_counts
                .entry("scene".to_string())
                .or_default() += 1;
        }
        if !entity.modality_support.text_labels.is_empty() {
            *modality_support_counts
                .entry("text".to_string())
                .or_default() += 1;
        }
        for edge in &entity.constellation.binding_edges {
            *constellation_edges_by_relation
                .entry(binding_relation_label(edge.relation.clone()).to_string())
                .or_default() += 1;
        }
    }

    let map_summary = local_map.summary();
    let point_cloud_summary = point_cloud.summary();
    if point_cloud_summary.observations == 0 {
        warnings.insert("point cloud received no usable observations".to_string());
    }

    let pose_graph_report = pose_graph.finish_report();
    let confidence_values = place_candidates
        .iter()
        .map(|candidate| candidate.confidence)
        .collect::<Vec<_>>();
    let mut candidate_kinds = HashMap::new();
    let mut same_place_cells: HashMap<(i32, i32), usize> = HashMap::new();
    for candidate in &place_candidates {
        let kind = match &candidate.kind {
            PlaceRecognitionKind::SamePlace => "same_place",
            PlaceRecognitionKind::SimilarPlace => "similar_place",
            PlaceRecognitionKind::EntityConstellation => "entity_constellation",
        };
        *candidate_kinds.entry(kind.to_string()).or_default() += 1;
        if matches!(&candidate.kind, PlaceRecognitionKind::SamePlace) {
            *same_place_cells
                .entry((candidate.cell.x, candidate.cell.y))
                .or_default() += 1;
        }
    }
    let repeated_place_hints = same_place_cells
        .iter()
        .filter(|(_, count)| **count > 1)
        .map(|((x, y), count)| format!("cell ({x}, {y}) recognized {count} times"))
        .take(5)
        .collect::<Vec<_>>();
    if place_candidates.is_empty() {
        place_recognition_warnings.insert("no place-recognition candidates emitted".to_string());
    }

    let return_to_start = return_to_start_validation(
        &local_map,
        &calibration_evidence,
        physical_reference,
    );
    Ok(RepresentationHealthReport {
        schema_version: 3,
        frame_count,
        input,
        warnings: warnings.into_iter().collect(),
        entity_memory: RepresentationEntityMemorySummary {
            total_entities: entity_report.total_entities,
            active_entities: entity_report.active_entities,
            occluded_entities: entity_report.occluded_entities,
            vanished_entities: entity_report.vanished_entities,
            revived_entities,
            modality_support_counts,
            constellation_edges_by_relation,
        },
        map: RepresentationMapSummary {
            local_occupancy_cell_count: map_summary.occupied_cells,
            pose_history_length: local_map.pose_history.len(),
            point_cloud_voxel_count: point_cloud_summary.voxels,
            stable_voxel_count: point_cloud_summary.stable_voxels,
            transient_voxel_count: point_cloud_summary.transient_voxels,
        },
        pose_graph: RepresentationPoseGraphSummary {
            node_count: pose_graph_report.nodes,
            odometry_edge_count: pose_graph_report.odometry_edges,
            loop_candidate_count: pose_graph_report.loop_candidate_edges,
            loop_accepted_count: pose_graph_report.active_loop_candidate_edges,
            loop_rejected_count: pose_graph_report.rejected_loop_candidates,
            confidence_distribution: RepresentationConfidenceDistribution {
                min: pose_graph_report.confidence_distribution.min,
                max: pose_graph_report.confidence_distribution.max,
                mean: pose_graph_report.confidence_distribution.mean,
                buckets: pose_graph_report
                    .confidence_distribution
                    .buckets
                    .into_iter()
                    .collect(),
            },
        },
        place_recognition: RepresentationPlaceRecognitionSummary {
            candidates_emitted: place_candidates.len(),
            candidate_kinds,
            confidence_distribution: summarize_confidence_distribution(&confidence_values),
            repeated_place_hints,
            warnings: place_recognition_warnings.into_iter().collect(),
        },
        return_to_start,
    })
}

fn return_to_start_validation(
    map: &LocalMap,
    calibration: &ReturnToStartCalibrationEvidence,
    physical_reference: Option<ReturnToStartPhysicalReference>,
) -> ReturnToStartValidation {
    const MAX_FINAL_DISTANCE_M: f32 = 0.25;
    let registration = map.pose_graph.edges.iter().rev().find_map(|edge| {
        if !edge.active {
            return None;
        }
        match &edge.source {
            PoseEdgeSource::LoopClosureCandidate { registration, .. } => registration.as_ref(),
            _ => None,
        }
    });
    let optimization = map.pose_graph_optimization;
    let graph_error_reduced = optimization.initial_mean_error > 0.0
        && optimization.final_mean_error < optimization.initial_mean_error;
    let wall_overlap_before = registration.map(|measurement| measurement.odometry_geometric_overlap);
    let wall_overlap_after = registration.map(|measurement| measurement.geometric_overlap);
    let wall_overlap_improved = matches!(
        (wall_overlap_before, wall_overlap_after),
        (Some(before), Some(after)) if after > before
    );
    let raw_poses = map
        .observations
        .iter()
        .filter_map(|observation| {
            let pose = observation.source_snapshot.pointer("/body/odometry")?;
            Some((
                pose.get("x_m")?.as_f64()? as f32,
                pose.get("y_m")?.as_f64()? as f32,
                pose.get("heading_rad")?.as_f64()? as f32,
            ))
        })
        .collect::<Vec<_>>();
    let raw_final_distance_to_start_m = raw_poses.first().zip(raw_poses.last()).map(|(start, end)| {
        (end.0 - start.0).hypot(end.1 - start.1)
    });
    let raw_final_heading_error_deg = raw_poses
        .first()
        .zip(raw_poses.last())
        .map(|(start, end)| angle_difference(end.2, start.2).abs().to_degrees());
    let corrected_final_distance_to_start_m = map
        .pose_graph
        .nodes
        .first()
        .zip(map.pose_graph.nodes.last())
        .map(|(start, end)| {
            (end.pose_estimate.pose.x_m - start.pose_estimate.pose.x_m)
                .hypot(end.pose_estimate.pose.y_m - start.pose_estimate.pose.y_m)
        });
    let corrected_final_heading_error_deg = map
        .pose_graph
        .nodes
        .first()
        .zip(map.pose_graph.nodes.last())
        .map(|(start, end)| {
            angle_difference(
                end.pose_estimate.pose.heading_rad,
                start.pose_estimate.pose.heading_rad,
            )
            .abs()
            .to_degrees()
        });
    let corrected_endpoint_improves_over_raw = raw_final_distance_to_start_m
        .zip(corrected_final_distance_to_start_m)
        .is_some_and(|(raw, corrected)| corrected < raw);
    let corrected_pose_near_start = corrected_final_distance_to_start_m
        .is_some_and(|distance| distance <= MAX_FINAL_DISTANCE_M);
    let evaluated = map.pose_graph.nodes.len() >= 3 && registration.is_some();
    let mut reasons = Vec::new();
    if registration.is_none() {
        reasons.push("no active measured loop registration".to_string());
    }
    if !graph_error_reduced {
        reasons.push("pose-graph error did not decrease".to_string());
    }
    if !wall_overlap_improved {
        reasons.push("registered wall overlap did not improve over raw odometry".to_string());
    }
    if !corrected_pose_near_start {
        reasons.push(format!(
            "corrected final pose is not within {MAX_FINAL_DISTANCE_M:.2}m of the start"
        ));
    }
    if !corrected_endpoint_improves_over_raw {
        reasons.push("corrected endpoint did not improve over raw odometry".to_string());
    }
    let loop_direction = loop_direction(&raw_poses);
    let physical_measurement_passed = physical_reference.as_ref().is_some_and(|reference| {
        reference.actual_endpoint_distance_m.is_finite()
            && reference.actual_orientation_error_deg.is_finite()
            && reference.distance_tolerance_m.is_finite()
            && reference.orientation_tolerance_deg.is_finite()
            && reference.actual_endpoint_distance_m >= 0.0
            && reference.distance_tolerance_m >= 0.0
            && reference.orientation_tolerance_deg >= 0.0
            && reference.actual_endpoint_distance_m <= reference.distance_tolerance_m
            && reference.actual_orientation_error_deg.abs()
                <= reference.orientation_tolerance_deg
    });
    let physical_direction_matches = physical_reference.as_ref().is_some_and(|reference| {
        loop_direction == "unobservable"
            || reference.direction.trim().eq_ignore_ascii_case(&loop_direction)
    });
    if physical_reference.is_none() {
        reasons.push("independent physical endpoint/orientation sidecar is missing".to_string());
    } else if !physical_measurement_passed {
        reasons.push("physical endpoint or orientation exceeds declared tolerance".to_string());
    }
    if physical_reference.is_some() && !physical_direction_matches {
        reasons.push("physical direction label contradicts the replayed loop direction".to_string());
    }
    let remount_detected = calibration.epoch_ids.len() > 1 || calibration.saw_invalidated;
    if remount_detected && !calibration.last_epoch_trusted {
        reasons.push("mount calibration did not reconverge after epoch change".to_string());
    }
    if !calibration.kinect_present {
        reasons.push("Kinect geometry is absent".to_string());
    }
    if !calibration.uncertainty_reported {
        reasons.push("calibration covariance/uncertainty is absent".to_string());
    }
    let passed = evaluated
        && graph_error_reduced
        && wall_overlap_improved
        && corrected_pose_near_start
        && corrected_endpoint_improves_over_raw;
    let navigation_trusted = passed
        && physical_measurement_passed
        && physical_direction_matches
        && calibration.kinect_present
        && calibration.uncertainty_reported
        && (!remount_detected || calibration.last_epoch_trusted);
    ReturnToStartValidation {
        evaluated,
        passed,
        measured_loop_constraint: registration.is_some(),
        graph_error_before: optimization.initial_mean_error,
        graph_error_after: optimization.final_mean_error,
        graph_error_reduced,
        wall_overlap_before,
        wall_overlap_after,
        wall_overlap_improved,
        raw_final_distance_to_start_m,
        corrected_final_distance_to_start_m,
        raw_final_heading_error_deg,
        corrected_final_heading_error_deg,
        corrected_endpoint_improves_over_raw,
        max_corrected_distance_to_start_m: MAX_FINAL_DISTANCE_M,
        corrected_pose_near_start,
        loop_direction,
        physical_reference,
        physical_measurement_passed,
        physical_direction_matches,
        calibration_epoch_ids: calibration.epoch_ids.clone(),
        remount_detected,
        reconverged_after_remount: remount_detected && calibration.last_epoch_trusted,
        kinect_present: calibration.kinect_present,
        lidar_present: calibration.lidar_present,
        geometry_mode: if calibration.lidar_present {
            "kinect_with_optional_lidar".to_string()
        } else {
            "kinect_only".to_string()
        },
        calibration_uncertainty_reported: calibration.uncertainty_reported,
        navigation_trusted,
        navigation_trust_decision: if navigation_trusted {
            "trusted".to_string()
        } else {
            "withheld".to_string()
        },
        reasons,
    }
}

fn observe_return_to_start_calibration(
    evidence: &mut ReturnToStartCalibrationEvidence,
    now: &Now,
) {
    evidence.kinect_present |= !now.kinect.depth_m.is_empty();
    evidence.lidar_present |= now.range.source.as_deref().is_some_and(|source| {
        source.contains("lidar") || source.contains("lfcd")
    });
    if let Some(estimate) = now.kinect.live_geometry_calibration.as_ref() {
        if !evidence.epoch_ids.contains(&estimate.epoch.id) {
            evidence.epoch_ids.push(estimate.epoch.id);
        }
        evidence.saw_invalidated |=
            estimate.trust_state == pete_now::CalibrationTrustState::Invalidated;
        evidence.last_epoch_trusted =
            estimate.trust_state == pete_now::CalibrationTrustState::Trusted;
        evidence.uncertainty_reported |= estimate
            .covariance
            .iter()
            .all(|value| value.is_finite());
    }
}

fn loop_direction(poses: &[(f32, f32, f32)]) -> String {
    let signed_area = poses
        .windows(2)
        .map(|window| window[0].0 * window[1].1 - window[1].0 * window[0].1)
        .sum::<f32>();
    if signed_area > 0.001 {
        "counter_clockwise".to_string()
    } else if signed_area < -0.001 {
        "clockwise".to_string()
    } else {
        "unobservable".to_string()
    }
}

fn angle_difference(left: f32, right: f32) -> f32 {
    (left - right + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
        - std::f32::consts::PI
}

fn summarize_confidence_distribution(values: &[f32]) -> RepresentationConfidenceDistribution {
    let mut buckets = HashMap::new();
    for value in values {
        let bucket = if *value < 0.25 {
            "0.00-0.24"
        } else if *value < 0.5 {
            "0.25-0.49"
        } else if *value < 0.75 {
            "0.50-0.74"
        } else {
            "0.75-1.00"
        };
        *buckets.entry(bucket.to_string()).or_default() += 1;
    }
    let min = values
        .iter()
        .copied()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let max = values
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = (!values.is_empty()).then_some(values.iter().sum::<f32>() / values.len() as f32);
    RepresentationConfidenceDistribution {
        min,
        max,
        mean,
        buckets,
    }
}

fn binding_relation_label(relation: BindingRelation) -> &'static str {
    match relation {
        BindingRelation::CooccursInTime => "cooccurs_in_time",
        BindingRelation::CooccursInEstimatedSpace => "cooccurs_in_estimated_space",
        BindingRelation::MovesTogether => "moves_together",
        BindingRelation::PredictsSameFutureEvents => "predicts_same_future_events",
        BindingRelation::NamedBy => "named_by",
        BindingRelation::ProjectsTo => "projects_to",
        BindingRelation::HasColorAtPose => "has_color_at_pose",
        BindingRelation::LikelySameEntity => "likely_same_entity",
        BindingRelation::ExplainsOutcome => "explains_outcome",
        BindingRelation::Contradicts => "contradicts",
        BindingRelation::RequiresReview => "requires_review",
    }
}

async fn generate_pose_graph_report(args: &PoseGraphReportArgs) -> Result<PoseGraphReport> {
    let config = PoseGraphConfig {
        min_node_distance_m: args.min_node_distance_m,
        min_node_heading_delta_rad: args.min_node_degrees.to_radians(),
        max_ticks_between_nodes: args.max_ticks_between_nodes,
        min_loop_confidence: args.min_loop_confidence,
        ..PoseGraphConfig::default()
    };
    let mut builder = PoseGraphBuilder::new(config);
    let mut memory = PlaceMemory::new();

    if let Some(capture) = args.capture.as_deref() {
        let reader = CaptureReader::open(capture).await?;
        let mut records = reader.read_frames().await?;
        records.sort_by_key(|record| record.t_ms);
        for record in &records {
            let frame_id = format!("capture-frame-{}", record.index);
            let mut now = record.snapshot.to_now(record.t_ms);
            set_now_frame_id(&mut now, &frame_id);
            observe_pose_graph_now(&mut builder, &mut memory, &now, Some(frame_id));
            memory.observe_now(&now);
        }
    } else {
        let ledger = JsonlLedger::new(&args.ledger);
        let mut frames = ledger.range(0, u64::MAX).await?;
        frames.sort_by_key(|frame| frame.t_ms);
        for frame in &frames {
            observe_pose_graph_frame(&mut builder, &mut memory, frame);
            memory.observe_frame(frame);
        }
    }

    Ok(builder.finish_report())
}

fn observe_pose_graph_now(
    builder: &mut PoseGraphBuilder,
    memory: &mut PlaceMemory,
    now: &Now,
    source_frame_id: Option<String>,
) {
    let current_key = Some(memory.quantize(now.body.odometry.x_m, now.body.odometry.y_m));
    let place_candidates = memory.recognize_places(current_key, &now.eye.scene_vectors, 0.0, 20);
    let entity_labels = entity_labels_from_now(now);
    let entity_candidates =
        memory.recognize_entity_constellations(current_key, &entity_labels, 0.0, 10);
    let loop_candidates = place_candidates
        .iter()
        .chain(entity_candidates.iter())
        .map(|candidate| place_candidate_to_loop_input(candidate, source_frame_id.clone()))
        .collect::<Vec<_>>();
    builder.observe(
        now.body.odometry,
        now.t_ms,
        source_frame_id,
        &loop_candidates,
    );
}

fn live_loop_candidates_from_now(
    memory: &PlaceMemory,
    now: &Now,
    current_key: Option<pete_memory::PlaceCellKey>,
    source_frame_id: Option<String>,
) -> Vec<LoopClosureCandidateInput> {
    let place_candidates = memory.recognize_places(current_key, &now.eye.scene_vectors, 0.85, 10);
    let entity_labels = entity_labels_from_now(now);
    let entity_candidates =
        memory.recognize_entity_constellations(current_key, &entity_labels, 0.85, 10);
    place_candidates
        .iter()
        .chain(entity_candidates.iter())
        .map(|candidate| place_candidate_to_loop_input(candidate, source_frame_id.clone()))
        .collect()
}

fn observe_pose_graph_frame(
    builder: &mut PoseGraphBuilder,
    memory: &mut PlaceMemory,
    frame: &ExperienceFrame,
) {
    let current_key =
        Some(memory.quantize(frame.now.body.odometry.x_m, frame.now.body.odometry.y_m));
    let place_input = place_recognition_input_from_frame(frame);
    let mut query_vectors = frame.now.eye.scene_vectors.clone();
    query_vectors.extend(place_recognition_vectors_from_input(&place_input));
    let place_candidates = memory.recognize_places(current_key, &query_vectors, 0.0, 20);
    let entity_labels = entity_labels_from_place_input(&place_input);
    let entity_candidates =
        memory.recognize_entity_constellations(current_key, &entity_labels, 0.0, 10);
    let loop_candidates = place_candidates
        .iter()
        .chain(entity_candidates.iter())
        .map(|candidate| place_candidate_to_loop_input(candidate, Some(frame.id.to_string())))
        .collect::<Vec<_>>();
    builder.observe(
        frame.now.body.odometry,
        frame.t_ms,
        Some(frame.id.to_string()),
        &loop_candidates,
    );
}

fn live_loop_candidates_from_frame(
    memory: &PlaceMemory,
    frame: &ExperienceFrame,
    current_key: Option<pete_memory::PlaceCellKey>,
) -> Vec<LoopClosureCandidateInput> {
    let place_input = place_recognition_input_from_frame(frame);
    let mut query_vectors = frame.now.eye.scene_vectors.clone();
    query_vectors.extend(place_recognition_vectors_from_input(&place_input));
    let place_candidates = memory.recognize_places(current_key, &query_vectors, 0.85, 10);
    let entity_labels = entity_labels_from_place_input(&place_input);
    let entity_candidates =
        memory.recognize_entity_constellations(current_key, &entity_labels, 0.85, 10);
    place_candidates
        .iter()
        .chain(entity_candidates.iter())
        .map(|candidate| place_candidate_to_loop_input(candidate, Some(frame.id.to_string())))
        .collect()
}

fn now_with_frame_id(now: &Now, frame_id: &str) -> Now {
    let mut now = now.clone();
    set_now_frame_id(&mut now, frame_id);
    now
}

fn set_now_frame_id(now: &mut Now, frame_id: &str) {
    now.extensions.insert(
        "frame_id".to_string(),
        serde_json::Value::String(frame_id.to_string()),
    );
}

fn entity_labels_from_now(now: &Now) -> Vec<String> {
    let mut labels: Vec<String> = now
        .objects
        .observations
        .iter()
        .filter(|obs| obs.confidence >= 0.3)
        .map(|obs| obs.label.clone())
        .collect();
    labels.sort();
    labels.dedup();
    labels
}

fn entity_labels_from_place_input(input: &pete_memory::PlaceRecognitionInput) -> Vec<String> {
    let mut labels: Vec<String> = input
        .object_labels
        .iter()
        .chain(input.person_labels.iter())
        .cloned()
        .collect();
    labels.sort();
    labels.dedup();
    labels
}

fn place_candidate_to_loop_input(
    candidate: &PlaceRecognitionCandidate,
    source_frame_id: Option<String>,
) -> LoopClosureCandidateInput {
    LoopClosureCandidateInput {
        target_pose: pete_core::Pose2 {
            x_m: candidate.cell.center_x_m,
            y_m: candidate.cell.center_y_m,
            heading_rad: 0.0,
        },
        confidence: candidate.confidence,
        similarity: candidate.similarity,
        kind: match candidate.kind {
            PlaceRecognitionKind::SamePlace => "same_place",
            PlaceRecognitionKind::SimilarPlace => "similar_place",
            PlaceRecognitionKind::EntityConstellation => "entity_constellation",
        }
        .to_string(),
        target_frame_id: candidate
            .source_instant_frame_id
            .clone()
            .or_else(|| candidate.source_frame_id.clone()),
        source_frame_id,
        source_experience_id: candidate.source_experience_id.clone(),
        source_instant_frame_id: candidate.source_instant_frame_id.clone(),
        source_vector_refs: candidate.source_vector_refs.clone(),
        source_vector_id: Some(candidate.source_vector_id.clone()),
        query_vector_id: candidate.query_vector_id.clone(),
        query_experience_id: candidate.query_experience_id.clone(),
    }
}
