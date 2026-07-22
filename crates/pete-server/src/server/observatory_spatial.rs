const MAX_RETAINED_DEPTH_SAMPLES: usize = 1_024;

fn strip_observatory_heavy_payloads(now: &mut pete_now::Now) {
    now.eye_frame = None;
    now.kinect.color_frame = None;
    now.kinect.ir.clear();
    now.kinect.player_index.clear();
    for detection in &mut now.objects.detections {
        detection.crop_rgb8.clear();
    }
    now.objects.vectors.clear();
    now.face.vectors.clear();
    now.voice.vectors.clear();
    let depth_len = now.kinect.depth_m.len();
    if depth_len > MAX_RETAINED_DEPTH_SAMPLES {
        let stride = depth_len.div_ceil(MAX_RETAINED_DEPTH_SAMPLES);
        now.kinect.depth_m = now.kinect.depth_m.iter().step_by(stride).copied().collect();
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
struct SpatialViewQuery {
    snapshot_id: Option<String>,
    at_or_before_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpatialDetectionView {
    pub source_frame_id: String,
    pub source_snapshot_id: String,
    pub track_id: Option<String>,
    pub labels: Vec<pete_now::VisionLabelHypothesis>,
    pub bbox: pete_now::VisionBoundingBox,
    pub image_width: u32,
    pub image_height: u32,
    pub model: pete_now::VisionModelIdentity,
    pub calibration_epoch: Option<u64>,
    pub geometry_trust: String,
    pub position: Option<pete_now::VisionPositionEstimate>,
    pub position_unavailable_reasons: Vec<String>,
    pub crop_asset_available: bool,
    pub downstream_event_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpatialDepthView {
    pub original_width: u32,
    pub original_height: u32,
    pub original_sample_count: usize,
    pub retained_samples: Vec<f32>,
    pub sample_stride: usize,
    pub min_depth_m: f32,
    pub max_depth_m: f32,
    pub registration_trusted: bool,
    pub registration_reasons: Vec<String>,
    pub calibration_epoch: Option<u64>,
    pub captured_at_ms: u64,
    pub frame_alignment: Option<pete_now::KinectFusionAlignment>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpatialViewResponse {
    pub snapshot_id: String,
    pub at_ms: u64,
    pub rgb_asset_url: Option<String>,
    pub rgb_asset_is_current: bool,
    pub detections: Vec<SpatialDetectionView>,
    pub depth: SpatialDepthView,
    pub raw_body_pose: pete_core::Pose2,
    pub map: Option<serde_json::Value>,
    pub map_alignment: String,
    pub raw_corrected_comparison_available: bool,
    pub navigation_trusted: bool,
    pub navigation_block_reasons: Vec<String>,
    pub lidar: String,
    pub missing_assets: Vec<String>,
}

fn build_spatial_view(
    selection: &ObservatoryNowSelection,
    map: Option<serde_json::Value>,
    rgb_asset_available: bool,
    events: &[BrainEvent],
) -> SpatialViewResponse {
    let now = &selection.selected.now;
    let has_depth = !now.kinect.depth_m.is_empty();
    let registration_trusted = !has_depth
        || (now
            .kinect
            .geometry_calibration
            .as_ref()
            .is_some_and(|geometry| geometry.physical_validation_ready())
            && pete_now::DepthGeometry::live_transform_trusted(&now.kinect));
    let mut registration_reasons = Vec::new();
    if has_depth && now.kinect.geometry_calibration.is_none() {
        registration_reasons.push("depth has no measured RGB-D geometry calibration".into());
    }
    if has_depth && !pete_now::DepthGeometry::live_transform_trusted(&now.kinect) {
        registration_reasons.push("live mount transform is not fully trusted".into());
    }
    if let Some(alignment) = &now.kinect.fusion_alignment {
        if alignment.captured_at_ms != now.kinect.captured_at_ms {
            registration_reasons.push("body/IMU alignment belongs to a mismatched frame".into());
        }
    }
    let depth_count =
        (now.kinect.depth_width as usize).saturating_mul(now.kinect.depth_height as usize);
    let sample_stride = if now.kinect.depth_m.is_empty() {
        1
    } else {
        depth_count
            .max(now.kinect.depth_m.len())
            .div_ceil(now.kinect.depth_m.len())
    };
    let geometry_epoch = now
        .kinect
        .live_geometry_calibration
        .as_ref()
        .map(|calibration| calibration.epoch.id);
    let detections: Vec<SpatialDetectionView> = now
        .objects
        .detections
        .iter()
        .map(|detection| SpatialDetectionView {
            source_frame_id: detection.source_frame_id.clone(),
            source_snapshot_id: detection.source_snapshot_id.clone(),
            track_id: detection.track_id.clone(),
            labels: detection.labels.clone(),
            bbox: detection.bbox,
            image_width: detection.image_width,
            image_height: detection.image_height,
            model: detection.model.clone(),
            calibration_epoch: detection.calibration_epoch,
            geometry_trust: detection.geometry_trust.clone(),
            position: detection.position.clone(),
            position_unavailable_reasons: detection.position_unavailable_reasons.clone(),
            crop_asset_available: false,
            downstream_event_ids: events
                .iter()
                .filter(|event| {
                    event.references.frame_id.as_deref() == Some(&detection.source_frame_id)
                        || event.references.snapshot_id.as_deref()
                            == Some(&detection.source_snapshot_id)
                })
                .map(|event| event.event_id.0.clone())
                .collect(),
        })
        .collect();
    let map_navigation_trusted = map
        .as_ref()
        .and_then(|map| map.pointer("/world_projection/navigation_trusted"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let mut navigation_block_reasons: Vec<String> = map
        .as_ref()
        .and_then(|map| map.pointer("/world_projection/reasons"))
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_string)
        .collect();
    if map.is_none() {
        navigation_block_reasons
            .push("map/point-cloud history is unavailable for this retained snapshot".into());
    }
    if !registration_trusted {
        navigation_block_reasons.extend(registration_reasons.clone());
    }
    navigation_block_reasons.sort();
    navigation_block_reasons.dedup();
    let mut missing_assets = Vec::new();
    if !rgb_asset_available {
        missing_assets.push("RGB frame is not retained for this snapshot".into());
    }
    if !detections.is_empty() {
        missing_assets.push(
            "detection crops are not retained; source frame references remain available".into(),
        );
    }
    let raw_corrected_comparison_available = map
        .as_ref()
        .and_then(|map| map.pointer("/pose_graph/optimization/max_node_update_m"))
        .and_then(serde_json::Value::as_f64)
        .is_some_and(|update| update > 0.0);
    SpatialViewResponse {
        snapshot_id: selection.selected.snapshot_id.clone(),
        at_ms: now.t_ms,
        rgb_asset_url: rgb_asset_available.then(|| "/view/retina/latest.png".into()),
        rgb_asset_is_current: rgb_asset_available,
        detections,
        depth: SpatialDepthView {
            original_width: now.kinect.depth_width,
            original_height: now.kinect.depth_height,
            original_sample_count: depth_count,
            retained_samples: now.kinect.depth_m.clone(),
            sample_stride,
            min_depth_m: now.kinect.min_depth_m,
            max_depth_m: now.kinect.max_depth_m,
            registration_trusted,
            registration_reasons,
            calibration_epoch: geometry_epoch,
            captured_at_ms: now.kinect.captured_at_ms,
            frame_alignment: now.kinect.fusion_alignment.clone(),
        },
        raw_body_pose: now.body.odometry,
        map_alignment: if map.is_some() {
            "exact live map at selected snapshot".into()
        } else {
            "unavailable; not substituted with current map".into()
        },
        raw_corrected_comparison_available,
        navigation_trusted: registration_trusted && map_navigation_trusted,
        navigation_block_reasons,
        map,
        lidar: "optional corroboration; not required and not observed by this view".into(),
        missing_assets,
    }
}

async fn get_observatory_spatial(
    State(state): State<LiveViewState>,
    Query(query): Query<SpatialViewQuery>,
) -> Result<Json<SpatialViewResponse>, ObservatoryHttpError> {
    let selection = match (query.snapshot_id.as_deref(), query.at_or_before_ms) {
        (Some(id), None) => state.observatory_now_snapshot(id),
        (None, Some(t_ms)) => state.observatory_now_at_or_before(t_ms),
        _ => {
            return Err(ObservatoryHttpError::bad_request(
                "provide exactly one of snapshot_id or at_or_before_ms",
            ))
        }
    }
    .ok_or_else(|| {
        ObservatoryHttpError::unavailable("requested spatial snapshot is not retained")
    })?;
    let latest = state.latest();
    let exact_live = latest
        .as_ref()
        .is_some_and(|latest| latest.body.last_update_ms == selection.selected.now.t_ms);
    let map = if exact_live {
        latest.as_ref().and_then(|latest| {
            serde_json::to_value(map_response_from_parts(
                &state.map_snapshot(),
                &state.point_cloud_snapshot(),
                latest,
                state.scene_metadata().as_ref(),
                &state.entity_memory_report(),
            ))
            .ok()
        })
    } else {
        None
    };
    let rgb_asset_available = exact_live
        && latest
            .as_ref()
            .is_some_and(|latest| latest.eye_frame.is_some());
    let history = state
        .observatory()
        .query(&BrainEventQuery {
            occurred_to_ms: Some(selection.selected.now.t_ms),
            limit: Some(MAX_OBSERVATORY_QUERY_LIMIT),
            ..Default::default()
        })
        .map_err(|error| ObservatoryHttpError::bad_request(error.to_string()))?;
    let events: Vec<BrainEvent> = history
        .records
        .into_iter()
        .map(|record| match record {
            BrainEventStreamRecord::Event { envelope } => envelope.event,
            BrainEventStreamRecord::Gap { gap } => gap.event,
        })
        .collect();
    Ok(Json(build_spatial_view(
        &selection,
        map,
        rgb_asset_available,
        &events,
    )))
}
