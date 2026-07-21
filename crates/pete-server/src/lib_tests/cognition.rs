#[tokio::test]
async fn capture_frame_to_scene_conversion_works_for_tiny_capture() {
    let root = unique_test_dir("capture-scene");
    let mut writer = CaptureWriter::create(&root, CaptureSource::Sim, Some(100))
        .await
        .unwrap();
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.last_update_ms = 99;
    snapshot.body.odometry.x_m = 1.0;
    snapshot.range.beams = vec![0.5];
    writer
        .append_snapshot(99, snapshot, Vec::new())
        .await
        .unwrap();
    writer.finish().await.unwrap();

    let Json(scene) = get_capture_scene(Query(CaptureSceneQuery {
        capture: root.clone(),
        frame: 0,
    }))
    .await
    .unwrap();

    assert_eq!(scene.t_ms, 99);
    assert_eq!(scene.body.x_m, 1.0);
    assert_eq!(scene.range.beams.len(), 1);
    std::fs::remove_dir_all(root).ok();
}

#[tokio::test]
async fn live_routes_include_3d_and_scene_endpoints() {
    assert!(HTTP_ENDPOINTS.contains(&"/view"));
    assert!(HTTP_ENDPOINTS.contains(&"/view/3d"));
    assert!(HTTP_ENDPOINTS.contains(&"/view/scene"));
    assert!(HTTP_ENDPOINTS.contains(&"/view/map"));
    assert!(HTTP_ENDPOINTS.contains(&"/view/embodied"));
    assert!(HTTP_ENDPOINTS.contains(&"/view/embodied/graph"));
    assert!(HTTP_ENDPOINTS.contains(&"/api/experience/lineage"));
    assert!(HTTP_ENDPOINTS.contains(&"/debug/embodied"));
    assert!(HTTP_ENDPOINTS.contains(&"/debug/embodied/graph"));
    assert!(HTTP_ENDPOINTS.contains(&"/models"));
    assert!(HTTP_ENDPOINTS.contains(&"/stream/llm"));
    assert!(HTTP_ENDPOINTS.contains(&"/reign/command/ws"));
    let Html(live_page) = live_view_page().await;
    assert!(live_page.contains("Embodied lineage"));
    assert!(live_page.contains("/api/experience/lineage"));
    assert!(live_page.contains("graph_query"));
    assert!(live_page.contains("graph_modality"));
    let Html(page) = live_view_3d_page().await;
    assert!(page.contains("Instant 3D"));
    assert!(page.contains("/view/snapshot"));
    assert!(page.contains("/models"));
    assert!(page.contains("Training stats"));
    assert!(page.contains("Connections"));
    assert!(page.contains("/stream/llm"));
    assert!(page.contains("LLM streams"));
    assert!(page.contains("navigator.xr"));
    assert!(page.contains("window.isSecureContext"));
    assert!(page.contains("/reign/command"));
    assert!(page.contains("/reign/command/ws"));
    assert!(page.contains("/view/behavior-nodes"));
    assert!(page.contains("behaviorInspector"));
    assert!(page.contains("packet.behavior_nodes"));
    assert!(page.contains("const nodes = {...graphLayout}"));
    assert!(page.contains("source='Gamepad'"));
    assert!(page.contains("type:'Drive'"));
    assert!(page.contains("data-reign-turn=\"Left\""));
    assert!(page.contains("data-reign-turn=\"Right\""));
    assert!(page.contains("function postTurnOnly"));
    assert!(page.contains("type:'Turn', direction"));
    assert!(page.contains("type:'Speak'"));
    assert!(page.contains("type:'Chirp'"));
    assert!(page.contains("data-chirp=\"Confirm\""));
    assert!(page.contains("data-chirp=\"GoalAcquired\""));
    assert!(page.contains("data-chirp=\"PersonRecognized\""));
    assert!(page.contains("notes 72,79,84,91"));
    assert!(page.contains("WASD / arrow keys"));
    assert!(page.contains("keyboardReignCodes"));
    assert!(page.contains("KeyW"));
    assert!(page.contains("ArrowUp"));
    assert!(page.contains("function syncVisualFloor"));
    assert!(page.contains("const center = world(centerX, centerY, 0);"));
    assert!(page.contains("viewerCamera.setTarget(center);"));
    assert!(!page.contains("selectProjectedFloor"));
    assert!(!page.contains("quaternionFromFloorNormal"));
    assert!(page.contains("Real hardware armed"));
    assert!(page.contains("id=\"reign-voice-panel\""));
    assert!(page.contains("id=\"reign-hardware\""));
    assert!(page.contains("id=\"reign-map\""));
    assert!(page.contains("id=\"reign-constellation\""));
    assert!(page.contains("'reign-voice-panel'"));
    assert!(page.contains("'reign-hardware'"));
    assert!(page.contains("'reign-map'"));
    assert!(page.contains("'reign-constellation'"));
    assert!(page.contains("/reign/hardware-arm"));
    let Html(reign_page) = reign_page().await;
    assert!(reign_page.contains("chirp('Confirm')"));
    assert!(reign_page.contains("chirp('GoalAcquired')"));
    assert!(reign_page.contains("72,79,84,91; little fanfare"));
    assert!(reign_page.contains("chirp('DidntUnderstand')"));
    assert!(page.contains("window.addEventListener('pagehide'"));
    assert!(page.contains("navigator.sendBeacon"));
    assert!(!page.contains("id=\"reign-dock\""));
    assert!(!page.contains("id=\"reign-explore\""));
    assert!(!page.contains("data-cockpit"));
    assert!(!page.contains("data-heading-nudge"));
    assert!(!page.contains("function startCockpitHold"));
    assert!(!page.contains("function commandForHeadingTarget"));
    assert!(page.contains("'WebRemote'"));
    assert!(page.contains("function projectRangeBeam"));
    assert!(page.contains("/view/map"));
    assert!(page.contains("data-map-layer=\"occupancy\""));
    assert!(page.contains("data-map-layer=\"rays\""));
    assert!(page.contains("data-map-layer=\"raw point cloud\""));
    assert!(page.contains("data-map-layer=\"accumulated occupancy\""));
    assert!(page.contains("data-map-layer=\"flat image\""));
    assert!(page.contains("data-map-layer=\"hypotheses\""));
    assert!(page.contains("function syncDisplayToggles"));
    assert!(page.contains("data-map-layer=\"stable wall candidates\""));
    assert!(page.contains("function renderWorldBeliefPoints"));
    assert!(page.contains("function packetHeadingRad"));
    assert!(page.contains("function robotYawToBabylon"));
    assert!(page.contains("function renderRobotMotion"));
    assert!(page.contains("motionState.connections"));
    assert!(page.contains("scanConnections"));
    assert!(page.contains("function pointCloudFrameKind"));
    assert!(page.contains("function robotRenderPointToBabylonLocal"));
    assert!(page.contains("function kinectCameraPointToBabylonLocal"));
    assert!(page.contains("function worldMathPointToBabylonWorld"));
    assert!(page.contains("new BABYLON.Vector3(p.x, -p.y, -p.z)"));
    assert!(page.contains("new BABYLON.Vector3(-p.y, p.z, p.x)"));
    assert!(page.contains("TransformCoordinates(kinectCameraPointToBabylonLocal(p), robotMatrix)"));
    assert!(page.contains("return worldMathPointToBabylonWorld(p);"));
    assert!(!page.contains("eyePanel.scaling.x = -1"));
    assert!(page.contains("function drawMirroredImageToEyeCanvas"));
    assert!(page.contains("mirroredImageTargetOffset"));
    assert!(page.contains("function renderPersistentWorldBelief"));
    assert!(page.contains("local_world_belief"));
    assert!(page.contains("roll_pitch_corrected"));
    assert!(page.contains("id=\"map-trust\""));
    assert!(page.contains("function updateMapTrust"));
    assert!(page.contains("liveMap?.world_projection?.cells"));
    assert!(page
        .contains("Obstacle cells project the same calibrated odometry-world voxels shown in 3D."));
    assert!(page.contains("const traceLocal = (x, y) =>"));
    assert!(page.contains("forward: dx * headingCos + dy * headingSin"));
    assert!(page.contains("function occupancyGridCellCenter"));
    assert!(page.contains("forward: (Number(cell.x) + .5) * res"));
    assert!(page.contains("left: (Number(cell.y) + .5) * res"));
    assert!(page.contains("const center = occupancyGridCellCenter(cell, grid);"));
    assert!(page.contains("const gridPose = packet.body || latest;"));
    assert!(page.contains("occupancyGridCellToWorld(gridPose, cell, grid)"));
    assert!(page.contains("traceCtx.rotate(-Math.PI / 2);"));
    assert!(page.contains("id=\"entity-graph\""));
    assert!(page.contains("drawEntityGraph"));
    assert!(page.contains("createDefaultXRExperienceAsync"));
    assert!(page.contains("if(!eye?.data_url)"));
    assert!(page.contains(
            ".panel-window.is-shaded{height:32px!important;min-height:32px!important;max-height:32px!important;"
        ));
}

#[tokio::test]
async fn live_embodied_endpoint_returns_latest_context() {
    let state = LiveViewState::new();
    let context = EmbodiedContext {
        experience_id: Some(uuid::Uuid::new_v4()),
        summary: "I see a frame.".to_string(),
        ..EmbodiedContext::default()
    };
    state.update_embodied_context(context.clone());

    let Json(response) = get_live_embodied(State(state)).await.unwrap();

    assert_eq!(response, context);
}

#[test]
fn embodied_lineage_graph_traces_current_experience() {
    let primary = Sensation::primary(
        Modality::Vision,
        SensationSource::new("unit-camera"),
        100,
        101,
        SensationPayload::image_metadata(32, 24, "rgb8", 32 * 24 * 3),
    )
    .with_summary("I receive a visual frame.");
    let child = Sensation::descendant(
        &primary,
        "vision.crop.focus",
        SensationPayloadKind::Crop,
        serde_json::json!({"x": 4, "y": 3, "width": 12, "height": 9}),
        SensationMetadata::default(),
        "focus",
    )
    .with_summary("I focus on a patch.")
    .with_vector(VectorEmbedding::new(
        vec![0.1, 0.2, 0.3],
        "unit.crop.v0",
        Modality::Vision,
        SensationPayloadKind::Crop,
        primary.id,
        102,
    ));
    let impression = Impression::new(
        "vision.focus.impression",
        "I see and focus.",
        vec![primary.id, child.id],
        100,
        102,
    );
    let summary_impression = Impression::new(
        "experience.summary",
        "I see and focus.",
        vec![primary.id, child.id],
        100,
        102,
    )
    .with_vector(VectorEmbedding::new(
        vec![0.1, 0.2, 0.3, 0.4],
        "unit.fuser.v0",
        Modality::Other,
        SensationPayloadKind::Structured,
        child.id,
        103,
    ));
    let mut experience = Experience::new(
        "embodied.now",
        "I see and focus.",
        vec![impression.id, summary_impression.id],
        vec![primary.id, child.id],
        100,
        102,
    );
    experience.summary_impression = Some(summary_impression.clone().for_experience(experience.id));
    experience.predictions.push(Prediction {
        offset_ms: 750,
        text: "The focused view should remain stable.".to_string(),
        confidence: 0.6,
        vector: None,
    });
    experience.memory_links.push(MemoryLink {
        target_id: "memory-1".to_string(),
        relation: "similar".to_string(),
        score: 0.8,
        payload: serde_json::json!({"text": "A previous focused camera moment."}),
    });
    let context = EmbodiedContext::from_current_experience(
        Some(&experience),
        &[primary.clone(), child.clone()],
        &[impression.clone(), summary_impression.clone()],
        &[],
        &[],
    );

    let graph = EmbodiedLineageGraph::from_context(&context);

    assert_eq!(graph.schema_version, 1);
    assert_eq!(graph.experience_id, Some(experience.id.to_string()));
    assert!(graph.nodes.iter().any(|node| {
        node.id == format!("sensation:{}", child.id)
            && node.derived
            && node.modality.as_deref() == Some("vision")
    }));
    assert!(graph.edges.iter().any(|edge| {
        edge.from == format!("sensation:{}", primary.id)
            && edge.to == format!("sensation:{}", child.id)
            && edge.relation == EmbodiedGraphEdgeType::ParentSensation
    }));
    assert!(graph.edges.iter().any(|edge| {
        edge.from == format!("sensation:{}", child.id)
            && edge.to == format!("experience:{}", experience.id)
            && edge.relation == EmbodiedGraphEdgeType::SensationMember
    }));
    assert!(graph.edges.iter().any(|edge| {
        edge.from == format!("impression:{}", impression.id)
            && edge.to == format!("experience:{}", experience.id)
            && edge.relation == EmbodiedGraphEdgeType::ImpressionMember
    }));
    assert!(graph
        .nodes
        .iter()
        .any(|node| node.node_type == EmbodiedGraphNodeType::Prediction));
    assert!(graph
        .vector_metadata
        .iter()
        .any(|vector| { vector.model_id == "unit.fuser.v0" && vector.dim == 4 }));
    assert!(graph
        .vector_metadata
        .iter()
        .any(|vector| { vector.model_id == "unit.crop.v0" && vector.dim == 3 }));
    assert_eq!(graph.recent_memories.len(), 1);
    assert_eq!(graph.recent_memories[0].target_id, "memory-1");
}

#[test]
fn model_summary_reads_training_artifacts() {
    let model_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/models");
    let packet = read_models_response(&model_root);

    assert_eq!(packet.schema_version, 1);
    assert!(packet
        .connections
        .iter()
        .any(|edge| edge.from == "Ledger" && edge.to == "Training"));
    assert!(packet
        .behavior_nodes
        .iter()
        .any(|node| node.node_id == "ActionValue"));
    assert!(packet
        .registry
        .iter()
        .any(|entry| entry.behavior == "danger"));
    for (name, behavior_report, scenario_report) in [
        (
            "danger_golden_column_v0",
            "data/reports/danger-golden-column-heldout-eval.json",
            "data/reports/golden-column-trap-model-assisted-full-shadow.json",
        ),
        (
            "action_value_golden_column_v0",
            "data/reports/action-value-golden-column-heldout-eval.json",
            "data/reports/golden-column-trap-model-assisted-full-shadow.json",
        ),
        (
            "charge_golden_charger_v0",
            "data/reports/charge-golden-charger-heldout-eval.json",
            "data/reports/golden-charger-seeking-heldout.json",
        ),
    ] {
        let entry = packet
            .registry
            .iter()
            .find(|entry| entry.name == name)
            .unwrap_or_else(|| panic!("missing registry entry {name}"));
        assert_eq!(entry.status, "shadow");
        assert_eq!(entry.behavior_report_path.as_deref(), Some(behavior_report));
        assert_eq!(entry.scenario_report_path.as_deref(), Some(scenario_report));
        assert!(entry
            .allowed_modes
            .iter()
            .any(|mode| mode == "shadow-infer"));
        assert!(!entry.allowed_modes.iter().any(|mode| mode == "model-infer"));
    }
}

#[tokio::test]
async fn live_view_page_draws_common_camera_formats() {
    let Html(page) = live_view_page().await;

    assert!(page.contains("isBgr"));
    assert!(page.contains("isGray"));
    assert!(page.contains("isYuyv"));
    assert!(page.contains("writeYuvPixel"));
    assert!(page.contains("drawGeneratedEye"));
    assert!(page.contains("/view/scene"));
    assert!(page.contains("session.source === 'sim'"));
    assert!(page.contains("includes('virtual')"));
}

#[test]
fn yuyv_eye_frame_encodes_to_png_data_url() {
    let frame = EyeFrame {
        rgbd_frame_id: None,
        device_timestamp_ms: None,
        captured_at_ms: 1,
        width: 2,
        height: 1,
        format: EyeFrameFormat::Yuyv422,
        bytes: vec![82, 90, 145, 240],
        source: None,
    };

    let (eye, warnings) = scene_eye_from_frame(&frame, None, 1);

    assert!(warnings.is_empty());
    assert_eq!(eye.width, 2);
    assert_eq!(eye.height, 1);
    assert!(eye
        .data_url
        .as_deref()
        .unwrap_or_default()
        .starts_with("data:image/png;base64,"));
}

#[test]
fn grbg_bayer_eye_frame_encodes_to_png_data_url() {
    let frame = EyeFrame {
        rgbd_frame_id: None,
        device_timestamp_ms: None,
        captured_at_ms: 1,
        width: 2,
        height: 2,
        format: EyeFrameFormat::BayerGrbg8,
        bytes: vec![90, 220, 40, 110],
        source: None,
    };

    let (eye, warnings) = scene_eye_from_frame(&frame, None, 1);

    assert!(warnings.is_empty());
    assert_eq!(eye.width, 2);
    assert_eq!(eye.height, 2);
    assert!(eye
        .data_url
        .as_deref()
        .unwrap_or_default()
        .starts_with("data:image/png;base64,"));
}

#[test]
fn grbg_bayer_eye_frame_encodes_latest_png_bytes() {
    let frame = EyeFrame {
        rgbd_frame_id: None,
        device_timestamp_ms: None,
        captured_at_ms: 1,
        width: 2,
        height: 2,
        format: EyeFrameFormat::BayerGrbg8,
        bytes: vec![90, 220, 40, 110],
        source: None,
    };

    let bytes = encode_eye_png_bytes(&frame).unwrap();

    assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
}

#[tokio::test]
async fn cognitive_summary_endpoint_returns_valid_json() {
    let state = LiveViewState::new();
    let Json(report) = get_cognitive_summary(State(state)).await;
    let value = serde_json::to_value(&report).expect("cognitive report serializes");

    assert!(value.get("summary").is_some());
    assert!(value.get("bindings").is_some());
    assert_eq!(value["summary"]["feature_count"], 0);
    assert!(HTTP_ENDPOINTS.contains(&"/api/cognitive/summary"));
    assert!(HTTP_ENDPOINTS.contains(&"/api/cognitive/bindings"));
    let Html(page) = cognitive_view_page().await;
    assert!(page.contains("Cognitive Inspector"));
    assert!(page.contains("/api/cognitive/summary"));
}

fn unique_test_dir(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "pete-server-{name}-{}-{}",
        std::process::id(),
        wall_now_ms()
    ));
    std::fs::remove_dir_all(&path).ok();
    path
}
