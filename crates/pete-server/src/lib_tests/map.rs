#[tokio::test]
async fn live_scene_returns_503_before_first_snapshot() {
    let err = get_live_scene(State(LiveViewState::new()))
        .await
        .unwrap_err();

    assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn inline_learning_control_updates_live_state() {
    let state = LiveViewState::new();
    let config = InlineLearningConfig {
        mode: InlineLearningMode::WorldOutcome,
        behaviors: pete_runtime::InlineLearningBehaviors {
            danger: false,
            charge: false,
            future: true,
            action_value: true,
            eye_next: false,
            ear_next: false,
            experience: false,
        },
        max_train_steps_per_tick: 2,
    };

    let Json(response) = post_inline_learning(State(state.clone()), Json(config.clone()))
        .await
        .unwrap();
    let Json(readback) = get_inline_learning(State(state)).await;

    assert!(response.enabled);
    assert_eq!(response.training_mode, "inline-world-outcome");
    assert_eq!(readback.config, config);
    assert!(readback.weights_updating);
}

#[tokio::test]
async fn behavior_node_endpoints_round_trip_config() {
    let state = LiveViewState::new();

    let Json(initial) = get_behavior_nodes(State(state.clone())).await;
    assert!(initial.nodes.iter().any(|node| node.node_id == "Conductor"));

    let Json(updated) = post_behavior_node(
        State(state.clone()),
        AxumPath("Conductor".to_string()),
        Json(BehaviorNodeUpdate {
            selected_regime: Some(BehaviorRegime::ShadowTrain),
            selected_hardcoded: Some("reign.teacher".to_string()),
            selected_model: Some("conductor.burn.v0".to_string()),
            checkpoint_path: Some("data/models/conductor_v0".to_string()),
            fallback_policy: Some(pete_behaviors::FallbackPolicy::UseHardcoded),
            training_enabled: Some(true),
        }),
    )
    .await
    .unwrap();
    let Json(readback) = get_behavior_nodes(State(state.clone())).await;
    let conductor = readback
        .nodes
        .iter()
        .find(|node| node.node_id == "Conductor")
        .unwrap();

    assert_eq!(updated.selected_regime, BehaviorRegime::ShadowTrain);
    assert_eq!(conductor.selected_hardcoded, "reign.teacher");
    assert_eq!(
        conductor.checkpoint_path.as_deref(),
        Some("data/models/conductor_v0")
    );
    assert!(conductor.training_enabled);

    let Json(updated_event) = post_behavior_node(
        State(state.clone()),
        AxumPath("EventBump".to_string()),
        Json(BehaviorNodeUpdate {
            selected_regime: Some(BehaviorRegime::ShadowTrain),
            selected_hardcoded: Some("script.on_bump.v0".to_string()),
            selected_model: Some("event.bump.shadow.v0".to_string()),
            checkpoint_path: Some("data/models/event_bump_v0".to_string()),
            fallback_policy: Some(pete_behaviors::FallbackPolicy::UseHardcoded),
            training_enabled: Some(true),
        }),
    )
    .await
    .unwrap();

    assert_eq!(updated_event.selected_regime, BehaviorRegime::ShadowTrain);
    assert_eq!(updated_event.selected_hardcoded, "script.on_bump.v0");
    assert_eq!(
        updated_event.selected_model.as_deref(),
        Some("event.bump.shadow.v0")
    );
}

#[tokio::test]
async fn live_scene_returns_body_pose_and_range_beams() {
    let state = LiveViewState::new();
    state.update_scene_metadata(LiveSceneMetadata {
        arena: Some(SceneArena {
            width_m: 4.0,
            height_m: 3.0,
        }),
        objects: vec![SceneObject {
            id: "charger-0".to_string(),
            kind: "charger".to_string(),
            x_m: 1.2,
            y_m: 0.4,
            radius_m: 0.25,
            label: Some("charger".to_string()),
            color_rgb: Some([80, 220, 130]),
        }],
        sensor_calibration: Some(SceneSensorCalibration::sim_default()),
    });
    state.update_session(SceneSession {
        mode: "virtual-live".to_string(),
        scenario: Some("charger-seeking".to_string()),
        seed: Some(99),
        source: "sim".to_string(),
        tick_ms: Some(100),
    });
    state.update_training_status(LiveTrainingStatus {
        training_mode: "collecting".to_string(),
        ledger_path: Some("data/ledger/virtual-live".to_string()),
        frames_written: 12,
        transitions_written: 11,
        models_loaded: vec!["danger".to_string()],
        model_modes: HashMap::from([("danger".to_string(), "shadow-infer".to_string())]),
        action_selector_mode: "baseline".to_string(),
        weights_updating: false,
    });
    state.update_prod_state(NudgeStatus {
        idle_ms: 4_200,
        last_nudge_ms: Some(1_000),
        nudge_count_recent: 1,
        nudge_blocked_reason: Some("prod cooldown active".to_string()),
        active_nudge: false,
    });
    let mut snapshot = WorldSnapshot::default();
    snapshot.body.odometry.x_m = 0.5;
    snapshot.body.odometry.y_m = 0.75;
    snapshot.body.odometry.heading_rad = 1.25;
    snapshot.body.battery_level = 0.82;
    snapshot.body.last_update_ms = 1234;
    snapshot.final_selected_action = Some(ActionPrimitive::Explore {
        style: pete_actions::ExploreStyle::RandomWalk,
        duration_ms: 1_000,
    });
    snapshot.llm_action_proposal = Some(pete_actions::LlmActionProposal {
        proposed_action: None,
        advisory_action: Some(pete_actions::LlmAdvisoryAction {
            action: ActionPrimitive::Go {
                intensity: 0.4,
                duration_ms: 800,
            },
            source: pete_actions::LlmAdvisoryActionSource::ProviderDecision,
            input_snapshot_ref: "provider-input-1200".to_string(),
            disposition: pete_actions::LlmAdvisoryActionDisposition::DiscardedAtAdvisoryBoundary,
        }),
        accepted: false,
        safety_vetoed: false,
        final_action: snapshot.final_selected_action.clone(),
        ignored_reason: Some("provider suggested Go; discarded at advisory boundary".to_string()),
        safety_reason: None,
    });
    snapshot.range.beams = vec![1.0, 2.0, 3.0];
    snapshot.range.nearest_m = Some(1.0);
    snapshot.extensions.push(pete_now::ExtensionSense {
        schema_version: 1,
        name: "sim.stuck".to_string(),
        values: vec![
            1.0, 1.0, 6.0, 300.0, 3.0, -1.0, 1.0, 0.0, 1.0, 0.0, 3.0, 2.0, 1.0, 0.2,
        ],
    });
    snapshot.eye_frame = Some(EyeFrame {
        captured_at_ms: 1200,
        width: 1,
        height: 1,
        format: EyeFrameFormat::Rgb8,
        bytes: vec![255, 0, 0],
        source: None,
    });
    let expected_llm_action = snapshot
        .llm_action_proposal
        .as_ref()
        .and_then(|proposal| proposal.proposed_action.clone());
    let expected_llm_advisory_action = snapshot
        .llm_action_proposal
        .as_ref()
        .and_then(|proposal| proposal.advisory_action.clone());
    let expected_final_action = snapshot.final_selected_action.clone();
    state.update(snapshot);

    let Json(scene) = get_live_scene(State(state)).await.unwrap();

    assert_eq!(scene.schema_version, 1);
    assert_eq!(scene.session.as_ref().unwrap().mode, "virtual-live");
    assert_eq!(
        scene.session.as_ref().unwrap().scenario.as_deref(),
        Some("charger-seeking")
    );
    assert_eq!(scene.t_ms, 1234);
    assert_eq!(scene.body.x_m, 0.5);
    assert_eq!(scene.body.y_m, 0.75);
    assert_eq!(scene.body.heading_rad, 1.25);
    assert_eq!(scene.action.latest_llm_proposed_action, expected_llm_action);
    assert_eq!(
        scene.action.latest_llm_advisory_action,
        expected_llm_advisory_action
    );
    assert_eq!(scene.action.llm_action_accepted, Some(false));
    assert_eq!(scene.action.llm_action_safety_vetoed, Some(false));
    assert_eq!(
        scene.action.llm_action_ignored_reason.as_deref(),
        Some("provider suggested Go; discarded at advisory boundary")
    );
    assert_eq!(scene.action.final_selected_action, expected_final_action);
    assert_eq!(scene.range.nearest_m, Some(1.0));
    assert_eq!(scene.range.beams.len(), 3);
    assert_eq!(scene.training_mode, "collecting");
    assert_eq!(
        scene.ledger_path.as_deref(),
        Some("data/ledger/virtual-live")
    );
    assert_eq!(scene.frames_written, 12);
    assert_eq!(scene.transitions_written, 11);
    assert_eq!(scene.models_loaded, vec!["danger"]);
    assert!(!scene.weights_updating);
    assert!(scene.stuck);
    assert_eq!(scene.idle_ms, 4_200);
    assert_eq!(scene.last_nudge_ms, Some(1_000));
    assert_eq!(scene.nudge_count_recent, 1);
    assert_eq!(
        scene.nudge_blocked_reason.as_deref(),
        Some("prod cooldown active")
    );
    assert_eq!(scene.prod.idle_ms, 4_200);
    assert!(scene.dead_battery);
    assert_eq!(scene.recovery_mode.as_deref(), Some("turn-away"));
    assert_eq!(scene.stuck_ticks, 6);
    assert_eq!(scene.stuck_detail.class.as_deref(), Some("column-trap"));
    assert_eq!(scene.stuck_detail.trap_kind.as_deref(), Some("column"));
    assert_eq!(scene.stuck_detail.recovery_attempts, 2);
    assert_eq!(scene.stuck_detail.repeated_trap_count, 1);
    assert_eq!(
        scene.stuck_detail.recovery_phase.as_deref(),
        Some("turn-away")
    );
    assert_eq!(scene.objects[0].kind, "charger");
    assert!(scene
        .eye
        .unwrap()
        .data_url
        .unwrap()
        .starts_with("data:image/png;base64,"));
}

#[tokio::test]
async fn live_map_endpoint_returns_pose_projected_beams_and_overlays() {
    let state = LiveViewState::new();
    state.update_scene_metadata(LiveSceneMetadata {
        arena: None,
        objects: vec![
            SceneObject {
                id: "charger-0".to_string(),
                kind: "charger".to_string(),
                x_m: 2.0,
                y_m: 0.5,
                radius_m: 0.2,
                label: Some("charger".to_string()),
                color_rgb: None,
            },
            SceneObject {
                id: "person-0".to_string(),
                kind: "person".to_string(),
                x_m: 0.2,
                y_m: 1.8,
                radius_m: 0.25,
                label: Some("person".to_string()),
                color_rgb: None,
            },
            SceneObject {
                id: "speaker-0".to_string(),
                kind: "speaker".to_string(),
                x_m: -0.4,
                y_m: 1.2,
                radius_m: 0.15,
                label: Some("speaker".to_string()),
                color_rgb: None,
            },
        ],
        sensor_calibration: None,
    });

    let mut snapshot = WorldSnapshot::default();
    snapshot.body.odometry.x_m = 0.5;
    snapshot.body.odometry.y_m = 0.75;
    snapshot.body.odometry.heading_rad = std::f32::consts::FRAC_PI_2;
    snapshot.body.last_update_ms = 1234;
    snapshot.body.flags.bump_left = true;
    snapshot.body.flags.cliff_front_left = true;
    snapshot.range.beams = vec![2.0, 1.0, 2.0];
    snapshot.range.nearest_m = Some(1.0);
    snapshot
        .objects
        .observations
        .push(pete_now::ObjectObservation {
            label: "person-nearby".to_string(),
            class: pete_now::ObjectClass::Person,
            bearing_rad: 0.1,
            distance_m: Some(1.2),
            confidence: 0.82,
            source: pete_now::ObjectObservationSource::Sim,
        });
    snapshot.ear.transcript = Some("Travis".to_string());
    snapshot.llm_action_proposal = Some(pete_actions::LlmActionProposal {
        safety_vetoed: true,
        ..pete_actions::LlmActionProposal::default()
    });
    snapshot.extensions.push(pete_now::ExtensionSense {
        schema_version: 1,
        name: "sim.stuck".to_string(),
        values: vec![1.0, 0.0, 4.0, 200.0, 1.0],
    });
    state.update(snapshot.clone());
    snapshot.body.last_update_ms = 1334;
    state.update(snapshot);

    let Json(map) = get_live_map(State(state)).await.unwrap();

    assert_eq!(map.schema_version, 1);
    assert_eq!(map.label, MAP_LABEL);
    assert_eq!(map.world_projection.coordinate_frame, "odometry_world");
    assert!(!map.world_projection.aligned_with_3d);
    assert!(map.summary.label.contains("scan-matched"));
    assert!(map.summary.label.contains("occupancy"));
    assert_eq!(map.pose_trail.len(), 2);
    assert_eq!(map.current_pose.as_ref().unwrap().x_m, 0.5);
    assert_eq!(map.pose_graph.nodes, map.summary.pose_graph_nodes);
    assert_eq!(map.pose_graph.edges, map.summary.pose_graph_edges);
    assert_eq!(map.pose_graph.nodes, 1);
    assert_eq!(map.remap.submaps, map.summary.remap.submaps);
    assert_eq!(map.remap.cells, map.summary.remap.cells);
    assert!(map.remap.submaps >= 1);
    assert_eq!(
        map.overlays,
        vec![
            "occupancy",
            "rays",
            "raw point cloud",
            "accumulated occupancy",
            "stable wall candidates",
            "danger",
            "charger/charge",
            "social",
            "novelty",
            "events"
        ]
    );
    assert_eq!(map.range_beams.len(), 3);
    assert!(!map.cells.is_empty());
    assert!(map
        .cells
        .iter()
        .any(|cell| cell.occupied_score > cell.free_score));
    assert!(map
        .cells
        .iter()
        .any(|cell| cell.free_score >= cell.occupied_score));
    assert!(map
        .semantic_cells
        .iter()
        .any(|cell| cell.kind == "charger/charge" && cell.label.as_deref() == Some("charger")));
    assert!(map
        .semantic_cells
        .iter()
        .any(|cell| cell.kind == "social" && cell.label.as_deref() == Some("person")));
    assert!(map
        .semantic_cells
        .iter()
        .any(|cell| cell.kind == "social" && cell.label.as_deref() == Some("speaker")));
    assert!(map.events.iter().any(|event| event.kind == "bump"));
    assert!(map.events.iter().any(|event| event.kind == "cliff"));
    assert!(map.events.iter().any(|event| event.kind == "stuck"));
    assert!(map
        .events
        .iter()
        .any(|event| event.kind == "safety_override"));
    assert!(map.events.iter().any(|event| event.kind == "charger"));
    assert!(map.events.iter().any(|event| event.kind == "person"));
    assert!(map.events.iter().any(|event| event.kind == "speaker"));
    assert_eq!(map.entity_graph.schema_version, 1);
    assert!(map
        .entity_graph
        .nodes
        .iter()
        .any(|node| node.node_type == "entity"));
    assert!(map
        .entity_graph
        .nodes
        .iter()
        .any(|node| node.node_type == "text_label"));
    assert!(map
        .entity_graph
        .nodes
        .iter()
        .any(|node| node.node_type == "observation" && node.source_channel.is_some()));
    assert!(map
        .entity_graph
        .edges
        .iter()
        .any(|edge| edge.edge_type == "named_by"));
    assert!(map
        .entity_graph
        .edges
        .iter()
        .any(|edge| edge.observed_at_ms.is_some()));
    assert!(!map.entity_graph.events.is_empty());
    assert!(map
        .entity_graph
        .events
        .iter()
        .any(|event| event.event_type == "create"));
    assert!(map
        .entity_graph
        .events
        .iter()
        .any(|event| event.event_type == "strengthen"));
    assert!(map
        .entity_graph
        .events
        .iter()
        .all(|event| event.timestamp_ms.is_some()));

    let forward_hit = map
        .range_beams
        .iter()
        .find(|beam| beam.hit && beam.angle_rad.abs() < 0.001)
        .expect("center beam should be marked as a hit");
    assert!((forward_hit.origin_x_m - 0.5).abs() < 0.001);
    assert!((forward_hit.origin_y_m - 0.75).abs() < 0.001);
    assert!((forward_hit.end_x_m - 0.5).abs() < 0.001);
    assert!((forward_hit.end_y_m - 1.75).abs() < 0.001);
    assert!((forward_hit.angle_rad - 0.0).abs() < 0.001);
}

#[tokio::test]
async fn live_map_projects_depth_only_world_voxels_in_the_same_frame_as_3d() {
    let state = LiveViewState::new();
    state.update_scene_metadata(LiveSceneMetadata {
        sensor_calibration: Some(SceneSensorCalibration::sim_default()),
        ..LiveSceneMetadata::default()
    });
    for t_ms in [100, 200, 300, 400] {
        let mut snapshot = WorldSnapshot::default();
        snapshot.body.odometry = Pose2 {
            x_m: 0.5,
            y_m: 0.75,
            heading_rad: std::f32::consts::FRAC_PI_2,
        };
        snapshot.body.last_update_ms = t_ms;
        snapshot.kinect = KinectSense {
            captured_at_ms: t_ms,
            depth_m: vec![1.0],
            depth_width: 1,
            depth_height: 1,
            depth_fx: 1.0,
            depth_fy: 1.0,
            min_depth_m: 0.1,
            max_depth_m: 8.0,
            depth_coordinate_system: Some("kinect_camera".to_string()),
            ..KinectSense::default()
        };
        snapshot.imu.orientation = vec![0.0, 0.0];
        state.update(snapshot);
    }

    let cloud = state.point_cloud_snapshot();
    let world_point = cloud
        .points()
        .into_iter()
        .find(|point| point.stable)
        .expect("repeated calibrated depth point should be stable");
    let Json(map) = get_live_map(State(state)).await.unwrap();

    assert!(map.cells.is_empty(), "range-only map should have no cells");
    assert!(map.world_projection.aligned_with_3d);
    assert!(map.world_projection.geometry_trusted);
    assert!(!map.world_projection.navigation_trusted);
    assert_eq!(map.world_projection.source_voxels, 1);
    assert_eq!(map.world_projection.projected_cells, 1);
    assert_eq!(map.world_projection.stable_cells, 1);
    let projected = &map.world_projection.cells[0];
    let half_cell = map.world_projection.resolution_m * 0.5;
    assert!((projected.center_x_m - world_point.position.x_m).abs() <= half_cell);
    assert!((projected.center_y_m - world_point.position.y_m).abs() <= half_cell);
    assert!(projected.stable);
}
