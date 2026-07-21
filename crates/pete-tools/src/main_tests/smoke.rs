#[tokio::test]
async fn eval_scenario_empty_room_smoke_runs() {
    let args = eval_args(ScenarioArg::EmptyRoom, 1, 3, None);
    run_eval_scenario(args).await.unwrap();
}

#[tokio::test]
async fn eval_scenario_obstacle_writes_report() {
    let temp_dir = temp_path("pete_eval_scenario_obstacle");
    let out = temp_dir.join("obstacle.json");
    let mut args = eval_args(
        ScenarioArg::ObstacleAvoidance,
        3,
        5,
        Some(out.to_string_lossy().to_string()),
    );
    args.memory_report = true;
    run_eval_scenario(args).await.unwrap();
    let report: serde_json::Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(report["scenario"], "obstacle-avoidance");
    assert_eq!(report["action_selector_mode"], "baseline");
    assert_eq!(report["episodes_detail"].as_array().unwrap().len(), 3);
    assert!(report["memory"]["places_visited"].as_u64().unwrap_or(0) > 0);
    assert!(
        report["episodes_detail"][0]["memory"]["danger_memory_ticks"]
            .as_u64()
            .unwrap_or(0)
            > 0
    );
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn eval_scenario_model_assisted_empty_room_runs_and_reports_stats() {
    let temp_dir = temp_path("pete_eval_scenario_model_assisted_empty");
    let out = temp_dir.join("empty-model-assisted.json");
    let mut args = eval_args(
        ScenarioArg::EmptyRoom,
        1,
        3,
        Some(out.to_string_lossy().to_string()),
    );
    args.action_selector = CliActionSelectorMode::ModelAssisted;
    run_eval_scenario(args).await.unwrap();
    let report: serde_json::Value = serde_json::from_slice(&fs::read(&out).unwrap()).unwrap();
    assert_eq!(report["action_selector_mode"], "model-assisted");
    assert_eq!(report["summary"]["model_assisted_decisions"], 3);
    assert!(report["summary"]["mean_candidate_score"].is_number());
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn eval_scenario_model_assisted_charger_seeking_runs() {
    let mut args = eval_args(ScenarioArg::ChargerSeeking, 1, 3, None);
    args.action_selector = CliActionSelectorMode::ModelAssisted;
    run_eval_scenario(args).await.unwrap();
}

#[tokio::test]
async fn eval_scenario_optional_ledger_writes_transitions() {
    let temp_dir = temp_path("pete_eval_scenario_ledger");
    let ledger_dir = temp_dir.join("ledger");
    let out = temp_dir.join("empty.json");
    let mut args = eval_args(
        ScenarioArg::EmptyRoom,
        1,
        4,
        Some(out.to_string_lossy().to_string()),
    );
    args.ledger = Some(ledger_dir.to_string_lossy().to_string());
    run_eval_scenario(args).await.unwrap();
    let transitions = JsonlLedger::new(&ledger_dir).transitions().await.unwrap();
    assert!(!transitions.is_empty());
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn sim_curriculum_writes_one_capture_per_episode() {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let temp_dir = std::env::temp_dir().join(format!("pete_curriculum_test_{now_ms}"));
    let ledger_dir = temp_dir.join("ledger");
    let capture_root = temp_dir.join("captures");

    let args = SimCurriculumArgs {
        scenario: ScenarioArg::PersonSpeakerRoom,
        episodes: 2,
        steps: 3,
        seed: 7,
        out: ledger_dir.to_str().unwrap().to_string(),
        capture_root: Some(capture_root.to_str().unwrap().to_string()),
        tick_ms: 100,
        validation_ratio: 0.25,
        test_ratio: 0.25,
        llm: LlmArgs::default(),
    };

    run_sim_curriculum(args).await.unwrap();

    let manifest_path = ledger_dir.join("manifest.json");
    assert!(manifest_path.exists());
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["scenario"], "person-speaker-room");
    assert_eq!(manifest["episodes"], 2);
    assert_eq!(manifest["splits"]["train"], 0);
    assert_eq!(manifest["splits"]["validation"], 1);
    assert_eq!(manifest["splits"]["test"], 1);
    assert_eq!(
        manifest["episodes_detail"][0]["capture"],
        capture_root
            .join("episode-000")
            .to_string_lossy()
            .to_string()
    );
    assert!(capture_root
        .join("episode-000")
        .join("manifest.json")
        .exists());
    assert!(capture_root
        .join("episode-001")
        .join("manifest.json")
        .exists());
    let transitions = JsonlLedger::new(&ledger_dir).transitions().await.unwrap();
    assert!(!transitions.is_empty());

    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn test_evaluate_behavior_command_writes_json_to_out() {
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let temp_dir = std::env::temp_dir().join(format!("pete_eval_test_{}", now_ms));
    let ledger_dir = temp_dir.join("ledger");
    let session_dir = ledger_dir.join("2026-06-24");
    fs::create_dir_all(&session_dir).unwrap();

    let checkpoint_dir = temp_dir.join("checkpoint");
    fs::create_dir_all(&checkpoint_dir).unwrap();

    // Write 5 mock transitions to have enough data for training and validation splits
    let mut transitions = Vec::new();
    for i in 0..5 {
        let transition = ExperienceTransition {
            id: uuid::Uuid::new_v4(),
            before_frame_id: uuid::Uuid::new_v4(),
            before: Now::blank(100 + i * 100, BodySense::default()),
            before_z: ExperienceLatent {
                t_ms: 100 + i * 100,
                z: vec![0.1; 4],
                ..ExperienceLatent::default()
            },
            action: Some(ActionPrimitive::Stop),
            predicted_futures: Vec::new(),
            after: Now::blank(200 + i * 100, BodySense::default()),
            after_z: ExperienceLatent {
                t_ms: 200 + i * 100,
                z: vec![0.2; 4],
                ..ExperienceLatent::default()
            },
            reward: Reward { value: 0.0 },
            surprise: SurpriseSense::default(),
            created_at_ms: 200 + i * 100,
        };
        transitions.push(transition);
    }

    let transitions_file = session_dir.join("transitions.jsonl");
    let mut content = String::new();
    for t in &transitions {
        content.push_str(&serde_json::to_string(t).unwrap());
        content.push('\n');
    }
    fs::write(&transitions_file, content).unwrap();

    // Train first to create the checkpoint and metadata
    pete_training::train_behavior(pete_training::TrainBehaviorRequest {
        behavior: pete_training::TrainableBehavior::Danger,
        ledger_path: ledger_dir.clone(),
        checkpoint_path: checkpoint_dir.clone(),
        epochs: 1,
        validation_split: 0.2,
        seed: 42,
    })
    .await
    .unwrap();

    // Prepare output path
    let out_json_path = temp_dir.join("report.json");

    let args = EvaluateBehaviorArgs {
        behavior: "danger".to_string(),
        ledger: ledger_dir.to_str().unwrap().to_string(),
        checkpoint: Some(checkpoint_dir.to_str().unwrap().to_string()),
        max_samples: None,
        out: Some(out_json_path.to_str().unwrap().to_string()),
    };

    let cmd = EvaluateCommand {
        model: EvaluateModel::Behavior(args),
    };

    let res = run_evaluate(cmd).await;
    assert!(res.is_ok(), "run_evaluate failed: {:?}", res.err());

    // Verify report file exists and has correct behavior name
    assert!(out_json_path.exists());
    let report_content = fs::read_to_string(&out_json_path).unwrap();
    let report: serde_json::Value = serde_json::from_str(&report_content).unwrap();
    assert_eq!(report["behavior"], "danger");

    // Clean up
    let _ = fs::remove_dir_all(&temp_dir);
}
