fn eval_args(
    scenario: ScenarioArg,
    episodes: usize,
    steps: usize,
    out: Option<String>,
) -> EvalScenarioArgs {
    EvalScenarioArgs {
        scenario,
        episodes,
        steps,
        seed: 7,
        tick_ms: 100,
        out,
        ledger: None,
        capture_root: None,
        memory_report: false,
        danger_checkpoint: None,
        danger_mode: DangerMode::Off,
        charge_checkpoint: None,
        charge_mode: ChargeMode::Off,
        action_value_checkpoint: None,
        action_value_mode: ActionValueMode::Off,
        future_checkpoint: None,
        future_mode: FutureMode::Hardcoded,
        eye_next_checkpoint: None,
        eye_next_mode: EyeNextMode::Off,
        ear_next_checkpoint: None,
        ear_next_mode: EarNextMode::Off,
        experience_checkpoint: None,
        experience_mode: ExperienceMode::Off,
        action_selector: CliActionSelectorMode::Baseline,
        llm: LlmArgs::default(),
    }
}

fn replay_counterfactual_args(
    capture: String,
    out_ledger: Option<String>,
    out_report: Option<String>,
) -> ReplayCounterfactualArgs {
    ReplayCounterfactualArgs {
        capture,
        edit: Vec::new(),
        policy: "baseline".to_string(),
        actions: None,
        steps: Some(4),
        out_ledger,
        out_report,
        llm: LlmArgs::default(),
    }
}

#[test]
fn counterfactual_edit_parser_parses_supported_edits() {
    assert_eq!(
        parse_counterfactual_edit("move-charger:x=1.0,y=2.0").unwrap(),
        CounterfactualEdit::MoveObject {
            kind: CounterfactualObjectKind::Charger,
            id: None,
            x_m: 1.0,
            y_m: 2.0,
        }
    );
    assert_eq!(
        parse_counterfactual_edit("set-battery:value=0.42").unwrap(),
        CounterfactualEdit::SetBattery { value: 0.42 }
    );
    assert!(parse_counterfactual_edit("move-moon:x=1,y=2")
        .unwrap_err()
        .to_string()
        .contains("unknown counterfactual edit"));
}

#[test]
fn counterfactual_edits_move_charger_and_set_battery() {
    let scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ChargerSeeking, 9));
    let mut metadata = scenario.metadata;
    let edits = vec![
        parse_counterfactual_edit("move-charger:x=1.0,y=1.0").unwrap(),
        parse_counterfactual_edit("set-battery:value=0.75").unwrap(),
    ];
    let mut warnings = Vec::new();

    apply_counterfactual_edits(&mut metadata, &edits, &mut warnings).unwrap();

    let charger = metadata
        .objects
        .iter()
        .find(|object| matches!(object.kind, pete_sim::SimObjectKind::Charger))
        .unwrap();
    assert_eq!((charger.x_m, charger.y_m), (1.0, 1.0));
    assert_eq!(metadata.body.battery_level, 0.75);
    assert!(warnings
        .iter()
        .any(|warning| warning.contains("first matching object")));
}

#[test]
fn counterfactual_report_serializes_schema() {
    let report = CounterfactualReport {
        schema_version: 1,
        source_capture: "capture".to_string(),
        reconstructable: true,
        edits: vec!["set-battery:value=0.5".to_string()],
        policy: "stop".to_string(),
        steps: 3,
        summary: CounterfactualSummary {
            collisions: 0,
            charging_ticks: 1,
            battery_delta: 0.1,
            distance_traveled: 0.2,
            final_distance_to_charger_m: Some(0.3),
        },
        warnings: Vec::new(),
    };

    let value: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&report).unwrap()).unwrap();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(value["summary"]["charging_ticks"], 1);
}

#[tokio::test]
async fn replay_counterfactual_baseline_writes_ledger_and_report() {
    let temp_dir = temp_path("pete_counterfactual_baseline");
    let capture_dir = temp_dir.join("capture");
    let ledger_dir = temp_dir.join("ledger");
    let report_path = temp_dir.join("report.json");
    let mut scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ChargerSeeking, 77));
    let snapshot = scenario.world.snapshot().await.unwrap();
    let mut writer = CaptureWriter::create(&capture_dir, CaptureSource::Sim, Some(100))
        .await
        .unwrap();
    writer.manifest_mut().scenario = Some(scenario.metadata);
    writer
        .append_snapshot(snapshot.body.last_update_ms, snapshot, Vec::new())
        .await
        .unwrap();
    writer.finish().await.unwrap();

    let args = replay_counterfactual_args(
        capture_dir.to_string_lossy().to_string(),
        Some(ledger_dir.to_string_lossy().to_string()),
        Some(report_path.to_string_lossy().to_string()),
    );
    replay_counterfactual(args).await.unwrap();

    let transitions = JsonlLedger::new(&ledger_dir).transitions().await.unwrap();
    assert!(!transitions.is_empty());
    let report: CounterfactualReport =
        serde_json::from_slice(&fs::read(&report_path).unwrap()).unwrap();
    assert!(report.reconstructable);
    assert_eq!(report.steps, 4);
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn replay_counterfactual_with_moved_charger_writes_report() {
    let temp_dir = temp_path("pete_counterfactual_moved_charger");
    let capture_dir = temp_dir.join("capture");
    let report_path = temp_dir.join("report.json");
    let mut scenario = build_scenario(ScenarioConfig::new(ScenarioKind::ChargerSeeking, 78));
    let snapshot = scenario.world.snapshot().await.unwrap();
    let mut writer = CaptureWriter::create(&capture_dir, CaptureSource::Sim, Some(100))
        .await
        .unwrap();
    writer.manifest_mut().scenario = Some(scenario.metadata);
    writer
        .append_snapshot(snapshot.body.last_update_ms, snapshot, Vec::new())
        .await
        .unwrap();
    writer.finish().await.unwrap();

    let mut args = replay_counterfactual_args(
        capture_dir.to_string_lossy().to_string(),
        None,
        Some(report_path.to_string_lossy().to_string()),
    );
    args.edit = vec!["move-charger:x=1.0,y=1.0".to_string()];
    args.policy = "seek-charge".to_string();
    replay_counterfactual(args).await.unwrap();

    let report: CounterfactualReport =
        serde_json::from_slice(&fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(report.edits, vec!["move-charger:x=1.0,y=1.0"]);
    assert_eq!(report.policy, "seek-charge");
    assert!(report
        .warnings
        .iter()
        .any(|warning| warning.contains("first matching object")));
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn replay_counterfactual_passive_capture_fails_clearly() {
    let temp_dir = temp_path("pete_counterfactual_passive");
    let capture_dir = temp_dir.join("capture");
    let mut writer = CaptureWriter::create(&capture_dir, CaptureSource::Replay, Some(100))
        .await
        .unwrap();
    writer
        .append_snapshot(0, WorldSnapshot::default(), Vec::new())
        .await
        .unwrap();
    writer.finish().await.unwrap();

    let err = replay_counterfactual(replay_counterfactual_args(
        capture_dir.to_string_lossy().to_string(),
        None,
        None,
    ))
    .await
    .unwrap_err()
    .to_string();
    assert!(err.contains(
            "passive captures without reconstructable sim metadata cannot yet be counterfactually replayed"
        ));
    let _ = fs::remove_dir_all(&temp_dir);
}

