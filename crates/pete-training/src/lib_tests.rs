
use super::*;
use pete_actions::ActionPrimitive;
use pete_body::BodySense;
use pete_core::Reward;
use pete_experience::ExperienceLatent;
use pete_now::{SurpriseSense, VectorArtifact};
use std::fs;

#[tokio::test]
async fn test_train_behavior_writes_evaluation_json() {
    let temp_dir = std::env::temp_dir().join(format!("pete_train_test_{}", now_ms()));
    let ledger_dir = temp_dir.join("ledger");
    let session_dir = ledger_dir.join("2026-06-24");
    fs::create_dir_all(&session_dir).unwrap();

    let checkpoint_dir = temp_dir.join("checkpoint");
    fs::create_dir_all(&checkpoint_dir).unwrap();

    // Construct 5 mock transitions to have enough data for training and validation splits
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

    let request = TrainBehaviorRequest {
        behavior: TrainableBehavior::Danger,
        ledger_path: ledger_dir,
        checkpoint_path: checkpoint_dir.clone(),
        epochs: 1,
        validation_split: 0.2,
        seed: 42,
    };

    let summary = train_behavior(request).await.unwrap();
    assert_eq!(summary.behavior, TrainableBehavior::Danger);

    // Verify that evaluation.json was created
    let eval_json_path = checkpoint_dir.join("evaluation.json");
    assert!(eval_json_path.exists());

    let eval_content = fs::read_to_string(&eval_json_path).unwrap();
    let report: BehaviorEvaluationReport = serde_json::from_str(&eval_content).unwrap();
    assert_eq!(report.behavior, TrainableBehavior::Danger);

    // Clean up
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn test_train_latent_round_trip_writes_predictive_report() {
    let temp_dir = std::env::temp_dir().join(format!("pete_latent_round_trip_test_{}", now_ms()));
    let ledger_dir = temp_dir.join("ledger");
    let session_dir = ledger_dir.join("2026-06-27");
    fs::create_dir_all(&session_dir).unwrap();

    let mut transitions = Vec::new();
    for i in 0..6 {
        let mut before_body = BodySense::default();
        before_body.battery_level = 0.8;
        before_body.odometry.x_m = i as f32 * 0.01;
        let mut after_body = before_body.clone();
        after_body.odometry.x_m += 0.01;
        let before = Now::blank(100 + i * 100, before_body);
        let after = Now::blank(200 + i * 100, after_body);
        transitions.push(ExperienceTransition {
            id: uuid::Uuid::new_v4(),
            before_frame_id: uuid::Uuid::new_v4(),
            before,
            before_z: ExperienceLatent {
                t_ms: 100 + i * 100,
                z: vec![0.1 + i as f32 * 0.01, 0.2],
                ..ExperienceLatent::default()
            },
            action: Some(ActionPrimitive::Stop),
            predicted_futures: Vec::new(),
            after,
            after_z: ExperienceLatent {
                t_ms: 200 + i * 100,
                z: vec![0.11 + i as f32 * 0.01, 0.2],
                ..ExperienceLatent::default()
            },
            reward: Reward { value: 0.0 },
            surprise: SurpriseSense::default(),
            created_at_ms: 200 + i * 100,
        });
    }

    let transitions_file = session_dir.join("transitions.jsonl");
    let content = transitions
        .iter()
        .map(|transition| serde_json::to_string(transition).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&transitions_file, format!("{content}\n")).unwrap();

    let report_path = temp_dir.join("latent-report.json");
    let report = train_latent_round_trip(TrainLatentRoundTripRequest {
        ledger_path: ledger_dir,
        checkpoint_path: temp_dir.join("checkpoint"),
        report_path: report_path.clone(),
        epochs: 0,
        validation_split: 0.34,
        seed: 7,
        z_dim: 2,
        codebook_size: Some(2),
    })
    .await
    .unwrap();

    assert!(report_path.exists());
    assert_eq!(report.schema_version, 2);
    assert_eq!(report.architecture.owned_latent.name, "ExperienceLatent");
    assert_eq!(report.architecture.owned_latent.owner, "Pete");
    assert!(report.architecture.owned_latent.teacher_independent);
    assert!(report
        .architecture
        .pipeline
        .contains(&"mechanically_assembled_instant".to_string()));
    assert!(report.predictors.len() >= 2);
    assert!(report
        .predictors
        .iter()
        .any(|predictor| predictor.encoder == "trainable-autoencoder"));
    assert!(report.reconstruction.sample_count > 0);
    assert_eq!(report.codebook.unwrap().code_count, 2);

    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn test_train_unified_experience_reports_multimodal_teacher_instant() {
    let temp_dir = std::env::temp_dir().join(format!("pete_unified_experience_test_{}", now_ms()));
    let ledger_dir = temp_dir.join("ledger");
    let session_dir = ledger_dir.join("2026-06-28");
    fs::create_dir_all(&session_dir).unwrap();

    let mut transitions = Vec::new();
    for i in 0..8 {
        let before = multimodal_now(100 + i * 100, i as f32);
        let after = multimodal_now(200 + i * 100, i as f32 + 0.5);
        transitions.push(ExperienceTransition {
            id: uuid::Uuid::new_v4(),
            before_frame_id: uuid::Uuid::new_v4(),
            before,
            before_z: ExperienceLatent {
                t_ms: 100 + i * 100,
                z: vec![0.1, 0.2, 0.3, 0.4],
                ..ExperienceLatent::default()
            },
            action: Some(ActionPrimitive::Drive {
                forward: 0.25,
                turn: 0.05,
                duration_ms: 100,
            }),
            predicted_futures: Vec::new(),
            after,
            after_z: ExperienceLatent {
                t_ms: 200 + i * 100,
                z: vec![0.2, 0.3, 0.4, 0.5],
                ..ExperienceLatent::default()
            },
            reward: Reward { value: 0.0 },
            surprise: SurpriseSense::default(),
            created_at_ms: 200 + i * 100,
        });
    }
    let transitions_file = session_dir.join("transitions.jsonl");
    let content = transitions
        .iter()
        .map(|transition| serde_json::to_string(transition).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&transitions_file, format!("{content}\n")).unwrap();

    let report_path = temp_dir.join("unified-report.json");
    let report = train_unified_experience(TrainUnifiedExperienceRequest {
        ledger_path: ledger_dir,
        checkpoint_path: temp_dir.join("checkpoint"),
        report_path: report_path.clone(),
        epochs: 1,
        validation_split: 0.25,
        seed: 17,
        z_dim: 6,
        teacher_dim: 8,
    })
    .await
    .unwrap();

    assert!(report_path.exists());
    assert_eq!(report.schema_version, 1);
    assert_eq!(report.instant.representation, "UnifiedExperienceInstant");
    assert_eq!(report.instant.mask_dim, UNIFIED_TEACHER_SLOTS.len());
    assert!(report.instant.input_dim > report.teacher_dim * UNIFIED_TEACHER_SLOTS.len());
    assert_eq!(report.latent_dim, 6);
    assert_eq!(
        report.instant.teacher_slots,
        vec![
            "scene".to_string(),
            "face".to_string(),
            "voice".to_string(),
            "transcript".to_string(),
            "depth_range".to_string(),
            "memory".to_string(),
        ]
    );
    assert!(report
        .modality_coverage
        .iter()
        .any(|slot| slot.slot == "scene" && slot.present_count > 0));
    assert_eq!(report.modality_coverage.len(), 6);
    assert!(report
        .reconstruction
        .head_losses
        .contains_key("teacher_vectors"));
    assert!(report
        .reconstruction
        .head_losses
        .contains_key("modality_mask"));
    assert!(report
        .predictors
        .iter()
        .any(|predictor| predictor.encoder == "unified-experience-latent"));
    assert!(report
        .predictors
        .iter()
        .any(|predictor| predictor.encoder == "mechanical-instant"));
    assert!(report.baselines.trained_loss_mean.is_some());
    assert_eq!(report.learned_loop.canonical_instant, "ExperienceInstant");
    assert_eq!(report.learned_loop.canonical_latent, "ExperienceLatent");
    assert!(report.learned_loop.sample_count > 0);
    assert!(report
        .learned_loop
        .records
        .iter()
        .all(|record| record.encoded_latent.len() == report.latent_dim));

    let _ = fs::remove_dir_all(&temp_dir);
}

fn multimodal_now(t_ms: u64, value: f32) -> Now {
    let mut body = BodySense::default();
    body.battery_level = (0.6 + value * 0.01).clamp(0.0, 1.0);
    body.flags.bump_left = value as i32 % 4 == 0;
    body.odometry.x_m = value * 0.1;
    body.velocity.forward_m_s = 0.05 + value * 0.01;
    let mut now = Now::blank(t_ms, body);
    now.eye.scene_vectors.push(
        VectorArtifact::new(
            "scene_vectors",
            format!("scene-{t_ms}"),
            vec![0.1 + value, 0.2, 0.3, 0.4],
        )
        .with_model("test.scene"),
    );
    now.face.vectors.push(
        VectorArtifact::new("faces", format!("face-{t_ms}"), vec![0.4, 0.2 + value, 0.1])
            .with_model("test.face"),
    );
    now.voice.vectors.push(
        VectorArtifact::new("voices", format!("voice-{t_ms}"), vec![0.3, 0.7, value])
            .with_model("test.voice"),
    );
    now.ear.asr.transcript = Some(format!("hello pete {value:.1}"));
    now.ear.asr.confidence = 0.8;
    now.range.nearest_m = Some(0.4 + value * 0.02);
    now.range.beams = vec![0.4 + value * 0.01, 0.9, 1.4];
    now.kinect.depth_m = vec![0.5 + value * 0.01, 1.0, 1.5];
    now.kinect.depth_width = 3;
    now.kinect.depth_height = 1;
    now.memory.place_familiarity = 0.3 + value * 0.01;
    now.memory.place_novelty = 0.2;
    now.memory.similar_situation_count = 2;
    now.predictions.uncertainty = 0.15;
    now
}
