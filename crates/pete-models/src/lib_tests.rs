use super::*;
use pete_actions::ActionPrimitive;
use pete_body::BodySense;
use pete_experience::{
    experience_decode_target_from_now, experience_encode_input_from_now, ActionValueInput,
    ActionValueTarget, ChargeInput, ChargeTarget, DangerInput, EarNextInput, EarNextTarget,
    EyeNextInput, EyeNextTarget, FutureInput, FuturePrediction,
};

#[test]
fn hardcoded_uses_current_now_for_body_danger() {
    let mut now = Now::blank(1, BodySense::default());
    now.body.flags.bump_left = true;
    let input = DangerInput::from_parts(vec![0.0], Some(&ActionPrimitive::Stop), &now);

    let output = HardcodedDangerPredictor.predict_from_now(&now, &input);

    assert_eq!(output.bump_risk, 1.0);
    assert!(output.confidence > 0.0);
}

#[test]
fn danger_net_forward_returns_unit_risks() {
    let now = Now::blank(1, BodySense::default());
    let input = DangerInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Stop), &now);
    let trainer = DangerNetTrainer::new(input.flat_features().len());

    let output = trainer.predict(&input).unwrap();

    for risk in output.risks() {
        assert!((0.0..=1.0).contains(&risk));
    }
}

#[test]
fn one_train_step_records_loss() {
    let now = Now::blank(1, BodySense::default());
    let input = DangerInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Stop), &now);
    let mut trainer = DangerNetTrainer::new(input.flat_features().len());
    let target = DangerTarget {
        bump: 1.0,
        ..DangerTarget::default()
    };

    let stats = trainer.train_step(&input, &target).unwrap();

    assert_eq!(stats.samples_seen, 1);
    assert!(stats.loss.is_finite());
}

#[test]
fn shadow_comparison_writes_metric_shape() {
    let now = Now::blank(10, BodySense::default());
    let input = DangerInput::from_parts(vec![0.1], Some(&ActionPrimitive::Stop), &now);
    let mut trainer = DangerNetTrainer::new(input.flat_features().len());
    let target = DangerTarget::default();

    let metric = trainer.shadow_compare(10, &now, &input, &target).unwrap();

    assert_eq!(metric.observed_at_ms, 10);
    assert!(metric.loss.is_finite());
}

#[test]
fn danger_checkpoint_round_trips_prediction_shape() {
    let dir = std::env::temp_dir().join(format!("pete-danger-checkpoint-{}", now_ms()));
    let now = Now::blank(1, BodySense::default());
    let input = DangerInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Stop), &now);
    let mut trainer = DangerNetTrainer::new(input.flat_features().len());
    trainer
        .train_step(
            &input,
            &DangerTarget {
                bump: 1.0,
                ..DangerTarget::default()
            },
        )
        .unwrap();

    trainer.save_checkpoint(&dir).unwrap();
    let loaded = DangerNetTrainer::load_checkpoint(&dir, input.flat_features().len()).unwrap();
    let output = loaded.predict(&input).unwrap();

    assert!(dir.join("model.bin").exists());
    assert!(dir.join("metadata.json").exists());
    assert_eq!(loaded.samples_seen(), 1);
    for risk in output.risks() {
        assert!((0.0..=1.0).contains(&risk));
    }

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn danger_checkpoint_rejects_dimension_mismatch() {
    let dir = std::env::temp_dir().join(format!("pete-danger-checkpoint-mismatch-{}", now_ms()));
    let trainer = DangerNetTrainer::new(3);

    trainer.save_checkpoint(&dir).unwrap();
    let err = match DangerNetTrainer::load_checkpoint(&dir, 4) {
        Ok(_) => panic!("expected dimension mismatch"),
        Err(err) => err,
    };

    assert!(err
        .to_string()
        .contains("danger checkpoint input dimension mismatch"));

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn hardcoded_charge_uses_current_charging_state() {
    let mut now = Now::blank(1, BodySense::default());
    now.body.charging = true;
    let input = ChargeInput::from_parts(vec![0.0], Some(&ActionPrimitive::Stop), &now);

    let output = HardcodedChargePredictor.predict_from_now(&now, &input);

    assert!(output.charge_probability > 0.9);
    assert!(output.expected_battery_delta > 0.0);
}

#[test]
fn hardcoded_charge_distinguishes_visible_from_dock_contact() {
    let mut now = Now::blank(1, BodySense::default());
    now.body.battery_level = 0.2;
    let mut input = ChargeInput::from_parts(vec![0.0], Some(&ActionPrimitive::Dock), &now);
    input.body_features.resize(10, 0.0);
    input.body_features[7] = 0.8;

    let output = HardcodedChargePredictor.predict_from_now(&now, &input);

    assert!(output.charge_probability >= 0.8);
    assert!(output.dock_likelihood < 0.6);
}

#[test]
fn charge_net_forward_returns_bounded_outputs() {
    let now = Now::blank(1, BodySense::default());
    let input = ChargeInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Dock), &now);
    let trainer = ChargeNetTrainer::new(input.flat_features().len());

    let output = trainer.predict(&input).unwrap();

    assert!((0.0..=1.0).contains(&output.charge_probability));
    assert!((-1.0..=1.0).contains(&output.expected_battery_delta));
    assert!((0.0..=1.0).contains(&output.dock_likelihood));
}

#[test]
fn charge_checkpoint_round_trips_prediction_shape() {
    let dir = std::env::temp_dir().join(format!("pete-charge-checkpoint-{}", now_ms()));
    let now = Now::blank(1, BodySense::default());
    let input = ChargeInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Dock), &now);
    let mut trainer = ChargeNetTrainer::new(input.flat_features().len());
    trainer
        .train_step(
            &input,
            &ChargeTarget {
                charging_started: 1.0,
                battery_delta: 0.03,
                charging_after: 1.0,
            },
        )
        .unwrap();

    trainer.save_checkpoint(&dir).unwrap();
    let loaded = ChargeNetTrainer::load_checkpoint(&dir, input.flat_features().len()).unwrap();
    let output = loaded.predict(&input).unwrap();

    assert!(dir.join("model.bin").exists());
    assert!(dir.join("metadata.json").exists());
    assert_eq!(loaded.samples_seen(), 1);
    assert!((0.0..=1.0).contains(&output.charge_probability));

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn action_value_net_forward_returns_finite_value() {
    let now = Now::blank(1, BodySense::default());
    let input = ActionValueInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Dock), &now);
    let trainer = ActionValueNetTrainer::new(input.flat_features().len());

    let output = trainer.predict(&input).unwrap();

    assert!(output.value.is_finite());
    assert!(output.confidence.is_finite());
    assert!((-1.0..=1.0).contains(&output.value));
    assert!((0.0..=1.0).contains(&output.confidence));
}

#[test]
fn action_value_checkpoint_round_trips_prediction_shape() {
    let dir = std::env::temp_dir().join(format!("pete-action-value-checkpoint-{}", now_ms()));
    let now = Now::blank(1, BodySense::default());
    let input = ActionValueInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Dock), &now);
    let mut trainer = ActionValueNetTrainer::new(input.flat_features().len());
    trainer
        .train_step(&input, &ActionValueTarget { value: 0.4 })
        .unwrap();

    trainer.save_checkpoint(&dir).unwrap();
    let loaded = ActionValueNetTrainer::load_checkpoint(&dir, input.flat_features().len()).unwrap();
    let output = loaded.predict(&input).unwrap();

    assert!(dir.join("model.bin").exists());
    assert!(dir.join("metadata.json").exists());
    assert_eq!(loaded.samples_seen(), 1);
    assert!(output.value.is_finite());

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn future_net_forward_returns_latent_dim_outputs() {
    let input = FutureInput {
        latent: pete_experience::ExperienceLatent {
            t_ms: 10,
            z: vec![0.1, 0.2, 0.3],
            confidence: 0.8,
            ..pete_experience::ExperienceLatent::default()
        },
        action: ActionPrimitive::Stop,
        offset_ms: 1_000,
    };
    let trainer = FutureNetTrainer::new(input.flat_features().len(), input.latent.z.len());

    let output = trainer.predict(&input).unwrap();

    assert_eq!(output.predicted_z.len(), input.latent.z.len());
    assert_eq!(output.offset_ms, input.offset_ms);
    assert!(output.predicted_z.iter().all(|value| value.is_finite()));
}

#[test]
fn future_train_step_and_shadow_compare_record_finite_loss() {
    let input = FutureInput {
        latent: pete_experience::ExperienceLatent {
            t_ms: 10,
            z: vec![0.1, 0.2],
            confidence: 0.8,
            ..pete_experience::ExperienceLatent::default()
        },
        action: ActionPrimitive::Go {
            intensity: 0.2,
            duration_ms: 500,
        },
        offset_ms: 1_000,
    };
    let target = vec![0.2, 0.4];
    let mut trainer = FutureNetTrainer::new(input.flat_features().len(), target.len());
    let hardcoded = FuturePrediction {
        offset_ms: input.offset_ms,
        predicted_z: input.latent.z.clone(),
        confidence: 0.7,
        summary: None,
    };

    let metric = trainer.shadow_compare(&input, &hardcoded, &target).unwrap();
    let stats = trainer.train_step(&input, &target).unwrap();

    assert_eq!(metric.offset_ms, 1_000);
    assert!(metric.model_loss.is_finite());
    assert_eq!(stats.samples_seen, 1);
    assert!(stats.loss.is_finite());
}

#[test]
fn future_checkpoint_round_trips_prediction_shape() {
    let dir = std::env::temp_dir().join(format!("pete-future-checkpoint-{}", now_ms()));
    let input = FutureInput {
        latent: pete_experience::ExperienceLatent {
            t_ms: 10,
            z: vec![0.1, 0.2, 0.3],
            confidence: 0.8,
            ..pete_experience::ExperienceLatent::default()
        },
        action: ActionPrimitive::Stop,
        offset_ms: 1_000,
    };
    let mut trainer = FutureNetTrainer::new(input.flat_features().len(), input.latent.z.len());
    trainer.train_step(&input, &[0.3, 0.2, 0.1]).unwrap();

    trainer.save_checkpoint(&dir).unwrap();
    let loaded =
        FutureNetTrainer::load_checkpoint(&dir, input.flat_features().len(), input.latent.z.len())
            .unwrap();
    let output = loaded.predict(&input).unwrap();

    assert!(dir.join("model.bin").exists());
    assert!(dir.join("metadata.json").exists());
    assert_eq!(loaded.samples_seen(), 1);
    assert_eq!(output.predicted_z.len(), input.latent.z.len());

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn future_checkpoint_rejects_dimension_mismatch() {
    let dir = std::env::temp_dir().join(format!("pete-future-checkpoint-mismatch-{}", now_ms()));
    let trainer = FutureNetTrainer::new(4, 2);

    trainer.save_checkpoint(&dir).unwrap();
    let err = match FutureNetTrainer::load_checkpoint(&dir, 5, 2) {
        Ok(_) => panic!("expected dimension mismatch"),
        Err(err) => err,
    };

    assert!(err
        .to_string()
        .contains("future checkpoint input dimension mismatch"));

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn eye_next_checkpoint_round_trips_prediction_shape() {
    let dir = std::env::temp_dir().join(format!("pete-eye-next-checkpoint-{}", now_ms()));
    let mut now = Now::blank(1, BodySense::default());
    now.eye.frames = vec![vec![0.2, 0.4, 0.6, 0.8]];
    let input = EyeNextInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Stop), &now, 100);
    let mut trainer = EyeNextNetTrainer::new(input.flat_features().len(), 4, 4);
    let target = EyeNextTarget {
        width: 4,
        height: 4,
        rgb: vec![128; 4 * 4 * 3],
    };
    trainer.train_step(&input, &target).unwrap();

    trainer.save_checkpoint(&dir).unwrap();
    let loaded = EyeNextNetTrainer::load_checkpoint(&dir, input.flat_features().len()).unwrap();
    let output = loaded.predict(&input).unwrap();

    assert!(dir.join("model.bin").exists());
    assert!(dir.join("metadata.json").exists());
    assert_eq!(loaded.samples_seen(), 1);
    assert_eq!(output.width, 4);
    assert_eq!(output.height, 4);
    assert_eq!(output.rgb.len(), 4 * 4 * 3);

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn copy_current_ear_predictor_returns_current_features() {
    let mut now = Now::blank(1, BodySense::default());
    now.ear.features = vec![vec![0.2, 0.4], vec![0.6, 0.8]];
    let input = EarNextInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Stop), &now, 100);

    let output = CopyCurrentEarPredictor.predict_from_now(&now, &input);

    assert_eq!(output.features, vec![0.2, 0.4, 0.6, 0.8]);
    assert!(output.pcm.is_empty());
    assert!(output.confidence > 0.0);
}

#[test]
fn copy_current_ear_predictor_returns_zero_features_without_audio() {
    let now = Now::blank(1, BodySense::default());
    let input = EarNextInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Stop), &now, 100);

    let output = CopyCurrentEarPredictor.predict_from_now(&now, &input);

    assert_eq!(output.features, vec![0.0; input.ear_features.len()]);
    assert!(output.pcm.is_empty());
    assert_eq!(output.confidence, 0.0);
}

#[test]
fn ear_next_net_forward_returns_bounded_features() {
    let mut now = Now::blank(1, BodySense::default());
    now.ear.features = vec![vec![0.2, 0.4, 0.6, 0.8]];
    let input = EarNextInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Stop), &now, 100);
    let trainer = EarNextNetTrainer::new(input.flat_features().len(), 4);

    let output = trainer.predict(&input).unwrap();

    assert_eq!(output.features.len(), 4);
    assert!(output
        .features
        .iter()
        .all(|value| (0.0..=1.0).contains(value)));
}

#[test]
fn ear_next_checkpoint_round_trips_prediction_shape() {
    let dir = std::env::temp_dir().join(format!("pete-ear-next-checkpoint-{}", now_ms()));
    let mut now = Now::blank(1, BodySense::default());
    now.ear.features = vec![vec![0.2, 0.4, 0.6, 0.8]];
    let input = EarNextInput::from_parts(vec![0.1, 0.2], Some(&ActionPrimitive::Stop), &now, 100);
    let mut trainer = EarNextNetTrainer::new(input.flat_features().len(), 4);
    let target = EarNextTarget {
        features: vec![0.1, 0.3, 0.5, 0.7],
        ..EarNextTarget::default()
    };
    trainer.train_step(&input, &target).unwrap();

    trainer.save_checkpoint(&dir).unwrap();
    let loaded = EarNextNetTrainer::load_checkpoint(&dir, input.flat_features().len()).unwrap();
    let output = loaded.predict(&input).unwrap();

    assert!(dir.join("model.bin").exists());
    assert!(dir.join("metadata.json").exists());
    assert_eq!(loaded.samples_seen(), 1);
    assert_eq!(output.features.len(), 4);
    assert!(output.pcm.is_empty());

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn experience_autoencoder_forward_returns_fixed_size_z_and_decode_lengths() {
    let mut now = Now::blank(1, BodySense::default());
    now.eye.frames = vec![vec![0.2, 0.4, 0.6, 0.8]];
    now.ear.features = vec![vec![0.2, 0.4, 0.6, 0.8]];
    let input = experience_encode_input_from_now(&now);
    let target = experience_decode_target_from_now(&now);
    let trainer = ExperienceAutoencoderTrainer::new(
        input.flat_features().len(),
        12,
        target.feature_lengths(),
    );

    let prediction = trainer.predict(&input).unwrap();

    assert_eq!(prediction.encoded.z.len(), 12);
    assert_eq!(
        prediction.decoded.feature_lengths(),
        target.feature_lengths()
    );
    assert_eq!(
        prediction.decoded.eye_features.len(),
        target.eye_features.len()
    );
    assert_eq!(
        prediction.decoded.ear_features.len(),
        target.ear_features.len()
    );
}

#[test]
fn experience_autoencoder_train_step_records_loss() {
    let mut now = Now::blank(1, BodySense::default());
    now.memory.place_familiarity = 0.7;
    now.drives.curiosity = 0.5;
    let input = experience_encode_input_from_now(&now);
    let target = experience_decode_target_from_now(&now);
    let mut trainer =
        ExperienceAutoencoderTrainer::new(input.flat_features().len(), 8, target.feature_lengths());

    let stats = trainer.train_step(&input, &target).unwrap();

    assert_eq!(stats.samples_seen, 1);
    assert!(stats.loss.is_finite());
}

#[test]
fn experience_autoencoder_checkpoint_round_trips_prediction_shape() {
    let dir = std::env::temp_dir().join(format!("pete-experience-checkpoint-{}", now_ms()));
    let mut now = Now::blank(1, BodySense::default());
    now.eye.frames = vec![vec![0.2, 0.4, 0.6, 0.8]];
    now.ear.features = vec![vec![0.2, 0.4, 0.6, 0.8]];
    let input = experience_encode_input_from_now(&now);
    let target = experience_decode_target_from_now(&now);
    let mut trainer = ExperienceAutoencoderTrainer::new(
        input.flat_features().len(),
        10,
        target.feature_lengths(),
    );
    trainer.train_step(&input, &target).unwrap();

    trainer.save_checkpoint(&dir).unwrap();
    let loaded =
        ExperienceAutoencoderTrainer::load_checkpoint(&dir, input.flat_features().len()).unwrap();
    let prediction = loaded.predict(&input).unwrap();

    assert!(dir.join("model.bin").exists());
    assert!(dir.join("metadata.json").exists());
    assert_eq!(loaded.samples_seen(), 1);
    assert_eq!(prediction.encoded.z.len(), 10);
    assert_eq!(
        prediction.decoded.feature_lengths(),
        target.feature_lengths()
    );

    let _ = std::fs::remove_dir_all(dir);
}
