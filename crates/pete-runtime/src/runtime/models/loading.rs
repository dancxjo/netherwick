fn load_locomotion_behavior(behavior: &BehaviorConfig) -> Result<Option<NeatLocomotionBehavior>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    let checkpoint_path = Path::new(checkpoint);
    let artifact = if checkpoint_path.extension().is_some() {
        checkpoint_path.to_path_buf()
    } else {
        checkpoint_path.join("locomotion-neat.json")
    };
    if !artifact.exists() {
        return Ok(None);
    }
    Ok(Some(NeatLocomotionBehavior::load(checkpoint)?))
}

fn load_danger_behavior_trainer(behavior: &BehaviorConfig) -> Result<Option<DangerNetTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_danger_metadata(checkpoint)?;
    Ok(Some(DangerNetTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
    )?))
}

fn load_charge_behavior_trainer(behavior: &BehaviorConfig) -> Result<Option<ChargeNetTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_charge_metadata(checkpoint)?;
    Ok(Some(ChargeNetTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
    )?))
}

fn load_action_value_behavior_trainer(
    behavior: &BehaviorConfig,
) -> Result<Option<ActionValueNetTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_action_value_metadata(checkpoint)?;
    Ok(Some(ActionValueNetTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
    )?))
}

fn load_future_behavior_trainer(behavior: &BehaviorConfig) -> Result<Option<FutureNetTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_future_metadata(checkpoint)?;
    Ok(Some(FutureNetTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
        metadata.latent_dim,
    )?))
}

fn load_eye_next_behavior_trainer(behavior: &BehaviorConfig) -> Result<Option<EyeNextNetTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_eye_next_metadata(checkpoint)?;
    Ok(Some(EyeNextNetTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
    )?))
}

fn load_ear_next_behavior_trainer(behavior: &BehaviorConfig) -> Result<Option<EarNextNetTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_ear_next_metadata(checkpoint)?;
    Ok(Some(EarNextNetTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
    )?))
}

fn load_experience_behavior_trainer(
    behavior: &BehaviorConfig,
) -> Result<Option<ExperienceAutoencoderTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_experience_autoencoder_metadata(checkpoint)?;
    Ok(Some(ExperienceAutoencoderTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
    )?))
}

fn danger_behavior_input(
    now: &Now,
    latent: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
) -> SituatedDangerInput {
    SituatedDangerInput {
        input: DangerInput::from_parts(latent.z.clone(), action, now),
        now: now.clone(),
    }
}

fn charge_behavior_input(
    now: &Now,
    latent: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
) -> SituatedChargeInput {
    SituatedChargeInput {
        input: ChargeInput::from_parts(latent.z.clone(), action, now),
        now: now.clone(),
    }
}

fn action_value_behavior_input(
    now: &Now,
    latent: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
    danger: Option<DangerOutput>,
    charge: Option<ChargeOutput>,
) -> SituatedActionValueInput {
    SituatedActionValueInput {
        input: ActionValueInput::from_parts_with_predictions(
            latent.z.clone(),
            action,
            now,
            danger,
            charge,
        ),
        now: now.clone(),
    }
}

fn eye_next_behavior_input(
    now: &Now,
    latent: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
    offset_ms: TimeMs,
) -> SituatedEyeNextInput {
    SituatedEyeNextInput {
        input: EyeNextInput::from_parts(latent.z.clone(), action, now, offset_ms),
        now: now.clone(),
    }
}

fn ear_next_behavior_input(
    now: &Now,
    latent: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
    offset_ms: TimeMs,
) -> SituatedEarNextInput {
    SituatedEarNextInput {
        input: EarNextInput::from_parts(latent.z.clone(), action, now, offset_ms),
        now: now.clone(),
    }
}

fn danger_disagreement(left: &DangerOutput, right: &DangerOutput) -> f32 {
    let deltas = [
        (left.bump_risk - right.bump_risk).abs(),
        (left.cliff_risk - right.cliff_risk).abs(),
        (left.wheel_drop_risk - right.wheel_drop_risk).abs(),
        (left.stuck_risk - right.stuck_risk).abs(),
    ];
    deltas.iter().sum::<f32>() / deltas.len() as f32
}

fn action_value_disagreement(left: &ActionValueOutput, right: &ActionValueOutput) -> f32 {
    (left.value - right.value).abs()
}

fn charge_disagreement(left: &ChargeOutput, right: &ChargeOutput) -> f32 {
    let deltas = [
        (left.charge_probability - right.charge_probability).abs(),
        (left.expected_battery_delta - right.expected_battery_delta).abs(),
        (left.dock_likelihood - right.dock_likelihood).abs(),
    ];
    deltas.iter().sum::<f32>() / deltas.len() as f32
}

fn eye_next_disagreement(left: &EyeNextOutput, right: &EyeNextOutput) -> f32 {
    let len = left.rgb.len().max(right.rgb.len());
    if len == 0 {
        return 0.0;
    }
    (0..len)
        .map(|idx| {
            let left = left.rgb.get(idx).copied().unwrap_or_default() as f32 / 255.0;
            let right = right.rgb.get(idx).copied().unwrap_or_default() as f32 / 255.0;
            (left - right).abs()
        })
        .sum::<f32>()
        / len as f32
}

fn ear_next_disagreement(left: &EarNextOutput, right: &EarNextOutput) -> f32 {
    let len = left.features.len().max(right.features.len());
    if len == 0 {
        return 0.0;
    }
    (0..len)
        .map(|idx| {
            let left = left.features.get(idx).copied().unwrap_or_default();
            let right = right.features.get(idx).copied().unwrap_or_default();
            (left - right).abs()
        })
        .sum::<f32>()
        / len as f32
}

fn experience_reconstruction_loss_flat(
    output: &ExperienceDecodeOutput,
    target: &ExperienceDecodeOutput,
) -> f32 {
    let output = output.flat_features();
    let target = target.flat_features();
    let len = output.len().max(target.len());
    if len == 0 {
        return 0.0;
    }
    (0..len)
        .map(|idx| {
            let actual = output.get(idx).copied().unwrap_or_default();
            let expected = target.get(idx).copied().unwrap_or_default();
            let delta = actual - expected;
            delta * delta
        })
        .sum::<f32>()
        / len as f32
}

fn experience_disagreement(
    left: &ExperienceBehaviorOutput,
    right: &ExperienceBehaviorOutput,
) -> f32 {
    let a = &left.latent.z;
    let b = &right.latent.z;
    let len = a.len().max(b.len());
    if len == 0 {
        return 0.0;
    }
    let sum: f32 = (0..len)
        .map(|idx| {
            let delta =
                a.get(idx).copied().unwrap_or_default() - b.get(idx).copied().unwrap_or_default();
            delta * delta
        })
        .sum();
    sum.sqrt()
}
