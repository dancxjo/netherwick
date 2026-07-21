fn tensor_to_danger_output<B: Backend>(tensor: Tensor<B, 2>) -> Result<DangerOutput> {
    let values = tensor.into_data().to_vec::<f32>()?;
    if values.len() != 4 {
        return Err(anyhow!(
            "danger net emitted {} outputs, expected 4",
            values.len()
        ));
    }
    Ok(DangerOutput {
        bump_risk: values[0].clamp(0.0, 1.0),
        cliff_risk: values[1].clamp(0.0, 1.0),
        wheel_drop_risk: values[2].clamp(0.0, 1.0),
        stuck_risk: values[3].clamp(0.0, 1.0),
        confidence: 0.5,
    })
}

fn tensor_to_charge_output<B: Backend>(tensor: Tensor<B, 2>) -> Result<ChargeOutput> {
    let values = tensor.into_data().to_vec::<f32>()?;
    if values.len() != 3 {
        return Err(anyhow!(
            "charge net emitted {} outputs, expected 3",
            values.len()
        ));
    }
    Ok(ChargeOutput {
        charge_probability: values[0].clamp(0.0, 1.0),
        expected_battery_delta: (values[1] * 2.0 - 1.0).clamp(-1.0, 1.0),
        dock_likelihood: values[2].clamp(0.0, 1.0),
        confidence: 0.5,
    })
}

fn tensor_to_action_value_output<B: Backend>(tensor: Tensor<B, 2>) -> Result<ActionValueOutput> {
    let values = tensor.into_data().to_vec::<f32>()?;
    if values.len() != 2 {
        return Err(anyhow!(
            "action-value net emitted {} outputs, expected 2",
            values.len()
        ));
    }
    Ok(ActionValueOutput {
        value: (values[0] * 2.0 - 1.0).clamp(-1.0, 1.0),
        confidence: values[1].clamp(0.0, 1.0),
    })
}

fn tensor_to_future_prediction<B: Backend>(
    tensor: Tensor<B, 2>,
    offset_ms: TimeMs,
    latent_dim: usize,
) -> Result<FuturePrediction> {
    let values = tensor.into_data().to_vec::<f32>()?;
    if values.len() != latent_dim {
        return Err(anyhow!(
            "future net emitted {} outputs, expected {}",
            values.len(),
            latent_dim
        ));
    }
    Ok(FuturePrediction {
        offset_ms,
        predicted_z: values
            .into_iter()
            .map(|value| value.clamp(0.0, 1.0))
            .collect(),
        confidence: 0.5,
        summary: Some("Learned latent future prediction.".to_string()),
    })
}

fn tensor_to_eye_next_output<B: Backend>(
    tensor: Tensor<B, 2>,
    width: u32,
    height: u32,
) -> Result<EyeNextOutput> {
    let values = tensor.into_data().to_vec::<f32>()?;
    let expected_len = width as usize * height as usize * 3;
    if values.len() != expected_len {
        return Err(anyhow!(
            "eye-next net emitted {} outputs, expected {}",
            values.len(),
            expected_len
        ));
    }
    Ok(EyeNextOutput {
        width,
        height,
        rgb: values
            .into_iter()
            .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect(),
        confidence: 0.5,
    })
}

fn tensor_to_ear_next_output<B: Backend>(
    tensor: Tensor<B, 2>,
    output_dim: usize,
    sample_rate_hz: u32,
    channels: u16,
) -> Result<EarNextOutput> {
    let values = tensor.into_data().to_vec::<f32>()?;
    if values.len() != output_dim {
        return Err(anyhow!(
            "ear-next net emitted {} outputs, expected {}",
            values.len(),
            output_dim
        ));
    }
    Ok(EarNextOutput {
        sample_rate_hz,
        channels,
        pcm: Vec::new(),
        features: values
            .into_iter()
            .map(|value| value.clamp(0.0, 1.0))
            .collect(),
        confidence: 0.5,
    })
}

fn tensor_to_experience_encode_output<B: Backend>(
    tensor: Tensor<B, 2>,
    z_dim: usize,
) -> Result<ExperienceEncodeOutput> {
    let values = tensor.into_data().to_vec::<f32>()?;
    if values.len() != z_dim {
        return Err(anyhow!(
            "experience autoencoder emitted {} z outputs, expected {}",
            values.len(),
            z_dim
        ));
    }
    Ok(ExperienceEncodeOutput {
        z: values
            .into_iter()
            .map(|value| value.clamp(0.0, 1.0))
            .collect(),
        confidence: 0.5,
    })
}

fn tensor_to_experience_decode_output<B: Backend>(
    tensor: Tensor<B, 2>,
    output_dim: usize,
    lengths: ExperienceDecodeFeatureLengths,
) -> Result<ExperienceDecodeOutput> {
    let values = tensor.into_data().to_vec::<f32>()?;
    if values.len() != output_dim {
        return Err(anyhow!(
            "experience autoencoder emitted {} reconstruction outputs, expected {}",
            values.len(),
            output_dim
        ));
    }
    Ok(split_experience_decode_values(
        values
            .into_iter()
            .map(|value| value.clamp(0.0, 1.0))
            .collect(),
        lengths,
    ))
}

fn charge_target_train_values(target: &ChargeTarget) -> [f32; 3] {
    [
        target.charging_started.clamp(0.0, 1.0),
        ((target.battery_delta.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0),
        target.charging_after.clamp(0.0, 1.0),
    ]
}

fn action_value_target_train_values(target: &ActionValueTarget) -> [f32; 2] {
    [
        ((target.value.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0),
        1.0,
    ]
}

fn future_target_train_values(target_z: &[f32], latent_dim: usize) -> Vec<f32> {
    let mut values = target_z
        .iter()
        .take(latent_dim)
        .copied()
        .map(|value| value.clamp(0.0, 1.0))
        .collect::<Vec<_>>();
    values.resize(latent_dim, 0.0);
    values
}

fn eye_target_train_values(target: &EyeNextTarget, output_dim: usize) -> Vec<f32> {
    let mut values = target
        .rgb
        .iter()
        .take(output_dim)
        .map(|byte| *byte as f32 / 255.0)
        .collect::<Vec<_>>();
    values.resize(output_dim, 0.0);
    values
}

fn ear_target_train_values(target: &EarNextTarget, output_dim: usize) -> Vec<f32> {
    let mut values = target
        .features
        .iter()
        .take(output_dim)
        .map(|value| value.clamp(0.0, 1.0))
        .collect::<Vec<_>>();
    values.resize(output_dim, 0.0);
    values
}

fn experience_decode_target_values(target: &ExperienceDecodeOutput, output_dim: usize) -> Vec<f32> {
    let mut values = target.flat_features();
    values.resize(output_dim, 0.0);
    values.truncate(output_dim);
    values
        .into_iter()
        .map(|value| value.clamp(0.0, 1.0))
        .collect()
}

fn split_experience_decode_values(
    values: Vec<f32>,
    lengths: ExperienceDecodeFeatureLengths,
) -> ExperienceDecodeOutput {
    let mut cursor = 0;
    let mut take = |len: usize| {
        let end = (cursor + len).min(values.len());
        let mut out = values[cursor..end].to_vec();
        out.resize(len, 0.0);
        cursor = cursor.saturating_add(len);
        out
    };
    ExperienceDecodeOutput {
        body_features: take(lengths.body),
        memory_features: take(lengths.memory),
        drive_features: take(lengths.drive),
        prediction_features: take(lengths.prediction),
        eye_features: take(lengths.eye),
        ear_features: take(lengths.ear),
    }
}

fn mse_output_target(output: DangerOutput, target: DangerTarget) -> f32 {
    let output = output.risks();
    let target = target.risks();
    output
        .iter()
        .zip(target.iter())
        .map(|(actual, expected)| {
            let delta = actual - expected;
            delta * delta
        })
        .sum::<f32>()
        / 4.0
}

fn mse_charge_output_target(output: ChargeOutput, target: ChargeTarget) -> f32 {
    let output = output.values();
    let target = target.values();
    output
        .iter()
        .zip(target.iter())
        .map(|(actual, expected)| {
            let delta = actual - expected;
            delta * delta
        })
        .sum::<f32>()
        / 3.0
}

fn mse_action_value_output_target(output: ActionValueOutput, target: ActionValueTarget) -> f32 {
    let delta = output.value - target.value;
    delta * delta
}

fn mse_vec_target(output: &[f32], target: &[f32]) -> f32 {
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

fn mse_eye_next_output_target(output: &EyeNextOutput, target: &EyeNextTarget) -> f32 {
    let len = output.rgb.len().max(target.rgb.len());
    if len == 0 {
        return 0.0;
    }
    (0..len)
        .map(|idx| {
            let actual = output.rgb.get(idx).copied().unwrap_or_default() as f32 / 255.0;
            let expected = target.rgb.get(idx).copied().unwrap_or_default() as f32 / 255.0;
            let delta = actual - expected;
            delta * delta
        })
        .sum::<f32>()
        / len as f32
}

fn mse_ear_next_output_target(output: &EarNextOutput, target: &EarNextTarget) -> f32 {
    let len = output.features.len().max(target.features.len());
    if len == 0 {
        return 0.0;
    }
    (0..len)
        .map(|idx| {
            let actual = output.features.get(idx).copied().unwrap_or_default();
            let expected = target.features.get(idx).copied().unwrap_or_default();
            let delta = actual - expected;
            delta * delta
        })
        .sum::<f32>()
        / len as f32
}

fn mse_experience_decode_output_target(
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
