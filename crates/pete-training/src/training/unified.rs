#[derive(Clone, Copy, Debug)]
struct UnifiedTeacherSlot {
    name: &'static str,
    source: &'static str,
    purpose: &'static str,
}

const UNIFIED_TEACHER_SLOTS: [UnifiedTeacherSlot; 6] = [
    UnifiedTeacherSlot {
        name: "scene",
        source: "eye image/scene vectors",
        purpose: "visual scene similarity",
    },
    UnifiedTeacherSlot {
        name: "face",
        source: "face identity vectors",
        purpose: "person identity",
    },
    UnifiedTeacherSlot {
        name: "voice",
        source: "voice/audio identity vectors",
        purpose: "speaker identity",
    },
    UnifiedTeacherSlot {
        name: "transcript",
        source: "ASR/transcript text hash",
        purpose: "text semantic bridge",
    },
    UnifiedTeacherSlot {
        name: "depth_range",
        source: "range and Kinect depth summaries",
        purpose: "near-field geometry",
    },
    UnifiedTeacherSlot {
        name: "memory",
        source: "memory recall/state vector",
        purpose: "remembered context",
    },
];

#[derive(Clone, Debug)]
struct UnifiedInstant {
    input: ExperienceEncodeInput,
    target: ExperienceDecodeOutput,
    slot_presence: Vec<f32>,
}

#[derive(Clone, Debug)]
struct UnifiedExperienceExample {
    t_ms: TimeMs,
    input: ExperienceEncodeInput,
    target: ExperienceDecodeOutput,
    next_input: ExperienceEncodeInput,
    slot_presence: Vec<f32>,
    action: ActionPrimitive,
    offset_ms: TimeMs,
}

fn unified_examples_from_transitions(
    transitions: &[ExperienceTransition],
    teacher_dim: usize,
) -> Result<Vec<UnifiedExperienceExample>> {
    let teacher_dim = teacher_dim.max(2);
    let mut examples = Vec::new();
    for transition in transitions {
        let Some(action) = transition.action.clone() else {
            continue;
        };
        let offset_ms = transition
            .after
            .t_ms
            .saturating_sub(transition.before.t_ms)
            .max(1);
        let now =
            unified_instant_from_now(&transition.before, teacher_dim, Some(&action), offset_ms);
        let next_now = unified_instant_from_now(&transition.after, teacher_dim, None, offset_ms);
        examples.push(UnifiedExperienceExample {
            t_ms: transition.before.t_ms,
            input: now.input,
            target: now.target,
            next_input: next_now.input,
            slot_presence: now.slot_presence,
            action,
            offset_ms,
        });
    }
    Ok(examples)
}

fn unified_instant_from_now(
    now: &Now,
    teacher_dim: usize,
    action: Option<&ActionPrimitive>,
    offset_ms: TimeMs,
) -> UnifiedInstant {
    let mut slot_vectors = Vec::new();
    let mut slot_presence = Vec::new();
    for slot_index in 0..UNIFIED_TEACHER_SLOTS.len() {
        let (vector, present) = unified_slot_vector(now, slot_index, teacher_dim);
        slot_vectors.push(vector);
        slot_presence.push(bool_feature(present));
    }
    let mut sense_vectors = slot_vectors.clone();
    sense_vectors.push(slot_presence.clone());
    sense_vectors.push(action_features(action));
    sense_vectors.push(unified_time_features(now, offset_ms));
    sense_vectors.push(unified_compact_sensor_summary(now));
    UnifiedInstant {
        input: ExperienceEncodeInput { sense_vectors },
        target: unified_decode_target(now, &slot_vectors, &slot_presence),
        slot_presence,
    }
}

fn unified_time_features(now: &Now, offset_ms: TimeMs) -> Vec<f32> {
    let seconds = now.t_ms as f32 / 1_000.0;
    vec![
        (seconds / 60.0).sin(),
        (seconds / 60.0).cos(),
        (offset_ms as f32 / 5_000.0).clamp(0.0, 1.0),
    ]
}

fn unified_compact_sensor_summary(now: &Now) -> Vec<f32> {
    compact_contact_features(now)
        .into_iter()
        .chain(compact_range_features(now))
        .chain(compact_depth_features(now))
        .chain([
            now.body.battery_level,
            bool_feature(now.body.charging),
            now.body.velocity.forward_m_s.clamp(-1.0, 1.0),
            now.body.velocity.turn_rad_s.clamp(-1.0, 1.0),
        ])
        .map(clean_feature)
        .collect()
}

fn unified_slot_vector(now: &Now, slot_index: usize, teacher_dim: usize) -> (Vec<f32>, bool) {
    let raw = match slot_index {
        0 => average_artifacts(
            now.eye
                .scene_vectors
                .iter()
                .chain(now.eye.image_vectors.iter())
                .chain(now.eye.image_description_vectors.iter()),
        ),
        1 => average_artifacts(now.face.vectors.iter()),
        2 => average_artifacts(now.voice.vectors.iter()),
        3 => transcript_vector(now, teacher_dim),
        4 => Some(
            compact_range_features(now)
                .into_iter()
                .chain(compact_depth_features(now))
                .collect(),
        )
        .filter(|_| !now.range.beams.is_empty() || !now.kinect.depth_m.is_empty()),
        5 => Some(memory_teacher_vector(now)).filter(|values| values.iter().any(|v| *v != 0.0)),
        _ => None,
    };
    let present = raw.as_ref().is_some_and(|values| !values.is_empty());
    (fit_vector(raw.unwrap_or_default(), teacher_dim), present)
}

fn average_artifacts<'a>(
    artifacts: impl Iterator<Item = &'a pete_now::VectorArtifact>,
) -> Option<Vec<f32>> {
    let vectors = artifacts
        .filter(|artifact| !artifact.vector.is_empty())
        .map(|artifact| artifact.vector.as_slice())
        .collect::<Vec<_>>();
    average_slices(&vectors)
}

fn average_slices(vectors: &[&[f32]]) -> Option<Vec<f32>> {
    let dim = vectors.iter().map(|vector| vector.len()).max()?;
    let mut out = vec![0.0; dim];
    let mut count = 0.0_f32;
    for vector in vectors {
        for (slot, value) in out.iter_mut().zip(vector.iter().copied()) {
            *slot += clean_feature(value);
        }
        count += 1.0;
    }
    if count == 0.0 {
        return None;
    }
    for value in &mut out {
        *value /= count;
    }
    Some(out)
}

fn transcript_vector(now: &Now, teacher_dim: usize) -> Option<Vec<f32>> {
    let text = now
        .ear
        .asr
        .transcript
        .as_deref()
        .or(now.ear.transcript.as_deref())
        .map(str::trim)
        .filter(|text| !text.is_empty())?;
    let mut out = vec![0.0; teacher_dim.max(1)];
    for (index, byte) in text.bytes().enumerate() {
        let slot = index % out.len();
        let signed = (byte as f32 / 127.5) - 1.0;
        out[slot] = (out[slot] + signed).tanh();
    }
    Some(out)
}

fn memory_teacher_vector(now: &Now) -> Vec<f32> {
    vec![
        now.memory.place_familiarity,
        now.memory.place_danger,
        now.memory.place_charge_value,
        now.memory.place_social_value,
        now.memory.place_novelty,
        now.memory.recent_trap_confidence,
        now.memory.similar_situation_count as f32 / 32.0,
        bool_feature(now.memory.remembered_warning.is_some()),
        bool_feature(now.memory.graph_context_summary.is_some()),
        now.memory
            .nearby_best_safe_direction_rad
            .unwrap_or_default()
            .sin(),
        now.memory
            .nearby_best_charge_direction_rad
            .unwrap_or_default()
            .sin(),
        now.memory
            .nearby_frontier_direction_rad
            .unwrap_or_default()
            .sin(),
    ]
    .into_iter()
    .map(clean_feature)
    .collect()
}

fn extension_vector_values(value: &serde_json::Value) -> Option<Vec<f32>> {
    if let Some(values) = value.get("values").and_then(|value| value.as_array()) {
        return Some(
            values
                .iter()
                .filter_map(|value| value.as_f64())
                .map(|value| clean_feature(value as f32))
                .collect(),
        );
    }
    value.as_array().map(|values| {
        values
            .iter()
            .filter_map(|value| value.as_f64())
            .map(|value| clean_feature(value as f32))
            .collect()
    })
}

fn fit_vector(values: Vec<f32>, dim: usize) -> Vec<f32> {
    let dim = dim.max(1);
    if values.is_empty() {
        return vec![0.0; dim];
    }
    let original_len = values.len();
    if values.len() == dim {
        return values.into_iter().map(clean_feature).collect();
    }
    let mut out = vec![0.0; dim];
    if values.len() < dim {
        for (slot, value) in out.iter_mut().zip(values.into_iter()) {
            *slot = clean_feature(value);
        }
        return out;
    }
    for (index, value) in values.into_iter().enumerate() {
        out[index % dim] += clean_feature(value);
    }
    let folds = (values_len_for_dim(original_len, dim) as f32).max(1.0);
    for value in &mut out {
        *value = (*value / folds).tanh();
    }
    out
}

fn values_len_for_dim(len: usize, dim: usize) -> usize {
    len.div_ceil(dim.max(1))
}

fn unified_decode_target(
    now: &Now,
    slot_vectors: &[Vec<f32>],
    slot_presence: &[f32],
) -> ExperienceDecodeOutput {
    let teacher_summary = slot_vectors
        .iter()
        .zip(slot_presence.iter().copied())
        .flat_map(|(vector, present)| {
            let mean_abs = if vector.is_empty() {
                0.0
            } else {
                vector.iter().map(|value| value.abs()).sum::<f32>() / vector.len() as f32
            };
            let max_abs = vector
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f32, f32::max);
            [present, mean_abs, max_abs]
        })
        .collect::<Vec<_>>();
    ExperienceDecodeOutput {
        body_features: compact_contact_features(now)
            .into_iter()
            .chain([now.body.battery_level, bool_feature(now.body.charging)])
            .map(clean_feature)
            .collect(),
        memory_features: teacher_summary,
        drive_features: slot_presence.to_vec(),
        prediction_features: vec![
            bool_feature(now.body.flags.bump_left || now.body.flags.bump_right),
            now.extensions
                .get("sim.stuck")
                .and_then(|value| {
                    value.as_f64().map(|value| value as f32).or_else(|| {
                        extension_vector_values(value).and_then(|values| values.first().copied())
                    })
                })
                .unwrap_or_default(),
            now.memory.place_novelty,
            bool_feature(now.reign.active),
            now.predictions.uncertainty,
        ],
        eye_features: slot_vectors.iter().flatten().copied().collect(),
        ear_features: compact_range_features(now)
            .into_iter()
            .chain(compact_depth_features(now))
            .map(clean_feature)
            .collect(),
    }
}

fn unified_modality_coverage(
    examples: &[UnifiedExperienceExample],
) -> Vec<UnifiedModalityCoverage> {
    UNIFIED_TEACHER_SLOTS
        .iter()
        .enumerate()
        .map(|(index, slot)| {
            let present_count = examples
                .iter()
                .filter(|example| {
                    example
                        .slot_presence
                        .get(index)
                        .copied()
                        .unwrap_or_default()
                        > 0.0
                })
                .count();
            let missing_count = examples.len().saturating_sub(present_count);
            UnifiedModalityCoverage {
                slot: slot.name.to_string(),
                source: slot.source.to_string(),
                purpose: slot.purpose.to_string(),
                dim: examples
                    .first()
                    .and_then(|example| example.input.sense_vectors.get(index))
                    .map(Vec::len)
                    .unwrap_or_default(),
                placeholder: slot.source.contains("placeholder"),
                present_count,
                missing_count,
                coverage: present_count as f32 / examples.len().max(1) as f32,
            }
        })
        .collect()
}

fn evaluate_unified_reconstruction(
    autoencoder: &ExperienceAutoencoderTrainer,
    examples: &[UnifiedExperienceExample],
) -> Result<UnifiedReconstructionReport> {
    let mut total_losses = Vec::new();
    let mut zero_losses = Vec::new();
    let mut head_losses: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut zero_head_losses: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for sample in examples {
        if sample.input.flat_features().len() != autoencoder.input_dim()
            || sample.target.feature_lengths() != autoencoder.decode_lengths()
        {
            continue;
        }
        let prediction = autoencoder.predict(&sample.input)?;
        let predicted = &prediction.decoded;
        let target = &sample.target;
        total_losses.push(mse_vec(&predicted.flat_features(), &target.flat_features()));
        zero_losses.push(mse_vec(
            &vec![0.0; target.flat_features().len()],
            &target.flat_features(),
        ));
        push_head_loss(
            &mut head_losses,
            "sensor_body",
            &predicted.body_features,
            &target.body_features,
        );
        push_head_loss(
            &mut head_losses,
            "teacher_summary",
            &predicted.memory_features,
            &target.memory_features,
        );
        push_head_loss(
            &mut head_losses,
            "modality_mask",
            &predicted.drive_features,
            &target.drive_features,
        );
        push_head_loss(
            &mut head_losses,
            "outcomes",
            &predicted.prediction_features,
            &target.prediction_features,
        );
        push_head_loss(
            &mut head_losses,
            "teacher_vectors",
            &predicted.eye_features,
            &target.eye_features,
        );
        push_head_loss(
            &mut head_losses,
            "range_depth",
            &predicted.ear_features,
            &target.ear_features,
        );
        push_head_loss(
            &mut zero_head_losses,
            "sensor_body",
            &[],
            &target.body_features,
        );
        push_head_loss(
            &mut zero_head_losses,
            "teacher_summary",
            &[],
            &target.memory_features,
        );
        push_head_loss(
            &mut zero_head_losses,
            "modality_mask",
            &[],
            &target.drive_features,
        );
        push_head_loss(
            &mut zero_head_losses,
            "outcomes",
            &[],
            &target.prediction_features,
        );
        push_head_loss(
            &mut zero_head_losses,
            "teacher_vectors",
            &[],
            &target.eye_features,
        );
        push_head_loss(
            &mut zero_head_losses,
            "range_depth",
            &[],
            &target.ear_features,
        );
    }
    if total_losses.is_empty() {
        bail!("no usable unified Experience reconstruction samples");
    }
    let total_loss_mean = mean(&total_losses);
    let zero_loss_mean = mean(&zero_losses);
    Ok(UnifiedReconstructionReport {
        sample_count: total_losses.len(),
        total_loss_mean,
        zero_loss_mean,
        head_losses: mean_loss_map(head_losses),
        zero_head_losses: mean_loss_map(zero_head_losses),
        reconstructive: total_loss_mean < zero_loss_mean,
    })
}

fn unified_learned_loop_report(
    autoencoder: &ExperienceAutoencoderTrainer,
    future: &FutureNetTrainer,
    examples: &[UnifiedExperienceExample],
    baselines: &UnifiedBaselineReport,
    random_projection_prediction_loss: f32,
    mechanical_instant_prediction_loss: f32,
) -> Result<UnifiedLearnedLoopReport> {
    let mut records = Vec::new();
    let mut reconstruction_losses = Vec::new();
    let mut prediction_losses = Vec::new();
    let mut combined_surprises = Vec::new();
    let mut confidences = Vec::new();

    for sample in examples {
        if sample.input.flat_features().len() != autoencoder.input_dim()
            || sample.next_input.flat_features().len() != autoencoder.input_dim()
            || sample.target.feature_lengths() != autoencoder.decode_lengths()
        {
            continue;
        }

        let prediction = autoencoder.predict(&sample.input)?;
        let next_z = autoencoder.encode(&sample.next_input)?.z;
        let reconstruction_loss = mse_vec(
            &prediction.decoded.flat_features(),
            &sample.target.flat_features(),
        );
        let input = FutureInput {
            latent: ExperienceLatent {
                t_ms: sample.t_ms,
                z: prediction.encoded.z.clone(),
                reconstruction_error: reconstruction_loss,
                prediction_error: 0.0,
                confidence: prediction.encoded.confidence,
            },
            action: sample.action.clone(),
            offset_ms: sample.offset_ms,
        };
        if input.flat_features().len() != future.input_dim() || next_z.len() != future.latent_dim()
        {
            continue;
        }
        let predicted = future.predict(&input)?;
        let prediction_loss = mse_vec(&predicted.predicted_z, &next_z);
        let copy_current_prediction_loss = mse_vec(&input.latent.z, &next_z);
        let reconstruction_norm = normalize_loss(
            reconstruction_loss,
            baselines
                .copy_current_loss_mean
                .unwrap_or(reconstruction_loss)
                .max(reconstruction_loss),
        );
        let prediction_norm = normalize_loss(prediction_loss, copy_current_prediction_loss);
        let combined_surprise = (0.4 * reconstruction_norm + 0.6 * prediction_norm).clamp(0.0, 1.0);
        let coverage =
            sample.slot_presence.iter().sum::<f32>() / sample.slot_presence.len().max(1) as f32;
        let confidence = ((prediction.encoded.confidence + predicted.confidence) * 0.5 * coverage)
            .clamp(0.0, 1.0);
        let surprise = ExperienceSurprise {
            t_ms: sample.t_ms,
            reconstruction_loss,
            prediction_loss,
            combined_surprise,
            confidence,
            reconstruction_weight: 0.4,
            prediction_weight: 0.6,
        };

        reconstruction_losses.push(reconstruction_loss);
        prediction_losses.push(prediction_loss);
        combined_surprises.push(combined_surprise);
        confidences.push(confidence);
        records.push(UnifiedExperienceLoopRecord {
            t_ms: sample.t_ms,
            offset_ms: sample.offset_ms,
            encoded_latent: input.latent.z,
            predicted_next_latent: predicted.predicted_z,
            actual_next_latent: next_z,
            reconstruction_loss,
            prediction_loss,
            combined_surprise,
            confidence,
            teacher_coverage: coverage,
            missing_modality_mask: sample
                .slot_presence
                .iter()
                .map(|present| if *present > 0.0 { 0.0 } else { 1.0 })
                .collect(),
            baseline_comparisons: UnifiedExperienceLoopBaselines {
                copy_current_prediction_loss,
                random_projection_prediction_loss: Some(random_projection_prediction_loss),
                mechanical_instant_prediction_loss: Some(mechanical_instant_prediction_loss),
            },
            surprise,
        });
    }

    if records.is_empty() {
        bail!("no usable unified Experience learned-loop records");
    }

    Ok(UnifiedLearnedLoopReport {
        canonical_instant: "ExperienceInstant".to_string(),
        canonical_latent: "ExperienceLatent".to_string(),
        prediction: "ExperiencePrediction".to_string(),
        surprise: "ExperienceSurprise".to_string(),
        sample_count: records.len(),
        reconstruction_loss_mean: mean(&reconstruction_losses),
        prediction_loss_mean: mean(&prediction_losses),
        combined_surprise_mean: mean(&combined_surprises),
        confidence_mean: mean(&confidences),
        records,
    })
}

fn normalize_loss(loss: f32, baseline: f32) -> f32 {
    if baseline.is_finite() && baseline > 1.0e-6 {
        (loss / baseline).clamp(0.0, 1.0)
    } else {
        loss.clamp(0.0, 1.0)
    }
}

fn push_head_loss(
    losses: &mut BTreeMap<String, Vec<f32>>,
    head: &str,
    predicted: &[f32],
    target: &[f32],
) {
    let zero;
    let predicted = if predicted.is_empty() && !target.is_empty() {
        zero = vec![0.0; target.len()];
        &zero
    } else {
        predicted
    };
    losses
        .entry(head.to_string())
        .or_default()
        .push(mse_vec(predicted, target));
}

fn mean_loss_map(losses: BTreeMap<String, Vec<f32>>) -> BTreeMap<String, f32> {
    losses
        .into_iter()
        .map(|(head, values)| (head, mean(&values)))
        .collect()
}

fn unified_trained_future_samples(
    autoencoder: &ExperienceAutoencoderTrainer,
    examples: &[UnifiedExperienceExample],
) -> Result<Vec<(TimeMs, FutureInput, Vec<f32>)>> {
    let mut samples = Vec::new();
    for sample in examples {
        if sample.input.flat_features().len() != autoencoder.input_dim()
            || sample.next_input.flat_features().len() != autoencoder.input_dim()
        {
            continue;
        }
        let before_z = autoencoder.encode(&sample.input)?.z;
        let after_z = autoencoder.encode(&sample.next_input)?.z;
        samples.push((
            sample.t_ms,
            FutureInput {
                latent: ExperienceLatent {
                    t_ms: sample.t_ms,
                    z: before_z,
                    reconstruction_error: 0.0,
                    prediction_error: 0.0,
                    confidence: 0.65,
                },
                action: sample.action.clone(),
                offset_ms: sample.offset_ms,
            },
            after_z,
        ));
    }
    Ok(samples)
}

fn unified_encoded_future_samples(
    encoder: &mut impl LatentEncoder,
    examples: &[UnifiedExperienceExample],
) -> Result<Vec<(TimeMs, FutureInput, Vec<f32>)>> {
    let mut samples = Vec::new();
    for sample in examples {
        let before_z = encoder.encode_input(&sample.input, sample.t_ms)?;
        let after_z = encoder.encode_input(
            &sample.next_input,
            sample.t_ms.saturating_add(sample.offset_ms),
        )?;
        samples.push((
            sample.t_ms,
            FutureInput {
                latent: before_z,
                action: sample.action.clone(),
                offset_ms: sample.offset_ms,
            },
            after_z.z,
        ));
    }
    Ok(samples)
}

fn unified_mechanical_future_samples(
    examples: &[UnifiedExperienceExample],
) -> Vec<(TimeMs, FutureInput, Vec<f32>)> {
    examples
        .iter()
        .map(|sample| {
            (
                sample.t_ms,
                FutureInput {
                    latent: ExperienceLatent {
                        t_ms: sample.t_ms,
                        z: sample.input.flat_features(),
                        reconstruction_error: 0.0,
                        prediction_error: 0.0,
                        confidence: 0.5,
                    },
                    action: sample.action.clone(),
                    offset_ms: sample.offset_ms,
                },
                sample.next_input.flat_features(),
            )
        })
        .collect()
}

fn unified_latent_variance(
    autoencoder: &ExperienceAutoencoderTrainer,
    examples: &[UnifiedExperienceExample],
) -> Result<f32> {
    let mut latents = Vec::new();
    for sample in examples {
        if sample.input.flat_features().len() == autoencoder.input_dim() {
            latents.push(autoencoder.encode(&sample.input)?.z);
        }
    }
    let Some(first) = latents.first() else {
        return Ok(0.0);
    };
    let dim = first.len();
    if dim == 0 {
        return Ok(0.0);
    }
    let mut means = vec![0.0; dim];
    for latent in &latents {
        for (mean, value) in means.iter_mut().zip(latent.iter().copied()) {
            *mean += value;
        }
    }
    for mean in &mut means {
        *mean /= latents.len().max(1) as f32;
    }
    let mut variance = 0.0;
    for latent in &latents {
        for (index, value) in latent.iter().copied().enumerate() {
            variance += (value - means[index]).powi(2);
        }
    }
    Ok(variance / (latents.len().max(1) * dim).max(1) as f32)
}

fn latent_architecture_report(
    source_kind: &str,
    encoder_name: &str,
    checkpoint_path: &Path,
    predict_checkpoint_path: &Path,
    decode_target_kind: &str,
    samples: &[(&ExperienceEncodeInput, &ExperienceDecodeOutput)],
    z_dim: usize,
) -> Result<LatentRoundTripArchitectureReport> {
    let (input, target) = samples
        .first()
        .ok_or_else(|| anyhow!("no samples available for latent architecture report"))?;
    let input_dim = input.flat_features().len();
    let decode_target_dim = target.flat_features().len();
    let teacher_vectors = input
        .sense_vectors
        .iter()
        .enumerate()
        .map(|(index, vector)| {
            teacher_vector_report(source_kind, index, vector.len(), samples.len())
        })
        .collect::<Vec<_>>();

    Ok(LatentRoundTripArchitectureReport {
        pipeline: vec![
            "teacher_vectors".to_string(),
            "mechanically_assembled_instant".to_string(),
            "experience_encoder".to_string(),
            "experience_latent".to_string(),
            "decode_predict_compare".to_string(),
        ],
        teacher_vectors,
        instant: MechanicalInstantReport {
            representation: "ExperienceInstant".to_string(),
            assembly:
                "assemble deterministic modality teacher vectors, masks, action, time, and compact sensor summaries; keep reconstruction targets as separate supervision"
                    .to_string(),
            sample_count: samples.len(),
            input_dim,
            decode_target_dim,
            decode_target_kind: decode_target_kind.to_string(),
        },
        encoder: ExperienceEncoderReport {
            name: encoder_name.to_string(),
            input_dim,
            z_dim,
            checkpoint_path: checkpoint_path.to_path_buf(),
        },
        owned_latent: OwnedExperienceLatentReport {
            name: "ExperienceLatent".to_string(),
            owner: "Pete".to_string(),
            dim: z_dim,
            teacher_independent: true,
            evidence: vec![
                "decoder reconstructs compact sensor summaries from z".to_string(),
                "future predictor consumes z to predict the next ExperienceLatent".to_string(),
                "comparison report measures z against copy-current and research-only random-projection baselines".to_string(),
            ],
        },
        heads: vec![
            LatentHeadReport {
                name: "decode".to_string(),
                target: decode_target_kind.to_string(),
                checkpoint_path: Some(checkpoint_path.to_path_buf()),
            },
            LatentHeadReport {
                name: "predict".to_string(),
                target: "next ExperienceLatent".to_string(),
                checkpoint_path: Some(predict_checkpoint_path.to_path_buf()),
            },
            LatentHeadReport {
                name: "compare".to_string(),
                target: "copy-current and research-only random-projection baselines".to_string(),
                checkpoint_path: None,
            },
        ],
    })
}

fn teacher_vector_report(
    source_kind: &str,
    index: usize,
    dim: usize,
    sample_count: usize,
) -> TeacherVectorReport {
    let (name, source, purpose) = match (source_kind, index) {
        (_, 0) => (
            "teacher.now_sense_vectors",
            "ledger Now vector assembly",
            "teacher/fallback present-moment vector",
        ),
        (_, _) => (
            "teacher.aux_sense_vector",
            "ledger Now vector assembly",
            "auxiliary instant feature vector",
        ),
    };
    TeacherVectorReport {
        name: name.to_string(),
        source: source.to_string(),
        purpose: purpose.to_string(),
        dim,
        sample_count,
    }
}

fn compact_contact_features(now: &Now) -> Vec<f32> {
    vec![
        bool_feature(now.body.flags.bump_left),
        bool_feature(now.body.flags.bump_right),
        bool_feature(now.body.flags.bump_left || now.body.flags.bump_right),
        bool_feature(
            now.body.flags.cliff_left
                || now.body.flags.cliff_front_left
                || now.body.flags.cliff_front_right
                || now.body.flags.cliff_right
                || now.body.cliff_sensors.left > 0.5
                || now.body.cliff_sensors.front_left > 0.5
                || now.body.cliff_sensors.front_right > 0.5
                || now.body.cliff_sensors.right > 0.5,
        ),
        bool_feature(now.body.flags.wheel_drop),
        bool_feature(now.body.flags.wall),
        bool_feature(now.body.flags.virtual_wall),
        now.extensions
            .get("sim.stuck")
            .and_then(|value| value.as_f64())
            .unwrap_or(0.0)
            .clamp(0.0, 1.0) as f32,
    ]
}

fn compact_range_features(now: &Now) -> Vec<f32> {
    let beams = now
        .range
        .beams
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let nearest = now
        .range
        .nearest_m
        .filter(|value| value.is_finite())
        .or_else(|| beams.iter().copied().reduce(f32::min));
    let mean = mean(&beams);
    let len = beams.len().max(1);
    let third = len / 3;
    vec![
        nearest.map(inverse_distance_feature).unwrap_or_default(),
        (beams.len() as f32 / 128.0).clamp(0.0, 1.0),
        inverse_distance_feature(mean),
        inverse_distance_feature(window_mean_feature(&beams, 0, third.max(1))),
        inverse_distance_feature(window_mean_feature(
            &beams,
            third,
            len.saturating_sub(third * 2).max(1),
        )),
        inverse_distance_feature(window_mean_feature(
            &beams,
            len.saturating_sub(third.max(1)),
            third.max(1),
        )),
    ]
}

fn compact_depth_features(now: &Now) -> Vec<f32> {
    let depths = now
        .kinect
        .depth_m
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let nonzero = depths.iter().filter(|value| **value > 0.01).count();
    let min = depths.iter().copied().reduce(f32::min).unwrap_or_default();
    let max = depths.iter().copied().reduce(f32::max).unwrap_or_default();
    let avg = mean(&depths);
    vec![
        inverse_distance_feature(min),
        inverse_distance_feature(avg),
        inverse_distance_feature(max),
        nonzero as f32 / depths.len().max(1) as f32,
        (now.kinect.depth_width as f32 / 640.0).clamp(0.0, 1.0),
        (now.kinect.depth_height as f32 / 480.0).clamp(0.0, 1.0),
        now.kinect.audio_confidence.clamp(0.0, 1.0),
        now.kinect.audio_angle_rad.unwrap_or_default().sin(),
        now.kinect.audio_angle_rad.unwrap_or_default().cos(),
    ]
}

fn split_samples<T>(mut samples: Vec<T>, validation_split: f32, seed: u64) -> (Vec<T>, Vec<T>) {
    let validation_split = validation_split.clamp(0.0, 0.9);
    let mut rng = StdRng::seed_from_u64(seed);
    samples.shuffle(&mut rng);
    let eval_len = ((samples.len() as f32) * validation_split).round() as usize;
    let eval_len = eval_len.min(samples.len().saturating_sub(1));
    let eval = samples.split_off(samples.len().saturating_sub(eval_len));
    (samples, eval)
}

fn bool_feature(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

fn clean_feature(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-10.0, 10.0)
    } else {
        0.0
    }
}

fn inverse_distance_feature(value: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        (1.0 / (1.0 + value)).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn window_mean_feature(values: &[f32], start: usize, len: usize) -> f32 {
    if values.is_empty() || len == 0 {
        return 0.0;
    }
    let end = start.saturating_add(len).min(values.len());
    if start >= end {
        return 0.0;
    }
    mean(&values[start..end])
}

fn trained_latent_future_samples(
    autoencoder: &ExperienceAutoencoderTrainer,
    transitions: &[ExperienceTransition],
) -> Result<Vec<(TimeMs, FutureInput, Vec<f32>)>> {
    let mut samples = Vec::new();
    for transition in transitions {
        let Some(action) = transition.action.clone() else {
            continue;
        };
        let before_input = experience_encode_input_from_now(&transition.before);
        let after_input = experience_encode_input_from_now(&transition.after);
        if before_input.flat_features().len() != autoencoder.input_dim()
            || after_input.flat_features().len() != autoencoder.input_dim()
        {
            continue;
        }
        let before_z = autoencoder.encode(&before_input)?.z;
        let after_z = autoencoder.encode(&after_input)?.z;
        samples.push((
            transition.created_at_ms,
            FutureInput {
                latent: ExperienceLatent {
                    t_ms: transition.before.t_ms,
                    z: before_z,
                    reconstruction_error: 0.0,
                    prediction_error: 0.0,
                    confidence: 0.6,
                },
                action,
                offset_ms: transition
                    .after
                    .t_ms
                    .saturating_sub(transition.before.t_ms)
                    .max(1),
            },
            after_z,
        ));
    }
    Ok(samples)
}

fn encoded_future_samples(
    encoder: &mut impl LatentEncoder,
    transitions: &[ExperienceTransition],
) -> Result<Vec<(TimeMs, FutureInput, Vec<f32>)>> {
    let mut samples = Vec::new();
    for transition in transitions {
        let Some(action) = transition.action.clone() else {
            continue;
        };
        let before_input = experience_encode_input_from_now(&transition.before);
        let after_input = experience_encode_input_from_now(&transition.after);
        let before_z = encoder.encode_input(&before_input, transition.before.t_ms)?;
        let after_z = encoder.encode_input(&after_input, transition.after.t_ms)?;
        samples.push((
            transition.created_at_ms,
            FutureInput {
                latent: before_z,
                action,
                offset_ms: transition
                    .after
                    .t_ms
                    .saturating_sub(transition.before.t_ms)
                    .max(1),
            },
            after_z.z,
        ));
    }
    Ok(samples)
}

fn evaluate_trained_reconstruction(
    autoencoder: &ExperienceAutoencoderTrainer,
    transitions: &[ExperienceTransition],
) -> Result<LatentReconstructionReport> {
    let mut losses = Vec::new();
    let mut zero_losses = Vec::new();
    for (_, input, target, _) in experience_samples(transitions) {
        if input.flat_features().len() != autoencoder.input_dim()
            || target.feature_lengths() != autoencoder.decode_lengths()
        {
            continue;
        }
        let prediction = autoencoder.predict(&input)?;
        let target_features = target.flat_features();
        losses.push(mse_vec(
            &prediction.decoded.flat_features(),
            &target_features,
        ));
        zero_losses.push(mse_vec(&vec![0.0; target_features.len()], &target_features));
    }
    if losses.is_empty() {
        bail!("no usable reconstruction samples for latent round-trip evaluation");
    }
    Ok(LatentReconstructionReport {
        sample_count: losses.len(),
        trained_decoder_loss_mean: mean(&losses),
        zero_decoder_loss_mean: mean(&zero_losses),
        target_kind: "compact body/memory/drive/prediction/range-depth/audio-summary features"
            .to_string(),
    })
}

fn train_and_evaluate_future_latents(
    encoder: &str,
    train_samples: Vec<(TimeMs, FutureInput, Vec<f32>)>,
    eval_samples: Vec<(TimeMs, FutureInput, Vec<f32>)>,
    epochs: usize,
    checkpoint_path: &Path,
    target_kind: &str,
) -> Result<LatentPredictorReport> {
    let input_dim = first_dim(&train_samples, |(_, input, _)| input.flat_features().len())?;
    let latent_dim = first_dim(&train_samples, |(_, input, _)| input.latent.z.len())?;
    let target_dim = first_dim(&train_samples, |(_, _, target)| target.len())?;
    let mut trainer = FutureNetTrainer::new(input_dim, target_dim);
    for _epoch in 0..epochs {
        for (_, input, target) in &train_samples {
            if input.flat_features().len() == trainer.input_dim()
                && target.len() == trainer.latent_dim()
            {
                trainer.train_step(input, target)?;
            }
        }
    }
    trainer.save_checkpoint(checkpoint_path)?;

    let mut stasis = StasisFuturePredictor;
    let mut model_losses = Vec::new();
    let mut stasis_losses = Vec::new();
    for (_, input, target) in eval_samples.iter().filter(|(_, input, target)| {
        input.flat_features().len() == trainer.input_dim() && target.len() == trainer.latent_dim()
    }) {
        let model = trainer.predict(input)?;
        let hard = stasis.predict(&input.latent, &input.action, input.offset_ms)?;
        model_losses.push(mse_vec(&model.predicted_z, target));
        stasis_losses.push(mse_vec(&hard.predicted_z, target));
    }
    if model_losses.is_empty() {
        bail!("no usable future samples for {encoder} latent evaluation");
    }
    let model_loss_mean = mean(&model_losses);
    let stasis_loss_mean = mean(&stasis_losses);
    let improvement_ratio =
        (stasis_loss_mean > 0.0).then(|| (stasis_loss_mean - model_loss_mean) / stasis_loss_mean);
    Ok(LatentPredictorReport {
        encoder: encoder.to_string(),
        target_kind: target_kind.to_string(),
        train_sample_count: train_samples.len(),
        eval_sample_count: model_losses.len(),
        latent_dim,
        target_dim,
        model_loss_mean,
        stasis_loss_mean,
        improvement_ratio,
        predictive: model_loss_mean < stasis_loss_mean,
    })
}

fn latent_baseline_comparisons(
    predictors: &[LatentPredictorReport],
    trained_encoder: &str,
    random_encoder: &str,
    evolved_encoder: Option<&str>,
) -> LatentBaselineComparisons {
    let trained = predictors
        .iter()
        .find(|report| report.encoder == trained_encoder);
    let random = predictors
        .iter()
        .find(|report| report.encoder == random_encoder);
    let evolved = evolved_encoder
        .and_then(|encoder| predictors.iter().find(|report| report.encoder == encoder));

    let trained_loss = trained.map(|report| report.model_loss_mean);
    let copy_loss = trained.map(|report| report.stasis_loss_mean);
    let random_loss = random.map(|report| report.model_loss_mean);
    let evolved_loss = evolved.map(|report| report.model_loss_mean);

    LatentBaselineComparisons {
        trained_encoder: trained_encoder.to_string(),
        copy_current_loss_mean: copy_loss,
        random_projection_loss_mean: random_loss,
        evolved_vector_loss_mean: evolved_loss,
        trained_loss_mean: trained_loss,
        trained_beats_copy_current: trained_loss
            .zip(copy_loss)
            .is_some_and(|(trained, baseline)| trained < baseline),
        trained_beats_random_projection: trained_loss
            .zip(random_loss)
            .is_some_and(|(trained, baseline)| trained < baseline),
        trained_beats_evolved_vector: trained_loss
            .zip(evolved_loss)
            .is_some_and(|(trained, baseline)| trained < baseline),
    }
}
