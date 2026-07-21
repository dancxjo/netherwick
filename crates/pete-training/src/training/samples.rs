fn danger_samples(
    transitions: &[ExperienceTransition],
) -> Vec<(TimeMs, Now, DangerInput, DangerTarget)> {
    transitions
        .iter()
        .filter(|transition| !transition.before_z.z.is_empty() && !transition.after_z.z.is_empty())
        .map(|transition| {
            (
                transition.created_at_ms,
                transition.before.clone(),
                danger_input_from_transition_like(
                    &transition.before_z,
                    transition.action.as_ref(),
                    &transition.before,
                ),
                danger_target_from_transition_like(
                    &transition.before,
                    transition.action.as_ref(),
                    &transition.after,
                ),
            )
        })
        .collect()
}

fn charge_samples(
    transitions: &[ExperienceTransition],
) -> Vec<(TimeMs, Now, ChargeInput, ChargeTarget)> {
    transitions
        .iter()
        .filter(|transition| !transition.before_z.z.is_empty() && !transition.after_z.z.is_empty())
        .map(|transition| {
            (
                transition.created_at_ms,
                transition.before.clone(),
                charge_input_from_transition_like(
                    &transition.before_z,
                    transition.action.as_ref(),
                    &transition.before,
                ),
                charge_target_from_transition_like(
                    &transition.before,
                    transition.action.as_ref(),
                    &transition.after,
                ),
            )
        })
        .collect()
}

fn action_value_samples(
    transitions: &[ExperienceTransition],
) -> Vec<(TimeMs, Now, ActionValueInput, ActionValueTarget)> {
    transitions
        .iter()
        .filter(|transition| !transition.before_z.z.is_empty() && !transition.after_z.z.is_empty())
        .map(|transition| {
            (
                transition.created_at_ms,
                transition.before.clone(),
                action_value_input_from_transition_like(
                    &transition.before_z,
                    transition.action.as_ref(),
                    &transition.before,
                ),
                action_value_target_from_reward_surprise(&transition.reward, &transition.surprise),
            )
        })
        .collect()
}

fn future_samples(transitions: &[ExperienceTransition]) -> Vec<(TimeMs, FutureInput, Vec<f32>)> {
    transitions
        .iter()
        .filter_map(|transition| {
            let input = future_input_from_transition(transition, 1_000)?;
            let target = future_target_from_transition(transition);
            (!target.is_empty()).then_some((transition.created_at_ms, input, target))
        })
        .collect()
}

fn eye_next_samples(
    transitions: &[ExperienceTransition],
) -> Vec<(TimeMs, Now, EyeNextInput, EyeNextTarget)> {
    transitions
        .iter()
        .filter_map(|transition| {
            let target = eye_next_target_from_now(&transition.after)?;
            let input = eye_next_input_from_transition_like(
                &transition.before_z,
                transition.action.as_ref(),
                &transition.before,
                100,
            );
            Some((
                transition.created_at_ms,
                transition.before.clone(),
                input,
                target,
            ))
        })
        .collect()
}

fn ear_next_samples(
    transitions: &[ExperienceTransition],
) -> Vec<(TimeMs, Now, EarNextInput, EarNextTarget)> {
    transitions
        .iter()
        .filter_map(|transition| {
            let target = ear_next_target_from_now(&transition.after)?;
            let input = ear_next_input_from_transition_like(
                &transition.before_z,
                transition.action.as_ref(),
                &transition.before,
                100,
            );
            Some((
                transition.created_at_ms,
                transition.before.clone(),
                input,
                target,
            ))
        })
        .collect()
}

fn experience_samples(
    transitions: &[ExperienceTransition],
) -> Vec<(
    TimeMs,
    ExperienceEncodeInput,
    ExperienceDecodeOutput,
    Vec<f32>,
)> {
    let mut samples = Vec::new();
    for transition in transitions {
        for (t_ms, now, baseline_z) in [
            (
                transition.created_at_ms,
                &transition.before,
                transition.before_z.z.clone(),
            ),
            (
                transition.created_at_ms,
                &transition.after,
                transition.after_z.z.clone(),
            ),
        ] {
            let input = experience_encode_input_from_now(now);
            let target = experience_decode_target_from_now(now);
            if input.flat_features().is_empty() || target.flat_features().is_empty() {
                continue;
            }
            samples.push((t_ms, input, target, baseline_z));
        }
    }
    samples
}

fn first_dim<T>(samples: &[T], f: impl Fn(&T) -> usize) -> Result<usize> {
    samples
        .first()
        .map(f)
        .filter(|dim| *dim > 0)
        .ok_or_else(|| anyhow!("no usable samples"))
}

fn mse<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f32>()
        / N.max(1) as f32
}

fn mse_vec(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    a.iter()
        .take(len)
        .zip(b.iter().take(len))
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f32>()
        / len as f32
}

fn mse_bytes(a: &[u8], b: &[u8]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    a.iter()
        .take(len)
        .zip(b.iter().take(len))
        .map(|(left, right)| ((*left as f32 / 255.0) - (*right as f32 / 255.0)).powi(2))
        .sum::<f32>()
        / len as f32
}

fn eye_current_loss(now: &Now, target: &EyeNextTarget) -> Option<f32> {
    eye_next_target_from_now(now).map(|current| mse_bytes(&current.rgb, &target.rgb))
}

fn ear_current_loss(now: &Now, target: &EarNextTarget) -> Option<f32> {
    ear_next_target_from_now(now).map(|current| mse_vec(&current.features, &target.features))
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len().max(1) as f32
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
