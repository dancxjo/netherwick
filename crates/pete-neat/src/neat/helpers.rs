fn range_sectors(range: &RangeSense) -> (f32, f32, f32) {
    let mut left = MAX_RANGE_M;
    let mut front = MAX_RANGE_M;
    let mut right = MAX_RANGE_M;
    let count = range.beams.len();
    for (index, distance) in range.beams.iter().copied().enumerate() {
        if !distance.is_finite() || distance <= 0.0 {
            continue;
        }
        let angle = range
            .beam_angles_rad
            .get(index)
            .copied()
            .unwrap_or_else(|| {
                let ratio = if count <= 1 {
                    0.5
                } else {
                    index as f32 / (count - 1) as f32
                };
                -std::f32::consts::FRAC_PI_2 + ratio * std::f32::consts::PI
            });
        if angle > std::f32::consts::FRAC_PI_6 {
            left = left.min(distance);
        } else if angle < -std::f32::consts::FRAC_PI_6 {
            right = right.min(distance);
        } else {
            front = front.min(distance);
        }
    }
    if range.beams.is_empty() {
        if let Some(nearest) = range.nearest_m.filter(|value| value.is_finite()) {
            front = front.min(nearest.max(0.0));
        }
    }
    (left, front, right)
}

fn pose_delta(last: Pose2, current: Pose2) -> (f32, f32) {
    let dx = current.x_m - last.x_m;
    let dy = current.y_m - last.y_m;
    let mut distance = dx.hypot(dy);
    if dy.abs() < 1.0e-6 && dx.abs() > 0.0 {
        // Real Create odometry currently stores cumulative distance in x.
        distance = dx;
    }
    (distance, wrap_angle(current.heading_rad - last.heading_rad))
}

fn wrap_angle(mut angle: f32) -> f32 {
    while angle > std::f32::consts::PI {
        angle -= std::f32::consts::TAU;
    }
    while angle < -std::f32::consts::PI {
        angle += std::f32::consts::TAU;
    }
    angle
}

fn prefer_measured(measured: f32, derived: f32) -> f32 {
    if measured.is_finite() && measured.abs() > 1.0e-5 {
        measured
    } else {
        finite_or_zero(derived)
    }
}

fn best_index(indices: &[usize], fitness: &[f32]) -> usize {
    indices
        .iter()
        .copied()
        .max_by(|left, right| fitness[*left].total_cmp(&fitness[*right]))
        .unwrap_or(0)
}

fn allocate_offspring(scores: &[f32], population_size: usize) -> Vec<usize> {
    if scores.is_empty() || population_size == 0 {
        return Vec::new();
    }
    let survivor_count = scores.len().min(population_size);
    let mut counts = vec![1usize; survivor_count];
    let remaining = population_size.saturating_sub(survivor_count);
    if remaining == 0 {
        return counts;
    }
    let usable_scores = scores
        .iter()
        .take(survivor_count)
        .map(|score| finite_or_zero(*score))
        .collect::<Vec<_>>();
    let min = usable_scores.iter().copied().fold(f32::INFINITY, f32::min);
    let shift = if min <= 0.0 { -min + 1.0e-3 } else { 0.0 };
    let total = usable_scores.iter().map(|score| score + shift).sum::<f32>();
    if !total.is_finite() || total <= 0.0 {
        for offset in 0..remaining {
            counts[offset % survivor_count] += 1;
        }
        return counts;
    }
    let mut fractional = Vec::with_capacity(survivor_count);
    let mut assigned = 0usize;
    for (index, score) in usable_scores.iter().enumerate() {
        let quota = ((*score + shift) / total) * remaining as f32;
        let whole = quota.floor() as usize;
        counts[index] += whole;
        assigned += whole;
        fractional.push((index, quota - whole as f32));
    }
    fractional.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    for (index, _) in fractional
        .into_iter()
        .cycle()
        .take(remaining.saturating_sub(assigned))
    {
        counts[index] += 1;
    }
    counts
}

fn select_parent_from_indices<R: Rng + ?Sized>(
    indices: &[usize],
    fitness: &[f32],
    rng: &mut R,
) -> usize {
    if indices.is_empty() {
        return 0;
    }
    let min = indices
        .iter()
        .map(|index| finite_or_zero(fitness[*index]))
        .fold(f32::INFINITY, f32::min);
    let shift = if min <= 0.0 { -min + 1.0e-3 } else { 0.0 };
    let total = indices
        .iter()
        .map(|index| finite_or_zero(fitness[*index]) + shift)
        .sum::<f32>();
    if !total.is_finite() || total <= 0.0 {
        return indices[rng.gen_range(0..indices.len())];
    }
    let mut needle = rng.gen_range(0.0..total);
    for index in indices {
        needle -= finite_or_zero(fitness[*index]) + shift;
        if needle <= 0.0 {
            return *index;
        }
    }
    *indices.last().unwrap_or(&0)
}

fn constraint_violation_score(traits: FitnessTraits, constraints: SelectionConstraints) -> f32 {
    let safety_excess = traits
        .safety_invariant_violations
        .saturating_sub(constraints.maximum_safety_invariant_violations)
        as f32;
    if safety_excess > 0.0 {
        return 10_000.0 + safety_excess;
    }
    let veto_excess = (traits.safety_veto_rate - constraints.maximum_safety_veto_rate).max(0.0);
    if veto_excess > 0.0 {
        return 5_000.0 + veto_excess * 1_000.0;
    }
    let collision_excess = (traits.collision_rate - constraints.maximum_collision_rate).max(0.0);
    if collision_excess > 0.0 {
        return 1_000.0 + collision_excess * 1_000.0;
    }
    let escape_deficit = (constraints.minimum_escape_rate - traits.escape_rate).max(0.0);
    if escape_deficit > 0.0 {
        return 100.0 + escape_deficit * 100.0;
    }
    0.0
}

fn infeasible_tiebreak(traits: FitnessTraits) -> f32 {
    let exploration = (traits.exploration / 28.0).clamp(0.0, 1.0);
    let progress = (traits.forward_progress / 20.0).clamp(0.0, 1.0);
    let low_repetition = (1.0 - traits.repetition_rate).clamp(0.0, 1.0);
    let low_veto = (1.0 - traits.safety_veto_rate).clamp(0.0, 1.0);
    (0.40 * traits.escape_rate.clamp(0.0, 1.0)
        + 0.20 * exploration
        + 0.15 * progress
        + 0.15 * low_repetition
        + 0.10 * low_veto)
        .clamp(0.0, 0.99)
}

fn pareto_fronts(traits: &[FitnessTraits], candidates: &[usize]) -> Vec<Vec<usize>> {
    let mut remaining = candidates.to_vec();
    let mut fronts = Vec::new();
    while !remaining.is_empty() {
        let mut front = Vec::new();
        for candidate in remaining.iter().copied() {
            let dominated = remaining
                .iter()
                .copied()
                .any(|other| other != candidate && dominates(traits[other], traits[candidate]));
            if !dominated {
                front.push(candidate);
            }
        }
        if front.is_empty() {
            fronts.push(remaining);
            break;
        }
        remaining.retain(|index| !front.contains(index));
        fronts.push(front);
    }
    fronts
}

fn dominates(left: FitnessTraits, right: FitnessTraits) -> bool {
    let mut strictly_better = false;
    for objective in SELECTION_OBJECTIVES {
        let left_value = finite_or_zero(left.objective(objective));
        let right_value = finite_or_zero(right.objective(objective));
        if left_value < right_value {
            return false;
        }
        if left_value > right_value {
            strictly_better = true;
        }
    }
    strictly_better
}

fn crowding_distances(traits: &[FitnessTraits], front: &[usize]) -> Vec<f32> {
    let mut distances = vec![0.0; front.len()];
    if front.len() <= 2 {
        distances.fill(f32::INFINITY);
        return distances;
    }

    for objective in SELECTION_OBJECTIVES {
        let mut ordered = front
            .iter()
            .copied()
            .enumerate()
            .map(|(offset, index)| (offset, finite_or_zero(traits[index].objective(objective))))
            .collect::<Vec<_>>();
        ordered.sort_by(|left, right| left.1.total_cmp(&right.1));
        let first = ordered.first().copied().unwrap();
        let last = ordered.last().copied().unwrap();
        distances[first.0] = f32::INFINITY;
        distances[last.0] = f32::INFINITY;
        let span = last.1 - first.1;
        if span.abs() <= f32::EPSILON {
            continue;
        }
        for window in ordered.windows(3) {
            let previous = window[0].1;
            let current_offset = window[1].0;
            let next = window[2].1;
            if distances[current_offset].is_finite() {
                distances[current_offset] += (next - previous).abs() / span.abs();
            }
        }
    }

    distances
}

fn normalized_crowding_bonus(crowding_distance: f32) -> f32 {
    if crowding_distance.is_finite() {
        crowding_distance.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

fn checkpoint_path(path: &Path) -> std::path::PathBuf {
    if path.extension().is_some() {
        path.to_path_buf()
    } else {
        path.join("locomotion-neat.json")
    }
}

fn normalize_clearance(value: f32) -> f32 {
    if value.is_finite() {
        (value / MAX_RANGE_M).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

fn unit(value: f32) -> f32 {
    finite_or_zero(value).clamp(0.0, 1.0)
}

fn bool_unit(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

fn bin(value: f32, thresholds: &[f32]) -> u8 {
    let value = finite_or_zero(value);
    thresholds
        .iter()
        .position(|threshold| value < *threshold)
        .unwrap_or(thresholds.len()) as u8
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}
