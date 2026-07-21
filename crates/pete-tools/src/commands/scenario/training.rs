async fn run_dream_train(args: DreamTrainArgs) -> Result<()> {
    let checkpoint_dir = PathBuf::from(&args.checkpoint_dir);
    if args.clear && checkpoint_dir.exists() {
        fs::remove_dir_all(&checkpoint_dir).with_context(|| {
            format!(
                "failed to clear checkpoint dir {}",
                checkpoint_dir.display()
            )
        })?;
        println!(
            "cleared dream checkpoint dir for fresh evolve run: {}",
            checkpoint_dir.display()
        );
    }

    let evolve_best = checkpoint_dir.join("evolve-best.json");
    let incumbent = if !args.clear && evolve_best.exists() {
        Some(load_best_genome(&evolve_best).with_context(|| {
            format!(
                "failed to load incumbent evolve checkpoint {}",
                evolve_best.display()
            )
        })?)
    } else {
        None
    };

    let config = DreamTrainingConfig {
        population_size: args.population,
        generations: args.generations,
        base_seed: args.seed,
        start_level: args.start_level.into(),
        hidden_dim: args.hidden_dim,
        checkpoint_dir: checkpoint_dir.clone(),
        dataset_dir: PathBuf::from(args.dataset_dir),
        export_dataset: args.export_dataset,
        detailed_logs: args.detailed_logs,
    };
    let report = train_dream_policy(config).await?;

    let candidate = load_best_genome(&report.best_checkpoint).with_context(|| {
        format!(
            "failed to load candidate checkpoint {}",
            report.best_checkpoint.display()
        )
    })?;
    let promote = incumbent.as_ref().map_or(true, |current| {
        if candidate.level.id() != current.level.id() {
            candidate.level.id() > current.level.id()
        } else {
            candidate.best_score > current.best_score
        }
    });

    if promote {
        fs::copy(&report.best_checkpoint, &evolve_best).with_context(|| {
            format!(
                "failed to publish evolve checkpoint alias from {} to {}",
                report.best_checkpoint.display(),
                evolve_best.display()
            )
        })?;
        println!(
            "published evolve checkpoint alias: {}",
            evolve_best.display()
        );
    } else if let Some(current) = &incumbent {
        println!(
            "kept incumbent evolve checkpoint: {} (incumbent level={} score={:.3}, candidate level={} score={:.3})",
            evolve_best.display(),
            current.level.name(),
            current.best_score,
            candidate.level.name(),
            candidate.best_score,
        );
    }

    fn comma_count(value: u64) -> String {
        let digits = value.to_string();
        let mut out = String::with_capacity(digits.len() + (digits.len().saturating_sub(1) / 3));
        let mut since_comma = 0usize;
        for ch in digits.chars().rev() {
            if since_comma == 3 {
                out.push(',');
                since_comma = 0;
            }
            out.push(ch);
            since_comma += 1;
        }
        out.chars().rev().collect()
    }

    let unlocked = report
        .unlocked_levels
        .iter()
        .map(|level| level.name())
        .collect::<Vec<_>>()
        .join(" -> ");
    println!(
        "dream policy training complete: level {}, generation {}, best score {:.3}, genome {}, checkpoint {}, dataset {}, unlocked {}",
        report.status.current_level.name(),
        comma_count(report.status.generation as u64),
        report.status.best_score,
        comma_count(report.status.selected_genome_id),
        report.best_checkpoint.display(),
        report.dataset_dir.display(),
        unlocked,
    );
    if let Some(reason) = report.status.blocked_reason {
        println!("last safety block: {reason}");
    }
    Ok(())
}

fn scenario_recommendation(episodes: usize, summary: &ScenarioEvaluationSummary) -> String {
    if episodes < 3 {
        "insufficient_data".to_string()
    } else if summary.collision_rate > 0.10 || summary.mean_collisions_per_episode > 5.0 {
        "reject_or_continue_training".to_string()
    } else if summary.success_rate >= 0.80 && summary.collision_rate <= 0.02 {
        "candidate_for_more_eval".to_string()
    } else {
        "continue_training".to_string()
    }
}

fn nearest_object_distance<F>(
    position: (f32, f32),
    objects: &[pete_sim::SimObject],
    matches_kind: F,
) -> Option<f32>
where
    F: Fn(&pete_sim::SimObjectKind) -> bool,
{
    objects
        .iter()
        .filter(|object| matches_kind(&object.kind))
        .map(|object| {
            (distance_between(position, (object.x_m, object.y_m)) - object.radius_m).max(0.0)
        })
        .min_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
}

fn nearest_object_bearing<F>(
    position: (f32, f32),
    heading_rad: f32,
    objects: &[pete_sim::SimObject],
    matches_kind: F,
) -> Option<f32>
where
    F: Fn(&pete_sim::SimObjectKind) -> bool,
{
    objects
        .iter()
        .filter(|object| matches_kind(&object.kind))
        .min_by(|left, right| {
            let left_distance = distance_between(position, (left.x_m, left.y_m));
            let right_distance = distance_between(position, (right.x_m, right.y_m));
            left_distance
                .partial_cmp(&right_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|object| {
            let dx = object.x_m - position.0;
            let dy = object.y_m - position.1;
            (dy.atan2(dx) - heading_rad + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
                - std::f32::consts::PI
        })
}

fn distance_between(left: (f32, f32), right: (f32, f32)) -> f32 {
    let dx = left.0 - right.0;
    let dy = left.1 - right.1;
    ((dx * dx) + (dy * dy)).sqrt()
}

fn hit_rate(hits: usize, opportunities: usize) -> Option<f32> {
    (opportunities > 0).then_some(hits.min(opportunities) as f32 / opportunities as f32)
}

fn aggregate_hit_rate(pairs: impl Iterator<Item = (usize, usize)>) -> Option<f32> {
    let (hits, opportunities) = pairs.fold((0usize, 0usize), |acc, pair| {
        (acc.0.saturating_add(pair.0), acc.1.saturating_add(pair.1))
    });
    hit_rate(hits, opportunities)
}

fn sim_world_score(snapshot: &WorldSnapshot, index: usize) -> f32 {
    snapshot
        .extensions
        .iter()
        .find(|extension| extension.name == "sim.world")
        .and_then(|extension| extension.values.get(index).copied())
        .unwrap_or(0.0)
}

fn mean(values: impl Iterator<Item = f32>) -> f32 {
    let mut count = 0usize;
    let mut sum = 0.0;
    for value in values {
        count = count.saturating_add(1);
        sum += value;
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

fn mean_optional(values: impl Iterator<Item = f32>) -> Option<f32> {
    let mut count = 0usize;
    let mut sum = 0.0;
    for value in values {
        count = count.saturating_add(1);
        sum += value;
    }
    (count > 0).then_some(sum / count as f32)
}
