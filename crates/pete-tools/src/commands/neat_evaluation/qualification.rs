async fn qualify_neat_stage_archive(
    archive: &mut [NeatNicheArchiveEntry],
    episodes: usize,
    steps: usize,
    seed: u64,
) -> Result<()> {
    if archive.is_empty() || episodes == 0 {
        return Ok(());
    }
    let mut indices = (0..archive.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        archive[*right]
            .selection_fitness
            .total_cmp(&archive[*left].selection_fitness)
    });
    indices.truncate(16);
    let trap_challenges = niche_challenges(
        CurriculumStage::EscapeCorners,
        ScenarioKind::ConcaveTrap,
        episodes,
        seed ^ 0x7A_A9,
        None,
        None,
        false,
    );
    let low_battery_challenges = niche_challenges(
        CurriculumStage::NavigateVariedRooms,
        ScenarioKind::MixedRoom,
        episodes,
        seed ^ 0xBA77,
        None,
        Some(0.18),
        true,
    );
    let corridor_challenges = niche_challenges(
        CurriculumStage::NavigateVariedRooms,
        ScenarioKind::EmptyRoom,
        episodes,
        seed ^ 0x0C07_71D0,
        Some((12.0, 3.5)),
        None,
        false,
    );

    for index in indices {
        let genome = archive[index].genome.clone();
        let nominal = evaluate_neat_locomotion(
            Some(&genome),
            CurriculumStage::TransferCandidatesToPete,
            episodes,
            steps,
            seed.saturating_add(index as u64 * 0x1000),
            NeatPerturbation::default(),
            None,
            false,
        )
        .await?;
        let degraded = evaluate_neat_locomotion(
            Some(&genome),
            CurriculumStage::TransferCandidatesToPete,
            episodes,
            steps,
            seed.saturating_add(index as u64 * 0x1000),
            NeatPerturbation {
                sensor_noise: 0.08,
                latency_steps: 1,
                ..NeatPerturbation::default()
            },
            None,
            false,
        )
        .await?;
        let mismatch = evaluate_neat_locomotion(
            Some(&genome),
            CurriculumStage::TransferCandidatesToPete,
            episodes,
            steps,
            seed.saturating_add(index as u64 * 0x1000),
            NeatPerturbation {
                left_motor_scale: 0.82,
                right_motor_scale: 1.0,
                wheel_gain_jitter: 0.12,
                deadband_m_s: 0.015,
                ..NeatPerturbation::default()
            },
            None,
            false,
        )
        .await?;
        let traps = evaluate_neat_locomotion(
            Some(&genome),
            CurriculumStage::EscapeCorners,
            episodes,
            steps,
            seed ^ 0x7A_A9,
            NeatPerturbation::default(),
            Some(&trap_challenges),
            false,
        )
        .await?;
        let low_battery = evaluate_neat_locomotion(
            Some(&genome),
            CurriculumStage::NavigateVariedRooms,
            episodes,
            steps,
            seed ^ 0xBA77,
            NeatPerturbation::default(),
            Some(&low_battery_challenges),
            false,
        )
        .await?;
        let corridor = evaluate_neat_locomotion(
            Some(&genome),
            CurriculumStage::NavigateVariedRooms,
            episodes,
            steps,
            seed ^ 0x0C07_71D0,
            NeatPerturbation::default(),
            Some(&corridor_challenges),
            false,
        )
        .await?;
        let low_battery_progress = if low_battery.metrics.battery_depleted == 0 {
            low_battery.metrics.distance_without_collision_m / episodes as f32
        } else {
            0.0
        };
        let evidence = NicheQualificationEvidence {
            degraded_sensor_retention: success_retention(&nominal, &degraded),
            motor_mismatch_retention: success_retention(&nominal, &mismatch),
            heldout_trap_success_rate: traps.success_rate(),
            low_battery_progress_m: low_battery_progress,
            corridor_success_rate: corridor.success_rate(),
        };
        archive[index].niche = archive[index]
            .descriptor
            .evidence_based_niche_label(archive[index].evaluation.traits, evidence);
        archive[index].qualification_evidence = Some(evidence);
    }
    Ok(())
}

fn success_retention(nominal: &NeatPolicyEvaluation, perturbed: &NeatPolicyEvaluation) -> f32 {
    if nominal.success_rate() <= f32::EPSILON {
        0.0
    } else {
        (perturbed.success_rate() / nominal.success_rate()).clamp(0.0, 1.0)
    }
}

fn niche_challenges(
    stage: CurriculumStage,
    kind: ScenarioKind,
    episodes: usize,
    seed: u64,
    arena_override_m: Option<(f32, f32)>,
    initial_battery: Option<f32>,
    disable_chargers: bool,
) -> Vec<NeatEnvironmentChallenge> {
    (0..episodes)
        .map(|episode| NeatEnvironmentChallenge {
            stage,
            kind,
            seed: seed.saturating_add(episode as u64),
            arena_override_m,
            initial_battery,
            disable_chargers,
        })
        .collect()
}

fn update_global_repertoire(
    repertoire: &mut Vec<NeatNicheArchiveEntry>,
    stage_archive: &[NeatNicheArchiveEntry],
) {
    for entry in stage_archive {
        upsert_niche_archive_entry(repertoire, entry.clone());
    }
}

fn upsert_niche_archive_entry(
    archive: &mut Vec<NeatNicheArchiveEntry>,
    candidate: NeatNicheArchiveEntry,
) {
    if let Some(existing) = archive
        .iter_mut()
        .find(|entry| entry.descriptor == candidate.descriptor)
    {
        if candidate.selection_fitness > existing.selection_fitness {
            *existing = candidate;
        }
    } else {
        archive.push(candidate);
    }
}

fn write_stage_niche_checkpoints(
    archive: &[NeatNicheArchiveEntry],
    generation: u64,
    report_dir: &Path,
) -> Result<()> {
    let archive_dir = report_dir.join("niche-archive");
    fs::create_dir_all(&archive_dir)?;
    for entry in archive {
        let file_name = format!(
            "{}-{}-c{}-t{}-a{}-e{}-r{}.json",
            neat_stage_slug(entry.stage),
            niche_slug(entry.niche),
            entry.descriptor.collision_frequency_bin,
            entry.descriptor.turning_intensity_bin,
            entry.descriptor.area_coverage_bin,
            entry.descriptor.energy_consumption_bin,
            entry.descriptor.recovery_aggressiveness_bin
        );
        let checkpoint =
            LocomotionCheckpoint::new(generation, entry.selection_fitness, entry.genome.clone());
        checkpoint.save(archive_dir.join(file_name))?;
    }
    Ok(())
}

fn niche_slug(niche: pete_neat::NicheLabel) -> &'static str {
    match niche {
        pete_neat::NicheLabel::OpenRoomExplorer => "open-room-explorer",
        pete_neat::NicheLabel::NarrowCorridorNavigator => "narrow-corridor-navigator",
        pete_neat::NicheLabel::ConcaveTrapEscapeSpecialist => "concave-trap-escape-specialist",
        pete_neat::NicheLabel::ClutterSpecialist => "clutter-specialist",
        pete_neat::NicheLabel::LowBatteryConservativeMover => "low-battery-conservative-mover",
        pete_neat::NicheLabel::DegradedSensorNavigator => "degraded-sensor-navigator",
        pete_neat::NicheLabel::AsymmetricMotorCompensator => "asymmetric-motor-compensator",
        pete_neat::NicheLabel::Generalist => "generalist",
    }
}

fn add_neat_metrics(total: &mut NeatEpisodeMetrics, episode: NeatEpisodeMetrics) {
    total.new_area_cells = total.new_area_cells.saturating_add(episode.new_area_cells);
    total.distance_without_collision_m += episode.distance_without_collision_m;
    total.successful_escapes = total
        .successful_escapes
        .saturating_add(episode.successful_escapes);
    total.escape_boundary_crossings = total
        .escape_boundary_crossings
        .saturating_add(episode.escape_boundary_crossings);
    total.trap_mouth_progress_m += episode.trap_mouth_progress_m;
    total.collisions = total.collisions.saturating_add(episode.collisions);
    total.repeated_state_steps = total
        .repeated_state_steps
        .saturating_add(episode.repeated_state_steps);
    total.short_cycle_count = total
        .short_cycle_count
        .saturating_add(episode.short_cycle_count);
    total.short_cycle_steps = total
        .short_cycle_steps
        .saturating_add(episode.short_cycle_steps);
    total.recent_repetition_steps = total
        .recent_repetition_steps
        .saturating_add(episode.recent_repetition_steps);
    total.maximum_displacement_m += episode.maximum_displacement_m;
    total.radius_bands_reached = total
        .radius_bands_reached
        .saturating_add(episode.radius_bands_reached);
    total.arena_sectors_visited = total
        .arena_sectors_visited
        .saturating_add(episode.arena_sectors_visited);
    total.wheel_motion_m += episode.wheel_motion_m;
    total.angular_motion_rad += episode.angular_motion_rad;
    total.recovery_activation_sum += episode.recovery_activation_sum;
    total.stalled_steps = total.stalled_steps.saturating_add(episode.stalled_steps);
    total.safety_vetoes = total.safety_vetoes.saturating_add(episode.safety_vetoes);
    total.safety_invariant_violations = total
        .safety_invariant_violations
        .saturating_add(episode.safety_invariant_violations);
    total.resource_energy_used += episode.resource_energy_used;
    total.sensor_energy_cost += episode.sensor_energy_cost;
    total.computation_energy_cost += episode.computation_energy_cost;
    total.collision_energy_cost += episode.collision_energy_cost;
    total.minimum_resource_battery = if total.minimum_resource_battery == 0.0 {
        episode.minimum_resource_battery
    } else {
        total
            .minimum_resource_battery
            .min(episode.minimum_resource_battery)
    };
    total.final_resource_battery += episode.final_resource_battery;
    total.minimum_resource_health = if total.minimum_resource_health == 0.0 {
        episode.minimum_resource_health
    } else {
        total
            .minimum_resource_health
            .min(episode.minimum_resource_health)
    };
    total.final_resource_health += episode.final_resource_health;
    total.battery_depleted = total
        .battery_depleted
        .saturating_add(episode.battery_depleted);
    total.health_depleted = total
        .health_depleted
        .saturating_add(episode.health_depleted);
}

fn write_json_report(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("writing {}", path.display()))
}

fn curriculum_split(
    episode_index: usize,
    episode_count: usize,
    validation_ratio: f32,
    test_ratio: f32,
) -> &'static str {
    let validation_count = ((episode_count as f32) * validation_ratio).round() as usize;
    let test_count = ((episode_count as f32) * test_ratio).round() as usize;
    let train_count = episode_count.saturating_sub(validation_count + test_count);
    if episode_index < train_count {
        "train"
    } else if episode_index < train_count + validation_count {
        "validation"
    } else {
        "test"
    }
}

fn scenario_object_summary(objects: &[pete_sim::SimObject]) -> serde_json::Value {
    let mut chargers = 0usize;
    let mut obstacles = 0usize;
    let mut people = 0usize;
    let mut speakers = 0usize;
    let mut landmarks = 0usize;

    for object in objects {
        match &object.kind {
            pete_sim::SimObjectKind::Charger => chargers += 1,
            pete_sim::SimObjectKind::Obstacle => obstacles += 1,
            pete_sim::SimObjectKind::Person { .. } => people += 1,
            pete_sim::SimObjectKind::SoundSource { .. } => speakers += 1,
            pete_sim::SimObjectKind::Landmark { .. } => landmarks += 1,
        }
    }

    serde_json::json!({
        "chargers": chargers,
        "obstacles": obstacles,
        "people": people,
        "speakers": speakers,
        "landmarks": landmarks,
    })
}
