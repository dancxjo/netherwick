async fn validate_neat_curriculum(
    genome: &Genome,
    current_stage_index: usize,
    evolving_stages: &[CurriculumStage],
    steps: usize,
    validation_seed: u64,
    validation_round: u64,
) -> Result<Vec<NeatStageValidationReport>> {
    let mut reports = Vec::with_capacity(current_stage_index + 1);
    for (stage_index, stage) in evolving_stages
        .iter()
        .copied()
        .enumerate()
        .take(current_stage_index + 1)
    {
        let criteria = stage.promotion_criteria();
        let episodes = criteria.minimum_seeded_episodes as usize;
        let seed = validation_seed
            .saturating_add(validation_round << 32)
            .saturating_add((stage_index as u64) << 20);
        let evaluation = evaluate_neat_locomotion(
            Some(genome),
            stage,
            episodes,
            steps,
            seed,
            NeatPerturbation::default(),
            None,
            false,
        )
        .await?;
        let passed = evaluation.success_rate() >= criteria.minimum_success_rate
            && evaluation.collision_rate <= criteria.maximum_collision_rate
            && evaluation.traits.safety_veto_rate <= stage.maximum_safety_veto_rate()
            && evaluation.metrics.safety_invariant_violations
                <= criteria.maximum_safety_invariant_violations;
        reports.push(NeatStageValidationReport {
            candidate_kind: String::new(),
            stage,
            validation_round,
            seeded_episodes: episodes,
            success_rate: evaluation.success_rate(),
            collision_rate: evaluation.collision_rate,
            safety_veto_rate: evaluation.traits.safety_veto_rate,
            safety_invariant_violations: evaluation.metrics.safety_invariant_violations,
            passed,
        });
    }
    Ok(reports)
}

fn validation_candidates(
    current_genome: &Genome,
    current_evaluation: &NeatPolicyEvaluation,
    stage_best: &Option<(f32, Genome, NeatPolicyEvaluation)>,
    stage_archive: &[NeatNicheArchiveEntry],
    repertoire: &[NeatNicheArchiveEntry],
) -> Vec<(String, Genome, NeatPolicyEvaluation)> {
    let mut candidates = vec![(
        "reproductive-champion".to_string(),
        current_genome.clone(),
        current_evaluation.clone(),
    )];
    if let Some((_, genome, evaluation)) = stage_best.as_ref().filter(|(_, _, evaluation)| {
        evaluation.metrics.safety_invariant_violations == 0
            && evaluation.traits.safety_veto_rate <= 0.05
    }) {
        candidates.push((
            "structured-stage-best".to_string(),
            genome.clone(),
            evaluation.clone(),
        ));
    }
    if let Some(entry) = stage_archive
        .iter()
        .chain(repertoire.iter())
        .filter(|entry| {
            entry.niche == pete_neat::NicheLabel::Generalist && archived_evaluation_is_safe(entry)
        })
        .max_by(|left, right| {
            left.evaluation
                .success_rate()
                .total_cmp(&right.evaluation.success_rate())
                .then_with(|| left.evaluation.fitness.total_cmp(&right.evaluation.fitness))
        })
    {
        candidates.push((
            "archived-generalist".to_string(),
            entry.genome.clone(),
            entry.evaluation.clone(),
        ));
    }
    if let Some(entry) = stage_archive
        .iter()
        .filter(|entry| archived_evaluation_is_safe(entry))
        .max_by(|left, right| {
            left.evaluation
                .success_rate()
                .total_cmp(&right.evaluation.success_rate())
                .then_with(|| left.evaluation.fitness.total_cmp(&right.evaluation.fitness))
        })
    {
        candidates.push((
            "current-stage-specialist".to_string(),
            entry.genome.clone(),
            entry.evaluation.clone(),
        ));
    }
    let mut unique = Vec::<(String, Genome, NeatPolicyEvaluation)>::new();
    for candidate in candidates {
        if !unique.iter().any(|(_, genome, _)| genome == &candidate.1) {
            unique.push(candidate);
        }
    }
    unique
}

async fn select_neat_transfer_candidate(
    stage_champion: &Genome,
    repertoire: &[NeatNicheArchiveEntry],
    steps: usize,
    seed: u64,
) -> Result<Genome> {
    let mut candidates = vec![stage_champion.clone()];
    let mut ranked = repertoire.iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.selection_fitness.total_cmp(&left.selection_fitness));
    for entry in ranked {
        if candidates.len() >= 8 {
            break;
        }
        if !candidates.contains(&entry.genome) {
            candidates.push(entry.genome.clone());
        }
    }
    let candidate_count = candidates.len();
    let mut best: Option<(Genome, NeatPolicyEvaluation)> = None;
    for (index, genome) in candidates.into_iter().enumerate() {
        let evaluation = evaluate_neat_locomotion(
            Some(&genome),
            CurriculumStage::TransferCandidatesToPete,
            64,
            steps,
            seed,
            NeatPerturbation::default(),
            None,
            false,
        )
        .await?;
        println!(
            "transfer shortlist {}/{} success={:.1}% collision={:.4} invariants={}",
            index + 1,
            candidate_count,
            evaluation.success_rate() * 100.0,
            evaluation.collision_rate,
            evaluation.metrics.safety_invariant_violations
        );
        if best
            .as_ref()
            .is_none_or(|(_, current)| transfer_evaluation_better(&evaluation, current))
        {
            best = Some((genome, evaluation));
        }
    }
    best.map(|(genome, _)| genome)
        .context("empty transfer candidate shortlist")
}

async fn capture_neat_champion(
    genome: &Genome,
    stage: CurriculumStage,
    steps: usize,
    seed: u64,
    path: &Path,
    fitness: f32,
) -> Result<()> {
    let evaluation = evaluate_neat_locomotion(
        Some(genome),
        stage,
        1,
        steps,
        seed,
        NeatPerturbation::default(),
        None,
        true,
    )
    .await?;
    let mut writer = CaptureWriter::create(path, CaptureSource::Sim, Some(100)).await?;
    writer.manifest_mut().scenario = evaluation.scenario;
    writer.manifest_mut().notes.push(format!(
        "NEAT locomotion champion stage={} selection={fitness:.4} nodes={} connections={}",
        neat_stage_slug(stage),
        genome.nodes.len(),
        genome.connections.len()
    ));
    writer.manifest_mut().command_args = std::env::args().collect();
    for snapshot in evaluation.snapshots {
        writer
            .append_snapshot(snapshot.body.last_update_ms, snapshot, Vec::new())
            .await?;
    }
    writer.finish().await?;
    Ok(())
}

fn neat_stage_scenario(stage: CurriculumStage, seed: u64) -> ScenarioKind {
    match stage {
        CurriculumStage::BackAwayReliably => ScenarioKind::ObstacleAvoidance,
        CurriculumStage::ChooseUsefulTurn => match seed % 3 {
            0 => ScenarioKind::ObstacleAvoidance,
            1 => ScenarioKind::ColumnTrap,
            _ => ScenarioKind::ConcaveTrap,
        },
        CurriculumStage::EscapeCorners => match seed % 3 {
            0 => ScenarioKind::CornerTrap,
            1 => ScenarioKind::ColumnTrap,
            _ => ScenarioKind::ConcaveTrap,
        },
        CurriculumStage::LeaveStartRegion => match seed % 2 {
            0 => ScenarioKind::EmptyRoom,
            _ => ScenarioKind::ObstacleAvoidance,
        },
        CurriculumStage::ExpandLocalCoverage => match seed % 3 {
            0 => ScenarioKind::EmptyRoom,
            1 => ScenarioKind::ObstacleAvoidance,
            _ => ScenarioKind::MixedRoom,
        },
        CurriculumStage::BreakShortCycles => match seed % 3 {
            0 => ScenarioKind::EmptyRoom,
            1 => ScenarioKind::ObstacleAvoidance,
            _ => ScenarioKind::MixedRoom,
        },
        CurriculumStage::ExploreWithoutLooping => {
            if seed % 4 == 0 {
                ScenarioKind::ConcaveTrap
            } else {
                ScenarioKind::MixedRoom
            }
        }
        CurriculumStage::NavigateVariedRooms | CurriculumStage::TransferCandidatesToPete => {
            match seed % 6 {
                0 => ScenarioKind::EmptyRoom,
                1 => ScenarioKind::ObstacleAvoidance,
                2 => ScenarioKind::CornerTrap,
                3 => ScenarioKind::ColumnTrap,
                4 => ScenarioKind::ConcaveTrap,
                _ => ScenarioKind::Dream,
            }
        }
    }
}

fn neat_stage_slug(stage: CurriculumStage) -> &'static str {
    match stage {
        CurriculumStage::BackAwayReliably => "back-away-reliably",
        CurriculumStage::ChooseUsefulTurn => "choose-a-useful-turn",
        CurriculumStage::EscapeCorners => "escape-corners",
        CurriculumStage::LeaveStartRegion => "leave-start-region",
        CurriculumStage::ExpandLocalCoverage => "expand-local-coverage",
        CurriculumStage::BreakShortCycles => "break-short-cycles",
        CurriculumStage::ExploreWithoutLooping => "explore-without-looping",
        CurriculumStage::NavigateVariedRooms => "navigate-varied-rooms",
        CurriculumStage::TransferCandidatesToPete => "transfer-candidates-to-pete",
    }
}

fn constrained_episode_score(metrics: NeatEpisodeMetrics, succeeded: bool, steps: usize) -> f32 {
    let step_count = steps.max(1) as f32;
    let collision_rate = metrics.collisions as f32 / step_count;
    let repetition_rate = metrics.repeated_state_steps as f32 / step_count;
    let efficiency = (metrics.new_area_cells as f32 + metrics.distance_without_collision_m)
        / (1.0 + metrics.wheel_motion_m);
    (succeeded as u8 as f32) * 20.0
        + metrics.escape_boundary_crossings as f32 * 8.0
        + metrics.successful_escapes as f32 * 6.0
        + efficiency
        + metrics.trap_mouth_progress_m
        - collision_rate * 20.0
        - repetition_rate * 5.0
        - metrics.safety_vetoes as f32 * 100.0
        - metrics.resource_energy_used * 4.0
        - metrics.battery_depleted as f32 * 50.0
        - metrics.health_depleted as f32 * 75.0
}

const RECENT_CELL_WINDOW: usize = 24;

fn observe_short_cycles(
    recent: &mut VecDeque<(i32, i32)>,
    cell: (i32, i32),
    metrics: &mut NeatEpisodeMetrics,
) {
    if recent.iter().any(|candidate| *candidate == cell) {
        metrics.recent_repetition_steps = metrics.recent_repetition_steps.saturating_add(1);
    }
    let cycle_length = (2..=4).find(|length| {
        recent
            .len()
            .checked_sub(*length)
            .and_then(|index| recent.get(index))
            .is_some_and(|candidate| *candidate == cell)
    });
    if let Some(length) = cycle_length {
        metrics.short_cycle_count = metrics.short_cycle_count.saturating_add(1);
        metrics.short_cycle_steps = metrics.short_cycle_steps.saturating_add(length as u32);
    }
    recent.push_back(cell);
    if recent.len() > RECENT_CELL_WINDOW {
        recent.pop_front();
    }
}

fn neat_plasticity_reward(step_distance: f32, collision: bool, safety_vetoed: bool) -> f32 {
    let mut reward = (step_distance * 5.0).clamp(0.0, 0.5);
    if collision {
        reward -= 0.5;
    }
    if safety_vetoed {
        reward -= 0.5;
    }
    reward.clamp(-1.0, 1.0)
}

fn neat_safety_invariant_violated(decision: &pete_autonomic::SafetyDecision) -> bool {
    let command = decision.command;
    let finite = command.forward.is_finite() && command.turn.is_finite();
    let bounded =
        command.forward.abs() <= 0.6 + f32::EPSILON && command.turn.abs() <= 1.0 + f32::EPSILON;
    let stopped_when_vetoed = !decision.vetoed
        || (command.forward.abs() <= f32::EPSILON && command.turn.abs() <= f32::EPSILON);
    !finite || !bounded || !stopped_when_vetoed
}

fn neat_sensor_energy_cost(range: &RangeSense) -> f32 {
    0.00025 + range.beams.len() as f32 * 0.000002
}

fn neat_topology_metabolic_cost(genome: &Genome) -> f32 {
    (genome.nodes.len() as f32 * 0.000015 + genome.connections.len() as f32 * 0.000008).min(0.01)
}
