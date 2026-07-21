fn neat_generation_challenges(
    stage: CurriculumStage,
    episodes: usize,
    seed: u64,
    archive: &[NeatWorldArchiveEntry],
    replay_ratio: f32,
    mutation_ratio: f32,
    rehearsal_ratio: f32,
) -> Vec<NeatEnvironmentChallenge> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0xA11CE_5757);
    let current_stage_index = CurriculumStage::ORDER
        .iter()
        .position(|candidate| *candidate == stage)
        .unwrap_or_default();
    let prior_stages = &CurriculumStage::ORDER[..current_stage_index];
    let retention_enabled = rehearsal_ratio > 0.0 && !prior_stages.is_empty();
    // A small configured batch is expanded only as far as necessary to give
    // every retained prerequisite one episode while keeping >= 50% fresh
    // current-stage signal.
    let total_episodes = if retention_enabled {
        episodes.max(prior_stages.len().saturating_mul(2))
    } else {
        episodes
    };
    let fresh_count = (total_episodes + 1) / 2;
    let rehearsal_count = if retention_enabled {
        prior_stages.len()
    } else {
        0
    };
    let auxiliary_budget = total_episodes.saturating_sub(fresh_count + rehearsal_count);
    let wants_replay = replay_ratio > 0.0;
    let wants_mutation = mutation_ratio > 0.0;
    let replay_count = auxiliary_budget.min(wants_replay as usize);
    let mutation_count = auxiliary_budget
        .saturating_sub(replay_count)
        .min(wants_mutation as usize);
    let extra_fresh = auxiliary_budget.saturating_sub(replay_count + mutation_count);
    let mut ranked_archive = archive
        .iter()
        .filter(|entry| entry.challenge.stage == stage)
        .collect::<Vec<_>>();
    ranked_archive.sort_by(|left, right| {
        right
            .distinction
            .total_cmp(&left.distinction)
            .then_with(|| right.difficulty.total_cmp(&left.difficulty))
    });

    let mut challenges = Vec::with_capacity(total_episodes);
    for current_index in 0..fresh_count + extra_fresh {
        let episode_seed = seed
            .saturating_add((current_index as u64) << 8)
            .saturating_add(rng.gen_range(0..=u16::MAX) as u64);
        challenges.push(NeatEnvironmentChallenge {
            stage,
            kind: neat_stage_scenario(stage, episode_seed),
            seed: episode_seed,
            arena_override_m: None,
            initial_battery: None,
            disable_chargers: false,
        });
    }
    for entry in ranked_archive.iter().take(replay_count) {
        challenges.push(entry.challenge);
    }
    for entry in ranked_archive.iter().cycle().take(mutation_count) {
        challenges.push(mutate_neat_environment(entry.challenge, &mut rng));
    }
    // If no current-stage archive exists yet, auxiliary slots become fresh
    // current-stage worlds rather than leaking prior-stage archive material.
    while challenges.len() < fresh_count + extra_fresh + replay_count + mutation_count {
        let episode_seed = seed
            .saturating_add((challenges.len() as u64) << 8)
            .saturating_add(rng.gen_range(0..=u16::MAX) as u64);
        challenges.push(NeatEnvironmentChallenge {
            stage,
            kind: neat_stage_scenario(stage, episode_seed),
            seed: episode_seed,
            arena_override_m: None,
            initial_battery: None,
            disable_chargers: false,
        });
    }
    for rehearsal_index in 0..rehearsal_count {
        let rehearsal_stage = prior_stages[rehearsal_index];
        let archived = archive
            .iter()
            .filter(|entry| entry.challenge.stage == rehearsal_stage)
            .max_by(|left, right| {
                left.difficulty
                    .total_cmp(&right.difficulty)
                    .then_with(|| left.distinction.total_cmp(&right.distinction))
            });
        if let Some(entry) = archived {
            challenges.push(entry.challenge);
            continue;
        }
        let episode_seed = seed
            .saturating_add((challenges.len() as u64) << 8)
            .saturating_add(rng.gen_range(0..=u16::MAX) as u64);
        challenges.push(NeatEnvironmentChallenge {
            stage: rehearsal_stage,
            kind: neat_stage_scenario(rehearsal_stage, episode_seed),
            seed: episode_seed,
            arena_override_m: None,
            initial_battery: None,
            disable_chargers: false,
        });
    }
    while challenges.len() < total_episodes {
        let episode_seed = seed
            .saturating_add((challenges.len() as u64) << 8)
            .saturating_add(rng.gen_range(0..=u16::MAX) as u64);
        challenges.push(NeatEnvironmentChallenge {
            stage,
            kind: neat_stage_scenario(stage, episode_seed),
            seed: episode_seed,
            arena_override_m: None,
            initial_battery: None,
            disable_chargers: false,
        });
    }
    challenges
}

fn mutate_neat_environment<R: Rng + ?Sized>(
    challenge: NeatEnvironmentChallenge,
    rng: &mut R,
) -> NeatEnvironmentChallenge {
    let kinds = match challenge.stage {
        CurriculumStage::BackAwayReliably => &[ScenarioKind::ObstacleAvoidance][..],
        CurriculumStage::ChooseUsefulTurn => &[
            ScenarioKind::ObstacleAvoidance,
            ScenarioKind::ColumnTrap,
            ScenarioKind::ConcaveTrap,
        ],
        CurriculumStage::EscapeCorners => &[
            ScenarioKind::CornerTrap,
            ScenarioKind::ColumnTrap,
            ScenarioKind::ConcaveTrap,
        ],
        CurriculumStage::LeaveStartRegion => {
            &[ScenarioKind::EmptyRoom, ScenarioKind::ObstacleAvoidance]
        }
        CurriculumStage::ExpandLocalCoverage | CurriculumStage::BreakShortCycles => &[
            ScenarioKind::EmptyRoom,
            ScenarioKind::ObstacleAvoidance,
            ScenarioKind::MixedRoom,
        ],
        CurriculumStage::ExploreWithoutLooping => {
            &[ScenarioKind::MixedRoom, ScenarioKind::ConcaveTrap]
        }
        CurriculumStage::NavigateVariedRooms | CurriculumStage::TransferCandidatesToPete => &[
            ScenarioKind::EmptyRoom,
            ScenarioKind::ObstacleAvoidance,
            ScenarioKind::CornerTrap,
            ScenarioKind::ColumnTrap,
            ScenarioKind::ConcaveTrap,
            ScenarioKind::Dream,
        ],
    };
    NeatEnvironmentChallenge {
        stage: challenge.stage,
        kind: kinds.choose(rng).copied().unwrap_or(challenge.kind),
        seed: challenge.seed.wrapping_add(rng.gen_range(1..=10_000)) ^ rng.gen::<u64>(),
        arena_override_m: challenge.arena_override_m,
        initial_battery: challenge.initial_battery,
        disable_chargers: challenge.disable_chargers,
    }
}

fn update_neat_world_archive(
    archive: &mut Vec<NeatWorldArchiveEntry>,
    challenges: &[NeatEnvironmentChallenge],
    evaluations: &[NeatPolicyEvaluation],
    generation: u64,
    archive_limit: usize,
) -> usize {
    let mut retained = 0usize;
    for (episode_index, challenge) in challenges.iter().copied().enumerate() {
        let scores = evaluations
            .iter()
            .filter_map(|evaluation| evaluation.environment_scores.get(episode_index).copied())
            .filter(|score| score.is_finite())
            .collect::<Vec<_>>();
        if scores.len() < 2 {
            continue;
        }
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores
            .iter()
            .map(|score| {
                let delta = score - mean;
                delta * delta
            })
            .sum::<f32>()
            / scores.len() as f32;
        let distinction = variance.sqrt();
        let difficulty = (1.0 - ((mean + 20.0) / 80.0)).clamp(0.0, 1.0);
        let discriminates = distinction >= 0.25 && (0.10..=0.90).contains(&difficulty);
        if !discriminates {
            continue;
        }
        let candidate = NeatWorldArchiveEntry {
            challenge,
            difficulty,
            distinction,
            retained_generation: generation,
        };
        if let Some(existing) = archive
            .iter_mut()
            .find(|entry| entry.challenge == candidate.challenge)
        {
            if candidate.distinction > existing.distinction {
                *existing = candidate;
                retained = retained.saturating_add(1);
            }
        } else {
            archive.push(candidate);
            retained = retained.saturating_add(1);
        }
    }
    archive.sort_by(|left, right| {
        let left_score = left.distinction * (1.0 - (left.difficulty - 0.5).abs());
        let right_score = right.distinction * (1.0 - (right.difficulty - 0.5).abs());
        right_score
            .total_cmp(&left_score)
            .then_with(|| right.retained_generation.cmp(&left.retained_generation))
    });
    if archive.len() > archive_limit {
        archive.truncate(archive_limit);
    }
    retained
}

fn update_neat_niche_archive(
    archive: &mut Vec<NeatNicheArchiveEntry>,
    stage: CurriculumStage,
    generation_archive: &[QualityDiversityEntry],
    genomes: &[Genome],
    evaluations: &[NeatPolicyEvaluation],
) {
    for entry in generation_archive {
        let Some(genome) = genomes.get(entry.genome_index) else {
            continue;
        };
        let Some(evaluation) = evaluations.get(entry.genome_index) else {
            continue;
        };
        let candidate = NeatNicheArchiveEntry {
            stage,
            niche: entry.niche,
            descriptor: entry.descriptor,
            selection_fitness: entry.selection_fitness,
            base_selection_fitness: evaluation.base_selection_fitness,
            novelty_score: evaluation.novelty_score,
            diagnostic_fitness: evaluation.fitness,
            qualification_evidence: None,
            genome: genome.clone(),
            evaluation: evaluation.clone(),
        };
        upsert_niche_archive_entry(archive, candidate);
    }
}
