#[derive(Clone, Copy, Debug)]
struct NeatPerturbation {
    sensor_noise: f32,
    left_motor_scale: f32,
    right_motor_scale: f32,
    wheel_gain_jitter: f32,
    deadband_m_s: f32,
    latency_steps: usize,
}

impl Default for NeatPerturbation {
    fn default() -> Self {
        Self {
            sensor_noise: 0.0,
            left_motor_scale: 1.0,
            right_motor_scale: 1.0,
            wheel_gain_jitter: 0.08,
            deadband_m_s: 0.01,
            latency_steps: 0,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct NeatPolicyEvaluation {
    fitness: f32,
    selection_fitness: f32,
    base_selection_fitness: f32,
    novelty_score: f32,
    lifetime_selection: NeatLifetimeSelection,
    traits: FitnessTraits,
    selection_summary: Option<SelectionSummary>,
    metrics: NeatEpisodeMetrics,
    successful_episodes: usize,
    episodes: usize,
    collision_rate: f32,
    environment_scores: Vec<f32>,
    #[serde(default)]
    stage_competence: BTreeMap<CurriculumStage, StageCompetence>,
    #[serde(skip)]
    snapshots: Vec<WorldSnapshot>,
    #[serde(skip)]
    scenario: Option<pete_sim::ScenarioMetadata>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct StageCompetence {
    episodes: usize,
    successes: usize,
    weighted_score: f32,
    collision_rate: f32,
    invariant_violations: u32,
}

impl StageCompetence {
    fn success_rate(self) -> f32 {
        if self.episodes == 0 {
            0.0
        } else {
            self.successes as f32 / self.episodes as f32
        }
    }
}

impl NeatPolicyEvaluation {
    fn success_rate(&self) -> f32 {
        if self.episodes == 0 {
            0.0
        } else {
            self.successful_episodes as f32 / self.episodes as f32
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct NeatLifetimeSelection {
    mean_score: f32,
    lower_quartile_score: f32,
    worst_score: f32,
    qualification_probability: f32,
    robustness_score: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NeatGenerationReport {
    stage: CurriculumStage,
    generation_in_stage: usize,
    population_generation: u64,
    species: usize,
    best_fitness: f32,
    mean_fitness: f32,
    worst_fitness: f32,
    best_selection_fitness: f32,
    mean_selection_fitness: f32,
    worst_selection_fitness: f32,
    best_novelty: f32,
    mean_novelty: f32,
    champion_novelty: f32,
    champion_selection_summary: SelectionSummary,
    champion_lifetime_selection: NeatLifetimeSelection,
    champion_traits: FitnessTraits,
    champion_nodes: usize,
    champion_connections: usize,
    champion_metrics: NeatEpisodeMetrics,
    champion_success_rate: f32,
    champion_collision_rate: f32,
    archive_cells: usize,
    world_archive_size: usize,
    replayed_worlds: usize,
    retained_worlds: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NeatStageReport {
    stage: CurriculumStage,
    best_fitness: f32,
    best_evaluation: NeatPolicyEvaluation,
    niche_archive: Vec<NeatNicheArchiveEntry>,
    generations: Vec<NeatGenerationReport>,
    capture: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NeatNicheArchiveEntry {
    stage: CurriculumStage,
    niche: pete_neat::NicheLabel,
    descriptor: QualityDiversityDescriptor,
    selection_fitness: f32,
    base_selection_fitness: f32,
    novelty_score: f32,
    diagnostic_fitness: f32,
    #[serde(default)]
    qualification_evidence: Option<NicheQualificationEvidence>,
    genome: Genome,
    evaluation: NeatPolicyEvaluation,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct NeatEnvironmentChallenge {
    stage: CurriculumStage,
    kind: ScenarioKind,
    seed: u64,
    #[serde(default)]
    arena_override_m: Option<(f32, f32)>,
    #[serde(default)]
    initial_battery: Option<f32>,
    #[serde(default)]
    disable_chargers: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NeatWorldArchiveEntry {
    challenge: NeatEnvironmentChallenge,
    difficulty: f32,
    distinction: f32,
    retained_generation: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NeatStageValidationReport {
    #[serde(default)]
    candidate_kind: String,
    stage: CurriculumStage,
    validation_round: u64,
    seeded_episodes: usize,
    success_rate: f32,
    collision_rate: f32,
    #[serde(default)]
    safety_veto_rate: f32,
    safety_invariant_violations: u32,
    passed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NeatTrainerState {
    schema_version: u32,
    behavior: String,
    seed: u64,
    validation_seed: u64,
    heldout_seed: u64,
    steps: usize,
    episodes_per_genome: usize,
    settings: NeatTrainerSettings,
    population: Population,
    #[serde(default)]
    stage: Option<CurriculumStage>,
    stage_index: usize,
    generation_in_stage: usize,
    validation_round: u64,
    qualification_streak: usize,
    stage_qualified: bool,
    #[serde(default)]
    low_species_generations: usize,
    validations: Vec<NeatStageValidationReport>,
    generation_reports: Vec<NeatGenerationReport>,
    stage_best: Option<(f32, Genome, NeatPolicyEvaluation)>,
    stage_archive: Vec<NeatNicheArchiveEntry>,
    stage_reports: Vec<NeatStageReport>,
    repertoire: Vec<NeatNicheArchiveEntry>,
    novelty_archive: NoveltyArchive,
    world_archive: Vec<NeatWorldArchiveEntry>,
    transfer_genome: Genome,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct NeatTrainerSettings {
    initial_compatibility_threshold: f32,
    #[serde(default = "default_compatibility_threshold_floor")]
    compatibility_threshold_floor: f32,
    target_species_min: usize,
    target_species_max: usize,
    novelty_weight: f32,
    novelty_neighbors: usize,
    novelty_archive_limit: usize,
    world_archive_limit: usize,
    world_replay_ratio: f32,
    world_mutation_ratio: f32,
    rehearsal_ratio: f32,
    validation_every: usize,
    validation_passes: usize,
    niche_audit_episodes: usize,
}

impl From<&NeatTrainArgs> for NeatTrainerSettings {
    fn from(args: &NeatTrainArgs) -> Self {
        Self {
            initial_compatibility_threshold: args.compatibility_threshold,
            compatibility_threshold_floor: args.compatibility_threshold_floor,
            target_species_min: args.target_species_min,
            target_species_max: args.target_species_max,
            novelty_weight: args.novelty_weight,
            novelty_neighbors: args.novelty_neighbors,
            novelty_archive_limit: args.novelty_archive_limit,
            world_archive_limit: args.world_archive_limit,
            world_replay_ratio: args.world_replay_ratio,
            world_mutation_ratio: args.world_mutation_ratio,
            rehearsal_ratio: args.rehearsal_ratio,
            validation_every: args.validation_every,
            validation_passes: args.validation_passes,
            niche_audit_episodes: args.niche_audit_episodes,
        }
    }
}

fn default_compatibility_threshold_floor() -> f32 {
    0.05
}

const NEAT_TRAINER_STATE_SCHEMA_VERSION: u32 = 3;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NeatTrainingReport {
    behavior: String,
    seed: u64,
    heldout_seed_root: u64,
    #[serde(default)]
    validation_seed_root: u64,
    checkpoint: String,
    novelty_weight: f32,
    novelty_neighbors: usize,
    novelty_archive_limit: usize,
    world_archive_limit: usize,
    world_replay_ratio: f32,
    world_mutation_ratio: f32,
    stages: Vec<NeatStageReport>,
    repertoire: Vec<NeatNicheArchiveEntry>,
    world_archive: Vec<NeatWorldArchiveEntry>,
    #[serde(default)]
    validations: Vec<NeatStageValidationReport>,
    #[serde(default)]
    trainer_state: Option<String>,
    transfer_candidate: CandidateEvaluation,
    transfer_eligible: bool,
    transfer_criteria: pete_neat::PromotionCriteria,
    hardcoded_transfer_fitness: f32,
    candidate_transfer_fitness: f32,
    noisy_transfer_fitness: f32,
    motor_mismatch_transfer_fitness: f32,
    promotion: NeatPromotionReport,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NeatPromotionReport {
    enabled: bool,
    promoted: bool,
    reason: String,
    baseline_kind: String,
    baseline_checkpoint: Option<String>,
    baseline_fitness: f32,
    candidate_checkpoint: String,
    candidate_artifact: Option<String>,
    promoted_regime: Option<BehaviorRegime>,
    models_config: String,
}

fn load_or_initialize_neat_trainer_state(
    args: &NeatTrainArgs,
    config: NeatConfig,
) -> Result<NeatTrainerState> {
    if let Some(path) = args.resume.as_deref() {
        let bytes = fs::read(path).with_context(|| format!("reading NEAT trainer state {path}"))?;
        let mut state: NeatTrainerState = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing NEAT trainer state {path}"))?;
        if state.schema_version == 2 {
            state = migrate_neat_trainer_state_v2(state, args);
            println!(
                "migrated trainer state schema 2: explore-without-looping -> leave-start-region; preserved population and archives"
            );
        } else if state.schema_version != NEAT_TRAINER_STATE_SCHEMA_VERSION {
            anyhow::bail!(
                "unsupported NEAT trainer state schema {}; expected {}",
                state.schema_version,
                NEAT_TRAINER_STATE_SCHEMA_VERSION
            );
        }
        if state.behavior != args.behavior
            || state.seed != args.seed
            || state.validation_seed != args.validation_seed
            || state.heldout_seed != args.heldout_seed
            || state.steps != args.steps
            || state.episodes_per_genome != args.episodes_per_genome
            || state.population.config.population_size != args.population
            || state.settings != NeatTrainerSettings::from(args)
        {
            anyhow::bail!(
                "resume configuration differs from {}; behavior, seeds, steps, episodes, and population must match",
                path
            );
        }
        println!(
            "resuming trainer state {} at stage={} generation={}",
            path,
            state.stage.map(neat_stage_slug).unwrap_or("unknown"),
            state.generation_in_stage
        );
        return Ok(state);
    }

    let default_start = if args.founders_report.is_some() {
        CurriculumStage::LeaveStartRegion
    } else {
        CurriculumStage::BackAwayReliably
    };
    let start_stage = args
        .start_stage
        .as_deref()
        .map(parse_neat_stage)
        .transpose()?
        .unwrap_or(default_start);
    let evolving_stages = CurriculumStage::ORDER
        .into_iter()
        .filter(|stage| stage.evolves_population())
        .collect::<Vec<_>>();
    let stage_index = evolving_stages
        .iter()
        .position(|stage| *stage == start_stage)
        .context("transfer audit cannot be used as an evolutionary start stage")?;
    let mut rng = StdRng::seed_from_u64(args.seed ^ 0xF0_0D_E2);

    let (population, stage_reports, repertoire, world_archive, mut novelty_archive) =
        if let Some(report_path) = args.founders_report.as_deref() {
            let legacy: NeatTrainingReport = serde_json::from_slice(
                &fs::read(report_path)
                    .with_context(|| format!("reading founder report {report_path}"))?,
            )
            .with_context(|| format!("parsing founder report {report_path}"))?;
            let founders = legacy_founder_genomes(&legacy, args)?;
            let population = Population::from_founders(config, &founders, &mut rng)?;
            let mut novelty = NoveltyArchive::new(args.novelty_archive_limit);
            let traits = legacy
                .repertoire
                .iter()
                .map(|entry| entry.evaluation.traits)
                .collect::<Vec<_>>();
            let metrics = legacy
                .repertoire
                .iter()
                .map(|entry| entry.evaluation.metrics)
                .collect::<Vec<_>>();
            novelty.observe(&pete_neat::behavioral_descriptors(
                &traits,
                &metrics,
                args.episodes_per_genome,
                args.steps,
            ));
            println!(
                "reconstructed {}-member population from {} imported founders",
                population.genomes.len(),
                founders.len()
            );
            (
                population,
                legacy.stages.into_iter().take(stage_index).collect(),
                legacy.repertoire,
                legacy.world_archive,
                novelty,
            )
        } else {
            (
                Population::seeded(config, &mut rng),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                NoveltyArchive::new(args.novelty_archive_limit),
            )
        };
    novelty_archive.observe(&[]);
    let transfer_genome = population.genomes[0].clone();
    Ok(NeatTrainerState {
        schema_version: NEAT_TRAINER_STATE_SCHEMA_VERSION,
        behavior: args.behavior.clone(),
        seed: args.seed,
        validation_seed: args.validation_seed,
        heldout_seed: args.heldout_seed,
        steps: args.steps,
        episodes_per_genome: args.episodes_per_genome,
        settings: NeatTrainerSettings::from(args),
        population,
        stage: Some(start_stage),
        stage_index,
        generation_in_stage: 0,
        validation_round: 0,
        qualification_streak: 0,
        stage_qualified: false,
        low_species_generations: 0,
        validations: Vec::new(),
        generation_reports: Vec::new(),
        stage_best: None,
        stage_archive: Vec::new(),
        stage_reports,
        repertoire,
        novelty_archive,
        world_archive,
        transfer_genome,
    })
}

fn migrate_neat_trainer_state_v2(
    mut state: NeatTrainerState,
    args: &NeatTrainArgs,
) -> NeatTrainerState {
    // Schema 2 index 3 was ExploreWithoutLooping. The inserted transition
    // stages deliberately restart learning at LeaveStartRegion while retaining
    // the complete evolutionary lineage and all historical evidence.
    if state.stage_index == 3 || state.stage == Some(CurriculumStage::ExploreWithoutLooping) {
        for founder in &state.stage_archive {
            if !state
                .repertoire
                .iter()
                .any(|entry| entry.genome == founder.genome)
            {
                state.repertoire.push(founder.clone());
            }
        }
        state.stage = Some(CurriculumStage::LeaveStartRegion);
        state.stage_index = CurriculumStage::ORDER
            .iter()
            .filter(|stage| stage.evolves_population())
            .position(|stage| *stage == CurriculumStage::LeaveStartRegion)
            .expect("transition stage is evolutionary");
        state.generation_in_stage = 0;
        state.qualification_streak = 0;
        state.stage_qualified = false;
        state.generation_reports.clear();
        state.stage_best = None;
        state.stage_archive.clear();
    } else {
        if let Some(stage) = state
            .stage
            .or_else(|| schema2_curriculum_stage(state.stage_index))
        {
            state.stage = Some(stage);
            state.stage_index = CurriculumStage::ORDER
                .iter()
                .filter(|candidate| candidate.evolves_population())
                .position(|candidate| *candidate == stage)
                .unwrap_or_default();
        }
    }
    state.schema_version = NEAT_TRAINER_STATE_SCHEMA_VERSION;
    // Episode count is evaluation policy, not population state. Adopt the safer
    // schema-3 minimum on migration without rebuilding genomes or archives.
    state.episodes_per_genome = args.episodes_per_genome;
    if state.settings.compatibility_threshold_floor <= 0.0 {
        state.settings.compatibility_threshold_floor = default_compatibility_threshold_floor();
    }
    state
}

fn schema2_curriculum_stage(index: usize) -> Option<CurriculumStage> {
    [
        CurriculumStage::BackAwayReliably,
        CurriculumStage::ChooseUsefulTurn,
        CurriculumStage::EscapeCorners,
        CurriculumStage::ExploreWithoutLooping,
        CurriculumStage::NavigateVariedRooms,
    ]
    .get(index)
    .copied()
}

fn parse_neat_stage(value: &str) -> Result<CurriculumStage> {
    CurriculumStage::ORDER
        .into_iter()
        .find(|stage| {
            neat_stage_slug(*stage) == value || format!("{stage:?}").eq_ignore_ascii_case(value)
        })
        .with_context(|| format!("unknown NEAT curriculum stage {value:?}"))
}

fn legacy_founder_genomes(
    report: &NeatTrainingReport,
    args: &NeatTrainArgs,
) -> Result<Vec<Genome>> {
    let mut founders = Vec::<Genome>::new();
    let add_unique = |founders: &mut Vec<Genome>, genome: Genome| {
        if !founders.contains(&genome) {
            founders.push(genome);
        }
    };
    for stage in &report.stages {
        if let Some(entry) = stage
            .niche_archive
            .iter()
            .max_by(|left, right| left.selection_fitness.total_cmp(&right.selection_fitness))
        {
            add_unique(&mut founders, entry.genome.clone());
        }
    }
    if let Some(active_path) = active_locomotion_model_checkpoint(Path::new(&args.models_config))? {
        if let Ok(checkpoint) = LocomotionCheckpoint::load(active_path) {
            add_unique(&mut founders, checkpoint.genome);
        }
    }
    if let Some(candidate_path) = report.promotion.candidate_artifact.as_deref() {
        if let Ok(checkpoint) = LocomotionCheckpoint::load(candidate_path) {
            add_unique(&mut founders, checkpoint.genome);
        }
    }

    let mut seen_cells = HashSet::new();
    let mut candidates = report.repertoire.iter().collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.selection_fitness.total_cmp(&left.selection_fitness));
    for entry in candidates {
        if founders.len() >= 16 {
            break;
        }
        if seen_cells.insert((neat_stage_slug(entry.stage), entry.niche, entry.descriptor)) {
            add_unique(&mut founders, entry.genome.clone());
        }
    }
    if founders.is_empty() {
        anyhow::bail!("founder report contains no recoverable genomes");
    }
    founders.truncate(16);
    Ok(founders)
}

#[allow(clippy::too_many_arguments)]
fn write_neat_trainer_state(
    path: &Path,
    args: &NeatTrainArgs,
    population: &Population,
    stage_index: usize,
    generation_in_stage: usize,
    validation_round: u64,
    qualification_streak: usize,
    stage_qualified: bool,
    low_species_generations: usize,
    validations: &[NeatStageValidationReport],
    generation_reports: &[NeatGenerationReport],
    stage_best: &Option<(f32, Genome, NeatPolicyEvaluation)>,
    stage_archive: &[NeatNicheArchiveEntry],
    stage_reports: &[NeatStageReport],
    repertoire: &[NeatNicheArchiveEntry],
    novelty_archive: &NoveltyArchive,
    world_archive: &[NeatWorldArchiveEntry],
    transfer_genome: &Genome,
) -> Result<()> {
    let state = NeatTrainerState {
        schema_version: NEAT_TRAINER_STATE_SCHEMA_VERSION,
        behavior: args.behavior.clone(),
        seed: args.seed,
        validation_seed: args.validation_seed,
        heldout_seed: args.heldout_seed,
        steps: args.steps,
        episodes_per_genome: args.episodes_per_genome,
        settings: NeatTrainerSettings::from(args),
        population: population.clone(),
        stage: CurriculumStage::ORDER
            .into_iter()
            .filter(|stage| stage.evolves_population())
            .nth(stage_index),
        stage_index,
        generation_in_stage,
        validation_round,
        qualification_streak,
        stage_qualified,
        low_species_generations,
        validations: validations.to_vec(),
        generation_reports: generation_reports.to_vec(),
        stage_best: stage_best.clone(),
        stage_archive: stage_archive.to_vec(),
        stage_reports: stage_reports.to_vec(),
        repertoire: repertoire.to_vec(),
        novelty_archive: novelty_archive.clone(),
        world_archive: world_archive.to_vec(),
        transfer_genome: transfer_genome.clone(),
    };
    write_json_report_atomic(path, &state)
}

fn write_json_report_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("writing {}", temporary.display()))?;
    fs::rename(&temporary, path).with_context(|| format!("replacing {}", path.display()))
}

