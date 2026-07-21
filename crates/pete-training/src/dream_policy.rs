
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use pete_actions::{
    action_to_motor_command, ActionPrimitive, ApproachTarget, InspectTarget, TurnDir,
};
use pete_autonomic::{SafetyConfig, SafetyDecision, SafetyLayer, SimpleSafety};
use pete_cockpit::{MotionCommand, MotorCommand};
use pete_now::Now;
use pete_sensors::World;
use pete_sim::{
    build_scenario, DreamConfig, ScenarioConfig, ScenarioKind, ScenarioWorld, SimObjectKind,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

pub const OBSERVATION_DIM: usize = 64;
pub const POLICY_OUTPUT_DIM: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DreamLevel {
    Motion = 0,
    ObstacleAvoidance = 1,
    EscapeTrap = 2,
    ChargerSeeking = 3,
    SocialInspection = 4,
    PlaceMemory = 5,
    WeirdDream = 6,
}

impl DreamLevel {
    pub fn id(self) -> u8 {
        self as u8
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Motion => "motion",
            Self::ObstacleAvoidance => "obstacle_avoidance",
            Self::EscapeTrap => "escape_trap",
            Self::ChargerSeeking => "charger_seeking",
            Self::SocialInspection => "social_inspection",
            Self::PlaceMemory => "place_memory",
            Self::WeirdDream => "weird_dream",
        }
    }

    pub fn config(self) -> DreamLevelConfig {
        match self {
            Self::Motion => DreamLevelConfig {
                level: self,
                episode_steps: 80,
                unlock_threshold: 18.0,
                success_score: 18.0,
                seeds_per_genome: 2,
            },
            Self::ObstacleAvoidance => DreamLevelConfig {
                level: self,
                episode_steps: 120,
                unlock_threshold: 24.0,
                success_score: 24.0,
                seeds_per_genome: 3,
            },
            Self::EscapeTrap => DreamLevelConfig {
                level: self,
                episode_steps: 140,
                unlock_threshold: 18.0,
                success_score: 18.0,
                seeds_per_genome: 3,
            },
            Self::ChargerSeeking => DreamLevelConfig {
                level: self,
                episode_steps: 160,
                unlock_threshold: 20.0,
                success_score: 20.0,
                seeds_per_genome: 3,
            },
            Self::SocialInspection => DreamLevelConfig {
                level: self,
                episode_steps: 140,
                unlock_threshold: 16.0,
                success_score: 16.0,
                seeds_per_genome: 3,
            },
            Self::PlaceMemory => DreamLevelConfig {
                level: self,
                episode_steps: 180,
                unlock_threshold: 28.0,
                success_score: 28.0,
                seeds_per_genome: 3,
            },
            Self::WeirdDream => DreamLevelConfig {
                level: self,
                episode_steps: 220,
                unlock_threshold: 30.0,
                success_score: 30.0,
                seeds_per_genome: 4,
            },
        }
    }

    pub fn next(self) -> Option<Self> {
        match self {
            Self::Motion => Some(Self::ObstacleAvoidance),
            Self::ObstacleAvoidance => Some(Self::EscapeTrap),
            Self::EscapeTrap => Some(Self::ChargerSeeking),
            Self::ChargerSeeking => Some(Self::SocialInspection),
            Self::SocialInspection => Some(Self::PlaceMemory),
            Self::PlaceMemory => Some(Self::WeirdDream),
            Self::WeirdDream => None,
        }
    }

    pub fn scenario_config(self, seed: u64) -> ScenarioConfig {
        match self {
            Self::Motion => ScenarioConfig::new(ScenarioKind::EmptyRoom, seed),
            Self::ObstacleAvoidance => ScenarioConfig::new(ScenarioKind::ObstacleAvoidance, seed),
            Self::EscapeTrap => {
                let kind = if seed % 2 == 0 {
                    ScenarioKind::CornerTrap
                } else {
                    ScenarioKind::ColumnTrap
                };
                ScenarioConfig::new(kind, seed)
            }
            Self::ChargerSeeking => ScenarioConfig::new(ScenarioKind::ChargerSeeking, seed),
            Self::SocialInspection => ScenarioConfig::new(ScenarioKind::PersonAndSpeaker, seed),
            Self::PlaceMemory => ScenarioConfig::new(ScenarioKind::MixedRoom, seed),
            Self::WeirdDream => {
                let mut config = ScenarioConfig::new(ScenarioKind::Dream, seed);
                let weirdness = 0.55 + ((seed % 41) as f32 / 100.0);
                let density = 0.45 + ((seed % 29) as f32 / 100.0);
                config.dream = Some(DreamConfig {
                    seed,
                    weirdness: weirdness.clamp(0.0, 1.0),
                    density: density.clamp(0.0, 1.0),
                    sociality: 0.35,
                    hazard_bias: 0.32,
                    charger_bias: 0.25,
                });
                config
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DreamLevelConfig {
    pub level: DreamLevel,
    pub episode_steps: usize,
    pub unlock_threshold: f32,
    pub success_score: f32,
    pub seeds_per_genome: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DreamPolicyMode {
    Conductor,
    Neat,
    Hybrid,
    Learned,
}

impl DreamPolicyMode {
    pub fn from_env() -> Self {
        match std::env::var("PETE_POLICY")
            .unwrap_or_else(|_| "conductor".to_string())
            .as_str()
        {
            "neat" => Self::Neat,
            "hybrid" => Self::Hybrid,
            "learned" => Self::Learned,
            _ => Self::Conductor,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DreamObservation {
    pub values: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DreamPolicyOutput {
    pub forward: f32,
    pub turn: f32,
    pub stop_probability: f32,
    pub inspect_probability: f32,
    pub dock_preference: f32,
}

impl DreamPolicyOutput {
    pub fn bounded(self) -> Self {
        Self {
            forward: self.forward.clamp(0.0, 1.0),
            turn: self.turn.clamp(-1.0, 1.0),
            stop_probability: self.stop_probability.clamp(0.0, 1.0),
            inspect_probability: self.inspect_probability.clamp(0.0, 1.0),
            dock_preference: self.dock_preference.clamp(0.0, 1.0),
        }
    }

    pub fn to_action(self, level: DreamLevel) -> ActionPrimitive {
        let output = self.bounded();
        if output.stop_probability > 0.72 {
            return ActionPrimitive::Stop;
        }
        if output.dock_preference > 0.70 && matches!(level, DreamLevel::ChargerSeeking) {
            return ActionPrimitive::Dock;
        }
        if output.inspect_probability > 0.66 {
            let target = if matches!(level, DreamLevel::SocialInspection) {
                InspectTarget::Sound
            } else {
                InspectTarget::Novelty
            };
            return ActionPrimitive::Inspect { target };
        }
        if output.dock_preference > 0.55 && matches!(level, DreamLevel::ChargerSeeking) {
            return ActionPrimitive::Approach {
                target: ApproachTarget::Charger,
            };
        }
        if output.forward > 0.12 {
            return ActionPrimitive::Go {
                intensity: (output.forward * 0.28).clamp(0.04, 0.28),
                duration_ms: 300,
            };
        }
        ActionPrimitive::Turn {
            direction: if output.turn >= 0.0 {
                TurnDir::Left
            } else {
                TurnDir::Right
            },
            intensity: output.turn.abs().max(0.12).min(0.45),
            duration_ms: 300,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DreamGenome {
    pub id: u64,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub input_hidden: Vec<f32>,
    pub hidden_output: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    pub output_bias: Vec<f32>,
    pub fitness: f32,
}

impl DreamGenome {
    pub fn random(id: u64, input_dim: usize, hidden_dim: usize, rng: &mut StdRng) -> Self {
        let mut weights = |len| {
            (0..len)
                .map(|_| rng.gen_range(-0.55..0.55))
                .collect::<Vec<f32>>()
        };
        Self {
            id,
            input_dim,
            hidden_dim,
            output_dim: POLICY_OUTPUT_DIM,
            input_hidden: weights(input_dim * hidden_dim),
            hidden_output: weights(hidden_dim * POLICY_OUTPUT_DIM),
            hidden_bias: weights(hidden_dim),
            output_bias: weights(POLICY_OUTPUT_DIM),
            fitness: 0.0,
        }
    }

    pub fn infer(&self, observation: &DreamObservation) -> DreamPolicyOutput {
        let mut hidden = vec![0.0; self.hidden_dim];
        for h in 0..self.hidden_dim {
            let mut sum = self.hidden_bias[h];
            for i in 0..self.input_dim {
                let input = observation.values.get(i).copied().unwrap_or(0.0);
                sum += input * self.input_hidden[h * self.input_dim + i];
            }
            hidden[h] = sum.tanh();
        }

        let mut raw = [0.0; POLICY_OUTPUT_DIM];
        for o in 0..POLICY_OUTPUT_DIM {
            let mut sum = self.output_bias[o];
            for h in 0..self.hidden_dim {
                sum += hidden[h] * self.hidden_output[o * self.hidden_dim + h];
            }
            raw[o] = sum;
        }

        DreamPolicyOutput {
            forward: sigmoid(raw[0]),
            turn: raw[1].tanh(),
            stop_probability: sigmoid(raw[2]),
            inspect_probability: sigmoid(raw[3]),
            dock_preference: sigmoid(raw[4]),
        }
        .bounded()
    }

    pub fn mutate(&mut self, rng: &mut StdRng, rate: f32, scale: f32) {
        mutate_weights(&mut self.input_hidden, rng, rate, scale);
        mutate_weights(&mut self.hidden_output, rng, rate, scale);
        mutate_weights(&mut self.hidden_bias, rng, rate, scale);
        mutate_weights(&mut self.output_bias, rng, rate, scale);
        if self.hidden_dim < 32 && rng.gen_bool(0.04) {
            self.add_hidden_node(rng);
        }
    }

    pub fn crossover(id: u64, left: &Self, right: &Self, rng: &mut StdRng) -> Self {
        if left.hidden_dim != right.hidden_dim
            || left.input_dim != right.input_dim
            || left.output_dim != right.output_dim
        {
            let mut child = if left.fitness >= right.fitness {
                left.clone()
            } else {
                right.clone()
            };
            child.id = id;
            child.fitness = 0.0;
            return child;
        }
        let choose = |a: &[f32], b: &[f32], rng: &mut StdRng| {
            a.iter()
                .zip(b)
                .map(|(left, right)| if rng.gen_bool(0.5) { *left } else { *right })
                .collect::<Vec<f32>>()
        };
        Self {
            id,
            input_dim: left.input_dim,
            hidden_dim: left.hidden_dim,
            output_dim: left.output_dim,
            input_hidden: choose(&left.input_hidden, &right.input_hidden, rng),
            hidden_output: choose(&left.hidden_output, &right.hidden_output, rng),
            hidden_bias: choose(&left.hidden_bias, &right.hidden_bias, rng),
            output_bias: choose(&left.output_bias, &right.output_bias, rng),
            fitness: 0.0,
        }
    }

    fn add_hidden_node(&mut self, rng: &mut StdRng) {
        let old_hidden_dim = self.hidden_dim;
        self.input_hidden.extend(
            (0..self.input_dim)
                .map(|_| rng.gen_range(-0.35..0.35))
                .collect::<Vec<f32>>(),
        );

        let mut expanded_hidden_output = Vec::with_capacity((old_hidden_dim + 1) * self.output_dim);
        for output_index in 0..self.output_dim {
            let start = output_index * old_hidden_dim;
            expanded_hidden_output
                .extend_from_slice(&self.hidden_output[start..start + old_hidden_dim]);
            expanded_hidden_output.push(rng.gen_range(-0.35..0.35));
        }
        self.hidden_output = expanded_hidden_output;
        self.hidden_bias.push(rng.gen_range(-0.20..0.20));
        self.hidden_dim += 1;
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RewardComponents {
    pub safe_motion: f32,
    pub coverage: f32,
    pub charger_progress: f32,
    pub social_alignment: f32,
    pub collision: f32,
    pub stuck: f32,
    pub idleness: f32,
    pub unsafe_proximity: f32,
    pub energy_time: f32,
}

impl RewardComponents {
    pub fn total(&self) -> f32 {
        self.safe_motion
            + self.coverage
            + self.charger_progress
            + self.social_alignment
            + self.collision
            + self.stuck
            + self.idleness
            + self.unsafe_proximity
            + self.energy_time
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DreamEpisodeRecord {
    pub observation: Vec<f32>,
    pub policy_output: DreamPolicyOutput,
    pub chosen_action: ActionPrimitive,
    pub reward_components: RewardComponents,
    pub next_observation: Vec<f32>,
    pub done: bool,
    pub success: bool,
    pub level: DreamLevel,
    pub seed: u64,
    pub safety_blocked_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DreamEpisodeReport {
    pub level: DreamLevel,
    pub seed: u64,
    pub genome_id: u64,
    pub score: f32,
    pub success: bool,
    pub steps: usize,
    pub covered_cells: usize,
    pub collisions: usize,
    pub stuck_steps: usize,
    pub last_output: DreamPolicyOutput,
    pub last_reward: RewardComponents,
    pub records: Vec<DreamEpisodeRecord>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DreamTrainingConfig {
    pub population_size: usize,
    pub generations: usize,
    pub base_seed: u64,
    pub start_level: DreamLevel,
    pub hidden_dim: usize,
    pub checkpoint_dir: PathBuf,
    pub dataset_dir: PathBuf,
    pub export_dataset: bool,
    pub detailed_logs: bool,
}

impl Default for DreamTrainingConfig {
    fn default() -> Self {
        Self {
            population_size: 32,
            generations: 30,
            base_seed: 7,
            start_level: DreamLevel::Motion,
            hidden_dim: 12,
            checkpoint_dir: PathBuf::from("data/models/dream-policy/neat"),
            dataset_dir: PathBuf::from("datasets/dream-policy/v0/episodes"),
            export_dataset: true,
            detailed_logs: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DreamTrainingStatus {
    pub current_level: DreamLevel,
    pub generation: usize,
    pub episode_score: f32,
    pub best_score: f32,
    pub selected_genome_id: u64,
    pub current_policy_output: DreamPolicyOutput,
    pub reward_components: RewardComponents,
    pub blocked_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DreamTrainingReport {
    pub status: DreamTrainingStatus,
    pub best_checkpoint: PathBuf,
    pub dataset_dir: PathBuf,
    pub unlocked_levels: Vec<DreamLevel>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DreamPolicyCheckpoint {
    pub schema_version: u32,
    pub level: DreamLevel,
    pub generation: usize,
    pub best_score: f32,
    pub genome: DreamGenome,
}

fn format_count(value: u64) -> String {
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

fn render_status_bar(current: usize, total: usize, width: usize) -> String {
    let total = total.max(1);
    let current = current.min(total);
    let filled = current.saturating_mul(width) / total;
    let mut bar = String::with_capacity(width);
    for _ in 0..filled {
        bar.push('#');
    }
    for _ in filled..width {
        bar.push('-');
    }
    bar
}

pub async fn train_dream_policy(config: DreamTrainingConfig) -> Result<DreamTrainingReport> {
    fs::create_dir_all(&config.checkpoint_dir).with_context(|| {
        format!(
            "failed to create checkpoint dir {}",
            config.checkpoint_dir.display()
        )
    })?;
    if config.export_dataset {
        fs::create_dir_all(&config.dataset_dir).with_context(|| {
            format!(
                "failed to create dataset dir {}",
                config.dataset_dir.display()
            )
        })?;
    }

    let mut rng = StdRng::seed_from_u64(config.base_seed);
    let population_size = config.population_size.max(4);
    let hidden_dim = config.hidden_dim.clamp(4, 32);
    let mut population = (0..population_size)
        .map(|index| DreamGenome::random(index as u64, OBSERVATION_DIM, hidden_dim, &mut rng))
        .collect::<Vec<_>>();
    let mut next_genome_id = population_size as u64;
    let mut level = config.start_level;
    let mut unlocked = vec![level];
    let mut best_checkpoint = config.checkpoint_dir.join("best.json");
    let mut best_score = f32::NEG_INFINITY;
    let mut best_draft_ratio = f32::NEG_INFINITY;
    let mut total_episodes_run = 0u64;
    let mut total_records_exported = 0u64;
    let started_at = std::time::Instant::now();
    let mut last_status = DreamTrainingStatus {
        current_level: level,
        generation: 0,
        episode_score: 0.0,
        best_score,
        selected_genome_id: 0,
        current_policy_output: DreamPolicyOutput {
            forward: 0.0,
            turn: 0.0,
            stop_probability: 1.0,
            inspect_probability: 0.0,
            dock_preference: 0.0,
        },
        reward_components: RewardComponents::default(),
        blocked_reason: None,
    };

    if config.detailed_logs {
        println!(
                "dream-train start: level={} generations={} population={} hidden_dim={} seed={} checkpoint_dir={} dataset_dir={} export_dataset={}",
                level.name(),
                format_count(config.generations as u64),
                format_count(population_size as u64),
                hidden_dim,
                config.base_seed,
                config.checkpoint_dir.display(),
                config.dataset_dir.display(),
                config.export_dataset,
            );
    }

    for generation in 0..config.generations {
        let generation_started_at = std::time::Instant::now();
        let level_config = level.config();
        let mut scored = Vec::with_capacity(population.len());
        for mut genome in population {
            let mut score = 0.0;
            let mut reports = Vec::new();
            for seed_index in 0..level_config.seeds_per_genome {
                let seed = config
                    .base_seed
                    .wrapping_add((generation as u64) * 10_000)
                    .wrapping_add((seed_index as u64) * 37)
                    .wrapping_add(genome.id);
                total_episodes_run = total_episodes_run.saturating_add(1);
                let report =
                    evaluate_genome_episode(&genome, level, seed, level_config.episode_steps)
                        .await?;
                score += report.score;
                reports.push(report);
            }
            genome.fitness = score / level_config.seeds_per_genome as f32;
            if config.export_dataset {
                for report in &reports {
                    export_episode_records(report, &config.dataset_dir).await?;
                    total_records_exported =
                        total_records_exported.saturating_add(report.records.len() as u64);
                }
            }
            let fitness_ratio = genome.fitness / level_config.unlock_threshold.max(0.001);
            if fitness_ratio > best_draft_ratio {
                best_draft_ratio = fitness_ratio;
                best_checkpoint = config
                    .checkpoint_dir
                    .join(format!("level-{}-best.json", level.id()));
            }
            if genome.fitness > best_score {
                best_score = genome.fitness;
                let checkpoint = DreamPolicyCheckpoint {
                    schema_version: 1,
                    level,
                    generation,
                    best_score,
                    genome: genome.clone(),
                };
                let level_best_checkpoint = config
                    .checkpoint_dir
                    .join(format!("level-{}-best.json", level.id()));
                save_best_genome(&checkpoint, &level_best_checkpoint)?;
            }
            scored.push(genome);
        }

        scored.sort_by(|left, right| {
            right
                .fitness
                .partial_cmp(&left.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let best = scored[0].clone();
        let worst_fitness = scored
            .last()
            .map(|genome| genome.fitness)
            .unwrap_or(best.fitness);
        let mean_fitness =
            scored.iter().map(|genome| genome.fitness).sum::<f32>() / scored.len() as f32;
        let probe =
            evaluate_genome_episode(&best, level, config.base_seed ^ generation as u64, 24).await?;
        last_status = DreamTrainingStatus {
            current_level: level,
            generation,
            episode_score: probe.score,
            best_score,
            selected_genome_id: best.id,
            current_policy_output: probe.last_output,
            reward_components: probe.last_reward,
            blocked_reason: probe
                .records
                .last()
                .and_then(|r| r.safety_blocked_reason.clone()),
        };

        if config.detailed_logs {
            let generation_number = generation.saturating_add(1);
            let bar = render_status_bar(generation_number, config.generations, 28);
            println!(
                    "[{bar}] gen {}/{} level={} best_id={} best={:.3} avg={:.3} min={:.3} unlock={:.3} probe={:.3} episodes={} exported_records={} gen_s={:.2} total_s={:.2}",
                    format_count(generation_number as u64),
                    format_count(config.generations as u64),
                    level.name(),
                    best.id,
                    best.fitness,
                    mean_fitness,
                    worst_fitness,
                    level_config.unlock_threshold,
                    probe.score,
                    format_count(total_episodes_run),
                    format_count(total_records_exported),
                    generation_started_at.elapsed().as_secs_f32(),
                    started_at.elapsed().as_secs_f32(),
                );
        }

        if best.fitness >= level_config.unlock_threshold {
            if let Some(next) = level.next() {
                if config.detailed_logs {
                    println!(
                        "level unlocked: {} -> {} (fitness {:.3} >= {:.3})",
                        level.name(),
                        next.name(),
                        best.fitness,
                        level_config.unlock_threshold,
                    );
                }
                level = next;
                best_score = f32::NEG_INFINITY;
                unlocked.push(level);
            }
        }

        population = breed_next_population(
            &scored,
            population_size,
            &mut next_genome_id,
            &mut rng,
            generation,
        );
    }

    if config.detailed_logs {
        println!(
            "dream-train done: generations={} episodes={} exported_records={} elapsed_s={:.2}",
            format_count(config.generations as u64),
            format_count(total_episodes_run),
            format_count(total_records_exported),
            started_at.elapsed().as_secs_f32(),
        );
    }

    Ok(DreamTrainingReport {
        status: last_status,
        best_checkpoint,
        dataset_dir: config.dataset_dir,
        unlocked_levels: unlocked,
    })
}

pub fn save_best_genome(checkpoint: &DreamPolicyCheckpoint, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(checkpoint)?)?;
    Ok(())
}

pub fn load_best_genome(path: impl AsRef<Path>) -> Result<DreamPolicyCheckpoint> {
    let path = path.as_ref();
    let bytes = fs::read(path)
        .with_context(|| format!("failed to read genome checkpoint {}", path.display()))?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub async fn evaluate_genome_episode(
    genome: &DreamGenome,
    level: DreamLevel,
    seed: u64,
    max_steps: usize,
) -> Result<DreamEpisodeReport> {
    let scenario = build_scenario(level.scenario_config(seed));
    run_episode(genome, level, seed, max_steps, scenario).await
}

async fn run_episode(
    genome: &DreamGenome,
    level: DreamLevel,
    seed: u64,
    max_steps: usize,
    mut scenario: ScenarioWorld,
) -> Result<DreamEpisodeReport> {
    let mut safety = SimpleSafety {
        config: SafetyConfig {
            stale_sensor_ms: 5_000,
            ..SafetyConfig::default()
        },
    };
    let mut covered = BTreeSet::new();
    let mut records = Vec::new();
    let mut previous_action = MotorCommand::stop();
    let mut previous_distance_to_charger = None;
    let mut previous_pose = None;
    let mut score = 0.0;
    let mut collisions = 0;
    let mut stuck_steps = 0;
    let mut last_output = DreamPolicyOutput {
        forward: 0.0,
        turn: 0.0,
        stop_probability: 1.0,
        inspect_probability: 0.0,
        dock_preference: 0.0,
    };
    let mut last_reward = RewardComponents::default();

    for step in 0..max_steps {
        let snapshot = scenario.world.snapshot().await?;
        let now = snapshot.to_now(snapshot.body.last_update_ms);
        let observation = build_observation(
            &now,
            &scenario,
            level,
            previous_action,
            previous_pose,
            &covered,
        );
        let output = genome.infer(&observation);
        let action = output.to_action(level);
        let desired = action_to_motor_command(Some(&action));
        let decision = safety.filter(&now, desired);
        let before_pose = now.body.odometry;
        scenario.motors.apply_motion(MotionCommand::Drive {
            forward_m_s: decision.command.forward,
            turn_rad_s: decision.command.turn,
        })?;
        let next_snapshot = scenario.world.snapshot().await?;
        let next_now = next_snapshot.to_now(next_snapshot.body.last_update_ms);
        let next_observation = build_observation(
            &next_now,
            &scenario,
            level,
            decision.command,
            Some(before_pose),
            &covered,
        );
        let reward = score_step(
            level,
            &now,
            &next_now,
            &scenario,
            desired,
            decision.command,
            &mut covered,
            &mut previous_distance_to_charger,
        );
        if next_now.body.flags.bump_left
            || next_now.body.flags.bump_right
            || next_now.body.flags.wall
        {
            collisions += 1;
        }
        if reward.stuck < 0.0 {
            stuck_steps += 1;
        }
        score += reward.total();
        let done = step + 1 == max_steps;
        let success = score >= level.config().success_score;
        last_output = output;
        last_reward = reward.clone();
        records.push(DreamEpisodeRecord {
            observation: observation.values,
            policy_output: output,
            chosen_action: action,
            reward_components: reward,
            next_observation: next_observation.values,
            done,
            success,
            level,
            seed,
            safety_blocked_reason: safety_block_reason(&decision),
        });
        previous_pose = Some(next_now.body.odometry);
        previous_action = decision.command;
        if done || success {
            break;
        }
    }

    Ok(DreamEpisodeReport {
        level,
        seed,
        genome_id: genome.id,
        score,
        success: score >= level.config().success_score,
        steps: records.len(),
        covered_cells: covered.len(),
        collisions,
        stuck_steps,
        last_output,
        last_reward,
        records,
    })
}

pub fn build_observation(
    now: &Now,
    scenario: &ScenarioWorld,
    level: DreamLevel,
    previous_action: MotorCommand,
    previous_pose: Option<pete_core::Pose2>,
    covered: &BTreeSet<(i16, i16)>,
) -> DreamObservation {
    let mut values = Vec::with_capacity(OBSERVATION_DIM);
    for beam in now.range.beams.iter().take(16) {
        values.push(beam.clamp(0.0, 1.0));
    }
    while values.len() < 16 {
        values.push(1.0);
    }
    values.extend([
        bool01(now.body.flags.bump_left),
        bool01(now.body.flags.bump_right),
        bool01(now.body.flags.cliff_left || now.body.flags.cliff_front_left),
        bool01(now.body.flags.cliff_right || now.body.flags.cliff_front_right),
        bool01(now.body.flags.wheel_drop),
        now.body.battery_level.clamp(0.0, 1.0),
        bool01(now.body.charging),
        now.body.velocity.forward_m_s.clamp(-0.5, 0.5) / 0.5,
        now.body.velocity.turn_rad_s.clamp(-1.0, 1.0),
    ]);
    let (charger_dir, charger_value) = target_direction_and_value(now, scenario, Target::Charger);
    let (social_dir, social_value) = target_direction_and_value(now, scenario, Target::Social);
    values.extend([
        charger_dir.sin(),
        charger_dir.cos(),
        charger_value,
        social_dir.sin(),
        social_dir.cos(),
        social_value,
        novelty_value(now, covered),
        familiarity_value(now, covered),
        previous_action.forward.clamp(-0.5, 0.5) / 0.5,
        previous_action.turn.clamp(-1.0, 1.0),
        stuck_signal(now, previous_pose),
        level.id() as f32 / 6.0,
    ]);
    append_now_latent(now, &mut values, 24);
    values.resize(OBSERVATION_DIM, 0.0);
    DreamObservation { values }
}

fn score_step(
    level: DreamLevel,
    before: &Now,
    after: &Now,
    scenario: &ScenarioWorld,
    desired: MotorCommand,
    applied: MotorCommand,
    covered: &mut BTreeSet<(i16, i16)>,
    previous_distance_to_charger: &mut Option<f32>,
) -> RewardComponents {
    let distance = pose_distance(before.body.odometry, after.body.odometry);
    let collision =
        after.body.flags.bump_left || after.body.flags.bump_right || after.body.flags.wall;
    let cell = coverage_cell(after);
    let new_cell = covered.insert(cell);
    let charger_distance = nearest_target_distance(after, scenario, Target::Charger);
    let charger_progress = if matches!(level, DreamLevel::ChargerSeeking) {
        let previous = previous_distance_to_charger.replace(charger_distance.unwrap_or(99.0));
        previous
            .zip(charger_distance)
            .map(|(p, c)| (p - c).clamp(-0.2, 0.2) * 8.0)
            .unwrap_or(0.0)
    } else {
        0.0
    };
    let social_alignment = if matches!(level, DreamLevel::SocialInspection) {
        let (dir, value) = target_direction_and_value(before, scenario, Target::Social);
        let turning_toward = if dir.sin().signum() == applied.turn.signum() {
            applied.turn.abs().min(1.0)
        } else {
            -applied.turn.abs().min(1.0)
        };
        turning_toward * value
    } else {
        0.0
    };
    RewardComponents {
        safe_motion: if !collision && distance > 0.01 {
            distance * 8.0
        } else {
            0.0
        },
        coverage: if new_cell { 0.65 } else { 0.0 },
        charger_progress,
        social_alignment,
        collision: if collision { -3.0 } else { 0.0 },
        stuck: if desired.forward.abs() > 0.04 && distance < 0.003 {
            -0.65
        } else {
            0.0
        },
        idleness: if desired.forward.abs() < 0.02 && desired.turn.abs() < 0.03 {
            -0.30
        } else {
            0.0
        },
        unsafe_proximity: before
            .range
            .nearest_m
            .map(|nearest| if nearest < 0.25 { -0.8 } else { 0.0 })
            .unwrap_or(0.0),
        energy_time: -0.015 - (applied.forward.abs() + applied.turn.abs()) * 0.01,
    }
}

async fn export_episode_records(report: &DreamEpisodeReport, dataset_dir: &Path) -> Result<()> {
    fs::create_dir_all(dataset_dir)?;
    let path = dataset_dir.join(format!(
        "level-{}-seed-{}-genome-{}.jsonl",
        report.level.id(),
        report.seed,
        report.genome_id
    ));
    let mut file = tokio::fs::File::create(path).await?;
    for record in &report.records {
        file.write_all(serde_json::to_string(record)?.as_bytes())
            .await?;
        file.write_all(b"\n").await?;
    }
    Ok(())
}

fn breed_next_population(
    scored: &[DreamGenome],
    population_size: usize,
    next_genome_id: &mut u64,
    rng: &mut StdRng,
    generation: usize,
) -> Vec<DreamGenome> {
    let elite_count = (population_size / 5).max(2).min(scored.len());
    let mut next = scored.iter().take(elite_count).cloned().collect::<Vec<_>>();
    while next.len() < population_size {
        let left = tournament(scored, rng);
        let right = tournament(scored, rng);
        let mut child = DreamGenome::crossover(*next_genome_id, left, right, rng);
        *next_genome_id += 1;
        let rate = 0.08 + (generation as f32 * 0.001).min(0.07);
        child.mutate(rng, rate, 0.22);
        next.push(child);
    }
    next
}

fn tournament<'a>(scored: &'a [DreamGenome], rng: &mut StdRng) -> &'a DreamGenome {
    let mut best = &scored[rng.gen_range(0..scored.len().min(8))];
    for _ in 0..2 {
        let candidate = &scored[rng.gen_range(0..scored.len())];
        if candidate.fitness > best.fitness {
            best = candidate;
        }
    }
    best
}

fn mutate_weights(weights: &mut [f32], rng: &mut StdRng, rate: f32, scale: f32) {
    for weight in weights {
        if rng.gen_bool(rate as f64) {
            *weight = (*weight + rng.gen_range(-scale..scale)).clamp(-4.0, 4.0);
        }
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn bool01(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

fn append_now_latent(now: &Now, values: &mut Vec<f32>, max: usize) {
    for vector in now
        .eye
        .scene_vectors
        .iter()
        .map(|artifact| &artifact.vector)
    {
        for value in vector.iter().take(max) {
            values.push(value.clamp(-1.0, 1.0));
            if values.len() >= OBSERVATION_DIM {
                return;
            }
        }
    }
    for frame in now.eye.frames.iter() {
        for value in frame.iter().take(max) {
            values.push((*value / 3.0).clamp(0.0, 1.0));
            if values.len() >= OBSERVATION_DIM {
                return;
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Target {
    Charger,
    Social,
}

fn target_direction_and_value(now: &Now, scenario: &ScenarioWorld, target: Target) -> (f32, f32) {
    let heading = now.body.odometry.heading_rad;
    scenario
        .metadata
        .objects
        .iter()
        .filter(|object| match target {
            Target::Charger => matches!(object.kind, SimObjectKind::Charger),
            Target::Social => {
                object.emits_sound || matches!(object.kind, SimObjectKind::Person { .. })
            }
        })
        .map(|object| {
            let dx = object.x_m - now.body.odometry.x_m;
            let dy = object.y_m - now.body.odometry.y_m;
            let distance = (dx * dx + dy * dy).sqrt().max(0.001);
            let angle = (dy.atan2(dx) - heading + std::f32::consts::PI)
                .rem_euclid(std::f32::consts::TAU)
                - std::f32::consts::PI;
            (angle, (1.0 - distance / 6.0).clamp(0.0, 1.0))
        })
        .max_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or((0.0, 0.0))
}

fn nearest_target_distance(now: &Now, scenario: &ScenarioWorld, target: Target) -> Option<f32> {
    scenario
        .metadata
        .objects
        .iter()
        .filter(|object| match target {
            Target::Charger => matches!(object.kind, SimObjectKind::Charger),
            Target::Social => {
                object.emits_sound || matches!(object.kind, SimObjectKind::Person { .. })
            }
        })
        .map(|object| {
            let dx = object.x_m - now.body.odometry.x_m;
            let dy = object.y_m - now.body.odometry.y_m;
            (dx * dx + dy * dy).sqrt()
        })
        .min_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
}

fn novelty_value(now: &Now, covered: &BTreeSet<(i16, i16)>) -> f32 {
    if covered.contains(&coverage_cell(now)) {
        0.0
    } else {
        1.0
    }
}

fn familiarity_value(now: &Now, covered: &BTreeSet<(i16, i16)>) -> f32 {
    1.0 - novelty_value(now, covered)
}

fn stuck_signal(now: &Now, previous_pose: Option<pete_core::Pose2>) -> f32 {
    previous_pose
        .map(|previous| {
            if pose_distance(previous, now.body.odometry) < 0.005
                && now.body.velocity.forward_m_s.abs() < 0.02
            {
                1.0
            } else {
                0.0
            }
        })
        .unwrap_or(0.0)
}

fn coverage_cell(now: &Now) -> (i16, i16) {
    (
        (now.body.odometry.x_m / 0.5).floor() as i16,
        (now.body.odometry.y_m / 0.5).floor() as i16,
    )
}

fn pose_distance(left: pete_core::Pose2, right: pete_core::Pose2) -> f32 {
    let dx = left.x_m - right.x_m;
    let dy = left.y_m - right.y_m;
    (dx * dx + dy * dy).sqrt()
}

fn safety_block_reason(decision: &SafetyDecision) -> Option<String> {
    if decision.vetoed {
        decision.reason.as_ref().map(|reason| format!("{reason:?}"))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pete_body::BodySense;

    #[test]
    fn level_generation_is_deterministic_by_seed() {
        let left = build_scenario(DreamLevel::WeirdDream.scenario_config(42));
        let right = build_scenario(DreamLevel::WeirdDream.scenario_config(42));
        assert_eq!(left.metadata.arena, right.metadata.arena);
        assert_eq!(left.metadata.objects, right.metadata.objects);
    }

    #[test]
    fn output_is_bounded() {
        let output = DreamPolicyOutput {
            forward: 9.0,
            turn: -8.0,
            stop_probability: 2.0,
            inspect_probability: -1.0,
            dock_preference: 3.0,
        }
        .bounded();
        assert_eq!(output.forward, 1.0);
        assert_eq!(output.turn, -1.0);
        assert_eq!(output.stop_probability, 1.0);
        assert_eq!(output.inspect_probability, 0.0);
        assert_eq!(output.dock_preference, 1.0);
    }

    #[test]
    fn reward_penalizes_sitting_still() {
        let scenario = build_scenario(DreamLevel::Motion.scenario_config(1));
        let mut body = BodySense::default();
        body.last_update_ms = 1;
        let before = Now::blank(1, body.clone());
        let after = Now::blank(2, body);
        let mut covered = BTreeSet::new();
        covered.insert(coverage_cell(&before));
        let mut charger = None;
        let reward = score_step(
            DreamLevel::Motion,
            &before,
            &after,
            &scenario,
            MotorCommand::stop(),
            MotorCommand::stop(),
            &mut covered,
            &mut charger,
        );
        assert!(reward.idleness < 0.0);
        assert!(reward.total() < 0.0);
    }

    #[test]
    fn reward_penalizes_collision() {
        let scenario = build_scenario(DreamLevel::ObstacleAvoidance.scenario_config(2));
        let body = BodySense::default();
        let before = Now::blank(1, body.clone());
        let mut after_body = body;
        after_body.flags.bump_left = true;
        let after = Now::blank(2, after_body);
        let mut covered = BTreeSet::new();
        let mut charger = None;
        let reward = score_step(
            DreamLevel::ObstacleAvoidance,
            &before,
            &after,
            &scenario,
            MotorCommand {
                forward: 0.2,
                turn: 0.0,
            },
            MotorCommand::stop(),
            &mut covered,
            &mut charger,
        );
        assert!(reward.collision < -1.0);
    }

    #[test]
    fn reward_gives_coverage_bonus() {
        let scenario = build_scenario(DreamLevel::Motion.scenario_config(3));
        let mut before_body = BodySense::default();
        before_body.odometry.x_m = 1.0;
        before_body.odometry.y_m = 1.0;
        let mut after_body = before_body.clone();
        after_body.odometry.x_m = 1.6;
        let before = Now::blank(1, before_body);
        let after = Now::blank(2, after_body);
        let mut covered = BTreeSet::new();
        let mut charger = None;
        let reward = score_step(
            DreamLevel::Motion,
            &before,
            &after,
            &scenario,
            MotorCommand {
                forward: 0.2,
                turn: 0.0,
            },
            MotorCommand {
                forward: 0.2,
                turn: 0.0,
            },
            &mut covered,
            &mut charger,
        );
        assert!(reward.coverage > 0.0);
    }

    #[test]
    fn safety_gate_prevents_unsafe_forward() {
        let mut safety = SimpleSafety::default();
        let mut body = BodySense::default();
        body.flags.wheel_drop = true;
        body.last_update_ms = 10;
        let now = Now::blank(10, body);
        let decision = safety.filter(
            &now,
            MotorCommand {
                forward: 0.2,
                turn: 0.0,
            },
        );
        assert!(decision.vetoed);
        assert_eq!(decision.command, MotorCommand::stop());
    }

    #[test]
    fn saved_best_genome_can_be_reloaded() {
        let mut rng = StdRng::seed_from_u64(4);
        let genome = DreamGenome::random(99, OBSERVATION_DIM, 4, &mut rng);
        let checkpoint = DreamPolicyCheckpoint {
            schema_version: 1,
            level: DreamLevel::Motion,
            generation: 2,
            best_score: 12.0,
            genome,
        };
        let path =
            std::env::temp_dir().join(format!("pete-dream-genome-{}.json", std::process::id()));
        save_best_genome(&checkpoint, &path).unwrap();
        let loaded = load_best_genome(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(loaded.level, DreamLevel::Motion);
        assert_eq!(loaded.genome.id, 99);
    }
}
