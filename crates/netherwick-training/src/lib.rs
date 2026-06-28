use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use netherwick_behaviors::{
    BehaviorConfig, BehaviorRegime, BehaviorRegistryConfig, FallbackPolicy,
};
use netherwick_core::TimeMs;
use netherwick_experience::{
    action_value_input_from_transition_like, action_value_target_from_reward_surprise,
    charge_input_from_transition_like, charge_target_from_transition_like,
    danger_input_from_transition_like, danger_target_from_transition_like,
    ear_next_input_from_transition_like, ear_next_target_from_now,
    experience_decode_target_from_now, experience_encode_input_from_now,
    eye_next_input_from_transition_like, eye_next_target_from_now, ActionValueInput,
    ActionValueTarget, ChargeInput, ChargeTarget, CodebookQuantizer, CodebookUsageReport,
    DangerInput, DangerTarget, EarNextInput, EarNextTarget, ExperienceDecodeOutput,
    ExperienceEncodeInput, ExperienceLatent, EyeNextInput, EyeNextTarget, FutureInput,
    FuturePredictor, LatentEncoder, RandomProjectionExperienceEncoder, StasisFuturePredictor,
};
use netherwick_ledger::{
    future_input_from_transition, future_target_from_transition, ExperienceTransition, JsonlLedger,
};
use netherwick_models::{
    ActionValueNetTrainer, ChargeNetTrainer, DangerNetTrainer, EarNextNetTrainer,
    ExperienceAutoencoderTrainer, EyeNextNetTrainer, FutureNetTrainer,
    HardcodedActionValuePredictor, HardcodedChargePredictor, HardcodedDangerPredictor, TrainStats,
};
use netherwick_now::Now;
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;

pub mod dream_policy {
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::{Path, PathBuf};

    use anyhow::{Context, Result};
    use netherwick_actions::{
        action_to_motor_command, ActionPrimitive, ApproachTarget, InspectTarget, TurnDir,
    };
    use netherwick_autonomic::{SafetyConfig, SafetyDecision, SafetyLayer, SimpleSafety};
    use netherwick_body::{MotionCommand, MotorCommand, MotorComplex};
    use netherwick_now::Now;
    use netherwick_sensors::World;
    use netherwick_sim::{
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
                Self::ObstacleAvoidance => {
                    ScenarioConfig::new(ScenarioKind::ObstacleAvoidance, seed)
                }
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
            match std::env::var("NETHERWICK_POLICY")
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

            let mut expanded_hidden_output =
                Vec::with_capacity((old_hidden_dim + 1) * self.output_dim);
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
                evaluate_genome_episode(&best, level, config.base_seed ^ generation as u64, 24)
                    .await?;
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
            scenario
                .motors
                .send(MotionCommand::Drive {
                    forward_m_s: decision.command.forward,
                    turn_rad_s: decision.command.turn,
                })
                .await?;
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
        previous_pose: Option<netherwick_core::Pose2>,
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
        let (charger_dir, charger_value) =
            target_direction_and_value(now, scenario, Target::Charger);
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

    fn target_direction_and_value(
        now: &Now,
        scenario: &ScenarioWorld,
        target: Target,
    ) -> (f32, f32) {
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

    fn stuck_signal(now: &Now, previous_pose: Option<netherwick_core::Pose2>) -> f32 {
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

    fn pose_distance(left: netherwick_core::Pose2, right: netherwick_core::Pose2) -> f32 {
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
        use netherwick_body::BodySense;

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
            let path = std::env::temp_dir().join(format!(
                "netherwick-dream-genome-{}.json",
                std::process::id()
            ));
            save_best_genome(&checkpoint, &path).unwrap();
            let loaded = load_best_genome(&path).unwrap();
            let _ = fs::remove_file(&path);
            assert_eq!(loaded.level, DreamLevel::Motion);
            assert_eq!(loaded.genome.id, 99);
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TrainableBehavior {
    Danger,
    Charge,
    ActionValue,
    EyeNext,
    EarNext,
    Experience,
    Future,
}

impl TrainableBehavior {
    pub fn config_key(&self) -> &'static str {
        match self {
            Self::Danger => "danger",
            Self::Charge => "charge",
            Self::ActionValue => "action_value",
            Self::EyeNext => "eye_next",
            Self::EarNext => "ear_next",
            Self::Experience => "experience",
            Self::Future => "future",
        }
    }

    pub fn cli_name(&self) -> &'static str {
        match self {
            Self::Danger => "danger",
            Self::Charge => "charge",
            Self::ActionValue => "action-value",
            Self::EyeNext => "eye-next",
            Self::EarNext => "ear-next",
            Self::Experience => "experience",
            Self::Future => "future",
        }
    }

    pub fn default_model_id(&self) -> &'static str {
        match self {
            Self::Danger => "danger.burn.v0",
            Self::Charge => "charge.burn.v0",
            Self::ActionValue => "action_value.burn.v0",
            Self::EyeNext => "eye.burn.next_v0",
            Self::EarNext => "ear.burn.next_v0",
            Self::Experience => "experience.autoencoder.v0",
            Self::Future => "future.burn.v0",
        }
    }

    pub fn default_hardcoded_id(&self) -> &'static str {
        match self {
            Self::Danger => "danger.range_bumper",
            Self::Charge => "charge.sensor_battery_delta",
            Self::ActionValue => "action_value.handcoded",
            Self::EyeNext => "eye.copy_current",
            Self::EarNext => "ear.copy_current",
            Self::Experience => "experience.feature_encoder",
            Self::Future => "future.stasis",
        }
    }

    fn is_safety_critical(&self) -> bool {
        matches!(
            self,
            Self::Danger | Self::ActionValue | Self::Experience | Self::Future
        )
    }
}

impl fmt::Display for TrainableBehavior {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.cli_name())
    }
}

impl FromStr for TrainableBehavior {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "danger" => Ok(Self::Danger),
            "charge" => Ok(Self::Charge),
            "action-value" | "action_value" => Ok(Self::ActionValue),
            "eye-next" | "eye_next" => Ok(Self::EyeNext),
            "ear-next" | "ear_next" => Ok(Self::EarNext),
            "experience" => Ok(Self::Experience),
            "future" => Ok(Self::Future),
            other => bail!("unknown trainable behavior {other:?}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainBehaviorRequest {
    pub behavior: TrainableBehavior,
    pub ledger_path: PathBuf,
    pub checkpoint_path: PathBuf,
    pub epochs: usize,
    pub validation_split: f32,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct EvaluateBehaviorRequest {
    pub behavior: TrainableBehavior,
    pub ledger_path: PathBuf,
    pub checkpoint_path: PathBuf,
    pub max_samples: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BehaviorEvaluationReport {
    pub behavior: TrainableBehavior,
    pub checkpoint_path: PathBuf,
    pub sample_count: usize,
    pub model_loss_mean: f32,
    pub hardcoded_loss_mean: Option<f32>,
    pub selected_loss_mean: Option<f32>,
    pub model_better_than_hardcoded: Option<bool>,
    pub improvement_ratio: Option<f32>,
    pub warnings: Vec<String>,
    pub recommendation: PromotionRecommendation,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PromotionRecommendation {
    KeepHardcoded,
    ShadowInfer,
    ShadowTrain,
    PromoteToModelInfer,
    RejectCheckpoint,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BehaviorMetricRecord {
    pub t_ms: TimeMs,
    pub behavior: TrainableBehavior,
    pub epoch: usize,
    pub sample_index: usize,
    pub train_loss: Option<f32>,
    pub eval_loss: Option<f32>,
    pub hardcoded_loss: Option<f32>,
    pub model_loss: Option<f32>,
    pub selected_loss: Option<f32>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainSummary {
    pub behavior: TrainableBehavior,
    pub transition_count: usize,
    pub train_sample_count: usize,
    pub eval_sample_count: usize,
    pub epochs: usize,
    pub samples_seen: u64,
    pub last_loss: Option<f32>,
    pub best_loss: Option<f32>,
    pub metrics_path: PathBuf,
    pub checkpoint_path: PathBuf,
    pub evaluation: BehaviorEvaluationReport,
}

#[derive(Clone, Debug)]
pub struct TrainLatentRoundTripRequest {
    pub ledger_path: PathBuf,
    pub checkpoint_path: PathBuf,
    pub report_path: PathBuf,
    pub epochs: usize,
    pub validation_split: f32,
    pub seed: u64,
    pub z_dim: usize,
    pub codebook_size: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainLatentRoundTripReport {
    pub schema_version: u32,
    pub transition_count: usize,
    pub train_transition_count: usize,
    pub eval_transition_count: usize,
    pub epochs: usize,
    pub z_dim: usize,
    pub checkpoints: LatentRoundTripCheckpoints,
    pub reconstruction: LatentReconstructionReport,
    pub predictors: Vec<LatentPredictorReport>,
    pub codebook: Option<CodebookUsageReport>,
    pub verdict: String,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentRoundTripCheckpoints {
    pub experience: PathBuf,
    pub future_evolved: PathBuf,
    pub future_trained: PathBuf,
    pub future_random: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentReconstructionReport {
    pub sample_count: usize,
    pub trained_decoder_loss_mean: f32,
    pub target_kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentPredictorReport {
    pub encoder: String,
    pub train_sample_count: usize,
    pub eval_sample_count: usize,
    pub latent_dim: usize,
    pub model_loss_mean: f32,
    pub stasis_loss_mean: f32,
    pub improvement_ratio: Option<f32>,
    pub predictive: bool,
}

pub trait BehaviorTrainer {
    type Input;
    type Output;
    type Target;

    fn behavior(&self) -> TrainableBehavior;

    fn train_step(&mut self, input: &Self::Input, target: &Self::Target) -> Result<TrainStats>;

    fn evaluate_sample(&self, input: &Self::Input, target: &Self::Target) -> Result<f32>;

    fn hardcoded_loss(&self, input: &Self::Input, target: &Self::Target) -> Result<Option<f32>>;

    fn save_checkpoint(&self, path: &Path) -> Result<()>;
}

pub async fn load_transitions(path: impl AsRef<Path>) -> Result<Vec<ExperienceTransition>> {
    let path = path.as_ref();
    if !path.exists() {
        bail!("missing ledger at {}", path.display());
    }
    if !path.is_dir() {
        bail!(
            "wrong ledger path shape: {} is not a directory",
            path.display()
        );
    }

    let transitions = JsonlLedger::new(path)
        .transitions()
        .await
        .with_context(|| {
            format!(
                "invalid JSONL while reading transitions below {}",
                path.display()
            )
        })?;
    if transitions.is_empty() {
        bail!("no transitions found below {}", path.display());
    }
    let without_z = transitions
        .iter()
        .filter(|transition| transition.before_z.z.is_empty() || transition.after_z.z.is_empty())
        .count();
    if without_z == transitions.len() {
        bail!(
            "transitions below {} do not contain usable before_z/after_z vectors",
            path.display()
        );
    }
    Ok(transitions)
}

pub fn split_transitions(
    mut transitions: Vec<ExperienceTransition>,
    validation_split: f32,
    seed: u64,
) -> (Vec<ExperienceTransition>, Vec<ExperienceTransition>) {
    let validation_split = validation_split.clamp(0.0, 0.9);
    let mut rng = StdRng::seed_from_u64(seed);
    transitions.shuffle(&mut rng);
    let eval_len = ((transitions.len() as f32) * validation_split).round() as usize;
    let eval_len = eval_len.min(transitions.len().saturating_sub(1));
    let eval = transitions.split_off(transitions.len().saturating_sub(eval_len));
    (transitions, eval)
}

pub async fn train_behavior(request: TrainBehaviorRequest) -> Result<TrainSummary> {
    let transitions = load_transitions(&request.ledger_path).await?;
    let transition_count = transitions.len();
    let (train, eval) = split_transitions(transitions, request.validation_split, request.seed);
    tokio::fs::create_dir_all(&request.checkpoint_path).await?;
    let metrics_path = request.checkpoint_path.join("metrics.jsonl");

    let mut writer = MetricWriter::open(&metrics_path).await?;
    let mut last_loss = None;
    let mut samples_seen = 0;
    let mut best_loss = None;
    let train_sample_count;

    match request.behavior {
        TrainableBehavior::Danger => {
            let samples = danger_samples(&train);
            train_sample_count = samples.len();
            let mut trainer = DangerNetTrainer::new(first_dim(&samples, |(_, _, input, _)| {
                input.flat_features().len()
            })?);
            for epoch in 0..request.epochs {
                for (sample_index, (t_ms, before, input, target)) in samples.iter().enumerate() {
                    if input.flat_features().len() != trainer.input_dim() {
                        continue;
                    }
                    let hardcoded_loss = Some(mse(
                        &HardcodedDangerPredictor
                            .predict_from_now(before, input)
                            .risks(),
                        &target.risks(),
                    ));
                    let model_loss = trainer.evaluate_sample(input, target)?;
                    let stats = trainer.train_step(input, target)?;
                    last_loss = Some(stats.loss);
                    samples_seen = stats.samples_seen;
                    best_loss = trainer.best_loss();
                    writer
                        .write(train_metric(
                            *t_ms,
                            TrainableBehavior::Danger,
                            epoch,
                            sample_index,
                            stats.loss,
                            hardcoded_loss,
                            model_loss,
                        ))
                        .await?;
                }
            }
            trainer.save_checkpoint(&request.checkpoint_path)?;
        }
        TrainableBehavior::Charge => {
            let samples = charge_samples(&train);
            train_sample_count = samples.len();
            let mut trainer = ChargeNetTrainer::new(first_dim(&samples, |(_, _, input, _)| {
                input.flat_features().len()
            })?);
            for epoch in 0..request.epochs {
                for (sample_index, (t_ms, before, input, target)) in samples.iter().enumerate() {
                    if input.flat_features().len() != trainer.input_dim() {
                        continue;
                    }
                    let hardcoded_loss = Some(mse(
                        &HardcodedChargePredictor
                            .predict_from_now(before, input)
                            .values(),
                        &target.values(),
                    ));
                    let model_loss = trainer.evaluate_sample(input, target)?;
                    let stats = trainer.train_step(input, target)?;
                    last_loss = Some(stats.loss);
                    samples_seen = stats.samples_seen;
                    best_loss = trainer.best_loss();
                    writer
                        .write(train_metric(
                            *t_ms,
                            TrainableBehavior::Charge,
                            epoch,
                            sample_index,
                            stats.loss,
                            hardcoded_loss,
                            model_loss,
                        ))
                        .await?;
                }
            }
            trainer.save_checkpoint(&request.checkpoint_path)?;
        }
        TrainableBehavior::ActionValue => {
            let samples = action_value_samples(&train);
            train_sample_count = samples.len();
            let mut trainer =
                ActionValueNetTrainer::new(first_dim(&samples, |(_, _, input, _)| {
                    input.flat_features().len()
                })?);
            for epoch in 0..request.epochs {
                for (sample_index, (t_ms, before, input, target)) in samples.iter().enumerate() {
                    if input.flat_features().len() != trainer.input_dim() {
                        continue;
                    }
                    let hardcoded = HardcodedActionValuePredictor.predict_from_now(before, input);
                    let hardcoded_loss = Some((hardcoded.value - target.value).powi(2));
                    let model_loss = trainer.evaluate_sample(input, target)?;
                    let stats = trainer.train_step(input, target)?;
                    last_loss = Some(stats.loss);
                    samples_seen = stats.samples_seen;
                    best_loss = trainer.best_loss();
                    writer
                        .write(train_metric(
                            *t_ms,
                            TrainableBehavior::ActionValue,
                            epoch,
                            sample_index,
                            stats.loss,
                            hardcoded_loss,
                            model_loss,
                        ))
                        .await?;
                }
            }
            trainer.save_checkpoint(&request.checkpoint_path)?;
        }
        TrainableBehavior::Future => {
            let samples = future_samples(&train);
            train_sample_count = samples.len();
            let input_dim = first_dim(&samples, |(_, input, _)| input.flat_features().len())?;
            let latent_dim = first_dim(&samples, |(_, _, target)| target.len())?;
            let mut trainer = FutureNetTrainer::new(input_dim, latent_dim);
            let mut stasis = StasisFuturePredictor;
            for epoch in 0..request.epochs {
                for (sample_index, (t_ms, input, target)) in samples.iter().enumerate() {
                    if input.flat_features().len() != trainer.input_dim()
                        || target.len() != trainer.latent_dim()
                    {
                        continue;
                    }
                    let hardcoded =
                        stasis.predict(&input.latent, &input.action, input.offset_ms)?;
                    let hardcoded_loss = Some(mse_vec(&hardcoded.predicted_z, target));
                    let model_loss = trainer.evaluate_sample(input, target)?;
                    let stats = trainer.train_step(input, target)?;
                    last_loss = Some(stats.loss);
                    samples_seen = stats.samples_seen;
                    best_loss = trainer.best_loss();
                    writer
                        .write(train_metric(
                            *t_ms,
                            TrainableBehavior::Future,
                            epoch,
                            sample_index,
                            stats.loss,
                            hardcoded_loss,
                            model_loss,
                        ))
                        .await?;
                }
            }
            trainer.save_checkpoint(&request.checkpoint_path)?;
        }
        TrainableBehavior::EyeNext => {
            let samples = eye_next_samples(&train);
            train_sample_count = samples.len();
            let (input_dim, width, height) = samples
                .first()
                .map(|(_, _, input, target)| {
                    (input.flat_features().len(), target.width, target.height)
                })
                .ok_or_else(|| anyhow!("no usable eye-next samples"))?;
            let mut trainer = EyeNextNetTrainer::new(input_dim, width, height);
            for epoch in 0..request.epochs {
                for (sample_index, (t_ms, before, input, target)) in samples.iter().enumerate() {
                    if input.flat_features().len() != trainer.input_dim() {
                        continue;
                    }
                    let hardcoded_loss = eye_current_loss(before, target);
                    let model_loss = trainer.evaluate_sample(input, target)?;
                    let stats = trainer.train_step(input, target)?;
                    last_loss = Some(stats.loss);
                    samples_seen = stats.samples_seen;
                    best_loss = trainer.best_loss();
                    writer
                        .write(train_metric(
                            *t_ms,
                            TrainableBehavior::EyeNext,
                            epoch,
                            sample_index,
                            stats.loss,
                            hardcoded_loss,
                            model_loss,
                        ))
                        .await?;
                }
            }
            trainer.save_checkpoint(&request.checkpoint_path)?;
        }
        TrainableBehavior::EarNext => {
            let samples = ear_next_samples(&train);
            train_sample_count = samples.len();
            let (input_dim, output_dim, sample_rate_hz, channels) = samples
                .first()
                .map(|(_, _, input, target)| {
                    (
                        input.flat_features().len(),
                        target.features.len(),
                        target.sample_rate_hz,
                        target.channels,
                    )
                })
                .ok_or_else(|| anyhow!("no usable ear-next samples"))?;
            let mut trainer = EarNextNetTrainer::with_audio_shape(
                input_dim,
                output_dim,
                sample_rate_hz,
                channels,
            );
            for epoch in 0..request.epochs {
                for (sample_index, (t_ms, before, input, target)) in samples.iter().enumerate() {
                    if input.flat_features().len() != trainer.input_dim()
                        || target.features.len() != trainer.output_dim()
                    {
                        continue;
                    }
                    let hardcoded_loss = ear_current_loss(before, target);
                    let model_loss = trainer.evaluate_sample(input, target)?;
                    let stats = trainer.train_step(input, target)?;
                    last_loss = Some(stats.loss);
                    samples_seen = stats.samples_seen;
                    best_loss = trainer.best_loss();
                    writer
                        .write(train_metric(
                            *t_ms,
                            TrainableBehavior::EarNext,
                            epoch,
                            sample_index,
                            stats.loss,
                            hardcoded_loss,
                            model_loss,
                        ))
                        .await?;
                }
            }
            trainer.save_checkpoint(&request.checkpoint_path)?;
        }
        TrainableBehavior::Experience => {
            let samples = experience_samples(&train);
            train_sample_count = samples.len();
            let (input_dim, decode_lengths) = samples
                .first()
                .map(|(_, input, target, _)| {
                    (input.flat_features().len(), target.feature_lengths())
                })
                .ok_or_else(|| anyhow!("no usable experience samples"))?;
            let z_dim = input_dim.clamp(8, 32);
            let mut trainer = ExperienceAutoencoderTrainer::new(input_dim, z_dim, decode_lengths);
            for epoch in 0..request.epochs {
                for (sample_index, (t_ms, input, target, _baseline_z)) in samples.iter().enumerate()
                {
                    if input.flat_features().len() != trainer.input_dim()
                        || target.feature_lengths() != trainer.decode_lengths()
                    {
                        continue;
                    }
                    let model_loss = trainer.evaluate_sample(input, target)?;
                    let stats = trainer.train_step(input, target)?;
                    last_loss = Some(stats.loss);
                    samples_seen = stats.samples_seen;
                    best_loss = trainer.best_loss();
                    writer
                        .write(train_metric(
                            *t_ms,
                            TrainableBehavior::Experience,
                            epoch,
                            sample_index,
                            stats.loss,
                            None,
                            model_loss,
                        ))
                        .await?;
                }
            }
            trainer.save_checkpoint(&request.checkpoint_path)?;
        }
    }

    let eval_transitions = if eval.is_empty() { &train } else { &eval };
    let eval_request = EvaluateBehaviorRequest {
        behavior: request.behavior.clone(),
        ledger_path: request.ledger_path,
        checkpoint_path: request.checkpoint_path.clone(),
        max_samples: None,
    };
    let mut evaluation = evaluate_behavior_on_transitions(eval_request, eval_transitions)?;
    evaluation.checkpoint_path = request.checkpoint_path.clone();

    let evaluation_path = request.checkpoint_path.join("evaluation.json");
    if let Ok(json) = serde_json::to_string_pretty(&evaluation) {
        let _ = std::fs::write(&evaluation_path, json);
    }

    Ok(TrainSummary {
        behavior: request.behavior,
        transition_count,
        train_sample_count,
        eval_sample_count: evaluation.sample_count,
        epochs: request.epochs,
        samples_seen,
        last_loss,
        best_loss,
        metrics_path,
        checkpoint_path: request.checkpoint_path,
        evaluation,
    })
}

pub async fn train_latent_round_trip(
    request: TrainLatentRoundTripRequest,
) -> Result<TrainLatentRoundTripReport> {
    let transitions = load_transitions(&request.ledger_path).await?;
    let transition_count = transitions.len();
    let (train, eval) = split_transitions(transitions, request.validation_split, request.seed);
    let eval_transitions = if eval.is_empty() { &train } else { &eval };
    let checkpoints = LatentRoundTripCheckpoints {
        experience: request.checkpoint_path.join("experience"),
        future_evolved: request.checkpoint_path.join("future-evolved"),
        future_trained: request.checkpoint_path.join("future-trained"),
        future_random: request.checkpoint_path.join("future-random"),
    };

    let experience_train = experience_samples(&train);
    let (input_dim, decode_lengths) = experience_train
        .first()
        .map(|(_, input, target, _)| (input.flat_features().len(), target.feature_lengths()))
        .ok_or_else(|| anyhow!("no usable experience samples for latent round-trip training"))?;
    let z_dim = request.z_dim.clamp(2, input_dim.max(2));
    let mut autoencoder = ExperienceAutoencoderTrainer::new(input_dim, z_dim, decode_lengths);
    for _epoch in 0..request.epochs {
        for (_, input, target, _) in &experience_train {
            if input.flat_features().len() == autoencoder.input_dim()
                && target.feature_lengths() == autoencoder.decode_lengths()
            {
                autoencoder.train_step(input, target)?;
            }
        }
    }
    autoencoder.save_checkpoint(&checkpoints.experience)?;

    let reconstruction = evaluate_trained_reconstruction(&autoencoder, eval_transitions)?;
    let evolved_report = train_and_evaluate_future_latents(
        "online-evolved-filters",
        replay_latent_future_samples(&train),
        replay_latent_future_samples(eval_transitions),
        request.epochs,
        &checkpoints.future_evolved,
    )?;

    let trained_train = trained_latent_future_samples(&autoencoder, &train)?;
    let trained_eval = trained_latent_future_samples(&autoencoder, eval_transitions)?;
    let trained_report = train_and_evaluate_future_latents(
        "trainable-autoencoder",
        trained_train.clone(),
        trained_eval,
        request.epochs,
        &checkpoints.future_trained,
    )?;

    let mut random_train_encoder = RandomProjectionExperienceEncoder::new(z_dim, request.seed);
    let mut random_eval_encoder = RandomProjectionExperienceEncoder::new(z_dim, request.seed);
    let random_train = encoded_future_samples(&mut random_train_encoder, &train)?;
    let random_eval = encoded_future_samples(&mut random_eval_encoder, eval_transitions)?;
    let random_report = train_and_evaluate_future_latents(
        "random-projection",
        random_train,
        random_eval,
        request.epochs,
        &checkpoints.future_random,
    )?;

    let codebook = if let Some(codebook_size) = request.codebook_size {
        let mut quantizer = CodebookQuantizer::from_latents(
            &trained_train
                .iter()
                .map(|(_, input, _)| input.latent.z.clone())
                .collect::<Vec<_>>(),
            codebook_size,
        );
        for (_, input, target) in &trained_train {
            let code_id = quantizer.encode(&input.latent.z);
            let decoded = quantizer.decode(code_id);
            let _ = mse_vec(&decoded, target);
        }
        Some(quantizer.report())
    } else {
        None
    };

    let predictors = vec![evolved_report, trained_report, random_report];
    let mut warnings = Vec::new();
    if transition_count < 50 {
        warnings.push(format!(
            "insufficient data: {transition_count} transitions is below the conservative 50-transition floor"
        ));
    }
    if predictors.iter().all(|report| !report.predictive) {
        warnings.push("no encoder beat the stasis baseline on held-out prediction".to_string());
    }
    let verdict = if predictors
        .iter()
        .any(|report| report.encoder == "trainable-autoencoder" && report.predictive)
    {
        "trained latent is predictive on held-out replay".to_string()
    } else if predictors.iter().any(|report| report.predictive) {
        "a latent is predictive, but the trained encoder is not yet strongest".to_string()
    } else {
        "latent remains compact but not proven predictive".to_string()
    };

    let report = TrainLatentRoundTripReport {
        schema_version: 1,
        transition_count,
        train_transition_count: train.len(),
        eval_transition_count: eval_transitions.len(),
        epochs: request.epochs,
        z_dim,
        checkpoints,
        reconstruction,
        predictors,
        codebook,
        verdict,
        warnings,
    };
    if let Some(parent) = request.report_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&request.report_path, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub async fn evaluate_behavior(
    request: EvaluateBehaviorRequest,
) -> Result<BehaviorEvaluationReport> {
    let transitions = load_transitions(&request.ledger_path).await?;
    evaluate_behavior_on_transitions(request, &transitions)
}

pub fn load_models_config(path: &Path) -> Result<BehaviorRegistryConfig> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

pub fn set_behavior_checkpoint(
    config: &mut BehaviorRegistryConfig,
    behavior: TrainableBehavior,
    checkpoint: PathBuf,
) -> Result<()> {
    let entry = behavior_config_entry(config, &behavior);
    entry.checkpoint = Some(checkpoint.to_string_lossy().to_string());
    Ok(())
}

pub fn set_behavior_regime(
    config: &mut BehaviorRegistryConfig,
    behavior: TrainableBehavior,
    regime: BehaviorRegime,
) -> Result<()> {
    let entry = behavior_config_entry(config, &behavior);
    entry.regime = regime;
    entry.fallback = FallbackPolicy::UseHardcoded;
    Ok(())
}

pub fn write_models_config(path: &Path, config: &BehaviorRegistryConfig) -> Result<()> {
    let text = toml::to_string_pretty(config)?;
    std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

pub fn promote_behavior_config(
    behavior: TrainableBehavior,
    checkpoint: PathBuf,
    config_path: &Path,
    regime: BehaviorRegime,
) -> Result<()> {
    if regime == BehaviorRegime::ModelInfer && behavior.is_safety_critical() {
        eprintln!(
            "warning: {} is safety-critical; model-infer was explicitly requested",
            behavior
        );
    }
    let mut config = load_models_config(config_path)?;
    set_behavior_checkpoint(&mut config, behavior.clone(), checkpoint)?;
    set_behavior_regime(&mut config, behavior, regime)?;
    write_models_config(config_path, &config)
}

fn behavior_config_entry<'a>(
    config: &'a mut BehaviorRegistryConfig,
    behavior: &TrainableBehavior,
) -> &'a mut BehaviorConfig {
    config
        .behavior
        .entry(behavior.config_key().to_string())
        .or_insert_with(|| BehaviorConfig {
            regime: BehaviorRegime::Hardcoded,
            hardcoded: behavior.default_hardcoded_id().to_string(),
            model: Some(behavior.default_model_id().to_string()),
            checkpoint: None,
            fallback: FallbackPolicy::UseHardcoded,
        })
}

fn evaluate_behavior_on_transitions(
    request: EvaluateBehaviorRequest,
    transitions: &[ExperienceTransition],
) -> Result<BehaviorEvaluationReport> {
    let max_samples = request.max_samples.unwrap_or(usize::MAX);
    let mut warnings = Vec::new();
    let (model_losses, hardcoded_losses): (Vec<f32>, Vec<Option<f32>>) = match request.behavior {
        TrainableBehavior::Danger => {
            let samples = danger_samples(transitions);
            let input_dim = first_dim(&samples, |(_, _, input, _)| input.flat_features().len())?;
            let trainer = DangerNetTrainer::load_checkpoint(&request.checkpoint_path, input_dim)?;
            samples
                .into_iter()
                .take(max_samples)
                .filter(|(_, _, input, _)| input.flat_features().len() == trainer.input_dim())
                .map(|(_, before, input, target)| {
                    let model = trainer.predict(&input)?;
                    let hard = HardcodedDangerPredictor.predict_from_now(&before, &input);
                    Ok((
                        mse(&model.risks(), &target.risks()),
                        Some(mse(&hard.risks(), &target.risks())),
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip()
        }
        TrainableBehavior::Charge => {
            let samples = charge_samples(transitions);
            let input_dim = first_dim(&samples, |(_, _, input, _)| input.flat_features().len())?;
            let trainer = ChargeNetTrainer::load_checkpoint(&request.checkpoint_path, input_dim)?;
            samples
                .into_iter()
                .take(max_samples)
                .filter(|(_, _, input, _)| input.flat_features().len() == trainer.input_dim())
                .map(|(_, before, input, target)| {
                    let model = trainer.predict(&input)?;
                    let hard = HardcodedChargePredictor.predict_from_now(&before, &input);
                    Ok((
                        mse(&model.values(), &target.values()),
                        Some(mse(&hard.values(), &target.values())),
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip()
        }
        TrainableBehavior::ActionValue => {
            let samples = action_value_samples(transitions);
            let input_dim = first_dim(&samples, |(_, _, input, _)| input.flat_features().len())?;
            let trainer =
                ActionValueNetTrainer::load_checkpoint(&request.checkpoint_path, input_dim)?;
            samples
                .into_iter()
                .take(max_samples)
                .filter(|(_, _, input, _)| input.flat_features().len() == trainer.input_dim())
                .map(|(_, before, input, target)| {
                    let model = trainer.predict(&input)?;
                    let hard = HardcodedActionValuePredictor.predict_from_now(&before, &input);
                    Ok((
                        (model.value - target.value).powi(2),
                        Some((hard.value - target.value).powi(2)),
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip()
        }
        TrainableBehavior::Future => {
            let samples = future_samples(transitions);
            let input_dim = first_dim(&samples, |(_, input, _)| input.flat_features().len())?;
            let latent_dim = first_dim(&samples, |(_, _, target)| target.len())?;
            let trainer =
                FutureNetTrainer::load_checkpoint(&request.checkpoint_path, input_dim, latent_dim)?;
            let mut stasis = StasisFuturePredictor;
            samples
                .into_iter()
                .take(max_samples)
                .filter(|(_, input, target)| {
                    input.flat_features().len() == trainer.input_dim()
                        && target.len() == trainer.latent_dim()
                })
                .map(|(_, input, target)| {
                    let model = trainer.predict(&input)?;
                    let hard = stasis.predict(&input.latent, &input.action, input.offset_ms)?;
                    Ok((
                        mse_vec(&model.predicted_z, &target),
                        Some(mse_vec(&hard.predicted_z, &target)),
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip()
        }
        TrainableBehavior::EyeNext => {
            let samples = eye_next_samples(transitions);
            let input_dim = first_dim(&samples, |(_, _, input, _)| input.flat_features().len())?;
            let trainer = EyeNextNetTrainer::load_checkpoint(&request.checkpoint_path, input_dim)?;
            samples
                .into_iter()
                .take(max_samples)
                .filter(|(_, _, input, _)| input.flat_features().len() == trainer.input_dim())
                .map(|(_, before, input, target)| {
                    let model = trainer.predict(&input)?;
                    Ok((
                        mse_bytes(&model.rgb, &target.rgb),
                        eye_current_loss(&before, &target),
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip()
        }
        TrainableBehavior::EarNext => {
            let samples = ear_next_samples(transitions);
            let input_dim = first_dim(&samples, |(_, _, input, _)| input.flat_features().len())?;
            let trainer = EarNextNetTrainer::load_checkpoint(&request.checkpoint_path, input_dim)?;
            samples
                .into_iter()
                .take(max_samples)
                .filter(|(_, _, input, target)| {
                    input.flat_features().len() == trainer.input_dim()
                        && target.features.len() == trainer.output_dim()
                })
                .map(|(_, before, input, target)| {
                    let model = trainer.predict(&input)?;
                    Ok((
                        mse_vec(&model.features, &target.features),
                        ear_current_loss(&before, &target),
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip()
        }
        TrainableBehavior::Experience => {
            let samples = experience_samples(transitions);
            let input_dim = first_dim(&samples, |(_, input, _, _)| input.flat_features().len())?;
            let trainer =
                ExperienceAutoencoderTrainer::load_checkpoint(&request.checkpoint_path, input_dim)?;
            samples
                .into_iter()
                .take(max_samples)
                .filter(|(_, input, target, _)| {
                    input.flat_features().len() == trainer.input_dim()
                        && target.feature_lengths() == trainer.decode_lengths()
                })
                .map(|(_, input, target, _)| {
                    let prediction = trainer.predict(&input)?;
                    Ok((
                        mse_vec(&prediction.decoded.flat_features(), &target.flat_features()),
                        None,
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip()
        }
    };

    let sample_count = model_losses.len();
    if sample_count == 0 {
        bail!("no usable {} samples for evaluation", request.behavior);
    }
    if sample_count < 50 {
        warnings.push(format!(
            "insufficient data: {sample_count} samples is below the conservative 50-sample floor"
        ));
    }
    let model_loss_mean = mean(&model_losses);
    let hardcoded_values: Vec<f32> = hardcoded_losses.into_iter().flatten().collect();
    let hardcoded_loss_mean = (!hardcoded_values.is_empty()).then(|| mean(&hardcoded_values));
    let selected_loss_mean = hardcoded_loss_mean.or(Some(model_loss_mean));
    let model_better_than_hardcoded = hardcoded_loss_mean.map(|hard| model_loss_mean < hard);
    let improvement_ratio =
        hardcoded_loss_mean.and_then(|hard| (hard > 0.0).then(|| (hard - model_loss_mean) / hard));
    let recommendation = recommend(
        &request.behavior,
        sample_count,
        model_loss_mean,
        hardcoded_loss_mean,
        improvement_ratio,
    );

    Ok(BehaviorEvaluationReport {
        behavior: request.behavior,
        checkpoint_path: request.checkpoint_path,
        sample_count,
        model_loss_mean,
        hardcoded_loss_mean,
        selected_loss_mean,
        model_better_than_hardcoded,
        improvement_ratio,
        warnings,
        recommendation,
    })
}

fn recommend(
    behavior: &TrainableBehavior,
    sample_count: usize,
    model_loss_mean: f32,
    hardcoded_loss_mean: Option<f32>,
    improvement_ratio: Option<f32>,
) -> PromotionRecommendation {
    if sample_count < 50 {
        return PromotionRecommendation::KeepHardcoded;
    }
    if !model_loss_mean.is_finite() || model_loss_mean > 1.0e6 {
        return PromotionRecommendation::RejectCheckpoint;
    }
    match (hardcoded_loss_mean, improvement_ratio) {
        (Some(_), Some(ratio))
            if ratio > 0.25 && sample_count > 500 && !behavior.is_safety_critical() =>
        {
            PromotionRecommendation::PromoteToModelInfer
        }
        (Some(_), Some(ratio)) if ratio > 0.10 => PromotionRecommendation::ShadowInfer,
        (Some(_), _) => PromotionRecommendation::KeepHardcoded,
        (None, _) if sample_count >= 50 => PromotionRecommendation::ShadowInfer,
        _ => PromotionRecommendation::KeepHardcoded,
    }
}

impl BehaviorTrainer for DangerNetTrainer {
    type Input = DangerInput;
    type Output = ();
    type Target = DangerTarget;

    fn behavior(&self) -> TrainableBehavior {
        TrainableBehavior::Danger
    }

    fn train_step(&mut self, input: &Self::Input, target: &Self::Target) -> Result<TrainStats> {
        let stats = DangerNetTrainer::train_step(self, input, target)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }

    fn evaluate_sample(&self, input: &Self::Input, target: &Self::Target) -> Result<f32> {
        Ok(mse(&self.predict(input)?.risks(), &target.risks()))
    }

    fn hardcoded_loss(&self, _input: &Self::Input, _target: &Self::Target) -> Result<Option<f32>> {
        Ok(None)
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        DangerNetTrainer::save_checkpoint(self, path)
    }
}

impl BehaviorTrainer for ChargeNetTrainer {
    type Input = ChargeInput;
    type Output = ();
    type Target = ChargeTarget;

    fn behavior(&self) -> TrainableBehavior {
        TrainableBehavior::Charge
    }

    fn train_step(&mut self, input: &Self::Input, target: &Self::Target) -> Result<TrainStats> {
        let stats = ChargeNetTrainer::train_step(self, input, target)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }

    fn evaluate_sample(&self, input: &Self::Input, target: &Self::Target) -> Result<f32> {
        Ok(mse(&self.predict(input)?.values(), &target.values()))
    }

    fn hardcoded_loss(&self, _input: &Self::Input, _target: &Self::Target) -> Result<Option<f32>> {
        Ok(None)
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        ChargeNetTrainer::save_checkpoint(self, path)
    }
}

impl BehaviorTrainer for ActionValueNetTrainer {
    type Input = ActionValueInput;
    type Output = ();
    type Target = ActionValueTarget;

    fn behavior(&self) -> TrainableBehavior {
        TrainableBehavior::ActionValue
    }

    fn train_step(&mut self, input: &Self::Input, target: &Self::Target) -> Result<TrainStats> {
        let stats = ActionValueNetTrainer::train_step(self, input, target)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }

    fn evaluate_sample(&self, input: &Self::Input, target: &Self::Target) -> Result<f32> {
        Ok((self.predict(input)?.value - target.value).powi(2))
    }

    fn hardcoded_loss(&self, _input: &Self::Input, _target: &Self::Target) -> Result<Option<f32>> {
        Ok(None)
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        ActionValueNetTrainer::save_checkpoint(self, path)
    }
}

impl BehaviorTrainer for FutureNetTrainer {
    type Input = FutureInput;
    type Output = ();
    type Target = Vec<f32>;

    fn behavior(&self) -> TrainableBehavior {
        TrainableBehavior::Future
    }

    fn train_step(&mut self, input: &Self::Input, target: &Self::Target) -> Result<TrainStats> {
        FutureNetTrainer::train_step(self, input, target)
    }

    fn evaluate_sample(&self, input: &Self::Input, target: &Self::Target) -> Result<f32> {
        Ok(mse_vec(&self.predict(input)?.predicted_z, target))
    }

    fn hardcoded_loss(&self, _input: &Self::Input, _target: &Self::Target) -> Result<Option<f32>> {
        Ok(None)
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        FutureNetTrainer::save_checkpoint(self, path)
    }
}

impl BehaviorTrainer for EyeNextNetTrainer {
    type Input = EyeNextInput;
    type Output = ();
    type Target = EyeNextTarget;

    fn behavior(&self) -> TrainableBehavior {
        TrainableBehavior::EyeNext
    }

    fn train_step(&mut self, input: &Self::Input, target: &Self::Target) -> Result<TrainStats> {
        let stats = EyeNextNetTrainer::train_step(self, input, target)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }

    fn evaluate_sample(&self, input: &Self::Input, target: &Self::Target) -> Result<f32> {
        Ok(mse_bytes(&self.predict(input)?.rgb, &target.rgb))
    }

    fn hardcoded_loss(&self, _input: &Self::Input, _target: &Self::Target) -> Result<Option<f32>> {
        Ok(None)
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        EyeNextNetTrainer::save_checkpoint(self, path)
    }
}

impl BehaviorTrainer for EarNextNetTrainer {
    type Input = EarNextInput;
    type Output = ();
    type Target = EarNextTarget;

    fn behavior(&self) -> TrainableBehavior {
        TrainableBehavior::EarNext
    }

    fn train_step(&mut self, input: &Self::Input, target: &Self::Target) -> Result<TrainStats> {
        let stats = EarNextNetTrainer::train_step(self, input, target)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }

    fn evaluate_sample(&self, input: &Self::Input, target: &Self::Target) -> Result<f32> {
        Ok(mse_vec(&self.predict(input)?.features, &target.features))
    }

    fn hardcoded_loss(&self, _input: &Self::Input, _target: &Self::Target) -> Result<Option<f32>> {
        Ok(None)
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        EarNextNetTrainer::save_checkpoint(self, path)
    }
}

impl BehaviorTrainer for ExperienceAutoencoderTrainer {
    type Input = ExperienceEncodeInput;
    type Output = ();
    type Target = ExperienceDecodeOutput;

    fn behavior(&self) -> TrainableBehavior {
        TrainableBehavior::Experience
    }

    fn train_step(&mut self, input: &Self::Input, target: &Self::Target) -> Result<TrainStats> {
        let stats = ExperienceAutoencoderTrainer::train_step(self, input, target)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }

    fn evaluate_sample(&self, input: &Self::Input, target: &Self::Target) -> Result<f32> {
        Ok(mse_vec(
            &self.predict(input)?.decoded.flat_features(),
            &target.flat_features(),
        ))
    }

    fn hardcoded_loss(&self, _input: &Self::Input, _target: &Self::Target) -> Result<Option<f32>> {
        Ok(None)
    }

    fn save_checkpoint(&self, path: &Path) -> Result<()> {
        ExperienceAutoencoderTrainer::save_checkpoint(self, path)
    }
}

struct MetricWriter {
    file: tokio::fs::File,
}

impl MetricWriter {
    async fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
            .with_context(|| format!("open metrics {}", path.display()))?;
        Ok(Self { file })
    }

    async fn write(&mut self, record: BehaviorMetricRecord) -> Result<()> {
        let line = serde_json::to_string(&record)?;
        self.file.write_all(line.as_bytes()).await?;
        self.file.write_all(b"\n").await?;
        Ok(())
    }
}

fn train_metric(
    t_ms: TimeMs,
    behavior: TrainableBehavior,
    epoch: usize,
    sample_index: usize,
    train_loss: f32,
    hardcoded_loss: Option<f32>,
    model_loss: f32,
) -> BehaviorMetricRecord {
    BehaviorMetricRecord {
        t_ms: if t_ms == 0 { now_ms() } else { t_ms },
        behavior,
        epoch,
        sample_index,
        train_loss: Some(train_loss),
        eval_loss: None,
        hardcoded_loss,
        model_loss: Some(model_loss),
        selected_loss: hardcoded_loss,
        notes: Vec::new(),
    }
}

fn replay_latent_future_samples(
    transitions: &[ExperienceTransition],
) -> Vec<(TimeMs, FutureInput, Vec<f32>)> {
    future_samples(transitions)
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
    for (_, input, target, _) in experience_samples(transitions) {
        if input.flat_features().len() != autoencoder.input_dim()
            || target.feature_lengths() != autoencoder.decode_lengths()
        {
            continue;
        }
        let prediction = autoencoder.predict(&input)?;
        losses.push(mse_vec(
            &prediction.decoded.flat_features(),
            &target.flat_features(),
        ));
    }
    if losses.is_empty() {
        bail!("no usable reconstruction samples for latent round-trip evaluation");
    }
    Ok(LatentReconstructionReport {
        sample_count: losses.len(),
        trained_decoder_loss_mean: mean(&losses),
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
) -> Result<LatentPredictorReport> {
    let input_dim = first_dim(&train_samples, |(_, input, _)| input.flat_features().len())?;
    let latent_dim = first_dim(&train_samples, |(_, _, target)| target.len())?;
    let mut trainer = FutureNetTrainer::new(input_dim, latent_dim);
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
        train_sample_count: train_samples.len(),
        eval_sample_count: model_losses.len(),
        latent_dim,
        model_loss_mean,
        stasis_loss_mean,
        improvement_ratio,
        predictive: model_loss_mean < stasis_loss_mean,
    })
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use netherwick_actions::ActionPrimitive;
    use netherwick_body::BodySense;
    use netherwick_core::Reward;
    use netherwick_experience::ExperienceLatent;
    use netherwick_now::SurpriseSense;
    use std::fs;

    #[tokio::test]
    async fn test_train_behavior_writes_evaluation_json() {
        let temp_dir = std::env::temp_dir().join(format!("netherwick_train_test_{}", now_ms()));
        let ledger_dir = temp_dir.join("ledger");
        let session_dir = ledger_dir.join("2026-06-24");
        fs::create_dir_all(&session_dir).unwrap();

        let checkpoint_dir = temp_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).unwrap();

        // Construct 5 mock transitions to have enough data for training and validation splits
        let mut transitions = Vec::new();
        for i in 0..5 {
            let transition = ExperienceTransition {
                id: uuid::Uuid::new_v4(),
                before_frame_id: uuid::Uuid::new_v4(),
                before: Now::blank(100 + i * 100, BodySense::default()),
                before_z: ExperienceLatent {
                    t_ms: 100 + i * 100,
                    z: vec![0.1; 4],
                    ..ExperienceLatent::default()
                },
                action: Some(ActionPrimitive::Stop),
                predicted_futures: Vec::new(),
                after: Now::blank(200 + i * 100, BodySense::default()),
                after_z: ExperienceLatent {
                    t_ms: 200 + i * 100,
                    z: vec![0.2; 4],
                    ..ExperienceLatent::default()
                },
                reward: Reward { value: 0.0 },
                surprise: SurpriseSense::default(),
                created_at_ms: 200 + i * 100,
            };
            transitions.push(transition);
        }

        let transitions_file = session_dir.join("transitions.jsonl");
        let mut content = String::new();
        for t in &transitions {
            content.push_str(&serde_json::to_string(t).unwrap());
            content.push('\n');
        }
        fs::write(&transitions_file, content).unwrap();

        let request = TrainBehaviorRequest {
            behavior: TrainableBehavior::Danger,
            ledger_path: ledger_dir,
            checkpoint_path: checkpoint_dir.clone(),
            epochs: 1,
            validation_split: 0.2,
            seed: 42,
        };

        let summary = train_behavior(request).await.unwrap();
        assert_eq!(summary.behavior, TrainableBehavior::Danger);

        // Verify that evaluation.json was created
        let eval_json_path = checkpoint_dir.join("evaluation.json");
        assert!(eval_json_path.exists());

        let eval_content = fs::read_to_string(&eval_json_path).unwrap();
        let report: BehaviorEvaluationReport = serde_json::from_str(&eval_content).unwrap();
        assert_eq!(report.behavior, TrainableBehavior::Danger);

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_train_latent_round_trip_writes_predictive_report() {
        let temp_dir =
            std::env::temp_dir().join(format!("netherwick_latent_round_trip_test_{}", now_ms()));
        let ledger_dir = temp_dir.join("ledger");
        let session_dir = ledger_dir.join("2026-06-27");
        fs::create_dir_all(&session_dir).unwrap();

        let mut transitions = Vec::new();
        for i in 0..6 {
            let mut before_body = BodySense::default();
            before_body.battery_level = 0.8;
            before_body.odometry.x_m = i as f32 * 0.01;
            let mut after_body = before_body.clone();
            after_body.odometry.x_m += 0.01;
            let before = Now::blank(100 + i * 100, before_body);
            let after = Now::blank(200 + i * 100, after_body);
            transitions.push(ExperienceTransition {
                id: uuid::Uuid::new_v4(),
                before_frame_id: uuid::Uuid::new_v4(),
                before,
                before_z: ExperienceLatent {
                    t_ms: 100 + i * 100,
                    z: vec![0.1 + i as f32 * 0.01, 0.2],
                    ..ExperienceLatent::default()
                },
                action: Some(ActionPrimitive::Stop),
                predicted_futures: Vec::new(),
                after,
                after_z: ExperienceLatent {
                    t_ms: 200 + i * 100,
                    z: vec![0.11 + i as f32 * 0.01, 0.2],
                    ..ExperienceLatent::default()
                },
                reward: Reward { value: 0.0 },
                surprise: SurpriseSense::default(),
                created_at_ms: 200 + i * 100,
            });
        }

        let transitions_file = session_dir.join("transitions.jsonl");
        let content = transitions
            .iter()
            .map(|transition| serde_json::to_string(transition).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&transitions_file, format!("{content}\n")).unwrap();

        let report_path = temp_dir.join("latent-report.json");
        let report = train_latent_round_trip(TrainLatentRoundTripRequest {
            ledger_path: ledger_dir,
            checkpoint_path: temp_dir.join("checkpoint"),
            report_path: report_path.clone(),
            epochs: 0,
            validation_split: 0.34,
            seed: 7,
            z_dim: 2,
            codebook_size: Some(2),
        })
        .await
        .unwrap();

        assert!(report_path.exists());
        assert_eq!(report.predictors.len(), 3);
        assert!(report
            .predictors
            .iter()
            .any(|predictor| predictor.encoder == "trainable-autoencoder"));
        assert!(report.reconstruction.sample_count > 0);
        assert_eq!(report.codebook.unwrap().code_count, 2);

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
