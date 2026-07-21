use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use pete_actions::ActionPrimitive;
use pete_behaviors::{BehaviorConfig, BehaviorRegime, BehaviorRegistryConfig, FallbackPolicy};
use pete_core::TimeMs;
use pete_experience::{
    action_features, action_value_input_from_transition_like,
    action_value_target_from_reward_surprise, charge_input_from_transition_like,
    charge_target_from_transition_like, danger_input_from_transition_like,
    danger_target_from_transition_like, ear_next_input_from_transition_like,
    ear_next_target_from_now, experience_decode_target_from_now, experience_encode_input_from_now,
    eye_next_input_from_transition_like, eye_next_target_from_now, ActionValueInput,
    ActionValueTarget, ChargeInput, ChargeTarget, CodebookQuantizer, CodebookUsageReport,
    DangerInput, DangerTarget, EarNextInput, EarNextTarget, ExperienceDecodeOutput,
    ExperienceEncodeInput, ExperienceLatent, ExperienceSurprise, EyeNextInput, EyeNextTarget,
    FutureInput, FuturePredictor, LatentEncoder, RandomProjectionExperienceEncoder,
    StasisFuturePredictor,
};
use pete_ledger::{
    future_input_from_transition, future_target_from_transition, ExperienceTransition, JsonlLedger,
};
use pete_models::{
    ActionValueNetTrainer, ChargeNetTrainer, DangerNetTrainer, EarNextNetTrainer,
    ExperienceAutoencoderTrainer, EyeNextNetTrainer, FutureNetTrainer,
    HardcodedActionValuePredictor, HardcodedChargePredictor, HardcodedDangerPredictor, TrainStats,
};
use pete_now::Now;
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
            Self::Experience => "experience.no_latent_yet",
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

#[derive(Clone, Debug)]
pub struct TrainUnifiedExperienceRequest {
    pub ledger_path: PathBuf,
    pub checkpoint_path: PathBuf,
    pub report_path: PathBuf,
    pub epochs: usize,
    pub validation_split: f32,
    pub seed: u64,
    pub z_dim: usize,
    pub teacher_dim: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainLatentRoundTripReport {
    pub schema_version: u32,
    pub input_source: String,
    pub architecture: LatentRoundTripArchitectureReport,
    pub transition_count: usize,
    pub train_transition_count: usize,
    pub eval_transition_count: usize,
    pub epochs: usize,
    pub z_dim: usize,
    pub checkpoints: LatentRoundTripCheckpoints,
    pub reconstruction: LatentReconstructionReport,
    pub predictors: Vec<LatentPredictorReport>,
    pub baseline_comparisons: LatentBaselineComparisons,
    pub codebook: Option<CodebookUsageReport>,
    pub verdict: String,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentRoundTripCheckpoints {
    pub experience: PathBuf,
    pub future_trained: PathBuf,
    pub future_random: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentRoundTripArchitectureReport {
    pub pipeline: Vec<String>,
    pub teacher_vectors: Vec<TeacherVectorReport>,
    pub instant: MechanicalInstantReport,
    pub encoder: ExperienceEncoderReport,
    pub owned_latent: OwnedExperienceLatentReport,
    pub heads: Vec<LatentHeadReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TeacherVectorReport {
    pub name: String,
    pub source: String,
    pub purpose: String,
    pub dim: usize,
    pub sample_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MechanicalInstantReport {
    pub representation: String,
    pub assembly: String,
    pub sample_count: usize,
    pub input_dim: usize,
    pub decode_target_dim: usize,
    pub decode_target_kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperienceEncoderReport {
    pub name: String,
    pub input_dim: usize,
    pub z_dim: usize,
    pub checkpoint_path: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OwnedExperienceLatentReport {
    pub name: String,
    pub owner: String,
    pub dim: usize,
    pub teacher_independent: bool,
    pub evidence: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentHeadReport {
    pub name: String,
    pub target: String,
    pub checkpoint_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentReconstructionReport {
    pub sample_count: usize,
    pub trained_decoder_loss_mean: f32,
    pub zero_decoder_loss_mean: f32,
    pub target_kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentPredictorReport {
    pub encoder: String,
    pub target_kind: String,
    pub train_sample_count: usize,
    pub eval_sample_count: usize,
    pub latent_dim: usize,
    pub target_dim: usize,
    pub model_loss_mean: f32,
    pub stasis_loss_mean: f32,
    pub improvement_ratio: Option<f32>,
    pub predictive: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatentBaselineComparisons {
    pub trained_encoder: String,
    pub copy_current_loss_mean: Option<f32>,
    pub random_projection_loss_mean: Option<f32>,
    pub evolved_vector_loss_mean: Option<f32>,
    pub trained_loss_mean: Option<f32>,
    pub trained_beats_copy_current: bool,
    pub trained_beats_random_projection: bool,
    pub trained_beats_evolved_vector: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainUnifiedExperienceReport {
    pub schema_version: u32,
    pub input_source: String,
    pub example_count: usize,
    pub train_example_count: usize,
    pub eval_example_count: usize,
    pub transition_count: usize,
    pub epochs: usize,
    pub teacher_dim: usize,
    pub latent_dim: usize,
    pub checkpoint_path: PathBuf,
    pub future_checkpoint_path: PathBuf,
    pub instant: UnifiedInstantReport,
    pub modality_coverage: Vec<UnifiedModalityCoverage>,
    pub reconstruction: UnifiedReconstructionReport,
    pub predictors: Vec<LatentPredictorReport>,
    pub learned_loop: UnifiedLearnedLoopReport,
    pub baselines: UnifiedBaselineReport,
    pub verdict: String,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedInstantReport {
    pub representation: String,
    pub teacher_slots: Vec<String>,
    pub input_dim: usize,
    pub mask_dim: usize,
    pub target_dim: usize,
    pub assembly: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedModalityCoverage {
    pub slot: String,
    pub source: String,
    pub purpose: String,
    pub dim: usize,
    pub placeholder: bool,
    pub present_count: usize,
    pub missing_count: usize,
    pub coverage: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedReconstructionReport {
    pub sample_count: usize,
    pub total_loss_mean: f32,
    pub zero_loss_mean: f32,
    pub head_losses: BTreeMap<String, f32>,
    pub zero_head_losses: BTreeMap<String, f32>,
    pub reconstructive: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedBaselineReport {
    pub copy_current_loss_mean: Option<f32>,
    pub random_projection_loss_mean: Option<f32>,
    pub mechanical_instant_loss_mean: Option<f32>,
    pub trained_loss_mean: Option<f32>,
    pub trained_beats_copy_current: bool,
    pub trained_beats_random_projection: bool,
    pub trained_beats_mechanical_instant: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedLearnedLoopReport {
    pub canonical_instant: String,
    pub canonical_latent: String,
    pub prediction: String,
    pub surprise: String,
    pub sample_count: usize,
    pub reconstruction_loss_mean: f32,
    pub prediction_loss_mean: f32,
    pub combined_surprise_mean: f32,
    pub confidence_mean: f32,
    pub records: Vec<UnifiedExperienceLoopRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedExperienceLoopRecord {
    pub t_ms: TimeMs,
    pub offset_ms: TimeMs,
    pub encoded_latent: Vec<f32>,
    pub predicted_next_latent: Vec<f32>,
    pub actual_next_latent: Vec<f32>,
    pub reconstruction_loss: f32,
    pub prediction_loss: f32,
    pub combined_surprise: f32,
    pub confidence: f32,
    pub teacher_coverage: f32,
    pub missing_modality_mask: Vec<f32>,
    pub baseline_comparisons: UnifiedExperienceLoopBaselines,
    pub surprise: ExperienceSurprise,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnifiedExperienceLoopBaselines {
    pub copy_current_prediction_loss: f32,
    pub random_projection_prediction_loss: Option<f32>,
    pub mechanical_instant_prediction_loss: Option<f32>,
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
    let architecture = latent_architecture_report(
        "ledger-replay",
        "trainable-autoencoder",
        &checkpoints.experience,
        &checkpoints.future_trained,
        "compact body/memory/drive/prediction/range-depth/audio-summary features",
        &experience_train
            .iter()
            .map(|(_, input, target, _)| (input, target))
            .collect::<Vec<_>>(),
        z_dim,
    )?;
    let trained_train = trained_latent_future_samples(&autoencoder, &train)?;
    let trained_eval = trained_latent_future_samples(&autoencoder, eval_transitions)?;
    let trained_report = train_and_evaluate_future_latents(
        "trainable-autoencoder",
        trained_train.clone(),
        trained_eval,
        request.epochs,
        &checkpoints.future_trained,
        "next trained latent",
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
        "next random-projected latent",
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

    let predictors = vec![trained_report, random_report];
    let baseline_comparisons = latent_baseline_comparisons(
        &predictors,
        "trainable-autoencoder",
        "random-projection",
        None,
    );
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
        schema_version: 2,
        input_source: format!("ledger:{}", request.ledger_path.display()),
        architecture,
        transition_count,
        train_transition_count: train.len(),
        eval_transition_count: eval_transitions.len(),
        epochs: request.epochs,
        z_dim,
        checkpoints,
        reconstruction,
        predictors,
        baseline_comparisons,
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

pub async fn train_unified_experience(
    request: TrainUnifiedExperienceRequest,
) -> Result<TrainUnifiedExperienceReport> {
    let transitions = load_transitions(&request.ledger_path).await?;
    let transition_count = transitions.len();
    let mut examples = unified_examples_from_transitions(&transitions, request.teacher_dim)?;
    let example_count = examples.len();
    if example_count == 0 {
        bail!("no usable unified Experience teacher-vector examples");
    }
    let coverage = unified_modality_coverage(&examples);
    let (train, eval) = split_samples(
        std::mem::take(&mut examples),
        request.validation_split,
        request.seed,
    );
    let eval_examples = if eval.is_empty() { &train } else { &eval };
    let first = train
        .first()
        .ok_or_else(|| anyhow!("no training examples for unified Experience"))?;
    let input_dim = first.input.flat_features().len();
    let decode_lengths = first.target.feature_lengths();
    let z_dim = request.z_dim.clamp(2, input_dim.max(2));
    let mut autoencoder = ExperienceAutoencoderTrainer::new(input_dim, z_dim, decode_lengths);
    for _epoch in 0..request.epochs {
        for sample in &train {
            if sample.input.flat_features().len() == autoencoder.input_dim()
                && sample.target.feature_lengths() == autoencoder.decode_lengths()
            {
                autoencoder.train_step(&sample.input, &sample.target)?;
            }
        }
    }
    autoencoder.save_checkpoint(&request.checkpoint_path)?;

    let reconstruction = evaluate_unified_reconstruction(&autoencoder, eval_examples)?;
    let trained_train = unified_trained_future_samples(&autoencoder, &train)?;
    let trained_eval = unified_trained_future_samples(&autoencoder, eval_examples)?;
    let future_checkpoint_path = request.checkpoint_path.join("future-trained");
    let trained_report = train_and_evaluate_future_latents(
        "unified-experience-latent",
        trained_train.clone(),
        trained_eval,
        request.epochs,
        &future_checkpoint_path,
        "next unified Experience latent",
    )?;
    let future_input_dim = first_dim(&trained_train, |(_, input, _)| input.flat_features().len())?;
    let future_trainer =
        FutureNetTrainer::load_checkpoint(&future_checkpoint_path, future_input_dim, z_dim)?;
    let mut random_train_encoder = RandomProjectionExperienceEncoder::new(z_dim, request.seed);
    let mut random_eval_encoder = RandomProjectionExperienceEncoder::new(z_dim, request.seed);
    let random_report = train_and_evaluate_future_latents(
        "random-projection",
        unified_encoded_future_samples(&mut random_train_encoder, &train)?,
        unified_encoded_future_samples(&mut random_eval_encoder, eval_examples)?,
        request.epochs,
        &request.checkpoint_path.join("future-random"),
        "next random-projected unified latent",
    )?;
    let mechanical_report = train_and_evaluate_future_latents(
        "mechanical-instant",
        unified_mechanical_future_samples(&train),
        unified_mechanical_future_samples(eval_examples),
        request.epochs,
        &request.checkpoint_path.join("future-mechanical-instant"),
        "next mechanical Instant",
    )?;
    let predictors = vec![
        trained_report.clone(),
        random_report.clone(),
        mechanical_report.clone(),
    ];
    let baselines = UnifiedBaselineReport {
        copy_current_loss_mean: Some(trained_report.stasis_loss_mean),
        random_projection_loss_mean: Some(random_report.model_loss_mean),
        mechanical_instant_loss_mean: Some(mechanical_report.model_loss_mean),
        trained_loss_mean: Some(trained_report.model_loss_mean),
        trained_beats_copy_current: trained_report.model_loss_mean
            < trained_report.stasis_loss_mean,
        trained_beats_random_projection: trained_report.model_loss_mean
            < random_report.model_loss_mean,
        trained_beats_mechanical_instant: trained_report.model_loss_mean
            < mechanical_report.model_loss_mean,
    };
    let learned_loop = unified_learned_loop_report(
        &autoencoder,
        &future_trainer,
        eval_examples,
        &baselines,
        random_report.model_loss_mean,
        mechanical_report.model_loss_mean,
    )?;
    let mut warnings = Vec::new();
    if example_count < 50 {
        warnings.push(format!(
            "insufficient data: {example_count} examples is below the conservative 50-example floor"
        ));
    }
    for slot in &coverage {
        if slot.present_count == 0 {
            warnings.push(format!(
                "teacher slot {} was explicitly masked as missing for every example",
                slot.slot
            ));
        }
    }
    let collapsed_latent = unified_latent_variance(&autoencoder, eval_examples)? < 1.0e-6;
    if collapsed_latent {
        warnings.push(
            "learned unified Experience latent appears collapsed on held-out examples".to_string(),
        );
    }
    let verdict = if reconstruction.reconstructive && trained_report.predictive && !collapsed_latent
    {
        "unified Experience latent is reconstructive and predictive".to_string()
    } else if reconstruction.reconstructive {
        "unified Experience latent reconstructs teacher/sensor heads but is not yet predictive"
            .to_string()
    } else {
        "unified Experience latent is not yet proven reconstructive or predictive".to_string()
    };
    let report = TrainUnifiedExperienceReport {
        schema_version: 1,
        input_source: format!("ledger:{}", request.ledger_path.display()),
        example_count,
        train_example_count: train.len(),
        eval_example_count: eval_examples.len(),
        transition_count,
        epochs: request.epochs,
        teacher_dim: request.teacher_dim,
        latent_dim: z_dim,
        checkpoint_path: request.checkpoint_path.clone(),
        future_checkpoint_path,
        instant: UnifiedInstantReport {
            representation: "UnifiedExperienceInstant".to_string(),
            teacher_slots: UNIFIED_TEACHER_SLOTS.iter().map(|slot| slot.name.to_string()).collect(),
            input_dim,
            mask_dim: UNIFIED_TEACHER_SLOTS.len(),
            target_dim: first.target.flat_features().len(),
            assembly: "fixed teacher-vector slots plus explicit presence mask; missing modalities stay masked instead of disappearing".to_string(),
        },
        modality_coverage: coverage,
        reconstruction,
        predictors,
        learned_loop,
        baselines,
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
#[path = "lib_tests.rs"]
mod tests;
