//! A deliberately small NEAT nervous system for PETE locomotion.
//!
//! This crate owns policy proposals, never physical authority. Its output is
//! still expected to pass through `pete-autonomic`, the cockpit lease and the
//! brainstem safety/reflex layer.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use pete_behaviors::{FunctionBehavior, OutputDistance};
use pete_body::BodySense;
use pete_core::Pose2;
use pete_now::RangeSense;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

pub const LOCOMOTION_SCHEMA_VERSION: u32 = 1;
pub const LOCOMOTION_INPUT_COUNT: usize = 17;
pub const LOCOMOTION_OUTPUT_COUNT: usize = 3;
pub const LOCOMOTION_INPUT_NAMES: [&str; LOCOMOTION_INPUT_COUNT] = [
    "bump_left",
    "bump_right",
    "bump_front",
    "left_wheel_travel",
    "right_wheel_travel",
    "forward_velocity",
    "angular_velocity",
    "distance_since_collision",
    "time_since_collision",
    "recent_turn_direction",
    "clearance_left",
    "clearance_front",
    "clearance_right",
    "last_forward_command",
    "last_turn_command",
    "battery_level",
    "recent_collision_rate",
];
pub const LOCOMOTION_OUTPUT_NAMES: [&str; LOCOMOTION_OUTPUT_COUNT] = [
    "forward_velocity",
    "angular_velocity",
    "recovery_activation",
];

const CREATE_WHEEL_BASE_M: f32 = 0.235;
const MAX_RANGE_M: f32 = 4.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocomotionInput {
    pub schema_version: u32,
    pub bump_left: f32,
    pub bump_right: f32,
    pub bump_front: f32,
    pub left_wheel_travel_m: f32,
    pub right_wheel_travel_m: f32,
    pub forward_velocity_m_s: f32,
    pub angular_velocity_rad_s: f32,
    pub distance_since_collision_m: f32,
    pub time_since_collision_s: f32,
    /// Negative is a recent right turn; positive is a recent left turn.
    pub recent_turn_direction: f32,
    pub clearance_left_m: f32,
    pub clearance_front_m: f32,
    pub clearance_right_m: f32,
    pub last_forward_command_m_s: f32,
    pub last_turn_command_rad_s: f32,
    pub battery_level: f32,
    pub recent_collision_rate: f32,
}

impl Default for LocomotionInput {
    fn default() -> Self {
        Self {
            schema_version: LOCOMOTION_SCHEMA_VERSION,
            bump_left: 0.0,
            bump_right: 0.0,
            bump_front: 0.0,
            left_wheel_travel_m: 0.0,
            right_wheel_travel_m: 0.0,
            forward_velocity_m_s: 0.0,
            angular_velocity_rad_s: 0.0,
            distance_since_collision_m: 0.0,
            time_since_collision_s: 0.0,
            recent_turn_direction: 0.0,
            clearance_left_m: MAX_RANGE_M,
            clearance_front_m: MAX_RANGE_M,
            clearance_right_m: MAX_RANGE_M,
            last_forward_command_m_s: 0.0,
            last_turn_command_rad_s: 0.0,
            battery_level: 1.0,
            recent_collision_rate: 0.0,
        }
    }
}

impl LocomotionInput {
    /// Stable, normalized network order. Changing it requires a schema bump.
    pub fn features(&self) -> [f32; LOCOMOTION_INPUT_COUNT] {
        [
            unit(self.bump_left),
            unit(self.bump_right),
            unit(self.bump_front),
            (self.left_wheel_travel_m / 5.0).tanh(),
            (self.right_wheel_travel_m / 5.0).tanh(),
            (self.forward_velocity_m_s / 0.6).clamp(-1.0, 1.0),
            (self.angular_velocity_rad_s / 1.5).clamp(-1.0, 1.0),
            (self.distance_since_collision_m / 4.0).clamp(0.0, 1.0),
            (self.time_since_collision_s / 20.0).clamp(0.0, 1.0),
            self.recent_turn_direction.clamp(-1.0, 1.0),
            normalize_clearance(self.clearance_left_m),
            normalize_clearance(self.clearance_front_m),
            normalize_clearance(self.clearance_right_m),
            (self.last_forward_command_m_s / 0.6).clamp(-1.0, 1.0),
            (self.last_turn_command_rad_s / 1.5).clamp(-1.0, 1.0),
            unit(self.battery_level),
            unit(self.recent_collision_rate),
        ]
    }

    pub fn collision_active(&self) -> bool {
        self.bump_left > 0.5 || self.bump_right > 0.5 || self.bump_front > 0.5
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LocomotionOutput {
    pub forward_velocity_m_s: f32,
    pub angular_velocity_rad_s: f32,
    pub recovery_activation: f32,
}

impl LocomotionOutput {
    pub fn bounded(self, max_forward_m_s: f32, max_turn_rad_s: f32) -> Self {
        Self {
            forward_velocity_m_s: finite_or_zero(self.forward_velocity_m_s)
                .clamp(-max_forward_m_s.abs(), max_forward_m_s.abs()),
            angular_velocity_rad_s: finite_or_zero(self.angular_velocity_rad_s)
                .clamp(-max_turn_rad_s.abs(), max_turn_rad_s.abs()),
            recovery_activation: finite_or_zero(self.recovery_activation).clamp(0.0, 1.0),
        }
    }

    /// Recovery is an intent, not a safety-latch override. It only makes the
    /// proposed linear component non-positive; downstream gates still decide.
    pub fn with_recovery_intent(mut self, threshold: f32) -> Self {
        if self.recovery_activation >= threshold {
            self.forward_velocity_m_s = -self.forward_velocity_m_s.abs();
        }
        self
    }
}

impl OutputDistance for LocomotionOutput {
    fn distance(&self, other: &Self) -> f32 {
        let df = self.forward_velocity_m_s - other.forward_velocity_m_s;
        let dt = self.angular_velocity_rad_s - other.angular_velocity_rad_s;
        let dr = self.recovery_activation - other.recovery_activation;
        (df.mul_add(df, dt.mul_add(dt, dr * dr))).sqrt()
    }
}

#[derive(Clone, Debug)]
pub struct HardcodedLocomotionBehavior {
    pub forward_velocity_m_s: f32,
    pub angular_velocity_rad_s: f32,
}

impl Default for HardcodedLocomotionBehavior {
    fn default() -> Self {
        Self {
            forward_velocity_m_s: 0.2,
            angular_velocity_rad_s: 0.1,
        }
    }
}

impl FunctionBehavior<LocomotionInput, LocomotionOutput> for HardcodedLocomotionBehavior {
    fn id(&self) -> &'static str {
        "locomotion.hardcoded_wander.v0"
    }

    fn infer(&mut self, _input: &LocomotionInput) -> Result<LocomotionOutput> {
        // Preserve the exact ancestral Explore motor mapping. Bump/cliff
        // recovery remains in the existing simulator/possession reflex paths.
        Ok(LocomotionOutput {
            forward_velocity_m_s: self.forward_velocity_m_s,
            angular_velocity_rad_s: self.angular_velocity_rad_s,
            recovery_activation: 0.0,
        })
    }
}

/// Stateful conversion from body/range snapshots to the small nervous system.
#[derive(Clone, Debug, Default)]
pub struct LocomotionTracker {
    started_at_ms: Option<u64>,
    last_t_ms: Option<u64>,
    last_pose: Option<Pose2>,
    cumulative_distance_m: f32,
    cumulative_heading_rad: f32,
    distance_since_collision_m: f32,
    last_collision_ms: Option<u64>,
    collision_times_ms: Vec<u64>,
    collision_active: bool,
    recent_turn_direction: f32,
    last_forward_command_m_s: f32,
    last_turn_command_rad_s: f32,
}

impl LocomotionTracker {
    pub fn observe(&mut self, t_ms: u64, body: &BodySense, range: &RangeSense) -> LocomotionInput {
        let started_at_ms = *self.started_at_ms.get_or_insert(t_ms);
        let dt_s = self
            .last_t_ms
            .map(|last| t_ms.saturating_sub(last) as f32 / 1_000.0)
            .filter(|dt| *dt > 0.0)
            .unwrap_or(0.0);

        let (distance_delta, heading_delta) = self
            .last_pose
            .map(|last| pose_delta(last, body.odometry))
            .unwrap_or((0.0, 0.0));
        self.cumulative_distance_m += distance_delta;
        self.cumulative_heading_rad += heading_delta;
        self.distance_since_collision_m += distance_delta.abs();

        let collision = body.flags.bump_left || body.flags.bump_right;
        if collision && !self.collision_active {
            self.last_collision_ms = Some(t_ms);
            self.distance_since_collision_m = 0.0;
            self.collision_times_ms.push(t_ms);
        }
        self.collision_active = collision;
        self.collision_times_ms
            .retain(|stamp| t_ms.saturating_sub(*stamp) <= 10_000);

        let derived_forward = if dt_s > 0.0 {
            distance_delta / dt_s
        } else {
            0.0
        };
        let derived_turn = if dt_s > 0.0 {
            heading_delta / dt_s
        } else {
            0.0
        };
        let forward_velocity = prefer_measured(body.velocity.forward_m_s, derived_forward);
        let angular_velocity = prefer_measured(body.velocity.turn_rad_s, derived_turn);
        self.recent_turn_direction =
            self.recent_turn_direction * 0.8 + (angular_velocity / 1.5).clamp(-1.0, 1.0) * 0.2;

        let (clearance_left_m, clearance_front_m, clearance_right_m) = range_sectors(range);
        let left_wheel_travel_m =
            self.cumulative_distance_m - self.cumulative_heading_rad * CREATE_WHEEL_BASE_M * 0.5;
        let right_wheel_travel_m =
            self.cumulative_distance_m + self.cumulative_heading_rad * CREATE_WHEEL_BASE_M * 0.5;

        let input = LocomotionInput {
            schema_version: LOCOMOTION_SCHEMA_VERSION,
            bump_left: bool_unit(body.flags.bump_left),
            bump_right: bool_unit(body.flags.bump_right),
            bump_front: bool_unit(body.flags.bump_left && body.flags.bump_right),
            left_wheel_travel_m,
            right_wheel_travel_m,
            forward_velocity_m_s: forward_velocity,
            angular_velocity_rad_s: angular_velocity,
            distance_since_collision_m: self.distance_since_collision_m,
            time_since_collision_s: self
                .last_collision_ms
                .map(|stamp| t_ms.saturating_sub(stamp) as f32 / 1_000.0)
                .unwrap_or_else(|| t_ms.saturating_sub(started_at_ms) as f32 / 1_000.0),
            recent_turn_direction: self.recent_turn_direction,
            clearance_left_m,
            clearance_front_m,
            clearance_right_m,
            last_forward_command_m_s: self.last_forward_command_m_s,
            last_turn_command_rad_s: self.last_turn_command_rad_s,
            battery_level: body.battery_level,
            recent_collision_rate: self.collision_times_ms.len() as f32 / 10.0,
        };

        self.last_t_ms = Some(t_ms);
        self.last_pose = Some(body.odometry);
        input
    }

    pub fn observe_command(&mut self, output: LocomotionOutput) {
        self.last_forward_command_m_s = output.forward_velocity_m_s;
        self.last_turn_command_rad_s = output.angular_velocity_rad_s;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Input,
    Bias,
    Hidden,
    Output,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeGene {
    pub id: u32,
    pub kind: NodeKind,
    pub layer: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConnectionGene {
    pub innovation: u64,
    pub from: u32,
    pub to: u32,
    pub weight: f32,
    pub enabled: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Genome {
    pub input_count: usize,
    pub output_count: usize,
    pub nodes: Vec<NodeGene>,
    pub connections: Vec<ConnectionGene>,
}

impl Genome {
    pub fn minimal<R: Rng + ?Sized>(
        input_count: usize,
        output_count: usize,
        innovations: &mut InnovationTracker,
        rng: &mut R,
    ) -> Self {
        let mut nodes = Vec::with_capacity(input_count + output_count + 1);
        for id in 0..input_count as u32 {
            nodes.push(NodeGene {
                id,
                kind: NodeKind::Input,
                layer: 0.0,
            });
        }
        let bias_id = input_count as u32;
        nodes.push(NodeGene {
            id: bias_id,
            kind: NodeKind::Bias,
            layer: 0.0,
        });
        for index in 0..output_count as u32 {
            nodes.push(NodeGene {
                id: bias_id + 1 + index,
                kind: NodeKind::Output,
                layer: 1.0,
            });
        }
        let mut connections = Vec::new();
        for from in 0..=bias_id {
            for index in 0..output_count as u32 {
                let to = bias_id + 1 + index;
                connections.push(ConnectionGene {
                    innovation: innovations.connection(from, to),
                    from,
                    to,
                    weight: rng.gen_range(-1.0..1.0),
                    enabled: true,
                });
            }
        }
        Self {
            input_count,
            output_count,
            nodes,
            connections,
        }
    }

    pub fn activate(&self, inputs: &[f32]) -> Result<Vec<f32>> {
        if inputs.len() != self.input_count {
            bail!(
                "NEAT input mismatch: genome expects {}, received {}",
                self.input_count,
                inputs.len()
            );
        }
        let mut values: HashMap<u32, f32> = HashMap::with_capacity(self.nodes.len());
        for (index, value) in inputs.iter().copied().enumerate() {
            values.insert(index as u32, finite_or_zero(value));
        }
        values.insert(self.input_count as u32, 1.0);

        let mut ordered = self.nodes.iter().collect::<Vec<_>>();
        ordered.sort_by(|left, right| {
            left.layer
                .total_cmp(&right.layer)
                .then_with(|| left.id.cmp(&right.id))
        });
        for node in ordered {
            if matches!(node.kind, NodeKind::Input | NodeKind::Bias) {
                continue;
            }
            let sum = self
                .connections
                .iter()
                .filter(|edge| edge.enabled && edge.to == node.id)
                .map(|edge| values.get(&edge.from).copied().unwrap_or(0.0) * edge.weight)
                .sum::<f32>();
            values.insert(node.id, sum.tanh());
        }
        let first_output = self.input_count as u32 + 1;
        Ok((0..self.output_count as u32)
            .map(|index| values.get(&(first_output + index)).copied().unwrap_or(0.0))
            .collect())
    }

    pub fn compatibility_distance(&self, other: &Self, config: NeatConfig) -> f32 {
        let left = self
            .connections
            .iter()
            .map(|gene| (gene.innovation, gene))
            .collect::<BTreeMap<_, _>>();
        let right = other
            .connections
            .iter()
            .map(|gene| (gene.innovation, gene))
            .collect::<BTreeMap<_, _>>();
        let left_max = left.keys().next_back().copied().unwrap_or(0);
        let right_max = right.keys().next_back().copied().unwrap_or(0);
        let mut excess = 0;
        let mut disjoint = 0;
        let mut matching = 0;
        let mut weight_delta = 0.0;
        let innovations = left
            .keys()
            .chain(right.keys())
            .copied()
            .collect::<BTreeSet<_>>();
        for innovation in innovations {
            match (left.get(&innovation), right.get(&innovation)) {
                (Some(a), Some(b)) => {
                    matching += 1;
                    weight_delta += (a.weight - b.weight).abs();
                }
                (Some(_), None) => {
                    if innovation > right_max {
                        excess += 1
                    } else {
                        disjoint += 1
                    }
                }
                (None, Some(_)) => {
                    if innovation > left_max {
                        excess += 1
                    } else {
                        disjoint += 1
                    }
                }
                (None, None) => {}
            }
        }
        let size = self.connections.len().max(other.connections.len()).max(1) as f32;
        config.excess_coefficient * excess as f32 / size
            + config.disjoint_coefficient * disjoint as f32 / size
            + config.weight_coefficient
                * if matching == 0 {
                    0.0
                } else {
                    weight_delta / matching as f32
                }
    }

    fn mutate<R: Rng + ?Sized>(
        &mut self,
        config: NeatConfig,
        innovations: &mut InnovationTracker,
        rng: &mut R,
    ) {
        for connection in &mut self.connections {
            if rng.gen::<f32>() < config.weight_mutation_rate {
                if rng.gen::<f32>() < config.weight_reset_rate {
                    connection.weight = rng.gen_range(-1.0..1.0);
                } else {
                    connection.weight = (connection.weight
                        + rng.gen_range(-config.weight_perturbation..config.weight_perturbation))
                    .clamp(-5.0, 5.0);
                }
            }
        }
        if rng.gen::<f32>() < config.add_connection_rate {
            self.mutate_add_connection(innovations, rng);
        }
        if rng.gen::<f32>() < config.add_node_rate {
            self.mutate_add_node(innovations, rng);
        }
    }

    fn mutate_add_connection<R: Rng + ?Sized>(
        &mut self,
        innovations: &mut InnovationTracker,
        rng: &mut R,
    ) {
        let mut candidates = Vec::new();
        for from in &self.nodes {
            for to in &self.nodes {
                if from.layer >= to.layer
                    || matches!(to.kind, NodeKind::Input | NodeKind::Bias)
                    || self
                        .connections
                        .iter()
                        .any(|edge| edge.from == from.id && edge.to == to.id)
                {
                    continue;
                }
                candidates.push((from.id, to.id));
            }
        }
        if let Some(&(from, to)) = candidates.choose(rng) {
            self.connections.push(ConnectionGene {
                innovation: innovations.connection(from, to),
                from,
                to,
                weight: rng.gen_range(-1.0..1.0),
                enabled: true,
            });
        }
    }

    fn mutate_add_node<R: Rng + ?Sized>(
        &mut self,
        innovations: &mut InnovationTracker,
        rng: &mut R,
    ) {
        let enabled = self
            .connections
            .iter()
            .enumerate()
            .filter(|(_, edge)| edge.enabled)
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let Some(&edge_index) = enabled.choose(rng) else {
            return;
        };
        let old = self.connections[edge_index].clone();
        self.connections[edge_index].enabled = false;
        let node_id = innovations.split_node(old.innovation);
        if !self.nodes.iter().any(|node| node.id == node_id) {
            let from_layer = self
                .nodes
                .iter()
                .find(|node| node.id == old.from)
                .map(|node| node.layer)
                .unwrap_or(0.0);
            let to_layer = self
                .nodes
                .iter()
                .find(|node| node.id == old.to)
                .map(|node| node.layer)
                .unwrap_or(1.0);
            self.nodes.push(NodeGene {
                id: node_id,
                kind: NodeKind::Hidden,
                layer: (from_layer + to_layer) * 0.5,
            });
        }
        for (from, to, weight) in [(old.from, node_id, 1.0), (node_id, old.to, old.weight)] {
            let innovation = innovations.connection(from, to);
            if let Some(existing) = self
                .connections
                .iter_mut()
                .find(|edge| edge.innovation == innovation)
            {
                existing.enabled = true;
                existing.weight = weight;
            } else {
                self.connections.push(ConnectionGene {
                    innovation,
                    from,
                    to,
                    weight,
                    enabled: true,
                });
            }
        }
    }

    fn crossover<R: Rng + ?Sized>(fitter: &Self, other: &Self, rng: &mut R) -> Self {
        let other_genes = other
            .connections
            .iter()
            .map(|gene| (gene.innovation, gene))
            .collect::<HashMap<_, _>>();
        let mut connections = Vec::with_capacity(fitter.connections.len());
        for gene in &fitter.connections {
            let mut child = other_genes
                .get(&gene.innovation)
                .filter(|_| rng.gen::<bool>())
                .copied()
                .unwrap_or(gene)
                .clone();
            if !gene.enabled
                || other_genes
                    .get(&gene.innovation)
                    .is_some_and(|other| !other.enabled)
            {
                child.enabled = rng.gen::<f32>() >= 0.75;
            }
            connections.push(child);
        }
        let mut nodes = fitter.nodes.clone();
        for edge in &connections {
            for id in [edge.from, edge.to] {
                if !nodes.iter().any(|node| node.id == id) {
                    if let Some(node) = other.nodes.iter().find(|node| node.id == id) {
                        nodes.push(node.clone());
                    }
                }
            }
        }
        Self {
            input_count: fitter.input_count,
            output_count: fitter.output_count,
            nodes,
            connections,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NeatConfig {
    pub population_size: usize,
    pub compatibility_threshold: f32,
    pub excess_coefficient: f32,
    pub disjoint_coefficient: f32,
    pub weight_coefficient: f32,
    pub weight_mutation_rate: f32,
    pub weight_reset_rate: f32,
    pub weight_perturbation: f32,
    pub add_connection_rate: f32,
    pub add_node_rate: f32,
    pub crossover_rate: f32,
    pub elitism: usize,
}

impl Default for NeatConfig {
    fn default() -> Self {
        Self {
            population_size: 64,
            compatibility_threshold: 3.0,
            excess_coefficient: 1.0,
            disjoint_coefficient: 1.0,
            weight_coefficient: 0.4,
            weight_mutation_rate: 0.8,
            weight_reset_rate: 0.1,
            weight_perturbation: 0.35,
            add_connection_rate: 0.08,
            add_node_rate: 0.03,
            crossover_rate: 0.75,
            elitism: 2,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InnovationTracker {
    next_innovation: u64,
    next_node_id: u32,
    connections: HashMap<(u32, u32), u64>,
    split_nodes: HashMap<u64, u32>,
}

impl InnovationTracker {
    pub fn new(first_node_id: u32) -> Self {
        Self {
            next_innovation: 0,
            next_node_id: first_node_id,
            connections: HashMap::new(),
            split_nodes: HashMap::new(),
        }
    }

    fn connection(&mut self, from: u32, to: u32) -> u64 {
        if let Some(innovation) = self.connections.get(&(from, to)) {
            return *innovation;
        }
        let innovation = self.next_innovation;
        self.next_innovation += 1;
        self.connections.insert((from, to), innovation);
        innovation
    }

    fn split_node(&mut self, innovation: u64) -> u32 {
        if let Some(id) = self.split_nodes.get(&innovation) {
            return *id;
        }
        let id = self.next_node_id;
        self.next_node_id += 1;
        self.split_nodes.insert(innovation, id);
        id
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Population {
    pub generation: u64,
    pub genomes: Vec<Genome>,
    pub config: NeatConfig,
    innovations: InnovationTracker,
}

impl Population {
    pub fn seeded<R: Rng + ?Sized>(config: NeatConfig, rng: &mut R) -> Self {
        let first_hidden = (LOCOMOTION_INPUT_COUNT + LOCOMOTION_OUTPUT_COUNT + 1) as u32;
        let mut innovations = InnovationTracker::new(first_hidden);
        let seed = Genome::minimal(
            LOCOMOTION_INPUT_COUNT,
            LOCOMOTION_OUTPUT_COUNT,
            &mut innovations,
            rng,
        );
        let genomes = (0..config.population_size)
            .map(|index| {
                let mut genome = seed.clone();
                if index > 0 {
                    genome.mutate(config, &mut innovations, rng);
                }
                genome
            })
            .collect();
        Self {
            generation: 0,
            genomes,
            config,
            innovations,
        }
    }

    pub fn species(&self) -> Vec<Vec<usize>> {
        let mut species: Vec<Vec<usize>> = Vec::new();
        for (index, genome) in self.genomes.iter().enumerate() {
            if let Some(group) = species.iter_mut().find(|group| {
                genome.compatibility_distance(&self.genomes[group[0]], self.config)
                    < self.config.compatibility_threshold
            }) {
                group.push(index);
            } else {
                species.push(vec![index]);
            }
        }
        species
    }

    pub fn evolve<R: Rng + ?Sized>(&mut self, fitness: &[f32], rng: &mut R) -> Result<()> {
        if fitness.len() != self.genomes.len() {
            bail!(
                "fitness count {} does not match population {}",
                fitness.len(),
                self.genomes.len()
            );
        }
        if self.genomes.is_empty() {
            bail!("cannot evolve an empty population");
        }
        let species = self.species();
        let mut species_size = vec![1usize; self.genomes.len()];
        for group in species {
            for index in &group {
                species_size[*index] = group.len();
            }
        }
        let adjusted = fitness
            .iter()
            .enumerate()
            .map(|(index, value)| finite_or_zero(*value) / species_size[index] as f32)
            .collect::<Vec<_>>();
        let mut ranked = (0..self.genomes.len()).collect::<Vec<_>>();
        ranked.sort_by(|a, b| fitness[*b].total_cmp(&fitness[*a]));

        let mut next = ranked
            .iter()
            .take(self.config.elitism.min(self.config.population_size))
            .map(|index| self.genomes[*index].clone())
            .collect::<Vec<_>>();
        while next.len() < self.config.population_size {
            let first = select_parent(&adjusted, rng);
            let mut child = if rng.gen::<f32>() < self.config.crossover_rate {
                let second = select_parent(&adjusted, rng);
                if fitness[first] >= fitness[second] {
                    Genome::crossover(&self.genomes[first], &self.genomes[second], rng)
                } else {
                    Genome::crossover(&self.genomes[second], &self.genomes[first], rng)
                }
            } else {
                self.genomes[first].clone()
            };
            child.mutate(self.config, &mut self.innovations, rng);
            next.push(child);
        }
        self.genomes = next;
        self.generation += 1;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocomotionCheckpoint {
    pub schema_version: u32,
    pub generation: u64,
    pub fitness: f32,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub genome: Genome,
}

impl LocomotionCheckpoint {
    pub fn new(generation: u64, fitness: f32, genome: Genome) -> Self {
        Self {
            schema_version: LOCOMOTION_SCHEMA_VERSION,
            generation,
            fitness,
            input_names: LOCOMOTION_INPUT_NAMES
                .iter()
                .map(|name| name.to_string())
                .collect(),
            output_names: LOCOMOTION_OUTPUT_NAMES
                .iter()
                .map(|name| name.to_string())
                .collect(),
            genome,
        }
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = checkpoint_path(path.as_ref());
        let checkpoint: Self = serde_json::from_slice(
            &fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", path.display()))?;
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let path = checkpoint_path(path.as_ref());
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, serde_json::to_vec_pretty(self)?)
            .with_context(|| format!("failed to write {}", path.display()))
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != LOCOMOTION_SCHEMA_VERSION {
            bail!(
                "unsupported locomotion schema {}, expected {}",
                self.schema_version,
                LOCOMOTION_SCHEMA_VERSION
            );
        }
        if self.input_names != LOCOMOTION_INPUT_NAMES
            || self.output_names != LOCOMOTION_OUTPUT_NAMES
        {
            bail!("locomotion checkpoint feature order does not match this runtime");
        }
        if self.genome.input_count != LOCOMOTION_INPUT_COUNT
            || self.genome.output_count != LOCOMOTION_OUTPUT_COUNT
        {
            bail!("locomotion checkpoint network dimensions do not match this runtime");
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct NeatLocomotionBehavior {
    pub checkpoint: LocomotionCheckpoint,
    pub max_forward_m_s: f32,
    pub max_turn_rad_s: f32,
}

impl NeatLocomotionBehavior {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            checkpoint: LocomotionCheckpoint::load(path)?,
            max_forward_m_s: 0.6,
            max_turn_rad_s: 1.0,
        })
    }
}

impl FunctionBehavior<LocomotionInput, LocomotionOutput> for NeatLocomotionBehavior {
    fn id(&self) -> &'static str {
        "locomotion.neat.v0"
    }

    fn infer(&mut self, input: &LocomotionInput) -> Result<LocomotionOutput> {
        let values = self.checkpoint.genome.activate(&input.features())?;
        Ok(LocomotionOutput {
            forward_velocity_m_s: values[0] * self.max_forward_m_s,
            angular_velocity_rad_s: values[1] * self.max_turn_rad_s,
            recovery_activation: (values[2] + 1.0) * 0.5,
        }
        .bounded(self.max_forward_m_s, self.max_turn_rad_s)
        .with_recovery_intent(0.5))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EpisodeMetrics {
    pub new_area_cells: u32,
    pub distance_without_collision_m: f32,
    pub successful_escapes: u32,
    pub escape_boundary_crossings: u32,
    pub trap_mouth_progress_m: f32,
    pub collisions: u32,
    pub repeated_state_steps: u32,
    pub wheel_motion_m: f32,
    pub angular_motion_rad: f32,
    pub stalled_steps: u32,
    pub safety_vetoes: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitnessWeights {
    pub new_area: f32,
    pub collision_free_distance: f32,
    pub successful_escape: f32,
    pub escape_boundary: f32,
    pub trap_mouth_progress: f32,
    pub collision: f32,
    pub repeated_state: f32,
    pub energy: f32,
    pub angular_motion: f32,
    pub stalled: f32,
    pub safety_veto: f32,
}

impl FitnessWeights {
    pub fn collision_recovery() -> Self {
        Self {
            new_area: 0.2,
            collision_free_distance: 1.0,
            successful_escape: 8.0,
            escape_boundary: 12.0,
            trap_mouth_progress: 4.0,
            collision: 3.0,
            repeated_state: 0.1,
            energy: 0.05,
            angular_motion: 0.03,
            stalled: 0.1,
            safety_veto: 2.0,
        }
    }

    pub fn efficient_wandering() -> Self {
        Self {
            new_area: 2.0,
            collision_free_distance: 1.5,
            successful_escape: 3.0,
            escape_boundary: 6.0,
            trap_mouth_progress: 2.0,
            collision: 5.0,
            repeated_state: 0.3,
            energy: 0.15,
            angular_motion: 0.08,
            stalled: 0.2,
            safety_veto: 3.0,
        }
    }

    pub fn score(self, metrics: EpisodeMetrics) -> f32 {
        self.new_area * metrics.new_area_cells as f32
            + self.collision_free_distance * metrics.distance_without_collision_m
            + self.successful_escape * metrics.successful_escapes as f32
            + self.escape_boundary * metrics.escape_boundary_crossings as f32
            + self.trap_mouth_progress * metrics.trap_mouth_progress_m
            - self.collision * metrics.collisions as f32
            - self.repeated_state * metrics.repeated_state_steps as f32
            - self.energy * metrics.wheel_motion_m
            - self.angular_motion * metrics.angular_motion_rad
            - self.stalled * metrics.stalled_steps as f32
            - self.safety_veto * metrics.safety_vetoes as f32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurriculumStage {
    BackAwayReliably,
    ChooseUsefulTurn,
    EscapeCorners,
    ExploreWithoutLooping,
    NavigateVariedRooms,
    TransferCandidatesToPete,
}

impl CurriculumStage {
    pub const ORDER: [Self; 6] = [
        Self::BackAwayReliably,
        Self::ChooseUsefulTurn,
        Self::EscapeCorners,
        Self::ExploreWithoutLooping,
        Self::NavigateVariedRooms,
        Self::TransferCandidatesToPete,
    ];

    pub fn next(self) -> Option<Self> {
        Self::ORDER
            .iter()
            .position(|stage| *stage == self)
            .and_then(|index| Self::ORDER.get(index + 1))
            .copied()
    }

    pub fn evolves_population(self) -> bool {
        self != Self::TransferCandidatesToPete
    }

    pub fn weights(self) -> FitnessWeights {
        match self {
            Self::BackAwayReliably => FitnessWeights {
                successful_escape: 12.0,
                collision: 5.0,
                safety_veto: 5.0,
                ..FitnessWeights::collision_recovery()
            },
            Self::ChooseUsefulTurn => FitnessWeights {
                successful_escape: 10.0,
                angular_motion: 0.08,
                repeated_state: 0.2,
                ..FitnessWeights::collision_recovery()
            },
            Self::EscapeCorners => FitnessWeights {
                successful_escape: 14.0,
                escape_boundary: 18.0,
                trap_mouth_progress: 8.0,
                collision: 5.0,
                repeated_state: 0.25,
                ..FitnessWeights::collision_recovery()
            },
            Self::ExploreWithoutLooping => FitnessWeights::efficient_wandering(),
            Self::NavigateVariedRooms => FitnessWeights {
                new_area: 2.5,
                escape_boundary: 8.0,
                trap_mouth_progress: 3.0,
                collision: 6.0,
                safety_veto: 4.0,
                ..FitnessWeights::efficient_wandering()
            },
            Self::TransferCandidatesToPete => FitnessWeights {
                escape_boundary: 10.0,
                trap_mouth_progress: 4.0,
                safety_veto: 20.0,
                collision: 8.0,
                ..FitnessWeights::efficient_wandering()
            },
        }
    }

    pub fn promotion_criteria(self) -> PromotionCriteria {
        match self {
            Self::BackAwayReliably => PromotionCriteria::new(40, 0.90, 0.20, false),
            Self::ChooseUsefulTurn => PromotionCriteria::new(60, 0.85, 0.18, false),
            Self::EscapeCorners => PromotionCriteria::new(100, 0.80, 0.15, false),
            Self::ExploreWithoutLooping => PromotionCriteria::new(150, 0.80, 0.10, true),
            Self::NavigateVariedRooms => PromotionCriteria::new(300, 0.85, 0.08, true),
            Self::TransferCandidatesToPete => PromotionCriteria::new(500, 0.90, 0.05, true),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromotionCriteria {
    pub minimum_seeded_episodes: u32,
    pub minimum_success_rate: f32,
    pub maximum_collision_rate: f32,
    pub must_beat_hardcoded: bool,
    pub maximum_safety_invariant_violations: u32,
}

impl PromotionCriteria {
    pub const fn new(
        minimum_seeded_episodes: u32,
        minimum_success_rate: f32,
        maximum_collision_rate: f32,
        must_beat_hardcoded: bool,
    ) -> Self {
        Self {
            minimum_seeded_episodes,
            minimum_success_rate,
            maximum_collision_rate,
            must_beat_hardcoded,
            maximum_safety_invariant_violations: 0,
        }
    }

    pub fn accepts(self, evaluation: CandidateEvaluation) -> bool {
        evaluation.seeded_episodes >= self.minimum_seeded_episodes
            && evaluation.success_rate >= self.minimum_success_rate
            && evaluation.collision_rate <= self.maximum_collision_rate
            && evaluation.safety_invariant_violations <= self.maximum_safety_invariant_violations
            && (!self.must_beat_hardcoded || evaluation.beats_hardcoded)
            && evaluation.noise_robust
            && evaluation.motor_mismatch_robust
            && evaluation.fallback_verified
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CandidateEvaluation {
    pub seeded_episodes: u32,
    pub success_rate: f32,
    pub collision_rate: f32,
    pub safety_invariant_violations: u32,
    pub beats_hardcoded: bool,
    pub noise_robust: bool,
    pub motor_mismatch_robust: bool,
    pub fallback_verified: bool,
}

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

fn select_parent<R: Rng + ?Sized>(fitness: &[f32], rng: &mut R) -> usize {
    let min = fitness.iter().copied().fold(f32::INFINITY, f32::min);
    let shift = if min <= 0.0 { -min + 1.0e-3 } else { 0.0 };
    let total = fitness.iter().map(|value| value + shift).sum::<f32>();
    if !total.is_finite() || total <= 0.0 {
        return rng.gen_range(0..fitness.len());
    }
    let mut needle = rng.gen_range(0.0..total);
    for (index, value) in fitness.iter().enumerate() {
        needle -= value + shift;
        if needle <= 0.0 {
            return index;
        }
    }
    fitness.len() - 1
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

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn hardcoded_policy_preserves_ancestral_wander_during_contact() {
        let mut behavior = HardcodedLocomotionBehavior::default();
        let left = behavior
            .infer(&LocomotionInput {
                bump_left: 1.0,
                ..LocomotionInput::default()
            })
            .unwrap();
        assert_eq!(left.forward_velocity_m_s, 0.2);
        assert_eq!(left.angular_velocity_rad_s, 0.1);
        assert_eq!(left.recovery_activation, 0.0);
    }

    #[test]
    fn tracker_resets_collision_distance_and_derives_wheels() {
        let mut tracker = LocomotionTracker::default();
        let range = RangeSense::default();
        let mut body = BodySense::default();
        tracker.observe(100, &body, &range);
        body.odometry.x_m = 1.0;
        let moving = tracker.observe(1_100, &body, &range);
        assert!((moving.distance_since_collision_m - 1.0).abs() < 0.001);
        body.flags.bump_left = true;
        let collided = tracker.observe(1_200, &body, &range);
        assert_eq!(collided.distance_since_collision_m, 0.0);
        assert!(collided.left_wheel_travel_m > 0.9);
    }

    #[test]
    fn minimal_genome_activates_and_population_evolves() {
        let mut rng = StdRng::seed_from_u64(7);
        let config = NeatConfig {
            population_size: 12,
            ..NeatConfig::default()
        };
        let mut population = Population::seeded(config, &mut rng);
        let output = population.genomes[0]
            .activate(&[0.0; LOCOMOTION_INPUT_COUNT])
            .unwrap();
        assert_eq!(output.len(), LOCOMOTION_OUTPUT_COUNT);
        let before = population.genomes.len();
        population
            .evolve(
                &(0..before).map(|index| index as f32).collect::<Vec<_>>(),
                &mut rng,
            )
            .unwrap();
        assert_eq!(population.generation, 1);
        assert_eq!(population.genomes.len(), before);
    }

    #[test]
    fn checkpoint_round_trip_validates_feature_order() {
        let mut rng = StdRng::seed_from_u64(9);
        let mut innovations =
            InnovationTracker::new((LOCOMOTION_INPUT_COUNT + LOCOMOTION_OUTPUT_COUNT + 1) as u32);
        let genome = Genome::minimal(
            LOCOMOTION_INPUT_COUNT,
            LOCOMOTION_OUTPUT_COUNT,
            &mut innovations,
            &mut rng,
        );
        let checkpoint = LocomotionCheckpoint::new(3, 12.5, genome);
        let directory = tempfile::tempdir().unwrap();
        checkpoint.save(directory.path()).unwrap();
        assert_eq!(
            LocomotionCheckpoint::load(directory.path()).unwrap(),
            checkpoint
        );
    }

    #[test]
    fn curriculum_scores_reward_escape_and_penalize_vetoes() {
        let weights = FitnessWeights::collision_recovery();
        let good = weights.score(EpisodeMetrics {
            successful_escapes: 2,
            ..EpisodeMetrics::default()
        });
        let unsafe_run = weights.score(EpisodeMetrics {
            safety_vetoes: 2,
            ..EpisodeMetrics::default()
        });
        assert!(good > unsafe_run);
    }

    #[test]
    fn curriculum_has_the_required_order_and_transfer_is_a_gate() {
        assert_eq!(
            CurriculumStage::BackAwayReliably.next(),
            Some(CurriculumStage::ChooseUsefulTurn)
        );
        assert_eq!(
            CurriculumStage::NavigateVariedRooms.next(),
            Some(CurriculumStage::TransferCandidatesToPete)
        );
        assert!(!CurriculumStage::TransferCandidatesToPete.evolves_population());
        assert_eq!(CurriculumStage::TransferCandidatesToPete.next(), None);
    }

    #[test]
    fn transfer_gate_requires_robustness_fallback_and_zero_violations() {
        let criteria = CurriculumStage::TransferCandidatesToPete.promotion_criteria();
        let passing = CandidateEvaluation {
            seeded_episodes: 500,
            success_rate: 0.95,
            collision_rate: 0.02,
            safety_invariant_violations: 0,
            beats_hardcoded: true,
            noise_robust: true,
            motor_mismatch_robust: true,
            fallback_verified: true,
        };
        assert!(criteria.accepts(passing));
        assert!(!criteria.accepts(CandidateEvaluation {
            safety_invariant_violations: 1,
            ..passing
        }));
        assert!(!criteria.accepts(CandidateEvaluation {
            fallback_verified: false,
            ..passing
        }));
    }
}
