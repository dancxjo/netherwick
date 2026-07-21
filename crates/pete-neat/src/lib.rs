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
    #[serde(default)]
    pub recurrent: bool,
    #[serde(default)]
    pub plasticity: PlasticityMode,
    #[serde(default)]
    pub plasticity_rate: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlasticityMode {
    #[default]
    Fixed,
    Hebbian,
    AntiHebbian,
    RewardModulated,
    Habituating,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Genome {
    pub input_count: usize,
    pub output_count: usize,
    pub nodes: Vec<NodeGene>,
    pub connections: Vec<ConnectionGene>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GenomeState {
    values: HashMap<u32, f32>,
    effective_weights: HashMap<u64, f32>,
    last_activity: HashMap<u64, (f32, f32)>,
}

impl GenomeState {
    pub fn reset(&mut self) {
        self.values.clear();
        self.effective_weights.clear();
        self.last_activity.clear();
    }
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
                    innovation: innovations.connection(from, to, false),
                    from,
                    to,
                    weight: rng.gen_range(-1.0..1.0),
                    enabled: true,
                    recurrent: false,
                    plasticity: PlasticityMode::Fixed,
                    plasticity_rate: 0.0,
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
        let mut state = GenomeState::default();
        self.activate_stateful(inputs, &mut state)
    }

    pub fn activate_stateful(&self, inputs: &[f32], state: &mut GenomeState) -> Result<Vec<f32>> {
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
            let mut activity = Vec::new();
            let sum = self
                .connections
                .iter()
                .filter(|edge| edge.enabled && edge.to == node.id)
                .map(|edge| {
                    let source = if edge.recurrent {
                        state.values.get(&edge.from).copied().unwrap_or(0.0)
                    } else {
                        values.get(&edge.from).copied().unwrap_or(0.0)
                    };
                    let weight = state
                        .effective_weights
                        .get(&edge.innovation)
                        .copied()
                        .unwrap_or(edge.weight);
                    activity.push((edge.innovation, source));
                    source * weight
                })
                .sum::<f32>();
            let post = sum.tanh();
            for (innovation, pre) in activity {
                state.last_activity.insert(innovation, (pre, post));
            }
            values.insert(node.id, post);
        }
        let first_output = self.input_count as u32 + 1;
        let output = (0..self.output_count as u32)
            .map(|index| values.get(&(first_output + index)).copied().unwrap_or(0.0))
            .collect::<Vec<_>>();
        state.values = values;
        Ok(output)
    }

    pub fn apply_plasticity(&self, state: &mut GenomeState, reward: f32) {
        let reward = finite_or_zero(reward).clamp(-1.0, 1.0);
        for edge in self.connections.iter().filter(|edge| edge.enabled) {
            if edge.plasticity == PlasticityMode::Fixed || edge.plasticity_rate <= 0.0 {
                continue;
            }
            let Some((pre, post)) = state.last_activity.get(&edge.innovation).copied() else {
                continue;
            };
            let current = state
                .effective_weights
                .get(&edge.innovation)
                .copied()
                .unwrap_or(edge.weight);
            let rate = finite_or_zero(edge.plasticity_rate).clamp(0.0, 1.0);
            let delta = match edge.plasticity {
                PlasticityMode::Fixed => 0.0,
                PlasticityMode::Hebbian => rate * pre * post,
                PlasticityMode::AntiHebbian => -rate * pre * post,
                PlasticityMode::RewardModulated => rate * reward * pre * post,
                PlasticityMode::Habituating => -rate * current.signum() * (pre * post).abs(),
            };
            state
                .effective_weights
                .insert(edge.innovation, (current + delta).clamp(-5.0, 5.0));
        }
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
            if rng.gen::<f32>() < config.plasticity_mutation_rate {
                connection.plasticity = *[
                    PlasticityMode::Fixed,
                    PlasticityMode::Hebbian,
                    PlasticityMode::AntiHebbian,
                    PlasticityMode::RewardModulated,
                    PlasticityMode::Habituating,
                ]
                .choose(rng)
                .unwrap_or(&PlasticityMode::Fixed);
                connection.plasticity_rate = if connection.plasticity == PlasticityMode::Fixed {
                    0.0
                } else {
                    rng.gen_range(0.001..0.05)
                };
            } else if connection.plasticity != PlasticityMode::Fixed
                && rng.gen::<f32>() < config.weight_mutation_rate
            {
                connection.plasticity_rate =
                    (connection.plasticity_rate + rng.gen_range(-0.01..0.01)).clamp(0.001, 0.10);
            }
        }
        if rng.gen::<f32>() < config.add_connection_rate {
            self.mutate_add_connection(innovations, rng);
        }
        if rng.gen::<f32>() < config.add_recurrent_connection_rate {
            self.mutate_add_recurrent_connection(innovations, rng);
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
                innovation: innovations.connection(from, to, false),
                from,
                to,
                weight: rng.gen_range(-1.0..1.0),
                enabled: true,
                recurrent: false,
                plasticity: PlasticityMode::Fixed,
                plasticity_rate: 0.0,
            });
        }
    }

    fn mutate_add_recurrent_connection<R: Rng + ?Sized>(
        &mut self,
        innovations: &mut InnovationTracker,
        rng: &mut R,
    ) {
        let mut candidates = Vec::new();
        for from in &self.nodes {
            for to in &self.nodes {
                if matches!(to.kind, NodeKind::Input | NodeKind::Bias)
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
                innovation: innovations.connection(from, to, true),
                from,
                to,
                weight: rng.gen_range(-1.0..1.0),
                enabled: true,
                recurrent: true,
                plasticity: PlasticityMode::Fixed,
                plasticity_rate: 0.0,
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
            .filter(|(_, edge)| edge.enabled && !edge.recurrent)
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
            let innovation = innovations.connection(from, to, false);
            if let Some(existing) = self
                .connections
                .iter_mut()
                .find(|edge| edge.innovation == innovation)
            {
                existing.enabled = true;
                existing.weight = weight;
                existing.recurrent = false;
                existing.plasticity = PlasticityMode::Fixed;
                existing.plasticity_rate = 0.0;
            } else {
                self.connections.push(ConnectionGene {
                    innovation,
                    from,
                    to,
                    weight,
                    enabled: true,
                    recurrent: false,
                    plasticity: PlasticityMode::Fixed,
                    plasticity_rate: 0.0,
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
#[serde(default)]
pub struct NeatConfig {
    pub population_size: usize,
    pub compatibility_threshold: f32,
    pub excess_coefficient: f32,
    pub disjoint_coefficient: f32,
    pub weight_coefficient: f32,
    pub weight_mutation_rate: f32,
    pub weight_reset_rate: f32,
    pub weight_perturbation: f32,
    pub plasticity_mutation_rate: f32,
    pub add_connection_rate: f32,
    pub add_recurrent_connection_rate: f32,
    pub add_node_rate: f32,
    pub crossover_rate: f32,
    pub elitism: usize,
    pub interspecies_mating_rate: f32,
    pub species_stagnation_generations: u64,
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
            plasticity_mutation_rate: 0.04,
            add_connection_rate: 0.08,
            add_recurrent_connection_rate: 0.03,
            add_node_rate: 0.03,
            crossover_rate: 0.75,
            elitism: 2,
            interspecies_mating_rate: 0.005,
            species_stagnation_generations: 15,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct InnovationTracker {
    next_innovation: u64,
    next_node_id: u32,
    #[serde(with = "connection_innovation_map")]
    connections: ConnectionInnovationMap,
    split_nodes: HashMap<u64, u32>,
}

type ConnectionInnovationKey = (u32, u32, bool);
type ConnectionInnovationMap = HashMap<ConnectionInnovationKey, u64>;

mod connection_innovation_map {
    use super::*;

    pub fn serialize<S>(
        value: &ConnectionInnovationMap,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut entries = value
            .iter()
            .map(|(&(from, to, recurrent), &innovation)| (from, to, recurrent, innovation))
            .collect::<Vec<_>>();
        entries.sort_unstable();
        entries.serialize(serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> std::result::Result<ConnectionInnovationMap, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let entries = Vec::<(u32, u32, bool, u64)>::deserialize(deserializer)?;
        Ok(entries
            .into_iter()
            .map(|(from, to, recurrent, innovation)| ((from, to, recurrent), innovation))
            .collect())
    }
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

    fn connection(&mut self, from: u32, to: u32, recurrent: bool) -> u64 {
        if let Some(innovation) = self.connections.get(&(from, to, recurrent)) {
            return *innovation;
        }
        let innovation = self.next_innovation;
        self.next_innovation += 1;
        self.connections.insert((from, to, recurrent), innovation);
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpeciesRecord {
    pub id: u64,
    pub representative: Genome,
    pub age: u64,
    #[serde(
        default = "negative_infinity",
        deserialize_with = "deserialize_best_fitness"
    )]
    pub best_fitness: f32,
    pub generations_without_improvement: u64,
}

fn negative_infinity() -> f32 {
    f32::NEG_INFINITY
}

fn deserialize_best_fitness<'de, D>(deserializer: D) -> std::result::Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<f32>::deserialize(deserializer)?.unwrap_or(f32::NEG_INFINITY))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpeciesSnapshot {
    pub id: u64,
    pub representative: Genome,
    pub age: u64,
    pub best_fitness: f32,
    pub generations_without_improvement: u64,
    pub member_indices: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Population {
    pub generation: u64,
    pub genomes: Vec<Genome>,
    pub config: NeatConfig,
    innovations: InnovationTracker,
    #[serde(default)]
    species_records: Vec<SpeciesRecord>,
    #[serde(default)]
    genome_species: Vec<u64>,
    #[serde(default)]
    next_species_id: u64,
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
        let mut population = Self {
            generation: 0,
            genomes,
            config,
            innovations,
            species_records: Vec::new(),
            genome_species: Vec::new(),
            next_species_id: 0,
        };
        population.assign_species_members();
        population
    }

    pub fn from_founders<R: Rng + ?Sized>(
        config: NeatConfig,
        founders: &[Genome],
        rng: &mut R,
    ) -> Result<Self> {
        if founders.is_empty() {
            bail!("cannot reconstruct a population without founders");
        }
        let founder_count = founders.len().min(config.population_size);
        let mut canonical_founders = founders[..founder_count].to_vec();
        let mut next_innovation = canonical_founders
            .iter()
            .flat_map(|genome| genome.connections.iter())
            .map(|connection| connection.innovation.saturating_add(1))
            .max()
            .unwrap_or_default();
        let mut key_innovations = HashMap::<(u32, u32, bool), u64>::new();
        let mut innovation_keys = HashMap::<u64, (u32, u32, bool)>::new();
        for genome in &mut canonical_founders {
            for connection in &mut genome.connections {
                let key = (connection.from, connection.to, connection.recurrent);
                let innovation = if let Some(existing) = key_innovations.get(&key) {
                    *existing
                } else if innovation_keys
                    .get(&connection.innovation)
                    .is_none_or(|existing| *existing == key)
                {
                    connection.innovation
                } else {
                    let fresh = next_innovation;
                    next_innovation = next_innovation.saturating_add(1);
                    fresh
                };
                connection.innovation = innovation;
                key_innovations.insert(key, innovation);
                innovation_keys.insert(innovation, key);
            }
        }
        let mut innovations = InnovationTracker::default();
        for genome in &canonical_founders {
            innovations.next_node_id = innovations.next_node_id.max(
                genome
                    .nodes
                    .iter()
                    .map(|node| node.id.saturating_add(1))
                    .max()
                    .unwrap_or_default(),
            );
            for connection in &genome.connections {
                innovations.next_innovation = innovations
                    .next_innovation
                    .max(connection.innovation.saturating_add(1));
                innovations
                    .connections
                    .entry((connection.from, connection.to, connection.recurrent))
                    .or_insert(connection.innovation);
            }
        }

        let mut genomes = canonical_founders;
        let mut descendant = 0usize;
        while genomes.len() < config.population_size {
            let mut genome = genomes[descendant % founder_count].clone();
            genome.mutate(config, &mut innovations, rng);
            genomes.push(genome);
            descendant += 1;
        }
        let mut population = Self {
            generation: 0,
            genomes,
            config,
            innovations,
            species_records: Vec::new(),
            genome_species: Vec::new(),
            next_species_id: 0,
        };
        population.assign_species_members();
        Ok(population)
    }

    pub fn species(&self) -> Vec<SpeciesSnapshot> {
        if self.genome_species.len() != self.genomes.len() || self.species_records.is_empty() {
            return self.transient_species();
        }
        let groups = self.species_member_groups();
        self.species_records
            .iter()
            .filter_map(|record| {
                groups.get(&record.id).map(|members| SpeciesSnapshot {
                    id: record.id,
                    representative: record.representative.clone(),
                    age: record.age,
                    best_fitness: record.best_fitness,
                    generations_without_improvement: record.generations_without_improvement,
                    member_indices: members.clone(),
                })
            })
            .collect()
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
        self.ensure_species_members();
        let all_indices = (0..self.genomes.len()).collect::<Vec<_>>();
        let global_champion = best_index(&all_indices, fitness);
        let groups = self.species_member_groups();
        let mut species_generation = Vec::new();
        for (species_id, members) in groups {
            if members.is_empty() {
                continue;
            }
            let champion_index = best_index(&members, fitness);
            let raw_best_fitness = finite_or_zero(fitness[champion_index]);
            let adjusted_fitness = members
                .iter()
                .map(|index| finite_or_zero(fitness[*index]) / members.len() as f32)
                .sum::<f32>();
            let record = self
                .species_records
                .iter_mut()
                .find(|record| record.id == species_id)
                .expect("species assignment must have a record");
            record.age = record.age.saturating_add(1);
            record.representative = self.genomes[champion_index].clone();
            if raw_best_fitness > record.best_fitness + 1.0e-6 {
                record.best_fitness = raw_best_fitness;
                record.generations_without_improvement = 0;
            } else {
                record.generations_without_improvement =
                    record.generations_without_improvement.saturating_add(1);
            }
            let contains_global_champion = members.contains(&global_champion);
            let stagnant = record.generations_without_improvement
                >= self.config.species_stagnation_generations
                && !contains_global_champion;
            if !stagnant {
                species_generation.push(SpeciesGeneration {
                    id: species_id,
                    members,
                    adjusted_fitness,
                    champion_index,
                    contains_global_champion,
                });
            }
        }
        if species_generation.is_empty() {
            let species_id = self.genome_species[global_champion];
            species_generation.push(SpeciesGeneration {
                id: species_id,
                members: vec![global_champion],
                adjusted_fitness: finite_or_zero(fitness[global_champion]).max(0.0),
                champion_index: global_champion,
                contains_global_champion: true,
            });
        }
        species_generation.sort_by(|left, right| {
            right
                .contains_global_champion
                .cmp(&left.contains_global_champion)
                .then_with(|| right.adjusted_fitness.total_cmp(&left.adjusted_fitness))
                .then_with(|| left.id.cmp(&right.id))
        });
        species_generation.truncate(self.config.population_size);
        let species_scores = species_generation
            .iter()
            .map(|species| species.adjusted_fitness)
            .collect::<Vec<_>>();
        let offspring_counts = allocate_offspring(&species_scores, self.config.population_size);
        let all_surviving_members = species_generation
            .iter()
            .flat_map(|species| species.members.iter().copied())
            .collect::<Vec<_>>();
        let mut next = Vec::with_capacity(self.config.population_size);
        let mut next_species = Vec::with_capacity(self.config.population_size);
        let mut surviving_ids = BTreeSet::new();
        for (species, offspring_count) in species_generation.iter().zip(offspring_counts) {
            if offspring_count == 0 {
                continue;
            }
            surviving_ids.insert(species.id);
            next.push(self.genomes[species.champion_index].clone());
            next_species.push(species.id);
            for _ in 1..offspring_count {
                let first = select_parent_from_indices(&species.members, fitness, rng);
                let mate_pool =
                    if rng.gen::<f32>() < self.config.interspecies_mating_rate.clamp(0.0, 1.0) {
                        all_surviving_members.as_slice()
                    } else {
                        species.members.as_slice()
                    };
                let mut child = if rng.gen::<f32>() < self.config.crossover_rate {
                    let second = select_parent_from_indices(mate_pool, fitness, rng);
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
                next_species.push(species.id);
            }
        }
        while next.len() < self.config.population_size {
            let first = select_parent_from_indices(&[global_champion], fitness, rng);
            let mut child = self.genomes[first].clone();
            child.mutate(self.config, &mut self.innovations, rng);
            next.push(child);
            next_species.push(self.genome_species[global_champion]);
        }
        self.species_records
            .retain(|record| surviving_ids.contains(&record.id));
        self.genomes = next;
        self.genome_species = next_species;
        self.generation += 1;
        self.assign_species_members();
        Ok(())
    }

    /// Evolves normally, then carries the supplied protected repertoire into the
    /// next population without mutation. Protected genomes replace descendants
    /// at the tail, never the newly selected global champion at index zero.
    pub fn evolve_with_elites<R: Rng + ?Sized>(
        &mut self,
        fitness: &[f32],
        protected: &[Genome],
        rng: &mut R,
    ) -> Result<()> {
        let protected = protected
            .iter()
            .filter(|genome| self.genomes.iter().any(|candidate| candidate == *genome))
            .cloned()
            .collect::<Vec<_>>();
        self.evolve(fitness, rng)?;
        let mut replacement = self.genomes.len();
        for elite in protected {
            if self.genomes.iter().any(|genome| genome == &elite) {
                continue;
            }
            if replacement <= 1 {
                break;
            }
            replacement -= 1;
            self.genomes[replacement] = elite;
        }
        self.assign_species_members();
        Ok(())
    }

    /// Replaces low-priority descendants with mutated archive founders. This is
    /// intended for bounded diversity recovery, not wholesale reseeding.
    pub fn inject_archive_descendants<R: Rng + ?Sized>(
        &mut self,
        founders: &[Genome],
        count: usize,
        rng: &mut R,
    ) -> usize {
        if founders.is_empty() || self.genomes.len() <= 1 {
            return 0;
        }
        let count = count.min(self.genomes.len() - 1);
        for offset in 0..count {
            let mut descendant = founders[offset % founders.len()].clone();
            // Two mutations give recovery founders room to clear a collapsed
            // compatibility basin while retaining their archived behavior.
            descendant.mutate(self.config, &mut self.innovations, rng);
            descendant.mutate(self.config, &mut self.innovations, rng);
            let index = 1 + offset;
            self.genomes[index] = descendant;
        }
        self.assign_species_members();
        count
    }

    pub fn compatibility_distance_distribution(&self) -> (f32, f32, f32) {
        let mut distances = Vec::new();
        for left in 0..self.genomes.len() {
            for right in left + 1..self.genomes.len() {
                distances.push(
                    self.genomes[left].compatibility_distance(&self.genomes[right], self.config),
                );
            }
        }
        if distances.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        distances.sort_by(|left, right| left.total_cmp(right));
        let mean = distances.iter().sum::<f32>() / distances.len() as f32;
        (distances[0], mean, distances[distances.len() - 1])
    }

    pub fn evolve_ranked<R: Rng + ?Sized>(
        &mut self,
        traits: &[FitnessTraits],
        constraints: SelectionConstraints,
        rng: &mut R,
    ) -> Result<Vec<f32>> {
        let fitness = rank_fitness(traits, constraints);
        self.evolve(&fitness, rng)?;
        Ok(fitness)
    }

    fn ensure_species_members(&mut self) {
        if self.species_records.is_empty() || self.genome_species.len() != self.genomes.len() {
            self.assign_species_members();
        }
    }

    fn assign_species_members(&mut self) {
        let mut assignments = Vec::with_capacity(self.genomes.len());
        let mut member_counts = BTreeMap::<u64, usize>::new();
        for genome in &self.genomes {
            let species_id = self
                .species_records
                .iter()
                .find(|record| {
                    genome.compatibility_distance(&record.representative, self.config)
                        < self.config.compatibility_threshold
                })
                .map(|record| record.id)
                .unwrap_or_else(|| {
                    let id = self.next_species_id;
                    self.next_species_id = self.next_species_id.saturating_add(1);
                    self.species_records.push(SpeciesRecord {
                        id,
                        representative: genome.clone(),
                        age: 0,
                        best_fitness: f32::NEG_INFINITY,
                        generations_without_improvement: 0,
                    });
                    id
                });
            assignments.push(species_id);
            *member_counts.entry(species_id).or_default() += 1;
        }
        self.genome_species = assignments;
        self.species_records
            .retain(|record| member_counts.contains_key(&record.id));
    }

    fn species_member_groups(&self) -> BTreeMap<u64, Vec<usize>> {
        let mut groups = BTreeMap::<u64, Vec<usize>>::new();
        for (index, species_id) in self.genome_species.iter().copied().enumerate() {
            groups.entry(species_id).or_default().push(index);
        }
        groups
    }

    fn transient_species(&self) -> Vec<SpeciesSnapshot> {
        let mut species: Vec<(Genome, Vec<usize>)> = Vec::new();
        for (index, genome) in self.genomes.iter().enumerate() {
            if let Some((_, members)) = species.iter_mut().find(|(representative, _)| {
                genome.compatibility_distance(representative, self.config)
                    < self.config.compatibility_threshold
            }) {
                members.push(index);
            } else {
                species.push((genome.clone(), vec![index]));
            }
        }
        species
            .into_iter()
            .enumerate()
            .map(
                |(index, (representative, member_indices))| SpeciesSnapshot {
                    id: index as u64,
                    representative,
                    age: 0,
                    best_fitness: f32::NEG_INFINITY,
                    generations_without_improvement: 0,
                    member_indices,
                },
            )
            .collect()
    }
}

#[derive(Clone, Debug)]
struct SpeciesGeneration {
    id: u64,
    members: Vec<usize>,
    adjusted_fitness: f32,
    champion_index: usize,
    contains_global_champion: bool,
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
    state: GenomeState,
    last_input: Option<LocomotionInput>,
}

impl NeatLocomotionBehavior {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            checkpoint: LocomotionCheckpoint::load(path)?,
            max_forward_m_s: 0.6,
            max_turn_rad_s: 1.0,
            state: GenomeState::default(),
            last_input: None,
        })
    }

    pub fn infer_with_reward(
        &mut self,
        input: &LocomotionInput,
        reward: f32,
    ) -> Result<LocomotionOutput> {
        self.checkpoint
            .genome
            .apply_plasticity(&mut self.state, reward);
        let values = self
            .checkpoint
            .genome
            .activate_stateful(&input.features(), &mut self.state)?;
        self.last_input = Some(input.clone());
        Ok(LocomotionOutput {
            forward_velocity_m_s: values[0] * self.max_forward_m_s,
            angular_velocity_rad_s: values[1] * self.max_turn_rad_s,
            recovery_activation: (values[2] + 1.0) * 0.5,
        }
        .bounded(self.max_forward_m_s, self.max_turn_rad_s)
        .with_recovery_intent(0.5))
    }
}

impl FunctionBehavior<LocomotionInput, LocomotionOutput> for NeatLocomotionBehavior {
    fn id(&self) -> &'static str {
        "locomotion.neat.v0"
    }

    fn infer(&mut self, input: &LocomotionInput) -> Result<LocomotionOutput> {
        let reward = self
            .last_input
            .as_ref()
            .map(|previous| runtime_locomotion_reward(previous, input))
            .unwrap_or(0.0);
        self.infer_with_reward(input, reward)
    }
}

fn runtime_locomotion_reward(previous: &LocomotionInput, current: &LocomotionInput) -> f32 {
    let left_delta = current.left_wheel_travel_m - previous.left_wheel_travel_m;
    let right_delta = current.right_wheel_travel_m - previous.right_wheel_travel_m;
    let travel = ((left_delta + right_delta) * 0.5).abs();
    let mut reward = (travel * 5.0).clamp(0.0, 0.5);
    if current.collision_active() && !previous.collision_active() {
        reward -= 0.5;
    }
    let collision_rate_delta =
        (current.recent_collision_rate - previous.recent_collision_rate).max(0.0);
    reward -= collision_rate_delta.clamp(0.0, 0.5);
    reward.clamp(-1.0, 1.0)
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
    #[serde(default)]
    pub short_cycle_count: u32,
    #[serde(default)]
    pub short_cycle_steps: u32,
    #[serde(default)]
    pub recent_repetition_steps: u32,
    #[serde(default)]
    pub maximum_displacement_m: f32,
    #[serde(default)]
    pub radius_bands_reached: u32,
    #[serde(default)]
    pub arena_sectors_visited: u32,
    pub wheel_motion_m: f32,
    pub angular_motion_rad: f32,
    pub recovery_activation_sum: f32,
    pub stalled_steps: u32,
    pub safety_vetoes: u32,
    #[serde(default)]
    pub safety_invariant_violations: u32,
    pub resource_energy_used: f32,
    pub sensor_energy_cost: f32,
    pub computation_energy_cost: f32,
    pub collision_energy_cost: f32,
    pub minimum_resource_battery: f32,
    pub final_resource_battery: f32,
    pub minimum_resource_health: f32,
    pub final_resource_health: f32,
    pub battery_depleted: u32,
    pub health_depleted: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitnessTraits {
    pub exploration: f32,
    pub escape_rate: f32,
    pub collision_rate: f32,
    pub energy_use: f32,
    pub forward_progress: f32,
    pub repetition_rate: f32,
    pub worst_environment_score: f32,
    #[serde(default)]
    pub safety_veto_rate: f32,
    #[serde(default)]
    pub safety_invariant_violations: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NicheLabel {
    OpenRoomExplorer,
    NarrowCorridorNavigator,
    ConcaveTrapEscapeSpecialist,
    ClutterSpecialist,
    LowBatteryConservativeMover,
    DegradedSensorNavigator,
    AsymmetricMotorCompensator,
    Generalist,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct QualityDiversityDescriptor {
    pub collision_frequency_bin: u8,
    pub turning_intensity_bin: u8,
    pub area_coverage_bin: u8,
    pub energy_consumption_bin: u8,
    pub recovery_aggressiveness_bin: u8,
}

impl QualityDiversityDescriptor {
    pub fn from_traits_and_metrics(
        traits: FitnessTraits,
        metrics: EpisodeMetrics,
        episodes: usize,
        steps: usize,
    ) -> Self {
        let step_count = episodes.max(1).saturating_mul(steps.max(1)) as f32;
        Self {
            collision_frequency_bin: bin(traits.collision_rate, &[0.002, 0.01, 0.04]),
            turning_intensity_bin: bin(
                metrics.angular_motion_rad / step_count,
                &[0.01, 0.04, 0.12],
            ),
            area_coverage_bin: bin(traits.exploration, &[6.0, 14.0, 28.0]),
            energy_consumption_bin: bin(traits.energy_use, &[2.0, 6.0, 14.0]),
            recovery_aggressiveness_bin: bin(
                metrics.recovery_activation_sum / step_count,
                &[0.05, 0.20, 0.50],
            ),
        }
    }

    pub fn niche_label(self, traits: FitnessTraits) -> NicheLabel {
        self.evidence_based_niche_label(traits, NicheQualificationEvidence::default())
    }

    pub fn evidence_based_niche_label(
        self,
        traits: FitnessTraits,
        evidence: NicheQualificationEvidence,
    ) -> NicheLabel {
        if let Some(niche) = evidence.label_for_descriptor(self, traits) {
            return niche;
        }
        if self.area_coverage_bin >= 3
            && self.collision_frequency_bin <= 1
            && self.energy_consumption_bin <= 2
        {
            NicheLabel::OpenRoomExplorer
        } else if self.collision_frequency_bin >= 2
            && self.recovery_aggressiveness_bin >= 1
            && traits.escape_rate >= 0.6
        {
            NicheLabel::ClutterSpecialist
        } else {
            NicheLabel::Generalist
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NicheQualificationEvidence {
    pub degraded_sensor_retention: f32,
    pub motor_mismatch_retention: f32,
    pub heldout_trap_success_rate: f32,
    pub low_battery_progress_m: f32,
    pub corridor_success_rate: f32,
}

impl NicheQualificationEvidence {
    pub fn from_selection_retention(degraded_sensor: f32, motor_mismatch: f32) -> Self {
        Self {
            degraded_sensor_retention: finite_or_zero(degraded_sensor).clamp(0.0, 1.0),
            motor_mismatch_retention: finite_or_zero(motor_mismatch).clamp(0.0, 1.0),
            ..Self::default()
        }
    }

    fn label_for_descriptor(
        self,
        descriptor: QualityDiversityDescriptor,
        traits: FitnessTraits,
    ) -> Option<NicheLabel> {
        if self.heldout_trap_success_rate >= 0.80 && descriptor.recovery_aggressiveness_bin >= 2 {
            Some(NicheLabel::ConcaveTrapEscapeSpecialist)
        } else if self.low_battery_progress_m > 0.5
            && descriptor.energy_consumption_bin <= 1
            && descriptor.collision_frequency_bin <= 1
        {
            Some(NicheLabel::LowBatteryConservativeMover)
        } else if self.corridor_success_rate >= 0.80
            && descriptor.turning_intensity_bin >= 2
            && descriptor.collision_frequency_bin <= 1
        {
            Some(NicheLabel::NarrowCorridorNavigator)
        } else if self.degraded_sensor_retention >= 0.75
            && descriptor.collision_frequency_bin <= 1
            && traits.worst_environment_score > 0.0
        {
            Some(NicheLabel::DegradedSensorNavigator)
        } else if self.motor_mismatch_retention >= 0.75 && descriptor.collision_frequency_bin <= 1 {
            Some(NicheLabel::AsymmetricMotorCompensator)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct QualityDiversityEntry {
    pub genome_index: usize,
    pub selection_fitness: f32,
    pub traits: FitnessTraits,
    pub descriptor: QualityDiversityDescriptor,
    pub niche: NicheLabel,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BehavioralDescriptor {
    pub coverage: f32,
    pub collision_rate: f32,
    pub mean_curvature: f32,
    pub escape_style: f32,
    pub energy: f32,
}

impl BehavioralDescriptor {
    pub fn from_traits_and_metrics(
        traits: FitnessTraits,
        metrics: EpisodeMetrics,
        episodes: usize,
        steps: usize,
    ) -> Self {
        let episode_count = episodes.max(1) as f32;
        let step_count = episodes.max(1).saturating_mul(steps.max(1)) as f32;
        let forward = traits.forward_progress.abs().max(0.05);
        Self {
            coverage: traits.exploration / 40.0,
            collision_rate: traits.collision_rate * 100.0,
            mean_curvature: (metrics.angular_motion_rad / episode_count) / forward,
            escape_style: (metrics.recovery_activation_sum / step_count)
                + metrics.successful_escapes as f32 / episode_count
                + metrics.escape_boundary_crossings as f32 / episode_count,
            energy: traits.energy_use / 12.0,
        }
        .normalized()
    }

    pub fn distance(self, other: Self) -> f32 {
        let dc = self.coverage - other.coverage;
        let dcol = self.collision_rate - other.collision_rate;
        let dcurv = self.mean_curvature - other.mean_curvature;
        let desc = self.escape_style - other.escape_style;
        let de = self.energy - other.energy;
        (dc.mul_add(
            dc,
            dcol.mul_add(dcol, dcurv.mul_add(dcurv, desc.mul_add(desc, de * de))),
        ))
        .sqrt()
    }

    fn normalized(self) -> Self {
        Self {
            coverage: finite_or_zero(self.coverage).clamp(0.0, 1.0),
            collision_rate: finite_or_zero(self.collision_rate).clamp(0.0, 1.0),
            mean_curvature: (finite_or_zero(self.mean_curvature) / 8.0).clamp(0.0, 1.0),
            escape_style: finite_or_zero(self.escape_style).clamp(0.0, 2.0) * 0.5,
            energy: finite_or_zero(self.energy).clamp(0.0, 1.0),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NoveltyArchive {
    descriptors: Vec<BehavioralDescriptor>,
    max_descriptors: usize,
}

impl NoveltyArchive {
    pub fn new(max_descriptors: usize) -> Self {
        Self {
            descriptors: Vec::new(),
            max_descriptors,
        }
    }

    pub fn descriptors(&self) -> &[BehavioralDescriptor] {
        &self.descriptors
    }

    pub fn observe(&mut self, descriptors: &[BehavioralDescriptor]) {
        if self.max_descriptors == 0 {
            return;
        }
        self.descriptors.extend(descriptors.iter().copied());
        if self.descriptors.len() > self.max_descriptors {
            let excess = self.descriptors.len() - self.max_descriptors;
            self.descriptors.drain(0..excess);
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NoveltySummary {
    pub novelty: f32,
    pub selection_fitness: f32,
}

pub fn behavioral_descriptors(
    traits: &[FitnessTraits],
    metrics: &[EpisodeMetrics],
    episodes: usize,
    steps: usize,
) -> Vec<BehavioralDescriptor> {
    traits
        .iter()
        .copied()
        .zip(metrics.iter().copied())
        .map(|(traits, metrics)| {
            BehavioralDescriptor::from_traits_and_metrics(traits, metrics, episodes, steps)
        })
        .collect()
}

pub fn novelty_scores(
    descriptors: &[BehavioralDescriptor],
    archive: &[BehavioralDescriptor],
    nearest_neighbors: usize,
) -> Vec<f32> {
    descriptors
        .iter()
        .enumerate()
        .map(|(index, descriptor)| {
            let mut distances = archive
                .iter()
                .chain(
                    descriptors
                        .iter()
                        .enumerate()
                        .filter_map(|(other_index, other)| (other_index != index).then_some(other)),
                )
                .map(|other| descriptor.distance(*other))
                .filter(|distance| distance.is_finite() && *distance > 0.0)
                .collect::<Vec<_>>();
            if distances.is_empty() {
                return 0.0;
            }
            distances.sort_by(|left, right| left.total_cmp(right));
            let neighbor_count = nearest_neighbors.max(1).min(distances.len());
            distances.iter().take(neighbor_count).sum::<f32>() / neighbor_count as f32
        })
        .collect()
}

pub fn apply_novelty_pressure(
    selection_fitness: &[f32],
    novelty: &[f32],
    novelty_weight: f32,
) -> Vec<NoveltySummary> {
    selection_fitness
        .iter()
        .copied()
        .zip(novelty.iter().copied())
        .map(|(selection_fitness, novelty)| NoveltySummary {
            novelty,
            selection_fitness: selection_fitness + novelty_weight.max(0.0) * novelty,
        })
        .collect()
}

pub fn quality_diversity_archive(
    traits: &[FitnessTraits],
    metrics: &[EpisodeMetrics],
    selection_fitness: &[f32],
    episodes: usize,
    steps: usize,
) -> Vec<QualityDiversityEntry> {
    quality_diversity_archive_with_evidence(
        traits,
        metrics,
        selection_fitness,
        &[],
        episodes,
        steps,
    )
}

pub fn quality_diversity_archive_with_evidence(
    traits: &[FitnessTraits],
    metrics: &[EpisodeMetrics],
    selection_fitness: &[f32],
    evidence: &[NicheQualificationEvidence],
    episodes: usize,
    steps: usize,
) -> Vec<QualityDiversityEntry> {
    let mut archive = BTreeMap::<QualityDiversityDescriptor, QualityDiversityEntry>::new();
    for index in 0..traits.len().min(metrics.len()).min(selection_fitness.len()) {
        let descriptor = QualityDiversityDescriptor::from_traits_and_metrics(
            traits[index],
            metrics[index],
            episodes,
            steps,
        );
        let niche = evidence
            .get(index)
            .and_then(|evidence| evidence.label_for_descriptor(descriptor, traits[index]))
            .unwrap_or_else(|| descriptor.niche_label(traits[index]));
        let entry = QualityDiversityEntry {
            genome_index: index,
            selection_fitness: selection_fitness[index],
            traits: traits[index],
            descriptor,
            niche,
        };
        archive
            .entry(descriptor)
            .and_modify(|current| {
                if entry.selection_fitness > current.selection_fitness {
                    *current = entry;
                }
            })
            .or_insert(entry);
    }
    archive.into_values().collect()
}

impl FitnessTraits {
    pub fn from_metrics(
        metrics: EpisodeMetrics,
        episodes: usize,
        steps: usize,
        successful_episodes: usize,
        worst_environment_score: f32,
    ) -> Self {
        let episode_count = episodes.max(1) as f32;
        let step_count = episodes.max(1).saturating_mul(steps.max(1)) as f32;
        let exploration = metrics.new_area_cells as f32 / episode_count;
        let forward_progress = metrics.distance_without_collision_m / episode_count;
        let resource_energy = if metrics.resource_energy_used > 0.0 {
            metrics.resource_energy_used
        } else {
            metrics.wheel_motion_m
        };
        let energy_use = resource_energy / episode_count;
        Self {
            exploration,
            escape_rate: successful_episodes as f32 / episode_count,
            collision_rate: metrics.collisions as f32 / step_count,
            energy_use,
            forward_progress,
            repetition_rate: metrics.repeated_state_steps as f32 / step_count,
            worst_environment_score: finite_or_zero(worst_environment_score),
            safety_veto_rate: metrics.safety_vetoes as f32 / step_count,
            safety_invariant_violations: metrics.safety_invariant_violations,
        }
    }

    fn objective(self, objective: SelectionObjective) -> f32 {
        match objective {
            SelectionObjective::EscapeRate => self.escape_rate,
            SelectionObjective::Exploration => self.exploration,
            SelectionObjective::Efficiency => {
                (self.exploration + self.forward_progress) / (1.0 + self.energy_use)
            }
            SelectionObjective::ForwardProgress => self.forward_progress,
            SelectionObjective::WorstEnvironment => self.worst_environment_score,
            SelectionObjective::CollisionAvoidance => -self.collision_rate,
            SelectionObjective::LowEnergy => -self.energy_use,
            SelectionObjective::LowRepetition => -self.repetition_rate,
            SelectionObjective::LowSafetyVeto => -self.safety_veto_rate,
        }
    }
}

impl Default for FitnessTraits {
    fn default() -> Self {
        Self {
            exploration: 0.0,
            escape_rate: 0.0,
            collision_rate: 1.0,
            energy_use: f32::INFINITY,
            forward_progress: 0.0,
            repetition_rate: 1.0,
            worst_environment_score: f32::NEG_INFINITY,
            safety_veto_rate: 1.0,
            safety_invariant_violations: u32::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelectionConstraints {
    pub maximum_safety_invariant_violations: u32,
    pub maximum_collision_rate: f32,
    pub minimum_escape_rate: f32,
    #[serde(default = "default_maximum_safety_veto_rate")]
    pub maximum_safety_veto_rate: f32,
}

fn default_maximum_safety_veto_rate() -> f32 {
    1.0
}

impl SelectionConstraints {
    pub const fn new(
        maximum_safety_invariant_violations: u32,
        maximum_collision_rate: f32,
        minimum_escape_rate: f32,
    ) -> Self {
        Self {
            maximum_safety_invariant_violations,
            maximum_collision_rate,
            minimum_escape_rate,
            maximum_safety_veto_rate: 1.0,
        }
    }

    pub const fn with_maximum_safety_veto_rate(mut self, maximum_rate: f32) -> Self {
        self.maximum_safety_veto_rate = maximum_rate;
        self
    }
}

impl Default for SelectionConstraints {
    fn default() -> Self {
        Self::new(0, 0.10, 0.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SelectionObjective {
    EscapeRate,
    Exploration,
    Efficiency,
    ForwardProgress,
    WorstEnvironment,
    CollisionAvoidance,
    LowEnergy,
    LowRepetition,
    LowSafetyVeto,
}

const SELECTION_OBJECTIVES: [SelectionObjective; 9] = [
    SelectionObjective::EscapeRate,
    SelectionObjective::Exploration,
    SelectionObjective::Efficiency,
    SelectionObjective::ForwardProgress,
    SelectionObjective::WorstEnvironment,
    SelectionObjective::CollisionAvoidance,
    SelectionObjective::LowEnergy,
    SelectionObjective::LowRepetition,
    SelectionObjective::LowSafetyVeto,
];

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelectionSummary {
    pub fitness: f32,
    pub constraint_violations: u32,
    #[serde(default)]
    pub stage_success_rate: f32,
    #[serde(default)]
    pub prerequisite_floor: f32,
    #[serde(default)]
    pub stage_score: f32,
    pub pareto_front: u32,
    #[serde(default, deserialize_with = "deserialize_optional_f32")]
    pub crowding_distance: f32,
}

fn deserialize_optional_f32<'de, D>(deserializer: D) -> std::result::Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<f32>::deserialize(deserializer)?.unwrap_or_default())
}

pub fn rank_fitness(traits: &[FitnessTraits], constraints: SelectionConstraints) -> Vec<f32> {
    selection_summaries(traits, constraints)
        .into_iter()
        .map(|summary| summary.fitness)
        .collect()
}

pub fn selection_summaries(
    traits: &[FitnessTraits],
    constraints: SelectionConstraints,
) -> Vec<SelectionSummary> {
    if traits.is_empty() {
        return Vec::new();
    }

    let violations = traits
        .iter()
        .map(|traits| constraint_violation_score(*traits, constraints))
        .collect::<Vec<_>>();
    let feasible = violations
        .iter()
        .enumerate()
        .filter_map(|(index, violation)| (*violation == 0.0).then_some(index))
        .collect::<Vec<_>>();
    let fronts = pareto_fronts(traits, &feasible);
    let mut summaries = vec![
        SelectionSummary {
            fitness: 0.0,
            constraint_violations: 0,
            stage_success_rate: 0.0,
            prerequisite_floor: 0.0,
            stage_score: 0.0,
            pareto_front: u32::MAX,
            crowding_distance: 0.0,
        };
        traits.len()
    ];

    for (index, violation) in violations.iter().copied().enumerate() {
        if violation > 0.0 {
            summaries[index] = SelectionSummary {
                fitness: -violation + infeasible_tiebreak(traits[index]),
                constraint_violations: violation.ceil() as u32,
                stage_success_rate: 0.0,
                prerequisite_floor: 0.0,
                stage_score: 0.0,
                pareto_front: u32::MAX,
                crowding_distance: 0.0,
            };
        }
    }

    let front_count = fronts.len().max(1) as f32;
    for (front_index, front) in fronts.iter().enumerate() {
        let crowding = crowding_distances(traits, front);
        for (member_offset, genome_index) in front.iter().copied().enumerate() {
            let crowding_distance = crowding[member_offset];
            let crowding_bonus = normalized_crowding_bonus(crowding_distance);
            summaries[genome_index] = SelectionSummary {
                fitness: 1_000.0
                    + (front_count - front_index as f32) * 100.0
                    + crowding_bonus * 50.0,
                constraint_violations: 0,
                stage_success_rate: 0.0,
                prerequisite_floor: 0.0,
                stage_score: 0.0,
                pareto_front: front_index as u32,
                crowding_distance,
            };
        }
    }

    summaries
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
    pub resource_energy: f32,
    pub battery_depletion: f32,
    pub health_depletion: f32,
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
            resource_energy: 0.6,
            battery_depletion: 25.0,
            health_depletion: 35.0,
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
            resource_energy: 1.0,
            battery_depletion: 35.0,
            health_depletion: 45.0,
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
            - self.resource_energy * metrics.resource_energy_used
            - self.battery_depletion * metrics.battery_depleted as f32
            - self.health_depletion * metrics.health_depleted as f32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CurriculumStage {
    BackAwayReliably,
    ChooseUsefulTurn,
    EscapeCorners,
    LeaveStartRegion,
    ExpandLocalCoverage,
    BreakShortCycles,
    ExploreWithoutLooping,
    NavigateVariedRooms,
    TransferCandidatesToPete,
}

impl CurriculumStage {
    pub const ORDER: [Self; 9] = [
        Self::BackAwayReliably,
        Self::ChooseUsefulTurn,
        Self::EscapeCorners,
        Self::LeaveStartRegion,
        Self::ExpandLocalCoverage,
        Self::BreakShortCycles,
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
            Self::LeaveStartRegion => FitnessWeights {
                new_area: 3.0,
                collision_free_distance: 2.0,
                collision: 7.0,
                repeated_state: 0.5,
                angular_motion: 0.15,
                stalled: 0.5,
                ..FitnessWeights::efficient_wandering()
            },
            Self::ExpandLocalCoverage => FitnessWeights {
                new_area: 4.0,
                collision_free_distance: 2.0,
                collision: 7.0,
                repeated_state: 0.7,
                angular_motion: 0.12,
                stalled: 0.5,
                ..FitnessWeights::efficient_wandering()
            },
            Self::BreakShortCycles => FitnessWeights {
                new_area: 4.0,
                collision_free_distance: 2.0,
                collision: 7.0,
                repeated_state: 1.5,
                angular_motion: 0.2,
                stalled: 0.5,
                ..FitnessWeights::efficient_wandering()
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

    pub fn selection_constraints(self) -> SelectionConstraints {
        let criteria = self.promotion_criteria();
        SelectionConstraints::new(
            criteria.maximum_safety_invariant_violations,
            criteria.maximum_collision_rate,
            0.0,
        )
        .with_maximum_safety_veto_rate(self.maximum_safety_veto_rate())
    }

    pub const fn maximum_safety_veto_rate(self) -> f32 {
        // A few isolated interventions remain a cost, while policies that ask
        // the downstream safety layer to suppress more than one command in
        // twenty are not considered reproductively feasible or promotable.
        0.05
    }

    pub fn promotion_criteria(self) -> PromotionCriteria {
        match self {
            Self::BackAwayReliably => PromotionCriteria::new(40, 0.90, 0.20, false),
            Self::ChooseUsefulTurn => PromotionCriteria::new(60, 0.85, 0.18, false),
            Self::EscapeCorners => PromotionCriteria::new(100, 0.80, 0.15, false),
            Self::LeaveStartRegion => PromotionCriteria::new(120, 0.80, 0.10, false),
            Self::ExpandLocalCoverage => PromotionCriteria::new(120, 0.80, 0.10, false),
            Self::BreakShortCycles => PromotionCriteria::new(150, 0.80, 0.10, false),
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
    #[serde(default = "default_promotion_maximum_safety_veto_rate")]
    pub maximum_safety_veto_rate: f32,
    pub must_beat_hardcoded: bool,
    pub maximum_safety_invariant_violations: u32,
}

fn default_promotion_maximum_safety_veto_rate() -> f32 {
    0.05
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
            maximum_safety_veto_rate: 0.05,
            must_beat_hardcoded,
            maximum_safety_invariant_violations: 0,
        }
    }

    pub fn accepts(self, evaluation: CandidateEvaluation) -> bool {
        evaluation.seeded_episodes >= self.minimum_seeded_episodes
            && evaluation.success_rate >= self.minimum_success_rate
            && evaluation.collision_rate <= self.maximum_collision_rate
            && evaluation.safety_veto_rate <= self.maximum_safety_veto_rate
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
    #[serde(default)]
    pub safety_veto_rate: f32,
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

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
