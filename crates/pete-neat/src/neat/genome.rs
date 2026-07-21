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
