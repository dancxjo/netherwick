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
