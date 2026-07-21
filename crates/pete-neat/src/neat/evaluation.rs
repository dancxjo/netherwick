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
