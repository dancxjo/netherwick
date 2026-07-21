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
