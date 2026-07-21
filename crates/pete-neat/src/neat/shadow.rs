pub const LOCOMOTION_SHADOW_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocomotionShadowFrame {
    pub schema_version: u32,
    pub frame_id: String,
    pub t_ms: u64,
    pub input_id: String,
    pub input: LocomotionInput,
    pub baseline_output: LocomotionOutput,
    pub candidate_output: LocomotionOutput,
    pub executed_output: LocomotionOutput,
    pub baseline_provenance: String,
    pub candidate_provenance: String,
    pub baseline_inference_us: u64,
    pub candidate_inference_us: u64,
    pub candidate_confidence: Option<f32>,
    pub candidate_error: Option<String>,
    pub disagreement: f32,
    pub baseline_executed_only: bool,
}

impl LocomotionShadowFrame {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frame_id: impl Into<String>,
        t_ms: u64,
        input: LocomotionInput,
        baseline_output: LocomotionOutput,
        candidate_output: LocomotionOutput,
        executed_output: LocomotionOutput,
        baseline_provenance: impl Into<String>,
        candidate_provenance: impl Into<String>,
        baseline_inference_us: u64,
        candidate_inference_us: u64,
        candidate_confidence: Option<f32>,
        candidate_error: Option<String>,
    ) -> Self {
        let disagreement = baseline_output.distance(&candidate_output);
        let baseline_executed_only = executed_output == baseline_output;
        Self {
            schema_version: LOCOMOTION_SHADOW_SCHEMA_VERSION,
            frame_id: frame_id.into(),
            t_ms,
            input_id: locomotion_input_id(&input),
            input,
            baseline_output,
            candidate_output,
            executed_output,
            baseline_provenance: baseline_provenance.into(),
            candidate_provenance: candidate_provenance.into(),
            baseline_inference_us,
            candidate_inference_us,
            candidate_confidence,
            candidate_error,
            disagreement,
            baseline_executed_only,
        }
    }
}

/// A stable identity for the exact normalized input presented to both policies.
pub fn locomotion_input_id(input: &LocomotionInput) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in input.schema_version.to_le_bytes().into_iter().chain(
        input
            .features()
            .into_iter()
            .flat_map(|value| value.to_bits().to_le_bytes()),
    ) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("locomotion-input-v{}-{hash:016x}", input.schema_version)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LocomotionPolicyMetrics {
    pub collision_rate: f32,
    pub progress_m: f32,
    pub oscillations_per_m: f32,
    pub energy_per_m: f32,
    pub recovery_success_rate: f32,
    pub command_instability: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowEnvironment {
    HeldOutSimulation,
    Physical,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocomotionShadowReport {
    pub schema_version: u32,
    pub environment: ShadowEnvironment,
    pub baseline_id: String,
    pub candidate_id: String,
    pub capture_ids: Vec<String>,
    pub episodes: u32,
    pub total_frames: u64,
    pub aligned_input_frames: u64,
    pub baseline_executed_only: bool,
    pub proposal_only: bool,
    pub conductor_gate_observed: bool,
    pub autonomic_gate_observed: bool,
    pub final_motor_gate_observed: bool,
    pub possession_lease_observed: bool,
    pub brainstem_gate_observed: bool,
    pub safety_invariant_violations: u32,
    pub hardcoded_fallback_verified: bool,
    pub atomic_activation_verified: bool,
    pub rollback_verified: bool,
    pub baseline: LocomotionPolicyMetrics,
    pub candidate: LocomotionPolicyMetrics,
}

impl LocomotionShadowReport {
    fn safety_chain_complete(&self) -> bool {
        self.proposal_only
            && self.conductor_gate_observed
            && self.autonomic_gate_observed
            && self.final_motor_gate_observed
            && self.possession_lease_observed
            && self.brainstem_gate_observed
            && self.safety_invariant_violations == 0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocomotionPromotionEvidence {
    pub schema_version: u32,
    pub simulation: LocomotionShadowReport,
    pub physical: LocomotionShadowReport,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocomotionPromotionPolicy {
    pub minimum_simulation_episodes: u32,
    pub minimum_physical_episodes: u32,
    pub minimum_aligned_fraction: f32,
    pub minimum_progress_gain: f32,
    pub maximum_collision_regression: f32,
    pub maximum_oscillation_regression: f32,
    pub maximum_energy_regression: f32,
    pub maximum_instability_regression: f32,
    pub minimum_recovery_gain: f32,
    pub maximum_sim_physical_gain_delta: f32,
}

impl Default for LocomotionPromotionPolicy {
    fn default() -> Self {
        Self {
            minimum_simulation_episodes: 20,
            minimum_physical_episodes: 10,
            minimum_aligned_fraction: 1.0,
            minimum_progress_gain: 0.02,
            maximum_collision_regression: 0.0,
            maximum_oscillation_regression: 0.0,
            maximum_energy_regression: 0.05,
            maximum_instability_regression: 0.0,
            minimum_recovery_gain: 0.0,
            maximum_sim_physical_gain_delta: 0.25,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocomotionPromotionDecision {
    pub promote: bool,
    pub reasons: Vec<String>,
}

pub fn evaluate_locomotion_promotion(
    evidence: &LocomotionPromotionEvidence,
    policy: LocomotionPromotionPolicy,
) -> LocomotionPromotionDecision {
    let mut reasons = Vec::new();
    validate_report(
        &evidence.simulation,
        ShadowEnvironment::HeldOutSimulation,
        policy.minimum_simulation_episodes,
        policy,
        &mut reasons,
    );
    validate_report(
        &evidence.physical,
        ShadowEnvironment::Physical,
        policy.minimum_physical_episodes,
        policy,
        &mut reasons,
    );
    if evidence.simulation.baseline_id != evidence.physical.baseline_id
        || evidence.simulation.candidate_id != evidence.physical.candidate_id
    {
        reasons.push("simulation and physical reports do not identify the same policy pair".into());
    }
    let sim_gain = relative_gain(
        evidence.simulation.baseline.progress_m,
        evidence.simulation.candidate.progress_m,
    );
    let physical_gain = relative_gain(
        evidence.physical.baseline.progress_m,
        evidence.physical.candidate.progress_m,
    );
    if (sim_gain - physical_gain).abs() > policy.maximum_sim_physical_gain_delta {
        reasons.push(format!(
            "physical progress gain {physical_gain:.3} is inconsistent with simulation gain {sim_gain:.3}"
        ));
    }
    if !evidence.physical.hardcoded_fallback_verified {
        reasons.push("hardcoded fallback was not verified on the physical path".into());
    }
    if !evidence.physical.atomic_activation_verified || !evidence.physical.rollback_verified {
        reasons.push("atomic activation and rollback were not both verified".into());
    }
    LocomotionPromotionDecision {
        promote: reasons.is_empty(),
        reasons,
    }
}

fn validate_report(
    report: &LocomotionShadowReport,
    expected_environment: ShadowEnvironment,
    minimum_episodes: u32,
    policy: LocomotionPromotionPolicy,
    reasons: &mut Vec<String>,
) {
    let label = match expected_environment {
        ShadowEnvironment::HeldOutSimulation => "simulation",
        ShadowEnvironment::Physical => "physical",
    };
    if report.environment != expected_environment {
        reasons.push(format!("{label} report has the wrong environment"));
    }
    if report.episodes < minimum_episodes {
        reasons.push(format!(
            "{label} report has {} episodes; {minimum_episodes} required",
            report.episodes
        ));
    }
    let aligned_fraction = if report.total_frames == 0 {
        0.0
    } else {
        report.aligned_input_frames as f32 / report.total_frames as f32
    };
    if aligned_fraction < policy.minimum_aligned_fraction {
        reasons.push(format!(
            "{label} exact-input alignment is {aligned_fraction:.3}; {:.3} required",
            policy.minimum_aligned_fraction
        ));
    }
    if !report.baseline_executed_only {
        reasons.push(format!("{label} shadow run did not execute baseline only"));
    }
    if !report.safety_chain_complete() {
        reasons.push(format!("{label} safety-authority chain is incomplete"));
    }
    let baseline = report.baseline;
    let candidate = report.candidate;
    if relative_gain(baseline.progress_m, candidate.progress_m) < policy.minimum_progress_gain {
        reasons.push(format!("{label} candidate progress did not beat baseline"));
    }
    reject_regression(
        label,
        "collision rate",
        baseline.collision_rate,
        candidate.collision_rate,
        policy.maximum_collision_regression,
        reasons,
    );
    reject_regression(
        label,
        "oscillation",
        baseline.oscillations_per_m,
        candidate.oscillations_per_m,
        policy.maximum_oscillation_regression,
        reasons,
    );
    reject_regression(
        label,
        "energy",
        baseline.energy_per_m,
        candidate.energy_per_m,
        policy.maximum_energy_regression,
        reasons,
    );
    reject_regression(
        label,
        "command instability",
        baseline.command_instability,
        candidate.command_instability,
        policy.maximum_instability_regression,
        reasons,
    );
    if candidate.recovery_success_rate + policy.minimum_recovery_gain
        < baseline.recovery_success_rate
    {
        reasons.push(format!("{label} candidate recovery regressed"));
    }
}

fn reject_regression(
    label: &str,
    metric: &str,
    baseline: f32,
    candidate: f32,
    allowed_relative_regression: f32,
    reasons: &mut Vec<String>,
) {
    let ceiling = baseline * (1.0 + allowed_relative_regression.max(0.0));
    if candidate > ceiling + f32::EPSILON {
        reasons.push(format!("{label} candidate {metric} regressed"));
    }
}

fn relative_gain(baseline: f32, candidate: f32) -> f32 {
    if baseline.abs() <= f32::EPSILON {
        if candidate > baseline { 1.0 } else { 0.0 }
    } else {
        (candidate - baseline) / baseline.abs()
    }
}
