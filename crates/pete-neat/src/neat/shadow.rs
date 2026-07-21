pub const LOCOMOTION_SHADOW_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocomotionShadowFrame {
    pub schema_version: u32,
    pub frame_id: String,
    pub t_ms: u64,
    pub input_id: String,
    pub input: LocomotionInput,
    pub baseline_output: LocomotionOutput,
    pub candidate_output: Option<LocomotionOutput>,
    pub executed_output: LocomotionOutput,
    pub baseline_provenance: String,
    pub candidate_provenance: String,
    pub baseline_inference_us: Option<u64>,
    pub candidate_inference_us: Option<u64>,
    pub candidate_confidence: Option<f32>,
    pub candidate_error: Option<String>,
    pub disagreement: Option<f32>,
    pub baseline_executed_only: bool,
}

impl LocomotionShadowFrame {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frame_id: impl Into<String>,
        t_ms: u64,
        input: LocomotionInput,
        baseline_output: LocomotionOutput,
        candidate_output: Option<LocomotionOutput>,
        executed_output: LocomotionOutput,
        baseline_provenance: impl Into<String>,
        candidate_provenance: impl Into<String>,
        baseline_inference_us: Option<u64>,
        candidate_inference_us: Option<u64>,
        candidate_confidence: Option<f32>,
        candidate_error: Option<String>,
    ) -> Self {
        let disagreement = candidate_output
            .as_ref()
            .map(|candidate| baseline_output.distance(candidate));
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

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocomotionPromotionEvidence {
    pub schema_version: u32,
    pub simulation: LocomotionShadowReport,
    pub physical: LocomotionShadowReport,
    pub artifacts: LocomotionPromotionArtifacts,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromotionArtifactRef {
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShadowCaptureArtifacts {
    pub capture_id: String,
    pub manifest: PromotionArtifactRef,
    pub shadow_frames: PromotionArtifactRef,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocomotionPromotionArtifacts {
    pub simulation_captures: Vec<ShadowCaptureArtifacts>,
    pub physical_captures: Vec<ShadowCaptureArtifacts>,
    pub candidate_identity: PromotionArtifactRef,
    pub candidate_checkpoint: PromotionArtifactRef,
    pub activation_ledger: PromotionArtifactRef,
    pub rollback_record: PromotionArtifactRef,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocomotionCaptureManifestEvidence {
    pub capture_id: String,
    pub environment: ShadowEnvironment,
    pub episodes: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub possession_lease_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brainstem_firmware_identity: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocomotionShadowSafetyTrace {
    pub proposal_only: bool,
    pub conductor_gate_executed: bool,
    pub autonomic_gate_executed: bool,
    pub final_motor_gate_executed: bool,
    pub safety_invariant_violation: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecordedLocomotionShadowFrame {
    pub shadow: LocomotionShadowFrame,
    pub safety: LocomotionShadowSafetyTrace,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocomotionCandidateIdentityArtifact {
    pub candidate_id: String,
    pub checkpoint_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocomotionActivationTransition {
    pub sequence: u64,
    pub from_policy_id: String,
    pub to_policy_id: String,
    pub candidate_id: String,
    pub checkpoint_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocomotionRollbackRecord {
    pub activation_sequence: u64,
    pub rollback_sequence: u64,
    pub candidate_id: String,
    pub checkpoint_sha256: String,
    pub restored_policy_id: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VerifiedShadowReport {
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocomotionPromotionArtifactVerification {
    simulation: VerifiedShadowReport,
    physical: VerifiedShadowReport,
    candidate_identity_verified: bool,
    atomic_activation_verified: bool,
    rollback_verified: bool,
}

pub fn verify_locomotion_promotion_artifacts(
    evidence: &LocomotionPromotionEvidence,
    artifact_root: &Path,
) -> Result<LocomotionPromotionArtifactVerification> {
    let simulation = verify_shadow_report_artifacts(
        &evidence.simulation,
        &evidence.artifacts.simulation_captures,
        ShadowEnvironment::HeldOutSimulation,
        artifact_root,
    )?;
    let physical = verify_shadow_report_artifacts(
        &evidence.physical,
        &evidence.artifacts.physical_captures,
        ShadowEnvironment::Physical,
        artifact_root,
    )?;
    let checkpoint_bytes = verified_artifact_bytes(
        artifact_root,
        &evidence.artifacts.candidate_checkpoint,
    )?;
    let checkpoint_sha256 = sha256_hex(&checkpoint_bytes);
    let identity: LocomotionCandidateIdentityArtifact = serde_json::from_slice(
        &verified_artifact_bytes(artifact_root, &evidence.artifacts.candidate_identity)?,
    )
    .context("parsing candidate identity artifact")?;
    let candidate_identity_verified = identity.candidate_id == evidence.simulation.candidate_id
        && identity.candidate_id == evidence.physical.candidate_id
        && normalized_sha256(&identity.checkpoint_sha256) == checkpoint_sha256;

    let transitions: Vec<LocomotionActivationTransition> = serde_json::from_slice(
        &verified_artifact_bytes(artifact_root, &evidence.artifacts.activation_ledger)?,
    )
    .context("parsing activation ledger artifact")?;
    if transitions
        .windows(2)
        .any(|window| window[0].sequence >= window[1].sequence)
    {
        bail!("activation ledger sequences are not strictly increasing");
    }
    let activation = transitions.iter().find(|transition| {
        transition.from_policy_id == evidence.physical.baseline_id
            && transition.to_policy_id == evidence.physical.candidate_id
            && transition.candidate_id == identity.candidate_id
            && normalized_sha256(&transition.checkpoint_sha256) == checkpoint_sha256
    });
    let atomic_activation_verified = activation.is_some();

    let rollback: LocomotionRollbackRecord = serde_json::from_slice(&verified_artifact_bytes(
        artifact_root,
        &evidence.artifacts.rollback_record,
    )?)
    .context("parsing rollback record artifact")?;
    let rollback_transition = transitions.iter().find(|transition| {
        transition.sequence == rollback.rollback_sequence
            && transition.from_policy_id == evidence.physical.candidate_id
            && transition.to_policy_id == evidence.physical.baseline_id
            && transition.candidate_id == identity.candidate_id
            && normalized_sha256(&transition.checkpoint_sha256) == checkpoint_sha256
    });
    let rollback_verified = activation.is_some_and(|activation| {
        rollback.activation_sequence == activation.sequence
            && rollback.rollback_sequence > activation.sequence
            && rollback.candidate_id == identity.candidate_id
            && normalized_sha256(&rollback.checkpoint_sha256) == checkpoint_sha256
            && rollback.restored_policy_id == evidence.physical.baseline_id
            && rollback_transition.is_some()
    });

    Ok(LocomotionPromotionArtifactVerification {
        simulation,
        physical,
        candidate_identity_verified,
        atomic_activation_verified,
        rollback_verified,
    })
}

fn verify_shadow_report_artifacts(
    report: &LocomotionShadowReport,
    captures: &[ShadowCaptureArtifacts],
    expected_environment: ShadowEnvironment,
    artifact_root: &Path,
) -> Result<VerifiedShadowReport> {
    if captures.is_empty() {
        bail!("shadow report has no capture artifacts");
    }
    let mut verified = VerifiedShadowReport {
        baseline_executed_only: true,
        proposal_only: true,
        conductor_gate_observed: true,
        autonomic_gate_observed: true,
        final_motor_gate_observed: true,
        possession_lease_observed: expected_environment == ShadowEnvironment::Physical,
        brainstem_gate_observed: expected_environment == ShadowEnvironment::Physical,
        hardcoded_fallback_verified: true,
        ..VerifiedShadowReport::default()
    };
    for capture in captures {
        let manifest: LocomotionCaptureManifestEvidence = serde_json::from_slice(
            &verified_artifact_bytes(artifact_root, &capture.manifest)?,
        )
        .with_context(|| format!("parsing capture manifest for {}", capture.capture_id))?;
        if manifest.capture_id != capture.capture_id
            || manifest.environment != expected_environment
        {
            bail!("capture {} manifest identity/environment mismatch", capture.capture_id);
        }
        verified.capture_ids.push(capture.capture_id.clone());
        verified.episodes = verified.episodes.saturating_add(manifest.episodes);
        if expected_environment == ShadowEnvironment::Physical {
            verified.possession_lease_observed &= manifest
                .possession_lease_id
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty());
            verified.brainstem_gate_observed &= manifest
                .brainstem_firmware_identity
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty());
        }
        let bytes = verified_artifact_bytes(artifact_root, &capture.shadow_frames)?;
        let text = std::str::from_utf8(&bytes)
            .with_context(|| format!("shadow frames for {} are not UTF-8", capture.capture_id))?;
        for (line_index, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let recorded: RecordedLocomotionShadowFrame = serde_json::from_str(line)
                .with_context(|| {
                    format!(
                        "parsing shadow frame {}:{}",
                        capture.capture_id,
                        line_index + 1
                    )
                })?;
            let frame = recorded.shadow;
            verified.total_frames = verified.total_frames.saturating_add(1);
            if frame.input_id == locomotion_input_id(&frame.input)
                && frame.baseline_provenance == report.baseline_id
                && frame.candidate_provenance == report.candidate_id
            {
                verified.aligned_input_frames =
                    verified.aligned_input_frames.saturating_add(1);
            }
            verified.baseline_executed_only &= frame.executed_output == frame.baseline_output;
            verified.hardcoded_fallback_verified &= frame.executed_output == frame.baseline_output
                && frame.baseline_provenance == report.baseline_id;
            verified.proposal_only &= recorded.safety.proposal_only;
            verified.conductor_gate_observed &= recorded.safety.conductor_gate_executed;
            verified.autonomic_gate_observed &= recorded.safety.autonomic_gate_executed;
            verified.final_motor_gate_observed &= recorded.safety.final_motor_gate_executed;
            verified.safety_invariant_violations = verified
                .safety_invariant_violations
                .saturating_add(u32::from(recorded.safety.safety_invariant_violation));
        }
    }
    verified.capture_ids.sort();
    let mut report_capture_ids = report.capture_ids.clone();
    report_capture_ids.sort();
    if verified.capture_ids != report_capture_ids {
        bail!("report capture IDs do not match checksummed capture artifacts");
    }
    if verified.total_frames == 0 {
        bail!("checksummed shadow frame artifacts contain no frames");
    }
    Ok(verified)
}

fn verified_artifact_bytes(root: &Path, artifact: &PromotionArtifactRef) -> Result<Vec<u8>> {
    if artifact.path.trim().is_empty() || artifact.sha256.trim().is_empty() {
        bail!("promotion artifact path/checksum is empty");
    }
    let root = root
        .canonicalize()
        .with_context(|| format!("resolving promotion artifact root {}", root.display()))?;
    let candidate = root.join(&artifact.path);
    let resolved = candidate
        .canonicalize()
        .with_context(|| format!("resolving promotion artifact {}", candidate.display()))?;
    if !resolved.starts_with(&root) {
        bail!("promotion artifact escapes its evidence root: {}", artifact.path);
    }
    let bytes = fs::read(&resolved)
        .with_context(|| format!("reading promotion artifact {}", resolved.display()))?;
    let actual = sha256_hex(&bytes);
    if actual != normalized_sha256(&artifact.sha256) {
        bail!("promotion artifact checksum mismatch: {}", artifact.path);
    }
    Ok(bytes)
}

fn normalized_sha256(value: &str) -> String {
    value
        .trim()
        .strip_prefix("sha256:")
        .unwrap_or(value.trim())
        .to_ascii_lowercase()
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
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
    verification: Option<&LocomotionPromotionArtifactVerification>,
) -> LocomotionPromotionDecision {
    let mut reasons = Vec::new();
    if evidence.schema_version != LOCOMOTION_SHADOW_SCHEMA_VERSION {
        reasons.push(format!(
            "promotion evidence schema {} is unsupported; expected {}",
            evidence.schema_version, LOCOMOTION_SHADOW_SCHEMA_VERSION
        ));
    }
    if !valid_policy(policy) {
        reasons.push("promotion policy contains invalid thresholds".into());
    }
    validate_report(
        &evidence.simulation,
        verification.map(|value| &value.simulation),
        ShadowEnvironment::HeldOutSimulation,
        policy.minimum_simulation_episodes,
        policy,
        &mut reasons,
    );
    validate_report(
        &evidence.physical,
        verification.map(|value| &value.physical),
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
    if !verification.is_some_and(|value| value.candidate_identity_verified) {
        reasons.push("candidate identity is not bound to the checksummed checkpoint".into());
    }
    if !verification.is_some_and(|value| value.physical.hardcoded_fallback_verified) {
        reasons.push("hardcoded fallback was not verified on the physical path".into());
    }
    if !verification.is_some_and(|value| {
        value.atomic_activation_verified && value.rollback_verified
    }) {
        reasons.push("atomic activation and rollback were not both verified".into());
    }
    LocomotionPromotionDecision {
        promote: reasons.is_empty(),
        reasons,
    }
}

fn validate_report(
    report: &LocomotionShadowReport,
    verified: Option<&VerifiedShadowReport>,
    expected_environment: ShadowEnvironment,
    minimum_episodes: u32,
    policy: LocomotionPromotionPolicy,
    reasons: &mut Vec<String>,
) {
    let label = match expected_environment {
        ShadowEnvironment::HeldOutSimulation => "simulation",
        ShadowEnvironment::Physical => "physical",
    };
    if report.schema_version != LOCOMOTION_SHADOW_SCHEMA_VERSION {
        reasons.push(format!(
            "{label} report schema {} is unsupported; expected {}",
            report.schema_version, LOCOMOTION_SHADOW_SCHEMA_VERSION
        ));
    }
    if report.environment != expected_environment {
        reasons.push(format!("{label} report has the wrong environment"));
    }
    if report.baseline_id.trim().is_empty()
        || report.candidate_id.trim().is_empty()
        || report.baseline_id == report.candidate_id
    {
        reasons.push(format!("{label} report has invalid policy identities"));
    }
    if report.capture_ids.is_empty() || report.capture_ids.iter().any(|id| id.trim().is_empty()) {
        reasons.push(format!("{label} report has invalid capture provenance"));
    }
    let Some(verified) = verified else {
        reasons.push(format!(
            "{label} report has no independently verified artifact evidence"
        ));
        return;
    };
    if report.capture_ids != verified.capture_ids
        || report.episodes != verified.episodes
        || report.total_frames != verified.total_frames
        || report.aligned_input_frames != verified.aligned_input_frames
        || report.baseline_executed_only != verified.baseline_executed_only
        || report.proposal_only != verified.proposal_only
        || report.conductor_gate_observed != verified.conductor_gate_observed
        || report.autonomic_gate_observed != verified.autonomic_gate_observed
        || report.final_motor_gate_observed != verified.final_motor_gate_observed
        || report.possession_lease_observed != verified.possession_lease_observed
        || report.brainstem_gate_observed != verified.brainstem_gate_observed
        || report.safety_invariant_violations != verified.safety_invariant_violations
        || report.hardcoded_fallback_verified != verified.hardcoded_fallback_verified
    {
        reasons.push(format!(
            "{label} report claims do not match checksummed artifacts"
        ));
    }
    if verified.episodes < minimum_episodes {
        reasons.push(format!(
            "{label} report has {} verified episodes; {minimum_episodes} required",
            verified.episodes
        ));
    }
    if verified.total_frames == 0 || verified.aligned_input_frames > verified.total_frames {
        reasons.push(format!("{label} report has invalid frame counts"));
    }
    let aligned_fraction = if verified.total_frames == 0 {
        0.0
    } else {
        verified.aligned_input_frames as f32 / verified.total_frames as f32
    };
    if aligned_fraction < policy.minimum_aligned_fraction {
        reasons.push(format!(
            "{label} exact-input alignment is {aligned_fraction:.3}; {:.3} required",
            policy.minimum_aligned_fraction
        ));
    }
    if !verified.baseline_executed_only {
        reasons.push(format!("{label} shadow run did not execute baseline only"));
    }
    let safety_chain_complete = verified.proposal_only
        && verified.conductor_gate_observed
        && verified.autonomic_gate_observed
        && verified.final_motor_gate_observed
        && verified.safety_invariant_violations == 0
        && (expected_environment != ShadowEnvironment::Physical
            || (verified.possession_lease_observed && verified.brainstem_gate_observed));
    if !safety_chain_complete {
        reasons.push(format!("{label} safety-authority chain is incomplete"));
    }
    let baseline = report.baseline;
    let candidate = report.candidate;
    if !valid_metrics(baseline) || !valid_metrics(candidate) {
        reasons.push(format!("{label} report contains invalid policy metrics"));
        return;
    }
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

fn valid_policy(policy: LocomotionPromotionPolicy) -> bool {
    let values = [
        policy.minimum_aligned_fraction,
        policy.minimum_progress_gain,
        policy.maximum_collision_regression,
        policy.maximum_oscillation_regression,
        policy.maximum_energy_regression,
        policy.maximum_instability_regression,
        policy.minimum_recovery_gain,
        policy.maximum_sim_physical_gain_delta,
    ];
    values.into_iter().all(f32::is_finite)
        && (0.0..=1.0).contains(&policy.minimum_aligned_fraction)
        && policy.minimum_progress_gain >= 0.0
        && policy.maximum_collision_regression >= 0.0
        && policy.maximum_oscillation_regression >= 0.0
        && policy.maximum_energy_regression >= 0.0
        && policy.maximum_instability_regression >= 0.0
        && policy.minimum_recovery_gain >= 0.0
        && policy.maximum_sim_physical_gain_delta >= 0.0
}

fn valid_metrics(metrics: LocomotionPolicyMetrics) -> bool {
    let values = [
        metrics.collision_rate,
        metrics.progress_m,
        metrics.oscillations_per_m,
        metrics.energy_per_m,
        metrics.recovery_success_rate,
        metrics.command_instability,
    ];
    values.into_iter().all(f32::is_finite)
        && (0.0..=1.0).contains(&metrics.collision_rate)
        && metrics.progress_m >= 0.0
        && metrics.oscillations_per_m >= 0.0
        && metrics.energy_per_m >= 0.0
        && (0.0..=1.0).contains(&metrics.recovery_success_rate)
        && metrics.command_instability >= 0.0
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
