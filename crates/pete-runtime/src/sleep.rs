use std::collections::{BTreeMap, BTreeSet, VecDeque};

use pete_now::{ClockDomain, EvidenceRef, TypedTimestamp};
use serde::{Deserialize, Serialize};

const DEFAULT_SLEEP_WALL_BUDGET_MS: u64 = 30 * 60 * 1_000;
const FATIGUE_ENTRY_THRESHOLD: f32 = 0.80;
const FATIGUE_REARM_THRESHOLD: f32 = 0.65;
const MAX_CONSUMED_SLEEP_INPUT_REFS: usize = 4_096;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SleepPhase {
    #[default]
    Awake,
    Preparing,
    Quiescent,
    Consolidating,
    Training,
    Evaluating,
    Finalizing,
    Waking,
    Interrupted,
}

impl SleepPhase {
    pub fn is_asleep(self) -> bool {
        !matches!(self, Self::Awake | Self::Waking | Self::Interrupted)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SleepTrigger {
    HighFatigue,
    SustainedCharging,
    StableDocked,
    IdleWindow,
    DeferredWork,
    EpisodeEnded,
    #[default]
    OperatorRequest,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WakePriority {
    #[default]
    Routine,
    Social,
    Operator,
    Homeostasis,
    Communication,
    Safety,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WakeReason {
    WorkPlanComplete,
    ResourceBudgetExhausted,
    DirectOperatorCommand,
    ImportantSocialCue,
    CriticalBattery,
    ExternalPowerLost,
    BodyCommunicationLost,
    SafetyEvent(String),
    ScheduledDeadline,
    #[default]
    ExplicitWake,
}

impl WakeReason {
    pub fn priority(&self) -> WakePriority {
        match self {
            Self::SafetyEvent(_) => WakePriority::Safety,
            Self::BodyCommunicationLost => WakePriority::Communication,
            Self::CriticalBattery | Self::ExternalPowerLost => WakePriority::Homeostasis,
            Self::DirectOperatorCommand | Self::ExplicitWake => WakePriority::Operator,
            Self::ImportantSocialCue => WakePriority::Social,
            Self::WorkPlanComplete | Self::ResourceBudgetExhausted | Self::ScheduledDeadline => {
                WakePriority::Routine
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SleepWorkKind {
    FlushDurableState,
    ConsolidateEpisodes,
    ReplayRecentFailures,
    TrainCandidate,
    EvaluateCandidate,
    RebuildIndexes,
    DryRunPruning,
    #[default]
    SummarizeChanges,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkLocality {
    #[default]
    Local,
    AcceleratorPreferred,
    AcceleratorRequired,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkCancellationPolicy {
    #[default]
    Restartable,
    Resumable,
    Atomic,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidatePromotionPolicy {
    #[default]
    EvaluateOnly,
    RecommendForShadow,
    RequiresAttendedValidation,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SleepResourceEstimate {
    pub wall_time_ms: u64,
    pub cpu_time_ms: u64,
    pub memory_mb: u64,
    pub disk_growth_mb: u64,
    pub energy_wh: f32,
    pub network_mb: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SleepResourceBudget {
    pub wall_time_ms: u64,
    pub cpu_time_ms: u64,
    pub memory_mb: u64,
    pub disk_growth_mb: u64,
    pub energy_wh: f32,
    pub network_mb: u64,
    pub max_thermal_fraction: f32,
    pub used: SleepResourceEstimate,
}

impl Default for SleepResourceBudget {
    fn default() -> Self {
        Self {
            wall_time_ms: DEFAULT_SLEEP_WALL_BUDGET_MS,
            cpu_time_ms: 10 * 60 * 1_000,
            memory_mb: 1_024,
            disk_growth_mb: 512,
            energy_wh: 5.0,
            network_mb: 256,
            max_thermal_fraction: 0.80,
            used: SleepResourceEstimate::default(),
        }
    }
}

impl SleepResourceBudget {
    fn can_reserve(&self, estimate: &SleepResourceEstimate) -> bool {
        self.used.wall_time_ms.saturating_add(estimate.wall_time_ms) <= self.wall_time_ms
            && self.used.cpu_time_ms.saturating_add(estimate.cpu_time_ms) <= self.cpu_time_ms
            && self.used.memory_mb.max(estimate.memory_mb) <= self.memory_mb
            && self
                .used
                .disk_growth_mb
                .saturating_add(estimate.disk_growth_mb)
                <= self.disk_growth_mb
            && self.used.energy_wh + estimate.energy_wh <= self.energy_wh
            && self.used.network_mb.saturating_add(estimate.network_mb) <= self.network_mb
    }

    fn reserve(&mut self, estimate: &SleepResourceEstimate) {
        self.used.wall_time_ms = self.used.wall_time_ms.saturating_add(estimate.wall_time_ms);
        self.used.cpu_time_ms = self.used.cpu_time_ms.saturating_add(estimate.cpu_time_ms);
        self.used.memory_mb = self.used.memory_mb.max(estimate.memory_mb);
        self.used.disk_growth_mb = self
            .used
            .disk_growth_mb
            .saturating_add(estimate.disk_growth_mb);
        self.used.energy_wh += estimate.energy_wh;
        self.used.network_mb = self.used.network_mb.saturating_add(estimate.network_mb);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SleepWorkItem {
    pub id: String,
    pub kind: SleepWorkKind,
    #[serde(default)]
    pub input_artifact_refs: Vec<String>,
    #[serde(default)]
    pub input_schema_versions: BTreeMap<String, u32>,
    #[serde(default)]
    pub dependencies: Vec<String>,
    pub estimate: SleepResourceEstimate,
    pub locality: WorkLocality,
    pub cancellation: WorkCancellationPolicy,
    pub output_contract: String,
    pub verification: String,
    pub promotion_policy: CandidatePromotionPolicy,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SleepWorkStatus {
    #[default]
    Pending,
    Completed,
    Failed,
    Deferred,
    Cancelled,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReplayArtifact {
    pub artifact_id: String,
    #[serde(default)]
    pub source_episode_refs: Vec<String>,
    pub historical_time_domain: ClockDomain,
    pub replay_computed_at: TypedTimestamp,
    pub injected_into_live_now: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConsolidationArtifact {
    pub artifact_id: String,
    #[serde(default)]
    pub source_episode_refs: Vec<String>,
    #[serde(default)]
    pub semantic_relation_refs: Vec<String>,
    pub source_history_preserved: bool,
    pub deterministic_index_entries: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CandidateArtifact {
    pub artifact_id: String,
    pub role: String,
    pub interface_version: u32,
    #[serde(default)]
    pub training_data_refs: Vec<String>,
    pub configuration: String,
    pub seed: u64,
    #[serde(default)]
    pub metrics: BTreeMap<String, f32>,
    #[serde(default)]
    pub warnings: Vec<String>,
    pub baseline_comparison: String,
    #[serde(default)]
    pub known_failure_slices: Vec<String>,
    pub fallback_artifact_ref: Option<String>,
    pub promotion_policy: CandidatePromotionPolicy,
    pub promotion_recommended: bool,
    pub automatically_promoted: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SleepWorkResult {
    pub work_item_id: String,
    pub kind: SleepWorkKind,
    pub status: SleepWorkStatus,
    pub started_at_ms: u64,
    pub completed_at_ms: Option<u64>,
    pub executor: String,
    #[serde(default)]
    pub output_artifact_refs: Vec<String>,
    pub summary: String,
    pub replay: Option<ReplayArtifact>,
    pub consolidation: Option<ConsolidationArtifact>,
    pub candidate: Option<CandidateArtifact>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SleepSessionId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SleepSession {
    pub id: SleepSessionId,
    pub started_at_ms: u64,
    pub phase: SleepPhase,
    pub trigger: SleepTrigger,
    pub started_on_external_power: bool,
    pub resource_budget: SleepResourceBudget,
    #[serde(default)]
    pub work_plan: Vec<SleepWorkItem>,
    #[serde(default)]
    pub completed: Vec<SleepWorkResult>,
    pub interrupted_by: Option<WakeReason>,
    #[serde(default)]
    pub provenance: Vec<EvidenceRef>,
    #[serde(default)]
    pub claimed_input_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SleepEligibility {
    pub eligible: bool,
    pub trigger: Option<SleepTrigger>,
    #[serde(default)]
    pub blocking_reasons: Vec<String>,
    pub expensive_work_allowed: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SleepReport {
    pub schema_version: u32,
    pub session_id: SleepSessionId,
    pub started_at_ms: u64,
    pub ended_at_ms: u64,
    pub trigger: SleepTrigger,
    pub wake_reason: WakeReason,
    #[serde(default)]
    pub completed: Vec<SleepWorkResult>,
    #[serde(default)]
    pub failed: Vec<String>,
    #[serde(default)]
    pub deferred: Vec<String>,
    #[serde(default)]
    pub produced_artifacts: Vec<String>,
    pub promoted_artifact: Option<String>,
    pub fresh_world_model_required: bool,
    pub stale_skill_resumed: bool,
    #[serde(default)]
    pub consumed_input_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SleepSnapshot {
    pub schema_version: u32,
    pub t_ms: u64,
    pub phase: SleepPhase,
    pub eligibility: SleepEligibility,
    pub session: Option<SleepSession>,
    pub last_report: Option<SleepReport>,
}

#[derive(Clone, Debug, Default)]
pub struct SleepTickInput {
    pub now_ms: u64,
    pub fatigue_activation: f32,
    pub charging: bool,
    pub docked: bool,
    pub stopped: bool,
    pub direct_reign_active: bool,
    pub unresolved_urgent_need: bool,
    pub body_communication_stable: bool,
    pub active_skill_interruptible: bool,
    pub critical_battery: bool,
    pub external_power_lost: bool,
    pub safety_event: Option<String>,
    pub important_social_cue: bool,
    pub operator_sleep_request: bool,
    pub operator_wake_request: bool,
    pub accelerator_available: bool,
    pub thermal_fraction: f32,
    pub completed_episode_refs: Vec<String>,
    pub failed_behavior_refs: Vec<String>,
    pub semantic_relation_refs: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct SleepController {
    sequence: u64,
    session: Option<SleepSession>,
    last_report: Option<SleepReport>,
    consumed_input_refs: VecDeque<String>,
    fatigue_entry_armed: bool,
    operator_sleep_request_active: bool,
}

impl Default for SleepController {
    fn default() -> Self {
        Self {
            sequence: 0,
            session: None,
            last_report: None,
            consumed_input_refs: VecDeque::new(),
            fatigue_entry_armed: true,
            operator_sleep_request_active: false,
        }
    }
}

impl SleepController {
    pub fn snapshot(
        &self,
        now_ms: u64,
        eligibility: SleepEligibility,
    ) -> SleepSnapshot {
        SleepSnapshot {
            schema_version: 1,
            t_ms: now_ms,
            phase: self
                .session
                .as_ref()
                .map(|session| session.phase)
                .unwrap_or(SleepPhase::Awake),
            eligibility,
            session: self.session.clone(),
            last_report: self.last_report.clone(),
        }
    }

    pub fn requires_quiescence(&self) -> bool {
        self.session
            .as_ref()
            .is_some_and(|session| session.phase.is_asleep())
    }

    pub fn expects_external_power(&self) -> bool {
        self.session
            .as_ref()
            .is_some_and(|session| session.started_on_external_power)
    }

    pub fn tick(&mut self, input: SleepTickInput) -> SleepSnapshot {
        if input.fatigue_activation <= FATIGUE_REARM_THRESHOLD {
            self.fatigue_entry_armed = true;
        }
        let operator_request_edge =
            input.operator_sleep_request && !self.operator_sleep_request_active;
        self.operator_sleep_request_active = input.operator_sleep_request;
        let pending_input = self.pending_input(&input);
        let eligibility = sleep_eligibility_with_arms(
            &pending_input,
            operator_request_edge,
            self.fatigue_entry_armed,
        );
        if self.session.is_none() {
            if eligibility.eligible {
                self.start_session(&pending_input, &eligibility);
            }
            return self.snapshot(input.now_ms, eligibility);
        }

        if let Some(reason) = wake_reason(&input) {
            let session = self.session.as_mut().expect("checked session");
            session.interrupted_by = Some(reason);
            session.phase = SleepPhase::Interrupted;
            return self.snapshot(input.now_ms, eligibility);
        }

        let phase = self.session.as_ref().expect("checked session").phase;
        match phase {
            SleepPhase::Preparing => self.session_mut().phase = SleepPhase::Quiescent,
            SleepPhase::Quiescent => self.session_mut().phase = SleepPhase::Consolidating,
            SleepPhase::Consolidating => {
                self.execute_phase(
                    &input,
                    &[
                        SleepWorkKind::FlushDurableState,
                        SleepWorkKind::ConsolidateEpisodes,
                        SleepWorkKind::ReplayRecentFailures,
                    ],
                );
                self.session_mut().phase = SleepPhase::Training;
            }
            SleepPhase::Training => {
                self.execute_phase(&input, &[SleepWorkKind::TrainCandidate]);
                self.session_mut().phase = SleepPhase::Evaluating;
            }
            SleepPhase::Evaluating => {
                self.execute_phase(&input, &[SleepWorkKind::EvaluateCandidate]);
                self.session_mut().phase = SleepPhase::Finalizing;
            }
            SleepPhase::Finalizing => self.session_mut().phase = SleepPhase::Waking,
            SleepPhase::Interrupted => self.session_mut().phase = SleepPhase::Waking,
            SleepPhase::Waking => self.finish_session(input.now_ms),
            SleepPhase::Awake => self.session = None,
        }
        let eligibility = if self.session.is_none() {
            let pending_input = self.pending_input(&input);
            sleep_eligibility_with_arms(&pending_input, false, self.fatigue_entry_armed)
        } else {
            eligibility
        };
        self.snapshot(input.now_ms, eligibility)
    }

    fn start_session(&mut self, input: &SleepTickInput, eligibility: &SleepEligibility) {
        self.sequence = self.sequence.saturating_add(1);
        let trigger = eligibility.trigger.unwrap_or(SleepTrigger::OperatorRequest);
        if input.fatigue_activation >= FATIGUE_ENTRY_THRESHOLD {
            self.fatigue_entry_armed = false;
        }
        let id = SleepSessionId(format!("sleep:{}:{}", input.now_ms, self.sequence));
        let claimed_input_refs = sleep_input_refs(input);
        self.session = Some(SleepSession {
            id: id.clone(),
            started_at_ms: input.now_ms,
            phase: SleepPhase::Preparing,
            trigger,
            started_on_external_power: input.charging,
            resource_budget: SleepResourceBudget::default(),
            work_plan: deterministic_work_plan(&id, input),
            completed: Vec::new(),
            interrupted_by: None,
            provenance: vec![EvidenceRef {
                id: format!("sleep:trigger:{:?}:{}", trigger, input.now_ms).to_lowercase(),
                source: if input.operator_sleep_request {
                    "operator.sleep_request".to_string()
                } else {
                    "self.homeostasis".to_string()
                },
                key: "sleep.trigger".to_string(),
                observed_at_ms: input.now_ms,
                transformation_lineage: vec!["pete_runtime::SleepController".to_string()],
                implementation_version: Some("1".to_string()),
            }],
            claimed_input_refs,
        });
    }

    fn session_mut(&mut self) -> &mut SleepSession {
        self.session.as_mut().expect("sleep session exists")
    }

    fn execute_phase(&mut self, input: &SleepTickInput, kinds: &[SleepWorkKind]) {
        let pending = self
            .session
            .as_ref()
            .expect("sleep session exists")
            .work_plan
            .iter()
            .filter(|item| kinds.contains(&item.kind))
            .cloned()
            .collect::<Vec<_>>();
        for item in pending {
            let already_completed = self
                .session
                .as_ref()
                .expect("sleep session exists")
                .completed
                .iter()
                .any(|result| result.work_item_id == item.id);
            if already_completed {
                continue;
            }
            let result = execute_work_item(self.session_mut(), &item, input);
            self.session_mut().completed.push(result);
        }
    }

    fn finish_session(&mut self, now_ms: u64) {
        let session = self.session.take().expect("sleep session exists");
        let wake_reason = session
            .interrupted_by
            .clone()
            .unwrap_or(WakeReason::WorkPlanComplete);
        let completed = session.completed.clone();
        let produced_artifacts = completed
            .iter()
            .flat_map(|result| result.output_artifact_refs.clone())
            .collect();
        let failed = completed
            .iter()
            .filter(|result| result.status == SleepWorkStatus::Failed)
            .map(|result| result.work_item_id.clone())
            .collect();
        let deferred = completed
            .iter()
            .filter(|result| result.status == SleepWorkStatus::Deferred)
            .map(|result| result.work_item_id.clone())
            .collect();
        let consumed_input_refs = if session.interrupted_by.is_none() {
            session.claimed_input_refs.clone()
        } else {
            Vec::new()
        };
        for input_ref in &consumed_input_refs {
            self.mark_consumed(input_ref.clone());
        }
        self.last_report = Some(SleepReport {
            schema_version: 1,
            session_id: session.id,
            started_at_ms: session.started_at_ms,
            ended_at_ms: now_ms,
            trigger: session.trigger,
            wake_reason,
            completed,
            failed,
            deferred,
            produced_artifacts,
            promoted_artifact: None,
            fresh_world_model_required: true,
            stale_skill_resumed: false,
            consumed_input_refs,
        });
    }

    fn pending_input(&self, input: &SleepTickInput) -> SleepTickInput {
        let mut pending = input.clone();
        pending
            .completed_episode_refs
            .retain(|input_ref| !self.consumed_input_refs.contains(input_ref));
        pending
            .failed_behavior_refs
            .retain(|input_ref| !self.consumed_input_refs.contains(input_ref));
        pending
            .semantic_relation_refs
            .retain(|input_ref| !self.consumed_input_refs.contains(input_ref));
        pending
    }

    fn mark_consumed(&mut self, input_ref: String) {
        if self.consumed_input_refs.contains(&input_ref) {
            return;
        }
        self.consumed_input_refs.push_back(input_ref);
        while self.consumed_input_refs.len() > MAX_CONSUMED_SLEEP_INPUT_REFS {
            self.consumed_input_refs.pop_front();
        }
    }
}

pub fn sleep_eligibility(input: &SleepTickInput) -> SleepEligibility {
    sleep_eligibility_with_arms(input, input.operator_sleep_request, true)
}

fn sleep_eligibility_with_arms(
    input: &SleepTickInput,
    operator_request_edge: bool,
    fatigue_entry_armed: bool,
) -> SleepEligibility {
    let trigger = if operator_request_edge {
        Some(SleepTrigger::OperatorRequest)
    } else if fatigue_entry_armed && input.fatigue_activation >= FATIGUE_ENTRY_THRESHOLD {
        Some(SleepTrigger::HighFatigue)
    } else if has_deferred_work(input) && input.charging {
        Some(SleepTrigger::DeferredWork)
    } else {
        None
    };
    let mut blocking_reasons = Vec::new();
    if !input.stopped {
        blocking_reasons.push("body is moving".to_string());
    }
    if input.direct_reign_active {
        blocking_reasons.push("Direct Reign is active".to_string());
    }
    if input.unresolved_urgent_need {
        blocking_reasons.push("an urgent safety or homeostatic need is unresolved".to_string());
    }
    if !input.body_communication_stable {
        blocking_reasons.push("body communication is not stable".to_string());
    }
    if !input.active_skill_interruptible {
        blocking_reasons.push("the active skill cannot be safely interrupted".to_string());
    }
    if input.safety_event.is_some() {
        blocking_reasons.push("a safety event is active".to_string());
    }
    if input.thermal_fraction > 0.80 {
        blocking_reasons.push("thermal state exceeds the sleep work budget".to_string());
    }
    SleepEligibility {
        eligible: trigger.is_some() && blocking_reasons.is_empty(),
        trigger,
        blocking_reasons,
        expensive_work_allowed: input.charging && input.docked && input.thermal_fraction <= 0.80,
    }
}

fn has_deferred_work(input: &SleepTickInput) -> bool {
    !input.completed_episode_refs.is_empty() || !input.failed_behavior_refs.is_empty()
}

fn sleep_input_refs(input: &SleepTickInput) -> Vec<String> {
    input
        .completed_episode_refs
        .iter()
        .chain(input.failed_behavior_refs.iter())
        .chain(input.semantic_relation_refs.iter())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn wake_reason(input: &SleepTickInput) -> Option<WakeReason> {
    let mut reasons = Vec::new();
    if let Some(event) = input.safety_event.as_ref() {
        reasons.push(WakeReason::SafetyEvent(event.clone()));
    }
    if !input.body_communication_stable {
        reasons.push(WakeReason::BodyCommunicationLost);
    }
    if input.critical_battery {
        reasons.push(WakeReason::CriticalBattery);
    }
    if input.external_power_lost {
        reasons.push(WakeReason::ExternalPowerLost);
    }
    if input.direct_reign_active || input.operator_wake_request {
        reasons.push(WakeReason::DirectOperatorCommand);
    }
    if input.important_social_cue {
        reasons.push(WakeReason::ImportantSocialCue);
    }
    reasons.into_iter().max_by_key(WakeReason::priority)
}

fn deterministic_work_plan(id: &SleepSessionId, input: &SleepTickInput) -> Vec<SleepWorkItem> {
    let item = |suffix: &str,
                kind,
                dependencies: Vec<String>,
                estimate,
                locality,
                output_contract: &str,
                promotion_policy| SleepWorkItem {
        id: format!("{}:{suffix}", id.0),
        kind,
        input_artifact_refs: input
            .completed_episode_refs
            .iter()
            .chain(input.failed_behavior_refs.iter())
            .chain(input.semantic_relation_refs.iter())
            .cloned()
            .collect(),
        input_schema_versions: BTreeMap::from([
            ("experience_frame".to_string(), 1),
            ("world_model".to_string(), 2),
        ]),
        dependencies,
        estimate,
        locality,
        cancellation: WorkCancellationPolicy::Restartable,
        output_contract: output_contract.to_string(),
        verification: "stable artifact id and deterministic source refs".to_string(),
        promotion_policy,
    };
    let flush_id = format!("{}:flush", id.0);
    let consolidate_id = format!("{}:consolidate", id.0);
    let train_id = format!("{}:train", id.0);
    vec![
        item(
            "flush",
            SleepWorkKind::FlushDurableState,
            Vec::new(),
            SleepResourceEstimate {
                wall_time_ms: 50,
                cpu_time_ms: 10,
                memory_mb: 8,
                disk_growth_mb: 1,
                ..SleepResourceEstimate::default()
            },
            WorkLocality::Local,
            "durability verification report",
            CandidatePromotionPolicy::EvaluateOnly,
        ),
        item(
            "consolidate",
            SleepWorkKind::ConsolidateEpisodes,
            vec![flush_id],
            SleepResourceEstimate {
                wall_time_ms: 100,
                cpu_time_ms: 50,
                memory_mb: 32,
                disk_growth_mb: 2,
                ..SleepResourceEstimate::default()
            },
            WorkLocality::Local,
            "provenance-carrying episode index",
            CandidatePromotionPolicy::EvaluateOnly,
        ),
        item(
            "replay",
            SleepWorkKind::ReplayRecentFailures,
            vec![consolidate_id.clone()],
            SleepResourceEstimate {
                wall_time_ms: 100,
                cpu_time_ms: 75,
                memory_mb: 32,
                disk_growth_mb: 1,
                ..SleepResourceEstimate::default()
            },
            WorkLocality::Local,
            "replay artifact preserving historical event time",
            CandidatePromotionPolicy::EvaluateOnly,
        ),
        item(
            "train",
            SleepWorkKind::TrainCandidate,
            vec![consolidate_id],
            SleepResourceEstimate {
                wall_time_ms: 2_000,
                cpu_time_ms: 1_000,
                memory_mb: 256,
                disk_growth_mb: 16,
                energy_wh: 0.1,
                network_mb: 1,
            },
            WorkLocality::AcceleratorPreferred,
            "versioned candidate artifact",
            CandidatePromotionPolicy::EvaluateOnly,
        ),
        item(
            "evaluate",
            SleepWorkKind::EvaluateCandidate,
            vec![train_id],
            SleepResourceEstimate {
                wall_time_ms: 500,
                cpu_time_ms: 250,
                memory_mb: 64,
                disk_growth_mb: 2,
                ..SleepResourceEstimate::default()
            },
            WorkLocality::Local,
            "fixed-seed evaluation and promotion recommendation",
            CandidatePromotionPolicy::RecommendForShadow,
        ),
    ]
}

fn execute_work_item(
    session: &mut SleepSession,
    item: &SleepWorkItem,
    input: &SleepTickInput,
) -> SleepWorkResult {
    let dependencies_complete = item.dependencies.iter().all(|dependency| {
        session.completed.iter().any(|result| {
            &result.work_item_id == dependency && result.status == SleepWorkStatus::Completed
        })
    });
    let mut result = SleepWorkResult {
        work_item_id: item.id.clone(),
        kind: item.kind,
        status: SleepWorkStatus::Pending,
        started_at_ms: input.now_ms,
        executor: match item.locality {
            WorkLocality::Local => "organism.local".to_string(),
            WorkLocality::AcceleratorPreferred | WorkLocality::AcceleratorRequired => {
                if input.accelerator_available {
                    "cognitive_provider.mock".to_string()
                } else {
                    "deferred.no_accelerator".to_string()
                }
            }
        },
        ..SleepWorkResult::default()
    };
    if !dependencies_complete {
        result.status = SleepWorkStatus::Deferred;
        result.summary = "dependency did not complete".to_string();
        return result;
    }
    if matches!(
        item.locality,
        WorkLocality::AcceleratorPreferred | WorkLocality::AcceleratorRequired
    ) && !input.accelerator_available
    {
        result.status = SleepWorkStatus::Deferred;
        result.summary = "accelerator unavailable; useful local work remains complete".to_string();
        return result;
    }
    if !session.resource_budget.can_reserve(&item.estimate) {
        result.status = SleepWorkStatus::Cancelled;
        result.summary = "deterministic resource budget gate rejected work".to_string();
        return result;
    }
    session.resource_budget.reserve(&item.estimate);
    let artifact_id = format!("artifact:{}:{:?}", session.id.0, item.kind).to_lowercase();
    result.status = SleepWorkStatus::Completed;
    result.completed_at_ms = Some(input.now_ms);
    result.output_artifact_refs.push(artifact_id.clone());
    result.summary = match item.kind {
        SleepWorkKind::FlushDurableState => "verified pending durable state".to_string(),
        SleepWorkKind::ConsolidateEpisodes => {
            result.consolidation = Some(ConsolidationArtifact {
                artifact_id,
                source_episode_refs: input.completed_episode_refs.clone(),
                semantic_relation_refs: input.semantic_relation_refs.clone(),
                source_history_preserved: true,
                deterministic_index_entries: input.completed_episode_refs.len()
                    + input.semantic_relation_refs.len(),
            });
            format!(
                "indexed {} episode refs and {} semantic refs without replacing source history",
                input.completed_episode_refs.len(),
                input.semantic_relation_refs.len()
            )
        }
        SleepWorkKind::ReplayRecentFailures => {
            result.replay = Some(ReplayArtifact {
                artifact_id,
                source_episode_refs: input.completed_episode_refs.clone(),
                historical_time_domain: ClockDomain::Event,
                replay_computed_at: TypedTimestamp {
                    domain: ClockDomain::Replay,
                    ms: input.now_ms,
                },
                injected_into_live_now: false,
            });
            "replayed historical evidence without injecting it as current observation".to_string()
        }
        SleepWorkKind::TrainCandidate => {
            result.candidate = Some(CandidateArtifact {
                artifact_id,
                role: "goal_progress_predictor".to_string(),
                interface_version: 1,
                training_data_refs: item.input_artifact_refs.clone(),
                configuration: "deterministic_mock_teacher_v1".to_string(),
                seed: 7,
                metrics: BTreeMap::from([("training_loss".to_string(), 0.25)]),
                warnings: vec!["mock candidate requires offline evaluation".to_string()],
                baseline_comparison: "not evaluated".to_string(),
                known_failure_slices: vec!["held_out_physical_contact".to_string()],
                fallback_artifact_ref: Some("teacher:goal_progress:v1".to_string()),
                promotion_policy: CandidatePromotionPolicy::EvaluateOnly,
                promotion_recommended: false,
                automatically_promoted: false,
            });
            "produced a versioned candidate without promotion authority".to_string()
        }
        SleepWorkKind::EvaluateCandidate => {
            let trained = session.completed.iter().find_map(|completed| {
                (completed.kind == SleepWorkKind::TrainCandidate)
                    .then_some(completed.candidate.as_ref())
                    .flatten()
            });
            result.candidate = trained.cloned().map(|mut candidate| {
                candidate
                    .metrics
                    .insert("fixed_seed_score".to_string(), 0.72);
                candidate.baseline_comparison = "candidate 0.72 vs teacher 0.75".to_string();
                candidate.promotion_recommended = false;
                candidate.automatically_promoted = false;
                candidate
            });
            "evaluated candidate; current promoted implementation is unchanged".to_string()
        }
        SleepWorkKind::RebuildIndexes
        | SleepWorkKind::DryRunPruning
        | SleepWorkKind::SummarizeChanges => "completed bounded maintenance".to_string(),
    };
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn safe_input(now_ms: u64) -> SleepTickInput {
        SleepTickInput {
            now_ms,
            fatigue_activation: 0.9,
            charging: true,
            docked: true,
            stopped: true,
            body_communication_stable: true,
            active_skill_interruptible: true,
            accelerator_available: true,
            completed_episode_refs: vec!["episode:charging:1".to_string()],
            failed_behavior_refs: vec!["behavior:approach_charger:failed:1".to_string()],
            ..SleepTickInput::default()
        }
    }

    fn run_to_wake(controller: &mut SleepController, start_ms: u64, accelerator: bool) {
        for offset in 0..8 {
            let mut input = safe_input(start_ms + offset);
            input.accelerator_available = accelerator;
            if offset > 0 {
                input.fatigue_activation = 0.0;
            }
            controller.tick(input);
        }
    }

    #[test]
    fn high_fatigue_while_safely_docked_enters_sleep() {
        let mut controller = SleepController::default();
        let snapshot = controller.tick(safe_input(100));
        assert_eq!(snapshot.phase, SleepPhase::Preparing);
        assert!(snapshot.eligibility.eligible);
        assert!(controller.requires_quiescence());
    }

    #[test]
    fn direct_control_and_unsafe_body_block_entry() {
        let mut input = safe_input(100);
        input.direct_reign_active = true;
        input.safety_event = Some("wheel_drop".to_string());
        let eligibility = sleep_eligibility(&input);
        assert!(!eligibility.eligible);
        assert!(eligibility
            .blocking_reasons
            .iter()
            .any(|reason| reason.contains("Direct Reign")));
        assert!(eligibility
            .blocking_reasons
            .iter()
            .any(|reason| reason.contains("safety")));
    }

    #[test]
    fn safety_event_interrupts_sleep_with_highest_priority() {
        let mut controller = SleepController::default();
        controller.tick(safe_input(100));
        let mut interrupted = safe_input(101);
        interrupted.direct_reign_active = true;
        interrupted.safety_event = Some("cliff".to_string());
        let snapshot = controller.tick(interrupted);
        assert_eq!(snapshot.phase, SleepPhase::Interrupted);
        assert_eq!(
            snapshot.session.unwrap().interrupted_by.unwrap(),
            WakeReason::SafetyEvent("cliff".to_string())
        );
    }

    #[test]
    fn no_accelerator_defers_training_but_completes_local_maintenance() {
        let mut controller = SleepController::default();
        run_to_wake(&mut controller, 100, false);
        let report = controller.last_report.as_ref().unwrap();
        assert!(report.completed.iter().any(|result| {
            result.kind == SleepWorkKind::ConsolidateEpisodes
                && result.status == SleepWorkStatus::Completed
                && result
                    .consolidation
                    .as_ref()
                    .is_some_and(|artifact| artifact.source_history_preserved)
        }));
        assert!(report.completed.iter().any(|result| {
            result.kind == SleepWorkKind::TrainCandidate
                && result.status == SleepWorkStatus::Deferred
        }));
    }

    #[test]
    fn replay_preserves_historical_time_and_never_enters_live_now() {
        let mut controller = SleepController::default();
        run_to_wake(&mut controller, 100, true);
        let replay = controller
            .last_report
            .as_ref()
            .unwrap()
            .completed
            .iter()
            .find_map(|result| result.replay.as_ref())
            .unwrap();
        assert_eq!(replay.historical_time_domain, ClockDomain::Event);
        assert_eq!(replay.replay_computed_at.domain, ClockDomain::Replay);
        assert!(!replay.injected_into_live_now);
    }

    #[test]
    fn sleep_evaluates_candidate_without_automatic_promotion() {
        let mut controller = SleepController::default();
        run_to_wake(&mut controller, 100, true);
        let report = controller.last_report.as_ref().unwrap();
        let candidates = report
            .completed
            .iter()
            .filter_map(|result| result.candidate.as_ref())
            .collect::<Vec<_>>();
        assert!(!candidates.is_empty());
        assert!(candidates
            .iter()
            .all(|candidate| !candidate.automatically_promoted));
        assert!(report.promoted_artifact.is_none());
        assert!(report.fresh_world_model_required);
        assert!(!report.stale_skill_resumed);
    }

    #[test]
    fn resource_budget_cancellation_is_deterministic() {
        let mut controller = SleepController::default();
        controller.tick(safe_input(100));
        controller.session_mut().resource_budget.cpu_time_ms = 0;
        controller.tick(SleepTickInput {
            now_ms: 101,
            body_communication_stable: true,
            active_skill_interruptible: true,
            ..SleepTickInput::default()
        });
        controller.tick(SleepTickInput {
            now_ms: 102,
            body_communication_stable: true,
            active_skill_interruptible: true,
            ..SleepTickInput::default()
        });
        controller.tick(SleepTickInput {
            now_ms: 103,
            body_communication_stable: true,
            active_skill_interruptible: true,
            ..SleepTickInput::default()
        });
        assert!(controller
            .session
            .as_ref()
            .unwrap()
            .completed
            .iter()
            .any(|result| result.status == SleepWorkStatus::Cancelled));
    }
}
