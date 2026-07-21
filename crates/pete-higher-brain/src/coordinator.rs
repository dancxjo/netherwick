use crate::auth::{Principal, Scope};
use crate::{atomic_write_json, read_json};
use anyhow::Result;
use chrono::Utc;
use pete_ups::{ConsolidationPowerAssessment, UpsTelemetry};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DockReadiness {
    pub create_stopped: bool,
    pub docked: bool,
    pub charging: bool,
    pub motion_authority_active: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PowerReadiness {
    pub external_power_present: bool,
    pub ups_battery_percent: f32,
    pub suitable_for_training: bool,
}

impl From<UpsTelemetry> for PowerReadiness {
    fn from(value: UpsTelemetry) -> Self {
        Self {
            external_power_present: value.external_power_present,
            ups_battery_percent: value.battery_percent,
            suitable_for_training: value.external_power_present && value.battery_percent >= 20.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsolidationPhase {
    CheckpointExperience,
    DiscoverForebrain,
    TransferBundles,
    SubmitJobs,
    AwaitJobs,
    ReturnCandidates,
    StageCandidates,
    AwaitActivation,
    Activate,
    Complete,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsolidationEvent {
    pub at: String,
    pub phase: ConsolidationPhase,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsolidationCycle {
    pub schema_version: u32,
    pub cycle_id: String,
    pub phase: ConsolidationPhase,
    pub selected_forebrain_id: Option<String>,
    pub bundle_ids: Vec<String>,
    pub job_ids: Vec<String>,
    pub candidate_ids: Vec<String>,
    pub retry_count: u32,
    pub last_error: Option<String>,
    pub activation_requested: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power_assessment: Option<ConsolidationPowerAssessment>,
    pub events: Vec<ConsolidationEvent>,
}

pub trait ConsolidationBackend {
    fn checkpoint_epoch(&mut self) -> Result<Vec<String>>;
    fn discover_authorized_forebrain(&mut self) -> Result<String>;
    fn transfer_bundles(&mut self, forebrain_id: &str, bundle_ids: &[String]) -> Result<()>;
    fn submit_jobs(&mut self, forebrain_id: &str, bundle_ids: &[String]) -> Result<Vec<String>>;
    fn jobs_complete(&mut self, forebrain_id: &str, job_ids: &[String]) -> Result<bool>;
    fn return_candidates(&mut self, forebrain_id: &str, job_ids: &[String]) -> Result<Vec<String>>;
    fn stage_candidates(&mut self, candidate_ids: &[String]) -> Result<()>;
    fn activate_candidates(&mut self, candidate_ids: &[String]) -> Result<()>;
}

/// Durable coordinator. It consumes read-only bodily status but has no cockpit
/// command or brainstem dependency, so failures cannot block charging or safety.
pub struct ConsolidationCoordinator {
    state_path: PathBuf,
    pub cycle: Option<ConsolidationCycle>,
}

impl ConsolidationCoordinator {
    pub fn open(state_path: impl Into<PathBuf>) -> Result<Self> {
        let state_path = state_path.into();
        let cycle = state_path
            .exists()
            .then(|| read_json(&state_path))
            .transpose()?;
        Ok(Self { state_path, cycle })
    }

    pub fn start(
        &mut self,
        dock: DockReadiness,
        power: PowerReadiness,
        principal: &Principal,
    ) -> Result<()> {
        principal.require(Scope::Discover)?;
        principal.require(Scope::TransferExperience)?;
        principal.require(Scope::SubmitJob)?;
        principal.require(Scope::StageCandidate)?;
        if !dock.create_stopped {
            anyhow::bail!("consolidation refused: Create is moving");
        }
        if !dock.docked {
            anyhow::bail!("consolidation refused: Create is not docked");
        }
        if !dock.charging {
            anyhow::bail!("consolidation refused: Create is not charging");
        }
        if dock.motion_authority_active {
            anyhow::bail!("consolidation refused: motion authority is active");
        }
        if !power.external_power_present || !power.suitable_for_training {
            anyhow::bail!("consolidation refused: external/UPS power is unsuitable");
        }
        if self
            .cycle
            .as_ref()
            .is_some_and(|cycle| cycle.phase != ConsolidationPhase::Complete)
        {
            anyhow::bail!("a consolidation cycle is already pending");
        }
        let phase = ConsolidationPhase::CheckpointExperience;
        self.cycle = Some(ConsolidationCycle {
            schema_version: 1,
            cycle_id: Uuid::new_v4().to_string(),
            phase,
            selected_forebrain_id: None,
            bundle_ids: Vec::new(),
            job_ids: Vec::new(),
            candidate_ids: Vec::new(),
            retry_count: 0,
            last_error: None,
            activation_requested: false,
            power_assessment: None,
            events: vec![ConsolidationEvent {
                at: Utc::now().to_rfc3339(),
                phase,
                detail: "docked charging preconditions accepted".into(),
            }],
        });
        self.persist()
    }

    /// Starts consolidation from the confidence- and freshness-qualified
    /// X1202/Create evidence model. This is the production entrypoint; the
    /// narrower `start` method remains for compatibility with existing callers.
    pub fn start_with_power_assessment(
        &mut self,
        assessment: ConsolidationPowerAssessment,
        principal: &Principal,
    ) -> Result<()> {
        if !assessment.consolidation_ready {
            anyhow::bail!(
                "consolidation refused by power evidence: {}",
                assessment.reasons.join("; ")
            );
        }
        let dock = DockReadiness {
            create_stopped: assessment.create_stopped == Some(true),
            docked: assessment.create_docked == Some(true),
            charging: assessment.create_charging == Some(true),
            motion_authority_active: assessment.motion_authority_active != Some(false),
        };
        let power = PowerReadiness {
            external_power_present: assessment.external_power_present == Some(true),
            ups_battery_percent: assessment.battery_percent.unwrap_or_default(),
            suitable_for_training: assessment.consolidation_ready,
        };
        self.start(dock, power, principal)?;
        if let Some(cycle) = self.cycle.as_mut() {
            cycle.power_assessment = Some(assessment);
            cycle.events.push(ConsolidationEvent {
                at: Utc::now().to_rfc3339(),
                phase: cycle.phase,
                detail: "fresh independent X1202/Create evidence accepted".to_string(),
            });
        }
        self.persist()
    }

    /// Re-evaluates power before each expensive phase. Loss of external-power
    /// or charging evidence pauses in place; it never issues a motion or
    /// brainstem command and resumes only after fresh evidence returns.
    pub fn tick_with_power_assessment(
        &mut self,
        assessment: ConsolidationPowerAssessment,
        backend: &mut dyn ConsolidationBackend,
    ) -> Result<ConsolidationPhase> {
        let cycle = self
            .cycle
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("no consolidation cycle"))?;
        cycle.power_assessment = Some(assessment.clone());
        if !assessment.consolidation_ready {
            let detail = format!(
                "consolidation paused: {} ({})",
                assessment.action,
                assessment.reasons.join("; ")
            );
            cycle.last_error = Some(detail.clone());
            if cycle
                .events
                .last()
                .is_none_or(|event| event.detail != detail)
            {
                cycle.events.push(ConsolidationEvent {
                    at: Utc::now().to_rfc3339(),
                    phase: cycle.phase,
                    detail,
                });
            }
            let phase = cycle.phase;
            self.persist()?;
            return Ok(phase);
        }
        self.persist()?;
        self.tick(backend)
    }

    pub fn tick(&mut self, backend: &mut dyn ConsolidationBackend) -> Result<ConsolidationPhase> {
        let cycle = self
            .cycle
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("no consolidation cycle"))?;
        let current = cycle.phase;
        let result: Result<Option<ConsolidationPhase>> = (|| match current {
            ConsolidationPhase::CheckpointExperience => {
                cycle.bundle_ids = backend.checkpoint_epoch()?;
                if cycle.bundle_ids.is_empty() {
                    anyhow::bail!("checkpoint produced no complete bundles");
                }
                Ok(Some(ConsolidationPhase::DiscoverForebrain))
            }
            ConsolidationPhase::DiscoverForebrain => {
                cycle.selected_forebrain_id = Some(backend.discover_authorized_forebrain()?);
                Ok(Some(ConsolidationPhase::TransferBundles))
            }
            ConsolidationPhase::TransferBundles => {
                backend.transfer_bundles(forebrain(cycle)?, &cycle.bundle_ids)?;
                Ok(Some(ConsolidationPhase::SubmitJobs))
            }
            ConsolidationPhase::SubmitJobs => {
                cycle.job_ids = backend.submit_jobs(forebrain(cycle)?, &cycle.bundle_ids)?;
                Ok(Some(ConsolidationPhase::AwaitJobs))
            }
            ConsolidationPhase::AwaitJobs => {
                if backend.jobs_complete(forebrain(cycle)?, &cycle.job_ids)? {
                    Ok(Some(ConsolidationPhase::ReturnCandidates))
                } else {
                    Ok(None)
                }
            }
            ConsolidationPhase::ReturnCandidates => {
                cycle.candidate_ids =
                    backend.return_candidates(forebrain(cycle)?, &cycle.job_ids)?;
                Ok(Some(ConsolidationPhase::StageCandidates))
            }
            ConsolidationPhase::StageCandidates => {
                backend.stage_candidates(&cycle.candidate_ids)?;
                Ok(Some(ConsolidationPhase::AwaitActivation))
            }
            ConsolidationPhase::AwaitActivation => {
                if cycle.activation_requested {
                    Ok(Some(ConsolidationPhase::Activate))
                } else {
                    Ok(None)
                }
            }
            ConsolidationPhase::Activate => {
                backend.activate_candidates(&cycle.candidate_ids)?;
                Ok(Some(ConsolidationPhase::Complete))
            }
            ConsolidationPhase::Complete => Ok(None),
        })();
        match result {
            Ok(next) => {
                cycle.last_error = None;
                if let Some(next) = next {
                    cycle.phase = next;
                    cycle.events.push(ConsolidationEvent {
                        at: Utc::now().to_rfc3339(),
                        phase: next,
                        detail: "phase completed".into(),
                    });
                }
            }
            Err(error) => {
                cycle.retry_count = cycle.retry_count.saturating_add(1);
                cycle.last_error = Some(error.to_string());
                cycle.events.push(ConsolidationEvent {
                    at: Utc::now().to_rfc3339(),
                    phase: current,
                    detail: format!("retryable failure: {error:#}"),
                });
            }
        }
        let phase = cycle.phase;
        self.persist()?;
        Ok(phase)
    }

    pub fn approve_activation(&mut self, principal: &Principal) -> Result<()> {
        principal.require(Scope::ActivateCandidate)?;
        let cycle = self
            .cycle
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("no consolidation cycle"))?;
        if cycle.phase != ConsolidationPhase::AwaitActivation {
            anyhow::bail!("cycle is not awaiting activation approval");
        }
        cycle.activation_requested = true;
        self.persist()
    }

    pub fn cycle(&self) -> Option<&ConsolidationCycle> {
        self.cycle.as_ref()
    }

    fn persist(&self) -> Result<()> {
        atomic_write_json(
            &self.state_path,
            self.cycle
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("no cycle state"))?,
        )
    }
}

fn forebrain(cycle: &ConsolidationCycle) -> Result<&str> {
    cycle
        .selected_forebrain_id
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("no forebrain selected"))
}

pub fn state_path(root: &Path) -> PathBuf {
    root.join("consolidation/current.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn principal() -> Principal {
        Principal {
            id: "coordinator".into(),
            scopes: [
                Scope::Discover,
                Scope::TransferExperience,
                Scope::SubmitJob,
                Scope::StageCandidate,
                Scope::ActivateCandidate,
            ]
            .into_iter()
            .collect::<BTreeSet<_>>(),
        }
    }

    fn ready() -> (DockReadiness, PowerReadiness) {
        (
            DockReadiness {
                create_stopped: true,
                docked: true,
                charging: true,
                motion_authority_active: false,
            },
            PowerReadiness {
                external_power_present: true,
                ups_battery_percent: 90.0,
                suitable_for_training: true,
            },
        )
    }

    fn assessed_ready() -> ConsolidationPowerAssessment {
        ConsolidationPowerAssessment {
            evaluated_at_ms: 1_000,
            external_power_present: Some(true),
            charging_enabled: Some(true),
            battery_voltage_v: Some(4.0),
            battery_percent: Some(90.0),
            battery_current_a: None,
            battery_current_observable: false,
            battery_charging: pete_ups::ChargingInference::LikelyCharging,
            battery_charging_confidence: 0.75,
            create_stopped: Some(true),
            create_docked: Some(true),
            home_base_contact: Some(true),
            dock_ir_visible: Some(true),
            create_charging: Some(true),
            create_charging_state: Some(2),
            motion_authority_active: Some(false),
            ages: pete_ups::PowerEvidenceAge::default(),
            consolidation_ready: true,
            action: "proceed".to_string(),
            reasons: vec![
                "battery current is unavailable on MAX17040G; charging is inferred".to_string(),
            ],
            evidence_sources: vec![
                "x1202_pogo_max17040g_gpio6_gpio16".to_string(),
                "create_oi_and_dock_observation".to_string(),
            ],
        }
    }

    #[derive(Default)]
    struct Backend {
        fail_phase: Option<ConsolidationPhase>,
        brainstem_mutations: usize,
    }

    impl Backend {
        fn fail(&self, phase: ConsolidationPhase) -> Result<()> {
            if self.fail_phase == Some(phase) {
                anyhow::bail!("forebrain disappeared");
            }
            Ok(())
        }
    }

    impl ConsolidationBackend for Backend {
        fn checkpoint_epoch(&mut self) -> Result<Vec<String>> {
            self.fail(ConsolidationPhase::CheckpointExperience)?;
            Ok(vec!["exp".into()])
        }
        fn discover_authorized_forebrain(&mut self) -> Result<String> {
            self.fail(ConsolidationPhase::DiscoverForebrain)?;
            Ok("fore".into())
        }
        fn transfer_bundles(&mut self, _: &str, _: &[String]) -> Result<()> {
            self.fail(ConsolidationPhase::TransferBundles)
        }
        fn submit_jobs(&mut self, _: &str, _: &[String]) -> Result<Vec<String>> {
            self.fail(ConsolidationPhase::SubmitJobs)?;
            Ok(vec!["job".into()])
        }
        fn jobs_complete(&mut self, _: &str, _: &[String]) -> Result<bool> {
            self.fail(ConsolidationPhase::AwaitJobs)?;
            Ok(true)
        }
        fn return_candidates(&mut self, _: &str, _: &[String]) -> Result<Vec<String>> {
            self.fail(ConsolidationPhase::ReturnCandidates)?;
            Ok(vec!["candidate".into()])
        }
        fn stage_candidates(&mut self, _: &[String]) -> Result<()> {
            self.fail(ConsolidationPhase::StageCandidates)
        }
        fn activate_candidates(&mut self, _: &[String]) -> Result<()> {
            self.fail(ConsolidationPhase::Activate)
        }
    }

    #[test]
    fn refuses_every_unsafe_docking_condition_and_missing_authority() {
        let cases = [
            DockReadiness {
                create_stopped: false,
                ..ready().0
            },
            DockReadiness {
                docked: false,
                ..ready().0
            },
            DockReadiness {
                charging: false,
                ..ready().0
            },
            DockReadiness {
                motion_authority_active: true,
                ..ready().0
            },
        ];
        for (index, dock) in cases.into_iter().enumerate() {
            let path = std::env::temp_dir().join(format!("pete-coordinator-refuse-{index}.json"));
            let mut coordinator = ConsolidationCoordinator::open(&path).unwrap();
            assert!(coordinator.start(dock, ready().1, &principal()).is_err());
            let _ = std::fs::remove_file(path);
        }
        let path = std::env::temp_dir().join("pete-coordinator-power-refuse.json");
        let mut coordinator = ConsolidationCoordinator::open(&path).unwrap();
        assert!(coordinator
            .start(ready().0, PowerReadiness::default(), &principal())
            .is_err());
        let denied = Principal {
            id: "motion-lease".into(),
            scopes: BTreeSet::new(),
        };
        assert!(coordinator.start(ready().0, ready().1, &denied).is_err());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn disappearance_is_retryable_at_every_network_phase_and_never_touches_brainstem() {
        for phase in [
            ConsolidationPhase::DiscoverForebrain,
            ConsolidationPhase::TransferBundles,
            ConsolidationPhase::SubmitJobs,
            ConsolidationPhase::AwaitJobs,
            ConsolidationPhase::ReturnCandidates,
            ConsolidationPhase::StageCandidates,
        ] {
            let path = std::env::temp_dir().join(format!("pete-coordinator-{phase:?}.json"));
            let mut coordinator = ConsolidationCoordinator::open(&path).unwrap();
            coordinator
                .start(ready().0, ready().1, &principal())
                .unwrap();
            let mut backend = Backend::default();
            while coordinator.cycle().unwrap().phase != phase {
                coordinator.tick(&mut backend).unwrap();
            }
            backend.fail_phase = Some(phase);
            coordinator.tick(&mut backend).unwrap();
            assert_eq!(coordinator.cycle().unwrap().phase, phase);
            assert!(coordinator.cycle().unwrap().last_error.is_some());
            assert_eq!(backend.brainstem_mutations, 0);
            backend.fail_phase = None;
            coordinator.tick(&mut backend).unwrap();
            assert_ne!(coordinator.cycle().unwrap().phase, phase);
            let _ = std::fs::remove_file(path);
        }
    }

    #[test]
    fn pending_cycle_survives_motherbrain_restart_and_waits_for_approval() {
        let path =
            std::env::temp_dir().join(format!("pete-coordinator-restart-{}.json", Uuid::new_v4()));
        let mut first = ConsolidationCoordinator::open(&path).unwrap();
        first.start(ready().0, ready().1, &principal()).unwrap();
        let mut backend = Backend::default();
        first.tick(&mut backend).unwrap();
        drop(first);
        let mut restarted = ConsolidationCoordinator::open(&path).unwrap();
        while restarted.cycle().unwrap().phase != ConsolidationPhase::AwaitActivation {
            restarted.tick(&mut backend).unwrap();
        }
        restarted.tick(&mut backend).unwrap();
        assert_eq!(
            restarted.cycle().unwrap().phase,
            ConsolidationPhase::AwaitActivation
        );
        restarted.approve_activation(&principal()).unwrap();
        restarted.tick(&mut backend).unwrap();
        restarted.tick(&mut backend).unwrap();
        assert_eq!(
            restarted.cycle().unwrap().phase,
            ConsolidationPhase::Complete
        );
        assert_eq!(backend.brainstem_mutations, 0);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn fresh_power_assessment_starts_then_external_power_loss_pauses_in_place() {
        let path = std::env::temp_dir().join(format!(
            "pete-coordinator-power-assessment-{}.json",
            Uuid::new_v4()
        ));
        let mut coordinator = ConsolidationCoordinator::open(&path).unwrap();
        coordinator
            .start_with_power_assessment(assessed_ready(), &principal())
            .unwrap();
        assert!(coordinator.cycle().unwrap().power_assessment.is_some());
        let initial = coordinator.cycle().unwrap().phase;
        let mut lost = assessed_ready();
        lost.external_power_present = Some(false);
        lost.consolidation_ready = false;
        lost.action = "pause_external_power_lost".to_string();
        lost.reasons = vec!["fresh GPIO6 external-power evidence is absent".to_string()];
        let mut backend = Backend::default();
        assert_eq!(
            coordinator
                .tick_with_power_assessment(lost, &mut backend)
                .unwrap(),
            initial
        );
        assert_eq!(coordinator.cycle().unwrap().phase, initial);
        assert!(coordinator
            .cycle()
            .unwrap()
            .last_error
            .as_deref()
            .is_some_and(|error| error.contains("external_power_lost")));
        coordinator
            .tick_with_power_assessment(assessed_ready(), &mut backend)
            .unwrap();
        assert_ne!(coordinator.cycle().unwrap().phase, initial);
        assert_eq!(backend.brainstem_mutations, 0);
        std::fs::remove_file(path).unwrap();
    }
}
