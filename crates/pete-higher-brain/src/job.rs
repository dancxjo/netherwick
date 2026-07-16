use crate::auth::{Principal, Scope};
use crate::bundle::validate_bundle;
use crate::candidate::{create_candidate, CandidateRequest};
use crate::capability::AcceleratorCapabilities;
use crate::{atomic_write_json, canonical_json, read_json, sha256_bytes, JOB_SCHEMA_VERSION};
use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobClass {
    FixtureDigest,
    DatasetConstruction,
    LongHistoryReplay,
    GraphAnalysis,
    RepresentationTraining,
    PerceptionTraining,
    FineTuning,
    HyperparameterSearch,
    Consolidation,
    FullEvaluation,
}

impl JobClass {
    pub fn capability_name(&self) -> &'static str {
        match self {
            Self::FixtureDigest => "fixture_digest",
            Self::DatasetConstruction => "dataset_construction",
            Self::LongHistoryReplay => "replay",
            Self::GraphAnalysis => "graph_analysis",
            Self::RepresentationTraining => "representation_training",
            Self::PerceptionTraining => "perception_training",
            Self::FineTuning => "fine_tuning",
            Self::HyperparameterSearch => "hyperparameter_search",
            Self::Consolidation => "consolidation",
            Self::FullEvaluation => "evaluation",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_cores: usize,
    pub ram_bytes: u64,
    pub gpu_memory_bytes: u64,
    pub workspace_bytes: u64,
    pub required_runtimes: BTreeSet<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JobEnvelope {
    pub schema_version: u32,
    pub job_id: String,
    pub job_class: JobClass,
    pub source_experience_bundle_ids: Vec<String>,
    pub resources: ResourceRequirements,
    pub parameters: BTreeMap<String, Value>,
    pub submitter_node_id: String,
    pub software_identity: String,
}

impl JobEnvelope {
    pub fn deterministic(
        job_class: JobClass,
        source_experience_bundle_ids: Vec<String>,
        resources: ResourceRequirements,
        parameters: BTreeMap<String, Value>,
        submitter_node_id: impl Into<String>,
        software_identity: impl Into<String>,
    ) -> Result<Self> {
        let submitter_node_id = submitter_node_id.into();
        let software_identity = software_identity.into();
        let identity = sha256_bytes(&canonical_json(&(
            &job_class,
            &source_experience_bundle_ids,
            &resources,
            &parameters,
            &submitter_node_id,
            &software_identity,
        ))?);
        Ok(Self {
            schema_version: JOB_SCHEMA_VERSION,
            job_id: format!("job-{}", &identity[..32]),
            job_class,
            source_experience_bundle_ids,
            resources,
            parameters,
            submitter_node_id,
            software_identity,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
    Interrupted,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobTransition {
    pub status: JobStatus,
    pub at: String,
    pub progress_percent: u8,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DurableJob {
    pub envelope: JobEnvelope,
    pub status: JobStatus,
    pub progress_percent: u8,
    pub detail: String,
    pub candidate_id: Option<String>,
    pub transitions: Vec<JobTransition>,
}

impl DurableJob {
    fn new(envelope: JobEnvelope) -> Self {
        let mut job = Self {
            envelope,
            status: JobStatus::Queued,
            progress_percent: 0,
            detail: "queued".into(),
            candidate_id: None,
            transitions: Vec::new(),
        };
        job.transition(JobStatus::Queued, 0, "queued");
        job
    }

    fn transition(&mut self, status: JobStatus, progress: u8, detail: impl Into<String>) {
        self.status = status;
        self.progress_percent = progress.min(100);
        self.detail = detail.into();
        self.transitions.push(JobTransition {
            status,
            at: Utc::now().to_rfc3339(),
            progress_percent: self.progress_percent,
            detail: self.detail.clone(),
        });
    }
}

#[derive(Clone, Debug)]
pub struct WorkerPaths {
    pub bundles: PathBuf,
    pub jobs: PathBuf,
    pub workspaces: PathBuf,
    pub candidates: PathBuf,
}

pub struct ForebrainWorker {
    pub paths: WorkerPaths,
    pub capabilities: AcceleratorCapabilities,
}

impl ForebrainWorker {
    pub fn open(paths: WorkerPaths, capabilities: AcceleratorCapabilities) -> Result<Self> {
        for path in [
            &paths.bundles,
            &paths.jobs,
            &paths.workspaces,
            &paths.candidates,
        ] {
            fs::create_dir_all(path)?;
        }
        let worker = Self {
            paths,
            capabilities,
        };
        worker.recover_interrupted()?;
        Ok(worker)
    }

    pub fn enqueue(&self, envelope: JobEnvelope, principal: &Principal) -> Result<DurableJob> {
        principal.require(Scope::SubmitJob)?;
        self.validate_envelope(&envelope)?;
        let path = self.job_path(&envelope.job_id);
        if path.exists() {
            let existing: DurableJob = read_json(&path)?;
            if existing.envelope != envelope {
                anyhow::bail!("job id already exists with a different envelope");
            }
            return Ok(existing);
        }
        let job = DurableJob::new(envelope);
        atomic_write_json(&path, &job)?;
        Ok(job)
    }

    pub fn cancel(&self, job_id: &str, principal: &Principal) -> Result<()> {
        principal.require(Scope::CancelJob)?;
        let mut job = self.load(job_id)?;
        if matches!(
            job.status,
            JobStatus::Succeeded | JobStatus::Failed | JobStatus::Cancelled
        ) {
            anyhow::bail!("job is already terminal");
        }
        job.transition(
            JobStatus::Cancelled,
            job.progress_percent,
            "cancelled by authorized request",
        );
        self.save(&job)
    }

    pub fn retry_interrupted(&self, job_id: &str, principal: &Principal) -> Result<()> {
        principal.require(Scope::SubmitJob)?;
        let mut job = self.load(job_id)?;
        if job.status != JobStatus::Interrupted && job.status != JobStatus::Failed {
            anyhow::bail!("only interrupted or failed jobs may be retried");
        }
        job.transition(JobStatus::Queued, 0, "retry queued");
        self.save(&job)
    }

    pub fn run_once(&self) -> Result<Option<DurableJob>> {
        self.ingest_incoming()?;
        let mut queued = fs::read_dir(&self.paths.jobs)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("json"))
            .filter_map(|path| read_json::<DurableJob>(&path).ok())
            .filter(|job| job.status == JobStatus::Queued)
            .collect::<Vec<_>>();
        queued.sort_by(|left, right| left.envelope.job_id.cmp(&right.envelope.job_id));
        let Some(mut job) = queued.into_iter().next() else {
            return Ok(None);
        };
        job.transition(JobStatus::Running, 5, "validating source bundles");
        self.save(&job)?;
        let result = self.execute(&mut job);
        match result {
            Ok(candidate_id) => {
                job.candidate_id = Some(candidate_id);
                job.transition(JobStatus::Succeeded, 100, "candidate produced and verified");
            }
            Err(error) => {
                job.transition(
                    JobStatus::Failed,
                    job.progress_percent,
                    format!("{error:#}"),
                );
            }
        }
        self.save(&job)?;
        Ok(Some(job))
    }

    pub fn load(&self, job_id: &str) -> Result<DurableJob> {
        read_json(&self.job_path(job_id))
    }

    fn execute(&self, job: &mut DurableJob) -> Result<String> {
        self.validate_envelope(&job.envelope)?;
        let mut manifests = Vec::new();
        for id in &job.envelope.source_experience_bundle_ids {
            let path = self.paths.bundles.join(format!("{id}.bundle"));
            let manifest = validate_bundle(&path)?;
            if manifest.bundle_id != *id {
                anyhow::bail!("bundle directory and manifest identity differ");
            }
            manifests.push(manifest);
        }
        job.transition(JobStatus::Running, 35, "source bundles verified");
        self.save(job)?;
        match job.envelope.job_class {
            JobClass::FixtureDigest => self.run_fixture_digest(job, &manifests),
            _ => anyhow::bail!(
                "job class {} is declared but no trainer adapter is configured",
                job.envelope.job_class.capability_name()
            ),
        }
    }

    fn run_fixture_digest(
        &self,
        job: &mut DurableJob,
        manifests: &[crate::bundle::ExperienceBundleManifest],
    ) -> Result<String> {
        let workspace = self.paths.workspaces.join(&job.envelope.job_id);
        fs::create_dir_all(&workspace)?;
        let total_files = manifests
            .iter()
            .map(|manifest| manifest.files.len())
            .sum::<usize>();
        let total_bytes = manifests
            .iter()
            .flat_map(|manifest| &manifest.files)
            .map(|file| file.bytes)
            .sum::<u64>();
        let source_digest = sha256_bytes(&canonical_json(
            &manifests
                .iter()
                .flat_map(|manifest| manifest.files.iter().map(|file| &file.sha256))
                .collect::<Vec<_>>(),
        )?);
        let artifact = workspace.join("fixture-model.json");
        let learned = json!({
            "schema_version": 1,
            "algorithm": "fixture_digest_v1",
            "source_digest": source_digest,
            "source_bundle_count": manifests.len(),
            "source_file_count": total_files,
            "source_bytes": total_bytes,
        });
        fs::write(&artifact, canonical_json(&learned)?)?;
        job.transition(
            JobStatus::Running,
            80,
            "deterministic fixture artifact trained",
        );
        self.save(job)?;
        let candidate = create_candidate(
            &self.paths.candidates,
            &CandidateRequest {
                algorithm_family: "fixture_digest_v1".into(),
                preprocessing_version: "fixture_digest/1".into(),
                input_schema_version: "experience_bundle/1".into(),
                output_schema_version: "fixture_model/1".into(),
                training_build_identity: job.envelope.software_identity.clone(),
                source_experience_bundle_ids: job.envelope.source_experience_bundle_ids.clone(),
                training_parameters: job.envelope.parameters.clone(),
                evaluation_results: [
                    ("deterministic".into(), Value::Bool(true)),
                    ("source_files".into(), json!(total_files)),
                    ("source_bytes".into(), json!(total_bytes)),
                ]
                .into_iter()
                .collect(),
                hardware_requirements: BTreeMap::new(),
                runtime_requirements: BTreeSet::new(),
                intended_deployment_target: "motherbrain_fixture".into(),
                rollback_compatibility: [
                    ("atomic_symlink_activation".into(), Value::Bool(true)),
                    ("previous_candidate_retained".into(), Value::Bool(true)),
                ]
                .into_iter()
                .collect(),
            },
            &[(artifact, PathBuf::from("fixture-model.json"))],
        )?;
        Ok(crate::candidate::validate_candidate(&candidate, None)?.candidate_id)
    }

    fn validate_envelope(&self, envelope: &JobEnvelope) -> Result<()> {
        if envelope.schema_version != JOB_SCHEMA_VERSION {
            anyhow::bail!("incompatible job schema {}", envelope.schema_version);
        }
        if !self
            .capabilities
            .job_classes
            .contains(envelope.job_class.capability_name())
        {
            anyhow::bail!("forebrain does not advertise requested job class");
        }
        let resources = &envelope.resources;
        if resources.cpu_cores > self.capabilities.cpu_cores
            || resources.ram_bytes > self.capabilities.available_ram_bytes
            || resources.workspace_bytes > self.capabilities.storage_available_bytes
            || resources.gpu_memory_bytes
                > self
                    .capabilities
                    .gpu
                    .as_ref()
                    .map(|gpu| gpu.usable_memory_bytes)
                    .unwrap_or(0)
            || !resources
                .required_runtimes
                .iter()
                .all(|runtime| self.capabilities.runtimes.contains(runtime))
        {
            anyhow::bail!("forebrain does not satisfy declared resource requirements");
        }
        Ok(())
    }

    /// Envelopes arrive through the restricted SFTP inbox. Renaming them to
    /// `.accepted`/`.rejected` is an audit trail outside the immutable request.
    fn ingest_incoming(&self) -> Result<()> {
        let incoming = self.paths.jobs.join("incoming");
        fs::create_dir_all(&incoming)?;
        let principal = Principal {
            id: "restricted-sftp-inbox".into(),
            scopes: [Scope::SubmitJob].into_iter().collect(),
        };
        for entry in fs::read_dir(&incoming)? {
            let path = entry?.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let outcome = match read_json::<JobEnvelope>(&path)
                .and_then(|envelope| self.enqueue(envelope, &principal).map(|_| ()))
            {
                Ok(()) => "accepted",
                Err(error) => {
                    let error_path = path.with_extension("rejected.txt");
                    fs::write(error_path, format!("{error:#}\n"))?;
                    "rejected"
                }
            };
            fs::rename(&path, path.with_extension(outcome))?;
        }
        Ok(())
    }

    fn recover_interrupted(&self) -> Result<()> {
        for entry in fs::read_dir(&self.paths.jobs)? {
            let path = entry?.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let mut job: DurableJob = read_json(&path)?;
            if job.status == JobStatus::Running {
                job.transition(
                    JobStatus::Interrupted,
                    job.progress_percent,
                    "worker restarted while job was running",
                );
                atomic_write_json(&path, &job)?;
            }
        }
        Ok(())
    }

    fn save(&self, job: &DurableJob) -> Result<()> {
        atomic_write_json(&self.job_path(&job.envelope.job_id), job)
    }

    fn job_path(&self, job_id: &str) -> PathBuf {
        self.paths.jobs.join(format!("{job_id}.json"))
    }
}

/// Motherbrain work is bounded separately from forebrain job classes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LiveLearningLimits {
    pub enabled_classes: BTreeSet<String>,
    pub max_cpu_percent: u8,
    pub max_memory_bytes: u64,
    pub max_step_millis: u64,
    pub pause_during_persistence_pressure: bool,
    pub pause_during_network_pressure: bool,
}

impl Default for LiveLearningLimits {
    fn default() -> Self {
        Self {
            enabled_classes: [
                "replay_accumulation",
                "embedding_insert",
                "graph_mutation",
                "centroid_update",
                "calibration",
                "novelty_statistics",
                "small_classifier_head",
                "short_horizon_adaptation",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            max_cpu_percent: 20,
            max_memory_bytes: 256 * 1024 * 1024,
            max_step_millis: 25,
            pause_during_persistence_pressure: true,
            pause_during_network_pressure: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::Scope;
    use crate::bundle::{
        BundleBuilder, BundleContentKind, BundleSourceAdapter, ExperienceBundleRequest,
        FileExportAdapter,
    };
    use crate::capability::{AcceleratorCapabilities, CapabilityProbe};
    use std::path::Path;
    use uuid::Uuid;

    fn temp_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("pete-job-{name}-{}", Uuid::new_v4()))
    }

    fn caps() -> AcceleratorCapabilities {
        AcceleratorCapabilities::from_probe(
            "fore",
            "boot",
            "v",
            CapabilityProbe {
                architecture: "x86_64".into(),
                cpu_cores: 4,
                total_ram_bytes: 1000,
                available_ram_bytes: 1000,
                storage_capacity_bytes: 1000,
                storage_available_bytes: 1000,
                ..Default::default()
            },
        )
    }

    fn principal() -> Principal {
        Principal {
            id: "mother".into(),
            scopes: [Scope::SubmitJob, Scope::CancelJob].into_iter().collect(),
        }
    }

    fn paths(root: &Path) -> WorkerPaths {
        WorkerPaths {
            bundles: root.join("bundles"),
            jobs: root.join("jobs"),
            workspaces: root.join("work"),
            candidates: root.join("candidates"),
        }
    }

    fn bundle(root: &Path) -> String {
        fs::create_dir_all(root).unwrap();
        let source = root.join("fixture.json");
        fs::write(&source, b"{\"t_ms\":1,\"observation\":\"fixture\"}\n").unwrap();
        let request = ExperienceBundleRequest {
            pete_id: "pete".into(),
            source_node_id: "mother".into(),
            begin_timestamp_ms: 1,
            end_timestamp_ms: 1,
            event_range: None,
            source_checkpoints: BTreeMap::new(),
            software_identity: "commit".into(),
            schema_versions: BTreeMap::new(),
            active_model_versions: BTreeMap::new(),
            configuration_identity: "cfg".into(),
            calibration_identity: "cal".into(),
        };
        let adapters: Vec<Box<dyn BundleSourceAdapter>> = vec![Box::new(FileExportAdapter {
            source,
            destination: "fixture.json".into(),
            kind: BundleContentKind::SensorRecords,
            required: true,
        })];
        let path = BundleBuilder {
            root: root.to_path_buf(),
        }
        .create(&request, &adapters)
        .unwrap();
        validate_bundle(&path).unwrap().bundle_id
    }

    fn envelope(id: String) -> JobEnvelope {
        JobEnvelope::deterministic(
            JobClass::FixtureDigest,
            vec![id],
            ResourceRequirements::default(),
            BTreeMap::new(),
            "mother",
            "commit",
        )
        .unwrap()
    }

    #[test]
    fn durable_running_job_becomes_interrupted_after_restart() {
        let root = temp_root("restart");
        let worker = ForebrainWorker::open(paths(&root), caps()).unwrap();
        let mut job = worker
            .enqueue(envelope("exp-missing".into()), &principal())
            .unwrap();
        job.transition(JobStatus::Running, 20, "fixture crash point");
        worker.save(&job).unwrap();
        let reopened = ForebrainWorker::open(paths(&root), caps()).unwrap();
        assert_eq!(
            reopened.load(&job.envelope.job_id).unwrap().status,
            JobStatus::Interrupted
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fixture_job_produces_deterministic_candidate() {
        let root = temp_root("fixture");
        let bundle_id = bundle(&paths(&root).bundles);
        let worker = ForebrainWorker::open(paths(&root), caps()).unwrap();
        let env = envelope(bundle_id);
        worker.enqueue(env.clone(), &principal()).unwrap();
        let first = worker.run_once().unwrap().unwrap();
        assert_eq!(first.status, JobStatus::Succeeded);
        let first_id = first.candidate_id.unwrap();

        let second_root = temp_root("fixture-second");
        fs::create_dir_all(paths(&second_root).bundles).unwrap();
        copy_dir_for_test(&paths(&root).bundles, &paths(&second_root).bundles);
        let worker2 = ForebrainWorker::open(paths(&second_root), caps()).unwrap();
        worker2.enqueue(env, &principal()).unwrap();
        let second_id = worker2.run_once().unwrap().unwrap().candidate_id.unwrap();
        assert_eq!(first_id, second_id);
        fs::remove_dir_all(root).unwrap();
        fs::remove_dir_all(second_root).unwrap();
    }

    fn copy_dir_for_test(source: &Path, destination: &Path) {
        fs::create_dir_all(destination).unwrap();
        for entry in fs::read_dir(source).unwrap().filter_map(Result::ok) {
            let path = entry.path();
            let target = destination.join(entry.file_name());
            if path.is_dir() {
                copy_dir_for_test(&path, &target);
            } else {
                fs::copy(path, target).unwrap();
            }
        }
    }
}
