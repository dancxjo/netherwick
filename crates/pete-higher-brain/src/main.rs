use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use pete_higher_brain::auth::{Principal, Scope};
use pete_higher_brain::bundle::{
    validate_bundle, BundleBuilder, BundleContentKind, BundleSourceAdapter,
    ExperienceBundleRequest, FileExportAdapter, JsonlLedgerAdapter, PhysicalCaptureAdapter,
};
use pete_higher_brain::candidate::{
    validate_candidate, ActivationPolicy, CandidateCompatibility, CandidateStore,
};
use pete_higher_brain::capability::detect_local;
use pete_higher_brain::discovery::{
    advertise_once, local_interfaces, DataPlaneConfig, DiscoveryAdvertisement,
};
use pete_higher_brain::failover::{acceptance_matrix, FailoverConfig};
use pete_higher_brain::job::{ForebrainWorker, JobEnvelope, WorkerPaths};
use pete_higher_brain::job::{JobClass, ResourceRequirements};
use pete_higher_brain::transfer::transfer_bundle;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

#[derive(Parser)]
#[command(about = "Netherwick motherbrain/forebrain data-plane tools")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Capabilities {
        #[arg(long, default_value = "forebrain-development")]
        node_id: String,
        #[arg(long, default_value = ".")]
        workspace: PathBuf,
    },
    ValidateNode {
        #[arg(long, default_value = "/etc/netherwick/forebrain.toml")]
        config: PathBuf,
    },
    Advertise {
        #[arg(long, default_value = "/etc/netherwick/forebrain.toml")]
        config: PathBuf,
        #[arg(long)]
        once: bool,
    },
    Worker {
        #[arg(long, default_value = "/etc/netherwick/forebrain.toml")]
        config: PathBuf,
        #[arg(long)]
        once: bool,
    },
    BundleCreate {
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        ledger: Option<PathBuf>,
        #[arg(long)]
        graph: Option<PathBuf>,
        #[arg(long)]
        vectors: Option<PathBuf>,
        #[arg(long)]
        sensors: Option<PathBuf>,
        /// Complete WorldLab capture whose manifest source is RealRobot.
        #[arg(long)]
        capture: Option<PathBuf>,
    },
    BundleVerify {
        path: PathBuf,
    },
    BundleTransfer {
        source: PathBuf,
        #[arg(long)]
        destination: PathBuf,
        #[arg(long)]
        acknowledgements: PathBuf,
        #[arg(long, default_value = "forebrain")]
        receiver: String,
        /// Stop after this many bytes to exercise resumable transfer.
        #[arg(long)]
        byte_budget: Option<u64>,
    },
    JobCreate {
        #[arg(long)]
        bundle_id: Vec<String>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "motherbrain")]
        submitter: String,
        #[arg(long, default_value = "development")]
        software_identity: String,
        #[arg(long, value_enum, default_value_t = JobClassArg::FixtureDigest)]
        job_class: JobClassArg,
        /// Modality each physical dataset row must contain for the coverage metric.
        #[arg(long)]
        required_modality: Vec<String>,
        #[arg(long, default_value_t = 0.0)]
        minimum_required_modality_frame_ratio: f64,
    },
    JobSubmit {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        envelope: PathBuf,
    },
    JobStatus {
        #[arg(long)]
        config: PathBuf,
        job_id: String,
    },
    JobRetry {
        #[arg(long)]
        config: PathBuf,
        job_id: String,
    },
    CandidateValidate {
        path: PathBuf,
    },
    CandidateReceive {
        #[arg(long)]
        store: PathBuf,
        candidate: PathBuf,
    },
    CandidateStage {
        #[arg(long)]
        store: PathBuf,
        candidate_id: String,
    },
    CandidateActivate {
        #[arg(long)]
        store: PathBuf,
        #[arg(long)]
        approve: bool,
        candidate_id: String,
    },
    CandidateRollback {
        #[arg(long)]
        store: PathBuf,
    },
    /// Run deterministic host-link failure injection and print the matrix.
    FailoverCheck,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum JobClassArg {
    FixtureDigest,
    DatasetConstruction,
}

impl From<JobClassArg> for JobClass {
    fn from(value: JobClassArg) -> Self {
        match value {
            JobClassArg::FixtureDigest => Self::FixtureDigest,
            JobClassArg::DatasetConstruction => Self::DatasetConstruction,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
struct ForebrainConfig {
    node_id: String,
    workspace: PathBuf,
    bundles: PathBuf,
    datasets: PathBuf,
    jobs: PathBuf,
    candidates: PathBuf,
    logs: PathBuf,
    temporary: PathBuf,
    poll_interval_seconds: u64,
    data_plane: DataPlaneConfig,
    failover: FailoverConfig,
}

impl Default for ForebrainConfig {
    fn default() -> Self {
        let root = PathBuf::from("/var/lib/netherwick/forebrain");
        Self {
            node_id: "forebrain-unenrolled".into(),
            workspace: root.join("workspace"),
            bundles: root.join("experience"),
            datasets: root.join("datasets"),
            jobs: root.join("jobs"),
            candidates: root.join("candidates"),
            logs: PathBuf::from("/var/log/netherwick"),
            temporary: root.join("tmp"),
            poll_interval_seconds: 5,
            data_plane: DataPlaneConfig::default(),
            failover: FailoverConfig::default(),
        }
    }
}

impl ForebrainConfig {
    fn load(path: &Path) -> Result<Self> {
        Ok(toml::from_str(
            &fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?,
        )?)
    }

    fn worker_paths(&self) -> WorkerPaths {
        WorkerPaths {
            bundles: self.bundles.clone(),
            jobs: self.jobs.clone(),
            workspaces: self.workspace.clone(),
            candidates: self.candidates.clone(),
        }
    }
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Capabilities { node_id, workspace } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&detect_local(&workspace, node_id)?)?
            );
        }
        Command::ValidateNode { config } => validate_node(&config)?,
        Command::Advertise { config, once } => advertise(&config, once)?,
        Command::Worker { config, once } => worker(&config, once)?,
        Command::BundleCreate {
            request,
            output,
            ledger,
            graph,
            vectors,
            sensors,
            capture,
        } => {
            let request: ExperienceBundleRequest = pete_higher_brain::read_json(&request)?;
            let mut adapters: Vec<Box<dyn BundleSourceAdapter>> = Vec::new();
            if let Some(ledger_root) = ledger {
                adapters.push(Box::new(JsonlLedgerAdapter { ledger_root }));
            }
            if let Some(capture_root) = capture {
                adapters.push(Box::new(PhysicalCaptureAdapter { capture_root }));
            }
            for (source, destination, kind) in [
                (graph, "stores/graph.json", BundleContentKind::GraphExport),
                (
                    vectors,
                    "stores/vectors.json",
                    BundleContentKind::VectorRecords,
                ),
                (
                    sensors,
                    "perception/sensors.jsonl",
                    BundleContentKind::SensorRecords,
                ),
            ] {
                if let Some(source) = source {
                    adapters.push(Box::new(FileExportAdapter {
                        source,
                        destination: destination.into(),
                        kind,
                        required: true,
                    }));
                }
            }
            let path = BundleBuilder { root: output }.create(&request, &adapters)?;
            println!("{}", path.display());
        }
        Command::BundleVerify { path } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&validate_bundle(&path)?)?
            );
        }
        Command::BundleTransfer {
            source,
            destination,
            acknowledgements,
            receiver,
            byte_budget,
        } => {
            println!(
                "{:?}",
                transfer_bundle(
                    &source,
                    &destination,
                    &acknowledgements,
                    &local_principal([Scope::TransferExperience]),
                    &receiver,
                    byte_budget,
                )?
            );
        }
        Command::JobCreate {
            bundle_id,
            output,
            submitter,
            software_identity,
            job_class,
            required_modality,
            minimum_required_modality_frame_ratio,
        } => {
            let mut parameters = BTreeMap::new();
            if matches!(job_class, JobClassArg::DatasetConstruction) {
                if !required_modality.is_empty() {
                    parameters.insert(
                        "required_modalities".into(),
                        serde_json::json!(required_modality),
                    );
                }
                parameters.insert(
                    "minimum_required_modality_frame_ratio".into(),
                    serde_json::json!(minimum_required_modality_frame_ratio),
                );
            }
            let envelope = JobEnvelope::deterministic(
                job_class.into(),
                bundle_id,
                ResourceRequirements::default(),
                parameters,
                submitter,
                software_identity,
            )?;
            pete_higher_brain::atomic_write_json(&output, &envelope)?;
            println!("{}", envelope.job_id);
        }
        Command::JobSubmit { config, envelope } => {
            let config = ForebrainConfig::load(&config)?;
            let capabilities = detect_local(&config.workspace, config.node_id.clone())?;
            let worker = ForebrainWorker::open(config.worker_paths(), capabilities)?;
            let envelope: JobEnvelope = pete_higher_brain::read_json(&envelope)?;
            let job = worker.enqueue(envelope, &local_principal([Scope::SubmitJob]))?;
            println!("{}", serde_json::to_string_pretty(&job)?);
        }
        Command::JobStatus { config, job_id } => {
            let config = ForebrainConfig::load(&config)?;
            let capabilities = detect_local(&config.workspace, config.node_id.clone())?;
            let worker = ForebrainWorker::open(config.worker_paths(), capabilities)?;
            println!("{}", serde_json::to_string_pretty(&worker.load(&job_id)?)?);
        }
        Command::JobRetry { config, job_id } => {
            let config = ForebrainConfig::load(&config)?;
            let capabilities = detect_local(&config.workspace, config.node_id.clone())?;
            let worker = ForebrainWorker::open(config.worker_paths(), capabilities)?;
            worker.retry_interrupted(&job_id, &local_principal([Scope::SubmitJob]))?;
            println!("retry queued {job_id}");
        }
        Command::CandidateValidate { path } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&validate_candidate(&path, None)?)?
            );
        }
        Command::CandidateReceive { store, candidate } => {
            let store = candidate_store(store);
            println!(
                "{}",
                store.receive(&candidate, &local_principal([Scope::ReturnCandidate]))?
            );
        }
        Command::CandidateStage {
            store,
            candidate_id,
        } => {
            let store = candidate_store(store);
            let p = local_principal([Scope::StageCandidate]);
            store.validate(&candidate_id, &p)?;
            store.stage(&candidate_id, &p)?;
            println!("staged {candidate_id}");
        }
        Command::CandidateActivate {
            store,
            approve,
            candidate_id,
        } => {
            candidate_store(store).activate(
                &candidate_id,
                &local_principal([Scope::ActivateCandidate]),
                approve,
            )?;
            println!("activated {candidate_id}");
        }
        Command::CandidateRollback { store } => {
            println!(
                "activated {}",
                candidate_store(store).rollback(&local_principal([Scope::RollbackModel]))?
            );
        }
        Command::FailoverCheck => {
            let checks = acceptance_matrix()?;
            println!("{}", serde_json::to_string_pretty(&checks)?);
            if checks.iter().any(|check| !check.passed) {
                anyhow::bail!("one or more failover acceptance scenarios failed");
            }
        }
    }
    Ok(())
}

fn validate_node(path: &Path) -> Result<()> {
    let config = ForebrainConfig::load(path)?;
    let required = [
        &config.workspace,
        &config.bundles,
        &config.datasets,
        &config.jobs,
        &config.candidates,
        &config.logs,
        &config.temporary,
    ];
    let mut directory_status = BTreeMap::new();
    for directory in required {
        directory_status.insert(
            directory.display().to_string(),
            directory.is_dir() && !fs::metadata(directory)?.permissions().readonly(),
        );
    }
    let interfaces = local_interfaces().unwrap_or_default();
    let selected = config.data_plane.select_interfaces(&interfaces);
    let failover = config.failover.validate();
    let capabilities = detect_local(&config.workspace, config.node_id)?;
    let ready = directory_status.values().all(|ready| *ready)
        && selected.is_ok()
        && failover.is_ok()
        && capabilities.ready;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "ready": ready,
            "directories": directory_status,
            "data_plane": selected.as_ref().map(|items| items.iter().map(|item| &item.name).collect::<Vec<_>>()).ok(),
            "data_plane_error": selected.err().map(|error| error.to_string()),
            "failover_ready": failover.is_ok(),
            "failover_error": failover.err().map(|error| error.to_string()),
            "capabilities": capabilities,
        }))?
    );
    if !ready {
        anyhow::bail!("forebrain validation failed");
    }
    Ok(())
}

fn advertise(path: &Path, once: bool) -> Result<()> {
    let config = ForebrainConfig::load(path)?;
    loop {
        let interfaces = local_interfaces()?;
        let capabilities = detect_local(&config.workspace, config.node_id.clone())?;
        let endpoints = config.data_plane.endpoints(&interfaces)?;
        let advertisement = DiscoveryAdvertisement {
            schema_version: 1,
            role: "forebrain".into(),
            capabilities,
            endpoints,
        };
        advertise_once(&config.data_plane, &interfaces, &advertisement)?;
        if once {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(config.poll_interval_seconds.max(1)));
    }
}

fn worker(path: &Path, once: bool) -> Result<()> {
    let config = ForebrainConfig::load(path)?;
    let capabilities = detect_local(&config.workspace, config.node_id.clone())?;
    let worker = ForebrainWorker::open(config.worker_paths(), capabilities)?;
    loop {
        if let Some(job) = worker.run_once()? {
            println!(
                "{} {}: {}",
                job.envelope.job_id, job.progress_percent, job.detail
            );
        }
        if once {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(config.poll_interval_seconds.max(1)));
    }
}

fn local_principal(scopes: impl IntoIterator<Item = Scope>) -> Principal {
    Principal {
        id: "local-service-account".into(),
        scopes: scopes.into_iter().collect(),
    }
}

fn candidate_store(root: PathBuf) -> CandidateStore {
    CandidateStore {
        root,
        compatibility: CandidateCompatibility {
            input_schema_versions: ["experience_bundle/1".into()].into_iter().collect(),
            output_schema_versions: [
                "fixture_model/1".into(),
                "physical_experience_dataset/1".into(),
            ]
            .into_iter()
            .collect(),
            preprocessing_versions: ["fixture_digest/1".into(), "physical_capture_index/1".into()]
                .into_iter()
                .collect(),
            deployment_targets: [
                "motherbrain_fixture".into(),
                "motherbrain_dataset_library".into(),
            ]
            .into_iter()
            .collect(),
            runtimes: BTreeSet::new(),
        },
        policy: ActivationPolicy::OperatorApproval,
    }
}
