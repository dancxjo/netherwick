use crate::auth::{Principal, Scope};
use crate::{
    atomic_replace_symlink, atomic_write_json, canonical_json, read_json, safe_relative_path,
    sha256_bytes, sha256_file, sync_dir, CANDIDATE_SCHEMA_VERSION,
};
use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub const CANDIDATE_MANIFEST: &str = "candidate.json";
pub const CANDIDATE_COMPLETE: &str = ".complete";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateFile {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelCandidateManifest {
    pub schema_version: u32,
    pub candidate_id: String,
    pub algorithm_family: String,
    pub artifacts: Vec<CandidateFile>,
    pub preprocessing_version: String,
    pub input_schema_version: String,
    pub output_schema_version: String,
    pub training_build_identity: String,
    pub source_experience_bundle_ids: Vec<String>,
    pub training_parameters: BTreeMap<String, Value>,
    pub evaluation_results: BTreeMap<String, Value>,
    pub hardware_requirements: BTreeMap<String, Value>,
    pub runtime_requirements: BTreeSet<String>,
    pub intended_deployment_target: String,
    pub rollback_compatibility: BTreeMap<String, Value>,
    pub creation_status: String,
}

#[derive(Clone, Debug)]
pub struct CandidateRequest {
    pub algorithm_family: String,
    pub preprocessing_version: String,
    pub input_schema_version: String,
    pub output_schema_version: String,
    pub training_build_identity: String,
    pub source_experience_bundle_ids: Vec<String>,
    pub training_parameters: BTreeMap<String, Value>,
    pub evaluation_results: BTreeMap<String, Value>,
    pub hardware_requirements: BTreeMap<String, Value>,
    pub runtime_requirements: BTreeSet<String>,
    pub intended_deployment_target: String,
    pub rollback_compatibility: BTreeMap<String, Value>,
}

pub fn create_candidate(
    root: &Path,
    request: &CandidateRequest,
    artifact_sources: &[(PathBuf, PathBuf)],
) -> Result<PathBuf> {
    if request.intended_deployment_target == "brainstem" {
        anyhow::bail!("arbitrary learned candidates may not target the brainstem");
    }
    if request.source_experience_bundle_ids.is_empty() {
        anyhow::bail!("candidate must identify its source experience bundles");
    }
    fs::create_dir_all(root)?;
    let staging = root.join(format!(".candidate-{}.staging", std::process::id()));
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(staging.join("artifacts"))?;
    let result = (|| {
        let mut artifacts = Vec::new();
        for (source, relative) in artifact_sources {
            let relative = safe_relative_path(relative)?;
            let relative = PathBuf::from("artifacts").join(relative);
            let destination = staging.join(&relative);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(source, &destination)?;
            artifacts.push(CandidateFile {
                path: relative,
                sha256: sha256_file(&destination)?,
                bytes: fs::metadata(&destination)?.len(),
            });
        }
        artifacts.sort_by(|left, right| left.path.cmp(&right.path));
        let identity = sha256_bytes(&canonical_json(&(
            &request.algorithm_family,
            &artifacts,
            &request.preprocessing_version,
            &request.input_schema_version,
            &request.output_schema_version,
            &request.training_build_identity,
            &request.source_experience_bundle_ids,
            &request.training_parameters,
            &request.evaluation_results,
            &request.hardware_requirements,
            &request.runtime_requirements,
            &request.intended_deployment_target,
            &request.rollback_compatibility,
        ))?);
        let candidate_id = format!("candidate-{}", &identity[..32]);
        let manifest = ModelCandidateManifest {
            schema_version: CANDIDATE_SCHEMA_VERSION,
            candidate_id: candidate_id.clone(),
            algorithm_family: request.algorithm_family.clone(),
            artifacts,
            preprocessing_version: request.preprocessing_version.clone(),
            input_schema_version: request.input_schema_version.clone(),
            output_schema_version: request.output_schema_version.clone(),
            training_build_identity: request.training_build_identity.clone(),
            source_experience_bundle_ids: request.source_experience_bundle_ids.clone(),
            training_parameters: request.training_parameters.clone(),
            evaluation_results: request.evaluation_results.clone(),
            hardware_requirements: request.hardware_requirements.clone(),
            runtime_requirements: request.runtime_requirements.clone(),
            intended_deployment_target: request.intended_deployment_target.clone(),
            rollback_compatibility: request.rollback_compatibility.clone(),
            creation_status: "complete".into(),
        };
        let bytes = serde_json::to_vec_pretty(&manifest)?;
        fs::write(staging.join(CANDIDATE_MANIFEST), &bytes)?;
        fs::write(
            staging.join(CANDIDATE_COMPLETE),
            format!("{}\n", sha256_bytes(&bytes)),
        )?;
        sync_tree(&staging)?;
        let destination = root.join(format!("{candidate_id}.candidate"));
        if destination.exists() {
            if validate_candidate(&destination, None)? != manifest {
                anyhow::bail!("candidate identity collision");
            }
            fs::remove_dir_all(&staging)?;
        } else {
            fs::rename(&staging, &destination)?;
            sync_dir(root)?;
        }
        Ok(destination)
    })();
    if result.is_err() && staging.exists() {
        let _ = fs::remove_dir_all(staging);
    }
    result
}

#[derive(Clone, Debug, Default)]
pub struct CandidateCompatibility {
    pub input_schema_versions: BTreeSet<String>,
    pub output_schema_versions: BTreeSet<String>,
    pub preprocessing_versions: BTreeSet<String>,
    pub deployment_targets: BTreeSet<String>,
    pub runtimes: BTreeSet<String>,
}

pub fn validate_candidate(
    root: &Path,
    compatibility: Option<&CandidateCompatibility>,
) -> Result<ModelCandidateManifest> {
    let bytes = fs::read(root.join(CANDIDATE_MANIFEST))?;
    let manifest: ModelCandidateManifest = serde_json::from_slice(&bytes)?;
    if manifest.schema_version != CANDIDATE_SCHEMA_VERSION || manifest.creation_status != "complete"
    {
        anyhow::bail!("incompatible or incomplete candidate");
    }
    if manifest.intended_deployment_target == "brainstem" {
        anyhow::bail!("brainstem candidate target is forbidden");
    }
    if fs::read_to_string(root.join(CANDIDATE_COMPLETE))?.trim() != sha256_bytes(&bytes) {
        anyhow::bail!("candidate completion marker mismatch");
    }
    for artifact in &manifest.artifacts {
        let path = root.join(safe_relative_path(&artifact.path)?);
        if fs::metadata(&path)?.len() != artifact.bytes || sha256_file(&path)? != artifact.sha256 {
            anyhow::bail!(
                "candidate artifact checksum mismatch: {}",
                artifact.path.display()
            );
        }
    }
    if let Some(compatibility) = compatibility {
        let supported = |set: &BTreeSet<String>, value: &str| set.is_empty() || set.contains(value);
        if !supported(
            &compatibility.input_schema_versions,
            &manifest.input_schema_version,
        ) || !supported(
            &compatibility.output_schema_versions,
            &manifest.output_schema_version,
        ) || !supported(
            &compatibility.preprocessing_versions,
            &manifest.preprocessing_version,
        ) || !supported(
            &compatibility.deployment_targets,
            &manifest.intended_deployment_target,
        ) || !manifest
            .runtime_requirements
            .iter()
            .all(|runtime| compatibility.runtimes.contains(runtime))
        {
            anyhow::bail!("candidate is incompatible with this motherbrain");
        }
    }
    Ok(manifest)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateState {
    Received,
    Validated,
    Rejected,
    Staged,
    Activated,
    RolledBack,
    Superseded,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateTransition {
    pub state: CandidateState,
    pub at: String,
    pub principal: String,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateLifecycle {
    pub schema_version: u32,
    pub candidate_id: String,
    pub transitions: Vec<CandidateTransition>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationPolicy {
    OperatorApproval,
    SafeDevelopmentAuto,
}

pub struct CandidateStore {
    pub root: PathBuf,
    pub compatibility: CandidateCompatibility,
    pub policy: ActivationPolicy,
}

impl CandidateStore {
    pub fn receive(&self, candidate: &Path, principal: &Principal) -> Result<String> {
        principal.require(Scope::ReturnCandidate)?;
        let manifest = validate_candidate(candidate, None)?;
        let destination = self
            .root
            .join("received")
            .join(format!("{}.candidate", manifest.candidate_id));
        copy_immutable_tree(candidate, &destination)?;
        self.record(
            &manifest.candidate_id,
            CandidateState::Received,
            principal,
            None,
        )?;
        Ok(manifest.candidate_id)
    }

    pub fn validate(&self, candidate_id: &str, principal: &Principal) -> Result<()> {
        principal.require(Scope::StageCandidate)?;
        let path = self
            .root
            .join("received")
            .join(format!("{candidate_id}.candidate"));
        match validate_candidate(&path, Some(&self.compatibility)) {
            Ok(_) => self.record(candidate_id, CandidateState::Validated, principal, None),
            Err(error) => {
                self.record(
                    candidate_id,
                    CandidateState::Rejected,
                    principal,
                    Some(error.to_string()),
                )?;
                Err(error)
            }
        }
    }

    pub fn stage(&self, candidate_id: &str, principal: &Principal) -> Result<()> {
        principal.require(Scope::StageCandidate)?;
        let source = self
            .root
            .join("received")
            .join(format!("{candidate_id}.candidate"));
        validate_candidate(&source, Some(&self.compatibility))?;
        let destination = self
            .root
            .join("library")
            .join(format!("{candidate_id}.candidate"));
        copy_immutable_tree(&source, &destination)?;
        self.record(candidate_id, CandidateState::Staged, principal, None)
    }

    pub fn activate(
        &self,
        candidate_id: &str,
        principal: &Principal,
        operator_approved: bool,
    ) -> Result<()> {
        principal.require(Scope::ActivateCandidate)?;
        if self.policy == ActivationPolicy::OperatorApproval && !operator_approved {
            anyhow::bail!("operator approval is required by activation policy");
        }
        let target = self
            .root
            .join("library")
            .join(format!("{candidate_id}.candidate"));
        validate_candidate(&target, Some(&self.compatibility))?;
        let links = self.root.join("active");
        fs::create_dir_all(&links)?;
        let active = links.join("current");
        let previous = links.join("previous");
        let old = fs::read_link(&active).ok();
        if let Some(old) = &old {
            atomic_replace_symlink(old, &previous)?;
        }
        atomic_replace_symlink(&target, &active)?;
        if let Some(old) = old {
            if let Some(old_id) = candidate_id_from_path(&old) {
                self.record(&old_id, CandidateState::Superseded, principal, None)?;
            }
        }
        self.record(candidate_id, CandidateState::Activated, principal, None)
    }

    pub fn rollback(&self, principal: &Principal) -> Result<String> {
        principal.require(Scope::RollbackModel)?;
        let links = self.root.join("active");
        let active_link = links.join("current");
        let previous_link = links.join("previous");
        let active = fs::read_link(&active_link)?;
        let previous = fs::read_link(&previous_link)?;
        validate_candidate(&previous, Some(&self.compatibility))?;
        atomic_replace_symlink(&previous, &active_link)?;
        atomic_replace_symlink(&active, &previous_link)?;
        let previous_id = candidate_id_from_path(&previous)
            .ok_or_else(|| anyhow::anyhow!("invalid previous candidate link"))?;
        if let Some(active_id) = candidate_id_from_path(&active) {
            self.record(&active_id, CandidateState::RolledBack, principal, None)?;
        }
        self.record(
            &previous_id,
            CandidateState::Activated,
            principal,
            Some("rollback".into()),
        )?;
        Ok(previous_id)
    }

    pub fn lifecycle(&self, candidate_id: &str) -> Result<CandidateLifecycle> {
        read_json(&self.lifecycle_path(candidate_id))
    }

    fn lifecycle_path(&self, id: &str) -> PathBuf {
        self.root.join("state").join(format!("{id}.json"))
    }

    fn record(
        &self,
        id: &str,
        state: CandidateState,
        principal: &Principal,
        reason: Option<String>,
    ) -> Result<()> {
        let path = self.lifecycle_path(id);
        let mut lifecycle = if path.exists() {
            read_json(&path)?
        } else {
            CandidateLifecycle {
                schema_version: 1,
                candidate_id: id.into(),
                transitions: Vec::new(),
            }
        };
        lifecycle.transitions.push(CandidateTransition {
            state,
            at: Utc::now().to_rfc3339(),
            principal: principal.id.clone(),
            reason,
        });
        atomic_write_json(&path, &lifecycle)
    }
}

fn candidate_id_from_path(path: &Path) -> Option<String> {
    path.file_name()?
        .to_str()?
        .strip_suffix(".candidate")
        .map(str::to_string)
}

fn copy_immutable_tree(source: &Path, destination: &Path) -> Result<()> {
    if destination.exists() {
        let source_manifest = validate_candidate(source, None)?;
        if validate_candidate(destination, None)? != source_manifest {
            anyhow::bail!("existing candidate copy contradicts identity");
        }
        return Ok(());
    }
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let staging = parent.join(format!(".candidate-copy-{}.staging", std::process::id()));
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    copy_dir(source, &staging)?;
    validate_candidate(&staging, None)?;
    fs::rename(&staging, destination)?;
    sync_dir(parent)
}

fn copy_dir(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let path = entry.path();
        let target = destination.join(entry.file_name());
        if path.is_dir() {
            copy_dir(&path, &target)?;
        } else {
            fs::copy(path, target)?;
        }
    }
    Ok(())
}

fn sync_tree(root: &Path) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            sync_tree(&path)?;
        } else {
            fs::File::open(path)?.sync_all()?;
        }
    }
    sync_dir(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn temp() -> PathBuf {
        std::env::temp_dir().join(format!("pete-candidate-{}", Uuid::new_v4()))
    }

    fn request(tag: &str) -> CandidateRequest {
        CandidateRequest {
            algorithm_family: "fixture_digest_v1".into(),
            preprocessing_version: "fixture/1".into(),
            input_schema_version: "experience_bundle/1".into(),
            output_schema_version: "fixture_model/1".into(),
            training_build_identity: tag.into(),
            source_experience_bundle_ids: vec!["exp-a".into()],
            training_parameters: BTreeMap::new(),
            evaluation_results: BTreeMap::new(),
            hardware_requirements: BTreeMap::new(),
            runtime_requirements: BTreeSet::new(),
            intended_deployment_target: "motherbrain_fixture".into(),
            rollback_compatibility: BTreeMap::new(),
        }
    }

    fn principal() -> Principal {
        Principal {
            id: "operator".into(),
            scopes: [
                Scope::ReturnCandidate,
                Scope::StageCandidate,
                Scope::ActivateCandidate,
                Scope::RollbackModel,
            ]
            .into_iter()
            .collect(),
        }
    }

    fn compatibility() -> CandidateCompatibility {
        CandidateCompatibility {
            input_schema_versions: ["experience_bundle/1".into()].into_iter().collect(),
            output_schema_versions: ["fixture_model/1".into()].into_iter().collect(),
            preprocessing_versions: ["fixture/1".into()].into_iter().collect(),
            deployment_targets: ["motherbrain_fixture".into()].into_iter().collect(),
            runtimes: BTreeSet::new(),
        }
    }

    #[test]
    fn corrupt_and_incompatible_candidates_are_rejected() {
        let root = temp();
        fs::create_dir_all(&root).unwrap();
        let artifact = root.join("artifact");
        fs::write(&artifact, b"learned").unwrap();
        let candidate = create_candidate(
            &root.join("out"),
            &request("a"),
            &[(artifact, "weights".into())],
        )
        .unwrap();
        assert!(validate_candidate(&candidate, Some(&compatibility())).is_ok());
        let mut incompatible = compatibility();
        incompatible.output_schema_versions = ["another_schema/1".into()].into_iter().collect();
        assert!(validate_candidate(&candidate, Some(&incompatible)).is_err());
        fs::write(candidate.join("artifacts/weights"), b"corrupt").unwrap();
        assert!(validate_candidate(&candidate, Some(&compatibility())).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn activation_is_atomic_and_rollback_retains_previous() {
        let root = temp();
        fs::create_dir_all(&root).unwrap();
        let artifact_a = root.join("a");
        let artifact_b = root.join("b");
        fs::write(&artifact_a, b"a").unwrap();
        fs::write(&artifact_b, b"b").unwrap();
        let a = create_candidate(
            &root.join("out-a"),
            &request("a"),
            &[(artifact_a, "w".into())],
        )
        .unwrap();
        let b = create_candidate(
            &root.join("out-b"),
            &request("b"),
            &[(artifact_b, "w".into())],
        )
        .unwrap();
        let store = CandidateStore {
            root: root.join("store"),
            compatibility: compatibility(),
            policy: ActivationPolicy::OperatorApproval,
        };
        let p = principal();
        let a_id = store.receive(&a, &p).unwrap();
        store.validate(&a_id, &p).unwrap();
        store.stage(&a_id, &p).unwrap();
        store.activate(&a_id, &p, true).unwrap();
        let old_target = fs::read_link(store.root.join("active/current")).unwrap();
        let b_id = store.receive(&b, &p).unwrap();
        store.validate(&b_id, &p).unwrap();
        store.stage(&b_id, &p).unwrap();
        assert!(store.activate(&b_id, &p, false).is_err());
        assert_eq!(
            fs::read_link(store.root.join("active/current")).unwrap(),
            old_target
        );
        store.activate(&b_id, &p, true).unwrap();
        assert_eq!(store.rollback(&p).unwrap(), a_id);
        fs::remove_dir_all(root).unwrap();
    }
}
