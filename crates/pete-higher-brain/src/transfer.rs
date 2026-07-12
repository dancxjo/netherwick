use crate::auth::{Principal, Scope};
use crate::bundle::{validate_bundle, COMPLETE_FILE, MANIFEST_FILE};
use crate::{atomic_write_json, sha256_file, sync_dir};
use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransferOutcome {
    Completed { destination: PathBuf },
    Interrupted { bytes_copied: u64 },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransferAcknowledgement {
    pub schema_version: u32,
    pub bundle_id: String,
    pub sender_principal: String,
    pub receiver_id: String,
    pub manifest_sha256: String,
    pub acknowledged_at: String,
}

/// Resumable local transport core. Production uses the same immutable layout
/// over rsync/SFTP via an enrolled SSH key; this function is also the simulator
/// and post-transfer verification path.
pub fn transfer_bundle(
    source: &Path,
    destination_root: &Path,
    acknowledgement_root: &Path,
    sender: &Principal,
    receiver_id: &str,
    byte_budget: Option<u64>,
) -> Result<TransferOutcome> {
    sender.require(Scope::TransferExperience)?;
    let manifest = validate_bundle(source)?;
    fs::create_dir_all(destination_root)?;
    let destination = destination_root.join(format!("{}.bundle", manifest.bundle_id));
    if destination.exists() {
        let received = validate_bundle(&destination)?;
        if received != manifest {
            anyhow::bail!("existing destination contradicts bundle identity");
        }
        acknowledge(
            acknowledgement_root,
            source,
            &manifest.bundle_id,
            sender,
            receiver_id,
        )?;
        return Ok(TransferOutcome::Completed { destination });
    }

    let partial = destination_root.join(".partial").join(&manifest.bundle_id);
    fs::create_dir_all(&partial)?;
    let mut copied_this_attempt = 0u64;
    let mut paths = manifest
        .files
        .iter()
        .map(|file| file.path.clone())
        .collect::<Vec<_>>();
    paths.extend([PathBuf::from(MANIFEST_FILE), PathBuf::from(COMPLETE_FILE)]);
    for relative in paths {
        let src = source.join(&relative);
        let dst = partial.join(&relative);
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        let remaining_budget = byte_budget.map(|budget| budget.saturating_sub(copied_this_attempt));
        let copied = resume_copy(&src, &dst, remaining_budget)?;
        copied_this_attempt = copied_this_attempt.saturating_add(copied);
        if byte_budget.is_some_and(|budget| copied_this_attempt >= budget)
            && fs::metadata(&dst)?.len() < fs::metadata(&src)?.len()
        {
            return Ok(TransferOutcome::Interrupted {
                bytes_copied: copied_this_attempt,
            });
        }
    }
    validate_bundle(&partial)?;
    sync_tree(&partial)?;
    fs::rename(&partial, &destination)?;
    sync_dir(destination_root)?;
    acknowledge(
        acknowledgement_root,
        source,
        &manifest.bundle_id,
        sender,
        receiver_id,
    )?;
    Ok(TransferOutcome::Completed { destination })
}

fn resume_copy(source: &Path, destination: &Path, budget: Option<u64>) -> Result<u64> {
    let source_len = fs::metadata(source)?.len();
    let mut offset = fs::metadata(destination).map(|m| m.len()).unwrap_or(0);
    if offset > source_len {
        fs::remove_file(destination)?;
        offset = 0;
    }
    if offset == source_len {
        if sha256_file(source)? == sha256_file(destination)? {
            return Ok(0);
        }
        fs::remove_file(destination)?;
        offset = 0;
    }
    let mut input = fs::File::open(source)?;
    input.seek(SeekFrom::Start(offset))?;
    let mut output = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(destination)?;
    let mut remaining = budget.unwrap_or(u64::MAX);
    let mut copied = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    while remaining > 0 {
        let read_limit = buffer.len().min(remaining as usize);
        let read = input.read(&mut buffer[..read_limit])?;
        if read == 0 {
            break;
        }
        output.write_all(&buffer[..read])?;
        copied += read as u64;
        remaining -= read as u64;
    }
    output.sync_all()?;
    Ok(copied)
}

fn acknowledge(
    root: &Path,
    source: &Path,
    bundle_id: &str,
    sender: &Principal,
    receiver_id: &str,
) -> Result<()> {
    fs::create_dir_all(root)?;
    let path = root.join(format!("{bundle_id}.{receiver_id}.ack.json"));
    if path.exists() {
        return Ok(());
    }
    atomic_write_json(
        &path,
        &TransferAcknowledgement {
            schema_version: 1,
            bundle_id: bundle_id.into(),
            sender_principal: sender.id.clone(),
            receiver_id: receiver_id.into(),
            manifest_sha256: sha256_file(&source.join(MANIFEST_FILE))?,
            acknowledged_at: Utc::now().to_rfc3339(),
        },
    )
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
    use crate::auth::Scope;
    use crate::bundle::{
        BundleBuilder, BundleContentKind, BundleSourceAdapter, ExperienceBundleRequest,
        FileExportAdapter,
    };
    use std::collections::{BTreeMap, BTreeSet};
    use uuid::Uuid;

    fn principal() -> Principal {
        Principal {
            id: "motherbrain".into(),
            scopes: [Scope::TransferExperience]
                .into_iter()
                .collect::<BTreeSet<_>>(),
        }
    }

    fn fixture(root: &Path) -> PathBuf {
        fs::create_dir_all(root).unwrap();
        let source = root.join("source");
        fs::write(&source, vec![7u8; 200_000]).unwrap();
        let request = ExperienceBundleRequest {
            pete_id: "pete".into(),
            source_node_id: "mother".into(),
            begin_timestamp_ms: 1,
            end_timestamp_ms: 2,
            event_range: None,
            source_checkpoints: BTreeMap::new(),
            software_identity: "a".into(),
            schema_versions: BTreeMap::new(),
            active_model_versions: BTreeMap::new(),
            configuration_identity: "a".into(),
            calibration_identity: "a".into(),
        };
        let adapters: Vec<Box<dyn BundleSourceAdapter>> = vec![Box::new(FileExportAdapter {
            source,
            destination: "sensor.bin".into(),
            kind: BundleContentKind::SensorRecords,
            required: true,
        })];
        BundleBuilder {
            root: root.join("bundles"),
        }
        .create(&request, &adapters)
        .unwrap()
    }

    #[test]
    fn interrupted_transfer_resumes_and_repeats_safely() {
        let root = std::env::temp_dir().join(format!("pete-transfer-{}", Uuid::new_v4()));
        let source = fixture(&root);
        let received = root.join("received");
        let acks = root.join("acks");
        assert!(matches!(
            transfer_bundle(&source, &received, &acks, &principal(), "fore", Some(1024)).unwrap(),
            TransferOutcome::Interrupted { .. }
        ));
        let completed =
            transfer_bundle(&source, &received, &acks, &principal(), "fore", None).unwrap();
        assert!(matches!(completed, TransferOutcome::Completed { .. }));
        let repeated =
            transfer_bundle(&source, &received, &acks, &principal(), "fore", None).unwrap();
        assert!(matches!(repeated, TransferOutcome::Completed { .. }));
        assert_eq!(fs::read_dir(acks).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn transfer_requires_specific_scope() {
        let root = std::env::temp_dir().join(format!("pete-transfer-auth-{}", Uuid::new_v4()));
        let source = fixture(&root);
        let denied = Principal {
            id: "x".into(),
            scopes: BTreeSet::new(),
        };
        assert!(transfer_bundle(
            &source,
            &root.join("r"),
            &root.join("a"),
            &denied,
            "f",
            None
        )
        .is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
