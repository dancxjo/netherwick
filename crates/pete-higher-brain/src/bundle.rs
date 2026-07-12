use crate::{
    atomic_write_json, canonical_json, read_json, safe_relative_path, sha256_bytes, sha256_file,
    sync_dir, BUNDLE_SCHEMA_VERSION,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

pub const MANIFEST_FILE: &str = "manifest.json";
pub const COMPLETE_FILE: &str = ".complete";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventRange {
    pub first_cursor: Option<String>,
    pub last_cursor: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceBundleRequest {
    pub pete_id: String,
    pub source_node_id: String,
    pub begin_timestamp_ms: u64,
    pub end_timestamp_ms: u64,
    pub event_range: Option<EventRange>,
    pub source_checkpoints: BTreeMap<String, String>,
    pub software_identity: String,
    pub schema_versions: BTreeMap<String, String>,
    pub active_model_versions: BTreeMap<String, String>,
    pub configuration_identity: String,
    pub calibration_identity: String,
}

impl ExperienceBundleRequest {
    pub fn validate(&self) -> Result<()> {
        if self.pete_id.trim().is_empty() || self.source_node_id.trim().is_empty() {
            anyhow::bail!("Pete and source node identities are required");
        }
        if self.begin_timestamp_ms > self.end_timestamp_ms {
            anyhow::bail!("bundle beginning is after ending");
        }
        if self.software_identity.trim().is_empty()
            || self.configuration_identity.trim().is_empty()
            || self.calibration_identity.trim().is_empty()
        {
            anyhow::bail!("software, configuration, and calibration identities are required");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BundleContentKind {
    LedgerFrames,
    LedgerTransitions,
    GraphExport,
    VectorRecords,
    SensorRecords,
    BlobReferences,
    Other,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleFile {
    pub path: PathBuf,
    pub kind: BundleContentKind,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceBundleManifest {
    pub schema_version: u32,
    pub bundle_id: String,
    pub pete_id: String,
    pub source_node_id: String,
    pub begin_timestamp_ms: u64,
    pub end_timestamp_ms: u64,
    pub event_range: Option<EventRange>,
    pub source_checkpoints: BTreeMap<String, String>,
    pub software_identity: String,
    pub schema_versions: BTreeMap<String, String>,
    pub active_model_versions: BTreeMap<String, String>,
    pub configuration_identity: String,
    pub calibration_identity: String,
    pub files: Vec<BundleFile>,
    pub completeness_notes: Vec<String>,
    pub creation_status: String,
}

pub trait BundleSourceAdapter {
    fn export(
        &self,
        payload_root: &Path,
        request: &ExperienceBundleRequest,
    ) -> Result<Vec<ExportedFile>>;
}

#[derive(Clone, Debug)]
pub struct ExportedFile {
    pub path: PathBuf,
    pub kind: BundleContentKind,
    pub completeness_note: Option<String>,
}

#[derive(Clone, Debug)]
pub struct FileExportAdapter {
    pub source: PathBuf,
    pub destination: PathBuf,
    pub kind: BundleContentKind,
    pub required: bool,
}

impl BundleSourceAdapter for FileExportAdapter {
    fn export(
        &self,
        payload_root: &Path,
        _request: &ExperienceBundleRequest,
    ) -> Result<Vec<ExportedFile>> {
        let destination = safe_relative_path(&self.destination)?;
        if !self.source.exists() {
            if self.required {
                anyhow::bail!("required bundle source missing: {}", self.source.display());
            }
            return Ok(vec![ExportedFile {
                path: destination,
                kind: self.kind.clone(),
                completeness_note: Some(format!(
                    "optional source unavailable: {}",
                    self.source.display()
                )),
            }]);
        }
        let target = payload_root.join(&destination);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&self.source, &target)?;
        Ok(vec![ExportedFile {
            path: destination,
            kind: self.kind.clone(),
            completeness_note: None,
        }])
    }
}

/// Adapter around Netherwick's real JSONL ledger. It retains raw records and
/// filters by `t_ms`/`created_at_ms`, so provenance embedded in frames (sensor
/// vectors, preprocessing labels, actions, and observations) is not flattened.
#[derive(Clone, Debug)]
pub struct JsonlLedgerAdapter {
    pub ledger_root: PathBuf,
}

impl BundleSourceAdapter for JsonlLedgerAdapter {
    fn export(
        &self,
        payload_root: &Path,
        request: &ExperienceBundleRequest,
    ) -> Result<Vec<ExportedFile>> {
        let mut paths = Vec::new();
        collect_named(&self.ledger_root, "frames.jsonl", &mut paths)?;
        collect_named(&self.ledger_root, "session.jsonl", &mut paths)?;
        let frames = PathBuf::from("ledger/frames.jsonl");
        write_filtered_jsonl(
            &paths,
            &payload_root.join(&frames),
            request.begin_timestamp_ms,
            request.end_timestamp_ms,
        )?;

        paths.clear();
        collect_named(&self.ledger_root, "transitions.jsonl", &mut paths)?;
        let transitions = PathBuf::from("ledger/transitions.jsonl");
        write_filtered_jsonl(
            &paths,
            &payload_root.join(&transitions),
            request.begin_timestamp_ms,
            request.end_timestamp_ms,
        )?;
        Ok(vec![
            ExportedFile {
                path: frames,
                kind: BundleContentKind::LedgerFrames,
                completeness_note: None,
            },
            ExportedFile {
                path: transitions,
                kind: BundleContentKind::LedgerTransitions,
                completeness_note: None,
            },
        ])
    }
}

#[derive(Clone, Debug)]
pub struct BlobReferenceAdapter {
    pub references: Vec<BlobReference>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobReference {
    pub media_type: String,
    pub source_id: String,
    pub uri: String,
    pub sha256: Option<String>,
    pub preprocessing_identity: Option<String>,
    pub calibration_identity: Option<String>,
}

impl BundleSourceAdapter for BlobReferenceAdapter {
    fn export(
        &self,
        payload_root: &Path,
        _request: &ExperienceBundleRequest,
    ) -> Result<Vec<ExportedFile>> {
        let path = PathBuf::from("blobs/references.json");
        let target = payload_root.join(&path);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&target, canonical_json(&self.references)?)?;
        Ok(vec![ExportedFile {
            path,
            kind: BundleContentKind::BlobReferences,
            completeness_note: None,
        }])
    }
}

pub struct BundleBuilder {
    pub root: PathBuf,
}

impl BundleBuilder {
    pub fn create(
        &self,
        request: &ExperienceBundleRequest,
        adapters: &[Box<dyn BundleSourceAdapter>],
    ) -> Result<PathBuf> {
        request.validate()?;
        fs::create_dir_all(&self.root)?;
        let range_key = sha256_bytes(&canonical_json(request)?);
        let index_path = self.root.join("index").join(format!("{range_key}.json"));
        if index_path.exists() {
            let index: RangeIndex = read_json(&index_path)?;
            let existing = self.root.join(format!("{}.bundle", index.bundle_id));
            validate_bundle(&existing)?;
            return Ok(existing);
        }

        let staging = self.root.join(format!(
            ".export-{range_key}-{}.staging",
            std::process::id()
        ));
        if staging.exists() {
            fs::remove_dir_all(&staging)?;
        }
        let payload = staging.join("payload");
        fs::create_dir_all(&payload)?;
        let result = (|| {
            let mut exported = Vec::new();
            for adapter in adapters {
                exported.extend(adapter.export(&payload, request)?);
            }
            let mut files = Vec::new();
            let mut notes = Vec::new();
            for item in exported {
                if let Some(note) = item.completeness_note {
                    notes.push(note);
                }
                let relative = PathBuf::from("payload").join(safe_relative_path(&item.path)?);
                let full = staging.join(&relative);
                if !full.exists() {
                    continue;
                }
                files.push(BundleFile {
                    path: relative,
                    kind: item.kind,
                    sha256: sha256_file(&full)?,
                    bytes: fs::metadata(full)?.len(),
                });
            }
            files.sort_by(|left, right| left.path.cmp(&right.path));
            let identity = sha256_bytes(&canonical_json(&(request, &files))?);
            let bundle_id = format!("exp-{}", &identity[..32]);
            let manifest = ExperienceBundleManifest {
                schema_version: BUNDLE_SCHEMA_VERSION,
                bundle_id: bundle_id.clone(),
                pete_id: request.pete_id.clone(),
                source_node_id: request.source_node_id.clone(),
                begin_timestamp_ms: request.begin_timestamp_ms,
                end_timestamp_ms: request.end_timestamp_ms,
                event_range: request.event_range.clone(),
                source_checkpoints: request.source_checkpoints.clone(),
                software_identity: request.software_identity.clone(),
                schema_versions: request.schema_versions.clone(),
                active_model_versions: request.active_model_versions.clone(),
                configuration_identity: request.configuration_identity.clone(),
                calibration_identity: request.calibration_identity.clone(),
                files,
                completeness_notes: notes,
                creation_status: "complete".into(),
            };
            let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
            fs::write(staging.join(MANIFEST_FILE), &manifest_bytes)?;
            fs::write(
                staging.join(COMPLETE_FILE),
                format!("{}\n", sha256_bytes(&manifest_bytes)),
            )?;
            sync_tree(&staging)?;
            let final_path = self.root.join(format!("{bundle_id}.bundle"));
            if final_path.exists() {
                let existing = validate_bundle(&final_path)?;
                if existing != manifest {
                    anyhow::bail!("bundle identity collision for {bundle_id}");
                }
                fs::remove_dir_all(&staging)?;
            } else {
                fs::rename(&staging, &final_path)?;
                sync_dir(&self.root)?;
            }
            atomic_write_json(
                &index_path,
                &RangeIndex {
                    bundle_id: bundle_id.clone(),
                },
            )?;
            Ok(final_path)
        })();
        if result.is_err() && staging.exists() {
            let _ = fs::remove_dir_all(&staging);
        }
        result
    }
}

#[derive(Serialize, Deserialize)]
struct RangeIndex {
    bundle_id: String,
}

pub fn validate_bundle(root: &Path) -> Result<ExperienceBundleManifest> {
    if !root.is_dir() {
        anyhow::bail!("bundle is not a directory: {}", root.display());
    }
    let manifest_path = root.join(MANIFEST_FILE);
    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest: ExperienceBundleManifest = serde_json::from_slice(&manifest_bytes)?;
    if manifest.schema_version != BUNDLE_SCHEMA_VERSION {
        anyhow::bail!(
            "incompatible experience bundle schema {}",
            manifest.schema_version
        );
    }
    if manifest.creation_status != "complete" {
        anyhow::bail!("experience bundle is not complete");
    }
    let marker = fs::read_to_string(root.join(COMPLETE_FILE))?;
    if marker.trim() != sha256_bytes(&manifest_bytes) {
        anyhow::bail!("bundle completion marker does not match manifest");
    }
    for file in &manifest.files {
        let relative = safe_relative_path(&file.path)?;
        let path = root.join(relative);
        let metadata = fs::metadata(&path)
            .with_context(|| format!("missing bundle file {}", path.display()))?;
        if metadata.len() != file.bytes || sha256_file(&path)? != file.sha256 {
            anyhow::bail!("bundle checksum mismatch: {}", file.path.display());
        }
    }
    Ok(manifest)
}

fn collect_named(root: &Path, name: &str, output: &mut Vec<PathBuf>) -> Result<()> {
    if !root.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_named(&path, name, output)?;
        } else if path.file_name().and_then(|value| value.to_str()) == Some(name) {
            output.push(path);
        }
    }
    output.sort();
    output.dedup();
    Ok(())
}

fn write_filtered_jsonl(paths: &[PathBuf], target: &Path, begin: u64, end: u64) -> Result<()> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut output = fs::File::create(target)?;
    for path in paths {
        for line in BufReader::new(fs::File::open(path)?).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(&line)
                .with_context(|| format!("parse ledger record from {}", path.display()))?;
            let timestamp = value
                .get("t_ms")
                .or_else(|| value.get("created_at_ms"))
                .and_then(Value::as_u64);
            if timestamp.is_some_and(|timestamp| timestamp >= begin && timestamp <= end) {
                output.write_all(line.as_bytes())?;
                output.write_all(b"\n")?;
            }
        }
    }
    output.sync_all()?;
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

    fn temp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("pete-bundle-{name}-{}", Uuid::new_v4()))
    }

    fn request() -> ExperienceBundleRequest {
        ExperienceBundleRequest {
            pete_id: "pete".into(),
            source_node_id: "motherbrain".into(),
            begin_timestamp_ms: 10,
            end_timestamp_ms: 20,
            event_range: Some(EventRange {
                first_cursor: Some("1".into()),
                last_cursor: Some("2".into()),
            }),
            source_checkpoints: [("ledger".into(), "checkpoint-a".into())]
                .into_iter()
                .collect(),
            software_identity: "commit-a".into(),
            schema_versions: [("ledger".into(), "1".into())].into_iter().collect(),
            active_model_versions: BTreeMap::new(),
            configuration_identity: "config-a".into(),
            calibration_identity: "cal-a".into(),
        }
    }

    #[test]
    fn manifests_are_deterministic_and_repeat_safe() {
        let root = temp("deterministic");
        let source = root.join("source.json");
        fs::create_dir_all(&root).unwrap();
        fs::write(&source, b"{\"value\":1}\n").unwrap();
        let adapter: Box<dyn BundleSourceAdapter> = Box::new(FileExportAdapter {
            source,
            destination: "graph/export.json".into(),
            kind: BundleContentKind::GraphExport,
            required: true,
        });
        let builder = BundleBuilder {
            root: root.join("out"),
        };
        let first = builder.create(&request(), &[adapter]).unwrap();
        let bytes = fs::read(first.join(MANIFEST_FILE)).unwrap();
        let second = builder.create(&request(), &[]).unwrap();
        assert_eq!(first, second);
        assert_eq!(bytes, fs::read(second.join(MANIFEST_FILE)).unwrap());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn partial_bundle_is_never_published() {
        let root = temp("partial");
        let builder = BundleBuilder { root: root.clone() };
        let missing: Box<dyn BundleSourceAdapter> = Box::new(FileExportAdapter {
            source: root.join("missing"),
            destination: "required.json".into(),
            kind: BundleContentKind::Other,
            required: true,
        });
        assert!(builder.create(&request(), &[missing]).is_err());
        let published = fs::read_dir(&root)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .any(|entry| entry.path().extension().and_then(|v| v.to_str()) == Some("bundle"));
        assert!(!published);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn corruption_is_detected() {
        let root = temp("corrupt");
        fs::create_dir_all(&root).unwrap();
        let source = root.join("source");
        fs::write(&source, b"ok").unwrap();
        let builder = BundleBuilder {
            root: root.join("out"),
        };
        let path = builder
            .create(
                &request(),
                &[Box::new(FileExportAdapter {
                    source,
                    destination: "x".into(),
                    kind: BundleContentKind::Other,
                    required: true,
                })],
            )
            .unwrap();
        fs::write(path.join("payload/x"), b"bad").unwrap();
        assert!(validate_bundle(&path).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn incompatible_bundle_schema_is_rejected() {
        let root = temp("schema");
        fs::create_dir_all(&root).unwrap();
        let source = root.join("source");
        fs::write(&source, b"ok").unwrap();
        let builder = BundleBuilder {
            root: root.join("out"),
        };
        let path = builder
            .create(
                &request(),
                &[Box::new(FileExportAdapter {
                    source,
                    destination: "x".into(),
                    kind: BundleContentKind::Other,
                    required: true,
                })],
            )
            .unwrap();
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join(MANIFEST_FILE)).unwrap()).unwrap();
        manifest["schema_version"] = serde_json::json!(999);
        let bytes = serde_json::to_vec_pretty(&manifest).unwrap();
        fs::write(path.join(MANIFEST_FILE), &bytes).unwrap();
        fs::write(
            path.join(COMPLETE_FILE),
            format!("{}\n", sha256_bytes(&bytes)),
        )
        .unwrap();
        assert!(validate_bundle(&path).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
