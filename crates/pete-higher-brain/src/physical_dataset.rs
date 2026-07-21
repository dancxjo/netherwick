use crate::bundle::{BundleContentKind, ExperienceBundleManifest};
use crate::{canonical_json, safe_relative_path, sha256_bytes};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

pub const PREPROCESSING_VERSION: &str = "physical_capture_index/1";
pub const OUTPUT_SCHEMA_VERSION: &str = "physical_experience_dataset/1";
pub const DEPLOYMENT_TARGET: &str = "motherbrain_dataset_library";

const SUPPORTED_MODALITIES: [&str; 9] = [
    "audio",
    "body",
    "calibration",
    "depth",
    "imu",
    "odometry",
    "perception",
    "range",
    "rgb",
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhysicalDatasetRow {
    pub schema_version: u32,
    pub source_bundle_id: String,
    pub capture_id: String,
    pub frame_index: u64,
    pub t_ms: u64,
    pub modalities: BTreeSet<String>,
    pub asset_paths: BTreeMap<String, PathBuf>,
    pub source_frame_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhysicalDatasetEvaluation {
    pub schema_version: u32,
    pub physical_capture_count: usize,
    pub frame_count: usize,
    pub required_modalities: BTreeSet<String>,
    pub modality_frame_counts: BTreeMap<String, usize>,
    pub required_modality_frame_ratio: f64,
    pub mean_required_modality_coverage: f64,
    pub temporal_monotonicity_ratio: f64,
    pub asset_reference_count: usize,
    pub capture_writer_dropped_frames: u64,
}

pub struct PhysicalDatasetOutput {
    pub dataset_path: PathBuf,
    pub evaluation_path: PathBuf,
    pub evaluation: PhysicalDatasetEvaluation,
}

pub fn construct_physical_dataset(
    bundles_root: &Path,
    manifests: &[ExperienceBundleManifest],
    workspace: &Path,
    parameters: &BTreeMap<String, Value>,
) -> Result<PhysicalDatasetOutput> {
    let required_modalities = required_modalities(parameters)?;
    let minimum_ratio = parameters
        .get("minimum_required_modality_frame_ratio")
        .map(|value| {
            value
                .as_f64()
                .context("minimum_required_modality_frame_ratio must be a number")
        })
        .transpose()?
        .unwrap_or(0.0);
    if !(0.0..=1.0).contains(&minimum_ratio) {
        anyhow::bail!("minimum_required_modality_frame_ratio must be between 0 and 1");
    }

    fs::create_dir_all(workspace)?;
    let dataset_path = workspace.join("physical-experience-dataset.jsonl");
    let evaluation_path = workspace.join("physical-experience-evaluation.json");
    let mut dataset = fs::File::create(&dataset_path)?;
    let mut modality_frame_counts = BTreeMap::<String, usize>::new();
    let mut frame_count = 0usize;
    let mut complete_required_frames = 0usize;
    let mut required_modality_observations = 0usize;
    let mut temporal_edges = 0usize;
    let mut monotonic_edges = 0usize;
    let mut asset_reference_count = 0usize;
    let mut capture_writer_dropped_frames = 0u64;

    for manifest in manifests {
        let bundle_root = bundles_root.join(format!("{}.bundle", manifest.bundle_id));
        let capture_manifest_file = manifest
            .files
            .iter()
            .find(|file| file.kind == BundleContentKind::CaptureManifest)
            .context("dataset construction requires a physical capture manifest")?;
        let capture_frames_file = manifest
            .files
            .iter()
            .find(|file| file.kind == BundleContentKind::CaptureFrames)
            .context("dataset construction requires physical capture frames")?;
        let capture_manifest: Value =
            serde_json::from_slice(&fs::read(bundle_root.join(&capture_manifest_file.path))?)?;
        if capture_manifest.get("source").and_then(Value::as_str) != Some("RealRobot") {
            anyhow::bail!(
                "bundle {} is not physical real-robot experience",
                manifest.bundle_id
            );
        }
        let capture_id = capture_manifest
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("physical-capture")
            .to_string();
        capture_writer_dropped_frames = capture_writer_dropped_frames.saturating_add(
            capture_manifest
                .pointer("/writer_health/dropped_frames")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        );

        let frames_path = bundle_root.join(&capture_frames_file.path);
        let capture_payload_root = frames_path
            .parent()
            .context("capture frames have no parent directory")?;
        let mut previous_timestamp = None;
        let mut capture_frames = 0u64;
        for (line_index, line) in BufReader::new(fs::File::open(&frames_path)?)
            .lines()
            .enumerate()
        {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let frame: Value = serde_json::from_str(&line).with_context(|| {
                format!("parse {} line {}", frames_path.display(), line_index + 1)
            })?;
            let frame_index = frame
                .get("index")
                .and_then(Value::as_u64)
                .context("physical capture frame has no index")?;
            let t_ms = frame
                .get("t_ms")
                .and_then(Value::as_u64)
                .context("physical capture frame has no t_ms")?;
            if let Some(previous) = previous_timestamp {
                temporal_edges += 1;
                if t_ms > previous {
                    monotonic_edges += 1;
                }
            }
            previous_timestamp = Some(t_ms);

            let (modalities, asset_paths) = inspect_frame(&frame, capture_payload_root)?;
            for modality in &modalities {
                *modality_frame_counts.entry(modality.clone()).or_default() += 1;
            }
            let required_present = required_modalities
                .iter()
                .filter(|modality| modalities.contains(*modality))
                .count();
            required_modality_observations += required_present;
            if required_present == required_modalities.len() {
                complete_required_frames += 1;
            }
            asset_reference_count += asset_paths.len();
            let row = PhysicalDatasetRow {
                schema_version: 1,
                source_bundle_id: manifest.bundle_id.clone(),
                capture_id: capture_id.clone(),
                frame_index,
                t_ms,
                modalities,
                asset_paths,
                source_frame_sha256: sha256_bytes(&canonical_json(&frame)?),
            };
            dataset.write_all(&canonical_json(&row)?)?;
            dataset.write_all(b"\n")?;
            frame_count += 1;
            capture_frames += 1;
        }
        let declared_frames = capture_manifest
            .get("frame_count")
            .and_then(Value::as_u64)
            .context("physical capture manifest has no frame_count")?;
        if capture_frames != declared_frames {
            anyhow::bail!(
                "physical capture frame count changed inside bundle: manifest={declared_frames}, parsed={capture_frames}"
            );
        }
    }
    dataset.sync_all()?;
    if frame_count == 0 {
        anyhow::bail!("physical dataset contains no frames");
    }

    let required_modality_frame_ratio = complete_required_frames as f64 / frame_count as f64;
    let mean_required_modality_coverage = if required_modalities.is_empty() {
        1.0
    } else {
        required_modality_observations as f64 / (frame_count * required_modalities.len()) as f64
    };
    let temporal_monotonicity_ratio = if temporal_edges == 0 {
        1.0
    } else {
        monotonic_edges as f64 / temporal_edges as f64
    };
    let evaluation = PhysicalDatasetEvaluation {
        schema_version: 1,
        physical_capture_count: manifests.len(),
        frame_count,
        required_modalities,
        modality_frame_counts,
        required_modality_frame_ratio,
        mean_required_modality_coverage,
        temporal_monotonicity_ratio,
        asset_reference_count,
        capture_writer_dropped_frames,
    };
    fs::write(&evaluation_path, serde_json::to_vec_pretty(&evaluation)?)?;
    if required_modality_frame_ratio < minimum_ratio {
        anyhow::bail!(
            "physical dataset required-modality frame ratio {required_modality_frame_ratio:.3} is below minimum {minimum_ratio:.3}"
        );
    }
    Ok(PhysicalDatasetOutput {
        dataset_path,
        evaluation_path,
        evaluation,
    })
}

fn required_modalities(parameters: &BTreeMap<String, Value>) -> Result<BTreeSet<String>> {
    let values = parameters
        .get("required_modalities")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| vec![Value::String("body".into())]);
    let mut modalities = BTreeSet::new();
    for value in values {
        let modality = value
            .as_str()
            .context("required_modalities must contain strings")?;
        if !SUPPORTED_MODALITIES.contains(&modality) {
            anyhow::bail!("unsupported required modality {modality}");
        }
        modalities.insert(modality.to_string());
    }
    if modalities.is_empty() {
        anyhow::bail!("required_modalities must not be empty");
    }
    Ok(modalities)
}

fn inspect_frame(
    frame: &Value,
    capture_payload_root: &Path,
) -> Result<(BTreeSet<String>, BTreeMap<String, PathBuf>)> {
    let mut modalities = BTreeSet::new();
    let snapshot = frame
        .get("snapshot")
        .context("physical frame has no snapshot")?;
    if snapshot.get("body").is_some_and(|value| !value.is_null()) {
        modalities.insert("body".to_string());
    }
    if snapshot
        .pointer("/body/odometry")
        .is_some_and(|value| value.is_object())
    {
        modalities.insert("odometry".to_string());
    }
    if snapshot
        .get("eye_frame")
        .is_some_and(|value| !value.is_null())
    {
        modalities.insert("rgb".to_string());
    }
    if nonempty_array(snapshot.pointer("/kinect/depth_m")) {
        modalities.insert("depth".to_string());
    }
    if nonempty_array(snapshot.pointer("/range/beams"))
        || snapshot
            .pointer("/range/nearest_m")
            .is_some_and(|value| value.is_number())
    {
        modalities.insert("range".to_string());
    }
    if nonempty_array(snapshot.pointer("/imu/orientation"))
        || snapshot
            .pointer("/imu/captured_at_ms")
            .and_then(Value::as_u64)
            .is_some_and(|timestamp| timestamp > 0)
    {
        modalities.insert("imu".to_string());
    }
    if snapshot
        .get("ear_pcm")
        .is_some_and(|value| !value.is_null())
        || nonempty_array(snapshot.pointer("/ear/features"))
        || snapshot
            .pointer("/ear/transcript")
            .is_some_and(|value| !value.is_null())
    {
        modalities.insert("audio".to_string());
    }
    if nonempty_array(snapshot.pointer("/objects/observations")) {
        modalities.insert("perception".to_string());
    }
    if snapshot
        .pointer("/kinect/geometry_calibration")
        .is_some_and(|value| !value.is_null())
    {
        modalities.insert("calibration".to_string());
    }

    let mut asset_paths = BTreeMap::new();
    if let Some(assets) = frame.get("assets").and_then(Value::as_object) {
        for (kind, value) in assets {
            let Some(path) = value.as_str() else {
                continue;
            };
            let relative = safe_relative_path(Path::new(path))?;
            if !capture_payload_root.join(&relative).is_file() {
                anyhow::bail!("physical frame references missing asset {path}");
            }
            let modality = match kind.as_str() {
                "rgb" | "camera" => Some("rgb"),
                "depth" => Some("depth"),
                "lidar" => Some("range"),
                "imu" => Some("imu"),
                "audio" | "transcript" => Some("audio"),
                "perception" => Some("perception"),
                "calibration" => Some("calibration"),
                _ => None,
            };
            if let Some(modality) = modality {
                modalities.insert(modality.to_string());
            }
            asset_paths.insert(kind.clone(), relative);
        }
    }
    Ok((modalities, asset_paths))
}

fn nonempty_array(value: Option<&Value>) -> bool {
    value
        .and_then(Value::as_array)
        .is_some_and(|items| !items.is_empty())
}
