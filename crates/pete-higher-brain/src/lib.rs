//! Cognitive-accelerator data-plane primitives.
//!
//! This crate deliberately has no dependency on `pete-cockpit`: discovery,
//! training authority, bulk transfer, and model activation cannot acquire or
//! reuse a brainstem motion lease. Role-neutral APIs use
//! `CognitiveRole::CognitiveAccelerator`.

pub mod auth;
pub mod bundle;
pub mod candidate;
pub mod capability;
pub mod coordinator;
pub mod discovery;
pub mod failover;
pub mod job;
pub mod transfer;

use anyhow::{Context, Result};
use serde::{de::DeserializeOwned, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const PROTOCOL_VERSION: &str = "netherwick-higher-brain/1";
pub const BUNDLE_SCHEMA_VERSION: u32 = 1;
pub const JOB_SCHEMA_VERSION: u32 = 1;
pub const CANDIDATE_SCHEMA_VERSION: u32 = 1;

pub fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    // All protocol maps use BTreeMap. Struct field order is stable, making this
    // compact representation deterministic without an extra canonical-JSON
    // dependency.
    Ok(serde_json::to_vec(value)?)
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )?)
}

pub fn atomic_write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("state.json");
    let temporary = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    let bytes = serde_json::to_vec_pretty(value)?;
    let mut file = fs::File::create(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    sync_dir(parent)?;
    Ok(())
}

pub fn atomic_replace_symlink(target: &Path, link: &Path) -> Result<()> {
    let parent = link
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.tmp-{}",
        link.file_name().and_then(|v| v.to_str()).unwrap_or("link"),
        std::process::id()
    ));
    if temporary.exists() {
        fs::remove_file(&temporary)?;
    }
    #[cfg(unix)]
    std::os::unix::fs::symlink(target, &temporary)?;
    #[cfg(not(unix))]
    compile_error!("atomic model activation currently requires Unix symlinks");
    fs::rename(&temporary, link)?;
    sync_dir(parent)?;
    Ok(())
}

pub fn sync_dir(path: &Path) -> Result<()> {
    fs::File::open(path)?.sync_all()?;
    Ok(())
}

pub fn safe_relative_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute()
        || path
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir))
    {
        anyhow::bail!("unsafe relative path: {}", path.display());
    }
    Ok(path.to_path_buf())
}
