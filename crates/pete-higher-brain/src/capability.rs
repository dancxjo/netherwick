use crate::PROTOCOL_VERSION;
use anyhow::Result;
use pete_cognition::{
    CapabilityDescriptor, CognitiveCapability, CognitiveProviderDescriptor, CognitiveRole, HostId,
    LatencyEstimate, Locality, ProcessId, ProviderHealth, ProviderHealthState, ProviderId,
    ResourceClass, TrustPolicy,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::process::Command;
use uuid::Uuid;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuProbe {
    pub vendor: String,
    pub model: String,
    pub runtime: String,
    pub usable_memory_bytes: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CapabilityProbe {
    pub architecture: String,
    pub cpu_cores: usize,
    pub total_ram_bytes: u64,
    pub available_ram_bytes: u64,
    pub gpu: Option<GpuProbe>,
    pub storage_capacity_bytes: u64,
    pub storage_available_bytes: u64,
    pub detected_commands: BTreeSet<String>,
    pub load_1m: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AcceleratorCapabilities {
    pub protocol_version: String,
    pub node_id: String,
    pub boot_id: String,
    pub architecture: String,
    pub cpu_cores: usize,
    pub total_ram_bytes: u64,
    pub available_ram_bytes: u64,
    pub gpu: Option<GpuProbe>,
    pub storage_capacity_bytes: u64,
    pub storage_available_bytes: u64,
    pub runtimes: BTreeSet<String>,
    pub job_classes: BTreeSet<String>,
    pub load_1m: f32,
    pub ready: bool,
    pub software_version: String,
    pub schema_versions: BTreeSet<String>,
}

/// Compatibility alias for existing deployment/configuration surfaces. New
/// role-neutral APIs should use `AcceleratorCapabilities`.
pub type ForebrainCapabilities = AcceleratorCapabilities;

impl AcceleratorCapabilities {
    pub fn from_probe(
        node_id: impl Into<String>,
        boot_id: impl Into<String>,
        software_version: impl Into<String>,
        probe: CapabilityProbe,
    ) -> Self {
        let mut runtimes = BTreeSet::new();
        for (command, runtime) in [
            ("cargo", "rust"),
            ("python3", "python"),
            ("docker", "oci"),
            ("nvidia-smi", "cuda"),
            ("rocminfo", "rocm"),
        ] {
            if probe.detected_commands.contains(command) {
                runtimes.insert(runtime.to_string());
            }
        }
        if let Some(gpu) = &probe.gpu {
            if !gpu.runtime.is_empty() {
                runtimes.insert(gpu.runtime.clone());
            }
        }
        let mut job_classes = [
            "fixture_digest",
            "dataset_construction",
            "replay",
            "evaluation",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
        if probe.gpu.is_some() {
            job_classes.extend(
                [
                    "representation_training",
                    "perception_training",
                    "fine_tuning",
                ]
                .into_iter()
                .map(str::to_string),
            );
        }
        let ready = probe.cpu_cores > 0
            && probe.available_ram_bytes > 0
            && probe.storage_available_bytes > 0;
        Self {
            protocol_version: PROTOCOL_VERSION.to_string(),
            node_id: node_id.into(),
            boot_id: boot_id.into(),
            architecture: probe.architecture,
            cpu_cores: probe.cpu_cores,
            total_ram_bytes: probe.total_ram_bytes,
            available_ram_bytes: probe.available_ram_bytes,
            gpu: probe.gpu,
            storage_capacity_bytes: probe.storage_capacity_bytes,
            storage_available_bytes: probe.storage_available_bytes,
            runtimes,
            job_classes,
            load_1m: probe.load_1m,
            ready,
            software_version: software_version.into(),
            schema_versions: ["experience_bundle/1", "job/1", "model_candidate/1"]
                .into_iter()
                .map(str::to_string)
                .collect(),
        }
    }

    pub fn compatible_with(&self, required_schemas: &[String]) -> bool {
        self.protocol_version == PROTOCOL_VERSION
            && required_schemas
                .iter()
                .all(|schema| self.schema_versions.contains(schema))
    }

    pub fn provider_descriptor(
        &self,
        provider_id: impl Into<String>,
        host_id: Option<String>,
        process_id: Option<String>,
        now_ms: u64,
    ) -> CognitiveProviderDescriptor {
        let mut capabilities = Vec::new();
        if self.job_classes.iter().any(|job| {
            matches!(
                job.as_str(),
                "representation_training"
                    | "perception_training"
                    | "fine_tuning"
                    | "hyperparameter_search"
            )
        }) {
            capabilities.push(CapabilityDescriptor {
                capability: CognitiveCapability::TrainModel,
                version: "1".to_string(),
                supports_partial: false,
                performance_confidence: 0.8,
            });
        }
        if self.job_classes.contains("consolidation") {
            capabilities.push(CapabilityDescriptor {
                capability: CognitiveCapability::ConsolidateMemory,
                version: "1".to_string(),
                supports_partial: false,
                performance_confidence: 0.8,
            });
        }
        if self.job_classes.contains("evaluation") || self.job_classes.contains("replay") {
            capabilities.push(CapabilityDescriptor {
                capability: CognitiveCapability::RunCounterfactual,
                version: "1".to_string(),
                supports_partial: false,
                performance_confidence: 0.7,
            });
        }
        CognitiveProviderDescriptor {
            provider_id: ProviderId(provider_id.into()),
            role: CognitiveRole::CognitiveAccelerator,
            host_id: host_id.map(HostId),
            process_id: process_id.map(ProcessId),
            implementation: "pete-higher-brain".to_string(),
            implementation_version: self.software_version.clone(),
            model_version: None,
            capabilities,
            health: ProviderHealth {
                state: if self.ready {
                    ProviderHealthState::Available
                } else {
                    ProviderHealthState::Disconnected
                },
                confidence: 1.0,
                observed_at_ms: now_ms,
                valid_until_ms: now_ms.saturating_add(5_000),
                reason: (!self.ready).then_some("accelerator capability probe is not ready".into()),
            },
            latency: LatencyEstimate::default(),
            resource_class: if self.gpu.is_some() {
                ResourceClass::Accelerated
            } else {
                ResourceClass::GeneralPurpose
            },
            locality: Locality::LocalNetwork,
            trust: TrustPolicy::TrustedProvider,
            energy_cost: 0.5,
            network_cost: 0.3,
        }
    }
}

pub fn detect_local(workspace: &Path, node_id: String) -> Result<AcceleratorCapabilities> {
    let architecture = std::env::consts::ARCH.to_string();
    let cpu_cores = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let (total_ram_bytes, available_ram_bytes) = read_meminfo().unwrap_or_default();
    let (storage_capacity_bytes, storage_available_bytes) =
        disk_space(workspace).unwrap_or_default();
    let detected_commands = ["cargo", "python3", "docker", "nvidia-smi", "rocminfo"]
        .into_iter()
        .filter(|command| command_exists(command))
        .map(str::to_string)
        .collect();
    let gpu = detect_gpu();
    let load_1m = fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|text| text.split_whitespace().next()?.parse().ok())
        .unwrap_or_default();
    let boot_id = fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .map(|value| value.trim().to_string())
        .unwrap_or_else(|_| Uuid::new_v4().to_string());
    Ok(AcceleratorCapabilities::from_probe(
        node_id,
        boot_id,
        env!("CARGO_PKG_VERSION"),
        CapabilityProbe {
            architecture,
            cpu_cores,
            total_ram_bytes,
            available_ram_bytes,
            gpu,
            storage_capacity_bytes,
            storage_available_bytes,
            detected_commands,
            load_1m,
        },
    ))
}

fn command_exists(command: &str) -> bool {
    Command::new("sh")
        .args(["-c", "command -v -- \"$1\" >/dev/null 2>&1", "sh", command])
        .status()
        .is_ok_and(|status| status.success())
}

fn read_meminfo() -> Option<(u64, u64)> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    let value = |name: &str| {
        text.lines().find_map(|line| {
            let (key, rest) = line.split_once(':')?;
            (key == name)
                .then(|| rest.split_whitespace().next()?.parse::<u64>().ok())
                .flatten()
        })
    };
    Some((value("MemTotal")? * 1024, value("MemAvailable")? * 1024))
}

fn disk_space(path: &Path) -> Option<(u64, u64)> {
    let output = Command::new("df")
        .args(["-Pk", "--"])
        .arg(path)
        .output()
        .ok()?;
    let line = String::from_utf8(output.stdout)
        .ok()?
        .lines()
        .last()?
        .to_string();
    let columns = line.split_whitespace().collect::<Vec<_>>();
    Some((
        columns.get(1)?.parse::<u64>().ok()? * 1024,
        columns.get(3)?.parse::<u64>().ok()? * 1024,
    ))
}

fn detect_gpu() -> Option<GpuProbe> {
    if let Ok(output) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        if output.status.success() {
            if let Some((model, memory)) = String::from_utf8_lossy(&output.stdout)
                .lines()
                .next()?
                .rsplit_once(',')
            {
                return Some(GpuProbe {
                    vendor: "nvidia".into(),
                    model: model.trim().into(),
                    runtime: "cuda".into(),
                    usable_memory_bytes: memory.trim().parse::<u64>().ok()? * 1024 * 1024,
                });
            }
        }
    }
    if command_exists("rocminfo") {
        return Some(GpuProbe {
            vendor: "amd".into(),
            model: "detected by rocminfo".into(),
            runtime: "rocm".into(),
            usable_memory_bytes: 0,
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_probe() -> CapabilityProbe {
        CapabilityProbe {
            architecture: "x86_64".into(),
            cpu_cores: 8,
            total_ram_bytes: 16_000,
            available_ram_bytes: 8_000,
            storage_capacity_bytes: 100_000,
            storage_available_bytes: 50_000,
            ..Default::default()
        }
    }

    #[test]
    fn cpu_only_is_ready_without_gpu_jobs() {
        let mut probe = base_probe();
        probe.detected_commands.insert("cargo".into());
        let caps = ForebrainCapabilities::from_probe("node", "boot", "v", probe);
        assert!(caps.ready);
        assert!(caps.runtimes.contains("rust"));
        assert!(!caps.job_classes.contains("perception_training"));
    }

    #[test]
    fn gpu_fixture_advertises_runtime_and_heavy_jobs() {
        let mut probe = base_probe();
        probe.gpu = Some(GpuProbe {
            vendor: "nvidia".into(),
            model: "fixture".into(),
            runtime: "cuda".into(),
            usable_memory_bytes: 8_000,
        });
        let caps = ForebrainCapabilities::from_probe("node", "boot", "v", probe);
        assert!(caps.runtimes.contains("cuda"));
        assert!(caps.job_classes.contains("perception_training"));
    }

    #[test]
    fn missing_runtime_fixture_does_not_fail_detection() {
        let caps = ForebrainCapabilities::from_probe("node", "boot", "v", base_probe());
        assert!(caps.ready);
        assert!(caps.runtimes.is_empty());
    }
}
