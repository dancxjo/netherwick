//! Real, effect-free Conduit robotics-profile implementations for Pete hosts.
//!
//! Linux and Pico W describe the same semantic profile through distinct exact
//! implementation, artifact, provider-bundle, build, target, and compiled
//! capability facts. Description performs no device or network I/O and creates
//! no boot observation, path observation, enrollment, possession, authority,
//! role promotion, plan activation, or actuation.

use serde::Serialize;

use crate::{build_identity, capabilities};

pub const SCHEMA_VERSION: u32 = 0;
pub const PROFILE_CONTRACT: &str = "conduit.robotics/profile";
pub const PROFILE_CONTRACT_HASH: &str =
    "sha256:8186fdd2be75eb23f4343f15b82a336bad951c19c803308162d150b45f67fd1e";
pub const PROFILE_IDENTITY: &str =
    "sha256:f4ccc52b6c40ef5106a0752cf8e2d926bbcb1bcebe5b19287c32733315767029";
pub const LINUX_HOST_PROFILE_IDENTITY: &str =
    "sha256:09a016735e026bbc9fc15a0af4f8023fae2dd67a31f172749f7da46037cdfb52";
pub const PICO_HOST_PROFILE_IDENTITY: &str =
    "sha256:46bb4f92740d3389b3e2e0be7f2a042d4c7f317fd134d9ec33133e0ea6cd1fd9";

const LINUX_IMPLEMENTATION_HASH: &str =
    "sha256:ca1176529b26599dfbcc16d5b9155ef979274d2bb3eccb6540375f1ec7e414a5";
const PICO_IMPLEMENTATION_HASH: &str =
    "sha256:9fb84fad0d18e673a7de4ed2c2676d4adedb210ffb6c9f57b53f82b965ae60ea";
const LINUX_ARTIFACT_HASH: &str =
    "sha256:5f661f42416ce1c99ad008da5242d6dbe1ac9277b794c07e2bf490a09d2b284f";
const PICO_ARTIFACT_HASH: &str =
    "sha256:76a9f275e22a56a2ed9be1ab59091d756d0509ba0f6d9d2396b60b145a2fef3e";
const LINUX_PROVIDER_BUNDLE_HASH: &str =
    "sha256:a09ded7cbcce5bddee388041d6425f0ea674d1b92254c4f1aa7141fbf0dceeae";
const PICO_PROVIDER_BUNDLE_HASH: &str =
    "sha256:d044a691cf807732418c4dd8517dc378c6a95fcf0cfecccc3e00aaa99e0595cb";
const DESCRIBE_ADAPTER_HASH: &str =
    "sha256:a446ad531079a0f5e8622302ed6f59592d2594b4eb5ed314060f09fede2f4c51";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ValueRole {
    Observation,
    Command,
    Acknowledgement,
    SafeOutcome,
    Possession,
    Terminal,
    Fault,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ValueTypeDescriptor {
    pub role: ValueRole,
    pub id: &'static str,
    pub semantic_hash: &'static str,
}

pub const VALUE_TYPES: [ValueTypeDescriptor; 7] = [
    ValueTypeDescriptor {
        role: ValueRole::Observation,
        id: "conduit.robotics/observation",
        semantic_hash: "sha256:f9a5f7a64f9c590d32e9e768ebf85c99f403b1e7b5fa6f818ad00ad626b3a61d",
    },
    ValueTypeDescriptor {
        role: ValueRole::Command,
        id: "conduit.robotics/command",
        semantic_hash: "sha256:91b6e322d47863cde4769c1173a601bd41d1128f246266105227812dc19f4b79",
    },
    ValueTypeDescriptor {
        role: ValueRole::Acknowledgement,
        id: "conduit.robotics/acknowledgement",
        semantic_hash: "sha256:5ca5190fc94997c45fe7ccf437f71a105609deeff6deef08fbcec1185e355981",
    },
    ValueTypeDescriptor {
        role: ValueRole::SafeOutcome,
        id: "conduit.robotics/safe-outcome",
        semantic_hash: "sha256:a66a4fdc328f6e1ce01e2ee72a5001cd67e6c0685b0217122ee6033650ec6a4a",
    },
    ValueTypeDescriptor {
        role: ValueRole::Possession,
        id: "conduit.robotics/possession",
        semantic_hash: "sha256:5ea91a556e18dbd01790acb97ee3635f0dd439939dfd3bb871f889b834873806",
    },
    ValueTypeDescriptor {
        role: ValueRole::Terminal,
        id: "conduit.robotics/terminal",
        semantic_hash: "sha256:d73fc8ee718b65c13c6b42c09d764ed43c176c436e8c729dd161e86992da6dae",
    },
    ValueTypeDescriptor {
        role: ValueRole::Fault,
        id: "conduit.robotics/fault",
        semantic_hash: "sha256:1c129536beafc20aac526f0ef48088e0fa57d5c730a3554c85a0c558165efbff",
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct QuantityDescriptor {
    pub id: &'static str,
    pub units: &'static str,
    pub frame: &'static str,
    pub clock: &'static str,
    pub uncertainty_bound: u64,
    pub maximum_age_ticks: u64,
}

const fn quantity(
    id: &'static str,
    units: &'static str,
    frame: &'static str,
    uncertainty_bound: u64,
    maximum_age_ticks: u64,
) -> QuantityDescriptor {
    QuantityDescriptor {
        id,
        units,
        frame,
        clock: "brainstem-monotonic-milliseconds",
        uncertainty_bound,
        maximum_age_ticks,
    }
}

pub const QUANTITIES: [QuantityDescriptor; 8] = [
    quantity("linear-velocity", "mm-per-second", "create-body", 25, 250),
    quantity(
        "angular-velocity",
        "milliradians-per-second",
        "create-body",
        50,
        250,
    ),
    quantity("distance", "millimetres", "create-body", 10, 250),
    quantity("heading", "milliradians", "create-body", 25, 250),
    quantity(
        "acceleration",
        "millimetres-per-second-squared",
        "imu-mount",
        100,
        100,
    ),
    quantity("voltage", "millivolts", "electrical", 50, 1_000),
    quantity("current", "milliamperes", "electrical", 100, 1_000),
    quantity("charge", "milliampere-hours", "battery", 100, 5_000),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct SafetyEnvelope {
    pub clock: &'static str,
    pub command_ttl_ticks: u32,
    pub maximum_linear_velocity_mm_per_second: u32,
    pub maximum_angular_velocity_milliradians_per_second: u32,
    pub maximum_command_queue: u16,
    pub possession_requirement: &'static str,
    pub motion_authority: &'static str,
    pub stop_requirement: &'static str,
    pub emergency_stop_requirement: &'static str,
    pub not_charging_requirement: &'static str,
    pub charging_interlock_requirement: &'static str,
    pub inhibit_clear_requirement: &'static str,
    pub capability_requirement: &'static str,
}

pub const SAFETY_ENVELOPE: SafetyEnvelope = SafetyEnvelope {
    clock: "brainstem-monotonic-milliseconds",
    command_ttl_ticks: 250,
    maximum_linear_velocity_mm_per_second: 500,
    maximum_angular_velocity_milliradians_per_second: 2_000,
    maximum_command_queue: 1,
    possession_requirement: "netherwick/requirement/current-possession",
    motion_authority: "netherwick/authority/motion",
    stop_requirement: "netherwick/requirement/bounded-stop",
    emergency_stop_requirement: "netherwick/requirement/emergency-stop",
    not_charging_requirement: "netherwick/requirement/not-charging",
    charging_interlock_requirement: "netherwick/requirement/charging-interlock",
    inhibit_clear_requirement: "netherwick/requirement/safety-inhibit-clear",
    capability_requirement: "netherwick/capability/create-motion",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct LogicalRelationship {
    pub id: &'static str,
    pub source_entity: &'static str,
    pub target_entity: &'static str,
    pub role: &'static str,
    pub allowed_carriers: &'static [&'static str],
}

pub const LOGICAL_RELATIONSHIPS: [LogicalRelationship; 2] = [
    LogicalRelationship {
        id: "motherbrain-to-brainstem",
        source_entity: "netherwick/motherbrain",
        target_entity: "netherwick/brainstem",
        role: "command-and-observation",
        allowed_carriers: &["usb", "ethernet", "wifi"],
    },
    LogicalRelationship {
        id: "brainstem-to-body",
        source_entity: "netherwick/brainstem",
        target_entity: "netherwick/create-body",
        role: "create-oi-control",
        allowed_carriers: &["uart"],
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum InventoryState {
    Unsupported,
    Compiled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct CompiledProvider {
    pub capability: &'static str,
    pub state: InventoryState,
    pub initialized_observation: Option<&'static str>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct CarrierCandidate {
    pub relationship: &'static str,
    pub carrier: &'static str,
    pub provider: &'static str,
    pub compiled: bool,
    pub admitted: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct GenericImplementationManifest {
    pub contract: &'static str,
    pub contract_hash: &'static str,
    pub implementation: &'static str,
    pub implementation_hash: &'static str,
    pub artifact: &'static str,
    pub artifact_descriptor_hash: &'static str,
    pub provider_bundle: &'static str,
    pub provider_bundle_hash: &'static str,
    pub adapter: &'static str,
    pub adapter_hash: &'static str,
    pub execution_mode: &'static str,
    pub host_profile: &'static str,
    pub host_profile_identity: &'static str,
    pub target: &'static str,
    pub source_commit: &'static str,
    pub current_build_id: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct IdentityFacts {
    pub entity: &'static str,
    pub boot: Option<&'static str>,
    pub role: &'static str,
    pub possession: Option<&'static str>,
    pub authority: Option<&'static str>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ContinuationDescriptor {
    pub katra_role: &'static str,
    pub organism_runtime_checkpoint: &'static str,
    pub loaded: bool,
    pub promoted: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct DescribeEffectAudit {
    pub device_open_count: u16,
    pub network_join_count: u16,
    pub relay_count: u16,
    pub possession_count: u16,
    pub role_promotion_count: u16,
    pub plan_activation_count: u16,
    pub actuation_count: u16,
}

impl DescribeEffectAudit {
    pub const NONE: Self = Self {
        device_open_count: 0,
        network_join_count: 0,
        relay_count: 0,
        possession_count: 0,
        role_promotion_count: 0,
        plan_activation_count: 0,
        actuation_count: 0,
    };

    #[must_use]
    pub const fn is_effect_free(self) -> bool {
        self.device_open_count == 0
            && self.network_join_count == 0
            && self.relay_count == 0
            && self.possession_count == 0
            && self.role_promotion_count == 0
            && self.plan_activation_count == 0
            && self.actuation_count == 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct SourceEvidence {
    pub subject: &'static str,
    pub source: &'static str,
    pub test: &'static str,
}

pub const SOURCE_EVIDENCE: [SourceEvidence; 10] = [
    SourceEvidence {
        subject: "body-and-build",
        source: "pete-brainstem/src/body.rs;pete-brainstem/src/build_identity.rs",
        test: "body::gpio_tests;build_identity::tests",
    },
    SourceEvidence {
        subject: "create-observation",
        source: "pete-brainstem/src/runtime/lifecycle.rs;pete-brainstem/src/status/telemetry.rs",
        test: "runtime::lifecycle::tests;status::tests",
    },
    SourceEvidence {
        subject: "motion-command",
        source: "pete-brainstem/src/runtime/motion.rs;pete-brainstem/src/runtime/safety.rs",
        test: "runtime::motion::tests;runtime::safety::tests",
    },
    SourceEvidence {
        subject: "acknowledgement-and-outcome",
        source: "pete-brainstem/src/runtime/execution.rs;pete-brainstem/src/events.rs",
        test: "runtime::execution::tests;events::tests",
    },
    SourceEvidence {
        subject: "sensors",
        source: "pete-brainstem/src/runtime/sensors.rs;pete-brainstem/src/drivers/imu.rs",
        test: "runtime::sensors::tests;drivers::imu::tests",
    },
    SourceEvidence {
        subject: "possession",
        source: "pete-cockpit/src/cockpit/possession",
        test: "pete-cockpit/src/lib_tests/possession.rs",
    },
    SourceEvidence {
        subject: "network",
        source: "pete-brainstem/src/conduit_network.rs",
        test: "conduit_network::tests",
    },
    SourceEvidence {
        subject: "linux-control",
        source: "pete-brainstem/src/rpi5_control.rs",
        test: "rpi5_control::tests",
    },
    SourceEvidence {
        subject: "higher-brain-role",
        source: "pete-higher-brain/src/capability.rs;pete-higher-brain/src/failover.rs",
        test: "pete-higher-brain capability and failover tests",
    },
    SourceEvidence {
        subject: "cockpit-session",
        source: "pete-cockpit/src/cockpit/possession;pete-cockpit-protocol/src",
        test: "pete-cockpit possession and protocol tests",
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct DescribeOnlyReport {
    pub schema: &'static str,
    pub schema_version: u32,
    pub profile_contract: &'static str,
    pub profile_contract_hash: &'static str,
    pub profile_identity: &'static str,
    pub body_kind: &'static str,
    pub implementation: GenericImplementationManifest,
    pub value_types: [ValueTypeDescriptor; 7],
    pub quantities: [QuantityDescriptor; 8],
    pub safety: SafetyEnvelope,
    pub logical_relationships: [LogicalRelationship; 2],
    pub compiled_providers: [CompiledProvider; 4],
    pub carrier_candidates: [CarrierCandidate; 3],
    pub current_path_observations: [&'static str; 0],
    pub identities: IdentityFacts,
    pub continuation: ContinuationDescriptor,
    pub hidden_device_handles: u16,
    pub secret_or_sensitive_topology_fields: u16,
    pub effect_audit: DescribeEffectAudit,
    pub source_evidence: [SourceEvidence; 10],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RenderError {
    BufferTooSmall,
    InvalidUtf8,
}

/// Serializes one already-constructed descriptor into caller-owned storage.
/// Serialization performs no host observation or external I/O.
pub fn render_json<'a>(
    report: &DescribeOnlyReport,
    buffer: &'a mut [u8],
) -> Result<&'a str, RenderError> {
    let length =
        serde_json_core::to_slice(report, buffer).map_err(|_| RenderError::BufferTooSmall)?;
    core::str::from_utf8(&buffer[..length]).map_err(|_| RenderError::InvalidUtf8)
}

/// Returns the current build's real describe implementation without I/O.
#[must_use]
pub fn describe_only() -> DescribeOnlyReport {
    if cfg!(feature = "rpi5") {
        linux_report()
    } else {
        pico_w_report()
    }
}

/// Returns the Linux candidate's exact static report without opening UART,
/// USB, networking, cockpit, or higher-brain services.
#[must_use]
pub fn linux_report() -> DescribeOnlyReport {
    report(
        GenericImplementationManifest {
            contract: PROFILE_CONTRACT,
            contract_hash: PROFILE_CONTRACT_HASH,
            implementation: "netherwick/implementation/pete-linux-robotics-describe",
            implementation_hash: LINUX_IMPLEMENTATION_HASH,
            artifact: "netherwick/artifact/pete-brainstem-linux",
            artifact_descriptor_hash: LINUX_ARTIFACT_HASH,
            provider_bundle: "netherwick/provider/pete-linux-robotics-describe",
            provider_bundle_hash: LINUX_PROVIDER_BUNDLE_HASH,
            adapter: "conduit.adapter/robotics-profile-describe",
            adapter_hash: DESCRIBE_ADAPTER_HASH,
            execution_mode: "describe-only",
            host_profile: "netherwick/pete-linux-describe-only",
            host_profile_identity: LINUX_HOST_PROFILE_IDENTITY,
            target: "aarch64-unknown-linux-gnu",
            source_commit: build_identity::CURRENT.git_commit,
            current_build_id: build_identity::CURRENT.build_id,
        },
        [
            compiled("netherwick/capability/usb", InventoryState::Compiled),
            compiled("netherwick/capability/cyw43", InventoryState::Unsupported),
            compiled("netherwick/capability/network", InventoryState::Compiled),
            compiled(
                "netherwick/capability/create-motion",
                InventoryState::Compiled,
            ),
        ],
        [
            carrier("usb", "netherwick/provider/linux-usb", true),
            carrier("ethernet", "netherwick/provider/linux-ethernet", true),
            carrier("wifi", "netherwick/provider/linux-wifi", false),
        ],
    )
}

/// Returns the Pico W candidate's exact static report without reading live
/// CYW43, USB, UART, session, possession, or control state.
#[must_use]
pub fn pico_w_report() -> DescribeOnlyReport {
    let cyw43_compiled = cfg!(feature = "pico-w");
    report(
        GenericImplementationManifest {
            contract: PROFILE_CONTRACT,
            contract_hash: PROFILE_CONTRACT_HASH,
            implementation: "netherwick/implementation/pete-pico-robotics-describe",
            implementation_hash: PICO_IMPLEMENTATION_HASH,
            artifact: "netherwick/artifact/pete-brainstem-pico-w",
            artifact_descriptor_hash: PICO_ARTIFACT_HASH,
            provider_bundle: "netherwick/provider/pete-pico-robotics-describe",
            provider_bundle_hash: PICO_PROVIDER_BUNDLE_HASH,
            adapter: "conduit.adapter/robotics-profile-describe",
            adapter_hash: DESCRIBE_ADAPTER_HASH,
            execution_mode: "describe-only",
            host_profile: "netherwick/pete-pico-w-describe-only",
            host_profile_identity: PICO_HOST_PROFILE_IDENTITY,
            target: "thumbv6m-none-eabi",
            source_commit: build_identity::CURRENT.git_commit,
            current_build_id: build_identity::CURRENT.build_id,
        },
        [
            compiled("netherwick/capability/usb", InventoryState::Compiled),
            compiled(
                "netherwick/capability/cyw43",
                if cyw43_compiled {
                    InventoryState::Compiled
                } else {
                    InventoryState::Unsupported
                },
            ),
            compiled(
                "netherwick/capability/network",
                if cyw43_compiled {
                    InventoryState::Compiled
                } else {
                    InventoryState::Unsupported
                },
            ),
            compiled(
                "netherwick/capability/create-motion",
                InventoryState::Compiled,
            ),
        ],
        [
            carrier("usb", "netherwick/provider/pico-usb", true),
            carrier("ethernet", "netherwick/provider/pico-ethernet", false),
            carrier("wifi", "netherwick/provider/pico-wifi", cyw43_compiled),
        ],
    )
}

const fn compiled(capability: &'static str, state: InventoryState) -> CompiledProvider {
    CompiledProvider {
        capability,
        state,
        initialized_observation: None,
    }
}

const fn carrier(
    carrier: &'static str,
    provider: &'static str,
    compiled: bool,
) -> CarrierCandidate {
    CarrierCandidate {
        relationship: "motherbrain-to-brainstem",
        carrier,
        provider,
        compiled,
        admitted: false,
    }
}

fn report(
    implementation: GenericImplementationManifest,
    compiled_providers: [CompiledProvider; 4],
    carrier_candidates: [CarrierCandidate; 3],
) -> DescribeOnlyReport {
    DescribeOnlyReport {
        schema: "conduit.netherwick-robotics-profile-report",
        schema_version: SCHEMA_VERSION,
        profile_contract: PROFILE_CONTRACT,
        profile_contract_hash: PROFILE_CONTRACT_HASH,
        profile_identity: PROFILE_IDENTITY,
        body_kind: capabilities::current().body_kind,
        implementation,
        value_types: VALUE_TYPES,
        quantities: QUANTITIES,
        safety: SAFETY_ENVELOPE,
        logical_relationships: LOGICAL_RELATIONSHIPS,
        compiled_providers,
        carrier_candidates,
        current_path_observations: [],
        identities: IdentityFacts {
            entity: "netherwick/entity/pete-brainstem",
            boot: None,
            role: "netherwick/role/katra-custodian",
            possession: None,
            authority: None,
        },
        continuation: ContinuationDescriptor {
            katra_role: "netherwick/role/katra-custodian",
            organism_runtime_checkpoint: "netherwick/checkpoint/organism-runtime-continuation",
            loaded: false,
            promoted: false,
        },
        hidden_device_handles: 0,
        secret_or_sensitive_topology_fields: 0,
        effect_audit: DescribeEffectAudit::NONE,
        source_evidence: SOURCE_EVIDENCE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linux_and_pico_are_distinct_implementations_of_one_exact_contract() {
        let linux = linux_report();
        let pico = pico_w_report();
        assert_eq!(linux.profile_contract, pico.profile_contract);
        assert_eq!(linux.profile_contract_hash, pico.profile_contract_hash);
        assert_eq!(linux.profile_identity, pico.profile_identity);
        assert_ne!(
            linux.implementation.implementation,
            pico.implementation.implementation
        );
        assert_ne!(linux.implementation.artifact, pico.implementation.artifact);
        assert_ne!(
            linux.implementation.provider_bundle,
            pico.implementation.provider_bundle
        );
        assert_ne!(
            linux.implementation.host_profile_identity,
            pico.implementation.host_profile_identity
        );
    }

    #[test]
    fn value_roles_and_physical_quantity_metadata_are_distinct_and_complete() {
        let report = describe_only();
        assert_eq!(report.value_types.len(), 7);
        for role in [
            ValueRole::Observation,
            ValueRole::Command,
            ValueRole::Acknowledgement,
            ValueRole::SafeOutcome,
            ValueRole::Possession,
            ValueRole::Terminal,
            ValueRole::Fault,
        ] {
            assert_eq!(
                report
                    .value_types
                    .iter()
                    .filter(|descriptor| descriptor.role == role)
                    .count(),
                1
            );
        }
        assert_ne!(report.value_types[2].id, report.value_types[3].id);
        assert_ne!(report.value_types[5].id, report.value_types[6].id);
        assert!(report.quantities.iter().all(|quantity| {
            !quantity.units.is_empty()
                && !quantity.frame.is_empty()
                && !quantity.clock.is_empty()
                && quantity.maximum_age_ticks > 0
        }));
    }

    #[test]
    fn finite_safety_envelope_is_compatible_with_current_body_limits() {
        let current = capabilities::current();
        assert!(SAFETY_ENVELOPE.command_ttl_ticks >= current.min_ttl_ms);
        assert!(SAFETY_ENVELOPE.command_ttl_ticks <= current.max_ttl_ms);
        assert!(
            SAFETY_ENVELOPE.maximum_linear_velocity_mm_per_second
                <= u32::from(current.max_linear_mm_s.unsigned_abs())
        );
        assert!(
            SAFETY_ENVELOPE.maximum_angular_velocity_milliradians_per_second
                <= u32::from(current.max_angular_mrad_s.unsigned_abs())
        );
        assert_eq!(SAFETY_ENVELOPE.maximum_command_queue, 1);
        for requirement in [
            SAFETY_ENVELOPE.possession_requirement,
            SAFETY_ENVELOPE.motion_authority,
            SAFETY_ENVELOPE.stop_requirement,
            SAFETY_ENVELOPE.emergency_stop_requirement,
            SAFETY_ENVELOPE.not_charging_requirement,
            SAFETY_ENVELOPE.charging_interlock_requirement,
            SAFETY_ENVELOPE.inhibit_clear_requirement,
            SAFETY_ENVELOPE.capability_requirement,
        ] {
            assert!(!requirement.is_empty());
        }
    }

    #[test]
    fn compiled_inventory_is_not_initialized_or_admitted_state() {
        for report in [linux_report(), pico_w_report()] {
            assert!(report
                .compiled_providers
                .iter()
                .all(|provider| provider.initialized_observation.is_none()));
            assert!(report
                .carrier_candidates
                .iter()
                .all(|candidate| !candidate.admitted));
            assert!(report.current_path_observations.is_empty());
            assert!(report.identities.boot.is_none());
            assert!(report.identities.possession.is_none());
            assert!(report.identities.authority.is_none());
        }
    }

    #[test]
    fn describe_only_is_redacted_and_has_no_effects_or_loaded_continuation() {
        for report in [linux_report(), pico_w_report(), describe_only()] {
            assert_eq!(report.hidden_device_handles, 0);
            assert_eq!(report.secret_or_sensitive_topology_fields, 0);
            assert!(report.effect_audit.is_effect_free());
            assert!(!report.continuation.loaded);
            assert!(!report.continuation.promoted);
        }
    }

    #[test]
    fn inventory_points_to_exact_current_source_and_tests() {
        let report = describe_only();
        assert_eq!(report.source_evidence.len(), 10);
        assert!(report.source_evidence.iter().all(|evidence| {
            !evidence.subject.is_empty() && !evidence.source.is_empty() && !evidence.test.is_empty()
        }));
        assert_eq!(
            report.implementation.source_commit,
            build_identity::CURRENT.git_commit
        );
        assert_eq!(
            report.implementation.current_build_id,
            build_identity::CURRENT.build_id
        );
    }

    #[test]
    fn reports_are_inspectable_in_caller_owned_storage_without_secret_fields() {
        let mut storage = [0_u8; 24 * 1024];
        let json = render_json(&describe_only(), &mut storage).unwrap();
        assert!(json.starts_with("{\"schema\":\"conduit.netherwick-robotics-profile-report\""));
        assert!(json.contains(PROFILE_IDENTITY));
        assert!(json.contains("\"current_path_observations\":[]"));
        assert!(json.contains("\"possession\":null"));
        assert!(json.contains("\"authority\":null"));
        for forbidden in [
            "\"device_handle\":",
            "\"credential\":",
            "\"bearer_token\":",
            "\"private_endpoint\":",
        ] {
            assert!(!json.contains(forbidden));
        }
    }
}
