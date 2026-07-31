//! Truthful Conduit inventory and live observations for the Pico W network lane.
//!
//! Compiled inventory is always safe to inspect. Runnable observations exist
//! only after the real CYW43/AP services have initialized and remain coupled to
//! the firmware Wi-Fi state. This module conveys no robot or motion authority.

use portable_atomic::{AtomicU32, AtomicU8, Ordering};
use serde::Serialize;

use crate::{build_identity, status};

pub const PROVIDER_IDS: [&str; 4] = [
    "net/wifi/access-point",
    "net/dhcp/server",
    "net/reachability",
    "net/dns-sd",
];
pub const INTERFACE: &str = "cyw43/ap0";
pub const AP_ADDRESS: [u8; 4] = [192, 168, 4, 1];
pub const OBSERVATION_VALIDITY_MS: u32 = 1_000;

const STATE_UNINITIALIZED: u8 = 0;
const STATE_INITIALIZED: u8 = 1;
const STATE_LOST: u8 = 2;

static PROVIDER_STATE: AtomicU8 = AtomicU8::new(STATE_UNINITIALIZED);
static INTERFACE_GENERATION: AtomicU32 = AtomicU32::new(0);
static INITIALIZED_AT_MS: AtomicU32 = AtomicU32::new(0);
static DEVICE_ID: AtomicU32 = AtomicU32::new(0);
static BOOT_ID: AtomicU32 = AtomicU32::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct CompiledNetworkInventory {
    pub firmware_build_id: &'static str,
    pub firmware_git_commit: &'static str,
    pub firmware_target: &'static str,
    pub interface: &'static str,
    pub providers: [&'static str; 4],
    pub access_point_address: [u8; 4],
    pub client_capacity: u8,
    pub routing: bool,
    pub bridging: bool,
    pub nat: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct InitializedNetworkObservation {
    pub device_id: u32,
    pub boot_id: u32,
    pub interface_generation: u32,
    pub initialized_at_ms: u32,
    pub observed_at_ms: u32,
    pub valid_until_ms: u32,
    pub providers: [&'static str; 4],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct PicoNetworkReport {
    pub inventory: CompiledNetworkInventory,
    pub observation: Option<InitializedNetworkObservation>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ObservationUnavailable {
    NotInitialized,
    ProviderLost,
}

pub const fn compiled_inventory() -> CompiledNetworkInventory {
    CompiledNetworkInventory {
        firmware_build_id: build_identity::CURRENT.build_id,
        firmware_git_commit: build_identity::CURRENT.git_commit,
        firmware_target: build_identity::CURRENT.build_target,
        interface: INTERFACE,
        providers: PROVIDER_IDS,
        access_point_address: AP_ADDRESS,
        client_capacity: 8,
        routing: false,
        bridging: false,
        nat: false,
    }
}

pub const fn describe_only() -> PicoNetworkReport {
    PicoNetworkReport {
        inventory: compiled_inventory(),
        observation: None,
    }
}

/// Observe the current firmware-owned provider state.
///
/// The caller supplies only the current monotonic time. Availability, identity,
/// interface generation, and freshness are read from firmware state.
pub fn runnable(now_ms: u32) -> Result<PicoNetworkReport, ObservationUnavailable> {
    runnable_from_state(now_ms, status::network_services_ready())
}

fn runnable_from_state(
    now_ms: u32,
    network_services_ready: bool,
) -> Result<PicoNetworkReport, ObservationUnavailable> {
    let provider_state = PROVIDER_STATE.load(Ordering::Acquire);
    if provider_state != STATE_INITIALIZED {
        return Err(if provider_state == STATE_LOST {
            ObservationUnavailable::ProviderLost
        } else {
            ObservationUnavailable::NotInitialized
        });
    }
    if !network_services_ready {
        return Err(ObservationUnavailable::ProviderLost);
    }

    Ok(PicoNetworkReport {
        inventory: compiled_inventory(),
        observation: Some(InitializedNetworkObservation {
            device_id: DEVICE_ID.load(Ordering::Acquire),
            boot_id: BOOT_ID.load(Ordering::Acquire),
            interface_generation: INTERFACE_GENERATION.load(Ordering::Acquire),
            initialized_at_ms: INITIALIZED_AT_MS.load(Ordering::Acquire),
            observed_at_ms: now_ms,
            valid_until_ms: now_ms.wrapping_add(OBSERVATION_VALIDITY_MS),
            providers: PROVIDER_IDS,
        }),
    })
}

#[cfg(feature = "pico-w")]
pub(crate) fn observe_initialized(device_id: u32, boot_id: u32, now_ms: u32) {
    record_initialized(device_id, boot_id, now_ms);
}

fn record_initialized(device_id: u32, boot_id: u32, now_ms: u32) {
    DEVICE_ID.store(device_id, Ordering::Release);
    BOOT_ID.store(boot_id, Ordering::Release);
    INITIALIZED_AT_MS.store(now_ms, Ordering::Release);
    INTERFACE_GENERATION.fetch_add(1, Ordering::AcqRel);
    PROVIDER_STATE.store(STATE_INITIALIZED, Ordering::Release);
}

#[cfg(feature = "pico-w")]
pub(crate) fn observe_lost() {
    record_lost();
}

fn record_lost() {
    PROVIDER_STATE.store(STATE_LOST, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_only_never_claims_a_live_observation() {
        let report = describe_only();
        assert_eq!(report.observation, None);
        assert_eq!(report.inventory.providers, PROVIDER_IDS);
        assert!(!report.inventory.routing);
        assert!(!report.inventory.bridging);
        assert!(!report.inventory.nat);
    }

    #[test]
    fn public_runnable_api_has_no_authority_or_freshness_inputs() {
        let signature: fn(u32) -> Result<PicoNetworkReport, ObservationUnavailable> = runnable;
        let _ = signature;
    }

    #[test]
    fn runnable_observation_is_bound_to_initialization_and_loss() {
        PROVIDER_STATE.store(STATE_UNINITIALIZED, Ordering::Release);
        assert_eq!(
            runnable_from_state(10, true),
            Err(ObservationUnavailable::NotInitialized)
        );

        record_initialized(41, 73, 20);
        let observation = runnable_from_state(25, true).unwrap().observation.unwrap();
        assert_eq!(observation.device_id, 41);
        assert_eq!(observation.boot_id, 73);
        assert_eq!(observation.initialized_at_ms, 20);
        assert_eq!(observation.observed_at_ms, 25);
        assert_eq!(observation.valid_until_ms, 1_025);
        assert_eq!(
            runnable_from_state(26, false),
            Err(ObservationUnavailable::ProviderLost)
        );

        record_lost();
        assert_eq!(
            runnable_from_state(27, true),
            Err(ObservationUnavailable::ProviderLost)
        );
    }
}
