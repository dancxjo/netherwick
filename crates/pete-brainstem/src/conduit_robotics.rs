//! Describe-only Conduit-facing inventory for the current brainstem body.
//!
//! This is deliberately a compiled description, not a live capability report.
//! Calling [`describe_only`] neither opens a device nor obtains a possession,
//! motion, network, or safety authority. Live observations and any actuation
//! remain behind their existing firmware and host-specific boundaries.

use serde::Serialize;

use crate::{build_identity, capabilities};

/// A distinct observation, command, acknowledgement, safe-outcome, or host
/// boundary that a Conduit provider may describe. These identifiers are
/// inventory vocabulary only until a selected host binds them in an exact
/// plan; they are not grants or installed providers.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RoboticsBoundary {
    pub id: &'static str,
    pub contract: &'static str,
    pub role: &'static str,
    pub units: &'static str,
    pub frame: &'static str,
    pub clock: &'static str,
    pub uncertainty: &'static str,
    pub freshness: &'static str,
    pub required_authority: &'static str,
    pub effect: &'static str,
    pub host_requirement: &'static str,
    pub source: &'static str,
}

/// Compiled, effect-free inventory for the active brainstem build and body.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct RoboticsInventory {
    pub firmware_build_id: &'static str,
    pub firmware_git_commit: &'static str,
    pub firmware_target: &'static str,
    pub body_kind: &'static str,
    pub boundaries: &'static [RoboticsBoundary],
}

const BOUNDARIES: &[RoboticsBoundary] = &[
    RoboticsBoundary {
        id: "netherwick/brainstem-description",
        contract: "conduit.netherwick/robotics/brainstem-description",
        role: "describe-only compiled body and firmware inventory",
        units: "declared limits only",
        frame: "body",
        clock: "not-live",
        uncertainty: "not-applicable",
        freshness: "compiled build identity; not a current observation",
        required_authority: "none",
        effect: "none",
        host_requirement: "selected Pete brainstem firmware build",
        source: "pete-brainstem::current",
    },
    RoboticsBoundary {
        id: "netherwick/create-observation",
        contract: "conduit.netherwick/robotics/create-observation",
        role: "Create OI telemetry observation",
        units: "mm, mm/s, mrad, mrad/s, mV, mA, mAh",
        frame: "Create body frame",
        clock: "brainstem monotonic milliseconds",
        uncertainty: "packet decoding and sensor-specific bounds",
        freshness: "last complete packet and responsive timeout",
        required_authority: "none",
        effect: "none",
        host_requirement: "Create UART and current complete sensor packet",
        source: "runtime/lifecycle.rs and status/telemetry.rs",
    },
    RoboticsBoundary {
        id: "netherwick/motion-command",
        contract: "conduit.netherwick/robotics/motion-command",
        role: "bounded velocity-envelope command",
        units: "linear mm/s; angular mrad/s; TTL ms",
        frame: "Create body frame",
        clock: "brainstem monotonic milliseconds",
        uncertainty: "command acknowledgement is not a safe outcome",
        freshness: "finite command TTL from body limits",
        required_authority: "current possession plus exact motion grant",
        effect: "motion request only",
        host_requirement: "responsive Create, safety clear, and non-charging body",
        source: "runtime/motion.rs and runtime/safety.rs",
    },
    RoboticsBoundary {
        id: "netherwick/safe-motion-outcome",
        contract: "conduit.netherwick/robotics/safe-motion-outcome",
        role: "acknowledgement, stop, refusal, or terminal safety outcome",
        units: "terminal reason and stopped velocity",
        frame: "Create body frame",
        clock: "brainstem monotonic milliseconds",
        uncertainty: "does not claim physical HIL equivalence",
        freshness: "emitted for the bounded command lifecycle",
        required_authority: "none to observe; command authority to cause",
        effect: "none",
        host_requirement: "brainstem runtime evidence stream",
        source: "runtime/execution.rs and events.rs",
    },
    RoboticsBoundary {
        id: "netherwick/sensor-observation",
        contract: "conduit.netherwick/robotics/sensor-observation",
        role: "IMU, contact, cliff, wheel-drop, IR, battery, and odometry observation",
        units: "mrad, mrad/s, mm/s2, mm, mV, mA, mAh",
        frame: "body and declared IMU mounting frame",
        clock: "brainstem monotonic milliseconds",
        uncertainty: "source-specific calibration and orientation confidence",
        freshness: "per-sample timestamp and age",
        required_authority: "none",
        effect: "none",
        host_requirement: "selected sensor driver; unsupported sensors remain absent",
        source: "runtime/sensors.rs and drivers/imu.rs",
    },
    RoboticsBoundary {
        id: "netherwick/possession-lease",
        contract: "conduit.netherwick/robotics/possession-lease",
        role: "exclusive, revocable controller lease",
        units: "lease generation and expiry milliseconds",
        frame: "not-applicable",
        clock: "cockpit/session monotonic clock",
        uncertainty: "possession is not motor, safety, or network authority",
        freshness: "lease expiry and heartbeat",
        required_authority: "motherbrain possession policy",
        effect: "lease lifecycle only",
        host_requirement: "cockpit/motherbrain session provider",
        source: "pete-cockpit::cockpit::possession",
    },
    RoboticsBoundary {
        id: "netherwick/network-observation",
        contract: "conduit.netherwick/network/provider-observation",
        role: "isolated AP, DHCP, ICMP, and DNS-SD provider observation",
        units: "IPv4 address, leases, packets, and TTL milliseconds",
        frame: "network interface cyw43/ap0",
        clock: "firmware monotonic milliseconds",
        uncertainty: "attachment is not enrollment, possession, or authority",
        freshness: "one-second observation validity",
        required_authority: "none",
        effect: "none",
        host_requirement: "initialized CYW43 AP and service tasks",
        source: "conduit_network.rs",
    },
    RoboticsBoundary {
        id: "netherwick/higher-brain-provider",
        contract: "conduit.netherwick/robotics/higher-brain-provider",
        role: "optional cognitive provider description",
        units: "provider-specific bounded request and response limits",
        frame: "not-applicable",
        clock: "provider-declared monotonic clock",
        uncertainty: "selection is not authority or actuator admission",
        freshness: "provider health and capability validity interval",
        required_authority: "provider-specific grant",
        effect: "provider-specific; no implicit motion effect",
        host_requirement: "separately initialized higher-brain provider",
        source: "pete-higher-brain::capability",
    },
    RoboticsBoundary {
        id: "netherwick/cockpit-session",
        contract: "conduit.netherwick/robotics/cockpit-session",
        role: "operator session and requested control boundary",
        units: "session IDs, sequence numbers, and TTL milliseconds",
        frame: "not-applicable",
        clock: "cockpit/session monotonic clock",
        uncertainty: "transport authentication is not a motion grant",
        freshness: "session handshake and heartbeat",
        required_authority: "operator policy and explicit possession where required",
        effect: "requested control only",
        host_requirement: "cockpit transport/session implementation",
        source: "pete-cockpit protocol and possession session",
    },
];

/// Returns the current compiled inventory without interacting with hardware.
#[must_use]
pub fn describe_only() -> RoboticsInventory {
    let capabilities = capabilities::current();
    RoboticsInventory {
        firmware_build_id: build_identity::CURRENT.build_id,
        firmware_git_commit: build_identity::CURRENT.git_commit,
        firmware_target: build_identity::CURRENT.build_target,
        body_kind: capabilities.body_kind,
        boundaries: BOUNDARIES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describe_only_inventory_is_effect_free_and_complete() {
        let inventory = describe_only();
        assert!(!inventory.firmware_build_id.is_empty());
        assert_eq!(inventory.body_kind, "create_oi");
        assert_eq!(inventory.boundaries.len(), 9);
        assert!(inventory.boundaries.iter().all(|boundary| {
            boundary.required_authority != "none" || boundary.effect == "none"
        }));
    }

    #[test]
    fn observation_command_and_safe_outcome_are_not_one_boundary() {
        let inventory = describe_only();
        let ids: [&str; 3] = [
            "netherwick/create-observation",
            "netherwick/motion-command",
            "netherwick/safe-motion-outcome",
        ];
        for id in ids {
            assert_eq!(
                inventory
                    .boundaries
                    .iter()
                    .filter(|item| item.id == id)
                    .count(),
                1
            );
        }
    }

    #[test]
    fn no_boundary_turns_network_or_possession_into_motion_authority() {
        let inventory = describe_only();
        for id in [
            "netherwick/network-observation",
            "netherwick/possession-lease",
        ] {
            let boundary = inventory
                .boundaries
                .iter()
                .find(|item| item.id == id)
                .unwrap();
            assert_ne!(boundary.effect, "motion request only");
            assert_ne!(
                boundary.required_authority,
                "current possession plus exact motion grant"
            );
        }
    }
}
