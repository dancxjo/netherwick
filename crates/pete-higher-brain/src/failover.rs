//! Host-aware control-path failover.
//!
//! This module decides which path a Linux host should use and whether it is
//! eligible to ask the existing brainstem possession surface for control. It
//! deliberately cannot grant authority: a `TakeoverCandidate` must still win
//! an atomic acquisition from the brainstem.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostKind {
    Motherbrain,
    Forebrain,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostIdentity {
    pub node_id: String,
    pub boot_id: String,
    pub kind: HostKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathKind {
    DirectEthernet,
    OrdinaryWifi,
    BrainstemTransit,
    BrainstemUsb,
    BrainstemWifi,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdvertisedRole {
    Controlling,
    Subordinate,
    Candidate,
    Unavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectivityState {
    PreferredPath,
    SearchingOrdinaryNetwork,
    JoiningBrainstemNetwork,
    PeerReachableViaBrainstem,
    DegradedButControlled,
    NoControllerObserved,
    TakeoverCandidate,
    ControllingFallback,
    HandbackPending,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ControllerStatus {
    /// A transport failure or stale response is not evidence that nobody owns
    /// the body.
    Unknown,
    Uncontrolled {
        observed_at_ms: u64,
        generation: u64,
    },
    Active {
        node_id: String,
        boot_id: String,
        observed_at_ms: u64,
        lease_expires_at_ms: u64,
        generation: u64,
    },
}

impl ControllerStatus {
    fn fresh_at(&self, now_ms: u64, freshness_ms: u64) -> bool {
        match self {
            Self::Unknown => false,
            Self::Uncontrolled { observed_at_ms, .. } => {
                now_ms.saturating_sub(*observed_at_ms) <= freshness_ms
            }
            Self::Active {
                observed_at_ms,
                lease_expires_at_ms,
                ..
            } => {
                now_ms.saturating_sub(*observed_at_ms) <= freshness_ms
                    && now_ms < *lease_expires_at_ms
            }
        }
    }

    fn active_owner(&self, now_ms: u64, freshness_ms: u64) -> Option<(&str, &str)> {
        if !self.fresh_at(now_ms, freshness_ms) {
            return None;
        }
        match self {
            Self::Active {
                node_id, boot_id, ..
            } => Some((node_id, boot_id)),
            _ => None,
        }
    }

    fn fresh_uncontrolled(&self, now_ms: u64, freshness_ms: u64) -> Option<u64> {
        if !self.fresh_at(now_ms, freshness_ms) {
            return None;
        }
        match self {
            Self::Uncontrolled { generation, .. } => Some(*generation),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcquisitionResult {
    Granted,
    Refused,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConnectivityObservation {
    pub now_ms: u64,
    pub direct_peer_reachable: bool,
    pub ordinary_wifi_peer_reachable: bool,
    pub brainstem_network_associated: bool,
    pub peer_reachable_via_brainstem: bool,
    pub brainstem_usb_healthy: bool,
    pub brainstem_wifi_healthy: bool,
    pub peer_identity: Option<HostIdentity>,
    pub peer_observed_at_ms: Option<u64>,
    pub controller: ControllerStatus,
    pub acquisition_result: Option<AcquisitionResult>,
    /// Controller generation echoed by the completed atomic acquisition.
    /// Results without the generation of this machine's outstanding attempt
    /// are ignored as stale transport replies.
    pub acquisition_generation: Option<u64>,
    pub safe_handoff_boundary: bool,
    pub active_safe_sequence: bool,
}

impl ConnectivityObservation {
    pub fn disconnected(now_ms: u64) -> Self {
        Self {
            now_ms,
            direct_peer_reachable: false,
            ordinary_wifi_peer_reachable: false,
            brainstem_network_associated: false,
            peer_reachable_via_brainstem: false,
            brainstem_usb_healthy: false,
            brainstem_wifi_healthy: false,
            peer_identity: None,
            peer_observed_at_ms: None,
            controller: ControllerStatus::Unknown,
            acquisition_result: None,
            acquisition_generation: None,
            safe_handoff_boundary: false,
            active_safe_sequence: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum FailoverAction {
    SearchPeer { path: PathKind },
    JoinBrainstemNetwork,
    UsePath { path: PathKind },
    ObserveAuthoritativeController,
    AttemptAtomicAcquisition { expected_generation: u64 },
    StartBodyFacingRole,
    ContinueBodyFacingRole,
    WaitForSafeHandoffBoundary,
    ReleasePossession,
    RemainSubordinate,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailoverEvent {
    pub at_ms: u64,
    pub from: ConnectivityState,
    pub to: ConnectivityState,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FailoverDecision {
    pub state: ConnectivityState,
    pub state_age_ms: u64,
    pub role: AdvertisedRole,
    pub active_path: Option<PathKind>,
    pub controller_node_id: Option<String>,
    pub peer_visible: bool,
    pub actions: Vec<FailoverAction>,
    pub events: Vec<FailoverEvent>,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmergencyService {
    PeerDiscovery,
    PeerHealth,
    RoleCoordination,
    PossessionStatus,
    PossessionAcquire,
    Handback,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct FailoverConfig {
    pub enabled: bool,
    /// Takeover defaults off until both hosts and the brainstem possession
    /// surface are deployed with stable identities.
    pub takeover_enabled: bool,
    pub preferred_paths: Vec<PathKind>,
    pub peer_names: Vec<String>,
    pub brainstem_ssid: String,
    /// Name of an external secret, never the secret itself.
    pub brainstem_credential_ref: Option<String>,
    pub controller_freshness_ms: u64,
    pub peer_freshness_ms: u64,
    pub takeover_grace_ms: u64,
    pub ordinary_search_ms: u64,
    pub retry_backoff_ms: u64,
    pub path_hysteresis_ms: u64,
    pub permitted_emergency_services: BTreeMap<EmergencyService, u16>,
    pub handback_automatic: bool,
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            takeover_enabled: false,
            preferred_paths: vec![
                PathKind::DirectEthernet,
                PathKind::OrdinaryWifi,
                PathKind::BrainstemTransit,
            ],
            peer_names: vec![
                "motherbrain.local".into(),
                "motherbrain.pete.internal".into(),
            ],
            brainstem_ssid: "pete-<device>".into(),
            brainstem_credential_ref: None,
            controller_freshness_ms: 2_000,
            peer_freshness_ms: 3_000,
            takeover_grace_ms: 5_000,
            ordinary_search_ms: 3_000,
            retry_backoff_ms: 1_000,
            path_hysteresis_ms: 2_000,
            permitted_emergency_services: [
                (EmergencyService::PeerDiscovery, 5353),
                (EmergencyService::PeerHealth, 8787),
                (EmergencyService::RoleCoordination, 8787),
                (EmergencyService::PossessionStatus, 80),
                (EmergencyService::PossessionAcquire, 80),
                (EmergencyService::Handback, 8787),
            ]
            .into_iter()
            .collect(),
            handback_automatic: true,
        }
    }
}

impl FailoverConfig {
    pub fn validate(&self) -> Result<()> {
        if self.controller_freshness_ms == 0
            || self.peer_freshness_ms == 0
            || self.takeover_grace_ms == 0
            || self.ordinary_search_ms == 0
            || self.retry_backoff_ms == 0
            || self.path_hysteresis_ms == 0
        {
            bail!("failover timing thresholds must be nonzero");
        }
        if self.peer_names.is_empty() || self.brainstem_ssid.trim().is_empty() {
            bail!("failover discovery names and brainstem SSID must be explicit");
        }
        if self.preferred_paths
            != [
                PathKind::DirectEthernet,
                PathKind::OrdinaryWifi,
                PathKind::BrainstemTransit,
            ]
        {
            bail!("failover path order must be direct Ethernet, ordinary Wi-Fi, then brainstem transit");
        }
        if self
            .brainstem_credential_ref
            .as_deref()
            .is_some_and(|value| value.contains('=') || value.contains(' '))
        {
            bail!("brainstem_credential_ref must name an external secret, not contain one");
        }
        let required = [
            EmergencyService::PeerDiscovery,
            EmergencyService::PeerHealth,
            EmergencyService::RoleCoordination,
            EmergencyService::PossessionStatus,
            EmergencyService::PossessionAcquire,
            EmergencyService::Handback,
        ];
        if required
            .iter()
            .any(|service| !self.permitted_emergency_services.contains_key(service))
        {
            bail!("all bounded emergency services must have an explicit port");
        }
        Ok(())
    }

    pub fn emergency_port(&self, service: EmergencyService) -> Option<u16> {
        self.permitted_emergency_services.get(&service).copied()
    }

    /// Bulk transfer is intentionally absent. Callers cannot turn a missing
    /// service into an allow-all rule.
    pub fn emergency_ports(&self) -> BTreeSet<u16> {
        self.permitted_emergency_services
            .values()
            .copied()
            .collect()
    }
}

pub struct HostFailover {
    pub identity: HostIdentity,
    pub config: FailoverConfig,
    state: ConnectivityState,
    state_since_ms: u64,
    outage_since_ms: Option<u64>,
    uncontrolled_since_ms: Option<u64>,
    preferred_seen_since_ms: Option<u64>,
    last_acquisition_attempt_ms: Option<u64>,
    last_acquisition_generation: Option<u64>,
    active_path: Option<PathKind>,
}

impl HostFailover {
    pub fn new(identity: HostIdentity, config: FailoverConfig, now_ms: u64) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            identity,
            config,
            state: ConnectivityState::PreferredPath,
            state_since_ms: now_ms,
            outage_since_ms: None,
            uncontrolled_since_ms: None,
            preferred_seen_since_ms: None,
            last_acquisition_attempt_ms: None,
            last_acquisition_generation: None,
            active_path: None,
        })
    }

    pub fn state(&self) -> ConnectivityState {
        self.state
    }

    pub fn tick(&mut self, observation: &ConnectivityObservation) -> FailoverDecision {
        if !self.config.enabled {
            return self.decision(
                observation,
                AdvertisedRole::Unavailable,
                vec![FailoverAction::RemainSubordinate],
                "failover disabled",
                Vec::new(),
            );
        }
        match self.identity.kind {
            HostKind::Motherbrain => self.tick_motherbrain(observation),
            HostKind::Forebrain => self.tick_forebrain(observation),
        }
    }

    fn peer_observation_fresh(&self, observation: &ConnectivityObservation) -> bool {
        observation
            .peer_observed_at_ms
            .is_some_and(|observed_at_ms| {
                observation.now_ms.saturating_sub(observed_at_ms) <= self.config.peer_freshness_ms
            })
    }

    fn tick_motherbrain(&mut self, observation: &ConnectivityObservation) -> FailoverDecision {
        let self_controls = observation
            .controller
            .active_owner(observation.now_ms, self.config.controller_freshness_ms)
            .is_some_and(|(node, boot)| {
                node == self.identity.node_id && boot == self.identity.boot_id
            });
        let other_controls = observation
            .controller
            .active_owner(observation.now_ms, self.config.controller_freshness_ms)
            .is_some_and(|(node, boot)| {
                node != self.identity.node_id || boot != self.identity.boot_id
            });
        if other_controls {
            if self.peer_observation_fresh(observation)
                && (observation.direct_peer_reachable || observation.peer_reachable_via_brainstem)
            {
                let mut actions = vec![FailoverAction::RemainSubordinate];
                if self.config.handback_automatic {
                    actions.push(FailoverAction::WaitForSafeHandoffBoundary);
                }
                return self.transition_decision(
                    observation,
                    ConnectivityState::HandbackPending,
                    AdvertisedRole::Subordinate,
                    actions,
                    "returning motherbrain observed fallback controller; waiting for coordinated handback",
                );
            }
            return self.transition_decision(
                observation,
                ConnectivityState::DegradedButControlled,
                AdvertisedRole::Subordinate,
                vec![FailoverAction::RemainSubordinate],
                "another fresh controller remains authoritative",
            );
        }
        if observation.brainstem_usb_healthy {
            return self.tick_motherbrain_path(
                observation,
                PathKind::BrainstemUsb,
                ConnectivityState::PreferredPath,
                self_controls,
                "motherbrain body link is healthy; forebrain visibility does not affect control role",
            );
        }
        if observation.brainstem_wifi_healthy {
            return self.tick_motherbrain_path(
                observation,
                PathKind::BrainstemWifi,
                ConnectivityState::DegradedButControlled,
                self_controls,
                "USB transport failed; motherbrain retains its role over brainstem Wi-Fi",
            );
        }
        self.active_path = None;
        self.transition_decision(
            observation,
            ConnectivityState::DegradedButControlled,
            if self_controls {
                AdvertisedRole::Controlling
            } else {
                AdvertisedRole::Unavailable
            },
            vec![FailoverAction::ObserveAuthoritativeController],
            "body transport unavailable; role is not reassigned by transport loss",
        )
    }

    fn tick_motherbrain_path(
        &mut self,
        observation: &ConnectivityObservation,
        path: PathKind,
        state: ConnectivityState,
        self_controls: bool,
        controlling_reason: &str,
    ) -> FailoverDecision {
        let now = observation.now_ms;
        self.active_path = Some(path);
        let mut path_actions = vec![FailoverAction::UsePath { path }];
        if self_controls {
            self.last_acquisition_attempt_ms = None;
            self.last_acquisition_generation = None;
            path_actions.push(FailoverAction::ContinueBodyFacingRole);
            return self.transition_decision(
                observation,
                state,
                AdvertisedRole::Controlling,
                path_actions,
                controlling_reason,
            );
        }
        let generation = observation
            .controller
            .fresh_uncontrolled(now, self.config.controller_freshness_ms);
        let result_matches_attempt = observation.acquisition_generation.is_some()
            && observation.acquisition_generation == self.last_acquisition_generation;
        if observation.acquisition_result == Some(AcquisitionResult::Granted)
            && result_matches_attempt
        {
            self.last_acquisition_attempt_ms = None;
            self.last_acquisition_generation = None;
            path_actions.push(FailoverAction::StartBodyFacingRole);
            return self.transition_decision(
                observation,
                state,
                AdvertisedRole::Controlling,
                path_actions,
                "atomic possession acquisition granted to the preferred motherbrain",
            );
        }
        if observation.acquisition_result == Some(AcquisitionResult::Refused)
            && result_matches_attempt
        {
            self.last_acquisition_attempt_ms = Some(now);
            self.last_acquisition_generation = None;
            path_actions.push(FailoverAction::RemainSubordinate);
            return self.transition_decision(
                observation,
                state,
                AdvertisedRole::Subordinate,
                path_actions,
                "possession acquisition was refused; motherbrain will not actuate",
            );
        }
        let Some(generation) = generation else {
            path_actions.push(FailoverAction::ObserveAuthoritativeController);
            return self.transition_decision(
                observation,
                state,
                AdvertisedRole::Candidate,
                path_actions,
                "motherbrain transport is healthy but controller status is not fresh enough to acquire",
            );
        };
        if self
            .last_acquisition_attempt_ms
            .is_some_and(|last| now.saturating_sub(last) < self.config.retry_backoff_ms)
        {
            path_actions.push(FailoverAction::ObserveAuthoritativeController);
            return self.transition_decision(
                observation,
                state,
                AdvertisedRole::Candidate,
                path_actions,
                "motherbrain possession retry is within bounded backoff",
            );
        }
        self.last_acquisition_attempt_ms = Some(now);
        self.last_acquisition_generation = Some(generation);
        path_actions.push(FailoverAction::AttemptAtomicAcquisition {
            expected_generation: generation,
        });
        self.transition_decision(
            observation,
            state,
            AdvertisedRole::Candidate,
            path_actions,
            "fresh no-controller status permits the preferred motherbrain to acquire atomically",
        )
    }

    fn tick_forebrain(&mut self, observation: &ConnectivityObservation) -> FailoverDecision {
        let now = observation.now_ms;
        let active_owner = observation
            .controller
            .active_owner(now, self.config.controller_freshness_ms)
            .map(|(node, boot)| (node.to_owned(), boot.to_owned()));
        let self_controls = active_owner.as_ref().is_some_and(|(node, boot)| {
            node == &self.identity.node_id && boot == &self.identity.boot_id
        });
        let motherbrain_controls = active_owner.as_ref().is_some_and(|(node, boot)| {
            node != &self.identity.node_id || boot != &self.identity.boot_id
        });

        if self_controls {
            self.last_acquisition_attempt_ms = None;
            self.last_acquisition_generation = None;
            let motherbrain_returned = observation
                .peer_identity
                .as_ref()
                .is_some_and(|peer| peer.kind == HostKind::Motherbrain)
                && self.peer_observation_fresh(observation)
                && (observation.direct_peer_reachable
                    || observation.ordinary_wifi_peer_reachable
                    || observation.peer_reachable_via_brainstem);
            if motherbrain_returned {
                let actions = if observation.safe_handoff_boundary
                    && !observation.active_safe_sequence
                    && self.config.handback_automatic
                {
                    vec![FailoverAction::ReleasePossession]
                } else {
                    vec![FailoverAction::WaitForSafeHandoffBoundary]
                };
                return self.transition_decision(
                    observation,
                    ConnectivityState::HandbackPending,
                    AdvertisedRole::Controlling,
                    actions,
                    "preferred motherbrain returned; preserve one controller until a safe handoff boundary",
                );
            }
            self.active_path = Some(PathKind::BrainstemWifi);
            return self.transition_decision(
                observation,
                ConnectivityState::ControllingFallback,
                AdvertisedRole::Controlling,
                vec![FailoverAction::ContinueBodyFacingRole],
                "forebrain holds the authoritative fallback lease",
            );
        }

        if observation.direct_peer_reachable && self.peer_observation_fresh(observation) {
            if self.preferred_seen_since_ms.is_none() {
                self.preferred_seen_since_ms = Some(now);
            }
            if self.state == ConnectivityState::PreferredPath
                || now.saturating_sub(self.preferred_seen_since_ms.unwrap_or(now))
                    >= self.config.path_hysteresis_ms
            {
                self.outage_since_ms = None;
                self.uncontrolled_since_ms = None;
                self.active_path = Some(PathKind::DirectEthernet);
                return self.transition_decision(
                    observation,
                    ConnectivityState::PreferredPath,
                    AdvertisedRole::Subordinate,
                    vec![
                        FailoverAction::UsePath {
                            path: PathKind::DirectEthernet,
                        },
                        FailoverAction::RemainSubordinate,
                    ],
                    "direct Ethernet peer path is stable",
                );
            }
            let mut actions = self
                .active_path
                .map(|path| vec![FailoverAction::UsePath { path }])
                .unwrap_or_default();
            actions.push(FailoverAction::RemainSubordinate);
            return self.transition_decision(
                observation,
                ConnectivityState::DegradedButControlled,
                AdvertisedRole::Subordinate,
                actions,
                "direct Ethernet returned but has not met the failback hysteresis interval",
            );
        } else {
            self.preferred_seen_since_ms = None;
        }

        let outage_since = *self.outage_since_ms.get_or_insert(now);
        if observation.ordinary_wifi_peer_reachable && self.peer_observation_fresh(observation) {
            self.active_path = Some(PathKind::OrdinaryWifi);
            return self.transition_decision(
                observation,
                ConnectivityState::DegradedButControlled,
                AdvertisedRole::Subordinate,
                vec![
                    FailoverAction::UsePath {
                        path: PathKind::OrdinaryWifi,
                    },
                    FailoverAction::RemainSubordinate,
                ],
                "motherbrain rediscovered on ordinary Wi-Fi; control hierarchy is unchanged",
            );
        }
        if now.saturating_sub(outage_since) < self.config.ordinary_search_ms {
            self.active_path = None;
            return self.transition_decision(
                observation,
                ConnectivityState::SearchingOrdinaryNetwork,
                AdvertisedRole::Subordinate,
                vec![FailoverAction::SearchPeer {
                    path: PathKind::OrdinaryWifi,
                }],
                "direct path failed; ordinary Wi-Fi discovery precedes emergency association",
            );
        }
        if !observation.brainstem_network_associated {
            self.active_path = None;
            return self.transition_decision(
                observation,
                ConnectivityState::JoiningBrainstemNetwork,
                AdvertisedRole::Subordinate,
                vec![FailoverAction::JoinBrainstemNetwork],
                "ordinary paths exhausted; joining the brainstem network grants connectivity only",
            );
        }
        if observation.peer_reachable_via_brainstem && self.peer_observation_fresh(observation) {
            self.active_path = Some(PathKind::BrainstemTransit);
            self.uncontrolled_since_ms = None;
            return self.transition_decision(
                observation,
                ConnectivityState::PeerReachableViaBrainstem,
                AdvertisedRole::Subordinate,
                vec![
                    FailoverAction::UsePath {
                        path: PathKind::BrainstemTransit,
                    },
                    FailoverAction::RemainSubordinate,
                ],
                "motherbrain is reachable through brainstem transit and remains the body-facing host",
            );
        }
        if motherbrain_controls {
            self.active_path = Some(PathKind::BrainstemTransit);
            self.uncontrolled_since_ms = None;
            return self.transition_decision(
                observation,
                ConnectivityState::DegradedButControlled,
                AdvertisedRole::Subordinate,
                vec![
                    FailoverAction::ObserveAuthoritativeController,
                    FailoverAction::RemainSubordinate,
                ],
                "peer is not visible but the authoritative control surface reports a live controller",
            );
        }
        let Some(generation) = observation
            .controller
            .fresh_uncontrolled(now, self.config.controller_freshness_ms)
        else {
            self.uncontrolled_since_ms = None;
            return self.transition_decision(
                observation,
                ConnectivityState::NoControllerObserved,
                AdvertisedRole::Subordinate,
                vec![FailoverAction::ObserveAuthoritativeController],
                "controller status is unknown or stale; absence of peer visibility cannot authorize takeover",
            );
        };
        let uncontrolled_since = *self.uncontrolled_since_ms.get_or_insert(now);
        if !self.config.takeover_enabled
            || now.saturating_sub(uncontrolled_since) < self.config.takeover_grace_ms
        {
            return self.transition_decision(
                observation,
                ConnectivityState::NoControllerObserved,
                AdvertisedRole::Subordinate,
                vec![FailoverAction::ObserveAuthoritativeController],
                if self.config.takeover_enabled {
                    "fresh no-controller status observed; takeover grace interval has not elapsed"
                } else {
                    "fallback takeover is disabled"
                },
            );
        }
        let result_matches_attempt = observation.acquisition_generation.is_some()
            && observation.acquisition_generation == self.last_acquisition_generation;
        if observation.acquisition_result == Some(AcquisitionResult::Granted)
            && result_matches_attempt
        {
            self.active_path = Some(PathKind::BrainstemWifi);
            self.last_acquisition_attempt_ms = None;
            self.last_acquisition_generation = None;
            return self.transition_decision(
                observation,
                ConnectivityState::ControllingFallback,
                AdvertisedRole::Controlling,
                vec![FailoverAction::StartBodyFacingRole],
                "atomic brainstem acquisition granted; forebrain may start the ordinary body-facing role",
            );
        }
        if observation.acquisition_result == Some(AcquisitionResult::Refused)
            && result_matches_attempt
        {
            self.last_acquisition_attempt_ms = Some(now);
            self.last_acquisition_generation = None;
            return self.transition_decision(
                observation,
                ConnectivityState::NoControllerObserved,
                AdvertisedRole::Subordinate,
                vec![FailoverAction::RemainSubordinate],
                "atomic acquisition was refused; no ordinary actuation is permitted",
            );
        }
        if self
            .last_acquisition_attempt_ms
            .is_some_and(|last| now.saturating_sub(last) < self.config.retry_backoff_ms)
        {
            return self.transition_decision(
                observation,
                ConnectivityState::TakeoverCandidate,
                AdvertisedRole::Candidate,
                vec![FailoverAction::ObserveAuthoritativeController],
                "takeover retry is within bounded backoff",
            );
        }
        self.last_acquisition_attempt_ms = Some(now);
        self.last_acquisition_generation = Some(generation);
        self.transition_decision(
            observation,
            ConnectivityState::TakeoverCandidate,
            AdvertisedRole::Candidate,
            vec![FailoverAction::AttemptAtomicAcquisition {
                expected_generation: generation,
            }],
            "peer paths are exhausted and fresh authoritative status reports no controller",
        )
    }

    fn transition_decision(
        &mut self,
        observation: &ConnectivityObservation,
        next: ConnectivityState,
        role: AdvertisedRole,
        actions: Vec<FailoverAction>,
        reason: &str,
    ) -> FailoverDecision {
        let mut events = Vec::new();
        if next != self.state {
            events.push(FailoverEvent {
                at_ms: observation.now_ms,
                from: self.state,
                to: next,
                reason: reason.to_owned(),
            });
            self.state = next;
            self.state_since_ms = observation.now_ms;
        }
        self.decision(observation, role, actions, reason, events)
    }

    fn decision(
        &self,
        observation: &ConnectivityObservation,
        role: AdvertisedRole,
        actions: Vec<FailoverAction>,
        reason: &str,
        events: Vec<FailoverEvent>,
    ) -> FailoverDecision {
        FailoverDecision {
            state: self.state,
            state_age_ms: observation.now_ms.saturating_sub(self.state_since_ms),
            role,
            active_path: self.active_path,
            controller_node_id: observation
                .controller
                .active_owner(observation.now_ms, self.config.controller_freshness_ms)
                .map(|(node, _)| node.to_owned()),
            peer_visible: self.peer_observation_fresh(observation)
                && (observation.direct_peer_reachable
                    || observation.ordinary_wifi_peer_reachable
                    || observation.peer_reachable_via_brainstem),
            actions,
            events,
            reason: reason.to_owned(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct AcceptanceCheck {
    pub scenario: &'static str,
    pub passed: bool,
    pub detail: String,
}

/// Deterministic operator-visible acceptance harness. Unit tests exercise the
/// same transition engine in more detail.
pub fn acceptance_matrix() -> Result<Vec<AcceptanceCheck>> {
    let mother = HostIdentity {
        node_id: "motherbrain".into(),
        boot_id: "mother-boot".into(),
        kind: HostKind::Motherbrain,
    };
    let fore = HostIdentity {
        node_id: "forebrain".into(),
        boot_id: "fore-boot".into(),
        kind: HostKind::Forebrain,
    };
    let mut config = FailoverConfig::default();
    config.takeover_enabled = true;
    let active_mother = |at| ControllerStatus::Active {
        node_id: mother.node_id.clone(),
        boot_id: mother.boot_id.clone(),
        observed_at_ms: at,
        lease_expires_at_ms: at + 2_000,
        generation: 7,
    };
    let mut checks = Vec::new();

    let mut machine = HostFailover::new(fore.clone(), config.clone(), 0)?;
    let mut normal = ConnectivityObservation::disconnected(0);
    normal.direct_peer_reachable = true;
    normal.peer_identity = Some(mother.clone());
    normal.peer_observed_at_ms = Some(0);
    normal.controller = active_mother(0);
    let decision = machine.tick(&normal);
    checks.push(check(
        "normal",
        decision.state == ConnectivityState::PreferredPath
            && decision.role == AdvertisedRole::Subordinate,
        &decision,
    ));

    let mut ordinary = ConnectivityObservation::disconnected(4_000);
    ordinary.ordinary_wifi_peer_reachable = true;
    ordinary.peer_identity = Some(mother.clone());
    ordinary.peer_observed_at_ms = Some(4_000);
    ordinary.controller = active_mother(4_000);
    let decision = machine.tick(&ordinary);
    checks.push(check(
        "forebrain_ethernet_loss",
        decision.active_path == Some(PathKind::OrdinaryWifi)
            && decision.role == AdvertisedRole::Subordinate,
        &decision,
    ));

    let mut transit = ConnectivityObservation::disconnected(8_000);
    transit.brainstem_network_associated = true;
    transit.peer_reachable_via_brainstem = true;
    transit.peer_identity = Some(mother.clone());
    transit.peer_observed_at_ms = Some(8_000);
    transit.controller = active_mother(8_000);
    let decision = machine.tick(&transit);
    checks.push(check(
        "brainstem_transit_retains_motherbrain",
        decision.active_path == Some(PathKind::BrainstemTransit)
            && decision.role == AdvertisedRole::Subordinate
            && !decision
                .actions
                .iter()
                .any(|action| matches!(action, FailoverAction::AttemptAtomicAcquisition { .. })),
        &decision,
    ));

    let mut mother_machine = HostFailover::new(mother.clone(), config.clone(), 0)?;
    let mut usb_loss = ConnectivityObservation::disconnected(10_000);
    usb_loss.brainstem_wifi_healthy = true;
    usb_loss.controller = active_mother(10_000);
    let decision = mother_machine.tick(&usb_loss);
    checks.push(check(
        "motherbrain_usb_loss",
        decision.active_path == Some(PathKind::BrainstemWifi)
            && decision.role == AdvertisedRole::Controlling,
        &decision,
    ));

    let mut no_controller = ConnectivityObservation::disconnected(20_000);
    no_controller.brainstem_network_associated = true;
    no_controller.controller = ControllerStatus::Uncontrolled {
        observed_at_ms: 20_000,
        generation: 12,
    };
    machine.tick(&no_controller);
    no_controller.now_ms = 26_000;
    no_controller.controller = ControllerStatus::Uncontrolled {
        observed_at_ms: 26_000,
        generation: 12,
    };
    let candidate = machine.tick(&no_controller);
    no_controller.now_ms = 26_001;
    no_controller.controller = ControllerStatus::Uncontrolled {
        observed_at_ms: 26_001,
        generation: 12,
    };
    no_controller.acquisition_result = Some(AcquisitionResult::Granted);
    no_controller.acquisition_generation = Some(12);
    let controlling = machine.tick(&no_controller);
    checks.push(AcceptanceCheck {
        scenario: "last_resort_takeover",
        passed: candidate.state == ConnectivityState::TakeoverCandidate
            && candidate
                .actions
                .contains(&FailoverAction::AttemptAtomicAcquisition {
                    expected_generation: 12,
                })
            && controlling.state == ConnectivityState::ControllingFallback,
        detail: format!(
            "candidate={}, result={}",
            candidate.reason, controlling.reason
        ),
    });

    let mut returning = ConnectivityObservation::disconnected(30_000);
    returning.direct_peer_reachable = true;
    returning.peer_identity = Some(mother);
    returning.peer_observed_at_ms = Some(30_000);
    returning.controller = ControllerStatus::Active {
        node_id: fore.node_id,
        boot_id: fore.boot_id,
        observed_at_ms: 30_000,
        lease_expires_at_ms: 31_000,
        generation: 13,
    };
    returning.safe_handoff_boundary = true;
    let decision = machine.tick(&returning);
    checks.push(check(
        "orderly_handback",
        decision.state == ConnectivityState::HandbackPending
            && decision.actions == vec![FailoverAction::ReleasePossession],
        &decision,
    ));

    let mut released = ConnectivityObservation::disconnected(30_001);
    released.brainstem_usb_healthy = true;
    released.controller = ControllerStatus::Uncontrolled {
        observed_at_ms: 30_001,
        generation: 14,
    };
    let candidate = mother_machine.tick(&released);
    released.now_ms = 30_002;
    released.controller = ControllerStatus::Uncontrolled {
        observed_at_ms: 30_002,
        generation: 14,
    };
    released.acquisition_result = Some(AcquisitionResult::Granted);
    released.acquisition_generation = Some(14);
    let reacquired = mother_machine.tick(&released);
    checks.push(AcceptanceCheck {
        scenario: "motherbrain_reacquires_after_handback",
        passed: candidate
            .actions
            .contains(&FailoverAction::AttemptAtomicAcquisition {
                expected_generation: 14,
            })
            && reacquired.role == AdvertisedRole::Controlling
            && reacquired
                .actions
                .contains(&FailoverAction::StartBodyFacingRole),
        detail: format!(
            "candidate={}, result={}",
            candidate.reason, reacquired.reason
        ),
    });
    Ok(checks)
}

fn check(scenario: &'static str, passed: bool, decision: &FailoverDecision) -> AcceptanceCheck {
    AcceptanceCheck {
        scenario,
        passed,
        detail: decision.reason.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identities() -> (HostIdentity, HostIdentity) {
        (
            HostIdentity {
                node_id: "motherbrain".into(),
                boot_id: "mother-1".into(),
                kind: HostKind::Motherbrain,
            },
            HostIdentity {
                node_id: "forebrain".into(),
                boot_id: "fore-1".into(),
                kind: HostKind::Forebrain,
            },
        )
    }

    fn active(identity: &HostIdentity, at: u64) -> ControllerStatus {
        ControllerStatus::Active {
            node_id: identity.node_id.clone(),
            boot_id: identity.boot_id.clone(),
            observed_at_ms: at,
            lease_expires_at_ms: at + 2_000,
            generation: 4,
        }
    }

    #[test]
    fn association_never_implies_authority() {
        let (mother, fore) = identities();
        let mut config = FailoverConfig::default();
        config.takeover_enabled = true;
        config.ordinary_search_ms = 1;
        let mut machine = HostFailover::new(fore, config, 0).unwrap();
        let mut observation = ConnectivityObservation::disconnected(0);
        machine.tick(&observation);
        observation.now_ms = 10;
        observation.brainstem_network_associated = true;
        observation.controller = active(&mother, 10);
        let decision = machine.tick(&observation);
        assert_eq!(decision.role, AdvertisedRole::Subordinate);
        assert!(!decision
            .actions
            .iter()
            .any(|action| matches!(action, FailoverAction::AttemptAtomicAcquisition { .. })));
    }

    #[test]
    fn unknown_or_stale_controller_fails_closed() {
        let (_, fore) = identities();
        let mut config = FailoverConfig::default();
        config.takeover_enabled = true;
        config.ordinary_search_ms = 1;
        config.takeover_grace_ms = 1;
        let mut machine = HostFailover::new(fore, config, 0).unwrap();
        let mut observation = ConnectivityObservation::disconnected(0);
        machine.tick(&observation);
        observation.now_ms = 10;
        observation.brainstem_network_associated = true;
        let decision = machine.tick(&observation);
        assert_eq!(decision.state, ConnectivityState::NoControllerObserved);
        observation.now_ms = 3_000;
        observation.controller = ControllerStatus::Uncontrolled {
            observed_at_ms: 0,
            generation: 2,
        };
        let decision = machine.tick(&observation);
        assert_eq!(decision.role, AdvertisedRole::Subordinate);
    }

    #[test]
    fn acquisition_race_has_only_one_winner() {
        let (_, fore) = identities();
        let mut config = FailoverConfig::default();
        config.takeover_enabled = true;
        config.ordinary_search_ms = 1;
        config.takeover_grace_ms = 1;
        let mut a = HostFailover::new(fore.clone(), config.clone(), 0).unwrap();
        let mut b_identity = fore;
        b_identity.node_id = "forebrain-spare".into();
        let mut b = HostFailover::new(b_identity, config, 0).unwrap();
        let mut observation = ConnectivityObservation::disconnected(10);
        observation.brainstem_network_associated = true;
        observation.controller = ControllerStatus::Uncontrolled {
            observed_at_ms: 10,
            generation: 9,
        };
        a.tick(&observation);
        b.tick(&observation);
        observation.now_ms = 12;
        observation.controller = ControllerStatus::Uncontrolled {
            observed_at_ms: 12,
            generation: 9,
        };
        a.tick(&observation);
        b.tick(&observation);
        observation.now_ms = 14;
        observation.controller = ControllerStatus::Uncontrolled {
            observed_at_ms: 14,
            generation: 9,
        };
        assert!(matches!(
            a.tick(&observation).actions.as_slice(),
            [FailoverAction::AttemptAtomicAcquisition {
                expected_generation: 9
            }]
        ));
        assert!(matches!(
            b.tick(&observation).actions.as_slice(),
            [FailoverAction::AttemptAtomicAcquisition {
                expected_generation: 9
            }]
        ));
        observation.now_ms = 15;
        observation.controller = ControllerStatus::Active {
            node_id: "forebrain".into(),
            boot_id: "fore-1".into(),
            observed_at_ms: 15,
            lease_expires_at_ms: 1_000,
            generation: 10,
        };
        observation.acquisition_result = Some(AcquisitionResult::Granted);
        observation.acquisition_generation = Some(9);
        assert_eq!(a.tick(&observation).role, AdvertisedRole::Controlling);
        observation.acquisition_result = Some(AcquisitionResult::Refused);
        observation.acquisition_generation = Some(9);
        assert_eq!(b.tick(&observation).role, AdvertisedRole::Subordinate);
    }

    #[test]
    fn returning_motherbrain_waits_for_safe_handback() {
        let (mother, fore) = identities();
        let mut config = FailoverConfig::default();
        config.takeover_enabled = true;
        let mut machine = HostFailover::new(fore.clone(), config, 0).unwrap();
        let mut observation = ConnectivityObservation::disconnected(10);
        observation.direct_peer_reachable = true;
        observation.peer_identity = Some(mother);
        observation.peer_observed_at_ms = Some(10);
        observation.controller = active(&fore, 10);
        observation.active_safe_sequence = true;
        let decision = machine.tick(&observation);
        assert_eq!(decision.state, ConnectivityState::HandbackPending);
        assert_eq!(
            decision.actions,
            vec![FailoverAction::WaitForSafeHandoffBoundary]
        );
        observation.now_ms = 20;
        observation.active_safe_sequence = false;
        observation.safe_handoff_boundary = true;
        assert_eq!(
            machine.tick(&observation).actions,
            vec![FailoverAction::ReleasePossession]
        );
    }

    #[test]
    fn motherbrain_control_does_not_depend_on_forebrain_presence() {
        let (mother, _) = identities();
        let config = FailoverConfig::default();
        let mut machine = HostFailover::new(mother.clone(), config, 0).unwrap();
        let mut observation = ConnectivityObservation::disconnected(10);
        observation.brainstem_usb_healthy = true;
        observation.controller = active(&mother, 10);

        let decision = machine.tick(&observation);

        assert_eq!(decision.role, AdvertisedRole::Controlling);
        assert_eq!(decision.active_path, Some(PathKind::BrainstemUsb));
        assert!(!decision.peer_visible);
        assert!(decision
            .actions
            .contains(&FailoverAction::ContinueBodyFacingRole));
    }

    #[test]
    fn motherbrain_reacquires_only_after_fallback_release() {
        let (mother, fore) = identities();
        let config = FailoverConfig::default();
        let mut machine = HostFailover::new(mother, config, 0).unwrap();
        let mut observation = ConnectivityObservation::disconnected(10);
        observation.brainstem_usb_healthy = true;
        observation.direct_peer_reachable = true;
        observation.peer_observed_at_ms = Some(10);
        observation.controller = active(&fore, 10);
        assert_eq!(
            machine.tick(&observation).actions,
            vec![
                FailoverAction::RemainSubordinate,
                FailoverAction::WaitForSafeHandoffBoundary,
            ]
        );

        observation.now_ms = 11;
        observation.controller = ControllerStatus::Uncontrolled {
            observed_at_ms: 11,
            generation: 5,
        };
        let candidate = machine.tick(&observation);
        assert!(candidate
            .actions
            .contains(&FailoverAction::AttemptAtomicAcquisition {
                expected_generation: 5,
            }));
        assert_ne!(candidate.role, AdvertisedRole::Controlling);

        observation.now_ms = 12;
        observation.controller = ControllerStatus::Uncontrolled {
            observed_at_ms: 12,
            generation: 5,
        };
        observation.acquisition_result = Some(AcquisitionResult::Granted);
        observation.acquisition_generation = Some(5);
        let acquired = machine.tick(&observation);
        assert_eq!(acquired.role, AdvertisedRole::Controlling);
        assert!(acquired
            .actions
            .contains(&FailoverAction::StartBodyFacingRole));
    }

    #[test]
    fn stale_peer_presence_cannot_trigger_handback() {
        let (mother, fore) = identities();
        let config = FailoverConfig::default();
        let mut machine = HostFailover::new(fore.clone(), config, 0).unwrap();
        let mut observation = ConnectivityObservation::disconnected(10_000);
        observation.direct_peer_reachable = true;
        observation.peer_identity = Some(mother);
        observation.peer_observed_at_ms = Some(1_000);
        observation.controller = active(&fore, 10_000);
        let decision = machine.tick(&observation);
        assert_eq!(decision.state, ConnectivityState::ControllingFallback);
        assert!(!decision.peer_visible);
    }

    #[test]
    fn stale_boot_lease_is_not_mistaken_for_this_process() {
        let (_, fore) = identities();
        let mut config = FailoverConfig::default();
        config.ordinary_search_ms = 1;
        let mut machine = HostFailover::new(fore.clone(), config, 0).unwrap();
        let mut observation = ConnectivityObservation::disconnected(10);
        observation.brainstem_network_associated = true;
        observation.controller = ControllerStatus::Active {
            node_id: fore.node_id,
            boot_id: "previous-boot".into(),
            observed_at_ms: 10,
            lease_expires_at_ms: 2_000,
            generation: 3,
        };

        let decision = machine.tick(&observation);

        assert_eq!(decision.role, AdvertisedRole::Subordinate);
        assert!(!decision
            .actions
            .contains(&FailoverAction::ContinueBodyFacingRole));
        assert!(!decision
            .actions
            .contains(&FailoverAction::StartBodyFacingRole));
    }

    #[test]
    fn path_hysteresis_prevents_immediate_failback() {
        let (mother, fore) = identities();
        let mut config = FailoverConfig::default();
        config.path_hysteresis_ms = 100;
        let mut machine = HostFailover::new(fore, config, 0).unwrap();
        let mut observation = ConnectivityObservation::disconnected(1);
        observation.ordinary_wifi_peer_reachable = true;
        observation.peer_observed_at_ms = Some(1);
        observation.controller = active(&mother, 1);
        machine.tick(&observation);
        observation.now_ms = 10;
        observation.direct_peer_reachable = true;
        observation.ordinary_wifi_peer_reachable = false;
        observation.peer_observed_at_ms = Some(10);
        assert_ne!(
            machine.tick(&observation).state,
            ConnectivityState::PreferredPath
        );
        observation.now_ms = 111;
        observation.peer_observed_at_ms = Some(111);
        observation.controller = active(&mother, 111);
        assert_eq!(
            machine.tick(&observation).state,
            ConnectivityState::PreferredPath
        );
    }

    #[test]
    fn emergency_plane_has_no_bulk_service() {
        let config = FailoverConfig::default();
        config.validate().unwrap();
        assert_eq!(config.permitted_emergency_services.len(), 6);
        assert!(!config.emergency_ports().contains(&22));
    }

    #[test]
    fn deterministic_acceptance_matrix_passes() {
        let checks = acceptance_matrix().unwrap();
        assert!(checks.iter().all(|check| check.passed), "{checks:#?}");
    }
}
