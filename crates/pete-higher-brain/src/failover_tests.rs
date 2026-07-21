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
