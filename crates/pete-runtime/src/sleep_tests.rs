
use super::*;

fn safe_input(now_ms: u64) -> SleepTickInput {
    SleepTickInput {
        now_ms,
        fatigue_activation: 0.9,
        charging: true,
        docked: true,
        stopped: true,
        body_communication_stable: true,
        active_skill_interruptible: true,
        accelerator_available: true,
        completed_episode_refs: vec!["episode:charging:1".to_string()],
        failed_behavior_refs: vec!["behavior:approach_charger:failed:1".to_string()],
        ..SleepTickInput::default()
    }
}

fn run_to_wake(controller: &mut SleepController, start_ms: u64, accelerator: bool) {
    for offset in 0..8 {
        let mut input = safe_input(start_ms + offset);
        input.accelerator_available = accelerator;
        if offset > 0 {
            input.fatigue_activation = 0.0;
        }
        controller.tick(input);
    }
}

#[test]
fn high_fatigue_while_safely_docked_enters_sleep() {
    let mut controller = SleepController::default();
    let snapshot = controller.tick(safe_input(100));
    assert_eq!(snapshot.phase, SleepPhase::Preparing);
    assert!(snapshot.eligibility.eligible);
    assert!(controller.requires_quiescence());
}

#[test]
fn direct_control_and_unsafe_body_block_entry() {
    let mut input = safe_input(100);
    input.direct_reign_active = true;
    input.safety_event = Some("wheel_drop".to_string());
    let eligibility = sleep_eligibility(&input);
    assert!(!eligibility.eligible);
    assert!(eligibility
        .blocking_reasons
        .iter()
        .any(|reason| reason.contains("Direct Reign")));
    assert!(eligibility
        .blocking_reasons
        .iter()
        .any(|reason| reason.contains("safety")));
}

#[test]
fn safety_event_interrupts_sleep_with_highest_priority() {
    let mut controller = SleepController::default();
    controller.tick(safe_input(100));
    let mut interrupted = safe_input(101);
    interrupted.direct_reign_active = true;
    interrupted.safety_event = Some("cliff".to_string());
    let snapshot = controller.tick(interrupted);
    assert_eq!(snapshot.phase, SleepPhase::Interrupted);
    assert_eq!(
        snapshot.session.unwrap().interrupted_by.unwrap(),
        WakeReason::SafetyEvent("cliff".to_string())
    );
}

#[test]
fn no_accelerator_defers_training_but_completes_local_maintenance() {
    let mut controller = SleepController::default();
    run_to_wake(&mut controller, 100, false);
    let report = controller.last_report.as_ref().unwrap();
    assert!(report.completed.iter().any(|result| {
        result.kind == SleepWorkKind::ConsolidateEpisodes
            && result.status == SleepWorkStatus::Completed
            && result
                .consolidation
                .as_ref()
                .is_some_and(|artifact| artifact.source_history_preserved)
    }));
    assert!(report.completed.iter().any(|result| {
        result.kind == SleepWorkKind::TrainCandidate && result.status == SleepWorkStatus::Deferred
    }));
    assert!(report.consumed_input_refs.is_empty());
    assert!(report.consumed_inputs.iter().any(|consumption| {
        consumption.input_ref == "episode:charging:1"
            && consumption.work_kind == SleepWorkKind::ConsolidateEpisodes
    }));
    assert!(!report
        .consumed_inputs
        .iter()
        .any(|consumption| consumption.work_kind == SleepWorkKind::TrainCandidate));
}

#[test]
fn deferred_training_inputs_are_readmitted_when_acceleration_returns() {
    let mut controller = SleepController::default();
    run_to_wake(&mut controller, 100, false);

    let mut retry = safe_input(108);
    retry.fatigue_activation = 0.0;
    retry.accelerator_available = true;
    let restarted = controller.tick(retry);
    assert_eq!(restarted.phase, SleepPhase::Preparing);
    assert_eq!(
        restarted.session.as_ref().map(|session| session.trigger),
        Some(SleepTrigger::DeferredWork)
    );
    let plan = &restarted.session.as_ref().unwrap().work_plan;
    let training_inputs = &plan
        .iter()
        .find(|item| item.kind == SleepWorkKind::TrainCandidate)
        .unwrap()
        .input_artifact_refs;
    assert!(training_inputs.contains(&"episode:charging:1".to_string()));
    assert!(training_inputs.contains(&"behavior:approach_charger:failed:1".to_string()));
    assert!(plan
        .iter()
        .find(|item| item.kind == SleepWorkKind::ConsolidateEpisodes)
        .unwrap()
        .input_artifact_refs
        .is_empty());

    for now_ms in 109..=115 {
        let mut input = safe_input(now_ms);
        input.fatigue_activation = 0.0;
        controller.tick(input);
    }
    let report = controller.last_report.as_ref().unwrap();
    assert!(report.completed.iter().any(|result| {
        result.kind == SleepWorkKind::TrainCandidate && result.status == SleepWorkStatus::Completed
    }));
    assert!(report
        .consumed_input_refs
        .contains(&"episode:charging:1".to_string()));
    assert!(report
        .consumed_input_refs
        .contains(&"behavior:approach_charger:failed:1".to_string()));
}

#[test]
fn high_fatigue_sleep_on_battery_defers_expensive_work() {
    let mut controller = SleepController::default();
    for offset in 0..8 {
        let mut input = safe_input(100 + offset);
        input.charging = false;
        input.docked = false;
        input.external_power_lost = false;
        if offset > 0 {
            input.fatigue_activation = 0.0;
        }
        controller.tick(input);
    }

    let report = controller.last_report.as_ref().unwrap();
    let training = report
        .completed
        .iter()
        .find(|result| result.kind == SleepWorkKind::TrainCandidate)
        .unwrap();
    assert_eq!(training.status, SleepWorkStatus::Deferred);
    assert!(training.summary.contains("external power"));
    assert!(report.completed.iter().any(|result| {
        result.kind == SleepWorkKind::ConsolidateEpisodes
            && result.status == SleepWorkStatus::Completed
    }));
}

#[test]
fn rising_thermal_state_interrupts_sleep_before_more_work() {
    let mut controller = SleepController::default();
    controller.tick(safe_input(100));
    for now_ms in 101..=103 {
        let mut input = safe_input(now_ms);
        input.fatigue_activation = 0.0;
        controller.tick(input);
    }

    let mut hot = safe_input(104);
    hot.fatigue_activation = 0.0;
    hot.thermal_fraction = 0.81;
    let interrupted = controller.tick(hot);

    assert_eq!(interrupted.phase, SleepPhase::Interrupted);
    let session = interrupted.session.unwrap();
    assert_eq!(
        session.interrupted_by,
        Some(WakeReason::ThermalLimitExceeded)
    );
    assert!(!session
        .completed
        .iter()
        .any(|result| result.kind == SleepWorkKind::TrainCandidate));
}

#[test]
fn candidate_training_and_evaluation_declare_external_power_requirement() {
    let mut controller = SleepController::default();
    let snapshot = controller.tick(safe_input(100));
    let session = snapshot.session.unwrap();

    for kind in [
        SleepWorkKind::TrainCandidate,
        SleepWorkKind::EvaluateCandidate,
    ] {
        assert!(session
            .work_plan
            .iter()
            .find(|item| item.kind == kind)
            .is_some_and(|item| item.requires_external_power));
    }
    assert!(session
        .work_plan
        .iter()
        .all(|item| { item.input_schema_versions.get("world_model") == Some(&3) }));
}

#[test]
fn completed_session_consumes_deferred_refs_and_does_not_reenter() {
    let mut controller = SleepController::default();
    run_to_wake(&mut controller, 100, true);

    let mut same_work = safe_input(108);
    same_work.fatigue_activation = 0.0;
    let snapshot = controller.tick(same_work);

    assert_eq!(snapshot.phase, SleepPhase::Awake);
    assert!(snapshot.session.is_none());
    assert!(!snapshot.eligibility.eligible);
    assert_eq!(controller.sequence, 1);
    let consumed = &snapshot.last_report.unwrap().consumed_input_refs;
    assert!(consumed.contains(&"episode:charging:1".to_string()));
    assert!(consumed.contains(&"behavior:approach_charger:failed:1".to_string()));
}

#[test]
fn persistent_high_fatigue_must_recover_before_it_can_trigger_again() {
    let mut controller = SleepController::default();
    for offset in 0..8 {
        controller.tick(safe_input(100 + offset));
    }

    let still_fatigued = controller.tick(safe_input(108));
    assert_eq!(still_fatigued.phase, SleepPhase::Awake);
    assert!(!still_fatigued.eligibility.eligible);
    assert_eq!(controller.sequence, 1);

    let mut recovered = safe_input(109);
    recovered.fatigue_activation = 0.0;
    let recovered = controller.tick(recovered);
    assert_eq!(recovered.phase, SleepPhase::Awake);

    let fatigued_again = controller.tick(safe_input(110));
    assert_eq!(fatigued_again.phase, SleepPhase::Preparing);
    assert_eq!(
        fatigued_again.session.unwrap().trigger,
        SleepTrigger::HighFatigue
    );
    assert_eq!(controller.sequence, 2);
}

#[test]
fn increased_failure_count_is_new_deferred_work() {
    let mut controller = SleepController::default();
    run_to_wake(&mut controller, 100, true);

    let mut new_failure = safe_input(108);
    new_failure.fatigue_activation = 0.0;
    new_failure.failed_behavior_refs = vec!["behavior:approach_charger:failed:2".to_string()];
    let snapshot = controller.tick(new_failure);

    assert_eq!(snapshot.phase, SleepPhase::Preparing);
    assert_eq!(
        snapshot.session.unwrap().trigger,
        SleepTrigger::DeferredWork
    );
    assert_eq!(controller.sequence, 2);
}

#[test]
fn persistent_operator_request_is_edge_triggered() {
    let mut controller = SleepController::default();
    for offset in 0..8 {
        let mut input = safe_input(100 + offset);
        input.fatigue_activation = 0.0;
        input.operator_sleep_request = true;
        controller.tick(input);
    }

    let mut held_request = safe_input(108);
    held_request.fatigue_activation = 0.0;
    held_request.operator_sleep_request = true;
    let held = controller.tick(held_request);
    assert_eq!(held.phase, SleepPhase::Awake);
    assert!(!held.eligibility.eligible);

    let mut released = safe_input(109);
    released.fatigue_activation = 0.0;
    released.operator_sleep_request = false;
    controller.tick(released);

    let mut pressed_again = safe_input(110);
    pressed_again.fatigue_activation = 0.0;
    pressed_again.operator_sleep_request = true;
    let restarted = controller.tick(pressed_again);
    assert_eq!(restarted.phase, SleepPhase::Preparing);
    assert_eq!(
        restarted.session.unwrap().trigger,
        SleepTrigger::OperatorRequest
    );
}

#[test]
fn replay_preserves_historical_time_and_never_enters_live_now() {
    let mut controller = SleepController::default();
    run_to_wake(&mut controller, 100, true);
    let replay = controller
        .last_report
        .as_ref()
        .unwrap()
        .completed
        .iter()
        .find_map(|result| result.replay.as_ref())
        .unwrap();
    assert_eq!(replay.historical_time_domain, ClockDomain::Event);
    assert_eq!(replay.replay_computed_at.domain, ClockDomain::Replay);
    assert!(!replay.injected_into_live_now);
}

#[test]
fn sleep_evaluates_candidate_without_automatic_promotion() {
    let mut controller = SleepController::default();
    run_to_wake(&mut controller, 100, true);
    let report = controller.last_report.as_ref().unwrap();
    let candidates = report
        .completed
        .iter()
        .filter_map(|result| result.candidate.as_ref())
        .collect::<Vec<_>>();
    assert!(!candidates.is_empty());
    assert!(candidates
        .iter()
        .all(|candidate| !candidate.automatically_promoted));
    assert!(report.promoted_artifact.is_none());
    assert!(report.fresh_world_model_required);
    assert!(!report.stale_skill_resumed);
}

#[test]
fn resource_budget_cancellation_is_deterministic() {
    let mut controller = SleepController::default();
    controller.tick(safe_input(100));
    controller.session_mut().resource_budget.cpu_time_ms = 0;
    controller.tick(SleepTickInput {
        now_ms: 101,
        body_communication_stable: true,
        active_skill_interruptible: true,
        ..SleepTickInput::default()
    });
    controller.tick(SleepTickInput {
        now_ms: 102,
        body_communication_stable: true,
        active_skill_interruptible: true,
        ..SleepTickInput::default()
    });
    controller.tick(SleepTickInput {
        now_ms: 103,
        body_communication_stable: true,
        active_skill_interruptible: true,
        ..SleepTickInput::default()
    });
    assert!(controller
        .session
        .as_ref()
        .unwrap()
        .completed
        .iter()
        .any(|result| result.status == SleepWorkStatus::Cancelled));
}
