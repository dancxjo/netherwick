
use super::*;
use pete_body::BodySense;
use pete_now::{Now, ObjectClass, ObjectObservation, ObjectObservationSource, WorldModelUpdater};

fn evaluation(id: &str, activation: f32, urgency: f32) -> GoalEvaluation {
    GoalEvaluation {
        goal_id: GoalId::new(id),
        motivation: Motivation {
            activation,
            urgency,
            satisfaction: 0.0,
        },
        competence: Competence {
            confidence: 1.0,
            affordances: vec![affordance(
                "test",
                ActionPrimitive::Stop,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                100,
                None,
                &[],
            )],
        },
        ..GoalEvaluation::default()
    }
}

fn tick_with_canonical_world(
    system: &mut GoalSystem,
    updater: &mut WorldModelUpdater,
    now: Now,
) -> GoalCycle {
    let now = updater.update(now, system.world_model_update_context());
    system.tick(&now.world, &[]).unwrap()
}

#[test]
fn world_model_keeps_entity_identity_across_occlusion() {
    let mut updater = WorldModelUpdater::default();
    let mut now = Now::blank(100, BodySense::default());
    now.objects.observations.push(ObjectObservation {
        label: "dock 17".to_string(),
        class: ObjectClass::Charger,
        bearing_rad: 0.2,
        distance_m: Some(1.5),
        confidence: 0.9,
        source: ObjectObservationSource::Sim,
    });
    let first = updater
        .update(now.clone(), WorldModelUpdateContext::default())
        .world;
    now.t_ms = 500;
    now.objects.observations.clear();
    let second = updater
        .update(now, WorldModelUpdateContext::default())
        .world;
    assert_eq!(
        first.entities.keys().collect::<Vec<_>>(),
        second.entities.keys().collect::<Vec<_>>()
    );
    assert_eq!(second.entities.values().next().unwrap().confidence, 0.9);
}

#[test]
fn goal_interpretation_recomputes_relative_bearing_from_world_pose() {
    let mut system = GoalSystem::default();
    let mut updater = WorldModelUpdater::default();
    let mut now = Now::blank(100, BodySense::default());
    now.body.battery_level = 0.2;
    now.objects.observations.push(ObjectObservation {
        label: "dock".to_string(),
        class: ObjectClass::Charger,
        bearing_rad: 0.0,
        distance_m: Some(2.0),
        confidence: 0.9,
        source: ObjectObservationSource::Sim,
    });
    tick_with_canonical_world(&mut system, &mut updater, now.clone());

    now.t_ms = 200;
    now.objects.observations.clear();
    now.body.odometry.heading_rad = std::f32::consts::FRAC_PI_2;
    let cycle = tick_with_canonical_world(&mut system, &mut updater, now);
    let charge = cycle
        .interpretations
        .iter()
        .find(|interpretation| interpretation.goal_id == GoalId::new("seek_charger"))
        .unwrap();
    assert!((charge.target_bearing_rad.unwrap() + std::f32::consts::FRAC_PI_2).abs() < 0.001);
}

#[test]
fn goal_commitment_rejects_small_oscillations() {
    let mut arbiter = GoalArbiter::default();
    let first = arbiter.select(
        0,
        &[
            evaluation("explore", 0.51, 0.0),
            evaluation("charge", 0.50, 0.0),
        ],
    );
    assert_eq!(first.selected_goal, Some(GoalId::new("explore")));
    let second = arbiter.select(
        1_000,
        &[
            evaluation("explore", 0.49, 0.0),
            evaluation("charge", 0.52, 0.0),
        ],
    );
    assert_eq!(second.selected_goal, Some(GoalId::new("explore")));
    assert!(second.retained_by_commitment);
}

#[test]
fn arbitration_is_deterministic_and_does_not_modify_evaluations() {
    let mut alpha = evaluation("alpha", 0.5, 0.0);
    alpha.contributions.push(EvaluationContribution {
        source: "direct_observation".to_string(),
        value: 100.0,
    });
    alpha.competence.affordances[0]
        .provenance
        .push(EvidenceRef {
            id: "sensor:a".to_string(),
            ..EvidenceRef::default()
        });
    let mut beta = evaluation("beta", 0.5, 0.0);
    beta.contributions.push(EvaluationContribution {
        source: "memory_recall".to_string(),
        value: -100.0,
    });
    let evaluations = vec![alpha.clone(), beta.clone()];
    let original = evaluations.clone();
    let first = GoalArbiter::default().select(0, &evaluations);
    assert_eq!(evaluations, original);

    let reversed = vec![beta, alpha];
    let second = GoalArbiter::default().select(0, &reversed);
    assert_eq!(first.selected_goal, second.selected_goal);
}

#[test]
fn goal_components_are_independently_replaceable() {
    let id = GoalId::new("rest");
    let mut goal = GoalModule::new(id.clone());
    goal.interpreter_state.updates = 3;
    goal.evaluator_state.evaluations = 4;
    goal.executor_state.executions = 5;

    goal.replace_interpreter(Box::new(RuleGoalInterpreter { id: id.clone() }));
    assert_eq!(goal.interpreter_state, InterpreterState::default());
    assert_eq!(goal.evaluator_state.evaluations, 4);
    assert_eq!(goal.executor_state.executions, 5);

    goal.replace_evaluator(Box::new(RuleGoalEvaluator { id: id.clone() }));
    assert_eq!(goal.evaluator_state, EvaluatorState::default());
    assert_eq!(goal.executor_state.executions, 5);

    goal.replace_executor(Box::new(UtilityGoalExecutor { id }));
    assert_eq!(goal.executor_state, ExecutorState::default());
}

#[test]
fn adding_a_registered_goal_does_not_change_the_arbiter() {
    let mut system = GoalSystem::with_goals(vec![Box::new(GoalModule::new(GoalId::new("rest")))]);
    system
        .register_goal(Box::new(GoalModule::new(GoalId::new("explore"))))
        .unwrap();
    let mut updater = WorldModelUpdater::default();
    let cycle = tick_with_canonical_world(
        &mut system,
        &mut updater,
        Now::blank(0, BodySense::default()),
    );
    assert_eq!(cycle.evaluations.len(), 2);
    assert!(cycle
        .evaluations
        .iter()
        .any(|evaluation| evaluation.goal_id == GoalId::new("explore")));
}

#[test]
fn urgency_reduces_commitment_cost_without_becoming_activation() {
    let mut arbiter = GoalArbiter::default();
    arbiter.select(0, &[evaluation("explore", 0.4, 0.0)]);
    let switched = arbiter.select(
        10,
        &[
            evaluation("explore", 0.4, 0.0),
            evaluation("charge", 0.51, 1.0),
        ],
    );
    assert_eq!(switched.selected_goal, Some(GoalId::new("charge")));
    assert!(switched.switched);
    assert_eq!(switched.effective_minimum_dwell_ms, 0);
}

#[test]
fn completed_goal_releases_commitment_immediately() {
    let mut arbiter = GoalArbiter::default();
    arbiter.select(0, &[evaluation("charge", 0.9, 0.0)]);
    let mut completed = evaluation("charge", 0.9, 0.0);
    completed.disposition = GoalDisposition::Completed;
    let selection = arbiter.select(10, &[completed, evaluation("explore", 0.2, 0.0)]);
    assert_eq!(selection.selected_goal, Some(GoalId::new("explore")));
    assert_eq!(selection.exit_reason, Some(GoalExitReason::Completed));
    assert!(selection.switched);
}

#[test]
fn transient_drive_impulse_decays_and_ordinary_frames_do_not_reset_it() {
    let mut dynamics = DriveDynamics::default();
    let mut world = WorldModelSnapshot::default();
    let mut body = BodySense::default();
    body.battery_level = 0.8;
    world.self_model.battery_level = body.battery_level;
    dynamics.update(&world);
    dynamics.add_impulses(DriveSense {
        battery_hunger: 1.0,
        ..DriveSense::default()
    });
    world.t_ms = 100;
    let pulsed = dynamics.update(&world).energy.activation;
    world.t_ms = 200;
    let recovered = dynamics.update(&world).energy.activation;
    assert!(pulsed > 0.05);
    assert!(recovered < pulsed);
}

#[test]
fn low_confidence_urgent_charge_searches_instead_of_docking() {
    let mut system = GoalSystem::default();
    let mut updater = WorldModelUpdater::default();
    let mut body = BodySense::default();
    body.battery_level = 0.05;
    let now = Now::blank(1_000, body);
    let cycle = tick_with_canonical_world(&mut system, &mut updater, now);
    let behavior = cycle.behavior.unwrap();
    assert_eq!(behavior.goal_id, GoalId::new("seek_charger"));
    assert_eq!(behavior.behavior_id, "systematic_charger_search");
    assert!(matches!(behavior.action, ActionPrimitive::Explore { .. }));
    assert!(behavior.affordance.epistemic_question_id.is_some());
    assert!(behavior.affordance.expected_information_gain > 0.0);
}

#[test]
fn low_confidence_localized_charger_rejects_direct_locomotion() {
    let mut system = GoalSystem::default();
    let mut updater = WorldModelUpdater::default();
    let mut body = BodySense::default();
    body.battery_level = 0.05;
    let mut now = Now::blank(1_000, body);
    now.objects.observations.push(ObjectObservation {
        label: "uncertain dock".to_string(),
        class: ObjectClass::Charger,
        bearing_rad: 0.0,
        distance_m: Some(0.2),
        confidence: 0.1,
        source: ObjectObservationSource::Sim,
    });
    let cycle = tick_with_canonical_world(&mut system, &mut updater, now);
    assert_eq!(
        cycle.behavior.as_ref().unwrap().behavior_id,
        "systematic_charger_search"
    );
    let evaluation = cycle
        .evaluations
        .iter()
        .find(|evaluation| evaluation.goal_id == GoalId::new("seek_charger"))
        .unwrap();
    let dock = evaluation
        .competence
        .affordances
        .iter()
        .find(|affordance| affordance.behavior_id == "dock")
        .unwrap();
    assert!(!dock.available);
    assert!(dock.rejection_reason.is_some());
}

#[test]
fn goal_competence_uses_canonical_drive_capability() {
    let mut system = GoalSystem::default();
    let mut updater = WorldModelUpdater::default();
    let mut body = BodySense::default();
    body.battery_level = 0.05;
    body.health.health = 0.1;
    let mut now = Now::blank(1_000, body);
    now.objects.observations.push(ObjectObservation {
        label: "dock".to_string(),
        class: ObjectClass::Charger,
        bearing_rad: 0.0,
        distance_m: Some(1.0),
        confidence: 0.95,
        source: ObjectObservationSource::Sim,
    });
    let cycle = tick_with_canonical_world(&mut system, &mut updater, now);
    let evaluation = cycle
        .evaluations
        .iter()
        .find(|evaluation| evaluation.goal_id == GoalId::new("seek_charger"))
        .unwrap();
    let approach = evaluation
        .competence
        .affordances
        .iter()
        .find(|affordance| affordance.behavior_id == "approach_charger")
        .unwrap();
    assert!(!approach.available);
    assert_eq!(
        approach.rejection_reason.as_deref(),
        Some("drive is unsafe or body health is degraded")
    );
    assert!(!cycle
        .world
        .self_model
        .capabilities
        .is_available("actuator:drive"));
}

#[test]
fn occluded_charger_selects_search_instead_of_direct_approach() {
    let mut system = GoalSystem::default();
    let mut updater = WorldModelUpdater::default();
    let mut body = BodySense::default();
    body.battery_level = 0.05;
    let mut now = Now::blank(1_000, body);
    now.range.nearest_m = Some(0.5);
    now.objects.observations.extend([
        ObjectObservation {
            label: "blocking obstacle".to_string(),
            class: ObjectClass::Obstacle,
            bearing_rad: 0.02,
            distance_m: Some(0.5),
            confidence: 0.95,
            source: ObjectObservationSource::Sim,
        },
        ObjectObservation {
            label: "dock".to_string(),
            class: ObjectClass::Charger,
            bearing_rad: 0.0,
            distance_m: Some(2.0),
            confidence: 0.95,
            source: ObjectObservationSource::Sim,
        },
    ]);
    let cycle = tick_with_canonical_world(&mut system, &mut updater, now);
    assert_eq!(
        cycle.behavior.as_ref().unwrap().behavior_id,
        "systematic_charger_search"
    );
    let charge = cycle
        .evaluations
        .iter()
        .find(|evaluation| evaluation.goal_id == GoalId::new("seek_charger"))
        .unwrap();
    let approach = charge
        .competence
        .affordances
        .iter()
        .find(|affordance| affordance.behavior_id == "approach_charger")
        .unwrap();
    assert!(!approach.available);
    assert!(approach
        .rejection_reason
        .as_deref()
        .is_some_and(|reason| reason.contains("not currently reachable")));
}

#[test]
fn obstacle_contact_releases_charge_commitment_to_escape() {
    let mut system = GoalSystem::default();
    let mut updater = WorldModelUpdater::default();
    let mut body = BodySense::default();
    body.battery_level = 0.05;
    body.flags.bump_right = true;
    let mut now = Now::blank(1_000, body);
    now.objects.observations.push(ObjectObservation {
        label: "dock".to_string(),
        class: ObjectClass::Charger,
        bearing_rad: 0.0,
        distance_m: Some(1.5),
        confidence: 0.9,
        source: ObjectObservationSource::Sim,
    });
    let cycle = tick_with_canonical_world(&mut system, &mut updater, now);
    assert_eq!(
        cycle.selection.selected_goal,
        Some(GoalId::new("escape_danger"))
    );
    let charge = cycle
        .evaluations
        .iter()
        .find(|evaluation| evaluation.goal_id == GoalId::new("seek_charger"))
        .unwrap();
    assert!(charge
        .competence
        .affordances
        .iter()
        .all(|affordance| !affordance.available));
}

#[test]
fn escape_goal_sequences_behaviors_without_resetting_goal_commitment() {
    let mut system = GoalSystem::default();
    let mut updater = WorldModelUpdater::default();
    let mut body = BodySense::default();
    body.battery_level = 1.0;
    let mut now = Now::blank(0, body);
    now.memory.place_danger = 1.0;
    now.memory.map_confidence = 1.0;
    now.range.beams = vec![1.0; 9];
    let mut behaviors = Vec::new();
    for tick in 0..13 {
        now.t_ms = tick * 100;
        let cycle = tick_with_canonical_world(&mut system, &mut updater, now.clone());
        assert_eq!(
            cycle.selection.selected_goal,
            Some(GoalId::new("escape_danger"))
        );
        behaviors.push(cycle.behavior.unwrap().behavior_id);
    }
    assert!(behaviors[..9]
        .iter()
        .all(|behavior| behavior == "turn_toward_clearance"));
    assert!(behaviors[9..12]
        .iter()
        .all(|behavior| behavior == "probe_clearance"));
    assert_eq!(behaviors[12], "inspect_clearance");
}

#[test]
fn high_confidence_nearby_charger_affords_docking() {
    let mut system = GoalSystem::default();
    let mut updater = WorldModelUpdater::default();
    let mut body = BodySense::default();
    body.battery_level = 0.05;
    let mut now = Now::blank(1_000, body);
    now.objects.observations.push(ObjectObservation {
        label: "dock".to_string(),
        class: ObjectClass::Charger,
        bearing_rad: 0.0,
        distance_m: Some(0.2),
        confidence: 0.98,
        source: ObjectObservationSource::Sim,
    });
    let cycle = tick_with_canonical_world(&mut system, &mut updater, now);
    let behavior = cycle.behavior.unwrap();
    assert_eq!(behavior.goal_id, GoalId::new("seek_charger"));
    assert_eq!(behavior.behavior_id, "dock");
    assert_eq!(behavior.action, ActionPrimitive::Dock);
    assert_eq!(
        behavior
            .affordance
            .skill_request
            .as_ref()
            .map(|request| request.skill_id),
        Some(SkillId::AlignWithDock)
    );
}

#[test]
fn urgent_aligned_charger_approach_requests_possessor_skill() {
    let mut system = GoalSystem::default();
    let mut updater = WorldModelUpdater::default();
    let mut body = BodySense::default();
    body.battery_level = 0.05;
    let mut now = Now::blank(1_000, body);
    now.objects.observations.push(ObjectObservation {
        label: "dock".to_string(),
        class: ObjectClass::Charger,
        bearing_rad: 0.1,
        distance_m: Some(2.0),
        confidence: 0.98,
        source: ObjectObservationSource::Sim,
    });
    let cycle = tick_with_canonical_world(&mut system, &mut updater, now);
    let behavior = cycle.behavior.unwrap();
    assert_eq!(behavior.behavior_id, "approach_charger");
    assert_eq!(
        behavior.action,
        ActionPrimitive::Approach {
            target: ApproachTarget::Charger
        }
    );
    let skill = behavior.affordance.skill_request.unwrap();
    assert_eq!(skill.skill_id, SkillId::ApproachTarget);
    assert_eq!(skill.range_m, Some(2.0));
    assert_eq!(skill.stop_range_m, Some(0.30));
}

#[test]
fn failed_expected_progress_builds_runtime_frustration() {
    let mut system = GoalSystem::default();
    let mut updater = WorldModelUpdater::default();
    let mut body = BodySense::default();
    body.battery_level = 0.05;
    let first = Now::blank(1_000, body.clone());
    tick_with_canonical_world(&mut system, &mut updater, first);
    let second = Now::blank(2_100, body);
    tick_with_canonical_world(&mut system, &mut updater, second);
    let charge = system
        .goals
        .iter()
        .find(|goal| goal.id() == &GoalId::new("seek_charger"))
        .unwrap();
    assert_eq!(charge.runtime().failed_attempts, 1);
    assert!(charge.runtime().frustration > 0.0);
}

#[test]
fn possessor_terminal_failure_is_processed_once_per_execution() {
    let mut system = GoalSystem::default();
    let mut updater = WorldModelUpdater::default();
    let mut body = BodySense::default();
    body.battery_level = 0.05;
    let mut now = Now::blank(1_000, body);
    now.objects.observations.push(ObjectObservation {
        label: "dock".to_string(),
        class: ObjectClass::Charger,
        bearing_rad: 0.1,
        distance_m: Some(2.0),
        confidence: 0.98,
        source: ObjectObservationSource::Sim,
    });
    let cycle = tick_with_canonical_world(&mut system, &mut updater, now);
    let request = cycle.behavior.unwrap().affordance.skill_request.unwrap();

    let failure = SkillStatus {
        request,
        execution_id: 7,
        phase: SkillPhase::Terminal,
        outcome: Some(SkillOutcome::TimedOut),
        progress: None,
        attempts: 1,
        dispatch_count: 20,
        started_at_ms: Some(1_000),
        updated_at_ms: 2_000,
        reason: Some("no target progress".to_string()),
        script: None,
    };
    system.observe_skill_status(&failure);
    system.observe_skill_status(&failure);

    let charge = system
        .goals
        .iter()
        .find(|goal| goal.id() == &GoalId::new("seek_charger"))
        .unwrap();
    assert_eq!(charge.runtime().failed_attempts, 1);
    assert_eq!(charge.runtime().attempts, 1);
    assert_eq!(
        charge.runtime().last_skill_outcome,
        Some(SkillOutcome::TimedOut)
    );
    assert!(charge.runtime().last_progress_observation.is_some());
    assert_eq!(
        system.arbiter.current_goal(),
        Some(&GoalId::new("seek_charger"))
    );
}

#[test]
fn autonomic_preemption_is_not_counted_as_intended_skill_progress() {
    let mut system = GoalSystem::default();
    let mut updater = WorldModelUpdater::default();
    let mut body = BodySense::default();
    body.battery_level = 0.05;
    let cycle = tick_with_canonical_world(&mut system, &mut updater, Now::blank(1_000, body));
    let request = cycle.behavior.unwrap().affordance.skill_request.unwrap();

    system.observe_skill_status(&SkillStatus {
        request,
        execution_id: 8,
        phase: SkillPhase::Terminal,
        outcome: Some(SkillOutcome::SafetyPreempted),
        progress: Some(1.0),
        attempts: 1,
        dispatch_count: 1,
        started_at_ms: Some(1_000),
        updated_at_ms: 1_100,
        reason: Some("contact withdrawal preempted possessor control".to_string()),
        script: None,
    });

    let runtime = system
        .goals
        .iter()
        .find(|goal| goal.id() == &GoalId::new("seek_charger"))
        .unwrap()
        .runtime();
    assert_eq!(runtime.failed_attempts, 0);
    assert_eq!(runtime.recent_progress, 0.0);
    assert_eq!(runtime.last_progress_at_ms, None);
    assert_eq!(
        runtime
            .last_progress_observation
            .as_ref()
            .and_then(|observation| observation.progress),
        None
    );
    assert_eq!(
        runtime
            .last_progress_observation
            .as_ref()
            .map(|observation| observation.source.as_str()),
        Some("autonomic_safety_preemption")
    );
}

#[test]
fn possessor_progress_is_goal_scoped_and_only_fresh_when_it_advances() {
    let mut system = GoalSystem::default();
    let request = SkillRequest {
        skill_id: SkillId::ApproachTarget,
        goal_id: Some(GoalId::new("seek_charger")),
        progress_metric: "target_distance".to_string(),
        progress_baseline: Some(2.0),
        progress_tolerance: 0.1,
        ..SkillRequest::default()
    };
    let status = |progress, updated_at_ms| SkillStatus {
        request: request.clone(),
        execution_id: 9,
        phase: SkillPhase::Running,
        outcome: None,
        progress: Some(progress),
        attempts: 1,
        dispatch_count: 1,
        started_at_ms: Some(1_000),
        updated_at_ms,
        reason: None,
        script: None,
    };

    system.observe_skill_status(&status(0.25, 1_100));
    let charge = system
        .goals
        .iter()
        .find(|goal| goal.id() == &GoalId::new("seek_charger"))
        .unwrap()
        .runtime();
    assert_eq!(charge.last_progress_at_ms, Some(1_100));
    assert_eq!(
        charge
            .last_progress_observation
            .as_ref()
            .and_then(|observation| observation.progress),
        Some(0.25)
    );
    assert!(charge.recent_progress > 0.0);
    let explore_before = system
        .goals
        .iter()
        .find(|goal| goal.id() == &GoalId::new("explore"))
        .unwrap()
        .runtime()
        .clone();

    system.observe_skill_status(&status(0.25, 1_200));
    let charge = system
        .goals
        .iter()
        .find(|goal| goal.id() == &GoalId::new("seek_charger"))
        .unwrap()
        .runtime();
    assert_eq!(charge.last_progress_at_ms, Some(1_100));
    let explore_after = system
        .goals
        .iter()
        .find(|goal| goal.id() == &GoalId::new("explore"))
        .unwrap()
        .runtime();
    assert_eq!(explore_after, &explore_before);

    system.observe_skill_status(&status(0.5, 1_300));
    let charge = system
        .goals
        .iter()
        .find(|goal| goal.id() == &GoalId::new("seek_charger"))
        .unwrap()
        .runtime();
    assert_eq!(charge.last_progress_at_ms, Some(1_300));
    assert!(charge.progress_trend > 0.0);
}

#[test]
fn repeated_charger_failure_requests_help_then_abandons_at_bounded_limit() {
    let mut system = GoalSystem::default();
    let mut updater = WorldModelUpdater::default();
    let mut body = BodySense::default();
    body.battery_level = 0.05;
    let first =
        tick_with_canonical_world(&mut system, &mut updater, Now::blank(1_000, body.clone()));
    assert_eq!(first.schema_version, GOAL_CYCLE_SCHEMA_VERSION);
    let expectation = first
        .progress
        .iter()
        .find(|report| report.goal_id == GoalId::new("seek_charger"))
        .and_then(|report| report.expectation.as_ref())
        .unwrap();
    assert_eq!(expectation.metric, "uncertainty_reduction");
    assert!(expectation.baseline.is_some());
    assert_eq!(expectation.horizon_ms, 1_000);
    assert_eq!(expectation.tolerance, 0.1);
    let search = first.behavior.unwrap();
    assert_eq!(search.goal_id, GoalId::new("seek_charger"));
    assert_eq!(search.behavior_id, "systematic_charger_search");
    let request = search.affordance.skill_request.unwrap();
    assert_eq!(request.progress_metric, "frontier_coverage");
    assert_eq!(request.progress_baseline, Some(0.0));
    assert_eq!(request.progress_tolerance, 0.1);

    for attempt in 1..=4 {
        system.observe_skill_status(&SkillStatus {
            request: request.clone(),
            execution_id: attempt as u64,
            phase: SkillPhase::Terminal,
            outcome: Some(SkillOutcome::TimedOut),
            progress: Some(0.0),
            attempts: attempt,
            dispatch_count: 10,
            started_at_ms: Some(1_000),
            updated_at_ms: 1_000 + attempt as u64 * 25,
            reason: Some("charger search produced no evidence".to_string()),
            script: None,
        });
    }
    let charge_runtime = system
        .goals
        .iter()
        .find(|goal| goal.id() == &GoalId::new("seek_charger"))
        .unwrap()
        .runtime();
    assert_eq!(charge_runtime.failed_attempts, 4);
    assert!(charge_runtime.frustration > 0.6);

    let help =
        tick_with_canonical_world(&mut system, &mut updater, Now::blank(1_200, body.clone()));
    let help_evaluation = help
        .evaluations
        .iter()
        .find(|evaluation| evaluation.goal_id == GoalId::new("seek_charger"))
        .unwrap();
    assert!(help_evaluation
        .competence
        .affordances
        .iter()
        .any(|affordance| affordance.behavior_id == "request_charge_help"));
    assert_eq!(
        help.selection.selected_goal,
        Some(GoalId::new("seek_charger"))
    );
    assert_eq!(
        help.behavior
            .as_ref()
            .map(|behavior| behavior.behavior_id.as_str()),
        Some("request_charge_help")
    );
    let help_report = help
        .progress
        .iter()
        .find(|report| report.goal_id == GoalId::new("seek_charger"))
        .unwrap();
    assert_eq!(
        help_report.response,
        StrategyProgressResponse::HelpRequested
    );
    assert!(help_report.reason.contains("bounded escalation"));
    assert_eq!(
        help_report.previous_behavior.as_deref(),
        Some("systematic_charger_search")
    );

    let replayed: GoalCycle = serde_json::from_value(serde_json::to_value(&help).unwrap())
        .expect("progress trace should replay from a serialized goal cycle");
    assert_eq!(replayed.progress, help.progress);

    for attempt in 5..=8 {
        system.observe_skill_status(&SkillStatus {
            request: request.clone(),
            execution_id: attempt as u64,
            phase: SkillPhase::Terminal,
            outcome: Some(SkillOutcome::TimedOut),
            progress: Some(0.0),
            attempts: attempt,
            dispatch_count: 10,
            started_at_ms: Some(1_000),
            updated_at_ms: 1_200 + attempt as u64 * 25,
            reason: Some("bounded charger retry failed".to_string()),
            script: None,
        });
    }
    let abandoned = tick_with_canonical_world(&mut system, &mut updater, Now::blank(1_400, body));
    let abandoned_report = abandoned
        .progress
        .iter()
        .find(|report| report.goal_id == GoalId::new("seek_charger"))
        .unwrap();
    assert_eq!(
        abandoned_report.response,
        StrategyProgressResponse::Abandoned
    );
    assert_eq!(abandoned_report.failed_attempts, 8);
    assert!(abandoned_report.reason.contains("goal abandoned"));
    assert_ne!(
        abandoned.selection.selected_goal,
        Some(GoalId::new("seek_charger"))
    );
}

#[test]
fn stalled_explore_changes_strategy_without_switching_goal() {
    let mut system = GoalSystem::default();
    let mut updater = WorldModelUpdater::default();
    let body = BodySense {
        battery_level: 1.0,
        ..BodySense::default()
    };
    let first =
        tick_with_canonical_world(&mut system, &mut updater, Now::blank(1_000, body.clone()));
    let initial = first.behavior.unwrap();
    assert_eq!(initial.goal_id, GoalId::new("explore"));
    assert_eq!(initial.behavior_id, "random_walk_exploration");
    let request = initial.affordance.skill_request.unwrap();

    for attempt in 1..=4 {
        system.observe_skill_status(&SkillStatus {
            request: request.clone(),
            execution_id: attempt as u64,
            phase: SkillPhase::Terminal,
            outcome: Some(SkillOutcome::TimedOut),
            progress: Some(0.0),
            attempts: attempt,
            dispatch_count: 10,
            started_at_ms: Some(1_000),
            updated_at_ms: 1_000 + attempt as u64 * 25,
            reason: Some("frontier coverage did not increase".to_string()),
            script: None,
        });
    }

    let changed = tick_with_canonical_world(&mut system, &mut updater, Now::blank(1_200, body));
    assert_eq!(
        changed.selection.selected_goal,
        Some(GoalId::new("explore"))
    );
    assert_eq!(
        changed
            .behavior
            .as_ref()
            .map(|behavior| behavior.behavior_id.as_str()),
        Some("wall_follow_exploration")
    );
    let report = changed
        .progress
        .iter()
        .find(|report| report.goal_id == GoalId::new("explore"))
        .unwrap();
    assert_eq!(report.response, StrategyProgressResponse::Changed);
    assert_eq!(
        report.previous_behavior.as_deref(),
        Some("random_walk_exploration")
    );
    assert!(report.reason.contains("changed strategy"));
}

#[test]
fn stale_target_makes_progress_unknown_without_counting_failure() {
    let mut system = GoalSystem::default();
    let mut updater = WorldModelUpdater::default();
    let mut body = BodySense::default();
    body.battery_level = 0.05;
    let mut now = Now::blank(1_000, body);
    now.objects.observations.push(ObjectObservation {
        label: "dock".to_string(),
        class: ObjectClass::Charger,
        bearing_rad: 0.1,
        distance_m: Some(2.0),
        confidence: 0.98,
        source: ObjectObservationSource::Sim,
    });
    let first = tick_with_canonical_world(&mut system, &mut updater, now);
    assert_eq!(
        first
            .behavior
            .as_ref()
            .map(|behavior| behavior.behavior_id.as_str()),
        Some("approach_charger")
    );

    let mut stale_world = first.world.clone();
    stale_world.t_ms = 2_100;
    for entity in stale_world.entities.values_mut() {
        entity.distance_meta.as_mut().unwrap().freshness = Freshness::Stale;
    }
    let stale = system.tick(&stale_world, &[]).unwrap();
    let report = stale
        .progress
        .iter()
        .find(|report| report.goal_id == GoalId::new("seek_charger"))
        .unwrap();
    assert_eq!(report.failed_attempts, 0);
    assert_eq!(
        report
            .observation
            .as_ref()
            .and_then(|observation| observation.progress),
        None
    );
    assert!(report.reason.contains("unknown"));
}

#[test]
fn reusable_skills_are_claimed_by_multiple_goals() {
    let mut updater = WorldModelUpdater::default();

    let mut charger_now = Now::blank(1_000, BodySense::default());
    charger_now.body.battery_level = 0.05;
    charger_now.objects.observations.push(ObjectObservation {
        label: "dock".to_string(),
        class: ObjectClass::Charger,
        bearing_rad: 0.8,
        distance_m: Some(2.0),
        confidence: 0.95,
        source: ObjectObservationSource::Sim,
    });
    let mut charger_system = GoalSystem::default();
    let charger = tick_with_canonical_world(&mut charger_system, &mut updater, charger_now);
    let mut aligned_now = Now::blank(3_500, BodySense::default());
    aligned_now.body.battery_level = 0.05;
    aligned_now.objects.observations.push(ObjectObservation {
        label: "aligned dock".to_string(),
        class: ObjectClass::Charger,
        bearing_rad: 0.1,
        distance_m: Some(2.0),
        confidence: 0.95,
        source: ObjectObservationSource::Sim,
    });
    let mut aligned_system = GoalSystem::default();
    let aligned = tick_with_canonical_world(&mut aligned_system, &mut updater, aligned_now);
    assert!(charger
        .evaluations
        .iter()
        .find(|evaluation| evaluation.goal_id == GoalId::new("seek_charger"))
        .unwrap()
        .competence
        .affordances
        .iter()
        .any(|affordance| affordance
            .skill_request
            .as_ref()
            .map(|request| request.skill_id)
            == Some(SkillId::TurnTowardTarget)));

    let mut escape_now = Now::blank(2_000, BodySense::default());
    escape_now.body.flags.bump_left = true;
    escape_now.memory.place_danger = 1.0;
    escape_now.memory.map_confidence = 1.0;
    let mut escape_system = GoalSystem::default();
    let escape = tick_with_canonical_world(&mut escape_system, &mut updater, escape_now);
    assert!(escape
        .evaluations
        .iter()
        .find(|evaluation| evaluation.goal_id == GoalId::new("escape_danger"))
        .unwrap()
        .competence
        .affordances
        .iter()
        .any(|affordance| affordance
            .skill_request
            .as_ref()
            .map(|request| request.skill_id)
            == Some(SkillId::TurnTowardTarget)));

    let mut person_now = Now::blank(3_000, BodySense::default());
    person_now.objects.observations.push(ObjectObservation {
        label: "person".to_string(),
        class: ObjectClass::Person,
        bearing_rad: 0.1,
        distance_m: Some(2.0),
        confidence: 0.9,
        source: ObjectObservationSource::Sim,
    });
    let mut social_system = GoalSystem::default();
    let social = tick_with_canonical_world(&mut social_system, &mut updater, person_now);
    assert!(social
        .evaluations
        .iter()
        .find(|evaluation| evaluation.goal_id == GoalId::new("socialize"))
        .unwrap()
        .competence
        .affordances
        .iter()
        .any(|affordance| affordance
            .skill_request
            .as_ref()
            .map(|request| request.skill_id)
            == Some(SkillId::ApproachTarget)));
    assert!(aligned
        .evaluations
        .iter()
        .find(|evaluation| evaluation.goal_id == GoalId::new("seek_charger"))
        .unwrap()
        .competence
        .affordances
        .iter()
        .any(|affordance| affordance
            .skill_request
            .as_ref()
            .map(|request| request.skill_id)
            == Some(SkillId::ApproachTarget)));

    let task_now = Now::blank(4_000, BodySense::default());
    let task_world = updater.update(task_now, WorldModelUpdateContext::default());
    let mut task_system = GoalSystem::default();
    let task = task_system
        .tick(
            &task_world.world,
            &[ActionPrimitive::Go {
                intensity: -0.2,
                duration_ms: 300,
            }],
        )
        .unwrap();
    assert!(task
        .evaluations
        .iter()
        .find(|evaluation| evaluation.goal_id == GoalId::new("follow_task"))
        .unwrap()
        .competence
        .affordances
        .iter()
        .any(|affordance| affordance
            .skill_request
            .as_ref()
            .map(|request| request.skill_id)
            == Some(SkillId::BackAway)));
    assert!(escape
        .evaluations
        .iter()
        .find(|evaluation| evaluation.goal_id == GoalId::new("escape_danger"))
        .unwrap()
        .competence
        .affordances
        .iter()
        .any(|affordance| affordance
            .skill_request
            .as_ref()
            .map(|request| request.skill_id)
            == Some(SkillId::BackAway)));
}

#[test]
fn absent_llm_opinion_does_not_create_uncertainty_pressure() {
    let mut system = GoalSystem::default();
    let mut updater = WorldModelUpdater::default();
    let mut body = BodySense::default();
    body.battery_level = 0.35;
    let cycle = tick_with_canonical_world(&mut system, &mut updater, Now::blank(1_000, body));
    assert_eq!(cycle.drives.certainty.activation, 0.0);
    assert_eq!(
        cycle.selection.selected_goal,
        Some(GoalId::new("seek_charger"))
    );
    assert_eq!(
        cycle.behavior.unwrap().behavior_id,
        "systematic_charger_search"
    );
}

#[test]
fn investigate_publishes_three_targeted_information_gathering_behaviors() {
    let mut system = GoalSystem::default();
    let mut updater = WorldModelUpdater::default();
    let mut body = BodySense::default();
    body.battery_level = 1.0;
    let cycle = tick_with_canonical_world(&mut system, &mut updater, Now::blank(1_000, body));
    let investigate = cycle
        .evaluations
        .iter()
        .find(|evaluation| evaluation.goal_id == GoalId::new("investigate"))
        .unwrap();
    let available = investigate
        .competence
        .affordances
        .iter()
        .filter(|affordance| affordance.available)
        .collect::<Vec<_>>();
    assert!(available.len() >= 3);
    for expected in ["scan_clearance", "inspect_path", "stop_and_observe_path"] {
        let affordance = available
            .iter()
            .find(|affordance| affordance.behavior_id == expected)
            .unwrap();
        assert!(affordance.epistemic_question_id.is_some());
        assert!(affordance.expected_information_gain > 0.0);
    }
}

#[test]
fn recognized_person_proposes_encounter_scoped_lua_greeting_goal() {
    let mut uncertain_updater = WorldModelUpdater::default();
    let mut uncertain_now = Now::blank(1_000, BodySense::default());
    uncertain_now.objects.observations.push(ObjectObservation {
        label: "person".to_string(),
        class: ObjectClass::Person,
        bearing_rad: 0.0,
        distance_m: Some(0.6),
        confidence: 0.9,
        source: ObjectObservationSource::Kinect,
    });
    let uncertain_world = uncertain_updater
        .update(uncertain_now, WorldModelUpdateContext::default())
        .world;
    let mut uncertain_system = GoalSystem::default();
    let uncertain = uncertain_system.tick(&uncertain_world, &[]).unwrap();
    let uncertain_social = uncertain
        .evaluations
        .iter()
        .find(|evaluation| evaluation.goal_id == GoalId::new("socialize"))
        .unwrap();
    assert_eq!(
        uncertain_social.competence.affordances[0].action,
        Some(ActionPrimitive::Speak {
            text: "Hello. What should I call you?".to_string()
        })
    );

    let mut known_updater = WorldModelUpdater::default();
    let mut known_now = Now::blank(1_000, BodySense::default());
    known_now.objects.observations.push(ObjectObservation {
        label: "Alex".to_string(),
        class: ObjectClass::Person,
        bearing_rad: 0.0,
        distance_m: Some(0.6),
        confidence: 0.9,
        source: ObjectObservationSource::Kinect,
    });
    let known_world = known_updater
        .update(known_now, WorldModelUpdateContext::default())
        .world;
    let mut known_system = GoalSystem::default();
    let known = known_system.tick(&known_world, &[]).unwrap();
    let greeting = known
        .evaluations
        .iter()
        .find(|evaluation| evaluation.goal_id == GoalId::new("greet_person"))
        .unwrap();
    let affordance = &greeting.competence.affordances[0];
    assert_eq!(
        affordance.action,
        Some(ActionPrimitive::Speak {
            text: "Greet Alex".to_string()
        })
    );
    let request = affordance
        .skill_request
        .as_ref()
        .expect("greeting uses procedural memory");
    assert_eq!(request.skill_id, SkillId::RuntimeLoaded);
    assert_eq!(
        request.implementation_id.as_deref(),
        Some("motherbrain.greet")
    );
    assert_eq!(request.progress_metric, "social_acknowledgment");
    assert_eq!(
        known.selection.selected_goal,
        Some(GoalId::new("greet_person"))
    );
    assert!(known
        .behavior
        .as_ref()
        .unwrap()
        .behavior_id
        .starts_with("greet:person:alex:interaction:"));
    assert!(known
        .evaluations
        .iter()
        .find(|evaluation| evaluation.goal_id == GoalId::new("socialize"))
        .unwrap()
        .competence
        .affordances
        .is_empty());

    let mut acknowledged_world = known_world.clone();
    let interaction = acknowledged_world
        .social
        .active_interaction
        .as_mut()
        .unwrap();
    interaction
        .acknowledgments
        .push(pete_now::SocialAcknowledgment {
            acknowledgment_id: "greet-once".to_string(),
            kind: SocialAcknowledgmentKind::GreetingAttempted,
            person_id: pete_now::PersonId("person:alex".to_string()),
            occurred_at_ms: 1_100,
            skill_id: "motherbrain.greet".to_string(),
            skill_execution_id: 1,
            provenance: Vec::new(),
        });
    let mut acknowledged_system = GoalSystem::default();
    let acknowledged = acknowledged_system.tick(&acknowledged_world, &[]).unwrap();
    let greeting = acknowledged
        .evaluations
        .iter()
        .find(|evaluation| evaluation.goal_id == GoalId::new("greet_person"))
        .unwrap();
    assert_eq!(greeting.disposition, GoalDisposition::Satisfied);
    assert!(greeting.competence.affordances.is_empty());
    assert_ne!(
        acknowledged.selection.selected_goal,
        Some(GoalId::new("greet_person"))
    );
}

#[test]
fn immediate_danger_outranks_a_new_recognized_encounter() {
    let mut updater = WorldModelUpdater::default();
    let mut body = BodySense::default();
    body.flags.bump_left = true;
    let mut now = Now::blank(1_000, body);
    now.objects.observations.push(ObjectObservation {
        label: "Alex".to_string(),
        class: ObjectClass::Person,
        bearing_rad: 0.0,
        distance_m: Some(0.6),
        confidence: 0.9,
        source: ObjectObservationSource::Kinect,
    });
    let world = updater
        .update(now, WorldModelUpdateContext::default())
        .world;
    let mut system = GoalSystem::default();
    let cycle = system.tick(&world, &[]).unwrap();
    assert!(cycle
        .evaluations
        .iter()
        .find(|evaluation| evaluation.goal_id == GoalId::new("greet_person"))
        .unwrap()
        .competence
        .affordances
        .iter()
        .any(|affordance| affordance.available));
    assert_eq!(
        cycle.selection.selected_goal,
        Some(GoalId::new("escape_danger"))
    );
}

#[test]
fn behavior_expectations_use_the_predicted_clock_domain() {
    let mut system = GoalSystem::default();
    let mut updater = WorldModelUpdater::default();
    let mut body = BodySense::default();
    body.battery_level = 0.05;
    tick_with_canonical_world(&mut system, &mut updater, Now::blank(1_000, body));
    let context = system.world_model_update_context();
    assert_eq!(context.temporal_expectations.len(), 1);
    assert_eq!(
        context.temporal_expectations[0].expected_interval.domain,
        ClockDomain::Predicted
    );
}
