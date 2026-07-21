use super::*;
use pete_actions::{ReignCommand, ReignMode};

fn input_with_body(body: BodySense) -> ConductorInput {
    ConductorInput {
        latent: ExperienceLatent::default(),
        drives: DriveSense::default(),
        memory: MemorySense::default(),
        predictions: PredictionSense::default(),
        surprise: SurpriseSense::default(),
        llm: LlmSense::default(),
        safety: SafetySense::default(),
        reign: ReignSense::default(),
        range: RangeSense::default(),
        body,
        charger_near_score: 0.0,
        charger_visible_score: 0.0,
        proposals: Vec::new(),
    }
}

#[test]
fn critical_battery_stops_and_asks_when_charger_unknown() {
    let mut conductor = SimpleConductor::default();
    let mut body = BodySense::default();
    body.battery_level = 0.05;
    let input = input_with_body(body);

    let decision = conductor.choose_with_navigation_goal(input).unwrap();
    assert_eq!(
        decision.intent,
        NavigationIntent::StopAskForHelpWhenUncertain
    );
    assert_eq!(decision.action, ActionPrimitive::Stop);
    assert!(decision.confidence < 0.35);
    assert!(decision.reason.contains("no charger memory"));
}

#[test]
fn critical_battery_docks_only_when_charger_contact_is_plausible() {
    let mut conductor = SimpleConductor::default();
    let mut body = BodySense::default();
    body.battery_level = 0.05;
    let mut input = input_with_body(body);
    input.charger_near_score = 0.95;

    assert_eq!(conductor.choose(input).unwrap(), ActionPrimitive::Dock);
}

#[test]
fn critical_battery_remains_stopped_when_already_charging() {
    let mut conductor = SimpleConductor::default();
    let mut body = BodySense::default();
    body.battery_level = 0.05;
    body.charging = true;
    let mut input = input_with_body(body);
    input.charger_near_score = 0.95;

    let decision = conductor.choose_with_navigation_goal(input).unwrap();

    assert_eq!(decision.intent, NavigationIntent::RemainCharging);
    assert_eq!(decision.action, ActionPrimitive::Stop);
    assert!(decision.reason.contains("already established"));
}

#[test]
fn low_battery_remains_stopped_when_already_charging() {
    let mut conductor = SimpleConductor::default();
    let mut body = BodySense::default();
    body.battery_level = 0.15;
    body.charging = true;
    let input = input_with_body(body);

    let decision = conductor.choose_with_navigation_goal(input).unwrap();

    assert_eq!(decision.intent, NavigationIntent::RemainCharging);
    assert_eq!(decision.action, ActionPrimitive::Stop);
}

#[test]
fn visible_charger_is_approached_before_docking() {
    let mut conductor = SimpleConductor::default();
    let mut body = BodySense::default();
    body.battery_level = 0.15;
    let mut input = input_with_body(body);
    input.charger_visible_score = 0.45;

    assert_eq!(
        conductor.choose(input).unwrap(),
        ActionPrimitive::Approach {
            target: ApproachTarget::Charger
        }
    );
}

#[test]
fn low_confidence_charger_memory_searches_by_bearing() {
    let mut conductor = SimpleConductor::default();
    let mut body = BodySense::default();
    body.battery_level = 0.15;
    let mut input = input_with_body(body);
    input.memory.place_charge_value = 0.3;
    input.memory.nearby_best_charge_direction_rad = Some(-0.7);

    let decision = conductor.choose_with_navigation_goal(input).unwrap();
    assert_eq!(decision.intent, NavigationIntent::GoTowardKnownCharger);
    assert_eq!(
        decision.action,
        ActionPrimitive::Turn {
            direction: TurnDir::Right,
            intensity: 0.35,
            duration_ms: 700
        }
    );
    assert!(decision.reason.contains("charger memory"));
}

#[test]
fn bump_triggers_bounded_recovery_sequence() {
    let mut conductor = SimpleConductor::default();
    let mut body = BodySense::default();
    body.flags.bump_left = true;
    let mut input = input_with_body(body);
    input.range.beams = vec![0.2, 0.2, 0.8, 0.9, 0.9, 0.9];

    assert_eq!(
        conductor.choose(input.clone()).unwrap(),
        ActionPrimitive::Go {
            intensity: -0.05,
            duration_ms: 500
        }
    );
    input.body.flags.bump_left = false;
    input.body.odometry.x_m = -RECOVERY_REVERSE_BASE_TARGET_M;
    assert_eq!(
        conductor.choose(input).unwrap(),
        ActionPrimitive::Turn {
            direction: TurnDir::Right,
            intensity: 0.5,
            duration_ms: 500
        }
    );
}

#[test]
fn wheel_drop_vetoes_recovery() {
    let mut conductor = SimpleConductor::default();
    let mut body = BodySense::default();
    body.flags.bump_left = true;
    body.flags.wheel_drop = true;

    assert_eq!(
        conductor.choose(input_with_body(body)).unwrap(),
        ActionPrimitive::Stop
    );
}

#[test]
fn cramped_stationary_range_triggers_recovery() {
    let mut conductor = SimpleConductor::default();
    let body = BodySense::default();
    let mut input = input_with_body(body);
    input.range.nearest_m = Some(0.12);
    input.range.beams = vec![0.2, 0.2, 0.8, 0.8, 0.2, 0.2];

    assert_eq!(
        conductor.choose(input.clone()).unwrap(),
        ActionPrimitive::Turn {
            direction: TurnDir::Left,
            intensity: 0.5,
            duration_ms: 500
        }
    );
}

#[test]
fn contact_recovery_reverses_before_turning() {
    let mut conductor = SimpleConductor::default();
    let mut body = BodySense::default();
    body.flags.bump_right = true;
    let mut input = input_with_body(body);
    input.range.beams = vec![0.9, 0.9, 0.8, 0.2, 0.2, 0.2];

    assert_eq!(
        conductor.choose(input.clone()).unwrap(),
        ActionPrimitive::Go {
            intensity: -0.05,
            duration_ms: 500
        }
    );
    input.body.flags.bump_right = false;
    input.body.odometry.x_m = -RECOVERY_REVERSE_BASE_TARGET_M;
    assert_eq!(
        conductor.choose(input).unwrap(),
        ActionPrimitive::Turn {
            direction: TurnDir::Left,
            intensity: 0.5,
            duration_ms: 500
        }
    );
}

#[test]
fn contact_recovery_advances_only_after_observed_phase_progress() {
    let mut conductor = SimpleConductor::default();
    let mut body = BodySense::default();
    body.flags.bump_left = true;
    let mut input = input_with_body(body);
    input.range.beams = vec![0.9; 6];

    let reverse = conductor
        .choose_with_navigation_goal(input.clone())
        .unwrap();
    assert!(matches!(
        reverse.action,
        ActionPrimitive::Go {
            intensity: -0.05,
            ..
        }
    ));
    assert!(reverse.reason.contains("0/80 mm"));

    input.body.flags.bump_left = false;
    input.body.odometry.x_m = -0.08;
    let turn = conductor
        .choose_with_navigation_goal(input.clone())
        .unwrap();
    assert!(matches!(
        turn.action,
        ActionPrimitive::Turn {
            direction: TurnDir::Right,
            intensity: 0.5,
            ..
        }
    ));
    assert!(turn.reason.contains("0/1570 mrad"));

    input.body.odometry.heading_rad = -1.57;
    let probe = conductor
        .choose_with_navigation_goal(input.clone())
        .unwrap();
    assert!(matches!(
        probe.action,
        ActionPrimitive::Go {
            intensity: 0.05,
            ..
        }
    ));
    assert!(probe.reason.contains("0/50 mm"));

    input.body.odometry.x_m = -0.02;
    let inspect = conductor.choose_with_navigation_goal(input).unwrap();
    assert_eq!(
        inspect.action,
        ActionPrimitive::Inspect {
            target: InspectTarget::Novelty
        }
    );
    assert!(inspect
        .reason
        .contains("observed reverse, turn, and probe progress"));
}

#[test]
fn recovery_does_not_credit_motion_in_the_wrong_direction() {
    let mut conductor = SimpleConductor::default();
    let mut body = BodySense::default();
    body.flags.bump_left = true;
    let mut input = input_with_body(body);
    input.range.beams = vec![0.9; 6];
    let _ = conductor.choose(input.clone()).unwrap();

    input.body.flags.bump_left = false;
    input.body.odometry.x_m = 0.20;
    let wrong_way_reverse = conductor
        .choose_with_navigation_goal(input.clone())
        .unwrap();
    assert!(matches!(
        wrong_way_reverse.action,
        ActionPrimitive::Go {
            intensity: -0.05,
            ..
        }
    ));
    assert!(wrong_way_reverse.reason.contains("0/80 mm"));

    input.body.odometry.x_m = -0.08;
    let _ = conductor.choose(input.clone()).unwrap();
    input.body.odometry.heading_rad = 2.0;
    let wrong_way_turn = conductor.choose_with_navigation_goal(input).unwrap();
    assert!(matches!(
        wrong_way_turn.action,
        ActionPrimitive::Turn {
            direction: TurnDir::Right,
            ..
        }
    ));
    assert!(wrong_way_turn.reason.contains("0/1570 mrad"));
}

#[test]
fn absent_odometry_progress_escalates_then_stops() {
    let mut conductor = SimpleConductor::default();
    let mut body = BodySense::default();
    body.flags.bump_left = true;
    let mut input = input_with_body(body);
    input.range.beams = vec![0.9; 6];
    let mut saw_second_attempt = false;
    let mut saw_alternate_turn = false;
    let mut stopped = None;

    for _ in 0..350 {
        let decision = conductor
            .choose_with_navigation_goal(input.clone())
            .unwrap();
        saw_second_attempt |= decision.reason.contains("escape attempt 2");
        saw_alternate_turn |= matches!(
            decision.action,
            ActionPrimitive::Turn {
                direction: TurnDir::Left,
                ..
            }
        );
        if decision.action == ActionPrimitive::Stop
            && decision.reason.contains("no mechanically useful progress")
        {
            stopped = Some(decision);
            break;
        }
    }

    assert!(saw_second_attempt);
    assert!(saw_alternate_turn);
    let stopped = stopped.expect("recovery should stop after bounded escalation");
    assert!(stopped.reason.contains("3 attempts"));
    assert!(stopped.reason.contains("stalled odometry phases"));
}

#[test]
fn dangerous_place_turns_toward_remembered_safe_direction() {
    let mut conductor = SimpleConductor::default();
    let mut input = input_with_body(BodySense::default());
    input.memory.place_danger = 0.9;
    input.memory.nearby_best_safe_direction_rad = Some(-0.8);
    input.range.beams = vec![0.9, 0.9, 0.9, 0.1, 0.1, 0.1];

    let decision = conductor.choose_with_navigation_goal(input).unwrap();
    assert_eq!(decision.intent, NavigationIntent::AvoidKnownDangerCell);
    assert_eq!(
        decision.action,
        ActionPrimitive::Turn {
            direction: TurnDir::Right,
            intensity: 0.5,
            duration_ms: 1_000
        }
    );
    assert!(decision.reason.contains("danger memory"));
}

#[test]
fn low_battery_turns_toward_remembered_charger_before_approach() {
    let mut conductor = SimpleConductor::default();
    let mut body = BodySense::default();
    body.battery_level = 0.15;
    let mut input = input_with_body(body);
    input.memory.place_charge_value = 0.8;
    input.memory.nearby_best_charge_direction_rad = Some(0.7);

    assert_eq!(
        conductor.choose(input).unwrap(),
        ActionPrimitive::Turn {
            direction: TurnDir::Left,
            intensity: 0.4,
            duration_ms: 700
        }
    );
}

#[test]
fn low_battery_approaches_charger_when_memory_bearing_is_aligned() {
    let mut conductor = SimpleConductor::default();
    let mut body = BodySense::default();
    body.battery_level = 0.15;
    let mut input = input_with_body(body);
    input.memory.place_charge_value = 0.8;
    input.memory.nearby_best_charge_direction_rad = Some(0.05);

    assert_eq!(
        conductor.choose(input).unwrap(),
        ActionPrimitive::Approach {
            target: ApproachTarget::Charger
        }
    );
}

#[test]
fn safe_novel_place_inspects_before_default_explore() {
    let mut conductor = SimpleConductor::default();
    let mut input = input_with_body(BodySense::default());
    input.memory.place_novelty = 0.9;

    assert_eq!(
        conductor.choose(input).unwrap(),
        ActionPrimitive::Inspect {
            target: InspectTarget::Novelty
        }
    );
}

#[test]
fn safe_novel_frontier_turns_before_inspect() {
    let mut conductor = SimpleConductor::default();
    let mut input = input_with_body(BodySense::default());
    input.memory.place_novelty = 0.9;
    input.memory.nearby_frontier_direction_rad = Some(-0.6);

    assert_eq!(
        conductor.choose(input).unwrap(),
        ActionPrimitive::Turn {
            direction: TurnDir::Right,
            intensity: 0.35,
            duration_ms: 500
        }
    );
}

#[test]
fn recent_trap_turns_toward_remembered_safe_direction() {
    let mut conductor = SimpleConductor::default();
    let mut input = input_with_body(BodySense::default());
    input.memory.recent_trap_confidence = 0.8;
    input.memory.nearby_best_safe_direction_rad = Some(0.7);

    assert_eq!(
        conductor.choose(input).unwrap(),
        ActionPrimitive::Turn {
            direction: TurnDir::Left,
            intensity: 0.55,
            duration_ms: 800
        }
    );
}

#[test]
fn direct_reign_overrides_default_curiosity_drive() {
    let mut conductor = SimpleConductor::default();
    let command = ReignCommand::Turn {
        direction: TurnDir::Right,
        intensity: 0.4,
        duration_ms: 500,
    };
    let mut reign = ReignSense::default();
    reign.active = true;
    reign.mode = Some(ReignMode::Direct);
    reign.latest = Some(pete_actions::ReignInput {
        id: Default::default(),
        issued_at_ms: 100,
        expires_at_ms: 1_000,
        source: pete_actions::ReignSource::WebRemote,
        mode: ReignMode::Direct,
        command: command.clone(),
        priority: 1.0,
        note: None,
    });
    let mut drives = DriveSense::default();
    drives.curiosity = 1.0;
    let input = ConductorInput {
        latent: ExperienceLatent::default(),
        drives,
        memory: MemorySense::default(),
        predictions: PredictionSense::default(),
        surprise: SurpriseSense::default(),
        llm: LlmSense::default(),
        safety: SafetySense::default(),
        reign,
        range: RangeSense::default(),
        body: BodySense::default(),
        charger_near_score: 0.0,
        charger_visible_score: 0.0,
        proposals: Vec::new(),
    };

    assert_eq!(
        conductor.choose(input).unwrap(),
        command.to_action().unwrap()
    );
}

#[test]
fn assist_reign_overrides_default_curiosity_drive_without_proposal() {
    let mut conductor = SimpleConductor::default();
    let command = ReignCommand::Turn {
        direction: TurnDir::Right,
        intensity: 0.4,
        duration_ms: 500,
    };
    let mut reign = ReignSense::default();
    reign.active = true;
    reign.mode = Some(ReignMode::Assist);
    reign.latest = Some(pete_actions::ReignInput {
        id: Default::default(),
        issued_at_ms: 100,
        expires_at_ms: 1_000,
        source: pete_actions::ReignSource::WebRemote,
        mode: ReignMode::Assist,
        command: command.clone(),
        priority: 0.8,
        note: None,
    });
    let mut drives = DriveSense::default();
    drives.curiosity = 1.0;
    let input = ConductorInput {
        latent: ExperienceLatent::default(),
        drives,
        memory: MemorySense::default(),
        predictions: PredictionSense::default(),
        surprise: SurpriseSense::default(),
        llm: LlmSense::default(),
        safety: SafetySense::default(),
        reign,
        range: RangeSense::default(),
        body: BodySense::default(),
        charger_near_score: 0.0,
        charger_visible_score: 0.0,
        proposals: Vec::new(),
    };

    assert_eq!(
        conductor.choose(input).unwrap(),
        command.to_action().unwrap()
    );
}
