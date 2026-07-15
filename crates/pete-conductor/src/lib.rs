use anyhow::Result;
use pete_actions::{ActionPrimitive, ApproachTarget, ExploreStyle, InspectTarget, TurnDir};
use pete_body::BodySense;
use pete_experience::ExperienceLatent;
use pete_now::{
    DriveSense, LlmSense, MemorySense, PredictionSense, RangeSense, ReignSense, SafetySense,
    SurpriseSense,
};
use serde::{Deserialize, Serialize};

pub mod goal_architecture;

pub use goal_architecture::{
    Affordance, BehaviorDecision, Competence, DriveDynamics, DriveSnapshot, EvidenceRef,
    GoalArbiter, GoalArbiterConfig, GoalCycle, GoalEvaluation, GoalId, GoalRuntimeState,
    GoalSystem, Motivation, WorldEntity, WorldEntityKind, WorldModelSnapshot,
};

pub trait Conductor {
    fn choose(&mut self, input: ConductorInput) -> Result<ActionPrimitive>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConductorInput {
    pub latent: ExperienceLatent,
    pub drives: DriveSense,
    pub memory: MemorySense,
    pub predictions: PredictionSense,
    pub surprise: SurpriseSense,
    pub llm: LlmSense,
    pub safety: SafetySense,
    pub reign: ReignSense,
    pub range: RangeSense,
    pub body: BodySense,
    #[serde(default)]
    pub charger_near_score: f32,
    #[serde(default)]
    pub charger_visible_score: f32,
    pub proposals: Vec<ActionPrimitive>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConductorConfig {
    pub critical_battery: f32,
    pub low_battery: f32,
    pub danger_threshold: f32,
    pub novelty_threshold: f32,
}

impl Default for ConductorConfig {
    fn default() -> Self {
        Self {
            critical_battery: 0.10,
            low_battery: 0.20,
            danger_threshold: 0.70,
            novelty_threshold: 0.50,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NavigationIntent {
    GoTowardKnownCharger,
    RemainCharging,
    AvoidKnownDangerCell,
    InspectSafeNovelFrontier,
    ReturnToFamiliarSafeCell,
    StopAskForHelpWhenUncertain,
    FollowProposal,
    RecoverFromContact,
    ObeyDirectControl,
    Explore,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NavigationGoalDecision {
    pub intent: NavigationIntent,
    pub action: ActionPrimitive,
    pub confidence: f32,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum RecoveryStep {
    #[default]
    Idle,
    Reverse,
    Turn,
    Probe,
    Inspect,
}

#[derive(Clone, Debug, Default)]
struct RecoveryState {
    step: RecoveryStep,
    remaining_ticks: usize,
    turn_direction: Option<TurnDir>,
}

#[derive(Clone, Debug, Default)]
pub struct SimpleConductor {
    pub config: ConductorConfig,
    recovery: RecoveryState,
}

impl Conductor for SimpleConductor {
    fn choose(&mut self, input: ConductorInput) -> Result<ActionPrimitive> {
        Ok(self.choose_with_navigation_goal(input)?.action)
    }
}

impl SimpleConductor {
    pub fn choose_with_navigation_goal(
        &mut self,
        input: ConductorInput,
    ) -> Result<NavigationGoalDecision> {
        if let Some(action) = reign_action(&input) {
            self.recovery = RecoveryState::default();
            return Ok(navigation_goal(
                NavigationIntent::ObeyDirectControl,
                action,
                1.0,
                "direct Reign command is active",
            ));
        }
        if input.body.flags.wheel_drop {
            self.recovery = RecoveryState::default();
            return Ok(navigation_goal(
                NavigationIntent::StopAskForHelpWhenUncertain,
                ActionPrimitive::Stop,
                1.0,
                "wheel drop safety signal requires stopping",
            ));
        }
        if input.body.charging {
            self.recovery = RecoveryState::default();
            return Ok(navigation_goal(
                NavigationIntent::RemainCharging,
                ActionPrimitive::Stop,
                1.0,
                "charging is already established; remain stationary",
            ));
        }
        let charge_context = charge_context(&input);
        if charge_context.charging_established {
            self.recovery = RecoveryState::default();
            return Ok(navigation_goal(
                NavigationIntent::RemainCharging,
                ActionPrimitive::Stop,
                1.0,
                "charging is already established; remain stationary",
            ));
        }
        if input.body.battery_level <= self.config.critical_battery {
            self.recovery = RecoveryState::default();
            let decision = critical_battery_charge_goal(&input, charge_context);
            return Ok(decision);
        }
        if self.recovery.step == RecoveryStep::Idle {
            if contact_recovery_triggered(&input) {
                self.start_contact_recovery(contact_turn_direction(&input));
            } else if cramped_and_not_advancing(&input) {
                if side_escape_gap(&input.range.beams) {
                    self.start_contact_recovery(clearer_turn_direction(&input.range));
                } else {
                    self.start_range_recovery(clearer_turn_direction(&input.range));
                }
            }
        }
        if let Some(action) = self.next_recovery_action(&input) {
            return Ok(navigation_goal(
                NavigationIntent::RecoverFromContact,
                action,
                0.9,
                "contact or cramped-range recovery is active",
            ));
        }
        if input.memory.recent_trap_confidence >= 0.6 {
            if let Some(direction) = input.memory.nearby_best_safe_direction_rad {
                return Ok(navigation_goal(
                    NavigationIntent::ReturnToFamiliarSafeCell,
                    ActionPrimitive::Turn {
                        direction: direction_from_bearing(direction),
                        intensity: 0.55,
                        duration_ms: 800,
                    },
                    input.memory.recent_trap_confidence.clamp(0.0, 1.0),
                    format!("recent trap memory points to safe bearing {direction:.2} rad"),
                ));
            }
        }
        if input.memory.place_danger >= self.config.danger_threshold
            || input.drives.danger_avoidance >= self.config.danger_threshold
        {
            let remembered = input.memory.nearby_best_safe_direction_rad;
            let direction = remembered
                .map(direction_from_bearing)
                .unwrap_or_else(|| clearer_turn_direction(&input.range));
            return Ok(navigation_goal(
                NavigationIntent::AvoidKnownDangerCell,
                ActionPrimitive::Turn {
                    direction,
                    intensity: 0.5,
                    duration_ms: 1_000,
                },
                input
                    .memory
                    .place_danger
                    .max(input.drives.danger_avoidance)
                    .clamp(0.0, 1.0),
                remembered
                    .map(|bearing| {
                        format!("danger memory marks this place and safe bearing {bearing:.2} rad")
                    })
                    .unwrap_or_else(|| {
                        "danger signal is high; using range clearance as map hint".to_string()
                    }),
            ));
        }
        if input.body.battery_level <= self.config.low_battery {
            if charge_context.dock_plausible {
                return Ok(navigation_goal(
                    NavigationIntent::GoTowardKnownCharger,
                    ActionPrimitive::Dock,
                    0.95,
                    "charger contact is plausible from proximity and dock prediction",
                ));
            }
            if charge_context.should_approach {
                if let Some(direction) = input
                    .memory
                    .nearby_best_charge_direction_rad
                    .and_then(charge_alignment_turn)
                {
                    return Ok(navigation_goal(
                        NavigationIntent::GoTowardKnownCharger,
                        ActionPrimitive::Turn {
                            direction,
                            intensity: 0.4,
                            duration_ms: 700,
                        },
                        charger_goal_confidence(&input),
                        format!(
                            "charger memory/sensor signal says align toward bearing {:.2} rad",
                            input
                                .memory
                                .nearby_best_charge_direction_rad
                                .unwrap_or_default()
                        ),
                    ));
                }
                return Ok(navigation_goal(
                    NavigationIntent::GoTowardKnownCharger,
                    ActionPrimitive::Approach {
                        target: ApproachTarget::Charger,
                    },
                    charger_goal_confidence(&input),
                    "charger signal is present or remembered and bearing is aligned",
                ));
            }
            if charge_context.should_search {
                if let Some(direction) = input
                    .memory
                    .nearby_best_charge_direction_rad
                    .map(direction_from_bearing)
                {
                    return Ok(navigation_goal(
                        NavigationIntent::GoTowardKnownCharger,
                        ActionPrimitive::Turn {
                            direction,
                            intensity: 0.35,
                            duration_ms: 700,
                        },
                        charger_goal_confidence(&input).max(0.35),
                        format!(
                            "low-confidence charger memory suggests bearing {:.2} rad",
                            input
                                .memory
                                .nearby_best_charge_direction_rad
                                .unwrap_or_default()
                        ),
                    ));
                }
                return Ok(navigation_goal(
                    NavigationIntent::Explore,
                    ActionPrimitive::Explore {
                        style: ExploreStyle::Wander,
                        duration_ms: 1_000,
                    },
                    0.25,
                    "battery is low but no charger bearing is known",
                ));
            }
        }
        if let Some(action) = input.proposals.last() {
            return Ok(navigation_goal(
                NavigationIntent::FollowProposal,
                action.clone(),
                0.5,
                "using latest typed action proposal",
            ));
        }
        if input.drives.curiosity >= self.config.novelty_threshold
            || (input.memory.place_novelty >= self.config.novelty_threshold
                && input.memory.place_danger < self.config.danger_threshold)
        {
            if let Some(direction) = input.memory.nearby_frontier_direction_rad {
                return Ok(navigation_goal(
                    NavigationIntent::InspectSafeNovelFrontier,
                    ActionPrimitive::Turn {
                        direction: direction_from_bearing(direction),
                        intensity: 0.35,
                        duration_ms: 500,
                    },
                    input
                        .memory
                        .place_novelty
                        .max(input.drives.curiosity)
                        .clamp(0.0, 1.0),
                    format!("safe novelty memory points to frontier bearing {direction:.2} rad"),
                ));
            }
            return Ok(navigation_goal(
                NavigationIntent::InspectSafeNovelFrontier,
                ActionPrimitive::Inspect {
                    target: InspectTarget::Novelty,
                },
                input
                    .memory
                    .place_novelty
                    .max(input.drives.curiosity)
                    .clamp(0.0, 1.0),
                "place is novel and not remembered as dangerous",
            ));
        }
        Ok(navigation_goal(
            NavigationIntent::Explore,
            ActionPrimitive::Explore {
                style: ExploreStyle::RandomWalk,
                duration_ms: 1_000,
            },
            0.3,
            "no strong memory, map, or drive signal",
        ))
    }

    fn start_contact_recovery(&mut self, turn_direction: TurnDir) {
        self.recovery = RecoveryState {
            step: RecoveryStep::Reverse,
            remaining_ticks: 2,
            turn_direction: Some(turn_direction),
        };
    }

    fn start_range_recovery(&mut self, turn_direction: TurnDir) {
        self.recovery = RecoveryState {
            step: RecoveryStep::Turn,
            remaining_ticks: 9,
            turn_direction: Some(turn_direction),
        };
    }

    fn next_recovery_action(&mut self, input: &ConductorInput) -> Option<ActionPrimitive> {
        match self.recovery.step {
            RecoveryStep::Idle => None,
            RecoveryStep::Reverse => {
                self.advance_recovery(RecoveryStep::Turn, 7);
                Some(ActionPrimitive::Go {
                    intensity: -0.18,
                    duration_ms: 300,
                })
            }
            RecoveryStep::Turn => {
                let direction = self
                    .recovery
                    .turn_direction
                    .clone()
                    .unwrap_or(TurnDir::Left);
                self.advance_recovery(RecoveryStep::Probe, 3);
                Some(ActionPrimitive::Turn {
                    direction,
                    intensity: 0.75,
                    duration_ms: 500,
                })
            }
            RecoveryStep::Probe => {
                if center_clearance(&input.range.beams) < 0.30 {
                    self.recovery.step = RecoveryStep::Reverse;
                    self.recovery.remaining_ticks = 2;
                    return Some(ActionPrimitive::Go {
                        intensity: -0.16,
                        duration_ms: 300,
                    });
                }
                self.advance_recovery(RecoveryStep::Inspect, 1);
                Some(ActionPrimitive::Go {
                    intensity: 0.14,
                    duration_ms: 300,
                })
            }
            RecoveryStep::Inspect => {
                self.advance_recovery(RecoveryStep::Idle, 0);
                Some(ActionPrimitive::Inspect {
                    target: InspectTarget::Novelty,
                })
            }
        }
    }

    fn advance_recovery(&mut self, next_step: RecoveryStep, next_ticks: usize) {
        self.recovery.remaining_ticks = self.recovery.remaining_ticks.saturating_sub(1);
        if self.recovery.remaining_ticks == 0 {
            self.recovery.step = next_step;
            self.recovery.remaining_ticks = next_ticks;
        }
    }
}

fn reign_action(input: &ConductorInput) -> Option<ActionPrimitive> {
    let reign_input = input.reign.latest.as_ref()?;
    reign_input.command.to_action()
}

fn contact_recovery_triggered(input: &ConductorInput) -> bool {
    input.body.flags.bump_left || input.body.flags.bump_right || input.body.flags.wall
}

fn contact_turn_direction(input: &ConductorInput) -> TurnDir {
    if input.body.flags.bump_left && !input.body.flags.bump_right {
        TurnDir::Right
    } else if input.body.flags.bump_right && !input.body.flags.bump_left {
        TurnDir::Left
    } else {
        clearer_turn_direction(&input.range)
    }
}

fn cramped_and_not_advancing(input: &ConductorInput) -> bool {
    input
        .range
        .nearest_m
        .map(|nearest| nearest < 0.35)
        .unwrap_or(false)
        && input.body.velocity.forward_m_s.abs() < 0.02
}

fn clearer_turn_direction(range: &RangeSense) -> TurnDir {
    let (left, right) = range_clearance_sides(&range.beams);
    if right > left {
        TurnDir::Right
    } else {
        TurnDir::Left
    }
}

fn direction_from_bearing(bearing_rad: f32) -> TurnDir {
    if bearing_rad < 0.0 {
        TurnDir::Right
    } else {
        TurnDir::Left
    }
}

fn charge_alignment_turn(bearing_rad: f32) -> Option<TurnDir> {
    (bearing_rad.abs() > 0.20).then(|| direction_from_bearing(bearing_rad))
}

#[derive(Clone, Copy, Debug, Default)]
struct ChargeContext {
    charging_established: bool,
    dock_plausible: bool,
    should_approach: bool,
    should_search: bool,
}

fn charge_context(input: &ConductorInput) -> ChargeContext {
    let near = input.charger_near_score.clamp(0.0, 1.0);
    let visible = input.charger_visible_score.clamp(0.0, 1.0);
    let memory = input.memory.place_charge_value.clamp(0.0, 1.0);
    let prediction = input
        .predictions
        .charge_model
        .or(input.predictions.charge_hardcoded)
        .unwrap_or_default();
    let prediction_probability = prediction.charge_probability.clamp(0.0, 1.0);
    let dock_likelihood = prediction.dock_likelihood.clamp(0.0, 1.0);
    let charging_established = input.body.charging;
    let dock_plausible = near >= 0.92 || (near >= 0.80 && dock_likelihood >= 0.85);
    let should_approach = !charging_established
        && !dock_plausible
        && (visible >= 0.20 || near >= 0.25 || memory > 0.5 || prediction_probability >= 0.70);
    let should_search = !charging_established
        && !dock_plausible
        && !should_approach
        && (input.body.battery_level <= 0.20 || memory >= 0.25 || prediction_probability >= 0.35);
    ChargeContext {
        charging_established,
        dock_plausible,
        should_approach,
        should_search,
    }
}

fn navigation_goal(
    intent: NavigationIntent,
    action: ActionPrimitive,
    confidence: f32,
    reason: impl Into<String>,
) -> NavigationGoalDecision {
    NavigationGoalDecision {
        intent,
        action,
        confidence: confidence.clamp(0.0, 1.0),
        reason: reason.into(),
    }
}

fn charger_goal_confidence(input: &ConductorInput) -> f32 {
    let prediction = input
        .predictions
        .charge_model
        .or(input.predictions.charge_hardcoded)
        .unwrap_or_default();
    input
        .charger_near_score
        .max(input.charger_visible_score)
        .max(input.memory.place_charge_value)
        .max(prediction.charge_probability)
        .clamp(0.0, 1.0)
}

fn critical_battery_charge_goal(
    input: &ConductorInput,
    charge_context: ChargeContext,
) -> NavigationGoalDecision {
    if charge_context.dock_plausible {
        navigation_goal(
            NavigationIntent::GoTowardKnownCharger,
            ActionPrimitive::Dock,
            0.95,
            "critical battery and charger contact is plausible",
        )
    } else if charge_context.should_approach {
        navigation_goal(
            NavigationIntent::GoTowardKnownCharger,
            ActionPrimitive::Approach {
                target: ApproachTarget::Charger,
            },
            charger_goal_confidence(input).max(0.7),
            "critical battery and charger signal is strong enough to approach",
        )
    } else if let Some(direction) = input
        .memory
        .nearby_best_charge_direction_rad
        .map(direction_from_bearing)
    {
        navigation_goal(
            NavigationIntent::GoTowardKnownCharger,
            ActionPrimitive::Turn {
                direction,
                intensity: 0.35,
                duration_ms: 700,
            },
            charger_goal_confidence(input).max(0.35),
            format!(
                "critical battery and remembered charger bearing {:.2} rad",
                input
                    .memory
                    .nearby_best_charge_direction_rad
                    .unwrap_or_default()
            ),
        )
    } else {
        navigation_goal(
            NavigationIntent::StopAskForHelpWhenUncertain,
            ActionPrimitive::Stop,
            0.2,
            "critical battery but no charger memory, sensor, or map direction is reliable",
        )
    }
}

fn range_clearance_sides(beams: &[f32]) -> (f32, f32) {
    if beams.is_empty() {
        return (1.0, 1.0);
    }
    let third = (beams.len() / 3).max(1);
    let left_end = third.min(beams.len());
    let right_start = beams.len().saturating_sub(third);
    let left = beams[..left_end].iter().copied().fold(1.0, f32::min);
    let right = beams[right_start..].iter().copied().fold(1.0, f32::min);
    (left, right)
}

fn center_clearance(beams: &[f32]) -> f32 {
    if beams.is_empty() {
        return 1.0;
    }
    let third = (beams.len() / 3).max(1);
    let left_end = third.min(beams.len());
    let right_start = beams.len().saturating_sub(third);
    let center_start = left_end.saturating_sub(1).min(beams.len());
    let center_end = (right_start + 1).min(beams.len()).max(center_start + 1);
    beams[center_start..center_end]
        .iter()
        .copied()
        .fold(1.0, f32::min)
}

fn side_escape_gap(beams: &[f32]) -> bool {
    if beams.is_empty() {
        return false;
    }
    let third = (beams.len() / 3).max(1);
    let left_end = third.min(beams.len());
    let right_start = beams.len().saturating_sub(third);
    let left = beams[..left_end].iter().copied().fold(0.0, f32::max);
    let right = beams[right_start..].iter().copied().fold(0.0, f32::max);
    left.max(right) >= 0.75
}

#[cfg(test)]
mod tests {
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
                intensity: -0.18,
                duration_ms: 300
            }
        );
        input.body.flags.bump_left = false;
        for _ in 0..2 {
            let _ = conductor.choose(input.clone()).unwrap();
        }
        assert_eq!(
            conductor.choose(input).unwrap(),
            ActionPrimitive::Turn {
                direction: TurnDir::Right,
                intensity: 0.75,
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
                intensity: 0.75,
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
                intensity: -0.18,
                duration_ms: 300
            }
        );
        for _ in 0..2 {
            let _ = conductor.choose(input.clone()).unwrap();
        }
        assert_eq!(
            conductor.choose(input).unwrap(),
            ActionPrimitive::Turn {
                direction: TurnDir::Left,
                intensity: 0.75,
                duration_ms: 500
            }
        );
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
}
