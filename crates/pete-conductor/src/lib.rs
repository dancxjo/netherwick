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
    Affordance, BehaviorDecision, Competence, DriveDynamics, DriveSnapshot, Goal, GoalArbiter,
    GoalArbiterConfig, GoalCycle, GoalDisposition, GoalEvaluation, GoalEvaluationContext,
    GoalEvaluator, GoalExecutionContext, GoalExecutor, GoalExitReason, GoalId,
    GoalInterpretationContext, GoalInterpretationSnapshot, GoalInterpreter, GoalModule,
    GoalPerceptionContext, GoalPerceptionSnapshot, GoalProgressReport, GoalRuntimeState,
    GoalSystem, Motivation, ProgressExpectation, ProgressObservation, SkillId, SkillOutcome,
    SkillPhase, SkillRequest, SkillScriptStatus, SkillStatus, StrategyProgressResponse,
};
pub use pete_now::{EvidenceRef, WorldEntity, WorldEntityKind, WorldModelSnapshot};

pub trait Conductor {
    fn choose(&mut self, input: ConductorInput) -> Result<ActionPrimitive>;

    fn navigation_goal(&self) -> Option<&NavigationGoalDecision> {
        None
    }
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
    Stuck,
}

#[derive(Clone, Debug, Default)]
struct RecoveryState {
    step: RecoveryStep,
    remaining_ticks: usize,
    turn_direction: Option<TurnDir>,
    attempt: u8,
    phase_origin_distance_m: f32,
    phase_origin_heading_rad: f32,
    stalled_phases: u8,
}

#[derive(Clone, Debug, PartialEq)]
struct RecoveryDecision {
    action: ActionPrimitive,
    reason: String,
}

const RECOVERY_MAX_ATTEMPTS: u8 = 3;
const RECOVERY_REVERSE_BASE_TARGET_M: f32 = 0.08;
const RECOVERY_REVERSE_TARGET_STEP_M: f32 = 0.04;
const RECOVERY_REVERSE_MAX_TICKS: usize = 45;
const RECOVERY_TURN_TARGET_RAD: f32 = 1.57;
const RECOVERY_TURN_MIN_USEFUL_RAD: f32 = 0.30;
const RECOVERY_TURN_MAX_TICKS: usize = 45;
const RECOVERY_PROBE_TARGET_M: f32 = 0.05;
const RECOVERY_PROBE_MAX_TICKS: usize = 20;

#[derive(Clone, Debug, Default)]
pub struct SimpleConductor {
    pub config: ConductorConfig,
    recovery: RecoveryState,
    last_navigation_goal: Option<NavigationGoalDecision>,
}

impl Conductor for SimpleConductor {
    fn choose(&mut self, input: ConductorInput) -> Result<ActionPrimitive> {
        let decision = self.choose_with_navigation_goal(input)?;
        let action = decision.action.clone();
        self.last_navigation_goal = Some(decision);
        Ok(action)
    }

    fn navigation_goal(&self) -> Option<&NavigationGoalDecision> {
        self.last_navigation_goal.as_ref()
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
                self.start_contact_recovery(&input, contact_turn_direction(&input));
            } else if cramped_and_not_advancing(&input) {
                if side_escape_gap(&input.range.beams) {
                    self.start_contact_recovery(&input, clearer_turn_direction(&input.range));
                } else {
                    self.start_range_recovery(&input, clearer_turn_direction(&input.range));
                }
            }
        }
        if let Some(recovery) = self.next_recovery_action(&input) {
            return Ok(navigation_goal(
                NavigationIntent::RecoverFromContact,
                recovery.action,
                0.9,
                recovery.reason,
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

    fn start_contact_recovery(&mut self, input: &ConductorInput, turn_direction: TurnDir) {
        self.recovery = RecoveryState {
            step: RecoveryStep::Reverse,
            remaining_ticks: RECOVERY_REVERSE_MAX_TICKS,
            turn_direction: Some(turn_direction),
            attempt: 1,
            phase_origin_distance_m: input.body.odometry.x_m,
            phase_origin_heading_rad: input.body.odometry.heading_rad,
            stalled_phases: 0,
        };
    }

    fn start_range_recovery(&mut self, input: &ConductorInput, turn_direction: TurnDir) {
        self.recovery = RecoveryState {
            step: RecoveryStep::Turn,
            remaining_ticks: RECOVERY_TURN_MAX_TICKS,
            turn_direction: Some(turn_direction),
            attempt: 1,
            phase_origin_distance_m: input.body.odometry.x_m,
            phase_origin_heading_rad: input.body.odometry.heading_rad,
            stalled_phases: 0,
        };
    }

    fn next_recovery_action(&mut self, input: &ConductorInput) -> Option<RecoveryDecision> {
        // A transition can be recognized from fresh odometry at the start of
        // a tick. Loop only across state transitions; every returned motion
        // still corresponds to one short-lived primitive renewed by the
        // possession loop.
        loop {
            match self.recovery.step {
                RecoveryStep::Idle => return None,
                RecoveryStep::Reverse => {
                    let progress_m = self.recovery_reverse_progress(input);
                    let target_m = self.recovery_reverse_target_m();
                    if progress_m >= target_m || self.recovery.remaining_ticks == 0 {
                        if progress_m < 0.01 {
                            self.recovery.stalled_phases =
                                self.recovery.stalled_phases.saturating_add(1);
                        }
                        self.begin_recovery_phase(input, RecoveryStep::Turn);
                        continue;
                    }
                    self.recovery.remaining_ticks = self.recovery.remaining_ticks.saturating_sub(1);
                    return Some(RecoveryDecision {
                        action: ActionPrimitive::Go {
                            // Use the full default possession allowance. The
                            // hardware gate can still impose a lower operator
                            // limit; escape authority comes primarily from the
                            // observed distance target, not excess speed.
                            intensity: -0.05,
                            duration_ms: 500,
                        },
                        reason: format!(
                            "escape attempt {} reversing: {:.0}/{:.0} mm observed odometry",
                            self.recovery.attempt,
                            progress_m * 1_000.0,
                            target_m * 1_000.0
                        ),
                    });
                }
                RecoveryStep::Turn => {
                    let progress_rad = self.recovery_turn_progress(input);
                    if progress_rad >= RECOVERY_TURN_TARGET_RAD
                        || self.recovery.remaining_ticks == 0
                    {
                        if progress_rad < RECOVERY_TURN_MIN_USEFUL_RAD {
                            self.recovery.stalled_phases =
                                self.recovery.stalled_phases.saturating_add(1);
                            if self.begin_escalated_recovery_attempt(input) {
                                continue;
                            }
                            self.begin_recovery_phase(input, RecoveryStep::Stuck);
                            continue;
                        }
                        self.begin_recovery_phase(input, RecoveryStep::Probe);
                        continue;
                    }
                    self.recovery.remaining_ticks = self.recovery.remaining_ticks.saturating_sub(1);
                    let direction = self
                        .recovery
                        .turn_direction
                        .clone()
                        .unwrap_or(TurnDir::Left);
                    return Some(RecoveryDecision {
                        action: ActionPrimitive::Turn {
                            direction,
                            intensity: 0.5,
                            duration_ms: 500,
                        },
                        reason: format!(
                            "escape attempt {} turning {:?}: {:.0}/{:.0} mrad observed heading",
                            self.recovery.attempt,
                            self.recovery
                                .turn_direction
                                .as_ref()
                                .unwrap_or(&TurnDir::Left),
                            progress_rad * 1_000.0,
                            RECOVERY_TURN_TARGET_RAD * 1_000.0
                        ),
                    });
                }
                RecoveryStep::Probe => {
                    let progress_m = self.recovery_forward_progress(input);
                    let close_ahead = center_clearance(&input.range.beams) < 0.30;
                    if progress_m >= RECOVERY_PROBE_TARGET_M && !close_ahead {
                        self.begin_recovery_phase(input, RecoveryStep::Inspect);
                        continue;
                    }
                    if close_ahead || self.recovery.remaining_ticks == 0 {
                        if progress_m < 0.01 {
                            self.recovery.stalled_phases =
                                self.recovery.stalled_phases.saturating_add(1);
                        }
                        if self.begin_escalated_recovery_attempt(input) {
                            continue;
                        }
                        self.begin_recovery_phase(input, RecoveryStep::Stuck);
                        continue;
                    }
                    self.recovery.remaining_ticks = self.recovery.remaining_ticks.saturating_sub(1);
                    return Some(RecoveryDecision {
                        action: ActionPrimitive::Go {
                            intensity: 0.05,
                            duration_ms: 500,
                        },
                        reason: format!(
                            "escape attempt {} probing: {:.0}/{:.0} mm observed odometry",
                            self.recovery.attempt,
                            progress_m * 1_000.0,
                            RECOVERY_PROBE_TARGET_M * 1_000.0
                        ),
                    });
                }
                RecoveryStep::Inspect => {
                    self.recovery = RecoveryState::default();
                    return Some(RecoveryDecision {
                        action: ActionPrimitive::Inspect {
                            target: InspectTarget::Novelty,
                        },
                        reason: "escape completed with observed reverse, turn, and probe progress"
                            .to_string(),
                    });
                }
                RecoveryStep::Stuck => {
                    return Some(RecoveryDecision {
                        action: ActionPrimitive::Stop,
                        reason: format!(
                            "escape stopped after {} attempts and {} stalled odometry phases; no mechanically useful progress observed",
                            self.recovery.attempt, self.recovery.stalled_phases
                        ),
                    });
                }
            }
        }
    }

    fn recovery_reverse_target_m(&self) -> f32 {
        RECOVERY_REVERSE_BASE_TARGET_M
            + f32::from(self.recovery.attempt.saturating_sub(1)) * RECOVERY_REVERSE_TARGET_STEP_M
    }

    fn recovery_reverse_progress(&self, input: &ConductorInput) -> f32 {
        (self.recovery.phase_origin_distance_m - input.body.odometry.x_m).max(0.0)
    }

    fn recovery_forward_progress(&self, input: &ConductorInput) -> f32 {
        (input.body.odometry.x_m - self.recovery.phase_origin_distance_m).max(0.0)
    }

    fn recovery_turn_progress(&self, input: &ConductorInput) -> f32 {
        match self
            .recovery
            .turn_direction
            .as_ref()
            .unwrap_or(&TurnDir::Left)
        {
            TurnDir::Left => {
                (input.body.odometry.heading_rad - self.recovery.phase_origin_heading_rad).max(0.0)
            }
            TurnDir::Right => {
                (self.recovery.phase_origin_heading_rad - input.body.odometry.heading_rad).max(0.0)
            }
        }
    }

    fn begin_recovery_phase(&mut self, input: &ConductorInput, step: RecoveryStep) {
        self.recovery.step = step;
        self.recovery.remaining_ticks = match step {
            RecoveryStep::Reverse => RECOVERY_REVERSE_MAX_TICKS,
            RecoveryStep::Turn => RECOVERY_TURN_MAX_TICKS,
            RecoveryStep::Probe => RECOVERY_PROBE_MAX_TICKS,
            RecoveryStep::Inspect => 1,
            RecoveryStep::Stuck => 0,
            RecoveryStep::Idle => 0,
        };
        self.recovery.phase_origin_distance_m = input.body.odometry.x_m;
        self.recovery.phase_origin_heading_rad = input.body.odometry.heading_rad;
    }

    fn begin_escalated_recovery_attempt(&mut self, input: &ConductorInput) -> bool {
        if self.recovery.attempt >= RECOVERY_MAX_ATTEMPTS {
            return false;
        }
        self.recovery.attempt = self.recovery.attempt.saturating_add(1);
        self.recovery.turn_direction = Some(
            match self
                .recovery
                .turn_direction
                .clone()
                .unwrap_or(TurnDir::Left)
            {
                TurnDir::Left => TurnDir::Right,
                TurnDir::Right => TurnDir::Left,
            },
        );
        self.begin_recovery_phase(input, RecoveryStep::Reverse);
        true
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
#[path = "lib_tests.rs"]
mod tests;
