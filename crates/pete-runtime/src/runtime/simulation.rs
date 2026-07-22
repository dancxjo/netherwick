pub struct SimRunner<R> {
    pub runtime: R,
    pub world: VirtualWorld,
    pub cockpit: SafeCockpit<SimCockpit>,
    pub tick_count: usize,
    pub tick_ms: u64,
    stuck: StuckRecoveryController,
    possessor_skills: PossessorSkillRuntime,
}

const STUCK_LOW_DISPLACEMENT_TICKS: usize = 6;
const STUCK_WINDOW_DISPLACEMENT_EPSILON_M: f32 = 0.015;
const NEAR_ARENA_WALL_M: f32 = 0.32;
const SAME_TRAP_RADIUS_M: f32 = 0.18;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum RecoveryPhase {
    #[default]
    None,
    Stop,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum TrapKind {
    #[default]
    Unknown,
    Wall,
    Corner,
    Column,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct StuckStatus {
    active: bool,
    corner_trap: bool,
    trap_kind: TrapKind,
    stuck_ticks: usize,
    duration_ticks: usize,
    phase: RecoveryPhase,
    turn_sign: f32,
    recovery_attempts: usize,
    repeated_trap_count: usize,
    clearance_m: Option<f32>,
    event_started: bool,
    recovered: bool,
    dead_battery: bool,
    reset_due: bool,
}

#[derive(Clone, Debug, Default)]
struct StuckRecoveryController {
    last_position: Option<(f32, f32)>,
    displacement_window: VecDeque<f32>,
    commanded_window: VecDeque<bool>,
    stuck_ticks: usize,
    active: bool,
    corner_trap: bool,
    trap_kind: TrapKind,
    duration_ticks: usize,
    phase: RecoveryPhase,
    phase_ticks_remaining: usize,
    turn_sign: f32,
    recovery_attempts: usize,
    repeated_trap_count: usize,
    trap_anchor: Option<(f32, f32)>,
    last_failed_turn_sign: Option<f32>,
    clearance_m: Option<f32>,
    event_started: bool,
    recovered: bool,
    dead_battery: bool,
    reset_due: bool,
}

impl StuckRecoveryController {
    fn annotate_snapshot(&mut self, snapshot: &mut WorldSnapshot, tick_ms: u64) {
        self.dead_battery = is_dead_battery(snapshot);
        snapshot
            .extensions
            .retain(|extension| extension.name != "sim.stuck");
        snapshot.extensions.push(self.extension(tick_ms));
        self.event_started = false;
        self.recovered = false;
    }

    fn observe(&mut self, snapshot: &WorldSnapshot, action: Option<&ActionPrimitive>) {
        let was_active = self.active;
        let position = (snapshot.body.odometry.x_m, snapshot.body.odometry.y_m);
        let step_distance = self
            .last_position
            .map(|last| distance_between_points(last, position))
            .unwrap_or(f32::INFINITY);
        let commanded_motion = action_is_commanded_motion(action);
        self.dead_battery = is_dead_battery(snapshot);
        self.push_motion_sample(step_distance, commanded_motion);
        self.clearance_m = snapshot.range.nearest_m;
        let trap_kind = classify_trap_kind(snapshot);
        let trapped = trap_kind != TrapKind::Unknown;
        let stationary_column_or_corner = matches!(trap_kind, TrapKind::Column | TrapKind::Corner)
            && self.rolling_stationary()
            && matches!(action, Some(ActionPrimitive::Stop) | None);
        let low_displacement = (self.rolling_low_displacement() || stationary_column_or_corner)
            && !snapshot.body.charging;
        self.stuck_ticks = if low_displacement {
            self.stuck_ticks
                .saturating_add(1)
                .max(STUCK_LOW_DISPLACEMENT_TICKS)
        } else {
            if !self.active && step_distance > STUCK_WINDOW_DISPLACEMENT_EPSILON_M {
                self.recovery_attempts = 0;
            }
            0
        };

        if !self.active
            && (commanded_motion || stationary_column_or_corner)
            && self.stuck_ticks >= STUCK_LOW_DISPLACEMENT_TICKS
            && trapped
            && !self.dead_battery
        {
            self.active = true;
            self.corner_trap = trap_kind == TrapKind::Corner;
            self.trap_kind = trap_kind;
            self.duration_ticks = 1;
            self.phase = RecoveryPhase::Stop;
            self.phase_ticks_remaining = 1;
            if self
                .trap_anchor
                .map(|anchor| distance_between_points(anchor, position) <= SAME_TRAP_RADIUS_M)
                .unwrap_or(false)
            {
                self.repeated_trap_count = self.repeated_trap_count.saturating_add(1);
            } else {
                self.repeated_trap_count = 0;
                self.trap_anchor = Some(position);
                self.recovery_attempts = 0;
                self.last_failed_turn_sign = None;
            }
            self.recovery_attempts = self.recovery_attempts.saturating_add(1);
            self.turn_sign = recovery_turn_sign(snapshot, self.last_failed_turn_sign);
            self.event_started = true;
        } else if was_active {
            self.duration_ticks = self.duration_ticks.saturating_add(1);
        }

        let recovery_displacement = self
            .trap_anchor
            .map(|anchor| distance_between_points(anchor, position))
            .unwrap_or(0.0);
        if self.active && recovery_displacement >= STUCK_WINDOW_DISPLACEMENT_EPSILON_M {
            self.finish_recovery_success();
        }

        self.last_position = Some(position);
    }

    fn finish_recovery_success(&mut self) {
        self.active = false;
        self.corner_trap = false;
        self.trap_kind = TrapKind::Unknown;
        self.stuck_ticks = 0;
        self.phase = RecoveryPhase::None;
        self.phase_ticks_remaining = 0;
        self.trap_anchor = None;
        self.last_failed_turn_sign = None;
        self.recovered = true;
        self.displacement_window.clear();
        self.commanded_window.clear();
    }

    fn push_motion_sample(&mut self, step_distance: f32, commanded_motion: bool) {
        if step_distance.is_finite() {
            self.displacement_window.push_back(step_distance.max(0.0));
            self.commanded_window.push_back(commanded_motion);
        }
        while self.displacement_window.len() > STUCK_LOW_DISPLACEMENT_TICKS {
            self.displacement_window.pop_front();
        }
        while self.commanded_window.len() > STUCK_LOW_DISPLACEMENT_TICKS {
            self.commanded_window.pop_front();
        }
    }

    fn rolling_low_displacement(&self) -> bool {
        self.displacement_window.len() >= STUCK_LOW_DISPLACEMENT_TICKS
            && self.commanded_window.len() >= STUCK_LOW_DISPLACEMENT_TICKS
            && self.commanded_window.iter().all(|commanded| *commanded)
            && self.displacement_window.iter().sum::<f32>() < STUCK_WINDOW_DISPLACEMENT_EPSILON_M
    }

    fn rolling_stationary(&self) -> bool {
        self.displacement_window.len() >= STUCK_LOW_DISPLACEMENT_TICKS
            && self.displacement_window.iter().sum::<f32>() < STUCK_WINDOW_DISPLACEMENT_EPSILON_M
    }

    fn extension(&self, tick_ms: u64) -> ExtensionSense {
        let status = self.status();
        ExtensionSense {
            schema_version: 1,
            name: "sim.stuck".to_string(),
            values: vec![
                status.active as u8 as f32,
                status.corner_trap as u8 as f32,
                status.stuck_ticks as f32,
                (status.duration_ticks as u64).saturating_mul(tick_ms) as f32,
                recovery_phase_code(status.phase),
                status.turn_sign,
                status.event_started as u8 as f32,
                status.recovered as u8 as f32,
                status.dead_battery as u8 as f32,
                status.reset_due as u8 as f32,
                trap_kind_code(status.trap_kind),
                status.recovery_attempts as f32,
                status.repeated_trap_count as f32,
                status.clearance_m.unwrap_or(-1.0),
            ],
        }
    }

    fn status(&self) -> StuckStatus {
        StuckStatus {
            active: self.active,
            corner_trap: self.corner_trap,
            trap_kind: self.trap_kind,
            stuck_ticks: self.stuck_ticks,
            duration_ticks: self.duration_ticks,
            phase: self.phase,
            turn_sign: self.turn_sign,
            recovery_attempts: self.recovery_attempts,
            repeated_trap_count: self.repeated_trap_count,
            clearance_m: self.clearance_m,
            event_started: self.event_started,
            recovered: self.recovered,
            dead_battery: self.dead_battery,
            reset_due: self.reset_due,
        }
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

fn recovery_phase_code(phase: RecoveryPhase) -> f32 {
    match phase {
        RecoveryPhase::None => 0.0,
        RecoveryPhase::Stop => 1.0,
    }
}

fn trap_kind_code(kind: TrapKind) -> f32 {
    match kind {
        TrapKind::Unknown => 0.0,
        TrapKind::Wall => 1.0,
        TrapKind::Corner => 2.0,
        TrapKind::Column => 3.0,
    }
}

fn action_is_commanded_motion(action: Option<&ActionPrimitive>) -> bool {
    let motor = action_to_motor_command(action);
    motor.forward.abs() > 0.05 || motor.turn.abs() > 0.05
}

fn classify_trap_kind(snapshot: &WorldSnapshot) -> TrapKind {
    let body = &snapshot.body;
    let collision = body.flags.wall
        || body.flags.bump_left
        || body.flags.bump_right
        || body.flags.cliff_front_left
        || body.flags.cliff_front_right;
    let near = snapshot.range.nearest_m.unwrap_or(10.0) < NEAR_ARENA_WALL_M;
    let near_arena_wall = arena_bounds(snapshot)
        .map(|(width_m, height_m)| {
            snapshot.body.odometry.x_m < NEAR_ARENA_WALL_M
                || snapshot.body.odometry.y_m < NEAR_ARENA_WALL_M
                || width_m - snapshot.body.odometry.x_m < NEAR_ARENA_WALL_M
                || height_m - snapshot.body.odometry.y_m < NEAR_ARENA_WALL_M
        })
        .unwrap_or(false);
    let beams = &snapshot.range.beams;
    let (left, _center, right) = beam_clearance_buckets(beams);
    let side_constrained = left < 0.16 && right < 0.16;
    if near_arena_wall && side_constrained {
        TrapKind::Corner
    } else if near_arena_wall || body.flags.wall {
        TrapKind::Wall
    } else if collision || near {
        TrapKind::Column
    } else {
        TrapKind::Unknown
    }
}

fn recovery_turn_sign(snapshot: &WorldSnapshot, last_failed_turn_sign: Option<f32>) -> f32 {
    if let Some(sign) = bump_escape_turn_sign(snapshot) {
        return sign;
    }
    if let Some(last_failed) = last_failed_turn_sign {
        return -last_failed;
    }
    turn_toward_clearer_side(snapshot)
}

fn bump_escape_turn_sign(snapshot: &WorldSnapshot) -> Option<f32> {
    match (
        snapshot.body.flags.bump_left,
        snapshot.body.flags.bump_right,
        snapshot.body.flags.wall,
    ) {
        (true, false, _) => Some(-1.0),
        (false, true, _) => Some(1.0),
        (_, _, true) | (true, true, _) => Some(turn_toward_clearer_side(snapshot)),
        _ => None,
    }
}

fn turn_toward_clearer_side(snapshot: &WorldSnapshot) -> f32 {
    let beams = &snapshot.range.beams;
    if beams.len() < 2 {
        return 1.0;
    }
    let (left, _, right) = beam_clearance_buckets(beams);
    if left <= right {
        -1.0
    } else {
        1.0
    }
}

fn beam_clearance_buckets(beams: &[f32]) -> (f32, f32, f32) {
    if beams.is_empty() {
        return (1.0, 1.0, 1.0);
    }
    let third = (beams.len() / 3).max(1);
    let left_end = third.min(beams.len());
    let right_start = beams.len().saturating_sub(third);
    let center_start = left_end.saturating_sub(1).min(beams.len());
    let center_end = (right_start + 1).min(beams.len()).max(center_start + 1);
    let left = beams[..left_end].iter().copied().fold(1.0, f32::min);
    let center = beams[center_start..center_end]
        .iter()
        .copied()
        .fold(1.0, f32::min);
    let right = beams[right_start..].iter().copied().fold(1.0, f32::min);
    (left, center, right)
}

fn arena_bounds(snapshot: &WorldSnapshot) -> Option<(f32, f32)> {
    let world = snapshot
        .extensions
        .iter()
        .find(|extension| extension.name == "sim.world")?;
    let width_m = world.values.first().copied()?;
    let height_m = world.values.get(1).copied()?;
    (width_m > 0.0 && height_m > 0.0).then_some((width_m, height_m))
}

fn is_dead_battery(snapshot: &WorldSnapshot) -> bool {
    snapshot.body.battery_level <= f32::EPSILON && !snapshot.body.charging
}

fn sim_stuck_reset_due(snapshot: &WorldSnapshot) -> bool {
    snapshot
        .extensions
        .iter()
        .find(|extension| extension.name == "sim.stuck")
        .and_then(|extension| extension.values.get(9))
        .copied()
        .unwrap_or(0.0)
        > 0.0
}

fn distance_between_points(left: (f32, f32), right: (f32, f32)) -> f32 {
    let dx = left.0 - right.0;
    let dy = left.1 - right.1;
    (dx * dx + dy * dy).sqrt()
}

impl<R> SimRunner<R>
where
    R: RuntimeLoop + Send,
{
    pub fn new(runtime: R, world: VirtualWorld, motors: SimCockpit) -> Self {
        Self {
            runtime,
            world,
            cockpit: SafeCockpit::new(motors),
            tick_count: 0,
            tick_ms: 100,
            stuck: StuckRecoveryController::default(),
            possessor_skills: PossessorSkillRuntime::default(),
        }
    }

    pub async fn run_steps(&mut self, steps: usize) -> Result<()> {
        self.run_steps_observing(steps, |_| {}).await
    }

    pub async fn run_steps_observing<F>(&mut self, steps: usize, mut observe: F) -> Result<()>
    where
        F: FnMut(&WorldSnapshot),
    {
        self.run_steps_observing_ticks(steps, |snapshot, _tick| observe(snapshot))
            .await
    }

    pub async fn run_steps_observing_ticks<F>(&mut self, steps: usize, mut observe: F) -> Result<()>
    where
        F: FnMut(&WorldSnapshot, &RuntimeTick),
    {
        for _ in 0..steps {
            let mut snapshot = self.world.snapshot().await?;
            self.stuck.annotate_snapshot(&mut snapshot, self.tick_ms);
            let reset_after_tick = sim_stuck_reset_due(&snapshot);
            let body_pose_before = snapshot.body.clone();
            let mut now = snapshot.to_now(snapshot.body.last_update_ms);
            self.possessor_skills.annotate_now(&mut now);
            let mut tick = self
                .runtime
                .tick(now.clone(), ExperienceLatent::default(), Vec::new())
                .await?;
            let mut lua_skill_owns_motion = false;
            let selected_skill_request = self
                .possessor_skills
                .request_for_tick(tick.skill_request.clone());
            if let Some(request) = selected_skill_request {
                tick.skill_request = Some(request.clone());
                let status_summary = self.cockpit.refresh_status()?;
                let events = self.cockpit.poll_events_allowing_history_gap()?;
                let (status, _) = self.possessor_skills.step(
                    self.cockpit.client_mut(),
                    &request,
                    &tick.frame.now,
                    &status_summary,
                    status_summary.battery.home_base(),
                    &events,
                    now.t_ms,
                );
                self.runtime.observe_skill_status(&status);
                tick.skill_status = Some(status);
                self.possessor_skills.annotate_now(&mut tick.frame.now);
                lua_skill_owns_motion = true;
            }
            let final_motor = if lua_skill_owns_motion {
                self.world
                    .last_motion_sent()
                    .unwrap_or(MotionCommand::Stop)
                    .to_motor_command()
            } else {
                final_motor_from_tick(&tick)
            };
            let mut motion = motor_command_to_motion(final_motor);
            let mut motion_sent_to_sim = Some(serde_json::to_value(&motion)?);
            let reset_or_dead = is_dead_battery(&snapshot) || reset_after_tick;
            self.stuck.observe(&snapshot, tick.chosen_action.as_ref());
            let manual_reign_driving = tick
                .frame
                .reign_input
                .as_ref()
                .map(reign_input_drives_sim_directly)
                .unwrap_or(false);
            let observed_stuck_extension = self.stuck.extension(self.tick_ms);
            if is_dead_battery(&snapshot) || reset_after_tick {
                self.world.reset_body_to_spawn();
                self.stuck.reset();
                motion = MotionCommand::Stop;
                motion_sent_to_sim = None;
            } else if !lua_skill_owns_motion {
                let _ = manual_reign_driving;
                apply_safe_cockpit_motion(&mut self.cockpit, &motion)?;
            };
            let mut after_snapshot = self.world.snapshot().await?;
            annotate_snapshot_from_tick(&mut after_snapshot, &tick);
            let movement_delta = movement_delta_m(&body_pose_before, &after_snapshot.body);
            let why_not_moving = not_moving_reason(
                final_motor,
                &motion,
                &body_pose_before,
                &after_snapshot.body,
                movement_delta,
                reset_or_dead,
                &tick,
            );
            let mut action_debug = after_snapshot
                .action_debug
                .take()
                .unwrap_or_else(|| serde_json::json!({}));
            if !action_debug.is_object() {
                action_debug = serde_json::json!({});
            }
            if let Some(object) = action_debug.as_object_mut() {
                object.insert("body_pose_before".to_string(), pose_json(&body_pose_before));
                object.insert(
                    "body_pose_after".to_string(),
                    pose_json(&after_snapshot.body),
                );
                object.insert(
                    "movement_delta".to_string(),
                    serde_json::json!(movement_delta),
                );
                object.insert(
                    "motion_sent_to_sim".to_string(),
                    motion_sent_to_sim.unwrap_or(serde_json::Value::Null),
                );
                object.insert(
                    "motor_applied".to_string(),
                    serde_json::json!(movement_delta >= 0.005),
                );
                object.insert(
                    "why_not_moving".to_string(),
                    why_not_moving
                        .clone()
                        .map(serde_json::Value::String)
                        .unwrap_or(serde_json::Value::Null),
                );
            }
            after_snapshot.action_debug = Some(action_debug);
            append_actuator_outcome(
                &mut tick,
                Brain::Simulator,
                "sim.cockpit",
                after_snapshot.body.last_update_ms,
                after_snapshot
                    .action_debug
                    .clone()
                    .unwrap_or_else(|| serde_json::json!({"outcome": "not reported"})),
                if reset_or_dead {
                    EventDisposition::Rejected
                } else {
                    EventDisposition::Accepted
                },
            );
            after_snapshot
                .extensions
                .retain(|extension| extension.name != "sim.stuck");
            after_snapshot.extensions.push(observed_stuck_extension);
            self.stuck.event_started = false;
            self.stuck.recovered = false;
            observe(&after_snapshot, &tick);
            self.tick_count = self.tick_count.saturating_add(1);
        }
        Ok(())
    }
}
