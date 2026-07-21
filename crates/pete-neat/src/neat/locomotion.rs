pub const LOCOMOTION_SCHEMA_VERSION: u32 = 1;
pub const LOCOMOTION_INPUT_COUNT: usize = 17;
pub const LOCOMOTION_OUTPUT_COUNT: usize = 3;
pub const LOCOMOTION_INPUT_NAMES: [&str; LOCOMOTION_INPUT_COUNT] = [
    "bump_left",
    "bump_right",
    "bump_front",
    "left_wheel_travel",
    "right_wheel_travel",
    "forward_velocity",
    "angular_velocity",
    "distance_since_collision",
    "time_since_collision",
    "recent_turn_direction",
    "clearance_left",
    "clearance_front",
    "clearance_right",
    "last_forward_command",
    "last_turn_command",
    "battery_level",
    "recent_collision_rate",
];
pub const LOCOMOTION_OUTPUT_NAMES: [&str; LOCOMOTION_OUTPUT_COUNT] = [
    "forward_velocity",
    "angular_velocity",
    "recovery_activation",
];

const CREATE_WHEEL_BASE_M: f32 = 0.235;
const MAX_RANGE_M: f32 = 4.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocomotionInput {
    pub schema_version: u32,
    pub bump_left: f32,
    pub bump_right: f32,
    pub bump_front: f32,
    pub left_wheel_travel_m: f32,
    pub right_wheel_travel_m: f32,
    pub forward_velocity_m_s: f32,
    pub angular_velocity_rad_s: f32,
    pub distance_since_collision_m: f32,
    pub time_since_collision_s: f32,
    /// Negative is a recent right turn; positive is a recent left turn.
    pub recent_turn_direction: f32,
    pub clearance_left_m: f32,
    pub clearance_front_m: f32,
    pub clearance_right_m: f32,
    pub last_forward_command_m_s: f32,
    pub last_turn_command_rad_s: f32,
    pub battery_level: f32,
    pub recent_collision_rate: f32,
}

impl Default for LocomotionInput {
    fn default() -> Self {
        Self {
            schema_version: LOCOMOTION_SCHEMA_VERSION,
            bump_left: 0.0,
            bump_right: 0.0,
            bump_front: 0.0,
            left_wheel_travel_m: 0.0,
            right_wheel_travel_m: 0.0,
            forward_velocity_m_s: 0.0,
            angular_velocity_rad_s: 0.0,
            distance_since_collision_m: 0.0,
            time_since_collision_s: 0.0,
            recent_turn_direction: 0.0,
            clearance_left_m: MAX_RANGE_M,
            clearance_front_m: MAX_RANGE_M,
            clearance_right_m: MAX_RANGE_M,
            last_forward_command_m_s: 0.0,
            last_turn_command_rad_s: 0.0,
            battery_level: 1.0,
            recent_collision_rate: 0.0,
        }
    }
}

impl LocomotionInput {
    /// Stable, normalized network order. Changing it requires a schema bump.
    pub fn features(&self) -> [f32; LOCOMOTION_INPUT_COUNT] {
        [
            unit(self.bump_left),
            unit(self.bump_right),
            unit(self.bump_front),
            (self.left_wheel_travel_m / 5.0).tanh(),
            (self.right_wheel_travel_m / 5.0).tanh(),
            (self.forward_velocity_m_s / 0.6).clamp(-1.0, 1.0),
            (self.angular_velocity_rad_s / 1.5).clamp(-1.0, 1.0),
            (self.distance_since_collision_m / 4.0).clamp(0.0, 1.0),
            (self.time_since_collision_s / 20.0).clamp(0.0, 1.0),
            self.recent_turn_direction.clamp(-1.0, 1.0),
            normalize_clearance(self.clearance_left_m),
            normalize_clearance(self.clearance_front_m),
            normalize_clearance(self.clearance_right_m),
            (self.last_forward_command_m_s / 0.6).clamp(-1.0, 1.0),
            (self.last_turn_command_rad_s / 1.5).clamp(-1.0, 1.0),
            unit(self.battery_level),
            unit(self.recent_collision_rate),
        ]
    }

    pub fn collision_active(&self) -> bool {
        self.bump_left > 0.5 || self.bump_right > 0.5 || self.bump_front > 0.5
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LocomotionOutput {
    pub forward_velocity_m_s: f32,
    pub angular_velocity_rad_s: f32,
    pub recovery_activation: f32,
}

impl LocomotionOutput {
    pub fn bounded(self, max_forward_m_s: f32, max_turn_rad_s: f32) -> Self {
        Self {
            forward_velocity_m_s: finite_or_zero(self.forward_velocity_m_s)
                .clamp(-max_forward_m_s.abs(), max_forward_m_s.abs()),
            angular_velocity_rad_s: finite_or_zero(self.angular_velocity_rad_s)
                .clamp(-max_turn_rad_s.abs(), max_turn_rad_s.abs()),
            recovery_activation: finite_or_zero(self.recovery_activation).clamp(0.0, 1.0),
        }
    }

    /// Recovery is an intent, not a safety-latch override. It only makes the
    /// proposed linear component non-positive; downstream gates still decide.
    pub fn with_recovery_intent(mut self, threshold: f32) -> Self {
        if self.recovery_activation >= threshold {
            self.forward_velocity_m_s = -self.forward_velocity_m_s.abs();
        }
        self
    }
}

impl OutputDistance for LocomotionOutput {
    fn distance(&self, other: &Self) -> f32 {
        let df = self.forward_velocity_m_s - other.forward_velocity_m_s;
        let dt = self.angular_velocity_rad_s - other.angular_velocity_rad_s;
        let dr = self.recovery_activation - other.recovery_activation;
        (df.mul_add(df, dt.mul_add(dt, dr * dr))).sqrt()
    }
}

#[derive(Clone, Debug)]
pub struct HardcodedLocomotionBehavior {
    pub forward_velocity_m_s: f32,
    pub angular_velocity_rad_s: f32,
}

impl Default for HardcodedLocomotionBehavior {
    fn default() -> Self {
        Self {
            forward_velocity_m_s: 0.2,
            angular_velocity_rad_s: 0.1,
        }
    }
}

impl FunctionBehavior<LocomotionInput, LocomotionOutput> for HardcodedLocomotionBehavior {
    fn id(&self) -> &'static str {
        "locomotion.hardcoded_wander.v0"
    }

    fn infer(&mut self, _input: &LocomotionInput) -> Result<LocomotionOutput> {
        // Preserve the exact ancestral Explore motor mapping. Bump/cliff
        // recovery remains in the existing simulator/possession reflex paths.
        Ok(LocomotionOutput {
            forward_velocity_m_s: self.forward_velocity_m_s,
            angular_velocity_rad_s: self.angular_velocity_rad_s,
            recovery_activation: 0.0,
        })
    }
}

/// Stateful conversion from body/range snapshots to the small nervous system.
#[derive(Clone, Debug, Default)]
pub struct LocomotionTracker {
    started_at_ms: Option<u64>,
    last_t_ms: Option<u64>,
    last_pose: Option<Pose2>,
    cumulative_distance_m: f32,
    cumulative_heading_rad: f32,
    distance_since_collision_m: f32,
    last_collision_ms: Option<u64>,
    collision_times_ms: Vec<u64>,
    collision_active: bool,
    recent_turn_direction: f32,
    last_forward_command_m_s: f32,
    last_turn_command_rad_s: f32,
}

impl LocomotionTracker {
    pub fn observe(&mut self, t_ms: u64, body: &BodySense, range: &RangeSense) -> LocomotionInput {
        let started_at_ms = *self.started_at_ms.get_or_insert(t_ms);
        let dt_s = self
            .last_t_ms
            .map(|last| t_ms.saturating_sub(last) as f32 / 1_000.0)
            .filter(|dt| *dt > 0.0)
            .unwrap_or(0.0);

        let (distance_delta, heading_delta) = self
            .last_pose
            .map(|last| pose_delta(last, body.odometry))
            .unwrap_or((0.0, 0.0));
        self.cumulative_distance_m += distance_delta;
        self.cumulative_heading_rad += heading_delta;
        self.distance_since_collision_m += distance_delta.abs();

        let collision = body.flags.bump_left || body.flags.bump_right;
        if collision && !self.collision_active {
            self.last_collision_ms = Some(t_ms);
            self.distance_since_collision_m = 0.0;
            self.collision_times_ms.push(t_ms);
        }
        self.collision_active = collision;
        self.collision_times_ms
            .retain(|stamp| t_ms.saturating_sub(*stamp) <= 10_000);

        let derived_forward = if dt_s > 0.0 {
            distance_delta / dt_s
        } else {
            0.0
        };
        let derived_turn = if dt_s > 0.0 {
            heading_delta / dt_s
        } else {
            0.0
        };
        let forward_velocity = prefer_measured(body.velocity.forward_m_s, derived_forward);
        let angular_velocity = prefer_measured(body.velocity.turn_rad_s, derived_turn);
        self.recent_turn_direction =
            self.recent_turn_direction * 0.8 + (angular_velocity / 1.5).clamp(-1.0, 1.0) * 0.2;

        let (clearance_left_m, clearance_front_m, clearance_right_m) = range_sectors(range);
        let left_wheel_travel_m =
            self.cumulative_distance_m - self.cumulative_heading_rad * CREATE_WHEEL_BASE_M * 0.5;
        let right_wheel_travel_m =
            self.cumulative_distance_m + self.cumulative_heading_rad * CREATE_WHEEL_BASE_M * 0.5;

        let input = LocomotionInput {
            schema_version: LOCOMOTION_SCHEMA_VERSION,
            bump_left: bool_unit(body.flags.bump_left),
            bump_right: bool_unit(body.flags.bump_right),
            bump_front: bool_unit(body.flags.bump_left && body.flags.bump_right),
            left_wheel_travel_m,
            right_wheel_travel_m,
            forward_velocity_m_s: forward_velocity,
            angular_velocity_rad_s: angular_velocity,
            distance_since_collision_m: self.distance_since_collision_m,
            time_since_collision_s: self
                .last_collision_ms
                .map(|stamp| t_ms.saturating_sub(stamp) as f32 / 1_000.0)
                .unwrap_or_else(|| t_ms.saturating_sub(started_at_ms) as f32 / 1_000.0),
            recent_turn_direction: self.recent_turn_direction,
            clearance_left_m,
            clearance_front_m,
            clearance_right_m,
            last_forward_command_m_s: self.last_forward_command_m_s,
            last_turn_command_rad_s: self.last_turn_command_rad_s,
            battery_level: body.battery_level,
            recent_collision_rate: self.collision_times_ms.len() as f32 / 10.0,
        };

        self.last_t_ms = Some(t_ms);
        self.last_pose = Some(body.odometry);
        input
    }

    pub fn observe_command(&mut self, output: LocomotionOutput) {
        self.last_forward_command_m_s = output.forward_velocity_m_s;
        self.last_turn_command_rad_s = output.angular_velocity_rad_s;
    }
}
