#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ActionSelectorMode {
    #[default]
    Baseline,
    Random,
    ModelAssisted,
    Scripted,
    GoalShadow,
    Goal,
}

impl ActionSelectorMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Random => "random",
            Self::ModelAssisted => "model-assisted",
            Self::Scripted => "scripted",
            Self::GoalShadow => "goal-shadow",
            Self::Goal => "goal",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InlineLearningMode {
    #[default]
    Off,
    ShadowOnly,
    WorldOutcome,
}

impl InlineLearningMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::ShadowOnly => "shadow-only",
            Self::WorldOutcome => "world-outcome",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InlineLearningBehaviors {
    pub danger: bool,
    pub charge: bool,
    pub future: bool,
    pub action_value: bool,
    pub eye_next: bool,
    pub ear_next: bool,
    pub experience: bool,
}

impl Default for InlineLearningBehaviors {
    fn default() -> Self {
        Self {
            danger: true,
            charge: true,
            future: true,
            action_value: true,
            eye_next: true,
            ear_next: true,
            experience: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InlineLearningConfig {
    pub mode: InlineLearningMode,
    pub behaviors: InlineLearningBehaviors,
    pub max_train_steps_per_tick: usize,
}

impl Default for InlineLearningConfig {
    fn default() -> Self {
        Self {
            mode: InlineLearningMode::Off,
            behaviors: InlineLearningBehaviors::default(),
            max_train_steps_per_tick: 1,
        }
    }
}

impl InlineLearningConfig {
    pub fn disabled() -> Self {
        Self::default()
    }

    pub fn is_enabled(&self) -> bool {
        self.mode != InlineLearningMode::Off && self.max_train_steps_per_tick > 0
    }

    pub fn training_mode_label(&self) -> &'static str {
        match self.mode {
            InlineLearningMode::Off => "collecting",
            InlineLearningMode::ShadowOnly => "inline-shadow",
            InlineLearningMode::WorldOutcome => "inline-world-outcome",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct InlineLearningTickStatus {
    pub enabled: bool,
    pub mode: InlineLearningMode,
    pub samples_observed: usize,
    pub train_steps_used: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NudgePolicy {
    pub enabled: bool,
    pub idle_after_ms: u64,
    pub max_nudges_per_minute: u32,
    pub max_forward_intensity: f32,
    pub max_turn_intensity: f32,
    pub require_clearance_m: f32,
    pub prefer_turn_when_clearance_low: bool,
    pub cooldown_ms: u64,
}

impl Default for NudgePolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            idle_after_ms: 4_000,
            max_nudges_per_minute: 6,
            max_forward_intensity: 0.15,
            max_turn_intensity: 0.25,
            require_clearance_m: 0.35,
            prefer_turn_when_clearance_low: true,
            cooldown_ms: 5_000,
        }
    }
}

impl NudgePolicy {
    pub fn virtual_default() -> Self {
        let mut policy = Self::default();
        policy.enabled = true;
        policy.idle_after_ms = 1_200;
        policy.cooldown_ms = 2_500;
        policy
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct NudgeStatus {
    pub idle_ms: u64,
    pub last_nudge_ms: Option<u64>,
    pub nudge_count_recent: u32,
    pub nudge_blocked_reason: Option<String>,
    pub active_nudge: bool,
}

#[derive(Clone, Debug)]
struct NudgeController {
    status: NudgeStatus,
    last_pose: Option<pete_core::Pose2>,
    idle_started_at_ms: Option<u64>,
    recent_nudges: VecDeque<u64>,
    last_motor: MotorCommand,
}

impl Default for NudgeController {
    fn default() -> Self {
        Self {
            status: NudgeStatus::default(),
            last_pose: None,
            idle_started_at_ms: None,
            recent_nudges: VecDeque::new(),
            last_motor: MotorCommand::stop(),
        }
    }
}

impl NudgeController {
    fn propose(&mut self, now: &Now, policy: NudgePolicy) -> Option<ActionPrimitive> {
        self.prune_recent(now.t_ms);
        self.status.nudge_count_recent = self.recent_nudges.len() as u32;
        self.status.active_nudge = self
            .status
            .last_nudge_ms
            .map(|last| now.t_ms.saturating_sub(last) < 1_500)
            .unwrap_or(false);

        let low_motion =
            self.last_motor.forward.abs() < 0.02 && now.body.velocity.forward_m_s.abs() < 0.02;
        let low_pose_delta = self
            .last_pose
            .map(|pose| pose_delta_small(pose, now.body.odometry))
            .unwrap_or(true);
        if !low_motion || !low_pose_delta {
            self.idle_started_at_ms = Some(now.t_ms);
            self.status.idle_ms = 0;
            self.status.nudge_blocked_reason = None;
            self.last_pose = Some(now.body.odometry);
            return None;
        }

        let idle_started_at = *self.idle_started_at_ms.get_or_insert(now.t_ms);
        self.status.idle_ms = now.t_ms.saturating_sub(idle_started_at);
        self.last_pose = Some(now.body.odometry);

        if !policy.enabled {
            self.status.nudge_blocked_reason = Some("prod mode disabled".to_string());
            return None;
        }
        if let Some(last) = self.status.last_nudge_ms {
            if now.t_ms.saturating_sub(last) < policy.cooldown_ms {
                self.status.nudge_blocked_reason = Some("prod cooldown active".to_string());
                return None;
            }
        }
        if self.status.idle_ms < policy.idle_after_ms {
            self.status.nudge_blocked_reason = Some("not idle long enough".to_string());
            return None;
        }
        if self.recent_nudges.len() as u32 >= policy.max_nudges_per_minute {
            self.status.nudge_blocked_reason = Some("prod rate limit active".to_string());
            return None;
        }
        if let Some(reason) = nudge_general_block_reason(now) {
            self.status.nudge_blocked_reason = Some(reason);
            return None;
        }

        let action = choose_nudge_action(now, policy, self.recent_nudges.len());
        if let Some(reason) = nudge_action_block_reason(now, &action, policy) {
            self.status.nudge_blocked_reason = Some(reason);
            return None;
        }
        self.record_nudge(now.t_ms);
        self.status.nudge_blocked_reason = None;
        Some(action)
    }

    fn observe_motor(&mut self, motor: MotorCommand) {
        self.last_motor = motor;
    }

    fn record_nudge(&mut self, t_ms: u64) {
        self.status.last_nudge_ms = Some(t_ms);
        self.status.active_nudge = true;
        self.recent_nudges.push_back(t_ms);
        self.prune_recent(t_ms);
        self.status.nudge_count_recent = self.recent_nudges.len() as u32;
        self.idle_started_at_ms = Some(t_ms);
        self.status.idle_ms = 0;
    }

    fn prune_recent(&mut self, t_ms: u64) {
        while self
            .recent_nudges
            .front()
            .map(|stamp| t_ms.saturating_sub(*stamp) > 60_000)
            .unwrap_or(false)
        {
            self.recent_nudges.pop_front();
        }
    }
}

pub fn nudge_action_block_reason(
    now: &Now,
    action: &ActionPrimitive,
    policy: NudgePolicy,
) -> Option<String> {
    if let Some(reason) = nudge_general_block_reason(now) {
        return Some(reason);
    }
    let motor = action_to_motor_command(Some(action));
    if motor.forward > 0.0 && !forward_clear(now, policy.require_clearance_m) {
        return Some(format!(
            "forward path clearance is below {:.2} m",
            policy.require_clearance_m
        ));
    }
    None
}

pub fn nudge_action_block_reason_for_snapshot(
    snapshot: &WorldSnapshot,
    action: &ActionPrimitive,
    policy: NudgePolicy,
) -> Option<String> {
    nudge_action_block_reason(
        &snapshot.to_now(snapshot.body.last_update_ms),
        action,
        policy,
    )
}

fn choose_nudge_action(now: &Now, policy: NudgePolicy, recent_count: usize) -> ActionPrimitive {
    let turn_intensity = 0.20_f32.min(policy.max_turn_intensity);
    if !forward_clear(now, policy.require_clearance_m) && policy.prefer_turn_when_clearance_low {
        return ActionPrimitive::Turn {
            direction: clearer_turn_direction(now),
            intensity: turn_intensity,
            duration_ms: 600,
        };
    }

    match recent_count % 3 {
        0 => ActionPrimitive::Turn {
            direction: clearer_turn_direction(now),
            intensity: turn_intensity,
            duration_ms: 600,
        },
        1 => ActionPrimitive::Explore {
            style: ExploreStyle::RandomWalk,
            duration_ms: 1_000,
        },
        _ => ActionPrimitive::Go {
            intensity: 0.12_f32.min(policy.max_forward_intensity),
            duration_ms: 500,
        },
    }
}

fn nudge_general_block_reason(now: &Now) -> Option<String> {
    if now.body.flags.wheel_drop {
        return Some("wheel drop detected".to_string());
    }
    if now.body.battery_level <= SafetyConfigForNudge::CRITICAL_BATTERY {
        return Some("battery is critical".to_string());
    }
    if now
        .extensions
        .get("safety.vetoed")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
    {
        return Some("active safety override".to_string());
    }
    if sim_stuck_active(now) {
        return Some("stuck recovery active".to_string());
    }
    None
}

struct SafetyConfigForNudge;

impl SafetyConfigForNudge {
    const CRITICAL_BATTERY: f32 = 0.10;
}

fn is_near_zero_motor(motor: MotorCommand) -> bool {
    motor.forward.abs() < 0.02 && motor.turn.abs() < 0.04
}

fn pose_delta_small(left: pete_core::Pose2, right: pete_core::Pose2) -> bool {
    let dx = left.x_m - right.x_m;
    let dy = left.y_m - right.y_m;
    let distance = (dx * dx + dy * dy).sqrt();
    distance < 0.025
}

fn forward_clear(now: &Now, clearance_m: f32) -> bool {
    now.range
        .nearest_m
        .map(|nearest| nearest >= clearance_m)
        .unwrap_or(true)
}

fn clearer_turn_direction(now: &Now) -> TurnDir {
    let (left, _center, right) = beam_clearance_buckets(&now.range.beams);
    if right > left {
        TurnDir::Right
    } else {
        TurnDir::Left
    }
}

fn sim_stuck_active(now: &Now) -> bool {
    now.extensions
        .get("sim.stuck")
        .and_then(|value| value.get("values"))
        .and_then(|value| value.as_array())
        .and_then(|values| values.first())
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0)
        > 0.0
}

fn sim_world_extension_score(now: &Now, index: usize) -> f32 {
    now.extensions
        .get("sim.world")
        .and_then(|value| value.get("values"))
        .and_then(|value| value.as_array())
        .and_then(|values| values.get(index))
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0) as f32
}

fn apply_create_ir_charger_cue(now: &mut Now) {
    let Some(cue) = DockIrCue::from_character(now.body.infrared_character) else {
        return;
    };
    now.objects.observations.push(ObjectObservation {
        label: "home base IR".to_string(),
        class: ObjectClass::Charger,
        bearing_rad: cue.bearing_hint_rad(),
        // The beacon proves direction and identity, not metric range. In
        // particular, force-field reception must never be mistaken for dock
        // contact or charging.
        distance_m: None,
        confidence: cue.visible_score(),
        source: ObjectObservationSource::CreateIr,
    });
}

fn charger_signal_scores(now: &Now) -> (f32, f32) {
    let mut near = sim_world_extension_score(now, 3);
    let mut visible = sim_world_extension_score(now, 4);
    if let Some(cue) = DockIrCue::from_character(now.body.infrared_character) {
        near = near.max(cue.near_score());
        visible = visible.max(cue.visible_score());
    }
    (near, visible)
}

fn apply_recent_trap_memory_hints(now: &mut Now) {
    let Some(values) = now
        .extensions
        .get("sim.stuck")
        .and_then(|value| value.get("values"))
        .and_then(|value| value.as_array())
    else {
        return;
    };
    let active = values
        .first()
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0)
        > 0.0;
    let event_started = values
        .get(6)
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0)
        > 0.0;
    let repeated = values
        .get(12)
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0)
        .max(0.0) as f32;
    let trap_kind = values
        .get(10)
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0);
    if !(active || event_started || repeated > 0.0 || trap_kind > 0.0) {
        return;
    }
    let turn_sign = values
        .get(5)
        .and_then(|value| value.as_f64())
        .unwrap_or(0.0) as f32;
    now.memory.recent_trap_confidence = (0.6 + repeated.min(2.0) * 0.15).clamp(0.0, 1.0);
    now.memory.recent_trap_direction_rad = Some(if turn_sign < 0.0 {
        -std::f32::consts::FRAC_PI_2
    } else if turn_sign > 0.0 {
        std::f32::consts::FRAC_PI_2
    } else {
        0.0
    });
}

fn place_candidate_to_loop_input(
    candidate: &PlaceRecognitionCandidate,
    source_frame_id: Option<String>,
    query_input: Option<&pete_memory::PlaceRecognitionInput>,
) -> LoopClosureCandidateInput {
    LoopClosureCandidateInput {
        target_pose: Pose2 {
            x_m: candidate.cell.center_x_m,
            y_m: candidate.cell.center_y_m,
            heading_rad: query_input
                .and_then(|input| input.pose)
                .map(|pose| pose.heading_rad)
                .unwrap_or(0.0),
        },
        confidence: candidate.confidence,
        similarity: candidate.similarity,
        kind: match candidate.kind {
            PlaceRecognitionKind::SamePlace => "same_place",
            PlaceRecognitionKind::SimilarPlace => "similar_place",
            PlaceRecognitionKind::EntityConstellation => "entity_constellation",
        }
        .to_string(),
        target_frame_id: candidate
            .source_instant_frame_id
            .clone()
            .or_else(|| candidate.source_frame_id.clone()),
        source_frame_id,
        source_experience_id: candidate.source_experience_id.clone(),
        source_instant_frame_id: candidate.source_instant_frame_id.clone(),
        source_vector_refs: candidate.source_vector_refs.clone(),
        source_vector_id: Some(candidate.source_vector_id.clone()),
        query_vector_id: candidate.query_vector_id.clone(),
        query_experience_id: candidate
            .query_experience_id
            .clone()
            .or_else(|| query_input.and_then(|input| input.experience_id.clone())),
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ActionSelectionDecision {
    pub mode: ActionSelectorMode,
    pub candidates: Vec<ActionSelectionCandidateScore>,
    pub selected_action: Option<ActionPrimitive>,
    pub baseline_action: Option<ActionPrimitive>,
    pub selected_score: Option<f32>,
    pub safety_overrode: bool,
    pub fallback_warnings: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_goal: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_behavior: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_selected_goal: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_selected_behavior: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shadow_goal_action: Option<ActionPrimitive>,
    #[serde(default)]
    pub shadow_diverged_from_baseline: bool,
    #[serde(default)]
    pub goal_switched: bool,
    #[serde(default)]
    pub goal_retained_by_commitment: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goal_selection_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionSelectionCandidateScore {
    pub action: ActionPrimitive,
    pub score: f32,
    pub danger: f32,
    pub charge: f32,
    pub action_value: f32,
    pub curiosity: f32,
    pub collision_risk: f32,
    pub low_battery_risk: f32,
    pub repeat_penalty: f32,
    pub fallback_used: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MapMemoryDecisionDebug {
    pub influenced: bool,
    #[serde(default)]
    pub corrected_map_trusted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corrected_map_untrusted_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub navigation_intent: Option<NavigationIntent>,
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_string: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal_value: Option<f32>,
    #[serde(default)]
    pub signal_confidence: f32,
    #[serde(default)]
    pub confidence: f32,
    pub place_danger: f32,
    pub place_charge_value: f32,
    pub place_novelty: f32,
    pub safe_direction_rad: Option<f32>,
    pub charge_direction_rad: Option<f32>,
    pub frontier_direction_rad: Option<f32>,
    pub recent_trap_direction_rad: Option<f32>,
    pub map_confidence: f32,
    pub recent_trap_confidence: f32,
    pub selected_action: Option<ActionPrimitive>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen_action: Option<ActionPrimitive>,
    #[serde(default)]
    pub safety_overrode: bool,
}

impl Default for ActionSelectionCandidateScore {
    fn default() -> Self {
        Self {
            action: ActionPrimitive::Stop,
            score: 0.0,
            danger: 0.0,
            charge: 0.0,
            action_value: 0.0,
            curiosity: 0.0,
            collision_risk: 0.0,
            low_battery_risk: 0.0,
            repeat_penalty: 0.0,
            fallback_used: false,
        }
    }
}

