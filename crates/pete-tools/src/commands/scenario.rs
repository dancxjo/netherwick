#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ScenarioEvaluationReport {
    schema_version: u32,
    scenario: String,
    base_seed: u64,
    episodes: usize,
    steps_per_episode: usize,
    tick_ms: u64,
    action_selector_mode: String,
    model_modes: HashMap<String, String>,
    model_loading: RuntimeModelLoadReport,
    ledger: Option<String>,
    capture_root: Option<String>,
    summary: ScenarioEvaluationSummary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    memory: Option<ScenarioMemorySummary>,
    episodes_detail: Vec<ScenarioEpisodeReport>,
    recommendation: String,
    warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct ScenarioEvaluationSummary {
    success_rate: f32,
    collision_rate: f32,
    mean_collisions_per_episode: f32,
    mean_battery_delta: f32,
    mean_final_battery: f32,
    mean_distance_to_charger_final_m: Option<f32>,
    #[serde(default)]
    ticks_with_charger_visible: usize,
    #[serde(default)]
    ticks_with_charger_near: usize,
    #[serde(default)]
    ticks_approaching_charger: usize,
    #[serde(default)]
    ticks_docking_from_too_far: usize,
    mean_nearest_obstacle_m: Option<f32>,
    mean_distance_traveled_m: f32,
    #[serde(default)]
    action_histogram: HashMap<String, usize>,
    #[serde(default)]
    wall_cliff_veto_count: usize,
    #[serde(default)]
    escape_progress_score: f32,
    mean_ticks_survived: f32,
    #[serde(default)]
    stuck_count: usize,
    #[serde(default)]
    trap_kind_counts: HashMap<String, usize>,
    #[serde(default)]
    recovery_attempts: usize,
    #[serde(default)]
    stuck_duration: Option<f32>,
    #[serde(default)]
    mean_stuck_duration: Option<f32>,
    #[serde(default)]
    recovery_success_rate: Option<f32>,
    #[serde(default)]
    mean_recovery_ticks: Option<f32>,
    #[serde(default)]
    repeated_trap_count: usize,
    #[serde(default)]
    dead_battery_tick: Option<usize>,
    #[serde(default)]
    distance_after_recovery_m: Option<f32>,
    mean_safety_interventions: f32,
    behavior_run_records: usize,
    #[serde(default)]
    model_fallbacks: usize,
    #[serde(default)]
    action_selector_fallbacks: usize,
    #[serde(default)]
    action_selector_guard_yields: usize,
    #[serde(default)]
    map_memory_decisions: usize,
    #[serde(default)]
    danger_memory_decisions: usize,
    #[serde(default)]
    charge_memory_decisions: usize,
    #[serde(default)]
    novelty_memory_decisions: usize,
    #[serde(default)]
    frontier_memory_decisions: usize,
    #[serde(default)]
    trap_memory_decisions: usize,
    #[serde(default)]
    memory_navigation_intents: HashMap<String, usize>,
    #[serde(default)]
    memory_navigation_reasons: HashMap<String, usize>,
    #[serde(default)]
    map_memory_signals: HashMap<String, usize>,
    #[serde(default)]
    map_memory_safety_overrides: usize,
    #[serde(default)]
    low_confidence_navigation_fallbacks: usize,
    model_assisted_decisions: usize,
    action_selector_safety_overrides: usize,
    #[serde(default)]
    goal_switches: usize,
    #[serde(default)]
    goal_commitment_retained_ticks: usize,
    #[serde(default)]
    goal_behavior_transitions: usize,
    #[serde(default)]
    goal_shadow_divergences: usize,
    #[serde(default)]
    mean_goal_dwell_ms: Option<f32>,
    #[serde(default)]
    goal_histogram: HashMap<String, usize>,
    #[serde(default)]
    goal_behavior_histogram: HashMap<String, usize>,
    #[serde(default)]
    goal_progress_samples: usize,
    #[serde(default)]
    mean_goal_progress: Option<f32>,
    #[serde(default)]
    goal_no_progress_dwell_ticks: usize,
    #[serde(default)]
    goal_failed_attempts: usize,
    #[serde(default)]
    strategy_switches_within_goal: usize,
    #[serde(default)]
    goal_help_requests: usize,
    #[serde(default)]
    unmeasurable_progress_ticks: usize,
    #[serde(default)]
    false_stall_rate: Option<f32>,
    mean_chosen_score: Option<f32>,
    mean_candidate_score: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct ScenarioEpisodeReport {
    index: usize,
    seed: u64,
    success: bool,
    ticks: usize,
    collisions: usize,
    wall_hits: usize,
    bumper_hits: usize,
    cliff_hits: usize,
    charging_ticks: usize,
    first_charge_tick: Option<usize>,
    started_battery: f32,
    final_battery: f32,
    battery_delta: f32,
    min_nearest_obstacle_m: Option<f32>,
    mean_nearest_obstacle_m: Option<f32>,
    final_distance_to_charger_m: Option<f32>,
    #[serde(default)]
    final_heading_rad: Option<f32>,
    #[serde(default)]
    final_bearing_to_charger_rad: Option<f32>,
    final_distance_to_person_m: Option<f32>,
    final_distance_to_speaker_m: Option<f32>,
    distance_traveled_m: f32,
    #[serde(default)]
    ticks_with_charger_visible: usize,
    #[serde(default)]
    ticks_with_charger_near: usize,
    #[serde(default)]
    ticks_approaching_charger: usize,
    #[serde(default)]
    ticks_docking_from_too_far: usize,
    #[serde(default)]
    action_histogram: HashMap<String, usize>,
    #[serde(default)]
    wall_cliff_veto_count: usize,
    #[serde(default)]
    escape_progress_score: f32,
    #[serde(default)]
    stuck_ticks: usize,
    #[serde(default)]
    stuck_count: usize,
    #[serde(default)]
    trap_kind_counts: HashMap<String, usize>,
    #[serde(default)]
    recovery_attempts: usize,
    #[serde(default)]
    stuck_duration: Option<f32>,
    #[serde(default)]
    mean_stuck_duration: Option<f32>,
    #[serde(default)]
    recovery_success_rate: Option<f32>,
    #[serde(default)]
    mean_recovery_ticks: Option<f32>,
    #[serde(default)]
    repeated_trap_count: usize,
    #[serde(default)]
    dead_battery_tick: Option<usize>,
    #[serde(default)]
    distance_after_recovery_m: Option<f32>,
    unique_actions: Vec<String>,
    safety_interventions: usize,
    behavior_run_records: usize,
    model_fallbacks: usize,
    model_assisted_decisions: usize,
    action_selector_safety_overrides: usize,
    #[serde(default)]
    goal_switches: usize,
    #[serde(default)]
    goal_commitment_retained_ticks: usize,
    #[serde(default)]
    goal_behavior_transitions: usize,
    #[serde(default)]
    goal_shadow_divergences: usize,
    #[serde(default)]
    mean_goal_dwell_ms: Option<f32>,
    #[serde(default)]
    goal_histogram: HashMap<String, usize>,
    #[serde(default)]
    goal_behavior_histogram: HashMap<String, usize>,
    #[serde(default)]
    goal_progress_samples: usize,
    #[serde(default)]
    mean_goal_progress: Option<f32>,
    #[serde(default)]
    goal_no_progress_dwell_ticks: usize,
    #[serde(default)]
    goal_failed_attempts: usize,
    #[serde(default)]
    strategy_switches_within_goal: usize,
    #[serde(default)]
    goal_help_requests: usize,
    #[serde(default)]
    unmeasurable_progress_ticks: usize,
    #[serde(default)]
    stall_responses: usize,
    #[serde(default)]
    false_stall_count: usize,
    #[serde(default)]
    false_stall_rate: Option<f32>,
    action_selector_fallbacks: usize,
    #[serde(default)]
    action_selector_guard_yields: usize,
    #[serde(default)]
    map_memory_decisions: usize,
    #[serde(default)]
    danger_memory_decisions: usize,
    #[serde(default)]
    charge_memory_decisions: usize,
    #[serde(default)]
    novelty_memory_decisions: usize,
    #[serde(default)]
    frontier_memory_decisions: usize,
    #[serde(default)]
    trap_memory_decisions: usize,
    #[serde(default)]
    memory_navigation_intents: HashMap<String, usize>,
    #[serde(default)]
    memory_navigation_reasons: HashMap<String, usize>,
    #[serde(default)]
    map_memory_signals: HashMap<String, usize>,
    #[serde(default)]
    map_memory_safety_overrides: usize,
    #[serde(default)]
    map_memory_decision_samples: Vec<ScenarioMapMemoryDecisionReport>,
    #[serde(default)]
    low_confidence_navigation_fallbacks: usize,
    mean_chosen_score: Option<f32>,
    mean_candidate_score: Option<f32>,
    ticks_with_eye_frames: usize,
    ticks_with_ear_features: usize,
    ticks_with_voice_embeddings: usize,
    ticks_with_face_embeddings: usize,
    ticks_with_kinect_skeletons: usize,
    ticks_with_future_predictions: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    memory: Option<ScenarioEpisodeMemoryReport>,
    capture: Option<String>,
    ledger: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct ScenarioMemorySummary {
    places_visited: usize,
    mean_places_visited_per_episode: f32,
    charge_memory_hit_rate: Option<f32>,
    danger_memory_hit_rate: Option<f32>,
    social_memory_hit_rate: Option<f32>,
    novelty_decay_sane: bool,
    warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct ScenarioMapMemoryDecisionReport {
    signal: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    signal_value: Option<f32>,
    signal_confidence: f32,
    chosen_action: Option<ActionPrimitive>,
    reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reason_string: Option<String>,
    safety_overrode: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct ScenarioEpisodeMemoryReport {
    places_visited: usize,
    charge_memory_ticks: usize,
    charge_opportunity_ticks: usize,
    charge_memory_hit_rate: Option<f32>,
    danger_memory_ticks: usize,
    danger_opportunity_ticks: usize,
    danger_memory_hit_rate: Option<f32>,
    social_memory_ticks: usize,
    social_opportunity_ticks: usize,
    social_memory_hit_rate: Option<f32>,
    first_novelty: Option<f32>,
    final_novelty: Option<f32>,
    novelty_decayed: bool,
}

#[derive(Clone, Debug)]
struct EpisodeMetricBuilder {
    kind: ScenarioKind,
    metadata: pete_sim::ScenarioMetadata,
    index: usize,
    seed: u64,
    ledger: Option<String>,
    capture: Option<String>,
    ticks: usize,
    collisions: usize,
    wall_hits: usize,
    bumper_hits: usize,
    cliff_hits: usize,
    charging_ticks: usize,
    first_charge_tick: Option<usize>,
    started_battery: Option<f32>,
    final_battery: f32,
    min_nearest_obstacle_m: Option<f32>,
    nearest_obstacle_sum: f32,
    nearest_obstacle_count: usize,
    start_position: Option<(f32, f32)>,
    last_position: Option<(f32, f32)>,
    last_heading_rad: Option<f32>,
    distance_traveled_m: f32,
    ticks_with_charger_visible: usize,
    ticks_with_charger_near: usize,
    ticks_approaching_charger: usize,
    ticks_docking_from_too_far: usize,
    stuck_ticks: usize,
    stuck_count: usize,
    trap_kind_counts: HashMap<String, usize>,
    recovery_attempts: usize,
    stuck_duration_sum_ms: f32,
    stuck_duration_count: usize,
    active_stuck_duration_ms: Option<f32>,
    recovery_successes: usize,
    recovery_ticks_sum: usize,
    recovery_tick_count: usize,
    repeated_trap_count: usize,
    distance_at_last_recovery_m: Option<f32>,
    dead_battery_tick: Option<usize>,
    unique_actions: BTreeSet<String>,
    action_histogram: HashMap<String, usize>,
    wall_cliff_veto_count: usize,
    safety_interventions: usize,
    behavior_run_records: usize,
    model_fallbacks: usize,
    model_assisted_decisions: usize,
    action_selector_safety_overrides: usize,
    action_selector_fallbacks: usize,
    action_selector_guard_yields: usize,
    goal_switches: usize,
    goal_commitment_retained_ticks: usize,
    goal_behavior_transitions: usize,
    goal_shadow_divergences: usize,
    goal_dwell_ticks_sum: usize,
    goal_dwell_count: usize,
    current_goal: Option<String>,
    current_goal_ticks: usize,
    current_goal_behavior: Option<String>,
    goal_histogram: HashMap<String, usize>,
    goal_behavior_histogram: HashMap<String, usize>,
    goal_progress_sum: f32,
    goal_progress_samples: usize,
    goal_no_progress_dwell_ticks: usize,
    goal_failed_attempts: HashMap<String, u32>,
    strategy_switches_within_goal: usize,
    goal_help_requests: usize,
    unmeasurable_progress_ticks: usize,
    stall_responses: usize,
    false_stall_count: usize,
    map_memory_decisions: usize,
    danger_memory_decisions: usize,
    charge_memory_decisions: usize,
    novelty_memory_decisions: usize,
    frontier_memory_decisions: usize,
    trap_memory_decisions: usize,
    memory_navigation_intents: HashMap<String, usize>,
    memory_navigation_reasons: HashMap<String, usize>,
    map_memory_signals: HashMap<String, usize>,
    map_memory_safety_overrides: usize,
    map_memory_decision_samples: Vec<ScenarioMapMemoryDecisionReport>,
    low_confidence_navigation_fallbacks: usize,
    chosen_score_sum: f32,
    chosen_score_count: usize,
    candidate_score_sum: f32,
    candidate_score_count: usize,
    ticks_with_eye_frames: usize,
    ticks_with_ear_features: usize,
    ticks_with_voice_embeddings: usize,
    ticks_with_face_embeddings: usize,
    ticks_with_kinect_skeletons: usize,
    ticks_with_future_predictions: usize,
    memory: ScenarioEpisodeMemoryBuilder,
}

impl EpisodeMetricBuilder {
    fn new(
        kind: ScenarioKind,
        metadata: pete_sim::ScenarioMetadata,
        index: usize,
        seed: u64,
        ledger: Option<String>,
        capture: Option<String>,
    ) -> Self {
        Self {
            kind,
            metadata,
            index,
            seed,
            ledger,
            capture,
            ticks: 0,
            collisions: 0,
            wall_hits: 0,
            bumper_hits: 0,
            cliff_hits: 0,
            charging_ticks: 0,
            first_charge_tick: None,
            started_battery: None,
            final_battery: 0.0,
            min_nearest_obstacle_m: None,
            nearest_obstacle_sum: 0.0,
            nearest_obstacle_count: 0,
            start_position: None,
            last_position: None,
            last_heading_rad: None,
            distance_traveled_m: 0.0,
            ticks_with_charger_visible: 0,
            ticks_with_charger_near: 0,
            ticks_approaching_charger: 0,
            ticks_docking_from_too_far: 0,
            stuck_ticks: 0,
            stuck_count: 0,
            trap_kind_counts: HashMap::new(),
            recovery_attempts: 0,
            stuck_duration_sum_ms: 0.0,
            stuck_duration_count: 0,
            active_stuck_duration_ms: None,
            recovery_successes: 0,
            recovery_ticks_sum: 0,
            recovery_tick_count: 0,
            repeated_trap_count: 0,
            distance_at_last_recovery_m: None,
            dead_battery_tick: None,
            unique_actions: BTreeSet::new(),
            action_histogram: HashMap::new(),
            wall_cliff_veto_count: 0,
            safety_interventions: 0,
            behavior_run_records: 0,
            model_fallbacks: 0,
            model_assisted_decisions: 0,
            action_selector_safety_overrides: 0,
            action_selector_fallbacks: 0,
            action_selector_guard_yields: 0,
            goal_switches: 0,
            goal_commitment_retained_ticks: 0,
            goal_behavior_transitions: 0,
            goal_shadow_divergences: 0,
            goal_dwell_ticks_sum: 0,
            goal_dwell_count: 0,
            current_goal: None,
            current_goal_ticks: 0,
            current_goal_behavior: None,
            goal_histogram: HashMap::new(),
            goal_behavior_histogram: HashMap::new(),
            goal_progress_sum: 0.0,
            goal_progress_samples: 0,
            goal_no_progress_dwell_ticks: 0,
            goal_failed_attempts: HashMap::new(),
            strategy_switches_within_goal: 0,
            goal_help_requests: 0,
            unmeasurable_progress_ticks: 0,
            stall_responses: 0,
            false_stall_count: 0,
            map_memory_decisions: 0,
            danger_memory_decisions: 0,
            charge_memory_decisions: 0,
            novelty_memory_decisions: 0,
            frontier_memory_decisions: 0,
            trap_memory_decisions: 0,
            memory_navigation_intents: HashMap::new(),
            memory_navigation_reasons: HashMap::new(),
            map_memory_signals: HashMap::new(),
            map_memory_safety_overrides: 0,
            map_memory_decision_samples: Vec::new(),
            low_confidence_navigation_fallbacks: 0,
            chosen_score_sum: 0.0,
            chosen_score_count: 0,
            candidate_score_sum: 0.0,
            candidate_score_count: 0,
            ticks_with_eye_frames: 0,
            ticks_with_ear_features: 0,
            ticks_with_voice_embeddings: 0,
            ticks_with_face_embeddings: 0,
            ticks_with_kinect_skeletons: 0,
            ticks_with_future_predictions: 0,
            memory: ScenarioEpisodeMemoryBuilder::default(),
        }
    }

    fn observe(&mut self, snapshot: &WorldSnapshot, tick: &RuntimeTick) {
        self.ticks = self.ticks.saturating_add(1);
        let body = &snapshot.body;
        self.started_battery.get_or_insert(body.battery_level);
        self.final_battery = body.battery_level;
        if self.dead_battery_tick.is_none() && body.battery_level <= f32::EPSILON && !body.charging
        {
            self.dead_battery_tick = Some(self.ticks.saturating_sub(1));
        }
        let position = (body.odometry.x_m, body.odometry.y_m);
        if self.start_position.is_none() {
            self.start_position = Some(position);
        }
        self.last_heading_rad = Some(body.odometry.heading_rad);
        if let Some(last) = self.last_position.replace(position) {
            let step_distance = distance_between(last, position);
            self.distance_traveled_m += step_distance;
        }
        let charger_near_score = sim_world_score(snapshot, 3);
        let charger_visible_score = sim_world_score(snapshot, 4);
        if charger_visible_score >= 0.20 {
            self.ticks_with_charger_visible = self.ticks_with_charger_visible.saturating_add(1);
        }
        if charger_near_score >= 0.25 || body.charging {
            self.ticks_with_charger_near = self.ticks_with_charger_near.saturating_add(1);
        }

        let bumper = body.flags.bump_left || body.flags.bump_right;
        let cliff = body.flags.cliff_left
            || body.flags.cliff_front_left
            || body.flags.cliff_front_right
            || body.flags.cliff_right;
        let collision = bumper || body.flags.wall || cliff;
        if collision {
            self.collisions = self.collisions.saturating_add(1);
        }
        if body.flags.wall {
            self.wall_hits = self.wall_hits.saturating_add(1);
        }
        if bumper {
            self.bumper_hits = self.bumper_hits.saturating_add(1);
        }
        if cliff {
            self.cliff_hits = self.cliff_hits.saturating_add(1);
        }
        if body.charging {
            if self.first_charge_tick.is_none() {
                self.first_charge_tick = Some(self.ticks.saturating_sub(1));
            }
            self.charging_ticks = self.charging_ticks.saturating_add(1);
        }
        if let Some(nearest) = snapshot.range.nearest_m {
            self.min_nearest_obstacle_m = Some(
                self.min_nearest_obstacle_m
                    .map(|value| value.min(nearest))
                    .unwrap_or(nearest),
            );
            self.nearest_obstacle_sum += nearest;
            self.nearest_obstacle_count = self.nearest_obstacle_count.saturating_add(1);
        }
        if let Some(action) = &tick.chosen_action {
            self.unique_actions.insert(format!("{action:?}"));
            *self
                .action_histogram
                .entry(action_histogram_label(action).to_string())
                .or_default() += 1;
            if matches!(
                action,
                ActionPrimitive::Approach {
                    target: ApproachTarget::Charger
                }
            ) {
                self.ticks_approaching_charger = self.ticks_approaching_charger.saturating_add(1);
            }
            if matches!(action, ActionPrimitive::Dock)
                && !body.charging
                && charger_near_score < 0.80
            {
                self.ticks_docking_from_too_far = self.ticks_docking_from_too_far.saturating_add(1);
            }
        }
        self.observe_stuck(snapshot);
        if tick
            .frame
            .now
            .extensions
            .get("safety.vetoed")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            self.safety_interventions = self.safety_interventions.saturating_add(1);
            if wall_or_cliff_veto(tick) {
                self.wall_cliff_veto_count = self.wall_cliff_veto_count.saturating_add(1);
            }
        }
        self.observe_behavior_runs(&tick.frame.behavior_runs);
        self.observe_action_selector(tick);
        self.observe_goal_progress(tick);
        self.observe_map_memory_decision(tick);
        if snapshot.eye_frame.is_some() || !snapshot.eye.frames.is_empty() {
            self.ticks_with_eye_frames = self.ticks_with_eye_frames.saturating_add(1);
        }
        if !snapshot.ear.features.is_empty() || snapshot.ear_pcm.is_some() {
            self.ticks_with_ear_features = self.ticks_with_ear_features.saturating_add(1);
        }
        if !snapshot.voice.vectors.is_empty() {
            self.ticks_with_voice_embeddings = self.ticks_with_voice_embeddings.saturating_add(1);
        }
        if !snapshot.face.vectors.is_empty() {
            self.ticks_with_face_embeddings = self.ticks_with_face_embeddings.saturating_add(1);
        }
        if !snapshot.kinect.skeletons.is_empty() {
            self.ticks_with_kinect_skeletons = self.ticks_with_kinect_skeletons.saturating_add(1);
        }
        if !tick.frame.predicted_futures.is_empty() {
            self.ticks_with_future_predictions =
                self.ticks_with_future_predictions.saturating_add(1);
        }
        self.memory.observe(snapshot, tick);
    }

    fn observe_stuck(&mut self, snapshot: &WorldSnapshot) {
        let Some(extension) = snapshot
            .extensions
            .iter()
            .find(|extension| extension.name == "sim.stuck")
        else {
            return;
        };
        let values = &extension.values;
        let active = values.first().copied().unwrap_or(0.0) > 0.0;
        let duration_ms = values.get(3).copied().unwrap_or(0.0).max(0.0);
        let event_started = values.get(6).copied().unwrap_or(0.0) > 0.0;
        let recovered = values.get(7).copied().unwrap_or(0.0) > 0.0;
        let trap_kind = trap_kind_label(values.get(10).copied().unwrap_or(0.0));
        let attempts = values.get(11).copied().unwrap_or(0.0).max(0.0) as usize;
        let repeated = values.get(12).copied().unwrap_or(0.0).max(0.0) as usize;
        if event_started {
            self.stuck_count = self.stuck_count.saturating_add(1);
            self.recovery_attempts = self.recovery_attempts.saturating_add(1);
            self.active_stuck_duration_ms = Some(duration_ms);
            if let Some(kind) = trap_kind {
                *self.trap_kind_counts.entry(kind.to_string()).or_default() += 1;
            }
        }
        if active {
            self.stuck_ticks = self.stuck_ticks.saturating_add(1);
            self.active_stuck_duration_ms = Some(duration_ms);
        }
        self.recovery_attempts = self.recovery_attempts.max(attempts);
        self.repeated_trap_count = self.repeated_trap_count.max(repeated);
        if recovered {
            self.recovery_successes = self.recovery_successes.saturating_add(1);
            let duration = if duration_ms > 0.0 {
                Some(duration_ms)
            } else {
                self.active_stuck_duration_ms
            };
            self.active_stuck_duration_ms = None;
            if let Some(duration) = duration {
                self.stuck_duration_sum_ms += duration;
                self.stuck_duration_count = self.stuck_duration_count.saturating_add(1);
                self.recovery_ticks_sum = self
                    .recovery_ticks_sum
                    .saturating_add((duration / 100.0).round().max(0.0) as usize);
                self.recovery_tick_count = self.recovery_tick_count.saturating_add(1);
            }
            self.distance_at_last_recovery_m = Some(self.distance_traveled_m);
        }
    }

    fn observe_behavior_runs(&mut self, records: &[ErasedBehaviorRunRecord]) {
        self.behavior_run_records = self.behavior_run_records.saturating_add(records.len());
        self.model_fallbacks = self.model_fallbacks.saturating_add(
            records
                .iter()
                .filter(|record| {
                    matches!(
                        record.regime,
                        BehaviorRegime::ModelInfer | BehaviorRegime::ModelTrainAndInfer
                    ) && (record.error.is_some()
                        || (record.model_json.is_none() && record.hardcoded_json.is_some()))
                })
                .count(),
        );
    }

    fn observe_action_selector(&mut self, tick: &RuntimeTick) {
        let Some(value) = tick.frame.now.extensions.get("action_selector") else {
            return;
        };
        let Ok(decision) = serde_json::from_value::<ActionSelectionDecision>(value.clone()) else {
            return;
        };
        if decision.mode == ActionSelectorMode::ModelAssisted {
            self.model_assisted_decisions = self.model_assisted_decisions.saturating_add(1);
        }
        if decision.safety_overrode {
            self.action_selector_safety_overrides =
                self.action_selector_safety_overrides.saturating_add(1);
        }
        if decision.goal_switched {
            self.goal_switches = self.goal_switches.saturating_add(1);
        }
        if decision.goal_retained_by_commitment {
            self.goal_commitment_retained_ticks =
                self.goal_commitment_retained_ticks.saturating_add(1);
        }
        if decision.shadow_diverged_from_baseline {
            self.goal_shadow_divergences = self.goal_shadow_divergences.saturating_add(1);
        }
        let observed_goal = decision
            .selected_goal
            .clone()
            .or(decision.shadow_selected_goal.clone());
        let observed_behavior = decision.selected_behavior.clone().or_else(|| {
            tick.frame
                .now
                .extensions
                .get("goal_system")
                .and_then(|value| value.get("behavior"))
                .and_then(|value| value.get("behavior_id"))
                .and_then(|value| value.as_str())
                .map(str::to_string)
        });
        if let Some(goal) = observed_goal {
            *self.goal_histogram.entry(goal.clone()).or_default() += 1;
            if self.current_goal.as_deref() == Some(goal.as_str()) {
                self.current_goal_ticks = self.current_goal_ticks.saturating_add(1);
            } else {
                if self.current_goal.is_some() {
                    self.goal_dwell_ticks_sum = self
                        .goal_dwell_ticks_sum
                        .saturating_add(self.current_goal_ticks);
                    self.goal_dwell_count = self.goal_dwell_count.saturating_add(1);
                }
                self.current_goal = Some(goal);
                self.current_goal_ticks = 1;
                self.current_goal_behavior = None;
            }
        }
        if let Some(behavior) = observed_behavior {
            *self
                .goal_behavior_histogram
                .entry(behavior.clone())
                .or_default() += 1;
            if self.current_goal_behavior.is_some()
                && self.current_goal_behavior.as_deref() != Some(behavior.as_str())
            {
                self.goal_behavior_transitions = self.goal_behavior_transitions.saturating_add(1);
            }
            self.current_goal_behavior = Some(behavior);
        }
        if decision
            .candidates
            .iter()
            .any(|candidate| candidate.fallback_used)
        {
            self.action_selector_fallbacks = self.action_selector_fallbacks.saturating_add(1);
        }
        if decision
            .fallback_warnings
            .iter()
            .any(|warning| warning.contains("baseline trap recovery"))
        {
            self.action_selector_guard_yields = self.action_selector_guard_yields.saturating_add(1);
        }
        if let Some(score) = decision.selected_score {
            self.chosen_score_sum += score;
            self.chosen_score_count = self.chosen_score_count.saturating_add(1);
        }
        for candidate in decision.candidates {
            self.candidate_score_sum += candidate.score;
            self.candidate_score_count = self.candidate_score_count.saturating_add(1);
        }
    }

    fn observe_goal_progress(&mut self, tick: &RuntimeTick) {
        let Some(progress) = tick
            .frame
            .now
            .extensions
            .get("goal_system")
            .and_then(|cycle| cycle.get("progress"))
        else {
            return;
        };
        let Ok(reports) = serde_json::from_value::<Vec<GoalProgressReport>>(progress.clone())
        else {
            return;
        };
        for report in reports {
            let goal_id = report.goal_id.as_str().to_string();
            self.goal_failed_attempts
                .entry(goal_id)
                .and_modify(|attempts| *attempts = (*attempts).max(report.failed_attempts))
                .or_insert(report.failed_attempts);

            let observed_this_tick = report
                .observation
                .as_ref()
                .is_some_and(|observation| observation.observed_at_ms == tick.frame.now.t_ms);
            let meets_expectation = report
                .observation
                .as_ref()
                .and_then(|observation| observation.progress)
                .zip(report.expectation.as_ref())
                .is_some_and(|(observed, expected)| {
                    observed + expected.tolerance >= expected.expected_progress
                });
            if report.selected_behavior.is_some() && observed_this_tick {
                match report
                    .observation
                    .as_ref()
                    .and_then(|observation| observation.progress)
                {
                    Some(progress) => {
                        self.goal_progress_sum += progress;
                        self.goal_progress_samples = self.goal_progress_samples.saturating_add(1);
                        if !meets_expectation {
                            self.goal_no_progress_dwell_ticks =
                                self.goal_no_progress_dwell_ticks.saturating_add(1);
                        }
                    }
                    None => {
                        self.unmeasurable_progress_ticks =
                            self.unmeasurable_progress_ticks.saturating_add(1);
                    }
                }
            }

            match report.response {
                StrategyProgressResponse::Changed => {
                    self.strategy_switches_within_goal =
                        self.strategy_switches_within_goal.saturating_add(1);
                    self.stall_responses = self.stall_responses.saturating_add(1);
                    if meets_expectation {
                        self.false_stall_count = self.false_stall_count.saturating_add(1);
                    }
                }
                StrategyProgressResponse::HelpRequested => {
                    self.goal_help_requests = self.goal_help_requests.saturating_add(1);
                    self.stall_responses = self.stall_responses.saturating_add(1);
                    if meets_expectation {
                        self.false_stall_count = self.false_stall_count.saturating_add(1);
                    }
                }
                StrategyProgressResponse::Abandoned => {
                    self.stall_responses = self.stall_responses.saturating_add(1);
                    if meets_expectation {
                        self.false_stall_count = self.false_stall_count.saturating_add(1);
                    }
                }
                StrategyProgressResponse::Inactive
                | StrategyProgressResponse::Started
                | StrategyProgressResponse::Retained => {}
            }
        }
    }

    fn observe_map_memory_decision(&mut self, tick: &RuntimeTick) {
        let Some(decision) = tick
            .frame
            .now
            .extensions
            .get("action.motion_bridge")
            .and_then(|value| value.get("map_memory_decision"))
        else {
            return;
        };
        if !decision
            .get("influenced")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            return;
        }
        self.map_memory_decisions = self.map_memory_decisions.saturating_add(1);
        let reason = decision
            .get("reason")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        if let Some(intent) = decision
            .get("navigation_intent")
            .and_then(|value| value.as_str())
        {
            *self
                .memory_navigation_intents
                .entry(intent.to_string())
                .or_default() += 1;
        }
        if !reason.is_empty() {
            *self
                .memory_navigation_reasons
                .entry(reason.to_string())
                .or_default() += 1;
        }
        if let Some(signal) = decision.get("signal").and_then(|value| value.as_str()) {
            *self
                .map_memory_signals
                .entry(signal.to_string())
                .or_default() += 1;
        }
        if decision
            .get("safety_overrode")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            self.map_memory_safety_overrides = self.map_memory_safety_overrides.saturating_add(1);
        }
        if self.map_memory_decision_samples.len() < 16 {
            if let Some(sample) = scenario_map_memory_decision_report(decision) {
                self.map_memory_decision_samples.push(sample);
            }
        }
        if decision
            .get("confidence")
            .and_then(|value| value.as_f64())
            .map(|confidence| confidence < 0.35)
            .unwrap_or(false)
        {
            self.low_confidence_navigation_fallbacks =
                self.low_confidence_navigation_fallbacks.saturating_add(1);
        }
        if reason.starts_with("danger_") {
            self.danger_memory_decisions = self.danger_memory_decisions.saturating_add(1);
        } else if reason.starts_with("charge_") {
            self.charge_memory_decisions = self.charge_memory_decisions.saturating_add(1);
        } else if reason.starts_with("safe_novelty_") {
            self.novelty_memory_decisions = self.novelty_memory_decisions.saturating_add(1);
        } else if reason.starts_with("frontier_") {
            self.frontier_memory_decisions = self.frontier_memory_decisions.saturating_add(1);
        } else if reason.starts_with("recent_trap_") {
            self.trap_memory_decisions = self.trap_memory_decisions.saturating_add(1);
        }
    }

    fn finish(self) -> ScenarioEpisodeReport {
        let final_position = self.last_position.unwrap_or_else(|| {
            (
                self.metadata.body.odometry.x_m,
                self.metadata.body.odometry.y_m,
            )
        });
        let started_battery = self
            .started_battery
            .unwrap_or(self.metadata.body.battery_level);
        let final_distance_to_charger_m =
            nearest_object_distance(final_position, &self.metadata.objects, |kind| {
                matches!(kind, pete_sim::SimObjectKind::Charger)
            });
        let final_bearing_to_charger_rad = nearest_object_bearing(
            final_position,
            self.last_heading_rad.unwrap_or(0.0),
            &self.metadata.objects,
            |kind| matches!(kind, pete_sim::SimObjectKind::Charger),
        );
        let final_distance_to_person_m =
            nearest_object_distance(final_position, &self.metadata.objects, |kind| {
                matches!(kind, pete_sim::SimObjectKind::Person { .. })
            });
        let final_distance_to_speaker_m =
            nearest_object_distance(final_position, &self.metadata.objects, |kind| {
                matches!(kind, pete_sim::SimObjectKind::SoundSource { .. })
            });
        let mean_nearest_obstacle_m = if self.nearest_obstacle_count == 0 {
            None
        } else {
            Some(self.nearest_obstacle_sum / self.nearest_obstacle_count as f32)
        };
        let mut stuck_duration_sum_ms = self.stuck_duration_sum_ms;
        let mut stuck_duration_count = self.stuck_duration_count;
        if let Some(duration) = self.active_stuck_duration_ms {
            stuck_duration_sum_ms += duration;
            stuck_duration_count = stuck_duration_count.saturating_add(1);
        }
        let stuck_duration = (stuck_duration_count > 0)
            .then_some(stuck_duration_sum_ms / stuck_duration_count as f32);
        let mut report = ScenarioEpisodeReport {
            index: self.index,
            seed: self.seed,
            success: false,
            ticks: self.ticks,
            collisions: self.collisions,
            wall_hits: self.wall_hits,
            bumper_hits: self.bumper_hits,
            cliff_hits: self.cliff_hits,
            charging_ticks: self.charging_ticks,
            first_charge_tick: self.first_charge_tick,
            started_battery,
            final_battery: self.final_battery,
            battery_delta: self.final_battery - started_battery,
            min_nearest_obstacle_m: self.min_nearest_obstacle_m,
            mean_nearest_obstacle_m,
            final_distance_to_charger_m,
            final_heading_rad: self.last_heading_rad,
            final_bearing_to_charger_rad,
            final_distance_to_person_m,
            final_distance_to_speaker_m,
            distance_traveled_m: self.distance_traveled_m,
            ticks_with_charger_visible: self.ticks_with_charger_visible,
            ticks_with_charger_near: self.ticks_with_charger_near,
            ticks_approaching_charger: self.ticks_approaching_charger,
            ticks_docking_from_too_far: self.ticks_docking_from_too_far,
            action_histogram: self.action_histogram,
            wall_cliff_veto_count: self.wall_cliff_veto_count,
            escape_progress_score: escape_progress_score(
                self.kind,
                self.distance_traveled_m,
                self.distance_at_last_recovery_m,
                self.collisions,
                self.stuck_ticks,
                self.ticks,
            ),
            stuck_ticks: self.stuck_ticks,
            stuck_count: self.stuck_count,
            trap_kind_counts: self.trap_kind_counts,
            recovery_attempts: self.recovery_attempts,
            stuck_duration,
            mean_stuck_duration: stuck_duration,
            recovery_success_rate: (self.stuck_count > 0)
                .then_some(self.recovery_successes as f32 / self.stuck_count as f32),
            mean_recovery_ticks: (self.recovery_tick_count > 0)
                .then_some(self.recovery_ticks_sum as f32 / self.recovery_tick_count as f32),
            repeated_trap_count: self.repeated_trap_count,
            dead_battery_tick: self.dead_battery_tick,
            distance_after_recovery_m: self
                .distance_at_last_recovery_m
                .map(|distance| (self.distance_traveled_m - distance).max(0.0)),
            unique_actions: self.unique_actions.into_iter().collect(),
            safety_interventions: self.safety_interventions,
            behavior_run_records: self.behavior_run_records,
            model_fallbacks: self.model_fallbacks,
            model_assisted_decisions: self.model_assisted_decisions,
            action_selector_safety_overrides: self.action_selector_safety_overrides,
            goal_switches: self.goal_switches,
            goal_commitment_retained_ticks: self.goal_commitment_retained_ticks,
            goal_behavior_transitions: self.goal_behavior_transitions,
            goal_shadow_divergences: self.goal_shadow_divergences,
            mean_goal_dwell_ms: {
                let dwell_sum = self
                    .goal_dwell_ticks_sum
                    .saturating_add(self.current_goal_ticks);
                let dwell_count = self
                    .goal_dwell_count
                    .saturating_add(usize::from(self.current_goal.is_some()));
                (dwell_count > 0).then_some(dwell_sum as f32 * 100.0 / dwell_count as f32)
            },
            goal_histogram: self.goal_histogram,
            goal_behavior_histogram: self.goal_behavior_histogram,
            goal_progress_samples: self.goal_progress_samples,
            mean_goal_progress: (self.goal_progress_samples > 0)
                .then_some(self.goal_progress_sum / self.goal_progress_samples as f32),
            goal_no_progress_dwell_ticks: self.goal_no_progress_dwell_ticks,
            goal_failed_attempts: self
                .goal_failed_attempts
                .values()
                .map(|attempts| *attempts as usize)
                .sum(),
            strategy_switches_within_goal: self.strategy_switches_within_goal,
            goal_help_requests: self.goal_help_requests,
            unmeasurable_progress_ticks: self.unmeasurable_progress_ticks,
            stall_responses: self.stall_responses,
            false_stall_count: self.false_stall_count,
            false_stall_rate: (self.stall_responses > 0)
                .then_some(self.false_stall_count as f32 / self.stall_responses as f32),
            action_selector_fallbacks: self.action_selector_fallbacks,
            action_selector_guard_yields: self.action_selector_guard_yields,
            map_memory_decisions: self.map_memory_decisions,
            danger_memory_decisions: self.danger_memory_decisions,
            charge_memory_decisions: self.charge_memory_decisions,
            novelty_memory_decisions: self.novelty_memory_decisions,
            frontier_memory_decisions: self.frontier_memory_decisions,
            trap_memory_decisions: self.trap_memory_decisions,
            memory_navigation_intents: self.memory_navigation_intents,
            memory_navigation_reasons: self.memory_navigation_reasons,
            map_memory_signals: self.map_memory_signals,
            map_memory_safety_overrides: self.map_memory_safety_overrides,
            map_memory_decision_samples: self.map_memory_decision_samples,
            low_confidence_navigation_fallbacks: self.low_confidence_navigation_fallbacks,
            mean_chosen_score: (self.chosen_score_count > 0)
                .then_some(self.chosen_score_sum / self.chosen_score_count as f32),
            mean_candidate_score: (self.candidate_score_count > 0)
                .then_some(self.candidate_score_sum / self.candidate_score_count as f32),
            ticks_with_eye_frames: self.ticks_with_eye_frames,
            ticks_with_ear_features: self.ticks_with_ear_features,
            ticks_with_voice_embeddings: self.ticks_with_voice_embeddings,
            ticks_with_face_embeddings: self.ticks_with_face_embeddings,
            ticks_with_kinect_skeletons: self.ticks_with_kinect_skeletons,
            ticks_with_future_predictions: self.ticks_with_future_predictions,
            memory: Some(self.memory.finish()),
            capture: self.capture,
            ledger: self.ledger,
        };
        report.success = episode_success(self.kind, &report);
        report
    }
}

fn scenario_map_memory_decision_report(
    decision: &serde_json::Value,
) -> Option<ScenarioMapMemoryDecisionReport> {
    let signal = decision.get("signal")?.as_str()?.to_string();
    let reason = decision.get("reason")?.as_str()?.to_string();
    let chosen_action = decision
        .get("chosen_action")
        .or_else(|| decision.get("selected_action"))
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok());
    Some(ScenarioMapMemoryDecisionReport {
        signal,
        signal_value: decision
            .get("signal_value")
            .and_then(|value| value.as_f64())
            .map(|value| value as f32),
        signal_confidence: decision
            .get("signal_confidence")
            .or_else(|| decision.get("confidence"))
            .and_then(|value| value.as_f64())
            .unwrap_or(0.0) as f32,
        chosen_action,
        reason,
        reason_string: decision
            .get("reason_string")
            .and_then(|value| value.as_str())
            .map(str::to_string),
        safety_overrode: decision
            .get("safety_overrode")
            .and_then(|value| value.as_bool())
            .unwrap_or(false),
    })
}

#[derive(Clone, Debug, Default)]
struct ScenarioEpisodeMemoryBuilder {
    max_places_visited: usize,
    charge_memory_ticks: usize,
    charge_opportunity_ticks: usize,
    danger_memory_ticks: usize,
    danger_opportunity_ticks: usize,
    social_memory_ticks: usize,
    social_opportunity_ticks: usize,
    first_novelty: Option<f32>,
    final_novelty: Option<f32>,
}

impl ScenarioEpisodeMemoryBuilder {
    fn observe(&mut self, snapshot: &WorldSnapshot, tick: &RuntimeTick) {
        let memory = &tick.frame.now.memory;
        self.max_places_visited = self.max_places_visited.max(memory.places_visited as usize);
        self.first_novelty.get_or_insert(memory.place_novelty);
        self.final_novelty = Some(memory.place_novelty);

        let charger_near = snapshot.body.charging
            || sim_world_score(snapshot, 3).max(sim_world_score(snapshot, 4)) >= 0.3;
        if charger_near {
            self.charge_opportunity_ticks = self.charge_opportunity_ticks.saturating_add(1);
            if memory.place_charge_value >= 0.3 {
                self.charge_memory_ticks = self.charge_memory_ticks.saturating_add(1);
            }
        }

        let danger_near = snapshot.body.flags.bump_left
            || snapshot.body.flags.bump_right
            || snapshot.body.flags.wall
            || snapshot.body.flags.cliff_left
            || snapshot.body.flags.cliff_front_left
            || snapshot.body.flags.cliff_front_right
            || snapshot.body.flags.cliff_right
            || snapshot
                .range
                .nearest_m
                .map(|nearest| nearest <= 0.35)
                .unwrap_or(false);
        if danger_near {
            self.danger_opportunity_ticks = self.danger_opportunity_ticks.saturating_add(1);
            if memory.place_danger >= 0.3 {
                self.danger_memory_ticks = self.danger_memory_ticks.saturating_add(1);
            }
        }

        let social_seen = !snapshot.face.vectors.is_empty()
            || !snapshot.voice.vectors.is_empty()
            || !snapshot.kinect.skeletons.is_empty();
        if social_seen {
            self.social_opportunity_ticks = self.social_opportunity_ticks.saturating_add(1);
            if memory.place_social_value >= 0.3 {
                self.social_memory_ticks = self.social_memory_ticks.saturating_add(1);
            }
        }
    }

    fn finish(self) -> ScenarioEpisodeMemoryReport {
        ScenarioEpisodeMemoryReport {
            places_visited: self.max_places_visited,
            charge_memory_ticks: self.charge_memory_ticks,
            charge_opportunity_ticks: self.charge_opportunity_ticks,
            charge_memory_hit_rate: hit_rate(
                self.charge_memory_ticks,
                self.charge_opportunity_ticks,
            ),
            danger_memory_ticks: self.danger_memory_ticks,
            danger_opportunity_ticks: self.danger_opportunity_ticks,
            danger_memory_hit_rate: hit_rate(
                self.danger_memory_ticks,
                self.danger_opportunity_ticks,
            ),
            social_memory_ticks: self.social_memory_ticks,
            social_opportunity_ticks: self.social_opportunity_ticks,
            social_memory_hit_rate: hit_rate(
                self.social_memory_ticks,
                self.social_opportunity_ticks,
            ),
            first_novelty: self.first_novelty,
            final_novelty: self.final_novelty,
            novelty_decayed: self
                .first_novelty
                .zip(self.final_novelty)
                .map(|(first, final_value)| final_value <= first)
                .unwrap_or(false),
        }
    }
}

async fn run_social_exam_command(args: SocialExamArgs) -> Result<()> {
    let report = pete_runtime::run_social_exam().await?;
    for case in &report.cases {
        println!(
            "{:<28} {}",
            case.case,
            if case.passed { "PASS" } else { "FAIL" }
        );
        for failure in &case.failures {
            println!("  {failure}");
        }
    }
    if let Some(out) = args.out.as_deref() {
        let path = Path::new(out);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(path, serde_json::to_vec_pretty(&report)?)?;
        println!("social exam report written: {out}");
    }
    if !report.passed {
        anyhow::bail!("social exam failed");
    }
    println!("social exam passed: {} cases", report.cases.len());
    Ok(())
}

async fn run_eval_scenario(args: EvalScenarioArgs) -> Result<()> {
    let kind = ScenarioKind::from(args.scenario);
    let flags = RuntimeModelFlags::from(&args);
    let mut model_loading = load_runtime_models_from_flags(&flags)?.1;
    if args.future_mode == FutureMode::ModelInfer {
        model_loading.blocked_model_infer.push(
            "future model-infer is limited to prediction behavior; motor safety remains hardcoded"
                .to_string(),
        );
    }
    if args.experience_mode == ExperienceMode::ModelInfer {
        model_loading.blocked_model_infer.push(
            "experience model-infer changes latent encoding only; motor safety remains hardcoded"
                .to_string(),
        );
    }

    let mut episodes_detail = Vec::with_capacity(args.episodes);
    for episode_index in 0..args.episodes {
        let episode_seed = args.seed.saturating_add(episode_index as u64);
        let scenario = build_scenario(ScenarioConfig::new(kind, episode_seed));
        let capture = args.capture_root.as_ref().map(|root| {
            Path::new(root)
                .join(format!("episode-{episode_index:03}"))
                .to_string_lossy()
                .to_string()
        });
        let builder = EpisodeMetricBuilder::new(
            kind,
            scenario.metadata.clone(),
            episode_index,
            episode_seed,
            args.ledger.clone(),
            capture.clone(),
        );
        let (episode, warnings) = if let Some(ledger_path) = &args.ledger {
            let mut runtime = default_runtime(JsonlLedger::new(ledger_path), &args.llm)?;
            runtime = runtime.with_action_selector_mode(args.action_selector.into());
            if let Some(models) = load_runtime_models_from_flags(&flags)?.0 {
                runtime = runtime.with_models(models);
            }
            run_eval_episode(runtime, scenario.world, scenario.motors, &args, builder).await?
        } else {
            let mut runtime = default_noop_runtime(&args.llm)?;
            runtime = runtime.with_action_selector_mode(args.action_selector.into());
            if let Some(models) = load_runtime_models_from_flags(&flags)?.0 {
                runtime = runtime.with_models(models);
            }
            run_eval_episode(runtime, scenario.world, scenario.motors, &args, builder).await?
        };
        model_loading.warnings.extend(warnings);
        println!(
            "eval episode {} complete: scenario {}, seed {}, ticks {}, success {}, collisions {}",
            episode.index,
            kind.slug(),
            episode.seed,
            episode.ticks,
            episode.success,
            episode.collisions
        );
        episodes_detail.push(episode);
    }

    let summary = summarize_episodes(&episodes_detail);
    let memory = args
        .memory_report
        .then(|| summarize_episode_memory(&episodes_detail));
    let recommendation = scenario_recommendation(args.episodes, &summary);
    let report = ScenarioEvaluationReport {
        schema_version: 1,
        scenario: kind.slug().to_string(),
        base_seed: args.seed,
        episodes: args.episodes,
        steps_per_episode: args.steps,
        tick_ms: args.tick_ms,
        action_selector_mode: ActionSelectorMode::from(args.action_selector)
            .as_str()
            .to_string(),
        model_modes: model_modes_from_flags(&flags),
        model_loading: model_loading.clone(),
        ledger: args.ledger.clone(),
        capture_root: args.capture_root.clone(),
        summary,
        memory,
        episodes_detail,
        recommendation,
        warnings: model_loading.warnings.clone(),
    };

    let bytes = serde_json::to_vec_pretty(&report)?;
    if let Some(out) = &args.out {
        if let Some(parent) = Path::new(out).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(out, &bytes)?;
        println!("scenario evaluation report written: {out}");
    } else {
        println!("{}", String::from_utf8_lossy(&bytes));
    }
    Ok(())
}

async fn run_eval_episode<R>(
    runtime: R,
    world: pete_sim::VirtualWorld,
    motors: pete_sim::SimCockpit,
    args: &EvalScenarioArgs,
    mut metrics: EpisodeMetricBuilder,
) -> Result<(ScenarioEpisodeReport, Vec<String>)>
where
    R: RuntimeLoop + Send,
{
    let mut warnings = Vec::new();
    let mut runner = SimRunner::new(runtime, world, motors);
    runner.tick_ms = args.tick_ms;
    let mut snapshots = Vec::new();
    runner
        .run_steps_observing_ticks(args.steps, |snapshot, tick| {
            if metrics.capture.is_some() {
                snapshots.push(snapshot.clone());
            }
            metrics.observe(snapshot, tick);
        })
        .await?;

    if let Some(capture_path) = &metrics.capture {
        let mut writer =
            CaptureWriter::create(capture_path, CaptureSource::Sim, Some(args.tick_ms)).await?;
        writer.manifest_mut().scenario = Some(metrics.metadata.clone());
        for snapshot in snapshots {
            writer
                .append_snapshot(snapshot.body.last_update_ms, snapshot, Vec::new())
                .await?;
        }
        writer.finish().await?;
    }

    if runner.tick_count < args.steps {
        warnings.push(format!(
            "episode {} stopped after {} ticks before requested {} steps",
            metrics.index, runner.tick_count, args.steps
        ));
    }
    Ok((metrics.finish(), warnings))
}

fn configured_llm_config(args: &LlmArgs) -> Result<LlmConfig> {
    let mut config = match &args.llm_config {
        Some(path) => LlmConfig::load(path)?,
        None => LlmConfig::default(),
    };
    if let Some(provider) = args.llm_provider {
        config.provider = provider.into();
    }
    Ok(config)
}

fn configured_llm_agent(args: &LlmArgs) -> Result<ConfiguredLlmAgent> {
    let config = configured_llm_config(args)?;
    ConfiguredLlmAgent::from_config(config)
}

fn configured_llm_config_for_sim(args: &LlmArgs, live: bool) -> Result<LlmConfig> {
    let mut config = configured_llm_config(args)?;
    if live && args.llm_config.is_none() {
        let live_timeout_ms = std::env::var("PETE_LIVE_LLM_TIMEOUT_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_LIVE_LLM_TIMEOUT_MS);
        config.timeout_ms = config.timeout_ms.min(live_timeout_ms.max(1));
    }
    Ok(config)
}

fn configured_llm_agent_for_sim(args: &LlmArgs, live: bool) -> Result<ConfiguredLlmAgent> {
    let config = configured_llm_config_for_sim(args, live)?;
    ConfiguredLlmAgent::from_config(config)
}

fn default_noop_runtime(
    llm_args: &LlmArgs,
) -> Result<
    MinimalRuntime<
        NoopLedger,
        InMemoryExperienceStore,
        InMemoryExperienceStore,
        SimpleConductor,
        SimpleSafety,
        ConfiguredLlmAgent,
    >,
> {
    let memory = InMemoryExperienceStore::new();
    let recall = memory.clone();
    Ok(MinimalRuntime::with_default_events(
        NoopLedger,
        memory,
        recall,
        SimpleConductor::default(),
        SimpleSafety::default(),
        configured_llm_agent(llm_args)?,
    ))
}

fn episode_success(kind: ScenarioKind, episode: &ScenarioEpisodeReport) -> bool {
    match kind {
        ScenarioKind::EmptyRoom => episode.ticks > 0 && episode.collisions == 0,
        ScenarioKind::ObstacleAvoidance => {
            episode.ticks > 0
                && episode.collisions <= (episode.ticks / 50).max(1)
                && episode.stuck_ticks < episode.ticks / 2
                && episode.distance_traveled_m > 0.05
        }
        ScenarioKind::CornerTrap | ScenarioKind::ConcaveTrap => {
            episode.stuck_count > 0
                && episode.recovery_success_rate.unwrap_or(0.0) > 0.0
                && episode.distance_traveled_m > 0.10
                && episode.collisions <= (episode.ticks / 20).max(1)
        }
        ScenarioKind::ColumnTrap => {
            episode.stuck_count > 0
                && episode.recovery_success_rate.unwrap_or(0.0) > 0.0
                && episode.distance_traveled_m > 0.25
                && episode.collisions <= (episode.ticks / 20).max(1)
        }
        ScenarioKind::ChargerSeeking => {
            episode.charging_ticks > 0 && episode.dead_battery_tick.is_none()
        }
        ScenarioKind::PersonAndSpeaker => {
            episode.ticks > 0
                && episode.collisions == 0
                && (episode.ticks_with_face_embeddings > 0
                    || episode.ticks_with_voice_embeddings > 0
                    || episode.ticks_with_kinect_skeletons > 0
                    || episode.ticks_with_ear_features > 0)
        }
        ScenarioKind::MixedRoom => {
            episode.ticks > 0
                && episode.collisions <= (episode.ticks / 40).max(1)
                && (episode.charging_ticks > 0
                    || episode.ticks_with_face_embeddings > 0
                    || episode.ticks_with_voice_embeddings > 0)
        }
        ScenarioKind::Dream => {
            episode.ticks > 0
                && episode.collisions <= (episode.ticks / 30).max(1)
                && (episode.charging_ticks > 0
                    || episode.ticks_with_face_embeddings > 0
                    || episode.ticks_with_voice_embeddings > 0
                    || episode.ticks_with_ear_features > 0)
        }
    }
}

fn summarize_episodes(episodes: &[ScenarioEpisodeReport]) -> ScenarioEvaluationSummary {
    if episodes.is_empty() {
        return ScenarioEvaluationSummary::default();
    }
    let count = episodes.len() as f32;
    let total_ticks: usize = episodes.iter().map(|episode| episode.ticks).sum();
    let total_collisions: usize = episodes.iter().map(|episode| episode.collisions).sum();
    let mut trap_kind_counts = HashMap::new();
    for episode in episodes {
        for (kind, count) in &episode.trap_kind_counts {
            *trap_kind_counts.entry(kind.clone()).or_default() += count;
        }
    }
    let goal_progress_samples = episodes
        .iter()
        .map(|episode| episode.goal_progress_samples)
        .sum::<usize>();
    let goal_progress_sum = episodes
        .iter()
        .map(|episode| {
            episode.mean_goal_progress.unwrap_or(0.0) * episode.goal_progress_samples as f32
        })
        .sum::<f32>();
    let stall_responses = episodes
        .iter()
        .map(|episode| episode.stall_responses)
        .sum::<usize>();
    let false_stall_count = episodes
        .iter()
        .map(|episode| episode.false_stall_count)
        .sum::<usize>();
    ScenarioEvaluationSummary {
        success_rate: episodes.iter().filter(|episode| episode.success).count() as f32 / count,
        collision_rate: if total_ticks == 0 {
            0.0
        } else {
            total_collisions as f32 / total_ticks as f32
        },
        mean_collisions_per_episode: total_collisions as f32 / count,
        mean_battery_delta: mean(episodes.iter().map(|episode| episode.battery_delta)),
        mean_final_battery: mean(episodes.iter().map(|episode| episode.final_battery)),
        mean_distance_to_charger_final_m: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.final_distance_to_charger_m),
        ),
        ticks_with_charger_visible: episodes
            .iter()
            .map(|episode| episode.ticks_with_charger_visible)
            .sum(),
        ticks_with_charger_near: episodes
            .iter()
            .map(|episode| episode.ticks_with_charger_near)
            .sum(),
        ticks_approaching_charger: episodes
            .iter()
            .map(|episode| episode.ticks_approaching_charger)
            .sum(),
        ticks_docking_from_too_far: episodes
            .iter()
            .map(|episode| episode.ticks_docking_from_too_far)
            .sum(),
        mean_nearest_obstacle_m: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.mean_nearest_obstacle_m),
        ),
        mean_distance_traveled_m: mean(episodes.iter().map(|episode| episode.distance_traveled_m)),
        action_histogram: summarize_action_histogram(episodes),
        wall_cliff_veto_count: episodes
            .iter()
            .map(|episode| episode.wall_cliff_veto_count)
            .sum(),
        escape_progress_score: mean(episodes.iter().map(|episode| episode.escape_progress_score)),
        mean_ticks_survived: mean(episodes.iter().map(|episode| episode.ticks as f32)),
        stuck_count: episodes.iter().map(|episode| episode.stuck_count).sum(),
        trap_kind_counts,
        recovery_attempts: episodes
            .iter()
            .map(|episode| episode.recovery_attempts)
            .sum(),
        stuck_duration: mean_optional(episodes.iter().filter_map(|episode| episode.stuck_duration)),
        mean_stuck_duration: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.mean_stuck_duration),
        ),
        recovery_success_rate: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.recovery_success_rate),
        ),
        mean_recovery_ticks: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.mean_recovery_ticks),
        ),
        repeated_trap_count: episodes
            .iter()
            .map(|episode| episode.repeated_trap_count)
            .sum(),
        dead_battery_tick: episodes
            .iter()
            .filter_map(|episode| episode.dead_battery_tick)
            .min(),
        distance_after_recovery_m: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.distance_after_recovery_m),
        ),
        mean_safety_interventions: mean(
            episodes
                .iter()
                .map(|episode| episode.safety_interventions as f32),
        ),
        behavior_run_records: episodes
            .iter()
            .map(|episode| episode.behavior_run_records)
            .sum(),
        model_fallbacks: episodes.iter().map(|episode| episode.model_fallbacks).sum(),
        action_selector_fallbacks: episodes
            .iter()
            .map(|episode| episode.action_selector_fallbacks)
            .sum(),
        action_selector_guard_yields: episodes
            .iter()
            .map(|episode| episode.action_selector_guard_yields)
            .sum(),
        map_memory_decisions: episodes
            .iter()
            .map(|episode| episode.map_memory_decisions)
            .sum(),
        danger_memory_decisions: episodes
            .iter()
            .map(|episode| episode.danger_memory_decisions)
            .sum(),
        charge_memory_decisions: episodes
            .iter()
            .map(|episode| episode.charge_memory_decisions)
            .sum(),
        novelty_memory_decisions: episodes
            .iter()
            .map(|episode| episode.novelty_memory_decisions)
            .sum(),
        frontier_memory_decisions: episodes
            .iter()
            .map(|episode| episode.frontier_memory_decisions)
            .sum(),
        trap_memory_decisions: episodes
            .iter()
            .map(|episode| episode.trap_memory_decisions)
            .sum(),
        memory_navigation_intents: summarize_memory_navigation_intents(episodes),
        memory_navigation_reasons: summarize_memory_navigation_reasons(episodes),
        map_memory_signals: summarize_map_memory_signals(episodes),
        map_memory_safety_overrides: episodes
            .iter()
            .map(|episode| episode.map_memory_safety_overrides)
            .sum(),
        low_confidence_navigation_fallbacks: episodes
            .iter()
            .map(|episode| episode.low_confidence_navigation_fallbacks)
            .sum(),
        model_assisted_decisions: episodes
            .iter()
            .map(|episode| episode.model_assisted_decisions)
            .sum(),
        action_selector_safety_overrides: episodes
            .iter()
            .map(|episode| episode.action_selector_safety_overrides)
            .sum(),
        goal_switches: episodes.iter().map(|episode| episode.goal_switches).sum(),
        goal_commitment_retained_ticks: episodes
            .iter()
            .map(|episode| episode.goal_commitment_retained_ticks)
            .sum(),
        goal_behavior_transitions: episodes
            .iter()
            .map(|episode| episode.goal_behavior_transitions)
            .sum(),
        goal_shadow_divergences: episodes
            .iter()
            .map(|episode| episode.goal_shadow_divergences)
            .sum(),
        mean_goal_dwell_ms: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.mean_goal_dwell_ms),
        ),
        goal_histogram: summarize_string_histogram(
            episodes.iter().map(|episode| &episode.goal_histogram),
        ),
        goal_behavior_histogram: summarize_string_histogram(
            episodes
                .iter()
                .map(|episode| &episode.goal_behavior_histogram),
        ),
        goal_progress_samples,
        mean_goal_progress: (goal_progress_samples > 0)
            .then_some(goal_progress_sum / goal_progress_samples as f32),
        goal_no_progress_dwell_ticks: episodes
            .iter()
            .map(|episode| episode.goal_no_progress_dwell_ticks)
            .sum(),
        goal_failed_attempts: episodes
            .iter()
            .map(|episode| episode.goal_failed_attempts)
            .sum(),
        strategy_switches_within_goal: episodes
            .iter()
            .map(|episode| episode.strategy_switches_within_goal)
            .sum(),
        goal_help_requests: episodes
            .iter()
            .map(|episode| episode.goal_help_requests)
            .sum(),
        unmeasurable_progress_ticks: episodes
            .iter()
            .map(|episode| episode.unmeasurable_progress_ticks)
            .sum(),
        false_stall_rate: (stall_responses > 0)
            .then_some(false_stall_count as f32 / stall_responses as f32),
        mean_chosen_score: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.mean_chosen_score),
        ),
        mean_candidate_score: mean_optional(
            episodes
                .iter()
                .filter_map(|episode| episode.mean_candidate_score),
        ),
    }
}

fn summarize_action_histogram(episodes: &[ScenarioEpisodeReport]) -> HashMap<String, usize> {
    let mut histogram = HashMap::new();
    for episode in episodes {
        for (action, count) in &episode.action_histogram {
            *histogram.entry(action.clone()).or_default() += count;
        }
    }
    histogram
}

fn summarize_string_histogram<'a>(
    histograms: impl IntoIterator<Item = &'a HashMap<String, usize>>,
) -> HashMap<String, usize> {
    let mut combined = HashMap::new();
    for histogram in histograms {
        for (key, count) in histogram {
            *combined.entry(key.clone()).or_default() += count;
        }
    }
    combined
}

fn summarize_memory_navigation_intents(
    episodes: &[ScenarioEpisodeReport],
) -> HashMap<String, usize> {
    let mut histogram = HashMap::new();
    for episode in episodes {
        for (intent, count) in &episode.memory_navigation_intents {
            *histogram.entry(intent.clone()).or_default() += count;
        }
    }
    histogram
}

fn summarize_memory_navigation_reasons(
    episodes: &[ScenarioEpisodeReport],
) -> HashMap<String, usize> {
    let mut histogram = HashMap::new();
    for episode in episodes {
        for (reason, count) in &episode.memory_navigation_reasons {
            *histogram.entry(reason.clone()).or_default() += count;
        }
    }
    histogram
}

fn summarize_map_memory_signals(episodes: &[ScenarioEpisodeReport]) -> HashMap<String, usize> {
    let mut histogram = HashMap::new();
    for episode in episodes {
        for (signal, count) in &episode.map_memory_signals {
            *histogram.entry(signal.clone()).or_default() += count;
        }
    }
    histogram
}

fn action_histogram_label(action: &ActionPrimitive) -> &'static str {
    match action {
        ActionPrimitive::Stop => "Stop",
        ActionPrimitive::Go { intensity, .. } if *intensity < 0.0 => "Reverse",
        ActionPrimitive::Go { .. } => "Go",
        ActionPrimitive::Drive { .. } => "Drive",
        ActionPrimitive::Turn {
            direction: TurnDir::Left,
            ..
        } => "TurnLeft",
        ActionPrimitive::Turn {
            direction: TurnDir::Right,
            ..
        } => "TurnRight",
        ActionPrimitive::Inspect { .. } => "Inspect",
        ActionPrimitive::Approach { .. } => "Approach",
        ActionPrimitive::Dock => "Dock",
        ActionPrimitive::Explore { .. } => "Explore",
        ActionPrimitive::Speak { .. } => "Speak",
        ActionPrimitive::Chirp { .. } => "Chirp",
    }
}

fn wall_or_cliff_veto(tick: &RuntimeTick) -> bool {
    tick.frame
        .now
        .extensions
        .get("motor_gate")
        .and_then(|value| value.get("safety_reason"))
        .and_then(|value| value.as_str())
        .map(|reason| reason == "cliff")
        .unwrap_or(false)
        || tick.frame.now.body.flags.wall
        || tick.frame.now.body.flags.cliff_left
        || tick.frame.now.body.flags.cliff_front_left
        || tick.frame.now.body.flags.cliff_front_right
        || tick.frame.now.body.flags.cliff_right
}

fn escape_progress_score(
    kind: ScenarioKind,
    distance_traveled_m: f32,
    distance_at_last_recovery_m: Option<f32>,
    collisions: usize,
    stuck_ticks: usize,
    ticks: usize,
) -> f32 {
    let progress = match kind {
        ScenarioKind::ColumnTrap | ScenarioKind::CornerTrap | ScenarioKind::ConcaveTrap => {
            distance_at_last_recovery_m
                .map(|distance| (distance_traveled_m - distance).max(0.0))
                .filter(|distance| *distance >= 0.08)
                .unwrap_or(distance_traveled_m)
        }
        _ => distance_traveled_m,
    };
    let collision_penalty = collisions as f32 * 0.05;
    let stuck_penalty = if ticks == 0 {
        0.0
    } else {
        stuck_ticks as f32 / ticks as f32 * 0.25
    };
    (progress - collision_penalty - stuck_penalty).max(0.0)
}

fn trap_kind_label(code: f32) -> Option<&'static str> {
    match code.round() as i32 {
        1 => Some("wall"),
        2 => Some("corner"),
        3 => Some("column"),
        _ => None,
    }
}

fn summarize_episode_memory(episodes: &[ScenarioEpisodeReport]) -> ScenarioMemorySummary {
    let memory_reports = episodes
        .iter()
        .filter_map(|episode| episode.memory.as_ref())
        .collect::<Vec<_>>();
    if memory_reports.is_empty() {
        return ScenarioMemorySummary {
            novelty_decay_sane: false,
            warnings: vec!["no episode memory reports".to_string()],
            ..ScenarioMemorySummary::default()
        };
    }
    let places_visited = memory_reports
        .iter()
        .map(|memory| memory.places_visited)
        .max()
        .unwrap_or(0);
    let mut warnings = Vec::new();
    if places_visited == 0 {
        warnings.push("memory observed zero places".to_string());
    }
    let novelty_decay_sane = memory_reports.iter().any(|memory| memory.novelty_decayed);
    if !novelty_decay_sane {
        warnings.push("novelty did not decay in any episode".to_string());
    }
    ScenarioMemorySummary {
        places_visited,
        mean_places_visited_per_episode: mean(
            memory_reports
                .iter()
                .map(|memory| memory.places_visited as f32),
        ),
        charge_memory_hit_rate: aggregate_hit_rate(
            memory_reports
                .iter()
                .map(|memory| (memory.charge_memory_ticks, memory.charge_opportunity_ticks)),
        ),
        danger_memory_hit_rate: aggregate_hit_rate(
            memory_reports
                .iter()
                .map(|memory| (memory.danger_memory_ticks, memory.danger_opportunity_ticks)),
        ),
        social_memory_hit_rate: aggregate_hit_rate(
            memory_reports
                .iter()
                .map(|memory| (memory.social_memory_ticks, memory.social_opportunity_ticks)),
        ),
        novelty_decay_sane,
        warnings,
    }
}

async fn run_dream_train(args: DreamTrainArgs) -> Result<()> {
    let checkpoint_dir = PathBuf::from(&args.checkpoint_dir);
    if args.clear && checkpoint_dir.exists() {
        fs::remove_dir_all(&checkpoint_dir).with_context(|| {
            format!(
                "failed to clear checkpoint dir {}",
                checkpoint_dir.display()
            )
        })?;
        println!(
            "cleared dream checkpoint dir for fresh evolve run: {}",
            checkpoint_dir.display()
        );
    }

    let evolve_best = checkpoint_dir.join("evolve-best.json");
    let incumbent = if !args.clear && evolve_best.exists() {
        Some(load_best_genome(&evolve_best).with_context(|| {
            format!(
                "failed to load incumbent evolve checkpoint {}",
                evolve_best.display()
            )
        })?)
    } else {
        None
    };

    let config = DreamTrainingConfig {
        population_size: args.population,
        generations: args.generations,
        base_seed: args.seed,
        start_level: args.start_level.into(),
        hidden_dim: args.hidden_dim,
        checkpoint_dir: checkpoint_dir.clone(),
        dataset_dir: PathBuf::from(args.dataset_dir),
        export_dataset: args.export_dataset,
        detailed_logs: args.detailed_logs,
    };
    let report = train_dream_policy(config).await?;

    let candidate = load_best_genome(&report.best_checkpoint).with_context(|| {
        format!(
            "failed to load candidate checkpoint {}",
            report.best_checkpoint.display()
        )
    })?;
    let promote = incumbent.as_ref().map_or(true, |current| {
        if candidate.level.id() != current.level.id() {
            candidate.level.id() > current.level.id()
        } else {
            candidate.best_score > current.best_score
        }
    });

    if promote {
        fs::copy(&report.best_checkpoint, &evolve_best).with_context(|| {
            format!(
                "failed to publish evolve checkpoint alias from {} to {}",
                report.best_checkpoint.display(),
                evolve_best.display()
            )
        })?;
        println!(
            "published evolve checkpoint alias: {}",
            evolve_best.display()
        );
    } else if let Some(current) = &incumbent {
        println!(
            "kept incumbent evolve checkpoint: {} (incumbent level={} score={:.3}, candidate level={} score={:.3})",
            evolve_best.display(),
            current.level.name(),
            current.best_score,
            candidate.level.name(),
            candidate.best_score,
        );
    }

    fn comma_count(value: u64) -> String {
        let digits = value.to_string();
        let mut out = String::with_capacity(digits.len() + (digits.len().saturating_sub(1) / 3));
        let mut since_comma = 0usize;
        for ch in digits.chars().rev() {
            if since_comma == 3 {
                out.push(',');
                since_comma = 0;
            }
            out.push(ch);
            since_comma += 1;
        }
        out.chars().rev().collect()
    }

    let unlocked = report
        .unlocked_levels
        .iter()
        .map(|level| level.name())
        .collect::<Vec<_>>()
        .join(" -> ");
    println!(
        "dream policy training complete: level {}, generation {}, best score {:.3}, genome {}, checkpoint {}, dataset {}, unlocked {}",
        report.status.current_level.name(),
        comma_count(report.status.generation as u64),
        report.status.best_score,
        comma_count(report.status.selected_genome_id),
        report.best_checkpoint.display(),
        report.dataset_dir.display(),
        unlocked,
    );
    if let Some(reason) = report.status.blocked_reason {
        println!("last safety block: {reason}");
    }
    Ok(())
}

fn scenario_recommendation(episodes: usize, summary: &ScenarioEvaluationSummary) -> String {
    if episodes < 3 {
        "insufficient_data".to_string()
    } else if summary.collision_rate > 0.10 || summary.mean_collisions_per_episode > 5.0 {
        "reject_or_continue_training".to_string()
    } else if summary.success_rate >= 0.80 && summary.collision_rate <= 0.02 {
        "candidate_for_more_eval".to_string()
    } else {
        "continue_training".to_string()
    }
}

fn nearest_object_distance<F>(
    position: (f32, f32),
    objects: &[pete_sim::SimObject],
    matches_kind: F,
) -> Option<f32>
where
    F: Fn(&pete_sim::SimObjectKind) -> bool,
{
    objects
        .iter()
        .filter(|object| matches_kind(&object.kind))
        .map(|object| {
            (distance_between(position, (object.x_m, object.y_m)) - object.radius_m).max(0.0)
        })
        .min_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
}

fn nearest_object_bearing<F>(
    position: (f32, f32),
    heading_rad: f32,
    objects: &[pete_sim::SimObject],
    matches_kind: F,
) -> Option<f32>
where
    F: Fn(&pete_sim::SimObjectKind) -> bool,
{
    objects
        .iter()
        .filter(|object| matches_kind(&object.kind))
        .min_by(|left, right| {
            let left_distance = distance_between(position, (left.x_m, left.y_m));
            let right_distance = distance_between(position, (right.x_m, right.y_m));
            left_distance
                .partial_cmp(&right_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|object| {
            let dx = object.x_m - position.0;
            let dy = object.y_m - position.1;
            (dy.atan2(dx) - heading_rad + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
                - std::f32::consts::PI
        })
}

fn distance_between(left: (f32, f32), right: (f32, f32)) -> f32 {
    let dx = left.0 - right.0;
    let dy = left.1 - right.1;
    ((dx * dx) + (dy * dy)).sqrt()
}

fn hit_rate(hits: usize, opportunities: usize) -> Option<f32> {
    (opportunities > 0).then_some(hits.min(opportunities) as f32 / opportunities as f32)
}

fn aggregate_hit_rate(pairs: impl Iterator<Item = (usize, usize)>) -> Option<f32> {
    let (hits, opportunities) = pairs.fold((0usize, 0usize), |acc, pair| {
        (acc.0.saturating_add(pair.0), acc.1.saturating_add(pair.1))
    });
    hit_rate(hits, opportunities)
}

fn sim_world_score(snapshot: &WorldSnapshot, index: usize) -> f32 {
    snapshot
        .extensions
        .iter()
        .find(|extension| extension.name == "sim.world")
        .and_then(|extension| extension.values.get(index).copied())
        .unwrap_or(0.0)
}

fn mean(values: impl Iterator<Item = f32>) -> f32 {
    let mut count = 0usize;
    let mut sum = 0.0;
    for value in values {
        count = count.saturating_add(1);
        sum += value;
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

fn mean_optional(values: impl Iterator<Item = f32>) -> Option<f32> {
    let mut count = 0usize;
    let mut sum = 0.0;
    for value in values {
        count = count.saturating_add(1);
        sum += value;
    }
    (count > 0).then_some(sum / count as f32)
}

