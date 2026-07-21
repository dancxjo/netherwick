pub struct MinimalRuntime<L, M, R, C, S, A>
where
    L: LedgerWriter + Sync,
    M: MemoryStore,
    R: Recall + Sync,
    C: Conductor,
    S: SafetyLayer,
    A: LlmAgent + 'static,
{
    pub ledger: L,
    pub memory_store: M,
    pub memory_recall: R,
    pub conductor: C,
    pub safety: S,
    /// Optional higher cognition is shared only with a background job.  The
    /// control tick never takes this mutex; it is exclusively an ownership
    /// device for the spawned provider future.
    pub llm: Arc<tokio::sync::Mutex<A>>,
    pub extractor: EventExtractor,
    pub bus: EventBus,
    pub reign_queue: Arc<Mutex<ReignQueue>>,
    pub predictor: StasisFuturePredictor,
    pub models: RuntimeModelStack,
    pub action_selector_mode: ActionSelectorMode,
    pub surprise_computer: BaselineSurpriseComputer,
    pub reward_computer: BaselineRewardComputer,
    pub transition_builder: TransitionBuilder,
    pub behavior_training_hub: BehaviorTrainingHub,
    pub surface_extractor: SurfaceExtractor,
    pub inline_learning: InlineLearningConfig,
    pub nudge_policy: NudgePolicy,
    pub local_map: LocalMap,
    pub last_behavior_runs: Vec<ErasedBehaviorRunRecord>,
    locomotion_tracker: LocomotionTracker,
    chirp_events: ChirpEventState,
    nudge: NudgeController,
    goal_system: GoalSystem,
    world_model: WorldModelUpdater,
    sleep_controller: SleepController,
    semantic_outcomes: SemanticOutcomeTracker,
    last_active_control: Option<ActiveControlSummary>,
    cognition: RuntimeCognition,
}

const COGNITION_DEADLINE_MS: u64 = 2_000;
/// Leave a quiet period after every terminal provider outcome so a fast or
/// disabled provider cannot turn the organism tick into a request generator.
const COGNITION_COOLDOWN_MS: u64 = 2_000;

struct RuntimeCognition {
    pending: Option<PendingLlmCognition>,
    next_request_at_ms: u64,
    last_sense: pete_now::LlmSense,
    last_sense_valid_until_ms: u64,
    last_outcome: Option<CognitionOutcome>,
    provider_declared_available: bool,
    provider_unavailable_reason: Option<String>,
}

impl RuntimeCognition {
    fn from_agent(agent: &impl LlmAgent) -> Self {
        Self {
            pending: None,
            next_request_at_ms: 0,
            last_sense: pete_now::LlmSense::default(),
            last_sense_valid_until_ms: 0,
            last_outcome: None,
            provider_declared_available: agent.enhanced_cognition_available(),
            provider_unavailable_reason: agent
                .enhanced_cognition_unavailable_reason()
                .map(str::to_string),
        }
    }
}

struct PendingLlmCognition {
    snapshot_ref: String,
    requested_at_ms: u64,
    deadline_ms: u64,
    task: JoinHandle<Result<(Option<Combobulation>, LlmTickResult)>>,
}

#[derive(Clone, Debug)]
enum CognitionOutcome {
    Accepted,
    Expired,
    Failed(String),
    Cancelled,
}

struct AcceptedLlmCognition {
    reflection: Option<Combobulation>,
    tick: LlmTickResult,
    snapshot_ref: String,
    requested_at_ms: u64,
    observed_at_ms: u64,
}

#[derive(Clone, Debug, Default)]
struct SemanticOutcomeTracker {
    previous: Option<SemanticActionState>,
    pending: Vec<SemanticEvidenceObservation>,
}

#[derive(Clone, Debug)]
struct SemanticActionState {
    t_ms: u64,
    behavior_id: String,
    target_id: Option<EntityId>,
    charger_distance_m: Option<f32>,
    clearance_m: Option<f32>,
    charging: bool,
}

impl SemanticOutcomeTracker {
    fn take_pending(&mut self) -> Vec<SemanticEvidenceObservation> {
        std::mem::take(&mut self.pending)
    }

    fn observe_outcome(&mut self, world: &WorldModelSnapshot) {
        let Some(previous) = self.previous.as_ref() else {
            return;
        };
        let mut observations = Vec::new();
        let current_charger_distance = previous
            .target_id
            .as_ref()
            .and_then(|target_id| canonical_entity_distance(world, target_id));
        let progress_evidence = |key: &str| EvidenceRef {
            id: format!(
                "semantic:action-outcome:{key}:{}:{}",
                previous.t_ms, world.t_ms
            ),
            source: "runtime.action_outcome".to_string(),
            key: key.to_string(),
            observed_at_ms: world.t_ms,
            transformation_lineage: vec!["pete_runtime::SemanticOutcomeTracker".to_string()],
            implementation_version: Some("2".to_string()),
        };
        if previous.behavior_id == "approach_charger"
            && previous
                .charger_distance_m
                .zip(current_charger_distance)
                .is_some_and(|(before, after)| after + 0.02 < before)
        {
            observations.push(SemanticEvidenceObservation::supported(
                SemanticNodeRef::Behavior(SemanticBehaviorId("approach_charger".to_string())),
                SemanticPredicate::Predicts,
                SemanticNodeRef::Outcome(SemanticOutcomeId(
                    "target_distance_decreases".to_string(),
                )),
                0.85,
                SemanticGroundingKind::ActionOutcome,
                progress_evidence("approach_reduced_charger_distance"),
            ));
        }
        if previous.behavior_id == "dock" && !previous.charging && world.self_model.charging {
            observations.push(SemanticEvidenceObservation::supported(
                SemanticNodeRef::Behavior(SemanticBehaviorId("dock".to_string())),
                SemanticPredicate::Predicts,
                SemanticNodeRef::Outcome(SemanticOutcomeId("charging_started".to_string())),
                0.95,
                SemanticGroundingKind::ActionOutcome,
                progress_evidence("dock_started_charging"),
            ));
        }
        if previous.behavior_id == "back_away"
            && previous
                .clearance_m
                .zip(
                    world
                        .local_geometry
                        .nearest_m
                        .as_ref()
                        .filter(|belief| belief.meta.freshness == Freshness::Current)
                        .map(|belief| belief.value),
                )
                .is_some_and(|(before, after)| after > before + 0.02)
        {
            observations.push(SemanticEvidenceObservation::supported(
                SemanticNodeRef::Behavior(SemanticBehaviorId("back_away".to_string())),
                SemanticPredicate::Predicts,
                SemanticNodeRef::Outcome(SemanticOutcomeId("clearance_increases".to_string())),
                0.85,
                SemanticGroundingKind::ActionOutcome,
                progress_evidence("back_away_increased_clearance"),
            ));
        }
        self.pending.extend(observations);
    }

    fn remember(
        &mut self,
        world: &WorldModelSnapshot,
        behavior: Option<&pete_conductor::BehaviorDecision>,
    ) {
        self.previous = behavior.map(|behavior| {
            let target_id = behavior.affordance.target.clone();
            SemanticActionState {
                t_ms: world.t_ms,
                behavior_id: behavior.behavior_id.clone(),
                charger_distance_m: target_id
                    .as_ref()
                    .and_then(|target_id| canonical_entity_distance(world, target_id)),
                target_id,
                clearance_m: world
                    .local_geometry
                    .nearest_m
                    .as_ref()
                    .filter(|belief| belief.meta.freshness == Freshness::Current)
                    .map(|belief| belief.value),
                charging: world.self_model.charging,
            }
        });
    }
}

fn canonical_entity_distance(world: &WorldModelSnapshot, target_id: &EntityId) -> Option<f32> {
    world
        .entities
        .get(target_id)
        .filter(|entity| entity.kind == pete_now::WorldEntityKind::Charger)
        .filter(|entity| {
            entity.distance_meta.as_ref().is_some_and(|meta| {
                !matches!(
                    meta.freshness,
                    Freshness::Stale | Freshness::Invalidated | Freshness::Missing
                )
            })
        })
        .and_then(|entity| entity.distance_m)
}

fn runtime_sleep_input(
    now: &Now,
    expected_external_power: bool,
    accelerator_available: bool,
) -> SleepTickInput {
    let flags = &now.body.flags;
    let safety_event = if flags.wheel_drop {
        Some("wheel_drop".to_string())
    } else if flags.cliff_left
        || flags.cliff_front_left
        || flags.cliff_front_right
        || flags.cliff_right
    {
        Some("cliff".to_string())
    } else if flags.bump_left || flags.bump_right {
        Some("contact".to_string())
    } else {
        None
    };
    let extension_bool = |key: &str| {
        now.extensions
            .get(key)
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    };
    let fatigue_activation = now
        .world
        .self_model
        .motivation
        .drives
        .get("rest")
        .map(|drive| drive.activation)
        .unwrap_or(now.drives.fatigue)
        .max(now.drives.fatigue);
    let direct_reign_active = now.reign.active
        && (now.reign.mode == Some(ReignMode::Direct)
            || now
                .reign
                .latest
                .as_ref()
                .is_some_and(|input| input.mode == ReignMode::Direct));
    let stopped =
        now.body.velocity.forward_m_s.abs() <= 0.01 && now.body.velocity.turn_rad_s.abs() <= 0.01;
    let body_communication_stable =
        now.body.last_update_ms == 0 || now.t_ms.saturating_sub(now.body.last_update_ms) <= 2_000;
    let critical_battery = now.body.battery_level <= 0.08;
    let unresolved_urgent_need = safety_event.is_some()
        || (now.body.battery_level <= 0.15 && !now.body.charging)
        || now.drives.danger_avoidance >= 0.80;
    let completed_episode_refs = now
        .world
        .temporal
        .recently_completed
        .iter()
        .map(|episode| episode.episode_id.0.clone())
        .collect::<Vec<_>>();
    let failed_behavior_refs = now
        .world
        .self_model
        .goal_status
        .iter()
        .filter(|(_, status)| status.failed_attempts > 0)
        .map(|(goal_id, status)| format!("goal:{goal_id}:failures:{}", status.failed_attempts))
        .collect::<Vec<_>>();
    let semantic_relation_refs = now
        .world
        .semantic
        .relations
        .keys()
        .take(128)
        .map(|relation_id| relation_id.0.clone())
        .collect::<Vec<_>>();
    SleepTickInput {
        now_ms: now.t_ms,
        fatigue_activation,
        charging: now.body.charging,
        docked: now.body.charging,
        stopped,
        direct_reign_active,
        unresolved_urgent_need,
        body_communication_stable,
        active_skill_interruptible: true,
        critical_battery,
        external_power_lost: expected_external_power && !now.body.charging,
        safety_event,
        important_social_cue: extension_bool("sleep.important_social_cue"),
        operator_sleep_request: extension_bool("sleep.request"),
        operator_wake_request: extension_bool("wake.request"),
        accelerator_available,
        thermal_fraction: now
            .extensions
            .get("body.thermal_fraction")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0) as f32,
        completed_episode_refs,
        failed_behavior_refs,
        semantic_relation_refs,
    }
}

#[derive(Clone, Debug, Default)]
struct ChirpEventState {
    last_charging: Option<bool>,
    last_awake: Option<bool>,
    last_object_count: usize,
    last_object_familiarity: f32,
    last_place_familiarity: f32,
    last_places_visited: u32,
    last_similar_situation_count: u16,
    last_surprise_high: bool,
    last_chosen_docking: bool,
}

impl ChirpEventState {
    fn emit_pre_selection_chirps(&mut self, now: &mut Now, notes: &mut Vec<String>) -> Result<()> {
        let object_count = now.objects.observations.len() + now.objects.vectors.len();
        if object_count > 0 && self.last_object_count == 0 {
            append_event_script_chirp(now, notes, "saw-something", ChirpPattern::SawSomething)?;
        }
        if now.memory.object_familiarity >= 0.70 && self.last_object_familiarity < 0.70 {
            append_event_script_chirp(
                now,
                notes,
                "object-recognized",
                ChirpPattern::ObjectRecognized,
            )?;
        }
        if now.memory.place_familiarity >= 0.70 && self.last_place_familiarity < 0.70 {
            append_event_script_chirp(
                now,
                notes,
                "place-recognized",
                ChirpPattern::PlaceRecognized,
            )?;
        }
        let learned = (self.last_places_visited > 0
            && now.memory.places_visited > self.last_places_visited)
            || (self.last_similar_situation_count > 0
                && now.memory.similar_situation_count > self.last_similar_situation_count);
        if learned {
            append_event_script_chirp(now, notes, "learned", ChirpPattern::Learned)?;
        }
        let surprise_high = now.surprise.total >= 0.70 || now.surprise.prediction_error >= 0.70;
        if surprise_high && !self.last_surprise_high {
            append_event_script_chirp(now, notes, "surprise", ChirpPattern::Surprise)?;
        }
        if matches!(self.last_charging, Some(false)) && now.body.charging {
            append_event_script_chirp(
                now,
                notes,
                "charging-started",
                ChirpPattern::ChargingStarted,
            )?;
        }
        let awake = now.drives.fatigue < 0.80;
        if matches!(self.last_awake, Some(true)) && !awake {
            append_event_script_chirp(now, notes, "sleep", ChirpPattern::Sleep)?;
        } else if matches!(self.last_awake, Some(false)) && awake {
            append_event_script_chirp(now, notes, "wake", ChirpPattern::Wake)?;
        }

        self.last_charging = Some(now.body.charging);
        self.last_awake = Some(awake);
        self.last_object_count = object_count;
        self.last_object_familiarity = now.memory.object_familiarity;
        self.last_place_familiarity = now.memory.place_familiarity;
        self.last_places_visited = now.memory.places_visited;
        self.last_similar_situation_count = now.memory.similar_situation_count;
        self.last_surprise_high = surprise_high;
        Ok(())
    }

    fn emit_post_selection_chirps(
        &mut self,
        now: &mut Now,
        notes: &mut Vec<String>,
        chosen_action: &ActionPrimitive,
    ) -> Result<()> {
        let docking = matches!(chosen_action, ActionPrimitive::Dock);
        if docking && !self.last_chosen_docking {
            let pattern = if charger_visible(now) {
                ChirpPattern::GoalAcquired
            } else {
                ChirpPattern::Docking
            };
            append_event_script_chirp(now, notes, "docking", pattern)?;
        }
        self.last_chosen_docking = docking;
        Ok(())
    }
}

