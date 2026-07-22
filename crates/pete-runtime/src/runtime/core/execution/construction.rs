impl<L, M, R, C, S, A> MinimalRuntime<L, M, R, C, S, A>
where
    L: LedgerWriter + Sync,
    M: MemoryStore,
    R: Recall + Sync,
    C: Conductor,
    S: SafetyLayer,
    A: LlmAgent + 'static,
{
    pub fn new(
        ledger: L,
        memory_store: M,
        memory_recall: R,
        conductor: C,
        safety: S,
        llm: A,
    ) -> Self {
        let cognition = RuntimeCognition::from_agent(&llm);
        Self {
            ledger,
            memory_store,
            memory_recall,
            conductor,
            safety,
            llm: Arc::new(tokio::sync::Mutex::new(llm)),
            extractor: EventExtractor::default(),
            bus: default_event_bus(),
            reign_queue: Arc::new(Mutex::new(ReignQueue::default())),
            predictor: StasisFuturePredictor,
            models: RuntimeModelStack::default(),
            action_selector_mode: ActionSelectorMode::Baseline,
            surprise_computer: BaselineSurpriseComputer,
            reward_computer: BaselineRewardComputer,
            transition_builder: TransitionBuilder::new(),
            behavior_training_hub: BehaviorTrainingHub::default(),
            surface_extractor: SurfaceExtractor::default(),
            inline_learning: InlineLearningConfig::default(),
            nudge_policy: NudgePolicy::default(),
            local_map: LocalMap::default(),
            last_behavior_runs: Vec::new(),
            locomotion_tracker: LocomotionTracker::default(),
            chirp_events: ChirpEventState::default(),
            nudge: NudgeController::default(),
            goal_system: GoalSystem::default(),
            world_model: WorldModelUpdater::default(),
            sleep_controller: SleepController::default(),
            semantic_outcomes: SemanticOutcomeTracker::default(),
            last_active_control: None,
            cognition,
            next_frame_id: None,
            pending_actuator_outcomes: Vec::new(),
        }
    }

    pub fn with_reign_queue(
        ledger: L,
        memory_store: M,
        memory_recall: R,
        conductor: C,
        safety: S,
        llm: A,
        reign_queue: Arc<Mutex<ReignQueue>>,
    ) -> Self {
        let cognition = RuntimeCognition::from_agent(&llm);
        Self {
            ledger,
            memory_store,
            memory_recall,
            conductor,
            safety,
            llm: Arc::new(tokio::sync::Mutex::new(llm)),
            extractor: EventExtractor::default(),
            bus: default_event_bus(),
            reign_queue,
            predictor: StasisFuturePredictor,
            models: RuntimeModelStack::default(),
            action_selector_mode: ActionSelectorMode::Baseline,
            surprise_computer: BaselineSurpriseComputer,
            reward_computer: BaselineRewardComputer,
            transition_builder: TransitionBuilder::new(),
            behavior_training_hub: BehaviorTrainingHub::default(),
            surface_extractor: SurfaceExtractor::default(),
            inline_learning: InlineLearningConfig::default(),
            nudge_policy: NudgePolicy::default(),
            local_map: LocalMap::default(),
            last_behavior_runs: Vec::new(),
            locomotion_tracker: LocomotionTracker::default(),
            chirp_events: ChirpEventState::default(),
            nudge: NudgeController::default(),
            goal_system: GoalSystem::default(),
            world_model: WorldModelUpdater::default(),
            sleep_controller: SleepController::default(),
            semantic_outcomes: SemanticOutcomeTracker::default(),
            last_active_control: None,
            cognition,
            next_frame_id: None,
            pending_actuator_outcomes: Vec::new(),
        }
    }

    pub fn with_default_events(
        ledger: L,
        memory_store: M,
        memory_recall: R,
        conductor: C,
        safety: S,
        llm: A,
    ) -> Self {
        Self::new(ledger, memory_store, memory_recall, conductor, safety, llm)
    }

    pub fn with_models(mut self, models: RuntimeModelStack) -> Self {
        self.models = models;
        self
    }

    pub fn with_action_selector_mode(mut self, mode: ActionSelectorMode) -> Self {
        self.action_selector_mode = mode;
        self
    }

    pub fn with_inline_learning(mut self, config: InlineLearningConfig) -> Self {
        self.inline_learning = config;
        self
    }

    pub fn with_nudge_policy(mut self, policy: NudgePolicy) -> Self {
        self.nudge_policy = policy;
        self
    }

    /// Bind the next production tick to an immutable input-frame identity.
    /// Live operation leaves this unset and continues to allocate random IDs;
    /// replay and shadow-flight callers set it once immediately before a tick.
    pub fn set_next_frame_id(&mut self, frame_id: Uuid) {
        self.next_frame_id = Some(frame_id);
    }

    pub fn with_local_map(mut self, local_map: LocalMap) -> Self {
        self.local_map = local_map;
        self
    }

    pub fn nudge_status(&self) -> NudgeStatus {
        self.nudge.status.clone()
    }

    /// Cancel optional cognition without disturbing local control state.
    pub fn cancel_cognition(&mut self) {
        if let Some(pending) = self.cognition.pending.take() {
            self.cognition.next_request_at_ms = pending
                .requested_at_ms
                .saturating_add(COGNITION_COOLDOWN_MS);
            pending.task.abort();
            self.cognition.last_outcome = Some(CognitionOutcome::Cancelled);
        }
    }

    pub fn behavior_node_states(&self) -> Vec<BehaviorNodeState> {
        self.models.behavior_node_states(&self.last_behavior_runs)
    }

}
