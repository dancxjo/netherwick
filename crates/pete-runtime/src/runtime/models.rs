#[derive(Default)]
pub struct RuntimeModelStack {
    pub behaviors: BehaviorRegistry,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BumpEventInput {
    pub t_ms: TimeMs,
    pub bump_left: bool,
    pub bump_right: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RobotInitializedEventInput {
    pub t_ms: TimeMs,
    pub mode: String,
    pub body: String,
    pub battery_percent: Option<u32>,
    pub charging: Option<bool>,
    pub active_sensors: usize,
    pub requested_sensors: usize,
    pub ledger: String,
    pub tick_ms: u64,
    pub dashboard: Option<String>,
    pub capture: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventScriptAction {
    Say { text: String },
    Chirp { pattern: ChirpPattern },
    Song { name: String },
    Stop,
    Rotate { deg: i16 },
    Go,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EventScriptOutput {
    pub actions: Vec<EventScriptAction>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SafeScriptAction {
    pub requested: EventScriptAction,
    pub action: Option<ActionPrimitive>,
    pub desired_motor: MotorCommand,
    pub final_motor: MotorCommand,
    pub vetoed: bool,
    pub safety_reason: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SafeScriptSequence {
    pub actions: Vec<SafeScriptAction>,
}

impl RuntimeModelStack {
    pub fn with_danger_shadow_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_danger_metadata(path)?;
        let mut stack = Self::default();
        stack.behaviors.danger = danger_behavior(
            BehaviorRegime::ShadowInfer,
            Some(DangerNetTrainer::load_checkpoint(path, metadata.input_dim)?),
            FallbackPolicy::UseHardcoded,
        );
        Ok(stack)
    }

    pub fn with_charge_shadow_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_charge_metadata(path)?;
        let mut stack = Self::default();
        stack.behaviors.charge = charge_behavior(
            BehaviorRegime::ShadowInfer,
            Some(ChargeNetTrainer::load_checkpoint(path, metadata.input_dim)?),
            FallbackPolicy::UseHardcoded,
        );
        Ok(stack)
    }

    pub fn with_action_value_shadow_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_action_value_metadata(path)?;
        let mut stack = Self::default();
        stack.behaviors.action_value = action_value_behavior(
            BehaviorRegime::ShadowInfer,
            Some(ActionValueNetTrainer::load_checkpoint(
                path,
                metadata.input_dim,
            )?),
            FallbackPolicy::UseHardcoded,
        );
        Ok(stack)
    }

    pub fn with_eye_next_shadow_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_eye_next_metadata(path)?;
        let mut stack = Self::default();
        stack.behaviors.eye_next = eye_next_behavior(
            BehaviorRegime::ShadowInfer,
            Some(EyeNextNetTrainer::load_checkpoint(
                path,
                metadata.input_dim,
            )?),
            FallbackPolicy::UseHardcoded,
        );
        Ok(stack)
    }

    pub fn with_ear_next_shadow_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_ear_next_metadata(path)?;
        let mut stack = Self::default();
        stack.behaviors.ear_next = ear_next_behavior(
            BehaviorRegime::ShadowInfer,
            Some(EarNextNetTrainer::load_checkpoint(
                path,
                metadata.input_dim,
            )?),
            FallbackPolicy::UseHardcoded,
        );
        Ok(stack)
    }

    pub fn with_experience_shadow_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_experience_autoencoder_metadata(path)?;
        let mut stack = Self::default();
        stack.behaviors.experience = experience_behavior(
            BehaviorRegime::ShadowInfer,
            Some(ExperienceAutoencoderTrainer::load_checkpoint(
                path,
                metadata.input_dim,
            )?),
            FallbackPolicy::UseHardcoded,
        );
        Ok(stack)
    }

    pub fn with_future_shadow_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        Self::with_future_checkpoint(path, BehaviorRegime::ShadowInfer)
    }

    pub fn with_future_checkpoint(path: impl AsRef<Path>, mode: BehaviorRegime) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_future_metadata(path)?;
        let mut stack = Self::default();
        stack.behaviors.future = future_behavior(
            mode,
            Some(FutureNetTrainer::load_checkpoint(
                path,
                metadata.input_dim,
                metadata.latent_dim,
            )?),
            FallbackPolicy::UseHardcoded,
        );
        Ok(stack)
    }

    pub fn with_shadow_checkpoints(
        danger_path: Option<&Path>,
        charge_path: Option<&Path>,
        action_value_path: Option<&Path>,
        future_path: Option<&Path>,
        eye_next_path: Option<&Path>,
        ear_next_path: Option<&Path>,
        experience_path: Option<&Path>,
    ) -> Result<Self> {
        let mut stack = Self::default();
        if let Some(path) = danger_path {
            let metadata = read_danger_metadata(path)?;
            stack.behaviors.danger = danger_behavior(
                BehaviorRegime::ShadowInfer,
                Some(DangerNetTrainer::load_checkpoint(path, metadata.input_dim)?),
                FallbackPolicy::UseHardcoded,
            );
        }
        if let Some(path) = charge_path {
            let metadata = read_charge_metadata(path)?;
            stack.behaviors.charge = charge_behavior(
                BehaviorRegime::ShadowInfer,
                Some(ChargeNetTrainer::load_checkpoint(path, metadata.input_dim)?),
                FallbackPolicy::UseHardcoded,
            );
        }
        if let Some(path) = action_value_path {
            let metadata = read_action_value_metadata(path)?;
            stack.behaviors.action_value = action_value_behavior(
                BehaviorRegime::ShadowInfer,
                Some(ActionValueNetTrainer::load_checkpoint(
                    path,
                    metadata.input_dim,
                )?),
                FallbackPolicy::UseHardcoded,
            );
        }
        if let Some(path) = future_path {
            let metadata = read_future_metadata(path)?;
            stack.behaviors.future = future_behavior(
                BehaviorRegime::ShadowInfer,
                Some(FutureNetTrainer::load_checkpoint(
                    path,
                    metadata.input_dim,
                    metadata.latent_dim,
                )?),
                FallbackPolicy::UseHardcoded,
            );
        }
        if let Some(path) = eye_next_path {
            let metadata = read_eye_next_metadata(path)?;
            stack.behaviors.eye_next = eye_next_behavior(
                BehaviorRegime::ShadowInfer,
                Some(EyeNextNetTrainer::load_checkpoint(
                    path,
                    metadata.input_dim,
                )?),
                FallbackPolicy::UseHardcoded,
            );
        }
        if let Some(path) = ear_next_path {
            let metadata = read_ear_next_metadata(path)?;
            stack.behaviors.ear_next = ear_next_behavior(
                BehaviorRegime::ShadowInfer,
                Some(EarNextNetTrainer::load_checkpoint(
                    path,
                    metadata.input_dim,
                )?),
                FallbackPolicy::UseHardcoded,
            );
        }
        if let Some(path) = experience_path {
            let metadata = read_experience_autoencoder_metadata(path)?;
            stack.behaviors.experience = experience_behavior(
                BehaviorRegime::ShadowInfer,
                Some(ExperienceAutoencoderTrainer::load_checkpoint(
                    path,
                    metadata.input_dim,
                )?),
                FallbackPolicy::UseHardcoded,
            );
        }
        Ok(stack)
    }

    pub fn from_models_config(path: impl AsRef<Path>) -> Result<Self> {
        let config: BehaviorRegistryConfig = toml::from_str(&std::fs::read_to_string(path)?)?;
        Self::from_behavior_config(&config)
    }

    pub fn from_behavior_config(config: &BehaviorRegistryConfig) -> Result<Self> {
        let mut stack = Self::default();
        if let Some(behavior) = config.behavior.get("locomotion") {
            stack.behaviors.locomotion = locomotion_behavior(
                behavior.regime,
                load_locomotion_behavior(behavior)?,
                behavior.fallback,
            );
        }
        if let Some(behavior) = config.behavior.get("danger") {
            stack.behaviors.danger = danger_behavior(
                behavior.regime,
                load_danger_behavior_trainer(behavior)?,
                behavior.fallback,
            );
        }
        if let Some(behavior) = config.behavior.get("charge") {
            stack.behaviors.charge = charge_behavior(
                behavior.regime,
                load_charge_behavior_trainer(behavior)?,
                behavior.fallback,
            );
        }
        if let Some(behavior) = config.behavior.get("future") {
            stack.behaviors.future = future_behavior(
                behavior.regime,
                load_future_behavior_trainer(behavior)?,
                behavior.fallback,
            );
        }
        if let Some(behavior) = config.behavior.get("action_value") {
            stack.behaviors.action_value = action_value_behavior(
                behavior.regime,
                load_action_value_behavior_trainer(behavior)?,
                behavior.fallback,
            );
        }
        if let Some(behavior) = config.behavior.get("conductor") {
            stack.behaviors.conductor = conductor_behavior(
                behavior.regime,
                &behavior.hardcoded,
                behavior.model.clone(),
                behavior.fallback,
            );
        }
        if let Some(behavior) = config.behavior.get("eye_next") {
            stack.behaviors.eye_next = eye_next_behavior(
                behavior.regime,
                load_eye_next_behavior_trainer(behavior)?,
                behavior.fallback,
            );
        }
        if let Some(behavior) = config.behavior.get("ear_next") {
            stack.behaviors.ear_next = ear_next_behavior(
                behavior.regime,
                load_ear_next_behavior_trainer(behavior)?,
                behavior.fallback,
            );
        }
        if let Some(behavior) = config.behavior.get("experience") {
            stack.behaviors.experience = experience_behavior(
                behavior.regime,
                load_experience_behavior_trainer(behavior)?,
                behavior.fallback,
            );
        }
        if let Some(behavior) = config.behavior.get("event_bump") {
            stack.behaviors.event_bump =
                bump_event_behavior(behavior.regime, behavior.model.clone(), behavior.fallback);
        }
        if let Some(behavior) = config.behavior.get("event_robot_initialized") {
            stack.behaviors.event_robot_initialized = robot_initialized_event_behavior(
                behavior.regime,
                behavior.model.clone(),
                behavior.fallback,
            );
        }
        Ok(stack)
    }

    pub fn behavior_node_states(
        &self,
        last_runs: &[ErasedBehaviorRunRecord],
    ) -> Vec<BehaviorNodeState> {
        let last = |id: &str| {
            last_runs
                .iter()
                .rev()
                .find(|run| run.behavior_id == id)
                .cloned()
        };
        vec![
            behavior_node_state(
                "Locomotion",
                "locomotion",
                "Locomotion",
                self.behaviors.locomotion.regime,
                self.behaviors.locomotion.hardcoded_id(),
                self.behaviors.locomotion.model_id(),
                self.behaviors.locomotion.fallback,
                vec![impl_id(
                    "locomotion.hardcoded_wander.v0",
                    "Hardcoded wander/reflex",
                )],
                vec![impl_id("locomotion.neat.v0", "NEAT locomotion v0")],
                last("locomotion"),
            ),
            behavior_node_state(
                "Experience",
                "experience",
                "Experience",
                self.behaviors.experience.regime,
                self.behaviors.experience.hardcoded_id(),
                self.behaviors.experience.model_id(),
                self.behaviors.experience.fallback,
                vec![impl_id("experience.no_latent_yet", "No latent yet")],
                vec![impl_id("experience.autoencoder.v0", "Autoencoder v0")],
                last("experience"),
            ),
            behavior_node_state(
                "Danger",
                "danger",
                "Danger",
                self.behaviors.danger.regime,
                self.behaviors.danger.hardcoded_id(),
                self.behaviors.danger.model_id(),
                self.behaviors.danger.fallback,
                vec![impl_id("danger.range_bumper", "Range/bumper")],
                vec![impl_id("danger.burn.v0", "Burn v0")],
                last("danger"),
            ),
            behavior_node_state(
                "Charge",
                "charge",
                "Charge",
                self.behaviors.charge.regime,
                self.behaviors.charge.hardcoded_id(),
                self.behaviors.charge.model_id(),
                self.behaviors.charge.fallback,
                vec![impl_id(
                    "charge.sensor_battery_delta",
                    "Sensor/battery delta",
                )],
                vec![impl_id("charge.burn.v0", "Burn v0")],
                last("charge"),
            ),
            behavior_node_state(
                "Future",
                "future",
                "Future",
                self.behaviors.future.regime,
                self.behaviors.future.hardcoded_id(),
                self.behaviors.future.model_id(),
                self.behaviors.future.fallback,
                vec![impl_id("future.stasis", "Stasis")],
                vec![impl_id("future.burn.v0", "Burn v0")],
                last("future"),
            ),
            behavior_node_state(
                "Conductor",
                "conductor",
                "Conductor",
                self.behaviors.conductor.regime,
                self.behaviors.conductor.hardcoded_id(),
                self.behaviors.conductor.model_id(),
                self.behaviors.conductor.fallback,
                vec![
                    impl_id("conductor.simple_v0", "Simple conductor"),
                    impl_id("action_selector.baseline", "Baseline selector"),
                    impl_id("reign.teacher", "Reign teacher"),
                ],
                vec![
                    impl_id("conductor.burn.v0", "Conductor Burn v0"),
                    impl_id("action_selector.burn.v0", "Action selector Burn v0"),
                ],
                last("conductor"),
            ),
            behavior_node_state(
                "ActionValue",
                "action_value",
                "ActionValue",
                self.behaviors.action_value.regime,
                self.behaviors.action_value.hardcoded_id(),
                self.behaviors.action_value.model_id(),
                self.behaviors.action_value.fallback,
                vec![impl_id("action_value.handcoded", "Handcoded value")],
                vec![impl_id("action_value.burn.v0", "Burn v0")],
                last("action_value"),
            ),
            behavior_node_state(
                "EyeNext",
                "eye_next",
                "EyeNext",
                self.behaviors.eye_next.regime,
                self.behaviors.eye_next.hardcoded_id(),
                self.behaviors.eye_next.model_id(),
                self.behaviors.eye_next.fallback,
                vec![impl_id("eye.copy_current", "Copy current")],
                vec![impl_id("eye.burn.next_v0", "Burn next v0")],
                last("eye_next"),
            ),
            behavior_node_state(
                "EarNext",
                "ear_next",
                "EarNext",
                self.behaviors.ear_next.regime,
                self.behaviors.ear_next.hardcoded_id(),
                self.behaviors.ear_next.model_id(),
                self.behaviors.ear_next.fallback,
                vec![impl_id("ear.copy_current", "Copy current")],
                vec![impl_id("ear.burn.next_v0", "Burn next v0")],
                last("ear_next"),
            ),
            behavior_node_state(
                "EventRobotInitialized",
                "event_robot_initialized",
                "on(robot-initialized)",
                self.behaviors.event_robot_initialized.regime,
                self.behaviors.event_robot_initialized.hardcoded_id(),
                self.behaviors.event_robot_initialized.model_id(),
                self.behaviors.event_robot_initialized.fallback,
                vec![impl_id(
                    "script.on_robot_initialized.ts.v0",
                    "TypeScript script teacher",
                )],
                vec![impl_id("event.robot_initialized.shadow.v0", "Shadow model")],
                last("event_robot_initialized"),
            ),
            behavior_node_state(
                "EventBump",
                "event_bump",
                "on(bump)",
                self.behaviors.event_bump.regime,
                self.behaviors.event_bump.hardcoded_id(),
                self.behaviors.event_bump.model_id(),
                self.behaviors.event_bump.fallback,
                vec![impl_id("script.on_bump.ts.v0", "TypeScript script teacher")],
                vec![impl_id("event.bump.shadow.v0", "Shadow model")],
                last("event_bump"),
            ),
        ]
    }

    pub fn apply_behavior_node_update(&mut self, node_id: &str, update: &BehaviorNodeUpdate) {
        let id = normalize_behavior_node_id(node_id);
        if id == "conductor" {
            let regime = effective_training_regime(
                update
                    .selected_regime
                    .unwrap_or(self.behaviors.conductor.regime),
                update.training_enabled,
            );
            let hardcoded = update
                .selected_hardcoded
                .as_deref()
                .unwrap_or_else(|| self.behaviors.conductor.hardcoded_id());
            let model = update
                .selected_model
                .clone()
                .or_else(|| self.behaviors.conductor.model_id().map(str::to_string));
            let fallback = update
                .fallback_policy
                .unwrap_or(self.behaviors.conductor.fallback);
            self.behaviors.conductor = conductor_behavior(regime, hardcoded, model, fallback);
            return;
        }
        macro_rules! update_behavior {
            ($field:ident) => {{
                if let Some(regime) = update.selected_regime {
                    self.behaviors.$field.regime =
                        effective_training_regime(regime, update.training_enabled);
                }
                if let Some(fallback) = update.fallback_policy {
                    self.behaviors.$field.fallback = fallback;
                }
            }};
        }
        match id.as_str() {
            "locomotion" => update_behavior!(locomotion),
            "experience" => update_behavior!(experience),
            "danger" => update_behavior!(danger),
            "charge" => update_behavior!(charge),
            "future" => update_behavior!(future),
            "action_value" => update_behavior!(action_value),
            "eye_next" => update_behavior!(eye_next),
            "ear_next" => update_behavior!(ear_next),
            "event_robot_initialized" => update_behavior!(event_robot_initialized),
            "event_bump" => update_behavior!(event_bump),
            _ => {}
        }
    }
}

fn impl_id(id: &str, label: &str) -> BehaviorImplementation {
    BehaviorImplementation {
        id: id.to_string(),
        label: label.to_string(),
    }
}

fn behavior_node_state(
    node_id: &str,
    behavior_id: &str,
    label: &str,
    regime: BehaviorRegime,
    hardcoded_id: &str,
    model_id: Option<&str>,
    fallback: FallbackPolicy,
    hardcoded_implementations: Vec<BehaviorImplementation>,
    model_implementations: Vec<BehaviorImplementation>,
    last_run: Option<ErasedBehaviorRunRecord>,
) -> BehaviorNodeState {
    let training_enabled = matches!(
        regime,
        BehaviorRegime::ShadowTrain | BehaviorRegime::ModelTrainAndInfer
    );
    BehaviorNodeState {
        node_id: node_id.to_string(),
        behavior_id: behavior_id.to_string(),
        label: label.to_string(),
        allowed_regimes: vec![
            BehaviorRegime::Hardcoded,
            BehaviorRegime::ShadowTrain,
            BehaviorRegime::ShadowInfer,
            BehaviorRegime::ModelInfer,
            BehaviorRegime::ModelTrainAndInfer,
            BehaviorRegime::Compare,
        ],
        hardcoded_implementations,
        model_implementations,
        selected_regime: regime,
        selected_hardcoded: hardcoded_id.to_string(),
        selected_model: model_id.map(str::to_string),
        checkpoint_path: None,
        fallback_policy: fallback,
        training_enabled,
        last_run,
        samples_observed: 0,
        train_steps_used: 0,
        missing_model_or_checkpoint: model_id.is_none()
            && !matches!(regime, BehaviorRegime::Hardcoded),
    }
}

fn normalize_behavior_node_id(node_id: &str) -> String {
    match node_id {
        "ActionValue" => "action_value".to_string(),
        "EyeNext" => "eye_next".to_string(),
        "EarNext" => "ear_next".to_string(),
        "EventRobotInitialized" => "event_robot_initialized".to_string(),
        "EventBump" => "event_bump".to_string(),
        other => other.to_ascii_lowercase().replace('-', "_"),
    }
}

fn effective_training_regime(
    regime: BehaviorRegime,
    training_enabled: Option<bool>,
) -> BehaviorRegime {
    if training_enabled.unwrap_or(true) {
        return regime;
    }
    match regime {
        BehaviorRegime::ShadowTrain => BehaviorRegime::ShadowInfer,
        BehaviorRegime::ModelTrainAndInfer => BehaviorRegime::ModelInfer,
        other => other,
    }
}

pub struct BehaviorRegistry {
    pub locomotion: ReplaceableBehavior<LocomotionInput, LocomotionOutput>,
    pub experience: ReplaceableBehavior<ExperienceBehaviorInput, ExperienceBehaviorOutput>,
    pub danger: ReplaceableBehavior<SituatedDangerInput, DangerOutput>,
    pub charge: ReplaceableBehavior<SituatedChargeInput, ChargeOutput>,
    pub future: ReplaceableBehavior<FutureInput, FuturePrediction>,
    pub action_value: ReplaceableBehavior<SituatedActionValueInput, ActionValueOutput>,
    pub conductor: ReplaceableBehavior<ConductorInput, ActionPrimitive>,
    pub eye_next: ReplaceableBehavior<SituatedEyeNextInput, EyeNextOutput>,
    pub ear_next: ReplaceableBehavior<SituatedEarNextInput, EarNextOutput>,
    pub event_robot_initialized: ReplaceableBehavior<RobotInitializedEventInput, EventScriptOutput>,
    pub event_bump: ReplaceableBehavior<BumpEventInput, EventScriptOutput>,
}

impl Default for BehaviorRegistry {
    fn default() -> Self {
        Self {
            locomotion: locomotion_behavior(
                BehaviorRegime::Hardcoded,
                None,
                FallbackPolicy::UseHardcoded,
            ),
            experience: experience_behavior(
                BehaviorRegime::Hardcoded,
                None,
                FallbackPolicy::UseHardcoded,
            ),
            danger: danger_behavior(
                BehaviorRegime::Hardcoded,
                None,
                FallbackPolicy::UseHardcoded,
            ),
            charge: charge_behavior(
                BehaviorRegime::Hardcoded,
                None,
                FallbackPolicy::UseHardcoded,
            ),
            future: future_behavior(
                BehaviorRegime::Hardcoded,
                None,
                FallbackPolicy::UseHardcoded,
            ),
            action_value: action_value_behavior(
                BehaviorRegime::Hardcoded,
                None,
                FallbackPolicy::UseHardcoded,
            ),
            conductor: conductor_behavior(
                BehaviorRegime::Hardcoded,
                "conductor.simple_v0",
                Some("conductor.burn.v0".to_string()),
                FallbackPolicy::StopSafely,
            ),
            eye_next: eye_next_behavior(
                BehaviorRegime::Hardcoded,
                None,
                FallbackPolicy::UseHardcoded,
            ),
            ear_next: ear_next_behavior(
                BehaviorRegime::Hardcoded,
                None,
                FallbackPolicy::UseHardcoded,
            ),
            event_robot_initialized: robot_initialized_event_behavior(
                BehaviorRegime::ShadowTrain,
                Some("event.robot_initialized.shadow.v0".to_string()),
                FallbackPolicy::UseHardcoded,
            ),
            event_bump: bump_event_behavior(
                BehaviorRegime::ShadowTrain,
                Some("event.bump.shadow.v0".to_string()),
                FallbackPolicy::UseHardcoded,
            ),
        }
    }
}

pub struct BehaviorTrainingHub {
    pub danger_extractor: Box<dyn TargetExtractor<ExperienceTransition, DangerInput, DangerOutput>>,
    pub charge_extractor: Box<dyn TargetExtractor<ExperienceTransition, ChargeInput, ChargeOutput>>,
    pub future_extractor:
        Box<dyn TargetExtractor<ExperienceTransition, FutureInput, FuturePrediction>>,
    pub action_value_extractor:
        Box<dyn TargetExtractor<ExperienceTransition, ActionValueInput, ActionValueOutput>>,
    pub eye_next_extractor:
        Box<dyn TargetExtractor<ExperienceTransition, EyeNextInput, EyeNextOutput>>,
    pub ear_next_extractor:
        Box<dyn TargetExtractor<ExperienceTransition, EarNextInput, EarNextOutput>>,
}

impl Default for BehaviorTrainingHub {
    fn default() -> Self {
        Self {
            danger_extractor: Box::new(DangerTargetExtractor),
            charge_extractor: Box::new(ChargeTargetExtractor),
            future_extractor: Box::new(FutureTargetExtractor { offset_ms: 1_000 }),
            action_value_extractor: Box::new(ActionValueTargetExtractor),
            eye_next_extractor: Box::new(EyeNextTargetExtractor { offset_ms: 100 }),
            ear_next_extractor: Box::new(EarNextTargetExtractor { offset_ms: 100 }),
        }
    }
}

pub struct DangerTargetExtractor;

impl TargetExtractor<ExperienceTransition, DangerInput, DangerOutput> for DangerTargetExtractor {
    fn extract(
        &self,
        transition: &ExperienceTransition,
    ) -> Result<Option<TrainingSample<DangerInput, DangerOutput>>> {
        let input = danger_input_from_transition_like(
            &transition.before_z,
            transition.action.as_ref(),
            &transition.before,
        );
        let target = danger_target_from_transition_like(
            &transition.before,
            transition.action.as_ref(),
            &transition.after,
        );
        Ok(Some(TrainingSample {
            input,
            expected: DangerOutput {
                bump_risk: target.bump,
                cliff_risk: target.cliff,
                wheel_drop_risk: target.wheel_drop,
                stuck_risk: target.stuck,
                confidence: 1.0,
            },
            actual: None,
            reward: Some(transition.reward.value),
            weight: 1.0,
            source: TrainingSource::WorldOutcome,
            t_ms: transition.created_at_ms,
        }))
    }
}

pub struct ChargeTargetExtractor;

impl TargetExtractor<ExperienceTransition, ChargeInput, ChargeOutput> for ChargeTargetExtractor {
    fn extract(
        &self,
        transition: &ExperienceTransition,
    ) -> Result<Option<TrainingSample<ChargeInput, ChargeOutput>>> {
        let input = charge_input_from_transition_like(
            &transition.before_z,
            transition.action.as_ref(),
            &transition.before,
        );
        let target = charge_target_from_transition_like(
            &transition.before,
            transition.action.as_ref(),
            &transition.after,
        );
        Ok(Some(TrainingSample {
            input,
            expected: ChargeOutput {
                charge_probability: target.charging_started,
                expected_battery_delta: target.battery_delta,
                dock_likelihood: target.charging_after,
                confidence: 1.0,
            },
            actual: None,
            reward: Some(transition.reward.value),
            weight: 1.0,
            source: TrainingSource::WorldOutcome,
            t_ms: transition.created_at_ms,
        }))
    }
}

pub struct FutureTargetExtractor {
    pub offset_ms: TimeMs,
}

impl TargetExtractor<ExperienceTransition, FutureInput, FuturePrediction>
    for FutureTargetExtractor
{
    fn extract(
        &self,
        transition: &ExperienceTransition,
    ) -> Result<Option<TrainingSample<FutureInput, FuturePrediction>>> {
        let action = match transition.action.clone() {
            Some(action) => action,
            None => return Ok(None),
        };
        Ok(Some(TrainingSample {
            input: FutureInput {
                latent: transition.before_z.clone(),
                action,
                offset_ms: self.offset_ms,
            },
            expected: FuturePrediction {
                offset_ms: self.offset_ms,
                predicted_z: transition.after_z.z.clone(),
                confidence: transition.after_z.confidence,
                summary: Some("Observed next latent state.".to_string()),
            },
            actual: None,
            reward: Some(transition.reward.value),
            weight: 1.0,
            source: TrainingSource::WorldOutcome,
            t_ms: transition.created_at_ms,
        }))
    }
}

pub struct ActionValueTargetExtractor;

impl TargetExtractor<ExperienceTransition, ActionValueInput, ActionValueOutput>
    for ActionValueTargetExtractor
{
    fn extract(
        &self,
        transition: &ExperienceTransition,
    ) -> Result<Option<TrainingSample<ActionValueInput, ActionValueOutput>>> {
        let target =
            action_value_target_from_reward_surprise(&transition.reward, &transition.surprise);
        Ok(Some(TrainingSample {
            input: action_value_input_from_transition_like(
                &transition.before_z,
                transition.action.as_ref(),
                &transition.before,
            ),
            expected: ActionValueOutput {
                value: target.value.clamp(-1.0, 1.0),
                confidence: 1.0,
            },
            actual: None,
            reward: Some(transition.reward.value),
            weight: 1.0,
            source: TrainingSource::WorldOutcome,
            t_ms: transition.created_at_ms,
        }))
    }
}

pub struct EyeNextTargetExtractor {
    pub offset_ms: TimeMs,
}

impl TargetExtractor<ExperienceTransition, EyeNextInput, EyeNextOutput> for EyeNextTargetExtractor {
    fn extract(
        &self,
        transition: &ExperienceTransition,
    ) -> Result<Option<TrainingSample<EyeNextInput, EyeNextOutput>>> {
        let Some(target) = eye_next_target_from_now(&transition.after) else {
            return Ok(None);
        };
        Ok(Some(TrainingSample {
            input: eye_next_input_from_transition_like(
                &transition.before_z,
                transition.action.as_ref(),
                &transition.before,
                self.offset_ms,
            ),
            expected: EyeNextOutput {
                width: target.width,
                height: target.height,
                rgb: target.rgb,
                confidence: 1.0,
            },
            actual: None,
            reward: Some(transition.reward.value),
            weight: 1.0,
            source: TrainingSource::WorldOutcome,
            t_ms: transition.created_at_ms,
        }))
    }
}

pub struct EarNextTargetExtractor {
    pub offset_ms: TimeMs,
}

impl TargetExtractor<ExperienceTransition, EarNextInput, EarNextOutput> for EarNextTargetExtractor {
    fn extract(
        &self,
        transition: &ExperienceTransition,
    ) -> Result<Option<TrainingSample<EarNextInput, EarNextOutput>>> {
        let Some(target) = ear_next_target_from_now(&transition.after) else {
            return Ok(None);
        };
        Ok(Some(TrainingSample {
            input: ear_next_input_from_transition_like(
                &transition.before_z,
                transition.action.as_ref(),
                &transition.before,
                self.offset_ms,
            ),
            expected: EarNextOutput {
                sample_rate_hz: target.sample_rate_hz,
                channels: target.channels,
                pcm: target.pcm,
                features: target.features,
                confidence: 1.0,
            },
            actual: None,
            reward: Some(transition.reward.value),
            weight: 1.0,
            source: TrainingSource::WorldOutcome,
            t_ms: transition.created_at_ms,
        }))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SituatedDangerInput {
    pub input: DangerInput,
    pub now: Now,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SituatedChargeInput {
    pub input: ChargeInput,
    pub now: Now,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SituatedActionValueInput {
    pub input: ActionValueInput,
    pub now: Now,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SituatedEyeNextInput {
    pub input: EyeNextInput,
    pub now: Now,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SituatedEarNextInput {
    pub input: EarNextInput,
    pub now: Now,
}

struct HardcodedExperienceBehavior;

impl FunctionBehavior<ExperienceBehaviorInput, ExperienceBehaviorOutput>
    for HardcodedExperienceBehavior
{
    fn id(&self) -> &'static str {
        "experience.no_latent_yet"
    }

    fn infer(&mut self, input: &ExperienceBehaviorInput) -> Result<ExperienceBehaviorOutput> {
        Ok(ExperienceBehaviorOutput {
            latent: ExperienceLatent {
                t_ms: input.now.t_ms,
                z: Vec::new(),
                reconstruction_error: 0.0,
                prediction_error: 0.0,
                confidence: 0.0,
            },
            reconstruction: None,
            reconstruction_loss: None,
            confidence: 0.0,
        })
    }
}

struct LearnedExperienceBehavior {
    model: ExperienceAutoencoderTrainer,
}

impl FunctionBehavior<ExperienceBehaviorInput, ExperienceBehaviorOutput>
    for LearnedExperienceBehavior
{
    fn id(&self) -> &'static str {
        "experience.autoencoder.v0"
    }

    fn infer(&mut self, input: &ExperienceBehaviorInput) -> Result<ExperienceBehaviorOutput> {
        let encode_input = ExperienceEncodeInput {
            sense_vectors: input.sense_vectors.clone(),
        };
        let prediction = self.model.predict(&encode_input)?;
        let target = experience_decode_target_from_now(&input.now);
        let reconstruction_loss = experience_reconstruction_loss_flat(&prediction.decoded, &target);
        let latent = ExperienceLatent {
            t_ms: input.now.t_ms,
            z: prediction.encoded.z.clone(),
            reconstruction_error: reconstruction_loss,
            prediction_error: 0.0,
            confidence: prediction.encoded.confidence,
        };
        Ok(ExperienceBehaviorOutput {
            latent,
            reconstruction: Some(prediction.decoded),
            reconstruction_loss: Some(reconstruction_loss),
            confidence: prediction.encoded.confidence,
        })
    }

    fn observe(
        &mut self,
        sample: &TrainingSample<ExperienceBehaviorInput, ExperienceBehaviorOutput>,
    ) -> Result<()> {
        let encode_input = ExperienceEncodeInput {
            sense_vectors: sample.input.sense_vectors.clone(),
        };
        let target = experience_decode_target_from_now(&sample.input.now);
        self.model.train_step(&encode_input, &target)?;
        Ok(())
    }
}

struct HardcodedDangerBehavior;

impl FunctionBehavior<SituatedDangerInput, DangerOutput> for HardcodedDangerBehavior {
    fn id(&self) -> &'static str {
        "danger.range_bumper"
    }

    fn infer(&mut self, input: &SituatedDangerInput) -> Result<DangerOutput> {
        Ok(HardcodedDangerPredictor.predict_from_now(&input.now, &input.input))
    }
}

struct DangerModelBehavior {
    trainer: DangerNetTrainer,
}

impl FunctionBehavior<SituatedDangerInput, DangerOutput> for DangerModelBehavior {
    fn id(&self) -> &'static str {
        "danger.burn.v0"
    }

    fn infer(&mut self, input: &SituatedDangerInput) -> Result<DangerOutput> {
        self.trainer.predict(&input.input)
    }

    fn observe(
        &mut self,
        sample: &TrainingSample<SituatedDangerInput, DangerOutput>,
    ) -> Result<()> {
        let target = pete_experience::DangerTarget {
            bump: sample.expected.bump_risk,
            cliff: sample.expected.cliff_risk,
            wheel_drop: sample.expected.wheel_drop_risk,
            stuck: sample.expected.stuck_risk,
        };
        self.trainer.train_step(&sample.input.input, &target)?;
        Ok(())
    }
}

struct HardcodedChargeBehavior;

impl FunctionBehavior<SituatedChargeInput, ChargeOutput> for HardcodedChargeBehavior {
    fn id(&self) -> &'static str {
        "charge.sensor_battery_delta"
    }

    fn infer(&mut self, input: &SituatedChargeInput) -> Result<ChargeOutput> {
        Ok(HardcodedChargePredictor.predict_from_now(&input.now, &input.input))
    }
}

struct ChargeModelBehavior {
    trainer: ChargeNetTrainer,
}

impl FunctionBehavior<SituatedChargeInput, ChargeOutput> for ChargeModelBehavior {
    fn id(&self) -> &'static str {
        "charge.burn.v0"
    }

    fn infer(&mut self, input: &SituatedChargeInput) -> Result<ChargeOutput> {
        self.trainer.predict(&input.input)
    }

    fn observe(
        &mut self,
        sample: &TrainingSample<SituatedChargeInput, ChargeOutput>,
    ) -> Result<()> {
        let target = pete_experience::ChargeTarget {
            charging_started: sample.expected.charge_probability,
            battery_delta: sample.expected.expected_battery_delta,
            charging_after: sample.expected.dock_likelihood,
        };
        self.trainer.train_step(&sample.input.input, &target)?;
        Ok(())
    }
}

struct HardcodedActionValueBehavior;

impl FunctionBehavior<SituatedActionValueInput, ActionValueOutput>
    for HardcodedActionValueBehavior
{
    fn id(&self) -> &'static str {
        "action_value.handcoded"
    }

    fn infer(&mut self, input: &SituatedActionValueInput) -> Result<ActionValueOutput> {
        Ok(HardcodedActionValuePredictor.predict_from_now(&input.now, &input.input))
    }
}

struct ActionValueModelBehavior {
    trainer: ActionValueNetTrainer,
}

impl FunctionBehavior<SituatedActionValueInput, ActionValueOutput> for ActionValueModelBehavior {
    fn id(&self) -> &'static str {
        "action_value.burn.v0"
    }

    fn infer(&mut self, input: &SituatedActionValueInput) -> Result<ActionValueOutput> {
        self.trainer.predict(&input.input)
    }

    fn observe(
        &mut self,
        sample: &TrainingSample<SituatedActionValueInput, ActionValueOutput>,
    ) -> Result<()> {
        let target = pete_experience::ActionValueTarget {
            value: sample.expected.value,
        };
        self.trainer.train_step(&sample.input.input, &target)?;
        Ok(())
    }
}

struct HardcodedEyeNextBehavior;

impl FunctionBehavior<SituatedEyeNextInput, EyeNextOutput> for HardcodedEyeNextBehavior {
    fn id(&self) -> &'static str {
        "eye.copy_current"
    }

    fn infer(&mut self, input: &SituatedEyeNextInput) -> Result<EyeNextOutput> {
        Ok(CopyCurrentEyePredictor.predict_from_now(&input.now, &input.input))
    }
}

struct EyeNextModelBehavior {
    trainer: EyeNextNetTrainer,
}

impl FunctionBehavior<SituatedEyeNextInput, EyeNextOutput> for EyeNextModelBehavior {
    fn id(&self) -> &'static str {
        "eye.burn.next_v0"
    }

    fn infer(&mut self, input: &SituatedEyeNextInput) -> Result<EyeNextOutput> {
        self.trainer.predict(&input.input)
    }

    fn observe(
        &mut self,
        sample: &TrainingSample<SituatedEyeNextInput, EyeNextOutput>,
    ) -> Result<()> {
        let target = pete_experience::EyeNextTarget {
            width: sample.expected.width,
            height: sample.expected.height,
            rgb: sample.expected.rgb.clone(),
        };
        self.trainer.train_step(&sample.input.input, &target)?;
        Ok(())
    }
}

struct HardcodedEarNextBehavior;

impl FunctionBehavior<SituatedEarNextInput, EarNextOutput> for HardcodedEarNextBehavior {
    fn id(&self) -> &'static str {
        "ear.copy_current"
    }

    fn infer(&mut self, input: &SituatedEarNextInput) -> Result<EarNextOutput> {
        Ok(CopyCurrentEarPredictor.predict_from_now(&input.now, &input.input))
    }
}

struct EarNextModelBehavior {
    trainer: EarNextNetTrainer,
}

impl FunctionBehavior<SituatedEarNextInput, EarNextOutput> for EarNextModelBehavior {
    fn id(&self) -> &'static str {
        "ear.burn.next_v0"
    }

    fn infer(&mut self, input: &SituatedEarNextInput) -> Result<EarNextOutput> {
        self.trainer.predict(&input.input)
    }

    fn observe(
        &mut self,
        sample: &TrainingSample<SituatedEarNextInput, EarNextOutput>,
    ) -> Result<()> {
        let target = pete_experience::EarNextTarget {
            sample_rate_hz: sample.expected.sample_rate_hz,
            channels: sample.expected.channels,
            pcm: sample.expected.pcm.clone(),
            features: sample.expected.features.clone(),
        };
        self.trainer.train_step(&sample.input.input, &target)?;
        Ok(())
    }
}

struct StasisFutureBehavior {
    predictor: StasisFuturePredictor,
}

impl FunctionBehavior<FutureInput, FuturePrediction> for StasisFutureBehavior {
    fn id(&self) -> &'static str {
        "future.stasis"
    }

    fn infer(&mut self, input: &FutureInput) -> Result<FuturePrediction> {
        self.predictor
            .predict(&input.latent, &input.action, input.offset_ms)
    }
}

struct FutureModelBehavior {
    trainer: FutureNetTrainer,
}

impl FunctionBehavior<FutureInput, FuturePrediction> for FutureModelBehavior {
    fn id(&self) -> &'static str {
        "future.burn.v0"
    }

    fn infer(&mut self, input: &FutureInput) -> Result<FuturePrediction> {
        let mut input = input.clone();
        if input.flat_features().len() != self.trainer.input_dim() {
            input.latent.z.resize(self.trainer.latent_dim(), 0.0);
            input.latent.z.truncate(self.trainer.latent_dim());
            let expected_input_dim = self.trainer.latent_dim() + action_features(None).len() + 1;
            if expected_input_dim != self.trainer.input_dim() {
                return Err(anyhow::anyhow!(
                    "future checkpoint input dimension mismatch: checkpoint expects {}, adapted runtime input would be {}",
                    self.trainer.input_dim(),
                    expected_input_dim
                ));
            }
        }
        self.trainer.predict(&input)
    }

    fn observe(&mut self, sample: &TrainingSample<FutureInput, FuturePrediction>) -> Result<()> {
        self.trainer
            .train_step(&sample.input, &sample.expected.predicted_z)?;
        Ok(())
    }
}

fn locomotion_behavior(
    regime: BehaviorRegime,
    model: Option<NeatLocomotionBehavior>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<LocomotionInput, LocomotionOutput> {
    ReplaceableBehavior::new(
        "locomotion",
        regime,
        Box::new(HardcodedLocomotionBehavior::default()),
        model.map(|model| Box::new(model) as Box<_>),
        fallback,
    )
}

fn danger_behavior(
    regime: BehaviorRegime,
    trainer: Option<DangerNetTrainer>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<SituatedDangerInput, DangerOutput> {
    ReplaceableBehavior::new(
        "danger",
        regime,
        Box::new(HardcodedDangerBehavior),
        trainer.map(|trainer| Box::new(DangerModelBehavior { trainer }) as Box<_>),
        fallback,
    )
}

fn experience_behavior(
    regime: BehaviorRegime,
    trainer: Option<ExperienceAutoencoderTrainer>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<ExperienceBehaviorInput, ExperienceBehaviorOutput> {
    ReplaceableBehavior::new(
        "experience",
        regime,
        Box::new(HardcodedExperienceBehavior),
        trainer.map(|trainer| Box::new(LearnedExperienceBehavior { model: trainer }) as Box<_>),
        fallback,
    )
}

fn charge_behavior(
    regime: BehaviorRegime,
    trainer: Option<ChargeNetTrainer>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<SituatedChargeInput, ChargeOutput> {
    ReplaceableBehavior::new(
        "charge",
        regime,
        Box::new(HardcodedChargeBehavior),
        trainer.map(|trainer| Box::new(ChargeModelBehavior { trainer }) as Box<_>),
        fallback,
    )
}

fn future_behavior(
    regime: BehaviorRegime,
    trainer: Option<FutureNetTrainer>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<FutureInput, FuturePrediction> {
    ReplaceableBehavior::new(
        "future",
        regime,
        Box::new(StasisFutureBehavior {
            predictor: StasisFuturePredictor,
        }),
        trainer.map(|trainer| Box::new(FutureModelBehavior { trainer }) as Box<_>),
        fallback,
    )
}

fn action_value_behavior(
    regime: BehaviorRegime,
    trainer: Option<ActionValueNetTrainer>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<SituatedActionValueInput, ActionValueOutput> {
    ReplaceableBehavior::new(
        "action_value",
        regime,
        Box::new(HardcodedActionValueBehavior),
        trainer.map(|trainer| Box::new(ActionValueModelBehavior { trainer }) as Box<_>),
        fallback,
    )
}

fn conductor_behavior(
    regime: BehaviorRegime,
    hardcoded_id: &str,
    model_id: Option<String>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<ConductorInput, ActionPrimitive> {
    ReplaceableBehavior::new(
        "conductor",
        regime,
        Box::new(HardcodedConductorBehavior {
            id: known_conductor_hardcoded_id(hardcoded_id),
        }),
        model_id.map(|id| {
            Box::new(ShadowActionSelectorModel {
                id,
                last_observed: None,
                samples_seen: 0,
            }) as Box<_>
        }),
        fallback,
    )
}

fn known_conductor_hardcoded_id(id: &str) -> &'static str {
    match id {
        "action_selector.baseline" => "action_selector.baseline",
        "reign.teacher" => "reign.teacher",
        _ => "conductor.simple_v0",
    }
}

struct HardcodedConductorBehavior {
    id: &'static str,
}

impl FunctionBehavior<ConductorInput, ActionPrimitive> for HardcodedConductorBehavior {
    fn id(&self) -> &'static str {
        self.id
    }

    fn infer(&mut self, input: &ConductorInput) -> Result<ActionPrimitive> {
        match self.id {
            "reign.teacher" => input
                .reign
                .latest
                .as_ref()
                .and_then(|input| input.command.to_action())
                .or_else(|| input.proposals.last().cloned())
                .map(Ok)
                .unwrap_or_else(|| Ok(ActionPrimitive::Stop)),
            "action_selector.baseline" => {
                Ok(input
                    .proposals
                    .last()
                    .cloned()
                    .unwrap_or(ActionPrimitive::Explore {
                        style: ExploreStyle::RandomWalk,
                        duration_ms: 1_000,
                    }))
            }
            _ => SimpleConductor::default().choose(input.clone()),
        }
    }
}

struct ShadowActionSelectorModel {
    id: String,
    last_observed: Option<ActionPrimitive>,
    samples_seen: usize,
}

impl FunctionBehavior<ConductorInput, ActionPrimitive> for ShadowActionSelectorModel {
    fn id(&self) -> &'static str {
        "conductor.burn.v0"
    }

    fn infer(&mut self, _input: &ConductorInput) -> Result<ActionPrimitive> {
        self.last_observed
            .clone()
            .ok_or_else(|| anyhow::anyhow!("{} has no observed teacher samples", self.id))
    }

    fn observe(&mut self, sample: &TrainingSample<ConductorInput, ActionPrimitive>) -> Result<()> {
        if sample.source != TrainingSource::SafetyVeto {
            self.last_observed = Some(sample.expected.clone());
            self.samples_seen = self.samples_seen.saturating_add(1);
        }
        Ok(())
    }
}

struct BumpScriptBehavior;

impl FunctionBehavior<BumpEventInput, EventScriptOutput> for BumpScriptBehavior {
    fn id(&self) -> &'static str {
        "script.on_bump.ts.v0"
    }

    fn infer(&mut self, input: &BumpEventInput) -> Result<EventScriptOutput> {
        execute_event_script_typescript(BUMP_SCRIPT, input)
    }
}

const BUMP_SCRIPT: &str = r#"
const r = random();
const lament =
  r < 0.20 ? say("Uh-oh") :
  r < 0.40 ? say("Oh no!") :
  r < 0.60 ? say("Oopsie!") :
  r < 0.80 ? say("Oh dear!") :
             song("mournful_bump");

[
  chirp("Warning"),
  lament,
  stop(),
  rotate(180),
  go()
]
"#;

struct RobotInitializedScriptBehavior;

impl FunctionBehavior<RobotInitializedEventInput, EventScriptOutput>
    for RobotInitializedScriptBehavior
{
    fn id(&self) -> &'static str {
        "script.on_robot_initialized.ts.v0"
    }

    fn infer(&mut self, input: &RobotInitializedEventInput) -> Result<EventScriptOutput> {
        execute_event_script_typescript(ROBOT_INITIALIZED_SCRIPT, input)
    }
}

const ROBOT_INITIALIZED_SCRIPT: &str = r#"
[
  song("bring_up"),
  chirp("Wake"),
  chirp("Hello"),
  say(`Pete robot initialization complete in ${input.mode} mode.`),
  say(`${input.body}.`),
  input.battery_percent === null
    ? say("Battery status is unavailable.")
    : say(`Battery is ${input.battery_percent} percent and ${input.charging ? "charging" : "not charging"}.`),
  input.requested_sensors === 0
    ? say("No optional sensors requested.")
    : say(`${input.active_sensors} of ${input.requested_sensors} optional sensors initialized.`),
  say(`Ledger is ready at ${input.ledger}.`),
  say(`Tick rate is ${input.tick_ms} milliseconds.`),
  input.dashboard ? say(`Dashboard is listening at ${input.dashboard}.`) : say("Dashboard is not enabled."),
  input.capture ? say(`Capture recording is armed at ${input.capture}.`) : say("Capture recording is not enabled."),
  input.mode === "read-only"
    ? say("Read only mode is active. Motors are suppressed.")
    : say("Slow mode is active. Guarded motor commands are enabled."),
  chirp("Confirm")
]
"#;

struct EventScriptShadowModel {
    id: &'static str,
    last_observed: Option<EventScriptOutput>,
    samples_seen: usize,
}

impl<I> FunctionBehavior<I, EventScriptOutput> for EventScriptShadowModel
where
    I: Send,
{
    fn id(&self) -> &'static str {
        self.id
    }

    fn infer(&mut self, _input: &I) -> Result<EventScriptOutput> {
        self.last_observed
            .clone()
            .ok_or_else(|| anyhow::anyhow!("{} has no observed script samples", self.id))
    }

    fn observe(&mut self, sample: &TrainingSample<I, EventScriptOutput>) -> Result<()> {
        self.last_observed = Some(sample.expected.clone());
        self.samples_seen = self.samples_seen.saturating_add(1);
        Ok(())
    }
}

fn robot_initialized_event_behavior(
    regime: BehaviorRegime,
    model_id: Option<String>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<RobotInitializedEventInput, EventScriptOutput> {
    ReplaceableBehavior::new(
        "event_robot_initialized",
        regime,
        Box::new(RobotInitializedScriptBehavior),
        model_id.map(|_| {
            Box::new(EventScriptShadowModel {
                id: "event.robot_initialized.shadow.v0",
                last_observed: None,
                samples_seen: 0,
            }) as Box<_>
        }),
        fallback,
    )
}

fn bump_event_behavior(
    regime: BehaviorRegime,
    model_id: Option<String>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<BumpEventInput, EventScriptOutput> {
    ReplaceableBehavior::new(
        "event_bump",
        regime,
        Box::new(BumpScriptBehavior),
        model_id.map(|_| {
            Box::new(EventScriptShadowModel {
                id: "event.bump.shadow.v0",
                last_observed: None,
                samples_seen: 0,
            }) as Box<_>
        }),
        fallback,
    )
}

fn eye_next_behavior(
    regime: BehaviorRegime,
    trainer: Option<EyeNextNetTrainer>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<SituatedEyeNextInput, EyeNextOutput> {
    ReplaceableBehavior::new(
        "eye_next",
        regime,
        Box::new(HardcodedEyeNextBehavior),
        trainer.map(|trainer| Box::new(EyeNextModelBehavior { trainer }) as Box<_>),
        fallback,
    )
}

fn ear_next_behavior(
    regime: BehaviorRegime,
    trainer: Option<EarNextNetTrainer>,
    fallback: FallbackPolicy,
) -> ReplaceableBehavior<SituatedEarNextInput, EarNextOutput> {
    ReplaceableBehavior::new(
        "ear_next",
        regime,
        Box::new(HardcodedEarNextBehavior),
        trainer.map(|trainer| Box::new(EarNextModelBehavior { trainer }) as Box<_>),
        fallback,
    )
}

fn load_locomotion_behavior(behavior: &BehaviorConfig) -> Result<Option<NeatLocomotionBehavior>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    let checkpoint_path = Path::new(checkpoint);
    let artifact = if checkpoint_path.extension().is_some() {
        checkpoint_path.to_path_buf()
    } else {
        checkpoint_path.join("locomotion-neat.json")
    };
    if !artifact.exists() {
        return Ok(None);
    }
    Ok(Some(NeatLocomotionBehavior::load(checkpoint)?))
}

fn load_danger_behavior_trainer(behavior: &BehaviorConfig) -> Result<Option<DangerNetTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_danger_metadata(checkpoint)?;
    Ok(Some(DangerNetTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
    )?))
}

fn load_charge_behavior_trainer(behavior: &BehaviorConfig) -> Result<Option<ChargeNetTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_charge_metadata(checkpoint)?;
    Ok(Some(ChargeNetTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
    )?))
}

fn load_action_value_behavior_trainer(
    behavior: &BehaviorConfig,
) -> Result<Option<ActionValueNetTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_action_value_metadata(checkpoint)?;
    Ok(Some(ActionValueNetTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
    )?))
}

fn load_future_behavior_trainer(behavior: &BehaviorConfig) -> Result<Option<FutureNetTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_future_metadata(checkpoint)?;
    Ok(Some(FutureNetTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
        metadata.latent_dim,
    )?))
}

fn load_eye_next_behavior_trainer(behavior: &BehaviorConfig) -> Result<Option<EyeNextNetTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_eye_next_metadata(checkpoint)?;
    Ok(Some(EyeNextNetTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
    )?))
}

fn load_ear_next_behavior_trainer(behavior: &BehaviorConfig) -> Result<Option<EarNextNetTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_ear_next_metadata(checkpoint)?;
    Ok(Some(EarNextNetTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
    )?))
}

fn load_experience_behavior_trainer(
    behavior: &BehaviorConfig,
) -> Result<Option<ExperienceAutoencoderTrainer>> {
    let Some(checkpoint) = behavior.checkpoint.as_deref() else {
        return Ok(None);
    };
    if !Path::new(checkpoint).exists() {
        return Ok(None);
    }
    let metadata = read_experience_autoencoder_metadata(checkpoint)?;
    Ok(Some(ExperienceAutoencoderTrainer::load_checkpoint(
        checkpoint,
        metadata.input_dim,
    )?))
}

fn danger_behavior_input(
    now: &Now,
    latent: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
) -> SituatedDangerInput {
    SituatedDangerInput {
        input: DangerInput::from_parts(latent.z.clone(), action, now),
        now: now.clone(),
    }
}

fn charge_behavior_input(
    now: &Now,
    latent: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
) -> SituatedChargeInput {
    SituatedChargeInput {
        input: ChargeInput::from_parts(latent.z.clone(), action, now),
        now: now.clone(),
    }
}

fn action_value_behavior_input(
    now: &Now,
    latent: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
    danger: Option<DangerOutput>,
    charge: Option<ChargeOutput>,
) -> SituatedActionValueInput {
    SituatedActionValueInput {
        input: ActionValueInput::from_parts_with_predictions(
            latent.z.clone(),
            action,
            now,
            danger,
            charge,
        ),
        now: now.clone(),
    }
}

fn eye_next_behavior_input(
    now: &Now,
    latent: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
    offset_ms: TimeMs,
) -> SituatedEyeNextInput {
    SituatedEyeNextInput {
        input: EyeNextInput::from_parts(latent.z.clone(), action, now, offset_ms),
        now: now.clone(),
    }
}

fn ear_next_behavior_input(
    now: &Now,
    latent: &ExperienceLatent,
    action: Option<&ActionPrimitive>,
    offset_ms: TimeMs,
) -> SituatedEarNextInput {
    SituatedEarNextInput {
        input: EarNextInput::from_parts(latent.z.clone(), action, now, offset_ms),
        now: now.clone(),
    }
}

fn danger_disagreement(left: &DangerOutput, right: &DangerOutput) -> f32 {
    let deltas = [
        (left.bump_risk - right.bump_risk).abs(),
        (left.cliff_risk - right.cliff_risk).abs(),
        (left.wheel_drop_risk - right.wheel_drop_risk).abs(),
        (left.stuck_risk - right.stuck_risk).abs(),
    ];
    deltas.iter().sum::<f32>() / deltas.len() as f32
}

fn action_value_disagreement(left: &ActionValueOutput, right: &ActionValueOutput) -> f32 {
    (left.value - right.value).abs()
}

fn charge_disagreement(left: &ChargeOutput, right: &ChargeOutput) -> f32 {
    let deltas = [
        (left.charge_probability - right.charge_probability).abs(),
        (left.expected_battery_delta - right.expected_battery_delta).abs(),
        (left.dock_likelihood - right.dock_likelihood).abs(),
    ];
    deltas.iter().sum::<f32>() / deltas.len() as f32
}

fn eye_next_disagreement(left: &EyeNextOutput, right: &EyeNextOutput) -> f32 {
    let len = left.rgb.len().max(right.rgb.len());
    if len == 0 {
        return 0.0;
    }
    (0..len)
        .map(|idx| {
            let left = left.rgb.get(idx).copied().unwrap_or_default() as f32 / 255.0;
            let right = right.rgb.get(idx).copied().unwrap_or_default() as f32 / 255.0;
            (left - right).abs()
        })
        .sum::<f32>()
        / len as f32
}

fn ear_next_disagreement(left: &EarNextOutput, right: &EarNextOutput) -> f32 {
    let len = left.features.len().max(right.features.len());
    if len == 0 {
        return 0.0;
    }
    (0..len)
        .map(|idx| {
            let left = left.features.get(idx).copied().unwrap_or_default();
            let right = right.features.get(idx).copied().unwrap_or_default();
            (left - right).abs()
        })
        .sum::<f32>()
        / len as f32
}

fn experience_reconstruction_loss_flat(
    output: &ExperienceDecodeOutput,
    target: &ExperienceDecodeOutput,
) -> f32 {
    let output = output.flat_features();
    let target = target.flat_features();
    let len = output.len().max(target.len());
    if len == 0 {
        return 0.0;
    }
    (0..len)
        .map(|idx| {
            let actual = output.get(idx).copied().unwrap_or_default();
            let expected = target.get(idx).copied().unwrap_or_default();
            let delta = actual - expected;
            delta * delta
        })
        .sum::<f32>()
        / len as f32
}

fn experience_disagreement(
    left: &ExperienceBehaviorOutput,
    right: &ExperienceBehaviorOutput,
) -> f32 {
    let a = &left.latent.z;
    let b = &right.latent.z;
    let len = a.len().max(b.len());
    if len == 0 {
        return 0.0;
    }
    let sum: f32 = (0..len)
        .map(|idx| {
            let delta =
                a.get(idx).copied().unwrap_or_default() - b.get(idx).copied().unwrap_or_default();
            delta * delta
        })
        .sum();
    sum.sqrt()
}

