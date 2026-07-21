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
