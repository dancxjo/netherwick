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
