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
