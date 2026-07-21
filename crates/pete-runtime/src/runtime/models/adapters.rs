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
