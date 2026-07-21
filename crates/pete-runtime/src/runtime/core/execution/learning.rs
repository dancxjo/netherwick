impl<L, M, R, C, S, A> MinimalRuntime<L, M, R, C, S, A>
where
    L: LedgerWriter + Sync,
    M: MemoryStore,
    R: Recall + Sync,
    C: Conductor,
    S: SafetyLayer,
    A: LlmAgent + 'static,
{
    fn observe_inline_learning(
        &mut self,
        transition: &ExperienceTransition,
    ) -> Result<InlineLearningTickStatus> {
        let mut status = InlineLearningTickStatus {
            enabled: self.inline_learning.is_enabled(),
            mode: self.inline_learning.mode,
            samples_observed: 0,
            train_steps_used: 0,
        };
        if !self.inline_learning.is_enabled() {
            return Ok(status);
        }
        if self.inline_learning.mode != InlineLearningMode::WorldOutcome {
            return Ok(status);
        }

        let mut remaining = self.inline_learning.max_train_steps_per_tick;
        if self.inline_learning.behaviors.danger && remaining > 0 {
            if let Some(sample) = self
                .behavior_training_hub
                .danger_extractor
                .extract(transition)?
            {
                let sample = TrainingSample {
                    input: SituatedDangerInput {
                        input: sample.input,
                        now: transition.before.clone(),
                    },
                    expected: sample.expected,
                    actual: sample.actual,
                    reward: sample.reward,
                    weight: sample.weight,
                    source: sample.source,
                    t_ms: sample.t_ms,
                };
                self.models.behaviors.danger.observe(&sample)?;
                remaining = remaining.saturating_sub(1);
                status.samples_observed = status.samples_observed.saturating_add(1);
                status.train_steps_used = status.train_steps_used.saturating_add(1);
            }
        }
        if self.inline_learning.behaviors.charge && remaining > 0 {
            if let Some(sample) = self
                .behavior_training_hub
                .charge_extractor
                .extract(transition)?
            {
                let sample = TrainingSample {
                    input: SituatedChargeInput {
                        input: sample.input,
                        now: transition.before.clone(),
                    },
                    expected: sample.expected,
                    actual: sample.actual,
                    reward: sample.reward,
                    weight: sample.weight,
                    source: sample.source,
                    t_ms: sample.t_ms,
                };
                self.models.behaviors.charge.observe(&sample)?;
                remaining = remaining.saturating_sub(1);
                status.samples_observed = status.samples_observed.saturating_add(1);
                status.train_steps_used = status.train_steps_used.saturating_add(1);
            }
        }
        if self.inline_learning.behaviors.future && remaining > 0 {
            if let Some(sample) = self
                .behavior_training_hub
                .future_extractor
                .extract(transition)?
            {
                self.models.behaviors.future.observe(&sample)?;
                remaining = remaining.saturating_sub(1);
                status.samples_observed = status.samples_observed.saturating_add(1);
                status.train_steps_used = status.train_steps_used.saturating_add(1);
            }
        }
        if self.inline_learning.behaviors.action_value && remaining > 0 {
            if let Some(sample) = self
                .behavior_training_hub
                .action_value_extractor
                .extract(transition)?
            {
                let sample = TrainingSample {
                    input: SituatedActionValueInput {
                        input: sample.input,
                        now: transition.before.clone(),
                    },
                    expected: sample.expected,
                    actual: sample.actual,
                    reward: sample.reward,
                    weight: sample.weight,
                    source: sample.source,
                    t_ms: sample.t_ms,
                };
                self.models.behaviors.action_value.observe(&sample)?;
                remaining = remaining.saturating_sub(1);
                status.samples_observed = status.samples_observed.saturating_add(1);
                status.train_steps_used = status.train_steps_used.saturating_add(1);
            }
        }
        if self.inline_learning.behaviors.eye_next && remaining > 0 {
            if let Some(sample) = self
                .behavior_training_hub
                .eye_next_extractor
                .extract(transition)?
            {
                let sample = TrainingSample {
                    input: SituatedEyeNextInput {
                        input: sample.input,
                        now: transition.before.clone(),
                    },
                    expected: sample.expected,
                    actual: sample.actual,
                    reward: sample.reward,
                    weight: sample.weight,
                    source: sample.source,
                    t_ms: sample.t_ms,
                };
                self.models.behaviors.eye_next.observe(&sample)?;
                remaining = remaining.saturating_sub(1);
                status.samples_observed = status.samples_observed.saturating_add(1);
                status.train_steps_used = status.train_steps_used.saturating_add(1);
            }
        }
        if self.inline_learning.behaviors.ear_next && remaining > 0 {
            if let Some(sample) = self
                .behavior_training_hub
                .ear_next_extractor
                .extract(transition)?
            {
                let sample = TrainingSample {
                    input: SituatedEarNextInput {
                        input: sample.input,
                        now: transition.before.clone(),
                    },
                    expected: sample.expected,
                    actual: sample.actual,
                    reward: sample.reward,
                    weight: sample.weight,
                    source: sample.source,
                    t_ms: sample.t_ms,
                };
                self.models.behaviors.ear_next.observe(&sample)?;
                remaining = remaining.saturating_sub(1);
                status.samples_observed = status.samples_observed.saturating_add(1);
                status.train_steps_used = status.train_steps_used.saturating_add(1);
            }
        }
        if self.inline_learning.behaviors.experience && remaining > 0 {
            let input = ExperienceBehaviorInput::from_now(&transition.before);
            let sample = TrainingSample {
                input,
                expected: ExperienceBehaviorOutput {
                    latent: transition.before_z.clone(),
                    reconstruction: None,
                    reconstruction_loss: None,
                    confidence: transition.before_z.confidence,
                },
                actual: None,
                reward: Some(transition.reward.value),
                weight: 1.0,
                source: TrainingSource::WorldOutcome,
                t_ms: transition.created_at_ms,
            };
            self.models.behaviors.experience.observe(&sample)?;
            status.samples_observed = status.samples_observed.saturating_add(1);
            status.train_steps_used = status.train_steps_used.saturating_add(1);
        }

        Ok(status)
    }
}
