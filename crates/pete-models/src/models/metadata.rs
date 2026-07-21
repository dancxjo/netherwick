pub fn read_danger_metadata(path: impl AsRef<Path>) -> Result<DangerModelMetadata> {
    let path = path.as_ref();
    let bytes = std::fs::read(path.join("metadata.json"))
        .with_context(|| format!("read danger checkpoint metadata {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse danger checkpoint metadata {}", path.display()))
}

pub fn read_charge_metadata(path: impl AsRef<Path>) -> Result<ChargeModelMetadata> {
    let path = path.as_ref();
    let bytes = std::fs::read(path.join("metadata.json"))
        .with_context(|| format!("read charge checkpoint metadata {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse charge checkpoint metadata {}", path.display()))
}

pub fn read_action_value_metadata(path: impl AsRef<Path>) -> Result<ActionValueModelMetadata> {
    let path = path.as_ref();
    let bytes = std::fs::read(path.join("metadata.json"))
        .with_context(|| format!("read action-value checkpoint metadata {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse action-value checkpoint metadata {}", path.display()))
}

pub fn read_future_metadata(path: impl AsRef<Path>) -> Result<FutureModelMetadata> {
    let path = path.as_ref();
    let bytes = std::fs::read(path.join("metadata.json"))
        .with_context(|| format!("read future checkpoint metadata {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse future checkpoint metadata {}", path.display()))
}

pub fn read_eye_next_metadata(path: impl AsRef<Path>) -> Result<EyeNextModelMetadata> {
    let path = path.as_ref();
    let bytes = std::fs::read(path.join("metadata.json"))
        .with_context(|| format!("read eye-next checkpoint metadata {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse eye-next checkpoint metadata {}", path.display()))
}

pub fn read_ear_next_metadata(path: impl AsRef<Path>) -> Result<EarNextModelMetadata> {
    let path = path.as_ref();
    let bytes = std::fs::read(path.join("metadata.json"))
        .with_context(|| format!("read ear-next checkpoint metadata {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse ear-next checkpoint metadata {}", path.display()))
}

pub fn read_experience_autoencoder_metadata(
    path: impl AsRef<Path>,
) -> Result<ExperienceAutoencoderMetadata> {
    let path = path.as_ref();
    let bytes = std::fs::read(path.join("metadata.json")).with_context(|| {
        format!(
            "read experience autoencoder checkpoint metadata {}",
            path.display()
        )
    })?;
    serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "parse experience autoencoder checkpoint metadata {}",
            path.display()
        )
    })
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

impl<B: AutodiffBackend> OnlineTrainer<DangerInput, DangerTarget> for DangerNetTrainer<B> {
    fn train_step(
        &mut self,
        sample: TrainingSample<DangerInput, DangerTarget>,
    ) -> Result<TrainStats> {
        let stats = DangerNetTrainer::train_step(self, &sample.input, &sample.expected)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }
}

impl<B: AutodiffBackend> OnlineTrainer<ChargeInput, ChargeTarget> for ChargeNetTrainer<B> {
    fn train_step(
        &mut self,
        sample: TrainingSample<ChargeInput, ChargeTarget>,
    ) -> Result<TrainStats> {
        let stats = ChargeNetTrainer::train_step(self, &sample.input, &sample.expected)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }
}

impl<B: AutodiffBackend> OnlineTrainer<ActionValueInput, ActionValueTarget>
    for ActionValueNetTrainer<B>
{
    fn train_step(
        &mut self,
        sample: TrainingSample<ActionValueInput, ActionValueTarget>,
    ) -> Result<TrainStats> {
        let stats = ActionValueNetTrainer::train_step(self, &sample.input, &sample.expected)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }
}

impl<B: AutodiffBackend> OnlineTrainer<FutureInput, Vec<f32>> for FutureNetTrainer<B> {
    fn train_step(&mut self, sample: TrainingSample<FutureInput, Vec<f32>>) -> Result<TrainStats> {
        FutureNetTrainer::train_step(self, &sample.input, &sample.expected)
    }
}

impl<B: AutodiffBackend> OnlineTrainer<EyeNextInput, EyeNextTarget> for EyeNextNetTrainer<B> {
    fn train_step(
        &mut self,
        sample: TrainingSample<EyeNextInput, EyeNextTarget>,
    ) -> Result<TrainStats> {
        let stats = EyeNextNetTrainer::train_step(self, &sample.input, &sample.expected)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }
}

impl<B: AutodiffBackend> OnlineTrainer<EarNextInput, EarNextTarget> for EarNextNetTrainer<B> {
    fn train_step(
        &mut self,
        sample: TrainingSample<EarNextInput, EarNextTarget>,
    ) -> Result<TrainStats> {
        let stats = EarNextNetTrainer::train_step(self, &sample.input, &sample.expected)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }
}

impl<B: AutodiffBackend> OnlineTrainer<ExperienceEncodeInput, ExperienceDecodeOutput>
    for ExperienceAutoencoderTrainer<B>
{
    fn train_step(
        &mut self,
        sample: TrainingSample<ExperienceEncodeInput, ExperienceDecodeOutput>,
    ) -> Result<TrainStats> {
        let stats =
            ExperienceAutoencoderTrainer::train_step(self, &sample.input, &sample.expected)?;
        Ok(TrainStats {
            loss: stats.loss,
            samples_seen: stats.samples_seen,
            improved: stats.improved,
        })
    }
}

impl<B: AutodiffBackend> LatentEncoder for ExperienceAutoencoderTrainer<B> {
    fn encoder_kind(&self) -> &'static str {
        "trainable-autoencoder"
    }

    fn encode_input(
        &mut self,
        input: &ExperienceEncodeInput,
        t_ms: TimeMs,
    ) -> Result<ExperienceLatent> {
        let encoded = self.encode(input)?;
        Ok(ExperienceLatent {
            t_ms,
            z: encoded.z,
            reconstruction_error: 0.0,
            prediction_error: 0.0,
            confidence: encoded.confidence,
        })
    }
}
