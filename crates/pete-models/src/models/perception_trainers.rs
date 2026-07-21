impl EyeNextNetTrainer<EyeNextAutodiffBackend> {
    pub fn new(input_dim: usize, width: u32, height: u32) -> Self {
        Self::with_device(input_dim, width, height, Default::default())
    }

    pub fn load_checkpoint(path: impl AsRef<Path>, input_dim: usize) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_eye_next_metadata(path)?;
        if metadata.input_dim != input_dim {
            return Err(anyhow!(
                "eye-next checkpoint input dimension mismatch at {}: metadata has {}, runtime expected {}",
                path.display(),
                metadata.input_dim,
                input_dim
            ));
        }

        let device = Default::default();
        let model = EyeNextNet::init(input_dim, metadata.output_dim, &device).load_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
            &device,
        )?;
        Ok(Self {
            model,
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            output_dim: metadata.output_dim,
            width: metadata.width,
            height: metadata.height,
            learning_rate: 0.01,
            samples_seen: metadata.samples_seen,
            best_loss: metadata.best_loss,
        })
    }
}

impl<B: AutodiffBackend> EyeNextNetTrainer<B> {
    pub fn with_device(input_dim: usize, width: u32, height: u32, device: B::Device) -> Self {
        let output_dim = width as usize * height as usize * 3;
        Self {
            model: EyeNextNet::init(input_dim, output_dim, &device),
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            output_dim,
            width,
            height,
            learning_rate: 0.01,
            samples_seen: 0,
            best_loss: None,
        }
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn samples_seen(&self) -> u64 {
        self.samples_seen
    }

    pub fn best_loss(&self) -> Option<f32> {
        self.best_loss
    }

    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)
            .with_context(|| format!("create eye-next checkpoint dir {}", path.display()))?;
        self.model.clone().save_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
        )?;
        let metadata = EyeNextModelMetadata {
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            width: self.width,
            height: self.height,
            samples_seen: self.samples_seen,
            best_loss: self.best_loss,
            created_at_ms: now_ms(),
        };
        std::fs::write(
            path.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )
        .with_context(|| format!("write eye-next checkpoint metadata {}", path.display()))?;
        Ok(())
    }

    pub fn predict(&self, input: &EyeNextInput) -> Result<EyeNextOutput> {
        let features = self.checked_features(input)?;
        let tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let output = self.model.forward(tensor).inner();
        tensor_to_eye_next_output(output, self.width, self.height)
    }

    pub fn train_step(
        &mut self,
        input: &EyeNextInput,
        target: &EyeNextTarget,
    ) -> Result<EyeNextTrainStats> {
        let features = self.checked_features(input)?;
        let target_values = eye_target_train_values(target, self.output_dim);
        let input_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let target_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(target_values, [1, self.output_dim]),
            &self.device,
        );
        let output = self.model.forward(input_tensor);
        let loss = MseLoss::new().forward(output, target_tensor, Reduction::Mean);
        let loss_value = loss.clone().inner().into_data().to_vec::<f32>()?[0];
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = self
            .optimizer
            .step(self.learning_rate, self.model.clone(), grads);
        self.samples_seen = self.samples_seen.saturating_add(1);
        let improved = self.best_loss.map(|best| loss_value < best).unwrap_or(true);
        if improved {
            self.best_loss = Some(loss_value);
        }
        Ok(EyeNextTrainStats {
            loss: loss_value,
            samples_seen: self.samples_seen,
            improved,
        })
    }

    pub fn shadow_compare(
        &mut self,
        observed_at_ms: u64,
        now: &Now,
        input: &EyeNextInput,
        target: &EyeNextTarget,
    ) -> Result<EyeNextShadowMetric> {
        let hardcoded = CopyCurrentEyePredictor.predict_from_now(now, input);
        let model = self.predict(input)?;
        let loss = mse_eye_next_output_target(&model, target);
        Ok(EyeNextShadowMetric {
            observed_at_ms,
            hardcoded,
            model,
            target: target.clone(),
            loss,
        })
    }

    fn checked_features(&self, input: &EyeNextInput) -> Result<Vec<f32>> {
        let mut features = input.flat_features();
        if features.len() != self.input_dim {
            return Err(anyhow!(
                "eye-next input dimension mismatch: got {}, expected {}",
                features.len(),
                self.input_dim
            ));
        }
        for value in &mut features {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
        Ok(features)
    }
}

impl EarNextNetTrainer<EarNextAutodiffBackend> {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self::with_device(input_dim, output_dim, 0, 0, Default::default())
    }

    pub fn with_audio_shape(
        input_dim: usize,
        output_dim: usize,
        sample_rate_hz: u32,
        channels: u16,
    ) -> Self {
        Self::with_device(
            input_dim,
            output_dim,
            sample_rate_hz,
            channels,
            Default::default(),
        )
    }

    pub fn load_checkpoint(path: impl AsRef<Path>, input_dim: usize) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_ear_next_metadata(path)?;
        if metadata.input_dim != input_dim {
            return Err(anyhow!(
                "ear-next checkpoint input dimension mismatch at {}: metadata has {}, runtime expected {}",
                path.display(),
                metadata.input_dim,
                input_dim
            ));
        }

        let device = Default::default();
        let model = EarNextNet::init(input_dim, metadata.output_dim, &device).load_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
            &device,
        )?;
        Ok(Self {
            model,
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            output_dim: metadata.output_dim,
            sample_rate_hz: metadata.sample_rate_hz,
            channels: metadata.channels,
            learning_rate: 0.01,
            samples_seen: metadata.samples_seen,
            best_loss: metadata.best_loss,
        })
    }
}

impl<B: AutodiffBackend> EarNextNetTrainer<B> {
    pub fn with_device(
        input_dim: usize,
        output_dim: usize,
        sample_rate_hz: u32,
        channels: u16,
        device: B::Device,
    ) -> Self {
        Self {
            model: EarNextNet::init(input_dim, output_dim, &device),
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            output_dim,
            sample_rate_hz,
            channels,
            learning_rate: 0.01,
            samples_seen: 0,
            best_loss: None,
        }
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn samples_seen(&self) -> u64 {
        self.samples_seen
    }

    pub fn best_loss(&self) -> Option<f32> {
        self.best_loss
    }

    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)
            .with_context(|| format!("create ear-next checkpoint dir {}", path.display()))?;
        self.model.clone().save_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
        )?;
        let metadata = EarNextModelMetadata {
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            sample_rate_hz: self.sample_rate_hz,
            channels: self.channels,
            samples_seen: self.samples_seen,
            best_loss: self.best_loss,
            created_at_ms: now_ms(),
        };
        std::fs::write(
            path.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )
        .with_context(|| format!("write ear-next checkpoint metadata {}", path.display()))?;
        Ok(())
    }

    pub fn predict(&self, input: &EarNextInput) -> Result<EarNextOutput> {
        let features = self.checked_features(input)?;
        let tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let output = self.model.forward(tensor).inner();
        tensor_to_ear_next_output(output, self.output_dim, self.sample_rate_hz, self.channels)
    }

    pub fn train_step(
        &mut self,
        input: &EarNextInput,
        target: &EarNextTarget,
    ) -> Result<EarNextTrainStats> {
        let features = self.checked_features(input)?;
        let target_values = ear_target_train_values(target, self.output_dim);
        let input_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let target_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(target_values, [1, self.output_dim]),
            &self.device,
        );
        let output = self.model.forward(input_tensor);
        let loss = MseLoss::new().forward(output, target_tensor, Reduction::Mean);
        let loss_value = loss.clone().inner().into_data().to_vec::<f32>()?[0];
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = self
            .optimizer
            .step(self.learning_rate, self.model.clone(), grads);
        self.samples_seen = self.samples_seen.saturating_add(1);
        let improved = self.best_loss.map(|best| loss_value < best).unwrap_or(true);
        if improved {
            self.best_loss = Some(loss_value);
        }
        Ok(EarNextTrainStats {
            loss: loss_value,
            samples_seen: self.samples_seen,
            improved,
        })
    }

    pub fn shadow_compare(
        &mut self,
        observed_at_ms: u64,
        now: &Now,
        input: &EarNextInput,
        target: &EarNextTarget,
    ) -> Result<EarNextShadowMetric> {
        let hardcoded = CopyCurrentEarPredictor.predict_from_now(now, input);
        let model = self.predict(input)?;
        let loss = mse_ear_next_output_target(&model, target);
        Ok(EarNextShadowMetric {
            observed_at_ms,
            hardcoded,
            model,
            target: target.clone(),
            loss,
        })
    }

    fn checked_features(&self, input: &EarNextInput) -> Result<Vec<f32>> {
        let mut features = input.flat_features();
        if features.len() != self.input_dim {
            return Err(anyhow!(
                "ear-next input dimension mismatch: got {}, expected {}",
                features.len(),
                self.input_dim
            ));
        }
        for value in &mut features {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
        Ok(features)
    }
}

impl ExperienceAutoencoderTrainer<ExperienceAutoencoderAutodiffBackend> {
    pub fn new(
        input_dim: usize,
        z_dim: usize,
        decode_lengths: ExperienceDecodeFeatureLengths,
    ) -> Self {
        Self::with_device(input_dim, z_dim, decode_lengths, Default::default())
    }

    pub fn load_checkpoint(path: impl AsRef<Path>, input_dim: usize) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_experience_autoencoder_metadata(path)?;
        if metadata.input_dim != input_dim {
            return Err(anyhow!(
                "experience autoencoder checkpoint input dimension mismatch at {}: metadata has {}, runtime expected {}",
                path.display(),
                metadata.input_dim,
                input_dim
            ));
        }

        let device = Default::default();
        let model =
            ExperienceAutoencoderNet::init(input_dim, metadata.z_dim, metadata.output_dim, &device)
                .load_file(
                    path.join("model"),
                    &BinFileRecorder::<FullPrecisionSettings>::default(),
                    &device,
                )?;
        Ok(Self {
            model,
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            z_dim: metadata.z_dim,
            output_dim: metadata.output_dim,
            decode_lengths: metadata.decode_lengths,
            learning_rate: 0.01,
            samples_seen: metadata.samples_seen,
            best_loss: metadata.best_loss,
        })
    }
}

impl<B: AutodiffBackend> ExperienceAutoencoderTrainer<B> {
    pub fn with_device(
        input_dim: usize,
        z_dim: usize,
        decode_lengths: ExperienceDecodeFeatureLengths,
        device: B::Device,
    ) -> Self {
        let output_dim = decode_lengths.body
            + decode_lengths.memory
            + decode_lengths.drive
            + decode_lengths.prediction
            + decode_lengths.eye
            + decode_lengths.ear;
        Self {
            model: ExperienceAutoencoderNet::init(input_dim, z_dim, output_dim, &device),
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            z_dim,
            output_dim,
            decode_lengths,
            learning_rate: 0.01,
            samples_seen: 0,
            best_loss: None,
        }
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn z_dim(&self) -> usize {
        self.z_dim
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn samples_seen(&self) -> u64 {
        self.samples_seen
    }

    pub fn best_loss(&self) -> Option<f32> {
        self.best_loss
    }

    pub fn decode_lengths(&self) -> ExperienceDecodeFeatureLengths {
        self.decode_lengths
    }

    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        std::fs::create_dir_all(path).with_context(|| {
            format!(
                "create experience autoencoder checkpoint dir {}",
                path.display()
            )
        })?;
        self.model.clone().save_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
        )?;
        let metadata = ExperienceAutoencoderMetadata {
            input_dim: self.input_dim,
            z_dim: self.z_dim,
            output_dim: self.output_dim,
            decode_lengths: self.decode_lengths,
            samples_seen: self.samples_seen,
            best_loss: self.best_loss,
            created_at_ms: now_ms(),
        };
        std::fs::write(
            path.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )
        .with_context(|| {
            format!(
                "write experience autoencoder checkpoint metadata {}",
                path.display()
            )
        })?;
        Ok(())
    }

    pub fn predict(
        &self,
        input: &ExperienceEncodeInput,
    ) -> Result<ExperienceAutoencoderPrediction> {
        let features = self.checked_features(input)?;
        let tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let (z, decoded) = self.model.forward(tensor);
        Ok(ExperienceAutoencoderPrediction {
            encoded: tensor_to_experience_encode_output(z.inner(), self.z_dim)?,
            decoded: tensor_to_experience_decode_output(
                decoded.inner(),
                self.output_dim,
                self.decode_lengths,
            )?,
        })
    }

    pub fn encode(&self, input: &ExperienceEncodeInput) -> Result<ExperienceEncodeOutput> {
        Ok(self.predict(input)?.encoded)
    }

    pub fn train_step(
        &mut self,
        input: &ExperienceEncodeInput,
        target: &ExperienceDecodeOutput,
    ) -> Result<ExperienceAutoencoderTrainStats> {
        let features = self.checked_features(input)?;
        let target_values = experience_decode_target_values(target, self.output_dim);
        let input_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let target_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(target_values, [1, self.output_dim]),
            &self.device,
        );
        let (_z, decoded) = self.model.forward(input_tensor);
        let loss = MseLoss::new().forward(decoded, target_tensor, Reduction::Mean);
        let loss_value = loss.clone().inner().into_data().to_vec::<f32>()?[0];
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = self
            .optimizer
            .step(self.learning_rate, self.model.clone(), grads);
        self.samples_seen = self.samples_seen.saturating_add(1);
        let improved = self.best_loss.map(|best| loss_value < best).unwrap_or(true);
        if improved {
            self.best_loss = Some(loss_value);
        }
        Ok(ExperienceAutoencoderTrainStats {
            loss: loss_value,
            samples_seen: self.samples_seen,
            improved,
        })
    }

    pub fn shadow_compare(
        &mut self,
        t_ms: TimeMs,
        input: &ExperienceEncodeInput,
        target: &ExperienceDecodeOutput,
        baseline_z: &[f32],
        selected: String,
    ) -> Result<ExperienceAutoencoderShadowMetric> {
        let prediction = self.predict(input)?;
        let loss = mse_experience_decode_output_target(&prediction.decoded, target);

        fn l2_norm(z: &[f32]) -> f32 {
            let sum: f32 = z.iter().map(|&x| x * x).sum();
            sum.sqrt()
        }

        fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
            let len = a.len().max(b.len());
            if len == 0 {
                return 0.0;
            }
            let sum: f32 = (0..len)
                .map(|idx| {
                    let delta = a.get(idx).copied().unwrap_or_default()
                        - b.get(idx).copied().unwrap_or_default();
                    delta * delta
                })
                .sum();
            sum.sqrt()
        }

        let baseline_z_norm = l2_norm(baseline_z);
        let model_z = &prediction.encoded.z;
        let model_z_norm = l2_norm(model_z);
        let z_disagreement = euclidean_distance(baseline_z, model_z);

        Ok(ExperienceAutoencoderShadowMetric {
            t_ms,
            baseline_z_norm,
            model_z_norm,
            z_disagreement,
            reconstruction_loss: loss,
            selected,
        })
    }

    fn checked_features(&self, input: &ExperienceEncodeInput) -> Result<Vec<f32>> {
        let mut features = input.flat_features();
        if features.len() != self.input_dim {
            return Err(anyhow!(
                "experience autoencoder input dimension mismatch: got {}, expected {}",
                features.len(),
                self.input_dim
            ));
        }
        for value in &mut features {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
        Ok(features)
    }
}
