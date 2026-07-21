impl DangerNetTrainer<DangerAutodiffBackend> {
    pub fn new(input_dim: usize) -> Self {
        Self::with_device(input_dim, Default::default())
    }

    pub fn load_checkpoint(path: impl AsRef<Path>, input_dim: usize) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_danger_metadata(path)?;
        if metadata.input_dim != input_dim {
            return Err(anyhow!(
                "danger checkpoint input dimension mismatch at {}: metadata has {}, runtime expected {}",
                path.display(),
                metadata.input_dim,
                input_dim
            ));
        }

        let device = Default::default();
        let model = DangerNet::init(input_dim, &device).load_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
            &device,
        )?;
        Ok(Self {
            model,
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            learning_rate: 0.03,
            samples_seen: metadata.samples_seen,
            best_loss: metadata.best_loss,
        })
    }
}

impl<B: AutodiffBackend> DangerNetTrainer<B> {
    pub fn with_device(input_dim: usize, device: B::Device) -> Self {
        Self {
            model: DangerNet::init(input_dim, &device),
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            learning_rate: 0.03,
            samples_seen: 0,
            best_loss: None,
        }
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
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
            .with_context(|| format!("create danger checkpoint dir {}", path.display()))?;
        self.model.clone().save_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
        )?;
        let metadata = DangerModelMetadata {
            input_dim: self.input_dim,
            samples_seen: self.samples_seen,
            best_loss: self.best_loss,
            created_at_ms: now_ms(),
        };
        std::fs::write(
            path.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )
        .with_context(|| format!("write danger checkpoint metadata {}", path.display()))?;
        Ok(())
    }

    pub fn predict(&self, input: &DangerInput) -> Result<DangerOutput> {
        let features = self.checked_features(input)?;
        let tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let output = self.model.forward(tensor).inner();
        tensor_to_danger_output(output)
    }

    pub fn train_step(
        &mut self,
        input: &DangerInput,
        target: &DangerTarget,
    ) -> Result<DangerTrainStats> {
        let features = self.checked_features(input)?;
        let target_values = target.risks();
        let input_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let target_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(target_values.to_vec(), [1, 4]),
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
        Ok(DangerTrainStats {
            loss: loss_value,
            samples_seen: self.samples_seen,
            improved,
        })
    }

    pub fn shadow_compare(
        &mut self,
        observed_at_ms: u64,
        now: &Now,
        input: &DangerInput,
        target: &DangerTarget,
    ) -> Result<DangerShadowMetric> {
        let hardcoded = HardcodedDangerPredictor.predict_from_now(now, input);
        let model = self.predict(input)?;
        let loss = mse_output_target(model, *target);
        Ok(DangerShadowMetric {
            observed_at_ms,
            hardcoded,
            model,
            target: *target,
            loss,
        })
    }

    fn checked_features(&self, input: &DangerInput) -> Result<Vec<f32>> {
        let mut features = input.flat_features();
        if features.len() != self.input_dim {
            return Err(anyhow!(
                "danger input dimension mismatch: got {}, expected {}",
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

impl ChargeNetTrainer<ChargeAutodiffBackend> {
    pub fn new(input_dim: usize) -> Self {
        Self::with_device(input_dim, Default::default())
    }

    pub fn load_checkpoint(path: impl AsRef<Path>, input_dim: usize) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_charge_metadata(path)?;
        if metadata.input_dim != input_dim {
            return Err(anyhow!(
                "charge checkpoint input dimension mismatch at {}: metadata has {}, runtime expected {}",
                path.display(),
                metadata.input_dim,
                input_dim
            ));
        }

        let device = Default::default();
        let model = ChargeNet::init(input_dim, &device).load_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
            &device,
        )?;
        Ok(Self {
            model,
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            learning_rate: 0.03,
            samples_seen: metadata.samples_seen,
            best_loss: metadata.best_loss,
        })
    }
}

impl<B: AutodiffBackend> ChargeNetTrainer<B> {
    pub fn with_device(input_dim: usize, device: B::Device) -> Self {
        Self {
            model: ChargeNet::init(input_dim, &device),
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            learning_rate: 0.03,
            samples_seen: 0,
            best_loss: None,
        }
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
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
            .with_context(|| format!("create charge checkpoint dir {}", path.display()))?;
        self.model.clone().save_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
        )?;
        let metadata = ChargeModelMetadata {
            input_dim: self.input_dim,
            samples_seen: self.samples_seen,
            best_loss: self.best_loss,
            created_at_ms: now_ms(),
        };
        std::fs::write(
            path.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )
        .with_context(|| format!("write charge checkpoint metadata {}", path.display()))?;
        Ok(())
    }

    pub fn predict(&self, input: &ChargeInput) -> Result<ChargeOutput> {
        let features = self.checked_features(input)?;
        let tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let output = self.model.forward(tensor).inner();
        tensor_to_charge_output(output)
    }

    pub fn train_step(
        &mut self,
        input: &ChargeInput,
        target: &ChargeTarget,
    ) -> Result<ChargeTrainStats> {
        let features = self.checked_features(input)?;
        let target_values = charge_target_train_values(target);
        let input_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let target_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(target_values.to_vec(), [1, 3]),
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
        Ok(ChargeTrainStats {
            loss: loss_value,
            samples_seen: self.samples_seen,
            improved,
        })
    }

    pub fn shadow_compare(
        &mut self,
        observed_at_ms: u64,
        now: &Now,
        input: &ChargeInput,
        target: &ChargeTarget,
    ) -> Result<ChargeShadowMetric> {
        let hardcoded = HardcodedChargePredictor.predict_from_now(now, input);
        let model = self.predict(input)?;
        let loss = mse_charge_output_target(model, *target);
        Ok(ChargeShadowMetric {
            observed_at_ms,
            hardcoded,
            model,
            target: *target,
            loss,
        })
    }

    fn checked_features(&self, input: &ChargeInput) -> Result<Vec<f32>> {
        let mut features = input.flat_features();
        if features.len() != self.input_dim {
            return Err(anyhow!(
                "charge input dimension mismatch: got {}, expected {}",
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

impl ActionValueNetTrainer<ActionValueAutodiffBackend> {
    pub fn new(input_dim: usize) -> Self {
        Self::with_device(input_dim, Default::default())
    }

    pub fn load_checkpoint(path: impl AsRef<Path>, input_dim: usize) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_action_value_metadata(path)?;
        if metadata.input_dim != input_dim {
            return Err(anyhow!(
                "action-value checkpoint input dimension mismatch at {}: metadata has {}, runtime expected {}",
                path.display(),
                metadata.input_dim,
                input_dim
            ));
        }

        let device = Default::default();
        let model = ActionValueNet::init(input_dim, &device).load_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
            &device,
        )?;
        Ok(Self {
            model,
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            learning_rate: 0.03,
            samples_seen: metadata.samples_seen,
            best_loss: metadata.best_loss,
        })
    }
}

impl<B: AutodiffBackend> ActionValueNetTrainer<B> {
    pub fn with_device(input_dim: usize, device: B::Device) -> Self {
        Self {
            model: ActionValueNet::init(input_dim, &device),
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            learning_rate: 0.03,
            samples_seen: 0,
            best_loss: None,
        }
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
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
            .with_context(|| format!("create action-value checkpoint dir {}", path.display()))?;
        self.model.clone().save_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
        )?;
        let metadata = ActionValueModelMetadata {
            input_dim: self.input_dim,
            samples_seen: self.samples_seen,
            best_loss: self.best_loss,
            created_at_ms: now_ms(),
        };
        std::fs::write(
            path.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )
        .with_context(|| format!("write action-value checkpoint metadata {}", path.display()))?;
        Ok(())
    }

    pub fn predict(&self, input: &ActionValueInput) -> Result<ActionValueOutput> {
        let features = self.checked_features(input)?;
        let tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let output = self.model.forward(tensor).inner();
        tensor_to_action_value_output(output)
    }

    pub fn train_step(
        &mut self,
        input: &ActionValueInput,
        target: &ActionValueTarget,
    ) -> Result<ActionValueTrainStats> {
        let features = self.checked_features(input)?;
        let target_values = action_value_target_train_values(target);
        let input_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let target_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(target_values.to_vec(), [1, 2]),
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
        Ok(ActionValueTrainStats {
            loss: loss_value,
            samples_seen: self.samples_seen,
            improved,
        })
    }

    pub fn shadow_compare(
        &mut self,
        observed_at_ms: u64,
        now: &Now,
        input: &ActionValueInput,
        target: &ActionValueTarget,
    ) -> Result<ActionValueShadowMetric> {
        let hardcoded = HardcodedActionValuePredictor.predict_from_now(now, input);
        let model = self.predict(input)?;
        let loss = mse_action_value_output_target(model, *target);
        Ok(ActionValueShadowMetric {
            observed_at_ms,
            hardcoded,
            model,
            target: *target,
            loss,
        })
    }

    fn checked_features(&self, input: &ActionValueInput) -> Result<Vec<f32>> {
        let mut features = input.flat_features();
        if features.len() != self.input_dim {
            return Err(anyhow!(
                "action-value input dimension mismatch: got {}, expected {}",
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

impl FutureNetTrainer<FutureAutodiffBackend> {
    pub fn new(input_dim: usize, latent_dim: usize) -> Self {
        Self::with_device(input_dim, latent_dim, Default::default())
    }

    pub fn load_checkpoint(
        path: impl AsRef<Path>,
        input_dim: usize,
        latent_dim: usize,
    ) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_future_metadata(path)?;
        if metadata.input_dim != input_dim {
            return Err(anyhow!(
                "future checkpoint input dimension mismatch at {}: metadata has {}, runtime expected {}",
                path.display(),
                metadata.input_dim,
                input_dim
            ));
        }
        if metadata.latent_dim != latent_dim || metadata.output_dim != latent_dim {
            return Err(anyhow!(
                "future checkpoint latent dimension mismatch at {}: metadata has latent/output {}/{}, runtime expected {}",
                path.display(),
                metadata.latent_dim,
                metadata.output_dim,
                latent_dim
            ));
        }

        let device = Default::default();
        let model = FutureNet::init(input_dim, latent_dim, &device).load_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
            &device,
        )?;
        Ok(Self {
            model,
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            latent_dim,
            learning_rate: 0.03,
            samples_seen: metadata.samples_seen,
            best_loss: metadata.best_loss,
        })
    }
}

impl<B: AutodiffBackend> FutureNetTrainer<B> {
    pub fn with_device(input_dim: usize, latent_dim: usize, device: B::Device) -> Self {
        Self {
            model: FutureNet::init(input_dim, latent_dim, &device),
            optimizer: SgdConfig::new().init(),
            device,
            input_dim,
            latent_dim,
            learning_rate: 0.03,
            samples_seen: 0,
            best_loss: None,
        }
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
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
            .with_context(|| format!("create future checkpoint dir {}", path.display()))?;
        self.model.clone().save_file(
            path.join("model"),
            &BinFileRecorder::<FullPrecisionSettings>::default(),
        )?;
        let metadata = FutureModelMetadata {
            input_dim: self.input_dim,
            output_dim: self.latent_dim,
            latent_dim: self.latent_dim,
            samples_seen: self.samples_seen,
            best_loss: self.best_loss,
            created_at_ms: now_ms(),
        };
        std::fs::write(
            path.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )
        .with_context(|| format!("write future checkpoint metadata {}", path.display()))?;
        Ok(())
    }

    pub fn predict(&self, input: &FutureInput) -> Result<FuturePrediction> {
        let features = self.checked_features(input)?;
        let tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let output = self.model.forward(tensor).inner();
        tensor_to_future_prediction(output, input.offset_ms, self.latent_dim)
    }

    pub fn train_step(&mut self, input: &FutureInput, target_z: &[f32]) -> Result<TrainStats> {
        let features = self.checked_features(input)?;
        let target_values = future_target_train_values(target_z, self.latent_dim);
        let input_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(features, [1, self.input_dim]), &self.device);
        let target_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(target_values, [1, self.latent_dim]),
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
        Ok(TrainStats {
            loss: loss_value,
            samples_seen: self.samples_seen,
            improved,
        })
    }

    pub fn shadow_compare(
        &self,
        input: &FutureInput,
        hardcoded: &FuturePrediction,
        target_z: &[f32],
    ) -> Result<FutureShadowMetric> {
        let model = self.predict(input)?;
        let hardcoded_error = mse_vec_target(&hardcoded.predicted_z, target_z);
        let model_error = mse_vec_target(&model.predicted_z, target_z);
        Ok(FutureShadowMetric {
            t_ms: input.latent.t_ms,
            offset_ms: input.offset_ms,
            hardcoded_error,
            model_error,
            selected_error: hardcoded_error,
            model_loss: model_error,
        })
    }

    fn checked_features(&self, input: &FutureInput) -> Result<Vec<f32>> {
        let mut features = input.flat_features();
        if features.len() != self.input_dim {
            return Err(anyhow!(
                "future input dimension mismatch: got {}, expected {}",
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
