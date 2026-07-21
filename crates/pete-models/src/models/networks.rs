#[derive(Module, Debug)]
pub struct DangerNet<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
}

#[derive(Module, Debug)]
pub struct ChargeNet<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
}

#[derive(Module, Debug)]
pub struct ActionValueNet<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
}

#[derive(Module, Debug)]
pub struct FutureNet<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
}

#[derive(Module, Debug)]
pub struct EyeNextNet<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
}

#[derive(Module, Debug)]
pub struct EarNextNet<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
}

#[derive(Module, Debug)]
pub struct ExperienceAutoencoderNet<B: Backend> {
    encoder_input: Linear<B>,
    encoder_hidden: Linear<B>,
    z: Linear<B>,
    decoder_hidden: Linear<B>,
    decoder_output: Linear<B>,
}

impl<B: Backend> ChargeNet<B> {
    pub fn init(input_dim: usize, device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(input_dim, 32).init(device),
            hidden: LinearConfig::new(32, 16).init(device),
            output: LinearConfig::new(16, 3).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.input.forward(input));
        let x = activation::relu(self.hidden.forward(x));
        activation::sigmoid(self.output.forward(x))
    }
}

impl<B: Backend> DangerNet<B> {
    pub fn init(input_dim: usize, device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(input_dim, 32).init(device),
            hidden: LinearConfig::new(32, 16).init(device),
            output: LinearConfig::new(16, 4).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.input.forward(input));
        let x = activation::relu(self.hidden.forward(x));
        activation::sigmoid(self.output.forward(x))
    }
}

impl<B: Backend> ActionValueNet<B> {
    pub fn init(input_dim: usize, device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(input_dim, 64).init(device),
            hidden: LinearConfig::new(64, 32).init(device),
            output: LinearConfig::new(32, 2).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.input.forward(input));
        let x = activation::relu(self.hidden.forward(x));
        activation::sigmoid(self.output.forward(x))
    }
}

impl<B: Backend> FutureNet<B> {
    pub fn init(input_dim: usize, latent_dim: usize, device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(input_dim, 128).init(device),
            hidden: LinearConfig::new(128, 64).init(device),
            output: LinearConfig::new(64, latent_dim).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.input.forward(input));
        let x = activation::relu(self.hidden.forward(x));
        activation::sigmoid(self.output.forward(x))
    }
}

impl<B: Backend> EyeNextNet<B> {
    pub fn init(input_dim: usize, output_dim: usize, device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(input_dim, 128).init(device),
            hidden: LinearConfig::new(128, 128).init(device),
            output: LinearConfig::new(128, output_dim).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.input.forward(input));
        let x = activation::relu(self.hidden.forward(x));
        activation::sigmoid(self.output.forward(x))
    }
}

impl<B: Backend> EarNextNet<B> {
    pub fn init(input_dim: usize, output_dim: usize, device: &B::Device) -> Self {
        Self {
            input: LinearConfig::new(input_dim, 64).init(device),
            hidden: LinearConfig::new(64, 32).init(device),
            output: LinearConfig::new(32, output_dim).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = activation::relu(self.input.forward(input));
        let x = activation::relu(self.hidden.forward(x));
        activation::sigmoid(self.output.forward(x))
    }
}

impl<B: Backend> ExperienceAutoencoderNet<B> {
    pub fn init(input_dim: usize, z_dim: usize, output_dim: usize, device: &B::Device) -> Self {
        Self {
            encoder_input: LinearConfig::new(input_dim, 96).init(device),
            encoder_hidden: LinearConfig::new(96, 48).init(device),
            z: LinearConfig::new(48, z_dim).init(device),
            decoder_hidden: LinearConfig::new(z_dim, 48).init(device),
            decoder_output: LinearConfig::new(48, output_dim).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = activation::relu(self.encoder_input.forward(input));
        let x = activation::relu(self.encoder_hidden.forward(x));
        let z = activation::sigmoid(self.z.forward(x));
        let decoded = activation::relu(self.decoder_hidden.forward(z.clone()));
        let decoded = activation::sigmoid(self.decoder_output.forward(decoded));
        (z, decoded)
    }
}

pub type DangerBackend = NdArray<f32>;
pub type DangerAutodiffBackend = Autodiff<DangerBackend>;
pub type ChargeBackend = NdArray<f32>;
pub type ChargeAutodiffBackend = Autodiff<ChargeBackend>;
pub type ActionValueBackend = NdArray<f32>;
pub type ActionValueAutodiffBackend = Autodiff<ActionValueBackend>;
pub type FutureBackend = NdArray<f32>;
pub type FutureAutodiffBackend = Autodiff<FutureBackend>;
pub type EyeNextBackend = NdArray<f32>;
pub type EyeNextAutodiffBackend = Autodiff<EyeNextBackend>;
pub type EarNextBackend = NdArray<f32>;
pub type EarNextAutodiffBackend = Autodiff<EarNextBackend>;
pub type ExperienceAutoencoderBackend = NdArray<f32>;
pub type ExperienceAutoencoderAutodiffBackend = Autodiff<ExperienceAutoencoderBackend>;

pub struct DangerNetTrainer<B: AutodiffBackend = DangerAutodiffBackend> {
    model: DangerNet<B>,
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, DangerNet<B>, B>,
    device: B::Device,
    input_dim: usize,
    learning_rate: f64,
    samples_seen: u64,
    best_loss: Option<f32>,
}

pub struct ChargeNetTrainer<B: AutodiffBackend = ChargeAutodiffBackend> {
    model: ChargeNet<B>,
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, ChargeNet<B>, B>,
    device: B::Device,
    input_dim: usize,
    learning_rate: f64,
    samples_seen: u64,
    best_loss: Option<f32>,
}

pub struct ActionValueNetTrainer<B: AutodiffBackend = ActionValueAutodiffBackend> {
    model: ActionValueNet<B>,
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, ActionValueNet<B>, B>,
    device: B::Device,
    input_dim: usize,
    learning_rate: f64,
    samples_seen: u64,
    best_loss: Option<f32>,
}

pub struct FutureNetTrainer<B: AutodiffBackend = FutureAutodiffBackend> {
    model: FutureNet<B>,
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, FutureNet<B>, B>,
    device: B::Device,
    input_dim: usize,
    latent_dim: usize,
    learning_rate: f64,
    samples_seen: u64,
    best_loss: Option<f32>,
}

pub struct EyeNextNetTrainer<B: AutodiffBackend = EyeNextAutodiffBackend> {
    model: EyeNextNet<B>,
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, EyeNextNet<B>, B>,
    device: B::Device,
    input_dim: usize,
    output_dim: usize,
    width: u32,
    height: u32,
    learning_rate: f64,
    samples_seen: u64,
    best_loss: Option<f32>,
}

pub struct EarNextNetTrainer<B: AutodiffBackend = EarNextAutodiffBackend> {
    model: EarNextNet<B>,
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, EarNextNet<B>, B>,
    device: B::Device,
    input_dim: usize,
    output_dim: usize,
    sample_rate_hz: u32,
    channels: u16,
    learning_rate: f64,
    samples_seen: u64,
    best_loss: Option<f32>,
}

pub struct ExperienceAutoencoderTrainer<B: AutodiffBackend = ExperienceAutoencoderAutodiffBackend> {
    model: ExperienceAutoencoderNet<B>,
    optimizer: OptimizerAdaptor<Sgd<B::InnerBackend>, ExperienceAutoencoderNet<B>, B>,
    device: B::Device,
    input_dim: usize,
    z_dim: usize,
    output_dim: usize,
    decode_lengths: ExperienceDecodeFeatureLengths,
    learning_rate: f64,
    samples_seen: u64,
    best_loss: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DangerModelMetadata {
    pub input_dim: usize,
    pub samples_seen: u64,
    pub best_loss: Option<f32>,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChargeModelMetadata {
    pub input_dim: usize,
    pub samples_seen: u64,
    pub best_loss: Option<f32>,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionValueModelMetadata {
    pub input_dim: usize,
    pub samples_seen: u64,
    pub best_loss: Option<f32>,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FutureModelMetadata {
    pub input_dim: usize,
    pub output_dim: usize,
    pub latent_dim: usize,
    pub samples_seen: u64,
    pub best_loss: Option<f32>,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EyeNextModelMetadata {
    pub input_dim: usize,
    pub output_dim: usize,
    pub width: u32,
    pub height: u32,
    pub samples_seen: u64,
    pub best_loss: Option<f32>,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EarNextModelMetadata {
    pub input_dim: usize,
    pub output_dim: usize,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples_seen: u64,
    pub best_loss: Option<f32>,
    pub created_at_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExperienceAutoencoderMetadata {
    pub input_dim: usize,
    pub z_dim: usize,
    pub output_dim: usize,
    pub decode_lengths: ExperienceDecodeFeatureLengths,
    pub samples_seen: u64,
    pub best_loss: Option<f32>,
    pub created_at_ms: u64,
}
