#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LinuxWorldConfig {
    pub camera_device: Option<String>,
    pub gps_serial_port: Option<String>,
    pub gps_baud_rate: u32,
    pub microphone_name: Option<String>,
    pub audio_sample_rate_hz: u32,
    pub audio_channels: u16,
}

#[cfg(feature = "linux-hardware")]
pub struct LinuxWorld {
    snapshot: WorldSnapshot,
    microphone: Option<CpalMicrophone>,
    camera: Option<V4lCamera>,
    gps: Option<Ublox7Gps>,
}

#[cfg(feature = "linux-hardware")]
impl LinuxWorld {
    pub fn new(config: LinuxWorldConfig) -> Result<Self> {
        let microphone = CpalMicrophone::new(
            config.microphone_name.as_deref(),
            config.audio_sample_rate_hz.max(8_000),
            config.audio_channels.max(1),
        )
        .ok();
        let camera = config
            .camera_device
            .as_deref()
            .map(V4lCamera::new)
            .transpose()?;
        let gps = config
            .gps_serial_port
            .as_deref()
            .map(|port| Ublox7Gps::new(port, config.gps_baud_rate.max(9_600)))
            .transpose()?;
        Ok(Self {
            snapshot: WorldSnapshot::default(),
            microphone,
            camera,
            gps,
        })
    }

    pub fn snapshot_ref(&self) -> &WorldSnapshot {
        &self.snapshot
    }

    fn refresh_hardware(&mut self) -> Result<()> {
        if let Some(camera) = self.camera.as_mut() {
            if let Ok(frame) = camera.capture_frame() {
                self.snapshot.eye.frames = vec![bytes_to_unit_signal(&frame.bytes)];
                self.snapshot.eye_frame = Some(frame);
            }
        }

        if let Some(microphone) = self.microphone.as_ref() {
            if let Some(frame) = microphone.latest_frame() {
                self.snapshot.ear.features = vec![pcm_to_unit_signal(&frame.samples)];
                self.snapshot.ear_pcm = Some(frame);
            }
        }

        if let Some(gps) = self.gps.as_mut() {
            if let Some(fix) = gps.try_read_fix()? {
                self.snapshot.gps = Some(fix);
            }
        }

        Ok(())
    }
}

#[cfg(feature = "linux-hardware")]
#[async_trait]
impl World for LinuxWorld {
    async fn snapshot(&mut self) -> Result<WorldSnapshot> {
        self.refresh_hardware()?;
        Ok(self.snapshot.clone())
    }

    async fn apply_update(&mut self, update: WorldUpdate) -> Result<()> {
        update.apply_to(&mut self.snapshot);
        Ok(())
    }
}

#[cfg(feature = "linux-hardware")]
pub struct CpalMicrophone {
    latest: Arc<Mutex<Option<PcmAudioFrame>>>,
    _stream: cpal::Stream,
}

#[cfg(feature = "linux-hardware")]
unsafe impl Send for CpalMicrophone {}
#[cfg(feature = "linux-hardware")]
unsafe impl Sync for CpalMicrophone {}

#[cfg(feature = "linux-hardware")]
impl CpalMicrophone {
    pub fn new(preferred_name: Option<&str>, sample_rate_hz: u32, channels: u16) -> Result<Self> {
        let host = cpal::default_host();
        let mut errors = Vec::new();
        for device in input_device_candidates(&host, preferred_name)? {
            let device_name = device.name().unwrap_or_else(|_| "<unnamed>".to_string());
            match Self::open_device(device, sample_rate_hz, channels) {
                Ok(microphone) => {
                    eprintln!("microphone input active: {device_name}");
                    return Ok(microphone);
                }
                Err(error) => errors.push(format!("{device_name}: {error}")),
            }
        }
        anyhow::bail!(
            "no usable CPAL input device for requested mic {:?}; tried {}",
            preferred_name.unwrap_or("default"),
            if errors.is_empty() {
                "no input devices".to_string()
            } else {
                errors.join("; ")
            }
        )
    }

    fn open_device(device: cpal::Device, sample_rate_hz: u32, channels: u16) -> Result<Self> {
        let supported = select_input_config(&device, sample_rate_hz, channels)?;
        let sample_format = supported.sample_format();
        let config = supported.config();
        let actual_sample_rate_hz = config.sample_rate.0;
        let actual_channels = config.channels;
        let latest = Arc::new(Mutex::new(None));
        let shared = Arc::clone(&latest);
        let err_fn = |err: cpal::StreamError| {
            let message = err.to_string();
            if !is_muted_cpal_input_stream_error(&message) {
                eprintln!("cpal input stream error: {message}");
            }
        };
        let stream = match sample_format {
            cpal::SampleFormat::I16 => device.build_input_stream(
                &config,
                move |data: &[i16], _| {
                    store_i16_pcm_frame(&shared, data, actual_sample_rate_hz, actual_channels)
                },
                err_fn,
                None,
            )?,
            cpal::SampleFormat::U16 => device.build_input_stream(
                &config,
                move |data: &[u16], _| {
                    let pcm = data
                        .iter()
                        .map(|sample| (*sample as i32 - 32_768) as i16)
                        .collect::<Vec<_>>();
                    store_i16_pcm_frame(&shared, &pcm, actual_sample_rate_hz, actual_channels);
                },
                err_fn,
                None,
            )?,
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config,
                move |data: &[f32], _| {
                    let pcm = data
                        .iter()
                        .map(|sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
                        .collect::<Vec<_>>();
                    store_i16_pcm_frame(&shared, &pcm, actual_sample_rate_hz, actual_channels);
                },
                err_fn,
                None,
            )?,
            other => {
                anyhow::bail!("unsupported CPAL sample format: {:?}", other);
            }
        };
        stream.play()?;
        Ok(Self {
            latest,
            _stream: stream,
        })
    }

    pub fn latest_frame(&self) -> Option<PcmAudioFrame> {
        self.latest.lock().ok().and_then(|guard| guard.clone())
    }
}

#[cfg(any(feature = "linux-hardware", test))]
fn is_muted_cpal_input_stream_error(message: &str) -> bool {
    message.contains("snd_pcm_poll_descriptors") && message.contains("Unknown errno (-32)")
}

#[cfg(feature = "linux-hardware")]
pub struct V4lCamera {
    stream: MmapStream<'static>,
    format: Format,
}

#[cfg(feature = "linux-hardware")]
impl V4lCamera {
    pub fn new(path: &str) -> Result<Self> {
        let mut device = Device::with_path(path)?;
        configure_camera_format(&mut device);
        let format = device
            .format()
            .with_context(|| format!("failed to read V4L camera format for {path}"))?;
        let device = Box::leak(Box::new(device));
        let stream = MmapStream::with_buffers(device, Type::VideoCapture, 4)
            .with_context(|| format!("failed to create V4L mmap stream for {path}"))?;
        Ok(Self { stream, format })
    }

    pub fn capture_frame(&mut self) -> Result<EyeFrame> {
        let (bytes, _) = self.stream.next()?;
        Ok(EyeFrame {
            captured_at_ms: unix_time_ms(),
            width: self.format.width,
            height: self.format.height,
            format: eye_frame_format_from_fourcc(self.format.fourcc.str().unwrap_or_default()),
            bytes: bytes.to_vec(),
            source: Some("real-camera".to_string()),
        })
    }
}

#[cfg(feature = "linux-hardware")]
fn eye_frame_format_from_fourcc(fourcc: &str) -> EyeFrameFormat {
    match fourcc.trim_end_matches('\0') {
        "GREY" | "Y800" => EyeFrameFormat::Gray8,
        "RGB3" => EyeFrameFormat::Rgb8,
        "BGR3" => EyeFrameFormat::Bgr8,
        "YUYV" | "YUY2" => EyeFrameFormat::Yuyv422,
        "UYVY" => EyeFrameFormat::Uyvy422,
        "GRBG" => EyeFrameFormat::BayerGrbg8,
        "RGGB" => EyeFrameFormat::BayerRggb8,
        "BGGR" => EyeFrameFormat::BayerBggr8,
        "GBRG" => EyeFrameFormat::BayerGbrg8,
        "MJPG" | "JPEG" => EyeFrameFormat::Mjpeg,
        other => EyeFrameFormat::Unknown(other.to_string()),
    }
}

#[cfg(feature = "linux-hardware")]
fn configure_camera_format(device: &mut Device) {
    let candidates = [
        (640, 480, *b"UYVY"),
        (640, 480, *b"GRBG"),
        (320, 240, *b"MJPG"),
        (320, 240, *b"YUYV"),
        (640, 480, *b"MJPG"),
        (640, 480, *b"YUYV"),
        (640, 480, *b"RGB3"),
        (640, 480, *b"BGR3"),
        (640, 480, *b"GREY"),
        (1280, 1024, *b"GRBG"),
    ];
    for (width, height, fourcc) in candidates {
        let format = Format::new(width, height, FourCC::new(&fourcc));
        if device.set_format(&format).is_ok() {
            return;
        }
    }
}

#[cfg(feature = "linux-hardware")]
pub struct Ublox7Gps {
    port: Box<dyn SerialPort>,
    buffer: Vec<u8>,
}

#[cfg(feature = "linux-hardware")]
impl Ublox7Gps {
    pub fn new(path: &str, baud_rate: u32) -> Result<Self> {
        let port = serialport::new(path, baud_rate)
            .timeout(Duration::from_millis(25))
            .open()?;
        Ok(Self {
            port,
            buffer: Vec::new(),
        })
    }

    pub fn try_read_fix(&mut self) -> Result<Option<GpsSense>> {
        let mut chunk = [0u8; 512];
        match self.port.read(&mut chunk) {
            Ok(count) => self.buffer.extend_from_slice(&chunk[..count]),
            Err(error) if error.kind() == ErrorKind::TimedOut => {}
            Err(error) => return Err(error.into()),
        }

        while let Some(position) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let line = self.buffer.drain(..=position).collect::<Vec<_>>();
            if let Ok(text) = std::str::from_utf8(&line) {
                if let Some(fix) = parse_nmea_fix(text.trim()) {
                    return Ok(Some(fix));
                }
            }
        }

        Ok(None)
    }
}

#[cfg(feature = "linux-hardware")]
pub struct Mpu6050Imu {
    bus: File,
}

#[cfg(feature = "linux-hardware")]
impl Mpu6050Imu {
    pub fn new(device: &str) -> Result<Self> {
        let spec = parse_mpu6050_device_spec(device)?;
        let bus = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&spec.path)?;
        set_i2c_slave(&bus, spec.address)?;
        let mut imu = Self { bus };
        imu.write_register(MPU6050_PWR_MGMT_1, 0x00)?;
        imu.write_register(MPU6050_ACCEL_CONFIG, 0x00)?;
        imu.write_register(MPU6050_GYRO_CONFIG, 0x00)?;
        Ok(imu)
    }

    pub fn read_sense(&mut self) -> Result<ImuSense> {
        let mut bytes = [0u8; 14];
        self.read_registers(MPU6050_ACCEL_XOUT_H, &mut bytes)?;
        Ok(mpu6050_samples_to_imu(bytes, unix_time_ms()))
    }

    fn write_register(&mut self, register: u8, value: u8) -> Result<()> {
        self.bus.write_all(&[register, value])?;
        Ok(())
    }

    fn read_registers(&mut self, register: u8, buffer: &mut [u8]) -> Result<()> {
        self.bus.write_all(&[register])?;
        self.bus.read_exact(buffer)?;
        Ok(())
    }
}

#[cfg(feature = "linux-hardware")]
fn set_i2c_slave(bus: &File, address: u16) -> Result<()> {
    const I2C_SLAVE: libc::c_ulong = 0x0703;
    let result = unsafe { libc::ioctl(bus.as_raw_fd(), I2C_SLAVE, libc::c_ulong::from(address)) };
    if result < 0 {
        return Err(std::io::Error::last_os_error())
            .with_context(|| format!("failed to select I2C slave address 0x{address:02x}"));
    }
    Ok(())
}

#[cfg(any(feature = "linux-hardware", test))]
#[derive(Clone, Debug, PartialEq, Eq)]
struct Mpu6050DeviceSpec {
    path: String,
    address: u16,
}

#[cfg(any(feature = "linux-hardware", test))]
const MPU6050_DEFAULT_ADDRESS: u16 = 0x68;
#[cfg(feature = "linux-hardware")]
const MPU6050_PWR_MGMT_1: u8 = 0x6b;
#[cfg(feature = "linux-hardware")]
const MPU6050_ACCEL_CONFIG: u8 = 0x1c;
#[cfg(feature = "linux-hardware")]
const MPU6050_GYRO_CONFIG: u8 = 0x1b;
#[cfg(feature = "linux-hardware")]
const MPU6050_ACCEL_XOUT_H: u8 = 0x3b;

#[cfg(any(feature = "linux-hardware", test))]
fn parse_mpu6050_device_spec(device: &str) -> Result<Mpu6050DeviceSpec> {
    let (path, address) = device
        .rsplit_once('@')
        .or_else(|| device.rsplit_once(':'))
        .map(|(path, address)| (path, Some(address)))
        .unwrap_or((device, None));
    let address = address
        .map(parse_i2c_address)
        .transpose()?
        .unwrap_or(MPU6050_DEFAULT_ADDRESS);
    if path.trim().is_empty() {
        anyhow::bail!("MPU-6050 I2C device path is empty");
    }
    if !(0x03..=0x77).contains(&address) {
        anyhow::bail!("I2C address 0x{address:02x} is outside the 7-bit usable range");
    }
    Ok(Mpu6050DeviceSpec {
        path: path.to_string(),
        address,
    })
}

#[cfg(any(feature = "linux-hardware", test))]
fn parse_i2c_address(value: &str) -> Result<u16> {
    let trimmed = value.trim();
    let digits = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"));
    match digits {
        Some(hex) => u16::from_str_radix(hex, 16).map_err(anyhow::Error::from),
        None => trimmed.parse::<u16>().map_err(anyhow::Error::from),
    }
    .with_context(|| format!("invalid I2C address `{value}`"))
}

#[cfg(any(feature = "linux-hardware", test))]
fn mpu6050_samples_to_imu(bytes: [u8; 14], captured_at_ms: TimeMs) -> ImuSense {
    let accel_x = read_i16_be(bytes[0], bytes[1]) as f32 / 16_384.0;
    let accel_y = read_i16_be(bytes[2], bytes[3]) as f32 / 16_384.0;
    let accel_z = read_i16_be(bytes[4], bytes[5]) as f32 / 16_384.0;
    let gyro_x = (read_i16_be(bytes[8], bytes[9]) as f32 / 131.0).to_radians();
    let gyro_y = (read_i16_be(bytes[10], bytes[11]) as f32 / 131.0).to_radians();
    let gyro_z = (read_i16_be(bytes[12], bytes[13]) as f32 / 131.0).to_radians();

    let roll_rad = accel_y.atan2(accel_z);
    let pitch_rad = (-accel_x).atan2((accel_y * accel_y + accel_z * accel_z).sqrt());

    ImuSense {
        schema_version: 1,
        captured_at_ms,
        orientation: vec![roll_rad, pitch_rad],
        acceleration: vec![accel_x, accel_y, accel_z],
        angular_velocity: vec![gyro_x, gyro_y, gyro_z],
    }
}

#[cfg(any(feature = "linux-hardware", test))]
fn read_i16_be(high: u8, low: u8) -> i16 {
    i16::from_be_bytes([high, low])
}

#[cfg(feature = "linux-hardware")]
fn input_device_candidates(
    host: &cpal::Host,
    preferred_name: Option<&str>,
) -> Result<Vec<cpal::Device>> {
    let mut candidates = Vec::new();
    let devices = host.input_devices()?.collect::<Vec<_>>();
    if let Some(name) = preferred_name {
        candidates.extend(devices.iter().filter_map(|device| {
            let device_name = device.name().ok()?;
            if device_name == name || device_name.contains(name) {
                Some(device.clone())
            } else {
                None
            }
        }));
        if candidates.is_empty() {
            anyhow::bail!("requested CPAL input device '{name}' was not found");
        }
        return Ok(candidates);
    }
    for device in devices {
        let name = device.name().unwrap_or_default();
        let already_added = candidates
            .iter()
            .any(|candidate| candidate.name().unwrap_or_default() == name);
        if !already_added {
            candidates.push(device);
        }
    }
    if let Some(default) = host.default_input_device() {
        let name = default.name().unwrap_or_default();
        let already_added = candidates
            .iter()
            .any(|candidate| candidate.name().unwrap_or_default() == name);
        if !already_added {
            candidates.push(default);
        }
    }
    Ok(candidates)
}

#[cfg(feature = "linux-hardware")]
fn select_input_config(
    device: &cpal::Device,
    sample_rate_hz: u32,
    channels: u16,
) -> Result<cpal::SupportedStreamConfig> {
    let requested_rate = cpal::SampleRate(sample_rate_hz);
    if let Ok(configs) = device.supported_input_configs() {
        let mut fallback = None;
        for config in configs {
            if fallback.is_none() {
                fallback = Some(config.clone().with_max_sample_rate());
            }
            if config.channels() == channels
                && config.min_sample_rate() <= requested_rate
                && config.max_sample_rate() >= requested_rate
            {
                return Ok(config.with_sample_rate(requested_rate));
            }
        }
        if let Some(config) = fallback {
            return Ok(config);
        }
    }
    device
        .default_input_config()
        .context("reading default input config")
}

#[cfg(feature = "linux-hardware")]
fn store_i16_pcm_frame(
    shared: &Arc<Mutex<Option<PcmAudioFrame>>>,
    samples: &[i16],
    sample_rate_hz: u32,
    channels: u16,
) {
    if let Ok(mut guard) = shared.lock() {
        *guard = Some(PcmAudioFrame {
            captured_at_ms: unix_time_ms(),
            sample_rate_hz,
            channels,
            samples: samples.to_vec(),
        });
    }
}

fn pcm_rms(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum = samples
        .iter()
        .map(|sample| {
            let normalized = *sample as f32 / i16::MAX as f32;
            normalized * normalized
        })
        .sum::<f32>();
    (sum / samples.len() as f32).sqrt()
}

fn pcm_duration_ms(sample_count: usize, sample_rate_hz: u32, channels: u16) -> u64 {
    if sample_rate_hz == 0 || channels == 0 {
        return 0;
    }
    let frames = sample_count as u64 / channels as u64;
    frames.saturating_mul(1_000) / sample_rate_hz as u64
}

fn command_transcript_confidence(transcript: &str) -> f32 {
    let words = transcript.split_whitespace().count() as f32;
    (0.55 + words.min(8.0) * 0.04).clamp(0.55, 0.92)
}

const ASR_TRANSCRIPT_VECTOR_DIM: usize = 32;
const ASR_TRANSCRIPT_VECTOR_MODEL: &str = "pete.text.hashing.v1";

fn transcript_vector_artifact(
    transcript: &str,
    sequence: u64,
    start_ms: u64,
    end_ms: u64,
) -> VectorArtifact {
    let source_id = format!("asr-utterance-{sequence}");
    VectorArtifact::new(
        TRANSCRIPT_VECTOR_COLLECTION,
        format!("{source_id}-transcript"),
        text_hash_vector(transcript, ASR_TRANSCRIPT_VECTOR_DIM),
    )
    .with_model(ASR_TRANSCRIPT_VECTOR_MODEL)
    .with_source_id(source_id)
    .with_occurred_at_ms(end_ms.max(start_ms))
}

fn text_hash_vector(text: &str, dim: usize) -> Vec<f32> {
    let dim = dim.max(1);
    let mut vector = vec![0.0_f32; dim];
    let mut token_count = 0.0_f32;
    for token in text
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| !token.is_empty())
    {
        token_count += 1.0;
        let normalized = token.to_ascii_lowercase();
        for ngram in token_ngrams(&normalized) {
            let mut hash = 2166136261_u32;
            for byte in ngram.bytes() {
                hash = hash.wrapping_mul(16777619) ^ u32::from(byte);
            }
            let index = (hash as usize) % dim;
            let sign = if hash & 1 == 0 { 1.0 } else { -1.0 };
            vector[index] += sign;
        }
    }
    vector[0] += (text.chars().count() as f32 / 512.0).clamp(0.0, 1.0);
    if dim > 1 {
        vector[1] += (token_count / 96.0).clamp(0.0, 1.0);
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in &mut vector {
            *value = (*value / norm).clamp(-1.0, 1.0);
        }
    }
    vector
}

fn token_ngrams(token: &str) -> Vec<String> {
    let chars = token.chars().collect::<Vec<_>>();
    if chars.len() <= 3 {
        return vec![token.to_string()];
    }
    let mut ngrams = Vec::new();
    for window in chars.windows(3) {
        ngrams.push(window.iter().collect());
    }
    ngrams.push(token.to_string());
    ngrams
}

fn transcribe_with_command(
    command_line: &str,
    samples: &[i16],
    sample_rate_hz: u32,
    channels: u16,
) -> Result<String> {
    let mut parts = command_line.split_whitespace();
    let Some(program) = parts.next() else {
        anyhow::bail!("ASR command is empty");
    };
    let mut args = parts.map(str::to_string).collect::<Vec<_>>();
    let wav_path = std::env::temp_dir().join(format!(
        "pete-asr-{}-{}.wav",
        std::process::id(),
        unix_time_ms()
    ));
    write_pcm_wav(&wav_path, samples, sample_rate_hz, channels)?;
    args.push(wav_path.to_string_lossy().to_string());
    let output = Command::new(program).args(args).output();
    let _ = std::fs::remove_file(&wav_path);
    let output = output?;
    if !output.status.success() {
        anyhow::bail!(
            "ASR command exited with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn write_pcm_wav(path: &Path, samples: &[i16], sample_rate_hz: u32, channels: u16) -> Result<()> {
    let channels = channels.max(1);
    let sample_rate_hz = sample_rate_hz.max(1);
    let data_bytes = samples.len().saturating_mul(2);
    let riff_size = 36usize.saturating_add(data_bytes);
    let mut bytes = Vec::with_capacity(44 + data_bytes);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(riff_size as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&channels.to_le_bytes());
    bytes.extend_from_slice(&sample_rate_hz.to_le_bytes());
    let byte_rate = sample_rate_hz
        .saturating_mul(channels as u32)
        .saturating_mul(2);
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = channels.saturating_mul(2);
    bytes.extend_from_slice(&block_align.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(data_bytes as u32).to_le_bytes());
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn bytes_to_unit_signal(bytes: &[u8]) -> Vec<f32> {
    bytes
        .iter()
        .take(256)
        .map(|byte| *byte as f32 / 255.0)
        .collect()
}

fn pcm_to_unit_signal(samples: &[i16]) -> Vec<f32> {
    samples
        .iter()
        .take(256)
        .map(|sample| *sample as f32 / i16::MAX as f32)
        .collect()
}

#[cfg(feature = "linux-hardware")]
fn parse_nmea_fix(line: &str) -> Option<GpsSense> {
    if line.starts_with("$GPGGA") || line.starts_with("$GNGGA") {
        let fields = line.split(',').collect::<Vec<_>>();
        if fields.len() < 10 {
            return None;
        }
        let lat = parse_nmea_coord(fields[2], fields[3])?;
        let lon = parse_nmea_coord(fields[4], fields[5])?;
        let altitude_m = fields[9].parse::<f32>().ok();
        return Some(GpsSense {
            schema_version: 1,
            lat,
            lon,
            altitude_m,
        });
    }
    if line.starts_with("$GPRMC") || line.starts_with("$GNRMC") {
        let fields = line.split(',').collect::<Vec<_>>();
        if fields.len() < 7 || fields[2] != "A" {
            return None;
        }
        let lat = parse_nmea_coord(fields[3], fields[4])?;
        let lon = parse_nmea_coord(fields[5], fields[6])?;
        return Some(GpsSense {
            schema_version: 1,
            lat,
            lon,
            altitude_m: None,
        });
    }
    None
}

#[cfg(feature = "linux-hardware")]
fn parse_nmea_coord(value: &str, hemi: &str) -> Option<f64> {
    let dot = value.find('.')?;
    let degrees_len = if dot > 4 { 3 } else { 2 };
    let (degrees, minutes) = value.split_at(degrees_len);
    let degrees = degrees.parse::<f64>().ok()?;
    let minutes = minutes.parse::<f64>().ok()?;
    let mut decimal = degrees + minutes / 60.0;
    if matches!(hemi, "S" | "W") {
        decimal = -decimal;
    }
    Some(decimal)
}

fn unix_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
