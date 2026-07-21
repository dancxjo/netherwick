pub struct CameraSenseProvider {
    #[cfg(feature = "linux-hardware")]
    camera: V4lCamera,
}

impl CameraSenseProvider {
    pub fn new(device: &str) -> Result<Self> {
        #[cfg(feature = "linux-hardware")]
        {
            Ok(Self {
                camera: V4lCamera::new(device)?,
            })
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            let _ = device;
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

#[async_trait]
impl SenseProducer for CameraSenseProvider {
    async fn poll(&mut self) -> Result<SensePacket> {
        #[cfg(feature = "linux-hardware")]
        {
            let frame = self.camera.capture_frame()?;
            Ok(SensePacket::EyeFrame(frame))
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

pub struct MicrophoneSenseProvider {
    #[cfg(feature = "linux-hardware")]
    microphone: CpalMicrophone,
    #[cfg_attr(not(feature = "linux-hardware"), allow(dead_code))]
    asr: Option<AsrTool>,
    pending: VecDeque<SensePacket>,
    #[cfg_attr(not(feature = "linux-hardware"), allow(dead_code))]
    last_pcm_ms: Option<u64>,
}

impl MicrophoneSenseProvider {
    pub fn new(preferred_name: Option<&str>) -> Result<Self> {
        Self::with_asr_config(preferred_name, AsrToolConfig::default())
    }

    pub fn with_asr_config(
        preferred_name: Option<&str>,
        asr_config: AsrToolConfig,
    ) -> Result<Self> {
        #[cfg(feature = "linux-hardware")]
        {
            let asr = asr_config
                .command
                .is_some()
                .then(|| AsrTool::new(asr_config));
            Ok(Self {
                microphone: CpalMicrophone::new(preferred_name, 16000, 1)?,
                asr,
                pending: VecDeque::new(),
                last_pcm_ms: None,
            })
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            let _ = preferred_name;
            let _ = asr_config;
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

#[async_trait]
impl SenseProducer for MicrophoneSenseProvider {
    async fn poll(&mut self) -> Result<SensePacket> {
        if let Some(packet) = self.pending.pop_front() {
            return Ok(packet);
        }
        #[cfg(feature = "linux-hardware")]
        {
            let frame = self
                .microphone
                .latest_frame()
                .unwrap_or_else(|| PcmAudioFrame {
                    captured_at_ms: unix_time_ms(),
                    sample_rate_hz: 16000,
                    channels: 1,
                    samples: Vec::new(),
                });
            if self.last_pcm_ms != Some(frame.captured_at_ms) {
                self.last_pcm_ms = Some(frame.captured_at_ms);
                if let Some(asr) = self.asr.as_mut() {
                    if let Some(ear) = asr.observe_frame(&frame) {
                        self.pending.push_back(SensePacket::Ear(ear));
                    }
                }
            }
            Ok(SensePacket::EarPcm(frame))
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

pub struct GpsSenseProvider {
    #[cfg(feature = "linux-hardware")]
    gps: Ublox7Gps,
    #[cfg(feature = "linux-hardware")]
    last_fix: GpsSense,
}

/// Native serial provider for the Hitachi-LG HLS-LFCD2 / ROBOTIS LDS-01.
///
/// The sensor emits one 42-byte segment for each six degrees of a 360-degree
/// sweep. This provider follows the sensor's native clockwise ordering and
/// converts it to the counter-clockwise angle convention used by `RangeSense`.
pub struct Lfcd2SenseProvider {
    #[cfg(feature = "linux-hardware")]
    port: Box<dyn SerialPort>,
    #[cfg(feature = "linux-hardware")]
    parser: Lfcd2Parser,
    #[cfg(feature = "linux-hardware")]
    last_scan: Option<RangeSense>,
    #[cfg(feature = "linux-hardware")]
    last_scan_at: Option<Instant>,
}

impl Lfcd2SenseProvider {
    pub const BAUD_RATE: u32 = 230_400;

    pub fn new(port: &str) -> Result<Self> {
        Self::with_yaw_offset(port, 0.0)
    }

    /// Opens the lidar and rotates every beam by `yaw_offset_rad` in the robot
    /// base frame. Positive yaw is counter-clockwise.
    pub fn with_yaw_offset(port: &str, yaw_offset_rad: f32) -> Result<Self> {
        Self::with_extrinsics(
            port,
            RangeExtrinsics {
                yaw_rad: yaw_offset_rad,
                ..RangeExtrinsics::default()
            },
        )
    }

    pub fn with_extrinsics(port: &str, extrinsics: RangeExtrinsics) -> Result<Self> {
        #[cfg(feature = "linux-hardware")]
        {
            let mut port = serialport::new(port, Self::BAUD_RATE)
                .timeout(Duration::from_millis(4))
                .open()
                .with_context(|| format!("failed to open HLS-LFCD2 lidar at {port}"))?;
            // Older LFCD2 firmware requires this command. Newer firmware starts
            // on power-up and safely tolerates it.
            port.write_all(b"b")
                .context("failed to send HLS-LFCD2 start command")?;
            Ok(Self {
                port,
                parser: Lfcd2Parser::new(extrinsics),
                last_scan: None,
                last_scan_at: None,
            })
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            let _ = port;
            let _ = extrinsics;
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

#[async_trait]
impl SenseProducer for Lfcd2SenseProvider {
    async fn poll(&mut self) -> Result<SensePacket> {
        #[cfg(feature = "linux-hardware")]
        {
            // RealRobotRunner gives each sensor a 25 ms budget. Consume the
            // serial backlog incrementally and retain the latest scan between
            // native ~5 Hz updates without blocking a control tick.
            let deadline = Instant::now() + Duration::from_millis(20);
            let mut chunk = [0u8; 4096];
            loop {
                match self.port.read(&mut chunk) {
                    Ok(0) => {}
                    Ok(count) => {
                        if let Some(scan) = self.parser.push(&chunk[..count]) {
                            self.last_scan = Some(scan);
                            self.last_scan_at = Some(Instant::now());
                        }
                    }
                    Err(error) if error.kind() == ErrorKind::TimedOut => {}
                    Err(error) => return Err(error).context("failed to read HLS-LFCD2 lidar"),
                }
                if Instant::now() >= deadline {
                    break;
                }
            }
            if self
                .last_scan_at
                .is_some_and(|at| at.elapsed() <= Duration::from_millis(500))
            {
                return Ok(SensePacket::Range(
                    self.last_scan.clone().expect("scan timestamp without scan"),
                ));
            }
            anyhow::bail!("no fresh complete HLS-LFCD2 scan is available");
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

#[cfg(feature = "linux-hardware")]
impl Drop for Lfcd2SenseProvider {
    fn drop(&mut self) {
        let _ = self.port.write_all(b"e");
    }
}

#[cfg(any(feature = "linux-hardware", test))]
const LFCD2_SEGMENT_BYTES: usize = 42;
#[cfg(any(feature = "linux-hardware", test))]
const LFCD2_SEGMENTS_PER_SCAN: usize = 60;
#[cfg(any(feature = "linux-hardware", test))]
const LFCD2_BEAMS_PER_SEGMENT: usize = 6;
#[cfg(any(feature = "linux-hardware", test))]
const LFCD2_BEAMS_PER_SCAN: usize = 360;
#[cfg(any(feature = "linux-hardware", test))]
const LFCD2_MIN_RANGE_M: f32 = 0.12;
#[cfg(any(feature = "linux-hardware", test))]
const LFCD2_MAX_RANGE_M: f32 = 3.5;
#[cfg(any(feature = "linux-hardware", test))]
const LFCD2_SCAN_DURATION_MS: i32 = 200;

#[cfg(any(feature = "linux-hardware", test))]
#[derive(Clone, Debug)]
struct Lfcd2Parser {
    buffer: Vec<u8>,
    ranges_m: [f32; LFCD2_BEAMS_PER_SCAN],
    received_segments: [bool; LFCD2_SEGMENTS_PER_SCAN],
    received_count: usize,
    scan_started: bool,
    extrinsics: RangeExtrinsics,
}

#[cfg(any(feature = "linux-hardware", test))]
impl Lfcd2Parser {
    fn new(extrinsics: RangeExtrinsics) -> Self {
        Self {
            buffer: Vec::new(),
            ranges_m: [0.0; LFCD2_BEAMS_PER_SCAN],
            received_segments: [false; LFCD2_SEGMENTS_PER_SCAN],
            received_count: 0,
            scan_started: false,
            extrinsics,
        }
    }

    fn push(&mut self, bytes: &[u8]) -> Option<RangeSense> {
        self.buffer.extend_from_slice(bytes);
        loop {
            let Some(start) = self
                .buffer
                .windows(2)
                .position(|pair| pair[0] == 0xfa && (0xa0..=0xdb).contains(&pair[1]))
            else {
                let retain_sync_prefix = self.buffer.last() == Some(&0xfa);
                self.buffer.clear();
                if retain_sync_prefix {
                    self.buffer.push(0xfa);
                }
                return None;
            };
            if start > 0 {
                self.buffer.drain(..start);
            }
            if self.buffer.len() < LFCD2_SEGMENT_BYTES {
                return None;
            }

            let packet = self.buffer.drain(..LFCD2_SEGMENT_BYTES).collect::<Vec<_>>();
            let segment = usize::from(packet[1] - 0xa0);
            if segment == 0 {
                self.ranges_m.fill(0.0);
                self.received_segments.fill(false);
                self.received_count = 0;
                self.scan_started = true;
            } else if !self.scan_started {
                continue;
            }

            for beam_in_segment in 0..LFCD2_BEAMS_PER_SEGMENT {
                let raw_index = segment * LFCD2_BEAMS_PER_SEGMENT + beam_in_segment;
                let offset = 4 + beam_in_segment * 6;
                let range_mm = u16::from_le_bytes([packet[offset + 2], packet[offset + 3]]);
                let range_m = f32::from(range_mm) / 1000.0;
                // The official driver reverses raw indices so increasing output
                // angles are counter-clockwise (raw 0 degrees becomes 359).
                let output_index = LFCD2_BEAMS_PER_SCAN - 1 - raw_index;
                self.ranges_m[output_index] =
                    if (LFCD2_MIN_RANGE_M..=LFCD2_MAX_RANGE_M).contains(&range_m) {
                        range_m
                    } else {
                        0.0
                    };
            }

            if !self.received_segments[segment] {
                self.received_segments[segment] = true;
                self.received_count += 1;
            }
            if self.received_count == LFCD2_SEGMENTS_PER_SCAN {
                self.scan_started = false;
                let beams = self.ranges_m.to_vec();
                let nearest_m = beams
                    .iter()
                    .copied()
                    .filter(|range| *range > 0.0 && range.is_finite())
                    .min_by(f32::total_cmp);
                let beam_angles_rad = (0..LFCD2_BEAMS_PER_SCAN)
                    .map(|index| (index as f32).to_radians())
                    .collect();
                let beam_time_offsets_ms = (0..LFCD2_BEAMS_PER_SCAN)
                    .map(|output_index| {
                        let raw_index = LFCD2_BEAMS_PER_SCAN - 1 - output_index;
                        -LFCD2_SCAN_DURATION_MS
                            + (raw_index as i32 * LFCD2_SCAN_DURATION_MS
                                / (LFCD2_BEAMS_PER_SCAN as i32 - 1))
                    })
                    .collect();
                return Some(RangeSense {
                    schema_version: 1,
                    captured_at_ms: unix_time_ms(),
                    beams,
                    nearest_m,
                    beam_angles_rad,
                    beam_time_offsets_ms,
                    frame: Some("hls_lfcd2".to_string()),
                    source: Some("hls_lfcd2".to_string()),
                    extrinsics: Some(self.extrinsics),
                    ..RangeSense::default()
                });
            }
        }
    }
}

impl GpsSenseProvider {
    pub fn new(port: &str, baud_rate: u32) -> Result<Self> {
        #[cfg(feature = "linux-hardware")]
        {
            Ok(Self {
                gps: Ublox7Gps::new(port, baud_rate)?,
                last_fix: GpsSense::default(),
            })
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            let _ = port;
            let _ = baud_rate;
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

#[async_trait]
impl SenseProducer for GpsSenseProvider {
    async fn poll(&mut self) -> Result<SensePacket> {
        #[cfg(feature = "linux-hardware")]
        {
            if let Some(fix) = self.gps.try_read_fix()? {
                self.last_fix = fix;
            }
            Ok(SensePacket::Gps(self.last_fix.clone()))
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

pub struct ImuSenseProvider {
    #[cfg(feature = "linux-hardware")]
    imu: Mpu6050Imu,
    #[cfg(feature = "linux-hardware")]
    orientation_filter: Mpu6050OrientationFilter,
    motion_context: pete_now::ImuMotionContext,
}

impl ImuSenseProvider {
    pub fn new(device: &str) -> Result<Self> {
        #[cfg(feature = "linux-hardware")]
        {
            Ok(Self {
                imu: Mpu6050Imu::new(device)?,
                orientation_filter: Mpu6050OrientationFilter::from_env()?,
                motion_context: pete_now::ImuMotionContext::default(),
            })
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            let _ = device;
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

#[async_trait]
impl SenseProducer for ImuSenseProvider {
    fn set_motion_context(&mut self, motion: pete_now::ImuMotionContext) {
        self.motion_context = motion;
    }

    async fn poll(&mut self) -> Result<SensePacket> {
        #[cfg(feature = "linux-hardware")]
        {
            let sense = self
                .orientation_filter
                .update_with_context(self.imu.read_sense()?, self.motion_context);
            Ok(SensePacket::Imu(sense))
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

#[cfg(any(feature = "linux-hardware", test))]
const MPU6050_GYRO_BIAS_SAMPLES: u32 = 50;

#[cfg(any(feature = "linux-hardware", test))]
#[derive(Clone, Debug)]
struct Mpu6050OrientationFilter {
    calibration: pete_now::ImuCalibrationEstimator,
    last_t_ms: Option<TimeMs>,
    roll_rad: Option<f32>,
    pitch_rad: Option<f32>,
}

#[cfg(any(feature = "linux-hardware", test))]
impl Mpu6050OrientationFilter {
    #[cfg(feature = "linux-hardware")]
    fn from_env() -> Result<Self> {
        let mounting_calibrated = std::env::var("PETE_IMU_MOUNT_CALIBRATED")
            .ok()
            .is_some_and(|value| matches!(value.trim(), "1" | "true" | "yes" | "on"));
        let rotation_rpy_rad = match std::env::var("PETE_IMU_TO_BASE_RPY_DEG") {
            Ok(value) => parse_imu_mount_rpy_deg(&value)?,
            Err(_) => [0.0; 3],
        };
        Ok(Self::new(rotation_rpy_rad, mounting_calibrated))
    }

    fn new(rotation_rpy_rad: [f32; 3], mounting_calibrated: bool) -> Self {
        let prior = pete_now::RigidTransform3 {
            rotation_rpy_rad,
            ..pete_now::RigidTransform3::default()
        };
        Self {
            calibration: pete_now::ImuCalibrationEstimator::new(
                prior,
                mounting_calibrated,
                0,
                pete_now::ImuCalibrationConfig {
                    minimum_stationary_samples: u64::from(MPU6050_GYRO_BIAS_SAMPLES),
                    ..pete_now::ImuCalibrationConfig::default()
                },
            ),
            last_t_ms: None,
            roll_rad: None,
            pitch_rad: None,
        }
    }

    fn update(&mut self, sense: ImuSense) -> ImuSense {
        self.update_with_context(sense, pete_now::ImuMotionContext::default())
    }

    fn update_with_context(
        &mut self,
        mut sense: ImuSense,
        motion: pete_now::ImuMotionContext,
    ) -> ImuSense {
        let raw_acceleration = vector3(&sense.acceleration);
        let raw_angular_velocity = vector3(&sense.angular_velocity);
        self.calibration.observe(
            raw_acceleration,
            raw_angular_velocity,
            sense.temperature_c,
            motion,
            sense.captured_at_ms,
        );
        let calibration = self.calibration.state().clone();
        let acceleration = self.calibration.acceleration_in_base(raw_acceleration);
        let mut angular_velocity = self.calibration.corrected_gyro(raw_angular_velocity);
        if let Some(scale) = calibration.yaw_rate_scale {
            angular_velocity[2] *= scale;
        }
        let accel_norm = vector_norm(acceleration);
        let gyro_norm = vector_norm(angular_velocity);
        let stationary = (0.96..=1.04).contains(&accel_norm) && gyro_norm <= 0.08;
        let dt_s = self
            .last_t_ms
            .map(|last| sense.captured_at_ms.abs_diff(last) as f32 / 1000.0)
            .unwrap_or(0.0)
            .clamp(0.0, 0.1);
        self.last_t_ms = Some(sense.captured_at_ms);
        let accel_trusted = (0.90..=1.10).contains(&accel_norm);
        let accel_orientation = accel_trusted.then(|| {
            let roll = acceleration[1].atan2(acceleration[2]);
            let pitch = (-acceleration[0]).atan2(
                (acceleration[1] * acceleration[1] + acceleration[2] * acceleration[2]).sqrt(),
            );
            [roll, pitch]
        });
        let predicted_roll = self
            .roll_rad
            .unwrap_or_else(|| accel_orientation.map(|v| v[0]).unwrap_or(0.0))
            + angular_velocity[0] * dt_s;
        let predicted_pitch = self
            .pitch_rad
            .unwrap_or_else(|| accel_orientation.map(|v| v[1]).unwrap_or(0.0))
            + angular_velocity[1] * dt_s;
        let (roll, pitch) = if let Some([accel_roll, accel_pitch]) = accel_orientation {
            let gyro_weight = if stationary { 0.96 } else { 0.985 };
            (
                blend_angle(predicted_roll, accel_roll, 1.0 - gyro_weight),
                blend_angle(predicted_pitch, accel_pitch, 1.0 - gyro_weight),
            )
        } else {
            (predicted_roll, predicted_pitch)
        };
        self.roll_rad = Some(roll);
        self.pitch_rad = Some(pitch);
        sense.schema_version = 3;
        sense.orientation = vec![roll, pitch];
        sense.acceleration = acceleration.to_vec();
        sense.angular_velocity = angular_velocity.to_vec();
        sense.gyro_bias_calibrated = matches!(
            calibration.trust_state,
            pete_now::ImuCalibrationTrustState::Trusted
                | pete_now::ImuCalibrationTrustState::Degraded
        );
        sense.mounting_calibrated = calibration.roll_pitch_observable
            && calibration.trust_state != pete_now::ImuCalibrationTrustState::Invalidated;
        sense.orientation_confidence = if accel_trusted {
            calibration.confidence
        } else {
            calibration.confidence * 0.75
        };
        sense.orientation_source = Some(format!(
            "local_i2c_mpu6050@{}:adaptive_accel_gyro",
            calibration.epoch.id
        ));
        sense.calibration = Some(calibration);
        sense
    }
}

#[cfg(any(feature = "linux-hardware", test))]
fn vector3(values: &[f32]) -> [f32; 3] {
    [
        values.first().copied().unwrap_or(0.0),
        values.get(1).copied().unwrap_or(0.0),
        values.get(2).copied().unwrap_or(0.0),
    ]
}

#[cfg(any(feature = "linux-hardware", test))]
#[cfg(feature = "linux-hardware")]
fn parse_imu_mount_rpy_deg(value: &str) -> Result<[f32; 3]> {
    let values = value
        .split(',')
        .map(|part| part.trim().parse::<f32>().map(f32::to_radians))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if values.len() != 3 || values.iter().any(|value| !value.is_finite()) {
        anyhow::bail!("PETE_IMU_TO_BASE_RPY_DEG must contain three finite comma-separated values");
    }
    Ok([values[0], values[1], values[2]])
}

#[cfg(any(feature = "linux-hardware", test))]
fn vector_norm(vector: [f32; 3]) -> f32 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

#[cfg(any(feature = "linux-hardware", test))]
fn blend_angle(predicted: f32, measured: f32, measured_weight: f32) -> f32 {
    let delta = (measured - predicted + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
        - std::f32::consts::PI;
    predicted + delta * measured_weight
}
