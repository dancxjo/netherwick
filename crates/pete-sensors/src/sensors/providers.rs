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
                return Some(RangeSense {
                    schema_version: 1,
                    captured_at_ms: unix_time_ms(),
                    beams,
                    nearest_m,
                    beam_angles_rad,
                    frame: Some("hls_lfcd2".to_string()),
                    source: Some("hls_lfcd2".to_string()),
                    extrinsics: Some(self.extrinsics),
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
    flat_orientation_zero: Option<Mpu6050GravityBaseline>,
}

impl ImuSenseProvider {
    pub fn new(device: &str) -> Result<Self> {
        #[cfg(feature = "linux-hardware")]
        {
            Ok(Self {
                imu: Mpu6050Imu::new(device)?,
                flat_orientation_zero: None,
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
    async fn poll(&mut self) -> Result<SensePacket> {
        #[cfg(feature = "linux-hardware")]
        {
            let mut sense = self.imu.read_sense()?;
            zero_mpu6050_orientation_to_flat(&mut sense, &mut self.flat_orientation_zero);
            Ok(SensePacket::Imu(sense))
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            anyhow::bail!("linux-hardware feature is not enabled");
        }
    }
}

#[cfg(any(feature = "linux-hardware", test))]
#[derive(Clone, Copy, Debug)]
struct Mpu6050GravityBaseline {
    gravity_unit: Vec3Unit,
}

#[cfg(any(feature = "linux-hardware", test))]
#[derive(Clone, Copy, Debug)]
struct Vec3Unit {
    x: f32,
    y: f32,
    z: f32,
}

#[cfg(any(feature = "linux-hardware", test))]
fn zero_mpu6050_orientation_to_flat(
    sense: &mut ImuSense,
    flat_orientation_zero: &mut Option<Mpu6050GravityBaseline>,
) {
    let Some(gravity_unit) = normalized_mpu6050_gravity(sense) else {
        return;
    };
    let baseline = flat_orientation_zero.get_or_insert(Mpu6050GravityBaseline { gravity_unit });
    let leveled_gravity = rotate_gravity_to_flat_baseline(gravity_unit, baseline.gravity_unit);
    let (roll, pitch) = roll_pitch_from_gravity(leveled_gravity);
    if sense.orientation.len() < 2 {
        sense.orientation.resize(2, 0.0);
    }
    sense.orientation[0] = roll;
    sense.orientation[1] = pitch;
}

#[cfg(any(feature = "linux-hardware", test))]
fn normalized_mpu6050_gravity(sense: &ImuSense) -> Option<Vec3Unit> {
    let x = sense.acceleration.first().copied()?;
    let y = sense.acceleration.get(1).copied()?;
    let z = sense.acceleration.get(2).copied()?;
    normalized_vec3(x, y, z)
}

#[cfg(any(feature = "linux-hardware", test))]
fn normalized_vec3(x: f32, y: f32, z: f32) -> Option<Vec3Unit> {
    if !(x.is_finite() && y.is_finite() && z.is_finite()) {
        return None;
    }
    let norm = (x * x + y * y + z * z).sqrt();
    if norm <= 0.001 {
        return None;
    }
    Some(Vec3Unit {
        x: x / norm,
        y: y / norm,
        z: z / norm,
    })
}

#[cfg(any(feature = "linux-hardware", test))]
fn rotate_gravity_to_flat_baseline(gravity: Vec3Unit, baseline: Vec3Unit) -> Vec3Unit {
    const Z_UP: Vec3Unit = Vec3Unit {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };
    rotate_vec3_between_unit_vectors(gravity, baseline, Z_UP)
}

#[cfg(any(feature = "linux-hardware", test))]
fn rotate_vec3_between_unit_vectors(value: Vec3Unit, from: Vec3Unit, to: Vec3Unit) -> Vec3Unit {
    let axis_x = from.y * to.z - from.z * to.y;
    let axis_y = from.z * to.x - from.x * to.z;
    let axis_z = from.x * to.y - from.y * to.x;
    let sin_angle = (axis_x * axis_x + axis_y * axis_y + axis_z * axis_z).sqrt();
    let cos_angle = (from.x * to.x + from.y * to.y + from.z * to.z).clamp(-1.0, 1.0);

    if sin_angle <= 0.000001 {
        if cos_angle > 0.0 {
            return value;
        }
        return Vec3Unit {
            x: value.x,
            y: -value.y,
            z: -value.z,
        };
    }

    let ux = axis_x / sin_angle;
    let uy = axis_y / sin_angle;
    let uz = axis_z / sin_angle;
    let cross_x = uy * value.z - uz * value.y;
    let cross_y = uz * value.x - ux * value.z;
    let cross_z = ux * value.y - uy * value.x;
    let dot = ux * value.x + uy * value.y + uz * value.z;
    let one_minus_cos = 1.0 - cos_angle;

    Vec3Unit {
        x: value.x * cos_angle + cross_x * sin_angle + ux * dot * one_minus_cos,
        y: value.y * cos_angle + cross_y * sin_angle + uy * dot * one_minus_cos,
        z: value.z * cos_angle + cross_z * sin_angle + uz * dot * one_minus_cos,
    }
}

#[cfg(any(feature = "linux-hardware", test))]
fn roll_pitch_from_gravity(gravity: Vec3Unit) -> (f32, f32) {
    let roll_rad = gravity.y.atan2(gravity.z);
    let pitch_rad = (-gravity.x).atan2((gravity.y * gravity.y + gravity.z * gravity.z).sqrt());
    (roll_rad, pitch_rad)
}
