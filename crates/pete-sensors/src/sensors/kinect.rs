#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct KinectReplayFrame {
    pub t_ms: u64,
    pub rgb_path: Option<String>,
    #[serde(default)]
    pub rgb_width: Option<u32>,
    #[serde(default)]
    pub rgb_height: Option<u32>,
    pub depth_path: Option<String>,
    pub color_features: Option<Vec<Vec<f32>>>,
    pub depth_m: Option<Vec<f32>>,
    pub audio_angle_rad: Option<f32>,
    pub audio_confidence: Option<f32>,
}

pub struct KinectReplayProvider {
    root: PathBuf,
    frames: Vec<KinectReplayFrame>,
    cursor: usize,
    pending: VecDeque<SensePacket>,
}

impl KinectReplayProvider {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        let manifest_path = root.join("timestamps.jsonl");
        let manifest = File::open(&manifest_path)?;
        let frames = BufReader::new(manifest)
            .lines()
            .filter_map(|line| match line {
                Ok(line) if !line.trim().is_empty() => {
                    Some(serde_json::from_str(&line).map_err(anyhow::Error::from))
                }
                Ok(_) => None,
                Err(error) => Some(Err(error.into())),
            })
            .collect::<Result<Vec<KinectReplayFrame>>>()?;
        Ok(Self {
            root,
            frames,
            cursor: 0,
            pending: VecDeque::new(),
        })
    }

    fn packet_for_frame(
        &self,
        frame: &KinectReplayFrame,
    ) -> Result<(SensePacket, Option<SensePacket>)> {
        let rgb_bytes = frame
            .rgb_path
            .as_ref()
            .map(|path| std::fs::read(self.root.join(path)))
            .transpose()?;
        let depth_m = match &frame.depth_m {
            Some(depth) => depth.clone(),
            None => frame
                .depth_path
                .as_ref()
                .map(|path| read_depth_values(&self.root.join(path)))
                .transpose()?
                .unwrap_or_default(),
        };
        let color_features = frame
            .color_features
            .clone()
            .or_else(|| {
                rgb_bytes
                    .as_deref()
                    .map(|bytes| vec![bytes_to_unit_signal(bytes)])
            })
            .unwrap_or_default();
        let frame_id = format!("kinect-replay-rgbd-{}", frame.t_ms);
        let eye_frame = rgb_bytes.map(|bytes| {
            let (width, height) = match (frame.rgb_width, frame.rgb_height) {
                (Some(width), Some(height)) => (width, height),
                _ if bytes.len() == 640 * 480 * 3 => (640, 480),
                _ => ((bytes.len() / 3).max(1) as u32, 1),
            };
            EyeFrame {
                captured_at_ms: frame.t_ms,
                rgbd_frame_id: Some(frame_id.clone()),
                device_timestamp_ms: None,
                width,
                height,
                format: EyeFrameFormat::Rgb8,
                bytes,
                source: Some("kinect_replay_rgb".to_string()),
            }
        });
        let eye = eye_frame.clone().map(SensePacket::EyeFrame);
        let kinect = KinectSense {
            schema_version: 1,
            captured_at_ms: frame.t_ms,
            rgbd_frame_id: Some(frame_id),
            color_frame: eye_frame,
            color_features,
            depth_m,
            audio_angle_rad: frame.audio_angle_rad,
            audio_confidence: frame.audio_confidence.unwrap_or(0.0),
            ..KinectSense::default()
        };
        Ok((SensePacket::Kinect(kinect), eye))
    }
}

#[async_trait]
impl SenseProducer for KinectReplayProvider {
    async fn poll(&mut self) -> Result<SensePacket> {
        if let Some(packet) = self.pending.pop_front() {
            return Ok(packet);
        }
        if self.frames.is_empty() {
            anyhow::bail!("kinect replay has no frames");
        }
        let frame = &self.frames[self.cursor % self.frames.len()];
        self.cursor += 1;
        let (kinect, eye) = self.packet_for_frame(frame)?;
        if let Some(eye) = eye {
            self.pending.push_back(eye);
        }
        Ok(kinect)
    }
}

#[cfg(feature = "kinect-freenect")]
pub struct FreenectKinectProvider {
    index: i32,
    pending: VecDeque<SensePacket>,
    last_rgb_error: Option<String>,
    rgb_adjustment: KinectRgbAdjustment,
    clock: FreenectClockAligner,
    geometry_calibration: Option<pete_now::DepthGeometryCalibration>,
}

#[cfg(feature = "kinect-freenect")]
impl FreenectKinectProvider {
    pub fn new() -> Result<Self> {
        Self::with_index(0)
    }

    pub fn with_index(index: i32) -> Result<Self> {
        Ok(Self {
            index,
            pending: VecDeque::new(),
            last_rgb_error: None,
            rgb_adjustment: KinectRgbAdjustment::default(),
            clock: FreenectClockAligner::default(),
            geometry_calibration: load_kinect_geometry_calibration()?,
        })
    }

    pub fn with_rgb_adjustment(mut self, rgb_adjustment: KinectRgbAdjustment) -> Self {
        self.rgb_adjustment = rgb_adjustment;
        self
    }
}

#[cfg(feature = "kinect-freenect")]
#[async_trait]
impl SenseProducer for FreenectKinectProvider {
    fn source_name(&self) -> &'static str {
        "kinect-depth"
    }

    async fn poll(&mut self) -> Result<SensePacket> {
        if let Some(packet) = self.pending.pop_front() {
            return Ok(packet);
        }
        let depth = match read_freenect_depth_m(self.index) {
            Ok(depth) => depth,
            Err(error) => {
                // libfreenect_sync leaves its worker alive after an open/read
                // failure.  That worker retries noisily on stderr (including
                // "Invalid index" and subdevice-open messages) even though
                // the caller has already received the failure.  Stop it here;
                // the outer optional-sensor worker will retry with backoff.
                unsafe {
                    freenect_sync_stop();
                }
                return Err(error);
            }
        };
        let depth_captured_at_ms = self
            .clock
            .host_time(depth.device_timestamp_ms, depth.received_at_ms);
        let frame_id = format!("kinect-rgbd-{}", depth.device_timestamp_ms);
        let mut paired_color = None;
        match read_freenect_rgb_frame(self.index, self.rgb_adjustment) {
            Ok(mut rgb_frame) => {
                rgb_frame.frame.captured_at_ms = self
                    .clock
                    .host_time(rgb_frame.device_timestamp_ms, rgb_frame.received_at_ms);
                rgb_frame.frame.device_timestamp_ms = Some(rgb_frame.device_timestamp_ms);
                let skew_ms = depth_captured_at_ms.abs_diff(rgb_frame.frame.captured_at_ms);
                if skew_ms <= MAX_FREENECT_RGBD_SKEW_MS {
                    rgb_frame.frame.rgbd_frame_id = Some(frame_id.clone());
                    paired_color = Some(rgb_frame.frame.clone());
                    self.pending
                        .push_back(SensePacket::EyeFrame(rgb_frame.frame));
                } else {
                    eprintln!(
                        "Kinect RGB frame rejected: RGB-D device-clock skew {skew_ms} ms exceeds {MAX_FREENECT_RGBD_SKEW_MS} ms"
                    );
                }
                self.last_rgb_error = None;
            }
            Err(error) => {
                let error = error.to_string();
                if self.last_rgb_error.as_deref() != Some(error.as_str()) {
                    eprintln!(
                        "Kinect RGB frame unavailable; continuing with depth-only frame: {error}"
                    );
                }
                self.last_rgb_error = Some(error);
            }
        }
        let depth_intrinsics = self
            .geometry_calibration
            .map(|calibration| calibration.depth);
        Ok(SensePacket::Kinect(KinectSense {
            schema_version: 2,
            captured_at_ms: depth_captured_at_ms,
            rgbd_frame_id: Some(frame_id),
            color_frame: paired_color,
            device_timestamp_ms: Some(depth.device_timestamp_ms),
            depth_m: depth.depth_m,
            depth_width: depth_intrinsics
                .map(|intrinsics| intrinsics.width)
                .unwrap_or(FREENECT_DEPTH_WIDTH as u32),
            depth_height: depth_intrinsics
                .map(|intrinsics| intrinsics.height)
                .unwrap_or(FREENECT_DEPTH_HEIGHT as u32),
            depth_fx: depth_intrinsics
                .map(|intrinsics| intrinsics.fx)
                .unwrap_or(KINECT_V1_DEPTH_FX),
            depth_fy: depth_intrinsics
                .map(|intrinsics| intrinsics.fy)
                .unwrap_or(KINECT_V1_DEPTH_FY),
            depth_cx: depth_intrinsics
                .map(|intrinsics| intrinsics.cx)
                .unwrap_or(KINECT_V1_DEPTH_CX),
            depth_cy: depth_intrinsics
                .map(|intrinsics| intrinsics.cy)
                .unwrap_or(KINECT_V1_DEPTH_CY),
            depth_distortion: depth_intrinsics
                .map(|intrinsics| intrinsics.distortion)
                .unwrap_or_default(),
            geometry_calibration: self.geometry_calibration,
            min_depth_m: 0.4,
            max_depth_m: 8.0,
            depth_coordinate_system: Some("kinect_depth_image".to_string()),
            ..KinectSense::default()
        }))
    }
}

#[cfg(feature = "kinect-freenect")]
fn load_kinect_geometry_calibration() -> Result<Option<pete_now::DepthGeometryCalibration>> {
    let Ok(path) = std::env::var("PETE_KINECT_CALIBRATION_JSON") else {
        return Ok(None);
    };
    let calibration: pete_now::DepthGeometryCalibration = serde_json::from_slice(
        &std::fs::read(&path)
            .with_context(|| format!("failed to read Kinect calibration {path}"))?,
    )
    .with_context(|| format!("failed to parse Kinect calibration {path}"))?;
    if calibration.depth.width != FREENECT_DEPTH_WIDTH as u32
        || calibration.depth.height != FREENECT_DEPTH_HEIGHT as u32
        || calibration.depth.fx <= 0.0
        || calibration.depth.fy <= 0.0
    {
        anyhow::bail!("Kinect calibration must provide finite positive 640x480 depth intrinsics");
    }
    Ok(Some(calibration))
}

#[cfg(feature = "kinect-freenect")]
struct FreenectDepthFrame {
    device_timestamp_ms: u32,
    received_at_ms: TimeMs,
    depth_m: Vec<f32>,
}

#[cfg(feature = "kinect-freenect")]
struct FreenectRgbFrame {
    device_timestamp_ms: u32,
    received_at_ms: TimeMs,
    frame: EyeFrame,
}

#[cfg(feature = "kinect-freenect")]
const MAX_FREENECT_RGBD_SKEW_MS: u64 = 50;

#[cfg(feature = "kinect-freenect")]
#[derive(Default)]
struct FreenectClockAligner {
    anchor: Option<(u32, TimeMs)>,
}

#[cfg(feature = "kinect-freenect")]
impl FreenectClockAligner {
    fn host_time(&mut self, device_timestamp_ms: u32, received_at_ms: TimeMs) -> TimeMs {
        let (anchor_device_ms, anchor_host_ms) = *self
            .anchor
            .get_or_insert((device_timestamp_ms, received_at_ms));
        let delta_ms = device_timestamp_ms.wrapping_sub(anchor_device_ms) as i32 as i64;
        if delta_ms >= 0 {
            anchor_host_ms.saturating_add(delta_ms as u64)
        } else {
            anchor_host_ms.saturating_sub(delta_ms.unsigned_abs())
        }
    }
}

#[cfg(feature = "kinect-freenect")]
fn read_freenect_depth_m(index: i32) -> Result<FreenectDepthFrame> {
    let mut depth_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let mut timestamp = 0u32;
    let result = unsafe {
        freenect_sync_get_depth_with_res(
            &mut depth_ptr,
            &mut timestamp,
            index,
            FREENECT_RESOLUTION_MEDIUM,
            FREENECT_DEPTH_MM,
        )
    };
    if result != 0 {
        anyhow::bail!(
            "libfreenect failed to read Kinect depth frame from device index {index}: {result}"
        );
    }
    if depth_ptr.is_null() {
        anyhow::bail!("libfreenect returned a null Kinect depth frame for device index {index}");
    }
    let received_at_ms = unix_time_ms();
    let depth_mm =
        unsafe { std::slice::from_raw_parts(depth_ptr as *const u16, FREENECT_DEPTH_PIXELS) };
    let depth_m = depth_mm
        .iter()
        .map(|value| {
            if *value == 0 {
                0.0
            } else {
                (*value as f32 * 0.001).clamp(0.0, 8.0)
            }
        })
        .collect();
    Ok(FreenectDepthFrame {
        device_timestamp_ms: timestamp,
        received_at_ms,
        depth_m,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct KinectRgbAdjustment {
    pub enabled: bool,
    pub gain: f32,
    pub gamma: f32,
    pub target_luma: f32,
    pub auto_gain_max: f32,
    pub brightness: f32,
}

impl Default for KinectRgbAdjustment {
    fn default() -> Self {
        Self {
            enabled: true,
            gain: 1.0,
            gamma: 0.80,
            target_luma: 0.32,
            auto_gain_max: 3.0,
            brightness: 0.0,
        }
    }
}

#[cfg(feature = "kinect-freenect")]
fn read_freenect_rgb_frame(
    index: i32,
    adjustment: KinectRgbAdjustment,
) -> Result<FreenectRgbFrame> {
    let mut video_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let mut timestamp = 0u32;
    let result = unsafe {
        freenect_sync_get_video_with_res(
            &mut video_ptr,
            &mut timestamp,
            index,
            FREENECT_RESOLUTION_MEDIUM,
            FREENECT_VIDEO_RGB,
        )
    };
    if result != 0 {
        anyhow::bail!(
            "libfreenect failed to read Kinect RGB frame from device index {index}: {result}"
        );
    }
    if video_ptr.is_null() {
        anyhow::bail!("libfreenect returned a null Kinect RGB frame for device index {index}");
    }
    let rgb = unsafe { std::slice::from_raw_parts(video_ptr as *const u8, FREENECT_RGB_BYTES) };
    let bytes = adjust_kinect_rgb(rgb, adjustment);
    Ok(FreenectRgbFrame {
        device_timestamp_ms: timestamp,
        received_at_ms: unix_time_ms(),
        frame: EyeFrame {
            captured_at_ms: 0,
            rgbd_frame_id: None,
            device_timestamp_ms: None,
            width: FREENECT_DEPTH_WIDTH as u32,
            height: FREENECT_DEPTH_HEIGHT as u32,
            format: EyeFrameFormat::Rgb8,
            bytes,
            source: Some(if adjustment.enabled {
                "kinect-freenect-rgb-adjusted".to_string()
            } else {
                "kinect-freenect-rgb".to_string()
            }),
        },
    })
}

pub fn adjust_kinect_rgb(rgb: &[u8], adjustment: KinectRgbAdjustment) -> Vec<u8> {
    if !adjustment.enabled || rgb.is_empty() {
        return rgb.to_vec();
    }
    let mean = mean_rgb_luma(rgb);
    let auto_gain = if mean > f32::EPSILON {
        (adjustment.target_luma.clamp(0.0, 1.0) / mean)
            .clamp(1.0, adjustment.auto_gain_max.max(1.0))
    } else {
        adjustment.auto_gain_max.max(1.0)
    };
    let gain = (adjustment.gain.max(0.0) * auto_gain).max(0.0);
    let gamma = adjustment.gamma.clamp(0.10, 5.0);
    let brightness = adjustment.brightness.clamp(-1.0, 1.0);
    rgb.iter()
        .map(|byte| {
            let linear = (*byte as f32 / 255.0) * gain + brightness;
            let corrected = linear.clamp(0.0, 1.0).powf(gamma);
            (corrected * 255.0).round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

pub fn mean_rgb_luma(rgb: &[u8]) -> f32 {
    let mut sum = 0.0;
    let mut pixels = 0usize;
    for pixel in rgb.chunks_exact(3) {
        sum += (0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32)
            / 255.0;
        pixels += 1;
    }
    if pixels == 0 {
        0.0
    } else {
        sum / pixels as f32
    }
}

#[cfg(feature = "kinect-freenect")]
impl Drop for FreenectKinectProvider {
    fn drop(&mut self) {
        unsafe {
            freenect_sync_stop();
        }
    }
}

fn read_depth_values(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if let Ok(values) = serde_json::from_slice::<Vec<f32>>(&bytes) {
        return Ok(values);
    }
    let text = String::from_utf8(bytes)?;
    text.split_whitespace()
        .map(|value| value.parse::<f32>().map_err(anyhow::Error::from))
        .collect()
}
