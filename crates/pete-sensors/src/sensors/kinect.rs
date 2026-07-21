#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct KinectReplayFrame {
    pub t_ms: u64,
    pub rgb_path: Option<String>,
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
        let eye = rgb_bytes.map(|bytes| {
            SensePacket::Eye(EyeSense {
                schema_version: 1,
                frames: vec![bytes_to_unit_signal(&bytes)],
                image_vectors: Vec::new(),
                image_description_vectors: Vec::new(),
                scene_vectors: Vec::new(),
            })
        });
        let kinect = KinectSense {
            schema_version: 1,
            captured_at_ms: frame.t_ms,
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
        match read_freenect_rgb_frame(self.index, self.rgb_adjustment) {
            Ok(rgb_frame) => {
                self.last_rgb_error = None;
                self.pending.push_back(SensePacket::EyeFrame(rgb_frame));
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
        Ok(SensePacket::Kinect(KinectSense {
            schema_version: 1,
            captured_at_ms: depth.captured_at_ms,
            depth_m: depth.depth_m,
            depth_width: FREENECT_DEPTH_WIDTH as u32,
            depth_height: FREENECT_DEPTH_HEIGHT as u32,
            depth_fx: KINECT_V1_DEPTH_FX,
            depth_fy: KINECT_V1_DEPTH_FY,
            depth_cx: KINECT_V1_DEPTH_CX,
            depth_cy: KINECT_V1_DEPTH_CY,
            min_depth_m: 0.4,
            max_depth_m: 8.0,
            depth_coordinate_system: Some("kinect_depth_image".to_string()),
            ..KinectSense::default()
        }))
    }
}

#[cfg(feature = "kinect-freenect")]
struct FreenectDepthFrame {
    captured_at_ms: TimeMs,
    depth_m: Vec<f32>,
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
    let captured_at_ms = unix_time_ms();
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
        captured_at_ms,
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
fn read_freenect_rgb_frame(index: i32, adjustment: KinectRgbAdjustment) -> Result<EyeFrame> {
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
    Ok(EyeFrame {
        captured_at_ms: unix_time_ms(),
        width: FREENECT_DEPTH_WIDTH as u32,
        height: FREENECT_DEPTH_HEIGHT as u32,
        format: EyeFrameFormat::Rgb8,
        bytes,
        source: Some(if adjustment.enabled {
            "kinect-freenect-rgb-adjusted".to_string()
        } else {
            "kinect-freenect-rgb".to_string()
        }),
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
