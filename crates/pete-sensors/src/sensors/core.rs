#[async_trait]
pub trait SenseProducer {
    fn source_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Producer-owned health details such as queue depth and drop counts.
    /// The runtime records this alongside its own timeout/failure accounting.
    fn health_diagnostics(&self) -> Option<serde_json::Value> {
        None
    }

    /// Supplies current robot motion evidence before polling. Producers that
    /// estimate stationary calibration may consume this; all others ignore it.
    fn set_motion_context(&mut self, _motion: pete_now::ImuMotionContext) {}

    async fn poll(&mut self) -> Result<SensePacket>;
}

#[async_trait]
pub trait World: Send {
    async fn snapshot(&mut self) -> Result<WorldSnapshot>;
    async fn apply_update(&mut self, update: WorldUpdate) -> Result<()>;

    async fn set_body(&mut self, body: BodySense) -> Result<()> {
        self.apply_update(WorldUpdate {
            body: Some(body),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_eye_frame(&mut self, frame: EyeFrame) -> Result<()> {
        self.apply_update(WorldUpdate {
            eye_frame: Some(frame),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_eye_sense(&mut self, eye: EyeSense) -> Result<()> {
        self.apply_update(WorldUpdate {
            eye: Some(eye),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_ear_pcm_frame(&mut self, frame: PcmAudioFrame) -> Result<()> {
        self.apply_update(WorldUpdate {
            ear_pcm: Some(frame),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_ear_sense(&mut self, ear: EarSense) -> Result<()> {
        self.apply_update(WorldUpdate {
            ear: Some(ear),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_range_sense(&mut self, range: RangeSense) -> Result<()> {
        self.apply_update(WorldUpdate {
            range: Some(range),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_imu_sense(&mut self, imu: ImuSense) -> Result<()> {
        self.apply_update(WorldUpdate {
            imu: Some(imu),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_gps_sense(&mut self, gps: Option<GpsSense>) -> Result<()> {
        self.apply_update(WorldUpdate {
            gps,
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_kinect_sense(&mut self, kinect: KinectSense) -> Result<()> {
        self.apply_update(WorldUpdate {
            kinect: Some(kinect),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_object_sense(&mut self, objects: ObjectSense) -> Result<()> {
        self.apply_update(WorldUpdate {
            objects: Some(objects),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_face_sense(&mut self, face: FaceSense) -> Result<()> {
        self.apply_update(WorldUpdate {
            face: Some(face),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_voice_sense(&mut self, voice: VoiceSense) -> Result<()> {
        self.apply_update(WorldUpdate {
            voice: Some(voice),
            ..WorldUpdate::default()
        })
        .await
    }

    async fn set_extensions(&mut self, extensions: Vec<ExtensionSense>) -> Result<()> {
        self.apply_update(WorldUpdate {
            extensions: Some(extensions),
            ..WorldUpdate::default()
        })
        .await
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SensePacket {
    Eye(EyeSense),
    EyeFrame(EyeFrame),
    Ear(EarSense),
    EarPcm(PcmAudioFrame),
    Range(RangeSense),
    Imu(ImuSense),
    Gps(GpsSense),
    Kinect(KinectSense),
    Face(FaceSense),
    Voice(VoiceSense),
    Objects(ObjectSense),
    Extension(ExtensionSense),
}

#[derive(Clone, Debug, Default)]
pub struct NowBuilder {
    last_snapshot: WorldSnapshot,
    last_updates: SensorUpdateTimes,
    pose_history: VecDeque<(TimeMs, Pose2)>,
    imu_history: VecDeque<ImuSense>,
    latency_calibration: SensorLatencyRegistry,
    body_rotation_sign: i8,
    stream_rotation_signs: BTreeMap<String, i8>,
    locomotion_calibration: LocomotionCalibrationEstimator,
}

#[derive(Clone, Default)]
pub struct FrameProcessor {
    last_processed_frame_key: Option<FrameKey>,
    face_detector: Option<Arc<dyn FaceDetector>>,
    object_detector: Option<Arc<dyn ObjectDetector>>,
    kinect_range_projection: Option<DepthRangeProjectionConfig>,
    kinect_calibration: Option<CalibrationStateMachine>,
    vision_pipeline: Option<VisionPipeline>,
}

impl std::fmt::Debug for FrameProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameProcessor")
            .field("last_processed_frame_key", &self.last_processed_frame_key)
            .field("face_detector", &self.face_detector.is_some())
            .field("object_detector", &self.object_detector.is_some())
            .field("kinect_range_projection", &self.kinect_range_projection)
            .field("kinect_calibration", &self.kinect_calibration.is_some())
            .field("vision_pipeline", &self.vision_pipeline.is_some())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DepthRangeProjectionConfig {
    pub compact_depth_beam_count: usize,
    pub compact_depth_fov_rad: f32,
    pub depth_scale: f32,
    pub camera_forward_m: f32,
    pub camera_height_m: f32,
    pub camera_pitch_rad: f32,
    pub camera_roll_rad: f32,
    pub camera_yaw_rad: f32,
    pub min_depth_m: f32,
    pub max_depth_m: f32,
}

impl Default for DepthRangeProjectionConfig {
    fn default() -> Self {
        Self {
            compact_depth_beam_count: 32,
            compact_depth_fov_rad: std::f32::consts::PI * 0.75,
            depth_scale: 1.0,
            camera_forward_m: 0.0,
            camera_height_m: 0.0,
            camera_pitch_rad: 0.0,
            camera_roll_rad: 0.0,
            camera_yaw_rad: 0.0,
            min_depth_m: 0.2,
            max_depth_m: 8.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FrameKey {
    captured_at_ms: u64,
    width: u32,
    height: u32,
    format: String,
    byte_len: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProcessedFrame {
    pub eye: EyeSense,
    pub face: FaceSense,
    pub objects: ObjectSense,
    pub summary: String,
    pub source_frame_id: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensorUpdateTimes {
    pub body: Option<TimeMs>,
    pub eye: Option<TimeMs>,
    pub ear: Option<TimeMs>,
    pub range: Option<TimeMs>,
    pub imu: Option<TimeMs>,
    pub gps: Option<TimeMs>,
    pub kinect: Option<TimeMs>,
    pub face: Option<TimeMs>,
    pub objects: Option<TimeMs>,
    pub voice: Option<TimeMs>,
}

impl NowBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Invalidates canonical IMU interpolation without disturbing body pose history.
    /// Call this on source or brainstem clock-epoch changes.
    pub fn clear_imu_history(&mut self) {
        self.imu_history.clear();
        self.last_snapshot.imu = ImuSense::default();
        self.last_snapshot.kinect.fusion_alignment = None;
        self.last_updates.imu = None;
    }

    pub fn observe_latency_reference_event(&mut self, name: &str, occurred_at_ms: TimeMs) {
        self.latency_calibration
            .observe_reference_event(name, occurred_at_ms);
    }

    pub fn observe_sensor_timing(
        &mut self,
        stream: &str,
        observation: SensorTimingObservation,
    ) {
        self.latency_calibration.observe(stream, observation);
    }

    pub fn observe_straight_calibration_episode(
        &mut self,
        episode: StraightCalibrationEpisode,
    ) -> bool {
        self.locomotion_calibration.observe_straight(episode)
    }

    pub fn observe_rotation_calibration_episode(
        &mut self,
        episode: RotationCalibrationEpisode,
    ) -> bool {
        self.locomotion_calibration.observe_rotation(episode)
    }

    pub fn set_power_assessment(&mut self, assessment: Option<serde_json::Value>) {
        self.last_snapshot.power_assessment = assessment;
    }

    pub fn build(
        &mut self,
        t_ms: TimeMs,
        mut body: BodySense,
        packets: Vec<SensePacket>,
    ) -> Result<Now> {
        self.last_snapshot.calibration_transitions.clear();
        let body_sample_ms = if body.last_update_ms == 0 {
            t_ms
        } else {
            body.last_update_ms
        };
        body.last_update_ms = body_sample_ms;
        self.last_updates.body = Some(body_sample_ms);
        observe_body_timing(
            &mut self.latency_calibration,
            &mut self.body_rotation_sign,
            &body,
            body_sample_ms,
            t_ms,
        );
        push_timestamped_pose(&mut self.pose_history, body_sample_ms, body.odometry);
        self.last_snapshot.body = body;
        self.last_snapshot.extensions.clear();
        let mut saw_range_packet = false;

        for packet in packets {
            match packet {
                SensePacket::Eye(eye) => {
                    observe_packet_timing(&mut self.latency_calibration, "camera", None, t_ms, None, 0.0, Vec::new());
                    self.last_snapshot.eye = eye;
                    self.last_updates.eye = Some(t_ms);
                }
                SensePacket::EyeFrame(frame) => {
                    observe_packet_timing(
                        &mut self.latency_calibration,
                        "camera",
                        nonzero_time(frame.captured_at_ms),
                        t_ms,
                        None,
                        1.0,
                        Vec::new(),
                    );
                    self.last_snapshot.eye.frames = vec![bytes_to_unit_signal(&frame.bytes)];
                    let captured_at_ms = frame.captured_at_ms;
                    self.last_snapshot.eye_frame = Some(frame);
                    self.last_updates.eye = Some(if captured_at_ms > 0 {
                        captured_at_ms
                    } else {
                        t_ms
                    });
                }
                SensePacket::Ear(ear) => {
                    observe_packet_timing(&mut self.latency_calibration, "audio", None, t_ms, None, 0.0, Vec::new());
                    self.last_snapshot.ear = ear;
                    self.last_updates.ear = Some(t_ms);
                }
                SensePacket::EarPcm(frame) => {
                    observe_packet_timing(
                        &mut self.latency_calibration,
                        "audio",
                        nonzero_time(frame.captured_at_ms),
                        t_ms,
                        None,
                        1.0,
                        Vec::new(),
                    );
                    self.last_snapshot.ear.features = vec![pcm_to_unit_signal(&frame.samples)];
                    self.last_snapshot.ear_pcm = Some(frame);
                    self.last_updates.ear = Some(t_ms);
                }
                SensePacket::Range(mut range) => {
                    let timing_stream = if range
                        .source
                        .as_deref()
                        .is_some_and(|source| source.contains("lidar") || source.contains("lfcd"))
                    {
                        "lidar"
                    } else {
                        "range"
                    };
                    observe_packet_timing(
                        &mut self.latency_calibration,
                        timing_stream,
                        nonzero_time(range.captured_at_ms),
                        t_ms,
                        None,
                        1.0,
                        Vec::new(),
                    );
                    align_range_beam_poses(&mut range, &self.pose_history);
                    let captured_at_ms = range.captured_at_ms;
                    self.last_snapshot.range = range;
                    self.last_updates.range = Some(if captured_at_ms > 0 {
                        captured_at_ms
                    } else {
                        t_ms
                    });
                    saw_range_packet = true;
                }
                SensePacket::Imu(imu) => {
                    self.last_snapshot
                        .calibration_transitions
                        .extend(imu.calibration_transitions.iter().cloned());
                    let source_epoch = imu.source_epoch();
                    let source_name = imu.source_id().unwrap_or("imu").to_string();
                    let event_features = rotation_event_features(
                        &mut self.stream_rotation_signs,
                        &source_name,
                        imu.angular_velocity.get(2).copied().unwrap_or(0.0),
                        imu.captured_at_ms,
                    );
                    observe_packet_timing(
                        &mut self.latency_calibration,
                        "imu",
                        nonzero_time(imu.captured_at_ms),
                        t_ms,
                        source_epoch,
                        1.0,
                        event_features,
                    );
                    push_timestamped_imu(&mut self.imu_history, &imu, t_ms);
                    let captured_at_ms = imu.captured_at_ms;
                    self.last_snapshot.imu = imu;
                    self.last_updates.imu = Some(if captured_at_ms > 0 {
                        captured_at_ms
                    } else {
                        t_ms
                    });
                }
                SensePacket::Gps(gps) => {
                    self.last_snapshot.gps = Some(gps);
                    self.last_updates.gps = Some(t_ms);
                }
                SensePacket::Kinect(kinect) => {
                    self.last_snapshot
                        .calibration_transitions
                        .extend(kinect.calibration_transitions.iter().cloned());
                    observe_packet_timing(
                        &mut self.latency_calibration,
                        "kinect",
                        nonzero_time(kinect.captured_at_ms),
                        t_ms,
                        None,
                        1.0,
                        Vec::new(),
                    );
                    let captured_at_ms = kinect.captured_at_ms;
                    if let Some(color_frame) = kinect.color_frame.clone() {
                        self.last_updates.eye = Some(color_frame.captured_at_ms);
                        self.last_snapshot.eye_frame = Some(color_frame);
                    }
                    if !saw_range_packet {
                        if let Some(range) = range_from_kinect_depth(&kinect) {
                            self.last_snapshot.range = range;
                            self.last_updates.range = Some(t_ms);
                        }
                    }
                    self.last_snapshot.kinect = kinect;
                    self.last_updates.kinect = Some(if captured_at_ms > 0 {
                        captured_at_ms
                    } else {
                        t_ms
                    });
                }
                SensePacket::Face(face) => {
                    self.last_snapshot.face = face;
                    self.last_updates.face = Some(t_ms);
                }
                SensePacket::Voice(voice) => {
                    self.last_snapshot.voice = voice;
                    self.last_updates.voice = Some(t_ms);
                }
                SensePacket::Objects(objects) => {
                    self.last_snapshot.objects = objects;
                    self.last_updates.objects = Some(t_ms);
                }
                SensePacket::Extension(extension) => {
                    self.last_snapshot.extensions.push(extension);
                }
            }
        }

        let latency_calibration = self.latency_calibration.snapshot(t_ms);
        self.last_snapshot
            .calibration_transitions
            .extend(self.latency_calibration.take_transitions());
        self.last_snapshot
            .calibration_transitions
            .extend(self.locomotion_calibration.take_transitions());
        self.last_snapshot.latency_calibration = latency_calibration.clone();
        self.last_snapshot.locomotion_calibration = self.locomotion_calibration.state().clone();
        if self.last_snapshot.kinect.captured_at_ms > 0 {
            let captured_at_ms = self.last_snapshot.kinect.captured_at_ms;
            self.last_snapshot.kinect.fusion_alignment =
                fusion_alignment_at(&self.pose_history, &self.imu_history, captured_at_ms);
            apply_latency_fusion_trust(
                &mut self.last_snapshot.kinect.fusion_alignment,
                &latency_calibration,
            );
        }

        let mut now = self.last_snapshot.to_now(t_ms);
        now.extensions.insert(
            "sensor_status".to_string(),
            serde_json::json!({
                "last_update_ms": self.last_updates,
                "age_ms": self.last_updates.age_ms(t_ms),
            }),
        );
        Ok(now)
    }

    pub fn snapshot(&self) -> WorldSnapshot {
        self.last_snapshot.clone()
    }
}

fn nonzero_time(value: TimeMs) -> Option<TimeMs> {
    (value > 0).then_some(value)
}

fn observe_packet_timing(
    registry: &mut SensorLatencyRegistry,
    stream: &str,
    producer_time_ms: Option<TimeMs>,
    receive_time_ms: TimeMs,
    clock_epoch: Option<u64>,
    clock_confidence: f32,
    event_features: Vec<LatencyEventFeature>,
) {
    registry.observe(
        stream,
        SensorTimingObservation {
            producer_time_ms,
            receive_time_ms,
            canonical_frame_time_ms: receive_time_ms,
            clock_epoch,
            clock_confidence,
            event_features,
        },
    );
}

fn observe_body_timing(
    registry: &mut SensorLatencyRegistry,
    last_rotation_sign: &mut i8,
    body: &BodySense,
    producer_time_ms: TimeMs,
    receive_time_ms: TimeMs,
) {
    let sign = rotation_sign(body.velocity.turn_rad_s);
    if sign != 0 && sign != *last_rotation_sign {
        registry.observe_reference_event(rotation_event_name(sign), producer_time_ms);
    }
    *last_rotation_sign = sign;
    observe_packet_timing(
        registry,
        "body",
        Some(producer_time_ms),
        receive_time_ms,
        None,
        1.0,
        Vec::new(),
    );
}

fn rotation_event_features(
    signs: &mut BTreeMap<String, i8>,
    source: &str,
    angular_rate: f32,
    occurred_at_ms: TimeMs,
) -> Vec<LatencyEventFeature> {
    let sign = rotation_sign(angular_rate);
    let previous = signs.insert(source.to_string(), sign).unwrap_or(0);
    if sign == 0 || sign == previous {
        return Vec::new();
    }
    vec![LatencyEventFeature {
        name: rotation_event_name(sign).to_string(),
        value: angular_rate,
        occurred_at_ms,
    }]
}

fn rotation_sign(rate: f32) -> i8 {
    if rate > 0.08 {
        1
    } else if rate < -0.08 {
        -1
    } else {
        0
    }
}

fn rotation_event_name(sign: i8) -> &'static str {
    if sign > 0 {
        "rotation_ccw_started"
    } else {
        "rotation_cw_started"
    }
}

fn apply_latency_fusion_trust(
    alignment: &mut Option<KinectFusionAlignment>,
    states: &BTreeMap<String, pete_now::StreamLatencyCalibration>,
) {
    if ["kinect", "imu"].iter().any(|stream| {
        states.get(*stream).is_some_and(|state| {
            matches!(
                state.trust_state,
                LatencyTrustState::Degraded | LatencyTrustState::Invalidated
            )
        })
    }) {
        *alignment = None;
        return;
    }
    let Some(alignment) = alignment.as_mut() else {
        return;
    };
    for stream in ["kinect", "imu"] {
        let Some(state) = states.get(stream) else {
            alignment.confidence *= 0.25;
            continue;
        };
        match state.trust_state {
            LatencyTrustState::Trusted => alignment.confidence *= state.confidence.max(0.5),
            LatencyTrustState::Estimating | LatencyTrustState::Unobservable => {
                alignment.confidence *= 0.25
            }
            LatencyTrustState::Degraded | LatencyTrustState::Invalidated => unreachable!(),
        }
    }
}

fn push_timestamped_pose(history: &mut VecDeque<(TimeMs, Pose2)>, t_ms: TimeMs, pose: Pose2) {
    if history
        .back()
        .is_some_and(|(last_t_ms, last_pose)| *last_t_ms == t_ms && *last_pose == pose)
    {
        return;
    }
    history.push_back((t_ms, pose));
    while history.len() > FUSION_HISTORY_LIMIT {
        history.pop_front();
    }
}

fn push_timestamped_imu(history: &mut VecDeque<ImuSense>, imu: &ImuSense, fallback_t_ms: TimeMs) {
    let mut sample = imu.clone();
    if sample.captured_at_ms == 0 {
        sample.captured_at_ms = fallback_t_ms;
    }
    if history
        .back()
        .is_some_and(|last| last.captured_at_ms == sample.captured_at_ms)
    {
        history.pop_back();
    }
    history.push_back(sample);
    while history.len() > FUSION_HISTORY_LIMIT {
        history.pop_front();
    }
}

fn fusion_alignment_at(
    poses: &VecDeque<(TimeMs, Pose2)>,
    imus: &VecDeque<ImuSense>,
    target_ms: TimeMs,
) -> Option<KinectFusionAlignment> {
    let (pose, pose_skew, pose_span) = interpolate_pose(poses, target_ms)?;
    let (imu, imu_skew, imu_span) = interpolate_imu(imus, target_ms)?;
    if pose_skew > MAX_FUSION_SAMPLE_SKEW_MS || imu_skew > MAX_FUSION_SAMPLE_SKEW_MS {
        return None;
    }
    let worst_skew = pose_skew.max(imu_skew) as f32;
    Some(KinectFusionAlignment {
        pose,
        imu,
        captured_at_ms: target_ms,
        pose_sample_skew_ms: pose_skew,
        imu_sample_skew_ms: imu_skew,
        pose_bracket_span_ms: pose_span,
        imu_bracket_span_ms: imu_span,
        confidence: (1.0 - worst_skew / MAX_FUSION_SAMPLE_SKEW_MS as f32).clamp(0.0, 1.0),
    })
}

fn interpolate_pose(
    history: &VecDeque<(TimeMs, Pose2)>,
    target_ms: TimeMs,
) -> Option<(Pose2, u64, u64)> {
    let samples = history.iter().copied().collect::<Vec<_>>();
    let (before, after) = bracketing_samples(&samples, target_ms, |sample| sample.0)?;
    let (before_t, before_pose) = before;
    let (after_t, after_pose) = after;
    let span = after_t.abs_diff(before_t);
    let nearest = target_ms
        .abs_diff(before_t)
        .min(target_ms.abs_diff(after_t));
    let alpha = interpolation_alpha(before_t, after_t, target_ms);
    Some((
        Pose2 {
            x_m: lerp(before_pose.x_m, after_pose.x_m, alpha),
            y_m: lerp(before_pose.y_m, after_pose.y_m, alpha),
            heading_rad: lerp_angle(before_pose.heading_rad, after_pose.heading_rad, alpha),
        },
        nearest,
        span,
    ))
}

fn interpolate_imu(
    history: &VecDeque<ImuSense>,
    target_ms: TimeMs,
) -> Option<(ImuSense, u64, u64)> {
    let samples = history.iter().cloned().collect::<Vec<_>>();
    let (before, after) = bracketing_samples(&samples, target_ms, |sample| sample.captured_at_ms)?;
    if before.source_id() != after.source_id() || before.source_epoch() != after.source_epoch() {
        return None;
    }
    let span = after.captured_at_ms.abs_diff(before.captured_at_ms);
    let nearest = target_ms
        .abs_diff(before.captured_at_ms)
        .min(target_ms.abs_diff(after.captured_at_ms));
    let alpha = interpolation_alpha(before.captured_at_ms, after.captured_at_ms, target_ms);
    let mut imu = before.clone();
    imu.captured_at_ms = target_ms;
    imu.orientation = interpolate_vector(&before.orientation, &after.orientation, alpha, true);
    imu.acceleration = interpolate_vector(&before.acceleration, &after.acceleration, alpha, false);
    imu.angular_velocity = interpolate_vector(
        &before.angular_velocity,
        &after.angular_velocity,
        alpha,
        false,
    );
    imu.orientation_confidence = lerp(
        before.orientation_confidence,
        after.orientation_confidence,
        alpha,
    );
    imu.gyro_bias_calibrated = before.gyro_bias_calibrated && after.gyro_bias_calibrated;
    imu.mounting_calibrated = before.mounting_calibrated && after.mounting_calibrated;
    Some((imu, nearest, span))
}

fn bracketing_samples<T: Clone>(
    samples: &[T],
    target_ms: TimeMs,
    timestamp: impl Fn(&T) -> TimeMs,
) -> Option<(T, T)> {
    let before = samples
        .iter()
        .filter(|sample| timestamp(sample) <= target_ms)
        .max_by_key(|sample| timestamp(sample));
    let after = samples
        .iter()
        .filter(|sample| timestamp(sample) >= target_ms)
        .min_by_key(|sample| timestamp(sample));
    match (before, after) {
        (Some(before), Some(after)) => Some((before.clone(), after.clone())),
        (Some(sample), None) | (None, Some(sample)) => Some((sample.clone(), sample.clone())),
        (None, None) => None,
    }
}

fn interpolation_alpha(before_ms: u64, after_ms: u64, target_ms: u64) -> f32 {
    if before_ms == after_ms {
        0.0
    } else {
        (target_ms.saturating_sub(before_ms) as f32 / after_ms.abs_diff(before_ms) as f32)
            .clamp(0.0, 1.0)
    }
}

fn interpolate_vector(before: &[f32], after: &[f32], alpha: f32, angular: bool) -> Vec<f32> {
    before
        .iter()
        .zip(after)
        .map(|(before, after)| {
            if angular {
                lerp_angle(*before, *after, alpha)
            } else {
                lerp(*before, *after, alpha)
            }
        })
        .collect()
}

fn lerp(before: f32, after: f32, alpha: f32) -> f32 {
    before + (after - before) * alpha
}

fn lerp_angle(before: f32, after: f32, alpha: f32) -> f32 {
    let delta = (after - before + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
        - std::f32::consts::PI;
    before + delta * alpha
}

fn align_range_beam_poses(range: &mut RangeSense, poses: &VecDeque<(TimeMs, Pose2)>) {
    if range.captured_at_ms == 0
        || range.beam_time_offsets_ms.len() != range.beams.len()
        || range.beams.is_empty()
    {
        return;
    }
    let aligned = range
        .beam_time_offsets_ms
        .iter()
        .map(|offset_ms| {
            let beam_t_ms = if *offset_ms >= 0 {
                range.captured_at_ms.saturating_add(*offset_ms as u64)
            } else {
                range
                    .captured_at_ms
                    .saturating_sub(offset_ms.unsigned_abs() as u64)
            };
            interpolate_pose(poses, beam_t_ms)
        })
        .collect::<Option<Vec<_>>>();
    let Some(aligned) = aligned else {
        return;
    };
    let max_skew = aligned.iter().map(|(_, skew, _)| *skew).max().unwrap_or(0);
    if max_skew > MAX_FUSION_SAMPLE_SKEW_MS {
        return;
    }
    range.beam_poses = aligned.into_iter().map(|(pose, _, _)| pose).collect();
    range.beam_pose_max_skew_ms = Some(max_skew);
}

fn range_from_kinect_depth(kinect: &KinectSense) -> Option<RangeSense> {
    range_from_kinect_depth_with_config(kinect, None)
}

fn range_from_kinect_depth_with_config(
    kinect: &KinectSense,
    config: Option<DepthRangeProjectionConfig>,
) -> Option<RangeSense> {
    const FALLBACK_BEAM_COUNT: usize = 32;
    let depth = &kinect.depth_m;
    if depth.is_empty() {
        return None;
    }
    let transform = config.unwrap_or_default();
    let min_depth = positive_or(kinect.min_depth_m, transform.min_depth_m);
    let max_depth = if kinect.max_depth_m > min_depth {
        kinect.max_depth_m
    } else {
        transform.max_depth_m.max(min_depth)
    };
    let valid_depth = |value: f32| value.is_finite() && value >= min_depth && value <= max_depth;

    if let Some(projection) = RangeDepthProjection::from_kinect(kinect, min_depth, max_depth) {
        let beam_count = config
            .map(|config| config.compact_depth_beam_count)
            .filter(|count| *count > 0)
            .unwrap_or(FALLBACK_BEAM_COUNT)
            .min(projection.width.max(1));
        return range_from_depth_image(kinect, projection, beam_count, transform).or_else(|| {
            range_from_depth_image_without_intrinsics(kinect, projection, beam_count, transform)
        });
    }

    if config.is_some() && depth.len() == transform.compact_depth_beam_count {
        return range_from_compact_depth(depth, transform, min_depth, max_depth);
    }

    let beams = depth
        .iter()
        .copied()
        .filter(|value| valid_depth(*value))
        .take(FALLBACK_BEAM_COUNT)
        .collect::<Vec<_>>();

    if beams.is_empty() {
        return None;
    }
    let nearest_m = beams.iter().copied().reduce(f32::min);
    Some(RangeSense {
        schema_version: 1,
        captured_at_ms: kinect.captured_at_ms,
        beams,
        nearest_m,
        beam_angles_rad: Vec::new(),
        frame: None,
        source: Some("kinect_depth_legacy_range".to_string()),
        extrinsics: None,
        ..RangeSense::default()
    })
}

fn range_from_depth_image_without_intrinsics(
    kinect: &KinectSense,
    projection: RangeDepthProjection,
    beam_count: usize,
    transform: DepthRangeProjectionConfig,
) -> Option<RangeSense> {
    let beam_count = beam_count.max(1);
    let row_start = projection.height / 3;
    let row_end = (projection.height * 2 / 3)
        .max(row_start + 1)
        .min(projection.height);
    let mut beams = vec![projection.max_depth_m; beam_count];
    let mut saw_valid = false;
    for y in row_start..row_end {
        let row = y * projection.width;
        for x in 0..projection.width {
            let depth_m = kinect.depth_m[row + x] * transform.depth_scale;
            if !depth_m.is_finite()
                || depth_m < projection.min_depth_m
                || depth_m > projection.max_depth_m
            {
                continue;
            }
            let beam = (x * beam_count / projection.width).min(beam_count - 1);
            beams[beam] = beams[beam].min(depth_m);
            saw_valid = true;
        }
    }
    if !saw_valid {
        return None;
    }
    let fov = transform.compact_depth_fov_rad;
    let angles = (0..beam_count)
        .map(|beam| {
            if beam_count == 1 {
                0.0
            } else {
                -fov * 0.5 + fov * beam as f32 / (beam_count - 1) as f32
            }
        })
        .collect();
    Some(RangeSense {
        schema_version: 1,
        captured_at_ms: kinect.captured_at_ms,
        nearest_m: beams.iter().copied().reduce(f32::min),
        beams,
        beam_angles_rad: angles,
        frame: Some("robot_base".to_string()),
        source: Some("kinect_depth_image".to_string()),
        extrinsics: None,
        ..RangeSense::default()
    })
}

#[derive(Clone, Copy, Debug)]
struct RangeDepthProjection {
    width: usize,
    height: usize,
    min_depth_m: f32,
    max_depth_m: f32,
}

impl RangeDepthProjection {
    fn from_kinect(kinect: &KinectSense, min_depth_m: f32, max_depth_m: f32) -> Option<Self> {
        let width = usize::try_from(kinect.depth_width).ok()?;
        let height = usize::try_from(kinect.depth_height).ok()?;
        if width == 0 || height == 0 || width.checked_mul(height)? != kinect.depth_m.len() {
            return None;
        }
        Some(Self {
            width,
            height,
            min_depth_m,
            max_depth_m,
        })
    }
}

fn range_from_depth_image(
    kinect: &KinectSense,
    projection: RangeDepthProjection,
    beam_count: usize,
    transform: DepthRangeProjectionConfig,
) -> Option<RangeSense> {
    let geometry = pete_now::DepthGeometry::from_kinect(kinect)?;
    let beam_count = beam_count.max(1);
    let row_start = projection.height / 3;
    let row_end = (projection.height * 2 / 3)
        .max(row_start + 1)
        .min(projection.height);
    let mut beams = vec![projection.max_depth_m; beam_count];
    let mut angles = (0..beam_count)
        .map(|beam| {
            let u = ((beam as f32 + 0.5) * projection.width as f32 / beam_count as f32)
                .clamp(0.0, projection.width.saturating_sub(1) as f32);
            let v = (projection.height as f32 - 1.0) * 0.5;
            let raw_depth_m = if kinect.geometry_calibration.is_some() {
                1.0
            } else {
                transform.depth_scale
            };
            let camera = geometry
                .depth_pixel_to_camera(u, v, raw_depth_m)
                .unwrap_or([0.0, 0.0, 1.0]);
            let robot = range_camera_point_to_robot(kinect, geometry, camera, transform);
            robot[1].atan2(robot[0])
        })
        .collect::<Vec<_>>();
    let mut saw_valid = false;

    for y in row_start..row_end {
        let row = y * projection.width;
        for x in 0..projection.width {
            let raw_depth_m = kinect.depth_m[row + x];
            let input_depth_m = if kinect.geometry_calibration.is_some() {
                raw_depth_m
            } else {
                raw_depth_m * transform.depth_scale
            };
            let Some(camera) = geometry.depth_pixel_to_camera(x as f32, y as f32, input_depth_m)
            else {
                continue;
            };
            if camera[2] < projection.min_depth_m || camera[2] > projection.max_depth_m {
                continue;
            }
            let beam = (x * beam_count / projection.width).min(beam_count - 1);
            let robot = range_camera_point_to_robot(kinect, geometry, camera, transform);
            let planar_distance = robot[0].hypot(robot[1]);
            if planar_distance.is_finite() && planar_distance < beams[beam] {
                beams[beam] = planar_distance;
                angles[beam] = robot[1].atan2(robot[0]);
                saw_valid = true;
            }
        }
    }

    if !saw_valid {
        return None;
    }
    let nearest_m = beams.iter().copied().reduce(f32::min);
    Some(RangeSense {
        schema_version: 1,
        captured_at_ms: kinect.captured_at_ms,
        beams,
        nearest_m,
        beam_angles_rad: angles,
        frame: Some("robot_base".to_string()),
        source: Some("kinect_depth_image".to_string()),
        extrinsics: None,
        ..RangeSense::default()
    })
}

fn range_from_compact_depth(
    depth: &[f32],
    transform: DepthRangeProjectionConfig,
    min_depth_m: f32,
    max_depth_m: f32,
) -> Option<RangeSense> {
    let beam_count = depth.len().max(1);
    let fov_rad = transform
        .compact_depth_fov_rad
        .clamp(0.01, std::f32::consts::TAU);
    let start = if beam_count == 1 { 0.0 } else { -fov_rad * 0.5 };
    let step = if beam_count == 1 {
        0.0
    } else {
        fov_rad / (beam_count - 1) as f32
    };
    let mut beams = Vec::with_capacity(depth.len());
    let mut angles = Vec::with_capacity(depth.len());

    for (index, depth_m) in depth.iter().enumerate() {
        let scaled = *depth_m * transform.depth_scale;
        if !scaled.is_finite() || scaled < min_depth_m || scaled > max_depth_m {
            continue;
        }
        let angle = start + step * index as f32;
        let robot = depth_apply_robot_extrinsics(
            [angle.cos() * scaled, angle.sin() * scaled, 0.0],
            transform,
        );
        let planar_distance = robot[0].hypot(robot[1]);
        if !planar_distance.is_finite() {
            continue;
        }
        beams.push(planar_distance);
        angles.push(robot[1].atan2(robot[0]));
    }

    if beams.is_empty() {
        return None;
    }
    let nearest_m = beams.iter().copied().reduce(f32::min);
    Some(RangeSense {
        schema_version: 1,
        captured_at_ms: 0,
        beams,
        nearest_m,
        beam_angles_rad: angles,
        frame: Some("robot_base".to_string()),
        source: Some("kinect_compact_depth".to_string()),
        extrinsics: None,
        ..RangeSense::default()
    })
}

fn range_camera_point_to_robot(
    kinect: &KinectSense,
    geometry: pete_now::DepthGeometry,
    camera: [f32; 3],
    transform: DepthRangeProjectionConfig,
) -> [f32; 3] {
    if kinect.geometry_calibration.is_some() {
        geometry.depth_point_to_base(camera)
    } else {
        depth_camera_point_to_robot(camera, transform)
    }
}

fn depth_camera_point_to_robot(
    camera: [f32; 3],
    transform: DepthRangeProjectionConfig,
) -> [f32; 3] {
    depth_apply_robot_extrinsics([camera[2], -camera[0], -camera[1]], transform)
}

fn depth_apply_robot_extrinsics(base: [f32; 3], transform: DepthRangeProjectionConfig) -> [f32; 3] {
    let rotated = depth_rotate_robot_extrinsic(
        base,
        transform.camera_pitch_rad,
        transform.camera_roll_rad,
        transform.camera_yaw_rad,
    );
    [
        rotated[0] + transform.camera_forward_m,
        rotated[1],
        rotated[2] + transform.camera_height_m,
    ]
}

fn depth_rotate_robot_extrinsic(
    point: [f32; 3],
    pitch_rad: f32,
    roll_rad: f32,
    yaw_rad: f32,
) -> [f32; 3] {
    let (pitch_sin, pitch_cos) = pitch_rad.sin_cos();
    let mut x = point[0] * pitch_cos + point[2] * pitch_sin;
    let y = point[1];
    let mut z = -point[0] * pitch_sin + point[2] * pitch_cos;

    let (roll_sin, roll_cos) = roll_rad.sin_cos();
    let rolled_y = y * roll_cos - z * roll_sin;
    z = y * roll_sin + z * roll_cos;

    let (yaw_sin, yaw_cos) = yaw_rad.sin_cos();
    let yawed_x = x * yaw_cos - rolled_y * yaw_sin;
    let yawed_y = x * yaw_sin + rolled_y * yaw_cos;
    x = yawed_x;

    [x, yawed_y, z]
}

fn positive_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

impl FrameProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_face_detector(mut self, detector: Arc<dyn FaceDetector>) -> Self {
        self.face_detector = Some(detector);
        self
    }

    pub fn with_object_detector(mut self, detector: Arc<dyn ObjectDetector>) -> Self {
        self.object_detector = Some(detector);
        self
    }

    pub fn with_vision_pipeline(mut self, pipeline: VisionPipeline) -> Self {
        self.vision_pipeline = Some(pipeline);
        self
    }

    pub fn with_kinect_range_projection(mut self, config: DepthRangeProjectionConfig) -> Self {
        self.kinect_range_projection = Some(config);
        self
    }

    pub fn process_packets(&mut self, t_ms: TimeMs, packets: &mut Vec<SensePacket>) {
        self.process_kinect_calibration_packets(t_ms, packets);
        self.process_kinect_range_packets(packets);

        let frame = packets
            .iter()
            .rev()
            .find_map(|packet| match packet {
                SensePacket::EyeFrame(frame) => Some(frame),
                _ => None,
            })
            .cloned();
        let kinect = packets
            .iter()
            .rev()
            .find_map(|packet| match packet {
                SensePacket::Kinect(kinect) => Some(kinect),
                _ => None,
            })
            .cloned();

        if let (Some(pipeline), Some(frame)) = (&self.vision_pipeline, frame.as_ref()) {
            pipeline.enqueue(t_ms, frame.clone(), kinect.as_ref().cloned());
        }

        let mut combined_objects = ObjectSense {
            schema_version: 2,
            ..ObjectSense::default()
        };
        if let Some(frame) = frame.as_ref() {
            if let Some(processed) = self.process_frame(t_ms, frame) {
                let summary_values = summary_extension_values(&processed);
                packets.push(SensePacket::Eye(processed.eye));
                if !processed.face.vectors.is_empty() {
                    packets.push(SensePacket::Face(processed.face));
                }
                combined_objects
                    .observations
                    .extend(processed.objects.observations);
                combined_objects.vectors.extend(processed.objects.vectors);
                packets.push(SensePacket::Extension(ExtensionSense {
                    schema_version: 1,
                    name: "vision.frame_summary".to_string(),
                    values: summary_values,
                }));
            }
        }

        if let Some(pipeline) = &self.vision_pipeline {
            let objects = pipeline.drain(t_ms, kinect.as_ref());
            combined_objects.observations.extend(objects.observations);
            combined_objects.detections.extend(objects.detections);
            combined_objects.vision_health = objects.vision_health;
        }
        if !combined_objects.observations.is_empty()
            || !combined_objects.vectors.is_empty()
            || !combined_objects.detections.is_empty()
            || combined_objects.vision_health.is_some()
        {
            packets.push(SensePacket::Objects(combined_objects));
        }
    }

    /// Supplies transform observations from odometry, persistent surfaces,
    /// map consistency, loop closure, or optional lidar adapters. Trust and
    /// epoch transitions remain centralized here.
    pub fn observe_kinect_calibration_evidence(
        &mut self,
        evidence: TransformEstimateEvidence,
        now_ms: u64,
    ) -> Option<&pete_now::LiveCalibrationEstimate> {
        self.kinect_calibration
            .as_mut()
            .map(|state| state.observe(evidence, now_ms))
    }

    fn process_kinect_calibration_packets(
        &mut self,
        t_ms: TimeMs,
        packets: &mut [SensePacket],
    ) {
        let Some(kinect) = packets.iter_mut().rev().find_map(|packet| match packet {
            SensePacket::Kinect(kinect) => Some(kinect),
            _ => None,
        }) else {
            return;
        };
        self.update_kinect_calibration(t_ms, kinect);
    }

    fn update_kinect_calibration(&mut self, t_ms: TimeMs, kinect: &mut KinectSense) {
        if self.kinect_calibration.is_none() {
            let Some(configured) = kinect.geometry_calibration else {
                return;
            };
            let started_at_ms = if kinect.captured_at_ms > 0 {
                kinect.captured_at_ms
            } else {
                t_ms
            };
            self.kinect_calibration = Some(CalibrationStateMachine::new(
                configured.depth_to_base,
                started_at_ms,
                CalibrationStateConfig::default(),
            ));
        }
        if let Some(state) = self.kinect_calibration.as_mut() {
            if let Some(evidence) = floor_plane_calibration_evidence(kinect) {
                state.observe(evidence, t_ms);
            } else {
                state.refresh(t_ms);
            }
            kinect
                .calibration_transitions
                .extend(state.take_transitions());
        }
        kinect.live_geometry_calibration = self
            .kinect_calibration
            .as_ref()
            .map(|state| state.estimate().clone());
    }

    fn process_kinect_range_packets(&self, packets: &mut Vec<SensePacket>) {
        if packets
            .iter()
            .any(|packet| matches!(packet, SensePacket::Range(_)))
        {
            return;
        }
        let Some(config) = self.kinect_range_projection else {
            return;
        };
        let Some(kinect) = packets.iter().rev().find_map(|packet| match packet {
            SensePacket::Kinect(kinect) => Some(kinect),
            _ => None,
        }) else {
            return;
        };
        let Some(range) = range_from_kinect_depth_with_config(kinect, Some(config)) else {
            return;
        };
        packets.insert(0, SensePacket::Range(range));
    }

    pub fn process_snapshot(&mut self, t_ms: TimeMs, snapshot: &mut WorldSnapshot) {
        self.update_kinect_calibration(t_ms, &mut snapshot.kinect);
        let Some(frame) = snapshot.eye_frame.clone() else {
            if let Some(pipeline) = &self.vision_pipeline {
                let objects = pipeline.drain(t_ms, Some(&snapshot.kinect));
                if !objects.detections.is_empty() || objects.vision_health.is_some() {
                    snapshot.objects = objects;
                }
            }
            return;
        };
        if let Some(pipeline) = &self.vision_pipeline {
            pipeline.enqueue(t_ms, frame.clone(), Some(snapshot.kinect.clone()));
        }
        let Some(processed) = self.process_frame(t_ms, &frame) else {
            return;
        };
        let summary_values = summary_extension_values(&processed);
        snapshot.eye = processed.eye;
        if !processed.face.vectors.is_empty() {
            snapshot.face = processed.face;
        }
        if !processed.objects.observations.is_empty() || !processed.objects.vectors.is_empty() {
            snapshot.objects = processed.objects;
        }
        snapshot.extensions.push(ExtensionSense {
            schema_version: 1,
            name: "vision.frame_summary".to_string(),
            values: summary_values,
        });
        if let Some(pipeline) = &self.vision_pipeline {
            let detected = pipeline.drain(t_ms, Some(&snapshot.kinect));
            if !detected.detections.is_empty() || detected.vision_health.is_some() {
                snapshot.objects.detections.extend(detected.detections);
                snapshot.objects.observations.extend(detected.observations);
                snapshot.objects.vision_health = detected.vision_health;
            }
        }
    }

    pub fn process_frame(&mut self, t_ms: TimeMs, frame: &EyeFrame) -> Option<ProcessedFrame> {
        let key = FrameKey::from(frame);
        if self.last_processed_frame_key.as_ref() == Some(&key) {
            return None;
        }
        self.last_processed_frame_key = Some(key);
        Some(process_eye_frame(
            t_ms,
            frame,
            self.face_detector.as_deref(),
            self.object_detector.as_deref(),
        ))
    }
}

fn floor_plane_calibration_evidence(kinect: &KinectSense) -> Option<TransformEstimateEvidence> {
    let configured = kinect.geometry_calibration?.depth_to_base;
    let [a, b, c, d] = *<&[f32; 4]>::try_from(kinect.floor_clip_plane.as_slice()).ok()?;
    let norm = (a * a + b * b + c * c).sqrt();
    if !norm.is_finite() || norm <= f32::EPSILON {
        return None;
    }
    let normal_camera = [a / norm, b / norm, c / norm];
    let normal_base = configured.rotate_vector(normal_camera);
    let mut transform = configured;
    transform.translation_m[2] = (d / norm).abs();
    transform.rotation_rpy_rad[0] -= normal_base[1].atan2(normal_base[2]);
    transform.rotation_rpy_rad[1] +=
        normal_base[0].atan2(normal_base[1].hypot(normal_base[2]));
    let tilt_residual = normal_base[2].clamp(-1.0, 1.0).acos();
    Some(TransformEstimateEvidence {
        source: CalibrationEvidenceSource::FloorPlane,
        captured_at_ms: kinect.captured_at_ms,
        transform,
        observable_dofs: [false, false, true, true, true, false],
        covariance: [1.0, 1.0, 0.0025, 0.0012, 0.0012, 1.0],
        residuals: CalibrationResiduals {
            gravity_rad: Some(tilt_residual),
            ..CalibrationResiduals::default()
        },
    })
}
