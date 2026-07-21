/// Explicit resource envelope for background recognition. The Pi 5 profile is
/// conservative enough to coexist with possession, sensor polling, and STOP.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisionPipelineConfig {
    pub profile: String,
    pub input_width: u32,
    pub input_height: u32,
    pub maximum_fps: f32,
    pub queue_capacity: usize,
    pub result_capacity: usize,
    pub inference_deadline_ms: u64,
    pub model_threads: usize,
    pub memory_limit_mb: usize,
    pub maximum_detections: usize,
    pub track_max_age_ms: u64,
    pub track_iou_threshold: f32,
}

impl VisionPipelineConfig {
    pub fn raspberry_pi_5() -> Self {
        Self {
            profile: "raspberry-pi-5".to_string(),
            input_width: 320,
            input_height: 240,
            maximum_fps: 5.0,
            queue_capacity: 2,
            result_capacity: 4,
            inference_deadline_ms: 180,
            model_threads: 2,
            memory_limit_mb: 96,
            maximum_detections: 8,
            track_max_age_ms: 750,
            track_iou_threshold: 0.3,
        }
    }
}

impl Default for VisionPipelineConfig {
    fn default() -> Self {
        Self::raspberry_pi_5()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedVisionFrame {
    pub width: u32,
    pub height: u32,
    pub rgb8: Vec<u8>,
    pub source_width: u32,
    pub source_height: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VisionProposal {
    pub bbox: VisionBoundingBox,
    pub labels: Vec<VisionLabelHypothesis>,
}

/// Backends own image preprocessing and proposals/classification, but return
/// Pete domain-neutral values rather than OpenCV matrices or model tensors.
pub trait VisionBackend: Send + Sync {
    fn identity(&self) -> VisionModelIdentity;
    fn state(&self) -> VisionBackendState {
        VisionBackendState::Ready
    }
    fn preprocess(
        &self,
        frame: &EyeFrame,
        config: &VisionPipelineConfig,
    ) -> Result<PreparedVisionFrame>;
    fn detect(
        &self,
        frame: &PreparedVisionFrame,
        maximum_detections: usize,
    ) -> Result<Vec<VisionProposal>>;
}

/// A deterministic, offline connected-component proposal backend. It is a real
/// local implementation and also the bounded fallback when no neural model is
/// installed.
#[derive(Clone, Debug, Default)]
pub struct ClassicalSaliencyBackend;

impl VisionBackend for ClassicalSaliencyBackend {
    fn identity(&self) -> VisionModelIdentity {
        VisionModelIdentity {
            backend: "pete.classical-saliency".to_string(),
            model_id: "luma-components".to_string(),
            version: "1".to_string(),
            checksum: Some("builtin:fnv1a64:vision-saliency-v1".to_string()),
        }
    }

    fn preprocess(
        &self,
        frame: &EyeFrame,
        config: &VisionPipelineConfig,
    ) -> Result<PreparedVisionFrame> {
        let source_pixels = usize::try_from(frame.width)
            .ok()
            .and_then(|width| {
                usize::try_from(frame.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or_else(|| anyhow::anyhow!("vision frame dimensions overflowed"))?;
        let source_rgb = match frame.format {
            EyeFrameFormat::Rgb8 if frame.bytes.len() >= source_pixels * 3 => {
                frame.bytes[..source_pixels * 3].to_vec()
            }
            EyeFrameFormat::Bgr8 if frame.bytes.len() >= source_pixels * 3 => frame
                .bytes
                .chunks_exact(3)
                .take(source_pixels)
                .flat_map(|pixel| [pixel[2], pixel[1], pixel[0]])
                .collect(),
            EyeFrameFormat::Gray8 if frame.bytes.len() >= source_pixels => frame
                .bytes
                .iter()
                .take(source_pixels)
                .flat_map(|value| [*value, *value, *value])
                .collect(),
            _ => anyhow::bail!(
                "unsupported or incomplete frame format for local vision: {:?}",
                frame.format
            ),
        };
        let width = frame.width.min(config.input_width).max(1);
        let height = frame.height.min(config.input_height).max(1);
        let mut rgb8 = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            let source_y = y as usize * frame.height as usize / height as usize;
            for x in 0..width {
                let source_x = x as usize * frame.width as usize / width as usize;
                let offset = (source_y * frame.width as usize + source_x) * 3;
                rgb8.extend_from_slice(&source_rgb[offset..offset + 3]);
            }
        }
        Ok(PreparedVisionFrame {
            width,
            height,
            rgb8,
            source_width: frame.width,
            source_height: frame.height,
        })
    }

    fn detect(
        &self,
        frame: &PreparedVisionFrame,
        maximum_detections: usize,
    ) -> Result<Vec<VisionProposal>> {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let pixel_count = width.saturating_mul(height);
        if pixel_count < 16 || frame.rgb8.len() < pixel_count * 3 {
            return Ok(Vec::new());
        }
        let mut luma = Vec::with_capacity(pixel_count);
        let mut mean = 0.0_f32;
        for pixel in frame.rgb8.chunks_exact(3).take(pixel_count) {
            let value =
                (0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32)
                    / 255.0;
            mean += value;
            luma.push(value);
        }
        mean /= pixel_count as f32;
        let variance =
            luma.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / pixel_count as f32;
        if variance < 0.0015 {
            return Ok(Vec::new());
        }
        let threshold = (mean + variance.sqrt().max(0.12)).clamp(0.16, 0.88);
        let mut visited = vec![false; pixel_count];
        let mut proposals = Vec::new();
        for start in 0..pixel_count {
            if visited[start] || luma[start] < threshold {
                continue;
            }
            let mut stack = vec![start];
            visited[start] = true;
            let (mut min_x, mut max_x, mut min_y, mut max_y) = (width, 0, height, 0);
            let (mut count, mut brightness) = (0_usize, 0.0_f32);
            while let Some(index) = stack.pop() {
                let x = index % width;
                let y = index / width;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                count += 1;
                brightness += luma[index];
                for neighbor in [
                    (x > 0).then_some(index - 1),
                    (x + 1 < width).then_some(index + 1),
                    (y > 0).then_some(index - width),
                    (y + 1 < height).then_some(index + width),
                ]
                .into_iter()
                .flatten()
                {
                    if !visited[neighbor] && luma[neighbor] >= threshold {
                        visited[neighbor] = true;
                        stack.push(neighbor);
                    }
                }
            }
            let box_width = max_x.saturating_sub(min_x) + 1;
            let box_height = max_y.saturating_sub(min_y) + 1;
            let area_ratio = count as f32 / pixel_count as f32;
            if count < 8 || area_ratio < 0.004 || box_width < 2 || box_height < 2 {
                continue;
            }
            let scale_x = frame.source_width as f32 / frame.width as f32;
            let scale_y = frame.source_height as f32 / frame.height as f32;
            let confidence = (0.25
                + area_ratio.sqrt() * 0.65
                + (brightness / count as f32 - mean).max(0.0) * 0.35)
                .clamp(0.05, 0.9);
            proposals.push(VisionProposal {
                bbox: VisionBoundingBox {
                    x: (min_x as f32 * scale_x).round() as u32,
                    y: (min_y as f32 * scale_y).round() as u32,
                    width: (box_width as f32 * scale_x).round().max(1.0) as u32,
                    height: (box_height as f32 * scale_y).round().max(1.0) as u32,
                },
                labels: vec![VisionLabelHypothesis {
                    label: "salient object".to_string(),
                    confidence,
                }],
            });
        }
        proposals.sort_by(|left, right| {
            right.labels[0]
                .confidence
                .partial_cmp(&left.labels[0].confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        proposals.truncate(maximum_detections);
        Ok(proposals)
    }
}

#[derive(Clone, Debug)]
pub struct UnavailableVisionBackend {
    reason: String,
}

impl UnavailableVisionBackend {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl VisionBackend for UnavailableVisionBackend {
    fn identity(&self) -> VisionModelIdentity {
        VisionModelIdentity {
            backend: "unavailable".to_string(),
            model_id: "none".to_string(),
            version: "0".to_string(),
            checksum: None,
        }
    }

    fn state(&self) -> VisionBackendState {
        VisionBackendState::Missing
    }

    fn preprocess(
        &self,
        _frame: &EyeFrame,
        _config: &VisionPipelineConfig,
    ) -> Result<PreparedVisionFrame> {
        anyhow::bail!("vision backend unavailable: {}", self.reason)
    }

    fn detect(
        &self,
        _frame: &PreparedVisionFrame,
        _maximum_detections: usize,
    ) -> Result<Vec<VisionProposal>> {
        anyhow::bail!("vision backend unavailable: {}", self.reason)
    }
}

#[derive(Clone, Debug)]
struct VisionJob {
    enqueued_at_ms: u64,
    deadline_ms: u64,
    frame: EyeFrame,
    kinect: Option<KinectSense>,
    source_frame_id: String,
    source_sensation_id: String,
    source_snapshot_id: String,
    source_stream: String,
    calibration_epoch: Option<u64>,
}

impl VisionJob {
    fn estimated_bytes(&self) -> usize {
        self.frame.bytes.len()
            + self.kinect.as_ref().map_or(0, |kinect| {
                kinect.depth_m.len() * std::mem::size_of::<f32>()
            })
    }
}

#[derive(Clone, Debug)]
struct VisionBatch {
    detections: Vec<VisionDetection>,
    calibration_epoch: Option<u64>,
}

impl VisionBatch {
    fn estimated_bytes(&self) -> usize {
        self.detections
            .iter()
            .map(|detection| {
                detection.crop_rgb8.len()
                    + detection.source_frame_id.len()
                    + detection.source_sensation_id.len()
                    + detection.descendant_sensation_id.len()
                    + detection.source_snapshot_id.len()
                    + detection.source_stream.len()
                    + detection.geometry_trust.len()
                    + detection
                        .labels
                        .iter()
                        .map(|label| label.label.len())
                        .sum::<usize>()
                    + detection
                        .position_unavailable_reasons
                        .iter()
                        .map(String::len)
                        .sum::<usize>()
            })
            .sum()
    }
}

#[derive(Clone, Debug, Default)]
struct VisionStats {
    last_queued_at_ms: Option<u64>,
    queued_frames: u64,
    processed_frames: u64,
    replaced_frames: u64,
    dropped_frames: u64,
    expired_frames: u64,
    stale_results: u64,
    failed_frames: u64,
    inference_ms: VecDeque<u64>,
    latest_error: Option<String>,
}

#[derive(Debug, Default)]
struct VisionQueues {
    pending: VecDeque<VisionJob>,
    completed: VecDeque<VisionBatch>,
    pending_bytes: usize,
    completed_bytes: usize,
}

struct VisionPipelineState {
    config: VisionPipelineConfig,
    backend: Arc<dyn VisionBackend>,
    queues: std::sync::Mutex<VisionQueues>,
    stats: std::sync::Mutex<VisionStats>,
    wake: std::sync::Condvar,
}

#[derive(Clone)]
pub struct VisionPipeline {
    state: Arc<VisionPipelineState>,
}

impl std::fmt::Debug for VisionPipeline {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VisionPipeline")
            .field("config", &self.state.config)
            .field("backend", &self.state.backend.identity())
            .finish()
    }
}

impl VisionPipeline {
    pub fn spawn(config: VisionPipelineConfig, backend: Arc<dyn VisionBackend>) -> Self {
        let state = Arc::new(VisionPipelineState {
            config,
            backend,
            queues: std::sync::Mutex::new(VisionQueues::default()),
            stats: std::sync::Mutex::new(VisionStats::default()),
            wake: std::sync::Condvar::new(),
        });
        let weak = Arc::downgrade(&state);
        std::thread::Builder::new()
            .name("pete-vision".to_string())
            .spawn(move || vision_worker(weak))
            .expect("failed to spawn bounded vision worker");
        Self { state }
    }

    pub fn classical_pi5() -> Self {
        Self::spawn(
            VisionPipelineConfig::raspberry_pi_5(),
            Arc::new(ClassicalSaliencyBackend),
        )
    }

    pub fn enqueue(&self, now_ms: u64, frame: EyeFrame, kinect: Option<KinectSense>) {
        let mut stats = self
            .state
            .stats
            .lock()
            .expect("vision stats mutex poisoned");
        let minimum_interval_ms = if self.state.config.maximum_fps > 0.0 {
            (1000.0 / self.state.config.maximum_fps) as u64
        } else {
            0
        };
        if stats
            .last_queued_at_ms
            .is_some_and(|last| now_ms.saturating_sub(last) < minimum_interval_ms)
        {
            stats.dropped_frames += 1;
            return;
        }
        stats.last_queued_at_ms = Some(now_ms);
        stats.queued_frames += 1;
        drop(stats);

        let source_frame_id = frame_id(&frame);
        let source_sensation_id = format!(
            "vision-source-{:016x}",
            stable_hash64(source_frame_id.as_bytes())
        );
        let calibration_epoch = kinect
            .as_ref()
            .and_then(|value| value.live_geometry_calibration.as_ref())
            .map(|value| value.epoch.id);
        let source_snapshot_id = format!(
            "{}:epoch:{}",
            frame.rgbd_frame_id.as_deref().unwrap_or(&source_frame_id),
            calibration_epoch.map_or_else(|| "none".to_string(), |epoch| epoch.to_string())
        );
        let source_stream = frame.source.clone().unwrap_or_else(|| {
            if frame.rgbd_frame_id.is_some() {
                "kinect_rgb"
            } else {
                "camera_rgb"
            }
            .to_string()
        });
        let job = VisionJob {
            enqueued_at_ms: now_ms,
            deadline_ms: now_ms.saturating_add(self.state.config.inference_deadline_ms),
            frame,
            kinect,
            source_frame_id,
            source_sensation_id,
            source_snapshot_id,
            source_stream,
            calibration_epoch,
        };
        let memory_limit = self
            .state
            .config
            .memory_limit_mb
            .saturating_mul(1024 * 1024);
        let job_bytes = job.estimated_bytes();
        if job_bytes > memory_limit {
            let mut stats = self
                .state
                .stats
                .lock()
                .expect("vision stats mutex poisoned");
            stats.dropped_frames += 1;
            stats.latest_error = Some(format!(
                "frame needs {job_bytes} bytes, exceeding the {} MiB vision profile",
                self.state.config.memory_limit_mb
            ));
            return;
        }
        let mut queues = self
            .state
            .queues
            .lock()
            .expect("vision queue mutex poisoned");
        while queues.pending.len() >= self.state.config.queue_capacity.max(1)
            || queues
                .pending_bytes
                .saturating_add(queues.completed_bytes)
                .saturating_add(job_bytes)
                > memory_limit
        {
            let Some(replaced) = queues.pending.pop_front() else {
                break;
            };
            queues.pending_bytes = queues
                .pending_bytes
                .saturating_sub(replaced.estimated_bytes());
            self.state
                .stats
                .lock()
                .expect("vision stats mutex poisoned")
                .replaced_frames += 1;
        }
        queues.pending_bytes = queues.pending_bytes.saturating_add(job_bytes);
        queues.pending.push_back(job);
        drop(queues);
        self.state.wake.notify_one();
    }

    pub fn drain(&self, _now_ms: u64, current_kinect: Option<&KinectSense>) -> ObjectSense {
        let current_epoch = current_kinect
            .and_then(|value| value.live_geometry_calibration.as_ref())
            .map(|value| value.epoch.id);
        let mut detections = Vec::new();
        let mut queues = self
            .state
            .queues
            .lock()
            .expect("vision queue mutex poisoned");
        while let Some(batch) = queues.completed.pop_front() {
            queues.completed_bytes = queues
                .completed_bytes
                .saturating_sub(batch.estimated_bytes());
            let mut batch = batch;
            if batch.calibration_epoch.is_some()
                && current_epoch.is_some()
                && batch.calibration_epoch != current_epoch
            {
                self.state
                    .stats
                    .lock()
                    .expect("vision stats mutex poisoned")
                    .stale_results += 1;
                for detection in &mut batch.detections {
                    detection.position = None;
                    detection.geometry_trust = "invalidated_by_epoch_change".to_string();
                    if !detection
                        .position_unavailable_reasons
                        .iter()
                        .any(|reason| reason == "calibration epoch changed after inference")
                    {
                        detection
                            .position_unavailable_reasons
                            .push("calibration epoch changed after inference".to_string());
                    }
                }
            }
            detections.extend(batch.detections);
        }
        let queue_depth = queues.pending.len();
        drop(queues);
        let health = self.health_with_queue_depth(queue_depth);
        let observations = detections
            .iter()
            .filter_map(|detection| {
                let hypothesis = detection.labels.first()?;
                let center_x = detection.bbox.x as f32 + detection.bbox.width as f32 * 0.5;
                let bearing_rad = ((center_x / detection.image_width.max(1) as f32) - 0.5) * 1.0;
                Some(ObjectObservation {
                    label: hypothesis.label.clone(),
                    class: ObjectClass::Unknown,
                    bearing_rad,
                    distance_m: detection.position.as_ref().map(|position| position.depth_m),
                    confidence: hypothesis.confidence,
                    source: if detection.source_stream.contains("kinect") {
                        ObjectObservationSource::Kinect
                    } else {
                        ObjectObservationSource::Unknown
                    },
                })
            })
            .collect();
        ObjectSense {
            schema_version: 2,
            observations,
            vectors: Vec::new(),
            detections,
            vision_health: Some(health),
        }
    }

    pub fn health(&self) -> VisionPipelineHealth {
        let queue_depth = self
            .state
            .queues
            .lock()
            .expect("vision queue mutex poisoned")
            .pending
            .len();
        self.health_with_queue_depth(queue_depth)
    }

    fn health_with_queue_depth(&self, queue_depth: usize) -> VisionPipelineHealth {
        let stats = self
            .state
            .stats
            .lock()
            .expect("vision stats mutex poisoned")
            .clone();
        let mut samples = stats.inference_ms.iter().copied().collect::<Vec<_>>();
        samples.sort_unstable();
        let percentile = |fraction: f32| {
            (!samples.is_empty()).then(|| {
                let index = ((samples.len() - 1) as f32 * fraction).round() as usize;
                samples[index]
            })
        };
        let backend_state = self.state.backend.state();
        VisionPipelineHealth {
            backend: self.state.backend.identity(),
            state: if stats.latest_error.is_some() && backend_state == VisionBackendState::Ready {
                VisionBackendState::Degraded
            } else {
                backend_state
            },
            profile: self.state.config.profile.clone(),
            input_width: self.state.config.input_width,
            input_height: self.state.config.input_height,
            maximum_fps: self.state.config.maximum_fps,
            queue_capacity: self.state.config.queue_capacity,
            inference_deadline_ms: self.state.config.inference_deadline_ms,
            model_threads: self.state.config.model_threads,
            memory_limit_mb: self.state.config.memory_limit_mb,
            queue_depth,
            queued_frames: stats.queued_frames,
            processed_frames: stats.processed_frames,
            replaced_frames: stats.replaced_frames,
            dropped_frames: stats.dropped_frames,
            expired_frames: stats.expired_frames,
            stale_results: stats.stale_results,
            failed_frames: stats.failed_frames,
            p50_inference_ms: percentile(0.5),
            p95_inference_ms: percentile(0.95),
            latest_error: stats.latest_error,
        }
    }
}

fn vision_worker(weak: std::sync::Weak<VisionPipelineState>) {
    let mut tracker = ShortTermTracker::default();
    loop {
        let Some(state) = weak.upgrade() else { break };
        let mut queues = state.queues.lock().expect("vision queue mutex poisoned");
        if queues.pending.is_empty() {
            let (guard, _) = state
                .wake
                .wait_timeout(queues, std::time::Duration::from_millis(100))
                .expect("vision queue mutex poisoned while waiting");
            queues = guard;
        }
        let Some(job) = queues.pending.pop_front() else {
            drop(queues);
            drop(state);
            continue;
        };
        queues.pending_bytes = queues.pending_bytes.saturating_sub(job.estimated_bytes());
        drop(queues);

        let started_wall_ms = vision_wall_time_ms();
        if started_wall_ms > job.deadline_ms && job.deadline_ms > 1_000_000_000_000 {
            state
                .stats
                .lock()
                .expect("vision stats mutex poisoned")
                .expired_frames += 1;
            continue;
        }
        let started = std::time::Instant::now();
        let processed = state
            .backend
            .preprocess(&job.frame, &state.config)
            .and_then(|prepared| {
                state
                    .backend
                    .detect(&prepared, state.config.maximum_detections)
                    .map(|proposals| (prepared, proposals))
            });
        let duration_ms = (started.elapsed().as_millis() as u64).max(1);
        let completed_at_ms = if job.enqueued_at_ms > 1_000_000_000_000 {
            vision_wall_time_ms()
        } else {
            job.enqueued_at_ms.saturating_add(duration_ms)
        };
        let (_prepared, mut proposals) = match processed {
            Ok(value) => value,
            Err(error) => {
                let mut stats = state.stats.lock().expect("vision stats mutex poisoned");
                stats.failed_frames += 1;
                stats.latest_error = Some(error.to_string());
                continue;
            }
        };
        if completed_at_ms > job.deadline_ms {
            state
                .stats
                .lock()
                .expect("vision stats mutex poisoned")
                .expired_frames += 1;
            continue;
        }
        let track_ids = tracker.assign(
            &proposals,
            job.producer_timestamp_ms(),
            job.calibration_epoch,
            &state.config,
        );
        let depth_associations = associate_depth_batch(
            &job,
            &proposals.iter().map(|proposal| proposal.bbox).collect::<Vec<_>>(),
        );
        let model = state.backend.identity();
        let detections = proposals
            .drain(..)
            .zip(track_ids)
            .zip(depth_associations)
            .enumerate()
            .map(|(index, ((proposal, track_id), depth_association))| {
                let (position, position_unavailable_reasons) = depth_association;
                VisionDetection {
                    source_frame_id: job.source_frame_id.clone(),
                    source_sensation_id: job.source_sensation_id.clone(),
                    descendant_sensation_id: format!(
                        "{}-detection-{index}",
                        job.source_sensation_id
                    ),
                    source_snapshot_id: job.source_snapshot_id.clone(),
                    source_stream: job.source_stream.clone(),
                    producer_timestamp_ms: job.frame.captured_at_ms,
                    image_width: job.frame.width,
                    image_height: job.frame.height,
                    bbox: proposal.bbox,
                    labels: proposal.labels,
                    model: model.clone(),
                    inference_started_at_ms: completed_at_ms.saturating_sub(duration_ms),
                    inference_completed_at_ms: completed_at_ms,
                    inference_duration_ms: duration_ms,
                    deadline_ms: job.deadline_ms,
                    track_id: Some(track_id),
                    calibration_epoch: job.calibration_epoch,
                    geometry_trust: geometry_trust(&job),
                    position,
                    position_unavailable_reasons,
                    crop_rgb8: crop_source_rgb8(&job.frame, proposal.bbox),
                }
            })
            .collect();
        {
            let mut stats = state.stats.lock().expect("vision stats mutex poisoned");
            stats.processed_frames += 1;
            stats.latest_error = None;
            stats.inference_ms.push_back(duration_ms);
            while stats.inference_ms.len() > 256 {
                stats.inference_ms.pop_front();
            }
        }
        let mut queues = state.queues.lock().expect("vision queue mutex poisoned");
        let batch = VisionBatch {
            detections,
            calibration_epoch: job.calibration_epoch,
        };
        let batch_bytes = batch.estimated_bytes();
        let memory_limit = state.config.memory_limit_mb.saturating_mul(1024 * 1024);
        while queues.completed.len() >= state.config.result_capacity.max(1)
            || queues
                .pending_bytes
                .saturating_add(queues.completed_bytes)
                .saturating_add(batch_bytes)
                > memory_limit
        {
            let Some(dropped) = queues.completed.pop_front() else {
                break;
            };
            queues.completed_bytes = queues
                .completed_bytes
                .saturating_sub(dropped.estimated_bytes());
            state
                .stats
                .lock()
                .expect("vision stats mutex poisoned")
                .dropped_frames += 1;
        }
        if batch_bytes <= memory_limit {
            queues.completed_bytes = queues.completed_bytes.saturating_add(batch_bytes);
            queues.completed.push_back(batch);
        } else {
            let mut stats = state.stats.lock().expect("vision stats mutex poisoned");
            stats.dropped_frames += 1;
            stats.latest_error = Some(format!(
                "completed evidence needs {batch_bytes} bytes, exceeding the {} MiB queued-evidence budget",
                state.config.memory_limit_mb
            ));
        }
    }
}

impl VisionJob {
    fn producer_timestamp_ms(&self) -> u64 {
        if self.frame.captured_at_ms > 0 {
            self.frame.captured_at_ms
        } else {
            self.enqueued_at_ms
        }
    }
}

#[derive(Clone, Debug)]
struct Track {
    id: String,
    bbox: VisionBoundingBox,
    last_seen_ms: u64,
    calibration_epoch: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct ShortTermTracker {
    next_id: u64,
    tracks: Vec<Track>,
}

impl ShortTermTracker {
    pub fn assign(
        &mut self,
        proposals: &[VisionProposal],
        timestamp_ms: u64,
        calibration_epoch: Option<u64>,
        config: &VisionPipelineConfig,
    ) -> Vec<String> {
        self.tracks.retain(|track| {
            track.calibration_epoch == calibration_epoch
                && timestamp_ms.saturating_sub(track.last_seen_ms) <= config.track_max_age_ms
        });
        let matchable_track_count = self.tracks.len();
        let mut used = vec![false; matchable_track_count];
        let mut ids = Vec::with_capacity(proposals.len());
        for proposal in proposals {
            let matched = self
                .tracks
                .iter()
                .take(matchable_track_count)
                .enumerate()
                .filter(|(index, _)| !used[*index])
                .map(|(index, track)| (index, bbox_iou(proposal.bbox, track.bbox)))
                .filter(|(_, iou)| *iou >= config.track_iou_threshold)
                .max_by(|left, right| {
                    left.1
                        .partial_cmp(&right.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some((index, _)) = matched {
                used[index] = true;
                self.tracks[index].bbox = proposal.bbox;
                self.tracks[index].last_seen_ms = timestamp_ms;
                ids.push(self.tracks[index].id.clone());
            } else {
                self.next_id += 1;
                let id = format!("vision-track-{}", self.next_id);
                self.tracks.push(Track {
                    id: id.clone(),
                    bbox: proposal.bbox,
                    last_seen_ms: timestamp_ms,
                    calibration_epoch,
                });
                ids.push(id);
            }
        }
        ids
    }
}

fn bbox_iou(left: VisionBoundingBox, right: VisionBoundingBox) -> f32 {
    let x0 = left.x.max(right.x);
    let y0 = left.y.max(right.y);
    let x1 = left
        .x
        .saturating_add(left.width)
        .min(right.x.saturating_add(right.width));
    let y1 = left
        .y
        .saturating_add(left.height)
        .min(right.y.saturating_add(right.height));
    let intersection = x1.saturating_sub(x0) as u64 * y1.saturating_sub(y0) as u64;
    let union = left.width as u64 * left.height as u64 + right.width as u64 * right.height as u64
        - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

#[cfg(test)]
fn associate_depth(
    job: &VisionJob,
    bbox: VisionBoundingBox,
) -> (Option<VisionPositionEstimate>, Vec<String>) {
    associate_depth_batch(job, &[bbox])
        .pop()
        .unwrap_or_else(|| (None, vec!["detection bounding box unavailable".to_string()]))
}

fn associate_depth_batch(
    job: &VisionJob,
    bboxes: &[VisionBoundingBox],
) -> Vec<(Option<VisionPositionEstimate>, Vec<String>)> {
    let unavailable = |reason: &str| {
        bboxes
            .iter()
            .map(|_| (None, vec![reason.to_string()]))
            .collect()
    };
    let Some(kinect) = job.kinect.as_ref() else {
        return unavailable("depth snapshot unavailable");
    };
    if job.frame.rgbd_frame_id.is_none() || job.frame.rgbd_frame_id != kinect.rgbd_frame_id {
        return unavailable("RGB and depth frame identities do not match");
    }
    let Some(calibration) = kinect.geometry_calibration else {
        return unavailable("depth calibration unavailable");
    };
    if !calibration.physical_validation_ready() {
        return unavailable("depth calibration is not physically validated");
    }
    if !pete_now::DepthGeometry::live_transform_trusted(kinect) {
        return unavailable("active calibration epoch is not trusted");
    }
    let Some(geometry) = pete_now::DepthGeometry::from_kinect(kinect) else {
        return unavailable("depth geometry is invalid");
    };
    let Some(rgb) = calibration.rgb else {
        return unavailable("color-to-depth registration intrinsics are unavailable");
    };
    if calibration.depth_to_rgb.is_none() {
        return unavailable("color-to-depth camera extrinsics are unavailable");
    }
    if rgb.width != job.frame.width || rgb.height != job.frame.height {
        return unavailable("RGB frame dimensions do not match registration calibration");
    }
    if calibration.depth.width != kinect.depth_width
        || calibration.depth.height != kinect.depth_height
    {
        return unavailable("depth frame dimensions do not match registration calibration");
    }

    // Reproject each depth sample into the RGB optical frame once, then assign
    // it to the central region of matching RGB detections. Equal image sizes
    // are deliberately irrelevant: Kinect color and depth pixels are not the
    // same rays.
    let windows = bboxes
        .iter()
        .map(|bbox| {
            let center_x = bbox.x as f32 + bbox.width as f32 * 0.5;
            let center_y = bbox.y as f32 + bbox.height as f32 * 0.5;
            let radius_x = (bbox.width as f32 * 0.125).max(1.0);
            let radius_y = (bbox.height as f32 * 0.125).max(1.0);
            [
                center_x - radius_x,
                center_y - radius_y,
                center_x + radius_x,
                center_y + radius_y,
            ]
        })
        .collect::<Vec<_>>();
    let mut samples = vec![Vec::<[f32; 3]>::new(); bboxes.len()];
    for y in 0..kinect.depth_height {
        for x in 0..kinect.depth_width {
            let index = y as usize * kinect.depth_width as usize + x as usize;
            let Some(raw_depth_m) = kinect
                .depth_m
                .get(index)
                .copied()
                .filter(|value| value.is_finite() && *value > 0.0)
            else {
                continue;
            };
            let Some(camera) = geometry.depth_pixel_to_camera(x as f32, y as f32, raw_depth_m)
            else {
                continue;
            };
            if (kinect.min_depth_m > 0.0 && camera[2] < kinect.min_depth_m)
                || (kinect.max_depth_m > 0.0 && camera[2] > kinect.max_depth_m)
            {
                continue;
            }
            let Some([rgb_x, rgb_y]) = geometry.depth_point_to_rgb_pixel(camera) else {
                continue;
            };
            for (index, [x0, y0, x1, y1]) in windows.iter().copied().enumerate() {
                if rgb_x >= x0 && rgb_x <= x1 && rgb_y >= y0 && rgb_y <= y1 {
                    samples[index].push(camera);
                }
            }
        }
    }

    samples
        .into_iter()
        .map(|mut samples| {
            if samples.is_empty() {
                return (
                    None,
                    vec!["no registered depth samples inside RGB detection".to_string()],
                );
            }
            samples.sort_by(|left, right| {
                left[2]
                    .partial_cmp(&right[2])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            position_from_registered_depth(job, geometry, calibration, samples[samples.len() / 2])
        })
        .collect()
}

fn position_from_registered_depth(
    job: &VisionJob,
    geometry: pete_now::DepthGeometry,
    calibration: pete_now::DepthGeometryCalibration,
    camera: [f32; 3],
) -> (Option<VisionPositionEstimate>, Vec<String>) {
    let mut reasons = Vec::new();
    let kinect = job
        .kinect
        .as_ref()
        .expect("registered depth requires a Kinect snapshot");
    let robot_position_m = geometry.depth_point_to_base(camera);
    let world_position_m = kinect.fusion_alignment.as_ref().and_then(|alignment| {
        if alignment.confidence < 0.5 {
            reasons.push("world pose alignment confidence is too low".to_string());
            None
        } else {
            Some(pete_now::DepthGeometry::base_point_to_world(
                robot_position_m,
                alignment.pose,
                None,
                None,
            ))
        }
    });
    if kinect.fusion_alignment.is_none() {
        reasons.push("world pose alignment unavailable".to_string());
    }
    let uncertainty_m = kinect
        .live_geometry_calibration
        .as_ref()
        .map(|estimate| {
            estimate.covariance[..3]
                .iter()
                .sum::<f32>()
                .sqrt()
                .max(0.01)
        })
        .or_else(|| {
            calibration
                .validation
                .map(|validation| validation.max_plane_distance_error_m.max(0.01))
        })
        .unwrap_or(0.1);
    (
        Some(VisionPositionEstimate {
            depth_m: camera[2],
            robot_position_m,
            world_position_m,
            uncertainty_m,
        }),
        reasons,
    )
}

fn geometry_trust(job: &VisionJob) -> String {
    let Some(kinect) = job.kinect.as_ref() else {
        return "unavailable".to_string();
    };
    let Some(calibration) = kinect.geometry_calibration else {
        return "missing".to_string();
    };
    if !calibration.physical_validation_ready() {
        return "unvalidated".to_string();
    }
    kinect
        .live_geometry_calibration
        .as_ref()
        .map(|estimate| format!("{:?}", estimate.trust_state).to_ascii_lowercase())
        .unwrap_or_else(|| "configured".to_string())
}

fn crop_source_rgb8(frame: &EyeFrame, bbox: VisionBoundingBox) -> Vec<u8> {
    let pixel_count = frame.width as usize * frame.height as usize;
    let rgb = match frame.format {
        EyeFrameFormat::Rgb8 if frame.bytes.len() >= pixel_count * 3 => frame.bytes.clone(),
        EyeFrameFormat::Bgr8 if frame.bytes.len() >= pixel_count * 3 => frame
            .bytes
            .chunks_exact(3)
            .take(pixel_count)
            .flat_map(|pixel| [pixel[2], pixel[1], pixel[0]])
            .collect(),
        EyeFrameFormat::Gray8 if frame.bytes.len() >= pixel_count => frame
            .bytes
            .iter()
            .take(pixel_count)
            .flat_map(|value| [*value, *value, *value])
            .collect(),
        _ => return Vec::new(),
    };
    let x0 = bbox.x as usize;
    let y0 = bbox.y as usize;
    let x1 = bbox.x.saturating_add(bbox.width).min(frame.width) as usize;
    let y1 = bbox.y.saturating_add(bbox.height).min(frame.height) as usize;
    if x0 >= x1 || y0 >= y1 {
        return Vec::new();
    }
    let mut crop = Vec::with_capacity((x1 - x0) * (y1 - y0) * 3);
    for y in y0..y1 {
        let start = (y * frame.width as usize + x0) * 3;
        let end = (y * frame.width as usize + x1) * 3;
        crop.extend_from_slice(&rgb[start..end]);
    }
    crop
}

fn frame_id(frame: &EyeFrame) -> String {
    frame.rgbd_frame_id.clone().unwrap_or_else(|| {
        format!(
            "eye-{}-{}x{}-{:016x}",
            frame.captured_at_ms,
            frame.width,
            frame.height,
            stable_hash64(&frame.bytes)
        )
    })
}

fn vision_wall_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
