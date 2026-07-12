#[cfg(any(feature = "face", feature = "linux-hardware", test))]
use anyhow::Context;
use anyhow::Result;
use async_trait::async_trait;
use pete_actions::{ActionPrimitive, LlmActionProposal};
use pete_body::BodySense;
use pete_now::{
    AsrSense, EarSense, ExtensionSense, EyeSense, FaceSense, GpsSense, ImuSense, KinectSense,
    ObjectClass, ObjectObservation, ObjectObservationSource, ObjectSense, RangeSense,
    TranscriptCandidateEvent, TranscriptCandidateTracker, TranscriptStabilityState, VectorArtifact,
    VoiceSense, FACE_VECTOR_COLLECTION, IMAGE_DESCRIPTION_VECTOR_COLLECTION,
    IMAGE_VECTOR_COLLECTION, OBJECT_VECTOR_COLLECTION, SCENE_VECTOR_COLLECTION,
    TRANSCRIPT_VECTOR_COLLECTION,
};
use pete_now::{Now, PredictionSense, SurpriseSense};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
#[cfg(any(feature = "face", feature = "linux-hardware"))]
use std::sync::Mutex;

mod surface;

pub use surface::{
    anticipate_surface_frame, anticipate_surfaces, AnticipatedNavigation, Bounds2,
    ClusterObservation, OccupancyCell, OccupancyGrid, OccupancyState, PlaneObservation, Point3,
    ProjectedCluster, ProjectedSurface, SceneGraphSummary, SurfaceAnticipationFrame,
    SurfaceExtractor, SurfaceExtractorConfig, SurfaceExtractorDiagnostics, SurfaceExtractorOutput,
    SurfaceHypothesis, SurfaceKind, SurfacePrimitiveKind, SurfaceTrack, Vec3,
};

type TimeMs = u64;

#[cfg(feature = "linux-hardware")]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(feature = "linux-hardware")]
use serialport::SerialPort;
#[cfg(feature = "linux-hardware")]
use std::io::Write;
#[cfg(feature = "linux-hardware")]
use std::io::{ErrorKind, Read};
#[cfg(feature = "linux-hardware")]
use std::os::fd::AsRawFd;
#[cfg(feature = "linux-hardware")]
use std::time::{Duration, Instant};
#[cfg(feature = "linux-hardware")]
use v4l::buffer::Type;
#[cfg(feature = "linux-hardware")]
use v4l::io::traits::CaptureStream;
#[cfg(feature = "linux-hardware")]
use v4l::prelude::{MmapStream, *};
#[cfg(feature = "linux-hardware")]
use v4l::video::Capture;
#[cfg(feature = "linux-hardware")]
use v4l::{Format, FourCC};

#[cfg(feature = "kinect-freenect")]
#[link(name = "freenect_sync")]
unsafe extern "C" {
    fn freenect_sync_get_depth_with_res(
        depth: *mut *mut std::ffi::c_void,
        timestamp: *mut u32,
        index: i32,
        resolution: i32,
        format: i32,
    ) -> i32;
    fn freenect_sync_get_video_with_res(
        video: *mut *mut std::ffi::c_void,
        timestamp: *mut u32,
        index: i32,
        resolution: i32,
        format: i32,
    ) -> i32;
    fn freenect_sync_stop();
}

#[cfg(feature = "kinect-freenect")]
const FREENECT_RESOLUTION_MEDIUM: i32 = 1;
#[cfg(feature = "kinect-freenect")]
const FREENECT_DEPTH_MM: i32 = 5;
#[cfg(feature = "kinect-freenect")]
const FREENECT_VIDEO_RGB: i32 = 0;
#[cfg(feature = "kinect-freenect")]
const FREENECT_DEPTH_WIDTH: usize = 640;
#[cfg(feature = "kinect-freenect")]
const FREENECT_DEPTH_HEIGHT: usize = 480;
#[cfg(feature = "kinect-freenect")]
const FREENECT_DEPTH_PIXELS: usize = FREENECT_DEPTH_WIDTH * FREENECT_DEPTH_HEIGHT;
#[cfg(feature = "kinect-freenect")]
const FREENECT_RGB_BYTES: usize = FREENECT_DEPTH_PIXELS * 3;
#[cfg(feature = "kinect-freenect")]
const KINECT_V1_DEPTH_FX: f32 = 594.0;
#[cfg(feature = "kinect-freenect")]
const KINECT_V1_DEPTH_FY: f32 = 591.0;
#[cfg(feature = "kinect-freenect")]
const KINECT_V1_DEPTH_CX: f32 = 339.0;
#[cfg(feature = "kinect-freenect")]
const KINECT_V1_DEPTH_CY: f32 = 242.0;

#[async_trait]
pub trait SenseProducer {
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
}

#[derive(Clone, Default)]
pub struct FrameProcessor {
    last_processed_frame_key: Option<FrameKey>,
    face_detector: Option<Arc<dyn FaceDetector>>,
    object_detector: Option<Arc<dyn ObjectDetector>>,
    kinect_range_projection: Option<DepthRangeProjectionConfig>,
}

impl std::fmt::Debug for FrameProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameProcessor")
            .field("last_processed_frame_key", &self.last_processed_frame_key)
            .field("face_detector", &self.face_detector.is_some())
            .field("object_detector", &self.object_detector.is_some())
            .field("kinect_range_projection", &self.kinect_range_projection)
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

    pub fn build(
        &mut self,
        t_ms: TimeMs,
        mut body: BodySense,
        packets: Vec<SensePacket>,
    ) -> Result<Now> {
        body.last_update_ms = body.last_update_ms.max(t_ms);
        self.last_updates.body = Some(body.last_update_ms);
        self.last_snapshot.body = body;
        self.last_snapshot.extensions.clear();
        let mut saw_range_packet = false;

        for packet in packets {
            match packet {
                SensePacket::Eye(eye) => {
                    self.last_snapshot.eye = eye;
                    self.last_updates.eye = Some(t_ms);
                }
                SensePacket::EyeFrame(frame) => {
                    self.last_snapshot.eye.frames = vec![bytes_to_unit_signal(&frame.bytes)];
                    self.last_snapshot.eye_frame = Some(frame);
                    self.last_updates.eye = Some(t_ms);
                }
                SensePacket::Ear(ear) => {
                    self.last_snapshot.ear = ear;
                    self.last_updates.ear = Some(t_ms);
                }
                SensePacket::EarPcm(frame) => {
                    self.last_snapshot.ear.features = vec![pcm_to_unit_signal(&frame.samples)];
                    self.last_snapshot.ear_pcm = Some(frame);
                    self.last_updates.ear = Some(t_ms);
                }
                SensePacket::Range(range) => {
                    self.last_snapshot.range = range;
                    self.last_updates.range = Some(t_ms);
                    saw_range_packet = true;
                }
                SensePacket::Imu(imu) => {
                    self.last_snapshot.imu = imu;
                    self.last_updates.imu = Some(t_ms);
                }
                SensePacket::Gps(gps) => {
                    self.last_snapshot.gps = Some(gps);
                    self.last_updates.gps = Some(t_ms);
                }
                SensePacket::Kinect(kinect) => {
                    if !saw_range_packet {
                        if let Some(range) = range_from_kinect_depth(&kinect) {
                            self.last_snapshot.range = range;
                            self.last_updates.range = Some(t_ms);
                        }
                    }
                    self.last_snapshot.kinect = kinect;
                    self.last_updates.kinect = Some(t_ms);
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
        return range_from_depth_image(depth, projection, beam_count, transform);
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
        beams,
        nearest_m,
        beam_angles_rad: Vec::new(),
        frame: None,
        source: Some("kinect_depth_legacy_range".to_string()),
    })
}

#[derive(Clone, Copy, Debug)]
struct RangeDepthProjection {
    width: usize,
    height: usize,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
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
            fx: positive_or(kinect.depth_fx, 594.0),
            fy: positive_or(kinect.depth_fy, 591.0),
            cx: positive_or(kinect.depth_cx, (width as f32 - 1.0) * 0.5),
            cy: positive_or(kinect.depth_cy, (height as f32 - 1.0) * 0.5),
            min_depth_m,
            max_depth_m,
        })
    }
}

fn range_from_depth_image(
    depth: &[f32],
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
    let mut angles = (0..beam_count)
        .map(|beam| {
            let u = ((beam as f32 + 0.5) * projection.width as f32 / beam_count as f32)
                .clamp(0.0, projection.width.saturating_sub(1) as f32);
            let v = (projection.height as f32 - 1.0) * 0.5;
            let camera = depth_image_camera_point(u, v, 1.0, projection);
            robot_angle_for_camera_point(camera, transform)
        })
        .collect::<Vec<_>>();
    let mut saw_valid = false;

    for y in row_start..row_end {
        let row = y * projection.width;
        for x in 0..projection.width {
            let depth_m = depth[row + x] * transform.depth_scale;
            if !depth_m.is_finite()
                || depth_m < projection.min_depth_m
                || depth_m > projection.max_depth_m
            {
                continue;
            }
            let beam = (x * beam_count / projection.width).min(beam_count - 1);
            let camera = depth_image_camera_point(x as f32, y as f32, depth_m, projection);
            let robot = depth_camera_point_to_robot(camera, transform);
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
        beams,
        nearest_m,
        beam_angles_rad: angles,
        frame: Some("robot_base".to_string()),
        source: Some("kinect_depth_image".to_string()),
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
        beams,
        nearest_m,
        beam_angles_rad: angles,
        frame: Some("robot_base".to_string()),
        source: Some("kinect_compact_depth".to_string()),
    })
}

fn depth_image_camera_point(
    u: f32,
    v: f32,
    depth_m: f32,
    projection: RangeDepthProjection,
) -> [f32; 3] {
    [
        (u - projection.cx) * depth_m / projection.fx.max(f32::EPSILON),
        (v - projection.cy) * depth_m / projection.fy.max(f32::EPSILON),
        depth_m,
    ]
}

fn robot_angle_for_camera_point(camera: [f32; 3], transform: DepthRangeProjectionConfig) -> f32 {
    let robot = depth_camera_point_to_robot(camera, transform);
    robot[1].atan2(robot[0])
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

    pub fn with_kinect_range_projection(mut self, config: DepthRangeProjectionConfig) -> Self {
        self.kinect_range_projection = Some(config);
        self
    }

    pub fn process_packets(&mut self, t_ms: TimeMs, packets: &mut Vec<SensePacket>) {
        self.process_kinect_range_packets(packets);

        let Some(frame) = packets.iter().rev().find_map(|packet| match packet {
            SensePacket::EyeFrame(frame) => Some(frame),
            _ => None,
        }) else {
            return;
        };
        let Some(processed) = self.process_frame(t_ms, frame) else {
            return;
        };
        let summary_values = summary_extension_values(&processed);
        packets.push(SensePacket::Eye(processed.eye));
        if !processed.face.embeddings.is_empty() || !processed.face.vectors.is_empty() {
            packets.push(SensePacket::Face(processed.face));
        }
        if !processed.objects.observations.is_empty()
            || !processed.objects.embeddings.is_empty()
            || !processed.objects.vectors.is_empty()
        {
            packets.push(SensePacket::Objects(processed.objects));
        }
        packets.push(SensePacket::Extension(ExtensionSense {
            schema_version: 1,
            name: "vision.frame_summary".to_string(),
            values: summary_values,
        }));
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
        let Some(frame) = snapshot.eye_frame.clone() else {
            return;
        };
        let Some(processed) = self.process_frame(t_ms, &frame) else {
            return;
        };
        let summary_values = summary_extension_values(&processed);
        snapshot.eye = processed.eye;
        if !processed.face.embeddings.is_empty() || !processed.face.vectors.is_empty() {
            snapshot.face = processed.face;
        }
        if !processed.objects.observations.is_empty()
            || !processed.objects.embeddings.is_empty()
            || !processed.objects.vectors.is_empty()
        {
            snapshot.objects = processed.objects;
        }
        snapshot.extensions.push(ExtensionSense {
            schema_version: 1,
            name: "vision.frame_summary".to_string(),
            values: summary_values,
        });
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

pub trait FaceDetector: Send + Sync {
    fn detect_faces(&self, frame: &EyeFrame) -> Result<Vec<FaceDetection>>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct FaceDetection {
    pub face_id: String,
    pub source_frame_id: Option<String>,
    pub embedding: Vec<f32>,
    pub model: String,
}

pub trait ObjectDetector: Send + Sync {
    fn detect_objects(&self, frame: &EyeFrame) -> Result<Vec<ObjectDetection>>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObjectDetection {
    pub object_id: String,
    pub label: String,
    pub class: ObjectClass,
    pub bearing_rad: f32,
    pub distance_m: Option<f32>,
    pub confidence: f32,
    pub source: ObjectObservationSource,
    pub source_frame_id: Option<String>,
    pub embedding: Vec<f32>,
    pub model: String,
}

#[cfg(feature = "face")]
pub struct FaceIdDetector {
    analyzer: Arc<Mutex<face_id::analyzer::FaceAnalyzer>>,
}

#[cfg(feature = "face")]
impl FaceIdDetector {
    pub async fn from_hf() -> Result<Self> {
        let analyzer = face_id::analyzer::FaceAnalyzer::from_hf()
            .build()
            .await
            .context("failed to initialize face_id analyzer")?;
        Ok(Self {
            analyzer: Arc::new(Mutex::new(analyzer)),
        })
    }
}

#[cfg(feature = "face")]
impl FaceDetector for FaceIdDetector {
    fn detect_faces(&self, frame: &EyeFrame) -> Result<Vec<FaceDetection>> {
        let image = dynamic_image_from_eye_frame(frame)?;
        let faces = self
            .analyzer
            .lock()
            .map_err(|_| anyhow::anyhow!("face analyzer lock poisoned"))?
            .analyze(&image)
            .context("face_id analysis failed")?;
        Ok(faces
            .into_iter()
            .enumerate()
            .map(|(index, face)| FaceDetection {
                face_id: face_detection_id(frame, index, &face.embedding),
                source_frame_id: None,
                embedding: face.embedding,
                model: "face_id.0.4.1".to_string(),
            })
            .collect())
    }
}

fn process_eye_frame(
    t_ms: TimeMs,
    frame: &EyeFrame,
    face_detector: Option<&dyn FaceDetector>,
    object_detector: Option<&dyn ObjectDetector>,
) -> ProcessedFrame {
    let source_frame_id = format!(
        "eye-{}-{}x{}-{}",
        frame.captured_at_ms,
        frame.width,
        frame.height,
        frame.bytes.len()
    );
    let signal = bytes_to_unit_signal(&frame.bytes);
    let mut eye = EyeSense {
        schema_version: 1,
        frames: vec![signal.clone()],
        ..EyeSense::default()
    };
    eye.image_vectors.push(
        VectorArtifact::new(
            IMAGE_VECTOR_COLLECTION,
            source_frame_id.clone(),
            signal.clone(),
        )
        .with_model("raw-byte-unit-signal-v0")
        .with_source_frame_id(source_frame_id.clone())
        .with_occurred_at_ms(t_ms),
    );
    eye.image_description_vectors.push(
        VectorArtifact::new(
            IMAGE_DESCRIPTION_VECTOR_COLLECTION,
            format!("{source_frame_id}-summary"),
            frame_summary_vector(frame, &signal),
        )
        .with_model("frame-summary-v0")
        .with_source_frame_id(source_frame_id.clone())
        .with_occurred_at_ms(t_ms),
    );
    eye.scene_vectors.push(
        VectorArtifact::new(
            SCENE_VECTOR_COLLECTION,
            format!("{source_frame_id}-scene"),
            frame_summary_vector(frame, &signal),
        )
        .with_model("scene-summary-v0")
        .with_source_frame_id(source_frame_id.clone())
        .with_occurred_at_ms(t_ms),
    );

    let face = match face_detector {
        Some(detector) => detected_face_sense(t_ms, frame, &source_frame_id, detector),
        None => Ok(FaceSense {
            schema_version: 1,
            ..FaceSense::default()
        }),
    }
    .unwrap_or_else(|_| FaceSense {
        schema_version: 1,
        ..FaceSense::default()
    });

    let objects = match object_detector {
        Some(detector) => detected_object_sense(t_ms, frame, &source_frame_id, detector),
        None => Ok(ObjectSense {
            schema_version: 1,
            ..ObjectSense::default()
        }),
    }
    .unwrap_or_else(|_| ObjectSense {
        schema_version: 1,
        ..ObjectSense::default()
    });

    ProcessedFrame {
        eye,
        face,
        objects,
        summary: format!(
            "{:?} frame {}x{}, {} bytes",
            frame.format,
            frame.width,
            frame.height,
            frame.bytes.len()
        ),
        source_frame_id,
    }
}

fn detected_face_sense(
    t_ms: TimeMs,
    frame: &EyeFrame,
    source_frame_id: &str,
    detector: &dyn FaceDetector,
) -> Result<FaceSense> {
    let detections = detector.detect_faces(frame)?;
    let mut face = FaceSense {
        schema_version: 1,
        ..FaceSense::default()
    };
    for detection in detections {
        if detection.embedding.is_empty() {
            continue;
        }
        face.embeddings.push(detection.embedding.clone());
        face.vectors.push(
            VectorArtifact::new(
                FACE_VECTOR_COLLECTION,
                detection.face_id,
                detection.embedding,
            )
            .with_model(detection.model)
            .with_source_frame_id(
                detection
                    .source_frame_id
                    .unwrap_or_else(|| source_frame_id.to_string()),
            )
            .with_occurred_at_ms(t_ms),
        );
    }
    Ok(face)
}

fn detected_object_sense(
    t_ms: TimeMs,
    frame: &EyeFrame,
    source_frame_id: &str,
    detector: &dyn ObjectDetector,
) -> Result<ObjectSense> {
    let detections = detector.detect_objects(frame)?;
    let mut objects = ObjectSense {
        schema_version: 1,
        ..ObjectSense::default()
    };
    for detection in detections {
        let source_frame_id = detection
            .source_frame_id
            .clone()
            .unwrap_or_else(|| source_frame_id.to_string());
        objects.observations.push(ObjectObservation {
            label: detection.label,
            class: detection.class,
            bearing_rad: detection.bearing_rad,
            distance_m: detection.distance_m,
            confidence: detection.confidence,
            source: detection.source,
        });
        if detection.embedding.is_empty() {
            continue;
        }
        objects.embeddings.push(detection.embedding.clone());
        objects.vectors.push(
            VectorArtifact::new(
                OBJECT_VECTOR_COLLECTION,
                detection.object_id,
                detection.embedding,
            )
            .with_model(detection.model)
            .with_source_frame_id(source_frame_id)
            .with_occurred_at_ms(t_ms),
        );
    }
    Ok(objects)
}

fn face_detection_id(frame: &EyeFrame, index: usize, embedding: &[f32]) -> String {
    let mut hash = stable_hash64(&frame.captured_at_ms.to_le_bytes());
    hash ^= stable_hash64(&frame.width.to_le_bytes()).rotate_left(7);
    hash ^= stable_hash64(&frame.height.to_le_bytes()).rotate_left(13);
    hash ^= stable_hash64(&index.to_le_bytes()).rotate_left(19);
    for value in embedding.iter().take(32) {
        hash ^= stable_hash64(&value.to_bits().to_le_bytes()).rotate_left(3);
    }
    format!("face-{hash:016x}-{index}")
}

fn stable_hash64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    bytes.iter().fold(FNV_OFFSET, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

#[cfg(feature = "face")]
fn dynamic_image_from_eye_frame(frame: &EyeFrame) -> Result<image::DynamicImage> {
    match frame.format {
        EyeFrameFormat::Mjpeg => {
            image::load_from_memory(&frame.bytes).context("failed to decode MJPEG eye frame")
        }
        EyeFrameFormat::Rgb8 => {
            image::RgbImage::from_raw(frame.width, frame.height, frame.bytes.clone())
                .map(image::DynamicImage::ImageRgb8)
                .context("RGB eye frame byte length did not match dimensions")
        }
        EyeFrameFormat::Bgr8 => {
            let mut rgb = frame.bytes.clone();
            for pixel in rgb.chunks_exact_mut(3) {
                pixel.swap(0, 2);
            }
            image::RgbImage::from_raw(frame.width, frame.height, rgb)
                .map(image::DynamicImage::ImageRgb8)
                .context("BGR eye frame byte length did not match dimensions")
        }
        EyeFrameFormat::Gray8 => {
            image::GrayImage::from_raw(frame.width, frame.height, frame.bytes.clone())
                .map(image::DynamicImage::ImageLuma8)
                .context("gray eye frame byte length did not match dimensions")
        }
        _ => anyhow::bail!(
            "unsupported eye frame format for face detection: {:?}",
            frame.format
        ),
    }
}

fn summary_extension_values(processed: &ProcessedFrame) -> Vec<f32> {
    let signal = processed
        .eye
        .frames
        .first()
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mean = if signal.is_empty() {
        0.0
    } else {
        signal.iter().sum::<f32>() / signal.len() as f32
    };
    vec![
        signal.len() as f32,
        mean,
        processed.eye.image_vectors.len() as f32,
        processed.face.vectors.len() as f32,
    ]
}

fn frame_summary_vector(frame: &EyeFrame, signal: &[f32]) -> Vec<f32> {
    let mean = if signal.is_empty() {
        0.0
    } else {
        signal.iter().sum::<f32>() / signal.len() as f32
    };
    vec![
        frame.width as f32,
        frame.height as f32,
        frame.bytes.len() as f32,
        mean,
    ]
}

impl SensorUpdateTimes {
    fn age_ms(&self, t_ms: TimeMs) -> serde_json::Value {
        serde_json::json!({
            "body": self.body.map(|value| t_ms.saturating_sub(value)),
            "eye": self.eye.map(|value| t_ms.saturating_sub(value)),
            "ear": self.ear.map(|value| t_ms.saturating_sub(value)),
            "range": self.range.map(|value| t_ms.saturating_sub(value)),
            "imu": self.imu.map(|value| t_ms.saturating_sub(value)),
            "gps": self.gps.map(|value| t_ms.saturating_sub(value)),
            "kinect": self.kinect.map(|value| t_ms.saturating_sub(value)),
            "face": self.face.map(|value| t_ms.saturating_sub(value)),
            "voice": self.voice.map(|value| t_ms.saturating_sub(value)),
        })
    }
}

pub use pete_now::{EyeFrame, EyeFrameFormat};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PcmAudioFrame {
    pub captured_at_ms: u64,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples: Vec<i16>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AsrToolConfig {
    pub command: Option<String>,
    pub min_voice_rms: f32,
    pub min_chunk_ms: u64,
    pub max_chunk_ms: u64,
    pub silence_finalize_ms: u64,
}

impl Default for AsrToolConfig {
    fn default() -> Self {
        Self {
            command: std::env::var("PETE_ASR_COMMAND")
                .ok()
                .filter(|value| !value.trim().is_empty()),
            min_voice_rms: std::env::var("PETE_ASR_MIN_RMS")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .unwrap_or(0.012),
            min_chunk_ms: std::env::var("PETE_ASR_MIN_CHUNK_MS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(450),
            max_chunk_ms: std::env::var("PETE_ASR_MAX_CHUNK_MS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(8_000),
            silence_finalize_ms: std::env::var("PETE_ASR_SILENCE_FINALIZE_MS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(700),
        }
    }
}

#[derive(Clone, Debug)]
pub struct AsrTool {
    config: AsrToolConfig,
    chunk: Vec<i16>,
    chunk_start_ms: Option<u64>,
    chunk_end_ms: Option<u64>,
    last_voice_ms: Option<u64>,
    sequence: u64,
    transcript_tracker: TranscriptCandidateTracker,
}

impl AsrTool {
    pub fn new(config: AsrToolConfig) -> Self {
        Self {
            config,
            chunk: Vec::new(),
            chunk_start_ms: None,
            chunk_end_ms: None,
            last_voice_ms: None,
            sequence: 0,
            transcript_tracker: TranscriptCandidateTracker::new(),
        }
    }

    pub fn observe_frame(&mut self, frame: &PcmAudioFrame) -> Option<EarSense> {
        if frame.samples.is_empty() || frame.sample_rate_hz == 0 || frame.channels == 0 {
            return self.try_finalize(frame.captured_at_ms, frame.sample_rate_hz, frame.channels);
        }
        let rms = pcm_rms(&frame.samples);
        let voice = rms >= self.config.min_voice_rms;
        let duration_ms =
            pcm_duration_ms(frame.samples.len(), frame.sample_rate_hz, frame.channels);
        let frame_start = frame.captured_at_ms;
        let frame_end = frame_start.saturating_add(duration_ms);

        if voice {
            if self.chunk_start_ms.is_none() {
                self.chunk_start_ms = Some(frame_start);
            }
            self.chunk.extend_from_slice(&frame.samples);
            self.chunk_end_ms = Some(frame_end);
            self.last_voice_ms = Some(frame_end);
        } else if self.chunk_start_ms.is_some() {
            self.chunk_end_ms = Some(frame_end);
        }

        let chunk_duration = self
            .chunk_start_ms
            .zip(self.chunk_end_ms)
            .map(|(start, end)| end.saturating_sub(start))
            .unwrap_or_default();
        let silence_ms = self
            .last_voice_ms
            .map(|last| frame_end.saturating_sub(last))
            .unwrap_or_default();
        let should_finalize = chunk_duration >= self.config.max_chunk_ms
            || (chunk_duration >= self.config.min_chunk_ms
                && silence_ms >= self.config.silence_finalize_ms);
        if should_finalize {
            self.try_finalize(frame_end, frame.sample_rate_hz, frame.channels)
        } else {
            None
        }
    }

    fn try_finalize(
        &mut self,
        fallback_end_ms: u64,
        sample_rate_hz: u32,
        channels: u16,
    ) -> Option<EarSense> {
        let start_ms = self.chunk_start_ms?;
        let end_ms = self.chunk_end_ms.unwrap_or(fallback_end_ms);
        if end_ms.saturating_sub(start_ms) < self.config.min_chunk_ms || self.chunk.is_empty() {
            self.clear_chunk();
            return None;
        }
        let transcript = self
            .config
            .command
            .as_deref()
            .and_then(|command| {
                transcribe_with_command(command, &self.chunk, sample_rate_hz, channels).ok()
            })
            .map(|text| text.trim().to_string())
            .filter(|text| !text.is_empty())?;

        let sequence = self.sequence;
        self.sequence = self.sequence.saturating_add(1);
        let word_count = transcript.split_whitespace().count().min(u16::MAX as usize) as u16;
        let confidence = command_transcript_confidence(&transcript);
        let candidate_events =
            self.transcript_tracker
                .ingest_candidate(transcript.clone(), Some(confidence), true);
        let stability = transcript_stability_from_events(&candidate_events);
        let transcript_vector = transcript_vector_artifact(&transcript, sequence, start_ms, end_ms);
        self.clear_chunk();
        Some(EarSense {
            schema_version: 1,
            features: Vec::new(),
            transcript: Some(transcript.clone()),
            transcript_vectors: vec![transcript_vector],
            asr: AsrSense {
                transcript: Some(transcript.clone()),
                possible_transcript: None,
                committed_transcript: Some(transcript),
                is_final: true,
                confidence,
                sequence_start: Some(sequence),
                sequence_end: Some(sequence),
                candidate_id: stability.as_ref().map(|state| state.candidate_id.0),
                stable_text: stability.as_ref().map(|state| state.stable_text.clone()),
                unstable_text: stability.as_ref().map(|state| state.unstable_text.clone()),
                stable_word_prefix: stability
                    .as_ref()
                    .and_then(|state| state.stable_word_prefix.clone()),
                stable_word_count: stability
                    .as_ref()
                    .map(|state| state.stable_word_count.min(u16::MAX as usize) as u16),
                start_ms: Some(start_ms),
                end_ms: Some(end_ms),
                duration_ms: Some(end_ms.saturating_sub(start_ms)),
                sample_rate_hz: Some(sample_rate_hz),
                word_count: Some(word_count),
                speaker_confidence: None,
                candidate_events,
            },
        })
    }

    fn clear_chunk(&mut self) {
        self.chunk.clear();
        self.chunk_start_ms = None;
        self.chunk_end_ms = None;
        self.last_voice_ms = None;
    }
}

fn transcript_stability_from_events(
    events: &[TranscriptCandidateEvent],
) -> Option<TranscriptStabilityState> {
    events.iter().rev().find_map(|event| match event {
        TranscriptCandidateEvent::CandidateUpdated {
            id,
            text,
            stable_prefix_len,
            confidence,
        } => Some(TranscriptStabilityState::from_parts(
            *id,
            text,
            *stable_prefix_len,
            *confidence,
        )),
        TranscriptCandidateEvent::CandidateFinalized {
            id,
            text,
            confidence,
        } => Some(TranscriptStabilityState::from_parts(
            *id,
            text,
            text.len(),
            *confidence,
        )),
        TranscriptCandidateEvent::CandidateStarted { .. }
        | TranscriptCandidateEvent::CandidateReplaced { .. }
        | TranscriptCandidateEvent::CandidateCancelled { .. } => None,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldSnapshot {
    pub body: BodySense,
    pub final_selected_action: Option<ActionPrimitive>,
    pub llm_action_proposal: Option<LlmActionProposal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_debug: Option<serde_json::Value>,
    pub eye_frame: Option<EyeFrame>,
    pub ear_pcm: Option<PcmAudioFrame>,
    pub eye: EyeSense,
    pub ear: EarSense,
    pub range: RangeSense,
    pub imu: ImuSense,
    pub gps: Option<GpsSense>,
    pub kinect: KinectSense,
    pub objects: ObjectSense,
    pub face: FaceSense,
    pub voice: VoiceSense,
    pub extensions: Vec<ExtensionSense>,
}

impl Default for WorldSnapshot {
    fn default() -> Self {
        Self {
            body: BodySense::default(),
            final_selected_action: None,
            llm_action_proposal: None,
            action_debug: None,
            eye_frame: None,
            ear_pcm: None,
            eye: EyeSense {
                schema_version: 1,
                ..EyeSense::default()
            },
            ear: EarSense {
                schema_version: 1,
                ..EarSense::default()
            },
            range: RangeSense {
                schema_version: 1,
                ..RangeSense::default()
            },
            imu: ImuSense {
                schema_version: 1,
                ..ImuSense::default()
            },
            gps: None,
            kinect: KinectSense {
                schema_version: 1,
                ..KinectSense::default()
            },
            objects: ObjectSense {
                schema_version: 1,
                ..ObjectSense::default()
            },
            face: FaceSense {
                schema_version: 1,
                ..FaceSense::default()
            },
            voice: VoiceSense {
                schema_version: 1,
                ..VoiceSense::default()
            },
            extensions: Vec::new(),
        }
    }
}

impl WorldSnapshot {
    pub fn to_now(&self, t_ms: u64) -> Now {
        let mut now = Now::blank(t_ms, self.body.clone());
        now.eye = self.eye.clone();
        now.eye_frame = self.eye_frame.clone();
        now.ear = self.ear.clone();
        now.face = self.face.clone();
        now.voice = self.voice.clone();
        now.range = self.range.clone();
        now.imu = self.imu.clone();
        now.gps = self.gps.clone();
        now.kinect = self.kinect.clone();
        now.objects = self.objects.clone();
        now.predictions = PredictionSense {
            schema_version: 1,
            ..PredictionSense::default()
        };
        now.surprise = SurpriseSense {
            schema_version: 1,
            ..SurpriseSense::default()
        };
        for extension in &self.extensions {
            now.extensions.insert(
                extension.name.clone(),
                serde_json::json!({
                    "schema_version": extension.schema_version,
                    "values": extension.values,
                }),
            );
        }
        now
    }
}

impl From<&EyeFrame> for FrameKey {
    fn from(frame: &EyeFrame) -> Self {
        Self {
            captured_at_ms: frame.captured_at_ms,
            width: frame.width,
            height: frame.height,
            format: format!("{:?}", frame.format),
            byte_len: frame.bytes.len(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorldUpdate {
    pub body: Option<BodySense>,
    pub eye_frame: Option<EyeFrame>,
    pub ear_pcm: Option<PcmAudioFrame>,
    pub eye: Option<EyeSense>,
    pub ear: Option<EarSense>,
    pub range: Option<RangeSense>,
    pub imu: Option<ImuSense>,
    pub gps: Option<GpsSense>,
    pub kinect: Option<KinectSense>,
    pub objects: Option<ObjectSense>,
    pub face: Option<FaceSense>,
    pub voice: Option<VoiceSense>,
    pub extensions: Option<Vec<ExtensionSense>>,
}

impl WorldUpdate {
    pub fn apply_to(self, snapshot: &mut WorldSnapshot) {
        if let Some(body) = self.body {
            snapshot.body = body;
        }
        if let Some(frame) = self.eye_frame {
            snapshot.eye_frame = Some(frame);
        }
        if let Some(frame) = self.ear_pcm {
            snapshot.ear_pcm = Some(frame);
        }
        if let Some(eye) = self.eye {
            snapshot.eye = eye;
        }
        if let Some(ear) = self.ear {
            snapshot.ear = ear;
        }
        if let Some(range) = self.range {
            snapshot.range = range;
        }
        if let Some(imu) = self.imu {
            snapshot.imu = imu;
        }
        if self.gps.is_some() {
            snapshot.gps = self.gps;
        }
        if let Some(kinect) = self.kinect {
            snapshot.kinect = kinect;
        }
        if let Some(objects) = self.objects {
            snapshot.objects = objects;
        }
        if let Some(face) = self.face {
            snapshot.face = face;
        }
        if let Some(voice) = self.voice {
            snapshot.voice = voice;
        }
        if let Some(extensions) = self.extensions {
            snapshot.extensions = extensions;
        }
    }
}

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
    async fn poll(&mut self) -> Result<SensePacket> {
        if let Some(packet) = self.pending.pop_front() {
            return Ok(packet);
        }
        let depth = read_freenect_depth_m(self.index)?;
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
                parser: Lfcd2Parser::new(yaw_offset_rad),
                last_scan: None,
                last_scan_at: None,
            })
        }
        #[cfg(not(feature = "linux-hardware"))]
        {
            let _ = port;
            let _ = yaw_offset_rad;
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
    yaw_offset_rad: f32,
}

#[cfg(any(feature = "linux-hardware", test))]
impl Lfcd2Parser {
    fn new(yaw_offset_rad: f32) -> Self {
        Self {
            buffer: Vec::new(),
            ranges_m: [0.0; LFCD2_BEAMS_PER_SCAN],
            received_segments: [false; LFCD2_SEGMENTS_PER_SCAN],
            received_count: 0,
            scan_started: false,
            yaw_offset_rad,
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
                    .map(|index| (index as f32).to_radians() + self.yaw_offset_rad)
                    .collect();
                return Some(RangeSense {
                    schema_version: 1,
                    beams,
                    nearest_m,
                    beam_angles_rad,
                    frame: Some("robot_base".to_string()),
                    source: Some("hls_lfcd2".to_string()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Arc;

    struct StaticFaceDetector;

    impl FaceDetector for StaticFaceDetector {
        fn detect_faces(&self, _frame: &EyeFrame) -> Result<Vec<FaceDetection>> {
            Ok(vec![FaceDetection {
                face_id: "face-static".to_string(),
                source_frame_id: None,
                embedding: vec![0.1, 0.2, 0.3],
                model: "test.face.detector".to_string(),
            }])
        }
    }

    struct StaticObjectDetector;

    impl ObjectDetector for StaticObjectDetector {
        fn detect_objects(&self, _frame: &EyeFrame) -> Result<Vec<ObjectDetection>> {
            Ok(vec![ObjectDetection {
                object_id: "object-static".to_string(),
                label: "red cup".to_string(),
                class: ObjectClass::Landmark,
                bearing_rad: 0.25,
                distance_m: Some(1.5),
                confidence: 0.9,
                source: ObjectObservationSource::Kinect,
                source_frame_id: None,
                embedding: vec![0.7, 0.2, 0.1],
                model: "test.object.detector".to_string(),
            }])
        }
    }

    #[test]
    fn frame_processor_vectorizes_detected_faces_into_face_collection() {
        let frame = EyeFrame {
            captured_at_ms: 42,
            width: 1,
            height: 1,
            format: EyeFrameFormat::Rgb8,
            bytes: vec![255, 128, 0],
            source: Some("unit-camera".to_string()),
        };
        let mut processor = FrameProcessor::new().with_face_detector(Arc::new(StaticFaceDetector));

        let processed = processor
            .process_frame(100, &frame)
            .expect("processed frame");

        assert_eq!(processed.face.embeddings, vec![vec![0.1, 0.2, 0.3]]);
        assert_eq!(processed.face.vectors.len(), 1);
        let artifact = &processed.face.vectors[0];
        assert_eq!(artifact.collection, FACE_VECTOR_COLLECTION);
        assert_eq!(artifact.point_id, "face-static");
        assert_eq!(artifact.vector, vec![0.1, 0.2, 0.3]);
        assert_eq!(artifact.model.as_deref(), Some("test.face.detector"));
        assert_eq!(artifact.source_frame_id.as_deref(), Some("eye-42-1x1-3"));
        assert_eq!(artifact.occurred_at_ms, Some(100));
    }

    #[test]
    fn frame_processor_vectorizes_detected_objects_into_object_collection() {
        let frame = EyeFrame {
            captured_at_ms: 42,
            width: 1,
            height: 1,
            format: EyeFrameFormat::Rgb8,
            bytes: vec![255, 128, 0],
            source: Some("unit-camera".to_string()),
        };
        let mut processor =
            FrameProcessor::new().with_object_detector(Arc::new(StaticObjectDetector));

        let processed = processor
            .process_frame(100, &frame)
            .expect("processed frame");

        assert_eq!(processed.objects.observations.len(), 1);
        assert_eq!(processed.objects.observations[0].label, "red cup");
        assert_eq!(processed.objects.embeddings, vec![vec![0.7, 0.2, 0.1]]);
        assert_eq!(processed.objects.vectors.len(), 1);
        let artifact = &processed.objects.vectors[0];
        assert_eq!(artifact.collection, OBJECT_VECTOR_COLLECTION);
        assert_eq!(artifact.point_id, "object-static");
        assert_eq!(artifact.vector, vec![0.7, 0.2, 0.1]);
        assert_eq!(artifact.model.as_deref(), Some("test.object.detector"));
        assert_eq!(artifact.source_frame_id.as_deref(), Some("eye-42-1x1-3"));
        assert_eq!(artifact.occurred_at_ms, Some(100));
    }

    #[test]
    fn mutes_repeated_alsa_poll_descriptor_input_stream_error() {
        assert!(is_muted_cpal_input_stream_error(
            "A backend-specific error has occurred: ALSA function 'snd_pcm_poll_descriptors' failed with error 'Unknown errno (-32)'"
        ));
    }

    #[test]
    fn keeps_other_cpal_input_stream_errors_visible() {
        assert!(!is_muted_cpal_input_stream_error(
            "A backend-specific error has occurred: ALSA function 'snd_pcm_readi' failed with error 'Input/output error'"
        ));
        assert!(!is_muted_cpal_input_stream_error(
            "A backend-specific error has occurred: ALSA function 'snd_pcm_poll_descriptors' failed with error 'Input/output error'"
        ));
    }

    #[tokio::test]
    async fn kinect_replay_emits_kinect_then_eye_packet() {
        let root = std::env::temp_dir().join(format!("pete-kinect-replay-{}", unix_time_ms()));
        std::fs::create_dir_all(root.join("rgb")).unwrap();
        std::fs::create_dir_all(root.join("depth")).unwrap();
        std::fs::write(root.join("rgb/frame.raw"), [0u8, 128, 255]).unwrap();
        std::fs::write(root.join("depth/frame.json"), "[1.0,2.0]").unwrap();
        let mut manifest = File::create(root.join("timestamps.jsonl")).unwrap();
        writeln!(
            manifest,
            "{}",
            serde_json::json!({
                "t_ms": 1,
                "rgb_path": "rgb/frame.raw",
                "depth_path": "depth/frame.json"
            })
        )
        .unwrap();

        let mut provider = KinectReplayProvider::new(&root).unwrap();
        let first = provider.poll().await.unwrap();
        let second = provider.poll().await.unwrap();

        assert!(
            matches!(first, SensePacket::Kinect(KinectSense { depth_m, .. }) if depth_m == vec![1.0, 2.0])
        );
        assert!(matches!(second, SensePacket::Eye(EyeSense { frames, .. }) if frames.len() == 1));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn now_builder_maps_packets_and_marks_stale_sensor_ages() {
        let mut builder = NowBuilder::new();
        let first = builder
            .build(
                100,
                BodySense::default(),
                vec![
                    SensePacket::Ear(EarSense {
                        transcript: Some("hello".to_string()),
                        ..EarSense::default()
                    }),
                    SensePacket::Range(RangeSense {
                        beams: vec![0.4],
                        nearest_m: Some(0.4),
                        ..RangeSense::default()
                    }),
                ],
            )
            .unwrap();

        assert_eq!(first.ear.transcript.as_deref(), Some("hello"));
        assert_eq!(first.range.nearest_m, Some(0.4));

        let second = builder
            .build(250, BodySense::default(), Vec::new())
            .unwrap();
        assert_eq!(second.ear.transcript.as_deref(), Some("hello"));
        assert_eq!(second.range.nearest_m, Some(0.4));
        assert_eq!(
            second
                .extensions
                .get("sensor_status")
                .and_then(|status| status.get("age_ms"))
                .and_then(|age| age.get("ear"))
                .and_then(|age| age.as_u64()),
            Some(150)
        );
    }

    #[test]
    fn now_builder_derives_range_beams_from_kinect_depth_when_range_absent() {
        let mut builder = NowBuilder::new();
        let now = builder
            .build(
                100,
                BodySense::default(),
                vec![SensePacket::Kinect(KinectSense {
                    depth_m: vec![1.0, 2.0, 0.0, 3.0],
                    depth_width: 4,
                    depth_height: 1,
                    min_depth_m: 0.2,
                    max_depth_m: 4.0,
                    ..KinectSense::default()
                })],
            )
            .unwrap();

        assert_eq!(now.range.beams.len(), 4);
        for (actual, expected) in now.range.beams.iter().zip([1.0, 2.0, 4.0, 3.0]) {
            assert!((actual - expected).abs() < 0.0001);
        }
        assert!(now
            .range
            .nearest_m
            .is_some_and(|nearest| (nearest - 1.0).abs() < 0.0001));
        assert_eq!(now.range.beam_angles_rad.len(), 4);
        assert_eq!(now.range.frame.as_deref(), Some("robot_base"));
        assert_eq!(now.range.source.as_deref(), Some("kinect_depth_image"));
        assert_eq!(
            now.extensions
                .get("sensor_status")
                .and_then(|status| status.get("last_update_ms"))
                .and_then(|updates| updates.get("range"))
                .and_then(|value| value.as_u64()),
            Some(100)
        );
    }

    #[test]
    fn frame_processor_injects_calibrated_range_angles_from_kinect_depth() {
        let mut processor =
            FrameProcessor::new().with_kinect_range_projection(DepthRangeProjectionConfig {
                compact_depth_beam_count: 3,
                compact_depth_fov_rad: std::f32::consts::FRAC_PI_2,
                camera_yaw_rad: -std::f32::consts::FRAC_PI_2,
                min_depth_m: 0.1,
                max_depth_m: 3.0,
                ..DepthRangeProjectionConfig::default()
            });
        let mut packets = vec![SensePacket::Kinect(KinectSense {
            depth_m: vec![1.0, 1.0, 1.0],
            ..KinectSense::default()
        })];

        processor.process_packets(100, &mut packets);

        let range = packets
            .iter()
            .find_map(|packet| match packet {
                SensePacket::Range(range) => Some(range),
                _ => None,
            })
            .expect("calibrated range packet");
        assert_eq!(range.frame.as_deref(), Some("robot_base"));
        assert_eq!(range.source.as_deref(), Some("kinect_compact_depth"));
        assert_eq!(range.beam_angles_rad.len(), 3);
        assert!((range.beam_angles_rad[1] + std::f32::consts::FRAC_PI_2).abs() < 0.001);
    }

    #[test]
    fn now_builder_keeps_explicit_range_over_kinect_fallback() {
        let mut builder = NowBuilder::new();
        let now = builder
            .build(
                100,
                BodySense::default(),
                vec![
                    SensePacket::Range(RangeSense {
                        beams: vec![0.7],
                        nearest_m: Some(0.7),
                        ..RangeSense::default()
                    }),
                    SensePacket::Kinect(KinectSense {
                        depth_m: vec![1.0, 2.0, 3.0, 4.0],
                        depth_width: 4,
                        depth_height: 1,
                        ..KinectSense::default()
                    }),
                ],
            )
            .unwrap();

        assert_eq!(now.range.beams, vec![0.7]);
        assert_eq!(now.range.nearest_m, Some(0.7));
    }

    #[test]
    fn lfcd2_parser_builds_clockwise_native_segments_into_a_full_scan() {
        let mut parser = Lfcd2Parser::new(0.25);
        let mut stream = vec![0x00, 0x11, 0xfa, 0x01];
        for segment in 0..LFCD2_SEGMENTS_PER_SCAN {
            stream.extend_from_slice(&lfcd2_test_segment(segment));
        }

        assert!(parser.push(&stream[..777]).is_none());
        let scan = parser.push(&stream[777..]).expect("complete LFCD2 scan");

        assert_eq!(scan.schema_version, 1);
        assert_eq!(scan.beams.len(), LFCD2_BEAMS_PER_SCAN);
        assert_eq!(scan.beam_angles_rad.len(), LFCD2_BEAMS_PER_SCAN);
        assert_eq!(scan.frame.as_deref(), Some("robot_base"));
        assert_eq!(scan.source.as_deref(), Some("hls_lfcd2"));
        assert!((scan.nearest_m.expect("nearest range") - 0.12).abs() < 0.0001);
        assert!((scan.beams[359] - 0.12).abs() < 0.0001);
        assert!((scan.beams[0] - 0.479).abs() < 0.0001);
        assert!((scan.beam_angles_rad[0] - 0.25).abs() < 0.0001);
        assert!((scan.beam_angles_rad[359] - (359.0_f32.to_radians() + 0.25)).abs() < 0.0001);
    }

    #[test]
    fn lfcd2_parser_rejects_out_of_range_measurements() {
        let mut parser = Lfcd2Parser::new(0.0);
        let mut stream = Vec::new();
        for segment in 0..LFCD2_SEGMENTS_PER_SCAN {
            let mut packet = lfcd2_test_segment(segment);
            if segment == 0 {
                packet[6..8].copy_from_slice(&119_u16.to_le_bytes());
                packet[12..14].copy_from_slice(&3501_u16.to_le_bytes());
            }
            stream.extend_from_slice(&packet);
        }

        let scan = parser.push(&stream).expect("complete LFCD2 scan");
        assert_eq!(scan.beams[359], 0.0);
        assert_eq!(scan.beams[358], 0.0);
        assert!((scan.nearest_m.expect("nearest valid range") - 0.122).abs() < 0.0001);
    }

    fn lfcd2_test_segment(segment: usize) -> [u8; LFCD2_SEGMENT_BYTES] {
        let mut packet = [0u8; LFCD2_SEGMENT_BYTES];
        packet[0] = 0xfa;
        packet[1] = 0xa0 + segment as u8;
        packet[2..4].copy_from_slice(&3000_u16.to_le_bytes());
        for beam_in_segment in 0..LFCD2_BEAMS_PER_SEGMENT {
            let raw_index = segment * LFCD2_BEAMS_PER_SEGMENT + beam_in_segment;
            let offset = 4 + beam_in_segment * 6;
            packet[offset..offset + 2]
                .copy_from_slice(&(1000_u16 + raw_index as u16).to_le_bytes());
            packet[offset + 2..offset + 4]
                .copy_from_slice(&(120_u16 + raw_index as u16).to_le_bytes());
        }
        packet
    }

    #[test]
    fn parses_mpu6050_device_specs() {
        assert_eq!(
            parse_mpu6050_device_spec("/dev/i2c-1").unwrap(),
            Mpu6050DeviceSpec {
                path: "/dev/i2c-1".to_string(),
                address: 0x68,
            }
        );
        assert_eq!(
            parse_mpu6050_device_spec("/dev/i2c-1@0x69").unwrap(),
            Mpu6050DeviceSpec {
                path: "/dev/i2c-1".to_string(),
                address: 0x69,
            }
        );
        assert_eq!(
            parse_mpu6050_device_spec("/dev/i2c-2:105").unwrap(),
            Mpu6050DeviceSpec {
                path: "/dev/i2c-2".to_string(),
                address: 0x69,
            }
        );
    }

    #[test]
    fn converts_mpu6050_raw_samples_to_imu_sense() {
        let imu = mpu6050_samples_to_imu(
            [
                0x00, 0x00, // accel x = 0 g
                0x00, 0x00, // accel y = 0 g
                0x40, 0x00, // accel z = 1 g
                0x00, 0x00, // temperature, ignored
                0x00, 0x83, // gyro x = 1 deg/s => rad/s
                0xff, 0x7d, // gyro y = -1 deg/s => rad/s
                0x01, 0x06, // gyro z = 2 deg/s => rad/s
            ],
            1234,
        );

        assert_eq!(imu.schema_version, 1);
        assert_eq!(imu.captured_at_ms, 1234);
        assert_eq!(imu.acceleration, vec![0.0, 0.0, 1.0]);
        assert!((imu.angular_velocity[0] - 1.0_f32.to_radians()).abs() < 0.0001);
        assert!((imu.angular_velocity[1] - (-1.0_f32).to_radians()).abs() < 0.0001);
        assert!((imu.angular_velocity[2] - 2.0_f32.to_radians()).abs() < 0.0001);
        assert_eq!(imu.orientation, vec![0.0, -0.0]);
    }

    #[test]
    fn mpu6050_orientation_is_zeroed_to_first_flat_sample() {
        let mut zero = None;
        let baseline_gravity =
            test_gravity_for_roll_pitch(120.0_f32.to_radians(), 62.0_f32.to_radians());
        let mut first = ImuSense {
            orientation: vec![120.0_f32.to_radians(), 62.0_f32.to_radians()],
            acceleration: vec![baseline_gravity.x, baseline_gravity.y, baseline_gravity.z],
            ..ImuSense::default()
        };
        zero_mpu6050_orientation_to_flat(&mut first, &mut zero);
        assert!(first.orientation[0].abs() < 0.0001);
        assert!(first.orientation[1].abs() < 0.0001);

        let expected_roll = 5.0_f32.to_radians();
        let expected_pitch = (-10.0_f32).to_radians();
        let expected_leveled_gravity = test_gravity_for_roll_pitch(expected_roll, expected_pitch);
        let next_gravity = rotate_vec3_between_unit_vectors(
            expected_leveled_gravity,
            test_z_axis(),
            baseline_gravity,
        );
        let mut next = ImuSense {
            orientation: vec![0.0, 0.0],
            acceleration: vec![next_gravity.x, next_gravity.y, next_gravity.z],
            ..ImuSense::default()
        };
        zero_mpu6050_orientation_to_flat(&mut next, &mut zero);
        assert!((next.orientation[0] - expected_roll).abs() < 0.0001);
        assert!((next.orientation[1] - expected_pitch).abs() < 0.0001);
    }

    fn test_gravity_for_roll_pitch(roll_rad: f32, pitch_rad: f32) -> Vec3Unit {
        Vec3Unit {
            x: -pitch_rad.sin(),
            y: roll_rad.sin() * pitch_rad.cos(),
            z: roll_rad.cos() * pitch_rad.cos(),
        }
    }

    fn test_z_axis() -> Vec3Unit {
        Vec3Unit {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        }
    }

    #[test]
    fn kinect_rgb_adjustment_brightens_dark_frames_without_changing_length() {
        let dark = vec![12_u8, 10, 8, 20, 18, 16];
        let adjusted = adjust_kinect_rgb(
            &dark,
            KinectRgbAdjustment {
                enabled: true,
                target_luma: 0.35,
                auto_gain_max: 4.0,
                gamma: 0.75,
                ..KinectRgbAdjustment::default()
            },
        );

        assert_eq!(adjusted.len(), dark.len());
        assert!(mean_rgb_luma(&adjusted) > mean_rgb_luma(&dark));
        assert!(adjusted.iter().zip(dark.iter()).any(|(new, old)| new > old));
    }

    #[test]
    fn disabled_kinect_rgb_adjustment_preserves_bytes() {
        let rgb = vec![8_u8, 16, 24, 128, 96, 64];
        let adjusted = adjust_kinect_rgb(
            &rgb,
            KinectRgbAdjustment {
                enabled: false,
                ..KinectRgbAdjustment::default()
            },
        );

        assert_eq!(adjusted, rgb);
    }

    #[test]
    fn asr_tool_emits_final_ear_sense_from_command_transcript() {
        let mut adapter = AsrTool::new(AsrToolConfig {
            command: Some("printf hello".to_string()),
            min_voice_rms: 0.01,
            min_chunk_ms: 100,
            max_chunk_ms: 8_000,
            silence_finalize_ms: 0,
        });
        let frame = PcmAudioFrame {
            captured_at_ms: 1_000,
            sample_rate_hz: 1_000,
            channels: 1,
            samples: vec![10_000; 250],
        };

        let ear = adapter.observe_frame(&frame).expect("final ASR sense");

        assert_eq!(ear.transcript.as_deref(), Some("hello"));
        assert_eq!(ear.asr.transcript.as_deref(), Some("hello"));
        assert!(ear.asr.is_final);
        assert_eq!(ear.asr.start_ms, Some(1_000));
        assert_eq!(ear.asr.duration_ms, Some(250));
        assert_eq!(ear.asr.sample_rate_hz, Some(1_000));
        assert_eq!(ear.asr.word_count, Some(1));
        assert_eq!(ear.asr.committed_transcript.as_deref(), Some("hello"));
        assert!(ear.asr.possible_transcript.is_none());
        assert_eq!(ear.transcript_vectors.len(), 1);
        let transcript_vector = &ear.transcript_vectors[0];
        assert_eq!(transcript_vector.collection, TRANSCRIPT_VECTOR_COLLECTION);
        assert_eq!(
            transcript_vector.model.as_deref(),
            Some(ASR_TRANSCRIPT_VECTOR_MODEL)
        );
        assert_eq!(transcript_vector.vector.len(), ASR_TRANSCRIPT_VECTOR_DIM);
        assert_eq!(transcript_vector.occurred_at_ms, Some(1_250));
        assert_eq!(ear.asr.candidate_id, Some(1));
        assert_eq!(ear.asr.stable_text.as_deref(), Some("hello"));
        assert!(ear
            .asr
            .candidate_events
            .iter()
            .any(|event| matches!(event, TranscriptCandidateEvent::CandidateFinalized { .. })));
    }

    #[test]
    fn asr_tool_without_command_does_not_fabricate_words() {
        let mut adapter = AsrTool::new(AsrToolConfig {
            command: None,
            min_voice_rms: 0.01,
            min_chunk_ms: 100,
            max_chunk_ms: 8_000,
            silence_finalize_ms: 0,
        });
        let frame = PcmAudioFrame {
            captured_at_ms: 1_000,
            sample_rate_hz: 1_000,
            channels: 1,
            samples: vec![10_000; 250],
        };

        assert!(adapter.observe_frame(&frame).is_none());
    }
}
