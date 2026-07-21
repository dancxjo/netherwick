#[cfg(any(
    feature = "face",
    feature = "linux-hardware",
    feature = "kinect-freenect",
    test
))]
use anyhow::Context;
use anyhow::Result;
use async_trait::async_trait;
use pete_actions::{ActionPrimitive, LlmActionProposal};
use pete_body::BodySense;
use pete_core::Pose2;
use pete_now::{
    AsrSense, EarSense, ExtensionSense, EyeSense, FaceSense, GpsSense, ImuSense,
    KinectFusionAlignment, KinectSense, ObjectClass, ObjectObservation, ObjectObservationSource,
    ObjectSense, RangeExtrinsics, RangeSense, TranscriptCandidateEvent, TranscriptCandidateTracker,
    TranscriptStabilityState, VectorArtifact, VoiceSense, FACE_VECTOR_COLLECTION,
    IMAGE_DESCRIPTION_VECTOR_COLLECTION, IMAGE_VECTOR_COLLECTION, OBJECT_VECTOR_COLLECTION,
    SCENE_VECTOR_COLLECTION, TRANSCRIPT_VECTOR_COLLECTION,
};
use pete_now::{Now, PredictionSense, SurpriseSense};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
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

const FUSION_HISTORY_LIMIT: usize = 256;
const MAX_FUSION_SAMPLE_SKEW_MS: u64 = 200;

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

// Sensor domains share one namespace to preserve the crate API.
include!("sensors/core.rs");
include!("sensors/imu_arbitration.rs");
include!("sensors/vision.rs");
include!("sensors/audio.rs");
include!("sensors/world.rs");
include!("sensors/linux.rs");
include!("sensors/kinect.rs");
include!("sensors/providers.rs");

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
