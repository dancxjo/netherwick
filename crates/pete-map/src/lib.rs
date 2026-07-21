use std::collections::BTreeMap;

use pete_core::{Pose2, TimeMs};
use pete_now::{EyeFrame, EyeFrameFormat, ImuSense, KinectSense, Now, RangeExtrinsics, RangeSense};
use pete_sensors::WorldSnapshot;
use serde::{Deserialize, Serialize};

// Map domains share one namespace to preserve the crate API.
include!("map/types.rs");
include!("map/point_cloud.rs");
include!("map/pose_graph.rs");
include!("map/local.rs");
include!("map/observations.rs");
include!("map/projection.rs");
include!("map/world.rs");
include!("map/helpers.rs");

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
