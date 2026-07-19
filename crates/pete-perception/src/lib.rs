use pete_core::TimeMs;
use pete_now::KinectSense;
use pete_sensors::{EyeFrameFormat, WorldSnapshot};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const DEFAULT_MAX_SPARSE_POINTS: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameId(pub String);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointXyz {
    pub x_m: f32,
    pub y_m: f32,
    pub z_m: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DepthSample {
    pub index: usize,
    pub u: u32,
    pub v: u32,
    pub depth_m: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RgbSample {
    pub u: u32,
    pub v: u32,
    pub rgb: [u8; 3],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoxelRef {
    pub key: [i32; 3],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfaceRef {
    pub id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusterRef {
    pub id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemanticRef {
    pub label: String,
    pub source: String,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointSample {
    pub depth: DepthSample,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgb: Option<RgbSample>,
    pub camera_point: PointXyz,
    pub robot_point: PointXyz,
    pub world_point: PointXyz,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voxel: Option<VoxelRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<SurfaceRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster: Option<ClusterRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic: Option<SemanticRef>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerceptionFrame {
    pub frame_id: FrameId,
    pub t_ms: TimeMs,
    #[serde(default)]
    pub points: Vec<PointSample>,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub metadata: Value,
}

impl PerceptionFrame {
    pub fn from_world_snapshot(snapshot: &WorldSnapshot, t_ms: TimeMs) -> Option<Self> {
        Self::from_world_snapshot_sparse(snapshot, t_ms, DEFAULT_MAX_SPARSE_POINTS)
    }

    pub fn from_world_snapshot_sparse(
        snapshot: &WorldSnapshot,
        t_ms: TimeMs,
        max_points: usize,
    ) -> Option<Self> {
        let projection = DepthProjection::from_kinect(&snapshot.kinect)?;
        let rgb = RgbImageView::from_snapshot(snapshot);
        let min_depth_m = positive_or(snapshot.kinect.min_depth_m, 0.1);
        let max_depth_m = positive_or(snapshot.kinect.max_depth_m, 8.0);
        let stride = snapshot
            .kinect
            .depth_m
            .len()
            .div_ceil(max_points.max(1))
            .max(1);
        let pose = snapshot.body.odometry;
        let yaw = pose.heading_rad;
        let (sin_yaw, cos_yaw) = yaw.sin_cos();
        let mut skipped_depth_count = 0usize;
        let mut clipped_depth_count = 0usize;
        let mut points = Vec::new();

        for (index, depth_m) in snapshot.kinect.depth_m.iter().enumerate().step_by(stride) {
            let depth_m = *depth_m;
            if !depth_m.is_finite() || depth_m <= 0.0 {
                skipped_depth_count = skipped_depth_count.saturating_add(1);
                continue;
            }
            if depth_m < min_depth_m || depth_m > max_depth_m {
                clipped_depth_count = clipped_depth_count.saturating_add(1);
                continue;
            }

            let u = index % projection.width;
            let v = index / projection.width;
            let camera_point = PointXyz {
                x_m: (u as f32 - projection.cx) * depth_m / projection.fx.max(f32::EPSILON),
                y_m: (v as f32 - projection.cy) * depth_m / projection.fy.max(f32::EPSILON),
                z_m: depth_m,
            };
            let robot_point = camera_point_to_robot(camera_point);
            let world_point = PointXyz {
                x_m: pose.x_m + robot_point.x_m * cos_yaw - robot_point.y_m * sin_yaw,
                y_m: pose.y_m + robot_point.x_m * sin_yaw + robot_point.y_m * cos_yaw,
                z_m: robot_point.z_m,
            };

            points.push(PointSample {
                depth: DepthSample {
                    index,
                    u: u as u32,
                    v: v as u32,
                    depth_m,
                },
                rgb: rgb
                    .as_ref()
                    .and_then(|rgb| rgb.sample_scaled(u, v, projection.width, projection.height)),
                camera_point,
                robot_point,
                world_point,
                voxel: None,
                surface: None,
                cluster: None,
                semantic: None,
            });
        }

        if points.is_empty() {
            return None;
        }

        Some(Self {
            frame_id: FrameId(frame_id(snapshot, t_ms)),
            t_ms,
            points,
            metadata: serde_json::json!({
                "schema": "pete.perception.v1",
                "depth_width": projection.width,
                "depth_height": projection.height,
                "depth_fx": projection.fx,
                "depth_fy": projection.fy,
                "depth_cx": projection.cx,
                "depth_cy": projection.cy,
                "depth_projection": projection.source,
                "sample_stride": stride,
                "max_sparse_points": max_points.max(1),
                "skipped_depth_count": skipped_depth_count,
                "clipped_depth_count": clipped_depth_count,
                "rgb_mapped": rgb.is_some(),
                "robot_transform": "camera_z_forward_x_left_y_down_to_robot_x_forward_y_left_z_up_no_extrinsics",
                "world_transform": "odometry_xy_yaw"
            }),
        })
    }
}

fn frame_id(snapshot: &WorldSnapshot, t_ms: TimeMs) -> String {
    if snapshot.kinect.captured_at_ms > 0 {
        format!("kinect-depth-{}", snapshot.kinect.captured_at_ms)
    } else {
        format!("perception-frame-{t_ms}")
    }
}

fn positive_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        fallback
    }
}

fn camera_point_to_robot(point: PointXyz) -> PointXyz {
    PointXyz {
        x_m: point.z_m,
        y_m: -point.x_m,
        z_m: -point.y_m,
    }
}

#[derive(Clone, Copy, Debug)]
struct DepthProjection {
    width: usize,
    height: usize,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    source: &'static str,
}

impl DepthProjection {
    fn from_kinect(kinect: &KinectSense) -> Option<Self> {
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
            cx: if kinect.depth_cx > 0.0 {
                kinect.depth_cx
            } else {
                (width as f32 - 1.0) * 0.5
            },
            cy: if kinect.depth_cy > 0.0 {
                kinect.depth_cy
            } else {
                (height as f32 - 1.0) * 0.5
            },
            source: "kinect_intrinsics",
        })
    }
}

struct RgbImageView<'a> {
    width: usize,
    height: usize,
    format: EyeFrameFormat,
    bytes: &'a [u8],
}

impl<'a> RgbImageView<'a> {
    fn from_snapshot(snapshot: &'a WorldSnapshot) -> Option<Self> {
        let frame = snapshot.eye_frame.as_ref()?;
        let width = usize::try_from(frame.width).ok()?;
        let height = usize::try_from(frame.height).ok()?;
        let expected = match &frame.format {
            EyeFrameFormat::Rgb8 | EyeFrameFormat::Bgr8 => {
                width.checked_mul(height)?.checked_mul(3)?
            }
            EyeFrameFormat::Gray8 => width.checked_mul(height)?,
            EyeFrameFormat::Yuyv422
            | EyeFrameFormat::Uyvy422
            | EyeFrameFormat::BayerGrbg8
            | EyeFrameFormat::BayerRggb8
            | EyeFrameFormat::BayerBggr8
            | EyeFrameFormat::BayerGbrg8
            | EyeFrameFormat::Mjpeg
            | EyeFrameFormat::Unknown(_) => return None,
        };
        (width > 0 && height > 0 && frame.bytes.len() == expected).then_some(Self {
            width,
            height,
            format: frame.format.clone(),
            bytes: &frame.bytes,
        })
    }

    fn sample_scaled(
        &self,
        depth_u: usize,
        depth_v: usize,
        depth_width: usize,
        depth_height: usize,
    ) -> Option<RgbSample> {
        if depth_width == 0 || depth_height == 0 {
            return None;
        }
        let u = depth_u.saturating_mul(self.width) / depth_width;
        let v = depth_v.saturating_mul(self.height) / depth_height;
        self.sample(u.min(self.width - 1), v.min(self.height - 1))
    }

    fn sample(&self, u: usize, v: usize) -> Option<RgbSample> {
        let pixel = v.checked_mul(self.width)?.checked_add(u)?;
        let rgb = match &self.format {
            EyeFrameFormat::Rgb8 => {
                let offset = pixel.checked_mul(3)?;
                [
                    *self.bytes.get(offset)?,
                    *self.bytes.get(offset + 1)?,
                    *self.bytes.get(offset + 2)?,
                ]
            }
            EyeFrameFormat::Bgr8 => {
                let offset = pixel.checked_mul(3)?;
                [
                    *self.bytes.get(offset + 2)?,
                    *self.bytes.get(offset + 1)?,
                    *self.bytes.get(offset)?,
                ]
            }
            EyeFrameFormat::Gray8 => {
                let value = *self.bytes.get(pixel)?;
                [value, value, value]
            }
            EyeFrameFormat::Yuyv422
            | EyeFrameFormat::Uyvy422
            | EyeFrameFormat::BayerGrbg8
            | EyeFrameFormat::BayerRggb8
            | EyeFrameFormat::BayerBggr8
            | EyeFrameFormat::BayerGbrg8
            | EyeFrameFormat::Mjpeg
            | EyeFrameFormat::Unknown(_) => return None,
        };
        Some(RgbSample {
            u: u as u32,
            v: v as u32,
            rgb,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pete_now::KinectSense;
    use pete_sensors::EyeFrame;

    #[test]
    fn sparse_frame_serializes_depth_geometry_and_empty_optional_fields() {
        let mut snapshot = WorldSnapshot::default();
        snapshot.kinect = KinectSense {
            captured_at_ms: 123,
            depth_m: vec![1.0, 2.0, 0.0, 3.0],
            depth_width: 2,
            depth_height: 2,
            depth_fx: 2.0,
            depth_fy: 2.0,
            depth_cx: 0.0,
            depth_cy: 0.0,
            min_depth_m: 0.1,
            max_depth_m: 8.0,
            ..KinectSense::default()
        };

        let frame = PerceptionFrame::from_world_snapshot_sparse(&snapshot, 500, 16).unwrap();
        let encoded = serde_json::to_value(&frame).unwrap();
        let decoded: PerceptionFrame = serde_json::from_value(encoded.clone()).unwrap();

        assert_eq!(decoded.frame_id, FrameId("kinect-depth-123".to_string()));
        assert_eq!(decoded.points.len(), 3);
        assert_eq!(decoded.points[1].depth.index, 1);
        assert_eq!(decoded.points[1].depth.u, 1);
        assert_eq!(decoded.points[1].depth.v, 0);
        assert!(decoded.points[1].rgb.is_none());
        assert!(decoded.points[1].semantic.is_none());
        assert_eq!(encoded["points"][0]["depth"]["index"], 0);
    }

    #[test]
    fn sparse_frame_maps_scaled_rgb_when_available() {
        let mut snapshot = WorldSnapshot::default();
        snapshot.kinect = KinectSense {
            depth_m: vec![1.0, 1.0, 1.0, 1.0],
            depth_width: 2,
            depth_height: 2,
            depth_fx: 1.0,
            depth_fy: 1.0,
            min_depth_m: 0.1,
            max_depth_m: 8.0,
            ..KinectSense::default()
        };
        snapshot.eye_frame = Some(EyeFrame {
            width: 2,
            height: 2,
            format: EyeFrameFormat::Rgb8,
            bytes: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            captured_at_ms: 100,
            source: Some("test".to_string()),
        });

        let frame = PerceptionFrame::from_world_snapshot_sparse(&snapshot, 100, 16).unwrap();

        assert_eq!(frame.points[0].rgb.as_ref().unwrap().rgb, [1, 2, 3]);
        assert_eq!(frame.points[3].rgb.as_ref().unwrap().rgb, [10, 11, 12]);
    }
}
