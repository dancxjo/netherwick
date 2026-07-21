use serde::{Deserialize, Serialize};

use crate::{CalibrationTrustState, KinectSense};
use pete_core::Pose2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CameraIntrinsics {
    pub width: u32,
    pub height: u32,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    /// Brown-Conrady `[k1, k2, p1, p2, k3]` coefficients.
    #[serde(default)]
    pub distortion: [f32; 5],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RigidTransform3 {
    /// Translation of the source origin expressed in the destination frame.
    pub translation_m: [f32; 3],
    /// Extrinsic roll, pitch, yaw applied in X, Y, Z order.
    pub rotation_rpy_rad: [f32; 3],
}

impl RigidTransform3 {
    pub fn rotate_vector(self, point: [f32; 3]) -> [f32; 3] {
        let [roll, pitch, yaw] = self.rotation_rpy_rad;
        let (roll_sin, roll_cos) = roll.sin_cos();
        let (pitch_sin, pitch_cos) = pitch.sin_cos();
        let (yaw_sin, yaw_cos) = yaw.sin_cos();
        let [x, y, z] = point;
        let rolled = [x, y * roll_cos - z * roll_sin, y * roll_sin + z * roll_cos];
        let pitched = [
            rolled[0] * pitch_cos + rolled[2] * pitch_sin,
            rolled[1],
            -rolled[0] * pitch_sin + rolled[2] * pitch_cos,
        ];
        [
            pitched[0] * yaw_cos - pitched[1] * yaw_sin,
            pitched[0] * yaw_sin + pitched[1] * yaw_cos,
            pitched[2],
        ]
    }

    pub fn transform_point(self, point: [f32; 3]) -> [f32; 3] {
        let rotated = self.rotate_vector(point);
        [
            rotated[0] + self.translation_m[0],
            rotated[1] + self.translation_m[1],
            rotated[2] + self.translation_m[2],
        ]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DepthGeometryCalibration {
    /// True only for coefficients measured for this physical camera/mount.
    pub calibrated: bool,
    pub depth: CameraIntrinsics,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rgb: Option<CameraIntrinsics>,
    #[serde(default)]
    pub depth_scale: f32,
    #[serde(default)]
    pub depth_bias_m: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_to_rgb: Option<RigidTransform3>,
    /// Optical depth-camera coordinates to robot base coordinates.
    pub depth_to_base: RigidTransform3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation: Option<DepthCalibrationValidation>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DepthCalibrationValidation {
    pub distance_sample_count: u32,
    pub min_test_distance_m: f32,
    pub max_test_distance_m: f32,
    pub max_plane_distance_error_m: f32,
    pub rgb_depth_boundary_error_px: f32,
}

impl DepthGeometryCalibration {
    pub fn physical_validation_ready(self) -> bool {
        self.calibrated
            && self.validation.is_some_and(|validation| {
                validation.distance_sample_count >= 4
                    && validation.min_test_distance_m <= 0.5
                    && validation.max_test_distance_m >= 3.0
                    && validation.max_plane_distance_error_m <= 0.02
                    && validation.rgb_depth_boundary_error_px <= 3.0
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthGeometry {
    pub calibration: DepthGeometryCalibration,
}

impl DepthGeometry {
    pub fn from_kinect(kinect: &KinectSense) -> Option<Self> {
        let mut calibration = kinect
            .geometry_calibration
            .unwrap_or(DepthGeometryCalibration {
                calibrated: false,
                depth: CameraIntrinsics {
                    width: kinect.depth_width,
                    height: kinect.depth_height,
                    fx: kinect.depth_fx,
                    fy: kinect.depth_fy,
                    cx: kinect.depth_cx,
                    cy: kinect.depth_cy,
                    distortion: kinect.depth_distortion,
                },
                depth_scale: 1.0,
                ..DepthGeometryCalibration::default()
            });
        if let Some(estimate) = kinect.live_geometry_calibration.as_ref() {
            if estimate.trust_state != CalibrationTrustState::Invalidated {
                // Estimating/degraded transforms remain usable for conservative
                // perception. Navigation trust is gated separately.
                calibration.depth_to_base = estimate.transform;
            }
        }
        let depth = calibration.depth;
        (depth.width > 0
            && depth.height > 0
            && depth.fx.is_finite()
            && depth.fx > 0.0
            && depth.fy.is_finite()
            && depth.fy > 0.0)
            .then_some(Self { calibration })
    }

    pub fn live_transform_trusted(kinect: &KinectSense) -> bool {
        kinect
            .live_geometry_calibration
            .as_ref()
            .map_or(true, |estimate| {
                estimate.trust_state == CalibrationTrustState::Trusted
            })
    }

    pub fn depth_pixel_to_camera(self, u: f32, v: f32, raw_depth_m: f32) -> Option<[f32; 3]> {
        let intrinsics = self.calibration.depth;
        let scale = if self.calibration.depth_scale > 0.0 {
            self.calibration.depth_scale
        } else {
            1.0
        };
        let z = raw_depth_m * scale + self.calibration.depth_bias_m;
        if !z.is_finite() || z <= 0.0 {
            return None;
        }
        let distorted = [
            (u - intrinsics.cx) / intrinsics.fx,
            (v - intrinsics.cy) / intrinsics.fy,
        ];
        let [x, y] = undistort_normalized(distorted, intrinsics.distortion);
        Some([x * z, y * z, z])
    }

    pub fn depth_point_to_base(self, point: [f32; 3]) -> [f32; 3] {
        self.calibration.depth_to_base.transform_point(point)
    }

    pub fn depth_point_to_rgb_pixel(self, point: [f32; 3]) -> Option<[f32; 2]> {
        let intrinsics = self.calibration.rgb?;
        let point = self.calibration.depth_to_rgb?.transform_point(point);
        if !point[2].is_finite() || point[2] <= 0.0 {
            return None;
        }
        let normalized = [point[0] / point[2], point[1] / point[2]];
        let [x, y] = distort_normalized(normalized, intrinsics.distortion);
        let pixel = [
            intrinsics.fx * x + intrinsics.cx,
            intrinsics.fy * y + intrinsics.cy,
        ];
        (pixel[0] >= 0.0
            && pixel[1] >= 0.0
            && pixel[0] < intrinsics.width as f32
            && pixel[1] < intrinsics.height as f32)
            .then_some(pixel)
    }

    pub fn base_point_to_world(
        point: [f32; 3],
        pose: Pose2,
        roll_rad: Option<f32>,
        pitch_rad: Option<f32>,
    ) -> [f32; 3] {
        let [mut x, mut y, mut z] = point;
        if let Some(roll) = roll_rad {
            let (sin, cos) = roll.sin_cos();
            [y, z] = [y * cos - z * sin, y * sin + z * cos];
        }
        if let Some(pitch) = pitch_rad {
            let (sin, cos) = pitch.sin_cos();
            [x, z] = [x * cos + z * sin, -x * sin + z * cos];
        }
        let (sin, cos) = pose.heading_rad.sin_cos();
        [
            pose.x_m + x * cos - y * sin,
            pose.y_m + x * sin + y * cos,
            z,
        ]
    }
}

fn distort_normalized(point: [f32; 2], coefficients: [f32; 5]) -> [f32; 2] {
    let [x, y] = point;
    let [k1, k2, p1, p2, k3] = coefficients;
    let r2 = x * x + y * y;
    let radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    [
        x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
        y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
    ]
}

fn undistort_normalized(distorted: [f32; 2], coefficients: [f32; 5]) -> [f32; 2] {
    let mut estimate = distorted;
    for _ in 0..6 {
        let projected = distort_normalized(estimate, coefficients);
        estimate[0] += distorted[0] - projected[0];
        estimate[1] += distorted[1] - projected[1];
    }
    estimate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_reprojection_changes_with_depth_for_translated_cameras() {
        let geometry = DepthGeometry {
            calibration: DepthGeometryCalibration {
                rgb: Some(CameraIntrinsics {
                    width: 640,
                    height: 480,
                    fx: 500.0,
                    fy: 500.0,
                    cx: 320.0,
                    cy: 240.0,
                    ..CameraIntrinsics::default()
                }),
                depth_to_rgb: Some(RigidTransform3 {
                    translation_m: [0.05, 0.0, 0.0],
                    ..RigidTransform3::default()
                }),
                ..DepthGeometryCalibration::default()
            },
        };
        let near = geometry.depth_point_to_rgb_pixel([0.0, 0.0, 1.0]).unwrap();
        let far = geometry.depth_point_to_rgb_pixel([0.0, 0.0, 2.0]).unwrap();
        assert!(near[0] > far[0]);
        assert!((near[0] - far[0] - 12.5).abs() < 0.001);
    }
}
