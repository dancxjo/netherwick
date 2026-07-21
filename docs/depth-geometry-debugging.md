# Depth Geometry Debugging

For live mount epochs, transform covariance, observability, shift detection,
and remount closure trials, see [Adaptive sensor calibration](adaptive-calibration.md).

Pete uses these depth geometry conventions:

- Kinect camera frame: `+x` right, `+y` down, `+z` forward.
- Robot/base math frame: `+x` forward, `+y` left, `+z` up. The floor is `z = 0`.
- World/odometry math frame: `+x` and `+y` are odometry planar axes, `+z` is up.
- Live 3D scene frame: Babylon `+y` is vertical. Calibrated `ScenePoint` depth uses `scene_robot_render`, with `x = robot_y`, `y = robot_z`, `z = robot_x`. The browser must convert every point cloud through the named frame helpers in `renderPoints`: `robotRenderPointToBabylonLocal`, `kinectCameraPointToBabylonLocal`, and `worldMathPointToBabylonWorld`.

The coordinated-grid rule is:

```text
raw depth image -> kinect_camera -> robot/base math -> world/odometry math -> Babylon display
```

Only the boundary between two named frames may change handedness. Do not add sign
flips at call sites. If a cloud appears mirrored or behind the robot, fix the
named conversion for that source frame and add or update the corresponding
viewer regression.

## Report First

Generate a report from a real capture before tuning the UI:

```bash
cargo run -p pete-tools -- geometry-debug \
  --capture data/captures/real/rpi5-smoke \
  --out data/reports/geometry/rpi5-smoke.json
```

Or report directly from a running live robot dashboard:

```bash
cargo run -p pete-tools -- geometry-debug \
  --live-now-url http://127.0.0.1:3000/now \
  --out data/reports/geometry/live-now.json
```

Read the warnings first. Fallback intrinsics, unknown depth dimensions, or an assumed IMU contract mean the geometry is not trustworthy yet.

The report also includes `sensor_truth.ready_for_real_slam`. Do not start real SLAM integration until it is `true`. The gate list must show:

- `depth_intrinsics_non_fallback`: depth has real dimensions and `fx/fy` values.
- `below_floor_ratio`: current-frame transformed floor leakage is under the selected threshold.
- `frame_timestamps_monotonic`: capture frame timestamps are ordered and sane.
- `body_timestamp_fresh`: odometry/body time is close to the selected depth frame time.
- `multi_frame_depth_capture`: the report contains at least two distinct depth frames.
- `camera_geometry_calibrated`: the current camera supplies measured intrinsics, distortion, depth scale/bias, RGB-D extrinsics, and full 6-DoF depth-to-base extrinsics.
- `rgb_depth_paired`: color and depth share one RGB-D frame ID and meet the device-clock skew limit.
- `kinect_timestamp_fresh`: Kinect depth has its own `captured_at_ms` and it is within the selected freshness limit.
- `imu_timestamp_fresh`: IMU has its own `captured_at_ms` and it is within the selected freshness limit.
- `kinect_imu_synchronized`: the Kinect and IMU capture timestamps are within the selected skew limit.
- `kinect_body_pose_synchronized`: a buffered body pose was interpolated sufficiently close to the depth exposure.
- `imu_roll_pitch_contract`: IMU shape is recognized as `[roll, pitch]` or `[roll, pitch, yaw]` radians and roll/pitch correction is active.
- `stationary_rotation_cloud_stability`: a rotate-in-place capture has enough stable voxels with limited vertical spread.

For the stationary rotation gate, use a capture where the robot turns at least 45 degrees while translating less than 0.20 m. A normal driving capture will be marked `not_applicable` for that gate and is not enough to clear SLAM readiness.

Old captures made before sensor `captured_at_ms` fields were added deserialize
those timestamps as `0` and fail the freshness and synchronization gates. A
present but stale timestamp also fails. Re-capture after this change before
judging SLAM readiness.

The live runtime `LocalMap` consumes place/entity loop candidates, registers the
current scan against the target submap, optimizes the anchored pose graph, and
rebuilds occupancy from corrected submaps. The dashboard receives that same map
instead of constructing a second one. A loop candidate without a measured
scan/submap registration stays rejected. On real captures, require
`sensor_truth.ready_for_real_slam = true` first; failed gates should be fixed in
projection, extrinsics, timestamp plumbing, IMU interpretation, or the
camera/world transform chain. Do not change renderer coordinates to hide
below-floor leakage or unstable accumulated clouds.

After the stationary-rotation gate passes, record a small return-to-start route
and run:

```bash
cargo run -p pete-tools -- representation-report \
  --capture data/captures/real/return-to-start \
  --out data/reports/representation/return-to-start.json
```

Require `return_to_start.passed = true`. That gate requires a measured loop
constraint, lower final RMS graph error, improved registered wall overlap, and
a corrected final pose within 0.25 m of the start. Depth observations are
retained and the voxel world is rebuilt through corrected graph-node poses. The
dashboard keeps the projection untrusted if any retained observation cannot be
associated with the corrected graph.

## Calibration Procedure

1. Disable accumulation or visually separate it from current-frame points.
2. Show raw camera-frame depth and verify the image is not mirrored, transposed, or clipped.
3. Verify depth image metadata: width, height, vector length, `fx`, `fy`, `cx`, `cy`, min/median/max depth, skipped count, and clipped count.
4. If real Kinect depth is using fallback projection, fix intrinsics before adjusting extrinsics.
5. Enable camera-to-base extrinsics. With zero rotation, a centered point at 1 m should land near robot `x = 1 m`, `y = 0`, `z = camera_height`.
6. Adjust camera height and pitch until observed floor samples land near robot/world `z = 0`.
7. Adjust roll until left and right floor samples have matching height.
8. Check yaw by rotating a known forward point into the expected world axis.
9. Supply the measured IMU-to-base mounting rotation, hold the robot stationary through gyro-bias acquisition, and require the filtered orientation confidence gate. The first sample is never treated as an arbitrary "flat" zero.
10. Enable world accumulation. Rotate in place and verify stable voxels stay fixed instead of smearing or sinking.
11. Re-run `geometry-debug` on the rotation capture and require `sensor_truth.ready_for_real_slam = true`.
12. Trust `LocalWorldBelief` surfaces/blobs only when below-floor ratio is near zero and accumulated voxels remain stable under in-place rotation.

Copy `configs/kinect-calibration.example.json`, replace every nominal/zero value
with per-device measurements, set `calibrated` to `true`, and start the hardware
runtime with `PETE_KINECT_CALIBRATION_JSON=/absolute/path/to/calibration.json`.
The example deliberately has `calibrated: false` and cannot clear the physical
gate. Configure the IMU mount with `PETE_IMU_TO_BASE_RPY_DEG=roll,pitch,yaw` and
`PETE_IMU_MOUNT_CALIBRATED=true`; the orientation remains untrusted until the
stationary gyro-bias sample window also completes.

## Live View Checks

In `/view/3d`, watch:

- current and accumulated coordinate frame labels
- IMU raw vector and interpreted roll/pitch/yaw in degrees
- calibration height, forward offset, pitch, roll, yaw
- below-floor ratio
- math-frame `z` min/median and render-frame vertical min/median
- fallback-intrinsics and unknown-IMU-contract warnings

The floor plane is only a reference. Do not tune the renderer to hide points under the floor; fix projection, extrinsics, IMU interpretation, or the camera/world transform chain first.
