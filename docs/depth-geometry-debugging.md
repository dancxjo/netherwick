# Depth Geometry Debugging

Netherwick uses these depth geometry conventions:

- Kinect camera frame: `+x` right, `+y` down, `+z` forward.
- Robot/base math frame: `+x` forward, `+y` left, `+z` up. The floor is `z = 0`.
- World/odometry math frame: `+x` and `+y` are odometry planar axes, `+z` is up.
- Live 3D scene frame: Babylon `+y` is vertical. Calibrated `ScenePoint` depth uses `scene_robot_render`, with `x = robot_y`, `y = robot_z`, `z = robot_x`, and the browser maps local forward to robot local `-z`.

## Report First

Generate a report from a real capture before tuning the UI:

```bash
cargo run -p netherwick-tools -- geometry-debug \
  --capture data/captures/real/rpi5-smoke \
  --out data/reports/geometry/rpi5-smoke.json
```

Read the warnings first. Fallback intrinsics, unknown depth dimensions, or an assumed IMU contract mean the geometry is not trustworthy yet.

The report also includes `sensor_truth.ready_for_real_slam`. Do not start real SLAM integration until it is `true`. The gate list must show:

- `depth_intrinsics_non_fallback`: depth has real dimensions and `fx/fy` values.
- `below_floor_ratio`: current-frame transformed floor leakage is under the selected threshold.
- `frame_timestamps_monotonic`: capture frame timestamps are ordered and sane.
- `body_timestamp_fresh`: odometry/body time is close to the selected depth frame time.
- `kinect_timestamp_carried`: Kinect depth has its own capture timestamp, not only an enclosing capture-frame timestamp.
- `imu_timestamp_carried`: IMU has its own capture timestamp, not only an enclosing capture-frame timestamp.
- `imu_roll_pitch_contract`: IMU shape is recognized as `[roll, pitch]` or `[roll, pitch, yaw]` radians and roll/pitch correction is active.
- `stationary_rotation_cloud_stability`: a rotate-in-place capture has enough stable voxels with limited vertical spread.

For the stationary rotation gate, use a capture where the robot turns at least 45 degrees while translating less than 0.20 m. A normal driving capture will be marked `not_applicable` for that gate and is not enough to clear SLAM readiness.

## Calibration Procedure

1. Disable accumulation or visually separate it from current-frame points.
2. Show raw camera-frame depth and verify the image is not mirrored, transposed, or clipped.
3. Verify depth image metadata: width, height, vector length, `fx`, `fy`, `cx`, `cy`, min/median/max depth, skipped count, and clipped count.
4. If real Kinect depth is using fallback projection, fix intrinsics before adjusting extrinsics.
5. Enable camera-to-base extrinsics. With zero rotation, a centered point at 1 m should land near robot `x = 1 m`, `y = 0`, `z = camera_height`.
6. Adjust camera height and pitch until observed floor samples land near robot/world `z = 0`.
7. Adjust roll until left and right floor samples have matching height.
8. Check yaw by rotating a known forward point into the expected world axis.
9. Enable IMU roll/pitch correction only after confirming the producer emits radians in `[roll, pitch]` or `[roll, pitch, yaw]` order.
10. Enable world accumulation. Rotate in place and verify stable voxels stay fixed instead of smearing or sinking.
11. Re-run `geometry-debug` on the rotation capture and require `sensor_truth.ready_for_real_slam = true`.
12. Trust `LocalWorldBelief` surfaces/blobs only when below-floor ratio is near zero and accumulated voxels remain stable under in-place rotation.

## Live View Checks

In `/view/3d`, watch:

- current and accumulated coordinate frame labels
- IMU raw vector and interpreted roll/pitch/yaw in degrees
- calibration height, forward offset, pitch, roll, yaw
- below-floor ratio
- math-frame `z` min/median and render-frame vertical min/median
- fallback-intrinsics and unknown-IMU-contract warnings

The floor plane is only a reference. Do not tune the renderer to hide points under the floor; fix projection, extrinsics, IMU interpretation, or the camera/world transform chain first.
