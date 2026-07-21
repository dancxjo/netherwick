# Brainstem IMU handoff

## Ownership boundary

Brainstem owns Create wheel odometry, MPU-6050 acquisition, sample and body-packet timestamps,
stationary gyro-bias estimation, board mounting identity, health, and immediate tilt/impact/contact/
watchdog/motion interlocks. Those reflexes remain active when Motherbrain is disconnected, rejects
the sample, disables fusion IMU, or restarts.

Motherbrain owns mapping the 32-bit brainstem clock into its host clock, candidate discovery and
selection, sensor-to-depth alignment, 3D fusion, pose-graph correction, mapping, and SLAM. Wheel
odometry supplies planar heading; the MPU supplies roll, pitch, angular velocity, and acceleration;
scan registration/SLAM correct heading. SLAM-corrected poses never feed back into raw brainstem
odometry. The MPU-6050 gyro-integrated yaw is diagnostic firmware state and is not published as an
absolute `ImuSense` yaw.

## Wiring and mounting contract

The brainstem MPU-6050 shares Pico I2C1: SDA is GP2/physical pin 4 and SCL is GP3/physical pin 5.
The current fixed installation is declared in `crates/pete-brainstem/board.toml` as sensor X/Y/Z
aligned with base forward/left/up. Change that board declaration when the physical installation
changes; do not add scattered sign flips. Set `mounting_calibrated` true only after verifying the
declared axes on the assembled robot.

At boot, keep the robot stationary and level. Firmware collects 50 plausible-gravity, low-angular-
rate samples to estimate gyro bias. The existing attended `zero_imu_orientation`/orientation probe
establishes the gravity/down reference. Gravity zeroing is not gyro-bias calibration: status reports
the two gates separately, and Motherbrain rejects geometry trust until both are true.

## Discovery and overrides

Normal operation needs no IMU option:

```bash
just possess
```

Brainstem telemetry is discovered as `brainstem_board_imu`. A future local MPU registers only when
Linux has enumerated a supported I2C address or an operator explicitly supplies a device such as
`--imu /dev/i2c-1@0x69`; `/dev/i2c-1` is not assumed to exist. Automatic selection is the default.
Diagnostic/recovery overrides are:

```bash
just possess --imu-source brainstem
just possess --imu-source local-i2c --imu /dev/i2c-1@0x68
just possess --imu-source none
```

`--imu none` remains a backward-compatible fusion-disable spelling. A forced source still has to
pass health, freshness, clock, mounting, gyro-bias, confidence, completeness, and plausible-tilt
gates. It cannot bypass trust or brainstem safety.

## Clock and source epochs

Each status contains uptime at generation, exact MPU sample time, exact Create body-packet time,
and a firmware-local clock epoch. Motherbrain timestamps the request start and response receipt,
uses their midpoint as the bounded status-generation estimate, and maps sensor timestamps by their
wrapping age from uptime. The host clock is a process-stable monotonic clock anchored once to Unix
time so it remains in the `Now` timestamp domain without following later wall-clock adjustments.
Excessive round-trip latency lowers clock confidence. `sample_age_ms` remains a diagnostic and a
lower-confidence fallback; timestamp presence alone never proves fresh data.

The mapper handles 32-bit wrap without declaring a reboot. Uptime regression, a changed firmware
epoch, or transport reconnect advances the host source epoch. Source/clock epoch changes clear the
canonical IMU interpolation history and current Kinect alignment; samples across epochs never join.
Out-of-order or future samples are rejected.

## Selection and bring-up evidence

Motherbrain retains separate candidate histories. Fresh, healthy, mounting-calibrated,
gyro-bias-calibrated, well-timed candidates clear mandatory gates; orientation confidence ranks
the survivors, and otherwise equivalent candidates prefer the brainstem. A challenger must remain
materially better for three evaluations before a live source switch. A switch advances the fusion
source epoch and waits for two samples from the new isolated history.

Inspect the runtime `sensor.imu_selection` extension. It reports candidates, rejection reasons,
selected source, source epoch, switch reason, sample age, calibration/confidence, per-source history
count, and `kinect_history_ready`. Expected normal observations are:

- `brainstem_board_imu` appears even before it is trusted;
- gyro-bias and mounting gates become true, orientation confidence reaches the trusted range, and
  `selected_source` becomes `brainstem_board_imu`;
- after two fresh samples, `kinect_history_ready` is true;
- `Now.imu.source_id` and `KinectFusionAlignment.imu.source_id` both read
  `brainstem_board_imu`, with bounded pose/IMU skew;
- reboot, reconnect, stale/unhealthy telemetry, or lost calibration removes alignment instead of
  reusing the old sample.

If a local provider is explicitly present, its candidate remains visible with its own history.
Seeing `selected_source: brainstem_board_imu` plus the same provenance inside Kinect alignment is
the direct proof that fusion is not using local I2C.
