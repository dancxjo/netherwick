# Sensor latency calibration

`NowBuilder` records timing evidence for body, camera, audio, range, IMU, Kinect, and optional
lidar streams. Each observation retains producer time, Motherbrain receive time, canonical frame
time, producer clock epoch/confidence, and any correlated event features. No estimator work occurs
on the brainstem, sensor poll, heartbeat, or motor-safety path.

For each enabled stream, `Now.extensions["sensor.latency_calibration"]` and the captured calibration
asset report a rolling latency distribution: median, p95, median-absolute-deviation jitter,
uncertainty, evidence count, correlated-event count, confidence, trust state, epoch, and rejection
reasons. Missing timestamps and low-confidence clocks remain explicitly unobservable. Lidar is
declared optional and disabled lidar does not block any other stream.

The real aggregation path automatically correlates safe rotation-start edges from Create odometry
with IMU rotation edges. Other producers can submit domain-specific repeated events through
`NowBuilder::observe_latency_reference_event` and `NowBuilder::observe_sensor_timing`; this keeps
camera, audio, and depth event extraction independent from the timing estimator. A held-out event
validator checks new offsets against the learned distribution and an explicit tolerance.

A producer clock-epoch change or a median shift beyond the configured bound advances only that
stream's calibration epoch and clears its evidence window. Stale or invalidated Kinect/IMU timing
removes fusion alignment; unobservable or estimating timing lowers its confidence. Replay restores
the selected per-stream epoch and raw timing observation from the captured snapshot rather than
relying on a process-global constant.

Physical acceptance still requires repeated measured events on the assembled robot across each
enabled producer, followed by a deliberate latency step and held-out trials. Those trials should
confirm the declared tolerances and verify that sensor load never delays brainstem polling or motor
safety.
