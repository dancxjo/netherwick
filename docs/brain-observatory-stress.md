# Brain Observatory stress and real-time validation

The stress harness produces a versioned JSON report rather than a pass message
without evidence. It runs a deterministic mixed stream with 64 coalescible
projection keys and periodic loss-intolerant transitions, preserves a sensor
clock reset in source sequence order, leaves one subscriber stalled, reconnects
another, queries an expired cursor, serializes history, restarts from the
durable critical log, and injects an independent writer failure. Baseline event
construction and enabled publication are timed separately.

Run the software/CI profile:

```bash
just observatory-stress profile=ci events=20000 \
  output=data/reports/observatory-stress/software.json
```

The report directory beside the JSON contains the exact checksummed durable
fixtures used for restart/replay and failure injection. Each run gets a fresh
UUID directory, so a previous stable event identity cannot make a later run
silently pass as a duplicate.

## Software thresholds

The CI profile fails unless all of these remain true:

- ingress depth is at most 512 and live history at most 512 records;
- no critical ingress rejection or normal durable-write failure occurs;
- a stalled subscriber reports viewer-local lag while a reconnect receives the
  current live event;
- old-cursor gaps remain distinct from critical rejection and viewer lag;
- recovered critical `(sequence, event_id)` pairs exactly match original order;
- the clock reset retains its changed epoch and does not reorder causality;
- injected writer failure is visible while ingestion drains;
- enabled p99 publication overhead above event construction is at most 1 ms;
- measured process RSS growth is at most 128 MiB;
- the configured publication deadline has no misses.

Latency sampling is capped at 100,000 observations, live queues and history are
bounded by configuration, coalescible projections never enter the durable log,
and durable retention is byte/segment bounded. The harness therefore does not
create an unbounded observer while trying to prove boundedness.

Focused regression command:

```bash
cargo test -p pete-server \
  software_observatory_stress_harness_reports_bounded_truthful_behavior \
  -- --test-threads=1
```

## Physical Pi 5 soak: intentionally pending

The software result does not prove Pi scheduling, storage, camera/IMU provider,
brainstem, or thermal behavior. On the deployed Pi 5, first record the hardware
environment, then run the two-hour paced profile (360,000 20 ms publications)
under the process resource recorder:

```bash
just hardware-env
/usr/bin/time -v just observatory-stress profile=pi5-soak events=360000 \
  output=data/reports/observatory-stress/pi5-soak.json \
  2>&1 | tee data/reports/observatory-stress/pi5-soak.time.txt
```

The Pi profile enables `sync_data`, requires p99 added publication time at or
below 500 microseconds, zero 20 ms publication-deadline misses, zero critical
or durable loss, bounded queues/history, and RSS growth at or below 192 MiB.
Retain the JSON, per-run durable fixture directory, `/usr/bin/time` output, and
`just hardware-env` report.

That resource soak must be followed by a real sensor/control session so Pi,
brainstem, Kinect/camera, IMU, capture, and runtime clock epochs are represented:

```bash
just possess-rpi5 --dashboard 127.0.0.1:8787 --tick-ms 20 \
  --capture data/captures/observatory-pi-soak
```

Keep it running for at least two hours with normal stationary and cautious
operator-driven motion, periodically pause/reconnect a browser, and export the
full diagnostic interval before shutdown. Acceptance requires no new runtime
deadline miss or sensor backlog, no durability/security failure, truthful clock
epoch transitions across any reconnect, and no change to navigation,
calibration-promotion, watchdog, hardware-arm, or physical-safety gates. Lidar
is optional; its absence is not a soak failure.
