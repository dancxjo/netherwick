# Adaptive sensor calibration

Netherwick treats configured sensor transforms as initial guesses. The shared
calibration state machine in `pete-now` owns transform covariance, evidence
counts, residuals, trust, mount epochs, and invalidation. Projection still uses
the shared `DepthGeometry`; adaptive calibration does not create another depth
math path.

## Kinect mount state

`KinectSense.live_geometry_calibration` exposes these states:

- `configured`: a file supplied the initial transform, but experience has not
  made it observable;
- `estimating`: accepted evidence is reducing covariance, but one or more of
  x/y/z/roll/pitch/yaw is not sufficiently observed;
- `trusted`: every degree of freedom has enough independent evidence and its
  covariance is within the declared limits;
- `degraded`: an estimate remains usable for conservative perception while
  residuals exceed navigation-trust limits;
- `invalidated`: a discontinuity indicates a mount shift and starts a new
  epoch.

Floor-plane observations automatically contribute height, roll, and pitch at
the sensor-processing boundary. `TransformEstimator` and
`TransformEstimateEvidence` are the common interfaces for gravity, wheel
odometry, persistent walls/surfaces, map consistency, loop closure, and
optional lidar adapters. Lidar is never part of the minimum evidence set;
startup output reports a missing lidar stream as optional unless the operator
explicitly passes `--require-lidar`.

The state machine rejects stale, out-of-order, non-finite, zero-covariance, and
unobservable observations. Sudden transform, floor, gravity, reprojection, or
map-consistency changes invalidate the current epoch. Trust can return only
after the new epoch reconverges across all six degrees of freedom.

## Trust boundary

An estimating or degraded transform may support conservative perception. It
does not grant geometry or navigation trust. The live map requires a measured
camera calibration and a `trusted` live mount estimate when the new estimate is
present. This trust only gates derived geometry; it has no path to brainstem
motion, possession, reflex, or safety limits.

WorldLab capture frames retain the complete live estimate. Each calibration
asset includes the current transform, covariance, evidence counts, floor/wall
and other residuals, trust state, timestamp, epoch, and invalidation reason.
`geometry-debug` schema 4 summarizes all recorded epochs, state counts,
residual maxima, covariance maxima, and evidence counts across replay:

```bash
cargo run -p pete-tools -- geometry-debug \
  --capture data/captures/real/kinect-remount \
  --out data/reports/geometry/kinect-remount.json
```

## Physical closure for issue #89

The deterministic state machine, floor adapter, capture/replay diagnostics,
trust gate, and synthetic shift/reconvergence tests are software-complete. Do
not close the physical acceptance gate until one capture demonstrates all of
the following:

1. run without lidar and record that its absence is optional;
2. collect observable translation and rotation before a remount;
3. deliberately move the Velcro-mounted Kinect during the same session;
4. record an invalidated epoch followed by a distinct reconverged trusted
   epoch;
5. compare held-out floor/wall/map residuals before and after the remount
   against declared tolerances;
6. confirm deterministic motion and safety limits are unchanged.
