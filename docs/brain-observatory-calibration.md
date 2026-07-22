# Brain Observatory calibration and trust console

The calibration console is synchronized to the retained canonical `Now`
snapshot selected by the timeline. Its endpoint reads the same snapshot for
live and replay and reports four estimator families: Kinect geometry/mount,
IMU bias/noise/mounting, per-stream timing, and locomotion/wheel calibration.

Each estimator card keeps the configured prior separate from the live estimate
and shows recorded trust, confidence, covariance/uncertainty, observable and
unobservable degrees of freedom, evidence counts and age, residuals,
thresholds, rejection/invalidation reasons, epoch, and held-out validation.
Missing validation or estimator state is labeled `not_recorded` or
`not_observed`; the console does not manufacture a pass.

Consumer gates explain why depth association, mapping, orientation,
cross-stream association, or navigation correction is allowed or blocked.
Partial observability is never equivalent to full transform trust. An IMU may,
for example, allow trusted roll/pitch correction while absolute yaw remains
blocked. Locomotion learning is always shown as advisory and cannot grant
brainstem motor or safety authority.

An epoch change between the preceding and selected snapshot receives a red
remount treatment. Surface/tire changes are called out explicitly. Mini-plots
cover mount transform parameters, gyro bias/noise, latency distribution,
wheel scales, CW/CCW rotation scale, wheelbase, and confidence. Lidar is
identified as optional corroboration: missing lidar is visible but is not an
independent trust requirement.

## Canonical transition ownership

Calibration history is authored where the decision is made, never by comparing
Observatory snapshots. The Kinect transform state machine, IMU mounting/bias
estimator, locomotion distance/rotation estimator, and each latency estimator
emit a loss-intolerant transition only for an accepted or rejected candidate,
degradation, invalidation/remount, rollback, or newly trusted state. Periodic
state projection with no decision or trust change remains silent.

Each transition retains the complete prior/new calibration view, epoch,
per-degree observability and uncertainty, evidence window, supporting evidence
event and artifact checksums, candidate/prior/accepted artifact identities,
consumer impact, reason, and occurred/observed clock domains. Runtime converts
the estimator-authored records into canonical `BrainEvent` evidence and
transition envelopes; `pete-server` only retains and serves them. Replay applies
the same records in order to reconstruct the identical epoch/trust sequence.
