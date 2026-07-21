# Bounded local visual recognition

Pete's object recognizer is advisory perception. It cannot issue an action,
select a goal, acquire possession, or bypass conductor, autonomic safety,
Cockpit, or brainstem authority. The physical control loop only submits an
immutable RGB snapshot to a background worker and drains completed evidence
without waiting for inference.

## Runtime path

The real path is:

1. Kinect publishes an atomically identified RGB-D pair; an optional V4L camera
   can publish an independent `EyeFrame`.
2. `FrameProcessor` retains the ordinary eye/face path and offers the immutable
   RGB frame plus its matching Kinect snapshot to `VisionPipeline`.
3. The bounded queue replaces its oldest pending frame when full. Frequency
   limiting drops excess frames before enqueue, and deadline checks reject work
   before or after inference. None of these paths wait in the possession tick.
4. A backend preprocesses the frame and returns domain-neutral proposals. The
   built-in `classical` backend performs local RGB conversion, bounded resize,
   luma segmentation, and connected-component proposals without network or
   model weights.
5. The generic IoU tracker assigns an explicitly short-lived track id. A
   calibration-epoch transition clears continuity; the id is not an entity id.
6. Trusted depth is associated only when RGB-D frame ids and dimensions match,
   physical calibration validation passes, and the active geometry transform
   is trusted. Robot/world positions retain uncertainty and explicit reasons
   when depth or world placement is unavailable.
7. `ObjectSense` stores the detections and health report. The same snapshot is
   captured by WorldLab, available at `/view/vision`, and converted into labeled
   descendant sensations/crops for vectorization, impressions, and memory.

Each result retains source frame/snapshot/sensation ids, stream and producer
time, image dimensions and bounds, ranked label hypotheses, backend/model
identity and checksum, inference times and deadline, track id, calibration
epoch/trust, crop bytes, and optional depth/robot/world position.

Entity memory keys detector evidence by its short-term track (or unique
descendant evidence id when there is no track). Repeated frames can strengthen
that hypothesis, but a label such as `chair` never becomes permanent identity.

## Raspberry Pi 5 profile

The default `raspberry-pi-5` profile is explicit and serialized in runtime and
evaluation health reports:

| Limit | Default |
| --- | ---: |
| Preprocessed input | 320 x 240 RGB |
| Detection frequency | 5 fps maximum |
| Pending queue | 2 frames, oldest replaced |
| Completed-result queue | 4 batches, oldest dropped |
| Inference deadline | 180 ms |
| Model threads | 2 |
| Vision memory envelope | 96 MiB |
| Detections per frame | 8 |
| Track age | 750 ms |

The health payload reports queue depth; queued, processed, replaced, dropped,
expired, stale, and failed counts; p50/p95 inference duration; backend state;
and the latest error. A crashed worker cannot stall control because it owns no
control-loop lock or channel receive. Dropped and expired work is expected under
load and is preferable to accumulating latency.

`PETE_VISION_BACKEND=classical` is the default physical profile.
`PETE_VISION_BACKEND=off` keeps the worker's explicit `missing` health state but
does not prevent conservative operation. Unknown backend names also degrade to
that state with a warning. MJPEG or partial frames that the built-in preprocessor
cannot decode are counted as failures rather than guessed into existence.

## Replay-first evaluation

Run the checked-in positive, negative, occlusion, blur, low-light, empty-scene,
and two-frame tracking fixtures:

```sh
just vision-eval
```

Compare different configured backend names on exactly the same ordered inputs:

```sh
just vision-eval \
  --candidate classical --baseline unavailable --out /tmp/vision-eval.json
```

Or run on every hydrated Kinect/camera RGB frame in a recorded WorldLab capture
(annotations are optional, so precision/recall remain absent when there is no
ground truth):

```sh
just vision-eval \
  --capture data/captures/real/rpi5-smoke \
  --candidate classical --baseline classical
```

The JSON report preserves ordered per-frame detections and track ids, fixture
coverage tags, label precision/recall, duplicate and fragmentation counts,
p50/p95 inference time in microseconds and milliseconds, throughput, failure
reasons, backend identity, and the complete resource profile for both candidate
and baseline. The checked-in seven-frame fixture gate measures the built-in
backend at sub-millisecond p50/p95 on the current x86 development host while
finding all five annotated objects (one deliberate duplicate under occlusion).
That measurement is a software regression reference, not a claimed Pi 5
benchmark; run the same command on the deployed Pi 5 and retain its JSON report
before raising the physical profile's frequency or deadline.

No model weights are committed or downloaded by this implementation. A future
OpenCV DNN or ONNX backend belongs behind the same `VisionBackend` interface and
must use a license-reviewed artifact with a pinned SHA-256 checksum; missing or
mismatched weights must select an explicit unavailable state rather than fall
back silently.
