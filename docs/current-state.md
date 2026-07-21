# Current Pete State

This note records the current working shape of Pete after physical motherbrain possession became operational. Behavior validation is the active milestone.

## What works now

Pete can render aligned, colored 3D voxels from real sensor data. The output is blocky and Minecraft-like, but the important feature is not visual polish. The important feature is correspondence: colored voxel structures line up with real things in the room.

This means the perception stack has crossed from raw capture and loose point cloud experiments into an inspectable spatial model.

Pete can also take guarded physical possession through `just possess`. The
motherbrain lease is possession; bounded commands reach the brainstem, body
telemetry reaches `BodySense`, and orderly shutdown performs acknowledged STOP
and exorcize. Body freshness is tied to complete Create packet age rather than
to the time at which cached status was read. Reconnect begins stopped and does
not reopen the motor path until a newly received packet-0 body frame is fresh.

The LLM loop is also doing real work now. It is predicting counterfactual outcomes, critiquing training data, and suggesting motion intents. Some of those critiques are weird, but they are useful: they expose suspicious examples, unstable hypotheses, and tests the robot could run to become more certain.

## What is being generalized

### Constellations

A constellation should not be vision-only. The first obvious examples are stable arrangements of colored voxels, planes, corners, blobs, and depth edges, but the abstraction should apply to every modality Pete can observe or remember.

A constellation is a recurrent signature across experience. It may include:

- geometry and voxel structure,
- color and image evidence,
- motion and odometry,
- body state,
- audio and speech events,
- text labels and image descriptions,
- memory recalls,
- surprise and prediction error,
- LLM critiques,
- counterfactual predictions,
- suggested actions or motion intents.

The first question is not "What object is this?" The first question is "Have I seen this arrangement before, and what evidence would make me more sure?"

See [constellations.md](constellations.md) for the dedicated model.

## What is being validated

### 2D map

The live 2D map is now an explicit projection of the same calibrated
odometry-world voxels used by the 3D view, rather than a separate mapping
universe. The panel reports shared-frame alignment, geometry trust, and
navigation trust independently so a working visualization cannot silently
grant motion authority.

Debug order:

1. Confirm the shared-world projection continues updating from depth-only and
   range-assisted inputs.
2. Require stable projected cells and an acceptable below-floor ratio before
   calling the sensor geometry trusted.
3. Keep navigation gated until loop-closed corrected SLAM is ready.
4. Replay a synchronized physical capture and compare voxel output against map
   output.
5. Save any physical mismatch as a regression fixture.

### Behavior validation

The command-to-base path is working. The active task is validating that normal
behavior, autonomic policy, the final hardware gate, brainstem reflexes, and
recorded experience agree under real disturbances. Automated coverage includes
normal random walk → bump stop → bounded conductor recovery. The remaining
physical cases are:

- charging interlock,
- bumper recovery,
- left, front-left, front-right, and right cliff sensors,
- wheel drop,
- heartbeat loss,
- transport loss and stopped reconnect with fresh body telemetry.

Validation order:

1. Record the generated action and autonomic decision.
2. Record the final regular-possession gate and outbound bounded command.
3. Record brainstem safety and motion events.
4. Record complete-packet age and the resulting `BodySense`.
5. Confirm the observed body outcome and stopped shutdown state.

The goal is not merely wheel motion. The goal is agreement between intent,
refusal or execution, physical outcome, and the experience used for later
behavior training.

## Capture target

The next golden run should include:

- screenshot of the 3D voxel world,
- short video capture of live voxel updates,
- raw RGB frames,
- raw depth frames,
- IMU and odometry if available,
- command intents,
- LLM counterfactuals and critiques,
- suggested motion intents,
- safety decisions,
- body mode and serial/base status,
- 2D map output, even if broken,
- compact scene JSON.

A golden run gives the project a fixed specimen. When a later change breaks spatial alignment, the capture can answer whether the failure came from input, projection, mapping, rendering, LLM hypothesis generation, or action plumbing.

## Near-term definition of success

Pete should be considered healthy when:

- the voxel view shows real objects in stable 3D,
- the 2D map agrees with the voxel/world frame,
- recurrent cross-modal constellations can be saved and matched again,
- the LLM can critique candidates and suggest useful tests,
- possession and reconnect fail closed on stale or incomplete body telemetry,
- movement commands either execute or produce explicit safety/mode refusal reasons,
- the pending physical safety checklist has captured evidence for every case,
- captures can be replayed without the robot present,
- the WebXR view can be used as an inspection surface rather than a novelty.
