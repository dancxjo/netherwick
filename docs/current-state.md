# Current Netherwick State

This note records the current working shape of Netherwick after the first convincing real-world voxel milestone.

## What works now

Netherwick can render aligned, colored 3D voxels from real sensor data. The output is blocky and Minecraft-like, but the important feature is not visual polish. The important feature is correspondence: colored voxel structures line up with real things in the room.

This means the perception stack has crossed from raw capture and loose point cloud experiments into an inspectable spatial model.

## What is being restored

### 2D map

The 2D map path drifted out of alignment and then stopped working. It should now be treated as a projection of the same coordinate truth used by the 3D voxel world, not as a separate mapping universe.

Debug order:

1. Confirm map update events still exist.
2. Confirm the renderer is subscribed to the current event shape.
3. Confirm the map uses the same robot/world frame as the voxel projection.
4. Replay a known capture and compare voxel output against map output.
5. Save the failure as a regression fixture.

### Movement

Movement appears to have broken somewhere in the command-to-base path. The likely explanations are:

- safety veto,
- wrong robot mode,
- stale or missing base connection,
- command path regression,
- body state that correctly refuses movement, such as docked, cliff, bumper, or fault state.

Debug order:

1. Log the generated movement intent.
2. Log the controller receipt.
3. Log the safety decision with reason codes.
4. Log the outbound base command.
5. Log the robot's reported mode and body state.

The desired fix is not to bypass safety. The desired fix is a legible refusal path, so Pete can say why he will not move.

## Capture target

The next golden run should include:

- screenshot of the 3D voxel world,
- short video capture of live voxel updates,
- raw RGB frames,
- raw depth frames,
- IMU and odometry if available,
- command intents,
- safety decisions,
- body mode and serial/base status,
- 2D map output, even if broken,
- compact scene JSON.

A golden run gives the project a fixed specimen. When a later change breaks spatial alignment, the capture can answer whether the failure came from input, projection, mapping, rendering, or action plumbing.

## Near-term definition of success

Netherwick should be considered healthy when:

- the voxel view shows real objects in stable 3D,
- the 2D map agrees with the voxel/world frame,
- movement commands either execute or produce explicit safety/mode refusal reasons,
- captures can be replayed without the robot present,
- the WebXR view can be used as an inspection surface rather than a novelty.
