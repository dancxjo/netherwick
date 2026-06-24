# WebXR Sensorium Viewer

Netherwick serves a read-only browser sensorium for live sessions at `/view/3d`. The existing 2D page at `/view` remains available.

## Start The Live Server

```bash
just live-server
```

For a local cargo smoke run:

```bash
cargo run -p netherwick-tools -- sim \
  --live \
  --live-addr 127.0.0.1:8787 \
  --steps 100000 \
  --tick-delay-ms 100
```

Open:

- `http://127.0.0.1:8787/view` for the original 2D panel.
- `http://127.0.0.1:8787/view/3d` for the 3D sensorium.
- `http://127.0.0.1:8787/view/scene` for the compact scene JSON packet.

## WebXR Mode

The 3D page runs in desktop mode everywhere WebGL and module scripts are available. Drag to orbit, scroll to zoom, and right-drag to pan.

If `navigator.xr` reports `immersive-vr` support, the page enables an Enter VR button. Most browsers require a secure origin for WebXR. Localhost is normally treated as secure; LAN IPs may require HTTPS or browser/device-specific setup.

The page does not expose motor commands, WebXR controller driving, or other actuation controls.

## Coordinate System

The scene uses this mapping:

- Netherwick sim/world `x_m` maps to Three.js `x`.
- Netherwick sim/world `y_m` maps to Three.js `z`.
- Three.js `y` is up.
- `heading_rad` rotates around the Three.js `y` axis.

## Rendered Now

The v0 viewer renders:

- ground plane and arena when scenario metadata is available
- robot body marker and heading arrow
- range beams as 3D rays with hit markers
- scenario objects such as obstacles, chargers, people, and speakers
- RGB eye frame as a floating textured panel
- coarse Kinect/depth point cloud when compact depth values exist
- Kinect skeleton records in the scene JSON
- audio bearing from Kinect, or from a simulated speaker when metadata is available
- HUD text for pose, battery, nearest obstacle, points, audio, and available warnings

If a field is unavailable, `/view/scene` returns `null`, an empty array, or a warning rather than failing.

## Capture Scene Hook

Recorded captures can be converted to the same scene schema with:

```text
/view/capture-scene?capture=data/captures/real/mock-assets-smoke&frame=0
```

The endpoint reads `frames.jsonl` from the capture directory and returns the selected frame as a scene packet. If the capture manifest includes scenario metadata, arena and objects are included. PLY point-cloud asset loading is noted in warnings and is planned as a follow-up browser import path.

## Planned Later

The schema has room for richer action state, safety state, memory heatmaps, predicted futures, counterfactual ghosts, calibrated point clouds, and direct PLY replay.

## Privacy

This view exposes robot sensor data over the HTTP address it is bound to. If the server binds to `0.0.0.0`, anyone on the network who can reach the port may see the robot's sensor view.
