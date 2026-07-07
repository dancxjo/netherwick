# WebXR Instant Viewer

Pete serves a browser Instant viewer for live embodied sessions at `/view/3d`. The existing 2D page at `/view` remains available.

## Start The Live Server

```bash
just live-server
```

For a local cargo smoke run:

```bash
cargo run -p pete-tools -- sim \
  --live \
  --live-addr 127.0.0.1:8787 \
  --steps 100000 \
  --tick-delay-ms 100
```

For a LAN headset-friendly HTTPS virtual theater:

```bash
just go virtual
```

See [go-virtual.md](go-virtual.md) for certificate and headset setup.

Open:

- `http://127.0.0.1:8787/view` for the original 2D panel.
- `http://127.0.0.1:8787/view/3d` for the 3D Instant view.
- `http://127.0.0.1:8787/view/scene` for the compact scene JSON packet.
- `http://127.0.0.1:8787/api/experience/lineage` for the current embodied lineage graph payload.

## WebXR Mode

The 3D page runs in desktop mode everywhere WebGL and module scripts are available. Drag to orbit, scroll to zoom, and right-drag to pan.

If `navigator.xr` reports `immersive-vr` support, the page enables an Enter VR button. Most browsers require a secure origin for WebXR. Localhost is normally treated as secure; LAN IPs usually need HTTPS or browser/device-specific certificate setup.

In immersive VR, WebXR controllers can feed the Reign queue through `/reign/command`:

- Thumbstick up sends a short Direct `Go` lease.
- Thumbstick left or right sends a short Direct `Turn` lease.
- Squeeze or secondary button sends `Stop`.
- Primary auxiliary buttons request `Dock` or `Explore` where supported by the controller mapping.

Commands are sent with `source: "Gamepad"` and short TTLs so steering expires quickly when the controller returns to neutral.

## Coordinate System

The scene uses this mapping:

- Pete sim/world `x_m` maps to Three.js `x`.
- Pete sim/world `y_m` maps to Three.js `z`.
- Three.js `y` is up.
- `heading_rad` rotates around the Three.js `y` axis.

## Rendered Now

The v0 viewer renders:

- ground plane and arena when scenario metadata is available
- robot body marker and heading arrow
- range beams as 3D rays with hit markers
- scenario objects such as obstacles, chargers, speaking dream people, and distant voices
- RGB eye frame as a floating textured panel
- coarse Kinect/depth point cloud when compact depth values exist
- Kinect skeleton records in the scene JSON
- audio bearing from Kinect, or from a speaking dream person when metadata is available
- HUD text for session mode, scenario, seed, tick, URL scheme, secure-context status, WebXR support, pose, battery, nearest obstacle, points, audio, and available warnings
- XR Reign status while immersive controller input is available

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

This view exposes robot sensor data over the HTTP address it is bound to. If the server binds to `0.0.0.0`, anyone on the network who can reach the port may see the robot's sensor view. When a VR session is active, controller input can also submit Reign commands to the same server.
