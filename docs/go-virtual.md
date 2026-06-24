# `just go virtual`

`just go virtual` starts Netherwick as a virtual live training theater: a simulated scenario updates the live view state, the server binds to `0.0.0.0`, and HTTPS pages are served for desktop browsers and LAN headset browsers.

This mode collects experience into the ledger. It does not update model weights online. Startup prints:

```text
Virtual training theater is collecting experience.
Models are not updated online in this run.
Train later with `cargo run --bin netherwick -- train behavior ...`
```

The `/view/scene` packet and `/view/3d` HUD expose `training_mode`, `ledger_path`, written frame/transition counts, loaded models, model modes, action selector mode, and `weights_updating`. For the default virtual run, expect `training_mode: "collecting"` and `weights_updating: false`.

Open:

- Desktop: `https://127.0.0.1:8787/view/3d`
- Headset/LAN: `https://<machine-lan-ip>:8787/view/3d`
- Scene JSON: `https://<machine-lan-ip>:8787/view/scene`

The 3D page shows live sensors and, in immersive VR, maps supported controller inputs into the Reign queue. Thumbstick steering sends short `Go` and `Turn` leases, squeeze/secondary sends `Stop`, and auxiliary buttons can request `Dock` or `Explore`.

## Why HTTPS

WebXR immersive VR requires a secure browser context. `localhost` is often treated as secure, but a headset visiting a LAN IP usually needs HTTPS and a trusted local development certificate.

The Justfile generates `certs/netherwick-dev.crt` and `certs/netherwick-dev.key` with SAN entries for `localhost`, `127.0.0.1`, `netherwick.local`, and the first detected LAN IP when available.

If the headset warns about the certificate, accept the warning for local testing or install/trust the generated dev certificate on the headset.

## Options

Change port:

```bash
NETHERWICK_LIVE_PORT=9443 just go virtual
```

Change scenario:

```bash
NETHERWICK_SCENARIO=charger-seeking just go virtual
```

Useful scenario slugs:

- `empty-room`
- `obstacle-avoidance`
- `corner-trap`
- `column-trap`
- `charger-seeking`
- `person-speaker-room`
- `mixed-room`

Change the long-running step budget or tick delay:

```bash
NETHERWICK_SIM_STEPS=500000 NETHERWICK_TICK_DELAY_MS=50 just go virtual
```

Open the desktop URL automatically when `xdg-open` is available:

```bash
NETHERWICK_OPEN_BROWSER=1 just go virtual
```

Print the detected headset URL without starting the theater:

```bash
just virtual-url
```

Stop it with `Ctrl-C`.

## Finding The Machine IP

The Justfile uses:

```bash
hostname -I | awk '{print $1}'
```

You can also run `ip addr` and use the address assigned to your Wi-Fi or Ethernet interface.

## Security

`just go virtual` intentionally binds to `0.0.0.0` so a headset on the LAN can connect. This serves robot/sim sensor data to any device that can reach the port, and the same origin exposes Reign command endpoints. Use only on trusted networks.

## Static Viewer Assets

The viewer imports vendored Three.js r166 files through local `/static/vendor/three/...` module paths, so `/view/3d` does not need the headset browser to reach a CDN.
