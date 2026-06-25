# `just go virtual`

`just go virtual` starts Netherwick's Dream World: a simulated dream scenario updates the live view state, the server binds to `0.0.0.0`, and HTTPS pages are served for desktop browsers and LAN headset browsers.

By default this mode collects experience into the ledger, starts Dream NEAT policy
training in the background, and runs inline world-outcome learning. Startup
prints:

```text
Virtual training theater is collecting experience.
Dream NEAT policy training starts automatically. Set NETHERWICK_NEAT_TRAINING=0 to disable.
Inline learning defaults to world-outcome. Set NETHERWICK_INLINE_LEARNING_MODE=off for collect-only.
Offline training still exists: `cargo run --bin netherwick -- train behavior ...`
```

The `/view/scene` packet and `/view/3d` HUD expose `training_mode`, `ledger_path`, written frame/transition counts, loaded models, model modes, action selector mode, and `weights_updating`. For the default virtual run, expect `training_mode: "inline-world-outcome"` and `weights_updating: true`.

Ledger writing remains on; Pete still writes memory while learning from the stream.

Collect-only:

```bash
NETHERWICK_INLINE_LEARNING_MODE=off \
just go virtual
```

Inline world-outcome learning:

```bash
NETHERWICK_INLINE_LEARNING_MODE=world-outcome \
NETHERWICK_INLINE_TRAIN_STEPS_PER_TICK=1 \
NETHERWICK_INLINE_BEHAVIORS=future,action_value \
just go virtual
```

Shadow-only learning/status:

```bash
NETHERWICK_INLINE_LEARNING_MODE=shadow-only \
NETHERWICK_INLINE_BEHAVIORS=danger,charge,future \
just go virtual
```

Offline training still exists and is still useful. Use `train virtual` or behavior training commands when you want repeatable batches, more epochs, promotion checks, and registry updates from a collected Dream World ledger.

Dream NEAT training writes checkpoints to `data/models/dream-policy/neat` and
distillation rollouts to `datasets/dream-policy/v0/episodes`. When a
`level-*-best.json` checkpoint exists, the visible Dream World robot is started
with the newest checkpoint as its controller. The browser HUD/scene JSON reports
this as an action selector mode like `dream-neat+baseline` and includes the
loaded `dream-neat:<level>:genome-<id>` model entry.

Tune or disable the automatic Dream NEAT run:

```bash
NETHERWICK_NEAT_GENERATIONS=60 \
NETHERWICK_NEAT_POPULATION=48 \
NETHERWICK_NEAT_START_LEVEL=motion \
just go virtual
```

```bash
NETHERWICK_NEAT_TRAINING=0 just go virtual
```

Use a specific visible-controller checkpoint:

```bash
NETHERWICK_DREAM_POLICY_CHECKPOINT=data/models/dream-policy/neat/level-1-best.json \
just go virtual
```

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

The default is `dream`, a seeded randomized Dream World. Re-run with the same
`--seed`/environment to get the same arena, object population, labels, colors,
voices, charger signals, and collision geometry.

Useful scenario slugs:

- `dream`
- `empty-room`
- `obstacle-avoidance`
- `corner-trap`
- `column-trap`
- `charger-seeking`
- `person-speaker-room`
- `mixed-room`

Dream generation knobs:

```bash
NETHERWICK_SCENARIO=dream \
NETHERWICK_DREAM_WEIRDNESS=0.65 \
NETHERWICK_DREAM_DENSITY=0.7 \
NETHERWICK_DREAM_SOCIALITY=0.5 \
NETHERWICK_DREAM_HAZARD_BIAS=0.35 \
NETHERWICK_DREAM_CHARGER_BIAS=0.4 \
just go virtual
```

- `NETHERWICK_DREAM_WEIRDNESS`: increases odd labels, colors, size variance,
  sound sources, landmarks, and asymmetric placement.
- `NETHERWICK_DREAM_DENSITY`: increases object population.
- `NETHERWICK_DREAM_SOCIALITY`: increases people and voices.
- `NETHERWICK_DREAM_HAZARD_BIAS`: increases blockers and charger-adjacent decoys.
- `NETHERWICK_DREAM_CHARGER_BIAS`: increases charger likelihood.

Dream objects are not decorative-only. Obstacles and landmarks project into
range, eye frames, Kinect depth/color, and collision; people project into face,
voice, ear, Kinect skeleton, depth, and collision; sound sources project into
voice and ear; chargers project into visible color, proximity, and charging
signal.

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

`just go virtual` intentionally binds to `0.0.0.0` so a headset on the LAN can connect. This serves robot/dream-world sensor data to any device that can reach the port, and the same origin exposes Reign command endpoints. Use only on trusted networks.

## Static Viewer Assets

The viewer imports vendored Three.js r166 files through local `/static/vendor/three/...` module paths, so `/view/3d` does not need the headset browser to reach a CDN.
