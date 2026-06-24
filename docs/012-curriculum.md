# Simulation Curricula

`sim-curriculum` creates repeatable families of virtual worlds for training Pete before real hardware is available. Each episode is deterministic from the base `--seed` plus the episode index, so a curriculum can be regenerated and compared across model changes.

Examples:

```bash
just run sim-curriculum --scenario obstacle-avoidance --episodes 50 --steps 300 --out data/ledger/curriculum/obstacle-v0
just run sim-curriculum --scenario charger-seeking --episodes 50 --steps 300 --out data/ledger/curriculum/charge-v0
just run sim-curriculum --scenario person-speaker-room --episodes 50 --steps 300 --out data/ledger/curriculum/social-v0
```

The output path is a normal `JsonlLedger` root. It also gets a `manifest.json` that records the
scenario slug, base seed, per-episode seed, train/validation/test split, object counts, spawn pose,
tick count, and optional capture path. Existing training commands can consume the ledger directly:

```bash
just run train behavior danger --ledger data/ledger/curriculum/obstacle-v0
just run train behavior charge --ledger data/ledger/curriculum/charge-v0
just run train behavior eye-next --ledger data/ledger/curriculum/social-v0
just run train behavior ear-next --ledger data/ledger/curriculum/social-v0
```

Use `--capture-root` to also write one Worldlab capture per episode:

```bash
just run sim-curriculum \
  --scenario mixed-room \
  --episodes 5 \
  --steps 100 \
  --out data/ledger/curriculum/mixed-smoke \
  --capture-root data/captures/curriculum/mixed-smoke
```

Those captures can be replayed through the normal worldlab path, while the ledger remains the training fuel.

By default, `sim-curriculum` assigns the first 80% of episodes to `train`, the next 10% to
`validation`, and the remaining 10% to `test`. Tune that with `--validation-ratio` and
`--test-ratio` when you need a different split.
