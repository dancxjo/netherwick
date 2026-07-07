# Counterfactual Replay

Worldlab passive replay feeds recorded `WorldSnapshot` frames back through the runtime exactly as captured. It answers “what happened?” Counterfactual replay reconstructs a small simulator world from capture metadata, applies controlled edits, runs forward with a selected policy, and writes a fresh ledger and/or JSON report. It answers first-step “what if?” questions.

For a hardware-free embodied pipeline gate, use `cargo run --bin pete -- embodied-eval --fixture deterministic --json` instead. That command does not reconstruct or edit a world; it proves the sensor-to-sensation-to-vector-to-experience-to-prediction-to-memory-recall path is intact using a tiny deterministic fixture. In issue terms, it builds on #34's embodied prediction input path and #35's vector metadata/vectorizer work, then adds the #36 persistence and recall coverage check.

## Required Capture Metadata

Counterfactual replay currently requires sim capture manifests with a `scenario` object containing:

- scenario kind and seed
- arena dimensions
- initial robot body pose and battery state
- sim objects with stable ids, positions, radii, and kinds
- capture `tick_ms`

New `capture-sim`, `sim-curriculum --capture-root`, and `eval-scenario --capture-root` captures include this metadata. Older passive captures, real robot captures, and captures without the `scenario` field fail clearly:

```text
passive captures without reconstructable sim metadata cannot yet be counterfactually replayed
```

## Edits

Supported v0 edits:

```text
move-charger:x=1.0,y=1.0
move-person:x=-1.0,y=0.5
move-speaker:x=-1.0,y=0.5
remove-obstacle:id=obstacle-0
add-obstacle:x=2.0,y=3.0,radius=0.3
set-battery:value=0.5
```

`move-charger`, `move-person`, `move-speaker`, and `remove-obstacle` accept `id=...`. If no id is provided, replay uses the first matching object and records a warning in the report.

## Policies

Supported v0 policies:

- `baseline`: normal conductor/runtime behavior
- `stop`: no movement
- `turn-left-on-danger`
- `turn-right-on-danger`
- `seek-charge`
- `random-walk` or `random-walk:seed=7`
- scripted action list with `--actions forward,forward,left,forward`

These are probe policies, not learned planning. They exist to create alternative ledgers and compare simple outcomes.

## Examples

```bash
cargo run --bin pete -- sim-curriculum \
  --scenario charger-seeking \
  --episodes 1 \
  --steps 100 \
  --seed 900 \
  --out data/ledger/curriculum/cf-charge-source \
  --capture-root data/captures/counterfactual/source

cargo run --bin pete -- replay-counterfactual \
  --capture data/captures/counterfactual/source/episode-000 \
  --edit move-charger:x=1.0,y=1.0 \
  --policy seek-charge \
  --out-ledger data/ledger/counterfactual/charge-moved \
  --out-report data/reports/counterfactual/charge-moved.json
```

The report has schema version 1 and summarizes collisions, charging ticks, battery delta, distance traveled, final charger distance, and warnings.

## Limitations

This is not a full physical simulator or game engine. It reconstructs only the simple `VirtualWorld` state available in sim metadata, uses the existing runtime tick loop, and applies basic motor effects. It does not infer missing real-world geometry, use point clouds, command hardware, or fabricate world state from passive captures.
