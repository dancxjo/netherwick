# Motherbrain shadow flight

`just shadow-flight` runs the production Motherbrain runtime, conductor, safety
layer, memory/learning ledger, simulator cockpit, and live Observatory event
adapter without opening a network or physical actuator transport. The default
fixture is seeded and deterministic. It does not require lidar or any attached
hardware.

```bash
just shadow-flight ticks=1000 seed=7 output=data/reports/shadow-flight/latest
```

The output directory contains:

- `manifest.json`, including the exact source, production components, transport
  isolation, timing mode, artifact checksums, and any explicit substitutions;
- `input-frames.jsonl`, preserving immutable input identity, runtime frame ID,
  timestamps, clock metadata, and injected faults;
- `events.jsonl`, the same canonical `BrainEvent` records published by the live
  adapter, including simulator-authored dispatch and motion outcomes;
- `summary.json`, with causal-stage counts, safety gates, outcome counts, and
  higher-brain authority violations;
- `ledger/`, written by the normal runtime memory and learning path.

The run fails unless it observes the complete
Evidence → Interpretation → Belief → Proposal → Gate → Command → Outcome path,
at least one safety gate and simulator outcome, and zero higher-brain control
authority. A failed CLI run writes `failure.json` and still states that no
physical transport was opened.

`--higher-brain disabled` is the offline default. `--higher-brain
advisory-stub` exercises the production asynchronous higher-cognition boundary
with a deterministic, network-free advisory provider. Reports hash local gate
and command authority separately; tests require that hash to remain identical
with the provider enabled or disabled and reject any higher-brain gate or
command authority.

## Sources and time control

Direct CLI use supports deterministic fixtures, arbitrary seeded simulations,
capture directories, and JSONL ledger directories:

```bash
ulimit -s 32768
cargo run -q -p pete-tools -- shadow-flight \
  --source capture --input data/captures/example \
  --ticks 10000 --clock recorded \
  --output data/reports/shadow-flight/capture-example
```

`--clock recorded` follows input deltas, `--clock accelerated --speed N` scales
them, and `--clock step` waits for operator input before each frame. `--pause-at`
accepts comma-separated frame indexes. Runs are always bounded by `--ticks`; a
360,000-tick run covers ten simulated hours at the simulator's 100 ms cadence.

Seeded simulations accept repeatable `--fault tick:kind` injections. Supported
kinds are `battery_depleted`, `wheel_drop`, `cliff`, and `charging`. The normal
safety layer sees the resulting body evidence; the runner does not manufacture
gate decisions after the fact.

Capture replay derives a stable runtime frame UUID from capture identity and
frame index. Ledger replay retains the original ledger frame UUID exactly.
Known clock metadata is copied to input provenance; absent epochs remain absent.

The physical Pi 5 soak tracked by issue #117 is separate. Shadow flight is the
software-first validation surface and does not claim physical timing, thermal,
or hardware reliability evidence.
