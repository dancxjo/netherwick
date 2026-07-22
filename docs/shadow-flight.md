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
  timestamps, clock metadata, injected faults, and the exact prior outcome IDs
  consumed by the transition's memory and inline-learning path;
- `events.jsonl`, the same canonical `BrainEvent` records published by the live
  adapter, including simulator-authored dispatch and motion outcomes. Long runs
  retain a declared rolling tail while full-run counts and authority hashes are
  accumulated incrementally;
- `summary.json`, with causal-stage counts, safety gates, outcome counts, and
  higher-brain authority violations;
- `ledger/`, written by the normal runtime memory and learning path.

The default rolling replay window retains 64 production-ledger frames and
transitions, 100,000 canonical events, and 10,000 input provenance records.
`--ledger-retained-frames`, `--ledger-retained-transitions`,
`--retained-events`, and `--retained-input-frames` make those limits explicit.
The manifest reports both retained and dropped counts, so bounded soak output
never presents a truncated history as complete. Behavior-run inputs reference
their enclosing immutable ledger frame instead of embedding the same `Now`
snapshot once per candidate.

The run fails unless it observes the complete
Evidence → Interpretation → Belief → Proposal → Gate → Command → Outcome path,
at least one safety gate and simulator outcome, and zero higher-brain control
authority. A failed CLI run writes `failure.json` and still states that no
physical transport was opened.

`--higher-brain disabled` is the offline default. `--higher-brain
advisory-stub --allow-substitution higher_brain` explicitly authorizes and
records a deterministic, network-free test double that produces a substantive
interpretation and decision through the production asynchronous
higher-cognition boundary. Without that named authorization the run fails
before creating output. Reports hash local gate
and command authority separately; tests require that hash to remain identical
with the provider enabled or disabled and reject any higher-brain gate or
command authority.

Every shadow-flight run is enclosed in the cockpit crate's process-wide
physical-transport denial scope. UART, UDP, WebSocket, and HTTP cockpit paths
all check this scope before opening, binding, resolving, or connecting. This is
the enforced boundary behind the manifest's simulator-only transport claim.

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
