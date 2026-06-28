# Scenario Evaluation

Scenario evaluation is Netherwick's simulated exam loop. It runs deterministic scenarios, observes the runtime tick by tick, and writes a stable JSON report that says whether Pete completed the task better than a baseline policy or checkpoint configuration.

This is different from `sim-curriculum`: curriculum runs generate training ledgers and optional Worldlab captures. Scenario evaluation can also write ledgers and captures, but its main output is a report card with task metrics, model loading status, and a recommendation.

## Examples

```bash
just run sim \
  --scenario column-trap \
  --steps 300 \
  --ledger data/ledger/golden-column-trap \
  --action-selector baseline

just run eval-scenario \
  --scenario column-trap \
  --episodes 20 \
  --steps 300 \
  --out data/reports/golden-column-trap.json \
  --memory-report

just run eval-scenario \
  --scenario corner-trap \
  --episodes 20 \
  --steps 300 \
  --out data/reports/golden-corner-trap.json \
  --memory-report

just run eval-scenario \
  --scenario obstacle-avoidance \
  --episodes 20 \
  --steps 300 \
  --seed 100 \
  --out data/reports/scenario/obstacle-baseline.json

just run eval-scenario \
  --scenario charger-seeking \
  --episodes 20 \
  --steps 300 \
  --seed 200 \
  --charge-checkpoint data/models/charge_v0 \
  --charge-mode shadow-infer \
  --out data/reports/scenario/charge-shadow.json

just run eval-scenario \
  --scenario mixed-room \
  --episodes 20 \
  --steps 300 \
  --seed 300 \
  --future-checkpoint data/models/future_v0 \
  --future-mode shadow-infer \
  --experience-checkpoint data/models/experience_v0 \
  --experience-mode shadow-infer \
  --out data/reports/scenario/mixed-shadow.json
```

Use `--ledger data/ledger/eval/foo` when you also want normal `ExperienceFrame` and `ExperienceTransition` output. Use `--capture-root data/captures/eval/foo` to write one Worldlab capture per episode.

Ledger frames include the embodied prediction path. Inspect `experiences[-1].fused_vector` and `experiences[-1].predictions` to confirm that future, hazard, charge, action-value, event-change, and uncertainty predictions were attached to the embodied experience that memory will store.

## Embodied Pipeline Coverage

Use `embodied-eval` when you want a fast, deterministic check of the embodied data path without Create hardware, Kinect hardware, camera, microphone, model checkpoints, or a sim episode:

```bash
cargo run --bin netherwick -- embodied-eval \
  --fixture deterministic \
  --json
```

The deterministic fixture contains body odometry/contact, range beams, Kinect-style depth samples, visual frame bytes, ASR transcript/audio metadata, face/voice vectors, and a prior persisted experience. The command writes a coverage report and exits non-zero if a required stage is missing. For negative checks, `--omit vectors`, `--omit recall`, `--omit predictions`, and the other stage names deliberately remove a stage and should produce failures.

Example JSON shape:

```json
{
  "schema_version": 1,
  "fixture": "deterministic",
  "frame_count": 2,
  "primary_sensation_count": 15,
  "descendant_sensation_count": 8,
  "vectorized_sensation_count": 21,
  "impression_count": 23,
  "summary_impression_count": 2,
  "fused_experience_count": 2,
  "prediction_count": 4,
  "memory_link_count": 9,
  "recall_sensation_count": 1,
  "recall_impression_count": 1,
  "lineage_edge_count": 8,
  "input_modalities": ["audio", "depth", "lidar", "memory", "touch", "vision"],
  "warnings": [],
  "failures": []
}
```

Metric meanings:

- `frame_count`: frames persisted or evaluated in the deterministic replay.
- `primary_sensation_count`: direct sensor/body/memory sensations created from `Now`.
- `descendant_sensation_count`: derived sensations such as visual crops and audio transcript spans.
- `vectorized_sensation_count`: sensations carrying vector metadata with model id, dimension, purpose, and collection.
- `impression_count`: sensation-level impressions.
- `summary_impression_count`: fused experience summary impressions.
- `fused_experience_count`: embodied experiences with fused vectors.
- `prediction_count`: attached embodied or fallback future predictions.
- `memory_link_count`: links attached before persistence, such as place/person/object/recollection links.
- `recall_sensation_count` and `recall_impression_count`: memory recall materialized back into embodied sensation/impression form.
- `lineage_edge_count`: parent-to-child sensation edges preserved in the embodied context.
- `warnings` and `failures`: missing or intentionally omitted stages.

This closes the end-to-end coverage gap after #34 and #35: #34 made the embodied fused vector usable by prediction inputs, #35 made vectorizer metadata explicit and deterministic, and this check proves the whole hardware-free loop reaches memory persistence and recall.

## Golden Behavior Training

Train `danger` and `action-value` only after the golden locomotion baseline is passing. Keep checkpoints in shadow/off modes for scenario control; this step proves prediction quality, not motor authority.

```bash
just run sim \
  --scenario column-trap \
  --steps 300 \
  --ledger data/ledger/golden-column-trap \
  --action-selector baseline

just run train danger \
  --ledger data/ledger/golden-column-trap \
  --epochs 5 \
  --checkpoint data/models/danger_golden_column_v0

just run eval-scenario \
  --scenario column-trap \
  --episodes 5 \
  --steps 300 \
  --seed 1007 \
  --ledger data/ledger/golden-column-trap-heldout \
  --out data/reports/golden-column-trap-heldout.json \
  --memory-report

just run evaluate behavior danger \
  --ledger data/ledger/golden-column-trap-heldout \
  --checkpoint data/models/danger_golden_column_v0 \
  --out data/reports/danger-golden-column-heldout-eval.json

just run train action-value \
  --ledger data/ledger/golden-column-trap \
  --epochs 5 \
  --checkpoint data/models/action_value_golden_column_v0

just run evaluate behavior action-value \
  --ledger data/ledger/golden-column-trap-heldout \
  --checkpoint data/models/action_value_golden_column_v0 \
  --out data/reports/action-value-golden-column-heldout-eval.json

just run eval-scenario \
  --scenario column-trap \
  --episodes 5 \
  --steps 300 \
  --seed 2007 \
  --danger-checkpoint data/models/danger_golden_column_v0 \
  --danger-mode shadow-infer \
  --action-value-checkpoint data/models/action_value_golden_column_v0 \
  --action-value-mode shadow-infer \
  --out data/reports/golden-column-trap-shadow-models.json \
  --memory-report

just run eval-scenario \
  --scenario column-trap \
  --episodes 5 \
  --steps 300 \
  --seed 6007 \
  --action-selector model-assisted \
  --danger-checkpoint data/models/danger_golden_column_v0 \
  --danger-mode shadow-infer \
  --charge-checkpoint data/models/charge_golden_charger_v0 \
  --charge-mode shadow-infer \
  --action-value-checkpoint data/models/action_value_golden_column_v0 \
  --action-value-mode shadow-infer \
  --out data/reports/golden-column-trap-model-assisted-full-shadow.json \
  --memory-report
```

The held-out behavior reports must show `model_better_than_hardcoded: true` before checkpoints are registered as shadow candidates. The combined shadow scenario report should preserve baseline success, collision, and veto metrics. The model-assisted scenario report is only a candidate-scoring gate: it may choose among typed actions, and active `sim.stuck` recovery is injected as a scored candidate rather than a hard selector yield. Contact, cliff, wheel-drop, and critical battery still belong to the hard safety path. Do not use `danger-mode model-infer`, `charge-mode model-infer`, or `action-value-mode model-infer` for the golden loop.

Place-memory steering is part of the baseline conductor, not a learned-model privilege. A dangerous current place should turn toward `nearby_best_safe_direction_rad` when available. Low battery with known charger memory should first turn toward `nearby_best_charge_direction_rad`, then approach the charger once roughly aligned. Novel, low-danger places may inspect before default exploration.

## Current Salvage Checkpoint

As of 2026-06-25, the golden locomotion path has a measurable baseline and a guarded model-assisted shadow path:

- `data/reports/golden-column-trap.json` and `data/reports/golden-corner-trap.json` establish the hardcoded recovery baseline for the trap scenarios.
- `data/models/danger_golden_column_v0` and `data/models/action_value_golden_column_v0` have held-out behavior reports showing model predictions beat the hardcoded behavior targets.
- `data/models/charge_golden_charger_v0` has a held-out charger-seeking behavior report at `data/reports/charge-golden-charger-heldout-eval.json`; keep it in shadow mode for locomotion reports.
- `data/reports/golden-column-trap-model-assisted-full-shadow.json` and `data/reports/golden-corner-trap-model-assisted-full-shadow.json` run `danger`, `charge`, and `action-value` checkpoints in shadow with `--action-selector model-assisted`.

The latest 5-episode full-shadow column-trap report has:

```json
{
  "success_rate": 1.0,
  "mean_collisions_per_episode": 5.0,
  "recovery_attempts": 140,
  "mean_recovery_ticks": 3.8148148,
  "model_fallbacks": 0,
  "action_selector_fallbacks": 0,
  "action_selector_guard_yields": 0,
  "model_assisted_decisions": 1500
}
```

The latest 5-episode full-shadow corner-trap report has:

```json
{
  "success_rate": 1.0,
  "mean_collisions_per_episode": 0.4,
  "recovery_attempts": 150,
  "mean_recovery_ticks": 3.0896554,
  "model_fallbacks": 0,
  "action_selector_fallbacks": 0,
  "action_selector_guard_yields": 0,
  "model_assisted_decisions": 1500
}
```

Interpretation: all requested behavior checkpoints are loading and producing candidate signals. Active `sim.stuck` recovery is represented as a scored model-assisted candidate, so the selector no longer needs baseline trap-recovery guard yields in this report. Hard safety still overrides contact, cliff, wheel-drop, and critical battery conditions.

## Comparing Runs

Run a baseline first with no model checkpoints, then run the same scenario and episode count with shadow checkpoints. Compare:

- `summary.success_rate`: task-level pass rate.
- `summary.collision_rate`: collision frames divided by total frames.
- `summary.mean_distance_traveled_m`: total locomotion during the episode.
- `summary.action_histogram`: counts of Stop, Go, Reverse, TurnLeft, TurnRight, Inspect, and other action classes.
- `summary.wall_cliff_veto_count`: safety vetoes tied to cliff or wall contact conditions.
- `summary.mean_nearest_obstacle_m`: average nearest range reading.
- `summary.escape_progress_score`: progress after trap recovery minus collision/stuck penalties.
- `summary.mean_battery_delta`: useful for charger-seeking.
- `summary.mean_distance_to_charger_final_m`: whether the policy ends closer to dock.
- `summary.model_fallbacks` and `warnings`: whether requested behavior checkpoints actually ran.
- `summary.action_selector_fallbacks`: model-assisted selector ticks where at least one candidate score used fallback estimates because a model signal was missing.
- `summary.action_selector_guard_yields`: model-assisted selector ticks that deliberately yielded to a hardcoded guard. This should stay near zero in the golden loop now that trap recovery is a scored candidate.

Shadow inference never grants a model direct motor authority for safety-critical behavior. Model-controlled modes are limited to the replaceable behavior they configure, and the hardcoded safety layer still filters actions.

## Recommendations

The `recommendation` field is intentionally conservative:

- `insufficient_data`: fewer than three episodes.
- `candidate_for_more_eval`: high success with low collisions.
- `continue_training`: mixed or weak evidence.
- `reject_or_continue_training`: collision-heavy behavior.

These reports are simulation evidence, not hardware proof. They are meant to decide whether a checkpoint deserves more virtual testing or cautious real-robot read-only rehearsal, not to bypass RPi5 safety gates.
