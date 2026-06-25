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
  --seed 3007 \
  --action-selector model-assisted \
  --danger-checkpoint data/models/danger_golden_column_v0 \
  --danger-mode shadow-infer \
  --action-value-checkpoint data/models/action_value_golden_column_v0 \
  --action-value-mode shadow-infer \
  --out data/reports/golden-column-trap-model-assisted.json \
  --memory-report
```

The held-out behavior reports must show `model_better_than_hardcoded: true` before checkpoints are registered as shadow candidates. The combined shadow scenario report should preserve baseline success, collision, and veto metrics. The model-assisted scenario report is only a candidate-scoring gate: it may choose among typed actions, but close-range trap recovery must still yield to the baseline conductor. Do not use `danger-mode model-infer` or `action-value-mode model-infer` for the golden loop.

Place-memory steering is part of the baseline conductor, not a learned-model privilege. A dangerous current place should turn toward `nearby_best_safe_direction_rad` when available. Low battery with known charger memory should first turn toward `nearby_best_charge_direction_rad`, then approach the charger once roughly aligned. Novel, low-danger places may inspect before default exploration.

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
- `summary.model_fallbacks` and `warnings`: whether requested checkpoints actually ran.

Shadow inference never grants a model direct motor authority for safety-critical behavior. Model-controlled modes are limited to the replaceable behavior they configure, and the hardcoded safety layer still filters actions.

## Recommendations

The `recommendation` field is intentionally conservative:

- `insufficient_data`: fewer than three episodes.
- `candidate_for_more_eval`: high success with low collisions.
- `continue_training`: mixed or weak evidence.
- `reject_or_continue_training`: collision-heavy behavior.

These reports are simulation evidence, not hardware proof. They are meant to decide whether a checkpoint deserves more virtual testing or cautious real-robot read-only rehearsal, not to bypass RPi5 safety gates.
