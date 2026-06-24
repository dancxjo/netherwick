# Scenario Evaluation

Scenario evaluation is Netherwick's simulated exam loop. It runs deterministic scenarios, observes the runtime tick by tick, and writes a stable JSON report that says whether Pete completed the task better than a baseline policy or checkpoint configuration.

This is different from `sim-curriculum`: curriculum runs generate training ledgers and optional Worldlab captures. Scenario evaluation can also write ledgers and captures, but its main output is a report card with task metrics, model loading status, and a recommendation.

## Examples

```bash
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

## Comparing Runs

Run a baseline first with no model checkpoints, then run the same scenario and episode count with shadow checkpoints. Compare:

- `summary.success_rate`: task-level pass rate.
- `summary.collision_rate`: collision frames divided by total frames.
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
