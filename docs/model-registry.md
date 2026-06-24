# Model Registry

The model registry is Netherwick's file-backed card catalog for learned behavior checkpoints. It records which checkpoint exists, which behavior it belongs to, what ledger and reports produced it, what scenario metrics it has earned, and which runtime modes it may enter.

The default registry lives at:

```bash
data/models/registry.json
```

It is deliberately just JSON. There is no database, no hidden promotion state, and no automatic checkpoint deletion.

## Registering A Model

Register after training and evaluation artifacts exist:

```bash
cargo run --bin netherwick -- model-register \
  --behavior danger \
  --checkpoint data/models/danger_obstacle_v0 \
  --training-ledger data/ledger/curriculum/obstacle-v0 \
  --behavior-report data/reports/behavior/danger-obstacle-v0.json \
  --scenario-report data/reports/scenario/obstacle-shadow-v0.json \
  --name danger_obstacle_v0
```

If the registry is missing, the CLI creates it. If the same behavior and model name already exist, pass `--overwrite` to replace that card. Registration records warnings for missing evidence so the card is visible, but promotion gates still decide whether it may shadow or infer.

## Statuses

- `registered`: the checkpoint is known to the catalog.
- `shadow`: the model may run beside hardcoded behavior and record predictions, but hardcoded behavior still decides.
- `inference`: the model may be used on approved inference surfaces after passing stricter gates.
- `retired`: the card is kept for history but should not be selected for new runs.
- `rejected`: the checkpoint failed review or evaluation.

Check the catalog with:

```bash
cargo run --bin netherwick -- model-status
```

## Promotion Gates

Promotion is conservative by default. `shadow` requires a checkpoint and scenario evidence. `inference` requires enough scenario episodes, low or zero fallbacks, acceptable collision rate, and baseline comparison evidence for safety-critical behaviors.

Danger, action-value, future, and experience are safety-critical because they can influence motor choices or core state used by action selection. Danger inference is refused unless the operator explicitly passes:

```bash
--allow-safety-critical-inference
```

This flag does not bypass metrics. It only says the human reviewer understands this is a safety-critical promotion.

## Comparing Scenarios

Compare a baseline report with a candidate report:

```bash
cargo run --bin netherwick -- compare-scenario-reports \
  --baseline data/reports/scenario/obstacle-baseline.json \
  --candidate data/reports/scenario/obstacle-shadow.json
```

The output is `improved`, `regressed`, or `inconclusive`, plus deltas for success rate, collision rate, battery delta, and fallback count. Promotion gates reuse this comparison when both reports are supplied.

## Example Workflow

1. Generate curriculum data:

```bash
cargo run --bin netherwick -- sim-curriculum \
  --scenario obstacle-avoidance \
  --episodes 50 \
  --steps 300 \
  --out data/ledger/curriculum/obstacle-v0
```

2. Train a behavior checkpoint:

```bash
cargo run --bin netherwick -- train behavior danger \
  --ledger data/ledger/curriculum/obstacle-v0 \
  --checkpoint data/models/danger_obstacle_v0
```

3. Evaluate behavior loss and write a behavior report:

```bash
cargo run --bin netherwick -- evaluate behavior danger \
  --ledger data/ledger/curriculum/obstacle-v0 \
  --checkpoint data/models/danger_obstacle_v0 \
  --out data/reports/behavior/danger-obstacle-v0.json
```

4. Run baseline and shadow scenario exams:

```bash
cargo run --bin netherwick -- eval-scenario \
  --scenario obstacle-avoidance \
  --episodes 20 \
  --steps 300 \
  --out data/reports/scenario/obstacle-baseline-v0.json

cargo run --bin netherwick -- eval-scenario \
  --scenario obstacle-avoidance \
  --episodes 20 \
  --steps 300 \
  --danger-checkpoint data/models/danger_obstacle_v0 \
  --danger-mode shadow-infer \
  --out data/reports/scenario/obstacle-shadow-v0.json
```

5. Register and promote only as far as the evidence supports:

```bash
cargo run --bin netherwick -- model-register \
  --behavior danger \
  --checkpoint data/models/danger_obstacle_v0 \
  --training-ledger data/ledger/curriculum/obstacle-v0 \
  --behavior-report data/reports/behavior/danger-obstacle-v0.json \
  --scenario-report data/reports/scenario/obstacle-shadow-v0.json \
  --name danger_obstacle_v0

cargo run --bin netherwick -- model-promote \
  --behavior danger \
  --name danger_obstacle_v0 \
  --target shadow
```

Shadow means Pete can listen to the learned mind without letting it steer. Inference means a reviewed model may affect its approved runtime surface. For safety-critical behavior, that second step needs much stronger evidence.
