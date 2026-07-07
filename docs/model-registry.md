# Model Registry

The model registry is Pete's file-backed card catalog for learned behavior checkpoints. It records which checkpoint exists, which behavior it belongs to, what ledger and reports produced it, what scenario metrics it has earned, and which runtime modes it may enter.

Embodied sensation vectorizers are configured beside behavior models in `configs/models.toml` under `[vectorizer.*]`. They are deliberately lightweight selectors: when an upstream sensor has already produced a model-backed `VectorArtifact`, the embodied pipeline preserves that vector, model id, dimension, source sensation id, timestamp, modality, payload kind, collection, and purpose. Face artifacts use `face_identity`/`faces`; voice artifacts use `voice_identity`/`voices`. When no upstream vector is present, configured feature vectorizers produce bounded deterministic embeddings for image frames/crops, audio/voice windows, transcript spans, and depth scenes. Setting `enabled = false` for a vectorizer, or pointing `model_path` at a missing local asset, keeps the deterministic placeholder fallback available and labels the emitted vector as `is_fallback = true`.

Every embodied vector now records:

- `vectorizer_id`: the Pete registry entry that produced or preserved the vector.
- `model_id` and `model_label`: the configured model identity or upstream sensor model label.
- `dim`: the emitted vector width.
- `source_kind` and `source_sensation_id`: what local object the vector represents.
- `purpose`: why this vector should be searched, such as `visual_similarity`, `scene_similarity`, `face_identity`, `transcript_semantic`, `voice_identity`, or `experience_semantic`.
- `collection`: the vector collection or logical search lane for that purpose.
- `input_summary`: a small audit string, never the raw frame/audio payload.
- `provenance`: whether the vector came from an upstream artifact, Pete feature vectorizer, summary/impression vectorizer, or placeholder fallback.

The implementation follows Daringsby's image/face vector work (`psyche/src/sensors/image_vector.rs`, `psyche/src/sensors/face.rs`) and scene vector loop (`pete/src/bin/scene_vec.rs`) for model labels, graceful vectorization failure, and duplicate or near-identical visual frame suppression. Listenbury contributes the lightweight voice-signature and text embedding seams used as baseline vocabulary. Mortar-Sea is the semantic contract: vectors are purpose-specific, and every vector declares what it represents, which model produced it, which collection it belongs to, what purpose it serves, and what input was vectorized.

#31 is considered closed when image, face, voice, and text vectorization are represented in the embodied pipeline while placeholders remain deterministic and available. Pete currently registers deterministic feature-backed vectorizers for image, crop, audio, transcript, and depth payloads; preserves real model-backed upstream face, voice, and ASR transcript `VectorArtifact`s when sensors provide them; and emits deterministic semantic hash vectors for transcript, impression, and experience-summary text. The command-backed ASR tool attaches transcript vectors to `EarSense.transcript_vectors`, mirroring how processed image frames carry image-description and scene vectors. CLIP/OpenCLIP, ArcFace runtime loading, SpeakerNet, LLM embeddings, and other neural model-backed vectorizers remain follow-up work.

The default registry lives at:

```bash
data/models/registry.json
```

It is deliberately just JSON. There is no database, no hidden promotion state, and no automatic checkpoint deletion.

## Registering A Model

Register after training and evaluation artifacts exist:

```bash
cargo run --bin pete -- model-register \
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
cargo run --bin pete -- model-status
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
cargo run --bin pete -- compare-scenario-reports \
  --baseline data/reports/scenario/obstacle-baseline.json \
  --candidate data/reports/scenario/obstacle-shadow.json
```

The output is `improved`, `regressed`, or `inconclusive`, plus deltas for success rate, collision rate, battery delta, and fallback count. Promotion gates reuse this comparison when both reports are supplied.

## Example Workflow

1. Generate curriculum data:

```bash
cargo run --bin pete -- sim-curriculum \
  --scenario obstacle-avoidance \
  --episodes 50 \
  --steps 300 \
  --out data/ledger/curriculum/obstacle-v0
```

2. Train a behavior checkpoint:

```bash
cargo run --bin pete -- train behavior danger \
  --ledger data/ledger/curriculum/obstacle-v0 \
  --checkpoint data/models/danger_obstacle_v0
```

3. Evaluate behavior loss and write a behavior report:

```bash
cargo run --bin pete -- evaluate behavior danger \
  --ledger data/ledger/curriculum/obstacle-v0 \
  --checkpoint data/models/danger_obstacle_v0 \
  --out data/reports/behavior/danger-obstacle-v0.json
```

4. Run baseline and shadow scenario exams:

```bash
cargo run --bin pete -- eval-scenario \
  --scenario obstacle-avoidance \
  --episodes 20 \
  --steps 300 \
  --out data/reports/scenario/obstacle-baseline-v0.json

cargo run --bin pete -- eval-scenario \
  --scenario obstacle-avoidance \
  --episodes 20 \
  --steps 300 \
  --danger-checkpoint data/models/danger_obstacle_v0 \
  --danger-mode shadow-infer \
  --out data/reports/scenario/obstacle-shadow-v0.json
```

5. Register and promote only as far as the evidence supports:

```bash
cargo run --bin pete -- model-register \
  --behavior danger \
  --checkpoint data/models/danger_obstacle_v0 \
  --training-ledger data/ledger/curriculum/obstacle-v0 \
  --behavior-report data/reports/behavior/danger-obstacle-v0.json \
  --scenario-report data/reports/scenario/obstacle-shadow-v0.json \
  --name danger_obstacle_v0

cargo run --bin pete -- model-promote \
  --behavior danger \
  --name danger_obstacle_v0 \
  --target shadow
```

Shadow means Pete can listen to the learned mind without letting it steer. Inference means a reviewed model may affect its approved runtime surface. For safety-critical behavior, that second step needs much stronger evidence.
