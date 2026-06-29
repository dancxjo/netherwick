# Latent Thought Evaluation

This document turns Netherwick's `ExperienceLatent` into something measurable instead of merely suggestive.

Netherwick already compresses `Now` into `ExperienceLatent`, then uses it for future prediction, action choice, and consequence learning. The goal of this evaluation track is to test whether those latent states behave like useful intermediate thought representations.

## Target representation

Primary target:

- `netherwick_experience::ExperienceLatent.z`

Primary input/output path:

- `Now` -> `ExperienceEncoder` / `LatentEncoder` -> `ExperienceLatent`
- `ExperienceLatent` + `ActionPrimitive` -> `FuturePrediction`
- `ExperienceLatent` -> `NowReconstruction` when a decoder is available
- scenario ledger frames/transitions -> repeated deterministic samples

Secondary probes may also inspect:

- `ExperienceFrame.tiny_now_vector`
- `ExperienceFrame.filter_outputs`
- fused embodied vectors persisted into memory
- prediction outputs attached to embodied experiences

## Evaluation axes

### 1. Causality

Question: does the latent actually matter for behavior or prediction?

Probe:

1. Encode a ledger transition into `z`.
2. Run downstream prediction/action scoring with the original `z`.
3. Perturb selected dimensions, zero selected dimensions, or swap in a matched-control `z` from another frame.
4. Re-run the same downstream path.
5. Report how much the output changes.

Metrics:

- `mean_prediction_delta_l2`
- `mean_action_value_delta`
- `action_rank_flip_rate`
- `hazard_prediction_flip_rate`
- `causal_effect_score`

Acceptance sketch:

- Perturbing high-salience latent components should change relevant predictions more than perturbing shuffled/control components.
- Fully zeroing or swapping `z` should degrade prediction/action consistency.
- If output is unchanged, the latent is decorative wiring.

### 2. Minimality

Question: does `z` contain compact task-relevant state, or is it just a padded copy of the input?

Probe:

1. Train or fit lightweight probes from `z` to known labels, such as bump, stuck, blocked-forward, free-motion, novelty, intervention, action-changed-scene, and stable-scene.
2. Fit matched probes from raw/teacher vectors and `tiny_now_vector`.
3. Measure whether `z` predicts target labels without preserving irrelevant nuisance details.
4. Estimate redundancy by ablation, sparsity, effective rank, and mutual-nearest-neighbor overlap with raw input vectors.

Metrics:

- `label_probe_auc` or `label_probe_accuracy`
- `nuisance_probe_accuracy`
- `effective_rank`
- `mean_abs_correlation_to_input`
- `compression_ratio`
- `minimality_score`

Acceptance sketch:

- `z` should keep outcome-relevant information.
- `z` should not trivially preserve every raw input channel.
- Smaller or ablated `z` should fail gracefully rather than collapse unpredictably.

### 3. Separability

Question: do different situations produce distinguishable latent states?

Probe:

1. Collect balanced frames from deterministic scenarios.
2. Group by scenario, outcome label, action class, and place/object condition.
3. Compare within-group and between-group distances.
4. Run kNN or simple linear probes from `z` to labels.

Metrics:

- `between_within_distance_ratio`
- `knn_label_accuracy`
- `linear_probe_accuracy`
- `silhouette_by_scenario`
- `silhouette_by_outcome`
- `collapse_rate`

Acceptance sketch:

- Obstacle, charger, empty-room, trap, and recovery states should not all map to the same fog-bank vector.
- Within-task distinctions matter: two different obstacle bearings should be more separable than prompt-level category labels alone.

### 4. Stability

Question: do irrelevant or small input changes preserve nearby latent states?

Probe:

1. Generate paired frames from deterministic scenario seeds.
2. Apply controlled perturbations that should not change the semantic state much:
   - small sensor noise
   - transcript paraphrase or casing changes
   - small odometry jitter
   - harmless memory recall ordering changes
3. Re-encode and compare latent distance.
4. Compare against semantically meaningful changes, such as obstacle appearing, contact, charger becoming visible, or action class changes.

Metrics:

- `same_state_distance_mean`
- `changed_state_distance_mean`
- `stability_margin`
- `nearest_neighbor_preservation_rate`
- `semantic_change_sensitivity`

Acceptance sketch:

- Small nuisance changes should keep `z` nearby.
- Meaningful world changes should move `z` more.

## Report shape

The first implementation should emit a stable JSON file:

```json
{
  "schema_version": 1,
  "ledger": "data/ledger/eval/foo",
  "encoder": "experience_v0",
  "sample_count": 0,
  "latent_dim": 0,
  "summary": {
    "causality_score": 0.0,
    "minimality_score": 0.0,
    "separability_score": 0.0,
    "stability_score": 0.0,
    "overall_recommendation": "insufficient_data"
  },
  "causality": {},
  "minimality": {},
  "separability": {},
  "stability": {},
  "warnings": [],
  "failures": []
}
```

Recommendation values should be conservative:

- `insufficient_data`
- `instrumentation_missing`
- `latent_decorative`
- `continue_training`
- `candidate_for_more_eval`
- `candidate_for_shadow_use`

## Proposed CLI

```bash
just latent-axiom-smoke

cargo run -p netherwick-tools -- latent-axioms \
  --ledger data/ledger/eval/obstacle-smoke \
  --out data/reports/latent/obstacle-smoke.json \
  --encoder experience_v0 \
  --seed 7
```

Suggested smoke recipe:

```bash
just run sim --scenario obstacle-avoidance --steps 120 --seed 7001 --ledger data/ledger/latent/obstacle-smoke
just run sim --scenario charger-seeking --steps 120 --seed 7002 --ledger data/ledger/latent/charger-smoke
just run latent-axioms --ledger data/ledger/latent/obstacle-smoke --out data/reports/latent/obstacle-smoke.json
just run latent-axioms --ledger data/ledger/latent/charger-smoke --out data/reports/latent/charger-smoke.json
```

## Implementation checklist

1. Add `LatentAxiomReport` structs in the tools or training crate.
2. Add ledger loading for `ExperienceFrame` / `ExperienceTransition` samples.
3. Add deterministic perturbation helpers for `ExperienceLatent.z` and controlled `Now` variants.
4. Implement distance helpers: L2, cosine distance, effective rank approximation, nearest-neighbor preservation.
5. Implement causality first using available downstream prediction/action-value paths.
6. Implement separability second because scenario labels and outcome labels already exist in `ExperienceOutcomeLabels`.
7. Implement stability with deterministic paired fixtures.
8. Implement minimality last, initially with cheap linear/kNN probes before adding heavier learned probes.
9. Add report comparison support so latent reports can be gated like scenario reports.

## Success condition

A checkpoint is not promoted because it "feels smarter." It is promoted only when the latent report shows that `z` causally affects downstream outputs, separates task-relevant states, stays stable under nuisance changes, and preserves less irrelevant input detail than the raw sensor/state bundle.
