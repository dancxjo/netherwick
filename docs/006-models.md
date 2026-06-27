# 006 Models

Burn-backed models live in `netherwick-models`, while the rest of the workspace depends only on stable traits and schemas. The simulator should compile lightly without forcing GPU features.

Embodied vectorizers live on the stable schema side in `netherwick-experience`. They can consume model-backed vectors already surfaced by sensors (`VectorArtifact`) without pulling heavyweight runtimes into the experience crate. If those artifacts are absent or a vectorizer is disabled in `configs/models.toml`, deterministic placeholder vectorizers remain available for repeatable tests and demos.
