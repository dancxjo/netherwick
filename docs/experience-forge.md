# Experience Forge

The Experience Forge is the online feature-discovery layer for #22. It keeps a population of lightweight scalar filters over the current `Now` channels and emits a fixed 16-slot `TinyNowVector`.

The live dashboard already shows the current vector, top filters, scores, channels, and recent fired events. The forge can also be replayed and persisted from a saved ledger:

```bash
cargo run --bin netherwick -- experience-forge-replay \
  --ledger data/ledger/virtual-live \
  --checkpoint data/models/experience_forge/latest \
  --log data/reports/experience-forge/latest.jsonl \
  --report data/reports/experience-forge/latest.json \
  --json
```

Outputs:

- `checkpoint/forge.json`: durable forge state, including filters, scores, champion set, replay buffer, latest channels, and RNG state.
- `log`: optional JSONL snapshots after each replayed frame.
- `report`: replay summary with frame count, final snapshot, checkpoint path, and warnings.

The first #25 latent round-trip slice trains directly from those forge artifacts:

```bash
cargo run --bin netherwick -- latent-roundtrip \
  --forge-log data/reports/experience-forge/latest.jsonl \
  --checkpoint-out data/models/latent_roundtrip/latest \
  --report data/reports/latent-roundtrip/latest.json \
  --json
```

The top-level command defaults `--forge-checkpoint` to `data/models/experience_forge/latest`; pass it explicitly when training from another Forge run. In forge mode the replay buffer in `forge.json` is the training source because it preserves paired `TinyNowVector` values, actions, and original `Now` frames. The JSONL log is validated and counted for provenance, but it intentionally contains snapshots rather than full transition examples.

Training examples use `[TinyNowVector, action features]` as input, predict the next `TinyNowVector`, and decode a compact range/depth/contact sensor summary. The report compares copy-current, random projection, evolved Forge vector, and trained `Experience` latent candidates. It records whether the trained latent is predictive, not merely smaller, and writes separate checkpoints for the autoencoder and each future predictor baseline.

This keeps #22 focused on the online/evolved filter machine. #25 then consumes replay data after #22: train or compare learned encoders/decoders/predictors against the evolved forge and random baselines. In that framing, #31's vectorizers and `TinyNowVector` are teacher/fallback bridge signals, while the learned `Experience` latent in #25 is the second-stage compression of the whole present moment. The next handoff is #34: feed this learned latent into embodied prediction input. Later #23/#24 can use those latents as compact state for entity-SLAM and the constellation viewer.
