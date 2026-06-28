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

This keeps #22 focused on the online/evolved filter machine. #25 then consumes replay data after #22: train or compare learned encoders/decoders/predictors against the evolved forge and random baselines. In that framing, #31's vectorizers are teacher/fallback scaffolding, while the learned `Experience` latent in #25 is the second-stage compression of the whole present moment.
