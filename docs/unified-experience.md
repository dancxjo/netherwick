# Unified Experience Autoencoder

#37 trains Pete's replay-first unified `Experience` autoencoder over mechanically assembled multimodal teacher vectors:

```text
raw sensation -> modality / teacher encoder -> teacher vector
teacher vectors + masks -> UnifiedExperienceInstant
UnifiedExperienceInstant -> Experience encoder -> ExperienceLatent
ExperienceLatent -> decode / predict / compare
```

Teacher vectors are scaffolding, not Pete's learned state. The trainer assembles fixed slots for scene, face, voice, transcript, depth/range, and memory. Missing slots are recorded in an explicit mask and in report coverage instead of silently disappearing.

Run it from existing replay ledgers:

```bash
cargo run --bin netherwick -- train unified-experience \
  --ledger data/ledger/sim1 \
  --checkpoint data/models/unified_experience/latest \
  --report data/reports/unified-experience/latest.json \
  --epochs 1 \
  --z-dim 16 \
  --teacher-dim 16
```

The JSON report includes example counts, per-slot modality coverage, latent dimension, per-head reconstruction losses, next-latent prediction loss, combined surprise, confidence, copy-current/random/mechanical-Instant research baselines, warnings for insufficient data or missing slots, and a verdict stating whether the learned latent is reconstructive and predictive.
