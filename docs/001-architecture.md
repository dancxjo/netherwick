# 001 Architecture

The main flow is sensors → Features → `Now` → Sensations → teacher vectors → `ExperienceInstant` → latent encoding → imagined futures → conductor choice → autonomic safety → body actuation → next `Now` → reward/surprise → ledger → training.

The canonical moment representation is [ExperienceInstant](instant.md), not a separate Sensorium layer.

The canonical observation representation is [Feature](013-feature-registry.md): every perception, body, memory, language, and prediction subsystem should be able to emit Features before downstream clustering, binding, constellations, memory, graph storage, or prediction consumes them.
