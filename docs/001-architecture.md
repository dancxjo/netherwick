# 001 Architecture

The main flow is sensors → Features → `Now` → Sensations → teacher vectors → `ExperienceInstant` → latent encoding → imagined futures → conductor choice → autonomic safety → body actuation → next `Now` → reward/surprise → ledger → training.

The canonical moment representation is [ExperienceInstant](instant.md), not a separate Sensorium layer.

The canonical observation representation is [Feature](013-feature-registry.md): every perception, body, memory, language, and prediction subsystem should be able to emit Features before downstream clustering, binding, constellations, memory, graph storage, or prediction consumes them.

## Events and Mouth Output

Runtime events are the boundary between sensed state changes and behavior selection. `EventExtractor` turns `Now` state into typed events such as bump, charging, face recognition, Reign commands, safety vetoes, and robot initialization. Event responders may add sensations, impressions, memory notes, teaching, or proposed actions. Event-script behaviors then map selected events into `EventScriptAction` sequences.

Event scripts are replaceable behaviors, not one-off side effects. A script can emit `Say`, `Chirp`, `Song`, or motion-oriented script actions; these lower to typed `ActionPrimitive` values. Motor primitives still pass through autonomic safety. Mouth primitives pass through the robot mouth gate and are rendered by the queued Piper/CPAL mouth when available.

The current TypeScript script path uses `tsrun` as an embedded, no-Node runtime for compact event behaviors. A TypeScript script receives structured event input and returns JSON-like script actions. This is useful for expressive startup/status/mouth behaviors, but it is still a teacher implementation behind `ReplaceableBehavior`; learned models must be able to shadow it, compare against it, and eventually replace it under the normal behavior promotion rules.
