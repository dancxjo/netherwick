# 002 Now Schema

`Now` is PETE's immutable, typed snapshot of its current best beliefs about
self and world. Raw and derived sensor fields may remain during migration as
evidence inputs and diagnostics, but policy consumers use `Now.world`, not
provider payloads or `extensions`. A new tick produces a new snapshot.

`Now.world` contains typed self/body beliefs, persistent entities, local
geometry, hazards, context, external authority, and a deterministic update
trace. Action-relevant beliefs carry confidence, freshness, provenance, source
kind, observation/validity time, optional coordinate frame, and explicit
contradiction references. Missing evidence remains absent rather than becoming
a false or zero-valued fact.

## ObjectSense

`Now.objects` carries first-class object observations as evidence for the
world-model updater:

```rust
ObjectObservation {
    label: String,
    class: ObjectClass,
    bearing_rad: f32,
    distance_m: Option<f32>,
    confidence: f32,
    source: ObjectObservationSource,
}
```

The simulator emits `ObjectObservationSource::Sim` observations from known world objects when they are inside the visible field of view. Kinect, captioner, and human labels should write the same `ObjectSense` shape later instead of hiding object facts in generic image vectors or prose summaries.

Use vectors for embeddings and similarity. Use `ObjectSense` to publish object
evidence. Goals act on the resulting typed entity and geometry beliefs in
`Now.world`; they do not read `ObjectSense` directly.

See [020-now-world-model.md](020-now-world-model.md) for the normative boundary
and freshness/replay policy.
