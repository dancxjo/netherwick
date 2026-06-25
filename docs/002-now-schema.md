# 002 Now Schema

`Now` stores typed, versioned sensory and internal state as structured vectors instead of a single brittle mega-vector. New sensors extend adapters and `extensions` rather than forcing architecture changes.

## ObjectSense

`Now.objects` carries first-class object observations for actionable geometry:

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

Use vectors for embeddings and similarity. Use `ObjectSense` when the robot needs geometry it can act on.
