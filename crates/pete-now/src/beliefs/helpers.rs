fn integrate_cognitive_registry(now: &Now, services: &mut CognitiveServiceSummary) {
    let Some(value) = now.extensions.get("cognition.registry") else {
        return;
    };
    let Ok(registry) = serde_json::from_value::<ProviderRegistrySnapshot>(value.clone()) else {
        return;
    };
    for provider in registry.providers.values() {
        for capability in &provider.capabilities {
            let key = capability.capability.as_str().to_string();
            let available = matches!(
                provider.health.state,
                ProviderHealthState::Available | ProviderHealthState::Degraded
            ) && now.t_ms <= provider.health.valid_until_ms;
            let candidate = CognitiveServiceBelief {
                provider_id: Some(provider.provider_id.0.clone()),
                role: Some(provider.role.as_str().to_string()),
                capability: Some(key.clone()),
                capability_version: Some(capability.version.clone()),
                available,
                busy: false,
                confidence: (provider.health.confidence * capability.performance_confidence)
                    .clamp(0.0, 1.0),
                unavailable_reason: (!available).then(|| {
                    provider.health.reason.clone().unwrap_or_else(|| {
                        format!("provider health is {:?}", provider.health.state)
                            .to_ascii_lowercase()
                    })
                }),
                host_id: provider.host_id.as_ref().map(|id| HostId(id.0.clone())),
                process_id: provider
                    .process_id
                    .as_ref()
                    .map(|id| ProcessId(id.0.clone())),
                implementation: Some(provider.implementation.clone()),
                implementation_version: Some(provider.implementation_version.clone()),
                model_version: provider.model_version.clone(),
                locality: Some(provider.locality.as_str().to_string()),
                resource_class: Some(provider.resource_class.as_str().to_string()),
                meta: simple_meta(
                    now.t_ms,
                    BeliefSourceKind::DerivedPerception,
                    &format!("cognition.registry.{}", provider.provider_id.0),
                ),
            };
            let replace = services.services.get(&key).map_or(true, |incumbent| {
                (candidate.available && !incumbent.available)
                    || (candidate.available == incumbent.available
                        && (candidate.confidence > incumbent.confidence
                            || (candidate.confidence == incumbent.confidence
                                && candidate.provider_id < incumbent.provider_id)))
            });
            if replace {
                services.services.insert(key, candidate);
            }
        }
    }
}

fn insert_capability(
    model: &mut CapabilitySelfModel,
    id: &str,
    kind: CapabilityKind,
    available: bool,
    unavailable_reason: Option<&str>,
    now_ms: u64,
) {
    let id = CapabilityId(id.to_string());
    model.capabilities.insert(
        id.clone(),
        CapabilityBelief {
            id,
            kind,
            availability: if available {
                CapabilityAvailability::Available
            } else {
                CapabilityAvailability::Unavailable
            },
            confidence: 1.0,
            unavailable_reason: unavailable_reason.map(ToOwned::to_owned),
            authorized: available,
            meta: simple_meta(now_ms, BeliefSourceKind::Map, "self.capability.registry"),
            ..CapabilityBelief::default()
        },
    );
}

fn record_meta_evidence(trace: &mut BeliefUpdateTrace, meta: &BeliefMeta) {
    trace
        .input_evidence_ids
        .extend(meta.provenance.iter().map(|evidence| evidence.id.clone()));
    trace.input_evidence_ids.extend(
        meta.contradiction_refs
            .iter()
            .map(|evidence| evidence.id.clone()),
    );
}

fn context_beliefs(now: &Now) -> ContextBeliefs {
    let memory_present = now.memory.map_confidence > 0.0
        || now.memory.places_visited > 0
        || !now.memory.remembered_entities.is_empty();
    let predictions_present = !now.predictions.expected_events.is_empty()
        || now.predictions.danger_model.is_some()
        || now.predictions.danger_hardcoded.is_some()
        || now.predictions.charge_model.is_some()
        || now.predictions.charge_hardcoded.is_some();
    ContextBeliefs {
        novelty: memory_present.then(|| Belief {
            value: now.memory.place_novelty.clamp(0.0, 1.0),
            meta: simple_meta(now.t_ms, BeliefSourceKind::MemoryRecall, "memory.novelty"),
        }),
        surprise: Some(Belief {
            value: now.surprise.total.clamp(0.0, 1.0),
            meta: simple_meta(
                now.t_ms,
                BeliefSourceKind::DerivedPerception,
                "surprise.total",
            ),
        }),
        prediction_uncertainty: predictions_present.then(|| Belief {
            value: now.predictions.uncertainty.clamp(0.0, 1.0),
            meta: simple_meta(
                now.t_ms,
                BeliefSourceKind::LearnedPrediction,
                "prediction.uncertainty",
            ),
        }),
        map_confidence: memory_present.then(|| Belief {
            value: now.memory.map_confidence.clamp(0.0, 1.0),
            meta: simple_meta(now.t_ms, BeliefSourceKind::Map, "memory.map_confidence"),
        }),
        safe_bearing_rad: now
            .memory
            .nearby_best_safe_direction_rad
            .map(|value| Belief {
                value,
                meta: simple_meta(
                    now.t_ms,
                    BeliefSourceKind::MemoryRecall,
                    "memory.safe_bearing",
                ),
            }),
        frontier_bearing_rad: now
            .memory
            .nearby_frontier_direction_rad
            .map(|value| Belief {
                value,
                meta: simple_meta(
                    now.t_ms,
                    BeliefSourceKind::MemoryRecall,
                    "memory.frontier_bearing",
                ),
            }),
        llm_confidence: (now.llm.command_summary.is_some() || now.llm.critique.is_some()).then(
            || Belief {
                value: now.llm.confidence.clamp(0.0, 1.0),
                meta: simple_meta(now.t_ms, BeliefSourceKind::LlmClaim, "llm.confidence"),
            },
        ),
        expected_battery_delta: now
            .predictions
            .charge_model
            .or(now.predictions.charge_hardcoded)
            .map(|prediction| Belief {
                value: prediction.expected_battery_delta,
                meta: simple_meta(
                    now.t_ms,
                    BeliefSourceKind::LearnedPrediction,
                    "prediction.expected_battery_delta",
                ),
            }),
    }
}

fn local_geometry(now: &Now) -> LocalGeometrySnapshot {
    let range_belief = |key: &str, value: f32| Belief {
        value,
        meta: simple_meta(
            now.t_ms,
            BeliefSourceKind::DirectObservation,
            &format!("range.{key}"),
        ),
    };
    let (left, center, right) = clearance_buckets(&now.range.beams);
    LocalGeometrySnapshot {
        nearest_m: now
            .range
            .nearest_m
            .map(|value| range_belief("nearest", value)),
        left_clearance_m: left.map(|value| range_belief("left_clearance", value)),
        center_clearance_m: center.map(|value| range_belief("center_clearance", value)),
        right_clearance_m: right.map(|value| range_belief("right_clearance", value)),
    }
}

fn clearance_buckets(beams: &[f32]) -> (Option<f32>, Option<f32>, Option<f32>) {
    if beams.is_empty() {
        return (None, None, None);
    }
    let third = (beams.len() / 3).max(1);
    let left_end = third.min(beams.len());
    let right_start = beams.len().saturating_sub(third);
    let center_start = left_end.saturating_sub(1).min(beams.len());
    let center_end = (right_start + 1).min(beams.len()).max(center_start + 1);
    let nearest = |slice: &[f32]| slice.iter().copied().reduce(f32::min);
    (
        nearest(&beams[..left_end]),
        nearest(&beams[center_start..center_end]),
        nearest(&beams[right_start..]),
    )
}

fn authority_belief(now: &Now) -> Option<AuthorityBelief> {
    now.reign.latest.clone().map(|input| AuthorityBelief {
        meta: simple_meta(now.t_ms, BeliefSourceKind::HumanClaim, "authority.reign"),
        input,
    })
}

#[derive(Clone, Copy)]
struct FreshnessPolicy {
    current_for_ms: u64,
    aging_after_ms: u64,
    invalidate_after_ms: u64,
}

fn identity_policy(kind: &WorldEntityKind) -> FreshnessPolicy {
    match kind {
        WorldEntityKind::Person | WorldEntityKind::SoundSource => FreshnessPolicy {
            current_for_ms: 1_000,
            aging_after_ms: 5_000,
            invalidate_after_ms: 15_000,
        },
        _ => FreshnessPolicy {
            current_for_ms: 2_000,
            aging_after_ms: 15_000,
            invalidate_after_ms: 60_000,
        },
    }
}

fn bearing_policy(_kind: &WorldEntityKind) -> FreshnessPolicy {
    FreshnessPolicy {
        current_for_ms: 500,
        aging_after_ms: 2_000,
        invalidate_after_ms: 3_000,
    }
}

fn distance_policy(_kind: &WorldEntityKind) -> FreshnessPolicy {
    FreshnessPolicy {
        current_for_ms: 500,
        aging_after_ms: 2_000,
        invalidate_after_ms: 3_000,
    }
}

fn freshness(age_ms: u64, policy: FreshnessPolicy) -> Freshness {
    if age_ms <= policy.current_for_ms {
        Freshness::Current
    } else if age_ms <= policy.aging_after_ms {
        Freshness::Aging
    } else if age_ms <= policy.invalidate_after_ms {
        Freshness::Stale
    } else {
        Freshness::Invalidated
    }
}

fn decayed_confidence(base: f32, age_ms: u64, policy: FreshnessPolicy) -> f32 {
    if age_ms <= policy.current_for_ms {
        base.clamp(0.0, 1.0)
    } else {
        let span = policy
            .invalidate_after_ms
            .saturating_sub(policy.current_for_ms)
            .max(1);
        let elapsed = age_ms.saturating_sub(policy.current_for_ms);
        (base * (1.0 - elapsed as f32 / span as f32)).clamp(0.0, 1.0)
    }
}

fn hazard_beliefs(now: &Now) -> HazardBeliefs {
    let contact = now.body.flags.bump_left
        || now.body.flags.bump_right
        || now.body.flags.wall
        || now.body.flags.wheel_drop;
    let range_risk = now
        .range
        .nearest_m
        .map(|distance| ((0.35 - distance) / 0.35).clamp(0.0, 1.0));
    let immediate = if contact { Some(1.0) } else { range_risk };
    let predicted = now
        .predictions
        .danger_model
        .or(now.predictions.danger_hardcoded)
        .map(|prediction| {
            prediction
                .bump_risk
                .max(prediction.cliff_risk)
                .max(prediction.wheel_drop_risk)
                .max(prediction.stuck_risk)
        });
    HazardBeliefs {
        immediate_risk: immediate.map(|value| Belief {
            value,
            meta: simple_meta(now.t_ms, BeliefSourceKind::DirectObservation, "range/body"),
        }),
        remembered_risk: (now.memory.map_confidence > 0.0).then(|| Belief {
            value: now.memory.place_danger.clamp(0.0, 1.0),
            meta: simple_meta(
                now.t_ms,
                BeliefSourceKind::MemoryRecall,
                "memory.place_danger",
            ),
        }),
        predicted_risk: predicted.map(|value| Belief {
            value,
            meta: simple_meta(
                now.t_ms,
                BeliefSourceKind::LearnedPrediction,
                "prediction.danger",
            ),
        }),
    }
}

fn temporal_beliefs(now: &Now) -> Vec<TemporalBelief> {
    let vectors = now
        .eye
        .image_vectors
        .iter()
        .chain(now.eye.image_description_vectors.iter())
        .chain(now.eye.scene_vectors.iter())
        .chain(now.face.vectors.iter())
        .chain(now.voice.vectors.iter())
        .chain(now.objects.vectors.iter())
        .chain(now.ear.transcript_vectors.iter());
    let mut beliefs = Vec::new();
    for vector in vectors {
        let Some(occurred_at_ms) = vector.occurred_at_ms else {
            continue;
        };
        let evidence = evidence_ref(
            &format!("vector.{}", vector.collection),
            &vector.point_id,
            now.t_ms,
            "temporal-evidence-v1",
        );
        let subject = format!("vector:{}:{}", vector.collection, vector.point_id);
        beliefs.push(TemporalBelief {
            interval: TimeInterval {
                domain: ClockDomain::Event,
                start_ms: occurred_at_ms,
                end_ms: Some(occurred_at_ms),
                uncertainty_ms: 0,
            },
            relation: TemporalRelation::OccurredDuring,
            subject: subject.clone(),
            confidence: 1.0,
            provenance: vec![evidence.clone()],
        });
        beliefs.push(TemporalBelief {
            interval: TimeInterval {
                domain: ClockDomain::Observation,
                start_ms: now.t_ms,
                end_ms: Some(now.t_ms),
                uncertainty_ms: 0,
            },
            relation: if occurred_at_ms < now.t_ms {
                TemporalRelation::After
            } else {
                TemporalRelation::Overlaps
            },
            subject,
            confidence: 1.0,
            provenance: vec![evidence],
        });
    }
    beliefs
}

fn simple_meta(now_ms: u64, source_kind: BeliefSourceKind, key: &str) -> BeliefMeta {
    let evidence = evidence_ref(key, key, now_ms, "world-model-v1");
    belief_meta(1.0, now_ms, source_kind, evidence, None)
}

fn belief_meta(
    confidence: f32,
    now_ms: u64,
    source_kind: BeliefSourceKind,
    evidence: EvidenceRef,
    coordinate_frame: Option<FrameId>,
) -> BeliefMeta {
    BeliefMeta {
        confidence: confidence.clamp(0.0, 1.0),
        observed_at_ms: now_ms,
        valid_at_ms: now_ms,
        freshness: Freshness::Current,
        provenance: vec![evidence],
        contradiction_refs: Vec::new(),
        coordinate_frame,
        source_kind,
    }
}

fn evidence_ref(source: &str, key: &str, now_ms: u64, implementation: &str) -> EvidenceRef {
    EvidenceRef {
        id: format!("{source}:{key}:{now_ms}"),
        source: source.to_string(),
        key: key.to_string(),
        observed_at_ms: now_ms,
        transformation_lineage: vec![implementation.to_string()],
        implementation_version: Some("1".to_string()),
    }
}

fn entity_kind_key(kind: &WorldEntityKind) -> &'static str {
    match kind {
        WorldEntityKind::Charger => "charger",
        WorldEntityKind::Person => "person",
        WorldEntityKind::Obstacle => "obstacle",
        WorldEntityKind::SoundSource => "sound_source",
        WorldEntityKind::Landmark => "landmark",
        WorldEntityKind::Door => "door",
        WorldEntityKind::Region => "region",
        WorldEntityKind::Unknown => "unknown",
    }
}

fn normalized_label(label: &str) -> String {
    let normalized = label
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    if normalized.is_empty() {
        "unlabeled".to_string()
    } else {
        normalized
    }
}

fn normalize_angle(mut angle: f32) -> f32 {
    while angle > std::f32::consts::PI {
        angle -= std::f32::consts::TAU;
    }
    while angle < -std::f32::consts::PI {
        angle += std::f32::consts::TAU;
    }
    angle
}
