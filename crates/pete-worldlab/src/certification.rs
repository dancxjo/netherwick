use std::collections::{BTreeMap, BTreeSet};

use pete_events::{BrainEvent, BrainEventType, EventDisposition};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CertificationRunIdentity {
    pub run_id: String,
    pub input_sha256: String,
    pub software_artifact: String,
    pub config_artifact: String,
    pub source_identity: String,
    pub seed: Option<u64>,
}

impl CertificationRunIdentity {
    pub fn deterministic(
        input_sha256: impl Into<String>,
        software_artifact: impl Into<String>,
        config_artifact: impl Into<String>,
        source_identity: impl Into<String>,
        seed: Option<u64>,
    ) -> Self {
        let mut identity = Self {
            run_id: String::new(),
            input_sha256: input_sha256.into(),
            software_artifact: software_artifact.into(),
            config_artifact: config_artifact.into(),
            source_identity: source_identity.into(),
            seed,
        };
        identity.run_id = format!(
            "cert:{:x}",
            Sha256::digest(serde_json::to_vec(&identity).unwrap_or_default())
        );
        identity
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricAvailability {
    Available,
    Unavailable,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CertificationMetric {
    pub name: String,
    pub availability: MetricAvailability,
    pub value: Option<f64>,
    pub unit: String,
    pub confidence: f32,
    pub coverage: f32,
    pub supporting_event_ids: Vec<String>,
    #[serde(default)]
    pub missing_evidence_event_ids: Vec<String>,
    pub bundle_references: Vec<String>,
    pub unavailable_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CertificationGate {
    pub invariant: String,
    pub metric: String,
    pub passed: bool,
    pub threshold: String,
    pub observed: Option<f64>,
    pub supporting_event_ids: Vec<String>,
    pub failure_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShadowCertificationReport {
    pub schema_version: u32,
    pub identity: CertificationRunIdentity,
    pub passed: bool,
    pub metrics: BTreeMap<String, CertificationMetric>,
    pub gates: Vec<CertificationGate>,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CertificationComparison {
    pub schema_version: u32,
    pub baseline_run_id: String,
    pub candidate_run_id: String,
    pub comparable: bool,
    pub incompatibilities: Vec<String>,
    pub deltas: BTreeMap<String, Option<f64>>,
    pub regressions: Vec<CertificationGate>,
}

struct MetricDefinition {
    name: &'static str,
    unit: &'static str,
    terms: &'static [&'static str],
    event_type: Option<BrainEventType>,
}

const METRICS: &[MetricDefinition] = &[
    MetricDefinition {
        name: "safety_violations",
        unit: "events",
        terms: &["unsafe", "safety_violation"],
        event_type: None,
    },
    MetricDefinition {
        name: "veto_correctness",
        unit: "ratio",
        terms: &["veto", "safety"],
        event_type: Some(BrainEventType::GateDecision),
    },
    MetricDefinition {
        name: "task_progress_rate",
        unit: "ratio",
        terms: &["progress", "goal"],
        event_type: Some(BrainEventType::Outcome),
    },
    MetricDefinition {
        name: "collision_contact_incidence",
        unit: "events",
        terms: &["collision", "contact", "bumper"],
        event_type: None,
    },
    MetricDefinition {
        name: "trap_recovery_success",
        unit: "ratio",
        terms: &["trap", "recovery"],
        event_type: Some(BrainEventType::Outcome),
    },
    MetricDefinition {
        name: "oscillation_indecision_stopping",
        unit: "events",
        terms: &["oscillation", "indecision", "stop"],
        event_type: None,
    },
    MetricDefinition {
        name: "energy_docking_behavior",
        unit: "ratio",
        terms: &["energy", "battery", "dock", "charging"],
        event_type: None,
    },
    MetricDefinition {
        name: "command_to_motion_response",
        unit: "ratio",
        terms: &["motion.response", "dispatch_outcome"],
        event_type: Some(BrainEventType::Outcome),
    },
    MetricDefinition {
        name: "deadline_compliance",
        unit: "ratio",
        terms: &["deadline", "expired"],
        event_type: None,
    },
    MetricDefinition {
        name: "queue_loss",
        unit: "events",
        terms: &["dropped", "transport_gap", "queue.loss"],
        event_type: None,
    },
    MetricDefinition {
        name: "provenance_evidence_completeness",
        unit: "ratio",
        terms: &[],
        event_type: None,
    },
    MetricDefinition {
        name: "calibration_honesty",
        unit: "ratio",
        terms: &["calibration"],
        event_type: Some(BrainEventType::CalibrationTransition),
    },
    MetricDefinition {
        name: "trust_gate_correctness",
        unit: "ratio",
        terms: &["trust"],
        event_type: Some(BrainEventType::GateDecision),
    },
    MetricDefinition {
        name: "higher_brain_usefulness",
        unit: "ratio",
        terms: &["brain.exchange.higher_to_mother"],
        event_type: None,
    },
    MetricDefinition {
        name: "higher_brain_distraction",
        unit: "events",
        terms: &["contradict", "irrelevant", "nonsense"],
        event_type: None,
    },
    MetricDefinition {
        name: "higher_brain_latency_ms",
        unit: "milliseconds",
        terms: &["brain.exchange.higher_to_mother"],
        event_type: None,
    },
    MetricDefinition {
        name: "higher_brain_resource_cost",
        unit: "events",
        terms: &["resource", "higher_brain"],
        event_type: Some(BrainEventType::ResourceState),
    },
];

pub fn score_shadow_events(
    identity: CertificationRunIdentity,
    events: &[BrainEvent],
    bundle_references: &[String],
) -> ShadowCertificationReport {
    let mut metrics = BTreeMap::new();
    for definition in METRICS {
        let supporting = metric_events(definition, events);
        let metric = if definition.name == "provenance_evidence_completeness" {
            provenance_metric(events, bundle_references)
        } else if supporting.is_empty()
            && is_zero_observable_count(definition)
            && !events.is_empty()
        {
            zero_metric(definition, bundle_references)
        } else if supporting.is_empty() {
            unavailable_metric(definition, bundle_references)
        } else {
            available_metric(definition, supporting, events.len(), bundle_references)
        };
        metrics.insert(definition.name.to_string(), metric);
    }
    let gates = vec![
        upper_gate(&metrics, "no_safety_violation", "safety_violations", 0.0),
        upper_gate(&metrics, "no_queue_loss", "queue_loss", 0.0),
        lower_gate(
            &metrics,
            "complete_provenance",
            "provenance_evidence_completeness",
            1.0,
        ),
        lower_gate(
            &metrics,
            "motion_outcomes_observable",
            "command_to_motion_response",
            0.000_001,
        ),
    ];
    let mut report = ShadowCertificationReport {
        schema_version: 1,
        identity,
        passed: gates.iter().all(|gate| gate.passed),
        metrics,
        gates,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_hash(&report);
    report
}

fn is_zero_observable_count(definition: &MetricDefinition) -> bool {
    matches!(
        definition.name,
        "safety_violations"
            | "collision_contact_incidence"
            | "oscillation_indecision_stopping"
            | "queue_loss"
            | "higher_brain_distraction"
            | "higher_brain_resource_cost"
    )
}

fn zero_metric(definition: &MetricDefinition, bundles: &[String]) -> CertificationMetric {
    CertificationMetric {
        name: definition.name.into(),
        availability: MetricAvailability::Available,
        value: Some(0.0),
        unit: definition.unit.into(),
        confidence: 1.0,
        coverage: 1.0,
        supporting_event_ids: Vec::new(),
        missing_evidence_event_ids: Vec::new(),
        bundle_references: bundles.to_vec(),
        unavailable_reason: None,
    }
}

fn metric_events<'a>(
    definition: &MetricDefinition,
    events: &'a [BrainEvent],
) -> Vec<&'a BrainEvent> {
    events
        .iter()
        .filter(|event| {
            definition
                .event_type
                .is_none_or(|kind| event.event_type == kind)
        })
        .filter(|event| {
            definition.terms.is_empty()
                || definition
                    .terms
                    .iter()
                    .any(|term| event.kind.contains(term))
        })
        .collect()
}

fn available_metric(
    definition: &MetricDefinition,
    supporting: Vec<&BrainEvent>,
    total: usize,
    bundle_references: &[String],
) -> CertificationMetric {
    let negative = matches!(
        definition.name,
        "safety_violations"
            | "collision_contact_incidence"
            | "oscillation_indecision_stopping"
            | "queue_loss"
            | "higher_brain_distraction"
            | "higher_brain_resource_cost"
    );
    let value = if definition.name == "higher_brain_latency_ms" {
        supporting
            .iter()
            .map(|event| {
                event
                    .times
                    .observed
                    .t_ms
                    .saturating_sub(event.times.occurred.t_ms) as f64
            })
            .sum::<f64>()
            / supporting.len() as f64
    } else if definition.unit == "events" || negative {
        supporting.len() as f64
    } else {
        let accepted = supporting
            .iter()
            .filter(|event| event.disposition == EventDisposition::Accepted)
            .count();
        accepted as f64 / supporting.len() as f64
    };
    CertificationMetric {
        name: definition.name.into(),
        availability: MetricAvailability::Available,
        value: Some(value),
        unit: definition.unit.into(),
        confidence: 1.0,
        coverage: (supporting.len() as f32 / total.max(1) as f32).clamp(0.0, 1.0),
        supporting_event_ids: supporting
            .iter()
            .map(|event| event.event_id.to_string())
            .collect(),
        missing_evidence_event_ids: Vec::new(),
        bundle_references: bundle_references.to_vec(),
        unavailable_reason: None,
    }
}

fn unavailable_metric(definition: &MetricDefinition, bundles: &[String]) -> CertificationMetric {
    CertificationMetric {
        name: definition.name.into(),
        availability: MetricAvailability::Unavailable,
        value: None,
        unit: definition.unit.into(),
        confidence: 0.0,
        coverage: 0.0,
        supporting_event_ids: Vec::new(),
        missing_evidence_event_ids: Vec::new(),
        bundle_references: bundles.to_vec(),
        unavailable_reason: Some("no supporting canonical events were present".into()),
    }
}

fn provenance_metric(events: &[BrainEvent], bundles: &[String]) -> CertificationMetric {
    if events.is_empty() {
        return unavailable_metric(&METRICS[10], bundles);
    }
    let relevant = events
        .iter()
        .filter(|event| {
            matches!(
                event.event_type,
                BrainEventType::Evidence
                    | BrainEventType::Interpretation
                    | BrainEventType::BeliefUpdate
                    | BrainEventType::Proposal
                    | BrainEventType::GateDecision
                    | BrainEventType::Command
                    | BrainEventType::Outcome
                    | BrainEventType::CalibrationTransition
            )
        })
        .collect::<Vec<_>>();
    if relevant.is_empty() {
        return unavailable_metric(&METRICS[10], bundles);
    }
    let complete = relevant
        .iter()
        .filter(|event| {
            event.references.frame_id.is_some()
                && (!event.links.parents.is_empty()
                    || matches!(
                        event.event_type,
                        BrainEventType::Evidence | BrainEventType::ProviderState
                    ))
        })
        .collect::<Vec<_>>();
    let complete_ids = complete
        .iter()
        .map(|event| event.event_id.to_string())
        .collect::<BTreeSet<_>>();
    let missing_evidence_event_ids = relevant
        .iter()
        .map(|event| event.event_id.to_string())
        .filter(|event_id| !complete_ids.contains(event_id))
        .collect();
    CertificationMetric {
        name: "provenance_evidence_completeness".into(),
        availability: MetricAvailability::Available,
        value: Some(complete.len() as f64 / relevant.len() as f64),
        unit: "ratio".into(),
        confidence: 1.0,
        coverage: 1.0,
        supporting_event_ids: complete
            .iter()
            .map(|event| event.event_id.to_string())
            .collect(),
        missing_evidence_event_ids,
        bundle_references: bundles.to_vec(),
        unavailable_reason: None,
    }
}

fn upper_gate(
    metrics: &BTreeMap<String, CertificationMetric>,
    invariant: &str,
    metric: &str,
    maximum: f64,
) -> CertificationGate {
    gate(
        metrics,
        invariant,
        metric,
        format!("<= {maximum}"),
        |value| value <= maximum,
    )
}

fn lower_gate(
    metrics: &BTreeMap<String, CertificationMetric>,
    invariant: &str,
    metric: &str,
    minimum: f64,
) -> CertificationGate {
    gate(
        metrics,
        invariant,
        metric,
        format!(">= {minimum}"),
        |value| value >= minimum,
    )
}

fn gate(
    metrics: &BTreeMap<String, CertificationMetric>,
    invariant: &str,
    metric: &str,
    threshold: String,
    predicate: impl FnOnce(f64) -> bool,
) -> CertificationGate {
    let measurement = &metrics[metric];
    let passed = measurement.value.is_some_and(predicate);
    CertificationGate {
        invariant: invariant.into(),
        metric: metric.into(),
        passed,
        threshold,
        observed: measurement.value,
        supporting_event_ids: if passed || measurement.missing_evidence_event_ids.is_empty() {
            measurement.supporting_event_ids.clone()
        } else {
            measurement.missing_evidence_event_ids.clone()
        },
        failure_reason: (!passed).then(|| {
            measurement
                .unavailable_reason
                .clone()
                .unwrap_or_else(|| format!("metric {metric} violated its threshold"))
        }),
    }
}

pub fn compare_certification_reports(
    baseline: &ShadowCertificationReport,
    candidate: &ShadowCertificationReport,
) -> CertificationComparison {
    let mut incompatibilities = Vec::new();
    if baseline.identity.input_sha256 != candidate.identity.input_sha256 {
        incompatibilities.push("input artifact differs".into());
    }
    if baseline.identity.config_artifact != candidate.identity.config_artifact {
        incompatibilities.push("configuration artifact differs".into());
    }
    if baseline.identity.source_identity != candidate.identity.source_identity
        || baseline.identity.seed != candidate.identity.seed
    {
        incompatibilities.push("source identity or seed differs".into());
    }
    let comparable = incompatibilities.is_empty();
    let names = baseline
        .metrics
        .keys()
        .chain(candidate.metrics.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let deltas = names
        .into_iter()
        .map(|name| {
            let delta = baseline
                .metrics
                .get(&name)
                .and_then(|metric| metric.value)
                .zip(candidate.metrics.get(&name).and_then(|metric| metric.value))
                .map(|(baseline, candidate)| candidate - baseline);
            (name, comparable.then_some(delta).flatten())
        })
        .collect();
    let regressions = if comparable {
        candidate
            .gates
            .iter()
            .filter(|gate| !gate.passed)
            .cloned()
            .collect()
    } else {
        Vec::new()
    };
    CertificationComparison {
        schema_version: 1,
        baseline_run_id: baseline.identity.run_id.clone(),
        candidate_run_id: candidate.identity.run_id.clone(),
        comparable,
        incompatibilities,
        deltas,
        regressions,
    }
}

pub fn authorize_baseline_update(
    report: &ShadowCertificationReport,
    explicitly_reviewed: bool,
) -> Result<Vec<u8>, String> {
    if !explicitly_reviewed {
        return Err("baseline replacement requires explicit review authorization".into());
    }
    serde_json::to_vec_pretty(report).map_err(|error| error.to_string())
}

fn report_hash(report: &ShadowCertificationReport) -> String {
    let mut unhashed = report.clone();
    unhashed.report_sha256.clear();
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&unhashed).unwrap_or_default())
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use pete_events::{Brain, BrainEventId, BrainEventPayload, EventTimes, ProducerIdentity};
    use serde_json::json;

    fn event(index: u64, kind: &str, event_type: BrainEventType) -> BrainEvent {
        let mut event = BrainEvent::historical(
            BrainEventId::from_domain("cert-test", index),
            event_type,
            ProducerIdentity::new(Brain::Motherbrain, "cert.test"),
            EventTimes::observed(index, index + 2),
        );
        event.kind = kind.into();
        event.disposition = EventDisposition::Accepted;
        event.references.frame_id = Some(format!("frame-{index}"));
        if event_type != BrainEventType::Evidence {
            event.links.parents.push(pete_events::TypedEventRef::new(
                BrainEventId::from_domain("parent", index),
                BrainEventType::Evidence,
            ));
        }
        event.payload = BrainEventPayload::inline(json!({"test": true}));
        event
    }

    fn identity(input: &str) -> CertificationRunIdentity {
        CertificationRunIdentity::deterministic(input, "git:abc", "config:1", "seeded:7", Some(7))
    }

    #[test]
    fn scoring_is_deterministic_and_missing_evidence_cannot_pass() {
        let mut events = vec![
            event(1, "actuator.dispatch_outcome", BrainEventType::Outcome),
            event(2, "motion.response", BrainEventType::Outcome),
        ];
        let mut ungrounded = event(
            3,
            "interpretation.ungrounded",
            BrainEventType::Interpretation,
        );
        ungrounded.links.parents.clear();
        events.push(ungrounded);
        let left = score_shadow_events(identity("input"), &events, &["bundle://one".into()]);
        let right = score_shadow_events(identity("input"), &events, &["bundle://one".into()]);
        assert_eq!(left, right);
        assert!(!left.passed);
        assert!(left
            .gates
            .iter()
            .any(|gate| !gate.passed && gate.failure_reason.is_some()));
        assert!(left.metrics["calibration_honesty"].value.is_none());
    }

    #[test]
    fn comparison_rejects_different_inputs_and_regressions_link_events() {
        let events = vec![
            event(1, "unsafe.safety_violation", BrainEventType::Outcome),
            event(2, "motion.response", BrainEventType::Outcome),
        ];
        let baseline = score_shadow_events(identity("input-a"), &events, &[]);
        let candidate = score_shadow_events(identity("input-b"), &events, &[]);
        assert!(!compare_certification_reports(&baseline, &candidate).comparable);

        let comparable = score_shadow_events(identity("input-a"), &events, &[]);
        let comparison = compare_certification_reports(&baseline, &comparable);
        assert!(comparison.comparable);
        let regression = comparison
            .regressions
            .iter()
            .find(|gate| gate.invariant == "no_safety_violation")
            .unwrap();
        assert!(!regression.supporting_event_ids.is_empty());
    }

    #[test]
    fn baseline_updates_require_explicit_review() {
        let report = score_shadow_events(identity("input"), &[], &[]);
        assert!(authorize_baseline_update(&report, false).is_err());
        assert!(authorize_baseline_update(&report, true).is_ok());
    }
}
