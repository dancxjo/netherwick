use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    EntityId, EvidenceRef, LocalGeometrySnapshot, Now, SocialWorldSnapshot, WorldEntity,
    WorldEntityKind,
};

const OUTCOME_HISTORY_LIMIT: usize = 16;

macro_rules! epistemic_id {
    ($name:ident) => {
        #[derive(
            Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub String);
    };
}

epistemic_id!(QuestionId);
epistemic_id!(BeliefRef);
epistemic_id!(HypothesisRef);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicQuestionFamily {
    ChargerIdentityOrBearing,
    PathPassability,
    ClearanceSide,
    PersonIdentity,
    SoundDirection,
    PlaceFamiliarity,
    SkillFailureCause,
    PredictedDanger,
    #[default]
    Other,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EpistemicQuestion {
    pub question_id: QuestionId,
    pub family: EpistemicQuestionFamily,
    pub subject: BeliefRef,
    #[serde(default)]
    pub alternatives: Vec<HypothesisRef>,
    pub current_uncertainty: f32,
    pub initial_uncertainty: f32,
    pub importance: f32,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub expiry_ms: Option<u64>,
    pub attempts: u32,
    #[serde(default)]
    pub attempted_behaviors: Vec<String>,
    #[serde(default)]
    pub provenance: Vec<EvidenceRef>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EpistemicAttempt {
    pub question_id: QuestionId,
    pub behavior_id: String,
    pub started_at_ms: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicActionKind {
    OrientToBearing,
    InspectTarget,
    ScanClearance,
    Listen,
    AskPerson,
    SystematicSearch,
    ComparePrediction,
    StopAndObserve,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EpistemicAffordance {
    pub question_id: QuestionId,
    pub behavior_id: String,
    pub action_kind: EpistemicActionKind,
    pub available: bool,
    pub rejection_reason: Option<String>,
    pub target: Option<EntityId>,
    pub bearing_rad: Option<f32>,
    pub expected_information_gain: f32,
    pub expected_uncertainty_after: f32,
    pub action_cost: f32,
    pub energy_cost: f32,
    pub risk: f32,
    pub duration_ms: u64,
    pub confidence: f32,
    pub affected_belief: BeliefRef,
    #[serde(default)]
    pub required_evidence: Vec<String>,
}

impl EpistemicAffordance {
    pub fn epistemic_utility(&self) -> f32 {
        (self.expected_information_gain * self.confidence
            - 0.35 * self.action_cost
            - 0.25 * self.energy_cost
            - 0.65 * self.risk)
            .clamp(-1.0, 1.0)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EpistemicOutcome {
    pub question_id: QuestionId,
    pub observed_at_ms: u64,
    pub uncertainty_before: f32,
    pub uncertainty_after: f32,
    pub information_gain: f32,
    #[serde(default)]
    pub evidence_refs: Vec<EvidenceRef>,
    pub resolved: bool,
    pub unanswerable: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EpistemicMetrics {
    pub total_information_gain: f32,
    pub resolved_questions: u64,
    pub unresolved_questions: usize,
    pub repeated_question_count: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EpistemicSnapshot {
    pub schema_version: u32,
    pub t_ms: u64,
    #[serde(default)]
    pub active_questions: Vec<EpistemicQuestion>,
    #[serde(default)]
    pub affordances: Vec<EpistemicAffordance>,
    #[serde(default)]
    pub recent_outcomes: Vec<EpistemicOutcome>,
    pub metrics: EpistemicMetrics,
}

impl EpistemicSnapshot {
    pub fn most_important_question(&self) -> Option<&EpistemicQuestion> {
        self.active_questions.iter().max_by(|left, right| {
            (left.importance * left.current_uncertainty)
                .total_cmp(&(right.importance * right.current_uncertainty))
        })
    }

    pub fn affordances_for<'a>(
        &'a self,
        question_id: &'a QuestionId,
    ) -> impl Iterator<Item = &'a EpistemicAffordance> {
        self.affordances
            .iter()
            .filter(move |affordance| &affordance.question_id == question_id)
    }

    pub fn weighted_uncertainty(&self) -> f32 {
        self.active_questions
            .iter()
            .map(|question| question.importance * question.current_uncertainty)
            .fold(0.0f32, f32::max)
            .clamp(0.0, 1.0)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct EpistemicModelBuilder {
    active: BTreeMap<QuestionId, EpistemicQuestion>,
    recent_outcomes: Vec<EpistemicOutcome>,
    metrics: EpistemicMetrics,
    unanswerable_until_ms: BTreeMap<QuestionId, u64>,
}

impl EpistemicModelBuilder {
    pub(crate) fn update(
        &mut self,
        now: &Now,
        entities: &BTreeMap<EntityId, WorldEntity>,
        geometry: &LocalGeometrySnapshot,
        social: &SocialWorldSnapshot,
        strategy_failure_pressure: f32,
        attempt: Option<&EpistemicAttempt>,
    ) -> EpistemicSnapshot {
        self.unanswerable_until_ms
            .retain(|_, until_ms| *until_ms > now.t_ms);
        let generated =
            generate_questions(now, entities, geometry, social, strategy_failure_pressure)
                .into_iter()
                .filter(|question| {
                    !self
                        .unanswerable_until_ms
                        .contains_key(&question.question_id)
                })
                .collect::<Vec<_>>();
        let mut next = BTreeMap::new();
        for mut question in generated {
            if let Some(previous) = self.active.get(&question.question_id) {
                question.created_at_ms = previous.created_at_ms;
                question.attempts = previous.attempts;
                question.initial_uncertainty = previous.initial_uncertainty;
                question.attempted_behaviors = previous.attempted_behaviors.clone();
                if let Some(attempt) = attempt.filter(|attempt| {
                    attempt.question_id == question.question_id
                        && !question.attempted_behaviors.contains(&attempt.behavior_id)
                }) {
                    question.attempts = question.attempts.saturating_add(1);
                    question
                        .attempted_behaviors
                        .push(attempt.behavior_id.clone());
                }
                if previous.current_uncertainty > question.current_uncertainty + 0.01 {
                    self.record_outcome(EpistemicOutcome {
                        question_id: question.question_id.clone(),
                        observed_at_ms: now.t_ms,
                        uncertainty_before: previous.current_uncertainty,
                        uncertainty_after: question.current_uncertainty,
                        information_gain: (previous.current_uncertainty
                            - question.current_uncertainty)
                            .clamp(0.0, 1.0),
                        evidence_refs: question.provenance.clone(),
                        resolved: question.current_uncertainty <= 0.20,
                        unanswerable: false,
                    });
                } else if previous.updated_at_ms != now.t_ms {
                    self.metrics.repeated_question_count =
                        self.metrics.repeated_question_count.saturating_add(1);
                }
            }
            if question.attempts >= 3
                && question.current_uncertainty + 0.05 >= question.initial_uncertainty
            {
                self.unanswerable_until_ms.insert(
                    question.question_id.clone(),
                    now.t_ms.saturating_add(30_000),
                );
                self.record_outcome(EpistemicOutcome {
                    question_id: question.question_id.clone(),
                    observed_at_ms: now.t_ms,
                    uncertainty_before: question.initial_uncertainty,
                    uncertainty_after: question.current_uncertainty,
                    information_gain: (question.initial_uncertainty - question.current_uncertainty)
                        .max(0.0),
                    evidence_refs: question.provenance.clone(),
                    resolved: true,
                    unanswerable: true,
                });
                continue;
            }
            if question.current_uncertainty > 0.20 {
                next.insert(question.question_id.clone(), question);
            }
        }
        let resolved_missing = self
            .active
            .iter()
            .filter(|(question_id, _)| {
                !next.contains_key(*question_id)
                    && !self.recent_outcomes.last().is_some_and(|outcome| {
                        &outcome.question_id == *question_id && outcome.resolved
                    })
            })
            .map(|(question_id, previous)| EpistemicOutcome {
                question_id: question_id.clone(),
                observed_at_ms: now.t_ms,
                uncertainty_before: previous.current_uncertainty,
                uncertainty_after: 0.0,
                information_gain: previous.current_uncertainty,
                evidence_refs: previous.provenance.clone(),
                resolved: true,
                unanswerable: false,
            })
            .collect::<Vec<_>>();
        for outcome in resolved_missing {
            self.record_outcome(outcome);
        }
        self.active = next;
        self.metrics.unresolved_questions = self.active.len();
        let mut active_questions = self.active.values().cloned().collect::<Vec<_>>();
        active_questions.sort_by(|left, right| left.question_id.cmp(&right.question_id));
        let affordances = active_questions
            .iter()
            .flat_map(epistemic_affordances)
            .collect();
        EpistemicSnapshot {
            schema_version: 1,
            t_ms: now.t_ms,
            active_questions,
            affordances,
            recent_outcomes: self.recent_outcomes.clone(),
            metrics: self.metrics.clone(),
        }
    }

    fn record_outcome(&mut self, outcome: EpistemicOutcome) {
        self.metrics.total_information_gain += outcome.information_gain;
        if outcome.resolved {
            self.metrics.resolved_questions = self.metrics.resolved_questions.saturating_add(1);
        }
        self.recent_outcomes.push(outcome);
        if self.recent_outcomes.len() > OUTCOME_HISTORY_LIMIT {
            self.recent_outcomes
                .drain(..self.recent_outcomes.len() - OUTCOME_HISTORY_LIMIT);
        }
    }
}

fn generate_questions(
    now: &Now,
    entities: &BTreeMap<EntityId, WorldEntity>,
    geometry: &LocalGeometrySnapshot,
    social: &SocialWorldSnapshot,
    strategy_failure_pressure: f32,
) -> Vec<EpistemicQuestion> {
    let mut questions = Vec::new();
    let charger = entities
        .values()
        .filter(|entity| entity.kind == WorldEntityKind::Charger)
        .max_by(|left, right| left.confidence.total_cmp(&right.confidence));
    let charger_uncertainty = charger
        .map(|entity| {
            (1.0 - entity.confidence).max(if entity.bearing_rad.is_none() {
                0.75
            } else {
                0.0
            })
        })
        .unwrap_or(1.0);
    if charger_uncertainty > 0.20 && (now.body.battery_level < 0.80 || charger.is_some()) {
        questions.push(question(
            now.t_ms,
            EpistemicQuestionFamily::ChargerIdentityOrBearing,
            charger
                .map(|entity| format!("entity:{}", entity.id.0))
                .unwrap_or_else(|| "entity:charger:unknown".to_string()),
            charger_uncertainty,
            (1.0 - now.body.battery_level).max(0.35),
            charger
                .map(|entity| entity.provenance.clone())
                .unwrap_or_default(),
            &["charger", "not_charger", "bearing_unknown"],
        ));
    }

    let left = geometry
        .left_clearance_m
        .as_ref()
        .map(|belief| belief.value);
    let right = geometry
        .right_clearance_m
        .as_ref()
        .map(|belief| belief.value);
    let clearance_uncertainty = match (left, right) {
        (Some(left), Some(right)) if (left - right).abs() >= 0.15 => 0.15,
        (Some(_), Some(_)) => 0.65,
        _ => 0.90,
    };
    if clearance_uncertainty > 0.20 {
        questions.push(question(
            now.t_ms,
            EpistemicQuestionFamily::ClearanceSide,
            "geometry:clearance_side".to_string(),
            clearance_uncertainty,
            if now.body.velocity.forward_m_s.abs() > 0.01 {
                0.85
            } else {
                0.55
            },
            geometry
                .left_clearance_m
                .iter()
                .chain(geometry.right_clearance_m.iter())
                .flat_map(|belief| belief.meta.provenance.clone())
                .collect(),
            &["left_clearer", "right_clearer", "both_blocked"],
        ));
    }

    if let Some(person) = social
        .present_people()
        .filter(|person| person.identity_is_uncertain())
        .max_by(|left, right| {
            left.presence
                .confidence
                .total_cmp(&right.presence.confidence)
        })
    {
        let uncertainty = person
            .best_identity()
            .map(|identity| 1.0 - identity.confidence)
            .unwrap_or(1.0)
            .max(if person.identity_is_uncertain() {
                0.35
            } else {
                0.0
            });
        questions.push(question(
            now.t_ms,
            EpistemicQuestionFamily::PersonIdentity,
            format!("person:{}:identity", person.person_id.0),
            uncertainty,
            0.60,
            person.meta.provenance.clone(),
            &["known_person", "unfamiliar_person", "identity_conflict"],
        ));
    }

    if let Some(sound) = entities
        .values()
        .find(|entity| entity.kind == WorldEntityKind::SoundSource && entity.bearing_rad.is_none())
    {
        questions.push(question(
            now.t_ms,
            EpistemicQuestionFamily::SoundDirection,
            format!("entity:{}:bearing", sound.id.0),
            0.80,
            0.45,
            sound.provenance.clone(),
            &["left", "right", "diffuse"],
        ));
    }

    if strategy_failure_pressure > 0.35 {
        questions.push(question(
            now.t_ms,
            EpistemicQuestionFamily::SkillFailureCause,
            "control:last_skill_failure".to_string(),
            strategy_failure_pressure,
            0.75,
            Vec::new(),
            &["target_moved", "route_blocked", "sensing_stale"],
        ));
    }
    if now.memory.place_novelty > 0.55 && now.memory.place_familiarity < 0.45 {
        questions.push(question(
            now.t_ms,
            EpistemicQuestionFamily::PlaceFamiliarity,
            "place:current:familiarity".to_string(),
            (1.0 - now.memory.place_familiarity).clamp(0.0, 1.0),
            now.memory.place_novelty.clamp(0.0, 1.0),
            Vec::new(),
            &["familiar_place", "novel_place", "insufficient_memory"],
        ));
    }
    if let Some(prediction) = now
        .predictions
        .danger_model
        .or(now.predictions.danger_hardcoded)
        .filter(|prediction| {
            prediction.confidence < 0.65
                && prediction
                    .bump_risk
                    .max(prediction.cliff_risk)
                    .max(prediction.wheel_drop_risk)
                    .max(prediction.stuck_risk)
                    > 0.30
        })
    {
        questions.push(question(
            now.t_ms,
            EpistemicQuestionFamily::PredictedDanger,
            "prediction:danger:current".to_string(),
            1.0 - prediction.confidence,
            0.80,
            Vec::new(),
            &["danger_present", "danger_absent", "prediction_stale"],
        ));
    }
    questions
}

fn question(
    now_ms: u64,
    family: EpistemicQuestionFamily,
    subject: String,
    uncertainty: f32,
    importance: f32,
    provenance: Vec<EvidenceRef>,
    alternatives: &[&str],
) -> EpistemicQuestion {
    EpistemicQuestion {
        question_id: QuestionId(format!("question:{}:{}", family_key(family), subject)),
        family,
        subject: BeliefRef(subject),
        alternatives: alternatives
            .iter()
            .map(|alternative| HypothesisRef((*alternative).to_string()))
            .collect(),
        current_uncertainty: uncertainty.clamp(0.0, 1.0),
        initial_uncertainty: uncertainty.clamp(0.0, 1.0),
        importance: importance.clamp(0.0, 1.0),
        created_at_ms: now_ms,
        updated_at_ms: now_ms,
        expiry_ms: Some(now_ms.saturating_add(30_000)),
        attempts: 0,
        attempted_behaviors: Vec::new(),
        provenance,
    }
}

fn epistemic_affordances(question: &EpistemicQuestion) -> Vec<EpistemicAffordance> {
    let target = question
        .subject
        .0
        .strip_prefix("entity:")
        .and_then(|value| {
            value
                .split_once(":bearing")
                .map(|(id, _)| id)
                .or(Some(value))
        })
        .filter(|id| !id.ends_with(":unknown"))
        .map(|id| EntityId(id.to_string()));
    let make = |behavior: &str,
                action_kind: EpistemicActionKind,
                gain: f32,
                cost: f32,
                energy: f32,
                risk: f32,
                duration_ms: u64,
                required: &[&str]| EpistemicAffordance {
        question_id: question.question_id.clone(),
        behavior_id: behavior.to_string(),
        action_kind,
        available: true,
        target: target.clone(),
        expected_information_gain: gain.min(question.current_uncertainty),
        expected_uncertainty_after: (question.current_uncertainty - gain).max(0.0),
        action_cost: cost,
        energy_cost: energy,
        risk,
        duration_ms,
        confidence: 0.75,
        affected_belief: question.subject.clone(),
        required_evidence: required.iter().map(|value| (*value).to_string()).collect(),
        ..EpistemicAffordance::default()
    };
    let mut affordances = match question.family {
        EpistemicQuestionFamily::ChargerIdentityOrBearing => vec![
            make(
                "orient_for_charger_evidence",
                EpistemicActionKind::OrientToBearing,
                0.35,
                0.10,
                0.03,
                0.05,
                500,
                &["fresh_target_bearing"],
            ),
            make(
                "inspect_charger_hypothesis",
                EpistemicActionKind::InspectTarget,
                0.55,
                0.15,
                0.02,
                0.02,
                750,
                &["object_classification", "target_range"],
            ),
            make(
                "search_for_charger_evidence",
                EpistemicActionKind::SystematicSearch,
                0.45,
                0.35,
                0.18,
                0.10,
                2_000,
                &["charger_candidate"],
            ),
        ],
        EpistemicQuestionFamily::ClearanceSide | EpistemicQuestionFamily::PathPassability => vec![
            make(
                "scan_clearance",
                EpistemicActionKind::ScanClearance,
                0.60,
                0.18,
                0.04,
                0.04,
                900,
                &["left_range", "right_range"],
            ),
            make(
                "inspect_path",
                EpistemicActionKind::InspectTarget,
                0.40,
                0.10,
                0.01,
                0.02,
                600,
                &["center_range", "obstacle_classification"],
            ),
            make(
                "stop_and_observe_path",
                EpistemicActionKind::StopAndObserve,
                0.30,
                0.08,
                0.0,
                0.0,
                500,
                &["stable_range_frame"],
            ),
        ],
        EpistemicQuestionFamily::PersonIdentity => vec![
            make(
                "inspect_person_identity",
                EpistemicActionKind::InspectTarget,
                0.40,
                0.12,
                0.01,
                0.01,
                650,
                &["face_or_gesture_evidence"],
            ),
            make(
                "listen_for_identity",
                EpistemicActionKind::Listen,
                0.30,
                0.10,
                0.0,
                0.0,
                1_000,
                &["voice_or_self_identification"],
            ),
            make(
                "ask_identity_clarification",
                EpistemicActionKind::AskPerson,
                0.65,
                0.25,
                0.0,
                0.0,
                2_000,
                &["person_response"],
            ),
        ],
        EpistemicQuestionFamily::SoundDirection => vec![
            make(
                "listen_for_direction",
                EpistemicActionKind::Listen,
                0.45,
                0.10,
                0.0,
                0.0,
                1_000,
                &["stereo_sound_evidence"],
            ),
            make(
                "orient_for_sound_parallax",
                EpistemicActionKind::OrientToBearing,
                0.35,
                0.18,
                0.03,
                0.03,
                700,
                &["changed_sound_bearing"],
            ),
        ],
        EpistemicQuestionFamily::SkillFailureCause => vec![
            make(
                "inspect_failure_context",
                EpistemicActionKind::InspectTarget,
                0.35,
                0.10,
                0.0,
                0.0,
                700,
                &["fresh_target", "fresh_route"],
            ),
            make(
                "compare_failure_prediction",
                EpistemicActionKind::ComparePrediction,
                0.45,
                0.05,
                0.0,
                0.0,
                250,
                &["skill_outcome", "prediction_error"],
            ),
        ],
        _ => vec![make(
            "gather_targeted_evidence",
            EpistemicActionKind::InspectTarget,
            0.30,
            0.10,
            0.01,
            0.01,
            750,
            &["relevant_observation"],
        )],
    };
    for affordance in &mut affordances {
        if question
            .attempted_behaviors
            .contains(&affordance.behavior_id)
        {
            affordance.confidence *= 0.25;
            affordance.expected_information_gain *= 0.25;
            affordance.expected_uncertainty_after =
                (question.current_uncertainty - affordance.expected_information_gain).max(0.0);
        }
    }
    affordances
}

fn family_key(family: EpistemicQuestionFamily) -> &'static str {
    match family {
        EpistemicQuestionFamily::ChargerIdentityOrBearing => "charger",
        EpistemicQuestionFamily::PathPassability => "path",
        EpistemicQuestionFamily::ClearanceSide => "clearance",
        EpistemicQuestionFamily::PersonIdentity => "person_identity",
        EpistemicQuestionFamily::SoundDirection => "sound_direction",
        EpistemicQuestionFamily::PlaceFamiliarity => "place",
        EpistemicQuestionFamily::SkillFailureCause => "skill_failure",
        EpistemicQuestionFamily::PredictedDanger => "predicted_danger",
        EpistemicQuestionFamily::Other => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Belief, BeliefMeta};
    use pete_body::BodySense;

    #[test]
    fn targeted_clearance_observation_scores_above_random_motion() {
        let question = question(
            0,
            EpistemicQuestionFamily::ClearanceSide,
            "geometry:clearance_side".to_string(),
            0.9,
            0.8,
            Vec::new(),
            &["left", "right"],
        );
        let affordances = epistemic_affordances(&question);
        let scan = affordances
            .iter()
            .find(|affordance| affordance.action_kind == EpistemicActionKind::ScanClearance)
            .unwrap();
        let random_motion_utility = 0.10 - 0.35 * 0.40 - 0.25 * 0.20 - 0.65 * 0.10;
        assert!(scan.epistemic_utility() > random_motion_utility);
    }

    #[test]
    fn world_change_measures_actual_information_gain() {
        let mut builder = EpistemicModelBuilder::default();
        let mut now = Now::blank(0, BodySense::default());
        now.body.battery_level = 0.4;
        let first = builder.update(
            &now,
            &BTreeMap::new(),
            &LocalGeometrySnapshot::default(),
            &SocialWorldSnapshot::default(),
            0.0,
            None,
        );
        assert!(first
            .active_questions
            .iter()
            .any(|question| question.family == EpistemicQuestionFamily::ChargerIdentityOrBearing));

        let charger = WorldEntity {
            id: EntityId("charger:dock".to_string()),
            kind: WorldEntityKind::Charger,
            confidence: 0.95,
            bearing_rad: Some(0.1),
            meta: BeliefMeta::default(),
            ..WorldEntity::default()
        };
        now.t_ms = 100;
        let second = builder.update(
            &now,
            &BTreeMap::from([(charger.id.clone(), charger)]),
            &LocalGeometrySnapshot {
                left_clearance_m: Some(Belief {
                    value: 0.2,
                    meta: BeliefMeta::default(),
                }),
                right_clearance_m: Some(Belief {
                    value: 0.6,
                    meta: BeliefMeta::default(),
                }),
                ..LocalGeometrySnapshot::default()
            },
            &SocialWorldSnapshot::default(),
            0.0,
            None,
        );
        assert!(second
            .recent_outcomes
            .iter()
            .any(|outcome| outcome.resolved && outcome.information_gain > 0.0));
    }

    #[test]
    fn repeated_methods_without_gain_mark_question_unanswerable() {
        let mut builder = EpistemicModelBuilder::default();
        let mut now = Now::blank(0, BodySense::default());
        now.body.battery_level = 0.4;
        let mut snapshot = builder.update(
            &now,
            &BTreeMap::new(),
            &LocalGeometrySnapshot::default(),
            &SocialWorldSnapshot::default(),
            0.0,
            None,
        );
        let question_id = snapshot
            .active_questions
            .iter()
            .find(|question| question.family == EpistemicQuestionFamily::ChargerIdentityOrBearing)
            .unwrap()
            .question_id
            .clone();
        for (tick, behavior) in [
            "orient_for_charger_evidence",
            "inspect_charger_hypothesis",
            "search_for_charger_evidence",
        ]
        .into_iter()
        .enumerate()
        {
            now.t_ms = tick as u64 + 1;
            snapshot = builder.update(
                &now,
                &BTreeMap::new(),
                &LocalGeometrySnapshot::default(),
                &SocialWorldSnapshot::default(),
                0.0,
                Some(&EpistemicAttempt {
                    question_id: question_id.clone(),
                    behavior_id: behavior.to_string(),
                    started_at_ms: now.t_ms,
                }),
            );
        }
        assert!(snapshot.recent_outcomes.iter().any(|outcome| {
            outcome.question_id == question_id && outcome.resolved && outcome.unanswerable
        }));
        assert!(!snapshot
            .active_questions
            .iter()
            .any(|question| question.question_id == question_id));
    }
}
