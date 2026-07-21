use std::collections::{BTreeMap, BTreeSet};
use std::future;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use pete_autonomic::SimpleSafety;
use pete_body::BodySense;
use pete_conductor::SimpleConductor;
use pete_experience::{EmbodiedContext, ExperienceLatent, FuturePrediction, Impression};
use pete_ledger::{ExperienceFrame, ExperienceTransition, LedgerWriter};
use pete_llm::{Combobulation, LlmAgent, LlmReviewRequest, LlmScientificReview, LlmTickResult};
use pete_memory::InMemoryExperienceStore;
use pete_now::{
    EpistemicAttempt, EpistemicQuestionFamily, InteractionPhase, Now, WorldModelUpdateContext,
    WorldModelUpdater,
};
use pete_sensors::World;
use pete_sim::{ArenaConfig, SimObject, SimObjectKind, VirtualWorld};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{MinimalRuntime, SleepPhase, SleepSnapshot, WakeReason};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SocialExamCaseReport {
    pub case: String,
    pub passed: bool,
    #[serde(default)]
    pub observations: BTreeMap<String, Value>,
    #[serde(default)]
    pub failures: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SocialExamReport {
    pub schema_version: u32,
    pub passed: bool,
    pub cases: Vec<SocialExamCaseReport>,
}

/// Run the deterministic social-world acceptance exam against real simulator
/// projections, the canonical world-model updater, and the production runtime.
pub async fn run_social_exam() -> Result<SocialExamReport> {
    let cases = vec![
        arrival_departure_case().await?,
        identity_conflict_case().await?,
        conversation_case().await?,
        sleep_interruption_case().await?,
        epistemic_strategy_case().await?,
        cognition_delay_case().await?,
        forebrain_failure_case().await?,
    ];
    Ok(SocialExamReport {
        schema_version: 1,
        passed: cases.iter().all(|case| case.passed),
        cases,
    })
}

async fn arrival_departure_case() -> Result<SocialExamCaseReport> {
    let (mut world, person) = social_world("Alex", "Hello, Pete.");
    let mut updater = WorldModelUpdater::default();
    let arrived = updater.update(
        simulator_now(&mut world, 100).await?,
        WorldModelUpdateContext::default(),
    );
    let first_interaction = arrived
        .world
        .social
        .active_interaction
        .as_ref()
        .map(|interaction| interaction.interaction_id.0.clone());

    world.set_objects(Vec::new());
    let departed = updater.update(
        simulator_now(&mut world, 1_200).await?,
        WorldModelUpdateContext::default(),
    );
    world.set_objects(vec![person]);
    let returned = updater.update(
        simulator_now(&mut world, 1_300).await?,
        WorldModelUpdateContext::default(),
    );
    let returned_interaction = returned
        .world
        .social
        .active_interaction
        .as_ref()
        .map(|interaction| interaction.interaction_id.0.clone());

    let observations = BTreeMap::from([
        ("arrival_interaction".to_string(), json!(first_interaction)),
        (
            "departure_closed_interactions".to_string(),
            json!(departed.world.social.recent_interactions.len()),
        ),
        (
            "return_interaction".to_string(),
            json!(returned_interaction),
        ),
    ]);
    let mut failures = Vec::new();
    check(
        first_interaction.is_some(),
        "arrival did not open an interaction",
        &mut failures,
    );
    check(
        departed.world.social.active_interaction.is_none()
            && departed
                .world
                .social
                .recent_interactions
                .last()
                .is_some_and(|interaction| interaction.phase == InteractionPhase::Ended),
        "departure did not close the active interaction",
        &mut failures,
    );
    check(
        returned_interaction.is_some() && returned_interaction != first_interaction,
        "return did not create a new encounter identity",
        &mut failures,
    );
    Ok(case_report("arrival_departure", observations, failures))
}

async fn identity_conflict_case() -> Result<SocialExamCaseReport> {
    let (mut world, _) = social_world("Alex", "My name is Bob.");
    let mut updater = WorldModelUpdater::default();
    let now = updater.update(
        simulator_now(&mut world, 100).await?,
        WorldModelUpdateContext::default(),
    );
    let person = now.world.social.people.values().next();
    let contradictory_hypotheses = person
        .map(|person| {
            person
                .identity_hypotheses
                .iter()
                .filter(|hypothesis| !hypothesis.contradiction_refs.is_empty())
                .count()
        })
        .unwrap_or_default();
    let identity_question = now
        .world
        .epistemic
        .active_questions
        .iter()
        .any(|question| question.family == EpistemicQuestionFamily::PersonIdentity);
    let observations = BTreeMap::from([
        (
            "identity_uncertain".to_string(),
            json!(person.is_some_and(|person| person.identity_is_uncertain())),
        ),
        (
            "contradictory_hypotheses".to_string(),
            json!(contradictory_hypotheses),
        ),
        ("identity_question".to_string(), json!(identity_question)),
    ]);
    let mut failures = Vec::new();
    check(
        person.is_some_and(|person| person.identity_is_uncertain()),
        "conflicting face and self-identification evidence was collapsed",
        &mut failures,
    );
    check(
        contradictory_hypotheses > 0,
        "identity conflict lacks explicit contradiction provenance",
        &mut failures,
    );
    check(
        identity_question,
        "identity conflict did not create an epistemic question",
        &mut failures,
    );
    Ok(case_report("identity_conflict", observations, failures))
}

async fn conversation_case() -> Result<SocialExamCaseReport> {
    let (mut world, _) = social_world("Alex", "Could you help me, Pete?");
    let mut updater = WorldModelUpdater::default();
    let now = updater.update(
        simulator_now(&mut world, 100).await?,
        WorldModelUpdateContext::default(),
    );
    let interaction = now.world.social.active_interaction.as_ref();
    let conversation_episode = now
        .world
        .temporal
        .active_episode(pete_now::EpisodeKind::Conversation);
    let pending_turns = interaction
        .map(|interaction| interaction.pending_turns.len())
        .unwrap_or_default();
    let unresolved_requests = interaction
        .map(|interaction| interaction.unresolved_requests.len())
        .unwrap_or_default();
    let observations = BTreeMap::from([
        (
            "phase".to_string(),
            json!(interaction.map(|interaction| interaction.phase)),
        ),
        ("pending_turns".to_string(), json!(pending_turns)),
        (
            "unresolved_requests".to_string(),
            json!(unresolved_requests),
        ),
        (
            "conversation_episode".to_string(),
            json!(conversation_episode.map(|episode| &episode.episode_id.0)),
        ),
    ]);
    let mut failures = Vec::new();
    check(
        interaction
            .is_some_and(|interaction| interaction.phase == InteractionPhase::AwaitingResponse),
        "addressed speech did not put the interaction into awaiting-response state",
        &mut failures,
    );
    check(
        pending_turns == 1 && unresolved_requests == 1,
        "conversation turn or attributed request was not retained",
        &mut failures,
    );
    check(
        conversation_episode.is_some(),
        "social interaction did not open a temporal conversation episode",
        &mut failures,
    );
    Ok(case_report("conversation", observations, failures))
}

async fn sleep_interruption_case() -> Result<SocialExamCaseReport> {
    let memory = InMemoryExperienceStore::new();
    let mut runtime = MinimalRuntime::new(
        NoopLedger,
        memory.clone(),
        memory,
        SimpleConductor::default(),
        SimpleSafety::default(),
        pete_llm::NoopLlmAgent,
    );
    let mut sleep_now = idle_now(100);
    sleep_now.body.charging = true;
    sleep_now
        .extensions
        .insert("sleep.request".to_string(), Value::Bool(true));
    let started = runtime
        .tick(sleep_now, ExperienceLatent::default(), Vec::new())
        .await?;
    let started_sleep: SleepSnapshot = serde_json::from_value(
        started
            .frame
            .now
            .extensions
            .get("sleep")
            .cloned()
            .ok_or_else(|| anyhow!("sleep snapshot missing after request"))?,
    )?;

    let mut cue_now = idle_now(200);
    cue_now.body.charging = true;
    cue_now
        .extensions
        .insert("sleep.important_social_cue".to_string(), Value::Bool(true));
    let interrupted = runtime
        .tick(cue_now, ExperienceLatent::default(), Vec::new())
        .await?;
    let interrupted_sleep: SleepSnapshot = serde_json::from_value(
        interrupted
            .frame
            .now
            .extensions
            .get("sleep")
            .cloned()
            .ok_or_else(|| anyhow!("sleep snapshot missing after social cue"))?,
    )?;
    let wake_reason = interrupted_sleep
        .session
        .as_ref()
        .and_then(|session| session.interrupted_by.clone());
    let observations = BTreeMap::from([
        ("started_phase".to_string(), json!(started_sleep.phase)),
        (
            "interrupted_phase".to_string(),
            json!(interrupted_sleep.phase),
        ),
        ("wake_reason".to_string(), json!(wake_reason)),
    ]);
    let mut failures = Vec::new();
    check(
        started_sleep.phase == SleepPhase::Preparing,
        "operator sleep request did not start a sleep session",
        &mut failures,
    );
    check(
        interrupted_sleep.phase == SleepPhase::Interrupted
            && wake_reason == Some(WakeReason::ImportantSocialCue),
        "important social cue did not interrupt sleep with the social wake reason",
        &mut failures,
    );
    Ok(case_report("sleep_interruption", observations, failures))
}

async fn epistemic_strategy_case() -> Result<SocialExamCaseReport> {
    let (mut world, _) = social_world("Alex", "My name is Bob.");
    let mut updater = WorldModelUpdater::default();
    let mut now = updater.update(
        simulator_now(&mut world, 100).await?,
        WorldModelUpdateContext::default(),
    );
    let question_id = now
        .world
        .epistemic
        .active_questions
        .iter()
        .find(|question| question.family == EpistemicQuestionFamily::PersonIdentity)
        .map(|question| question.question_id.clone())
        .ok_or_else(|| anyhow!("identity-conflict fixture did not create a question"))?;
    let mut strategies = Vec::new();
    for step in 0..3u64 {
        let strategy = now
            .world
            .epistemic
            .affordances_for(&question_id)
            .max_by(|left, right| {
                left.epistemic_utility()
                    .total_cmp(&right.epistemic_utility())
            })
            .map(|affordance| affordance.behavior_id.clone())
            .ok_or_else(|| anyhow!("identity question has no information-gathering strategy"))?;
        strategies.push(strategy.clone());
        now = updater.update(
            simulator_now(&mut world, 200 + step * 100).await?,
            WorldModelUpdateContext {
                epistemic_attempt: Some(EpistemicAttempt {
                    question_id: question_id.clone(),
                    behavior_id: strategy,
                    started_at_ms: 200 + step * 100,
                }),
                ..WorldModelUpdateContext::default()
            },
        );
    }
    let unique_strategies = strategies.iter().collect::<BTreeSet<_>>().len();
    let unanswerable = now.world.epistemic.recent_outcomes.iter().any(|outcome| {
        outcome.question_id == question_id && outcome.resolved && outcome.unanswerable
    });
    let observations = BTreeMap::from([
        ("strategies".to_string(), json!(strategies)),
        (
            "unique_strategy_count".to_string(),
            json!(unique_strategies),
        ),
        ("unanswerable".to_string(), json!(unanswerable)),
    ]);
    let mut failures = Vec::new();
    check(
        unique_strategies == 3,
        "failed inquiry did not change to each remaining identity strategy",
        &mut failures,
    );
    check(
        unanswerable,
        "three no-gain strategies did not suspend the sterile identity question",
        &mut failures,
    );
    Ok(case_report(
        "epistemic_strategy_changes",
        observations,
        failures,
    ))
}

async fn cognition_delay_case() -> Result<SocialExamCaseReport> {
    let memory = InMemoryExperienceStore::new();
    let mut runtime = MinimalRuntime::new(
        NoopLedger,
        memory.clone(),
        memory,
        SimpleConductor::default(),
        SimpleSafety::default(),
        PendingCognitionAgent,
    );
    let first = runtime
        .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
        .await?;
    let second = runtime
        .tick(idle_now(200), ExperienceLatent::default(), Vec::new())
        .await?;
    let service = &second.frame.now.world.self_model.service_state.services["rich_language"];
    let observations = BTreeMap::from([
        ("available".to_string(), json!(service.available)),
        ("busy".to_string(), json!(service.busy)),
        (
            "local_ticks_completed".to_string(),
            json!([
                first.frame.now.world.revision,
                second.frame.now.world.revision
            ]),
        ),
    ]);
    let mut failures = Vec::new();
    check(
        service.available && service.busy,
        "delayed cognition was reported as an outage instead of healthy occupancy",
        &mut failures,
    );
    check(
        first.frame.now.world.revision == 1 && second.frame.now.world.revision == 2,
        "local cognition stopped advancing while the forebrain was delayed",
        &mut failures,
    );
    runtime.cancel_cognition();
    Ok(case_report("cognition_delay", observations, failures))
}

async fn forebrain_failure_case() -> Result<SocialExamCaseReport> {
    let memory = InMemoryExperienceStore::new();
    let mut runtime = MinimalRuntime::new(
        NoopLedger,
        memory.clone(),
        memory,
        SimpleConductor::default(),
        SimpleSafety::default(),
        FailingCognitionAgent,
    );
    runtime
        .tick(idle_now(100), ExperienceLatent::default(), Vec::new())
        .await?;
    for _ in 0..16 {
        if runtime
            .cognition
            .pending
            .as_ref()
            .is_some_and(|pending| pending.task.is_finished())
        {
            break;
        }
        tokio::task::yield_now().await;
    }
    let failure_tick = runtime
        .tick(idle_now(200), ExperienceLatent::default(), Vec::new())
        .await?;
    let service = &failure_tick
        .frame
        .now
        .world
        .self_model
        .service_state
        .services["rich_language"];
    let reason = service.unavailable_reason.clone();
    let observations = BTreeMap::from([
        ("available".to_string(), json!(service.available)),
        ("busy".to_string(), json!(service.busy)),
        ("unavailable_reason".to_string(), json!(reason.clone())),
        (
            "local_world_revision".to_string(),
            json!(failure_tick.frame.now.world.revision),
        ),
    ]);
    let mut failures = Vec::new();
    check(
        !service.available
            && !service.busy
            && reason
                .as_deref()
                .is_some_and(|reason: &str| reason.contains("simulated forebrain outage")),
        "forebrain failure did not become explicit service unavailability",
        &mut failures,
    );
    check(
        failure_tick.frame.now.world.revision == 2,
        "local world-model processing did not survive forebrain failure",
        &mut failures,
    );
    Ok(case_report("forebrain_failure", observations, failures))
}

fn social_world(name: &str, spoken_text: &str) -> (VirtualWorld, SimObject) {
    let (mut world, _) = VirtualWorld::new_with_cockpit(
        7,
        ArenaConfig {
            width_m: 8.0,
            height_m: 8.0,
        },
    );
    let mut body = BodySense::default();
    body.battery_level = 1.0;
    body.odometry.x_m = 1.0;
    body.odometry.y_m = 4.0;
    body.odometry.heading_rad = 0.0;
    world.set_body(body);
    let person = SimObject {
        id: "social-exam-person".to_string(),
        label: name.to_string(),
        kind: SimObjectKind::Person {
            identity: Some(name.to_string()),
        },
        x_m: 2.0,
        y_m: 4.0,
        radius_m: 0.22,
        color_rgb: [220, 180, 140],
        emits_sound: true,
        spoken_text: Some(spoken_text.to_string()),
        charge_rate: 0.0,
    };
    world.set_objects(vec![person.clone()]);
    (world, person)
}

async fn simulator_now(world: &mut VirtualWorld, t_ms: u64) -> Result<Now> {
    let mut body = world.body();
    body.last_update_ms = t_ms;
    world.set_body(body);
    Ok(world.snapshot().await?.to_now(t_ms))
}

fn idle_now(t_ms: u64) -> Now {
    let mut body = BodySense::default();
    body.last_update_ms = t_ms;
    body.battery_level = 1.0;
    Now::blank(t_ms, body)
}

fn check(condition: bool, message: &str, failures: &mut Vec<String>) {
    if !condition {
        failures.push(message.to_string());
    }
}

fn case_report(
    case: &str,
    observations: BTreeMap<String, Value>,
    failures: Vec<String>,
) -> SocialExamCaseReport {
    SocialExamCaseReport {
        case: case.to_string(),
        passed: failures.is_empty(),
        observations,
        failures,
    }
}

#[derive(Clone, Debug, Default)]
struct NoopLedger;

#[async_trait]
impl LedgerWriter for NoopLedger {
    async fn append(&self, _frame: &ExperienceFrame) -> Result<()> {
        Ok(())
    }

    async fn append_transition(&self, _transition: &ExperienceTransition) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug)]
struct PendingCognitionAgent;

#[async_trait]
impl LlmAgent for PendingCognitionAgent {
    async fn combobulate(
        &mut self,
        _now: &Now,
        _impressions: &[Impression],
        _embodied: Option<&EmbodiedContext>,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
    ) -> Result<Option<Combobulation>> {
        future::pending().await
    }

    async fn maybe_tick(
        &mut self,
        _now: &Now,
        _embodied: Option<&EmbodiedContext>,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
        _awareness_summary: Option<&str>,
    ) -> Result<LlmTickResult> {
        Ok(LlmTickResult::default())
    }

    async fn scientific_review(
        &mut self,
        _request: &LlmReviewRequest,
    ) -> Result<Option<LlmScientificReview>> {
        Ok(None)
    }
}

#[derive(Debug)]
struct FailingCognitionAgent;

#[async_trait]
impl LlmAgent for FailingCognitionAgent {
    async fn combobulate(
        &mut self,
        _now: &Now,
        _impressions: &[Impression],
        _embodied: Option<&EmbodiedContext>,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
    ) -> Result<Option<Combobulation>> {
        Err(anyhow!("simulated forebrain outage"))
    }

    async fn maybe_tick(
        &mut self,
        _now: &Now,
        _embodied: Option<&EmbodiedContext>,
        _z: &ExperienceLatent,
        _futures: &[FuturePrediction],
        _recall_summary: &str,
        _awareness_summary: Option<&str>,
    ) -> Result<LlmTickResult> {
        Ok(LlmTickResult::default())
    }

    async fn scientific_review(
        &mut self,
        _request: &LlmReviewRequest,
    ) -> Result<Option<LlmScientificReview>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_social_exam_passes_all_cases() {
        std::thread::Builder::new()
            .name("social-exam-test".to_string())
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
                    .block_on(async {
                        let report = run_social_exam().await.unwrap();
                        assert!(
                            report.passed,
                            "failed cases: {:?}",
                            report
                                .cases
                                .iter()
                                .filter(|case| !case.passed)
                                .collect::<Vec<_>>()
                        );
                        assert_eq!(report.cases.len(), 7);
                    });
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
