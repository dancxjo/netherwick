use anyhow::Result;
use async_trait::async_trait;
use netherwick_actions::ActionPrimitive;
use netherwick_autonomic::SafetyLayer;
use netherwick_behaviors::{BehaviorRun, Replaceable};
use netherwick_body::{BodySense, MotorCommand};
use netherwick_conductor::{Conductor, ConductorInput};
use netherwick_core::{Provenance, Reward, TimeMs};
use netherwick_events::{
    EventBus, EventContext, EventExtractor, Response, default_event_bus, safety_veto_event,
};
use netherwick_experience::{
    Experience, ExperienceLatent, FuturePrediction, Impression, Sensation,
};
use netherwick_ledger::{ExperienceFrame, LedgerWriter};
use netherwick_llm::{ConsciousCommand, LlmAgent, LlmTeaching, LlmTickResult};
use netherwick_memory::{MemoryStore, Recall, RecallBundle, RecallQuery};
use netherwick_now::{DriveSense, MemorySense, Now, SafetySense};
use uuid::Uuid;

pub const CORE_LOOP: &[&str] = &[
    "Now_t",
    "encode to latent z_t",
    "predict futures",
    "recall relevant experiences",
    "extract events",
    "dispatch responders",
    "choose action",
    "safety-filter",
    "act",
    "observe Now_t+1",
    "compute surprise/reward",
    "write ledger",
    "train",
];

pub const CADENCES: &[(&str, &str)] = &[
    ("fast", "sensors/safety/motors"),
    ("medium", "Now/prediction/conductor/events"),
    ("slow", "LLM/reflection/memory summaries"),
    ("idle", "replay/dream/model comparison"),
];

#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeTick {
    pub now: Now,
    pub latent: ExperienceLatent,
    pub predicted_futures: Vec<FuturePrediction>,
    pub recall: RecallBundle,
    pub experience: Experience,
    pub chosen_action: ActionPrimitive,
    pub safety: SafetySense,
    pub conscious_command: Option<ConsciousCommand>,
}

pub struct MinimalRuntime<L, M, R, C, S, A> {
    pub ledger: L,
    pub memory_store: M,
    pub memory_recall: R,
    pub conductor: C,
    pub safety: S,
    pub llm: A,
    pub event_extractor: EventExtractor,
    pub event_bus: EventBus,
}

impl<L, M, R, C, S, A> MinimalRuntime<L, M, R, C, S, A>
where
    L: LedgerWriter,
    M: MemoryStore,
    R: Recall,
    C: Conductor,
    S: SafetyLayer,
    A: LlmAgent,
{
    pub async fn tick(
        &mut self,
        now: Now,
        latent: ExperienceLatent,
        predicted_futures: Vec<FuturePrediction>,
    ) -> Result<RuntimeTick> {
        let recall_query = RecallQuery::from_now(&now);
        let recall = self.memory_recall.recall(recall_query).await?;
        let llm_tick = self
            .llm
            .maybe_tick(&now, &latent, &predicted_futures, &recall.first_person_summary)
            .await?;
        let ctx = EventContext {
            now: &now,
            latent: Some(&latent),
            recall: Some(&recall),
            predicted_futures: &predicted_futures,
        };

        let base_events = self.event_extractor.events_from_now(&now, Some(&recall));
        let dispatch = self.event_bus.dispatch_all(&ctx, base_events)?;
        let mut dispatched_events = dispatch.events;
        let mut aggregate = ResponseAggregate::from_responses(dispatch.responses);
        let action = choose_action(
            &mut self.conductor,
            &now,
            &latent,
            &llm_tick,
            &recall,
            &aggregate.proposed_actions,
            &aggregate.memory_override,
            &aggregate.drive_updates,
        )?;
        let safety_decision = self.safety.filter(&now, action_to_motor(&action));
        let safety = SafetySense {
            schema_version: 1,
            vetoed: safety_decision.vetoed,
            reasons: safety_decision
                .reason
                .iter()
                .map(|reason| format!("{reason:?}"))
                .collect(),
        };

        if safety_decision.vetoed {
            let veto_event = safety_veto_event(&now, &action, safety_decision.reason.clone());
            let veto_dispatch = self.event_bus.dispatch_all(&ctx, [veto_event])?;
            dispatched_events.extend(veto_dispatch.events);
            aggregate.merge(ResponseAggregate::from_responses(veto_dispatch.responses));
        }

        let mut sensations = synthesize_sensations(&now, &recall);
        sensations.extend(aggregate.sensations.clone());
        let mut impressions = synthesize_impressions(&sensations, now.t_ms);
        impressions.extend(aggregate.impressions.clone());
        let mut experiences = aggregate.experiences.clone();
        let experience = synthesize_experience(&sensations, &impressions, now.t_ms);
        experiences.push(experience.clone());

        let reward = Reward {
            value: if safety_decision.vetoed { -0.1 } else { 0.0 },
        };
        let mut notes = vec![recall.first_person_summary.clone()];
        notes.extend(aggregate.notes.clone());
        notes.extend(
            dispatched_events
                .iter()
                .filter_map(|event| event.summary.clone()),
        );
        notes.extend(
            aggregate
                .teaching
                .iter()
                .flat_map(|teaching| teaching.memory_notes.clone()),
        );

        let frame = ExperienceFrame {
            id: Uuid::new_v4(),
            t_ms: now.t_ms,
            now: now.clone(),
            sensations,
            impressions,
            experiences,
            z: Some(latent.clone()),
            chosen_action: Some(action.clone()),
            conscious_command: llm_tick.conscious_command.clone(),
            predicted_futures: predicted_futures.clone(),
            actual_next: None,
            reward,
            surprise: now.surprise.clone(),
            memory_recall: recall.hits.clone(),
            recollections: recall.recollections.clone(),
            llm_teaching: aggregate.merged_teaching(&llm_tick.teaching),
            counterfactuals: llm_tick
                .teaching
                .iter()
                .chain(aggregate.teaching.iter())
                .flat_map(|teaching| teaching.counterfactuals.clone())
                .collect(),
            notes,
        };
        self.ledger.append(&frame).await?;
        self.memory_store.store(&frame).await?;

        Ok(RuntimeTick {
            now,
            latent,
            predicted_futures,
            recall,
            experience,
            chosen_action: action,
            safety,
            conscious_command: llm_tick.conscious_command,
        })
    }
}

#[derive(Default)]
struct ResponseAggregate {
    sensations: Vec<Sensation>,
    impressions: Vec<Impression>,
    experiences: Vec<Experience>,
    proposed_actions: Vec<ActionPrimitive>,
    drive_updates: Vec<(String, f32)>,
    memory_override: Option<MemorySense>,
    teaching: Vec<LlmTeaching>,
    notes: Vec<String>,
}

impl ResponseAggregate {
    fn from_responses(responses: Vec<Response>) -> Self {
        let mut out = Self::default();
        for response in responses {
            match response {
                Response::None | Response::Emit(_) => {}
                Response::AddSensation(sensation) => out.sensations.push(sensation),
                Response::AddImpression(impression) => out.impressions.push(impression),
                Response::AddExperience(experience) => out.experiences.push(experience),
                Response::ProposeAction(action) => out.proposed_actions.push(action),
                Response::SetDrive { name, value } => out.drive_updates.push((name, value)),
                Response::SetMemorySense(memory) => out.memory_override = Some(memory),
                Response::Teach(teaching) => out.teaching.push(teaching),
                Response::RememberNote(note) => out.notes.push(note),
            }
        }
        out
    }

    fn merge(&mut self, mut other: Self) {
        self.sensations.append(&mut other.sensations);
        self.impressions.append(&mut other.impressions);
        self.experiences.append(&mut other.experiences);
        self.proposed_actions.append(&mut other.proposed_actions);
        self.drive_updates.append(&mut other.drive_updates);
        if other.memory_override.is_some() {
            self.memory_override = other.memory_override;
        }
        self.teaching.append(&mut other.teaching);
        self.notes.append(&mut other.notes);
    }

    fn merged_teaching(&self, llm_teaching: &[LlmTeaching]) -> Vec<LlmTeaching> {
        let mut merged = llm_teaching.to_vec();
        merged.extend(self.teaching.clone());
        merged
    }
}

pub fn synthesize_sensations(now: &Now, _recall: &RecallBundle) -> Vec<Sensation> {
    let mut sensations = Vec::new();
    if let Some(transcript) = &now.ear.transcript {
        if !transcript.trim().is_empty() {
            sensations.push(
                Sensation::new(
                    "audio.transcript",
                    "ear",
                    now.t_ms,
                    now.t_ms,
                    serde_json::json!({ "transcript": transcript }),
                )
                .with_summary(format!("I hear: {}", transcript.trim()))
                .with_provenance(Provenance::direct().with_stage("ear")),
            );
        }
    }
    if sensations.is_empty() {
        sensations.push(
            Sensation::new(
                "body.state",
                "body",
                now.t_ms,
                now.t_ms,
                serde_json::json!({
                    "battery": now.body.battery_level,
                    "charging": now.body.charging,
                }),
            )
            .with_summary(format!(
                "I feel my battery at {:.2}.",
                now.body.battery_level
            ))
            .with_provenance(Provenance::direct().with_stage("body")),
        );
    }
    sensations
}

pub fn synthesize_impressions(sensations: &[Sensation], observed_at_ms: TimeMs) -> Vec<Impression> {
    sensations
        .iter()
        .map(|sensation| {
            let text = sensation
                .summary
                .clone()
                .unwrap_or_else(|| format!("I notice {}.", sensation.kind));
            Impression::new(
                format!("{}.observation", sensation.kind),
                text,
                vec![sensation.id],
                sensation.occurred_at_ms,
                observed_at_ms,
            )
            .with_confidence(0.8)
        })
        .collect()
}

pub fn synthesize_experience(
    sensations: &[Sensation],
    impressions: &[Impression],
    observed_at_ms: TimeMs,
) -> Experience {
    let occurred_at_ms = sensations
        .iter()
        .map(|sensation| sensation.occurred_at_ms)
        .min()
        .unwrap_or(observed_at_ms);
    let text = impressions
        .iter()
        .map(|impression| impression.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    Experience::new(
        "realtime.situation",
        if text.trim().is_empty() {
            "I cannot tell what is happening yet.".to_string()
        } else {
            text
        },
        impressions.iter().map(|impression| impression.id).collect(),
        sensations.iter().map(|sensation| sensation.id).collect(),
        occurred_at_ms,
        observed_at_ms,
    )
}

fn choose_action<C: Conductor>(
    conductor: &mut C,
    now: &Now,
    latent: &ExperienceLatent,
    llm_tick: &LlmTickResult,
    recall: &RecallBundle,
    proposed_actions: &[ActionPrimitive],
    memory_override: &Option<MemorySense>,
    drive_updates: &[(String, f32)],
) -> Result<ActionPrimitive> {
    if let Some(action) = proposed_actions.first() {
        return Ok(action.clone());
    }
    if let Some(command) = &llm_tick.conscious_command {
        if let Some(action) = &command.action {
            return Ok(action.clone());
        }
    }
    conductor.choose(ConductorInput {
        latent: latent.clone(),
        drives: apply_drive_updates(now.drives.clone(), drive_updates),
        memory: memory_override.clone().unwrap_or_else(|| recall.sense.clone()),
        predictions: now.predictions.clone(),
        surprise: now.surprise.clone(),
        llm: llm_tick.sense.clone(),
        safety: SafetySense::default(),
        body: now.body.clone(),
    })
}

fn apply_drive_updates(mut drives: DriveSense, updates: &[(String, f32)]) -> DriveSense {
    for (name, value) in updates {
        match name.as_str() {
            "battery_hunger" => drives.battery_hunger = *value,
            "danger_avoidance" => drives.danger_avoidance = *value,
            "curiosity" => drives.curiosity = *value,
            "social_interest" => drives.social_interest = *value,
            "fatigue" => drives.fatigue = *value,
            "uncertainty_pressure" => drives.uncertainty_pressure = *value,
            _ => {}
        }
    }
    drives
}

pub fn action_to_motor(action: &ActionPrimitive) -> MotorCommand {
    match action {
        ActionPrimitive::Stop
        | ActionPrimitive::Dock
        | ActionPrimitive::Speak { .. }
        | ActionPrimitive::Chirp { .. } => MotorCommand::stop(),
        ActionPrimitive::Go { intensity, .. } => MotorCommand {
            forward: *intensity,
            turn: 0.0,
        },
        ActionPrimitive::Turn {
            direction,
            intensity,
            ..
        } => MotorCommand {
            forward: 0.0,
            turn: match direction {
                netherwick_actions::TurnDir::Left => *intensity,
                netherwick_actions::TurnDir::Right => -*intensity,
            },
        },
        ActionPrimitive::Inspect { .. } => MotorCommand {
            forward: 0.0,
            turn: 0.3,
        },
        ActionPrimitive::Approach { .. } => MotorCommand {
            forward: 0.3,
            turn: 0.0,
        },
        ActionPrimitive::Explore { .. } => MotorCommand {
            forward: 0.2,
            turn: 0.2,
        },
    }
}

pub fn apply_replaceable_behavior<I, O>(
    wrapper: &mut Replaceable<I, O>,
    input: &I,
    observed_at_ms: TimeMs,
) -> Result<BehaviorRun<O>>
where
    I: Clone,
    O: Clone + PartialEq,
{
    wrapper.run(input, observed_at_ms)
}

pub fn blank_now(t_ms: TimeMs, body: BodySense) -> Now {
    Now::blank(t_ms, body)
}

impl<L, M, R, C, S, A> MinimalRuntime<L, M, R, C, S, A> {
    pub fn with_default_events(
        ledger: L,
        memory_store: M,
        memory_recall: R,
        conductor: C,
        safety: S,
        llm: A,
    ) -> Self {
        Self {
            ledger,
            memory_store,
            memory_recall,
            conductor,
            safety,
            llm,
            event_extractor: EventExtractor::default(),
            event_bus: default_event_bus(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    use netherwick_actions::TurnDir;
    use netherwick_autonomic::{SafetyDecision, SafetyReason};
    use netherwick_body::{BodyFlags, BodySense};
    use netherwick_conductor::SimpleConductor;
    use netherwick_experience::RecalledExperience;
    use netherwick_ledger::ExperienceFrame;
    use netherwick_llm::{LlmTickResult, NoopLlmAgent};
    use netherwick_memory::{InMemoryExperienceStore, MemoryStore, Recall};
    use netherwick_now::RecallHit;

    #[derive(Clone, Default)]
    struct RecordingLedger {
        frames: Arc<Mutex<Vec<ExperienceFrame>>>,
    }

    impl RecordingLedger {
        fn frames(&self) -> Vec<ExperienceFrame> {
            self.frames.lock().expect("ledger mutex poisoned").clone()
        }
    }

    #[async_trait]
    impl LedgerWriter for RecordingLedger {
        async fn append(&self, frame: &ExperienceFrame) -> Result<()> {
            self.frames
                .lock()
                .expect("ledger mutex poisoned")
                .push(frame.clone());
            Ok(())
        }
    }

    #[derive(Clone, Default)]
    struct StaticRecall {
        bundle: RecallBundle,
    }

    #[async_trait]
    impl Recall for StaticRecall {
        async fn recall(&self, _query: RecallQuery) -> Result<RecallBundle> {
            Ok(self.bundle.clone())
        }
    }

    #[async_trait]
    impl MemoryStore for StaticRecall {
        async fn store(&self, _frame: &ExperienceFrame) -> Result<()> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct StaticLlm {
        result: LlmTickResult,
    }

    #[async_trait]
    impl LlmAgent for StaticLlm {
        async fn maybe_tick(
            &mut self,
            _now: &Now,
            _z: &ExperienceLatent,
            _futures: &[FuturePrediction],
            _recall_summary: &str,
        ) -> Result<LlmTickResult> {
            Ok(self.result.clone())
        }
    }

    #[derive(Default)]
    struct AlwaysVetoSafety;

    impl SafetyLayer for AlwaysVetoSafety {
        fn filter(&mut self, _now: &Now, _desired: MotorCommand) -> netherwick_autonomic::SafetyDecision {
            SafetyDecision {
                command: MotorCommand::stop(),
                vetoed: true,
                reason: Some(SafetyReason::Cliff),
                events: Vec::new(),
            }
        }
    }

    #[tokio::test]
    async fn low_battery_emits_battery_low_and_proposes_dock() {
        let ledger = RecordingLedger::default();
        let recall = InMemoryExperienceStore::new();
        let mut runtime = MinimalRuntime::with_default_events(
            ledger.clone(),
            recall.clone(),
            recall,
            SimpleConductor::default(),
            netherwick_autonomic::SimpleSafety::default(),
            NoopLlmAgent,
        );
        let mut body = BodySense::default();
        body.battery_level = 0.1;
        body.last_update_ms = 100;
        let now = Now::blank(100, body);

        let tick = runtime.tick(now, ExperienceLatent::default(), Vec::new()).await.unwrap();
        let frames = ledger.frames();

        assert_eq!(tick.chosen_action, ActionPrimitive::Dock);
        assert!(frames[0]
            .notes
            .iter()
            .any(|note| note.contains("Battery low")));
    }

    #[tokio::test]
    async fn bump_emits_danger_sensation_and_proposes_turn() {
        let ledger = RecordingLedger::default();
        let recall = InMemoryExperienceStore::new();
        let mut runtime = MinimalRuntime::with_default_events(
            ledger.clone(),
            recall.clone(),
            recall,
            SimpleConductor::default(),
            netherwick_autonomic::SimpleSafety::default(),
            NoopLlmAgent,
        );
        let mut body = BodySense::default();
        body.flags = BodyFlags {
            bump_left: true,
            ..BodyFlags::default()
        };
        body.last_update_ms = 10;
        let now = Now::blank(10, body);

        let tick = runtime.tick(now, ExperienceLatent::default(), Vec::new()).await.unwrap();
        let frames = ledger.frames();

        assert!(matches!(
            tick.chosen_action,
            ActionPrimitive::Turn {
                direction: TurnDir::Right,
                ..
            }
        ));
        assert!(frames[0]
            .sensations
            .iter()
            .any(|sensation| sensation.summary.as_deref() == Some("I bumped my left side.")));
    }

    #[tokio::test]
    async fn recall_hit_emits_memory_related_sensation() {
        let ledger = RecordingLedger::default();
        let recall = StaticRecall {
            bundle: RecallBundle {
                hits: vec![RecallHit {
                    frame_id: None,
                    score: 0.9,
                    summary: "remembered danger".to_string(),
                    warning: None,
                }],
                first_person_summary: "I remember a dangerous place.".to_string(),
                recollections: vec![{
                    let experience = Experience::new(
                        "memory",
                        "A wall scraped my shell.",
                        Vec::new(),
                        Vec::new(),
                        1,
                        1,
                    );
                    RecalledExperience {
                        score: 0.9,
                        sensation: experience.to_recall_sensation(2, 0.9, "test"),
                        experience,
                    }
                }],
                ..RecallBundle::default()
            },
        };
        let mut runtime = MinimalRuntime::with_default_events(
            ledger.clone(),
            recall.clone(),
            recall,
            SimpleConductor::default(),
            netherwick_autonomic::SimpleSafety::default(),
            NoopLlmAgent,
        );
        let mut body = BodySense::default();
        body.last_update_ms = 2;
        let now = Now::blank(2, body);

        let _ = runtime.tick(now, ExperienceLatent::default(), Vec::new()).await.unwrap();
        let frames = ledger.frames();

        assert!(frames[0]
            .sensations
            .iter()
            .any(|sensation| sensation.kind == "memory.related_experience"));
    }

    #[tokio::test]
    async fn safety_veto_emits_event_and_records_it() {
        let ledger = RecordingLedger::default();
        let recall = InMemoryExperienceStore::new();
        let mut runtime = MinimalRuntime::with_default_events(
            ledger.clone(),
            recall.clone(),
            recall,
            SimpleConductor::default(),
            AlwaysVetoSafety,
            StaticLlm::default(),
        );
        let mut body = BodySense::default();
        body.last_update_ms = 8;
        let now = Now::blank(8, body);

        let tick = runtime.tick(now, ExperienceLatent::default(), Vec::new()).await.unwrap();
        let frames = ledger.frames();

        assert!(tick.safety.vetoed);
        assert!(frames[0]
            .experiences
            .iter()
            .any(|experience| experience.kind == "safety.veto"));
        assert!(frames[0]
            .notes
            .iter()
            .any(|note| note.contains("Safety vetoed")));
    }
}
