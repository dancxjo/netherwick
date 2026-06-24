use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use netherwick_actions::{
    action_to_motion, action_to_motor_command, ActionPrimitive, ExploreStyle, ReignInput,
    ReignOutcome,
};
use netherwick_autonomic::{SafetyLayer, SafetyReason};
use netherwick_body::MotorComplex;
use netherwick_conductor::{Conductor, ConductorInput};
use netherwick_core::{Provenance, Reward, TimeMs};
use netherwick_events::{
    default_event_bus, DriveName, EventBus, EventContext, EventExtractor, Response,
};
use netherwick_experience::{
    BaselineRewardComputer, BaselineSurpriseComputer, Experience, ExperienceEncoder,
    ExperienceLatent, FeatureExperienceEncoder, FuturePrediction, FuturePredictor, Impression,
    RewardComputer, Sensation, StasisFuturePredictor, SurpriseComputer,
};
use netherwick_ledger::{ExperienceFrame, LedgerWriter, PendingFrame, TransitionBuilder};
use netherwick_llm::{Combobulation, LlmAgent, LlmTickResult};
use netherwick_memory::{MemoryStore, Recall, RecallBundle, RecallQuery};
use netherwick_now::{DriveSense, Now, ReignSense, SafetySense};
use netherwick_sensors::World;
use netherwick_sim::{SimMotorComplex, VirtualWorld};
use uuid::Uuid;

pub struct MinimalRuntime<L, M, R, C, S, A>
where
    L: LedgerWriter + Sync,
    M: MemoryStore,
    R: Recall,
    C: Conductor,
    S: SafetyLayer,
    A: LlmAgent,
{
    pub ledger: L,
    pub memory_store: M,
    pub memory_recall: R,
    pub conductor: C,
    pub safety: S,
    pub llm: A,
    pub extractor: EventExtractor,
    pub bus: EventBus,
    pub reign_queue: Arc<Mutex<ReignQueue>>,
    pub encoder: FeatureExperienceEncoder,
    pub predictor: StasisFuturePredictor,
    pub surprise_computer: BaselineSurpriseComputer,
    pub reward_computer: BaselineRewardComputer,
    pub transition_builder: TransitionBuilder,
}

#[derive(Clone, Debug, Default)]
pub struct ReignQueue {
    pending: VecDeque<ReignInput>,
    latest: Option<ReignInput>,
    clear_sequence: u64,
}

impl ReignQueue {
    pub fn push(&mut self, input: ReignInput) {
        self.latest = Some(input.clone());
        self.pending.push_back(input);
    }

    pub fn latest_active(&self, now_ms: TimeMs) -> Option<ReignInput> {
        self.pending
            .iter()
            .rev()
            .find(|input| input.expires_at_ms > now_ms)
            .cloned()
    }

    pub fn drain_expired(&mut self, now_ms: TimeMs) {
        self.pending.retain(|input| input.expires_at_ms > now_ms);
        if self
            .latest
            .as_ref()
            .map(|input| input.expires_at_ms <= now_ms)
            .unwrap_or(false)
        {
            self.latest = self.latest_active(now_ms);
        }
    }

    pub fn clear(&mut self) {
        self.pending.clear();
        self.latest = None;
        self.clear_sequence = self.clear_sequence.saturating_add(1);
    }

    pub fn sense(&self, now_ms: TimeMs) -> ReignSense {
        let latest = self.latest_active(now_ms);
        let active = latest.is_some();
        ReignSense {
            active,
            mode: latest.as_ref().map(|input| input.mode.clone()),
            last_command_age_ms: latest
                .as_ref()
                .map(|input| now_ms.saturating_sub(input.issued_at_ms)),
            human_override_pressure: latest
                .as_ref()
                .map(|input| input.priority.clamp(0.0, 1.0))
                .unwrap_or(0.0),
            latest,
            pending_count: self
                .pending
                .iter()
                .filter(|input| input.expires_at_ms > now_ms)
                .count(),
            clear_sequence: self.clear_sequence,
        }
    }
}

impl<L, M, R, C, S, A> MinimalRuntime<L, M, R, C, S, A>
where
    L: LedgerWriter + Sync,
    M: MemoryStore,
    R: Recall,
    C: Conductor,
    S: SafetyLayer,
    A: LlmAgent,
{
    pub fn new(
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
            extractor: EventExtractor::default(),
            bus: default_event_bus(),
            reign_queue: Arc::new(Mutex::new(ReignQueue::default())),
            encoder: FeatureExperienceEncoder::new(),
            predictor: StasisFuturePredictor,
            surprise_computer: BaselineSurpriseComputer,
            reward_computer: BaselineRewardComputer,
            transition_builder: TransitionBuilder::new(),
        }
    }

    pub fn with_reign_queue(
        ledger: L,
        memory_store: M,
        memory_recall: R,
        conductor: C,
        safety: S,
        llm: A,
        reign_queue: Arc<Mutex<ReignQueue>>,
    ) -> Self {
        Self {
            ledger,
            memory_store,
            memory_recall,
            conductor,
            safety,
            llm,
            extractor: EventExtractor::default(),
            bus: default_event_bus(),
            reign_queue,
            encoder: FeatureExperienceEncoder::new(),
            predictor: StasisFuturePredictor,
            surprise_computer: BaselineSurpriseComputer,
            reward_computer: BaselineRewardComputer,
            transition_builder: TransitionBuilder::new(),
        }
    }

    pub fn with_default_events(
        ledger: L,
        memory_store: M,
        memory_recall: R,
        conductor: C,
        safety: S,
        llm: A,
    ) -> Self {
        Self::new(ledger, memory_store, memory_recall, conductor, safety, llm)
    }

    pub async fn tick(
        &mut self,
        mut now: Now,
        mut latent: ExperienceLatent,
        mut futures: Vec<FuturePrediction>,
    ) -> Result<RuntimeTick> {
        {
            let mut reign_queue = self
                .reign_queue
                .lock()
                .map_err(|_| anyhow::anyhow!("reign queue lock poisoned"))?;
            reign_queue.drain_expired(now.t_ms);
            now.reign = reign_queue.sense(now.t_ms);
        }
        let reign_input = now.reign.latest.clone();
        let reign_action = reign_input
            .as_ref()
            .and_then(|input| input.command.to_action());

        if latent.z.is_empty() {
            latent = self.encoder.encode(&now)?;
        }
        if futures.is_empty() {
            futures = predict_baseline_futures(&mut self.predictor, &latent)?;
        }

        let recall = self
            .memory_recall
            .recall(RecallQuery::from_now(&now))
            .await?;

        let mut sensations = derive_direct_sensations(&now);
        let mut impressions = derive_direct_impressions(&sensations, now.t_ms);
        let mut experiences = derive_direct_experiences(&impressions, &sensations, now.t_ms);
        let mut teachings = Vec::new();
        let mut notes = Vec::new();
        let mut proposed_actions = Vec::new();

        let events = self.extractor.events_from_now(&now, Some(&recall));
        let ctx = EventContext {
            now: &now,
            latent: Some(&latent),
            recall: Some(&recall),
            predicted_futures: &futures,
            llm: Some(&now.llm),
            safety: None,
        };
        let event_output = self.bus.dispatch_all(&ctx, events)?;
        apply_responses(
            &mut now,
            event_output.responses,
            &mut sensations,
            &mut impressions,
            &mut experiences,
            &mut teachings,
            &mut notes,
            &mut proposed_actions,
        );

        let combobulation = self
            .llm
            .combobulate(&now, &latent, &futures, &recall.first_person_summary)
            .await?;

        let awareness_summary = combobulation.as_ref().map(|value| value.summary.as_str());
        let llm_tick = self
            .llm
            .maybe_tick(
                &now,
                &latent,
                &futures,
                &recall.first_person_summary,
                awareness_summary,
            )
            .await?;
        now.llm = llm_tick.sense.clone();
        apply_llm_tick(
            &llm_tick,
            &mut sensations,
            &mut impressions,
            &mut experiences,
            &mut teachings,
        );

        let mut proposals = proposed_actions.clone();
        if let Some(action) = llm_tick
            .conscious_command
            .as_ref()
            .and_then(|cmd| cmd.action.clone())
        {
            proposals.push(action);
        }

        let chosen_action = self.conductor.choose(ConductorInput {
            latent: latent.clone(),
            drives: now.drives.clone(),
            memory: now.memory.clone(),
            predictions: now.predictions.clone(),
            surprise: now.surprise.clone(),
            llm: now.llm.clone(),
            safety: SafetySense::default(),
            reign: now.reign.clone(),
            body: now.body.clone(),
            proposals,
        })?;

        let safety = self
            .safety
            .filter(&now, action_to_motor_command(Some(&chosen_action)));
        if safety.vetoed {
            now.extensions
                .insert("safety.vetoed".to_string(), serde_json::Value::Bool(true));
            let veto_ctx = EventContext {
                now: &now,
                latent: Some(&latent),
                recall: Some(&recall),
                predicted_futures: &futures,
                llm: Some(&now.llm),
                safety: Some(&safety),
            };
            let veto_events = self
                .extractor
                .events_from_safety(&now, &chosen_action, &safety);
            let veto_output = self.bus.dispatch_all(&veto_ctx, veto_events)?;
            apply_responses(
                &mut now,
                veto_output.responses,
                &mut sensations,
                &mut impressions,
                &mut experiences,
                &mut teachings,
                &mut notes,
                &mut proposed_actions,
            );
            notes.push(format!(
                "Safety vetoed {:?}: {}",
                chosen_action,
                describe_safety_reason(safety.reason.clone())
            ));
        }

        if let Some(combobulation) = &combobulation {
            append_combobulation(
                &mut sensations,
                &mut impressions,
                &mut experiences,
                now.t_ms,
                combobulation,
            );
        }

        let reign_outcome = reign_input.as_ref().map(|input| {
            let accepted_by_conductor = reign_action
                .as_ref()
                .map(|action| action == &chosen_action)
                .unwrap_or(false);
            ReignOutcome {
                input_id: input.id,
                accepted_by_conductor,
                vetoed_by_safety: safety.vetoed,
                final_action: Some(chosen_action.clone()),
                reason: if safety.vetoed {
                    Some(describe_safety_reason(safety.reason.clone()).to_string())
                } else if accepted_by_conductor {
                    None
                } else {
                    Some("conductor chose another action".to_string())
                },
            }
        });

        if experiences.is_empty() {
            experiences.push(Experience::new(
                "realtime.state",
                format!(
                    "I am at t={}ms with battery {:.2}.",
                    now.t_ms, now.body.battery_level
                ),
                Vec::new(),
                Vec::new(),
                now.t_ms,
                now.t_ms,
            ));
        }

        let frame = ExperienceFrame {
            id: Uuid::new_v4(),
            t_ms: now.t_ms,
            now: now.clone(),
            sensations,
            impressions,
            experiences: experiences.clone(),
            z: Some(latent.clone()),
            chosen_action: Some(chosen_action.clone()),
            conscious_command: llm_tick.conscious_command.clone(),
            reign_input,
            reign_outcome,
            predicted_futures: futures.clone(),
            actual_next: None,
            reward: Reward::default(),
            surprise: now.surprise.clone(),
            memory_recall: recall.hits.clone(),
            recollections: recall.recollections.clone(),
            llm_teaching: teachings.clone(),
            counterfactuals: teachings
                .iter()
                .flat_map(|teaching| teaching.counterfactuals.clone())
                .collect(),
            notes,
        };

        self.ledger.append(&frame).await?;
        let surprise_computer = self.surprise_computer.clone();
        let reward_computer = self.reward_computer.clone();
        if let Some(transition) = self.transition_builder.observe(
            PendingFrame {
                frame_id: frame.id,
                now: frame.now.clone(),
                z: latent,
                action: frame.chosen_action.clone(),
                predicted_futures: frame.predicted_futures.clone(),
            },
            |previous, current| {
                let surprise = surprise_computer.compute(
                    &previous.predicted_futures,
                    &current.z,
                    &current.now,
                );
                reward_computer.compute(
                    &previous.now,
                    previous.action.as_ref(),
                    &current.now,
                    &surprise,
                )
            },
            |previous, current| {
                surprise_computer.compute(&previous.predicted_futures, &current.z, &current.now)
            },
        ) {
            self.ledger.append_transition(&transition).await?;
        }
        self.memory_store.store(&frame).await?;

        Ok(RuntimeTick {
            frame,
            experience: experiences.last().cloned().unwrap_or_else(|| {
                Experience::new(
                    "realtime.state",
                    "I am active.",
                    Vec::new(),
                    Vec::new(),
                    now.t_ms,
                    now.t_ms,
                )
            }),
            chosen_action: Some(chosen_action),
            recall,
            llm: llm_tick,
            combobulation,
        })
    }
}

pub struct RuntimeTick {
    pub frame: ExperienceFrame,
    pub experience: Experience,
    pub chosen_action: Option<ActionPrimitive>,
    pub recall: RecallBundle,
    pub llm: LlmTickResult,
    pub combobulation: Option<Combobulation>,
}

#[async_trait::async_trait]
pub trait RuntimeLoop {
    async fn tick(
        &mut self,
        now: Now,
        latent: ExperienceLatent,
        futures: Vec<FuturePrediction>,
    ) -> Result<RuntimeTick>;
}

#[async_trait::async_trait]
impl<L, M, R, C, S, A> RuntimeLoop for MinimalRuntime<L, M, R, C, S, A>
where
    L: LedgerWriter + Sync + Send,
    M: MemoryStore + Send,
    R: Recall + Send,
    C: Conductor + Send,
    S: SafetyLayer + Send,
    A: LlmAgent + Send,
{
    async fn tick(
        &mut self,
        now: Now,
        latent: ExperienceLatent,
        futures: Vec<FuturePrediction>,
    ) -> Result<RuntimeTick> {
        MinimalRuntime::tick(self, now, latent, futures).await
    }
}

pub struct SimRunner<R> {
    pub runtime: R,
    pub world: VirtualWorld,
    pub motors: SimMotorComplex,
    pub tick_count: usize,
    pub tick_ms: u64,
}

impl<R> SimRunner<R>
where
    R: RuntimeLoop + Send,
{
    pub fn new(runtime: R, world: VirtualWorld, motors: SimMotorComplex) -> Self {
        Self {
            runtime,
            world,
            motors,
            tick_count: 0,
            tick_ms: 100,
        }
    }

    pub async fn run_steps(&mut self, steps: usize) -> Result<()> {
        for _ in 0..steps {
            let snapshot = self.world.snapshot().await?;
            let now = snapshot.to_now(snapshot.body.last_update_ms);
            let tick = self
                .runtime
                .tick(now, ExperienceLatent::default(), Vec::new())
                .await?;
            let motion = action_to_motion(tick.chosen_action.as_ref());
            self.motors.send(motion).await?;
            self.tick_count = self.tick_count.saturating_add(1);
        }
        Ok(())
    }
}

fn predict_baseline_futures(
    predictor: &mut impl FuturePredictor,
    latent: &ExperienceLatent,
) -> Result<Vec<FuturePrediction>> {
    let mut out = Vec::new();
    for action in default_candidate_actions() {
        for offset_ms in [1_000, 5_000] {
            out.push(predictor.predict(latent, &action, offset_ms)?);
        }
    }
    Ok(out)
}

fn default_candidate_actions() -> Vec<ActionPrimitive> {
    vec![
        ActionPrimitive::Stop,
        ActionPrimitive::Explore {
            style: ExploreStyle::Wander,
            duration_ms: 2_000,
        },
        ActionPrimitive::Go {
            intensity: 0.15,
            duration_ms: 1_000,
        },
        ActionPrimitive::Dock,
    ]
}

fn apply_responses(
    now: &mut Now,
    responses: Vec<Response>,
    sensations: &mut Vec<Sensation>,
    impressions: &mut Vec<Impression>,
    experiences: &mut Vec<Experience>,
    teachings: &mut Vec<netherwick_llm::LlmTeaching>,
    notes: &mut Vec<String>,
    proposed_actions: &mut Vec<ActionPrimitive>,
) {
    for response in responses {
        match response {
            Response::Emit(_) => {}
            Response::AddSensation(sensation) => sensations.push(sensation),
            Response::AddImpression(impression) => impressions.push(impression),
            Response::AddExperience(experience) => experiences.push(experience),
            Response::ProposeAction(action) => proposed_actions.push(action),
            Response::SetDrive { name, value } => set_drive(&mut now.drives, &name, value),
            Response::SetMemorySense(memory) => now.memory = memory,
            Response::Teach(teaching) => teachings.push(teaching),
            Response::AddMemoryNote(note) => notes.push(note),
        }
    }
}

fn apply_llm_tick(
    llm_tick: &LlmTickResult,
    sensations: &mut Vec<Sensation>,
    impressions: &mut Vec<Impression>,
    experiences: &mut Vec<Experience>,
    teachings: &mut Vec<netherwick_llm::LlmTeaching>,
) {
    if let Some(command) = &llm_tick.conscious_command {
        let sensation = Sensation::new(
            "llm.command",
            "llm",
            llm_tick
                .teaching
                .first()
                .map(|value| value.t_ms)
                .unwrap_or_default(),
            llm_tick
                .teaching
                .first()
                .map(|value| value.t_ms)
                .unwrap_or_default(),
            serde_json::json!({
                "summary": command.summary,
                "action": command.action,
            }),
        )
        .with_summary(command.summary.clone())
        .with_provenance(Provenance::direct().with_stage("llm"));
        let impression = Impression::new(
            "llm.command.observation",
            command.summary.clone(),
            vec![sensation.id],
            sensation.occurred_at_ms,
            sensation.observed_at_ms,
        )
        .with_confidence(llm_tick.sense.confidence);
        let experience = Experience::new(
            "llm.command",
            command.summary.clone(),
            vec![impression.id],
            vec![sensation.id],
            sensation.occurred_at_ms,
            sensation.observed_at_ms,
        );
        sensations.push(sensation);
        impressions.push(impression);
        experiences.push(experience);
    }

    if let Some(critique) = &llm_tick.sense.critique {
        let sensation = Sensation::new(
            "llm.critique",
            "llm",
            llm_tick
                .teaching
                .first()
                .map(|value| value.t_ms)
                .unwrap_or_default(),
            llm_tick
                .teaching
                .first()
                .map(|value| value.t_ms)
                .unwrap_or_default(),
            serde_json::json!({ "critique": critique }),
        )
        .with_summary(critique.clone())
        .with_provenance(Provenance::direct().with_stage("llm"));
        let impression = Impression::new(
            "llm.critique.observation",
            critique.clone(),
            vec![sensation.id],
            sensation.occurred_at_ms,
            sensation.observed_at_ms,
        )
        .with_confidence(llm_tick.sense.confidence);
        sensations.push(sensation);
        impressions.push(impression);
    }

    teachings.extend(llm_tick.teaching.clone());
}

fn append_combobulation(
    sensations: &mut Vec<Sensation>,
    impressions: &mut Vec<Impression>,
    experiences: &mut Vec<Experience>,
    t_ms: u64,
    combobulation: &Combobulation,
) {
    let sensation = Sensation::new(
        "llm.combobulation",
        "llm",
        t_ms,
        t_ms,
        serde_json::json!({
            "summary": combobulation.summary,
            "confidence": combobulation.confidence,
        }),
    )
    .with_summary(combobulation.summary.clone())
    .with_provenance(Provenance::direct().with_stage("combobulator"));
    let impression = Impression::new(
        "llm.combobulation.observation",
        combobulation.summary.clone(),
        vec![sensation.id],
        t_ms,
        t_ms,
    )
    .with_confidence(combobulation.confidence);
    let experience = Experience::new(
        "llm.combobulation",
        combobulation.summary.clone(),
        vec![impression.id],
        vec![sensation.id],
        t_ms,
        t_ms,
    );
    sensations.push(sensation);
    impressions.push(impression);
    experiences.push(experience);
}

fn derive_direct_sensations(now: &Now) -> Vec<Sensation> {
    let mut out = Vec::new();
    if let Some(transcript) = &now.ear.transcript {
        let transcript = transcript.trim();
        if !transcript.is_empty() {
            out.push(
                Sensation::new(
                    "audio.transcript",
                    "ear",
                    now.t_ms,
                    now.t_ms,
                    serde_json::json!({ "transcript": transcript }),
                )
                .with_summary(format!("I hear: {transcript}")),
            );
        }
    }
    out
}

fn derive_direct_impressions(sensations: &[Sensation], t_ms: u64) -> Vec<Impression> {
    sensations
        .iter()
        .filter_map(|sensation| {
            if sensation.kind == "audio.transcript" {
                Some(
                    Impression::new(
                        "audio.transcript.observation",
                        sensation.summary.clone().unwrap_or_default(),
                        vec![sensation.id],
                        t_ms,
                        t_ms,
                    )
                    .with_confidence(0.8),
                )
            } else {
                None
            }
        })
        .collect()
}

fn derive_direct_experiences(
    impressions: &[Impression],
    sensations: &[Sensation],
    t_ms: u64,
) -> Vec<Experience> {
    if impressions.is_empty() || sensations.is_empty() {
        return Vec::new();
    }
    vec![Experience::new(
        "realtime.situation",
        impressions
            .last()
            .map(|value| value.text.clone())
            .unwrap_or_default(),
        impressions.iter().map(|value| value.id).collect(),
        sensations.iter().map(|value| value.id).collect(),
        t_ms,
        t_ms,
    )]
}

fn set_drive(drives: &mut DriveSense, name: &DriveName, value: f32) {
    match name {
        DriveName::BatteryHunger => drives.battery_hunger = value,
        DriveName::DangerAvoidance => drives.danger_avoidance = value,
        DriveName::Curiosity => drives.curiosity = value,
        DriveName::SocialInterest => drives.social_interest = value,
        DriveName::Fatigue => drives.fatigue = value,
        DriveName::UncertaintyPressure => drives.uncertainty_pressure = value,
    }
}

fn describe_safety_reason(reason: Option<SafetyReason>) -> &'static str {
    match reason {
        Some(SafetyReason::WheelDrop) => "wheel drop",
        Some(SafetyReason::Cliff) => "cliff",
        Some(SafetyReason::BatteryCritical) => "critical battery",
        Some(SafetyReason::StaleSensors) => "stale sensors",
        Some(SafetyReason::LostBodyComms) => "lost body comms",
        Some(SafetyReason::MotorOutOfRange) => "motor out of range",
        Some(SafetyReason::HighDanger) => "high danger",
        Some(SafetyReason::RawLlmMotorRejected) => "raw llm motor rejected",
        None => "unknown reason",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use netherwick_actions::{ReignCommand, ReignMode, ReignSource};
    use netherwick_autonomic::SimpleSafety;
    use netherwick_body::BodySense;
    use netherwick_conductor::{Conductor, ConductorInput, SimpleConductor};
    use netherwick_ledger::{ExperienceTransition, JsonlLedger, LedgerReader};
    use netherwick_memory::InMemoryExperienceStore;
    use netherwick_now::Now;
    use netherwick_sensors::World;
    use netherwick_sim::{ArenaConfig, SimObject, VirtualWorld};
    use std::fs;
    use std::path::{Path, PathBuf};

    #[tokio::test]
    async fn tick_adds_combobulated_experience() {
        let ledger = JsonlLedger::new("/tmp/netherwick-runtime-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            netherwick_llm::NoopLlmAgent,
        );
        let mut now = Now::blank(100, BodySense::default());
        now.ear.transcript = Some("hello world".to_string());

        let tick = runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();

        assert!(tick
            .frame
            .experiences
            .iter()
            .any(|experience| experience.text.contains("hello world")));
    }

    #[tokio::test]
    async fn stop_reign_becomes_now_event_and_chosen_action() {
        let ledger = JsonlLedger::new("/tmp/netherwick-runtime-reign-stop-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            netherwick_llm::NoopLlmAgent,
        );
        runtime.reign_queue.lock().unwrap().push(test_reign_input(
            100,
            ReignMode::Direct,
            ReignCommand::Stop,
            2_000,
        ));
        let mut body = BodySense::default();
        body.last_update_ms = 100;
        let now = Now::blank(100, body);

        let tick = runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();

        assert!(tick.frame.now.reign.active);
        assert_eq!(tick.chosen_action, Some(ActionPrimitive::Stop));
        assert!(tick
            .frame
            .sensations
            .iter()
            .any(|sensation| sensation.kind == "reign.command"));
        assert!(tick
            .frame
            .reign_input
            .as_ref()
            .map(|input| matches!(input.command, ReignCommand::Stop))
            .unwrap_or(false));
        assert!(tick
            .frame
            .reign_outcome
            .as_ref()
            .map(|outcome| outcome.accepted_by_conductor)
            .unwrap_or(false));
    }

    #[test]
    fn expired_reign_disappears_from_sense() {
        let mut queue = ReignQueue::default();
        queue.push(test_reign_input(
            100,
            ReignMode::Direct,
            ReignCommand::Stop,
            100,
        ));

        queue.drain_expired(250);
        let sense = queue.sense(250);

        assert!(!sense.active);
        assert!(sense.latest.is_none());
        assert_eq!(sense.pending_count, 0);
    }

    #[test]
    fn clear_marks_reign_sense_for_event_extraction() {
        let mut queue = ReignQueue::default();
        queue.push(test_reign_input(
            100,
            ReignMode::Direct,
            ReignCommand::Stop,
            1_000,
        ));

        queue.clear();
        let sense = queue.sense(150);

        assert!(!sense.active);
        assert!(sense.latest.is_none());
        assert_eq!(sense.clear_sequence, 1);
    }

    #[tokio::test]
    async fn safety_veto_beats_direct_go_reign_at_cliff() {
        let ledger = JsonlLedger::new("/tmp/netherwick-runtime-reign-safety-test");
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let mut runtime = MinimalRuntime::new(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            netherwick_llm::NoopLlmAgent,
        );
        runtime.reign_queue.lock().unwrap().push(test_reign_input(
            100,
            ReignMode::Direct,
            ReignCommand::Go {
                intensity: 0.5,
                duration_ms: 500,
            },
            2_000,
        ));
        let mut body = BodySense::default();
        body.flags.cliff_left = true;
        body.last_update_ms = 100;
        let now = Now::blank(100, body);

        let tick = runtime
            .tick(now, ExperienceLatent::default(), Vec::new())
            .await
            .unwrap();

        assert_eq!(
            tick.chosen_action,
            Some(ActionPrimitive::Go {
                intensity: 0.5,
                duration_ms: 500,
            })
        );
        assert!(tick
            .frame
            .reign_outcome
            .as_ref()
            .map(|outcome| outcome.vetoed_by_safety)
            .unwrap_or(false));
        assert!(tick
            .frame
            .notes
            .iter()
            .any(|note| note.contains("Safety vetoed")));
    }

    #[tokio::test]
    async fn sim_runner_writes_frames_and_transitions() {
        let root = test_ledger_root("sim-runner-writes");
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(ledger.clone(), SimpleConductor::default());
        let (world, motors) = VirtualWorld::new_with_motor(7, arena());
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(10).await.unwrap();

        let frames = ledger.recent(20).await.unwrap();
        let transitions = read_transitions(&root);
        assert!(frames.len() >= 10);
        assert!(transitions.len() >= 9);
        assert!(transitions.iter().any(|transition| {
            transition.before.body.odometry.x_m != transition.after.body.odometry.x_m
                || transition.before.body.odometry.y_m != transition.after.body.odometry.y_m
        }));
    }

    #[tokio::test]
    async fn sim_runner_applies_chosen_action_to_world() {
        let ledger = JsonlLedger::new(test_ledger_root("sim-runner-action-world"));
        let runtime = test_runtime(
            ledger,
            FixedConductor::new(ActionPrimitive::Go {
                intensity: 0.4,
                duration_ms: 1_000,
            }),
        );
        let (mut world, motors) = VirtualWorld::new_with_motor(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.5, 7));
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(1).await.unwrap();
        let snapshot = runner.world.snapshot().await.unwrap();

        assert!(snapshot.body.odometry.x_m > 1.0);
        assert_eq!(runner.tick_count, 1);
    }

    #[tokio::test]
    async fn sim_runner_reaches_charger_gets_positive_reward() {
        let root = test_ledger_root("sim-runner-charger-reward");
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(
            ledger,
            FixedConductor::new(ActionPrimitive::Go {
                intensity: 0.4,
                duration_ms: 1_000,
            }),
        );
        let (mut world, motors) = VirtualWorld::new_with_motor(7, arena());
        let mut body = test_body(1.0, 1.0, 0.2, 7);
        body.battery_level = 0.2;
        world.set_body(body);
        world.add_object(SimObject::charger("charger", "charger", 1.38, 1.0, 0.18));
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(2).await.unwrap();
        let transitions = read_transitions(&root);

        let transition = transitions.last().unwrap();
        assert!(transition.after.body.charging);
        assert!(transition.reward.value > 0.0);
        assert!(transition.surprise.total > 0.0);
    }

    #[tokio::test]
    async fn sim_runner_collision_sets_bump_and_negative_reward() {
        let root = test_ledger_root("sim-runner-collision-reward");
        let ledger = JsonlLedger::new(&root);
        let runtime = test_runtime(
            ledger,
            FixedConductor::new(ActionPrimitive::Go {
                intensity: 0.4,
                duration_ms: 1_000,
            }),
        );
        let (mut world, motors) = VirtualWorld::new_with_motor(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.8, 7));
        world.add_object(SimObject::obstacle("box", "box", 1.31, 1.0, 0.1));
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(2).await.unwrap();
        let transitions = read_transitions(&root);

        let transition = transitions.last().unwrap();
        assert!(transition.after.body.flags.bump_left || transition.after.body.flags.bump_right);
        assert!(transition.reward.value < 0.0);
        assert!(transition.surprise.total > 0.0);
    }

    #[tokio::test]
    async fn shared_reign_queue_controls_next_sim_tick() {
        let root = test_ledger_root("sim-runner-shared-reign");
        let ledger = JsonlLedger::new(&root);
        let queue = Arc::new(Mutex::new(ReignQueue::default()));
        queue.lock().unwrap().push(test_reign_input(
            7,
            ReignMode::Direct,
            ReignCommand::Turn {
                direction: netherwick_actions::TurnDir::Left,
                intensity: 0.5,
                duration_ms: 500,
            },
            2_000,
        ));
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        let runtime = MinimalRuntime::with_reign_queue(
            ledger,
            memory,
            recall,
            SimpleConductor::default(),
            SimpleSafety::default(),
            netherwick_llm::NoopLlmAgent,
            queue,
        );
        let (mut world, motors) = VirtualWorld::new_with_motor(7, arena());
        world.set_body(test_body(1.0, 1.0, 0.8, 7));
        let mut runner = SimRunner::new(runtime, world, motors);

        runner.run_steps(1).await.unwrap();
        let snapshot = runner.world.snapshot().await.unwrap();
        let frames = JsonlLedger::new(&root).recent(5).await.unwrap();
        let frame = frames.last().unwrap();

        assert!(snapshot.body.odometry.heading_rad > 0.0);
        assert!(frame.now.reign.active);
        assert!(frame
            .sensations
            .iter()
            .any(|sensation| sensation.kind == "reign.command"));
        assert!(frame.reign_input.is_some());
        assert!(frame
            .reign_outcome
            .as_ref()
            .map(|outcome| outcome.accepted_by_conductor)
            .unwrap_or(false));
    }

    #[derive(Clone, Debug)]
    struct FixedConductor {
        action: ActionPrimitive,
    }

    impl FixedConductor {
        fn new(action: ActionPrimitive) -> Self {
            Self { action }
        }
    }

    impl Conductor for FixedConductor {
        fn choose(&mut self, _input: ConductorInput) -> Result<ActionPrimitive> {
            Ok(self.action.clone())
        }
    }

    fn test_runtime<C>(
        ledger: JsonlLedger,
        conductor: C,
    ) -> MinimalRuntime<
        JsonlLedger,
        InMemoryExperienceStore,
        InMemoryExperienceStore,
        C,
        SimpleSafety,
        netherwick_llm::NoopLlmAgent,
    >
    where
        C: Conductor,
    {
        let memory = InMemoryExperienceStore::new();
        let recall = memory.clone();
        MinimalRuntime::new(
            ledger,
            memory,
            recall,
            conductor,
            SimpleSafety::default(),
            netherwick_llm::NoopLlmAgent,
        )
    }

    fn arena() -> ArenaConfig {
        ArenaConfig {
            width_m: 4.0,
            height_m: 4.0,
        }
    }

    fn test_body(x_m: f32, y_m: f32, battery_level: f32, last_update_ms: u64) -> BodySense {
        let mut body = BodySense::default();
        body.odometry.x_m = x_m;
        body.odometry.y_m = y_m;
        body.battery_level = battery_level;
        body.last_update_ms = last_update_ms;
        body
    }

    fn test_ledger_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("netherwick-{name}-{}", Uuid::new_v4()));
        let _ = fs::remove_dir_all(&root);
        root
    }

    fn read_transitions(root: &Path) -> Vec<ExperienceTransition> {
        let mut out = Vec::new();
        read_transition_paths(root, &mut out);
        out
    }

    fn read_transition_paths(path: &Path, out: &mut Vec<ExperienceTransition>) {
        let Ok(entries) = fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                read_transition_paths(&path, out);
            } else if path.file_name().and_then(|name| name.to_str()) == Some("transitions.jsonl") {
                let Ok(contents) = fs::read_to_string(path) else {
                    continue;
                };
                out.extend(
                    contents
                        .lines()
                        .filter(|line| !line.trim().is_empty())
                        .filter_map(|line| serde_json::from_str(line).ok()),
                );
            }
        }
    }

    fn test_reign_input(
        issued_at_ms: u64,
        mode: ReignMode,
        command: ReignCommand,
        ttl_ms: u64,
    ) -> ReignInput {
        ReignInput {
            id: Uuid::new_v4(),
            issued_at_ms,
            expires_at_ms: issued_at_ms + ttl_ms,
            source: ReignSource::WebRemote,
            mode,
            command,
            priority: 1.0,
            note: None,
        }
    }
}
