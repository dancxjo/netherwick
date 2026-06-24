use anyhow::Result;
use netherwick_actions::ActionPrimitive;
use netherwick_autonomic::SafetyLayer;
use netherwick_behaviors::{BehaviorRun, Replaceable};
use netherwick_body::{BodySense, MotorCommand};
use netherwick_conductor::{Conductor, ConductorInput};
use netherwick_core::{Provenance, Reward, TimeMs};
use netherwick_experience::{
    Experience, ExperienceLatent, FuturePrediction, Impression, Sensation,
};
use netherwick_ledger::{ExperienceFrame, LedgerWriter};
use netherwick_llm::{ConsciousCommand, LlmAgent, LlmTickResult};
use netherwick_memory::{MemoryStore, Recall, RecallBundle, RecallQuery};
use netherwick_now::{Now, SafetySense};
use uuid::Uuid;

pub const CORE_LOOP: &[&str] = &[
    "Now_t",
    "encode to latent z_t",
    "predict futures",
    "recall relevant experiences",
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
    ("medium", "Now/prediction/conductor"),
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
        let sensations = synthesize_sensations(&now, &recall);
        let impressions = synthesize_impressions(&sensations, now.t_ms);
        let experience = synthesize_experience(&sensations, &impressions, now.t_ms);
        let llm_tick = self
            .llm
            .maybe_tick(&now, &latent, &predicted_futures, &recall.first_person_summary)
            .await?;
        let action = choose_action(
            &mut self.conductor,
            &now,
            &latent,
            &predicted_futures,
            &llm_tick,
            &recall,
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
        let reward = Reward {
            value: if safety_decision.vetoed { -0.1 } else { 0.0 },
        };
        let frame = ExperienceFrame {
            id: Uuid::new_v4(),
            t_ms: now.t_ms,
            now: now.clone(),
            sensations,
            impressions,
            experiences: vec![experience.clone()],
            z: Some(latent.clone()),
            chosen_action: Some(action.clone()),
            conscious_command: llm_tick.conscious_command.clone(),
            predicted_futures: predicted_futures.clone(),
            actual_next: None,
            reward,
            surprise: now.surprise.clone(),
            memory_recall: recall.hits.clone(),
            recollections: recall.recollections.clone(),
            llm_teaching: llm_tick.teaching.clone(),
            counterfactuals: llm_tick
                .teaching
                .iter()
                .flat_map(|teaching| teaching.counterfactuals.clone())
                .collect(),
            notes: vec![recall.first_person_summary.clone()],
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

pub fn synthesize_sensations(now: &Now, recall: &RecallBundle) -> Vec<Sensation> {
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
    if !recall.first_person_summary.trim().is_empty() {
        sensations.extend(recall.recollections.iter().map(|recalled| recalled.sensation.clone()));
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
    predicted_futures: &[FuturePrediction],
    llm_tick: &LlmTickResult,
    recall: &RecallBundle,
) -> Result<ActionPrimitive> {
    if let Some(command) = &llm_tick.conscious_command {
        if let Some(action) = &command.action {
            return Ok(action.clone());
        }
    }
    conductor.choose(ConductorInput {
        latent: latent.clone(),
        drives: now.drives.clone(),
        memory: recall.sense.clone(),
        predictions: now.predictions.clone(),
        surprise: now.surprise.clone(),
        llm: llm_tick.sense.clone(),
        safety: SafetySense::default(),
        body: now.body.clone(),
    })
}

pub fn action_to_motor(action: &ActionPrimitive) -> MotorCommand {
    match action {
        ActionPrimitive::Stop | ActionPrimitive::Dock | ActionPrimitive::Speak { .. } | ActionPrimitive::Chirp { .. } => {
            MotorCommand::stop()
        }
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
